//! Complete CUDA fast backbone for Inkling-2-Mini.
//!
//! Wires all 8 Wave 2 CUDA primitives into one clean forward pass supporting:
//! - Linear chunked prefill (`CUDA_PREFILL_CHUNK = 1024`, matching SWA `ring_cap = 2048`)
//!   wrapped in [`CudaGraphsGuard::disable_for_prefill`]
//! - Single-token AR decode (`T = 1`, zero host syncs on `ids_arr`, stable SWA ring pointers)
//! - Speculative linear & tree verification (`T <= 16`, `CudaTreeInputs` + `Inkling2ShortConvTape`)
//! - Optional hidden-state tap capture (`[5, 23, 41]` pre-visit, `48` post-`output_norm`) for DSpark.

use std::sync::OnceLock;
use std::time::Instant;

use mlx_sys as sys;
use napi::bindgen_prelude::{Error, Result};

use crate::accelerator::CudaGraphsGuard;
use crate::array::{DType, MxArray};
use crate::kernels::inkling2::{
    KvConvPrepParams, LmHeadOutMode, MoeForwardParams, MoeQuantMode, SdpaParams,
    block_attn_res_norm_cuda, embed_rmsnorm_cuda, kv_conv_prep_store_cuda, lm_head_softcap_cuda,
    loop_blend_rms_norm_cuda, loop_ctrl_norm_cuda, loop_pair_rms_norm_cuda,
    loop_up_blend_rms_norm_cuda, moe_forward_cuda, sconv_k4_residual_cuda, sdpa_few_cuda,
    sdpa_prefill_cuda,
};
use crate::models::quantized_linear::{
    DEFAULT_QUANT_MODE, LinearProj, MXFP8_MODE, NVFP4_MODE, QuantizedSwitchLinear,
};
use crate::moe::SwitchProj;
use crate::nn::Embedding;
use crate::nn::embedding::EmbeddingRawView;
use crate::tools::inkling::IMAGE_PAD_ID;

use super::attention::forward_linear_proj_fast;
use super::attn_res::{ATTN_RES_EPS, apply_sublayer_rms_norm};
use super::config::Inkling2Config;
use super::cuda_kv::Inkling2CudaCache;
use super::model::{AttnInputNorm, Inkling2Inner, Inkling2PhysicalLayer, overwrite_image_pad_rows};
use super::moe::ExpertGateUp;
use super::short_conv::{
    Inkling2ShortConvTape, StreamShortConvTape, VisitShortConvState, VisitShortConvTape,
};

/// Prefill chunk size matching SWA `ring_cap = 2048` (`1024` window + `1024` chunk tokens).
pub(crate) const CUDA_PREFILL_CHUNK: usize = 1024;

static INKLING2_CUDA_PROFILE: OnceLock<bool> = OnceLock::new();

/// Opt-in per-stage GPU wall-clock X-ray (`MLX_INKLING2_PROFILE=1`), checked once per process.
/// When unset (default), introduces zero syncs and zero overhead.
#[inline]
fn cuda_profile_enabled() -> bool {
    *INKLING2_CUDA_PROFILE.get_or_init(|| {
        std::env::var("MLX_INKLING2_PROFILE").is_ok_and(|v| {
            matches!(
                v.trim(),
                "1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES"
            )
        })
    })
}

/// Accumulated per-stage GPU microseconds across one forward pass (all 48 logical visits).
#[derive(Default)]
struct CudaStageProfile {
    embed_us: u64,
    attn_res_us: u64,
    loop_ctrl_us: u64,
    attn_proj_us: u64,
    kv_prep_store_us: u64,
    sdpa_us: u64,
    o_proj_us: u64,
    attn_sconv_us: u64,
    moe_nvfp4_us: u64,
    moe_l20_23_us: u64,
    lm_head_us: u64,
}

impl CudaStageProfile {
    #[inline]
    fn finish_stage(acc: &mut u64, started: Instant, arrays: &[&MxArray]) -> Result<()> {
        if !arrays.is_empty() {
            MxArray::eval_arrays_with_context(arrays, "inkling2::cuda_profile_stage")?;
        }
        crate::array::synchronize();
        *acc = acc.saturating_add(started.elapsed().as_micros() as u64);
        Ok(())
    }

    fn print(&self, tokens: usize, lm_head_mode: CudaLmHeadMode, is_tree: bool) {
        let total_us = self.embed_us
            + self.attn_res_us
            + self.loop_ctrl_us
            + self.attn_proj_us
            + self.kv_prep_store_us
            + self.sdpa_us
            + self.o_proj_us
            + self.attn_sconv_us
            + self.moe_nvfp4_us
            + self.moe_l20_23_us
            + self.lm_head_us;
        let ms = |us: u64| us as f64 / 1000.0;
        eprintln!(
            "[inkling2-profile] T={tokens} tree={is_tree} mode={lm_head_mode:?} total={:.2}ms | embed={:.2} attn_res={:.2} loop_ctrl={:.2} attn_proj={:.2} kv_prep_store={:.2} sdpa={:.2} o_proj={:.2} attn_sconv={:.2} moe_nvfp4={:.2} moe_l20_23={:.2} lm_head={:.2}",
            ms(total_us),
            ms(self.embed_us),
            ms(self.attn_res_us),
            ms(self.loop_ctrl_us),
            ms(self.attn_proj_us),
            ms(self.kv_prep_store_us),
            ms(self.sdpa_us),
            ms(self.o_proj_us),
            ms(self.attn_sconv_us),
            ms(self.moe_nvfp4_us),
            ms(self.moe_l20_23_us),
            ms(self.lm_head_us),
        );
    }
}

/// Whether the Inkling-2-Mini CUDA fast path is enabled on the active device.
///
/// Requires CUDA ops + SM >= 8.0 Tensor Core kernels, and respects `MLX_INKLING2_CUDA_FAST=0`
/// (which forces the legacy flat fallback for A/B debugging).
#[inline]
pub(crate) fn cuda_fast_enabled() -> bool {
    if !crate::kernels::inkling2::available() || !crate::kernels::inkling2::sm80_available() {
        return false;
    }
    match std::env::var("MLX_INKLING2_CUDA_FAST") {
        Ok(v) => !matches!(v.trim(), "0" | "off" | "false" | "FALSE"),
        Err(_) => true,
    }
}

/// Hardcoded numerical constants and kernel bounds compiled into the Wave 2 CUDA kernels:
/// - `inkling2_ops.cu.inc`: `kEps = 1e-6f`, `kQkClip = 22.627416997969522f`, `kRelClip = 8.0f`
/// - `inkling2_moe_{few,prefill}.cu.inc`: `mlp_gate_softcap = 4.0f`, `mlp_up_softcap = 25.0f`
/// - `ops.cpp` (`Inkling2BlockAttnResNormCuda`): max `8` AttnRes sources (`<= 7` completed blocks + `1` partial)
const CUDA_FAST_RMS_NORM_EPS: f64 = 1e-6;
const CUDA_FAST_QK_L2_CLIP: f64 = 22.627_416_997_969_522;
const CUDA_FAST_REL_L2_CLIP: f64 = 8.0;
const CUDA_FAST_MLP_GATE_SOFTCAP: f64 = 4.0;
const CUDA_FAST_MLP_UP_SOFTCAP: f64 = 25.0;
const CUDA_FAST_MAX_ATTN_RES_SOURCES: usize = 8;

/// Pure config predicate for [`cuda_fast_supported_for`] (independent of GPU/env availability):
/// verifies that `config` matches the Inkling-2-Mini geometry, hardcoded numerical constants,
/// and kernel bounds compiled into the Wave 2 CUDA kernels:
/// - Geometry: `H=2304, Hq=24, Hkv=8, D=128, Dv=128, d_rel=16, E=128, top_k=8, I=1152, k=4, rank=64`
/// - Hardcoded constants: `rms_norm_eps=1e-6`, `qk_l2_clip_threshold=sqrt(512)`,
///   `rel_l2_clip_threshold=8.0`, `mlp_gate_softcap=4.0`, `mlp_up_softcap=25.0`
/// - Kernel limits: `num_physical_layers > 0`, `n_layers_per_attn_res_block > 0`,
///   `num_physical_layers.div_ceil(n_layers_per_attn_res_block) + 1 <= 8`, and
///   `sliding_window > 0` whenever any layer visit uses local sliding-window attention.
#[inline]
pub(crate) fn cuda_fast_config_supported(config: &Inkling2Config) -> bool {
    let n_phys = config.num_physical_layers();
    let block_size = config.n_layers_per_attn_res_block;
    let attn_res_ok = n_phys > 0
        && block_size > 0
        && n_phys.div_ceil(block_size).saturating_add(1) <= CUDA_FAST_MAX_ATTN_RES_SOURCES;
    let swa_ok = config.sliding_window > 0 || !config.visits().iter().any(|v| v.is_local);

    config.hidden_size == 2304
        && config.num_attention_heads == 24
        && config.num_key_value_heads == 8
        && config.head_dim == 128
        && config.effective_value_head_dim() == 128
        && config.d_rel == 16
        && config.num_experts == 128
        && config.num_experts_per_tok == 8
        && config.effective_moe_intermediate_size() == 1152
        && config.sconv_kernel_size == 4
        && config
            .loop_config
            .as_ref()
            .is_none_or(|lc| lc.gate_rank == 64)
        && (config.rms_norm_eps - CUDA_FAST_RMS_NORM_EPS).abs() <= 1e-9
        && (config.qk_l2_clip_threshold - CUDA_FAST_QK_L2_CLIP).abs() <= 1e-4
        && (config.rel_l2_clip_threshold - CUDA_FAST_REL_L2_CLIP).abs() <= 1e-4
        && (config.mlp_gate_softcap - CUDA_FAST_MLP_GATE_SOFTCAP).abs() <= 1e-4
        && (config.mlp_up_softcap - CUDA_FAST_MLP_UP_SOFTCAP).abs() <= 1e-4
        && attn_res_ok
        && swa_ok
}

/// Whether `config` matches the Inkling-2-Mini geometry, hardcoded numerical constants, and kernel
/// bounds compiled into the Wave 2 CUDA kernels, and the CUDA fast path is enabled on this host.
#[inline]
pub(crate) fn cuda_fast_supported_for(config: &Inkling2Config) -> bool {
    cuda_fast_enabled() && cuda_fast_config_supported(config)
}

/// Output mode for the final LM head in [`Inkling2Inner::forward_cuda_ids`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaLmHeadMode {
    /// `[T, unpadded_vocab]` FP32 when `T <= 16`; last-row `[1, unpadded_vocab]` FP32 when `T > 16`.
    LogitsF32,
    /// Always compute only the last row `[1, unpadded_vocab]` FP32 logits (`M = 1`, ideal for prefill).
    LogitsF32Last,
    /// Full `[T, unpadded_vocab]` FP32 logits for any `T` (parity tests / logprobs).
    #[allow(dead_code)]
    LogitsF32All,
    /// Greedy argmax over all `T` rows (`[T]` Int32; uses the fused zero-logit-DRAM kernel when
    /// `T <= 16` and falls back to full-sequence `lm_head` argmax when `T > 16`).
    #[allow(dead_code)]
    GreedyArgmax,
    /// Skip final LM head (and output AttnRes/norm unless tap `num_visits` is requested).
    Skip,
}

/// Device-resident tree verification inputs for [`Inkling2Inner::forward_cuda_ids`].
#[derive(Clone, Copy)]
pub(crate) struct CudaTreeInputs<'a> {
    /// `[T]` Int32 tree depths (`pos = start_pos + depths[t]`).
    pub depths_i32: &'a MxArray,
    /// `[T]` Uint32 ancestor bitmasks for causal tree SDPA.
    pub ancestors_mask_u32: &'a MxArray,
    /// `[T, 4]` Int32 tree ShortConv tap indices (`< 0` reads `hist_in[3 + idx]`).
    pub conv_taps_i32: &'a MxArray,
}

/// Output of [`Inkling2Inner::forward_cuda_ids`].
pub(crate) struct CudaForwardOutput {
    pub logits_or_tokens: Option<MxArray>,
    pub taps: Vec<MxArray>,
    #[allow(dead_code)]
    pub tape: Option<Inkling2ShortConvTape>,
}

/// Pre-computed multimodal splice info for one prefill chunk (avoids reading `ids_arr` to host).
struct ChunkImageSplice<'a> {
    embeddings: &'a MxArray,
    pad_positions: &'a [usize],
}

/// Single-launch fused token embedding lookup + precise FP32 RMSNorm -> BF16 on CUDA
/// (`embed_rmsnorm_cuda`), supporting dense BF16 and packed affine Q8_0 (`bits == 8,
/// group_size == 32, mode == "affine"`, explicit `biases: Some(...)`). Returns `Ok(None)`
/// cleanly off CUDA or for unsupported/incomplete embedding backends so callers fall back
/// to `embed.forward` + RMSNorm (never assuming missing affine biases are symmetric Q8).
pub(crate) fn try_embed_rmsnorm_cuda(
    embed: &Embedding,
    token_ids: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
) -> Result<Option<MxArray>> {
    if !crate::kernels::inkling2::available() || norm_weight.dtype()? != DType::Float32 {
        return Ok(None);
    }
    let Some(raw) = embed.raw_view() else {
        return Ok(None);
    };
    match raw {
        EmbeddingRawView::Dense(w) if w.dtype()? == DType::BFloat16 => {
            embed_rmsnorm_cuda(token_ids, w, None, None, norm_weight, eps)
        }
        EmbeddingRawView::QuantizedPacked {
            weight,
            scales,
            biases: Some(biases),
            group_size: 32,
            bits: 8,
            mode: DEFAULT_QUANT_MODE,
        } => embed_rmsnorm_cuda(
            token_ids,
            weight,
            Some(scales),
            Some(biases),
            norm_weight,
            eps,
        ),
        _ => Ok(None),
    }
}

/// Single-launch FP32-accumulated RMSNorm returning BF16 via `mlx_rmsnorm_precise_f32_cuda`,
/// falling back to `apply_sublayer_rms_norm` when input is not BF16 or weight is not FP32.
#[inline]
fn precise_f32_rms_norm_bf16(x: &MxArray, weight: &MxArray, eps: f32) -> Result<MxArray> {
    if crate::accelerator::cuda_ops_available()
        && x.dtype()? == DType::BFloat16
        && weight.dtype()? == DType::Float32
    {
        let handle =
            unsafe { sys::mlx_rmsnorm_precise_f32_cuda(x.as_raw_ptr(), weight.as_raw_ptr(), eps) };
        if !handle.is_null() {
            return MxArray::from_handle(handle, "inkling2_cuda_rms_norm_precise_f32");
        }
    }
    apply_sublayer_rms_norm(x, weight, eps, DType::BFloat16)
}

/// Classify a `SwitchProj` into one of the 5 Wave 2 `moe_forward_cuda` quantization formats.
fn classify_cuda_moe_proj(
    proj: &SwitchProj,
    allow_q5k: bool,
) -> Result<Option<(&QuantizedSwitchLinear, MoeQuantMode)>> {
    let SwitchProj::Quantized(q) = proj else {
        return Ok(None);
    };
    let s_dt = q.get_scales().dtype()?;
    let mode = if q.mode() == NVFP4_MODE
        && q.bits() == 4
        && q.group_size() == 16
        && q.get_biases().is_none()
        && s_dt == DType::Uint8
    {
        MoeQuantMode::Nvfp4
    } else if q.mode() == DEFAULT_QUANT_MODE
        && q.bits() == 8
        && q.group_size() == 32
        && q.is_symmetric_q8()
        && q.get_biases().is_none()
        && matches!(s_dt, DType::Float16 | DType::BFloat16)
    {
        MoeQuantMode::SymQ8
    } else if q.mode() == DEFAULT_QUANT_MODE
        && q.bits() == 8
        && q.group_size() == 32
        && q.get_biases().is_some()
        && matches!(s_dt, DType::Float16 | DType::BFloat16)
        && q.get_biases().is_some_and(|b| b.dtype().ok() == Some(s_dt))
    {
        MoeQuantMode::AffineQ8
    } else if allow_q5k
        && q.mode() == "q5k"
        && q.bits() == 5
        && q.group_size() == 32
        && s_dt == DType::Uint8
        && q.get_biases()
            .is_some_and(|b| b.dtype().ok() == Some(DType::Float16))
    {
        MoeQuantMode::Q5K
    } else if q.mode() == MXFP8_MODE
        && q.bits() == 8
        && q.group_size() == 32
        && q.get_biases().is_none()
        && s_dt == DType::Uint8
    {
        MoeQuantMode::Mxfp8
    } else {
        return Ok(None);
    };
    Ok(Some((q, mode)))
}

/// Slice row `row_idx` out of a 3D `[G, T, D]` tensor as contiguous `[T, D]`.
#[inline]
fn slice_stream_2d(stacked: &MxArray, idx: i64, t: i64, d: i64) -> Result<MxArray> {
    stacked
        .slice(&[idx, 0, 0], &[idx + 1, t, d])?
        .squeeze(Some(&[0]))
}

/// Run LoopController + sublayer RMSNorm on CUDA:
/// - `t_usize <= 16` (decode & verify tree): single-launch fused `loop_ctrl_norm_cuda`.
/// - `t_usize > 16` (prefill): `loop_pair_rms_norm_cuda` -> cuBLAS `h = pair @ down_proj_t` ->
///   fused `loop_up_blend_rms_norm_cuda` (eliminating the `[G, T, 2*D]` `gf` DRAM intermediate
///   and second cuBLAS launch for BF16 weights on SM >= 8.0), falling back cleanly to
///   `gf = h @ gain_forget_proj_t` -> `loop_blend_rms_norm_cuda` when declined (e.g. FP32
///   loop controller weights via `MLX_INKLING2_LOOP_BF16=0`).
#[allow(clippy::too_many_arguments)]
fn run_cuda_loop_ctrl_norm(
    dest: &MxArray,
    msg: &MxArray,
    down_w: &MxArray,
    down_proj_t: &MxArray,
    gain_forget_w: &MxArray,
    gain_forget_proj_t: &MxArray,
    norm_w: &MxArray,
    norm_eps: f32,
    g_count: i32,
    squeeze_g1: bool,
    t_usize: usize,
) -> Result<Option<MxArray>> {
    if t_usize > 16 && (g_count == 3 || (g_count == 1 && squeeze_g1)) {
        let w_dtype = down_proj_t.dtype()?;
        if (w_dtype == DType::BFloat16 || w_dtype == DType::Float32)
            && gain_forget_proj_t.dtype()? == w_dtype
        {
            let out_is_bf16 = w_dtype == DType::BFloat16;
            if let Some(pair) =
                loop_pair_rms_norm_cuda(dest, msg, g_count, ATTN_RES_EPS, out_is_bf16)?
            {
                let h = pair.matmul(down_proj_t)?;
                if let Some(out) = loop_up_blend_rms_norm_cuda(
                    dest,
                    msg,
                    &h,
                    gain_forget_w,
                    norm_w,
                    g_count,
                    norm_eps,
                )? {
                    return Ok(Some(out));
                }
                let gf = h.matmul(gain_forget_proj_t)?;
                if let Some(out) =
                    loop_blend_rms_norm_cuda(dest, msg, &gf, norm_w, g_count, norm_eps)?
                {
                    return Ok(Some(out));
                }
            }
        }
    }
    loop_ctrl_norm_cuda(
        dest,
        msg,
        down_w,
        gain_forget_w,
        norm_w,
        norm_eps,
        g_count,
        squeeze_g1,
    )
}

/// Execute one logical visit (`0..48`) on CUDA using all Wave 2 fused primitives.
#[allow(clippy::too_many_arguments)]
fn forward_cuda_visit(
    layer: &Inkling2PhysicalLayer,
    blocks: &mut Vec<MxArray>,
    partial: &mut MxArray,
    block_size: usize,
    start_pos: usize,
    visit_cache: &mut super::cuda_kv::Inkling2CudaVisitCache,
    visit_conv: &mut VisitShortConvState,
    bank_sources: Option<&[MxArray]>,
    tree: Option<CudaTreeInputs<'_>>,
    mut visit_tape: Option<&mut VisitShortConvTape>,
    mut profile: Option<&mut CudaStageProfile>,
) -> Result<()> {
    let p_shape = partial.shape()?;
    let t_i64 = p_shape[0];
    let d_i64 = p_shape[1];
    let t_usize = usize::try_from(t_i64).map_err(|_| {
        Error::from_reason(format!("forward_cuda_visit: invalid token count {t_i64}"))
    })?;
    let record_tape = visit_tape.is_some();
    let update_hist = !record_tape;
    let conv_taps = tree.map(|tr| tr.conv_taps_i32);

    // Ensure KV storage capacity BEFORE capturing write_offset & launching kv_conv_prep_store.
    visit_cache.ensure_capacity_for(t_usize)?;
    let write_offset = visit_cache.write_offset();

    // 1. Optional cross-pass message read `[4, T, D]` FP32 (core passes 1 & 2, 1 CUDA launch).
    let messages = match (bank_sources, &layer.loop_ctrl) {
        (Some(bank), Some(ctrl)) => {
            let t0 = profile.is_some().then(Instant::now);
            let bank_refs: Vec<&MxArray> = bank.iter().collect();
            let msgs = block_attn_res_norm_cuda(
                &bank_refs,
                &ctrl.msg_queries,
                None,
                ATTN_RES_EPS,
                false,
            )?
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "forward_cuda_visit(phys={}): block_attn_res_norm_cuda declined cross-pass bank (n={})",
                    layer.phys_idx,
                    bank_refs.len()
                ))
            })?;
            if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t0) {
                CudaStageProfile::finish_stage(&mut prof.attn_res_us, started, &[&msgs])?;
            }
            Some((msgs, ctrl))
        }
        _ => None,
    };

    // 2. Attention Block AttnRes + optional batched G=3 LoopController + sublayer RMSNorm -> BF16.
    let (x_q, x_k_v) = if let Some(ref q_res) = layer.attn_res_query {
        let mut sources: Vec<&MxArray> = Vec::with_capacity(blocks.len() + 1);
        for b in blocks.iter() {
            sources.push(b);
        }
        sources.push(partial);

        if let Some((ref msgs, ctrl)) = messages {
            // Core pass > 0: 1 launch for attn_dest + 1 launch for batched G=3 loop_ctrl_norm!
            let t0 = profile.is_some().then(Instant::now);
            let attn_dest = block_attn_res_norm_cuda(
                &sources,
                q_res,
                None,
                ATTN_RES_EPS,
                layer.is_local,
            )?
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "forward_cuda_visit(phys={}): block_attn_res_norm_cuda declined attn_dest",
                    layer.phys_idx
                ))
            })?;
            if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t0) {
                CudaStageProfile::finish_stage(&mut prof.attn_res_us, started, &[&attn_dest])?;
            }

            let t_lc = profile.is_some().then(Instant::now);
            let msgs_qkv = msgs.slice(&[0, 0, 0], &[3, t_i64, d_i64])?;
            let norm_w = match &layer.attn_inp_norm {
                AttnInputNorm::Local(w) => w,
                AttnInputNorm::Global { qkv_norm, .. } => qkv_norm,
            };
            let x_qkv = run_cuda_loop_ctrl_norm(
                &attn_dest,
                &msgs_qkv,
                &ctrl.qkv_down_w,
                &ctrl.qkv_down_proj_t,
                &ctrl.qkv_gain_forget_w,
                &ctrl.qkv_gain_forget_proj_t,
                norm_w,
                layer.rms_norm_eps,
                3,
                false,
                t_usize,
            )?
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "forward_cuda_visit(phys={}): loop_ctrl_norm_cuda declined G=3 attention",
                    layer.phys_idx
                ))
            })?;
            let xq = slice_stream_2d(&x_qkv, 0, t_i64, d_i64)?;
            let xk = slice_stream_2d(&x_qkv, 1, t_i64, d_i64)?;
            let xv = slice_stream_2d(&x_qkv, 2, t_i64, d_i64)?;
            if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t_lc) {
                CudaStageProfile::finish_stage(&mut prof.loop_ctrl_us, started, &[&xq, &xk, &xv])?;
            }
            (xq, Some((xk, xv)))
        } else {
            // Pass 0 / Prelude / Coda: 1 fused launch for Block AttnRes + sublayer RMSNorm!
            let t0 = profile.is_some().then(Instant::now);
            let res = match &layer.attn_inp_norm {
                AttnInputNorm::Local(w) => {
                    let xq = block_attn_res_norm_cuda(
                        &sources,
                        q_res,
                        Some(w),
                        layer.rms_norm_eps,
                        true,
                    )?
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "forward_cuda_visit(phys={}): block_attn_res_norm_cuda declined SWA attn",
                            layer.phys_idx
                        ))
                    })?;
                    (xq, None)
                }
                AttnInputNorm::Global { qkv_norm, .. } => {
                    let x_qkv = block_attn_res_norm_cuda(
                        &sources,
                        q_res,
                        Some(qkv_norm),
                        layer.rms_norm_eps,
                        false,
                    )?
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "forward_cuda_visit(phys={}): block_attn_res_norm_cuda declined Global attn",
                            layer.phys_idx
                        ))
                    })?;
                    let xq = slice_stream_2d(&x_qkv, 0, t_i64, d_i64)?;
                    let xk = slice_stream_2d(&x_qkv, 1, t_i64, d_i64)?;
                    let xv = slice_stream_2d(&x_qkv, 2, t_i64, d_i64)?;
                    (xq, Some((xk, xv)))
                }
            };
            if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t0) {
                if let Some((ref xk, ref xv)) = res.1 {
                    CudaStageProfile::finish_stage(
                        &mut prof.attn_res_us,
                        started,
                        &[&res.0, xk, xv],
                    )?;
                } else {
                    CudaStageProfile::finish_stage(&mut prof.attn_res_us, started, &[&res.0])?;
                }
            }
            res
        }
    } else {
        // Physical layer 0 (`attn_res_query` is `None`): direct single-launch FP32 RMSNorm.
        let t0 = profile.is_some().then(Instant::now);
        let res = match &layer.attn_inp_norm {
            AttnInputNorm::Local(w) => (
                precise_f32_rms_norm_bf16(partial, w, layer.rms_norm_eps)?,
                None,
            ),
            AttnInputNorm::Global {
                q_norm,
                k_norm,
                v_norm,
                ..
            } => (
                precise_f32_rms_norm_bf16(partial, q_norm, layer.rms_norm_eps)?,
                Some((
                    precise_f32_rms_norm_bf16(partial, k_norm, layer.rms_norm_eps)?,
                    precise_f32_rms_norm_bf16(partial, v_norm, layer.rms_norm_eps)?,
                )),
            ),
        };
        if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t0) {
            if let Some((ref xk, ref xv)) = res.1 {
                CudaStageProfile::finish_stage(&mut prof.attn_res_us, started, &[&res.0, xk, xv])?;
            } else {
                CudaStageProfile::finish_stage(&mut prof.attn_res_us, started, &[&res.0])?;
            }
        }
        res
    };

    // 3. Attention linear projections (hits Wave 1A SymQ8 / Q8 prefill / BF16 few kernels).
    let t_proj = profile.is_some().then(Instant::now);
    let (q_raw, r_raw, gate_raw, k_raw, v_raw, sink_raw, dyn_k, dyn_v) = layer
        .self_attn
        .project_qrgkv_sink_and_dyn_conv(&x_q, x_k_v.as_ref().map(|(k, v)| (k, v)))?;
    if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t_proj) {
        let mut outs: Vec<&MxArray> = vec![&q_raw, &r_raw, &gate_raw, &k_raw, &v_raw, &sink_raw];
        if let Some(ref dk) = dyn_k {
            outs.push(dk);
        }
        if let Some(ref dv) = dyn_v {
            outs.push(dv);
        }
        CudaStageProfile::finish_stage(&mut prof.attn_proj_us, started, &outs)?;
    }

    if let Some(ref mut tape) = visit_tape {
        tape.k = Some(StreamShortConvTape {
            hist_before: visit_conv.k.clone(),
            rows: k_raw.clone(),
        });
        tape.v = Some(StreamShortConvTape {
            hist_before: visit_conv.v.clone(),
            rows: v_raw.clone(),
        });
    }

    // 4. Single-launch fused Q/R L2-clip + K/V ShortConv + K clip + V norm + Global tau + KV store.
    let start_pos_i32 = i32::try_from(start_pos).map_err(|_| {
        Error::from_reason(format!(
            "forward_cuda_visit: start_pos {start_pos} exceeds i32::MAX"
        ))
    })?;
    let write_off_i32 = i32::try_from(write_offset).map_err(|_| {
        Error::from_reason(format!(
            "forward_cuda_visit: write_offset {write_offset} exceeds i32::MAX"
        ))
    })?;
    let ring_cap_i32 = i32::try_from(visit_cache.ring_cap).map_err(|_| {
        Error::from_reason(format!(
            "forward_cuda_visit: ring_cap {} exceeds i32::MAX",
            visit_cache.ring_cap
        ))
    })?;

    let t_kv = profile.is_some().then(Instant::now);
    let prep_out = kv_conv_prep_store_cuda(&KvConvPrepParams {
        q_raw: &q_raw,
        r_raw: &r_raw,
        k_raw: &k_raw,
        v_raw: &v_raw,
        q_clip: &layer.self_attn.q_clip,
        rel_clip: &layer.self_attn.rel_clip,
        k_conv_w: &layer.self_attn.k_conv,
        v_conv_w: &layer.self_attn.v_conv,
        dyn_k: dyn_k.as_ref(),
        dyn_v: dyn_v.as_ref(),
        k_hist_in: &visit_conv.k,
        v_hist_in: &visit_conv.v,
        kv_k_cache: &visit_cache.k,
        kv_v_cache: &visit_cache.v,
        kv_k_scales: visit_cache.k_scales.as_ref(),
        kv_v_scales: visit_cache.v_scales.as_ref(),
        conv_taps,
        depths: tree.map(|tr| tr.depths_i32),
        start_pos: start_pos_i32,
        write_offset: write_off_i32,
        ring_cap: ring_cap_i32,
        is_global: !layer.is_local,
        log_n_floor: layer.self_attn.log_scaling_n_floor as f32,
        log_alpha: layer.self_attn.log_scaling_alpha as f32,
        update_hist,
    })?
    .ok_or_else(|| {
        Error::from_reason(format!(
            "forward_cuda_visit(phys={}): kv_conv_prep_store_cuda declined inputs \
             [q={:?}/{:?}, r={:?}/{:?}, k={:?}/{:?}, v={:?}/{:?}, \
              q_clip={:?}/{:?}, r_clip={:?}/{:?}, k_conv={:?}/{:?}, v_conv={:?}/{:?}, \
              kh={:?}/{:?}, vh={:?}/{:?}, kc={:?}/{:?}, vc={:?}/{:?}, ks={:?}, \
              start_pos={start_pos_i32}, write_off={write_off_i32}, ring_cap={ring_cap_i32}]",
            layer.phys_idx,
            q_raw.shape().ok().map(|s| s.to_vec()),
            q_raw.dtype().ok(),
            r_raw.shape().ok().map(|s| s.to_vec()),
            r_raw.dtype().ok(),
            k_raw.shape().ok().map(|s| s.to_vec()),
            k_raw.dtype().ok(),
            v_raw.shape().ok().map(|s| s.to_vec()),
            v_raw.dtype().ok(),
            layer.self_attn.q_clip.shape().ok().map(|s| s.to_vec()),
            layer.self_attn.q_clip.dtype().ok(),
            layer.self_attn.rel_clip.shape().ok().map(|s| s.to_vec()),
            layer.self_attn.rel_clip.dtype().ok(),
            layer.self_attn.k_conv.shape().ok().map(|s| s.to_vec()),
            layer.self_attn.k_conv.dtype().ok(),
            layer.self_attn.v_conv.shape().ok().map(|s| s.to_vec()),
            layer.self_attn.v_conv.dtype().ok(),
            visit_conv.k.shape().ok().map(|s| s.to_vec()),
            visit_conv.k.dtype().ok(),
            visit_conv.v.shape().ok().map(|s| s.to_vec()),
            visit_conv.v.dtype().ok(),
            visit_cache.k.shape().ok().map(|s| s.to_vec()),
            visit_cache.k.dtype().ok(),
            visit_cache.v.shape().ok().map(|s| s.to_vec()),
            visit_cache.v.dtype().ok(),
            visit_cache.k_scales.as_ref().and_then(|s| s.dtype().ok()),
        ))
    })?;

    // Propagate in-place cache alias handles into the cache so downstream SDPA & eval track writes!
    visit_cache.k = prep_out.kv_k_alias;
    visit_cache.v = prep_out.kv_v_alias;
    if let (Some(ks), Some(vs)) = (prep_out.ks_alias, prep_out.vs_alias) {
        visit_cache.k_scales = Some(ks);
        visit_cache.v_scales = Some(vs);
    }
    if let (Some(kh), Some(vh)) = (prep_out.k_hist, prep_out.v_hist) {
        visit_conv.k = kh;
        visit_conv.v = vh;
    }
    if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t_kv) {
        let mut outs: Vec<&MxArray> =
            vec![&prep_out.q, &prep_out.r, &visit_cache.k, &visit_cache.v];
        if let Some(ref ks) = visit_cache.k_scales {
            outs.push(ks);
        }
        if let Some(ref vs) = visit_cache.v_scales {
            outs.push(vs);
        }
        if update_hist {
            outs.push(&visit_conv.k);
            outs.push(&visit_conv.v);
        }
        CudaStageProfile::finish_stage(&mut prof.kv_prep_store_us, started, &outs)?;
    }

    // 5. Fused Tensor Core SDPA (`sdpa_few_cuda` for T <= 16, `sdpa_prefill_cuda` for T > 16)
    //    with fused relative bias, per-token sink, and attention gate -> `[T, 3072]` BF16!
    let cache_len_i32 = i32::try_from(visit_cache.cache_len).map_err(|_| {
        Error::from_reason(format!(
            "forward_cuda_visit: cache_len {} exceeds i32::MAX",
            visit_cache.cache_len
        ))
    })?;
    let kv_start_i32 = i32::try_from(visit_cache.kv_start_pos).map_err(|_| {
        Error::from_reason(format!(
            "forward_cuda_visit: kv_start_pos {} exceeds i32::MAX",
            visit_cache.kv_start_pos
        ))
    })?;
    let kv_phys_i32 = i32::try_from(visit_cache.kv_phys_base).map_err(|_| {
        Error::from_reason(format!(
            "forward_cuda_visit: kv_phys_base {} exceeds i32::MAX",
            visit_cache.kv_phys_base
        ))
    })?;
    let sw_i32 = i32::try_from(visit_cache.sliding_window).map_err(|_| {
        Error::from_reason(format!(
            "forward_cuda_visit: sliding_window {} exceeds i32::MAX",
            visit_cache.sliding_window
        ))
    })?;

    let sdpa_params = SdpaParams {
        q: &prep_out.q,
        r: &prep_out.r,
        k_cache: &visit_cache.k,
        v_cache: &visit_cache.v,
        k_scales: visit_cache.k_scales.as_ref(),
        v_scales: visit_cache.v_scales.as_ref(),
        rel_bias_w: &layer.self_attn.rel_bias_w,
        gate: &gate_raw,
        sink: &sink_raw,
        tree_depths: tree.map(|tr| tr.depths_i32),
        tree_mask: tree.map(|tr| tr.ancestors_mask_u32),
        start_pos: start_pos_i32,
        cache_len: cache_len_i32,
        kv_start_pos: kv_start_i32,
        kv_phys_base: kv_phys_i32,
        write_offset: write_off_i32,
        ring_cap: ring_cap_i32,
        is_local: layer.is_local,
        sliding_window: sw_i32,
    };

    let t_sdpa = profile.is_some().then(Instant::now);
    let attn_gated = if t_usize <= 16 {
        sdpa_few_cuda(&sdpa_params)?.ok_or_else(|| {
            Error::from_reason(format!(
                "forward_cuda_visit(phys={}): sdpa_few_cuda declined inputs (T={t_usize})",
                layer.phys_idx
            ))
        })?
    } else {
        if tree.is_some() {
            return Err(Error::from_reason(format!(
                "forward_cuda_visit: tree verification requires T <= 16, got T={t_usize}"
            )));
        }
        sdpa_prefill_cuda(&sdpa_params)?.ok_or_else(|| {
            Error::from_reason(format!(
                "forward_cuda_visit(phys={}): sdpa_prefill_cuda declined inputs (T={t_usize})",
                layer.phys_idx
            ))
        })?
    };
    if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t_sdpa) {
        CudaStageProfile::finish_stage(&mut prof.sdpa_us, started, &[&attn_gated])?;
    }

    // 6. Output projection + fused `attn_post_conv` (+ residual add unless block boundary).
    let t_oproj = profile.is_some().then(Instant::now);
    let o_proj_out = forward_linear_proj_fast(&layer.self_attn.o_proj, &attn_gated)?;
    if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t_oproj) {
        CudaStageProfile::finish_stage(&mut prof.o_proj_us, started, &[&o_proj_out])?;
    }
    if let Some(ref mut tape) = visit_tape {
        tape.attn_post = Some(StreamShortConvTape {
            hist_before: visit_conv.attn_post.clone(),
            rows: o_proj_out.clone(),
        });
    }

    let is_block_boundary = layer.phys_idx.is_multiple_of(block_size);
    let attn_residual = if is_block_boundary {
        blocks.push(partial.clone());
        None
    } else {
        Some(&*partial)
    };
    let t_sconv = profile.is_some().then(Instant::now);
    let sconv_attn = sconv_k4_residual_cuda(
        &o_proj_out,
        &layer.self_attn.post_conv,
        &visit_conv.attn_post,
        attn_residual,
        conv_taps,
        update_hist,
    )?
    .ok_or_else(|| {
        Error::from_reason(format!(
            "forward_cuda_visit(phys={}): sconv_k4_residual_cuda declined attn_post",
            layer.phys_idx
        ))
    })?;
    let ffn_inp = sconv_attn.y;
    if let Some(h) = sconv_attn.hist {
        visit_conv.attn_post = h;
    }
    if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t_sconv) {
        if update_hist {
            CudaStageProfile::finish_stage(
                &mut prof.attn_sconv_us,
                started,
                &[&ffn_inp, &visit_conv.attn_post],
            )?;
        } else {
            CudaStageProfile::finish_stage(&mut prof.attn_sconv_us, started, &[&ffn_inp])?;
        }
    }

    // 7. MLP Block AttnRes + optional LoopController (G=1) + `ffn_norm` -> `[T, D]` BF16.
    let mut mlp_sources: Vec<&MxArray> = Vec::with_capacity(blocks.len() + 1);
    for b in blocks.iter() {
        mlp_sources.push(b);
    }
    mlp_sources.push(&ffn_inp);

    let x_mlp = if let Some((ref msgs, ctrl)) = messages {
        let t0 = profile.is_some().then(Instant::now);
        let dest_mlp =
            block_attn_res_norm_cuda(&mlp_sources, &layer.ffn_res_query, None, ATTN_RES_EPS, true)?
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "forward_cuda_visit(phys={}): block_attn_res_norm_cuda declined mlp_dest",
                        layer.phys_idx
                    ))
                })?;
        if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t0) {
            CudaStageProfile::finish_stage(&mut prof.attn_res_us, started, &[&dest_mlp])?;
        }
        let t_lc = profile.is_some().then(Instant::now);
        let msg_mlp = slice_stream_2d(msgs, 3, t_i64, d_i64)?;
        let out = run_cuda_loop_ctrl_norm(
            &dest_mlp,
            &msg_mlp,
            &ctrl.mlp_ctrl.down_proj_w,
            &ctrl.mlp_ctrl.down_proj_t,
            &ctrl.mlp_ctrl.gain_forget_w,
            &ctrl.mlp_ctrl.gain_forget_proj_t,
            &layer.ffn_norm,
            layer.rms_norm_eps,
            1,
            true,
            t_usize,
        )?
        .ok_or_else(|| {
            Error::from_reason(format!(
                "forward_cuda_visit(phys={}): loop_ctrl_norm_cuda declined G=1 mlp",
                layer.phys_idx
            ))
        })?;
        if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t_lc) {
            CudaStageProfile::finish_stage(&mut prof.loop_ctrl_us, started, &[&out])?;
        }
        out
    } else {
        let t0 = profile.is_some().then(Instant::now);
        let out = block_attn_res_norm_cuda(
            &mlp_sources,
            &layer.ffn_res_query,
            Some(&layer.ffn_norm),
            layer.rms_norm_eps,
            true,
        )?
        .ok_or_else(|| {
            Error::from_reason(format!(
                "forward_cuda_visit(phys={}): block_attn_res_norm_cuda declined mlp_norm",
                layer.phys_idx
            ))
        })?;
        if let (Some(prof), Some(started)) = (profile.as_deref_mut(), t0) {
            CudaStageProfile::finish_stage(&mut prof.attn_res_us, started, &[&out])?;
        }
        out
    };

    // 8. Fused CUDA Sparse MoE + `ffn_post_conv` + residual `+ ffn_inp` (+ optional rollback tape).
    let t_moe = profile.is_some().then(Instant::now);
    let gu_classified = match &layer.moe.gate_up_proj {
        ExpertGateUp::Fused(proj) => classify_cuda_moe_proj(proj, true)?,
        ExpertGateUp::Split { .. } => None,
    };
    let fused_moe_out = if let (Some((gu, gu_mode)), Some((dn, dn_mode))) = (
        gu_classified,
        classify_cuda_moe_proj(&layer.moe.down_proj, false)?,
    ) {
        let res = moe_forward_cuda(&MoeForwardParams {
            x: &x_mlp,
            router_w: &layer.moe.router_weight_f32,
            router_bias: layer.moe.router_bias_f32.as_ref(),
            gu_w: gu.get_weight(),
            gu_scales: gu.get_scales(),
            gu_biases: gu.get_biases(),
            gu_outer_scale: gu.outer_scale(),
            gu_mode,
            gu_interleaved: layer.moe.gate_up_interleaved,
            dn_w: dn.get_weight(),
            dn_scales: dn.get_scales(),
            dn_biases: dn.get_biases(),
            dn_outer_scale: dn.outer_scale(),
            dn_mode,
            route_scale: layer.moe.route_scale as f32,
            post_conv_w: Some(&layer.moe.post_conv),
            hist_in: Some(&visit_conv.mlp_post),
            residual: Some(&ffn_inp),
            conv_taps,
            update_hist,
            emit_pre_conv: record_tape,
        })?;
        if res.is_none() {
            tracing::warn!(
                "forward_cuda_visit(phys={}): moe_forward_cuda declined classified modes gu={gu_mode:?} dn={dn_mode:?}; falling back",
                layer.phys_idx
            );
        }
        res
    } else {
        None
    };

    if let Some(moe_res) = fused_moe_out {
        if let Some(ref mut tape) = visit_tape {
            let pre_conv = moe_res.pre_conv.ok_or_else(|| {
                Error::from_reason(format!(
                    "forward_cuda_visit(phys={}): moe_forward_cuda missing pre_conv tape",
                    layer.phys_idx
                ))
            })?;
            tape.mlp_post = Some(StreamShortConvTape {
                hist_before: visit_conv.mlp_post.clone(),
                rows: pre_conv,
            });
        }
        if let Some(h) = moe_res.hist {
            visit_conv.mlp_post = h;
        }
        *partial = moe_res.y;
    } else {
        // Fallback for non-fused expert quant formats (e.g., Q4K/Q6K/dense): run generic MoE
        // pre-conv then fuse `post_conv` + residual + tree taps via `sconv_k4_residual_cuda`.
        let pre_conv = layer.moe.forward_pre_conv(&x_mlp)?;
        if let Some(ref mut tape) = visit_tape {
            tape.mlp_post = Some(StreamShortConvTape {
                hist_before: visit_conv.mlp_post.clone(),
                rows: pre_conv.clone(),
            });
        }
        let sconv_mlp = sconv_k4_residual_cuda(
            &pre_conv,
            &layer.moe.post_conv,
            &visit_conv.mlp_post,
            Some(&ffn_inp),
            conv_taps,
            update_hist,
        )?
        .ok_or_else(|| {
            Error::from_reason(format!(
                "forward_cuda_visit(phys={}): sconv_k4_residual_cuda declined mlp_post fallback",
                layer.phys_idx
            ))
        })?;
        if let Some(h) = sconv_mlp.hist {
            visit_conv.mlp_post = h;
        }
        *partial = sconv_mlp.y;
    }
    if let (Some(prof), Some(started)) = (profile, t_moe) {
        let mut outs: Vec<&MxArray> = vec![&*partial];
        if update_hist {
            outs.push(&visit_conv.mlp_post);
        }
        if let Some(ref tape) = visit_tape
            && let Some(ref s) = tape.mlp_post
        {
            outs.push(&s.rows);
        }
        let acc = if layer.phys_idx < 20 {
            &mut prof.moe_nvfp4_us
        } else {
            &mut prof.moe_l20_23_us
        };
        CudaStageProfile::finish_stage(acc, started, &outs)?;
    }

    Ok(())
}

/// Run the final LM head + unpadded vocab slice + softcap (or zero-DRAM greedy argmax) on
/// `[M, D]` BF16 `normed_out`. Uses `lm_head_softcap_cuda` whenever `M <= 16` and weights match
/// SymQ8 / AffineQ8 / BF16, with automatic fallback for other formats or `M > 16`.
fn run_cuda_lm_head(
    lm_head: &LinearProj,
    normed_out: &MxArray,
    unpadded_vocab: usize,
    softcap: f64,
    want_argmax: bool,
) -> Result<MxArray> {
    let shape = normed_out.shape()?;
    let m = usize::try_from(shape[0]).map_err(|_| {
        Error::from_reason(format!("run_cuda_lm_head: invalid row count {}", shape[0]))
    })?;
    let unpadded_i32 = i32::try_from(unpadded_vocab).map_err(|_| {
        Error::from_reason(format!(
            "run_cuda_lm_head: unpadded_vocab {unpadded_vocab} exceeds i32::MAX"
        ))
    })?;
    let out_mode = if want_argmax {
        LmHeadOutMode::GreedyArgmax
    } else {
        LmHeadOutMode::LogitsF32
    };

    if m <= 16 {
        let fast = match lm_head {
            LinearProj::Quantized(ql)
                if ql.grouped_moe_plain_projection()
                    && ql.mode() == DEFAULT_QUANT_MODE
                    && ql.bits() == 8
                    && ql.group_size() == 32 =>
            {
                let biases = if ql.is_symmetric_q8() {
                    None
                } else {
                    ql.get_biases()
                };
                lm_head_softcap_cuda(
                    normed_out,
                    ql.get_weight(),
                    Some(ql.get_scales()),
                    biases,
                    unpadded_i32,
                    softcap as f32,
                    out_mode,
                )?
            }
            LinearProj::Standard(lin)
                if lin.get_bias().is_none() && lin.get_weight().dtype()? == DType::BFloat16 =>
            {
                let w = lin.get_weight();
                lm_head_softcap_cuda(
                    normed_out,
                    &w,
                    None,
                    None,
                    unpadded_i32,
                    softcap as f32,
                    out_mode,
                )?
            }
            _ => None,
        };
        if let Some(res) = fast {
            return Ok(res);
        }
    }

    // Fallback (M > 16 full-sequence logits or unsupported lm_head quant format):
    // always stay packed and keep full FP32 logit precision via `forward_logits_keep_f32`
    // (avoids materializing a dense BF16 copy of `lm_head` or rounding logits before softcap).
    let logits_full = lm_head.forward_logits_keep_f32(normed_out)?;
    let vocab_cols = logits_full.shape()?[1];
    let unpadded_i64 = unpadded_vocab as i64;
    let sliced = if unpadded_i64 < vocab_cols {
        logits_full.slice(&[0, 0], &[shape[0], unpadded_i64])?
    } else {
        logits_full
    };
    let mut logits = if sliced.dtype()? == DType::Float32 {
        sliced
    } else {
        sliced.astype(DType::Float32)?
    };
    if softcap > 0.0 {
        logits = logits
            .mul_scalar(1.0 / softcap)?
            .tanh()?
            .mul_scalar(softcap)?;
    }
    if want_argmax {
        let idx = logits.argmax(-1, Some(false))?;
        if idx.dtype()? == DType::Int32 {
            Ok(idx)
        } else {
            idx.astype(DType::Int32)
        }
    } else {
        Ok(logits)
    }
}

#[inline]
fn lm_head_last_row_only(lm_head_mode: CudaLmHeadMode, t_usize: usize) -> bool {
    match lm_head_mode {
        CudaLmHeadMode::LogitsF32Last => true,
        CudaLmHeadMode::LogitsF32 => t_usize > 16,
        CudaLmHeadMode::LogitsF32All | CudaLmHeadMode::GreedyArgmax | CudaLmHeadMode::Skip => false,
    }
}

impl Inkling2Inner {
    /// Whether this model instance should execute on the Inkling-2-Mini CUDA fast path.
    #[inline]
    pub(crate) fn use_cuda_fast(&self) -> bool {
        self.paged.is_none() && cuda_fast_supported_for(&self.config)
    }

    /// Execute one forward pass over device-resident 1D token IDs `ids_arr` (`[T]` Int32/Uint32)
    /// on the Inkling-2-Mini CUDA fast backbone.
    ///
    /// - When `!record_tape` (normal prefill & AR decode): commits `T` linear tokens into `cache`
    ///   (updating SWA circular ring cursors & `cache.pos_offset += T`).
    /// - When `record_tape` (speculative linear/tree verification): keeps `cache` cursors and
    ///   ShortConv histories unadvanced and returns `Some(Inkling2ShortConvTape)` so the caller
    ///   can commit only the accepted tokens/branch after verification.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_cuda_ids(
        &self,
        ids_arr: &MxArray,
        image_embeddings: Option<&MxArray>,
        cache: &mut Inkling2CudaCache,
        tap_visits: Option<&[usize]>,
        tree: Option<CudaTreeInputs<'_>>,
        record_tape: bool,
        lm_head_mode: CudaLmHeadMode,
    ) -> Result<CudaForwardOutput> {
        // Resolve multimodal IMAGE_PAD positions ONLY when image embeddings are actually supplied
        // (never touches host memory during text prefill or AR decode!).
        let pad_positions_owned: Vec<usize>;
        let splice = if let Some(img_emb) = image_embeddings {
            let ids_host = if ids_arr.dtype()? == DType::Int32 {
                ids_arr.to_int32()?
            } else {
                ids_arr.astype(DType::Int32)?.to_int32()?
            };
            pad_positions_owned = ids_host
                .iter()
                .enumerate()
                .filter_map(|(idx, &tok)| (tok as u32 == IMAGE_PAD_ID).then_some(idx))
                .collect();
            let img_rows = img_emb.shape()?[0] as usize;
            if pad_positions_owned.len() != img_rows {
                return Err(Error::from_reason(format!(
                    "Inkling2Inner::forward_cuda_ids: input has {} IMAGE_PAD tokens but image_embeddings has {img_rows} rows",
                    pad_positions_owned.len()
                )));
            }
            Some(ChunkImageSplice {
                embeddings: img_emb,
                pad_positions: &pad_positions_owned,
            })
        } else {
            None
        };

        self.forward_cuda_ids_inner(
            ids_arr,
            splice,
            cache,
            tap_visits,
            tree,
            record_tape,
            lm_head_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_cuda_ids_inner(
        &self,
        ids_arr: &MxArray,
        image_splice: Option<ChunkImageSplice<'_>>,
        cache: &mut Inkling2CudaCache,
        tap_visits: Option<&[usize]>,
        tree: Option<CudaTreeInputs<'_>>,
        record_tape: bool,
        lm_head_mode: CudaLmHeadMode,
    ) -> Result<CudaForwardOutput> {
        let ids_1d = if ids_arr.ndim()? == 1 {
            ids_arr.clone()
        } else {
            ids_arr.reshape(&[-1])?
        };
        let t_i64 = ids_1d.shape_at(0)?;
        let t_usize = usize::try_from(t_i64).map_err(|_| {
            Error::from_reason(format!(
                "Inkling2Inner::forward_cuda_ids: invalid token count {t_i64}"
            ))
        })?;
        if t_usize == 0 {
            return Err(Error::from_reason(
                "Inkling2Inner::forward_cuda_ids: empty ids_arr",
            ));
        }

        let num_visits = self.config.num_layer_visits();
        if cache.visits.len() != num_visits || cache.conv_state.visits.len() != num_visits {
            return Err(Error::from_reason(format!(
                "Inkling2Inner::forward_cuda_ids: cache visit count mismatch (kv={}, conv={}, expected={num_visits})",
                cache.visits.len(),
                cache.conv_state.visits.len()
            )));
        }

        let eps = self.config.rms_norm_eps as f32;
        let mut profile = cuda_profile_enabled().then(CudaStageProfile::default);
        let _prof_graphs_guard = profile.is_some().then(CudaGraphsGuard::disable_for_prefill);

        // 1. Single-launch fused token embeddings + precise FP32 `embed_norm` -> BF16
        //    (covers both dense BF16 and GGUF affine Q8_0 with fallback for other formats).
        let t_emb = profile.is_some().then(Instant::now);
        let mut hidden = if let Some(h) =
            try_embed_rmsnorm_cuda(&self.embed_tokens, &ids_1d, &self.embed_norm, eps)?
        {
            h
        } else {
            let raw_emb = self.embed_tokens.forward(&ids_1d)?;
            if raw_emb.dtype()? == DType::BFloat16 {
                precise_f32_rms_norm_bf16(&raw_emb, &self.embed_norm, eps)?
            } else {
                apply_sublayer_rms_norm(&raw_emb, &self.embed_norm, eps, DType::BFloat16)?
            }
        };

        // 2. Optional multimodal image embedding splice AFTER `embed_norm`.
        if let Some(splice) = image_splice
            && !splice.pad_positions.is_empty()
        {
            hidden = overwrite_image_pad_rows(
                &hidden,
                splice.embeddings,
                splice.pad_positions,
                DType::BFloat16,
            )?;
        }
        if let (Some(prof), Some(started)) = (profile.as_mut(), t_emb) {
            CudaStageProfile::finish_stage(&mut prof.embed_us, started, &[&hidden])?;
        }

        // 3. Execute Prelude -> Core (3 passes with raw cross-pass bank) -> Coda.
        let start_pos = cache.pos_offset;
        let block_size = self.config.n_layers_per_attn_res_block;
        let taps_slice = tap_visits.unwrap_or(&[]);
        let mut captured_taps: Vec<MxArray> = Vec::with_capacity(taps_slice.len());
        let mut tape_opt = record_tape.then(|| Inkling2ShortConvTape::new(num_visits));

        let mut blocks: Vec<MxArray> = Vec::with_capacity(6);
        let mut partial = hidden;

        if let Some(ref lc) = self.config.loop_config {
            let mut visit_idx = 0usize;

            // Prelude: physical `0..core_start`, pass 0
            for layer in &self.layers[..lc.core_start] {
                if taps_slice.contains(&visit_idx) {
                    captured_taps.push(partial.clone());
                }
                forward_cuda_visit(
                    layer,
                    &mut blocks,
                    &mut partial,
                    block_size,
                    start_pos,
                    &mut cache.visits[visit_idx],
                    &mut cache.conv_state.visits[visit_idx],
                    None,
                    tree,
                    tape_opt.as_mut().map(|t| &mut t.visits[visit_idx]),
                    profile.as_mut(),
                )?;
                visit_idx += 1;
            }

            let prelude_blocks = blocks.clone();
            let prelude_partial = partial.clone();
            let prelude_block_count = prelude_blocks.len();

            for pass in 0..lc.num_passes {
                // Cross-pass bank contains previous pass's core blocks ONLY (raw BF16 sources;
                // `block_attn_res_norm_cuda` normalizes them on the fly with zero copies!).
                let bank: Option<Vec<MxArray>> = if pass > 0 {
                    let core_tail = blocks.len().saturating_sub(prelude_block_count + 1);
                    let mut bank_vec = Vec::with_capacity(core_tail + 1);
                    bank_vec.extend_from_slice(&blocks[prelude_block_count + 1..]);
                    bank_vec.push(partial.clone());
                    blocks = prelude_blocks.clone();
                    partial = prelude_partial.clone();
                    Some(bank_vec)
                } else {
                    None
                };

                for layer in &self.layers[lc.core_start..lc.core_end] {
                    if taps_slice.contains(&visit_idx) {
                        captured_taps.push(partial.clone());
                    }
                    forward_cuda_visit(
                        layer,
                        &mut blocks,
                        &mut partial,
                        block_size,
                        start_pos,
                        &mut cache.visits[visit_idx],
                        &mut cache.conv_state.visits[visit_idx],
                        bank.as_deref(),
                        tree,
                        tape_opt.as_mut().map(|t| &mut t.visits[visit_idx]),
                        profile.as_mut(),
                    )?;
                    visit_idx += 1;
                }
            }

            // Coda: physical `core_end..num_physical_layers`, pass 0
            for layer in &self.layers[lc.core_end..] {
                if taps_slice.contains(&visit_idx) {
                    captured_taps.push(partial.clone());
                }
                forward_cuda_visit(
                    layer,
                    &mut blocks,
                    &mut partial,
                    block_size,
                    start_pos,
                    &mut cache.visits[visit_idx],
                    &mut cache.conv_state.visits[visit_idx],
                    None,
                    tree,
                    tape_opt.as_mut().map(|t| &mut t.visits[visit_idx]),
                    profile.as_mut(),
                )?;
                visit_idx += 1;
            }
        } else {
            for (visit_idx, layer) in self.layers.iter().enumerate() {
                if taps_slice.contains(&visit_idx) {
                    captured_taps.push(partial.clone());
                }
                forward_cuda_visit(
                    layer,
                    &mut blocks,
                    &mut partial,
                    block_size,
                    start_pos,
                    &mut cache.visits[visit_idx],
                    &mut cache.conv_state.visits[visit_idx],
                    None,
                    tree,
                    tape_opt.as_mut().map(|t| &mut t.visits[visit_idx]),
                    profile.as_mut(),
                )?;
            }
        }

        // 4. Advance KV cache cursors on normal linear prefill/AR steps (NOT speculative verify).
        if !record_tape {
            cache.commit_linear(t_usize);
        }

        // 5. Final Output AttnRes + `output_norm` + optional tap `num_visits` (48) + LM Head.
        let capture_final_norm = taps_slice.contains(&num_visits);
        if lm_head_mode == CudaLmHeadMode::Skip && !capture_final_norm {
            if let Some(ref prof) = profile {
                prof.print(t_usize, lm_head_mode, tree.is_some());
            }
            return Ok(CudaForwardOutput {
                logits_or_tokens: None,
                taps: captured_taps,
                tape: tape_opt,
            });
        }

        let last_row_only = lm_head_last_row_only(lm_head_mode, t_usize);

        let t_out_norm = profile.is_some().then(Instant::now);
        let normed_for_head = if last_row_only && !capture_final_norm && t_usize > 1 {
            // Fast prefill tail: slice ONLY the final row across all 5 sources before output AttnRes!
            let last_start = t_i64 - 1;
            let mut last_blocks: Vec<MxArray> = Vec::with_capacity(blocks.len() + 1);
            for b in &blocks {
                last_blocks.push(b.slice_axis(0, last_start, t_i64)?);
            }
            last_blocks.push(partial.slice_axis(0, last_start, t_i64)?);
            let src_refs: Vec<&MxArray> = last_blocks.iter().collect();
            block_attn_res_norm_cuda(
                &src_refs,
                &self.output_res_query,
                Some(&self.output_norm),
                eps,
                true,
            )?
            .ok_or_else(|| {
                Error::from_reason(
                    "Inkling2Inner::forward_cuda_ids: block_attn_res_norm_cuda declined output_norm (last row)",
                )
            })?
        } else {
            let mut src_refs: Vec<&MxArray> = Vec::with_capacity(blocks.len() + 1);
            for b in &blocks {
                src_refs.push(b);
            }
            src_refs.push(&partial);
            let normed_all = block_attn_res_norm_cuda(
                &src_refs,
                &self.output_res_query,
                Some(&self.output_norm),
                eps,
                true,
            )?
            .ok_or_else(|| {
                Error::from_reason(
                    "Inkling2Inner::forward_cuda_ids: block_attn_res_norm_cuda declined output_norm",
                )
            })?;
            if capture_final_norm {
                captured_taps.push(normed_all.clone());
            }
            if lm_head_mode == CudaLmHeadMode::Skip {
                if let (Some(prof), Some(started)) = (profile.as_mut(), t_out_norm) {
                    CudaStageProfile::finish_stage(&mut prof.attn_res_us, started, &[&normed_all])?;
                    prof.print(t_usize, lm_head_mode, tree.is_some());
                }
                return Ok(CudaForwardOutput {
                    logits_or_tokens: None,
                    taps: captured_taps,
                    tape: tape_opt,
                });
            }
            if last_row_only && t_usize > 1 {
                normed_all.slice_axis(0, t_i64 - 1, t_i64)?
            } else {
                normed_all
            }
        };
        if let (Some(prof), Some(started)) = (profile.as_mut(), t_out_norm) {
            CudaStageProfile::finish_stage(&mut prof.attn_res_us, started, &[&normed_for_head])?;
        }

        let unpadded_vocab = self.config.effective_unpadded_vocab_size();
        let softcap = self.config.final_logit_softcapping;
        let want_argmax = lm_head_mode == CudaLmHeadMode::GreedyArgmax;
        let t_head = profile.is_some().then(Instant::now);
        let logits_or_tokens = run_cuda_lm_head(
            &self.lm_head,
            &normed_for_head,
            unpadded_vocab,
            softcap,
            want_argmax,
        )?;
        if let (Some(prof), Some(started)) = (profile.as_mut(), t_head) {
            CudaStageProfile::finish_stage(&mut prof.lm_head_us, started, &[&logits_or_tokens])?;
            prof.print(t_usize, lm_head_mode, tree.is_some());
        }

        Ok(CudaForwardOutput {
            logits_or_tokens: Some(logits_or_tokens),
            taps: captured_taps,
            tape: tape_opt,
        })
    }

    /// Chunked CUDA prefill over `prompt_tokens`, wrapped in [`CudaGraphsGuard::disable_for_prefill`].
    ///
    /// Processes `CUDA_PREFILL_CHUNK = 1024` tokens per chunk (matching SWA `ring_cap = 2048`),
    /// skips LM head on intermediate chunks, evaluates KV + conv state cleanly between chunks so
    /// peak memory stays flat, and concatenates captured taps across chunks when `tap_visits` is
    /// requested. When `tap_tail_limit` is `Some(max_rows)` (e.g. DFlash2's bounded sliding
    /// window `sliding_window - 1`), chunks completely before `total_len - max_rows` skip tap
    /// capture altogether and only the trailing `min(total_len, max_rows)` rows are retained.
    ///
    /// Returns `(last_logits, taps)` where `last_logits` has shape `[1, unpadded_vocab]` FP32.
    pub(crate) fn prefill_cuda_tokens(
        &self,
        prompt_tokens: &[u32],
        image_embeddings: Option<&MxArray>,
        cache: &mut Inkling2CudaCache,
        tap_visits: Option<&[usize]>,
        tap_tail_limit: Option<usize>,
    ) -> Result<(MxArray, Vec<MxArray>)> {
        if prompt_tokens.is_empty() {
            return Err(Error::from_reason(
                "Inkling2Inner::prefill_cuda_tokens: empty prompt_tokens",
            ));
        }

        // Disable CUDA Graph capture for variable-length prefill chunks; re-enabled on drop!
        let _graphs_guard = CudaGraphsGuard::disable_for_prefill();

        let total_len = prompt_tokens.len();
        if let Some(img_emb) = image_embeddings {
            let total_pads = prompt_tokens
                .iter()
                .filter(|&&tok| tok == IMAGE_PAD_ID)
                .count();
            let img_rows = img_emb.shape()?[0] as usize;
            if total_pads != img_rows {
                return Err(Error::from_reason(format!(
                    "Inkling2Inner::prefill_cuda_tokens: prompt has {total_pads} IMAGE_PAD tokens but image_embeddings has {img_rows} rows"
                )));
            }
        }

        let want_taps = tap_visits.is_some_and(|tv| !tv.is_empty());
        let mut tap_chunks: Vec<Vec<MxArray>> = Vec::new();
        let mut img_cursor = 0usize;
        let mut last_logits: Option<MxArray> = None;

        let mut offset = 0usize;
        while offset < total_len {
            if self.is_turn_cancelled() {
                return Err(Error::from_reason("prefill cancelled"));
            }
            let end = (offset + CUDA_PREFILL_CHUNK).min(total_len);
            let chunk = &prompt_tokens[offset..end];
            let is_last = end == total_len;

            let mut chunk_i32 = Vec::with_capacity(chunk.len());
            for &tok in chunk {
                chunk_i32.push(tok as i32);
            }
            let ids_arr = MxArray::from_int32(&chunk_i32, &[chunk.len() as i64])?;

            let pad_positions: Vec<usize>;
            let chunk_img_owned: MxArray;
            let splice = if let Some(img_emb) = image_embeddings {
                pad_positions = chunk
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &tok)| (tok == IMAGE_PAD_ID).then_some(idx))
                    .collect();
                if pad_positions.is_empty() {
                    None
                } else {
                    let count = pad_positions.len();
                    chunk_img_owned =
                        img_emb.slice_axis(0, img_cursor as i64, (img_cursor + count) as i64)?;
                    img_cursor += count;
                    Some(ChunkImageSplice {
                        embeddings: &chunk_img_owned,
                        pad_positions: &pad_positions,
                    })
                }
            } else {
                None
            };

            let lm_mode = if is_last {
                CudaLmHeadMode::LogitsF32Last
            } else {
                CudaLmHeadMode::Skip
            };

            let tap_range = if want_taps {
                chunk_tap_slice_range(offset, end, total_len, tap_tail_limit)
            } else {
                None
            };
            let chunk_tap_visits = tap_range.as_ref().and(tap_visits);

            let out = self.forward_cuda_ids_inner(
                &ids_arr,
                splice,
                cache,
                chunk_tap_visits,
                None,
                false,
                lm_mode,
            )?;

            if let Some(ref range) = tap_range {
                if tap_chunks.is_empty() {
                    tap_chunks.resize_with(out.taps.len(), Vec::new);
                }
                for (dst, tap) in tap_chunks.iter_mut().zip(out.taps) {
                    let kept = if range.start > 0 {
                        tap.slice_axis(0, range.start as i64, range.end as i64)?
                    } else {
                        tap
                    };
                    dst.push(kept);
                }
            }

            if !is_last {
                cache.eval_state()?;
                if tap_range.is_some() {
                    let tap_refs: Vec<&MxArray> =
                        tap_chunks.iter().filter_map(|v| v.last()).collect();
                    if !tap_refs.is_empty() {
                        MxArray::eval_arrays_with_context(
                            &tap_refs,
                            "inkling2 cuda prefill chunk taps",
                        )?;
                    }
                }
            } else {
                last_logits = out.logits_or_tokens;
            }

            offset = end;
        }

        if self.is_turn_cancelled() {
            return Err(Error::from_reason("prefill cancelled"));
        }

        let logits = last_logits.ok_or_else(|| {
            Error::from_reason("Inkling2Inner::prefill_cuda_tokens: missing final chunk logits")
        })?;

        let mut merged_taps = Vec::with_capacity(tap_chunks.len());
        for mut parts in tap_chunks {
            if parts.len() == 1 {
                if let Some(only) = parts.pop() {
                    merged_taps.push(only);
                }
            } else if !parts.is_empty() {
                let refs: Vec<&MxArray> = parts.iter().collect();
                merged_taps.push(MxArray::concatenate_many(refs, Some(0))?);
            }
        }

        // Materialize the final prefill chunk's logits, KV/conv cache updates, and taps BEFORE
        // `_graphs_guard` drops so variable-length prefill never triggers CUDA Graph capture.
        let mut to_eval: Vec<&MxArray> =
            Vec::with_capacity(1 + merged_taps.len() + cache.visits.len() * 8);
        to_eval.push(&logits);
        for tap in &merged_taps {
            to_eval.push(tap);
        }
        cache.collect_pending_state_arrays(&mut to_eval);
        MxArray::eval_arrays_with_context(&to_eval, "inkling2 cuda prefill final chunk")?;

        if self.is_turn_cancelled() {
            return Err(Error::from_reason("prefill cancelled"));
        }

        Ok((logits, merged_taps))
    }

    /// Flat non-speculative CUDA prefill with fail-closed cache ownership:
    /// if prefill is cancelled or fails mid-way, the partially-advanced `cuda_cache` is dropped
    /// (leaving `self.cuda_cache == None`) instead of being put back.
    pub(crate) fn prefill_cuda_flat(&mut self, prompt_tokens: &[u32]) -> Result<MxArray> {
        if self.is_turn_cancelled() {
            self.cuda_cache = None;
            return Err(Error::from_reason("prefill cancelled"));
        }
        let mut cache = match self.cuda_cache.take() {
            Some(c) => c,
            None => self.new_cuda_cache()?,
        };
        let (logits, _) = self.prefill_cuda_tokens(prompt_tokens, None, &mut cache, None, None)?;
        self.cuda_cache = Some(cache);
        Ok(logits)
    }
}

/// Compute the local row range `[local_start..chunk_len]` of a prefill chunk `[offset..end]`
/// that falls within the trailing `tap_tail_limit` rows of a `total_len`-token prefill.
///
/// Returns `None` when the chunk lies completely before the retained tail so tap capture
/// can be skipped altogether for earlier chunks.
#[inline]
fn chunk_tap_slice_range(
    offset: usize,
    end: usize,
    total_len: usize,
    tap_tail_limit: Option<usize>,
) -> Option<std::ops::Range<usize>> {
    let tap_start = match tap_tail_limit {
        Some(limit) => total_len.saturating_sub(limit),
        None => 0,
    };
    if end <= tap_start {
        return None;
    }
    let local_start = tap_start.saturating_sub(offset);
    let chunk_len = end - offset;
    (local_start < chunk_len).then_some(local_start..chunk_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;

    #[test]
    fn greedy_argmax_handles_t16_and_t17_without_truncation() {
        assert!(!lm_head_last_row_only(CudaLmHeadMode::GreedyArgmax, 16));
        assert!(!lm_head_last_row_only(CudaLmHeadMode::GreedyArgmax, 17));
        assert!(!lm_head_last_row_only(CudaLmHeadMode::LogitsF32, 16));
        assert!(lm_head_last_row_only(CudaLmHeadMode::LogitsF32, 17));
        assert!(lm_head_last_row_only(CudaLmHeadMode::LogitsF32Last, 1));
        assert!(lm_head_last_row_only(CudaLmHeadMode::LogitsF32Last, 17));
        assert!(!lm_head_last_row_only(CudaLmHeadMode::LogitsF32All, 17));

        let d = 64usize;
        let v_pad = 128usize;
        let v_unpadded = 96usize;
        let softcap = 30.0f64;

        // Build deterministic BF16 LM head weights [v_pad, d].
        // Give padded columns [v_unpadded..v_pad] huge positive values to ensure slicing works.
        let mut w_f32 = vec![0.0f32; v_pad * d];
        for r in 0..v_pad {
            for c in 0..d {
                w_f32[r * d + c] = if r >= v_unpadded {
                    8.0
                } else {
                    (((r * 37 + c * 13) % 29) as f32 - 14.0) * 0.05
                };
            }
        }
        let w_bf16 = MxArray::from_float32(&w_f32, &[v_pad as i64, d as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let lm_head = LinearProj::Standard(Linear::from_weights(&w_bf16, None).unwrap());

        for t in [16usize, 17usize] {
            let mut x_f32 = vec![0.0f32; t * d];
            for row in 0..t {
                for c in 0..d {
                    x_f32[row * d + c] = (((row * 19 + c * 7 + 3) % 23) as f32 - 11.0) * 0.1 + 0.25;
                }
            }
            let x_bf16 = MxArray::from_float32(&x_f32, &[t as i64, d as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();

            let argmax_arr =
                run_cuda_lm_head(&lm_head, &x_bf16, v_unpadded, softcap, true).unwrap();
            let logits_arr =
                run_cuda_lm_head(&lm_head, &x_bf16, v_unpadded, softcap, false).unwrap();

            assert_eq!(argmax_arr.shape().unwrap().as_ref(), [t as i64]);
            assert_eq!(argmax_arr.dtype().unwrap(), DType::Int32);
            assert_eq!(
                logits_arr.shape().unwrap().as_ref(),
                [t as i64, v_unpadded as i64]
            );

            let got = argmax_arr.to_int32().unwrap();
            let expected = logits_arr
                .argmax(-1, Some(false))
                .unwrap()
                .astype(DType::Int32)
                .unwrap()
                .to_int32()
                .unwrap();
            assert_eq!(got.as_ref(), expected.as_ref());
            assert!(
                got.iter()
                    .all(|&tok| tok >= 0 && (tok as usize) < v_unpadded)
            );
        }
    }

    #[test]
    fn cuda_fast_config_supported_guards_hardcoded_constants_and_kernel_bounds() {
        let base_json = serde_json::json!({
            "model_type": "inkling2",
            "hidden_size": 2304,
            "num_hidden_layers": 48,
            "num_physical_layers": 24,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 201024,
            "unpadded_vocab_size": 200058,
            "sliding_window": 1024,
            "n_layers_per_attn_res_block": 6,
            "loop_config": {
                "core_start": 6,
                "core_end": 18,
                "num_passes": 3,
                "gate_rank": 64,
                "is_varloop": false
            }
        })
        .to_string();
        let base = Inkling2Config::from_json_str(&base_json, true).unwrap();
        assert!(cuda_fast_config_supported(&base));

        // GGUF float32 round-trip of `1e-6` must still pass within tolerance.
        let mut f32_eps = base.clone();
        f32_eps.rms_norm_eps = f64::from(1e-6_f32);
        assert!(cuda_fast_config_supported(&f32_eps));

        // Dynamic kernel arguments may vary freely without disabling the CUDA fast path.
        let mut dynamic_ok = base.clone();
        dynamic_ok.route_scale = 4.0;
        dynamic_ok.final_logit_softcapping = 50.0;
        dynamic_ok.sliding_window = 512;
        dynamic_ok.rel_extent = 512;
        assert!(cuda_fast_config_supported(&dynamic_ok));

        // Every hardcoded numerical constant in CUDA kernels must gate cleanly.
        let mut bad_eps = base.clone();
        bad_eps.rms_norm_eps = 1e-5;
        assert!(!cuda_fast_config_supported(&bad_eps));

        let mut nan_eps = base.clone();
        nan_eps.rms_norm_eps = f64::NAN;
        assert!(!cuda_fast_config_supported(&nan_eps));

        let mut bad_qk_clip = base.clone();
        bad_qk_clip.qk_l2_clip_threshold = 20.0;
        assert!(!cuda_fast_config_supported(&bad_qk_clip));

        let mut bad_rel_clip = base.clone();
        bad_rel_clip.rel_l2_clip_threshold = 12.0;
        assert!(!cuda_fast_config_supported(&bad_rel_clip));

        let mut bad_gate_cap = base.clone();
        bad_gate_cap.mlp_gate_softcap = 6.0;
        assert!(!cuda_fast_config_supported(&bad_gate_cap));

        let mut bad_up_cap = base.clone();
        bad_up_cap.mlp_up_softcap = 30.0;
        assert!(!cuda_fast_config_supported(&bad_up_cap));

        // Block AttnRes kernel limit: `num_physical_layers.div_ceil(block_size) + 1 <= 8`.
        // 24 / 4 + 1 = 7 <= 8 (supported); 24 / 3 + 1 = 9 > 8 (rejected).
        let mut ok_blocks = base.clone();
        ok_blocks.n_layers_per_attn_res_block = 4;
        assert!(cuda_fast_config_supported(&ok_blocks));

        let mut too_many_blocks = base.clone();
        too_many_blocks.n_layers_per_attn_res_block = 3;
        assert!(!cuda_fast_config_supported(&too_many_blocks));

        let mut zero_block = base.clone();
        zero_block.n_layers_per_attn_res_block = 0;
        assert!(!cuda_fast_config_supported(&zero_block));

        // Local SWA layers require `sliding_window > 0`.
        let mut zero_window = base.clone();
        zero_window.sliding_window = 0;
        assert!(!cuda_fast_config_supported(&zero_window));
    }

    #[test]
    fn chunk_tap_slice_range_bounds_dflash2_tail_and_preserves_full_dspark() {
        // 10,000-token prompt chunked by 1024 with DFlash2 ring_rows = 2047 (tap_start = 7953).
        let total_len = 10_000usize;
        let ring_rows = 2047usize;
        let mut retained = Vec::new();
        let mut offset = 0usize;
        let mut skipped_chunks = 0usize;
        while offset < total_len {
            let end = (offset + CUDA_PREFILL_CHUNK).min(total_len);
            match chunk_tap_slice_range(offset, end, total_len, Some(ring_rows)) {
                None => skipped_chunks += 1,
                Some(r) => {
                    for local in r {
                        retained.push(offset + local);
                    }
                }
            }
            offset = end;
        }
        // Chunks 0..7 ([0..7168]) lie completely before tap_start=7953 -> skipped.
        assert_eq!(skipped_chunks, 7);
        assert_eq!(retained.len(), ring_rows);
        assert_eq!(retained.first().copied(), Some(total_len - ring_rows));
        assert_eq!(retained.last().copied(), Some(total_len - 1));

        // DSpark (`None`) keeps every row across all chunks.
        let mut full_rows = 0usize;
        let mut off = 0usize;
        while off < total_len {
            let end = (off + CUDA_PREFILL_CHUNK).min(total_len);
            let r = chunk_tap_slice_range(off, end, total_len, None)
                .expect("unbounded capture must keep every chunk");
            assert_eq!(r, 0..(end - off));
            full_rows += r.len();
            off = end;
        }
        assert_eq!(full_rows, total_len);

        // Short prompt (< ring_rows) keeps all rows; zero limit skips all chunks.
        assert_eq!(
            chunk_tap_slice_range(0, 500, 500, Some(ring_rows)),
            Some(0..500)
        );
        assert_eq!(chunk_tap_slice_range(0, 500, 500, Some(0)), None);
    }

    #[test]
    fn try_embed_rmsnorm_cuda_declines_affine_without_biases() {
        let vocab = 16u32;
        let d = 64u32;
        let mut emb = Embedding::new_uninitialized(vocab, d).unwrap();
        let w = MxArray::zeros(&[i64::from(vocab), i64::from(d / 4)], Some(DType::Uint32)).unwrap();
        let s =
            MxArray::ones(&[i64::from(vocab), i64::from(d / 32)], Some(DType::Float16)).unwrap();
        emb.load_quantized_packed(&w, &s, None, 32, 8, DEFAULT_QUANT_MODE)
            .unwrap();
        let ids = MxArray::from_int32(&[0, 1], &[2]).unwrap();
        let norm_w = MxArray::ones(&[i64::from(d)], Some(DType::Float32)).unwrap();
        // Missing affine biases must NEVER be silently decoded as symmetric Q8 (`q - 128`).
        assert!(
            try_embed_rmsnorm_cuda(&emb, &ids, &norm_w, 1e-6)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn run_cuda_lm_head_m17_q8_preserves_f32_logit_precision() {
        use crate::models::quantized_linear::QuantizedLinear;

        // Construct an asymmetric Q8_0 (bits=8, group_size=32) head with FP32 sidecars
        // where row 1 exceeds row 0 by 0.02 at magnitude ~16.0 (BF16 ULP at 16.0 is 0.125,
        // so routing M=17 through BF16 `forward` would collapse row 0 and row 1 to a tie!).
        let m = 17usize;
        let d = 64usize;
        let v = 8usize;
        let g = d / 32;
        let w = MxArray::zeros(&[v as i64, (d / 4) as i64], Some(DType::Uint32)).unwrap();
        let s = MxArray::ones(&[v as i64, g as i64], Some(DType::Float32)).unwrap();
        let mut biases_f32 = vec![0.0f32; v * g];
        // x is all 1.0 (sum = 64), so with q == 0, logit[row] = sum_g (32 * bias[row, g]).
        // Row 0: 64 * 0.25 = 16.000; Row 1: 32 * 0.25 + 32 * 0.250625 = 16.020 -> winner is 1.
        biases_f32[0] = 0.25;
        biases_f32[1] = 0.25;
        biases_f32[g] = 0.25;
        biases_f32[g + 1] = 0.250_625;
        let b = MxArray::from_float32(&biases_f32, &[v as i64, g as i64]).unwrap();
        let ql = QuantizedLinear::new(w, s, Some(b), None, 32, 8, DEFAULT_QUANT_MODE.to_string());
        let lm_head = LinearProj::Quantized(ql);
        let x = MxArray::ones(&[m as i64, d as i64], Some(DType::BFloat16)).unwrap();

        let logits = run_cuda_lm_head(&lm_head, &x, v, 0.0, false).unwrap();
        assert_eq!(logits.dtype().unwrap(), DType::Float32);
        let vals = logits.to_float32().unwrap();
        for row in 0..m {
            let l0 = vals[row * v];
            let l1 = vals[row * v + 1];
            assert!(
                (l0 - 16.0).abs() < 1e-4 && (l1 - 16.02).abs() < 1e-4 && l1 > l0,
                "row {row}: expected l0=16.0, l1=16.02 (FP32 preserved), got l0={l0}, l1={l1}"
            );
        }
        let argmax = run_cuda_lm_head(&lm_head, &x, v, 0.0, true)
            .unwrap()
            .to_int32()
            .unwrap();
        assert_eq!(argmax.as_ref(), vec![1i32; m].as_slice());
    }
}
