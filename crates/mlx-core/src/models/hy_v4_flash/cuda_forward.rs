//! Full Windows/Linux CUDA fast forward pass for Hy4-flash (`hy_v4_flash`):
//!
//! - BF16 hidden states (`X: bf16 [M, HC, D]`) and dense/MoE/MLA activations with
//!   FP32 accumulation inside every CUDA kernel.
//! - FP16 KV and indexer pooled caches held directly in CUDA device VRAM (`device == 0`)
//!   with fixed buffer pointers across decode steps for CUDA Graph capture.
//! - FP32 logits (`[1 or M, vocab]`) and FP32 trunk hidden outputs (`[M, D]`).
//! - Pre-swizzled 128x4 cuBLASLt NVFP4 scales built once at model load so prefill
//!   `CublasQQMM` pays zero per-call weight-scale swizzling.
//! - MTP-3 CUDA fast `commit` and `draft` paths sharing the same GPU-native primitives.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::ffi::CString;
use std::sync::OnceLock;

use napi::{Error, Result};

use super::bank::{CudaExpertLayer, KqBank};
use super::cache::{LayerCache, ROW_DTYPE, row_dtype};
use super::config::{Attn, HyV4FlashConfig};
use super::engram::keeps;
use super::model::{Logits, Model, Output, Session, Want, call_engram_rows, head_rows};
use super::moe::{AffineExperts, ExpertSource, RowKind, qlinear};
use super::mtp::{Draft, MtpLayer};
use super::ops::rms_norm;
use super::rows::PrefetchHandle;
use super::scan::ExpertClass;
use super::stats::{DeferScope, Phase, prof_end, prof_start};
use super::weights::{AttnWeights, Dense, Ffn, LayerWeights, MoeWeights, Residual};
use crate::accelerator::{CudaGraphsGuard, cuda_cache_clear_threshold_bytes};
use crate::array::{DType, MxArray};
use crate::kernels::hy4::{self, HcFnMode, LmHeadWeightMode, MoeWeightMode, Nvfp4PrefillMode};
use crate::kernels::linear;
use crate::models::moe_common::{Weight, inv_freqs};
use crate::nn::Activations;

/// Master switch for the Hy4 CUDA fast forward pass (default enabled on CUDA GPU).
pub const CUDA_FAST_ENV: &str = "MLX_HY4_CUDA_FAST";
/// NVFP4 prefill kernel mode override (`auto`, `bf16`, `fp4`).
pub const NVFP4_PREFILL_MODE_ENV: &str = "MLX_HY4_NVFP4_PREFILL_MODE";

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 CUDA forward: {msg}"))
}

/// Whether the CUDA fast forward pass is enabled for this call.
///
/// Defaults to `true` on CUDA GPUs when `kernels::hy4::available()` is true and
/// flat KV cache storage uses `Float16`. Setting `MLX_HY4_CUDA_FAST=0` forces
/// the reference FP32 eager path for A/B parity testing.
pub fn cuda_fast_enabled() -> bool {
    if !hy4::available() || row_dtype() != ROW_DTYPE {
        return false;
    }
    match std::env::var(CUDA_FAST_ENV) {
        Ok(raw) => {
            let v = raw.trim();
            !(v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no"))
        }
        Err(_) => true,
    }
}

/// Parses `MLX_HY4_NVFP4_PREFILL_MODE` (`"auto"` -> `Auto`, `"bf16"` -> `Bf16Tc`, `"fp4"` -> `CublasFp4`).
pub fn nvfp4_prefill_mode() -> Nvfp4PrefillMode {
    match std::env::var(NVFP4_PREFILL_MODE_ENV) {
        Ok(raw) => {
            let v = raw.trim();
            if v.eq_ignore_ascii_case("bf16")
                || v.eq_ignore_ascii_case("bf16tc")
                || v.eq_ignore_ascii_case("bf16_tc")
            {
                Nvfp4PrefillMode::Bf16Tc
            } else if v.eq_ignore_ascii_case("fp4")
                || v.eq_ignore_ascii_case("cublas")
                || v.eq_ignore_ascii_case("cublasfp4")
                || v.eq_ignore_ascii_case("cublas_fp4")
            {
                Nvfp4PrefillMode::CublasFp4
            } else {
                Nvfp4PrefillMode::Auto
            }
        }
        Err(_) => Nvfp4PrefillMode::Auto,
    }
}

/// Cloned lightweight handles to resident 8-bit g32 routed experts (MTP or unit-test trunk).
pub(crate) struct CudaSymQ8Experts {
    pub gate_w: MxArray,
    pub gate_s: MxArray,
    pub up_w: MxArray,
    pub up_s: MxArray,
    pub down_w: MxArray,
    pub down_s: MxArray,
}

impl CudaSymQ8Experts {
    pub(crate) fn try_from_affine(experts: &AffineExperts) -> Result<Option<Self>> {
        if experts.bits() != 8 || experts.group_size() != 32 {
            return Ok(None);
        }
        let projs = experts.projs();
        for p in projs {
            let dt = p.s.dtype()?;
            if !matches!(dt, DType::Float16 | DType::BFloat16) {
                return Ok(None);
            }
        }
        let mut gate_w = projs[0].w.clone();
        let mut gate_s = projs[0].s.clone();
        let mut up_w = projs[1].w.clone();
        let mut up_s = projs[1].s.clone();
        let mut down_w = projs[2].w.clone();
        let mut down_s = projs[2].s.clone();
        gate_w.promote_cuda_device()?;
        gate_s.promote_cuda_device()?;
        up_w.promote_cuda_device()?;
        up_s.promote_cuda_device()?;
        down_w.promote_cuda_device()?;
        down_s.promote_cuda_device()?;
        Ok(Some(Self {
            gate_w,
            gate_s,
            up_w,
            up_s,
            down_w,
            down_s,
        }))
    }
}

struct LayerCudaCache {
    engram_kv: Option<MxArray>,
    /// Cached GPU BF16 HC projection weights when resident `hc_*_fn.weight` is not already BF16.
    hc_attn_w: Option<MxArray>,
    hc_ffn_w: Option<MxArray>,
    attn_q_kv_gate: Option<MxArray>,
    attn_q_kv: Option<MxArray>,
    attn_q_a: Option<MxArray>,
    attn_kv_a: Option<MxArray>,
    attn_gate: Option<MxArray>,
    attn_q_b: Option<MxArray>,
    attn_output: Option<MxArray>,
    ffn_gate_up: Option<MxArray>,
    ffn_gate: Option<MxArray>,
    ffn_up: Option<MxArray>,
    ffn_down: Option<MxArray>,
    moe_down_shared_gu: Option<MxArray>,
    moe_routed_down: Option<MxArray>,
    moe_shared_gate: Option<MxArray>,
    moe_shared_up: Option<MxArray>,
    moe_routed_up: Option<MxArray>,
    moe_shared_down: Option<MxArray>,
    /// Cached GPU BF16 (or F32 when `MLX_HY4_ROUTER_F32=1`) router weight when `MoeWeights.router`
    /// is not already stored in that target dtype.
    router_w: Option<MxArray>,
    /// Cached GPU BF16 copy of `indexer.proj` when resident `iw.proj` is F32
    /// (eliminates `CublasGemm` `CaptureContext` on all 12 indexed layers).
    indexer_proj_w: Option<MxArray>,
    /// Optional custom per-layer inverse frequencies if a layer's `rope_theta` differs from global.
    custom_inv_freq: Option<MxArray>,
    /// Optional resident SymQ8 experts (for unit tests without a `CudaDirect` `KqBank`).
    resident_experts: Option<CudaSymQ8Experts>,
}

impl LayerCudaCache {
    fn push_pending<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        let fields = [
            &self.engram_kv,
            &self.hc_attn_w,
            &self.hc_ffn_w,
            &self.attn_q_kv_gate,
            &self.attn_q_kv,
            &self.attn_q_a,
            &self.attn_kv_a,
            &self.attn_gate,
            &self.attn_q_b,
            &self.attn_output,
            &self.ffn_gate_up,
            &self.ffn_gate,
            &self.ffn_up,
            &self.ffn_down,
            &self.moe_down_shared_gu,
            &self.moe_routed_down,
            &self.moe_shared_gate,
            &self.moe_shared_up,
            &self.moe_routed_up,
            &self.moe_shared_down,
            &self.router_w,
            &self.indexer_proj_w,
        ];
        for f in fields {
            if let Some(a) = f.as_ref() {
                out.push(a);
            }
        }
    }

    fn promote_all(&mut self) -> Result<()> {
        let fields = [
            &mut self.engram_kv,
            &mut self.hc_attn_w,
            &mut self.hc_ffn_w,
            &mut self.attn_q_kv_gate,
            &mut self.attn_q_kv,
            &mut self.attn_q_a,
            &mut self.attn_kv_a,
            &mut self.attn_gate,
            &mut self.attn_q_b,
            &mut self.attn_output,
            &mut self.ffn_gate_up,
            &mut self.ffn_gate,
            &mut self.ffn_up,
            &mut self.ffn_down,
            &mut self.moe_down_shared_gu,
            &mut self.moe_routed_down,
            &mut self.moe_shared_gate,
            &mut self.moe_shared_up,
            &mut self.moe_routed_up,
            &mut self.moe_shared_down,
            &mut self.router_w,
            &mut self.indexer_proj_w,
        ];
        for f in fields {
            if let Some(a) = f.as_mut() {
                a.promote_cuda_device()?;
            }
        }
        Ok(())
    }
}

/// Precomputed CUDA metadata and pre-swizzled NVFP4 scale tables held once per `Model`.
pub(crate) struct CudaForwardCache {
    inv_freq_swa: MxArray,
    inv_freq_idx: MxArray,
    /// Lazily cached GPU BF16 copy of `output_hc_fn.weight` when resident weight is not BF16.
    output_hc_w: OnceLock<MxArray>,
    layers: Vec<LayerCudaCache>,
    /// Holds MLX CUDA Graphs disabled for the lifetime of `Model` on Windows WDDM
    /// and CC 12.1 Spark hosts unless `MLX_USE_CUDA_GRAPHS` is explicitly set.
    _graphs_guard: Option<CudaGraphsGuard>,
}

/// Disables MLX CUDA Graph capture for `hy_v4_flash` on Windows (WDDM graph
/// build/update/destroy adds +12–52 ms/cycle on a 42-layer Hy4 trunk) and on
/// CC 12.1 (RTX/DGX Spark, where MLX's `device.cpp` defaults to `ops=100, mb=25`
/// and shatters every forward into ~15–20 graphs), unless `MLX_USE_CUDA_GRAPHS`
/// is explicitly set in the environment.
pub(crate) fn disable_cuda_graphs_for_hy4_default() -> Option<CudaGraphsGuard> {
    if !crate::accelerator::cuda_ops_available()
        || std::env::var_os("MLX_USE_CUDA_GRAPHS").is_some()
    {
        return None;
    }
    let is_spark_cc121 = crate::accelerator::caps().cuda_sm == Some((12, 1));
    if cfg!(target_os = "windows") || is_spark_cc121 {
        Some(CudaGraphsGuard::disable_for_prefill())
    } else {
        None
    }
}

/// Precomputed CUDA metadata for `MtpLayer`.
pub(crate) struct MtpCudaCache {
    pub(crate) inv_freq: MxArray,
    pub(crate) router_w: Option<MxArray>,
    pub(crate) indexer_proj_w: Option<MxArray>,
    pub(crate) sym_q8_experts: Option<CudaSymQ8Experts>,
}

fn build_gpu_inv_freq(rope_dim: usize, theta: f32) -> Result<MxArray> {
    let vals = inv_freqs(rope_dim, theta, None);
    let mut arr = MxArray::from_float32(&vals, &[vals.len() as i64])?;
    arr.promote_cuda_device()?;
    Ok(arr)
}

/// Lazily builds pre-swizzled 2D `Dense::Nvfp4` scales (`N >= 128 && N % 128 == 0 && K % 64 == 0`)
/// into cuBLASLt 128x4 tiled layout; caller batches one `eval` + `promote_cuda_device`.
fn maybe_swizzle_nvfp4_lazy(dense: &Dense) -> Result<Option<MxArray>> {
    let Some((_, scales)) = dense.nvfp4_parts() else {
        return Ok(None);
    };
    if scales.ndim()? != 2 || scales.dtype()? != DType::Uint8 {
        return Ok(None);
    }
    let n = scales.shape_at(0)?;
    let k16 = scales.shape_at(1)?;
    let k = k16 * 16;
    if n < 128 || n % 128 != 0 || k < 64 || k % 64 != 0 {
        return Ok(None);
    }
    hy4::nvfp4_swizzle_scales(scales)
}

fn router_f32_forced() -> bool {
    std::env::var("MLX_HY4_ROUTER_F32").as_deref() == Ok("1")
}

/// Lazily builds a GPU-resident BF16 copy of `hc_*_fn.weight` when the resident
/// weight is not already BF16.
fn cache_hc_bf16_weight_lazy(w: &MxArray) -> Result<Option<MxArray>> {
    if w.dtype()? == DType::BFloat16 {
        return Ok(None);
    }
    Ok(Some(w.astype(DType::BFloat16)?))
}

/// Caches a GPU-resident BF16 copy of `hc_*_fn.weight` when the resident weight
/// is not already BF16 (used by lazy `output_hc_bf16`).
fn cache_hc_bf16_weight(w: &MxArray) -> Result<Option<MxArray>> {
    let Some(mut bf16) = cache_hc_bf16_weight_lazy(w)? else {
        return Ok(None);
    };
    MxArray::eval_arrays_with_context(&[&bf16], "hy_v4_flash cuda hc bf16 weight")?;
    bf16.promote_cuda_device()?;
    Ok(Some(bf16))
}

/// Lazily builds a GPU-resident BF16 (or F32 when `MLX_HY4_ROUTER_F32=1`) router weight.
fn cache_router_weight_lazy(router: &Dense) -> Result<Option<MxArray>> {
    if router_f32_forced() {
        if let Dense::Array(a) = router
            && a.dtype()? == DType::Float32
        {
            return Ok(None);
        }
        return Ok(Some(router.f32()?));
    }
    if let Dense::Array(a) = router
        && a.dtype()? == DType::BFloat16
    {
        return Ok(None);
    }
    let w = match router {
        Dense::Array(a) => a.astype(DType::BFloat16)?,
        _ => router.f32()?.astype(DType::BFloat16)?,
    };
    Ok(Some(w))
}

/// Lazily builds a GPU-resident BF16 copy of `indexer.proj` when stored as F32,
/// avoiding `CublasGemm` `CaptureContext` on indexed layers & MTP.
fn cache_indexer_proj_weight_lazy(proj: &Dense) -> Result<Option<MxArray>> {
    let Dense::Array(a) = proj else {
        return Ok(None);
    };
    if a.dtype()? == DType::BFloat16 {
        return Ok(None);
    }
    Ok(Some(a.astype(DType::BFloat16)?))
}

/// Maximum routed experts per token supported by the Hy4 CUDA router/MoE kernels
/// (`kMaxTopK` in `crates/mlx-sys/src/cuda/hy4/moe.cpp`).
const CUDA_MOE_MAX_TOP_K: usize = 16;

fn moe_dims_fit_cuda(cfg: &HyV4FlashConfig) -> bool {
    cfg.moe.latent >= 64
        && cfg.moe.latent.is_multiple_of(64)
        && cfg.moe.ff >= 64
        && cfg.moe.ff.is_multiple_of(64)
        && cfg.moe.n_experts > 0
        && cfg.moe.n_experts <= 512
        && cfg.moe.n_used > 0
        && cfg.moe.n_used <= CUDA_MOE_MAX_TOP_K
        && cfg.moe.n_used <= cfg.moe.n_experts
}

impl CudaForwardCache {
    /// Builds `CudaForwardCache` for `Model::new` when running on CUDA and geometry
    /// satisfies all CUDA kernel contracts. Returns `Ok(None)` off CUDA or for
    /// non-CUDA-direct / toy-geometry models so they fall back cleanly to the eager path.
    pub(crate) fn build(
        cfg: &HyV4FlashConfig,
        layer_weights: &[LayerWeights],
        expert_sources: &[Option<&ExpertSource>],
    ) -> Result<Option<Self>> {
        if !hy4::available() {
            return Ok(None);
        }
        if cfg.hc.count == 0
            || cfg.hc.count > 8
            || cfg.dim < 32
            || !cfg.dim.is_multiple_of(8)
            || cfg.mla.kv_lora_rank == 0
            || cfg.mla.rope_dim < 2
            || !cfg.mla.rope_dim.is_multiple_of(2)
        {
            return Ok(None);
        }
        let has_moe = cfg.trunk().iter().any(|l| !l.dense_ffn);
        if has_moe && !moe_dims_fit_cuda(cfg) {
            return Ok(None);
        }

        let _graphs_guard = disable_cuda_graphs_for_hy4_default();
        let inv_freq_swa = build_gpu_inv_freq(cfg.mla.rope_dim, cfg.rope_theta_swa)?;
        let inv_freq_idx = build_gpu_inv_freq(cfg.mla.rope_dim, cfg.rope_theta)?;
        let mut layers = Vec::with_capacity(layer_weights.len());

        for (il, (lw, lc)) in layer_weights.iter().zip(cfg.trunk().iter()).enumerate() {
            let resident_experts = if lc.dense_ffn {
                None
            } else {
                match expert_sources.get(il).and_then(|s| *s) {
                    Some(ExpertSource::Bank(b)) if b.is_cuda_direct() => None,
                    Some(ExpertSource::Resident(ae)) => {
                        match CudaSymQ8Experts::try_from_affine(ae)? {
                            Some(re) => Some(re),
                            None => return Ok(None),
                        }
                    }
                    _ => return Ok(None),
                }
            };

            let expected_theta = match lc.attn {
                Attn::Sliding => cfg.rope_theta_swa,
                Attn::Indexed => cfg.rope_theta,
            };
            let custom_inv_freq = if (lc.rope_theta - expected_theta).abs() > 1e-3 {
                Some(build_gpu_inv_freq(cfg.mla.rope_dim, lc.rope_theta)?)
            } else {
                None
            };

            let engram_kv = match &lw.engram {
                Some(ew) => maybe_swizzle_nvfp4_lazy(&ew.kv)?,
                None => None,
            };
            let (hc_attn_w, hc_ffn_w) = match &lw.residual {
                Residual::Hc { attn, ffn } => (
                    cache_hc_bf16_weight_lazy(&attn.weight)?,
                    cache_hc_bf16_weight_lazy(&ffn.weight)?,
                ),
                Residual::Mtp(_) => (None, None),
            };
            let attn_q_kv_gate = match &lw.attn.q_kv_gate {
                Some(d) => maybe_swizzle_nvfp4_lazy(d)?,
                None => None,
            };
            let attn_q_kv = match &lw.attn.q_kv {
                Some(d) => maybe_swizzle_nvfp4_lazy(d)?,
                None => None,
            };
            let attn_q_a = if lw.attn.q_kv_gate.is_none() && lw.attn.q_kv.is_none() {
                maybe_swizzle_nvfp4_lazy(&lw.attn.q_a)?
            } else {
                None
            };
            let attn_kv_a = if lw.attn.q_kv_gate.is_none() && lw.attn.q_kv.is_none() {
                maybe_swizzle_nvfp4_lazy(&lw.attn.kv_a)?
            } else {
                None
            };
            let attn_gate = if lw.attn.q_kv_gate.is_none() {
                maybe_swizzle_nvfp4_lazy(&lw.attn.gate)?
            } else {
                None
            };
            let attn_q_b = maybe_swizzle_nvfp4_lazy(&lw.attn.q_b)?;
            let attn_output = maybe_swizzle_nvfp4_lazy(&lw.attn.output)?;
            let indexer_proj_w = match &lw.indexer {
                Some(iw) => cache_indexer_proj_weight_lazy(&iw.proj)?,
                None => None,
            };

            let (
                ffn_gate_up,
                ffn_gate,
                ffn_up,
                ffn_down,
                moe_down_shared_gu,
                moe_routed_down,
                moe_shared_gate,
                moe_shared_up,
                moe_routed_up,
                moe_shared_down,
                router_w,
            ) = match &lw.ffn {
                Ffn::Mlp {
                    gate,
                    up,
                    down,
                    gate_up,
                } => {
                    let gu = match gate_up {
                        Some(d) => maybe_swizzle_nvfp4_lazy(d)?,
                        None => None,
                    };
                    let g = if gate_up.is_none() {
                        maybe_swizzle_nvfp4_lazy(gate)?
                    } else {
                        None
                    };
                    let u = if gate_up.is_none() {
                        maybe_swizzle_nvfp4_lazy(up)?
                    } else {
                        None
                    };
                    let d = maybe_swizzle_nvfp4_lazy(down)?;
                    (gu, g, u, d, None, None, None, None, None, None, None)
                }
                Ffn::Moe(mw) => {
                    let dgu = match &mw.down_shared_gu {
                        Some(d) => maybe_swizzle_nvfp4_lazy(d)?,
                        None => None,
                    };
                    let rd = if mw.down_shared_gu.is_none() {
                        maybe_swizzle_nvfp4_lazy(&mw.routed_down)?
                    } else {
                        None
                    };
                    let sg = if mw.down_shared_gu.is_none() {
                        maybe_swizzle_nvfp4_lazy(&mw.shared_gate)?
                    } else {
                        None
                    };
                    let su = if mw.down_shared_gu.is_none() {
                        maybe_swizzle_nvfp4_lazy(&mw.shared_up)?
                    } else {
                        None
                    };
                    let ru = maybe_swizzle_nvfp4_lazy(&mw.routed_up)?;
                    let sd = maybe_swizzle_nvfp4_lazy(&mw.shared_down)?;
                    let rw = cache_router_weight_lazy(&mw.router)?;
                    (None, None, None, None, dgu, rd, sg, su, ru, sd, rw)
                }
            };

            layers.push(LayerCudaCache {
                engram_kv,
                hc_attn_w,
                hc_ffn_w,
                attn_q_kv_gate,
                attn_q_kv,
                attn_q_a,
                attn_kv_a,
                attn_gate,
                attn_q_b,
                attn_output,
                ffn_gate_up,
                ffn_gate,
                ffn_up,
                ffn_down,
                moe_down_shared_gu,
                moe_routed_down,
                moe_shared_gate,
                moe_shared_up,
                moe_routed_up,
                moe_shared_down,
                router_w,
                indexer_proj_w,
                custom_inv_freq,
                resident_experts,
            });
        }

        // Evaluate all pre-swizzled NVFP4 scales and cached BF16 weights across
        // every layer in ONE batched GPU pass instead of ~450 per-tensor syncs.
        {
            let mut pending: Vec<&MxArray> = Vec::with_capacity(layers.len() * 12);
            for l in &layers {
                l.push_pending(&mut pending);
            }
            if !pending.is_empty() {
                MxArray::eval_arrays_with_context(
                    &pending,
                    "hy_v4_flash cuda forward cache build",
                )?;
            }
        }
        for l in &mut layers {
            l.promote_all()?;
        }

        Ok(Some(Self {
            inv_freq_swa,
            inv_freq_idx,
            output_hc_w: OnceLock::new(),
            layers,
            _graphs_guard,
        }))
    }

    fn inv_freq_for(&self, il: usize, attn: Attn) -> &MxArray {
        if let Some(custom) = self.layers[il].custom_inv_freq.as_ref() {
            return custom;
        }
        match attn {
            Attn::Sliding => &self.inv_freq_swa,
            Attn::Indexed => &self.inv_freq_idx,
        }
    }

    fn output_hc_bf16(&self, w_fn: &MxArray) -> Result<(MxArray, HcFnMode)> {
        if w_fn.dtype()? == DType::BFloat16 {
            return Ok((w_fn.clone(), HcFnMode::Bf16));
        }
        if let Some(cached) = self.output_hc_w.get() {
            return Ok((cached.clone(), HcFnMode::Bf16));
        }
        let _graphs_guard = CudaGraphsGuard::disable_for_prefill();
        let Some(bf16) = cache_hc_bf16_weight(w_fn)? else {
            return Ok((w_fn.clone(), HcFnMode::Bf16));
        };
        let _ = self.output_hc_w.set(bf16.clone());
        let w = self.output_hc_w.get().cloned().unwrap_or(bf16);
        Ok((w, HcFnMode::Bf16))
    }
}

impl MtpCudaCache {
    /// Builds CUDA metadata for `MtpLayer::new_with_head` when on CUDA and geometry fits.
    pub(crate) fn build(
        cfg: &HyV4FlashConfig,
        layer: usize,
        w: &LayerWeights,
        experts: &AffineExperts,
    ) -> Result<Option<Self>> {
        if !hy4::available() {
            return Ok(None);
        }
        if cfg.dim < 32
            || !cfg.dim.is_multiple_of(8)
            || cfg.mla.kv_lora_rank == 0
            || cfg.mla.rope_dim < 2
            || !cfg.mla.rope_dim.is_multiple_of(2)
        {
            return Ok(None);
        }
        if matches!(w.ffn, Ffn::Moe(_)) && !moe_dims_fit_cuda(cfg) {
            return Ok(None);
        }
        let lc = cfg
            .layers
            .get(layer)
            .ok_or_else(|| bad(format!("MTP layer {layer} out of range")))?;
        let inv_freq = build_gpu_inv_freq(cfg.mla.rope_dim, lc.rope_theta)?;
        let mut router_w = match &w.ffn {
            Ffn::Moe(mw) => cache_router_weight_lazy(&mw.router)?,
            Ffn::Mlp { .. } => None,
        };
        let mut indexer_proj_w = match &w.indexer {
            Some(iw) => cache_indexer_proj_weight_lazy(&iw.proj)?,
            None => None,
        };
        {
            let mut pending: Vec<&MxArray> = Vec::with_capacity(2);
            if let Some(a) = router_w.as_ref() {
                pending.push(a);
            }
            if let Some(a) = indexer_proj_w.as_ref() {
                pending.push(a);
            }
            if !pending.is_empty() {
                MxArray::eval_arrays_with_context(&pending, "hy_v4_flash cuda mtp cache build")?;
            }
        }
        if let Some(a) = router_w.as_mut() {
            a.promote_cuda_device()?;
        }
        if let Some(a) = indexer_proj_w.as_mut() {
            a.promote_cuda_device()?;
        }
        let sym_q8_experts = if cfg.moe.latent >= 64
            && cfg.moe.latent.is_multiple_of(64)
            && cfg.moe.ff >= 64
            && cfg.moe.ff.is_multiple_of(64)
            && cfg.moe.n_experts > 0
            && cfg.moe.n_experts <= 512
        {
            CudaSymQ8Experts::try_from_affine(experts)?
        } else {
            None
        };
        Ok(Some(Self {
            inv_freq,
            router_w,
            indexer_proj_w,
            sym_q8_experts,
        }))
    }
}

#[inline]
fn ensure_bf16(x: &MxArray) -> Result<MxArray> {
    if x.dtype()? == DType::BFloat16 {
        Ok(x.clone())
    } else {
        x.astype(DType::BFloat16)
    }
}

fn qmm_nvfp4_bf16_fallback(x_bf16: &MxArray, w: &MxArray, scales: &MxArray) -> Result<MxArray> {
    let mode = CString::new("nvfp4").map_err(|e| bad(e.to_string()))?;
    // SAFETY: live arrays for the call; biases is null for nvfp4; `mlx_dequantize` is AOT-compiled
    // and does not invoke MLX's CUDA NVRTC `cute/` header path.
    let h = unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            std::ptr::null_mut(),
            16,
            4,
            DType::Float32 as i32,
            mode.as_ptr(),
        )
    };
    let w_bf16 = MxArray::from_handle(h, "hy_v4_flash cuda nvfp4 dequant fallback")?
        .astype(DType::BFloat16)?;
    x_bf16.matmul(&w_bf16.transpose(Some(&[1, 0]))?)
}

fn qmm4_bf16_fallback(
    x_bf16: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<MxArray> {
    let dt = scales.dtype()?;
    let xq = if x_bf16.dtype()? == dt {
        x_bf16.clone()
    } else {
        x_bf16.astype(dt)?
    };
    let mode = CString::new("affine").map_err(|e| bad(e.to_string()))?;
    // SAFETY: live arrays for the call.
    let h = unsafe {
        mlx_sys::mlx_quantized_matmul(
            xq.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            true,
            64,
            4,
            mode.as_ptr(),
        )
    };
    let out = MxArray::from_handle(h, "hy_v4_flash cuda qmm4 fallback")?;
    ensure_bf16(&out)
}

/// BF16-in / BF16-out dense linear dispatch for CUDA across NVFP4, SymQ8/Q8, Qmm,
/// Qmm4, and plain arrays.
pub(crate) fn cuda_linear_bf16(
    x: &MxArray,
    dense: &Dense,
    swizzled: Option<&MxArray>,
    prefill_mode: Nvfp4PrefillMode,
) -> Result<MxArray> {
    let x_bf16 = ensure_bf16(x)?;
    let m = x_bf16.shape_at(0)? as usize;
    match dense {
        Dense::Nvfp4 { w, scales } => {
            if m <= 16
                && let Some(out) = hy4::nvfp4_few(&x_bf16, w, scales)?
            {
                return Ok(out);
            }
            if m >= 17
                && let Some(out) = hy4::nvfp4_prefill(&x_bf16, w, scales, swizzled, prefill_mode)?
            {
                return Ok(out);
            }
            qmm_nvfp4_bf16_fallback(&x_bf16, w, scales)
        }
        Dense::SymQ8 { w, scales } | Dense::Q8 { w, scales } => {
            if m <= 16
                && let Some(out) = linear::affine_bf16_matmul_few(&x_bf16, w, scales, None, 32, 8)?
            {
                return Ok(out);
            }
            if m >= 17
                && let Some(out) = linear::q8_prefill_cuda(&x_bf16, w, scales, None, -1, false)?
            {
                return Ok(out);
            }
            if let Some(w_bf16) = linear::dequant_q8_bf16_cuda(w, scales, None)? {
                return x_bf16.matmul(&w_bf16.transpose(Some(&[1, 0]))?);
            }
            qlinear(&x_bf16.astype(DType::Float32)?, dense)?.astype(DType::BFloat16)
        }
        Dense::Qmm { w, scales, biases } => {
            if m <= 16
                && let Some(out) =
                    linear::affine_bf16_matmul_few(&x_bf16, w, scales, Some(biases), 32, 8)?
            {
                return Ok(out);
            }
            if m >= 17
                && let Some(out) =
                    linear::q8_prefill_cuda(&x_bf16, w, scales, Some(biases), -1, true)?
            {
                return Ok(out);
            }
            if let Some(w_bf16) = linear::dequant_q8_bf16_cuda(w, scales, Some(biases))? {
                return x_bf16.matmul(&w_bf16.transpose(Some(&[1, 0]))?);
            }
            qlinear(&x_bf16.astype(DType::Float32)?, dense)?.astype(DType::BFloat16)
        }
        Dense::Qmm4 { w, scales, biases } => {
            if m <= 8
                && let Some(out) = linear::affine_q4_g64_matmul_few(&x_bf16, w, scales, biases)?
            {
                return Ok(out);
            }
            qmm4_bf16_fallback(&x_bf16, w, scales, biases)
        }
        Dense::Array(a) => {
            let dt = a.dtype()?;
            if dt == DType::BFloat16 {
                if m <= 16
                    && let Some(out) = linear::bf16_matmul_few(&x_bf16, a, a.shape_at(0)?)?
                {
                    return Ok(out);
                }
                x_bf16.matmul(&a.transpose(Some(&[1, 0]))?)
            } else {
                let xm = x_bf16.astype(dt)?;
                xm.matmul(&a.transpose(Some(&[1, 0]))?)?
                    .astype(DType::BFloat16)
            }
        }
        Dense::MmapBf16 { .. } | Dense::MmapQ8_0 { .. } => {
            qlinear(&x_bf16.astype(DType::Float32)?, dense)?.astype(DType::BFloat16)
        }
    }
}

/// BF16-in / BF16-out dense linear using a cached GPU-resident BF16 weight matrix.
fn cuda_linear_bf16_cached_array(x: &MxArray, w_bf16: &MxArray) -> Result<MxArray> {
    let x_bf16 = ensure_bf16(x)?;
    let m = x_bf16.shape_at(0)? as usize;
    if m <= 16
        && let Some(out) = linear::bf16_matmul_few(&x_bf16, w_bf16, w_bf16.shape_at(0)?)?
    {
        return Ok(out);
    }
    x_bf16.matmul(&w_bf16.transpose(Some(&[1, 0]))?)
}

/// Fused CUDA LM head returning `f32 [M, V]` logits.
fn cuda_lm_head_f32(h: &MxArray, head: &Dense) -> Result<MxArray> {
    let h_bf16 = ensure_bf16(h)?;
    let m = h_bf16.shape_at(0)? as usize;
    match head {
        Dense::Nvfp4 { w, scales } => {
            if let Some(out) =
                hy4::lm_head_logits(&h_bf16, w, Some(scales), LmHeadWeightMode::Nvfp4)?
            {
                return Ok(out);
            }
        }
        Dense::SymQ8 { w, scales } | Dense::Q8 { w, scales } => {
            if let Some(out) =
                hy4::lm_head_logits(&h_bf16, w, Some(scales), LmHeadWeightMode::SymQ8)?
            {
                return Ok(out);
            }
        }
        Dense::Array(w) if w.dtype()? == DType::BFloat16 => {
            if let Some(out) = hy4::lm_head_logits(&h_bf16, w, None, LmHeadWeightMode::Bf16)? {
                return Ok(out);
            }
        }
        Dense::Qmm4 { w, scales, biases } if m <= 8 => {
            if let Some(out) = linear::affine_q4_g64_matmul_few(&h_bf16, w, scales, biases)? {
                return out.astype(DType::Float32);
            }
        }
        _ => {}
    }
    head_rows(&h_bf16.astype(DType::Float32)?, head)
}

/// Grouped SymQ8 projection for MLA `attn_k_b` / `attn_v_b` with optional fused `* sigmoid(gate)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_grouped_q8(
    x_bf16: &MxArray,
    w: &Dense,
    gate: Option<&MxArray>,
    n_heads: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<MxArray> {
    if let Some((qw, qs)) = w.sym_q8_parts()
        && qs.dtype()? == DType::Float16
        && let Some(out) = hy4::grouped_q8(
            x_bf16,
            qw,
            qs,
            gate,
            n_heads as i32,
            in_dim as i32,
            out_dim as i32,
        )?
    {
        return Ok(out);
    }
    // `grouped_qmm_t` expects head-major `[H, M, in_dim]` and returns `[H, M, out_dim]`.
    let x_hm = x_bf16.astype(DType::Float32)?.transpose(Some(&[1, 0, 2]))?;
    let y_hm = match w.grouped_qmm_t(&x_hm, n_heads as i64)? {
        Some(out) => out,
        None => {
            let wf = w
                .f32()?
                .reshape(&[n_heads as i64, out_dim as i64, in_dim as i64])?;
            x_hm.matmul(&wf.transpose(Some(&[0, 2, 1]))?)?
        }
    };
    let y = y_hm.transpose(Some(&[1, 0, 2]))?.astype(DType::BFloat16)?;
    if let Some(g) = gate {
        let m = x_bf16.shape_at(0)?;
        let g3 = ensure_bf16(g)?.reshape(&[m, n_heads as i64, out_dim as i64])?;
        Activations::sigmoid_mul_compiled(&g3, &y)
    } else {
        Ok(y)
    }
}

fn rms_norm_bf16(x_bf16: &MxArray, weight: &MxArray, eps: f32) -> Result<MxArray> {
    if weight.dtype()? == DType::Float32 {
        // SAFETY: live BF16 input and F32 weight handles; null signals contract miss.
        let h = unsafe {
            mlx_sys::mlx_rmsnorm_precise_f32_cuda(x_bf16.as_raw_ptr(), weight.as_raw_ptr(), eps)
        };
        if !h.is_null() {
            return MxArray::from_handle(h, "hy4_cuda_rmsnorm_precise_f32");
        }
    }
    rms_norm(&x_bf16.astype(DType::Float32)?, weight, eps)?.astype(DType::BFloat16)
}

fn fused_add_rms_norm_bf16(
    x_bf16: &MxArray,
    res_bf16: &MxArray,
    weight: &MxArray,
    eps: f32,
) -> Result<(MxArray, MxArray)> {
    if weight.dtype()? == DType::Float32 {
        let mut h_ptr = std::ptr::null_mut();
        let mut n_ptr = std::ptr::null_mut();
        // SAFETY: live BF16 inputs and F32 weight; returns false on contract miss.
        let ok = unsafe {
            mlx_sys::mlx_fused_add_rmsnorm_precise_f32_cuda(
                x_bf16.as_raw_ptr(),
                res_bf16.as_raw_ptr(),
                weight.as_raw_ptr(),
                eps,
                &mut h_ptr,
                &mut n_ptr,
            )
        };
        if ok && !h_ptr.is_null() && !n_ptr.is_null() {
            return Ok((
                MxArray::from_handle(h_ptr, "hy4_cuda_add_rmsnorm_h")?,
                MxArray::from_handle(n_ptr, "hy4_cuda_add_rmsnorm_norm")?,
            ));
        }
    }
    let h = x_bf16.add(res_bf16)?;
    let n = rms_norm_bf16(&h, weight, eps)?;
    Ok((h, n))
}

fn hc_fn_mode(w_fn: &MxArray, cached_bf16: Option<&MxArray>) -> Result<(MxArray, HcFnMode)> {
    if let Some(bf16) = cached_bf16 {
        return Ok((bf16.clone(), HcFnMode::Bf16));
    }
    match w_fn.dtype()? {
        DType::BFloat16 => Ok((w_fn.clone(), HcFnMode::Bf16)),
        DType::Float32 => Ok((w_fn.clone(), HcFnMode::F32)),
        _ => Ok((w_fn.astype(DType::Float32)?, HcFnMode::F32)),
    }
}

#[allow(clippy::too_many_arguments)]
fn project_q_kv_gate(
    x_norm: &MxArray,
    attn: &AttnWeights,
    q_lora: i64,
    kv_row: i64,
    gate_dim: i64,
    sw_qkvg: Option<&MxArray>,
    sw_qkv: Option<&MxArray>,
    sw_qa: Option<&MxArray>,
    sw_kva: Option<&MxArray>,
    sw_gate: Option<&MxArray>,
    prefill_mode: Nvfp4PrefillMode,
) -> Result<(MxArray, MxArray, MxArray)> {
    let m = x_norm.shape_at(0)?;
    if let Some(qkvg) = &attn.q_kv_gate {
        let out = cuda_linear_bf16(x_norm, qkvg, sw_qkvg, prefill_mode)?;
        let q_a = out.slice(&[0, 0], &[m, q_lora])?;
        let kv_a = out.slice(&[0, q_lora], &[m, q_lora + kv_row])?;
        let gate = out.slice(&[0, q_lora + kv_row], &[m, q_lora + kv_row + gate_dim])?;
        return Ok((q_a, kv_a, gate));
    }
    if let Some(qkv) = &attn.q_kv {
        let out = cuda_linear_bf16(x_norm, qkv, sw_qkv, prefill_mode)?;
        let q_a = out.slice(&[0, 0], &[m, q_lora])?;
        let kv_a = out.slice(&[0, q_lora], &[m, q_lora + kv_row])?;
        let gate = cuda_linear_bf16(x_norm, &attn.gate, sw_gate, prefill_mode)?;
        return Ok((q_a, kv_a, gate));
    }
    let q_a = cuda_linear_bf16(x_norm, &attn.q_a, sw_qa, prefill_mode)?;
    let kv_a = cuda_linear_bf16(x_norm, &attn.kv_a, sw_kva, prefill_mode)?;
    let gate = cuda_linear_bf16(x_norm, &attn.gate, sw_gate, prefill_mode)?;
    Ok((q_a, kv_a, gate))
}

fn resolve_router_weight<'a>(
    mw: &'a MoeWeights,
    cached: Option<&'a MxArray>,
    scratch: &'a mut Option<MxArray>,
) -> Result<&'a MxArray> {
    if let Some(c) = cached {
        return Ok(c);
    }
    if let Dense::Array(a) = &mw.router
        && matches!(a.dtype()?, DType::Float32 | DType::BFloat16)
    {
        return Ok(a);
    }
    Ok(scratch.insert(mw.router.f32()?))
}

#[allow(clippy::too_many_arguments)]
fn run_moe_cuda(
    ffn_in: &MxArray,
    mw: &MoeWeights,
    cfg: &HyV4FlashConfig,
    swiglu_limit: f32,
    router_w: &MxArray,
    gate_w: &MxArray,
    gate_s: &MxArray,
    up_w: &MxArray,
    up_s: &MxArray,
    down_w: &MxArray,
    down_s: &MxArray,
    moe_mode: MoeWeightMode,
    sw_dgu: Option<&MxArray>,
    sw_rd: Option<&MxArray>,
    sw_sg: Option<&MxArray>,
    sw_su: Option<&MxArray>,
    sw_ru: Option<&MxArray>,
    sw_sd: Option<&MxArray>,
    prefill_mode: Nvfp4PrefillMode,
) -> Result<MxArray> {
    let m_i64 = ffn_in.shape_at(0)?;
    let m = m_i64 as usize;
    let lat = cfg.moe.latent as i64;
    let sh_ff = mw.shared_gate.shape_at(0)?;
    let tp_shared_in = prof_start();
    let (z, sh_g, sh_u) = if let Some(dgu) = &mw.down_shared_gu {
        let dgu_out = cuda_linear_bf16(ffn_in, dgu, sw_dgu, prefill_mode)?;
        let z = dgu_out.slice(&[0, 0], &[m_i64, lat])?;
        let sh_g = dgu_out.slice(&[0, lat], &[m_i64, lat + sh_ff])?;
        let sh_u = dgu_out.slice(&[0, lat + sh_ff], &[m_i64, lat + 2 * sh_ff])?;
        (z, sh_g, sh_u)
    } else {
        let z = cuda_linear_bf16(ffn_in, &mw.routed_down, sw_rd, prefill_mode)?;
        let sh_g = cuda_linear_bf16(ffn_in, &mw.shared_gate, sw_sg, prefill_mode)?;
        let sh_u = cuda_linear_bf16(ffn_in, &mw.shared_up, sw_su, prefill_mode)?;
        (z, sh_g, sh_u)
    };
    prof_end(Phase::Shared, tp_shared_in, &[&z, &sh_g, &sh_u])?;

    let tp_router = prof_start();
    let (topk_ids, topk_weights) = hy4::moe_router_topk(
        ffn_in,
        router_w,
        mw.router_bias.as_ref(),
        cfg.moe.n_used as i32,
        cfg.moe.route_scale,
        cfg.moe.norm_topk,
    )?
    .ok_or_else(|| bad("hy4::moe_router_topk CUDA kernel declined input"))?;
    prof_end(Phase::Router, tp_router, &[&topk_ids, &topk_weights])?;

    let tp_routed = prof_start();
    let r = if m <= 16 {
        hy4::moe_routed_few(
            &z,
            &topk_ids,
            &topk_weights,
            gate_w,
            gate_s,
            up_w,
            up_s,
            down_w,
            down_s,
            moe_mode,
            swiglu_limit,
        )?
        .ok_or_else(|| bad("hy4::moe_routed_few CUDA kernel declined input"))?
    } else {
        hy4::moe_routed_prefill(
            &z,
            &topk_ids,
            &topk_weights,
            gate_w,
            gate_s,
            up_w,
            up_s,
            down_w,
            down_s,
            moe_mode,
            swiglu_limit,
        )?
        .ok_or_else(|| bad("hy4::moe_routed_prefill CUDA kernel declined input"))?
    };

    let y_routed = cuda_linear_bf16(&r, &mw.routed_up, sw_ru, prefill_mode)?;
    prof_end(Phase::Routed, tp_routed, &[&y_routed])?;

    let tp_shared_out = prof_start();
    let sh_act = Activations::swiglu_compiled(&sh_g, &sh_u)?;
    let y_shared = cuda_linear_bf16(&sh_act, &mw.shared_down, sw_sd, prefill_mode)?;
    let y_out = y_routed.add(&y_shared)?;
    prof_end(Phase::Shared, tp_shared_out, &[&y_out])?;
    Ok(y_out)
}

fn find_cuda_direct_bank(banks: &[std::sync::Arc<KqBank>]) -> Option<&KqBank> {
    banks
        .iter()
        .find(|b| b.is_cuda_direct())
        .map(|b| b.as_ref())
}

fn expert_mode_of(layer: &CudaExpertLayer) -> Result<MoeWeightMode> {
    match layer.class {
        ExpertClass::Nvfp4 => Ok(MoeWeightMode::Nvfp4),
        ExpertClass::Q8_0 => Ok(MoeWeightMode::SymQ8),
        other => Err(bad(format!(
            "unsupported CUDA direct expert class {other:?}"
        ))),
    }
}

/// Full CUDA fast forward pass for a single-owner trunk call (`prefill`, `decode`,
/// or `MTP-3 verify`).
///
/// Wraps `RowKind::Prefill` (`M > 16`) in `CudaGraphsGuard::disable_for_prefill()`
/// so decode & verify (`M <= 16`) stay inside CUDA Graphs with fixed GPU buffer pointers.
pub(crate) fn forward_trunk_cuda(
    model: &Model,
    sess: &mut Session,
    ids: &[u32],
    kind: RowKind,
    want: Want,
    cancel: Option<&dyn Fn(usize) -> bool>,
) -> Result<Output> {
    let cfg = model.cfg();
    if let Some(&bad_id) = ids.iter().find(|&&id| id as usize >= cfg.vocab_size) {
        return Err(bad(format!(
            "token id {bad_id} outside the vocab of {}",
            cfg.vocab_size
        )));
    }
    let cache = model
        .cuda_cache()
        .ok_or_else(|| bad("Model has no CudaForwardCache"))?;

    let _graphs_guard =
        (kind == RowKind::Prefill || ids.len() > 16).then(CudaGraphsGuard::disable_for_prefill);

    // Snapshot evaluated pre-step GPU buffer handles so a failed or cancelled
    // deferred write never leaves a poisoned lazy MxArray in the session cache.
    // On short steps (`sess.can_short_mark(ids.len())`), every sliding layer uses
    // `Hy4MlaPrepKvStoreCuda` (`copy_shared_buffer`, no refcount-1 requirement);
    // on longer prefills, `model.rs` takes `StepMark::Full` and replaces
    // `WindowRing` from its snapshot.
    let save_window = sess.can_short_mark(ids.len());
    let saved_bufs: Vec<(Option<MxArray>, Option<MxArray>, Option<MxArray>)> = sess
        .layers()
        .iter()
        .map(|lst| {
            let kv = match &lst.cache {
                LayerCache::Window(r) if save_window => r.cuda_buf_clone(),
                LayerCache::Window(_) => None,
                LayerCache::Latent(b) => b.cuda_buf_clone(),
            };
            let pooled = lst.index.as_ref().and_then(|ix| ix.cuda_pooled_buf_clone());
            let tail = lst.index.as_ref().and_then(|ix| ix.cuda_tail_rows_clone());
            (kv, pooled, tail)
        })
        .collect();

    let scope = DeferScope::enter();
    let res = forward_trunk_cuda_inner(model, cache, sess, ids, kind, want, cancel);
    if res.is_err() {
        crate::array::synchronize();
        for (lst, (prev_kv, prev_pooled, prev_tail)) in sess.layers_mut().iter_mut().zip(saved_bufs)
        {
            match &mut lst.cache {
                LayerCache::Window(r) if save_window => r.cuda_restore_buf_opt(prev_kv),
                LayerCache::Window(_) => {}
                LayerCache::Latent(b) => b.cuda_restore_buf_opt(prev_kv),
            }
            if let Some(ix) = lst.index.as_mut() {
                ix.cuda_restore_pooled_buf(prev_pooled);
                ix.cuda_restore_tail_rows(prev_tail);
            }
        }
    }
    for b in model.banks() {
        b.settle_all();
    }
    drop(scope);
    res
}

#[allow(clippy::too_many_arguments)]
fn forward_trunk_cuda_inner(
    model: &Model,
    cache: &CudaForwardCache,
    sess: &mut Session,
    ids: &[u32],
    kind: RowKind,
    want: Want,
    cancel: Option<&dyn Fn(usize) -> bool>,
) -> Result<Output> {
    let cfg = model.cfg();
    let w = model.weights();
    let table = model.table();
    let hasher = model.hasher();
    let blocks = model.blocks();
    let cuda_bank = find_cuda_direct_bank(model.banks());
    let prefill_mode = nvfp4_prefill_mode();

    let start = sess.len();
    let m = ids.len();
    let m_i64 = m as i64;
    let hc_i64 = cfg.hc.count as i64;
    let dim_i64 = cfg.dim as i64;
    let eps = cfg.norm_eps;
    let hc_eps = cfg.hc.eps;
    let hc_mag = cfg.hc.magnitude;

    // 1. Engram hashing & staggered async prefetch (layer 0 before blk.0, ei+1 as ei is consumed).
    let tp_eng = prof_start();
    let keep_bool = keeps(ids, cfg.engram.pad_token_id);
    let all_keep = keep_bool.iter().all(|&k| k);
    let keep_mask_arr = if all_keep {
        None
    } else {
        let mask_u8: Vec<u8> = keep_bool.iter().map(|&k| u8::from(k)).collect();
        Some(MxArray::from_uint8(&mask_u8, &[m_i64])?)
    };

    let n_engram = hasher.n_layers();
    let mut engram_layer_ids: Vec<Vec<u32>> = Vec::with_capacity(n_engram);
    let mut engram_handles: Vec<Option<PrefetchHandle>> = Vec::with_capacity(n_engram);
    let matched_lookahead = if kind == RowKind::Prefill {
        if let Some((layer_ids, pre_handles)) =
            model.take_engram_chunk_cuda(start, ids, sess.tokens())
        {
            engram_layer_ids = layer_ids;
            engram_handles = pre_handles;
            if n_engram > 0 && engram_handles.first().is_some_and(Option::is_none) {
                engram_handles[0] = table.prefetch(0, &engram_layer_ids[0])?;
            }
            true
        } else {
            false
        }
    } else {
        false
    };
    if !matched_lookahead && n_engram > 0 {
        let rows = call_engram_rows(hasher, cfg.engram.max_ngram - 1, sess.tokens(), ids)?;
        for e in 0..n_engram {
            engram_layer_ids.push(hasher.layer_ids(&rows, e)?);
            engram_handles.push(None);
        }
        // On CUDA decode/verify (m <= 16), submit all Engram layers upfront in ONE
        // batched `prefetch_all` call (issuing a single `PrefetchVirtualMemory` /
        // `MADV_WILLNEED` across all 4 layers when OS page-cached mmap is active).
        let prefetch_count = if m <= 16 { n_engram } else { n_engram.min(2) };
        let reqs: Vec<(usize, &[u32])> = (0..prefetch_count)
            .map(|e| (e, engram_layer_ids[e].as_slice()))
            .collect();
        let prefetched = table.prefetch_all(&reqs)?;
        for (e, h) in prefetched.into_iter().enumerate() {
            engram_handles[e] = h;
        }
    }
    prof_end(Phase::Engram, tp_eng, &[])?;

    // 2. Token embedding gather -> BF16 -> broadcast [M, 1, D] -> [M, HC, D].
    let tok = MxArray::from_uint32(ids, &[m_i64])?;
    let emb = w.token_embd().rows(&tok)?.astype(DType::BFloat16)?;
    let mut x = emb
        .reshape(&[m_i64, 1, dim_i64])?
        .broadcast_to(&[m_i64, hc_i64, dim_i64])?;

    let q_lora = cfg.mla.q_lora_rank as i64;
    let kv_lora = cfg.mla.kv_lora_rank;
    let kv_lora_i64 = kv_lora as i64;
    let nope = cfg.mla.qk_nope_dim;
    let rope = cfg.mla.rope_dim;
    let kv_row = cfg.mla.kv_row();
    let kv_row_i64 = kv_row as i64;
    let head_v = cfg.mla.head_dim_v;
    let mla_scale = cfg.mla.scale;
    let h_idx_i64 = cfg.indexer.n_heads as i64;
    let d_idx_i64 = cfg.indexer.head_dim as i64;

    let layers_w = w.layers();
    let sess_layers = sess.layers_mut();

    // 3. Trunk layer loop.
    // Carry `(y_ffn, post_ffn, post_ffw_norm)` across layer boundaries when the
    // next layer has no Engram and no prefill memory flush is needed so
    // `hc_post_norm(ffn)` fuses with the next layer's `hc_pre_norm(attn)`.
    let mut pending_ffn_post: Option<(MxArray, MxArray, &MxArray)> = None;
    for il in 0..cfg.n_layers {
        if cancel.is_some_and(|c| c(il)) {
            return Err(bad(format!("cancelled before layer {il}")));
        }
        let lw = &layers_w[il];
        let lc = &cfg.layers[il];
        let blk = &blocks[il];
        let lcache = &cache.layers[il];
        let lst = &mut sess_layers[il];

        // 3a. Engram residual (before HC attention mix).
        if let (Some(ei), Some(ew)) = (blk.engram(), lw.engram.as_ref()) {
            let tp = prof_start();
            let raw_f32 = match engram_handles.get_mut(ei).and_then(Option::take) {
                Some(h) => {
                    // On cold NVMe misses (ei > 0), workers may still be faulting
                    // pages while the CPU builds layers 0..il lazily. Kick off GPU
                    // work for preceding layers asynchronously ONLY when this layer's
                    // prefetch is not yet ready: zero graph split on OS page-cache
                    // hits, full GPU/NVMe overlap (~12 ms/block) on cold misses.
                    if ei > 0 && !h.is_ready() && !super::stats::profiling() {
                        MxArray::eval_arrays_with_context(&[&x], "hy_v4_flash engram overlap")?;
                    }
                    table.take_prefetched(h)?
                }
                None => table.rows(ei, &engram_layer_ids[ei])?,
            };
            // Keep up to 2 Engram layers in flight on NVMe during prefill
            // (decode m <= 16 already submitted all n_engram layers upfront).
            for ahead in 1..=2 {
                if let (Some(next_slot), Some(next_ids)) = (
                    engram_handles.get_mut(ei + ahead),
                    engram_layer_ids.get(ei + ahead),
                ) && next_slot.is_none()
                {
                    *next_slot = table.prefetch(ei + ahead, next_ids)?;
                }
            }
            if matched_lookahead {
                model.prefetch_next_engram_layer_cuda(ei);
            }
            let n_cols = ew.lookup_norm.shape_at(0)?;
            let e_hd = ew.lookup_norm.shape_at(1)?;
            let raw_arr = MxArray::from_float32(&raw_f32, &[m_i64, n_cols, e_hd])?;
            let rn = hy4::engram_lookup_norm(&raw_arr, &ew.lookup_norm)?
                .ok_or_else(|| bad(format!("layer {il}: hy4::engram_lookup_norm declined")))?;
            let kv = cuda_linear_bf16(&rn, &ew.kv, lcache.engram_kv.as_ref(), prefill_mode)?;
            x = hy4::engram_gate_add(
                &x,
                &kv,
                &ew.kv_b,
                &ew.gate,
                &ew.pre_gamma,
                &ew.hc_scale,
                keep_mask_arr.as_ref(),
            )?
            .ok_or_else(|| bad(format!("layer {il}: hy4::engram_gate_add declined")))?;
            prof_end(Phase::Engram, tp, &[&x])?;
        }

        let Residual::Hc {
            attn: hc_attn,
            ffn: hc_ffn,
        } = &lw.residual
        else {
            return Err(bad(format!("layer {il}: expected Residual::Hc")));
        };

        // 3b. HC Pre + Attention Norm (fused with previous layer's FFN HC Post when eligible).
        let tp_hc_attn = prof_start();
        let (hc_attn_w, hc_attn_mode) = hc_fn_mode(&hc_attn.weight, lcache.hc_attn_w.as_ref())?;
        let (x_norm, post_attn) =
            if let Some((prev_y, prev_post, prev_post_norm)) = pending_ffn_post.take() {
                let (x_next, h_norm, post_a) = hy4::hc_post_pre_norm(
                    &x,
                    &prev_y,
                    &prev_post,
                    Some(prev_post_norm),
                    &hc_attn_w,
                    None,
                    &hc_attn.scale,
                    &hc_attn.base,
                    Some(&lw.attn.norm),
                    hc_attn_mode,
                    eps,
                    hc_eps,
                    hc_mag,
                )?
                .ok_or_else(|| {
                    bad(format!(
                        "layer {il}: hy4::hc_post_pre_norm (ffn->attn) declined"
                    ))
                })?;
                x = x_next;
                prof_end(Phase::Hc, tp_hc_attn, &[&x, &h_norm, &post_a])?;
                (h_norm, post_a)
            } else {
                let (h_norm, post_a) = hy4::hc_pre_norm(
                    &x,
                    &hc_attn_w,
                    None,
                    &hc_attn.scale,
                    &hc_attn.base,
                    Some(&lw.attn.norm),
                    hc_attn_mode,
                    eps,
                    hc_eps,
                    hc_mag,
                )?
                .ok_or_else(|| bad(format!("layer {il}: hy4::hc_pre_norm (attn) declined")))?;
                prof_end(Phase::Hc, tp_hc_attn, &[&h_norm, &post_a])?;
                (h_norm, post_a)
            };

        // 3c. Absorbed MLA.
        let tp_attn_proj = prof_start();
        let n_heads = lc.n_heads;
        let h_i64 = n_heads as i64;
        let gate_dim = (n_heads * head_v) as i64;
        let (q_a, kv_a, gate) = project_q_kv_gate(
            &x_norm,
            &lw.attn,
            q_lora,
            kv_row_i64,
            gate_dim,
            lcache.attn_q_kv_gate.as_ref(),
            lcache.attn_q_kv.as_ref(),
            lcache.attn_q_a.as_ref(),
            lcache.attn_kv_a.as_ref(),
            lcache.attn_gate.as_ref(),
            prefill_mode,
        )?;

        let inv_freq = cache.inv_freq_for(il, lc.attn);
        let end = start + m;

        // KV store (ring / large-prefill linear / growable indexed slab) + q_a_norm.
        let linear_keys_holder: Option<MxArray>;
        let (qr, sdpa_slot_start, sdpa_total_len, sdpa_ring_cap, sdpa_window, idx_opt) =
            match (&mut lst.cache, lst.index.as_mut()) {
                (LayerCache::Window(ring), None) => {
                    let cap = ring.cap();
                    let kept = ring.kept();
                    let first_needed = (start + 1).saturating_sub(cfg.window).max(start - kept);
                    if end - first_needed <= cap {
                        let ring_buf = ring.cuda_take_buf_or_alloc()?;
                        let prep = hy4::mla_prep_kv_store(
                            Some(&q_a),
                            &kv_a,
                            Some(&lw.attn.q_a_norm),
                            &lw.attn.kv_a_norm,
                            inv_freq,
                            &ring_buf,
                            start as i64,
                            start as i64,
                            cap as i32,
                            eps,
                        );
                        let (qr_opt, out_kv) = match prep {
                            Ok(Some(pair)) => pair,
                            Ok(None) => {
                                ring.cuda_restore_buf(ring_buf);
                                return Err(bad(format!(
                                    "layer {il}: hy4::mla_prep_kv_store (ring) declined"
                                )));
                            }
                            Err(e) => {
                                ring.cuda_restore_buf(ring_buf);
                                return Err(e);
                            }
                        };
                        drop(ring_buf);
                        ring.cuda_commit(out_kv, m)?;
                        let qr = qr_opt.ok_or_else(|| bad("mla_prep_kv_store missing qr"))?;
                        linear_keys_holder = None;
                        (
                            qr,
                            start as i64,
                            end as i32,
                            cap as i32,
                            cfg.window as i32,
                            None,
                        )
                    } else {
                        // Large prefill exceeding ring capacity: linear keys for SDPA + tail into ring.
                        let tmp_kv =
                            match MxArray::alloc_gpu_uninit(&[m_i64, kv_row_i64], ROW_DTYPE)? {
                                Some(b) => b,
                                None => MxArray::zeros(&[m_i64, kv_row_i64], Some(ROW_DTYPE))?,
                            };
                        let (qr_opt, new_kv) = hy4::mla_prep_kv_store(
                            Some(&q_a),
                            &kv_a,
                            Some(&lw.attn.q_a_norm),
                            &lw.attn.kv_a_norm,
                            inv_freq,
                            &tmp_kv,
                            0,
                            start as i64,
                            0,
                            eps,
                        )?
                        .ok_or_else(|| {
                            bad(format!(
                                "layer {il}: hy4::mla_prep_kv_store (linear) declined"
                            ))
                        })?;
                        drop(tmp_kv);
                        let qr = qr_opt.ok_or_else(|| bad("mla_prep_kv_store missing qr"))?;
                        let prefix_len = start - first_needed;
                        let keys = if prefix_len > 0 {
                            let prefix = ring.cuda_linear_prefix(first_needed, prefix_len)?;
                            MxArray::concatenate(&prefix, &new_kv, 0)?
                        } else {
                            new_kv.clone()
                        };
                        let (src, write_start) = if m >= cap {
                            (
                                new_kv.slice(&[(m - cap) as i64, 0], &[m_i64, kv_row_i64])?,
                                end - cap,
                            )
                        } else {
                            (new_kv, start)
                        };
                        let prev = ring.cuda_take_buf_or_alloc()?;
                        // When `prefix_len > 0`, `keys` lazily references `prev`. Because
                        // `Hy4RingWriteCuda` writes in place via `copy_shared_buffer`,
                        // write into a fresh buffer (when `m >= cap`, all `cap` rows are
                        // overwritten) or a detached copy (when `m < cap`) so `ring_write`
                        // never overwrites `prefix` before `mla_sdpa` reads `keys`.
                        let write_target = if prefix_len > 0 {
                            if m >= cap {
                                match MxArray::alloc_gpu_uninit(
                                    &[cap as i64, kv_row_i64],
                                    ROW_DTYPE,
                                )? {
                                    Some(b) => b,
                                    None => {
                                        MxArray::zeros(&[cap as i64, kv_row_i64], Some(ROW_DTYPE))?
                                    }
                                }
                            } else {
                                prev.deep_copy()?
                            }
                        } else {
                            prev.clone()
                        };
                        let next_res = hy4::ring_write_in_place(
                            &write_target,
                            &src,
                            write_start as i64,
                            cap as i32,
                        );
                        let next = match next_res {
                            Ok(Some(u)) => u,
                            Ok(None) => {
                                super::ops::ring_write(&write_target, &src, write_start % cap, cap)?
                            }
                            Err(e) => {
                                ring.cuda_restore_buf(prev);
                                return Err(e);
                            }
                        };
                        drop(write_target);
                        drop(prev);
                        ring.cuda_commit(next, m)?;
                        linear_keys_holder = Some(keys);
                        (
                            qr,
                            (start - first_needed) as i64,
                            (end - first_needed) as i32,
                            0,
                            cfg.window as i32,
                            None,
                        )
                    }
                }
                (LayerCache::Latent(buf), Some(ix_state)) => {
                    let slab_buf = buf.cuda_ensure_buf(end)?;
                    let (qr_opt, out_kv) = hy4::mla_prep_kv_store(
                        Some(&q_a),
                        &kv_a,
                        Some(&lw.attn.q_a_norm),
                        &lw.attn.kv_a_norm,
                        inv_freq,
                        slab_buf,
                        start as i64,
                        start as i64,
                        0,
                        eps,
                    )?
                    .ok_or_else(|| {
                        bad(format!(
                            "layer {il}: hy4::mla_prep_kv_store (latent) declined"
                        ))
                    })?;
                    buf.cuda_commit(out_kv, end)?;
                    let qr = qr_opt.ok_or_else(|| bad("mla_prep_kv_store missing qr"))?;
                    prof_end(Phase::AttnProj, tp_attn_proj, &[&qr, &gate, buf.rows()?])?;

                    let tp_idx = prof_start();
                    let iw = lw
                        .indexer
                        .as_ref()
                        .ok_or_else(|| bad(format!("layer {il}: missing indexer weights")))?;
                    let indexer = blk
                        .indexer()
                        .ok_or_else(|| bad(format!("layer {il}: missing indexer module")))?;
                    let tok_k_raw = cuda_linear_bf16(&x_norm, &iw.attn_k, None, prefill_mode)?;
                    let tok_k = rms_norm(&tok_k_raw.astype(DType::Float32)?, &iw.k_norm, eps)?;
                    let q_idx_raw = cuda_linear_bf16(&qr, &iw.q_b, None, prefill_mode)?
                        .reshape(&[m_i64, h_idx_i64, d_idx_i64])?;
                    let w_proj = match lcache.indexer_proj_w.as_ref() {
                        Some(cached) => cuda_linear_bf16_cached_array(&x_norm, cached)?,
                        None => cuda_linear_bf16(&x_norm, &iw.proj, None, prefill_mode)?,
                    };
                    let idx = indexer.cuda_select(
                        iw, &tok_k, &q_idx_raw, &w_proj, inv_freq, start, start, ix_state,
                    )?;
                    prof_end(Phase::Indexer, tp_idx, &[&idx])?;
                    linear_keys_holder = None;
                    (qr, start as i64, end as i32, 0, 0, Some(idx))
                }
                _ => {
                    return Err(bad(format!(
                        "layer {il}: cache / indexer state kind mismatch"
                    )));
                }
            };
        if idx_opt.is_none() {
            let k_ref: &MxArray = match (&linear_keys_holder, &lst.cache) {
                (Some(lin), _) => lin,
                (None, LayerCache::Window(ring)) => ring
                    .buf()
                    .ok_or_else(|| bad(format!("layer {il}: window ring buffer missing")))?,
                (None, LayerCache::Latent(buf)) => buf.rows()?,
            };
            prof_end(Phase::AttnProj, tp_attn_proj, &[&qr, &gate, k_ref])?;
        }

        // Q expand + absorb + NORM-RoPE.
        let tp_q_exp = prof_start();
        let q_raw = cuda_linear_bf16(&qr, &lw.attn.q_b, lcache.attn_q_b.as_ref(), prefill_mode)?
            .reshape(&[m_i64, h_i64, (nope + rope) as i64])?;
        let q_nope = q_raw.slice(&[0, 0, 0], &[m_i64, h_i64, nope as i64])?;
        let q_abs = cuda_grouped_q8(&q_nope, &lw.attn.k_b, None, n_heads, nope, kv_lora)?;
        let q = hy4::rope_norm_q(&q_abs, &q_raw, inv_freq, start as i64, nope as i32)?
            .ok_or_else(|| bad(format!("layer {il}: hy4::rope_norm_q declined")))?;
        prof_end(Phase::AttnProj, tp_q_exp, &[&q])?;

        // MLA Sink Softmax SDPA.
        let tp_sdpa = prof_start();
        let keys_buf: &MxArray = match (&linear_keys_holder, &lst.cache) {
            (Some(lin), _) => lin,
            (None, LayerCache::Window(ring)) => ring
                .buf()
                .ok_or_else(|| bad(format!("layer {il}: window ring buffer missing")))?,
            (None, LayerCache::Latent(buf)) => buf.rows()?,
        };
        let o_lat = hy4::mla_sdpa(
            &q,
            keys_buf,
            &lw.attn.sinks,
            idx_opt.as_ref(),
            sdpa_slot_start,
            sdpa_total_len,
            sdpa_ring_cap,
            sdpa_window,
            mla_scale,
            kv_lora_i64 as i32,
        )?
        .ok_or_else(|| bad(format!("layer {il}: hy4::mla_sdpa declined")))?;
        prof_end(Phase::Sparse, tp_sdpa, &[&o_lat])?;

        // V expand + fused * sigmoid(gate) + output + HC post + post_attention_norm.
        let tp_v_out = prof_start();
        let o_gated = cuda_grouped_q8(&o_lat, &lw.attn.v_b, Some(&gate), n_heads, kv_lora, head_v)?
            .reshape(&[m_i64, gate_dim])?;
        let y_attn = cuda_linear_bf16(
            &o_gated,
            &lw.attn.output,
            lcache.attn_output.as_ref(),
            prefill_mode,
        )?;
        prof_end(Phase::AttnProj, tp_v_out, &[&y_attn])?;

        // 3d. Fused HC Post (attn) + HC Pre (ffn) + FFN Norm.
        let tp_hc_ffn = prof_start();
        let (hc_ffn_w, hc_ffn_mode) = hc_fn_mode(&hc_ffn.weight, lcache.hc_ffn_w.as_ref())?;
        let (x_next, ffn_in, post_ffn) = hy4::hc_post_pre_norm(
            &x,
            &y_attn,
            &post_attn,
            Some(&lw.attn.post_norm),
            &hc_ffn_w,
            None,
            &hc_ffn.scale,
            &hc_ffn.base,
            Some(&lw.ffn_norm),
            hc_ffn_mode,
            eps,
            hc_eps,
            hc_mag,
        )?
        .ok_or_else(|| {
            bad(format!(
                "layer {il}: hy4::hc_post_pre_norm (attn->ffn) declined"
            ))
        })?;
        x = x_next;
        prof_end(Phase::Hc, tp_hc_ffn, &[&x, &ffn_in, &post_ffn])?;

        // 3e. FFN (Dense MLP on leading layers, Routed + Shared MoE on remaining layers).
        let y_ffn = match &lw.ffn {
            Ffn::Mlp {
                gate,
                up,
                down,
                gate_up,
            } => {
                let tp_mlp = prof_start();
                let (g, u) = if let Some(gu) = gate_up {
                    let gu_out =
                        cuda_linear_bf16(&ffn_in, gu, lcache.ffn_gate_up.as_ref(), prefill_mode)?;
                    let ff = gu_out.shape_at(1)? / 2;
                    (
                        gu_out.slice(&[0, 0], &[m_i64, ff])?,
                        gu_out.slice(&[0, ff], &[m_i64, 2 * ff])?,
                    )
                } else {
                    (
                        cuda_linear_bf16(&ffn_in, gate, lcache.ffn_gate.as_ref(), prefill_mode)?,
                        cuda_linear_bf16(&ffn_in, up, lcache.ffn_up.as_ref(), prefill_mode)?,
                    )
                };
                let act = Activations::swiglu_compiled(&g, &u)?;
                let out = cuda_linear_bf16(&act, down, lcache.ffn_down.as_ref(), prefill_mode)?;
                prof_end(Phase::Shared, tp_mlp, &[&out])?;
                out
            }
            Ffn::Moe(mw) => {
                let mut router_scratch = None;
                let router_w =
                    resolve_router_weight(mw, lcache.router_w.as_ref(), &mut router_scratch)?;
                if let Some(re) = lcache.resident_experts.as_ref() {
                    run_moe_cuda(
                        &ffn_in,
                        mw,
                        cfg,
                        lc.swiglu_limit,
                        router_w,
                        &re.gate_w,
                        &re.gate_s,
                        &re.up_w,
                        &re.up_s,
                        &re.down_w,
                        &re.down_s,
                        MoeWeightMode::SymQ8,
                        lcache.moe_down_shared_gu.as_ref(),
                        lcache.moe_routed_down.as_ref(),
                        lcache.moe_shared_gate.as_ref(),
                        lcache.moe_shared_up.as_ref(),
                        lcache.moe_routed_up.as_ref(),
                        lcache.moe_shared_down.as_ref(),
                        prefill_mode,
                    )?
                } else {
                    let bank = cuda_bank.ok_or_else(|| {
                        bad(format!("layer {il}: missing CUDA direct expert bank"))
                    })?;
                    let ce = bank.cuda_layer_experts(il)?.ok_or_else(|| {
                        bad(format!("layer {il}: bank has no CUDA direct experts"))
                    })?;
                    let moe_mode = expert_mode_of(ce)?;
                    run_moe_cuda(
                        &ffn_in,
                        mw,
                        cfg,
                        lc.swiglu_limit,
                        router_w,
                        &ce.gate.w,
                        &ce.gate.scales,
                        &ce.up.w,
                        &ce.up.scales,
                        &ce.down.w,
                        &ce.down.scales,
                        moe_mode,
                        lcache.moe_down_shared_gu.as_ref(),
                        lcache.moe_routed_down.as_ref(),
                        lcache.moe_shared_gate.as_ref(),
                        lcache.moe_shared_up.as_ref(),
                        lcache.moe_routed_up.as_ref(),
                        lcache.moe_shared_down.as_ref(),
                        prefill_mode,
                    )?
                }
            }
        };

        // 3f. HC Post + Post FFW Norm (deferred to fuse with next layer's HC Pre
        // when the next layer has no Engram and no prefill memory flush happens here).
        // Bound peak prefill intermediate memory so all 42 layers' buffers do
        // not stay live simultaneously against WDDM / unified-memory headroom.
        // `stats::eval` atomically evaluates `x` plus all deferred KV/indexer
        // writes in one call, and `synchronize` drains MLX's CUDA completion
        // worker so layer `il`'s temporaries return to `buffer_cache_` before
        // layer `il + 1` allocates.
        let prefill_flush = kind == RowKind::Prefill
            && !super::stats::profiling()
            && (m > 1024 || (m > 16 && il >= 2 && (il - 2).is_multiple_of(3)));
        let can_fuse_next = !prefill_flush
            && il + 1 < cfg.n_layers
            && !(blocks[il + 1].engram().is_some() && layers_w[il + 1].engram.is_some());
        if can_fuse_next {
            pending_ffn_post = Some((y_ffn, post_ffn, &lw.post_ffw_norm));
        } else {
            let tp_hc_post_f = prof_start();
            x = hy4::hc_post_norm(&x, &y_ffn, &post_ffn, Some(&lw.post_ffw_norm), eps)?
                .ok_or_else(|| bad(format!("layer {il}: hy4::hc_post_norm (ffn) declined")))?;
            prof_end(Phase::Hc, tp_hc_post_f, &[&x])?;
            if prefill_flush {
                super::stats::eval(&[&x], "hy_v4_flash cuda prefill layer")?;
                crate::array::synchronize();
            }
        }
    }

    // 4. Output HC + output_norm + LM head (f32 logits + optional f32 hidden).
    let tp_head = prof_start();
    if !want.hidden && want.logits == Logits::None {
        super::stats::eval(&[&x], "hy_v4_flash cuda head")?;
        prof_end(Phase::Head, tp_head, &[])?;
        return Ok(Output {
            logits: None,
            hidden: None,
        });
    }

    let ohc = w.output_hc();
    let (ohc_w, ohc_mode) = cache.output_hc_bf16(&ohc.weight)?;

    let (logits, hidden) = if !want.hidden && want.logits == Logits::Last && m > 1 {
        let x_last = x.slice(&[m_i64 - 1, 0, 0], &[m_i64, hc_i64, dim_i64])?;
        let h_last = hy4::hc_head_norm(
            &x_last,
            &ohc_w,
            None,
            &ohc.scale,
            &ohc.base,
            w.output_norm(),
            ohc_mode,
            eps,
            hc_eps,
        )?
        .ok_or_else(|| bad("hy4::hc_head_norm (last) declined"))?;
        let logits = cuda_lm_head_f32(&h_last, w.output())?;
        (Some(logits), None)
    } else {
        let h_trunk = hy4::hc_head_norm(
            &x,
            &ohc_w,
            None,
            &ohc.scale,
            &ohc.base,
            w.output_norm(),
            ohc_mode,
            eps,
            hc_eps,
        )?
        .ok_or_else(|| bad("hy4::hc_head_norm declined"))?;
        let logits = match want.logits {
            Logits::None => None,
            Logits::All => Some(cuda_lm_head_f32(&h_trunk, w.output())?),
            Logits::Last => {
                let h_last = if m == 1 {
                    h_trunk.clone()
                } else {
                    h_trunk.slice(&[m_i64 - 1, 0], &[m_i64, dim_i64])?
                };
                Some(cuda_lm_head_f32(&h_last, w.output())?)
            }
        };
        let hidden = want
            .hidden
            .then(|| h_trunk.astype(DType::Float32))
            .transpose()?;
        (logits, hidden)
    };

    let mut live: Vec<&MxArray> = Vec::with_capacity(3);
    live.push(&x);
    if let Some(l) = logits.as_ref() {
        live.push(l);
    }
    if let Some(h) = hidden.as_ref() {
        live.push(h);
    }
    super::stats::eval(&live, "hy_v4_flash cuda head")?;
    prof_end(Phase::Head, tp_head, &[])?;

    if want.logits != Logits::None {
        maybe_clear_cuda_cache_after_prefill(kind);
    }
    Ok(Output { logits, hidden })
}

fn maybe_clear_cuda_cache_after_prefill(kind: RowKind) {
    if kind == RowKind::Prefill
        && crate::array::get_cache_memory() > cuda_cache_clear_threshold_bytes()
    {
        crate::array::clear_cache();
    }
}

/// CUDA fast path for `MtpLayer::commit`: updates MTP KV rows (at `slot=at, rope=at+1`)
/// and MTP indexer state (at `slot=at`).
pub(crate) fn mtp_commit_cuda(
    mtp: &MtpLayer,
    model: &Model,
    d: &mut Draft,
    h: &MxArray,
    ids: &[u32],
) -> Result<()> {
    mtp_commit_cuda_with_flush(mtp, model, d, h, ids, true)
}

/// Like [`mtp_commit_cuda`]; when `flush` is false (used by `Flat::commit_mtp` and
/// `Lagged::step` inside an active `DeferScope`), leaves deferred KV/indexer updates
/// in the caller's scope so they chain into the following draft / trunk forward with
/// zero extra host sync.
pub(crate) fn mtp_commit_cuda_with_flush(
    mtp: &MtpLayer,
    model: &Model,
    d: &mut Draft,
    h: &MxArray,
    ids: &[u32],
    flush: bool,
) -> Result<()> {
    let at = d.len();
    let m = ids.len();
    if m == 0 || h.shape_at(0)? as usize != m {
        return Err(bad(format!(
            "MTP commit of {m} ids with {:?} hiddens",
            h.shape()?.as_ref()
        )));
    }
    if at + m > d.cap() {
        return Err(bad(format!(
            "MTP commit to {} past the {} rows a draft holds",
            at + m,
            d.cap()
        )));
    }
    let mtp_cache = mtp
        .cuda_cache()
        .ok_or_else(|| bad("MtpLayer has no MtpCudaCache"))?;
    let tp_mtp_commit = prof_start();
    let _graphs_guard = (m > 16).then(CudaGraphsGuard::disable_for_prefill);

    let prefill_mode = nvfp4_prefill_mode();
    let w = mtp.weights();
    let nextn = mtp.nextn_weights()?;
    let iw = mtp.indexer_weights_ref()?;
    let eps = mtp.eps();
    let inv_freq = &mtp_cache.inv_freq;

    let tok = MxArray::from_uint32(ids, &[m as i64])?;
    let e_bf16 = model
        .weights()
        .token_embd()
        .rows(&tok)?
        .astype(DType::BFloat16)?;
    let h_bf16 = ensure_bf16(h)?;
    let eh_cat = hy4::mtp_eh_norm_concat(&e_bf16, &h_bf16, &nextn.enorm, &nextn.hnorm, eps)?
        .ok_or_else(|| bad("hy4::mtp_eh_norm_concat declined"))?;
    let x = cuda_linear_bf16(&eh_cat, &nextn.eh_proj, None, prefill_mode)?;
    let x_norm = rms_norm_bf16(&x, &w.attn.norm, eps)?;
    let kv_a = cuda_linear_bf16(&x_norm, &w.attn.kv_a, None, prefill_mode)?;

    let _scope = flush.then(DeferScope::enter);
    let (kv, index) = d.kv_and_index_mut();
    let prev_kv = kv.cuda_buf_clone();
    let prev_pooled = index.cuda_pooled_buf_clone();
    let prev_tail = index.cuda_tail_rows_clone();
    let prev_held = index.held();
    let fast_rb = (index.is_empty() && m <= index.tail().target_ring_cap())
        || (index.tail().ring_cap() > 0 && m <= index.tail().rollback_slack());
    let had_tail_buf = index.tail().rows().is_some();
    let mark = if fast_rb {
        None
    } else {
        let m_snap = index.mark();
        if let Some(rows) = m_snap.tail().rows() {
            MxArray::eval_arrays_with_context(&[rows], "hy_v4_flash MTP commit mark")?;
        }
        Some(m_snap)
    };

    let commit_res: Result<()> = (|| {
        let kv_buf = kv.cuda_ensure_buf(at + m)?;
        let (_, out_kv_buf) = hy4::mla_prep_kv_store(
            None,
            &kv_a,
            None,
            &w.attn.kv_a_norm,
            inv_freq,
            kv_buf,
            at as i64,
            (at + 1) as i64,
            0,
            eps,
        )?
        .ok_or_else(|| bad("MTP commit: hy4::mla_prep_kv_store declined"))?;

        let tok_k_raw = cuda_linear_bf16(&x_norm, &iw.attn_k, None, prefill_mode)?;
        let tok_k = rms_norm(&tok_k_raw.astype(DType::Float32)?, &iw.k_norm, eps)?;
        mtp.indexer().cuda_update(iw, &tok_k, inv_freq, at, index)?;
        kv.cuda_commit(out_kv_buf, at + m)?;
        if flush || super::stats::profiling() {
            // Evaluate deferred KV + indexer writes atomically before dropping the
            // pre-commit handles so any GPU failure rolls back without poisoning `d`.
            super::stats::flush("hy_v4_flash MTP commit")?;
        }
        Ok(())
    })();

    if let Err(e) = commit_res {
        crate::array::synchronize();
        kv.cuda_restore_buf_opt(prev_kv);
        kv.truncate(at)?;
        index.cuda_restore_pooled_buf(prev_pooled);
        index.cuda_restore_tail_rows(prev_tail);
        if let Some(m_snap) = mark {
            index.restore(m_snap)?;
        } else if !had_tail_buf {
            index.reset_empty()?;
        } else {
            index.rollback_push(at, prev_held)?;
        }
        return Err(e);
    }
    prof_end(Phase::Mtp, tp_mtp_commit, &[])?;
    Ok(())
}

/// CUDA fast path for `MtpLayer::draft_with_flush`: single-row MTP draft step
/// with attention RoPE at `at + 1`, KV/indexer slots at `at`, and transactional
/// rollback on error.
pub(crate) fn mtp_draft_cuda(
    mtp: &MtpLayer,
    model: &Model,
    d: &mut Draft,
    h: &MxArray,
    emb: &MxArray,
    flush: bool,
    persist_state: bool,
) -> Result<(MxArray, MxArray)> {
    let tp_mtp_draft = prof_start();
    let _scope = flush.then(DeferScope::enter);
    let at = d.len();
    if at >= d.cap() {
        return Err(bad(format!(
            "MTP draft row {at} past the {} rows held",
            d.cap()
        )));
    }
    let mtp_cache = mtp
        .cuda_cache()
        .ok_or_else(|| bad("MtpLayer has no MtpCudaCache"))?;

    let prefill_mode = nvfp4_prefill_mode();
    let cfg = model.cfg();
    let w = mtp.weights();
    let nextn = mtp.nextn_weights()?;
    let iw = mtp.indexer_weights_ref()?;
    let eps = mtp.eps();
    let dim = mtp.dim();
    let g = mtp.mla().geometry();
    let ix_g = mtp.indexer().geometry();
    let inv_freq = &mtp_cache.inv_freq;

    let e_bf16 = ensure_bf16(&emb.reshape(&[1, dim])?)?;
    let h_bf16 = ensure_bf16(&h.reshape(&[1, dim])?)?;
    let eh_cat = hy4::mtp_eh_norm_concat(&e_bf16, &h_bf16, &nextn.enorm, &nextn.hnorm, eps)?
        .ok_or_else(|| bad("hy4::mtp_eh_norm_concat declined"))?;
    let x = cuda_linear_bf16(&eh_cat, &nextn.eh_proj, None, prefill_mode)?;
    let x_norm = rms_norm_bf16(&x, &w.attn.norm, eps)?;

    let n_heads = g.n_heads;
    let h_i64 = n_heads as i64;
    let q_lora = w.attn.q_a_norm.shape_at(0)?;
    let kv_lora = g.kv_lora;
    let nope = g.nope;
    let rope = g.rope_dim;
    let kv_row = g.row() as i64;
    let head_v = g.head_v;
    let gate_dim = (n_heads * head_v) as i64;

    let (q_a, kv_a, gate) = project_q_kv_gate(
        &x_norm,
        &w.attn,
        q_lora,
        kv_row,
        gate_dim,
        None,
        None,
        None,
        None,
        None,
        prefill_mode,
    )?;

    let (kv, index) = d.kv_and_index_mut();
    let prev_kv = kv.cuda_buf_clone();
    let prev_pooled = index.cuda_pooled_buf_clone();
    let prev_tail = index.cuda_tail_rows_clone();
    let kv_buf = kv.cuda_ensure_buf(at + 1)?;
    let (qr_opt, out_kv_buf) = hy4::mla_prep_kv_store(
        Some(&q_a),
        &kv_a,
        Some(&w.attn.q_a_norm),
        &w.attn.kv_a_norm,
        inv_freq,
        kv_buf,
        at as i64,
        (at + 1) as i64,
        0,
        eps,
    )?
    .ok_or_else(|| bad("MTP draft: hy4::mla_prep_kv_store declined"))?;
    let qr = qr_opt.ok_or_else(|| bad("MTP draft: mla_prep_kv_store missing qr"))?;

    let q_raw = cuda_linear_bf16(&qr, &w.attn.q_b, None, prefill_mode)?.reshape(&[
        1,
        h_i64,
        (nope + rope) as i64,
    ])?;
    let q_nope = q_raw.slice(&[0, 0, 0], &[1, h_i64, nope as i64])?;
    let q_abs = cuda_grouped_q8(&q_nope, &w.attn.k_b, None, n_heads, nope, kv_lora)?;
    let q = hy4::rope_norm_q(&q_abs, &q_raw, inv_freq, (at + 1) as i64, nope as i32)?
        .ok_or_else(|| bad("MTP draft: hy4::rope_norm_q declined"))?;

    let mutates_index = persist_state || !ix_g.select_tail;
    let prev_held = index.held();
    let fast_rb = mutates_index && index.tail().rollback_slack() >= 1;
    let mark = (mutates_index && !fast_rb).then(|| index.mark());

    let q_idx_raw = cuda_linear_bf16(&qr, &iw.q_b, None, prefill_mode)?.reshape(&[
        1,
        ix_g.n_heads as i64,
        ix_g.head_dim as i64,
    ])?;
    let w_proj = match mtp_cache.indexer_proj_w.as_ref() {
        Some(cached) => cuda_linear_bf16_cached_array(&x_norm, cached)?,
        None => cuda_linear_bf16(&x_norm, &iw.proj, None, prefill_mode)?,
    };

    // MTP indexer pooled keys use slot-based RoPE `g * kpool`, so `q_idx` uses
    // `rope_start = at` (matching `mtp.rs` `Call { start: at }` and `kpool_compress`).
    let idx = if mutates_index {
        let tok_k_raw = cuda_linear_bf16(&x_norm, &iw.attn_k, None, prefill_mode)?;
        let tok_k = rms_norm(&tok_k_raw.astype(DType::Float32)?, &iw.k_norm, eps)?;
        mtp.indexer()
            .cuda_select(iw, &tok_k, &q_idx_raw, &w_proj, inv_freq, at, at, index)?
    } else {
        mtp.indexer()
            .cuda_select_readonly(&q_idx_raw, &w_proj, inv_freq, at, at, index)?
    };

    if let Err(e) = kv.cuda_commit(out_kv_buf, at + 1) {
        kv.cuda_restore_buf_opt(prev_kv);
        index.cuda_restore_pooled_buf(prev_pooled);
        index.cuda_restore_tail_rows(prev_tail);
        if let Some(m) = mark {
            index.restore(m)?;
        } else if fast_rb {
            index.rollback_push(at, prev_held)?;
        }
        return Err(e);
    }

    let tail_res: Result<(MxArray, MxArray)> = (|| {
        let o_lat = hy4::mla_sdpa(
            &q,
            kv.rows()?,
            &w.attn.sinks,
            Some(&idx),
            at as i64,
            (at + 1) as i32,
            0,
            0,
            g.scale,
            kv_lora as i32,
        )?
        .ok_or_else(|| bad("MTP draft: hy4::mla_sdpa declined"))?;

        let o_gated = cuda_grouped_q8(&o_lat, &w.attn.v_b, Some(&gate), n_heads, kv_lora, head_v)?
            .reshape(&[1, gate_dim])?;
        let attn_raw = cuda_linear_bf16(&o_gated, &w.attn.output, None, prefill_mode)?;
        let attn_post = rms_norm_bf16(&attn_raw, &w.attn.post_norm, eps)?;
        let (x1, ffn_in) = fused_add_rms_norm_bf16(&x, &attn_post, &w.ffn_norm, eps)?;

        let swiglu_limit = cfg
            .layers
            .get(mtp.layer())
            .map_or(0.0, |lc| lc.swiglu_limit);
        let y_ffn = match (&w.ffn, mtp_cache.sym_q8_experts.as_ref()) {
            (Ffn::Moe(mw), Some(re)) => {
                let mut router_scratch = None;
                let router_w =
                    resolve_router_weight(mw, mtp_cache.router_w.as_ref(), &mut router_scratch)?;
                run_moe_cuda(
                    &ffn_in,
                    mw,
                    cfg,
                    swiglu_limit,
                    router_w,
                    &re.gate_w,
                    &re.gate_s,
                    &re.up_w,
                    &re.up_s,
                    &re.down_w,
                    &re.down_s,
                    MoeWeightMode::SymQ8,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    prefill_mode,
                )?
            }
            _ => {
                // Affine4 MTP experts (BF16 GGUF) or dense FFN: run resident LayerFfn.
                let (f_raw, _) = mtp.ffn().forward_normed_raw(
                    w,
                    &ffn_in.astype(DType::Float32)?,
                    RowKind::Decode,
                    false,
                    None,
                )?;
                f_raw.astype(DType::BFloat16)?
            }
        };

        let f_post = rms_norm_bf16(&y_ffn, &w.post_ffw_norm, eps)?;
        let (_, h_mtp_bf16) = fused_add_rms_norm_bf16(&x1, &f_post, &nextn.shared_head_norm, eps)?;

        let head = mtp.draft_head().unwrap_or_else(|| model.weights().output());
        let logits = cuda_lm_head_f32(&h_mtp_bf16, head)?;
        let h_mtp = h_mtp_bf16.astype(DType::Float32)?.reshape(&[1, 1, dim])?;
        if flush {
            super::stats::flush("hy_v4_flash MTP draft")?;
        }
        prof_end(Phase::Mtp, tp_mtp_draft, &[&h_mtp, &logits])?;
        Ok((h_mtp, logits))
    })();

    if tail_res.is_err() {
        crate::array::synchronize();
        kv.cuda_restore_buf_opt(prev_kv);
        kv.truncate(at)?;
        index.cuda_restore_pooled_buf(prev_pooled);
        index.cuda_restore_tail_rows(prev_tail);
        if let Some(m) = mark {
            index.restore(m)?;
        } else if fast_rb {
            index.rollback_push(at, prev_held)?;
        }
    }
    tail_res
}
