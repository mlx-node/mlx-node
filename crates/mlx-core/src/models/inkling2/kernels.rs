//! Safe Rust wrappers for Inkling-2-Mini Metal `Custom` primitives:
//! - `inkling2_paged_attention` (INT8 32-token KTM & BF16 vLLM, relative bias + natural-log LSE, varlen)
//! - `inkling2_paged_prefill_attention` (in-place G=3 D=128 MPP prefill with SWA skip, zero-bias `q8q` fast path, portable fallback)
//! - `inkling2_gate_sink` (`O * sigmoid(gate) * sigmoid(lse - sink)`)
//! - `inkling2_situ` (SiTU activation with optional per-expert `outer_scale`)
//! - `inkling2_loop_blend_rms_norm` (FP32 no-FMA loop blend + sublayer RMSNorm)
//! - `inkling2_sconv_k4` (static & dynamic global causal k=4 ShortConv returning `(y, history_out)`)

use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};

/// On-cache KV dtype wire code (matches `mlx_paged_ops.h::KvDtype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Inkling2KvDtype {
    Bf16 = 1,
    Int8 = 3,
}

/// Returns true when the G=3, D=128 Inkling2 in-place MPP prefill pipelines build on this GPU.
pub fn inkling2_paged_prefill_mpp_supported() -> bool {
    // SAFETY: nullary FFI query that catches internally.
    unsafe { mlx_sys::mlx_inkling2_paged_prefill_mpp_supported() }
}

/// Inkling-2-Mini varlen relative-bias + LSE paged attention (head_size = 128) with optional
/// on-the-fly `r_for_rel @ rel_bias_w^T` (`rel_bias_w = Some([rel_extent, 16] F32)` and
/// `rel_bias = Some(r_for_rel [T, H, 16] F32)`) and optional fused gate+sink epilogue.
///
/// When `want_lse == false`, skips LSE buffer allocation and DRAM write (returns `None`).
#[allow(clippy::too_many_arguments)]
pub fn inkling2_paged_attention_ex(
    q: &MxArray,
    k_pool: &MxArray,
    v_pool: &MxArray,
    block_table: &MxArray,
    seq_lens: &MxArray,
    cu_seqlens_q: &MxArray,
    k_scale: &MxArray,
    v_scale: &MxArray,
    rel_bias: Option<&MxArray>,
    rel_bias_w: Option<&MxArray>,
    gate_sink: Option<(&MxArray, &MxArray)>,
    want_lse: bool,
    scale: f32,
    sliding_window: i32,
    rel_extent: i32,
    block_size: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_size: i32,
    kv_dtype: Inkling2KvDtype,
    score_scales: Option<&MxArray>,
) -> Result<(MxArray, Option<MxArray>)> {
    let (gate_ptr, sink_ptr) = match gate_sink {
        Some((g, s)) => (g.as_raw_ptr(), s.as_raw_ptr()),
        None => (std::ptr::null_mut(), std::ptr::null_mut()),
    };
    let mut out_attn = std::ptr::null_mut();
    let mut out_lse = std::ptr::null_mut();
    let out_lse_ptr = if want_lse {
        &mut out_lse as *mut *mut mlx_sys::mlx_array
    } else {
        std::ptr::null_mut()
    };
    // SAFETY: all input pointers are live `MxArray` handles; optional pointers may be null.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_paged_attention_forward(
            q.as_raw_ptr(),
            k_pool.as_raw_ptr(),
            v_pool.as_raw_ptr(),
            block_table.as_raw_ptr(),
            seq_lens.as_raw_ptr(),
            cu_seqlens_q.as_raw_ptr(),
            k_scale.as_raw_ptr(),
            v_scale.as_raw_ptr(),
            rel_bias.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            rel_bias_w.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            gate_ptr,
            sink_ptr,
            scale,
            sliding_window,
            rel_extent,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_size,
            kv_dtype as u8,
            score_scales.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            &mut out_attn,
            out_lse_ptr,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_inkling2_paged_attention_forward failed (invalid contract or Metal unavailable)",
        ));
    }
    let attn = MxArray::from_handle(out_attn, "inkling2_paged_attention.out")?;
    let lse = if want_lse {
        Some(MxArray::from_handle(
            out_lse,
            "inkling2_paged_attention.lse",
        )?)
    } else {
        None
    };
    Ok((attn, lse))
}

/// Inkling-2-Mini varlen relative-bias + natural-log LSE paged attention (head_size = 128).
///
/// Supports INT8 32-token KTM (`k_pool [nb, H_kv, 32, 128]`, `v_pool [nb, H_kv, 128, 32]`,
/// `k_scale/v_scale [nb, H_kv, 32]`) and BF16 vLLM (`k_pool [nb, H_kv, 16, bs, 8]`,
/// `v_pool [nb, H_kv, 128, bs]`, `bs` in `{16, 32}`).
///
/// Returns `(out [total_q, num_q_heads, 128] bf16, lse [total_q, num_q_heads] f32)`.
#[allow(clippy::too_many_arguments)]
pub fn inkling2_paged_attention(
    q: &MxArray,
    k_pool: &MxArray,
    v_pool: &MxArray,
    block_table: &MxArray,
    seq_lens: &MxArray,
    cu_seqlens_q: &MxArray,
    k_scale: &MxArray,
    v_scale: &MxArray,
    rel_bias: Option<&MxArray>,
    scale: f32,
    sliding_window: i32,
    rel_extent: i32,
    block_size: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_size: i32,
    kv_dtype: Inkling2KvDtype,
    score_scales: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    let (attn, lse) = inkling2_paged_attention_ex(
        q,
        k_pool,
        v_pool,
        block_table,
        seq_lens,
        cu_seqlens_q,
        k_scale,
        v_scale,
        rel_bias,
        None,
        None,
        true,
        scale,
        sliding_window,
        rel_extent,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_size,
        kv_dtype,
        score_scales,
    )?;
    let lse = lse.ok_or_else(|| Error::from_reason("inkling2_paged_attention missing lse"))?;
    Ok((attn, lse))
}

/// Inkling-2-Mini in-place MPP paged prefill / verify attention (G=3, D=128) with optional
/// on-the-fly `r_for_rel @ rel_bias_w^T`, optional fused gate+sink epilogue, and optional LSE.
#[allow(clippy::too_many_arguments)]
pub fn inkling2_paged_prefill_attention_ex(
    q: &MxArray,
    k_pool: &MxArray,
    v_pool: &MxArray,
    k_scale: &MxArray,
    v_scale: &MxArray,
    block_table: &MxArray,
    workspace: &MxArray,
    rel_bias: Option<&MxArray>,
    rel_bias_w: Option<&MxArray>,
    gate_sink: Option<(&MxArray, &MxArray)>,
    want_lse: bool,
    committed_tokens: i32,
    splits: i32,
    max_splits: i32,
    block_size: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_size: i32,
    sliding_window: i32,
    rel_extent: i32,
    kv_dtype: Inkling2KvDtype,
    int8_queries: bool,
    scale: f32,
) -> Result<(MxArray, Option<MxArray>, MxArray)> {
    let (gate_ptr, sink_ptr) = match gate_sink {
        Some((g, s)) => (g.as_raw_ptr(), s.as_raw_ptr()),
        None => (std::ptr::null_mut(), std::ptr::null_mut()),
    };
    let mut out_attn = std::ptr::null_mut();
    let mut out_lse = std::ptr::null_mut();
    let mut out_ws = std::ptr::null_mut();
    let out_lse_ptr = if want_lse {
        &mut out_lse as *mut *mut mlx_sys::mlx_array
    } else {
        std::ptr::null_mut()
    };
    // SAFETY: all input pointers are live `MxArray` handles; optional pointers may be null.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_paged_prefill_attention_forward(
            q.as_raw_ptr(),
            k_pool.as_raw_ptr(),
            v_pool.as_raw_ptr(),
            k_scale.as_raw_ptr(),
            v_scale.as_raw_ptr(),
            block_table.as_raw_ptr(),
            workspace.as_raw_ptr(),
            rel_bias.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            rel_bias_w.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            gate_ptr,
            sink_ptr,
            committed_tokens,
            splits,
            max_splits,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_size,
            sliding_window,
            rel_extent,
            kv_dtype as u8,
            int8_queries,
            scale,
            &mut out_attn,
            out_lse_ptr,
            &mut out_ws,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_inkling2_paged_prefill_attention_forward failed",
        ));
    }
    let attn = MxArray::from_handle(out_attn, "inkling2_paged_prefill.out")?;
    let lse = if want_lse {
        Some(MxArray::from_handle(out_lse, "inkling2_paged_prefill.lse")?)
    } else {
        None
    };
    let ws = MxArray::from_handle(out_ws, "inkling2_paged_prefill.workspace")?;
    Ok((attn, lse, ws))
}

/// Inkling-2-Mini in-place MPP paged prefill / verify attention (G=3, D=128) with SWA page skip,
/// relative bias + zero-bias fast path (including `q8q` INT8 queries on Apple10/M5+),
/// and portable non-MPP fallback.
///
/// Returns `(out [rows, num_q_heads, 128] bf16, lse [rows, num_q_heads] f32, workspace')`.
#[allow(clippy::too_many_arguments)]
pub fn inkling2_paged_prefill_attention(
    q: &MxArray,
    k_pool: &MxArray,
    v_pool: &MxArray,
    k_scale: &MxArray,
    v_scale: &MxArray,
    block_table: &MxArray,
    workspace: &MxArray,
    rel_bias: Option<&MxArray>,
    committed_tokens: i32,
    splits: i32,
    max_splits: i32,
    block_size: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_size: i32,
    sliding_window: i32,
    rel_extent: i32,
    kv_dtype: Inkling2KvDtype,
    int8_queries: bool,
    scale: f32,
) -> Result<(MxArray, MxArray, MxArray)> {
    let (attn, lse, ws) = inkling2_paged_prefill_attention_ex(
        q,
        k_pool,
        v_pool,
        k_scale,
        v_scale,
        block_table,
        workspace,
        rel_bias,
        None,
        None,
        true,
        committed_tokens,
        splits,
        max_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_size,
        sliding_window,
        rel_extent,
        kv_dtype,
        int8_queries,
        scale,
    )?;
    let lse = lse.ok_or_else(|| Error::from_reason("inkling2_paged_prefill missing lse"))?;
    Ok((attn, lse, ws))
}

/// Fused Inkling2 gate + sink epilogue: `Y = O * sigmoid(gate) * sigmoid(lse - sink)`.
pub fn inkling2_gate_sink(
    o: &MxArray,
    gate: &MxArray,
    lse: &MxArray,
    sink: &MxArray,
) -> Result<MxArray> {
    // SAFETY: all input pointers are live `MxArray` handles.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_gate_sink(
            o.as_raw_ptr(),
            gate.as_raw_ptr(),
            lse.as_raw_ptr(),
            sink.as_raw_ptr(),
        )
    };
    MxArray::from_handle(ptr, "inkling2_gate_sink")
}

/// Fused Inkling2 SiTU activation: `softcap(gate * s, gate_cap) * sigmoid(gate * s) * softcap(up * s, up_cap)`
/// with optional per-expert `outer_scale[expert_indices]`.
pub fn inkling2_situ(
    gate_up: &MxArray,
    outer_scale: Option<(&MxArray, &MxArray)>,
    gate_softcap: f32,
    up_softcap: f32,
    interleaved: bool,
) -> Result<MxArray> {
    let (scale_ptr, idx_ptr) = match outer_scale {
        Some((s, i)) => (s.as_raw_ptr(), i.as_raw_ptr()),
        None => (std::ptr::null_mut(), std::ptr::null_mut()),
    };
    // SAFETY: input handles are live; optional pointers are paired null or non-null.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_situ(
            gate_up.as_raw_ptr(),
            scale_ptr,
            idx_ptr,
            gate_softcap,
            up_softcap,
            interleaved,
        )
    };
    MxArray::from_handle(ptr, "inkling2_situ")
}

/// Fused Inkling2 FP32 no-FMA loop blend `(1 - tanh(forget)) * dest + tanh(gain) * msg`
/// followed by sublayer RMSNorm -> `out_dtype` (`BFloat16` or `Float32`).
pub fn inkling2_loop_blend_rms_norm(
    dest: &MxArray,
    msg: &MxArray,
    gain_raw: &MxArray,
    forget_raw: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    out_dtype: DType,
) -> Result<MxArray> {
    let output_f32 = match out_dtype {
        DType::Float32 => true,
        DType::BFloat16 => false,
        other => {
            return Err(Error::from_reason(format!(
                "inkling2_loop_blend_rms_norm: unsupported out_dtype {other:?}"
            )));
        }
    };
    // SAFETY: all input pointers are live `MxArray` handles.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_loop_blend_rms_norm(
            dest.as_raw_ptr(),
            msg.as_raw_ptr(),
            gain_raw.as_raw_ptr(),
            forget_raw.as_raw_ptr(),
            norm_weight.as_raw_ptr(),
            eps,
            output_f32,
        )
    };
    MxArray::from_handle(ptr, "inkling2_loop_blend_rms_norm")
}

/// Fused Inkling2 causal k=4 ShortConv (static when `dyn_taps` is `None`, dynamic global when `Some`).
/// Returns `(y [T, C], updated_history [3, C])`.
pub fn inkling2_sconv_k4(
    x: &MxArray,
    weight: &MxArray,
    history_in: &MxArray,
    dyn_taps: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    let mut out_y = std::ptr::null_mut();
    let mut out_hist = std::ptr::null_mut();
    // SAFETY: all input pointers are live; `dyn_taps` may be null.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_sconv_k4(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            history_in.as_raw_ptr(),
            dyn_taps.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            &mut out_y,
            &mut out_hist,
        )
    };
    if !ok {
        return Err(Error::from_reason("mlx_inkling2_sconv_k4 failed"));
    }
    let y = MxArray::from_handle(out_y, "inkling2_sconv_k4.y")?;
    let hist = MxArray::from_handle(out_hist, "inkling2_sconv_k4.history")?;
    Ok((y, hist))
}

/// Fused Inkling2 strided Q & R L2-clip + optional global `tau` -> `(Q [T, H, D], R [T, H, d_rel] F32)`.
#[allow(clippy::too_many_arguments)]
pub fn inkling2_qr_clip(
    q_in: &MxArray,
    r_in: &MxArray,
    q_clip: &MxArray,
    rel_clip: &MxArray,
    tau: Option<&MxArray>,
    num_heads: usize,
    head_dim: usize,
    d_rel: usize,
    qk_thresh: f32,
    rel_thresh: f32,
    eps: f32,
    q_out_dtype: DType,
    r_round_bf16: bool,
) -> Result<(MxArray, MxArray)> {
    let output_f32 = match q_out_dtype {
        DType::Float32 => true,
        DType::BFloat16 => false,
        other => {
            return Err(Error::from_reason(format!(
                "inkling2_qr_clip: unsupported q_out_dtype {other:?}"
            )));
        }
    };
    let mut out_q = std::ptr::null_mut();
    let mut out_r = std::ptr::null_mut();
    // SAFETY: all input handles are live; `tau` may be null.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_qr_clip(
            q_in.as_raw_ptr(),
            r_in.as_raw_ptr(),
            q_clip.as_raw_ptr(),
            rel_clip.as_raw_ptr(),
            tau.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            num_heads as u32,
            head_dim as u32,
            d_rel as u32,
            qk_thresh,
            rel_thresh,
            eps,
            output_f32,
            r_round_bf16,
            &mut out_q,
            &mut out_r,
        )
    };
    if !ok {
        return Err(Error::from_reason("mlx_inkling2_qr_clip failed"));
    }
    let q = MxArray::from_handle(out_q, "inkling2_qr_clip.q")?;
    let r = MxArray::from_handle(out_r, "inkling2_qr_clip.r")?;
    Ok((q, r))
}

/// Fused Inkling2 single/ragged static/dynamic k=4 K+V ShortConv + K L2-clip + V RMSNorm.
///
/// Supports 1..=4 unstacked `[3, kv_dim]` histories per stream (zero `stack` overhead!) or
/// packed `[N, 3, kv_dim]` histories when `packed_hist == true`.
/// Returns `(k_cur [T, H_kv, head_dim], v_cur [T, H_kv, head_dim], k_hist_out, v_hist_out)`.
#[allow(clippy::too_many_arguments)]
pub fn inkling2_sconv_k4_kv_prep(
    k_in: &MxArray,
    v_in: &MxArray,
    k_weight: &MxArray,
    v_weight: &MxArray,
    dyn_k: Option<&MxArray>,
    dyn_v: Option<&MxArray>,
    cu_seqlens: Option<&MxArray>,
    k_hists: &[&MxArray],
    v_hists: &[&MxArray],
    num_seqs: usize,
    packed_hist: bool,
    num_kv_heads: usize,
    head_dim: usize,
    qk_thresh: f32,
    v_eps: f32,
    l2_eps: f32,
) -> Result<(MxArray, MxArray, MxArray, MxArray)> {
    let expected_h = if packed_hist { 1 } else { num_seqs };
    if num_seqs == 0 || k_hists.len() != expected_h || v_hists.len() != expected_h {
        return Err(Error::from_reason(
            "inkling2_sconv_k4_kv_prep: history slice count mismatch",
        ));
    }
    let k_ptrs: Vec<*mut mlx_sys::mlx_array> = k_hists.iter().map(|a| a.as_raw_ptr()).collect();
    let v_ptrs: Vec<*mut mlx_sys::mlx_array> = v_hists.iter().map(|a| a.as_raw_ptr()).collect();
    let mut out_k = std::ptr::null_mut();
    let mut out_v = std::ptr::null_mut();
    let mut out_kh = std::ptr::null_mut();
    let mut out_vh = std::ptr::null_mut();
    // SAFETY: all input pointers live through the FFI call.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_sconv_k4_kv_prep(
            k_in.as_raw_ptr(),
            v_in.as_raw_ptr(),
            k_weight.as_raw_ptr(),
            v_weight.as_raw_ptr(),
            dyn_k.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            dyn_v.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            cu_seqlens.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            k_ptrs.as_ptr(),
            v_ptrs.as_ptr(),
            num_seqs as u32,
            packed_hist,
            num_kv_heads as u32,
            head_dim as u32,
            qk_thresh,
            v_eps,
            l2_eps,
            &mut out_k,
            &mut out_v,
            &mut out_kh,
            &mut out_vh,
        )
    };
    if !ok {
        return Err(Error::from_reason("mlx_inkling2_sconv_k4_kv_prep failed"));
    }
    Ok((
        MxArray::from_handle(out_k, "inkling2_sconv_k4_kv_prep.k")?,
        MxArray::from_handle(out_v, "inkling2_sconv_k4_kv_prep.v")?,
        MxArray::from_handle(out_kh, "inkling2_sconv_k4_kv_prep.k_hist")?,
        MxArray::from_handle(out_vh, "inkling2_sconv_k4_kv_prep.v_hist")?,
    ))
}

/// Fused Inkling2 single/ragged static/dynamic k=4 K+V ShortConv + K L2-clip + V RMSNorm +
/// in-place PagedKVWrite (INT8-KTM & BF16), writing directly into the paged Metal KV pools
/// with ZERO intermediate `[T, H_kv, head_dim]` DRAM roundtrip.
///
/// `kv_mode`: `0` = INT8-KTM, `1` = BF16 vLLM (x=8), `2` = BF16 values_token_major.
/// Returns `(k_pool', v_pool', k_scale', v_scale', k_hist_out, v_hist_out)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn inkling2_sconv_k4_kv_prep_paged(
    k_in: &MxArray,
    v_in: &MxArray,
    k_weight: &MxArray,
    v_weight: &MxArray,
    dyn_k: Option<&MxArray>,
    dyn_v: Option<&MxArray>,
    cu_seqlens: Option<&MxArray>,
    k_hists: &[&MxArray],
    v_hists: &[&MxArray],
    k_pool: &MxArray,
    v_pool: &MxArray,
    k_scale: &MxArray,
    v_scale: &MxArray,
    slot_mapping: &MxArray,
    num_seqs: usize,
    packed_hist: bool,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    kv_mode: u32,
    qk_thresh: f32,
    v_eps: f32,
    l2_eps: f32,
) -> Result<(MxArray, MxArray, MxArray, MxArray, MxArray, MxArray)> {
    let expected_h = if packed_hist { 1 } else { num_seqs };
    if num_seqs == 0 || k_hists.len() != expected_h || v_hists.len() != expected_h {
        return Err(Error::from_reason(
            "inkling2_sconv_k4_kv_prep_paged: history slice count mismatch",
        ));
    }
    let k_ptrs: Vec<*mut mlx_sys::mlx_array> = k_hists.iter().map(|a| a.as_raw_ptr()).collect();
    let v_ptrs: Vec<*mut mlx_sys::mlx_array> = v_hists.iter().map(|a| a.as_raw_ptr()).collect();
    let mut out_kp = std::ptr::null_mut();
    let mut out_vp = std::ptr::null_mut();
    let mut out_ks = std::ptr::null_mut();
    let mut out_vs = std::ptr::null_mut();
    let mut out_kh = std::ptr::null_mut();
    let mut out_vh = std::ptr::null_mut();
    // SAFETY: all input pointers live through the FFI call.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_sconv_k4_kv_prep_paged(
            k_in.as_raw_ptr(),
            v_in.as_raw_ptr(),
            k_weight.as_raw_ptr(),
            v_weight.as_raw_ptr(),
            dyn_k.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            dyn_v.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            cu_seqlens.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            k_ptrs.as_ptr(),
            v_ptrs.as_ptr(),
            k_pool.as_raw_ptr(),
            v_pool.as_raw_ptr(),
            k_scale.as_raw_ptr(),
            v_scale.as_raw_ptr(),
            slot_mapping.as_raw_ptr(),
            num_seqs as u32,
            packed_hist,
            num_kv_heads as u32,
            head_dim as u32,
            block_size as u32,
            kv_mode,
            qk_thresh,
            v_eps,
            l2_eps,
            &mut out_kp,
            &mut out_vp,
            &mut out_ks,
            &mut out_vs,
            &mut out_kh,
            &mut out_vh,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_inkling2_sconv_k4_kv_prep_paged failed",
        ));
    }
    Ok((
        MxArray::from_handle(out_kp, "inkling2_sconv_k4_kv_prep_paged.k_pool")?,
        MxArray::from_handle(out_vp, "inkling2_sconv_k4_kv_prep_paged.v_pool")?,
        MxArray::from_handle(out_ks, "inkling2_sconv_k4_kv_prep_paged.k_scale")?,
        MxArray::from_handle(out_vs, "inkling2_sconv_k4_kv_prep_paged.v_scale")?,
        MxArray::from_handle(out_kh, "inkling2_sconv_k4_kv_prep_paged.k_hist")?,
        MxArray::from_handle(out_vh, "inkling2_sconv_k4_kv_prep_paged.v_hist")?,
    ))
}

/// Fused Inkling2 Block AttnRes over `1..=8` source `[T, D]` tensors (+ optional `addend` on last
/// source) with `queries [Q, D]` FP32 and optional fused sublayer RMSNorm (`norm_weight`).
#[allow(clippy::too_many_arguments)]
pub fn inkling2_block_attn_res(
    sources: &[&MxArray],
    addend: Option<&MxArray>,
    queries: &MxArray,
    norm_weight: Option<&MxArray>,
    attn_res_eps: f32,
    norm_eps: f32,
    squeeze_q1: bool,
    out_dtype: DType,
) -> Result<MxArray> {
    if sources.is_empty() || sources.len() > 8 {
        return Err(Error::from_reason(format!(
            "inkling2_block_attn_res: expected 1..=8 sources, got {}",
            sources.len()
        )));
    }
    let output_f32 = match out_dtype {
        DType::Float32 => true,
        DType::BFloat16 => false,
        other => {
            return Err(Error::from_reason(format!(
                "inkling2_block_attn_res: unsupported out_dtype {other:?}"
            )));
        }
    };
    let mut ptrs = [std::ptr::null_mut(); 8];
    for (i, s) in sources.iter().enumerate() {
        ptrs[i] = s.as_raw_ptr();
    }
    // SAFETY: `ptrs[..sources.len()]` and optional pointers are valid live handles.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_block_attn_res(
            ptrs.as_ptr(),
            sources.len() as u32,
            addend.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            queries.as_raw_ptr(),
            norm_weight.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            attn_res_eps,
            norm_eps,
            squeeze_q1,
            output_f32,
        )
    };
    MxArray::from_handle(ptr, "inkling2_block_attn_res")
}

/// Fused Inkling2 `[rms_norm(dest, eps), rms_norm(msg, eps)]` -> `[T, 2*D]` or `[S, T, 2*D]`.
pub fn inkling2_pair_rms_norm(
    dest: &MxArray,
    msg: &MxArray,
    eps: f32,
    out_dtype: DType,
) -> Result<MxArray> {
    let output_f32 = match out_dtype {
        DType::Float32 => true,
        DType::BFloat16 => false,
        other => {
            return Err(Error::from_reason(format!(
                "inkling2_pair_rms_norm: unsupported out_dtype {other:?} (expected Float32 or BFloat16)"
            )));
        }
    };
    // SAFETY: both handles are live.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_pair_rms_norm(dest.as_raw_ptr(), msg.as_raw_ptr(), eps, output_f32)
    };
    MxArray::from_handle(ptr, "inkling2_pair_rms_norm")
}

/// Coalesced exact-FP32 MoE router projection `x [T, D] (BF16/F32) @ weight^T [D, E] (F32) -> [T, E] F32`.
/// Avoids MLX's slow M=8/32 64x64-padded FP32 `steel_matmul` on small-M decode/verify!
pub fn inkling2_moe_router_proj(x: &MxArray, weight_f32: &MxArray) -> Result<MxArray> {
    // SAFETY: both handles are live.
    let ptr =
        unsafe { mlx_sys::mlx_inkling2_moe_router_proj(x.as_raw_ptr(), weight_f32.as_raw_ptr()) };
    MxArray::from_handle(ptr, "inkling2_moe_router_proj")
}

/// Fused pair RMSNorm + coalesced exact-FP32 LoopController down projection:
/// `h[s, t, r] = [rms_norm(dest[s, t]), rms_norm(msg[s, t])] @ down_w[s, r]^T`.
/// Supports 2D single-stream (`S=1`) and 3D multi-stream (`S=3`, with `dest` broadcast or 3D).
pub fn inkling2_loop_ctrl_down(
    dest: &MxArray,
    msg: &MxArray,
    down_w: &MxArray,
    eps: f32,
) -> Result<MxArray> {
    // SAFETY: all input handles are live.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_loop_ctrl_down(
            dest.as_raw_ptr(),
            msg.as_raw_ptr(),
            down_w.as_raw_ptr(),
            eps,
        )
    };
    MxArray::from_handle(ptr, "inkling2_loop_ctrl_down")
}

/// Fused coalesced exact-FP32 LoopController up projection (`h @ gain_forget_w`)
/// + no-FMA FP32 blend `(1 - tanh(forget)) * dest + tanh(gain) * msg`
/// + sublayer RMSNorm -> `out_dtype` (`BFloat16` or `Float32`).
#[allow(clippy::too_many_arguments)]
pub fn inkling2_loop_ctrl_up_blend_norm(
    h: &MxArray,
    gain_forget_w: &MxArray,
    dest: &MxArray,
    msg: &MxArray,
    norm_weight: &MxArray,
    eps: f32,
    out_dtype: DType,
) -> Result<MxArray> {
    let output_f32 = match out_dtype {
        DType::Float32 => true,
        DType::BFloat16 => false,
        other => {
            return Err(Error::from_reason(format!(
                "inkling2_loop_ctrl_up_blend_norm: unsupported out_dtype {other:?}"
            )));
        }
    };
    // SAFETY: all input handles are live.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_loop_ctrl_up_blend_norm(
            h.as_raw_ptr(),
            gain_forget_w.as_raw_ptr(),
            dest.as_raw_ptr(),
            msg.as_raw_ptr(),
            norm_weight.as_raw_ptr(),
            eps,
            output_f32,
        )
    };
    MxArray::from_handle(ptr, "inkling2_loop_ctrl_up_blend_norm")
}

/// Fused Inkling2 multi-sequence ragged causal k=4 ShortConv with optional FP32 residual epilogue
/// (`res_a` + optional `res_b` + `conv_out`).
#[allow(clippy::too_many_arguments)]
pub fn inkling2_sconv_k4_ragged(
    x: &MxArray,
    weight: &MxArray,
    dyn_taps: Option<&MxArray>,
    cu_seqlens: Option<&MxArray>,
    res_a: Option<&MxArray>,
    res_b: Option<&MxArray>,
    hists: &[&MxArray],
    num_seqs: usize,
    packed_hist: bool,
) -> Result<(MxArray, MxArray)> {
    let expected_h = if packed_hist { 1 } else { num_seqs };
    if num_seqs == 0 || hists.len() != expected_h {
        return Err(Error::from_reason(
            "inkling2_sconv_k4_ragged: history slice count mismatch",
        ));
    }
    let num_seqs_u32 = u32::try_from(num_seqs)
        .map_err(|_| Error::from_reason("inkling2_sconv_k4_ragged: num_seqs exceeds u32::MAX"))?;
    let h_ptrs: Vec<*mut mlx_sys::mlx_array> = hists.iter().map(|a| a.as_raw_ptr()).collect();
    let mut out_y = std::ptr::null_mut();
    let mut out_hist = std::ptr::null_mut();
    // SAFETY: all input pointers live through the FFI call.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_sconv_k4_ragged(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            dyn_taps.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            cu_seqlens.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            res_a.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            res_b.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            h_ptrs.as_ptr(),
            num_seqs_u32,
            packed_hist,
            &mut out_y,
            &mut out_hist,
        )
    };
    if !ok {
        return Err(Error::from_reason("mlx_inkling2_sconv_k4_ragged failed"));
    }
    let y = MxArray::from_handle(out_y, "inkling2_sconv_k4_ragged.y")?;
    let h = MxArray::from_handle(out_hist, "inkling2_sconv_k4_ragged.history")?;
    Ok((y, h))
}

/// Fused Inkling2 MoE router: `sigmoid(logits) + bias` -> exact top-K (`int32`) with smaller-index
/// tie-break -> normalized weights `* route_scale` (with optional `down_scale` folded in!).
pub fn inkling2_moe_route(
    logits: &MxArray,
    bias: Option<&MxArray>,
    down_scale: Option<&MxArray>,
    top_k: usize,
    route_scale: f32,
) -> Result<(MxArray, MxArray)> {
    let mut out_idx = std::ptr::null_mut();
    let mut out_w = std::ptr::null_mut();
    // SAFETY: all input pointers are live; optional pointers may be null.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_moe_route(
            logits.as_raw_ptr(),
            bias.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            down_scale.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            top_k as u32,
            route_scale,
            &mut out_idx,
            &mut out_w,
        )
    };
    if !ok {
        return Err(Error::from_reason("mlx_inkling2_moe_route failed"));
    }
    let indices = MxArray::from_handle(out_idx, "inkling2_moe_route.indices")?;
    let weights = MxArray::from_handle(out_w, "inkling2_moe_route.weights")?;
    Ok((indices, weights))
}

/// Fused Inkling2 MoE combine: optional `inv_order` unsort + FP32 weighted sum -> `[T, D]`.
pub fn inkling2_moe_combine(
    down_out: &MxArray,
    weights: &MxArray,
    inv_order: Option<&MxArray>,
    out_dtype: DType,
) -> Result<MxArray> {
    let output_f32 = match out_dtype {
        DType::Float32 => true,
        DType::BFloat16 => false,
        other => {
            return Err(Error::from_reason(format!(
                "inkling2_moe_combine: unsupported out_dtype {other:?}"
            )));
        }
    };
    // SAFETY: all input pointers are live; `inv_order` may be null.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_moe_combine(
            down_out.as_raw_ptr(),
            weights.as_raw_ptr(),
            inv_order.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            output_f32,
        )
    };
    MxArray::from_handle(ptr, "inkling2_moe_combine")
}

/// Fused Inkling2 vocab slice `[:out_cols]` + optional bias + `cap * tanh(x / cap)` -> `[T, out_cols]` F32.
pub fn inkling2_lm_head_softcap(
    logits: &MxArray,
    bias: Option<&MxArray>,
    out_cols: usize,
    softcap: f32,
    bias_round_bf16: bool,
) -> Result<MxArray> {
    // SAFETY: all input pointers are live; `bias` may be null.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_lm_head_softcap(
            logits.as_raw_ptr(),
            bias.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            out_cols as u32,
            softcap,
            bias_round_bf16,
        )
    };
    MxArray::from_handle(ptr, "inkling2_lm_head_softcap")
}

/// Fused Inkling2 vocab slice `[:out_cols]` + optional bias + `cap * tanh(x / cap)` + 2-stage
/// GPU argmax -> `[T]` Int32 (zero `[T, V]` logits DRAM write!).
pub fn inkling2_lm_head_softcap_argmax(
    logits: &MxArray,
    bias: Option<&MxArray>,
    out_cols: usize,
    softcap: f32,
    bias_round_bf16: bool,
) -> Result<MxArray> {
    // SAFETY: all input pointers are live; `bias` may be null.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_lm_head_softcap_argmax(
            logits.as_raw_ptr(),
            bias.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            out_cols as u32,
            softcap,
            bias_round_bf16,
        )
    };
    MxArray::from_handle(ptr, "inkling2_lm_head_softcap_argmax")
}

/// Fused Inkling2 DSpark Markov RNN gate (`[N, 3*R]` + optional `[N, 3*R]` ctx + `[N, R]` state)
/// -> `(next_state [N, R] BF16, latent [N, R] BF16)`.
pub fn inkling2_markov_gate(
    raw: &MxArray,
    ctx: Option<&MxArray>,
    state: &MxArray,
) -> Result<(MxArray, MxArray)> {
    let mut out_state = std::ptr::null_mut();
    let mut out_latent = std::ptr::null_mut();
    // SAFETY: all input pointers are live; `ctx` may be null; output pointers are valid stack locals.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_markov_gate(
            raw.as_raw_ptr(),
            ctx.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            state.as_raw_ptr(),
            &mut out_state,
            &mut out_latent,
        )
    };
    if !ok {
        return Err(Error::from_reason("mlx_inkling2_markov_gate failed"));
    }
    let next_state = MxArray::from_handle(out_state, "inkling2_markov_gate.next_state")?;
    let latent = MxArray::from_handle(out_latent, "inkling2_markov_gate.latent")?;
    Ok((next_state, latent))
}

/// Fused Inkling2 Q8_0 (group_size=32) folded Markov head + `fold_bias` + `base_logits` + softcap
/// -> `[B, out_cols]` FP32 (zero `[B, V]` intermediate bias DRAM write/read!).
#[allow(clippy::too_many_arguments)]
pub fn inkling2_markov_fold_softcap(
    latent: &MxArray,
    fold_w_q8: &MxArray,
    fold_scales: &MxArray,
    fold_biases: Option<&MxArray>,
    fold_bias: &MxArray,
    base_logits: &MxArray,
    out_cols: usize,
    softcap: f32,
) -> Result<MxArray> {
    // SAFETY: all input handles are live; `fold_biases` may be null.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_markov_fold_softcap(
            latent.as_raw_ptr(),
            fold_w_q8.as_raw_ptr(),
            fold_scales.as_raw_ptr(),
            fold_biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            fold_bias.as_raw_ptr(),
            base_logits.as_raw_ptr(),
            out_cols as u32,
            softcap,
        )
    };
    MxArray::from_handle(ptr, "inkling2_markov_fold_softcap")
}

/// Fused Inkling2 Q8_0 (group_size=32) folded Markov head + `fold_bias` + `base_logits` + softcap
/// + 2-stage GPU argmax -> `[B]` Int32 (zero `[B, V]` intermediate DRAM traffic!).
#[allow(clippy::too_many_arguments)]
pub fn inkling2_markov_fold_softcap_argmax(
    latent: &MxArray,
    fold_w_q8: &MxArray,
    fold_scales: &MxArray,
    fold_biases: Option<&MxArray>,
    fold_bias: &MxArray,
    base_logits: &MxArray,
    out_cols: usize,
    softcap: f32,
) -> Result<MxArray> {
    // SAFETY: all input handles are live; `fold_biases` may be null.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_markov_fold_softcap_argmax(
            latent.as_raw_ptr(),
            fold_w_q8.as_raw_ptr(),
            fold_scales.as_raw_ptr(),
            fold_biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            fold_bias.as_raw_ptr(),
            base_logits.as_raw_ptr(),
            out_cols as u32,
            softcap,
        )
    };
    MxArray::from_handle(ptr, "inkling2_markov_fold_softcap_argmax")
}

/// Coalesced single-pass BF16 linear `x [M, K] @ w [N, K]^T -> [M, N] BF16` (`1 <= M <= 32`,
/// `K % 4 == 0`), 100% 0-ULP batch-size invariant across every `M` in `1..=32`.
pub fn inkling2_bf16_linear(x: &MxArray, w: &MxArray) -> Result<MxArray> {
    // SAFETY: both input handles are live.
    let ptr = unsafe { mlx_sys::mlx_inkling2_bf16_linear(x.as_raw_ptr(), w.as_raw_ptr()) };
    MxArray::from_handle(ptr, "inkling2_bf16_linear")
}

/// Fused single-pass BF16 `lm_head` + softcap + 2-stage GPU argmax -> `[M]` Int32 (`1 <= M <= 32`,
/// `K % 4 == 0`, zero `[M, V]` logits DRAM write, 0-ULP batch-size invariant across `M in 1..=32`).
pub fn inkling2_bf16_lm_head_softcap_argmax(
    x: &MxArray,
    w: &MxArray,
    out_cols: usize,
    softcap: f32,
) -> Result<MxArray> {
    // SAFETY: both input handles are live.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_bf16_lm_head_softcap_argmax(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            out_cols as u32,
            softcap,
        )
    };
    MxArray::from_handle(ptr, "inkling2_bf16_lm_head_softcap_argmax")
}

/// Fused single-pass Symmetric Q8_0 (native Float16 scales, zero biases) `lm_head` + softcap +
/// 2-stage GPU argmax -> `[M]` Int32 (`1 <= M <= 32`, `K % 32 == 0`, 0-ULP batch-size invariant).
pub fn inkling2_lm_head_q8_softcap_argmax(
    x: &MxArray,
    w_q8: &MxArray,
    scales_f16: &MxArray,
    out_cols: usize,
    softcap: f32,
) -> Result<MxArray> {
    // SAFETY: all input handles are live.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_lm_head_q8_softcap_argmax(
            x.as_raw_ptr(),
            w_q8.as_raw_ptr(),
            scales_f16.as_raw_ptr(),
            out_cols as u32,
            softcap,
        )
    };
    MxArray::from_handle(ptr, "inkling2_lm_head_q8_softcap_argmax")
}

/// Single-pass Symmetric Q8_0 (native Float16 scales, zero biases) `lm_head` -> `[M, out_cols]`
/// BF16 (when `out_dtype == BFloat16`) OR FP32 (+ optional softcap when `softcap > 0`).
pub fn inkling2_lm_head_q8_bf16(
    x: &MxArray,
    w_q8: &MxArray,
    scales_f16: &MxArray,
    out_cols: usize,
    softcap: f32,
    out_dtype: DType,
) -> Result<MxArray> {
    let output_f32 = out_dtype == DType::Float32;
    // SAFETY: all input handles are live.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_lm_head_q8_bf16(
            x.as_raw_ptr(),
            w_q8.as_raw_ptr(),
            scales_f16.as_raw_ptr(),
            out_cols as u32,
            softcap,
            output_f32,
        )
    };
    MxArray::from_handle(ptr, "inkling2_lm_head_q8_bf16")
}

/// Single-pass Symmetric Q8_0 (native Float16 scales, zero biases) dense linear (`1 <= M <= 32`,
/// `K % 32 == 0`, 0-ULP batch-size invariant across `M in 1..=32`) -> `[M, N]` BF16/F32.
pub fn inkling2_q8_sym_matmul_bf16(
    x: &MxArray,
    w_q8: &MxArray,
    scales_f16: &MxArray,
    out_dtype: DType,
) -> Result<MxArray> {
    let output_f32 = out_dtype == DType::Float32;
    // SAFETY: all input handles are live.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_q8_sym_matmul_bf16(
            x.as_raw_ptr(),
            w_q8.as_raw_ptr(),
            scales_f16.as_raw_ptr(),
            output_f32,
        )
    };
    MxArray::from_handle(ptr, "inkling2_q8_sym_matmul_bf16")
}

/// Single-pass Symmetric Q8_0 dense linear + fused exact-BF16 tail projection (`w_aux_bf16 [N_aux, K]`)
/// -> `(out_main [M, N], out_aux [M, N_aux] BF16)` in ONE Metal dispatch (`1 <= M <= 32`, `K % 32 == 0`).
pub fn inkling2_q8_sym_matmul_bf16_with_aux(
    x: &MxArray,
    w_q8: &MxArray,
    scales_f16: &MxArray,
    w_aux_bf16: &MxArray,
    out_dtype: DType,
) -> Result<(MxArray, MxArray)> {
    let output_f32 = out_dtype == DType::Float32;
    let mut out_main = std::ptr::null_mut();
    let mut out_aux = std::ptr::null_mut();
    // SAFETY: all input handles are live.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_q8_sym_matmul_bf16_with_aux(
            x.as_raw_ptr(),
            w_q8.as_raw_ptr(),
            scales_f16.as_raw_ptr(),
            w_aux_bf16.as_raw_ptr(),
            output_f32,
            &mut out_main,
            &mut out_aux,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_inkling2_q8_sym_matmul_bf16_with_aux failed",
        ));
    }
    let main_arr = MxArray::from_handle(out_main, "inkling2_q8_sym_matmul_bf16_with_aux.main")?;
    let aux_arr = MxArray::from_handle(out_aux, "inkling2_q8_sym_matmul_bf16_with_aux.aux")?;
    Ok((main_arr, aux_arr))
}

/// MoE expert weight quantization mode wire code (matches `mlx_inkling2_grouped_moe*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Inkling2MoeQuantMode {
    Nvfp4 = 0,
    Q8 = 1,
    Mxfp8 = 2,
}

/// Check whether Inkling2 grouped/fused NVFP4/Q8_0/MXFP8 MoE Metal kernels are available for `rows`.
pub fn inkling2_grouped_moe_available(rows: usize) -> bool {
    #[cfg(target_os = "macos")]
    {
        u32::try_from(rows).is_ok_and(|r| unsafe { mlx_sys::mlx_inkling2_grouped_moe_available(r) })
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = rows;
        false
    }
}

/// Grouped/fused Inkling2 NVFP4 (`group_size=16`, `uint8` E4M3 scales + optional F32 `[E]`
/// `gu_outer_scale`), Q8_0 affine (`group_size=32`, BF16/F32 scales & biases), or MXFP8
/// (`group_size=32`, `uint8` UE8M0 scales) MoE forward.
///
/// Executes:
/// - `T == 1`: zero-gather fused `m1_gate_up_situ` -> `m1_down_combine` (2 dispatches!)
/// - `T >= 2`: GPU `group` -> indirect `gather` -> indirect hybrid SIMD/NAX `gate_up+SiTU`
///   -> indirect hybrid SIMD/NAX `down` -> `combine` (zero host sync, reads each active expert ONCE!).
#[allow(clippy::too_many_arguments)]
pub fn inkling2_grouped_moe(
    x: &MxArray,
    top_k_indices: &MxArray,
    gamma: &MxArray,
    gu_w: &MxArray,
    gu_s: &MxArray,
    gu_aux: Option<&MxArray>,
    dn_w: &MxArray,
    dn_s: &MxArray,
    dn_aux: Option<&MxArray>,
    quant_mode: Inkling2MoeQuantMode,
    interleaved_gate_up: bool,
    gate_softcap: f32,
    up_softcap: f32,
    round_bf16_before_situ: bool,
) -> Result<MxArray> {
    // SAFETY: all non-null array handles are valid and live for the duration of the FFI call.
    let ptr = unsafe {
        mlx_sys::mlx_inkling2_grouped_moe(
            x.as_raw_ptr(),
            top_k_indices.as_raw_ptr(),
            gamma.as_raw_ptr(),
            gu_w.as_raw_ptr(),
            gu_s.as_raw_ptr(),
            gu_aux.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            dn_w.as_raw_ptr(),
            dn_s.as_raw_ptr(),
            dn_aux.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            quant_mode as u32,
            interleaved_gate_up,
            gate_softcap,
            up_softcap,
            round_bf16_before_situ,
        )
    };
    MxArray::from_handle(ptr, "inkling2_grouped_moe")
}

/// End-to-end fused `1 <= T <= 32` Inkling2 MoE forward:
/// `route+group -> direct-gather gate_up+SiTU -> down -> combine+post_conv+residual (+ optional tape)`.
///
/// Returns `(y [T, D] BF16, hist_out [3, D] or [N, 3, D] BF16, Option<tape_out [T, D] BF16>)`.
#[allow(clippy::too_many_arguments)]
pub fn inkling2_grouped_moe_fused(
    x: &MxArray,
    router_input: &MxArray,
    router_bias: Option<&MxArray>,
    down_outer_scale: Option<&MxArray>,
    gu_w: &MxArray,
    gu_s: &MxArray,
    gu_aux: Option<&MxArray>,
    dn_w: &MxArray,
    dn_s: &MxArray,
    dn_aux: Option<&MxArray>,
    post_conv_w: &MxArray,
    residual: Option<&MxArray>,
    cu_seqlens: &[usize],
    hists: &[&MxArray],
    num_seqs: usize,
    packed_hist: bool,
    top_k: usize,
    route_scale: f32,
    quant_mode: Inkling2MoeQuantMode,
    interleaved_gate_up: bool,
    gate_softcap: f32,
    up_softcap: f32,
    round_bf16_before_situ: bool,
    emit_tape: bool,
    router_input_is_weight: bool,
) -> Result<(MxArray, MxArray, Option<MxArray>)> {
    let expected_h = if packed_hist { 1 } else { num_seqs };
    if num_seqs == 0 || hists.len() != expected_h {
        return Err(Error::from_reason(
            "inkling2_grouped_moe_fused: history slice count mismatch",
        ));
    }
    let x_shape = x.shape()?;
    if x_shape.len() != 2 || x_shape[0] <= 0 {
        return Err(Error::from_reason(
            "inkling2_grouped_moe_fused: x must be 2D with positive rows",
        ));
    }
    let total_rows = usize::try_from(x_shape[0])
        .map_err(|_| Error::from_reason("inkling2_grouped_moe_fused: invalid row count"))?;
    if cu_seqlens.len() != num_seqs + 1
        || cu_seqlens.first() != Some(&0)
        || cu_seqlens.last() != Some(&total_rows)
        || !cu_seqlens.windows(2).all(|w| w[1] > w[0])
    {
        return Err(Error::from_reason(format!(
            "inkling2_grouped_moe_fused: invalid cu_seqlens {cu_seqlens:?} for num_seqs={num_seqs}, rows={total_rows}"
        )));
    }
    let cu_seqlens_i32: Vec<i32> = cu_seqlens
        .iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| {
                Error::from_reason("inkling2_grouped_moe_fused: cu_seqlens value exceeds i32::MAX")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let cu_seqlens_len_u32 = u32::try_from(cu_seqlens_i32.len()).map_err(|_| {
        Error::from_reason("inkling2_grouped_moe_fused: cu_seqlens.len() exceeds u32::MAX")
    })?;
    let num_seqs_u32 = u32::try_from(num_seqs)
        .map_err(|_| Error::from_reason("inkling2_grouped_moe_fused: num_seqs exceeds u32::MAX"))?;
    let num_hists_u32 = u32::try_from(hists.len()).map_err(|_| {
        Error::from_reason("inkling2_grouped_moe_fused: hists.len() exceeds u32::MAX")
    })?;
    let top_k_u32 = u32::try_from(top_k)
        .map_err(|_| Error::from_reason("inkling2_grouped_moe_fused: top_k exceeds u32::MAX"))?;
    let hist_ptrs: Vec<*mut mlx_sys::mlx_array> = hists.iter().map(|h| h.as_raw_ptr()).collect();
    let mut out_y = std::ptr::null_mut();
    let mut out_hist = std::ptr::null_mut();
    let mut out_tape = std::ptr::null_mut();
    // SAFETY: `hist_ptrs.len() == num_hists_u32 as usize == expected_h`,
    // `cu_seqlens_i32.len() == cu_seqlens_len_u32 as usize == num_seqs + 1`, and all non-null
    // handles remain live for the duration of the FFI call.
    let ok = unsafe {
        mlx_sys::mlx_inkling2_grouped_moe_fused(
            x.as_raw_ptr(),
            router_input.as_raw_ptr(),
            router_bias.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            down_outer_scale.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            gu_w.as_raw_ptr(),
            gu_s.as_raw_ptr(),
            gu_aux.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            dn_w.as_raw_ptr(),
            dn_s.as_raw_ptr(),
            dn_aux.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            post_conv_w.as_raw_ptr(),
            residual.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
            cu_seqlens_i32.as_ptr(),
            cu_seqlens_len_u32,
            hist_ptrs.as_ptr(),
            num_hists_u32,
            num_seqs_u32,
            packed_hist,
            top_k_u32,
            route_scale,
            quant_mode as u32,
            interleaved_gate_up,
            gate_softcap,
            up_softcap,
            round_bf16_before_situ,
            emit_tape,
            router_input_is_weight,
            &mut out_y,
            &mut out_hist,
            &mut out_tape,
        )
    };
    if !ok {
        return Err(Error::from_reason("mlx_inkling2_grouped_moe_fused failed"));
    }
    let y = MxArray::from_handle(out_y, "inkling2_grouped_moe_fused.y")?;
    let hist = MxArray::from_handle(out_hist, "inkling2_grouped_moe_fused.hist")?;
    let tape = if emit_tape {
        Some(MxArray::from_handle(
            out_tape,
            "inkling2_grouped_moe_fused.tape",
        )?)
    } else {
        None
    };
    Ok((y, hist, tape))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::models::inkling2::attn_res::apply_sublayer_rms_norm;
    use crate::models::inkling2::moe::situ_activation;
    use crate::models::inkling2::short_conv::{dynamic_short_conv, static_short_conv};
    use crate::nn::Activations;
    use mlx_paged_attn::metal::MetalDtype;
    use mlx_paged_attn::prefill_plan::{self, Pass};
    use mlx_paged_attn::{LayerKVPool, PageLayout, PagedAttentionConfig};

    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next_f32() * scale).collect()
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
    }

    fn metal_available() -> bool {
        unsafe { mlx_sys::mlx_metal_is_available() }
    }

    #[test]
    fn inkling2_gate_sink_matches_mlx_reference() {
        if !metal_available() {
            return;
        }
        let mut rng = Rng(0x101);
        let (t, h, d) = (9usize, 24usize, 128usize);
        let o_f32 =
            MxArray::from_float32(&rng.vec(t * h * d, 1.5), &[t as i64, h as i64, d as i64])
                .unwrap();
        let g_f32 =
            MxArray::from_float32(&rng.vec(t * h * d, 3.0), &[t as i64, h as i64, d as i64])
                .unwrap();
        let mut lse_v = rng.vec(t * h, 5.0);
        lse_v[3] = f32::NEG_INFINITY; // edge case: empty window -> sink scale 0
        let lse = MxArray::from_float32(&lse_v, &[t as i64, h as i64]).unwrap();
        let sink_f32 = MxArray::from_float32(&rng.vec(t * h, 2.0), &[t as i64, h as i64]).unwrap();

        // BFloat16 has 7 mantissa bits: 1 ULP is 2^-8 (~0.00391) in [0.5, 1.0)
        // and 2^-7 (~0.00781) in [1.0, 2.0), so allow 1 ULP for cross-GPU fast-math exp.
        for (dtype, tol) in [(DType::BFloat16, 1e-2_f32), (DType::Float32, 1e-6_f32)] {
            let o = o_f32.astype(dtype).unwrap();
            let g = g_f32.astype(dtype).unwrap();
            let sink = sink_f32.astype(dtype).unwrap();

            // Pure-MLX reference
            let sink_scale =
                Activations::sigmoid(&lse.sub(&sink.astype(DType::Float32).unwrap()).unwrap())
                    .unwrap()
                    .expand_dims(-1)
                    .unwrap();
            let gate_scale = Activations::sigmoid(&g.astype(DType::Float32).unwrap()).unwrap();
            let ref_out = o
                .astype(DType::Float32)
                .unwrap()
                .mul(&gate_scale)
                .unwrap()
                .mul(&sink_scale)
                .unwrap()
                .astype(dtype)
                .unwrap();

            let got = inkling2_gate_sink(&o, &g, &lse, &sink).unwrap();
            let diff = max_abs_diff(&got.to_float32().unwrap(), &ref_out.to_float32().unwrap());
            assert!(diff <= tol, "gate_sink {dtype:?} max_diff={diff} > {tol}");
        }
    }

    #[test]
    fn inkling2_situ_matches_mlx_reference_split_interleaved_and_scaled() {
        if !metal_available() {
            return;
        }
        let mut rng = Rng(0x202);
        let (rows, inter, num_exp) = (16usize, 1152usize, 128usize);
        let gu_vals = rng.vec(rows * 2 * inter, 12.0); // large values to exercise softcap + un-softcapped sigmoid
        let outer_vals = rng
            .vec(num_exp, 0.5)
            .into_iter()
            .map(|v| v.abs() + 0.1)
            .collect::<Vec<_>>();
        let idx_vals: Vec<i32> = (0..rows).map(|i| ((i * 7 + 3) % num_exp) as i32).collect();

        let gu_f32 =
            MxArray::from_float32(&gu_vals, &[rows as i64, 1, (2 * inter) as i64]).unwrap();
        let outer = MxArray::from_float32(&outer_vals, &[num_exp as i64]).unwrap();
        let idx = MxArray::from_int32(&idx_vals, &[rows as i64]).unwrap();

        for interleaved in [false, true] {
            for with_scale in [false, true] {
                // BFloat16 has 7 mantissa bits: 1 ULP scales with |x| (e.g. 2^-8 at 0.5,
                // 2^-6 at 2.0), so allow ~1-2 BF16 ULP for fast-math tanh/sigmoid.
                for (dtype, tol) in [(DType::BFloat16, 2e-2_f32), (DType::Float32, 1e-5_f32)] {
                    let gu = gu_f32.astype(dtype).unwrap();
                    // Reference
                    let mut ref_gu = gu.astype(DType::Float32).unwrap();
                    if with_scale {
                        let gs = outer
                            .take(&idx, 0)
                            .unwrap()
                            .reshape(&[rows as i64, 1, 1])
                            .unwrap();
                        ref_gu = ref_gu.mul(&gs).unwrap();
                    }
                    let (g_ref, u_ref) = if interleaved {
                        let paired = ref_gu.reshape(&[rows as i64, 1, inter as i64, 2]).unwrap();
                        (
                            paired
                                .slice(&[0, 0, 0, 0], &[rows as i64, 1, inter as i64, 1])
                                .unwrap()
                                .squeeze(Some(&[-1]))
                                .unwrap(),
                            paired
                                .slice(&[0, 0, 0, 1], &[rows as i64, 1, inter as i64, 2])
                                .unwrap()
                                .squeeze(Some(&[-1]))
                                .unwrap(),
                        )
                    } else {
                        (
                            ref_gu
                                .slice(&[0, 0, 0], &[rows as i64, 1, inter as i64])
                                .unwrap(),
                            ref_gu
                                .slice(&[0, 0, inter as i64], &[rows as i64, 1, (2 * inter) as i64])
                                .unwrap(),
                        )
                    };
                    let ref_out = situ_activation(&g_ref, &u_ref, 4.0, 25.0)
                        .unwrap()
                        .astype(dtype)
                        .unwrap();

                    let got = inkling2_situ(
                        &gu,
                        with_scale.then_some((&outer, &idx)),
                        4.0,
                        25.0,
                        interleaved,
                    )
                    .unwrap();
                    let diff =
                        max_abs_diff(&got.to_float32().unwrap(), &ref_out.to_float32().unwrap());
                    assert!(
                        diff <= tol,
                        "situ interleaved={interleaved} scaled={with_scale} {dtype:?} diff={diff} > {tol}"
                    );
                }
            }
        }
    }

    #[test]
    fn inkling2_loop_blend_rms_norm_matches_mlx_reference() {
        if !metal_available() {
            return;
        }
        let mut rng = Rng(0x303);
        let (t, d) = (8usize, 2304usize);
        let dest = MxArray::from_float32(&rng.vec(t * d, 2.0), &[t as i64, d as i64]).unwrap();
        let msg = MxArray::from_float32(&rng.vec(t * d, 2.0), &[t as i64, d as i64]).unwrap();
        let gain_raw = MxArray::from_float32(&rng.vec(t * d, 3.0), &[t as i64, d as i64]).unwrap();
        let forget_raw =
            MxArray::from_float32(&rng.vec(t * d, 3.0), &[t as i64, d as i64]).unwrap();
        let weight = MxArray::from_float32(&rng.vec(d, 1.0), &[d as i64]).unwrap();

        // Reference FP32 blend + sublayer RMSNorm
        let g = gain_raw.tanh().unwrap();
        let f = forget_raw.tanh().unwrap();
        let one_minus_f = MxArray::scalar_float(1.0).unwrap().sub(&f).unwrap();
        let blended = one_minus_f
            .mul(&dest)
            .unwrap()
            .add(&g.mul(&msg).unwrap())
            .unwrap();

        // BFloat16 has 7 mantissa bits: allow ~1-2 BF16 ULP for fast-math tanh/rsqrt.
        for (dtype, tol) in [(DType::BFloat16, 2e-2_f32), (DType::Float32, 1e-5_f32)] {
            let ref_out = apply_sublayer_rms_norm(&blended, &weight, 1e-6, dtype).unwrap();
            let got = inkling2_loop_blend_rms_norm(
                &dest,
                &msg,
                &gain_raw,
                &forget_raw,
                &weight,
                1e-6,
                dtype,
            )
            .unwrap();
            let diff = max_abs_diff(&got.to_float32().unwrap(), &ref_out.to_float32().unwrap());
            assert!(
                diff <= tol,
                "loop_blend_rms_norm {dtype:?} diff={diff} > {tol}"
            );
        }
    }

    #[test]
    fn inkling2_sconv_k4_static_and_dynamic_match_mlx_reference() {
        if !metal_available() {
            return;
        }
        let mut rng = Rng(0x404);
        let (c_static, n_kv, head_dim) = (2304usize, 8usize, 128usize);
        let c_dyn = n_kv * head_dim;

        for seq_len in [1usize, 2, 8, 37] {
            // Static
            let x = MxArray::from_float32(
                &rng.vec(seq_len * c_static, 1.5),
                &[seq_len as i64, c_static as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let w =
                MxArray::from_float32(&rng.vec(c_static * 4, 0.5), &[c_static as i64, 4]).unwrap();
            let h0 = MxArray::from_float32(&rng.vec(3 * c_static, 1.5), &[3, c_static as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();

            let mut ref_h = h0.clone();
            let ref_y = static_short_conv(&x, &w, &mut ref_h).unwrap();
            let (got_y, got_h) = inkling2_sconv_k4(&x, &w, &h0, None).unwrap();
            let dy = max_abs_diff(&got_y.to_float32().unwrap(), &ref_y.to_float32().unwrap());
            let dh = max_abs_diff(&got_h.to_float32().unwrap(), &ref_h.to_float32().unwrap());
            assert!(dy <= 1e-3, "static sconv T={seq_len} y diff={dy}");
            assert!(dh == 0.0, "static sconv T={seq_len} hist diff={dh}");

            // Dynamic global
            let xd = MxArray::from_float32(
                &rng.vec(seq_len * c_dyn, 1.5),
                &[seq_len as i64, c_dyn as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let wd = MxArray::from_float32(&rng.vec(c_dyn * 4, 0.5), &[c_dyn as i64, 4]).unwrap();
            let hd0 = MxArray::from_float32(&rng.vec(3 * c_dyn, 1.5), &[3, c_dyn as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let dyn_taps = MxArray::from_float32(
                &rng.vec(seq_len * n_kv * 4, 0.4),
                &[seq_len as i64, n_kv as i64, 4],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

            let mut ref_hd = hd0.clone();
            let ref_yd = dynamic_short_conv(&xd, &wd, &dyn_taps, &mut ref_hd).unwrap();
            let (got_yd, got_hd) = inkling2_sconv_k4(&xd, &wd, &hd0, Some(&dyn_taps)).unwrap();
            let dyd = max_abs_diff(&got_yd.to_float32().unwrap(), &ref_yd.to_float32().unwrap());
            let dhd = max_abs_diff(&got_hd.to_float32().unwrap(), &ref_hd.to_float32().unwrap());
            assert!(dyd <= 1e-3, "dynamic sconv T={seq_len} y diff={dyd}");
            assert!(dhd == 0.0, "dynamic sconv T={seq_len} hist diff={dhd}");
        }
    }

    /// Reference FP64/FP32 attention with causal mask, optional SWA, relative bias for `d < rel_extent`,
    /// returning `(out [total_q, H, D], lse [total_q, H])`.
    #[allow(clippy::too_many_arguments)]
    fn ref_varlen_attention(
        q: &[f32],
        seq_k: &[Vec<f32>],
        seq_v: &[Vec<f32>],
        cu_seqlens_q: &[i32],
        seq_lens: &[i32],
        rel_bias: &[f32],
        rel_extent: usize,
        sliding_window: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let total_q = *cu_seqlens_q.last().unwrap() as usize;
        let group = num_q_heads / num_kv_heads;
        let mut out = vec![0.0_f32; total_q * num_q_heads * head_dim];
        let mut lse = vec![f32::NEG_INFINITY; total_q * num_q_heads];

        for seq_idx in 0..seq_lens.len() {
            let q_start = cu_seqlens_q[seq_idx] as usize;
            let q_end = cu_seqlens_q[seq_idx + 1] as usize;
            let q_len = q_end - q_start;
            let ctx = seq_lens[seq_idx] as usize;
            let k_all = &seq_k[seq_idx];
            let v_all = &seq_v[seq_idx];

            for q_pos_in_seq in 0..q_len {
                let q_row = q_start + q_pos_in_seq;
                let eff_ctx = ctx - q_len + q_pos_in_seq + 1;
                let q_pos = eff_ctx - 1;
                let lower = if sliding_window > 0 && eff_ctx > sliding_window {
                    eff_ctx - sliding_window
                } else {
                    0
                };
                for h in 0..num_q_heads {
                    let kv_h = h / group;
                    let q_base = (q_row * num_q_heads + h) * head_dim;
                    let mut scores = Vec::with_capacity(eff_ctx - lower);
                    let mut m = f64::NEG_INFINITY;
                    for k_pos in lower..eff_ctx {
                        let k_base = (k_pos * num_kv_heads + kv_h) * head_dim;
                        let mut dot = 0.0_f64;
                        for d in 0..head_dim {
                            dot += f64::from(q[q_base + d]) * f64::from(k_all[k_base + d]);
                        }
                        let mut s = dot * f64::from(scale);
                        let dist = q_pos - k_pos;
                        if rel_extent > 0 && dist < rel_extent {
                            s += f64::from(rel_bias[(q_row * num_q_heads + h) * rel_extent + dist]);
                        }
                        m = m.max(s);
                        scores.push(s);
                    }
                    let mut sum_exp = 0.0_f64;
                    let mut weights = Vec::with_capacity(scores.len());
                    for s in &scores {
                        let w = (s - m).exp();
                        sum_exp += w;
                        weights.push(w);
                    }
                    lse[q_row * num_q_heads + h] = (m + sum_exp.ln()) as f32;
                    for d in 0..head_dim {
                        let mut acc = 0.0_f64;
                        for (idx, &w) in weights.iter().enumerate() {
                            let k_pos = lower + idx;
                            let v_base = (k_pos * num_kv_heads + kv_h) * head_dim;
                            acc += (w / sum_exp) * f64::from(v_all[v_base + d]);
                        }
                        out[q_base + d] = acc as f32;
                    }
                }
            }
        }
        (out, lse)
    }

    struct TestPool {
        _pool: LayerKVPool,
        k_pool: MxArray,
        v_pool: MxArray,
        k_scale: MxArray,
        v_scale: MxArray,
    }

    fn build_pool(
        kv_dtype: Inkling2KvDtype,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_blocks: usize,
    ) -> TestPool {
        let layout = match kv_dtype {
            Inkling2KvDtype::Int8 => PageLayout {
                keys_token_major: true,
                ..PageLayout::default()
            },
            Inkling2KvDtype::Bf16 => PageLayout::default(),
        };
        let metal_dtype = match kv_dtype {
            Inkling2KvDtype::Int8 => MetalDtype::Int8,
            Inkling2KvDtype::Bf16 => MetalDtype::BFloat16,
        };
        let cfg = PagedAttentionConfig {
            block_size: block_size as u32,
            gpu_memory_mb: 512,
            head_size: head_dim as u32,
            num_kv_heads: num_kv_heads as u32,
            num_layers: 1,
            max_seq_len: Some((num_blocks * block_size) as u32),
            max_batch_size: Some(4),
        };
        let pool = LayerKVPool::new_with_layout(
            cfg,
            num_blocks as u32,
            num_blocks as u32,
            metal_dtype,
            layout,
        )
        .unwrap();
        // Fill spare sentinel blocks (`0` and `num_blocks - 1`) with 0xFF (NaN in BF16 and F32
        // scales) so any evicted SWA read or out-of-bounds block read fails loudly; fill active
        // blocks with finite non-zero sentinel bytes (0x42 = BF16 ~0.008 / INT8 66, finite) so
        // unwritten tail slots still catch masking bugs without triggering `0.0 * NaN = NaN` on
        // in-place BF16 MPP device reads (matching `paged_prefill_attention.rs`).
        let (kb, vb) = pool.block_blob_bytes().unwrap();
        let k_nan = vec![0xffu8; kb as usize];
        let v_nan = vec![0xffu8; vb as usize];
        let k_fin = vec![0x42u8; kb as usize];
        let v_fin = vec![0x42u8; vb as usize];
        let uploads: Vec<_> = (0..num_blocks as u32)
            .map(|b| {
                let is_spare = b == 0 || b == (num_blocks - 1) as u32;
                let (ks, vs) = if is_spare {
                    (k_nan.as_slice(), v_nan.as_slice())
                } else {
                    (k_fin.as_slice(), v_fin.as_slice())
                };
                (b, vec![(ks, vs)])
            })
            .collect();
        pool.write_blocks_all_layers(&uploads).unwrap();

        let k_pool = MxArray::from_handle(pool.key_cache_array_raw(0).unwrap(), "k_pool").unwrap();
        let v_pool =
            MxArray::from_handle(pool.value_cache_array_raw(0).unwrap(), "v_pool").unwrap();
        let (k_scale, v_scale) = match kv_dtype {
            Inkling2KvDtype::Int8 => (
                MxArray::from_handle(pool.key_scale_array_raw(0).unwrap(), "k_scale").unwrap(),
                MxArray::from_handle(pool.value_scale_array_raw(0).unwrap(), "v_scale").unwrap(),
            ),
            Inkling2KvDtype::Bf16 => (
                MxArray::from_float32(&[1.0], &[1]).unwrap(),
                MxArray::from_float32(&[1.0], &[1]).unwrap(),
            ),
        };
        TestPool {
            _pool: pool,
            k_pool,
            v_pool,
            k_scale,
            v_scale,
        }
    }

    fn store_kv(
        pool: &mut TestPool,
        k: &MxArray,
        v: &MxArray,
        slots: &[i64],
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dtype: Inkling2KvDtype,
    ) {
        let slot_arr = MxArray::from_int64(slots, &[slots.len() as i64]).unwrap();
        let mut outs = [std::ptr::null_mut(); 4];
        let ok = unsafe {
            mlx_sys::mlx_paged_kv_write_forward(
                pool.k_pool.as_raw_ptr(),
                pool.v_pool.as_raw_ptr(),
                k.as_raw_ptr(),
                v.as_raw_ptr(),
                slot_arr.as_raw_ptr(),
                pool.k_scale.as_raw_ptr(),
                pool.v_scale.as_raw_ptr(),
                block_size as i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dtype as u8,
                false,
                &mut outs[0],
                &mut outs[1],
                &mut outs[2],
                &mut outs[3],
            )
        };
        assert!(ok, "mlx_paged_kv_write_forward failed for {kv_dtype:?}");
        pool.k_pool = MxArray::from_handle(outs[0], "kp").unwrap();
        pool.v_pool = MxArray::from_handle(outs[1], "vp").unwrap();
        pool.k_scale = MxArray::from_handle(outs[2], "ks").unwrap();
        pool.v_scale = MxArray::from_handle(outs[3], "vs").unwrap();
        MxArray::eval_arrays(&[&pool.k_pool, &pool.v_pool, &pool.k_scale, &pool.v_scale]).unwrap();
    }

    #[test]
    fn inkling2_paged_decode_varlen_bf16_and_int8_ktm_parity() {
        if !metal_available() {
            return;
        }
        let num_q_heads = 24usize;
        let num_kv_heads = 8usize;
        let head_dim = 128usize;
        let rel_extent = 1024usize;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        // Test matrix:
        // - V1 short ctx (ctx <= 512) and V2 long ctx (ctx = 1200 > 1024 crossing both SWA & rel_extent!)
        // - Global (sliding_window=0 -> tests d >= 1024 zero-bias path!) and SWA (sliding_window=1024)
        // - Multi-sequence varlen batch (seq 0: decode q=1, seq 1: DSpark verify q=8, seq 2: q=3)
        // - Both BF16 (bs=16 & bs=32) and INT8 32-token KTM!
        let cases: &[(&str, Inkling2KvDtype, usize, &[i32], &[i32], usize, DType)] = &[
            (
                "bf16_bs16_v1_varlen_global",
                Inkling2KvDtype::Bf16,
                16,
                &[97, 260, 45],
                &[0, 1, 9, 12],
                0,
                DType::BFloat16,
            ),
            (
                "bf16_bs32_v2_global_cross1024",
                Inkling2KvDtype::Bf16,
                32,
                &[1200, 1050],
                &[0, 8, 9],
                0,
                DType::Float32,
            ),
            (
                "bf16_bs32_v2_swa1024",
                Inkling2KvDtype::Bf16,
                32,
                &[1200, 1050],
                &[0, 8, 9],
                1024,
                DType::BFloat16,
            ),
            (
                "int8_ktm_v1_varlen_global",
                Inkling2KvDtype::Int8,
                32,
                &[97, 260, 45],
                &[0, 1, 9, 12],
                0,
                DType::BFloat16,
            ),
            (
                "int8_ktm_v2_global_cross1024",
                Inkling2KvDtype::Int8,
                32,
                &[1200, 1050],
                &[0, 8, 9],
                0,
                DType::Float32,
            ),
            (
                "int8_ktm_v2_swa1024",
                Inkling2KvDtype::Int8,
                32,
                &[1200, 1050],
                &[0, 8, 9],
                1024,
                DType::BFloat16,
            ),
        ];

        for &(label, kv_dtype, bs, seq_lens, cu_q, swa, bias_dtype) in cases {
            let mut rng = Rng(0x5050 + bs as u64 + swa as u64 + kv_dtype as u64);
            let num_seqs = seq_lens.len();
            let total_q = *cu_q.last().unwrap() as usize;
            let max_blocks = seq_lens
                .iter()
                .map(|&s| (s as usize).div_ceil(bs))
                .max()
                .unwrap();
            let total_used_blocks: usize =
                seq_lens.iter().map(|&s| (s as usize).div_ceil(bs)).sum();
            let num_blocks = total_used_blocks + 2; // +2 spare NaN blocks

            let mut pool = build_pool(kv_dtype, bs, num_kv_heads, head_dim, num_blocks);
            let mut block_table_flat = vec![(num_blocks - 1) as i32; num_seqs * max_blocks];
            let mut next_phys = total_used_blocks; // reverse-allocate physical blocks
            let mut seq_k_ref = Vec::with_capacity(num_seqs);
            let mut seq_v_ref = Vec::with_capacity(num_seqs);

            for (s_idx, &ctx) in seq_lens.iter().enumerate() {
                let ctx_u = ctx as usize;
                let nb_s = ctx_u.div_ceil(bs);
                let mut slots = Vec::with_capacity(ctx_u);
                for lb in 0..nb_s {
                    next_phys -= 1;
                    let phys = next_phys + 1; // avoid block 0 (leave 0 as NaN sentinel)
                    block_table_flat[s_idx * max_blocks + lb] = phys as i32;
                }
                for tok in 0..ctx_u {
                    let phys = block_table_flat[s_idx * max_blocks + tok / bs] as usize;
                    slots.push((phys * bs + tok % bs) as i64);
                }
                let k_arr = MxArray::from_float32(
                    &rng.vec(ctx_u * num_kv_heads * head_dim, 0.4),
                    &[ctx_u as i64, num_kv_heads as i64, head_dim as i64],
                )
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
                let v_arr = MxArray::from_float32(
                    &rng.vec(ctx_u * num_kv_heads * head_dim, 0.4),
                    &[ctx_u as i64, num_kv_heads as i64, head_dim as i64],
                )
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
                seq_k_ref.push(k_arr.to_float32().unwrap().to_vec());
                seq_v_ref.push(v_arr.to_float32().unwrap().to_vec());
                store_kv(
                    &mut pool,
                    &k_arr,
                    &v_arr,
                    &slots,
                    bs,
                    num_kv_heads,
                    head_dim,
                    kv_dtype,
                );
            }

            let q_arr = MxArray::from_float32(
                &rng.vec(total_q * num_q_heads * head_dim, 0.4),
                &[total_q as i64, num_q_heads as i64, head_dim as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let rel_bias_arr = MxArray::from_float32(
                &rng.vec(total_q * num_q_heads * rel_extent, 0.5),
                &[total_q as i64, num_q_heads as i64, rel_extent as i64],
            )
            .unwrap()
            .astype(bias_dtype)
            .unwrap();
            let bt_arr =
                MxArray::from_int32(&block_table_flat, &[num_seqs as i64, max_blocks as i64])
                    .unwrap();
            let sl_arr = MxArray::from_int32(seq_lens, &[num_seqs as i64]).unwrap();
            let cu_arr = MxArray::from_int32(cu_q, &[cu_q.len() as i64]).unwrap();

            let (got_out, got_lse) = inkling2_paged_attention(
                &q_arr,
                &pool.k_pool,
                &pool.v_pool,
                &bt_arr,
                &sl_arr,
                &cu_arr,
                &pool.k_scale,
                &pool.v_scale,
                Some(&rel_bias_arr),
                scale,
                swa as i32,
                rel_extent as i32,
                bs as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dtype,
                None,
            )
            .unwrap();

            let (ref_out, ref_lse) = ref_varlen_attention(
                &q_arr.to_float32().unwrap(),
                &seq_k_ref,
                &seq_v_ref,
                cu_q,
                seq_lens,
                &rel_bias_arr.to_float32().unwrap(),
                rel_extent,
                swa,
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
            );

            let got_out_v = got_out.to_float32().unwrap();
            let got_lse_v = got_lse.to_float32().unwrap();
            assert!(
                got_out_v.iter().all(|v| v.is_finite()),
                "{label}: non-finite out"
            );
            assert!(
                got_lse_v.iter().all(|v| v.is_finite()),
                "{label}: non-finite lse"
            );

            let out_diff = max_abs_diff(&got_out_v, &ref_out);
            let lse_diff = max_abs_diff(&got_lse_v, &ref_lse);
            let (out_tol, lse_tol) = match kv_dtype {
                Inkling2KvDtype::Bf16 => (1e-3_f32, 2e-3_f32),
                Inkling2KvDtype::Int8 => (2e-2_f32, 5e-2_f32),
            };
            assert!(
                out_diff <= out_tol,
                "{label}: out max_diff={out_diff} > {out_tol}"
            );
            assert!(
                lse_diff <= lse_tol,
                "{label}: lse max_diff={lse_diff} > {lse_tol}"
            );
        }
    }

    #[test]
    fn inkling2_paged_prefill_mpp_ktm_q8_q8q_bf16_swa_and_global_parity() {
        if !metal_available() {
            return;
        }
        let num_q_heads = 24usize;
        let num_kv_heads = 8usize;
        let head_dim = 128usize;
        let rel_extent = 1024usize;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let mpp_ok = inkling2_paged_prefill_mpp_supported();
        let apple10 = unsafe { mlx_sys::mlx_gpu_architecture_gen() } >= 17;
        eprintln!("inkling2 MPP supported={mpp_ok}, apple10(q8q)={apple10}");

        // Cases cover:
        // - SWA=1024 at committed=1500, rows=40 (first 14 pages completely skipped; we even poison
        //   evicted pages with NaN in block_table/pool to PROVE tile skip never touches them!)
        // - Global (swa=0) at committed=1500, rows=40 (pages 0..14 take ZERO-BIAS fast path,
        //   pages 15..48 take rel_bias path!)
        // - Both INT8 KTM q8 (BF16 queries) and q8q (INT8 queries on M5+/Apple10)
        // - BF16 vLLM pool (bs=16 & bs=32)
        // - Multiple splits (splits=1, splits=4)
        let cases: &[(
            &str,
            Inkling2KvDtype,
            usize,
            usize,
            usize,
            usize,
            usize,
            bool,
        )] = &[
            (
                "ktm_q8_swa1024_skip",
                Inkling2KvDtype::Int8,
                32,
                1500,
                40,
                1024,
                4,
                false,
            ),
            (
                "ktm_q8_global_cross1024",
                Inkling2KvDtype::Int8,
                32,
                1500,
                40,
                0,
                4,
                false,
            ),
            (
                "ktm_q8q_global_cross1024",
                Inkling2KvDtype::Int8,
                32,
                1500,
                40,
                0,
                4,
                true,
            ),
            (
                "ktm_q8_verify8_swa1024",
                Inkling2KvDtype::Int8,
                32,
                1200,
                8,
                1024,
                8,
                false,
            ),
            (
                "bf16_bs16_swa1024_skip",
                Inkling2KvDtype::Bf16,
                16,
                1500,
                40,
                1024,
                4,
                false,
            ),
            (
                "bf16_bs32_global_cross1024",
                Inkling2KvDtype::Bf16,
                32,
                1500,
                40,
                0,
                4,
                false,
            ),
        ];

        for &(label, kv_dtype, bs, committed, rows, swa, splits, int8_queries) in cases {
            let mut rng = Rng(0x7070 + bs as u64 + swa as u64 + (int8_queries as u64) * 17);
            let total_tokens = committed + rows;
            let used_blocks = total_tokens.div_ceil(bs);
            let num_blocks = used_blocks + 2; // spare NaN blocks at 0 and used_blocks+1

            let mut pool = build_pool(kv_dtype, bs, num_kv_heads, head_dim, num_blocks);
            let mut table: Vec<i32> = (0..used_blocks)
                .map(|lb| (used_blocks - lb) as i32)
                .collect();
            let slots: Vec<i64> = (0..total_tokens)
                .map(|t| (table[t / bs] as usize * bs + t % bs) as i64)
                .collect();

            let k_arr = MxArray::from_float32(
                &rng.vec(total_tokens * num_kv_heads * head_dim, 0.35),
                &[total_tokens as i64, num_kv_heads as i64, head_dim as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let v_arr = MxArray::from_float32(
                &rng.vec(total_tokens * num_kv_heads * head_dim, 0.35),
                &[total_tokens as i64, num_kv_heads as i64, head_dim as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            store_kv(
                &mut pool,
                &k_arr,
                &v_arr,
                &slots,
                bs,
                num_kv_heads,
                head_dim,
                kv_dtype,
            );

            // For SWA cases, repoint completely evicted blocks (strictly before
            // `committed + 1 - swa`) to block 0 (filled with 0xFF NaNs) to prove
            // the SWA tile skip never reads evicted pages!
            if swa > 0 && committed + 1 > swa {
                let first_needed_page = (committed + 1 - swa) / 32;
                let first_needed_block = first_needed_page * (32 / bs);
                for entry in table.iter_mut().take(first_needed_block) {
                    *entry = 0;
                }
            }

            let q_arr = MxArray::from_float32(
                &rng.vec(rows * num_q_heads * head_dim, 0.35),
                &[rows as i64, num_q_heads as i64, head_dim as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let rel_bias_arr = MxArray::from_float32(
                &rng.vec(rows * num_q_heads * rel_extent, 0.5),
                &[rows as i64, num_q_heads as i64, rel_extent as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let bt_arr = MxArray::from_int32(&table, &[table.len() as i64]).unwrap();
            let pass = if rows <= 8 {
                Pass::Verify
            } else {
                Pass::Prefill
            };
            let ws_floats = prefill_plan::workspace_floats(
                rows as u32,
                splits as u32,
                num_q_heads as u32,
                head_dim as u32,
            );
            let workspace = MxArray::zeros(&[ws_floats as i64], Some(DType::Float32)).unwrap();

            let (got_out, got_lse, _ws_out) = inkling2_paged_prefill_attention(
                &q_arr,
                &pool.k_pool,
                &pool.v_pool,
                &pool.k_scale,
                &pool.v_scale,
                &bt_arr,
                &workspace,
                Some(&rel_bias_arr),
                committed as i32,
                splits as i32,
                pass.max_splits() as i32,
                bs as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                swa as i32,
                rel_extent as i32,
                kv_dtype,
                int8_queries,
                scale,
            )
            .unwrap();

            let (ref_out, ref_lse) = ref_varlen_attention(
                &q_arr.to_float32().unwrap(),
                &[k_arr.to_float32().unwrap().to_vec()],
                &[v_arr.to_float32().unwrap().to_vec()],
                &[0, rows as i32],
                &[total_tokens as i32],
                &rel_bias_arr.to_float32().unwrap(),
                rel_extent,
                swa,
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
            );

            let got_out_v = got_out.to_float32().unwrap();
            let got_lse_v = got_lse.to_float32().unwrap();
            assert!(
                got_out_v.iter().all(|v| v.is_finite()),
                "{label}: non-finite out"
            );
            assert!(
                got_lse_v.iter().all(|v| v.is_finite()),
                "{label}: non-finite lse"
            );

            let out_diff = max_abs_diff(&got_out_v, &ref_out);
            let lse_diff = max_abs_diff(&got_lse_v, &ref_lse);
            let (out_tol, lse_tol) = match (kv_dtype, int8_queries) {
                (Inkling2KvDtype::Bf16, _) => (1e-3_f32, 2e-3_f32),
                (Inkling2KvDtype::Int8, false) => (2e-2_f32, 5e-2_f32),
                (Inkling2KvDtype::Int8, true) => (2.5e-2_f32, 6e-2_f32),
            };
            assert!(
                out_diff <= out_tol,
                "{label}: out max_diff={out_diff} > {out_tol}"
            );
            assert!(
                lse_diff <= lse_tol,
                "{label}: lse max_diff={lse_diff} > {lse_tol}"
            );
        }
    }

    #[test]
    fn inkling2_qr_clip_and_strided_gate_sink_match_reference() {
        if !metal_available() {
            return;
        }
        use crate::models::inkling2::attention::l2_clip;
        let mut rng = Rng(0x6060);
        let (t, h, hd, dr) = (8usize, 24usize, 128usize, 16usize);
        let q_dim = h * hd;
        let r_dim = h * dr;
        let total_cols = q_dim + r_dim + q_dim + 2048;

        // Pack into a fused QRGKV-like [T, 8600] buffer and slice strided views!
        let packed = MxArray::from_float32(
            &rng.vec(t * total_cols, 4.0),
            &[t as i64, total_cols as i64],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let q_view = packed.slice(&[0, 0], &[t as i64, q_dim as i64]).unwrap();
        let r_view = packed
            .slice(&[0, q_dim as i64], &[t as i64, (q_dim + r_dim) as i64])
            .unwrap();
        let g_view = packed
            .slice(
                &[0, (q_dim + r_dim) as i64],
                &[t as i64, (q_dim + r_dim + q_dim) as i64],
            )
            .unwrap();

        let q_clip =
            MxArray::from_float32(&rng.vec(q_dim, 1.5), &[1, h as i64, hd as i64]).unwrap();
        let r_clip =
            MxArray::from_float32(&rng.vec(r_dim, 1.5), &[1, h as i64, dr as i64]).unwrap();
        let tau = MxArray::from_float32(
            &[1.0, 1.05, 1.12, 1.2, 1.0, 1.03, 1.18, 1.25],
            &[t as i64, 1, 1],
        )
        .unwrap();
        let qk_thresh = 22.627416997969522_f64;
        let rel_thresh = 8.0_f64;

        for use_tau in [false, true] {
            let (got_q, got_r) = inkling2_qr_clip(
                &q_view,
                &r_view,
                &q_clip,
                &r_clip,
                use_tau.then_some(&tau),
                h,
                hd,
                dr,
                qk_thresh as f32,
                rel_thresh as f32,
                1e-6,
                DType::BFloat16,
                true,
            )
            .unwrap();

            let q_3d = q_view.reshape(&[t as i64, h as i64, hd as i64]).unwrap();
            let r_3d = r_view.reshape(&[t as i64, h as i64, dr as i64]).unwrap();
            let mut ref_q = l2_clip(&q_3d, Some(&q_clip), qk_thresh, DType::BFloat16).unwrap();
            let ref_r_bf = l2_clip(&r_3d, Some(&r_clip), rel_thresh, DType::BFloat16).unwrap();
            let ref_r = if use_tau {
                ref_q = ref_q
                    .astype(DType::Float32)
                    .unwrap()
                    .mul(&tau)
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                ref_r_bf.astype(DType::Float32).unwrap().mul(&tau).unwrap()
            } else {
                ref_r_bf.astype(DType::Float32).unwrap()
            };

            let dq = max_abs_diff(&got_q.to_float32().unwrap(), &ref_q.to_float32().unwrap());
            let dr_diff = max_abs_diff(&got_r.to_float32().unwrap(), &ref_r.to_float32().unwrap());
            assert!(dq <= 1e-2, "qr_clip use_tau={use_tau} q diff={dq}");
            assert!(
                dr_diff <= 1e-4,
                "qr_clip use_tau={use_tau} r diff={dr_diff}"
            );
        }

        // Also verify zero-copy strided gate in inkling2_gate_sink
        let o = MxArray::from_float32(&rng.vec(t * q_dim, 1.5), &[t as i64, h as i64, hd as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let lse = MxArray::from_float32(&rng.vec(t * h, 4.0), &[t as i64, h as i64]).unwrap();
        let sink = MxArray::from_float32(&rng.vec(t * h, 2.0), &[t as i64, h as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let got_gs = inkling2_gate_sink(&o, &g_view, &lse, &sink).unwrap();
        let g_contig = g_view.reshape(&[t as i64, h as i64, hd as i64]).unwrap();
        let ref_gs = inkling2_gate_sink(&o, &g_contig, &lse, &sink).unwrap();
        assert_eq!(
            max_abs_diff(&got_gs.to_float32().unwrap(), &ref_gs.to_float32().unwrap()),
            0.0,
            "strided gate_sink must be bit-identical to contiguous"
        );
    }

    #[test]
    fn inkling2_sconv_k4_kv_prep_and_ragged_match_reference() {
        if !metal_available() {
            return;
        }
        use crate::models::inkling2::attention::l2_clip;
        use crate::nn::normalization::rms_norm_unscaled;
        let mut rng = Rng(0x7070);
        let (n_kv, hd) = (8usize, 128usize);
        let kv_dim = n_kv * hd;
        let qk_thresh = 22.627416997969522_f64;

        // Test N=1 and N=4 ragged (with variable lengths [1, 8, 3, 5]), both static and dynamic,
        // and strided K/V views from a packed [T, 8600] tensor!
        for cu_vec in [vec![0i32, 8], vec![0i32, 1, 9, 12, 17]] {
            let num_seqs = cu_vec.len() - 1;
            let total_t = *cu_vec.last().unwrap() as usize;
            let cu_arr = MxArray::from_int32(&cu_vec, &[cu_vec.len() as i64]).unwrap();

            let packed_cols = 6528 + 2 * kv_dim;
            let packed = MxArray::from_float32(
                &rng.vec(total_t * packed_cols, 3.0),
                &[total_t as i64, packed_cols as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let k_view = packed
                .slice(&[0, 6528], &[total_t as i64, (6528 + kv_dim) as i64])
                .unwrap();
            let v_view = packed
                .slice(
                    &[0, (6528 + kv_dim) as i64],
                    &[total_t as i64, packed_cols as i64],
                )
                .unwrap();

            let kw = MxArray::from_float32(&rng.vec(kv_dim * 4, 0.4), &[kv_dim as i64, 4]).unwrap();
            let vw = MxArray::from_float32(&rng.vec(kv_dim * 4, 0.4), &[kv_dim as i64, 4]).unwrap();
            let dk = MxArray::from_float32(
                &rng.vec(total_t * n_kv * 4, 0.3),
                &[total_t as i64, n_kv as i64, 4],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let dv = MxArray::from_float32(
                &rng.vec(total_t * n_kv * 4, 0.3),
                &[total_t as i64, n_kv as i64, 4],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

            let k_hists: Vec<MxArray> = (0..num_seqs)
                .map(|_| {
                    MxArray::from_float32(&rng.vec(3 * kv_dim, 2.0), &[3, kv_dim as i64])
                        .unwrap()
                        .astype(DType::BFloat16)
                        .unwrap()
                })
                .collect();
            let v_hists: Vec<MxArray> = (0..num_seqs)
                .map(|_| {
                    MxArray::from_float32(&rng.vec(3 * kv_dim, 2.0), &[3, kv_dim as i64])
                        .unwrap()
                        .astype(DType::BFloat16)
                        .unwrap()
                })
                .collect();
            let kh_refs: Vec<&MxArray> = k_hists.iter().collect();
            let vh_refs: Vec<&MxArray> = v_hists.iter().collect();

            for is_dyn in [false, true] {
                let (got_k, got_v, got_kh, got_vh) = inkling2_sconv_k4_kv_prep(
                    &k_view,
                    &v_view,
                    &kw,
                    &vw,
                    is_dyn.then_some(&dk),
                    is_dyn.then_some(&dv),
                    (num_seqs > 1).then_some(&cu_arr),
                    &kh_refs,
                    &vh_refs,
                    num_seqs,
                    false,
                    n_kv,
                    hd,
                    qk_thresh as f32,
                    1e-6,
                    1e-6,
                )
                .unwrap();

                // Reference per-sequence
                let mut ref_k_parts = Vec::new();
                let mut ref_v_parts = Vec::new();
                let mut ref_kh_parts = Vec::new();
                let mut ref_vh_parts = Vec::new();
                for s in 0..num_seqs {
                    let st = cu_vec[s] as i64;
                    let en = cu_vec[s + 1] as i64;
                    let ks = k_view.slice(&[st, 0], &[en, kv_dim as i64]).unwrap();
                    let vs = v_view.slice(&[st, 0], &[en, kv_dim as i64]).unwrap();
                    let mut kh = k_hists[s].clone();
                    let mut vh = v_hists[s].clone();
                    let (kc, vc) = if is_dyn {
                        let dks = dk.slice(&[st, 0, 0], &[en, n_kv as i64, 4]).unwrap();
                        let dvs = dv.slice(&[st, 0, 0], &[en, n_kv as i64, 4]).unwrap();
                        (
                            dynamic_short_conv(&ks, &kw, &dks, &mut kh).unwrap(),
                            dynamic_short_conv(&vs, &vw, &dvs, &mut vh).unwrap(),
                        )
                    } else {
                        (
                            static_short_conv(&ks, &kw, &mut kh).unwrap(),
                            static_short_conv(&vs, &vw, &mut vh).unwrap(),
                        )
                    };
                    let len = en - st;
                    let k3 = kc.reshape(&[len, n_kv as i64, hd as i64]).unwrap();
                    let v3 = vc.reshape(&[len, n_kv as i64, hd as i64]).unwrap();
                    ref_k_parts.push(l2_clip(&k3, None, qk_thresh, DType::BFloat16).unwrap());
                    ref_v_parts.push(
                        rms_norm_unscaled(&v3.astype(DType::Float32).unwrap(), 1e-6)
                            .unwrap()
                            .astype(DType::BFloat16)
                            .unwrap(),
                    );
                    ref_kh_parts.push(kh);
                    ref_vh_parts.push(vh);
                }
                let rk = MxArray::concatenate_many(ref_k_parts.iter().collect(), Some(0)).unwrap();
                let rv = MxArray::concatenate_many(ref_v_parts.iter().collect(), Some(0)).unwrap();
                let rkh = if num_seqs == 1 {
                    ref_kh_parts[0].clone()
                } else {
                    MxArray::stack(ref_kh_parts.iter().collect(), Some(0)).unwrap()
                };
                let rvh = if num_seqs == 1 {
                    ref_vh_parts[0].clone()
                } else {
                    MxArray::stack(ref_vh_parts.iter().collect(), Some(0)).unwrap()
                };
                assert!(
                    max_abs_diff(&got_k.to_float32().unwrap(), &rk.to_float32().unwrap()) <= 2e-2,
                    "kv_prep N={num_seqs} dyn={is_dyn} k mismatch"
                );
                assert!(
                    max_abs_diff(&got_v.to_float32().unwrap(), &rv.to_float32().unwrap()) <= 2e-2,
                    "kv_prep N={num_seqs} dyn={is_dyn} v mismatch"
                );
                assert_eq!(
                    max_abs_diff(&got_kh.to_float32().unwrap(), &rkh.to_float32().unwrap()),
                    0.0
                );
                assert_eq!(
                    max_abs_diff(&got_vh.to_float32().unwrap(), &rvh.to_float32().unwrap()),
                    0.0
                );
            }

            // Test sconv_k4_ragged with residual epilogue (res_a + res_b + conv_out)
            let c_post = 2304usize;
            let x_post = MxArray::from_float32(
                &rng.vec(total_t * c_post, 1.5),
                &[total_t as i64, c_post as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let w_post =
                MxArray::from_float32(&rng.vec(c_post * 4, 0.4), &[c_post as i64, 4]).unwrap();
            let res_a = MxArray::from_float32(
                &rng.vec(total_t * c_post, 1.5),
                &[total_t as i64, c_post as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let res_b = MxArray::from_float32(
                &rng.vec(total_t * c_post, 1.5),
                &[total_t as i64, c_post as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let p_hists: Vec<MxArray> = (0..num_seqs)
                .map(|_| {
                    MxArray::from_float32(&rng.vec(3 * c_post, 1.5), &[3, c_post as i64])
                        .unwrap()
                        .astype(DType::BFloat16)
                        .unwrap()
                })
                .collect();
            let ph_refs: Vec<&MxArray> = p_hists.iter().collect();
            let (got_y, got_h) = inkling2_sconv_k4_ragged(
                &x_post,
                &w_post,
                None,
                (num_seqs > 1).then_some(&cu_arr),
                Some(&res_a),
                Some(&res_b),
                &ph_refs,
                num_seqs,
                false,
            )
            .unwrap();

            let mut ref_y_parts = Vec::new();
            let mut ref_h_parts = Vec::new();
            for s in 0..num_seqs {
                let st = cu_vec[s] as i64;
                let en = cu_vec[s + 1] as i64;
                let xs = x_post.slice(&[st, 0], &[en, c_post as i64]).unwrap();
                let mut hs = p_hists[s].clone();
                let ys = static_short_conv(&xs, &w_post, &mut hs).unwrap();
                ref_y_parts.push(ys);
                ref_h_parts.push(hs);
            }
            let conv_ref =
                MxArray::concatenate_many(ref_y_parts.iter().collect(), Some(0)).unwrap();
            let ref_y = res_a
                .astype(DType::Float32)
                .unwrap()
                .add(&res_b.astype(DType::Float32).unwrap())
                .unwrap()
                .add(&conv_ref.astype(DType::Float32).unwrap())
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let ref_h = if num_seqs == 1 {
                ref_h_parts[0].clone()
            } else {
                MxArray::stack(ref_h_parts.iter().collect(), Some(0)).unwrap()
            };
            assert!(
                max_abs_diff(&got_y.to_float32().unwrap(), &ref_y.to_float32().unwrap()) <= 2e-2,
                "sconv_k4_ragged residual N={num_seqs} mismatch"
            );
            assert_eq!(
                max_abs_diff(&got_h.to_float32().unwrap(), &ref_h.to_float32().unwrap()),
                0.0
            );
        }
    }

    #[test]
    fn inkling2_block_attn_res_and_pair_rms_norm_match_reference() {
        if !metal_available() {
            return;
        }
        use crate::models::inkling2::attn_res::ATTN_RES_EPS;
        use crate::nn::normalization::rms_norm_unscaled;
        let mut rng = Rng(0x8080);
        let (t, d) = (6usize, 2304usize);

        let srcs: Vec<MxArray> = (0..5)
            .map(|_| {
                MxArray::from_float32(&rng.vec(t * d, 1.5), &[t as i64, d as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
            })
            .collect();
        let addend = MxArray::from_float32(&rng.vec(t * d, 1.0), &[t as i64, d as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let inv_sqrt_d = 1.0_f32 / (d as f32).sqrt();

        for (k, q, with_addend, with_norm) in [
            (1usize, 1usize, false, true),
            (5, 1, false, true),
            (4, 3, false, true),
            (2, 4, false, false),
            (3, 1, true, true),
            (3, 3, true, false),
        ] {
            let q_vals: Vec<f32> = rng
                .vec(q * d, 1.0)
                .into_iter()
                .map(|v| v * inv_sqrt_d)
                .collect();
            let queries = MxArray::from_float32(&q_vals, &[q as i64, d as i64]).unwrap();
            let nw = MxArray::from_float32(&rng.vec(q * d, 1.0), &[q as i64, d as i64]).unwrap();

            let src_refs: Vec<&MxArray> = srcs[..k].iter().collect();
            let out_dtype = if with_norm {
                DType::BFloat16
            } else {
                DType::Float32
            };
            let got = inkling2_block_attn_res(
                &src_refs,
                with_addend.then_some(&addend),
                &queries,
                with_norm.then_some(&nw),
                ATTN_RES_EPS,
                1e-6,
                q == 1,
                out_dtype,
            )
            .unwrap();

            // Reference
            let last_eff = if with_addend {
                srcs[k - 1]
                    .astype(DType::Float32)
                    .unwrap()
                    .add(&addend.astype(DType::Float32).unwrap())
                    .unwrap()
            } else {
                srcs[k - 1].clone()
            };
            let mut ref_srcs: Vec<&MxArray> = srcs[..k - 1].iter().collect();
            ref_srcs.push(&last_eff);
            // Exact CPU FP64 reference (MLX FP32 matmul on Apple10/M5 uses TF32 tensor ops with ~2e-3 error).
            let mut src_f64: Vec<Vec<f64>> = Vec::new();
            for s in &ref_srcs {
                src_f64.push(
                    s.to_float32()
                        .unwrap()
                        .iter()
                        .map(|&x| f64::from(x))
                        .collect(),
                );
            }
            let nw_f64: Vec<f64> = nw
                .to_float32()
                .unwrap()
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            let mut cpu_ref = vec![0.0_f32; q * t * d];
            for tok in 0..t {
                let mut v_norm = Vec::with_capacity(k);
                for src in src_f64.iter().take(k) {
                    let row_s = &src[tok * d..(tok + 1) * d];
                    let ms: f64 = row_s.iter().map(|x| x * x).sum::<f64>() / (d as f64);
                    let ir = 1.0 / (ms + f64::from(ATTN_RES_EPS)).sqrt();
                    v_norm.push(row_s.iter().map(|&x| x * ir).collect::<Vec<f64>>());
                }
                for qi in 0..q {
                    let mut probs = vec![0.0_f64; k];
                    if k == 1 {
                        probs[0] = 1.0;
                    } else {
                        let mut sc = vec![0.0_f64; k];
                        let mut mx = f64::NEG_INFINITY;
                        for si in 0..k {
                            let dot: f64 = (0..d)
                                .map(|j| f64::from(q_vals[qi * d + j]) * v_norm[si][j])
                                .sum();
                            sc[si] = dot;
                            mx = mx.max(dot);
                        }
                        let exps: Vec<f64> = sc.iter().map(|&x| (x - mx).exp()).collect();
                        let den: f64 = exps.iter().sum();
                        for si in 0..k {
                            probs[si] = exps[si] / den;
                        }
                    }
                    let mut mix_row = vec![0.0_f64; d];
                    for j in 0..d {
                        let mut acc = 0.0_f64;
                        for si in 0..k {
                            acc += probs[si] * v_norm[si][j];
                        }
                        mix_row[j] = acc;
                    }
                    if with_norm {
                        let ms: f64 = mix_row.iter().map(|x| x * x).sum::<f64>() / (d as f64);
                        let ir = 1.0 / (ms + 1e-6_f64).sqrt();
                        for j in 0..d {
                            cpu_ref[(qi * t + tok) * d + j] =
                                (mix_row[j] * ir * nw_f64[qi * d + j]) as f32;
                        }
                    } else {
                        for j in 0..d {
                            cpu_ref[(qi * t + tok) * d + j] = mix_row[j] as f32;
                        }
                    }
                }
            }

            let got_v = got.to_float32().unwrap();
            let diff = max_abs_diff(&got_v, &cpu_ref);
            let tol = if with_norm { 2e-2 } else { 1e-5 };
            assert!(
                diff <= tol,
                "block_attn_res K={k} Q={q} add={with_addend} norm={with_norm} diff={diff} > {tol}"
            );
        }

        // Test inkling2_pair_rms_norm (S=1 and S=3 with broadcast dest + 3D strided loop_blend_rms_norm!)
        let dest_1 = MxArray::from_float32(&rng.vec(t * d, 1.5), &[t as i64, d as i64]).unwrap();
        let msgs_4 =
            MxArray::from_float32(&rng.vec(4 * t * d, 1.5), &[4, t as i64, d as i64]).unwrap();
        let msgs_3 = msgs_4.slice(&[0, 0, 0], &[3, t as i64, d as i64]).unwrap();
        let pair_3 =
            inkling2_pair_rms_norm(&dest_1, &msgs_3, ATTN_RES_EPS, DType::Float32).unwrap();
        let pair_3_bf16 =
            inkling2_pair_rms_norm(&dest_1, &msgs_3, ATTN_RES_EPS, DType::BFloat16).unwrap();
        let pair_bf16_diff = max_abs_diff(
            &pair_3_bf16.to_float32().unwrap(),
            &pair_3
                .astype(DType::BFloat16)
                .unwrap()
                .to_float32()
                .unwrap(),
        );
        assert!(
            pair_bf16_diff <= 1e-6,
            "pair_rms_norm_bf16 mismatch diff={pair_bf16_diff}"
        );

        // Test coalesced exact-FP32 inkling2_moe_router_proj across M=1, 4, 8, 32 against exact FP32 CPU dot
        let w_vals = rng.vec(128 * 256, 0.4);
        let w_router = MxArray::from_float32(&w_vals, &[128, 256]).unwrap();
        for &m in &[1i64, 4, 8, 32] {
            let x_r = MxArray::from_float32(&rng.vec(m as usize * 256, 0.8), &[m, 256])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let got_r = inkling2_moe_router_proj(&x_r, &w_router).unwrap();
            let x_f32 = x_r.to_float32().unwrap();
            let mut ref_cpu = vec![0.0_f32; m as usize * 128];
            for r in 0..m as usize {
                for e in 0..128 {
                    let mut sum = 0.0_f64;
                    for c in 0..256 {
                        sum += f64::from(x_f32[r * 256 + c]) * f64::from(w_vals[e * 256 + c]);
                    }
                    ref_cpu[r * 128 + e] = sum as f32;
                }
            }
            let r_diff = max_abs_diff(&got_r.to_float32().unwrap(), &ref_cpu);
            assert!(r_diff <= 1e-4, "moe_router_proj M={m} diff={r_diff}");
        }
        assert_eq!(
            pair_3.shape().unwrap().as_ref(),
            &[3, t as i64, (2 * d) as i64]
        );
        let dh_ref = rms_norm_unscaled(&dest_1, ATTN_RES_EPS).unwrap();
        for s in 0..3 {
            let ms = msgs_3
                .slice(&[s, 0, 0], &[s + 1, t as i64, d as i64])
                .unwrap()
                .squeeze(Some(&[0]))
                .unwrap();
            let mh_ref = rms_norm_unscaled(&ms, ATTN_RES_EPS).unwrap();
            let ref_p = MxArray::concatenate(&dh_ref, &mh_ref, -1).unwrap();
            let got_p = pair_3
                .slice(&[s, 0, 0], &[s + 1, t as i64, (2 * d) as i64])
                .unwrap()
                .squeeze(Some(&[0]))
                .unwrap();
            let diff = max_abs_diff(&got_p.to_float32().unwrap(), &ref_p.to_float32().unwrap());
            assert!(diff <= 1e-5, "pair_rms_norm S=3 stream={s} diff={diff}");
        }

        // Test 3D strided inkling2_loop_blend_rms_norm with gf_3 [3, T, 2D] sliced without copy!
        let gf_3 =
            MxArray::from_float32(&rng.vec(3 * t * 2 * d, 2.0), &[3, t as i64, (2 * d) as i64])
                .unwrap();
        let gain_3 = gf_3.slice(&[0, 0, 0], &[3, t as i64, d as i64]).unwrap();
        let forget_3 = gf_3
            .slice(&[0, 0, d as i64], &[3, t as i64, (2 * d) as i64])
            .unwrap();
        let w_3 = MxArray::from_float32(&rng.vec(3 * d, 1.0), &[3, d as i64]).unwrap();
        let got_blend_3 = inkling2_loop_blend_rms_norm(
            &dest_1,
            &msgs_3,
            &gain_3,
            &forget_3,
            &w_3,
            1e-6,
            DType::BFloat16,
        )
        .unwrap();
        assert_eq!(
            got_blend_3.shape().unwrap().as_ref(),
            &[3, t as i64, d as i64]
        );
        for s in 0..3 {
            let ms = msgs_3
                .slice(&[s, 0, 0], &[s + 1, t as i64, d as i64])
                .unwrap()
                .squeeze(Some(&[0]))
                .unwrap();
            let gs = gain_3
                .slice(&[s, 0, 0], &[s + 1, t as i64, d as i64])
                .unwrap()
                .squeeze(Some(&[0]))
                .unwrap();
            let fs = forget_3
                .slice(&[s, 0, 0], &[s + 1, t as i64, d as i64])
                .unwrap()
                .squeeze(Some(&[0]))
                .unwrap();
            let ws = w_3
                .slice(&[s, 0], &[s + 1, d as i64])
                .unwrap()
                .squeeze(Some(&[0]))
                .unwrap();
            let ref_s =
                inkling2_loop_blend_rms_norm(&dest_1, &ms, &gs, &fs, &ws, 1e-6, DType::BFloat16)
                    .unwrap();
            let got_s = got_blend_3
                .slice(&[s, 0, 0], &[s + 1, t as i64, d as i64])
                .unwrap()
                .squeeze(Some(&[0]))
                .unwrap();
            assert_eq!(
                max_abs_diff(&got_s.to_float32().unwrap(), &ref_s.to_float32().unwrap()),
                0.0,
                "3D strided loop_blend_rms_norm stream={s} must be bit-exact"
            );
        }
    }

    #[test]
    fn inkling2_moe_route_and_combine_match_reference_including_ties_and_sort() {
        if !metal_available() {
            return;
        }
        let mut rng = Rng(0x9090);
        let (t, e, k, d) = (10usize, 128usize, 8usize, 2304usize);
        let mut logits_v = rng.vec(t * e, 3.0);
        // Inject exact score ties on row 0 across different lanes (indices 5, 37, 69, 101)!
        for &idx in &[5usize, 37, 69, 101] {
            logits_v[idx] = 2.5;
        }
        let mut bias_v = rng.vec(e, 0.2);
        for &idx in &[5usize, 37, 69, 101] {
            bias_v[idx] = 0.5;
        }
        let down_scale_v: Vec<f32> = rng.vec(e, 0.5).into_iter().map(|v| v.abs() + 0.1).collect();

        let logits = MxArray::from_float32(&logits_v, &[t as i64, e as i64]).unwrap();
        let bias = MxArray::from_float32(&bias_v, &[e as i64]).unwrap();
        let down_scale = MxArray::from_float32(&down_scale_v, &[e as i64]).unwrap();

        let (got_idx, got_w) =
            inkling2_moe_route(&logits, Some(&bias), Some(&down_scale), k, 8.0).unwrap();
        assert_eq!(got_idx.dtype().unwrap(), DType::Int32);
        assert_eq!(got_w.dtype().unwrap(), DType::Float32);

        // Reference router
        let probs = Activations::sigmoid(&logits).unwrap();
        let sel = probs.add(&bias).unwrap();
        let ref_idx = sel
            .negative()
            .unwrap()
            .argsort(Some(-1))
            .unwrap()
            .slice_axis(1, 0, k as i64)
            .unwrap();
        let ps = probs.take_along_axis(&ref_idx, -1).unwrap();
        let den = ps
            .sum(Some(&[-1]), Some(true))
            .unwrap()
            .add_scalar(1e-20)
            .unwrap();
        let ds_g = down_scale.take(&ref_idx, 0).unwrap();
        let ref_w = ps
            .div(&den)
            .unwrap()
            .mul_scalar(8.0)
            .unwrap()
            .mul(&ds_g)
            .unwrap();

        assert_eq!(
            got_idx.to_int32().unwrap().as_ref(),
            ref_idx
                .astype(DType::Int32)
                .unwrap()
                .to_int32()
                .unwrap()
                .as_ref(),
            "moe_route indices & tie-break mismatch"
        );
        let dw = max_abs_diff(&got_w.to_float32().unwrap(), &ref_w.to_float32().unwrap());
        assert!(dw <= 1e-5, "moe_route weights max_diff={dw}");

        // Test moe_combine (both unsorted and sorted with uint32 inv_order)
        let down_unsorted =
            MxArray::from_float32(&rng.vec(t * k * d, 1.5), &[t as i64, k as i64, 1, d as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
        let ref_comb = down_unsorted
            .reshape(&[t as i64, k as i64, d as i64])
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .mul(&got_w.expand_dims(-1).unwrap())
            .unwrap()
            .sum(Some(&[1]), Some(false))
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        let got_comb_unsorted =
            inkling2_moe_combine(&down_unsorted, &got_w, None, DType::BFloat16).unwrap();
        assert!(
            max_abs_diff(
                &got_comb_unsorted.to_float32().unwrap(),
                &ref_comb.to_float32().unwrap()
            ) <= 1e-2
        );

        let flat_idx = got_idx.reshape(&[-1]).unwrap();
        let order = flat_idx.argsort(Some(-1)).unwrap();
        let inv = order.argsort(Some(-1)).unwrap();
        let down_sorted = down_unsorted
            .reshape(&[(t * k) as i64, 1, d as i64])
            .unwrap()
            .take(&order, 0)
            .unwrap();
        let got_comb_sorted =
            inkling2_moe_combine(&down_sorted, &got_w, Some(&inv), DType::BFloat16).unwrap();
        assert_eq!(
            max_abs_diff(
                &got_comb_sorted.to_float32().unwrap(),
                &got_comb_unsorted.to_float32().unwrap()
            ),
            0.0,
            "sorted moe_combine must be bit-identical to unsorted"
        );
    }

    #[test]
    fn inkling2_lm_head_softcap_and_argmax_match_reference() {
        if !metal_available() {
            return;
        }
        let mut rng = Rng(0xa0a0);
        let (t, padded_v, unpadded_v) = (4usize, 4096usize, 4003usize);
        let mut lv = rng.vec(t * padded_v, 25.0);
        // Row 0: place a huge value in padded tail (must be ignored!) and a saturation tie at idx 50 & 3000!
        lv[unpadded_v + 5] = 999.0;
        lv[50] = 400.0;
        lv[3000] = 500.0; // both saturate 30*tanh(x/30) to 30.0f -> tie must pick smaller idx 50!
        let logits = MxArray::from_float32(&lv, &[t as i64, padded_v as i64]).unwrap();
        let bias = MxArray::from_float32(&rng.vec(padded_v, 5.0), &[1, padded_v as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        // 1. Target lm_head mode (no bias)
        let got_sc = inkling2_lm_head_softcap(&logits, None, unpadded_v, 30.0, false).unwrap();
        let ref_sc = logits
            .slice(&[0, 0], &[t as i64, unpadded_v as i64])
            .unwrap()
            .mul_scalar(1.0 / 30.0)
            .unwrap()
            .tanh()
            .unwrap()
            .mul_scalar(30.0)
            .unwrap();
        assert!(max_abs_diff(&got_sc.to_float32().unwrap(), &ref_sc.to_float32().unwrap()) <= 1e-5);
        let got_am =
            inkling2_lm_head_softcap_argmax(&logits, None, unpadded_v, 30.0, false).unwrap();
        let ref_am = ref_sc
            .argmax(-1, Some(false))
            .unwrap()
            .astype(DType::Int32)
            .unwrap();
        assert_eq!(
            got_am.to_int32().unwrap().as_ref(),
            ref_am.to_int32().unwrap().as_ref()
        );
        assert_eq!(got_am.to_int32().unwrap()[0], 50);

        // 2. DSpark Markov mode (BF16 base + BF16 bias with BF16 round + softcap + argmax)
        let base_bf = logits.astype(DType::BFloat16).unwrap();
        let got_m_sc =
            inkling2_lm_head_softcap(&base_bf, Some(&bias), unpadded_v, 30.0, true).unwrap();
        let ref_m_sum = base_bf
            .astype(DType::Float32)
            .unwrap()
            .add(&bias.astype(DType::Float32).unwrap())
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
            .slice(&[0, 0], &[t as i64, unpadded_v as i64])
            .unwrap()
            .astype(DType::Float32)
            .unwrap();
        let ref_m_sc = ref_m_sum
            .mul_scalar(1.0 / 30.0)
            .unwrap()
            .tanh()
            .unwrap()
            .mul_scalar(30.0)
            .unwrap();
        assert!(
            max_abs_diff(
                &got_m_sc.to_float32().unwrap(),
                &ref_m_sc.to_float32().unwrap()
            ) <= 1e-5
        );
        let got_m_am =
            inkling2_lm_head_softcap_argmax(&base_bf, Some(&bias), unpadded_v, 30.0, true).unwrap();
        let ref_m_am = ref_m_sc
            .argmax(-1, Some(false))
            .unwrap()
            .astype(DType::Int32)
            .unwrap();
        assert_eq!(
            got_m_am.to_int32().unwrap().as_ref(),
            ref_m_am.to_int32().unwrap().as_ref()
        );
    }

    #[test]
    fn inkling2_stage2_kernels_handle_non_contiguous_inputs_without_clobber() {
        if !metal_available() {
            return;
        }
        let mut rng = Rng(0xc10b);

        // 1. sconv_k4_kv_prep with strided (non-contiguous) k_hist / v_hist views (the exact
        //    DSpark `StreamShortConvTape::commit_prefix` layout from fused `qrgkv`!).
        let (t, n_kv, hd) = (8usize, 8usize, 128usize);
        let kv_dim = n_kv * hd;
        let fused_cols = 8600usize;
        let big = MxArray::from_float32(
            &rng.vec(t * fused_cols, 1.0),
            &[t as i64, fused_cols as i64],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let k_in = big
            .slice(&[0, 6552], &[t as i64, (6552 + kv_dim) as i64])
            .unwrap();
        let v_in = big
            .slice(&[0, 7576], &[t as i64, (7576 + kv_dim) as i64])
            .unwrap();
        // Non-contiguous `[3, kv_dim]` histories sliced directly out of `big` (row stride = 8600!)
        let kh_nc = big.slice(&[1, 6552], &[4, (6552 + kv_dim) as i64]).unwrap();
        let vh_nc = big.slice(&[2, 7576], &[5, (7576 + kv_dim) as i64]).unwrap();
        // Contiguous copies via reshape/astype roundtrip for reference
        let kh_c = MxArray::from_float32(&kh_nc.to_float32().unwrap(), &[3, kv_dim as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let vh_c = MxArray::from_float32(&vh_nc.to_float32().unwrap(), &[3, kv_dim as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let kw = MxArray::from_float32(&rng.vec(kv_dim * 4, 0.2), &[kv_dim as i64, 4]).unwrap();
        let vw = MxArray::from_float32(&rng.vec(kv_dim * 4, 0.2), &[kv_dim as i64, 4]).unwrap();
        let (k_got, v_got, kh_got, vh_got) = inkling2_sconv_k4_kv_prep(
            &k_in,
            &v_in,
            &kw,
            &vw,
            None,
            None,
            None,
            &[&kh_nc],
            &[&vh_nc],
            1,
            false,
            n_kv,
            hd,
            22.627417,
            1e-6,
            1e-6,
        )
        .unwrap();
        let (k_ref, v_ref, kh_ref, vh_ref) = inkling2_sconv_k4_kv_prep(
            &k_in,
            &v_in,
            &kw,
            &vw,
            None,
            None,
            None,
            &[&kh_c],
            &[&vh_c],
            1,
            false,
            n_kv,
            hd,
            22.627417,
            1e-6,
            1e-6,
        )
        .unwrap();
        assert_eq!(
            max_abs_diff(&k_got.to_float32().unwrap(), &k_ref.to_float32().unwrap()),
            0.0
        );
        assert_eq!(
            max_abs_diff(&v_got.to_float32().unwrap(), &v_ref.to_float32().unwrap()),
            0.0
        );
        assert_eq!(
            max_abs_diff(&kh_got.to_float32().unwrap(), &kh_ref.to_float32().unwrap()),
            0.0
        );
        assert_eq!(
            max_abs_diff(&vh_got.to_float32().unwrap(), &vh_ref.to_float32().unwrap()),
            0.0
        );

        // 2. block_attn_res with transposed (non-contiguous) queries & norm_weight
        let (rows, dim) = (4usize, 256usize);
        let s0 = MxArray::from_float32(&rng.vec(rows * dim, 1.0), &[rows as i64, dim as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let s1 = MxArray::from_float32(&rng.vec(rows * dim, 1.0), &[rows as i64, dim as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let q_t = MxArray::from_float32(&rng.vec(3 * dim, 0.1), &[dim as i64, 3]).unwrap();
        let q_nc = q_t.transpose(None).unwrap(); // [3, dim] non-contiguous!
        let q_c = MxArray::from_float32(&q_nc.to_float32().unwrap(), &[3, dim as i64]).unwrap();
        let nw_t = MxArray::from_float32(&rng.vec(3 * dim, 1.0), &[dim as i64, 3]).unwrap();
        let nw_nc = nw_t.transpose(None).unwrap();
        let nw_c = MxArray::from_float32(&nw_nc.to_float32().unwrap(), &[3, dim as i64]).unwrap();
        let out_nc = inkling2_block_attn_res(
            &[&s0, &s1],
            None,
            &q_nc,
            Some(&nw_nc),
            1e-6,
            1e-6,
            false,
            DType::BFloat16,
        )
        .unwrap();
        let out_c = inkling2_block_attn_res(
            &[&s0, &s1],
            None,
            &q_c,
            Some(&nw_c),
            1e-6,
            1e-6,
            false,
            DType::BFloat16,
        )
        .unwrap();
        assert_eq!(
            max_abs_diff(&out_nc.to_float32().unwrap(), &out_c.to_float32().unwrap()),
            0.0
        );

        // 3. sconv_k4_ragged with non-contiguous transposed history
        let h_t = MxArray::from_float32(&rng.vec(3 * dim, 1.0), &[dim as i64, 3])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let h_nc = h_t.transpose(None).unwrap();
        let h_c = MxArray::from_float32(&h_nc.to_float32().unwrap(), &[3, dim as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let w_s = MxArray::from_float32(&rng.vec(dim * 4, 0.2), &[dim as i64, 4]).unwrap();
        let (ry_nc, rh_nc) =
            inkling2_sconv_k4_ragged(&s0, &w_s, None, None, Some(&s1), None, &[&h_nc], 1, false)
                .unwrap();
        let (ry_c, rh_c) =
            inkling2_sconv_k4_ragged(&s0, &w_s, None, None, Some(&s1), None, &[&h_c], 1, false)
                .unwrap();
        assert_eq!(
            max_abs_diff(&ry_nc.to_float32().unwrap(), &ry_c.to_float32().unwrap()),
            0.0
        );
        assert_eq!(
            max_abs_diff(&rh_nc.to_float32().unwrap(), &rh_c.to_float32().unwrap()),
            0.0
        );
    }

    #[test]
    fn inkling2_markov_gate_matches_reference() {
        if !metal_available() {
            return;
        }
        use crate::models::inkling2::dspark::markov_gate;

        let mut rng = Rng(0x909);
        let rank = 256usize;
        for rows in [1usize, 4usize] {
            let rec = MxArray::from_float32(
                &rng.vec(rows * 3 * rank, 1.5),
                &[rows as i64, (3 * rank) as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let ctx = MxArray::from_float32(
                &rng.vec(rows * 3 * rank, 1.5),
                &[rows as i64, (3 * rank) as i64],
            )
            .unwrap();
            let state =
                MxArray::from_float32(&rng.vec(rows * rank, 1.0), &[rows as i64, rank as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();

            // Reference: `raw = f32(rec) + ctx` -> `markov_gate(&raw, &state, rank)`
            let raw = rec.astype(DType::Float32).unwrap().add(&ctx).unwrap();
            let (ref_state, ref_lat) = markov_gate(&raw, &state, rank as i64).unwrap();

            // Fused with ctx
            let (got_state, got_lat) = inkling2_markov_gate(&rec, Some(&ctx), &state).unwrap();
            assert_eq!(
                max_abs_diff(
                    &got_state.to_float32().unwrap(),
                    &ref_state.to_float32().unwrap()
                ),
                0.0,
                "markov_gate fused state mismatch (rows={rows})"
            );
            assert_eq!(
                max_abs_diff(
                    &got_lat.to_float32().unwrap(),
                    &ref_lat.to_float32().unwrap()
                ),
                0.0,
                "markov_gate fused latent mismatch (rows={rows})"
            );

            // Without ctx (pre-summed raw)
            let (got_state2, got_lat2) = inkling2_markov_gate(&raw, None, &state).unwrap();
            assert_eq!(
                max_abs_diff(
                    &got_state2.to_float32().unwrap(),
                    &ref_state.to_float32().unwrap()
                ),
                0.0,
                "markov_gate raw state mismatch (rows={rows})"
            );
            assert_eq!(
                max_abs_diff(
                    &got_lat2.to_float32().unwrap(),
                    &ref_lat.to_float32().unwrap()
                ),
                0.0,
                "markov_gate raw latent mismatch (rows={rows})"
            );

            // Non-contiguous strided ctx (slice from wider [rows, 2, 3*rank] tensor)
            let wide_ctx = MxArray::concatenate(&ctx, &ctx, -1).unwrap();
            let strided_ctx = wide_ctx
                .slice(&[0, 0], &[rows as i64, (3 * rank) as i64])
                .unwrap();
            let (got_state3, got_lat3) =
                inkling2_markov_gate(&rec, Some(&strided_ctx), &state).unwrap();
            assert_eq!(
                max_abs_diff(
                    &got_state3.to_float32().unwrap(),
                    &ref_state.to_float32().unwrap()
                ),
                0.0,
                "markov_gate strided ctx state mismatch (rows={rows})"
            );
            assert_eq!(
                max_abs_diff(
                    &got_lat3.to_float32().unwrap(),
                    &ref_lat.to_float32().unwrap()
                ),
                0.0,
                "markov_gate strided ctx latent mismatch (rows={rows})"
            );
        }
    }

    #[test]
    fn inkling2_grouped_moe_nvfp4_and_q8_match_reference() {
        if !metal_available() {
            return;
        }
        use crate::models::quantized_linear::{
            DEFAULT_QUANT_MODE, MXFP8_MODE, NVFP4_MODE, QuantizedSwitchLinear,
        };

        let mut rng = Rng(0x2026_0303);
        let e = 16usize;
        let top_k = 4usize;
        let d = 256usize;
        let inter = 128usize;
        let gate_cap = 4.0_f32;
        let up_cap = 25.0_f32;

        // Build NVFP4 weights (uint32 [E, N, K/8], uint8 scales [E, N, K/16], f32 outer [E])
        let nv_gu_w_f32 = MxArray::from_float32(
            &rng.vec(e * 2 * inter * d, 0.5),
            &[e as i64, (2 * inter) as i64, d as i64],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let nv_dn_w_f32 = MxArray::from_float32(
            &rng.vec(e * d * inter, 0.5),
            &[e as i64, d as i64, inter as i64],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let quantize_tensor = |w: &MxArray,
                               group_size: i32,
                               bits: i32,
                               mode: &std::ffi::CStr|
         -> (MxArray, MxArray, Option<MxArray>) {
            let mut out_q = std::ptr::null_mut();
            let mut out_s = std::ptr::null_mut();
            let mut out_b = std::ptr::null_mut();
            let ok = unsafe {
                mlx_sys::mlx_quantize(
                    w.as_raw_ptr(),
                    group_size,
                    bits,
                    mode.as_ptr(),
                    &mut out_q,
                    &mut out_s,
                    &mut out_b,
                )
            };
            assert!(ok, "mlx_quantize failed for {mode:?}");
            let q = MxArray::from_handle(out_q, "q").unwrap();
            let s = MxArray::from_handle(out_s, "s").unwrap();
            let b = if out_b.is_null() {
                None
            } else {
                Some(MxArray::from_handle(out_b, "b").unwrap())
            };
            (q, s, b)
        };

        let (nv_gu_w, nv_gu_s, _) = quantize_tensor(&nv_gu_w_f32, 16, 4, c"nvfp4");
        let (nv_dn_w, nv_dn_s, _) = quantize_tensor(&nv_dn_w_f32, 16, 4, c"nvfp4");
        let nv_gu_outer_vals: Vec<f32> = (0..e).map(|i| 0.5 + 0.05 * i as f32).collect();
        let nv_gu_outer = MxArray::from_float32(&nv_gu_outer_vals, &[e as i64]).unwrap();
        let nv_gu_lin = QuantizedSwitchLinear::new(
            nv_gu_w.clone(),
            nv_gu_s.clone(),
            None,
            16,
            4,
            NVFP4_MODE.to_string(),
        )
        .with_outer_scale(nv_gu_outer.clone());
        let nv_dn_lin = QuantizedSwitchLinear::new(
            nv_dn_w.clone(),
            nv_dn_s.clone(),
            None,
            16,
            4,
            NVFP4_MODE.to_string(),
        );

        // Build Q8_0 affine weights (BF16 sidecars, matching Inkling-2-Mini blk.20..23)
        let (q8_gu_w, q8_gu_s, q8_gu_b_opt) = quantize_tensor(&nv_gu_w_f32, 32, 8, c"affine");
        let (q8_dn_w, q8_dn_s, q8_dn_b_opt) = quantize_tensor(&nv_dn_w_f32, 32, 8, c"affine");
        let q8_gu_s = q8_gu_s.astype(DType::BFloat16).unwrap();
        let q8_gu_b = q8_gu_b_opt.unwrap().astype(DType::BFloat16).unwrap();
        let q8_dn_s = q8_dn_s.astype(DType::BFloat16).unwrap();
        let q8_dn_b = q8_dn_b_opt.unwrap().astype(DType::BFloat16).unwrap();
        let q8_gu_lin = QuantizedSwitchLinear::new(
            q8_gu_w.clone(),
            q8_gu_s.clone(),
            Some(q8_gu_b.clone()),
            32,
            8,
            DEFAULT_QUANT_MODE.to_string(),
        );
        let q8_dn_lin = QuantizedSwitchLinear::new(
            q8_dn_w.clone(),
            q8_dn_s.clone(),
            Some(q8_dn_b.clone()),
            32,
            8,
            DEFAULT_QUANT_MODE.to_string(),
        );

        // Build exact symmetric Q8_0 weights (native Float16 scales, biases == None,
        // matching compacted Inkling-2-Mini GGUF blk.20..23)
        let q8s_gu_s = q8_gu_s.astype(DType::Float16).unwrap();
        let q8s_dn_s = q8_dn_s.astype(DType::Float16).unwrap();
        let q8s_gu_b_ref = q8s_gu_s
            .mul_scalar(-128.0)
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        let q8s_dn_b_ref = q8s_dn_s
            .mul_scalar(-128.0)
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        let mut q8s_gu_lin = QuantizedSwitchLinear::new(
            q8_gu_w.clone(),
            q8s_gu_s.clone(),
            Some(q8s_gu_b_ref),
            32,
            8,
            DEFAULT_QUANT_MODE.to_string(),
        );
        let mut q8s_dn_lin = QuantizedSwitchLinear::new(
            q8_dn_w.clone(),
            q8s_dn_s.clone(),
            Some(q8s_dn_b_ref),
            32,
            8,
            DEFAULT_QUANT_MODE.to_string(),
        );
        assert!(q8s_gu_lin.try_compact_symmetric_q8_f16().unwrap());
        assert!(q8s_dn_lin.try_compact_symmetric_q8_f16().unwrap());
        assert!(q8s_gu_lin.is_symmetric_q8() && q8s_gu_lin.get_biases().is_none());
        assert!(q8s_dn_lin.is_symmetric_q8() && q8s_dn_lin.get_biases().is_none());

        // Build MXFP8 weights (E4M3FN uint32 [E, N, K/4] + UE8M0 uint8 scales [E, N, K/32],
        // matching SafeTensors Inkling-2-Mini L20..23)
        let (mx_gu_w, mx_gu_s, _) = quantize_tensor(&nv_gu_w_f32, 32, 8, c"mxfp8");
        let (mx_dn_w, mx_dn_s, _) = quantize_tensor(&nv_dn_w_f32, 32, 8, c"mxfp8");
        let mx_gu_lin = QuantizedSwitchLinear::new(
            mx_gu_w.clone(),
            mx_gu_s.clone(),
            None,
            32,
            8,
            MXFP8_MODE.to_string(),
        );
        let mx_dn_lin = QuantizedSwitchLinear::new(
            mx_dn_w.clone(),
            mx_dn_s.clone(),
            None,
            32,
            8,
            MXFP8_MODE.to_string(),
        );

        for &rows in &[1usize, 4, 5, 6, 8, 16, 32, 64, 128] {
            if !inkling2_grouped_moe_available(rows) {
                continue;
            }
            let x = MxArray::from_float32(&rng.vec(rows * d, 0.8), &[rows as i64, d as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();

            // Construct valid distinct top_k expert IDs per token (with deliberate expert reuse across rows!)
            let mut idx_vec = Vec::with_capacity(rows * top_k);
            let mut gam_vec = Vec::with_capacity(rows * top_k);
            for r in 0..rows {
                // Force hot experts 0..4 to hit >4 rows when rows >= 8 (exercising NAX path)
                // AND sparse experts to hit 1..4 rows (exercising SIMD QMV path)!
                let base = if r % 3 == 0 { 0 } else { (r * 3) % (e - top_k) };
                for k in 0..top_k {
                    idx_vec.push((base + k) as i32);
                    gam_vec.push(0.25 + 0.1 * k as f32);
                }
            }
            let topk = MxArray::from_int32(&idx_vec, &[rows as i64, top_k as i64]).unwrap();
            let gamma = MxArray::from_float32(&gam_vec, &[rows as i64, top_k as i64]).unwrap();

            // Reference helper using MLX gather_qmm + inkling2_situ + inkling2_moe_combine
            let run_ref = |gu: &QuantizedSwitchLinear,
                           dn: &QuantizedSwitchLinear,
                           interleaved: bool|
             -> Vec<f32> {
                let x_4d = x.reshape(&[rows as i64, 1, 1, d as i64]).unwrap();
                let gu_out = gu.forward_without_outer_scale(&x_4d, &topk, false).unwrap();
                let outer_arg = gu.outer_scale().map(|s| (s, &topk));
                let act = inkling2_situ(&gu_out, outer_arg, gate_cap, up_cap, interleaved).unwrap();
                let dn_out = dn.forward_without_outer_scale(&act, &topk, false).unwrap();
                let comb = inkling2_moe_combine(&dn_out, &gamma, None, DType::BFloat16).unwrap();
                comb.to_float32().unwrap().as_ref().to_vec()
            };

            let err_stats = |a: &[f32], b: &[f32]| -> (f32, f32, f32) {
                let mut max_abs = 0.0_f32;
                let mut max_rel = 0.0_f32;
                let mut sum_sq = 0.0_f64;
                for (&x, &y) in a.iter().zip(b.iter()) {
                    let abs_e = (x - y).abs();
                    let denom = x.abs().max(y.abs()).max(1.0_f32);
                    max_abs = max_abs.max(abs_e);
                    max_rel = max_rel.max(abs_e / denom);
                    sum_sq += f64::from(abs_e) * f64::from(abs_e);
                }
                let rmse = (sum_sq / a.len().max(1) as f64).sqrt() as f32;
                (max_abs, max_rel, rmse)
            };

            for &interleaved in &[false, true] {
                // 1. NVFP4 parity (split & interleaved)
                let ref_nv = run_ref(&nv_gu_lin, &nv_dn_lin, interleaved);
                let got_nv = inkling2_grouped_moe(
                    &x,
                    &topk,
                    &gamma,
                    &nv_gu_w,
                    &nv_gu_s,
                    Some(&nv_gu_outer),
                    &nv_dn_w,
                    &nv_dn_s,
                    None,
                    Inkling2MoeQuantMode::Nvfp4,
                    interleaved,
                    gate_cap,
                    up_cap,
                    true,
                )
                .unwrap()
                .to_float32()
                .unwrap();
                let (abs_nv, rel_nv, rmse_nv) = err_stats(got_nv.as_ref(), &ref_nv);
                eprintln!(
                    "  [grouped_moe NVFP4 il={interleaved} M={rows:3}] max_abs={abs_nv:.5}, max_rel={rel_nv:.5}, rmse={rmse_nv:.6}"
                );
                assert!(
                    rmse_nv <= 0.01 && rel_nv <= 0.05,
                    "NVFP4 grouped MoE mismatch at il={interleaved} M={rows}: max_abs={abs_nv}, max_rel={rel_nv}, rmse={rmse_nv}"
                );

                // 2. Q8_0 parity (split & interleaved)
                let ref_q8 = run_ref(&q8_gu_lin, &q8_dn_lin, interleaved);
                let got_q8 = inkling2_grouped_moe(
                    &x,
                    &topk,
                    &gamma,
                    &q8_gu_w,
                    &q8_gu_s,
                    Some(&q8_gu_b),
                    &q8_dn_w,
                    &q8_dn_s,
                    Some(&q8_dn_b),
                    Inkling2MoeQuantMode::Q8,
                    interleaved,
                    gate_cap,
                    up_cap,
                    true,
                )
                .unwrap()
                .to_float32()
                .unwrap();
                let (abs_q8, rel_q8, rmse_q8) = err_stats(got_q8.as_ref(), &ref_q8);
                eprintln!(
                    "  [grouped_moe Q8_0  il={interleaved} M={rows:3}] max_abs={abs_q8:.5}, max_rel={rel_q8:.5}, rmse={rmse_q8:.6}"
                );
                assert!(
                    rmse_q8 <= 0.01 && rel_q8 <= 0.05,
                    "Q8_0 grouped MoE mismatch at il={interleaved} M={rows}: max_abs={abs_q8}, max_rel={rel_q8}, rmse={rmse_q8}"
                );

                // 2b. Q8_0 symmetric F16 scales (zero biases, split & interleaved)
                let ref_q8s = run_ref(&q8s_gu_lin, &q8s_dn_lin, interleaved);
                let got_q8s = inkling2_grouped_moe(
                    &x,
                    &topk,
                    &gamma,
                    &q8_gu_w,
                    &q8s_gu_s,
                    None,
                    &q8_dn_w,
                    &q8s_dn_s,
                    None,
                    Inkling2MoeQuantMode::Q8,
                    interleaved,
                    gate_cap,
                    up_cap,
                    true,
                )
                .unwrap()
                .to_float32()
                .unwrap();
                let (abs_q8s, rel_q8s, rmse_q8s) = err_stats(got_q8s.as_ref(), &ref_q8s);
                eprintln!(
                    "  [grouped_moe Q8SYM il={interleaved} M={rows:3}] max_abs={abs_q8s:.5}, max_rel={rel_q8s:.5}, rmse={rmse_q8s:.6}"
                );
                assert!(
                    rmse_q8s <= 0.01 && rel_q8s <= 0.05,
                    "Q8_0 symmetric grouped MoE mismatch at il={interleaved} M={rows}: max_abs={abs_q8s}, max_rel={rel_q8s}, rmse={rmse_q8s}"
                );

                // 3. MXFP8 parity (split & interleaved, M <= 32)
                if rows <= 32 {
                    let ref_mx = run_ref(&mx_gu_lin, &mx_dn_lin, interleaved);
                    let got_mx = inkling2_grouped_moe(
                        &x,
                        &topk,
                        &gamma,
                        &mx_gu_w,
                        &mx_gu_s,
                        None,
                        &mx_dn_w,
                        &mx_dn_s,
                        None,
                        Inkling2MoeQuantMode::Mxfp8,
                        interleaved,
                        gate_cap,
                        up_cap,
                        true,
                    )
                    .unwrap()
                    .to_float32()
                    .unwrap();
                    let (abs_mx, rel_mx, rmse_mx) = err_stats(got_mx.as_ref(), &ref_mx);
                    eprintln!(
                        "  [grouped_moe MXFP8 il={interleaved} M={rows:3}] max_abs={abs_mx:.5}, max_rel={rel_mx:.5}, rmse={rmse_mx:.6}"
                    );
                    assert!(
                        rmse_mx <= 0.01 && rel_mx <= 0.05,
                        "MXFP8 grouped MoE mismatch at il={interleaved} M={rows}: max_abs={abs_mx}, max_rel={rel_mx}, rmse={rmse_mx}"
                    );
                }
            }
        }
    }

    #[test]
    fn inkling2_grouped_moe_fused_matches_unfused_bit_exact() {
        if !inkling2_grouped_moe_available(1) {
            return;
        }
        let mut rng = Rng(0xB0E5_F05E);
        let d = 128usize;
        let inter = 128usize;
        let e = 16usize;
        let top_k = 4usize;
        let gate_cap = 4.0_f32;
        let up_cap = 25.0_f32;
        let route_scale = 8.0_f32;

        let gu_w_f32 = MxArray::from_float32(
            &rng.vec(e * (2 * inter) * d, 0.3),
            &[e as i64, (2 * inter) as i64, d as i64],
        )
        .unwrap();
        let dn_w_f32 = MxArray::from_float32(
            &rng.vec(e * d * inter, 0.3),
            &[e as i64, d as i64, inter as i64],
        )
        .unwrap();

        let quantize_tensor = |w: &MxArray, gs: i32, bits: i32, mode: &std::ffi::CStr| {
            let mut out_q = std::ptr::null_mut();
            let mut out_s = std::ptr::null_mut();
            let mut out_b = std::ptr::null_mut();
            unsafe {
                mlx_sys::mlx_quantize(
                    w.as_raw_ptr(),
                    gs,
                    bits,
                    mode.as_ptr(),
                    &mut out_q,
                    &mut out_s,
                    &mut out_b,
                );
            }
            let q = MxArray::from_handle(out_q, "q").unwrap();
            let s = MxArray::from_handle(out_s, "s").unwrap();
            let b = if out_b.is_null() {
                None
            } else {
                Some(MxArray::from_handle(out_b, "b").unwrap())
            };
            (q, s, b)
        };

        let (nv_gu_w, nv_gu_s, _) = quantize_tensor(&gu_w_f32, 16, 4, c"nvfp4");
        let (nv_dn_w, nv_dn_s, _) = quantize_tensor(&dn_w_f32, 16, 4, c"nvfp4");
        let nv_gu_outer = MxArray::from_float32(
            &(0..e).map(|i| 0.5 + 0.05 * i as f32).collect::<Vec<_>>(),
            &[e as i64],
        )
        .unwrap();
        let nv_dn_outer = MxArray::from_float32(
            &(0..e).map(|i| 0.7 + 0.03 * i as f32).collect::<Vec<_>>(),
            &[e as i64],
        )
        .unwrap();

        let (q8_gu_w, q8_gu_s, q8_gu_b_opt) = quantize_tensor(&gu_w_f32, 32, 8, c"affine");
        let (q8_dn_w, q8_dn_s, q8_dn_b_opt) = quantize_tensor(&dn_w_f32, 32, 8, c"affine");
        let q8_gu_s = q8_gu_s.astype(DType::BFloat16).unwrap();
        let q8_gu_b = q8_gu_b_opt.unwrap().astype(DType::BFloat16).unwrap();
        let q8_dn_s = q8_dn_s.astype(DType::BFloat16).unwrap();
        let q8_dn_b = q8_dn_b_opt.unwrap().astype(DType::BFloat16).unwrap();

        let q8s_gu_s = q8_gu_s.astype(DType::Float16).unwrap();
        let q8s_dn_s = q8_dn_s.astype(DType::Float16).unwrap();

        let (mx_gu_w, mx_gu_s, _) = quantize_tensor(&gu_w_f32, 32, 8, c"mxfp8");
        let (mx_dn_w, mx_dn_s, _) = quantize_tensor(&dn_w_f32, 32, 8, c"mxfp8");

        let router_w = MxArray::from_float32(&rng.vec(e * d, 0.2), &[e as i64, d as i64]).unwrap();
        let router_bias = MxArray::from_float32(&rng.vec(e, 0.25), &[e as i64]).unwrap();
        let post_conv_w = MxArray::from_float32(&rng.vec(d * 4, 0.35), &[d as i64, 4]).unwrap();

        // Test (rows, cu_seqlens) covering M=1, M=4 (4x1 AR), M=5, M=6, M=8 (1x8 verify),
        // M=16 (4x4 verify), M=32 (4x8 verify)
        let cases: &[(usize, &[usize])] = &[
            (1, &[0, 1]),
            (4, &[0, 1, 2, 3, 4]),
            (5, &[0, 2, 5]),
            (6, &[0, 3, 6]),
            (8, &[0, 8]),
            (16, &[0, 4, 8, 12, 16]),
            (32, &[0, 8, 16, 24, 32]),
        ];

        for &(rows, cu) in cases {
            if !inkling2_grouped_moe_available(rows) {
                continue;
            }
            let num_seqs = cu.len() - 1;
            let x = MxArray::from_float32(&rng.vec(rows * d, 0.8), &[rows as i64, d as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let logits = inkling2_moe_router_proj(&x, &router_w).unwrap();
            let res = MxArray::from_float32(&rng.vec(rows * d, 0.6), &[rows as i64, d as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let cu_i32: Vec<i32> = cu.iter().map(|&v| v as i32).collect();
            let cu_arr = MxArray::from_int32(&cu_i32, &[cu_i32.len() as i64]).unwrap();
            let hists: Vec<MxArray> = (0..num_seqs)
                .map(|_| {
                    MxArray::from_float32(&rng.vec(3 * d, 0.5), &[3, d as i64])
                        .unwrap()
                        .astype(DType::BFloat16)
                        .unwrap()
                })
                .collect();
            let h_refs: Vec<&MxArray> = hists.iter().collect();

            for &(quant_mode, sym_q8) in &[
                (Inkling2MoeQuantMode::Nvfp4, false),
                (Inkling2MoeQuantMode::Q8, false),
                (Inkling2MoeQuantMode::Q8, true),
                (Inkling2MoeQuantMode::Mxfp8, false),
            ] {
                let (gu_w, gu_s, gu_aux, dn_w, dn_s, dn_aux, dn_outer) = match (quant_mode, sym_q8)
                {
                    (Inkling2MoeQuantMode::Nvfp4, _) => (
                        &nv_gu_w,
                        &nv_gu_s,
                        Some(&nv_gu_outer),
                        &nv_dn_w,
                        &nv_dn_s,
                        None,
                        Some(&nv_dn_outer),
                    ),
                    (Inkling2MoeQuantMode::Q8, false) => (
                        &q8_gu_w,
                        &q8_gu_s,
                        Some(&q8_gu_b),
                        &q8_dn_w,
                        &q8_dn_s,
                        Some(&q8_dn_b),
                        None,
                    ),
                    (Inkling2MoeQuantMode::Q8, true) => {
                        (&q8_gu_w, &q8s_gu_s, None, &q8_dn_w, &q8s_dn_s, None, None)
                    }
                    (Inkling2MoeQuantMode::Mxfp8, _) => {
                        (&mx_gu_w, &mx_gu_s, None, &mx_dn_w, &mx_dn_s, None, None)
                    }
                };

                for &interleaved in &[false, true] {
                    // Unfused reference: inkling2_moe_route -> inkling2_grouped_moe -> inkling2_sconv_k4_ragged
                    let (topk, gamma) = inkling2_moe_route(
                        &logits,
                        Some(&router_bias),
                        dn_outer,
                        top_k,
                        route_scale,
                    )
                    .unwrap();
                    let moe_ref = inkling2_grouped_moe(
                        &x,
                        &topk,
                        &gamma,
                        gu_w,
                        gu_s,
                        gu_aux,
                        dn_w,
                        dn_s,
                        dn_aux,
                        quant_mode,
                        interleaved,
                        gate_cap,
                        up_cap,
                        true,
                    )
                    .unwrap();
                    let (y_ref, hist_ref) = inkling2_sconv_k4_ragged(
                        &moe_ref,
                        &post_conv_w,
                        None,
                        if num_seqs > 1 { Some(&cu_arr) } else { None },
                        Some(&res),
                        None,
                        &h_refs,
                        num_seqs,
                        false,
                    )
                    .unwrap();

                    for &router_is_weight in &[false, true] {
                        let router_in = if router_is_weight { &router_w } else { &logits };
                        for &emit_tape in &[false, true] {
                            let (y_got, hist_got, tape_got) = inkling2_grouped_moe_fused(
                                &x,
                                router_in,
                                Some(&router_bias),
                                dn_outer,
                                gu_w,
                                gu_s,
                                gu_aux,
                                dn_w,
                                dn_s,
                                dn_aux,
                                &post_conv_w,
                                Some(&res),
                                cu,
                                &h_refs,
                                num_seqs,
                                false,
                                top_k,
                                route_scale,
                                quant_mode,
                                interleaved,
                                gate_cap,
                                up_cap,
                                true,
                                emit_tape,
                                router_is_weight,
                            )
                            .unwrap();

                            assert_eq!(
                                y_got.to_float32().unwrap().as_ref(),
                                y_ref.to_float32().unwrap().as_ref(),
                                "fused MoE y mismatch (mode={quant_mode:?}, sym_q8={sym_q8}, rw={router_is_weight}, il={interleaved}, rows={rows}, emit_tape={emit_tape})"
                            );
                            assert_eq!(
                                hist_got.to_float32().unwrap().as_ref(),
                                hist_ref.to_float32().unwrap().as_ref(),
                                "fused MoE hist mismatch (mode={quant_mode:?}, sym_q8={sym_q8}, rw={router_is_weight}, il={interleaved}, rows={rows}, emit_tape={emit_tape})"
                            );
                            if emit_tape {
                                let tape_arr = tape_got.expect("tape must be returned");
                                assert_eq!(
                                    tape_arr.to_float32().unwrap().as_ref(),
                                    moe_ref.to_float32().unwrap().as_ref(),
                                    "fused MoE tape mismatch (mode={quant_mode:?}, sym_q8={sym_q8}, rw={router_is_weight}, il={interleaved}, rows={rows})"
                                );
                            } else {
                                assert!(tape_got.is_none());
                            }
                        }
                    }

                    // Batch-size invariance: row 0 of M>1 (seq 0) MUST match M=1 bit-for-bit!
                    if rows > 1 {
                        let x0 = x.slice(&[0, 0], &[1, d as i64]).unwrap();
                        let r0 = res.slice(&[0, 0], &[1, d as i64]).unwrap();
                        let (y0, _, _) = inkling2_grouped_moe_fused(
                            &x0,
                            &router_w,
                            Some(&router_bias),
                            dn_outer,
                            gu_w,
                            gu_s,
                            gu_aux,
                            dn_w,
                            dn_s,
                            dn_aux,
                            &post_conv_w,
                            Some(&r0),
                            &[0, 1],
                            &[&hists[0]],
                            1,
                            false,
                            top_k,
                            route_scale,
                            quant_mode,
                            interleaved,
                            gate_cap,
                            up_cap,
                            true,
                            false,
                            true,
                        )
                        .unwrap();
                        assert_eq!(
                            &y_ref.to_float32().unwrap()[..d],
                            y0.to_float32().unwrap().as_ref(),
                            "batch-size invariance failed between M=1 and M={rows} (mode={quant_mode:?}, sym_q8={sym_q8}, il={interleaved})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn inkling2_loop_ctrl_down_and_up_blend_norm_match_exact_cpu_reference() {
        if !metal_available() {
            return;
        }
        use crate::models::inkling2::attn_res::{
            ATTN_RES_EPS, LoopController, LoopLayerControllers,
        };

        let mut rng = Rng(0x9191);
        let d = 2304usize;
        let rank = 64usize;
        let norm_eps = 1e-6_f32;

        // Exact FP64 CPU reference for one stream `[T, D]`.
        let cpu_loop_ctrl_stream = |dest: &[f32],
                                    msg: &[f32],
                                    down_w: &[f32],
                                    gf_w: &[f32],
                                    nw: &[f32],
                                    t: usize|
         -> (Vec<f32>, Vec<f32>) {
            let mut h_ref = vec![0.0_f32; t * rank];
            let mut out_ref = vec![0.0_f32; t * d];
            for row in 0..t {
                let d_row = &dest[row * d..(row + 1) * d];
                let m_row = &msg[row * d..(row + 1) * d];
                let sq_d: f64 = d_row.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
                let sq_m: f64 = m_row.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
                let inv_d = 1.0 / (sq_d / (d as f64) + f64::from(ATTN_RES_EPS)).sqrt();
                let inv_m = 1.0 / (sq_m / (d as f64) + f64::from(ATTN_RES_EPS)).sqrt();
                for r in 0..rank {
                    let wr = &down_w[r * 2 * d..(r + 1) * 2 * d];
                    let mut acc = 0.0_f64;
                    for j in 0..d {
                        acc += (f64::from(d_row[j]) * inv_d) * f64::from(wr[j]);
                        acc += (f64::from(m_row[j]) * inv_m) * f64::from(wr[d + j]);
                    }
                    h_ref[row * rank + r] = acc as f32;
                }
                let mut blended = vec![0.0_f64; d];
                let mut sum_sq = 0.0_f64;
                for j in 0..d {
                    let mut g_raw = 0.0_f64;
                    let mut f_raw = 0.0_f64;
                    for r in 0..rank {
                        let hv = f64::from(h_ref[row * rank + r]);
                        g_raw += hv * f64::from(gf_w[j * rank + r]);
                        f_raw += hv * f64::from(gf_w[(d + j) * rank + r]);
                    }
                    let g = g_raw.clamp(-20.0, 20.0).tanh();
                    let f = f_raw.clamp(-20.0, 20.0).tanh();
                    let b = (1.0 - f) * f64::from(d_row[j]) + g * f64::from(m_row[j]);
                    blended[j] = b;
                    sum_sq += b * b;
                }
                let inv_rms = 1.0 / (sum_sq / (d as f64) + f64::from(norm_eps)).sqrt();
                for j in 0..d {
                    out_ref[row * d + j] = (blended[j] * inv_rms * f64::from(nw[j])) as f32;
                }
            }
            (h_ref, out_ref)
        };

        // Test S=1 (MLP) across T in {1, 4, 8, 9, 32} and both F32 + BF16 outputs.
        let down_1_v = rng.vec(rank * 2 * d, 0.05);
        let gf_1_v = rng.vec(2 * d * rank, 0.05);
        let nw_1_v = rng.vec(d, 1.0);
        let down_1 = MxArray::from_float32(&down_1_v, &[rank as i64, (2 * d) as i64]).unwrap();
        let gf_1 = MxArray::from_float32(&gf_1_v, &[(2 * d) as i64, rank as i64]).unwrap();
        let nw_1 = MxArray::from_float32(&nw_1_v, &[d as i64]).unwrap();

        for &t in &[1usize, 4, 8, 9, 32] {
            let dest_v = rng.vec(t * d, 1.2);
            let msg_v = rng.vec(t * d, 1.2);
            let dest = MxArray::from_float32(&dest_v, &[t as i64, d as i64]).unwrap();
            let msg = MxArray::from_float32(&msg_v, &[t as i64, d as i64]).unwrap();

            let h_got = inkling2_loop_ctrl_down(&dest, &msg, &down_1, ATTN_RES_EPS).unwrap();
            assert_eq!(h_got.shape().unwrap().as_ref(), &[t as i64, rank as i64]);
            let y_f32 = inkling2_loop_ctrl_up_blend_norm(
                &h_got,
                &gf_1,
                &dest,
                &msg,
                &nw_1,
                norm_eps,
                DType::Float32,
            )
            .unwrap();
            let y_bf16 = inkling2_loop_ctrl_up_blend_norm(
                &h_got,
                &gf_1,
                &dest,
                &msg,
                &nw_1,
                norm_eps,
                DType::BFloat16,
            )
            .unwrap();

            let (h_ref, y_ref) =
                cpu_loop_ctrl_stream(&dest_v, &msg_v, &down_1_v, &gf_1_v, &nw_1_v, t);
            let h_diff = max_abs_diff(&h_got.to_float32().unwrap(), &h_ref);
            assert!(h_diff <= 1e-5, "loop_ctrl_down S=1 T={t} h_diff={h_diff}");
            let y_diff = max_abs_diff(&y_f32.to_float32().unwrap(), &y_ref);
            assert!(
                y_diff <= 1e-5,
                "loop_ctrl_up_blend_norm S=1 T={t} y_diff={y_diff}"
            );
            let y_bf16_v = y_bf16.to_float32().unwrap();
            let y_f32_to_bf16 = y_f32.astype(DType::BFloat16).unwrap().to_float32().unwrap();
            assert_eq!(
                y_bf16_v.as_ref(),
                y_f32_to_bf16.as_ref(),
                "loop_ctrl_up_blend_norm BF16 must be bit-exact vs F32->BF16 for T={t}"
            );

            // Bit-exact (0-ULP) batch-size invariance: row 0 of T>1 MUST match T=1 bit-for-bit!
            if t > 1 {
                let d0 = dest.slice(&[0, 0], &[1, d as i64]).unwrap();
                let m0 = msg.slice(&[0, 0], &[1, d as i64]).unwrap();
                let h0 = inkling2_loop_ctrl_down(&d0, &m0, &down_1, ATTN_RES_EPS).unwrap();
                let y0 = inkling2_loop_ctrl_up_blend_norm(
                    &h0,
                    &gf_1,
                    &d0,
                    &m0,
                    &nw_1,
                    norm_eps,
                    DType::Float32,
                )
                .unwrap();
                assert_eq!(
                    &h_got.to_float32().unwrap()[..rank],
                    h0.to_float32().unwrap().as_ref(),
                    "loop_ctrl_down row 0 must be bit-exact between T=1 and T={t}"
                );
                assert_eq!(
                    &y_f32.to_float32().unwrap()[..d],
                    y0.to_float32().unwrap().as_ref(),
                    "loop_ctrl_up_blend_norm row 0 must be bit-exact between T=1 and T={t}"
                );
            }
        }

        // Test S=3 (Q/K/V) with BOTH S_dest=1 broadcast (SWA) AND S_dest=3 (Global) + sliced msgs[0..3] from [4, T, D]!
        let down_3_v = rng.vec(3 * rank * 2 * d, 0.05);
        let gf_3_v = rng.vec(3 * 2 * d * rank, 0.05);
        let nw_1b_v = rng.vec(d, 1.0);
        let nw_3_v = rng.vec(3 * d, 1.0);
        let down_3 = MxArray::from_float32(&down_3_v, &[3, rank as i64, (2 * d) as i64]).unwrap();
        let gf_3 = MxArray::from_float32(&gf_3_v, &[3, (2 * d) as i64, rank as i64]).unwrap();
        let nw_1b = MxArray::from_float32(&nw_1b_v, &[d as i64]).unwrap();
        let nw_3 = MxArray::from_float32(&nw_3_v, &[3, d as i64]).unwrap();

        for &t in &[1usize, 8usize] {
            let msgs_4_v = rng.vec(4 * t * d, 1.2);
            let msgs_4 = MxArray::from_float32(&msgs_4_v, &[4, t as i64, d as i64]).unwrap();
            let msgs_3 = msgs_4.slice(&[0, 0, 0], &[3, t as i64, d as i64]).unwrap();

            // SWA: broadcast dest [T, D] and norm_weight [D]
            let dest_b_v = rng.vec(t * d, 1.2);
            let dest_b = MxArray::from_float32(&dest_b_v, &[t as i64, d as i64]).unwrap();
            let h_swa = inkling2_loop_ctrl_down(&dest_b, &msgs_3, &down_3, ATTN_RES_EPS).unwrap();
            let y_swa = inkling2_loop_ctrl_up_blend_norm(
                &h_swa,
                &gf_3,
                &dest_b,
                &msgs_3,
                &nw_1b,
                norm_eps,
                DType::Float32,
            )
            .unwrap();
            let h_swa_v = h_swa.to_float32().unwrap();
            let y_swa_v = y_swa.to_float32().unwrap();
            for s in 0..3 {
                let (hr, yr) = cpu_loop_ctrl_stream(
                    &dest_b_v,
                    &msgs_4_v[s * t * d..(s + 1) * t * d],
                    &down_3_v[s * rank * 2 * d..(s + 1) * rank * 2 * d],
                    &gf_3_v[s * rank * 2 * d..(s + 1) * rank * 2 * d],
                    &nw_1b_v,
                    t,
                );
                let hd = max_abs_diff(&h_swa_v[s * t * rank..(s + 1) * t * rank], &hr);
                let yd = max_abs_diff(&y_swa_v[s * t * d..(s + 1) * t * d], &yr);
                assert!(hd <= 1e-5, "SWA S=3 T={t} stream={s} h_diff={hd}");
                assert!(yd <= 1e-5, "SWA S=3 T={t} stream={s} y_diff={yd}");
            }

            // Global: 3-stream dest [3, T, D] and norm_weight [3, D]
            let dest_g_v = rng.vec(3 * t * d, 1.2);
            let dest_g = MxArray::from_float32(&dest_g_v, &[3, t as i64, d as i64]).unwrap();
            let h_glb = inkling2_loop_ctrl_down(&dest_g, &msgs_3, &down_3, ATTN_RES_EPS).unwrap();
            let y_glb = inkling2_loop_ctrl_up_blend_norm(
                &h_glb,
                &gf_3,
                &dest_g,
                &msgs_3,
                &nw_3,
                norm_eps,
                DType::Float32,
            )
            .unwrap();
            let h_glb_v = h_glb.to_float32().unwrap();
            let y_glb_v = y_glb.to_float32().unwrap();
            for s in 0..3 {
                let (hr, yr) = cpu_loop_ctrl_stream(
                    &dest_g_v[s * t * d..(s + 1) * t * d],
                    &msgs_4_v[s * t * d..(s + 1) * t * d],
                    &down_3_v[s * rank * 2 * d..(s + 1) * rank * 2 * d],
                    &gf_3_v[s * rank * 2 * d..(s + 1) * rank * 2 * d],
                    &nw_3_v[s * d..(s + 1) * d],
                    t,
                );
                let hd = max_abs_diff(&h_glb_v[s * t * rank..(s + 1) * t * rank], &hr);
                let yd = max_abs_diff(&y_glb_v[s * t * d..(s + 1) * t * d], &yr);
                assert!(hd <= 1e-5, "Global S=3 T={t} stream={s} h_diff={hd}");
                assert!(yd <= 1e-5, "Global S=3 T={t} stream={s} y_diff={yd}");
            }
        }

        // End-to-end LoopLayerControllers packing test (verifies LoopController::new + LoopLayerControllers::new layout)
        let make_ctrl = |rng: &mut Rng| {
            let dw =
                MxArray::from_float32(&rng.vec(rank * 2 * d, 0.05), &[rank as i64, (2 * d) as i64])
                    .unwrap();
            let gw =
                MxArray::from_float32(&rng.vec(d * rank, 0.05), &[d as i64, rank as i64]).unwrap();
            let fw =
                MxArray::from_float32(&rng.vec(d * rank, 0.05), &[d as i64, rank as i64]).unwrap();
            LoopController::new(dw, gw, fw).unwrap()
        };
        let msg_q = MxArray::from_float32(&rng.vec(4 * d, 0.05), &[4, d as i64]).unwrap();
        let ctrls = LoopLayerControllers::new(
            msg_q,
            make_ctrl(&mut rng),
            make_ctrl(&mut rng),
            make_ctrl(&mut rng),
            make_ctrl(&mut rng),
        )
        .unwrap();
        let t_e2e = 8usize;
        let dest_e2e =
            MxArray::from_float32(&rng.vec(t_e2e * d, 1.0), &[t_e2e as i64, d as i64]).unwrap();
        let msgs_e2e =
            MxArray::from_float32(&rng.vec(3 * t_e2e * d, 1.0), &[3, t_e2e as i64, d as i64])
                .unwrap();
        let (xq, xk, xv) = ctrls
            .forward_qkv_fused_rms_norm(&dest_e2e, &msgs_e2e, &nw_1b, norm_eps, DType::Float32)
            .unwrap();
        let ref_q = ctrls
            .q_ctrl
            .forward_fused_rms_norm_with_hats(
                &dest_e2e,
                &msgs_e2e
                    .slice(&[0, 0, 0], &[1, t_e2e as i64, d as i64])
                    .unwrap()
                    .squeeze(Some(&[0]))
                    .unwrap(),
                None,
                None,
                &nw_1b,
                norm_eps,
                DType::Float32,
            )
            .unwrap();
        let ref_k = ctrls
            .k_ctrl
            .forward_fused_rms_norm_with_hats(
                &dest_e2e,
                &msgs_e2e
                    .slice(&[1, 0, 0], &[2, t_e2e as i64, d as i64])
                    .unwrap()
                    .squeeze(Some(&[0]))
                    .unwrap(),
                None,
                None,
                &nw_1b,
                norm_eps,
                DType::Float32,
            )
            .unwrap();
        let ref_v = ctrls
            .v_ctrl
            .forward_fused_rms_norm_with_hats(
                &dest_e2e,
                &msgs_e2e
                    .slice(&[2, 0, 0], &[3, t_e2e as i64, d as i64])
                    .unwrap()
                    .squeeze(Some(&[0]))
                    .unwrap(),
                None,
                None,
                &nw_1b,
                norm_eps,
                DType::Float32,
            )
            .unwrap();
        // MLX FP32 matmul on M5 uses TF32 (~1e-3), so compare within 5e-3 for end-to-end layout check.
        assert!(max_abs_diff(&xq.to_float32().unwrap(), &ref_q.to_float32().unwrap()) <= 5e-3);
        assert!(max_abs_diff(&xk.to_float32().unwrap(), &ref_k.to_float32().unwrap()) <= 5e-3);
        assert!(max_abs_diff(&xv.to_float32().unwrap(), &ref_v.to_float32().unwrap()) <= 5e-3);

        // Native BF16 LoopController weight path (used by SafeTensors): verify BF16 weights produce
        // identical FP32 accumulation to casting BF16 weights -> FP32 (0-ULP bit-exact!) for S=1 and S=3.
        let down_1_bf16 = down_1.astype(DType::BFloat16).unwrap();
        let gf_1_bf16 = gf_1.astype(DType::BFloat16).unwrap();
        let down_1_f32_from_bf16 = down_1_bf16.astype(DType::Float32).unwrap();
        let gf_1_f32_from_bf16 = gf_1_bf16.astype(DType::Float32).unwrap();
        for &t in &[1usize, 4, 8, 16] {
            let dest = MxArray::from_float32(&rng.vec(t * d, 1.2), &[t as i64, d as i64]).unwrap();
            let msg = MxArray::from_float32(&rng.vec(t * d, 1.2), &[t as i64, d as i64]).unwrap();
            let h_bf16 = inkling2_loop_ctrl_down(&dest, &msg, &down_1_bf16, ATTN_RES_EPS).unwrap();
            let h_f32 =
                inkling2_loop_ctrl_down(&dest, &msg, &down_1_f32_from_bf16, ATTN_RES_EPS).unwrap();
            assert_eq!(
                h_bf16.to_float32().unwrap().as_ref(),
                h_f32.to_float32().unwrap().as_ref(),
                "BF16 loop_ctrl_down S=1 T={t} must match FP32-cast weights bit-for-bit"
            );
            let y_bf16_w = inkling2_loop_ctrl_up_blend_norm(
                &h_bf16,
                &gf_1_bf16,
                &dest,
                &msg,
                &nw_1,
                norm_eps,
                DType::BFloat16,
            )
            .unwrap();
            let y_f32_w = inkling2_loop_ctrl_up_blend_norm(
                &h_f32,
                &gf_1_f32_from_bf16,
                &dest,
                &msg,
                &nw_1,
                norm_eps,
                DType::BFloat16,
            )
            .unwrap();
            assert_eq!(
                y_bf16_w.to_float32().unwrap().as_ref(),
                y_f32_w.to_float32().unwrap().as_ref(),
                "BF16 loop_ctrl_up_blend_norm S=1 T={t} must match FP32-cast weights bit-for-bit"
            );
        }

        let down_3_bf16 = down_3.astype(DType::BFloat16).unwrap();
        let gf_3_bf16 = gf_3.astype(DType::BFloat16).unwrap();
        let down_3_f32_from_bf16 = down_3_bf16.astype(DType::Float32).unwrap();
        let gf_3_f32_from_bf16 = gf_3_bf16.astype(DType::Float32).unwrap();
        let dest_g8 = MxArray::from_float32(&rng.vec(3 * 8 * d, 1.0), &[3, 8, d as i64]).unwrap();
        let msgs_g8 = MxArray::from_float32(&rng.vec(3 * 8 * d, 1.0), &[3, 8, d as i64]).unwrap();
        let h3_bf16 =
            inkling2_loop_ctrl_down(&dest_g8, &msgs_g8, &down_3_bf16, ATTN_RES_EPS).unwrap();
        let h3_f32 =
            inkling2_loop_ctrl_down(&dest_g8, &msgs_g8, &down_3_f32_from_bf16, ATTN_RES_EPS)
                .unwrap();
        assert_eq!(
            h3_bf16.to_float32().unwrap().as_ref(),
            h3_f32.to_float32().unwrap().as_ref(),
            "BF16 loop_ctrl_down S=3 T=8 must match FP32-cast weights bit-for-bit"
        );
        let y3_bf16 = inkling2_loop_ctrl_up_blend_norm(
            &h3_bf16,
            &gf_3_bf16,
            &dest_g8,
            &msgs_g8,
            &nw_3,
            norm_eps,
            DType::Float32,
        )
        .unwrap();
        let y3_f32 = inkling2_loop_ctrl_up_blend_norm(
            &h3_f32,
            &gf_3_f32_from_bf16,
            &dest_g8,
            &msgs_g8,
            &nw_3,
            norm_eps,
            DType::Float32,
        )
        .unwrap();
        assert_eq!(
            y3_bf16.to_float32().unwrap().as_ref(),
            y3_f32.to_float32().unwrap().as_ref(),
            "BF16 loop_ctrl_up_blend_norm S=3 T=8 must match FP32-cast weights bit-for-bit"
        );
    }

    #[test]
    fn inkling2_markov_fold_softcap_and_argmax_match_dequant_ref() {
        if !metal_available() {
            return;
        }
        let mut rng = Rng(0x2026_C0DE);
        let softcap = 30.0_f32;
        for &(batch, rank, vocab_unpadded, vocab_padded, sidecar_dt) in &[
            (1usize, 256usize, 4099usize, 4128usize, DType::BFloat16),
            (4usize, 256usize, 4099usize, 4128usize, DType::BFloat16),
            (5usize, 256usize, 517usize, 544usize, DType::Float16),
            (2usize, 64usize, 513usize, 544usize, DType::Float32),
            (2usize, 512usize, 513usize, 544usize, DType::BFloat16),
        ] {
            let lat_v = rng.vec(batch * rank, 0.75);
            let w_v = rng.vec(vocab_unpadded * rank, 0.35);
            let fb_v = rng.vec(vocab_unpadded, 0.5);
            let base_v = rng.vec(batch * vocab_padded, 1.5);

            let latent = MxArray::from_float32(&lat_v, &[batch as i64, rank as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let w_src = MxArray::from_float32(&w_v, &[vocab_unpadded as i64, rank as i64])
                .unwrap()
                .astype(sidecar_dt)
                .unwrap();

            let mut out_q = std::ptr::null_mut();
            let mut out_s = std::ptr::null_mut();
            let mut out_b = std::ptr::null_mut();
            let ok = unsafe {
                mlx_sys::mlx_quantize(
                    w_src.as_raw_ptr(),
                    32,
                    8,
                    c"affine".as_ptr(),
                    &mut out_q,
                    &mut out_s,
                    &mut out_b,
                )
            };
            assert!(ok, "mlx_quantize affine Q8_0 failed");
            let w_q = MxArray::from_handle(out_q, "q").unwrap();
            let scales = MxArray::from_handle(out_s, "s").unwrap();
            let biases = MxArray::from_handle(out_b, "b").unwrap();

            let fold_bias = MxArray::from_float32(&fb_v, &[1, vocab_unpadded as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            // Exercise a strided slice of a `[batch, 2, vocab_padded]` tensor (matching `propose_blocks_batched`).
            let base_logits_contig =
                MxArray::from_float32(&base_v, &[batch as i64, vocab_padded as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
            let base_logits = if batch > 1 {
                let zeros = MxArray::zeros(
                    &[batch as i64, 1, vocab_padded as i64],
                    Some(DType::BFloat16),
                )
                .unwrap();
                let stacked = MxArray::concatenate(
                    &zeros,
                    &base_logits_contig
                        .reshape(&[batch as i64, 1, vocab_padded as i64])
                        .unwrap(),
                    1,
                )
                .unwrap();
                stacked
                    .slice(&[0, 1, 0], &[batch as i64, 2, vocab_padded as i64])
                    .unwrap()
                    .squeeze(Some(&[1]))
                    .unwrap()
            } else {
                base_logits_contig
            };

            // Exact CPU FP32 reference matching the fused Q8_0 8-lane tree reduction + BF16 rounding path.
            let q_u32 = w_q.to_uint32().unwrap();
            let s_f32 = scales.astype(DType::Float32).unwrap().to_float32().unwrap();
            let b_f32 = biases.astype(DType::Float32).unwrap().to_float32().unwrap();
            let lat_f32 = latent.astype(DType::Float32).unwrap().to_float32().unwrap();
            let fb_f32 = fold_bias
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let base_f32 = base_logits
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let to_bf16 = |x: f32| -> f32 {
                let bits = x.to_bits();
                let lsb = (bits >> 16) & 1;
                let rounded = bits.wrapping_add(0x7fff + lsb) & 0xffff_0000;
                f32::from_bits(rounded)
            };
            let bf16_ulp = |x: f32| -> f32 {
                let bits = x.abs().to_bits() & 0xffff_0000;
                let next = f32::from_bits(bits.wrapping_add(0x0001_0000));
                (next - f32::from_bits(bits)).max(1e-4)
            };
            let n_groups = rank / 32;
            let row_u32_stride = rank / 4;
            let mut ref_l = vec![0.0_f32; batch * vocab_unpadded];
            let mut ref_tol = vec![1e-4_f32; batch * vocab_unpadded];
            let mut ref_am = vec![0_i32; batch];
            for b in 0..batch {
                let mut best_v = f32::NEG_INFINITY;
                let mut best_i = -1_i32;
                let lat_row = &lat_f32[b * rank..(b + 1) * rank];
                let base_row = &base_f32[b * vocab_padded..(b + 1) * vocab_padded];
                for v in 0..vocab_unpadded {
                    let mut lane_acc = [0.0_f32; 8];
                    for g in 0..n_groups {
                        let sc = s_f32[v * n_groups + g];
                        let bi = b_f32[v * n_groups + g];
                        let mut q_dot = 0.0_f32;
                        let mut s4 = [0.0_f32; 4];
                        for u in 0..8 {
                            let packed = q_u32[v * row_u32_stride + g * 8 + u];
                            let mut d4 = 0.0_f32;
                            for byte_i in 0..4 {
                                let q = ((packed >> (byte_i * 8)) & 0xff) as f32;
                                let x = lat_row[g * 32 + u * 4 + byte_i];
                                d4 += q * x;
                                s4[byte_i] += x;
                            }
                            q_dot += d4;
                        }
                        let x_sum = (s4[0] + s4[1]) + (s4[2] + s4[3]);
                        lane_acc[g & 7] += sc * q_dot + bi * x_sum;
                    }
                    // Match `simd_shuffle_down(4/2/1)` reduction order.
                    let l0 = (lane_acc[0] + lane_acc[4]) + (lane_acc[2] + lane_acc[6]);
                    let l1 = (lane_acc[1] + lane_acc[5]) + (lane_acc[3] + lane_acc[7]);
                    let dot_val = l0 + l1;
                    let dot_bf16 = to_bf16(dot_val);
                    let bias_v = to_bf16(dot_bf16 + fb_f32[v]);
                    let summed = to_bf16(base_row[v] + bias_v);
                    let val = softcap * (summed / softcap).tanh();
                    let idx = b * vocab_unpadded + v;
                    ref_l[idx] = val;
                    // If `dot_val` is within 1e-4 of a BF16 half-ULP midpoint, a 1-FP32-ULP
                    // difference in `dot(float4)` can flip `bfloat(dot_val)` by 1 BF16 ULP.
                    let half_ulp = 0.5 * bf16_ulp(dot_val);
                    if ((dot_val - dot_bf16).abs() - half_ulp).abs() <= 1e-4 {
                        ref_tol[idx] = (bf16_ulp(dot_val) + bf16_ulp(summed)).max(1e-3);
                    }
                    if best_i < 0 || val > best_v {
                        best_v = val;
                        best_i = v as i32;
                    }
                }
                ref_am[b] = best_i;
            }

            let got_logits = inkling2_markov_fold_softcap(
                &latent,
                &w_q,
                &scales,
                Some(&biases),
                &fold_bias,
                &base_logits,
                vocab_unpadded,
                softcap,
            )
            .unwrap();
            let got_argmax = inkling2_markov_fold_softcap_argmax(
                &latent,
                &w_q,
                &scales,
                Some(&biases),
                &fold_bias,
                &base_logits,
                vocab_unpadded,
                softcap,
            )
            .unwrap();

            let got_l = got_logits.to_float32().unwrap();
            assert_eq!(got_l.len(), ref_l.len());
            for (idx, (&g, (&r, &tol))) in got_l.iter().zip(ref_l.iter().zip(&ref_tol)).enumerate()
            {
                let d = (g - r).abs();
                assert!(
                    d <= tol,
                    "markov_fold_softcap (B={batch}, R={rank}, dt={sidecar_dt:?}) idx={idx} got={g} ref={r} diff={d} tol={tol}"
                );
            }
            // Also verify argmax matches `got_logits.argmax(-1)` bit-for-bit!
            let from_logits_am = got_logits
                .argmax(-1, Some(false))
                .unwrap()
                .to_int32()
                .unwrap();
            let got_am = got_argmax.to_int32().unwrap();
            assert_eq!(
                got_am.as_ref(),
                from_logits_am.as_ref(),
                "markov_fold_softcap_argmax != markov_fold_softcap.argmax (B={batch}, R={rank})"
            );
            assert_eq!(
                got_am.as_ref(),
                ref_am.as_slice(),
                "markov_fold_softcap_argmax != exact CPU Q8 reference argmax (B={batch}, R={rank})"
            );
        }

        // Exact smaller-index tie-break check when all logits are equal
        let batch = 2usize;
        let rank = 256usize;
        let vocab = 128usize;
        let latent = MxArray::zeros(&[batch as i64, rank as i64], Some(DType::BFloat16)).unwrap();
        let w_zero = MxArray::zeros(&[vocab as i64, rank as i64], Some(DType::BFloat16)).unwrap();
        let mut out_q = std::ptr::null_mut();
        let mut out_s = std::ptr::null_mut();
        let mut out_b = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_quantize(
                w_zero.as_raw_ptr(),
                32,
                8,
                c"affine".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        });
        let w_q = MxArray::from_handle(out_q, "q").unwrap();
        let scales = MxArray::from_handle(out_s, "s").unwrap();
        let biases = MxArray::from_handle(out_b, "b").unwrap();
        let fold_bias = MxArray::zeros(&[1, vocab as i64], Some(DType::BFloat16)).unwrap();
        let mut base_vals = vec![1.0_f32; batch * vocab];
        base_vals[17] = 5.0;
        base_vals[91] = 5.0; // tie at 17 vs 91 -> must choose 17
        base_vals[vocab + 64] = 7.0;
        base_vals[vocab + 100] = 7.0; // tie at 64 vs 100 -> must choose 64
        let base_logits = MxArray::from_float32(&base_vals, &[batch as i64, vocab as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let tie_am = inkling2_markov_fold_softcap_argmax(
            &latent,
            &w_q,
            &scales,
            Some(&biases),
            &fold_bias,
            &base_logits,
            vocab,
            softcap,
        )
        .unwrap()
        .to_int32()
        .unwrap();
        assert_eq!(tie_am.as_ref(), &[17, 64]);
    }

    #[test]
    fn inkling2_bf16_linear_and_lm_head_argmax_match_ref_and_batch_invariant() {
        if !metal_available() || !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
            return;
        }
        let mut rng = Rng(0xBF16_1234);
        let k = 2304usize;
        let n = 517usize;
        let out_cols = 509usize;
        let softcap = 30.0_f32;

        let w = MxArray::from_float32(&rng.vec(n * k, 0.25), &[n as i64, k as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let w_f32_t = w.astype(DType::Float32).unwrap().transpose(None).unwrap();

        let x_all = MxArray::from_float32(&rng.vec(32 * k, 0.6), &[32, k as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let x0 = x_all.slice(&[0, 0], &[1, k as i64]).unwrap();
        let y0 = inkling2_bf16_linear(&x0, &w).unwrap().to_float32().unwrap();
        let am0 = inkling2_bf16_lm_head_softcap_argmax(&x0, &w, out_cols, softcap)
            .unwrap()
            .to_int32()
            .unwrap();

        for &m in &[1usize, 2, 4, 5, 6, 7, 8, 16, 20, 32] {
            let x = x_all.slice(&[0, 0], &[m as i64, k as i64]).unwrap();
            let got = inkling2_bf16_linear(&x, &w).unwrap();
            let got_f32 = got.to_float32().unwrap();

            // 1. Accuracy vs FP32 matmul reference -> BF16
            let ref_bf16 = x
                .astype(DType::Float32)
                .unwrap()
                .matmul(&w_f32_t)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap()
                .to_float32()
                .unwrap();
            let diff = max_abs_diff(got_f32.as_ref(), ref_bf16.as_ref());
            assert!(
                diff <= 0.125,
                "bf16_linear M={m} max_abs_diff={diff} vs FP32->BF16 ref"
            );

            // 2. 0-ULP batch-size invariance: row 0 at M MUST match M=1 bit-for-bit!
            assert_eq!(
                &got_f32.as_ref()[..n],
                y0.as_ref(),
                "bf16_linear row 0 not 0-ULP invariant between M=1 and M={m}"
            );

            // 3. Fused BF16 lm_head + softcap + argmax MUST match `bf16_linear` -> `lm_head_softcap_argmax` bit-for-bit!
            let got_am = inkling2_bf16_lm_head_softcap_argmax(&x, &w, out_cols, softcap)
                .unwrap()
                .to_int32()
                .unwrap();
            let ref_am = inkling2_lm_head_softcap_argmax(&got, None, out_cols, softcap, false)
                .unwrap()
                .to_int32()
                .unwrap();
            assert_eq!(
                got_am.as_ref(),
                ref_am.as_ref(),
                "bf16_lm_head_softcap_argmax mismatch at M={m}"
            );
            assert_eq!(
                got_am[0], am0[0],
                "bf16_lm_head_softcap_argmax row 0 not invariant between M=1 and M={m}"
            );
        }

        // Also test K % 8 == 4 tail path (e.g. K = 260)
        let k_tail = 260usize;
        let wt = MxArray::from_float32(&rng.vec(64 * k_tail, 0.3), &[64, k_tail as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let xt = MxArray::from_float32(&rng.vec(8 * k_tail, 0.5), &[8, k_tail as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let yt1 = inkling2_bf16_linear(&xt.slice(&[0, 0], &[1, k_tail as i64]).unwrap(), &wt)
            .unwrap()
            .to_float32()
            .unwrap();
        let yt8 = inkling2_bf16_linear(&xt, &wt)
            .unwrap()
            .to_float32()
            .unwrap();
        assert_eq!(&yt8.as_ref()[..64], yt1.as_ref());
    }

    #[test]
    fn inkling2_q8_sym_kernels_match_mlx_ref_0ulp_and_batch_invariant() {
        if !metal_available() {
            return;
        }
        use crate::models::quantized_linear::{DEFAULT_QUANT_MODE, QuantizedLinear};

        let mut rng = Rng(0x5180_9A2B);
        let softcap = 30.0_f32;

        // Build exact symmetric Q8_0 weights (`w_u32` storing `q_u8 = (q_i8 as u8) ^ 0x80`,
        // native Float16 `scales`, `biases = scales * -128.0` in F16) covering BOTH
        // K = 2304 (72 groups, exact Inkling2 hidden dim) AND K = 96 (3 groups, non-multiple of 8).
        for &(k, n, out_cols, aux_n) in
            &[(2304usize, 517usize, 509usize, 24usize), (96, 67, 61, 32)]
        {
            let n_groups = k / 32;
            let row_u32 = k / 4;
            let mut w_seed = 0x5180_9A2Bu64 ^ (k as u64);
            let mut next_u8 = || -> u32 {
                w_seed = w_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((w_seed >> 32) & 0xff) as u32
            };
            let mut w_u32 = vec![0u32; n * row_u32];
            for word in &mut w_u32 {
                // Exercise all 256 byte values 0..=255 including 0 (-128), 128 (0), 255 (+127).
                let b0 = next_u8();
                let b1 = next_u8();
                let b2 = next_u8();
                let b3 = next_u8();
                *word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            }
            // Pin row 0 first 64 words to every exact byte value 0..=255 to prove 0-ULP coverage!
            for (idx, word) in w_u32.iter_mut().enumerate().take(64.min(row_u32)) {
                let base = (idx * 4) as u32;
                *word = base | ((base + 1) << 8) | ((base + 2) << 16) | ((base + 3) << 24);
            }
            let scales_f32_src = rng.vec(n * n_groups, 0.02);
            let scales_f16 = MxArray::from_float32(&scales_f32_src, &[n as i64, n_groups as i64])
                .unwrap()
                .astype(DType::Float16)
                .unwrap();
            let biases_f16 = scales_f16
                .mul_scalar(-128.0)
                .unwrap()
                .astype(DType::Float16)
                .unwrap();
            let w_q8 = MxArray::from_uint32(&w_u32, &[n as i64, row_u32 as i64]).unwrap();

            // Verify QuantizedLinear symmetric Q8_0 compaction drops biases & retains exact FP32 dequant!
            let mut ql = QuantizedLinear::new(
                w_q8.clone(),
                scales_f16.clone(),
                Some(biases_f16.clone()),
                None,
                32,
                8,
                DEFAULT_QUANT_MODE.to_string(),
            );
            assert!(ql.try_compact_symmetric_q8_f16().unwrap());
            assert!(ql.is_symmetric_q8());
            assert!(ql.get_biases().is_none());
            assert_eq!(ql.get_scales().dtype().unwrap(), DType::Float16);
            let deq_after = ql.dequantized_weight().unwrap().to_float32().unwrap();

            // Exact FP32 MLX dequant reference (using F32-cast F16 sidecars, matching MLX's
            // `promote_types(bf16, f16) -> f32` dequant arithmetic) proves `w ^ 0x80808080u`
            // is 0-ULP bit-exact to `int(u8) - 128` for EVERY byte 0..=255!
            let ql_f32_ref = QuantizedLinear::new(
                w_q8.clone(),
                scales_f16.astype(DType::Float32).unwrap(),
                Some(biases_f16.astype(DType::Float32).unwrap()),
                None,
                32,
                8,
                DEFAULT_QUANT_MODE.to_string(),
            );
            let deq_f32_ref = ql_f32_ref
                .dequantized_weight()
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(deq_after.as_ref(), deq_f32_ref.as_ref());
            let sc_vals = scales_f16
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            for r in 0..n.min(4) {
                for g in 0..n_groups {
                    let s = sc_vals[r * n_groups + g];
                    for u in 0..8 {
                        let word = w_u32[r * row_u32 + g * 8 + u];
                        let xor_word = word ^ 0x8080_8080u32;
                        for byte_i in 0..4 {
                            let q_u8 = ((word >> (byte_i * 8)) & 0xff) as i32;
                            let q_i8 = ((xor_word >> (byte_i * 8)) & 0xff) as i8;
                            assert_eq!(q_i8 as i32, q_u8 - 128);
                            let expected = s * (q_i8 as f32);
                            let got = deq_f32_ref[r * k + g * 32 + u * 4 + byte_i];
                            assert_eq!(
                                (got - expected).abs(),
                                0.0,
                                "0-ULP dequant mismatch at r={r} g={g} u={u} b={byte_i}"
                            );
                        }
                    }
                }
            }

            let w_aux = MxArray::from_float32(&rng.vec(aux_n * k, 0.2), &[aux_n as i64, k as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let w_aux_f32_t = w_aux
                .astype(DType::Float32)
                .unwrap()
                .transpose(None)
                .unwrap();

            let x_all = MxArray::from_float32(&rng.vec(32 * k, 0.5), &[32, k as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let x0 = x_all.slice(&[0, 0], &[1, k as i64]).unwrap();
            let y0_bf16 = inkling2_q8_sym_matmul_bf16(&x0, &w_q8, &scales_f16, DType::BFloat16)
                .unwrap()
                .to_float32()
                .unwrap();
            let (ym0, ya0) = inkling2_q8_sym_matmul_bf16_with_aux(
                &x0,
                &w_q8,
                &scales_f16,
                &w_aux,
                DType::BFloat16,
            )
            .unwrap();
            let ym0_v = ym0.to_float32().unwrap();
            let ya0_v = ya0.to_float32().unwrap();
            assert_eq!(ym0_v.as_ref(), y0_bf16.as_ref());
            let lm0_am =
                inkling2_lm_head_q8_softcap_argmax(&x0, &w_q8, &scales_f16, out_cols, softcap)
                    .unwrap()
                    .to_int32()
                    .unwrap();

            for &m in &[1usize, 2, 4, 5, 7, 8, 9, 15, 16, 17, 24, 28, 32] {
                let x = x_all.slice(&[0, 0], &[m as i64, k as i64]).unwrap();
                let mlx_ref_f32 = ql.forward_keep_f32(&x).unwrap().to_float32().unwrap();

                // 1. `inkling2_q8_sym_matmul_bf16` (F32 & BF16) vs MLX `quantized_matmul` + 0-ULP M invariance!
                let got_f32 = inkling2_q8_sym_matmul_bf16(&x, &w_q8, &scales_f16, DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let diff_f32 = max_abs_diff(got_f32.as_ref(), mlx_ref_f32.as_ref());
                assert!(
                    diff_f32 <= 1e-4,
                    "q8_sym_matmul F32 K={k} M={m} max_abs_diff={diff_f32} vs MLX quantized_matmul"
                );
                let got_bf16 = inkling2_q8_sym_matmul_bf16(&x, &w_q8, &scales_f16, DType::BFloat16)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                assert_eq!(
                    &got_bf16.as_ref()[..n],
                    y0_bf16.as_ref(),
                    "q8_sym_matmul row 0 not 0-ULP invariant between M=1 and M={m} (K={k})"
                );

                // 2. Fused BF16 aux tail matches exact Q8 main + exact BF16 aux & 0-ULP M invariance!
                let (got_m, got_a) = inkling2_q8_sym_matmul_bf16_with_aux(
                    &x,
                    &w_q8,
                    &scales_f16,
                    &w_aux,
                    DType::BFloat16,
                )
                .unwrap();
                let got_m_v = got_m.to_float32().unwrap();
                let got_a_v = got_a.to_float32().unwrap();
                assert_eq!(got_m_v.as_ref(), got_bf16.as_ref());
                assert_eq!(
                    &got_a_v.as_ref()[..aux_n],
                    ya0_v.as_ref(),
                    "q8_sym_matmul aux row 0 not 0-ULP invariant between M=1 and M={m}"
                );
                let ref_aux = x
                    .astype(DType::Float32)
                    .unwrap()
                    .matmul(&w_aux_f32_t)
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let aux_diff = max_abs_diff(got_a_v.as_ref(), ref_aux.as_ref());
                assert!(
                    aux_diff <= 0.0625,
                    "q8_sym_matmul aux K={k} M={m} max_abs_diff={aux_diff}"
                );

                // 3. `inkling2_lm_head_q8_bf16` (BF16 & FP32+softcap) + `inkling2_lm_head_q8_softcap_argmax`!
                let lm_bf16 = inkling2_lm_head_q8_bf16(
                    &x,
                    &w_q8,
                    &scales_f16,
                    out_cols,
                    0.0,
                    DType::BFloat16,
                )
                .unwrap()
                .to_float32()
                .unwrap();
                for r in 0..m {
                    assert_eq!(
                        &lm_bf16.as_ref()[r * out_cols..(r + 1) * out_cols],
                        &got_bf16.as_ref()[r * n..r * n + out_cols],
                        "lm_head_q8_bf16 != q8_sym_matmul slice at K={k} M={m} r={r}"
                    );
                }
                let lm_sc_f32 = inkling2_lm_head_q8_bf16(
                    &x,
                    &w_q8,
                    &scales_f16,
                    out_cols,
                    softcap,
                    DType::Float32,
                )
                .unwrap();
                let ref_am = lm_sc_f32
                    .argmax(-1, Some(false))
                    .unwrap()
                    .to_int32()
                    .unwrap();
                let got_am =
                    inkling2_lm_head_q8_softcap_argmax(&x, &w_q8, &scales_f16, out_cols, softcap)
                        .unwrap()
                        .to_int32()
                        .unwrap();
                assert_eq!(
                    got_am.as_ref(),
                    ref_am.as_ref(),
                    "lm_head_q8_softcap_argmax mismatch at K={k} M={m}"
                );
                assert_eq!(
                    got_am[0], lm0_am[0],
                    "lm_head_q8_softcap_argmax row 0 not 0-ULP invariant between M=1 and M={m}"
                );
            }
        }

        // Exact smaller-index tie-break check when all weights are zero (`q_u8 = 128` -> `q_i8 = 0`)
        let w_zero = MxArray::from_uint32(&vec![0x8080_8080u32; 128 * 8], &[128, 8]).unwrap();
        let sc_one = MxArray::from_float32(&vec![1.0f32; 128], &[128, 1])
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        let x_tie = MxArray::from_float32(&vec![1.0f32; 20 * 32], &[20, 32])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let am_tie = inkling2_lm_head_q8_softcap_argmax(&x_tie, &w_zero, &sc_one, 128, softcap)
            .unwrap()
            .to_int32()
            .unwrap();
        assert_eq!(am_tie.as_ref(), &[0i32; 20]);
    }

    #[test]
    fn inkling2_grouped_moe_fused_rejects_mismatched_history_slice() {
        let dummy = MxArray::zeros(&[1, 64], Some(DType::BFloat16)).unwrap();
        let dummy_u32 = MxArray::zeros(&[1, 1, 1], Some(DType::Uint32)).unwrap();
        let dummy_u8 = MxArray::zeros(&[1, 1, 1], Some(DType::Uint8)).unwrap();
        let dummy_post = MxArray::zeros(&[64, 4], Some(DType::BFloat16)).unwrap();
        let hist = MxArray::zeros(&[3, 64], Some(DType::BFloat16)).unwrap();

        // Empty history slice for num_seqs=1 (unpacked) -> Err (must never read past slice)
        let res_empty = inkling2_grouped_moe_fused(
            &dummy,
            &dummy,
            None,
            None,
            &dummy_u32,
            &dummy_u8,
            None,
            &dummy_u32,
            &dummy_u8,
            None,
            &dummy_post,
            None,
            &[0, 1],
            &[],
            1,
            false,
            1,
            1.0,
            Inkling2MoeQuantMode::Nvfp4,
            false,
            0.0,
            0.0,
            false,
            false,
            false,
        );
        assert!(res_empty.is_err());

        // Short history slice (1 hist for num_seqs=3 unpacked) -> Err
        let res_short = inkling2_grouped_moe_fused(
            &dummy,
            &dummy,
            None,
            None,
            &dummy_u32,
            &dummy_u8,
            None,
            &dummy_u32,
            &dummy_u8,
            None,
            &dummy_post,
            None,
            &[0, 1, 2, 3],
            &[&hist],
            3,
            false,
            1,
            1.0,
            Inkling2MoeQuantMode::Nvfp4,
            false,
            0.0,
            0.0,
            false,
            false,
            false,
        );
        assert!(res_short.is_err());

        // Empty history slice for packed_hist=true -> Err
        let res_packed_empty = inkling2_grouped_moe_fused(
            &dummy,
            &dummy,
            None,
            None,
            &dummy_u32,
            &dummy_u8,
            None,
            &dummy_u32,
            &dummy_u8,
            None,
            &dummy_post,
            None,
            &[0, 1, 2],
            &[],
            2,
            true,
            1,
            1.0,
            Inkling2MoeQuantMode::Nvfp4,
            false,
            0.0,
            0.0,
            false,
            false,
            false,
        );
        assert!(res_packed_empty.is_err());

        // Excess history slice (2 hists for packed_hist=true) -> Err
        let res_packed_excess = inkling2_grouped_moe_fused(
            &dummy,
            &dummy,
            None,
            None,
            &dummy_u32,
            &dummy_u8,
            None,
            &dummy_u32,
            &dummy_u8,
            None,
            &dummy_post,
            None,
            &[0, 1, 2],
            &[&hist, &hist],
            2,
            true,
            1,
            1.0,
            Inkling2MoeQuantMode::Nvfp4,
            false,
            0.0,
            0.0,
            false,
            false,
            false,
        );
        assert!(res_packed_excess.is_err());

        // Invalid ragged `cu_seqlens` (non-zero start, overlapping/non-increasing, OOB end,
        // short end, wrong length) must all be rejected before GPU dispatch.
        let x4 = MxArray::zeros(&[4, 64], Some(DType::BFloat16)).unwrap();
        let logits4 = MxArray::zeros(&[4, 2], Some(DType::Float32)).unwrap();
        for (bad_cu, n_seqs) in [
            (&[1usize, 4][..], 1usize), // non-zero start
            (&[0, 2, 2, 4][..], 3),     // zero-length sequence
            (&[0, 3, 2, 4][..], 3),     // overlapping / decreasing offsets
            (&[0, 2, 5][..], 2),        // end exceeds T=4 (would OOB on GPU)
            (&[0, 2, 3][..], 2),        // end < T=4
            (&[0, 4][..], 2),           // length != num_seqs + 1
            (&[0, 1, 2, 3, 4][..], 2),  // length != num_seqs + 1
        ] {
            let hists_vec: Vec<&MxArray> = vec![&hist; n_seqs];
            let res = inkling2_grouped_moe_fused(
                &x4,
                &logits4,
                None,
                None,
                &dummy_u32,
                &dummy_u8,
                None,
                &dummy_u32,
                &dummy_u8,
                None,
                &dummy_post,
                None,
                bad_cu,
                &hists_vec,
                n_seqs,
                false,
                1,
                1.0,
                Inkling2MoeQuantMode::Nvfp4,
                false,
                0.0,
                0.0,
                false,
                false,
                false,
            );
            assert!(
                res.is_err(),
                "expected Err for invalid cu_seqlens={bad_cu:?} (num_seqs={n_seqs}, T=4)"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn inkling2_sconv_k4_kv_prep_paged_matches_unfused_bit_exact() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return;
        }
        let mut rng = Rng(0x5C0B_9A6E_D123_4567);
        let num_kv_heads = 8usize;
        let head_dim = 128usize;
        let kv_dim = num_kv_heads * head_dim;
        let num_blocks = 8usize;

        for &(kv_mode, block_size, is_dynamic) in &[
            (0u32, 32usize, false), // INT8-KTM Local static
            (0u32, 32usize, true),  // INT8-KTM Global dynamic
            (1u32, 16usize, false), // BF16 vLLM bs=16 Local static
            (1u32, 32usize, true),  // BF16 vLLM bs=32 Global dynamic
            (2u32, 32usize, false), // BF16 vt bs=32 Local static
        ] {
            for &(num_seqs, lens) in &[(1usize, &[5usize][..]), (4usize, &[3, 1, 4, 2][..])] {
                let total_t: usize = lens.iter().sum();
                // Strided k_in/v_in (simulating slices of `fused_qrgkv`)!
                let wide_stride = kv_dim * 2 + 64;
                let wide = MxArray::from_float32(
                    &rng.vec(total_t * wide_stride, 0.7),
                    &[total_t as i64, wide_stride as i64],
                )
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
                let k_in = wide
                    .slice(&[0, 32], &[total_t as i64, 32 + kv_dim as i64])
                    .unwrap();
                let v_in = wide
                    .slice(
                        &[0, 32 + kv_dim as i64],
                        &[total_t as i64, 32 + 2 * kv_dim as i64],
                    )
                    .unwrap();
                let kw =
                    MxArray::from_float32(&rng.vec(kv_dim * 4, 0.5), &[kv_dim as i64, 4]).unwrap();
                let vw =
                    MxArray::from_float32(&rng.vec(kv_dim * 4, 0.5), &[kv_dim as i64, 4]).unwrap();
                let dyn_k = is_dynamic.then(|| {
                    MxArray::from_float32(
                        &rng.vec(total_t * num_kv_heads * 4, 0.3),
                        &[total_t as i64, num_kv_heads as i64, 4],
                    )
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
                });
                let dyn_v = is_dynamic.then(|| {
                    MxArray::from_float32(
                        &rng.vec(total_t * num_kv_heads * 4, 0.3),
                        &[total_t as i64, num_kv_heads as i64, 4],
                    )
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
                });
                let mut cu_vals = vec![0i32];
                for &l in lens {
                    cu_vals.push(cu_vals.last().copied().unwrap() + l as i32);
                }
                let cu_arr = (num_seqs > 1)
                    .then(|| MxArray::from_int32(&cu_vals, &[cu_vals.len() as i64]).unwrap());
                let k_hists: Vec<MxArray> = (0..num_seqs)
                    .map(|_| {
                        MxArray::from_float32(&rng.vec(3 * kv_dim, 0.4), &[3, kv_dim as i64])
                            .unwrap()
                            .astype(DType::BFloat16)
                            .unwrap()
                    })
                    .collect();
                let v_hists: Vec<MxArray> = (0..num_seqs)
                    .map(|_| {
                        MxArray::from_float32(&rng.vec(3 * kv_dim, 0.4), &[3, kv_dim as i64])
                            .unwrap()
                            .astype(DType::BFloat16)
                            .unwrap()
                    })
                    .collect();
                let kh_refs: Vec<&MxArray> = k_hists.iter().collect();
                let vh_refs: Vec<&MxArray> = v_hists.iter().collect();

                // Include one negative padding slot (-1) to verify skip behavior!
                let mut slots: Vec<i64> = (0..total_t).map(|i| (i * 3 + 5) as i64).collect();
                if total_t >= 3 {
                    slots[1] = -1;
                }
                let slot_mapping = MxArray::from_int64(&slots, &[total_t as i64]).unwrap();

                let make_pools = || -> (MxArray, MxArray, MxArray, MxArray) {
                    if kv_mode == 0 {
                        let kp = MxArray::zeros(
                            &[
                                num_blocks as i64,
                                num_kv_heads as i64,
                                block_size as i64,
                                head_dim as i64,
                            ],
                            Some(DType::Int8),
                        )
                        .unwrap();
                        let vp = MxArray::zeros(
                            &[
                                num_blocks as i64,
                                num_kv_heads as i64,
                                head_dim as i64,
                                block_size as i64,
                            ],
                            Some(DType::Int8),
                        )
                        .unwrap();
                        let ks = MxArray::zeros(
                            &[num_blocks as i64, num_kv_heads as i64, block_size as i64],
                            Some(DType::Float32),
                        )
                        .unwrap();
                        let vs = MxArray::zeros(
                            &[num_blocks as i64, num_kv_heads as i64, block_size as i64],
                            Some(DType::Float32),
                        )
                        .unwrap();
                        (kp, vp, ks, vs)
                    } else if kv_mode == 1 {
                        let kp = MxArray::zeros(
                            &[
                                num_blocks as i64,
                                num_kv_heads as i64,
                                (head_dim / 8) as i64,
                                block_size as i64,
                                8,
                            ],
                            Some(DType::BFloat16),
                        )
                        .unwrap();
                        let vp = MxArray::zeros(
                            &[
                                num_blocks as i64,
                                num_kv_heads as i64,
                                head_dim as i64,
                                block_size as i64,
                            ],
                            Some(DType::BFloat16),
                        )
                        .unwrap();
                        let ks = MxArray::ones(&[1], Some(DType::Float32)).unwrap();
                        let vs = MxArray::ones(&[1], Some(DType::Float32)).unwrap();
                        (kp, vp, ks, vs)
                    } else {
                        let kp = MxArray::zeros(
                            &[
                                num_blocks as i64,
                                num_kv_heads as i64,
                                block_size as i64,
                                head_dim as i64,
                            ],
                            Some(DType::BFloat16),
                        )
                        .unwrap();
                        let vp = MxArray::zeros(
                            &[
                                num_blocks as i64,
                                num_kv_heads as i64,
                                block_size as i64,
                                head_dim as i64,
                            ],
                            Some(DType::BFloat16),
                        )
                        .unwrap();
                        let ks = MxArray::ones(&[1], Some(DType::Float32)).unwrap();
                        let vs = MxArray::ones(&[1], Some(DType::Float32)).unwrap();
                        (kp, vp, ks, vs)
                    }
                };

                // Reference: separate `inkling2_sconv_k4_kv_prep` + `mlx_paged_kv_write_forward`
                let (ref_kp0, ref_vp0, ref_ks0, ref_vs0) = make_pools();
                let (ref_k, ref_v, ref_kh, ref_vh) = inkling2_sconv_k4_kv_prep(
                    &k_in,
                    &v_in,
                    &kw,
                    &vw,
                    dyn_k.as_ref(),
                    dyn_v.as_ref(),
                    cu_arr.as_ref(),
                    &kh_refs,
                    &vh_refs,
                    num_seqs,
                    false,
                    num_kv_heads,
                    head_dim,
                    22.627417,
                    1e-6,
                    1e-6,
                )
                .unwrap();
                let mut r_kp = std::ptr::null_mut();
                let mut r_vp = std::ptr::null_mut();
                let mut r_ks = std::ptr::null_mut();
                let mut r_vs = std::ptr::null_mut();
                let kv_dtype_u8 = if kv_mode == 0 {
                    Inkling2KvDtype::Int8 as u8
                } else {
                    Inkling2KvDtype::Bf16 as u8
                };
                let ok = unsafe {
                    mlx_sys::mlx_paged_kv_write_forward(
                        ref_kp0.as_raw_ptr(),
                        ref_vp0.as_raw_ptr(),
                        ref_k.as_raw_ptr(),
                        ref_v.as_raw_ptr(),
                        slot_mapping.as_raw_ptr(),
                        ref_ks0.as_raw_ptr(),
                        ref_vs0.as_raw_ptr(),
                        block_size as i32,
                        num_kv_heads as i32,
                        head_dim as i32,
                        kv_dtype_u8,
                        kv_mode == 2,
                        &mut r_kp,
                        &mut r_vp,
                        &mut r_ks,
                        &mut r_vs,
                    )
                };
                assert!(ok, "reference mlx_paged_kv_write_forward failed");
                let ref_kp = MxArray::from_handle(r_kp, "ref_kp").unwrap();
                let ref_vp = MxArray::from_handle(r_vp, "ref_vp").unwrap();
                let ref_ks = MxArray::from_handle(r_ks, "ref_ks").unwrap();
                let ref_vs = MxArray::from_handle(r_vs, "ref_vs").unwrap();

                // Fused: single `inkling2_sconv_k4_kv_prep_paged`
                let (f_kp0, f_vp0, f_ks0, f_vs0) = make_pools();
                let (f_kp, f_vp, f_ks, f_vs, f_kh, f_vh) = inkling2_sconv_k4_kv_prep_paged(
                    &k_in,
                    &v_in,
                    &kw,
                    &vw,
                    dyn_k.as_ref(),
                    dyn_v.as_ref(),
                    cu_arr.as_ref(),
                    &kh_refs,
                    &vh_refs,
                    &f_kp0,
                    &f_vp0,
                    &f_ks0,
                    &f_vs0,
                    &slot_mapping,
                    num_seqs,
                    false,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    kv_mode,
                    22.627417,
                    1e-6,
                    1e-6,
                )
                .unwrap();

                // Verify 0-ULP bit-exact equality on histories AND every pool/scale byte!
                assert_eq!(
                    f_kh.astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    ref_kh
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    "k_hist mismatch (kv_mode={kv_mode}, bs={block_size}, dyn={is_dynamic}, seqs={num_seqs})"
                );
                assert_eq!(
                    f_vh.astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    ref_vh
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    "v_hist mismatch (kv_mode={kv_mode}, bs={block_size}, dyn={is_dynamic}, seqs={num_seqs})"
                );
                assert_eq!(
                    f_kp.astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    ref_kp
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    "k_pool mismatch (kv_mode={kv_mode}, bs={block_size}, dyn={is_dynamic}, seqs={num_seqs})"
                );
                assert_eq!(
                    f_vp.astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    ref_vp
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    "v_pool mismatch (kv_mode={kv_mode}, bs={block_size}, dyn={is_dynamic}, seqs={num_seqs})"
                );
                if kv_mode == 0 {
                    assert_eq!(
                        f_ks.to_float32().unwrap().as_ref(),
                        ref_ks.to_float32().unwrap().as_ref(),
                        "k_scale mismatch"
                    );
                    assert_eq!(
                        f_vs.to_float32().unwrap().as_ref(),
                        ref_vs.to_float32().unwrap().as_ref(),
                        "v_scale mismatch"
                    );
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn inkling2_paged_attention_fused_gate_sink_and_otf_rel_bias_parity() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return;
        }
        let mut rng = Rng(0xA77E_514C_BEEF_CAFE);
        let num_q_heads = 24usize;
        let num_kv_heads = 8usize;
        let head_dim = 128usize;
        let rel_extent = 1024usize;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        // Exercise BOTH V1 (ctx <= 512) and V2 (ctx > 512) on INT8-KTM and BF16 (bs=16, bs=32)!
        for &(kv_dtype, block_size, ctx_len, q_len) in &[
            (Inkling2KvDtype::Int8, 32usize, 128usize, 4usize),
            (Inkling2KvDtype::Int8, 32usize, 640usize, 8usize),
            (Inkling2KvDtype::Bf16, 16usize, 128usize, 4usize),
            (Inkling2KvDtype::Bf16, 32usize, 640usize, 4usize),
        ] {
            let needed_blocks = ctx_len.div_ceil(block_size);
            let num_blocks = needed_blocks + 2; // block 0 and last block are NaN sentinels
            let k_all = MxArray::from_float32(
                &rng.vec(ctx_len * num_kv_heads * head_dim, 0.35),
                &[ctx_len as i64, num_kv_heads as i64, head_dim as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let v_all = MxArray::from_float32(
                &rng.vec(ctx_len * num_kv_heads * head_dim, 0.35),
                &[ctx_len as i64, num_kv_heads as i64, head_dim as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let mut pool = build_pool(kv_dtype, block_size, num_kv_heads, head_dim, num_blocks);
            let slots: Vec<i64> = (0..ctx_len).map(|i| (block_size + i) as i64).collect();
            store_kv(
                &mut pool,
                &k_all,
                &v_all,
                &slots,
                block_size,
                num_kv_heads,
                head_dim,
                kv_dtype,
            );
            let (k_pool, v_pool, k_scale, v_scale) = (
                pool.k_pool.clone(),
                pool.v_pool.clone(),
                pool.k_scale.clone(),
                pool.v_scale.clone(),
            );

            let q = MxArray::from_float32(
                &rng.vec(q_len * num_q_heads * head_dim, 0.5),
                &[q_len as i64, num_q_heads as i64, head_dim as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let r_for_rel = MxArray::from_float32(
                &rng.vec(q_len * num_q_heads * 16, 0.3),
                &[q_len as i64, num_q_heads as i64, 16],
            )
            .unwrap();
            let rel_bias_w =
                MxArray::from_float32(&rng.vec(rel_extent * 16, 0.2), &[rel_extent as i64, 16])
                    .unwrap();
            let rel_bias_t = rel_bias_w.transpose(None).unwrap();
            let rel_bias_precomp = r_for_rel.matmul(&rel_bias_t).unwrap();

            // Zero-copy strided gate & sink sliced from wider [T, q_dim + H + 64] BF16 buffer!
            let q_dim = num_q_heads * head_dim;
            let wide_cols = q_dim + num_q_heads + 64;
            // Wide [-12, +12] gate/sink range stresses sigmoid precision & catches `-ffast-math` reassociation!
            let wide = MxArray::from_float32(
                &rng.vec(q_len * wide_cols, 12.0),
                &[q_len as i64, wide_cols as i64],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let gate_strided = wide
                .slice(&[0, 32], &[q_len as i64, (32 + q_dim) as i64])
                .unwrap();
            let sink_strided = wide
                .slice(
                    &[0, (32 + q_dim) as i64],
                    &[q_len as i64, (32 + q_dim + num_q_heads) as i64],
                )
                .unwrap();

            let bt_vals: Vec<i32> = (1..=(needed_blocks as i32)).collect();
            let bt_2d = MxArray::from_int32(&bt_vals, &[1, bt_vals.len() as i64]).unwrap();
            let seq_lens = MxArray::from_int32(&[ctx_len as i32], &[1]).unwrap();
            let cu_q = MxArray::from_int32(&[0, q_len as i32], &[2]).unwrap();

            // 1. Reference unfused: paged_attention (precomp rel_bias) + inkling2_gate_sink
            let (ref_o, ref_lse) = inkling2_paged_attention(
                &q,
                &k_pool,
                &v_pool,
                &bt_2d,
                &seq_lens,
                &cu_q,
                &k_scale,
                &v_scale,
                Some(&rel_bias_precomp),
                scale,
                0,
                rel_extent as i32,
                block_size as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dtype,
                None,
            )
            .unwrap();
            let ref_y = inkling2_gate_sink(&ref_o, &gate_strided, &ref_lse, &sink_strided).unwrap();
            let ref_y_f32 = ref_y.astype(DType::Float32).unwrap().to_float32().unwrap();

            // 2. Fused gate_sink ONLY (with precomp rel_bias) -> MUST be 0-ULP BIT-EXACT!
            let (fused_gs_y, no_lse) = inkling2_paged_attention_ex(
                &q,
                &k_pool,
                &v_pool,
                &bt_2d,
                &seq_lens,
                &cu_q,
                &k_scale,
                &v_scale,
                Some(&rel_bias_precomp),
                None,
                Some((&gate_strided, &sink_strided)),
                false,
                scale,
                0,
                rel_extent as i32,
                block_size as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dtype,
                None,
            )
            .unwrap();
            assert!(no_lse.is_none());
            let fused_gs_f32 = fused_gs_y
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(
                fused_gs_f32.as_ref(),
                ref_y_f32.as_ref(),
                "fused gate_sink must be 0-ULP bit-exact (dtype={kv_dtype:?}, bs={block_size}, ctx={ctx_len})"
            );

            // 3. Fused gate_sink + on-the-fly `r_for_rel @ rel_bias_w^T` -> matches reference tightly!
            let (fused_both_y, _) = inkling2_paged_attention_ex(
                &q,
                &k_pool,
                &v_pool,
                &bt_2d,
                &seq_lens,
                &cu_q,
                &k_scale,
                &v_scale,
                Some(&r_for_rel),
                Some(&rel_bias_w),
                Some((&gate_strided, &sink_strided)),
                false,
                scale,
                0,
                rel_extent as i32,
                block_size as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                kv_dtype,
                None,
            )
            .unwrap();
            let fused_both_f32 = fused_both_y
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let diff = max_abs_diff(fused_both_f32.as_ref(), ref_y_f32.as_ref());
            assert!(
                diff <= 1e-2,
                "fused gate_sink + OTF rel_bias max_abs_diff={diff} (dtype={kv_dtype:?}, bs={block_size}, ctx={ctx_len})"
            );

            // 4. Also test MPP prefill kernel with fused gate_sink + OTF rel_bias when supported!
            if inkling2_paged_prefill_mpp_supported() {
                let bt_1d = MxArray::from_int32(&bt_vals, &[bt_vals.len() as i64]).unwrap();
                let ws_floats = mlx_paged_attn::prefill_plan::workspace_floats(
                    q_len as u32,
                    2,
                    num_q_heads as u32,
                    head_dim as u32,
                );
                let ws0 = MxArray::zeros(&[ws_floats as i64], Some(DType::Float32)).unwrap();
                let (pref_o, pref_lse, ws1) = inkling2_paged_prefill_attention(
                    &q,
                    &k_pool,
                    &v_pool,
                    &k_scale,
                    &v_scale,
                    &bt_1d,
                    &ws0,
                    Some(&rel_bias_precomp),
                    (ctx_len - q_len) as i32,
                    2,
                    32,
                    block_size as i32,
                    num_q_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    0,
                    rel_extent as i32,
                    kv_dtype,
                    false,
                    scale,
                )
                .unwrap();
                let pref_ref_y =
                    inkling2_gate_sink(&pref_o, &gate_strided, &pref_lse, &sink_strided)
                        .unwrap()
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap();

                let (pref_fused_gs, no_lse, ws2) = inkling2_paged_prefill_attention_ex(
                    &q,
                    &k_pool,
                    &v_pool,
                    &k_scale,
                    &v_scale,
                    &bt_1d,
                    &ws1,
                    Some(&rel_bias_precomp),
                    None,
                    Some((&gate_strided, &sink_strided)),
                    false,
                    (ctx_len - q_len) as i32,
                    2,
                    32,
                    block_size as i32,
                    num_q_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    0,
                    rel_extent as i32,
                    kv_dtype,
                    false,
                    scale,
                )
                .unwrap();
                assert!(no_lse.is_none());
                assert_eq!(
                    pref_fused_gs
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    pref_ref_y.as_ref(),
                    "MPP prefill fused gate_sink must be 0-ULP bit-exact (dtype={kv_dtype:?}, bs={block_size}, ctx={ctx_len})"
                );

                let (pref_fused_both, _, _) = inkling2_paged_prefill_attention_ex(
                    &q,
                    &k_pool,
                    &v_pool,
                    &k_scale,
                    &v_scale,
                    &bt_1d,
                    &ws2,
                    Some(&r_for_rel),
                    Some(&rel_bias_w),
                    Some((&gate_strided, &sink_strided)),
                    false,
                    (ctx_len - q_len) as i32,
                    2,
                    32,
                    block_size as i32,
                    num_q_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    0,
                    rel_extent as i32,
                    kv_dtype,
                    false,
                    scale,
                )
                .unwrap();
                let pref_diff = max_abs_diff(
                    pref_fused_both
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .as_ref(),
                    pref_ref_y.as_ref(),
                );
                assert!(
                    pref_diff <= 1e-2,
                    "MPP prefill fused gate_sink + OTF rel_bias max_abs_diff={pref_diff}"
                );
            }
        }
    }
}
