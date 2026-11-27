//! Safe Rust FFI wrappers for Hy4-flash (`hy_v4_flash`) CUDA kernels.
//!
//! Targets RTX Spark (Windows ARM64 `sm_121a`, CUDA 13.4+) and DGX Spark
//! (Linux aarch64 `sm_121a`, CUDA 13.0+). Every function returns `Ok(None)`
//! off CUDA or when a shape/dtype declines the fast path.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::MxArray;

#[inline]
pub fn available() -> bool {
    active_backend() == ActiveBackend::CudaGpu
}

/// NVFP4 prefill mode selector (`MLX_HY4_NVFP4_PREFILL_MODE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Nvfp4PrefillMode {
    Auto = 0,
    Bf16Tc = 1,
    CublasFp4 = 2,
}

/// Weight format selector for Hy4 LM head kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum LmHeadWeightMode {
    Nvfp4 = 0,
    SymQ8 = 1,
    Bf16 = 2,
}

/// Weight format selector for Hy4 MoE kernels (`NVFP4` trunk, `SymQ8` MTP).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum MoeWeightMode {
    Nvfp4 = 0,
    SymQ8 = 1,
}

/// Weight format selector for Hy4 HC `*_fn` matrices.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum HcFnMode {
    Nvfp4 = 0,
    SymQ8 = 1,
    F32 = 2,
    Bf16 = 3,
}

/// Pre-swizzle 2D `uint8 [N, K/16]` NVFP4 scales into 128×4 cuBLASLt tiled layout.
#[inline]
pub fn nvfp4_swizzle_scales(scales: &MxArray) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h = unsafe { sys::mlx_hy4_nvfp4_swizzle_scales_cuda(scales.as_raw_ptr()) };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            h,
            "hy4_nvfp4_swizzle_scales_cuda",
        )?))
    }
}

/// Dense NVFP4 small-M matmul (`M in 1..=16`, `K % 64 == 0`, BF16 -> BF16).
#[inline]
pub fn nvfp4_few(x: &MxArray, w: &MxArray, scales: &MxArray) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h =
        unsafe { sys::mlx_hy4_nvfp4_few_cuda(x.as_raw_ptr(), w.as_raw_ptr(), scales.as_raw_ptr()) };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_nvfp4_few_cuda")?))
    }
}

/// Dense NVFP4 prefill matmul (`CublasQQMM` FP4×FP4 or exact BF16 TC MMA).
#[inline]
pub fn nvfp4_prefill(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    swizzled_scales: Option<&MxArray>,
    mode: Nvfp4PrefillMode,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let sw_ptr = swizzled_scales.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let h = unsafe {
        sys::mlx_hy4_nvfp4_prefill_cuda(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            sw_ptr,
            mode as i32,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_nvfp4_prefill_cuda")?))
    }
}

/// Fused LM head returning `f32 [M, V]` logits.
#[inline]
pub fn lm_head_logits(
    x: &MxArray,
    w: &MxArray,
    scales: Option<&MxArray>,
    weight_mode: LmHeadWeightMode,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let s_ptr = scales.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_a: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_hy4_lm_head_cuda(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            s_ptr,
            weight_mode as i32,
            0,
            &mut out_a,
            &mut out_b,
        )
    };
    if !ok || out_a.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            out_a,
            "hy4_lm_head_logits_cuda",
        )?))
    }
}

/// 2-stage zero-DRAM GreedyArgmax LM head returning `(argmax_ids: i32 [M], max_vals: f32 [M])`.
#[allow(dead_code)]
#[inline]
pub fn lm_head_greedy_argmax(
    x: &MxArray,
    w: &MxArray,
    scales: Option<&MxArray>,
    weight_mode: LmHeadWeightMode,
) -> Result<Option<(MxArray, MxArray)>> {
    if !available() {
        return Ok(None);
    }
    let s_ptr = scales.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_idx: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_val: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_hy4_lm_head_cuda(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            s_ptr,
            weight_mode as i32,
            1,
            &mut out_idx,
            &mut out_val,
        )
    };
    if !ok || out_idx.is_null() || out_val.is_null() {
        Ok(None)
    } else {
        Ok(Some((
            MxArray::from_handle(out_idx, "hy4_lm_head_argmax_idx")?,
            MxArray::from_handle(out_val, "hy4_lm_head_argmax_val")?,
        )))
    }
}

/// Grouped SymQ8 matmul for MLA `attn_k_b` (`[H, 512, 192]`) and `attn_v_b`
/// (`[H, 256, 512]` with optional fused `* sigmoid(gate)`).
#[inline]
pub fn grouped_q8(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    gate: Option<&MxArray>,
    n_heads: i32,
    in_dim: i32,
    out_dim: i32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let g_ptr = gate.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let h = unsafe {
        sys::mlx_hy4_grouped_q8_cuda(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            g_ptr,
            n_heads,
            in_dim,
            out_dim,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_grouped_q8_cuda")?))
    }
}

/// Hy4 MoE sigmoid + optional bias top-K router (ties -> lowest ID, ascending expert order).
#[inline]
pub fn moe_router_topk(
    x: &MxArray,
    router_w: &MxArray,
    exp_probs_b: Option<&MxArray>,
    top_k: i32,
    route_scale: f32,
    norm_topk: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    if !available() {
        return Ok(None);
    }
    let b_ptr = exp_probs_b.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_ids: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_weights: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_hy4_moe_router_topk_cuda(
            x.as_raw_ptr(),
            router_w.as_raw_ptr(),
            b_ptr,
            top_k,
            route_scale,
            norm_topk,
            &mut out_ids,
            &mut out_weights,
        )
    };
    if !ok || out_ids.is_null() || out_weights.is_null() {
        Ok(None)
    } else {
        Ok(Some((
            MxArray::from_handle(out_ids, "hy4_moe_router_ids")?,
            MxArray::from_handle(out_weights, "hy4_moe_router_weights")?,
        )))
    }
}

/// Hy4 MoE decode/verify (`M in 1..=16`, NVFP4 or SymQ8, asymmetric clamped SwiGLU).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn moe_routed_few(
    z: &MxArray,
    topk_ids: &MxArray,
    topk_weights: &MxArray,
    gate_w: &MxArray,
    gate_s: &MxArray,
    up_w: &MxArray,
    up_s: &MxArray,
    down_w: &MxArray,
    down_s: &MxArray,
    mode: MoeWeightMode,
    swiglu_limit: f32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h = unsafe {
        sys::mlx_hy4_moe_routed_few_cuda(
            z.as_raw_ptr(),
            topk_ids.as_raw_ptr(),
            topk_weights.as_raw_ptr(),
            gate_w.as_raw_ptr(),
            gate_s.as_raw_ptr(),
            up_w.as_raw_ptr(),
            up_s.as_raw_ptr(),
            down_w.as_raw_ptr(),
            down_s.as_raw_ptr(),
            mode as i32,
            swiglu_limit,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_moe_routed_few_cuda")?))
    }
}

/// Hy4 MoE prefill (`M >= 1`, Zipfian `BM=64` hot / `BM=32` cold, NVFP4 or SymQ8).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn moe_routed_prefill(
    z: &MxArray,
    topk_ids: &MxArray,
    topk_weights: &MxArray,
    gate_w: &MxArray,
    gate_s: &MxArray,
    up_w: &MxArray,
    up_s: &MxArray,
    down_w: &MxArray,
    down_s: &MxArray,
    mode: MoeWeightMode,
    swiglu_limit: f32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h = unsafe {
        sys::mlx_hy4_moe_routed_prefill_cuda(
            z.as_raw_ptr(),
            topk_ids.as_raw_ptr(),
            topk_weights.as_raw_ptr(),
            gate_w.as_raw_ptr(),
            gate_s.as_raw_ptr(),
            up_w.as_raw_ptr(),
            up_s.as_raw_ptr(),
            down_w.as_raw_ptr(),
            down_s.as_raw_ptr(),
            mode as i32,
            swiglu_limit,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            h,
            "hy4_moe_routed_prefill_cuda",
        )?))
    }
}

/// Fused 4-stream HC pre-gate + mix + optional RMSNorm -> `(h_normed: bf16 [M, D], post: f32 [M, HC])`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn hc_pre_norm(
    x: &MxArray,
    w_fn: &MxArray,
    s_fn: Option<&MxArray>,
    scale: &MxArray,
    base: &MxArray,
    norm_w: Option<&MxArray>,
    fn_mode: HcFnMode,
    norm_eps: f32,
    hc_eps: f32,
    hc_mag: f32,
) -> Result<Option<(MxArray, MxArray)>> {
    if !available() {
        return Ok(None);
    }
    let s_ptr = s_fn.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let nw_ptr = norm_w.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_h: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_post: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_hy4_hc_pre_cuda(
            x.as_raw_ptr(),
            w_fn.as_raw_ptr(),
            s_ptr,
            scale.as_raw_ptr(),
            base.as_raw_ptr(),
            nw_ptr,
            fn_mode as i32,
            norm_eps,
            hc_eps,
            hc_mag,
            false,
            &mut out_h,
            &mut out_post,
        )
    };
    if !ok || out_h.is_null() || out_post.is_null() {
        Ok(None)
    } else {
        Ok(Some((
            MxArray::from_handle(out_h, "hy4_hc_pre_h")?,
            MxArray::from_handle(out_post, "hy4_hc_pre_post")?,
        )))
    }
}

/// Fused 4-stream output HC head + `output_norm` -> `bf16 [M, D]`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn hc_head_norm(
    x: &MxArray,
    w_fn: &MxArray,
    s_fn: Option<&MxArray>,
    scale: &MxArray,
    base: &MxArray,
    output_norm: &MxArray,
    fn_mode: HcFnMode,
    norm_eps: f32,
    hc_eps: f32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let s_ptr = s_fn.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let h = unsafe {
        sys::mlx_hy4_hc_head_cuda(
            x.as_raw_ptr(),
            w_fn.as_raw_ptr(),
            s_ptr,
            scale.as_raw_ptr(),
            base.as_raw_ptr(),
            output_norm.as_raw_ptr(),
            fn_mode as i32,
            norm_eps,
            hc_eps,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_hc_head_cuda")?))
    }
}

/// Fused post-norm + 4-stream HC identity scatter (`X + post * rms_norm(y, post_norm_w)`).
#[inline]
pub fn hc_post_norm(
    x: &MxArray,
    y: &MxArray,
    post: &MxArray,
    post_norm_w: Option<&MxArray>,
    norm_eps: f32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let nw_ptr = post_norm_w.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let h = unsafe {
        sys::mlx_hy4_hc_post_cuda(
            x.as_raw_ptr(),
            y.as_raw_ptr(),
            post.as_raw_ptr(),
            nw_ptr,
            norm_eps,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_hc_post_cuda")?))
    }
}

/// Fused 4-stream HC post-norm + residual scatter + next HC pre-gate + mix + optional RMSNorm
/// -> `(x_next: bf16 [M, HC, D], h_out: bf16 [M, D], next_post: f32 [M, HC])`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn hc_post_pre_norm(
    x: &MxArray,
    y: &MxArray,
    prev_post: &MxArray,
    prev_post_norm_w: Option<&MxArray>,
    w_fn: &MxArray,
    s_fn: Option<&MxArray>,
    scale: &MxArray,
    base: &MxArray,
    next_norm_w: Option<&MxArray>,
    fn_mode: HcFnMode,
    norm_eps: f32,
    hc_eps: f32,
    hc_mag: f32,
) -> Result<Option<(MxArray, MxArray, MxArray)>> {
    if !available() {
        return Ok(None);
    }
    let pnw_ptr = prev_post_norm_w.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let s_ptr = s_fn.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let nnw_ptr = next_norm_w.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_x: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_h: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_post: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_hy4_hc_post_pre_cuda(
            x.as_raw_ptr(),
            y.as_raw_ptr(),
            prev_post.as_raw_ptr(),
            pnw_ptr,
            w_fn.as_raw_ptr(),
            s_ptr,
            scale.as_raw_ptr(),
            base.as_raw_ptr(),
            nnw_ptr,
            fn_mode as i32,
            norm_eps,
            hc_eps,
            hc_mag,
            &mut out_x,
            &mut out_h,
            &mut out_post,
        )
    };
    if !ok || out_x.is_null() || out_h.is_null() || out_post.is_null() {
        Ok(None)
    } else {
        Ok(Some((
            MxArray::from_handle(out_x, "hy4_hc_post_pre_x")?,
            MxArray::from_handle(out_h, "hy4_hc_post_pre_h")?,
            MxArray::from_handle(out_post, "hy4_hc_post_pre_post")?,
        )))
    }
}

/// Engram per-column lookup RMSNorm (`kEngramEps`) -> flattened `bf16 [M, n_cols * e_hd]`.
#[inline]
pub fn engram_lookup_norm(raw_rows: &MxArray, lookup_norm: &MxArray) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h = unsafe {
        sys::mlx_hy4_engram_lookup_norm_cuda(raw_rows.as_raw_ptr(), lookup_norm.as_raw_ptr())
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            h,
            "hy4_engram_lookup_norm_cuda",
        )?))
    }
}

/// Fused Engram stream dot + `sign_{+0}` gate + `pre_gamma` + `hc_scale` + `keep_mask` residual add.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn engram_gate_add(
    x: &MxArray,
    kv: &MxArray,
    kv_b: &MxArray,
    gate_w: &MxArray,
    pre_gamma: &MxArray,
    hc_scale: &MxArray,
    keep_mask: Option<&MxArray>,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let km_ptr = keep_mask.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let h = unsafe {
        sys::mlx_hy4_engram_gate_add_cuda(
            x.as_raw_ptr(),
            kv.as_raw_ptr(),
            kv_b.as_raw_ptr(),
            gate_w.as_raw_ptr(),
            pre_gamma.as_raw_ptr(),
            hc_scale.as_raw_ptr(),
            km_ptr,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_engram_gate_add_cuda")?))
    }
}

/// Fused MTP `[rms_norm(emb, enorm) | rms_norm(hidden, hnorm)]` -> `bf16 [M, 2*D]`.
#[inline]
pub fn mtp_eh_norm_concat(
    emb: &MxArray,
    hidden: &MxArray,
    enorm: &MxArray,
    hnorm: &MxArray,
    norm_eps: f32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h = unsafe {
        sys::mlx_hy4_mtp_eh_norm_concat_cuda(
            emb.as_raw_ptr(),
            hidden.as_raw_ptr(),
            enorm.as_raw_ptr(),
            hnorm.as_raw_ptr(),
            norm_eps,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            h,
            "hy4_mtp_eh_norm_concat_cuda",
        )?))
    }
}

/// GPU-native ring / slab write (`f16`/`f32`/`bf16`, `ring_cap > 0` = circular, `<= 0` = linear).
/// Preserves copy-on-write when `buf` has other live references (`is_donatable`).
#[inline]
pub fn ring_write(
    buf: &MxArray,
    rows: &MxArray,
    start_pos: i64,
    ring_cap: i32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h = unsafe {
        sys::mlx_hy4_ring_write_cuda(
            buf.as_raw_ptr(),
            rows.as_raw_ptr(),
            start_pos,
            ring_cap,
            false,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_ring_write_cuda")?))
    }
}

/// GPU-native aliased in-place ring / slab write (`f16`/`f32`/`bf16`, `ring_cap > 0` = circular, `<= 0` = linear).
/// Caller MUST guarantee that no lazy reader depends on overwritten rows of `buf`.
#[inline]
pub fn ring_write_in_place(
    buf: &MxArray,
    rows: &MxArray,
    start_pos: i64,
    ring_cap: i32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h = unsafe {
        sys::mlx_hy4_ring_write_cuda(
            buf.as_raw_ptr(),
            rows.as_raw_ptr(),
            start_pos,
            ring_cap,
            true,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            h,
            "hy4_ring_write_in_place_cuda",
        )?))
    }
}

/// Fused MLA `q_a_norm` (optional) + `kv_a_norm` + NORM-RoPE (`rope_start`) + in-place `f16` KV store (`slot_start`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn mla_prep_kv_store(
    q_a: Option<&MxArray>,
    kv_a: &MxArray,
    q_a_norm: Option<&MxArray>,
    kv_a_norm: &MxArray,
    inv_freq: &MxArray,
    kv_buf: &MxArray,
    slot_start: i64,
    rope_start: i64,
    ring_cap: i32,
    norm_eps: f32,
) -> Result<Option<(Option<MxArray>, MxArray)>> {
    if !available() {
        return Ok(None);
    }
    let qa_ptr = q_a.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let qn_ptr = q_a_norm.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out_qr: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_kv: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_hy4_mla_prep_kv_store_cuda(
            qa_ptr,
            kv_a.as_raw_ptr(),
            qn_ptr,
            kv_a_norm.as_raw_ptr(),
            inv_freq.as_raw_ptr(),
            kv_buf.as_raw_ptr(),
            slot_start,
            rope_start,
            ring_cap,
            norm_eps,
            &mut out_qr,
            &mut out_kv,
        )
    };
    if !ok || out_kv.is_null() {
        Ok(None)
    } else {
        let qr = if out_qr.is_null() {
            None
        } else {
            Some(MxArray::from_handle(out_qr, "hy4_mla_prep_qr")?)
        };
        let kv = MxArray::from_handle(out_kv, "hy4_mla_prep_kv_buf")?;
        Ok(Some((qr, kv)))
    }
}

/// Fused NORM-RoPE on `q_raw[..., nope_dim:]` + concat `[q_abs | q_pe]` -> `bf16 [M, H, kv_lora + rope]`.
#[inline]
pub fn rope_norm_q(
    q_abs: &MxArray,
    q_raw: &MxArray,
    inv_freq: &MxArray,
    rope_start: i64,
    nope_dim: i32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h = unsafe {
        sys::mlx_hy4_rope_norm_q_cuda(
            q_abs.as_raw_ptr(),
            q_raw.as_raw_ptr(),
            inv_freq.as_raw_ptr(),
            rope_start,
            nope_dim,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_rope_norm_q_cuda")?))
    }
}

/// Absorbed MLA Sink Softmax SDPA (`D_qk=576, D_v=512`, SWA ring / linear / sparse indexed).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn mla_sdpa(
    q: &MxArray,
    kv_buf: &MxArray,
    sinks: &MxArray,
    idx: Option<&MxArray>,
    slot_start: i64,
    total_len: i32,
    ring_cap: i32,
    window: i32,
    scale: f32,
    kv_lora: i32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let idx_ptr = idx.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let h = unsafe {
        sys::mlx_hy4_mla_sdpa_cuda(
            q.as_raw_ptr(),
            kv_buf.as_raw_ptr(),
            sinks.as_raw_ptr(),
            idx_ptr,
            slot_start,
            total_len,
            ring_cap,
            window,
            scale,
            kv_lora,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_mla_sdpa_cuda")?))
    }
}

/// Fused K-pool compression (`kpool=4` per-channel softmax + `comp_norm` + tail NORM-RoPE -> `f16` slab).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn kpool_compress(
    tok_k: &MxArray,
    w_val: &MxArray,
    w_gate: &MxArray,
    ape: &MxArray,
    comp_norm: &MxArray,
    inv_freq: &MxArray,
    pooled_buf: &MxArray,
    start_group: i64,
    kpool: i32,
    norm_eps: f32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h = unsafe {
        sys::mlx_hy4_kpool_compress_cuda(
            tok_k.as_raw_ptr(),
            w_val.as_raw_ptr(),
            w_gate.as_raw_ptr(),
            ape.as_raw_ptr(),
            comp_norm.as_raw_ptr(),
            inv_freq.as_raw_ptr(),
            pooled_buf.as_raw_ptr(),
            start_group,
            kpool,
            norm_eps,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_kpool_compress_cuda")?))
    }
}

/// Fused `16×128` Lightning Indexer tail NORM-RoPE + scoring + Radix top-K + ascending expand (`i32 [M, kmax]`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn indexer_select(
    q_idx_raw: &MxArray,
    w_proj: &MxArray,
    pooled_buf: &MxArray,
    inv_freq: &MxArray,
    slot_start: i64,
    rope_start: i64,
    kpool: i32,
    top_k: i32,
    select_tail: bool,
    weight_scale: f32,
) -> Result<Option<MxArray>> {
    if !available() {
        return Ok(None);
    }
    let h = unsafe {
        sys::mlx_hy4_indexer_select_cuda(
            q_idx_raw.as_raw_ptr(),
            w_proj.as_raw_ptr(),
            pooled_buf.as_raw_ptr(),
            inv_freq.as_raw_ptr(),
            slot_start,
            rope_start,
            kpool,
            top_k,
            select_tail,
            weight_scale,
        )
    };
    if h.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(h, "hy4_indexer_select_cuda")?))
    }
}
