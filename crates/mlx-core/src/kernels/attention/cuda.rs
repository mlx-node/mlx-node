//! CUDA Tensor Core attention & QK-Norm/RoPE kernel wrappers.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};

#[inline]
pub(super) fn sdpa_causal_few_q1_to_16(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    head_dim: i64,
    gqa: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 96;
    if !(1..=16).contains(&seq_len)
        || head_dim != 256
        || gqa <= 1
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_tc_few_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            scale,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "sdpa_tc_few_cuda")?))
    }
}

#[inline]
pub(super) fn sdpa_prefill_d256(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    if seq_len < 9
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_prefill_d256_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            scale,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_prefill_d256_gated(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    gate: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    if seq_len < 9
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || gate.dtype()? != DType::BFloat16
        || gate.ndim()? != 3
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_cuda_sdpa_prefill_d256_gated(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            gate.as_raw_ptr(),
            std::ptr::null_mut(),
            scale,
            true,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_gated_cuda",
        )?))
    }
}

#[inline]
pub(super) fn sdpa_unmasked_few(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    head_dim: i64,
    gqa: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 96;
    if !(1..=16).contains(&seq_len)
        || !(head_dim == 128 || head_dim == 256)
        || gqa < 1
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_tc_few_masked_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            std::ptr::null_mut(),
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_tc_few_unmasked_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_masked_few(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: &MxArray,
    head_dim: i64,
    gqa: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 96;
    // Note: Rust `DType` has no `Bool` variant (`mlx_array_dtype` returns -1
    // for `bool_`, which `MxArray::dtype()` maps to `Err`), so C++ enforces
    // `mask.dtype() == bool_` and returns null on non-boolean masks.
    if !(1..=16).contains(&seq_len)
        || !(head_dim == 128 || head_dim == 256)
        || gqa < 1
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || mask.ndim()? != 4
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_tc_few_masked_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            mask.as_raw_ptr(),
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_tc_few_masked_cuda",
        )?))
    }
}

#[inline]
pub(super) fn sdpa_prefill_d256_masked(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    // C++ enforces `mask.dtype() == bool_` and returns null on non-boolean masks.
    if seq_len < 9
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || mask.ndim()? != 4
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_prefill_d256_masked_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            mask.as_raw_ptr(),
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_masked_cuda",
        )?))
    }
}

#[inline]
pub(super) fn sdpa_prefill_d256_causal_valid(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    valid_mask: &MxArray,
    scale: f32,
) -> Result<Option<MxArray>> {
    // C++ enforces `valid_mask.dtype() == bool_` (Rust `DType` has no `Bool`
    // variant, so do not call `valid_mask.dtype()`) and returns null on
    // non-boolean masks or unsupported shapes.
    if queries.ndim()? != 4
        || keys.ndim()? != 4
        || values.ndim()? != 4
        || valid_mask.ndim()? != 4
        || valid_mask.shape_at(2)? != 1
    {
        return Ok(None);
    }
    let seq_len = queries.shape_at(2)?;
    let head_dim = queries.shape_at(3)?;
    let kv_len = keys.shape_at(2)?;
    if seq_len < 9
        || kv_len < seq_len
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_prefill_d256_masked_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            valid_mask.as_raw_ptr(),
            scale,
            true,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_causal_valid_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_prefill_d256_masked_gated(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    gate: &MxArray,
    mask: &MxArray,
    head_dim: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    // C++ enforces `mask.dtype() == bool_` and returns null on non-boolean masks.
    if seq_len < 9
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || gate.dtype()? != DType::BFloat16
        || gate.ndim()? != 3
        || mask.ndim()? != 4
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_cuda_sdpa_prefill_d256_gated(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            gate.as_raw_ptr(),
            mask.as_raw_ptr(),
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_masked_gated_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_prefill_d256_causal_valid_gated(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    gate: &MxArray,
    valid_mask: &MxArray,
    scale: f32,
) -> Result<Option<MxArray>> {
    // C++ enforces `valid_mask.dtype() == bool_` (Rust `DType` has no `Bool`
    // variant, so do not call `valid_mask.dtype()`) and returns null on
    // non-boolean masks or unsupported shapes.
    if queries.ndim()? != 4
        || keys.ndim()? != 4
        || values.ndim()? != 4
        || gate.ndim()? != 3
        || valid_mask.ndim()? != 4
        || valid_mask.shape_at(2)? != 1
    {
        return Ok(None);
    }
    let seq_len = queries.shape_at(2)?;
    let head_dim = queries.shape_at(3)?;
    let kv_len = keys.shape_at(2)?;
    if seq_len < 9
        || kv_len < seq_len
        || head_dim != 256
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || gate.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_cuda_sdpa_prefill_d256_gated(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            gate.as_raw_ptr(),
            valid_mask.as_raw_ptr(),
            scale,
            true,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "sdpa_prefill_d256_causal_valid_gated_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_draft_masked(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: Option<&MxArray>,
    head_dim: i64,
    num_heads: i64,
    num_kv_heads: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    let gqa = num_heads / num_kv_heads;
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 96;
    if !(1..=16).contains(&seq_len)
        || !(head_dim == 128 || head_dim == 256)
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let mask_ptr = match mask {
        Some(m) if m.ndim()? == 4 && m.shape_at(0)? == 1 && m.shape_at(1)? == 1 => m.as_raw_ptr(),
        None => std::ptr::null_mut(),
        _ => return Ok(None),
    };
    let handle = unsafe {
        sys::mlx_sdpa_tc_few_masked_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            mask_ptr,
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "dflash2 sdpa_tc_few_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_draft_masked_bthd(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    mask: Option<&MxArray>,
    head_dim: i64,
    num_heads: i64,
    num_kv_heads: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    if num_kv_heads <= 0 || num_heads < num_kv_heads || num_heads % num_kv_heads != 0 {
        return Ok(None);
    }
    let gqa = num_heads / num_kv_heads;
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 96;
    if !(1..=16).contains(&seq_len)
        || !(head_dim == 128 || head_dim == 256)
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
    {
        return Ok(None);
    }
    let mask_ptr = match mask {
        Some(m) if m.ndim()? == 4 && m.shape_at(0)? == 1 && m.shape_at(1)? == 1 => m.as_raw_ptr(),
        None => std::ptr::null_mut(),
        _ => return Ok(None),
    };
    let handle = unsafe {
        sys::mlx_sdpa_tc_few_masked_bthd_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            mask_ptr,
            scale,
            false,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "dflash2 sdpa_tc_few_bthd_cuda",
        )?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn qk_norm_rope_offsets(
    queries: &MxArray,
    keys: &MxArray,
    q_weight: &MxArray,
    k_weight: &MxArray,
    offsets: &MxArray,
    eps: f32,
    rope_base: f32,
    rope_scale: f32,
    rope_dims: i32,
) -> Result<Option<(MxArray, MxArray)>> {
    let mut out_q = std::ptr::null_mut();
    let mut out_k = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qk_norm_rope_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            q_weight.as_raw_ptr(),
            k_weight.as_raw_ptr(),
            offsets.as_raw_ptr(),
            eps,
            rope_base,
            rope_scale,
            rope_dims,
            &mut out_q,
            &mut out_k,
        )
    };
    if !ok || out_q.is_null() || out_k.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_q, "qk_norm_rope:q")?,
        MxArray::from_handle(out_k, "qk_norm_rope:k")?,
    )))
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn qk_norm_rope_positions(
    queries: &MxArray,
    keys: &MxArray,
    q_norm_w: &MxArray,
    k_norm_w: &MxArray,
    positions: &MxArray,
    eps: f32,
    rope_dims: i32,
    rope_base: f32,
    rope_scale: f32,
    precise_f32: bool,
) -> Result<Option<(MxArray, MxArray)>> {
    let expected_w_dt = if precise_f32 {
        DType::Float32
    } else {
        DType::BFloat16
    };
    if queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || q_norm_w.dtype()? != expected_w_dt
        || k_norm_w.dtype()? != expected_w_dt
        || positions.dtype()? != DType::Int32
        || positions.ndim()? != 2
    {
        return Ok(None);
    }
    let mut out_q = std::ptr::null_mut();
    let mut out_k = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_qk_norm_rope_positions_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            q_norm_w.as_raw_ptr(),
            k_norm_w.as_raw_ptr(),
            positions.as_raw_ptr(),
            eps,
            rope_dims,
            rope_base,
            rope_scale,
            precise_f32,
            &mut out_q,
            &mut out_k,
        )
    };
    if !ok || out_q.is_null() || out_k.is_null() {
        return Ok(None);
    }
    Ok(Some((
        MxArray::from_handle(out_q, "qk_norm_rope_positions:q")?,
        MxArray::from_handle(out_k, "qk_norm_rope_positions:k")?,
    )))
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sdpa_tree_few(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    tree_mask: &MxArray,
    cache_len: i32,
    head_dim: i64,
    gqa: i64,
    seq_len: i64,
    scale: f32,
) -> Result<Option<MxArray>> {
    let m = gqa * seq_len;
    let ok_single = (1..=96).contains(&m);
    let ok_two_tile = seq_len > 8 && gqa * 8 <= 96;
    if !(1..=16).contains(&seq_len)
        || !(head_dim == 128 || head_dim == 256)
        || cache_len < 0
        || (!ok_single && !ok_two_tile)
        || queries.dtype()? != DType::BFloat16
        || keys.dtype()? != DType::BFloat16
        || values.dtype()? != DType::BFloat16
        || tree_mask.dtype()? != DType::Uint32
    {
        return Ok(None);
    }
    let handle = unsafe {
        sys::mlx_sdpa_tc_tree_few_cuda(
            queries.as_raw_ptr(),
            keys.as_raw_ptr(),
            values.as_raw_ptr(),
            tree_mask.as_raw_ptr(),
            cache_len,
            scale,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "sdpa_tc_tree_few_cuda")?))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{sdpa_draft_masked, sdpa_draft_masked_bthd};
    use crate::accelerator::{ActiveBackend, active_backend};
    use crate::array::{DType, MxArray};
    use half::bf16;

    /// Pure FP32 CPU reference SDPA (`[B, Hq, T, D]` x `[B, Hkv, K, D]` -> `[B, T, Hq * D]`),
    /// independent of cuDNN so `cargo test -p mlx-core --lib kernels::attention::` runs with
    /// default env vars on every CUDA platform.
    fn ref_sdpa_bthd_cpu(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[u8]>,
        b: usize,
        hq: usize,
        hkv: usize,
        t: usize,
        k_len: usize,
        d: usize,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = hq / hkv;
        let mut out = vec![0.0f32; b * t * hq * d];
        let mut scores = vec![0.0f32; k_len];
        for bi in 0..b {
            for h in 0..hq {
                let kv_h = h / gqa;
                for qi in 0..t {
                    let q_off = ((bi * hq + h) * t + qi) * d;
                    let mut m = f32::NEG_INFINITY;
                    for ki in 0..k_len {
                        let visible = mask.is_none_or(|msk| msk[qi * k_len + ki] != 0);
                        if !visible {
                            scores[ki] = f32::NEG_INFINITY;
                            continue;
                        }
                        let k_off = ((bi * hkv + kv_h) * k_len + ki) * d;
                        let mut dot = 0.0f32;
                        for di in 0..d {
                            dot += q[q_off + di] * k[k_off + di];
                        }
                        let s = dot * scale;
                        scores[ki] = s;
                        if s > m {
                            m = s;
                        }
                    }
                    let mut l = 0.0f32;
                    for s in &mut scores {
                        if s.is_finite() {
                            *s = (*s - m).exp();
                            l += *s;
                        } else {
                            *s = 0.0;
                        }
                    }
                    let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                    let out_off = ((bi * t + qi) * hq + h) * d;
                    for di in 0..d {
                        let mut acc = 0.0f32;
                        for (ki, &sc) in scores.iter().enumerate().take(k_len) {
                            let w = sc * inv_l;
                            if w != 0.0 {
                                let v_off = ((bi * hkv + kv_h) * k_len + ki) * d;
                                acc += w * v[v_off + di];
                            }
                        }
                        out[out_off + di] = acc;
                    }
                }
            }
        }
        out
    }

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * ((self.next_u32() & 0x00FF_FFFF) as f32 / 16_777_216.0)
        }
        fn bf16(&mut self, shape: &[i64], scale: f32) -> MxArray {
            let len: usize = shape.iter().map(|&d| d as usize).product();
            let vals: Vec<f32> = (0..len).map(|_| self.uniform(-scale, scale)).collect();
            MxArray::from_float32(&vals, shape)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap()
        }
    }

    fn read_bf16_bits(arr: &MxArray) -> Vec<u16> {
        arr.eval();
        arr.to_uint16_native().unwrap().to_vec()
    }

    fn bits_to_f32(bits: &[u16]) -> Vec<f32> {
        bits.iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
    }

    #[test]
    fn cuda_sdpa_draft_masked_bthd_matches_bhtd_and_reference() {
        if active_backend() != ActiveBackend::CudaGpu {
            return;
        }
        let mut rng = Lcg::new(0x5D9A_B74D);

        // Inkling-2-Mini DFlash2 exact shape (`D=128, B=1, Hq=32, Hkv=8, T=8`),
        // plus two-tile `T=16` and `D=256`, both unmasked and boolean-masked.
        for &(b, hq, hkv, t, k_len, d) in &[
            (1i64, 32i64, 8i64, 8i64, 96i64, 128i64),
            (1, 32, 8, 8, 520, 128),
            (1, 32, 8, 16, 128, 128),
            (1, 24, 4, 8, 80, 256),
        ] {
            let q = rng.bf16(&[b, hq, t, d], 0.5);
            let k = rng.bf16(&[b, hkv, k_len, d], 0.5);
            let v = rng.bf16(&[b, hkv, k_len, d], 0.5);
            let scale = 1.0f32 / (d as f32).sqrt();

            // Non-trivial sliding + tail boolean mask `[1, 1, T, K_len]`.
            let mut mask_u8 = vec![0u8; (t * k_len) as usize];
            for row in 0..t as usize {
                for col in 0..k_len as usize {
                    let keep = (col >= row
                        && col < (k_len as usize).saturating_sub(t as usize) + row)
                        || col >= (k_len as usize).saturating_sub(t as usize);
                    mask_u8[row * k_len as usize + col] = u8::from(keep);
                }
            }
            let mask = MxArray::from_uint8(&mask_u8, &[1, 1, t, k_len])
                .unwrap()
                .not_equal(&MxArray::from_uint8(&[0], &[1]).unwrap())
                .unwrap();

            for use_mask in [false, true] {
                let m_opt = use_mask.then_some(&mask);
                let out_bthd = sdpa_draft_masked_bthd(&q, &k, &v, m_opt, d, hq, hkv, t, scale)
                    .unwrap()
                    .expect("sdpa_draft_masked_bthd must dispatch on CUDA");
                assert_eq!(out_bthd.dtype().unwrap(), DType::BFloat16);
                assert_eq!(out_bthd.shape().unwrap().as_ref(), &[b, t, hq * d]);

                let out_bhtd = sdpa_draft_masked(&q, &k, &v, m_opt, d, hq, hkv, t, scale, true)
                    .unwrap()
                    .expect("sdpa_draft_masked must dispatch on CUDA")
                    .transpose(Some(&[0, 2, 1, 3]))
                    .unwrap()
                    .reshape(&[b, t, hq * d])
                    .unwrap();

                // Same split + reduce kernels with fused BTHD output indexing -> 0-ULP bit-exact!
                let got_bits = read_bf16_bits(&out_bthd);
                let want_bits = read_bf16_bits(&out_bhtd);
                assert_eq!(
                    got_bits, want_bits,
                    "sdpa_draft_masked_bthd must be bit-exact vs BHTD+transpose (d={d}, t={t}, k={k_len}, mask={use_mask})"
                );

                // Cross-check against exact FP32 CPU reference SDPA.
                let q_f = bits_to_f32(&read_bf16_bits(&q));
                let k_f = bits_to_f32(&read_bf16_bits(&k));
                let v_f = bits_to_f32(&read_bf16_bits(&v));
                let ref_f = ref_sdpa_bthd_cpu(
                    &q_f,
                    &k_f,
                    &v_f,
                    use_mask.then_some(mask_u8.as_slice()),
                    b as usize,
                    hq as usize,
                    hkv as usize,
                    t as usize,
                    k_len as usize,
                    d as usize,
                    scale,
                );
                let got_f = bits_to_f32(&got_bits);
                let max_abs = got_f
                    .iter()
                    .zip(ref_f.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs <= 2.5e-2,
                    "sdpa_draft_masked_bthd vs CPU SDPA ref max_abs={max_abs:.4e} (d={d}, t={t}, k={k_len}, mask={use_mask})"
                );
            }
        }
    }
}
