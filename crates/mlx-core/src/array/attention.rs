//! Scaled Dot-Product Attention operations
//!
//! This module provides efficient attention implementations using MLX's optimized kernels.

use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::ffi::CString;

use super::MxArray;

/// Scaled dot-product attention using MLX's optimized kernel.
///
/// Computes: O = softmax(scale * (Q @ K^T)) @ V
///
/// # Arguments
/// * `queries` - Query tensor [batch, n_heads, seq_len, head_dim]
/// * `keys` - Key tensor [batch, n_heads, seq_len, head_dim]
/// * `values` - Value tensor [batch, n_heads, seq_len, head_dim]
/// * `scale` - Scale factor (typically 1/sqrt(head_dim))
/// * `mask` - Optional attention mask (None for no mask)
///
/// # Returns
/// Attention output with same shape as values
#[inline]
pub fn scaled_dot_product_attention(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    scale: f64,
    mask: Option<&MxArray>,
) -> Result<MxArray> {
    let handle = unsafe {
        if let Some(m) = mask {
            // Use empty string with mask array - MLX will apply it as an array mask
            let mask_mode = c"";
            sys::mlx_fast_scaled_dot_product_attention(
                queries.handle.0,
                keys.handle.0,
                values.handle.0,
                scale as f32,
                mask_mode.as_ptr(),
                m.handle.0,
                true,
            )
        } else {
            // No mask - pass empty string
            let mask_mode = c"";
            sys::mlx_fast_scaled_dot_product_attention(
                queries.handle.0,
                keys.handle.0,
                values.handle.0,
                scale as f32,
                mask_mode.as_ptr(),
                std::ptr::null_mut(),
                false,
            )
        }
    };
    MxArray::from_handle(handle, "scaled_dot_product_attention")
}

/// Scaled dot-product attention with "causal" mask mode.
///
/// Uses MLX's optimized internal causal masking (no explicit mask array needed).
/// This is faster than passing an explicit mask because MLX handles it internally
/// with an optimized kernel.
///
/// # Arguments
/// * `queries` - Query tensor [batch, n_heads, seq_len, head_dim]
/// * `keys` - Key tensor [batch, n_heads, kv_len, head_dim]
/// * `values` - Value tensor [batch, n_heads, kv_len, head_dim]
/// * `scale` - Scale factor (typically 1/sqrt(head_dim))
///
/// # Returns
/// Attention output with same shape as values
pub fn scaled_dot_product_attention_causal(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    scale: f64,
) -> Result<MxArray> {
    let handle = unsafe {
        // Use "causal" mode - MLX handles causal masking internally with optimized kernel
        let mask_mode = CString::new("causal").unwrap();
        sys::mlx_fast_scaled_dot_product_attention(
            queries.handle.0,
            keys.handle.0,
            values.handle.0,
            scale as f32,
            mask_mode.as_ptr(),
            std::ptr::null_mut(),
            false,
        )
    };
    MxArray::from_handle(handle, "scaled_dot_product_attention_causal")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::mask::create_causal_mask;

    fn deterministic_data(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = i as f32 + phase;
                (x.sin() * 0.5) + (x.cos() * 0.25)
            })
            .collect()
    }

    #[test]
    fn causal_attention_matches_explicit_offset_mask_when_kv_is_longer() {
        let batch = 1;
        let heads = 2;
        let q_len = 4;
        let kv_len = 9;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q = MxArray::from_float32(
            &deterministic_data(batch * heads * q_len * head_dim, 0.0),
            &[batch as i64, heads as i64, q_len as i64, head_dim as i64],
        )
        .unwrap();
        let k = MxArray::from_float32(
            &deterministic_data(batch * heads * kv_len * head_dim, 1.0),
            &[batch as i64, heads as i64, kv_len as i64, head_dim as i64],
        )
        .unwrap();
        let v = MxArray::from_float32(
            &deterministic_data(batch * heads * kv_len * head_dim, 2.0),
            &[batch as i64, heads as i64, kv_len as i64, head_dim as i64],
        )
        .unwrap();

        let mask = create_causal_mask(q_len as i32, Some((kv_len - q_len) as i32), None).unwrap();
        let explicit = scaled_dot_product_attention(&q, &k, &v, scale, Some(&mask)).unwrap();
        let causal = scaled_dot_product_attention_causal(&q, &k, &v, scale).unwrap();

        let explicit = explicit.to_float32().unwrap();
        let causal = causal.to_float32().unwrap();
        assert_eq!(explicit.len(), causal.len());
        for (idx, (a, b)) in explicit.iter().zip(causal.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= 1e-4,
                "causal SDPA diverged from explicit offset mask at {idx}: {a} vs {b} (diff {diff})"
            );
        }
    }
}
