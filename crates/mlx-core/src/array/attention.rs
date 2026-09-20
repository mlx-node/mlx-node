//! Scaled Dot-Product Attention operations
//!
//! This module provides efficient attention implementations using MLX's optimized kernels.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

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
        let mask_mode = c"causal";
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

    /// The verify-block split used by `Qwen3_5Attention::forward` when
    /// `q_len * gqa > 32` overflows the fused vector kernel's threadgroup:
    /// the head chunk attends to keys truncated at `kv_len - tail`, the tail
    /// chunk sees the full cache. Both pieces keep causal alignment because
    /// the kernel derives the offset as `kL - qL` per call.
    #[test]
    fn split_causal_attention_matches_unsplit() {
        let batch = 1i64;
        let heads = 24i64;
        let kv_heads = 4i64;
        let head_dim = 64i64;
        let q_len = 7i64;
        let kv_len = 1100i64; // past the 2-pass threshold on Apple Silicon
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q = MxArray::from_float32(
            &deterministic_data((batch * heads * q_len * head_dim) as usize, 0.0),
            &[batch, heads, q_len, head_dim],
        )
        .unwrap();
        let k = MxArray::from_float32(
            &deterministic_data((batch * kv_heads * kv_len * head_dim) as usize, 1.0),
            &[batch, kv_heads, kv_len, head_dim],
        )
        .unwrap();
        let v = MxArray::from_float32(
            &deterministic_data((batch * kv_heads * kv_len * head_dim) as usize, 2.0),
            &[batch, kv_heads, kv_len, head_dim],
        )
        .unwrap();

        let whole = scaled_dot_product_attention_causal(&q, &k, &v, scale).unwrap();

        let gqa = heads / kv_heads;
        let tail = (32 / gqa).min(q_len - 1);
        let head_len = q_len - tail;
        let q_parts = q.split_sections(&[head_len], 2).unwrap();
        let k_head = k.split_sections(&[kv_len - tail], 2).unwrap()[0].clone();
        let v_head = v.split_sections(&[kv_len - tail], 2).unwrap()[0].clone();
        let out_head = if head_len > 1 {
            scaled_dot_product_attention_causal(&q_parts[0], &k_head, &v_head, scale).unwrap()
        } else {
            scaled_dot_product_attention(&q_parts[0], &k_head, &v_head, scale, None).unwrap()
        };
        let out_tail = scaled_dot_product_attention_causal(&q_parts[1], &k, &v, scale).unwrap();
        let split = MxArray::concatenate(&out_head, &out_tail, 2).unwrap();

        let whole = whole.to_float32().unwrap();
        let split = split.to_float32().unwrap();
        assert_eq!(whole.len(), split.len());
        for (idx, (a, b)) in whole.iter().zip(split.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= 1e-3,
                "split causal SDPA diverged from unsplit at {idx}: {a} vs {b} (diff {diff})"
            );
        }
    }
}
