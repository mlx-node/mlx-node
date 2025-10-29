//! Scaled Dot-Product Attention operations
//!
//! This module provides efficient attention implementations using MLX's optimized kernels.

use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
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
#[napi(js_name = "scaledDotProductAttention")]
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
            let mask_mode = CString::new("").unwrap();
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
            let mask_mode = CString::new("").unwrap();
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
#[napi(js_name = "scaledDotProductAttentionCausal")]
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
