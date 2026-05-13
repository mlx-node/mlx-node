//! Bidirectional banded attention with per-head sinks — reference implementation.
//!
//! This is the **correctness oracle** for the future Metal kernel (Task B2). It
//! composes existing MLX ops (matmul, softmax, mask construction, concat,
//! slice) instead of running a single fused kernel, so it is intentionally
//! slow but provably correct.
//!
//! Architectural note: the eventual kernel-backed primitive lives in
//! `mlx-paged-attn`, but adding `mlx-core` as a dependency there would create a
//! cycle (`mlx-core → mlx-paged-attn` already exists). We therefore keep this
//! reference alongside `attention.rs` in `mlx-core`. The B2 kernel will be able
//! to depend on this crate via `dev-dependencies` for its tests, or duplicate
//! a minimal Rust-side scaffold for input prep.
//!
//! ## Math
//!
//! For query position `q` and key index `k`:
//!
//!   mask(q, k) = 0       if `|q - k| <= band`
//!              = -inf    otherwise
//!
//! The softmax denominator runs over the in-band keys **plus one virtual
//! per-head "sink" entry** with logit `sinks[h]`. The sink contributes to the
//! denominator but its V-projection is implicitly zero, so it shrinks the
//! attention mass routed to real tokens without contributing any value.
//!
//! ## Implementation outline (no manual loops)
//!
//! 1. `logits = (Q @ K^T) * scale`                              → `[B,H,T,T]`
//! 2. additive `[T,T]` band mask (broadcast to `[1,1,T,T]`)
//! 3. concat `sinks` as a `[B,H,T,1]` virtual column            → `[B,H,T,T+1]`
//! 4. softmax over last axis (sink mass is lost to real tokens)
//! 5. slice off the sink column                                 → `[B,H,T,T]`
//! 6. `out = attn @ V`                                          → `[B,H,T,D]`
//!
//! ## GQA
//!
//! When `num_q_heads != num_kv_heads`, K and V are repeat-interleaved along the
//! head axis by `group_size = num_q_heads / num_kv_heads`. This mirrors what
//! the eventual fused kernel will do internally.

use super::{DType, MxArray, scaled_dot_product_attention};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Build the `[T, T]` band mask with `0` inside the band and `-inf` outside.
///
/// Constructed as: `where(|q - k| <= band, 0, -inf)`. The caller is expected to
/// reshape to `[1, 1, T, T]` and add to the (B, H, T, T) logits tensor.
fn build_band_mask(t: i64, band: i32) -> Result<MxArray> {
    if t <= 0 {
        return Err(Error::from_reason(format!(
            "build_band_mask: T must be positive, got {t}"
        )));
    }
    let band_f = band as f64;

    // positions: [T]
    let positions = MxArray::arange(0.0, t as f64, Some(1.0), Some(DType::Float32))?;

    // q_pos: [T, 1], k_pos: [1, T]
    let q_pos = positions.reshape(&[t, 1])?;
    let k_pos = positions.reshape(&[1, t])?;

    // diff = |q - k|  → [T, T]
    let diff = q_pos.sub(&k_pos)?.abs()?;

    // band_scalar broadcasts against diff.
    let band_scalar = MxArray::scalar_float(band_f)?;
    let in_band = diff.less_equal(&band_scalar)?;

    let zero = MxArray::scalar_float(0.0)?;
    let neg_inf = MxArray::scalar_float(f64::NEG_INFINITY)?;

    // where(in_band, 0, -inf)  → [T, T] f32
    in_band.where_(&zero, &neg_inf)
}

/// Softmax along the last axis using MLX's fused primitive (no array dep on
/// `crate::nn::Activations`, to keep `array` self-contained).
fn softmax_last_axis(x: &MxArray) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_array_softmax(x.handle.0, -1) };
    MxArray::from_handle(handle, "banded_attention_softmax")
}

/// Bidirectional banded attention with per-head sinks (reference impl).
///
/// Shapes:
/// - `q`:  `[B, num_q_heads, T, head_dim]`
/// - `k`:  `[B, num_kv_heads, T, head_dim]`
/// - `v`:  `[B, num_kv_heads, T, head_dim]`
/// - `sinks`: `[num_q_heads]`
/// - returns `[B, num_q_heads, T, head_dim]`
///
/// Requires `num_q_heads % num_kv_heads == 0`. K and V must share their first
/// three dimensions (batch, num_kv_heads, T). Query and key sequence lengths
/// must match (this is bidirectional self-attention, not cross-attention).
pub fn banded_attention_reference(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    sinks: &MxArray,
    band: i32,
) -> Result<MxArray> {
    if band < 0 {
        return Err(Error::from_reason(format!(
            "banded_attention_reference: band must be non-negative, got {band}"
        )));
    }

    // Shape extraction + validation.
    if q.ndim()? != 4 || k.ndim()? != 4 || v.ndim()? != 4 {
        return Err(Error::from_reason(
            "banded_attention_reference: Q, K, V must be 4D [B, H, T, D]",
        ));
    }
    if sinks.ndim()? != 1 {
        return Err(Error::from_reason(
            "banded_attention_reference: sinks must be 1D [num_q_heads]",
        ));
    }

    let q_shape = q.shape()?;
    let k_shape = k.shape()?;
    let v_shape = v.shape()?;
    let sinks_shape = sinks.shape()?;
    let q_shape = q_shape.as_ref();
    let k_shape = k_shape.as_ref();
    let v_shape = v_shape.as_ref();
    let sinks_shape = sinks_shape.as_ref();

    let b = q_shape[0];
    let h_q = q_shape[1];
    let t_q = q_shape[2];
    let d = q_shape[3];

    let h_kv = k_shape[1];
    let t_kv = k_shape[2];

    if k_shape[0] != b || v_shape[0] != b {
        return Err(Error::from_reason(
            "banded_attention_reference: batch dim must match between Q, K, V",
        ));
    }
    if k_shape[1] != h_kv || v_shape[1] != h_kv {
        return Err(Error::from_reason(
            "banded_attention_reference: num_kv_heads must match between K and V",
        ));
    }
    if k_shape[2] != t_kv || v_shape[2] != t_kv {
        return Err(Error::from_reason(
            "banded_attention_reference: sequence length must match between K and V",
        ));
    }
    if t_q != t_kv {
        return Err(Error::from_reason(format!(
            "banded_attention_reference: Q/K sequence lengths must match (self-attn), got T_q={t_q}, T_kv={t_kv}"
        )));
    }
    if k_shape[3] != d || v_shape[3] != d {
        return Err(Error::from_reason(
            "banded_attention_reference: head_dim must match between Q, K, V",
        ));
    }
    if h_q % h_kv != 0 {
        return Err(Error::from_reason(format!(
            "banded_attention_reference: num_q_heads ({h_q}) must be a multiple of num_kv_heads ({h_kv})"
        )));
    }
    if sinks_shape[0] != h_q {
        return Err(Error::from_reason(format!(
            "banded_attention_reference: sinks length ({}) must equal num_q_heads ({h_q})",
            sinks_shape[0]
        )));
    }

    let group_size = (h_q / h_kv) as i32;
    let t = t_q;
    let dtype = q.dtype()?;

    // Repeat K, V along the head axis when GQA. `mlx::core::repeat` is
    // repeat_interleave-style: each element is repeated `group_size` times in
    // place, so head h_q == h_kv*group_size + r maps back to KV head h_kv.
    let (k_expanded, v_expanded) = if group_size == 1 {
        (k.clone(), v.clone())
    } else {
        (k.repeat(group_size, 1)?, v.repeat(group_size, 1)?)
    };

    // logits = (Q @ K^T) * scale, computed as Q @ K.transpose(-1, -2).
    // K^T over last two dims: swap dims 2 and 3.
    let k_t = k_expanded.transpose(Some(&[0, 1, 3, 2]))?;
    let scale = 1.0 / (d as f64).sqrt();
    let logits = q.matmul(&k_t)?.mul_scalar(scale)?;

    // Build band mask [T, T] in f32 and reshape to [1, 1, T, T] for broadcast.
    // Cast to logits dtype to avoid the f32-promotes-bf16 footgun.
    let band_mask = build_band_mask(t, band)?
        .reshape(&[1, 1, t, t])?
        .astype(dtype)?;
    let masked_logits = logits.add(&band_mask)?;

    // sinks: [H_q] → [1, H_q, 1, 1] → broadcast to [B, H_q, T, 1], cast to logits dtype.
    let sinks_col = sinks
        .reshape(&[1, h_q, 1, 1])?
        .astype(dtype)?
        .broadcast_to(&[b, h_q, t, 1])?;

    // Concatenate sink as the last "virtual" key column → [B, H_q, T, T+1].
    let with_sinks = MxArray::concatenate(&masked_logits, &sinks_col, -1)?;

    // Softmax along the augmented key axis. Sink absorbs some mass; real-key
    // weights now sum to (1 - sink_weight) instead of 1.
    let attn_with_sinks = softmax_last_axis(&with_sinks)?;

    // Slice off the sink column to recover [B, H_q, T, T] attention weights.
    let attn_kv = attn_with_sinks.slice(&[0, 0, 0, 0], &[b, h_q, t, t])?;

    // out = attn @ V  → [B, H_q, T, D]
    attn_kv.matmul(&v_expanded)
}

/// Build the `[H_q, T, T+1]` augmented mask used by `banded_attention`.
///
/// Layout:
/// - `mask[h, q, k] = 0`         if `k < T && |q - k| <= band`
/// - `mask[h, q, k] = -inf`      if `k < T && |q - k| >  band`
/// - `mask[h, q, T] = sinks[h]`  for the appended virtual sink column
///
/// `out_dtype` is the dtype the mask is cast to — must match the SDPA output
/// dtype (`final_type` in MLX), otherwise SDPA rejects it because f32 does not
/// promote to bf16/f16.
fn build_band_plus_sink_mask(
    t: i64,
    band: i32,
    sinks: &MxArray,
    h_q: i64,
    out_dtype: DType,
) -> Result<MxArray> {
    if t <= 0 {
        return Err(Error::from_reason(format!(
            "build_band_plus_sink_mask: T must be positive, got {t}"
        )));
    }

    // Band mask: [T, T] f32, 0 inside the band, -inf outside.
    let band_mask = build_band_mask(t, band)?;

    // Broadcast to [H_q, T, T]. The band mask doesn't depend on the head index,
    // but the final mask does (sink column varies per head), so we materialise
    // a per-head copy via broadcast_to.
    let band_mask_per_head = band_mask.reshape(&[1, t, t])?.broadcast_to(&[h_q, t, t])?;

    // Sink column: sinks[h] → [H_q, T, 1] (broadcast over T).
    let sinks_col = sinks
        .reshape(&[h_q, 1, 1])?
        .astype(DType::Float32)?
        .broadcast_to(&[h_q, t, 1])?;

    // Concatenate along the last axis → [H_q, T, T+1] in f32.
    let mask_f32 = MxArray::concatenate(&band_mask_per_head, &sinks_col, -1)?;

    // Cast to the SDPA output dtype. MLX requires that the mask dtype promotes
    // to the output dtype; bf16/f16 don't accept an f32 mask.
    mask_f32.astype(out_dtype)
}

/// Bidirectional banded attention with per-head sinks — fast path.
///
/// Implemented by augmenting K and V with a zero "sink slot" and encoding the
/// sink logits in the mask, then calling MLX's fused
/// `scaled_dot_product_attention` (Apple's optimised Metal SDPA kernel).
///
/// Correctness sketch:
/// - Append a zero column to K → K_aug `[B, H_kv, T+1, D]`, same for V.
/// - Build a `[H_q, T, T+1]` mask where the last column carries `sinks[h]` and
///   the first T columns carry the band mask (0 in-band, -inf out-of-band).
/// - The fused kernel computes `softmax((Q @ K_aug^T) * scale + mask) @ V_aug`.
///   Because `V_aug[:, :, T, :] = 0`, the sink slot contributes nothing to the
///   numerator while still showing up in the softmax denominator — exactly the
///   sink semantics encoded in `banded_attention_reference`.
///
/// Equivalent (up to bf16/f16 rounding) to `banded_attention_reference`, but
/// substantially faster because softmax + (Q@K^T) + (attn@V) fuse into one
/// Metal shader invocation.
///
/// Shapes (same as `banded_attention_reference`):
/// - `q`:  `[B, num_q_heads, T, head_dim]`
/// - `k`:  `[B, num_kv_heads, T, head_dim]`
/// - `v`:  `[B, num_kv_heads, T, head_dim]`
/// - `sinks`: `[num_q_heads]`
/// - returns `[B, num_q_heads, T, head_dim]`
pub fn banded_attention(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    sinks: &MxArray,
    band: i32,
) -> Result<MxArray> {
    if band < 0 {
        return Err(Error::from_reason(format!(
            "banded_attention: band must be non-negative, got {band}"
        )));
    }

    if q.ndim()? != 4 || k.ndim()? != 4 || v.ndim()? != 4 {
        return Err(Error::from_reason(
            "banded_attention: Q, K, V must be 4D [B, H, T, D]",
        ));
    }
    if sinks.ndim()? != 1 {
        return Err(Error::from_reason(
            "banded_attention: sinks must be 1D [num_q_heads]",
        ));
    }

    let q_shape = q.shape()?;
    let k_shape = k.shape()?;
    let v_shape = v.shape()?;
    let sinks_shape = sinks.shape()?;
    let q_shape = q_shape.as_ref();
    let k_shape = k_shape.as_ref();
    let v_shape = v_shape.as_ref();
    let sinks_shape = sinks_shape.as_ref();

    let b = q_shape[0];
    let h_q = q_shape[1];
    let t_q = q_shape[2];
    let d = q_shape[3];

    let h_kv = k_shape[1];
    let t_kv = k_shape[2];

    if k_shape[0] != b || v_shape[0] != b {
        return Err(Error::from_reason(
            "banded_attention: batch dim must match between Q, K, V",
        ));
    }
    if k_shape[1] != h_kv || v_shape[1] != h_kv {
        return Err(Error::from_reason(
            "banded_attention: num_kv_heads must match between K and V",
        ));
    }
    if k_shape[2] != t_kv || v_shape[2] != t_kv {
        return Err(Error::from_reason(
            "banded_attention: sequence length must match between K and V",
        ));
    }
    if t_q != t_kv {
        return Err(Error::from_reason(format!(
            "banded_attention: Q/K sequence lengths must match (self-attn), got T_q={t_q}, T_kv={t_kv}"
        )));
    }
    if k_shape[3] != d || v_shape[3] != d {
        return Err(Error::from_reason(
            "banded_attention: head_dim must match between Q, K, V",
        ));
    }
    if h_q % h_kv != 0 {
        return Err(Error::from_reason(format!(
            "banded_attention: num_q_heads ({h_q}) must be a multiple of num_kv_heads ({h_kv})"
        )));
    }
    if sinks_shape[0] != h_q {
        return Err(Error::from_reason(format!(
            "banded_attention: sinks length ({}) must equal num_q_heads ({h_q})",
            sinks_shape[0]
        )));
    }

    let t = t_q;
    let dtype = q.dtype()?;

    // K_aug = concat([K, zeros[B, H_kv, 1, D]], axis=-2)  → [B, H_kv, T+1, D]
    // V_aug = concat([V, zeros[B, H_kv, 1, D]], axis=-2)  → [B, H_kv, T+1, D]
    //
    // V_aug's sink slot is zero so even though softmax weight may be non-zero
    // for the sink column, the resulting V contribution is zero — only the
    // denominator share matters.
    let pad_shape = [b, h_kv, 1, d];
    let pad_k = MxArray::zeros(&pad_shape, Some(dtype))?;
    let pad_v = MxArray::zeros(&pad_shape, Some(dtype))?;
    let k_aug = MxArray::concatenate(k, &pad_k, -2)?;
    let v_aug = MxArray::concatenate(v, &pad_v, -2)?;

    // Build [H_q, T, T+1] mask, cast to the SDPA output dtype (= q's dtype).
    let mask = build_band_plus_sink_mask(t, band, sinks, h_q, dtype)?;

    let scale = 1.0 / (d as f64).sqrt();
    scaled_dot_product_attention(q, &k_aug, &v_aug, scale, Some(&mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_data(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = i as f32 + phase;
                (x.sin() * 0.5) + (x.cos() * 0.25)
            })
            .collect()
    }

    fn assert_arrays_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= tolerance,
                "{label}: divergence at {i}: actual={a}, expected={e}, diff={diff} > tol={tolerance}"
            );
        }
    }

    /// With `band >= T` every key is in-window, and with sinks set very
    /// negative the virtual sink column adds ~0 to the softmax denominator —
    /// the function should equal plain (unmasked) scaled dot-product attention.
    #[test]
    fn banded_reference_with_huge_band_matches_full_attention() {
        let batch = 1i64;
        let heads = 2i64;
        let t = 6i64;
        let head_dim = 8i64;

        let q = MxArray::from_float32(
            &deterministic_data((batch * heads * t * head_dim) as usize, 0.0),
            &[batch, heads, t, head_dim],
        )
        .unwrap();
        let k = MxArray::from_float32(
            &deterministic_data((batch * heads * t * head_dim) as usize, 1.0),
            &[batch, heads, t, head_dim],
        )
        .unwrap();
        let v = MxArray::from_float32(
            &deterministic_data((batch * heads * t * head_dim) as usize, 2.0),
            &[batch, heads, t, head_dim],
        )
        .unwrap();
        // Sinks essentially -inf so exp(sinks) ≈ 0 and softmax denominator is
        // unaffected.
        let sinks = MxArray::from_float32(&[-1e9, -1e9], &[heads]).unwrap();

        // band = T-1 covers the entire bidirectional window for every q.
        let banded = banded_attention_reference(&q, &k, &v, &sinks, (t - 1) as i32).unwrap();
        let banded_data = banded.to_float32().unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let full = crate::array::scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
        let full_data = full.to_float32().unwrap();

        assert_arrays_close(
            banded_data.as_ref(),
            full_data.as_ref(),
            1e-4,
            "banded(band>=T, sinks=-inf) vs SDPA",
        );
    }

    /// Construct K so the last token has a much larger key magnitude than all
    /// others. With a tight band (1) the query at position 0 cannot attend to
    /// k=T-1, so output[0] must not be dominated by V[T-1] — concretely, it
    /// should be much closer to V[0..=band] than to V[T-1].
    #[test]
    fn banded_reference_zeroes_out_of_window_keys() {
        let batch = 1i64;
        let heads = 1i64;
        let t = 8i64;
        let head_dim = 4i64;
        let band = 1;

        // Q: all-ones (so logits = sum(K[t,:]) * scale per token)
        let q = MxArray::from_float32(
            &vec![1.0f32; (batch * heads * t * head_dim) as usize],
            &[batch, heads, t, head_dim],
        )
        .unwrap();

        // K: zero everywhere EXCEPT the last token which has very large values.
        // Without the band mask, the softmax for q=0 would collapse to k=T-1.
        let mut k_data = vec![0.0f32; (batch * heads * t * head_dim) as usize];
        let last_row_start = ((t - 1) * head_dim) as usize;
        for j in 0..head_dim as usize {
            k_data[last_row_start + j] = 50.0;
        }
        let k = MxArray::from_float32(&k_data, &[batch, heads, t, head_dim]).unwrap();

        // V: distinct per-token unit vectors. V[t, :] = [t+1, t+1, ...].
        let mut v_data = vec![0.0f32; (batch * heads * t * head_dim) as usize];
        for tok in 0..t as usize {
            for j in 0..head_dim as usize {
                v_data[tok * head_dim as usize + j] = (tok as f32) + 1.0;
            }
        }
        let v = MxArray::from_float32(&v_data, &[batch, heads, t, head_dim]).unwrap();

        // Sinks very negative so they don't shift mass.
        let sinks = MxArray::from_float32(&[-1e9], &[heads]).unwrap();

        let out = banded_attention_reference(&q, &k, &v, &sinks, band).unwrap();
        let out_data = out.to_float32().unwrap();

        // For q=0, only k=0,1 are in-band. K[0,:] = K[1,:] = 0 (all-zero rows),
        // so logits are equal → softmax weights are uniform 0.5/0.5 over k=0,1.
        // V[0,:] = 1, V[1,:] = 2 → output[0,:] should be 1.5.
        // The huge V[T-1, :] = T must NOT contribute (it's out of band).
        let v_last = t as f32; // V[T-1, j] = T
        for j in 0..head_dim as usize {
            let val = out_data[j];
            assert!(
                (val - 1.5).abs() < 1e-3,
                "q=0 output dim {j} = {val}; expected ≈ 1.5 (avg of V[0], V[1]); \
                 if the band mask is broken we'd see ≈ {v_last} from out-of-band V[T-1]"
            );
            assert!(
                (val - v_last).abs() > 1.0,
                "q=0 output dim {j} = {val} is suspiciously close to V[T-1] = {v_last}; \
                 the band mask may be leaking out-of-window keys"
            );
        }
    }

    /// With sinks set to a large positive value, most softmax mass routes to
    /// the sink (which has no V contribution), so the output magnitude should
    /// shrink dramatically compared to sinks = -inf.
    #[test]
    fn banded_reference_sinks_reduce_attention_mass() {
        let batch = 1i64;
        let heads = 1i64;
        let t = 4i64;
        let head_dim = 4i64;

        let q = MxArray::from_float32(
            &deterministic_data((batch * heads * t * head_dim) as usize, 0.0),
            &[batch, heads, t, head_dim],
        )
        .unwrap();
        let k = MxArray::from_float32(
            &deterministic_data((batch * heads * t * head_dim) as usize, 1.0),
            &[batch, heads, t, head_dim],
        )
        .unwrap();
        // V values are all roughly O(1); pick a constant non-zero vector so
        // the output magnitude is essentially the sum of softmax weights times
        // a fixed bias.
        let v = MxArray::from_float32(
            &vec![3.0f32; (batch * heads * t * head_dim) as usize],
            &[batch, heads, t, head_dim],
        )
        .unwrap();

        let sinks_neg = MxArray::from_float32(&[-1e9], &[heads]).unwrap();
        let sinks_pos = MxArray::from_float32(&[10.0], &[heads]).unwrap();

        let out_no_sink =
            banded_attention_reference(&q, &k, &v, &sinks_neg, (t - 1) as i32).unwrap();
        let out_big_sink =
            banded_attention_reference(&q, &k, &v, &sinks_pos, (t - 1) as i32).unwrap();

        let no_sink = out_no_sink.to_float32().unwrap();
        let big_sink = out_big_sink.to_float32().unwrap();

        // With sinks=-inf: real-token softmax weights sum to 1, output ≈ 3.0.
        // With sinks=10: logits Q@K^T*scale are ~O(1), so sink dominates and
        // real weights sum to ≈ 1/(1 + exp(10 - max_logit)) ≪ 1. Output should
        // therefore be much smaller than 3.0.
        let mut max_abs_no = 0.0f32;
        let mut max_abs_big = 0.0f32;
        for (a, b) in no_sink.iter().zip(big_sink.iter()) {
            max_abs_no = max_abs_no.max(a.abs());
            max_abs_big = max_abs_big.max(b.abs());
        }
        // Strong sanity check: big-sink output is at least 10x smaller in
        // magnitude. (Tight bound is ~exp(-10) ≈ 4.5e-5x but we leave room.)
        assert!(
            max_abs_no > 1.0,
            "sinks=-inf baseline has unexpectedly tiny output (max_abs={max_abs_no}); \
             expected ≈ 3.0 since all real weights sum to 1 and V is constant 3.0"
        );
        assert!(
            max_abs_big * 10.0 < max_abs_no,
            "sinks=+10 should heavily reduce attention mass; \
             max_abs no_sink={max_abs_no}, max_abs big_sink={max_abs_big}"
        );
    }

    /// Max abs diff between two arrays returned as f32 buffers. Both inputs
    /// are converted to f32 before comparison so we can compare bf16 outputs
    /// without bit-extraction shenanigans. Also returns the max abs value of
    /// either array — useful for setting magnitude-aware tolerances.
    fn max_abs_diff_and_magnitude(a: &MxArray, b: &MxArray) -> (f32, f32) {
        let av = a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let bv = b.astype(DType::Float32).unwrap().to_float32().unwrap();
        assert_eq!(
            av.len(),
            bv.len(),
            "length mismatch {} vs {}",
            av.len(),
            bv.len()
        );
        let mut max_diff = 0.0f32;
        let mut max_abs = 0.0f32;
        for (x, y) in av.iter().zip(bv.iter()) {
            let d = (x - y).abs();
            if d > max_diff {
                max_diff = d;
            }
            let m = x.abs().max(y.abs());
            if m > max_abs {
                max_abs = m;
            }
        }
        (max_diff, max_abs)
    }

    /// Fast SDPA-based `banded_attention` should agree with the reference impl
    /// across the shapes used by the privacy-filter design (T=257, band=128)
    /// plus a couple of neighbours that stress GQA + multi-batch behaviour.
    #[test]
    fn banded_attention_matches_reference_on_random_inputs() {
        // Seed for reproducibility — bf16 parity is intrinsically tight to
        // the magnitude-scaled tolerance, so we don't want a flaky test on
        // unlucky random draws.
        unsafe { sys::mlx_seed(0x6261_6e64) };

        // (B, H_q, H_kv, head_dim, T, band)
        let cases: &[(i64, i64, i64, i64, i64, i32)] = &[
            (1, 14, 2, 64, 64, 32),    // small, GQA group_size=7
            (1, 14, 2, 64, 257, 128),  // exact privacy-filter window
            (1, 14, 2, 64, 1024, 128), // longer sequence
            (2, 14, 2, 64, 512, 128),  // multi-batch
        ];

        for &(b, h_q, h_kv, d, t, band) in cases {
            let q =
                MxArray::random_normal(&[b, h_q, t, d], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
            let k =
                MxArray::random_normal(&[b, h_kv, t, d], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
            let v =
                MxArray::random_normal(&[b, h_kv, t, d], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
            // Sinks are conventionally f32 (and small in magnitude so we
            // actually exercise softmax mixing, not the degenerate -inf case).
            let sinks = MxArray::random_normal(&[h_q], 0.0, 1.0, Some(DType::Float32)).unwrap();

            let fast = banded_attention(&q, &k, &v, &sinks, band).unwrap();
            let slow = banded_attention_reference(&q, &k, &v, &sinks, band).unwrap();

            let fast_shape = fast.shape().unwrap();
            let fast_shape = fast_shape.as_ref();
            assert_eq!(
                fast_shape,
                &[b, h_q, t, d],
                "banded_attention shape mismatch for case (b={b}, h_q={h_q}, h_kv={h_kv}, d={d}, t={t}, band={band})"
            );

            // Both impls produce bf16 outputs but follow different
            // accumulation paths: the reference materialises intermediate bf16
            // attention weights before the second matmul, whereas SDPA keeps
            // running max/sum in f32 inside the fused kernel. Per-element
            // disagreement is therefore bounded by a small number of bf16 ULPs
            // scaled by the output magnitude. (bf16 ULP at magnitude M is
            // M * 2^-7 ≈ 8e-3·M; we allow ~3 ULPs to cover the sqrt(T)-scaled
            // accumulator term, with a 5e-3 absolute floor for the small-T
            // cases where outputs land near zero. The f32 parity test
            // `banded_attention_matches_reference_f32` enforces a far tighter
            // 5e-5 bound, so algorithmic bugs cannot hide under this band.)
            let (diff, max_abs) = max_abs_diff_and_magnitude(&fast, &slow);
            let tol = 5e-3f32.max(max_abs * (3.0 / 128.0));
            assert!(
                diff <= tol,
                "banded_attention vs reference diverged for case \
                 (b={b}, h_q={h_q}, h_kv={h_kv}, d={d}, t={t}, band={band}): \
                 max_abs_diff={diff} > tol={tol} (max_abs={max_abs})"
            );
        }
    }

    /// Same parity sweep as the bf16 test but in f32, where both paths use
    /// identical precision throughout. This catches algorithmic bugs that
    /// would be hidden under the bf16 magnitude-scaled tolerance.
    #[test]
    fn banded_attention_matches_reference_f32() {
        unsafe { sys::mlx_seed(0x6632_6e64) };

        let cases: &[(i64, i64, i64, i64, i64, i32)] = &[
            (1, 14, 2, 64, 64, 32),
            (1, 14, 2, 64, 257, 128),
            (2, 4, 2, 32, 128, 16),
        ];

        for &(b, h_q, h_kv, d, t, band) in cases {
            let q =
                MxArray::random_normal(&[b, h_q, t, d], 0.0, 1.0, Some(DType::Float32)).unwrap();
            let k =
                MxArray::random_normal(&[b, h_kv, t, d], 0.0, 1.0, Some(DType::Float32)).unwrap();
            let v =
                MxArray::random_normal(&[b, h_kv, t, d], 0.0, 1.0, Some(DType::Float32)).unwrap();
            let sinks = MxArray::random_normal(&[h_q], 0.0, 1.0, Some(DType::Float32)).unwrap();

            let fast = banded_attention(&q, &k, &v, &sinks, band).unwrap();
            let slow = banded_attention_reference(&q, &k, &v, &sinks, band).unwrap();

            let (diff, max_abs) = max_abs_diff_and_magnitude(&fast, &slow);
            // f32 path: agree to ~1e-5 in absolute terms even after
            // accumulation across T~257 keys.
            assert!(
                diff <= 5e-5,
                "banded_attention f32 vs reference diverged for case \
                 (b={b}, h_q={h_q}, h_kv={h_kv}, d={d}, t={t}, band={band}): \
                 max_abs_diff={diff} > 5e-5 (max_abs={max_abs})"
            );
        }
    }

    /// Spot-check: the fast path should pass the same "huge-band, sinks≈-inf"
    /// invariant the reference satisfies — output equals plain SDPA. This is a
    /// useful smoke test independent of the parity oracle.
    #[test]
    fn banded_attention_fast_with_huge_band_matches_full_attention() {
        let batch = 1i64;
        let heads = 2i64;
        let t = 6i64;
        let head_dim = 8i64;

        let q = MxArray::from_float32(
            &deterministic_data((batch * heads * t * head_dim) as usize, 0.0),
            &[batch, heads, t, head_dim],
        )
        .unwrap();
        let k = MxArray::from_float32(
            &deterministic_data((batch * heads * t * head_dim) as usize, 1.0),
            &[batch, heads, t, head_dim],
        )
        .unwrap();
        let v = MxArray::from_float32(
            &deterministic_data((batch * heads * t * head_dim) as usize, 2.0),
            &[batch, heads, t, head_dim],
        )
        .unwrap();
        let sinks = MxArray::from_float32(&[-1e9, -1e9], &[heads]).unwrap();

        let banded = banded_attention(&q, &k, &v, &sinks, (t - 1) as i32).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let full = crate::array::scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();

        assert_arrays_close(
            banded.to_float32().unwrap().as_ref(),
            full.to_float32().unwrap().as_ref(),
            1e-4,
            "banded_attention(band>=T, sinks=-inf) vs SDPA",
        );
    }
}
