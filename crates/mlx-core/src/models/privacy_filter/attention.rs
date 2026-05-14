//! AttentionLayer for the OpenAI Privacy Filter.
//!
//! Composes:
//! - Q/K/V projections with bias (GQA: 14 query heads, 2 KV heads, head_dim=64
//!   for the shipped 8-layer checkpoint).
//! - YaRN-scaled RoPE applied to Q and K (frequencies precomputed once at
//!   model load and shared across all 8 layers — see [`super::yarn`]).
//! - [`banded_attention`] with per-head attention sinks and a bidirectional
//!   band equal to `sliding_window`.
//! - Output projection with bias.
//!
//! The layer is intentionally a thin borrow-only wrapper around
//! [`AttnWeights`] and [`PrivacyFilterConfig`]; the model assembles it on the
//! fly during the forward pass. No mutable state lives here — KV caches are
//! not applicable to the bidirectional token-classification setting.

use crate::array::{MxArray, banded_attention};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::config::PrivacyFilterConfig;
use super::persistence::AttnWeights;
use super::quantized_linear::project_2d;

/// One privacy-filter attention block, parameterised by borrowed weights /
/// config / shared YaRN frequencies.
pub struct AttentionLayer<'a> {
    pub weights: &'a AttnWeights,
    pub config: &'a PrivacyFilterConfig,
    /// Shape `[head_dim / 2]`, `f32`. Built once at model load.
    pub yarn_freqs: &'a MxArray,
}

impl<'a> AttentionLayer<'a> {
    /// Compute one block's attention forward pass.
    ///
    /// Input shape:  `[B, T, hidden_size]`
    /// Output shape: `[B, T, hidden_size]`
    ///
    /// `band` is the bidirectional attention window in tokens —
    /// `|q_pos - k_pos| <= band`. For `sliding_attention` layers pass
    /// `config.sliding_window`; for `full_attention` layers pass an
    /// effectively-unbounded value (use
    /// [`PrivacyFilterConfig::band_for_layer`]). Per gpt-oss defaults,
    /// half the layers alternate to full attention — applying the sliding
    /// band to every layer cripples the bidirectional receptive field.
    pub fn forward(&self, hidden: &MxArray, band: i32) -> Result<MxArray> {
        let batch = hidden.shape_at(0)?;
        let seq_len = hidden.shape_at(1)?;

        let hidden_size = self.config.hidden_size as i64;
        let head_dim = self.config.head_dim as i64;
        let num_q_heads = self.config.num_attention_heads as i64;
        let num_kv_heads = self.config.num_key_value_heads as i64;
        let q_dim = num_q_heads * head_dim; // 14 * 64 = 896

        // 1. Q, K, V projections with bias.
        //    For plain bf16 checkpoints each weight is stored as
        //    `[out, in]`, so `project_2d` transposes to `[in, out]`
        //    before the fused matmul (mirroring `nn::Linear`). For
        //    quantized checkpoints `project_2d` dispatches to
        //    `mlx_quantized_matmul(transpose=true)` and adds the bias
        //    after dequantize; both branches return the same shape so
        //    the rest of this forward pass is unchanged.
        let q = project_2d(hidden, &self.weights.q_proj)?;
        let k = project_2d(hidden, &self.weights.k_proj)?;
        let v = project_2d(hidden, &self.weights.v_proj)?;

        // 2. Reshape `[B, T, *]` to `[B, T, H, D]` and permute to `[B, H, T, D]`.
        let q = q
            .reshape(&[batch, seq_len, num_q_heads, head_dim])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let k = k
            .reshape(&[batch, seq_len, num_kv_heads, head_dim])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let v = v
            .reshape(&[batch, seq_len, num_kv_heads, head_dim])?
            .transpose(Some(&[0, 2, 1, 3]))?;

        // 3. YaRN `attention_factor` (a.k.a. `mscale`) — multiply Q and K
        //    by the YaRN attention factor BEFORE the rotation. This
        //    mirrors mlx-lm `YarnRoPE.__call__`
        //    (`x[..., :dims] = self.mscale * x[..., :dims]` then
        //    `mx.fast.rope(...)`) and HF transformers
        //    (`cos *= attention_factor; sin *= attention_factor`, which
        //    is algebraically equivalent to pre-scaling Q/K). Without it,
        //    rotated Q/K are ~1.35× smaller than reference for
        //    `factor=32`. `mul_scalar` keeps the bf16 dtype intact —
        //    MLX handles the scalar promotion internally rather than
        //    materialising an f32 constant that would force bf16→f32.
        let attention_factor = self.config.rope_parameters.attention_factor();
        let (q, k) = if (attention_factor - 1.0).abs() > 1e-7 {
            let q = q.mul_scalar(attention_factor as f64)?;
            let k = k.mul_scalar(attention_factor as f64)?;
            (q, k)
        } else {
            (q, k)
        };

        // 4. YaRN RoPE on Q and K. Offset is always 0 (bidirectional
        //    self-attention over a single chunk — no KV cache). `dims` is the
        //    full head dimension; the freqs array is `[head_dim/2]`.
        //
        //    `mlx_fast_rope_with_freqs` accepts an array-valued offset for
        //    compile-friendliness; we materialise a 1-element `i32` array.
        let offset = MxArray::from_int32(&[0], &[1])?;
        let q = unsafe {
            sys::mlx_fast_rope_with_freqs(
                q.handle.0,
                head_dim as i32,
                true, // traditional=true: interleaved [::2]/[1::2] (HF _apply_rotary_emb)
                0.0,  // base ignored when freqs provided
                1.0,  // scale=1.0
                offset.handle.0,
                self.yarn_freqs.handle.0,
            )
        };
        let q = MxArray::from_handle(q, "privacy_filter yarn rope (q)")?;
        let k = unsafe {
            sys::mlx_fast_rope_with_freqs(
                k.handle.0,
                head_dim as i32,
                true,
                0.0,
                1.0,
                offset.handle.0,
                self.yarn_freqs.handle.0,
            )
        };
        let k = MxArray::from_handle(k, "privacy_filter yarn rope (k)")?;

        // 5. Banded attention with per-head sinks. Sliding window is
        //    bidirectional: |q - k| <= band. `band` is supplied by the
        //    caller because gpt-oss alternates sliding / full attention
        //    per layer — see [`PrivacyFilterConfig::band_for_layer`].
        if band < 0 {
            return Err(Error::from_reason(format!(
                "privacy_filter::AttentionLayer: band must be non-negative, got {band}",
            )));
        }
        let attn = banded_attention(&q, &k, &v, &self.weights.sinks, band)?;

        // 6. Merge heads `[B, H, T, D]` → `[B, T, H*D]` and apply the output
        //    projection (`o_proj.weight` shape `[hidden_size, num_q_heads *
        //    head_dim]`, so its transpose maps `q_dim → hidden_size`).
        let merged = attn
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&[batch, seq_len, q_dim])?;

        let out = project_2d(&merged, &self.weights.o_proj)?;

        // Output shape sanity: `[B, T, hidden_size]`. Use a debug_assert so
        // tests pin the contract without paying for a shape check on every
        // production forward.
        debug_assert_eq!(
            out.shape()?.as_ref(),
            &[batch, seq_len, hidden_size],
            "AttentionLayer::forward output shape mismatch"
        );
        let _ = hidden_size; // silence unused warning in release builds
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;
    use crate::models::privacy_filter::persistence::load_from_directory;
    use crate::models::privacy_filter::yarn::compute_yarn_freqs;

    fn checkpoint_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models/privacy-filter")
    }

    /// Layer-0 forward against the real checkpoint. Verifies:
    /// 1. Output shape is `[B, T, hidden_size]`.
    /// 2. Every element is finite (no NaN / Inf escaping RoPE, banded attn,
    ///    or the bias-projected matmuls).
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn forward_output_shape_and_finite() {
        let loaded = load_from_directory(&checkpoint_dir()).expect("load model");
        let freqs = compute_yarn_freqs(loaded.config.head_dim, &loaded.config.rope_parameters)
            .expect("yarn freqs");

        let layer = AttentionLayer {
            weights: &loaded.weights.layers[0].self_attn,
            config: &loaded.config,
            yarn_freqs: &freqs,
        };

        // Random hidden state in the weights' native dtype (bf16 for this
        // checkpoint) so the matmuls don't trip the f32-promotes-bf16
        // footgun.
        let hidden_dtype = match &loaded.weights.layers[0].self_attn.q_proj {
            crate::models::privacy_filter::LoadedProj::Plain { weight, .. } => {
                weight.dtype().unwrap()
            }
            crate::models::privacy_filter::LoadedProj::Quantized { weight, .. } => {
                weight.dtype().unwrap()
            }
        };
        let hidden = MxArray::random_normal(
            &[1, 8, loaded.config.hidden_size as i64],
            0.0,
            1.0,
            Some(hidden_dtype),
        )
        .expect("random hidden");

        let band = loaded.config.band_for_layer(0);
        let out = layer.forward(&hidden, band).expect("attention forward");

        let shape = out.shape().unwrap().to_vec();
        assert_eq!(shape, vec![1, 8, loaded.config.hidden_size as i64]);

        // Finiteness check via GPU reduction (`has_nan_or_inf` already exists
        // in `array::ops`; transfers a single bool).
        // Promote to f32 first — bf16's slightly larger dynamic range
        // shouldn't be relied on for the finiteness check semantics.
        let out_f32 = out.astype(DType::Float32).unwrap();
        assert!(
            !out_f32.has_nan_or_inf().unwrap(),
            "AttentionLayer::forward produced NaN or Inf"
        );
    }

    /// Determinism: same input + same weights ⇒ same output across two
    /// calls. Guards against accidental RNG / cache reuse in any of the
    /// underlying ops.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn forward_deterministic() {
        let loaded = load_from_directory(&checkpoint_dir()).expect("load model");
        let freqs = compute_yarn_freqs(loaded.config.head_dim, &loaded.config.rope_parameters)
            .expect("yarn freqs");

        let layer = AttentionLayer {
            weights: &loaded.weights.layers[0].self_attn,
            config: &loaded.config,
            yarn_freqs: &freqs,
        };

        let hidden_dtype = match &loaded.weights.layers[0].self_attn.q_proj {
            crate::models::privacy_filter::LoadedProj::Plain { weight, .. } => {
                weight.dtype().unwrap()
            }
            crate::models::privacy_filter::LoadedProj::Quantized { weight, .. } => {
                weight.dtype().unwrap()
            }
        };
        let hidden = MxArray::random_normal(
            &[1, 16, loaded.config.hidden_size as i64],
            0.0,
            1.0,
            Some(hidden_dtype),
        )
        .expect("random hidden");

        let band = loaded.config.band_for_layer(0);
        let a = layer.forward(&hidden, band).expect("forward 1");
        let b = layer.forward(&hidden, band).expect("forward 2");

        let av = a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let bv = b.astype(DType::Float32).unwrap().to_float32().unwrap();
        assert_eq!(av.len(), bv.len(), "output length mismatch");
        for (i, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "non-deterministic output at index {i}: {x} vs {y}"
            );
        }
    }
}
