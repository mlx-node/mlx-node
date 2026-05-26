//! Inspector recorder for the education-app "Try it now" panels.
//!
//! The frontend lessons (see `packages/browser/src/inspector-types.ts` for the
//! canonical TS contract) need to peek inside the model while it runs — most
//! prominently the post-softmax attention scores per layer. This module owns
//! the Rust side of that hook.
//!
//! # Hot-path safety
//!
//! [`InspectorRecorder`] is *only* constructed when an inspector-enabled
//! generation path runs (currently `Qwen3_5Model::run_for_inspector`). Every
//! forward / attention site that accepts a recorder takes
//! `Option<&mut InspectorRecorder>` and defaults to `None`, so the existing
//! `generate` / chat-stream paths skip all inspector code unconditionally.
//!
//! # Memory cost
//!
//! Attention scores are captured as `Vec<f32>` per full-attention layer with
//! shape `[num_heads, seq_q, seq_k]`. For Qwen3.5-0.8B (16 query heads, full
//! attention every 4 layers, 1 prefill pass) a 1k-token context costs
//! ~64 MiB per captured layer (`16 * 1024 * 1024 * 4 bytes`). Multiple layers
//! return all data in a single `AttentionRunNapi` struct — no streaming.
//! Frontends typically restrict captures to one or two layers; the
//! `attention_layers` request field is the throttle.

use std::collections::BTreeSet;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::MxArray;
use crate::array::mask::create_causal_mask;
use crate::nn::Activations;

/// Per-full-attention-layer attention score capture.
///
/// `scores` is the post-softmax attention matrix in row-major
/// `[num_heads, seq_q, seq_k]` layout, matching the contract documented in
/// `packages/browser/src/inspector-types.ts`. Layers that ran but were not
/// requested (or were filtered out) are *not* present in the recorder — the
/// final serialization step fills in `linear` layer entries from the model
/// config.
#[derive(Debug, Clone)]
pub struct CapturedAttention {
    pub layer_index: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    /// Query-side sequence length used in the capture. Retained for
    /// downstream assertions even when not directly read from the
    /// serialization path.
    #[allow(dead_code)]
    pub seq_q: i32,
    /// Key-side sequence length used in the capture. Same rationale as
    /// [`Self::seq_q`].
    #[allow(dead_code)]
    pub seq_k: i32,
    pub scores: Vec<f32>,
}

/// State carried through a single inspector-enabled forward run.
///
/// The recorder lives on the stack of `run_for_inspector_sync` and is threaded
/// down into the model forward through an `Option<&mut InspectorRecorder>`
/// parameter that is `None` for every other code path.
#[derive(Debug)]
pub struct InspectorRecorder {
    /// Whether to capture attention scores at all. When `false`, the recorder
    /// is effectively a no-op (the attention-side `should_capture_layer` short-
    /// circuits and avoids any extra GPU work).
    pub attention_enabled: bool,
    /// If `Some`, only capture for layer indices in this set. When `None`,
    /// capture every full-attention layer.
    pub attention_layers: Option<BTreeSet<i32>>,
    /// Captured attention payloads, ordered by the layer's position in the
    /// recorder's encounter sequence (matches the forward-pass layer iteration
    /// order).
    pub attention: Vec<CapturedAttention>,
}

impl InspectorRecorder {
    pub fn new(attention_enabled: bool, attention_layers: Option<Vec<i32>>) -> Self {
        let attention_layers = attention_layers.map(|ids| ids.into_iter().collect());
        Self {
            attention_enabled,
            attention_layers,
            attention: Vec::new(),
        }
    }

    /// Whether attention scores should be captured for `layer_index`.
    ///
    /// `false` skips the entire side-branch computation — important because
    /// the capture path runs a parallel non-fused `softmax(scale * Q@K^T)` and
    /// is the dominant cost when the inspector is on.
    pub fn should_capture_layer(&self, layer_index: i32) -> bool {
        if !self.attention_enabled {
            return false;
        }
        match &self.attention_layers {
            None => true,
            Some(set) => set.contains(&layer_index),
        }
    }

    /// Capture a full-attention layer's softmaxed attention scores.
    ///
    /// `queries` and `keys` are expected post-RoPE, post-QK-norm, in
    /// `[B=1, H, T, D]` layout. `num_heads` is the query-head count;
    /// `num_kv_heads` is the K/V-head count (the GQA ratio). The function
    /// recomputes `softmax(scale * Q @ K^T + causal_mask)` explicitly — i.e.,
    /// *not* the fused SDPA kernel — because the fused kernel never
    /// materializes the intermediate score tensor.
    ///
    /// The materialized scores are forced to f32 and copied to CPU
    /// synchronously here; the surrounding generation must `eval` lazy state
    /// before this call returns or `to_float32` will block the device.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_attention(
        &mut self,
        layer_index: i32,
        queries_bhtd: &MxArray,
        keys_bhtd: &MxArray,
        scale: f32,
        num_heads: i32,
        num_kv_heads: i32,
        kv_offset: i32,
    ) -> Result<()> {
        // Shapes — queries: [B, H_q, T_q, D], keys: [B, H_kv, T_k, D].
        let q_ndim = queries_bhtd.ndim()?;
        let k_ndim = keys_bhtd.ndim()?;
        if q_ndim != 4 || k_ndim != 4 {
            return Err(Error::from_reason(format!(
                "inspector capture expects 4-D Q/K, got Q.ndim={} K.ndim={}",
                q_ndim, k_ndim
            )));
        }
        let batch = queries_bhtd.shape_at(0)?;
        if batch != 1 {
            return Err(Error::from_reason(format!(
                "inspector capture only supports batch=1, got {}",
                batch
            )));
        }
        let seq_q = queries_bhtd.shape_at(2)?;
        let seq_k = keys_bhtd.shape_at(2)?;
        let head_dim_q = queries_bhtd.shape_at(3)?;
        let head_dim_k = keys_bhtd.shape_at(3)?;
        if head_dim_q != head_dim_k {
            return Err(Error::from_reason(format!(
                "inspector capture head-dim mismatch: q.D={} k.D={}",
                head_dim_q, head_dim_k
            )));
        }

        // Promote to f32 for the side-branch math so the softmax we emit is
        // identical bit-for-bit on every host. The fused SDPA kernel preserves
        // the input dtype; ours doesn't have to.
        use crate::array::DType;
        let q = queries_bhtd.astype(DType::Float32)?;
        let mut k = keys_bhtd.astype(DType::Float32)?;

        // GQA expansion: tile KV heads up to the query-head count when they
        // disagree. Without this the QK^T matmul broadcast would fail.
        if num_heads != num_kv_heads {
            if num_kv_heads <= 0 || num_heads % num_kv_heads != 0 {
                return Err(Error::from_reason(format!(
                    "GQA mismatch in inspector capture: num_heads={} num_kv_heads={}",
                    num_heads, num_kv_heads
                )));
            }
            let group = num_heads / num_kv_heads;
            // Insert axis after H_kv, repeat, then collapse: [B, H_kv, T, D]
            //   -> [B, H_kv, 1, T, D] -> [B, H_kv, group, T, D] -> [B, H_q, T, D]
            let head_dim = head_dim_k;
            let k_expanded = k.expand_dims(2)?;
            let k_repeated = k_expanded.broadcast_to(&[
                batch,
                num_kv_heads as i64,
                group as i64,
                seq_k,
                head_dim,
            ])?;
            k = k_repeated.reshape(&[batch, num_heads as i64, seq_k, head_dim])?;
        }

        // QK^T: [B, H, T_q, D] @ [B, H, D, T_k] -> [B, H, T_q, T_k]
        let keys_t = k.transpose(Some(&[0, 1, 3, 2]))?;
        let scores = q.matmul(&keys_t)?;
        let scores = scores.mul_scalar(scale as f64)?;

        // Causal mask: emulate the fused kernel's "causal" mode by adding
        // -inf to masked entries before softmax. For decode (seq_q == 1) we
        // skip the mask entirely — the single query token already attends to
        // all of the cached K/V, with no future positions to mask.
        let scores = if seq_q > 1 {
            // `create_causal_mask` returns a bool [seq_q, seq_k] tensor where
            // TRUE = keep score. Translate to an additive mask: 0 where keep,
            // -1e9 where mask. (-inf would propagate NaN through any padding.)
            let bool_mask = create_causal_mask(
                seq_q as i32,
                Some(kv_offset),
                None,
            )?;
            let bool_f32 = bool_mask.astype(DType::Float32)?;
            // additive_mask = (bool_f32 - 1) * 1e9
            //   keep => 0, mask => -1e9
            let one = MxArray::ones(&[1], Some(DType::Float32))?;
            let neg_inf_mask = bool_f32.sub(&one)?.mul_scalar(1.0e9)?;
            // Broadcast [seq_q, seq_k] over [B, H, seq_q, seq_k] (leading 1s).
            let mask_4d = neg_inf_mask.reshape(&[1, 1, seq_q, seq_k])?;
            scores.add(&mask_4d)?
        } else {
            scores
        };

        let probs = Activations::softmax(&scores, Some(-1))?;
        probs.eval();
        let mut buffer = probs.to_float32_vec()?;

        // Squeeze the batch axis: layout becomes [H, T_q, T_k] row-major.
        let expected_len = (num_heads as usize) * (seq_q as usize) * (seq_k as usize);
        if buffer.len() != expected_len {
            return Err(Error::from_reason(format!(
                "inspector capture got {} floats, expected {} ([{}, {}, {}])",
                buffer.len(),
                expected_len,
                num_heads,
                seq_q,
                seq_k,
            )));
        }
        // Pre-shrink any speculative over-allocation so the downstream copy
        // into a NAPI Float32Array is exact.
        buffer.shrink_to_fit();

        self.attention.push(CapturedAttention {
            layer_index,
            num_heads,
            num_kv_heads,
            seq_q: seq_q as i32,
            seq_k: seq_k as i32,
            scores: buffer,
        });
        Ok(())
    }
}

// ============================================================================
// NAPI surface
// ============================================================================

/// Inspector-run options. Mirrors the TS `InspectorRequest` (minus the wire
/// fields `type` / `id` / `prompt`, which are part of the worker protocol
/// rather than the NAPI signature).
#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct InspectorRunOptions {
    /// Capture full-attention layer scores. Default: false.
    #[napi(ts_type = "boolean | undefined")]
    pub attention: Option<bool>,
    /// Restrict capture to specific full-attention layer indices.
    /// Default: all full-attention layers.
    #[napi(ts_type = "number[] | undefined")]
    pub attention_layers: Option<Vec<i32>>,
    /// How many new tokens to generate before returning. Default: 1.
    /// Hard-capped at 8 inside the native code — this is a visualization
    /// hook, not a chat path.
    #[napi(ts_type = "number | undefined")]
    pub max_new_tokens: Option<i32>,
}

/// Tokenized token entry. Matches `TokenInfo` in inspector-types.ts.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct TokenInfoNapi {
    pub id: i32,
    pub text: String,
}

/// Per-layer attention payload. Matches `AttentionLayer` in
/// inspector-types.ts. `kind == "linear"` layers carry an empty `scores`
/// Float32Array so the frontend can preserve layer ordering.
#[napi(object)]
pub struct AttentionLayerNapi {
    pub layer_index: i32,
    pub kind: String,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub scores: Float32Array,
}

/// Model metadata. Matches `ModelMeta` in inspector-types.ts.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ModelMetaNapi {
    pub name: String,
    pub num_layers: i32,
    pub full_attention_layer_indices: Vec<i32>,
}

/// Top-level inspector-run result. Matches `AttentionRun` in
/// inspector-types.ts.
#[napi(object)]
pub struct AttentionRunNapi {
    pub prompt: String,
    pub tokens: Vec<TokenInfoNapi>,
    pub generated_token: TokenInfoNapi,
    pub attention: Vec<AttentionLayerNapi>,
    pub model_meta: ModelMetaNapi,
}

/// Hard cap on `max_new_tokens` for an inspector run. The inspector hook is
/// a visualization device; capturing 1k token x 1k token attention matrices
/// for 64+ layers is a non-trivial memory footprint even at 4 bytes per cell.
/// Refer to the module-level docs for the per-layer cost.
pub const INSPECTOR_MAX_NEW_TOKENS_CAP: i32 = 8;
