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

use std::collections::{BTreeSet, HashSet};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::MxArray;
use crate::array::mask::create_causal_mask;
use crate::nn::Activations;

/// Canonical names of the seven hidden-state capture points per decoder layer.
/// Order matches the forward-pass order and is the default capture set when
/// no `points` filter is supplied.
pub const HIDDEN_STATE_POINTS: [&str; 7] = [
    "pre_attn_input",
    "post_attn_norm",
    "attn_output",
    "post_attn_residual",
    "post_mlp_norm",
    "mlp_output",
    "post_mlp_residual",
];

/// Soft cap on cumulative attention bytes captured by a single inspector run.
///
/// The browser deploy runs in a 4 GiB WASM heap shared with model weights,
/// KV cache, and lesson scratch buffers. A single Qwen3.5-0.8B full-attention
/// layer at 1k context costs `32 heads * 1024 * 1024 * 4 bytes = 128 MiB`
/// in fp32, so 256 MiB allows roughly two such layers — enough for the
/// lesson UI's typical "compare layer A vs layer B" view while preventing a
/// misconfigured request (capture every full layer at the context cap) from
/// silently blowing past the heap budget.
pub const INSPECTOR_TOTAL_BYTES_SOFT_CAP: usize = 256 * 1024 * 1024;

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

/// Per-generation-step top-K logits capture.
///
/// `top_k_ids` are token ids in descending logit order; `top_k_logits` are the
/// matching raw (pre-softmax) logits in the same order. `token_id` is the
/// token actually sampled at this step. Under greedy sampling — which the
/// inspector run uses — `token_id == top_k_ids[0]`. Text decoding happens
/// later in `run_for_inspector_sync` where the tokenizer is available.
#[derive(Debug, Clone)]
pub struct CapturedLogits {
    pub step: u32,
    pub token_id: u32,
    pub top_k_ids: Vec<u32>,
    pub top_k_logits: Vec<f32>,
}

/// Per-point hidden-state summary statistics. Stats are computed in f32 over
/// every element of the layer activation tensor (flattened across `seq` and
/// `hidden_dim`). `std` is the population standard deviation — divides by
/// `count`, not `count - 1`. The capture is unbiased for visualization use
/// and avoids a degenerate NaN when `count == 1`.
#[derive(Debug, Clone)]
pub struct CapturedHiddenStatePointStats {
    pub point: &'static str,
    pub mean: f32,
    pub std: f32,
    pub abs_max: f32,
    pub l2_norm: f32,
    pub count: u32,
}

/// One layer's worth of hidden-state captures for a single generation step.
#[derive(Debug, Clone)]
pub struct CapturedHiddenStateLayer {
    pub layer_index: u32,
    pub stats: Vec<CapturedHiddenStatePointStats>,
}

/// One generation step's worth of hidden-state captures, accumulating layers
/// as the forward pass progresses through them.
#[derive(Debug, Clone)]
pub struct CapturedHiddenStateStep {
    pub step: u32,
    pub layers: Vec<CapturedHiddenStateLayer>,
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
    /// Whether to capture per-step top-K logits. When `false`, all logits
    /// capture calls are no-ops.
    pub logits_enabled: bool,
    /// How many top-K entries to retain per captured step.
    pub logits_top_k: u32,
    /// Captured per-step logits payloads, in generation order.
    pub logits: Vec<CapturedLogits>,
    /// Whether to capture per-layer hidden-state summary statistics.
    pub hidden_states_enabled: bool,
    /// If `Some`, only capture hidden states for layer indices in this set.
    /// When `None`, capture every decoder layer.
    pub hidden_states_layers: Option<HashSet<u32>>,
    /// Which named points to capture per layer. Always populated — defaults
    /// to all seven names from [`HIDDEN_STATE_POINTS`] when the request omits
    /// a filter.
    pub hidden_states_points: HashSet<&'static str>,
    /// One entry per generation step that produced any captures. Layers are
    /// appended in forward-pass order; new steps are appended whenever
    /// [`Self::current_step`] advances past the last recorded step.
    pub hidden_states_steps: Vec<CapturedHiddenStateStep>,
    /// The generation step currently in flight. `run_for_inspector_sync`
    /// updates this before each forward pass (prefill = 0, then 1, 2, ...).
    /// The decoder layer reads it through `capture_hidden_state` so callers
    /// don't have to thread the value through every forward signature.
    pub current_step: u32,
    /// Cumulative bytes captured into [`Self::attention`]. Used to enforce
    /// [`INSPECTOR_TOTAL_BYTES_SOFT_CAP`] across all captured layers in a
    /// single inspector run.
    pub captured_bytes: usize,
}

impl InspectorRecorder {
    pub fn new(
        attention_enabled: bool,
        attention_layers: Option<Vec<i32>>,
        logits_enabled: bool,
        logits_top_k: u32,
        hidden_states_enabled: bool,
        hidden_states_layers: Option<Vec<i32>>,
        hidden_states_points: Option<Vec<String>>,
    ) -> Self {
        let attention_layers = attention_layers.map(|ids| ids.into_iter().collect());
        let hidden_states_layers = hidden_states_layers.map(|ids| {
            ids.into_iter()
                .filter_map(|v| if v < 0 { None } else { Some(v as u32) })
                .collect()
        });
        let hidden_states_points = match hidden_states_points {
            None => HIDDEN_STATE_POINTS.iter().copied().collect(),
            Some(requested) => {
                let requested_set: HashSet<String> = requested.into_iter().collect();
                HIDDEN_STATE_POINTS
                    .iter()
                    .copied()
                    .filter(|p| requested_set.contains(*p))
                    .collect()
            }
        };
        Self {
            attention_enabled,
            attention_layers,
            attention: Vec::new(),
            logits_enabled,
            logits_top_k,
            logits: Vec::new(),
            hidden_states_enabled,
            hidden_states_layers,
            hidden_states_points,
            hidden_states_steps: Vec::new(),
            current_step: 0,
            captured_bytes: 0,
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

        // Soft-cap guard. Compute the f32 byte cost of this capture before
        // we touch the GPU and reject early if adding it would exceed the
        // per-run budget. This protects the WASM heap from a misconfigured
        // "capture every full-attention layer at 1k context" request.
        let expected_bytes = (num_heads as usize)
            .saturating_mul(seq_q as usize)
            .saturating_mul(seq_k as usize)
            .saturating_mul(4);
        let captured_bytes_after = self.captured_bytes.saturating_add(expected_bytes);
        if captured_bytes_after > INSPECTOR_TOTAL_BYTES_SOFT_CAP {
            return Err(Error::from_reason(format!(
                "Inspector capture would exceed soft cap: requested {} MiB, total {} MiB, cap {} MiB. \
                 Capture fewer layers or reduce context length.",
                expected_bytes / 1024 / 1024,
                captured_bytes_after / 1024 / 1024,
                INSPECTOR_TOTAL_BYTES_SOFT_CAP / 1024 / 1024,
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
            // -1e9 where mask.
            // Additive mask: -1e9 on masked positions; softmax(-1e9) ≈ 0 in f32.
            // The fused SDPA kernel uses -INF internally, so reconstructed
            // scores match to f32 precision on unmasked cells and are exactly
            // zero on masked cells.
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
        let buffer = probs.to_float32_vec()?;

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

        self.attention.push(CapturedAttention {
            layer_index,
            num_heads,
            num_kv_heads,
            seq_q: seq_q as i32,
            seq_k: seq_k as i32,
            scores: buffer,
        });
        self.captured_bytes = captured_bytes_after;
        Ok(())
    }

    /// Capture top-K logits for one generation step.
    ///
    /// `logits` is the 1-D logits vector for the token position being sampled
    /// (shape `[vocab]`). For Qwen3.5 the vocab is ~151k, so we pull the full
    /// vector to host once per step and run an O(vocab) partial sort — vocab
    /// is too large for an on-device gather to be worth the kernel-launch
    /// overhead for `top_k` ≤ 64. `token_id` is the id actually sampled at
    /// this step (typically `top_k_ids[0]` under greedy sampling).
    pub fn capture_logits(
        &mut self,
        step: u32,
        token_id: u32,
        logits: &MxArray,
    ) -> Result<()> {
        if !self.logits_enabled || self.logits_top_k == 0 {
            return Ok(());
        }
        let ndim = logits.ndim()?;
        let vocab = match ndim {
            1 => logits.shape_at(0)? as usize,
            2 => {
                let batch = logits.shape_at(0)?;
                if batch != 1 {
                    return Err(Error::from_reason(format!(
                        "inspector capture_logits 2-D requires batch=1, got {}",
                        batch
                    )));
                }
                logits.shape_at(1)? as usize
            }
            _ => {
                return Err(Error::from_reason(format!(
                    "inspector capture_logits expects 1-D or [1, vocab] logits, got ndim={}",
                    ndim
                )));
            }
        };
        if vocab == 0 {
            return Err(Error::from_reason("inspector capture_logits: empty vocab"));
        }
        let top_k = (self.logits_top_k as usize).min(vocab);

        let expected_bytes = top_k.saturating_mul(4 + 4);
        let captured_bytes_after = self.captured_bytes.saturating_add(expected_bytes);
        if captured_bytes_after > INSPECTOR_TOTAL_BYTES_SOFT_CAP {
            return Err(Error::from_reason(format!(
                "Inspector capture would exceed soft cap: requested {} B, total {} MiB, cap {} MiB.",
                expected_bytes,
                captured_bytes_after / 1024 / 1024,
                INSPECTOR_TOTAL_BYTES_SOFT_CAP / 1024 / 1024,
            )));
        }

        use crate::array::DType;
        let logits_f32 = logits.astype(DType::Float32)?;
        logits_f32.eval();
        let buffer = logits_f32.to_float32_vec()?;
        if buffer.len() != vocab {
            return Err(Error::from_reason(format!(
                "inspector capture_logits got {} floats, expected vocab={}",
                buffer.len(),
                vocab
            )));
        }

        let (top_k_ids, top_k_logits) = top_k_from_row(&buffer, top_k);

        self.logits.push(CapturedLogits {
            step,
            token_id,
            top_k_ids,
            top_k_logits,
        });
        self.captured_bytes = captured_bytes_after;
        Ok(())
    }

    /// Whether the recorder is configured to capture this layer/point combo.
    /// Cheap check used by the decoder forward to skip the f32 promotion + CPU
    /// copy when filtered out.
    pub fn should_capture_hidden_state(&self, layer_idx: u32, point: &str) -> bool {
        if !self.hidden_states_enabled {
            return false;
        }
        if let Some(set) = &self.hidden_states_layers {
            if !set.contains(&layer_idx) {
                return false;
            }
        }
        self.hidden_states_points.contains(point)
    }

    /// Capture summary statistics for one named point in one layer at the
    /// recorder's `current_step`. `point` must be one of [`HIDDEN_STATE_POINTS`]
    /// — unknown names are rejected with an error so the wire contract stays
    /// in sync with the frontend's union type.
    ///
    /// Stats are computed in one f32 pass over the host buffer; the mean is
    /// promoted to f64 mid-loop to keep the variance numerically stable for
    /// large `count` (Qwen3.5-0.8B prefill at 1k tokens is ~1.5M floats).
    pub fn capture_hidden_state(
        &mut self,
        layer_idx: u32,
        point: &str,
        x: &MxArray,
    ) -> Result<()> {
        if !self.should_capture_hidden_state(layer_idx, point) {
            return Ok(());
        }
        // Resolve to the static string slice so the captured record can hold a
        // 'static reference rather than allocating a String per capture.
        let point_static: &'static str = match HIDDEN_STATE_POINTS.iter().find(|p| **p == point) {
            Some(p) => *p,
            None => {
                return Err(Error::from_reason(format!(
                    "inspector capture_hidden_state: unknown point {:?}",
                    point
                )));
            }
        };

        // Each captured record is ~32 bytes — vanishingly small next to
        // attention scores — but we still account for it to keep the soft-cap
        // invariant honest.
        let expected_bytes: usize = std::mem::size_of::<CapturedHiddenStatePointStats>();
        let captured_bytes_after = self.captured_bytes.saturating_add(expected_bytes);
        if captured_bytes_after > INSPECTOR_TOTAL_BYTES_SOFT_CAP {
            return Err(Error::from_reason(format!(
                "Inspector capture would exceed soft cap: requested {} B, total {} MiB, cap {} MiB.",
                expected_bytes,
                captured_bytes_after / 1024 / 1024,
                INSPECTOR_TOTAL_BYTES_SOFT_CAP / 1024 / 1024,
            )));
        }

        use crate::array::DType;
        let x_f32 = x.astype(DType::Float32)?;
        x_f32.eval();
        let buffer = x_f32.to_float32_vec()?;
        let count = buffer.len();
        if count == 0 {
            return Err(Error::from_reason(format!(
                "inspector capture_hidden_state: empty tensor at point {:?} layer {}",
                point_static, layer_idx
            )));
        }
        if count > u32::MAX as usize {
            return Err(Error::from_reason(format!(
                "inspector capture_hidden_state: element count {} exceeds u32 cap",
                count
            )));
        }

        // Single pass: sum, sum of squares, max(|x|). Variance is computed via
        // E[x^2] - E[x]^2 in f64 to avoid catastrophic cancellation on large
        // tensors where the elementwise mean dwarfs individual deviations.
        let mut sum: f64 = 0.0;
        let mut sumsq: f64 = 0.0;
        let mut abs_max: f32 = 0.0;
        for &v in &buffer {
            let vd = v as f64;
            sum += vd;
            sumsq += vd * vd;
            let a = v.abs();
            if v.is_nan() {
                abs_max = f32::NAN;
            } else if a > abs_max {
                abs_max = a;
            }
        }
        let n = count as f64;
        let mean = sum / n;
        let variance = (sumsq / n) - (mean * mean);
        let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };
        let l2_norm = sumsq.sqrt() as f32;

        let stats = CapturedHiddenStatePointStats {
            point: point_static,
            mean: mean as f32,
            std: std as f32,
            abs_max,
            l2_norm,
            count: count as u32,
        };

        // Append to the step record, creating one if `current_step` advanced
        // past whatever we last saw.
        let step = self.current_step;
        let step_entry = match self.hidden_states_steps.last_mut() {
            Some(last) if last.step == step => last,
            _ => {
                self.hidden_states_steps.push(CapturedHiddenStateStep {
                    step,
                    layers: Vec::new(),
                });
                self.hidden_states_steps.last_mut().unwrap()
            }
        };
        // Within a step the forward pass walks layers in order; keep the last
        // entry hot to coalesce repeated points on the same layer.
        let layer_entry = match step_entry.layers.last_mut() {
            Some(last) if last.layer_index == layer_idx => last,
            _ => {
                step_entry.layers.push(CapturedHiddenStateLayer {
                    layer_index: layer_idx,
                    stats: Vec::new(),
                });
                step_entry.layers.last_mut().unwrap()
            }
        };
        layer_entry.stats.push(stats);
        self.captured_bytes = captured_bytes_after;
        Ok(())
    }
}

// ============================================================================
// Pure helpers shared by the inspector capture path and the `scoreTokens`
// single-pass speculative-decode verify. Kept free of MxArray / model state
// so they can be unit-tested without loaded weights.
// ============================================================================

/// Select the top-`top_k` `(token_id, logit)` pairs from one logits row,
/// descending by logit.
///
/// Extracted verbatim from the original `capture_logits` body so the
/// inspector's per-step capture and the `scoreTokens` verify path share one
/// definition: O(vocab) partial select, then a full sort of the retained K.
/// NaN-vs-number comparisons fall back to `Ordering::Equal`, matching the
/// original behavior. `top_k` is clamped to `row.len()`; an empty row (or
/// `top_k == 0`) yields empty outputs rather than panicking.
pub fn top_k_from_row(row: &[f32], top_k: usize) -> (Vec<u32>, Vec<f32>) {
    let top_k = top_k.min(row.len());
    if top_k == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut indexed: Vec<(u32, f32)> = row
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect();
    indexed.select_nth_unstable_by(top_k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.truncate(top_k);
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut top_k_ids = Vec::with_capacity(top_k);
    let mut top_k_logits = Vec::with_capacity(top_k);
    for (id, logit) in indexed {
        top_k_ids.push(id);
        top_k_logits.push(logit);
    }
    (top_k_ids, top_k_logits)
}

/// Per-row top-K over a flat row-major `[num_rows, vocab]` logits buffer.
///
/// Returns one `(top_k_ids, top_k_logits)` pair per row, in row order, with
/// the same ordering semantics as [`top_k_from_row`]. `top_k` is clamped to
/// `vocab`. Errors on shape mismatch instead of silently truncating so a bad
/// device-side slice surfaces loudly.
pub fn top_k_rows(
    buffer: &[f32],
    num_rows: usize,
    vocab: usize,
    top_k: usize,
) -> Result<Vec<(Vec<u32>, Vec<f32>)>> {
    if vocab == 0 {
        return Err(Error::from_reason("top_k_rows: empty vocab"));
    }
    if top_k == 0 {
        return Err(Error::from_reason("top_k_rows: top_k must be >= 1"));
    }
    if buffer.len() != num_rows * vocab {
        return Err(Error::from_reason(format!(
            "top_k_rows: got {} floats, expected {} ({} rows x {} vocab)",
            buffer.len(),
            num_rows * vocab,
            num_rows,
            vocab,
        )));
    }
    let top_k = top_k.min(vocab);
    let mut out = Vec::with_capacity(num_rows);
    for r in 0..num_rows {
        let row = &buffer[r * vocab..(r + 1) * vocab];
        out.push(top_k_from_row(row, top_k));
    }
    Ok(out)
}

/// Row bounds (on the sequence axis) of the D verify positions inside the
/// full-sequence `[1, T, vocab]` logits of a `[prefix + draft]` forced-token
/// forward pass.
///
/// The logits row at sequence position `p` is the model's next-token
/// distribution for position `p + 1`, so draft token `i` (0-based) is
/// predicted by row `prefix_len - 1 + i`. The returned `(start, end)` is a
/// half-open range suitable for `MxArray::slice_axis(1, start, end)`:
/// `start = prefix_len - 1`, `end = start + draft_len`. The last row used is
/// `prefix_len + draft_len - 2 == T - 2` — the `[1, T, vocab]` logits always
/// cover the range (row `T - 1` predicts the token *after* the draft and is
/// deliberately excluded).
pub fn verify_logit_row_bounds(prefix_len: usize, draft_len: usize) -> Result<(i64, i64)> {
    if prefix_len == 0 {
        return Err(Error::from_reason(
            "verify_logit_row_bounds: prefix must contain at least 1 token",
        ));
    }
    if draft_len == 0 {
        return Err(Error::from_reason(
            "verify_logit_row_bounds: draft must contain at least 1 token",
        ));
    }
    let start = prefix_len - 1;
    Ok((start as i64, (start + draft_len) as i64))
}

/// Reference accept rule for the single-pass speculative-decode verify:
/// `accept[i] ⇔ argmax_ids[i] == draft_ids[i]` (greedy / temp-0 only).
///
/// `argmax_ids[i]` is the greedy prediction at verify position `i` (i.e.
/// `top_k_ids[0]` of row `prefix_len - 1 + i`). Returns the per-position
/// flags without prefix-truncation — the standard speculative-decode
/// "longest accepted prefix" is a caller-side fold over these flags. In
/// production the JS consumer of `scoreTokens` computes this itself; the
/// helper is kept here as the canonical, unit-tested definition.
pub fn verify_accept_flags(argmax_ids: &[u32], draft_ids: &[u32]) -> Vec<bool> {
    argmax_ids
        .iter()
        .zip(draft_ids.iter())
        .map(|(a, d)| a == d)
        .collect()
}

// ============================================================================
// NAPI surface
// ============================================================================

/// Per-step top-K logits inspector request. Mirrors `LogitsInspectorRequest`
/// in `packages/browser/src/inspector-types.ts`.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct LogitsInspectorRequest {
    /// Number of top-ranked token ids to retain per generation step.
    pub top_k: i32,
}

/// Per-layer hidden-state summary capture request. Mirrors
/// `HiddenStatesInspectorRequest` in `packages/browser/src/inspector-types.ts`.
#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct HiddenStatesInspectorRequest {
    /// Restrict capture to specific decoder-layer indices.
    /// Default (`None`): every layer.
    #[napi(ts_type = "number[] | undefined")]
    pub layers: Option<Vec<i32>>,
    /// Restrict capture to a subset of the seven canonical capture points
    /// (see `HIDDEN_STATE_POINTS` in inspector.rs). Default: all seven.
    #[napi(ts_type = "string[] | undefined")]
    pub points: Option<Vec<String>>,
}

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
    /// Capture per-step top-K logits. Default: not captured.
    #[napi(ts_type = "LogitsInspectorRequest | undefined")]
    pub logits: Option<LogitsInspectorRequest>,
    /// Capture per-layer hidden-state summary stats at the seven canonical
    /// capture points around attention / MLP / residuals. Default: not captured.
    #[napi(ts_type = "HiddenStatesInspectorRequest | undefined")]
    pub hidden_states: Option<HiddenStatesInspectorRequest>,
    /// How many new tokens to generate before returning. Default: 1.
    /// Hard-capped at 8 inside the native code — this is a visualization
    /// hook, not a chat path.
    #[napi(ts_type = "number | undefined")]
    pub max_new_tokens: Option<i32>,
    /// Wrap the prompt in the model's Jinja2 chat template (system+user roles,
    /// assistant generation prompt appended) before tokenizing. Default: false,
    /// which feeds the raw string directly to the BPE tokenizer — useful for
    /// the Tokenization chapter and any lesson that wants to show "what does
    /// the model see for this exact string?" Turning this on makes the model
    /// run in its natural chat distribution, so the predicted next token will
    /// reflect what the chat-tuned model would actually say in a conversation
    /// (e.g. `"Hello, my name is" → " ChatGPT"`-ish) instead of whatever
    /// argmax happens to find in raw-completion mode (often weird tokens like
    /// `"1"` because the instruct fine-tuning is out-of-distribution without
    /// the template wrapping).
    #[napi(ts_type = "boolean | undefined")]
    pub apply_chat_template: Option<bool>,
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

/// Per-step top-K logits payload. Matches `LogitsStep` in inspector-types.ts.
#[napi(object)]
pub struct LogitsStepNapi {
    pub step: i32,
    pub token_id: i32,
    pub top_k_ids: Vec<i32>,
    pub top_k_logits: Float32Array,
    pub top_k_texts: Vec<String>,
}

/// Per-position payload for `scoreTokens` — the single-pass
/// speculative-decode verify. Sibling of [`LogitsStepNapi`]: same top-K
/// contract (ids descending by logit, raw pre-softmax f32 logits — the
/// consumer softmaxes — plus decoded texts), but keyed by draft position
/// instead of decode step.
#[napi(object)]
pub struct ScorePositionNapi {
    /// 0-based draft index `i`. Entry `i` verifies `draft_ids[i]`.
    pub index: i32,
    /// Absolute sequence position of the logits row this entry was read
    /// from: `prefix_len - 1 + index`. The next-token distribution at that
    /// position is the model's prediction for `draft_ids[index]`.
    pub position: i32,
    /// Echo of `draft_ids[index]` — the token that was forced at this
    /// position.
    pub draft_token_id: i32,
    /// Greedy / temp-0 prediction at this position. Always equals
    /// `top_k_ids[0]` (host-side descending sort, ties broken by the sort —
    /// deterministic for a given logits buffer). The accept rule — computed
    /// by the CALLER, not here — is `argmax_token_id == draft_token_id`.
    pub argmax_token_id: i32,
    /// Top-K token ids at this position, descending by logit.
    pub top_k_ids: Vec<i32>,
    /// Raw (pre-softmax) logits matching `top_k_ids`, same order. NOT
    /// softmax probabilities — same contract as `LogitsStepNapi`.
    pub top_k_logits: Float32Array,
    /// Decoded text fragment for each id in `top_k_ids`.
    pub top_k_texts: Vec<String>,
}

/// Result of a `scoreTokens` single-pass forced-token verify. One forward
/// pass over `[prefix + draft]`, per-position top-K logits for exactly the
/// D draft positions. Caches are fully reset before the call returns
/// (single-shot, no persistent session — the hybrid GDN layers make cache
/// rewind unsafe, so the full reset mirrors `runForInspector`).
#[napi(object)]
pub struct ScoreTokensResult {
    /// Echo of the request's prefix length in tokens.
    pub prefix_len: u32,
    /// Echo of the request's draft length D (`1 ..= 8`).
    pub draft_len: u32,
    /// Effective top-K after clamping: `min(requested, 64, vocab)`.
    pub top_k: u32,
    /// One entry per draft position, index-aligned with the request's
    /// `draft_ids` (length == `draft_len`).
    pub positions: Vec<ScorePositionNapi>,
}

/// Options for a `lensReadout` call — the "logit lens" (and, when a fitted
/// lens pack is loaded, the "Jacobian lens"): read the model's vocabulary out
/// of an intermediate residual `h_ℓ` at chosen depths through the tied
/// unembedding `W_U`.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct LensReadoutOptions {
    /// Residual-stream boundaries to read, each in `0..=24` (`h_0` = embedding
    /// output … `h_24` = pre-final-norm). Default (`None`/empty): all 25.
    /// Duplicates are de-duplicated; order is preserved as first-seen.
    #[napi(ts_type = "number[] | undefined")]
    pub layers: Option<Vec<i32>>,
    /// Top-K token ids to return per cell. Clamped to
    /// `min(topK, LENS_TOP_K_CAP=32, vocab)`.
    pub top_k: u32,
    /// Vocab ids whose full-vocab rank (1-based) is returned per cell. Clamped
    /// to at most `LENS_MAX_PINNED=8`. Each id must be `< vocab`.
    #[napi(ts_type = "number[] | undefined")]
    pub pinned_ids: Option<Vec<u32>>,
    /// Apply the fitted per-layer Jacobian `J_ℓ` before the unembedding. When
    /// `true`, EVERY requested non-final layer must have a fitted `J_ℓ` in the
    /// loaded pack; if any is missing the call errors (naming the missing
    /// layers) rather than silently falling back to a logit lens. The final
    /// boundary (`ℓ==24`) is `J=I` by definition and needs no pack, so a
    /// final-boundary-only request is always allowed even with no pack loaded.
    pub use_jacobian: bool,
}

/// Per-(layer, position) readout cell. Sibling of [`ScorePositionNapi`]: same
/// top-K contract (ids descending by logit, raw pre-softmax f32 logits,
/// decoded texts), extended with full-vocab-normalized probabilities.
#[napi(object)]
pub struct LensCellNapi {
    /// Residual-stream boundary this cell was read at (`0..=24`).
    pub layer: i32,
    /// 0-based prompt position (`0..prompt_len`).
    pub position: i32,
    /// Greedy argmax token id at this cell (`= top_k_ids[0]`).
    pub argmax_id: i32,
    /// Top-K token ids, descending by logit.
    pub top_k_ids: Vec<i32>,
    /// Raw (pre-softmax) logits matching `top_k_ids`, same order. NOT
    /// probabilities — same contract as `ScorePositionNapi::top_k_logits`.
    pub top_k_logits: Float32Array,
    /// Full-vocab-normalized softmax probabilities for `top_k_ids` (computed
    /// via a device `logsumexp` over the whole `[vocab]` row — an honest
    /// probability, NOT a softmax over just the top-K).
    pub top_k_probs: Float32Array,
    /// Decoded text fragment for each id in `top_k_ids`.
    pub top_k_texts: Vec<String>,
}

/// Full-vocab rank track for a single pinned token id across every cell.
#[napi(object)]
pub struct LensPinnedNapi {
    /// The pinned vocab id.
    pub token_id: i32,
    /// Decoded text fragment for `token_id`.
    pub token_text: String,
    /// 1-based full-vocab rank of `token_id` at each cell, in the same
    /// (layer, position) order as `LensReadoutResult::cells`. Display-capped
    /// at 999. Rank = (count of tokens with strictly greater logit) + 1.
    pub ranks: Int32Array,
}

/// Result of a `lensReadout` — a `layers × positions` grid of per-cell top-K
/// readouts plus per-pinned-token rank tracks. Caches are fully reset before
/// the call returns (single-shot, no persistent session — the hybrid GDN
/// layers make cache rewind unsafe, so the full reset mirrors `scoreTokens`).
#[napi(object)]
pub struct LensReadoutResult {
    /// Prompt length `P` in tokens.
    pub prompt_len: u32,
    /// Effective top-K after clamping: `min(requested, 32, vocab)`.
    pub top_k: u32,
    /// Echo of the request's `useJacobian` flag.
    pub use_jacobian: bool,
    /// `true` iff `useJacobian` was requested AND a real per-layer `J` was
    /// actually applied to at least one requested non-final boundary. Because a
    /// missing Jacobian for a requested non-final layer is now a hard error,
    /// this equals `useJacobian && (any requested layer != 24)`. When only the
    /// final boundary is requested (`J=I@24`), this is `false` and the readout
    /// is the plain logit lens — never a Jacobian readout mislabeled as applied.
    pub jacobian_applied: bool,
    /// The readout depths actually returned, in cell-grid order.
    pub layers: Vec<i32>,
    /// Decoded prompt tokens (the UI position axis).
    pub tokens: Vec<TokenInfoNapi>,
    /// One entry per (layer, position), layer-major then position-minor.
    /// Length == `layers.len() * prompt_len`.
    pub cells: Vec<LensCellNapi>,
    /// One entry per pinned id, in request order.
    pub pinned: Vec<LensPinnedNapi>,
}

/// Summary stats for a single capture point. Matches `HiddenStatePointStats`
/// in inspector-types.ts.
///
/// NAPI does not surface `f32`, so the floats are serialized as `f64` — the
/// underlying capture is f32 (see `CapturedHiddenStatePointStats`), so the
/// extra precision is purely cosmetic and the wire size cost is trivial at
/// the volumes this hook produces.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct HiddenStatePointStatsNapi {
    /// One of the seven names from `HIDDEN_STATE_POINTS` in inspector.rs.
    pub point: String,
    pub mean: f64,
    pub std: f64,
    pub abs_max: f64,
    pub l2_norm: f64,
    pub count: u32,
}

/// One layer's hidden-state captures for a single generation step. Matches
/// `HiddenStateLayer` in inspector-types.ts.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct HiddenStateLayerNapi {
    pub layer_idx: u32,
    pub stats: Vec<HiddenStatePointStatsNapi>,
}

/// One generation step's hidden-state captures. Matches `HiddenStateStep` in
/// inspector-types.ts.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct HiddenStateStepNapi {
    /// 0-based generation step (0 == prefill, 1+ == decode steps).
    pub step: u32,
    pub layers: Vec<HiddenStateLayerNapi>,
}

/// Top-level inspector-run result. Matches `AttentionRun` in
/// inspector-types.ts.
#[napi(object)]
pub struct AttentionRunNapi {
    pub prompt: String,
    pub tokens: Vec<TokenInfoNapi>,
    /// The first generated token, kept for back-compat with callers that
    /// only need to display "the next token." For multi-token inspector
    /// runs prefer `generated_tokens` which holds the full decode sequence.
    pub generated_token: TokenInfoNapi,
    /// All tokens generated by the inspector run, in order. Length equals
    /// the effective `max_new_tokens` (may be shorter if EOS was hit). The
    /// first element matches `generated_token`. The forward-pass diagram
    /// uses this to render the model's *full reply* after a single inspector
    /// call rather than just the literal next token — important for the
    /// chat-template case, where the first token of an assistant reply is
    /// often a scaffold word ("The", "Hello") and the actually-meaningful
    /// content shows up over the next several tokens.
    pub generated_tokens: Vec<TokenInfoNapi>,
    /// Number of leading tokens in `generated_tokens` whose cumulative
    /// text matches the user's prompt as a prefix. Chat-tuned models
    /// (Qwen3.5, Gemma, LFM2) routinely start their assistant reply by
    /// echoing the user's input verbatim — `echo_token_count` tells
    /// the frontend exactly how many leading tokens to treat as echo so
    /// it can render them differently from the model's "new" content
    /// without re-doing the string match. Zero when no echo.
    pub echo_token_count: u32,
    pub attention: Vec<AttentionLayerNapi>,
    pub model_meta: ModelMetaNapi,
    /// Per-step top-K logits, in generation order. Present iff
    /// `opts.logits` was set on the inspector request.
    #[napi(ts_type = "LogitsStepNapi[] | undefined")]
    pub logits: Option<Vec<LogitsStepNapi>>,
    /// Per-step hidden-state summary stats, in generation order. Present iff
    /// `opts.hiddenStates` was set on the inspector request.
    #[napi(ts_type = "HiddenStateStepNapi[] | undefined")]
    pub hidden_states: Option<Vec<HiddenStateStepNapi>>,
}

/// Hard cap on `max_new_tokens` for an inspector run. The previous value of 8
/// was sized for "show one next token" demos; with the chat-template flag the
/// forward-pass diagram wants to surface the full assistant reply (typically
/// 10–30 tokens for short prompts), so we lift the cap to a value that keeps
/// the worker call comfortably sub-second on M3 while still bounding worst-
/// case memory if someone requests attention capture on a long sequence.
/// Refer to the module-level docs for the per-layer cost.
pub const INSPECTOR_MAX_NEW_TOKENS_CAP: i32 = 64;

/// Hard cap on the number of forced draft tokens (`D`) per `scoreTokens`
/// verify call. Speculative-decode drafts beyond ~8 tokens have vanishing
/// acceptance probability, and the cap bounds the host-side logits copy at
/// `8 * vocab * 4` bytes (~4.8 MiB for Qwen3.5's ~151k vocab).
pub const SCORE_TOKENS_MAX_DRAFT: usize = 8;

/// Cap on the requested top-K for `scoreTokens`, mirroring the inspector's
/// "top_k ≤ 64" convention (see [`InspectorRecorder::capture_logits`] — the
/// O(vocab) host-side partial sort is sized for K in that range).
///
/// Unit tests for the pure verify helpers above live in
/// `crates/mlx-core/tests/score_tokens_verify.rs` (an integration test:
/// this branch's in-crate `#[cfg(test)]` target has pre-existing unrelated
/// breakage, and the helpers are `pub` anyway).
pub const SCORE_TOKENS_TOP_K_CAP: u32 = 64;

/// Cap on the requested top-K per cell for the J-lens readout
/// (`lensReadout`). Smaller than [`SCORE_TOKENS_TOP_K_CAP`] because the lens
/// grid multiplies `top_k` by (layers × positions) cells — 32 keeps the
/// decoded-text + probability payload compact while still showing a full
/// ranked column per cell.
pub const LENS_TOP_K_CAP: u32 = 32;

/// Cap on the number of pinned vocab ids whose full-vocab rank the readout
/// tracks per cell. Each pinned id adds one device gt-count reduce per layer
/// plus one `Int32Array` of length `cells` in the result.
pub const LENS_MAX_PINNED: usize = 8;

/// Cap on the prompt length (`P`, positions) accepted by `lensReadout`. Bounds
/// the transient per-layer `[P, vocab]` logits materialized on device and the
/// number of cells (`layers × P`) crossed to host.
pub const LENS_MAX_POSITIONS: usize = 48;

/// The number of residual-stream boundaries the J-lens can read: `h_0`
/// (embedding output) through `h_24` (pre-final-norm hidden state, = the
/// model's actual pre-unembed residual). Equals `num_hidden_layers + 1`.
pub const LENS_NUM_BOUNDARIES: usize = 25;
