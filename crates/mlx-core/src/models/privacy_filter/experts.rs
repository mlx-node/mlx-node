//! Gpt-oss-style sparse Mixture-of-Experts FFN for the privacy-filter model.
//!
//! Routes each token to `top_k` experts via the per-layer `TopKRouter`
//! (built with `RoutingMode::GptOss`), then dispatches the selected
//! tokens through a fused gate-up projection, a clamped SwiGLU
//! activation, and a final down projection — all using MLX's
//! `gather_mm` to keep the matmuls dense.
//!
//! ## Weight layout
//!
//! Privacy-filter ships its expert weights in the `[E, in, out]`
//! orientation, which is the layout `gather_mm` already expects on its
//! right-hand-side argument. This **differs** from Qwen3.5's
//! [`crate::models::qwen3_5_moe::switch_linear::SwitchLinear`], whose
//! Python-style `[E, out, in]` weights have to be transposed to
//! `[E, in, out]` before every `gather_mm` call. Here we skip that
//! transpose entirely.
//!
//! - `gate_up_proj`: `[E=128, hidden=640, 2*intermediate=1280]`
//! - `gate_up_bias`:  `[E=128, 2*intermediate=1280]`
//! - `down_proj`:    `[E=128, intermediate=640, hidden=640]`
//! - `down_bias`:    `[E=128, hidden=640]`
//!
//! ## Activation
//!
//! gpt-oss's gated activation interleaves gate / up channels rather than
//! storing them as contiguous halves. The Python reference splits the
//! `[2*I]` fused output as `gate_up[..., ::2]` (gate) and
//! `gate_up[..., 1::2]` (up). We reproduce that **alternating** split by
//! reshaping the last axis `2*I → (I, 2)` and selecting the two
//! components along the trailing length-2 axis (this is the standard
//! reshape-to-pairs trick — `::2 / 1::2` slicing without strided slice
//! support).
//!
//! The activation itself is `gpt_oss.swiglu` from
//! `mlx-lm/mlx_lm/models/gpt_oss.py:49` (cross-referenced against the
//! canonical HF transformers implementation in
//! `transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py:82`).
//! In gpt-oss notation:
//!
//! ```text
//! gate = clip(gate, max=limit)
//! up   = clip(up,   min=-limit, max=limit)
//! out  = (gate * sigmoid(alpha * gate)) * (up + 1)
//! ```
//!
//! with `alpha = 1.702` and `limit = 7.0` (the hard-coded gpt-oss
//! constants, not exposed through `config.json`).
//!
//! ## Dispatch flow (mirrors `SwitchGLU::forward`)
//!
//! 1. Router → `(top_weights, top_indices)` with shape `[B, T, top_k]`.
//! 2. Sort `top_indices` so per-expert token blocks are contiguous,
//!    permuting tokens with the same `gather_sort` helper used by
//!    Qwen3.5. The branch threshold (`do_sort` when
//!    `indices.size() >= 64`) matches mlx-lm so behaviour is identical
//!    at the small-prefill and long-prefill ends.
//! 3. `gather_mm` for `gate_up`, add per-token gathered bias, split into
//!    `gate / up`, apply the clamped activation, `gather_mm` for the
//!    down projection, add per-token gathered down bias.
//! 4. Scatter back to original token order, weight by `top_weights`,
//!    sum over the `top_k` axis, reshape to `[B, T, H]`.

use crate::array::MxArray;
use crate::nn::Activations;
use napi::bindgen_prelude::*;

use super::config::PrivacyFilterConfig;
use super::persistence::MlpWeights;

/// gpt-oss SwiGLU clamp / scale constants. Hard-coded in both mlx-lm's
/// `gpt_oss.swiglu` and HF transformers' `GptOssExperts._apply_gate`;
/// not exposed via `config.json`, so we mirror them as `const`s here.
const GPT_OSS_SWIGLU_ALPHA: f64 = 1.702;
const GPT_OSS_SWIGLU_LIMIT: f64 = 7.0;

/// Threshold (in number of expert slots, i.e. `N * top_k`) above which
/// the dispatch sorts tokens by expert so that each expert's token
/// block is contiguous in memory. Matches mlx-lm's
/// `SwitchGLU.__call__` heuristic (`do_sort = indices.size >= 64`).
const SORT_THRESHOLD: u64 = 64;

/// Borrow-only view onto a single layer's MoE FFN weights. Constructed
/// fresh per forward pass alongside an `AttentionLayer`; the actual
/// tensors live in [`MlpWeights`] on the loaded model.
pub struct GptOssMlp<'a> {
    pub weights: &'a MlpWeights,
    pub config: &'a PrivacyFilterConfig,
}

/// Result of the gather-and-sort pass: tokens permuted so that every
/// expert's slice of the dispatch is contiguous, plus the inverse
/// permutation needed to undo the sort. Mirrors
/// `crate::models::qwen3_5_moe::switch_glu::GatherSortResult`.
struct SortedDispatch {
    /// Tokens replicated and reordered to align with `idx_sorted`.
    /// Shape `[N*K, 1, H]`.
    x_sorted: MxArray,
    /// Per-slot expert ids, sorted ascending (so `gather_mm(..., true)`
    /// can stream over each expert's block in one pass). Shape `[N*K]`.
    idx_sorted: MxArray,
    /// Inverse permutation used by [`scatter_unsort`] to restore the
    /// original `(B, T, K)` ordering. Shape `[N*K]`.
    inv_order: MxArray,
}

/// Permute `x` so that tokens routed to the same expert are contiguous.
///
/// This is structurally identical to
/// `crate::models::qwen3_5_moe::switch_glu::gather_sort`. We re-derive
/// it locally so the two MoE families stay independent — and because
/// the function is small enough that copying is preferable to leaking
/// it into a shared module (it has no MLX-internal dependencies beyond
/// `MxArray`).
fn gather_sort(x: &MxArray, indices: &MxArray) -> Result<SortedDispatch> {
    let idx_shape = indices.shape()?;
    let m = *idx_shape
        .last()
        .ok_or_else(|| Error::from_reason("gather_sort: empty indices"))?;

    let flat_indices = indices.reshape(&[-1])?;
    let order = flat_indices.argsort(Some(-1))?;
    let inv_order = order.argsort(Some(-1))?;
    let idx_sorted = flat_indices.take(&order, 0)?;

    let x_shape = x.shape()?;
    let d = *x_shape
        .last()
        .ok_or_else(|| Error::from_reason("gather_sort: empty x"))?;
    // Collapse all leading dims into a single token axis; insert a unit
    // axis so the subsequent gather_mm sees the canonical
    // `[tokens, 1, d]` layout used by `mx.gather_mm`.
    let x_flat = x.reshape(&[-1, 1, d])?;
    let m_scalar = MxArray::scalar_int(m as i32)?;
    let token_indices = order.floor_divide(&m_scalar)?;
    let x_sorted = x_flat.take(&token_indices, 0)?;

    Ok(SortedDispatch {
        x_sorted,
        idx_sorted,
        inv_order,
    })
}

/// Undo `gather_sort`: reorder rows of `x` by `inv_order` and reshape the
/// leading flat-token axis back into `orig_shape` (the unflattened
/// `(B, T, K)` shape).
fn scatter_unsort(x: &MxArray, inv_order: &MxArray, orig_shape: &[i64]) -> Result<MxArray> {
    let unsorted = x.take(inv_order, 0)?;
    let x_shape = unsorted.shape()?;
    let mut new_shape = orig_shape.to_vec();
    for &dim in &x_shape[1..] {
        new_shape.push(dim);
    }
    unsorted.reshape(&new_shape)
}

/// Gather a per-expert bias vector for each dispatched token. Wraps a
/// `take(axis=0)` plus an `expand_dims(-2)` to align with the
/// `[N*K, 1, out]` shape produced by `gather_mm`.
fn gather_bias(bias: &MxArray, idx_sorted: &MxArray) -> Result<MxArray> {
    // bias: [E, out], idx_sorted: [N*K] → [N*K, out]
    let gathered = bias.take(idx_sorted, 0)?;
    // Insert the matmul-style unit axis so we can broadcast-add onto
    // the `[N*K, 1, out]` matmul result.
    gathered.expand_dims(-2)
}

/// gpt-oss's clamped gated linear unit, operating on the alternating
/// `gate / up` split. See module-level docs for the source references.
///
/// Inputs:
/// - `gate`: `[N*K, 1, I]` (the `::2` slice of the fused projection)
/// - `up`:   `[N*K, 1, I]` (the `1::2` slice)
///
/// Output: `[N*K, 1, I]`.
fn gpt_oss_glu(gate: &MxArray, up: &MxArray) -> Result<MxArray> {
    // Clamp gate from above only; clamp up symmetrically. The asymmetric
    // gate clamp matches the HF / mlx-lm references and is intentional:
    // gpt-oss only worries about gate blowing up positive (because
    // sigmoid saturates) but needs full two-sided clamping on up since
    // it enters the output linearly as `(up + 1)`.
    let gate_c = gate.clip(None, Some(GPT_OSS_SWIGLU_LIMIT))?;
    let up_c = up.clip(Some(-GPT_OSS_SWIGLU_LIMIT), Some(GPT_OSS_SWIGLU_LIMIT))?;

    // glu = gate * sigmoid(alpha * gate). Multiplied via `mul_scalar` so
    // the bf16 dtype is preserved (an f32 scalar in a binary op would
    // promote the whole tensor to f32 — the well-known footgun).
    let gate_scaled = gate_c.mul_scalar(GPT_OSS_SWIGLU_ALPHA)?;
    let sig = Activations::sigmoid(&gate_scaled)?;
    let glu = gate_c.mul(&sig)?;

    // (up + 1) * glu. `add_scalar` preserves dtype the same way.
    let up_plus_one = up_c.add_scalar(1.0)?;
    up_plus_one.mul(&glu)
}

impl<'a> GptOssMlp<'a> {
    /// Forward pass: gpt-oss sparse MoE FFN with fused gate-up projections.
    ///
    /// Input:  `hidden` shape `[B, T, hidden_size]`
    /// Output: `[B, T, hidden_size]`
    pub fn forward(&self, hidden: &MxArray) -> Result<MxArray> {
        let shape = hidden.shape()?;
        if shape.len() != 3 {
            return Err(Error::from_reason(format!(
                "GptOssMlp::forward expects 3D input [B, T, H], got {}D",
                shape.len()
            )));
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];

        if hidden_size != self.config.hidden_size as i64 {
            return Err(Error::from_reason(format!(
                "GptOssMlp::forward hidden_size mismatch: input has {}, config says {}",
                hidden_size, self.config.hidden_size
            )));
        }

        let intermediate = self.config.intermediate_size as i64;
        let two_intermediate = 2 * intermediate;
        let top_k = self.config.num_experts_per_tok as i64;

        // ---- 1. Route. (B, T, K) weights + indices. ----
        let (top_weights, top_indices) = self.weights.router.route(hidden)?;
        let idx_shape: Vec<i64> = top_indices.shape()?.as_ref().to_vec();
        debug_assert_eq!(idx_shape, vec![batch, seq_len, top_k]);

        // ---- 2. Sort dispatch (or skip for very small batches). ----
        //
        // The threshold matches mlx-lm's `SwitchGLU`. Sorting pays a
        // small overhead per call but enables `gather_mm(sorted=true)`,
        // which streams each expert's tokens in one matmul instead of
        // re-binding the right-hand-side for every dispatch slot.
        let n_slots = top_indices.size()?;
        let do_sort = n_slots >= SORT_THRESHOLD;

        let (x_dispatch, idx_dispatch, inv_order) = if do_sort {
            let sorted = gather_sort(hidden, &top_indices)?;
            (sorted.x_sorted, sorted.idx_sorted, Some(sorted.inv_order))
        } else {
            // Unsorted path: reshape hidden to `[B*T, 1, 1, H]` so the
            // gather_mm's per-slot expansion still works. `top_indices`
            // is passed as `[B, T, K]` and gather_mm broadcasts
            // appropriately. This mirrors `SwitchGLU`'s small-batch
            // branch.
            let x_expanded = hidden.reshape(&[batch * seq_len, 1, 1, hidden_size])?;
            (x_expanded, top_indices.clone(), None)
        };

        // ---- 3. Fused gate-up projection. ----
        //
        // Weights are already `[E, in, out]`, which is exactly what
        // gather_mm wants — no transpose required. (Contrast with
        // Qwen3.5's `SwitchLinear`, which transposes from `[E, out, in]`
        // on every call.)
        let gate_up = x_dispatch.gather_mm(&self.weights.gate_up_proj, &idx_dispatch, do_sort)?;
        let gate_up_bias = gather_bias(&self.weights.gate_up_bias, &idx_dispatch)?;
        let gate_up = gate_up.add(&gate_up_bias)?;

        // ---- 4. Alternating gate / up split. ----
        //
        // The trailing axis (length `2*I`) interleaves gate and up
        // channels. Reshape the last axis to `(I, 2)` so position 0 is
        // gate (the `::2` slice) and position 1 is up (the `1::2`
        // slice). We then index each half with a length-1 slice along
        // the new last axis and squeeze it out — `MxArray` doesn't yet
        // expose strided slicing, but this reshape-and-slice trick is
        // mathematically equivalent.
        let gate_up_shape: Vec<i64> = gate_up.shape()?.as_ref().to_vec();
        // gate_up_shape = [..., 1, 2*I]; replace the last dim with (I, 2).
        let mut paired_shape = gate_up_shape.clone();
        let last = paired_shape
            .pop()
            .ok_or_else(|| Error::from_reason("unexpected scalar gate_up"))?;
        if last != two_intermediate {
            return Err(Error::from_reason(format!(
                "gate_up trailing dim {last} != 2 * intermediate_size {two_intermediate}"
            )));
        }
        paired_shape.push(intermediate);
        paired_shape.push(2);
        let gate_up_paired = gate_up.reshape(&paired_shape)?;

        // slice along the trailing length-2 axis to extract gate (idx 0)
        // and up (idx 1), then drop the now-length-1 final axis.
        let ndim = paired_shape.len();
        let mut starts = vec![0i64; ndim];
        let mut stops = paired_shape.clone();
        // gate: last axis [0, 1)
        stops[ndim - 1] = 1;
        let gate = gate_up_paired.slice(&starts, &stops)?;
        // up: last axis [1, 2)
        starts[ndim - 1] = 1;
        stops[ndim - 1] = 2;
        let up = gate_up_paired.slice(&starts, &stops)?;
        // Drop the trailing length-1 axis so shapes match what
        // `gpt_oss_glu` and the downstream `gather_mm` expect.
        let gate = gate.squeeze(Some(&[-1]))?;
        let up = up.squeeze(Some(&[-1]))?;

        // ---- 5. Clamped gpt-oss SwiGLU. ----
        let activated = gpt_oss_glu(&gate, &up)?;

        // ---- 6. Down projection (+ per-expert bias). ----
        let down = activated.gather_mm(&self.weights.down_proj, &idx_dispatch, do_sort)?;
        let down_bias = gather_bias(&self.weights.down_bias, &idx_dispatch)?;
        let expert_out = down.add(&down_bias)?;

        // ---- 7. Restore token order. ----
        //
        // `expert_out` is `[N*K, 1, H]` (sorted) or, in the unsorted
        // branch, whatever the gather_mm broadcasting produced.
        let unsorted = match inv_order {
            Some(inv) => scatter_unsort(&expert_out, &inv, &idx_shape)?,
            None => {
                // Unsorted branch already preserves original ordering; we
                // just need to reshape so the per-token / per-k axes are
                // separated for the weighted sum.
                let out_shape = [batch, seq_len, top_k, 1, hidden_size];
                expert_out.reshape(&out_shape)?
            }
        };

        // ---- 8. Squeeze the matmul unit axis. ----
        // Sorted branch: `[B, T, K, 1, H]` → `[B, T, K, H]`.
        // Unsorted branch: same, since we explicitly inserted the unit
        // axis above.
        let per_expert = unsorted.squeeze(Some(&[-2]))?;

        // ---- 9. Weighted combine over the top_k axis. ----
        let weights_expanded = top_weights.reshape(&[batch, seq_len, top_k, 1])?;
        let weighted = per_expert.mul(&weights_expanded)?;
        weighted.sum(Some(&[-2]), Some(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;
    use crate::models::privacy_filter::persistence::load_from_directory;

    fn checkpoint_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models/privacy-filter")
    }

    /// Layer-0 MoE forward against the real checkpoint. We deliberately
    /// keep this light — full numerical parity is C7's job. The two
    /// invariants we pin here are the ones most likely to break during
    /// refactors of the dispatch / split logic:
    ///
    /// 1. Output shape is `[B, T, hidden_size]`.
    /// 2. Output is finite (no NaN / Inf produced by clamp, sigmoid,
    ///    gather_mm, or the bias adds).
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn forward_output_shape_and_finite() {
        let loaded = load_from_directory(&checkpoint_dir()).expect("load model");
        let mlp = GptOssMlp {
            weights: &loaded.weights.layers[0].mlp,
            config: &loaded.config,
        };

        // Generate hidden in the layer's native dtype (bf16) so the
        // matmuls don't trip the f32-promotes-bf16 footgun. We use the
        // gate_up_proj dtype rather than the LayerNorm one: bf16 vs
        // bf16 here, but the matmul is the one that actually cares.
        let hidden_dtype = loaded.weights.layers[0].mlp.gate_up_proj.dtype().unwrap();
        let hidden = MxArray::random_normal(
            &[1, 8, loaded.config.hidden_size as i64],
            0.0,
            1.0,
            Some(hidden_dtype),
        )
        .expect("random hidden");

        let out = mlp.forward(&hidden).expect("mlp forward");
        let shape = out.shape().unwrap().to_vec();
        assert_eq!(shape, vec![1, 8, loaded.config.hidden_size as i64]);

        // Finiteness via the GPU-resident reduction (same idiom as the
        // attention test) — promotes to f32 only for the reduction.
        let out_f32 = out.astype(DType::Float32).unwrap();
        assert!(
            !out_f32.has_nan_or_inf().unwrap(),
            "GptOssMlp::forward produced NaN or Inf"
        );
    }
}
