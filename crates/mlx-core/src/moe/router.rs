//! Shared top-k router for Mixture-of-Experts.
//!
//! Two routing conventions are supported, mirroring the upstream Python
//! implementations in `mlx-lm`:
//!
//! - [`RoutingMode::Qwen35`] — softmax-then-top-k, with conditional
//!   renormalization (mirrors `Qwen3MoeSparseMoeBlock` in
//!   `mlx-lm/mlx_lm/models/qwen3_moe.py` and that model's
//!   `config.norm_topk_prob` flag):
//!     1. `logits = hidden @ weight.T + bias`  → `[B, T, E]`
//!     2. `probs = softmax(logits, axis=-1)`
//!     3. `(top_probs, top_indices) = topk(probs)` along axis=-1
//!     4. If `renormalize_topk`, divide `top_probs` by their sum.
//!
//! - [`RoutingMode::GptOss`] — top-k-then-softmax (mirrors `MLPBlock` in
//!   `mlx-lm/mlx_lm/models/gpt_oss.py`):
//!     1. `logits = hidden @ weight.T + bias`  → `[B, T, E]`
//!     2. `(top_logits, top_indices) = topk(logits)` along axis=-1
//!     3. `weights = softmax(top_logits, axis=-1)` — auto-sums to 1, no
//!        renormalization needed. Mathematically the cheaper path because
//!        softmax is taken over `k`, not the full `E`.
//!
//! Both modes return `(weights, indices)` with shape `[B, T, top_k]` each.
//!
//! This module is intentionally minimal: it owns no expert MLPs and no
//! shared-expert branch. Dispatch (token → expert routing) lives in
//! [`crate::moe::dispatch`].

use crate::array::MxArray;
use crate::nn::Activations;
use napi::bindgen_prelude::*;

/// Routing convention selector.
#[derive(Debug, Clone, Copy)]
pub enum RoutingMode {
    /// gpt-oss style: top-k of logits, then softmax over top-k. No
    /// renormalization (softmax over the top-k already sums to 1).
    GptOss,
    /// Qwen3.5 style: softmax over all experts, then top-k.
    ///
    /// If `renormalize_topk` is true, divide the top-k weights by their sum
    /// (so they sum to 1.0). Mirrors Qwen3.5's `config.norm_topk_prob`.
    Qwen35 { renormalize_topk: bool },
}

/// Compute top-k routing weights and indices from raw router logits.
///
/// `logits` must be 2D with shape `[N, num_experts]` where `N` is any flat
/// batch (typically `batch * seq_len`). Returns `(weights, indices)` each of
/// shape `[N, top_k]`. Indices are MLX `uint32` (the dtype returned by
/// `argpartition`).
///
/// This function is the single source of truth for the topk + softmax +
/// renormalize math. It is consumed in two places:
///
/// - [`TopKRouter::route`], the all-in-one router used by models that own
///   their gate weight as a plain `MxArray` (e.g. privacy-filter, gpt-oss).
/// - `Qwen3_5MoeSparseMoeBlock::forward`, which retains its own
///   `LinearProj` gate so it can run a quantized router matmul, but
///   delegates the routing math here for parity with the shared router.
///
/// The dispatch on `mode` mirrors the description in this module's docs.
pub fn topk_from_logits(
    logits: &MxArray,
    num_experts: i32,
    top_k: i32,
    mode: RoutingMode,
) -> Result<(MxArray, MxArray)> {
    if top_k <= 0 || top_k > num_experts {
        return Err(Error::from_reason(format!(
            "topk_from_logits requires 0 < top_k <= num_experts, got top_k={}, num_experts={}",
            top_k, num_experts
        )));
    }
    let num_experts_i64 = num_experts as i64;
    let top_k_i64 = top_k as i64;

    match mode {
        RoutingMode::GptOss => {
            // Top-k of logits, then softmax over the k selected logits.
            // argpartition with kth=-k puts the k largest at the tail.
            let top_indices_full = logits.argpartition(-top_k, Some(-1))?;
            let top_indices =
                top_indices_full.slice_axis(1, num_experts_i64 - top_k_i64, num_experts_i64)?;
            let top_logits = logits.take_along_axis(&top_indices, -1)?;
            let top_weights = Activations::softmax(&top_logits, Some(-1))?;
            Ok((top_weights, top_indices))
        }
        RoutingMode::Qwen35 { renormalize_topk } => {
            // Softmax over all experts, then top-k of probs, with
            // optional renormalization.
            let routing_weights = Activations::softmax(logits, Some(-1))?;
            let top_indices_full = routing_weights.argpartition(-top_k, Some(-1))?;
            let top_indices =
                top_indices_full.slice_axis(1, num_experts_i64 - top_k_i64, num_experts_i64)?;
            let top_weights = routing_weights.take_along_axis(&top_indices, -1)?;
            let top_weights = if renormalize_topk {
                let sum = top_weights.sum(Some(&[-1]), Some(true))?;
                top_weights.div(&sum)?
            } else {
                top_weights
            };
            Ok((top_weights, top_indices))
        }
    }
}

/// Compute top-k routing weights and indices using sigmoid gating with an
/// optional selection-only expert bias (DeepSeek-V3 / LFM2 / NemotronH /
/// Inkling2 convention):
///
/// 1. `scores = sigmoid(logits.astype(Float32))` (skips cast when already F32)
/// 2. `selection = scores + bias` when `bias` is `Some`, else `scores`
/// 3. Top-k selection along axis `-1`:
///    - `deterministic_tie_break = true`: descending `argsort(-selection)[:, :top_k]`
///      (breaks exact ties in favor of lower expert indices; used by Inkling2)
///    - `deterministic_tie_break = false`: `argpartition(selection, -top_k)[:, E-top_k:]`
///      (used by LFM2 and NemotronH)
/// 4. Gather routing weights from the **unbiased** `scores` at `top_indices`.
/// 5. If `norm_topk`, divide by `sum(weights, axis=-1, keepdims=true) + renorm_eps`.
/// 6. If `route_scale != 1.0`, multiply weights by `route_scale`.
///
/// `logits` must be 2D `[N, num_experts]`. Returns `(weights, indices)` each
/// of shape `[N, top_k]`, with `weights` in `Float32` so each caller can cast
/// back to its activation dtype at its own boundary.
#[allow(clippy::too_many_arguments)]
pub fn sigmoid_biased_topk_from_logits(
    logits: &MxArray,
    bias: Option<&MxArray>,
    top_k: i32,
    renorm_eps: f64,
    route_scale: f64,
    norm_topk: bool,
    deterministic_tie_break: bool,
) -> Result<(MxArray, MxArray)> {
    let shape = logits.shape()?;
    if shape.len() != 2 {
        return Err(Error::from_reason(format!(
            "sigmoid_biased_topk_from_logits expects 2D [N, num_experts] logits, got {}D",
            shape.len()
        )));
    }
    let num_experts = shape[1];
    let top_k_i64 = i64::from(top_k);
    if top_k <= 0 || top_k_i64 > num_experts {
        return Err(Error::from_reason(format!(
            "sigmoid_biased_topk_from_logits requires 0 < top_k <= num_experts, got top_k={}, num_experts={}",
            top_k, num_experts
        )));
    }

    let logits_f32 = if logits.dtype()? == crate::array::DType::Float32 {
        logits.clone()
    } else {
        logits.astype(crate::array::DType::Float32)?
    };
    let scores = Activations::sigmoid(&logits_f32)?;
    let selection = match bias {
        Some(b) => scores.add(b)?,
        None => scores.clone(),
    };

    let top_indices = if deterministic_tie_break {
        selection
            .negative()?
            .argsort(Some(-1))?
            .slice_axis(1, 0, top_k_i64)?
    } else {
        let inds_full = selection.argpartition(-top_k, Some(-1))?;
        inds_full.slice_axis(1, num_experts - top_k_i64, num_experts)?
    };

    let mut top_weights = scores.take_along_axis(&top_indices, -1)?;
    if norm_topk {
        let denom = top_weights
            .sum(Some(&[-1]), Some(true))?
            .add_scalar(renorm_eps)?;
        top_weights = top_weights.div(&denom)?;
    }
    if route_scale != 1.0 {
        top_weights = top_weights.mul_scalar(route_scale)?;
    }
    Ok((top_weights, top_indices))
}

/// Static configuration for a [`TopKRouter`].
#[derive(Debug, Clone, Copy)]
pub struct RouterConfig {
    pub num_experts: usize,
    pub hidden: usize,
    pub top_k: usize,
    pub mode: RoutingMode,
}

/// Top-k MoE router.
///
/// Stores its own `weight` (`[num_experts, hidden]`) and `bias`
/// (`[num_experts]`) tensors plus a cached transposed view of the weight to
/// avoid rebuilding the transpose graph node on every forward pass. The
/// `route` method computes routing weights and indices for an input
/// `hidden [B, T, H]` according to `config.mode`.
pub struct TopKRouter {
    pub config: RouterConfig,
    /// Router gate weight, shape `[num_experts, hidden]`.
    pub weight: MxArray,
    /// Cached transposed weight, shape `[hidden, num_experts]`. Built once
    /// at construction; used by `route` via `addmm` for the fused matmul +
    /// bias add. Mirrors the `weight_t` caching pattern in
    /// `crate::nn::linear::Linear`.
    weight_t: MxArray,
    /// Router gate bias, shape `[num_experts]`.
    pub bias: MxArray,
}

impl TopKRouter {
    /// Construct a new router. The caller is responsible for the weight and
    /// bias shapes matching `config`; validation happens lazily on the first
    /// `route` call (via MLX shape errors) to avoid evaluating lazy tensors.
    pub fn new(config: RouterConfig, weight: MxArray, bias: MxArray) -> Result<Self> {
        let weight_t = weight.transpose(Some(&[1, 0]))?;
        Ok(Self {
            config,
            weight,
            weight_t,
            bias,
        })
    }

    /// Compute the top-k routing weights and indices for `hidden`.
    ///
    /// `hidden` must have shape `[B, T, H]`. The result is
    /// `(top_k_weights, top_k_indices)` with shape `[B, T, top_k]` each.
    /// Internally this flattens the leading dims to a 2D `[B*T, H]` matmul,
    /// then delegates the topk + softmax + (optional) renormalize math to
    /// [`topk_from_logits`], reshaping the result back to `[B, T, top_k]`.
    pub fn route(&self, hidden: &MxArray) -> Result<(MxArray, MxArray)> {
        let shape = hidden.shape()?;
        if shape.len() != 3 {
            return Err(Error::from_reason(format!(
                "TopKRouter::route expects 3D input [B, T, H], got {}D",
                shape.len()
            )));
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_dim = shape[2];

        let num_experts = self.config.num_experts as i32;
        let top_k = self.config.top_k as i32;

        // Flatten leading dims for the matmul, like sparse_moe.rs.
        let ne = batch * seq_len;
        let x_flat = hidden.reshape(&[ne, hidden_dim])?;

        // Fused logits = x_flat @ weight_t + bias  → [ne, num_experts]
        // weight_t is cached; addmm fuses the bias add. Mirrors
        // `Linear::forward` in `crate::nn::linear`.
        let logits = x_flat.addmm(&self.bias, &self.weight_t, None, None)?;

        let (top_weights_flat, top_indices_flat) =
            topk_from_logits(&logits, num_experts, top_k, self.config.mode)?;

        // Reshape back to [B, T, top_k].
        let out_shape = [batch, seq_len, top_k as i64];
        let top_weights = top_weights_flat.reshape(&out_shape)?;
        let top_indices = top_indices_flat.reshape(&out_shape)?;

        Ok((top_weights, top_indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    fn router(num_experts: usize, hidden: usize, top_k: usize, mode: RoutingMode) -> TopKRouter {
        let config = RouterConfig {
            num_experts,
            hidden,
            top_k,
            mode,
        };
        let weight = MxArray::random_normal(
            &[num_experts as i64, hidden as i64],
            0.0,
            0.02,
            Some(DType::Float32),
        )
        .expect("random_normal weight");
        let bias = MxArray::zeros(&[num_experts as i64], Some(DType::Float32)).expect("zeros bias");
        TopKRouter::new(config, weight, bias).expect("router new")
    }

    #[test]
    fn test_route_shapes() {
        let r = router(
            8,
            4,
            2,
            RoutingMode::Qwen35 {
                renormalize_topk: true,
            },
        );
        let hidden = MxArray::random_normal(&[2, 3, 4], 0.0, 1.0, Some(DType::Float32))
            .expect("random_normal hidden");
        let (weights, indices) = r.route(&hidden).expect("route");

        let w_shape: Vec<i64> = weights.shape().expect("weights shape").as_ref().to_vec();
        let i_shape: Vec<i64> = indices.shape().expect("indices shape").as_ref().to_vec();
        assert_eq!(w_shape, vec![2, 3, 2]);
        assert_eq!(i_shape, vec![2, 3, 2]);
    }

    #[test]
    fn test_route_gptoss_mode_no_renormalization() {
        // gpt-oss mode: softmax-over-top-k auto-normalizes, so per-token
        // weight sum must be ~1.0 with no explicit renormalization.
        let num_experts = 8usize;
        let top_k = 4usize;
        let r = router(num_experts, 4, top_k, RoutingMode::GptOss);
        let hidden = MxArray::random_normal(&[2, 3, 4], 0.0, 1.0, Some(DType::Float32))
            .expect("random_normal hidden");
        let (weights, _indices) = r.route(&hidden).expect("route");

        let sums = weights.sum(Some(&[-1]), Some(false)).expect("sum over -1");
        sums.eval();
        let flat = sums.to_float32().expect("to_float32");
        assert_eq!(flat.len(), 2 * 3);
        for (i, &v) in flat.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "gpt-oss row {} weight sum is {} (expected 1.0 +/- 1e-4)",
                i,
                v
            );
        }
    }

    #[test]
    fn test_route_qwen35_renormalize_off_does_not_sum_to_one() {
        // Qwen35 mode without renormalization: weights are the top-k probs
        // of the full softmax distribution. Their sum equals the mass of
        // the top-k experts, which is < 1.0 unless top_k == num_experts.
        let num_experts = 16usize;
        let top_k = 4usize;
        let r = router(
            num_experts,
            8,
            top_k,
            RoutingMode::Qwen35 {
                renormalize_topk: false,
            },
        );
        let hidden = MxArray::random_normal(&[2, 3, 8], 0.0, 1.0, Some(DType::Float32))
            .expect("random_normal hidden");
        let (weights, _indices) = r.route(&hidden).expect("route");

        let sums = weights.sum(Some(&[-1]), Some(false)).expect("sum over -1");
        sums.eval();
        let flat = sums.to_float32().expect("to_float32");
        assert_eq!(flat.len(), 2 * 3);
        for (i, &v) in flat.iter().enumerate() {
            // Strictly less than 1.0 (some mass lives outside the top-k).
            // Also positive: softmax values are non-negative and at least
            // one entry is selected.
            assert!(
                v > 0.0 && v < 1.0 - 1e-6,
                "qwen35(renorm=false) row {} weight sum is {} (expected (0, 1))",
                i,
                v
            );
        }
    }

    #[test]
    fn test_route_qwen35_renormalize_on_sums_to_one() {
        let num_experts = 8usize;
        let top_k = 2usize;
        let r = router(
            num_experts,
            4,
            top_k,
            RoutingMode::Qwen35 {
                renormalize_topk: true,
            },
        );
        let hidden = MxArray::random_normal(&[2, 3, 4], 0.0, 1.0, Some(DType::Float32))
            .expect("random_normal hidden");
        let (weights, _indices) = r.route(&hidden).expect("route");

        let sums = weights.sum(Some(&[-1]), Some(false)).expect("sum over -1");
        sums.eval();
        let flat = sums.to_float32().expect("to_float32");
        assert_eq!(flat.len(), 2 * 3);
        for (i, &v) in flat.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "qwen35(renorm=true) row {} weight sum is {} (expected 1.0 +/- 1e-4)",
                i,
                v
            );
        }
    }

    #[test]
    fn test_route_indices_in_range() {
        let num_experts = 32usize;
        let top_k = 4usize;
        let hidden_dim = 8usize;
        let r = router(num_experts, hidden_dim, top_k, RoutingMode::GptOss);
        let hidden =
            MxArray::random_normal(&[4, 16, hidden_dim as i64], 0.0, 1.0, Some(DType::Float32))
                .expect("random_normal hidden");
        let (_weights, indices) = r.route(&hidden).expect("route");

        indices.eval();
        // argpartition returns Uint32 indices in MLX.
        let dtype = indices.dtype().expect("indices dtype");
        match dtype {
            DType::Uint32 => {
                let flat = indices.to_uint32().expect("to_uint32");
                let max_exclusive = num_experts as u32;
                for (i, &idx) in flat.iter().enumerate() {
                    assert!(
                        idx < max_exclusive,
                        "index {} at pos {} is out of range [0, {})",
                        idx,
                        i,
                        max_exclusive
                    );
                }
            }
            DType::Int32 => {
                let flat = indices.to_int32().expect("to_int32");
                let max_exclusive = num_experts as i32;
                for (i, &idx) in flat.iter().enumerate() {
                    assert!(
                        idx >= 0 && idx < max_exclusive,
                        "index {} at pos {} is out of range [0, {})",
                        idx,
                        i,
                        max_exclusive
                    );
                }
            }
            other => panic!("unexpected index dtype: {:?}", other),
        }
    }

    // ---------- topk_from_logits direct tests ----------
    //
    // These cover the shared helper's public contract (used by both
    // `TopKRouter` and `Qwen3_5MoeSparseMoeBlock`) without going through
    // the router's matmul. Shape: 2D `[N, num_experts]` logits in;
    // `[N, top_k]` weights + indices out.

    fn logits(n: i64, num_experts: i64) -> MxArray {
        MxArray::random_normal(&[n, num_experts], 0.0, 1.0, Some(DType::Float32))
            .expect("random_normal logits")
    }

    #[test]
    fn test_topk_from_logits_shapes() {
        let num_experts = 8i32;
        let top_k = 3i32;
        let n: i64 = 7;
        let l = logits(n, num_experts as i64);
        let (weights, indices) = topk_from_logits(
            &l,
            num_experts,
            top_k,
            RoutingMode::Qwen35 {
                renormalize_topk: true,
            },
        )
        .expect("topk_from_logits");

        let w_shape: Vec<i64> = weights.shape().expect("weights shape").as_ref().to_vec();
        let i_shape: Vec<i64> = indices.shape().expect("indices shape").as_ref().to_vec();
        assert_eq!(w_shape, vec![n, top_k as i64]);
        assert_eq!(i_shape, vec![n, top_k as i64]);
    }

    #[test]
    fn test_topk_from_logits_gptoss_sums_to_one() {
        let num_experts = 8i32;
        let top_k = 4i32;
        let n: i64 = 6;
        let l = logits(n, num_experts as i64);
        let (weights, _indices) = topk_from_logits(&l, num_experts, top_k, RoutingMode::GptOss)
            .expect("topk_from_logits");

        let sums = weights.sum(Some(&[-1]), Some(false)).expect("sum over -1");
        sums.eval();
        let flat = sums.to_float32().expect("to_float32");
        assert_eq!(flat.len(), n as usize);
        for (i, &v) in flat.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "gpt-oss row {} weight sum is {} (expected 1.0 +/- 1e-4)",
                i,
                v
            );
        }
    }

    #[test]
    fn test_topk_from_logits_qwen35_renorm_off_in_open_unit() {
        let num_experts = 16i32;
        let top_k = 4i32;
        let n: i64 = 6;
        let l = logits(n, num_experts as i64);
        let (weights, _indices) = topk_from_logits(
            &l,
            num_experts,
            top_k,
            RoutingMode::Qwen35 {
                renormalize_topk: false,
            },
        )
        .expect("topk_from_logits");

        let sums = weights.sum(Some(&[-1]), Some(false)).expect("sum over -1");
        sums.eval();
        let flat = sums.to_float32().expect("to_float32");
        assert_eq!(flat.len(), n as usize);
        for (i, &v) in flat.iter().enumerate() {
            assert!(
                v > 0.0 && v < 1.0 - 1e-6,
                "qwen35(renorm=false) row {} weight sum is {} (expected (0, 1))",
                i,
                v
            );
        }
    }

    #[test]
    fn test_topk_from_logits_qwen35_renorm_on_sums_to_one() {
        let num_experts = 8i32;
        let top_k = 2i32;
        let n: i64 = 6;
        let l = logits(n, num_experts as i64);
        let (weights, _indices) = topk_from_logits(
            &l,
            num_experts,
            top_k,
            RoutingMode::Qwen35 {
                renormalize_topk: true,
            },
        )
        .expect("topk_from_logits");

        let sums = weights.sum(Some(&[-1]), Some(false)).expect("sum over -1");
        sums.eval();
        let flat = sums.to_float32().expect("to_float32");
        assert_eq!(flat.len(), n as usize);
        for (i, &v) in flat.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "qwen35(renorm=true) row {} weight sum is {} (expected 1.0 +/- 1e-4)",
                i,
                v
            );
        }
    }

    #[test]
    fn test_topk_from_logits_indices_in_range() {
        let num_experts = 32i32;
        let top_k = 5i32;
        let n: i64 = 12;
        let l = logits(n, num_experts as i64);
        let (_weights, indices) = topk_from_logits(&l, num_experts, top_k, RoutingMode::GptOss)
            .expect("topk_from_logits");
        indices.eval();
        let dtype = indices.dtype().expect("indices dtype");
        match dtype {
            DType::Uint32 => {
                let flat = indices.to_uint32().expect("to_uint32");
                let max_exclusive = num_experts as u32;
                for (i, &idx) in flat.iter().enumerate() {
                    assert!(
                        idx < max_exclusive,
                        "index {} at pos {} is out of range [0, {})",
                        idx,
                        i,
                        max_exclusive
                    );
                }
            }
            DType::Int32 => {
                let flat = indices.to_int32().expect("to_int32");
                let max_exclusive = num_experts;
                for (i, &idx) in flat.iter().enumerate() {
                    assert!(
                        idx >= 0 && idx < max_exclusive,
                        "index {} at pos {} is out of range [0, {})",
                        idx,
                        i,
                        max_exclusive
                    );
                }
            }
            other => panic!("unexpected index dtype: {:?}", other),
        }
    }

    #[test]
    fn test_topk_from_logits_rejects_bad_top_k() {
        let l = logits(2, 4);
        // top_k > num_experts
        let r = topk_from_logits(&l, 4, 5, RoutingMode::GptOss);
        assert!(r.is_err(), "expected error when top_k > num_experts");
        // top_k = 0
        let r = topk_from_logits(&l, 4, 0, RoutingMode::GptOss);
        assert!(r.is_err(), "expected error when top_k == 0");
    }

    // ---------- sigmoid_biased_topk_from_logits direct tests ----------

    #[test]
    fn test_sigmoid_biased_topk_bias_selects_without_contaminating_weights() {
        // Logits favor experts 0 and 1, while a large bias on experts 2 and 3
        // flips top-2 selection to {2, 3}. Returned weights must be derived
        // from the UNBIASED sigmoid(logits) of experts {2, 3}.
        let l = MxArray::from_float32(&[2.0, 1.5, -1.0, -2.0], &[1, 4]).unwrap();
        let bias = MxArray::from_float32(&[0.0, 0.0, 5.0, 5.0], &[4]).unwrap();

        let (weights, indices) =
            sigmoid_biased_topk_from_logits(&l, Some(&bias), 2, 1e-6, 2.5, true, false).unwrap();
        weights.eval();
        indices.eval();
        let idx = indices.to_uint32().unwrap();
        let w = weights.to_float32().unwrap();

        let mut sorted_idx = idx.to_vec();
        sorted_idx.sort_unstable();
        assert_eq!(sorted_idx, vec![2, 3]);

        let sig = |z: f32| 1.0f32 / (1.0 + (-z).exp());
        let s2 = sig(-1.0);
        let s3 = sig(-2.0);
        let denom = s2 + s3 + 1e-6;
        for (&e, &got) in idx.iter().zip(w.iter()) {
            let expected = (if e == 2 { s2 } else { s3 }) / denom * 2.5;
            assert!(
                (got - expected).abs() < 1e-5,
                "expert {e}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_sigmoid_biased_topk_deterministic_tie_break_prefers_lower_indices() {
        // All logits and biases equal -> deterministic tie-break must pick [0, 1].
        let l = MxArray::from_float32(&[0.0, 0.0, 0.0, 0.0], &[1, 4]).unwrap();
        let (weights, indices) =
            sigmoid_biased_topk_from_logits(&l, None, 2, 1e-20, 1.0, true, true).unwrap();
        weights.eval();
        indices.eval();
        assert_eq!(indices.to_uint32().unwrap().as_ref(), &[0, 1]);
        let w = weights.to_float32().unwrap();
        assert!((w[0] - 0.5).abs() < 1e-5);
        assert!((w[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_biased_topk_unnormalized_and_validation() {
        let l = MxArray::from_float32(&[0.0, 1.0, 2.0, -1.0], &[1, 4]).unwrap();
        let (weights, indices) =
            sigmoid_biased_topk_from_logits(&l, None, 2, 1e-6, 1.0, false, true).unwrap();
        weights.eval();
        indices.eval();
        assert_eq!(indices.to_uint32().unwrap().as_ref(), &[2, 1]);
        let sig = |z: f32| 1.0f32 / (1.0 + (-z).exp());
        let w = weights.to_float32().unwrap();
        assert!((w[0] - sig(2.0)).abs() < 1e-5);
        assert!((w[1] - sig(1.0)).abs() < 1e-5);

        assert!(sigmoid_biased_topk_from_logits(&l, None, 0, 1e-6, 1.0, true, false).is_err());
        assert!(sigmoid_biased_topk_from_logits(&l, None, 5, 1e-6, 1.0, true, false).is_err());
    }
}
