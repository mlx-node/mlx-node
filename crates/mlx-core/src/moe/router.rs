//! Shared top-k softmax router for Mixture-of-Experts.
//!
//! Convention (gpt-oss / Qwen3.5 MoE):
//! 1. Linear projection: `logits = hidden @ weight.T + bias`  → `[B, T, E]`
//! 2. Softmax over all experts (axis=-1).
//! 3. Take top-k indices via `argpartition(-k)` and `take_along_axis`.
//! 4. Renormalize the top-k weights so they sum to 1 along axis=-1.
//!
//! This module is intentionally minimal: it owns no expert MLPs and no
//! shared-expert branch. Dispatch (token → expert routing) lives in
//! `crate::moe::dispatch`.

use crate::array::MxArray;
use crate::nn::Activations;
use napi::bindgen_prelude::*;

/// Static configuration for a [`TopKRouter`].
#[derive(Debug, Clone, Copy)]
pub struct RouterConfig {
    pub num_experts: usize,
    pub hidden: usize,
    pub top_k: usize,
}

/// Top-k softmax router.
///
/// Stores its own `weight` (`[num_experts, hidden]`) and `bias`
/// (`[num_experts]`) tensors. The `route` method computes routing
/// weights and indices for an input `hidden [B, T, H]`.
pub struct TopKRouter {
    pub config: RouterConfig,
    /// Router gate weight, shape `[num_experts, hidden]`.
    pub weight: MxArray,
    /// Router gate bias, shape `[num_experts]`.
    pub bias: MxArray,
}

impl TopKRouter {
    /// Construct a new router. The caller is responsible for the weight and
    /// bias shapes matching `config`; validation happens lazily on the first
    /// `route` call (via MLX shape errors) to avoid evaluating lazy tensors.
    pub fn new(config: RouterConfig, weight: MxArray, bias: MxArray) -> Self {
        Self {
            config,
            weight,
            bias,
        }
    }

    /// Compute the top-k routing weights and indices for `hidden`.
    ///
    /// `hidden` must have shape `[B, T, H]`. The result is
    /// `(top_k_weights, top_k_indices)` with shape `[B, T, top_k]` each.
    ///
    /// `top_k_weights` are softmax-then-top-k normalized: a full softmax is
    /// taken across all `num_experts`, the top-`k` are kept, and the kept
    /// values are renormalized to sum to 1 along the last axis.
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

        let num_experts = self.config.num_experts as i64;
        let top_k = self.config.top_k as i64;
        if top_k <= 0 || top_k > num_experts {
            return Err(Error::from_reason(format!(
                "TopKRouter::route requires 0 < top_k <= num_experts, got top_k={}, num_experts={}",
                top_k, num_experts
            )));
        }

        // Flatten leading dims for the matmul, like sparse_moe.rs.
        let ne = batch * seq_len;
        let x_flat = hidden.reshape(&[ne, hidden_dim])?;

        // logits = x_flat @ weight.T + bias  → [ne, num_experts]
        let weight_t = self.weight.transpose(Some(&[1, 0]))?;
        let logits = x_flat.matmul(&weight_t)?.add(&self.bias)?;

        // Softmax over all experts along the last axis.
        let routing_weights = Activations::softmax(&logits, Some(-1))?;

        // Top-k via argpartition: kth = -k partitions so the last k positions
        // along axis=-1 are the k largest. slice_axis on axis=1 keeps those.
        let top_indices_full = routing_weights.argpartition(-(top_k as i32), Some(-1))?;
        let top_indices_flat = top_indices_full.slice_axis(1, num_experts - top_k, num_experts)?;
        let top_weights_flat = routing_weights.take_along_axis(&top_indices_flat, -1)?;

        // Renormalize the top-k weights to sum to 1 along axis=-1.
        let sum = top_weights_flat.sum(Some(&[-1]), Some(true))?;
        let top_weights_flat = top_weights_flat.div(&sum)?;

        // Reshape back to [B, T, top_k].
        let out_shape = [batch, seq_len, top_k];
        let top_weights = top_weights_flat.reshape(&out_shape)?;
        let top_indices = top_indices_flat.reshape(&out_shape)?;

        Ok((top_weights, top_indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    fn router(num_experts: usize, hidden: usize, top_k: usize) -> TopKRouter {
        let config = RouterConfig {
            num_experts,
            hidden,
            top_k,
        };
        let weight = MxArray::random_normal(
            &[num_experts as i64, hidden as i64],
            0.0,
            0.02,
            Some(DType::Float32),
        )
        .expect("random_normal weight");
        let bias = MxArray::zeros(&[num_experts as i64], Some(DType::Float32)).expect("zeros bias");
        TopKRouter::new(config, weight, bias)
    }

    #[test]
    fn test_route_shapes() {
        let r = router(8, 4, 2);
        let hidden = MxArray::random_normal(&[2, 3, 4], 0.0, 1.0, Some(DType::Float32))
            .expect("random_normal hidden");
        let (weights, indices) = r.route(&hidden).expect("route");

        let w_shape: Vec<i64> = weights.shape().expect("weights shape").as_ref().to_vec();
        let i_shape: Vec<i64> = indices.shape().expect("indices shape").as_ref().to_vec();
        assert_eq!(w_shape, vec![2, 3, 2]);
        assert_eq!(i_shape, vec![2, 3, 2]);
    }

    #[test]
    fn test_route_renormalization_sums_to_one() {
        let num_experts = 8usize;
        let top_k = 2usize;
        let r = router(num_experts, 4, top_k);
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
                "row {} of weight sum is {} (expected 1.0 +/- 1e-4)",
                i,
                v
            );
        }
    }

    #[test]
    fn test_route_indices_in_range() {
        let num_experts = 8usize;
        let r = router(num_experts, 4, 2);
        let hidden = MxArray::random_normal(&[2, 3, 4], 0.0, 1.0, Some(DType::Float32))
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
}
