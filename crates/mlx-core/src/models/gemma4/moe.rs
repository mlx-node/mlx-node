use crate::array::MxArray;
use crate::nn::{Activations, Linear, RMSNorm};
use napi::bindgen_prelude::*;

/// Gemma4 MoE Router.
///
/// Computes expert routing logits from hidden states.
/// Forward: `proj(norm(x, no_weight) * root_size * scale)`
///
/// The norm is a scale-free RMSNorm (weight stays at ones, never loaded from checkpoint).
/// `scale` is a learnable vector [hidden_size] loaded from `router.scale`.
/// `root_size` = hidden_size^(-0.5), a constant scaling factor.
pub struct Gemma4Router {
    /// Scale-free RMSNorm (weight=ones, never loaded). Normalizes input before routing.
    norm: RMSNorm,
    /// Learnable scale vector [hidden_size], loaded from checkpoint.
    scale: MxArray,
    /// Pre-created scalar: hidden_size^(-0.5).
    root_size: MxArray,
    /// Projects to [num_experts] logits.
    proj: Linear,
}

impl Gemma4Router {
    pub fn new(hidden_size: u32, num_experts: u32, eps: f64) -> Result<Self> {
        let norm = RMSNorm::new(hidden_size, Some(eps))?;
        // Scale initialized to ones; will be loaded from checkpoint
        let scale = MxArray::ones(&[hidden_size as i64], None)?;
        let root_size = MxArray::scalar_float((hidden_size as f64).powf(-0.5))?;
        let proj = Linear::new(hidden_size, num_experts, Some(false))?;

        Ok(Self {
            norm,
            scale,
            root_size,
            proj,
        })
    }

    /// Compute router logits.
    ///
    /// Input: `x` [B*T, hidden_size] or [B, T, hidden_size]
    /// Output: logits [B*T, num_experts] or [B, T, num_experts]
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let normed = self.norm.forward(x)?;
        let root = self.root_size.astype(normed.dtype()?)?;
        let scaled = normed.mul(&root)?.mul(&self.scale)?;
        let logits = self.proj.forward(&scaled)?;
        // Cast to f32 for stable routing (matches vLLM's GateLinear out_dtype=f32)
        logits.astype(crate::array::DType::Float32)
    }

    // ========== Weight setters ==========

    pub fn set_scale(&mut self, w: &MxArray) -> Result<()> {
        self.scale = w.clone();
        Ok(())
    }

    pub fn set_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.proj.set_weight(w)
    }
}

/// Gemma4 MoE Experts block.
///
/// Uses fused gate_up_proj weights [num_experts, 2*moe_inter, hidden] and
/// down_proj weights [num_experts, hidden, moe_inter].
///
/// Routing uses a custom scheme different from standard top-k:
/// 1. topk_ids = topk(logits, k)
/// 2. probs = softmax(logits) over ALL experts
/// 3. indicator = one_hot(topk_ids, E).sum(dim=-2)  (binary mask)
/// 4. gate_weights = indicator * probs  (masked probabilities)
/// 5. gate_weights = gate_weights / sum(gate_weights)  (renormalize)
/// 6. topk_weights = gather(gate_weights, topk_ids)
/// 7. topk_weights *= per_expert_scale[topk_ids]
pub struct Gemma4MoE {
    /// Fused gate+up projection, stored TRANSPOSED: [num_experts, hidden, 2*moe_inter]
    gate_up_proj_t: MxArray,
    /// Down projection, stored TRANSPOSED: [num_experts, moe_inter, hidden]
    down_proj_t: MxArray,
    /// Per-expert scaling factors: [num_experts]
    per_expert_scale: MxArray,
    /// Pre-created scalar of top_k for floor_divide.
    k_scalar: MxArray,
    num_experts: i32,
    top_k: i32,
    moe_intermediate_size: i32,
}

impl Gemma4MoE {
    pub fn new(
        hidden_size: u32,
        num_experts: u32,
        top_k: u32,
        moe_intermediate_size: u32,
    ) -> Result<Self> {
        let fused_inter = (2 * moe_intermediate_size) as i64;
        // Store in transposed form: [E, hidden, 2*moe_inter] and [E, moe_inter, hidden]
        let gate_up_proj_t =
            MxArray::zeros(&[num_experts as i64, hidden_size as i64, fused_inter], None)?;
        let down_proj_t = MxArray::zeros(
            &[
                num_experts as i64,
                moe_intermediate_size as i64,
                hidden_size as i64,
            ],
            None,
        )?;
        let per_expert_scale = MxArray::ones(&[num_experts as i64], None)?;
        let k_scalar = MxArray::scalar_int(top_k as i32)?;

        Ok(Self {
            gate_up_proj_t,
            down_proj_t,
            per_expert_scale,
            k_scalar,
            num_experts: num_experts as i32,
            top_k: top_k as i32,
            moe_intermediate_size: moe_intermediate_size as i32,
        })
    }

    /// Forward pass: route tokens to top-k experts and compute weighted expert outputs.
    ///
    /// # Arguments
    /// * `x` - Input hidden states [B, T, hidden_size]
    /// * `router_logits` - Router output [B, T, num_experts]
    pub fn forward(&self, x: &MxArray, router_logits: &MxArray) -> Result<MxArray> {
        let shape = x.shape()?;
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let ne = batch * seq_len;
        let k = self.top_k as i64;
        let num_exp = self.num_experts as i64;
        let moe_inter = self.moe_intermediate_size as i64;
        let x_dtype = x.dtype()?;

        // Flatten to [ne, hidden_size] and [ne, num_experts]
        let x_flat = x.reshape(&[ne, hidden_size])?;
        let logits_flat = router_logits.reshape(&[ne, num_exp])?;

        // Step 1: Find top-k expert indices using argpartition
        // argpartition with kth=-k puts the k largest in the last k positions
        let top_indices_full = logits_flat.argpartition(-(k as i32), Some(-1))?;
        let top_indices = top_indices_full.slice_axis(1, num_exp - k, num_exp)?;

        // Step 2: Softmax over ALL experts (not just top-k)
        let probs = Activations::softmax_precise(&logits_flat, Some(-1))?;

        // Fused equivalent of one_hot->indicator->gather: directly select top-k probabilities.
        let top_weights = probs.take_along_axis(&top_indices, -1)?;
        // Renormalize with zero-division guard (matches vLLM: torch.where(renorm_factor > 0.0, renorm_factor, 1.0))
        let weight_sum = top_weights.sum(Some(&[-1]), Some(true))?;
        let eps = MxArray::full(&[1], Either::A(1e-9), Some(weight_sum.dtype()?))?;
        let weight_sum = weight_sum.maximum(&eps)?;
        let top_weights = top_weights.div(&weight_sum)?;

        // Step 7: Apply per-expert scaling
        let expert_scales = self
            .per_expert_scale
            .take(&top_indices.reshape(&[-1])?, 0)?;
        let expert_scales = expert_scales.reshape(&[ne, k])?;
        let top_weights = top_weights.mul(&expert_scales)?;

        // Sort by expert index for gather_mm efficiency
        let flat_indices = top_indices.reshape(&[-1])?;
        let order = flat_indices.argsort(Some(-1))?;
        let inv_order = order.argsort(Some(-1))?;
        let idx_sorted = flat_indices.take(&order, 0)?;

        // Expand x for expert selection: [ne, hidden] -> [ne, 1, hidden] (3D for gather_mm)
        let x_expanded = x_flat.reshape(&[ne, 1, hidden_size])?;
        let token_indices = order.floor_divide(&self.k_scalar)?;
        let x_sorted = x_expanded.take(&token_indices, 0)?;

        // gate_up = gather_mm(x_sorted, gate_up_proj_t, idx_sorted) -> [ne*k, 1, 2*moe_inter]
        let gate_up = x_sorted.gather_mm(&self.gate_up_proj_t, &idx_sorted, true)?;

        // Split gate and up along last axis (axis=2): each [ne*k, 1, moe_inter]
        let gate = gate_up.slice_axis(2, 0, moe_inter)?;
        let up = gate_up.slice_axis(2, moe_inter, 2 * moe_inter)?;

        // Apply exact GELU activation (vLLM uses activation="gelu" which maps to
        // F.gelu(x, approximate="none"), not the tanh approximation used by the dense MLP)
        let activated = Activations::gelu_exact(&gate)?;
        let hidden = activated.mul(&up)?;

        // down = gather_mm(hidden, down_proj_t, idx_sorted) -> [ne*k, 1, hidden]
        let down = hidden.gather_mm(&self.down_proj_t, &idx_sorted, true)?;

        // Unsort back to original order
        let down_unsorted = down.take(&inv_order, 0)?;
        // Reshape to [ne, k, hidden]
        let expert_out = down_unsorted.reshape(&[ne, k, hidden_size])?;

        // Apply routing weights: [ne, k, 1] * [ne, k, hidden]
        let weights_expanded = top_weights.reshape(&[ne, k, 1])?;
        let weights_expanded = weights_expanded.astype(x_dtype)?;
        let weighted = expert_out.mul(&weights_expanded)?;

        // Sum over experts: [ne, hidden]
        let output = weighted.sum(Some(&[1]), None)?;

        // Reshape back to [B, T, hidden]
        output.reshape(&[batch, seq_len, hidden_size])
    }

    // ========== Weight setters ==========

    pub fn set_gate_up_proj(&mut self, w: &MxArray) -> Result<()> {
        // Store transposed: [E, 2*moe_inter, hidden] -> [E, hidden, 2*moe_inter]
        self.gate_up_proj_t = w.transpose(Some(&[0, 2, 1]))?;
        Ok(())
    }

    pub fn set_down_proj(&mut self, w: &MxArray) -> Result<()> {
        // Store transposed: [E, hidden, moe_inter] -> [E, moe_inter, hidden]
        self.down_proj_t = w.transpose(Some(&[0, 2, 1]))?;
        Ok(())
    }

    pub fn set_per_expert_scale(&mut self, w: &MxArray) -> Result<()> {
        self.per_expert_scale = w.clone();
        Ok(())
    }
}
