use crate::array::MxArray;
use crate::nn::{Activations, Linear};
use crate::transformer::MLP;
use napi::bindgen_prelude::*;

use super::config::Qwen3_5Config;
use super::switch_glu::SwitchGLU;

/// SparseMoeBlock: Mixture-of-Experts block with shared expert.
///
/// Routes tokens to top-k experts via learned gating, then adds
/// a dedicated shared expert (gated by sigmoid) to all tokens.
pub struct SparseMoeBlock {
    gate: Linear,              // hidden → num_experts (routing logits)
    switch_mlp: SwitchGLU,     // expert MLP with gather_mm
    shared_expert: MLP,        // dedicated shared expert
    shared_expert_gate: Linear, // hidden → 1 (sigmoid gating for shared expert)
    num_experts: i32,
    num_experts_per_tok: i32,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    pub fn new(config: &Qwen3_5Config) -> Result<Self> {
        let num_experts = config.num_experts.unwrap_or(0);
        let num_experts_per_tok = config.num_experts_per_tok.unwrap_or(1);
        let hidden_size = config.hidden_size;
        let moe_intermediate = config.moe_intermediate_size.unwrap_or(config.intermediate_size);
        let shared_expert_intermediate = config
            .shared_expert_intermediate_size
            .unwrap_or(config.intermediate_size);
        let norm_topk_prob = config.norm_topk_prob.unwrap_or(true);

        let gate = Linear::new(hidden_size as u32, num_experts as u32, Some(false))?;
        let switch_mlp = SwitchGLU::new(
            hidden_size as u32,
            moe_intermediate as u32,
            num_experts as u32,
        )?;
        let shared_expert = MLP::new(hidden_size as u32, shared_expert_intermediate as u32)?;
        let shared_expert_gate = Linear::new(hidden_size as u32, 1, Some(false))?;

        Ok(Self {
            gate,
            switch_mlp,
            shared_expert,
            shared_expert_gate,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    ///
    /// # Returns
    /// Output [B, T, hidden_size]
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let shape = x.shape()?;
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let ne = batch * seq_len;

        // Flatten: [B, T, D] → [B*T, D]
        let x_flat = x.reshape(&[ne, hidden_size])?;

        // Compute routing logits: [B*T, num_experts]
        let router_logits = self.gate.forward(&x_flat)?;

        // Softmax over experts: [B*T, num_experts]
        let routing_weights = Activations::softmax(&router_logits, Some(-1))?;

        // Top-k expert selection
        // argpartition to find top-k indices: [B*T, num_experts_per_tok]
        let k = self.num_experts_per_tok as i64;
        let neg_k = -k; // argpartition returns smallest, we want largest, so negate

        // Get top-k indices via argpartition on negative weights
        let neg_weights = routing_weights.negative()?;
        let top_indices_full = neg_weights.argpartition(neg_k as i32, Some(-1))?;
        // Take last k: [B*T, k]
        let num_exp = self.num_experts as i64;
        let top_indices = top_indices_full.slice_axis(1, num_exp - k, num_exp)?;

        // Gather top-k weights: [B*T, k]
        let top_weights = routing_weights.take_along_axis(&top_indices, -1)?;

        // Normalize weights if configured
        let top_weights = if self.norm_topk_prob {
            let sum = top_weights.sum(Some(&[-1]), Some(true))?;
            top_weights.div(&sum)?
        } else {
            top_weights
        };

        // Expand dims for SwitchGLU: [B*T, k, 1, 1]
        let indices_expanded = top_indices.reshape(&[ne, k, 1, 1])?;

        // For each expert slot, compute expert output and weight it
        let mut weighted_sum = MxArray::zeros(&[ne, hidden_size], None)?;

        for i in 0..k {
            // Get indices for this expert slot: [B*T, 1, 1]
            let slot_indices = indices_expanded.slice_axis(1, i, i + 1)?;
            let slot_indices = slot_indices.reshape(&[ne, 1, 1])?;

            // Get weights for this slot: [B*T, 1]
            let slot_weights = top_weights.slice_axis(1, i, i + 1)?;

            // Prepare input: [B*T, 1, D]
            let x_expanded = x_flat.reshape(&[ne, 1, hidden_size])?;

            // Expert forward: [B*T, 1, D]
            let expert_out = self.switch_mlp.forward(&x_expanded, &slot_indices)?;
            let expert_out = expert_out.reshape(&[ne, hidden_size])?;

            // Weight and accumulate
            let weighted = expert_out.mul(&slot_weights)?;
            weighted_sum = weighted_sum.add(&weighted)?;
        }

        // Shared expert contribution
        let shared_out = self.shared_expert.forward(&x_flat)?;
        let shared_gate = self.shared_expert_gate.forward(&x_flat)?;
        let shared_gate = Activations::sigmoid(&shared_gate)?;
        let shared_contribution = shared_out.mul(&shared_gate)?;

        // Combine: expert_output + shared_expert_output
        let output = weighted_sum.add(&shared_contribution)?;

        // Reshape back: [B*T, D] → [B, T, D]
        output.reshape(&[batch, seq_len, hidden_size])
    }

    // ========== Weight accessors ==========

    pub fn set_gate_weight(&mut self, w: &MxArray) -> Result<()> {
        self.gate.set_weight(w)
    }

    pub fn set_switch_mlp_gate_proj_weight(&mut self, w: &MxArray) {
        self.switch_mlp.set_gate_proj_weight(w);
    }
    pub fn set_switch_mlp_up_proj_weight(&mut self, w: &MxArray) {
        self.switch_mlp.set_up_proj_weight(w);
    }
    pub fn set_switch_mlp_down_proj_weight(&mut self, w: &MxArray) {
        self.switch_mlp.set_down_proj_weight(w);
    }

    pub fn set_shared_expert_gate_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.shared_expert.set_gate_proj_weight(w)
    }
    pub fn set_shared_expert_up_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.shared_expert.set_up_proj_weight(w)
    }
    pub fn set_shared_expert_down_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.shared_expert.set_down_proj_weight(w)
    }
    pub fn set_shared_expert_gate_weight(&mut self, w: &MxArray) -> Result<()> {
        self.shared_expert_gate.set_weight(w)
    }
}
