use crate::array::MxArray;
use crate::moe::{RoutingMode, SwitchGLU, topk_from_logits};
use crate::nn::{Activations, Linear};
use crate::transformer::MLP;
use napi::bindgen_prelude::*;

use super::config::{Qwen3_5Config, Qwen35MoeParams};
use crate::models::quantized_linear::{LinearProj, MLPVariant, QuantizedLinear};

/// SparseMoeBlock: Mixture-of-Experts block with shared expert.
pub struct SparseMoeBlock {
    gate: LinearProj,
    switch_mlp: SwitchGLU,
    shared_expert: MLPVariant,
    shared_expert_gate: LinearProj,
    num_experts: i32,
    num_experts_per_tok: i32,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    pub fn new(config: &crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig) -> Result<Self> {
        Self::build(
            config.hidden_size,
            config.intermediate_size,
            config.num_experts,
            config.num_experts_per_tok,
            config.moe_intermediate_size,
            config.shared_expert_intermediate_size,
            config.norm_topk_prob,
        )
    }

    pub(crate) fn new_with_params(base: &Qwen3_5Config, moe: &Qwen35MoeParams) -> Result<Self> {
        Self::build(
            base.hidden_size,
            base.intermediate_size,
            moe.num_experts,
            moe.num_experts_per_tok,
            moe.moe_intermediate_size,
            moe.shared_expert_intermediate_size,
            moe.norm_topk_prob,
        )
    }

    fn build(
        hidden_size: i32,
        intermediate_size: i32,
        num_experts: i32,
        num_experts_per_tok: i32,
        moe_intermediate_size: Option<i32>,
        shared_expert_intermediate_size: Option<i32>,
        norm_topk_prob: bool,
    ) -> Result<Self> {
        if num_experts <= 0 {
            return Err(Error::from_reason(format!(
                "SparseMoeBlock requires num_experts > 0, got {}",
                num_experts
            )));
        }
        if num_experts_per_tok <= 0 || num_experts_per_tok > num_experts {
            return Err(Error::from_reason(format!(
                "SparseMoeBlock requires 0 < num_experts_per_tok <= num_experts, got {} (num_experts={})",
                num_experts_per_tok, num_experts
            )));
        }

        let moe_intermediate = moe_intermediate_size.unwrap_or(intermediate_size);
        let shared_expert_intermediate =
            shared_expert_intermediate_size.unwrap_or(intermediate_size);

        let gate = Linear::new(hidden_size as u32, num_experts as u32, Some(false))?;
        let switch_mlp = SwitchGLU::new(
            hidden_size as u32,
            moe_intermediate as u32,
            num_experts as u32,
        )?;
        let shared_expert = MLP::new(hidden_size as u32, shared_expert_intermediate as u32)?;
        let shared_expert_gate = Linear::new(hidden_size as u32, 1, Some(false))?;

        Ok(Self {
            gate: LinearProj::Standard(gate),
            switch_mlp,
            shared_expert: MLPVariant::Standard(shared_expert),
            shared_expert_gate: LinearProj::Standard(shared_expert_gate),
            num_experts,
            num_experts_per_tok,
            norm_topk_prob,
        })
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        self.forward_with_budget(x, None)
    }

    /// The grouped path skips expert work outside the device-selected prefix.
    /// Generic projections retain full evaluation; acceptance still caps the
    /// committed prefix, while the physical verifier/cache shapes stay fixed.
    pub fn forward_with_budget(
        &self,
        x: &MxArray,
        row_budget: Option<&MxArray>,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        if shape.len() != 3 {
            return Err(Error::from_reason(format!(
                "SparseMoeBlock::forward expects 3D input [B, T, D], got {}D input",
                shape.len()
            )));
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let ne = batch * seq_len;
        let k = self.num_experts_per_tok as i64;

        let x_flat = x.reshape(&[ne, hidden_size])?;
        let router_logits = self.gate.forward(&x_flat)?;

        if self.num_experts_per_tok <= 8
            && self
                .switch_mlp
                .supports_grouped(&self.shared_expert, &x_flat)?
        {
            let shared_logits = self.shared_expert_gate.forward(&x_flat)?;
            return self
                .switch_mlp
                .forward_grouped(
                    &x_flat,
                    &router_logits,
                    &shared_logits,
                    &self.shared_expert,
                    self.num_experts_per_tok,
                    self.norm_topk_prob,
                    row_budget,
                )?
                .reshape(&[batch, seq_len, hidden_size]);
        }

        // Routing math (softmax-then-topk, optionally renormalized) is shared
        // with the gpt-oss / privacy-filter path via `moe::topk_from_logits`.
        // The gate matmul stays here because it may be quantized (8-bit affine) —
        // the shared `TopKRouter` only handles plain `MxArray` gate weights.
        let (top_weights, top_indices) = topk_from_logits(
            &router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            RoutingMode::Qwen35 {
                renormalize_topk: self.norm_topk_prob,
            },
        )?;

        let expert_out = self.switch_mlp.forward(&x_flat, &top_indices)?;
        let weights_expanded = top_weights.reshape(&[ne, k, 1])?;
        let weighted = expert_out.mul(&weights_expanded)?;
        let expert_output = weighted.sum(Some(&[1]), None)?;

        let shared_out = self.shared_expert.forward(&x_flat)?;
        let shared_gate = self.shared_expert_gate.forward(&x_flat)?;
        let shared_gate = Activations::sigmoid(&shared_gate)?;
        let shared_contribution = shared_out.mul(&shared_gate)?;

        let output = expert_output.add(&shared_contribution)?;
        output.reshape(&[batch, seq_len, hidden_size])
    }

    // ========== Weight getters (for training parameter extraction) ==========

    pub fn get_gate_weight(&self) -> MxArray {
        self.gate.get_weight()
    }

    pub fn get_switch_mlp(&self) -> &SwitchGLU {
        &self.switch_mlp
    }

    pub fn get_shared_expert_gate_proj_weight(&self) -> MxArray {
        self.shared_expert.get_gate_proj_weight()
    }

    pub fn get_shared_expert_up_proj_weight(&self) -> MxArray {
        self.shared_expert.get_up_proj_weight()
    }

    pub fn get_shared_expert_down_proj_weight(&self) -> MxArray {
        self.shared_expert.get_down_proj_weight()
    }

    pub fn get_shared_expert_gate_weight(&self) -> MxArray {
        self.shared_expert_gate.get_weight()
    }

    /// Whether any sub-projection (router gate, expert switch_mlp, shared
    /// expert, or shared-expert gate) holds quantized weights.
    ///
    /// Used by the dense/bf16-only MTP save path to refuse serializing a
    /// quantized MoE MTP layer's stale dense weights (see
    /// `Qwen3_5MTPModule::has_quantized_weights`).
    pub fn is_quantized(&self) -> bool {
        self.gate.is_quantized()
            || self.switch_mlp.is_quantized()
            || self.shared_expert.is_quantized()
            || self.shared_expert_gate.is_quantized()
    }

    // ========== Weight setters ==========

    pub fn set_gate_weight(&mut self, w: &MxArray) -> Result<()> {
        self.gate.set_weight(w, "gate")
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
        match &mut self.shared_expert {
            MLPVariant::Standard(mlp) => mlp.set_gate_proj_weight(w),
            MLPVariant::Quantized { .. } => Err(Error::from_reason(
                "Cannot set weight on quantized shared expert",
            )),
        }
    }
    pub fn set_shared_expert_up_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match &mut self.shared_expert {
            MLPVariant::Standard(mlp) => mlp.set_up_proj_weight(w),
            MLPVariant::Quantized { .. } => Err(Error::from_reason(
                "Cannot set weight on quantized shared expert",
            )),
        }
    }
    pub fn set_shared_expert_down_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match &mut self.shared_expert {
            MLPVariant::Standard(mlp) => mlp.set_down_proj_weight(w),
            MLPVariant::Quantized { .. } => Err(Error::from_reason(
                "Cannot set weight on quantized shared expert",
            )),
        }
    }
    pub fn set_shared_expert_gate_weight(&mut self, w: &MxArray) -> Result<()> {
        self.shared_expert_gate.set_weight(w, "shared_expert_gate")
    }

    #[cfg(test)]
    pub fn switch_mlp_mut(&mut self) -> &mut SwitchGLU {
        &mut self.switch_mlp
    }

    pub fn set_switch_mlp(&mut self, mlp: SwitchGLU) {
        self.switch_mlp = mlp;
    }

    pub fn set_quantized_gate(&mut self, gate: QuantizedLinear) {
        self.gate.set_quantized(gate);
    }

    pub fn set_quantized_shared_expert_gate(&mut self, gate: QuantizedLinear) {
        self.shared_expert_gate.set_quantized(gate);
    }

    pub fn set_quantized_shared_expert(
        &mut self,
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
    ) -> Result<()> {
        self.shared_expert = MLPVariant::Quantized {
            gate_proj,
            up_proj,
            down_proj,
            gate_up: None,
        };
        self.shared_expert.finalize_gate_up()
    }

    pub(crate) fn finalize_grouped_layout(&mut self) -> Result<bool> {
        if self.num_experts_per_tok > 8 {
            return Ok(false);
        }
        self.switch_mlp.tile_grouped_layout(&self.shared_expert)
    }
}
