use crate::array::MxArray;
use crate::nn::{Activations, Linear};
use crate::transformer::MLP;
use napi::bindgen_prelude::*;

use super::config::Qwen3_5MoeConfig;
use super::quantized_linear::{LinearProj, MLPVariant, QuantizedLinear};
use super::switch_glu::SwitchGLU;

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
    pub fn new(config: &Qwen3_5MoeConfig) -> Result<Self> {
        let num_experts = config.num_experts;
        let num_experts_per_tok = config.num_experts_per_tok;

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

        let hidden_size = config.hidden_size;
        let moe_intermediate = config
            .moe_intermediate_size
            .unwrap_or(config.intermediate_size);
        let shared_expert_intermediate = config
            .shared_expert_intermediate_size
            .unwrap_or(config.intermediate_size);
        let norm_topk_prob = config.norm_topk_prob;

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

    pub fn new_quantized(
        config: &Qwen3_5MoeConfig,
        gate: QuantizedLinear,
        switch_mlp: SwitchGLU,
        shared_expert: MLPVariant,
        shared_expert_gate: QuantizedLinear,
    ) -> Result<Self> {
        let num_experts = config.num_experts;
        let num_experts_per_tok = config.num_experts_per_tok;
        let norm_topk_prob = config.norm_topk_prob;

        Ok(Self {
            gate: LinearProj::Quantized(gate),
            switch_mlp,
            shared_expert,
            shared_expert_gate: LinearProj::Quantized(shared_expert_gate),
            num_experts,
            num_experts_per_tok,
            norm_topk_prob,
        })
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
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
        let num_exp = self.num_experts as i64;

        let x_flat = x.reshape(&[ne, hidden_size])?;
        let router_logits = self.gate.forward(&x_flat)?;
        let routing_weights = Activations::softmax(&router_logits, Some(-1))?;

        let top_indices_full = routing_weights.argpartition(-(k as i32), Some(-1))?;
        let top_indices = top_indices_full.slice_axis(1, num_exp - k, num_exp)?;
        let top_weights = routing_weights.take_along_axis(&top_indices, -1)?;

        let top_weights = if self.norm_topk_prob {
            let sum = top_weights.sum(Some(&[-1]), Some(true))?;
            top_weights.div(&sum)?
        } else {
            top_weights
        };

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

    // ========== Weight accessors ==========

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
    ) {
        self.shared_expert = MLPVariant::Quantized {
            gate_proj,
            up_proj,
            down_proj,
        };
    }
}
