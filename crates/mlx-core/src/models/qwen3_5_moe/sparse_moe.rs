use crate::array::MxArray;
use crate::moe::{RoutingMode, topk_from_logits};
use crate::nn::{Activations, Linear};
use crate::transformer::MLP;
use napi::bindgen_prelude::*;

use super::config::Qwen3_5MoeConfig;
use super::switch_glu::SwitchGLU;
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
    source_residual_compatible: bool,
    source_moe_policy: super::source_math_policy::SourceMoePolicy,
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
            source_residual_compatible: source_residual_profile(config),
            source_moe_policy: Default::default(),
        })
    }

    pub(crate) fn configure_source_moe_policy(
        &mut self,
        policy: super::source_math_policy::SourceMoePolicy,
    ) {
        self.source_moe_policy = policy;
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
        if let Some((router, shared)) = self.source_router_weights(x)? {
            return self
                .switch_mlp
                .forward_grouped(
                    &x_flat,
                    &router,
                    &shared,
                    &self.shared_expert,
                    self.num_experts_per_tok,
                    self.norm_topk_prob,
                    row_budget,
                    true,
                    true,
                    None,
                )?
                .reshape(&[batch, seq_len, hidden_size]);
        }
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
                    self.source_moe_policy.body,
                    false,
                    None,
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

    /// Load-captured, M8-only arithmetic diagnostic. None preserves the
    /// ordinary FFN+residual path; Some is already the residual-inclusive hidden.
    pub(crate) fn forward_source_residual(
        &self,
        x: &MxArray,
        residual: &MxArray,
        row_budget: Option<&MxArray>,
    ) -> Result<Option<MxArray>> {
        if !self.source_moe_policy.residual
            || !self.source_residual_compatible
            || x.shape()?.as_ref() != [1, 8, 2048]
            || residual.shape()?.as_ref() != x.shape()?.as_ref()
            || x.dtype()? != crate::array::DType::BFloat16
            || residual.dtype()? != crate::array::DType::BFloat16
            || !unsafe { mlx_sys::mlx_splash_moe_source_residual_available() }
            || !self
                .switch_mlp
                .supports_source_residual(&self.shared_expert)?
        {
            return Ok(None);
        }
        let flat = x.reshape(&[8, 2048])?;
        let residual = residual.reshape(&[8, 2048])?;
        let source_weights = self.source_router_weights(x)?;
        let source_routing = source_weights.is_some();
        let (logits, shared) = match source_weights {
            Some(weights) => weights,
            None => (
                self.gate.forward(&flat)?,
                self.shared_expert_gate.forward(&flat)?,
            ),
        };
        if logits.dtype()? != crate::array::DType::BFloat16
            || shared.dtype()? != crate::array::DType::BFloat16
        {
            return Ok(None);
        }
        self.switch_mlp
            .forward_grouped(
                &flat,
                &logits,
                &shared,
                &self.shared_expert,
                self.num_experts_per_tok,
                self.norm_topk_prob,
                row_budget,
                self.source_moe_policy.body,
                source_routing,
                Some(&residual),
            )?
            .reshape(&[1, 8, 2048])
            .map(Some)
    }

    /// The copied source router needs an exact, unquantized BF16 matrix. The
    /// checkpoint's F32 router weights are exactly BF16-representable, so this
    /// widens the existing operands without a duplicate F32 weight allocation.
    /// Other shapes, biased/quantized gates, and devices keep native routing.
    fn source_router_weights(&self, x: &MxArray) -> Result<Option<(MxArray, MxArray)>> {
        if !self.source_moe_policy.routing
            || !self.source_residual_compatible
            || !self.norm_topk_prob
            || x.shape()?.as_ref() != [1, 8, 2048]
            || x.dtype()? != crate::array::DType::BFloat16
            || !unsafe { mlx_sys::mlx_splash_moe_source_residual_available() }
            || !self
                .switch_mlp
                .supports_source_residual(&self.shared_expert)?
        {
            return Ok(None);
        }
        source_router_weights_from_projections(&self.gate, &self.shared_expert_gate)
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
    /// `Qwen3_5MoeMTPModule::has_quantized_weights`).
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

pub(super) fn source_residual_profile(config: &Qwen3_5MoeConfig) -> bool {
    config.hidden_size == 2048
        && config.num_layers == 40
        && config.num_experts == 256
        && config.num_experts_per_tok == 8
        && config
            .moe_intermediate_size
            .unwrap_or(config.intermediate_size)
            == 512
        && config
            .shared_expert_intermediate_size
            .unwrap_or(config.intermediate_size)
            == 512
}

fn source_router_weights_from_projections(
    gate: &LinearProj,
    shared_gate: &LinearProj,
) -> Result<Option<(MxArray, MxArray)>> {
    let (LinearProj::Standard(gate), LinearProj::Standard(shared)) = (gate, shared_gate) else {
        return Ok(None);
    };
    if gate.is_quantized()
        || shared.is_quantized()
        || gate.get_bias().is_some()
        || shared.get_bias().is_some()
    {
        return Ok(None);
    }
    let router = gate.get_weight();
    let shared = shared.get_weight();
    if router.shape()?.as_ref() != [256, 2048]
        || shared.shape()?.as_ref() != [1, 2048]
        || router.dtype()? != crate::array::DType::BFloat16
        || shared.dtype()? != crate::array::DType::BFloat16
    {
        return Ok(None);
    }
    Ok(Some((router, shared)))
}

#[cfg(test)]
mod source_router_tests {
    use super::*;
    use crate::array::DType;

    fn gate(rows: u32, dtype: DType, bias: bool) -> Result<LinearProj> {
        let mut gate = Linear::new(2048, rows, Some(bias))?;
        gate.set_weight(&MxArray::zeros(&[rows as i64, 2048], Some(dtype))?)?;
        Ok(LinearProj::Standard(gate))
    }

    #[test]
    fn source_router_rejects_biased_wrong_shape_and_non_bf16_projections() -> Result<()> {
        let router = gate(256, DType::BFloat16, false)?;
        let shared = gate(1, DType::BFloat16, false)?;
        let original = source_router_weights_from_projections(&router, &shared)?.unwrap();
        assert_eq!(original.0.shape()?.as_ref(), [256, 2048]);
        assert_eq!(original.1.shape()?.as_ref(), [1, 2048]);
        for bad_router in [
            gate(255, DType::BFloat16, false)?,
            gate(256, DType::Float32, false)?,
            gate(256, DType::BFloat16, true)?,
        ] {
            assert!(source_router_weights_from_projections(&bad_router, &shared)?.is_none());
        }
        for bad_shared in [
            gate(2, DType::BFloat16, false)?,
            gate(1, DType::Float32, false)?,
            gate(1, DType::BFloat16, true)?,
        ] {
            assert!(source_router_weights_from_projections(&router, &bad_shared)?.is_none());
        }
        Ok(())
    }
}
