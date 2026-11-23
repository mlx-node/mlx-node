use super::super::*;
use super::Banks;

impl Decoder {
    /// Resident shared-expert `[gate, up, down]` banks for layer `i` (GGUF).
    fn cuda_shared_banks(&self, i: usize) -> Option<Banks> {
        if !self.gguf() {
            return None;
        }
        let [Some(gate), Some(up), Some(down)] = ["gate", "up", "down"].map(|part| {
            self.weights
                .resident_bank(&format!("blk.{i}.ffn_{part}_shexp.weight"))
        }) else {
            return None;
        };
        Some([gate, up, down])
    }

    /// Fused CUDA routed + shared MoE for layer `i` on direct/resident banks:
    /// - `M in 1..=16`: `moe_routed_shared_few_cuda` (routed + shared in one primitive!)
    /// - `M >= 17`: `shared_expert_parts` + `moe_prefill_cuda` (routed + shared combine!)
    ///
    /// Returns `Ok(None)` when disabled or off-contract so callers fall back cleanly.
    pub(in super::super) fn cuda_fused_moe(
        &mut self,
        x: &MxArray,
        selected: &MxArray,
        scores: &MxArray,
        i: usize,
    ) -> Result<Option<MxArray>> {
        if !super::super::super::cuda_qwen4_fast_enabled() || !self.gguf() || !self.scope.is_empty()
        {
            return Ok(None);
        }
        let c = self.config.clone();
        if c.hidden_size != 2560
            || c.moe_intermediate_size != 640
            || c.num_experts_per_tok == 0
            || c.num_experts_per_tok > 10
            || x.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let (banks, local) = if let Some(banks) = self.resident_banks(i) {
            (banks, selected.reshape(&[-1])?)
        } else if self.weights.plan.direct {
            let Some((banks, local)) = self.weights.device_expert_slots(i, selected)? else {
                return Ok(None);
            };
            (banks, local)
        } else {
            return Ok(None);
        };
        let tokens = x.shape_at(1)?;
        if (1..=16).contains(&tokens) {
            if let Some(shared_banks) = self.cuda_shared_banks(i)
                && shared_banks
                    .iter()
                    .all(|w| w.mode == "affine" && w.bits == 8 && w.group == 32)
            {
                let shared_gate = self.linear_rows(
                    x,
                    &format!("layers.{i}.mlp.shared_expert_gate.weight"),
                    &format!("blk.{i}.ffn_gate_inp_shexp.weight"),
                )?;
                if let Some(out) = math::cuda_moe_routed_shared_few(
                    x,
                    &local,
                    scores,
                    &banks,
                    Some((&shared_banks, &shared_gate)),
                )? {
                    return Ok(Some(out));
                }
            }
            if let Some(routed) = math::cuda_moe_routed_shared_few(x, &local, scores, &banks, None)?
            {
                return self.with_shared_expert(x, &routed, i).map(Some);
            }
            return Ok(None);
        }
        let (shared_out, shared_gate) = self.shared_expert_parts(x, i)?;
        if let Some(out) =
            math::cuda_moe_prefill(x, &local, scores, &banks, Some((&shared_out, &shared_gate)))?
        {
            return Ok(Some(out));
        }
        Ok(None)
    }

    /// Non-Metal direct residency (Windows CUDA): wait for layer `i`'s
    /// direct fill if still running and run routed experts directly on
    /// the GPU-resident banks without host route readbacks.
    pub(super) fn cuda_direct_resident_banks(
        &mut self,
        i: usize,
        selected: &MxArray,
    ) -> Result<Option<(Banks, MxArray, usize)>> {
        if self.weights.plan.direct
            && !crate::engine::persistence::compiled_forward_backend_available()
        {
            let Some((banks, local)) = self.weights.device_expert_slots(i, selected)? else {
                return Ok(None);
            };
            return Ok(Some((banks, local, self.config.num_experts)));
        }
        Ok(None)
    }

    pub(super) fn cuda_resident_moe(
        x: &MxArray,
        ids: &MxArray,
        scores: &MxArray,
        banks: &Banks,
        tokens: i64,
        top: usize,
        h: usize,
    ) -> Result<Option<MxArray>> {
        if super::super::super::cuda_qwen4_fast_enabled() && top <= 10 && h == 2560 {
            let cuda_out = if (1..=16).contains(&tokens) {
                math::cuda_moe_routed_shared_few(x, ids, scores, banks, None)?
            } else {
                math::cuda_moe_prefill(x, ids, scores, banks, None)?
            };
            if let Some(out) = cuda_out {
                return Ok(Some(out));
            }
        }
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn cuda_gdn_batch(
        &mut self,
        prepared: Option<&[MxArray; 5]>,
        z: &MxArray,
        norm: &MxArray,
        state: &MxArray,
        t: i64,
        p: &str,
        g: &str,
        cache: &mut LayerCache,
    ) -> Result<Option<MxArray>> {
        let c = self.config.clone();
        if super::super::super::cuda_qwen4_fast_enabled()
            && let Some([q, k, v, decay, beta]) = prepared
            && c.rms_norm_eps.is_finite()
            && c.rms_norm_eps > 0.0
            && z.dtype()? == DType::BFloat16
        {
            let tiled_heads = self.gguf();
            let sigmoid_gate = c.output_gate_type == "sigmoid";
            let cuda_out = if t <= 16 {
                crate::kernels::gdn::fused_step_with_gate(
                    q,
                    k,
                    v,
                    decay,
                    beta,
                    state,
                    z,
                    norm,
                    c.rms_norm_eps as f32,
                    tiled_heads,
                    sigmoid_gate,
                    None,
                )?
            } else {
                crate::kernels::gdn::prefill_chain_gated_with_gate(
                    q,
                    k,
                    v,
                    decay,
                    beta,
                    state,
                    z,
                    norm,
                    c.rms_norm_eps as f32,
                    tiled_heads,
                    sigmoid_gate,
                )?
            };
            if let Some((out, next_state)) = cuda_out {
                self.submit_state(&[&next_state], "qwen4::batch::recurrent_state")?;
                cache.recurrent = Some(next_state);
                return self
                    .linear(
                        &out,
                        &format!("{p}.out_proj.weight"),
                        &format!("{g}.ssm_out.weight"),
                    )
                    .map(Some);
            }
        }
        Ok(None)
    }
}
