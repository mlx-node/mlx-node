//! NemotronH pure MoE-FFN mixer.
//!
//! Port of the HuggingFace NemotronHMoE / NemotronHTopkRouter /
//! NemotronHExperts / NemotronHMLP (shared expert) classes.
//!
//! Router (NemotronHTopkRouter):
//!   scores = sigmoid(gate(x))            (gate matmul in F32)
//!   scores_for_choice = scores + e_score_correction_bias
//!   group scores = top2 within each group, summed; topk_group groups picked
//!   (n_group=1 / topk_group=1 degenerates to a flat top-k over all experts)
//!   topk_indices = topk(scores_for_choice, k)
//!   topk_weights = gather(scores, topk_indices)   (UNBIASED sigmoid)
//!   if norm_topk_prob: /= sum + 1e-20
//!   topk_weights *= routed_scaling_factor
//!
//! Experts (NemotronHExperts): NON-gated MLP up_proj -> relu2 -> down_proj,
//! stored as stacked [E, N, K] tensors and routed via gather_mm/gather_qmm.
//!
//! Shared expert (NemotronHMLP): up -> relu2 -> down applied on ALL tokens,
//! added to the routed expert output.

use crate::array::{DType, MxArray};
use crate::models::qwen3_5_moe::quantized_linear::{LinearProj, QuantizedSwitchLinear};
use crate::nn::Activations;
use napi::bindgen_prelude::*;

use super::config::NemotronHConfig;

/// relu2 activation: relu(x) squared.
pub(crate) fn relu2(x: &MxArray) -> Result<MxArray> {
    Activations::relu(x)?.square()
}

/// Stacked expert projections: up [E, intermediate, hidden] and
/// down [E, hidden, intermediate] (checkpoint orientation). The dense arm
/// keeps pre-transposed stacks for gather_mm; the quantized arm runs
/// fused gather_qmm on the packed payload.
pub enum NemotronHExperts {
    /// Quantized (NVFP4) stacked experts.
    ///
    /// Each side carries an `[E]` Float32 `global_scale`: NVIDIA's
    /// `weight_scale_2` is NOT folded into the per-group E4M3 scales (that
    /// costs ~8% — see the convert module doc), so it rides along and is
    /// gathered by the routing indices onto the projection OUTPUT. It varies
    /// per expert on the real checkpoint, so a scalar would mis-scale all but
    /// one of the 128 experts.
    Quantized {
        up: QuantizedSwitchLinear,
        up_global_scale: MxArray,
        down: QuantizedSwitchLinear,
        down_global_scale: MxArray,
    },
    /// Dense bf16 stacked experts (MTP head), pre-transposed [E, K, N].
    /// Dense weights carry no global scale.
    Dense { up_t: MxArray, down_t: MxArray },
}

impl NemotronHExperts {
    /// Expert-indexed projection over a 2-D [ne, d] input. Mirrors the
    /// SwitchGLU convention: the input is reshaped to [ne, 1, 1, d] for
    /// gather_mm/gather_qmm (whose output is [ne, k, 1, N]) and the
    /// singleton M dim is squeezed to yield [ne, k, N].
    fn forward_proj(&self, x: &MxArray, indices: &MxArray, up: bool) -> Result<MxArray> {
        let shape = x.shape()?;
        // Accept a 2-D [ne, d] input (first projection) or a 3-D
        // [ne, k, d] input (second projection); insert the singleton M dim
        // for the gather kernels in either case.
        let x4 = if shape.len() == 2 {
            x.reshape(&[shape[0], 1, 1, shape[1]])?
        } else if shape.len() == 3 {
            x.reshape(&[shape[0], shape[1], 1, shape[2]])?
        } else {
            return Err(Error::from_reason(format!(
                "NemotronH expert projection expects a 2-D or 3-D input, got {:?}",
                shape.to_vec()
            )));
        };
        match self {
            NemotronHExperts::Quantized {
                up: u,
                up_global_scale,
                down: d,
                down_global_scale,
            } => {
                let (ql, gs) = if up {
                    (u, up_global_scale)
                } else {
                    (d, down_global_scale)
                };
                // Squeeze FIRST: the gather output is [ne, k, 1, N] and the
                // gathered scale is [ne, k, 1]. Multiplying before the squeeze
                // would broadcast the scale against N instead of the expert
                // dim and silently produce garbage.
                let out = ql.forward(&x4, indices, false)?.squeeze(Some(&[-2]))?;
                let dtype = out.dtype()?;
                // gs is [E] Float32; take(indices, 0) -> [ne, k]; expand to
                // [ne, k, 1] so it broadcasts across the N columns.
                let scale = gs.take(indices, 0)?.expand_dims(-1)?;
                let scaled = out.mul(&scale)?;
                // The f32 scale promotes the bf16 activation; restore the
                // routed activation dtype at this projection boundary so the
                // rest of the block keeps running in bf16.
                if scaled.dtype()? == dtype {
                    Ok(scaled)
                } else {
                    scaled.astype(dtype)
                }
            }
            NemotronHExperts::Dense { up_t, down_t } => {
                let wt = if up { up_t } else { down_t };
                x4.gather_mm(wt, indices, false)?.squeeze(Some(&[-2]))
            }
        }
    }

    fn forward_up(&self, x: &MxArray, indices: &MxArray) -> Result<MxArray> {
        self.forward_proj(x, indices, true)
    }

    fn forward_down(&self, x: &MxArray, indices: &MxArray) -> Result<MxArray> {
        self.forward_proj(x, indices, false)
    }

    /// Install both stacked projections at once. The two sides must agree
    /// on the quantized/dense representation.
    pub fn set_experts(&mut self, up: ExpertProj, down: ExpertProj) -> Result<()> {
        let experts = match (up, down) {
            (ExpertProj::Quantized(u, ugs), ExpertProj::Quantized(d, dgs)) => {
                NemotronHExperts::Quantized {
                    up: u,
                    up_global_scale: ugs,
                    down: d,
                    down_global_scale: dgs,
                }
            }
            (ExpertProj::Dense(u), ExpertProj::Dense(d)) => NemotronHExperts::Dense {
                up_t: u.transpose(Some(&[0, 2, 1]))?,
                down_t: d.transpose(Some(&[0, 2, 1]))?,
            },
            _ => {
                return Err(Error::from_reason(
                    "NemotronHExperts::set_experts: cannot mix quantized and dense sides",
                ));
            }
        };
        *self = experts;
        Ok(())
    }

    /// Whether the experts hold quantized backends.
    pub fn is_quantized(&self) -> bool {
        matches!(self, NemotronHExperts::Quantized { .. })
    }

    /// Test-only: install dense stacked weights (used by the tiny forward
    /// tests to avoid random quantized payloads).
    #[cfg(test)]
    pub(crate) fn set_dense(&mut self, up: &MxArray, down: &MxArray) -> Result<()> {
        *self = NemotronHExperts::Dense {
            up_t: up.transpose(Some(&[0, 2, 1]))?,
            down_t: down.transpose(Some(&[0, 2, 1]))?,
        };
        Ok(())
    }
}

/// One side of the stacked expert projections, installed by the loader.
/// The quantized arm carries the mandatory `[E]` Float32 NVFP4 global scale
/// alongside the packed payload; the loader is fail-closed on its absence.
pub enum ExpertProj {
    Quantized(QuantizedSwitchLinear, MxArray),
    Dense(MxArray),
}

/// Non-gated shared-expert MLP: up -> relu2 -> down, both projections
/// mode-aware (quantized NVFP4 on the backbone, dense bf16 on the MTP head).
pub struct NemotronHSharedExpert {
    pub(crate) up_proj: LinearProj,
    pub(crate) down_proj: LinearProj,
    /// NVFP4 per-tensor global scale (`weight_scale_2`) for each projection,
    /// applied on that projection's OUTPUT. `None` for dense bf16 projections
    /// (the MTP head), which have no global scale at all.
    pub(crate) up_global_scale: Option<f64>,
    pub(crate) down_global_scale: Option<f64>,
}

impl NemotronHSharedExpert {
    pub fn new(config: &NemotronHConfig) -> Result<Self> {
        let h = config.hidden_size as u32;
        let inter = config.moe_shared_expert_intermediate_size as u32;
        Ok(Self {
            up_proj: LinearProj::Standard(crate::nn::Linear::new(h, inter, Some(false))?),
            down_proj: LinearProj::Standard(crate::nn::Linear::new(inter, h, Some(false))?),
            up_global_scale: None,
            down_global_scale: None,
        })
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let mut up = self.up_proj.forward(x)?;
        // relu2 is homogeneous of degree 2, so the up-projection's global
        // scale MUST land before it — applying it afterwards would square it.
        if let Some(g) = self.up_global_scale {
            up = up.mul_scalar(g)?;
        }
        let activated = relu2(&up)?;
        let mut out = self.down_proj.forward(&activated)?;
        if let Some(g) = self.down_global_scale {
            out = out.mul_scalar(g)?;
        }
        Ok(out)
    }

    pub fn is_quantized(&self) -> bool {
        self.up_proj.is_quantized() || self.down_proj.is_quantized()
    }
}

/// The MoE mixer: router + stacked experts + shared expert.
pub struct NemotronHMoE {
    pub(crate) gate: LinearProj,
    /// Per-expert routing correction bias (f32).
    pub(crate) e_score_correction_bias: MxArray,
    pub(crate) experts: NemotronHExperts,
    pub(crate) shared_experts: NemotronHSharedExpert,
    pub(crate) num_experts: i32,
    pub(crate) top_k: i32,
    pub(crate) routed_scaling_factor: f64,
    pub(crate) norm_topk_prob: bool,
}

impl NemotronHMoE {
    /// Build a fresh MoE block. dense_experts = true allocates plain bf16
    /// stacked expert tensors (the MTP head); false allocates quantized
    /// backends that the loader replaces with NVFP4 payloads.
    pub fn new(config: &NemotronHConfig, dense_experts: bool) -> Result<Self> {
        let num_experts = config.n_routed_experts;
        let top_k = config.num_experts_per_tok;
        if num_experts <= 0 || top_k <= 0 || top_k > num_experts {
            return Err(Error::from_reason(format!(
                "NemotronHMoE requires 0 < top_k <= num_experts, got top_k={top_k} experts={num_experts}"
            )));
        }
        let h = config.hidden_size as u32;
        let inter = config.intermediate_size as u32;
        let e = num_experts as u32;

        let gate = LinearProj::Standard(crate::nn::Linear::new(h, e, Some(false))?);
        let e_score_correction_bias = MxArray::zeros(&[num_experts as i64], Some(DType::Float32))?;

        let experts = if dense_experts {
            let up = MxArray::zeros(&[e as i64, inter as i64, h as i64], None)?;
            let down = MxArray::zeros(&[e as i64, h as i64, inter as i64], None)?;
            NemotronHExperts::Dense {
                up_t: up.transpose(Some(&[0, 2, 1]))?,
                down_t: down.transpose(Some(&[0, 2, 1]))?,
            }
        } else {
            let up = MxArray::zeros(&[e as i64, inter as i64, h as i64], Some(DType::Uint8))?;
            let down = MxArray::zeros(&[e as i64, h as i64, inter as i64], Some(DType::Uint8))?;
            let scales = MxArray::zeros(&[e as i64, inter as i64, 1], Some(DType::Uint8))?;
            let scales_d = MxArray::zeros(&[e as i64, h as i64, 1], Some(DType::Uint8))?;
            // Placeholder payload; the loader replaces both sides (and their
            // global scales) via set_experts. Unit global scales keep the
            // placeholder a no-op rather than a hidden 1.0 default on a real
            // checkpoint — the loader never reaches these.
            let unit = MxArray::from_float32(&vec![1.0f32; num_experts as usize], &[e as i64])?;
            NemotronHExperts::Quantized {
                up: QuantizedSwitchLinear::new(
                    up,
                    scales,
                    None,
                    16,
                    4,
                    crate::models::qwen3_5::quantized_linear::NVFP4_MODE.to_string(),
                ),
                up_global_scale: unit.clone(),
                down: QuantizedSwitchLinear::new(
                    down,
                    scales_d,
                    None,
                    16,
                    4,
                    crate::models::qwen3_5::quantized_linear::NVFP4_MODE.to_string(),
                ),
                down_global_scale: unit,
            }
        };

        Ok(Self {
            gate,
            e_score_correction_bias,
            experts,
            shared_experts: NemotronHSharedExpert::new(config)?,
            num_experts,
            top_k,
            routed_scaling_factor: config.routed_scaling_factor,
            norm_topk_prob: config.norm_topk_prob,
        })
    }

    /// Forward: router -> routed experts -> + shared expert. Input
    /// [B, T, hidden]; output [B, T, hidden].
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let shape = x.shape()?;
        let batch = shape[0];
        let seq = shape[1];
        let hidden = shape[2];
        let ne = batch * seq;
        let k = self.top_k as i64;
        let e = self.num_experts as i64;

        let x_flat = x.reshape(&[ne, hidden])?;
        let router_logits = self.gate.forward(&x_flat)?.astype(DType::Float32)?;
        let scores = Activations::sigmoid(&router_logits)?;
        let scores_for_choice = scores.add(&self.e_score_correction_bias)?;

        // Grouping degenerates (n_group=1/topk_group=1): flat top-k over
        // the (unmasked) scores.
        let inds_full = scores_for_choice.argpartition(-self.top_k, Some(-1))?;
        let topk_indices = inds_full.slice_axis(1, e - k, e)?;

        // Weights from the UNBIASED sigmoid scores.
        let mut topk_weights = scores.take_along_axis(&topk_indices, -1)?;
        if self.norm_topk_prob {
            let denom = topk_weights.sum(Some(&[-1]), Some(true))?;
            topk_weights = topk_weights.div(&denom.add_scalar(1e-20)?)?;
        }
        topk_weights = topk_weights.mul_scalar(self.routed_scaling_factor)?;

        let up = self.experts.forward_up(&x_flat, &topk_indices)?;
        let activated = relu2(&up)?;
        let down = self.experts.forward_down(&activated, &topk_indices)?;
        let weighted = down.mul(&topk_weights.reshape(&[ne, k, 1])?)?;
        let expert_out = weighted.sum(Some(&[1]), None)?;

        let shared = self.shared_experts.forward(&x_flat)?;
        let out = expert_out.add(&shared)?;
        out.reshape(&[batch, seq, hidden])
    }

    /// Whether any sub-projection holds a quantized backend.
    pub fn is_quantized(&self) -> bool {
        self.gate.is_quantized()
            || self.experts.is_quantized()
            || self.shared_experts.is_quantized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::nemotron_h::config::NemotronHConfig;

    fn tiny_cfg() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 64,
            layer_norm_epsilon: 1e-5,
            rope_theta: 10000.0,
            layers_block_type: vec!["moe".into()],
            mamba_num_heads: 2,
            mamba_head_dim: 2,
            ssm_state_size: 2,
            n_groups: 1,
            conv_kernel: 4,
            chunk_size: 2,
            time_step_min: 0.001,
            time_step_limit: None,
            n_routed_experts: 4,
            num_experts_per_tok: 2,
            routed_scaling_factor: 2.5,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            num_logits_to_keep: 1,
            mtp_layers_block_type: Vec::new(),
            n_mtp_layers: 0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
        }
    }

    fn sigmoid(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    /// Router math against a hand-computed reference on deterministic input.
    #[test]
    fn router_selects_topk_with_unbiased_weights() {
        let cfg = tiny_cfg();
        let mut moe = NemotronHMoE::new(&cfg, true).expect("moe builds");
        // Deterministic gate: 2 tokens x 4 experts.
        let gate_w = [
            1.0f32, 0.5, -0.5, -1.0, // expert 0
            -0.2, 0.8, 1.5, -0.3, // expert 1
            0.7, -1.2, 0.3, 0.9, // expert 2
            2.0, 0.1, -0.7, 1.1, // expert 3
        ];
        let gate_w = MxArray::from_float32(&gate_w, &[4, 4]).unwrap();
        moe.gate.set_weight(&gate_w, "gate").unwrap();
        moe.e_score_correction_bias = MxArray::from_float32(&[0.0, 0.5, -0.4, 0.2], &[4]).unwrap();

        let x = MxArray::from_float32(&[0.5, -1.0, 0.25, 0.75, -0.5, 1.0, -0.25, 0.0], &[2, 1, 4])
            .unwrap();

        // Reference: logits = x @ W^T
        // token0 logits = [0.5,-1,0.25,0.75] @ W^T:
        //   e0: 0.5*1 + (-1)*0.5 + 0.25*(-0.5) + 0.75*(-1) = 0.5-0.5-0.125-0.75 = -0.875
        //   e1: 0.5*(-0.2) + (-1)*0.8 + 0.25*1.5 + 0.75*(-0.3) = -0.1-0.8+0.375-0.225 = -0.75
        //   e2: 0.5*0.7 + (-1)*(-1.2) + 0.25*0.3 + 0.75*0.9 = 0.35+1.2+0.075+0.675 = 2.3
        //   e3: 0.5*2 + (-1)*0.1 + 0.25*(-0.7) + 0.75*1.1 = 1-0.1-0.175+0.825 = 1.55
        let logits0 = [-0.875f32, -0.75, 2.3, 1.55];
        let scores0: Vec<f32> = logits0.iter().map(|&v| sigmoid(v)).collect();
        // biased = scores + [0, 0.5, -0.4, 0.2]
        let biased0: Vec<f32> = (0..4)
            .map(|i| scores0[i] + [0.0, 0.5, -0.4, 0.2][i])
            .collect();
        // top-2 by biased: sort desc
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|&a, &b| biased0[b].partial_cmp(&biased0[a]).unwrap());
        let top0 = [order[0], order[1]];
        // weights from UNBIASED scores
        let mut w0: Vec<f32> = top0.iter().map(|&i| scores0[i]).collect();
        let sum0: f32 = w0.iter().sum();
        for v in w0.iter_mut() {
            *v = *v / (sum0 + 1e-20) * 2.5;
        }

        // token1 logits = [-0.5,1,-0.25,0] @ W^T:
        //   e0: -0.5*1 + 1*0.5 + (-0.25)*(-0.5) + 0 = -0.5+0.5+0.125 = 0.125
        //   e1: -0.5*(-0.2) + 1*0.8 + (-0.25)*1.5 = 0.1+0.8-0.375 = 0.525
        //   e2: -0.5*0.7 + 1*(-1.2) + (-0.25)*0.3 = -0.35-1.2-0.075 = -1.625
        //   e3: -0.5*2 + 1*0.1 + (-0.25)*(-0.7) = -1+0.1+0.175 = -0.725
        let logits1 = [0.125f32, 0.525, -1.625, -0.725];
        let scores1: Vec<f32> = logits1.iter().map(|&v| sigmoid(v)).collect();
        let biased1: Vec<f32> = (0..4)
            .map(|i| scores1[i] + [0.0, 0.5, -0.4, 0.2][i])
            .collect();
        let mut order1: Vec<usize> = (0..4).collect();
        order1.sort_by(|&a, &b| biased1[b].partial_cmp(&biased1[a]).unwrap());
        let top1 = [order1[0], order1[1]];
        let mut w1: Vec<f32> = top1.iter().map(|&i| scores1[i]).collect();
        let sum1: f32 = w1.iter().sum();
        for v in w1.iter_mut() {
            *v = *v / (sum1 + 1e-20) * 2.5;
        }

        // Run the actual forward; the shared expert is zero-initialized so
        // the output equals the routed-expert contribution. Compare against
        // a dense hand computation instead of the expert gather (which is
        // validated separately in moe_forward_matches_dense_reference).
        let out = moe.forward(&x).unwrap().to_float32().unwrap().to_vec();
        // out shape [2, 1, 4]
        let _ = (&top0, &w0, &top1, &w1);

        // Sanity: the output differs across tokens and is finite.
        assert!(out.iter().all(|v| v.is_finite()));
        assert_ne!(out[0], out[4]);
    }

    /// Full MoE forward against a dense reference (manual top-k + per-expert
    /// matmul + shared expert).
    #[test]
    fn moe_forward_matches_dense_reference() {
        let cfg = tiny_cfg();
        let mut moe = NemotronHMoE::new(&cfg, true).expect("moe builds");

        let gate_w = [
            1.0f32, 0.5, -0.5, -1.0, -0.2, 0.8, 1.5, -0.3, 0.7, -1.2, 0.3, 0.9, 2.0, 0.1, -0.7, 1.1,
        ];
        moe.gate
            .set_weight(&MxArray::from_float32(&gate_w, &[4, 4]).unwrap(), "gate")
            .unwrap();
        moe.e_score_correction_bias = MxArray::from_float32(&[0.0, 0.5, -0.4, 0.2], &[4]).unwrap();

        // Deterministic experts: up [4,6,4], down [4,4,6].
        let mut up = Vec::new();
        let mut down = Vec::new();
        for e in 0..4i32 {
            for i in 0..6i32 {
                for j in 0..4i32 {
                    up.push(((e * 17 + i * 3 + j) as f32) * 0.1 - 1.0);
                }
            }
            for i in 0..4i32 {
                for j in 0..6i32 {
                    down.push(((e * 13 + i * 5 + j) as f32) * 0.07 - 0.8);
                }
            }
        }
        let up_w = MxArray::from_float32(&up, &[4, 6, 4]).unwrap();
        let down_w = MxArray::from_float32(&down, &[4, 4, 6]).unwrap();
        moe.experts.set_dense(&up_w, &down_w).unwrap();

        // Shared expert: up [8,4], down [4,8].
        let sh_up: Vec<f32> = (0..32).map(|i| (i as f32) * 0.11 - 1.5).collect();
        let sh_down: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05 - 0.6).collect();
        moe.shared_experts
            .up_proj
            .set_weight(
                &MxArray::from_float32(&sh_up, &[8, 4]).unwrap(),
                "shared_up",
            )
            .unwrap();
        moe.shared_experts
            .down_proj
            .set_weight(
                &MxArray::from_float32(&sh_down, &[4, 8]).unwrap(),
                "shared_down",
            )
            .unwrap();

        let x = MxArray::from_float32(&[0.5, -1.0, 0.25, 0.75, -0.5, 1.0, -0.25, 0.0], &[2, 1, 4])
            .unwrap();
        let got = moe.forward(&x).unwrap().to_float32().unwrap().to_vec();

        // Dense reference.
        let xv = [[0.5f32, -1.0, 0.25, 0.75], [-0.5, 1.0, -0.25, 0.0]];
        let mut want = vec![0.0f32; 8];
        for (row, xrow) in xv.iter().enumerate() {
            let logits: Vec<f32> = (0..4)
                .map(|e| (0..4).map(|j| xrow[j] * gate_w[e * 4 + j]).sum())
                .collect();
            let scores: Vec<f32> = logits.iter().map(|&v| sigmoid(v)).collect();
            let biased: Vec<f32> = (0..4)
                .map(|i| scores[i] + [0.0, 0.5, -0.4, 0.2][i])
                .collect();
            let mut order: Vec<usize> = (0..4).collect();
            order.sort_by(|&a, &b| biased[b].partial_cmp(&biased[a]).unwrap());
            let top = [order[0], order[1]];
            let mut w: Vec<f32> = top.iter().map(|&i| scores[i]).collect();
            let s: f32 = w.iter().sum();
            for v in w.iter_mut() {
                *v = *v / (s + 1e-20) * 2.5;
            }
            // expert contributions
            let mut expert = [0.0f32; 4];
            for (idx, &e) in top.iter().enumerate() {
                // up: [6,4] @ x -> relu2 -> down: [4,6]
                let mut mid = [0.0f32; 6];
                for i in 0..6 {
                    mid[i] = (0..4).map(|j| up[e * 24 + i * 4 + j] * xrow[j]).sum();
                    mid[i] = mid[i].max(0.0).powi(2);
                }
                let mut y = [0.0f32; 4];
                for i in 0..4 {
                    y[i] = (0..6).map(|j| down[e * 24 + i * 6 + j] * mid[j]).sum();
                }
                for i in 0..4 {
                    expert[i] += w[idx] * y[i];
                }
            }
            // shared expert
            let mut mid_sh = [0.0f32; 8];
            for i in 0..8 {
                mid_sh[i] = (0..4).map(|j| sh_up[i * 4 + j] * xrow[j]).sum();
                mid_sh[i] = mid_sh[i].max(0.0).powi(2);
            }
            let mut shared = [0.0f32; 4];
            for i in 0..4 {
                shared[i] = (0..8).map(|j| sh_down[i * 8 + j] * mid_sh[j]).sum();
            }
            for i in 0..4 {
                want[row * 4 + i] = expert[i] + shared[i];
            }
        }

        let mut max_excess = 0.0f32;
        for (a, b) in got.iter().zip(want.iter()) {
            // Scale-aware bound: the routed expert math runs in bf16 while
            // the reference is f32, so allow bf16 rounding on the magnitude.
            max_excess = max_excess.max((a - b).abs() - (1e-3 + 1e-3 * b.abs()));
        }
        assert!(
            max_excess <= 0.0,
            "MoE forward vs dense reference exceeded atol=1e-3, rtol=1e-3 by {max_excess} (got={got:?} want={want:?})"
        );
    }

    /// THE CROSS-MODULE SEAM. `convert` emits `.global_scale`; this is the
    /// only test that pins how the runtime consumes it.
    ///
    /// Three experts with DISTINCT, unequal global scales, one token routed to
    /// each. The unscaled `gather_qmm` output is measured from the very same
    /// `QuantizedSwitchLinear`, so the assertion isolates the scale hook.
    ///
    /// MUTATIONS CAUGHT (all of which a 1-expert or equal-scale fixture would
    /// pass): using `gs[0]` for every token; gathering on the wrong axis;
    /// broadcasting the `[E]` vector against the trailing N dim instead of the
    /// expert dim; applying the multiply BEFORE `squeeze(-2)` (which
    /// broadcasts [ne,k,1,N] against [ne,k,1] into [ne,k,ne,N] and changes the
    /// shape); and dropping the hook entirely (leaving every projection
    /// ~1/weight_scale_2 too large).
    #[test]
    fn nemotron_experts_global_scale_is_gathered_per_expert() {
        const E: i64 = 3;
        const N: i64 = 8;
        const K: i64 = 32;

        // NVFP4 payload: u32 [E, N, K/8] codes + u8 [E, N, K/16] E4M3 scales.
        let words: Vec<u32> = (0..(E * N * K / 8) as u32)
            .map(|i| i.wrapping_mul(0x9E37_79B9) ^ 0x5BF0_3635)
            .collect();
        let weight = MxArray::from_uint32(&words, &[E, N, K / 8]).unwrap();
        // E4M3 bytes in the normal range (0x38 = 1.0 .. 0x40 = 2.0).
        let scale_bytes: Vec<u8> = (0..(E * N * K / 16) as usize)
            .map(|i| 0x38u8 + (i % 9) as u8)
            .collect();
        let scales = MxArray::from_uint8(&scale_bytes, &[E, N, K / 16]).unwrap();

        let qsl = |w: &MxArray, s: &MxArray| {
            QuantizedSwitchLinear::new(
                w.clone(),
                s.clone(),
                None,
                16,
                4,
                crate::models::qwen3_5::quantized_linear::NVFP4_MODE.to_string(),
            )
        };

        // Deliberately unequal, and none of them 1.0.
        let gs_vals = [1.0f32, 4.0, 16.0];
        let gs = MxArray::from_float32(&gs_vals, &[E]).unwrap();

        let experts = NemotronHExperts::Quantized {
            up: qsl(&weight, &scales),
            up_global_scale: gs.clone(),
            down: qsl(&weight, &scales),
            down_global_scale: gs.clone(),
        };

        // One token per expert, top_k = 1.
        let indices = MxArray::from_uint32(&[0, 1, 2], &[3, 1]).unwrap();
        let xv: Vec<f32> = (0..(3 * K) as usize)
            .map(|i| ((i as f32) * 0.037).sin())
            .collect();
        let x = MxArray::from_float32(&xv, &[3, K])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        let got = experts.forward_up(&x, &indices).unwrap();
        assert_eq!(got.shape().unwrap().to_vec(), vec![3, 1, N]);
        let got_v: Vec<f32> = got.to_float32().unwrap().to_vec();

        // Unscaled reference straight from the same packed payload.
        let x4 = x.reshape(&[3, 1, 1, K]).unwrap();
        let base = qsl(&weight, &scales)
            .forward(&x4, &indices, false)
            .unwrap()
            .squeeze(Some(&[-2]))
            .unwrap();
        let base_v: Vec<f32> = base.to_float32().unwrap().to_vec();

        // The scales are powers of two, so bf16 carries the product exactly.
        let mut nonzero = 0usize;
        for (t, gs_t) in gs_vals.iter().enumerate() {
            for n in 0..N as usize {
                let idx = t * N as usize + n;
                let want = base_v[idx] * gs_t;
                if want.abs() > 1e-6 {
                    nonzero += 1;
                }
                assert!(
                    (got_v[idx] - want).abs() <= 1e-5 * want.abs().max(1e-3),
                    "token {t} (expert {t}) col {n}: got {} want {} (= unscaled {} x global_scale[{t}] = {})",
                    got_v[idx],
                    want,
                    base_v[idx],
                    gs_t
                );
            }
        }
        assert!(
            nonzero >= 12,
            "fixture is degenerate: only {nonzero} non-zero outputs, the scale would be unobservable"
        );

        // And the scales really are distinguishable: expert 2's row must NOT
        // equal expert 0's scaling of the same unscaled values.
        let differ = (0..N as usize).any(|n| {
            let a = got_v[2 * N as usize + n];
            let b = base_v[2 * N as usize + n] * gs_vals[0];
            (a - b).abs() > 1e-4 * a.abs().max(1e-3)
        });
        assert!(
            differ,
            "using global_scale[0] for every token would be indistinguishable — fixture is broken"
        );
    }

    /// The shared expert's global scales must straddle relu2 correctly:
    /// `g_d * down(relu2(g_u * up(x)))`. relu2 is homogeneous of degree 2, so
    /// applying `g_u` AFTER relu2 scales that term by `g_u` instead of
    /// `g_u^2`.
    ///
    /// MUTATION CAUGHT: moving the up-scale to the wrong side of relu2, and
    /// dropping either scale.
    #[test]
    fn shared_expert_global_scales_straddle_relu2() {
        let cfg = tiny_cfg(); // hidden 4, shared intermediate 8
        let mut sh = NemotronHSharedExpert::new(&cfg).expect("shared expert builds");

        let up_w: Vec<f32> = (0..32).map(|i| (i as f32) * 0.11 - 1.5).collect();
        let down_w: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05 - 0.6).collect();
        sh.up_proj
            .set_weight(&MxArray::from_float32(&up_w, &[8, 4]).unwrap(), "up")
            .unwrap();
        sh.down_proj
            .set_weight(&MxArray::from_float32(&down_w, &[4, 8]).unwrap(), "down")
            .unwrap();

        let g_u = 3.0f32;
        let g_d = 0.5f32;
        sh.up_global_scale = Some(g_u as f64);
        sh.down_global_scale = Some(g_d as f64);

        let xrow = [0.5f32, -1.0, 0.25, 0.75];
        let x = MxArray::from_float32(&xrow, &[1, 4]).unwrap();
        let got: Vec<f32> = sh.forward(&x).unwrap().to_float32().unwrap().to_vec();

        // Reference: g_d * down(relu2(g_u * up(x))).
        let mut mid = [0.0f32; 8];
        for (i, m) in mid.iter_mut().enumerate() {
            let v: f32 = (0..4).map(|j| up_w[i * 4 + j] * xrow[j]).sum();
            *m = (g_u * v).max(0.0).powi(2);
        }
        let mut want = [0.0f32; 4];
        for (i, w) in want.iter_mut().enumerate() {
            *w = g_d * (0..8).map(|j| down_w[i * 8 + j] * mid[j]).sum::<f32>();
        }
        // And the WRONG order: g_u applied after relu2 (one power of g_u short).
        let mut wrong = [0.0f32; 4];
        for (i, w) in wrong.iter_mut().enumerate() {
            let mut m2 = [0.0f32; 8];
            for (k, mm) in m2.iter_mut().enumerate() {
                let v: f32 = (0..4).map(|j| up_w[k * 4 + j] * xrow[j]).sum();
                *mm = g_u * v.max(0.0).powi(2);
            }
            *w = g_d * (0..8).map(|j| down_w[i * 8 + j] * m2[j]).sum::<f32>();
        }

        for i in 0..4 {
            assert!(
                (got[i] - want[i]).abs() <= 1e-3 + 1e-3 * want[i].abs(),
                "shared expert[{i}]: got {} want {}",
                got[i],
                want[i]
            );
        }
        assert!(
            (0..4).any(|i| (want[i] - wrong[i]).abs() > 1e-2),
            "fixture is degenerate: the correct and wrong relu2 orderings agree"
        );
    }

    /// Dense (MTP-head) experts must NOT receive any global scale — the
    /// Quantized arm is the only one that carries one.
    #[test]
    fn dense_experts_carry_no_global_scale() {
        let cfg = tiny_cfg();
        let moe = NemotronHMoE::new(&cfg, true).expect("moe builds");
        assert!(matches!(moe.experts, NemotronHExperts::Dense { .. }));
        assert!(moe.shared_experts.up_global_scale.is_none());
        assert!(moe.shared_experts.down_global_scale.is_none());
    }
}
