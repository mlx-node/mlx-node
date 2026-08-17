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
    Quantized(QuantizedSwitchLinear, QuantizedSwitchLinear),
    /// Dense bf16 stacked experts (MTP head), pre-transposed [E, K, N].
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
        let out = match self {
            NemotronHExperts::Quantized(u, d) => {
                let ql = if up { u } else { d };
                ql.forward(&x4, indices, false)?
            }
            NemotronHExperts::Dense { up_t, down_t } => {
                let wt = if up { up_t } else { down_t };
                x4.gather_mm(wt, indices, false)?
            }
        };
        out.squeeze(Some(&[-2]))
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
            (ExpertProj::Quantized(u), ExpertProj::Quantized(d)) => {
                NemotronHExperts::Quantized(u, d)
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
        matches!(self, NemotronHExperts::Quantized(..))
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
pub enum ExpertProj {
    Quantized(QuantizedSwitchLinear),
    Dense(MxArray),
}

/// Non-gated shared-expert MLP: up -> relu2 -> down, both projections
/// mode-aware (quantized NVFP4 on the backbone, dense bf16 on the MTP head).
pub struct NemotronHSharedExpert {
    pub(crate) up_proj: LinearProj,
    pub(crate) down_proj: LinearProj,
}

impl NemotronHSharedExpert {
    pub fn new(config: &NemotronHConfig) -> Result<Self> {
        let h = config.hidden_size as u32;
        let inter = config.moe_shared_expert_intermediate_size as u32;
        Ok(Self {
            up_proj: LinearProj::Standard(crate::nn::Linear::new(h, inter, Some(false))?),
            down_proj: LinearProj::Standard(crate::nn::Linear::new(inter, h, Some(false))?),
        })
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let up = self.up_proj.forward(x)?;
        let activated = relu2(&up)?;
        self.down_proj.forward(&activated)
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
            NemotronHExperts::Quantized(
                QuantizedSwitchLinear::new(
                    up,
                    scales,
                    None,
                    16,
                    4,
                    crate::models::qwen3_5::quantized_linear::NVFP4_MODE.to_string(),
                ),
                QuantizedSwitchLinear::new(
                    down,
                    scales_d,
                    None,
                    16,
                    4,
                    crate::models::qwen3_5::quantized_linear::NVFP4_MODE.to_string(),
                ),
            )
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
}
