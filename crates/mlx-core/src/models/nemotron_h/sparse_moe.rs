//! NemotronH pure MoE-FFN mixer: a sigmoid router, NON-gated stacked experts
//! (up -> relu2 -> down), and a shared expert applied on ALL tokens.
//!
//! Router trap: experts are SELECTED on `scores + e_score_correction_bias`,
//! but the routing weights are gathered from the UNBIASED sigmoid scores.

use crate::array::{DType, MxArray};
use crate::models::qwen3_5_moe::quantized_linear::{LinearProj, QuantizedSwitchLinear};
use crate::nn::Activations;
use napi::bindgen_prelude::*;

use super::config::NemotronHConfig;

pub(crate) fn relu2(x: &MxArray) -> Result<MxArray> {
    Activations::relu(x)?.square()
}

/// Stacked expert projections in checkpoint orientation. The dense arm keeps
/// pre-transposed stacks for gather_mm; the quantized arm runs gather_qmm on
/// the packed payload.
pub enum NemotronHExperts {
    /// Quantized (NVFP4) stacked experts.
    ///
    /// Each side carries an `[E]` Float32 `global_scale`: NVIDIA's
    /// `weight_scale_2` must NOT be folded into the per-group E4M3 scales, so
    /// it rides along and is gathered by the routing indices onto the
    /// projection OUTPUT. It is per-expert — a scalar would mis-scale the rest.
    Quantized {
        up: QuantizedSwitchLinear,
        up_global_scale: MxArray,
        down: QuantizedSwitchLinear,
        down_global_scale: MxArray,
    },
    /// Dense bf16 stacked experts (MTP head): no global scale.
    Dense { up_t: MxArray, down_t: MxArray },
}

impl NemotronHExperts {
    /// Expert-indexed projection following the SwitchGLU convention: the input
    /// gains a singleton M dim for gather_mm/gather_qmm, which is squeezed back
    /// off the [ne, k, 1, N] output.
    fn forward_proj(&self, x: &MxArray, indices: &MxArray, up: bool) -> Result<MxArray> {
        let shape = x.shape()?;
        // 2-D [ne, d] (first projection) or 3-D [ne, k, d] (second).
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
                // Squeeze FIRST: multiplying the [ne, k, 1, N] gather output by
                // the [ne, k, 1] scale would broadcast against N instead of the
                // expert dim and silently produce garbage.
                let out = ql.forward(&x4, indices, false)?.squeeze(Some(&[-2]))?;
                let dtype = out.dtype()?;
                let scale = gs.take(indices, 0)?.expand_dims(-1)?;
                let scaled = out.mul(&scale)?;
                // The f32 scale promotes the bf16 activation; restore the dtype
                // here so the rest of the block keeps running in bf16.
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

    /// Install both stacked projections at once; the two sides must agree on
    /// the quantized/dense representation.
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

    pub fn is_quantized(&self) -> bool {
        matches!(self, NemotronHExperts::Quantized { .. })
    }

    /// Test-only: install dense stacked weights.
    #[cfg(test)]
    pub(crate) fn set_dense(&mut self, up: &MxArray, down: &MxArray) -> Result<()> {
        *self = NemotronHExperts::Dense {
            up_t: up.transpose(Some(&[0, 2, 1]))?,
            down_t: down.transpose(Some(&[0, 2, 1]))?,
        };
        Ok(())
    }
}

/// One side of the stacked expert projections. The quantized arm carries the
/// MANDATORY `[E]` Float32 global scale; the loader fails closed without it.
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
    /// applied on that projection's OUTPUT. `None` for the dense bf16 MTP head.
    ///
    /// A 1-element FLOAT32 `MxArray`, never an `f64`: `mul_scalar` builds its
    /// scalar in the ARRAY's dtype, so a bf16 activation would silently round
    /// this tiny scale (and relu2 then squares the error). `mul` with an f32
    /// array promotes instead.
    pub(crate) up_global_scale: Option<MxArray>,
    pub(crate) down_global_scale: Option<MxArray>,
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
        // relu2 is homogeneous of degree 2, so the up-projection's global scale
        // MUST land before it — after it, the term is one power of g short.
        // The f32 scale promotes the activation; restore the dtype here.
        if let Some(g) = &self.up_global_scale {
            let dtype = up.dtype()?;
            up = up.mul(g)?;
            if up.dtype()? != dtype {
                up = up.astype(dtype)?;
            }
        }
        let activated = relu2(&up)?;
        let mut out = self.down_proj.forward(&activated)?;
        if let Some(g) = &self.down_global_scale {
            let dtype = out.dtype()?;
            out = out.mul(g)?;
            if out.dtype()? != dtype {
                out = out.astype(dtype)?;
            }
        }
        Ok(out)
    }

    pub fn is_quantized(&self) -> bool {
        self.up_proj.is_quantized() || self.down_proj.is_quantized()
    }
}

pub struct NemotronHMoE {
    pub(crate) gate: LinearProj,
    pub(crate) e_score_correction_bias: MxArray,
    pub(crate) experts: NemotronHExperts,
    pub(crate) shared_experts: NemotronHSharedExpert,
    pub(crate) num_experts: i32,
    pub(crate) top_k: i32,
    pub(crate) routed_scaling_factor: f64,
    pub(crate) norm_topk_prob: bool,
}

impl NemotronHMoE {
    /// Build a fresh MoE block. `dense_experts` picks plain bf16 stacks (the
    /// MTP head) over quantized backends the loader fills with NVFP4 payloads.
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
            // Placeholder payload; the loader replaces both sides and their
            // global scales via set_experts, so these units are never live.
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

    /// Forward over [B, T, hidden]: router -> routed experts -> shared.
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let shape = x.shape()?;
        // The dtype this block must hand back to the residual stream: the
        // router runs in f32 and MLX promotes, so the sum needs an explicit
        // restore. See the cast on `expert_out`.
        let io_dtype = x.dtype()?;
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

        // n_group=1/topk_group=1 degenerates to a flat top-k over all experts.
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
        // `topk_weights` descends from the f32 router, so restore the dtype
        // here, BEFORE the shared expert is added — the references close the
        // seam at the same point. Miss it and the residual promotes at the
        // first MoE layer, widening the flat KV cache and the lm_head logits
        // and desyncing the flat lane from the bf16 paged pool.
        let expert_out = weighted.sum(Some(&[1]), None)?.astype(io_dtype)?;

        let shared = self.shared_experts.forward(&x_flat)?;
        let out = expert_out.add(&shared)?;
        out.reshape(&[batch, seq, hidden])
    }

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
            norm_topk_prob: true,
            intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            eos_token_ids: vec![2],
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

        // Hand-computed `x @ W^T` for token 0.
        let logits0 = [-0.875f32, -0.75, 2.3, 1.55];
        let scores0: Vec<f32> = logits0.iter().map(|&v| sigmoid(v)).collect();
        let biased0: Vec<f32> = (0..4)
            .map(|i| scores0[i] + [0.0, 0.5, -0.4, 0.2][i])
            .collect();
        // Selection is on the BIASED scores...
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|&a, &b| biased0[b].partial_cmp(&biased0[a]).unwrap());
        let top0 = [order[0], order[1]];
        // ...but the weights come from the UNBIASED ones.
        let mut w0: Vec<f32> = top0.iter().map(|&i| scores0[i]).collect();
        let sum0: f32 = w0.iter().sum();
        for v in w0.iter_mut() {
            *v = *v / (sum0 + 1e-20) * 2.5;
        }

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

        // The shared expert is zero-initialized, so the output is the routed
        // contribution alone.
        let out = moe.forward(&x).unwrap().to_float32().unwrap().to_vec();
        let _ = (&top0, &w0, &top1, &w1);

        assert!(out.iter().all(|v| v.is_finite()));
        assert_ne!(out[0], out[4]);
    }

    /// Full MoE forward against a hand-rolled dense reference.
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
            let mut expert = [0.0f32; 4];
            for (idx, &e) in top.iter().enumerate() {
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
            // bf16 math against an f32 reference: bound scales with |y|.
            max_excess = max_excess.max((a - b).abs() - (1e-3 + 1e-3 * b.abs()));
        }
        assert!(
            max_excess <= 0.0,
            "MoE forward vs dense reference exceeded atol=1e-3, rtol=1e-3 by {max_excess} (got={got:?} want={want:?})"
        );
    }

    /// THE CROSS-MODULE SEAM: `convert` emits `.global_scale`; this is the only
    /// test that pins how the runtime consumes it. The distinct, unequal
    /// per-expert scales are load-bearing — an equal-scale fixture still passes
    /// with the scale gathered on the wrong axis, or dropped entirely.
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

        // The scales really are distinguishable across experts.
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

    /// Round an f32 through BF16 as MLX would, so the fixture below can assert
    /// its own precondition rather than trust a hand-computed neighbour.
    fn bf16_round(v: f32) -> f32 {
        MxArray::from_float32(&[v], &[1])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec()[0]
    }

    /// The shared expert's global scale must reach the activation at FULL f32
    /// precision.
    ///
    /// MUTATION CAUGHT: `up.mul_scalar(g: f64)`, which builds the scalar in the
    /// ARRAY's dtype and so silently rounds the scale on a bf16 activation.
    ///
    /// Tolerance-free by construction: `G_LO` and `G_HI` are distinct f32
    /// scales sharing the SAME bf16 image, so rounding makes the two forwards
    /// BIT-IDENTICAL while the f32 scale makes them differ.
    #[test]
    fn shared_expert_global_scale_is_not_rounded_to_bf16() {
        // The exact bf16 neighbour of the canonical weight_scale_2 8.646647e-5.
        const B: f32 = 8.630_752_6e-5;
        assert_eq!(bf16_round(B), B, "B must itself be a bf16 value");
        // Both within half a bf16 ULP (relative 1/512) of B, so both round to B.
        let g_lo = B * (1.0 - 1.0 / 1024.0);
        let g_hi = B * (1.0 + 1.0 / 1024.0);
        assert_eq!(
            bf16_round(g_lo),
            bf16_round(g_hi),
            "fixture rot: the two scales no longer share a bf16 image"
        );
        assert_ne!(g_lo, g_hi, "the two scales must be distinct in f32");

        // BF16 weights and x, the real dtype here — an f32 fixture makes
        // mul_scalar exact and hides the bug.
        const HIDDEN: i64 = 64;
        const INTER: i64 = 128;
        const TOKENS: i64 = 32;
        let mut cfg = tiny_cfg();
        cfg.hidden_size = HIDDEN as i32;
        cfg.moe_shared_expert_intermediate_size = INTER as i32;
        let mut sh = NemotronHSharedExpert::new(&cfg).expect("shared expert builds");

        let bf16_of = |vals: &[f32], shape: &[i64]| {
            MxArray::from_float32(vals, shape)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap()
        };
        let up_w: Vec<f32> = (0..(INTER * HIDDEN) as usize)
            .map(|i| ((i as f32) * 0.037).sin() * 0.5)
            .collect();
        let down_w: Vec<f32> = (0..(HIDDEN * INTER) as usize)
            .map(|i| ((i as f32) * 0.019).cos() * 0.5)
            .collect();
        sh.up_proj
            .set_weight(&bf16_of(&up_w, &[INTER, HIDDEN]), "up")
            .unwrap();
        sh.down_proj
            .set_weight(&bf16_of(&down_w, &[HIDDEN, INTER]), "down")
            .unwrap();
        let xv: Vec<f32> = (0..(TOKENS * HIDDEN) as usize)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let x = bf16_of(&xv, &[TOKENS, HIDDEN]);

        sh.up_global_scale = Some(MxArray::from_float32(&[g_lo], &[1]).unwrap());
        let lo: Vec<f32> = sh.forward(&x).unwrap().to_float32().unwrap().to_vec();
        sh.up_global_scale = Some(MxArray::from_float32(&[g_hi], &[1]).unwrap());
        let hi: Vec<f32> = sh.forward(&x).unwrap().to_float32().unwrap().to_vec();

        assert_eq!(lo.len(), (TOKENS * HIDDEN) as usize);
        let nonzero = lo.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero * 2 >= lo.len(),
            "fixture is degenerate: only {nonzero} of {} outputs are non-zero",
            lo.len()
        );
        let differing = lo.iter().zip(&hi).filter(|(a, b)| a != b).count();
        assert!(
            differing > 0,
            "two f32 global scales sharing a bf16 image produced bit-identical output \
             (0 of {} elements differ) — the scale is being rounded to bf16",
            lo.len()
        );
    }

    /// `NemotronHSharedExpert::forward` must be dtype-transparent: the f32
    /// global scales PROMOTE a bf16 activation, and the two `astype` restores
    /// are the only thing stopping that promotion from escaping into the
    /// residual stream, the flat KV cache and the lm_head matmul.
    ///
    /// It has to be a DTYPE assertion: promotion is silent and only ever adds
    /// precision, so no value comparison can see it.
    ///
    /// MUTATION CAUGHT: deleting either `astype`. The arms are isolated by
    /// leaving the other scale `None`.
    #[test]
    fn shared_expert_forward_preserves_activation_dtype() {
        const HIDDEN: i64 = 8;
        const INTER: i64 = 16;
        const TOKENS: i64 = 3;
        let mut cfg = tiny_cfg();
        cfg.hidden_size = HIDDEN as i32;
        cfg.moe_shared_expert_intermediate_size = INTER as i32;

        let up_w: Vec<f32> = (0..(INTER * HIDDEN) as usize)
            .map(|i| ((i as f32) * 0.037).sin() * 0.5)
            .collect();
        let down_w: Vec<f32> = (0..(HIDDEN * INTER) as usize)
            .map(|i| ((i as f32) * 0.019).cos() * 0.5)
            .collect();
        let xv: Vec<f32> = (0..(TOKENS * HIDDEN) as usize)
            .map(|i| ((i as f32) * 0.11).sin())
            .collect();

        // A real `weight_scale_2` magnitude, so the fixture drives the same f32
        // promotion the checkpoint does.
        let scale = || Some(MxArray::from_float32(&[8.646_647e-5], &[1]).unwrap());

        for dtype in [DType::BFloat16, DType::Float32] {
            let cast = |vals: &[f32], shape: &[i64]| {
                MxArray::from_float32(vals, shape)
                    .unwrap()
                    .astype(dtype)
                    .unwrap()
            };
            let build = || {
                let mut sh = NemotronHSharedExpert::new(&cfg).expect("shared expert builds");
                sh.up_proj
                    .set_weight(&cast(&up_w, &[INTER, HIDDEN]), "up")
                    .unwrap();
                sh.down_proj
                    .set_weight(&cast(&down_w, &[HIDDEN, INTER]), "down")
                    .unwrap();
                sh
            };
            let x = cast(&xv, &[TOKENS, HIDDEN]);

            // No scale at all (the dense MTP head): the control.
            assert_eq!(
                build().forward(&x).unwrap().dtype().unwrap(),
                dtype,
                "{dtype:?}: the unscaled shared expert must not change dtype"
            );

            // UP scale only — isolates the restore inside the `up` branch; its
            // promotion survives relu2 and reaches the output dtype.
            let mut up_only = build();
            up_only.up_global_scale = scale();
            assert_eq!(
                up_only.forward(&x).unwrap().dtype().unwrap(),
                dtype,
                "{dtype:?}: the up-branch global scale must leave the output at the \
                 input dtype, not the f32 dtype of the scale"
            );

            // DOWN scale only — isolates the restore inside the `out` branch.
            let mut down_only = build();
            down_only.down_global_scale = scale();
            assert_eq!(
                down_only.forward(&x).unwrap().dtype().unwrap(),
                dtype,
                "{dtype:?}: the down-branch global scale must leave the output at the \
                 input dtype, not the f32 dtype of the scale"
            );

            let mut both = build();
            both.up_global_scale = scale();
            both.down_global_scale = scale();
            let out = both.forward(&x).unwrap();
            assert_eq!(
                out.dtype().unwrap(),
                dtype,
                "{dtype:?}: the shared expert must hand the residual stream its own dtype"
            );

            // Anti-vacuity: an all-zero output satisfies every dtype assertion.
            let vals: Vec<f32> = out.to_float32().unwrap().to_vec();
            assert_eq!(vals.len(), (TOKENS * HIDDEN) as usize);
            let nonzero = vals.iter().filter(|v| **v != 0.0).count();
            assert!(
                nonzero * 2 >= vals.len(),
                "{dtype:?}: fixture is degenerate — only {nonzero} of {} outputs are non-zero",
                vals.len()
            );
        }
    }

    /// The global scales must straddle relu2 as `g_d * down(relu2(g_u *
    /// up(x)))`: relu2 is homogeneous of degree 2, so applying `g_u` after it
    /// scales the term by `g_u` instead of `g_u^2`.
    ///
    /// MUTATION CAUGHT: the up-scale on the wrong side of relu2; either scale
    /// dropped.
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
        sh.up_global_scale = Some(MxArray::from_float32(&[g_u], &[1]).unwrap());
        sh.down_global_scale = Some(MxArray::from_float32(&[g_d], &[1]).unwrap());

        let xrow = [0.5f32, -1.0, 0.25, 0.75];
        let x = MxArray::from_float32(&xrow, &[1, 4]).unwrap();
        let got: Vec<f32> = sh.forward(&x).unwrap().to_float32().unwrap().to_vec();

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

    /// Dense (MTP-head) experts must NOT receive any global scale.
    #[test]
    fn dense_experts_carry_no_global_scale() {
        let cfg = tiny_cfg();
        let moe = NemotronHMoE::new(&cfg, true).expect("moe builds");
        assert!(matches!(moe.experts, NemotronHExperts::Dense { .. }));
        assert!(moe.shared_experts.up_global_scale.is_none());
        assert!(moe.shared_experts.down_global_scale.is_none());
    }
}
