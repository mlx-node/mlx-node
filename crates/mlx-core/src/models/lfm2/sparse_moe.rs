//! LFM2.5 sparse Mixture-of-Experts feed-forward block.
//!
//! Mirrors `mlx-lm/mlx_lm/models/lfm2_moe.py::Lfm2MoeSparseMoeBlock`
//! (lines 189-226) EXACTLY:
//!
//! ```text
//! gates = softmax(gate(x).astype(f32), axis=-1)
//! if use_expert_bias: gates += expert_bias            # post-softmax, pre-topk
//! inds = argpartition(gates, kth=-k, axis=-1)[..., -k:]
//! scores = take_along_axis(gates, inds, axis=-1)       # from POST-BIAS gates
//! if norm_topk_prob: scores /= sum(scores, -1, keepdims) + 1e-20
//! scores = scores.astype(x.dtype)
//! y = switch_mlp(x, inds)
//! y = (y * scores[..., None]).sum(axis=-2)
//! ```
//!
//! Differences from `qwen3_5_moe::SparseMoeBlock`:
//! - NO shared expert / shared_expert_gate / routed_scaling_factor.
//! - The learned `expert_bias` is added to the post-softmax gates BEFORE
//!   top-k selection, and the routing scores are gathered from those
//!   biased gates. This cannot reuse `moe::topk_from_logits` (which has no
//!   bias step), so the gate math is inlined here.
//! - `expert_bias` stays f32 (matches Python `cast_predicate`).

use crate::array::{DType, MxArray};
use crate::models::qwen3_5::quantized_linear::{LinearProj, QuantizedLinear};
use crate::models::qwen3_5_moe::switch_glu::SwitchGLU;
use crate::nn::{Activations, Linear};
use napi::bindgen_prelude::*;

use super::config::Lfm2Config;

/// LFM2.5 sparse MoE block.
pub struct Lfm2SparseMoeBlock {
    /// Router gate: `Standard(Linear)` for bf16/f16, `Quantized` for quantized
    /// checkpoints. Convert forces `feed_forward.gate` to 8-bit affine (never
    /// mxfp8) via `is_router_gate`, so the quantized form is 8-bit affine.
    gate: LinearProj,
    /// Expert-indexed SwiGLU (gather_mm / gather_qmm). Reused from qwen3_5_moe.
    switch_mlp: SwitchGLU,
    /// Per-expert routing bias `(num_experts,)`, kept f32. `None` when
    /// `use_expert_bias` is false.
    expert_bias: Option<MxArray>,
    num_experts: i32,
    top_k: i32,
    norm_topk_prob: bool,
}

impl Lfm2SparseMoeBlock {
    /// Build a bf16/f16 MoE block from config. Quantized experts/gate are
    /// installed afterward by the loader via the setters.
    pub fn new(config: &Lfm2Config) -> Result<Self> {
        let hidden = config.hidden_size;
        let num_experts = config
            .num_experts
            .ok_or_else(|| Error::from_reason("Lfm2SparseMoeBlock::new requires num_experts"))?;
        let top_k = config.num_experts_per_tok.ok_or_else(|| {
            Error::from_reason("Lfm2SparseMoeBlock::new requires num_experts_per_tok")
        })?;
        if num_experts <= 0 || top_k <= 0 || top_k > num_experts {
            return Err(Error::from_reason(format!(
                "Lfm2SparseMoeBlock requires 0 < num_experts_per_tok <= num_experts, \
                 got top_k={top_k} num_experts={num_experts}"
            )));
        }
        let moe_inter = config.moe_intermediate_size.ok_or_else(|| {
            Error::from_reason("Lfm2SparseMoeBlock::new requires moe_intermediate_size")
        })?;

        let gate = Linear::new(hidden as u32, num_experts as u32, Some(false))?;
        let switch_mlp = SwitchGLU::new(hidden as u32, moe_inter as u32, num_experts as u32)?;
        let expert_bias = if config.use_expert_bias.unwrap_or(true) {
            Some(MxArray::zeros(&[num_experts as i64], Some(DType::Float32))?)
        } else {
            None
        };

        Ok(Self {
            gate: LinearProj::Standard(gate),
            switch_mlp,
            expert_bias,
            num_experts,
            top_k,
            norm_topk_prob: config.norm_topk_prob.unwrap_or(true),
        })
    }

    // ========== Weight setters (used by persistence) ==========

    pub fn set_gate_weight(&mut self, w: &MxArray) -> Result<()> {
        self.gate.set_weight(w, "feed_forward.gate")
    }

    pub fn set_quantized_gate(&mut self, q: QuantizedLinear) {
        self.gate.set_quantized(q);
    }

    pub fn set_switch_mlp(&mut self, m: SwitchGLU) {
        self.switch_mlp = m;
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

    /// Set the per-expert routing bias. Defensively re-cast to f32 to match
    /// Python `cast_predicate` (which excludes `expert_bias` from the bf16
    /// cast).
    pub fn set_expert_bias(&mut self, b: &MxArray) -> Result<()> {
        self.expert_bias = Some(b.astype(DType::Float32)?);
        Ok(())
    }

    /// Test-only inspector: whether an expert bias is currently installed.
    /// Used by persistence tests to assert the loader honors
    /// `config.use_expert_bias` for version-skewed checkpoints.
    #[cfg(test)]
    pub fn expert_bias_is_some(&self) -> bool {
        self.expert_bias.is_some()
    }

    /// Forward pass. Input `x` is `[B, T, D]`; output is `[B, T, D]`.
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let shape = x.shape()?;
        if shape.len() != 3 {
            return Err(Error::from_reason(format!(
                "Lfm2SparseMoeBlock::forward expects 3D input [B, T, D], got {}D",
                shape.len()
            )));
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden = shape[2];
        let ne = batch * seq_len;
        let k = self.top_k as i64;
        let ne_e = self.num_experts as i64;

        let x_flat = x.reshape(&[ne, hidden])?;
        let x_dtype = x.dtype()?;

        // gates = softmax(gate(x).astype(f32), axis=-1)
        let logits = self.gate.forward(&x_flat)?; // [ne, num_experts]
        let logits = logits.astype(DType::Float32)?;
        let mut gates = Activations::softmax(&logits, Some(-1))?;

        // if use_expert_bias: gates += expert_bias  (f32 + f32 stays f32;
        // broadcasts [ne, E] + [E])
        if let Some(bias) = &self.expert_bias {
            gates = gates.add(bias)?;
        }

        // inds = argpartition(gates, kth=-k)[..., -k:]  (UNSORTED top-k)
        let inds_full = gates.argpartition(-self.top_k, Some(-1))?;
        let inds = inds_full.slice_axis(1, ne_e - k, ne_e)?; // [ne, k]

        // scores = take_along_axis(gates, inds, -1)  — from POST-BIAS gates
        let mut scores = gates.take_along_axis(&inds, -1)?; // [ne, k]

        // if norm_topk_prob: scores /= sum(scores, -1, keepdims) + 1e-20
        if self.norm_topk_prob {
            let denom = scores.sum(Some(&[-1]), Some(true))?; // [ne, 1]
            let eps = MxArray::scalar_float(1e-20)?.astype(DType::Float32)?;
            let denom = denom.add(&eps)?;
            scores = scores.div(&denom)?;
        }
        let scores = scores.astype(x_dtype)?;

        // y = switch_mlp(x, inds);  y = (y * scores[..., None]).sum(axis=-2)
        let y = self.switch_mlp.forward(&x_flat, &inds)?; // [ne, k, hidden]
        let scores_exp = scores.reshape(&[ne, k, 1])?;
        let weighted = y.mul(&scores_exp)?;
        let out = weighted.sum(Some(&[1]), None)?; // [ne, hidden]
        out.reshape(&[batch, seq_len, hidden])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal MoE config: hidden=4, num_experts=4, top_k=2,
    /// moe_intermediate_size=4.
    fn tiny_moe_config(
        num_dense_layers: i32,
        norm_topk_prob: bool,
        use_expert_bias: bool,
    ) -> Lfm2Config {
        Lfm2Config {
            vocab_size: 32,
            hidden_size: 4,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            max_position_embeddings: 128,
            norm_eps: 1e-5,
            conv_bias: false,
            conv_l_cache: 3,
            block_dim: 4,
            block_ff_dim: 4,
            block_multiple_of: 256,
            block_ffn_dim_multiplier: 1.0,
            block_auto_adjust_ff_dim: false,
            rope_theta: 1_000_000.0,
            layer_types: vec!["conv".into(), "full_attention".into()],
            tie_embedding: true,
            eos_token_id: 7,
            bos_token_id: 1,
            pad_token_id: 0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
            intermediate_size: Some(4),
            moe_intermediate_size: Some(4),
            num_experts: Some(4),
            num_experts_per_tok: Some(2),
            num_dense_layers: Some(num_dense_layers),
            norm_topk_prob: Some(norm_topk_prob),
            use_expert_bias: Some(use_expert_bias),
        }
    }

    // ----- reference math in plain Rust f32 -----

    const HIDDEN: usize = 4;
    const N_EXP: usize = 4;
    const INTER: usize = 4;
    const TOP_K: usize = 2;

    /// Gate weight `(num_experts, hidden)` row-major. Chosen so the four
    /// experts get clearly separated logits for a fixed input.
    fn gate_w() -> Vec<f32> {
        // logits[e] = dot(gate_w[e], x)
        vec![
            1.0, 0.0, 0.0, 0.0, // expert 0
            0.0, 2.0, 0.0, 0.0, // expert 1
            0.0, 0.0, 0.5, 0.0, // expert 2
            0.0, 0.0, 0.0, 0.3, // expert 3
        ]
    }

    /// Per-expert gate_proj weights `(num_experts, inter, hidden)`.
    fn expert_gate() -> Vec<f32> {
        (0..N_EXP * INTER * HIDDEN)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1 + ((i / 13) as f32) * 0.05)
            .collect()
    }
    /// Per-expert up_proj weights `(num_experts, inter, hidden)`.
    fn expert_up() -> Vec<f32> {
        (0..N_EXP * INTER * HIDDEN)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.15 + 0.1)
            .collect()
    }
    /// Per-expert down_proj weights `(num_experts, hidden, inter)`.
    fn expert_down() -> Vec<f32> {
        (0..N_EXP * HIDDEN * INTER)
            .map(|i| ((i % 6) as f32 - 2.5) * 0.12)
            .collect()
    }

    fn softmax(v: &[f32]) -> Vec<f32> {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = v.iter().map(|&z| (z - m).exp()).collect();
        let s: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / s).collect()
    }

    fn silu(z: f32) -> f32 {
        z / (1.0 + (-z).exp())
    }

    /// Compute one expert's SwiGLU MLP output for a single token `x`.
    fn expert_forward(e: usize, x: &[f32]) -> Vec<f32> {
        let g = expert_gate();
        let u = expert_up();
        let d = expert_down();
        // gate_proj: (inter, hidden) @ x -> (inter,)
        let mut gate_out = [0.0f32; INTER];
        let mut up_out = [0.0f32; INTER];
        for r in 0..INTER {
            let mut gs = 0.0;
            let mut us = 0.0;
            for c in 0..HIDDEN {
                gs += g[e * INTER * HIDDEN + r * HIDDEN + c] * x[c];
                us += u[e * INTER * HIDDEN + r * HIDDEN + c] * x[c];
            }
            gate_out[r] = gs;
            up_out[r] = us;
        }
        // swiglu = silu(gate) * up
        let act: Vec<f32> = (0..INTER).map(|r| silu(gate_out[r]) * up_out[r]).collect();
        // down_proj: (hidden, inter) @ act -> (hidden,)
        let mut out = vec![0.0f32; HIDDEN];
        for r in 0..HIDDEN {
            let mut s = 0.0;
            for c in 0..INTER {
                s += d[e * HIDDEN * INTER + r * INTER + c] * act[c];
            }
            out[r] = s;
        }
        out
    }

    /// Full reference MoE forward for a single token.
    fn reference_forward(x: &[f32], bias: &[f32; N_EXP], norm_topk_prob: bool) -> Vec<f32> {
        let gw = gate_w();
        // logits
        let mut logits = vec![0.0f32; N_EXP];
        for e in 0..N_EXP {
            let mut s = 0.0;
            for c in 0..HIDDEN {
                s += gw[e * HIDDEN + c] * x[c];
            }
            logits[e] = s;
        }
        // softmax then + bias
        let mut gates = softmax(&logits);
        for e in 0..N_EXP {
            gates[e] += bias[e];
        }
        // top-k (by biased gate value); ties broken by lower index for
        // determinism (won't occur with our weights).
        let mut idx: Vec<usize> = (0..N_EXP).collect();
        idx.sort_by(|&a, &b| {
            gates[b]
                .partial_cmp(&gates[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let top: Vec<usize> = idx[..TOP_K].to_vec();
        let mut scores: Vec<f32> = top.iter().map(|&e| gates[e]).collect();
        if norm_topk_prob {
            let s: f32 = scores.iter().sum::<f32>() + 1e-20;
            for v in scores.iter_mut() {
                *v /= s;
            }
        }
        // weighted sum of expert outputs
        let mut out = vec![0.0f32; HIDDEN];
        for (slot, &e) in top.iter().enumerate() {
            let eo = expert_forward(e, x);
            for h in 0..HIDDEN {
                out[h] += eo[h] * scores[slot];
            }
        }
        out
    }

    fn build_block(
        norm_topk_prob: bool,
        use_expert_bias: bool,
        bias: Option<&[f32; N_EXP]>,
    ) -> Lfm2SparseMoeBlock {
        let cfg = tiny_moe_config(0, norm_topk_prob, use_expert_bias);
        let mut block = Lfm2SparseMoeBlock::new(&cfg).expect("new block");

        // gate weight (num_experts, hidden)
        let gw = MxArray::from_float32(&gate_w(), &[N_EXP as i64, HIDDEN as i64]).expect("gate w");
        let gw = gw.astype(DType::BFloat16).expect("gate bf16");
        block.set_gate_weight(&gw).expect("set gate w");

        // stacked expert weights
        let g = MxArray::from_float32(&expert_gate(), &[N_EXP as i64, INTER as i64, HIDDEN as i64])
            .expect("gate proj");
        let u = MxArray::from_float32(&expert_up(), &[N_EXP as i64, INTER as i64, HIDDEN as i64])
            .expect("up proj");
        let d = MxArray::from_float32(&expert_down(), &[N_EXP as i64, HIDDEN as i64, INTER as i64])
            .expect("down proj");
        let g = g.astype(DType::BFloat16).expect("g bf16");
        let u = u.astype(DType::BFloat16).expect("u bf16");
        let d = d.astype(DType::BFloat16).expect("d bf16");
        block.set_switch_mlp_gate_proj_weight(&g);
        block.set_switch_mlp_up_proj_weight(&u);
        block.set_switch_mlp_down_proj_weight(&d);

        if let Some(b) = bias {
            let ba = MxArray::from_float32(b, &[N_EXP as i64]).expect("bias");
            block.set_expert_bias(&ba).expect("set bias");
        }
        block
    }

    fn run_single(block: &Lfm2SparseMoeBlock, x: &[f32]) -> Vec<f32> {
        let xa = MxArray::from_float32(x, &[1, 1, HIDDEN as i64]).expect("x");
        let xa = xa.astype(DType::BFloat16).expect("x bf16");
        let out = block.forward(&xa).expect("forward");
        let out = out.astype(DType::Float32).expect("out f32");
        out.eval();
        out.to_float32().expect("to_float32").to_vec()
    }

    #[test]
    fn forward_zero_bias_matches_reference() {
        let x = [0.5f32, -1.0, 2.0, 0.25];
        let bias = [0.0f32; N_EXP];
        let block = build_block(true, true, Some(&bias));
        let got = run_single(&block, &x);
        let want = reference_forward(&x, &bias, true);
        assert_eq!(got.len(), HIDDEN);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() < 2e-3,
                "zero-bias mismatch got={got:?} want={want:?}"
            );
        }
    }

    #[test]
    fn forward_bias_flips_top_k() {
        // With x, the un-biased top-2 are experts {1, 0} (largest softmax).
        // A large positive bias on experts {2,3} flips the top-2 to {2,3},
        // proving the bias is applied BEFORE argpartition and scores are
        // gathered from the biased gates.
        let x = [0.5f32, -1.0, 2.0, 0.25];

        let no_bias = [0.0f32; N_EXP];
        let unbiased_ref = reference_forward(&x, &no_bias, true);

        let bias = [0.0f32, 0.0, 5.0, 5.0];
        let block = build_block(true, true, Some(&bias));
        let got = run_single(&block, &x);
        let want = reference_forward(&x, &bias, true);

        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() < 2e-3,
                "biased mismatch got={got:?} want={want:?}"
            );
        }
        // Sanity: the flipped result must differ from the un-biased result
        // (different experts selected).
        let differs = got
            .iter()
            .zip(unbiased_ref.iter())
            .any(|(g, u)| (g - u).abs() > 1e-2);
        assert!(
            differs,
            "bias flip produced identical output to unbiased (got={got:?} unbiased={unbiased_ref:?})"
        );
    }

    #[test]
    fn forward_no_renorm_differs_from_renorm() {
        let x = [0.5f32, -1.0, 2.0, 0.25];
        let bias = [0.0f32; N_EXP];

        let block_renorm = build_block(true, true, Some(&bias));
        let got_renorm = run_single(&block_renorm, &x);
        let want_renorm = reference_forward(&x, &bias, true);

        let block_no = build_block(false, true, Some(&bias));
        let got_no = run_single(&block_no, &x);
        let want_no = reference_forward(&x, &bias, false);

        for (g, w) in got_no.iter().zip(want_no.iter()) {
            assert!(
                (g - w).abs() < 2e-3,
                "no-renorm mismatch got={got_no:?} want={want_no:?}"
            );
        }
        // Un-renormalized scores sum to < 1 here, so the weighted output must
        // differ from the renormalized output.
        let differs = got_no
            .iter()
            .zip(got_renorm.iter())
            .any(|(a, b)| (a - b).abs() > 1e-3);
        assert!(
            differs,
            "norm_topk_prob=false produced same output as true (no-renorm={got_no:?} renorm={got_renorm:?} want_renorm={want_renorm:?})"
        );
    }

    /// Skips on hosts where either NAX half-precision kernel class this test
    /// depends on is broken (gen>=17 GPUs on the current vendored-MLX pin):
    ///
    /// 1. `test_support::half_gemm_untrustworthy` — with 32 tokens the router
    ///    gate projection is a bf16 `[32,4] @ [4,4]` GEMM (M>1, K=4), which
    ///    the NAX unaligned-K bug turns into garbage (probed vs host truth:
    ///    max_err 3.2; e.g. token-0 logit for expert 2 computes 0 where the
    ///    true value is 1.0), corrupting every token's routing scores before
    ///    the experts even run. The single-token sibling tests stay green
    ///    because M=1 dispatches the correct GEMV.
    /// 2. `test_support::sorted_gather_mm_untrustworthy` — the >= 64-index
    ///    branch also switches expert dispatch to sorted `gather_mm`
    ///    (`gather_mm_rhs_nax`), which is garbage at every K on such hosts.
    ///
    /// The gather-sort logic itself is NOT under suspicion: the identical
    /// 32-token forward with f32 weights/input (same gather_sort /
    /// scatter_unsort code, non-NAX kernels via MLX_ENABLE_TF32=0) matches
    /// the per-token reference to 1.5e-8. Both canaries must be healthy
    /// before the bf16 assertion means anything again, hence the OR.
    #[test]
    fn forward_64_index_gather_sort_branch() {
        if crate::test_support::half_gemm_untrustworthy()
            || crate::test_support::sorted_gather_mm_untrustworthy()
        {
            eprintln!(
                "skipping forward_64_index_gather_sort_branch: this host's NAX \
                 half-precision kernels are broken (plain unaligned-K GEMM \
                 corrupts the [32,4]@[4,4] router gate, and/or sorted \
                 gather_mm_rhs corrupts the expert matmuls), so the bf16 \
                 gather-sort parity assertion is not meaningful here"
            );
            return;
        }
        // 32 tokens x top_k=2 = 64 indices >= 64 → exercises the gather-sort
        // path in SwitchGLU::forward. Must equal the per-token reference.
        let bias = [0.0f32; N_EXP];
        let block = build_block(true, true, Some(&bias));

        let n_tok = 32usize;
        let mut xs = Vec::with_capacity(n_tok * HIDDEN);
        for t in 0..n_tok {
            // vary inputs so different experts win across tokens
            xs.push(0.5 + 0.1 * t as f32);
            xs.push(-1.0 + 0.05 * t as f32);
            xs.push(2.0 - 0.07 * t as f32);
            xs.push(0.25 + 0.03 * t as f32);
        }
        let xa = MxArray::from_float32(&xs, &[1, n_tok as i64, HIDDEN as i64]).expect("xs");
        let xa = xa.astype(DType::BFloat16).expect("xs bf16");
        let out = block.forward(&xa).expect("forward");
        let out = out.astype(DType::Float32).expect("out f32");
        out.eval();
        let got = out.to_float32().expect("to_float32").to_vec();
        assert_eq!(got.len(), n_tok * HIDDEN);

        for t in 0..n_tok {
            let x = &xs[t * HIDDEN..(t + 1) * HIDDEN];
            let want = reference_forward(x, &bias, true);
            for h in 0..HIDDEN {
                let g = got[t * HIDDEN + h];
                assert!(
                    (g - want[h]).abs() < 3e-3,
                    "gather-sort token {t} dim {h}: got={g} want={}",
                    want[h]
                );
            }
        }
    }
}
