use crate::array::MxArray;
use crate::nn::{Activations, Linear};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::quantized_linear::QuantizedSwitchLinear;

/// Gemma4 MoE Router.
///
/// Matches mlx-lm gemma4_text.py Router:
/// Forward: `proj(rms_norm(x, scale * root_size, eps))` → argpartition top-k → softmax(top-k scores) → per_expert_scale
///
/// Key: softmax is applied to the TOP-K scores ONLY, not all experts.
/// `scale` is a learnable vector [hidden_size] loaded from `router.scale`.
/// `root_size` = hidden_size^(-0.5), fused into the rms_norm weight.
pub struct Gemma4Router {
    /// Learnable scale vector [hidden_size], loaded from checkpoint.
    /// Fused with root_size as the rms_norm weight: `weight = scale * root_size`.
    scale: MxArray,
    /// Pre-computed scalar: hidden_size^(-0.5).
    root_size: f64,
    /// RMSNorm epsilon.
    eps: f64,
    /// Projects to [num_experts] logits.
    proj: Linear,
    /// Per-expert scaling factors: [num_experts]
    per_expert_scale: MxArray,
    /// Number of experts to route to.
    top_k: i32,
}

impl Gemma4Router {
    pub fn new(hidden_size: u32, num_experts: u32, top_k: u32, eps: f64) -> Result<Self> {
        // Scale initialized to ones; will be loaded from checkpoint
        let scale = MxArray::ones(&[hidden_size as i64], None)?;
        let root_size = (hidden_size as f64).powf(-0.5);
        let proj = Linear::new(hidden_size, num_experts, Some(false))?;
        let per_expert_scale = MxArray::ones(&[num_experts as i64], None)?;

        Ok(Self {
            scale,
            root_size,
            eps,
            proj,
            per_expert_scale,
            top_k: top_k as i32,
        })
    }

    /// Compute top-k expert indices and weights.
    ///
    /// Matches mlx-lm Router.__call__:
    /// 1. rms_norm(x, scale * root_size, eps)
    /// 2. proj → expert_scores
    /// 3. argpartition → top-k indices
    /// 4. softmax over TOP-K scores only (NOT all experts)
    /// 5. Multiply by per_expert_scale
    ///
    /// Returns: (top_k_indices [B, T, K], top_k_weights [B, T, K])
    pub fn forward(&self, x: &MxArray) -> Result<(MxArray, MxArray)> {
        // Fused norm: rms_norm(x, scale * root_size, eps)
        // Matches mlx-lm: mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)
        let fused_weight = self.scale.mul_scalar(self.root_size)?;
        let normed = {
            let handle = unsafe {
                sys::mlx_fast_rms_norm(x.handle.0, fused_weight.handle.0, self.eps as f32)
            };
            MxArray::from_handle(handle, "router_rms_norm")?
        };

        let expert_scores = self.proj.forward(&normed)?;

        // Top-k via argpartition (matches mlx-lm: kth=-top_k, axis=-1)
        let ndim = expert_scores.ndim()? as usize;
        let last_axis = ndim - 1;
        let num_experts = expert_scores.shape_at(last_axis as u32)?;
        let top_k_indices_full = expert_scores.argpartition(-self.top_k, Some(-1))?;
        let top_k_indices = top_k_indices_full.slice_axis(
            last_axis,
            num_experts - self.top_k as i64,
            num_experts,
        )?;

        // Extract top-k scores and softmax over them ONLY (not all experts)
        // This is the key difference from the old implementation which did softmax over ALL experts
        let top_k_scores = expert_scores.take_along_axis(&top_k_indices, -1)?;
        let top_k_weights = Activations::softmax_precise(&top_k_scores, Some(-1))?;

        // Apply per-expert scaling: per_expert_scale[top_k_indices]
        // per_expert_scale is [num_experts], top_k_indices is [B, T, K] or [B*T, K]
        // Use take on flattened indices, then reshape back
        let idx_shape = top_k_indices.shape()?;
        let flat_idx = top_k_indices.reshape(&[-1])?;
        let top_k_scales = self.per_expert_scale.take(&flat_idx, 0)?;
        let top_k_scales = top_k_scales.reshape(&idx_shape)?;
        let top_k_weights = top_k_weights.mul(&top_k_scales)?;

        Ok((top_k_indices, top_k_weights))
    }

    // ========== Weight setters ==========

    pub fn set_scale(&mut self, w: &MxArray) -> Result<()> {
        self.scale = w.clone();
        Ok(())
    }

    pub fn set_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.proj.set_weight(w)
    }

    pub fn set_per_expert_scale(&mut self, w: &MxArray) -> Result<()> {
        self.per_expert_scale = w.clone();
        Ok(())
    }
}

/// Expert projection: either a dense weight (gather_mm) or quantized (gather_qmm).
pub enum ExpertProj {
    /// Dense weight tensor for gather_mm.
    Dense(MxArray),
    /// Quantized expert weights for gather_qmm.
    Quantized(QuantizedSwitchLinear),
}

/// Gemma4 MoE Experts block.
///
/// Weights are stored in their ORIGINAL layout (no pre-transpose):
///   gate_up_proj: [num_experts, 2*moe_inter, hidden]
///   down_proj:    [num_experts, hidden, moe_inter]
///
/// Dense path: transpose to [E, in, out] happens lazily inside gather_mm via
/// `swapaxes(-1, -2)`, matching Python's SwitchLinear which does:
///   `mx.gather_mm(x, self["weight"].swapaxes(-1, -2), ...)`
///
/// Quantized path: gather_qmm handles the transpose internally (transpose=true).
pub struct Gemma4MoE {
    /// Fused gate+up projection: [num_experts, 2*moe_inter, hidden]
    gate_up_proj: ExpertProj,
    /// Down projection: [num_experts, hidden, moe_inter]
    down_proj: ExpertProj,
    /// Pre-created scalar of top_k for floor_divide.
    k_scalar: MxArray,
    /// Pre-computed token indices for single-token decode: [K] zeros.
    /// Avoids arange allocation on every forward call in the hot path.
    single_token_indices: MxArray,
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
        let gate_up_proj = ExpertProj::Dense(MxArray::zeros(
            &[num_experts as i64, fused_inter, hidden_size as i64],
            None,
        )?);
        let down_proj = ExpertProj::Dense(MxArray::zeros(
            &[
                num_experts as i64,
                hidden_size as i64,
                moe_intermediate_size as i64,
            ],
            None,
        )?);
        let k_scalar = MxArray::scalar_int(top_k as i32)?;
        // Pre-compute token indices for single-token unsorted path: all zeros
        let single_token_indices =
            MxArray::from_int32(&vec![0i32; top_k as usize], &[top_k as i64])?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            k_scalar,
            single_token_indices,
            top_k: top_k as i32,
            moe_intermediate_size: moe_intermediate_size as i32,
        })
    }

    /// Forward pass: dispatch tokens to experts via pre-computed routing.
    ///
    /// # Arguments
    /// * `x` - Input hidden states [B, T, hidden_size]
    /// * `top_k_indices` - Expert indices from Router [B, T, K]
    /// * `top_k_weights` - Expert weights from Router [B, T, K]
    pub fn forward(
        &self,
        x: &MxArray,
        top_k_indices: &MxArray,
        top_k_weights: &MxArray,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let ne = batch * seq_len;
        let k = self.top_k as i64;
        let moe_inter = self.moe_intermediate_size as i64;
        let x_dtype = x.dtype()?;

        let x_flat = x.reshape(&[ne, hidden_size])?;

        // Sort by expert index for gather_mm efficiency.
        // Skip sorting when indices.size < 64 (single-token decode).
        let flat_indices = top_k_indices.reshape(&[-1])?;
        let do_sort = ne * k >= 64;

        let (x_for_gather, idx_for_gather, needs_unsort) = if do_sort {
            let order = flat_indices.argsort(Some(-1))?;
            let inv_order_arr = order.argsort(Some(-1))?;
            let idx_sorted = flat_indices.take(&order, 0)?;
            let x_expanded = x_flat.reshape(&[ne, 1, hidden_size])?;
            let token_indices = order.floor_divide(&self.k_scalar)?;
            let x_sorted = x_expanded.take(&token_indices, 0)?;
            (x_sorted, idx_sorted, Some(inv_order_arr))
        } else {
            let x_expanded = x_flat.reshape(&[ne, 1, hidden_size])?;
            // Positional indices (arange // k), NOT expert indices — expert IDs are
            // arbitrary (e.g. [81, 7, 38, ...]) and would index out-of-bounds.
            let token_indices = if ne == 1 {
                // Single-token decode hot path: reuse pre-computed zeros
                self.single_token_indices.clone()
            } else {
                let positions =
                    MxArray::arange(0.0, (ne * k) as f64, None, Some(crate::array::DType::Int32))?;
                positions.floor_divide(&self.k_scalar)?
            };
            let x_rep = x_expanded.take(&token_indices, 0)?;
            (x_rep, flat_indices, None)
        };

        // gate_up projection: Dense uses gather_mm with lazy transpose,
        // Quantized uses gather_qmm (transpose handled internally).
        let gate_up = match &self.gate_up_proj {
            ExpertProj::Dense(w) => {
                let w_t = w.transpose(Some(&[0, 2, 1]))?;
                x_for_gather.gather_mm(&w_t, &idx_for_gather, do_sort)?
            }
            ExpertProj::Quantized(qsl) => qsl.forward(&x_for_gather, &idx_for_gather, do_sort)?,
        };

        let gate = gate_up.slice_axis(2, 0, moe_inter)?;
        let up = gate_up.slice_axis(2, moe_inter, 2 * moe_inter)?;

        // Fused geglu: gelu_approx(gate) * up in one compiled kernel
        let hidden = {
            let handle = unsafe { sys::mlx_geglu(gate.handle.0, up.handle.0) };
            MxArray::from_handle(handle, "moe_geglu")?
        };

        // down projection: Dense uses gather_mm with lazy transpose,
        // Quantized uses gather_qmm (transpose handled internally).
        let down = match &self.down_proj {
            ExpertProj::Dense(w) => {
                let w_t = w.transpose(Some(&[0, 2, 1]))?;
                hidden.gather_mm(&w_t, &idx_for_gather, do_sort)?
            }
            ExpertProj::Quantized(qsl) => qsl.forward(&hidden, &idx_for_gather, do_sort)?,
        };

        let down_final = if let Some(inv_order) = needs_unsort {
            down.take(&inv_order, 0)?
        } else {
            down
        };
        let expert_out = down_final.reshape(&[ne, k, hidden_size])?;

        // Apply routing weights: [ne, k, 1] * [ne, k, hidden]
        let weights_flat = top_k_weights.reshape(&[ne, k, 1])?;
        let weights_flat = weights_flat.astype(x_dtype)?;
        let weighted = expert_out.mul(&weights_flat)?;

        // Sum over experts: [ne, hidden]
        let output = weighted.sum(Some(&[1]), None)?;

        output.reshape(&[batch, seq_len, hidden_size])
    }

    // ========== Weight setters ==========

    pub fn set_gate_up_proj(&mut self, w: &MxArray) -> Result<()> {
        self.gate_up_proj = ExpertProj::Dense(w.clone());
        Ok(())
    }

    pub fn set_gate_up_proj_quantized(&mut self, qsl: QuantizedSwitchLinear) {
        self.gate_up_proj = ExpertProj::Quantized(qsl);
    }

    pub fn set_down_proj(&mut self, w: &MxArray) -> Result<()> {
        self.down_proj = ExpertProj::Dense(w.clone());
        Ok(())
    }

    pub fn set_down_proj_quantized(&mut self, qsl: QuantizedSwitchLinear) {
        self.down_proj = ExpertProj::Quantized(qsl);
    }
}
