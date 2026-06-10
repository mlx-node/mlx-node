use crate::array::MxArray;
// Only the test-only `forward_with_cache` uses Activations on the Rust side;
// the production forward routes through the fused C++ SwiGLU kernels.
#[cfg(test)]
use crate::nn::Activations;
use crate::nn::Linear;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Multi-Layer Perceptron with SwiGLU activation.
///
/// Uses the gated linear unit activation popularized by models like Llama and Qwen:
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
///
/// This is more expressive than standard FFN and is the default in modern LLMs.
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    /// E39: pre-stacked `[w_gate; w_up]` then transposed to `[hidden, 2*intermediate]`.
    /// Populated by `finalize_gate_up()` after weights are loaded. When present,
    /// `forward()` uses `mlx_swiglu_mlp_forward_stacked` (one matmul instead of two
    /// plus the per-call transposes baked in).
    gate_up_proj_wt: Option<MxArray>,
    /// E39: pre-transposed down_proj weight `[hidden, intermediate]`. Same idea:
    /// hoist the per-forward transpose to load time.
    down_proj_wt: Option<MxArray>,
}

impl MLP {
    /// Creates a new MLP (SwiGLU) layer.
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Hidden dimension (typically 4x or more of hidden_size)
    pub fn new(hidden_size: u32, intermediate_size: u32) -> Result<Self> {
        // All three projections have no bias (standard in modern architectures)
        let gate_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
        let up_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
        let down_proj = Linear::new(intermediate_size, hidden_size, Some(false))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_up_proj_wt: None,
            down_proj_wt: None,
        })
    }

    /// E39: precompute the stacked `[gate; up]^T` weight and the transposed
    /// `down_proj^T` weight once after all three projection weights are loaded.
    /// Forward will then use `mlx_swiglu_mlp_forward_stacked`, which does ONE
    /// (x @ wgu_t) matmul instead of two separate (x @ w_gate.T) + (x @ w_up.T)
    /// matmuls, and reads pre-transposed weights so the per-call `transpose`
    /// graph nodes vanish.
    ///
    /// Safe to call repeatedly (idempotent — overwrites). Callers from the
    /// persistence layer should invoke it once after the gate/up/down weights
    /// for a given layer have all been set.
    pub fn finalize_gate_up(&mut self) -> Result<()> {
        let w_gate = self.gate_proj.get_weight();
        let w_up = self.up_proj.get_weight();
        let w_down = self.down_proj.get_weight();
        // gate, up: [intermediate, hidden] → stacked: [2*intermediate, hidden] →
        // transpose to [hidden, 2*intermediate] for the matmul x @ wgu_t.
        let stacked = MxArray::concatenate(&w_gate, &w_up, 0)?;
        let wgu_t = stacked.transpose(Some(&[1, 0]))?;
        wgu_t.eval();
        // down: [hidden, intermediate] → [intermediate, hidden]
        let wd_t = w_down.transpose(Some(&[1, 0]))?;
        wd_t.eval();
        self.gate_up_proj_wt = Some(wgu_t);
        self.down_proj_wt = Some(wd_t);
        Ok(())
    }

    /// Forward pass: down(silu(gate(x)) * up(x))
    ///
    /// Uses fused C++ implementation for maximum performance (1 FFI call vs 8).
    ///
    /// # Arguments
    /// * `x` - Input tensor, shape: (batch, seq_len, hidden_size)
    ///
    /// # Returns
    /// Output tensor, shape: (batch, seq_len, hidden_size)
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        // E39: fast path — pre-stacked + pre-transposed weights.
        // Env-toggle MLX_DISABLE_E39_STACKED_MLP=1 reverts to the legacy
        // two-matmul path for A/B testing.
        if let (Some(wgu_t), Some(wd_t)) = (&self.gate_up_proj_wt, &self.down_proj_wt)
            && std::env::var("MLX_DISABLE_E39_STACKED_MLP").is_err()
        {
            let handle = unsafe {
                sys::mlx_swiglu_mlp_forward_stacked(x.handle.0, wgu_t.handle.0, wd_t.handle.0)
            };
            return MxArray::from_handle(handle, "swiglu_mlp_forward_stacked");
        }

        // Legacy path: two matmuls with per-call transposes.
        let w_gate = self.gate_proj.get_weight();
        let w_up = self.up_proj.get_weight();
        let w_down = self.down_proj.get_weight();

        let handle = unsafe {
            sys::mlx_swiglu_mlp_forward(x.handle.0, w_gate.handle.0, w_up.handle.0, w_down.handle.0)
        };
        MxArray::from_handle(handle, "swiglu_mlp_forward")
    }

    /// Forward pass with cached intermediates for backward pass
    ///
    /// Returns: [output, gate, up, gate_act, gated]
    /// - output: final output
    /// - gate: gate_proj(x)
    /// - up: up_proj(x)
    /// - gate_act: silu(gate)
    /// - gated: gate_act * up
    #[cfg(test)]
    pub fn forward_with_cache(&self, x: &MxArray) -> Result<Vec<MxArray>> {
        // Compute gate and up projections
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        // Apply SiLU activation to gate
        let gate_act = Activations::silu(&gate)?;

        // Element-wise multiplication
        let gated = gate_act.mul(&up)?;

        // Down projection
        let output = self.down_proj.forward(&gated)?;

        Ok(vec![output, gate, up, gate_act, gated])
    }

    // Weight setters for loading pretrained models

    pub fn set_gate_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.gate_proj.set_weight(weight)?;
        // Invalidate the E39 stacked cache — caller must call finalize_gate_up().
        self.gate_up_proj_wt = None;
        Ok(())
    }

    pub fn set_up_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.up_proj.set_weight(weight)?;
        self.gate_up_proj_wt = None;
        Ok(())
    }

    pub fn set_down_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.down_proj.set_weight(weight)?;
        self.down_proj_wt = None;
        Ok(())
    }

    // Mutable projection accessors.
    //
    // Expose the underlying `Linear`s so a persistence layer can drive
    // affine-quantized loads (`Linear::load_quantized`) or plain bf16 loads
    // uniformly without this shared module needing to know about each model's
    // quantization scheme. Each accessor invalidates the E39 stacked-MLP cache
    // (`gate_up_proj_wt` / `down_proj_wt`), because a caller obtaining a `&mut
    // Linear` may replace the weight; a stale stacked cache would otherwise be
    // served by `forward()`. The caller must re-run `finalize_gate_up()` if it
    // wants the stacked fast path after mutating a projection. This mirrors the
    // invalidation already done by `set_{gate,up,down}_proj_weight`.

    pub fn gate_proj_mut(&mut self) -> &mut Linear {
        self.gate_up_proj_wt = None;
        &mut self.gate_proj
    }

    pub fn up_proj_mut(&mut self) -> &mut Linear {
        self.gate_up_proj_wt = None;
        &mut self.up_proj
    }

    pub fn down_proj_mut(&mut self) -> &mut Linear {
        self.down_proj_wt = None;
        &mut self.down_proj
    }

    // Weight getters for backward pass

    pub fn get_gate_proj_weight(&self) -> MxArray {
        self.gate_proj.get_weight()
    }

    pub fn get_up_proj_weight(&self) -> MxArray {
        self.up_proj.get_weight()
    }

    pub fn get_down_proj_weight(&self) -> MxArray {
        self.down_proj.get_weight()
    }
}

impl Clone for MLP {
    fn clone(&self) -> Self {
        Self {
            gate_proj: self.gate_proj.clone(),
            up_proj: self.up_proj.clone(),
            down_proj: self.down_proj.clone(),
            gate_up_proj_wt: self.gate_up_proj_wt.clone(),
            down_proj_wt: self.down_proj_wt.clone(),
        }
    }
}
