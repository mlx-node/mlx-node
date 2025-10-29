use crate::array::MxArray;
use crate::nn::{Activations, Linear};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Multi-Layer Perceptron with SwiGLU activation.
///
/// Uses the gated linear unit activation popularized by models like Llama and Qwen:
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
///
/// This is more expressive than standard FFN and is the default in modern LLMs.
#[napi(js_name = "MLP")]
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

#[napi]
impl MLP {
    /// Creates a new MLP (SwiGLU) layer.
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Hidden dimension (typically 4x or more of hidden_size)
    #[napi(constructor)]
    pub fn new(hidden_size: u32, intermediate_size: u32) -> Result<Self> {
        // All three projections have no bias (standard in modern architectures)
        let gate_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
        let up_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
        let down_proj = Linear::new(intermediate_size, hidden_size, Some(false))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
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
    #[napi]
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        // Use fused C++ implementation: reduces 8 FFI calls to 1
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
    #[napi]
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

    #[napi]
    pub fn set_gate_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.gate_proj.set_weight(weight)?;
        Ok(())
    }

    #[napi]
    pub fn set_up_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.up_proj.set_weight(weight)?;
        Ok(())
    }

    #[napi]
    pub fn set_down_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.down_proj.set_weight(weight)?;
        Ok(())
    }

    // Weight getters for backward pass

    #[napi]
    pub fn get_gate_proj_weight(&self) -> MxArray {
        self.gate_proj.get_weight()
    }

    #[napi]
    pub fn get_up_proj_weight(&self) -> MxArray {
        self.up_proj.get_weight()
    }

    #[napi]
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
        }
    }
}
