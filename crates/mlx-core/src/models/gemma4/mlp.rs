use crate::array::MxArray;
use crate::nn::{Activations, Linear};
use napi::bindgen_prelude::*;

/// Gemma4 MLP with GELU activation (GeGLU).
///
/// output = down_proj(gelu(gate_proj(x)) * up_proj(x))
///
/// Unlike the standard SwiGLU MLP used by Qwen3.5 (which uses SiLU),
/// Gemma 4 uses GELU (gelu_pytorch_tanh approximation).
pub struct GemmaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GemmaMLP {
    pub fn new(hidden_size: u32, intermediate_size: u32) -> Result<Self> {
        let gate_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
        let up_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
        let down_proj = Linear::new(intermediate_size, hidden_size, Some(false))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass: down(gelu(gate(x)) * up(x))
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = Activations::gelu(&gate)?;
        let gated = activated.mul(&up)?;
        self.down_proj.forward(&gated)
    }

    // Weight setters

    pub fn set_gate_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.gate_proj.set_weight(weight)
    }

    pub fn set_up_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.up_proj.set_weight(weight)
    }

    pub fn set_down_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.down_proj.set_weight(weight)
    }
}
