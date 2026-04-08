use crate::array::MxArray;
use crate::nn::Linear;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Gemma4 MLP with GeGLU activation.
///
/// output = down_proj(geglu(gate_proj(x), up_proj(x)))
///
/// Uses a compiled (fused) `geglu` kernel matching Python's
/// `@partial(mx.compile, shapeless=True) def geglu(gate, x): return nn.gelu_approx(gate) * x`
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

    /// Forward pass: down(geglu(gate(x), up(x)))
    /// Uses compiled fused geglu kernel (gelu_approx(gate) * up → single Metal dispatch).
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // Fused geglu: gelu_approx(gate) * up in one compiled kernel
        let handle = unsafe { sys::mlx_geglu(gate.handle.0, up.handle.0) };
        let gated = MxArray::from_handle(handle, "geglu")?;
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
