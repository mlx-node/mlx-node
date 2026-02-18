use crate::array::MxArray;
use crate::nn::Activations;
use napi::bindgen_prelude::*;

use super::switch_linear::SwitchLinear;

/// SwitchGLU: Expert-indexed SwiGLU MLP using SwitchLinear (gather_mm).
///
/// Each of the three projections (gate, up, down) has per-expert weights.
/// Uses expand_dims to broadcast input across expert slots, matching
/// the mlx-lm Python implementation.
pub struct SwitchGLU {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGLU {
    pub fn new(input_dims: u32, hidden_dims: u32, num_experts: u32) -> Result<Self> {
        let gate_proj = SwitchLinear::new(input_dims, hidden_dims, num_experts)?;
        let up_proj = SwitchLinear::new(input_dims, hidden_dims, num_experts)?;
        let down_proj = SwitchLinear::new(hidden_dims, input_dims, num_experts)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B*T, D]
    /// * `indices` - Expert indices [B*T, k] (int32)
    ///
    /// # Returns
    /// Output tensor [B*T, k, D]
    pub fn forward(&self, x: &MxArray, indices: &MxArray) -> Result<MxArray> {
        let x_shape = x.shape()?;
        let ne = x_shape[0];
        let d = x_shape[1];

        // Expand x: (ne, D) -> (ne, 1, 1, D) for gather_mm broadcasting.
        // batch dims (ne, 1) broadcast with indices (ne, k) -> (ne, k)
        // matrix dims (1, D) stay as-is for the matmul
        let x_expanded = x.reshape(&[ne, 1, 1, d])?;

        // SwitchLinear via gather_mm: (ne, 1, 1, D) x (E, D, H) -> (ne, k, 1, H)
        let gate_out = self.gate_proj.forward(&x_expanded, indices, false)?;
        let up_out = self.up_proj.forward(&x_expanded, indices, false)?;

        // SwiGLU activation: silu(gate) * up -> (ne, k, 1, H)
        let activated = Activations::swiglu(&gate_out, &up_out)?;

        // Down projection: (ne, k, 1, H) -> (ne, k, 1, D)
        let out = self.down_proj.forward(&activated, indices, false)?;

        // Squeeze: (ne, k, 1, D) -> (ne, k, D)
        out.squeeze(Some(&[-2]))
    }

    // Weight accessors
    pub fn set_gate_proj_weight(&mut self, w: &MxArray) {
        self.gate_proj.set_weight(w);
    }
    pub fn set_up_proj_weight(&mut self, w: &MxArray) {
        self.up_proj.set_weight(w);
    }
    pub fn set_down_proj_weight(&mut self, w: &MxArray) {
        self.down_proj.set_weight(w);
    }
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
