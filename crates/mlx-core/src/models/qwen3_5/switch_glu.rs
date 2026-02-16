use crate::array::MxArray;
use crate::nn::Activations;
use napi::bindgen_prelude::*;

use super::switch_linear::SwitchLinear;

/// SwitchGLU: Expert-indexed SwiGLU MLP using SwitchLinear.
///
/// Each of the three projections (gate, up, down) has per-expert weights.
/// When `indices.size >= 64`, sorts indices for memory-efficient gather_mm.
pub struct SwitchGLU {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGLU {
    pub fn new(
        input_dims: u32,
        hidden_dims: u32,
        num_experts: u32,
    ) -> Result<Self> {
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
    /// * `x` - Input tensor [B*T, 1, input_dims] (already expanded)
    /// * `indices` - Expert indices [B*T, 1, 1] (int32)
    ///
    /// # Returns
    /// Output tensor [B*T, 1, input_dims]
    pub fn forward(&self, x: &MxArray, indices: &MxArray) -> Result<MxArray> {
        // Determine if we should sort for efficiency
        let idx_shape = indices.shape()?;
        let idx_size: i64 = idx_shape.iter().product();
        let do_sort = idx_size >= 64;

        if do_sort {
            // Sort indices for memory-efficient gather_mm
            let flat_indices = indices.reshape(&[-1])?;
            let sort_order = flat_indices.argsort(None)?;
            let sorted_indices = flat_indices.take_along_axis(&sort_order, -1)?;

            // Reshape sorted indices back
            let sorted_indices = sorted_indices.reshape(idx_shape.as_ref())?;

            // Sort x to match
            // Flatten x for reordering, then reshape back
            let x_shape = x.shape()?;
            let x_flat = x.reshape(&[idx_size, x_shape[x_shape.len() - 1]])?;
            let sorted_x = x_flat.take_along_axis(
                &sort_order.reshape(&[idx_size, 1])?,
                0,
            )?;
            let sorted_x = sorted_x.reshape(x_shape.as_ref())?;

            // Apply SwitchLinear with sorted=true
            let gate_out = self.gate_proj.forward(&sorted_x, &sorted_indices, true)?;
            let up_out = self.up_proj.forward(&sorted_x, &sorted_indices, true)?;

            // SwiGLU activation
            let activated = Activations::swiglu(&gate_out, &up_out)?;

            // Down projection
            let out = self.down_proj.forward(&activated, &sorted_indices, true)?;

            // Unsort: reverse the permutation
            let unsort_order = sort_order.argsort(None)?;
            let out_shape = out.shape()?;
            let out_flat = out.reshape(&[idx_size, out_shape[out_shape.len() - 1]])?;
            let unsorded = out_flat.take_along_axis(
                &unsort_order.reshape(&[idx_size, 1])?,
                0,
            )?;
            unsorded.reshape(out_shape.as_ref())
        } else {
            // Direct (unsorted) path
            let gate_out = self.gate_proj.forward(x, indices, false)?;
            let up_out = self.up_proj.forward(x, indices, false)?;

            let activated = Activations::swiglu(&gate_out, &up_out)?;

            self.down_proj.forward(&activated, indices, false)
        }
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
