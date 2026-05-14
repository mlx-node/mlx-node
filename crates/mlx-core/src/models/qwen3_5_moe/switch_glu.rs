use crate::array::MxArray;
use crate::moe::{gather_sort, scatter_unsort};
use crate::nn::Activations;
use napi::bindgen_prelude::*;

use super::quantized_linear::QuantizedSwitchLinear;
use super::switch_linear::SwitchLinear;

/// A projection layer that can be either standard or quantized.
pub enum SwitchProj {
    Standard(SwitchLinear),
    Quantized(QuantizedSwitchLinear),
}

impl SwitchProj {
    fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        match self {
            SwitchProj::Standard(l) => l.forward(x, indices, sorted),
            SwitchProj::Quantized(l) => l.forward(x, indices, sorted),
        }
    }
}

/// SwitchGLU: Expert-indexed SwiGLU MLP using SwitchLinear (gather_mm) or
/// QuantizedSwitchLinear (gather_qmm).
pub struct SwitchGLU {
    pub(crate) gate_proj: SwitchProj,
    pub(crate) up_proj: SwitchProj,
    pub(crate) down_proj: SwitchProj,
}

impl SwitchGLU {
    pub fn new(input_dims: u32, hidden_dims: u32, num_experts: u32) -> Result<Self> {
        let gate_proj = SwitchLinear::new(input_dims, hidden_dims, num_experts)?;
        let up_proj = SwitchLinear::new(input_dims, hidden_dims, num_experts)?;
        let down_proj = SwitchLinear::new(hidden_dims, input_dims, num_experts)?;

        Ok(Self {
            gate_proj: SwitchProj::Standard(gate_proj),
            up_proj: SwitchProj::Standard(up_proj),
            down_proj: SwitchProj::Standard(down_proj),
        })
    }

    pub fn new_quantized(
        gate_proj: QuantizedSwitchLinear,
        up_proj: QuantizedSwitchLinear,
        down_proj: QuantizedSwitchLinear,
    ) -> Self {
        Self {
            gate_proj: SwitchProj::Quantized(gate_proj),
            up_proj: SwitchProj::Quantized(up_proj),
            down_proj: SwitchProj::Quantized(down_proj),
        }
    }

    pub fn forward(&self, x: &MxArray, indices: &MxArray) -> Result<MxArray> {
        let x_shape = x.shape()?;
        let ne = x_shape[0];
        let d = x_shape[1];
        let idx_shape = indices.shape()?;

        let x_expanded = x.reshape(&[ne, 1, 1, d])?;
        let do_sort = indices.size()? >= 64;

        let out = if do_sort {
            let sorted = gather_sort(&x_expanded, indices)?;
            let idx = &sorted.idx_sorted;

            let gate_out = self.gate_proj.forward(&sorted.x_sorted, idx, true)?;
            let up_out = self.up_proj.forward(&sorted.x_sorted, idx, true)?;
            let activated = Activations::swiglu(&gate_out, &up_out)?;
            let result = self.down_proj.forward(&activated, idx, true)?;
            scatter_unsort(&result, &sorted.inv_order, &idx_shape)?
        } else {
            let gate_out = self.gate_proj.forward(&x_expanded, indices, false)?;
            let up_out = self.up_proj.forward(&x_expanded, indices, false)?;
            let activated = Activations::swiglu(&gate_out, &up_out)?;
            self.down_proj.forward(&activated, indices, false)?
        };

        out.squeeze(Some(&[-2]))
    }

    pub fn set_gate_proj_weight(&mut self, w: &MxArray) {
        if let SwitchProj::Standard(ref mut l) = self.gate_proj {
            l.set_weight(w);
        }
    }
    pub fn set_up_proj_weight(&mut self, w: &MxArray) {
        if let SwitchProj::Standard(ref mut l) = self.up_proj {
            l.set_weight(w);
        }
    }
    pub fn set_down_proj_weight(&mut self, w: &MxArray) {
        if let SwitchProj::Standard(ref mut l) = self.down_proj {
            l.set_weight(w);
        }
    }
    pub fn get_gate_proj_weight(&self) -> MxArray {
        match &self.gate_proj {
            SwitchProj::Standard(l) => l.get_weight(),
            SwitchProj::Quantized(l) => l.get_weight().clone(),
        }
    }
    pub fn get_up_proj_weight(&self) -> MxArray {
        match &self.up_proj {
            SwitchProj::Standard(l) => l.get_weight(),
            SwitchProj::Quantized(l) => l.get_weight().clone(),
        }
    }
    pub fn get_down_proj_weight(&self) -> MxArray {
        match &self.down_proj {
            SwitchProj::Standard(l) => l.get_weight(),
            SwitchProj::Quantized(l) => l.get_weight().clone(),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(&self.gate_proj, SwitchProj::Quantized(_))
    }
}
