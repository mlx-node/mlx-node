use crate::array::MxArray;
use crate::models::gemma4::quantized_linear::LinearProj;
use crate::nn::Activations;
use napi::bindgen_prelude::*;

pub struct MuseGlimmerMlp {
    gate_proj: LinearProj,
    up_proj: LinearProj,
    down_proj: LinearProj,
}

impl MuseGlimmerMlp {
    pub fn new(gate_proj: LinearProj, up_proj: LinearProj, down_proj: LinearProj) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let gate = Activations::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.mul(&up)?)
    }
}
