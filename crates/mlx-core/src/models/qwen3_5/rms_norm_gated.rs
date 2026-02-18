use crate::array::MxArray;
use crate::nn::Activations;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// RMSNorm with optional SwiGLU gating.
///
/// When `gate` is provided: `swiglu(gate, rms_norm(x))`
/// When `gate` is None: `rms_norm(x)`
pub struct RMSNormGated {
    weight: MxArray,
    eps: f32,
}

impl RMSNormGated {
    pub fn new(dims: u32, eps: Option<f64>) -> Result<Self> {
        let weight = MxArray::ones(&[dims as i64], None)?;
        Ok(Self {
            weight,
            eps: eps.unwrap_or(1e-6) as f32,
        })
    }

    /// Forward pass with optional gating.
    pub fn forward(&self, x: &MxArray, gate: Option<&MxArray>) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_fast_rms_norm(x.handle.0, self.weight.handle.0, self.eps) };
        let normed = MxArray::from_handle(handle, "rms_norm_gated")?;
        match gate {
            Some(g) => Activations::swiglu(g, &normed),
            None => Ok(normed),
        }
    }

    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    pub fn set_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.weight = weight.clone();
        Ok(())
    }
}

impl Clone for RMSNormGated {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            eps: self.eps,
        }
    }
}
