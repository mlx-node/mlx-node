use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ============================================
// Normalization Layers
// ============================================

#[napi(js_name = "RMSNorm")]
pub struct RMSNorm {
    weight: MxArray,
    eps: f64,
}

#[napi]
impl RMSNorm {
    /// Create a new RMSNorm layer
    #[napi(constructor)]
    pub fn new(dims: u32, eps: Option<f64>) -> Result<Self> {
        let weight_shape = vec![dims as i64];
        let weight = MxArray::ones(&weight_shape, None)?;

        Ok(Self {
            weight,
            eps: eps.unwrap_or(1e-5),
        })
    }

    /// Forward pass: RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
    /// Uses mx.fast.rms_norm for optimal performance (single fused Metal kernel)
    #[napi]
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_fast_rms_norm(input.handle.0, self.weight.handle.0, self.eps as f32)
        };
        MxArray::from_handle(handle, "fast_rms_norm")
    }

    /// Get the weight (scale) parameter
    #[napi]
    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    /// Set the weight (scale) parameter
    #[napi]
    pub fn set_weight(&mut self, weight: &MxArray) -> Result<()> {
        let shape = weight.shape()?;
        if shape.len() != 1 {
            return Err(Error::from_reason(format!(
                "RMSNorm weight must be 1D, got shape {:?}",
                shape.as_ref()
            )));
        }
        // Clone the Arc reference (no need to copy the underlying MLX array)
        self.weight = weight.clone();
        Ok(())
    }
}

#[napi]
pub struct LayerNorm {
    weight: MxArray,
    bias: MxArray,
    eps: f64,
}

#[napi]
impl LayerNorm {
    /// Create a new LayerNorm layer
    #[napi(constructor)]
    pub fn new(dims: u32, eps: Option<f64>) -> Result<Self> {
        let shape = vec![dims as i64];
        let weight = MxArray::ones(&shape, None)?;
        let bias = MxArray::zeros(&shape, None)?;

        Ok(Self {
            weight,
            bias,
            eps: eps.unwrap_or(1e-5),
        })
    }

    /// Forward pass: LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
    /// Uses mx.fast.layer_norm for optimal performance (single fused Metal kernel)
    #[napi]
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_fast_layer_norm(
                input.handle.0,
                self.weight.handle.0,
                self.bias.handle.0,
                self.eps as f32,
            )
        };
        MxArray::from_handle(handle, "fast_layer_norm")
    }
}

impl Clone for RMSNorm {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            eps: self.eps,
        }
    }
}
