use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// ============================================
// Normalization Layers
// ============================================

pub struct RMSNorm {
    weight: MxArray,
    eps: f64,
}

impl RMSNorm {
    /// Create a new RMSNorm layer
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
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_fast_rms_norm(input.handle.0, self.weight.handle.0, self.eps as f32)
        };
        MxArray::from_handle(handle, "fast_rms_norm")
    }

    /// Get the weight (scale) parameter
    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    /// Set the weight (scale) parameter
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

pub struct LayerNorm {
    weight: MxArray,
    bias: MxArray,
    eps: f64,
}

impl LayerNorm {
    /// Create a new LayerNorm layer
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

impl RMSNorm {
    /// Create an RMSNorm layer from pre-loaded weight
    ///
    /// # Arguments
    /// * `weight` - Scale parameter [dims]
    /// * `eps` - Small constant for numerical stability
    pub fn from_weight(weight: &MxArray, eps: Option<f64>) -> Result<Self> {
        let shape = weight.shape()?;
        if shape.len() != 1 {
            return Err(Error::from_reason(format!(
                "RMSNorm weight must be 1D, got shape {:?}",
                shape.as_ref()
            )));
        }

        Ok(Self {
            weight: weight.clone(),
            eps: eps.unwrap_or(1e-5),
        })
    }
}

impl Clone for LayerNorm {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            eps: self.eps,
        }
    }
}

impl LayerNorm {
    /// Create a LayerNorm layer from pre-loaded weights
    ///
    /// # Arguments
    /// * `weight` - Scale parameter [dims]
    /// * `bias` - Bias parameter [dims] (optional, defaults to zeros)
    /// * `eps` - Small constant for numerical stability
    pub fn from_weights(
        weight: &MxArray,
        bias: Option<&MxArray>,
        eps: Option<f64>,
    ) -> Result<Self> {
        let shape = weight.shape()?;
        if shape.len() != 1 {
            return Err(Error::from_reason(format!(
                "LayerNorm weight must be 1D, got shape {:?}",
                shape.as_ref()
            )));
        }

        let bias_arr = if let Some(b) = bias {
            let bias_shape = b.shape()?;
            if bias_shape.as_ref() != shape.as_ref() {
                return Err(Error::from_reason(format!(
                    "LayerNorm bias shape {:?} must match weight shape {:?}",
                    bias_shape.as_ref(),
                    shape.as_ref()
                )));
            }
            b.clone()
        } else {
            MxArray::zeros(&shape, None)?
        };

        Ok(Self {
            weight: weight.clone(),
            bias: bias_arr,
            eps: eps.unwrap_or(1e-5),
        })
    }
}
