use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ============================================
// Linear Layer
// ============================================

#[napi]
pub struct Linear {
    weight: MxArray,
    bias: Option<MxArray>,
    in_features: u32,
    out_features: u32,
}

#[napi]
impl Linear {
    /// Create a new Linear layer
    #[napi(constructor)]
    pub fn new(in_features: u32, out_features: u32, use_bias: Option<bool>) -> Result<Self> {
        // Initialize weight with Xavier/Glorot uniform initialization
        let scale = (6.0 / (in_features + out_features) as f64).sqrt();

        // Create weight matrix [out_features, in_features]
        let weight_shape = [out_features as i64, in_features as i64];
        let weight = MxArray::random_uniform(&weight_shape, -scale, scale, None)?;

        // Create bias if needed
        let bias = if use_bias.unwrap_or(true) {
            let bias_shape = [out_features as i64];
            Some(MxArray::zeros(&bias_shape, None)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Forward pass: y = xW^T + b
    /// Uses fused addmm operation when bias is present for better performance
    #[napi]
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        // For 2D arrays, transpose swaps axes 0 and 1
        let axes = [1, 0];
        let weight_t = self.weight.transpose(Some(&axes))?;

        // Use fused addmm when bias is present: bias + input @ weight.T
        // This is more efficient than separate matmul and add operations
        if let Some(ref b) = self.bias {
            input.addmm(b, &weight_t, None, None)
        } else {
            input.matmul(&weight_t)
        }
    }

    /// Set new weights
    #[napi]
    pub fn set_weight(&mut self, weight: &MxArray) -> Result<()> {
        let ndim = weight.ndim()?;
        if ndim != 2
            || weight.shape_at(0)? != self.out_features as i64
            || weight.shape_at(1)? != self.in_features as i64
        {
            return Err(Error::from_reason(format!(
                "Weight shape mismatch: expected [{}, {}], got {:?}",
                self.out_features,
                self.in_features,
                weight.shape()?.as_ref()
            )));
        }
        // Clone the Arc reference (no need to copy the underlying MLX array)
        self.weight = weight.clone();
        Ok(())
    }

    /// Set new bias
    #[napi]
    pub fn set_bias(&mut self, bias: Option<&MxArray>) -> Result<()> {
        if let Some(b) = bias {
            let ndim = b.ndim()?;
            if ndim != 1 || b.shape_at(0)? != self.out_features as i64 {
                return Err(Error::from_reason(format!(
                    "Bias shape mismatch: expected [{}], got {:?}",
                    self.out_features,
                    b.shape()?.as_ref()
                )));
            }
            // Use copy() to create a new array handle, avoiding handle aliasing
            self.bias = Some(b.copy()?);
        } else {
            self.bias = None;
        }
        Ok(())
    }

    /// Get the weight matrix
    #[napi]
    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    /// Get the bias vector (if present)
    #[napi]
    pub fn get_bias(&self) -> Option<MxArray> {
        self.bias.clone()
    }
}

impl Clone for Linear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            in_features: self.in_features,
            out_features: self.out_features,
        }
    }
}
