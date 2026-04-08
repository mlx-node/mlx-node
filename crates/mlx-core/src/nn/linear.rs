use crate::array::MxArray;
use napi::bindgen_prelude::*;

// ============================================
// Linear Layer (supports optional quantized backend)
// ============================================

/// Quantized weight storage for Linear.
struct QuantizedBackend {
    weight: MxArray,         // Packed uint32 [out, in_packed]
    scales: MxArray,         // Quantization scales
    biases: Option<MxArray>, // Quantization biases (affine mode)
    group_size: i32,
    bits: i32,
}

pub struct Linear {
    weight: MxArray,
    /// Pre-transposed weight [in_features, out_features] for efficient matmul.
    /// Avoids creating a transpose graph node on every forward() call.
    weight_t: MxArray,
    bias: Option<MxArray>,
    in_features: u32,
    out_features: u32,
    /// When set, `forward()` uses quantized_matmul instead of plain matmul.
    quantized: Option<QuantizedBackend>,
}

impl Linear {
    /// Create a new Linear layer
    pub fn new(in_features: u32, out_features: u32, use_bias: Option<bool>) -> Result<Self> {
        // Initialize weight with Xavier/Glorot uniform initialization
        let scale = (6.0 / (in_features + out_features) as f64).sqrt();

        // Create weight matrix [out_features, in_features]
        let weight_shape = [out_features as i64, in_features as i64];
        let weight = MxArray::random_uniform(&weight_shape, -scale, scale, None)?;
        let weight_t = weight.transpose(Some(&[1, 0]))?;

        // Create bias if needed
        let bias = if use_bias.unwrap_or(true) {
            let bias_shape = [out_features as i64];
            Some(MxArray::zeros(&bias_shape, None)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            weight_t,
            bias,
            in_features,
            out_features,
            quantized: None,
        })
    }

    /// Forward pass: y = xW^T + b
    /// When quantized, uses fused dequantize+matmul Metal kernel.
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        if let Some(ref q) = self.quantized {
            let mode_c = c"affine";
            let biases_ptr = q
                .biases
                .as_ref()
                .map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());

            let handle = unsafe {
                mlx_sys::mlx_quantized_matmul(
                    input.as_raw_ptr(),
                    q.weight.as_raw_ptr(),
                    q.scales.as_raw_ptr(),
                    biases_ptr,
                    true, // transpose
                    q.group_size,
                    q.bits,
                    mode_c.as_ptr(),
                )
            };
            let mut result = MxArray::from_handle(handle, "quantized_linear_forward")?;

            if let Some(ref b) = self.bias {
                result = result.add(b)?;
            }
            Ok(result)
        } else if let Some(ref b) = self.bias {
            input.addmm(b, &self.weight_t, None, None)
        } else {
            input.matmul(&self.weight_t)
        }
    }

    /// Set new weights (dense bf16)
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
        self.weight_t = weight.transpose(Some(&[1, 0]))?;
        self.weight = weight.clone();
        self.quantized = None;
        Ok(())
    }

    /// Load quantized weights. `forward()` will use quantized_matmul.
    /// The dense `weight` is lazily dequantized for `get_weight()`.
    pub fn load_quantized(
        &mut self,
        weight: &MxArray,
        scales: &MxArray,
        biases: Option<&MxArray>,
        group_size: i32,
        bits: i32,
    ) -> Result<()> {
        // Verify out_features matches
        if weight.shape_at(0)? != self.out_features as i64 {
            return Err(Error::from_reason(format!(
                "Quantized weight out_features mismatch: expected {}, got {}",
                self.out_features,
                weight.shape_at(0)?
            )));
        }

        // Dequantize for get_weight() (used by tied embeddings path)
        let biases_ptr = biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());
        let handle = unsafe {
            mlx_sys::mlx_dequantize(
                weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases_ptr,
                group_size,
                bits,
                -1,
                c"affine".as_ptr(),
            )
        };
        self.weight = MxArray::from_handle(handle, "dequantize_linear")?;

        self.quantized = Some(QuantizedBackend {
            weight: weight.clone(),
            scales: scales.clone(),
            biases: biases.cloned(),
            group_size,
            bits,
        });
        Ok(())
    }

    /// Set new bias
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
            self.bias = Some(b.copy()?);
        } else {
            self.bias = None;
        }
        Ok(())
    }

    /// Get the weight matrix (always dense bf16)
    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    /// Get the bias vector (if present)
    pub fn get_bias(&self) -> Option<MxArray> {
        self.bias.clone()
    }

    /// Whether this linear layer uses quantized weights
    pub fn is_quantized(&self) -> bool {
        self.quantized.is_some()
    }
}

impl Clone for Linear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            weight_t: self.weight_t.clone(),
            bias: self.bias.clone(),
            in_features: self.in_features,
            out_features: self.out_features,
            quantized: self.quantized.as_ref().map(|q| QuantizedBackend {
                weight: q.weight.clone(),
                scales: q.scales.clone(),
                biases: q.biases.clone(),
                group_size: q.group_size,
                bits: q.bits,
            }),
        }
    }
}

impl Linear {
    /// Create a Linear layer from pre-loaded weights
    pub fn from_weights(weight: &MxArray, bias: Option<&MxArray>) -> Result<Self> {
        let shape = weight.shape()?;
        if shape.len() != 2 {
            return Err(Error::from_reason(format!(
                "Linear weight must be 2D, got shape {:?}",
                shape.as_ref()
            )));
        }

        let out_features = shape[0] as u32;
        let in_features = shape[1] as u32;

        Ok(Self {
            weight_t: weight.transpose(Some(&[1, 0]))?,
            weight: weight.clone(),
            bias: bias.cloned(),
            in_features,
            out_features,
            quantized: None,
        })
    }
}
