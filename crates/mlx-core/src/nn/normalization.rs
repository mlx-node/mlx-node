use super::activations::Activations;
use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// ============================================
// Normalization Layers
// ============================================

/// RMS normalization without a learnable weight (`weight=None` in Python).
///
/// Uses the fused `mlx_fast_rms_norm` kernel with a nullptr weight (the C++
/// side maps nullptr → `std::nullopt`), computing in f32 accumulators and
/// returning the input dtype. This is the canonical inference-path version of
/// the per-model `rms_norm_no_weight` / `scaleless_rms_norm` copies it
/// replaces; the autograd functional code keeps its own elementwise spell-out
/// so gradients flow through ordinary VJP nodes.
pub fn rms_norm_unscaled(x: &MxArray, eps: f32) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_fast_rms_norm(x.handle.0, std::ptr::null_mut(), eps) };
    MxArray::from_handle(handle, "rms_norm_unscaled")
}

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
    /// Get the weight (scale) parameter.
    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    /// Get the bias parameter
    pub fn get_bias(&self) -> MxArray {
        self.bias.clone()
    }

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

/// RMSNorm with optional SwiGLU gating.
///
/// When `gate` is provided: `swiglu(gate, rms_norm(x))`
/// When `gate` is None: `rms_norm(x)`
///
/// Lives beside `RMSNorm`/`LayerNorm`/`GroupedRMSNorm` — it has no model-
/// family dependencies (moved out of `models::qwen3_5`, which re-exports it
/// via `qwen3_5::rms_norm_gated`).
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

/// Grouped RMSNorm (K2-Horizon).
///
/// HF remote-code semantics (`K2HorizonRMSNorm`, verified against
/// `IFM/K2-Horizon-7B-FP8` modeling file):
///
/// ```python
/// def forward(self, hidden_states):
///     input_dtype = hidden_states.dtype
///     hidden_states = hidden_states.to(torch.float32)
///     hidden_states = hidden_states.reshape(*hidden_states.shape[:-1], self.num_groups, -1)
///     variance = hidden_states.pow(2).mean(-1, keepdim=True)
///     hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
///     return (hidden_states.reshape(input_shape) * self.weight).to(input_dtype)
/// ```
///
/// Unlike plain [`RMSNorm`] (one variance over the whole hidden dim), the
/// variance is computed **per group** of `hidden_size / num_groups`
/// channels, so each group's statistics are independent. With
/// `num_groups == 1` this degenerates to ordinary RMSNorm.
///
/// Uses MLX's fused RMSNorm on a grouped f32 view, followed by the
/// full-width learned scale and a cast back to the input dtype.
pub struct GroupedRMSNorm {
    weight: MxArray,
    eps: f64,
    num_groups: i64,
}

impl GroupedRMSNorm {
    /// `hidden_size` is the last-dim extent of inputs; `num_groups`
    /// partitions it. `eps` is added to the per-group variance.
    pub fn new(hidden_size: i64, num_groups: i64, eps: f64) -> Result<Self> {
        if num_groups <= 0 || hidden_size % num_groups != 0 {
            return Err(Error::from_reason(format!(
                "GroupedRMSNorm: hidden_size ({hidden_size}) must be divisible by num_groups ({num_groups})"
            )));
        }
        let weight = MxArray::ones(&[hidden_size], Some(crate::array::DType::Float32))?;
        Ok(Self {
            weight,
            eps,
            num_groups,
        })
    }

    /// x: `[..., hidden_size]` → grouped-RMS-normalized, same shape.
    ///
    /// Normalization runs in f32 (per-group variance → `x * rsqrt`); the
    /// learned `weight` multiplies the f32-normalized tensor, then the
    /// result casts back to the input dtype — matching HF's
    /// `K2HorizonRMSNorm` op-for-op. The variance/normalize core uses the
    /// fused `fast.rms_norm` kernel on the `[..., G, C]` view (its Metal
    /// kernel accumulates in f32), so a call is 4 real kernels instead of
    /// the ~9 an elementwise port needs — this fires twice per layer per
    /// decode token, so the shape is load-bearing for decode speed.
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let in_dtype = x.dtype()?;
        let shape = x.shape()?;
        let dims = shape.as_ref();
        let hidden = *dims
            .last()
            .ok_or_else(|| Error::from_reason("GroupedRMSNorm: input must have at least 1 dim"))?;
        let group_size = hidden / self.num_groups;

        // Cast to f32, reshape [..., H] → [..., G, C]. The fused kernel
        // normalizes over the last axis (per group) in f32 accumulators;
        // weight=None keeps the per-element multiply outside since the
        // binding requires a 1-D weight of size C while ours is H-sized.
        let x32 = x.astype(crate::array::DType::Float32)?;
        let mut grouped = dims.to_vec();
        let last = grouped.len() - 1;
        grouped[last] = self.num_groups;
        grouped.push(group_size);
        let xg = x32.reshape(&grouped)?;

        let handle =
            unsafe { sys::mlx_fast_rms_norm(xg.handle.0, std::ptr::null_mut(), self.eps as f32) };
        let normed = MxArray::from_handle(handle, "grouped_fast_rms_norm")?;

        // Back to [..., H], apply learned weight in f32, restore dtype.
        normed.reshape(dims)?.mul(&self.weight)?.astype(in_dtype)
    }

    /// Set the weight (scale) parameter
    pub fn set_weight(&mut self, weight: &MxArray) -> Result<()> {
        let shape = weight.shape()?;
        if shape.as_ref().len() != 1 {
            return Err(Error::from_reason(format!(
                "GroupedRMSNorm weight must be 1D, got shape {:?}",
                shape.as_ref()
            )));
        }
        self.weight = weight.astype(crate::array::DType::Float32)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    /// Reference in plain f32 element ops (no fused kernel anywhere) so the
    /// test verifies the *grouped* math, not just shape plumbing.
    fn reference_grouped_rms(
        x: &[f32],
        shape: &[i64],
        weight: &[f32],
        num_groups: i64,
        eps: f64,
    ) -> Vec<f32> {
        let hidden = *shape.last().unwrap() as usize;
        let rows = x.len() / hidden;
        let group_size = hidden / num_groups as usize;
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            for g in 0..num_groups as usize {
                let grp = &row[g * group_size..(g + 1) * group_size];
                let var = grp.iter().map(|v| v * v).sum::<f32>() / group_size as f32;
                let inv = (var as f64 + eps).sqrt().recip() as f32;
                for (i, v) in grp.iter().enumerate() {
                    let idx = g * group_size + i;
                    out[r * hidden + idx] = v * inv * weight[idx];
                }
            }
        }
        out
    }

    #[test]
    fn test_grouped_norm_matches_reference() {
        let hidden = 8i64;
        let num_groups = 2i64;
        let eps = 1e-6;
        let mut norm = GroupedRMSNorm::new(hidden, num_groups, eps).unwrap();
        let weight: Vec<f32> = (0..hidden).map(|i| 0.5 + i as f32 * 0.25).collect();
        norm.set_weight(&MxArray::from_float32(&weight, &[hidden]).unwrap())
            .unwrap();

        let x: Vec<f32> = (0..2 * 4 * hidden)
            .map(|i| (i as f32 * 0.37).sin() * 3.0)
            .collect();
        let input = MxArray::from_float32(&x, &[2, 4, hidden]).unwrap();
        let out = norm.forward(&input).unwrap();
        assert_eq!(out.shape().unwrap().as_ref(), &[2, 4, hidden]);

        let got = out.to_float32().unwrap().to_vec();
        let want = reference_grouped_rms(&x, &[2, 4, hidden], &weight, num_groups, eps);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn test_single_group_equals_plain_rmsnorm() {
        // num_groups == 1 degenerates to ordinary RMSNorm.
        let hidden = 16i64;
        let mut grouped = GroupedRMSNorm::new(hidden, 1, 1e-6).unwrap();
        let weight: Vec<f32> = vec![1.0; hidden as usize];
        grouped
            .set_weight(&MxArray::from_float32(&weight, &[hidden]).unwrap())
            .unwrap();
        let plain = RMSNorm::new(hidden as u32, Some(1e-6)).unwrap();

        let x: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.5 - 2.0).collect();
        let input = MxArray::from_float32(&x, &[hidden]).unwrap();
        let a = grouped
            .forward(&input)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        let b = plain
            .forward(&input)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "grouped {x} vs plain {y}");
        }
    }

    #[test]
    fn test_dtype_roundtrip_bf16() {
        let mut norm = GroupedRMSNorm::new(8, 4, 1e-6).unwrap();
        norm.set_weight(&MxArray::from_float32(&[1.0; 8], &[8]).unwrap())
            .unwrap();
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let input = MxArray::from_float32(&x, &[8])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let out = norm.forward(&input).unwrap();
        assert_eq!(out.dtype().unwrap(), DType::BFloat16);
        assert_eq!(out.shape().unwrap().as_ref(), &[8]);
    }
}
