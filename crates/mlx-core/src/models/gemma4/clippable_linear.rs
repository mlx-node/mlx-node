use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::nn::Linear;

/// Linear layer with optional input/output clamping.
///
/// When `use_clipping=true`, applies `clip(input_min, input_max)` before
/// the matmul and `clip(output_min, output_max)` after. Clip bounds are
/// loaded as f64 scalars from checkpoint tensors.
///
/// Weight key layout when clipped:
///   `*.linear.weight`, `*.input_min`, `*.input_max`, `*.output_min`, `*.output_max`
///
/// Weight key layout when NOT clipped:
///   `*.weight` directly (standard Linear layout)
pub struct ClippableLinear {
    pub linear: Linear,
    pub use_clipping: bool,
    /// Cached clip bounds (loaded from checkpoint scalars). ±inf when unbounded.
    input_min: f64,
    input_max: f64,
    output_min: f64,
    output_max: f64,
}

impl ClippableLinear {
    pub fn new(linear: Linear, use_clipping: bool) -> Self {
        Self {
            linear,
            use_clipping,
            input_min: f64::NEG_INFINITY,
            input_max: f64::INFINITY,
            output_min: f64::NEG_INFINITY,
            output_max: f64::INFINITY,
        }
    }

    /// Set clip bounds from checkpoint scalar arrays.
    /// Each array should be a scalar (0-d or 1-element).
    pub fn set_clip_bounds(
        &mut self,
        input_min: f64,
        input_max: f64,
        output_min: f64,
        output_max: f64,
    ) {
        self.input_min = input_min;
        self.input_max = input_max;
        self.output_min = output_min;
        self.output_max = output_max;
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        if self.use_clipping {
            let x = x.clip(Some(self.input_min), Some(self.input_max))?;
            let out = self.linear.forward(&x)?;
            out.clip(Some(self.output_min), Some(self.output_max))
        } else {
            self.linear.forward(x)
        }
    }
}

impl Clone for ClippableLinear {
    fn clone(&self) -> Self {
        Self {
            linear: self.linear.clone(),
            use_clipping: self.use_clipping,
            input_min: self.input_min,
            input_max: self.input_max,
            output_min: self.output_min,
            output_max: self.output_max,
        }
    }
}
