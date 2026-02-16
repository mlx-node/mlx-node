use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// SwitchLinear: Expert-indexed linear layer using gather_mm.
///
/// Weight shape: [num_experts, output_dims, input_dims]
/// Uses mx.gather_mm for efficient expert selection.
pub struct SwitchLinear {
    weight: MxArray, // [num_experts, output_dims, input_dims]
}

impl SwitchLinear {
    pub fn new(input_dims: u32, output_dims: u32, num_experts: u32) -> Result<Self> {
        let scale = (2.0 / input_dims as f64).sqrt();
        let weight = MxArray::random_uniform(
            &[num_experts as i64, output_dims as i64, input_dims as i64],
            -scale,
            scale,
            None,
        )?;
        Ok(Self { weight })
    }

    pub fn from_weight(weight: &MxArray) -> Self {
        Self {
            weight: weight.clone(),
        }
    }

    /// Forward pass using gather_mm.
    ///
    /// # Arguments
    /// * `x` - Input tensor [..., input_dims]
    /// * `indices` - Expert indices [...] (int32)
    /// * `sorted` - Whether indices are pre-sorted
    ///
    /// # Returns
    /// Output tensor [..., output_dims]
    pub fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        // Transpose weight: [num_experts, output_dims, input_dims] → [num_experts, input_dims, output_dims]
        let weight_t = self.weight.transpose(Some(&[0, 2, 1]))?;

        // gather_mm: x @ weight_t[indices]
        // x: [..., input_dims], rhs_indices = indices → [..., output_dims]
        let handle = unsafe {
            sys::mlx_gather_mm(
                x.handle.0,
                weight_t.handle.0,
                std::ptr::null_mut(), // no lhs indexing
                indices.handle.0,     // rhs indexing by expert
                sorted,
            )
        };
        MxArray::from_handle(handle, "switch_linear")
    }

    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    pub fn set_weight(&mut self, weight: &MxArray) {
        self.weight = weight.clone();
    }
}

impl Clone for SwitchLinear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
        }
    }
}
