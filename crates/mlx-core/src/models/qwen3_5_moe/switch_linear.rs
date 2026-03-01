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
    pub fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        let weight_t = self.weight.transpose(Some(&[0, 2, 1]))?;

        let handle = unsafe {
            sys::mlx_gather_mm(
                x.handle.0,
                weight_t.handle.0,
                std::ptr::null_mut(),
                indices.handle.0,
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
