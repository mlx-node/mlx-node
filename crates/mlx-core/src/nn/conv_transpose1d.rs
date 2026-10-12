use crate::array::MxArray;
use napi::{Error, Result};

/// MLX NLC transposed convolution. Weight layout is [output, kernel, input].
pub struct ConvTranspose1d {
    weight: MxArray,
    bias: Option<MxArray>,
    pub stride: i32,
    pub kernel: i32,
}

impl ConvTranspose1d {
    pub fn from_weights(weight: MxArray, bias: Option<MxArray>, stride: i32) -> Result<Self> {
        let shape = weight.shape()?;
        if shape.len() != 3 || stride <= 0 || shape[1] < i64::from(stride) {
            return Err(Error::from_reason(
                "Invalid transposed convolution geometry",
            ));
        }
        Ok(Self {
            kernel: shape[1] as i32,
            weight,
            bias,
            stride,
        })
    }

    pub fn forward_without_bias(&self, input: &MxArray) -> Result<MxArray> {
        MxArray::from_handle(
            unsafe {
                mlx_sys::mlx_conv_transpose1d(
                    input.as_raw_ptr(),
                    self.weight.as_raw_ptr(),
                    self.stride,
                    0,
                    1,
                    0,
                    1,
                )
            },
            "conv_transpose1d",
        )
    }

    pub fn add_bias(&self, output: MxArray) -> Result<MxArray> {
        match &self.bias {
            Some(bias) => output.add(bias),
            None => Ok(output),
        }
    }
}
