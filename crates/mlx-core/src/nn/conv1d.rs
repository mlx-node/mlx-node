use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// 1D Convolution layer.
///
/// Applies a 1D convolution over an input signal composed of several input planes.
/// Used by GatedDeltaNet for depthwise convolution in Qwen3.5.
///
/// Input shape: `[batch, seq_len, in_channels]`
/// Weight shape: `[out_channels, kernel_size, in_channels/groups]`  (MLX convention after sanitization)
/// Output shape: `[batch, seq_len_out, out_channels]`
pub struct Conv1d {
    weight: MxArray,
    bias: Option<MxArray>,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
}

impl Conv1d {
    /// Create a new Conv1d layer with random initialization.
    pub fn new(
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        stride: Option<u32>,
        padding: Option<u32>,
        dilation: Option<u32>,
        groups: Option<u32>,
        use_bias: Option<bool>,
    ) -> Result<Self> {
        let groups = groups.unwrap_or(1) as i32;
        if groups <= 0 {
            return Err(Error::from_reason(format!(
                "Conv1d: groups must be > 0, got {}",
                groups
            )));
        }
        let stride = stride.unwrap_or(1) as i32;
        let padding = padding.unwrap_or(0) as i32;
        let dilation = dilation.unwrap_or(1) as i32;

        // Weight shape: [out_channels, kernel_size, in_channels/groups]
        // For depthwise conv (groups == in_channels == out_channels):
        //   weight shape = [out_channels, kernel_size, 1]
        let weight_shape = [
            out_channels as i64,
            kernel_size as i64,
            (in_channels as i64) / (groups as i64),
        ];
        let scale = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let weight = MxArray::random_uniform(&weight_shape, -scale, scale, None)?;

        let bias = if use_bias.unwrap_or(false) {
            Some(MxArray::zeros(&[out_channels as i64], None)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        })
    }

    /// Create a Conv1d layer from pre-loaded weights.
    pub fn from_weights(
        weight: &MxArray,
        bias: Option<&MxArray>,
        stride: Option<u32>,
        padding: Option<u32>,
        dilation: Option<u32>,
        groups: Option<u32>,
    ) -> Result<Self> {
        Ok(Self {
            weight: weight.clone(),
            bias: bias.cloned(),
            stride: stride.unwrap_or(1) as i32,
            padding: padding.unwrap_or(0) as i32,
            dilation: dilation.unwrap_or(1) as i32,
            groups: groups.unwrap_or(1) as i32,
        })
    }

    /// Forward pass: applies 1D convolution.
    ///
    /// Input shape: `[batch, seq_len, in_channels]`
    /// Output shape: `[batch, seq_len_out, out_channels]`
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_conv1d(
                input.handle.0,
                self.weight.handle.0,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        };
        let result = MxArray::from_handle(handle, "conv1d")?;

        if let Some(ref b) = self.bias {
            result.add(b)
        } else {
            Ok(result)
        }
    }

    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    pub fn set_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.weight = weight.clone();
        Ok(())
    }

    pub fn set_bias(&mut self, bias: Option<&MxArray>) -> Result<()> {
        self.bias = bias.cloned();
        Ok(())
    }
}

impl Clone for Conv1d {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        }
    }
}
