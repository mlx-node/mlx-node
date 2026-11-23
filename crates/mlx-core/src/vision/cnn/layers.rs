//! Shared Native MLX CNN Layers (NHWC)
//!
//! Weight tensors follow MLX conventions:
//! - Conv2d / ConvTranspose2d weights: `[out_channels, kernel_h, kernel_w, in_channels/groups]`
//! - BatchNorm params: `[channels]`

use crate::array::MxArray;
use crate::nn::activations::Activations;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// Apply activation function by name.
pub fn apply_activation(input: &MxArray, activation: &str) -> Result<MxArray> {
    match activation {
        "relu" => Activations::relu(input),
        "silu" | "swish" => Activations::silu(input),
        "gelu" => Activations::gelu(input),
        "hardswish" | "hard_swish" => Activations::hard_swish(input),
        "hardsigmoid" | "hard_sigmoid" => Activations::hard_sigmoid(input),
        "none" | "identity" => Ok(input.clone()),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unsupported activation: {activation}"),
        )),
    }
}

// ============================================================================
// Frozen Batch Normalization
// ============================================================================

/// Frozen Batch Normalization layer (inference only).
///
/// Uses fixed running_mean and running_var statistics.
/// Formula: `output = (input - mean) / sqrt(var + eps) * weight + bias`.
///
/// All tensors are 1D with shape `[channels]`.
/// In NHWC format, normalization is applied to the last dimension.
pub struct FrozenBatchNorm2d {
    /// Scale parameter (gamma) `[channels]`
    weight: Arc<MxArray>,
    /// Shift parameter (beta) `[channels]`
    bias: Arc<MxArray>,
    /// Running mean `[channels]`
    running_mean: Arc<MxArray>,
    /// Running variance `[channels]`
    running_var: Arc<MxArray>,
    /// Epsilon for numerical stability
    eps: f64,
}

impl FrozenBatchNorm2d {
    pub fn new(
        weight: &MxArray,
        bias: &MxArray,
        running_mean: &MxArray,
        running_var: &MxArray,
        eps: f64,
    ) -> Self {
        Self {
            weight: Arc::new(weight.clone()),
            bias: Arc::new(bias.clone()),
            running_mean: Arc::new(running_mean.clone()),
            running_var: Arc::new(running_var.clone()),
            eps,
        }
    }

    /// Forward pass: normalize input using frozen statistics.
    ///
    /// Input/Output shape: `[batch, height, width, channels]` (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let centered = input.sub(&self.running_mean)?;
        let var_plus_eps = self.running_var.add_scalar(self.eps)?;
        let std_dev = var_plus_eps.sqrt()?;
        let normalized = centered.div(&std_dev)?;
        let scaled = normalized.mul(&self.weight)?;
        scaled.add(&self.bias)
    }
}

// ============================================================================
// Conv2d (native MLX)
// ============================================================================

/// 2D Convolution layer using MLX's native `conv2d` in NHWC format.
///
/// Weight shape: `[out_channels, kernel_h, kernel_w, in_channels/groups]`
pub struct NativeConv2d {
    weight: Arc<MxArray>,
    bias: Option<Arc<MxArray>>,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
}

impl NativeConv2d {
    pub fn new(
        weight: &MxArray,
        bias: Option<&MxArray>,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        groups: i32,
    ) -> Self {
        Self {
            weight: Arc::new(weight.clone()),
            bias: bias.map(|b| Arc::new(b.clone())),
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Forward pass: `[batch, height, width, in_channels]` -> `[batch, out_h, out_w, out_channels]`
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_conv2d(
                input.handle.0,
                self.weight.handle.0,
                self.stride.0,
                self.stride.1,
                self.padding.0,
                self.padding.1,
                self.dilation.0,
                self.dilation.1,
                self.groups,
            )
        };
        let mut output = MxArray::from_handle(handle, "conv2d")?;

        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }
}

// ============================================================================
// ConvTranspose2d (native MLX)
// ============================================================================

/// 2D Transposed Convolution layer using MLX's native `conv_transpose2d` in NHWC format.
///
/// Weight shape: `[out_channels, kernel_h, kernel_w, in_channels/groups]`.
/// Note: The underlying FFI binding hardcodes `output_padding` to `(0, 0)`.
pub struct NativeConvTranspose2d {
    weight: Arc<MxArray>,
    bias: Option<Arc<MxArray>>,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
}

impl NativeConvTranspose2d {
    pub fn new(
        weight: &MxArray,
        bias: Option<&MxArray>,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        groups: i32,
    ) -> Self {
        Self {
            weight: Arc::new(weight.clone()),
            bias: bias.map(|b| Arc::new(b.clone())),
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Forward pass: `[batch, height, width, in_channels]` -> `[batch, out_h, out_w, out_channels]`
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_conv_transpose2d(
                input.handle.0,
                self.weight.handle.0,
                self.stride.0,
                self.stride.1,
                self.padding.0,
                self.padding.1,
                self.dilation.0,
                self.dilation.1,
                self.groups,
            )
        };
        let mut output = MxArray::from_handle(handle, "conv_transpose2d")?;

        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }
}

// ============================================================================
// Learnable Affine Block
// ============================================================================

/// Learnable Affine Block: `output = scale * input + bias`.
pub struct LearnableAffineBlock {
    scale: Arc<MxArray>,
    bias: Arc<MxArray>,
}

impl LearnableAffineBlock {
    pub fn new(scale: &MxArray, bias: &MxArray) -> Self {
        Self {
            scale: Arc::new(scale.clone()),
            bias: Arc::new(bias.clone()),
        }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let scaled = input.mul(&self.scale)?;
        scaled.add(&self.bias)
    }
}

// ============================================================================
// ConvBNAct: Conv2d + BatchNorm + Activation (+ optional LAB)
// ============================================================================

/// Convolution + Frozen Batch Normalization + Activation layer (+ optional LAB).
///
/// Unified layer replacing `ConvNormLayer`, `ConvLayer`, `ConvBNLayer`,
/// `ConvBNRelu`, and `SVTRConvBN` across Paddle vision models.
/// Pass `"none"` (or `"identity"`) for no activation.
pub struct ConvBNAct {
    conv: NativeConv2d,
    norm: FrozenBatchNorm2d,
    activation: String,
    lab: Option<LearnableAffineBlock>,
}

impl ConvBNAct {
    pub fn new(
        conv: NativeConv2d,
        norm: FrozenBatchNorm2d,
        activation: &str,
        lab: Option<LearnableAffineBlock>,
    ) -> Self {
        Self {
            conv,
            norm,
            activation: activation.to_string(),
            lab,
        }
    }

    /// Forward pass: Conv -> BN -> Activation -> (optional LAB)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = self.conv.forward(input)?;
        x = self.norm.forward(&x)?;
        x = apply_activation(&x, &self.activation)?;
        if let Some(ref lab) = self.lab {
            x = lab.forward(&x)?;
        }
        Ok(x)
    }
}

// ============================================================================
// ConvBNActLight: Depthwise separable convolution
// ============================================================================

/// Light (depthwise separable) convolution block:
/// 1x1 pointwise conv (no activation) -> kxk depthwise conv with activation.
pub struct ConvBNActLight {
    conv1: ConvBNAct,
    conv2: ConvBNAct,
}

impl ConvBNActLight {
    pub fn new(conv1: ConvBNAct, conv2: ConvBNAct) -> Self {
        Self { conv1, conv2 }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let x = self.conv1.forward(input)?;
        self.conv2.forward(&x)
    }
}

// ============================================================================
// MaxPool2d (pure Rust implementation for MLX NHWC)
// ============================================================================

/// Max pooling implementation for NHWC format.
pub struct MaxPool2d {
    kernel_size: i32,
    stride: i32,
    ceil_mode: bool,
}

impl MaxPool2d {
    pub fn new(kernel_size: i32, stride: i32, ceil_mode: bool) -> Self {
        Self {
            kernel_size,
            stride,
            ceil_mode,
        }
    }

    /// Forward pass: `[batch, height, width, channels]` -> `[batch, out_h, out_w, channels]`
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let shape = input.shape()?;
        let shape: Vec<i64> = shape.as_ref().to_vec();
        if shape.len() != 4 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("MaxPool2d input must be 4D [N,H,W,C], got {}D", shape.len()),
            ));
        }

        let batch = shape[0];
        let in_h = shape[1];
        let in_w = shape[2];
        let channels = shape[3];

        let k = self.kernel_size as i64;
        let s = self.stride as i64;

        let out_h = if self.ceil_mode {
            (in_h - k + s - 1) / s + 1
        } else {
            (in_h - k) / s + 1
        };
        let out_w = if self.ceil_mode {
            (in_w - k + s - 1) / s + 1
        } else {
            (in_w - k) / s + 1
        };

        let pad_h = (out_h - 1) * s + k - in_h;
        let pad_w = (out_w - 1) * s + k - in_w;

        let padded = if pad_h > 0 || pad_w > 0 {
            let pad_width = vec![0i32, 0, 0, pad_h as i32, 0, pad_w as i32, 0, 0];
            input.pad(&pad_width, f64::NEG_INFINITY)?
        } else {
            input.clone()
        };

        if k == 2 && s == 1 {
            let tl = padded.slice(&[0, 0, 0, 0], &[batch, out_h, out_w, channels])?;
            let tr = padded.slice(&[0, 0, 1, 0], &[batch, out_h, out_w + 1, channels])?;
            let bl = padded.slice(&[0, 1, 0, 0], &[batch, out_h + 1, out_w, channels])?;
            let br = padded.slice(&[0, 1, 1, 0], &[batch, out_h + 1, out_w + 1, channels])?;

            let max_top = tl.maximum(&tr)?;
            let max_bot = bl.maximum(&br)?;
            return max_top.maximum(&max_bot);
        }

        let mut result: Option<MxArray> = None;
        for kh in 0..k {
            for kw in 0..k {
                let window = padded.slice(
                    &[0, kh, kw, 0],
                    &[batch, kh + out_h * s, kw + out_w * s, channels],
                )?;

                if s > 1 {
                    return Err(Error::new(
                        Status::GenericFailure,
                        "MaxPool2d with stride > 1 and kernel > 2 not yet optimized",
                    ));
                }

                result = Some(match result {
                    None => window,
                    Some(prev) => prev.maximum(&window)?,
                });
            }
        }

        result.ok_or_else(|| Error::new(Status::GenericFailure, "Empty max pool kernel"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frozen_batch_norm() {
        let weight = MxArray::from_float32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let bias = MxArray::from_float32(&[0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
        let running_mean = MxArray::from_float32(&[0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
        let running_var = MxArray::from_float32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();

        let bn = FrozenBatchNorm2d::new(&weight, &bias, &running_mean, &running_var, 1e-5);

        let input = MxArray::from_float32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 2, 2, 4],
        )
        .unwrap();

        let output = bn.forward(&input).unwrap();
        let output_shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(output_shape, vec![1, 2, 2, 4]);
    }

    #[test]
    fn test_max_pool_2d_k2_s1_ceil() {
        let pool = MaxPool2d::new(2, 1, true);

        let input = MxArray::from_float32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 3, 3, 1],
        )
        .unwrap();

        let output = pool.forward(&input).unwrap();
        let output_shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(output_shape, vec![1, 2, 2, 1]);

        output.eval();
        let data: Vec<f32> = output.to_float32().unwrap().to_vec();
        assert_eq!(data[0], 5.0);
        assert_eq!(data[1], 6.0);
        assert_eq!(data[2], 8.0);
        assert_eq!(data[3], 9.0);
    }
}
