//! UVDoc Document Unwarping Model
//!
//! Encoder-decoder CNN that predicts a 2D displacement field for correcting
//! perspective distortion. Uses residual blocks with dilated convolutions and
//! a multi-scale bridge module.
//!
//! All operations use NHWC format (MLX native).

use crate::array::MxArray;
use crate::models::pp_doclayout_v3::backbone::{FrozenBatchNorm2d, NativeConv2d};
use crate::nn::activations::Activations;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::config::UVDocConfig;
use super::processing;

// ============================================================================
// Building blocks
// ============================================================================

/// Conv + BN + ReLU block (used in head and bridge)
pub struct ConvBNRelu {
    conv: NativeConv2d,
    bn: FrozenBatchNorm2d,
}

impl ConvBNRelu {
    pub fn new(conv: NativeConv2d, bn: FrozenBatchNorm2d) -> Self {
        Self { conv, bn }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let x = self.conv.forward(input)?;
        let x = self.bn.forward(&x)?;
        Activations::relu(&x)
    }
}

/// Residual block with optional dilation.
///
/// When stride != 1 or is_top: uses standard conv3x3 with given stride
/// Otherwise: uses dilated conv with dilation=3
pub struct ResidualBlock {
    conv1: NativeConv2d,
    bn1: FrozenBatchNorm2d,
    conv2: NativeConv2d,
    bn2: FrozenBatchNorm2d,
    downsample: Option<(NativeConv2d, FrozenBatchNorm2d)>,
}

impl ResidualBlock {
    pub fn new(
        conv1: NativeConv2d,
        bn1: FrozenBatchNorm2d,
        conv2: NativeConv2d,
        bn2: FrozenBatchNorm2d,
        downsample: Option<(NativeConv2d, FrozenBatchNorm2d)>,
    ) -> Self {
        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let residual = if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            let x = ds_conv.forward(input)?;
            ds_bn.forward(&x)?
        } else {
            input.clone()
        };

        let out = self.conv1.forward(input)?;
        let out = self.bn1.forward(&out)?;
        let out = Activations::relu(&out)?;

        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward(&out)?;

        let out = out.add(&residual)?;
        Activations::relu(&out)
    }
}

/// A layer of residual blocks (ResnetStraight layer)
pub struct ResLayer {
    blocks: Vec<ResidualBlock>,
}

impl ResLayer {
    pub fn new(blocks: Vec<ResidualBlock>) -> Self {
        Self { blocks }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = input.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

/// ResnetStraight: 3 layers of residual blocks
pub struct ResnetEncoder {
    layers: Vec<ResLayer>,
}

impl ResnetEncoder {
    pub fn new(layers: Vec<ResLayer>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

/// Bridge module: multi-scale dilated convolutions
/// bridge_1 through bridge_6 with different dilation patterns, concatenated + 1x1 conv
pub struct Bridge {
    /// bridge_1: single dilated conv (dilation=1)
    bridge_1: Vec<ConvBNRelu>,
    /// bridge_2: single dilated conv (dilation=2)
    bridge_2: Vec<ConvBNRelu>,
    /// bridge_3: single dilated conv (dilation=5)
    bridge_3: Vec<ConvBNRelu>,
    /// bridge_4: 3 dilated convs (dilation=8,3,2)
    bridge_4: Vec<ConvBNRelu>,
    /// bridge_5: 3 dilated convs (dilation=12,7,4)
    bridge_5: Vec<ConvBNRelu>,
    /// bridge_6: 3 dilated convs (dilation=18,12,6)
    bridge_6: Vec<ConvBNRelu>,
    /// 1x1 conv to reduce concatenated channels
    bridge_concat: ConvBNRelu,
}

impl Bridge {
    pub fn new(
        bridge_1: Vec<ConvBNRelu>,
        bridge_2: Vec<ConvBNRelu>,
        bridge_3: Vec<ConvBNRelu>,
        bridge_4: Vec<ConvBNRelu>,
        bridge_5: Vec<ConvBNRelu>,
        bridge_6: Vec<ConvBNRelu>,
        bridge_concat: ConvBNRelu,
    ) -> Self {
        Self {
            bridge_1,
            bridge_2,
            bridge_3,
            bridge_4,
            bridge_5,
            bridge_6,
            bridge_concat,
        }
    }

    fn forward_branch(branch: &[ConvBNRelu], input: &MxArray) -> Result<MxArray> {
        let mut x = input.clone();
        for layer in branch {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let b1 = Self::forward_branch(&self.bridge_1, input)?;
        let b2 = Self::forward_branch(&self.bridge_2, input)?;
        let b3 = Self::forward_branch(&self.bridge_3, input)?;
        let b4 = Self::forward_branch(&self.bridge_4, input)?;
        let b5 = Self::forward_branch(&self.bridge_5, input)?;
        let b6 = Self::forward_branch(&self.bridge_6, input)?;

        // Concatenate along channel axis (axis=3 in NHWC)
        let refs: Vec<&MxArray> = vec![&b1, &b2, &b3, &b4, &b5, &b6];
        let concatenated = MxArray::concatenate_many(refs, Some(3))?;

        // 1x1 conv to reduce
        self.bridge_concat.forward(&concatenated)
    }
}

/// Output head: Conv + BN + PReLU + Conv (outputs 2-channel displacement field)
///
/// Uses reflect padding (padding_mode="reflect") before each convolution,
/// matching the original UVDoc PyTorch implementation.
pub struct OutputHead {
    /// Conv with padding=0 (reflect padding applied manually before)
    conv1: NativeConv2d,
    bn: FrozenBatchNorm2d,
    /// PReLU learnable parameter [channels]
    prelu_weight: MxArray,
    /// Conv with padding=0 (reflect padding applied manually before)
    conv2: NativeConv2d,
    /// Padding amount for reflect padding (kernel_size / 2)
    pad: i64,
}

impl OutputHead {
    pub fn new(
        conv1: NativeConv2d,
        bn: FrozenBatchNorm2d,
        prelu_weight: MxArray,
        conv2: NativeConv2d,
        pad: i64,
    ) -> Self {
        Self {
            conv1,
            bn,
            prelu_weight,
            conv2,
            pad,
        }
    }

    /// PReLU: max(0, x) + weight * min(0, x)
    fn prelu(&self, input: &MxArray) -> Result<MxArray> {
        let pos = Activations::relu(input)?;
        let neg = {
            let zero = MxArray::scalar_float(0.0)?;
            let min_x = input.minimum(&zero)?;
            min_x.mul(&self.prelu_weight)?
        };
        pos.add(&neg)
    }

    /// Apply 2D reflect padding to NHWC tensor along H and W axes.
    ///
    /// For padding p, reflects p rows/cols at each boundary:
    ///   [a₂, a₁, a₀, a₁, ..., aₙ₋₁, aₙ₋₂, aₙ₋₃]
    fn reflect_pad_2d(input: &MxArray, pad: i64) -> Result<MxArray> {
        if pad == 0 {
            return Ok(input.clone());
        }

        // Pad height (axis 1)
        let mut parts_h: Vec<MxArray> = Vec::with_capacity(2 * pad as usize + 1);
        // Top: reflect rows [pad, pad-1, ..., 1]
        for i in (1..=pad).rev() {
            parts_h.push(input.slice_axis(1, i, i + 1)?);
        }
        parts_h.push(input.clone());
        // Bottom: reflect rows [H-2, H-3, ..., H-pad-1]
        let h = input.shape_at(1)?;
        for i in (h - pad - 1..h - 1).rev() {
            parts_h.push(input.slice_axis(1, i, i + 1)?);
        }
        let refs_h: Vec<&MxArray> = parts_h.iter().collect();
        let x = MxArray::concatenate_many(refs_h, Some(1))?;

        // Pad width (axis 2)
        let w = x.shape_at(2)?;
        let mut parts_w: Vec<MxArray> = Vec::with_capacity(2 * pad as usize + 1);
        // Left: reflect cols [pad, pad-1, ..., 1]
        for i in (1..=pad).rev() {
            parts_w.push(x.slice_axis(2, i, i + 1)?);
        }
        // Right: reflect cols [W-2, W-3, ..., W-pad-1]
        let mut right_parts: Vec<MxArray> = Vec::with_capacity(pad as usize);
        for i in (w - pad - 1..w - 1).rev() {
            right_parts.push(x.slice_axis(2, i, i + 1)?);
        }
        parts_w.push(x);
        parts_w.extend(right_parts);
        let refs_w: Vec<&MxArray> = parts_w.iter().collect();
        MxArray::concatenate_many(refs_w, Some(2))
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let x = Self::reflect_pad_2d(input, self.pad)?;
        let x = self.conv1.forward(&x)?;
        let x = self.bn.forward(&x)?;
        let x = self.prelu(&x)?;
        let x = Self::reflect_pad_2d(&x, self.pad)?;
        self.conv2.forward(&x)
    }
}

// ============================================================================
// Full UVDoc model
// ============================================================================

/// UVDoc document unwarping model.
pub struct UVDocNet {
    /// Initial head: 2x (Conv5x5/s2 + BN + ReLU)
    head: Vec<ConvBNRelu>,
    /// ResNet encoder with dilated residual blocks
    encoder: ResnetEncoder,
    /// Multi-scale dilated bridge
    bridge: Bridge,
    /// 2D output head (displacement field)
    out_2d: OutputHead,
}

impl UVDocNet {
    pub fn new(
        head: Vec<ConvBNRelu>,
        encoder: ResnetEncoder,
        bridge: Bridge,
        out_2d: OutputHead,
    ) -> Self {
        Self {
            head,
            encoder,
            bridge,
            out_2d,
        }
    }

    /// Forward pass: returns displacement field [1, H', W', 2] in NHWC.
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        // Head: 4x spatial downsample
        let mut x = input.clone();
        for layer in &self.head {
            x = layer.forward(&x)?;
        }

        // Encoder
        x = self.encoder.forward(&x)?;

        // Bridge
        x = self.bridge.forward(&x)?;

        // Output: 2D displacement field
        self.out_2d.forward(&x)
    }
}

// ============================================================================
// NAPI types
// ============================================================================

/// Result from document unwarping.
#[napi(object)]
pub struct UnwarpResult {
    /// Unwarped image as PNG bytes
    pub image: Buffer,
}

/// UVDoc Document Unwarping model.
///
/// Predicts a 2D displacement field and applies it to correct perspective
/// distortion in camera-captured documents.
#[napi(js_name = "DocUnwarpModel")]
pub struct DocUnwarpModel {
    net: UVDocNet,
    config: UVDocConfig,
}

#[napi]
impl DocUnwarpModel {
    /// Load a DocUnwarpModel from a directory containing model.safetensors.
    #[napi(factory)]
    pub fn load(model_path: String) -> Result<Self> {
        super::persistence::load_model(&model_path)
    }

    /// Unwarp a document image and return the corrected image bytes.
    #[napi]
    pub fn unwarp(&self, image_data: Buffer) -> Result<UnwarpResult> {
        // Preprocess
        let (pixel_values, _orig_w, _orig_h) = processing::preprocess(&image_data, &self.config)?;

        // Forward pass - get displacement field [1, H', W', 2] in NHWC
        let displacement = self.net.forward(&pixel_values)?;

        displacement.eval();
        let shape = displacement.shape()?;
        let grid_h = shape[1] as usize;
        let grid_w = shape[2] as usize;

        let grid_data = displacement.to_float32()?;
        let grid_vec: Vec<f32> = grid_data.to_vec();

        // Apply displacement field to original image
        let image_bytes =
            processing::apply_displacement_field(&image_data, &grid_vec, grid_h, grid_w)?;

        Ok(UnwarpResult {
            image: image_bytes.into(),
        })
    }
}

/// Create an initialized DocUnwarpModel (called from persistence).
pub(super) fn create_model(net: UVDocNet, config: UVDocConfig) -> DocUnwarpModel {
    DocUnwarpModel { net, config }
}
