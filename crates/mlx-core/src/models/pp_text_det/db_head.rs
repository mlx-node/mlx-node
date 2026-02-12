//! DBHead and PFHeadLocal (Differentiable Binarization Head)
//!
//! Produces shrink map + threshold map from neck features.
//!
//! ## Head (binarize/thresh branch):
//!   Conv2d(in_ch->inner_ch, k=3, pad=1, bias=False) -> BN -> ReLU
//!   ConvTranspose2d(inner_ch->inner_ch, k=2, stride=2) -> BN -> ReLU
//!   ConvTranspose2d(inner_ch->1, k=2, stride=2) -> Sigmoid
//!
//! ## PFHeadLocal (inference):
//!   1. binarize head runs and returns both sigmoid output AND intermediate features f
//!      (after deconv1 + BN + ReLU, before deconv2 + sigmoid)
//!   2. up_conv: nearest-neighbor 2x upsample of f
//!   3. cbn_layer (LocalModule): [shrink_maps, up_conv(f)] -> conv1(BN+ReLU) -> conv2
//!   4. sigmoid(cbn_output) -> cbn_maps
//!   5. return 0.5 * (base_maps + cbn_maps)

use crate::array::MxArray;
use crate::nn::activations::Activations;
use napi::bindgen_prelude::*;

use crate::models::pp_doclayout_v3::backbone::{
    FrozenBatchNorm2d, NativeConv2d, NativeConvTranspose2d,
};

/// Head: a single branch (binarize or threshold) of the DB head.
///
/// Corresponds to PaddleOCR's `Head` class in `det_db_head.py`.
pub struct Head {
    /// Conv2d(in_ch->inner_ch, k=3, pad=1, bias=False) + BN + ReLU
    conv1: NativeConv2d,
    bn1: FrozenBatchNorm2d,
    /// ConvTranspose2d(inner_ch->inner_ch, k=2, stride=2) + BN + ReLU
    deconv1: NativeConvTranspose2d,
    bn2: FrozenBatchNorm2d,
    /// ConvTranspose2d(inner_ch->1, k=2, stride=2) -> Sigmoid
    deconv2: NativeConvTranspose2d,
}

impl Head {
    pub fn new(
        conv1: NativeConv2d,
        bn1: FrozenBatchNorm2d,
        deconv1: NativeConvTranspose2d,
        bn2: FrozenBatchNorm2d,
        deconv2: NativeConvTranspose2d,
    ) -> Self {
        Self {
            conv1,
            bn1,
            deconv1,
            bn2,
            deconv2,
        }
    }

    /// Forward pass: produces probability map.
    ///
    /// Input: [B, H, W, C] where H,W are at stride-4 relative to input image
    /// Output: [B, H*4, W*4, 1] probability map at original resolution
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        // Conv + BN + ReLU
        let x = self.conv1.forward(input)?;
        let x = self.bn1.forward(&x)?;
        let x = Activations::relu(&x)?;

        // Deconv (2x upsample) + BN + ReLU
        let x = self.deconv1.forward(&x)?;
        let x = self.bn2.forward(&x)?;
        let x = Activations::relu(&x)?;

        // Deconv (2x upsample) + Sigmoid
        let x = self.deconv2.forward(&x)?;
        Activations::sigmoid(&x)
    }

    /// Forward pass with intermediate feature return (for PFHeadLocal).
    ///
    /// Returns (sigmoid_output, intermediate_features) where intermediate_features
    /// is the output of deconv1 + BN + ReLU (before deconv2 + sigmoid).
    pub fn forward_with_features(&self, input: &MxArray) -> Result<(MxArray, MxArray)> {
        // Conv + BN + ReLU
        let x = self.conv1.forward(input)?;
        let x = self.bn1.forward(&x)?;
        let x = Activations::relu(&x)?;

        // Deconv (2x upsample) + BN + ReLU
        let x = self.deconv1.forward(&x)?;
        let x = self.bn2.forward(&x)?;
        let f = Activations::relu(&x)?;

        // Deconv (2x upsample) + Sigmoid
        let x = self.deconv2.forward(&f)?;
        let shrink_maps = Activations::sigmoid(&x)?;

        Ok((shrink_maps, f))
    }
}

/// LocalModule: refinement branch of PFHeadLocal.
///
/// Corresponds to PaddleOCR's `LocalModule` class.
///
/// Architecture:
///   Input: concatenate [shrink_maps(1ch), up_conv(f)(inner_ch)] -> (inner_channels + 1) channels
///   conv1: Conv2d(inner_ch+1 -> mid_ch, k=3, pad=1, bias=False) + BN + ReLU
///   conv2: Conv2d(mid_ch -> 1, k=1, pad=0, bias=False)
///   Output: 1-channel feature map (before sigmoid)
pub struct LocalModule {
    /// Conv2d(inner_ch+1 -> mid_ch, k=3, pad=1) + BN + ReLU
    conv1: NativeConv2d,
    bn1: FrozenBatchNorm2d,
    /// Conv2d(mid_ch -> 1, k=1, pad=0)
    conv2: NativeConv2d,
}

impl LocalModule {
    pub fn new(conv1: NativeConv2d, bn1: FrozenBatchNorm2d, conv2: NativeConv2d) -> Self {
        Self { conv1, bn1, conv2 }
    }

    /// Forward pass: refine the shrink map.
    ///
    /// # Arguments
    /// * `x` - Upsampled intermediate features [B, H, W, inner_ch]
    /// * `init_map` - Shrink map from binarize head [B, H, W, 1]
    ///
    /// # Returns
    /// * Refined feature map [B, H, W, 1] (before sigmoid)
    pub fn forward(&self, x: &MxArray, init_map: &MxArray) -> Result<MxArray> {
        // Concatenate [init_map, x] along channel axis (axis=3 in NHWC)
        let outf = MxArray::concatenate(init_map, x, 3)?;
        // Conv1 + BN + ReLU
        let out = self.conv1.forward(&outf)?;
        let out = self.bn1.forward(&out)?;
        let out = Activations::relu(&out)?;
        // Conv2
        self.conv2.forward(&out)
    }
}

/// PFHeadLocal: DB head with local refinement branch.
///
/// Corresponds to PaddleOCR's `PFHeadLocal` class.
///
/// At inference time:
/// 1. binarize head produces shrink_maps AND intermediate features f
/// 2. f is 2x nearest-upsampled
/// 3. LocalModule refines using [shrink_maps, upsample(f)]
/// 4. Returns 0.5 * (base_maps + sigmoid(cbn_output))
pub struct PFHeadLocal {
    /// Binarize (shrink map) head
    binarize: Head,
    /// CBN refinement layer
    cbn_layer: LocalModule,
}

impl PFHeadLocal {
    pub fn new(binarize: Head, cbn_layer: LocalModule) -> Self {
        Self {
            binarize,
            cbn_layer,
        }
    }

    /// Forward pass (inference only).
    ///
    /// Input: [B, H, W, C] where H,W are at stride-4 relative to input image
    /// Output: [B, H*4, W*4, 1] probability map at original resolution
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        // 1. Run binarize head with intermediate feature return
        let (shrink_maps, f) = self.binarize.forward_with_features(input)?;

        // 2. Nearest-neighbor 2x upsample of intermediate features
        let f_up = upsample_nearest_2x(&f)?;

        // 3. Run cbn_layer (LocalModule)
        let cbn_out = self.cbn_layer.forward(&f_up, &shrink_maps)?;
        let cbn_maps = Activations::sigmoid(&cbn_out)?;

        // 4. Return 0.5 * (base_maps + cbn_maps)
        let sum = shrink_maps.add(&cbn_maps)?;
        sum.mul_scalar(0.5)
    }
}

/// 2x nearest-neighbor upsample for NHWC tensors.
///
/// Input: [B, H, W, C]
/// Output: [B, H*2, W*2, C]
fn upsample_nearest_2x(input: &MxArray) -> Result<MxArray> {
    let shape = input.shape()?;
    let batch = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];

    // Reshape to [B, H, 1, W, 1, C], broadcast to [B, H, 2, W, 2, C], reshape back
    let expanded = input.reshape(&[batch, h, 1, w, 1, c])?;
    let upsampled = expanded.broadcast_to(&[batch, h, 2, w, 2, c])?;
    upsampled.reshape(&[batch, h * 2, w * 2, c])
}

// Legacy type alias for backward compatibility in persistence.rs
pub type DBHead = Head;
