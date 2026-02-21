//! PP-LCNet_x1_0 Document Orientation Model
//!
//! NAPI class providing document orientation classification.
//! Architecture: Stem Conv -> DepthwiseSeparable blocks (2-6) -> AdaptiveAvgPool -> FC head
//!
//! All operations use NHWC format (MLX native).

use crate::array::MxArray;
use crate::models::pp_doclayout_v3::backbone::{FrozenBatchNorm2d, NativeConv2d};
use crate::nn::activations::Activations;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

use super::processing;

// ============================================================================
// ConvBNLayer: Conv2d + BatchNorm + HardSwish
// ============================================================================

/// Conv + BatchNorm + HardSwish layer (PP-LCNet building block).
pub struct ConvBNLayer {
    conv: NativeConv2d,
    bn: FrozenBatchNorm2d,
}

impl ConvBNLayer {
    pub fn new(conv: NativeConv2d, bn: FrozenBatchNorm2d) -> Self {
        Self { conv, bn }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let x = self.conv.forward(input)?;
        let x = self.bn.forward(&x)?;
        Activations::hard_swish(&x)
    }
}

// ============================================================================
// SEModule: Squeeze-and-Excitation
// ============================================================================

/// Squeeze-and-Excitation module.
/// AdaptiveAvgPool2D(1) -> Conv1x1(c, c/4) -> ReLU -> Conv1x1(c/4, c) -> HardSigmoid
pub struct SEModule {
    /// 1x1 conv: channels -> channels/4
    conv1: NativeConv2d,
    /// 1x1 conv: channels/4 -> channels
    conv2: NativeConv2d,
}

impl SEModule {
    pub fn new(conv1: NativeConv2d, conv2: NativeConv2d) -> Self {
        Self { conv1, conv2 }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let identity = input;

        // AdaptiveAvgPool2D(1): reduce spatial dims to 1x1
        // Input: [B, H, W, C] -> mean over H,W dims -> [B, 1, 1, C]
        // Mean over axes 1 and 2 (H and W), keeping dims
        let pooled = input.mean(Some(&[1, 2]), Some(true))?;

        // Conv1x1 (channels -> channels/reduction)
        let x = self.conv1.forward(&pooled)?;
        // ReLU
        let x = Activations::relu(&x)?;
        // Conv1x1 (channels/reduction -> channels)
        let x = self.conv2.forward(&x)?;
        // HardSigmoid
        let x = Activations::hard_sigmoid(&x)?;

        // Scale: identity * attention weights
        identity.mul(&x)
    }
}

// ============================================================================
// DepthwiseSeparable block
// ============================================================================

/// Depthwise Separable Convolution block.
/// dw_conv (depthwise, groups=in_channels) -> [SE] -> pw_conv (pointwise, 1x1)
pub struct DepthwiseSeparable {
    dw_conv: ConvBNLayer,
    se: Option<SEModule>,
    pw_conv: ConvBNLayer,
}

impl DepthwiseSeparable {
    pub fn new(dw_conv: ConvBNLayer, se: Option<SEModule>, pw_conv: ConvBNLayer) -> Self {
        Self {
            dw_conv,
            se,
            pw_conv,
        }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = self.dw_conv.forward(input)?;
        if let Some(ref se) = self.se {
            x = se.forward(&x)?;
        }
        self.pw_conv.forward(&x)
    }
}

// ============================================================================
// PPLCNet backbone
// ============================================================================

/// PP-LCNet backbone: stem conv + blocks2-6 + optional last_conv (1x1 expansion).
pub struct PPLCNetBackbone {
    /// Initial 3x3 conv with stride 2
    conv1: ConvBNLayer,
    /// Block groups 2-6
    block_groups: Vec<Vec<DepthwiseSeparable>>,
    /// Optional 1x1 conv expanding channels (e.g. 512 -> 1280), no BN
    last_conv: Option<NativeConv2d>,
}

impl PPLCNetBackbone {
    pub fn new(
        conv1: ConvBNLayer,
        block_groups: Vec<Vec<DepthwiseSeparable>>,
        last_conv: Option<NativeConv2d>,
    ) -> Self {
        Self {
            conv1,
            block_groups,
            last_conv,
        }
    }

    /// Forward pass through the backbone.
    /// Returns the final feature map (output of blocks6, optionally expanded by last_conv).
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = self.conv1.forward(input)?;
        for group in &self.block_groups {
            for block in group {
                x = block.forward(&x)?;
            }
        }
        if let Some(ref last_conv) = self.last_conv {
            x = last_conv.forward(&x)?;
            x = Activations::hard_swish(&x)?;
        }
        Ok(x)
    }
}

// ============================================================================
// Classification head
// ============================================================================

/// Classification head: AdaptiveAvgPool2D(1) -> Flatten -> Linear -> Softmax
pub struct ClassificationHead {
    /// FC weight [in_channels, num_classes]
    weight: Arc<MxArray>,
    /// FC bias [num_classes]
    bias: Arc<MxArray>,
}

impl ClassificationHead {
    pub fn new(weight: &MxArray, bias: &MxArray) -> Self {
        Self {
            weight: Arc::new(weight.clone()),
            bias: Arc::new(bias.clone()),
        }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        // AdaptiveAvgPool2D(1): [B, H, W, C] -> mean over H,W -> [B, 1, 1, C]
        let pooled = input.mean(Some(&[1, 2]), Some(true))?;

        // Flatten: [B, 1, 1, C] -> [B, C]
        let shape = pooled.shape()?;
        let batch = shape[0];
        let channels = shape[3];
        let flat = pooled.reshape(&[batch, channels])?;

        // Linear: [B, C] @ [C, num_classes] + bias = [B, num_classes]
        // Weight is stored as [in_features, out_features] (Paddle convention)
        let logits = flat.matmul(&self.weight)?;
        let logits = logits.add(&self.bias)?;

        // Softmax
        Activations::softmax(&logits, Some(-1))
    }
}

// ============================================================================
// NAPI OrientationResult
// ============================================================================

/// Result from document orientation classification.
#[napi(object)]
pub struct OrientationResult {
    /// Detected rotation angle (0, 90, 180, or 270 degrees)
    pub angle: u32,
    /// Confidence score
    pub score: f64,
    /// Angle label as string
    pub label: String,
}

/// Result from classify_and_rotate: orientation info + corrected image bytes.
#[napi(object)]
pub struct ClassifyRotateResult {
    /// Detected rotation angle (0, 90, 180, or 270 degrees)
    pub angle: u32,
    /// Confidence score
    pub score: f64,
    /// Angle label as string
    pub label: String,
    /// Corrected image as PNG bytes (or original bytes if angle=0)
    pub image: Buffer,
}

// ============================================================================
// NAPI DocOrientationModel
// ============================================================================

/// PP-LCNet_x1_0 Document Orientation Classification model.
///
/// Classifies document images into 4 orientation classes (0/90/180/270 degrees).
/// Uses depthwise separable convolutions with HardSwish activation.
#[napi(js_name = "DocOrientationModel")]
pub struct DocOrientationModel {
    backbone: PPLCNetBackbone,
    head: ClassificationHead,
}

/// The 4 orientation labels in order
const ORIENTATION_LABELS: [u32; 4] = [0, 90, 180, 270];

#[napi]
impl DocOrientationModel {
    /// Load a DocOrientationModel from a directory containing model.safetensors and config.json.
    #[napi(factory)]
    pub fn load(model_path: String) -> Result<Self> {
        super::persistence::load_model(&model_path)
    }

    /// Classify the orientation of a document image.
    ///
    /// Returns the detected orientation angle (0, 90, 180, 270) and confidence.
    #[napi]
    pub fn classify(&self, image_data: Buffer) -> Result<OrientationResult> {
        self.classify_bytes(&image_data)
    }

    /// Classify orientation and return the corrected (upright) image bytes.
    ///
    /// Returns classification result plus corrected PNG image bytes.
    #[napi]
    pub fn classify_and_rotate(&self, image_data: Buffer) -> Result<ClassifyRotateResult> {
        let result = self.classify_bytes(&image_data)?;

        let image = if result.angle != 0 {
            let correction = processing::correction_angle(result.angle);
            processing::rotate(&image_data, correction)?
        } else {
            image_data.to_vec()
        };

        Ok(ClassifyRotateResult {
            angle: result.angle,
            score: result.score,
            label: result.label,
            image: image.into(),
        })
    }
}

impl DocOrientationModel {
    fn classify_bytes(&self, data: &[u8]) -> Result<OrientationResult> {
        let pixel_values = processing::preprocess(data)?;
        let probs = self.forward(&pixel_values)?;

        probs.eval();
        let prob_data = probs.to_float32()?;
        let prob_vec: Vec<f32> = prob_data.to_vec();

        let (max_idx, max_score) = prob_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let angle = *ORIENTATION_LABELS.get(max_idx).ok_or_else(|| {
            Error::from_reason(format!(
                "Model output has {} classes but expected {}",
                prob_vec.len(),
                ORIENTATION_LABELS.len()
            ))
        })?;

        Ok(OrientationResult {
            angle,
            score: *max_score as f64,
            label: angle.to_string(),
        })
    }

    fn forward(&self, pixel_values: &MxArray) -> Result<MxArray> {
        let features = self.backbone.forward(pixel_values)?;
        self.head.forward(&features)
    }
}

/// Create an initialized DocOrientationModel (called from persistence).
pub(super) fn create_model(
    backbone: PPLCNetBackbone,
    head: ClassificationHead,
) -> DocOrientationModel {
    DocOrientationModel { backbone, head }
}
