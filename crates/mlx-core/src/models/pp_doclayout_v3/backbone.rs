//! HGNetV2 Backbone for PP-DocLayoutV3
//!
//! Convolutional backbone implementing the HGNetV2-L architecture.
//! Ported from the HuggingFace Transformers HGNetV2 implementation.
//!
//! Architecture overview:
//! - HGNetV2Embeddings (Stem): Initial convolutions + MaxPool to downsample 4x
//! - HGNetV2Stage x4: Each stage contains HGBlocks with feature aggregation
//! - Output: Feature maps at 4 scales [stage1, stage2, stage3, stage4]
//!
//! All operations use NHWC format (MLX native).
//! Weight tensors follow MLX conventions:
//! - Conv2d weights: [out_channels, kernel_h, kernel_w, in_channels/groups]
//! - BatchNorm params: [channels]

use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::sync::Arc;

use super::config::HGNetV2Config;

// ============================================================================
// Frozen Batch Normalization
// ============================================================================

/// Frozen Batch Normalization layer (inference only).
///
/// Uses fixed running_mean and running_var statistics.
/// Equivalent to PyTorch's FrozenBatchNorm2d or BatchNorm2d in eval mode.
///
/// Formula: output = (input - mean) / sqrt(var + eps) * weight + bias
///
/// All tensors are 1D with shape [channels].
/// In NHWC format, normalization is applied to the last dimension.
pub struct FrozenBatchNorm2d {
    /// Scale parameter (gamma) [channels]
    weight: Arc<MxArray>,
    /// Shift parameter (beta) [channels]
    bias: Arc<MxArray>,
    /// Running mean [channels]
    running_mean: Arc<MxArray>,
    /// Running variance [channels]
    running_var: Arc<MxArray>,
    /// Epsilon for numerical stability
    eps: f64,
}

impl FrozenBatchNorm2d {
    /// Create a new FrozenBatchNorm2d layer.
    ///
    /// # Arguments
    /// * `weight` - Scale parameter (gamma) [channels]
    /// * `bias` - Shift parameter (beta) [channels]
    /// * `running_mean` - Running mean [channels]
    /// * `running_var` - Running variance [channels]
    /// * `eps` - Epsilon for numerical stability
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
    /// Input shape: [batch, height, width, channels] (NHWC)
    /// Output shape: [batch, height, width, channels] (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        // Compute: (input - mean) / sqrt(var + eps) * weight + bias
        // In NHWC, the channel dimension is the last dimension, so
        // broadcasting of 1D tensors [C] over [N,H,W,C] works correctly
        // since they broadcast on the last dimension.

        // (input - running_mean)
        let centered = input.sub(&self.running_mean)?;

        // sqrt(running_var + eps)
        let var_plus_eps = self.running_var.add_scalar(self.eps)?;
        let std_dev = var_plus_eps.sqrt()?;

        // (input - mean) / std
        let normalized = centered.div(&std_dev)?;

        // normalized * weight + bias
        let scaled = normalized.mul(&self.weight)?;
        scaled.add(&self.bias)
    }
}

// ============================================================================
// Conv2d (native MLX)
// ============================================================================

/// 2D Convolution layer using MLX's native conv2d.
///
/// Uses `mlx::core::conv2d` which operates in NHWC format.
/// Weight shape: [out_channels, kernel_h, kernel_w, in_channels/groups]
pub struct NativeConv2d {
    /// Convolution weights [out_channels, kH, kW, in_channels/groups]
    weight: Arc<MxArray>,
    /// Optional bias [out_channels]
    bias: Option<Arc<MxArray>>,
    /// Stride (h, w)
    stride: (i32, i32),
    /// Padding (h, w)
    padding: (i32, i32),
    /// Dilation (h, w)
    dilation: (i32, i32),
    /// Groups for grouped/depthwise convolution
    groups: i32,
}

impl NativeConv2d {
    /// Create a new NativeConv2d layer.
    ///
    /// # Arguments
    /// * `weight` - Convolution weights [out_channels, kH, kW, in_channels/groups]
    /// * `bias` - Optional bias [out_channels]
    /// * `stride` - Stride as (stride_h, stride_w)
    /// * `padding` - Padding as (pad_h, pad_w)
    /// * `dilation` - Dilation as (dilation_h, dilation_w)
    /// * `groups` - Number of groups (1 for standard, out_channels for depthwise)
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

    /// Forward pass: apply convolution.
    ///
    /// Input shape: [batch, height, width, in_channels] (NHWC)
    /// Output shape: [batch, out_h, out_w, out_channels] (NHWC)
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

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }
}

// ============================================================================
// ConvTranspose2d (native MLX)
// ============================================================================

/// 2D Transposed Convolution layer using MLX's native conv_transpose2d.
///
/// Uses `mlx::core::conv_transpose2d` which operates in NHWC format.
/// Weight shape: [out_channels, kernel_h, kernel_w, in_channels/groups]
/// Transposed convolution layer with frozen batch norm.
///
/// **Note on output_padding**: The underlying FFI binding (`mlx_conv_transpose2d`) hardcodes
/// output_padding to (0, 0). This is correct for the current use case (DBHead upsampling
/// with kernel=2, stride=2, padding=0), which produces exact 2x upsampling. For future
/// use cases requiring non-zero output_padding, the C++ FFI and this struct must be updated
/// to expose it as a parameter.
pub struct NativeConvTranspose2d {
    /// Convolution weights [out_channels, kH, kW, in_channels/groups]
    weight: Arc<MxArray>,
    /// Optional bias [out_channels]
    bias: Option<Arc<MxArray>>,
    /// Stride (h, w)
    stride: (i32, i32),
    /// Padding (h, w)
    padding: (i32, i32),
    /// Dilation (h, w)
    dilation: (i32, i32),
    /// Groups for grouped/depthwise convolution
    groups: i32,
}

impl NativeConvTranspose2d {
    /// Create a new NativeConvTranspose2d layer.
    ///
    /// # Arguments
    /// * `weight` - Convolution weights [out_channels, kH, kW, in_channels/groups]
    /// * `bias` - Optional bias [out_channels]
    /// * `stride` - Stride as (stride_h, stride_w)
    /// * `padding` - Padding as (pad_h, pad_w)
    /// * `dilation` - Dilation as (dilation_h, dilation_w)
    /// * `groups` - Number of groups
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

    /// Forward pass: apply transposed convolution.
    ///
    /// Input shape: [batch, height, width, in_channels] (NHWC)
    /// Output shape: [batch, out_h, out_w, out_channels] (NHWC)
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

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }
}

// ============================================================================
// Learnable Affine Block
// ============================================================================

/// Learnable Affine Block: output = scale * input + bias
///
/// Adds learnable per-element scale and bias after activation.
/// Scale is initialized to 1.0, bias to 0.0.
pub struct LearnableAffineBlock {
    /// Scale parameter [1]
    scale: Arc<MxArray>,
    /// Bias parameter [1]
    bias: Arc<MxArray>,
}

impl LearnableAffineBlock {
    /// Create a new LearnableAffineBlock.
    pub fn new(scale: &MxArray, bias: &MxArray) -> Self {
        Self {
            scale: Arc::new(scale.clone()),
            bias: Arc::new(bias.clone()),
        }
    }

    /// Forward pass: scale * input + bias
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let scaled = input.mul(&self.scale)?;
        scaled.add(&self.bias)
    }
}

// ============================================================================
// ConvBNAct: Conv2d + BatchNorm + Activation (+ optional LAB)
// ============================================================================

/// Convolution + Batch Normalization + Activation layer.
///
/// Corresponds to HGNetV2ConvLayer in the reference implementation.
/// Optionally includes a LearnableAffineBlock after activation.
pub struct ConvBNAct {
    /// Convolution layer
    conv: NativeConv2d,
    /// Frozen batch normalization
    norm: FrozenBatchNorm2d,
    /// Activation function name ("relu", "silu", "none")
    activation: String,
    /// Optional learnable affine block
    lab: Option<LearnableAffineBlock>,
}

impl ConvBNAct {
    /// Create a new ConvBNAct layer.
    ///
    /// # Arguments
    /// * `conv` - NativeConv2d layer
    /// * `norm` - FrozenBatchNorm2d layer
    /// * `activation` - Activation name ("relu", "silu", "none"/identity)
    /// * `lab` - Optional LearnableAffineBlock
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
    ///
    /// Input/Output: [batch, height, width, channels] (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = self.conv.forward(input)?;
        x = self.norm.forward(&x)?;
        x = super::apply_activation(&x, &self.activation)?;
        if let Some(ref lab) = self.lab {
            x = lab.forward(&x)?;
        }
        Ok(x)
    }
}

// ============================================================================
// ConvBNActLight: Depthwise separable convolution
// ============================================================================

/// Light (depthwise separable) convolution block.
///
/// Corresponds to HGNetV2ConvLayerLight in the reference implementation.
/// Consists of:
/// 1. 1x1 pointwise convolution (no activation)
/// 2. kxk depthwise convolution (groups=out_channels) with activation
pub struct ConvBNActLight {
    /// 1x1 pointwise convolution (activation=None)
    conv1: ConvBNAct,
    /// kxk depthwise convolution (groups=out_channels)
    conv2: ConvBNAct,
}

impl ConvBNActLight {
    /// Create a new ConvBNActLight layer.
    pub fn new(conv1: ConvBNAct, conv2: ConvBNAct) -> Self {
        Self { conv1, conv2 }
    }

    /// Forward pass: pointwise conv -> depthwise conv
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let x = self.conv1.forward(input)?;
        self.conv2.forward(&x)
    }
}

// ============================================================================
// HGBlock: Basic building block with feature aggregation
// ============================================================================

/// A single convolutional layer within an HGBlock, which can be either
/// a standard ConvBNAct or a light (depthwise separable) ConvBNActLight.
pub enum HGBlockLayer {
    Standard(ConvBNAct),
    Light(ConvBNActLight),
}

impl HGBlockLayer {
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        match self {
            HGBlockLayer::Standard(layer) => layer.forward(input),
            HGBlockLayer::Light(layer) => layer.forward(input),
        }
    }
}

/// HGBlock: Feature aggregation block.
///
/// Corresponds to HGNetV2BasicLayer in the reference implementation.
///
/// Architecture:
/// 1. Sequential conv layers, each taking the previous output
/// 2. Concatenate all intermediate outputs (including input) along channel dim
/// 3. Aggregation: squeeze (1x1 conv to out/2) -> excitation (1x1 conv to out)
/// 4. Optional residual connection (for non-first blocks)
pub struct HGBlock {
    /// Sequential conv layers (standard or light)
    layers: Vec<HGBlockLayer>,
    /// Aggregation squeeze conv (total_channels -> out_channels/2)
    agg_squeeze: ConvBNAct,
    /// Aggregation excitation conv (out_channels/2 -> out_channels)
    agg_excitation: ConvBNAct,
    /// Whether to use residual connection
    residual: bool,
}

impl HGBlock {
    /// Create a new HGBlock.
    ///
    /// # Arguments
    /// * `layers` - Sequential conv layers
    /// * `agg_squeeze` - Aggregation squeeze conv
    /// * `agg_excitation` - Aggregation excitation conv
    /// * `residual` - Whether to add input as residual
    pub fn new(
        layers: Vec<HGBlockLayer>,
        agg_squeeze: ConvBNAct,
        agg_excitation: ConvBNAct,
        residual: bool,
    ) -> Self {
        Self {
            layers,
            agg_squeeze,
            agg_excitation,
            residual,
        }
    }

    /// Forward pass with feature aggregation.
    ///
    /// Input/Output: [batch, height, width, channels] (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let identity = input.clone();
        let mut outputs: Vec<MxArray> = vec![input.clone()];

        let mut hidden = input.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
            outputs.push(hidden.clone());
        }

        // Concatenate all intermediate outputs along channel axis (last dim in NHWC)
        // NHWC: axis=3 (channels) or axis=-1
        let output_refs: Vec<&MxArray> = outputs.iter().collect();
        let concatenated = MxArray::concatenate_many(output_refs, Some(3))?;

        // Aggregation: squeeze -> excitation
        let mut aggregated = self.agg_squeeze.forward(&concatenated)?;
        aggregated = self.agg_excitation.forward(&aggregated)?;

        // Residual connection
        if self.residual {
            aggregated = aggregated.add(&identity)?;
        }

        Ok(aggregated)
    }
}

// ============================================================================
// MaxPool2d (pure Rust implementation for MLX NHWC)
// ============================================================================

/// Max pooling implementation for NHWC format.
///
/// Since MLX does not have a native max_pool2d FFI binding, we implement
/// it using slicing and element-wise maximum operations.
///
/// For the specific case used in HGNetV2 (kernel=2, stride=1, ceil_mode=true),
/// this pads the input by (0,1) on height and width, then takes the maximum
/// of 4 overlapping positions.
pub struct MaxPool2d {
    kernel_size: i32,
    stride: i32,
    /// Whether to use ceil mode for output size calculation
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

    /// Forward pass: apply max pooling.
    ///
    /// Input: [batch, height, width, channels] (NHWC)
    /// Output: [batch, out_h, out_w, channels] (NHWC)
    ///
    /// For kernel=2, stride=1, ceil_mode=true:
    /// - Pads input with -inf on H,W dimensions
    /// - Takes max over 2x2 windows with stride 1
    /// - Output size = input size (due to ceil_mode + padding)
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

        // Calculate output dimensions
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

        // Pad input with -inf so that padded elements don't affect max
        // We need to pad H and W dimensions to ensure we can take k-sized windows
        // at all valid output positions.
        let pad_h = (out_h - 1) * s + k - in_h;
        let pad_w = (out_w - 1) * s + k - in_w;

        let padded = if pad_h > 0 || pad_w > 0 {
            // NHWC pad_width: [batch_before, batch_after, h_before, h_after, w_before, w_after, c_before, c_after]
            let pad_width = vec![0i32, 0, 0, pad_h as i32, 0, pad_w as i32, 0, 0];
            input.pad(&pad_width, f64::NEG_INFINITY)?
        } else {
            input.clone()
        };

        // For the common case of kernel_size=2, stride=1, we can implement
        // this efficiently using element-wise maximum of shifted windows.
        if k == 2 && s == 1 {
            // max(input[h:h+2, w:w+2]) for each output position
            // = max(max(padded[:, :out_h, :out_w, :],
            //          padded[:, 1:out_h+1, :out_w, :]),
            //       max(padded[:, :out_h, 1:out_w+1, :],
            //          padded[:, 1:out_h+1, 1:out_w+1, :]))

            let tl = padded.slice(&[0, 0, 0, 0], &[batch, out_h, out_w, channels])?;
            let tr = padded.slice(&[0, 0, 1, 0], &[batch, out_h, out_w + 1, channels])?;
            let bl = padded.slice(&[0, 1, 0, 0], &[batch, out_h + 1, out_w, channels])?;
            let br = padded.slice(&[0, 1, 1, 0], &[batch, out_h + 1, out_w + 1, channels])?;

            let max_top = tl.maximum(&tr)?;
            let max_bot = bl.maximum(&br)?;
            let result = max_top.maximum(&max_bot)?;

            return Ok(result);
        }

        // General case: use nested loops of slicing and max
        // For each (kh, kw) position in the kernel, slice the padded input
        // and accumulate the element-wise maximum.
        let mut result: Option<MxArray> = None;
        for kh in 0..k {
            for kw in 0..k {
                let window = padded.slice(
                    &[0, kh, kw, 0],
                    &[batch, kh + out_h * s, kw + out_w * s, channels],
                )?;

                // If stride > 1, we need to take every s-th element
                // For stride=1, the slice already gives us the right output
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

// ============================================================================
// HGNetV2Embeddings (Stem)
// ============================================================================

/// HGNetV2 Stem (Embeddings) layer.
///
/// Corresponds to HGNetV2Embeddings in the reference implementation.
///
/// Architecture:
/// 1. stem1: Conv(3->32, k=3, s=2) - initial 2x downsample
/// 2. Pad(0,1,0,1) + stem2a: Conv(32->16, k=2, s=1)
/// 3. Pad(0,1,0,1) + stem2b: Conv(16->32, k=2, s=1)
/// 4. MaxPool2d(k=2, s=1, ceil_mode=True) on output of stem1+pad
/// 5. Cat(pooled, stem2b) along channels -> 64 channels
/// 6. stem3: Conv(64->32, k=3, s=2) - second 2x downsample
/// 7. stem4: Conv(32->48, k=1, s=1) - channel projection
///
/// Total downsample: 4x (stride 2 * stride 2)
/// Output: [batch, H/4, W/4, stem_channels[2]]
pub struct HGNetV2Embeddings {
    stem1: ConvBNAct,
    stem2a: ConvBNAct,
    stem2b: ConvBNAct,
    stem3: ConvBNAct,
    stem4: ConvBNAct,
    pool: MaxPool2d,
}

impl HGNetV2Embeddings {
    /// Create a new HGNetV2Embeddings layer.
    pub fn new(
        stem1: ConvBNAct,
        stem2a: ConvBNAct,
        stem2b: ConvBNAct,
        stem3: ConvBNAct,
        stem4: ConvBNAct,
    ) -> Self {
        Self {
            stem1,
            stem2a,
            stem2b,
            stem3,
            stem4,
            pool: MaxPool2d::new(2, 1, true),
        }
    }

    /// Forward pass through the stem.
    ///
    /// Input: [batch, H, W, 3] (NHWC, RGB image)
    /// Output: [batch, H/4, W/4, stem_channels[2]] (NHWC)
    pub fn forward(&self, pixel_values: &MxArray) -> Result<MxArray> {
        // stem1: 3->32, k=3, s=2
        let embedding = self.stem1.forward(pixel_values)?;

        // Pad H and W by (0,1) each for the k=2 convolutions
        // In PyTorch NCHW: F.pad(x, (0, 1, 0, 1)) pads W_right=1, H_bottom=1
        // In MLX NHWC pad_width: [N_before, N_after, H_before, H_after, W_before, W_after, C_before, C_after]
        let padded = embedding.pad(&[0, 0, 0, 1, 0, 1, 0, 0], 0.0)?;

        // stem2a: 32->16, k=2, s=1
        let emb_stem_2a = self.stem2a.forward(&padded)?;

        // Pad again for stem2b
        let emb_stem_2a_padded = emb_stem_2a.pad(&[0, 0, 0, 1, 0, 1, 0, 0], 0.0)?;

        // stem2b: 16->32, k=2, s=1
        let emb_stem_2b = self.stem2b.forward(&emb_stem_2a_padded)?;

        // MaxPool on the first padded embedding (after stem1 + pad)
        let pooled_emb = self.pool.forward(&padded)?;

        // Concatenate pooled and stem2b along channel axis (axis=3 in NHWC)
        let concatenated = MxArray::concatenate(&pooled_emb, &emb_stem_2b, 3)?;

        // stem3: 64->32, k=3, s=2 (second 2x downsample)
        let embedding = self.stem3.forward(&concatenated)?;

        // stem4: 32->48, k=1, s=1 (channel projection)
        self.stem4.forward(&embedding)
    }
}

// ============================================================================
// HGNetV2Stage
// ============================================================================

/// HGNetV2 Stage: optional downsample + stack of HGBlocks.
///
/// Corresponds to HGNetV2Stage in the reference implementation.
///
/// If downsample is true, applies a depthwise Conv(k=3, s=2, groups=in_channels)
/// before the HGBlocks.
pub struct HGNetV2Stage {
    /// Optional downsample layer (depthwise conv with stride 2)
    downsample: Option<ConvBNAct>,
    /// Stack of HGBlocks
    blocks: Vec<HGBlock>,
}

impl HGNetV2Stage {
    /// Create a new HGNetV2Stage.
    pub fn new(downsample: Option<ConvBNAct>, blocks: Vec<HGBlock>) -> Self {
        Self { downsample, blocks }
    }

    /// Forward pass: downsample (if any) -> sequential HGBlocks.
    ///
    /// Input/Output: [batch, height, width, channels] (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = if let Some(ref ds) = self.downsample {
            ds.forward(input)?
        } else {
            input.clone()
        };

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        Ok(x)
    }
}

// ============================================================================
// HGNetV2Encoder
// ============================================================================

/// HGNetV2 Encoder: sequence of stages.
///
/// Corresponds to HGNetV2Encoder in the reference implementation.
pub struct HGNetV2Encoder {
    /// The 4 stages
    stages: Vec<HGNetV2Stage>,
}

impl HGNetV2Encoder {
    pub fn new(stages: Vec<HGNetV2Stage>) -> Self {
        Self { stages }
    }

    /// Forward pass: run through all stages, collecting hidden states.
    ///
    /// Input: [batch, H/4, W/4, stem_out_channels] (output of embeddings)
    /// Returns: Vec of hidden states - one before each stage plus the final output.
    ///          Index 0 = input to stage 0 (stem output)
    ///          Index i = output of stage i-1 (for i > 0)
    ///          Index num_stages = output of last stage
    pub fn forward(&self, input: &MxArray) -> Result<Vec<MxArray>> {
        let mut hidden_states: Vec<MxArray> = Vec::with_capacity(self.stages.len() + 1);
        hidden_states.push(input.clone());

        let mut x = input.clone();
        for stage in &self.stages {
            x = stage.forward(&x)?;
            hidden_states.push(x.clone());
        }

        Ok(hidden_states)
    }
}

// ============================================================================
// HGNetV2Backbone
// ============================================================================

/// HGNetV2 Backbone: Embeddings + Encoder + feature selection.
///
/// Corresponds to HGNetV2Backbone in the reference implementation.
/// Returns feature maps for the requested output stages.
pub struct HGNetV2Backbone {
    /// Stem / embeddings layer
    embedder: HGNetV2Embeddings,
    /// Encoder with all stages
    encoder: HGNetV2Encoder,
    /// Which features to output (indices into hidden_states)
    /// stage_names: ["stem", "stage1", "stage2", "stage3", "stage4"]
    /// out_features selects from these. "stage1" = hidden_states[1], etc.
    out_feature_indices: Vec<usize>,
    /// Channel sizes for the output features
    pub channels: Vec<i32>,
}

impl HGNetV2Backbone {
    /// Create a new HGNetV2Backbone.
    ///
    /// # Arguments
    /// * `embedder` - The stem/embeddings layer
    /// * `encoder` - The encoder with all stages
    /// * `config` - Configuration to determine output features
    pub fn new(
        embedder: HGNetV2Embeddings,
        encoder: HGNetV2Encoder,
        config: &HGNetV2Config,
    ) -> Self {
        // Map out_features names to indices in the hidden_states array
        // hidden_states[0] = "stem" output
        // hidden_states[1] = "stage1" output
        // hidden_states[2] = "stage2" output, etc.
        let stage_names: Vec<String> = std::iter::once("stem".to_string())
            .chain((1..=config.num_stages()).map(|i| format!("stage{i}")))
            .collect();

        let out_feature_indices: Vec<usize> = config
            .out_features
            .iter()
            .filter_map(|name| stage_names.iter().position(|s| s == name))
            .collect();

        // Compute channel sizes for output features
        let all_channels: Vec<i32> = std::iter::once(config.embedding_size)
            .chain(config.stage_out_channels.iter().copied())
            .collect();

        let channels: Vec<i32> = out_feature_indices
            .iter()
            .map(|&idx| all_channels[idx])
            .collect();

        Self {
            embedder,
            encoder,
            out_feature_indices,
            channels,
        }
    }

    /// Forward pass: extract feature maps from the backbone.
    ///
    /// Input: [batch, H, W, 3] (NHWC, RGB image)
    /// Returns: Vec of feature maps at requested scales.
    ///
    /// For default config (out_features = ["stage1", "stage2", "stage3", "stage4"]):
    /// - feature_maps[0]: [batch, H/4, W/4, 128]   (stage1)
    /// - feature_maps[1]: [batch, H/8, W/8, 512]   (stage2)
    /// - feature_maps[2]: [batch, H/16, W/16, 1024] (stage3)
    /// - feature_maps[3]: [batch, H/32, W/32, 2048] (stage4)
    pub fn forward(&self, pixel_values: &MxArray) -> Result<Vec<MxArray>> {
        let embedding_output = self.embedder.forward(pixel_values)?;
        let hidden_states = self.encoder.forward(&embedding_output)?;

        let feature_maps: Vec<MxArray> = self
            .out_feature_indices
            .iter()
            .map(|&idx| hidden_states[idx].clone())
            .collect();

        Ok(feature_maps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frozen_batch_norm() {
        // Create simple 1D tensors for a 4-channel batch norm
        let weight = MxArray::from_float32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let bias = MxArray::from_float32(&[0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
        let running_mean = MxArray::from_float32(&[0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
        let running_var = MxArray::from_float32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();

        let bn = FrozenBatchNorm2d::new(&weight, &bias, &running_mean, &running_var, 1e-5);

        // Input: [1, 2, 2, 4] - identity normalization should pass through
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

        // Input: [1, 3, 3, 1] - simple 3x3 with 1 channel
        let input = MxArray::from_float32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 3, 3, 1],
        )
        .unwrap();

        let output = pool.forward(&input).unwrap();
        let output_shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();

        // With k=2, s=1, ceil_mode=true:
        // out_h = ceil((3 - 2) / 1) + 1 = 2
        assert_eq!(output_shape, vec![1, 2, 2, 1]);

        // Check values: max of 2x2 windows
        output.eval();
        let data = output.to_float32().unwrap();
        let data: Vec<f32> = data.to_vec();
        // Position (0,0): max(1,2,4,5) = 5
        assert_eq!(data[0], 5.0);
        // Position (0,1): max(2,3,5,6) = 6
        assert_eq!(data[1], 6.0);
        // Position (1,0): max(4,5,7,8) = 8
        assert_eq!(data[2], 8.0);
        // Position (1,1): max(5,6,8,9) = 9
        assert_eq!(data[3], 9.0);
    }

    #[test]
    fn test_default_config() {
        let config = HGNetV2Config::default();
        assert_eq!(config.num_stages(), 4);
        assert_eq!(config.stem_channels, vec![3, 32, 48]);
        assert_eq!(config.output_channels(), vec![128, 512, 1024, 2048]);
    }
}
