//! PP-DocLayoutV3 Hybrid Encoder
//!
//! Implements the hybrid encoder for the PP-DocLayoutV3 model, which combines:
//! - AIFI (Attention-based Intra-scale Feature Interaction): Transformer encoder
//!   applied to the highest-level (lowest resolution) feature map
//! - FPN (Feature Pyramid Network): Top-down path with lateral connections
//! - PAN (Path Aggregation Network): Bottom-up path with downsampling
//! - MaskFeatFPN: Multi-level mask feature head producing 32-channel mask features
//!
//! All operations use NHWC format (MLX native).
//! Weight tensors follow MLX conventions:
//! - Conv2d weights: [out_channels, kernel_h, kernel_w, in_channels/groups]
//! - Linear weights: [out_features, in_features]
//! - BatchNorm params: [channels]
//! - LayerNorm params: [channels]

use crate::array::MxArray;
use crate::nn::activations::Activations;
use crate::nn::{LayerNorm, Linear};
use napi::Either;
use napi::bindgen_prelude::*;

use super::backbone::{FrozenBatchNorm2d, NativeConv2d};
use super::config::PPDocLayoutV3Config;

// ============================================================================
// ConvNormLayer: Conv2d + BatchNorm + optional Activation
// ============================================================================

/// Convolution + Batch Normalization + optional Activation layer.
///
/// Corresponds to PPDocLayoutV3ConvNormLayer in the reference implementation.
/// Uses Conv2d (no bias) + FrozenBatchNorm2d + optional activation.
///
/// Padding is automatically computed as (kernel_size - 1) / 2.
pub struct ConvNormLayer {
    /// Convolution layer (no bias)
    conv: NativeConv2d,
    /// Frozen batch normalization
    norm: FrozenBatchNorm2d,
    /// Activation function name (None for identity)
    activation: Option<String>,
}

impl ConvNormLayer {
    /// Create a new ConvNormLayer.
    ///
    /// # Arguments
    /// * `conv` - NativeConv2d layer (no bias)
    /// * `norm` - FrozenBatchNorm2d layer
    /// * `activation` - Optional activation name ("relu", "silu", "gelu")
    pub fn new(conv: NativeConv2d, norm: FrozenBatchNorm2d, activation: Option<&str>) -> Self {
        Self {
            conv,
            norm,
            activation: activation.map(|s| s.to_string()),
        }
    }

    /// Forward pass: Conv -> BN -> (optional Activation)
    ///
    /// Input/Output: [batch, height, width, channels] (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = self.conv.forward(input)?;
        x = self.norm.forward(&x)?;
        if let Some(ref act) = self.activation {
            x = super::apply_activation(&x, act)?;
        }
        Ok(x)
    }
}

// ============================================================================
// ConvLayer: Conv2d + BatchNorm + Activation (always has activation)
// ============================================================================

/// Convolution + Batch Normalization + Activation layer.
///
/// Corresponds to PPDocLayoutV3ConvLayer in the reference implementation.
/// Like ConvNormLayer but activation is always present.
/// Padding = kernel_size // 2.
pub struct ConvLayer {
    /// Convolution layer (no bias)
    conv: NativeConv2d,
    /// Frozen batch normalization
    norm: FrozenBatchNorm2d,
    /// Activation function name
    activation: String,
}

impl ConvLayer {
    /// Create a new ConvLayer.
    pub fn new(conv: NativeConv2d, norm: FrozenBatchNorm2d, activation: &str) -> Self {
        Self {
            conv,
            norm,
            activation: activation.to_string(),
        }
    }

    /// Forward pass: Conv -> BN -> Activation
    ///
    /// Input/Output: [batch, height, width, channels] (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = self.conv.forward(input)?;
        x = self.norm.forward(&x)?;
        super::apply_activation(&x, &self.activation)
    }
}

// ============================================================================
// 2D Sinusoidal Position Embedding
// ============================================================================

/// Build 2D sinusoidal position embeddings.
///
/// Creates a [1, H*W, embed_dim] position encoding tensor using sin/cos
/// of grid coordinates, divided into 4 quadrants:
/// [sin(h), cos(h), sin(w), cos(w)], each with embed_dim/4 frequencies.
///
/// # Arguments
/// * `width` - Feature map width
/// * `height` - Feature map height
/// * `embed_dim` - Embedding dimension (must be divisible by 4)
/// * `temperature` - Temperature for frequency scaling (default 10000)
pub fn build_2d_sincos_position_embedding(
    width: i64,
    height: i64,
    embed_dim: i64,
    temperature: f64,
) -> Result<MxArray> {
    if embed_dim % 4 != 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Embed dimension must be divisible by 4 for 2D sin-cos position embedding, got {embed_dim}"
            ),
        ));
    }

    let pos_dim = embed_dim / 4;

    // grid_w = arange(width), shape [width]
    let grid_w = MxArray::arange(0.0, width as f64, Some(1.0), None)?;
    // grid_h = arange(height), shape [height]
    let grid_h = MxArray::arange(0.0, height as f64, Some(1.0), None)?;

    // Create meshgrid: grid_w[i,j] = j, grid_h[i,j] = i
    // meshgrid with indexing="xy": grid_w varies along columns (axis=1), grid_h along rows (axis=0)
    // grid_w: [width] -> [1, width] -> [height, width]
    // grid_h: [height] -> [height, 1] -> [height, width]
    let grid_w_2d = grid_w
        .reshape(&[1, width])?
        .broadcast_to(&[height, width])?;
    let grid_h_2d = grid_h
        .reshape(&[height, 1])?
        .broadcast_to(&[height, width])?;

    // omega = arange(pos_dim) / pos_dim
    let omega =
        MxArray::arange(0.0, pos_dim as f64, Some(1.0), None)?.div_scalar(pos_dim as f64)?;
    // omega = 1.0 / (temperature ** omega)
    let temp_arr = MxArray::full(&[pos_dim], Either::A(temperature), None)?;
    let omega = temp_arr.power(&omega)?.reciprocal()?;

    // out_w = grid_w.flatten()[..., None] @ omega[None]
    // grid_w.flatten(): [H*W] -> [H*W, 1]
    // omega: [pos_dim] -> [1, pos_dim]
    // Result: [H*W, pos_dim]
    let hw = height * width;
    let grid_w_flat = grid_w_2d.reshape(&[hw, 1])?;
    let grid_h_flat = grid_h_2d.reshape(&[hw, 1])?;
    let omega_row = omega.reshape(&[1, pos_dim])?;

    let out_w = grid_w_flat.matmul(&omega_row)?; // [H*W, pos_dim]
    let out_h = grid_h_flat.matmul(&omega_row)?; // [H*W, pos_dim]

    // Compute sin and cos
    let sin_h = out_h.sin()?;
    let cos_h = out_h.cos()?;
    let sin_w = out_w.sin()?;
    let cos_w = out_w.cos()?;

    // Concatenate: [sin_h, cos_h, sin_w, cos_w] along dim=1
    // Each is [H*W, pos_dim], result is [H*W, embed_dim]
    let pos_embed = MxArray::concatenate_many(vec![&sin_h, &cos_h, &sin_w, &cos_w], Some(1))?;

    // Add batch dimension: [1, H*W, embed_dim]
    pos_embed.expand_dims(0)
}

// ============================================================================
// Multi-Head Attention
// ============================================================================

/// Multi-headed self-attention.
///
/// Corresponds to PPDocLayoutV3MultiheadAttention in the reference implementation.
/// Position embeddings are added to queries and keys before projection.
///
/// Input: [batch, seq_len, embed_dim]
/// Output: [batch, seq_len, embed_dim]
pub struct MultiheadAttention {
    /// Query projection [embed_dim, embed_dim]
    q_proj: Linear,
    /// Key projection [embed_dim, embed_dim]
    k_proj: Linear,
    /// Value projection [embed_dim, embed_dim]
    v_proj: Linear,
    /// Output projection [embed_dim, embed_dim]
    out_proj: Linear,
    /// Number of attention heads
    num_heads: i32,
    /// Dimension per head
    head_dim: i32,
    /// Scaling factor: 1/sqrt(head_dim)
    scaling: f64,
}

impl MultiheadAttention {
    /// Create a new MultiheadAttention layer.
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        num_heads: i32,
    ) -> Result<Self> {
        // Infer embed_dim from q_proj weight shape
        let q_weight = q_proj.get_weight();
        let embed_dim = q_weight.shape_at(0)? as i32;
        let head_dim = embed_dim / num_heads;
        if head_dim * num_heads != embed_dim {
            return Err(Error::new(
                Status::InvalidArg,
                format!("embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"),
            ));
        }
        let scaling = (head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scaling,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `hidden_states` - [batch, seq_len, embed_dim]
    /// * `position_embeddings` - Optional [1, seq_len, embed_dim] or [batch, seq_len, embed_dim]
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, embed_dim]
    pub fn forward(
        &self,
        hidden_states: &MxArray,
        position_embeddings: Option<&MxArray>,
    ) -> Result<MxArray> {
        let shape = hidden_states.shape()?;
        let batch_size = shape[0];
        let target_len = shape[1];
        let embed_dim = shape[2];

        // Add position embeddings to hidden states for Q/K projections
        let hidden_with_pos = if let Some(pos_emb) = position_embeddings {
            hidden_states.add(pos_emb)?
        } else {
            hidden_states.clone()
        };

        // Project queries (from hidden + pos), keys (from hidden + pos), values (from hidden only)
        // Q, K use position-enhanced states; V uses original states
        let query_states = self
            .q_proj
            .forward(&hidden_with_pos)?
            .mul_scalar(self.scaling)?;
        let key_states = self.k_proj.forward(&hidden_with_pos)?;
        let value_states = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq_len, num_heads, head_dim] and transpose to [batch, num_heads, seq_len, head_dim]
        let nh = self.num_heads as i64;
        let hd = self.head_dim as i64;

        let query_states = query_states
            .reshape(&[batch_size, target_len, nh, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?; // [B, nh, T, hd]

        let key_states = key_states
            .reshape(&[batch_size, target_len, nh, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?; // [B, nh, T, hd]

        let value_states = value_states
            .reshape(&[batch_size, target_len, nh, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?; // [B, nh, T, hd]

        // Attention: Q @ K^T -> [B, nh, T, T]
        let key_states_t = key_states.transpose(Some(&[0, 1, 3, 2]))?; // [B, nh, hd, T]
        let attn_weights = query_states.matmul(&key_states_t)?;

        // Softmax
        let attn_weights = Activations::softmax(&attn_weights, Some(-1))?;

        // Attention output: weights @ V -> [B, nh, T, hd]
        let attn_output = attn_weights.matmul(&value_states)?;

        // Reshape: [B, nh, T, hd] -> [B, T, nh, hd] -> [B, T, embed_dim]
        let attn_output = attn_output
            .transpose(Some(&[0, 2, 1, 3]))? // [B, T, nh, hd]
            .reshape(&[batch_size, target_len, embed_dim])?;

        // Output projection
        self.out_proj.forward(&attn_output)
    }
}

// ============================================================================
// Encoder Layer (Transformer)
// ============================================================================

/// Transformer encoder layer with self-attention and FFN.
///
/// Corresponds to PPDocLayoutV3EncoderLayer in the reference implementation.
///
/// Architecture (post-norm, normalize_before=false):
/// 1. Self-attention + residual + LayerNorm
/// 2. FFN (fc1 -> activation -> fc2) + residual + LayerNorm
pub struct EncoderLayer {
    /// Self-attention
    self_attn: MultiheadAttention,
    /// LayerNorm after self-attention
    self_attn_layer_norm: LayerNorm,
    /// FFN first linear (hidden_dim -> ffn_dim)
    fc1: Linear,
    /// FFN second linear (ffn_dim -> hidden_dim)
    fc2: Linear,
    /// LayerNorm after FFN
    final_layer_norm: LayerNorm,
    /// Activation function for FFN
    activation: String,
    /// Whether to normalize before (pre-norm) or after (post-norm)
    normalize_before: bool,
}

impl EncoderLayer {
    /// Create a new EncoderLayer.
    pub fn new(
        self_attn: MultiheadAttention,
        self_attn_layer_norm: LayerNorm,
        fc1: Linear,
        fc2: Linear,
        final_layer_norm: LayerNorm,
        activation: &str,
        normalize_before: bool,
    ) -> Self {
        Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation: activation.to_string(),
            normalize_before,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `hidden_states` - [batch, seq_len, embed_dim]
    /// * `position_embeddings` - Optional positional embeddings
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, embed_dim]
    pub fn forward(
        &self,
        hidden_states: &MxArray,
        position_embeddings: Option<&MxArray>,
    ) -> Result<MxArray> {
        // Self-attention block
        let residual = hidden_states.clone();

        let mut x = if self.normalize_before {
            self.self_attn_layer_norm.forward(hidden_states)?
        } else {
            hidden_states.clone()
        };

        x = self.self_attn.forward(&x, position_embeddings)?;
        x = residual.add(&x)?;

        let mut x = if !self.normalize_before {
            self.self_attn_layer_norm.forward(&x)?
        } else {
            x
        };

        // FFN block
        let residual = x.clone();
        if self.normalize_before {
            x = self.final_layer_norm.forward(&x)?;
        }

        x = self.fc1.forward(&x)?;
        x = super::apply_activation(&x, &self.activation)?;
        x = self.fc2.forward(&x)?;
        x = residual.add(&x)?;

        if !self.normalize_before {
            x = self.final_layer_norm.forward(&x)?;
        }

        Ok(x)
    }
}

// ============================================================================
// Encoder (stack of encoder layers)
// ============================================================================

/// Stack of transformer encoder layers.
///
/// Corresponds to PPDocLayoutV3Encoder in the reference implementation.
pub struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn new(layers: Vec<EncoderLayer>) -> Self {
        Self { layers }
    }

    /// Forward pass through all encoder layers.
    ///
    /// # Arguments
    /// * `src` - [batch, seq_len, embed_dim]
    /// * `pos_embed` - Optional positional embeddings
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, embed_dim]
    pub fn forward(&self, src: &MxArray, pos_embed: Option<&MxArray>) -> Result<MxArray> {
        let mut hidden_states = src.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, pos_embed)?;
        }
        Ok(hidden_states)
    }
}

// ============================================================================
// RepVGG Block
// ============================================================================

/// RepVGG block: parallel 3x3 and 1x1 convolutions with addition.
///
/// Corresponds to PPDocLayoutV3RepVggBlock in the reference implementation.
///
/// Architecture:
/// output = activation(conv3x3(x) + conv1x1(x))
pub struct RepVggBlock {
    /// 3x3 convolution branch
    conv1: ConvNormLayer,
    /// 1x1 convolution branch
    conv2: ConvNormLayer,
    /// Activation function
    activation: String,
}

impl RepVggBlock {
    /// Create a new RepVggBlock.
    pub fn new(conv1: ConvNormLayer, conv2: ConvNormLayer, activation: &str) -> Self {
        Self {
            conv1,
            conv2,
            activation: activation.to_string(),
        }
    }

    /// Forward pass: activation(conv3x3(x) + conv1x1(x))
    ///
    /// Input/Output: [batch, height, width, channels] (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let y1 = self.conv1.forward(input)?;
        let y2 = self.conv2.forward(input)?;
        let y = y1.add(&y2)?;
        super::apply_activation(&y, &self.activation)
    }
}

// ============================================================================
// CSPRepLayer (Cross-Stage Partial with RepVGG blocks)
// ============================================================================

/// Cross-Stage Partial network layer with RepVGG blocks.
///
/// Corresponds to PPDocLayoutV3CSPRepLayer in the reference implementation.
///
/// Architecture:
/// 1. Split input through conv1 (for bottleneck path) and conv2 (shortcut path)
/// 2. Pass conv1 output through a sequence of RepVGG blocks
/// 3. Add the two paths
/// 4. Optionally project with conv3 if hidden_channels != out_channels
///
/// Input: [batch, H, W, in_channels] where in_channels = encoder_hidden_dim * 2
/// Output: [batch, H, W, out_channels] where out_channels = encoder_hidden_dim
pub struct CSPRepLayer {
    /// 1x1 conv for bottleneck path (in_channels -> hidden_channels)
    conv1: ConvNormLayer,
    /// 1x1 conv for shortcut path (in_channels -> hidden_channels)
    conv2: ConvNormLayer,
    /// RepVGG bottleneck blocks
    bottlenecks: Vec<RepVggBlock>,
    /// Optional 1x1 conv for output projection (hidden_channels -> out_channels)
    /// None if hidden_channels == out_channels (identity)
    conv3: Option<ConvNormLayer>,
}

impl CSPRepLayer {
    /// Create a new CSPRepLayer.
    pub fn new(
        conv1: ConvNormLayer,
        conv2: ConvNormLayer,
        bottlenecks: Vec<RepVggBlock>,
        conv3: Option<ConvNormLayer>,
    ) -> Self {
        Self {
            conv1,
            conv2,
            bottlenecks,
            conv3,
        }
    }

    /// Forward pass.
    ///
    /// Input: [batch, H, W, in_channels] (NHWC)
    /// Output: [batch, H, W, out_channels] (NHWC)
    pub fn forward(&self, hidden_state: &MxArray) -> Result<MxArray> {
        // Bottleneck path
        let mut x1 = self.conv1.forward(hidden_state)?;
        for block in &self.bottlenecks {
            x1 = block.forward(&x1)?;
        }

        // Shortcut path
        let x2 = self.conv2.forward(hidden_state)?;

        // Merge
        let merged = x1.add(&x2)?;

        // Optional output projection
        if let Some(ref conv3) = self.conv3 {
            conv3.forward(&merged)
        } else {
            Ok(merged)
        }
    }
}

// ============================================================================
// Nearest-neighbor 2x Upsampling (NHWC)
// ============================================================================

/// Upsample a feature map by 2x using nearest-neighbor interpolation.
///
/// Input: [batch, height, width, channels] (NHWC)
/// Output: [batch, height*2, width*2, channels] (NHWC)
///
/// Uses reshape + broadcast approach:
/// [B, H, W, C] -> [B, H, 1, W, 1, C] -> [B, H, 2, W, 2, C] -> [B, H*2, W*2, C]
fn upsample_nearest_2x(input: &MxArray) -> Result<MxArray> {
    let shape = input.shape()?;
    let shape: Vec<i64> = shape.as_ref().to_vec();
    let batch = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];

    // Reshape to [B, H, 1, W, 1, C]
    let expanded = input.reshape(&[batch, h, 1, w, 1, c])?;

    // Broadcast to [B, H, 2, W, 2, C]
    let upsampled = expanded.broadcast_to(&[batch, h, 2, w, 2, c])?;

    // Reshape to [B, H*2, W*2, C]
    upsampled.reshape(&[batch, h * 2, w * 2, c])
}

// ============================================================================
// Bilinear 2x Upsampling (NHWC, align_corners=False)
// ============================================================================

/// Upsample a feature map by 2x using bilinear interpolation.
///
/// Matches PyTorch's `F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)`.
///
/// Input: [batch, height, width, channels] (NHWC)
/// Output: [batch, height*2, width*2, channels] (NHWC)
///
/// Uses separable bilinear interpolation via `take` + weighted blending:
/// 1. Interpolate along height (axis 1): gather rows by y0/y1, blend with wy
/// 2. Interpolate along width (axis 2): gather columns by x0/x1, blend with wx
///
/// For `align_corners=False`, output pixel `oy` maps to input coordinate:
///   `src_y = (oy + 0.5) / scale - 0.5 = (oy + 0.5) / 2.0 - 0.5`
fn upsample_bilinear_2x(input: &MxArray) -> Result<MxArray> {
    let shape = input.shape()?;
    let shape: Vec<i64> = shape.as_ref().to_vec();
    let h = shape[1];
    let w = shape[2];

    // --- Step 1: Interpolate along height (axis 1) ---
    // For each output row oy in [0, 2*H), compute:
    //   src_y = (oy + 0.5) / 2.0 - 0.5
    //   y0 = clamp(floor(src_y), 0, H-1)
    //   y1 = clamp(floor(src_y) + 1, 0, H-1)
    //   wy = src_y - floor(src_y)  (but when src_y < 0, wy is clamped via y0==y1==0)
    let out_h = h * 2;
    let mut y0_idx = Vec::with_capacity(out_h as usize);
    let mut y1_idx = Vec::with_capacity(out_h as usize);
    let mut wy_vals = Vec::with_capacity(out_h as usize);

    for oy in 0..out_h {
        let src_y = (oy as f64 + 0.5) / 2.0 - 0.5;
        let y0_raw = src_y.floor() as i64;
        let y0 = y0_raw.clamp(0, h - 1) as i32;
        let y1 = (y0_raw + 1).clamp(0, h - 1) as i32;
        let wy = (src_y - src_y.floor()) as f32;
        // When src_y < 0, floor(src_y) = -1, so wy = src_y - (-1) = src_y + 1.
        // But both y0 and y1 clamp to 0, so the weight doesn't matter (both point to same row).
        // We still store the correct wy for clarity.
        y0_idx.push(y0);
        y1_idx.push(y1);
        wy_vals.push(wy);
    }

    let y0_arr = MxArray::from_int32(&y0_idx, &[out_h])?;
    let y1_arr = MxArray::from_int32(&y1_idx, &[out_h])?;
    // wy shape: [1, 2*H, 1, 1] for broadcasting over [B, 2*H, W, C]
    let wy_arr = MxArray::from_float32(&wy_vals, &[1, out_h, 1, 1])?;

    // Gather rows: input.take(y0, axis=1) -> [B, 2*H, W, C]
    let rows_y0 = input.take(&y0_arr, 1)?;
    let rows_y1 = input.take(&y1_arr, 1)?;

    // Blend: result_h = rows_y0 * (1 - wy) + rows_y1 * wy
    let one_minus_wy = MxArray::from_float32(
        &wy_vals.iter().map(|w| 1.0 - w).collect::<Vec<f32>>(),
        &[1, out_h, 1, 1],
    )?;
    let h_interp = rows_y0.mul(&one_minus_wy)?.add(&rows_y1.mul(&wy_arr)?)?;

    // --- Step 2: Interpolate along width (axis 2) ---
    let out_w = w * 2;
    let mut x0_idx = Vec::with_capacity(out_w as usize);
    let mut x1_idx = Vec::with_capacity(out_w as usize);
    let mut wx_vals = Vec::with_capacity(out_w as usize);

    for ox in 0..out_w {
        let src_x = (ox as f64 + 0.5) / 2.0 - 0.5;
        let x0_raw = src_x.floor() as i64;
        let x0 = x0_raw.clamp(0, w - 1) as i32;
        let x1 = (x0_raw + 1).clamp(0, w - 1) as i32;
        let wx = (src_x - src_x.floor()) as f32;
        x0_idx.push(x0);
        x1_idx.push(x1);
        wx_vals.push(wx);
    }

    let x0_arr = MxArray::from_int32(&x0_idx, &[out_w])?;
    let x1_arr = MxArray::from_int32(&x1_idx, &[out_w])?;
    // wx shape: [1, 1, 2*W, 1] for broadcasting over [B, 2*H, 2*W, C]
    let wx_arr = MxArray::from_float32(&wx_vals, &[1, 1, out_w, 1])?;

    // Gather columns: h_interp.take(x0, axis=2) -> [B, 2*H, 2*W, C]
    let cols_x0 = h_interp.take(&x0_arr, 2)?;
    let cols_x1 = h_interp.take(&x1_arr, 2)?;

    // Blend: result = cols_x0 * (1 - wx) + cols_x1 * wx
    let one_minus_wx = MxArray::from_float32(
        &wx_vals.iter().map(|w| 1.0 - w).collect::<Vec<f32>>(),
        &[1, 1, out_w, 1],
    )?;
    cols_x0.mul(&one_minus_wx)?.add(&cols_x1.mul(&wx_arr)?)
}

// ============================================================================
// ScaleHead: Conv + optional Upsample chain
// ============================================================================

/// Scale head for the MaskFeatFPN.
///
/// Corresponds to PPDocLayoutV3ScaleHead in the reference implementation.
///
/// Contains a sequence of ConvLayer + optional Upsample pairs.
/// The number of layers depends on log2(fpn_stride / base_stride).
/// Each layer has a 3x3 ConvLayer followed by 2x bilinear upsampling
/// (if fpn_stride != base_stride). Uses align_corners=False to match Python reference.
pub enum ScaleHeadLayer {
    Conv(ConvLayer),
    Upsample,
}

pub struct ScaleHead {
    layers: Vec<ScaleHeadLayer>,
}

impl ScaleHead {
    pub fn new(layers: Vec<ScaleHeadLayer>) -> Self {
        Self { layers }
    }

    /// Forward pass through conv + upsample chain.
    ///
    /// Input/Output: [batch, height, width, channels] (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let mut x = input.clone();
        for layer in &self.layers {
            match layer {
                ScaleHeadLayer::Conv(conv) => {
                    x = conv.forward(&x)?;
                }
                ScaleHeadLayer::Upsample => {
                    x = upsample_bilinear_2x(&x)?;
                }
            }
        }
        Ok(x)
    }
}

// ============================================================================
// MaskFeatFPN
// ============================================================================

/// Multi-level mask feature head producing mask features.
///
/// Corresponds to PPDocLayoutV3MaskFeatFPN in the reference implementation.
///
/// Takes pan_feature_maps (ordered by stride ascending) and produces a
/// single feature map at the finest resolution among the inputs.
///
/// Architecture:
/// 1. Reorder inputs by stride (ascending)
/// 2. Process each level through a ScaleHead (conv + upsample to finest resolution)
/// 3. Sum all processed levels
/// 4. Final 3x3 conv to produce output
pub struct MaskFeatFPN {
    /// Reorder indices to sort by stride ascending
    reorder_index: Vec<usize>,
    /// Scale heads for each level
    scale_heads: Vec<ScaleHead>,
    /// Final output convolution
    output_conv: ConvLayer,
}

impl MaskFeatFPN {
    /// Create a new MaskFeatFPN.
    ///
    /// # Arguments
    /// * `reorder_index` - Indices to reorder inputs by ascending stride
    /// * `scale_heads` - Scale heads for each level
    /// * `output_conv` - Final 3x3 conv
    pub fn new(
        reorder_index: Vec<usize>,
        scale_heads: Vec<ScaleHead>,
        output_conv: ConvLayer,
    ) -> Self {
        Self {
            reorder_index,
            scale_heads,
            output_conv,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `inputs` - Vec of feature maps [batch, H_i, W_i, C] (NHWC), one per level
    ///
    /// # Returns
    /// * Mask feature map [batch, H_finest, W_finest, out_channels] (NHWC)
    pub fn forward(&self, inputs: &[MxArray]) -> Result<MxArray> {
        // Reorder by stride ascending
        let x: Vec<&MxArray> = self.reorder_index.iter().map(|&i| &inputs[i]).collect();

        // Process first level (finest stride)
        let mut output = self.scale_heads[0].forward(x[0])?;

        // Add other levels (upsampled to match output resolution)
        #[allow(clippy::needless_range_loop)]
        for i in 1..self.scale_heads.len() {
            let processed = self.scale_heads[i].forward(x[i])?;

            // Upsample processed to match output spatial size using bilinear interpolation
            // (align_corners=False) to match Python reference's F.interpolate(..., mode="bilinear").
            // The ScaleHead already handles upsampling internally, but we may need
            // to interpolate to match exactly if sizes differ.
            let out_shape = output.shape()?;
            let proc_shape = processed.shape()?;
            let out_h = out_shape[1];
            let out_w = out_shape[2];
            let proc_h = proc_shape[1];
            let proc_w = proc_shape[2];

            let processed = if proc_h != out_h || proc_w != out_w {
                // Use bilinear upsampling to match output size
                // Since we're dealing with power-of-2 relationships, repeated 2x upsampling works
                let mut p = processed;
                loop {
                    let ps = p.shape()?;
                    if ps[1] >= out_h && ps[2] >= out_w {
                        break;
                    }
                    p = upsample_bilinear_2x(&p)?;
                }
                // If we overshot, slice to exact size
                let ps = p.shape()?;
                if ps[1] != out_h || ps[2] != out_w {
                    p.slice(&[0, 0, 0, 0], &[ps[0], out_h, out_w, ps[3]])?
                } else {
                    p
                }
            } else {
                processed
            };

            output = output.add(&processed)?;
        }

        // Final output conv
        self.output_conv.forward(&output)
    }
}

// ============================================================================
// EncoderMaskOutput
// ============================================================================

/// Encoder mask output: ConvLayer + 1x1 Conv2d.
///
/// Corresponds to PPDocLayoutV3EncoderMaskOutput in the reference implementation.
///
/// Produces the final mask prototype features (num_prototypes channels).
pub struct EncoderMaskOutput {
    /// 3x3 ConvLayer (in_channels -> in_channels)
    base_conv: ConvLayer,
    /// 1x1 Conv2d (in_channels -> num_prototypes), with bias
    conv: NativeConv2d,
}

impl EncoderMaskOutput {
    pub fn new(base_conv: ConvLayer, conv: NativeConv2d) -> Self {
        Self { base_conv, conv }
    }

    /// Forward pass.
    ///
    /// Input: [batch, H, W, in_channels] (NHWC)
    /// Output: [batch, H, W, num_prototypes] (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let x = self.base_conv.forward(input)?;
        self.conv.forward(&x)
    }
}

// ============================================================================
// Input Projection: Conv2d(1x1, no bias) + BN
// ============================================================================

/// Input projection layer: 1x1 Conv2d (no bias) + BatchNorm2d.
///
/// Used to project backbone features to the encoder's hidden dimension.
/// Corresponds to `encoder_input_proj` in the model forward.
pub struct InputProjection {
    /// 1x1 Conv2d (no bias)
    conv: NativeConv2d,
    /// Frozen batch normalization
    norm: FrozenBatchNorm2d,
}

impl InputProjection {
    pub fn new(conv: NativeConv2d, norm: FrozenBatchNorm2d) -> Self {
        Self { conv, norm }
    }

    /// Forward pass: Conv2d(1x1) -> BN
    ///
    /// Input: [batch, H, W, in_channels] (NHWC)
    /// Output: [batch, H, W, hidden_dim] (NHWC)
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let x = self.conv.forward(input)?;
        self.norm.forward(&x)
    }
}

// ============================================================================
// HybridEncoder
// ============================================================================

/// PP-DocLayoutV3 Hybrid Encoder.
///
/// Corresponds to PPDocLayoutV3HybridEncoder in the reference implementation.
///
/// Architecture:
/// 1. Input projection (Conv2d 1x1 + BN) for each feature level (done outside, passed as projected)
/// 2. AIFI: Transformer encoder on the highest-level feature with 2D sinusoidal position encoding
/// 3. FPN (top-down): Lateral 1x1 convs + 2x upsample + concat + CSPRepLayer
/// 4. PAN (bottom-up): Stride-2 3x3 convs for downsampling + concat + CSPRepLayer
/// 5. MaskFeatFPN: Multi-level mask feature head
/// 6. Encoder mask lateral + output for final mask features
///
/// Input: Vec of projected feature maps [B, H_i, W_i, 256] (NHWC)
///        + x4_feat [B, H/4, W/4, x4_feat_dim] (stride-4 feature from backbone)
/// Output: (pan_feature_maps, mask_feat)
pub struct HybridEncoder {
    /// Hidden dimension (256)
    encoder_hidden_dim: i32,
    /// Indices of features to apply transformer encoder to
    encode_proj_layers: Vec<i32>,
    /// Temperature for positional encoding
    positional_encoding_temperature: i32,
    /// Number of FPN stages (len(in_channels) - 1)
    num_fpn_stages: usize,
    /// Number of PAN stages (len(in_channels) - 1)
    num_pan_stages: usize,
    /// Number of encoder layers
    encoder_layers: i32,

    // Transformer encoder(s) - one per encode_proj_layer
    encoders: Vec<Encoder>,

    // FPN (top-down)
    /// Lateral 1x1 convolutions for FPN
    lateral_convs: Vec<ConvNormLayer>,
    /// CSPRepLayer blocks for FPN
    fpn_blocks: Vec<CSPRepLayer>,

    // PAN (bottom-up)
    /// Stride-2 3x3 convolutions for downsampling
    downsample_convs: Vec<ConvNormLayer>,
    /// CSPRepLayer blocks for PAN
    pan_blocks: Vec<CSPRepLayer>,

    // Mask feature head
    /// MaskFeatFPN for multi-level mask features
    mask_feature_head: MaskFeatFPN,
    /// Lateral convolution for x4_feat
    encoder_mask_lateral: ConvLayer,
    /// Final mask output (ConvLayer + 1x1 Conv)
    encoder_mask_output: EncoderMaskOutput,
}

impl HybridEncoder {
    /// Create a new HybridEncoder.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &PPDocLayoutV3Config,
        encoders: Vec<Encoder>,
        lateral_convs: Vec<ConvNormLayer>,
        fpn_blocks: Vec<CSPRepLayer>,
        downsample_convs: Vec<ConvNormLayer>,
        pan_blocks: Vec<CSPRepLayer>,
        mask_feature_head: MaskFeatFPN,
        encoder_mask_lateral: ConvLayer,
        encoder_mask_output: EncoderMaskOutput,
    ) -> Self {
        let num_levels = config.encoder_in_channels.len();
        Self {
            encoder_hidden_dim: config.encoder_hidden_dim,
            encode_proj_layers: config.encode_proj_layers.clone(),
            positional_encoding_temperature: config.positional_encoding_temperature,
            num_fpn_stages: num_levels - 1,
            num_pan_stages: num_levels - 1,
            encoder_layers: config.encoder_layers,
            encoders,
            lateral_convs,
            fpn_blocks,
            downsample_convs,
            pan_blocks,
            mask_feature_head,
            encoder_mask_lateral,
            encoder_mask_output,
        }
    }

    /// Forward pass through the hybrid encoder.
    ///
    /// # Arguments
    /// * `inputs_embeds` - Vec of projected feature maps [B, H_i, W_i, 256] (NHWC)
    ///   Ordered from finest to coarsest resolution (stride 8, 16, 32)
    /// * `x4_feat` - Stride-4 feature map [B, H/4, W/4, x4_feat_dim] (NHWC)
    ///
    /// # Returns
    /// * `(pan_feature_maps, mask_feat)` where:
    ///   - pan_feature_maps: Vec of [B, H_i, W_i, 256] at each stride level
    ///   - mask_feat: [B, H/4, W/4, num_prototypes] mask prototype features
    pub fn forward(
        &self,
        inputs_embeds: &mut [MxArray],
        x4_feat: &MxArray,
    ) -> Result<(Vec<MxArray>, MxArray)> {
        // ---- AIFI: Transformer encoder on selected feature levels ----
        if self.encoder_layers > 0 {
            for (i, &enc_ind) in self.encode_proj_layers.iter().enumerate() {
                let enc_ind = enc_ind as usize;

                let feat_shape = inputs_embeds[enc_ind].shape()?;
                let batch = feat_shape[0];
                let height = feat_shape[1];
                let width = feat_shape[2];

                // Flatten spatial dims: [B, H, W, C] -> [B, H*W, C]
                let src_flatten = inputs_embeds[enc_ind].reshape(&[
                    batch,
                    height * width,
                    self.encoder_hidden_dim as i64,
                ])?;

                // Build 2D sinusoidal position embedding
                let pos_embed = build_2d_sincos_position_embedding(
                    width,
                    height,
                    self.encoder_hidden_dim as i64,
                    self.positional_encoding_temperature as f64,
                )?;

                // Run through transformer encoder
                let encoded = self.encoders[i].forward(&src_flatten, Some(&pos_embed))?;

                // Reshape back: [B, H*W, C] -> [B, H, W, C]
                inputs_embeds[enc_ind] =
                    encoded.reshape(&[batch, height, width, self.encoder_hidden_dim as i64])?;
            }
        }

        // ---- FPN: Top-down path ----
        // Start from the coarsest (highest-level) feature
        let num_levels = inputs_embeds.len();
        let mut fpn_feature_maps: Vec<MxArray> = vec![inputs_embeds[num_levels - 1].clone()];

        for idx in 0..self.num_fpn_stages {
            // Get the backbone feature at the corresponding level (going top-down)
            let backbone_feature_map = &inputs_embeds[self.num_fpn_stages - idx - 1];

            // Apply lateral conv to current top feature
            let top_fpn = self.lateral_convs[idx].forward(&fpn_feature_maps[idx])?;
            // Update in place
            fpn_feature_maps[idx] = top_fpn.clone();

            // Upsample by 2x
            let upsampled = upsample_nearest_2x(&top_fpn)?;

            // Concatenate along channel axis (NHWC: axis=3)
            let fused = MxArray::concatenate(&upsampled, backbone_feature_map, 3)?;

            // Apply CSPRepLayer
            let new_fpn = self.fpn_blocks[idx].forward(&fused)?;
            fpn_feature_maps.push(new_fpn);
        }

        // Reverse so finest resolution is first
        fpn_feature_maps.reverse();

        // ---- PAN: Bottom-up path ----
        let mut pan_feature_maps: Vec<MxArray> = vec![fpn_feature_maps[0].clone()];

        for idx in 0..self.num_pan_stages {
            let top_pan = &pan_feature_maps[idx];
            let fpn_feat = &fpn_feature_maps[idx + 1];

            // Downsample with stride-2 conv
            let downsampled = self.downsample_convs[idx].forward(top_pan)?;

            // Concatenate along channel axis (NHWC: axis=3)
            let fused = MxArray::concatenate(&downsampled, fpn_feat, 3)?;

            // Apply CSPRepLayer
            let new_pan = self.pan_blocks[idx].forward(&fused)?;
            pan_feature_maps.push(new_pan);
        }

        // ---- Mask feature head ----
        let mut mask_feat = self.mask_feature_head.forward(&pan_feature_maps)?;

        // Upsample mask_feat by 2x (to go from stride-8 to stride-4)
        // Uses bilinear interpolation (align_corners=False) to match Python reference
        mask_feat = upsample_bilinear_2x(&mask_feat)?;

        // Add x4_feat (stride-4 feature from backbone) via lateral conv
        let x4_lateral = self.encoder_mask_lateral.forward(x4_feat)?;
        mask_feat = mask_feat.add(&x4_lateral)?;

        // Final mask output
        mask_feat = self.encoder_mask_output.forward(&mask_feat)?;

        Ok((pan_feature_maps, mask_feat))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_2d_sincos_position_embedding() {
        let pos = build_2d_sincos_position_embedding(4, 3, 256, 10000.0).unwrap();
        let shape: Vec<i64> = pos.shape().unwrap().as_ref().to_vec();
        // [1, H*W, embed_dim] = [1, 12, 256]
        assert_eq!(shape, vec![1, 12, 256]);

        // Values should be in [-1, 1] (sin/cos range)
        pos.eval();
        let data = pos.to_float32().unwrap();
        let data: Vec<f32> = data.to_vec();
        for &v in &data {
            assert!(
                (-1.0..=1.0).contains(&v),
                "Position embedding value {v} out of range [-1, 1]"
            );
        }
    }

    #[test]
    fn test_build_2d_sincos_position_embedding_rejects_bad_dim() {
        let result = build_2d_sincos_position_embedding(4, 3, 255, 10000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_upsample_nearest_2x() {
        // Input: [1, 2, 2, 1]
        let input = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]).unwrap();

        let output = upsample_nearest_2x(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 4, 4, 1]);

        output.eval();
        let data = output.to_float32().unwrap();
        let data: Vec<f32> = data.to_vec();
        // Expected: each pixel repeated in a 2x2 block
        // [1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4]
        assert_eq!(
            data,
            vec![
                1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0
            ]
        );
    }

    #[test]
    fn test_upsample_nearest_2x_multichannel() {
        // Input: [1, 2, 2, 2]
        let input =
            MxArray::from_float32(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], &[1, 2, 2, 2])
                .unwrap();

        let output = upsample_nearest_2x(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 4, 4, 2]);
    }

    #[test]
    fn test_upsample_bilinear_2x_shape() {
        // Input: [1, 3, 4, 2]
        let input = MxArray::from_float32(&[1.0; 3 * 4 * 2], &[1, 3, 4, 2]).unwrap();

        let output = upsample_bilinear_2x(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 6, 8, 2]);
    }

    #[test]
    fn test_upsample_bilinear_2x_values() {
        // Input: [1, 2, 2, 1] with values [[1, 2], [3, 4]]
        // Test that bilinear interpolation produces smooth output
        let input = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]).unwrap();

        let output = upsample_bilinear_2x(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 4, 4, 1]);

        output.eval();
        let data: Vec<f32> = output.to_float32().unwrap().to_vec();

        // With align_corners=False and scale_factor=2:
        // Output pixel (oy, ox) maps to src = ((oy+0.5)/2 - 0.5, (ox+0.5)/2 - 0.5)
        //
        // oy=0: src_y = -0.25 -> y0=0,y1=0 (both clamped), wy=0.75 (irrelevant, same row)
        // oy=1: src_y =  0.25 -> y0=0,y1=1, wy=0.25
        // oy=2: src_y =  0.75 -> y0=0,y1=1, wy=0.75
        // oy=3: src_y =  1.25 -> y0=1,y1=1 (y1 clamped), wy=0.25 (irrelevant, same row)
        //
        // ox=0: src_x = -0.25 -> x0=0,x1=0, wx=0.75 (irrelevant)
        // ox=1: src_x =  0.25 -> x0=0,x1=1, wx=0.25
        // ox=2: src_x =  0.75 -> x0=0,x1=1, wx=0.75
        // ox=3: src_x =  1.25 -> x0=1,x1=1, wx=0.25 (irrelevant)
        //
        // Row 0 (src_y clamps to 0): input row = [1, 2]
        //   (0,0): 1.0*1.0 = 1.0
        //   (0,1): 0.75*1.0 + 0.25*2.0 = 1.25
        //   (0,2): 0.25*1.0 + 0.75*2.0 = 1.75
        //   (0,3): 1.0*2.0 = 2.0
        //
        // Row 1 (src_y=0.25): interp = 0.75*row0 + 0.25*row1
        //   row0=[1,2], row1=[3,4] -> h_interp = [1.5, 2.5]
        //   (1,0): 1.5
        //   (1,1): 0.75*1.5 + 0.25*2.5 = 1.75
        //   (1,2): 0.25*1.5 + 0.75*2.5 = 2.25
        //   (1,3): 2.5
        //
        // Row 2 (src_y=0.75): interp = 0.25*row0 + 0.75*row1
        //   -> h_interp = [2.5, 3.5]
        //   (2,0): 2.5
        //   (2,1): 0.75*2.5 + 0.25*3.5 = 2.75
        //   (2,2): 0.25*2.5 + 0.75*3.5 = 3.25
        //   (2,3): 3.5
        //
        // Row 3 (src_y clamps to 1): input row = [3, 4]
        //   (3,0): 3.0
        //   (3,1): 0.75*3.0 + 0.25*4.0 = 3.25
        //   (3,2): 0.25*3.0 + 0.75*4.0 = 3.75
        //   (3,3): 4.0
        let expected = vec![
            1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0,
        ];
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at index {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_upsample_bilinear_2x_uniform() {
        // A uniform input should produce uniform output
        let input = MxArray::from_float32(&[5.0; 3 * 3], &[1, 3, 3, 1]).unwrap();

        let output = upsample_bilinear_2x(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 6, 6, 1]);

        output.eval();
        let data: Vec<f32> = output.to_float32().unwrap().to_vec();
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "Uniform input should give uniform output, but index {i} = {v}"
            );
        }
    }

    #[test]
    fn test_upsample_bilinear_2x_multichannel() {
        // Input: [1, 2, 2, 3] — multi-channel should work independently per channel
        let input = MxArray::from_float32(
            &[
                // pixel (0,0): channels [1, 10, 100]
                1.0, 10.0, 100.0, // pixel (0,1): channels [2, 20, 200]
                2.0, 20.0, 200.0, // pixel (1,0): channels [3, 30, 300]
                3.0, 30.0, 300.0, // pixel (1,1): channels [4, 40, 400]
                4.0, 40.0, 400.0,
            ],
            &[1, 2, 2, 3],
        )
        .unwrap();

        let output = upsample_bilinear_2x(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 4, 4, 3]);

        output.eval();
        let data: Vec<f32> = output.to_float32().unwrap().to_vec();
        // Channel 1 should be 10x channel 0, channel 2 should be 100x channel 0
        // Check center pixel (1,1) which is a blend: channel 0 should be 1.75
        // (row 1: 0.75*[1,2] + 0.25*[3,4] = [1.5, 2.5], then col 1: 0.75*1.5 + 0.25*2.5 = 1.75)
        let ch0_11 = data[(4 + 1) * 3]; // row=1, col=1, ch=0
        let ch1_11 = data[(4 + 1) * 3 + 1]; // row=1, col=1, ch=1
        let ch2_11 = data[(4 + 1) * 3 + 2]; // row=1, col=1, ch=2
        assert!((ch0_11 - 1.75).abs() < 1e-5, "ch0 at (1,1) = {ch0_11}");
        assert!((ch1_11 - 17.5).abs() < 1e-4, "ch1 at (1,1) = {ch1_11}");
        assert!((ch2_11 - 175.0).abs() < 1e-3, "ch2 at (1,1) = {ch2_11}");
    }

    #[test]
    fn test_conv_norm_layer_shape() {
        // Create a simple 1x1 conv with identity BN
        let channels = 4;
        let weight = MxArray::from_float32(
            &vec![0.1; (channels * channels) as usize],
            &[channels as i64, 1, 1, channels as i64],
        )
        .unwrap();
        let bn_weight = MxArray::ones(&[channels as i64], None).unwrap();
        let bn_bias = MxArray::zeros(&[channels as i64], None).unwrap();
        let bn_mean = MxArray::zeros(&[channels as i64], None).unwrap();
        let bn_var = MxArray::ones(&[channels as i64], None).unwrap();

        let conv = NativeConv2d::new(&weight, None, (1, 1), (0, 0), (1, 1), 1);
        let norm = FrozenBatchNorm2d::new(&bn_weight, &bn_bias, &bn_mean, &bn_var, 1e-5);
        let layer = ConvNormLayer::new(conv, norm, Some("relu"));

        let input = MxArray::from_float32(
            &vec![1.0; (4 * 4 * channels) as usize],
            &[1, 4, 4, channels as i64],
        )
        .unwrap();

        let output = layer.forward(&input).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 4, 4, channels as i64]);
    }
}
