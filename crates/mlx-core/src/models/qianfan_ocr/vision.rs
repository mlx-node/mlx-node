//! InternViT Vision Encoder
//!
//! 24-layer Vision Transformer with:
//! - Learned absolute position embeddings (NOT rotary)
//! - CLS token (prepended to patches, dropped after encoding)
//! - Fused QKV attention (single linear for Q, K, V)
//! - Standard GELU MLP (NOT gated SiLU)
//! - LayerNorm (NOT RMSNorm)
//! - Layer scale (learnable per-element multipliers on attention/MLP outputs)
//! - DropPath is identity at inference (no-op, no weights)

use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::pp_doclayout_v3::persistence::get_tensor;
use crate::models::qianfan_ocr::config::InternVisionConfig;
use crate::nn::activations::Activations;
use crate::nn::{LayerNorm, Linear};
use crate::vision::conv2d::Conv2d;
use crate::vision::interpolate::bilinear_interpolate;

// ============================================================================
// InternVisionEmbeddings
// ============================================================================

/// Patch embedding + CLS token + learned absolute position embedding.
pub(crate) struct InternVisionEmbeddings {
    /// Conv2d(3, hidden_size, kernel=patch_size, stride=patch_size)
    patch_conv: Conv2d,
    /// Learnable CLS token [1, 1, hidden_size]
    cls_token: MxArray,
    /// Learnable position embedding [1, num_positions, hidden_size]
    /// where num_positions = (image_size / patch_size)^2 + 1 (patches + CLS)
    position_embedding: MxArray,
    patch_size: u32,
    /// Default grid size = image_size / patch_size (e.g. 32 for 448/14)
    default_grid_size: u32,
}

impl InternVisionEmbeddings {
    pub fn build(
        weights: &HashMap<String, MxArray>,
        prefix: &str,
        config: &InternVisionConfig,
    ) -> Result<Self> {
        // Conv2d weight: [out_channels, kernel_h, kernel_w, in_channels]  (MLX OHWI)
        let conv_weight = get_tensor(weights, &format!("{prefix}.patch_embedding.weight"))?;
        let conv_bias = get_tensor(weights, &format!("{prefix}.patch_embedding.bias"))?;
        let patch_size = config.patch_size as u32;
        let patch_conv = Conv2d::new(
            &conv_weight,
            Some(&conv_bias),
            Some(vec![patch_size, patch_size]),
            Some(vec![0, 0]),
            None,
            None,
        )?;

        // Accept both key names: HuggingFace uses "class_embedding",
        // some checkpoints/docs may use "cls_token"
        let cls_key = format!("{prefix}.class_embedding");
        let cls_token = weights
            .get(&cls_key)
            .or_else(|| weights.get(&format!("{prefix}.cls_token")))
            .cloned()
            .ok_or_else(|| {
                Error::from_reason(format!("Missing weight: {cls_key} (or cls_token)"))
            })?;
        let position_embedding = get_tensor(weights, &format!("{prefix}.position_embedding"))?;

        let default_grid_size = config.image_size as u32 / patch_size;

        Ok(Self {
            patch_conv,
            cls_token,
            position_embedding,
            patch_size,
            default_grid_size,
        })
    }

    /// Forward: pixel_values [B, H, W, 3] (NHWC) -> [B, num_patches+1, hidden_size]
    pub fn forward(&self, pixel_values: &MxArray) -> Result<MxArray> {
        let shape = pixel_values.shape()?;
        let batch = shape[0];
        let h = shape[1];
        let w = shape[2];

        // 1. Patch conv: [B, H, W, 3] -> [B, grid_h, grid_w, hidden_size]
        let patch_embeds = self.patch_conv.forward(pixel_values)?;

        let grid_h = h / self.patch_size as i64;
        let grid_w = w / self.patch_size as i64;
        let num_patches = grid_h * grid_w;
        let hidden_size = patch_embeds.shape()?[3];

        // 2. Flatten spatial: [B, num_patches, hidden_size]
        let patch_embeds = patch_embeds.reshape(&[batch, num_patches, hidden_size])?;

        // 3. Expand CLS token to batch: cls_token is [1, 1, hidden_size]
        let cls_expanded = self.cls_token.broadcast_to(&[batch, 1, hidden_size])?;

        // 4. Concatenate: [CLS, patches] -> [B, num_patches+1, hidden_size]
        let embeddings = MxArray::concatenate(&cls_expanded, &patch_embeds, 1)?;

        // 5. Add position embedding (with interpolation if grid size differs)
        let pos_embed = self.get_position_embedding(grid_h, grid_w)?;
        let embeddings = embeddings.add(&pos_embed)?;

        Ok(embeddings)
    }

    /// Get position embedding, interpolating if the grid size differs from default.
    fn get_position_embedding(&self, grid_h: i64, grid_w: i64) -> Result<MxArray> {
        let default = self.default_grid_size as i64;

        if grid_h == default && grid_w == default {
            return Ok(self.position_embedding.clone());
        }

        // position_embedding: [1, num_positions, hidden_size]
        // Split into CLS (first token) and patch positions (rest)
        let cls_pos = self.position_embedding.slice_axis(1, 0, 1)?; // [1, 1, D]
        let patch_pos = self
            .position_embedding
            .slice_axis(1, 1, default * default + 1)?; // [1, N, D]

        let hidden_size = self.position_embedding.shape()?[2];

        // Reshape patch_pos to [default, default, hidden_size] for interpolation
        let patch_pos_2d = patch_pos.reshape(&[default, default, hidden_size])?;

        // Bilinear interpolate to [grid_h, grid_w, hidden_size]
        let patch_pos_interp = bilinear_interpolate(&patch_pos_2d, grid_h, grid_w)?;

        // Flatten back to [1, grid_h * grid_w, hidden_size]
        let patch_pos_flat = patch_pos_interp.reshape(&[1, grid_h * grid_w, hidden_size])?;

        // Re-concatenate: [1, 1 + grid_h*grid_w, hidden_size]
        MxArray::concatenate(&cls_pos, &patch_pos_flat, 1)
    }
}

impl Clone for InternVisionEmbeddings {
    fn clone(&self) -> Self {
        Self {
            patch_conv: Conv2d::new(
                &self.patch_conv.weight(),
                self.patch_conv.bias().as_ref(),
                Some(vec![self.patch_size, self.patch_size]),
                Some(vec![0, 0]),
                None,
                None,
            )
            .expect("clone Conv2d"),
            cls_token: self.cls_token.clone(),
            position_embedding: self.position_embedding.clone(),
            patch_size: self.patch_size,
            default_grid_size: self.default_grid_size,
        }
    }
}

// ============================================================================
// InternVisionAttention
// ============================================================================

/// Fused QKV self-attention (no RoPE, no masking).
pub(crate) struct InternVisionAttention {
    /// Fused QKV projection [hidden_size, 3*hidden_size] with bias
    qkv: Linear,
    /// Output projection [hidden_size, hidden_size]
    out_proj: Linear,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
}

impl InternVisionAttention {
    pub fn build(
        weights: &HashMap<String, MxArray>,
        prefix: &str,
        config: &InternVisionConfig,
    ) -> Result<Self> {
        let qkv_weight = get_tensor(weights, &format!("{prefix}.attn.qkv.weight"))?;
        let qkv_bias = if config.qkv_bias {
            Some(get_tensor(weights, &format!("{prefix}.attn.qkv.bias"))?)
        } else {
            None
        };
        let qkv = Linear::from_weights(&qkv_weight, qkv_bias.as_ref())?;

        let proj_weight = get_tensor(weights, &format!("{prefix}.attn.proj.weight"))?;
        let proj_bias = get_tensor(weights, &format!("{prefix}.attn.proj.bias"))?;
        let out_proj = Linear::from_weights(&proj_weight, Some(&proj_bias))?;

        let num_heads = config.num_attention_heads as u32;
        let head_dim = config.hidden_size as u32 / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            qkv,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    /// Forward: input [B, N, D] -> [B, N, D]
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let shape = input.shape()?;
        let b = shape[0];
        let n = shape[1];
        let nh = self.num_heads as i64;
        let hd = self.head_dim as i64;

        // 1. QKV projection: [B, N, D] -> [B, N, 3*D]
        let qkv = self.qkv.forward(input)?;

        // 2. Reshape: [B, N, 3, num_heads, head_dim]
        let qkv = qkv.reshape(&[b, n, 3, nh, hd])?;

        // 3. Permute to [3, B, num_heads, N, head_dim]
        let qkv = qkv.transpose(Some(&[2, 0, 3, 1, 4]))?;

        // 4. Split Q, K, V: each [B, num_heads, N, head_dim]
        let q = qkv.slice_axis(0, 0, 1)?.squeeze(Some(&[0]))?;
        let k = qkv.slice_axis(0, 1, 2)?.squeeze(Some(&[0]))?;
        let v = qkv.slice_axis(0, 2, 3)?.squeeze(Some(&[0]))?;

        // 5. Scaled dot-product attention
        // Q @ K^T: [B, nh, N, hd] @ [B, nh, hd, N] -> [B, nh, N, N]
        let k_t = k.transpose(Some(&[0, 1, 3, 2]))?;
        let attn_weights = q.matmul(&k_t)?.mul_scalar(self.scale as f64)?;
        let attn_weights = Activations::softmax(&attn_weights, Some(-1))?;

        // attn_weights @ V: [B, nh, N, N] @ [B, nh, N, hd] -> [B, nh, N, hd]
        let attn_output = attn_weights.matmul(&v)?;

        // 6. Permute back: [B, N, nh, hd] -> [B, N, D]
        let attn_output = attn_output.transpose(Some(&[0, 2, 1, 3]))?;
        let attn_output = attn_output.reshape(&[b, n, nh * hd])?;

        // 7. Output projection
        self.out_proj.forward(&attn_output)
    }
}

impl Clone for InternVisionAttention {
    fn clone(&self) -> Self {
        Self {
            qkv: self.qkv.clone(),
            out_proj: self.out_proj.clone(),
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            scale: self.scale,
        }
    }
}

// ============================================================================
// InternVisionMLP
// ============================================================================

/// Standard MLP with GELU activation (not gated SiLU).
pub(crate) struct InternVisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl InternVisionMLP {
    pub fn build(weights: &HashMap<String, MxArray>, prefix: &str) -> Result<Self> {
        let fc1 = Linear::from_weights(
            &get_tensor(weights, &format!("{prefix}.mlp.fc1.weight"))?,
            Some(&get_tensor(weights, &format!("{prefix}.mlp.fc1.bias"))?),
        )?;
        let fc2 = Linear::from_weights(
            &get_tensor(weights, &format!("{prefix}.mlp.fc2.weight"))?,
            Some(&get_tensor(weights, &format!("{prefix}.mlp.fc2.bias"))?),
        )?;

        Ok(Self { fc1, fc2 })
    }

    /// Forward: x -> fc1 -> GELU -> fc2
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let hidden = self.fc1.forward(x)?;
        let activated = Activations::gelu(&hidden)?;
        self.fc2.forward(&activated)
    }
}

impl Clone for InternVisionMLP {
    fn clone(&self) -> Self {
        Self {
            fc1: self.fc1.clone(),
            fc2: self.fc2.clone(),
        }
    }
}

// ============================================================================
// InternVisionEncoderLayer
// ============================================================================

/// Pre-norm transformer block with layer scale.
///
/// Forward:
///   h = x + layer_scale_1 * attn(layer_norm1(x))
///   output = h + layer_scale_2 * mlp(layer_norm2(h))
pub(crate) struct InternVisionEncoderLayer {
    layer_norm1: LayerNorm,
    attn: InternVisionAttention,
    /// Per-element multiplier [hidden_size] applied to attention output
    layer_scale_1: MxArray,
    layer_norm2: LayerNorm,
    mlp: InternVisionMLP,
    /// Per-element multiplier [hidden_size] applied to MLP output
    layer_scale_2: MxArray,
}

impl InternVisionEncoderLayer {
    pub fn build(
        weights: &HashMap<String, MxArray>,
        prefix: &str,
        config: &InternVisionConfig,
    ) -> Result<Self> {
        let eps = Some(config.layer_norm_eps);

        let layer_norm1 = LayerNorm::from_weights(
            &get_tensor(weights, &format!("{prefix}.norm1.weight"))?,
            Some(&get_tensor(weights, &format!("{prefix}.norm1.bias"))?),
            eps,
        )?;
        let layer_norm2 = LayerNorm::from_weights(
            &get_tensor(weights, &format!("{prefix}.norm2.weight"))?,
            Some(&get_tensor(weights, &format!("{prefix}.norm2.bias"))?),
            eps,
        )?;

        let attn = InternVisionAttention::build(weights, prefix, config)?;
        let mlp = InternVisionMLP::build(weights, prefix)?;

        let layer_scale_1 = get_tensor(weights, &format!("{prefix}.ls1"))?;
        let layer_scale_2 = get_tensor(weights, &format!("{prefix}.ls2"))?;

        Ok(Self {
            layer_norm1,
            attn,
            layer_scale_1,
            layer_norm2,
            mlp,
            layer_scale_2,
        })
    }

    /// Forward pass with pre-norm and layer scale.
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        // h = x + layer_scale_1 * attn(layer_norm1(x))
        let normed1 = self.layer_norm1.forward(x)?;
        let attn_out = self.attn.forward(&normed1)?;
        let scaled_attn = attn_out.mul(&self.layer_scale_1)?;
        let h = x.add(&scaled_attn)?;

        // output = h + layer_scale_2 * mlp(layer_norm2(h))
        let normed2 = self.layer_norm2.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        let scaled_mlp = mlp_out.mul(&self.layer_scale_2)?;
        h.add(&scaled_mlp)
    }
}

impl Clone for InternVisionEncoderLayer {
    fn clone(&self) -> Self {
        Self {
            layer_norm1: self.layer_norm1.clone(),
            attn: self.attn.clone(),
            layer_scale_1: self.layer_scale_1.clone(),
            layer_norm2: self.layer_norm2.clone(),
            mlp: self.mlp.clone(),
            layer_scale_2: self.layer_scale_2.clone(),
        }
    }
}

// ============================================================================
// InternViTModel
// ============================================================================

/// Full InternViT vision encoder.
///
/// Forward pass:
/// 1. Embed patches + CLS + add position embedding
/// 2. Pass through encoder layers (up to select_layer)
/// 3. Drop CLS token (first position)
/// 4. Return [B, num_patches, hidden_size]
pub(crate) struct InternViTModel {
    embeddings: InternVisionEmbeddings,
    layers: Vec<InternVisionEncoderLayer>,
    /// Which layer to extract features from. -1 = last layer.
    select_layer: i32,
}

impl InternViTModel {
    pub fn build(
        weights: &HashMap<String, MxArray>,
        prefix: &str,
        config: &InternVisionConfig,
        select_layer: i32,
    ) -> Result<Self> {
        let embeddings =
            InternVisionEmbeddings::build(weights, &format!("{prefix}.embeddings"), config)?;

        let num_layers = config.num_hidden_layers as usize;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer =
                InternVisionEncoderLayer::build(weights, &format!("{prefix}.layers.{i}"), config)?;
            layers.push(layer);
        }

        Ok(Self {
            embeddings,
            layers,
            select_layer,
        })
    }

    /// Forward: pixel_values [B, H, W, 3] (NHWC) -> [B, num_patches, hidden_size]
    pub fn forward(&self, pixel_values: &MxArray) -> Result<MxArray> {
        // 1. Embeddings: [B, num_patches+1, hidden_size]
        let mut hidden_states = self.embeddings.forward(pixel_values)?;

        // 2. Determine how many layers to run
        let num_layers = self.layers.len() as i32;
        let end_layer = if self.select_layer < 0 {
            // -1 means last layer (all layers), -2 means second-to-last, etc.
            (num_layers + self.select_layer + 1).max(0) as usize
        } else {
            (self.select_layer + 1).min(num_layers) as usize
        };

        for layer in &self.layers[..end_layer] {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // 3. Drop CLS token (first position): [B, num_patches+1, D] -> [B, num_patches, D]
        let seq_len = hidden_states.shape()?[1];
        hidden_states.slice_axis(1, 1, seq_len)
    }

    /// Number of encoder layers
    #[cfg(test)]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Clone for InternViTModel {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            layers: self.layers.clone(),
            select_layer: self.select_layer,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a weight map with random weights for testing.
    fn make_test_weights(config: &InternVisionConfig, prefix: &str) -> HashMap<String, MxArray> {
        let d = config.hidden_size as i64;
        let inter = config.intermediate_size as i64;
        let ps = config.patch_size as i64;
        let ch = config.num_channels as i64;
        let grid = config.image_size as i64 / ps;
        let num_positions = grid * grid + 1; // patches + CLS

        let mut w: HashMap<String, MxArray> = HashMap::new();

        // Embeddings
        let ep = format!("{prefix}.embeddings");
        // Conv weight: [out_channels, kernel_h, kernel_w, in_channels] (MLX OHWI)
        w.insert(
            format!("{ep}.patch_embedding.weight"),
            MxArray::random_normal(&[d, ps, ps, ch], 0.0, 0.02, None).unwrap(),
        );
        w.insert(
            format!("{ep}.patch_embedding.bias"),
            MxArray::zeros(&[d], None).unwrap(),
        );
        w.insert(
            format!("{ep}.class_embedding"),
            MxArray::random_normal(&[1, 1, d], 0.0, 0.02, None).unwrap(),
        );
        w.insert(
            format!("{ep}.position_embedding"),
            MxArray::random_normal(&[1, num_positions, d], 0.0, 0.02, None).unwrap(),
        );

        // Encoder layers
        for i in 0..config.num_hidden_layers {
            let lp = format!("{prefix}.layers.{i}");
            // LayerNorm
            w.insert(
                format!("{lp}.norm1.weight"),
                MxArray::ones(&[d], None).unwrap(),
            );
            w.insert(
                format!("{lp}.norm1.bias"),
                MxArray::zeros(&[d], None).unwrap(),
            );
            w.insert(
                format!("{lp}.norm2.weight"),
                MxArray::ones(&[d], None).unwrap(),
            );
            w.insert(
                format!("{lp}.norm2.bias"),
                MxArray::zeros(&[d], None).unwrap(),
            );
            // Attention
            w.insert(
                format!("{lp}.attn.qkv.weight"),
                MxArray::random_normal(&[3 * d, d], 0.0, 0.02, None).unwrap(),
            );
            w.insert(
                format!("{lp}.attn.qkv.bias"),
                MxArray::zeros(&[3 * d], None).unwrap(),
            );
            w.insert(
                format!("{lp}.attn.proj.weight"),
                MxArray::random_normal(&[d, d], 0.0, 0.02, None).unwrap(),
            );
            w.insert(
                format!("{lp}.attn.proj.bias"),
                MxArray::zeros(&[d], None).unwrap(),
            );
            // MLP
            w.insert(
                format!("{lp}.mlp.fc1.weight"),
                MxArray::random_normal(&[inter, d], 0.0, 0.02, None).unwrap(),
            );
            w.insert(
                format!("{lp}.mlp.fc1.bias"),
                MxArray::zeros(&[inter], None).unwrap(),
            );
            w.insert(
                format!("{lp}.mlp.fc2.weight"),
                MxArray::random_normal(&[d, inter], 0.0, 0.02, None).unwrap(),
            );
            w.insert(
                format!("{lp}.mlp.fc2.bias"),
                MxArray::zeros(&[d], None).unwrap(),
            );
            // Layer scales
            w.insert(format!("{lp}.ls1"), MxArray::ones(&[d], None).unwrap());
            w.insert(format!("{lp}.ls2"), MxArray::ones(&[d], None).unwrap());
        }

        w
    }

    /// Create a small config for fast tests (2 layers, small dims).
    fn small_config() -> InternVisionConfig {
        InternVisionConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            image_size: 28,
            patch_size: 14,
            layer_norm_eps: 1e-6,
            qkv_bias: true,
            drop_path_rate: 0.0,
        }
    }

    #[test]
    fn test_attention_output_shape() {
        let config = small_config();
        let prefix = "vision_model";
        let weights = make_test_weights(&config, prefix);

        let layer_prefix = format!("{prefix}.layers.0");
        let attn = InternVisionAttention::build(&weights, &layer_prefix, &config).unwrap();

        let d = config.hidden_size as i64;
        let b = 1i64;
        let n = 5i64; // 4 patches + 1 CLS
        let input = MxArray::random_normal(&[b, n, d], 0.0, 1.0, None).unwrap();

        let output = attn.forward(&input).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();
        assert_eq!(shape, vec![b, n, d]);
    }

    #[test]
    fn test_mlp_output_shape() {
        let config = small_config();
        let prefix = "vision_model";
        let weights = make_test_weights(&config, prefix);

        let layer_prefix = format!("{prefix}.layers.0");
        let mlp = InternVisionMLP::build(&weights, &layer_prefix).unwrap();

        let d = config.hidden_size as i64;
        let b = 1i64;
        let n = 5i64;
        let input = MxArray::random_normal(&[b, n, d], 0.0, 1.0, None).unwrap();

        let output = mlp.forward(&input).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();
        assert_eq!(
            shape,
            vec![b, n, d],
            "MLP output shape must match input shape"
        );
    }

    #[test]
    fn test_encoder_layer_residual_shape() {
        let config = small_config();
        let prefix = "vision_model";
        let weights = make_test_weights(&config, prefix);

        let layer_prefix = format!("{prefix}.layers.0");
        let layer = InternVisionEncoderLayer::build(&weights, &layer_prefix, &config).unwrap();

        let d = config.hidden_size as i64;
        let b = 1i64;
        let n = 5i64;
        let input = MxArray::random_normal(&[b, n, d], 0.0, 1.0, None).unwrap();

        let output = layer.forward(&input).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();
        assert_eq!(
            shape,
            vec![b, n, d],
            "Encoder layer must preserve shape (residual)"
        );
    }

    #[test]
    fn test_full_model_drops_cls_token() {
        // With image_size=28, patch_size=14: grid = 2x2 = 4 patches
        // After embeddings: [B, 5, D] (4 patches + CLS)
        // After model: [B, 4, D] (CLS dropped)
        let config = small_config();
        let prefix = "vision_model";
        let weights = make_test_weights(&config, prefix);

        let model = InternViTModel::build(&weights, prefix, &config, -1).unwrap();

        let b = 1i64;
        let h = config.image_size as i64;
        let w = config.image_size as i64;
        let c = config.num_channels as i64;
        let pixel_values = MxArray::random_normal(&[b, h, w, c], 0.0, 1.0, None).unwrap();

        let output = model.forward(&pixel_values).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();

        let grid = config.image_size / config.patch_size;
        let expected_patches = (grid * grid) as i64;
        assert_eq!(
            shape,
            vec![b, expected_patches, config.hidden_size as i64],
            "Output should have num_patches (no CLS)"
        );
    }

    #[test]
    fn test_select_layer_subset() {
        let config = small_config();
        let prefix = "vision_model";
        let weights = make_test_weights(&config, prefix);

        // select_layer = 0 means run only the first layer
        let model = InternViTModel::build(&weights, prefix, &config, 0).unwrap();
        assert_eq!(model.num_layers(), 2, "Model should have 2 layers");

        let b = 1i64;
        let h = config.image_size as i64;
        let w = config.image_size as i64;
        let c = config.num_channels as i64;
        let pixel_values = MxArray::random_normal(&[b, h, w, c], 0.0, 1.0, None).unwrap();

        let output = model.forward(&pixel_values).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();

        let grid = config.image_size / config.patch_size;
        let expected_patches = (grid * grid) as i64;
        assert_eq!(shape, vec![b, expected_patches, config.hidden_size as i64],);
    }

    #[test]
    fn test_position_embedding_interpolation() {
        // Build a model with default grid 2x2 (image_size=28, patch_size=14)
        // Then feed a 42x42 image which gives a 3x3 grid -> needs interpolation
        let config = small_config();
        let prefix = "vision_model";
        let weights = make_test_weights(&config, prefix);

        let model = InternViTModel::build(&weights, prefix, &config, -1).unwrap();

        let b = 1i64;
        // 42x42 with patch_size=14 gives 3x3 = 9 patches
        let h = 42i64;
        let w = 42i64;
        let c = config.num_channels as i64;
        let pixel_values = MxArray::random_normal(&[b, h, w, c], 0.0, 1.0, None).unwrap();

        let output = model.forward(&pixel_values).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();

        let expected_patches = 9i64; // 3 * 3
        assert_eq!(
            shape,
            vec![b, expected_patches, config.hidden_size as i64],
            "Interpolated position embedding should produce correct patch count"
        );
    }

    #[test]
    fn test_attention_values_not_zero() {
        let config = small_config();
        let prefix = "vision_model";
        let weights = make_test_weights(&config, prefix);

        let layer_prefix = format!("{prefix}.layers.0");
        let attn = InternVisionAttention::build(&weights, &layer_prefix, &config).unwrap();

        let d = config.hidden_size as i64;
        let input = MxArray::random_normal(&[1, 5, d], 0.0, 1.0, None).unwrap();

        let output = attn.forward(&input).unwrap();
        output.eval();

        let abs_sum = output.abs().unwrap().sum(None, None).unwrap();
        abs_sum.eval();
        let sum_val: Vec<f32> = abs_sum.to_float32().unwrap().to_vec();
        assert!(sum_val[0] > 0.0, "Attention output should not be all zeros");
    }

    #[test]
    fn test_batch_dimension() {
        // Verify batch>1 works correctly
        let config = small_config();
        let prefix = "vision_model";
        let weights = make_test_weights(&config, prefix);

        let model = InternViTModel::build(&weights, prefix, &config, -1).unwrap();

        let b = 2i64;
        let h = config.image_size as i64;
        let w = config.image_size as i64;
        let c = config.num_channels as i64;
        let pixel_values = MxArray::random_normal(&[b, h, w, c], 0.0, 1.0, None).unwrap();

        let output = model.forward(&pixel_values).unwrap();
        output.eval();
        let shape: Vec<i64> = output.shape().unwrap().to_vec();

        let grid = config.image_size / config.patch_size;
        let expected_patches = (grid * grid) as i64;
        assert_eq!(shape, vec![b, expected_patches, config.hidden_size as i64]);
    }
}
