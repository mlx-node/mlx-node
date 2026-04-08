use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray, scaled_dot_product_attention};
use crate::nn::{Activations, Linear, RMSNorm};

use super::clippable_linear::ClippableLinear;
use super::vision_config::Gemma4VisionConfig;
use super::vision_rope::apply_multidimensional_rope;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Slice along a single axis: `x[..., start:stop, ...]`.
fn slice_axis(x: &MxArray, axis: usize, start: i64, stop: i64) -> Result<MxArray> {
    let handle = unsafe { mlx_sys::mlx_array_slice_axis(x.as_raw_ptr(), axis, start, stop) };
    MxArray::from_handle(handle, "slice_axis")
}

/// One-hot encoding: `(expand_dims(indices, -1) == arange(num_classes)).astype(f32)`
fn one_hot(indices: &MxArray, num_classes: i32) -> Result<MxArray> {
    let classes = MxArray::arange(0.0, num_classes as f64, None, Some(DType::Int32))?;
    let expanded = indices.expand_dims(-1)?;
    expanded.equal(&classes)?.astype(DType::Float32)
}

// ---------------------------------------------------------------------------
// VisionRMSNorm — learned scale, explicit f32 computation
// ---------------------------------------------------------------------------

/// RMSNorm with learned scale weight. Computes entirely in f32, casts back to
/// the input dtype.  Used for per-head Q/K norms in vision attention.
pub struct VisionRMSNorm {
    pub weight: MxArray,
    pub eps: f64,
}

impl VisionRMSNorm {
    pub fn new(dim: i32, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: MxArray::ones(&[dim as i64], Some(DType::Float32))?,
            eps,
        })
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let in_dtype = x.dtype()?;
        let x_f32 = x.astype(DType::Float32)?;
        let var = x_f32
            .power(&MxArray::scalar_float(2.0)?)?
            .mean(Some(&[-1]), Some(true))?;
        let eps = MxArray::scalar_float(self.eps)?;
        // rsqrt(var + eps) = (var + eps)^(-0.5)
        let inv_rms = var.add(&eps)?.power(&MxArray::scalar_float(-0.5)?)?;
        let normed = x_f32.mul(&inv_rms)?;
        let result = normed.mul(&self.weight.astype(DType::Float32)?)?;
        result.astype(in_dtype)
    }
}

// ---------------------------------------------------------------------------
// VisionRMSNormNoScale — parameter-free, explicit f32 computation
// ---------------------------------------------------------------------------

/// Parameter-free RMSNorm (no learnable weight). Computes in f32.
/// Used for V-norm in vision attention and pre-projection norm in the
/// multimodal embedder.
pub struct VisionRMSNormNoScale {
    pub eps: f64,
}

impl Default for VisionRMSNormNoScale {
    fn default() -> Self {
        Self { eps: 1e-6 }
    }
}

impl VisionRMSNormNoScale {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_eps(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let in_dtype = x.dtype()?;
        let x_f32 = x.astype(DType::Float32)?;
        let var = x_f32
            .power(&MxArray::scalar_float(2.0)?)?
            .mean(Some(&[-1]), Some(true))?;
        let eps = MxArray::scalar_float(self.eps)?;
        // rsqrt(var + eps) = (var + eps)^(-0.5)
        let inv_rms = var.add(&eps)?.power(&MxArray::scalar_float(-0.5)?)?;
        let normed = x_f32.mul(&inv_rms)?;
        normed.astype(in_dtype)
    }
}

// ---------------------------------------------------------------------------
// VisionAttention
// ---------------------------------------------------------------------------

/// Vision self-attention with separate Q/K/V ClippableLinear projections,
/// per-head RMSNorm on Q/K, RMSNormNoScale on V, 2D RoPE, and bidirectional
/// SDPA with scale=1.0.
pub struct VisionAttention {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub hidden_size: i32,
    pub rope_base_frequency: f64,
    pub q_proj: ClippableLinear,
    pub k_proj: ClippableLinear,
    pub v_proj: ClippableLinear,
    pub o_proj: ClippableLinear,
    pub q_norm: VisionRMSNorm,
    pub k_norm: VisionRMSNorm,
    pub v_norm: VisionRMSNormNoScale,
}

impl VisionAttention {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self> {
        let clip = config.use_clipped_linears;
        let h = config.hidden_size;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;

        let make = |inp: i32, out: i32| -> Result<ClippableLinear> {
            let lin = Linear::new(inp as u32, out as u32, Some(false))?;
            Ok(ClippableLinear::new(lin, clip))
        };

        Ok(Self {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            hidden_size: h,
            rope_base_frequency: config.rope_theta,
            q_proj: make(h, nh * hd)?,
            k_proj: make(h, nkv * hd)?,
            v_proj: make(h, nkv * hd)?,
            o_proj: make(nh * hd, h)?,
            q_norm: VisionRMSNorm::new(hd, config.rms_norm_eps)?,
            k_norm: VisionRMSNorm::new(hd, config.rms_norm_eps)?,
            v_norm: VisionRMSNormNoScale::new(),
        })
    }

    pub fn forward(
        &self,
        x: &MxArray,
        positions: &MxArray,
        mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        let b = x.shape_at(0)?;
        let l = x.shape_at(1)?;

        let q = self.q_proj.forward(x)?.reshape(&[
            b,
            l,
            self.num_heads as i64,
            self.head_dim as i64,
        ])?;
        let k = self.k_proj.forward(x)?.reshape(&[
            b,
            l,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let v = self.v_proj.forward(x)?.reshape(&[
            b,
            l,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_norm.forward(&v)?;

        // Apply 2D RoPE
        let q = apply_multidimensional_rope(&q, positions, self.rope_base_frequency)?;
        let k = apply_multidimensional_rope(&k, positions, self.rope_base_frequency)?;

        // Transpose to [B, H, L, D] for SDPA
        let q = q.transpose(Some(&[0, 2, 1, 3]))?;
        let k = k.transpose(Some(&[0, 2, 1, 3]))?;
        let v = v.transpose(Some(&[0, 2, 1, 3]))?;

        let attn_output = scaled_dot_product_attention(&q, &k, &v, 1.0, mask)?;

        // [B, H, L, D] -> [B, L, H*D]
        let attn_output = attn_output
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&[b, l, -1])?;

        self.o_proj.forward(&attn_output)
    }
}

// ---------------------------------------------------------------------------
// VisionMLP — Gate-GLU with gelu_approx
// ---------------------------------------------------------------------------

/// Gate-GLU MLP: `down_proj(gelu(gate_proj(x)) * up_proj(x))`.
pub struct VisionMLP {
    pub gate_proj: ClippableLinear,
    pub up_proj: ClippableLinear,
    pub down_proj: ClippableLinear,
}

impl VisionMLP {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self> {
        let clip = config.use_clipped_linears;
        let h = config.hidden_size;
        let im = config.intermediate_size;

        let make = |inp: i32, out: i32| -> Result<ClippableLinear> {
            let lin = Linear::new(inp as u32, out as u32, Some(false))?;
            Ok(ClippableLinear::new(lin, clip))
        };

        Ok(Self {
            gate_proj: make(h, im)?,
            up_proj: make(h, im)?,
            down_proj: make(im, h)?,
        })
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let gate = Activations::gelu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.mul(&up)?)
    }
}

// ---------------------------------------------------------------------------
// VisionBlock — transformer block with 4 RMSNorms
// ---------------------------------------------------------------------------

/// Vision transformer block: input_layernorm → attention → post_attention_norm
/// → residual → pre_feedforward_norm → MLP → post_feedforward_norm → residual.
///
/// The 4 norms use the standard crate `RMSNorm` (fast kernel), while the
/// per-head norms inside attention use `VisionRMSNorm`/`VisionRMSNormNoScale`.
pub struct VisionBlock {
    pub self_attn: VisionAttention,
    pub mlp: VisionMLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub pre_feedforward_layernorm: RMSNorm,
    pub post_feedforward_layernorm: RMSNorm,
}

impl VisionBlock {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self> {
        let eps = Some(config.rms_norm_eps);
        let dims = config.hidden_size as u32;
        Ok(Self {
            self_attn: VisionAttention::new(config)?,
            mlp: VisionMLP::new(config)?,
            input_layernorm: RMSNorm::new(dims, eps)?,
            post_attention_layernorm: RMSNorm::new(dims, eps)?,
            pre_feedforward_layernorm: RMSNorm::new(dims, eps)?,
            post_feedforward_layernorm: RMSNorm::new(dims, eps)?,
        })
    }

    pub fn forward(
        &self,
        x: &MxArray,
        positions: &MxArray,
        mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, positions, mask)?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let h = x.add(&attn_out)?;

        let normed_h = self.pre_feedforward_layernorm.forward(&h)?;
        let ffw_out = self.mlp.forward(&normed_h)?;
        let ffw_out = self.post_feedforward_layernorm.forward(&ffw_out)?;
        h.add(&ffw_out)
    }
}

// ---------------------------------------------------------------------------
// VisionPatchEmbedder
// ---------------------------------------------------------------------------

/// Patch embedding: patchify pixels via linear projection + learned 2D
/// position embedding table.
pub struct VisionPatchEmbedder {
    pub hidden_size: i32,
    pub patch_size: i32,
    pub position_embedding_size: i32,
    pub input_proj: Linear,
    pub position_embedding_table: MxArray,
}

impl VisionPatchEmbedder {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self> {
        let p = config.patch_size;
        let h = config.hidden_size;
        let pes = config.position_embedding_size;
        Ok(Self {
            hidden_size: h,
            patch_size: p,
            position_embedding_size: pes,
            input_proj: Linear::new((3 * p * p) as u32, h as u32, Some(false))?,
            position_embedding_table: MxArray::ones(
                &[2, pes as i64, h as i64],
                Some(DType::Float32),
            )?,
        })
    }

    /// Compute additive position embeddings from patch grid coordinates.
    fn position_embeddings(
        &self,
        patch_positions: &MxArray,
        padding_positions: &MxArray,
    ) -> Result<MxArray> {
        // one_hot: [B, num_patches, 2] -> [B, num_patches, 2, pos_size]
        let oh = one_hot(patch_positions, self.position_embedding_size)?;
        // Transpose to [B, 2, num_patches, pos_size]
        let oh = oh
            .transpose(Some(&[0, 2, 1, 3]))?
            .astype(self.position_embedding_table.dtype()?)?;
        // Matmul: [B, 2, num_patches, pos_size] @ [2, pos_size, hidden] -> [B, 2, num_patches, hidden]
        let pe = oh.matmul(&self.position_embedding_table)?;
        // Sum over dimension axis (axis=1): [B, num_patches, hidden]
        let pe = pe.sum(Some(&[1]), Some(false))?;
        // Zero out padding positions
        let pad_expanded = padding_positions.expand_dims(-1)?;
        let zero = MxArray::scalar_float(0.0)?;
        pad_expanded.where_(&zero, &pe)
    }

    /// Convert raw pixel values to patch tokens.
    fn patchify(&self, pixel_values: &MxArray) -> Result<MxArray> {
        // pixel_values: [B, C, H, W] (channel-first)
        let b = pixel_values.shape_at(0)?;
        let c = pixel_values.shape_at(1)?;
        let h = pixel_values.shape_at(2)?;
        let w = pixel_values.shape_at(3)?;
        let p = self.patch_size as i64;
        let p_h = h / p;
        let p_w = w / p;

        // [B, C, pH, p, pW, p]
        let patches = pixel_values.reshape(&[b, c, p_h, p, p_w, p])?;
        // Transpose to [B, pH, pW, p, p, C]
        let patches = patches.transpose(Some(&[0, 2, 4, 3, 5, 1]))?;
        // Flatten to [B, pH*pW, C*p*p]
        let patches = patches.reshape(&[b, p_h * p_w, c * p * p])?;
        // Normalize: 2 * (patches - 0.5) = 2*patches - 1
        let half = MxArray::scalar_float(0.5)?;
        let two = MxArray::scalar_float(2.0)?;
        let patches = two.mul(&patches.sub(&half)?)?;
        // Project to hidden_size
        let proj_dtype = self.input_proj.get_weight().dtype()?;
        self.input_proj.forward(&patches.astype(proj_dtype)?)
    }

    pub fn forward(
        &self,
        pixel_values: &MxArray,
        patch_positions: &MxArray,
        padding_positions: &MxArray,
    ) -> Result<MxArray> {
        let hidden_states = self.patchify(pixel_values)?;
        let pe = self.position_embeddings(patch_positions, padding_positions)?;
        hidden_states.add(&pe)
    }
}

// ---------------------------------------------------------------------------
// VisionPooler
// ---------------------------------------------------------------------------

/// Spatial average pooling via kernel index computation.  Reduces the sequence
/// length from `max_patches` to `default_output_length` (280 by default).
pub struct VisionPooler {
    pub hidden_size: i32,
    pub default_output_length: i32,
    pub root_hidden_size: f64,
}

impl VisionPooler {
    pub fn new(config: &Gemma4VisionConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            default_output_length: config.default_output_length,
            root_hidden_size: (config.hidden_size as f64).sqrt(),
        }
    }

    /// Average pool by spatial kernel positions.
    /// Returns `(output, mask)` where mask is `[B, length]` with True=valid.
    fn avg_pool_by_positions(
        &self,
        x: &MxArray,
        patch_positions: &MxArray,
        length: i32,
    ) -> Result<(MxArray, MxArray)> {
        let input_seq_len = x.shape_at(1)?;
        let k = ((input_seq_len as f64 / length as f64).sqrt()) as i64;
        let k_squared = k * k;

        // Clamp positions to >= 0
        let zero_i32 = MxArray::scalar_int(0)?;
        let clamped = patch_positions.maximum(&zero_i32)?;

        // max_x = max(clamped[..., 0], axis=-1, keepdims=True) + 1
        let clamped_x = slice_axis(&clamped, (clamped.ndim()? - 1) as usize, 0, 1)?;
        // clamped_x: [B, L, 1] -> squeeze last dim for max
        let clamped_x_squeezed =
            clamped_x.reshape(&[clamped.shape_at(0)?, clamped.shape_at(1)?])?;
        let max_x = clamped_x_squeezed
            .max(Some(&[-1]), Some(true))?
            .add(&MxArray::scalar_int(1)?)?;

        // kernel_idxs = floor(clamped.astype(f32) / k).astype(int32)
        let k_f = MxArray::scalar_float(k as f64)?;
        let kernel_idxs = clamped
            .astype(DType::Float32)?
            .div(&k_f)?
            .floor()?
            .astype(DType::Int32)?;

        // kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        let last_axis = (kernel_idxs.ndim()? - 1) as usize;
        let ki_0 = slice_axis(&kernel_idxs, last_axis, 0, 1)?;
        let ki_1 = slice_axis(&kernel_idxs, last_axis, 1, 2)?;
        // Squeeze last dims: [B, L, 1] -> [B, L]
        let ki_0 = ki_0.reshape(&[kernel_idxs.shape_at(0)?, kernel_idxs.shape_at(1)?])?;
        let ki_1 = ki_1.reshape(&[kernel_idxs.shape_at(0)?, kernel_idxs.shape_at(1)?])?;
        let k_int = MxArray::scalar_int(k as i32)?;
        let combined = ki_0.add(&max_x.div(&k_int)?.mul(&ki_1)?)?;

        // weights = one_hot(combined, length) / k_squared
        let weights = one_hot(&combined, length)?;
        let k_sq_f = MxArray::scalar_float(k_squared as f64)?;
        let weights = weights.div(&k_sq_f)?;

        // einsum("bLl,bLd->bld") = weights^T @ x
        // weights: [B, L, l], x: [B, L, d]
        // weights.transpose(0,2,1) @ x = [B, l, L] @ [B, L, d] = [B, l, d]
        let wt = weights.transpose(Some(&[0, 2, 1]))?;
        let output = wt.matmul(x)?.astype(x.dtype()?)?;

        // mask: True where any weight is non-zero along L axis
        // sum(weights, axis=1) != 0 -> [B, l]
        let zero_f = MxArray::scalar_float(0.0)?;
        let w_sum = weights.sum(Some(&[1]), Some(false))?;
        let mask = w_sum.not_equal(&zero_f)?;

        Ok((output, mask))
    }

    pub fn forward(
        &self,
        hidden_states: &MxArray,
        patch_positions: &MxArray,
        padding_positions: &MxArray,
        output_length: Option<i32>,
    ) -> Result<(MxArray, MxArray)> {
        // Zero out padding tokens
        let pad_expanded = padding_positions.expand_dims(-1)?;
        let zero = MxArray::scalar_float(0.0)?;
        let hidden_states = pad_expanded.where_(&zero, hidden_states)?;

        let length = output_length.unwrap_or(self.default_output_length);

        // mask convention: always True=valid, False=padding
        let (hidden_states, valid_mask) = if hidden_states.shape_at(1)? == length as i64 {
            // No pooling needed — invert padding_positions (True=padding) to True=valid
            let valid = padding_positions.logical_not()?;
            (hidden_states, valid)
        } else {
            // avg_pool_by_positions already returns True=valid
            self.avg_pool_by_positions(&hidden_states, patch_positions, length)?
        };

        // Scale by sqrt(hidden_size) — applied unconditionally after pool/no-pool
        let scale = MxArray::scalar_float(self.root_hidden_size)?;
        let hidden_states = hidden_states.mul(&scale)?;
        Ok((hidden_states, valid_mask))
    }
}

// ---------------------------------------------------------------------------
// Gemma4VisionModel — top-level vision encoder
// ---------------------------------------------------------------------------

/// Top-level vision encoder: patch_embedder -> pad to max_patches ->
/// build bidirectional mask -> encoder layers -> pooler -> strip padding.
///
/// Weight key structure:
///   `vision_tower.patch_embedder.*`
///   `vision_tower.encoder.layers.*`
pub struct Gemma4VisionModel {
    pub patch_size: i32,
    pub pooling_kernel_size: i32,
    pub default_output_length: i32,
    pub max_patches: i32,
    pub standardize: bool,

    pub patch_embedder: VisionPatchEmbedder,
    pub encoder_layers: Vec<VisionBlock>,
    pub pooler: VisionPooler,

    /// Optional standardization parameters (loaded from checkpoint).
    pub std_bias: Option<MxArray>,
    pub std_scale: Option<MxArray>,
}

impl Gemma4VisionModel {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self> {
        let max_patches = config.default_output_length * config.pooling_kernel_size.pow(2);
        let mut encoder_layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for _ in 0..config.num_hidden_layers {
            encoder_layers.push(VisionBlock::new(config)?);
        }

        let (std_bias, std_scale) = if config.standardize {
            (
                Some(MxArray::zeros(
                    &[config.hidden_size as i64],
                    Some(DType::Float32),
                )?),
                Some(MxArray::ones(
                    &[config.hidden_size as i64],
                    Some(DType::Float32),
                )?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            patch_size: config.patch_size,
            pooling_kernel_size: config.pooling_kernel_size,
            default_output_length: config.default_output_length,
            max_patches,
            standardize: config.standardize,
            patch_embedder: VisionPatchEmbedder::new(config)?,
            encoder_layers,
            pooler: VisionPooler::new(config),
            std_bias,
            std_scale,
        })
    }

    /// Build patch positions `[B, max_patches, 2]` (int32) and padding mask
    /// `[B, max_patches]` (bool, True=padding) from the pixel spatial dims.
    fn patch_positions(&self, b: i64, h: i64, w: i64) -> Result<(MxArray, MxArray)> {
        let p = self.patch_size as i64;
        let p_h = h / p;
        let p_w = w / p;
        let num_patches = p_h * p_w;
        let num_padding = self.max_patches as i64 - num_patches;

        // Build position grid: for each patch (y, x) -> store (x, y) matching
        // Python's meshgrid(arange(pW), arange(pH), indexing="xy").
        // gx varies fastest along columns, gy along rows.
        let mut pos_data = Vec::with_capacity((num_patches * 2) as usize);
        for row in 0..p_h {
            for col in 0..p_w {
                pos_data.push(col as i32); // x
                pos_data.push(row as i32); // y
            }
        }
        let real_positions = MxArray::from_int32(&pos_data, &[1, num_patches, 2])?;
        // Tile to batch: [B, num_patches, 2]
        let real_positions = real_positions.tile(&[b as i32, 1, 1])?;

        let (patch_positions, padding_positions) = if num_padding > 0 {
            // Pad positions with -1
            let pad_pos = MxArray::full(&[b, num_padding, 2], Either::A(-1.0), Some(DType::Int32))?;
            let patch_positions = MxArray::concatenate(&real_positions, &pad_pos, 1)?;

            // Padding mask as int32: 0 = real, 1 = padding
            // (MLX treats 0 as false, nonzero as true in logical ops and where_)
            let real_mask = MxArray::zeros(&[b, num_patches], Some(DType::Int32))?;
            let pad_mask = MxArray::ones(&[b, num_padding], Some(DType::Int32))?;
            let padding_positions = MxArray::concatenate(&real_mask, &pad_mask, 1)?;

            (patch_positions, padding_positions)
        } else {
            let padding_positions =
                MxArray::zeros(&[b, self.max_patches as i64], Some(DType::Int32))?;
            (real_positions, padding_positions)
        };

        Ok((patch_positions, padding_positions))
    }

    /// Run the full vision encoder on pixel values `[B, C, H, W]`.
    /// Returns `[1, total_valid_tokens, hidden_size]`.
    pub fn forward(&self, pixel_values: &MxArray) -> Result<MxArray> {
        let b = pixel_values.shape_at(0)?;
        let h = pixel_values.shape_at(2)?;
        let w = pixel_values.shape_at(3)?;
        let num_real = (h / self.patch_size as i64) * (w / self.patch_size as i64);
        let num_padding = self.max_patches as i64 - num_real;

        let (patch_positions, padding_positions) = self.patch_positions(b, h, w)?;

        // Embed real patches only (slice positions to [:, :num_real])
        let pp_real = slice_axis(&patch_positions, 1, 0, num_real)?;
        let pad_real = slice_axis(&padding_positions, 1, 0, num_real)?;

        let mut inputs_embeds = self
            .patch_embedder
            .forward(pixel_values, &pp_real, &pad_real)?;

        // Pad embeddings to max_patches
        if num_padding > 0 {
            let embed_dim = inputs_embeds.shape_at(2)?;
            let embed_dtype = inputs_embeds.dtype()?;
            let pad_embeds = MxArray::zeros(&[b, num_padding, embed_dim], Some(embed_dtype))?;
            inputs_embeds = MxArray::concatenate(&inputs_embeds, &pad_embeds, 1)?;
        }

        // Build bidirectional attention mask [B, 1, L, L]
        let valid_mask = padding_positions.logical_not()?; // True = valid
        let vm1 = valid_mask.expand_dims(1)?; // [B, 1, L]
        let vm2 = valid_mask.expand_dims(2)?; // [B, L, 1]
        let attn_mask_bool = vm1.mul(&vm2)?; // broadcast -> [B, L, L]

        let embed_dtype = inputs_embeds.dtype()?;
        let zero_typed = MxArray::zeros(&[], Some(embed_dtype))?;
        let neg_inf = MxArray::scalar_float(f64::NEG_INFINITY)?.astype(embed_dtype)?;
        let attn_mask = attn_mask_bool.where_(&zero_typed, &neg_inf)?;
        let attn_mask = attn_mask.expand_dims(1)?; // [B, 1, L, L]

        // Run encoder layers
        let mut hidden_states = inputs_embeds;
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states, &patch_positions, Some(&attn_mask))?;
        }

        // Pool: reduce from max_patches to default_output_length
        let (pooled, pool_mask) =
            self.pooler
                .forward(&hidden_states, &patch_positions, &padding_positions, None)?;

        // pool_mask is always True=valid (normalized in VisionPooler::forward)
        let seq_len = pooled.shape_at(1)?;
        let valid_mask = pool_mask;

        // Strip padding: for each batch item, count valid tokens and slice.
        // Pooling produces contiguous valid tokens followed by padding.
        let mut all_real: Vec<MxArray> = Vec::with_capacity(b as usize);
        for i in 0..b {
            let vm_i = slice_axis(&valid_mask, 0, i, i + 1)?;
            let vm_i = vm_i.reshape(&[seq_len])?;
            // Count valid tokens
            let n_valid_arr = vm_i.astype(DType::Int32)?.sum(None, None)?;
            n_valid_arr.eval();
            let n_valid = n_valid_arr.item_at_int32(0)? as i64;
            // Slice pooled[i, :n_valid]
            let row = slice_axis(&pooled, 0, i, i + 1)?;
            let row = slice_axis(&row, 1, 0, n_valid)?;
            // Squeeze batch dim: [1, n_valid, dim] -> [n_valid, dim]
            let dim = pooled.shape_at(2)?;
            all_real.push(row.reshape(&[n_valid, dim])?);
        }

        // Concatenate along token dimension and add batch dim
        let refs: Vec<&MxArray> = all_real.iter().collect();
        let mut result = MxArray::concatenate_many(refs, Some(0))?;
        result = result.expand_dims(0)?; // [1, total_valid, dim]

        // Optional standardization
        if self.standardize
            && let (Some(bias), Some(scale)) = (&self.std_bias, &self.std_scale)
        {
            result = result.sub(bias)?.mul(scale)?;
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Gemma4MultimodalEmbedder
// ---------------------------------------------------------------------------

/// Projects vision tokens into the text model's embedding space.
/// Applies RMSNormNoScale (via `mx.fast.rms_norm(x, None, eps)`) then a Linear
/// projection from vision_dim to text_dim.
pub struct Gemma4MultimodalEmbedder {
    pub embedding_pre_projection_norm: VisionRMSNormNoScale,
    pub embedding_projection: Linear,
}

impl Gemma4MultimodalEmbedder {
    pub fn new(vision_dim: i32, text_dim: i32, eps: f64) -> Result<Self> {
        Ok(Self {
            embedding_pre_projection_norm: VisionRMSNormNoScale::with_eps(eps),
            embedding_projection: Linear::new(vision_dim as u32, text_dim as u32, Some(false))?,
        })
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let normed = self.embedding_pre_projection_norm.forward(x)?;
        self.embedding_projection.forward(&normed)
    }
}
