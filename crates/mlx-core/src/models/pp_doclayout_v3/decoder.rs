//! PP-DocLayoutV3 Transformer Decoder
//!
//! Implements the decoder for PP-DocLayoutV3 object detection, including:
//! - **SelfAttention**: Multi-head attention with positional embeddings on Q/K (not V)
//! - **MultiscaleDeformableAttention**: Deformable attention sampling features at learned offsets
//! - **DecoderLayer**: self-attn → cross-attn (deformable) → FFN with residual connections
//! - **Decoder**: Stack of 6 decoder layers with iterative bbox refinement
//!
//! All operations use float32 for compute. Shapes follow [batch, seq, dim] convention.
//!
//! Reference: modeling_pp_doclayout_v3.py
//! (PPDocLayoutV3MultiheadAttention, PPDocLayoutV3MultiscaleDeformableAttention,
//!  PPDocLayoutV3DecoderLayer, PPDocLayoutV3Decoder)

use crate::array::MxArray;
use crate::nn::activations::Activations;
use crate::nn::linear::Linear;
use crate::nn::normalization::LayerNorm;
use napi::Either;
use napi::bindgen_prelude::*;

use super::heads::{GlobalPointer, MLPPredictionHead};

// ============================================================================
// Utility: inverse_sigmoid
// ============================================================================

/// Compute the inverse sigmoid (logit) function.
///
/// `inverse_sigmoid(x) = log(x / (1 - x))`
///
/// Clamps input to [eps, 1-eps] for numerical stability.
pub fn inverse_sigmoid(x: &MxArray, eps: f64) -> Result<MxArray> {
    // x = clamp(x, eps, 1 - eps)
    let x = x.clip(Some(eps), Some(1.0 - eps))?;
    // log(x / (1 - x))
    let one_minus_x = x.sub_scalar(1.0)?.negative()?; // 1 - x
    let ratio = x.div(&one_minus_x)?;
    ratio.log()
}

// ============================================================================
// Self-Attention (Multi-Head Attention with Position Embeddings)
// ============================================================================

/// Standard multi-head attention used in the decoder's self-attention.
///
/// Key behavior: Position embeddings are added to Q and K, but NOT to V.
///
/// Architecture:
/// 1. hidden_states_with_pos = hidden_states + position_embeddings
/// 2. Q = q_proj(hidden_states_with_pos) * scale
/// 3. K = k_proj(hidden_states_with_pos)
/// 4. V = v_proj(hidden_states)  (original, without position!)
/// 5. attn = softmax(Q @ K^T) @ V
/// 6. output = out_proj(attn)
///
/// Corresponds to PPDocLayoutV3MultiheadAttention in the reference.
pub struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: i32,
    head_dim: i32,
    scaling: f64,
}

impl SelfAttention {
    /// Create a new SelfAttention layer.
    ///
    /// # Arguments
    /// * `embed_dim` - Model dimension (d_model, default 256)
    /// * `num_heads` - Number of attention heads (default 8)
    pub fn new(embed_dim: u32, num_heads: i32) -> Result<Self> {
        let head_dim = embed_dim as i32 / num_heads;
        if head_dim * num_heads != embed_dim as i32 {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "embed_dim ({}) must be divisible by num_heads ({})",
                    embed_dim, num_heads
                ),
            ));
        }
        let scaling = (head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj: Linear::new(embed_dim, embed_dim, Some(true))?,
            k_proj: Linear::new(embed_dim, embed_dim, Some(true))?,
            v_proj: Linear::new(embed_dim, embed_dim, Some(true))?,
            out_proj: Linear::new(embed_dim, embed_dim, Some(true))?,
            num_heads,
            head_dim,
            scaling,
        })
    }

    /// Create from pre-loaded weights with explicit embed_dim.
    pub fn from_weights_with_dim(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        embed_dim: i32,
        num_heads: i32,
    ) -> Self {
        let head_dim = embed_dim / num_heads;
        let scaling = (head_dim as f64).powf(-0.5);
        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scaling,
        }
    }

    /// Forward pass for self-attention.
    ///
    /// # Arguments
    /// * `hidden_states` - [batch, target_len, embed_dim]
    /// * `position_embeddings` - Optional [batch, target_len, embed_dim], added to Q and K
    /// * `attention_mask` - Optional [target_len, target_len] attention mask
    ///
    /// # Returns
    /// Output tensor [batch, target_len, embed_dim]
    pub fn forward(
        &self,
        hidden_states: &MxArray,
        position_embeddings: Option<&MxArray>,
        attention_mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        let shape = hidden_states.shape()?;
        let shape: Vec<i64> = shape.as_ref().to_vec();
        let batch_size = shape[0];
        let target_len = shape[1];
        let embed_dim = shape[2];

        // Add position embeddings to Q/K input (but NOT to V input)
        let qk_input = if let Some(pos_emb) = position_embeddings {
            hidden_states.add(pos_emb)?
        } else {
            hidden_states.clone()
        };

        // Project Q, K, V
        // Q = q_proj(hidden_states + pos) * scale
        let query_states = self.q_proj.forward(&qk_input)?;
        let query_states = query_states.mul_scalar(self.scaling)?;

        // K = k_proj(hidden_states + pos)
        let key_states = self.k_proj.forward(&qk_input)?;

        // V = v_proj(hidden_states)  -- NO position embeddings on V!
        let value_states = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq, num_heads, head_dim] then transpose to [batch, num_heads, seq, head_dim]
        let nh = self.num_heads as i64;
        let hd = self.head_dim as i64;

        let query_states = query_states.reshape(&[batch_size, target_len, nh, hd])?;
        let query_states = query_states.transpose(Some(&[0, 2, 1, 3]))?;
        // Flatten to [batch * num_heads, target_len, head_dim]
        let query_states = query_states.reshape(&[batch_size * nh, target_len, hd])?;

        let key_states = key_states.reshape(&[batch_size, target_len, nh, hd])?;
        let key_states = key_states.transpose(Some(&[0, 2, 1, 3]))?;
        let key_states = key_states.reshape(&[batch_size * nh, target_len, hd])?;

        let value_states = value_states.reshape(&[batch_size, target_len, nh, hd])?;
        let value_states = value_states.transpose(Some(&[0, 2, 1, 3]))?;
        let value_states = value_states.reshape(&[batch_size * nh, target_len, hd])?;

        // Compute attention: Q @ K^T → [batch*num_heads, target_len, target_len]
        let key_states_t = key_states.transpose(Some(&[0, 2, 1]))?;
        let mut attn_weights = query_states.matmul(&key_states_t)?;

        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            // mask shape: [target_len, target_len] → expand to [batch, 1, target_len, target_len]
            let mask_expanded = mask
                .expand_dims(0)?
                .expand_dims(0)?
                .broadcast_to(&[batch_size, 1, target_len, target_len])?;

            // Reshape attn_weights to [batch, num_heads, target_len, target_len]
            attn_weights = attn_weights.reshape(&[batch_size, nh, target_len, target_len])?;

            // For the mask: where mask is True (non-zero), fill with -inf
            // The PyTorch code checks if mask is bool and fills True positions with -inf.
            // Here the mask values are already 0.0 (no mask) or -inf (mask).
            // So we add the mask directly.
            attn_weights = attn_weights.add(&mask_expanded)?;

            // Flatten back
            attn_weights = attn_weights.reshape(&[batch_size * nh, target_len, target_len])?;
        }

        // Softmax
        attn_weights = Activations::softmax(&attn_weights, Some(-1))?;

        // attn_output = attn_weights @ V → [batch*num_heads, target_len, head_dim]
        let attn_output = attn_weights.matmul(&value_states)?;

        // Reshape back: [batch, num_heads, target_len, head_dim] → [batch, target_len, embed_dim]
        let attn_output = attn_output.reshape(&[batch_size, nh, target_len, hd])?;
        let attn_output = attn_output.transpose(Some(&[0, 2, 1, 3]))?;
        let attn_output = attn_output.reshape(&[batch_size, target_len, embed_dim])?;

        // Output projection
        self.out_proj.forward(&attn_output)
    }
}

// ============================================================================
// Multi-Scale Deformable Attention
// ============================================================================

/// Multi-scale deformable attention mechanism.
///
/// For each query, instead of attending to all spatial positions (like standard attention),
/// this module:
/// 1. Predicts a small set of sampling offsets relative to reference points
/// 2. Samples features from the encoder feature maps at those locations
/// 3. Weights the sampled features with learned attention weights
///
/// This is much more efficient than full attention over all spatial positions,
/// and allows the model to focus on relevant regions.
///
/// Parameters:
/// - `num_heads=8, num_levels=3, num_points=4, d_model=256`
/// - Each query produces `num_heads * num_levels * num_points = 96` sampling locations
/// - Each location is a 2D offset relative to the reference point
///
/// Corresponds to PPDocLayoutV3MultiscaleDeformableAttention in the reference.
pub struct MultiscaleDeformableAttention {
    /// Project value features: Linear(d_model, d_model)
    value_proj: Linear,
    /// Predict sampling offsets: Linear(d_model, num_heads * num_levels * num_points * 2)
    sampling_offsets: Linear,
    /// Predict attention weights: Linear(d_model, num_heads * num_levels * num_points)
    attention_weights: Linear,
    /// Output projection: Linear(d_model, d_model)
    output_proj: Linear,
    /// Model dimension
    d_model: i32,
    /// Number of attention heads
    n_heads: i32,
    /// Number of feature levels
    n_levels: i32,
    /// Number of sampling points per head per level
    n_points: i32,
}

impl MultiscaleDeformableAttention {
    /// Create a new MultiscaleDeformableAttention.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (default 256)
    /// * `num_heads` - Number of attention heads (default 8)
    /// * `n_levels` - Number of feature levels (default 3)
    /// * `n_points` - Number of sampling points per head per level (default 4)
    pub fn new(d_model: i32, num_heads: i32, n_levels: i32, n_points: i32) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("d_model ({d_model}) must be divisible by num_heads ({num_heads})"),
            ));
        }
        let total_points = (num_heads * n_levels * n_points * 2) as u32;
        let total_weights = (num_heads * n_levels * n_points) as u32;
        let dm = d_model as u32;

        Ok(Self {
            value_proj: Linear::new(dm, dm, Some(true))?,
            sampling_offsets: Linear::new(dm, total_points, Some(true))?,
            attention_weights: Linear::new(dm, total_weights, Some(true))?,
            output_proj: Linear::new(dm, dm, Some(true))?,
            d_model,
            n_heads: num_heads,
            n_levels,
            n_points,
        })
    }

    /// Create from pre-loaded weights.
    pub fn from_weights(
        value_proj: Linear,
        sampling_offsets: Linear,
        attention_weights: Linear,
        output_proj: Linear,
        d_model: i32,
        n_heads: i32,
        n_levels: i32,
        n_points: i32,
    ) -> Self {
        Self {
            value_proj,
            sampling_offsets,
            attention_weights,
            output_proj,
            d_model,
            n_heads,
            n_levels,
            n_points,
        }
    }

    /// Forward pass for multi-scale deformable attention.
    ///
    /// # Arguments
    /// * `hidden_states` - Query features [batch, num_queries, d_model]
    /// * `encoder_hidden_states` - Flattened encoder features [batch, total_seq_len, d_model]
    /// * `position_embeddings` - Optional [batch, num_queries, d_model], added to hidden_states
    /// * `reference_points` - [batch, num_queries, 1, 2] or [batch, num_queries, num_levels, 2]
    ///   Normalized reference points in [0, 1]
    /// * `spatial_shapes` - [num_levels, 2] (height, width) for each feature level
    /// * `spatial_shapes_list` - Vec of (height, width) tuples
    /// * `level_start_index` - [num_levels] cumulative start indices
    ///
    /// # Returns
    /// Output tensor [batch, num_queries, d_model]
    pub fn forward(
        &self,
        hidden_states: &MxArray,
        encoder_hidden_states: &MxArray,
        position_embeddings: Option<&MxArray>,
        reference_points: &MxArray,
        spatial_shapes: &MxArray,
        spatial_shapes_list: &[(i64, i64)],
        _level_start_index: &MxArray,
    ) -> Result<MxArray> {
        // Add position embeddings to hidden states
        let query_input = if let Some(pos_emb) = position_embeddings {
            hidden_states.add(pos_emb)?
        } else {
            hidden_states.clone()
        };

        let q_shape = query_input.shape()?;
        let q_shape: Vec<i64> = q_shape.as_ref().to_vec();
        let batch_size = q_shape[0];
        let num_queries = q_shape[1];

        let hidden_dim = self.d_model as i64 / self.n_heads as i64;
        let nh = self.n_heads as i64;
        let nl = self.n_levels as i64;
        let np = self.n_points as i64;

        // Project values: [B, total_seq, d_model] → [B, total_seq, n_heads, hidden_dim]
        let value = self.value_proj.forward(encoder_hidden_states)?;
        let value = value.reshape(&[batch_size, -1, nh, hidden_dim])?;

        // Predict sampling offsets: [B, num_queries, n_heads * n_levels * n_points * 2]
        let offsets = self.sampling_offsets.forward(&query_input)?;
        let offsets = offsets.reshape(&[batch_size, num_queries, nh, nl, np, 2])?;

        // Predict attention weights: [B, num_queries, n_heads * n_levels * n_points]
        let attn_weights = self.attention_weights.forward(&query_input)?;
        let attn_weights = attn_weights.reshape(&[batch_size, num_queries, nh, nl * np])?;
        let attn_weights = Activations::softmax(&attn_weights, Some(-1))?;
        let attn_weights = attn_weights.reshape(&[batch_size, num_queries, nh, nl, np])?;

        // Compute sampling locations from reference points + offsets
        // reference_points: [B, num_queries, 1, 2] (for 2D case)
        // offset_normalizer: [width, height] per level (note: x=width, y=height)
        // sampling_locations = reference_points[:, :, None, :, None, :] + offsets / normalizer
        let ref_shape = reference_points.shape()?;
        let ref_shape: Vec<i64> = ref_shape.as_ref().to_vec();
        let num_coords = ref_shape[ref_shape.len() - 1];

        let sampling_locations = if num_coords == 2 {
            // offset_normalizer: stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            // This gives [width, height] for each level → [num_levels, 2]
            // We need to normalize offsets by the spatial dimensions
            //
            // spatial_shapes: [num_levels, 2] where [:,0]=height, [:,1]=width
            // offset_normalizer: [num_levels, 2] where [:,0]=width, [:,1]=height
            let shapes_h = spatial_shapes.slice(&[0, 0], &[nl, 1])?; // [nl, 1] heights
            let shapes_w = spatial_shapes.slice(&[0, 1], &[nl, 2])?; // [nl, 1] widths
            let offset_normalizer = MxArray::concatenate(&shapes_w, &shapes_h, 1)?; // [nl, 2] = [w, h]

            // Reshape normalizer for broadcasting: [1, 1, 1, nl, 1, 2]
            let normalizer = offset_normalizer.reshape(&[1, 1, 1, nl, 1, 2])?;

            // Normalize offsets
            let normalized_offsets = offsets.div(&normalizer)?;

            // reference_points has shape [B, num_queries, 1, 2]
            // Expand to [B, num_queries, 1, 1, 1, 2] for broadcasting
            // with normalized_offsets [B, num_queries, n_heads, n_levels, n_points, 2]
            let ref_expanded = reference_points.reshape(&[batch_size, num_queries, 1, 1, 1, 2])?;
            let ref_broadcast =
                ref_expanded.broadcast_to(&[batch_size, num_queries, nh, nl, np, 2])?;

            ref_broadcast.add(&normalized_offsets)?
        } else if num_coords == 4 {
            // 4-coordinate reference points: center_x, center_y, width, height
            // sampling_locations = ref[:,:,None,:,None,:2] +
            //   offsets / n_points * ref[:,:,None,:,None,2:] * 0.5
            let ref_xy =
                reference_points.slice(&[0, 0, 0, 0], &[batch_size, num_queries, nl, 2])?;
            let ref_wh =
                reference_points.slice(&[0, 0, 0, 2], &[batch_size, num_queries, nl, 4])?;

            let ref_xy_expanded = ref_xy.reshape(&[batch_size, num_queries, 1, nl, 1, 2])?;
            let ref_wh_expanded = ref_wh.reshape(&[batch_size, num_queries, 1, nl, 1, 2])?;

            let ref_xy_broadcast =
                ref_xy_expanded.broadcast_to(&[batch_size, num_queries, nh, nl, np, 2])?;
            let ref_wh_broadcast =
                ref_wh_expanded.broadcast_to(&[batch_size, num_queries, nh, nl, np, 2])?;

            let scaled_offsets = offsets
                .div_scalar(self.n_points as f64)?
                .mul(&ref_wh_broadcast)?
                .mul_scalar(0.5)?;

            ref_xy_broadcast.add(&scaled_offsets)?
        } else {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "reference_points last dim must be 2 or 4, got {}",
                    num_coords
                ),
            ));
        };

        // Now perform the actual deformable attention using bilinear interpolation.
        //
        // For each level, we need to:
        // 1. Extract the value features for that level
        // 2. Convert normalized sampling locations to absolute pixel coordinates
        // 3. Perform bilinear interpolation to sample features
        // 4. Weight by attention weights and sum

        // Convert sampling locations from [0,1] to grid coordinates [-1, 1] for grid_sample equiv
        // Actually, we'll do bilinear interpolation directly in absolute coordinates.

        // Split value by levels using spatial_shapes_list
        let mut level_values: Vec<MxArray> = Vec::with_capacity(self.n_levels as usize);
        let mut offset = 0i64;
        for &(h, w) in spatial_shapes_list {
            let len = h * w;
            // value_level: [B, h*w, n_heads, hidden_dim]
            let value_level = value.slice(
                &[0, offset, 0, 0],
                &[batch_size, offset + len, nh, hidden_dim],
            )?;
            level_values.push(value_level);
            offset += len;
        }

        // For each level, sample features at the predicted locations using bilinear interpolation
        // sampling_locations: [B, num_queries, n_heads, n_levels, n_points, 2] in [0, 1]
        let mut sampled_values_per_level: Vec<MxArray> = Vec::with_capacity(self.n_levels as usize);

        for (level_id, &(h, w)) in spatial_shapes_list.iter().enumerate() {
            let lid = level_id as i64;

            // Get value features for this level: [B, h*w, n_heads, hidden_dim]
            // Reshape to [B, h, w, n_heads, hidden_dim]
            let value_l = level_values[level_id].reshape(&[batch_size, h, w, nh, hidden_dim])?;

            // Get sampling locations for this level: [B, num_queries, n_heads, n_points, 2]
            let sample_locs = sampling_locations.slice(
                &[0, 0, 0, lid, 0, 0],
                &[batch_size, num_queries, nh, lid + 1, np, 2],
            )?;
            let sample_locs = sample_locs.squeeze(Some(&[3]))?;
            // sample_locs: [B, num_queries, n_heads, n_points, 2] in [0, 1]

            // Convert to absolute coordinates using PyTorch grid_sample align_corners=False convention:
            // pixel_coord = sample * size - 0.5
            let loc_x = sample_locs
                .slice(&[0, 0, 0, 0, 0], &[batch_size, num_queries, nh, np, 1])?
                .squeeze(Some(&[4]))?
                .mul_scalar(w as f64)?
                .sub_scalar(0.5)?;
            let loc_y = sample_locs
                .slice(&[0, 0, 0, 0, 1], &[batch_size, num_queries, nh, np, 2])?
                .squeeze(Some(&[4]))?
                .mul_scalar(h as f64)?
                .sub_scalar(0.5)?;

            // Bilinear interpolation
            // Floor and ceil coordinates
            let loc_x_floor = loc_x.floor()?;
            let loc_y_floor = loc_y.floor()?;
            let loc_x_ceil = loc_x_floor.add_scalar(1.0)?;
            let loc_y_ceil = loc_y_floor.add_scalar(1.0)?;

            // Interpolation weights
            let wx = loc_x.sub(&loc_x_floor)?; // fractional part x
            let wy = loc_y.sub(&loc_y_floor)?; // fractional part y
            let one_minus_wx = wx.sub_scalar(1.0)?.negative()?;
            let one_minus_wy = wy.sub_scalar(1.0)?.negative()?;

            // Weights for 4 corners
            let w_tl = one_minus_wx.mul(&one_minus_wy)?; // (1-wx) * (1-wy)
            let w_tr = wx.mul(&one_minus_wy)?; // wx * (1-wy)
            let w_bl = one_minus_wx.mul(&wy)?; // (1-wx) * wy
            let w_br = wx.mul(&wy)?; // wx * wy

            // Build per-corner validity masks BEFORE clamping (for zero-padding behavior).
            // PyTorch's F.grid_sample with padding_mode="zeros" checks each corner
            // independently: out-of-bounds corners contribute 0, but in-bounds corners
            // still contribute their real values weighted by their bilinear weights.
            let zero = MxArray::from_float32(&[0.0], &[1])?;
            let max_x = MxArray::from_float32(&[(w - 1) as f32], &[1])?;
            let max_y = MxArray::from_float32(&[(h - 1) as f32], &[1])?;

            // Per-coordinate validity (unclamped)
            let floor_x_valid = loc_x_floor
                .greater_equal(&zero)?
                .logical_and(&loc_x_floor.less_equal(&max_x)?)?;
            let ceil_x_valid = loc_x_ceil
                .greater_equal(&zero)?
                .logical_and(&loc_x_ceil.less_equal(&max_x)?)?;
            let floor_y_valid = loc_y_floor
                .greater_equal(&zero)?
                .logical_and(&loc_y_floor.less_equal(&max_y)?)?;
            let ceil_y_valid = loc_y_ceil
                .greater_equal(&zero)?
                .logical_and(&loc_y_ceil.less_equal(&max_y)?)?;

            // 4 per-corner masks: [B, num_queries, n_heads, n_points] (bool)
            let mask_tl = floor_x_valid.logical_and(&floor_y_valid)?; // top-left (x0, y0)
            let mask_tr = ceil_x_valid.logical_and(&floor_y_valid)?; // top-right (x1, y0)
            let mask_bl = floor_x_valid.logical_and(&ceil_y_valid)?; // bottom-left (x0, y1)
            let mask_br = ceil_x_valid.logical_and(&ceil_y_valid)?; // bottom-right (x1, y1)

            // Clamp coordinates to valid range for safe array indexing
            let x0 = loc_x_floor.maximum(&zero)?.minimum(&max_x)?;
            let y0 = loc_y_floor.maximum(&zero)?.minimum(&max_y)?;
            let x1 = loc_x_ceil.maximum(&zero)?.minimum(&max_x)?;
            let y1 = loc_y_ceil.maximum(&zero)?.minimum(&max_y)?;

            // Convert to integer indices for gathering
            // Compute linear indices into the [h, w] feature map
            // index = y * w + x
            let w_scalar = MxArray::from_float32(&[w as f32], &[1])?;

            // Compute 4 linear indices for the 4 corners
            let idx_tl = y0.mul(&w_scalar)?.add(&x0)?; // y0 * w + x0
            let idx_tr = y0.mul(&w_scalar)?.add(&x1)?; // y0 * w + x1
            let idx_bl = y1.mul(&w_scalar)?.add(&x0)?; // y1 * w + x0
            let idx_br = y1.mul(&w_scalar)?.add(&x1)?; // y1 * w + x1

            // Reshape value_l from [B, h, w, n_heads, hidden_dim] to [B, h*w, n_heads, hidden_dim]
            let value_hw = value_l.reshape(&[batch_size, h * w, nh, hidden_dim])?;

            // Per-head gather approach:
            // For each (query, head, point), we sample from value_hw[:, :, h, :].
            // Transpose to [B, n_heads, h*w, hidden_dim] then flatten batch and heads
            // so we can gather per-head spatial features efficiently.
            let value_hw_t = value_hw.transpose(Some(&[0, 2, 1, 3]))?;
            // [B, n_heads, h*w, hidden_dim]

            // Reshape to [B * n_heads, h*w, hidden_dim]
            let value_bh = value_hw_t.reshape(&[batch_size * nh, h * w, hidden_dim])?;

            // Reshape indices to [B, n_heads, num_queries * n_points]
            // Original idx shape: [B, num_queries, n_heads, n_points]
            // First reshape: [B, num_queries, n_heads, n_points] → transpose → [B, n_heads, num_queries, n_points]
            // → reshape → [B * n_heads, num_queries * n_points]
            let qp = num_queries * np;

            let idx_tl_4d = idx_tl
                .reshape(&[batch_size, num_queries, nh, np])?
                .astype(crate::array::DType::Int32)?;
            let idx_tl_t = idx_tl_4d
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp])?;

            let idx_tr_4d = idx_tr
                .reshape(&[batch_size, num_queries, nh, np])?
                .astype(crate::array::DType::Int32)?;
            let idx_tr_t = idx_tr_4d
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp])?;

            let idx_bl_4d = idx_bl
                .reshape(&[batch_size, num_queries, nh, np])?
                .astype(crate::array::DType::Int32)?;
            let idx_bl_t = idx_bl_4d
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp])?;

            let idx_br_4d = idx_br
                .reshape(&[batch_size, num_queries, nh, np])?
                .astype(crate::array::DType::Int32)?;
            let idx_br_t = idx_br_4d
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp])?;

            // Expand indices for broadcasting with hidden_dim
            let idx_tl_bh =
                idx_tl_t
                    .expand_dims(2)?
                    .broadcast_to(&[batch_size * nh, qp, hidden_dim])?;
            let idx_tr_bh =
                idx_tr_t
                    .expand_dims(2)?
                    .broadcast_to(&[batch_size * nh, qp, hidden_dim])?;
            let idx_bl_bh =
                idx_bl_t
                    .expand_dims(2)?
                    .broadcast_to(&[batch_size * nh, qp, hidden_dim])?;
            let idx_br_bh =
                idx_br_t
                    .expand_dims(2)?
                    .broadcast_to(&[batch_size * nh, qp, hidden_dim])?;

            // Gather: [B*n_heads, num_queries*n_points, hidden_dim]
            let g_tl = value_bh.take_along_axis(&idx_tl_bh, 1)?;
            let g_tr = value_bh.take_along_axis(&idx_tr_bh, 1)?;
            let g_bl = value_bh.take_along_axis(&idx_bl_bh, 1)?;
            let g_br = value_bh.take_along_axis(&idx_br_bh, 1)?;

            // Apply bilinear weights
            // Reshape weights to [B*n_heads, num_queries*n_points, 1]
            let w_tl_t = w_tl
                .reshape(&[batch_size, num_queries, nh, np])?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp, 1])?;
            let w_tr_t = w_tr
                .reshape(&[batch_size, num_queries, nh, np])?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp, 1])?;
            let w_bl_t = w_bl
                .reshape(&[batch_size, num_queries, nh, np])?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp, 1])?;
            let w_br_t = w_br
                .reshape(&[batch_size, num_queries, nh, np])?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp, 1])?;

            // Reshape per-corner validity masks to [B*n_heads, num_queries*n_points, 1]
            // to broadcast with gathered values [B*n_heads, qp, hidden_dim].
            let mask_tl_t = mask_tl
                .astype(crate::array::DType::Float32)?
                .reshape(&[batch_size, num_queries, nh, np])?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp, 1])?;
            let mask_tr_t = mask_tr
                .astype(crate::array::DType::Float32)?
                .reshape(&[batch_size, num_queries, nh, np])?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp, 1])?;
            let mask_bl_t = mask_bl
                .astype(crate::array::DType::Float32)?
                .reshape(&[batch_size, num_queries, nh, np])?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp, 1])?;
            let mask_br_t = mask_br
                .astype(crate::array::DType::Float32)?
                .reshape(&[batch_size, num_queries, nh, np])?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[batch_size * nh, qp, 1])?;

            // Bilinear interpolation with per-corner zero-padding:
            // Each corner's contribution is masked independently. Out-of-bounds corners
            // contribute 0, while in-bounds corners contribute their real values.
            let interpolated = g_tl
                .mul(&w_tl_t)?
                .mul(&mask_tl_t)?
                .add(&g_tr.mul(&w_tr_t)?.mul(&mask_tr_t)?)?
                .add(&g_bl.mul(&w_bl_t)?.mul(&mask_bl_t)?)?
                .add(&g_br.mul(&w_br_t)?.mul(&mask_br_t)?)?;
            // interpolated: [B*n_heads, num_queries*n_points, hidden_dim]

            // Reshape to [B*n_heads, hidden_dim, num_queries, n_points]
            let interpolated = interpolated
                .reshape(&[batch_size * nh, num_queries, np, hidden_dim])?
                .transpose(Some(&[0, 3, 1, 2]))?;
            // [B*n_heads, hidden_dim, num_queries, n_points]

            sampled_values_per_level.push(interpolated);
        }

        // Stack sampled values across levels: [B*n_heads, hidden_dim, num_queries, n_levels, n_points]
        // Then flatten last two dims: [B*n_heads, hidden_dim, num_queries, n_levels * n_points]
        let sampled_refs: Vec<&MxArray> = sampled_values_per_level.iter().collect();
        let stacked = MxArray::stack(sampled_refs, Some(3))?;
        // stacked: [B*n_heads, hidden_dim, num_queries, n_levels, n_points]
        let stacked_flat = stacked.reshape(&[batch_size * nh, hidden_dim, num_queries, nl * np])?;

        // Apply attention weights
        // attn_weights: [B, num_queries, n_heads, n_levels, n_points]
        // Reshape to [B*n_heads, 1, num_queries, n_levels * n_points]
        let attn_w = attn_weights.transpose(Some(&[0, 2, 1, 3, 4]))?.reshape(&[
            batch_size * nh,
            1,
            num_queries,
            nl * np,
        ])?;

        // Weighted sum: element-wise multiply and sum over last dim
        // stacked_flat: [B*n_heads, hidden_dim, num_queries, n_levels*n_points]
        // attn_w:       [B*n_heads, 1,          num_queries, n_levels*n_points]
        let weighted = stacked_flat.mul(&attn_w)?;
        // Sum over the last dimension (sampling points): [B*n_heads, hidden_dim, num_queries]
        let output = weighted.sum(Some(&[3]), Some(false))?;

        // Reshape: [B*n_heads, hidden_dim, num_queries] → [B, n_heads*hidden_dim, num_queries]
        let output = output.reshape(&[batch_size, nh * hidden_dim, num_queries])?;

        // Transpose to [B, num_queries, d_model]
        let output = output.transpose(Some(&[0, 2, 1]))?;

        // Output projection
        self.output_proj.forward(&output)
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

/// A single decoder layer combining self-attention, cross-attention, and FFN.
///
/// Architecture:
/// 1. Self-attention with position embeddings (on Q/K only) + residual + LayerNorm
/// 2. Multi-scale deformable cross-attention + residual + LayerNorm
/// 3. FFN (fc1 → activation → fc2) + residual + LayerNorm
///
/// Corresponds to PPDocLayoutV3DecoderLayer in the reference.
pub struct DecoderLayer {
    /// Self-attention
    self_attn: SelfAttention,
    /// Self-attention layer norm
    self_attn_layer_norm: LayerNorm,
    /// Cross-attention (deformable)
    encoder_attn: MultiscaleDeformableAttention,
    /// Cross-attention layer norm
    encoder_attn_layer_norm: LayerNorm,
    /// First feed-forward layer: d_model → ffn_dim
    fc1: Linear,
    /// Second feed-forward layer: ffn_dim → d_model
    fc2: Linear,
    /// Final layer norm
    final_layer_norm: LayerNorm,
    /// Activation function name
    activation: String,
}

impl DecoderLayer {
    /// Create a new DecoderLayer.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (default 256)
    /// * `ffn_dim` - FFN intermediate dimension (default 1024)
    /// * `num_heads` - Number of attention heads (default 8)
    /// * `n_levels` - Number of feature levels (default 3)
    /// * `n_points` - Number of sampling points (default 4)
    /// * `layer_norm_eps` - LayerNorm epsilon (default 1e-5)
    /// * `activation` - Activation function name (default "relu")
    pub fn new(
        d_model: i32,
        ffn_dim: i32,
        num_heads: i32,
        n_levels: i32,
        n_points: i32,
        layer_norm_eps: f64,
        activation: &str,
    ) -> Result<Self> {
        let dm = d_model as u32;
        let ff = ffn_dim as u32;
        let eps = Some(layer_norm_eps);

        Ok(Self {
            self_attn: SelfAttention::new(dm, num_heads)?,
            self_attn_layer_norm: LayerNorm::new(dm, eps)?,
            encoder_attn: MultiscaleDeformableAttention::new(
                d_model, num_heads, n_levels, n_points,
            )?,
            encoder_attn_layer_norm: LayerNorm::new(dm, eps)?,
            fc1: Linear::new(dm, ff, Some(true))?,
            fc2: Linear::new(ff, dm, Some(true))?,
            final_layer_norm: LayerNorm::new(dm, eps)?,
            activation: activation.to_string(),
        })
    }

    /// Create from pre-loaded components.
    pub fn from_components(
        self_attn: SelfAttention,
        self_attn_layer_norm: LayerNorm,
        encoder_attn: MultiscaleDeformableAttention,
        encoder_attn_layer_norm: LayerNorm,
        fc1: Linear,
        fc2: Linear,
        final_layer_norm: LayerNorm,
        activation: &str,
    ) -> Self {
        Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation: activation.to_string(),
        }
    }

    /// Forward pass through one decoder layer.
    ///
    /// # Arguments
    /// * `hidden_states` - [batch, num_queries, d_model]
    /// * `position_embeddings` - [batch, num_queries, d_model]
    /// * `encoder_hidden_states` - [batch, total_seq_len, d_model]
    /// * `reference_points` - [batch, num_queries, 1, 2] or [batch, num_queries, num_levels, 2]
    /// * `spatial_shapes` - [num_levels, 2]
    /// * `spatial_shapes_list` - Vec of (height, width)
    /// * `level_start_index` - [num_levels]
    /// * `encoder_attention_mask` - Optional attention mask for self-attention
    ///
    /// # Returns
    /// Output tensor [batch, num_queries, d_model]
    pub fn forward(
        &self,
        hidden_states: &MxArray,
        position_embeddings: &MxArray,
        encoder_hidden_states: &MxArray,
        reference_points: &MxArray,
        spatial_shapes: &MxArray,
        spatial_shapes_list: &[(i64, i64)],
        level_start_index: &MxArray,
        encoder_attention_mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        // 1. Self-Attention with position embeddings
        let residual = hidden_states.clone();
        let attn_output = self.self_attn.forward(
            hidden_states,
            Some(position_embeddings),
            encoder_attention_mask,
        )?;
        let hidden_states = residual.add(&attn_output)?;
        let hidden_states = self.self_attn_layer_norm.forward(&hidden_states)?;

        // 2. Cross-Attention (deformable)
        let second_residual = hidden_states.clone();
        let cross_output = self.encoder_attn.forward(
            &hidden_states,
            encoder_hidden_states,
            Some(position_embeddings),
            reference_points,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
        )?;
        let hidden_states = second_residual.add(&cross_output)?;
        let hidden_states = self.encoder_attn_layer_norm.forward(&hidden_states)?;

        // 3. FFN
        let residual = hidden_states.clone();
        let fc1_out = self.fc1.forward(&hidden_states)?;
        let activated = super::apply_activation(&fc1_out, &self.activation)?;
        let fc2_out = self.fc2.forward(&activated)?;
        let hidden_states = residual.add(&fc2_out)?;
        let hidden_states = self.final_layer_norm.forward(&hidden_states)?;

        Ok(hidden_states)
    }
}

// ============================================================================
// Decoder Output
// ============================================================================

/// Output of the PP-DocLayoutV3 decoder.
pub struct DecoderOutput {
    /// Last hidden state from the final decoder layer: [batch, num_queries, d_model]
    pub last_hidden_state: MxArray,
    /// Stacked intermediate hidden states: [batch, num_layers, num_queries, d_model]
    pub intermediate_hidden_states: MxArray,
    /// Stacked intermediate class logits: [batch, num_layers, num_queries, num_labels]
    pub intermediate_logits: MxArray,
    /// Stacked intermediate reference points: [batch, num_layers, num_queries, 4]
    pub intermediate_reference_points: MxArray,
    /// Stacked reading order logits: [batch, num_layers, num_queries, num_queries]
    pub decoder_out_order_logits: MxArray,
    /// Stacked mask predictions: [batch, num_layers, num_queries, mask_h, mask_w]
    pub decoder_out_masks: MxArray,
}

// ============================================================================
// Full Decoder
// ============================================================================

/// PP-DocLayoutV3 Transformer Decoder.
///
/// Consists of a stack of decoder layers with iterative bounding box refinement.
/// After each layer:
/// 1. Bbox refinement: predict offset → add to inverse_sigmoid(reference_points) → sigmoid
/// 2. Class prediction: class_embed(norm(hidden_states))
/// 3. Mask generation: mask_query_head(norm(hidden_states)) @ mask_feat
/// 4. Reading order: global_pointer(order_head(norm(hidden_states)))
///
/// Corresponds to PPDocLayoutV3Decoder in the reference.
pub struct Decoder {
    /// Stack of decoder layers
    layers: Vec<DecoderLayer>,
    /// Query position head: maps reference points (4D) to position embeddings
    query_pos_head: MLPPredictionHead,
    /// Bbox refinement head (one shared across layers, or set externally)
    pub bbox_embed: Option<MLPPredictionHead>,
    /// Class prediction head (one shared, or set externally)
    pub class_embed: Option<Linear>,
    /// Number of object queries
    num_queries: i32,
}

impl Decoder {
    /// Create a new Decoder.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (default 256)
    /// * `ffn_dim` - FFN dimension (default 1024)
    /// * `num_heads` - Number of attention heads (default 8)
    /// * `n_levels` - Number of feature levels (default 3)
    /// * `n_points` - Number of sampling points (default 4)
    /// * `num_layers` - Number of decoder layers (default 6)
    /// * `num_queries` - Number of object queries (default 300)
    /// * `layer_norm_eps` - LayerNorm epsilon
    /// * `activation` - Activation function name
    pub fn new(
        d_model: i32,
        ffn_dim: i32,
        num_heads: i32,
        n_levels: i32,
        n_points: i32,
        num_layers: i32,
        num_queries: i32,
        layer_norm_eps: f64,
        activation: &str,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers as usize);
        for _ in 0..num_layers {
            layers.push(DecoderLayer::new(
                d_model,
                ffn_dim,
                num_heads,
                n_levels,
                n_points,
                layer_norm_eps,
                activation,
            )?);
        }

        // query_pos_head: MLP(4, 2*d_model, d_model, 2 layers)
        let query_pos_head = MLPPredictionHead::new(4, (2 * d_model) as u32, d_model as u32, 2)?;

        Ok(Self {
            layers,
            query_pos_head,
            bbox_embed: None,
            class_embed: None,
            num_queries,
        })
    }

    /// Create from pre-loaded components.
    pub fn from_components(
        layers: Vec<DecoderLayer>,
        query_pos_head: MLPPredictionHead,
        bbox_embed: Option<MLPPredictionHead>,
        class_embed: Option<Linear>,
        num_queries: i32,
    ) -> Self {
        Self {
            layers,
            query_pos_head,
            bbox_embed,
            class_embed,
            num_queries,
        }
    }

    /// Forward pass through the full decoder.
    ///
    /// # Arguments
    /// * `inputs_embeds` - Query embeddings [batch, num_queries, d_model]
    /// * `encoder_hidden_states` - Flattened encoder features [batch, total_seq_len, d_model]
    /// * `reference_points` - Initial reference points (pre-sigmoid) [batch, num_queries, 4]
    /// * `spatial_shapes` - [num_levels, 2] (height, width)
    /// * `spatial_shapes_list` - Vec of (height, width)
    /// * `level_start_index` - [num_levels]
    /// * `encoder_attention_mask` - Optional mask for self-attention [target_len, target_len]
    /// * `order_heads` - Per-layer order head linear projections
    /// * `global_pointer` - Global pointer reading order head
    /// * `mask_query_head` - Mask query MLP head
    /// * `norm` - Output normalization (LayerNorm)
    /// * `mask_feat` - Mask features from encoder [batch, num_prototypes, mask_h, mask_w] (NCHW)
    ///
    /// # Returns
    /// DecoderOutput with all intermediate predictions
    pub fn forward(
        &self,
        inputs_embeds: &MxArray,
        encoder_hidden_states: &MxArray,
        reference_points_input: &MxArray,
        spatial_shapes: &MxArray,
        spatial_shapes_list: &[(i64, i64)],
        level_start_index: &MxArray,
        encoder_attention_mask: Option<&MxArray>,
        order_heads: &[Linear],
        global_pointer: &GlobalPointer,
        mask_query_head: &MLPPredictionHead,
        norm: &LayerNorm,
        mask_feat: &MxArray,
    ) -> Result<DecoderOutput> {
        let mut hidden_states = inputs_embeds.clone();

        // Apply sigmoid to get initial reference points in [0, 1]
        let mut reference_points = Activations::sigmoid(reference_points_input)?;

        let mut intermediate_hidden_states: Vec<MxArray> = Vec::new();
        let mut intermediate_reference_points: Vec<MxArray> = Vec::new();
        let mut intermediate_logits: Vec<MxArray> = Vec::new();
        let mut decoder_out_order_logits: Vec<MxArray> = Vec::new();
        let mut decoder_out_masks: Vec<MxArray> = Vec::new();

        for (idx, decoder_layer) in self.layers.iter().enumerate() {
            // Expand reference points: [B, Q, 4] → [B, Q, n_levels, 4] for deformable attention
            // The deformable attention slices along the level dimension, so we need to
            // broadcast the reference points across all feature levels.
            let n_levels = spatial_shapes_list.len() as i64;
            let rp_shape = reference_points.shape()?;
            let rp_shape: Vec<i64> = rp_shape.as_ref().to_vec();
            let reference_points_expanded = reference_points.expand_dims(2)?.broadcast_to(&[
                rp_shape[0],
                rp_shape[1],
                n_levels,
                rp_shape[2],
            ])?;

            // Compute position embeddings from reference points
            let position_embeddings = self.query_pos_head.forward(&reference_points)?;

            // Run decoder layer
            hidden_states = decoder_layer.forward(
                &hidden_states,
                &position_embeddings,
                encoder_hidden_states,
                &reference_points_expanded,
                spatial_shapes,
                spatial_shapes_list,
                level_start_index,
                encoder_attention_mask,
            )?;

            // Iterative bbox refinement
            if let Some(ref bbox_embed) = self.bbox_embed {
                let predicted_corners = bbox_embed.forward(&hidden_states)?;
                let inv_ref = inverse_sigmoid(&reference_points, 1e-5)?;
                let new_reference_points = Activations::sigmoid(&predicted_corners.add(&inv_ref)?)?;
                reference_points = new_reference_points.clone();

                intermediate_reference_points.push(new_reference_points);
            } else {
                intermediate_reference_points.push(reference_points.clone());
            }

            intermediate_hidden_states.push(hidden_states.clone());

            // Post-layer predictions: class, mask, order
            let out_query = norm.forward(&hidden_states)?;

            // Mask generation: mask_query_head(out_query) @ mask_feat.flatten(2)
            let mask_query_embed = mask_query_head.forward(&out_query)?;
            // mask_query_embed: [B, Q, num_prototypes]
            // mask_feat: [B, num_prototypes, mask_h, mask_w]

            let mf_shape = mask_feat.shape()?;
            let mf_shape: Vec<i64> = mf_shape.as_ref().to_vec();
            let mask_h = mf_shape[2];
            let mask_w = mf_shape[3];
            let batch_size = mf_shape[0];

            // Flatten mask_feat spatial dims: [B, num_prototypes, mask_h * mask_w]
            let mask_feat_flat = mask_feat.reshape(&[batch_size, mf_shape[1], mask_h * mask_w])?;

            // bmm: [B, Q, num_prototypes] @ [B, num_prototypes, mask_h*mask_w] → [B, Q, mask_h*mask_w]
            let out_mask = mask_query_embed.matmul(&mask_feat_flat)?;
            // Reshape to [B, Q, mask_h, mask_w]
            let mq_shape = mask_query_embed.shape()?;
            let mq_shape: Vec<i64> = mq_shape.as_ref().to_vec();
            let mask_dim = mq_shape[1];
            let out_mask = out_mask.reshape(&[batch_size, mask_dim, mask_h, mask_w])?;
            decoder_out_masks.push(out_mask);

            // Class prediction
            if let Some(ref class_embed) = self.class_embed {
                let logits = class_embed.forward(&out_query)?;
                intermediate_logits.push(logits);
            }

            // Reading order prediction
            if idx < order_heads.len() {
                // Extract valid queries (last num_queries queries, in case denoising added more)
                let q_shape = out_query.shape()?;
                let q_shape: Vec<i64> = q_shape.as_ref().to_vec();
                let total_queries = q_shape[1];
                let nq = self.num_queries as i64;
                let start = total_queries - nq;

                let valid_query = if start > 0 {
                    out_query.slice(&[0, start, 0], &[q_shape[0], total_queries, q_shape[2]])?
                } else {
                    out_query.clone()
                };

                let order_projected = order_heads[idx].forward(&valid_query)?;
                let order_logits = global_pointer.forward(&order_projected)?;
                decoder_out_order_logits.push(order_logits);
            }
        }

        // Stack all intermediate outputs along dim=1
        let inter_hidden_refs: Vec<&MxArray> = intermediate_hidden_states.iter().collect();
        let stacked_hidden = MxArray::stack(inter_hidden_refs, Some(1))?;

        let inter_ref_refs: Vec<&MxArray> = intermediate_reference_points.iter().collect();
        let stacked_refs = MxArray::stack(inter_ref_refs, Some(1))?;

        let stacked_logits = if !intermediate_logits.is_empty() {
            let inter_logit_refs: Vec<&MxArray> = intermediate_logits.iter().collect();
            MxArray::stack(inter_logit_refs, Some(1))?
        } else {
            MxArray::zeros(&[1], None)?
        };

        let stacked_order = if !decoder_out_order_logits.is_empty() {
            let order_refs: Vec<&MxArray> = decoder_out_order_logits.iter().collect();
            MxArray::stack(order_refs, Some(1))?
        } else {
            MxArray::zeros(&[1], None)?
        };

        let mask_refs: Vec<&MxArray> = decoder_out_masks.iter().collect();
        let stacked_masks = MxArray::stack(mask_refs, Some(1))?;

        Ok(DecoderOutput {
            last_hidden_state: hidden_states,
            intermediate_hidden_states: stacked_hidden,
            intermediate_logits: stacked_logits,
            intermediate_reference_points: stacked_refs,
            decoder_out_order_logits: stacked_order,
            decoder_out_masks: stacked_masks,
        })
    }
}

// ============================================================================
// Utility: mask_to_box_coordinate
// ============================================================================

/// Convert binary masks to bounding box coordinates in center format.
///
/// For each mask in the batch, finds the bounding box of the True region
/// and returns normalized (center_x, center_y, width, height) coordinates.
///
/// # Arguments
/// * `mask` - Boolean mask tensor [..., height, width]
///
/// # Returns
/// Bounding boxes [..., 4] in (cx, cy, w, h) format, normalized to [0, 1]
pub fn mask_to_box_coordinate(mask: &MxArray) -> Result<MxArray> {
    let shape = mask.shape()?;
    let shape: Vec<i64> = shape.as_ref().to_vec();
    let ndim = shape.len();

    let height = shape[ndim - 2];
    let width = shape[ndim - 1];

    // Create coordinate grids
    let y_coords = MxArray::arange(0.0, height as f64, Some(1.0), None)?;
    let x_coords = MxArray::arange(0.0, width as f64, Some(1.0), None)?;

    // y_coords: [H, 1], x_coords: [1, W]
    let y_grid = y_coords
        .reshape(&[height, 1])?
        .broadcast_to(&[height, width])?;
    let x_grid = x_coords
        .reshape(&[1, width])?
        .broadcast_to(&[height, width])?;

    // Mask the coordinates: where mask is True, keep coord; where False, use 0 for max, 1e8 for min
    let large = MxArray::full(&[1], Either::A(1e8), None)?;

    // x_coords_masked = x_grid * mask
    let mask_float = mask.astype(crate::array::DType::Float32)?;

    // For max: masked positions get 0 (won't affect max of valid coords)
    let x_for_max = x_grid.mul(&mask_float)?;
    let y_for_max = y_grid.mul(&mask_float)?;

    // For min: masked-out positions get 1e8 (won't affect min of valid coords)
    let x_for_min = mask_float.where_(&x_grid, &large)?;
    let y_for_min = mask_float.where_(&y_grid, &large)?;

    // Flatten spatial dims and compute min/max
    // We need to handle arbitrary batch dimensions.
    // Flatten the last 2 dims into one, then reduce.
    let batch_shape: Vec<i64> = shape[..ndim - 2].to_vec();
    let spatial = height * width;

    let mut flat_shape = batch_shape.clone();
    flat_shape.push(spatial);

    let x_max_flat = x_for_max.reshape(&flat_shape)?;
    let x_max = x_max_flat.max(Some(&[-1]), Some(false))?.add_scalar(1.0)?;

    let x_min_flat = x_for_min.reshape(&flat_shape)?;
    let x_min = x_min_flat.min(Some(&[-1]), Some(false))?;

    let y_max_flat = y_for_max.reshape(&flat_shape)?;
    let y_max = y_max_flat.max(Some(&[-1]), Some(false))?.add_scalar(1.0)?;

    let y_min_flat = y_for_min.reshape(&flat_shape)?;
    let y_min = y_min_flat.min(Some(&[-1]), Some(false))?;

    // Check for empty masks: any(mask, dim=(-2, -1))
    let any_mask = mask_float
        .reshape(&flat_shape)?
        .max(Some(&[-1]), Some(false))?;
    // any_mask > 0 means mask has at least one True value

    // Stack [x_min, y_min, x_max, y_max]
    let bbox_parts = [&x_min, &y_min, &x_max, &y_max];
    let bbox_refs: Vec<&MxArray> = bbox_parts.to_vec();
    let unnormalized_bbox = MxArray::stack(bbox_refs, Some(-1))?;

    // Zero out empty masks
    let any_mask_expanded = any_mask.expand_dims(-1)?;
    let unnormalized_bbox = unnormalized_bbox.mul(&any_mask_expanded)?;

    // Normalize by [width, height, width, height]
    let norm_data = [width as f32, height as f32, width as f32, height as f32];
    let norm_tensor = MxArray::from_float32(&norm_data, &[4])?;
    let normalized = unnormalized_bbox.div(&norm_tensor)?;

    // Convert from xyxy to cxcywh
    // Slice out components
    let mut bbox_shape = batch_shape.clone();
    bbox_shape.push(4);

    let normalized = normalized.reshape(&bbox_shape)?;

    // x_min_n, y_min_n, x_max_n, y_max_n
    let starts_x_min = vec![0i64; bbox_shape.len()];
    let mut stops_x_min: Vec<i64> = bbox_shape.clone();
    *stops_x_min.last_mut().unwrap() = 1;

    let mut starts_y_min = vec![0i64; bbox_shape.len()];
    *starts_y_min.last_mut().unwrap() = 1;
    let mut stops_y_min: Vec<i64> = bbox_shape.clone();
    *stops_y_min.last_mut().unwrap() = 2;

    let mut starts_x_max = vec![0i64; bbox_shape.len()];
    *starts_x_max.last_mut().unwrap() = 2;
    let mut stops_x_max: Vec<i64> = bbox_shape.clone();
    *stops_x_max.last_mut().unwrap() = 3;

    let mut starts_y_max = vec![0i64; bbox_shape.len()];
    *starts_y_max.last_mut().unwrap() = 3;
    let stops_y_max: Vec<i64> = bbox_shape.clone();

    let x_min_n = normalized.slice(&starts_x_min, &stops_x_min)?;
    let y_min_n = normalized.slice(&starts_y_min, &stops_y_min)?;
    let x_max_n = normalized.slice(&starts_x_max, &stops_x_max)?;
    let y_max_n = normalized.slice(&starts_y_max, &stops_y_max)?;

    // center_x = (x_min + x_max) / 2, center_y = (y_min + y_max) / 2
    let center_x = x_min_n.add(&x_max_n)?.div_scalar(2.0)?;
    let center_y = y_min_n.add(&y_max_n)?.div_scalar(2.0)?;
    let box_w = x_max_n.sub(&x_min_n)?;
    let box_h = y_max_n.sub(&y_min_n)?;

    // Stack [cx, cy, w, h] along last dim
    let parts = [&center_x, &center_y, &box_w, &box_h];
    let part_refs: Vec<&MxArray> = parts.to_vec();
    let result = MxArray::concatenate_many(part_refs, Some(-1))?;

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_sigmoid() {
        // inverse_sigmoid(0.5) should be ~0
        let x = MxArray::from_float32(&[0.5], &[1]).unwrap();
        let result = inverse_sigmoid(&x, 1e-5).unwrap();
        result.eval();
        let val = result.to_float32().unwrap();
        assert!((val[0]).abs() < 0.01);

        // inverse_sigmoid(sigmoid(2.0)) should be ~2.0
        let x = MxArray::from_float32(&[2.0], &[1]).unwrap();
        let sig = Activations::sigmoid(&x).unwrap();
        let inv = inverse_sigmoid(&sig, 1e-5).unwrap();
        inv.eval();
        let val = inv.to_float32().unwrap();
        assert!((val[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_self_attention_shapes() {
        let attn = SelfAttention::new(64, 4).unwrap();
        let hidden = MxArray::random_uniform(&[1, 10, 64], -1.0, 1.0, None).unwrap();
        let pos = MxArray::random_uniform(&[1, 10, 64], -1.0, 1.0, None).unwrap();

        let output = attn.forward(&hidden, Some(&pos), None).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 10, 64]);
    }

    #[test]
    fn test_self_attention_no_position() {
        let attn = SelfAttention::new(64, 4).unwrap();
        let hidden = MxArray::random_uniform(&[2, 5, 64], -1.0, 1.0, None).unwrap();

        let output = attn.forward(&hidden, None, None).unwrap();
        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![2, 5, 64]);
    }

    #[test]
    fn test_multiscale_deformable_attention_shapes() {
        let msda = MultiscaleDeformableAttention::new(64, 4, 2, 2).unwrap();

        let batch = 1i64;
        let num_queries = 10i64;
        let h1 = 4i64;
        let w1 = 4i64;
        let h2 = 2i64;
        let w2 = 2i64;
        let total_seq = h1 * w1 + h2 * w2;

        let hidden = MxArray::random_uniform(&[batch, num_queries, 64], -1.0, 1.0, None).unwrap();
        let encoder = MxArray::random_uniform(&[batch, total_seq, 64], -1.0, 1.0, None).unwrap();

        // Reference points: [B, Q, 1, 2] normalized
        let ref_points =
            MxArray::random_uniform(&[batch, num_queries, 1, 2], 0.1, 0.9, None).unwrap();

        let spatial_shapes =
            MxArray::from_float32(&[h1 as f32, w1 as f32, h2 as f32, w2 as f32], &[2, 2]).unwrap();
        let spatial_shapes_list = vec![(h1, w1), (h2, w2)];
        let level_start = MxArray::from_float32(&[0.0, (h1 * w1) as f32], &[2]).unwrap();

        let output = msda
            .forward(
                &hidden,
                &encoder,
                None,
                &ref_points,
                &spatial_shapes,
                &spatial_shapes_list,
                &level_start,
            )
            .unwrap();

        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![batch, num_queries, 64]);
    }

    #[test]
    fn test_decoder_layer_shapes() {
        let layer = DecoderLayer::new(64, 128, 4, 2, 2, 1e-5, "relu").unwrap();

        let batch = 1i64;
        let num_queries = 10i64;
        let h1 = 4i64;
        let w1 = 4i64;
        let h2 = 2i64;
        let w2 = 2i64;
        let total_seq = h1 * w1 + h2 * w2;

        let hidden = MxArray::random_uniform(&[batch, num_queries, 64], -1.0, 1.0, None).unwrap();
        let pos = MxArray::random_uniform(&[batch, num_queries, 64], -1.0, 1.0, None).unwrap();
        let encoder = MxArray::random_uniform(&[batch, total_seq, 64], -1.0, 1.0, None).unwrap();
        let ref_points =
            MxArray::random_uniform(&[batch, num_queries, 1, 2], 0.1, 0.9, None).unwrap();
        let spatial_shapes =
            MxArray::from_float32(&[h1 as f32, w1 as f32, h2 as f32, w2 as f32], &[2, 2]).unwrap();
        let spatial_shapes_list = vec![(h1, w1), (h2, w2)];
        let level_start = MxArray::from_float32(&[0.0, (h1 * w1) as f32], &[2]).unwrap();

        let output = layer
            .forward(
                &hidden,
                &pos,
                &encoder,
                &ref_points,
                &spatial_shapes,
                &spatial_shapes_list,
                &level_start,
                None,
            )
            .unwrap();

        let shape: Vec<i64> = output.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![batch, num_queries, 64]);
    }

    #[test]
    fn test_mask_to_box_coordinate() {
        // Create a simple 4x4 mask with a 2x2 region set to true
        // Region at rows 1-2, cols 1-2
        let mask_data: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 1.0, 0.0, // row 1
            0.0, 1.0, 1.0, 0.0, // row 2
            0.0, 0.0, 0.0, 0.0, // row 3
        ];
        let mask = MxArray::from_float32(&mask_data, &[1, 4, 4]).unwrap();

        let bbox = mask_to_box_coordinate(&mask).unwrap();
        bbox.eval();

        let shape: Vec<i64> = bbox.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![1, 4]);

        let data = bbox.to_float32().unwrap();
        let data: Vec<f32> = data.to_vec();

        // Expected: x_min=1, y_min=1, x_max=3, y_max=3 (exclusive)
        // Normalized by [4, 4, 4, 4]: [0.25, 0.25, 0.75, 0.75]
        // center_x = (0.25 + 0.75) / 2 = 0.5
        // center_y = (0.25 + 0.75) / 2 = 0.5
        // width = 0.75 - 0.25 = 0.5
        // height = 0.75 - 0.25 = 0.5
        assert!((data[0] - 0.5).abs() < 0.01, "cx: {}", data[0]);
        assert!((data[1] - 0.5).abs() < 0.01, "cy: {}", data[1]);
        assert!((data[2] - 0.5).abs() < 0.01, "w: {}", data[2]);
        assert!((data[3] - 0.5).abs() < 0.01, "h: {}", data[3]);
    }
}
