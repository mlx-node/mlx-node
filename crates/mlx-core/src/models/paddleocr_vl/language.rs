/**
 * ERNIE Language Model for PaddleOCR-VL
 *
 * The language model component uses multimodal RoPE (mRoPE) which splits
 * the head dimension into sections for temporal, height, and width positions.
 */
use crate::array::{MxArray, scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::models::paddleocr_vl::config::TextConfig;
use crate::nn::activations::Activations;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::transformer::KVCache;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// Multimodal Rotary Position Embedding (internal)
///
/// Unlike standard RoPE which uses 1D positions, mRoPE uses 3D positions
/// (temporal, height, width) to encode spatial relationships in vision tokens.
/// The head dimension is split into sections according to `mrope_section`.
///
/// Note: This is an internal implementation detail used by ERNIELanguageModel.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct MultimodalRoPE {
    /// mRoPE sections [temporal, height, width] (e.g., [16, 24, 24])
    mrope_section: [i32; 3],
    /// Pre-computed inverse frequencies, pre-shaped to [1, 1, half_dim, 1] for broadcasting
    inv_freq: Arc<MxArray>,
    /// Cached half_dim value to avoid FFI .shape() calls
    inv_freq_dim: i64,
    /// Attention scaling factor
    attention_scaling: f32,
}

impl MultimodalRoPE {
    /// Create a new Multimodal RoPE
    ///
    /// # Arguments
    /// * `dim` - Head dimension (e.g., 128)
    /// * `max_position_embeddings` - Maximum sequence length
    /// * `base` - Base theta (default 500000.0 for PaddleOCR-VL)
    /// * `mrope_section` - Section sizes [temporal, height, width]
    pub fn new(
        dim: i32,
        _max_position_embeddings: i32,
        base: Option<f64>,
        mrope_section: Vec<i32>,
    ) -> Result<Self> {
        let base = base.unwrap_or(500000.0) as f32;

        if mrope_section.len() != 3 {
            return Err(Error::new(
                Status::InvalidArg,
                "mrope_section must have exactly 3 elements [t, h, w]",
            ));
        }

        let section_sum: i32 = mrope_section.iter().map(|&x| x * 2).sum();
        if section_sum != dim {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "mrope_section sum ({}) * 2 = {} must equal dim ({})",
                    mrope_section.iter().sum::<i32>(),
                    section_sum,
                    dim
                ),
            ));
        }

        let mrope_section_arr: [i32; 3] = [mrope_section[0], mrope_section[1], mrope_section[2]];

        // Compute inverse frequencies: 1 / (base^(2i/dim))
        let half_dim = dim / 2;
        let inv_freq_dim = half_dim as i64;
        let mut inv_freq_data = Vec::with_capacity(half_dim as usize);
        for i in (0..dim).step_by(2) {
            let exp = i as f32 / dim as f32;
            inv_freq_data.push(1.0 / base.powf(exp));
        }
        // Pre-reshape to [1, 1, half_dim, 1] to avoid reshape+astype in every forward call.
        // Already float32, so no astype needed.
        let inv_freq = MxArray::from_float32(&inv_freq_data, &[1, 1, inv_freq_dim, 1])?;

        Ok(Self {
            mrope_section: mrope_section_arr,
            inv_freq: Arc::new(inv_freq),
            inv_freq_dim,
            attention_scaling: 1.0,
        })
    }

    /// Compute cos and sin for rotary embeddings
    ///
    /// # Arguments
    /// * `x` - Input tensor (used only for dtype)
    /// * `position_ids` - Position IDs [3, batch, seq_len] for t, h, w
    ///
    /// # Returns
    /// * Tuple of (cos, sin) each of shape [batch, 1, seq_len, head_dim]
    pub fn forward(&self, x: &MxArray, position_ids: &MxArray) -> Result<(MxArray, MxArray)> {
        let target_dtype = x.dtype()?; // 1 FFI call
        let pos_shape = position_ids.shape()?; // 1 FFI call

        // position_ids: [3, batch, seq_len]
        let batch_size = pos_shape[1];
        let seq_len = pos_shape[2];

        // inv_freq is pre-shaped to [1, 1, half_dim, 1] and already float32.
        // Broadcast to [3, batch, half_dim, 1] — uses cached inv_freq_dim.
        let inv_freq_expanded =
            MxArray::broadcast_to(&self.inv_freq, &[3, batch_size, self.inv_freq_dim, 1])?; // 1 FFI call

        // Expand position_ids: [3, batch, 1, seq_len] and cast to float32
        let pos_expanded = position_ids
            .reshape(&[3, batch_size, 1, seq_len])? // 1 FFI call
            .astype(crate::array::DType::Float32)?; // 1 FFI call

        // Compute freqs: inv_freq @ position_ids -> [3, batch, half_dim, seq_len]
        let freqs = inv_freq_expanded.matmul(&pos_expanded)?; // 1 FFI call

        // Transpose: [3, batch, seq_len, half_dim]
        let freqs = freqs.transpose(Some(&[0, 1, 3, 2]))?; // 1 FFI call

        // Concatenate freqs with itself: [3, batch, seq_len, dim]
        let emb = MxArray::concatenate_many(vec![&freqs, &freqs], Some(-1))?; // 1 FFI call

        // Compute cos and sin.
        // Skip mul_scalar when attention_scaling == 1.0 (saves 2 FFI calls).
        let (cos, sin) = if (self.attention_scaling - 1.0).abs() < 1e-8 {
            (emb.cos()?, emb.sin()?) // 2 FFI calls
        } else {
            let cos = emb.cos()?.mul_scalar(self.attention_scaling as f64)?;
            let sin = emb.sin()?.mul_scalar(self.attention_scaling as f64)?;
            (cos, sin)
        };

        // Cast to target dtype only if needed (saves 2 FFI calls when already float32)
        let cos = if target_dtype == crate::array::DType::Float32 {
            cos
        } else {
            cos.astype(target_dtype)? // conditional FFI call
        };
        let sin = if target_dtype == crate::array::DType::Float32 {
            sin
        } else {
            sin.astype(target_dtype)? // conditional FFI call
        };

        Ok((cos, sin))
    }

    /// Get mRoPE sections
    pub fn mrope_section(&self) -> Vec<i32> {
        self.mrope_section.to_vec()
    }

    /// Get raw inv_freq pointer for C++ forward pass
    pub fn get_inv_freq_ptr(&self) -> *mut sys::mlx_array {
        self.inv_freq.handle.0
    }

    /// Get mRoPE section array reference
    pub fn mrope_section_arr(&self) -> &[i32; 3] {
        &self.mrope_section
    }
}

/// Rotate half of the input tensor
///
/// Accepts pre-computed ndim and last_dim to avoid redundant .shape() FFI calls.
fn rotate_half(x: &MxArray, ndim: usize, last_dim: i64) -> Result<MxArray> {
    let half_dim = last_dim / 2;

    let x1 = x.slice_axis(ndim - 1, 0, half_dim)?;
    let x2 = x.slice_axis(ndim - 1, half_dim, last_dim)?;

    let neg_x2 = x2.mul_scalar(-1.0)?;
    MxArray::concatenate_many(vec![&neg_x2, &x1], Some(-1))
}

/// Apply multimodal rotary position embedding to Q and K (internal)
///
/// Uses split+concat pattern matching Python mlx-vlm for minimal FFI overhead.
/// cos/sin shape: [3, batch, seq_len, head_dim] where dim 0 = [temporal, height, width]
///
/// The mrope_section (e.g. [16, 24, 24]) defines how the head_dim is partitioned.
/// Each section i selects from spatial dimension i%3. After doubling (for sin/cos halves),
/// we split the last axis at cumulative indices and pick the right spatial row for each part,
/// then concatenate back. This mirrors Python's:
///   mrope_section = cumsum(mrope_section * 2)[:-1]
///   cos = concat([m[i%3] for i,m in enumerate(split(cos, mrope_section, axis=-1))], axis=-1)
///
/// Note: Internal implementation detail, not exposed to TypeScript.
pub fn apply_multimodal_rotary_pos_emb(
    q: &MxArray,
    k: &MxArray,
    cos: &MxArray,
    sin: &MxArray,
    mrope_section: Vec<i32>,
) -> Result<(MxArray, MxArray)> {
    // Compute cumulative section boundaries
    // e.g., [16, 24, 24] -> cumsum([32, 48, 48]) = [32, 80, 128]
    let mut boundaries: Vec<i64> = Vec::new();
    let mut cumsum = 0i64;
    for &s in &mrope_section {
        cumsum += (s * 2) as i64;
        boundaries.push(cumsum);
    }

    let cos_shape = cos.shape()?; // 1 FFI call — cache and reuse below
    let batch_size = cos_shape[1];
    let seq_len = cos_shape[2];
    let head_dim = cos_shape[3];

    // Split cos/sin by mRoPE sections and interleave
    // For each section i, take cos/sin from position i mod 3
    let mut cos_parts: Vec<MxArray> = Vec::new();
    let mut sin_parts: Vec<MxArray> = Vec::new();

    let mut start = 0i64;
    for (idx, &end) in boundaries.iter().enumerate() {
        let section_idx = idx % 3;

        // Extract section from cos/sin at position section_idx
        let cos_section = cos.slice_axis(0, section_idx as i64, section_idx as i64 + 1)?;
        let sin_section = sin.slice_axis(0, section_idx as i64, section_idx as i64 + 1)?;

        // Squeeze to remove the first dimension: [batch, seq_len, head_dim]
        let cos_section = cos_section.squeeze(Some(&[0]))?;
        let sin_section = sin_section.squeeze(Some(&[0]))?;

        // Slice the head_dim dimension
        let cos_slice = cos_section.slice_axis(2, start, end)?;
        let sin_slice = sin_section.slice_axis(2, start, end)?;

        cos_parts.push(cos_slice);
        sin_parts.push(sin_slice);

        start = end;
    }

    // Concatenate parts
    let cos_refs: Vec<&MxArray> = cos_parts.iter().collect();
    let sin_refs: Vec<&MxArray> = sin_parts.iter().collect();
    let cos_final = MxArray::concatenate_many(cos_refs, Some(-1))?;
    let sin_final = MxArray::concatenate_many(sin_refs, Some(-1))?;

    // Unsqueeze to [batch, 1, seq_len, head_dim] for broadcasting with heads
    let cos_final = cos_final.reshape(&[batch_size, 1, seq_len, head_dim])?;
    let sin_final = sin_final.reshape(&[batch_size, 1, seq_len, head_dim])?;

    // rotary_dim == head_dim (from cos_shape, already cached above)
    let rotary_dim = head_dim;
    let q_shape = q.shape()?; // 1 FFI call
    let q_dim = q_shape[3];
    let q_ndim = q_shape.len();

    // Split Q and K into rotary and pass-through parts
    let q_rot = q.slice_axis(3, 0, rotary_dim)?;
    let q_pass = if rotary_dim < q_dim {
        Some(q.slice_axis(3, rotary_dim, q_dim)?)
    } else {
        None
    };

    let k_rot = k.slice_axis(3, 0, rotary_dim)?;
    let k_pass = if rotary_dim < q_dim {
        Some(k.slice_axis(3, rotary_dim, q_dim)?)
    } else {
        None
    };

    // Apply rotary: q_rot * cos + rotate_half(q_rot) * sin
    // Pass pre-computed ndim and last_dim to avoid .shape() calls inside rotate_half
    let q_rotated = rotate_half(&q_rot, q_ndim, rotary_dim)?;
    let k_rotated = rotate_half(&k_rot, q_ndim, rotary_dim)?;

    let q_embed = q_rot.mul(&cos_final)?.add(&q_rotated.mul(&sin_final)?)?;
    let k_embed = k_rot.mul(&cos_final)?.add(&k_rotated.mul(&sin_final)?)?;

    // Concatenate rotary and pass-through parts
    let q_out = if let Some(q_pass) = q_pass {
        MxArray::concatenate_many(vec![&q_embed, &q_pass], Some(-1))?
    } else {
        q_embed
    };

    let k_out = if let Some(k_pass) = k_pass {
        MxArray::concatenate_many(vec![&k_embed, &k_pass], Some(-1))?
    } else {
        k_embed
    };

    Ok((q_out, k_out))
}

/// PaddleOCR Attention with mRoPE (internal)
///
/// Note: This is an internal implementation detail used by PaddleOCRDecoderLayer.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct PaddleOCRAttention {
    q_proj: Arc<Linear>,
    k_proj: Arc<Linear>,
    v_proj: Arc<Linear>,
    o_proj: Arc<Linear>,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    mrope_section: Vec<i32>,
}

impl PaddleOCRAttention {
    /// Get raw weight pointers for C++ forward pass
    pub fn get_weight_ptrs(&self) -> [*mut sys::mlx_array; 4] {
        [
            self.q_proj.get_weight().handle.0,
            self.k_proj.get_weight().handle.0,
            self.v_proj.get_weight().handle.0,
            self.o_proj.get_weight().handle.0,
        ]
    }

    pub fn new(
        config: TextConfig,
        q_weight: &MxArray,
        k_weight: &MxArray,
        v_weight: &MxArray,
        o_weight: &MxArray,
    ) -> Result<Self> {
        let q_proj = Linear::from_weights(q_weight, None)?;
        let k_proj = Linear::from_weights(k_weight, None)?;
        let v_proj = Linear::from_weights(v_weight, None)?;
        let o_proj = Linear::from_weights(o_weight, None)?;

        let head_dim = config.head_dim;
        let scale = (head_dim as f32).powf(-0.5);

        Ok(Self {
            q_proj: Arc::new(q_proj),
            k_proj: Arc::new(k_proj),
            v_proj: Arc::new(v_proj),
            o_proj: Arc::new(o_proj),
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            head_dim,
            scale,
            mrope_section: config.mrope_section.clone(),
        })
    }

    /// Forward pass without KV caching (for compatibility)
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        position_embeddings: &MxArray,
    ) -> Result<MxArray> {
        self.forward_with_cache(x, mask, position_embeddings, None)
    }

    /// Forward pass with optional KV caching
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `position_embeddings` - Position embeddings [2, 3, batch, seq, dim]
    /// * `cache` - Optional KV cache for incremental generation
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, hidden_size]
    pub fn forward_with_cache(
        &self,
        x: &MxArray,
        _mask: Option<&MxArray>,
        position_embeddings: &MxArray,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        let batch = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q
            .reshape(&[batch, seq_len, self.n_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let k = k
            .reshape(&[batch, seq_len, self.n_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let v = v
            .reshape(&[batch, seq_len, self.n_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;

        // Extract cos and sin from position embeddings (packed as [2, 3, batch, seq, dim])
        let cos = position_embeddings
            .slice_axis(0, 0, 1)?
            .squeeze(Some(&[0]))?;
        let sin = position_embeddings
            .slice_axis(0, 1, 2)?
            .squeeze(Some(&[0]))?;

        // Apply mRoPE
        let (q, k) =
            apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin, self.mrope_section.clone())?;

        // Update KV cache if provided
        let (k, v) = if let Some(cache) = cache {
            cache.update_and_fetch(&k, &v)?
        } else {
            (k, v)
        };

        // Compute attention with causal masking
        // Note: MLX's SDPA handles GQA broadcasting internally - no need to
        // tile/reshape KV heads to match Q heads. Q has n_heads, K/V have n_kv_heads.
        // Use causal SDPA for prefill (seq_len > 1) to enforce causality
        // For generation with cache (seq_len == 1), single token can attend to all cached K/V
        let output = if seq_len > 1 {
            // Prefill: use causal attention
            scaled_dot_product_attention_causal(&q, &k, &v, self.scale as f64)?
        } else {
            // Decode: use fused SDPA with no mask (most optimized Metal kernel path)
            // For single-token generation, the token can attend to all cached K/V
            scaled_dot_product_attention(&q, &k, &v, self.scale as f64, None)?
        };

        // Reshape back
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.n_heads * self.head_dim) as i64])?;

        self.o_proj.forward(&output)
    }
}

/// PaddleOCR Decoder Layer (internal)
///
/// Note: This is an internal implementation detail used by ERNIELanguageModel.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct PaddleOCRDecoderLayer {
    self_attn: Arc<PaddleOCRAttention>,
    mlp_gate: Arc<Linear>,
    mlp_up: Arc<Linear>,
    mlp_down: Arc<Linear>,
    input_layernorm: Arc<RMSNorm>,
    post_attention_layernorm: Arc<RMSNorm>,
}

impl PaddleOCRDecoderLayer {
    /// Get all 9 weight pointers for this layer in the order expected by C++:
    /// [input_norm, post_attn_norm, q, k, v, o, gate, up, down]
    pub fn get_weight_ptrs(&self) -> [*mut sys::mlx_array; 9] {
        let attn_ptrs = self.self_attn.get_weight_ptrs();
        [
            self.input_layernorm.get_weight().handle.0,
            self.post_attention_layernorm.get_weight().handle.0,
            attn_ptrs[0], // q_proj
            attn_ptrs[1], // k_proj
            attn_ptrs[2], // v_proj
            attn_ptrs[3], // o_proj
            self.mlp_gate.get_weight().handle.0,
            self.mlp_up.get_weight().handle.0,
            self.mlp_down.get_weight().handle.0,
        ]
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: TextConfig,
        q_weight: &MxArray,
        k_weight: &MxArray,
        v_weight: &MxArray,
        o_weight: &MxArray,
        gate_weight: &MxArray,
        up_weight: &MxArray,
        down_weight: &MxArray,
        input_norm_weight: &MxArray,
        post_attn_norm_weight: &MxArray,
    ) -> Result<Self> {
        let self_attn =
            PaddleOCRAttention::new(config.clone(), q_weight, k_weight, v_weight, o_weight)?;

        let mlp_gate = Linear::from_weights(gate_weight, None)?;
        let mlp_up = Linear::from_weights(up_weight, None)?;
        let mlp_down = Linear::from_weights(down_weight, None)?;

        let input_layernorm = RMSNorm::from_weight(input_norm_weight, Some(config.rms_norm_eps))?;
        let post_attention_layernorm =
            RMSNorm::from_weight(post_attn_norm_weight, Some(config.rms_norm_eps))?;

        Ok(Self {
            self_attn: Arc::new(self_attn),
            mlp_gate: Arc::new(mlp_gate),
            mlp_up: Arc::new(mlp_up),
            mlp_down: Arc::new(mlp_down),
            input_layernorm: Arc::new(input_layernorm),
            post_attention_layernorm: Arc::new(post_attention_layernorm),
        })
    }

    /// Forward pass without KV caching (for compatibility)
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        position_embeddings: &MxArray,
    ) -> Result<MxArray> {
        self.forward_with_cache(x, mask, position_embeddings, None)
    }

    /// Forward pass with optional KV caching
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `position_embeddings` - Position embeddings [2, 3, batch, seq, dim]
    /// * `cache` - Optional KV cache for incremental generation
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, hidden_size]
    pub fn forward_with_cache(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        position_embeddings: &MxArray,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        // Self attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out =
            self.self_attn
                .forward_with_cache(&normed, mask, position_embeddings, cache)?;
        let h = x.add(&attn_out)?;

        // MLP with residual (SiLU gated)
        let normed = self.post_attention_layernorm.forward(&h)?;
        let gate_proj = self.mlp_gate.forward(&normed)?;
        let gate = Activations::silu(&gate_proj)?;
        let up = self.mlp_up.forward(&normed)?;
        let hidden = gate.mul(&up)?;
        let mlp_out = self.mlp_down.forward(&hidden)?;

        h.add(&mlp_out)
    }
}

/// ERNIE Language Model (internal)
///
/// Note: This is an internal implementation detail used by VLModel.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct ERNIELanguageModel {
    embed_tokens: Arc<Embedding>,
    layers: Vec<Arc<PaddleOCRDecoderLayer>>,
    norm: Arc<RMSNorm>,
    lm_head: Arc<Linear>,
    rotary_emb: Arc<MultimodalRoPE>,
    config: TextConfig,
    /// KV caches for incremental generation (one per layer)
    kv_caches: Option<Vec<KVCache>>,
    /// Stored position IDs for decode phase (computed during prefill with images)
    position_ids: Option<MxArray>,
    /// Stored rope deltas for decode phase offset calculation
    rope_deltas: Option<i64>,
    /// Fused C++ KV cache keys (one per layer)
    fused_kv_keys: Vec<Option<MxArray>>,
    /// Fused C++ KV cache values (one per layer)
    fused_kv_values: Vec<Option<MxArray>>,
    /// Fused C++ cache write position
    fused_cache_idx: i32,
}

impl ERNIELanguageModel {
    /// Create language model (layers will be set separately)
    pub fn new(
        config: TextConfig,
        embed_tokens_weight: &MxArray,
        final_norm_weight: &MxArray,
        lm_head_weight: &MxArray,
    ) -> Result<Self> {
        let embed_tokens = Embedding::from_weight(embed_tokens_weight)?;

        let norm = RMSNorm::from_weight(final_norm_weight, Some(config.rms_norm_eps))?;
        let lm_head = Linear::from_weights(lm_head_weight, None)?;

        let rotary_emb = MultimodalRoPE::new(
            config.head_dim,
            config.max_position_embeddings,
            Some(config.rope_theta),
            config.mrope_section.clone(),
        )?;

        Ok(Self {
            embed_tokens: Arc::new(embed_tokens),
            layers: Vec::new(),
            norm: Arc::new(norm),
            lm_head: Arc::new(lm_head),
            rotary_emb: Arc::new(rotary_emb),
            config,
            kv_caches: None,
            position_ids: None,
            rope_deltas: None,
            fused_kv_keys: Vec::new(),
            fused_kv_values: Vec::new(),
            fused_cache_idx: 0,
        })
    }

    /// Add a decoder layer
    pub fn add_layer(&mut self, layer: &PaddleOCRDecoderLayer) {
        self.layers.push(Arc::new(PaddleOCRDecoderLayer {
            self_attn: layer.self_attn.clone(),
            mlp_gate: layer.mlp_gate.clone(),
            mlp_up: layer.mlp_up.clone(),
            mlp_down: layer.mlp_down.clone(),
            input_layernorm: layer.input_layernorm.clone(),
            post_attention_layernorm: layer.post_attention_layernorm.clone(),
        }));
    }

    /// Forward pass without KV caching (for compatibility)
    pub fn forward(
        &self,
        input_ids: &MxArray,
        inputs_embeds: Option<&MxArray>,
        mask: Option<&MxArray>,
        position_ids: Option<&MxArray>,
    ) -> Result<MxArray> {
        // Get embeddings
        let h = if let Some(embeds) = inputs_embeds {
            embeds.clone()
        } else {
            self.embed_tokens.forward(input_ids)?
        };

        // Compute position embeddings
        let pos_ids = if let Some(ids) = position_ids {
            ids.clone()
        } else {
            // Default position IDs: [3, batch, seq_len] all same positions
            let shape = h.shape()?;
            let batch = shape[0];
            let seq_len = shape[1];
            let pos = MxArray::arange(0.0, seq_len as f64, Some(1.0), None)?;
            let pos = pos.reshape(&[1, 1, seq_len])?;
            pos.broadcast_to(&[3, batch, seq_len])?
        };

        let (cos, sin) = self.rotary_emb.forward(&h, &pos_ids)?;
        // Pack cos and sin together for passing to layers
        let position_embeddings = MxArray::stack(vec![&cos, &sin], Some(0))?;

        // Forward through layers
        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(&h, mask, &position_embeddings)?;
        }

        // Final norm
        h = self.norm.forward(&h)?;

        // LM head
        self.lm_head.forward(&h)
    }

    /// Initialize KV caches for incremental generation
    ///
    /// Creates one KV cache per transformer layer. Call this before starting generation.
    pub fn init_kv_caches(&mut self) {
        let num_layers = self.layers.len();
        self.kv_caches = Some((0..num_layers).map(|_| KVCache::new()).collect());
    }

    /// Reset all KV caches
    ///
    /// Clears cached key-value states. Call this between different generation sequences.
    pub fn reset_kv_caches(&mut self) {
        if let Some(ref mut caches) = self.kv_caches {
            for cache in caches.iter_mut() {
                cache.reset();
            }
        }
    }

    /// Set position IDs for the current generation sequence
    ///
    /// These are stored during prefill and used for proper position slicing during decode.
    pub fn set_position_state(&mut self, position_ids: MxArray, rope_deltas: i64) {
        self.position_ids = Some(position_ids);
        self.rope_deltas = Some(rope_deltas);
    }

    /// Reset position state (call when processing new image)
    pub fn reset_position_state(&mut self) {
        self.position_ids = None;
        self.rope_deltas = None;
    }

    /// Get stored rope deltas
    pub fn get_rope_deltas(&self) -> Option<i64> {
        self.rope_deltas
    }

    /// Get the current cache offset (number of cached tokens)
    pub fn get_cache_offset(&self) -> i32 {
        self.kv_caches
            .as_ref()
            .and_then(|caches| caches.first())
            .map(|cache| cache.get_offset())
            .unwrap_or(0)
    }

    /// Forward pass with KV caching for incremental generation
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `inputs_embeds` - Optional pre-computed embeddings
    /// * `mask` - Optional attention mask
    /// * `position_ids` - Optional position IDs [3, batch, seq_len]
    /// * `use_cache` - Whether to use KV caching (must call init_kv_caches() first)
    ///
    /// # Returns
    /// * Logits [batch, seq_len, vocab_size]
    pub fn forward_with_cache(
        &mut self,
        input_ids: &MxArray,
        inputs_embeds: Option<&MxArray>,
        mask: Option<&MxArray>,
        position_ids: Option<&MxArray>,
        use_cache: bool,
    ) -> Result<MxArray> {
        // Get embeddings
        let h = if let Some(embeds) = inputs_embeds {
            embeds.clone()
        } else {
            self.embed_tokens.forward(input_ids)?
        };

        // Get cache offset for position calculation
        let cache_offset = if use_cache {
            self.get_cache_offset()
        } else {
            0
        };

        // Compute position embeddings
        // Priority: 1) Passed position_ids (prefill) 2) Stored position_ids (decode with multimodal) 3) Sequential (text-only)
        let shape = h.shape()?;
        let batch = shape[0];
        let seq_len = shape[1];

        let pos_ids = if let Some(ids) = position_ids {
            // Prefill case: use provided position_ids directly
            ids.clone()
        } else if self.position_ids.is_some() {
            // Decode case with multimodal: position_ids being set indicates mRoPE mode
            // We don't use the stored value directly - instead compute fresh positions with rope_deltas offset
            // During decode, we compute position = arange(seq_len) + cache_offset + rope_deltas
            let rope_deltas = self.rope_deltas.unwrap_or(0);
            let delta = (cache_offset as i64 + rope_deltas) as f64;
            if seq_len == 1 {
                // Fast path for single-token decode: create [3, batch, 1] directly
                // Avoids arange + scalar_float + add + reshape (4 FFI calls → 2)
                let pos = MxArray::from_float32(&[delta as f32], &[1, 1, 1])?;
                pos.broadcast_to(&[3, batch, 1])?
            } else {
                let pos = MxArray::arange(delta, delta + seq_len as f64, Some(1.0), None)?;
                let pos = pos.reshape(&[1, 1, seq_len])?;
                pos.broadcast_to(&[3, batch, seq_len])?
            }
        } else {
            // Text-only decode: simple sequential positions with cache offset
            if seq_len == 1 {
                // Fast path for single-token decode
                let pos = MxArray::from_float32(&[cache_offset as f32], &[1, 1, 1])?;
                pos.broadcast_to(&[3, batch, 1])?
            } else {
                let pos = MxArray::arange(
                    cache_offset as f64,
                    (cache_offset as i64 + seq_len) as f64,
                    Some(1.0),
                    None,
                )?;
                let pos = pos.reshape(&[1, 1, seq_len])?;
                pos.broadcast_to(&[3, batch, seq_len])?
            }
        };

        let (cos, sin) = self.rotary_emb.forward(&h, &pos_ids)?;
        // Pack cos and sin together for passing to layers
        let position_embeddings = MxArray::stack(vec![&cos, &sin], Some(0))?;

        // Forward through layers with optional caching
        let mut h = h;
        if use_cache {
            if let Some(ref mut caches) = self.kv_caches {
                for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
                    h = layer.forward_with_cache(&h, mask, &position_embeddings, Some(cache))?;
                }
            } else {
                return Err(Error::new(
                    Status::GenericFailure,
                    "KV caches not initialized. Call init_kv_caches() first.",
                ));
            }
        } else {
            for layer in &self.layers {
                h = layer.forward(&h, mask, &position_embeddings)?;
            }
        }

        // Final norm
        h = self.norm.forward(&h)?;

        // LM head
        self.lm_head.forward(&h)
    }

    /// Get token embeddings without passing through the model
    pub fn get_embeddings(&self, input_ids: &MxArray) -> Result<MxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Get fused cache write position
    pub fn get_fused_cache_offset(&self) -> i32 {
        self.fused_cache_idx
    }

    /// Initialize fused KV cache state for C++ forward pass
    pub fn init_fused_kv_caches(&mut self) {
        let n = self.layers.len();
        self.fused_kv_keys = (0..n).map(|_| None).collect();
        self.fused_kv_values = (0..n).map(|_| None).collect();
        self.fused_cache_idx = 0;
    }

    /// Evaluate all fused KV cache arrays to materialize them.
    /// This is critical for chunked prefill - it forces the computation graph
    /// to be evaluated between chunks, preventing unbounded graph growth.
    pub fn eval_fused_kv_caches(&self) {
        for arr in self.fused_kv_keys.iter().flatten() {
            arr.eval();
        }
        for arr in self.fused_kv_values.iter().flatten() {
            arr.eval();
        }
    }

    /// Reset fused KV cache state
    pub fn reset_fused_kv_caches(&mut self) {
        for k in self.fused_kv_keys.iter_mut() {
            *k = None;
        }
        for v in self.fused_kv_values.iter_mut() {
            *v = None;
        }
        self.fused_cache_idx = 0;
    }

    /// Fused forward pass through C++ for minimal FFI overhead.
    ///
    /// This replaces the per-layer Rust forward_with_cache calls with a single
    /// C++ function that builds the entire computation graph in one FFI call.
    ///
    /// # Arguments
    /// * `input_embeds` - Input embeddings [batch, seq_len, hidden_size]
    /// * `position_ids` - Position IDs [3, batch, seq_len] for mRoPE
    ///
    /// # Returns
    /// * Logits [batch, seq_len, vocab_size]
    pub fn forward_fused(
        &mut self,
        input_embeds: &MxArray,
        position_ids: &MxArray,
    ) -> Result<MxArray> {
        let num_layers = self.layers.len() as i32;

        // Collect all layer weight pointers (9 per layer)
        let mut all_weight_ptrs: Vec<*mut sys::mlx_array> =
            Vec::with_capacity(num_layers as usize * 9);
        for layer in &self.layers {
            let ptrs = layer.get_weight_ptrs();
            all_weight_ptrs.extend_from_slice(&ptrs);
        }

        // Model-level weights
        let final_norm_ptr = self.norm.get_weight().handle.0;
        let lm_head_ptr = self.lm_head.get_weight().handle.0;
        let inv_freq_ptr = self.rotary_emb.get_inv_freq_ptr();
        let mrope_section = self.rotary_emb.mrope_section_arr();

        // Prepare KV cache input pointers
        let mut kv_keys_ptrs: Vec<*mut sys::mlx_array> = Vec::with_capacity(num_layers as usize);
        let mut kv_values_ptrs: Vec<*mut sys::mlx_array> = Vec::with_capacity(num_layers as usize);

        for i in 0..num_layers as usize {
            kv_keys_ptrs.push(
                self.fused_kv_keys[i]
                    .as_ref()
                    .map(|a| a.handle.0)
                    .unwrap_or(std::ptr::null_mut()),
            );
            kv_values_ptrs.push(
                self.fused_kv_values[i]
                    .as_ref()
                    .map(|a| a.handle.0)
                    .unwrap_or(std::ptr::null_mut()),
            );
        }

        // Prepare output buffers
        let mut out_logits: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_kv_keys: Vec<*mut sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers as usize];
        let mut out_kv_values: Vec<*mut sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers as usize];
        let mut out_cache_idx: i32 = 0;

        let config = &self.config;

        unsafe {
            sys::mlx_paddleocr_vl_forward_step(
                input_embeds.handle.0,
                all_weight_ptrs.as_ptr(),
                num_layers,
                final_norm_ptr,
                lm_head_ptr,
                inv_freq_ptr,
                position_ids.handle.0,
                mrope_section.as_ptr(),
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps as f32,
                kv_keys_ptrs.as_ptr(),
                kv_values_ptrs.as_ptr(),
                self.fused_cache_idx,
                &mut out_logits,
                out_kv_keys.as_mut_ptr(),
                out_kv_values.as_mut_ptr(),
                &mut out_cache_idx,
            );
        }

        // Update fused KV cache state from output
        self.fused_cache_idx = out_cache_idx;
        for i in 0..num_layers as usize {
            if !out_kv_keys[i].is_null() {
                self.fused_kv_keys[i] = Some(MxArray::from_handle(out_kv_keys[i], "fused_kv_key")?);
            }
            if !out_kv_values[i].is_null() {
                self.fused_kv_values[i] =
                    Some(MxArray::from_handle(out_kv_values[i], "fused_kv_value")?);
            }
        }

        MxArray::from_handle(out_logits, "paddleocr_vl_forward_fused")
    }

    /// Run fused forward pass and extract KV cache arrays for batch merging.
    ///
    /// Runs prefill through the existing single-item FFI, then clones out the
    /// per-layer KV cache arrays and resets cache state for the next item.
    ///
    /// # Returns
    /// * (logits, kv_keys, kv_values, cache_idx) where kv_keys/values are per-layer
    pub fn forward_fused_extract_kv(
        &mut self,
        input_embeds: &MxArray,
        position_ids: &MxArray,
    ) -> Result<(MxArray, Vec<MxArray>, Vec<MxArray>, i32)> {
        let logits = self.forward_fused(input_embeds, position_ids)?;

        // Eval KV caches to materialize them before cloning
        self.eval_fused_kv_caches();

        // Clone out KV arrays
        let num_layers = self.layers.len();
        let mut kv_keys = Vec::with_capacity(num_layers);
        let mut kv_values = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            kv_keys.push(
                self.fused_kv_keys[i]
                    .as_ref()
                    .ok_or_else(|| {
                        Error::new(
                            Status::GenericFailure,
                            format!("KV key cache missing for layer {}", i),
                        )
                    })?
                    .clone(),
            );
            kv_values.push(
                self.fused_kv_values[i]
                    .as_ref()
                    .ok_or_else(|| {
                        Error::new(
                            Status::GenericFailure,
                            format!("KV value cache missing for layer {}", i),
                        )
                    })?
                    .clone(),
            );
        }

        let cache_idx = self.fused_cache_idx;

        // Reset cache state for next item
        self.reset_fused_kv_caches();

        Ok((logits, kv_keys, kv_values, cache_idx))
    }

    /// Batched fused forward pass through C++ with left-padding-aware attention.
    ///
    /// Like forward_fused but takes external KV cache arrays and left_padding
    /// for batched decode. Does NOT update internal cache state.
    ///
    /// # Arguments
    /// * `input_embeds` - Input embeddings [batch, seq_len, hidden_size]
    /// * `position_ids` - Position IDs [3, batch, seq_len] for mRoPE
    /// * `left_padding` - Left padding amounts [batch]
    /// * `kv_keys` - Per-layer KV keys [batch, heads, seq, dim]
    /// * `kv_values` - Per-layer KV values [batch, heads, seq, dim]
    /// * `cache_idx` - Current cache write position
    ///
    /// # Returns
    /// * (logits, updated_kv_keys, updated_kv_values, new_cache_idx)
    pub fn forward_fused_batched(
        &self,
        input_embeds: &MxArray,
        position_ids: &MxArray,
        left_padding: &MxArray,
        kv_keys: &[Option<MxArray>],
        kv_values: &[Option<MxArray>],
        cache_idx: i32,
    ) -> Result<(MxArray, Vec<Option<MxArray>>, Vec<Option<MxArray>>, i32)> {
        let num_layers = self.layers.len() as i32;

        // Collect layer weight pointers
        let mut all_weight_ptrs: Vec<*mut sys::mlx_array> =
            Vec::with_capacity(num_layers as usize * 9);
        for layer in &self.layers {
            let ptrs = layer.get_weight_ptrs();
            all_weight_ptrs.extend_from_slice(&ptrs);
        }

        let final_norm_ptr = self.norm.get_weight().handle.0;
        let lm_head_ptr = self.lm_head.get_weight().handle.0;
        let inv_freq_ptr = self.rotary_emb.get_inv_freq_ptr();
        let mrope_section = self.rotary_emb.mrope_section_arr();

        // KV cache input pointers
        let kv_keys_ptrs: Vec<*mut sys::mlx_array> = kv_keys
            .iter()
            .map(|k| {
                k.as_ref()
                    .map(|a| a.handle.0)
                    .unwrap_or(std::ptr::null_mut())
            })
            .collect();
        let kv_values_ptrs: Vec<*mut sys::mlx_array> = kv_values
            .iter()
            .map(|v| {
                v.as_ref()
                    .map(|a| a.handle.0)
                    .unwrap_or(std::ptr::null_mut())
            })
            .collect();

        // Output buffers
        let mut out_logits: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_kv_keys: Vec<*mut sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers as usize];
        let mut out_kv_values: Vec<*mut sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers as usize];
        let mut out_cache_idx: i32 = 0;

        let config = &self.config;

        unsafe {
            sys::mlx_paddleocr_vl_forward_step_batched(
                input_embeds.handle.0,
                all_weight_ptrs.as_ptr(),
                num_layers,
                final_norm_ptr,
                lm_head_ptr,
                inv_freq_ptr,
                position_ids.handle.0,
                mrope_section.as_ptr(),
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps as f32,
                left_padding.handle.0,
                kv_keys_ptrs.as_ptr(),
                kv_values_ptrs.as_ptr(),
                cache_idx,
                &mut out_logits,
                out_kv_keys.as_mut_ptr(),
                out_kv_values.as_mut_ptr(),
                &mut out_cache_idx,
            );
        }

        // Convert output pointers to MxArrays
        let mut new_kv_keys: Vec<Option<MxArray>> = Vec::with_capacity(num_layers as usize);
        let mut new_kv_values: Vec<Option<MxArray>> = Vec::with_capacity(num_layers as usize);
        for i in 0..num_layers as usize {
            new_kv_keys.push(if !out_kv_keys[i].is_null() {
                Some(MxArray::from_handle(out_kv_keys[i], "batched_kv_key")?)
            } else {
                None
            });
            new_kv_values.push(if !out_kv_values[i].is_null() {
                Some(MxArray::from_handle(out_kv_values[i], "batched_kv_value")?)
            } else {
                None
            });
        }

        let logits = MxArray::from_handle(out_logits, "paddleocr_vl_forward_batched")?;
        Ok((logits, new_kv_keys, new_kv_values, out_cache_idx))
    }

    /// Get number of layers (needed for batch KV cache setup)
    pub fn num_layers_usize(&self) -> usize {
        self.layers.len()
    }

    /// Get the embedding layer (for external use during batch generation)
    pub fn get_embedding_layer(&self) -> &Embedding {
        &self.embed_tokens
    }
}

impl Clone for ERNIELanguageModel {
    fn clone(&self) -> Self {
        Self {
            embed_tokens: self.embed_tokens.clone(),
            layers: self.layers.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
            rotary_emb: self.rotary_emb.clone(),
            config: self.config.clone(),
            kv_caches: None, // Don't clone caches - they should be initialized fresh
            position_ids: None, // Don't clone position state
            rope_deltas: None,
            fused_kv_keys: Vec::new(),
            fused_kv_values: Vec::new(),
            fused_cache_idx: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    // Helper to create a test TextConfig with smaller dimensions
    fn test_text_config() -> TextConfig {
        TextConfig {
            model_type: "paddleocr_vl".to_string(),
            hidden_size: 256,
            num_hidden_layers: 2,
            intermediate_size: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            vocab_size: 1000,
            max_position_embeddings: 1024,
            rope_theta: 500000.0,
            rope_traditional: false,
            use_bias: false,
            head_dim: 64,
            mrope_section: vec![8, 12, 12],
        }
    }

    #[test]
    fn test_mrope_section_sum() {
        let config = TextConfig::default();
        // [16, 24, 24] * 2 = [32, 48, 48] -> total = 128 = head_dim
        let total: i32 = config.mrope_section.iter().map(|&x| x * 2).sum();
        assert_eq!(total, config.head_dim);
    }

    #[test]
    fn test_mrope_forward_output_shapes() {
        // Test MultimodalRoPE forward pass produces correct shapes
        let mrope = MultimodalRoPE::new(128, 131072, Some(500000.0), vec![16, 24, 24]).unwrap();

        let x = MxArray::zeros(&[1, 4, 128], Some(DType::Float32)).unwrap();
        let position_ids = MxArray::zeros(&[3, 1, 4], Some(DType::Float32)).unwrap();

        let (cos, sin) = mrope.forward(&x, &position_ids).unwrap();

        let cos_shape: Vec<i64> = cos.shape().unwrap().as_ref().to_vec();
        let sin_shape: Vec<i64> = sin.shape().unwrap().as_ref().to_vec();
        assert_eq!(cos_shape, vec![3, 1, 4, 128]);
        assert_eq!(sin_shape, vec![3, 1, 4, 128]);
    }

    #[test]
    fn test_mrope_invalid_section_length() {
        // mRoPE section must have exactly 3 elements
        let result = MultimodalRoPE::new(128, 131072, Some(500000.0), vec![16, 24]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mrope_getter() {
        // Test mRoPE section getter
        let mrope = MultimodalRoPE::new(128, 131072, Some(500000.0), vec![16, 24, 24]).unwrap();
        assert_eq!(mrope.mrope_section(), vec![16, 24, 24]);
    }

    #[test]
    fn test_apply_mrope_output_shapes() {
        // Test apply_multimodal_rotary_pos_emb preserves tensor shapes
        let q = MxArray::random_uniform(&[1, 4, 2, 32], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let k = MxArray::random_uniform(&[1, 4, 2, 32], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let cos = MxArray::ones(&[3, 1, 2, 32], Some(DType::Float32)).unwrap();
        let sin = MxArray::zeros(&[3, 1, 2, 32], Some(DType::Float32)).unwrap();

        let (q_out, k_out) =
            apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin, vec![4, 6, 6]).unwrap();

        let q_out_shape: Vec<i64> = q_out.shape().unwrap().as_ref().to_vec();
        let k_out_shape: Vec<i64> = k_out.shape().unwrap().as_ref().to_vec();
        assert_eq!(q_out_shape, vec![1, 4, 2, 32]);
        assert_eq!(k_out_shape, vec![1, 4, 2, 32]);
    }

    #[test]
    fn test_ernie_language_model_creation() {
        // Test creating ERNIELanguageModel
        let config = test_text_config();

        let embed_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let lm_head_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();

        let lm =
            ERNIELanguageModel::new(config, &embed_weight, &norm_weight, &lm_head_weight).unwrap();

        assert_eq!(lm.num_layers(), 0); // No layers added yet
    }

    #[test]
    fn test_ernie_language_model_add_layers() {
        // Test adding decoder layers to ERNIELanguageModel
        let config = test_text_config();

        let embed_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let lm_head_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();

        let mut lm =
            ERNIELanguageModel::new(config.clone(), &embed_weight, &norm_weight, &lm_head_weight)
                .unwrap();

        // Create decoder layer weights
        // Q: [num_heads * head_dim, hidden_size] = [4 * 64, 256] = [256, 256]
        // K/V: [num_kv_heads * head_dim, hidden_size] = [2 * 64, 256] = [128, 256]
        let q_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let k_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let v_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let o_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let gate_weight =
            MxArray::random_uniform(&[512, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let up_weight =
            MxArray::random_uniform(&[512, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let down_weight =
            MxArray::random_uniform(&[256, 512], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let input_norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let post_attn_norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();

        let layer = PaddleOCRDecoderLayer::new(
            config,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            &gate_weight,
            &up_weight,
            &down_weight,
            &input_norm_weight,
            &post_attn_norm_weight,
        )
        .unwrap();

        lm.add_layer(&layer);
        assert_eq!(lm.num_layers(), 1);

        // Add another layer
        lm.add_layer(&layer);
        assert_eq!(lm.num_layers(), 2);
    }

    #[test]
    fn test_ernie_language_model_kv_cache_management() {
        // Test KV cache initialization and reset
        let config = test_text_config();

        let embed_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let lm_head_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();

        let mut lm =
            ERNIELanguageModel::new(config, &embed_weight, &norm_weight, &lm_head_weight).unwrap();

        // Initialize caches
        lm.init_kv_caches();
        assert_eq!(lm.get_cache_offset(), 0);

        // Reset caches
        lm.reset_kv_caches();
        assert_eq!(lm.get_cache_offset(), 0);
    }

    #[test]
    fn test_ernie_language_model_get_embeddings() {
        // Test getting token embeddings
        let config = test_text_config();

        let embed_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let lm_head_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();

        let lm =
            ERNIELanguageModel::new(config, &embed_weight, &norm_weight, &lm_head_weight).unwrap();

        // Create input token IDs [batch=1, seq_len=4]
        let input_ids = MxArray::from_int32(&[1, 2, 3, 4], &[1, 4]).unwrap();

        let embeddings = lm.get_embeddings(&input_ids).unwrap();
        let shape: Vec<i64> = embeddings.shape().unwrap().as_ref().to_vec();

        // Output shape should be [1, 4, 256] (batch, seq_len, hidden_size)
        assert_eq!(shape, vec![1, 4, 256]);
    }

    #[test]
    fn test_ernie_language_model_position_state() {
        // Test position state management for multimodal generation
        let config = test_text_config();

        let embed_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let lm_head_weight =
            MxArray::random_uniform(&[1000, 256], 0.0, 1.0, Some(DType::Float32)).unwrap();

        let mut lm =
            ERNIELanguageModel::new(config, &embed_weight, &norm_weight, &lm_head_weight).unwrap();

        // Initially no rope_deltas
        assert!(lm.get_rope_deltas().is_none());

        // Set position state
        let pos_ids = MxArray::zeros(&[3, 1, 4], Some(DType::Float32)).unwrap();
        lm.set_position_state(pos_ids, 42);
        assert_eq!(lm.get_rope_deltas(), Some(42));

        // Reset position state
        lm.reset_position_state();
        assert!(lm.get_rope_deltas().is_none());
    }

    #[test]
    fn test_decoder_layer_creation() {
        // Test creating PaddleOCRDecoderLayer
        let config = test_text_config();

        let q_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let k_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let v_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let o_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let gate_weight =
            MxArray::random_uniform(&[512, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let up_weight =
            MxArray::random_uniform(&[512, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let down_weight =
            MxArray::random_uniform(&[256, 512], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let input_norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();
        let post_attn_norm_weight = MxArray::ones(&[256], Some(DType::Float32)).unwrap();

        let layer = PaddleOCRDecoderLayer::new(
            config,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            &gate_weight,
            &up_weight,
            &down_weight,
            &input_norm_weight,
            &post_attn_norm_weight,
        );

        assert!(layer.is_ok());
    }

    #[test]
    fn test_attention_creation() {
        // Test creating PaddleOCRAttention
        let config = test_text_config();

        let q_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let k_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let v_weight =
            MxArray::random_uniform(&[128, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();
        let o_weight =
            MxArray::random_uniform(&[256, 256], 0.0, 0.1, Some(DType::Float32)).unwrap();

        let attn = PaddleOCRAttention::new(config, &q_weight, &k_weight, &v_weight, &o_weight);

        assert!(attn.is_ok());
    }
}
