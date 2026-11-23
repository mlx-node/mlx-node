/**
 * ERNIE Language Model for PaddleOCR-VL
 *
 * The language model component uses multimodal RoPE (mRoPE) which splits
 * the head dimension into sections for temporal, height, and width positions.
 */
use crate::array::MxArray;
use crate::models::paddleocr_vl::config::TextConfig;
use crate::nn::{Embedding, Linear, RMSNorm};
pub use crate::vision::rope_vision::{
    MultimodalRoPE, apply_interleaved_rotary, apply_multimodal_rotary_pos_emb_interleaved,
    select_interleaved_cos_sin,
};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::sync::Arc;

/// PaddleOCR Attention with mRoPE (internal)
///
/// Note: This is an internal implementation detail used by PaddleOCRDecoderLayer.
/// Not exposed to TypeScript - users interact with high-level VLModel API.
pub struct PaddleOCRAttention {
    q_proj: Arc<Linear>,
    k_proj: Arc<Linear>,
    v_proj: Arc<Linear>,
    o_proj: Arc<Linear>,
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
        _config: TextConfig,
        q_weight: &MxArray,
        k_weight: &MxArray,
        v_weight: &MxArray,
        o_weight: &MxArray,
    ) -> Result<Self> {
        let q_proj = Linear::from_weights(q_weight, None)?;
        let k_proj = Linear::from_weights(k_weight, None)?;
        let v_proj = Linear::from_weights(v_weight, None)?;
        let o_proj = Linear::from_weights(o_weight, None)?;

        Ok(Self {
            q_proj: Arc::new(q_proj),
            k_proj: Arc::new(k_proj),
            v_proj: Arc::new(v_proj),
            o_proj: Arc::new(o_proj),
        })
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
