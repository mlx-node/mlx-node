/**
 * PaddleOCR-VL Full Model
 *
 * Combines vision encoder and language model for vision-language tasks.
 */
use crate::array::{MxArray, clear_cache};
use crate::models::paddleocr_vl::chat::{ChatRole, VLMChatConfig, VLMChatMessage, VLMChatResult};
use crate::models::paddleocr_vl::config::{ModelConfig, TextConfig, VisionConfig};
use crate::models::paddleocr_vl::language::{ERNIELanguageModel, PaddleOCRDecoderLayer};
use crate::models::paddleocr_vl::persistence::load_paddleocr_vl_weights;
use crate::models::paddleocr_vl::processing::{ImageProcessor, ImageProcessorConfig};
use crate::models::paddleocr_vl::vision::PaddleOCRVisionModel;
use crate::models::qwen3::{GenerationConfig, GenerationResult};
use crate::nn::LayerNorm;
use crate::sampling::{SamplingConfig, apply_repetition_penalty, sample, sample_and_logprobs};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;
use crate::utils::safetensors::SafeTensorsFile;
use crate::vision::encoder::{VisionAttention, VisionEncoderLayer, VisionMLP};
use napi::{Env, Status, bindgen_prelude::*, tokio};
use napi_derive::napi;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};

/// Vision-Language Model
///
/// A generic VLM for OCR and document understanding tasks.
/// Currently supports PaddleOCR-VL architecture (vision encoder + ERNIE language model).
#[napi(js_name = "VLModel")]
pub struct VLModel {
    config: ModelConfig,
    visual: Option<Arc<PaddleOCRVisionModel>>,
    /// Language model wrapped in RwLock for mutable KV cache access
    language_model: Option<Arc<RwLock<ERNIELanguageModel>>>,
    /// Tokenizer for text encoding/decoding
    tokenizer: Option<Arc<Qwen3Tokenizer>>,
}

#[napi]
impl VLModel {
    /// Create a new PaddleOCR-VL model
    #[napi(constructor)]
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            visual: None,
            language_model: None,
            tokenizer: None,
        }
    }

    /// Set the vision model (internal - used by VLModel::load())
    pub fn set_visual(&mut self, visual: &PaddleOCRVisionModel) {
        self.visual = Some(Arc::new(visual.clone()));
    }

    /// Set the language model (internal - used by VLModel::load())
    pub fn set_language_model(&mut self, lm: &ERNIELanguageModel) {
        self.language_model = Some(Arc::new(RwLock::new(lm.clone())));
    }

    /// Set the tokenizer
    #[napi]
    pub fn set_tokenizer(&mut self, tokenizer: &Qwen3Tokenizer) {
        self.tokenizer = Some(Arc::new(tokenizer.clone()));
    }

    /// Check if tokenizer is available
    #[napi(getter)]
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Chat with the VLM model
    ///
    /// High-level API for conversational interaction with images.
    ///
    /// # Arguments
    /// * `messages` - Chat messages (role + content)
    /// * `config` - Chat configuration (including image_paths for automatic processing)
    ///
    /// # Returns
    /// * VLMChatResult with generated text
    ///
    /// # Example
    /// ```typescript
    /// const result = model.chat(
    ///   [{ role: 'user', content: 'Describe this image.' }],
    ///   { imagePaths: ['./photo.jpg'], maxNewTokens: 256 }
    /// );
    /// ```
    #[napi]
    pub fn chat(
        &self,
        messages: Vec<VLMChatMessage>,
        config: Option<VLMChatConfig>,
    ) -> Result<VLMChatResult> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::new(
                Status::GenericFailure,
                "Tokenizer not set. Use set_tokenizer() first.",
            )
        })?;

        // Merge passed config with defaults - any None fields use the default values
        let default_config = VLMChatConfig::default();
        let config = match config {
            Some(c) => VLMChatConfig {
                image_paths: c.image_paths,
                max_new_tokens: c.max_new_tokens.or(default_config.max_new_tokens),
                temperature: c.temperature.or(default_config.temperature),
                top_k: c.top_k.or(default_config.top_k),
                top_p: c.top_p.or(default_config.top_p),
                repetition_penalty: c.repetition_penalty.or(default_config.repetition_penalty),
                return_logprobs: c.return_logprobs.or(default_config.return_logprobs),
            },
            None => default_config,
        };

        // Process images if image_paths provided
        let (pixel_values, grid_thw) = if let Some(paths) = &config.image_paths {
            if paths.is_empty() {
                (None, None)
            } else {
                // Process ALL images using batch processing
                let processor_config = ImageProcessorConfig {
                    patch_size: self.config.vision_config.patch_size,
                    merge_size: self.config.vision_config.spatial_merge_size,
                    ..ImageProcessorConfig::default()
                };
                let processor = ImageProcessor::new(Some(processor_config));
                let processed = processor.process_files(paths.clone())?;

                // Add batch dimension: [total_patches, C, H, W] -> [1, total_patches, C, H, W]
                let pv = processed.pixel_values();
                let pv_shape = pv.shape()?;
                let new_shape = BigInt64Array::from(vec![
                    1i64,
                    pv_shape[0],
                    pv_shape[1],
                    pv_shape[2],
                    pv_shape[3],
                ]);
                let pixel_values = pv.reshape(&new_shape)?;

                // grid_thw already [num_images, 3] from process_files
                let grid_thw = processed.grid_thw();

                (Some(pixel_values), Some(grid_thw))
            }
        } else {
            (None, None)
        };

        // Count image tokens if image is provided
        let num_image_tokens = if let Some(ref grid) = grid_thw {
            grid.eval();
            let grid_data = grid.to_int32()?;
            // grid_thw has shape [num_images, 3] with [t, h, w] per image
            // Total tokens = sum of (t * h * w / spatial_merge_size^2) for each image
            let spatial_merge_size = self.config.vision_config.spatial_merge_size;
            let merge_factor = spatial_merge_size * spatial_merge_size;
            let mut total = 0i32;
            for i in 0..(grid_data.len() / 3) {
                let t = grid_data[i * 3];
                let h = grid_data[i * 3 + 1];
                let w = grid_data[i * 3 + 2];
                total += (t * h * w) / merge_factor;
            }
            Some(total as usize)
        } else {
            None
        };

        // Format messages with image placeholders
        let formatted =
            crate::models::paddleocr_vl::chat::format_vlm_chat(&messages, num_image_tokens);

        // Encode the text (use sync method for internal API)
        let token_ids = tokenizer.encode_sync(&formatted, None)?;
        let input_ids = MxArray::from_uint32(&token_ids, &[1, token_ids.len() as i64])?;

        // Build generation config
        let gen_config = GenerationConfig {
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: None,
            repetition_penalty: config.repetition_penalty,
            repetition_context_size: None,
            max_consecutive_tokens: None,
            max_ngram_repeats: None,
            ngram_size: None,
            eos_token_id: Some(self.config.eos_token_id),
            return_logprobs: config.return_logprobs,
            prefill_step_size: None,
            kv_cache_bits: None,
            kv_cache_group_size: None,
            num_draft_tokens: None,
        };

        // Generate
        let result = self.generate(
            &input_ids,
            pixel_values.as_ref(),
            grid_thw.as_ref(),
            Some(gen_config),
        )?;

        // Decode the generated tokens (use sync method for internal API)
        result.tokens.eval();
        let tokens_vec = result.tokens.to_uint32()?;
        let text = tokenizer.decode_sync(&tokens_vec, true)?;

        // Clean up the text (remove special tokens)
        // Note: Table tokens (<nl>, <lcel>, <ecel>, <fcel>, etc.) are preserved
        // for structured parsing. Use VlmOutputParser to convert to markdown/etc.
        let text = text.replace("<|im_end|>", "").trim().to_string();

        Ok(VLMChatResult {
            text,
            tokens: result.tokens,
            logprobs: result.logprobs,
            finish_reason: result.finish_reason,
            num_tokens: result.num_tokens,
        })
    }

    /// Simple OCR: extract text from an image file
    ///
    /// Convenience method that processes an image and extracts all text.
    ///
    /// # Arguments
    /// * `image_path` - Path to the image file
    /// * `prompt` - Optional custom prompt (default: "Extract all text from this image.")
    ///
    /// # Returns
    /// * Extracted text as a string
    ///
    /// # Example
    /// ```typescript
    /// const text = await model.ocr('./receipt.jpg');
    /// console.log(text);
    /// ```
    #[napi]
    pub fn ocr(&self, image_path: String, prompt: Option<String>) -> Result<String> {
        let prompt = prompt.unwrap_or_else(|| "Extract all text from this image.".to_string());

        let messages = vec![VLMChatMessage {
            role: ChatRole::User,
            content: prompt,
        }];

        let config = VLMChatConfig {
            image_paths: Some(vec![image_path]),
            ..Default::default()
        };

        let result = self.chat(messages, Some(config))?;

        // Clean up common LaTeX wrappers from OCR output
        let text = result.text;
        let text = text
            .strip_prefix("\\(\\text{")
            .and_then(|s| s.strip_suffix("}\\)"))
            .map(|s| s.to_string())
            .unwrap_or(text);

        Ok(text)
    }

    /// Get input embeddings with vision features merged
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `pixel_values` - Optional image patches [batch, seq, channels, patch_h, patch_w]
    /// * `image_grid_thw` - Optional grid dimensions [num_images, 3]
    ///
    /// # Returns
    /// * Input embeddings with vision features inserted at image token positions
    #[napi]
    pub fn get_input_embeddings(
        &self,
        input_ids: &MxArray,
        pixel_values: Option<&MxArray>,
        image_grid_thw: Option<&MxArray>,
    ) -> Result<MxArray> {
        let lm = self
            .language_model
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "Language model not set"))?;
        let lm_guard = lm.read().map_err(|_| {
            Error::new(
                Status::GenericFailure,
                "Failed to acquire language model read lock",
            )
        })?;

        // If no images, just get text embeddings
        if pixel_values.is_none() {
            return lm_guard.get_embeddings(input_ids);
        }

        let pixel_values = pixel_values.unwrap();
        let grid_thw = image_grid_thw.ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "image_grid_thw required when pixel_values provided",
            )
        })?;

        let visual = self
            .visual
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "Vision model not set"))?;

        // Get text embeddings
        let inputs_embeds = lm_guard.get_embeddings(input_ids)?;

        // Get vision features
        let hidden_states = visual.forward(pixel_values, grid_thw)?;

        // Cast vision features to match embedding dtype to prevent float32 promotion
        let embed_dtype = inputs_embeds.dtype()?;
        let hidden_states = if hidden_states.dtype()? != embed_dtype {
            hidden_states.astype(embed_dtype)?
        } else {
            hidden_states
        };

        // Merge vision features into text embeddings at image token positions
        self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            &hidden_states,
            &inputs_embeds,
            input_ids,
        )
    }

    /// Compute position IDs for multimodal RoPE
    ///
    /// This function computes proper 3D position IDs for image tokens:
    /// - Text tokens get sequential positions [0, 1, 2, ...]
    /// - Image tokens get 2D spatial positions based on grid_thw
    ///
    /// Returns (position_ids, rope_deltas) where:
    /// - position_ids: [3, batch, seq_len] position indices for t, h, w
    /// - rope_deltas: offset to add during decode phase
    fn get_rope_index(
        &self,
        input_ids: &MxArray,
        image_grid_thw: Option<&MxArray>,
    ) -> Result<(MxArray, i64)> {
        let shape = input_ids.shape()?;
        let batch_size = shape[0];
        let seq_len = shape[1];

        // If no images, use simple sequential positions
        if image_grid_thw.is_none() {
            let pos = MxArray::arange(0.0, seq_len as f64, Some(1.0), None)?;
            let pos = pos.reshape(&[1, 1, seq_len])?;
            let position_ids = MxArray::tile(&pos, &[3, batch_size as i32, 1])?;
            return Ok((position_ids, 0));
        }

        let grid_thw = image_grid_thw.unwrap();
        let spatial_merge_size = self.config.vision_config.spatial_merge_size;
        let image_token_id = self.config.image_token_id;

        // Get input IDs and grid data
        let input_ids_data = input_ids.to_int32()?;
        grid_thw.eval();
        let grid_data = grid_thw.to_int32()?;

        // Process batch (currently supports batch_size=1)
        // For each batch, we compute position IDs with proper 2D spatial encoding for images
        let mut all_position_ids: Vec<Vec<i64>> = vec![Vec::new(); 3]; // [t, h, w] components

        for batch_idx in 0..batch_size as usize {
            let start = batch_idx * seq_len as usize;
            let end = start + seq_len as usize;
            let batch_tokens: Vec<i32> = input_ids_data[start..end].to_vec();

            // Find positions of image tokens
            let mut image_positions: Vec<usize> = Vec::new();
            for (i, &token) in batch_tokens.iter().enumerate() {
                if token == image_token_id {
                    image_positions.push(i);
                }
            }

            // If no image tokens, use sequential positions
            if image_positions.is_empty() {
                for i in 0..seq_len {
                    all_position_ids[0].push(i);
                    all_position_ids[1].push(i);
                    all_position_ids[2].push(i);
                }
                continue;
            }

            // Get grid dimensions for ALL images
            let num_images = grid_data.len() / 3;
            if num_images == 0 || grid_data.len() % 3 != 0 {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "grid_data must have 3N elements for N images, got {} elements. \
                        Ensure image_grid_thw is properly set when image tokens are present in input.",
                        grid_data.len()
                    ),
                ));
            }

            // Calculate token info for each image
            let mut total_expected_tokens = 0usize;
            let mut image_token_info: Vec<(i64, i64, i64, usize)> = Vec::new();

            for img_idx in 0..num_images {
                let t = grid_data[img_idx * 3] as i64;
                let h = grid_data[img_idx * 3 + 1] as i64;
                let w = grid_data[img_idx * 3 + 2] as i64;

                let llm_grid_t = t;
                let llm_grid_h = h / spatial_merge_size as i64;
                let llm_grid_w = w / spatial_merge_size as i64;
                let num_tokens = (llm_grid_t * llm_grid_h * llm_grid_w) as usize;

                image_token_info.push((llm_grid_t, llm_grid_h, llm_grid_w, num_tokens));
                total_expected_tokens += num_tokens;
            }

            // Validate total image token count matches expected from grid dimensions
            if total_expected_tokens != image_positions.len() {
                return Err(Error::new(
                    Status::GenericFailure,
                    format!(
                        "Image token count mismatch: expected {} tokens from {} images, \
                        but found {} image tokens in prompt. \
                        This likely indicates a bug in prompt formatting. Check that: \
                        1) The image placeholder tokens match the expected count from grid_thw \
                        2) spatial_merge_size ({}) in config matches the vision encoder output \
                        3) grid_thw dimensions are computed correctly for the input images",
                        total_expected_tokens,
                        num_images,
                        image_positions.len(),
                        spatial_merge_size
                    ),
                ));
            }

            // Build position IDs
            let image_start = image_positions[0];
            let image_end = image_positions[image_positions.len() - 1] + 1;

            // Text tokens before images: sequential positions
            for i in 0..image_start {
                all_position_ids[0].push(i as i64);
                all_position_ids[1].push(i as i64);
                all_position_ids[2].push(i as i64);
            }

            // Each image's 2D spatial positions
            let mut current_pos = image_start as i64;
            let mut max_pos = image_start as i64;

            for (llm_grid_t, llm_grid_h, llm_grid_w, _) in &image_token_info {
                for t_idx in 0..*llm_grid_t {
                    for h_idx in 0..*llm_grid_h {
                        for w_idx in 0..*llm_grid_w {
                            all_position_ids[0].push(current_pos + t_idx);
                            all_position_ids[1].push(current_pos + h_idx);
                            all_position_ids[2].push(current_pos + w_idx);
                        }
                    }
                }
                let img_max = current_pos
                    + std::cmp::max(
                        *llm_grid_t - 1,
                        std::cmp::max(*llm_grid_h - 1, *llm_grid_w - 1),
                    );
                max_pos = std::cmp::max(max_pos, img_max);
                current_pos = img_max + 1;
            }

            // Text tokens after images: continue from max position
            let next_pos = max_pos + 1;
            for i in image_end..seq_len as usize {
                let pos = next_pos + (i - image_end) as i64;
                all_position_ids[0].push(pos);
                all_position_ids[1].push(pos);
                all_position_ids[2].push(pos);
            }
        }

        // Convert to MxArray [3, batch, seq_len]
        let total_len = all_position_ids[0].len();
        let expected_len = (3 * batch_size * seq_len) as usize;
        if total_len * 3 != expected_len {
            return Err(Error::new(
                Status::GenericFailure,
                format!(
                    "Position ID length mismatch: got {} * 3, expected {}",
                    total_len, expected_len
                ),
            ));
        }

        // Stack the three components
        let t_positions: Vec<i32> = all_position_ids[0].iter().map(|&x| x as i32).collect();
        let h_positions: Vec<i32> = all_position_ids[1].iter().map(|&x| x as i32).collect();
        let w_positions: Vec<i32> = all_position_ids[2].iter().map(|&x| x as i32).collect();

        let t_arr = MxArray::from_int32(&t_positions, &[batch_size, seq_len])?;
        let h_arr = MxArray::from_int32(&h_positions, &[batch_size, seq_len])?;
        let w_arr = MxArray::from_int32(&w_positions, &[batch_size, seq_len])?;

        let position_ids = MxArray::stack(vec![&t_arr, &h_arr, &w_arr], Some(0))?;

        // Compute rope_deltas: max position - seq_len (for decode phase offset)
        let max_position = *all_position_ids[0].iter().max().unwrap_or(&0);
        let rope_deltas = max_position + 1 - seq_len;

        Ok((position_ids, rope_deltas))
    }

    /// Merge image features into input embeddings at image token positions
    fn merge_input_ids_with_image_features(
        &self,
        image_token_id: i32,
        image_features: &MxArray,
        inputs_embeds: &MxArray,
        input_ids: &MxArray,
    ) -> Result<MxArray> {
        let input_shape = input_ids.shape()?;
        let batch_size = input_shape[0];
        let _seq_len = input_shape[1];

        // Create image token mask
        let image_token = MxArray::scalar_int(image_token_id)?;
        let image_positions = input_ids.equal(&image_token)?;

        let inputs_embeds_shape = inputs_embeds.shape()?;
        let hidden_dim = inputs_embeds_shape[2];

        let mut batch_outputs: Vec<MxArray> = Vec::new();
        let mut feature_start_idx = 0i64;

        for batch_idx in 0..batch_size {
            // Get mask for this batch item
            let batch_mask = image_positions.slice_axis(0, batch_idx, batch_idx + 1)?;
            let batch_mask = batch_mask.squeeze(Some(&[0]))?;

            // Count image tokens in this batch
            let mask_sum = batch_mask.sum(None, None)?;
            let num_positions = mask_sum.to_int32()?[0] as i64;

            if num_positions > 0 {
                // Extract features for this batch
                let batch_features = image_features.slice_axis(
                    0,
                    feature_start_idx,
                    feature_start_idx + num_positions,
                )?;

                // Get embeddings for this batch item
                let batch_embeds = inputs_embeds.slice_axis(0, batch_idx, batch_idx + 1)?;
                let batch_embeds = batch_embeds.squeeze(Some(&[0]))?;

                // Create cumsum for feature indexing
                let mask_int = batch_mask.astype(crate::array::DType::Int32)?;
                let cumsum = mask_int.cumsum(0)?;

                // Build feature indices: where mask is true, use cumsum-1; else use 0
                let ones = MxArray::scalar_int(1)?;
                let feature_indices = cumsum.sub(&ones)?;
                let zeros =
                    MxArray::zeros(&feature_indices.shape()?, Some(crate::array::DType::Int32))?;
                let feature_indices = batch_mask.where_(&feature_indices, &zeros)?;

                // Gather features using indices
                let gathered_features = batch_features.take(&feature_indices, 0)?;

                // Expand mask for broadcasting
                let mask_expanded = batch_mask.reshape(&[-1, 1])?;
                let mask_expanded =
                    MxArray::broadcast_to(&mask_expanded, &[batch_mask.shape()?[0], hidden_dim])?;

                // Combine: use gathered_features where mask is true, else use original embeds
                let batch_output = mask_expanded.where_(&gathered_features, &batch_embeds)?;

                batch_outputs.push(batch_output);
                feature_start_idx += num_positions;
            } else {
                // No image tokens in this batch item
                let batch_embeds = inputs_embeds.slice_axis(0, batch_idx, batch_idx + 1)?;
                batch_outputs.push(batch_embeds.squeeze(Some(&[0]))?);
            }
        }

        // Stack all batch outputs
        let refs: Vec<&MxArray> = batch_outputs.iter().collect();
        MxArray::stack(refs, Some(0))
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `pixel_values` - Optional image patches
    /// * `image_grid_thw` - Optional grid dimensions
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// * Logits [batch, seq_len, vocab_size]
    #[napi]
    pub fn forward(
        &self,
        input_ids: &MxArray,
        pixel_values: Option<&MxArray>,
        image_grid_thw: Option<&MxArray>,
        mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        let lm = self
            .language_model
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "Language model not set"))?;
        let lm_guard = lm.read().map_err(|_| {
            Error::new(
                Status::GenericFailure,
                "Failed to acquire language model read lock",
            )
        })?;

        // Get merged embeddings
        let inputs_embeds = self.get_input_embeddings(input_ids, pixel_values, image_grid_thw)?;

        // Forward through language model
        lm_guard.forward(input_ids, Some(&inputs_embeds), mask, None)
    }

    /// Generate text tokens given input tokens and optional image
    ///
    /// Uses KV caching for efficient generation - each step only processes the
    /// new token(s) while reusing cached key-value states from previous tokens.
    /// Vision features are computed once at the start and cached.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [1, seq_len]
    /// * `pixel_values` - Optional image patches [1, num_patches, C, H, W]
    /// * `image_grid_thw` - Optional grid dimensions [1, 3]
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// * GenerationResult with tokens, logprobs, and finish reason
    #[napi]
    pub fn generate(
        &self,
        input_ids: &MxArray,
        pixel_values: Option<&MxArray>,
        image_grid_thw: Option<&MxArray>,
        config: Option<GenerationConfig>,
    ) -> Result<GenerationResult> {
        let config = config.unwrap_or_default();

        // Extract config with defaults - aligned with mlx-vlm generate_step defaults
        let max_new_tokens = config.max_new_tokens.unwrap_or(256); // mlx-vlm DEFAULT_MAX_TOKENS
        let temperature = config.temperature.unwrap_or(0.0); // mlx-vlm: greedy by default
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(20); // Match Python default
        let eos_token_id = config.eos_token_id.unwrap_or(self.config.eos_token_id);
        let return_logprobs = config.return_logprobs.unwrap_or(false); // Default false for VLM (OCR use case)

        debug!(
            "Starting VLM generation with KV cache: max_tokens={}, temp={}, top_k={}, top_p={}, rep_penalty={}",
            max_new_tokens, temperature, top_k, top_p, repetition_penalty
        );

        // Create dedicated generation stream for GPU-CPU pipelining.
        // Forward pass runs on this stream, async_eval queues work, while CPU
        // extracts the previous token. Gives ~25% decode speedup.
        let generation_stream = Stream::new(DeviceType::Gpu);

        // Prepare sampling config
        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

        // Get language model with write access for KV cache
        let lm = self
            .language_model
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "Language model not set"))?;
        let mut lm_guard = lm.write().map_err(|_| {
            Error::new(
                Status::GenericFailure,
                "Failed to acquire language model write lock",
            )
        })?;

        // Initialize fused KV caches for this generation (C++ forward pass)
        lm_guard.init_fused_kv_caches();

        // Reset position state for new generation (critical for multimodal)
        lm_guard.reset_position_state();

        // === STEP 1: Compute vision features ONCE ===
        // Run vision encoding inside stream context for GPU-CPU overlap
        let vision_features = {
            let _stream_ctx = StreamContext::new(generation_stream);
            if let (Some(pv), Some(grid)) = (pixel_values, image_grid_thw) {
                let visual = self
                    .visual
                    .as_ref()
                    .ok_or_else(|| Error::new(Status::GenericFailure, "Vision model not set"))?;
                Some(visual.forward(pv, grid)?)
            } else {
                None
            }
        };

        // === STEP 2: Compute proper position IDs for mRoPE ===
        // This is critical - image tokens need 2D spatial positions, not sequential
        let (position_ids, rope_deltas) = self.get_rope_index(input_ids, image_grid_thw)?;
        debug!("Computed position_ids with rope_deltas: {}", rope_deltas);

        // Store position state for decode phase (critical for proper multimodal attention)
        lm_guard.set_position_state(position_ids.clone(), rope_deltas);

        // === STEP 3: Prefill - process prompt with vision features ===
        // Get text embeddings and merge with vision features (in stream context)
        let inputs_embeds = {
            let _stream_ctx = StreamContext::new(generation_stream);
            let embeds = lm_guard.get_embeddings(input_ids)?;
            if let Some(ref vf) = vision_features {
                // Cast vision features to match language model dtype (e.g. bfloat16).
                // Vision encoder outputs float32 (from float32 pixel inputs), but the
                // language model operates in bfloat16. Without this cast, the merged
                // embeddings become float32, causing the KV cache and all decode steps
                // to run in float32 (2x memory bandwidth → ~3-4x slower decode).
                let embed_dtype = embeds.dtype()?;
                let vf_cast = if vf.dtype()? != embed_dtype {
                    vf.astype(embed_dtype)?
                } else {
                    vf.clone()
                };
                self.merge_input_ids_with_image_features(
                    self.config.image_token_id,
                    &vf_cast,
                    &embeds,
                    input_ids,
                )?
            } else {
                embeds
            }
        };

        // Chunked prefill: process long sequences in chunks to bound memory usage
        // and keep computation graphs manageable. Matching Python mlx-vlm's approach.
        let prefill_step_size: i64 = 2048;
        let seq_len = inputs_embeds.shape_at(1)?;

        let mut last_logits = if seq_len > prefill_step_size {
            // Chunked prefill for long sequences (common with images)
            let mut offset: i64 = 0;
            let mut chunk_logits = None;

            while offset < seq_len {
                let chunk_end = std::cmp::min(offset + prefill_step_size, seq_len);
                // For all chunks except the last, process up to chunk_end
                // For the last chunk, process the rest
                let n_to_process = if chunk_end < seq_len {
                    // Not the last chunk: process prefill_step_size tokens
                    chunk_end - offset
                } else {
                    // Last chunk: process remaining
                    seq_len - offset
                };

                let chunk_embeds = inputs_embeds.slice_axis(1, offset, offset + n_to_process)?;
                let chunk_pos = position_ids.slice_axis(2, offset, offset + n_to_process)?;

                {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    chunk_logits = Some(lm_guard.forward_fused(&chunk_embeds, &chunk_pos)?);
                }

                // Eval KV caches between chunks to materialize results and bound graph size
                lm_guard.eval_fused_kv_caches();
                clear_cache();

                offset += n_to_process;
            }

            // Extract last position logits from the final chunk
            let logits = chunk_logits.unwrap();
            let last_seq = logits.shape_at(1)?;
            logits
                .slice_axis(1, last_seq - 1, last_seq)?
                .squeeze(Some(&[0, 1]))?
        } else {
            // Short sequence: single forward pass (text-only or short prompts)
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                lm_guard.forward_fused(&inputs_embeds, &position_ids)?
            };
            // Evaluate KV caches to materialize them and break dependency chain
            // from prefill (especially vision encoder graph). Without this, every
            // decode step drags the full prefill graph as a dependency.
            lm_guard.eval_fused_kv_caches();
            clear_cache();
            logits
                .slice_axis(1, seq_len - 1, seq_len)?
                .squeeze(Some(&[0, 1]))?
        };

        // Get input tokens for repetition penalty context
        let input_tokens = input_ids.to_uint32()?;
        let mut all_tokens: Vec<u32> = input_tokens.to_vec();

        // Apply repetition penalty to first token if enabled
        if repetition_penalty != 1.0 {
            last_logits = apply_repetition_penalty(
                &last_logits,
                &all_tokens,
                repetition_penalty,
                Some(repetition_context_size),
            )?;
        }

        // Sample first token
        let (mut token, mut logprobs_arr): (MxArray, Option<MxArray>) = if return_logprobs {
            let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
            (tok, Some(lp))
        } else {
            let tok = sample(&last_logits, Some(sampling_config))?;
            (tok, None)
        };

        // Synchronously evaluate the first token so that prefill cost is fully paid
        // before entering the decode loop. Without this, step 0's async_eval blocks
        // for 30-40ms waiting for the deferred prefill graph to complete.
        // This matches Python mlx-vlm's pattern: mx.eval(y) after first generate_step.
        if return_logprobs {
            if let Some(ref lp) = logprobs_arr {
                MxArray::async_eval_arrays(&[&token, lp]);
            } else {
                MxArray::async_eval_arrays(&[&token]);
            }
        } else {
            MxArray::async_eval_arrays(&[&token]);
        }
        token.eval();

        // Track generated tokens
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
        let mut generated_logprobs: Vec<f32> = if return_logprobs {
            Vec::with_capacity(max_new_tokens as usize)
        } else {
            Vec::new()
        };
        let mut finish_reason = "length";

        // N-gram repetition detection: max pattern length to check (stop on first repeat)
        // Default to 0 (disabled) for VLM - mlx-vlm does NOT have this aggressive check.
        // VLM outputs structured data that can trigger false positives.
        let ngram_size = config.ngram_size.unwrap_or(0);

        // === STEP 4: Pipelined decode loop ===
        // Structure: Build next graph → async_eval → extract current token
        // This overlaps GPU evaluation of the next step with CPU extraction of the current step,
        // matching the pipelining pattern used by Python mlx-vlm's generate_step.
        #[allow(clippy::needless_range_loop)]
        for step in 0..max_new_tokens {
            // --- Phase 1: Build next step's graph while GPU evaluates current token ---
            let (next_tok, next_lp) = {
                let _stream_ctx = StreamContext::new(generation_stream);

                // Zero-copy reshape: keep token on GPU instead of CPU round-trip
                let token_2d = token.reshape(&[1, 1])?;
                let input_embeds = lm_guard.get_embeddings(&token_2d)?;

                // Position from Rust-side state (no GPU dependency)
                let rope_deltas = lm_guard.get_rope_deltas().unwrap_or(0);
                let cache_offset = lm_guard.get_fused_cache_offset() as i64;
                let pos_value = (cache_offset + rope_deltas) as f32;
                let decode_pos =
                    MxArray::from_float32(&[pos_value], &[1, 1, 1])?.broadcast_to(&[3, 1, 1])?;

                let logits = lm_guard.forward_fused(&input_embeds, &decode_pos)?;
                let mut next_logits = logits.squeeze(Some(&[0, 1]))?;

                // Apply repetition penalty if enabled
                if repetition_penalty != 1.0 {
                    next_logits = apply_repetition_penalty(
                        &next_logits,
                        &all_tokens,
                        repetition_penalty,
                        Some(repetition_context_size),
                    )?;
                }

                // Sample next token
                let (tok, lp): (MxArray, Option<MxArray>) = if return_logprobs {
                    let (t, l) = sample_and_logprobs(&next_logits, Some(sampling_config))?;
                    (t, Some(l))
                } else {
                    (sample(&next_logits, Some(sampling_config))?, None)
                };

                (tok, lp)
            };

            // Submit next step's graph to GPU (starts processing while we do CPU work below)
            if return_logprobs {
                if let Some(ref lp) = next_lp {
                    MxArray::async_eval_arrays(&[&next_tok, lp]);
                } else {
                    MxArray::async_eval_arrays(&[&next_tok]);
                }
            } else {
                MxArray::async_eval_arrays(&[&next_tok]);
            }

            // --- Phase 2: Extract current token (GPU already working on next step) ---
            token.eval();
            let token_value = token.item_at_int32(0)? as u32;

            generated_tokens.push(token_value);
            all_tokens.push(token_value);

            // Extract logprob if needed
            if return_logprobs && let Some(ref lp) = logprobs_arr {
                lp.eval();
                let token_logprob = lp.item_at_float32(token_value as usize)?;
                generated_logprobs.push(token_logprob);
            }

            // Check for EOS
            if token_value == eos_token_id as u32 {
                finish_reason = "stop";
                break;
            }

            // Check for repetition (loop detection) - find any repeating pattern
            let min_pattern_len = 8; // Minimum pattern length to consider
            let max_pattern_len = ngram_size as usize;
            // Skip repetition check entirely if ngram_size is 0 (disabled)
            if ngram_size > 0 && generated_tokens.len() >= (min_pattern_len * 2) {
                let len = generated_tokens.len();

                // Check for patterns of various lengths
                for pattern_len in min_pattern_len..=max_pattern_len.min(len / 2) {
                    let pattern1_start = len - pattern_len * 2;
                    let pattern2_start = len - pattern_len;

                    let pattern1 = &generated_tokens[pattern1_start..pattern1_start + pattern_len];
                    let pattern2 = &generated_tokens[pattern2_start..];

                    if pattern1 == pattern2 {
                        debug!(
                            "Detected {}-token pattern repetition, stopping",
                            pattern_len
                        );
                        finish_reason = "repetition";
                        // Keep only up to the first occurrence of the repeated pattern
                        generated_tokens.truncate(pattern2_start);
                        generated_logprobs.truncate(pattern2_start);
                        break;
                    }
                }

                // If we broke from the inner loop due to repetition, break from outer loop too
                if finish_reason == "repetition" {
                    break;
                }
            }

            // Periodic cleanup to release intermediate tensors and prevent memory accumulation
            // Every 256 tokens is a good balance (aligned with mlx-vlm)
            // Only clear cache - do NOT synchronize, as that kills GPU-CPU pipelining
            if step > 0 && step % 256 == 0 {
                clear_cache();
            }

            token = next_tok;
            logprobs_arr = next_lp;
        }

        // Reset fused caches after generation
        lm_guard.reset_fused_kv_caches();

        // Build result
        let tokens_array =
            MxArray::from_uint32(&generated_tokens, &[generated_tokens.len() as i64])?;
        let logprobs_array = if return_logprobs {
            MxArray::from_float32(&generated_logprobs, &[generated_logprobs.len() as i64])?
        } else {
            MxArray::from_float32(&[], &[0])?
        };

        Ok(GenerationResult {
            text: String::new(), // Caller should decode with tokenizer
            tokens: tokens_array,
            logprobs: logprobs_array,
            finish_reason: finish_reason.to_string(),
            num_tokens: generated_tokens.len(),
        })
    }

    /// Get model configuration
    #[napi(getter)]
    pub fn config(&self) -> ModelConfig {
        self.config.clone()
    }

    /// Check if model is fully initialized
    #[napi(getter)]
    pub fn is_initialized(&self) -> bool {
        self.visual.is_some() && self.language_model.is_some()
    }

    /// Load a VLM from disk
    ///
    /// Loads a model from a directory containing:
    /// - config.json: Model configuration
    /// - model.safetensors or model-*.safetensors: Model weights in SafeTensors format
    ///
    /// # Arguments
    /// * `model_path` - Path to the model directory
    ///
    /// # Returns
    /// * A fully initialized VLModel with loaded weights
    ///
    /// # Example
    /// ```typescript
    /// import { VLModel } from '@mlx-node/vlm';
    /// const model = await VLModel.load('./models/paddleocr-vl');
    /// const result = model.chat(messages, { imagePaths: ['./image.jpg'] });
    /// ```
    #[napi]
    pub fn load<'env>(env: &'env Env, model_path: String) -> Result<PromiseRaw<'env, VLModel>> {
        env.spawn_future_with_callback(
            async move {
                tokio::task::spawn_blocking(move || {
                    let path = Path::new(&model_path);

                    // Check if path exists
                    if !path.exists() {
                        return Err(napi::Error::from_reason(format!(
                            "Model path does not exist: {}",
                            model_path
                        )));
                    }

                    // Load configuration
                    let config_path = path.join("config.json");
                    if !config_path.exists() {
                        return Err(napi::Error::from_reason(format!(
                            "Config file not found: {}",
                            config_path.display()
                        )));
                    }

                    let config_data = fs::read_to_string(&config_path)?;
                    let raw_config: Value = serde_json::from_str(&config_data)?;

                    // Parse vision config
                    let vision_raw = &raw_config["vision_config"];
                    let vision_config = VisionConfig {
                        model_type: vision_raw["model_type"]
                            .as_str()
                            .unwrap_or("paddleocr_vl")
                            .to_string(),
                        hidden_size: vision_raw["hidden_size"].as_i64().unwrap_or_else(|| {
                            warn!("vision_config.hidden_size not found in config.json, using default 1152");
                            1152
                        }) as i32,
                        intermediate_size: vision_raw["intermediate_size"].as_i64().unwrap_or(4304)
                            as i32,
                        num_hidden_layers: vision_raw["num_hidden_layers"].as_i64().unwrap_or_else(|| {
                            warn!("vision_config.num_hidden_layers not found in config.json, using default 27");
                            27
                        }) as i32,
                        num_attention_heads: vision_raw["num_attention_heads"]
                            .as_i64()
                            .unwrap_or_else(|| {
                                warn!("vision_config.num_attention_heads not found in config.json, using default 16");
                                16
                            }) as i32,
                        num_channels: vision_raw["num_channels"].as_i64().unwrap_or(3) as i32,
                        image_size: vision_raw["image_size"].as_i64().unwrap_or(384) as i32,
                        patch_size: vision_raw["patch_size"].as_i64().unwrap_or_else(|| {
                            warn!("vision_config.patch_size not found in config.json, using default 14");
                            14
                        }) as i32,
                        hidden_act: vision_raw["hidden_act"]
                            .as_str()
                            .unwrap_or("gelu_pytorch_tanh")
                            .to_string(),
                        layer_norm_eps: vision_raw["layer_norm_eps"].as_f64().unwrap_or(1e-6),
                        attention_dropout: vision_raw["attention_dropout"].as_f64().unwrap_or(0.0),
                        spatial_merge_size: vision_raw["spatial_merge_size"].as_i64().unwrap_or_else(|| {
                            warn!("vision_config.spatial_merge_size not found in config.json, using default 2");
                            2
                        }) as i32,
                    };

                    // Parse text config
                    let text_raw = &raw_config["text_config"];
                    let mrope_section: Vec<i32> = text_raw["mrope_section"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_i64().map(|x| x as i32))
                                .collect()
                        })
                        .unwrap_or_else(|| vec![16, 24, 24]);

                    let text_config = TextConfig {
                        model_type: text_raw["model_type"]
                            .as_str()
                            .unwrap_or("paddleocr_vl")
                            .to_string(),
                        hidden_size: text_raw["hidden_size"].as_i64().unwrap_or_else(|| {
                            warn!("text_config.hidden_size not found in config.json, using default 1024");
                            1024
                        }) as i32,
                        num_hidden_layers: text_raw["num_hidden_layers"].as_i64().unwrap_or_else(|| {
                            warn!("text_config.num_hidden_layers not found in config.json, using default 18");
                            18
                        }) as i32,
                        intermediate_size: text_raw["intermediate_size"].as_i64().unwrap_or(3072)
                            as i32,
                        num_attention_heads: text_raw["num_attention_heads"].as_i64().unwrap_or_else(|| {
                            warn!("text_config.num_attention_heads not found in config.json, using default 16");
                            16
                        }) as i32,
                        rms_norm_eps: text_raw["rms_norm_eps"].as_f64().unwrap_or(1e-5),
                        vocab_size: text_raw["vocab_size"].as_i64().unwrap_or_else(|| {
                            warn!("text_config.vocab_size not found in config.json, using default 103424");
                            103424
                        }) as i32,
                        num_key_value_heads: text_raw["num_key_value_heads"].as_i64().unwrap_or(2)
                            as i32,
                        max_position_embeddings: text_raw["max_position_embeddings"]
                            .as_i64()
                            .unwrap_or(131072)
                            as i32,
                        rope_theta: text_raw["rope_theta"].as_f64().unwrap_or(500000.0),
                        rope_traditional: text_raw["rope_traditional"].as_bool().unwrap_or(false),
                        use_bias: text_raw["use_bias"].as_bool().unwrap_or(false),
                        head_dim: text_raw["head_dim"].as_i64().unwrap_or(128) as i32,
                        mrope_section,
                    };

                    // Build model config
                    let config = ModelConfig {
                        vision_config: vision_config.clone(),
                        text_config: text_config.clone(),
                        model_type: raw_config["model_type"]
                            .as_str()
                            .unwrap_or("paddleocr_vl")
                            .to_string(),
                        ignore_index: raw_config["ignore_index"].as_i64().unwrap_or(-100) as i32,
                        image_token_id: raw_config["image_token_id"].as_i64().unwrap_or(100295)
                            as i32,
                        video_token_id: raw_config["video_token_id"].as_i64().unwrap_or(100296)
                            as i32,
                        vision_start_token_id: raw_config["vision_start_token_id"]
                            .as_i64()
                            .unwrap_or(101305)
                            as i32,
                        vision_end_token_id: raw_config["vision_end_token_id"]
                            .as_i64()
                            .unwrap_or(101306) as i32,
                        eos_token_id: raw_config["eos_token_id"].as_i64().unwrap_or(2) as i32,
                    };

                    info!("📦 Loading PaddleOCR-VL model from: {}", model_path);
                    info!(
                        "   Vision: {} layers, {} hidden, {} heads",
                        vision_config.num_hidden_layers,
                        vision_config.hidden_size,
                        vision_config.num_attention_heads
                    );
                    info!(
                        "   Language: {} layers, {} hidden, {} heads",
                        text_config.num_hidden_layers,
                        text_config.hidden_size,
                        text_config.num_attention_heads
                    );

                    // Load SafeTensors weights
                    // Support both single file and sharded format
                    let safetensors_path = path.join("model.safetensors");
                    let mut all_weights: HashMap<String, MxArray> = HashMap::new();

                    if safetensors_path.exists() {
                        let st_file = SafeTensorsFile::load(&safetensors_path)?;
                        info!(
                            "  Loading {} tensors from model.safetensors",
                            st_file.tensor_names().len()
                        );
                        all_weights = st_file.load_tensors(&safetensors_path)?;
                    } else {
                        // Try sharded format (model-00001-of-00002.safetensors, etc.)
                        let mut shard_index = 1;
                        loop {
                            // Find sharded safetensors file matching pattern model-XXXXX-of-*.safetensors
                            // Find matching file
                            let mut found_shard = None;
                            for entry in fs::read_dir(path)? {
                                let entry = entry?;
                                let name = entry.file_name().to_string_lossy().to_string();
                                if name.starts_with(&format!("model-{:05}-of-", shard_index))
                                    && name.ends_with(".safetensors")
                                {
                                    found_shard = Some(entry.path());
                                    break;
                                }
                            }

                            match found_shard {
                                Some(shard_path) => {
                                    info!("  Loading shard: {}", shard_path.display());
                                    let st_file = SafeTensorsFile::load(&shard_path)?;
                                    let shard_weights = st_file.load_tensors(&shard_path)?;
                                    all_weights.extend(shard_weights);
                                    shard_index += 1;
                                }
                                None => {
                                    if shard_index == 1 {
                                        return Err(Error::new(
                                            Status::InvalidArg,
                                            format!("No SafeTensors files found in {}", model_path),
                                        ));
                                    }
                                    break;
                                }
                            }
                        }
                    }

                    info!("  Loaded {} total tensors", all_weights.len());

                    // Transform keys for PaddleOCR-VL format
                    let weights = load_paddleocr_vl_weights(all_weights)?;
                    info!("  After transformation: {} tensors", weights.len());

                    Ok::<_, Error>((config, weights, vision_config, text_config, model_path))
                })
                .await
                .map_err(|err| {
                    Error::new(
                        Status::GenericFailure,
                        format!("Failed to load model: {err}"),
                    )
                })
                .flatten()
            },
            |_, (config, weights, vision_config, text_config, model_path)| {
                // Build the model from weights (includes tokenizer loading)
                build_paddleocr_vl_from_weights(
                    config,
                    weights,
                    vision_config,
                    text_config,
                    &model_path,
                )
            },
        )
    }

    /// Load model configuration from disk without loading weights
    ///
    /// This is useful for inspecting model configuration before loading the full model.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model directory containing config.json
    ///
    /// # Returns
    /// * ModelConfig with vision and text configuration
    ///
    /// # Example
    /// ```typescript
    /// import { VLModel } from '@mlx-node/vlm';
    /// const config = await VLModel.loadConfig('./models/paddleocr-vl');
    /// console.log(config.visionConfig.hiddenSize);
    /// ```
    #[napi]
    pub fn load_config<'env>(
        env: &'env Env,
        model_path: String,
    ) -> Result<PromiseRaw<'env, ModelConfig>> {
        env.spawn_future_with_callback(
            async move {
                tokio::task::spawn_blocking(move || {
                    let path = Path::new(&model_path);

                    // Check if path exists
                    if !path.exists() {
                        return Err(napi::Error::from_reason(format!(
                            "Model path does not exist: {}",
                            model_path
                        )));
                    }

                    // Load configuration
                    let config_path = path.join("config.json");
                    if !config_path.exists() {
                        return Err(napi::Error::from_reason(format!(
                            "Config file not found: {}",
                            config_path.display()
                        )));
                    }

                    let config_data = fs::read_to_string(&config_path)?;
                    let raw_config: Value = serde_json::from_str(&config_data)?;

                    // Parse vision config
                    let vision_raw = &raw_config["vision_config"];
                    let vision_config = VisionConfig {
                        model_type: vision_raw["model_type"]
                            .as_str()
                            .unwrap_or("paddleocr_vl")
                            .to_string(),
                        hidden_size: vision_raw["hidden_size"].as_i64().unwrap_or_else(|| {
                            warn!("vision_config.hidden_size not found in config.json, using default 1152");
                            1152
                        }) as i32,
                        intermediate_size: vision_raw["intermediate_size"].as_i64().unwrap_or(4304)
                            as i32,
                        num_hidden_layers: vision_raw["num_hidden_layers"].as_i64().unwrap_or_else(|| {
                            warn!("vision_config.num_hidden_layers not found in config.json, using default 27");
                            27
                        }) as i32,
                        num_attention_heads: vision_raw["num_attention_heads"]
                            .as_i64()
                            .unwrap_or_else(|| {
                                warn!("vision_config.num_attention_heads not found in config.json, using default 16");
                                16
                            }) as i32,
                        num_channels: vision_raw["num_channels"].as_i64().unwrap_or(3) as i32,
                        image_size: vision_raw["image_size"].as_i64().unwrap_or(384) as i32,
                        patch_size: vision_raw["patch_size"].as_i64().unwrap_or_else(|| {
                            warn!("vision_config.patch_size not found in config.json, using default 14");
                            14
                        }) as i32,
                        hidden_act: vision_raw["hidden_act"]
                            .as_str()
                            .unwrap_or("gelu_pytorch_tanh")
                            .to_string(),
                        layer_norm_eps: vision_raw["layer_norm_eps"].as_f64().unwrap_or(1e-6),
                        attention_dropout: vision_raw["attention_dropout"].as_f64().unwrap_or(0.0),
                        spatial_merge_size: vision_raw["spatial_merge_size"].as_i64().unwrap_or_else(|| {
                            warn!("vision_config.spatial_merge_size not found in config.json, using default 2");
                            2
                        }) as i32,
                    };

                    // Parse text config
                    let text_raw = &raw_config["text_config"];
                    let mrope_section: Vec<i32> = text_raw["mrope_section"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_i64().map(|x| x as i32))
                                .collect()
                        })
                        .unwrap_or_else(|| vec![16, 24, 24]);

                    let text_config = TextConfig {
                        model_type: text_raw["model_type"]
                            .as_str()
                            .unwrap_or("paddleocr_vl")
                            .to_string(),
                        hidden_size: text_raw["hidden_size"].as_i64().unwrap_or_else(|| {
                            warn!("text_config.hidden_size not found in config.json, using default 1024");
                            1024
                        }) as i32,
                        num_hidden_layers: text_raw["num_hidden_layers"].as_i64().unwrap_or_else(|| {
                            warn!("text_config.num_hidden_layers not found in config.json, using default 18");
                            18
                        }) as i32,
                        intermediate_size: text_raw["intermediate_size"].as_i64().unwrap_or(3072)
                            as i32,
                        num_attention_heads: text_raw["num_attention_heads"].as_i64().unwrap_or_else(|| {
                            warn!("text_config.num_attention_heads not found in config.json, using default 16");
                            16
                        }) as i32,
                        rms_norm_eps: text_raw["rms_norm_eps"].as_f64().unwrap_or(1e-5),
                        vocab_size: text_raw["vocab_size"].as_i64().unwrap_or_else(|| {
                            warn!("text_config.vocab_size not found in config.json, using default 103424");
                            103424
                        }) as i32,
                        num_key_value_heads: text_raw["num_key_value_heads"].as_i64().unwrap_or(2)
                            as i32,
                        max_position_embeddings: text_raw["max_position_embeddings"]
                            .as_i64()
                            .unwrap_or(131072)
                            as i32,
                        rope_theta: text_raw["rope_theta"].as_f64().unwrap_or(500000.0),
                        rope_traditional: text_raw["rope_traditional"].as_bool().unwrap_or(false),
                        use_bias: text_raw["use_bias"].as_bool().unwrap_or(false),
                        head_dim: text_raw["head_dim"].as_i64().unwrap_or(128) as i32,
                        mrope_section,
                    };

                    // Build model config
                    Ok(ModelConfig {
                        vision_config,
                        text_config,
                        model_type: raw_config["model_type"]
                            .as_str()
                            .unwrap_or("paddleocr_vl")
                            .to_string(),
                        ignore_index: raw_config["ignore_index"].as_i64().unwrap_or(-100) as i32,
                        image_token_id: raw_config["image_token_id"].as_i64().unwrap_or(100295)
                            as i32,
                        video_token_id: raw_config["video_token_id"].as_i64().unwrap_or(100296)
                            as i32,
                        vision_start_token_id: raw_config["vision_start_token_id"]
                            .as_i64()
                            .unwrap_or(101305)
                            as i32,
                        vision_end_token_id: raw_config["vision_end_token_id"]
                            .as_i64()
                            .unwrap_or(101306) as i32,
                        eos_token_id: raw_config["eos_token_id"].as_i64().unwrap_or(2) as i32,
                    })
                })
                .await
                .map_err(|err| {
                    Error::new(
                        Status::GenericFailure,
                        format!("Failed to load config: {err}"),
                    )
                })
                .flatten()
            },
            |_, config| Ok(config),
        )
    }
}

/// Build VLModel from loaded weights
fn build_paddleocr_vl_from_weights(
    config: ModelConfig,
    weights: HashMap<String, MxArray>,
    vision_config: VisionConfig,
    text_config: TextConfig,
    model_path: &str,
) -> Result<VLModel> {
    info!("🔧 Building PaddleOCR-VL model from weights...");

    // Helper to get weight with nice error message
    let get_weight = |key: &str| -> Result<&MxArray> {
        weights
            .get(key)
            .ok_or_else(|| Error::new(Status::InvalidArg, format!("Missing weight: {}", key)))
    };

    // === Build Vision Model ===
    info!("  Building vision model...");

    let patch_weight = get_weight("visual.embeddings.patch_embedding.weight")?;
    let position_weight = get_weight("visual.embeddings.position_embedding.weight")?;
    let post_norm_weight = get_weight("visual.post_layernorm.weight")?;
    let post_norm_bias = get_weight("visual.post_layernorm.bias")?;
    let projector_pre_norm_weight = get_weight("visual.projector.pre_norm.weight")?;
    let projector_pre_norm_bias = get_weight("visual.projector.pre_norm.bias")?;
    let projector_linear1_weight = get_weight("visual.projector.linear_1.weight")?;
    let projector_linear1_bias = get_weight("visual.projector.linear_1.bias")?;
    let projector_linear2_weight = get_weight("visual.projector.linear_2.weight")?;
    let projector_linear2_bias = get_weight("visual.projector.linear_2.bias")?;

    let mut vision_model = PaddleOCRVisionModel::new(
        vision_config.clone(),
        patch_weight,
        position_weight,
        post_norm_weight,
        post_norm_bias,
        projector_pre_norm_weight,
        projector_pre_norm_bias,
        projector_linear1_weight,
        projector_linear1_bias,
        projector_linear2_weight,
        projector_linear2_bias,
    )?;

    // Add vision encoder layers
    for i in 0..vision_config.num_hidden_layers {
        let prefix = format!("visual.layers.{}", i);

        let ln1_weight = get_weight(&format!("{}.layer_norm1.weight", prefix))?;
        let ln1_bias = get_weight(&format!("{}.layer_norm1.bias", prefix))?;
        let ln2_weight = get_weight(&format!("{}.layer_norm2.weight", prefix))?;
        let ln2_bias = get_weight(&format!("{}.layer_norm2.bias", prefix))?;
        let qkv_weight = get_weight(&format!("{}.self_attn.qkv.weight", prefix))?;
        let qkv_bias = weights.get(&format!("{}.self_attn.qkv.bias", prefix));
        let out_weight = get_weight(&format!("{}.self_attn.out_proj.weight", prefix))?;
        let out_bias = weights.get(&format!("{}.self_attn.out_proj.bias", prefix));
        let fc1_weight = get_weight(&format!("{}.mlp.fc1.weight", prefix))?;
        let fc1_bias = weights.get(&format!("{}.mlp.fc1.bias", prefix));
        let fc2_weight = get_weight(&format!("{}.mlp.fc2.weight", prefix))?;
        let fc2_bias = weights.get(&format!("{}.mlp.fc2.bias", prefix));

        // Create layer components
        let layer_norm1 = LayerNorm::from_weights(
            ln1_weight,
            Some(ln1_bias),
            Some(vision_config.layer_norm_eps),
        )?;
        let layer_norm2 = LayerNorm::from_weights(
            ln2_weight,
            Some(ln2_bias),
            Some(vision_config.layer_norm_eps),
        )?;

        let self_attn = VisionAttention::new(
            vision_config.hidden_size as u32,
            vision_config.num_attention_heads as u32,
            qkv_weight,
            qkv_bias,
            out_weight,
            out_bias,
        )?;

        let mlp = VisionMLP::new(fc1_weight, fc1_bias, fc2_weight, fc2_bias)?;

        let encoder_layer = VisionEncoderLayer::new(&layer_norm1, &layer_norm2, &self_attn, &mlp);
        vision_model.add_layer(&encoder_layer);
    }

    info!(
        "    Added {} vision encoder layers",
        vision_config.num_hidden_layers
    );

    // === Build Language Model ===
    info!("  Building language model...");

    let embed_tokens_weight = get_weight("language_model.model.embed_tokens.weight")?;
    let final_norm_weight = get_weight("language_model.model.norm.weight")?;
    let lm_head_weight = get_weight("language_model.lm_head.weight")?;

    let mut language_model = ERNIELanguageModel::new(
        text_config.clone(),
        embed_tokens_weight,
        final_norm_weight,
        lm_head_weight,
    )?;

    // Add decoder layers
    for i in 0..text_config.num_hidden_layers {
        let prefix = format!("language_model.model.layers.{}", i);

        let q_weight = get_weight(&format!("{}.self_attn.q_proj.weight", prefix))?;
        let k_weight = get_weight(&format!("{}.self_attn.k_proj.weight", prefix))?;
        let v_weight = get_weight(&format!("{}.self_attn.v_proj.weight", prefix))?;
        let o_weight = get_weight(&format!("{}.self_attn.o_proj.weight", prefix))?;
        let gate_weight = get_weight(&format!("{}.mlp.gate_proj.weight", prefix))?;
        let up_weight = get_weight(&format!("{}.mlp.up_proj.weight", prefix))?;
        let down_weight = get_weight(&format!("{}.mlp.down_proj.weight", prefix))?;
        let input_norm_weight = get_weight(&format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_norm_weight =
            get_weight(&format!("{}.post_attention_layernorm.weight", prefix))?;

        let decoder_layer = PaddleOCRDecoderLayer::new(
            text_config.clone(),
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            gate_weight,
            up_weight,
            down_weight,
            input_norm_weight,
            post_attn_norm_weight,
        )?;

        language_model.add_layer(&decoder_layer);
    }

    info!("    Added {} decoder layers", text_config.num_hidden_layers);

    // === Build full model ===
    let mut model = VLModel::new(config);
    model.set_visual(&vision_model);
    model.set_language_model(&language_model);

    // === Load tokenizer ===
    let tokenizer_path = Path::new(model_path).join("tokenizer.json");
    if tokenizer_path.exists() {
        info!("  Loading tokenizer from {:?}...", tokenizer_path);
        match crate::tokenizer::Qwen3Tokenizer::from_file(&tokenizer_path) {
            Ok(tokenizer) => {
                model.tokenizer = Some(Arc::new(tokenizer));
                info!("    ✅ Tokenizer loaded");
            }
            Err(e) => {
                // Log warning but don't fail - tokenizer can be set manually
                info!(
                    "    ⚠️ Could not load tokenizer: {}. Use setTokenizer() to set manually.",
                    e
                );
            }
        }
    } else {
        info!(
            "    ℹ️ No tokenizer.json found at {:?}. Use setTokenizer() to set manually.",
            tokenizer_path
        );
    }

    info!("✅ PaddleOCR-VL model built successfully");
    info!("   Vision: {} layers loaded", vision_model.num_layers());
    info!("   Language: {} layers loaded", language_model.num_layers());

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let config = ModelConfig::default();
        let model = VLModel::new(config);

        assert!(!model.is_initialized());
    }

    #[test]
    fn test_model_not_initialized_by_default() {
        let config = ModelConfig::default();
        let model = VLModel::new(config);

        // Model should not be initialized without vision and language models
        assert!(!model.is_initialized());
    }

    #[test]
    fn test_model_config_accessible() {
        let config = ModelConfig::default();
        let model = VLModel::new(config);

        // Config should be accessible after creation
        assert_eq!(model.config.image_token_id, 100295);
        assert_eq!(model.config.model_type, "paddleocr_vl");
        assert_eq!(model.config.vision_config.hidden_size, 1152);
        assert_eq!(model.config.text_config.hidden_size, 1024);
    }

    #[test]
    fn test_model_config_special_tokens() {
        let config = ModelConfig::default();
        let model = VLModel::new(config);

        assert_eq!(model.config.image_token_id, 100295);
        assert_eq!(model.config.video_token_id, 100296);
        assert_eq!(model.config.vision_start_token_id, 101305);
        assert_eq!(model.config.vision_end_token_id, 101306);
        assert_eq!(model.config.eos_token_id, 2);
    }

    #[test]
    fn test_model_config_mrope_section() {
        let config = ModelConfig::default();

        // mRoPE section should sum to head_dim when doubled
        // [16, 24, 24] * 2 = [32, 48, 48] -> total = 128 = head_dim
        assert_eq!(config.text_config.mrope_section, vec![16, 24, 24]);
        let total: i32 = config.text_config.mrope_section.iter().map(|x| x * 2).sum();
        assert_eq!(total, config.text_config.head_dim);
    }

    #[test]
    fn test_model_config_vision_defaults() {
        let config = ModelConfig::default();

        assert_eq!(config.vision_config.hidden_size, 1152);
        assert_eq!(config.vision_config.num_hidden_layers, 27);
        assert_eq!(config.vision_config.num_attention_heads, 16);
        assert_eq!(config.vision_config.patch_size, 14);
        assert_eq!(config.vision_config.spatial_merge_size, 2);
    }

    #[test]
    fn test_model_config_text_defaults() {
        let config = ModelConfig::default();

        assert_eq!(config.text_config.hidden_size, 1024);
        assert_eq!(config.text_config.num_hidden_layers, 18);
        assert_eq!(config.text_config.num_attention_heads, 16);
        assert_eq!(config.text_config.head_dim, 128);
    }
}
