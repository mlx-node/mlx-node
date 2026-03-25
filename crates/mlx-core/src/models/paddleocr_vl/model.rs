/**
 * PaddleOCR-VL Full Model
 *
 * Combines vision encoder and language model for vision-language tasks.
 */
use crate::array::{MxArray, clear_cache};
use crate::models::paddleocr_vl::chat::{
    ChatRole, VLMBatchItem, VLMChatConfig, VLMChatMessage, VLMChatResult,
};
use crate::models::paddleocr_vl::config::{ModelConfig, TextConfig, VisionConfig};
use crate::models::paddleocr_vl::language::{ERNIELanguageModel, PaddleOCRDecoderLayer};
use crate::models::paddleocr_vl::persistence::load_paddleocr_vl_weights;
use crate::models::paddleocr_vl::processing::{ImageProcessor, ImageProcessorConfig};
use crate::models::paddleocr_vl::vision::PaddleOCRVisionModel;
use crate::models::qwen3::{GenerationConfig, GenerationResult};
use crate::nn::LayerNorm;
use crate::sampling::{
    SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
    sample, sample_and_logprobs,
};
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
    /// * `config` - Chat configuration (including images for automatic processing)
    ///
    /// # Returns
    /// * VLMChatResult with generated text
    ///
    /// # Example
    /// ```typescript
    /// const result = await model.chat(
    ///   [{ role: 'user', content: 'Describe this image.' }],
    ///   { images: [readFileSync('./photo.jpg')], maxNewTokens: 256 }
    /// );
    /// ```
    #[napi]
    pub async fn chat(
        &self,
        messages: Vec<VLMChatMessage>,
        config: Option<VLMChatConfig>,
    ) -> Result<VLMChatResult> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::GenericFailure,
                "Tokenizer not set. Use set_tokenizer() first.",
            )
        })?;

        // Merge passed config with defaults - any None fields use the default values
        let default_config = VLMChatConfig::default();
        let config = match config {
            Some(c) => VLMChatConfig {
                images: c.images,
                max_new_tokens: c.max_new_tokens.or(default_config.max_new_tokens),
                temperature: c.temperature.or(default_config.temperature),
                top_k: c.top_k.or(default_config.top_k),
                top_p: c.top_p.or(default_config.top_p),
                repetition_penalty: c.repetition_penalty.or(default_config.repetition_penalty),
                presence_penalty: c.presence_penalty.or(default_config.presence_penalty),
                presence_context_size: c
                    .presence_context_size
                    .or(default_config.presence_context_size),
                frequency_penalty: c.frequency_penalty.or(default_config.frequency_penalty),
                frequency_context_size: c
                    .frequency_context_size
                    .or(default_config.frequency_context_size),
                return_logprobs: c.return_logprobs.or(default_config.return_logprobs),
            },
            None => default_config,
        };

        // spawn_blocking: image processing + tokenization (CPU-bound)
        let vision_config = self.config.vision_config.clone();
        let eos_token_id = self.config.eos_token_id;
        let tokenizer_clone = tokenizer.clone();
        let (input_ids, pixel_values, grid_thw, gen_config) =
            napi::bindgen_prelude::spawn_blocking(move || {
                // Process images if image buffers provided
                let (pixel_values, grid_thw) = if let Some(ref images) = config.images {
                    if images.is_empty() {
                        (None, None)
                    } else {
                        let processor_config = ImageProcessorConfig {
                            patch_size: vision_config.patch_size,
                            merge_size: vision_config.spatial_merge_size,
                            ..ImageProcessorConfig::default()
                        };
                        let processor = ImageProcessor::new(Some(processor_config));
                        let image_refs: Vec<&[u8]> = images.iter().map(|b| &b[..]).collect();
                        let processed = processor.process_many(&image_refs)?;

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
                        let grid_thw = processed.grid_thw();
                        (Some(pixel_values), Some(grid_thw))
                    }
                } else {
                    (None, None)
                };

                // Count image tokens
                let num_image_tokens = if let Some(ref grid) = grid_thw {
                    grid.eval();
                    let grid_data = grid.to_int32()?;
                    let spatial_merge_size = vision_config.spatial_merge_size;
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

                // Encode the text
                let token_ids = tokenizer_clone.encode_sync(&formatted, None)?;
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
                    presence_penalty: config.presence_penalty,
                    presence_context_size: config.presence_context_size,
                    frequency_penalty: config.frequency_penalty,
                    frequency_context_size: config.frequency_context_size,
                    max_consecutive_tokens: None,
                    max_ngram_repeats: None,
                    ngram_size: None,
                    eos_token_id: Some(eos_token_id),
                    return_logprobs: config.return_logprobs,
                    prefill_step_size: None,
                    kv_cache_bits: None,
                    kv_cache_group_size: None,
                    num_draft_tokens: None,
                    report_performance: None,
                };

                Ok::<_, Error>((input_ids, pixel_values, grid_thw, gen_config))
            })
            .await
            .map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("VLM chat preprocessing failed: {}", e),
                )
            })??;

        // Generate (already async)
        let result = self
            .generate(
                &input_ids,
                pixel_values.as_ref(),
                grid_thw.as_ref(),
                Some(gen_config),
            )
            .await?;

        // spawn_blocking: decode tokens → text (CPU-bound)
        let result_tokens = result.tokens.clone();
        let text = napi::bindgen_prelude::spawn_blocking(move || {
            result_tokens.eval();
            let tokens_vec = result_tokens.to_uint32()?;
            let text = tokenizer.decode_sync(&tokens_vec, true)?;
            let text = text.replace("<|im_end|>", "").trim().to_string();
            Ok::<String, Error>(text)
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("VLM chat decoding failed: {}", e),
            )
        })??;

        Ok(VLMChatResult {
            text,
            tokens: result.tokens,
            logprobs: result.logprobs,
            finish_reason: result.finish_reason,
            num_tokens: result.num_tokens,
        })
    }

    /// Simple OCR: extract text from encoded image bytes
    ///
    /// Convenience method that processes an image and extracts all text.
    ///
    /// # Arguments
    /// * `image_data` - Encoded image bytes (PNG/JPEG)
    /// * `prompt` - Optional custom prompt (default: "Extract all text from this image.")
    ///
    /// # Returns
    /// * Extracted text as a string
    ///
    /// # Example
    /// ```typescript
    /// const text = await model.ocr(imageBuffer);
    /// console.log(text);
    /// ```
    #[napi]
    pub async fn ocr(&self, image_data: Buffer, prompt: Option<String>) -> Result<String> {
        let prompt = prompt.unwrap_or_else(|| "Extract all text from this image.".to_string());

        let messages = vec![VLMChatMessage {
            role: ChatRole::User,
            content: prompt,
        }];

        let config = VLMChatConfig {
            images: Some(vec![image_data]),
            ..Default::default()
        };

        let result = self.chat(messages, Some(config)).await?;

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
        Self::merge_input_ids_with_image_features(
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
        input_ids: &MxArray,
        image_grid_thw: Option<&MxArray>,
        spatial_merge_size: i32,
        image_token_id: i32,
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
    pub async fn generate(
        &self,
        input_ids: &MxArray,
        pixel_values: Option<&MxArray>,
        image_grid_thw: Option<&MxArray>,
        config: Option<GenerationConfig>,
    ) -> Result<GenerationResult> {
        let config = config.unwrap_or_default();

        // Clone Arc fields and input arrays before the closure
        let visual_arc = self.visual.clone();
        let lm_arc = self
            .language_model
            .clone()
            .ok_or_else(|| Error::new(Status::GenericFailure, "Language model not set"))?;
        let model_config = self.config.clone();
        let input_ids = input_ids.clone();
        let pixel_values = pixel_values.cloned();
        let image_grid_thw = image_grid_thw.cloned();

        napi::bindgen_prelude::spawn_blocking(move || {
            // Extract config with defaults - aligned with mlx-vlm generate_step defaults
            let max_new_tokens = config.max_new_tokens.unwrap_or(256); // mlx-vlm DEFAULT_MAX_TOKENS
            let temperature = config.temperature.unwrap_or(0.0); // mlx-vlm: greedy by default
            let top_k = config.top_k.unwrap_or(0);
            let top_p = config.top_p.unwrap_or(1.0);
            let min_p = config.min_p.unwrap_or(0.0);
            let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
            let repetition_context_size = config.repetition_context_size.unwrap_or(20);
            let presence_penalty = config.presence_penalty.unwrap_or(0.0);
            let presence_context_size = config.presence_context_size.unwrap_or(20);
            let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
            let frequency_context_size = config.frequency_context_size.unwrap_or(20);
            let eos_token_id = config.eos_token_id.unwrap_or(model_config.eos_token_id);
            let return_logprobs = config.return_logprobs.unwrap_or(false);

            debug!(
                "Starting VLM generation with KV cache: max_tokens={}, temp={}, top_k={}, top_p={}, rep_penalty={}",
                max_new_tokens, temperature, top_k, top_p, repetition_penalty
            );

            // Create dedicated generation stream for GPU-CPU pipelining.
            let generation_stream = Stream::new(DeviceType::Gpu);

            // Prepare sampling config
            let sampling_config = SamplingConfig {
                temperature: Some(temperature),
                top_k: Some(top_k),
                top_p: Some(top_p),
                min_p: Some(min_p),
            };

            // Get language model with write access for KV cache
            let mut lm_guard = lm_arc.write().map_err(|_| {
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
            let vision_features = {
                let _stream_ctx = StreamContext::new(generation_stream);
                if let (Some(pv), Some(grid)) = (&pixel_values, &image_grid_thw) {
                    let visual = visual_arc
                        .as_ref()
                        .ok_or_else(|| Error::new(Status::GenericFailure, "Vision model not set"))?;
                    Some(visual.forward(pv, grid)?)
                } else {
                    None
                }
            };

            // === STEP 2: Compute proper position IDs for mRoPE ===
            let (position_ids, rope_deltas) = VLModel::get_rope_index(
                &input_ids,
                image_grid_thw.as_ref(),
                model_config.vision_config.spatial_merge_size,
                model_config.image_token_id,
            )?;
            debug!("Computed position_ids with rope_deltas: {}", rope_deltas);

            // Store position state for decode phase
            lm_guard.set_position_state(position_ids.clone(), rope_deltas);

            // === STEP 3: Prefill - process prompt with vision features ===
            let inputs_embeds = {
                let _stream_ctx = StreamContext::new(generation_stream);
                let embeds = lm_guard.get_embeddings(&input_ids)?;
                if let Some(ref vf) = vision_features {
                    let embed_dtype = embeds.dtype()?;
                    let vf_cast = if vf.dtype()? != embed_dtype {
                        vf.astype(embed_dtype)?
                    } else {
                        vf.clone()
                    };
                    VLModel::merge_input_ids_with_image_features(
                        model_config.image_token_id,
                        &vf_cast,
                        &embeds,
                        &input_ids,
                    )?
                } else {
                    embeds
                }
            };

            // Chunked prefill
            let prefill_step_size: i64 = 2048;
            let seq_len = inputs_embeds.shape_at(1)?;

            let mut last_logits = if seq_len > prefill_step_size {
                let mut offset: i64 = 0;
                let mut chunk_logits = None;

                while offset < seq_len {
                    let chunk_end = std::cmp::min(offset + prefill_step_size, seq_len);
                    let n_to_process = if chunk_end < seq_len {
                        chunk_end - offset
                    } else {
                        seq_len - offset
                    };

                    let chunk_embeds = inputs_embeds.slice_axis(1, offset, offset + n_to_process)?;
                    let chunk_pos = position_ids.slice_axis(2, offset, offset + n_to_process)?;

                    {
                        let _stream_ctx = StreamContext::new(generation_stream);
                        chunk_logits = Some(lm_guard.forward_fused(&chunk_embeds, &chunk_pos)?);
                    }

                    lm_guard.eval_fused_kv_caches();
                    clear_cache();

                    offset += n_to_process;
                }

                let logits = chunk_logits.unwrap();
                let last_seq = logits.shape_at(1)?;
                logits
                    .slice_axis(1, last_seq - 1, last_seq)?
                    .squeeze(Some(&[0, 1]))?
            } else {
                let logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    lm_guard.forward_fused(&inputs_embeds, &position_ids)?
                };
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
            if presence_penalty != 0.0 {
                last_logits = apply_presence_penalty(
                    &last_logits,
                    &all_tokens,
                    presence_penalty,
                    Some(presence_context_size),
                )?;
            }
            if frequency_penalty != 0.0 {
                last_logits = apply_frequency_penalty(
                    &last_logits,
                    &all_tokens,
                    frequency_penalty,
                    Some(frequency_context_size),
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

            // Synchronously evaluate the first token
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

            let ngram_size = config.ngram_size.unwrap_or(0);

            // === STEP 4: Pipelined decode loop ===
            #[allow(clippy::needless_range_loop)]
            for step in 0..max_new_tokens {
                let (next_tok, next_lp) = {
                    let _stream_ctx = StreamContext::new(generation_stream);

                    let token_2d = token.reshape(&[1, 1])?;
                    let input_embeds = lm_guard.get_embeddings(&token_2d)?;

                    let rope_deltas = lm_guard.get_rope_deltas().unwrap_or(0);
                    let cache_offset = lm_guard.get_fused_cache_offset() as i64;
                    let pos_value = (cache_offset + rope_deltas) as f32;
                    let decode_pos =
                        MxArray::from_float32(&[pos_value], &[1, 1, 1])?.broadcast_to(&[3, 1, 1])?;

                    let logits = lm_guard.forward_fused(&input_embeds, &decode_pos)?;
                    let mut next_logits = logits.squeeze(Some(&[0, 1]))?;

                    if repetition_penalty != 1.0 {
                        next_logits = apply_repetition_penalty(
                            &next_logits,
                            &all_tokens,
                            repetition_penalty,
                            Some(repetition_context_size),
                        )?;
                    }
                    if presence_penalty != 0.0 {
                        next_logits = apply_presence_penalty(
                            &next_logits,
                            &all_tokens,
                            presence_penalty,
                            Some(presence_context_size),
                        )?;
                    }
                    if frequency_penalty != 0.0 {
                        next_logits = apply_frequency_penalty(
                            &next_logits,
                            &all_tokens,
                            frequency_penalty,
                            Some(frequency_context_size),
                        )?;
                    }

                    let (tok, lp): (MxArray, Option<MxArray>) = if return_logprobs {
                        let (t, l) = sample_and_logprobs(&next_logits, Some(sampling_config))?;
                        (t, Some(l))
                    } else {
                        (sample(&next_logits, Some(sampling_config))?, None)
                    };

                    (tok, lp)
                };

                if return_logprobs {
                    if let Some(ref lp) = next_lp {
                        MxArray::async_eval_arrays(&[&next_tok, lp]);
                    } else {
                        MxArray::async_eval_arrays(&[&next_tok]);
                    }
                } else {
                    MxArray::async_eval_arrays(&[&next_tok]);
                }

                token.eval();
                let token_value = token.item_at_int32(0)? as u32;

                generated_tokens.push(token_value);
                all_tokens.push(token_value);

                if return_logprobs && let Some(ref lp) = logprobs_arr {
                    lp.eval();
                    let token_logprob = lp.item_at_float32(token_value as usize)?;
                    generated_logprobs.push(token_logprob);
                }

                if token_value == eos_token_id as u32 {
                    finish_reason = "stop";
                    break;
                }

                let min_pattern_len = 8;
                let max_pattern_len = ngram_size as usize;
                if ngram_size > 0 && generated_tokens.len() >= (min_pattern_len * 2) {
                    let len = generated_tokens.len();

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
                            generated_tokens.truncate(pattern2_start);
                            generated_logprobs.truncate(pattern2_start);
                            break;
                        }
                    }

                    if finish_reason == "repetition" {
                        break;
                    }
                }

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
                text: String::new(),
                tokens: tokens_array,
                logprobs: logprobs_array,
                finish_reason: finish_reason.to_string(),
                num_tokens: generated_tokens.len(),
                first_token_elapsed_ms: None,
            })
        })
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("VLM generation failed: {}", e)))?
    }

    /// Batch OCR: extract text from multiple images simultaneously
    ///
    /// Processes N images with sequential prefill + batched decode for ~N× decode throughput.
    ///
    /// # Arguments
    /// * `images` - Encoded image buffers
    /// * `config` - Optional chat configuration (shared across all items)
    ///
    /// # Returns
    /// * Vec of extracted text strings, one per image
    ///
    /// # Example
    /// ```typescript
    /// import { readFileSync } from 'fs';
    /// const images = ['page1.jpg', 'page2.jpg'].map(p => readFileSync(p));
    /// const texts = await model.ocrBatch(images);
    /// ```
    #[napi]
    pub async fn ocr_batch(
        &self,
        images: Vec<Buffer>,
        config: Option<VLMChatConfig>,
    ) -> Result<Vec<String>> {
        let prompt = "Extract all text from this image.".to_string();

        let batch: Vec<VLMBatchItem> = images
            .into_iter()
            .map(|image_data| VLMBatchItem {
                messages: vec![VLMChatMessage {
                    role: ChatRole::User,
                    content: prompt.clone(),
                }],
                images: Some(vec![image_data]),
            })
            .collect();

        let results = self.batch(batch, config).await?;

        Ok(results
            .into_iter()
            .map(|r| {
                let text = r.text;
                text.strip_prefix("\\(\\text{")
                    .and_then(|s| s.strip_suffix("}\\)"))
                    .map(|s| s.to_string())
                    .unwrap_or(text)
            })
            .collect())
    }

    /// Batch chat: process multiple items simultaneously
    ///
    /// Sequential prefill + batched decode. Each item can have different images/prompts.
    ///
    /// # Arguments
    /// * `batch` - Batch items, each with messages and optional images
    /// * `config` - Optional shared chat configuration
    ///
    /// # Returns
    /// * Vec of VLMChatResult, one per batch item
    #[napi]
    pub async fn batch(
        &self,
        batch: Vec<VLMBatchItem>,
        config: Option<VLMChatConfig>,
    ) -> Result<Vec<VLMChatResult>> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::GenericFailure,
                "Tokenizer not set. Use set_tokenizer() first.",
            )
        })?;

        if batch.is_empty() {
            return Ok(Vec::new());
        }

        // Fall back to sequential for batch_size == 1
        if batch.len() == 1 {
            let item = &batch[0];
            let default_cfg = VLMChatConfig::default();
            let base = config.as_ref().unwrap_or(&default_cfg);
            let c = Some(VLMChatConfig {
                images: item
                    .images
                    .as_ref()
                    .map(|imgs| imgs.iter().map(|b| Buffer::from(b.to_vec())).collect()),
                max_new_tokens: base.max_new_tokens,
                temperature: base.temperature,
                top_k: base.top_k,
                top_p: base.top_p,
                repetition_penalty: base.repetition_penalty,
                presence_penalty: base.presence_penalty,
                presence_context_size: base.presence_context_size,
                frequency_penalty: base.frequency_penalty,
                frequency_context_size: base.frequency_context_size,
                return_logprobs: base.return_logprobs,
            });
            let result = self.chat(item.messages.clone(), c).await?;
            return Ok(vec![result]);
        }

        let default_config = VLMChatConfig::default();
        let config = match config {
            Some(c) => VLMChatConfig {
                images: None, // Per-item
                max_new_tokens: c.max_new_tokens.or(default_config.max_new_tokens),
                temperature: c.temperature.or(default_config.temperature),
                top_k: c.top_k.or(default_config.top_k),
                top_p: c.top_p.or(default_config.top_p),
                repetition_penalty: c.repetition_penalty.or(default_config.repetition_penalty),
                presence_penalty: c.presence_penalty.or(default_config.presence_penalty),
                presence_context_size: c
                    .presence_context_size
                    .or(default_config.presence_context_size),
                frequency_penalty: c.frequency_penalty.or(default_config.frequency_penalty),
                frequency_context_size: c
                    .frequency_context_size
                    .or(default_config.frequency_context_size),
                return_logprobs: c.return_logprobs.or(default_config.return_logprobs),
            },
            None => default_config,
        };

        // Clone Arc fields for use inside spawn_blocking
        let visual_arc = self.visual.clone();
        let lm_arc = self
            .language_model
            .clone()
            .ok_or_else(|| Error::new(Status::GenericFailure, "Language model not set"))?;
        let model_config = self.config.clone();
        let tokenizer_clone = tokenizer.clone();

        // spawn_blocking: image processing + tokenization + generate_batch + decoding
        napi::bindgen_prelude::spawn_blocking(move || {
            let gen_config = GenerationConfig {
                max_new_tokens: config.max_new_tokens,
                temperature: config.temperature,
                top_k: config.top_k,
                top_p: config.top_p,
                min_p: None,
                repetition_penalty: config.repetition_penalty,
                repetition_context_size: None,
                presence_penalty: config.presence_penalty,
                presence_context_size: config.presence_context_size,
                frequency_penalty: config.frequency_penalty,
                frequency_context_size: config.frequency_context_size,
                max_consecutive_tokens: None,
                max_ngram_repeats: None,
                ngram_size: None,
                eos_token_id: Some(model_config.eos_token_id),
                return_logprobs: config.return_logprobs,
                prefill_step_size: None,
                kv_cache_bits: None,
                kv_cache_group_size: None,
                num_draft_tokens: None,
                report_performance: None,
            };

            // Prepare per-item inputs
            let batch_size = batch.len();
            let mut all_input_ids = Vec::with_capacity(batch_size);
            let mut all_pixel_values = Vec::with_capacity(batch_size);
            let mut all_grid_thws = Vec::with_capacity(batch_size);

            for item in &batch {
                let (pixel_values, grid_thw) = if let Some(ref images) = item.images {
                    if images.is_empty() {
                        (None, None)
                    } else {
                        let processor_config = ImageProcessorConfig {
                            patch_size: model_config.vision_config.patch_size,
                            merge_size: model_config.vision_config.spatial_merge_size,
                            ..ImageProcessorConfig::default()
                        };
                        let processor = ImageProcessor::new(Some(processor_config));
                        let image_refs: Vec<&[u8]> = images.iter().map(|b| &b[..]).collect();
                        let processed = processor.process_many(&image_refs)?;
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
                        let grid_thw = processed.grid_thw();
                        (Some(pixel_values), Some(grid_thw))
                    }
                } else {
                    (None, None)
                };

                let num_image_tokens = if let Some(ref grid) = grid_thw {
                    grid.eval();
                    let grid_data = grid.to_int32()?;
                    let spatial_merge_size = model_config.vision_config.spatial_merge_size;
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

                let formatted = crate::models::paddleocr_vl::chat::format_vlm_chat(
                    &item.messages,
                    num_image_tokens,
                );
                let token_ids = tokenizer_clone.encode_sync(&formatted, None)?;
                let input_ids = MxArray::from_uint32(&token_ids, &[1, token_ids.len() as i64])?;

                all_input_ids.push(input_ids);
                all_pixel_values.push(pixel_values);
                all_grid_thws.push(grid_thw);
            }

            // Run batch generation (generate_batch is private, stays sync)
            let results = VLModel::generate_batch_impl(
                &visual_arc,
                &lm_arc,
                &model_config,
                &all_input_ids,
                &all_pixel_values,
                &all_grid_thws,
                Some(gen_config),
            )?;

            // Decode results
            let mut chat_results = Vec::with_capacity(batch_size);
            for result in results {
                result.tokens.eval();
                let tokens_vec = result.tokens.to_uint32()?;
                let text = tokenizer_clone.decode_sync(&tokens_vec, true)?;
                let text = text.replace("<|im_end|>", "").trim().to_string();

                chat_results.push(VLMChatResult {
                    text,
                    tokens: result.tokens,
                    logprobs: result.logprobs,
                    finish_reason: result.finish_reason,
                    num_tokens: result.num_tokens,
                });
            }

            Ok(chat_results)
        })
        .await
        .map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("VLM batch processing failed: {}", e),
            )
        })?
    }

    /// Batch generate: sequential prefill + batched decode
    ///
    /// Each item is prefilled independently (different image sizes → different vision tokens),
    /// then KV caches are merged and decode runs in batch for ~N× throughput.
    fn generate_batch_impl(
        visual_arc: &Option<Arc<PaddleOCRVisionModel>>,
        lm_arc: &Arc<RwLock<ERNIELanguageModel>>,
        model_config: &ModelConfig,
        all_input_ids: &[MxArray],
        all_pixel_values: &[Option<MxArray>],
        all_grid_thws: &[Option<MxArray>],
        config: Option<GenerationConfig>,
    ) -> Result<Vec<GenerationResult>> {
        let config = config.unwrap_or_default();

        if all_input_ids.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = all_input_ids.len();
        let max_new_tokens = config.max_new_tokens.unwrap_or(256);
        let temperature = config.temperature.unwrap_or(0.0);
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(20);
        let presence_penalty = config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = config.presence_context_size.unwrap_or(20);
        let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = config.frequency_context_size.unwrap_or(20);
        let eos_token_id = config.eos_token_id.unwrap_or(model_config.eos_token_id);
        let return_logprobs = config.return_logprobs.unwrap_or(false);

        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

        let generation_stream = Stream::new(DeviceType::Gpu);

        let lm = lm_arc;
        let visual = visual_arc.as_ref();

        let num_layers = {
            let lm_guard = lm.read().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire LM read lock")
            })?;
            lm_guard.num_layers_usize()
        };

        // === STEP 1: Sequential per-item prefill ===
        let mut item_kv_keys: Vec<Vec<MxArray>> = Vec::with_capacity(batch_size);
        let mut item_kv_values: Vec<Vec<MxArray>> = Vec::with_capacity(batch_size);
        let mut item_rope_deltas: Vec<i64> = Vec::with_capacity(batch_size);
        let mut item_cache_idxs: Vec<i32> = Vec::with_capacity(batch_size);
        let mut item_first_tokens: Vec<MxArray> = Vec::with_capacity(batch_size);
        let mut item_first_logprobs: Vec<Option<MxArray>> = Vec::with_capacity(batch_size);
        let mut item_all_tokens: Vec<Vec<u32>> = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let input_ids = &all_input_ids[i];
            let pixel_values = all_pixel_values[i].as_ref();
            let grid_thw = all_grid_thws[i].as_ref();

            let mut lm_guard = lm.write().map_err(|_| {
                Error::new(Status::GenericFailure, "Failed to acquire LM write lock")
            })?;

            lm_guard.init_fused_kv_caches();
            lm_guard.reset_position_state();

            // Vision features
            let vision_features = {
                let _stream_ctx = StreamContext::new(generation_stream);
                if let (Some(pv), Some(grid)) = (pixel_values, grid_thw) {
                    let v = visual.ok_or_else(|| {
                        Error::new(Status::GenericFailure, "Vision model not set")
                    })?;
                    Some(v.forward(pv, grid)?)
                } else {
                    None
                }
            };

            // Position IDs with mRoPE
            let (position_ids, rope_deltas) = VLModel::get_rope_index(
                input_ids,
                grid_thw,
                model_config.vision_config.spatial_merge_size,
                model_config.image_token_id,
            )?;

            // Merge embeddings
            let inputs_embeds = {
                let _stream_ctx = StreamContext::new(generation_stream);
                let embeds = lm_guard.get_embeddings(input_ids)?;
                if let Some(ref vf) = vision_features {
                    let embed_dtype = embeds.dtype()?;
                    let vf_cast = if vf.dtype()? != embed_dtype {
                        vf.astype(embed_dtype)?
                    } else {
                        vf.clone()
                    };
                    VLModel::merge_input_ids_with_image_features(
                        model_config.image_token_id,
                        &vf_cast,
                        &embeds,
                        input_ids,
                    )?
                } else {
                    embeds
                }
            };

            // Prefill and extract KV
            let (logits, kv_keys, kv_values, cache_idx) = {
                let _stream_ctx = StreamContext::new(generation_stream);
                lm_guard.forward_fused_extract_kv(&inputs_embeds, &position_ids)?
            };

            clear_cache();

            // Sample first token (with logprobs if requested)
            let seq_len = logits.shape_at(1)?;
            let mut last_logits = logits
                .slice_axis(1, seq_len - 1, seq_len)?
                .squeeze(Some(&[0, 1]))?;

            // Apply repetition penalty to first token (matches single-item generate())
            let input_tokens = input_ids.to_uint32()?;
            {
                let all_tokens: Vec<u32> = input_tokens.to_vec();
                if repetition_penalty != 1.0 {
                    last_logits = apply_repetition_penalty(
                        &last_logits,
                        &all_tokens,
                        repetition_penalty,
                        Some(repetition_context_size),
                    )?;
                }
                if presence_penalty != 0.0 {
                    last_logits = apply_presence_penalty(
                        &last_logits,
                        &all_tokens,
                        presence_penalty,
                        Some(presence_context_size),
                    )?;
                }
                if frequency_penalty != 0.0 {
                    last_logits = apply_frequency_penalty(
                        &last_logits,
                        &all_tokens,
                        frequency_penalty,
                        Some(frequency_context_size),
                    )?;
                }
            }

            let (first_token, first_logprobs) = if return_logprobs {
                let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
                (tok, Some(lp))
            } else {
                (sample(&last_logits, Some(sampling_config))?, None)
            };
            first_token.eval();

            item_kv_keys.push(kv_keys);
            item_kv_values.push(kv_values);
            item_rope_deltas.push(rope_deltas);
            item_cache_idxs.push(cache_idx);
            item_first_tokens.push(first_token);
            item_first_logprobs.push(first_logprobs);
            item_all_tokens.push(input_tokens.to_vec());
        }

        // === STEP 2: Merge KV caches into batch ===
        let max_cache_idx = *item_cache_idxs.iter().max().unwrap();
        let mut left_padding_values: Vec<i32> = Vec::with_capacity(batch_size);

        for &cache_idx in &item_cache_idxs {
            left_padding_values.push(max_cache_idx - cache_idx);
        }

        // Left-pad and stack KV caches per layer
        let mut batch_kv_keys: Vec<Option<MxArray>> = Vec::with_capacity(num_layers);
        let mut batch_kv_values: Vec<Option<MxArray>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let mut padded_keys: Vec<MxArray> = Vec::with_capacity(batch_size);
            let mut padded_values: Vec<MxArray> = Vec::with_capacity(batch_size);

            for i in 0..batch_size {
                let k = &item_kv_keys[i][layer];
                let v = &item_kv_values[i][layer];
                let pad_amount = left_padding_values[i];

                if pad_amount > 0 {
                    // KV shape: [1, heads, seq, dim] - pad along axis 2
                    let k_shape = k.shape()?;
                    let heads = k_shape[1];
                    let dim = k_shape[3];
                    let pad_zeros =
                        MxArray::zeros(&[1, heads, pad_amount as i64, dim], Some(k.dtype()?))?;
                    let padded_k = MxArray::concatenate_many(vec![&pad_zeros, k], Some(2))?;
                    let padded_v = MxArray::concatenate_many(vec![&pad_zeros, v], Some(2))?;
                    // Slice to max_cache_idx to ensure uniform size
                    let padded_k = padded_k.slice_axis(2, 0, max_cache_idx as i64)?;
                    let padded_v = padded_v.slice_axis(2, 0, max_cache_idx as i64)?;
                    padded_keys.push(padded_k);
                    padded_values.push(padded_v);
                } else {
                    // Slice to max_cache_idx
                    let k_trimmed = k.slice_axis(2, 0, max_cache_idx as i64)?;
                    let v_trimmed = v.slice_axis(2, 0, max_cache_idx as i64)?;
                    padded_keys.push(k_trimmed);
                    padded_values.push(v_trimmed);
                }
            }

            // Stack along batch dim: [batch, heads, max_cache_idx, dim]
            let key_refs: Vec<&MxArray> = padded_keys.iter().collect();
            let value_refs: Vec<&MxArray> = padded_values.iter().collect();
            let stacked_keys = MxArray::concatenate_many(key_refs, Some(0))?;
            let stacked_values = MxArray::concatenate_many(value_refs, Some(0))?;

            batch_kv_keys.push(Some(stacked_keys));
            batch_kv_values.push(Some(stacked_values));
        }

        // Eval merged KV caches
        for k in batch_kv_keys.iter().flatten() {
            k.eval();
        }
        for v in batch_kv_values.iter().flatten() {
            v.eval();
        }
        clear_cache();

        // === STEP 3: Batched decode loop ===
        let mut generated_tokens: Vec<Vec<u32>> =
            vec![Vec::with_capacity(max_new_tokens as usize); batch_size];
        let mut generated_logprobs: Vec<Vec<f32>> = vec![Vec::new(); batch_size];
        let mut finish_reasons: Vec<Option<String>> = vec![None; batch_size];

        // Track active batch items
        let mut active_indices: Vec<usize> = (0..batch_size).collect();
        let mut current_cache_idx = max_cache_idx;

        // Build initial token + logprobs arrays from prefill results
        let first_token_refs: Vec<&MxArray> = item_first_tokens.iter().collect();
        let mut current_tokens = MxArray::stack(first_token_refs, Some(0))?;

        // Build initial logprobs: stack per-item logprob distributions [batch, vocab]
        let mut current_logprobs: Option<Vec<Option<MxArray>>> = if return_logprobs {
            Some(item_first_logprobs)
        } else {
            None
        };

        // Get read lock for batched decode
        let lm_guard = lm
            .read()
            .map_err(|_| Error::new(Status::GenericFailure, "Failed to acquire LM read lock"))?;

        for step in 0..max_new_tokens {
            if active_indices.is_empty() {
                break;
            }

            // --- Phase 1: Build next step's graph ---
            // Embed current tokens and run forward pass to get next logits
            let active_batch_size = active_indices.len() as i64;

            let embed_tokens: Vec<u32> = {
                current_tokens.eval();
                let tok_vals = current_tokens.to_int32()?;
                active_indices
                    .iter()
                    .enumerate()
                    .map(|(local_idx, _)| tok_vals[local_idx] as u32)
                    .collect()
            };
            let embed_input = MxArray::from_uint32(&embed_tokens, &[active_batch_size, 1])?;
            let token_embeds = lm_guard.get_embedding_layer().forward(&embed_input)?;

            // Build position_ids [3, active_batch, 1] for mRoPE decode
            let mut pos_data: Vec<f32> = Vec::with_capacity(3 * active_indices.len());
            for &global_idx in &active_indices {
                let pos = (current_cache_idx as i64 - left_padding_values[global_idx] as i64
                    + item_rope_deltas[global_idx]) as f32;
                pos_data.push(pos);
            }
            let pos_1d = MxArray::from_float32(&pos_data, &[1, active_batch_size, 1])?;
            let position_ids = MxArray::tile(&pos_1d, &[3, 1, 1])?;

            // Build left_padding for active items
            let active_left_padding: Vec<i32> = active_indices
                .iter()
                .map(|&idx| left_padding_values[idx])
                .collect();
            let left_padding_active =
                MxArray::from_int32(&active_left_padding, &[active_batch_size])?;

            // Batched forward
            let (logits, new_kv_keys, new_kv_values, new_cache_idx) = {
                let _stream_ctx = StreamContext::new(generation_stream);
                lm_guard.forward_fused_batched(
                    &token_embeds,
                    &position_ids,
                    &left_padding_active,
                    &batch_kv_keys,
                    &batch_kv_values,
                    current_cache_idx,
                )?
            };

            batch_kv_keys = new_kv_keys;
            batch_kv_values = new_kv_values;
            current_cache_idx = new_cache_idx;

            // Sample next tokens [active_batch, 1, vocab] -> [active_batch, vocab]
            let next_logits = logits.squeeze(Some(&[1]))?;

            // Apply repetition penalty per item
            let mut next_logits = if repetition_penalty != 1.0 {
                let mut rows = Vec::with_capacity(active_indices.len());
                for (local_idx, &global_idx) in active_indices.iter().enumerate() {
                    let row = next_logits.slice_axis(0, local_idx as i64, local_idx as i64 + 1)?;
                    let row = row.squeeze(Some(&[0]))?;
                    let penalized = apply_repetition_penalty(
                        &row,
                        &item_all_tokens[global_idx],
                        repetition_penalty,
                        Some(repetition_context_size),
                    )?;
                    rows.push(penalized);
                }
                let row_refs: Vec<&MxArray> = rows.iter().collect();
                MxArray::stack(row_refs, Some(0))?
            } else {
                next_logits
            };
            if presence_penalty != 0.0 {
                let mut rows = Vec::with_capacity(active_indices.len());
                for (local_idx, &global_idx) in active_indices.iter().enumerate() {
                    let row = next_logits.slice_axis(0, local_idx as i64, local_idx as i64 + 1)?;
                    let row = row.squeeze(Some(&[0]))?;
                    rows.push(apply_presence_penalty(
                        &row,
                        &item_all_tokens[global_idx],
                        presence_penalty,
                        Some(presence_context_size),
                    )?);
                }
                let row_refs: Vec<&MxArray> = rows.iter().collect();
                next_logits = MxArray::stack(row_refs, Some(0))?;
            }
            if frequency_penalty != 0.0 {
                let mut rows = Vec::with_capacity(active_indices.len());
                for (local_idx, &global_idx) in active_indices.iter().enumerate() {
                    let row = next_logits.slice_axis(0, local_idx as i64, local_idx as i64 + 1)?;
                    let row = row.squeeze(Some(&[0]))?;
                    rows.push(apply_frequency_penalty(
                        &row,
                        &item_all_tokens[global_idx],
                        frequency_penalty,
                        Some(frequency_context_size),
                    )?);
                }
                let row_refs: Vec<&MxArray> = rows.iter().collect();
                next_logits = MxArray::stack(row_refs, Some(0))?;
            }

            // Sample next tokens (with logprobs if requested)
            let (mut next_tokens, mut next_logprobs) = if return_logprobs {
                let (tok, lp) = sample_and_logprobs(&next_logits, Some(sampling_config))?;
                (tok, Some(lp))
            } else {
                (sample(&next_logits, Some(sampling_config))?, None)
            };

            // --- Phase 2: Extract CURRENT tokens and CURRENT logprobs (aligned pair) ---
            current_tokens.eval();
            let token_values = current_tokens.to_int32()?;

            // Extract logprobs for current tokens
            if return_logprobs && let Some(ref lp_vec) = current_logprobs {
                for (local_idx, &global_idx) in active_indices.iter().enumerate() {
                    let token_value = token_values[local_idx] as u32;
                    if let Some(ref lp) = lp_vec[global_idx] {
                        lp.eval();
                        let token_logprob = lp.item_at_float32(token_value as usize)?;
                        generated_logprobs[global_idx].push(token_logprob);
                    }
                }
            }

            // Track deactivations
            let mut to_deactivate: Vec<(usize, String)> = Vec::new();

            for (local_idx, &global_idx) in active_indices.iter().enumerate() {
                let token_value = token_values[local_idx] as u32;
                generated_tokens[global_idx].push(token_value);
                item_all_tokens[global_idx].push(token_value);

                // Check EOS
                if token_value == eos_token_id as u32 {
                    to_deactivate.push((local_idx, "stop".to_string()));
                }
            }

            // Build filter indices
            let positions_to_remove: std::collections::HashSet<usize> =
                to_deactivate.iter().map(|(idx, _)| *idx).collect();
            let old_batch_size = active_indices.len();
            let filter_indices: Vec<i32> = (0..old_batch_size)
                .filter(|i| !positions_to_remove.contains(i))
                .map(|i| i as i32)
                .collect();

            // Apply deactivations (reverse order)
            to_deactivate.sort_by(|a, b| b.0.cmp(&a.0));
            for (local_idx, reason) in to_deactivate {
                let global_idx = active_indices[local_idx];
                finish_reasons[global_idx] = Some(reason);
                active_indices.remove(local_idx);
            }

            if active_indices.is_empty() {
                break;
            }

            // Filter KV caches, tokens, and logprobs if batch shrank
            if filter_indices.len() < old_batch_size {
                let indices_array =
                    MxArray::from_int32(&filter_indices, &[filter_indices.len() as i64])?;
                for layer in 0..num_layers {
                    if let Some(ref keys) = batch_kv_keys[layer] {
                        batch_kv_keys[layer] = Some(keys.take(&indices_array, 0)?);
                    }
                    if let Some(ref values) = batch_kv_values[layer] {
                        batch_kv_values[layer] = Some(values.take(&indices_array, 0)?);
                    }
                }
                // Filter next_tokens to match the shrunken active set.
                // Without this, tok_vals[local_idx] on the next iteration reads
                // tokens from deactivated items, corrupting remaining outputs.
                next_tokens = next_tokens.take(&indices_array, 0)?;
                if let Some(ref lp) = next_logprobs {
                    next_logprobs = Some(lp.take(&indices_array, 0)?);
                }
            }

            // Advance: next becomes current
            current_tokens = next_tokens;

            // Update per-item logprobs for next iteration
            if return_logprobs && let Some(ref next_lp) = next_logprobs {
                let mut new_lp_vec: Vec<Option<MxArray>> = vec![None; batch_size];
                for (local_idx, &global_idx) in active_indices.iter().enumerate() {
                    let row = next_lp.slice_axis(0, local_idx as i64, local_idx as i64 + 1)?;
                    let row = row.squeeze(Some(&[0]))?;
                    new_lp_vec[global_idx] = Some(row);
                }
                current_logprobs = Some(new_lp_vec);
            }

            // Periodic cleanup
            if step > 0 && step % 256 == 0 {
                clear_cache();
            }
        }

        // Set finish reason for items still active
        for &global_idx in &active_indices {
            if finish_reasons[global_idx].is_none() {
                finish_reasons[global_idx] = Some("length".to_string());
            }
        }

        // Build results
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let tokens = &generated_tokens[i];
            let tokens_array = MxArray::from_uint32(tokens, &[tokens.len() as i64])?;
            let logprobs_array = if return_logprobs {
                MxArray::from_float32(
                    &generated_logprobs[i],
                    &[generated_logprobs[i].len() as i64],
                )?
            } else {
                MxArray::from_float32(&[], &[0])?
            };

            results.push(GenerationResult {
                text: String::new(),
                tokens: tokens_array,
                logprobs: logprobs_array,
                finish_reason: finish_reasons[i]
                    .clone()
                    .unwrap_or_else(|| "length".to_string()),
                num_tokens: tokens.len(),
                first_token_elapsed_ms: None,
            });
        }

        Ok(results)
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
    /// const result = await model.chat(messages, { images: [readFileSync('./image.jpg')] });
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

    /// Regression test: batch decode must filter next_tokens when items are deactivated.
    ///
    /// Before the fix, after EOS filtering removed batch items from active_indices
    /// and KV caches, next_tokens was NOT filtered. On the next iteration,
    /// tok_vals[local_idx] would read tokens from deactivated items, causing
    /// cross-item contamination.
    ///
    /// This test simulates the exact control flow from generate_batch's decode loop:
    ///   1. Start with 3 active items producing tokens [10, 20, 30]
    ///   2. Item 1 (middle) hits EOS → deactivated
    ///   3. Verify remaining items [0, 2] read correct tokens [10, 30] not [10, 20]
    #[test]
    fn test_batch_decode_eos_filter_tokens() {
        // Simulate next_tokens = [tok_A, tok_B, tok_C] from batch of 3
        let next_tokens = MxArray::from_int32(&[10, 20, 30], &[3]).unwrap();

        // Item B (local_idx=1) hits EOS. filter_indices keeps items 0 and 2.
        let filter_indices = MxArray::from_int32(&[0, 2], &[2]).unwrap();

        // The fix: filter next_tokens before assigning to current_tokens
        let filtered = next_tokens.take(&filter_indices, 0).unwrap();
        filtered.eval();
        let vals = filtered.to_int32().unwrap();

        // After filtering, position 0 should be tok_A (10), position 1 should be tok_C (30)
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], 10, "local_idx=0 should map to item A's token");
        assert_eq!(
            vals[1], 30,
            "local_idx=1 should map to item C's token (not B's)"
        );
    }

    /// Regression test: KV cache filtering must stay aligned with token filtering.
    ///
    /// Simulates the batch decode scenario where KV caches [batch, heads, seq, dim]
    /// and next_tokens [batch] must both be filtered by the same indices when
    /// items are deactivated.
    #[test]
    fn test_batch_decode_kv_and_token_alignment() {
        // Simulate KV cache: [3 items, 2 heads, 4 seq positions, 2 dim]
        // Each item has distinguishable values: item0=1.0, item1=2.0, item2=3.0
        let kv = MxArray::from_float32(
            &[
                // item 0: all 1s
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                // item 1: all 2s
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                // item 2: all 3s
                3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            ],
            &[3, 2, 4, 2],
        )
        .unwrap();
        let tokens = MxArray::from_int32(&[100, 200, 300], &[3]).unwrap();

        // Remove item 1 (middle)
        let filter = MxArray::from_int32(&[0, 2], &[2]).unwrap();
        let filtered_kv = kv.take(&filter, 0).unwrap();
        let filtered_tokens = tokens.take(&filter, 0).unwrap();

        filtered_kv.eval();
        filtered_tokens.eval();

        let kv_shape: Vec<i64> = filtered_kv.shape().unwrap().to_vec();
        assert_eq!(
            kv_shape,
            vec![2, 2, 4, 2],
            "KV batch dim should shrink to 2"
        );

        let tok_vals: Vec<i32> = filtered_tokens.to_int32().unwrap().to_vec();
        assert_eq!(
            tok_vals,
            vec![100, 300],
            "tokens should match filtered KV rows"
        );

        // Verify KV content: row 0 should be item 0 (1.0), row 1 should be item 2 (3.0)
        let kv_data = filtered_kv.to_float32().unwrap();
        assert_eq!(kv_data[0], 1.0, "KV row 0 should be item 0");
        assert_eq!(kv_data[16], 3.0, "KV row 1 should be item 2 (not item 1)");
    }
}
