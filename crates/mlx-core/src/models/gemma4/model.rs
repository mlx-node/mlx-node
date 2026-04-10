use std::collections::HashMap;
use std::sync::Arc;

use napi::Either;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::{DType, MxArray};
use crate::model_thread::ResponseTx;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};

use super::image_processor::{Gemma4ImageProcessor, ProcessedGemma4Image};
use super::vision::{Gemma4MultimodalEmbedder, Gemma4VisionModel};

/// Convert a JSON value to Gemma4's tool-call DSL format.
/// Strings → <|"|>str<|"|>, numbers/bools → bare, objects/arrays → recursive.
fn format_gemma4_value(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => format!("<|\"|>{}<|\"|>", s),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_gemma4_value).collect();
            format!("[{}]", items.join(","))
        }
        serde_json::Value::Object(map) => {
            let mut pairs: Vec<(String, String)> = map
                .iter()
                .map(|(k, v)| (k.clone(), format_gemma4_value(v)))
                .collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            let inner: Vec<String> = pairs.iter().map(|(k, v)| format!("{}:{}", k, v)).collect();
            format!("{{{}}}", inner.join(","))
        }
    }
}

/// Convert JSON arguments string to Gemma4 tool-call DSL.
/// Returns the inner key:value pairs (without outer braces).
fn json_args_to_gemma4_dsl(json_str: &str) -> String {
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str(json_str) {
        let mut pairs: Vec<(String, String)> = map
            .iter()
            .map(|(k, v)| (k.clone(), format_gemma4_value(v)))
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        pairs
            .iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect::<Vec<_>>()
            .join(",")
    } else {
        // If not valid JSON object, pass through as-is
        json_str.to_string()
    }
}

/// Strip Gemma4 control tokens from user-supplied content to prevent prompt injection.
///
/// Removes all Gemma4 delimiter tokens that could allow a malicious message to
/// hijack the turn structure or inject synthetic tool calls/responses.
fn escape_gemma4_content(s: &str) -> String {
    s.replace("<|turn>", "")
        .replace("<turn|>", "")
        .replace("<|tool_call>", "")
        .replace("<tool_call|>", "")
        .replace("<|tool_response>", "")
        .replace("<tool_response|>", "")
        .replace("<|tool>", "")
        .replace("<tool|>", "")
        .replace("<|channel>", "")
        .replace("<channel|>", "")
        .replace("<|think|>", "")
}

use super::config::Gemma4Config;
use super::decoder_layer::Gemma4DecoderLayer;
use super::layer_cache::Gemma4LayerCache;
use tracing::{debug, info};

/// Gemma4 generation configuration.
#[napi(object)]
pub struct Gemma4ChatConfig {
    pub max_new_tokens: Option<i32>,
    pub temperature: Option<f64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    /// Enable thinking mode. `None` = let the template decide,
    /// `Some(false)` = disabled, `Some(true)` = enabled.
    pub enable_thinking: Option<bool>,
}

/// Gemma4 chat result.
#[napi(object)]
pub struct Gemma4ChatResult {
    pub text: String,
    pub num_tokens: u32,
    pub finish_reason: String,
    /// Performance metrics (always present).
    pub performance: Option<crate::profiling::PerformanceMetrics>,
}

/// PLE (Per-Layer Embeddings) model-level components.
///
/// Provides per-layer token-level information to each decoder layer.
/// Present in E2B (2.3B) and E4B (4.5B) models.
pub(crate) struct PleComponents {
    /// Embedding table: [vocab_size_per_layer_input, num_layers * ple_dim]
    pub embed_tokens_per_layer: Embedding,
    /// Projection: [hidden_size, num_layers * ple_dim]
    pub per_layer_model_projection: Linear,
    /// Norm applied per ple_dim slice: weight shape [ple_dim]
    pub per_layer_projection_norm: RMSNorm,
    /// Scale factor: 2.0^(-0.5) = 1/sqrt(2) for per_layer_input_scale
    pub per_layer_input_scale: f64,
    /// Scale factor: hidden_size^(-0.5) for per_layer_model_projection_scale
    pub per_layer_model_projection_scale: f64,
    /// Dimension of per-layer embeddings
    pub ple_dim: i32,
    /// Number of layers
    pub num_layers: i32,
    /// PLE vocab size (may be smaller than main vocab_size)
    pub vocab_size_per_layer_input: i32,
}

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership.
pub(crate) struct Gemma4Inner {
    pub(crate) config: Gemma4Config,
    pub(crate) embed_tokens: Embedding,
    pub(crate) layers: Vec<Gemma4DecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) lm_head: Option<Linear>,
    /// Pre-transposed embedding weight for tied lm_head: [hidden_size, vocab_size].
    /// Only populated when tie_word_embeddings=true.
    pub(crate) embed_weight_t: Option<MxArray>,
    pub(crate) ple: Option<PleComponents>,
    // Vision components (None for text-only models)
    pub(crate) vision_tower: Option<Gemma4VisionModel>,
    pub(crate) embed_vision: Option<Gemma4MultimodalEmbedder>,
    pub(crate) image_processor: Option<Gemma4ImageProcessor>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    pub(crate) model_id: u64,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Gemma4Cmd {
    Chat {
        messages: Vec<ChatMessage>,
        config: Gemma4ChatConfig,
        processed_images: Vec<ProcessedGemma4Image>,
        reply: ResponseTx<Gemma4ChatResult>,
    },
}

/// Gemma 4 dense language model.
///
/// Supports E2B (2.3B), E4B (4.5B), and 31B variants.
/// Features: hybrid attention (sliding + global), GeGLU MLP, logit softcapping,
/// embedding scaling, and optional per-layer embeddings.
///
/// All model state lives on a dedicated OS thread. NAPI methods dispatch
/// commands via channels and await responses.
#[napi]
pub struct Gemma4Model {
    pub(crate) thread: crate::model_thread::ModelThread<Gemma4Cmd>,
    pub(crate) model_id: u64,
    pub(crate) image_processor: Option<Gemma4ImageProcessor>,
}

static MODEL_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Serializes compiled C++ forward calls across model instances.
/// Only matters if two Gemma4 models are loaded simultaneously (rare).
static COMPILED_FORWARD_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

impl Gemma4Inner {
    /// Create a new Gemma4Inner with empty (uninitialized) weights.
    pub(crate) fn new(config: Gemma4Config) -> Result<Self> {
        let num_layers = config.num_hidden_layers as usize;
        let hidden_size = config.hidden_size as u32;
        let vocab_size = config.vocab_size as u32;

        let embed_tokens = Embedding::new(vocab_size, hidden_size)?;
        let final_norm = RMSNorm::new(hidden_size, Some(config.rms_norm_eps))?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(Linear::new(hidden_size, vocab_size, Some(false))?)
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(Gemma4DecoderLayer::new(&config, i)?);
        }

        // Initialize PLE model-level components if enabled
        let ple = if config.per_layer_input_embeds {
            let ple_dim = config.ple_dim();
            let vocab_ple = config.vocab_size_per_layer_input.unwrap_or(0);
            if ple_dim > 0 && vocab_ple > 0 {
                let total_ple_dim = (num_layers as i32) * ple_dim;
                Some(PleComponents {
                    embed_tokens_per_layer: Embedding::new(vocab_ple as u32, total_ple_dim as u32)?,
                    per_layer_model_projection: Linear::new(
                        hidden_size,
                        total_ple_dim as u32,
                        Some(false),
                    )?,
                    per_layer_projection_norm: RMSNorm::new(
                        ple_dim as u32,
                        Some(config.rms_norm_eps),
                    )?,
                    per_layer_input_scale: 2.0_f64.powf(-0.5),
                    per_layer_model_projection_scale: (config.hidden_size as f64).powf(-0.5),
                    ple_dim,
                    num_layers: num_layers as i32,
                    vocab_size_per_layer_input: vocab_ple,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Initialize vision components if vision_config is present
        let (vision_tower, embed_vision, image_processor) = if let Some(ref vc) =
            config.vision_config
        {
            let vt = Gemma4VisionModel::new(vc)?;
            let ev =
                Gemma4MultimodalEmbedder::new(vc.hidden_size, config.hidden_size, vc.rms_norm_eps)?;
            let ip = Gemma4ImageProcessor::new(
                vc.patch_size,
                vc.default_output_length,
                vc.pooling_kernel_size,
            );
            (Some(vt), Some(ev), Some(ip))
        } else {
            (None, None, None)
        };

        let model_id = MODEL_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(Self {
            config,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            embed_weight_t: None,
            ple,
            vision_tower,
            embed_vision,
            image_processor,
            tokenizer: None,
            model_id,
        })
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Synchronous chat implementation. Runs on the dedicated model thread.
    fn chat_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: Gemma4ChatConfig,
        processed_images: Vec<ProcessedGemma4Image>,
    ) -> Result<Gemma4ChatResult> {
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        let has_images = !processed_images.is_empty();
        let sampling_config = make_sampling_config(&config, &self.config);
        let enable_thinking = config.enable_thinking;
        let eos_ids = self.config.eos_token_ids.clone();

        // Try the tokenizer's chat template if available (handles role mapping,
        // special tokens, and variant-specific formatting automatically).
        // Fall back to manual Gemma4 format if no template was loaded.
        let tokens = if tokenizer.has_chat_template() {
            tokenizer.apply_chat_template_sync(
                &messages,
                Some(true),      // add_generation_prompt
                None,            // no tools
                enable_thinking, // None = template default
            )?
        } else {
            // Manual fallback: thinking control requires a chat template
            if enable_thinking == Some(true) {
                return Err(Error::from_reason(
                    "enable_thinking=true requires a chat template (not found in tokenizer_config.json or chat_template.jinja)",
                ));
            }
            // Manual Gemma4 format matching the canonical template.
            // Role mapping: "assistant" → "model", "developer" → "system".
            // Tool calls serialized as <|tool_call>call:name{args}<tool_call|>.
            // Tool responses wrapped in <|tool_response>...<tool_response|>.
            // BOS prepended explicitly (matching {{ bos_token }} in template).
            let mut prompt_text = String::from("<bos>");
            for msg in &messages {
                let role = match msg.role.as_str() {
                    "assistant" => "model",
                    "developer" => "system",
                    other => other,
                };

                // All roles (including "tool") use the same <|turn>role\n...<turn|>\n format.
                // This matches the canonical tokenizer behavior verified against HF.
                {
                    prompt_text.push_str(&format!("<|turn>{}\n", role));

                    // Emit tool calls for assistant/model messages
                    if let Some(ref tool_calls) = msg.tool_calls {
                        for tc in tool_calls {
                            prompt_text.push_str(&format!(
                                "<|tool_call>call:{}{{{}}}<tool_call|>",
                                tc.name,
                                json_args_to_gemma4_dsl(&escape_gemma4_content(&tc.arguments))
                            ));
                        }
                    }

                    // Emit content (sanitized to prevent control-token injection)
                    prompt_text.push_str(&escape_gemma4_content(&msg.content));
                    prompt_text.push_str("<turn|>\n");
                }
            }
            prompt_text.push_str("<|turn>model\n");
            tokenizer.encode_sync(&prompt_text, Some(false))?
        };

        // Expand image tokens if images are present.
        // Gemma4 uses: <|image>  (BOI) + <|image|> × num_soft_tokens + <image|> (EOI)
        // The chat template inserts a single <|image|> per image; we expand it here.
        let tokens = if has_images && !processed_images.is_empty() {
            let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
            let boi_token_id = self.config.boi_token_id.unwrap_or(255999) as u32;
            let eoi_token_id = self.config.eoi_token_id.unwrap_or(258882) as u32;
            expand_image_tokens(
                &tokens,
                &processed_images,
                image_token_id,
                boi_token_id,
                eoi_token_id,
            )
        } else {
            tokens
        };

        // Create prompt tensor
        let token_arr: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, tokens.len() as i64])?;

        // Initialize caches
        let mut caches = init_caches_for_config(&self.config);

        // Create dedicated generation stream for GPU scheduling.
        let generation_stream = Stream::new(DeviceType::Gpu);

        // Wired memory: pin model weights in GPU memory (prevents paging for large models).
        // Uses usize::MAX to always set limit to max_recommended_working_set_size.
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();
        let prompt_token_count = tokens.len();

        // Vision prefill: if images present, build merged embeddings
        // (text embeddings with vision features scattered at image_token positions)
        let vision_embeds: Option<MxArray> = if has_images
            && !processed_images.is_empty()
            && let Some(ref vt) = self.vision_tower
            && let Some(ref ev) = self.embed_vision
        {
            let image_token_id = self.config.image_token_id.unwrap_or(258880);

            // Run vision tower on each image and collect features
            let mut all_features: Vec<MxArray> = Vec::new();
            for proc in &processed_images {
                let features = vt.forward(&proc.pixel_values)?;
                let projected = ev.forward(&features)?;
                all_features.push(projected);
            }

            // Concatenate all image features: [1, total_soft_tokens, hidden_size]
            let image_features = if all_features.len() == 1 {
                all_features.remove(0)
            } else {
                let refs: Vec<&MxArray> = all_features.iter().collect();
                MxArray::concatenate_many(refs, Some(1))?
            };

            // Build text embeddings
            let text_embeds = self.embed_tokens.forward(&prompt)?;
            let text_embeds = text_embeds.mul_scalar((self.config.hidden_size as f64).sqrt())?;

            // Cast image features to text embedding dtype
            let embed_dtype = text_embeds.dtype()?;
            let image_features = image_features.astype(embed_dtype)?;

            // masked_scatter: replace image_token positions with vision features
            let image_token = MxArray::scalar_int(image_token_id)?;
            let image_mask = prompt.equal(&image_token)?;

            // Validate: number of True positions in mask must match vision feature count
            let mask_count_arr = image_mask.astype(DType::Int32)?.sum(None, None)?;
            mask_count_arr.eval();
            let mask_count = mask_count_arr.item_at_int32(0)? as i64;
            let feature_count = image_features.shape_at(1)?;
            if mask_count != feature_count {
                return Err(Error::new(
                    Status::GenericFailure,
                    format!(
                        "Image token count ({mask_count}) does not match vision feature count ({feature_count}). \
                         Check that image token expansion produced the correct number of tokens."
                    ),
                ));
            }

            let image_mask_expanded = image_mask.expand_dims(-1)?;
            let image_mask_expanded = image_mask_expanded.broadcast_to(&text_embeds.shape()?)?;

            let merged = masked_scatter(&text_embeds, &image_mask_expanded, &image_features)?;
            Some(merged)
        } else {
            None
        };

        // Prefill: process tokens [0:N-1] through body only (no lm_head),
        // then run last token through full forward to get logits.
        // Matches mlx-lm generate_step pattern.
        {
            let _stream_ctx = StreamContext::new(generation_stream);
            if let Some(ref embeds) = vision_embeds {
                // Vision path: prefill with merged embeddings
                prefill_body_gemma4_with_embeds(
                    &prompt,
                    embeds,
                    &self.embed_tokens,
                    &self.layers,
                    &mut caches,
                    &self.final_norm,
                    self.ple.as_ref(),
                    &self.config,
                )?;
            } else {
                // Text-only path
                prefill_body_gemma4(
                    &prompt,
                    &self.embed_tokens,
                    &self.layers,
                    &mut caches,
                    &self.final_norm,
                    self.ple.as_ref(),
                    &self.config,
                )?;
            }
        }
        eval_gemma4_caches(&caches);

        // Last token → logits
        let last_token = prompt.slice_axis(1, tokens.len() as i64 - 1, tokens.len() as i64)?;
        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            forward_inner(
                &last_token,
                &self.embed_tokens,
                &self.layers,
                &mut caches,
                &self.final_norm,
                &self.lm_head,
                self.embed_weight_t.as_ref(),
                self.ple.as_ref(),
                &self.config,
            )?
        };
        let logits = logits.squeeze(Some(&[1]))?;
        let y = sample_next_token(&logits, sampling_config)?;
        y.eval();
        eval_gemma4_caches(&caches);

        // Mark first token time (TTFT = time to first token)
        let first_token_instant = std::time::Instant::now();

        // Decode loop — matches mlx-lm generate.py pattern:
        // 1. Build lazy graph per step via forward_inner
        // 2. async_eval the output token (caches materialize through dependency graph)
        // 3. Double-buffer: build step N+1 while GPU executes step N
        //
        // Set GEMMA4_USE_COMPILE=1 to use the old compiled C++ path for A/B testing.
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = "length".to_string();

        let use_compiled = std::env::var("GEMMA4_USE_COMPILE").is_ok()
            && self.config.num_kv_shared_layers.is_none_or(|n| n <= 0)
            && unsafe { mlx_sys::mlx_qwen35_get_model_id() } == self.model_id;

        if use_compiled {
            // Legacy compiled C++ path (opt-in via GEMMA4_USE_COMPILE=1)
            let _compiled_guard = COMPILED_FORWARD_MUTEX.lock().unwrap();
            let mut cache_arrays_owned: Vec<MxArray> = Vec::with_capacity(caches.len() * 2);
            for (layer_idx, cache) in caches.iter().enumerate() {
                let (k, v) = cache.get_cached_kv().ok_or_else(|| {
                    Error::from_reason(format!(
                        "Compiled Gemma4 decode expected cache for layer {} after prefill",
                        layer_idx
                    ))
                })?;
                cache_arrays_owned.push(k);
                cache_arrays_owned.push(v);
            }
            let mut cache_ptrs: Vec<*mut mlx_sys::mlx_array> =
                cache_arrays_owned.iter().map(|a| a.as_raw_ptr()).collect();

            let layer_types_i32: Vec<i32> = (0..self.config.num_hidden_layers as usize)
                .map(|i| if self.config.is_global_layer(i) { 1 } else { 0 })
                .collect();

            let max_kv_len =
                (tokens.len() as i32 + max_new_tokens).min(self.config.max_position_embeddings);

            unsafe {
                mlx_sys::mlx_gemma4_init_from_prefill(
                    self.config.num_hidden_layers,
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.effective_kv_heads(true),
                    self.config.effective_head_dim(true),
                    self.config.rope_theta as f32,
                    self.config.rope_local_base_freq as f32,
                    self.config.partial_rotary_factor as f32,
                    self.config.rms_norm_eps as f32,
                    self.config.sliding_window,
                    if self.config.tie_word_embeddings {
                        1
                    } else {
                        0
                    },
                    max_kv_len,
                    1,
                    self.config.num_experts.unwrap_or(0),
                    self.config.top_k_experts.unwrap_or(0),
                    self.config.moe_intermediate_size.unwrap_or(0),
                    self.config.intermediate_size,
                    self.config.final_logit_softcapping.unwrap_or(0.0) as f32,
                    layer_types_i32.as_ptr(),
                    layer_types_i32.len() as i32,
                    cache_ptrs.as_mut_ptr(),
                    tokens.len() as i32,
                );
            }

            let embed_weight = self.embed_tokens.get_weight();
            let mut current_y = y;
            for step in 0..max_new_tokens {
                let next_y = if step + 1 < max_new_tokens {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let next_ids = current_y.reshape(&[1, 1])?;
                    let logits = forward_gemma4_cpp(&next_ids, &embed_weight)?;
                    let next_token = sample_next_token(&logits, sampling_config)?;
                    eval_token_and_gemma4_caches(&next_token);
                    Some(next_token)
                } else {
                    None
                };

                let token_id = current_y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                if eos_ids.contains(&(token_id as i32)) {
                    finish_reason = "stop".to_string();
                    break;
                }
                if let Some(next_token) = next_y {
                    current_y = next_token;
                } else {
                    break;
                }
                if (step + 1) % 256 == 0 {
                    crate::array::synchronize_and_clear_cache();
                }
            }
            unsafe {
                mlx_sys::mlx_gemma4_reset();
            }
        } else {
            // Default: lazy eval decode (matches mlx-lm pattern)
            //
            // Double-buffered: build step N+1's graph while GPU executes step N.
            // Cache mutations (slice_assign_axis_inplace) are lazy side effects
            // in the computation graph — evaluating the token implicitly
            // materializes caches (no explicit cache eval needed during decode).
            //
            // Pattern from mlx-lm generate.py:
            //   mx.async_eval(next_y)   # fire and forget
            //   if n == 0: mx.eval(y)   # sync only for TTFT
            let mut current_y = y;
            for step in 0..max_new_tokens {
                let next_y = if step + 1 < max_new_tokens {
                    let _stream_ctx = StreamContext::new(generation_stream);

                    let next_ids = current_y.reshape(&[1, 1])?;
                    let logits = forward_inner(
                        &next_ids,
                        &self.embed_tokens,
                        &self.layers,
                        &mut caches,
                        &self.final_norm,
                        &self.lm_head,
                        self.embed_weight_t.as_ref(),
                        self.ple.as_ref(),
                        &self.config,
                    )?;
                    let logits = logits.squeeze(Some(&[1]))?;
                    let next_token = sample_next_token(&logits, sampling_config)?;
                    MxArray::async_eval_arrays(&[&next_token]);
                    Some(next_token)
                } else {
                    None
                };

                let token_id = current_y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                if eos_ids.contains(&(token_id as i32)) {
                    finish_reason = "stop".to_string();
                    break;
                }
                if let Some(next_token) = next_y {
                    current_y = next_token;
                } else {
                    break;
                }

                if (step + 1) % 256 == 0 {
                    crate::array::clear_cache();
                }
            }
        }

        // Decode text
        let text = tokenizer.decode_sync(&generated_tokens, true)?;

        // Compute performance metrics
        let generation_end = std::time::Instant::now();
        let ttft_ms = first_token_instant
            .duration_since(generation_start)
            .as_secs_f64()
            * 1000.0;
        let decode_ms = generation_end
            .duration_since(first_token_instant)
            .as_secs_f64()
            * 1000.0;
        let gen_toks = generated_tokens.len() as f64;
        let mem_after = crate::array::get_active_memory();
        debug!(
            "[gemma4-chat] after generate: {:.2} GB active",
            mem_after / 1e9
        );

        let performance = Some(crate::profiling::PerformanceMetrics {
            ttft_ms,
            prefill_tokens_per_second: if ttft_ms > 0.0 {
                prompt_token_count as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
        });

        Ok(Gemma4ChatResult {
            text,
            num_tokens: generated_tokens.len() as u32,
            finish_reason,
            performance,
        })
    }
}

/// Command handler for the dedicated model thread.
pub(crate) fn handle_gemma4_cmd(inner: &mut Gemma4Inner, cmd: Gemma4Cmd) {
    match cmd {
        Gemma4Cmd::Chat {
            messages,
            config,
            processed_images,
            reply,
        } => {
            let result = inner.chat_sync(messages, config, processed_images);
            let _ = reply.send(result);
        }
    }
}

#[napi]
impl Gemma4Model {
    #[napi(constructor)]
    pub fn new(config: Gemma4Config) -> Result<Self> {
        let image_processor = config.vision_config.as_ref().map(|vc| {
            Gemma4ImageProcessor::new(
                vc.patch_size,
                vc.default_output_length,
                vc.pooling_kernel_size,
            )
        });

        let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
            move || {
                let inner = Gemma4Inner::new(config)?;
                let model_id = inner.model_id;
                Ok((inner, model_id))
            },
            handle_gemma4_cmd,
        );

        let model_id = init_rx
            .blocking_recv()
            .map_err(|_| napi::Error::from_reason("Model thread exited during init"))??;

        Ok(Self {
            thread,
            model_id,
            image_processor,
        })
    }

    #[napi]
    pub fn model_id(&self) -> u32 {
        self.model_id as u32
    }

    /// Load a Gemma4 model from a directory.
    #[napi]
    pub async fn load(model_path: String) -> Result<Gemma4Model> {
        Self::load_from_dir(&model_path).await
    }

    /// Chat with the model using a list of messages.
    #[napi]
    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<Gemma4ChatConfig>,
    ) -> Result<Gemma4ChatResult> {
        let config = config.unwrap_or(Gemma4ChatConfig {
            max_new_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            enable_thinking: None,
        });

        // Process images before sending command (Uint8Array is !Send)
        let all_images = extract_images_from_messages(&messages);
        let processed_images = if !all_images.is_empty() {
            let ip = self.image_processor.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "Images provided but model has no vision support (no vision_config in config.json)",
                )
            })?;
            let mut results = Vec::with_capacity(all_images.len());
            for img_bytes in &all_images {
                results.push(ip.process_bytes(img_bytes)?);
            }
            results
        } else {
            Vec::new()
        };

        crate::model_thread::send_and_await(&self.thread, |reply| Gemma4Cmd::Chat {
            messages,
            config,
            processed_images,
            reply,
        })
        .await
    }
}

/// How many layers to batch per eval during warmup.
///
/// Larger GPUs can handle bigger Metal command buffers before timing out,
/// but the timeout is nondeterministic (thermal state, system load).
/// Uses `max_recommended_working_set_size` (GPU memory) as proxy:
///   ≤128 GB → 1  (base / Pro / Max)
///   ≤384 GB → 2  (Ultra variants)
///   >384 GB → 4  (future hardware)
fn warmup_layer_batch_size() -> usize {
    let gb = crate::stream::WiredLimitContext::get_max_working_set_size() / (1 << 30);
    match gb {
        0..=128 => 1,
        129..=384 => 2,
        _ => 4,
    }
}

/// Single-token forward pass to trigger Metal shader compilation at load time.
/// Layers are eval'd in batches (sized by GPU capability) to keep Metal
/// command buffers under the timeout limit on cold shader cache.
pub(crate) fn warmup_forward(inner: &Gemma4Inner) -> Result<()> {
    let config = &inner.config;
    let batch = warmup_layer_batch_size();
    let mem_before = crate::array::get_active_memory();
    info!(
        "[warmup] layer batch size: {} (GPU mem: query complete)",
        batch
    );

    {
        let mut caches = init_caches_for_config(config);
        let dummy = MxArray::from_int32(&[1i32], &[1, 1])?;

        let mut h = inner.embed_tokens.forward(&dummy)?;
        h = h.mul_scalar((config.hidden_size as f64).sqrt())?;
        h.eval();

        for (i, layer) in inner.layers.iter().enumerate() {
            h = layer.forward(&h, None, Some(&mut caches[i]), None, false)?;
            if (i + 1) % batch == 0 || i + 1 == inner.layers.len() {
                h.eval();
            }
        }

        h = inner.final_norm.forward(&h)?;
        let logits = if let Some(ref head) = inner.lm_head {
            head.forward(&h)?
        } else if let Some(ref w_t) = inner.embed_weight_t {
            h.matmul(w_t)?
        } else {
            let weight = inner.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            h.matmul(&weight_t)?
        };
        logits.eval();
    }

    crate::array::synchronize_and_clear_cache();
    let mem_after = crate::array::get_active_memory();
    info!(
        "[warmup] memory: {:.2} GB → {:.2} GB (delta: {:.2} GB)",
        mem_before / 1e9,
        mem_after / 1e9,
        (mem_after - mem_before) / 1e9
    );

    Ok(())
}

fn init_caches_for_config(config: &Gemma4Config) -> Vec<Gemma4LayerCache> {
    let num_layers = config.num_hidden_layers as usize;
    let mut caches = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if config.is_global_layer(i) {
            caches.push(Gemma4LayerCache::new_global());
        } else {
            caches.push(Gemma4LayerCache::new_sliding(config.sliding_window));
        }
    }
    caches
}

fn make_sampling_config(
    config: &Gemma4ChatConfig,
    model_config: &Gemma4Config,
) -> Option<SamplingConfig> {
    let temp = config
        .temperature
        .or(model_config.default_temperature)
        .unwrap_or(0.0);
    if temp <= 0.0 {
        // Greedy: use a near-zero temperature for argmax-like behavior.
        // Cannot pass None because sample() defaults to temperature=1.0.
        return Some(SamplingConfig {
            temperature: Some(0.0),
            top_k: None,
            top_p: None,
            min_p: None,
        });
    }
    Some(SamplingConfig {
        temperature: Some(temp),
        top_k: config.top_k.or(model_config.default_top_k),
        top_p: config.top_p.or(model_config.default_top_p),
        min_p: config.min_p,
    })
}

fn sample_next_token(logits: &MxArray, config: Option<SamplingConfig>) -> Result<MxArray> {
    if is_greedy_sampling(config) {
        return logits.argmax(-1, Some(false));
    }
    sample(logits, config)
}

fn is_greedy_sampling(config: Option<SamplingConfig>) -> bool {
    config.is_some_and(|cfg| {
        cfg.temperature.unwrap_or(1.0) <= 0.0
            && cfg.top_k.is_none()
            && cfg.top_p.is_none()
            && cfg.min_p.is_none()
    })
}

/// Call the compiled C++ forward for a single Gemma4 decode step.
fn forward_gemma4_cpp(input_ids: &MxArray, embedding_weight: &MxArray) -> Result<MxArray> {
    let mut logits_ptr: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut cache_offset: i32 = 0;
    unsafe {
        mlx_sys::mlx_gemma4_forward(
            input_ids.as_raw_ptr(),
            embedding_weight.as_raw_ptr(),
            &mut logits_ptr,
            &mut cache_offset,
        );
    }
    if logits_ptr.is_null() {
        return Err(Error::from_reason("Gemma4 compiled forward returned null"));
    }
    MxArray::from_handle(logits_ptr, "gemma4_compiled_forward")
}

fn eval_token_and_gemma4_caches(next_token: &MxArray) {
    unsafe {
        mlx_sys::mlx_gemma4_eval_token_and_caches(next_token.as_raw_ptr());
    }
}

/// Transformer body: embedding through decoder layers and final norm.
///
/// Matches mlx-vlm `Gemma4TextModel.__call__`. Does NOT run lm_head or softcap.
/// Used by chunked prefill for intermediate chunks and by the full forward.
///
/// When `inputs_embeds` is provided, uses it directly (skipping embedding lookup).
/// When `per_layer_inputs` is provided, uses it directly (skipping PLE computation).
fn forward_body(
    input_ids: Option<&MxArray>,
    inputs_embeds: Option<MxArray>,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    ple: Option<&PleComponents>,
    per_layer_inputs: Option<&MxArray>,
    config: &Gemma4Config,
) -> Result<MxArray> {
    // Step 1: Embedding (or use pre-computed embeddings)
    let mut h = if let Some(embeds) = inputs_embeds {
        embeds
    } else {
        let ids = input_ids.ok_or_else(|| {
            Error::from_reason("forward_body: either input_ids or inputs_embeds must be provided")
        })?;
        let emb = embedding.forward(ids)?;
        emb.mul_scalar((config.hidden_size as f64).sqrt())?
    };

    let seq_len = h.shape_at(1)?;

    // Step 2: PLE (per-layer embeddings) — compute or reuse
    let owned_ple: Option<MxArray>;
    let effective_ple: Option<&MxArray> = if let Some(ple_inputs) = per_layer_inputs {
        // Pre-computed: might need to slice for chunked prefill
        if ple_inputs.shape_at(1)? != seq_len {
            // Slice to match current chunk (chunked prefill)
            let cache_offset = caches
                .iter()
                .find_map(|c| {
                    let off = c.get_offset();
                    if off > 0 { Some(off as i64) } else { None }
                })
                .unwrap_or(0);
            let max_start = ple_inputs.shape_at(1)? - seq_len;
            let start = cache_offset.min(max_start);
            owned_ple = Some(ple_inputs.slice_axis(1, start, start + seq_len)?);
            owned_ple.as_ref()
        } else {
            Some(ple_inputs)
        }
    } else if let Some(ple) = ple {
        if let Some(ids) = input_ids {
            owned_ple = Some(compute_ple(ids, &h, ple, seq_len)?);
            owned_ple.as_ref()
        } else {
            None
        }
    } else {
        None
    };

    // Step 3: Project PLE if we have per-layer inputs
    // Matches mlx-vlm project_per_layer_inputs: projects h and combines with token PLEs
    let projected_ple: Option<MxArray> = if let Some(ple_data) = effective_ple {
        if let Some(ple) = ple {
            Some(project_per_layer_inputs(&h, ple_data, ple)?)
        } else {
            None
        }
    } else {
        None
    };

    // Step 4: Build masks
    // Global layers: None during prefill → triggers fused causal SDPA kernel
    // Sliding layers: explicit windowed mask during prefill
    // Decode (seq_len == 1): None for both
    //
    // Matches mlx-vlm create_attention_mask behavior:
    //   global → "causal" string → fused kernel
    //   sliding → explicit mask with window constraint
    // Sliding mask: only needed when seq_len > window_size.
    // Matches Python create_attention_mask: when N <= window_size, returns "causal".
    // When N > window_size, returns explicit causal+window mask.
    let sliding_mask = if seq_len > 1 && seq_len > config.sliding_window as i64 {
        let sliding_idx = (0..config.num_hidden_layers as usize)
            .find(|&i| config.is_sliding_layer(i))
            .unwrap_or(0);
        let offset = if sliding_idx < caches.len() {
            caches[sliding_idx].get_offset()
        } else {
            0
        };
        Some(create_sliding_mask(
            seq_len,
            offset,
            config.sliding_window as i64,
        )?)
    } else {
        None
    };

    // Step 5: Forward through layers with KV cache sharing
    let has_kv_sharing = config.num_kv_shared_layers.is_some_and(|n| n > 0);
    let mut shared_kv: HashMap<usize, (MxArray, MxArray)> = HashMap::new();

    for (i, layer) in layers.iter().enumerate() {
        let is_global = config.is_global_layer(i);

        // Global layers: None mask → attention module uses causal SDPA or no-mask path
        // Sliding layers: explicit windowed mask
        let mask: Option<&MxArray> = if is_global {
            None
        } else {
            sliding_mask.as_ref()
        };

        let ple_input = projected_ple.as_ref().map(|p| {
            // projected_ple shape: [B, T, num_layers, ple_dim], extract layer i
            p.slice_axis(2, i as i64, i as i64 + 1)
                .and_then(|s| s.squeeze(Some(&[2])))
        });
        let ple_input_ref = match &ple_input {
            Some(Ok(arr)) => Some(arr),
            _ => None,
        };

        if has_kv_sharing && config.is_kv_shared_layer(i) {
            let anchor_idx = config.kv_shared_anchor(i).ok_or_else(|| {
                Error::from_reason(format!(
                    "Layer {} is shared but has no anchor (missing layer type match)",
                    i
                ))
            })?;

            let (shared_keys, shared_values) = shared_kv.get(&anchor_idx).ok_or_else(|| {
                Error::from_reason(format!(
                    "Anchor layer {} K/V not found for shared layer {}",
                    anchor_idx, i
                ))
            })?;

            // Shared layer uses anchor's cache offset.
            // Subtract seq_len to get pre-update offset (queries need same positions as anchor).
            let cache_offset = caches[anchor_idx].get_offset() - seq_len as i32;

            h = layer.forward_shared(
                &h,
                mask,
                shared_keys,
                shared_values,
                cache_offset,
                ple_input_ref,
            )?;
        } else {
            let needs_stash = has_kv_sharing && config.should_store_shared_kv(i);
            h = layer.forward(&h, mask, Some(&mut caches[i]), ple_input_ref, needs_stash)?;

            if has_kv_sharing
                && config.should_store_shared_kv(i)
                && let Some((keys, values)) = caches[i].take_stashed_kv()
            {
                shared_kv.insert(i, (keys, values));
            }
        }
    }

    // Final norm
    final_norm.forward(&h)
}

/// Full forward pass: transformer body + lm_head + logit softcapping.
///
/// Used for the final prefill chunk and for each decode step.
fn forward_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embed_weight_t: Option<&MxArray>,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
) -> Result<MxArray> {
    let h = forward_body(
        Some(input_ids),
        None,
        embedding,
        layers,
        caches,
        final_norm,
        ple,
        None,
        config,
    )?;

    // LM head or tied embeddings
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else if let Some(w_t) = embed_weight_t {
        h.matmul(w_t)?
    } else {
        let weight = embedding.get_weight();
        let weight_t = weight.transpose(Some(&[1, 0]))?;
        h.matmul(&weight_t)?
    };

    // Logit softcapping — compiled fused kernel (matches Python's mx.compile logit_softcap)
    if let Some(cap) = config.final_logit_softcapping {
        let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
        let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
        Ok(MxArray::from_handle(handle, "logit_softcap")?)
    } else {
        Ok(logits)
    }
}

/// Compute PLE (per-layer embeddings) from input_ids.
/// Returns shape [B, T, num_layers, ple_dim].
fn compute_ple(
    input_ids: &MxArray,
    h: &MxArray,
    ple: &PleComponents,
    seq_len: i64,
) -> Result<MxArray> {
    let ple_dim = ple.ple_dim as i64;
    let num_layers = ple.num_layers as i64;

    // Mask OOV token IDs to 0 for PLE embedding
    let ple_vocab = MxArray::scalar_int(ple.vocab_size_per_layer_input)?;
    let zero = MxArray::scalar_int(0)?;
    let valid_mask = input_ids
        .greater_equal(&zero)?
        .logical_and(&input_ids.less(&ple_vocab)?)?;
    let masked_ids = valid_mask.where_(input_ids, &zero)?;

    // per_layer_embeds: [B, T, num_layers * ple_dim]
    let per_layer_embeds = ple.embed_tokens_per_layer.forward(&masked_ids)?;
    let per_layer_embeds = per_layer_embeds.mul_scalar((ple.ple_dim as f64).sqrt())?;
    let batch = per_layer_embeds.shape_at(0)?;
    let per_layer_embeds = per_layer_embeds.reshape(&[batch, seq_len, num_layers, ple_dim])?;

    // Project from main hidden state
    let projected = ple.per_layer_model_projection.forward(h)?;
    let projected = projected.mul_scalar(ple.per_layer_model_projection_scale)?;
    let projected = projected.reshape(&[batch, seq_len, num_layers, ple_dim])?;

    let projected = ple.per_layer_projection_norm.forward(&projected)?;

    // Combine: (normed_projection + per_layer_embeds) * 1/sqrt(2)
    let combined = projected.add(&per_layer_embeds)?;
    combined.mul_scalar(ple.per_layer_input_scale)
}

/// Project per-layer inputs: combine PLE data with hidden state projection.
/// Returns shape [B, T, num_layers, ple_dim].
fn project_per_layer_inputs(
    _h: &MxArray,
    per_layer_data: &MxArray,
    _ple: &PleComponents,
) -> Result<MxArray> {
    // PLE data is already fully computed (combined projection + token embeddings)
    Ok(per_layer_data.clone())
}

/// Default prefill chunk size (tokens per chunk).
/// Note: mlx-lm uses 2048 but the first eval triggers Metal shader compilation
/// which can GPU-timeout with very large graphs. Using 512 keeps individual
/// command buffers under Metal's timeout limit.
const GEMMA4_PREFILL_STEP_SIZE: i64 = 512;

/// Evaluate all Gemma4 cache arrays to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
fn eval_gemma4_caches(caches: &[Gemma4LayerCache]) {
    let mut arrays: Vec<&MxArray> = Vec::new();
    for cache in caches {
        cache.collect_cache_arrays(&mut arrays);
    }
    if !arrays.is_empty() {
        MxArray::eval_arrays(&arrays);
    }
}

/// Chunked prefill: process all tokens EXCEPT the last one.
///
/// Matches mlx-lm generate.py generate_step prefill pattern:
/// - The prefill loop processes tokens [0:N-1] (all but the last)
/// - The last token is processed by the caller via `forward_inner`, which
///   also produces the logits used to sample the first output token
///
/// This is CRITICAL for correctness: SDPA computes slightly different numerical
/// results for multi-token causal attention vs single-token attention with cached
/// K/V. These small differences compound through layers, causing divergent logits
/// if the last prompt token is processed in the same batch as the rest.
///
/// 1. Embed ALL tokens once upfront (including PLE if enabled)
/// 2. Run only the transformer body for each chunk (no lm_head)
/// 3. Stop BEFORE the last token — the caller handles it via forward_inner
fn prefill_body_gemma4(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
) -> Result<()> {
    let total_len = prompt.shape_at(1)?;

    // Must have at least 2 tokens (1 for prefill, 1 for caller to process)
    if total_len <= 1 {
        return Ok(());
    }

    // Process tokens [0:N-1] — leave last token for the caller
    let prefill_len = total_len - 1;

    // Step 1: Embed tokens [0:N-1]
    let prefill_ids = prompt.slice_axis(1, 0, prefill_len)?;
    let all_embeds = {
        let emb = embedding.forward(&prefill_ids)?;
        emb.mul_scalar((config.hidden_size as f64).sqrt())?
    };

    // Step 2: Compute PLE for prefill tokens (if enabled)
    let all_ple: Option<MxArray> = if let Some(ple) = ple {
        Some(compute_ple(&prefill_ids, &all_embeds, ple, prefill_len)?)
    } else {
        None
    };

    let mut offset: i64 = 0;

    // Process in chunks
    while prefill_len - offset > GEMMA4_PREFILL_STEP_SIZE {
        let chunk_embeds = all_embeds.slice_axis(1, offset, offset + GEMMA4_PREFILL_STEP_SIZE)?;
        let chunk_ple = all_ple
            .as_ref()
            .map(|p| p.slice_axis(1, offset, offset + GEMMA4_PREFILL_STEP_SIZE))
            .transpose()?;

        let _hidden = forward_body(
            None,
            Some(chunk_embeds),
            embedding,
            layers,
            caches,
            final_norm,
            ple,
            chunk_ple.as_ref(),
            config,
        )?;
        eval_gemma4_caches(caches);
        crate::array::clear_cache();
        offset += GEMMA4_PREFILL_STEP_SIZE;
    }

    // Final chunk (still body only — no lm_head needed)
    if offset < prefill_len {
        let remaining_embeds = all_embeds.slice_axis(1, offset, prefill_len)?;
        let remaining_ple = all_ple
            .as_ref()
            .map(|p| p.slice_axis(1, offset, prefill_len))
            .transpose()?;

        let _hidden = forward_body(
            None,
            Some(remaining_embeds),
            embedding,
            layers,
            caches,
            final_norm,
            ple,
            remaining_ple.as_ref(),
            config,
        )?;
    }

    Ok(())
}

fn create_sliding_mask(seq_len: i64, offset: i32, window_size: i64) -> Result<MxArray> {
    let total_len = seq_len + offset as i64;
    let rows = MxArray::arange(offset as f64, (offset as i64 + seq_len) as f64, None, None)?;
    let cols = MxArray::arange(0.0, total_len as f64, None, None)?;
    let rows = rows.reshape(&[seq_len, 1])?;
    let cols = cols.reshape(&[1, total_len])?;
    let distance = rows.sub(&cols)?;

    let zero = MxArray::scalar_int(0)?;
    let window = MxArray::scalar_int(window_size as i32)?;
    let causal = distance.greater_equal(&zero)?;
    let in_window = distance.less(&window)?;
    let valid = causal.logical_and(&in_window)?;

    let neg_inf = MxArray::full(
        &[1],
        Either::A(f64::NEG_INFINITY),
        Some(crate::array::DType::Float32),
    )?;
    let zero_f = MxArray::full(&[1], Either::A(0.0), Some(crate::array::DType::Float32))?;
    let mask = valid.where_(&zero_f, &neg_inf)?;
    mask.reshape(&[1, 1, seq_len, total_len])
}

// ---------------------------------------------------------------------------
// Vision helpers
// ---------------------------------------------------------------------------

/// Extract raw image bytes from ChatMessage.images fields.
fn extract_images_from_messages(messages: &[ChatMessage]) -> Vec<Vec<u8>> {
    let mut all_images: Vec<Vec<u8>> = Vec::new();
    for msg in messages {
        if let Some(ref images) = msg.images {
            for img in images {
                all_images.push(img.to_vec());
            }
        }
    }
    all_images
}

/// Expand image tokens in a token sequence.
///
/// The chat template inserts a single `<|image|>` per image. This function
/// replaces each occurrence with: `boi_token + image_token × num_soft_tokens + eoi_token`.
///
/// If there are fewer `<|image|>` tokens than processed images, the extra images
/// are ignored (manual fallback may not have inserted tokens).
/// If there are no `<|image|>` tokens but images exist, we insert the expanded
/// sequence after the first token (BOS).
fn expand_image_tokens(
    tokens: &[u32],
    processed_images: &[super::image_processor::ProcessedGemma4Image],
    image_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
) -> Vec<u32> {
    let image_count = tokens.iter().filter(|&&t| t == image_token_id).count();

    if image_count == 0 && !processed_images.is_empty() {
        // Manual fallback: insert expanded tokens after BOS (position 0)
        if tokens.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(
            tokens.len()
                + processed_images
                    .iter()
                    .map(|p| p.num_soft_tokens as usize + 2)
                    .sum::<usize>(),
        );
        result.push(tokens[0]); // BOS
        for proc in processed_images {
            result.push(boi_token_id);
            for _ in 0..proc.num_soft_tokens {
                result.push(image_token_id);
            }
            result.push(eoi_token_id);
        }
        result.extend_from_slice(&tokens[1..]);
        return result;
    }

    // Replace each <|image|> with the expanded BOI + N×image_token + EOI sequence
    let mut result = Vec::with_capacity(tokens.len() * 2);
    let mut img_idx = 0;
    for &t in tokens {
        if t == image_token_id && img_idx < processed_images.len() {
            let num_soft = processed_images[img_idx].num_soft_tokens;
            result.push(boi_token_id);
            for _ in 0..num_soft {
                result.push(image_token_id);
            }
            result.push(eoi_token_id);
            img_idx += 1;
        } else {
            result.push(t);
        }
    }
    result
}

/// masked_scatter: replace positions where mask=true with values from source.
///
/// Matches Python: `mx.where(mask_flat, aligned, input_flat).reshape(input.shape)`
/// where `aligned = source.flatten()[(cumsum(mask_flat) - 1) % source.size]`
fn masked_scatter(input: &MxArray, mask: &MxArray, source: &MxArray) -> Result<MxArray> {
    let input_shape = input.shape()?;
    let mask_flat = mask.reshape(&[-1])?.astype(DType::Int32)?;
    let input_flat = input.reshape(&[-1])?;

    let source_flat = source.reshape(&[-1])?;
    let source_size = source_flat.shape_at(0)?;

    // cumsum of mask gives 1-based indices into source; subtract 1 for 0-based
    let indices = mask_flat.cumsum(0)?.sub(&MxArray::scalar_int(1)?)?;
    // Modulo source_size to handle wrap-around safely
    let source_size_arr = MxArray::scalar_int(source_size as i32)?;
    let safe_indices = indices.remainder(&source_size_arr)?;
    let aligned = source_flat.take(&safe_indices, 0)?;

    // where mask=1 use aligned (source), else keep input
    let result = mask_flat.where_(&aligned, &input_flat)?;
    result.reshape(&input_shape)
}

/// Chunked prefill with pre-computed embeddings (for vision path).
///
/// Same as `prefill_body_gemma4` but uses pre-merged `inputs_embeds` instead
/// of looking up from the embedding table. PLE tokens at image positions are
/// zeroed to avoid confusing the per-layer embeddings with vision token IDs.
fn prefill_body_gemma4_with_embeds(
    prompt: &MxArray,
    inputs_embeds: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
) -> Result<()> {
    let total_len = inputs_embeds.shape_at(1)?;

    if total_len <= 1 {
        return Ok(());
    }

    // Process tokens [0:N-1] — leave last token for forward_inner
    let prefill_len = total_len - 1;
    let all_embeds = inputs_embeds.slice_axis(1, 0, prefill_len)?;

    // PLE: mask image token positions to 0 before computing per-layer embeddings
    let all_ple: Option<MxArray> = if let Some(ple) = ple {
        let prefill_ids = prompt.slice_axis(1, 0, prefill_len)?;
        let image_token_id = config.image_token_id.unwrap_or(258880);
        let image_token = MxArray::scalar_int(image_token_id)?;
        let image_mask = prefill_ids.equal(&image_token)?;
        let zero = MxArray::scalar_int(0)?;
        let masked_ids = image_mask.where_(&zero, &prefill_ids)?;
        Some(compute_ple(&masked_ids, &all_embeds, ple, prefill_len)?)
    } else {
        None
    };

    let mut offset: i64 = 0;

    while prefill_len - offset > GEMMA4_PREFILL_STEP_SIZE {
        let chunk_embeds = all_embeds.slice_axis(1, offset, offset + GEMMA4_PREFILL_STEP_SIZE)?;
        let chunk_ple = all_ple
            .as_ref()
            .map(|p| p.slice_axis(1, offset, offset + GEMMA4_PREFILL_STEP_SIZE))
            .transpose()?;

        let _hidden = forward_body(
            None,
            Some(chunk_embeds),
            embedding,
            layers,
            caches,
            final_norm,
            ple,
            chunk_ple.as_ref(),
            config,
        )?;
        eval_gemma4_caches(caches);
        crate::array::clear_cache();
        offset += GEMMA4_PREFILL_STEP_SIZE;
    }

    if offset < prefill_len {
        let remaining_embeds = all_embeds.slice_axis(1, offset, prefill_len)?;
        let remaining_ple = all_ple
            .as_ref()
            .map(|p| p.slice_axis(1, offset, prefill_len))
            .transpose()?;

        let _hidden = forward_body(
            None,
            Some(remaining_embeds),
            embedding,
            layers,
            caches,
            final_norm,
            ple,
            remaining_ple.as_ref(),
            config,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma4_chat_manual_fallback_format() {
        // When no chat template exists, manual format should:
        // 1. Start with <bos>
        // 2. Map "assistant" → "model"
        // 3. End with <|turn>model\n
        let messages = vec![
            ("system", "You are helpful."),
            ("user", "Hi"),
            ("assistant", "Hello!"),
            ("user", "Bye"),
        ];
        let mut prompt = String::from("<bos>");
        for (role, content) in &messages {
            let mapped = match *role {
                "assistant" => "model",
                other => other,
            };
            prompt.push_str(&format!("<|turn>{}\n{}<turn|>\n", mapped, content));
        }
        prompt.push_str("<|turn>model\n");

        assert!(prompt.starts_with("<bos><|turn>"), "must start with <bos>");
        assert!(prompt.contains("<|turn>system\nYou are helpful.<turn|>"));
        assert!(prompt.contains("<|turn>model\nHello!<turn|>"));
        assert!(!prompt.contains("<|turn>assistant"));
        assert!(prompt.ends_with("<|turn>model\n"));
    }

    #[test]
    fn test_gemma4_chat_role_mapping() {
        // Verify that "assistant" role gets mapped to "model" in Gemma4 format
        let messages = vec![
            ("user", "Hi"),
            ("assistant", "Hello!"),
            ("user", "How are you?"),
        ];

        let mut prompt_text = String::from("<bos>");
        for (role, content) in &messages {
            let mapped_role = match *role {
                "assistant" => "model",
                other => other,
            };
            prompt_text.push_str(&format!("<|turn>{}\n{}<turn|>\n", mapped_role, content));
        }
        prompt_text.push_str("<|turn>model\n");

        // Verify BOS is present and "assistant" was mapped to "model"
        assert!(prompt_text.starts_with("<bos>"), "must start with <bos>");
        assert!(
            !prompt_text.contains("<|turn>assistant"),
            "assistant role should be mapped to model"
        );
        assert!(
            prompt_text.contains("<|turn>model\nHello!<turn|>"),
            "assistant message should use model role"
        );

        // Verify the full format (with <bos> prefix)
        let expected = "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nHow are you?<turn|>\n<|turn>model\n";
        assert_eq!(prompt_text, expected);
    }

    #[test]
    fn test_ple_oov_masking() {
        // Simulate token IDs where some exceed PLE vocab or are negative
        let input_ids = MxArray::from_int32(&[5, 100, 262143, 0, -1], &[1, 5]).unwrap();
        let ple_vocab = 262144i32; // PLE vocab size

        let ple_vocab_arr = MxArray::scalar_int(ple_vocab).unwrap();
        let zero = MxArray::scalar_int(0).unwrap();
        let valid_mask = input_ids
            .greater_equal(&zero)
            .unwrap()
            .logical_and(&input_ids.less(&ple_vocab_arr).unwrap())
            .unwrap();
        let masked_ids = valid_mask.where_(&input_ids, &zero).unwrap();

        masked_ids.eval();
        // IDs within range: unchanged. IDs out of range (negative): mapped to 0.
        assert_eq!(masked_ids.item_at_int32(0).unwrap(), 5); // in range
        assert_eq!(masked_ids.item_at_int32(1).unwrap(), 100); // in range
        // 262143 < 262144, so it's valid
        assert_eq!(masked_ids.item_at_int32(2).unwrap(), 262143);
        assert_eq!(masked_ids.item_at_int32(3).unwrap(), 0); // in range (0 is valid)
        assert_eq!(masked_ids.item_at_int32(4).unwrap(), 0); // -1 is OOV, mapped to 0
    }

    #[test]
    fn test_gemma4_chat_tool_calls_serialization() {
        // Verify tool call args use Gemma4 DSL format (not raw JSON)
        // JSON: {"location": "Paris", "units": "celsius"}
        // DSL:  location:<|"|>Paris<|"|>,units:<|"|>celsius<|"|>  (keys sorted alphabetically)
        let args_json = r#"{"location": "Paris", "units": "celsius"}"#;
        let dsl = json_args_to_gemma4_dsl(args_json);
        assert_eq!(
            dsl, r#"location:<|"|>Paris<|"|>,units:<|"|>celsius<|"|>"#,
            "string values should be wrapped in <|\"|> delimiters, keys sorted alphabetically"
        );

        // Verify numeric and bool values are bare (no quotes)
        let args_with_number = r#"{"count": 5, "active": true}"#;
        let dsl2 = json_args_to_gemma4_dsl(args_with_number);
        assert_eq!(
            dsl2, "active:true,count:5",
            "numbers and bools should be bare (no <|\"|> wrapping), keys sorted alphabetically"
        );

        // Verify format_gemma4_value handles nested JSON objects correctly
        let nested_json = r#"{"temp": 20}"#;
        let nested_val: serde_json::Value = serde_json::from_str(nested_json).unwrap();
        let dsl3 = format_gemma4_value(&nested_val);
        assert_eq!(dsl3, "{temp:20}", "object with bare number value");

        // Build a full prompt matching the manual fallback path
        let mut prompt = String::from("<bos>");

        // user turn
        prompt.push_str("<|turn>user\nWhat's the weather?<turn|>\n");

        // model tool-call turn (assistant → model)
        let tc_dsl = json_args_to_gemma4_dsl(r#"{"location": "Paris", "units": "celsius"}"#);
        prompt.push_str(&format!(
            "<|turn>model\n<|tool_call>call:get_weather{{{}}}<tool_call|><turn|>\n",
            tc_dsl
        ));

        // tool response turn — plain <|turn>tool format (matches HF tokenizer behavior)
        prompt.push_str("<|turn>tool\n{\"temp\": 20}<turn|>\n");

        // final model answer
        prompt.push_str("<|turn>model\nIt's 20 degrees in Paris.<turn|>\n");
        prompt.push_str("<|turn>model\n");

        // Verify DSL format in tool call (no raw JSON quotes)
        assert!(
            prompt.contains(r#"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,units:<|"|>celsius<|"|>}<tool_call|>"#),
            "tool call args should use Gemma4 DSL with <|\"|> string delimiters"
        );
        assert!(
            !prompt.contains(r#""location""#),
            "tool call should NOT contain raw JSON quoted keys"
        );

        // Verify tool response uses simple <|turn>tool format (not rewritten)
        assert!(
            prompt.contains("<|turn>tool\n"),
            "tool response should use plain <|turn>tool format"
        );
        assert!(
            !prompt.contains("<|tool_response>"),
            "tool response should NOT use <|tool_response> rewriting"
        );

        // Verify assistant→model mapping
        assert!(!prompt.contains("<|turn>assistant"));
    }

    #[test]
    fn test_gemma4_chat_developer_role_mapping() {
        // "developer" role should be mapped to "system"
        let mut prompt = String::from("<bos>");
        let role = "developer";
        let mapped = match role {
            "assistant" => "model",
            "developer" => "system",
            other => other,
        };
        prompt.push_str(&format!(
            "<|turn>{}\nYou are a helpful bot.<turn|>\n",
            mapped
        ));
        prompt.push_str("<|turn>model\n");

        assert!(
            prompt.contains("<|turn>system\nYou are a helpful bot."),
            "developer role should be mapped to system"
        );
        assert!(
            !prompt.contains("<|turn>developer"),
            "developer should not appear as a raw role"
        );
    }
}
