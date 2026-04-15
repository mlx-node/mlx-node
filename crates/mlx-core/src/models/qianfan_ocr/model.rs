/**
 * Qianfan-OCR Main Model
 *
 * Integrates InternViT vision encoder, MLP bridge, and Qwen3 language model
 * for OCR and document understanding tasks.
 *
 * Provides NAPI-exposed load(), session chat methods (chatSessionStart /
 * chatSessionContinue / chatSessionContinueTool and streaming variants),
 * generate(), and resetCaches() APIs.
 *
 * # Architecture
 *
 * All model state (weights, KV caches, tokenizer, cached turn metadata) lives
 * on a dedicated OS thread owned by the `ModelThread<QianfanOCRCmd>` field on
 * [`QianfanOCRModel`]. NAPI methods are thin shells that marshal arguments
 * into a `QianfanOCRCmd` and dispatch them through the command channel —
 * responses flow back via oneshot channels (for the non-streaming
 * session commands, `Generate`, and `ResetCaches`) or an mpsc stream
 * (for the streaming session commands). This keeps MLX arrays off the
 * Tokio worker threads and removes the `Arc<RwLock<>>` plumbing the
 * legacy layout used to share mutable model state with `spawn_blocking`
 * closures.
 */
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{Env, Status, bindgen_prelude::*};
use napi_derive::napi;
use serde_json::Value;
use tracing::info;

use crate::array::{MxArray, synchronize_and_clear_cache};
use crate::model_thread::{ResponseTx, StreamTx};
use crate::models::qianfan_ocr::bridge::InternVLBridge;
use crate::models::qianfan_ocr::chat::format_qianfan_chat;
use crate::models::qianfan_ocr::config::{InternVisionConfig, QianfanOCRConfig, Qwen3LMConfig};
use crate::models::qianfan_ocr::language::InternVLLanguageModel;
use crate::models::qianfan_ocr::persistence::load_qianfan_ocr_weights;
use crate::models::qianfan_ocr::processing::QianfanImageProcessor;
use crate::models::qianfan_ocr::vision::InternViTModel;
use crate::models::qwen3_5::model::extract_images_from_messages;
use crate::models::qwen3_5::model::{ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle};
use crate::sampling::{
    SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
    check_repetition_cutoff, sample,
};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::tools;
use crate::transformer::kv_cache::KVCache;
use crate::utils::safetensors::SafeTensorsFile;

// ============================================================================
// QianfanOCRInner — dedicated-thread owned state
// ============================================================================

/// Internal Qianfan-OCR model state owned exclusively by the dedicated
/// model thread.
///
/// All fields are plain-owned (no `Arc<RwLock<>>`) because the model
/// thread has sole mutable access. `kv_caches`, `cached_token_history`,
/// `cached_image_key`, and `cached_cache_offset` are hoisted out of
/// [`InternVLLanguageModel`] and the old NAPI struct so the session
/// methods can read/mutate them directly alongside the other per-turn
/// metadata.
pub(crate) struct QianfanOCRInner {
    pub(crate) config: QianfanOCRConfig,
    pub(crate) vision: InternViTModel,
    pub(crate) bridge: InternVLBridge,
    pub(crate) language_model: InternVLLanguageModel,
    pub(crate) tokenizer: Arc<Qwen3Tokenizer>,
    /// Per-layer KV caches, promoted from [`InternVLLanguageModel`] so
    /// the session methods can inspect / clone / trim them in place
    /// without going through a wrapper method.
    pub(crate) kv_caches: Option<Vec<KVCache>>,
    /// Token history of the prompt + forwarded generated tokens from
    /// the most recent session turn. Used for prefix-match-based cache
    /// reuse on the next call.
    pub(crate) cached_token_history: Vec<u32>,
    /// Cached image set hash from the most recent session-start turn.
    /// Populated by the VLM-capable start path and cleared on reset —
    /// the TS `ChatSession` layer watches for changes and routes
    /// image-swap turns back through a fresh `chat_session_start`.
    pub(crate) cached_image_key: Option<u64>,
    /// Cache offset from the most recent call (number of tokens
    /// committed to the KV cache). Mirrors `kv_caches[0].get_offset()`
    /// at the end of the previous turn and is used by the session
    /// methods to validate session continuity without touching the
    /// caches.
    pub(crate) cached_cache_offset: i32,
}

// ============================================================================
// Commands dispatched from NAPI methods to the dedicated model thread
// ============================================================================

/// Commands dispatched from NAPI methods to the Qianfan-OCR model thread.
pub(crate) enum QianfanOCRCmd {
    /// Start a new session via the text-only / VLM jinja-render path with
    /// `<|im_end|>` as the stop token. See
    /// [`QianfanOCRInner::chat_session_start_sync`] for the behavioural
    /// contract (full cache reset, session-boundary eos, VLM-capable).
    ChatSessionStart {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session by appending a user turn. See
    /// [`QianfanOCRInner::chat_session_continue_sync`] — builds a raw
    /// ChatML delta from `user_message`, tokenizes it, and prefills on
    /// top of the live caches.
    ///
    /// `images` is an opt-in guard parameter: non-empty input is rejected
    /// with an `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed error so
    /// the TS `ChatSession` layer can route image-changes back through a
    /// fresh `chat_session_start`.
    ChatSessionContinue {
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session with a tool-result delta. See
    /// [`QianfanOCRInner::chat_session_continue_tool_sync`] — builds a
    /// ChatML `<tool_response>` delta and prefills on top of the live
    /// caches.
    ChatSessionContinueTool {
        tool_call_id: String,
        content: String,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Streaming session-start: same semantics as
    /// [`ChatSessionStart`](Self::ChatSessionStart) but streams token
    /// deltas through `stream_tx`.
    ChatStreamSessionStart {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    /// Streaming session-continue: same semantics as
    /// [`ChatSessionContinue`](Self::ChatSessionContinue) but streams
    /// token deltas through `stream_tx`. Carries the same opt-in
    /// `images` guard parameter.
    ChatStreamSessionContinue {
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    /// Streaming tool-result continuation: same semantics as
    /// [`ChatSessionContinueTool`](Self::ChatSessionContinueTool) but
    /// streams token deltas through `stream_tx`.
    ChatStreamSessionContinueTool {
        tool_call_id: String,
        content: String,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    Generate {
        input_ids: MxArray,
        max_new_tokens: i32,
        temperature: f64,
        reply: ResponseTx<Vec<u32>>,
    },
    ResetCaches {
        reply: ResponseTx<()>,
    },
}

/// Command handler for the dedicated model thread.
///
/// Dispatches each command variant to the matching `_sync` method on
/// [`QianfanOCRInner`] and forwards the result through the response
/// channel.
pub(crate) fn handle_qianfan_ocr_cmd(inner: &mut QianfanOCRInner, cmd: QianfanOCRCmd) {
    match cmd {
        QianfanOCRCmd::ChatSessionStart {
            messages,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_start_sync(messages, config));
        }
        QianfanOCRCmd::ChatSessionContinue {
            user_message,
            images,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_continue_sync(user_message, images, config));
        }
        QianfanOCRCmd::ChatSessionContinueTool {
            tool_call_id,
            content,
            config,
            reply,
        } => {
            let _ =
                reply.send(inner.chat_session_continue_tool_sync(tool_call_id, content, config));
        }
        QianfanOCRCmd::ChatStreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_start_sync(messages, config, stream_tx, cancelled);
        }
        QianfanOCRCmd::ChatStreamSessionContinue {
            user_message,
            images,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_continue_sync(
                user_message,
                images,
                config,
                stream_tx,
                cancelled,
            );
        }
        QianfanOCRCmd::ChatStreamSessionContinueTool {
            tool_call_id,
            content,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_continue_tool_sync(
                tool_call_id,
                content,
                config,
                stream_tx,
                cancelled,
            );
        }
        QianfanOCRCmd::Generate {
            input_ids,
            max_new_tokens,
            temperature,
            reply,
        } => {
            let _ = reply.send(inner.generate_sync(&input_ids, max_new_tokens, temperature));
        }
        QianfanOCRCmd::ResetCaches { reply } => {
            inner.reset_caches_sync();
            let _ = reply.send(Ok(()));
        }
    }
}

// ============================================================================
// QianfanOCRModel — NAPI shell around the dedicated model thread
// ============================================================================

/// Qianfan-OCR Vision-Language Model (InternVL architecture).
///
/// Combines InternViT vision encoder, MLP bridge with pixel shuffle,
/// and Qwen3 language model for OCR and document understanding.
///
/// All inference state lives on a dedicated OS thread. NAPI methods
/// dispatch commands via channels and await responses.
#[napi(js_name = "QianfanOCRModel")]
pub struct QianfanOCRModel {
    /// Dedicated model thread owning `QianfanOCRInner`. `None` when the
    /// model was constructed via `new(config)` without loading weights —
    /// in that uninitialized state only `isInitialized` is meaningful.
    thread: Option<crate::model_thread::ModelThread<QianfanOCRCmd>>,
    /// Cloned from inner for pure-getter NAPI methods (no command
    /// dispatch needed). For uninitialized instances this holds the
    /// bare constructor argument. Marked `allow(dead_code)` because
    /// nothing on the NAPI surface currently exposes a config getter —
    /// kept in place so one can be added without re-plumbing.
    #[allow(dead_code)]
    config: QianfanOCRConfig,
    /// Whether the model was loaded with real weights. `false` for
    /// `new QianfanOCRModel(config)` calls that predate `load()`.
    initialized: bool,
}

// ============================================================================
// QianfanOCRInner — core model logic (owned by the model thread)
// ============================================================================

/// Wrapper that adapts [`StreamTx<ChatStreamChunk>`] to the same `call()`
/// API as a napi [`ThreadsafeFunction`], so the streaming decode loop can
/// be reused verbatim when migrated off the callback path.
struct StreamSender(StreamTx<ChatStreamChunk>);

impl StreamSender {
    fn call(&self, result: napi::Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
        let _ = self.0.send(result);
    }
}

impl QianfanOCRInner {
    /// Reset the hoisted cache state (KV caches, token history, image key,
    /// cached offset). Used by both `reset_caches()` and the ResetCaches
    /// command path.
    fn reset_caches_sync(&mut self) {
        self.kv_caches = None;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_cache_offset = 0;
    }

    /// Allocate a fresh per-layer KV cache vector sized to match the
    /// current language model. Previously this lived on
    /// [`InternVLLanguageModel`]; it was hoisted so the session
    /// methods can share the same storage with other cached metadata.
    fn init_kv_caches(&mut self) {
        let num_layers = self.language_model.num_layers();
        self.kv_caches = Some((0..num_layers).map(|_| KVCache::new()).collect());
    }

    /// Current cache offset (number of tokens committed to the KV cache).
    /// Reads from the first layer's cache — all layers advance together.
    fn get_cache_offset(&self) -> i32 {
        self.kv_caches
            .as_ref()
            .and_then(|caches| caches.first())
            .map(|c| c.get_offset())
            .unwrap_or(0)
    }

    /// Core synchronous chat implementation with optional EOS override
    /// (runs on the model thread).
    ///
    /// Shared InternViT -> bridge -> language-model prefill/decode
    /// pipeline for the session surface: KV cache reuse via prefix
    /// matching, repetition/presence/frequency penalties, thinking/tool
    /// call parsing, and optional performance metrics.
    ///
    /// `eos_token_id` is the caller-supplied stop-on token id
    /// (`<|im_end|>` for ChatML boundaries) so the cached history ends on
    /// a clean delimiter that subsequent `chat_session_continue_*` calls
    /// can append a raw delta on top of.
    ///
    /// Only called from [`Self::chat_session_start_sync`]; there is no
    /// longer a non-session entry point.
    fn chat_sync_core(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        eos_token_id: u32,
    ) -> Result<ChatResult> {
        let max_new_tokens = config.max_new_tokens.unwrap_or(512);
        let temperature = config.temperature.unwrap_or(0.0);
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let presence_penalty = config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = config.presence_context_size.unwrap_or(20);
        let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = config.frequency_context_size.unwrap_or(20);
        let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
        let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
        let ngram_size = config.ngram_size.unwrap_or(64);
        let enable_thinking =
            crate::models::qwen3_5::chat_common::resolve_enable_thinking(&config).unwrap_or(false);
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Extract images from messages (CPU bound)
        let image_bytes = extract_images_from_messages(&messages);

        // --- Step 1: Process images ---
        let processor = QianfanImageProcessor::new(&self.config);
        let image_refs: Vec<&[u8]> = image_bytes.iter().map(|b| &b[..]).collect();
        let processed_images = processor.process_many(&image_refs)?;

        let num_patches_list: Vec<u32> = processed_images.iter().map(|p| p.num_tiles).collect();
        let num_image_token = self.config.num_image_token() as u32;

        // --- Step 2: Format chat template ---
        let prompt = format_qianfan_chat(
            &messages,
            &num_patches_list,
            num_image_token,
            enable_thinking,
            config.tools.as_deref(),
        )?;

        // --- Step 3: Tokenize ---
        let token_ids = self.tokenizer.encode_sync(&prompt, Some(false))?;
        let input_ids = MxArray::from_uint32(&token_ids, &[1, token_ids.len() as i64])?;

        // --- Step 4: Vision encoding ---
        let generation_stream = Stream::new(DeviceType::Gpu);
        let vision_features = if !processed_images.is_empty() {
            // Stack all tiles from all images: [total_tiles, H, W, C]
            let all_pixels = stack_processed_images(&processed_images)?;

            let vit_out = {
                let _ctx = StreamContext::new(generation_stream);
                self.vision.forward(&all_pixels)?
            };

            // Bridge: pixel shuffle + MLP projection
            let bridge_out = {
                let _ctx = StreamContext::new(generation_stream);
                self.bridge.forward(&vit_out)?
            };

            // bridge_out: [total_tiles, tokens_per_tile, llm_hidden]
            // Flatten to [total_visual_tokens, llm_hidden]
            let bridge_shape = bridge_out.shape()?;
            let total_tiles = bridge_shape[0];
            let tokens_per_tile = bridge_shape[1];
            let hidden = bridge_shape[2];
            Some(bridge_out.reshape(&[total_tiles * tokens_per_tile, hidden])?)
        } else {
            None
        };

        // --- Step 5: Prefix matching for KV cache reuse ---
        let prefix_len = if reuse_cache && image_bytes.is_empty() {
            // Only reuse cache for text-only; images need full re-prefill
            // since IMG_CONTEXT token IDs don't capture image content
            compute_prefix_match(&token_ids, &self.cached_token_history)
        } else {
            0
        };

        if prefix_len == 0 || !reuse_cache {
            // Full reset for fresh generation
            self.kv_caches = None;
            self.init_kv_caches();
        }
        // Trim happens below after we know seq_len — see clamped_prefix

        // --- Step 6: Prefill ---
        let merged_embeds = if let Some(ref vf) = vision_features {
            let text_embeds = {
                let _ctx = StreamContext::new(generation_stream);
                self.language_model.get_embeddings(&input_ids)?
            };
            let embed_dtype = text_embeds.dtype()?;
            let vf_cast = if vf.dtype()? != embed_dtype {
                vf.astype(embed_dtype)?
            } else {
                vf.clone()
            };
            merge_vision_features(
                &input_ids,
                &text_embeds,
                &vf_cast,
                self.config.img_context_token_id,
            )?
        } else {
            let _ctx = StreamContext::new(generation_stream);
            self.language_model.get_embeddings(&input_ids)?
        };

        // Clamp prefix to seq_len-1 so there's always at least 1 token to
        // forward for logits. Handles the full-prefix-hit case where
        // prefix_len == seq_len (identical prompt resent).
        let seq_len = merged_embeds.shape()?[1];
        let clamped_prefix = prefix_len.min(seq_len.saturating_sub(1) as usize);

        // Trim KV cache to clamped_prefix to discard stale suffix tokens
        if clamped_prefix > 0 && reuse_cache {
            let cache_offset = self.get_cache_offset();
            if cache_offset > clamped_prefix as i32
                && let Some(caches) = self.kv_caches.as_mut()
            {
                for c in caches.iter_mut() {
                    c.trim(clamped_prefix as i32);
                }
            }
        }

        let prefill_embeds = if clamped_prefix > 0 {
            merged_embeds.slice_axis(1, clamped_prefix as i64, seq_len)?
        } else {
            merged_embeds
        };

        let mut cache = self.kv_caches.take();
        let prefill_result: Result<MxArray> = {
            let _ctx = StreamContext::new(generation_stream);
            self.language_model
                .forward_from_embeddings(&prefill_embeds, &mut cache)
        };
        self.kv_caches = cache;
        let prefill_logits = prefill_result?;

        // Eval prefill logits -- caches materialize through dependency graph
        prefill_logits.eval();
        synchronize_and_clear_cache();

        // Get last logits for first token sampling
        let prefill_seq = prefill_logits.shape()?[1];
        let mut last_logits = prefill_logits
            .slice_axis(1, prefill_seq - 1, prefill_seq)?
            .squeeze(Some(&[0, 1]))?;

        // --- Step 7: Sampling config ---
        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

        // Track all tokens for repetition penalty
        let mut all_tokens: Vec<u32> = token_ids.clone();

        // Apply penalties to first token
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
        let mut token = sample(&last_logits, Some(sampling_config))?;
        token.eval();

        let first_token_instant = generation_start.map(|_| std::time::Instant::now());
        let prefill_token_count = token_ids.len();

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
        let mut finish_reason = "length".to_string();

        // --- Step 8: Decode loop ---
        for _step in 0..max_new_tokens {
            let token_value = token.item_at_int32(0)? as u32;
            generated_tokens.push(token_value);
            all_tokens.push(token_value);

            // Check EOS
            if token_value == eos_token_id {
                finish_reason = "stop".to_string();
                break;
            }

            // Check repetition cutoff
            if let Some(reason) = check_repetition_cutoff(
                &generated_tokens,
                max_consecutive_tokens,
                max_ngram_repeats,
                ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }

            // Forward single token
            let token_2d = token.reshape(&[1, 1])?;
            let mut cache = self.kv_caches.take();
            let step_result: Result<MxArray> = {
                let _ctx = StreamContext::new(generation_stream);
                self.language_model.forward(&token_2d, &mut cache)
            };
            self.kv_caches = cache;
            let logits = step_result?;

            let mut next_logits = logits.squeeze(Some(&[0, 1]))?;

            // Apply penalties
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

            token = sample(&next_logits, Some(sampling_config))?;
            token.eval();

            // Periodic cache clearing to prevent memory accumulation
            if (_step + 1) % 256 == 0 {
                synchronize_and_clear_cache();
            }
        }

        // --- Step 9: Sync token history with cache state ---
        // On "stop"/"repetition" exits, the terminal token was sampled and
        // pushed to generated_tokens but never forwarded into the KV cache.
        // Only include tokens that were actually forwarded so prefix matching
        // stays aligned with the live cache.
        if reuse_cache {
            let forwarded = if finish_reason == "stop" || finish_reason == "repetition" {
                generated_tokens.len().saturating_sub(1)
            } else {
                generated_tokens.len()
            };
            let mut full_history = token_ids.clone();
            full_history.extend_from_slice(&generated_tokens[..forwarded]);
            self.cached_token_history = full_history;
            self.cached_cache_offset = self.get_cache_offset();
            // Track image state so the session delta path's guard 4
            // (cached_image_key.is_some()) correctly rejects delta
            // continuations that would collide with prior image context.
            self.cached_image_key = if image_bytes.is_empty() {
                None
            } else {
                Some(crate::models::qwen3_5::chat_common::compute_image_cache_key(&image_bytes))
            };
        } else {
            // Not reusing — clear metadata to prevent stale prefix matches
            self.cached_token_history.clear();
            self.cached_cache_offset = 0;
            self.cached_image_key = None;
        }

        // --- Step 10: Decode and parse ---
        let raw_decoded = self.tokenizer.decode_sync(&generated_tokens, true)?;
        let cleaned = raw_decoded.replace("<|im_end|>", "").trim().to_string();

        let (text_after_thinking, thinking) = tools::parse_thinking(&cleaned);
        let (text, tool_calls) = tools::parse_tool_calls(&text_after_thinking);

        // Promote finish_reason to "tool_calls" when valid tool calls are parsed
        if tool_calls.iter().any(|tc| tc.status == "ok") {
            finish_reason = "tool_calls".to_string();
        }

        let performance =
            if let (Some(gen_start), Some(first_tok)) = (generation_start, first_token_instant) {
                let generation_end = std::time::Instant::now();
                let prefill_toks = prefill_token_count as f64;
                let gen_toks = generated_tokens.len() as f64;
                let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
                let decode_ms = generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
                Some(crate::profiling::PerformanceMetrics {
                    ttft_ms,
                    prefill_tokens_per_second: if ttft_ms > 0.0 {
                        prefill_toks / (ttft_ms / 1000.0)
                    } else {
                        0.0
                    },
                    decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                        (gen_toks - 1.0) / (decode_ms / 1000.0)
                    } else {
                        0.0
                    },
                })
            } else {
                None
            };

        let reasoning_tokens = tools::count_reasoning_tokens(
            &thinking,
            &generated_tokens,
            self.tokenizer.think_end_id(),
        );

        Ok(ChatResult {
            text: text.trim().to_string(),
            tool_calls,
            thinking,
            num_tokens: generated_tokens.len() as u32,
            prompt_tokens: prefill_token_count as u32,
            reasoning_tokens,
            finish_reason,
            raw_text: raw_decoded,
            performance,
        })
    }

    /// Streaming chat generation.
    ///
    /// Mirrors [`chat_sync_core`] but emits per-token deltas through
    /// `stream_tx` and checks `cancelled` on every decode iteration.
    /// Drives the same hoisted cache state so prefix matching and
    /// repetition penalties behave identically to the non-streaming path.
    ///
    /// `eos_token_id` threads through exactly as in
    /// [`chat_sync_core`](Self::chat_sync_core): session-start callers
    /// supply `<|im_end|>` via
    /// [`chat_stream_session_start_sync`](Self::chat_stream_session_start_sync).
    fn chat_stream_sync_core(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
        eos_token_id: u32,
    ) {
        let sender = StreamSender(stream_tx.clone());
        let emit = |chunk: ChatStreamChunk| {
            sender.call(Ok(chunk), ThreadsafeFunctionCallMode::NonBlocking);
        };

        let result: Result<()> = (|| {
            let max_new_tokens = config.max_new_tokens.unwrap_or(512);
            let temperature = config.temperature.unwrap_or(0.0);
            let top_k = config.top_k.unwrap_or(0);
            let top_p = config.top_p.unwrap_or(1.0);
            let min_p = config.min_p.unwrap_or(0.0);
            let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
            let repetition_context_size = config.repetition_context_size.unwrap_or(256);
            let presence_penalty = config.presence_penalty.unwrap_or(0.0);
            let presence_context_size = config.presence_context_size.unwrap_or(20);
            let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
            let frequency_context_size = config.frequency_context_size.unwrap_or(20);
            let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
            let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
            let ngram_size = config.ngram_size.unwrap_or(64);
            let enable_thinking =
                crate::models::qwen3_5::chat_common::resolve_enable_thinking(&config)
                    .unwrap_or(false);
            let reuse_cache = config.reuse_cache.unwrap_or(true);
            let report_perf = config.report_performance.unwrap_or(false);

            let generation_start = if report_perf {
                Some(std::time::Instant::now())
            } else {
                None
            };

            let image_bytes = extract_images_from_messages(&messages);

            // --- Process images ---
            let processor = QianfanImageProcessor::new(&self.config);
            let image_refs: Vec<&[u8]> = image_bytes.iter().map(|b| &b[..]).collect();
            let processed_images = processor.process_many(&image_refs)?;

            let num_patches_list: Vec<u32> = processed_images.iter().map(|p| p.num_tiles).collect();
            let num_image_token = self.config.num_image_token() as u32;

            // --- Format and tokenize ---
            let prompt = format_qianfan_chat(
                &messages,
                &num_patches_list,
                num_image_token,
                enable_thinking,
                config.tools.as_deref(),
            )?;
            let token_ids = self.tokenizer.encode_sync(&prompt, Some(false))?;
            let input_ids = MxArray::from_uint32(&token_ids, &[1, token_ids.len() as i64])?;

            // --- Vision encoding ---
            let generation_stream = Stream::new(DeviceType::Gpu);
            let vision_features = if !processed_images.is_empty() {
                let all_pixels = stack_processed_images(&processed_images)?;
                let vit_out = {
                    let _ctx = StreamContext::new(generation_stream);
                    self.vision.forward(&all_pixels)?
                };
                let bridge_out = {
                    let _ctx = StreamContext::new(generation_stream);
                    self.bridge.forward(&vit_out)?
                };
                let bridge_shape = bridge_out.shape()?;
                let total_tiles = bridge_shape[0];
                let tokens_per_tile = bridge_shape[1];
                let hidden = bridge_shape[2];
                Some(bridge_out.reshape(&[total_tiles * tokens_per_tile, hidden])?)
            } else {
                None
            };

            // --- Prefill (with cache reuse support) ---
            let prefix_len = if reuse_cache && image_bytes.is_empty() {
                compute_prefix_match(&token_ids, &self.cached_token_history)
            } else {
                0
            };

            if prefix_len == 0 || !reuse_cache {
                self.kv_caches = None;
                self.init_kv_caches();
            }

            let merged_embeds = if let Some(ref vf) = vision_features {
                let text_embeds = {
                    let _ctx = StreamContext::new(generation_stream);
                    self.language_model.get_embeddings(&input_ids)?
                };
                let embed_dtype = text_embeds.dtype()?;
                let vf_cast = if vf.dtype()? != embed_dtype {
                    vf.astype(embed_dtype)?
                } else {
                    vf.clone()
                };
                merge_vision_features(
                    &input_ids,
                    &text_embeds,
                    &vf_cast,
                    self.config.img_context_token_id,
                )?
            } else {
                let _ctx = StreamContext::new(generation_stream);
                self.language_model.get_embeddings(&input_ids)?
            };

            let seq_len = merged_embeds.shape()?[1];
            let clamped_prefix = prefix_len.min(seq_len.saturating_sub(1) as usize);

            if clamped_prefix > 0 && reuse_cache {
                let cache_offset = self.get_cache_offset();
                if cache_offset > clamped_prefix as i32
                    && let Some(caches) = self.kv_caches.as_mut()
                {
                    for c in caches.iter_mut() {
                        c.trim(clamped_prefix as i32);
                    }
                }
            }

            let prefill_embeds = if clamped_prefix > 0 {
                merged_embeds.slice_axis(1, clamped_prefix as i64, seq_len)?
            } else {
                merged_embeds
            };

            let mut cache = self.kv_caches.take();
            let prefill_result: Result<MxArray> = {
                let _ctx = StreamContext::new(generation_stream);
                self.language_model
                    .forward_from_embeddings(&prefill_embeds, &mut cache)
            };
            self.kv_caches = cache;
            let prefill_logits = prefill_result?;

            prefill_logits.eval();
            synchronize_and_clear_cache();

            let seq_len = prefill_logits.shape()?[1];
            let mut last_logits = prefill_logits
                .slice_axis(1, seq_len - 1, seq_len)?
                .squeeze(Some(&[0, 1]))?;

            let sampling_config = SamplingConfig {
                temperature: Some(temperature),
                top_k: Some(top_k),
                top_p: Some(top_p),
                min_p: Some(min_p),
            };

            let prompt_token_ids = token_ids.clone();
            let mut all_tokens: Vec<u32> = token_ids;

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

            let mut token = sample(&last_logits, Some(sampling_config))?;
            token.eval();

            let first_token_instant = generation_start.map(|_| std::time::Instant::now());
            let prefill_token_count = all_tokens.len();

            let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
            let mut finish_reason = "length".to_string();

            // Stateful decoder for correct multi-byte/CJK streaming
            let mut decode_stream = self.tokenizer.inner().decode_stream(true);
            let mut streamed_text_len: usize = 0;

            // --- Streaming decode loop ---
            for step in 0..max_new_tokens {
                if cancelled.load(Ordering::Relaxed) {
                    finish_reason = "cancelled".to_string();
                    break;
                }
                let token_value = token.item_at_int32(0)? as u32;
                generated_tokens.push(token_value);
                all_tokens.push(token_value);

                if token_value == eos_token_id {
                    finish_reason = "stop".to_string();
                    break;
                }

                // Decode and emit BEFORE repetition check so the
                // triggering token is streamed to clients
                let token_text = crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                    &mut decode_stream,
                    self.tokenizer.inner(),
                    token_value,
                    &generated_tokens,
                    streamed_text_len,
                );
                streamed_text_len += token_text.len();

                emit(ChatStreamChunk {
                    text: token_text,
                    done: false,
                    finish_reason: None,
                    tool_calls: None,
                    thinking: None,
                    num_tokens: None,
                    prompt_tokens: None,
                    reasoning_tokens: None,
                    raw_text: None,
                    performance: None,
                    is_reasoning: None,
                });

                // Check repetition cutoff (after emit so token is streamed)
                if let Some(reason) = check_repetition_cutoff(
                    &generated_tokens,
                    max_consecutive_tokens,
                    max_ngram_repeats,
                    ngram_size,
                ) {
                    finish_reason = reason.to_string();
                    break;
                }

                // Forward single token
                let token_2d = token.reshape(&[1, 1])?;
                let mut cache = self.kv_caches.take();
                let step_result: Result<MxArray> = {
                    let _ctx = StreamContext::new(generation_stream);
                    self.language_model.forward(&token_2d, &mut cache)
                };
                self.kv_caches = cache;
                let logits = step_result?;

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

                token = sample(&next_logits, Some(sampling_config))?;
                token.eval();

                if (step + 1) % 256 == 0 {
                    synchronize_and_clear_cache();
                }
            }

            // Sync token history with cache state.
            // "length" and "cancelled" break before/at the loop boundary
            // so all tokens in generated_tokens were forwarded.
            // "stop" and "repetition" push then break before forward,
            // so the last token was NOT forwarded.
            if reuse_cache {
                let forwarded = if finish_reason == "stop" || finish_reason == "repetition" {
                    generated_tokens.len().saturating_sub(1)
                } else {
                    generated_tokens.len()
                };
                let mut full_history = prompt_token_ids;
                full_history.extend_from_slice(&generated_tokens[..forwarded]);
                self.cached_token_history = full_history;
                self.cached_cache_offset = self.get_cache_offset();
                // Track image state — mirrors the non-streaming save block.
                self.cached_image_key = if image_bytes.is_empty() {
                    None
                } else {
                    Some(crate::models::qwen3_5::chat_common::compute_image_cache_key(&image_bytes))
                };
            } else {
                self.cached_token_history.clear();
                self.cached_cache_offset = 0;
                self.cached_image_key = None;
            }

            // Final chunk
            let raw_decoded = self.tokenizer.decode_sync(&generated_tokens, true)?;
            let cleaned = raw_decoded.replace("<|im_end|>", "").trim().to_string();
            let (text_after_thinking, thinking) = tools::parse_thinking(&cleaned);
            let (text, tool_calls) = tools::parse_tool_calls(&text_after_thinking);

            // Promote finish_reason to "tool_calls" when valid tool calls parsed
            if tool_calls.iter().any(|tc| tc.status == "ok") {
                finish_reason = "tool_calls".to_string();
            }

            let performance = if let (Some(gen_start), Some(first_tok)) =
                (generation_start, first_token_instant)
            {
                let generation_end = std::time::Instant::now();
                let prefill_toks = prefill_token_count as f64;
                let gen_toks = generated_tokens.len() as f64;
                let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
                let decode_ms = generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
                Some(crate::profiling::PerformanceMetrics {
                    ttft_ms,
                    prefill_tokens_per_second: if ttft_ms > 0.0 {
                        prefill_toks / (ttft_ms / 1000.0)
                    } else {
                        0.0
                    },
                    decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                        (gen_toks - 1.0) / (decode_ms / 1000.0)
                    } else {
                        0.0
                    },
                })
            } else {
                None
            };

            let reasoning_tokens = tools::count_reasoning_tokens(
                &thinking,
                &generated_tokens,
                self.tokenizer.think_end_id(),
            );

            emit(ChatStreamChunk {
                text: text.trim().to_string(),
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                num_tokens: Some(generated_tokens.len() as u32),
                prompt_tokens: Some(prefill_token_count as u32),
                reasoning_tokens: Some(reasoning_tokens),
                raw_text: Some(raw_decoded),
                performance,
                is_reasoning: None,
            });

            Ok(())
        })();

        if let Err(e) = result {
            // Propagate errors through the same stream; the tokio pump task
            // in `QianfanOCRModel::chat_stream` forwards them to the JS
            // callback's error channel.
            let _ = stream_tx.send(Err(e));
        }
    }

    // ========================================================================
    // Session chat API
    // ========================================================================

    /// Resolve the tokenizer id of `<|im_end|>`, the Qwen3 ChatML end-of-turn
    /// marker. Qianfan-OCR sits on Qwen3 for its language model, so the
    /// ChatML wire format applies directly — stopping on `<|im_end|>` keeps
    /// the cached history on a clean delta boundary for the session
    /// continuation paths.
    fn im_end_id(&self) -> Result<u32> {
        self.tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))
    }

    /// Start a new chat session.
    ///
    /// Fully resets the caches and delegates to [`Self::chat_sync_core`]
    /// with `<|im_end|>` as the stop token so the decode loop leaves the
    /// caches on a clean ChatML boundary that subsequent
    /// [`Self::chat_session_continue_sync`] /
    /// [`Self::chat_session_continue_tool_sync`] calls can append a raw
    /// delta on top of.
    ///
    /// Vision-capable: `messages` may carry images (they will be decoded
    /// through the InternViT → bridge pipeline inside `chat_sync_core`,
    /// same as the legacy chat path).
    pub(crate) fn chat_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // Mirror the symmetric guard in `chat_tokens_delta_sync`. The
        // session API only makes sense with cache reuse enabled: if we
        // silently accept `reuse_cache = false`, the post-decode save
        // block wipes the caches we just populated, and the next
        // `chat_session_continue` call fails with a cryptic guard error.
        // Fail fast before mutating any state.
        if config.reuse_cache == Some(false) {
            return Err(Error::from_reason(
                "chat_session_start requires reuse_cache=true (pass ChatConfig { reuse_cache: Some(true), .. } or leave as None). The session API only makes sense with cache reuse enabled.",
            ));
        }

        // Resolve `<|im_end|>` up front so session_continue can rely on the
        // cached history always terminating on a clean ChatML boundary.
        let im_end_id = self.im_end_id()?;

        // Full reset: the session-start path always begins from a clean
        // state. This matches the documented contract that the session is
        // owned end-to-end by the `chat_session_*` surface and
        // intentionally invalidates any prior cache.
        self.reset_caches_sync();

        self.chat_sync_core(messages, config, im_end_id)
    }

    /// Continue an existing chat session with a user turn.
    ///
    /// Builds a ChatML wire-format delta (`\n<|im_start|>user\n...
    /// <|im_end|>\n<|im_start|>assistant\n`), tokenizes it, and prefills
    /// on top of the live caches via [`Self::chat_tokens_delta_sync`].
    ///
    /// Text-only on the delta path: callers that need to change the
    /// image set must restart the session via
    /// [`Self::chat_session_start_sync`]. The `images` parameter is an
    /// opt-in guard that returns an
    /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed error when
    /// non-empty, letting the TS `ChatSession` layer pattern-match the
    /// prefix and route image-changes through a fresh session start.
    pub(crate) fn chat_session_continue_sync(
        &mut self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // Guard 1: text-only delta path.
        if images.as_ref().is_some_and(|v| !v.is_empty()) {
            return Err(Error::from_reason(format!(
                "{}chat_session_continue is text-only; start a new session with chat_session_start to change the image",
                crate::models::qwen3_5::chat_common::IMAGE_CHANGE_RESTART_PREFIX
            )));
        }

        let tokenizer = self.tokenizer.clone();

        // Subject the session path to the same role/content injection
        // sanitization as the legacy chat path so all entry points stay
        // uniform.
        let synthetic =
            crate::models::qwen3_5::chat_common::build_synthetic_user_message(&user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        let enable_thinking = crate::models::qwen3_5::chat_common::resolve_enable_thinking(&config);
        let delta_text = crate::models::qwen3_5::chat_common::build_chatml_continue_delta_text(
            sanitized_user,
            enable_thinking,
        );
        let delta_tokens = tokenizer.encode_sync(&delta_text, Some(false))?;

        self.chat_tokens_delta_sync(delta_tokens, config)
    }

    /// Continue an existing chat session with a tool-result turn.
    ///
    /// Builds a ChatML `<tool_response>`-wrapped delta from `content` and
    /// prefills it on top of the live session caches. The `tool_call_id`
    /// is intentionally dropped from the wire format — Qwen3.5's chat
    /// template identifies tool responses by position and wrapper tags,
    /// not an explicit id. Callers may still log it for bookkeeping.
    ///
    /// Text-only; delegates to [`Self::chat_tokens_delta_sync`] which
    /// inherits the same text-only-delta invariant (errors if the
    /// session currently holds image state).
    pub(crate) fn chat_session_continue_tool_sync(
        &mut self,
        tool_call_id: String,
        content: String,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        let tokenizer = self.tokenizer.clone();

        let enable_thinking = crate::models::qwen3_5::chat_common::resolve_enable_thinking(&config);
        let delta_text = crate::models::qwen3_5::chat_common::build_chatml_tool_delta_text(
            &tool_call_id,
            &content,
            enable_thinking,
        );
        let delta_tokens = tokenizer.encode_sync(&delta_text, Some(false))?;

        self.chat_tokens_delta_sync(delta_tokens, config)
    }

    /// Prefill a pre-tokenized delta on top of the existing KV caches and
    /// run the decode loop. Text-only session primitive used by
    /// [`Self::chat_session_continue_sync`] and
    /// [`Self::chat_session_continue_tool_sync`].
    ///
    /// Uses `<|im_end|>` as the eos token so the cached history continues
    /// to end on a clean ChatML boundary for the next turn.
    pub(crate) fn chat_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // --- Five guards (mirrors Gemma4/Qwen3). ---
        // The delta path is a session-reuse operation by construction: it
        // prefills on top of the existing caches. `reuse_cache = Some(false)`
        // would make the post-decode save block wipe those caches +
        // `cached_token_history`, making the delta turn both depend on
        // and then destroy the session. Reject early so no state is
        // mutated.
        if config.reuse_cache == Some(false) {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires reuse_cache to be enabled; \
                 the delta path operates on session state by construction",
            ));
        }
        if self.cached_token_history.is_empty() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires an initialized session (call chatSessionStart first)",
            ));
        }
        if delta_tokens.is_empty() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires at least one delta token",
            ));
        }
        if self.cached_image_key.is_some() {
            return Err(Error::from_reason(format!(
                "{}chat_tokens_delta_sync cannot be called while image state is cached; call chatSessionStart with the new images instead",
                crate::models::qwen3_5::chat_common::IMAGE_CHANGE_RESTART_PREFIX
            )));
        }
        if self.kv_caches.is_none() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires live KV caches; call chatSessionStart first",
            ));
        }

        // Session path: use `<|im_end|>` as eos, NOT config.eos_token_id.
        // This keeps the cached history aligned on a clean ChatML
        // boundary for the next `chat_session_continue*` call.
        let eos_token_id = self.im_end_id()?;

        let max_new_tokens = config.max_new_tokens.unwrap_or(512);
        let temperature = config.temperature.unwrap_or(0.0);
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let presence_penalty = config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = config.presence_context_size.unwrap_or(20);
        let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = config.frequency_context_size.unwrap_or(20);
        let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
        let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
        let ngram_size = config.ngram_size.unwrap_or(64);
        let report_perf = config.report_performance.unwrap_or(false);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Build the full token history = cached_history + delta. Used as
        // penalty context AND the snapshot saved back into
        // `cached_token_history` at the end.
        let mut all_tokens: Vec<u32> =
            Vec::with_capacity(self.cached_token_history.len() + delta_tokens.len());
        all_tokens.extend_from_slice(&self.cached_token_history);
        all_tokens.extend_from_slice(&delta_tokens);

        let prefill_token_count = all_tokens.len();

        // Text-only prefill of the delta on top of the existing caches.
        let generation_stream = Stream::new(DeviceType::Gpu);
        let input_ids = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
        let merged_embeds = {
            let _ctx = StreamContext::new(generation_stream);
            self.language_model.get_embeddings(&input_ids)?
        };

        let mut cache = self.kv_caches.take();
        let prefill_result: Result<MxArray> = {
            let _ctx = StreamContext::new(generation_stream);
            self.language_model
                .forward_from_embeddings(&merged_embeds, &mut cache)
        };
        self.kv_caches = cache;
        let prefill_logits = prefill_result?;

        prefill_logits.eval();
        synchronize_and_clear_cache();

        // Last logits for first sampled token
        let prefill_seq = prefill_logits.shape()?[1];
        let mut last_logits = prefill_logits
            .slice_axis(1, prefill_seq - 1, prefill_seq)?
            .squeeze(Some(&[0, 1]))?;

        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

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

        let mut token = sample(&last_logits, Some(sampling_config))?;
        token.eval();

        let first_token_instant = generation_start.map(|_| std::time::Instant::now());

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
        let mut finish_reason = "length".to_string();

        for step in 0..max_new_tokens {
            let token_value = token.item_at_int32(0)? as u32;
            generated_tokens.push(token_value);
            all_tokens.push(token_value);

            if token_value == eos_token_id {
                finish_reason = "stop".to_string();
                break;
            }

            if let Some(reason) = check_repetition_cutoff(
                &generated_tokens,
                max_consecutive_tokens,
                max_ngram_repeats,
                ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }

            let token_2d = token.reshape(&[1, 1])?;
            let mut cache = self.kv_caches.take();
            let step_result: Result<MxArray> = {
                let _ctx = StreamContext::new(generation_stream);
                self.language_model.forward(&token_2d, &mut cache)
            };
            self.kv_caches = cache;
            let logits = step_result?;

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

            token = sample(&next_logits, Some(sampling_config))?;
            token.eval();

            if (step + 1) % 256 == 0 {
                synchronize_and_clear_cache();
            }
        }

        // Sync token history with cache state (same drop-last idiom as
        // `chat_sync_core`: terminal stop/repetition tokens were sampled
        // but never forwarded into the cache).
        let forwarded = if finish_reason == "stop" || finish_reason == "repetition" {
            generated_tokens.len().saturating_sub(1)
        } else {
            generated_tokens.len()
        };
        let mut full_history =
            Vec::with_capacity(self.cached_token_history.len() + delta_tokens.len() + forwarded);
        full_history.extend_from_slice(&self.cached_token_history);
        full_history.extend_from_slice(&delta_tokens);
        full_history.extend_from_slice(&generated_tokens[..forwarded]);
        self.cached_token_history = full_history;
        self.cached_cache_offset = self.get_cache_offset();
        // The delta path is text-only (guarded above); the image key
        // invariant is preserved by construction, but we still explicitly
        // leave it as-is (always None here, because guard 4 rejected
        // any `Some(_)` state on entry).

        // Decode + tool/thinking parsing
        let raw_decoded = self.tokenizer.decode_sync(&generated_tokens, true)?;
        let cleaned = raw_decoded.replace("<|im_end|>", "").trim().to_string();

        let (text_after_thinking, thinking) = tools::parse_thinking(&cleaned);
        let (text, tool_calls) = tools::parse_tool_calls(&text_after_thinking);

        if tool_calls.iter().any(|tc| tc.status == "ok") {
            finish_reason = "tool_calls".to_string();
        }

        let performance =
            if let (Some(gen_start), Some(first_tok)) = (generation_start, first_token_instant) {
                let generation_end = std::time::Instant::now();
                let prefill_toks = prefill_token_count as f64;
                let gen_toks = generated_tokens.len() as f64;
                let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
                let decode_ms = generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
                Some(crate::profiling::PerformanceMetrics {
                    ttft_ms,
                    prefill_tokens_per_second: if ttft_ms > 0.0 {
                        prefill_toks / (ttft_ms / 1000.0)
                    } else {
                        0.0
                    },
                    decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                        (gen_toks - 1.0) / (decode_ms / 1000.0)
                    } else {
                        0.0
                    },
                })
            } else {
                None
            };

        let reasoning_tokens = tools::count_reasoning_tokens(
            &thinking,
            &generated_tokens,
            self.tokenizer.think_end_id(),
        );

        Ok(ChatResult {
            text: text.trim().to_string(),
            tool_calls,
            thinking,
            num_tokens: generated_tokens.len() as u32,
            prompt_tokens: prefill_token_count as u32,
            reasoning_tokens,
            finish_reason,
            raw_text: raw_decoded,
            performance,
        })
    }

    /// Streaming variant of [`Self::chat_session_start_sync`].
    pub(crate) fn chat_stream_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_session_start cancelled before start",
            );
            return;
        }

        if config.reuse_cache == Some(false) {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_session_start requires reuse_cache=true (leave as None or set to true). The session API only makes sense with cache reuse enabled.",
            );
            return;
        }

        let im_end_id = match self.im_end_id() {
            Ok(id) => id,
            Err(e) => {
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        // Full reset: the session always starts clean.
        self.reset_caches_sync();

        self.chat_stream_sync_core(messages, config, stream_tx, cancelled, im_end_id);
    }

    /// Streaming variant of [`Self::chat_session_continue_sync`].
    pub(crate) fn chat_stream_session_continue_sync(
        &mut self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_session_continue cancelled before start",
            );
            return;
        }

        if images.as_ref().is_some_and(|v| !v.is_empty()) {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                &format!(
                    "{}chat_stream_session_continue is text-only; start a new session with chat_stream_session_start to change the image",
                    crate::models::qwen3_5::chat_common::IMAGE_CHANGE_RESTART_PREFIX
                ),
            );
            return;
        }

        let tokenizer = self.tokenizer.clone();

        let synthetic =
            crate::models::qwen3_5::chat_common::build_synthetic_user_message(&user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        let enable_thinking = crate::models::qwen3_5::chat_common::resolve_enable_thinking(&config);
        let delta_text = crate::models::qwen3_5::chat_common::build_chatml_continue_delta_text(
            sanitized_user,
            enable_thinking,
        );

        let delta_tokens = match tokenizer.encode_sync(&delta_text, Some(false)) {
            Ok(t) => t,
            Err(e) => {
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        self.chat_stream_tokens_delta_sync(delta_tokens, config, stream_tx, cancelled);
    }

    /// Streaming variant of [`Self::chat_session_continue_tool_sync`].
    pub(crate) fn chat_stream_session_continue_tool_sync(
        &mut self,
        tool_call_id: String,
        content: String,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_session_continue_tool cancelled before start",
            );
            return;
        }

        let tokenizer = self.tokenizer.clone();

        let enable_thinking = crate::models::qwen3_5::chat_common::resolve_enable_thinking(&config);
        let delta_text = crate::models::qwen3_5::chat_common::build_chatml_tool_delta_text(
            &tool_call_id,
            &content,
            enable_thinking,
        );

        let delta_tokens = match tokenizer.encode_sync(&delta_text, Some(false)) {
            Ok(t) => t,
            Err(e) => {
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        self.chat_stream_tokens_delta_sync(delta_tokens, config, stream_tx, cancelled);
    }

    /// Streaming analog of [`Self::chat_tokens_delta_sync`]: prefill the
    /// caller-provided delta tokens on top of the existing KV caches and
    /// stream the reply through `stream_tx`.
    ///
    /// Applies the same five guards as the non-streaming path (routed via
    /// `send_stream_error` so they surface as an error-item rather than a
    /// fake done chunk). Uses `<|im_end|>` as the eos token so the cached
    /// history continues to end on a clean ChatML boundary.
    pub(crate) fn chat_stream_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta cancelled before start",
            );
            return;
        }

        if config.reuse_cache == Some(false) {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                "chat_tokens_delta_sync requires reuse_cache to be enabled; \
                 the delta path operates on session state by construction",
            );
            return;
        }
        if self.cached_token_history.is_empty() {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires an initialized session (call chatStreamSessionStart first)",
            );
            return;
        }
        if delta_tokens.is_empty() {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires at least one delta token",
            );
            return;
        }
        if self.cached_image_key.is_some() {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                &format!(
                    "{}chat_stream_tokens_delta cannot be called while image state is cached; call chatStreamSessionStart with the new images instead",
                    crate::models::qwen3_5::chat_common::IMAGE_CHANGE_RESTART_PREFIX
                ),
            );
            return;
        }
        if self.kv_caches.is_none() {
            crate::models::qwen3_5::chat_common::send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires live KV caches; call chatStreamSessionStart first",
            );
            return;
        }

        let result =
            self.chat_stream_tokens_delta_sync_inner(delta_tokens, config, &stream_tx, &cancelled);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    /// Inner body of [`Self::chat_stream_tokens_delta_sync`]: prefill
    /// delta tokens on top of the live caches, then run the streaming
    /// decode loop.
    fn chat_stream_tokens_delta_sync_inner(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        stream_tx: &StreamTx<ChatStreamChunk>,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<()> {
        let sender = StreamSender(stream_tx.clone());
        let emit = |chunk: ChatStreamChunk| {
            sender.call(Ok(chunk), ThreadsafeFunctionCallMode::NonBlocking);
        };

        let eos_token_id = self.im_end_id()?;

        let max_new_tokens = config.max_new_tokens.unwrap_or(512);
        let temperature = config.temperature.unwrap_or(0.0);
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let presence_penalty = config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = config.presence_context_size.unwrap_or(20);
        let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = config.frequency_context_size.unwrap_or(20);
        let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
        let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
        let ngram_size = config.ngram_size.unwrap_or(64);
        let report_perf = config.report_performance.unwrap_or(false);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Build full token history = cached_history + delta.
        let mut all_tokens: Vec<u32> =
            Vec::with_capacity(self.cached_token_history.len() + delta_tokens.len());
        all_tokens.extend_from_slice(&self.cached_token_history);
        all_tokens.extend_from_slice(&delta_tokens);

        let prefill_token_count = all_tokens.len();

        // Text-only prefill of the delta on top of the existing caches.
        let generation_stream = Stream::new(DeviceType::Gpu);
        let input_ids = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
        let merged_embeds = {
            let _ctx = StreamContext::new(generation_stream);
            self.language_model.get_embeddings(&input_ids)?
        };

        let mut cache = self.kv_caches.take();
        let prefill_result: Result<MxArray> = {
            let _ctx = StreamContext::new(generation_stream);
            self.language_model
                .forward_from_embeddings(&merged_embeds, &mut cache)
        };
        self.kv_caches = cache;
        let prefill_logits = prefill_result?;

        prefill_logits.eval();
        synchronize_and_clear_cache();

        let prefill_seq = prefill_logits.shape()?[1];
        let mut last_logits = prefill_logits
            .slice_axis(1, prefill_seq - 1, prefill_seq)?
            .squeeze(Some(&[0, 1]))?;

        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

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

        let mut token = sample(&last_logits, Some(sampling_config))?;
        token.eval();

        let first_token_instant = generation_start.map(|_| std::time::Instant::now());

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
        let mut finish_reason = "length".to_string();

        // Stateful decoder for correct multi-byte/CJK streaming.
        let mut decode_stream = self.tokenizer.inner().decode_stream(true);
        let mut streamed_text_len: usize = 0;

        for step in 0..max_new_tokens {
            if cancelled.load(Ordering::Relaxed) {
                finish_reason = "cancelled".to_string();
                break;
            }
            let token_value = token.item_at_int32(0)? as u32;
            generated_tokens.push(token_value);
            all_tokens.push(token_value);

            if token_value == eos_token_id {
                finish_reason = "stop".to_string();
                break;
            }

            let token_text = crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                &mut decode_stream,
                self.tokenizer.inner(),
                token_value,
                &generated_tokens,
                streamed_text_len,
            );
            streamed_text_len += token_text.len();

            emit(ChatStreamChunk {
                text: token_text,
                done: false,
                finish_reason: None,
                tool_calls: None,
                thinking: None,
                num_tokens: None,
                prompt_tokens: None,
                reasoning_tokens: None,
                raw_text: None,
                performance: None,
                is_reasoning: None,
            });

            if let Some(reason) = check_repetition_cutoff(
                &generated_tokens,
                max_consecutive_tokens,
                max_ngram_repeats,
                ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }

            let token_2d = token.reshape(&[1, 1])?;
            let mut cache = self.kv_caches.take();
            let step_result: Result<MxArray> = {
                let _ctx = StreamContext::new(generation_stream);
                self.language_model.forward(&token_2d, &mut cache)
            };
            self.kv_caches = cache;
            let logits = step_result?;

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

            token = sample(&next_logits, Some(sampling_config))?;
            token.eval();

            if (step + 1) % 256 == 0 {
                synchronize_and_clear_cache();
            }
        }

        // Save token history with the drop-last idiom (matches
        // non-streaming path).
        let forwarded = if finish_reason == "stop" || finish_reason == "repetition" {
            generated_tokens.len().saturating_sub(1)
        } else {
            generated_tokens.len()
        };
        let mut full_history =
            Vec::with_capacity(self.cached_token_history.len() + delta_tokens.len() + forwarded);
        full_history.extend_from_slice(&self.cached_token_history);
        full_history.extend_from_slice(&delta_tokens);
        full_history.extend_from_slice(&generated_tokens[..forwarded]);
        self.cached_token_history = full_history;
        self.cached_cache_offset = self.get_cache_offset();

        // Final chunk
        let raw_decoded = self.tokenizer.decode_sync(&generated_tokens, true)?;
        let cleaned = raw_decoded.replace("<|im_end|>", "").trim().to_string();
        let (text_after_thinking, thinking) = tools::parse_thinking(&cleaned);
        let (text, tool_calls) = tools::parse_tool_calls(&text_after_thinking);

        if tool_calls.iter().any(|tc| tc.status == "ok") {
            finish_reason = "tool_calls".to_string();
        }

        let performance =
            if let (Some(gen_start), Some(first_tok)) = (generation_start, first_token_instant) {
                let generation_end = std::time::Instant::now();
                let prefill_toks = prefill_token_count as f64;
                let gen_toks = generated_tokens.len() as f64;
                let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
                let decode_ms = generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
                Some(crate::profiling::PerformanceMetrics {
                    ttft_ms,
                    prefill_tokens_per_second: if ttft_ms > 0.0 {
                        prefill_toks / (ttft_ms / 1000.0)
                    } else {
                        0.0
                    },
                    decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                        (gen_toks - 1.0) / (decode_ms / 1000.0)
                    } else {
                        0.0
                    },
                })
            } else {
                None
            };

        let reasoning_tokens = tools::count_reasoning_tokens(
            &thinking,
            &generated_tokens,
            self.tokenizer.think_end_id(),
        );

        emit(ChatStreamChunk {
            text: text.trim().to_string(),
            done: true,
            finish_reason: Some(finish_reason),
            tool_calls: Some(tool_calls),
            thinking,
            num_tokens: Some(generated_tokens.len() as u32),
            prompt_tokens: Some(prefill_token_count as u32),
            reasoning_tokens: Some(reasoning_tokens),
            raw_text: Some(raw_decoded),
            performance,
            is_reasoning: None,
        });

        Ok(())
    }

    /// Low-level token generation given pre-tokenized input.
    ///
    /// Port of the legacy `generate()` body — always starts with a fresh
    /// cache (clears `kv_caches`, `cached_token_history`, and
    /// `cached_cache_offset`) and runs a pure greedy-style decode loop.
    fn generate_sync(
        &mut self,
        input_ids: &MxArray,
        max_new_tokens: i32,
        temperature: f64,
    ) -> Result<Vec<u32>> {
        let generation_stream = Stream::new(DeviceType::Gpu);

        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };

        self.kv_caches = None;
        self.init_kv_caches();

        // generate() always does fresh generation — clear cached metadata
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_cache_offset = 0;

        // Prefill
        let mut cache = self.kv_caches.take();
        let prefill_result: Result<MxArray> = {
            let _ctx = StreamContext::new(generation_stream);
            self.language_model.forward(input_ids, &mut cache)
        };
        self.kv_caches = cache;
        let logits = prefill_result?;

        // Eval prefill logits -- caches materialize through dependency graph
        logits.eval();
        synchronize_and_clear_cache();

        let seq_len = logits.shape()?[1];
        let last_logits = logits
            .slice_axis(1, seq_len - 1, seq_len)?
            .squeeze(Some(&[0, 1]))?;

        let mut token = sample(&last_logits, Some(sampling_config))?;

        let eos_token_id = self.config.eos_token_id;
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);

        for step in 0..max_new_tokens {
            token.eval();
            let token_value = token.item_at_int32(0)? as u32;
            generated.push(token_value);

            if token_value == eos_token_id as u32 {
                break;
            }

            let token_2d = token.reshape(&[1, 1])?;
            let mut cache = self.kv_caches.take();
            let step_result: Result<MxArray> = {
                let _ctx = StreamContext::new(generation_stream);
                self.language_model.forward(&token_2d, &mut cache)
            };
            self.kv_caches = cache;
            let logits = step_result?;

            let next_logits = logits.squeeze(Some(&[0, 1]))?;
            token = sample(&next_logits, Some(sampling_config))?;
            token.eval();

            if (step + 1) % 256 == 0 {
                synchronize_and_clear_cache();
            }
        }

        Ok(generated)
    }
}

#[napi]
impl QianfanOCRModel {
    /// Create a new QianfanOCRModel from config (uninitialized, no weights).
    ///
    /// This constructor path does not spawn a model thread — the returned
    /// instance is only useful for config inspection. Call
    /// [`QianfanOCRModel::load`] to actually run inference.
    #[napi(constructor)]
    pub fn new(config: QianfanOCRConfig) -> Self {
        Self {
            thread: None,
            config,
            initialized: false,
        }
    }

    /// Returns true if weights have been loaded via `load()`.
    #[napi(getter)]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Load a QianfanOCRModel from a directory.
    ///
    /// Reads config.json, loads SafeTensors weights (single or sharded),
    /// builds vision encoder, bridge, and language model, and loads tokenizer.
    /// All heavy work runs on the dedicated model thread.
    #[napi]
    pub fn load<'env>(
        env: &'env Env,
        model_path: String,
    ) -> Result<PromiseRaw<'env, QianfanOCRModel>> {
        env.spawn_future_with_callback(
            async move {
                let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
                    move || {
                        let inner = load_qianfan_ocr_inner_from_dir(&model_path)?;
                        let config = inner.config.clone();
                        Ok((inner, config))
                    },
                    handle_qianfan_ocr_cmd,
                );

                let config = init_rx
                    .await
                    .map_err(|_| napi::Error::from_reason("Model thread exited during load"))??;

                Ok((thread, config))
            },
            |_env, (thread, config)| {
                Ok(QianfanOCRModel {
                    thread: Some(thread),
                    config,
                    initialized: true,
                })
            },
        )
    }

    /// Generate text tokens given pre-tokenized input.
    ///
    /// Lower-level API — prefer the session chat methods
    /// (`chatSessionStart` / `chatSessionContinue` and their streaming
    /// variants) for typical usage.
    #[napi]
    pub async fn generate(
        &self,
        input_ids: &MxArray,
        max_new_tokens: Option<i32>,
        temperature: Option<f64>,
    ) -> Result<Vec<u32>> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let max_new_tokens = max_new_tokens.unwrap_or(256);
        let temperature = temperature.unwrap_or(0.0);
        let input_ids = input_ids.clone();

        crate::model_thread::send_and_await(thread, |reply| QianfanOCRCmd::Generate {
            input_ids,
            max_new_tokens,
            temperature,
            reply,
        })
        .await
    }

    /// Reset KV caches and token history.
    #[napi]
    pub fn reset_caches(&self) -> Result<()> {
        let Some(thread) = self.thread.as_ref() else {
            // Uninitialized model — nothing to reset.
            return Ok(());
        };
        crate::model_thread::send_and_block(thread, |reply| QianfanOCRCmd::ResetCaches { reply })
    }

    /// Start a new chat session.
    ///
    /// Runs the full chat template once, decodes until `<|im_end|>`,
    /// and leaves the KV caches on a clean turn boundary so subsequent
    /// `chatSessionContinue` / `chatSessionContinueTool` calls can
    /// append a raw ChatML delta on top without re-rendering the chat
    /// template.
    ///
    /// Qianfan-OCR is always a VLM (InternViT + Qwen3 language model), so
    /// this entry point accepts images in `messages` without the text-only
    /// fast-fail used by plain language models.
    #[napi]
    pub async fn chat_session_start(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(thread, |reply| QianfanOCRCmd::ChatSessionStart {
            messages,
            config,
            reply,
        })
        .await
    }

    /// Continue an existing chat session with a new user message.
    ///
    /// Appends a raw ChatML user/assistant delta to the session's cached
    /// KV state, then decodes the model reply. Stops on `<|im_end|>` so
    /// the cache remains on a clean turn boundary for the next turn.
    ///
    /// Requires a live session started via `chatSessionStart`. Errors
    /// if the session is empty or if `config.reuse_cache` is
    /// explicitly set to `false`.
    ///
    /// `images` is an opt-in guard parameter: when non-empty the native
    /// side returns an error whose message begins with
    /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` so the TypeScript
    /// `ChatSession` layer can catch the prefix and route image-changes
    /// back through a fresh `chatSessionStart` uniformly across all
    /// model backends. Qianfan-OCR is a VLM but the continue path cannot
    /// splice new vision features into a live KV cache — image changes
    /// always require a fresh session start.
    #[napi(
        ts_args_type = "userMessage: string, images: Uint8Array[] | null | undefined, config: ChatConfig | null | undefined"
    )]
    pub async fn chat_session_continue(
        &self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(thread, |reply| QianfanOCRCmd::ChatSessionContinue {
            user_message,
            images,
            config,
            reply,
        })
        .await
    }

    /// Continue an existing chat session with a tool-result turn.
    ///
    /// Builds a ChatML `<tool_response>` delta from `tool_call_id` and
    /// `content` and prefills it on top of the live session caches, then
    /// decodes the model reply. Stops on `<|im_end|>` so the cache stays
    /// on a clean turn boundary for the next turn.
    ///
    /// Requires a live session started via `chatSessionStart`.
    #[napi]
    pub async fn chat_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(thread, |reply| {
            QianfanOCRCmd::ChatSessionContinueTool {
                tool_call_id,
                content,
                config,
                reply,
            }
        })
        .await
    }

    /// Streaming variant of `chatSessionStart`.
    #[napi(
        ts_args_type = "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_start(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        thread.send(QianfanOCRCmd::ChatStreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;

        let callback = Arc::new(callback);
        tokio::spawn(async move {
            while let Some(result) = stream_rx.recv().await {
                callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(ChatStreamHandle { cancelled })
    }

    /// Streaming variant of `chatSessionContinue`.
    #[napi(
        ts_args_type = "userMessage: string, images: Uint8Array[] | null | undefined, config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_continue(
        &self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        thread.send(QianfanOCRCmd::ChatStreamSessionContinue {
            user_message,
            images,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;

        let callback = Arc::new(callback);
        tokio::spawn(async move {
            while let Some(result) = stream_rx.recv().await {
                callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(ChatStreamHandle { cancelled })
    }

    /// Streaming variant of `chatSessionContinueTool`.
    #[napi(
        ts_args_type = "toolCallId: string, content: string, config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        thread.send(QianfanOCRCmd::ChatStreamSessionContinueTool {
            tool_call_id,
            content,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;

        let callback = Arc::new(callback);
        tokio::spawn(async move {
            while let Some(result) = stream_rx.recv().await {
                callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(ChatStreamHandle { cancelled })
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Parse config.json into QianfanOCRConfig.
fn parse_config_json(raw: &Value) -> QianfanOCRConfig {
    let vision_raw = &raw["vision_config"];
    let vision_config = InternVisionConfig {
        hidden_size: vision_raw["hidden_size"].as_i64().unwrap_or(1024) as i32,
        intermediate_size: vision_raw["intermediate_size"].as_i64().unwrap_or(4096) as i32,
        num_hidden_layers: vision_raw["num_hidden_layers"].as_i64().unwrap_or(24) as i32,
        num_attention_heads: vision_raw["num_attention_heads"].as_i64().unwrap_or(16) as i32,
        num_channels: vision_raw["num_channels"].as_i64().unwrap_or(3) as i32,
        image_size: vision_raw["image_size"].as_i64().unwrap_or(448) as i32,
        patch_size: vision_raw["patch_size"].as_i64().unwrap_or(14) as i32,
        layer_norm_eps: vision_raw["layer_norm_eps"].as_f64().unwrap_or(1e-6),
        qkv_bias: vision_raw["qkv_bias"].as_bool().unwrap_or(true),
        drop_path_rate: vision_raw["drop_path_rate"].as_f64().unwrap_or(0.0),
    };

    let llm_raw = &raw["llm_config"];
    let llm_config = Qwen3LMConfig {
        hidden_size: llm_raw["hidden_size"].as_i64().unwrap_or(2560) as i32,
        num_hidden_layers: llm_raw["num_hidden_layers"].as_i64().unwrap_or(36) as i32,
        intermediate_size: llm_raw["intermediate_size"].as_i64().unwrap_or(9728) as i32,
        num_attention_heads: llm_raw["num_attention_heads"].as_i64().unwrap_or(32) as i32,
        num_key_value_heads: llm_raw["num_key_value_heads"].as_i64().unwrap_or(8) as i32,
        head_dim: llm_raw["head_dim"].as_i64().unwrap_or(128) as i32,
        rms_norm_eps: llm_raw["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        vocab_size: llm_raw["vocab_size"].as_i64().unwrap_or(153678) as i32,
        max_position_embeddings: llm_raw["max_position_embeddings"].as_i64().unwrap_or(32768)
            as i32,
        rope_theta: llm_raw["rope_theta"].as_f64().unwrap_or(5_000_000.0),
        use_qk_norm: llm_raw["use_qk_norm"].as_bool().unwrap_or(true),
        tie_word_embeddings: llm_raw["tie_word_embeddings"].as_bool().unwrap_or(false),
    };

    QianfanOCRConfig {
        vision_config,
        llm_config,
        model_type: raw["model_type"]
            .as_str()
            .unwrap_or("internvl_chat")
            .to_string(),
        img_context_token_id: raw["img_context_token_id"].as_i64().unwrap_or(151671) as i32,
        img_start_token_id: raw["img_start_token_id"].as_i64().unwrap_or(151669) as i32,
        img_end_token_id: raw["img_end_token_id"].as_i64().unwrap_or(151670) as i32,
        eos_token_id: raw["eos_token_id"].as_i64().unwrap_or(151645) as i32,
        select_layer: raw["select_layer"].as_i64().unwrap_or(-1) as i32,
        ps_version: raw["ps_version"].as_str().unwrap_or("v2").to_string(),
        downsample_ratio: raw["downsample_ratio"].as_f64().unwrap_or(0.5),
        dynamic_image_size: raw["dynamic_image_size"].as_bool().unwrap_or(true),
        use_thumbnail: raw["use_thumbnail"].as_bool().unwrap_or(true),
        max_dynamic_patch: raw["max_dynamic_patch"].as_i64().unwrap_or(12) as i32,
        min_dynamic_patch: raw["min_dynamic_patch"].as_i64().unwrap_or(1) as i32,
    }
}

/// Load SafeTensors weights from a model directory (single or sharded).
fn load_safetensors_weights(path: &Path) -> Result<HashMap<String, MxArray>> {
    let single = path.join("model.safetensors");
    let mut all_weights: HashMap<String, MxArray> = HashMap::new();

    if single.exists() {
        let st = SafeTensorsFile::load(&single)?;
        info!(
            "  Loading {} tensors from model.safetensors",
            st.tensor_names().len()
        );
        all_weights = st.load_tensors(&single)?;
    } else {
        // Try sharded format
        let mut shard_index = 1;
        loop {
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
                    let st = SafeTensorsFile::load(&shard_path)?;
                    let shard_weights = st.load_tensors(&shard_path)?;
                    all_weights.extend(shard_weights);
                    shard_index += 1;
                }
                None => {
                    if shard_index == 1 {
                        return Err(Error::new(
                            Status::InvalidArg,
                            format!("No SafeTensors files found in {}", path.display()),
                        ));
                    }
                    break;
                }
            }
        }
    }

    Ok(all_weights)
}

/// Load a `QianfanOCRInner` from a model directory.
///
/// Runs synchronously on the dedicated model thread inside
/// `ModelThread::spawn_with_init`. Parses `config.json`, loads
/// SafeTensors weights (single or sharded), transforms key formats if
/// still in HuggingFace layout, builds the InternViT vision encoder,
/// MLP bridge, and Qwen3 language model, and loads the tokenizer.
///
/// A tokenizer is required — unlike the paddleocr_vl path, Qianfan-OCR
/// has no `set_tokenizer()` NAPI method, so the model directory must
/// contain `tokenizer.json` for any of the session chat methods
/// (`chat_session_start`, `chat_session_continue`, their streaming
/// variants, and `chat_session_continue_tool`) to work. The loader
/// returns an error up front if `tokenizer.json` is missing rather
/// than deferring the failure to the first session call.
fn load_qianfan_ocr_inner_from_dir(model_path: &str) -> Result<QianfanOCRInner> {
    let path = Path::new(model_path);

    if !path.exists() {
        return Err(napi::Error::from_reason(format!(
            "Model path does not exist: {}",
            model_path
        )));
    }

    // --- Parse config.json ---
    let config_path = path.join("config.json");
    if !config_path.exists() {
        return Err(napi::Error::from_reason(format!(
            "Config file not found: {}",
            config_path.display()
        )));
    }

    let config_data = fs::read_to_string(&config_path)?;
    let raw: Value = serde_json::from_str(&config_data)?;

    let config = parse_config_json(&raw);

    info!(
        "Loading Qianfan-OCR model from: {} (vision: {} layers, LM: {} layers)",
        model_path, config.vision_config.num_hidden_layers, config.llm_config.num_hidden_layers
    );

    // --- Load SafeTensors weights ---
    let all_weights = load_safetensors_weights(path)?;
    info!("  Loaded {} total tensors", all_weights.len());

    // Transform keys if still in HuggingFace format (has vision_model. prefix)
    let needs_transform = all_weights.keys().any(|k| k.starts_with("vision_model."));
    let weights = if needs_transform {
        info!("  Transforming HuggingFace keys to internal format...");
        load_qianfan_ocr_weights(all_weights)?
    } else {
        info!("  Keys already in MLX format, skipping transformation");
        all_weights
    };

    info!("Building Qianfan-OCR model from weights...");

    // --- Build vision encoder ---
    info!(
        "  Building vision encoder ({} layers)...",
        config.vision_config.num_hidden_layers
    );
    let vision = InternViTModel::build(
        &weights,
        "vision",
        &config.vision_config,
        config.select_layer,
    )?;

    // --- Build bridge ---
    info!("  Building MLP bridge...");
    let bridge = InternVLBridge::build(&weights, "bridge", config.downsample_ratio)?;

    // --- Build language model ---
    info!(
        "  Building language model ({} layers)...",
        config.llm_config.num_hidden_layers
    );
    let language_model = InternVLLanguageModel::build(&weights, "lm", &config.llm_config)?;

    // --- Load tokenizer ---
    let tokenizer_path = Path::new(model_path).join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        info!("  Loading tokenizer from {}", tokenizer_path.display());
        Arc::new(Qwen3Tokenizer::load_from_file_sync(
            tokenizer_path
                .to_str()
                .ok_or_else(|| Error::from_reason("Non-UTF-8 tokenizer path"))?,
        )?)
    } else {
        return Err(Error::from_reason(format!(
            "Tokenizer not found: {}",
            tokenizer_path.display()
        )));
    };

    info!(
        "Qianfan-OCR model loaded: vision={} layers, LM={} layers, {} total weights",
        config.vision_config.num_hidden_layers,
        config.llm_config.num_hidden_layers,
        weights.len()
    );

    Ok(QianfanOCRInner {
        config,
        vision,
        bridge,
        language_model,
        tokenizer,
        kv_caches: None,
        cached_token_history: Vec::new(),
        cached_image_key: None,
        cached_cache_offset: 0,
    })
}

/// Merge vision features into text embeddings at image placeholder positions.
///
/// Replaces positions in `text_embeddings` where `input_ids == img_context_token_id`
/// with corresponding vision features from `vision_features`.
fn merge_vision_features(
    input_ids: &MxArray,
    text_embeddings: &MxArray,
    vision_features: &MxArray,
    img_context_token_id: i32,
) -> Result<MxArray> {
    let input_shape = input_ids.shape()?;
    let batch_size = input_shape[0];

    let image_token = MxArray::scalar_int(img_context_token_id)?;
    let image_positions = input_ids.equal(&image_token)?;

    let embed_shape = text_embeddings.shape()?;
    let hidden_dim = embed_shape[2];

    let mut batch_outputs: Vec<MxArray> = Vec::new();
    let mut feature_start_idx = 0i64;

    for batch_idx in 0..batch_size {
        let batch_mask = image_positions.slice_axis(0, batch_idx, batch_idx + 1)?;
        let batch_mask = batch_mask.squeeze(Some(&[0]))?;

        let mask_sum = batch_mask.sum(None, None)?;
        let num_positions = mask_sum.to_int32()?[0] as i64;

        if num_positions > 0 {
            let batch_features = vision_features.slice_axis(
                0,
                feature_start_idx,
                feature_start_idx + num_positions,
            )?;

            let batch_embeds = text_embeddings.slice_axis(0, batch_idx, batch_idx + 1)?;
            let batch_embeds = batch_embeds.squeeze(Some(&[0]))?;

            let mask_int = batch_mask.astype(crate::array::DType::Int32)?;
            let cumsum = mask_int.cumsum(0)?;

            let ones = MxArray::scalar_int(1)?;
            let feature_indices = cumsum.sub(&ones)?;
            let zeros =
                MxArray::zeros(&feature_indices.shape()?, Some(crate::array::DType::Int32))?;
            let feature_indices = batch_mask.where_(&feature_indices, &zeros)?;

            let gathered_features = batch_features.take(&feature_indices, 0)?;

            let mask_expanded = batch_mask.reshape(&[-1, 1])?;
            let mask_expanded =
                MxArray::broadcast_to(&mask_expanded, &[batch_mask.shape()?[0], hidden_dim])?;

            let batch_output = mask_expanded.where_(&gathered_features, &batch_embeds)?;
            batch_outputs.push(batch_output);
            feature_start_idx += num_positions;
        } else {
            let batch_embeds = text_embeddings.slice_axis(0, batch_idx, batch_idx + 1)?;
            batch_outputs.push(batch_embeds.squeeze(Some(&[0]))?);
        }
    }

    let refs: Vec<&MxArray> = batch_outputs.iter().collect();
    MxArray::stack(refs, Some(0))
}

/// Stack pixel values from multiple ProcessedImages into a single array.
fn stack_processed_images(
    images: &[crate::models::qianfan_ocr::processing::ProcessedImage],
) -> Result<MxArray> {
    if images.len() == 1 {
        return Ok(images[0].pixel_values.clone());
    }

    // Concatenate along batch dimension: [tiles_1, H, W, C] + [tiles_2, H, W, C] -> [total, H, W, C]
    let mut result = images[0].pixel_values.clone();
    for img in &images[1..] {
        result = MxArray::concatenate(&result, &img.pixel_values, 0)?;
    }
    Ok(result)
}

/// Compute the longest common prefix between two token sequences.
fn compute_prefix_match(new_tokens: &[u32], cached_tokens: &[u32]) -> usize {
    new_tokens
        .iter()
        .zip(cached_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qianfan_ocr::config::QianfanOCRConfig;

    #[test]
    fn test_config_defaults_work() {
        let config = QianfanOCRConfig::default();
        assert_eq!(config.model_type, "internvl_chat");
        assert_eq!(config.eos_token_id, 151645);
        assert_eq!(config.num_image_token(), 256);
    }

    #[test]
    fn test_model_construction_uninitialized() {
        let config = QianfanOCRConfig::default();
        let model = QianfanOCRModel::new(config);
        assert!(!model.is_initialized());
    }

    #[test]
    fn test_chat_result_creation() {
        let result = ChatResult {
            text: "Hello".to_string(),
            tool_calls: vec![],
            thinking: None,
            num_tokens: 1,
            prompt_tokens: 0,
            reasoning_tokens: 0,
            finish_reason: "stop".to_string(),
            raw_text: "Hello".to_string(),
            performance: None,
        };
        assert_eq!(result.text, "Hello");
        assert_eq!(result.num_tokens, 1);
        assert_eq!(result.finish_reason, "stop");
        assert!(result.thinking.is_none());
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_prefix_match_full() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        assert_eq!(compute_prefix_match(&a, &b), 5);
    }

    #[test]
    fn test_prefix_match_partial() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 6, 7];
        assert_eq!(compute_prefix_match(&a, &b), 3);
    }

    #[test]
    fn test_prefix_match_none() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert_eq!(compute_prefix_match(&a, &b), 0);
    }

    #[test]
    fn test_prefix_match_empty() {
        let a: Vec<u32> = vec![];
        let b: Vec<u32> = vec![1, 2, 3];
        assert_eq!(compute_prefix_match(&a, &b), 0);
        assert_eq!(compute_prefix_match(&b, &a), 0);
    }

    #[test]
    fn test_parse_config_json_defaults() {
        let raw: Value = serde_json::from_str("{}").unwrap();
        let config = parse_config_json(&raw);
        assert_eq!(config.model_type, "internvl_chat");
        assert_eq!(config.vision_config.hidden_size, 1024);
        assert_eq!(config.llm_config.hidden_size, 2560);
        assert_eq!(config.eos_token_id, 151645);
    }

    #[test]
    fn test_parse_config_json_custom_values() {
        let json = r#"{
            "model_type": "test_model",
            "eos_token_id": 12345,
            "downsample_ratio": 0.25,
            "vision_config": {
                "hidden_size": 512,
                "num_hidden_layers": 12
            },
            "llm_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 16,
                "vocab_size": 50000
            }
        }"#;
        let raw: Value = serde_json::from_str(json).unwrap();
        let config = parse_config_json(&raw);
        assert_eq!(config.model_type, "test_model");
        assert_eq!(config.eos_token_id, 12345);
        assert_eq!(config.downsample_ratio, 0.25);
        assert_eq!(config.vision_config.hidden_size, 512);
        assert_eq!(config.vision_config.num_hidden_layers, 12);
        assert_eq!(config.llm_config.hidden_size, 1024);
        assert_eq!(config.llm_config.num_hidden_layers, 16);
        assert_eq!(config.llm_config.vocab_size, 50000);
    }
}
