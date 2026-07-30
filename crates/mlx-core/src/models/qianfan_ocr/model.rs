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
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle};
use crate::model_thread::{ResponseTx, StreamTx};
use crate::models::qianfan_ocr::bridge::InternVLBridge;
use crate::models::qianfan_ocr::config::{InternVisionConfig, QianfanOCRConfig, Qwen3LMConfig};
use crate::models::qianfan_ocr::language::InternVLLanguageModel;
use crate::models::qianfan_ocr::persistence::load_qianfan_ocr_weights;
use crate::models::qianfan_ocr::processing::{ProcessedImage, QianfanImageProcessor};
use crate::models::qianfan_ocr::vision::InternViTModel;
use crate::models::qwen3_5::model::extract_images_from_messages;
use crate::sampling::{
    SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
    check_repetition_cutoff, sample,
};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, MultimodalContentOrder, Qwen3Tokenizer};
use crate::tools;
use crate::transformer::kv_cache::KVCache;
use crate::utils::safetensors::SafeTensorsFile;

/// Processor marker emitted once per image content part by Qianfan-OCR's
/// model-provided Jinja template. The vision processor replaces it with the
/// checkpoint-configured image span after template rendering.
const QIANFAN_IMAGE_TEMPLATE_PLACEHOLDER: &str = "<image>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QianfanPrefillPlan {
    prefix_len: usize,
    clamped_prefix: usize,
    suffix_requires_image_features: bool,
}

struct PreparedQianfanPrompt {
    token_ids: Vec<u32>,
    prefill_embeds: MxArray,
    current_image_key: Option<u64>,
    num_patches_list: Vec<u32>,
    cached_tokens: usize,
}

fn qianfan_session_state_is_live(
    cached_token_history: &[u32],
    cached_cache_offset: i32,
    kv_cache_offsets: &[i32],
) -> bool {
    let Ok(expected_offset) = i32::try_from(cached_token_history.len()) else {
        return false;
    };
    expected_offset > 0
        && cached_cache_offset == expected_offset
        && !kv_cache_offsets.is_empty()
        && kv_cache_offsets
            .iter()
            .all(|offset| *offset == expected_offset)
}

fn reusable_qianfan_patch_counts(
    image_count: usize,
    current_image_key: Option<u64>,
    cached_image_key: Option<u64>,
    cached_num_patches_list: Option<&[u32]>,
) -> Option<Vec<u32>> {
    if image_count == 0 {
        return Some(Vec::new());
    }
    (current_image_key.is_some() && current_image_key == cached_image_key)
        .then_some(cached_num_patches_list)
        .flatten()
        .filter(|counts| counts.len() == image_count)
        .map(<[u32]>::to_vec)
}

fn plan_qianfan_prefill(
    token_ids: &[u32],
    cached_token_history: &[u32],
    reuse_cache: bool,
    image_key_matches: bool,
    image_context_token_id: u32,
) -> QianfanPrefillPlan {
    let prefix_len = if reuse_cache && image_key_matches {
        compute_prefix_match(token_ids, cached_token_history)
    } else {
        0
    };
    let clamped_prefix = prefix_len.min(token_ids.len().saturating_sub(1));
    let suffix_requires_image_features =
        token_ids[clamped_prefix..].contains(&image_context_token_id);
    QianfanPrefillPlan {
        prefix_len,
        clamped_prefix,
        suffix_requires_image_features,
    }
}

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
    /// Per-image dynamic-tiling counts associated with `cached_image_key`.
    /// These let a same-image continuation reproduce the template's expanded
    /// token sequence and verify the reusable prefix before decoding or
    /// processing the historical image bytes again.
    pub(crate) cached_num_patches_list: Option<Vec<u32>>,
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
    /// Continue an existing session by rendering the caller's full history
    /// with the model-provided chat template. Prefix verification decides
    /// whether the live cache can serve the rendered extension.
    ChatSessionContinue {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Tool-result continuation. Tool structure lives in `messages` and is
    /// rendered exclusively by the model-provided chat template.
    ChatSessionContinueTool {
        messages: Vec<ChatMessage>,
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
    /// Streaming session-continue over a full template-rendered history.
    ChatStreamSessionContinue {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    /// Streaming tool-result continuation over a full history.
    ChatStreamSessionContinueTool {
        messages: Vec<ChatMessage>,
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
            // NOTE: no per-request cache drain here. On a multi-model
            // server the MLX allocator free-pool is process-wide, so
            // flushing after a request on model A discards blocks about
            // to be reused by model B. The TS idle sweeper in
            // `@mlx-node/server` handles between-turn drains.
            let _ = reply.send(inner.chat_session_start_sync(messages, config));
        }
        QianfanOCRCmd::ChatSessionContinue {
            messages,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_continue_sync(messages, config));
        }
        QianfanOCRCmd::ChatSessionContinueTool {
            messages,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_continue_tool_sync(messages, config));
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
            messages,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_continue_sync(messages, config, stream_tx, cancelled);
        }
        QianfanOCRCmd::ChatStreamSessionContinueTool {
            messages,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_continue_tool_sync(messages, config, stream_tx, cancelled);
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
    /// Whether the model was loaded with real weights. `false` for
    /// `new QianfanOCRModel(config)` calls that predate `load()`.
    initialized: bool,
    /// RAII: unregisters this model's baseline from the cache-limit
    /// coordinator on drop. `None` for instances constructed via
    /// `new(config)` that never loaded weights.
    _cache_limit_guard: Option<crate::cache_limit::CacheLimitGuard>,
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
        self.cached_num_patches_list = None;
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

    fn has_live_session(&self) -> bool {
        let Some(caches) = self.kv_caches.as_deref() else {
            return false;
        };
        let offsets: Vec<i32> = caches.iter().map(KVCache::get_offset).collect();
        qianfan_session_state_is_live(
            &self.cached_token_history,
            self.cached_cache_offset,
            &offsets,
        )
    }

    fn process_chat_images(&self, image_bytes: &[Vec<u8>]) -> Result<Vec<ProcessedImage>> {
        let processor = QianfanImageProcessor::new(&self.config);
        let image_refs: Vec<&[u8]> = image_bytes.iter().map(Vec::as_slice).collect();
        processor.process_many(&image_refs)
    }

    fn encode_chat_images(
        &self,
        processed_images: &[ProcessedImage],
        generation_stream: Stream,
    ) -> Result<MxArray> {
        let all_pixels = stack_processed_images(processed_images)?;
        let vit_out = {
            let _ctx = StreamContext::new(generation_stream);
            self.vision.forward(&all_pixels)?
        };
        let bridge_out = {
            let _ctx = StreamContext::new(generation_stream);
            self.bridge.forward(&vit_out)?
        };
        let bridge_shape = bridge_out.shape()?;
        bridge_out.reshape(&[bridge_shape[0] * bridge_shape[1], bridge_shape[2]])
    }

    /// Prepare the template-rendered prompt and only the embeddings that the
    /// LM still needs to prefill. Same-image continuations first reproduce
    /// image-token expansion from cached patch counts, then prefix-match. If
    /// every image-context token is already covered by the reusable prefix,
    /// historical images never reach the decoder, processor, ViT, or bridge.
    fn prepare_chat_prefill(
        &mut self,
        messages: &[ChatMessage],
        config: &ChatConfig,
        reuse_cache: bool,
        generation_stream: Stream,
    ) -> Result<PreparedQianfanPrompt> {
        let image_bytes = extract_images_from_messages(messages);
        let current_image_key =
            (!image_bytes.is_empty()).then(|| crate::engine::compute_image_cache_key(&image_bytes));
        let image_key_matches = current_image_key == self.cached_image_key;

        let mut processed_images: Option<Vec<ProcessedImage>> = None;
        let mut num_patches_list = if let Some(counts) = reusable_qianfan_patch_counts(
            image_bytes.len(),
            current_image_key,
            self.cached_image_key,
            self.cached_num_patches_list.as_deref(),
        ) {
            counts
        } else {
            let processed = self.process_chat_images(&image_bytes)?;
            let counts = processed.iter().map(|image| image.num_tiles).collect();
            processed_images = Some(processed);
            counts
        };

        let mut token_ids = self.render_prompt_tokens(messages, config, &num_patches_list)?;
        let mut plan = plan_qianfan_prefill(
            &token_ids,
            &self.cached_token_history,
            reuse_cache,
            image_key_matches,
            self.config.img_context_token_id as u32,
        );

        // A cached patch plan is sufficient for prefix verification, but a
        // suffix containing image-context positions still needs real visual
        // features. Reprocess only in that case. Revalidate the deterministic
        // counts defensively before using the cached expansion.
        if plan.suffix_requires_image_features && processed_images.is_none() {
            let processed = self.process_chat_images(&image_bytes)?;
            let actual_counts: Vec<u32> = processed.iter().map(|image| image.num_tiles).collect();
            if actual_counts != num_patches_list {
                num_patches_list = actual_counts;
                token_ids = self.render_prompt_tokens(messages, config, &num_patches_list)?;
                // The cached visual KV was produced with a different image
                // expansion. Even when the bytes hash identically, its
                // lineage is no longer provable, so force a complete visual
                // prefill instead of retaining a textual/image prefix.
                plan = plan_qianfan_prefill(
                    &token_ids,
                    &self.cached_token_history,
                    reuse_cache,
                    false,
                    self.config.img_context_token_id as u32,
                );
            }
            processed_images = Some(processed);
        }

        if plan.prefix_len == 0 || !reuse_cache {
            self.kv_caches = None;
            self.init_kv_caches();
        } else {
            let cache_offset = self.get_cache_offset();
            if cache_offset > plan.clamped_prefix as i32
                && let Some(caches) = self.kv_caches.as_mut()
            {
                for cache in caches {
                    cache.trim(plan.clamped_prefix as i32);
                }
            }
        }

        let prefill_embeds = if plan.suffix_requires_image_features {
            let processed = processed_images.as_deref().ok_or_else(|| {
                Error::from_reason(
                    "Qianfan-OCR image-bearing prompt suffix has no processed image features",
                )
            })?;
            let input_ids = MxArray::from_uint32(&token_ids, &[1, token_ids.len() as i64])?;
            let text_embeds = {
                let _ctx = StreamContext::new(generation_stream);
                self.language_model.get_embeddings(&input_ids)?
            };
            let vision_features = self.encode_chat_images(processed, generation_stream)?;
            let embed_dtype = text_embeds.dtype()?;
            let vision_features = if vision_features.dtype()? != embed_dtype {
                vision_features.astype(embed_dtype)?
            } else {
                vision_features
            };
            let merged_embeds = merge_vision_features(
                &input_ids,
                &text_embeds,
                &vision_features,
                self.config.img_context_token_id,
            )?;
            merged_embeds.slice_axis(1, plan.clamped_prefix as i64, token_ids.len() as i64)?
        } else {
            let suffix = &token_ids[plan.clamped_prefix..];
            let input_ids = MxArray::from_uint32(suffix, &[1, suffix.len() as i64])?;
            let _ctx = StreamContext::new(generation_stream);
            self.language_model.get_embeddings(&input_ids)?
        };

        Ok(PreparedQianfanPrompt {
            token_ids,
            prefill_embeds,
            current_image_key,
            num_patches_list,
            cached_tokens: plan.clamped_prefix,
        })
    }

    /// Render the complete conversation exclusively through the tokenizer's
    /// model-provided chat template, then expand the abstract image markers
    /// emitted by that template into model-configured visual-token spans.
    fn render_prompt_tokens(
        &self,
        messages: &[ChatMessage],
        config: &ChatConfig,
        num_patches_list: &[u32],
    ) -> Result<Vec<u32>> {
        if !self.tokenizer.has_chat_template() {
            return Err(Error::from_reason(
                "Qianfan-OCR requires a model-provided chat template in \
                 tokenizer_config.json or chat_template.jinja; no template was found",
            ));
        }

        let template_tokens = self.tokenizer.apply_chat_template_sync_with_content_order(
            messages,
            Some(true),
            config.tools.as_deref(),
            crate::engine::resolve_enable_thinking(config),
            MultimodalContentOrder::ImagesThenText,
            Some(QIANFAN_IMAGE_TEMPLATE_PLACEHOLDER),
        )?;
        let placeholder_tokens = self
            .tokenizer
            .encode_sync(QIANFAN_IMAGE_TEMPLATE_PLACEHOLDER, Some(false))?;

        expand_qianfan_image_placeholders(
            &template_tokens,
            &placeholder_tokens,
            num_patches_list,
            self.config.num_image_token() as usize,
            self.config.img_start_token_id as u32,
            self.config.img_context_token_id as u32,
            self.config.img_end_token_id as u32,
        )
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
    /// can verify before prefilling the newly rendered suffix.
    ///
    /// Only called by the session start/continue surface; there is no longer a
    /// non-session chat entry point.
    fn chat_turn_sync_core(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        eos_token_id: u32,
    ) -> Result<ChatResult> {
        let result = self.chat_turn_sync_core_inner(messages, config, eos_token_id);
        if result.is_err() {
            self.reset_caches_sync();
        }
        result
    }

    fn chat_turn_sync_core_inner(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        eos_token_id: u32,
    ) -> Result<ChatResult> {
        // Clamp a nonpositive budget to 0 so the `Vec::with_capacity(.. as
        // usize)` below never sees a negative `i32` (`-1 as usize` would
        // request `usize::MAX`); the `0..max_new_tokens` loop then emits 0.
        let max_new_tokens = config.max_new_tokens.unwrap_or(512).max(0);
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
        let max_consecutive_tokens = config
            .max_consecutive_tokens
            .unwrap_or(crate::sampling::DEFAULT_MAX_CONSECUTIVE_TOKENS);
        let max_ngram_repeats = config
            .max_ngram_repeats
            .unwrap_or(crate::sampling::DEFAULT_MAX_NGRAM_REPEATS);
        let ngram_size = config
            .ngram_size
            .unwrap_or(crate::sampling::DEFAULT_NGRAM_SIZE);
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let generation_stream = Stream::new(DeviceType::Gpu);
        let prepared =
            self.prepare_chat_prefill(&messages, &config, reuse_cache, generation_stream)?;
        let PreparedQianfanPrompt {
            token_ids,
            prefill_embeds,
            current_image_key,
            num_patches_list,
            cached_tokens,
        } = prepared;

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

        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(crate::engine::generated_capacity_hint(max_new_tokens));
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
            // Image identity and processor metadata are published together;
            // a matching hash never observes patch counts from another turn.
            self.cached_image_key = current_image_key;
            self.cached_num_patches_list = Some(num_patches_list);
        } else {
            // Not reusing — clear metadata to prevent stale prefix matches
            self.cached_token_history.clear();
            self.cached_cache_offset = 0;
            self.cached_image_key = None;
            self.cached_num_patches_list = None;
        }

        // --- Step 10: Decode and parse ---
        let raw_decoded = self.tokenizer.decode_sync(&generated_tokens, true)?;
        let include_reasoning = crate::engine::resolve_include_reasoning(&config);
        let thinking_enabled = crate::engine::resolve_enable_thinking(&config).unwrap_or(true);
        let (text, tool_calls, thinking) = crate::engine::parse_thinking_and_tools(
            &raw_decoded,
            &generated_tokens,
            thinking_enabled,
            self.tokenizer.think_end_id(),
            self.tokenizer.think_end_str(),
            true,
        );
        let public_raw_text = crate::engine::raw_text_with_reasoning_suppressed(
            &raw_decoded,
            &generated_tokens,
            thinking_enabled,
            self.tokenizer.think_end_id(),
            self.tokenizer.think_end_str(),
            false,
        );

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
                    // Qianfan-OCR has no MTP heads — acceptance stays None.
                    mtp_mean_accepted_tokens: None,
                    mtp_mean_accepted_tokens_total: None,
                    mtp_acceptance_by_position: None,
                    mtp_cycles: None,
                    mtp_mean_depth: None,
                    profile_phases: None,
                })
            } else {
                None
            };

        let reasoning_tokens = tools::count_reasoning_tokens(
            &thinking,
            &generated_tokens,
            self.tokenizer.think_end_id(),
        );
        let thinking = if include_reasoning { thinking } else { None };
        let raw_text = if include_reasoning {
            raw_decoded
        } else {
            public_raw_text.clone()
        };

        Ok(ChatResult {
            text: text.trim().to_string(),
            tool_calls,
            thinking,
            thinking_enabled,
            num_tokens: generated_tokens.len() as u32,
            prompt_tokens: prefill_token_count as u32,
            reasoning_tokens,
            finish_reason,
            raw_text,
            public_raw_text: Some(public_raw_text),
            cached_tokens: cached_tokens as u32,
            performance,
        })
    }

    /// Streaming chat generation.
    ///
    /// Mirrors [`chat_turn_sync_core`] but emits per-token deltas through
    /// `stream_tx` and checks `cancelled` on every decode iteration.
    /// Drives the same hoisted cache state so prefix matching and
    /// repetition penalties behave identically to the non-streaming path.
    ///
    /// `eos_token_id` threads through exactly as in
    /// [`chat_turn_sync_core`](Self::chat_turn_sync_core): session-start callers
    /// supply `<|im_end|>` via
    /// [`chat_stream_session_start_sync`](Self::chat_stream_session_start_sync).
    fn chat_turn_stream_core(
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
            // Clamp a nonpositive budget to 0 so the `Vec::with_capacity(..
            // as usize)` below never sees a negative `i32` (`-1 as usize`
            // would request `usize::MAX`); the `0..max_new_tokens` loop then
            // emits 0.
            let max_new_tokens = config.max_new_tokens.unwrap_or(512).max(0);
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
            let max_consecutive_tokens = config
                .max_consecutive_tokens
                .unwrap_or(crate::sampling::DEFAULT_MAX_CONSECUTIVE_TOKENS);
            let max_ngram_repeats = config
                .max_ngram_repeats
                .unwrap_or(crate::sampling::DEFAULT_MAX_NGRAM_REPEATS);
            let ngram_size = config
                .ngram_size
                .unwrap_or(crate::sampling::DEFAULT_NGRAM_SIZE);
            let reuse_cache = config.reuse_cache.unwrap_or(true);
            let report_perf = config.report_performance.unwrap_or(false);
            let include_reasoning = crate::engine::resolve_include_reasoning(&config);
            let thinking_enabled = crate::engine::resolve_enable_thinking(&config).unwrap_or(true);

            let generation_start = if report_perf {
                Some(std::time::Instant::now())
            } else {
                None
            };

            let generation_stream = Stream::new(DeviceType::Gpu);
            let prepared =
                self.prepare_chat_prefill(&messages, &config, reuse_cache, generation_stream)?;
            let PreparedQianfanPrompt {
                token_ids,
                prefill_embeds,
                current_image_key,
                num_patches_list,
                cached_tokens,
            } = prepared;

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

            let mut generated_tokens: Vec<u32> =
                Vec::with_capacity(crate::engine::generated_capacity_hint(max_new_tokens));
            let mut finish_reason = "length".to_string();
            let mut reasoning_tracker = crate::engine::ReasoningTracker::new(
                thinking_enabled,
                config.thinking_token_budget,
                self.tokenizer.think_end_id(),
            );

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
                let is_reasoning = reasoning_tracker.observe_token(token_value);

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

                if include_reasoning || !is_reasoning {
                    emit(ChatStreamChunk {
                        text: token_text,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(is_reasoning),
                    });
                }

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
                self.cached_image_key = current_image_key;
                self.cached_num_patches_list = Some(num_patches_list);
            } else {
                self.cached_token_history.clear();
                self.cached_cache_offset = 0;
                self.cached_image_key = None;
                self.cached_num_patches_list = None;
            }

            // Final chunk
            let raw_decoded = self.tokenizer.decode_sync(&generated_tokens, true)?;
            let (text, tool_calls, thinking) = crate::engine::parse_thinking_and_tools(
                &raw_decoded,
                &generated_tokens,
                thinking_enabled,
                self.tokenizer.think_end_id(),
                self.tokenizer.think_end_str(),
                true,
            );
            let public_raw_text = crate::engine::raw_text_with_reasoning_suppressed(
                &raw_decoded,
                &generated_tokens,
                thinking_enabled,
                self.tokenizer.think_end_id(),
                self.tokenizer.think_end_str(),
                false,
            );

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
                    // Qianfan-OCR has no MTP heads — acceptance stays None.
                    mtp_mean_accepted_tokens: None,
                    mtp_mean_accepted_tokens_total: None,
                    mtp_acceptance_by_position: None,
                    mtp_cycles: None,
                    mtp_mean_depth: None,
                    profile_phases: None,
                })
            } else {
                None
            };

            let reasoning_tokens = tools::count_reasoning_tokens(
                &thinking,
                &generated_tokens,
                self.tokenizer.think_end_id(),
            );
            let thinking = if include_reasoning { thinking } else { None };
            let raw_text = if include_reasoning {
                raw_decoded
            } else {
                public_raw_text.clone()
            };

            emit(ChatStreamChunk {
                text: text.trim().to_string(),
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                thinking_enabled: Some(thinking_enabled),
                num_tokens: Some(generated_tokens.len() as u32),
                prompt_tokens: Some(prefill_token_count as u32),
                reasoning_tokens: Some(reasoning_tokens),
                raw_text: Some(raw_text),
                public_raw_text: Some(public_raw_text),
                text_authoritative: Some(true),
                // Start path: report the matched prefix length. Zero on a miss or disabled
                // reuse, equal to the matched prefix length on a hit.
                cached_tokens: Some(cached_tokens as u32),
                performance,
                is_reasoning: None,
            });

            Ok(())
        })();

        if let Err(e) = result {
            self.reset_caches_sync();
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
    fn session_eos_id(&self) -> u32 {
        self.tokenizer.get_eos_token_id()
    }

    /// Start a new chat session.
    ///
    /// Fully resets the caches and delegates to [`Self::chat_turn_sync_core`]
    /// with `<|im_end|>` as the stop token so the decode loop leaves the
    /// caches on a clean ChatML boundary that subsequent
    /// [`Self::chat_session_continue_sync`] /
    /// [`Self::chat_session_continue_tool_sync`] calls can extend by
    /// re-rendering the complete structured history through the model
    /// template.
    ///
    /// Vision-capable: `messages` may carry images (they will be decoded
    /// through the InternViT → bridge pipeline inside `chat_turn_sync_core`,
    /// same as the legacy chat path).
    pub(crate) fn chat_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // The session API only makes sense with cache reuse enabled: if we
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
        let im_end_id = self.session_eos_id();

        // Full reset: the session-start path always begins from a clean
        // state. This matches the documented contract that the session is
        // owned end-to-end by the `chat_session_*` surface and
        // intentionally invalidates any prior cache.
        self.reset_caches_sync();

        self.chat_turn_sync_core(messages, config, im_end_id)
    }

    /// Continue a session from the caller's complete structured history.
    ///
    /// The full history is rendered by the checkpoint template. The core
    /// verifies that the rendered prompt strictly extends the committed token
    /// history before reusing the live cache, and otherwise performs a safe
    /// full prefill.
    pub(crate) fn chat_session_continue_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        if config.reuse_cache == Some(false) {
            return Err(Error::from_reason(
                "chat_session_continue requires reuse_cache=true (leave as None or set to true)",
            ));
        }
        if !self.has_live_session() {
            return Err(Error::from_reason(
                "chat_session_continue requires an initialized session (call chatSessionStart first)",
            ));
        }
        let eos_id = self.session_eos_id();
        self.chat_turn_sync_core(messages, config, eos_id)
    }

    /// Tool-result continuation uses the same complete-history template path;
    /// the model template alone decides how tool messages are represented.
    pub(crate) fn chat_session_continue_tool_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        self.chat_session_continue_sync(messages, config)
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
            crate::engine::send_stream_error(
                &stream_tx,
                "chat_stream_session_start cancelled before start",
            );
            return;
        }

        if config.reuse_cache == Some(false) {
            crate::engine::send_stream_error(
                &stream_tx,
                "chat_stream_session_start requires reuse_cache=true (leave as None or set to true). The session API only makes sense with cache reuse enabled.",
            );
            return;
        }

        let im_end_id = self.session_eos_id();

        // Full reset: the session always starts clean.
        self.reset_caches_sync();

        self.chat_turn_stream_core(messages, config, stream_tx, cancelled, im_end_id);
    }

    /// Streaming variant of [`Self::chat_session_continue_sync`].
    pub(crate) fn chat_stream_session_continue_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            crate::engine::send_stream_error(
                &stream_tx,
                "chat_stream_session_continue cancelled before start",
            );
            return;
        }

        if config.reuse_cache == Some(false) {
            crate::engine::send_stream_error(
                &stream_tx,
                "chat_stream_session_continue requires reuse_cache=true (leave as None or set to true)",
            );
            return;
        }
        if !self.has_live_session() {
            crate::engine::send_stream_error(
                &stream_tx,
                "chat_stream_session_continue requires an initialized session (call chatStreamSessionStart first)",
            );
            return;
        }

        let eos_id = self.session_eos_id();
        self.chat_turn_stream_core(messages, config, stream_tx, cancelled, eos_id);
    }

    /// Streaming tool-result continuation over the full structured history.
    pub(crate) fn chat_stream_session_continue_tool_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        self.chat_stream_session_continue_sync(messages, config, stream_tx, cancelled);
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
        self.cached_num_patches_list = None;
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
        let mut generated: Vec<u32> =
            Vec::with_capacity(crate::engine::generated_capacity_hint(max_new_tokens));

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
    /// instance is only useful for `is_initialized` queries until
    /// [`QianfanOCRModel::load`] is called to actually run inference. The
    /// `config` argument is accepted to preserve the `new
    /// QianfanOCRModel(config)` JS surface; the value is discarded because
    /// nothing on the uninitialized path consults it (any future config
    /// getter would forward to the inner thread state populated by
    /// `load()`).
    #[napi(constructor)]
    pub fn new(config: QianfanOCRConfig) -> Self {
        let _ = config;
        Self {
            thread: None,
            initialized: false,
            _cache_limit_guard: None,
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
                        // `load_qianfan_ocr_inner_from_dir` returns a
                        // deterministic weight-byte total alongside
                        // the inner; register it with the cache-limit
                        // coordinator here. No active-memory sampling
                        // — the deterministic path is race-free
                        // against concurrent inference. See
                        // `cache_limit.rs` module docs.
                        let (inner, weight_bytes) = load_qianfan_ocr_inner_from_dir(&model_path)?;
                        let cache_limit_guard =
                            crate::cache_limit::coordinator().register(weight_bytes);
                        Ok((inner, cache_limit_guard))
                    },
                    handle_qianfan_ocr_cmd,
                );

                let cache_limit_guard = init_rx
                    .await
                    .map_err(|_| napi::Error::from_reason("Model thread exited during load"))??;

                Ok((thread, cache_limit_guard))
            },
            |_env, (thread, cache_limit_guard)| {
                Ok(QianfanOCRModel {
                    thread: Some(thread),
                    initialized: true,
                    _cache_limit_guard: Some(cache_limit_guard),
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

        // Clamp a nonpositive budget to 0 before it reaches `generate_sync`
        // (which sizes `Vec::with_capacity(.. as usize)`); the
        // `0..max_new_tokens` loop then emits 0.
        let max_new_tokens = max_new_tokens.unwrap_or(256).max(0);
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
    /// Renders the complete structured conversation through the checkpoint
    /// template, decodes until `<|im_end|>`, and preserves KV state for an
    /// exact-prefix check against the next complete template render.
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
        reject_unsupported_audio_in_messages(&messages)?;

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

    /// Continue from the caller's complete conversation history. The
    /// checkpoint's chat template is rendered again and exact prefix matching
    /// decides whether the live cache can be reused.
    #[napi]
    pub async fn chat_session_continue(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        reject_unsupported_audio_in_messages(&messages)?;

        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(thread, |reply| QianfanOCRCmd::ChatSessionContinue {
            messages,
            config,
            reply,
        })
        .await
    }

    /// Tool-result continuation over a full history. Tool representation is
    /// owned entirely by the model-provided template.
    #[napi]
    pub async fn chat_session_continue_tool(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        reject_unsupported_audio_in_messages(&messages)?;
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(thread, |reply| {
            QianfanOCRCmd::ChatSessionContinueTool {
                messages,
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
        reject_unsupported_audio_in_messages(&messages)?;

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

    /// Streaming continuation over a complete conversation history.
    #[napi(
        ts_args_type = "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_continue(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        reject_unsupported_audio_in_messages(&messages)?;

        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        thread.send(QianfanOCRCmd::ChatStreamSessionContinue {
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

    /// Streaming tool-result continuation over a complete history.
    #[napi(
        ts_args_type = "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_continue_tool(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        reject_unsupported_audio_in_messages(&messages)?;
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call QianfanOCRModel.load() first.")
        })?;

        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        thread.send(QianfanOCRCmd::ChatStreamSessionContinueTool {
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
fn load_qianfan_ocr_inner_from_dir(model_path: &str) -> Result<(QianfanOCRInner, u64)> {
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

    // Validate the tokenizer and the checkpoint-owned template before loading
    // weights or constructing the vision/language stacks. Qianfan has no Rust
    // wire-format fallback: an incompatible checkpoint should fail quickly,
    // before doing the expensive model build.
    let tokenizer_path = path.join("tokenizer.json");
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
    if !tokenizer.has_chat_template() {
        return Err(Error::from_reason(format!(
            "Qianfan-OCR model at {} is missing a model-provided chat template; \
             add chat_template.jinja or tokenizer_config.json.chat_template",
            path.display()
        )));
    }

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

    info!(
        "Qianfan-OCR model loaded: vision={} layers, LM={} layers, {} total weights",
        config.vision_config.num_hidden_layers,
        config.llm_config.num_hidden_layers,
        weights.len()
    );

    // Deterministic weight-byte total for the cache-limit coordinator.
    // Computed while `weights` is still in scope — registration is
    // performed in `QianfanOCRModel::load` using this value.
    let weight_bytes: u64 = weights
        .values()
        .map(|a| a.nbytes() as u64)
        .fold(0u64, |acc, v| acc.saturating_add(v));

    let inner = QianfanOCRInner {
        config,
        vision,
        bridge,
        language_model,
        tokenizer,
        kv_caches: None,
        cached_token_history: Vec::new(),
        cached_image_key: None,
        cached_num_patches_list: None,
        cached_cache_offset: 0,
    };
    Ok((inner, weight_bytes))
}

/// Expand the abstract image markers emitted by the model's chat template.
///
/// Conversation structure and marker placement come from Jinja. This
/// processor step only substitutes each marker with the visual-token span
/// required by the checkpoint's image encoder.
fn expand_qianfan_image_placeholders(
    template_tokens: &[u32],
    placeholder_tokens: &[u32],
    num_patches_list: &[u32],
    image_tokens_per_patch: usize,
    image_start_id: u32,
    image_context_id: u32,
    image_end_id: u32,
) -> Result<Vec<u32>> {
    if placeholder_tokens.is_empty() {
        return Err(Error::from_reason(
            "Qianfan-OCR tokenizer encoded the template image marker to an empty sequence",
        ));
    }

    let mut expanded = Vec::with_capacity(template_tokens.len());
    let mut token_index = 0usize;
    let mut image_index = 0usize;

    while token_index < template_tokens.len() {
        if template_tokens[token_index..].starts_with(placeholder_tokens) {
            let patches = num_patches_list.get(image_index).ok_or_else(|| {
                Error::from_reason(format!(
                    "Qianfan-OCR chat template emitted more image markers than the {} supplied image(s)",
                    num_patches_list.len()
                ))
            })?;
            let context_tokens = (*patches as usize)
                .checked_mul(image_tokens_per_patch)
                .ok_or_else(|| Error::from_reason("Qianfan-OCR image token count overflow"))?;

            expanded.push(image_start_id);
            expanded.extend(std::iter::repeat_n(image_context_id, context_tokens));
            expanded.push(image_end_id);
            image_index += 1;
            token_index += placeholder_tokens.len();
        } else {
            expanded.push(template_tokens[token_index]);
            token_index += 1;
        }
    }

    if image_index != num_patches_list.len() {
        return Err(Error::from_reason(format!(
            "Qianfan-OCR chat template emitted {image_index} image marker(s) for {} supplied image(s)",
            num_patches_list.len()
        )));
    }

    Ok(expanded)
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

/// Boundary guard for the `audio` parameter on Qianfan-OCR's chat-continue
/// entry points.
///
/// The shared chat surface (`chat_napi_surface!`) carries an `audio`
/// positional argument between `images` and `config`. Qianfan-OCR keeps the
/// same positional ABI so it can be driven through the same `ChatSession` /
/// `makeStreamingModel` plumbing, but it has no audio support. A non-empty
/// `audio` is therefore rejected here with the shared no-audio message
/// (prefixed with [`crate::engine::IMAGE_CHANGE_RESTART_PREFIX`], matching the
/// fresh-turn audio rejection in `engine::session`). `None` / empty audio is a
/// complete no-op: it returns `Ok(())` and is never threaded into the model
/// thread, so existing text + image behaviour stays byte-identical.
fn reject_unsupported_audio(audio: Option<&[Uint8Array]>) -> Result<()> {
    if audio.is_some_and(|clips| !clips.is_empty()) {
        return Err(Error::from_reason(format!(
            "{} this model has no audio support; audio messages are not supported",
            crate::engine::IMAGE_CHANGE_RESTART_PREFIX
        )));
    }
    Ok(())
}

/// Boundary guard for the `audio` field carried by `ChatMessage` on
/// Qianfan-OCR's chat-start entry points.
///
/// The shared `ChatMessage` struct carries an `audio` field so the unified
/// `ChatSession` surface can ship audio clips to families that support them.
/// Qianfan-OCR bypasses the shared engine core (which rejects first-turn audio
/// for families that don't support it), so it must reject audio itself before
/// dispatching the message batch to the model thread. Any message with a
/// non-empty `audio` is rejected with the same shared no-audio message the
/// continue guard uses (prefixed with
/// [`crate::engine::IMAGE_CHANGE_RESTART_PREFIX`]); messages with no audio /
/// empty audio vecs are a complete no-op so text + image starts stay
/// byte-identical.
fn reject_unsupported_audio_in_messages(messages: &[ChatMessage]) -> Result<()> {
    for msg in messages {
        reject_unsupported_audio(msg.audio.as_deref())?;
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qianfan_ocr::config::QianfanOCRConfig;

    #[test]
    fn image_expansion_preserves_template_placement() {
        let expanded =
            expand_qianfan_image_placeholders(&[7, 90, 91, 8], &[90, 91], &[2], 3, 10, 11, 12)
                .unwrap();
        assert_eq!(expanded, vec![7, 10, 11, 11, 11, 11, 11, 11, 12, 8]);
    }

    #[test]
    fn image_expansion_rejects_template_image_count_mismatch() {
        let err =
            expand_qianfan_image_placeholders(&[7, 90, 91, 8], &[90, 91], &[1, 1], 2, 10, 11, 12)
                .expect_err("one marker cannot represent two images");
        assert!(err.reason.contains("emitted 1 image marker"));
    }

    #[test]
    fn image_expansion_preserves_multiple_manual_marker_order() {
        let expanded = expand_qianfan_image_placeholders(
            &[90, 91, 7, 90, 91],
            &[90, 91],
            &[1, 2],
            1,
            10,
            11,
            12,
        )
        .unwrap();
        assert_eq!(expanded, vec![10, 11, 12, 7, 10, 11, 11, 12]);
    }

    #[test]
    fn image_expansion_rejects_more_manual_markers_than_images() {
        let err =
            expand_qianfan_image_placeholders(&[90, 91, 7, 90, 91], &[90, 91], &[1], 2, 10, 11, 12)
                .expect_err("two markers cannot represent one image");
        assert!(
            err.reason
                .contains("more image markers than the 1 supplied image")
        );
    }

    #[test]
    fn session_state_requires_history_and_live_kv() {
        assert!(!qianfan_session_state_is_live(&[], 0, &[]));
        assert!(!qianfan_session_state_is_live(&[1], 0, &[0]));
        assert!(!qianfan_session_state_is_live(&[1], 1, &[]));
        assert!(!qianfan_session_state_is_live(&[1], 1, &[1, 0]));
        assert!(qianfan_session_state_is_live(&[1], 1, &[1, 1]));
    }

    #[test]
    fn patch_counts_reuse_requires_matching_image_identity_and_count() {
        let cached = [2, 4];
        assert_eq!(
            reusable_qianfan_patch_counts(2, Some(7), Some(7), Some(&cached)),
            Some(vec![2, 4])
        );
        assert_eq!(
            reusable_qianfan_patch_counts(2, Some(8), Some(7), Some(&cached)),
            None
        );
        assert_eq!(
            reusable_qianfan_patch_counts(1, Some(7), Some(7), Some(&cached)),
            None
        );
        assert_eq!(
            reusable_qianfan_patch_counts(0, None, Some(7), Some(&cached)),
            Some(Vec::new())
        );
    }

    #[test]
    fn warm_prefix_covering_images_skips_image_work() {
        let tokens = [10, 99, 99, 11, 20, 21];
        let cached = [10, 99, 99, 11, 20];
        let plan = plan_qianfan_prefill(&tokens, &cached, true, true, 99);
        assert_eq!(plan.prefix_len, cached.len());
        assert_eq!(plan.clamped_prefix, cached.len());
        assert!(!plan.suffix_requires_image_features);
    }

    #[test]
    fn image_work_is_required_on_prefix_or_image_identity_miss() {
        let tokens = [10, 99, 99, 11, 20];
        let prefix_before_image = plan_qianfan_prefill(&tokens, &[10], true, true, 99);
        assert_eq!(prefix_before_image.prefix_len, 1);
        assert!(prefix_before_image.suffix_requires_image_features);

        let changed_image = plan_qianfan_prefill(&tokens, &tokens, true, false, 99);
        assert_eq!(changed_image.prefix_len, 0);
        assert!(changed_image.suffix_requires_image_features);
    }

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
            thinking_enabled: true,
            num_tokens: 1,
            prompt_tokens: 0,
            reasoning_tokens: 0,
            finish_reason: "stop".to_string(),
            raw_text: "Hello".to_string(),
            public_raw_text: None,
            cached_tokens: 0,
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
    fn test_reject_unsupported_audio_none_is_noop() {
        // `None` audio (the only value the TS delta path ever ships) and an
        // explicitly empty audio vec are both complete no-ops so Qianfan's
        // text + image behaviour stays byte-identical.
        assert!(reject_unsupported_audio(None).is_ok());
        let empty: Vec<Uint8Array> = Vec::new();
        assert!(reject_unsupported_audio(Some(&empty)).is_ok());
    }

    #[test]
    fn test_reject_unsupported_audio_nonempty_rejects_with_prefix() {
        // A non-empty audio argument is rejected at the boundary with the
        // shared no-audio message. The error is prefixed so the TS
        // `ChatSession` layer treats it uniformly with the other families.
        let clips = vec![Uint8Array::new(vec![0u8, 1u8, 2u8])];
        let err = reject_unsupported_audio(Some(&clips)).expect_err("non-empty audio must reject");
        let msg = &err.reason;
        assert!(
            msg.starts_with(crate::engine::IMAGE_CHANGE_RESTART_PREFIX),
            "audio rejection must carry the restart prefix, got: {msg}"
        );
        assert!(
            msg.contains("no audio support"),
            "audio rejection must mention no audio support, got: {msg}"
        );
    }

    /// Build a minimal user `ChatMessage` with the given audio clips for the
    /// start-path guard tests. `audio: None` models the text/image path the
    /// TS layer normally ships.
    fn user_message_with_audio(audio: Option<Vec<Uint8Array>>) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: "describe this".to_string(),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            reasoning_content: None,
            thinking_enabled: None,
            images: None,
            audio,
        }
    }

    #[test]
    fn test_reject_unsupported_audio_in_messages_none_is_noop() {
        // Messages with no audio and messages whose audio vec is empty are
        // both complete no-ops so Qianfan's text + image start path stays
        // byte-identical.
        let no_audio = vec![user_message_with_audio(None)];
        assert!(reject_unsupported_audio_in_messages(&no_audio).is_ok());

        let empty_audio = vec![user_message_with_audio(Some(Vec::new()))];
        assert!(reject_unsupported_audio_in_messages(&empty_audio).is_ok());

        assert!(reject_unsupported_audio_in_messages(&[]).is_ok());
    }

    #[test]
    fn test_reject_unsupported_audio_in_messages_nonempty_rejects_with_prefix() {
        // A message carrying a non-empty audio clip is rejected at the start
        // boundary with the shared no-audio message, prefixed so the TS
        // `ChatSession` layer treats it uniformly with the other families.
        let clips = vec![Uint8Array::new(vec![0u8, 1u8, 2u8])];
        let messages = vec![
            user_message_with_audio(None),
            user_message_with_audio(Some(clips)),
        ];
        let err = reject_unsupported_audio_in_messages(&messages)
            .expect_err("a message with non-empty audio must reject");
        let msg = &err.reason;
        assert!(
            msg.starts_with(crate::engine::IMAGE_CHANGE_RESTART_PREFIX),
            "audio rejection must carry the restart prefix, got: {msg}"
        );
        assert!(
            msg.contains("no audio support"),
            "audio rejection must mention no audio support, got: {msg}"
        );
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
