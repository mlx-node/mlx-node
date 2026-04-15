use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tracing::{info, warn};

use crate::array::MxArray;
use crate::model_thread::{ResponseTx, StreamTx};
use crate::models::qwen3_5::chat_common::{
    IMAGE_CHANGE_RESTART_PREFIX, ReasoningTracker, apply_all_penalties,
    build_chatml_continue_delta_text, build_synthetic_user_message, compute_performance_metrics,
    extract_chat_params, finalize_chat_result, parse_thinking_and_tools, resolve_enable_thinking,
    resolve_include_reasoning, send_stream_error,
};
use crate::models::qwen3_5::model::{ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::sample;
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};

use super::config::Lfm2Config;
use super::decoder_layer::Lfm2DecoderLayer;
use super::layer_cache::Lfm2LayerCache;

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership.
pub(crate) struct Lfm2Inner {
    pub(crate) config: Lfm2Config,
    pub(crate) embed_tokens: Embedding,
    pub(crate) layers: Vec<Lfm2DecoderLayer>,
    /// Output norm (called "embedding_norm" in HF, applied AFTER all layers).
    pub(crate) embedding_norm: RMSNorm,
    /// Separate output projection when `tie_embedding: false`. None when tied.
    pub(crate) lm_head: Option<Linear>,
    pub(crate) caches: Vec<Lfm2LayerCache>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    /// Cached token history for KV cache reuse across chat-session turns.
    pub(crate) cached_token_history: Vec<u32>,
    /// Cached image key for structural uniformity with VLM-capable models.
    /// Always `None` for text-only LFM2; present so session-API code can treat
    /// all model backends uniformly.
    pub(crate) cached_image_key: Option<u64>,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Lfm2Cmd {
    /// Session-based chat continuation: prefill a pre-tokenized delta on top
    /// of the existing LFM2 caches (conv + KV), then decode. Text-only and
    /// requires an active session (prior `ChatSessionStart` call that
    /// populated `cached_token_history`).
    ///
    /// This bypasses the jinja chat template entirely — the caller is
    /// responsible for producing the correctly-formatted delta tokens
    /// (typically `\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`).
    ///
    /// Not currently wired through a NAPI method directly — external callers
    /// use `ChatSessionContinue` instead, which handles delta construction
    /// on the model thread. Kept as its own variant so the lower-level
    /// pre-tokenized entry point stays exposed for the gated integration
    /// test and future advanced use cases.
    #[allow(dead_code)]
    ChatTokensDelta {
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Start a new session via the jinja-render path with `<|im_end|>` as
    /// the stop token. See [`Lfm2Inner::chat_session_start_sync`] for the
    /// behavioural contract (full cache reset, session boundary on
    /// `<|im_end|>`).
    ChatSessionStart {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session by appending a user turn. See
    /// [`Lfm2Inner::chat_session_continue_sync`] — builds a raw ChatML delta
    /// from `user_message`, tokenizes it, and prefills on top of the live
    /// caches.
    ///
    /// LFM2 is text-only; `images` is an opt-in guard parameter that is
    /// rejected with an `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed
    /// error so the TS `ChatSession` layer can route image-changes back
    /// through a fresh `chat_session_start` uniformly across model backends.
    ChatSessionContinue {
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session with a tool-result delta. See
    /// [`Lfm2Inner::chat_session_continue_tool_sync`] — builds a plain
    /// `<|im_start|>tool\n{content}<|im_end|>` delta (matching LFM2's
    /// template which does NOT use Qwen3.5's `<tool_response>` wrapping)
    /// and prefills on top of the live caches.
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
    /// Reset all caches and clear cached token history. Exposed so tests
    /// and session-management code can start from a known clean state
    /// between turns.
    ResetCaches { reply: ResponseTx<()> },
}

/// Wrapper to adapt `StreamTx` to the same `call()` API as
/// napi `ThreadsafeFunction`, so decode loop code can use a uniform interface.
struct StreamSender(StreamTx<ChatStreamChunk>);

impl StreamSender {
    fn call(&self, result: napi::Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
        let _ = self.0.send(result);
    }
}

impl Lfm2Inner {
    /// Create a new Lfm2Inner with empty (uninitialized) weights.
    pub(crate) fn new(config: Lfm2Config) -> Result<Self> {
        let num_layers = config.num_hidden_layers as usize;
        let hidden_size = config.hidden_size as u32;
        let vocab_size = config.vocab_size as u32;

        let embed_tokens = Embedding::new(vocab_size, hidden_size)?;
        let embedding_norm = RMSNorm::new(hidden_size, Some(config.norm_eps))?;

        let lm_head = if config.tie_embedding {
            None
        } else {
            Some(Linear::new(hidden_size, vocab_size, Some(false))?)
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(Lfm2DecoderLayer::new(&config, i)?);
        }

        // Initialize caches
        let caches = init_caches(&config);

        Ok(Self {
            config,
            embed_tokens,
            layers,
            embedding_norm,
            lm_head,
            caches,
            tokenizer: None,
            cached_token_history: Vec::new(),
            cached_image_key: None,
        })
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Forward pass through the full model.
    ///
    /// Follows `lfm2.py:258-279` (Lfm2Model.__call__) + Model.__call__ (tied lm_head).
    ///
    /// Returns logits [B, T, vocab_size].
    fn forward(&mut self, input_ids: &MxArray) -> Result<MxArray> {
        // 1. Token embeddings (no scaling)
        let mut h = self.embed_tokens.forward(input_ids)?;

        // 2. Iterate through layers
        // No explicit causal mask — attention layers use the fused
        // `scaled_dot_product_attention_causal()` path when mask is None and
        // seq_len > 1 (prefill). This avoids O(T^2) mask memory.
        // Conv layers always get None mask.
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, None, Some(&mut self.caches[i]))?;
        }

        // 3. Output norm
        h = self.embedding_norm.forward(&h)?;

        // 4. LM head or tied embeddings
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&h)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            h.matmul(&weight_t)?
        };

        Ok(logits)
    }

    /// Chunked prefill: process prompt in chunks of PREFILL_STEP_SIZE tokens,
    /// evaluating caches after each chunk to avoid OOM on long prompts.
    fn chunked_prefill(&mut self, prompt: &MxArray, generation_stream: Stream) -> Result<MxArray> {
        let total_len = prompt.shape_at(1)?;
        let mut offset: i64 = 0;
        while total_len - offset > PREFILL_STEP_SIZE {
            let chunk = prompt.slice_axis(1, offset, offset + PREFILL_STEP_SIZE)?;
            {
                let _stream_ctx = StreamContext::new(generation_stream);
                let _logits = self.forward(&chunk)?;
            }
            eval_lfm2_caches(&self.caches);
            crate::array::clear_cache();
            offset += PREFILL_STEP_SIZE;
        }
        let remaining = prompt.slice_axis(1, offset, total_len)?;
        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            self.forward(&remaining)?
        };
        Ok(logits)
    }

    /// Reset all caches and cached token history.
    fn reset_caches(&mut self) {
        self.caches = init_caches(&self.config);
        self.cached_token_history.clear();
        self.cached_image_key = None;
    }

    /// Check if tokens share a prefix with cached_token_history and return the prefix length.
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        if !reuse_cache {
            return 0;
        }
        let cached = &self.cached_token_history;
        if !cached.is_empty()
            && tokens.len() >= cached.len()
            && tokens[..cached.len()] == cached[..]
        {
            cached.len()
        } else {
            0
        }
    }

    /// Save cache state for reuse in the next chat-session continue call.
    ///
    /// `last_token_in_cache` must reflect whether the final entry in
    /// `generated_tokens` was actually forwarded through the model and written
    /// into the KV/conv caches. The loop skips the forward pass for the token
    /// sampled at `step == max_new_tokens - 1`, so in that case the last pushed
    /// token is NOT in the caches even if the loop exits via EOS or another
    /// early-stop reason. Trimming based on the cache state (not the finish
    /// reason string) keeps `cached_token_history` aligned with the layer
    /// caches so a later `reuse_cache=true` call can't skip prefill for an
    /// uncached tail token.
    fn save_cache_state(
        &mut self,
        reuse_cache: bool,
        tokens: &[u32],
        generated_tokens: &[u32],
        last_token_in_cache: bool,
    ) {
        if reuse_cache {
            let mut full_history = tokens.to_vec();
            let history_tokens = if !last_token_in_cache && !generated_tokens.is_empty() {
                &generated_tokens[..generated_tokens.len() - 1]
            } else {
                generated_tokens
            };
            full_history.extend_from_slice(history_tokens);
            self.cached_token_history = full_history;
        } else {
            self.reset_caches();
        }
    }

    /// Core synchronous chat implementation.
    ///
    /// `eos_token_id` is the caller-supplied stop-on token id (e.g.
    /// `<|im_end|>` for Qwen-style ChatML delimiters). Session entry
    /// points always supply this explicitly.
    fn chat_sync_core(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        eos_token_id: u32,
    ) -> Result<ChatResult> {
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        let tool_defs = config.tools.as_deref();
        let enable_thinking = resolve_enable_thinking(&config);
        let include_reasoning = resolve_include_reasoning(&config);
        let p = extract_chat_params(&config);
        let reuse_cache = p.reuse_cache;
        let report_perf = p.report_performance;
        let max_new_tokens = p.max_new_tokens;

        let tokens = tokenizer.apply_chat_template_sync(
            &messages,
            Some(true),
            tool_defs,
            enable_thinking,
        )?;

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Cache reuse: prefix verification
        let cached_prefix_len = self.verify_cache_prefix(&tokens, reuse_cache);

        let prefill_tokens = if cached_prefix_len > 0 {
            info!(
                "Cache reuse: {} cached tokens, {} new tokens to prefill",
                cached_prefix_len,
                tokens.len() - cached_prefix_len,
            );
            tokens[cached_prefix_len..].to_vec()
        } else {
            self.reset_caches();
            tokens.clone()
        };

        // Zero-delta guard
        let (prefill_tokens, _cached_prefix_len) = if prefill_tokens.is_empty() {
            info!("Zero-delta cache hit: resetting caches for full re-prefill");
            self.reset_caches();
            (tokens.clone(), 0)
        } else {
            (prefill_tokens, cached_prefix_len)
        };

        let eos_id = eos_token_id;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = tokens.clone();
        let mut finish_reason = String::from("length");

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let prompt_token_count = tokens.len();

        // Reasoning tracker
        let thinking_enabled = enable_thinking.unwrap_or(true);
        let mut reasoning_tracker =
            ReasoningTracker::new(thinking_enabled, p.thinking_token_budget, think_end_id);

        // Prefill: process prompt tokens through chunked forward pass
        let token_arr: Vec<i32> = prefill_tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, prefill_tokens.len() as i64])?;

        let logits = self.chunked_prefill(&prompt, generation_stream)?;

        // Take logits for last token only (use actual returned seq len, not total prompt len)
        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;

        // Apply penalties and sample first token
        let sampling_config = p.sampling_config;
        let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, sampling_config)?;
        y.eval();

        // Eval all caches after prefill
        eval_lfm2_caches(&self.caches);

        // Mark first token time
        if report_perf {
            first_token_instant = Some(std::time::Instant::now());
        }

        // Decode loop — double-buffered lazy eval pattern
        let mut last_token_in_cache = false;
        for step in 0..max_new_tokens {
            let next_y = if step + 1 < max_new_tokens {
                let _stream_ctx = StreamContext::new(generation_stream);

                let next_ids = y.reshape(&[1, 1])?;
                let logits = self.forward(&next_ids)?;
                let logits = logits.squeeze(Some(&[1]))?;

                // Budget enforcement
                let (next_token, _budget_forced) = if reasoning_tracker.should_force_think_end() {
                    let forced_id = reasoning_tracker.forced_token_id() as i32;
                    (MxArray::from_int32(&[forced_id], &[1])?, true)
                } else {
                    let logits = apply_all_penalties(logits, &token_history, &p)?;
                    let t = sample(&logits, sampling_config)?;
                    (t, false)
                };

                MxArray::async_eval_arrays(&[&next_token]);
                Some(next_token)
            } else {
                None
            };

            // The forward pass inside the branch above writes the current `y`
            // into KV/conv caches, so the token we are about to push is cached
            // iff that branch ran (i.e. `next_y.is_some()`).
            last_token_in_cache = next_y.is_some();

            // Extract current token
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            reasoning_tracker.observe_token(token_id);

            // Check stop condition
            if token_id == eos_id {
                finish_reason = String::from("stop");
                break;
            }

            if let Some(reason) = crate::sampling::check_repetition_cutoff(
                &generated_tokens,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }

            match next_y {
                Some(next) => y = next,
                None => break,
            }

            if (step + 1) % 256 == 0 {
                crate::array::synchronize_and_clear_cache();
            }
        }

        // Save cache state for next call
        self.save_cache_state(reuse_cache, &tokens, &generated_tokens, last_token_in_cache);

        // Compute performance metrics
        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                prefill_tokens.len(),
                generated_tokens.len(),
            )
        } else {
            None
        };

        let reasoning_tokens = reasoning_tracker.reasoning_token_count();

        finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            include_reasoning,
            thinking_enabled,
            prompt_token_count as u32,
            reasoning_tokens,
        )
    }

    /// Core streaming chat implementation.
    ///
    /// `eos_token_id` is the caller-supplied stop-on token id (e.g.
    /// `<|im_end|>` for Qwen-style ChatML delimiters). Session entry
    /// points always supply this explicitly.
    fn chat_stream_sync_core(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        eos_token_id: u32,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<()> {
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        let tool_defs = config.tools.as_deref();
        let enable_thinking = resolve_enable_thinking(&config);
        let include_reasoning = resolve_include_reasoning(&config);
        let p = extract_chat_params(&config);
        let reuse_cache = p.reuse_cache;
        let report_perf = p.report_performance;
        let max_new_tokens = p.max_new_tokens;

        let tokens = tokenizer.apply_chat_template_sync(
            &messages,
            Some(true),
            tool_defs,
            enable_thinking,
        )?;

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Cache reuse
        let cached_prefix_len = self.verify_cache_prefix(&tokens, reuse_cache);

        let prefill_tokens = if cached_prefix_len > 0 {
            tokens[cached_prefix_len..].to_vec()
        } else {
            self.reset_caches();
            tokens.clone()
        };

        let (prefill_tokens, _) = if prefill_tokens.is_empty() {
            self.reset_caches();
            (tokens.clone(), 0)
        } else {
            (prefill_tokens, cached_prefix_len)
        };

        let eos_id = eos_token_id;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut token_history: Vec<u32> = tokens.clone();
        let mut finish_reason = String::from("length");

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let prompt_token_count = tokens.len();

        // Reasoning tracker
        let thinking_enabled = enable_thinking.unwrap_or(true);
        let mut reasoning_tracker =
            ReasoningTracker::new(thinking_enabled, p.thinking_token_budget, think_end_id);

        // Streaming decode state
        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        // Prefill: chunked forward pass
        let token_arr: Vec<i32> = prefill_tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, prefill_tokens.len() as i64])?;

        let logits = self.chunked_prefill(&prompt, generation_stream)?;
        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;

        let sampling_config = p.sampling_config;
        let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, sampling_config)?;
        y.eval();
        eval_lfm2_caches(&self.caches);

        if report_perf {
            first_token_instant = Some(std::time::Instant::now());
        }

        // Decode loop
        let mut last_token_in_cache = false;
        for step in 0..max_new_tokens {
            let next_y = if step + 1 < max_new_tokens {
                let _stream_ctx = StreamContext::new(generation_stream);

                let next_ids = y.reshape(&[1, 1])?;
                let logits = self.forward(&next_ids)?;
                let logits = logits.squeeze(Some(&[1]))?;

                let (next_token, _budget_forced) = if reasoning_tracker.should_force_think_end() {
                    let forced_id = reasoning_tracker.forced_token_id() as i32;
                    (MxArray::from_int32(&[forced_id], &[1])?, true)
                } else {
                    let logits = apply_all_penalties(logits, &token_history, &p)?;
                    let t = sample(&logits, sampling_config)?;
                    (t, false)
                };

                MxArray::async_eval_arrays(&[&next_token]);
                Some(next_token)
            } else {
                None
            };

            // The forward pass inside the branch above writes the current `y`
            // into KV/conv caches, so the token we are about to push is cached
            // iff that branch ran (i.e. `next_y.is_some()`).
            last_token_in_cache = next_y.is_some();

            // Extract current token
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            let is_reasoning = reasoning_tracker.observe_token(token_id);
            last_is_reasoning = is_reasoning;

            // Check stop condition before streaming to avoid leaking EOS text
            if token_id == eos_id {
                finish_reason = String::from("stop");
                break;
            }

            // Check cancellation
            if cancelled.load(Ordering::Relaxed) {
                finish_reason = String::from("cancelled");
                break;
            }

            // Stream delta chunk
            let token_text = Qwen3Tokenizer::step_decode_stream(
                &mut decode_stream,
                tokenizer.inner(),
                token_id,
                &generated_tokens,
                streamed_text_len,
            );
            streamed_text_len += token_text.len();
            cb.call(
                Ok(ChatStreamChunk {
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
                    is_reasoning: Some(is_reasoning),
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );

            if let Some(reason) = crate::sampling::check_repetition_cutoff(
                &generated_tokens,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }

            match next_y {
                Some(next) => y = next,
                None => break,
            }

            if (step + 1) % 256 == 0 {
                crate::array::synchronize_and_clear_cache();
            }
        }

        // Save cache state
        self.save_cache_state(reuse_cache, &tokens, &generated_tokens, last_token_in_cache);

        // Flush residual buffered bytes from decode_stream
        let full_text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            let residual = full_text[streamed_text_len..].to_string();
            cb.call(
                Ok(ChatStreamChunk {
                    text: residual,
                    done: false,
                    finish_reason: None,
                    tool_calls: None,
                    thinking: None,
                    num_tokens: None,
                    prompt_tokens: None,
                    reasoning_tokens: None,
                    raw_text: None,
                    performance: None,
                    is_reasoning: Some(last_is_reasoning),
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
        }

        // Build final result
        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                prefill_tokens.len(),
                generated_tokens.len(),
            )
        } else {
            None
        };

        let reasoning_tokens = reasoning_tracker.reasoning_token_count();

        let result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            include_reasoning,
            thinking_enabled,
            prompt_token_count as u32,
            reasoning_tokens,
        )?;

        // Send final chunk
        cb.call(
            Ok(ChatStreamChunk {
                text: result.text.clone(),
                done: true,
                finish_reason: Some(result.finish_reason.clone()),
                tool_calls: Some(result.tool_calls.clone()),
                thinking: result.thinking.clone(),
                num_tokens: Some(result.num_tokens),
                prompt_tokens: Some(result.prompt_tokens),
                reasoning_tokens: Some(result.reasoning_tokens),
                raw_text: Some(result.raw_text.clone()),
                performance: result.performance.clone(),
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    // =================================================================
    // Session API (mirrors the Qwen3.5 MoE surface, text-only).
    // =================================================================

    /// Start a new chat session.
    ///
    /// Fully resets the caches and delegates to [`Self::chat_sync_core`]
    /// with `<|im_end|>` as the stop token so the decode loop leaves the
    /// caches on a clean ChatML boundary that subsequent
    /// [`Self::chat_session_continue_sync`] /
    /// [`Self::chat_session_continue_tool_sync`] calls can append a raw
    /// delta on top of.
    pub(crate) fn chat_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // Mirror the symmetric guard in `chat_tokens_delta_sync`. The
        // session API only makes sense with cache reuse enabled.
        if config.reuse_cache == Some(false) {
            return Err(Error::from_reason(
                "chat_session_start requires reuse_cache=true (pass ChatConfig { reuse_cache: Some(true), .. } or leave as None). The session API only makes sense with cache reuse enabled.",
            ));
        }

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();
        let im_end_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        // Full reset: the session-start path always begins from a clean
        // state.
        self.reset_caches();

        self.chat_sync_core(messages, config, im_end_id)
    }

    /// Prefill a pre-tokenized delta on top of the existing LFM2 caches
    /// (conv state + KV) and run the decode loop. Text-only session
    /// primitive used by [`Self::chat_session_continue_sync`] and
    /// [`Self::chat_session_continue_tool_sync`].
    ///
    /// Uses `<|im_end|>` as the eos token (not `config.eos_token_id`) so
    /// the cached history continues to end on a clean ChatML boundary
    /// for the next turn. `save_cache_state` runs unconditionally at
    /// the end so the session stays consistent even on error.
    pub(crate) fn chat_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        // The delta path is a session-reuse operation by construction.
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
                "chat_tokens_delta_sync requires a non-empty delta",
            ));
        }
        if self.cached_image_key.is_some() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync is text-only; session currently holds image state",
            ));
        }

        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        let p = extract_chat_params(&config);
        let report_perf = p.report_performance;
        let max_new_tokens = p.max_new_tokens;
        let enable_thinking = resolve_enable_thinking(&config);
        let include_reasoning = resolve_include_reasoning(&config);
        let thinking_enabled = enable_thinking.unwrap_or(true);

        // Build full token history = cached_history + delta. Used for
        // penalty context AND as the running token history in the
        // decode loop.
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let prompt_token_count = full_token_history.len() as u32;

        // Save snapshot for save_cache_state (prior history + delta).
        let save_tokens = full_token_history.clone();

        let mut reasoning_tracker =
            ReasoningTracker::new(thinking_enabled, p.thinking_token_budget, think_end_id);

        // Prefill: chunked forward pass of the delta on top of existing caches.
        let token_arr: Vec<i32> = delta_tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, delta_tokens.len() as i64])?;

        let logits = self.chunked_prefill(&prompt, generation_stream)?;

        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;

        let sampling_config = p.sampling_config;
        let mut token_history: Vec<u32> = full_token_history;
        let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, sampling_config)?;
        y.eval();

        // Eval all caches after prefill so the prefix is materialized.
        eval_lfm2_caches(&self.caches);

        if report_perf {
            first_token_instant = Some(std::time::Instant::now());
        }

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");

        // Decode loop — double-buffered lazy eval pattern (same as chat_sync_core).
        let mut last_token_in_cache = false;
        for step in 0..max_new_tokens {
            let next_y = if step + 1 < max_new_tokens {
                let _stream_ctx = StreamContext::new(generation_stream);

                let next_ids = y.reshape(&[1, 1])?;
                let logits = self.forward(&next_ids)?;
                let logits = logits.squeeze(Some(&[1]))?;

                let (next_token, _budget_forced) = if reasoning_tracker.should_force_think_end() {
                    let forced_id = reasoning_tracker.forced_token_id() as i32;
                    (MxArray::from_int32(&[forced_id], &[1])?, true)
                } else {
                    let logits = apply_all_penalties(logits, &token_history, &p)?;
                    let t = sample(&logits, sampling_config)?;
                    (t, false)
                };

                MxArray::async_eval_arrays(&[&next_token]);
                Some(next_token)
            } else {
                None
            };

            last_token_in_cache = next_y.is_some();

            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            reasoning_tracker.observe_token(token_id);

            if token_id == eos_id {
                finish_reason = String::from("stop");
                break;
            }

            if let Some(reason) = crate::sampling::check_repetition_cutoff(
                &generated_tokens,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }

            match next_y {
                Some(next) => y = next,
                None => break,
            }

            if (step + 1) % 256 == 0 {
                crate::array::synchronize_and_clear_cache();
            }
        }

        // Save cache state unconditionally so the session stays
        // consistent for the next turn. The delta path always runs with
        // reuse_cache enabled (guarded above), so pass `true` directly.
        self.save_cache_state(true, &save_tokens, &generated_tokens, last_token_in_cache);

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                delta_tokens.len(),
                generated_tokens.len(),
            )
        } else {
            None
        };

        let reasoning_tokens = reasoning_tracker.reasoning_token_count();

        finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tokens,
        )
    }

    /// Session-based chat continuation via a plain user message string.
    ///
    /// Builds the ChatML delta (closes the previous `<|im_end|>` line,
    /// opens a new user turn with `user_message`, and opens a fresh
    /// assistant turn). Delegates to
    /// [`Self::chat_tokens_delta_sync`] which handles the actual
    /// prefill-on-top-of-cache + decode path.
    ///
    /// LFM2 is text-only; `images` is an opt-in guard parameter:
    /// non-empty input is rejected with an
    /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed error so the
    /// TS `ChatSession` layer can route image-changes back through a
    /// fresh `chat_session_start` uniformly across all model backends.
    pub(crate) fn chat_session_continue_sync(
        &mut self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        if images.as_ref().is_some_and(|v| !v.is_empty()) {
            return Err(Error::from_reason(format!(
                "{} chat_session_continue is text-only; start a new session with chat_session_start to change the image",
                IMAGE_CHANGE_RESTART_PREFIX
            )));
        }

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Match `chat_sync`'s sanitization so the session path is
        // subject to the same role/content injection protection as the
        // legacy path.
        let synthetic = build_synthetic_user_message(&user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        // LFM2's chat template does NOT inject `<think>\n` after the
        // assistant opener — the model emits `<think>` tags on its own
        // when reasoning. Always suppress the prefix by passing
        // `Some(false)` to the shared builder so the delta stays
        // template-equivalent with the LFM2 jinja output.
        let delta_text = build_chatml_continue_delta_text(sanitized_user, Some(false));

        let delta_tokens = tokenizer.encode_sync(&delta_text, Some(false))?;

        self.chat_tokens_delta_sync(delta_tokens, config)
    }

    /// Session-based chat continuation via a tool-result turn.
    ///
    /// LFM2's chat template renders tool-role messages as a plain
    /// `<|im_start|>tool\n{content}<|im_end|>` block — it does NOT use
    /// Qwen3.5's `<tool_response>`-wrapped `user`-role variant. We
    /// therefore build the delta inline rather than calling
    /// `chat_common::build_chatml_tool_delta_text` (which is
    /// Qwen3.5-specific). The `tool_call_id` is intentionally dropped
    /// from the wire format — LFM2's template identifies tool responses
    /// positionally, like Qwen3.5 does.
    ///
    /// Delegates to [`Self::chat_tokens_delta_sync`] which inherits the
    /// same text-only-delta invariant (errors if the session currently
    /// holds image state).
    pub(crate) fn chat_session_continue_tool_sync(
        &mut self,
        _tool_call_id: String,
        content: String,
        config: ChatConfig,
    ) -> Result<ChatResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Build the LFM2-specific tool delta inline. Leading `\n`
        // closes the cached `<|im_end|>` line, then we open a `tool`
        // role turn, close it, and open an assistant turn ready for
        // generation. No `<think>\n` prefix because LFM2's template
        // does not inject one.
        let delta_text =
            format!("\n<|im_start|>tool\n{content}<|im_end|>\n<|im_start|>assistant\n");

        let delta_tokens = tokenizer.encode_sync(&delta_text, Some(false))?;

        self.chat_tokens_delta_sync(delta_tokens, config)
    }

    /// Streaming chat (session-start variant): same semantics as
    /// [`Self::chat_session_start_sync`] but streams token deltas
    /// through `stream_tx`.
    pub(crate) fn chat_stream_session_start_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        let cb = StreamSender(stream_tx.clone());

        if cancelled.load(Ordering::Relaxed) {
            send_stream_error(
                &stream_tx,
                "chat_stream_session_start cancelled before start",
            );
            return;
        }

        if config.reuse_cache == Some(false) {
            send_stream_error(
                &stream_tx,
                "chat_stream_session_start requires reuse_cache=true (leave as None or set to true). \
                 The session API only makes sense with cache reuse enabled.",
            );
            return;
        }

        let im_end_id = match self.tokenizer.as_ref().and_then(|t| t.im_end_id()) {
            Some(id) => id,
            None => {
                send_stream_error(
                    &stream_tx,
                    "chat_stream_session_start requires a tokenizer with an <|im_end|> special token",
                );
                return;
            }
        };

        // Full reset: the session-start path always begins clean.
        self.reset_caches();

        let result = self.chat_stream_sync_core(messages, config, im_end_id, &cb, &cancelled);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    /// Streaming chat (session-continue variant): same semantics as
    /// [`Self::chat_session_continue_sync`] but streams token deltas
    /// through `stream_tx`.
    pub(crate) fn chat_stream_session_continue_sync(
        &mut self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            send_stream_error(
                &stream_tx,
                "chat_stream_session_continue cancelled before start",
            );
            return;
        }

        if images.as_ref().is_some_and(|v| !v.is_empty()) {
            send_stream_error(
                &stream_tx,
                &format!(
                    "{} chat_stream_session_continue is text-only; start a new session with chat_stream_session_start to change the image",
                    IMAGE_CHANGE_RESTART_PREFIX
                ),
            );
            return;
        }

        let tokenizer = match self.tokenizer.as_ref() {
            Some(t) => t.clone(),
            None => {
                send_stream_error(&stream_tx, "Tokenizer not loaded");
                return;
            }
        };

        let synthetic = build_synthetic_user_message(&user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        // LFM2 template does NOT inject `<think>\n`; always suppress.
        let delta_text = build_chatml_continue_delta_text(sanitized_user, Some(false));

        let delta_tokens = match tokenizer.encode_sync(&delta_text, Some(false)) {
            Ok(t) => t,
            Err(e) => {
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        self.chat_stream_tokens_delta_sync(delta_tokens, config, stream_tx, cancelled);
    }

    /// Streaming analog of [`Self::chat_session_continue_tool_sync`].
    pub(crate) fn chat_stream_session_continue_tool_sync(
        &mut self,
        _tool_call_id: String,
        content: String,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            send_stream_error(
                &stream_tx,
                "chat_stream_session_continue_tool cancelled before start",
            );
            return;
        }

        let tokenizer = match self.tokenizer.as_ref() {
            Some(t) => t.clone(),
            None => {
                send_stream_error(&stream_tx, "Tokenizer not loaded");
                return;
            }
        };

        // LFM2-specific plain tool delta (no `<tool_response>`
        // wrapper). See `chat_session_continue_tool_sync` for the
        // rationale.
        let delta_text =
            format!("\n<|im_start|>tool\n{content}<|im_end|>\n<|im_start|>assistant\n");

        let delta_tokens = match tokenizer.encode_sync(&delta_text, Some(false)) {
            Ok(t) => t,
            Err(e) => {
                let _ = stream_tx.send(Err(e));
                return;
            }
        };

        self.chat_stream_tokens_delta_sync(delta_tokens, config, stream_tx, cancelled);
    }

    /// Streaming analog of [`Self::chat_tokens_delta_sync`]: prefill
    /// the caller-provided delta tokens on top of the existing LFM2
    /// caches and stream the reply through `stream_tx`.
    ///
    /// Applies the same four guards as the non-streaming path and
    /// still calls `save_cache_state` at the end regardless of whether
    /// cancellation fired, so the cache stays consistent for the next
    /// turn even on early abort.
    pub(crate) fn chat_stream_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        if cancelled.load(Ordering::Relaxed) {
            send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta cancelled before start",
            );
            return;
        }

        // --- Same four guards as chat_tokens_delta_sync ---
        if config.reuse_cache == Some(false) {
            send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires reuse_cache to be enabled; \
                 the delta path operates on session state by construction",
            );
            return;
        }
        if self.cached_token_history.is_empty() {
            send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires an initialized session (call chatStreamSessionStart first)",
            );
            return;
        }
        if delta_tokens.is_empty() {
            send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta requires a non-empty delta",
            );
            return;
        }
        if self.cached_image_key.is_some() {
            send_stream_error(
                &stream_tx,
                "chat_stream_tokens_delta is text-only; session currently holds image state",
            );
            return;
        }

        let cb = StreamSender(stream_tx.clone());
        let result =
            self.chat_stream_tokens_delta_sync_inner(delta_tokens, config, &cb, &cancelled);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    /// Prefill the delta tokens and run the streaming decode loop.
    ///
    /// Mirrors [`Self::chat_stream_sync_core`] but skips the message
    /// rendering + prefix verification stages — the caller owns cache
    /// coherence by construction. Uses `<|im_end|>` as eos so the
    /// cached history continues to end on a clean ChatML boundary
    /// after the reply is saved.
    fn chat_stream_tokens_delta_sync_inner(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<()> {
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        let p = extract_chat_params(&config);
        let report_perf = p.report_performance;
        let max_new_tokens = p.max_new_tokens;
        let enable_thinking = resolve_enable_thinking(&config);
        let include_reasoning = resolve_include_reasoning(&config);
        let thinking_enabled = enable_thinking.unwrap_or(true);

        // Build full token history = cached_history + delta.
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let prompt_token_count = full_token_history.len() as u32;

        // Save snapshot for save_cache_state (prior history + delta).
        let save_tokens = full_token_history.clone();

        let mut reasoning_tracker =
            ReasoningTracker::new(thinking_enabled, p.thinking_token_budget, think_end_id);

        // Streaming decode state
        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        // Prefill: chunked forward pass of the delta on top of existing caches.
        let token_arr: Vec<i32> = delta_tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, delta_tokens.len() as i64])?;

        let logits = self.chunked_prefill(&prompt, generation_stream)?;
        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;

        let sampling_config = p.sampling_config;
        let mut token_history: Vec<u32> = full_token_history;
        let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, sampling_config)?;
        y.eval();
        eval_lfm2_caches(&self.caches);

        if report_perf {
            first_token_instant = Some(std::time::Instant::now());
        }

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");

        // Decode loop
        let mut last_token_in_cache = false;
        for step in 0..max_new_tokens {
            let next_y = if step + 1 < max_new_tokens {
                let _stream_ctx = StreamContext::new(generation_stream);

                let next_ids = y.reshape(&[1, 1])?;
                let logits = self.forward(&next_ids)?;
                let logits = logits.squeeze(Some(&[1]))?;

                let (next_token, _budget_forced) = if reasoning_tracker.should_force_think_end() {
                    let forced_id = reasoning_tracker.forced_token_id() as i32;
                    (MxArray::from_int32(&[forced_id], &[1])?, true)
                } else {
                    let logits = apply_all_penalties(logits, &token_history, &p)?;
                    let t = sample(&logits, sampling_config)?;
                    (t, false)
                };

                MxArray::async_eval_arrays(&[&next_token]);
                Some(next_token)
            } else {
                None
            };

            last_token_in_cache = next_y.is_some();

            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            let is_reasoning = reasoning_tracker.observe_token(token_id);
            last_is_reasoning = is_reasoning;

            if token_id == eos_id {
                finish_reason = String::from("stop");
                break;
            }

            if cancelled.load(Ordering::Relaxed) {
                finish_reason = String::from("cancelled");
                break;
            }

            // Stream delta chunk
            let token_text = Qwen3Tokenizer::step_decode_stream(
                &mut decode_stream,
                tokenizer.inner(),
                token_id,
                &generated_tokens,
                streamed_text_len,
            );
            streamed_text_len += token_text.len();
            cb.call(
                Ok(ChatStreamChunk {
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
                    is_reasoning: Some(is_reasoning),
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );

            if let Some(reason) = crate::sampling::check_repetition_cutoff(
                &generated_tokens,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }

            match next_y {
                Some(next) => y = next,
                None => break,
            }

            if (step + 1) % 256 == 0 {
                crate::array::synchronize_and_clear_cache();
            }
        }

        // Save cache state unconditionally — even on cancellation, the
        // partial generated_tokens must be appended so the session
        // stays consistent for the next turn.
        self.save_cache_state(true, &save_tokens, &generated_tokens, last_token_in_cache);

        // Flush residual buffered bytes from decode_stream
        let full_text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            let residual = full_text[streamed_text_len..].to_string();
            cb.call(
                Ok(ChatStreamChunk {
                    text: residual,
                    done: false,
                    finish_reason: None,
                    tool_calls: None,
                    thinking: None,
                    num_tokens: None,
                    prompt_tokens: None,
                    reasoning_tokens: None,
                    raw_text: None,
                    performance: None,
                    is_reasoning: Some(last_is_reasoning),
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
        }

        // Build the final done chunk with parsed tool/thinking info.
        let (clean_text, tool_calls, thinking) = parse_thinking_and_tools(
            &full_text,
            &generated_tokens,
            thinking_enabled,
            think_end_id,
            think_end_str.as_deref(),
            include_reasoning,
        );

        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                delta_tokens.len(),
                generated_tokens.len(),
            )
        } else {
            None
        };

        cb.call(
            Ok(ChatStreamChunk {
                text: clean_text,
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                num_tokens: Some(generated_tokens.len() as u32),
                prompt_tokens: Some(prompt_token_count),
                reasoning_tokens: Some(reasoning_tracker.reasoning_token_count()),
                raw_text: Some(full_text),
                performance,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }
}

/// Command handler for the dedicated model thread.
pub(crate) fn handle_lfm2_cmd(inner: &mut Lfm2Inner, cmd: Lfm2Cmd) {
    match cmd {
        Lfm2Cmd::ChatTokensDelta {
            delta_tokens,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_tokens_delta_sync(delta_tokens, config));
        }
        Lfm2Cmd::ChatSessionStart {
            messages,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_start_sync(messages, config));
        }
        Lfm2Cmd::ChatSessionContinue {
            user_message,
            images,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_session_continue_sync(user_message, images, config));
        }
        Lfm2Cmd::ChatSessionContinueTool {
            tool_call_id,
            content,
            config,
            reply,
        } => {
            let _ =
                reply.send(inner.chat_session_continue_tool_sync(tool_call_id, content, config));
        }
        Lfm2Cmd::ChatStreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_session_start_sync(messages, config, stream_tx, cancelled);
        }
        Lfm2Cmd::ChatStreamSessionContinue {
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
        Lfm2Cmd::ChatStreamSessionContinueTool {
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
        Lfm2Cmd::ResetCaches { reply } => {
            inner.reset_caches();
            let _ = reply.send(Ok(()));
        }
    }
}

/// Initialize caches matching the layer types.
fn init_caches(config: &Lfm2Config) -> Vec<Lfm2LayerCache> {
    let num_layers = config.num_hidden_layers as usize;
    let mut caches = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if config.is_attention_layer(i) {
            caches.push(Lfm2LayerCache::new_attention());
        } else {
            caches.push(Lfm2LayerCache::new_conv());
        }
    }
    caches
}

const PREFILL_STEP_SIZE: i64 = 2048;

/// Evaluate all cache arrays (after prefill).
fn eval_lfm2_caches(caches: &[Lfm2LayerCache]) {
    let mut arrays: Vec<&MxArray> = Vec::new();
    for cache in caches {
        cache.collect_arrays(&mut arrays);
    }
    if !arrays.is_empty() {
        MxArray::eval_arrays(&arrays);
    }
}

/// LFM2 language model (LFM2.5-1.2B-Thinking).
///
/// Hybrid conv+attention architecture from Liquid AI. 16 layers total:
/// 10 conv layers + 6 full_attention layers. Features gated short
/// convolutions for local processing and standard attention for global context.
///
/// All model state lives on a dedicated OS thread. NAPI methods dispatch
/// commands via channels and await responses.
#[napi]
pub struct Lfm2Model {
    pub(crate) thread: crate::model_thread::ModelThread<Lfm2Cmd>,
    pub(crate) config: Lfm2Config,
}

#[napi]
impl Lfm2Model {
    /// Load an LFM2 model from a directory containing safetensors and config.json.
    #[napi]
    pub async fn load(model_path: String) -> Result<Lfm2Model> {
        Lfm2Model::load_from_dir(&model_path).await
    }

    /// Reset all caches and clear cached token history. Exposed so
    /// tests and session-management code can start from a known clean
    /// state between turns.
    #[napi]
    pub fn reset_caches(&self) -> Result<()> {
        crate::model_thread::send_and_block(&self.thread, |reply| Lfm2Cmd::ResetCaches { reply })
    }

    /// Start a new chat session.
    ///
    /// Runs the full jinja chat template once, decodes until
    /// `<|im_end|>`, and leaves the KV/conv caches on a clean ChatML
    /// boundary so subsequent `chatSessionContinue` /
    /// `chatSessionContinueTool` calls can append a raw delta on top
    /// without re-rendering the chat template.
    ///
    /// Requires `config.reuse_cache` to be enabled (the default).
    #[napi]
    pub async fn chat_session_start(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        if messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()))
        {
            return Err(Error::from_reason(format!(
                "{} LFM2 is text-only; image messages are not supported",
                IMAGE_CHANGE_RESTART_PREFIX
            )));
        }
        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(&self.thread, |reply| Lfm2Cmd::ChatSessionStart {
            messages,
            config,
            reply,
        })
        .await
    }

    /// Continue an existing chat session with a new user message.
    ///
    /// Appends a raw ChatML user/assistant delta to the session's
    /// cached KV/conv state, then decodes the assistant reply. Stops
    /// on `<|im_end|>` so the cache remains on a clean boundary for
    /// the next turn.
    ///
    /// Requires a live session started via `chatSessionStart`. Errors
    /// if the session is empty, carries image state, or if
    /// `config.reuse_cache` is explicitly set to `false`.
    ///
    /// LFM2 is text-only; `images` is an opt-in guard parameter: when
    /// non-empty the native side returns an error whose message begins
    /// with `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` so the TypeScript
    /// `ChatSession` layer can catch the prefix and route
    /// image-changes back through a fresh `chatSessionStart`
    /// uniformly across all model backends.
    #[napi(
        ts_args_type = "userMessage: string, images: Uint8Array[] | null | undefined, config: ChatConfig | null | undefined"
    )]
    pub async fn chat_session_continue(
        &self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(&self.thread, |reply| Lfm2Cmd::ChatSessionContinue {
            user_message,
            images,
            config,
            reply,
        })
        .await
    }

    /// Continue an existing chat session with a tool-result turn.
    ///
    /// Builds an LFM2-format tool delta (`<|im_start|>tool\n{content}
    /// <|im_end|>`) from `content` and prefills it on top of the live
    /// session caches, then decodes the assistant reply. Stops on
    /// `<|im_end|>` so the cache stays on a clean boundary for the
    /// next turn.
    ///
    /// The `tool_call_id` is currently dropped by the wire format —
    /// LFM2's chat template identifies tool responses positionally,
    /// not via an explicit id. Callers may still log it for their own
    /// bookkeeping.
    ///
    /// Requires a live session started via `chatSessionStart`.
    #[napi]
    pub async fn chat_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(&self.thread, |reply| {
            Lfm2Cmd::ChatSessionContinueTool {
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
        ts_args_type = "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_start(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        if messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()))
        {
            return Err(Error::from_reason(format!(
                "{} LFM2 is text-only; image messages are not supported",
                IMAGE_CHANGE_RESTART_PREFIX
            )));
        }
        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        self.thread.send(Lfm2Cmd::ChatStreamSessionStart {
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
        ts_args_type = "userMessage: string, images: Uint8Array[] | null | undefined, config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_continue(
        &self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        self.thread.send(Lfm2Cmd::ChatStreamSessionContinue {
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
        ts_args_type = "toolCallId: string, content: string, config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void"
    )]
    pub async fn chat_stream_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        self.thread.send(Lfm2Cmd::ChatStreamSessionContinueTool {
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

    // ---------------------------------------------------------------
    // Test-only helpers: streaming session entry points that bypass
    // ThreadsafeFunction and expose the mpsc receiver directly. Used
    // by `crates/mlx-core/tests/lfm2_session.rs` to exercise the
    // streaming path from a pure-Rust integration test without a
    // NAPI host. Marked `#[doc(hidden)]` because they're not part of
    // the public API surface.
    // ---------------------------------------------------------------

    /// Test-only entry point that dispatches `ChatStreamSessionStart`
    /// and returns the raw mpsc receiver the model thread writes into.
    #[doc(hidden)]
    pub fn chat_stream_session_start_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        self.thread.send(Lfm2Cmd::ChatStreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Test-only entry point that dispatches
    /// `ChatStreamSessionContinue` and returns the raw mpsc receiver
    /// the model thread writes into.
    #[doc(hidden)]
    pub fn chat_stream_session_continue_for_test(
        &self,
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        self.thread.send(Lfm2Cmd::ChatStreamSessionContinue {
            user_message,
            images,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Test-only entry point that dispatches
    /// `ChatStreamSessionContinueTool` and returns the raw mpsc
    /// receiver the model thread writes into.
    #[doc(hidden)]
    pub fn chat_stream_session_continue_tool_for_test(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        self.thread.send(Lfm2Cmd::ChatStreamSessionContinueTool {
            tool_call_id,
            content,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Get the model configuration.
    #[napi]
    pub fn get_config(&self) -> Lfm2Config {
        self.config.clone()
    }

    /// Estimated number of model parameters.
    #[napi]
    pub fn num_parameters(&self) -> i64 {
        let h = self.config.hidden_size as i64;
        let v = self.config.vocab_size as i64;
        let ff = self.config.computed_ff_dim() as i64;
        let hd = self.config.head_dim() as i64;
        let nh = self.config.num_attention_heads as i64;
        let nkv = self.config.num_key_value_heads as i64;

        // Embedding
        let mut total = v * h;

        // Separate lm_head when not tied
        if !self.config.tie_embedding {
            total += h * v; // lm_head.weight
        }

        // Output norm
        total += h;

        for i in 0..self.config.num_hidden_layers as usize {
            // operator_norm + ffn_norm
            total += 2 * h;

            // MLP: gate_proj + up_proj + down_proj
            total += h * ff + h * ff + ff * h;

            if self.config.is_attention_layer(i) {
                // Attention: q_proj + k_proj + v_proj + out_proj + q_layernorm + k_layernorm
                let q_dim = nh * hd;
                let kv_dim = nkv * hd;
                total += h * q_dim + h * kv_dim + h * kv_dim + q_dim * h;
                total += 2 * hd; // layernorms
            } else {
                // ShortConv: in_proj + out_proj + conv
                let l_cache = self.config.conv_l_cache as i64;
                total += h * (3 * h); // in_proj
                total += h * h; // out_proj
                total += h * l_cache; // depthwise conv (groups=h)
            }
        }

        total
    }
}
