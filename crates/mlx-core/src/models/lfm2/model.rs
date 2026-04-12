use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tracing::info;

use crate::array::MxArray;
use crate::model_thread::{ResponseTx, StreamTx};
use crate::models::qwen3_5::chat_common::{
    ReasoningTracker, apply_all_penalties, compute_performance_metrics, extract_chat_params,
    finalize_chat_result, resolve_enable_thinking, resolve_include_reasoning,
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
    /// Cached token history for KV cache reuse across chat() calls.
    pub(crate) cached_token_history: Vec<u32>,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Lfm2Cmd {
    Chat {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    ChatStream {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
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

    /// Save cache state for reuse in the next chat() call.
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

    /// Synchronous chat implementation. Runs on the dedicated model thread.
    fn chat_sync(&mut self, messages: Vec<ChatMessage>, config: ChatConfig) -> Result<ChatResult> {
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

        let eos_id = self.config.eos_token_id as u32;
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

    /// Synchronous streaming chat implementation. Runs on the dedicated model thread.
    fn chat_stream_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    ) {
        let cb = StreamSender(stream_tx.clone());
        let result = self.chat_stream_sync_inner(messages, config, &cb, &cancelled);
        if let Err(e) = result {
            let _ = stream_tx.send(Err(e));
        }
    }

    fn chat_stream_sync_inner(
        &mut self,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
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

        let eos_id = self.config.eos_token_id as u32;
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
}

/// Command handler for the dedicated model thread.
pub(crate) fn handle_lfm2_cmd(inner: &mut Lfm2Inner, cmd: Lfm2Cmd) {
    match cmd {
        Lfm2Cmd::Chat {
            messages,
            config,
            reply,
        } => {
            let result = inner.chat_sync(messages, config);
            let _ = reply.send(result);
        }
        Lfm2Cmd::ChatStream {
            messages,
            config,
            stream_tx,
            cancelled,
        } => {
            inner.chat_stream_sync(messages, config, stream_tx, cancelled);
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

    /// Chat with the model using a list of messages.
    #[napi]
    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(&self.thread, |reply| Lfm2Cmd::Chat {
            messages,
            config,
            reply,
        })
        .await
    }

    /// Streaming chat with the model. Calls the callback for each token chunk.
    #[napi]
    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
        callback: napi::threadsafe_function::ThreadsafeFunction<ChatStreamChunk, ()>,
    ) -> Result<ChatStreamHandle> {
        let config = config.unwrap_or_default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();

        // Create mpsc channel to bridge model thread -> tokio task -> JS callback
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();

        // Send streaming command to model thread
        self.thread.send(Lfm2Cmd::ChatStream {
            messages,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;

        // Spawn tokio task that reads from stream_rx and calls the JS callback
        let callback = Arc::new(callback);
        tokio::spawn(async move {
            while let Some(result) = stream_rx.recv().await {
                callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
            }
        });

        Ok(ChatStreamHandle { cancelled })
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
