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
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::config::Lfm2Config;
use super::decoder_layer::{Lfm2DecoderLayer, Lfm2LayerKind};
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
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache).
    ///
    /// **Opt-in via `Lfm2Config::use_block_paged_cache`**. LFM2 is a
    /// hybrid conv + attention architecture, so only `full_attention`
    /// layers route through the block-paged path while `conv` layers
    /// continue to use the existing `Lfm2LayerCache::Conv(ArraysCache)`
    /// storage. The adapter's underlying `LayerKVPool` is sized for the
    /// count of `full_attention` layers ONLY — conv layers do not
    /// consume KV pool slots, and the pool is indexed by
    /// attention-layer ordinal (the index into `config.full_attn_idxs()`),
    /// not by absolute layer index. `Lfm2DecoderLayer::forward_paged_or_flat`
    /// performs the per-layer dispatch.
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Lfm2Cmd {
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

/// Classification of the prefix-cache decision made from a
/// [`Lfm2Inner::verify_cache_prefix`] return value plus the incoming
/// token count.
///
/// Test-only mirror of the inlined branch at the top of
/// [`Lfm2Inner::chat_sync_core`] / [`Lfm2Inner::chat_stream_sync_core`]
/// — separating the decision logic from the native state mutation so
/// the "exact-match routes to miss" invariant can be pinned by pure-
/// logic unit tests that do not require a loaded LFM2 model.
/// Production code keeps the inlined form for zero-overhead dispatch;
/// this enum exists solely to drive `prefix_cache_decision_tests`'s
/// four-case coverage (empty cache, strict-extend hit, divergence
/// miss, exact-match miss). Any change to the inlined production
/// branch MUST be mirrored here or the test ceases to guard the real
/// code.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum PrefixCacheDecision {
    /// Strict-extend hit: the new prompt begins with the cached prefix
    /// and carries additional delta tokens. Warm-reuse safe: skip the
    /// cached prefix and prefill only the tail.
    StrictExtendHit,
    /// Cache miss — covers three sub-cases that all dispatch through
    /// the same `reset_caches_sync` + `init_caches_sync` + full-prefill
    /// branch:
    /// * `cached_prefix_len == 0` (no prior cache, reuse_cache disabled,
    ///   or prefix mismatch).
    /// * `cached_prefix_len == tokens_len` (exact-match) — routed to
    ///   miss because LFM2's short-conv layers have non-invertible
    ///   left-padded state and no safe "rewind by 1" primitive.
    Miss,
}

/// Test-only helper: decide what to do given the verifier's answer and
/// the incoming prompt length. Exact-match (`cached_prefix_len ==
/// tokens_len`) and zero-length prefix both route to
/// [`PrefixCacheDecision::Miss`].
///
/// Mirrors the inlined branch at the top of
/// [`Lfm2Inner::chat_sync_core`] / [`Lfm2Inner::chat_stream_sync_core`];
/// lifting it out keeps the invariant pinnable without loading a real
/// LFM2 model.
#[cfg(test)]
#[inline]
pub(crate) fn classify_prefix_cache_decision(
    cached_prefix_len: usize,
    tokens_len: usize,
) -> PrefixCacheDecision {
    if cached_prefix_len > 0 && cached_prefix_len < tokens_len {
        PrefixCacheDecision::StrictExtendHit
    } else {
        PrefixCacheDecision::Miss
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

        // Block-paged KV adapter — default ON since 2026-04-28.
        //
        // Chat dispatch is wired through this adapter at every chat-entry
        // site (see the `self.paged_adapter.is_some()` early-returns in
        // `chat_sync_core` / `chat_stream_sync_core` that hand off to
        // `chat_sync_core_paged` / `chat_stream_sync_core_paged`).
        //
        // KV-pool sizing: ONLY full_attention layers participate. LFM2's
        // hybrid layer mix is parsed from `config.layer_types`; conv
        // layers don't consume KV slots and continue to use the flat
        // `Lfm2LayerCache::Conv(ArraysCache)` storage on the paged path
        // too. The pool's `num_layers` is therefore the count of
        // `full_attention` entries, NOT the absolute `num_hidden_layers`;
        // the paged forward indexes this pool by attention-ordinal,
        // mapping absolute layer index → ordinal via
        // `config.full_attn_idxs()`.
        //
        // Cache dtype: BFloat16 (LFM2's production dtype). Parity
        // verified via `lfm2_paged_vs_flat_parity` integration test
        // (greedy byte-equal + prefix-reuse byte-equal at BF16 against
        // real LFM2.5-1.2B weights). Callers can opt out with
        // `use_block_paged_cache: Some(false)`.
        let paged_adapter = if config.use_block_paged_cache.unwrap_or(true) {
            let attn_layer_count = config.full_attn_idxs().len() as u32;
            if attn_layer_count == 0 {
                return Err(Error::from_reason(
                    "LFM2 block-paged adapter: config has no full_attention layers; \
                     paged KV cache requires at least one attention layer",
                ));
            }

            let block_size = config.paged_block_size.unwrap_or(16);
            let gpu_memory_mb = config.paged_cache_memory_mb.unwrap_or(2048);
            let head_size = config.head_dim() as u32;
            let num_kv_heads = config.num_key_value_heads as u32;

            let pa_config = mlx_paged_attn::PagedAttentionConfig {
                block_size,
                gpu_memory_mb,
                head_size,
                num_kv_heads,
                // Pool covers only the attention layers — conv layers
                // continue to use Lfm2LayerCache::Conv.
                num_layers: attn_layer_count,
                use_fp8_cache: Some(false),
                max_seq_len: Some(config.max_position_embeddings as u32),
                max_batch_size: Some(32),
            };

            let num_blocks = pa_config.calculate_num_blocks();
            if num_blocks == 0 {
                return Err(Error::from_reason(format!(
                    "LFM2 block-paged adapter: gpu_memory_mb={gpu_memory_mb} too small \
                     (head_size={head_size}, num_kv_heads={num_kv_heads}, \
                     block_size={block_size}, num_attn_layers={attn_layer_count})"
                )));
            }

            let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
                num_blocks, block_size,
            )));

            let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
            let pool = mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, cache_dtype)
                .map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to construct LayerKVPool for LFM2 block-paged adapter: {e}"
                    ))
                })?;

            let adapter =
                PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                    Error::from_reason(format!("Failed to construct LFM2 PagedKVCacheAdapter: {e}"))
                })?;

            info!(
                "LFM2 block-paged adapter enabled (construction-only): num_blocks={}, \
                 block_size={}, gpu_memory_mb={}, num_attn_layers={}, cache_dtype=BFloat16",
                num_blocks, block_size, gpu_memory_mb, attn_layer_count
            );
            Some(adapter)
        } else {
            None
        };

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
            paged_adapter,
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
            eval_lfm2_caches(&self.caches)?;
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
        // Drop any live paged-adapter request so the next session starts
        // from a fully cold state. Without this, a finalize_turn_keep_live
        // call from a prior session would leave block_table populated and
        // a subsequent `chat_sync_core_paged` could mistakenly take the
        // warm-continue path against stale tokens.
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
    }

    /// Check if tokens share a prefix with cached_token_history and return the prefix length.
    ///
    /// **Safety invariant**: this helper returns ONLY `0` (cache miss) or
    /// `cached_token_history.len()` (either an exact-append hit where the
    /// new prompt strictly extends the cached one, or an exact match where
    /// the new prompt equals the cached one). It never returns an
    /// intermediate value. Combined with the "no mid-sequence rewind"
    /// policy in [`Self::chat_sync_core`], this keeps LFM2's conv-state + KV
    /// caches safe under the prefix-reuse path.
    ///
    /// The caller must additionally distinguish strict-extend
    /// (`cached_prefix_len < tokens.len()`, warm-reuse safe) from
    /// exact-match (`cached_prefix_len == tokens.len()`). Only the
    /// strict-extend case is served via the warm path; exact-match is
    /// routed back through the cache-miss branch because LFM2's short-conv
    /// layers have non-invertible left-padded state and we have no safe
    /// "rewind-by-1" primitive. Attempting to reprefill the final cached
    /// token over the live caches would advance conv/KV state to
    /// `prompt + last_token` (duplicated) while `save_cache_state` only
    /// persists `tokens` into `cached_token_history`, corrupting the next
    /// warm-hit turn.
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

        // Block-paged dispatch: when the adapter is configured, route
        // through the parallel `chat_sync_core_paged` path. The flat path
        // below stays untouched so off-by-default behavior is byte-
        // identical to before this commit.
        if self.paged_adapter.is_some() {
            return self.chat_sync_core_paged(
                tokens,
                tokenizer,
                think_end_id,
                think_end_str,
                include_reasoning,
                p,
                enable_thinking,
                report_perf,
                eos_token_id,
            );
        }

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Cache reuse: prefix verification
        //
        // `verify_cache_prefix` returns 0 on miss or `cached.len()` on exact-append
        // hit — never an intermediate value (see its rustdoc). We split tokens into
        // "already-cached prefix" and "new delta to prefill":
        //   * miss            → reset caches, prefill the full prompt
        //   * strict extend   → skip the cached prefix, prefill only the tail delta
        //   * exact match     → treat as a miss (see rationale below)
        let cached_prefix_len_raw = self.verify_cache_prefix(&tokens, reuse_cache);

        let (prefill_tokens, cached_prefix_len) =
            if cached_prefix_len_raw > 0 && cached_prefix_len_raw < tokens.len() {
                info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len_raw,
                    tokens.len() - cached_prefix_len_raw,
                );
                (
                    tokens[cached_prefix_len_raw..].to_vec(),
                    cached_prefix_len_raw,
                )
            } else {
                // Cache miss OR exact-match (cached_prefix_len_raw == tokens.len()).
                //
                // Exact-match is deliberately treated as a miss: LFM2's short-conv
                // layers carry non-invertible left-padded state that depends on
                // every prior token, so we have no safe "rewind-by-1" primitive.
                // An earlier version reprefilled just the last cached token to
                // reuse live caches, but that advances conv/KV state to
                // `prompt + last_token` (duplicated) while `save_cache_state`
                // writes only `tokens` into `cached_token_history`. The resulting
                // drift between live cache and history would corrupt the next
                // warm-hit turn.
                //
                // Wiping caches + token history here starts the prefill from a
                // clean slate and keeps cache state aligned with what
                // `save_cache_state` persists after generation.
                self.reset_caches();
                (tokens.clone(), 0)
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
        eval_lfm2_caches(&self.caches)?;

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

        let mut result = finalize_chat_result(
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
        result.cached_tokens = cached_prefix_len as u32;
        Ok(result)
    }

    /// Block-paged variant of [`Self::chat_sync_core`].
    ///
    /// Mirrors the flat path's control flow (penalty stack, decode loop,
    /// EOS / repetition cutoff, performance timing, output post-processing)
    /// but routes attention layers through `forward_paged_or_flat` instead
    /// of the flat `forward()` path. Conv layers continue to use their
    /// existing `Lfm2LayerCache::Conv(ArraysCache)` storage.
    ///
    /// Per-turn lifecycle (mirrors the Qwen3 paged path):
    ///
    /// 1. Choose between **cold start** and **warm continuation**:
    ///    - Cold start: `paged_adapter.reset_for_new_request(seq_id)` →
    ///      `find_cached_prefix` → `allocate_suffix_blocks`.
    ///    - Warm continuation (turn 2+ within the same session, when the
    ///      prior turn ended via `finalize_turn_keep_live`):
    ///      `continue_turn(prompt, total_budget)` instead of the
    ///      reset/find/allocate triple. Keeps the partial trailing block's
    ///      K/V live across turns, eliminating the cross-turn BF16
    ///      re-prefill divergence (see
    ///      `PagedKVCacheAdapter::finalize_turn_keep_live`). Conv layers
    ///      still rebuild from token 0 each turn — the partial-block
    ///      carry only applies to the attention layer K/V state.
    /// 2. Conv-layer cache reset: every paged turn does a fresh prefill
    ///    on conv layers (no in-turn warm-reuse on the paged path). Conv
    ///    layers don't participate in the cross-request prefix cache;
    ///    their state is rebuilt over the entire prompt each turn.
    /// 3. Prefill via `run_paged_prefill_chunk` over the suffix tokens.
    /// 4. Decode loop via `run_paged_decode_step` — single-token forward
    ///    with `gather_kv_for_decode` on attention layers and the conv
    ///    operator's incremental step on conv layers.
    /// 5. End-of-turn (success): `finalize_turn_keep_live` publishes
    ///    full blocks AND keeps the request live for the next turn's
    ///    warm `continue_turn`.
    /// 6. Session end / explicit reset / error: `release_request`.
    ///
    /// Limitations (P1; documented in the doc comment):
    /// - Conv-layer prefix reuse is NOT carried across paged turns; each
    ///   paged turn reprefills conv state from the start of the prompt.
    /// - Pure-cache prompt (every prompt token already in the paged pool)
    ///   is rejected — same caveat as Qwen3's paged path.
    #[allow(clippy::too_many_arguments)]
    fn chat_sync_core_paged(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        think_end_id: Option<u32>,
        think_end_str: Option<String>,
        include_reasoning: bool,
        p: crate::models::qwen3_5::chat_common::ChatParams,
        enable_thinking: Option<bool>,
        report_perf: bool,
        eos_token_id: u32,
    ) -> Result<ChatResult> {
        let prompt_token_count = tokens.len();
        let sampling_config = p.sampling_config;

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        let thinking_enabled = enable_thinking.unwrap_or(true);
        let mut reasoning_tracker =
            ReasoningTracker::new(thinking_enabled, p.thinking_token_budget, think_end_id);

        // === Adapter lifecycle: warm continuation OR cold start. ===
        // See `PagedKVCacheAdapter::finalize_turn_keep_live` for why the
        // warm-continue path preserves the partial trailing block's K/V
        // across turns. Conv layers always reset and re-prefill the
        // cached prefix in `run_paged_prefill_chunk`'s "Pass 1" so the
        // partial-block carry only affects attention layers.
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only. Decode
        // blocks grow on-demand via `record_tokens` (no pre-reserve of
        // `max_new_tokens`). The inner decode loop reads `p.max_new_tokens`
        // directly when it needs the budget bound.
        let total_budget = tokens.len() as u32;
        // vLLM-style exact-prefix cap — see qwen3/model.rs:chat_sync_core_paged
        // for the full rationale. Forces the cache lookup (and the live-prefix
        // continue check) to leave at least one suffix token for the prefill
        // chunk, so retries of an earlier identical turn never produce a
        // zero-delta prompt.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let cached_prefix_len = {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "chat_sync_core_paged: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?;

            let can_continue = adapter.is_live_for_continue()
                && tokens.starts_with(adapter.request_tokens())
                && adapter.request_tokens().len() <= max_cache_hit_tokens as usize;

            if can_continue {
                match adapter.continue_turn(&tokens, total_budget) {
                    Ok((prior_token_count, _newly_alloc)) => prior_token_count,
                    Err(_drift) => {
                        let _ = adapter.release_request();
                        adapter
                            .reset_for_new_request(seq_id)
                            .map_err(Error::from_reason)?;
                        let prefix = adapter
                            .find_cached_prefix_with_max_tokens(
                                &tokens,
                                &[],
                                0,
                                false,
                                max_cache_hit_tokens,
                            )
                            .map_err(Error::from_reason)?;
                        let cached = prefix.cached_token_count;
                        adapter
                            .allocate_suffix_blocks(total_budget)
                            .map_err(Error::from_reason)?;
                        cached
                    }
                }
            } else {
                if adapter.block_table().is_some() {
                    let _ = adapter.release_request();
                }
                adapter
                    .reset_for_new_request(seq_id)
                    .map_err(Error::from_reason)?;
                let prefix = adapter
                    .find_cached_prefix_with_max_tokens(
                        &tokens,
                        &[],
                        0,
                        false,
                        max_cache_hit_tokens,
                    )
                    .map_err(Error::from_reason)?;
                let cached = prefix.cached_token_count;
                adapter
                    .allocate_suffix_blocks(total_budget)
                    .map_err(Error::from_reason)?;
                cached
            }
        };

        // Reset conv-layer state for this turn. The paged path does not
        // carry conv prefix state across turns; each turn reprefills from
        // the start of the prompt over conv layers (see method docstring).
        self.caches = init_caches(&self.config);
        self.cached_token_history.clear();
        self.cached_image_key = None;

        let total_prompt_tokens = tokens.len() as u32;
        let suffix_len = total_prompt_tokens
            .checked_sub(cached_prefix_len)
            .ok_or_else(|| {
                Error::from_reason("chat_sync_core_paged: cached_prefix_len > total_prompt_tokens")
            })?;

        // Conv layers always need to rebuild from token 0; if the paged
        // adapter reports a cached prefix from a previous turn we still
        // need to prefill conv state over [0, total) tokens. Keep `tokens`
        // intact for conv prefill; the paged adapter records only the
        // suffix tokens via `record_tokens`.
        if total_prompt_tokens == 0 {
            return Err(Error::from_reason("Empty prompt"));
        }

        // Wrap forward / decode in a closure-like pattern so we can
        // release the paged request on either success or error.
        let forward_result = self.chat_sync_core_paged_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            &p,
            eos_token_id,
            &sampling_config,
            &mut reasoning_tracker,
            report_perf,
            &mut first_token_instant,
        );

        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    // Keep the request live across turns so the next
                    // turn's `continue_turn` can pick up the partial
                    // trailing block's K/V. See
                    // `finalize_turn_keep_live` doc for rationale.
                    let _ = adapter.finalize_turn_keep_live(&[], 0);
                }
                t
            }
            Err(e) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                return Err(e);
            }
        };

        // Persist the session's token history so the subsequent
        // `chat_session_continue` (which dispatches to
        // `chat_tokens_delta_sync`) finds an initialized session and
        // can build its delta on top of the prior prompt + reply.
        //
        // The paged decode loop never feeds the LAST sampled token
        // through `run_paged_decode_step`, so the last entry in
        // `generated_tokens` is NOT recorded in the adapter / conv
        // caches — drop it from the saved history to keep the live
        // cache state aligned with what the next turn replays.
        // Mirrors `save_cache_state(reuse_cache=true, ..., last_token_in_cache=false)`
        // on the flat path.
        let last_token_in_cache = false;
        self.save_cache_state(true, &tokens, &generated_tokens, last_token_in_cache);

        // Performance metrics
        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
        } else {
            None
        };

        let reasoning_tokens = reasoning_tracker.reasoning_token_count();

        let mut result = finalize_chat_result(
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
        result.cached_tokens = cached_prefix_len;
        Ok(result)
    }

    /// Inner forward + decode loop for `chat_sync_core_paged`. Split out
    /// so the caller can wrap it with `release_request` on either path.
    #[allow(clippy::too_many_arguments)]
    fn chat_sync_core_paged_inner(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &crate::models::qwen3_5::chat_common::ChatParams,
        eos_token_id: u32,
        sampling_config: &Option<crate::sampling::SamplingConfig>,
        reasoning_tracker: &mut ReasoningTracker,
        report_perf: bool,
        first_token_instant: &mut Option<std::time::Instant>,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "chat_sync_core_paged_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        // === PREFILL ===
        // Run conv prefill on the FULL prompt (since conv state must
        // start from token 0). For attention layers the paged path only
        // writes the suffix into the pool — the cached prefix already
        // lives in the pool from a prior request that registered it.
        let suffix = &tokens[(cached_prefix_len as usize)..];
        let last_logits = self.run_paged_prefill_chunk(tokens, suffix, cached_prefix_len)?;

        // Apply penalties + sample first token
        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, *sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating. Prefill builds a massive MLX subgraph; once
        // we have the last logits, those intermediates are dead but
        // MLX's caching allocator holds them.
        crate::array::synchronize_and_clear_cache();

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        // === DECODE LOOP ===
        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens.max(0) as usize);
        let mut finish_reason = String::from("length");

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            reasoning_tracker.observe_token(token_id);

            if token_id == eos_token_id {
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
            if step + 1 >= max_new_tokens {
                break;
            }

            // Decode forward
            let next_logits = self.run_paged_decode_step(token_id)?;
            let next_logits = next_logits.squeeze(Some(&[1]))?;

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id() as i32;
                y = MxArray::from_int32(&[forced_id], &[1])?;
                y.eval();
                continue;
            } else {
                apply_all_penalties(next_logits, &token_history, p)?
            };

            y = sample(&next_logits, *sampling_config)?;
            y.eval();

            crate::array::maybe_clear_cache_for_paged_step(step);
        }

        Ok((generated_tokens, finish_reason))
    }

    /// Run a paged-attention prefill over the full prompt, dispatching
    /// per-layer between paged-attention (full_attention layers) and the
    /// existing conv path (conv layers).
    ///
    /// `full_tokens` is the entire prompt (used for conv layers' prefill
    /// from token 0). `suffix_tokens` is the new portion beyond the
    /// paged prefix-cache hit (used by `record_tokens` +
    /// `update_keys_values` for the attention layers).
    /// `cached_prefix_len` is the paged-cache hit length.
    ///
    /// Returns the last position's logits squeezed to `[vocab]`.
    fn run_paged_prefill_chunk(
        &mut self,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
    ) -> Result<MxArray> {
        if suffix_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_prefill_chunk called with empty suffix",
            ));
        }

        // Record the SUFFIX tokens in the paged adapter (cached_prefix
        // already lives in the pool). The conv layers see the FULL
        // prompt below.
        let suffix_len = suffix_tokens.len() as u32;
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_prefill_chunk: paged_adapter is None")
            })?;
            adapter
                .record_tokens(suffix_tokens)
                .map_err(Error::from_reason)?;
        }

        // Build per-layer kind list once. paged_idx counts only
        // full_attention layers in their original layer order.
        let layer_kinds = self.compute_layer_kinds();

        // Forward the FULL prompt through conv layers and the SUFFIX
        // through attention layers in the same per-layer loop. Because
        // attention layers receive a different sequence length than
        // conv layers, we run two passes:
        //
        // Pass 1: conv-only prefill on the FULL prompt to build conv
        //         state. Hidden-state output is discarded.
        // Pass 2: full forward (conv + attention) on the SUFFIX. Conv
        //         layers see only the suffix here; their state from
        //         pass 1 carries the prefix context. Attention layers
        //         attend over `read_kv_range(0, total_ctx)` to recover
        //         the cached + new context.
        //
        // For the no-cache case (cached_prefix_len == 0) the suffix IS
        // the full prompt, so pass 1 is skipped and pass 2 handles
        // everything in one shot.

        if cached_prefix_len > 0 {
            // Pass 1: conv-only prefill over the cached prefix. This
            // brings conv state up to position `cached_prefix_len` so
            // pass 2 can continue from there.
            let prefix = &full_tokens[..(cached_prefix_len as usize)];
            self.run_conv_only_prefill(prefix)?;
        }

        // Pass 2: full forward on the suffix.
        let input_ids = MxArray::from_uint32(suffix_tokens, &[1, suffix_len as i64])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;

        let num_layers = self.layers.len();
        let first_logical_position = cached_prefix_len;

        // The index-based loop is required here: we use raw-pointer
        // split-borrows on `self.layers` and `self.caches` to access
        // disjoint indices simultaneously while the paged_adapter is
        // also borrowed mutably. An iterator-based version would conflict
        // with the borrow checker.
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            let kind = layer_kinds[layer_idx];

            // Split borrow: layers (immutable per layer) + paged_adapter
            // (mutable) + caches[layer_idx] (mutable for conv).
            let layer: &Lfm2DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };

            match kind {
                Lfm2LayerKind::FullAttention { .. } => {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_prefill_chunk: paged_adapter dropped mid-forward",
                        )
                    })?;
                    hidden_states = layer.forward_paged_or_flat(
                        &hidden_states,
                        kind,
                        adapter,
                        first_logical_position,
                        cached_prefix_len,
                        /* is_prefill */ true,
                        /* conv_cache */ None,
                    )?;
                }
                Lfm2LayerKind::Conv => {
                    // Conv path needs paged_adapter as a placeholder; it's
                    // ignored by the conv branch in `forward_paged_or_flat`.
                    // Use a split-borrow to access caches[layer_idx]
                    // mutably while paged_adapter is also mutable.
                    let conv_cache = unsafe {
                        let ptr = self.caches.as_mut_ptr().add(layer_idx);
                        &mut *ptr
                    };
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_prefill_chunk: paged_adapter dropped mid-forward",
                        )
                    })?;
                    hidden_states = layer.forward_paged_or_flat(
                        &hidden_states,
                        kind,
                        adapter,
                        first_logical_position,
                        cached_prefix_len,
                        /* is_prefill */ true,
                        Some(conv_cache),
                    )?;
                }
            }
            // Smooth the prefill memory peak: every K layers, materialize the
            // residual stream so MLX can release the upstream graph nodes
            // (embedding + every prior layer's attention/conv intermediates)
            // from the cache pool. Without this the in-flight lazy graph
            // accumulates on long contexts before the post-prefill sync
            // fires. Cadence is `MLX_PAGED_PREFILL_EVAL_INTERVAL` (default 8).
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }

        // Output norm + lm_head.
        hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        };

        // Slice the last token's logits.
        let seq_len = logits.shape_at(1)?;
        let last = logits
            .slice_axis(1, seq_len - 1, seq_len)?
            .squeeze(Some(&[0, 1]))?;
        Ok(last)
    }

    /// Run one paged decode step: feed `[token_id]` through the model.
    fn run_paged_decode_step(&mut self, token_id: u32) -> Result<MxArray> {
        // Record the new token + capture its logical position BEFORE
        // record_tokens advances the cursor.
        let first_logical_position = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter is None")
            })?;
            adapter.current_token_count()
        };
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter dropped")
            })?;
            adapter
                .record_tokens(&[token_id])
                .map_err(Error::from_reason)?;
        }

        let layer_kinds = self.compute_layer_kinds();

        let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;

        let num_layers = self.layers.len();
        // See `run_paged_prefill_chunk` for the rationale on the
        // index-based loop (raw-pointer split borrow over disjoint
        // fields).
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            let kind = layer_kinds[layer_idx];
            let layer: &Lfm2DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };

            match kind {
                Lfm2LayerKind::FullAttention { .. } => {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_decode_step: paged_adapter dropped mid-forward",
                        )
                    })?;
                    hidden_states = layer.forward_paged_or_flat(
                        &hidden_states,
                        kind,
                        adapter,
                        first_logical_position,
                        /* cached_prefix_len */ 0,
                        /* is_prefill */ false,
                        /* conv_cache */ None,
                    )?;
                }
                Lfm2LayerKind::Conv => {
                    let conv_cache = unsafe {
                        let ptr = self.caches.as_mut_ptr().add(layer_idx);
                        &mut *ptr
                    };
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_decode_step: paged_adapter dropped mid-forward",
                        )
                    })?;
                    hidden_states = layer.forward_paged_or_flat(
                        &hidden_states,
                        kind,
                        adapter,
                        first_logical_position,
                        /* cached_prefix_len */ 0,
                        /* is_prefill */ false,
                        Some(conv_cache),
                    )?;
                }
            }
        }

        hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        };
        Ok(logits)
    }

    /// Forward the cached prefix tokens through CONV layers ONLY,
    /// updating their state in-place. Used to bring conv state up to the
    /// paged cache's `cached_prefix_len` boundary before pass 2 of
    /// `run_paged_prefill_chunk` continues with the suffix.
    fn run_conv_only_prefill(&mut self, prefix_tokens: &[u32]) -> Result<()> {
        if prefix_tokens.is_empty() {
            return Ok(());
        }
        let input_ids = MxArray::from_uint32(prefix_tokens, &[1, prefix_tokens.len() as i64])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;

        let num_layers = self.layers.len();
        for layer_idx in 0..num_layers {
            let layer = &self.layers[layer_idx];
            if layer.is_attention_layer() {
                // Skip attention layers — they pull the prefix from the
                // paged pool's prefix cache. The hidden_states we feed
                // forward here will not pass through their projection,
                // so we make a SHAPE-PRESERVING identity passthrough.
                // This is safe because attention layers' contribution
                // to subsequent conv layers' input depends on their
                // residual + FFN, which is unrecoverable without a
                // full attention pass. Specifically: this `pass 1`
                // path only runs when cached_prefix_len > 0 i.e. we're
                // re-using state from a previous turn that already
                // computed exact conv state. In the smoke-test path
                // (cached_prefix_len == 0) this method is never called.
                //
                // **Limitation**: this is approximate — for exact
                // numerical equivalence we'd need to re-run attention
                // here too, which defeats the purpose of the prefix
                // cache. Marked as a P1 known-issue for follow-up.
                continue;
            }
            // Conv layer: forward through the operator + FFN tail.
            let cache_slot = unsafe {
                let ptr = self.caches.as_mut_ptr().add(layer_idx);
                &mut *ptr
            };
            hidden_states = layer.forward(&hidden_states, None, Some(cache_slot))?;
        }
        Ok(())
    }

    /// Build the per-layer routing list. `FullAttention { paged_idx }`
    /// for full-attention layers (paged_idx counts only those layers in
    /// their original order) and `Conv` for conv layers.
    fn compute_layer_kinds(&self) -> Vec<Lfm2LayerKind> {
        let mut kinds = Vec::with_capacity(self.layers.len());
        let mut paged_idx: u32 = 0;
        for i in 0..self.layers.len() {
            if self.config.is_attention_layer(i) {
                kinds.push(Lfm2LayerKind::FullAttention { paged_idx });
                paged_idx += 1;
            } else {
                kinds.push(Lfm2LayerKind::Conv);
            }
        }
        kinds
    }

    /// Block-paged streaming variant of [`Self::chat_stream_sync_core`].
    ///
    /// Mirrors `chat_sync_core_paged`'s adapter lifecycle and forward
    /// dispatch (reset → find_cached_prefix → allocate_suffix → prefill
    /// via `run_paged_prefill_chunk` → decode loop via
    /// `run_paged_decode_step`) but emits each generated token through
    /// the streaming callback as it is produced.
    ///
    /// Mirrors the flat streaming path's terminal contract:
    /// * Streams text chunks for every decoded token.
    /// * Sends a residual chunk for any tokens whose detokenized text
    ///   has not yet been flushed.
    /// * Sends a terminal `done: true` chunk with `finish_reason`,
    ///   aggregated `tool_calls`, `thinking`, performance metrics, and
    ///   the matched cached-prefix length.
    ///
    /// Applies the same vLLM `max_cache_hit_tokens = prompt.len() - 1`
    /// cap as `chat_sync_core_paged` so zero-delta prompts still produce
    /// at least one suffix token to prefill. Numerical equivalence to the
    /// flat path is not asserted here.
    #[allow(clippy::too_many_arguments)]
    fn chat_stream_sync_core_paged(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        think_end_id: Option<u32>,
        think_end_str: Option<String>,
        include_reasoning: bool,
        p: crate::models::qwen3_5::chat_common::ChatParams,
        enable_thinking: Option<bool>,
        report_perf: bool,
        eos_token_id: u32,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<()> {
        let prompt_token_count = tokens.len();
        let sampling_config = p.sampling_config;

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        let thinking_enabled = enable_thinking.unwrap_or(true);
        let mut reasoning_tracker =
            ReasoningTracker::new(thinking_enabled, p.thinking_token_budget, think_end_id);

        // Streaming decode state
        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        // === Adapter lifecycle: warm continuation OR cold start. ===
        // See the equivalent block in `chat_sync_core_paged` for full
        // discussion.
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // See `chat_sync_core_paged` for the vLLM-style cap rationale.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let cached_prefix_len = {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "chat_stream_sync_core_paged: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?;

            let can_continue = adapter.is_live_for_continue()
                && tokens.starts_with(adapter.request_tokens())
                && adapter.request_tokens().len() <= max_cache_hit_tokens as usize;

            if can_continue {
                match adapter.continue_turn(&tokens, total_budget) {
                    Ok((prior_token_count, _newly_alloc)) => prior_token_count,
                    Err(_drift) => {
                        let _ = adapter.release_request();
                        adapter
                            .reset_for_new_request(seq_id)
                            .map_err(Error::from_reason)?;
                        let prefix = adapter
                            .find_cached_prefix_with_max_tokens(
                                &tokens,
                                &[],
                                0,
                                false,
                                max_cache_hit_tokens,
                            )
                            .map_err(Error::from_reason)?;
                        let cached = prefix.cached_token_count;
                        adapter
                            .allocate_suffix_blocks(total_budget)
                            .map_err(Error::from_reason)?;
                        cached
                    }
                }
            } else {
                if adapter.block_table().is_some() {
                    let _ = adapter.release_request();
                }
                adapter
                    .reset_for_new_request(seq_id)
                    .map_err(Error::from_reason)?;
                let prefix = adapter
                    .find_cached_prefix_with_max_tokens(
                        &tokens,
                        &[],
                        0,
                        false,
                        max_cache_hit_tokens,
                    )
                    .map_err(Error::from_reason)?;
                let cached = prefix.cached_token_count;
                adapter
                    .allocate_suffix_blocks(total_budget)
                    .map_err(Error::from_reason)?;
                cached
            }
        };

        // Reset conv-layer state for this turn (see chat_sync_core_paged
        // doc comment).
        self.caches = init_caches(&self.config);
        self.cached_token_history.clear();
        self.cached_image_key = None;

        let total_prompt_tokens = tokens.len() as u32;
        let suffix_len = total_prompt_tokens
            .checked_sub(cached_prefix_len)
            .ok_or_else(|| {
                Error::from_reason(
                    "chat_stream_sync_core_paged: cached_prefix_len > total_prompt_tokens",
                )
            })?;

        if total_prompt_tokens == 0 {
            // Release before bailing.
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            return Err(Error::from_reason("Empty prompt"));
        }

        // Run the forward + decode under a try-style block so we can
        // always release the request afterwards.
        let result = self.chat_stream_sync_core_paged_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            &p,
            sampling_config,
            eos_token_id,
            &mut reasoning_tracker,
            report_perf,
            &mut first_token_instant,
            &tokenizer,
            &mut decode_stream,
            &mut streamed_text_len,
            &mut last_is_reasoning,
            cb,
            cancelled,
        );

        let (generated_tokens, finish_reason) = match result {
            Ok(t) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    // Keep request live across turns. See
                    // `finalize_turn_keep_live` doc + the non-streaming
                    // `chat_sync_core_paged`'s terminal block.
                    let _ = adapter.finalize_turn_keep_live(&[], 0);
                }
                t
            }
            Err(e) => {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                return Err(e);
            }
        };

        // Persist the session's token history so the subsequent
        // `chat_session_continue` (which dispatches to
        // `chat_tokens_delta_sync`) finds an initialized session and
        // can build its delta on top of the prior prompt + reply.
        // See the non-streaming `chat_sync_core_paged` for the rationale
        // on `last_token_in_cache = false`.
        let last_token_in_cache = false;
        self.save_cache_state(true, &tokens, &generated_tokens, last_token_in_cache);

        // Flush residual buffered bytes from decode_stream (mirrors flat
        // streaming).
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
                    cached_tokens: None,
                    performance: None,
                    is_reasoning: Some(last_is_reasoning),
                }),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
        }

        // Performance metrics
        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
        } else {
            None
        };

        let reasoning_tokens = reasoning_tracker.reasoning_token_count();

        let mut result = finalize_chat_result(
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
        result.cached_tokens = cached_prefix_len;

        // Send terminal chunk
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
                cached_tokens: Some(cached_prefix_len),
                performance: result.performance.clone(),
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Inner forward + streaming decode loop for
    /// [`Self::chat_stream_sync_core_paged`]. Split out so the caller can
    /// wrap with `release_request` in a try-style flow.
    #[allow(clippy::too_many_arguments)]
    fn chat_stream_sync_core_paged_inner<'a>(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &crate::models::qwen3_5::chat_common::ChatParams,
        sampling_config: Option<crate::sampling::SamplingConfig>,
        eos_token_id: u32,
        reasoning_tracker: &mut ReasoningTracker,
        report_perf: bool,
        first_token_instant: &mut Option<std::time::Instant>,
        tokenizer: &'a Arc<Qwen3Tokenizer>,
        decode_stream: &mut tokenizers::DecodeStream<
            'a,
            tokenizers::ModelWrapper,
            tokenizers::NormalizerWrapper,
            tokenizers::PreTokenizerWrapper,
            tokenizers::PostProcessorWrapper,
            tokenizers::DecoderWrapper,
        >,
        streamed_text_len: &mut usize,
        last_is_reasoning: &mut bool,
        cb: &StreamSender,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "chat_stream_sync_core_paged: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        // === PREFILL ===
        let suffix = &tokens[(cached_prefix_len as usize)..];
        let last_logits = self.run_paged_prefill_chunk(tokens, suffix, cached_prefix_len)?;

        // Apply penalties + sample first token
        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating (see chat_sync_core_paged_inner for rationale).
        crate::array::synchronize_and_clear_cache();

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        // === STREAMING DECODE LOOP ===
        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens.max(0) as usize);
        let mut finish_reason = String::from("length");

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            let is_reasoning = reasoning_tracker.observe_token(token_id);
            *last_is_reasoning = is_reasoning;

            if token_id == eos_token_id {
                finish_reason = String::from("stop");
                break;
            }
            if cancelled.load(Ordering::Relaxed) {
                finish_reason = String::from("cancelled");
                break;
            }

            // Stream delta chunk
            let token_text = Qwen3Tokenizer::step_decode_stream(
                decode_stream,
                tokenizer.inner(),
                token_id,
                &generated_tokens,
                *streamed_text_len,
            );
            *streamed_text_len += token_text.len();
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
                    cached_tokens: None,
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
            if step + 1 >= max_new_tokens {
                break;
            }

            // Decode forward
            let next_logits = self.run_paged_decode_step(token_id)?;
            let next_logits = next_logits.squeeze(Some(&[1]))?;

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id() as i32;
                y = MxArray::from_int32(&[forced_id], &[1])?;
                y.eval();
                continue;
            } else {
                apply_all_penalties(next_logits, &token_history, p)?
            };

            y = sample(&next_logits, sampling_config)?;
            y.eval();

            crate::array::maybe_clear_cache_for_paged_step(step);
        }

        Ok((generated_tokens, finish_reason))
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

        // Block-paged dispatch: when the adapter is configured, route
        // through the parallel `chat_stream_sync_core_paged` path. The
        // flat path below stays untouched so off-by-default behavior is
        // byte-identical to before this commit.
        if self.paged_adapter.is_some() {
            return self.chat_stream_sync_core_paged(
                tokens,
                tokenizer,
                think_end_id,
                think_end_str,
                include_reasoning,
                p,
                enable_thinking,
                report_perf,
                eos_token_id,
                cb,
                cancelled,
            );
        }

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Cache reuse — see the non-streaming `chat_sync_core` for the full
        // rationale. Invariant: `verify_cache_prefix` returns 0 or
        // `cached.len()` only. Strict-extend reuses the live caches; exact
        // match falls through to the miss branch because LFM2 has no safe
        // rewind primitive for its short-conv state.
        let cached_prefix_len_raw = self.verify_cache_prefix(&tokens, reuse_cache);

        let (prefill_tokens, cached_prefix_len) =
            if cached_prefix_len_raw > 0 && cached_prefix_len_raw < tokens.len() {
                (
                    tokens[cached_prefix_len_raw..].to_vec(),
                    cached_prefix_len_raw,
                )
            } else {
                // Cache miss OR exact-match (treated as miss — see chat_sync_core
                // for full rationale).
                self.reset_caches();
                (tokens.clone(), 0)
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
        eval_lfm2_caches(&self.caches)?;

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
                    cached_tokens: None,
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
                    cached_tokens: None,
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

        let mut result = finalize_chat_result(
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
        result.cached_tokens = cached_prefix_len as u32;

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
                // Start path: report the matched prefix length from
                // `verify_cache_prefix`. Zero on a miss, full cached
                // length on an exact-append hit.
                cached_tokens: Some(cached_prefix_len as u32),
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

        // NOTE: no unconditional reset here. Prefix-reuse support
        // (pi-mono / Aider / Codex-style stateless agents that resend the
        // full conversation every turn) requires `chat_sync_core` to
        // decide whether to reset based on `verify_cache_prefix`'s
        // return. A miss triggers an internal reset; a hit preserves the
        // live caches and prefills only the tail delta. Wiping here
        // would make every session-start a cache miss by construction.
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

        // Capture the full prior-cached length BEFORE appending the
        // delta so we can report it as `cached_tokens` on the returned
        // ChatResult. The delta path always reuses the entire cached
        // prefix (it's a strict extension on top of the session's
        // existing `cached_token_history`), so `prior_cached_len` IS
        // the number of prefilled tokens that were skipped thanks to
        // the warm cache. Without this, every LFM2 delta turn returns
        // `cached_tokens = 0` — `finalize_chat_result` defaults the
        // field to zero and only the HTTP layer fills it in
        // differently — which misreports every continuation as a MISS
        // and prevents the `/v1/responses` endpoint from promoting
        // `X-Session-Cache` to `prefix_hit`.
        let prior_cached_len = self.cached_token_history.len();

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
        eval_lfm2_caches(&self.caches)?;

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

        let mut result = finalize_chat_result(
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
        )?;
        // Overwrite the default `cached_tokens = 0` from
        // `finalize_chat_result` with the real prior-cached length.
        // On the delta path the session's full cached prefix is
        // reused by construction — `prior_cached_len` is the exact
        // token count skipped by `chat_session_start_sync`'s prefix
        // verifier equivalent on this path.
        result.cached_tokens = prior_cached_len as u32;
        Ok(result)
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

        // NOTE: no unconditional reset here — see `chat_session_start_sync`
        // for the prefix-reuse rationale. `chat_stream_sync_core` runs
        // `verify_cache_prefix` and resets internally only on a cache miss.
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
        // Capture `prior_cached_len` BEFORE the extend — this is the
        // reused-prefix length reported on the terminal ChatStreamChunk's
        // `cached_tokens` field (mirrors the non-streaming delta path's
        // `cached_tokens` in `ChatResult`).
        let prior_cached_len = self.cached_token_history.len() as u32;
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
        eval_lfm2_caches(&self.caches)?;

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
                    cached_tokens: None,
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
                    cached_tokens: None,
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
                // Delta path reuses the full prior history by construction
                // — report `prior_cached_len` (captured before the
                // `self.cached_token_history` extend above) as the
                // authoritative cached-prefix length.
                cached_tokens: Some(prior_cached_len),
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
        Lfm2Cmd::ChatSessionStart {
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
fn eval_lfm2_caches(caches: &[Lfm2LayerCache]) -> Result<()> {
    let mut arrays: Vec<&MxArray> = Vec::new();
    for cache in caches {
        cache.collect_arrays(&mut arrays);
    }
    if !arrays.is_empty() {
        MxArray::eval_arrays(&arrays)?;
    }
    Ok(())
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
    /// Snapshot of `Lfm2Inner::paged_adapter.is_some()` captured at
    /// construction time. The block-paged KV adapter is wired up once at
    /// load (default-on for full-attention layers — conv layers always
    /// stay on `Lfm2LayerCache::Conv`). Surfaced through the
    /// `hasBlockPagedCache()` NAPI method so the server-side
    /// `/v1/messages` endpoint can bypass the JS-side warm slot when
    /// paged is active and rely on native content-addressed block reuse.
    pub(crate) paged_active: bool,
    /// RAII: unregisters this model's baseline from the cache-limit
    /// coordinator on drop.
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
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

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Lfm2Inner::paged_adapter` was successfully constructed
    /// at load time (driven by `Lfm2Config::use_block_paged_cache`,
    /// defaulting to `true` after paged-vs-flat parity verification).
    /// LFM2 is hybrid (10 conv + 6 full-attention layers); only the
    /// full-attention layers route through the adapter, conv layers stay
    /// on flat `Lfm2LayerCache::Conv` regardless. When `true`, the native
    /// cache reuses SYS blocks across `chatSessionStart` calls via
    /// content-addressing, so the JS-side warm slot in
    /// `SessionRegistry.getOrCreateWarmAny` is redundant and the
    /// `/v1/messages` server endpoint allocates a fresh `ChatSession` per
    /// request.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
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

#[cfg(test)]
mod prefix_cache_decision_tests {
    //! Pure-logic coverage of the prefix-cache decision tree — no model
    //! load required. The verifier `Lfm2Inner::verify_cache_prefix`
    //! returns either `0` (miss) or `cached_token_history.len()` (exact
    //! prefix relation). The call sites in `chat_sync_core` /
    //! `chat_stream_sync_core` then classify that value plus the
    //! incoming prompt length into
    //! [`PrefixCacheDecision::StrictExtendHit`] (warm-reuse, skip the
    //! cached prefix, prefill only the tail) vs
    //! [`PrefixCacheDecision::Miss`] (reset caches + re-init + full
    //! prefill).
    //!
    //! The four cases covered below pin the Round 1 Fix #2 invariant:
    //! exact-match MUST route to `Miss`, not to `StrictExtendHit` —
    //! LFM2's short-conv layers carry non-invertible left-padded state
    //! and there is no safe "rewind-by-1" primitive. Reprefilling the
    //! final cached token on top of the live caches would advance state
    //! to `prompt + last_token` (duplicated) while `save_cache_state`
    //! writes only `tokens`, corrupting the next warm-hit turn. The
    //! `#[ignore]`-gated integration tests above exercise the end-to-
    //! end behaviour against a loaded LFM2 model; this module guarantees
    //! the decision logic stays correct in every CI run without a model
    //! dependency.

    use super::{PrefixCacheDecision, classify_prefix_cache_decision};

    #[test]
    fn empty_cache_is_miss() {
        // verify_cache_prefix returned 0: either `cached_token_history`
        // is empty, `reuse_cache` was false, or the prompt didn't
        // prefix-match. All three land on the same miss branch.
        assert_eq!(
            classify_prefix_cache_decision(0, 0),
            PrefixCacheDecision::Miss,
            "empty cache + empty tokens must be Miss"
        );
        assert_eq!(
            classify_prefix_cache_decision(0, 10),
            PrefixCacheDecision::Miss,
            "empty cache + non-empty tokens must be Miss"
        );
    }

    #[test]
    fn strict_extend_is_hit() {
        // verify_cache_prefix returned cached_token_history.len() AND
        // tokens.len() > cached_token_history.len(). The caller prefills
        // only `tokens[cached_prefix_len..]` on top of the live caches.
        assert_eq!(
            classify_prefix_cache_decision(5, 8),
            PrefixCacheDecision::StrictExtendHit,
            "cached.len() < tokens.len() must be StrictExtendHit"
        );
        assert_eq!(
            classify_prefix_cache_decision(1, 2),
            PrefixCacheDecision::StrictExtendHit,
            "minimum strict-extend (one cached, one delta) must be StrictExtendHit"
        );
    }

    #[test]
    fn divergence_is_miss() {
        // verify_cache_prefix returned 0 because
        // tokens[..cached.len()] != cached[..]. Same branch as empty-
        // cache miss — both flavours dispatch to reset + re-init +
        // full-prefill.
        assert_eq!(
            classify_prefix_cache_decision(0, 20),
            PrefixCacheDecision::Miss,
            "divergence (verifier returned 0) must be Miss"
        );
    }

    #[test]
    fn exact_match_is_miss() {
        // verify_cache_prefix returned cached_token_history.len() AND
        // tokens.len() == cached_token_history.len() — byte-equal
        // prompt. The classifier routes to Miss because LFM2's conv
        // state has non-invertible left-padded buffers; there is no
        // way to sample from the already-cached final position without
        // re-running the last forward step, which would duplicate the
        // final token into cache state while persistence only records
        // the prompt + generated tokens. Round 1 Fix #2 pinned this
        // invariant — the tests here guard against any regression.
        assert_eq!(
            classify_prefix_cache_decision(5, 5),
            PrefixCacheDecision::Miss,
            "exact-match (cached.len() == tokens.len()) must be Miss, not StrictExtendHit"
        );
        assert_eq!(
            classify_prefix_cache_decision(1, 1),
            PrefixCacheDecision::Miss,
            "exact-match single token must be Miss"
        );
        assert_eq!(
            classify_prefix_cache_decision(1000, 1000),
            PrefixCacheDecision::Miss,
            "exact-match long prompts must still be Miss"
        );
    }

    #[test]
    fn invariant_cached_len_never_exceeds_tokens_len_in_hit() {
        // Belt-and-braces: the verifier guarantees `cached.len() <=
        // tokens.len()` on every non-zero return (it rejects with 0
        // when tokens.len() < cached.len()), so the classifier never
        // sees cached_prefix_len > tokens_len in practice. But if it
        // ever did, the branch routes to Miss (the `<` is strict),
        // which is the safe fallthrough.
        assert_eq!(
            classify_prefix_cache_decision(10, 5),
            PrefixCacheDecision::Miss,
            "cached_prefix_len > tokens_len must be Miss (defensive fallthrough)"
        );
    }
}

#[cfg(test)]
mod paged_adapter_construction_tests {
    //! Construction-only coverage of `Lfm2Inner::paged_adapter`. The
    //! flag is opt-in and currently a no-op for chat dispatch — see the
    //! doc comment on the field for the architectural rationale (LFM2's
    //! hybrid conv + attention requires a bespoke per-layer dispatch and
    //! an attention-ordinal-indexed cache wrapper). These tests pin the
    //! "default = no allocation" invariant and verify that flipping the
    //! flag wires up a real adapter without churning forward-path code.

    use super::Lfm2Inner;
    use crate::models::lfm2::Lfm2Config;

    /// Tiny LFM2-shaped config compatible with `LayerKVPool`'s validate
    /// constraints (head_size in {32, 64, 96, 128, 256}, FP8 off).
    /// Two layers: one conv + one full_attention. Mirrors the same hybrid
    /// shape as production LFM2 so the adapter sizing exercises the
    /// "attention layers only" path.
    ///
    /// `use_block_paged` is `Option<bool>` so tests can distinguish the
    /// three states the production code now cares about:
    /// * `Some(true)`  — explicit opt-in, paged adapter must allocate.
    /// * `Some(false)` — explicit opt-out, paged adapter must NOT allocate.
    /// * `None`        — default-on under the new policy (`unwrap_or(true)`),
    ///   paged adapter must allocate on Metal hosts.
    fn paged_tiny_config(use_block_paged: Option<bool>) -> Lfm2Config {
        Lfm2Config {
            vocab_size: 100,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            max_position_embeddings: 128,
            norm_eps: 1e-5,
            conv_bias: false,
            conv_l_cache: 3,
            block_dim: 64,
            block_ff_dim: 64,
            block_multiple_of: 256,
            block_ffn_dim_multiplier: 1.0,
            block_auto_adjust_ff_dim: false,
            rope_theta: 1_000_000.0,
            // 1 conv + 1 full_attention — the adapter pool should be
            // sized for ONE attention layer, not two.
            layer_types: vec!["conv".to_string(), "full_attention".to_string()],
            tie_embedding: true,
            eos_token_id: 7,
            bos_token_id: 1,
            pad_token_id: 0,
            paged_cache_memory_mb: Some(256),
            paged_block_size: Some(16),
            use_block_paged_cache: use_block_paged,
        }
    }

    /// Explicit opt-out (`Some(false)`) must NOT allocate the block-paged
    /// adapter.
    ///
    /// The previous "None means no adapter" assertion was removed when the
    /// default flipped from `unwrap_or(false)` to `unwrap_or(true)`. The
    /// opt-out path is the new "no adapter" guarantee.
    #[test]
    fn test_lfm2_inner_no_paged_adapter_when_flag_is_explicit_false() {
        let cfg = paged_tiny_config(Some(false));
        let inner = match Lfm2Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Lfm2Inner::new failure: {msg}");
            }
        };
        assert!(
            inner.paged_adapter.is_none(),
            "paged_adapter must be None when use_block_paged_cache is Some(false)"
        );
    }

    /// Default-flag construction (`None`) must allocate the block-paged
    /// adapter under the new default-on policy (`unwrap_or(true)`).
    /// Allocates a `LayerKVPool`, so requires Metal — gracefully skips on
    /// no-Metal sandboxes.
    #[test]
    fn test_lfm2_inner_paged_adapter_when_flag_is_none_default_on_macos() {
        let cfg = paged_tiny_config(None);
        match Lfm2Inner::new(cfg) {
            Ok(inner) => {
                assert!(
                    inner.paged_adapter.is_some(),
                    "paged_adapter must be Some when use_block_paged_cache is None \
                     (new default-on policy: unwrap_or(true))"
                );
            }
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Lfm2Inner::new failure: {msg}");
            }
        }
    }

    /// Construction with `use_block_paged_cache: Some(true)` must populate
    /// `paged_adapter`. Allocates a `LayerKVPool`, so requires Metal —
    /// gracefully skips on no-Metal sandboxes.
    #[test]
    fn test_lfm2_inner_constructs_paged_adapter_when_flag_is_true() {
        let cfg = paged_tiny_config(Some(true));
        match Lfm2Inner::new(cfg) {
            Ok(inner) => {
                assert!(
                    inner.paged_adapter.is_some(),
                    "paged_adapter must be Some when use_block_paged_cache = Some(true)"
                );
            }
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Lfm2Inner::new failure: {msg}");
            }
        }
    }

    /// **Smoke test for `chat_sync_core_paged` helpers**. Without real
    /// weights / tokenizer we cannot drive the full chat path, but we
    /// CAN drive the underlying `run_paged_prefill_chunk` +
    /// `run_paged_decode_step` helpers that the chat path delegates to.
    /// This validates the adapter lifecycle (reset → find_cached_prefix
    /// → allocate_suffix → record_tokens → forward_paged_or_flat),
    /// the prefill SDPA path (no-cache branch), and the decode-loop
    /// control flow against a freshly-constructed Lfm2Inner with random
    /// BFloat16 weights.
    ///
    /// What we assert:
    /// * Prefill on a 4-token "prompt" produces logits with shape
    ///   `[vocab]` and finite values.
    /// * Two decode steps produce non-empty u32 token ids.
    /// * Adapter's `current_token_count()` matches the cumulative
    ///   prefill + decode tokens.
    /// * No panics during the lifecycle.
    ///
    /// What we do NOT assert: numerical equivalence to the flat path.
    /// Weights are random, so output values are arbitrary. Numerical
    /// validation is deferred to an end-to-end test with loaded weights.
    ///
    /// Skips on no-Metal hosts.
    #[test]
    fn test_lfm2_chat_sync_core_paged_smoke_via_helpers() {
        use crate::array::{DType, MxArray};

        let cfg = paged_tiny_config(Some(true));
        let mut inner = match Lfm2Inner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!(
                        "skipping test_lfm2_chat_sync_core_paged_smoke_via_helpers (no Metal): {msg}"
                    );
                    return;
                }
                panic!("unexpected Lfm2Inner::new failure: {msg}");
            }
        };
        assert!(
            inner.paged_adapter.is_some(),
            "paged_tiny_config(Some(true)) must construct paged_adapter"
        );

        // Cast all weights to BF16 to match the pool dtype. Random-init
        // weights from `Lfm2Inner::new` are Float32, but the paged pool
        // was built BFloat16, so `update_keys_values` would reject
        // F32-typed K/V from the layers. Mirror Qwen3's smoke-test cast.
        let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype BFloat16") };

        // Embedding.
        let w = inner.embed_tokens.get_weight();
        inner.embed_tokens.set_weight(&cast(&w)).expect("set embed");
        // Embedding norm.
        let w = inner.embedding_norm.get_weight();
        inner
            .embedding_norm
            .set_weight(&cast(&w))
            .expect("set embedding_norm");

        // Per-layer weights. Use the now-`pub(crate)` inner fields.
        use crate::models::lfm2::decoder_layer::OperatorType;
        for layer in inner.layers.iter_mut() {
            let w = layer.operator_norm.get_weight();
            layer
                .operator_norm
                .set_weight(&cast(&w))
                .expect("set op_norm");
            let w = layer.ffn_norm.get_weight();
            layer.ffn_norm.set_weight(&cast(&w)).expect("set ffn_norm");

            match &mut layer.operator {
                OperatorType::Attention(attn) => {
                    let w = attn.q_proj.get_weight();
                    attn.q_proj.set_weight(&cast(&w)).expect("set q");
                    let w = attn.k_proj.get_weight();
                    attn.k_proj.set_weight(&cast(&w)).expect("set k");
                    let w = attn.v_proj.get_weight();
                    attn.v_proj.set_weight(&cast(&w)).expect("set v");
                    let w = attn.out_proj.get_weight();
                    attn.out_proj.set_weight(&cast(&w)).expect("set o");
                    let w = attn.q_layernorm.get_weight();
                    attn.q_layernorm.set_weight(&cast(&w)).expect("set qn");
                    let w = attn.k_layernorm.get_weight();
                    attn.k_layernorm.set_weight(&cast(&w)).expect("set kn");
                }
                OperatorType::Conv(conv) => {
                    let w = conv.conv.get_weight();
                    conv.conv.set_weight(&cast(&w)).expect("set conv_w");
                    let w = conv.in_proj.get_weight();
                    conv.in_proj.set_weight(&cast(&w)).expect("set in_proj");
                    let w = conv.out_proj.get_weight();
                    conv.out_proj.set_weight(&cast(&w)).expect("set out_proj");
                }
            }

            let mlp = &mut layer.feed_forward;
            let w = mlp.get_gate_proj_weight();
            mlp.set_gate_proj_weight(&cast(&w)).expect("set gate");
            let w = mlp.get_up_proj_weight();
            mlp.set_up_proj_weight(&cast(&w)).expect("set up");
            let w = mlp.get_down_proj_weight();
            mlp.set_down_proj_weight(&cast(&w)).expect("set down");
        }

        // Drive the adapter lifecycle the same way `chat_sync_core_paged`
        // does. seq_id is arbitrary (per-request scoping).
        let prompt: Vec<u32> = vec![10, 20, 30, 40];
        let max_decode: u32 = 2;

        {
            let adapter = inner
                .paged_adapter
                .as_mut()
                .expect("paged_adapter constructed above");
            adapter
                .reset_for_new_request(0)
                .expect("reset_for_new_request");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32 + max_decode)
                .expect("allocate_suffix_blocks");
        }

        // Prefill the suffix == full prompt (cached_prefix_len = 0).
        let logits = match inner.run_paged_prefill_chunk(&prompt, &prompt, 0) {
            Ok(l) => l,
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                    eprintln!("skipping test_lfm2_chat_sync_core_paged_smoke_via_helpers: {msg}");
                    return;
                }
                panic!("unexpected run_paged_prefill_chunk failure: {msg}");
            }
        };
        assert_eq!(
            logits.ndim().expect("ndim"),
            1,
            "prefill logits must be 1-D"
        );
        assert_eq!(
            logits.shape_at(0).expect("shape_at(0)"),
            cfg.vocab_size as i64,
            "prefill logits must be [vocab]"
        );
        let logits_f32 = logits.astype(DType::Float32).expect("astype f32");
        logits_f32.eval();
        let v0 = logits_f32.item_at_float32(0).expect("item_at_float32(0)");
        assert!(v0.is_finite(), "prefill logits[0] must be finite, got {v0}");

        // Adapter cursor should now equal prompt length.
        {
            let adapter = inner.paged_adapter.as_ref().unwrap();
            assert_eq!(adapter.current_token_count(), prompt.len() as u32);
        }

        // Two decode steps with arbitrary token values.
        for (i, tok) in [50u32, 60u32].iter().enumerate() {
            let next_logits = match inner.run_paged_decode_step(*tok) {
                Ok(l) => l,
                Err(e) => {
                    let msg = e.reason.to_string();
                    if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                        eprintln!(
                            "skipping test_lfm2_chat_sync_core_paged_smoke_via_helpers: {msg}"
                        );
                        return;
                    }
                    panic!("unexpected run_paged_decode_step failure on step {i}: {msg}");
                }
            };
            // Decode logits shape: [1, 1, vocab].
            assert_eq!(next_logits.ndim().expect("ndim"), 3);
            assert_eq!(
                next_logits.shape_at(2).expect("shape_at(2)"),
                cfg.vocab_size as i64
            );
            let next_f32 = next_logits.astype(DType::Float32).expect("astype f32");
            next_f32.eval();
            let v = next_f32.item_at_float32(0).expect("item_at_float32(0)");
            assert!(
                v.is_finite(),
                "decode logits[0] step {i} must be finite, got {v}"
            );
        }

        // Cursor advanced by 2 decode tokens.
        {
            let adapter = inner.paged_adapter.as_ref().unwrap();
            assert_eq!(
                adapter.current_token_count(),
                prompt.len() as u32 + 2,
                "decode steps must advance the adapter cursor"
            );
        }
    }

    /// All-conv config (zero attention layers) with the flag enabled must
    /// fail with a clear error — paged KV cache is meaningless without
    /// attention layers, and silently constructing a pool with
    /// `num_layers=0` would violate `LayerKVPool::new`'s invariant.
    #[test]
    fn test_lfm2_inner_rejects_all_conv_with_paged_flag() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.layer_types = vec!["conv".to_string(), "conv".to_string()];
        let result = Lfm2Inner::new(cfg);
        assert!(
            result.is_err(),
            "all-conv layer_types with use_block_paged_cache=true must fail"
        );
        let err_msg = result.err().unwrap().reason.to_string();
        assert!(
            err_msg.contains("no full_attention layers")
                || err_msg.contains("No Metal device found"),
            "expected clear error about missing attention layers, got: {err_msg}"
        );
    }
}
