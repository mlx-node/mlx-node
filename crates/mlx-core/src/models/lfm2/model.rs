use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tracing::{info, warn};

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, ResetScope, SaveStateArgs, ThinkingSetup, TurnOutput,
    TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::ChatCmd;
use crate::engine::napi_glue::start_chat_stream;
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle};
use crate::engine::{
    IMAGE_CHANGE_RESTART_PREFIX, ReasoningTracker, ThinkingPolicy, apply_all_penalties,
    build_chatml_continue_delta_text, build_synthetic_user_message, compute_performance_metrics,
    finalize_chat_result, generated_capacity_hint, kv_capacity_round_up,
};
use crate::models::qwen3_5::arrays_cache::ArraysCache;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::profiling::PerformanceMetrics;
use crate::sampling::sample;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::transformer::KVCache;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::config::Lfm2Config;
use super::decoder_layer::{Lfm2DecoderLayer, Lfm2LayerKind};
use super::layer_cache::Lfm2LayerCache;

/// Whether the paged-prefill last-token slice optimization is enabled.
///
/// When ON (default), the eager paged prefill slices the residual stream to the
/// final token BEFORE the output norm + `lm_head` projection, so the largest
/// matmul only runs over the single row whose logits the caller consumes —
/// byte-identical, since the discarded rows are never read and all cache writes
/// already happened in the per-layer loop. Set `MLX_LFM2_DISABLE_LAST_TOKEN_SLICE`
/// (any value) to fall back to the old "project full T, then slice" behavior for
/// same-binary A/B baselining. The env var is read once on first call and cached;
/// subsequent reads hit the `OnceLock` fast path.
fn last_token_slice_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("MLX_LFM2_DISABLE_LAST_TOKEN_SLICE").is_none())
}

/// Escape hatch for the quantized compiled decode path (flat + paged). Returns
/// `true` (enabled) unless `MLX_LFM2_DISABLE_QUANT_COMPILED` is set, which forces
/// quantized checkpoints back onto the eager decode path (the A/B baseline and a
/// production escape hatch). Read once via `OnceLock` (process-global, NOT
/// per-request) matching the `last_token_slice_enabled` house style.
pub(crate) fn quant_compiled_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("MLX_LFM2_DISABLE_QUANT_COMPILED").is_none())
}

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
    /// Unique non-zero id for the compiled C++ forward path.
    /// Drawn from the shared `QWEN35_MODEL_ID_COUNTER` (see the assignment in
    /// `Lfm2Inner::new`) so ids are globally unique across every model that
    /// shares the compiled weight registry. Used by
    /// [`Lfm2Inner::compiled_path_active`] to gate the (not-yet-enabled)
    /// compiled path.
    pub(crate) model_id: u64,
    /// True iff EVERY registered floating weight is BFloat16 (with the sole
    /// intentional exception of `*.expert_bias`, kept F32 on MoE checkpoints).
    /// Computed once at load time over the full registered param map (see
    /// `all_registered_float_weights_are_bf16` in `persistence.rs`) and consulted
    /// by [`Lfm2Inner::paged_compiled_decode_setup`] to gate compiled-PAGED
    /// decode: the compiled-paged graph + paged KV pools are bf16-only
    /// (`KvDtype::Bf16`, bf16 static mask), and the paged graph consumes far more
    /// than q/k/v — operator/FFN/final norms, q/k norms, out_proj, conv
    /// weights/biases, dense-MLP or MoE router/expert weights, and (untied)
    /// lm_head — any of which, if non-bf16, would flow a non-bf16 hidden state /
    /// q / k / v into the bf16-only `paged_kv_write`/`paged_attention`. A
    /// load-time scan of the whole map is authoritative (it sees exactly what
    /// the C++ `get_weight` reads) and matches the graph, not a hand-picked
    /// subset. `false` until weights register (defaults safe: a model that
    /// somehow reached decode unregistered falls back to eager paged).
    pub(crate) all_float_weights_bf16: bool,
    /// True iff this checkpoint is quantized (any `.scales`-suffixed tensor).
    /// Computed once at load time (`persistence.rs`: `params.keys().any(.scales)`)
    /// and set unconditionally after construction, so it is authoritative for ALL
    /// checkpoints (independent of whether the compiled path registered).
    /// `paged_compiled_decode_setup` consults it to switch the compiled-paged
    /// bf16-only gate over to the `non_quant_floats_bf16` invariant — quantized
    /// `.weight` tensors are uint32-packed, not bf16, so the all-float scan is
    /// meaningless for them.
    pub(crate) is_quantized: bool,
    /// True iff every NON-quantized floating weight is BFloat16 — the
    /// quantized-checkpoint analogue of `all_float_weights_bf16`. The packed
    /// `.weight` tensors are uint32 (skipped); their float `.scales`/`.biases`
    /// companions plus the unquantized dense floats (norms, conv biases, untied
    /// lm_head, the dense bf16 embedding) must all be bf16 so the hidden state
    /// feeding the bf16-only `paged_kv_write`/`paged_attention` stays bf16.
    /// `false` until a quantized checkpoint registers; only meaningful then.
    pub(crate) non_quant_floats_bf16: bool,
}

/// Adapter giving the kept paged streaming cores their legacy
/// `cb.call(result, mode)` shape over the engine's [`ChunkSink`].
///
/// Pre-S11 this wrapped the raw `StreamTx`; the engine now owns the
/// channel and hands the paged probe a `&dyn ChunkSink`, so the wrapper
/// forwards `.call()` to [`ChunkSink::send`] (the call mode is
/// meaningless on the mpsc path and is dropped, exactly like the old
/// `StreamTx` wrapper did).
struct StreamSender<'a>(&'a dyn ChunkSink);

impl StreamSender<'_> {
    fn call(&self, result: napi::Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
        self.0.send(result);
    }
}

/// Classification of the prefix-cache decision made from a
/// [`Lfm2Inner::verify_cache_prefix`] return value plus the incoming
/// token count.
///
/// Test-only mirror of the inlined branch at the top of the deleted
/// flat `chat_sync_core` / `chat_stream_sync_core` (now the engine
/// session core's verify-prefix split)
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
/// Mirrors the inlined branch at the top of the deleted flat
/// `chat_sync_core` / `chat_stream_sync_core` (now the engine session
/// core's verify-prefix split);
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

/// Build the LFM2 wire-format delta text for a tool-result turn.
///
/// LFM2's chat template renders tool-role messages as plain
/// `<|im_start|>tool\n{content}<|im_end|>` blocks — no
/// `<tool_response>` wrapping (unlike Qwen3.5). Leading `\n` closes
/// the cached `<|im_end|>` line, then we open the `tool` turn, close
/// it, and open an assistant turn ready for generation. No
/// `<think>\n` prefix because LFM2's template never injects one.
///
/// `is_error` is the structured tool-error signal. When `Some(true)`,
/// the shared [`crate::tokenizer::TOOL_ERROR_MARKER`] is prepended to
/// `content` via [`crate::tokenizer::apply_tool_error_marker`]; `None`
/// / `Some(false)` keep the wire bytes byte-equal to the pre-feature
/// output.
///
/// Extracted from
/// [`Lfm2Inner::chat_session_continue_tool_sync`] /
/// [`Lfm2Inner::chat_stream_session_continue_tool_sync`] so the
/// wire-format choice can be pinned by pure-string unit tests that
/// don't need a loaded LFM2 model.
pub(crate) fn build_lfm2_tool_delta_text(content: &str, is_error: Option<bool>) -> String {
    let rendered_content = crate::tokenizer::apply_tool_error_marker(content, is_error);
    format!("\n<|im_start|>tool\n{rendered_content}<|im_end|>\n<|im_start|>assistant\n")
}

/// Holds the compiled-paged decode session state + RAII guards for one
/// paged decode turn. All guards are over GLOBAL statics / unit, so this
/// struct does NOT borrow `self`. Created by `paged_compiled_decode_setup`,
/// consumed across the whole decode loop, dropped at loop exit (the
/// `Lfm2PagedResetGuard` resets the C++ paged globals on EVERY exit path).
struct Lfm2PagedCompiledState {
    // DROP ORDER IS LOAD-BEARING. Struct fields drop in DECLARATION order, so
    // these three guards MUST be listed reset-guard → weight-guard → lock so
    // that `Lfm2PagedResetGuard::drop()` (→ `mlx_lfm2_paged_reset()`) runs
    // WHILE the lifecycle mutex + weight read lock are STILL held. This matches
    // the original local-variable reverse-drop order (locals declared
    // lock→weight→reset dropped reset→weight→lock). If the reset ran AFTER the
    // lifecycle mutex released, another compiled-paged request could acquire
    // the mutex and seed/use the shared process-global paged state in the
    // window before this request's delayed reset cleared it → cross-request
    // null forwards / decode corruption. Do NOT reorder these three fields.
    _paged_reset_guard: Option<Lfm2PagedResetGuard>,
    _weight_guard: Option<std::sync::RwLockReadGuard<'static, ()>>,
    _compiled_lock: Option<std::sync::MutexGuard<'static, ()>>,
    cpp_session_ready: bool,
    cpp_compiled_step_completed: bool,
    max_blocks_per_seq: u32,
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

        // Block-paged KV adapter — default ON.
        //
        // Chat dispatch is wired through this adapter at every chat-entry
        // site: the engine session core's `ChatBackend::paged_turn` probe
        // hands whole turns to `paged_turn_sync_core` /
        // `paged_turn_stream_core` whenever the adapter is live.
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
            // HARD INVARIANT (compiled-path gate correctness): the model id MUST
            // come from the SINGLE shared counter, not a per-family one. The
            // compiled C++ path keys ownership on one process-global atom
            // (`g_active_model_id`) shared with qwen3.5 (dense + MoE); if lfm2
            // drew from its own counter, an lfm2 id could equal a resident
            // qwen3.5 id and the id-equality gate would false-positive,
            // decoding the other family's weights → gibberish. Allocating from
            // qwen3.5's counter makes every live id globally unique, so a
            // non-match is a clean ownership eviction, never a collision. Any
            // future compiled model sharing the registry MUST also draw from
            // this counter.
            model_id: crate::models::qwen3_5::model::QWEN35_MODEL_ID_COUNTER
                .fetch_add(1, Ordering::Relaxed),
            // Safe default: not bf16-clean until the load path verifies the
            // registered weights (set in `persistence.rs` alongside the C++
            // weight registration). A non-match keeps compiled-PAGED OFF.
            all_float_weights_bf16: false,
            // Safe defaults: `Lfm2Inner::new` only sees `config` (no params/scales),
            // so both are set post-construction in `persistence.rs` — `is_quantized`
            // unconditionally, `non_quant_floats_bf16` only when a quantized
            // checkpoint registers. Defaults keep the quantized compiled path OFF.
            is_quantized: false,
            non_quant_floats_bf16: false,
        })
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Whether the compiled C++ forward path owns this model's weights and may
    /// be taken for decode. Active only when the C++ side's active model id
    /// (`mlx_lfm2_get_model_id()`, which reads the shared `g_active_model_id`
    /// atom) equals this model's [`Lfm2Inner::model_id`].
    ///
    /// The id is published ONLY by `register_weights_with_cpp` (load time), which
    /// is invoked for bf16/f16 checkpoints — DENSE or sparse-MoE, FLAT or PAGED —
    /// AND for QUANTIZED checkpoints (authoritative per-projection quant-info is
    /// published, so `linear_proj` / `lfm2_switch_linear` dispatch correctly),
    /// gated by the `MLX_LFM2_DISABLE_QUANT_COMPILED` hatch + a dense (non-packed)
    /// input embedding; see `should_register_compiled`. The single registered
    /// weight map + `model_id` serve BOTH the flat (`lfm2_decode_fn`) and paged
    /// (`lfm2_decode_fn_paged`) compiled graphs; the per-step dispatcher picks the
    /// right one. The gate is false until an lfm2 model has registered its weights
    /// into the shared map (and false for an instance evicted by a later load).
    pub(crate) fn compiled_path_active(&self) -> bool {
        let active = unsafe { mlx_sys::mlx_lfm2_get_model_id() };
        active != 0 && active == self.model_id
    }

    /// Forward pass through the full model.
    ///
    /// Follows `lfm2.py:258-279` (Lfm2Model.__call__) + Model.__call__ (tied lm_head).
    ///
    /// Returns logits [B, T, vocab_size].
    ///
    /// `pub(crate)` so the persistence-module test
    /// `production_compiled_decode_matches_native_with_conv_bias` can drive this
    /// pure-native per-step forward as the parity REFERENCE for the production
    /// compiled conv_bias=true decode path. Crate-internal only — no public/NAPI
    /// surface change.
    pub(crate) fn forward(&mut self, input_ids: &MxArray) -> Result<MxArray> {
        // PREFILL stays native. The compiled C++ decode path is wired ONLY into
        // the single-token decode stepper (`Lfm2Decode`, built by
        // `ChatBackend::begin_decode`; it seeds the
        // compiled graph from the post-prefill caches this native forward
        // builds, then drives `mlx_lfm2_moe_forward` per step). `forward()`
        // never calls the compiled path.

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

        // 4. LM head or tied embeddings. The tied path routes through
        // `Embedding::as_linear`, which handles BOTH a packed-quantized
        // embedding (`mlx_quantized_matmul` on the packed tensors — no dense
        // table) AND a dense bf16 embedding (`h @ get_weight()^T`, numerically
        // identical to the prior matmul).
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&h)?
        } else {
            self.embed_tokens.as_linear(&h)?
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
    ///
    /// Renamed from the pre-S11 inherent `reset_caches` so the
    /// [`ChatBackend::reset_caches`] trait method (which takes a
    /// [`ResetScope`]) cannot shadow it at concrete-typed call sites.
    /// Both engine scopes dispatch here for the SHARED clear; the
    /// explicit command reset additionally purges the paged prefix
    /// cache in the trait impl (see [`ChatBackend::reset_caches`]).
    fn reset_caches_internal(&mut self) {
        self.caches = init_caches(&self.config);
        self.cached_token_history.clear();
        self.cached_image_key = None;
        // Drop any live paged-adapter request. Without this, a
        // finalize_turn_keep_live call from a prior session would leave
        // block_table populated and a subsequent `paged_turn_sync_core`
        // could mistakenly take the warm-continue path against stale
        // tokens. NOTE: this alone is NOT a fully cold reset — released
        // full blocks stay content-addressed in the allocator's prefix
        // cache; the EXPLICIT command reset purges them on top of this
        // (`ResetScope::Command` branch of the trait impl).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
    }

    /// Export the compiled C++ decode caches back into `self.caches`, then
    /// MATERIALIZE the imported handles so subsequent native turns (or the
    /// next compiled seed) read live buffers — NOT lazy graph nodes that the
    /// `Lfm2CompiledResetGuard`'s `mlx_lfm2_moe_reset()` is about to free.
    ///
    /// Caller invariant: this must run BEFORE the reset guard drops, and the
    /// export → import → eval must complete with no `?`-early-return between
    /// the import and the eval (we collect every imported cache, install it,
    /// then eval the whole set in one shot).
    fn export_compiled_caches(&mut self) -> Result<()> {
        let num_layers = self.config.num_hidden_layers as usize;
        let mut export_ptrs: Vec<*mut mlx_sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers * 2];
        let exported = unsafe {
            mlx_sys::mlx_lfm2_moe_export_caches(export_ptrs.as_mut_ptr(), (num_layers * 2) as i32)
        };
        if exported <= 0 {
            // Nothing live to export (e.g. compiled init bailed). Leave the
            // native caches from prefill in place; native is the source of
            // truth in that case.
            return Ok(());
        }

        let cache_offset = unsafe { mlx_sys::mlx_lfm2_moe_get_cache_offset() };

        let mut new_caches: Vec<Lfm2LayerCache> = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if self.config.is_attention_layer(i) {
                let k_ptr = export_ptrs[i * 2];
                let v_ptr = export_ptrs[i * 2 + 1];
                if k_ptr.is_null() || v_ptr.is_null() {
                    return Err(Error::from_reason(
                        "lfm2 compiled cache export: null attn KV handle",
                    ));
                }
                // `from_handle` wraps the heap-allocated handle (null-checked).
                let keys = MxArray::from_handle(k_ptr, "lfm2 compiled export keys")?;
                let values = MxArray::from_handle(v_ptr, "lfm2 compiled export values")?;
                let mut kv = KVCache::new();
                kv.set_keys(keys);
                kv.set_values(values);
                kv.set_offset(cache_offset);
                new_caches.push(Lfm2LayerCache::Attention(kv));
            } else {
                let s_ptr = export_ptrs[i * 2];
                if s_ptr.is_null() {
                    return Err(Error::from_reason(
                        "lfm2 compiled cache export: null conv state handle",
                    ));
                }
                let state = MxArray::from_handle(s_ptr, "lfm2 compiled export conv state")?;
                // slot.b (export_ptrs[i*2+1]) is the unused scalar placeholder —
                // a freshly heap-allocated copy. Wrap + drop to release it so it
                // doesn't leak (export hands back an owned heap handle).
                let placeholder = export_ptrs[i * 2 + 1];
                if !placeholder.is_null() {
                    let _ =
                        MxArray::from_handle(placeholder, "lfm2 compiled export conv placeholder");
                }
                let mut conv = ArraysCache::new(1);
                conv.set(0, state);
                new_caches.push(Lfm2LayerCache::Conv(conv));
            }
        }

        // Materialize the freshly-imported handles BEFORE installing them as
        // `self.caches`. If the eval fails, the `?` returns while
        // `Lfm2CompiledResetGuard` still drops and frees the compiled globals
        // — so we must NOT have already replaced `self.caches` with lazy
        // handles that reference those soon-to-be-freed nodes. Keeping the old
        // (last-safe-native) caches on failure lets a subsequent native turn
        // fall back cleanly instead of feeding freed buffers to the GPU.
        eval_lfm2_caches(&new_caches)?;
        self.caches = new_caches;
        Ok(())
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
    /// Renamed from the pre-S11 inherent `save_cache_state` so the
    /// [`ChatBackend::save_cache_state`] trait method (which takes
    /// [`SaveStateArgs`]) cannot shadow it; the trait impl and the kept
    /// paged cores all dispatch here.
    fn save_cache_state_internal(
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
            self.reset_caches_internal();
        }
    }

    /// Block-paged whole-turn core (kept behind the engine's
    /// [`ChatBackend::paged_turn`] probe since S11; pre-S11 this was the
    /// block-paged variant of the deleted flat `chat_sync_core`).
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
    /// Limitations:
    /// - Conv-layer prefix reuse is NOT carried across paged turns; each
    ///   paged turn reprefills conv state from the start of the prompt.
    /// - Pure-cache prompt (every prompt token already in the paged pool)
    ///   is rejected — same caveat as Qwen3's paged path.
    #[allow(clippy::too_many_arguments)]
    fn paged_turn_sync_core(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        think_end_id: Option<u32>,
        think_end_str: Option<String>,
        include_reasoning: bool,
        p: &crate::engine::ChatParams,
        thinking: ThinkingSetup,
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

        // LFM2's chat template ignores enable_thinking; the model ALWAYS
        // emits a <think>…</think> block, so reasoning is always tracked AND
        // parsed. reasoningEffort controls the thinking BUDGET, not whether.
        // The (enabled, budget) pair is resolved ONCE at turn entry from
        // the family policy (AlwaysOnBudgetFromEffort) and threaded in as
        // `thinking`; this core no longer recomputes it.
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = ReasoningTracker::from_setup(&thinking, think_end_id);

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
        // vLLM-style exact-prefix cap — see qwen3/model.rs:paged_turn_sync_core
        // for the full rationale. Forces the cache lookup (and the live-prefix
        // continue check) to leave at least one suffix token for the prefill
        // chunk, so retries of an earlier identical turn never produce a
        // zero-delta prompt.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        // P3: adapter-owned warm-continue/cold-start lifecycle (gemma4 ships
        // this exact call; byte-identical). NOTE: lfm2's hand-rolled
        // `can_continue` had NO reuse_cache term, so pass LITERAL `true` —
        // the adapter ANDs reuse_cache into its can_continue, and `true`
        // preserves the original always-eligible warm predicate. Suffix
        // blocks are allocated inside prepare_turn; do not re-allocate.
        let cached_prefix_len = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| {
                Error::from_reason(
                    "paged_turn_sync_core: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?
            .prepare_turn_with_max_cache_hit_tokens(
                seq_id,
                &tokens,
                total_budget,
                true,
                &[],
                0,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason)?
            .cached_prefix_len;

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
                Error::from_reason("paged_turn_sync_core: cached_prefix_len > total_prompt_tokens")
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
        let forward_result = self.paged_turn_sync_core_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            p,
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
        self.save_cache_state_internal(true, &tokens, &generated_tokens, last_token_in_cache);

        // Performance metrics.
        // Paged prefill reprocesses the FULL prompt through conv layers
        // (run_paged_prefill_chunk Pass 1); only the attention suffix skips the
        // cached prefix. ttft measures full-prompt work, so the throughput
        // numerator must be the full prompt, not tokens.len()-cached_prefix_len.
        // (If a future LFM2 paged variant carries conv state across turns and
        // truly forwards only the delta during prefill, revert this to the
        // delta count.)
        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len(),
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

    /// Set up the compiled-PAGED decode session for one decode turn (shared by
    /// the non-streaming `paged_turn_sync_core_inner` and the streaming
    /// `paged_turn_stream_core_inner`).
    ///
    /// Acquires the cross-family compiled lifecycle lock + weight read lock,
    /// re-checks ownership, gates on block_size + bf16 weights, seeds the C++
    /// paged graph once from the live post-prefill adapter pools + conv state,
    /// and arms the RAII reset guard. Returns the populated
    /// [`Lfm2PagedCompiledState`], which the caller threads into
    /// `paged_compiled_decode_step` for every step and drops at loop exit.
    ///
    /// MUST be called AFTER prefill (the adapter pools + conv caches are read
    /// here) and BEFORE the decode loop.
    fn paged_compiled_decode_setup(&mut self) -> Result<Lfm2PagedCompiledState> {
        // ===== Compiled C++ PAGED decode-path dispatch =====
        //
        // Mirrors the FLAT path's lock contract (`Lfm2Decode` via
        // `ChatBackend::begin_decode`) and qwen3.5's
        // paged `cpp_session_ready` gate. The compiled-PAGED decode runs only
        // when ALL of:
        //   1. `compiled_path_active()` — weights registered for our model_id.
        //      (The registration gate publishes the id for ANY non-quantized
        //      bf16/f16 checkpoint, FLAT or PAGED, so this is the bf16/f16 +
        //      non-quant condition; a quantized checkpoint never registers and is
        //      structurally `false` here.)
        //   1b. The model weights are bf16. The paged adapter's `LayerKVPool` is
        //      always BFloat16 and the C++ paged graph hard-codes `KvDtype::Bf16`
        //      (its static attn mask + pool/scale dtype probes are bf16/f32), so
        //      an f16 checkpoint would be REJECTED by the first compiled-paged
        //      forward and silently fall back at step 0. Gate it OUT here so an
        //      f16 paged checkpoint takes the correct pure-Rust eager paged path
        //      with no wasted seed and no lying engagement signal.
        //   2. `adapter.block_size() == CPP_PAGED_REQUIRED_BLOCK_SIZE` (16): the
        //      compiled-paged graph hard-codes block_size=16 in its
        //      `paged_kv_write` / `paged_attention` calls.
        //   3. `init_lfm2_paged_compiled_session` succeeds (every attn layer has
        //      a usable `LayerKVPool` slot; every conv layer's
        //      `Lfm2LayerCache::Conv` state is populated by the eager paged
        //      prefill above; the C++ `g_lfm2_paged_inited` is set after init
        //      catches no exception).
        //
        // LOCK CONTRACT (same as the flat path): registration is the WRITER
        // (`register_weights_with_cpp` holds `COMPILED_WEIGHTS_RWLOCK.write()`,
        // clears + stores, bumps the compile epoch, publishes model_id LAST).
        // Decode is the READER: the `_weight_guard` (`.read()`, poison-recovered)
        // spans the `mlx_lfm2_get_model_id()` re-check, the seed
        // (`init_lfm2_paged_compiled_session` → `mlx_lfm2_moe_init_paged`), and
        // EVERY `forward_lfm2_cpp_paged` step, so the (epoch, id) pair validated
        // is exactly the one the compiled-paged graph executes against.
        //
        // CONV STATE: unlike qwen's GDN linear caches, lfm2 needs NO cross-turn
        // conv export — the eager paged path reprefills conv from token 0 each
        // turn (see `paged_turn_sync_core`'s per-turn `self.caches =
        // init_caches(..)`), so the compiled-paged graph threads conv state
        // WITHIN a turn only (in the C++ paged globals) and there is no
        // post-loop export step.
        use crate::models::qwen3_5::model::CPP_PAGED_REQUIRED_BLOCK_SIZE;
        let mut use_cpp_pre = self.compiled_path_active();
        // bf16-activation gate (1b): the compiled-PAGED graph + paged KV pools are
        // bf16-only (KvDtype::Bf16, bf16 static mask), so the hidden state feeding
        // `paged_kv_write`/`paged_attention` must be bf16. The gate MUST match what
        // the graph consumes, not a hand-picked subset: `lfm2_decode_fn_paged` reads
        // operator/FFN/final norms, q/k norms, attention out_proj, conv
        // weights/biases, dense-MLP or MoE router/expert weights, and the (untied)
        // lm_head in addition to q/k/v.
        //
        //  - bf16 checkpoint: EVERY float weight must be bf16
        //    (`all_float_weights_bf16`).
        //  - QUANTIZED checkpoint: the packed `.weight` tensors are uint32 (NOT part
        //    of the activation dtype); the invariant is that the NON-quantized floats
        //    (norms, conv biases, untied lm_head, dense bf16 embedding) plus the quant
        //    float companions are bf16 (`non_quant_floats_bf16`), AND the
        //    `MLX_LFM2_DISABLE_QUANT_COMPILED` escape hatch is not set. Packed-quant
        //    INPUT embeddings are already barred at registration (the C++ does a dense
        //    `take` over the embedding), so a registered quantized checkpoint reaching
        //    here has a usable dense embedding.
        //
        // `*.expert_bias` (intentional F32 on MoE) is the one allowed exception,
        // handled in the load-time scans.
        let activations_bf16 = if self.is_quantized {
            quant_compiled_enabled() && self.non_quant_floats_bf16
        } else {
            self.all_float_weights_bf16
        };
        if use_cpp_pre && !activations_bf16 {
            warn!(
                "lfm2 compiled paged decode: activation-dtype invariant unmet (a non-bf16 float \
                 weight — embedding, norm, projection, conv, FFN/MoE, or lm_head — or quantized \
                 compiled decode disabled); the compiled-PAGED graph + paged KV pools are \
                 bf16-only, so using the pure-Rust eager paged decode path for this request."
            );
            use_cpp_pre = false;
        }
        let mut _compiled_lock = if use_cpp_pre {
            Some(
                crate::engine::compiled_lock::COMPILED_LIFECYCLE_MUTEX
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
            )
        } else {
            None
        };
        let mut _weight_guard = None;
        let cpp_session_ready = if use_cpp_pre {
            let guard = crate::engine::compiled_lock::COMPILED_WEIGHTS_RWLOCK
                .read()
                .unwrap_or_else(|e| e.into_inner());
            // Re-check ownership under the read lock — a concurrent load of a
            // different model could have evicted us between the probe and here.
            if unsafe { mlx_sys::mlx_lfm2_get_model_id() } == self.model_id {
                _weight_guard = Some(guard);
                // Seed the compiled-paged graph ONCE from the live post-prefill
                // adapter pools + conv state. On any failure drop back to the
                // pure-Rust paged decode for the whole turn.
                let caches_ref = &self.caches;
                let adapter_ref = self.paged_adapter.as_ref().ok_or_else(|| {
                    Error::from_reason(
                        "paged_compiled_decode_setup: paged_adapter dropped post-prefill",
                    )
                })?;
                if adapter_ref.block_size() != CPP_PAGED_REQUIRED_BLOCK_SIZE {
                    warn!(
                        "lfm2 compiled paged decode: adapter block_size={} but compiled graph \
                         requires {}; falling back to pure-Rust paged decode",
                        adapter_ref.block_size(),
                        CPP_PAGED_REQUIRED_BLOCK_SIZE
                    );
                    false
                } else {
                    let prefill_offset = adapter_ref.current_token_count() as i32;
                    match init_lfm2_paged_compiled_session(
                        &self.config,
                        caches_ref,
                        adapter_ref,
                        prefill_offset,
                    ) {
                        Ok(()) => true,
                        Err(e) => {
                            warn!(
                                "lfm2 compiled paged decode: seed failed ({e}); falling back to \
                                 pure-Rust paged decode"
                            );
                            false
                        }
                    }
                }
            } else {
                false
            }
        } else {
            false
        };

        // When the compiled-paged session did NOT come up (model_id eviction,
        // block_size mismatch, or seed failure), the decode loop runs the pure-Rust
        // eager paged path, which touches none of the C++ compiled globals. Holding
        // the cross-family lifecycle mutex / weight read lock across that whole loop
        // is needless and blocks weight registration (.write()) and other compiled
        // startups (.lock()) for the entire generation. Drop them now. Safe because
        // cpp_session_ready==false here implies _paged_reset_guard is None (no seed
        // ran), so the reset-while-locks-held drop-order invariant is vacuous.
        if !cpp_session_ready {
            _weight_guard = None;
            _compiled_lock = None;
        }

        // RAII guard: resets the compiled-PAGED C++ globals
        // (`mlx_lfm2_paged_reset`) on EVERY exit path — including a `?`
        // early-return or a first-step fallback that flips `cpp_session_ready`
        // off — so the next request never seeds against stale paged pools.
        // Armed iff the seed succeeded.
        let _paged_reset_guard = cpp_session_ready.then_some(Lfm2PagedResetGuard);

        // Compile-cached `max_blocks_per_seq` shape — `max_position_embeddings`
        // div_ceil block_size keeps the compile-cache key stable across every
        // decode step within one turn (matches qwen's paged decode loop).
        let max_blocks_per_seq: u32 = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("paged_compiled_decode_setup: paged_adapter dropped pre-decode")
            })?;
            let max_seq = self.config.max_position_embeddings as u32;
            max_seq.div_ceil(adapter.block_size())
        };

        Ok(Lfm2PagedCompiledState {
            cpp_session_ready,
            // Tracks whether ANY compiled-paged step has completed this turn.
            // After a successful compiled step the C++ side has advanced its
            // per-conv-layer state global (and the adapter's paged pool via
            // `paged_kv_write`), but the conv state is NOT imported back into
            // `self.caches`. Falling back to the pure-Rust decode after that
            // point would run conv from stale pre-step state while
            // `paged_adapter` + `token_history` have advanced — silently
            // corrupting the response. So a mid-turn failure (after the first
            // successful compiled step) PROPAGATES; only a first-step failure
            // falls back (mirrors qwen's `should_propagate_compiled_paged_error`).
            cpp_compiled_step_completed: false,
            max_blocks_per_seq,
            _compiled_lock,
            _weight_guard,
            _paged_reset_guard,
        })
    }

    /// Run one compiled-PAGED decode step (shared by both paged decode loops).
    ///
    /// Dispatches to the compiled C++ paged forward when `st.cpp_session_ready`,
    /// else the pure-Rust paged decode. Returns the `[1, vocab]` logits (both
    /// branches produce the same 2D shape). On a FIRST-step compiled failure it
    /// rolls back the token cursor, flips `st.cpp_session_ready = false`, and
    /// re-runs the token through the pure-Rust path. After ANY compiled step has
    /// succeeded a mid-decode failure PROPAGATES as fatal (the C++ conv state has
    /// advanced but is never imported back into `self.caches`).
    fn paged_compiled_decode_step(
        &mut self,
        st: &mut Lfm2PagedCompiledState,
        token_id: u32,
        step: i32,
    ) -> Result<MxArray> {
        // Decode forward. Compiled-paged when `cpp_session_ready`, else the
        // pure-Rust paged decode.
        //
        // Defense-in-depth: if the C++ compiled-paged forward returns null on
        // the FIRST step, roll back the `record_tokens` cursor advance, flip
        // `cpp_session_ready = false`, and re-run this token through
        // `run_paged_decode_step` (which re-calls `record_tokens` on the
        // now-rolled-back cursor). After ANY compiled step has succeeded the
        // C++ conv state has advanced but is never imported back into
        // `self.caches`, so a Rust fallback would read stale state —
        // propagate the error as fatal instead of silently corrupting.
        if st.cpp_session_ready {
            let embedding_weight = self.embed_tokens.get_weight();
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "paged_compiled_decode_step: paged_adapter dropped mid-decode (cpp)",
                )
            })?;
            adapter
                .record_tokens(&[token_id])
                .map_err(Error::from_reason)?;
            let inputs = adapter
                .build_paged_attention_inputs(1, 1, st.max_blocks_per_seq)
                .map_err(Error::from_reason)?;
            let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
            match forward_lfm2_cpp_paged(&input_ids, &embedding_weight, &inputs) {
                Ok(logits) => {
                    st.cpp_compiled_step_completed = true;
                    // Compiled-paged logits are [B, vocab] (already 2D).
                    Ok(logits)
                }
                Err(e) => {
                    if crate::engine::should_propagate_compiled_paged_error(
                        st.cpp_compiled_step_completed,
                    ) {
                        warn!(
                            "lfm2 compiled paged forward failed mid-decode (step={step}) AFTER \
                             an earlier compiled step succeeded. The C++ conv state has \
                             advanced but is not imported back into self.caches, so a \
                             pure-Rust fallback would run from stale state and silently \
                             corrupt the response. Propagating as fatal. cause: {e}"
                        );
                        adapter
                            .rollback_last_tokens(1)
                            .map_err(Error::from_reason)?;
                        return Err(e);
                    }
                    warn!(
                        "lfm2 compiled paged forward failed on first decode step \
                         (step={step}); rolling back token cursor and falling back to \
                         pure-Rust paged decode for the rest of this request. cause: {e}"
                    );
                    adapter
                        .rollback_last_tokens(1)
                        .map_err(Error::from_reason)?;
                    st.cpp_session_ready = false;
                    // The rest of this turn runs pure-Rust eager paged decode
                    // (`run_paged_decode_step` touches none of the C++ compiled
                    // paged globals), so the seeded compiled-paged session is now
                    // dead. Tear it down and release the process-wide locks NOW
                    // instead of pinning them for the whole generation (they
                    // otherwise block weight registration `.write()` / other
                    // compiled startups `.lock()`).
                    //
                    // ORDER IS LOAD-BEARING (mirrors the struct field drop order,
                    // see `Lfm2PagedCompiledState`): drop the reset guard FIRST so
                    // `mlx_lfm2_paged_reset()` runs WHILE the lifecycle mutex +
                    // weight read lock are STILL held; only THEN release the locks.
                    // After this all three guards are None, so the struct's
                    // eventual drop is a no-op (no double reset / double release).
                    drop(st._paged_reset_guard.take());
                    st._weight_guard = None;
                    st._compiled_lock = None;
                    // Re-run this token through the pure-Rust paged decode
                    // (re-records the token on the rolled-back cursor).
                    self.run_paged_decode_step(token_id)?.squeeze(Some(&[1]))
                }
            }
        } else {
            // Pure-Rust paged decode fallback.
            self.run_paged_decode_step(token_id)?.squeeze(Some(&[1]))
        }
    }

    /// Inner forward + decode loop for `paged_turn_sync_core`. Split out
    /// so the caller can wrap it with `release_request` on either path.
    #[allow(clippy::too_many_arguments)]
    fn paged_turn_sync_core_inner(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &crate::engine::ChatParams,
        eos_token_id: u32,
        sampling_config: &Option<crate::sampling::SamplingConfig>,
        reasoning_tracker: &mut ReasoningTracker,
        report_perf: bool,
        first_token_instant: &mut Option<std::time::Instant>,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "paged_turn_sync_core_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
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

        // ===== Compiled C++ PAGED decode-path dispatch =====
        // Acquire the locks, gate, and seed the compiled-paged session for this
        // turn. The returned state holds the RAII guards (compiled lifecycle
        // mutex, weight read lock, paged reset guard) that span the whole decode
        // loop; the same setup + per-step dispatch is shared with the streaming
        // path (`paged_turn_stream_core_inner`). See
        // `paged_compiled_decode_setup` / `paged_compiled_decode_step`.
        let mut paged_st = self.paged_compiled_decode_setup()?;

        // === DECODE LOOP ===
        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(generated_capacity_hint(max_new_tokens));
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

            // Decode forward (compiled-paged when seeded, else pure-Rust paged)
            // via the shared per-step dispatcher (mirrored by the streaming path).
            let next_logits = self.paged_compiled_decode_step(&mut paged_st, token_id, step)?;

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id()? as i32;
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

        // Output norm + lm_head. Tied path → `Embedding::as_linear` (packed
        // quantized matmul or dense `h @ weight^T`).
        //
        // OPT (`MLX_LFM2_DISABLE_LAST_TOKEN_SLICE` unset, default ON): only the
        // FINAL token's logits are ever consumed by the caller, yet the output
        // norm + lm_head (vocab 65536 × hidden 2048 matmul) would otherwise run
        // over the full `[1, T, hidden]` residual stream. Slice to the last row
        // BEFORE the projection so the largest matmul does ~T× less work. This
        // is byte-identical: every attention / conv cache write already happened
        // in the per-layer loop above, and the discarded rows are never read.
        // Setting the toggle reproduces the old "project full T, then slice"
        // behavior for same-binary A/B baselining.
        let proj_input = if last_token_slice_enabled() {
            let seq_len_h = hidden_states.shape_at(1)?;
            hidden_states.slice_axis(1, seq_len_h - 1, seq_len_h)?
        } else {
            hidden_states
        };
        let hidden_states = self.embedding_norm.forward(&proj_input)?;
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else {
            self.embed_tokens.as_linear(&hidden_states)?
        };

        // Slice the last token's logits. When the opt is ON `logits` already has
        // T=1, so this is a no-op slice; when OFF it picks the final row as
        // before. Either way the returned shape is unchanged.
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

        // Tied path → `Embedding::as_linear` (packed quantized matmul or dense
        // `h @ weight^T`).
        hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else {
            self.embed_tokens.as_linear(&hidden_states)?
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
                // cache. Known issue for follow-up.
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
        compute_layer_kinds_for(&self.config, self.layers.len())
    }

    /// Block-paged streaming variant of the deleted flat
    /// `chat_stream_sync_core`.
    ///
    /// Mirrors `paged_turn_sync_core`'s adapter lifecycle and forward
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
    /// cap as `paged_turn_sync_core` so zero-delta prompts still produce
    /// at least one suffix token to prefill. Numerical equivalence to the
    /// flat path is not asserted here.
    #[allow(clippy::too_many_arguments)]
    fn paged_turn_stream_core(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        think_end_id: Option<u32>,
        think_end_str: Option<String>,
        include_reasoning: bool,
        p: &crate::engine::ChatParams,
        thinking: ThinkingSetup,
        report_perf: bool,
        eos_token_id: u32,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        let prompt_token_count = tokens.len();
        let sampling_config = p.sampling_config;

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // LFM2's chat template ignores enable_thinking; the model ALWAYS
        // emits a <think>…</think> block, so reasoning is always tracked AND
        // parsed. reasoningEffort controls the thinking BUDGET, not whether.
        // The (enabled, budget) pair is resolved ONCE at turn entry from
        // the family policy (AlwaysOnBudgetFromEffort) and threaded in as
        // `thinking`; this core no longer recomputes it.
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = ReasoningTracker::from_setup(&thinking, think_end_id);

        // Streaming decode state
        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        // === Adapter lifecycle: warm continuation OR cold start. ===
        // See the equivalent block in `paged_turn_sync_core` for full
        // discussion.
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // See `paged_turn_sync_core` for the vLLM-style cap rationale.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        // P3: adapter-owned lifecycle (see paged_turn_sync_core). LITERAL
        // `true` for reuse_cache — lfm2's hand-rolled can_continue had no
        // reuse term; the adapter ANDs reuse_cache, so true preserves it.
        let cached_prefix_len = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| {
                Error::from_reason(
                    "paged_turn_stream_core: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?
            .prepare_turn_with_max_cache_hit_tokens(
                seq_id,
                &tokens,
                total_budget,
                true,
                &[],
                0,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason)?
            .cached_prefix_len;

        // Reset conv-layer state for this turn (see paged_turn_sync_core
        // doc comment).
        self.caches = init_caches(&self.config);
        self.cached_token_history.clear();
        self.cached_image_key = None;

        let total_prompt_tokens = tokens.len() as u32;
        let suffix_len = total_prompt_tokens
            .checked_sub(cached_prefix_len)
            .ok_or_else(|| {
                Error::from_reason(
                    "paged_turn_stream_core: cached_prefix_len > total_prompt_tokens",
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
        let result = self.paged_turn_stream_core_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            p,
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
                    // `paged_turn_sync_core`'s terminal block.
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
        // See the non-streaming `paged_turn_sync_core` for the rationale
        // on `last_token_in_cache = false`.
        let last_token_in_cache = false;
        self.save_cache_state_internal(true, &tokens, &generated_tokens, last_token_in_cache);

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
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if include_reasoning || !last_is_reasoning {
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
        }

        // Performance metrics.
        // Paged prefill reprocesses the FULL prompt through conv layers
        // (run_paged_prefill_chunk Pass 1); only the attention suffix skips the
        // cached prefix. ttft measures full-prompt work, so the throughput
        // numerator must be the full prompt, not tokens.len()-cached_prefix_len.
        // (If a future LFM2 paged variant carries conv state across turns and
        // truly forwards only the delta during prefill, revert this to the
        // delta count.)
        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len(),
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
    /// [`Self::paged_turn_stream_core`]. Split out so the caller can
    /// wrap with `release_request` in a try-style flow.
    #[allow(clippy::too_many_arguments)]
    fn paged_turn_stream_core_inner<'a>(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &crate::engine::ChatParams,
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
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "paged_turn_stream_core: caller must cap max_cache_hit_tokens at prompt.len() - 1"
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
        // starts allocating (see paged_turn_sync_core_inner for rationale).
        crate::array::synchronize_and_clear_cache();

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        // ===== Compiled C++ PAGED decode-path dispatch =====
        // Same setup the non-streaming path uses: acquire the compiled lifecycle
        // + weight locks, gate, and seed the compiled-paged session for this
        // turn. The returned state holds the RAII guards spanning the whole
        // streaming decode loop. A fatal mid-turn compiled failure surfaced by
        // `paged_compiled_decode_step` propagates out of this streaming function
        // via `?` (correct — the C++ conv state has advanced and cannot be
        // recovered into `self.caches`).
        let mut paged_st = self.paged_compiled_decode_setup()?;

        // === STREAMING DECODE LOOP ===
        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(generated_capacity_hint(max_new_tokens));
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
            // Suppress reasoning deltas when include_reasoning == false.
            // Detokenize + length-advance above stay OUTSIDE this gate.
            if p.include_reasoning || !is_reasoning {
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

            // Decode forward (compiled-paged when seeded, else pure-Rust paged)
            // via the shared per-step dispatcher (mirrors the non-streaming path).
            let next_logits = self.paged_compiled_decode_step(&mut paged_st, token_id, step)?;

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id()? as i32;
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

    /// Block-paged whole-turn dispatch behind the engine's
    /// [`ChatBackend::paged_turn`] probe (S11).
    ///
    /// Routes the two FRESH turn shapes onto the kept paged cores
    /// verbatim: sync → [`Self::paged_turn_sync_core`], streaming →
    /// [`Self::paged_turn_stream_core`]. Delta turns never reach
    /// this helper — [`ChatBackend::paged_turn`] declines them (legacy
    /// lfm2 delta paths run the flat eager prefill+decode over
    /// `self.caches` even when `paged_adapter` is `Some`).
    ///
    /// Parameter mapping == the deleted flat-core dispatch:
    /// `think_end_id`/`think_end_str` from the tokenizer,
    /// `include_reasoning`/`report_perf` from the resolved params
    /// (`extract_chat_params` resolves `include_reasoning` via the same
    /// `resolve_include_reasoning` the legacy dispatch called), and the
    /// the resolved per-turn [`ThinkingSetup`] (`args.thinking`) for the
    /// reasoning tracker.
    fn paged_whole_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        let tokenizer = args.tokenizer.clone();
        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|st| st.to_string());
        // Resolved ONCE at turn entry (resolve(AlwaysOnBudgetFromEffort,
        // config)); `ThinkingSetup` is `Copy` so it moves into whichever
        // core arm fires without re-borrowing `args`.
        let thinking = args.thinking;
        match (args.sink, args.cancelled) {
            (Some(sink), Some(cancelled)) => {
                let cb = StreamSender(sink);
                self.paged_turn_stream_core(
                    args.tokens.to_vec(),
                    tokenizer,
                    think_end_id,
                    think_end_str,
                    args.params.include_reasoning,
                    args.params,
                    thinking,
                    args.params.report_performance,
                    args.eos_id,
                    &cb,
                    cancelled,
                )?;
                Ok(TurnOutput::Streamed)
            }
            _ => {
                let result = self.paged_turn_sync_core(
                    args.tokens.to_vec(),
                    tokenizer,
                    think_end_id,
                    think_end_str,
                    args.params.include_reasoning,
                    args.params,
                    thinking,
                    args.params.report_performance,
                    args.eos_id,
                )?;
                Ok(TurnOutput::Complete(Box::new(result)))
            }
        }
    }
}

/// Eager/compiled flat decode stepper for one lfm2 turn (S11,
/// [`ChatBackend::begin_decode`]).
///
/// Byte-identical port of the deleted flat cores' decode-loop step
/// bodies: the compiled C++ step (`mlx_lfm2_moe_forward` +
/// `mlx_lfm2_moe_eval_token_and_caches`) when the turn's seed engaged
/// the compiled path, else the native [`Lfm2Inner::forward`]. The RAII
/// guards captured here reproduce the legacy in-scope teardown.
pub(crate) struct Lfm2Decode<'a> {
    // DROP ORDER IS LOAD-BEARING. Struct fields drop in DECLARATION
    // order, so these three guards MUST be listed reset-guard →
    // weight-guard → lock so that `Lfm2CompiledResetGuard::drop()`
    // (→ `mlx_lfm2_moe_reset()`) runs WHILE the lifecycle mutex +
    // weight read lock are STILL held — the same reverse-drop order the
    // deleted cores' locals had (declared lock → weight → reset).
    _compiled_reset_guard: Option<Lfm2CompiledResetGuard>,
    _weight_guard: Option<std::sync::RwLockReadGuard<'static, ()>>,
    _compiled_lock: Option<std::sync::MutexGuard<'static, ()>>,
    inner: &'a mut Lfm2Inner,
    /// Tied dense embedding weight — captured only on the compiled path
    /// (the eager forward reads `self.embed_tokens` itself).
    embed_tokens_weight: Option<MxArray>,
    use_compiled: bool,
    /// `params.reuse_cache` captured at `begin_decode` — gates the
    /// compiled-cache export in [`DecodeStep::end_decode`].
    reuse_cache: bool,
}

impl DecodeStep for Lfm2Decode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        if self.use_compiled {
            // Compiled path returns [B, vocab] (already 2D) — `false` ==
            // no squeeze, matching the legacy compiled branch.
            let emb = self.embed_tokens_weight.as_ref().ok_or_else(|| {
                Error::from_reason("lfm2 compiled decode: missing embedding weight")
            })?;
            let mut out_ptr: *mut mlx_sys::mlx_array = std::ptr::null_mut();
            let mut off: i32 = 0;
            unsafe {
                mlx_sys::mlx_lfm2_moe_forward(
                    input_ids.as_raw_ptr(),
                    emb.as_raw_ptr(),
                    &mut out_ptr,
                    &mut off,
                );
            }
            if out_ptr.is_null() {
                return Err(Error::from_reason(
                    "lfm2 compiled decode: mlx_lfm2_moe_forward returned null logits",
                ));
            }
            Ok((
                MxArray::from_handle(out_ptr, "lfm2 compiled decode logits")?,
                false,
            ))
        } else {
            // Eager native forward returns [1, 1, vocab]; `true` == the
            // legacy loops' explicit `squeeze(Some(&[1]))`.
            Ok((self.inner.forward(input_ids)?, true))
        }
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        // Legacy lfm2 never force-evals the logits — not even on the
        // budget-forced path (the forced branch discards them lazily).
        if self.use_compiled {
            // Evaluating the token triggers the whole compiled graph
            // (logits + caches via the dependency edges).
            unsafe {
                mlx_sys::mlx_lfm2_moe_eval_token_and_caches(next_token.as_raw_ptr());
            }
        } else {
            MxArray::async_eval_arrays(&[next_token]);
        }
    }

    fn end_decode(&mut self) -> Result<()> {
        // TEARDOWN: when the compiled path ran and the caller wants to
        // reuse the cache, export the C++ caches back into
        // `inner.caches` and MATERIALIZE them BEFORE
        // `Lfm2CompiledResetGuard` drops (which calls
        // `mlx_lfm2_moe_reset()` and frees the compiled globals).
        // Exported handles are lazy copies whose graph still references
        // compiled nodes; without the eval inside
        // `export_compiled_caches` the next turn would feed freed
        // buffers to the GPU. Runs while the guards (fields above) are
        // still held; an `Err` aborts the turn BEFORE
        // `save_cache_state`, reproducing the legacy
        // reset-without-export error path.
        if self.use_compiled && self.reuse_cache {
            self.inner.export_compiled_caches()?;
        }
        Ok(())
    }
}

impl ChatBackend for Lfm2Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "lfm2"
    }

    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32> {
        tok.im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))
    }

    fn policy(&self) -> ThinkingPolicy {
        // LFM2's chat template ignores enable_thinking; the model ALWAYS
        // emits a <think>…</think> block, so reasoning is always tracked
        // AND parsed. reasoningEffort controls the thinking BUDGET, not
        // whether. `AlwaysOnBudgetFromEffort` resolves to
        // `{enabled:true, budget: thinking_token_budget.or_else(||
        //   default_thinking_budget_for_effort(reasoning_effort))}` —
        // explicit thinkingTokenBudget WINS, else derive from
        // reasoningEffort (footgun preserved: effort:"low" caps to 256
        // but does NOT disable thinking).
        ThinkingPolicy::AlwaysOnBudgetFromEffort
    }

    // `resolve_params`: engine default (`extract_chat_params`) == the
    // legacy `extract_chat_params(&config)` on every deleted core.
    //
    // `render_prompt`: engine default (jinja `apply_chat_template_sync`
    // with `add_generation_prompt = true`, the request tools, and
    // `resolve_enable_thinking`) == the deleted cores' render block
    // byte-for-byte (LFM2's template itself ignores `enable_thinking`).

    fn render_continue_delta(
        &self,
        tok: &Qwen3Tokenizer,
        user_message: &str,
        _config: &ChatConfig,
    ) -> Result<Vec<u32>> {
        // Match the legacy session path's sanitization so the delta is
        // subject to the same role/content injection protection as the
        // fresh-prompt path.
        let synthetic = build_synthetic_user_message(user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;

        // LFM2's chat template does NOT inject `<think>\n` after the
        // assistant opener — the model emits `<think>` tags on its own
        // when reasoning. Always suppress the prefix by passing
        // `Some(false)` to the shared builder so the delta stays
        // template-equivalent with the LFM2 jinja output (`config` is
        // deliberately ignored — the trait gained it for qwen3.5).
        let delta_text = build_chatml_continue_delta_text(sanitized_user, Some(false));
        tok.encode_sync(&delta_text, Some(false))
    }

    fn render_tool_delta(
        &self,
        tok: &Qwen3Tokenizer,
        _tool_call_id: &str,
        content: &str,
        is_error: Option<bool>,
        _config: &ChatConfig,
    ) -> Result<Vec<u32>> {
        // LFM2-specific plain tool delta — no `<tool_response>` wrapper,
        // `tool_call_id` dropped (LFM2's template identifies tool
        // responses positionally). See [`build_lfm2_tool_delta_text`].
        let delta_text = build_lfm2_tool_delta_text(content, is_error);
        tok.encode_sync(&delta_text, Some(false))
    }

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        // Shared clear for BOTH scopes (== the legacy inherent reset):
        // wipe flat caches + token history + image key and release any
        // live paged request.
        self.reset_caches_internal();
        // The EXPLICIT command reset must restore a fully cold state:
        // `release_request` alone leaves the request's full blocks
        // content-addressed in the allocator's prefix cache, so a
        // reset-then-rerun of the same prompt would take the prefix-hit
        // 1-token-suffix prefill path — whose bf16 reduction order
        // differs from the cold full prefill, enough to flip a greedy
        // near-tie (codex S12 finding; observed "says," vs "said" at
        // token ~6 on the 1.2b checkpoint). Purge the prefix cache so
        // the next turn replays the cold prefill byte-for-byte.
        // `PrefixMiss` (turn-internal) keeps the prefix cache:
        // cross-request block reuse after a history miss is the paged
        // design's entire point.
        if scope == ResetScope::Command
            && let Some(adapter) = self.paged_adapter.as_mut()
        {
            adapter
                .release_request_and_purge_prefix_cache()
                .map_err(|e| {
                    Error::from_reason(format!(
                        "lfm2 reset_caches: paged prefix-cache purge failed: {e}"
                    ))
                })?;
        }
        Ok(())
    }

    /// **Safety invariant** (moved verbatim from the deleted inherent
    /// `verify_cache_prefix`): returns ONLY `0` (cache miss) or
    /// `cached_token_history.len()` (exact-append hit or exact match) —
    /// never an intermediate value. The engine's session core routes the
    /// exact-match case (`hit == tokens.len()`) back through the
    /// miss/reset branch, reproducing the legacy behaviour: LFM2's
    /// short-conv layers have non-invertible left-padded state and no
    /// safe "rewind-by-1" primitive, so the qwen3 pure-KV exact-match
    /// rewind exception is FORBIDDEN here (see the all-or-nothing
    /// contract on [`ChatBackend::verify_cache_prefix`]).
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

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        // lfm2's persistence is identical on the fresh and delta paths
        // (one inherent helper; `args.is_delta` is irrelevant here).
        // `last_token_in_cache` is the lfm2-only trim signal — see
        // [`SaveStateArgs::last_token_in_cache`] and the derivation note
        // in the session core.
        self.save_cache_state_internal(
            args.reuse_cache,
            args.save_tokens,
            args.generated_tokens,
            args.last_token_in_cache,
        );
    }

    fn eval_caches(&self) -> Result<()> {
        // == the deleted cores' post-prefill `eval_lfm2_caches` sync.
        eval_lfm2_caches(&self.caches)
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        // Byte-identical port of the deleted cores' prefill block: lfm2
        // builds its prompt array as int32 (dtype is load-bearing for
        // parity), runs the chunked forward, and folds the last-token
        // slice into the impl (S6 trait contract) using the ACTUAL
        // returned seq len — `chunked_prefill` returns only the final
        // chunk's logits for prompts over `PREFILL_STEP_SIZE`.
        let token_arr: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, prompt_tokens.len() as i64])?;
        let logits = self.chunked_prefill(&prompt, stream)?;
        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        last_logits.squeeze(Some(&[1]))
    }

    type Decode<'a>
        = Lfm2Decode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        // lfm2's DELTA decode loop is ALWAYS eager (panel "paged/eager
        // delta" finding): the legacy delta cores ran the native forward
        // unconditionally, with no compiled dispatch. Only fresh flat
        // turns engage the compiled path below.
        if turn.is_delta {
            return Ok(Lfm2Decode {
                _compiled_reset_guard: None,
                _weight_guard: None,
                _compiled_lock: None,
                inner: self,
                embed_tokens_weight: None,
                use_compiled: false,
                reuse_cache: turn.params.reuse_cache,
            });
        }

        let max_new_tokens = turn.params.max_new_tokens;

        // ===== Compiled C++ decode-path dispatch =====
        // Serialize the compiled lifecycle across model instances on the SHARED
        // cross-family mutex (the same instance qwen3.5 locks), then re-validate
        // ownership under the weight RwLock read guard, held for the whole
        // decode loop. Poison-recover both locks (a panicked prior holder must
        // not wedge inference forever — banned `.unwrap()` on these paths).
        //
        // LOCK CONTRACT (compiled-closure lifecycle, mirrors the qwen3.5 dense
        // path): registration is the WRITER — `register_weights_with_cpp` holds
        // `COMPILED_WEIGHTS_RWLOCK.write()` and, in one critical section, clears
        // + re-stores weights, bumps the compile epoch
        // (`mlx_lfm2_invalidate_compiled`), then publishes the model id
        // (`mlx_set_model_id`). Decode is the READER — the weight read guard
        // (`.read()`, poison-recovered) below spans BOTH the
        // `mlx_lfm2_get_model_id()` re-check AND every subsequent
        // `mlx_lfm2_moe_*` invocation this turn (the stepper holds it until it
        // drops, after the decode loop and the post-loop cache export). So the
        // (epoch, id) pair this read guard validates is exactly the one the
        // compiled graph executes against — a registration cannot interleave
        // between the re-check and the forwards. The C++ side additionally
        // guards the epoch-check + recompile with its own `g_lfm2_compiled_mu`
        // and returns the closure BY VALUE, so even a hypothetical caller that
        // did NOT hold this read lock could not dangle its closure handle.
        let use_compiled_pre = self.compiled_path_active();
        let compiled_lock = if use_compiled_pre {
            Some(
                crate::engine::compiled_lock::COMPILED_LIFECYCLE_MUTEX
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
            )
        } else {
            None
        };
        let mut weight_guard = None;
        // `mut` so the seed step below can drop back to native on any failure.
        let mut use_compiled = if use_compiled_pre {
            let guard = crate::engine::compiled_lock::COMPILED_WEIGHTS_RWLOCK
                .read()
                .unwrap_or_else(|e| e.into_inner());
            // Re-check ownership under the read lock — a concurrent load of a
            // different model could have evicted us between the probe and here.
            if unsafe { mlx_sys::mlx_lfm2_get_model_id() } == self.model_id {
                weight_guard = Some(guard);
                true
            } else {
                false
            }
        } else {
            false
        };

        // Seed the compiled decode graph ONCE from the post-prefill caches.
        // On any failure (init bailed, missing handle), drop back to native by
        // clearing `use_compiled` for the loop. The distinct reset guard fires
        // `mlx_lfm2_moe_reset()` on EVERY exit path (including `?`
        // early-returns and a partial/failed seed) so no stale C++ state leaks.
        let compiled_reset_guard = if use_compiled {
            Some(Lfm2CompiledResetGuard)
        } else {
            None
        };
        let embed_tokens_weight = if use_compiled {
            // Tied dense embedding only — lfm2 compiled checkpoints (dense OR
            // MoE) are never packed-quantized on the compiled path (the gate
            // excludes any packed input embedding, and bf16 lfm2 ships a dense
            // embedding). `get_weight()` is infallible (returns MxArray, not
            // Result).
            Some(self.embed_tokens.get_weight())
        } else {
            None
        };
        if use_compiled {
            let num_layers = self.config.num_hidden_layers as usize;

            // CRITICAL: seed the compiled decode position from the LIVE
            // attention KV offset, NOT the prompt length. The `KVCache`
            // offset accumulates prefix + delta across every chunk and
            // across reuse, so it is the true sequence position (this is
            // why lfm2 deliberately ignores `turn.total_seq_len`). Seeding
            // from a logits/prompt length would build the C++ causal mask
            // and KV write index from a too-small offset, masking out valid
            // prefix tokens and overwriting live slots. All attention layers
            // must agree on the offset; if any disagrees (corrupt/partial
            // cache), fall back to native rather than seed a wrong position.
            let mut cache_offset: Option<i32> = None;
            let mut offset_ok = true;
            for cache in self.caches.iter() {
                if let Lfm2LayerCache::Attention(kv) = cache {
                    let off = kv.get_offset();
                    match cache_offset {
                        None => cache_offset = Some(off),
                        Some(prev) if prev != off => {
                            offset_ok = false;
                            break;
                        }
                        _ => {}
                    }
                }
            }
            let prefill_len = cache_offset.unwrap_or(0);
            if !offset_ok || cache_offset.is_none() {
                // No attention layer (impossible for a real lfm2 config — every
                // shipping checkpoint interleaves ≥1 full_attention layer) or
                // attention layers reporting inconsistent offsets (corrupt /
                // partial cache). Either way, refuse to seed a wrong position
                // and fall back to native. The debug log distinguishes this
                // from the missing-handle fallback below so a hypothetical
                // zero-attention config doesn't fail silently.
                tracing::debug!(
                    "lfm2 compiled decode: no consistent attention KV offset \
                     (offset_ok={offset_ok}, has_offset={}); using native path",
                    cache_offset.is_some()
                );
                offset_ok = false;
            }
            // Budget the fixed padded cache from the TRUE position so decode
            // can never exceed it (slice_update OOB / silent corruption).
            let max_kv_len = kv_capacity_round_up(prefill_len, max_new_tokens)?;

            // Per-layer attn/conv map — built DYNAMICALLY from config (lfm2
            // mixes conv/attn irregularly; never a modulo/hardcoded pattern).
            let is_attn: Vec<i32> = (0..num_layers)
                .map(|i| i32::from(self.config.is_attention_layer(i)))
                .collect();

            // Cache pointers, stride 2 by ABSOLUTE layer idx. attn layer ->
            // KVCache keys_ref()/values_ref() (MATERIALIZED via the engine's
            // post-prefill `eval_caches`); conv layer -> ArraysCache slot 0,
            // null at +1.
            let mut cache_ptrs: Vec<*mut mlx_sys::mlx_array> =
                vec![std::ptr::null_mut(); num_layers * 2];
            // Carry the offset-consistency check forward: a bad/inconsistent
            // attention offset must also block the seed (→ native fallback).
            let mut seed_ok = offset_ok;
            for (i, cache) in self.caches.iter().enumerate() {
                match cache {
                    Lfm2LayerCache::Attention(kv) => match (kv.keys_ref(), kv.values_ref()) {
                        (Some(k), Some(v)) => {
                            cache_ptrs[i * 2] = k.as_raw_ptr();
                            cache_ptrs[i * 2 + 1] = v.as_raw_ptr();
                        }
                        _ => {
                            seed_ok = false;
                            break;
                        }
                    },
                    Lfm2LayerCache::Conv(c) => match c.get(0) {
                        Some(state) => {
                            cache_ptrs[i * 2] = state.as_raw_ptr();
                            // slot.b stays null — conv branch never reads it.
                        }
                        None => {
                            seed_ok = false;
                            break;
                        }
                    },
                }
            }

            if seed_ok {
                unsafe {
                    mlx_sys::mlx_lfm2_moe_init_from_prefill(
                        self.config.num_hidden_layers,
                        self.config.hidden_size,
                        self.config.num_attention_heads,
                        self.config.num_key_value_heads,
                        self.config.head_dim(),
                        self.config.rope_theta as f32,
                        self.config.norm_eps as f32,
                        self.config.conv_l_cache,
                        self.config.num_experts.unwrap_or(0),
                        self.config.num_experts_per_tok.unwrap_or(0),
                        self.config.num_dense_layers.unwrap_or(0),
                        i32::from(self.config.norm_topk_prob.unwrap_or(true)),
                        i32::from(self.config.use_expert_bias.unwrap_or(true)),
                        i32::from(self.config.tie_embedding),
                        i32::from(self.config.conv_bias),
                        max_kv_len,
                        1,
                        is_attn.as_ptr(),
                        cache_ptrs.as_mut_ptr(),
                        prefill_len,
                    );
                }
                // C++ init is `void` but can still bail internally (a null
                // slot or a padding/concatenate exception sets g_lfm2_inited
                // = false). Confirm it actually seeded; if not, drop to native
                // BEFORE the loop rather than letting the first forward return
                // null logits and be treated as a fatal error.
                if unsafe { mlx_sys::mlx_lfm2_moe_is_initialized() } == 0 {
                    warn!("lfm2 compiled decode: C++ seed did not initialize; using native path");
                    use_compiled = false;
                }
            } else {
                warn!(
                    "lfm2 compiled decode: missing/inconsistent post-prefill cache state; using native path"
                );
                use_compiled = false;
            }
        }

        Ok(Lfm2Decode {
            _compiled_reset_guard: compiled_reset_guard,
            _weight_guard: weight_guard,
            _compiled_lock: compiled_lock,
            inner: self,
            embed_tokens_weight,
            use_compiled,
            reuse_cache: turn.params.reuse_cache,
        })
    }

    // `finalize_turn`: engine default (`finalize_chat_result`) == every
    // deleted core's finalization (the streaming delta core's manually
    // built done-chunk was field-for-field the same construction,
    // including the tool_calls finish-reason promotion and the raw_text
    // reasoning scrub).

    fn has_paged_adapter(&self) -> bool {
        self.paged_adapter.is_some()
    }

    // `supports_images`: engine default `false` — LFM2 is text-only.
    // The NAPI entry points additionally keep their legacy
    // "LFM2 is text-only" pre-checks, which fire before any command is
    // dispatched, so the engine's typed pre-render rejection is a
    // defense-in-depth backstop.
    //
    // `text_delta_image_guard`: engine default — `!supports_images() &&
    // session_holds_images()` with the parametrized strings
    // ("chat_tokens_delta_sync is text-only; session currently holds
    // image state" / the `chat_stream_tokens_delta` twin) byte-matches
    // the legacy lfm2 guards.
    //
    // `extra_eos_ids` / `stream_skip_special_tokens` / `stream_emitter`
    // / `wired_limit_bytes` (usize::MAX) / `profiler_label` /
    // `has_live_session` (`!cached_token_history.is_empty()`): engine
    // defaults == legacy lfm2 behaviour (lfm2's loops had no profiler;
    // the default labels only surface when profiling is enabled).

    fn eos_before_emit(&self) -> bool {
        // BOTH legacy streaming loops check EOS before the cancellation
        // check AND before the token's text is detokenized/emitted
        // ("Check stop condition before streaming to avoid leaking EOS
        // text") — which also resolves the EOS+cancel race as "stop".
        true
    }

    fn augment_performance(&self, _profiler: &DecodeProfiler, _metrics: &mut PerformanceMetrics) {
        // No-op override (gemma4 precedent): lfm2 has no MTP heads
        // (acceptance fields stay None) and its legacy metrics never
        // carried `profile_phases`; the default would add
        // profiling-gated extras. Keep the payload byte-stable.
    }

    fn session_holds_images(&self) -> bool {
        // Always `None` for text-only LFM2 in practice; feeds the
        // DEFAULT `text_delta_image_guard` policy.
        self.cached_image_key.is_some()
    }

    fn paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Option<Result<TurnOutput>> {
        // CRITICAL (engine paged-probe contract — see the DELIBERATE FIX
        // note at the probe): lfm2's delta paths NEVER touch the paged
        // adapter. The legacy `chat_tokens_delta_sync` /
        // `chat_stream_tokens_delta_sync` ran the flat eager
        // prefill+decode over `self.caches` even when `paged_adapter`
        // was `Some`. Decline delta turns so the generic flow (eager
        // flat delta) runs, exactly like legacy; fresh turns take the
        // kept paged cores for both sync and streaming.
        if args.is_delta {
            return None;
        }
        Some(self.paged_whole_turn(args))
    }
}

/// Initialize caches matching the layer types.
/// RAII guard that calls `mlx_lfm2_moe_reset()` on drop, tearing down the
/// compiled lfm2 decode globals (caches + offset + inited flag).
///
/// DISTINCT from qwen3.5's `CompiledResetGuard` (which calls
/// `mlx_qwen35_compiled_reset()`) — the two compiled families own separate
/// C++ state and must each reset their own. Ensures the compiled state is
/// always torn down even when the decode loop returns early via `?`, so the
/// next generation never sees stale compiled caches.
struct Lfm2CompiledResetGuard;

impl Drop for Lfm2CompiledResetGuard {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_lfm2_moe_reset();
        }
    }
}

/// RAII guard that calls `mlx_lfm2_paged_reset()` on drop, tearing down the
/// compiled-PAGED lfm2 decode globals (per-layer pools / scales / conv-state,
/// offset, inited flag).
///
/// DISTINCT from [`Lfm2CompiledResetGuard`] (the FLAT path, which calls
/// `mlx_lfm2_moe_reset()`): the flat and paged decode families own strictly
/// separate C++ state and must each reset their own. Like the flat guard, this
/// ensures the compiled-paged globals are always torn down even when the decode
/// loop returns early via `?`, so the next generation never seeds against stale
/// paged pools.
///
/// Armed by `paged_turn_sync_core_inner` when the compiled-paged seed succeeds.
struct Lfm2PagedResetGuard;

impl Drop for Lfm2PagedResetGuard {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_lfm2_paged_reset();
        }
    }
}

/// Initialize the compiled-PAGED lfm2 decode graph from the live post-prefill
/// state — the adapter's per-attention-layer paged KV pool/scale arrays AND the
/// per-conv-layer conv-state arrays already populated by the pure-Rust eager
/// paged prefill (`run_paged_prefill_chunk`).
///
/// Mirrors qwen3.5's `init_paged_dense_compiled_session` (qwen3_5/model.rs) but
/// for lfm2's irregular conv/attn interleave: instead of a modulo `is_linear`
/// test, the per-layer dispatch is driven by the explicit
/// `config.is_attention_layer(i)` map (built into `is_attn` and passed to C++).
///
/// # Layer-index contract
///
/// The C++ FFI (`mlx_lfm2_moe_init_paged`) accepts five per-layer handle arrays
/// of size `num_layers` (absolute decoder count). For each absolute layer `i`:
/// * Attention layer: `k_pool_handles[i]` / `v_pool_handles[i]` /
///   `k_scale_handles[i]` / `v_scale_handles[i]` come from the adapter's
///   `LayerKVPool` at the COMPACT (attention-layer) ordinal `paged_idx`; the
///   `conv_state_handles[i]` slot is null. The compact-ordinal mapping is
///   computed by [`Lfm2Inner::compute_layer_kinds`], the same helper the
///   production eager paged dispatch uses.
/// * Conv layer: `conv_state_handles[i]` is the layer's conv state
///   `[B, l_cache-1, hidden]` from its `Lfm2LayerCache::Conv(ArraysCache)` slot
///   0; the pool/scale slots are null. (No cross-turn conv export is needed —
///   the eager paged path reprefills conv from token 0 each turn — so the
///   compiled graph threads conv state WITHIN a turn only.)
///
/// # Caller contract
///
/// 1. `caches` is fully populated by a prior pure-Rust eager paged prefill.
/// 2. `adapter.block_size() == CPP_PAGED_REQUIRED_BLOCK_SIZE` (16): the compiled
///    paged graph hard-codes `block_size=16` into its `paged_kv_write` /
///    `paged_attention` calls. The caller MUST gate on this before invoking.
/// 3. The C++ weights for this model are still registered (caller verified
///    `mlx_lfm2_get_model_id() == self.model_id` and holds the read lock).
///
/// `prefill_offset` is the global token cursor the compiled paged graph's
/// `g_lfm2_paged_offset_int` starts incrementing from; the caller passes
/// `adapter.current_token_count() as i32`. Every attention layer shares this
/// single adapter cursor (paged attention has ONE logical sequence position, not
/// a per-layer KVCache offset like the flat path), so there is no per-layer
/// offset to disagree — the consistency invariant the flat seed checks is
/// structurally guaranteed here. We still validate the cursor is non-negative
/// and that the cache vector length matches the config so a corrupt session
/// falls back to native rather than seeding a wrong position.
///
/// On any failure (cache-length mismatch, missing pool/scale handle, missing
/// conv state, or the C++ FFI returning a non-zero status), returns `Err` so the
/// caller falls back to the pure-Rust eager paged decode path.
fn init_lfm2_paged_compiled_session(
    config: &Lfm2Config,
    caches: &[Lfm2LayerCache],
    paged_adapter: &PagedKVCacheAdapter,
    prefill_offset: i32,
) -> Result<()> {
    let num_layers_us = config.num_hidden_layers as usize;
    if caches.len() != num_layers_us {
        return Err(Error::from_reason(format!(
            "init_lfm2_paged_compiled_session: caches.len()={} but config.num_hidden_layers={}",
            caches.len(),
            num_layers_us
        )));
    }
    if prefill_offset < 0 {
        return Err(Error::from_reason(format!(
            "init_lfm2_paged_compiled_session: negative prefill_offset={prefill_offset}; refusing \
             to seed a wrong position. Caller must fall back to the pure-Rust paged path."
        )));
    }

    // Per-layer attn/conv dispatch — built DYNAMICALLY from config (lfm2 mixes
    // conv/attn irregularly; never a modulo/hardcoded pattern). Mirrors the flat
    // seed's `is_attn` construction.
    let is_attn: Vec<i32> = (0..num_layers_us)
        .map(|i| i32::from(config.is_attention_layer(i)))
        .collect();

    // Compact-ordinal mapping (paged_idx counts only attention layers), the same
    // helper the eager paged forward uses.
    let layer_kinds = compute_layer_kinds_for(config, num_layers_us);

    let mut k_pool_handles: Vec<*mut mlx_sys::mlx_array> =
        vec![std::ptr::null_mut(); num_layers_us];
    let mut v_pool_handles: Vec<*mut mlx_sys::mlx_array> =
        vec![std::ptr::null_mut(); num_layers_us];
    let mut k_scale_handles: Vec<*mut mlx_sys::mlx_array> =
        vec![std::ptr::null_mut(); num_layers_us];
    let mut v_scale_handles: Vec<*mut mlx_sys::mlx_array> =
        vec![std::ptr::null_mut(); num_layers_us];
    let mut conv_state_handles: Vec<*mut mlx_sys::mlx_array> =
        vec![std::ptr::null_mut(); num_layers_us];

    // Hold every wrapping `MxArray` alive across the FFI call so the C++ side has
    // time to copy each handle into its own globals before the temporaries drop.
    let mut held_arrays: Vec<MxArray> = Vec::with_capacity(num_layers_us * 4);

    for (i, kind) in layer_kinds.iter().enumerate() {
        match kind {
            Lfm2LayerKind::FullAttention { paged_idx } => {
                let k_arr = paged_adapter.key_pool_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_lfm2_paged_compiled_session: key_pool_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                let v_arr = paged_adapter.value_pool_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_lfm2_paged_compiled_session: value_pool_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                let ks_arr = paged_adapter.k_scale_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_lfm2_paged_compiled_session: k_scale_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                let vs_arr = paged_adapter.v_scale_array(*paged_idx).map_err(|e| {
                    Error::from_reason(format!(
                        "init_lfm2_paged_compiled_session: v_scale_array(layer={i}, \
                         paged_idx={paged_idx}): {e}",
                    ))
                })?;
                k_pool_handles[i] = k_arr.as_raw_ptr();
                v_pool_handles[i] = v_arr.as_raw_ptr();
                k_scale_handles[i] = ks_arr.as_raw_ptr();
                v_scale_handles[i] = vs_arr.as_raw_ptr();
                held_arrays.push(k_arr);
                held_arrays.push(v_arr);
                held_arrays.push(ks_arr);
                held_arrays.push(vs_arr);
            }
            Lfm2LayerKind::Conv => {
                // Conv state lives in `self.caches[i]` (NOT the adapter): the
                // eager paged prefill writes it into the layer's
                // `Lfm2LayerCache::Conv(ArraysCache)` slot 0 as
                // `[B, l_cache-1, hidden]`.
                let conv_cache = match &caches[i] {
                    Lfm2LayerCache::Conv(c) => c,
                    Lfm2LayerCache::Attention(_) => {
                        return Err(Error::from_reason(format!(
                            "init_lfm2_paged_compiled_session: layer {i} is Conv by config but \
                             cache slot is Attention",
                        )));
                    }
                };
                let state = conv_cache.get(0).ok_or_else(|| {
                    Error::from_reason(format!(
                        "init_lfm2_paged_compiled_session: layer {i} conv_state not populated; \
                         the eager paged prefill must run before C++ paged init",
                    ))
                })?;
                conv_state_handles[i] = state.as_raw_ptr();
            }
        }
    }

    let status = unsafe {
        mlx_sys::mlx_lfm2_moe_init_paged(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim(),
            config.rope_theta as f32,
            config.norm_eps as f32,
            config.conv_l_cache,
            config.num_experts.unwrap_or(0),
            config.num_experts_per_tok.unwrap_or(0),
            config.num_dense_layers.unwrap_or(0),
            i32::from(config.norm_topk_prob.unwrap_or(true)),
            i32::from(config.use_expert_bias.unwrap_or(true)),
            i32::from(config.tie_embedding),
            i32::from(config.conv_bias),
            // max_kv_len is accepted for ABI symmetry with the flat init; the
            // paged graph sizes its pools from the adapter, so any non-negative
            // value is fine here. Pass the global position so the field carries a
            // meaningful sequence length rather than 0.
            prefill_offset,
            1, // batch_size
            crate::models::qwen3_5::model::CPP_PAGED_REQUIRED_BLOCK_SIZE as i32,
            is_attn.as_ptr(),
            k_pool_handles.as_mut_ptr(),
            v_pool_handles.as_mut_ptr(),
            k_scale_handles.as_mut_ptr(),
            v_scale_handles.as_mut_ptr(),
            conv_state_handles.as_mut_ptr(),
            prefill_offset,
        )
    };

    // Drop the held wrappers only AFTER the FFI call returns: the C++ init copies
    // each handle into its globals (and force-evals the attn pools), so by here
    // it no longer needs our temporaries alive.
    drop(held_arrays);

    // The C++ side returns 0 on success, -1 on failure (missing handle, bad
    // block_size, dtype mismatch, or any caught exception). On failure it leaves
    // `g_lfm2_paged_inited` cleared so a subsequent `mlx_lfm2_moe_forward_paged`
    // would null its logits; surfacing the status here lets the caller fall back
    // to the pure-Rust paged path before any decode-step FFI is dispatched.
    if status != 0 {
        // Belt-and-suspenders: the C++ init now clears the paged globals on every
        // failure path itself, but a failed init mutates process-wide GPU-backed
        // state, so we also fire the reset from the Rust side. `mlx_lfm2_paged_reset`
        // is idempotent (clears config/is_attn/pool/scale vectors + offset + inited
        // flag), so a double reset is harmless. This guarantees no stale imported
        // array handles / populated pool vectors leak when the seed aborts and the
        // caller drops back to the pure-Rust paged path (no `Lfm2PagedResetGuard`
        // is armed on the Err branch).
        unsafe { mlx_sys::mlx_lfm2_paged_reset() };
        return Err(Error::from_reason(format!(
            "init_lfm2_paged_compiled_session: mlx_lfm2_moe_init_paged returned status={status} \
             (expected 0); see stderr for the C++ diagnostic. Caller must fall back to the \
             pure-Rust paged path."
        )));
    }

    Ok(())
}

/// Single-token decode step using the C++ compiled-PAGED forward pass.
///
/// Per-step wrapper mirroring qwen3.5's `forward_dense_cpp_paged`: threads the
/// paged-attention inputs (offset_arr, block_table, slot_mapping,
/// num_valid_tokens, num_valid_blocks, seq_lens) so each attention layer writes
/// its new K/V into the adapter's paged Metal pool via `paged_kv_write` and
/// gathers via `paged_attention`, while conv layers thread their state through
/// the compiled graph's cache slots.
///
/// Caller contract (enforced in the decode loop, not here):
/// * `init_lfm2_paged_compiled_session` has been called this turn (so
///   `g_lfm2_paged_inited == true`).
/// * `adapter.record_tokens(&[token_id])` has advanced the cursor (and lazily
///   allocated any new block) for this step.
/// * `inputs` was just built via
///   `adapter.build_paged_attention_inputs(1, 1, max_blocks_per_seq)`.
///
/// On any FFI failure (`output_logits == null`) returns `Err` so the dispatcher
/// can fall back to the pure-Rust paged decode (first step) or propagate (after
/// a compiled step has mutated the C++ paged globals).
fn forward_lfm2_cpp_paged(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    inputs: &crate::transformer::paged_attention_inputs::PagedAttentionInputs,
) -> Result<MxArray> {
    let mut output_ptr: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut cache_offset_out: i32 = 0;
    unsafe {
        mlx_sys::mlx_lfm2_moe_forward_paged(
            input_ids.as_raw_ptr(),
            embedding_weight.as_raw_ptr(),
            inputs.offset_arr.as_raw_ptr(),
            inputs.block_table.as_raw_ptr(),
            inputs.slot_mapping.as_raw_ptr(),
            inputs.num_valid_tokens.as_raw_ptr(),
            inputs.num_valid_blocks.as_raw_ptr(),
            inputs.seq_lens.as_raw_ptr(),
            &mut output_ptr,
            &mut cache_offset_out,
        );
    }

    if output_ptr.is_null() {
        return Err(Error::from_reason(
            "lfm2 compiled paged forward step returned null — check stderr for diagnostic. \
             (Common causes: g_lfm2_paged_inited = false, slot_mapping shape != [1], \
             input_ids size != 1, or weights cleared by another model load.)",
        ));
    }

    MxArray::from_handle(output_ptr, "lfm2 compiled paged forward logits")
}

/// Free-function form of [`Lfm2Inner::compute_layer_kinds`] usable without a
/// `self` borrow (the paged compiled session init only has `&Lfm2Config` +
/// `&[Lfm2LayerCache]`, not the whole `Lfm2Inner`). Identical mapping:
/// `FullAttention { paged_idx }` for attention layers (paged_idx counts only
/// those, in original order) and `Conv` otherwise. Called by
/// `init_lfm2_paged_compiled_session` (decode-loop wiring).
fn compute_layer_kinds_for(config: &Lfm2Config, num_layers: usize) -> Vec<Lfm2LayerKind> {
    let mut kinds = Vec::with_capacity(num_layers);
    let mut paged_idx: u32 = 0;
    for i in 0..num_layers {
        if config.is_attention_layer(i) {
            kinds.push(Lfm2LayerKind::FullAttention { paged_idx });
            paged_idx += 1;
        } else {
            kinds.push(Lfm2LayerKind::Conv);
        }
    }
    kinds
}

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
    /// Dedicated model thread owning `Lfm2Inner`. lfm2 is chat-only (no
    /// training/generate variants), so the thread dispatches the
    /// model-neutral [`ChatCmd`] directly via
    /// `engine::cmd::handle_chat_cmd::<Lfm2Inner>` — no per-family
    /// command enum (S11).
    pub(crate) thread: crate::model_thread::ModelThread<ChatCmd>,
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
        crate::model_thread::send_and_block(&self.thread, |reply| ChatCmd::ResetCaches { reply })
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

        crate::model_thread::send_and_await(&self.thread, |reply| ChatCmd::SessionStart {
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

        crate::model_thread::send_and_await(&self.thread, |reply| ChatCmd::SessionContinue {
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
    /// `is_error` is the structured tool-error signal. When `Some(true)`,
    /// the renderer prepends the shared
    /// [`crate::tokenizer::TOOL_ERROR_MARKER`] inside the
    /// `<|im_start|>tool` block so the model receives a clear text-level
    /// cue. `None` / `Some(false)` keep the wire bytes byte-equal to the
    /// pre-feature output.
    ///
    /// Requires a live session started via `chatSessionStart`.
    #[napi]
    pub async fn chat_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
        is_error: Option<bool>,
    ) -> Result<ChatResult> {
        let config = config.unwrap_or_default();

        crate::model_thread::send_and_await(&self.thread, |reply| ChatCmd::SessionContinueTool {
            tool_call_id,
            content,
            is_error,
            config,
            reply,
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

        let plumbing = start_chat_stream(callback);
        self.thread.send(ChatCmd::StreamSessionStart {
            messages,
            config,
            stream_tx: plumbing.stream_tx,
            cancelled: plumbing.cancelled,
        })?;

        Ok(plumbing.handle)
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

        let plumbing = start_chat_stream(callback);
        self.thread.send(ChatCmd::StreamSessionContinue {
            user_message,
            images,
            config,
            stream_tx: plumbing.stream_tx,
            cancelled: plumbing.cancelled,
        })?;

        Ok(plumbing.handle)
    }

    /// Streaming variant of `chatSessionContinueTool`.
    ///
    /// `is_error` mirrors the non-streaming entry point — when
    /// `Some(true)`, the renderer prepends the shared
    /// [`crate::tokenizer::TOOL_ERROR_MARKER`] inside the
    /// `<|im_start|>tool` block.
    #[napi(
        ts_args_type = "toolCallId: string, content: string, config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void, isError?: boolean | null | undefined"
    )]
    pub async fn chat_stream_session_continue_tool(
        &self,
        tool_call_id: String,
        content: String,
        config: Option<ChatConfig>,
        callback: ThreadsafeFunction<ChatStreamChunk, ()>,
        is_error: Option<bool>,
    ) -> Result<ChatStreamHandle> {
        let config = config.unwrap_or_default();

        let plumbing = start_chat_stream(callback);
        self.thread.send(ChatCmd::StreamSessionContinueTool {
            tool_call_id,
            content,
            is_error,
            config,
            stream_tx: plumbing.stream_tx,
            cancelled: plumbing.cancelled,
        })?;

        Ok(plumbing.handle)
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
        self.thread.send(ChatCmd::StreamSessionStart {
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
        self.thread.send(ChatCmd::StreamSessionContinue {
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
    //! prefix relation). The engine session core (and the paged turn
    //! cores) then classify that value plus the
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
mod tool_delta_marker_tests {
    //! Guard the structured `is_error` channel on LFM2's tool-result
    //! wire format. The shared
    //! [`crate::tokenizer::TOOL_ERROR_MARKER`] must be injected inside
    //! the `<|im_start|>tool` block only when the caller passes
    //! `Some(true)`. `None` and `Some(false)` keep the output
    //! byte-equal to the pre-feature behavior — guarding both the hot
    //! (successful) path and the explicit-false path against
    //! accidental drift.

    use super::build_lfm2_tool_delta_text;
    use crate::tokenizer::TOOL_ERROR_MARKER;

    #[test]
    fn injects_marker_when_is_error_true() {
        let payload = "boom: connection refused";
        let rendered = build_lfm2_tool_delta_text(payload, Some(true));
        let expected_inner = format!("{TOOL_ERROR_MARKER}{payload}");
        assert!(
            rendered.contains(&expected_inner),
            "expected error marker inside <|im_start|>tool block; got:\n{rendered}",
        );
        // Wrapper integrity stays correct on the marked path so we
        // don't ship a malformed delta that only the unflagged path
        // renders right.
        assert!(
            rendered.contains("<|im_start|>tool\n"),
            "tool block opener missing"
        );
        assert!(rendered.contains("<|im_end|>"), "im_end closer missing");
        assert!(
            rendered.contains("<|im_start|>assistant\n"),
            "assistant opener missing"
        );
    }

    #[test]
    fn skips_marker_when_is_error_none() {
        let payload = "{\"temperature\": 72}";
        let rendered = build_lfm2_tool_delta_text(payload, None);
        assert!(
            !rendered.contains(TOOL_ERROR_MARKER),
            "marker leaked into unflagged delta:\n{rendered}",
        );
        assert!(
            rendered.contains(payload),
            "original content missing from delta:\n{rendered}",
        );
    }

    #[test]
    fn skips_marker_when_is_error_some_false() {
        let payload = "ok";
        let rendered = build_lfm2_tool_delta_text(payload, Some(false));
        assert!(
            !rendered.contains(TOOL_ERROR_MARKER),
            "marker leaked into Some(false) delta:\n{rendered}",
        );
    }

    #[test]
    fn does_not_remark_content_that_resembles_marker() {
        // The structured channel removes the collision concern: a
        // successful tool result whose literal content begins with the
        // marker text must NOT double-prefix the marker on its way
        // through the renderer.
        let suspicious = format!("{TOOL_ERROR_MARKER}this is a successful payload");
        let rendered = build_lfm2_tool_delta_text(&suspicious, None);
        let occurrences = rendered.matches(TOOL_ERROR_MARKER).count();
        assert_eq!(
            occurrences, 1,
            "marker count should be 1 (the original literal); got {occurrences} in:\n{rendered}",
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
            intermediate_size: None,
            moe_intermediate_size: None,
            num_experts: None,
            num_experts_per_tok: None,
            num_dense_layers: None,
            norm_topk_prob: Some(true),
            use_expert_bias: Some(true),
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

    /// **Smoke test for `paged_turn_sync_core` helpers**. Without real
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
    fn test_lfm2_paged_turn_sync_core_smoke_via_helpers() {
        use crate::array::{DType, MxArray};

        let cfg = paged_tiny_config(Some(true));
        let mut inner = match Lfm2Inner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!(
                        "skipping test_lfm2_paged_turn_sync_core_smoke_via_helpers (no Metal): {msg}"
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
                    attn.q_proj.set_weight(&cast(&w), "q_proj").expect("set q");
                    let w = attn.k_proj.get_weight();
                    attn.k_proj.set_weight(&cast(&w), "k_proj").expect("set k");
                    let w = attn.v_proj.get_weight();
                    attn.v_proj.set_weight(&cast(&w), "v_proj").expect("set v");
                    let w = attn.out_proj.get_weight();
                    attn.out_proj
                        .set_weight(&cast(&w), "out_proj")
                        .expect("set o");
                    let w = attn.q_layernorm.get_weight();
                    attn.q_layernorm.set_weight(&cast(&w)).expect("set qn");
                    let w = attn.k_layernorm.get_weight();
                    attn.k_layernorm.set_weight(&cast(&w)).expect("set kn");
                }
                OperatorType::Conv(conv) => {
                    let w = conv.conv.get_weight();
                    conv.conv.set_weight(&cast(&w)).expect("set conv_w");
                    let w = conv.in_proj.get_weight();
                    conv.in_proj
                        .set_weight(&cast(&w), "in_proj")
                        .expect("set in_proj");
                    let w = conv.out_proj.get_weight();
                    conv.out_proj
                        .set_weight(&cast(&w), "out_proj")
                        .expect("set out_proj");
                }
            }

            let mlp = layer
                .dense_mlp_mut()
                .expect("paged_tiny_config layers are all dense MLPs");
            let w = mlp.get_gate_proj_weight();
            mlp.set_gate_proj_weight(&cast(&w)).expect("set gate");
            let w = mlp.get_up_proj_weight();
            mlp.set_up_proj_weight(&cast(&w)).expect("set up");
            let w = mlp.get_down_proj_weight();
            mlp.set_down_proj_weight(&cast(&w)).expect("set down");
        }

        // Drive the adapter lifecycle the same way `paged_turn_sync_core`
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
                    eprintln!("skipping test_lfm2_paged_turn_sync_core_smoke_via_helpers: {msg}");
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
                            "skipping test_lfm2_paged_turn_sync_core_smoke_via_helpers: {msg}"
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
