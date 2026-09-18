//! K2-Horizon dense causal LM — pure standard-KV transformer.
//!
//! `K2HorizonForCausalLM` (IFM K2-Horizon-7B): 36 dense decoder layers,
//! GQA 32/8, head_dim 128, SwiGLU MLP, grouped RMSNorm (`n_groups=4`),
//! full RoPE (theta 1e7), untied `lm_head`. No MoE, no conv/recurrent
//! sidecar, no MTP head — structurally the qwen3 shape, implemented on
//! the lfm2-style eager `KVCache` + `LinearProj` substrate so mxfp8
//! checkpoints load without a fused forward.
//!
//! Execution paths:
//!   * FLAT   — `Vec<KVCache>` + `scaled_dot_product_attention(_causal)`;
//!     qwen3's all-or-nothing prefix check with the pure-KV
//!     exact-match rewind (`trim` + re-forward last token).
//!   * PAGED  — `PagedKVCacheAdapter` (block-paged prefix cache), the
//!     qwen3 token-accounting convention (`record_tokens` before
//!     forward, `reconcile_paged_request_tokens` drop-last).
//!   * SCHED  — generic `HybridStepExecutor` batched paged decode
//!     (pure-KV: no per-sequence out-of-band state).

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::info;

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::ThinkingPolicy;
use crate::engine::backend::{
    ChatBackend, DecodeStep, FinalizeArgs, PagedBackend, ResetScope, SaveStateArgs,
    ThinkEndResolution, ThinkingSetup, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::ChatCmd;
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridSchedulerState, HybridStepExecutor, ScheduledPrefixAdmission,
    ScheduledRestoreResult, scheduler_max_num_seqs_for, scheduler_per_seq_context,
};
use crate::engine::paged_epilogue::{
    FinalTokenPolicy, SimplePagedPrefix, abort_single_adapter_turn, finalize_single_adapter_turn,
    prime_single_adapter_prefix, reconcile_paged_surplus, save_paged_token_history,
};
use crate::engine::paged_stepper::{EvalPolicy, PagedStepModel, PagedStepper};
use crate::engine::plan::{ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle};
use crate::models::forward as fwd;
use crate::nn::{Embedding, GroupedRMSNorm, Linear};
use crate::profiling::PerformanceMetrics;
use crate::stream::Stream;
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::transformer::KVCache;
use crate::transformer::paged_kv_cache_adapter::{
    PagedKVCacheAdapter, PagedRestorePoll, PagedRestoreTicket, PagedTurnAdmission, SeqId,
};

use super::config::K2HorizonConfig;
use super::layer::K2DecoderLayer;

/// Reasoning close-tag strings emitted by the K2 chat template, indexed by
/// `reasoning_effort`. `high` (and the template's `default('high')`) uses
/// `</ifm|think>`; `medium` uses `</ifm|think_fast>`; `low` uses
/// `</ifm|think_faster>`. Unsupported efforts raise in the template itself,
/// so this map never sees them on a successful render.
pub(crate) const K2_THINK_END_HIGH: &str = "</ifm|think>";
pub(crate) const K2_THINK_END_MEDIUM: &str = "</ifm|think_fast>";
pub(crate) const K2_THINK_END_LOW: &str = "</ifm|think_faster>";

/// K2 reasoning tag pairs `(open, close)` for the tag-parameterized tools
/// scrubbers. Order matters for history replay: the template always replays
/// `reasoning_content` inside the canonical `<ifm|think>` family.
pub(crate) const K2_REASONING_PAIRS: &[(&str, &str)] = &[
    ("<ifm|think>", "</ifm|think>"),
    ("<ifm|think_fast>", "</ifm|think_fast>"),
    ("<ifm|think_faster>", "</ifm|think_faster>"),
];

/// Project the generic effort vocabulary onto K2's valid set.
///
/// K2's chat template accepts exactly `high | medium | low` and raises on
/// any other DEFINED `reasoning_effort` (including `"none"`), while an
/// UNSET one falls to `default('high')`. The generic layer, however,
/// whitelists `none|minimal|low|medium|high|xhigh|max` through to render,
/// so pi's "reasoning unset → 'none'" default would crash the turn at
/// template render. Normalize at every effort reader instead:
///
///   * `'none' | 'minimal' | 'low'` → `'low'` — K2 cannot disable
///     thinking; `think_faster` is the least reasoning it emits (the
///     LFM2 `budget-0` analog), which honors "as little thinking as
///     possible" better than forcing `high`.
///   * `'xhigh' | 'max'` → `'high'` (clamp to K2's ceiling).
///   * unset / foreign → `None`, so the template's own `default('high')`
///     decides; [`K2_THINK_END_HIGH`] stays the matching close tag.
///
/// Applied inside `render_prompt` / `thinking_setup` /
/// `think_end_for_turn`, so render ctx, budget, and the tracker's close
/// token can never disagree on the SAME turn.
pub(crate) fn normalize_k2_effort(effort: Option<&str>) -> Option<String> {
    match effort {
        Some("none") | Some("minimal") | Some("low") => Some("low".to_string()),
        Some("medium") => Some("medium".to_string()),
        Some("high") | Some("xhigh") | Some("max") => Some("high".to_string()),
        _ => None,
    }
}

/// Commands owned by the K2 scheduler thread. Chat variants are lifted from
/// the model-neutral API; there is no family-specific payload (`Infallible`).
pub(crate) type K2Cmd = crate::engine::model_command::ModelCommand;

/// K2 scheduler state: the generic hybrid (paged-KV) continuous-batching
/// driver over `K2Inner`.
pub(crate) type K2SchedulerState = HybridSchedulerState<K2Inner>;

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership.
pub(crate) struct K2Inner {
    pub(crate) config: K2HorizonConfig,
    /// The in-flight turn's cooperative-cancel flag (installed by the
    /// session wrappers via `ChatBackend::set_turn_cancel_flag`; polled at
    /// flat `chunked_prefill` and paged chunk boundaries).
    pub(crate) turn_cancel: Option<Arc<AtomicBool>>,
    pub(crate) embed_tokens: Embedding,
    pub(crate) layers: Vec<K2DecoderLayer>,
    /// Final norm (`model.norm.weight`) — K2's grouped RMSNorm like the
    /// per-layer norms.
    pub(crate) norm: GroupedRMSNorm,
    /// Untied output projection (`lm_head.weight`). `LinearProj` so the
    /// persistence layer may install a quantized backend; the shipped
    /// checkpoint keeps it dense.
    pub(crate) lm_head: crate::models::quantized_linear::LinearProj,
    /// Flat-path KV caches (one per layer). Committed state lives here
    /// across turns; a failed turn resets them via `fail_closed_flat_turn`
    /// → `reset_caches`.
    pub(crate) caches: Vec<KVCache>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    /// Cached token history for KV cache reuse across chat-session turns.
    pub(crate) cached_token_history: Vec<u32>,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache). Every
    /// layer is full attention, so the pool has `num_hidden_layers` slots
    /// and the layer ordinal IS the adapter index.
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    /// Sampling + stop-token defaults parsed from `generation_config.json`.
    gen_defaults: crate::engine::ModelGenerationDefaults,
    /// Exact-match rewind arm for [`ChatBackend::verify_cache_prefix`]:
    /// when the new prompt equals the committed history byte-for-byte the
    /// flat path rewinds one slot and re-forwards the final token (the
    /// qwen3 pure-KV convention — `KVCache::trim` makes it safe).
    pending_exact_match_rewind: Cell<bool>,
}

/// Allocate the flat-path cache stack (one `KVCache` per layer).
pub(crate) fn init_caches(config: &K2HorizonConfig) -> Vec<KVCache> {
    (0..config.num_hidden_layers)
        .map(|_| KVCache::new())
        .collect()
}

/// Force-materialize every live flat-cache array (post-prefill sync).
fn eval_kv_caches(caches: &[KVCache]) -> Result<()> {
    fwd::eval_layer_caches(caches)
}

impl K2Inner {
    /// Construct `K2Inner` from the parsed config. Weight tensors are
    /// placeholders until `persistence::apply_weights` installs them.
    pub(crate) fn new(config: K2HorizonConfig) -> Result<Self> {
        let embed_tokens = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;

        let layers = (0..config.num_hidden_layers)
            .map(|_| K2DecoderLayer::new(&config))
            .collect::<Result<Vec<_>>>()?;

        let norm = GroupedRMSNorm::new(
            config.hidden_size as i64,
            config.layernorm_num_groups as i64,
            config.norm_eps,
        )?;

        // Untied lm_head (K2 ships `tie_word_embeddings: false`). Dense
        // `Linear` placeholder; persistence may upgrade to mxfp8.
        let lm_head = crate::models::quantized_linear::LinearProj::Standard(Linear::new(
            config.hidden_size as u32,
            config.vocab_size as u32,
            Some(false),
        )?);

        // Block-paged KV adapter — default ON (pure standard-KV model; the
        // Metal-only write/gather kernels mean non-Metal builds leave it
        // None and fall through to the flat path). Mirrors qwen3's
        // construction; sized adaptively after weights materialize.
        let paged_adapter = if config.use_block_paged_cache.unwrap_or(true)
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let block_size = config.paged_block_size.unwrap_or(16);
            let gpu_memory_mb = config.paged_cache_memory_mb.unwrap_or(2048);
            let pa_config = mlx_paged_attn::PagedAttentionConfig {
                block_size,
                gpu_memory_mb,
                head_size: config.head_dim() as u32,
                num_kv_heads: config.num_key_value_heads as u32,
                num_layers: config.num_hidden_layers as u32,
                use_fp8_cache: Some(false),
                max_seq_len: Some(config.max_position_embeddings as u32),
                max_batch_size: Some(scheduler_max_num_seqs_for(32) as u32),
            };

            let num_blocks = pa_config.calculate_num_blocks();
            if num_blocks == 0 {
                return Err(napi::Error::from_reason(format!(
                    "Block-paged adapter: gpu_memory_mb={gpu_memory_mb} too small to hold any \
                     blocks (head_size={}, num_kv_heads={}, block_size={}, num_layers={})",
                    pa_config.head_size,
                    pa_config.num_kv_heads,
                    pa_config.block_size,
                    pa_config.num_layers,
                )));
            }

            let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
                num_blocks, num_blocks, block_size,
            )));
            let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
            let pool =
                mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, num_blocks, cache_dtype)
                    .map_err(|e| {
                        napi::Error::from_reason(format!(
                            "Failed to construct LayerKVPool for block-paged adapter: {e}"
                        ))
                    })?;
            let adapter =
                PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                    napi::Error::from_reason(format!(
                        "Failed to construct PagedKVCacheAdapter: {e}"
                    ))
                })?;
            info!(
                "K2 block-paged adapter enabled: num_blocks={num_blocks}, block_size={block_size}, \
                 gpu_memory_mb={gpu_memory_mb}, cache_dtype=BFloat16"
            );
            Some(adapter)
        } else {
            None
        };

        Ok(Self {
            caches: init_caches(&config),
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            tokenizer: None,
            cached_token_history: Vec::new(),
            paged_adapter,
            turn_cancel: None,
            gen_defaults: crate::engine::ModelGenerationDefaults::default(),
            pending_exact_match_rewind: Cell::new(false),
        })
    }

    /// Rebuild the provisional constructor-time pool after weights are
    /// materialized, using the scheduler policy and live Metal budget
    /// (qwen3 convention: the load-time probe can finally see the model
    /// weights, so the selected pool is capped against the real working
    /// set instead of double-budgeting them).
    pub(crate) fn size_paged_pool_after_weight_load(&mut self) -> Result<()> {
        if self.paged_adapter.is_none() {
            return Ok(());
        }
        // An explicit pool size is an operator cap, not a sizing hint — the
        // constructor already built and co-resident-loaded that exact pool.
        if self.config.paged_cache_memory_mb.is_some() {
            return Ok(());
        }
        self.paged_adapter = None;

        let block_size = self.config.paged_block_size.unwrap_or(16);
        let trained_context = u32::try_from(self.config.max_position_embeddings)
            .unwrap_or(1)
            .max(1);
        let per_seq_context = trained_context.min(scheduler_per_seq_context());
        let requested_tokens =
            per_seq_context.saturating_mul(scheduler_max_num_seqs_for(32) as u32);
        let requested_blocks = requested_tokens.div_ceil(block_size).max(1);
        let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
        let sizing = mlx_paged_attn::profile::load_time_pool_sizing(
            requested_blocks,
            self.config.num_hidden_layers as u32,
            self.config.num_key_value_heads as u32,
            self.config.head_dim() as u32,
            block_size,
            cache_dtype,
        )
        .map_err(|error| {
            Error::from_reason(format!(
                "K2 adaptive paged cache sizing failed safely; refusing an uncapped pool request: {error}"
            ))
        })?;
        let selected_mb = sizing.selected_bytes.div_ceil(1024 * 1024).max(1) as u32;
        let pa_config = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            gpu_memory_mb: selected_mb,
            head_size: self.config.head_dim() as u32,
            num_kv_heads: self.config.num_key_value_heads as u32,
            num_layers: self.config.num_hidden_layers as u32,
            use_fp8_cache: Some(false),
            max_seq_len: Some(per_seq_context),
            max_batch_size: Some(scheduler_max_num_seqs_for(32) as u32),
        };
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            sizing.selected_blocks,
            sizing.selected_blocks,
            block_size,
        )));
        let pool = mlx_paged_attn::LayerKVPool::new(
            pa_config,
            sizing.selected_blocks,
            sizing.selected_blocks,
            cache_dtype,
        )
        .map_err(|error| Error::from_reason(format!("Failed to construct K2 KV pool: {error}")))?;
        self.paged_adapter = Some(
            PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|error| {
                Error::from_reason(format!("Failed to construct K2 paged adapter: {error}"))
            })?,
        );
        info!(
            "K2 scheduler pool enabled: requested_blocks={}, selected_blocks={}, bytes={:.2} GiB, per_seq_context={}, max_num_seqs={}",
            requested_blocks,
            sizing.selected_blocks,
            sizing.selected_bytes as f64 / (1u64 << 30) as f64,
            per_seq_context,
            scheduler_max_num_seqs_for(32),
        );
        Ok(())
    }

    pub(crate) fn paged_pool_allocated_bytes(&self) -> Result<u64> {
        self.paged_adapter
            .as_ref()
            .map(PagedKVCacheAdapter::pool_allocated_bytes)
            .transpose()
            .map(|bytes| bytes.unwrap_or(0))
            .map_err(Error::from_reason)
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Install sampling + stop-token defaults parsed from the checkpoint's
    /// `generation_config.json`.
    pub(crate) fn set_gen_defaults(&mut self, defaults: crate::engine::ModelGenerationDefaults) {
        self.gen_defaults = defaults;
    }

    // ================= Cold-tier (pure-KV) persistence =================
    //
    // K2 has no out-of-band recurrent state, so the cold context is the
    // qwen3 shape only: paged adapter + model fingerprint + loaded-weight
    // witness, no sidecar policy.

    /// Build the cold-tier context for this model dir. `None` (fail-open)
    /// when the paged adapter is absent, the global cold cache is off, or
    /// the fingerprint fails. `weights` pins this call after
    /// `materialize_weights` (qwen3 convention).
    pub(crate) fn build_cold_tier_context(
        &self,
        model_path: &str,
        weights: &crate::array::memory::WeightsResident,
    ) -> Option<crate::transformer::paged_kv_cache_adapter::ColdTierContext> {
        let adapter = self.paged_adapter.as_ref()?;
        let manager = crate::cold_tier::global_cold_cache()?;
        let config_json = serde_json::to_vec(&self.config).ok();
        let pool = adapter.layer_kv_pool();
        let geometry = crate::cold_tier::ColdTierGeometry {
            block_size: pool.block_size() as u64,
            num_layers: pool.num_layers() as u64,
            num_kv_heads: pool.config().num_kv_heads as u64,
            head_size: pool.config().head_size as u64,
            cache_dtype: format!("{:?}", pool.cache_dtype()),
        };
        match crate::cold_tier::build_model_fingerprint(
            "k2_horizon",
            model_path,
            config_json.as_deref(),
            &geometry,
            weights,
        ) {
            Some(fingerprint) => Some(
                crate::transformer::paged_kv_cache_adapter::ColdTierContext {
                    manager,
                    fingerprint,
                    // Pure standard-KV: the pool already holds every piece of
                    // per-token state the forward carries between turns — no
                    // sidecar to reconcile (the qwen3 `None` control).
                    sidecar_policy: None,
                },
            ),
            None => {
                tracing::warn!(
                    "cold-tier persistence disabled for {model_path}: could not establish a \
                     content fingerprint (unreadable or missing weight shard)"
                );
                None
            }
        }
    }

    /// Attach a previously-built cold-tier context to the paged adapter.
    /// Split from [`Self::build_cold_tier_context`] so the caller can verify
    /// shard identity is still stable between fingerprint read and commit.
    pub(crate) fn attach_cold_tier(
        &mut self,
        ctx: crate::transformer::paged_kv_cache_adapter::ColdTierContext,
        _weights: &crate::array::memory::WeightsResident,
    ) {
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter.set_cold_tier(ctx);
        }
    }

    // ============================ Flat path ============================

    /// Forward pass through the full model on the flat `KVCache` stack.
    /// Returns logits `[B, T, vocab]`.
    pub(crate) fn forward(&mut self, input_ids: &MxArray) -> Result<MxArray> {
        let h = fwd::forward_body_normed(
            input_ids,
            &self.embed_tokens,
            &mut self.layers,
            &mut self.caches,
            &self.norm,
            |layer, h, caches, i| layer.forward(h, None, Some(&mut caches[i])),
        )?;
        self.project_logits(&h)
    }

    /// Final logits projection. Untied checkpoints (the shipped card)
    /// use the loaded `lm_head` — `LinearProj::forward` dispatches dense
    /// or quantized transparently; `tie_word_embeddings: true` routes
    /// through the shared embedding table (`as_linear` covers dense and
    /// packed embeds alike), matching the qwen3/lfm2 convention.
    fn project_logits(&self, hidden_states: &MxArray) -> Result<MxArray> {
        fwd::project_logits(
            hidden_states,
            (!self.config.tie_word_embeddings).then_some(&self.lm_head),
            &self.embed_tokens,
        )
    }

    /// Chunked flat prefill (2048-token chunks, per-chunk cache evals).
    /// Returns last-token logits `[1, vocab]` (lm_head over the full final
    /// chunk, sliced after — the chunk-local logits are cheap enough not to
    /// warrant a pre-projection slice here).
    fn chunked_prefill(&mut self, prompt: &MxArray, generation_stream: Stream) -> Result<MxArray> {
        // Exact-match rewind: the session re-prefills the final history
        // token onto a cache whose last slot was trimmed, so the next
        // sampled token lands on the same boundary the committed history
        // recorded.
        if self.pending_exact_match_rewind.replace(false) {
            for cache in self.caches.iter_mut() {
                cache.trim(cache.get_offset() - 1);
            }
        }
        fwd::chunked_prefill(
            self,
            prompt,
            generation_stream,
            fwd::PREFILL_STEP_SIZE,
            true,
            |inner: &K2Inner| inner.turn_cancel.as_deref(),
            |inner, chunk, _is_final| inner.forward(chunk),
            |inner| fwd::eval_caches_and_clear(&inner.caches),
        )
    }

    /// Reset all caches and cached token history (both reset scopes share
    /// this clear; `ResetScope::Command` additionally purges the paged
    /// prefix cache in the trait impl).
    fn reset_caches_internal(&mut self) {
        self.caches = init_caches(&self.config);
        self.cached_token_history.clear();
        // Drop any live paged-adapter requests. Without this a prior
        // keep-live finalize would leave a block_table populated and a
        // subsequent paged turn could warm-continue against stale tokens.
        // Released full blocks stay content-addressed — the explicit
        // command reset purges them separately.
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_all_requests();
        }
    }

    /// Save the committed token history (flat path). `last_token_in_cache`
    /// mirrors the qwen3 convention: on a `length` exit the decode loop's
    /// `materialize_final` already forwarded the last token, so the full
    /// generated run is in-cache; on any other exit the final boundary
    /// token was never forwarded and is dropped.
    fn save_cache_state_internal(
        &mut self,
        reuse_cache: bool,
        tokens: &[u32],
        generated_tokens: &[u32],
        last_token_in_cache: bool,
    ) {
        if !fwd::save_flat_token_history(
            tokens,
            generated_tokens,
            last_token_in_cache,
            reuse_cache,
            FinalTokenPolicy::KeepAllOnLength,
            &mut self.cached_token_history,
        ) {
            self.reset_caches_internal();
        }
    }

    // ============================ Paged path ============================

    /// Single paged prefill chunk: `record_tokens` first (the adapter's
    /// cursor must already cover the chunk when `update_keys_values` runs),
    /// then embed + per-layer `forward_paged`. Returns the residual stream
    /// `[1, chunk_len, hidden]` — NOT logits.
    fn run_paged_prefill_one_chunk(
        &mut self,
        chunk_tokens: &[u32],
        chunk_first_position: u32,
    ) -> Result<MxArray> {
        let chunk_len = chunk_tokens.len() as u32;
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_prefill_chunk: paged_adapter is None")
            })?;
            adapter
                .record_tokens(chunk_tokens)
                .map_err(Error::from_reason)?;
        }

        let input_ids = MxArray::from_uint32(chunk_tokens, &[1, chunk_len as i64])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;

        // `cached_prefix_len = chunk_first_position`: everything already in
        // the pool (prior cache hit + earlier chunks) lives at logical
        // positions [0, chunk_first_position); this chunk occupies
        // [chunk_first_position, +chunk_len). The layer's `forward_paged`
        // handles both the cold (causal SDPA in-flight) and the cache-hit
        // (pool read + offset causal mask) branches.
        let num_layers = self.layers.len();
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            // Split-borrow: `self.layers` (immutable) vs `self.paged_adapter`
            // (mutable) are disjoint fields.
            let layer: &K2DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_prefill_chunk: paged_adapter dropped")
            })?;
            hidden_states = layer.forward_paged(
                &hidden_states,
                adapter,
                layer_idx as u32,
                chunk_first_position,
                chunk_first_position,
                /* is_prefill */ true,
            )?;
            // Bound the in-flight lazy graph on long prefills (byte-neutral
            // materialization cadence, MLX_PAGED_PREFILL_EVAL_INTERVAL).
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }
        Ok(hidden_states)
    }

    /// Chunk-size-parameterized paged prefill over the suffix. Returns the
    /// last token's logits `[vocab]`.
    fn run_paged_prefill_chunk_with_size(
        &mut self,
        suffix_tokens: &[u32],
        first_logical_position: u32,
        chunk_size: i32,
    ) -> Result<MxArray> {
        if suffix_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_prefill_chunk called with empty suffix",
            ));
        }

        if chunk_size <= 0 || suffix_tokens.len() <= chunk_size as usize {
            let hidden_states =
                self.run_paged_prefill_one_chunk(suffix_tokens, first_logical_position)?;
            return self.project_last_token_logits(&hidden_states);
        }

        let chunk_size_usize = chunk_size as usize;
        let total_chunks = suffix_tokens.len().div_ceil(chunk_size_usize);
        let mut last_hidden: Option<MxArray> = None;
        let mut tokens_consumed: u32 = 0;
        for (chunk_idx, chunk) in suffix_tokens.chunks(chunk_size_usize).enumerate() {
            // Cooperative-cancel checkpoint: abort at the chunk boundary —
            // the Err rides the paged engine's abort arm, which releases
            // the live request without registering its blocks.
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason("prefill cancelled"));
            }
            let chunk_start_pos = first_logical_position + tokens_consumed;
            let is_last_chunk = chunk_idx + 1 == total_chunks;
            let hidden = self.run_paged_prefill_one_chunk(chunk, chunk_start_pos)?;
            tokens_consumed += chunk.len() as u32;
            if is_last_chunk {
                last_hidden = Some(hidden);
            } else {
                // Materialize the residual stream so MLX releases every
                // upstream node before the next chunk's graph builds.
                hidden.eval();
                crate::array::memory::synchronize_and_clear_cache();
            }
        }
        let hidden_states = last_hidden
            .ok_or_else(|| Error::from_reason("K2 chunked prefill produced no hidden state"))?;
        self.project_last_token_logits(&hidden_states)
    }

    /// Paged prefill entry: reads `MLX_PAGED_PREFILL_CHUNK_SIZE` once via the
    /// shared helper and forwards to the size-parameterized worker.
    fn run_paged_prefill_chunk(
        &mut self,
        suffix_tokens: &[u32],
        first_logical_position: u32,
    ) -> Result<MxArray> {
        let chunk_size = crate::array::paged_prefill_chunk_size();
        self.run_paged_prefill_chunk_with_size(suffix_tokens, first_logical_position, chunk_size)
    }

    /// Final grouped norm + lm_head over the last position only.
    /// `hidden_states`: `[1, T, hidden]` → `[vocab]`.
    fn project_last_token_logits(&self, hidden_states: &MxArray) -> Result<MxArray> {
        fwd::project_last_token_logits(hidden_states, &self.norm, |normed| {
            self.project_logits(normed)
        })
    }

    /// One paged decode step: record the token, embed, per-layer paged
    /// forward, final norm + lm_head. Returns `[1, 1, vocab]`.
    fn run_paged_decode_step(&mut self, token_id: u32) -> Result<MxArray> {
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

        let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;

        let num_layers = self.layers.len();
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            let layer: &K2DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter dropped")
            })?;
            hidden_states = layer.forward_paged(
                &hidden_states,
                adapter,
                layer_idx as u32,
                first_logical_position,
                /* cached_prefix_len */ 0,
                /* is_prefill */ false,
            )?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        self.project_logits(&hidden_states)
    }

    /// Submit-ahead paged decode step for
    /// [`DecodeStep::forward_with_lazy_token`]: identical to
    /// [`Self::run_paged_decode_step`] except the input token id is not yet
    /// drained — `record_placeholder_token` reserves the write slot
    /// (record-first: the slot derives from the recorded COUNT, never the
    /// id value) and `input_ids` is the caller's lazy `[1, 1]` sample
    /// (embedding `take` accepts the int32 indices as-is). The caller
    /// patches the placeholder via `patch_last_recorded_token` at commit,
    /// or rewinds it via `rollback_last_tokens(1)` on a terminal step.
    ///
    /// Error contract: a returned `Err` carries NO pending speculative
    /// state — the placeholder record is rolled back here before the
    /// error propagates, so the decode loop can treat a speculative
    /// failure on a terminal step as never-attempted (the serial arm's
    /// terminal-before-forward ordering) instead of an abort.
    fn run_paged_decode_step_lazy(&mut self, input_ids: &MxArray) -> Result<MxArray> {
        let first_logical_position = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step_lazy: paged_adapter is None")
            })?;
            adapter.current_token_count()
        };
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step_lazy: paged_adapter dropped")
            })?;
            adapter
                .record_placeholder_token()
                .map_err(Error::from_reason)?;
        }

        // The placeholder record is the only committed host state this
        // step makes; graph-lazy KV writes may never have evaluated on an
        // error, which is fine (dead slot inside a still-allocated block).
        // Roll the record back best-effort and return the ORIGINAL error.
        let result = self.run_paged_decode_step_lazy_body(input_ids, first_logical_position);
        if result.is_err()
            && let Some(adapter) = self.paged_adapter.as_mut()
        {
            let _ = adapter.rollback_last_tokens(1);
        }
        result
    }

    /// Body of [`Self::run_paged_decode_step_lazy`] — embed the lazy ids
    /// and run the layer stack. Split out so the caller can roll the
    /// placeholder record back around a fallible build.
    fn run_paged_decode_step_lazy_body(
        &mut self,
        input_ids: &MxArray,
        first_logical_position: u32,
    ) -> Result<MxArray> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let num_layers = self.layers.len();
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            let layer: &K2DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step_lazy: paged_adapter dropped")
            })?;
            hidden_states = layer.forward_paged(
                &hidden_states,
                adapter,
                layer_idx as u32,
                first_logical_position,
                /* cached_prefix_len */ 0,
                /* is_prefill */ false,
            )?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        self.project_logits(&hidden_states)
    }

    /// Uniform batched paged decode: `[N,1,H]` rows, one token each.
    /// Pure-KV — no per-row out-of-band state to stack/scatter (the
    /// `HybridStepExecutor` contract LFM2 uses for its conv state simply
    /// never fires here).
    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        if rows.is_empty() {
            return Err(Error::from_reason(
                "run_paged_decode_step_batched requires at least one row",
            ));
        }
        // Record every row's token BEFORE the forward (record-first
        // contract); `record_tokens_batched` rolls back recorded rows on
        // failure so a partial record set never skews the pool, and returns
        // each row's pre-record write position as `planned_rows`.
        let planned_rows = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| {
                Error::from_reason("run_paged_decode_step_batched: paged adapter is unavailable")
            })?
            .record_tokens_batched(rows)
            .map_err(Error::from_reason)?;

        let token_ids = rows.iter().map(|&(_, token)| token).collect::<Vec<_>>();
        let input_ids = MxArray::from_uint32(&token_ids, &[rows.len() as i64, 1])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        if planned_rows.len() == 1 {
            // Single-row wave: the batched arm rebuilds per-row offsets and
            // ragged metadata in every layer; the singleton arm reuses the
            // now-active request's cached decode inputs instead.
            let position = planned_rows[0].1;
            for layer_idx in 0..self.layers.len() {
                let layer: &K2DecoderLayer = unsafe { &*self.layers.as_ptr().add(layer_idx) };
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason("run_paged_decode_step_batched: paged adapter dropped")
                })?;
                hidden_states = layer.forward_paged(
                    &hidden_states,
                    adapter,
                    layer_idx as u32,
                    position,
                    /* cached_prefix_len */ 0,
                    /* is_prefill */ false,
                )?;
            }
        } else {
            for layer_idx in 0..self.layers.len() {
                let layer: &K2DecoderLayer = unsafe { &*self.layers.as_ptr().add(layer_idx) };
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason("run_paged_decode_step_batched: paged adapter dropped")
                })?;
                hidden_states = layer.forward_paged_batched(
                    &hidden_states,
                    adapter,
                    layer_idx as u32,
                    &planned_rows,
                )?;
            }
        }
        hidden_states = self.norm.forward(&hidden_states)?;
        self.project_logits(&hidden_states)
    }

    fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        // Pure-KV: activation is just the adapter's request switch — no
        // per-sequence side state to swap in (unlike LFM2's conv table).
        self.paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("k2 paged adapter is unavailable"))?
            .activate_request(seq_id)
            .map_err(Error::from_reason)
    }
}

/// Flat decode stepper — owns nothing but the `&mut` borrow; the engine's
/// decode loop drives `forward` per sampled token.
pub(crate) struct K2Decode<'a> {
    inner: &'a mut K2Inner,
}

impl DecodeStep for K2Decode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // Eager native forward returns [1, 1, vocab]; `true` tells the loop
        // to `squeeze([1])` down to [1, vocab].
        Ok((self.inner.forward(input_ids)?, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token]);
    }

    fn supports_token_pipeline(&self) -> bool {
        // Pure-KV flat stepper: `forward` embeds the lazy ids directly
        // (no host token id is ever needed) and the only host-side state
        // a speculative step advances is each `KVCache`'s offset — a
        // `trim(offset - 1)` rewind, the same primitive the exact-match
        // path already uses (`pending_exact_match_rewind`).
        true
    }

    fn forward_with_lazy_token(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // `inner.forward` advances each `KVCache`'s offset incrementally
        // per layer, so a mid-forward error leaves PARTIAL advancement —
        // a blind `trim(offset - 1)` on every cache would corrupt layers
        // that never wrote. Snapshot the offsets up front and restore
        // each one on error (`trim` is a no-op when `new_len >= offset`,
        // so un-advanced caches are untouched). Contract: an `Err` return
        // carries no pending speculative state.
        let offsets: Vec<i32> = self
            .inner
            .caches
            .iter()
            .map(|cache| cache.get_offset())
            .collect();
        match self.inner.forward(input_ids) {
            Ok(logits) => Ok((logits, true)),
            Err(error) => {
                for (cache, offset) in self.inner.caches.iter_mut().zip(offsets) {
                    cache.trim(offset);
                }
                Err(error)
            }
        }
    }

    fn commit_lazy_token(&mut self, _token_id: u32) -> Result<()> {
        // Nothing to patch — the flat path consumed the lazy id array
        // directly and keeps no host-side token record.
        Ok(())
    }

    fn rollback_lazy_step(&mut self) -> Result<()> {
        // Rewind each layer's offset past the speculatively written slot;
        // the slot's dead bytes are overwritten in place by the next
        // `update_and_fetch` (KVCache::trim convention).
        for cache in self.inner.caches.iter_mut() {
            cache.trim(cache.get_offset() - 1);
        }
        Ok(())
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        // Pure-KV flat stepper (qwen3 convention): the decode loop's
        // forward gate skips the final committed token's forward on a
        // `length` exit, so the flat caches end one token SHORTER than the
        // keep-all history `save_cache_state` persists. One discard-logits
        // forward closes the gap.
        let input_ids = MxArray::from_int32(&[token_id as i32], &[1, 1])?;
        let logits = self.inner.forward(&input_ids)?;
        logits.eval();
        Ok(())
    }
}

/// Paged decode state (the paged analog of [`K2Decode`]); wrapped by
/// [`PagedStepper`] for the `DecodeStep` impl. Each `paged_step` runs the
/// eager paged step through `run_paged_decode_step`.
pub(crate) struct K2PagedDecode<'a> {
    inner: &'a mut K2Inner,
}

impl PagedStepModel for K2PagedDecode<'_> {
    /// `SyncToken`: a single synchronous eval pulls logits AND the paged
    /// K/V writes through the dependency chain (one sync wait); the
    /// loop-top `y.eval()` then no-ops.
    const EVAL: EvalPolicy = EvalPolicy::SyncToken;
    /// Pure-KV paged model: the placeholder record reserves the write
    /// slot (count-derived), the drained id patches it at commit, and a
    /// terminal step rewinds the record — no out-of-band state
    /// (conv/recurrent/MTP) to undo.
    const PIPELINED: bool = true;
    const FINAL_TOKEN_POLICY: FinalTokenPolicy = FinalTokenPolicy::KeepAllOnLength;

    fn paged_step(&mut self, token_id: u32) -> Result<MxArray> {
        self.inner.run_paged_decode_step(token_id)
    }

    fn paged_step_lazy(&mut self, input_ids: &MxArray) -> Result<MxArray> {
        self.inner.run_paged_decode_step_lazy(input_ids)
    }

    fn commit_placeholder(&mut self, token_id: u32) -> Result<()> {
        self.inner
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("commit_lazy_token: paged_adapter is None"))?
            .patch_last_recorded_token(token_id)
            .map_err(Error::from_reason)
    }

    fn rollback_placeholder(&mut self) -> Result<()> {
        // Bookkeeping-only rewind — the same idempotent primitive the
        // batched record-failure path already uses
        // (`run_paged_decode_step_batched`). The speculatively written KV
        // slot becomes dead space inside its still-allocated block and is
        // overwritten in place by the next recorded token.
        self.inner
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("rollback_lazy_step: paged_adapter is None"))?
            .rollback_last_tokens(1)
            .map_err(Error::from_reason)
    }

    fn materialize_final_token(&mut self, token_id: u32) -> Result<()> {
        // PAGED + pure-KV (qwen3 convention, NOT lfm2's): record + forward
        // the final length-exit token so `request_tokens()` equals the
        // keep-all history — the pipelined decode loop never forwards the
        // last sampled token, so the adapter would otherwise end one short.
        let logits = self.inner.run_paged_decode_step(token_id)?;
        logits.eval();
        Ok(())
    }
}

/// K2 paged prefix state — the shared two-usize shape (no sidecar fields;
/// K2 has no out-of-band state to reconcile).
pub(crate) type K2PrefixState = SimplePagedPrefix;

impl K2Inner {
    fn prime_prefix_state_for(
        &mut self,
        seq_id: SeqId,
        plan: &[u32],
        reuse_cache: bool,
        extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<K2PrefixState> {
        let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
            Error::from_reason(
                "prime_prefix_state: paged_adapter is None — caller must check \
                 use_block_paged_cache before dispatch",
            )
        })?;
        prime_single_adapter_prefix(adapter, seq_id, plan, reuse_cache, extra_keys, cache_salt)
            .map_err(Error::from_reason)
    }
}

impl PagedBackend for K2Inner {
    type PagedDecode<'a>
        = PagedStepper<K2PagedDecode<'a>>
    where
        Self: 'a;
    type PrefixState = K2PrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        reuse_cache: bool,
        _block_size: usize,
        extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        self.prime_prefix_state_for(0, plan, reuse_cache, extra_keys, cache_salt)
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        // Pure-KV: no conv Pass-1, no sidecar. `record_tokens` runs inside
        // the per-chunk worker; returns [vocab] last-token logits.
        self.run_paged_prefill_chunk(suffix_tokens, prefix.effective_cached_prefix_len as u32)
    }

    fn begin_paged_decode(&mut self) -> Result<Self::PagedDecode<'_>> {
        Ok(PagedStepper(K2PagedDecode { inner: self }))
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool, cache_salt: u64) {
        // Terminal lifecycle (qwen3 shape). Success: keep the request live
        // across turns when reuse is on so the next turn's `continue_turn`
        // builds on the partial trailing block's live K/V; otherwise
        // register full blocks for reuse + release. Infallible — a
        // teardown failure must not mask the turn result.
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = finalize_single_adapter_turn(adapter, reuse_cache, &[], cache_salt);
        }
    }

    fn abort_paged_turn(&mut self) {
        // Error-path teardown: release fully — partial block_table state is
        // unsafe to keep. Release ONLY, never register / keep live.
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = abort_single_adapter_turn(adapter);
        }
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        // Qwen3 token accounting: history only (the adapter pool owns the
        // K/V). `keep_all` is the flat rule (finish_reason == "length" AND
        // `materialize_final` forwarded the last token — K2's paged
        // stepper overrides `materialize_final`, so the last token IS in
        // the adapter on length exits).
        save_paged_token_history(
            save_tokens,
            generated,
            keep_all,
            reuse_cache,
            FinalTokenPolicy::KeepAllOnLength,
            &mut self.cached_token_history,
        );
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        keep_all: bool,
    ) -> bool {
        // Perf-parity warm-continue restore (qwen3 shape): the pipelined
        // decode loop records the stop token into the adapter (its forward
        // ran at the loop top BEFORE the stop-check), but saved history
        // DROPS it on a non-length exit. Roll the adapter back so
        // `request_tokens()` matches the persisted history.
        let Some(adapter) = self.paged_adapter.as_mut() else {
            return true;
        };
        if let Err((surplus, e)) = reconcile_paged_surplus(
            adapter.request_tokens().len(),
            prompt_len,
            generated.len(),
            keep_all,
            FinalTokenPolicy::KeepAllOnLength,
            |n| adapter.rollback_last_tokens(n),
        ) {
            tracing::warn!(
                target: "mlx_core::k2_horizon::paged",
                "reconcile_paged_request_tokens: rollback_last_tokens({surplus}) failed \
                 (finalize releases the request; next turn cold-prefills): {e}",
            );
            return false;
        }
        true
    }
}

impl ChatBackend for K2Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "k2_horizon"
    }

    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.turn_cancel = flag;
    }

    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32> {
        // K2's session terminator is `<|ifm|im_end|>` (id 250019), NOT the
        // ChatML `<|im_end|>` — the generic `im_end_id()` lookup would miss.
        tok.token_to_id("<|ifm|im_end|>".to_string())
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|ifm|im_end|> special token"))
    }

    fn generation_defaults(&self) -> Option<&crate::engine::ModelGenerationDefaults> {
        Some(&self.gen_defaults)
    }

    fn extra_eos_ids(&self) -> Vec<u32> {
        // generation_config.json ships eos [1, 250019] — both stop the turn.
        self.gen_defaults.eos_token_ids.clone()
    }

    fn policy(&self) -> ThinkingPolicy {
        // K2's template ALWAYS opens a reasoning tag (any valid effort);
        // `enable_thinking` is not consulted. Effort picks the tag variant
        // AND derives the budget via `default_thinking_budget_for_effort`
        // (explicit thinking_token_budget wins). `normalize_k2_effort` maps
        // none to low before render and budget resolution.
        ThinkingPolicy::AlwaysOnBudgetFromEffort
    }

    /// Per-turn think-end resolution: K2's close tag is a FUNCTION of
    /// `reasoning_effort` (high→`</ifm|think>`, medium→`</ifm|think_fast>`,
    /// low→`</ifm|think_faster>`), unlike the single static
    /// `tokenizer.think_end_id()` the engine defaults to. The template
    /// defaults an unset effort to `high`; unsupported values raise at
    /// render, so the fallback arm never survives to decode.
    /// The request's effort reaches the template only after
    /// [`normalize_k2_effort`] — applied here too so the tracker's close
    /// token always matches the tag the render actually opened.
    fn think_end_for_turn(&self, config: &ChatConfig, tok: &Qwen3Tokenizer) -> ThinkEndResolution {
        let tag = match normalize_k2_effort(config.reasoning_effort.as_deref()).as_deref() {
            Some("medium") => K2_THINK_END_MEDIUM,
            Some("low") => K2_THINK_END_LOW,
            _ => K2_THINK_END_HIGH,
        };
        // The rendered open is effort-matched, but the MODEL may emit any
        // member of the `</ifm|think*>` family to close — arm the tracker
        // on the rendered tag (it is also the budget-forced token) and
        // accept the other two as alternates so a cross-family close
        // still flips the reasoning boundary instead of leaking into
        // `reasoning_content`.
        let think_end_extra_ids = [K2_THINK_END_HIGH, K2_THINK_END_MEDIUM, K2_THINK_END_LOW]
            .iter()
            .filter(|t| **t != tag)
            .filter_map(|t| tok.token_to_id(t.to_string()))
            .collect();
        ThinkEndResolution {
            think_end_id: tok.token_to_id(tag.to_string()),
            think_end_str: Some(tag.to_string()),
            think_end_extra_ids,
        }
    }

    /// `AlwaysOnBudgetFromEffort` on the NORMALIZED effort: raw `'none'`
    /// would read as budget-0 (force-close) while the normalized render
    /// opens `think_faster` — the normalized `'low'` gives the intended
    /// small 256 cap instead.
    fn thinking_setup(&self, config: &ChatConfig) -> ThinkingSetup {
        let mut config = config.clone();
        config.reasoning_effort = normalize_k2_effort(config.reasoning_effort.as_deref());
        crate::engine::params::resolve(self.policy(), &config)
    }

    /// Render with the normalized effort — a raw `'none'`/`'minimal'`/
    /// `'xhigh'`/`'max'` would hit the template's `raise_exception` arm.
    /// (Only the `add_generation_prompt` render can raise; the
    /// continuation verifier's `false` renders skip the effort branch,
    /// so this override alone covers the turn.)
    fn render_prompt(
        &self,
        tok: &Qwen3Tokenizer,
        messages: &[ChatMessage],
        config: &ChatConfig,
        preserve_thinking: bool,
    ) -> Result<Vec<u32>> {
        let mut config = config.clone();
        config.reasoning_effort = normalize_k2_effort(config.reasoning_effort.as_deref());
        tok.apply_chat_template_with_config(messages, true, &config, preserve_thinking)
    }

    /// K2's canonical close tag — the one history replays land on
    /// (`reasoning_content` always re-renders inside `<ifm|think>`).
    /// Turns generated under `think_fast`/`think_faster` carry a different
    /// close tag, so the continuation verifier's normalized comparison
    /// fails closed to cold replay for them — safe by construction.
    const REASONING_CLOSE_TAG: &'static str = K2_THINK_END_HIGH;

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        self.reset_caches_internal();
        // The explicit command reset must restore a fully cold state:
        // `release_all_requests` alone leaves full blocks content-addressed
        // in the allocator's prefix cache, so a reset-then-rerun of the
        // same prompt would take the prefix-hit path whose bf16 reduction
        // order differs from the cold full prefill. Purge so the next turn
        // replays the cold prefill byte-for-byte. `PrefixMiss` keeps the
        // prefix cache (cross-request block reuse is the paged design's
        // entire point).
        if scope == ResetScope::Command
            && let Some(adapter) = self.paged_adapter.as_mut()
        {
            adapter
                .release_request_and_purge_prefix_cache()
                .map_err(|e| {
                    Error::from_reason(format!(
                        "k2_horizon reset_caches: paged prefix-cache purge failed: {e}"
                    ))
                })?;
        }
        Ok(())
    }

    /// All-or-nothing prefix match PLUS the sanctioned pure-KV exact-match
    /// rewind (qwen3 convention): on `tokens == cached_token_history`
    /// return `cached_len - 1` and arm `pending_exact_match_rewind`, which
    /// `chunked_prefill` consumes to `trim` each flat cache one slot before
    /// re-forwarding the last token. Safe because `KVCache::trim` +
    /// `update_and_fetch` overwrite the trimmed slot deterministically.
    /// The `cached_len == 1` corner returns 0 (miss → reset, then a
    /// 1-token prefill).
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        self.pending_exact_match_rewind.set(false);
        if !reuse_cache {
            return 0;
        }
        let cached = &self.cached_token_history;
        let has_kv = self.caches.first().is_some_and(|c| c.get_offset() > 0);
        let hit = if !cached.is_empty()
            && tokens.len() >= cached.len()
            && tokens[..cached.len()] == cached[..]
            && has_kv
        {
            cached.len()
        } else {
            0
        };
        if hit > 1 && hit == tokens.len() {
            self.pending_exact_match_rewind.set(true);
            return hit - 1;
        }
        hit
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        // Qwen3 pure-KV convention: keep the full generated run only on a
        // `length` exit (the stepper's `materialize_final` forwarded the
        // final token into `self.caches`); otherwise drop the last boundary
        // token, which was sampled but never forwarded.
        self.save_cache_state_internal(
            args.reuse_cache,
            args.save_tokens,
            args.generated_tokens,
            args.finish_reason == "length",
        );
    }

    fn eval_caches(&self) -> Result<()> {
        eval_kv_caches(&self.caches)
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        // int32 prompt (load-bearing dtype parity with lfm2/qwen3 eager).
        let token_arr: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
        let prompt = MxArray::from_int32(&token_arr, &[1, prompt_tokens.len() as i64])?;
        let logits = self.chunked_prefill(&prompt, stream)?;
        fwd::slice_last_logits_keep_batch(&logits)
    }

    type Decode<'a>
        = K2Decode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        Ok(K2Decode { inner: self })
    }

    fn finalize_turn(&self, args: FinalizeArgs<'_>) -> Result<ChatResult> {
        // K2 markup differs from the ChatML default: `<ifm|think*>` close
        // tags and `<ifm|tool_call>` blocks. The shared pipeline is
        // replicated in `k2_finalize` with the K2 tag spec.
        super::finalize::finalize_k2_chat_result(args)
    }

    fn execution_plan(&self) -> ExecutionPlan {
        ExecutionPlan {
            media: MediaPlan::NONE,
            paged_attention: self.paged_adapter.as_ref().map(|_| PagedAttentionPlan {
                // Delta turns stay PAGED (pure-KV: the engine rebuilds the
                // full token stream before plan resolution, so a delta
                // reaches `run_paged_turn` as the same strict extension a
                // resent growing conversation produces).
                supports_delta: true,
            }),
            speculative: None,
        }
    }

    fn eos_before_emit(&self) -> bool {
        // Check EOS before cancel + before detokenize/emit (no EOS text
        // leaks; EOS+cancel resolves as "stop").
        true
    }

    fn augment_performance(&self, _profiler: &DecodeProfiler, _metrics: &mut PerformanceMetrics) {
        // No MTP heads → no acceptance fields; no `profile_phases` — keep
        // the payload byte-stable.
    }

    fn session_media(&self) -> MediaCapabilities {
        MediaCapabilities::NONE
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // Fresh AND delta turns land here (`supports_delta: true`); the
        // model-neutral paged driver runs prime → prefill → decode →
        // reconcile → save through `<K2Inner as PagedBackend>`.
        crate::engine::paged_turn::run_paged_turn(self, args)
    }
}

impl HybridSchedulerBackend for K2Inner {
    type FamilyCommand = std::convert::Infallible;
    type RestoreTicket = PagedRestoreTicket;
    type OwnerState = Vec<u32>;
    type StepExecutor<'a> = HybridStepExecutor<'a, Self>;

    const SCHEDULER_NAME: &'static str = "K2Horizon";

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.paged_adapter.as_ref()
    }

    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.paged_adapter.as_mut()
    }

    fn max_position_embeddings(&self) -> i32 {
        self.config.max_position_embeddings
    }

    fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        self.activate_paged_seq(seq_id)
    }

    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        self.run_paged_decode_step_batched(rows)
    }

    fn replace_cached_token_history(&mut self, history: Vec<u32>) {
        self.cached_token_history = history;
    }

    fn owner_tokens(state: &Self::OwnerState) -> &[u32] {
        state
    }

    fn capture_owner_state(&mut self, _seq_id: SeqId) -> Self::OwnerState {
        self.cached_token_history.clone()
    }

    fn build_scheduled_prefix(
        &self,
        _base: &Self::PrefixState,
        effective_cached_prefix_len: usize,
        suffix_len: usize,
        _full_tokens: Vec<u32>,
        _first_chunk: bool,
    ) -> Self::PrefixState {
        K2PrefixState {
            effective_cached_prefix_len,
            suffix_len,
        }
    }

    fn prepare_scheduled_prefix(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        _owner_history: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
        _block_size: u32,
    ) -> Result<ScheduledPrefixAdmission<Self::PrefixState, Self::RestoreTicket>> {
        let total_budget = tokens.len() as u32;
        let admission = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("K2 paged adapter is unavailable"))?
            .prepare_turn_with_async_restore(
                seq_id,
                tokens,
                total_budget,
                reuse_cache,
                &[],
                cache_salt,
                false,
                total_budget.saturating_sub(1),
            )
            .map_err(Error::from_reason)?;
        Ok(match admission {
            PagedTurnAdmission::Ready(plan) => ScheduledPrefixAdmission::Ready(K2PrefixState {
                effective_cached_prefix_len: plan.cached_prefix_len as usize,
                suffix_len: plan.suffix_len as usize,
            }),
            PagedTurnAdmission::Waiting {
                provisional,
                restore,
            } => ScheduledPrefixAdmission::Waiting {
                provisional: K2PrefixState {
                    effective_cached_prefix_len: provisional.cached_prefix_len as usize,
                    suffix_len: provisional.suffix_len as usize,
                },
                restore: *restore,
            },
        })
    }

    fn poll_scheduled_restore(
        &mut self,
        seq_id: SeqId,
        restore: &mut Self::RestoreTicket,
        _prompt_tokens: &[u32],
        _owner_history: &[u32],
        _is_preemption_replay: bool,
    ) -> Result<Option<ScheduledRestoreResult<Self::PrefixState>>> {
        let outcome = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("K2 paged adapter is unavailable"))?
            .poll_restore(restore)
            .map_err(Error::from_reason)?;
        Ok(match outcome {
            PagedRestorePoll::Pending => None,
            PagedRestorePoll::Ready {
                plan,
                bytes_restored,
                wait,
            } => Some(ScheduledRestoreResult {
                prefix: K2PrefixState {
                    effective_cached_prefix_len: plan.cached_prefix_len as usize,
                    suffix_len: plan.suffix_len as usize,
                },
                bytes_restored,
                wait,
                materialized_blocks: self
                    .paged_adapter
                    .as_ref()
                    .and_then(|adapter| adapter.block_table_for(seq_id))
                    .map(|table| table.num_blocks() as u32)
                    .unwrap_or(0),
                profiler_prefill_tokens: plan.suffix_len,
                extra_prefill_breaks: Vec::new(),
            }),
        })
    }

    fn restore_reserved_blocks(restore: &Self::RestoreTicket) -> u32 {
        restore.reserved_blocks()
    }

    fn step_executor(&mut self) -> Self::StepExecutor<'_> {
        HybridStepExecutor::new(self)
    }
}

// ============================================================================
// NAPI surface
// ============================================================================

/// K2-Horizon language model (IFM K2-Horizon-7B).
///
/// Dense decoder-only transformer — GQA, SwiGLU, grouped RMSNorm, full
/// RoPE, untied lm_head. Pure standard-KV: flat KV caches, block-paged
/// adapter (default on), and the continuous-batching scheduler all share
/// this inner state on the dedicated model thread.
#[napi]
pub struct K2HorizonModel {
    /// Dedicated model thread owning `K2SchedulerState`. K2 is chat-only
    /// (no training/generate variants); `K2Cmd` wraps the model-neutral
    /// chat protocol without exposing model state outside the thread.
    pub(crate) thread: crate::model_thread::ModelThread<K2Cmd>,
    pub(crate) config: K2HorizonConfig,
    /// Snapshot of `K2Inner::paged_adapter.is_some()` captured at
    /// construction time — surfaced via `hasBlockPagedCache()` so the
    /// server endpoint can bypass the JS-side warm slot when paged is
    /// active and rely on native content-addressed block reuse.
    pub(crate) paged_active: bool,
    /// RAII: unregisters this model's baseline from the cache-limit
    /// coordinator on drop.
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
    /// RAII debit for the native paged KV pool, kept separate from weights
    /// so the global coordinator can account both deterministic residents.
    pub(crate) _pool_cache_limit_guard: Option<crate::cache_limit::PoolCacheLimitGuard>,
}

#[napi]
impl K2HorizonModel {
    /// Load a K2-Horizon model from a directory containing safetensors and
    /// config.json.
    #[napi]
    pub async fn load(model_path: String) -> Result<K2HorizonModel> {
        K2HorizonModel::load_from_dir(&model_path).await
    }

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance (`true` iff the adapter was constructed at load time —
    /// `use_block_paged_cache` defaults to true; a non-Metal build leaves
    /// it off).
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    /// Test-only entry point that dispatches `ChatStreamSessionStart` and
    /// returns the raw mpsc receiver the model thread writes into.
    #[doc(hidden)]
    pub fn chat_stream_session_start_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::Receiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) = crate::model_thread::stream_channel(
            crate::engine::napi_glue::CHAT_STREAM_NATIVE_QUEUE_LIMIT,
        );
        self.thread
            .send(K2Cmd::Chat(Box::new(ChatCmd::StreamSessionStart {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            })))?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Test-only entry point that dispatches `ChatStreamSessionContinue`.
    #[doc(hidden)]
    pub fn chat_stream_session_continue_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::Receiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) = crate::model_thread::stream_channel(
            crate::engine::napi_glue::CHAT_STREAM_NATIVE_QUEUE_LIMIT,
        );
        self.thread
            .send(K2Cmd::Chat(Box::new(ChatCmd::StreamSessionContinue {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            })))?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Get the model configuration.
    #[napi]
    pub fn get_config(&self) -> K2HorizonConfig {
        self.config.clone()
    }

    /// Native admission capacity for the server's per-model semaphore.
    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        if self.paged_active && !K2SchedulerState::force_serial() {
            scheduler_max_num_seqs_for(32) as u32
        } else {
            1
        }
    }

    /// Estimated number of model parameters.
    #[napi]
    pub fn num_parameters(&self) -> i64 {
        let h = self.config.hidden_size as i64;
        let v = self.config.vocab_size as i64;
        let ff = self.config.intermediate_size as i64;
        let hd = self.config.head_dim() as i64;
        let nh = self.config.num_attention_heads as i64;
        let nkv = self.config.num_key_value_heads as i64;
        let l = self.config.num_hidden_layers as i64;

        // embed_tokens (+ lm_head when untied) + final norm
        let mut total = v * h + h;
        if !self.config.tie_word_embeddings {
            total += v * h;
        }
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        total += l * (2 * h + h * q_dim + h * kv_dim * 2 + q_dim * h + h * ff * 2 + ff * h);
        total
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: K2HorizonModel,
    thread_cmd: crate::models::k2_horizon::model::K2Cmd,
    thread: direct,
    image_guard: text_only,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generic layer's full effort vocabulary must project onto K2's
    /// three-tag set: render ctx, thinking budget, and the close token all
    /// read the normalized value, so a drift here desyncs the turn.
    #[test]
    fn test_normalize_k2_effort() {
        assert_eq!(normalize_k2_effort(Some("high")), Some("high".to_string()));
        assert_eq!(
            normalize_k2_effort(Some("medium")),
            Some("medium".to_string())
        );
        assert_eq!(normalize_k2_effort(Some("low")), Some("low".to_string()));
        // pi's "reasoning unset" sentinel and the minimal level both pick
        // the least thinking K2 can emit.
        assert_eq!(normalize_k2_effort(Some("none")), Some("low".to_string()));
        assert_eq!(
            normalize_k2_effort(Some("minimal")),
            Some("low".to_string())
        );
        // Over-ceiling levels clamp.
        assert_eq!(normalize_k2_effort(Some("xhigh")), Some("high".to_string()));
        assert_eq!(normalize_k2_effort(Some("max")), Some("high".to_string()));
        // Unset/foreign → None so the template's `default('high')` decides.
        assert_eq!(normalize_k2_effort(None), None);
        assert_eq!(normalize_k2_effort(Some("bogus")), None);
    }
}
