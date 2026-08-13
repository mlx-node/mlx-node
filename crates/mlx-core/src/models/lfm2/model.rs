use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::info;

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::ThinkingPolicy;
use crate::engine::backend::{
    ChatBackend, DecodeStep, PagedBackend, PagedPrefix, ResetScope, SaveStateArgs, TurnOutput,
    TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::{ChatCmd, FromChatCmd, handle_chat_cmd};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridSchedulerCommand, HybridSchedulerState, HybridStepExecutor,
    ScheduledPrefixAdmission, ScheduledRestoreResult, SchedulerOwnerContext,
    scheduler_max_num_seqs_for, scheduler_per_seq_context,
};
use crate::engine::plan::{ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan};
use crate::engine::types::{ChatConfig, ChatStreamChunk, ChatStreamHandle};
use crate::engine::{self};
use crate::model_thread::{ResponseTx, send_and_await};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::profiling::PerformanceMetrics;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::transformer::paged_kv_cache_adapter::{
    PagedKVCacheAdapter, PagedRestorePoll, PagedRestoreTicket, PagedTurnAdmission, SeqId,
};

use super::config::Lfm2Config;
use super::conv_sidecar;
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

/// Deepest full block a future identical prompt can restore while still
/// recomputing its final prompt token to produce logits.
fn lfm2_cold_restore_boundary(prompt_tokens: u32, block_size: u32) -> u32 {
    if prompt_tokens == 0 || block_size == 0 {
        return 0;
    }
    prompt_tokens
        .saturating_sub(1)
        .checked_div(block_size)
        .unwrap_or(0)
        .saturating_mul(block_size)
}

/// Commands owned by the LFM2 scheduler thread. Chat variants are lifted from
/// the model-neutral API; scheduler telemetry is a barrier so it observes a
/// coherent step boundary.
pub(crate) enum Lfm2Cmd {
    Chat(Box<ChatCmd>),
    SchedulerStats {
        reply: ResponseTx<engine::SchedulerStatsJs>,
    },
}

impl FromChatCmd for Lfm2Cmd {
    fn from_chat(cmd: ChatCmd) -> Self {
        Self::Chat(Box::new(cmd))
    }
}

impl HybridSchedulerCommand for Lfm2Cmd {
    fn as_chat(&self) -> Option<&ChatCmd> {
        match self {
            Self::Chat(chat) => Some(chat),
            Self::SchedulerStats { .. } => None,
        }
    }

    fn into_chat(self) -> std::result::Result<ChatCmd, Self> {
        match self {
            Self::Chat(chat) => Ok(*chat),
            other => Err(other),
        }
    }

    fn into_scheduler_stats(
        self,
    ) -> std::result::Result<ResponseTx<engine::SchedulerStatsJs>, Self> {
        match self {
            Self::SchedulerStats { reply } => Ok(reply),
            other => Err(other),
        }
    }
}

fn handle_lfm2_cmd(inner: &mut Lfm2Inner, command: Lfm2Cmd) {
    match command {
        Lfm2Cmd::Chat(command) => handle_chat_cmd(inner, *command),
        Lfm2Cmd::SchedulerStats { reply } => {
            let _ = reply.send(Ok(engine::scheduler::SchedulerStats::default().to_js()));
        }
    }
}

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership.
pub(crate) struct Lfm2Inner {
    pub(crate) config: Lfm2Config,
    /// The in-flight turn's cooperative-cancel flag, installed by the
    /// sync and streaming session wrappers via
    /// [`ChatBackend::set_turn_cancel_flag`] and cleared (`None`) in their
    /// turn epilogue on every exit path.
    /// Polled at the top of each flat `chunked_prefill` chunk; `true`
    /// aborts with the distinguished `"prefill cancelled"` error, riding
    /// the engine's fail-closed prefill-`Err` arm. An LFM2 paged miss may
    /// perform a full cached-prefix convolution replay in Pass 1 before the
    /// single-shot suffix forward in Pass 2; neither has an internal chunk
    /// boundary, so each dispatched slice remains an uncancellable window.
    pub(crate) turn_cancel: Option<Arc<AtomicBool>>,
    /// Turn-constant layer classification (`FullAttention` vs `Conv`),
    /// computed once in [`Self::new`] instead of re-derived on every paged
    /// decode step. Pure function of the immutable `config` + layer count.
    /// Mirrors the `Gemma4Inner::layer_kinds` caching pattern.
    pub(crate) layer_kinds: Vec<Lfm2LayerKind>,
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
    /// Per-request recurrent state for the scheduler lane. The legacy
    /// whole-turn path continues to use `caches`; scheduled execution swaps one
    /// request into that field for serial prefill/finalize and stacks rows
    /// directly from this table for batched decode.
    scheduled_caches: HashMap<SeqId, Vec<Lfm2LayerCache>>,
    active_scheduled_seq: Option<SeqId>,
    /// Sampling + stop-token defaults parsed from the checkpoint's
    /// `generation_config.json` at load time. Empty for checkpoints that
    /// ship no such file. Consumed by the [`ChatBackend`] sampling/EOS
    /// hooks (`generation_defaults` / `extra_eos_ids`) to fold checkpoint
    /// defaults under explicit request params. The primary scalar
    /// `config.eos_token_id` is derived separately in `persistence.rs`;
    /// this carries the FULL eos list plus sampling defaults on top.
    gen_defaults: crate::engine::ModelGenerationDefaults,
    /// Whether the MOST RECENT `PagedBackend::prime_prefix_state` call
    /// decided `self.caches`'s conv-layer state already reflected the
    /// incoming prompt's cached prefix byte-for-byte (see
    /// [`conv_state_reusable`]), letting `run_paged_prefill_chunk` skip
    /// the full re-embed + causal-SDPA reconstruction pass
    /// (`run_conv_only_prefill`) over the cached prefix. Read back by
    /// `paged_perf_prefill_tokens` so `report_performance: true`
    /// telemetry reports the SUFFIX-scale numerator (the work this turn
    /// actually did) instead of the full-prompt-scale numerator that is
    /// only honest when the reconstruction pass actually ran.
    last_paged_prefill_reused_conv_state: bool,
    /// Materialized short-conv snapshots at exact block boundaries, keyed by
    /// scheduler sequence (`0` for the whole-turn path). The SSD writer picks
    /// the deepest checkpoint its K/V chain has actually reached.
    conv_cold_checkpoints: HashMap<SeqId, VecDeque<Lfm2ConvColdCheckpoint>>,
    /// Deepest boundary from the current PROMPT that a future request can
    /// actually name. Decode extends the K/V chain past the prompt; allowing
    /// those generated-token checkpoints to win would persist a sidecar no
    /// identical fresh request can reach because lookup is capped at
    /// `prompt_len - 1`.
    conv_cold_capture_boundaries: HashMap<SeqId, u32>,
}

struct Lfm2ConvColdCheckpoint {
    boundary: u32,
    tokens: Vec<u32>,
    states: Vec<MxArray>,
}

/// Classification of the prefix-cache decision made from a
/// [`Lfm2Inner::verify_cache_prefix`] return value plus the incoming
/// token count.
///
/// Test-only mirror of the inlined branch in the engine session core's
/// verify-prefix split — separating the decision logic from the native
/// state mutation so the "exact-match routes to miss" invariant can be
/// pinned by pure-logic unit tests that do not require a loaded LFM2
/// model. Production code keeps the inlined form for zero-overhead
/// dispatch; this enum exists solely to drive
/// `prefix_cache_decision_tests`'s four-case coverage (empty cache,
/// strict-extend hit, divergence miss, exact-match miss). Any change to
/// the inlined production branch MUST be mirrored here or the test
/// ceases to guard the real code.
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
/// Mirrors the inlined branch in the engine session core's
/// verify-prefix split; lifting it out keeps the invariant pinnable
/// without loading a real LFM2 model.
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

/// Decide whether `self.caches`'s conv-layer state ALREADY reflects the
/// token prefix `plan[..cached_prefix_len]` byte-for-byte, so
/// `run_paged_prefill_chunk`'s Pass 1 (`run_conv_only_prefill`) — a full
/// re-embed + causal-SDPA reconstruction over the ENTIRE cached prefix
/// through every layer, purely to rebuild conv state — can be skipped and
/// the live caches carried straight into Pass 2 over just the suffix.
///
/// Returns `true` only for the standard "resend growing conversation"
/// warm-continuation pattern: the new turn's paged-attention cache hit
/// (`cached_prefix_len`, from the content-addressed KV-block pool, which
/// may span foreign/cross-session blocks) EXACTLY equals the length AND
/// content of `cached_token_history` — the token sequence
/// `save_cache_state_internal` proved `self.caches`'s conv state was left
/// at after the immediately preceding successful turn on THIS model
/// instance. Any mismatch (first turn, a turn that aborted since the last
/// save, a partial/foreign prefix hit shorter or longer than
/// `cached_token_history`, or divergent content) safely returns `false`
/// and falls back to the full reconstruction — conv state is a pure
/// function of the token prefix, so this check is sufficient regardless
/// of which logical chat session produced `cached_token_history`.
///
/// `cached_prefix_len == 0` and the degenerate identical-resend case
/// (`cached_prefix_len` capped one below an exact match by
/// `prime_prefix_state`'s `max_cache_hit_tokens`) both return `false` —
/// mirrors [`classify_prefix_cache_decision`]'s exact-match-is-miss
/// invariant: LFM2 has no safe "rewind by one" primitive.
///
/// Both the legacy whole-turn path and the scheduler consume this predicate
/// directly. The sole oracle is the carried incremental state produced by that
/// sequence's preceding tokens, never a numerically different reconstruction.
#[inline]
fn conv_state_reusable(
    plan: &[u32],
    cached_token_history: &[u32],
    cached_prefix_len: usize,
) -> bool {
    cached_prefix_len > 0
        && cached_prefix_len == cached_token_history.len()
        && plan.len() >= cached_prefix_len
        && plan[..cached_prefix_len] == cached_token_history[..]
}

/// Paged decode stepper for lfm2 / lfm2_moe (the paged analog of the FLAT
/// [`Lfm2Decode`]). Drives [`crate::engine::decode::run_decode_loop`] through
/// the generic [`crate::engine::paged_turn::run_paged_turn`]: each `forward`
/// runs the pure-Rust eager paged step. Created by
/// `<Lfm2Inner as PagedBackend>::begin_paged_decode`.
pub(crate) struct Lfm2PagedDecode<'a> {
    inner: &'a mut Lfm2Inner,
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
        // site: the execution plan hands eligible fresh turns to the generic
        // `run_paged_turn` driving `<Lfm2Inner as PagedBackend>` whenever the
        // adapter is live.
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
        // The block-paged KV cache and its compiled decode path rely on
        // Metal-only kernels; on a non-Metal backend (the CUDA/Linux build) the
        // paged writes/gathers hit throwing stubs. Force flat eager there by
        // never building the adapter, mirroring how Qwen3.5 gates its compiled
        // paths. macOS is unaffected — the backend probe is always true, so the
        // `unwrap_or(true)` default still wins.
        let want_paged = config.use_block_paged_cache.unwrap_or(true)
            && crate::engine::persistence::compiled_forward_backend_available();
        let paged_adapter = if want_paged {
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

        // Layer classification is a pure function of the immutable config +
        // layer count; compute once here (see the field rustdoc).
        let layer_kinds = compute_layer_kinds_for(&config, num_layers);

        Ok(Self {
            config,
            turn_cancel: None,
            layer_kinds,
            embed_tokens,
            layers,
            embedding_norm,
            lm_head,
            caches,
            tokenizer: None,
            cached_token_history: Vec::new(),
            cached_image_key: None,
            paged_adapter,
            scheduled_caches: HashMap::new(),
            active_scheduled_seq: None,
            // Empty until the load path parses `generation_config.json`
            // (set via `set_gen_defaults` in `persistence.rs`).
            gen_defaults: crate::engine::ModelGenerationDefaults::default(),
            last_paged_prefill_reused_conv_state: false,
            conv_cold_checkpoints: HashMap::new(),
            conv_cold_capture_boundaries: HashMap::new(),
        })
    }

    /// Rebuild the construction-time pool after weight materialization so the
    /// Metal working-set probe caps cache residency against the already-live
    /// model instead of independently spending the same device budget twice.
    pub(crate) fn size_paged_pool_after_weight_load(&mut self) -> Result<()> {
        if self.paged_adapter.is_none() {
            return Ok(());
        }
        // Preserve an explicit operator/test cap. Adaptive sizing is only for
        // uncapped configurations; replacing a successfully co-resident fixed
        // pool here can fail spuriously as prior model instances tear down.
        if self.config.paged_cache_memory_mb.is_some() {
            return Ok(());
        }
        self.paged_adapter = None;

        let attn_layer_count = self.config.full_attn_idxs().len() as u32;
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
            attn_layer_count,
            self.config.num_key_value_heads as u32,
            self.config.head_dim() as u32,
            block_size,
            cache_dtype,
        )
        .map_err(|error| {
            Error::from_reason(format!(
                "LFM2 adaptive paged cache sizing failed safely; refusing an uncapped pool request: {error}"
            ))
        })?;
        let selected_mb = sizing.selected_bytes.div_ceil(1024 * 1024).max(1) as u32;
        let pa_config = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            gpu_memory_mb: selected_mb,
            head_size: self.config.head_dim() as u32,
            num_kv_heads: self.config.num_key_value_heads as u32,
            num_layers: attn_layer_count,
            use_fp8_cache: Some(false),
            max_seq_len: Some(per_seq_context),
            max_batch_size: Some(scheduler_max_num_seqs_for(32) as u32),
        };
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            sizing.selected_blocks,
            block_size,
        )));
        let pool = mlx_paged_attn::LayerKVPool::new(pa_config, sizing.selected_blocks, cache_dtype)
            .map_err(|error| {
                Error::from_reason(format!("Failed to construct LFM2 KV pool: {error}"))
            })?;
        self.paged_adapter = Some(
            PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size)
                .map_err(Error::from_reason)?,
        );
        info!(
            "LFM2 scheduler pool enabled: requested_blocks={}, selected_blocks={}, bytes={:.2} GiB, per_seq_context={}, max_num_seqs={}",
            requested_blocks,
            sizing.selected_blocks,
            sizing.selected_bytes as f64 / (1u64 << 30) as f64,
            per_seq_context,
            scheduler_max_num_seqs_for(32),
        );
        Ok(())
    }

    pub(crate) fn build_cold_tier_context(
        &self,
        model_path: &str,
        weights: &crate::array::memory::WeightsResident,
    ) -> Option<crate::transformer::paged_kv_cache_adapter::ColdTierContext> {
        let adapter = self.paged_adapter.as_ref()?;
        let manager = crate::cold_tier::global_cold_cache()?;
        let cache_dtype = format!("{:?}", adapter.layer_kv_pool().cache_dtype());
        let geometry = conv_sidecar::geometry(&self.config, &cache_dtype)?;
        let sidecar_policy = conv_sidecar::policy(&self.config, &cache_dtype)?;
        let mut config_json = serde_json::to_vec(&self.config).ok()?;
        config_json.extend_from_slice(&geometry.fingerprint_component());
        let pool = adapter.layer_kv_pool();
        let pool_geometry = crate::cold_tier::ColdTierGeometry {
            block_size: pool.block_size() as u64,
            num_layers: pool.num_layers() as u64,
            num_kv_heads: pool.config().num_kv_heads as u64,
            head_size: pool.config().head_size as u64,
            cache_dtype,
        };
        let family = if self.config.is_moe() {
            "lfm2_moe"
        } else {
            "lfm2"
        };
        crate::cold_tier::build_model_fingerprint(
            family,
            model_path,
            Some(&config_json),
            &pool_geometry,
            weights,
        )
        .map(
            |fingerprint| crate::transformer::paged_kv_cache_adapter::ColdTierContext {
                manager,
                fingerprint,
                sidecar_policy: Some(sidecar_policy),
            },
        )
    }

    pub(crate) fn attach_cold_tier(
        &mut self,
        context: crate::transformer::paged_kv_cache_adapter::ColdTierContext,
        _weights: &crate::array::memory::WeightsResident,
    ) {
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter.set_cold_tier(context);
        }
    }

    fn cold_restore_boundary(&self, prompt_tokens: u32) -> u32 {
        self.paged_adapter
            .as_ref()
            .filter(|adapter| {
                adapter.cold_tier().is_some_and(|cold| {
                    cold.sidecar_policy.as_ref().is_some_and(|policy| {
                        policy.group() == mlx_paged_attn::ColdGroup::ConvState
                    })
                })
            })
            .map_or(0, |adapter| {
                lfm2_cold_restore_boundary(prompt_tokens, adapter.block_size())
            })
    }

    fn remember_conv_cold_checkpoint_for(&mut self, seq_id: SeqId) {
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return;
        };
        let Some(boundary) = adapter.current_token_count_for(seq_id) else {
            return;
        };
        let block_size = adapter.block_size();
        if boundary == 0 || block_size == 0 || !boundary.is_multiple_of(block_size) {
            return;
        }
        let Some(tokens) = adapter
            .request_tokens_for(seq_id)
            .and_then(|tokens| tokens.get(..boundary as usize))
        else {
            return;
        };
        let caches = if self.active_scheduled_seq == Some(seq_id)
            || (seq_id == 0 && self.active_scheduled_seq.is_none())
        {
            &self.caches
        } else {
            let Some(caches) = self.scheduled_caches.get(&seq_id) else {
                return;
            };
            caches
        };
        let Some(states) = conv_sidecar::snapshot_states(&self.config, caches) else {
            return;
        };
        let checkpoints = self.conv_cold_checkpoints.entry(seq_id).or_default();
        if checkpoints
            .back()
            .is_some_and(|entry| entry.boundary == boundary && entry.tokens.as_slice() == tokens)
        {
            return;
        }
        checkpoints.push_back(Lfm2ConvColdCheckpoint {
            boundary,
            tokens: tokens.to_vec(),
            states,
        });
        while checkpoints.len() > 16 {
            checkpoints.pop_front();
        }
    }

    fn remember_conv_cold_checkpoint(&mut self) {
        self.remember_conv_cold_checkpoint_for(self.active_scheduled_seq.unwrap_or(0));
    }

    fn install_lfm2_conv_cold_sidecar(
        &mut self,
        seq_id: SeqId,
        cached_prefix_len: u32,
    ) -> Result<bool> {
        let (restored, cache_dtype) = {
            let Some(adapter) = self.paged_adapter.as_mut() else {
                return Ok(false);
            };
            let cache_dtype = format!("{:?}", adapter.layer_kv_pool().cache_dtype());
            (adapter.take_restored_sidecar(), cache_dtype)
        };
        let Some(geometry) = conv_sidecar::geometry(&self.config, &cache_dtype) else {
            return Ok(false);
        };
        let expected_layout = conv_sidecar::layout_at(&geometry, cached_prefix_len);
        crate::cold_tier::try_install_cold_sidecar(
            restored,
            &expected_layout,
            |tensors, _boundary| {
                let Some(caches) = conv_sidecar::decode_caches(&self.config, &geometry, tensors)?
                else {
                    return Ok(None);
                };
                if seq_id == 0 || self.active_scheduled_seq == Some(seq_id) {
                    self.caches = caches;
                } else {
                    self.scheduled_caches.insert(seq_id, caches);
                }
                Ok(Some(()))
            },
        )
        .map(|installed| installed.is_some())
    }

    fn capture_lfm2_conv_cold_sidecar(&self, cache_salt: u64) {
        crate::cold_tier::cold_sidecar_counters().record_capture_reached();
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return;
        };
        let Some(cold) = adapter.cold_tier() else {
            return;
        };
        let Some(policy) = cold.sidecar_policy.as_ref() else {
            return;
        };
        if policy.group() != mlx_paged_attn::ColdGroup::ConvState {
            return;
        }
        let block_size = adapter.block_size();
        if block_size == 0 {
            return;
        }
        let request_tokens = adapter.request_tokens();
        let seq_id = self.active_scheduled_seq.unwrap_or(0);
        let Some(prompt_boundary) = self.conv_cold_capture_boundaries.get(&seq_id).copied() else {
            crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
            return;
        };
        let ceiling = adapter
            .cold_captured_blocks()
            .saturating_mul(block_size)
            .min(prompt_boundary);
        let Some(checkpoints) = self.conv_cold_checkpoints.get(&seq_id) else {
            crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
            return;
        };
        let Some(checkpoint) = checkpoints.iter().rev().find(|checkpoint| {
            checkpoint.boundary <= ceiling
                && request_tokens
                    .get(..checkpoint.boundary as usize)
                    .is_some_and(|tokens| tokens == checkpoint.tokens)
        }) else {
            crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
            return;
        };
        let cache_dtype = format!("{:?}", adapter.layer_kv_pool().cache_dtype());
        let Some(geometry) = conv_sidecar::geometry(&self.config, &cache_dtype) else {
            return;
        };
        let blocks = checkpoint.boundary as usize / block_size as usize;
        let mut parent = None;
        for index in 0..blocks {
            let Some(tokens) = checkpoint
                .tokens
                .get(index * block_size as usize..(index + 1) * block_size as usize)
            else {
                return;
            };
            parent = Some(mlx_paged_attn::ColdCacheKey::chain(
                mlx_paged_attn::ColdGroup::ConvState,
                cold.fingerprint,
                parent,
                tokens,
                &[],
                cache_salt,
                index,
            ));
        }
        let Some(key) = parent else {
            return;
        };
        if cold
            .manager
            .contains_in(&key, mlx_paged_attn::ColdGroup::ConvState)
        {
            crate::cold_tier::cold_sidecar_counters().record_already_persisted();
            return;
        }
        let refs = checkpoint.states.iter().collect::<Vec<_>>();
        if MxArray::eval_arrays(&refs).is_err() {
            return;
        }
        let Ok(Some(tensors)) = conv_sidecar::encode_states(&geometry, &checkpoint.states) else {
            return;
        };
        let sidecar = mlx_paged_attn::ColdSidecar {
            key,
            fingerprint: cold.fingerprint,
            layout: conv_sidecar::layout_at(&geometry, checkpoint.boundary),
            tensors,
        };
        let deadline = std::time::Instant::now() + adapter.cold_capture_budget().max_walk;
        match cold.manager.enqueue_sidecar_before(sidecar, deadline) {
            Ok(true) => crate::cold_tier::cold_sidecar_counters().record_enqueued(),
            Ok(false) => crate::cold_tier::cold_sidecar_counters().record_queue_drop(),
            Err(error) => tracing::debug!(
                target: "mlx_core::lfm2::paged",
                "LFM2 conv sidecar enqueue failed: {error}"
            ),
        }
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
    /// `generation_config.json`. Called once by the load path after
    /// construction; the defaults are folded under explicit request params
    /// by the `ChatBackend` sampling/EOS hooks.
    pub(crate) fn set_gen_defaults(&mut self, defaults: crate::engine::ModelGenerationDefaults) {
        self.gen_defaults = defaults;
    }

    /// Forward pass through the full model.
    ///
    /// Follows `lfm2.py:258-279` (Lfm2Model.__call__) + Model.__call__ (tied lm_head).
    ///
    /// Returns logits [B, T, vocab_size].
    pub(crate) fn forward(&mut self, input_ids: &MxArray) -> Result<MxArray> {
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
            // Cooperative-cancel checkpoint (H1b): abort at the chunk
            // boundary. The Err rides the flat engine's
            // `fail_closed_flat_turn` arm — no `save_cache_state`, the
            // session is invalidated, so the partially-advanced
            // conv/attention caches never become a live prefix.
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason("prefill cancelled"));
            }
            let chunk = prompt.slice_axis(1, offset, offset + PREFILL_STEP_SIZE)?;
            {
                let _stream_ctx = StreamContext::new(generation_stream);
                let _logits = self.forward(&chunk)?;
            }
            eval_lfm2_caches(&self.caches)?;
            crate::array::clear_cache();
            offset += PREFILL_STEP_SIZE;
        }
        // The final remainder is a chunk boundary too once at least one
        // looped chunk ran: poll before forwarding it so a cancel landing
        // during the last looped chunk aborts instead of riding through the
        // remainder. `offset == 0` (single-shot) stays uncancellable by
        // design.
        if offset > 0
            && self
                .turn_cancel
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
        {
            return Err(Error::from_reason("prefill cancelled"));
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
    /// The `_internal` suffix keeps this inherent helper from being
    /// shadowed by the [`ChatBackend::reset_caches`] trait method (which
    /// takes a [`ResetScope`]) at concrete-typed call sites. Both engine
    /// scopes dispatch here for the SHARED clear; the explicit command
    /// reset additionally purges the paged prefix cache in the trait impl
    /// (see [`ChatBackend::reset_caches`]).
    fn reset_caches_internal(&mut self) {
        self.caches = init_caches(&self.config);
        self.scheduled_caches.clear();
        self.active_scheduled_seq = None;
        self.conv_cold_checkpoints.clear();
        self.conv_cold_capture_boundaries.clear();
        self.cached_token_history.clear();
        self.cached_image_key = None;
        // Drop any live paged-adapter request. Without this, a
        // finalize_turn_keep_live call from a prior session would leave
        // block_table populated and a subsequent paged turn (`run_paged_turn`
        // → `prime_prefix_state`) could mistakenly take the warm-continue
        // path against stale tokens. NOTE: this alone is NOT a fully cold
        // reset — released
        // full blocks stay content-addressed in the allocator's prefix
        // cache; the EXPLICIT command reset purges them on top of this
        // (`ResetScope::Command` branch of the trait impl).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_all_requests();
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
    /// The `_internal` suffix keeps this inherent helper from being
    /// shadowed by the [`ChatBackend::save_cache_state`] trait method
    /// (which takes [`SaveStateArgs`]); the trait impl and the paged
    /// backend all dispatch here.
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

    /// Run a paged-attention prefill over the full prompt, dispatching
    /// per-layer between paged-attention (full_attention layers) and the
    /// existing conv path (conv layers).
    ///
    /// `full_tokens` is the entire prompt (used for conv layers' prefill
    /// from token 0, UNLESS `skip_conv_reconstruction` is set — see
    /// below). `suffix_tokens` is the new portion beyond the paged
    /// prefix-cache hit (used by `record_tokens` + `update_keys_values`
    /// for the attention layers). `cached_prefix_len` is the paged-cache
    /// hit length. `skip_conv_reconstruction` is
    /// [`conv_state_reusable`]'s verdict from `prime_prefix_state`
    /// (threaded through [`Lfm2PrefixState::conv_state_reusable`]),
    /// is true when `self.caches`'s conv-layer state is already known to be at
    /// the `cached_prefix_len` boundary. The legacy path proves this against
    /// its one live history; the scheduler proves it against exact owner
    /// history plus the sequence-keyed recurrent-state table. In either case
    /// Pass 1 is skipped and Pass 2 runs on the carried state.
    ///
    /// Returns the last position's logits squeezed to `[vocab]`.
    fn run_paged_prefill_chunk(
        &mut self,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
        skip_conv_reconstruction: bool,
    ) -> Result<MxArray> {
        if suffix_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_prefill_chunk called with empty suffix",
            ));
        }

        let suffix_len = suffix_tokens.len() as u32;
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
        // everything in one shot. Pass 1 is ALSO skipped whenever
        // `skip_conv_reconstruction` is set — `prime_prefix_state`
        // already proved `self.caches`'s conv-layer state is at the
        // `cached_prefix_len` boundary for THIS EXACT token prefix (the
        // standard warm-continuation "resend growing conversation"
        // pattern), so re-deriving it via a full re-embed + causal-SDPA
        // reconstruction over the whole cached prefix would be
        // redundant work with no observable effect on the result.

        if cached_prefix_len > 0 && !skip_conv_reconstruction {
            // Pass 1: conv-only prefill over the cached prefix. This
            // brings conv state up to position `cached_prefix_len` so
            // pass 2 can continue from there.
            let prefix = &full_tokens[..(cached_prefix_len as usize)];
            self.run_conv_only_prefill(prefix)?;
        }

        // A hybrid cold/hot prefix is not writable until its out-of-pool
        // ShortConv half is established. A restored sidecar reaches this point
        // with no obligation; a warm foreign hit takes Pass 1 above, then this
        // exact-boundary acknowledgement makes it safe to append the suffix.
        // Arming this before replay would permit KV/conv skew, while recording
        // before acknowledgement fails closed in the adapter.
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_prefill_chunk: paged_adapter is None")
            })?;
            adapter
                .confirm_aux_prefix_primed(cached_prefix_len)
                .map_err(Error::from_reason)?;
            adapter
                .record_tokens(suffix_tokens)
                .map_err(Error::from_reason)?;
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
        self.remember_conv_cold_checkpoint();
        Ok(last)
    }

    fn park_active_scheduled_caches(&mut self) {
        let Some(seq_id) = self.active_scheduled_seq.take() else {
            return;
        };
        let replacement = init_caches(&self.config);
        let caches = std::mem::replace(&mut self.caches, replacement);
        self.scheduled_caches.insert(seq_id, caches);
    }

    fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        self.paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("lfm2 paged adapter is unavailable"))?
            .activate_request(seq_id)
            .map_err(Error::from_reason)?;
        if self.active_scheduled_seq == Some(seq_id) {
            return Ok(());
        }
        self.park_active_scheduled_caches();
        self.caches = self
            .scheduled_caches
            .remove(&seq_id)
            .unwrap_or_else(|| init_caches(&self.config));
        self.active_scheduled_seq = Some(seq_id);
        Ok(())
    }

    fn reset_scheduled_caches_for(&mut self, seq_id: SeqId) {
        if self.active_scheduled_seq == Some(seq_id) {
            self.caches = init_caches(&self.config);
        } else {
            self.scheduled_caches
                .insert(seq_id, init_caches(&self.config));
        }
    }

    fn release_scheduled_caches_for(&mut self, seq_id: SeqId) {
        if self.active_scheduled_seq == Some(seq_id) {
            self.active_scheduled_seq = None;
            self.caches = init_caches(&self.config);
        }
        self.scheduled_caches.remove(&seq_id);
        self.conv_cold_checkpoints.remove(&seq_id);
        self.conv_cold_capture_boundaries.remove(&seq_id);
    }

    fn recurrent_state_bytes_per_seq(&self) -> u64 {
        let conv_layers = self
            .layer_kinds
            .iter()
            .filter(|kind| matches!(kind, Lfm2LayerKind::Conv))
            .count() as u64;
        conv_layers
            .saturating_mul(self.config.conv_l_cache.saturating_sub(1).max(0) as u64)
            .saturating_mul(self.config.hidden_size.max(0) as u64)
            .saturating_mul(2) // production recurrent state is BF16
    }

    fn has_scheduled_caches_for(&self, seq_id: SeqId) -> bool {
        self.active_scheduled_seq == Some(seq_id) || self.scheduled_caches.contains_key(&seq_id)
    }

    fn scheduled_recurrent_bytes(&self) -> u64 {
        let rows =
            self.scheduled_caches.len() as u64 + u64::from(self.active_scheduled_seq.is_some());
        rows.saturating_mul(self.recurrent_state_bytes_per_seq())
    }

    fn stacked_conv_state(
        &mut self,
        seq_ids: &[SeqId],
        layer_idx: usize,
        _dtype: crate::array::DType,
    ) -> Result<MxArray> {
        self.park_active_scheduled_caches();
        let mut states = Vec::with_capacity(seq_ids.len());
        for &seq_id in seq_ids {
            let caches = self.scheduled_caches.get(&seq_id).ok_or_else(|| {
                Error::from_reason(format!(
                    "LFM2 sequence {seq_id} has no scheduled convolution state"
                ))
            })?;
            let state = caches
                .get(layer_idx)
                .and_then(|cache| match cache {
                    Lfm2LayerCache::Conv(cache) => cache.get(0).cloned(),
                    Lfm2LayerCache::Attention(_) => None,
                })
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "LFM2 sequence {seq_id} has no materialized convolution state for layer {layer_idx}"
                    ))
                })?;
            states.push(state);
        }
        MxArray::concatenate_many(states.iter().collect(), Some(0))
    }

    fn scatter_conv_state(
        &mut self,
        seq_ids: &[SeqId],
        layer_idx: usize,
        state: &MxArray,
    ) -> Result<()> {
        for (row, &seq_id) in seq_ids.iter().enumerate() {
            let row_state = state.slice_axis(0, row as i64, row as i64 + 1)?;
            let caches = self.scheduled_caches.get_mut(&seq_id).ok_or_else(|| {
                Error::from_reason(format!(
                    "LFM2 sequence {seq_id} disappeared during convolution-state scatter"
                ))
            })?;
            let cache = caches
                .get_mut(layer_idx)
                .and_then(Lfm2LayerCache::as_conv_cache_mut)
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "LFM2 layer {layer_idx} has no convolution-state slot"
                    ))
                })?;
            cache.set(0, row_state)?;
        }
        Ok(())
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

        // Turn-constant classification, cached once at construction (see the
        // field rustdoc) instead of re-derived every decode step.
        let layer_kinds = &self.layer_kinds;

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

        self.remember_conv_cold_checkpoint();

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

    /// Run one uniform paged decode step for multiple LFM2 requests.
    /// Attention and ShortConv layers both execute once over `[N,1,H]`; the
    /// convolution state is stacked/scattered around each conv layer.
    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        if rows.is_empty() {
            return Err(Error::from_reason(
                "run_paged_decode_step_batched requires at least one row",
            ));
        }
        self.park_active_scheduled_caches();
        let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
            Error::from_reason("run_paged_decode_step_batched: paged adapter is unavailable")
        })?;
        let mut seen = HashSet::with_capacity(rows.len());
        let mut planned_rows = Vec::with_capacity(rows.len());
        for &(seq_id, _) in rows {
            if !seen.insert(seq_id) {
                return Err(Error::from_reason(format!(
                    "run_paged_decode_step_batched received duplicate sequence {seq_id}"
                )));
            }
            let position = adapter.current_token_count_for(seq_id).ok_or_else(|| {
                Error::from_reason(format!(
                    "run_paged_decode_step_batched: unknown sequence {seq_id}"
                ))
            })?;
            planned_rows.push((seq_id, position));
        }

        let mut recorded = Vec::with_capacity(rows.len());
        for &(seq_id, token_id) in rows {
            let result = self
                .paged_adapter
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason("run_paged_decode_step_batched: paged adapter disappeared")
                })?
                .record_token_for(seq_id, token_id);
            if let Err(error) = result {
                for &recorded_seq in recorded.iter().rev() {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_decode_step_batched: paged adapter disappeared during rollback",
                        )
                    })?;
                    adapter
                        .activate_request(recorded_seq)
                        .map_err(Error::from_reason)?;
                    adapter
                        .rollback_last_tokens(1)
                        .map_err(Error::from_reason)?;
                }
                return Err(Error::from_reason(format!(
                    "run_paged_decode_step_batched failed to record sequence {seq_id}: {error}"
                )));
            }
            recorded.push(seq_id);
        }

        let token_ids = rows.iter().map(|&(_, token)| token).collect::<Vec<_>>();
        let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();
        let input_ids = MxArray::from_uint32(&token_ids, &[rows.len() as i64, 1])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        for layer_idx in 0..self.layers.len() {
            let kind = self.layer_kinds[layer_idx];
            let layer: &Lfm2DecoderLayer = unsafe { &*self.layers.as_ptr().add(layer_idx) };
            let conv_state = if kind == Lfm2LayerKind::Conv {
                Some(self.stacked_conv_state(&seq_ids, layer_idx, hidden_states.dtype()?)?)
            } else {
                None
            };
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step_batched: paged adapter dropped")
            })?;
            let (next_hidden, next_conv_state) = layer.forward_paged_or_flat_batched(
                &hidden_states,
                kind,
                adapter,
                &planned_rows,
                conv_state.as_ref(),
            )?;
            hidden_states = next_hidden;
            if let Some(state) = next_conv_state {
                self.scatter_conv_state(&seq_ids, layer_idx, &state)?;
            }
        }
        for &seq_id in &seq_ids {
            self.remember_conv_cold_checkpoint_for(seq_id);
        }
        hidden_states = self.embedding_norm.forward(&hidden_states)?;
        if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)
        } else {
            self.embed_tokens.as_linear(&hidden_states)
        }
    }

    /// Forward the cached prefix tokens through ALL layers (conv state
    /// updated in-place; attention run as a flat causal self-prefill whose
    /// K/V are discarded — the prefix K/V already live in the paged pool).
    /// Used to rebuild the EXACT inter-layer residual stream so conv state is
    /// brought up to the paged cache's `cached_prefix_len` boundary before
    /// pass 2 of `run_paged_prefill_chunk` continues with the suffix.
    ///
    /// Despite the name, this is NOT conv-only — attention must run to feed
    /// downstream conv layers the correct residual, otherwise their state
    /// drifts and paged-CONTINUE diverges from flat. See
    /// `tests/lfm2_paged_vs_flat_parity.rs::lfm2_paged_budget_forced_warm_continue_parity`.
    ///
    /// `run_paged_prefill_chunk` skips calling this entirely when
    /// [`conv_state_reusable`] already proved `self.caches` is at the
    /// `cached_prefix_len` boundary for this exact token prefix (the
    /// warm-continuation fast path) — this reconstruction is only needed to
    /// derive that same state from scratch when the live caches are NOT
    /// already known to hold it.
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
                // EXACT prefix reconstruction. This pass rebuilds the
                // inter-layer residual stream so DOWNSTREAM conv layers see
                // the same input the cold full-prefill (Pass-2) would. An
                // attention layer's contribution to that stream (out_proj +
                // residual + FFN) is NOT recoverable by an identity
                // passthrough, so we must actually run attention here.
                //
                // Run it as a FLAT causal self-prefill: `cache=None` +
                // `mask=None` + `seq_len>1` ⇒ `Lfm2Attention::forward` does
                // internal causal SDPA at RoPE positions 0..prefix_len
                // (attention.rs) with ZERO paged-pool I/O — so it does NOT
                // call `update_keys_values` and cannot trip its
                // already-recorded-suffix alignment check. The K/V computed
                // here are intentionally DISCARDED: Pass-2's suffix attention
                // reads the prefix K/V from the LIVE pool (written in the
                // prior turn, reused verbatim by continue_turn), which this
                // pass never touches. The result is byte-identical to
                // cold/flat for every downstream conv layer's state.
                //
                // A shape-preserving identity passthrough here would be wrong:
                // it would drop the attention contribution and drift downstream
                // conv state, diverging paged-CONTINUE from flat on a warm turn.
                // See tests/lfm2_paged_vs_flat_parity.rs
                // `lfm2_paged_budget_forced_warm_continue_parity`.
                hidden_states = layer.forward(&hidden_states, None, None)?;
            } else {
                // Conv layer: forward through the operator + FFN tail.
                let cache_slot = unsafe {
                    let ptr = self.caches.as_mut_ptr().add(layer_idx);
                    &mut *ptr
                };
                hidden_states = layer.forward(&hidden_states, None, Some(cache_slot))?;
            }
            // Bound the in-flight lazy graph, mirroring Pass-2 in
            // `run_paged_prefill_chunk`. Now that this pass runs causal SDPA
            // over the WHOLE cached prefix (not just conv), MLX would otherwise
            // retain a monolithic prefix DAG until the post-prefill sync —
            // risking a memory peak / OOM on long warm-continue prefixes. The
            // helper eval+clears every `MLX_PAGED_PREFILL_EVAL_INTERVAL` layers;
            // it is BYTE-NEUTRAL (an eval forces materialization, it does not
            // change values), so parity is unaffected.
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }
        Ok(())
    }

    /// Build the per-layer routing list. `FullAttention { paged_idx }`
    /// for full-attention layers (paged_idx counts only those layers in
    /// their original order) and `Conv` for conv layers.
    fn compute_layer_kinds(&self) -> Vec<Lfm2LayerKind> {
        compute_layer_kinds_for(&self.config, self.layers.len())
    }
}

pub(crate) type Lfm2SchedulerState = HybridSchedulerState<Lfm2Inner>;

impl HybridSchedulerBackend for Lfm2Inner {
    type Command = Lfm2Cmd;
    type RestoreTicket = PagedRestoreTicket;
    type OwnerState = Vec<u32>;
    type StepExecutor<'a> = HybridStepExecutor<'a, Self>;

    const SCHEDULER_NAME: &'static str = "LFM2";

    fn paged_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.paged_adapter.as_ref()
    }

    fn paged_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.paged_adapter.as_mut()
    }

    fn max_position_embeddings(&self) -> i32 {
        self.config.max_position_embeddings
    }

    fn recurrent_state_bytes(&self) -> u64 {
        self.recurrent_state_bytes_per_seq()
    }

    fn scheduled_recurrent_bytes(&self) -> u64 {
        self.scheduled_recurrent_bytes()
    }

    fn has_scheduled_recurrent(&self, seq_id: SeqId) -> bool {
        self.has_scheduled_caches_for(seq_id)
    }

    fn activate_scheduled_recurrent(&mut self, seq_id: SeqId) -> Result<()> {
        self.activate_paged_seq(seq_id)
    }

    fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        self.activate_paged_seq(seq_id)
    }

    fn park_active_scheduled_recurrent(&mut self) -> Result<()> {
        self.park_active_scheduled_caches();
        Ok(())
    }

    fn release_scheduled_recurrent_for(&mut self, seq_id: SeqId) {
        self.release_scheduled_caches_for(seq_id);
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
        base: &Self::PrefixState,
        effective_cached_prefix_len: usize,
        suffix_len: usize,
        full_tokens: Vec<u32>,
        first_chunk: bool,
    ) -> Self::PrefixState {
        Lfm2PrefixState {
            effective_cached_prefix_len,
            suffix_len,
            full_tokens,
            conv_state_reusable: !first_chunk || base.conv_state_reusable,
        }
    }

    fn prepare_scheduled_prefix(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        owner_history: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
        _block_size: u32,
    ) -> Result<ScheduledPrefixAdmission<Self::PrefixState, Self::RestoreTicket>> {
        self.activate_paged_seq(seq_id)?;
        let total_budget = tokens.len() as u32;
        let admission = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("LFM2 paged adapter is unavailable"))?
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
        let (plan, restore) = match admission {
            PagedTurnAdmission::Ready(plan) => (plan, None),
            PagedTurnAdmission::Waiting {
                provisional,
                restore,
            } => (provisional, Some(restore)),
        };
        let installed = if restore.is_none() {
            self.install_lfm2_conv_cold_sidecar(seq_id, plan.cached_prefix_len)?
        } else {
            false
        };
        let conv_state_reusable = installed
            || conv_state_reusable(tokens, owner_history, plan.cached_prefix_len as usize);
        if restore.is_none() && !conv_state_reusable {
            self.reset_scheduled_caches_for(seq_id);
        }
        let prefix = Lfm2PrefixState {
            effective_cached_prefix_len: plan.cached_prefix_len as usize,
            suffix_len: plan.suffix_len as usize,
            full_tokens: tokens.to_vec(),
            conv_state_reusable,
        };
        Ok(match restore {
            Some(restore) => ScheduledPrefixAdmission::Waiting {
                provisional: prefix,
                restore,
            },
            None => ScheduledPrefixAdmission::Ready(prefix),
        })
    }

    fn poll_scheduled_restore(
        &mut self,
        seq_id: SeqId,
        restore: &mut Self::RestoreTicket,
        prompt_tokens: &[u32],
        owner_history: &[u32],
        _is_preemption_replay: bool,
    ) -> Result<Option<ScheduledRestoreResult<Self::PrefixState>>> {
        let outcome = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("LFM2 paged adapter is unavailable"))?
            .poll_restore(restore)
            .map_err(Error::from_reason)?;
        let PagedRestorePoll::Ready {
            plan,
            bytes_restored,
            wait,
        } = outcome
        else {
            return Ok(None);
        };
        let installed = self.install_lfm2_conv_cold_sidecar(seq_id, plan.cached_prefix_len)?;
        let conv_state_reusable = installed
            || conv_state_reusable(
                prompt_tokens,
                owner_history,
                plan.cached_prefix_len as usize,
            );
        if !conv_state_reusable {
            self.reset_scheduled_caches_for(seq_id);
        }
        let cold_boundary = self.cold_restore_boundary(prompt_tokens.len() as u32);
        Ok(Some(ScheduledRestoreResult {
            prefix: Lfm2PrefixState {
                effective_cached_prefix_len: plan.cached_prefix_len as usize,
                suffix_len: plan.suffix_len as usize,
                full_tokens: prompt_tokens.to_vec(),
                conv_state_reusable,
            },
            bytes_restored,
            wait,
            materialized_blocks: self
                .paged_adapter
                .as_ref()
                .and_then(|adapter| adapter.block_table_for(seq_id))
                .map(|table| table.num_blocks() as u32)
                .unwrap_or(0),
            profiler_prefill_tokens: if conv_state_reusable {
                plan.suffix_len
            } else {
                prompt_tokens.len() as u32
            },
            extra_prefill_breaks: vec![cold_boundary],
        }))
    }

    fn restore_reserved_blocks(restore: &Self::RestoreTicket) -> u32 {
        restore.reserved_blocks()
    }

    fn extra_prefill_breaks(&self, prompt_tokens: u32, _cached_prefix: u32) -> Vec<u32> {
        vec![self.cold_restore_boundary(prompt_tokens)]
    }

    fn profiler_prefill_tokens(&self, prefix: &Self::PrefixState, prompt_tokens: u32) -> u32 {
        if prefix.conv_state_reusable {
            prefix.suffix_len as u32
        } else {
            prompt_tokens
        }
    }

    fn step_executor(&mut self) -> Self::StepExecutor<'_> {
        HybridStepExecutor::new(self)
    }

    fn execute_barrier(
        &mut self,
        command: Self::Command,
        _owners: SchedulerOwnerContext<'_, Self::OwnerState>,
    ) {
        handle_lfm2_cmd(self, command);
    }
}

pub(crate) struct Lfm2Decode<'a> {
    inner: &'a mut Lfm2Inner,
}

impl DecodeStep for Lfm2Decode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // Eager native forward returns [1, 1, vocab]; `true` signals the
        // caller to `squeeze(Some(&[1]))` down to [1, vocab].
        Ok((self.inner.forward(input_ids)?, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        // lfm2 never force-evals the logits — not even on the
        // budget-forced path (the forced branch discards them lazily).
        MxArray::async_eval_arrays(&[next_token]);
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

    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.turn_cancel = flag;
    }

    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32> {
        tok.im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))
    }

    fn generation_defaults(&self) -> Option<&crate::engine::ModelGenerationDefaults> {
        Some(&self.gen_defaults)
    }

    fn extra_eos_ids(&self) -> Vec<u32> {
        self.gen_defaults.eos_token_ids.clone()
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

    // `resolve_params`: engine default (`extract_chat_params`) is the
    // right behavior for lfm2 — no per-family override needed.
    //
    // `render_prompt`: engine default (jinja `apply_chat_template_sync`
    // with `add_generation_prompt = true`, the request tools, and
    // `resolve_enable_thinking`) is correct for lfm2 (its template itself
    // ignores `enable_thinking`).

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        // Shared clear for BOTH scopes: wipe flat caches + token history +
        // image key and release any live paged request.
        self.reset_caches_internal();
        // The EXPLICIT command reset must restore a fully cold state:
        // `release_request` alone leaves the request's full blocks
        // content-addressed in the allocator's prefix cache, so a
        // reset-then-rerun of the same prompt would take the prefix-hit
        // 1-token-suffix prefill path — whose bf16 reduction order
        // differs from the cold full prefill, enough to flip a greedy
        // near-tie (observed "says," vs "said" at token ~6 on the 1.2b
        // checkpoint). Purge the prefix cache so the next turn replays the
        // cold prefill byte-for-byte.
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

    /// **Safety invariant**: returns ONLY `0` (cache miss) or
    /// `cached_token_history.len()` (exact-append hit or exact match) —
    /// never an intermediate value. The engine's session core routes the
    /// exact-match case (`hit == tokens.len()`) back through the
    /// miss/reset branch: LFM2's short-conv layers have non-invertible
    /// left-padded state and no safe "rewind-by-1" primitive, so the
    /// qwen3 pure-KV exact-match rewind exception is FORBIDDEN here (see
    /// the all-or-nothing contract on
    /// [`ChatBackend::verify_cache_prefix`]).
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
        //
        // Drop-last ALWAYS: the shared decode loop's forward gate
        // (`step_idx + 1 < max_new_tokens && !is_terminal`) skips the final
        // committed token's forward on EVERY exit (length AND EOS / cancel /
        // repetition), so that token is never in the conv/KV cache. Unlike
        // qwen3/gemma4, lfm2 cannot materialize it (re-running a forward
        // would advance the non-invertible short-conv state past the saved
        // history), so the history must instead drop the uncached token to
        // stay aligned. This mirrors lfm2's own paged `save_paged_history`,
        // which already drops-last unconditionally.
        self.save_cache_state_internal(
            args.reuse_cache,
            args.save_tokens,
            args.generated_tokens,
            false,
        );
    }

    fn eval_caches(&self) -> Result<()> {
        // Post-prefill sync of every cache array.
        eval_lfm2_caches(&self.caches)
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        // lfm2 builds its prompt array as int32 (dtype is load-bearing for
        // parity), runs the chunked forward, and folds the last-token
        // slice into the impl using the ACTUAL returned seq len —
        // `chunked_prefill` returns only the final chunk's logits for
        // prompts over `PREFILL_STEP_SIZE`.
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

    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        Ok(Lfm2Decode { inner: self })
    }

    // `finalize_turn`: engine default (`finalize_chat_result`) is correct
    // for lfm2 — its done-chunk construction (the tool_calls finish-reason
    // promotion and the raw_text reasoning scrub) is what both the sync
    // and streaming paths need, so no per-family override.

    fn execution_plan(&self) -> ExecutionPlan {
        ExecutionPlan {
            media: MediaPlan::NONE,
            paged_attention: self.paged_adapter.as_ref().map(|_| PagedAttentionPlan {
                // Delta turns stay PAGED. The session core rebuilds the full
                // token stream (`cached_token_history ++ delta`) before the
                // plan resolves, so a delta reaches `run_paged_turn` as the
                // same strict extension a resent growing conversation
                // produces: the adapter warm-continues the live request and
                // exact-owner turns carry their incremental ShortConv state.
                // Foreign/partial prefix hits rebuild that state once via
                // Pass 1 (`run_conv_only_prefill`). The ShortConv state itself
                // is never represented by the paged adapter. Declaring
                // `supports_delta: false` would route deltas to the flat
                // tail, which tail-prefills onto EMPTY flat attention KV
                // (paged turns never fill it) while conv state sits at the
                // prior position — a corrupt, input-independent
                // continuation.
                supports_delta: true,
            }),
            speculative: None,
        }
    }

    // `execution_plan().media`: `NONE` — LFM2 is text-only.
    // The NAPI entry points additionally carry their own "LFM2 is
    // text-only" pre-checks, which fire before any command is dispatched,
    // so the engine's typed pre-render rejection is a defense-in-depth
    // backstop.
    // `extra_eos_ids` / `stream_skip_special_tokens` / `stream_emitter`
    // / `wired_limit_bytes` (usize::MAX) / `profiler_label` /
    // `has_live_session` (`!cached_token_history.is_empty()`): engine
    // defaults are correct for lfm2 (it has no profiler; the default
    // labels only surface when profiling is enabled).

    fn eos_before_emit(&self) -> bool {
        // lfm2 checks EOS before the cancellation check AND before the
        // token's text is detokenized/emitted, to avoid leaking EOS text —
        // which also resolves the EOS+cancel race as "stop".
        true
    }

    fn augment_performance(&self, _profiler: &DecodeProfiler, _metrics: &mut PerformanceMetrics) {
        // No-op override (gemma4 precedent): lfm2 has no MTP heads
        // (acceptance fields stay None) and carries no `profile_phases`;
        // the default would add profiling-gated extras. Keep the payload
        // byte-stable.
    }

    fn session_media(&self) -> MediaCapabilities {
        // LFM2 has no path that can populate media state. Keep the advertised
        // context truthful even though the legacy cache struct retains an
        // always-None image-key slot for layout symmetry.
        MediaCapabilities::NONE
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // Fresh AND delta turns land here (`supports_delta: true`). For a
        // delta, `args.tokens` is already the full stream
        // (`cached_token_history ++ delta`, rebuilt by the session core
        // before plan resolution) and the engine forces `reuse_cache = true`,
        // so `prime_prefix_state` sees the exact strict-extension shape a
        // resent growing conversation produces — the adapter warm-continues
        // the live request and `run_paged_prefill_chunk` carries the exact
        // incremental conv state (or performs one fail-closed Pass-1 rebuild
        // for a foreign/partial prefix hit).
        //
        // The model-neutral `run_paged_turn` drives the whole turn
        // through `<Lfm2Inner as PagedBackend>` (prime → prefill →
        // begin_paged_decode → decode loop → save).
        //
        // Think-budget force ordering: `run_decode_loop` forces `</think>`
        // FORCE-before-OBSERVE — it peeks `should_force_think_end()` for the
        // NEXT token before observing the current one, so a budget of N
        // yields N+1 reasoning tokens. That is the engine's
        // unit-test-locked contract (engine::decode
        // `budget_forcing_injects_think_end_token`; the observe-after-force
        // note at decode.rs ~412), shared by lfm2's flat path, paged path,
        // and qwen3 — so flat and paged agree token-for-token under a
        // mid-`<think>` budget force. `lfm2_paged_vs_flat_greedy_token_parity`
        // (budget 32) holds because both paths share this loop.
        crate::engine::paged_turn::run_paged_turn(self, args)
    }
}

impl DecodeStep for Lfm2PagedDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // NOT on the hot path — the engine drives decode via
        // `forward_with_token` (which hands the scalar the loop already
        // read). Kept only to satisfy the trait; extract then delegate.
        let token_id = input_ids.item_at_int32(0)? as u32;
        self.forward_with_token(input_ids, token_id)
    }

    fn forward_with_token(
        &mut self,
        _input_ids: &MxArray,
        token_id: u32,
    ) -> Result<(MxArray, bool)> {
        // `token_id` is HANDED by the engine (already read once at the loop
        // top via `y.item_at_int32`), so `_input_ids` is unused (kept for
        // signature parity); `run_paged_decode_step` re-records the token
        // from the scalar.
        let logits = self
            .inner
            .run_paged_decode_step(token_id)?
            .squeeze(Some(&[1]))?;

        // `run_paged_decode_step` returns [1, 1, vocab] and the `squeeze([1])`
        // above already reduces it to [1, vocab], so `needs_squeeze = FALSE`.
        // ⚠ POLARITY IS INVERTED vs qwen3 (whose `Qwen3PagedDecode::forward`
        // returns `true` from a direct `run_paged_decode_step`) — lfm2's
        // squeeze lives in this body, so returning `true` here would
        // double-squeeze and break the sampler.
        Ok((logits, false))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        // Single SYNCHRONOUS eval of `next_token` pulls the logits AND the
        // paged K/V writes through the dependency chain (one sync wait); the
        // loop-top `y.eval()` then no-ops on the already-materialized token.
        //
        // NOT `async_eval_arrays([next_token, logits])` (the qwen3-style
        // schedule): lfm2's compiled forward has negligible per-step CPU work
        // (~110us issue vs ~5.4ms GPU/token, bandwidth-bound), so the async
        // two-wait (bottom `async_eval` + loop-top `y.eval`) buys ZERO overlap
        // and costs ~5% vs the single sync wait.
        //
        // `_budget_forced` is unused: a forced final token does NOT leave its
        // compiled K/V co-output lazy (the single sync eval above pulls the
        // K/V writes through the dependency chain regardless). Cross-turn
        // parity on a budget-forced length exit instead depends on the conv
        // Pass-1 running attention in `run_conv_only_prefill`. See
        // tests/lfm2_paged_vs_flat_parity.rs
        // `lfm2_paged_budget_forced_warm_continue_parity`.
        next_token.eval();
    }

    fn maintain_cache(&mut self, step: i32) {
        // Paged cadence — per-step cache clear.
        crate::array::maybe_clear_cache_for_paged_step(step);
    }

    // `materialize_final` — DO NOT override (default no-op). lfm2 PAGED must
    // NOT re-run a decode step for the final length-exit token. The
    // drop-on-length `save_paged_history` already keeps history aligned with
    // the adapter (see the token-accounting proof on
    // `<Lfm2Inner as PagedBackend>::save_paged_history`), so no extra forward
    // is needed.
    //
    // `end_decode` — DO NOT override (default Ok(())). The decode loop updates
    // the active request's convolution state in place. The scheduler parks that
    // state in its sequence-keyed table after finalization; the legacy lane
    // leaves it in `self.caches` for the next exact-owner continuation.
}

/// lfm2 paged prefix state — the effective prefix/suffix split from
/// `prepare_turn_with_max_cache_hit_tokens`, PLUS the full prompt tokens.
///
/// `full_tokens` is the load-bearing gap vs qwen3's 2-usize
/// `Qwen3PrefixState`: the engine hands `paged_prefill` ONLY the suffix
/// (`tokens[effective_cached_prefix_len..]`), but lfm2's
/// `run_paged_prefill_chunk` needs the FULL prompt for the conv-only Pass-1
/// over `full_tokens[..cached_prefix_len]`. Stash `plan.to_vec()` during the
/// prime so `paged_prefill` can recover it.
pub(crate) struct Lfm2PrefixState {
    effective_cached_prefix_len: usize,
    suffix_len: usize,
    full_tokens: Vec<u32>,
    /// [`conv_state_reusable`]'s verdict, computed once in
    /// `prime_prefix_state` and consumed by `paged_prefill` to decide
    /// whether `run_paged_prefill_chunk` may skip Pass 1
    /// (`run_conv_only_prefill`) over the cached prefix.
    conv_state_reusable: bool,
}

impl PagedPrefix for Lfm2PrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

impl PagedBackend for Lfm2Inner {
    type PagedDecode<'a>
        = Lfm2PagedDecode<'a>
    where
        Self: 'a;
    type PrefixState = Lfm2PrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        _reuse_cache: bool,
        _block_size: usize,
        extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        // Lazy decode allocation: pass the prompt length only (decode blocks
        // grow on-demand via `record_tokens`). vLLM-style exact-prefix cap:
        // leave at least one prompt token to prefill so the decoder always
        // has something to consume.
        let total_budget = plan.len() as u32;
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let seq_id: u32 = 0;
        // Adapter-owned warm-continue/cold-start lifecycle. Pass LITERAL
        // `true` for lfm2's can_continue term — the adapter ANDs reuse_cache
        // into its own can_continue, so `true` keeps the warm predicate
        // always-eligible. ⚠ Do NOT thread the engine's `reuse_cache` here
        // (qwen3 does); for lfm2 the engine's `reuse_cache` instead drives
        // finalize/save. Suffix blocks are allocated inside prepare_turn; do
        // not re-allocate.
        let turn_plan = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| {
                Error::from_reason(
                    "prime_prefix_state: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?
            .prepare_turn_with_max_cache_hit_tokens(
                seq_id,
                plan,
                total_budget,
                true,
                extra_keys,
                cache_salt,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason)?;

        let cached_prefix_len = turn_plan.cached_prefix_len as usize;

        let installed_conv_sidecar =
            self.install_lfm2_conv_cold_sidecar(seq_id, turn_plan.cached_prefix_len)?;

        // Conv-state reuse fast path (see `conv_state_reusable`): when this
        // turn strictly extends the immediately preceding successful turn's
        // saved history byte-for-byte, `self.caches`'s conv-layer state is
        // ALREADY at the `cached_prefix_len` boundary — keep it live instead
        // of paying for a full re-embed + causal-SDPA reconstruction pass
        // over the whole cached prefix (`run_paged_prefill_chunk` Pass 1).
        // Any mismatch (first turn, aborted-since-last-save, or a
        // foreign/partial prefix hit) falls through to the existing
        // unconditional reset — `run_paged_turn` is family-neutral and will
        // NOT do this for us; skip it and conv state goes stale across
        // turns. Carried incremental state is the sole oracle.
        let reused_conv_state = installed_conv_sidecar
            || conv_state_reusable(plan, &self.cached_token_history, cached_prefix_len);
        if !reused_conv_state {
            self.caches = init_caches(&self.config);
            self.cached_token_history.clear();
            self.cached_image_key = None;
        }
        self.last_paged_prefill_reused_conv_state = reused_conv_state;

        Ok(Lfm2PrefixState {
            effective_cached_prefix_len: cached_prefix_len,
            suffix_len: turn_plan.suffix_len as usize,
            // Conv Pass-1 needs the FULL prompt (not just the suffix).
            full_tokens: plan.to_vec(),
            conv_state_reusable: reused_conv_state,
        })
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        // `run_paged_prefill_chunk` records the suffix in the adapter, runs
        // conv Pass-1 over the cached prefix from `full_tokens` (unless
        // `prefix.conv_state_reusable` says the live caches already cover
        // it), then the full forward over the suffix, and folds in the
        // last-token slice (returns `[vocab]`). The engine fires the
        // post-prefill `synchronize_and_clear_cache` AFTER this returns
        // (NOT here).
        let cached_prefix = prefix.effective_cached_prefix_len as u32;
        let split_boundary = self.cold_restore_boundary(prefix.full_tokens.len() as u32);
        self.conv_cold_capture_boundaries
            .insert(self.active_scheduled_seq.unwrap_or(0), split_boundary);
        if split_boundary > cached_prefix {
            let split = (split_boundary - cached_prefix) as usize;
            if split < suffix_tokens.len() {
                self.run_paged_prefill_chunk(
                    &prefix.full_tokens,
                    &suffix_tokens[..split],
                    cached_prefix,
                    prefix.conv_state_reusable,
                )?;
                return self.run_paged_prefill_chunk(
                    &prefix.full_tokens,
                    &suffix_tokens[split..],
                    split_boundary,
                    true,
                );
            }
        }
        self.run_paged_prefill_chunk(
            &prefix.full_tokens,
            suffix_tokens,
            cached_prefix,
            prefix.conv_state_reusable,
        )
    }

    fn begin_paged_decode(&mut self) -> Result<Self::PagedDecode<'_>> {
        Ok(Lfm2PagedDecode { inner: self })
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool, cache_salt: u64) {
        // Terminal lifecycle. Success: keep the request live across turns
        // when reuse is on so the next turn's `continue_turn` builds on the
        // partial trailing block's live K/V; otherwise register full blocks
        // for reuse + release. Infallible (`let _ =` every call — a teardown
        // failure must not mask the turn result).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            if reuse_cache {
                let _ = adapter.finalize_turn_keep_live(&[], cache_salt);
            } else {
                let _ = adapter.register_full_blocks_for_reuse(&[], cache_salt);
            }
        }
        // The K/V capture above establishes the authoritative cold frontier;
        // publish exactly the deepest matching ShortConv checkpoint beneath
        // it, then release the request. This is the same joint-boundary
        // contract as vLLM's KV+Mamba hybrid coordinator.
        self.capture_lfm2_conv_cold_sidecar(cache_salt);
        if !reuse_cache {
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            self.conv_cold_checkpoints.remove(&0);
            self.conv_cold_capture_boundaries.remove(&0);
        }
    }

    fn abort_paged_turn(&mut self) {
        // Error-path teardown: release fully, partial block_table state is
        // unsafe to keep around. Release ONLY — never register / keep live.
        // Infallible (`let _ =` — must not mask the turn's error).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
        self.conv_cold_checkpoints
            .remove(&self.active_scheduled_seq.unwrap_or(0));
        self.conv_cold_capture_boundaries
            .remove(&self.active_scheduled_seq.unwrap_or(0));
        // Invalidate the conv-state-reuse invariant `conv_state_reusable`
        // depends on: `self.caches`'s conv-layer state may now reflect a
        // partially-executed (aborted) turn rather than the exact position
        // `cached_token_history.len()` records (e.g. Pass 2 or a decode step
        // ran partway before the error). Clearing `cached_token_history`
        // forces the NEXT `prime_prefix_state` call's length/content check
        // to fail closed (`> 0` guard), taking the `init_caches` reset and
        // one safe Pass-1 rebuild.
        self.cached_token_history.clear();
    }

    fn paged_perf_prefill_tokens(&self, prompt_token_count: usize, suffix_len: usize) -> usize {
        // LFM2 reprefills the FULL prompt through conv layers on a foreign or
        // partial attention-prefix hit (`run_paged_prefill_chunk` Pass-1), so
        // ttft measures full-prompt work and the throughput numerator must
        // be the FULL prompt — NOT the attention suffix (pinned by the
        // `lfm2_paged_prefill_tps_is_full_prompt_scale_on_warm_reuse` guard).
        // The default (`suffix_len`) is the standard-KV qwen behavior, which
        // would under-report lfm2's prefill tok/s by the cache-hit ratio.
        //
        // When `prime_prefix_state` took the exact-owner carried-state path
        // this turn (`last_paged_prefill_reused_conv_state`),
        // Pass 1 never ran — the ACTUAL prefill work this turn was
        // suffix-scale, so reporting the full-prompt count here would
        // inflate `prefill_tokens_per_second` by the cache-hit ratio
        // instead of under-reporting it. Report the honest numerator for
        // whichever amount of work actually happened.
        if self.last_paged_prefill_reused_conv_state {
            suffix_len
        } else {
            prompt_token_count
        }
    }

    fn paged_decode_stream(&self, _generation_stream: Stream) -> Stream {
        // Run the compiled-paged DECODE on the canonical DEFAULT stream, NOT
        // the per-turn `generation_stream`. lfm2's compiled forward holds
        // persistent per-layer K/V pools; running it on a queue separate from
        // the shared loop's top-of-iteration `y.eval()` (always on the default
        // stream) forces a cross-queue completion-wait every token (~5% on
        // bandwidth-bound decode). Keeping paged decode on the default stream
        // — while `chunked_prefill` and `paged_prefill` use
        // `generation_stream` — gives a single-stream decode cadence. See the
        // `PagedBackend::paged_decode_stream` doc for the full mechanism.
        Stream::default(crate::stream::DeviceType::Gpu)
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        _keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        // lfm2 INVERSE convention vs qwen3 (Decision 1): lfm2 paged ALWAYS
        // drops the last token, regardless of the engine's `keep_all`
        // (length-exit) signal, because the paged decode loop NEVER forwards
        // the LAST sampled token through `run_paged_decode_step` — so the
        // last `generated` entry is NOT in the adapter / conv caches and must
        // be dropped to keep the saved history aligned with the live cache
        // state.
        //
        // We pass `last_token_in_cache=false` UNCONDITIONALLY to the FLAT
        // helper, which does the drop-last trim (model.rs `save_cache_state_internal`)
        // AND the `!reuse_cache → reset_caches_internal()` branch (Decision 2:
        // respect the engine's `reuse_cache`). This writes ONLY
        // `cached_token_history` (+ image-key reset inside the helper's reset
        // arm) — never the FLAT `cached_kv_*`, which the paged path never
        // fills. `_keep_all` is intentionally ignored (it is the qwen3 signal).
        //
        // Token accounting (adapter cursor == saved history), records-at-top
        // in run_paged_decode_step / forward; engine forward gate
        // `step_idx+1 < N && !is_terminal`:
        //  * LENGTH exit (k==N): adapter = prompt+(N-1) [last forward skipped];
        //    drop-last → history = prompt+(N-1). MATCH.
        //  * EARLY-STOP at g[k-1] (k<N): adapter = prompt+(k-1) [terminal
        //    forward skipped]; drop-last → history = prompt+(k-1). MATCH.
        self.save_cache_state_internal(reuse_cache, save_tokens, generated, false);
        // conv-state save is in-process and infallible (no recurrent-cache
        // eval/clone that can fail like the MoE GDN checkpoint).
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        _keep_all: bool,
    ) -> bool {
        // lfm2 ALWAYS drops the last token (see `save_paged_history`), so the
        // to-be-saved history length is `prompt_len + (generated.len() - 1)`
        // (or `prompt_len` when nothing was generated). Roll the adapter back
        // to that length so the next turn's warm-continue gate
        // (`prompt.starts_with(request_tokens())`) is not defeated by a
        // trailing token the pipelined loop recorded at the loop top before
        // the stop-check. `_keep_all` is intentionally ignored (qwen3 signal).
        //
        // Token accounting: on BOTH length and early-stop exits the to-be-saved
        // history equals the adapter cursor (the final/terminal forward was
        // skipped, see the proof in `save_paged_history`), so `surplus` is 0
        // and this is a true no-op for lfm2 — but the rollback is kept as the
        // defensive contract the trait mandates (and would fire if a future
        // change made the adapter over-record).
        let Some(adapter) = self.paged_adapter.as_mut() else {
            return true;
        };
        let history_len = if generated.is_empty() {
            0
        } else {
            generated.len() - 1
        };
        let target_len = prompt_len + history_len;
        let surplus = adapter.request_tokens().len().saturating_sub(target_len);
        if surplus > 0
            && let Err(e) = adapter.rollback_last_tokens(surplus as u32)
        {
            tracing::warn!(
                target: "mlx_core::lfm2::paged",
                "reconcile_paged_request_tokens: rollback_last_tokens({surplus}) failed \
                 (finalize releases the request; next turn cold-prefills): {e}",
            );
            return false;
        }
        true
    }
}

/// Free-function form of [`Lfm2Inner::compute_layer_kinds`] usable without a
/// `self` borrow. Identical mapping: `FullAttention { paged_idx }` for
/// attention layers (paged_idx counts only those, in original order) and
/// `Conv` otherwise.
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

pub(crate) fn init_caches(config: &Lfm2Config) -> Vec<Lfm2LayerCache> {
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
    /// Dedicated model thread owning `Lfm2SchedulerState`. LFM2 is chat-only
    /// (no training/generate variants); `Lfm2Cmd` wraps the model-neutral chat
    /// protocol and adds scheduler telemetry without exposing model state
    /// outside the thread.
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
    /// RAII debit for the native paged KV pool, kept separate from weights so
    /// the global coordinator can account both deterministic residents.
    pub(crate) _pool_cache_limit_guard: Option<crate::cache_limit::PoolCacheLimitGuard>,
}

#[napi]
impl Lfm2Model {
    /// Load an LFM2 model from a directory containing safetensors and config.json.
    #[napi]
    pub async fn load(model_path: String) -> Result<Lfm2Model> {
        Lfm2Model::load_from_dir(&model_path).await
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
        tokio::sync::mpsc::Receiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) = crate::model_thread::stream_channel(
            crate::engine::napi_glue::CHAT_STREAM_NATIVE_QUEUE_LIMIT,
        );
        self.thread
            .send(Lfm2Cmd::Chat(Box::new(ChatCmd::StreamSessionStart {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            })))?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Test-only entry point that dispatches
    /// `ChatStreamSessionContinue` and returns the raw mpsc receiver
    /// the model thread writes into.
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
            .send(Lfm2Cmd::Chat(Box::new(ChatCmd::StreamSessionContinue {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            })))?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Get the model configuration.
    #[napi]
    pub fn get_config(&self) -> Lfm2Config {
        self.config.clone()
    }

    /// Native admission capacity for the server's per-model semaphore.
    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        if self.paged_active && !Lfm2SchedulerState::force_serial() {
            scheduler_max_num_seqs_for(32) as u32
        } else {
            1
        }
    }

    /// Snapshot scheduler occupancy and paged-pool admission telemetry.
    #[napi]
    pub async fn scheduler_stats(&self) -> Result<engine::SchedulerStatsJs> {
        send_and_await(&self.thread, |reply| Lfm2Cmd::SchedulerStats { reply }).await
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

crate::models::chat_napi::chat_napi_surface! {
    class: Lfm2Model,
    thread_cmd: crate::models::lfm2::model::Lfm2Cmd,
    thread: direct,
    image_guard: text_only,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}

#[cfg(test)]
mod prefix_cache_decision_tests {
    //! Pure-logic coverage of the prefix-cache decision tree — no model
    //! load required. The verifier `Lfm2Inner::verify_cache_prefix`
    //! returns either `0` (miss) or `cached_token_history.len()` (exact
    //! prefix relation). The engine session core (and the paged turn
    //! path) then classify that value plus the
    //! incoming prompt length into
    //! [`PrefixCacheDecision::StrictExtendHit`] (warm-reuse, skip the
    //! cached prefix, prefill only the tail) vs
    //! [`PrefixCacheDecision::Miss`] (reset caches + re-init + full
    //! prefill).
    //!
    //! The four cases covered below pin the invariant:
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
        // the prompt + generated tokens. The tests here guard this
        // invariant against any regression.
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
mod conv_state_reuse_tests {
    //! Pure-logic coverage of [`conv_state_reusable`] — no model load
    //! required. Guards the `prime_prefix_state` fast path that skips
    //! `run_paged_prefill_chunk`'s Pass 1 (`run_conv_only_prefill`)
    //! whenever `self.caches`'s conv-layer state is provably already at
    //! the incoming cached-prefix boundary.

    use super::conv_state_reusable;

    #[test]
    fn zero_cached_prefix_is_not_reusable() {
        // First turn ever (or a cross-session/foreign miss the paged
        // adapter itself already reported as 0): nothing to reuse.
        assert!(!conv_state_reusable(&[1, 2, 3], &[], 0));
    }

    #[test]
    fn strict_extension_of_saved_history_is_reusable() {
        // The standard "resend growing conversation" pattern: the new
        // prompt is `cached_token_history` plus new tail tokens, and the
        // paged adapter's own hit length equals the saved history length
        // exactly.
        let history = vec![1u32, 2, 3, 4, 5];
        let plan = vec![1u32, 2, 3, 4, 5, 6, 7];
        assert!(conv_state_reusable(&plan, &history, history.len()));
    }

    #[test]
    fn length_mismatch_is_not_reusable() {
        // Paged-adapter hit length shorter than the saved history (a
        // partial/foreign prefix hit, or a branched conversation that
        // diverges before the end of the prior turn): the live caches
        // are NOT known to be at this exact boundary.
        let history = vec![1u32, 2, 3, 4, 5];
        let plan = vec![1u32, 2, 3, 9, 9, 9];
        assert!(!conv_state_reusable(&plan, &history, 3));

        // Hit length longer than the saved history (foreign cross-session
        // KV-block hit dwarfing this instance's own history): also unsafe.
        assert!(!conv_state_reusable(&plan, &history, 6));
    }

    #[test]
    fn content_divergence_is_not_reusable() {
        // Same lengths, but the prefix bytes differ — a coincidental
        // length match against an unrelated session's history.
        let history = vec![1u32, 2, 3, 4, 5];
        let plan = vec![1u32, 2, 9, 4, 5, 6];
        assert!(!conv_state_reusable(&plan, &history, history.len()));
    }

    #[test]
    fn exact_resend_capped_below_full_length_is_not_reusable() {
        // `prime_prefix_state` caps the paged adapter's own hit at
        // `plan.len() - 1` (never lets a cache hit consume the whole
        // prompt), so an identical resend with zero new tokens reports
        // `cached_prefix_len == history.len() - 1`, not `history.len()`.
        // Mirrors `classify_prefix_cache_decision`'s exact-match-is-miss
        // invariant.
        let history = vec![1u32, 2, 3, 4, 5];
        let plan = history.clone();
        assert!(!conv_state_reusable(&plan, &history, history.len() - 1));
    }

    #[test]
    fn chains_across_consecutive_growing_turns() {
        // Turn 2 reuses against turn 1's saved history; turn 3 reuses
        // against turn 2's (longer) saved history. Proves the invariant
        // is not a one-shot check.
        let history_after_turn1 = vec![1u32, 2, 3];
        let plan_turn2 = vec![1u32, 2, 3, 4, 5];
        assert!(conv_state_reusable(
            &plan_turn2,
            &history_after_turn1,
            history_after_turn1.len()
        ));

        let history_after_turn2 = vec![1u32, 2, 3, 4, 5, 6];
        let plan_turn3 = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        assert!(conv_state_reusable(
            &plan_turn3,
            &history_after_turn2,
            history_after_turn2.len()
        ));
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

    use super::{Lfm2Inner, Lfm2SchedulerState, compute_layer_kinds_for};
    use crate::array::DType;
    use crate::models::lfm2::Lfm2Config;

    #[test]
    fn cold_restore_boundary_keeps_one_prompt_token_uncached() {
        assert_eq!(super::lfm2_cold_restore_boundary(0, 16), 0);
        assert_eq!(super::lfm2_cold_restore_boundary(15, 16), 0);
        assert_eq!(super::lfm2_cold_restore_boundary(16, 16), 0);
        assert_eq!(super::lfm2_cold_restore_boundary(17, 16), 16);
        assert_eq!(super::lfm2_cold_restore_boundary(32, 16), 16);
        assert_eq!(super::lfm2_cold_restore_boundary(33, 16), 32);
        assert_eq!(super::lfm2_cold_restore_boundary(33, 0), 0);
    }

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
            persist_paged_cache: None,
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
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
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
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
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
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
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
        let logits = match inner.run_paged_prefill_chunk(&prompt, &prompt, 0, false) {
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
        // The all-conv rejection is a paged-path guard; it only runs when the
        // adapter is built, which needs the Metal backend. Skip elsewhere.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
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

    #[test]
    fn inner_caches_layer_kinds_matching_fresh_compute() {
        // The paged decode stepper consumes the cached classification
        // instead of re-deriving it per step; pin cache == fresh compute
        // (paged OFF so construction needs no Metal pool).
        let cfg = paged_tiny_config(Some(false));
        let inner = Lfm2Inner::new(cfg.clone()).expect("construct");
        let fresh = compute_layer_kinds_for(&cfg, cfg.num_hidden_layers as usize);
        assert_eq!(
            inner.layer_kinds, fresh,
            "cached layer classification must equal a fresh compute over the same config"
        );
    }

    #[test]
    fn cache_owner_release_drops_history_sequence_and_recurrent_state() {
        let inner = Lfm2Inner::new(paged_tiny_config(Some(false))).expect("construct");
        let mut state = Lfm2SchedulerState::new(inner).expect("construct scheduler state");
        state.owner_sequences.insert("stateless-owner".into(), 9);
        state
            .owner_states
            .insert("stateless-owner".into(), vec![7, 8]);
        state.inner.reset_scheduled_caches_for(9);

        state
            .release_cache_owner_now("stateless-owner")
            .expect("release owner");

        assert!(!state.owner_sequences.contains_key("stateless-owner"));
        assert!(!state.owner_states.contains_key("stateless-owner"));
        assert!(!state.inner.has_scheduled_caches_for(9));
    }

    #[test]
    fn stacked_conv_state_fails_closed_for_unknown_sequence() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = Lfm2Inner::new(cfg).expect("construct");
        let error = match inner.stacked_conv_state(&[99], 0, DType::BFloat16) {
            Ok(_) => panic!("missing scheduler state must not be fabricated as zeros"),
            Err(error) => error,
        };
        assert!(error.reason.contains("99"));
    }
}
