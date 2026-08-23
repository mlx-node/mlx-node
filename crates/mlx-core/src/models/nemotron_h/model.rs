//! NemotronH model: NemotronHInner + ChatBackend/MtpBackend + NAPI class.
//!
//! Mamba-2 SSM states and attention K/V live in per-layer NemotronHLayerCache
//! slots owned by the model thread; the speculative MTP head runs the engine's
//! flat run_mtp_turn loop with a depth-1 drafter that owns its own KV.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::MxArray;
use crate::engine::ThinkingPolicy;
use crate::engine::backend::{
    ChatBackend, DecodeStep, FinalizeArgs, MtpBackend, MtpStepper, MtpTurnSetup, PagedBackend,
    PagedPrefix, ResetScope, SaveStateArgs, SpecFrontier, StreamEmitter, TurnOutput, TurnSetup,
    WholeTurnArgs,
};
use crate::engine::cmd::{ChatCmd, FromChatCmd, handle_chat_cmd};
use crate::engine::decode::{DecodeLoopArgs, StreamingCtx, run_decode_loop};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridSchedulerCommand, HybridSchedulerState, HybridStepExecutor,
    NoRestoreTicket, ScheduledPrefixAdmission, pool_tokens_after_recurrent, scheduled_turn_context,
    scheduler_max_num_seqs_for, scheduler_per_seq_context_override,
};
use crate::engine::plan::{
    DecoderPlan, ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan, SpeculativeKind,
    SpeculativePlan,
};
use crate::engine::{self};
use crate::model_thread::{ResponseTx, send_and_await};
use crate::nn::{Embedding, RMSNorm};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, PagedTurnPlanReason, SeqId};

use super::config::NemotronHConfig;
use super::decoder_layer::{NemotronHDecoderLayer, NemotronHMixer};
use super::layer_cache::{NemotronHLayerCache, NemotronHLayerSnapshot};
use super::mamba2::Mamba2State;
use super::mtp::NemotronHMtpModule;

/// Chunk size for the chunked prefill (tokens per chunk).
pub(crate) const PREFILL_STEP_SIZE: i64 = 2048;

/// The distinguished error a chunk-boundary cancel poll aborts a prefill
/// with. Shared by the producers (every chunked prefill in this file) and
/// the one consumer that must tell a cancellation apart from a genuine
/// failure (`mtp_seed_aborted`), so the two cannot drift.
pub(crate) const PREFILL_CANCELLED: &str = "prefill cancelled";

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum NemotronHCmd {
    Chat(Box<ChatCmd>),
    SchedulerStats {
        reply: ResponseTx<engine::SchedulerStatsJs>,
    },
    /// Test-only: `(cached_token_history.len(), attention kv_offset,
    /// flat_mtp_caches_desynced, flat_mtp_last_rollback_unemitted)`. The first two
    /// are the seam invariant: the flat trunk must never sit ahead of the token
    /// history that keys it.
    #[doc(hidden)]
    MtpFlatStateForTest {
        reply: ResponseTx<(usize, i32, bool, usize)>,
    },
}

impl FromChatCmd for NemotronHCmd {
    #[inline]
    fn from_chat(cmd: ChatCmd) -> Self {
        NemotronHCmd::Chat(Box::new(cmd))
    }
}

impl HybridSchedulerCommand for NemotronHCmd {
    fn as_chat(&self) -> Option<&ChatCmd> {
        match self {
            Self::Chat(chat) => Some(chat),
            Self::SchedulerStats { .. } | Self::MtpFlatStateForTest { .. } => None,
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

/// Route a thread command to the engine's chat handler.
pub(crate) fn handle_nemotron_h_cmd(inner: &mut NemotronHInner, cmd: NemotronHCmd) {
    match cmd {
        NemotronHCmd::Chat(chat) => handle_chat_cmd(inner, *chat),
        NemotronHCmd::SchedulerStats { reply } => {
            let _ = reply.send(Ok(engine::scheduler::SchedulerStats::default().to_js()));
        }
        NemotronHCmd::MtpFlatStateForTest { reply } => {
            // `as_kv_cache_mut`, not `as_kv_cache`: the shared accessor is
            // `#[cfg(any(test, debug_assertions))]` and is gone from a release
            // build, which an integration test links WITHOUT `cfg(test)`.
            let kv_offset = inner
                .caches
                .iter_mut()
                .find_map(|c| c.as_kv_cache_mut())
                .map(|kv| kv.get_offset())
                .unwrap_or(-1);
            let _ = reply.send(Ok((
                inner.cached_token_history.len(),
                kv_offset,
                inner.flat_mtp_caches_desynced,
                inner.flat_mtp_last_rollback_unemitted,
            )));
        }
    }
}

/// Internal model state owned exclusively by the dedicated model thread.
pub(crate) struct NemotronHInner {
    pub(crate) config: NemotronHConfig,
    /// The in-flight turn's cooperative-cancel flag.
    pub(crate) turn_cancel: Option<Arc<AtomicBool>>,
    pub(crate) embedding: Embedding,
    pub(crate) layers: Vec<NemotronHDecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    /// Untied lm_head (tie_word_embeddings=false); NVFP4-quantized or dense.
    pub(crate) lm_head: Option<crate::models::qwen3_5_moe::quantized_linear::LinearProj>,
    pub(crate) caches: Vec<NemotronHLayerCache>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    pub(crate) cached_token_history: Vec<u32>,
    /// Multi-Token Prediction head - Some when config.n_mtp_layers > 0.
    pub(crate) mtp: Option<NemotronHMtpModule>,
    /// Set true by the loader only after the MTP weight set is COMPLETE, and never
    /// written again: `NemotronHModel::mtp_active` is a load-time snapshot of
    /// `has_mtp_weights()`, so a runtime writer here would make the `hasMtpWeights()`
    /// NAPI getter lie.
    pub(crate) mtp_weights_loaded: bool,
    /// Handover slot for the turn's seeded MTP drafter cache plus its committed
    /// length. `begin_mtp_decode` hard-errors on `None` rather than drafting from an
    /// empty history. Per-turn only: a surviving seed would describe the last turn.
    pub(crate) pending_mtp_draft_seed: Option<(Vec<NemotronHLayerCache>, i32)>,
    /// Flat MTP mid-cycle-stop desync latch: forces a full re-prefill on the
    /// next AR turn.
    pub(crate) flat_mtp_caches_desynced: bool,
    /// The `rollback_unemitted` of the most recent flat MTP turn (0 when it ended on
    /// a clean cycle boundary). Diagnostic and test seam ONLY: it is the
    /// latch-INDEPENDENT way to ask "did that turn stop mid-cycle?", so a gate keyed
    /// off it is not a probe of its own guard.
    pub(crate) flat_mtp_last_rollback_unemitted: usize,
    /// Lifetime count of drafter seeds that failed and degraded their turn to plain AR.
    /// A cancellation is not a failure and never increments it. Diagnostic and test
    /// seam ONLY: it is the observable that separates the cancel arm of
    /// `mtp_seed_aborted` from the fallback arm, which are otherwise
    /// post-condition-identical.
    pub(crate) mtp_seed_failures: u32,
    /// Parsed generation_config.json sampling/stop defaults.
    pub(crate) gen_defaults: crate::engine::ModelGenerationDefaults,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache) covering
    /// the six GQA attention layers. Only `full_attention` layers route
    /// through the adapter (indexed by attention-layer ordinal 0..6); mamba
    /// and moe layers keep their own per-request state / stateless forward.
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    /// Registered before Metal alloc (configured pools at `load_inner`, adaptive
    /// pools in `size_paged_pool_after_weight_load`) so a concurrent load cannot
    /// reuse the same headroom. Taken by `load_with_thread`.
    pub(crate) pool_cache_limit_guard: Option<crate::cache_limit::PoolCacheLimitGuard>,
    /// Per-request recurrent state for the scheduler lane: one full
    /// `Vec<NemotronHLayerCache>` per live sequence, swapped into `caches` for serial
    /// prefill and stacked into `[N, ...]` rows for batched decode.
    pub(crate) scheduled_caches: HashMap<SeqId, Vec<NemotronHLayerCache>>,
    /// The sequence whose per-request caches currently sit in `caches`.
    pub(crate) active_scheduled_seq: Option<SeqId>,
    /// Whether the most recent `prime_prefix_state_for` decided the live mamba states
    /// already reflected the prompt's cached prefix, so the turn only prefilled the
    /// suffix. Read by `paged_perf_prefill_tokens` for the telemetry numerator.
    pub(crate) last_paged_prefill_reused_mamba_state: bool,
    /// Whether the currently active scheduled sequence's recurrent (Mamba)
    /// state was RESTORED from parked caches (true) or freshly
    /// zero-initialized (false — e.g. after preemption released it).
    pub(crate) active_seq_recurrent_survived: bool,
    /// Quantized kernels dispatch differently for M=1 vs M>=2, and the 23 mamba layers
    /// amplify those few ULP into token flips by step ~8. When set, the batched lane
    /// runs the quantized projections per row (M=1, bit-identical to the scalar path).
    pub(crate) row_exact_decode_projections: bool,
}

impl NemotronHInner {
    /// Create a new inner with empty (uninitialized) weights.
    pub(crate) fn new(config: NemotronHConfig) -> Result<Self> {
        let num_layers = config.num_hidden_layers as usize;
        let hidden_size = config.hidden_size as u32;

        let embedding = Embedding::new(config.vocab_size as u32, hidden_size)?;
        let final_norm = RMSNorm::new(hidden_size, Some(config.layer_norm_epsilon))?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(NemotronHDecoderLayer::new(&config, i)?);
        }

        let caches = fresh_caches(&config, &layers)?;
        let paged_adapter = build_paged_adapter(&config)?;

        let mtp = if config.n_mtp_layers > 0 {
            Some(NemotronHMtpModule::new(&config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            turn_cancel: None,
            embedding,
            layers,
            final_norm,
            lm_head: None,
            caches,
            tokenizer: None,
            cached_token_history: Vec::new(),
            mtp,
            mtp_weights_loaded: false,
            pending_mtp_draft_seed: None,
            flat_mtp_caches_desynced: false,
            flat_mtp_last_rollback_unemitted: 0,
            mtp_seed_failures: 0,
            gen_defaults: crate::engine::ModelGenerationDefaults::default(),
            paged_adapter,
            pool_cache_limit_guard: None,
            scheduled_caches: HashMap::new(),
            active_scheduled_seq: None,
            last_paged_prefill_reused_mamba_state: false,
            active_seq_recurrent_survived: false,
            row_exact_decode_projections: false,
        })
    }
}

/// Build the per-layer flat caches for a fresh request (all mamba states
/// zero-initialized, attention K/V empty, moe stateless).
fn fresh_caches(
    config: &NemotronHConfig,
    layers: &[NemotronHDecoderLayer],
) -> Result<Vec<NemotronHLayerCache>> {
    let num_layers = config.num_hidden_layers as usize;
    let mut caches = Vec::with_capacity(num_layers);
    for (i, layer) in layers.iter().enumerate() {
        caches.push(if config.is_mamba_layer(i) {
            let m = match &layer.mixer {
                NemotronHMixer::Mamba(m) => m,
                _ => unreachable!("kind mismatch"),
            };
            NemotronHLayerCache::new_mamba(m.fresh_state(1)?)
        } else if config.is_attention_layer(i) {
            NemotronHLayerCache::new_attention()
        } else {
            NemotronHLayerCache::new_moe()
        });
    }
    Ok(caches)
}

struct ConfiguredPagedPoolPlan {
    pa_config: mlx_paged_attn::PagedAttentionConfig,
    num_blocks: u32,
    block_size: u32,
    cache_dtype: mlx_paged_attn::metal::MetalDtype,
    pool_bytes: u64,
}

/// Shared gates + byte math for a configured (`paged_cache_memory_mb`) pool.
/// Uncapped configs return `None` so `size_paged_pool_after_weight_load` can
/// apply `load_time_pool_sizing`.
fn configured_paged_pool_plan(config: &NemotronHConfig) -> Result<Option<ConfiguredPagedPoolPlan>> {
    if config.use_block_paged_cache == Some(false)
        || !crate::engine::persistence::compiled_forward_backend_available()
    {
        return Ok(None);
    }
    let attn_layer_count = config.attention_layer_idxs().len() as u32;
    if attn_layer_count == 0 {
        return Ok(None);
    }
    let block_size = config.paged_block_size.unwrap_or(16);
    if ![8, 16, 32].contains(&block_size) {
        return Err(Error::from_reason(format!(
            "NemotronH block-paged adapter: invalid paged_block_size {block_size} (must be 8, 16, or 32)"
        )));
    }
    let Some(gpu_memory_mb) = config.paged_cache_memory_mb else {
        return Ok(None);
    };
    let pa_config = mlx_paged_attn::PagedAttentionConfig {
        block_size,
        gpu_memory_mb,
        head_size: config.head_dim as u32,
        num_kv_heads: config.num_key_value_heads as u32,
        num_layers: attn_layer_count,
        use_fp8_cache: Some(false),
        max_seq_len: Some(config.max_position_embeddings as u32),
        max_batch_size: Some(32),
    };
    let num_blocks = pa_config.calculate_num_blocks();
    if num_blocks == 0 {
        return Err(Error::from_reason(format!(
            "NemotronH block-paged adapter: gpu_memory_mb={gpu_memory_mb} too small \
             (head_size={}, num_kv_heads={}, block_size={block_size}, num_attn_layers={attn_layer_count})",
            config.head_dim, config.num_key_value_heads
        )));
    }
    let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
    let pool_bytes = mlx_paged_attn::profile::bytes_per_block(
        attn_layer_count,
        config.num_key_value_heads as u32,
        config.head_dim as u32,
        block_size,
        cache_dtype,
    )
    .map_err(|error| {
        Error::from_reason(format!(
            "NemotronH configured paged pool byte accounting failed: {error}"
        ))
    })?
    .saturating_mul(u64::from(num_blocks));
    Ok(Some(ConfiguredPagedPoolPlan {
        pa_config,
        num_blocks,
        block_size,
        cache_dtype,
        pool_bytes,
    }))
}

/// Register a configured pool with the coordinator before `NemotronHInner::new`
/// allocates it, so a concurrent uncapped Nemotron load cannot treat those
/// private Metal bytes as free headroom.
pub(crate) fn reserve_configured_paged_pool(
    config: &NemotronHConfig,
) -> Result<Option<crate::cache_limit::PoolCacheLimitGuard>> {
    let Some(plan) = configured_paged_pool_plan(config)? else {
        return Ok(None);
    };
    Ok(Some(
        crate::cache_limit::coordinator().register_pool(plan.pool_bytes),
    ))
}

/// Construct the block-paged KV adapter covering only the GQA attention
/// layers (pool indexed by attention-layer ordinal). Gated on
/// `use_block_paged_cache` (default-on when None) and Metal availability;
/// returns `None` when either gate is closed.
fn build_paged_adapter(config: &NemotronHConfig) -> Result<Option<PagedKVCacheAdapter>> {
    let Some(plan) = configured_paged_pool_plan(config)? else {
        return Ok(None);
    };
    let allocator = Arc::new(Mutex::new(mlx_paged_attn::BlockAllocator::new(
        plan.num_blocks,
        plan.num_blocks,
        plan.block_size,
    )));
    let pool = mlx_paged_attn::LayerKVPool::new(
        plan.pa_config,
        plan.num_blocks,
        plan.num_blocks,
        plan.cache_dtype,
    )
    .map_err(|e| {
        Error::from_reason(format!(
            "Failed to construct LayerKVPool for NemotronH block-paged adapter: {e}"
        ))
    })?;
    let adapter =
        PagedKVCacheAdapter::new(allocator, Arc::new(pool), plan.block_size).map_err(|e| {
            Error::from_reason(format!(
                "Failed to construct NemotronH PagedKVCacheAdapter: {e}"
            ))
        })?;
    Ok(Some(adapter))
}

/// Whether the flat MTP core must drop its final generated token when saving the
/// session history, so it ends exactly where the physical caches end.
///
/// `finish_reason` alone is ambiguous: the engine leaves `last_in_cache` at `true`
/// on EVERY length exit. `rollback_unemitted` separates them — 0 is a token sampled
/// but never forwarded (drop), nonzero is a mid-cycle break where verify already
/// wrote that token's K/V (keep).
fn mtp_history_drop_last(
    last_in_cache: bool,
    finish_reason: &str,
    rollback_unemitted: usize,
) -> bool {
    !last_in_cache || (finish_reason == "length" && rollback_unemitted == 0)
}

/// Whether the live mamba states already sit at the `cached_prefix_len` boundary.
/// The Mamba-2 recurrent state is NON-INVERTIBLE, so only a strict byte-for-byte
/// extension of the saved history may reuse a cached K/V prefix; every other hit
/// must drive `cached_prefix_len` to 0 (there is no reconstruction path, and
/// `run_paged_prefill_chunk` hard-errors on the mismatch).
fn mamba_state_reusable(
    plan: &[u32],
    cached_token_history: &[u32],
    cached_prefix_len: usize,
) -> bool {
    cached_prefix_len > 0
        && cached_prefix_len == cached_token_history.len()
        && plan.len() >= cached_prefix_len
        && plan[..cached_prefix_len] == cached_token_history[..]
}

impl NemotronHInner {
    /// Install sampling + stop-token defaults from generation_config.json.
    pub(crate) fn set_gen_defaults(&mut self, defaults: crate::engine::ModelGenerationDefaults) {
        self.gen_defaults = defaults;
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Whether a complete MTP head was loaded.
    pub(crate) fn has_mtp_weights(&self) -> bool {
        self.mtp.is_some() && self.mtp_weights_loaded
    }

    /// Full forward over input_ids [1, T]: returns [1, T, vocab] logits.
    pub(crate) fn forward(&mut self, input_ids: &MxArray) -> Result<MxArray> {
        let mut h = self.embedding.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, Some(&mut self.caches[i]))?;
        }
        h = self.final_norm.forward(&h)?;
        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embedding.as_linear(&h)
        }
    }

    /// Forward with hidden: returns (logits [1, T, vocab], post-final-norm
    /// hidden [1, T, hidden]). Used by the MTP stepper (Step A + verify).
    pub(crate) fn forward_with_hidden(
        &mut self,
        input_ids: &MxArray,
        embedding: &Embedding,
    ) -> Result<(MxArray, MxArray)> {
        self.forward_with_hidden_3d(input_ids, embedding)
    }

    /// Raw forward with hidden, hidden kept as [1, T, hidden] (3D). Used by
    /// the MTP verify step, whose MtpVerifyOutput contract requires the
    /// engine's verify_hiddens[:, K, :] slice.
    pub(crate) fn forward_with_hidden_3d(
        &mut self,
        input_ids: &MxArray,
        embedding: &Embedding,
    ) -> Result<(MxArray, MxArray)> {
        let mut h = embedding.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, Some(&mut self.caches[i]))?;
        }
        let hidden = self.final_norm.forward(&h)?;
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden)?
        } else {
            embedding.as_linear(&hidden)?
        };
        Ok((logits, hidden))
    }

    /// Reset all caches and cached token history.
    pub(crate) fn reset_caches_internal(&mut self) {
        self.caches = fresh_caches(&self.config, &self.layers).expect("fresh caches rebuild");
        self.cached_token_history.clear();
        self.flat_mtp_caches_desynced = false;
        // The seed describes a backbone history that just went away.
        self.pending_mtp_draft_seed = None;
    }

    /// Eval every live cache array (post-prefill sync).
    fn eval_caches_internal(&self) -> Result<()> {
        let mut refs = Vec::new();
        for c in self.caches.iter() {
            c.collect_arrays(&mut refs);
        }
        MxArray::eval_arrays(&refs)
    }

    /// Save the session history, aligning it with the physical cache length.
    /// `drop_last` is the CALLER's decision, never re-derived here: AR/paged callers
    /// pass `true` unconditionally, the MTP caller passes [`mtp_history_drop_last`].
    fn save_cache_state_internal(
        &mut self,
        reuse_cache: bool,
        tokens: &[u32],
        generated_tokens: &[u32],
        drop_last: bool,
    ) {
        if reuse_cache {
            let mut full_history = tokens.to_vec();
            let history_tokens = if drop_last && !generated_tokens.is_empty() {
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

    /// Chunked prefill: process the prompt in chunks, evaluating caches
    /// after each chunk to bound peak GPU memory. Returns the full logits
    /// [1, T, vocab] of the FINAL chunk (the caller slices last-token).
    pub(crate) fn chunked_prefill(
        &mut self,
        prompt: &MxArray,
        generation_stream: Stream,
    ) -> Result<MxArray> {
        let total_len = prompt.shape_at(1)?;
        // Chunk-aligned slices: the Mamba-2 chunk scan pads the LAST chunk of each
        // forward, so every intermediate boundary must land on the configured chunk
        // grid or the recurrence's reduction grouping changes.
        let slices = chunk_aligned_prefill_slices(
            0,
            total_len as u32,
            PREFILL_STEP_SIZE as u32,
            self.config.chunk_size as u32,
        );
        let last_idx = slices.len().saturating_sub(1);
        let mut last = None;
        for (idx, (s, e)) in slices.into_iter().enumerate() {
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason(PREFILL_CANCELLED));
            }
            let chunk = prompt.slice_axis(1, s as i64, e as i64)?;
            {
                let _stream_ctx = StreamContext::new(generation_stream);
                last = Some(self.forward(&chunk)?);
            }
            if idx != last_idx {
                self.eval_caches_internal()?;
                crate::array::clear_cache();
            }
        }
        last.ok_or_else(|| Error::from_reason("chunked_prefill produced no chunks"))
    }

    /// Chunked prefill that ALSO seeds the MTP drafter's own KV cache, feeding it the
    /// vLLM EAGLE-shifted pairs `drafter slot p <- (h_p, emb(tokens[p + 1]))` for
    /// `p < T-1`. Slot `T-1` is NOT written here: its embedding is the first sampled
    /// token `y` — see [`seed_mtp_final_slot`](Self::seed_mtp_final_slot). Returns the
    /// final chunk's logits plus `h_{T-1}`.
    pub(crate) fn chunked_prefill_seeding_mtp(
        &mut self,
        prompt: &MxArray,
        generation_stream: Stream,
        mtp_caches: &mut [NemotronHLayerCache],
        committed_len: &mut i32,
    ) -> Result<(MxArray, MxArray)> {
        self.chunked_prefill_seeding_mtp_stepped(
            prompt,
            generation_stream,
            PREFILL_STEP_SIZE as u32,
            mtp_caches,
            committed_len,
        )
    }

    /// [`chunked_prefill_seeding_mtp`](Self::chunked_prefill_seeding_mtp) with an
    /// explicit chunk step, so a test can force the multi-chunk path. The seed must
    /// be invariant to where the chunk boundaries fall.
    pub(crate) fn chunked_prefill_seeding_mtp_stepped(
        &mut self,
        prompt: &MxArray,
        generation_stream: Stream,
        step_size: u32,
        mtp_caches: &mut [NemotronHLayerCache],
        committed_len: &mut i32,
    ) -> Result<(MxArray, MxArray)> {
        let total_len = prompt.shape_at(1)?;
        if total_len < 1 {
            return Err(Error::from_reason(
                "chunked_prefill_seeding_mtp: empty prompt",
            ));
        }
        let slices = chunk_aligned_prefill_slices(
            0,
            total_len as u32,
            step_size,
            self.config.chunk_size as u32,
        );
        let last_idx = slices.len().saturating_sub(1);
        let embedding = self.embedding.clone();
        let mut out: Option<(MxArray, MxArray)> = None;
        for (idx, (s, e)) in slices.into_iter().enumerate() {
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason(PREFILL_CANCELLED));
            }
            let chunk = prompt.slice_axis(1, s as i64, e as i64)?;
            let (logits, hidden) = {
                let _stream_ctx = StreamContext::new(generation_stream);
                self.forward_with_hidden_3d(&chunk, &embedding)?
            };
            // Rows [s, s + L') pair with ids [s+1, s+1+L'). The clamp drops
            // the very last prompt row (p == T-1): its partner id is `y`.
            let rows = (e - s) as i64;
            let l_prime = rows.min((total_len - 1) - s as i64);
            if l_prime > 0 {
                let _stream_ctx = StreamContext::new(generation_stream);
                let hidden_slice = hidden.slice_axis(1, 0, l_prime)?;
                let ids = prompt.slice_axis(1, s as i64 + 1, s as i64 + 1 + l_prime)?;
                let emb_seq = embedding.forward(&ids)?;
                let mtp = self.mtp.as_ref().ok_or_else(|| {
                    Error::from_reason(
                        "chunked_prefill_seeding_mtp: inner.mtp is None despite the \
                         has_mtp_weights() gate",
                    )
                })?;
                mtp.forward(&hidden_slice, &emb_seq, mtp_caches)?;
                *committed_len += l_prime as i32;
            }
            if idx == last_idx {
                let h_last = hidden.slice_axis(1, rows - 1, rows)?;
                out = Some((logits, h_last));
            } else {
                self.eval_caches_internal()?;
                // The drafter cache is lazy too: without this eval the next
                // `clear_cache()` strands its graph and peak memory grows.
                let mut refs = Vec::new();
                for c in mtp_caches.iter() {
                    c.collect_arrays(&mut refs);
                }
                MxArray::eval_arrays(&refs)?;
                crate::array::clear_cache();
            }
        }
        out.ok_or_else(|| Error::from_reason("chunked_prefill_seeding_mtp produced no chunks"))
    }

    /// Write the FINAL prompt slot of the drafter cache: `(h_{T-1}, emb(y))`, where
    /// `y` is the token sampled from the prefill logits. Completes the seed to
    /// `committed_len == T`.
    pub(crate) fn seed_mtp_final_slot(
        &mut self,
        h_last: &MxArray,
        y_id: u32,
        mtp_caches: &mut [NemotronHLayerCache],
        committed_len: &mut i32,
    ) -> Result<()> {
        let embedding = self.embedding.clone();
        let ids = MxArray::from_uint32(&[y_id], &[1, 1])?;
        let emb = embedding.forward(&ids)?;
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            Error::from_reason(
                "seed_mtp_final_slot: inner.mtp is None despite the has_mtp_weights() gate",
            )
        })?;
        mtp.forward(h_last, &emb, mtp_caches)?;
        *committed_len += 1;
        Ok(())
    }
}

impl NemotronHInner {
    /// f32 bytes of one request's mamba recurrent state (conv + SSM per mamba
    /// layer), used by the scheduler's unified-memory watermark.
    pub(crate) fn recurrent_state_bytes_per_seq(&self) -> u64 {
        let mut bytes = 0u64;
        for i in 0..self.layers.len() {
            if self.config.is_mamba_layer(i)
                && let Some(state) = self.caches[i].as_mamba_state()
            {
                bytes = bytes
                    .saturating_add(state.conv.nbytes() as u64)
                    .saturating_add(state.ssm.nbytes() as u64);
            }
        }
        bytes
    }

    pub(crate) fn scheduled_recurrent_bytes(&self) -> u64 {
        let rows =
            self.scheduled_caches.len() as u64 + u64::from(self.active_scheduled_seq.is_some());
        rows.saturating_mul(self.recurrent_state_bytes_per_seq())
    }

    pub(crate) fn has_scheduled_caches_for(&self, seq_id: SeqId) -> bool {
        self.active_scheduled_seq == Some(seq_id) || self.scheduled_caches.contains_key(&seq_id)
    }

    /// Whether `activate_paged_seq(seq_id)` will land on state that SURVIVED (as
    /// opposed to fresh zero-state caches). Read BEFORE the activation, while
    /// `active_seq_recurrent_survived` still describes the PREVIOUS sequence.
    pub(crate) fn scheduled_recurrent_state_live(&self, seq_id: SeqId) -> bool {
        if self.active_scheduled_seq == Some(seq_id) {
            self.active_seq_recurrent_survived
        } else {
            self.scheduled_caches.contains_key(&seq_id)
        }
    }

    /// Live recurrent rows resident in `caches` + `scheduled_caches`.
    /// `activate_paged_seq` removes from the map before setting the active
    /// slot, so no sequence is counted twice.
    pub(crate) fn scheduled_recurrent_units(&self) -> usize {
        self.scheduled_caches.len() + usize::from(self.active_scheduled_seq.is_some())
    }

    /// Physical vs trained context limits (qwen3_5/qwen3_5_moe contract):
    /// (trained window, effective window capped by the paged pool's token
    /// capacity, block capacity, block size). Without an adapter the flat
    /// cache owns no block pool and the trained window is the limit.
    pub(crate) fn paged_context_limits(&self) -> (u32, u32, u32, u32) {
        let trained = self.config.max_position_embeddings.max(0) as u32;
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return (trained, trained, 0, 0);
        };
        let blocks = adapter.block_capacity();
        let block_size = adapter.block_size();
        let bytes_per_block = adapter.bytes_per_block().unwrap_or(0);
        let usable = pool_tokens_after_recurrent(
            adapter.max_capacity_tokens(),
            block_size,
            bytes_per_block,
            self.recurrent_state_bytes_per_seq(),
        );
        (
            trained,
            scheduled_turn_context(trained, usable, scheduler_per_seq_context_override()),
            blocks,
            block_size,
        )
    }

    /// After weights are resident, allocate the uncapped default pool
    /// (one trained-length sequence) clipped by live Metal headroom.
    pub(crate) fn size_paged_pool_after_weight_load(&mut self) -> Result<()> {
        if self.paged_adapter.is_some() {
            return Ok(());
        }
        if self.config.paged_cache_memory_mb.is_some() {
            return Ok(());
        }
        if self.config.use_block_paged_cache == Some(false)
            || !crate::engine::persistence::compiled_forward_backend_available()
        {
            return Ok(());
        }
        let attn_layer_count = self.config.attention_layer_idxs().len() as u32;
        if attn_layer_count == 0 {
            return Ok(());
        }
        let block_size = self.config.paged_block_size.unwrap_or(16);
        if ![8, 16, 32].contains(&block_size) {
            return Err(Error::from_reason(format!(
                "NemotronH block-paged adapter: invalid paged_block_size {block_size} (must be 8, 16, or 32)"
            )));
        }
        let head_size = self.config.head_dim as u32;
        let num_kv_heads = self.config.num_key_value_heads as u32;
        let max_seq_len = self.config.max_position_embeddings.max(0) as u32;
        let requested_memory_mb =
            mlx_paged_attn::PagedAttentionConfig::memory_mb_for_one_full_sequence(
                max_seq_len,
                block_size,
                head_size,
                num_kv_heads,
                attn_layer_count,
            );
        let pa_config = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            gpu_memory_mb: requested_memory_mb,
            head_size,
            num_kv_heads,
            num_layers: attn_layer_count,
            use_fp8_cache: Some(false),
            max_seq_len: Some(max_seq_len),
            max_batch_size: Some(32),
        };
        let requested_blocks = pa_config.calculate_num_blocks();
        if requested_blocks == 0 {
            return Err(Error::from_reason(format!(
                "NemotronH block-paged adapter: requested paged cache {requested_memory_mb} MiB \
                 cannot hold one block"
            )));
        }
        let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
        let coordinator = crate::cache_limit::coordinator();
        let (selected_blocks, pool_guard) = {
            let mut reserved = None;
            for _ in 0..8 {
                let sibling = coordinator.registered_pool_bytes();
                let sizing = mlx_paged_attn::profile::load_time_pool_sizing_with_reserved(
                    requested_blocks,
                    attn_layer_count,
                    num_kv_heads,
                    head_size,
                    block_size,
                    cache_dtype,
                    sibling,
                )
                .map_err(|error| {
                    Error::from_reason(format!(
                        "NemotronH adaptive paged cache sizing failed safely; refusing an uncapped \
                         pool request: {error}"
                    ))
                })?;
                if let Some(guard) =
                    coordinator.try_register_pool_if_total_eq(sibling, sizing.selected_bytes)
                {
                    reserved = Some((sizing.selected_blocks, guard));
                    break;
                }
            }
            reserved.ok_or_else(|| {
                Error::from_reason(
                    "NemotronH adaptive paged cache sizing failed safely; concurrent pool \
                     reservation kept racing",
                )
            })?
        };
        let allocator = Arc::new(Mutex::new(mlx_paged_attn::BlockAllocator::new(
            selected_blocks,
            selected_blocks,
            block_size,
        )));
        let pool = mlx_paged_attn::LayerKVPool::new(
            pa_config,
            selected_blocks,
            selected_blocks,
            cache_dtype,
        )
        .map_err(|e| {
            Error::from_reason(format!(
                "Failed to construct LayerKVPool for NemotronH block-paged adapter: {e}"
            ))
        })?;
        self.paged_adapter = Some(
            PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                Error::from_reason(format!(
                    "Failed to construct NemotronH PagedKVCacheAdapter: {e}"
                ))
            })?,
        );
        self.pool_cache_limit_guard = Some(pool_guard);
        Ok(())
    }

    /// Activate the adapter request and swap the sequence's per-request caches
    /// into `caches`, parking the previously active sequence (lfm2 ShortConv
    /// pattern).
    pub(crate) fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        self.paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("nemotron_h paged adapter is unavailable"))?
            .activate_request(seq_id)
            .map_err(Error::from_reason)?;
        if self.active_scheduled_seq == Some(seq_id) {
            // Already live: LEAVE THE FLAG ALONE. The scheduler activates a
            // sequence and then activates it AGAIN through `prime_prefix_state_for`;
            // re-asserting `true` here would overwrite a preempted sequence's honest
            // `had_state = false` before that function reads it, resuming with mamba
            // state at zero against a full cached KV prefix.
            return Ok(());
        }
        // Capture BEFORE the remove: a preempted sequence's recurrent state was
        // released, so the remove falls back to FRESH zero-state caches and the reuse
        // predicate must know the state did not survive.
        let had_state = self.scheduled_caches.contains_key(&seq_id);
        self.park_active_scheduled_caches();
        self.caches = self
            .scheduled_caches
            .remove(&seq_id)
            .unwrap_or_else(|| fresh_caches(&self.config, &self.layers).expect("fresh caches"));
        self.active_scheduled_seq = Some(seq_id);
        self.active_seq_recurrent_survived = had_state;
        Ok(())
    }

    pub(crate) fn park_active_scheduled_caches(&mut self) {
        let Some(seq_id) = self.active_scheduled_seq.take() else {
            return;
        };
        let replacement = fresh_caches(&self.config, &self.layers).expect("fresh caches");
        let caches = std::mem::replace(&mut self.caches, replacement);
        self.scheduled_caches.insert(seq_id, caches);
    }

    fn reset_scheduled_caches_for(&mut self, seq_id: SeqId) {
        let fresh = || fresh_caches(&self.config, &self.layers).expect("fresh caches");
        if self.active_scheduled_seq == Some(seq_id) {
            self.caches = fresh();
        } else {
            self.scheduled_caches.insert(seq_id, fresh());
        }
    }

    pub(crate) fn release_scheduled_caches_for(&mut self, seq_id: SeqId) {
        if self.active_scheduled_seq == Some(seq_id) {
            self.active_scheduled_seq = None;
            self.caches = fresh_caches(&self.config, &self.layers).expect("fresh caches");
        }
        self.scheduled_caches.remove(&seq_id);
    }

    /// Attention-layer ordinal for the adapter's LayerKVPool (0..6).
    fn attention_ordinal(&self, layer_idx: usize) -> u32 {
        self.config
            .attention_layer_idxs()
            .iter()
            .position(|&a| a == layer_idx)
            .unwrap_or(0) as u32
    }

    /// Stack one mamba layer's per-request states into one batched state.
    fn stacked_mamba_state(&mut self, seq_ids: &[SeqId], layer_idx: usize) -> Result<Mamba2State> {
        self.park_active_scheduled_caches();
        let mut rows = Vec::with_capacity(seq_ids.len());
        for &seq_id in seq_ids {
            let caches = self.scheduled_caches.get(&seq_id).ok_or_else(|| {
                Error::from_reason(format!(
                    "NemotronH sequence {seq_id} has no scheduled caches"
                ))
            })?;
            let state = caches
                .get(layer_idx)
                .and_then(|c| c.as_mamba_state())
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "NemotronH sequence {seq_id} has no mamba state for layer {layer_idx}"
                    ))
                })?;
            rows.push(state);
        }
        Mamba2State::stack_rows(&rows)
    }

    /// Scatter a batched mamba state back into per-request slots.
    fn scatter_mamba_state(
        &mut self,
        seq_ids: &[SeqId],
        layer_idx: usize,
        state: &Mamba2State,
    ) -> Result<()> {
        for (row, &seq_id) in seq_ids.iter().enumerate() {
            let row_state = state.row(row as i64)?;
            let caches = self.scheduled_caches.get_mut(&seq_id).ok_or_else(|| {
                Error::from_reason(format!(
                    "NemotronH sequence {seq_id} disappeared during mamba scatter"
                ))
            })?;
            let slot = caches
                .get_mut(layer_idx)
                .and_then(|c| c.as_mamba_state_mut())
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "NemotronH layer {layer_idx} has no mamba-state slot"
                    ))
                })?;
            *slot = row_state;
        }
        Ok(())
    }

    /// One paged prefill slice: record the suffix in the adapter and forward it
    /// through all layers. Returns the last-token logits `[vocab]`. A nonzero
    /// `cached_prefix_len` is legal only when the live mamba state already sits on
    /// that boundary, which `skip_reconstruction` records.
    fn run_paged_prefill_chunk(
        &mut self,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
        skip_reconstruction: bool,
    ) -> Result<MxArray> {
        if suffix_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_prefill_chunk called with empty suffix",
            ));
        }
        let suffix_len = suffix_tokens.len() as u32;

        // Assertion, not a fallback. `prime_prefix_state_for` guarantees that a nonzero
        // `cached_prefix_len` always comes with a live mamba state at that boundary;
        // reaching here means that guarantee broke, and there is no reconstruction path
        // to fall back to.
        if cached_prefix_len > 0 && !skip_reconstruction {
            let planned = full_tokens.len();
            return Err(Error::from_reason(format!(
                "nemotron_h: paged prefill reached a {cached_prefix_len}-token cached prefix \
                 without a live mamba state (plan {planned} tokens); prime_prefix_state_for \
                 must force a cold prefill"
            )));
        }

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

        let input_ids = MxArray::from_uint32(suffix_tokens, &[1, suffix_len as i64])?;
        let mut hidden = self.embedding.forward(&input_ids)?;
        let num_layers = self.layers.len();
        let first_logical_position = cached_prefix_len;
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            let layer: &NemotronHDecoderLayer = unsafe { &*self.layers.as_ptr().add(layer_idx) };
            let attn_ordinal = self.attention_ordinal(layer_idx);
            let normed = layer.norm.forward(&hidden)?;
            let out = match &layer.mixer {
                NemotronHMixer::Mamba(m) => {
                    let cache: &mut NemotronHLayerCache =
                        unsafe { &mut *self.caches.as_mut_ptr().add(layer_idx) };
                    let state = cache.as_mamba_state_mut().ok_or_else(|| {
                        Error::from_reason("run_paged_prefill_chunk: mamba cache missing")
                    })?;
                    m.forward(&normed, Some(state))?
                }
                NemotronHMixer::Attention(a) => {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason("run_paged_prefill_chunk: paged_adapter dropped")
                    })?;
                    a.forward_paged(
                        &normed,
                        adapter,
                        attn_ordinal,
                        first_logical_position,
                        cached_prefix_len,
                        /* is_prefill */ true,
                    )?
                }
                NemotronHMixer::MoE(m) => m.forward(&normed)?,
            };
            hidden = hidden.add(&out)?;
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden)?;
        }

        hidden = self.final_norm.forward(&hidden)?;
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden)?
        } else {
            self.embedding.as_linear(&hidden)?
        };
        let seq_len = logits.shape_at(1)?;
        logits
            .slice_axis(1, seq_len - 1, seq_len)?
            .squeeze(Some(&[0, 1]))
    }

    /// Run one paged decode step (single request, exclusive/whole-turn lane).
    ///
    /// Records the token (reserving blocks, which is the only fallible
    /// allocation in a decode step) and then runs the forward.
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
        self.run_paged_decode_forward(token_id, first_logical_position)
    }

    /// Forward half of one paged decode step, for a token whose blocks are ALREADY
    /// reserved and whose cursor has ALREADY advanced; `first_logical_position` is the
    /// cursor captured BEFORE the record. The split lets the row-exact wave reserve
    /// every row's blocks (the only failable step) before any row's non-invertible
    /// mamba state is touched.
    fn run_paged_decode_forward(
        &mut self,
        token_id: u32,
        first_logical_position: u32,
    ) -> Result<MxArray> {
        let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
        let mut hidden = self.embedding.forward(&input_ids)?;
        let num_layers = self.layers.len();
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            let layer: &NemotronHDecoderLayer = unsafe { &*self.layers.as_ptr().add(layer_idx) };
            let attn_ordinal = self.attention_ordinal(layer_idx);
            let normed = layer.norm.forward(&hidden)?;
            let out = match &layer.mixer {
                NemotronHMixer::Mamba(m) => {
                    let cache: &mut NemotronHLayerCache =
                        unsafe { &mut *self.caches.as_mut_ptr().add(layer_idx) };
                    let state = cache.as_mamba_state_mut().ok_or_else(|| {
                        Error::from_reason("run_paged_decode_step: mamba cache missing")
                    })?;
                    m.forward(&normed, Some(state))?
                }
                NemotronHMixer::Attention(a) => {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason("run_paged_decode_step: paged_adapter dropped")
                    })?;
                    a.forward_paged(
                        &normed,
                        adapter,
                        attn_ordinal,
                        first_logical_position,
                        /* cached_prefix_len */ 0,
                        /* is_prefill */ false,
                    )?
                }
                NemotronHMixer::MoE(m) => m.forward(&normed)?,
            };
            hidden = hidden.add(&out)?;
        }

        hidden = self.final_norm.forward(&hidden)?;
        if let Some(ref head) = self.lm_head {
            head.forward(&hidden)
        } else {
            self.embedding.as_linear(&hidden)
        }
    }

    /// Roll back one recorded decode token per sequence, newest first, so an allocator
    /// squeeze leaves the wave where it started. Blocks the successful records
    /// allocated stay owned by their request — the retry writes into them.
    fn unwind_recorded_decode_rows(&mut self, recorded: &[SeqId]) -> Result<()> {
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
        Ok(())
    }

    /// Row-exact batched decode: N single-row decodes stacked into `[N, 1, vocab]`,
    /// bit-identical to N scalar decodes, committed ALL-OR-NOTHING.
    ///
    /// Ordering is the whole point: phase 1 records every row (the only step that
    /// reserves blocks, so the only one an allocator squeeze can fail) and unwinds its
    /// peers on failure, and only then does phase 2 fold each token into its
    /// non-invertible mamba state. Row-by-row instead lets the scheduler's
    /// blocked-wave retry re-feed a token a survivor already folded in.
    fn run_row_exact_decode_wave(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        // Pre-pass BEFORE any mutation: reject a duplicated sequence (two rows of one
        // wave would record two tokens against one cursor) and resolve every row's
        // write position while the cursors are still untouched.
        let mut seen = HashSet::with_capacity(rows.len());
        let mut positions = Vec::with_capacity(rows.len());
        {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step_batched: paged adapter is unavailable")
            })?;
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
                positions.push(position);
            }
        }

        // Phase 1: reserve every row's blocks, all or nothing.
        let mut recorded: Vec<SeqId> = Vec::with_capacity(rows.len());
        for &(seq_id, token_id) in rows {
            if let Err(error) = self
                .paged_adapter
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason("run_paged_decode_step_batched: paged adapter disappeared")
                })?
                .record_token_for(seq_id, token_id)
            {
                self.unwind_recorded_decode_rows(&recorded)?;
                return Err(Error::from_reason(format!(
                    "run_paged_decode_step_batched failed to record sequence {seq_id}: {error}"
                )));
            }
            recorded.push(seq_id);
        }

        // Phase 2: per-row scalar forwards. `activate_paged_seq` re-points the adapter
        // at each row and swaps in that row's per-request caches.
        let mut logits = Vec::with_capacity(rows.len());
        for (index, &(seq_id, token_id)) in rows.iter().enumerate() {
            self.activate_paged_seq(seq_id)?;
            logits.push(self.run_paged_decode_forward(token_id, positions[index])?);
        }
        MxArray::concatenate_many(logits.iter().collect(), Some(0))
    }

    /// Run one uniform paged decode step for multiple requests: mamba states
    /// stacked to `[N, ...]` rows around each SSM layer, attention through the
    /// batched paged kernels, MoE routed over the shared `[N,1,H]` tensor.
    /// Returns `[N, 1, vocab]` logits.
    fn run_paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        if rows.is_empty() {
            return Err(Error::from_reason(
                "run_paged_decode_step_batched requires at least one row",
            ));
        }
        // Preserve the scalar decode graph for a one-row wave (quantized kernels can
        // round differently when a singleton goes through a batched graph).
        if let [(seq_id, token_id)] = rows {
            self.activate_paged_seq(*seq_id)?;
            return self.run_paged_decode_step(*token_id);
        }
        // Quantized checkpoints: the batched [N,1] projections AND the batched
        // paged-attention kernels round differently from the single-row decode, and the
        // 23 mamba layers amplify a few ULP into token flips. Dense keeps the fused path.
        if self.row_exact_decode_projections {
            return self.run_row_exact_decode_wave(rows);
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
            if let Err(error) = self
                .paged_adapter
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason("run_paged_decode_step_batched: paged adapter disappeared")
                })?
                .record_token_for(seq_id, token_id)
            {
                self.unwind_recorded_decode_rows(&recorded)?;
                return Err(Error::from_reason(format!(
                    "run_paged_decode_step_batched failed to record sequence {seq_id}: {error}"
                )));
            }
            recorded.push(seq_id);
        }

        let token_ids = rows.iter().map(|&(_, token)| token).collect::<Vec<_>>();
        let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();
        let input_ids = MxArray::from_uint32(&token_ids, &[rows.len() as i64, 1])?;
        let mut hidden = self.embedding.forward(&input_ids)?;
        for layer_idx in 0..self.layers.len() {
            let layer: &NemotronHDecoderLayer = unsafe { &*self.layers.as_ptr().add(layer_idx) };
            let attn_ordinal = self.attention_ordinal(layer_idx);
            let mut stacked = if self.config.is_mamba_layer(layer_idx) {
                Some(self.stacked_mamba_state(&seq_ids, layer_idx)?)
            } else {
                None
            };
            let normed = layer.norm.forward(&hidden)?;
            let out = match &layer.mixer {
                NemotronHMixer::Mamba(m) => {
                    let state = stacked.as_mut().ok_or_else(|| {
                        Error::from_reason("NemotronH batched decode: mamba layer missing state")
                    })?;
                    m.forward(&normed, Some(state))?
                }
                NemotronHMixer::Attention(a) => {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason("NemotronH batched decode: paged adapter dropped")
                    })?;
                    a.forward_paged_batched(&normed, adapter, attn_ordinal, &planned_rows)?
                }
                NemotronHMixer::MoE(m) => m.forward(&normed)?,
            };
            hidden = hidden.add(&out)?;
            if let Some(state) = stacked.as_ref() {
                self.scatter_mamba_state(&seq_ids, layer_idx, state)?;
            }
        }

        hidden = self.final_norm.forward(&hidden)?;
        if let Some(ref head) = self.lm_head {
            head.forward(&hidden)
        } else {
            self.embedding.as_linear(&hidden)
        }
    }
}
/// Per-turn decode stepper for the engine's generic flat AR flow.
pub(crate) struct NemotronHDecode<'a> {
    inner: &'a mut NemotronHInner,
}

impl DecodeStep for NemotronHDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let logits = self.inner.forward(input_ids)?;
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token, logits]);
    }
}

impl ChatBackend for NemotronHInner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "nemotron_h"
    }

    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.turn_cancel = flag;
    }

    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32> {
        tok.im_end_id()
            .or_else(|| self.config.eos_token_ids.first().copied().map(|v| v as u32))
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))
    }

    fn generation_defaults(&self) -> Option<&crate::engine::ModelGenerationDefaults> {
        Some(&self.gen_defaults)
    }

    fn extra_eos_ids(&self) -> Vec<u32> {
        // UNION of the checkpoint config's EOS set and `generation_config.json` (the
        // engine contract: `eos_token_ids` is a union, never an override). The released
        // checkpoint adds a generation-config EOS absent from config.json; without this
        // every chat path would decode past it.
        let mut ids: Vec<u32> = self
            .config
            .eos_token_ids
            .iter()
            .map(|&v| v as u32)
            .collect();
        ids.extend(self.gen_defaults.eos_token_ids.iter().copied());
        ids
    }

    /// True when the physical flat caches hold tokens the saved history does NOT, so
    /// cache recovery must reset + re-prefill. NOT "a mid-cycle MTP stop happened":
    /// `rollback_unemitted` latches only when a token the backbone actually FORWARDED
    /// is dropped, and a cycle's LAST outcome token (bonus or residual) never goes
    /// through the layers, so a depth-1 mid-cycle stop must report `false`.
    fn flat_caches_desynced(&self) -> bool {
        self.flat_mtp_caches_desynced
    }

    fn clear_flat_caches_desynced(&mut self) {
        self.flat_mtp_caches_desynced = false;
    }

    fn policy(&self) -> ThinkingPolicy {
        ThinkingPolicy::TemplateHonoring
    }

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        // Shared clear for BOTH scopes: wipe flat caches + token history +
        // parked scheduled state.
        self.reset_caches_internal();
        self.scheduled_caches.clear();
        self.active_scheduled_seq = None;
        // The EXPLICIT command reset must restore a fully cold state: release the live
        // request AND purge the allocator prefix cache, so a reset-then-rerun of the
        // same prompt replays the cold prefill (a partial-prefix hit has a different
        // bf16 reduction order).
        if scope == ResetScope::Command
            && let Some(adapter) = self.paged_adapter.as_mut()
        {
            adapter
                .release_request_and_purge_prefix_cache()
                .map_err(|e| {
                    Error::from_reason(format!(
                        "nemotron_h reset_caches: paged prefix-cache purge failed: {e}"
                    ))
                })?;
        }
        Ok(())
    }

    /// All-or-nothing: the Mamba-2 recurrent state is non-invertible, so
    /// only an exact prefix append is reusable; exact-match is a miss.
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        if !reuse_cache {
            return 0;
        }
        let cached = &self.cached_token_history;
        if !cached.is_empty() && tokens.len() > cached.len() && tokens[..cached.len()] == cached[..]
        {
            cached.len()
        } else {
            0
        }
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        self.save_cache_state_internal(
            args.reuse_cache,
            args.save_tokens,
            args.generated_tokens,
            /* drop_last */ true,
        );
    }

    fn eval_caches(&self) -> Result<()> {
        self.eval_caches_internal()
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        let token_arr: Vec<u32> = prompt_tokens.to_vec();
        let prompt = MxArray::from_uint32(&token_arr, &[1, prompt_tokens.len() as i64])?;
        let logits = self.chunked_prefill(&prompt, stream)?;
        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        last_logits.squeeze(Some(&[1]))
    }

    type Decode<'a>
        = NemotronHDecode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        Ok(NemotronHDecode { inner: self })
    }

    fn execution_plan(&self) -> ExecutionPlan {
        ExecutionPlan {
            media: MediaPlan::NONE,
            paged_attention: self.paged_adapter.as_ref().map(|_| PagedAttentionPlan {
                supports_delta: true,
            }),
            speculative: self.has_mtp_weights().then_some(SpeculativePlan {
                kind: SpeculativeKind::NativeMtp,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::NONE,
                // Native MTP is flat-cache only here: the draft head reads the
                // FLAT KV, so an admitted MTP turn takes the flat lane with its
                // target and the paged adapter serves the autoregressive turns.
                supports_paged_attention: false,
                // `run_mtp_whole_turn` only ever returns `TurnOutput::Complete`.
                supports_streaming: false,
            }),
        }
    }

    fn wired_limit_bytes(&self) -> Option<usize> {
        Some(self.config.estimate_memory_bytes() as usize)
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        debug_assert!(matches!(args.plan.decoder, DecoderPlan::Autoregressive));
        crate::engine::paged_turn::run_paged_turn(self, args)
    }

    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        debug_assert!(
            args.sink.is_none(),
            "the flat MTP core has no streaming arm"
        );
        self.run_mtp_whole_turn(args)
    }
}

impl NemotronHInner {
    /// Async-eval every live cache array.
    fn async_eval_caches(&self) {
        let mut refs = Vec::new();
        for c in self.caches.iter() {
            c.collect_arrays(&mut refs);
        }
        MxArray::async_eval_arrays(&refs);
    }
}

/// Paged decode stepper for the exclusive/whole-turn lane, driving the
/// engine-owned run_paged_turn decode loop through PagedBackend.
pub(crate) struct NemotronHPagedDecode<'a> {
    inner: &'a mut NemotronHInner,
}

impl DecodeStep for NemotronHPagedDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // NOT on the hot path - the engine drives decode via
        // forward_with_token (which hands the scalar the loop already read).
        let token_id = input_ids.item_at_int32(0)? as u32;
        self.forward_with_token(input_ids, token_id)
    }

    fn forward_with_token(
        &mut self,
        _input_ids: &MxArray,
        token_id: u32,
    ) -> Result<(MxArray, bool)> {
        let logits = self
            .inner
            .run_paged_decode_step(token_id)?
            .squeeze(Some(&[1]))?;
        // run_paged_decode_step returns [1, 1, vocab]; the squeeze above
        // reduces it to [1, vocab], so needs_squeeze = false.
        Ok((logits, false))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        next_token.eval();
    }

    fn maintain_cache(&mut self, step: i32) {
        crate::array::maybe_clear_cache_for_paged_step(step);
    }
}

/// NemotronH paged prefix state: the effective prefix/suffix split the adapter
/// resolved, PLUS the full prompt tokens (which the engine never hands to
/// `paged_prefill`). The full tokens are used only to report the planned length
/// in `run_paged_prefill_chunk`'s fail-closed error.
pub(crate) struct NemotronHPrefixState {
    pub(crate) effective_cached_prefix_len: usize,
    pub(crate) suffix_len: usize,
    pub(crate) full_tokens: Vec<u32>,
    pub(crate) mamba_state_reusable: bool,
}

impl PagedPrefix for NemotronHPrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

impl PagedBackend for NemotronHInner {
    type PagedDecode<'a>
        = NemotronHPagedDecode<'a>
    where
        Self: 'a;
    type PrefixState = NemotronHPrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        _reuse_cache: bool,
        _block_size: usize,
        extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        debug_assert!(extra_keys.is_empty(), "nemotron_h is text-only");
        let owner_history = self.cached_token_history.clone();
        // The Mamba-2 recurrent state is non-invertible: a cached K/V prefix is
        // continuable only when `plan` strictly extends the previous turn's history AND
        // this sequence's recurrent row is still live (preemption, abort and
        // finalize-without-reuse all release it). Reconstruction is not bit-identical to
        // the cold prefill and the residual flips greedy argmaxes mid-turn.
        //
        // FAST PATH, not enforcement: `prime_prefix_state_for`'s backstop covers the same
        // condition, but this keeps the common preempted-resume off that backstop's
        // reallocating re-prepare, which can fail on a near-full pool.
        let continuation_eligible = !owner_history.is_empty()
            && plan.len() > owner_history.len()
            && plan[..owner_history.len()] == owner_history[..]
            && self.scheduled_recurrent_state_live(0);
        self.prime_prefix_state_for(0, plan, &owner_history, cache_salt, !continuation_eligible)
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        stream: Stream,
    ) -> Result<MxArray> {
        if suffix_tokens.is_empty() {
            return Err(Error::from_reason(
                "nemotron_h paged_prefill called with an empty suffix",
            ));
        }
        let cached = prefix.effective_cached_prefix_len as u32;
        let end = cached + suffix_tokens.len() as u32;
        let slices = chunk_aligned_prefill_slices(
            cached,
            end,
            PREFILL_STEP_SIZE as u32,
            self.config.chunk_size as u32,
        );
        let mut last = None;
        for (s, e) in slices {
            // Cancellation point: the engine calls `paged_prefill` ONCE with the
            // whole suffix on the single-session lane, so without this a long cold
            // prefill stalls the model thread for every peer session.
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason(PREFILL_CANCELLED));
            }
            let skip = prefix.mamba_state_reusable || s > cached;
            let local = (s - cached) as usize;
            let local_end = (e - cached) as usize;
            let _stream_ctx = StreamContext::new(stream);
            last = Some(self.run_paged_prefill_chunk(
                &prefix.full_tokens,
                &suffix_tokens[local..local_end],
                s,
                skip,
            )?);
        }
        last.ok_or_else(|| Error::from_reason("nemotron_h paged_prefill produced no logits"))
    }

    fn begin_paged_decode(&mut self) -> Result<Self::PagedDecode<'_>> {
        Ok(NemotronHPagedDecode { inner: self })
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool, cache_salt: u64) {
        if let Some(adapter) = self.paged_adapter.as_mut() {
            if reuse_cache {
                let _ = adapter.finalize_turn_keep_live(&[], cache_salt);
            } else {
                let _ = adapter.register_full_blocks_for_reuse(&[], cache_salt);
            }
        }
        if !reuse_cache {
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            self.release_scheduled_caches_for(0);
        }
    }

    fn abort_paged_turn(&mut self) {
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
        self.release_scheduled_caches_for(self.active_scheduled_seq.unwrap_or(0));
        self.cached_token_history.clear();
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        _keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        // NemotronH paged ALWAYS drops the last token: the decode loop never forwards
        // the final sampled token, so it is in neither the adapter nor the mamba states.
        self.save_cache_state_internal(
            reuse_cache,
            save_tokens,
            generated,
            /* drop_last */ true,
        );
        Ok(())
    }

    fn paged_perf_prefill_tokens(&self, prompt_token_count: usize, suffix_len: usize) -> usize {
        // A foreign/partial prefix hit is forced to a COLD prefill of the whole plan, so
        // ttft measures full-prompt work; the carried-state path prefills the suffix.
        if self.last_paged_prefill_reused_mamba_state {
            suffix_len
        } else {
            prompt_token_count
        }
    }
}

/// Engine-owned scheduler state for the NemotronH model thread.
pub(crate) type NemotronHSchedulerState = HybridSchedulerState<NemotronHInner>;

impl HybridSchedulerBackend for NemotronHInner {
    type Command = NemotronHCmd;
    type RestoreTicket = NoRestoreTicket;
    type OwnerState = Vec<u32>;
    type StepExecutor<'a> = HybridStepExecutor<'a, Self>;

    const SCHEDULER_NAME: &'static str = "NemotronH";

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

    /// Residency cap on parked recurrent rows. Without this override the trait default
    /// (`true`) makes `ensure_recurrent_slot`'s idle-victim body dead code and
    /// `scheduled_caches` grows one row per owner forever. The cap matches
    /// `max_concurrent_sequences()`, not `HYBRID_LIVE_STATE_UNITS = 2`, which would
    /// cut batching from 8 rows to 2. Deliberately not qwen3_5's hard error inside
    /// `activate_scheduled_recurrent`: `finish_completed` reaches the same function
    /// with no `ensure_recurrent_slot` gate, so erroring there would fail live turns.
    fn can_activate_scheduled_recurrent(&self, seq_id: SeqId) -> bool {
        self.has_scheduled_recurrent(seq_id)
            || self.scheduled_recurrent_units() < scheduler_max_num_seqs_for(32)
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
        NemotronHPrefixState {
            effective_cached_prefix_len,
            suffix_len,
            full_tokens,
            mamba_state_reusable: !first_chunk || base.mamba_state_reusable,
        }
    }

    fn prepare_scheduled_prefix(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        owner_history: &[u32],
        _reuse_cache: bool,
        cache_salt: u64,
        _block_size: u32,
    ) -> Result<ScheduledPrefixAdmission<Self::PrefixState, Self::RestoreTicket>> {
        // Same cold-prefill rule as the single-session lane (see `prime_prefix_state`),
        // plus the fact that makes it correct here: a PREEMPTED sequence keeps its owner
        // history and K/V blocks but loses its recurrent row (as can an idle row
        // `ensure_recurrent_slot` evicts), and `activate_scheduled_recurrent` already ran
        // for this seq_id, so the liveness answer is honest.
        let continuation_eligible = !owner_history.is_empty()
            && tokens.len() > owner_history.len()
            && tokens[..owner_history.len()] == owner_history[..]
            && self.scheduled_recurrent_state_live(seq_id);
        self.prime_prefix_state_for(
            seq_id,
            tokens,
            owner_history,
            cache_salt,
            !continuation_eligible,
        )
        .map(ScheduledPrefixAdmission::Ready)
    }

    fn run_scheduled_prefill_slice(
        &mut self,
        seq_id: SeqId,
        source: &[u32],
        base: &Self::PrefixState,
        start: usize,
        end: usize,
        generation_stream: Stream,
        first_chunk: bool,
    ) -> Result<Option<MxArray>> {
        self.activate_paged_seq(seq_id)?;
        // The engine's pinned break-set is a 2048-token grid; re-split every slice at
        // the CONFIGURED Mamba-2 chunk boundaries so no executed prefill forward splits
        // a chunk (a hard-coded 128 would misalign other checkpoints).
        let slices = chunk_aligned_prefill_slices(
            start as u32,
            end as u32,
            PREFILL_STEP_SIZE as u32,
            self.config.chunk_size as u32,
        );
        let mut last = None;
        for (index, (s, e)) in slices.into_iter().enumerate() {
            let prefix = self.build_scheduled_prefix(
                base,
                s as usize,
                (e - s) as usize,
                source.to_vec(),
                first_chunk && index == 0,
            );
            last = Some(self.paged_prefill(
                &source[s as usize..e as usize],
                &prefix,
                generation_stream,
            )?);
        }
        Ok(last)
    }

    fn profiler_prefill_tokens(&self, prefix: &Self::PrefixState, prompt_tokens: u32) -> u32 {
        if prefix.mamba_state_reusable {
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
        _owners: crate::engine::hybrid_scheduler::SchedulerOwnerContext<'_, Self::OwnerState>,
    ) {
        handle_nemotron_h_cmd(self, command);
    }
}

impl NemotronHInner {
    /// Shared prefix-priming for both lanes. Activates the sequence, resolves
    /// the adapter's effective prefix/suffix split, and decides whether the
    /// parked mamba states already sit at the prefix boundary.
    fn prime_prefix_state_for(
        &mut self,
        seq_id: SeqId,
        plan: &[u32],
        owner_history: &[u32],
        cache_salt: u64,
        skip_lookup: bool,
    ) -> Result<NemotronHPrefixState> {
        self.activate_paged_seq(seq_id)?;
        let total_budget = plan.len() as u32;
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let mut turn_plan = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| {
                Error::from_reason("nemotron_h prime_prefix_state: paged adapter is None")
            })?
            .prepare_turn_with_max_cache_hit_tokens(
                seq_id,
                plan,
                total_budget,
                true,
                &[],
                cache_salt,
                skip_lookup,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason)?;
        let mut cached_prefix_len = turn_plan.cached_prefix_len as usize;
        // Reuse requires BOTH a token-exact prefix match AND the sequence's recurrent
        // (Mamba) state surviving at that boundary: a preempted sequence releases its
        // state while its history and KV blocks stay reusable.
        let mut reused_state = self.active_seq_recurrent_survived
            && mamba_state_reusable(plan, owner_history, cached_prefix_len);
        // INVARIANT (enforced, not assumed): a kept K/V prefix ALWAYS has a live mamba
        // state at exactly its boundary. `skip_lookup` closes the hash-lookup route, but
        // `ContinuedLivePrefix` and `ContinueFailedReset` ignore it entirely. Drop the
        // prefix and re-prepare cold: same FLOPs, exact numerics.
        if !reused_state && cached_prefix_len > 0 {
            tracing::debug!(
                target: "mlx_core::nemotron_h",
                seq_id,
                cached_prefix_len,
                reason = ?turn_plan.reason,
                "nemotron_h: K/V prefix has no live mamba state; re-preparing cold"
            );
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("nemotron_h prime_prefix_state: paged adapter is None")
            })?;
            adapter.release_request().map_err(Error::from_reason)?;
            turn_plan = adapter
                .prepare_turn_with_max_cache_hit_tokens(
                    seq_id,
                    plan,
                    total_budget,
                    true,
                    &[],
                    cache_salt,
                    /* skip_lookup */ true,
                    max_cache_hit_tokens,
                )
                .map_err(Error::from_reason)?;
            cached_prefix_len = turn_plan.cached_prefix_len as usize;
            debug_assert_eq!(
                cached_prefix_len, 0,
                "skip_lookup must force a 0-token cold prefill"
            );
            reused_state = false;
        }
        self.last_paged_prefill_reused_mamba_state = reused_state;
        // From here this turn's prefill brings the live recurrent state to the plan
        // boundary either way, so the NEXT activation of this still-live sequence may
        // treat it as survived. A turn that fails after this point does not leak the
        // claim: every failure route releases the sequence.
        self.active_seq_recurrent_survived = true;
        // A turn that is NOT a live continuation of the currently live request (fresh
        // owner, or a same-owner continue that had to reset) cannot inherit the previous
        // request's token history: the decode loop applies penalties over it and labels
        // positions from it, so a stale cross-owner history shifts both.
        if turn_plan.reason != PagedTurnPlanReason::ContinuedLivePrefix {
            self.cached_token_history.clear();
        }
        if !reused_state {
            self.reset_scheduled_caches_for(seq_id);
        }
        Ok(NemotronHPrefixState {
            effective_cached_prefix_len: cached_prefix_len,
            suffix_len: turn_plan.suffix_len as usize,
            full_tokens: plan.to_vec(),
            mamba_state_reusable: reused_state,
        })
    }
}

/// Split [start, end) into sub-slices of at most `slice_tokens` tokens whose
/// internal boundaries are all multiples of the Mamba-2 `chunk_size`, so no
/// executed prefill forward splits a chunk relative to the chunk-scan arithmetic.
/// The first sub-slice starts at `start`, which may be unaligned.
pub(crate) fn chunk_aligned_prefill_slices(
    start: u32,
    end: u32,
    slice_tokens: u32,
    chunk_size: u32,
) -> Vec<(u32, u32)> {
    let mut slices = Vec::new();
    let mut s = start;
    if !s.is_multiple_of(chunk_size) {
        let a = s.div_ceil(chunk_size).saturating_mul(chunk_size);
        if a < end {
            slices.push((s, a));
            s = a;
        }
    }
    while s < end {
        let candidate = (s + slice_tokens).min(end);
        let e = if candidate < end {
            // NONTERMINAL boundary: round DOWN to a chunk multiple. (`slice_tokens`
            // need not divide the chunk size: 2048 = 10*192 + 128 would otherwise
            // end mid-chunk.)
            let snapped = (candidate / chunk_size) * chunk_size;
            if snapped > s {
                snapped
            } else {
                // slice_tokens smaller than one chunk: advance a full chunk.
                (s + chunk_size).min(end)
            }
        } else {
            // FINAL boundary: reach the range end exactly.
            end
        };
        slices.push((s, e));
        s = e;
    }
    slices
}

/// Flat MTP propose/verify stepper for the engine-owned run_mtp_turn loop.
///
/// The drafter is STATEFUL: the MTP head is a real decoder layer with its own K/V,
/// so this stepper owns the head's per-layer caches for the whole turn, seeded over
/// the prompt via [`NemotronHInner::pending_mtp_draft_seed`]. Two rewinds,
/// deliberately: the BACKBONE rolls back from the pre-verify snapshot (`snap`); the
/// DRAFTER never uses one, rewinding by cursor `trim`.
pub(crate) struct NemotronHMtpStepper<'a> {
    inner: &'a mut NemotronHInner,
    embedding: Embedding,
    /// The MTP head's OWN per-layer caches, seeded over the prompt.
    mtp_caches: Vec<NemotronHLayerCache>,
    /// Number of drafter slots holding committed (not drafted) pairs.
    /// Starts at the prompt length; `commit_mtp` advances it.
    committed_len: i32,
    /// Pre-verify snapshot of every layer cache.
    snap: Option<Vec<NemotronHLayerSnapshot>>,
    /// Stashed rollback/replay error (surfaced by restore_and_replay_main).
    replay_err: Option<Error>,
    /// Flat desync latch set by a mid-cycle stop.
    mtp_desynced: bool,
}

impl NemotronHMtpStepper<'_> {
    /// Rewind every drafter KV to `target` slots (cursor-only; the trimmed
    /// K/V stay allocated and are overwritten by the next write).
    fn trim_draft_caches(&mut self, target: i32) {
        for c in self.mtp_caches.iter_mut() {
            if let Some(kv) = c.as_kv_cache_mut() {
                kv.trim(target);
            }
        }
    }

    /// The BACKBONE attention side's ground-truth frontier: the first
    /// attention layer's offset, in absolute consumed tokens. The flat twin of
    /// the qwen3_5 / MoE `attention_frontier` — this family's native MTP is
    /// flat-cache only (`execution_plan` publishes
    /// `supports_paged_attention: false`), so there is no paged branch to pick.
    fn attention_frontier(&self) -> Option<u64> {
        self.inner
            .caches
            .iter()
            .find_map(|c| c.attention_offset())
            .map(|offset| offset.max(0) as u64)
    }

    /// The drafter attention slot's live offset. Compiled in every debug build so
    /// `begin_cycle`'s cursor-alignment `debug_assert!` can read it.
    #[cfg(any(test, debug_assertions))]
    pub(crate) fn draft_kv_offset(&self) -> i32 {
        self.mtp_caches
            .iter()
            .find_map(|c| c.as_kv_cache())
            .map(|kv| kv.get_offset())
            .unwrap_or(-1)
    }

    /// Test seam: the drafter's committed length. Stays `#[cfg(test)]` —
    /// `begin_cycle`'s debug assertion reads the FIELD, not this accessor.
    #[cfg(test)]
    pub(crate) fn committed_len(&self) -> i32 {
        self.committed_len
    }

    /// Bound the drafter's lazy graph alongside the backbone caches: the head writes
    /// K/V every draft AND every commit.
    fn async_eval_draft_caches(&self) {
        let mut refs = Vec::new();
        for c in self.mtp_caches.iter() {
            c.collect_arrays(&mut refs);
        }
        if !refs.is_empty() {
            MxArray::async_eval_arrays(&refs);
        }
    }
}

impl MtpStepper for NemotronHMtpStepper<'_> {
    fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    fn committed_history_active(&self) -> bool {
        // The drafter carries a persistent, prompt-seeded committed history. Left at
        // `false`, every chained cycle would re-commit its anchor and drift the drafter
        // cursor by +1 per cycle.
        true
    }

    fn forward_with_hidden(
        &mut self,
        ids: &MxArray,
        embedding: &Embedding,
    ) -> Result<(MxArray, MxArray, bool)> {
        // Step A contract: the engine seeds the next cycle with `hidden.shape_at(1)` as
        // the HIDDEN size, so the returned hidden must be [1, hidden]. Verify uses
        // forward_with_hidden_3d directly and keeps [1, T, hidden].
        let (logits, hidden) = self.inner.forward_with_hidden_3d(ids, embedding)?;
        let hidden = hidden.squeeze(Some(&[1]))?;
        Ok((logits, hidden, true))
    }

    fn draft_step(
        &mut self,
        prev_hidden: &MxArray,
        prev_emb: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let mtp = self.inner.mtp.as_ref().ok_or_else(|| {
            Error::from_reason(
                "NemotronH MTP draft_step: inner.mtp is None despite has_mtp_weights() gate",
            )
        })?;
        // The draft writes its own K/V at the slot `begin_cycle` just trimmed to, and
        // reads the head's OWN causal history — never the backbone's KV.
        let h_next = mtp.forward(prev_hidden, prev_emb, &mut self.mtp_caches)?;
        let head = self
            .inner
            .lm_head
            .as_ref()
            .ok_or_else(|| Error::from_reason("NemotronH MTP draft_step: lm_head missing"))?;
        let dl3 = head.forward(&h_next)?;
        let draft_logits = dl3.squeeze(Some(&[1]))?;
        Ok((h_next, draft_logits))
    }

    fn verify_step(
        &mut self,
        ids: &MxArray,
        embedding: &Embedding,
        depth: usize,
    ) -> Result<crate::models::qwen3_5::mtp_decode::MtpVerifyOutput> {
        // Fail closed on the depth-1 invariant. `run_mtp_whole_turn` pins
        // `mtp_adaptive_depth = false` and seeds `p.mtp_depth.min(1)`, and
        // `run_mtp_cycle` only ever SHORTENS a cycle's depth, so exactly 1 arrives here;
        // anything else means a new engine route bypassed that clamp.
        if depth != 1 {
            return Err(Error::from_reason(format!(
                "NemotronH MTP verify_step: cycle depth {depth} but this family is \
                 pinned to depth 1 (run_mtp_whole_turn clamps mtp_adaptive_depth \
                 off and seeds mtp_depth.min(1)); validate the deeper draft path \
                 and re-check NemotronHMtpStepper::rollback_unemitted before \
                 relaxing this"
            )));
        }
        let id_window = ids.to_int32().map_err(|e| {
            Error::from_reason(format!(
                "NemotronH MTP verify_step: ids to_int32: {}",
                e.reason
            ))
        })?;
        if id_window.len() < depth + 1 {
            return Err(Error::from_reason(format!(
                "NemotronH MTP verify_step: ids has {} elements, need {}",
                id_window.len(),
                depth + 1
            )));
        }
        let id_slice: Vec<i32> = id_window.iter().take(depth + 1).copied().collect();
        // Verify one token at a time through the AR DECODE path. A batched [1, depth+1]
        // forward routes every stateful mamba layer through the chunk-scan, whose
        // padded-chunk arithmetic differs from the recurrent decode_step in f32 rounding;
        // the drift flips near-tie argmaxes and breaks the T=0 lossless contract.
        let mut logits_rows: Vec<MxArray> = Vec::with_capacity(depth + 1);
        let mut hidden_rows: Vec<MxArray> = Vec::with_capacity(depth + 1);
        for &tok in &id_slice {
            let one = MxArray::from_int32(&[tok], &[1, 1])?;
            let (logits, hidden) = self.inner.forward_with_hidden_3d(&one, embedding)?;
            logits_rows.push(logits);
            hidden_rows.push(hidden);
        }
        let logits = MxArray::concatenate_many(logits_rows.iter().collect::<Vec<_>>(), Some(1))?;
        let hiddens = MxArray::concatenate_many(hidden_rows.iter().collect::<Vec<_>>(), Some(1))?;
        // The engine slices verify_hiddens[:, K, :], so hiddens must stay
        // [1, depth+1, hidden] (each row is the raw 3D per-token hidden).
        Ok(crate::models::qwen3_5::mtp_decode::MtpVerifyOutput::logits_only(logits, hiddens))
    }

    fn snapshot_main_linear(&mut self) {
        let snap = self
            .inner
            .caches
            .iter()
            .map(|c| c.snapshot())
            .collect::<Vec<_>>();
        self.snap = Some(snap);
    }

    fn rollback(&mut self, accepted_drafts: usize, depth: usize) {
        if self.replay_err.is_some() {
            return;
        }
        // Full accept: verify already left the main caches correctly
        // advanced (all depth+1 tokens committed) - discard the snapshot.
        if accepted_drafts == depth {
            self.snap = None;
            return;
        }
        let result: Result<()> = (|| {
            let snap = self
                .snap
                .as_ref()
                .ok_or_else(|| Error::from_reason("NemotronH MTP rollback: snapshot missing"))?;
            for (cache, snap) in self.inner.caches.iter_mut().zip(snap.iter()) {
                cache.restore(snap.clone()).map_err(Error::from_reason)?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.replay_err = Some(e);
        }
    }

    fn restore_and_replay_main(&mut self, replay_ids: &[u32], embedding: &Embedding) -> Result<()> {
        if let Some(e) = self.replay_err.take() {
            return Err(e);
        }
        if replay_ids.is_empty() {
            return Err(Error::from_reason(
                "NemotronH MTP restore_and_replay_main: empty replay prefix",
            ));
        }
        let arr = MxArray::from_uint32(replay_ids, &[1, replay_ids.len() as i64])?;
        let _ = self.inner.forward_with_hidden(&arr, embedding)?;
        Ok(())
    }

    /// Append the cycle's TRULY COMMITTED pairs to the drafter's history. Slot `p`
    /// holds `(h_p, emb(t_{p+1}))`, so the hidden run starts one position BEFORE the
    /// committed ids: `seed_hidden ++ verify_hiddens[..M-1]` for `IncludeAnchor`,
    /// `verify_hiddens[..M]` for `SkipAlreadyCommittedAnchor`. The `trim` first is
    /// load-bearing: this cycle's drafts wrote speculative K/V past `committed_len`,
    /// and a rejected draft's K/V must be overwritten.
    fn commit_mtp(
        &mut self,
        anchor: crate::models::qwen3_5::mtp_decode::MtpCommitAnchor,
        seed_h: &MxArray,
        verify_hiddens: &MxArray,
        committed_ids: &[u32],
        _k_accepted: usize,
        embedding: &Embedding,
    ) -> Result<()> {
        use crate::models::qwen3_5::mtp_decode::MtpCommitAnchor;
        let m = committed_ids.len();
        if m == 0 {
            return Ok(());
        }
        let hidden_dim = verify_hiddens.shape_at(2)?;
        let hidden_seq = match anchor {
            MtpCommitAnchor::IncludeAnchor => {
                let vh_prefix =
                    verify_hiddens.slice(&[0, 0, 0], &[1, (m - 1) as i64, hidden_dim])?;
                MxArray::concatenate(seed_h, &vh_prefix, 1)?
            }
            MtpCommitAnchor::SkipAlreadyCommittedAnchor => {
                verify_hiddens.slice(&[0, 0, 0], &[1, m as i64, hidden_dim])?
            }
        };
        let ids_i32: Vec<i32> = committed_ids.iter().map(|&v| v as i32).collect();
        let ids_arr = MxArray::from_int32(&ids_i32, &[m as i64])?;
        let gathered = embedding.forward(&ids_arr)?;
        let emb_seq = gathered.reshape(&[1, m as i64, hidden_dim])?;

        self.trim_draft_caches(self.committed_len);
        let mtp = self.inner.mtp.as_ref().ok_or_else(|| {
            Error::from_reason(
                "NemotronH MTP commit_mtp: inner.mtp is None despite has_mtp_weights() gate",
            )
        })?;
        mtp.forward(&hidden_seq, &emb_seq, &mut self.mtp_caches)?;
        self.committed_len += m as i32;
        Ok(())
    }

    /// Re-anchor the drafter cache before this cycle's draft steps: PERSISTENT, so
    /// this truncates the previous cycle's draft tail. A chained cycle's draft pair
    /// targets the slot the previous commit already wrote, so it anchors at
    /// `committed_len - 1`; Step-A cycles anchor at `committed_len`.
    fn begin_cycle(&mut self, chained_anchor: bool) {
        // The drafter must never enter a cycle BELOW the committed cursor:
        // `KVCache::trim` cannot GROW, so a cycle that anchored at `committed_len - 1`
        // and bailed without drafting would leave the cursor low for good and the next
        // draft would overwrite the last COMMITTED pair. `>=` not `==`: an all-rejected
        // cycle legitimately sits ABOVE `committed_len`.
        //
        // `#[cfg(debug_assertions)]` on a BLOCK, not a bare `debug_assert!`: the latter
        // still TYPE-CHECKS its body in release, where `draft_kv_offset` does not exist.
        #[cfg(debug_assertions)]
        {
            let cursor = self.draft_kv_offset();
            debug_assert!(
                cursor >= self.committed_len,
                "drafter cursor {} fell below committed_len {} at cycle entry",
                cursor,
                self.committed_len
            );
        }
        let target = if chained_anchor {
            (self.committed_len - 1).max(0)
        } else {
            self.committed_len
        };
        self.trim_draft_caches(target);
    }

    fn eval_step(&self, token: &MxArray, logits: &MxArray, budget_forced: bool) {
        self.inner.async_eval_caches();
        self.async_eval_draft_caches();
        token.eval();
        if budget_forced {
            logits.eval();
        }
    }

    fn eval_step_with_chained_hidden(&self, token: &MxArray, chained_hidden: &MxArray) {
        self.inner.async_eval_caches();
        self.async_eval_draft_caches();
        MxArray::async_eval_arrays(&[token, chained_hidden]);
    }

    fn rollback_unemitted(&mut self, unemitted: usize) {
        if unemitted == 0 {
            return;
        }
        // Latch ONLY when a FORWARDED token is dropped; `unemitted - 1` is that count at
        // every depth. A cycle's LAST outcome token is never fed through the backbone
        // (the bonus is argmax of an existing verify row; the residual is sampled after
        // `rollback` restored the snapshot), so `unemitted == 1` — every depth-1
        // mid-cycle stop — leaves the caches aligned with the saved history and must NOT
        // latch. The `> 1` arm is unreachable while depth is pinned to 1, and is kept
        // because it is the correct predicate at every depth.
        if unemitted > 1 {
            self.mtp_desynced = true;
        }
        // Cursor rewind ONLY, deliberately snapshot-free: on the one strandable depth-1
        // path (`accepted_drafts == depth`) `rollback` has already nulled `self.snap`,
        // so a snapshot-based undo here would silently no-op.
        self.committed_len = (self.committed_len - unemitted as i32).max(0);
        let target = self.committed_len;
        self.trim_draft_caches(target);
    }

    fn frontier(&self) -> Option<SpecFrontier> {
        // Hybrid stack, but the recurrent count is NOT nameable here, so it is
        // reported as unknown rather than fabricated: `Mamba2State` is a conv
        // tensor plus an ssm tensor with no consumed-token counter, and this
        // family rewinds the recurrent side by whole-snapshot RESTORE
        // (`rollback` restores every layer, Mamba and attention alike, from one
        // `snapshot_main_linear`) instead of replaying a step count, so there is
        // no bookkeeping to publish. Echoing `attn_tokens` back would turn the
        // I4 tripwire into a comparison of a value with itself.
        //
        // The tripwire is inert for the other reason too: `rollback_unemitted`
        // rewinds only the DRAFTER caches and never touches `inner.caches`, so
        // this family heals a mid-cycle stop through the `mtp_desynced` latch,
        // not by moving one side of the backbone.
        Some(SpecFrontier {
            attn_tokens: self.attention_frontier()?,
            recurrent_tokens: None,
        })
    }

    fn take_replay_error(&mut self) -> Option<Error> {
        self.replay_err.take()
    }

    fn into_desynced(self) -> bool {
        self.mtp_desynced
    }
}

impl NemotronHInner {
    /// Plain autoregressive whole-turn core over the FLAT caches, sync and streaming.
    /// Reached only from `mtp_seed_failed_fallback` on a model with no paged adapter:
    /// the engine's generic AR flow is unreachable from inside a specialized handler.
    /// Structure mirrors that generic flow one for one so the two cannot drift.
    fn run_flat_ar_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        if args.tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        let tokenizer = args.tokenizer.clone();
        let tokens = args.tokens.to_vec();
        let is_delta = args.plan.is_delta;
        let is_streaming = args.sink.is_some();
        let mut p = args.params.clone();
        p.extra_eos_ids = ChatBackend::extra_eos_ids(self);
        let eos_id = args.eos_id;
        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let max_new_tokens = p.max_new_tokens;
        let report_perf = p.report_performance;
        let generation_start = report_perf.then(std::time::Instant::now);
        let mut first_token_instant: Option<std::time::Instant> = None;

        // A prior mid-cycle MTP stop left the flat trunk ahead of the saved history; the
        // mamba state cannot rewind, so heal by re-prefilling.
        let desynced = self.flat_mtp_caches_desynced;
        let hit = if desynced {
            0
        } else {
            ChatBackend::verify_cache_prefix(self, &tokens, p.reuse_cache)
        };
        let (prefill_tokens, cached_prefix_len) = if hit > 0 && hit < tokens.len() {
            (tokens[hit..].to_vec(), hit)
        } else {
            ChatBackend::reset_caches(self, ResetScope::PrefixMiss)?;
            (tokens.clone(), 0)
        };

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = ChatBackend::wired_limit_bytes(self)
            .map(|bytes| crate::stream::WiredLimitContext::new(bytes, vec![generation_stream]));

        let mut profiler = crate::decode_profiler::DecodeProfiler::new(
            ChatBackend::profiler_label(self, is_delta, is_streaming),
            ChatBackend::family_name(self),
        );
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        let mut token_history = tokens.clone();
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(
            crate::engine::params::generated_capacity_hint(max_new_tokens),
        );
        let mut finish_reason = String::from("length");
        let mut reasoning_tracker =
            crate::engine::ReasoningTracker::from_setup(&args.thinking, think_end_id);
        let extra_eos_ids = ChatBackend::extra_eos_ids(self);
        let eos_before_emit = ChatBackend::eos_before_emit(self);
        let stream_skip_special = ChatBackend::stream_skip_special_tokens(self);
        let mut decode_stream = tokenizer.inner().decode_stream(stream_skip_special);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = args.thinking.enabled;
        let mut emitter: Option<Box<dyn StreamEmitter>> =
            args.sink.map(|_| ChatBackend::stream_emitter(self));
        let turn_token_observer = ChatBackend::turn_token_observer(self);

        profiler.begin_prefill();
        let last_logits = ChatBackend::prefill(self, &prefill_tokens, generation_stream)?;
        profiler.end_prefill();
        let last_logits = crate::engine::apply_all_penalties(last_logits, &token_history, &p)?;
        let y = crate::sampling::sample(&last_logits, p.sampling_config)?;
        y.eval();
        ChatBackend::eval_caches(self)?;
        if report_perf {
            first_token_instant = Some(std::time::Instant::now());
        }

        {
            let turn_setup = TurnSetup {
                params: &p,
                is_delta,
                has_images: false,
                total_seq_len: tokens.len(),
            };
            let mut step = ChatBackend::begin_decode(self, &turn_setup)?;
            let streaming_ctx = match (args.sink, args.cancelled, emitter.as_mut()) {
                (Some(callback), Some(cancelled), Some(emitter)) => Some(StreamingCtx {
                    callback,
                    cancelled,
                    decode_stream: &mut decode_stream,
                    tokenizer: tokenizer.inner(),
                    streamed_text_len: &mut streamed_text_len,
                    last_is_reasoning: &mut last_is_reasoning,
                    emitter: emitter.as_mut(),
                }),
                _ => None,
            };
            run_decode_loop(
                &mut step,
                DecodeLoopArgs {
                    y,
                    params: &p,
                    reasoning_tracker: &mut reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens,
                    eos_id,
                    extra_eos_ids: &extra_eos_ids,
                    eos_before_emit,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf,
                    generation_stream,
                    cancel_flag: args.cancelled,
                    turn_token_observer,
                },
                streaming_ctx,
            )?;
            step.end_decode()?;
        }

        // `NemotronHDecode` inherits the default no-op `materialize_final`: the mamba
        // state cannot re-run a forward for the final token, so this family drops-last
        // on EVERY exit.
        ChatBackend::save_cache_state(
            self,
            SaveStateArgs {
                reuse_cache: p.reuse_cache,
                is_delta,
                has_images: false,
                generated_tokens: &generated_tokens,
                finish_reason: &finish_reason,
                save_tokens: &tokens,
                save_expanded_tokens: None,
                image_cache_key: 0,
            },
        );
        if desynced {
            ChatBackend::clear_flat_caches_desynced(self);
        }

        let performance = report_perf
            .then(|| {
                crate::engine::finalize::compute_performance_metrics(
                    generation_start,
                    first_token_instant,
                    prefill_tokens.len(),
                    generated_tokens.len(),
                )
                .map(|mut metrics| {
                    ChatBackend::augment_performance(self, &profiler, &mut metrics);
                    metrics
                })
            })
            .flatten();

        if let (Some(sink), Some(emitter)) = (args.sink, emitter.as_mut()) {
            let full_text = tokenizer
                .decode_sync(&generated_tokens, stream_skip_special)
                .unwrap_or_default();
            if full_text.len() > streamed_text_len {
                emitter.on_residual(
                    &full_text[streamed_text_len..],
                    last_is_reasoning,
                    p.include_reasoning,
                    sink,
                );
            }
        }

        let mut result = ChatBackend::finalize_turn(
            self,
            FinalizeArgs {
                tokenizer: &tokenizer,
                generated_tokens: &generated_tokens,
                finish_reason,
                think_end_id,
                think_end_str: think_end_str.as_deref(),
                performance,
                include_reasoning: p.include_reasoning,
                thinking_enabled: args.thinking.enabled,
                prompt_tokens: tokens.len() as u32,
                reasoning_tokens: reasoning_tracker.reasoning_token_count(),
            },
        )?;
        result.cached_tokens = cached_prefix_len as u32;

        if let (Some(sink), Some(emitter)) = (args.sink, emitter.as_mut()) {
            emitter.finish(&result, sink);
            Ok(TurnOutput::Streamed)
        } else {
            Ok(TurnOutput::Complete(Box::new(result)))
        }
    }

    /// FAIL-CLOSED handling for a drafter seed that could not be built. A half-seeded
    /// cache would draft from the wrong state, so the seed is dropped and THIS turn is
    /// retried on the plain AR lane.
    ///
    /// Every non-cancellation error out of `chunked_prefill_seeding_mtp` /
    /// `seed_mtp_final_slot` arrives here, transient (an MLX allocation or eval failure)
    /// and permanent (a config/module mismatch: `NemotronHMtpModule::fresh_caches` sizes
    /// the cache vector from `config.mtp_layers_block_type`, and the head's `forward`
    /// checks that length against its built layers) alike. The degrade is per-TURN so
    /// the two need no telling apart: a permanent condition simply re-fails and
    /// re-degrades on every later turn, while disarming the head would spend one
    /// transient episode on the whole loaded model, since one inner serves every session
    /// on it.
    fn mtp_seed_failed_fallback(
        &mut self,
        args: &mut WholeTurnArgs<'_>,
        err: &Error,
    ) -> Result<TurnOutput> {
        self.mtp_seed_failures = self.mtp_seed_failures.saturating_add(1);
        tracing::warn!(
            target: "mlx_core::nemotron_h",
            seed_failures = self.mtp_seed_failures,
            "NemotronH MTP drafter seed failed ({}); running this turn autoregressively \
             and retrying the head on the next turn",
            err.reason
        );
        self.pending_mtp_draft_seed = None;
        // The partial seed left the backbone caches mid-prompt; the AR lane
        // must start from a clean slate.
        self.reset_caches_internal();
        // THIS turn, and only this turn, leaves the speculative lane. The paged
        // core reads `plan.decoder` to decide whether to admit a verify cycle, and
        // a stale `Speculative` here would send the fallback straight back into
        // speculation.
        args.plan.decoder = DecoderPlan::Autoregressive;
        args.plan.use_paged_attention = self.paged_adapter.is_some();
        if args.plan.use_paged_attention {
            crate::engine::paged_turn::run_paged_turn(self, args)
        } else {
            self.run_flat_ar_turn(args)
        }
    }

    /// Route a drafter seed that did not complete. A cooperative CANCELLATION must
    /// reach the caller as an ERROR, never be silently converted into a completed AR
    /// turn, so it cleans the caches the partial seed advanced and propagates the
    /// distinguished error. Anything else takes the fail-closed
    /// [`mtp_seed_failed_fallback`](Self::mtp_seed_failed_fallback).
    fn mtp_seed_aborted(&mut self, args: &mut WholeTurnArgs<'_>, err: Error) -> Result<TurnOutput> {
        if err.reason == PREFILL_CANCELLED {
            self.pending_mtp_draft_seed = None;
            self.reset_caches_internal();
            return Err(err);
        }
        self.mtp_seed_failed_fallback(args, &err)
    }

    /// Whole-turn speculative MTP core (fresh and delta turns). Prefills the FULL
    /// token stream (re-prefilling the cached history on warm deltas) while seeding the
    /// MTP head's own KV over the same stream, then drives run_mtp_turn at depth 1.
    /// Returns `TurnOutput` so the fail-closed seed fallback can hand back the AR
    /// lane's own output unchanged.
    fn run_mtp_whole_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        if args.tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        // The shared parameter resolver leaves `extra_eos_ids` empty for the executor to
        // populate; union in the checkpoint's full EOS set so MTP turns stop on the
        // alternate terminators, matching the paged executor.
        let mut p = args.params.clone();
        p.extra_eos_ids = self.extra_eos_ids();
        // GUARD for the depth-1 invariant this family's flat MTP core rests on. The
        // `p.mtp_depth.min(1)` at the call site below only SEEDS the turn: `run_mtp_cycle`
        // re-picks the depth from `AdaptiveDepthPolicy::pick_depth()` whenever
        // `mtp_adaptive_depth` is set, and that policy sweeps 1..=5 regardless of the
        // seed. The two pins are ONE guard.
        if p.mtp_adaptive_depth {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    target: "mlx_core::nemotron_h",
                    "mtpAdaptiveDepth is set on a NemotronH MTP turn: the flat MTP core \
                     is validated at draft depth 1 only, so the adaptive policy is \
                     disabled and every cycle runs at depth 1."
                );
            });
            p.mtp_adaptive_depth = false;
        }
        let p = &p;
        let eos_id = args.eos_id;
        let thinking = args.thinking;
        let tokenizer = args.tokenizer.clone();
        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let max_new_tokens = p.max_new_tokens;
        let report_perf = p.report_performance;
        let tokens = args.tokens.to_vec();
        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut generated_tokens: Vec<u32> = Vec::new();
        // Penalty history must start from the FULL rendered prompt:
        // `cached_token_history` is empty on fresh turns, so the repetition, presence,
        // frequency and ngram controls would otherwise ignore the prompt.
        let mut token_history = tokens.clone();
        // "length" like the other decode paths; run_mtp_turn overrides it on EOS.
        let mut finish_reason = String::from("length");
        let mut first_token_instant: Option<std::time::Instant> = None;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        // The flat MTP head reads the backbone KV, so the caches must hold the FULL
        // token stream: reset + prefill. On a paged model the flat core takes over
        // `self.caches` wholesale, so park any adapter-owned scheduled sequence FIRST or
        // its per-request state is lost to a later paged turn.
        if self.active_scheduled_seq.is_some() {
            self.park_active_scheduled_caches();
        }
        self.reset_caches_internal();
        self.flat_mtp_caches_desynced = false;

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("nemotron_chat", "nemotron_h");
        profiler.set_prompt_tokens(tokens.len() as u32);
        profiler.snapshot_memory_before();

        profiler.begin_prefill();
        let turn_cancel = self.turn_cancel.clone();
        let prompt = MxArray::from_uint32(&tokens, &[1, tokens.len() as i64])?;
        // The drafter's own KV history is seeded from the SAME chunked pass that fills
        // the backbone caches (no second prompt forward).
        let mut mtp_caches = super::mtp::NemotronHMtpModule::fresh_caches(&self.config);
        let mut committed_len = 0i32;
        let (prefill_logits, h_last) = match self.chunked_prefill_seeding_mtp(
            &prompt,
            generation_stream,
            &mut mtp_caches,
            &mut committed_len,
        ) {
            Ok(v) => v,
            Err(e) => {
                drop(wired_ctx);
                return self.mtp_seed_aborted(args, e);
            }
        };
        let seq_len = prefill_logits.shape_at(1)?;
        let mut last_logits = prefill_logits
            .slice_axis(1, seq_len - 1, seq_len)?
            .squeeze(Some(&[1]))?;
        profiler.end_prefill();

        last_logits = crate::engine::apply_all_penalties(last_logits, &token_history, p)?;
        let y = crate::sampling::sample(&last_logits, p.sampling_config)?;
        y.eval();
        let y_id = y.item_at_int32(0)? as u32;
        // Final prompt slot: (h_{T-1}, emb(y)). committed_len reaches T here.
        if let Err(e) = self.seed_mtp_final_slot(&h_last, y_id, &mut mtp_caches, &mut committed_len)
        {
            drop(wired_ctx);
            return self.mtp_seed_aborted(args, e);
        }
        debug_assert_eq!(committed_len as usize, tokens.len());
        self.pending_mtp_draft_seed = Some((mtp_caches, committed_len));

        let mut reasoning_tracker =
            crate::engine::ReasoningTracker::from_setup(&thinking, think_end_id);
        let mut rng = rand::rng();
        let outcome = crate::engine::mtp_turn::run_mtp_turn(
            self,
            &mut rng,
            crate::engine::mtp_turn::MtpTurnArgs {
                y: y.clone(),
                // Depth is clamped to 1 by POLICY, not by architecture. This
                // clamp only SEEDS the turn; it is a real ceiling only because
                // `p.mtp_adaptive_depth` was pinned false at the top of this
                // function — the two are ONE guard and must move together.
                depth: p.mtp_depth.min(1),
                params: p,
                reasoning_tracker: &mut reasoning_tracker,
                profiler: &mut profiler,
                max_new_tokens,
                eos_id,
                generated_tokens: &mut generated_tokens,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf,
                generation_stream,
                prompt_hidden: None,
                prompt_hidden_ids: None,
                cancel_flag: turn_cancel.as_deref(),
            },
            None,
        )?;
        let last_in_cache = outcome.last_in_cache;
        self.flat_mtp_last_rollback_unemitted = outcome.rollback_unemitted;
        if outcome.desynced {
            self.flat_mtp_caches_desynced = true;
        }

        self.save_cache_state_internal(
            p.reuse_cache,
            &tokens,
            &generated_tokens,
            mtp_history_drop_last(last_in_cache, &finish_reason, outcome.rollback_unemitted),
        );

        let performance = crate::engine::finalize::compute_performance_metrics(
            generation_start,
            first_token_instant,
            tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        let mut result = crate::engine::finalize::finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking.enabled,
            tokens.len() as u32,
            reasoning_tracker.reasoning_token_count(),
        )?;
        // The flat MTP path re-prefills the full prompt every turn.
        result.cached_tokens = 0;
        Ok(TurnOutput::Complete(Box::new(result)))
    }
}

impl MtpBackend for NemotronHInner {
    type MtpDecode<'a>
        = NemotronHMtpStepper<'a>
    where
        Self: 'a;

    fn begin_mtp_decode(&mut self, _setup: &MtpTurnSetup<'_>) -> Result<Self::MtpDecode<'_>> {
        let embedding = self.embedding.clone();
        // FAIL-CLOSED: the drafter owns a KV history and MUST have been seeded over the
        // prompt by `run_mtp_whole_turn`. Defaulting to an empty cache would silently
        // reintroduce the no-history bug this port exists to fix.
        let (mtp_caches, committed_len) = self.pending_mtp_draft_seed.take().ok_or_else(|| {
            Error::from_reason(
                "NemotronH MTP decode started without a seeded drafter cache: \
                     the turn must call chunked_prefill_seeding_mtp + seed_mtp_final_slot \
                     and install pending_mtp_draft_seed first",
            )
        })?;
        Ok(NemotronHMtpStepper {
            inner: self,
            embedding,
            mtp_caches,
            committed_len,
            snap: None,
            replay_err: None,
            mtp_desynced: false,
        })
    }
}

/// Physical and trained context limits captured at load time, surfaced through
/// `context_limits()` so the ChatSession preflight can compact or reject against
/// the paged pool's ACTUAL capacity instead of the trained window.
#[napi(object)]
#[derive(Clone, Copy)]
pub struct NemotronHContextLimits {
    pub trained_window_tokens: u32,
    pub effective_window_tokens: u32,
    pub paged_block_capacity: u32,
    pub paged_block_size: u32,
}

impl NemotronHContextLimits {
    pub(crate) fn from_tuple(value: (u32, u32, u32, u32)) -> Self {
        Self {
            trained_window_tokens: value.0,
            effective_window_tokens: value.1,
            paged_block_capacity: value.2,
            paged_block_size: value.3,
        }
    }
}

/// NVIDIA Nemotron 3.5 Lightning language model: hybrid Mamba-2 SSM + GQA + MoE-FFN
/// with an optional in-checkpoint MTP head. All model state lives on a dedicated OS
/// thread; NAPI methods dispatch commands via channels.
#[napi]
pub struct NemotronHModel {
    /// Dedicated model thread owning `NemotronHSchedulerState`.
    pub(crate) thread: crate::model_thread::ModelThread<NemotronHCmd>,
    pub(crate) config: NemotronHConfig,
    /// Snapshot of `NemotronHInner::has_mtp_weights()` at construction.
    pub(crate) mtp_active: bool,
    /// Snapshot of `NemotronHInner::paged_adapter.is_some()` at load time.
    /// Surfaced through `hasBlockPagedCache()` so the server can rely on
    /// native content-addressed block reuse instead of the JS warm slot.
    pub(crate) paged_active: bool,
    /// Load-time physical/trained context limits snapshot (paged adapter
    /// block capacity vs the checkpoint's trained window).
    pub(crate) context_limits: NemotronHContextLimits,
    /// RAII: unregisters this model's baseline from the cache-limit
    /// coordinator on drop.
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
    /// RAII: unregisters this model's private paged KV pool from the
    /// cache-limit coordinator on drop. `None` when paged cache is off.
    pub(crate) _pool_cache_limit_guard: Option<crate::cache_limit::PoolCacheLimitGuard>,
}

#[napi]
impl NemotronHModel {
    /// Load a NemotronH model from a directory containing safetensors and
    /// config.json.
    #[napi]
    pub async fn load(model_path: String) -> Result<NemotronHModel> {
        super::persistence::load_with_thread(&model_path).await
    }

    /// Whether this checkpoint shipped a complete MTP head (speculative
    /// decoding is available when enableMtp is set on the request).
    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.mtp_active
    }

    /// Whether `ChatSession` should turn MTP ON when the caller sets nothing. FALSE
    /// deliberately, about SCHEDULING not speed: this family's flat MTP core has no
    /// streaming arm, so `enable_mtp == Some(true)` costs a SYNC turn its scheduled
    /// slot — `chat_requires_barrier` routes it to the EXCLUSIVE lane and out of
    /// continuous batching. A streaming turn plans plain autoregressive and keeps its
    /// slot. `MLX_NEMOTRON_MTP_DEFAULT=1` flips the default.
    #[napi]
    pub fn mtp_auto_enabled(&self) -> bool {
        static OPT_IN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *OPT_IN.get_or_init(|| std::env::var("MLX_NEMOTRON_MTP_DEFAULT").is_ok_and(|v| v == "1"))
            && self.mtp_active
    }

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance (default-on unless `use_block_paged_cache: false`).
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    /// Physical/trained context limits captured at load, so the ChatSession preflight
    /// rejects long conversations instead of failing inside paged-cache allocation.
    #[napi]
    pub fn context_limits(&self) -> NemotronHContextLimits {
        self.context_limits
    }

    /// Get the model configuration.
    #[napi]
    pub fn get_config(&self) -> NemotronHConfig {
        self.config.clone()
    }

    /// Native admission capacity for the server's per-model semaphore.
    /// Paged models advertise the scheduler lane (up to 8 default); flat
    /// models and forced-serial processes report 1.
    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        if self.paged_active && !NemotronHSchedulerState::force_serial() {
            scheduler_max_num_seqs_for(32) as u32
        } else {
            1
        }
    }

    /// Snapshot scheduler occupancy and paged-pool admission telemetry.
    #[napi]
    pub async fn scheduler_stats(&self) -> Result<engine::SchedulerStatsJs> {
        send_and_await(&self.thread, |reply| NemotronHCmd::SchedulerStats { reply }).await
    }

    /// Test-only flat-MTP seam snapshot:
    /// `(history_len, attn_kv_offset, desynced, rollback_unemitted)`.
    /// `rollback_unemitted` says whether the preceding turn stopped mid-cycle WITHOUT
    /// consulting the latch. Serialized behind the model thread.
    #[doc(hidden)]
    pub async fn mtp_flat_state_for_test(&self) -> Result<(usize, i32, bool, usize)> {
        send_and_await(&self.thread, |reply| NemotronHCmd::MtpFlatStateForTest {
            reply,
        })
        .await
    }
}

#[napi]
impl NemotronHModel {
    /// Estimated number of model parameters.
    #[napi]
    pub fn num_parameters(&self) -> i64 {
        let h = self.config.hidden_size as i64;
        let v = self.config.vocab_size as i64;
        let mamba_i = self.config.mamba_intermediate_size() as i64;
        let conv_dim = self.config.mamba_conv_dim() as i64;
        let attn_q = self.config.num_attention_heads as i64 * self.config.head_dim as i64;
        let attn_kv = self.config.num_key_value_heads as i64 * self.config.head_dim as i64;
        let moe_i = self.config.intermediate_size as i64;
        let shared_i = self.config.moe_shared_expert_intermediate_size as i64;
        let e = self.config.n_routed_experts as i64;

        let mut total = v * h; // embedding
        total += h * v; // untied lm_head
        total += h; // final_norm
        for i in 0..self.config.num_hidden_layers as usize {
            total += h; // layer norm
            if self.config.is_mamba_layer(i) {
                total += h * (mamba_i + conv_dim + self.config.mamba_num_heads as i64);
                total += mamba_i * h;
                total += conv_dim * self.config.conv_kernel as i64;
                total += 3 * mamba_i; // A_log, D, dt_bias, norm weight
            } else if self.config.is_attention_layer(i) {
                total += h * attn_q + 2 * h * attn_kv + attn_q * h;
            } else if self.config.is_moe_layer(i) {
                total += e * 2 * h * moe_i + 2 * h * shared_i + e * h;
            }
        }
        total
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: NemotronHModel,
    thread_cmd: crate::models::nemotron_h::model::NemotronHCmd,
    thread: direct,
    image_guard: text_only,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}

#[cfg(test)]
mod scheduler_tests {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use super::*;
    use crate::engine::backend::{ChunkSink, ThinkingSetup};
    use crate::engine::persistence::compiled_forward_backend_available;
    use crate::engine::plan::{MediaInputs, TurnPath, TurnPlan, TurnRequest};
    use crate::engine::types::{ChatConfig, ChatStreamChunk};

    /// Tiny hybrid config: mamba(0) + moe(1) + attention(2), dense bf16
    /// weights (random init), paged adapter ON.
    fn tiny_paged_config() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 256,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            // Real checkpoint head_dim 128 (the generic batched paged route);
            // the mamba head_dim is a separate config field and stays tiny.
            head_dim: 128,
            max_position_embeddings: 512,
            layer_norm_epsilon: 1e-5,
            layers_block_type: vec![
                "linear_attention".into(),
                "moe".into(),
                "full_attention".into(),
            ],
            mamba_num_heads: 2,
            mamba_head_dim: 2,
            ssm_state_size: 2,
            n_groups: 1,
            conv_kernel: 4,
            chunk_size: 4,
            time_step_min: 0.001,
            time_step_limit: None,
            n_routed_experts: 4,
            num_experts_per_tok: 1,
            routed_scaling_factor: 1.0,
            norm_topk_prob: true,
            intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            eos_token_ids: vec![2],
            mtp_layers_block_type: Vec::new(),
            n_mtp_layers: 0,
            paged_cache_memory_mb: Some(256),
            paged_block_size: Some(16),
            use_block_paged_cache: Some(true),
        }
    }

    /// Greedy pick with PRODUCTION tie-break semantics: the FIRST maximal index.
    /// NOT `Iterator::max_by`, which returns the LAST — every lane this is an oracle
    /// for (`mx::argmax`, the compiled greedy sampler, the MTP accept gate) keeps the
    /// smaller index, and real ties made `max_by` report a phantom divergence.
    fn argmax(vec: &[f32]) -> usize {
        let mut best = 0usize;
        for (i, v) in vec.iter().enumerate() {
            if *v > vec[best] {
                best = i;
            }
        }
        best
    }

    /// The tiny fixture's MoE layer is built with quantized (uint8) expert backends;
    /// install deterministic dense bf16 stacks so the gather kernels run.
    fn install_dense_moe(inner: &mut NemotronHInner) -> Result<()> {
        let h = inner.config.hidden_size as i64;
        let e = inner.config.n_routed_experts as i64;
        let inter = inner.config.intermediate_size as i64;
        let up: Vec<f32> = (0..e * inter * h)
            .map(|i| ((i as f32) * 0.017) % 1.0 - 0.5)
            .collect();
        let down: Vec<f32> = (0..e * h * inter)
            .map(|i| ((i as f32) * 0.031) % 1.0 - 0.5)
            .collect();
        let up_w =
            MxArray::from_float32(&up, &[e, inter, h])?.astype(crate::array::DType::BFloat16)?;
        let down_w =
            MxArray::from_float32(&down, &[e, h, inter])?.astype(crate::array::DType::BFloat16)?;
        let moe = inner.layers[1]
            .moe_mut()
            .ok_or_else(|| Error::from_reason("fixture layer 1 must be MoE"))?;
        moe.experts.set_dense(&up_w, &down_w)
    }

    #[test]
    fn tiny_paged_pool_reports_nonzero_allocated_bytes() {
        let inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        let Some(adapter) = inner.paged_adapter.as_ref() else {
            eprintln!("skipping (no Metal backend)");
            return;
        };
        let pool_bytes = adapter
            .pool_allocated_bytes()
            .expect("paged pool byte accounting");
        assert!(
            pool_bytes > 0,
            "paged pool must report its private Metal bytes so load_with_thread can register_pool"
        );
    }

    #[test]
    fn uncapped_config_does_not_reserve_a_configured_pool() {
        let mut config = tiny_paged_config();
        config.paged_cache_memory_mb = None;
        let before = crate::cache_limit::coordinator().registered_pool_bytes();
        let guard = reserve_configured_paged_pool(&config).expect("uncapped plan");
        assert!(
            guard.is_none(),
            "uncapped loads must leave reservation to the adaptive CAS path"
        );
        assert_eq!(
            crate::cache_limit::coordinator().registered_pool_bytes(),
            before
        );
    }

    #[test]
    fn configured_pool_is_reserved_before_metal_alloc() {
        let config = tiny_paged_config();
        let Some(plan) = configured_paged_pool_plan(&config).expect("plan") else {
            eprintln!("skipping (no Metal backend or paged gated off)");
            return;
        };
        let before = crate::cache_limit::coordinator().registered_pool_bytes();
        let guard = reserve_configured_paged_pool(&config)
            .expect("reserve")
            .expect("configured pool must register");
        assert_eq!(
            crate::cache_limit::coordinator()
                .registered_pool_bytes()
                .saturating_sub(before),
            plan.pool_bytes,
            "configured reservation must be visible before Inner::new allocates"
        );
        let inner = NemotronHInner::new(config).expect("inner builds");
        let adapter = inner
            .paged_adapter
            .as_ref()
            .expect("configured plan existed so the adapter must build");
        assert_eq!(
            adapter
                .pool_allocated_bytes()
                .expect("paged pool byte accounting"),
            plan.pool_bytes
        );
        drop(guard);
    }

    /// Pure-function gate on the prefill break-set: no internal boundary inside a
    /// Mamba-2 chunk, and every slice at most `slice_tokens` long.
    #[test]
    fn chunk_aligned_prefill_slices_never_split_a_chunk() {
        let cases: &[(u32, u32, u32, u32)] = &[
            // Cold start: the 2048 grid from 0 lands on chunk multiples.
            (0, 3000, 2048, 128),
            // Warm partial-prefix hit at a non-chunk-multiple boundary.
            (112, 2160, 2048, 128),
            (112, 4096, 2048, 128),
            // Sub-chunk ranges.
            (0, 64, 2048, 128),
            (112, 140, 2048, 128),
            // Exact multiples.
            (128, 1280, 2048, 128),
            // A valid checkpoint may declare a different chunk_size.
            (0, 3000, 2048, 256),
            (112, 2160, 2048, 256),
            (256, 1280, 2048, 256),
            // chunk_size that does NOT divide the slice grid.
            (0, 3000, 2048, 192),
            (192, 2240, 2048, 192),
        ];
        for &(start, end, slice_tokens, chunk_size) in cases {
            let slices = chunk_aligned_prefill_slices(start, end, slice_tokens, chunk_size);
            assert!(!slices.is_empty(), "{start}..{end} must produce slices");
            assert_eq!(
                slices[0].0, start,
                "first slice starts at the effective prefix"
            );
            assert_eq!(slices.last().unwrap().1, end, "last slice reaches the end");
            for (i, &(s, e)) in slices.iter().enumerate() {
                assert!(s < e, "slice {i} of {start}..{end} is degenerate: {s}..{e}");
                assert!(
                    e - s <= slice_tokens,
                    "slice {i} of {start}..{end} exceeds the token budget: {} > {slice_tokens}",
                    e - s
                );
                // Every boundary except the very first start (which the family
                // cannot move) must be a chunk multiple.
                if i > 0 || s % chunk_size == 0 {
                    assert!(
                        s % chunk_size == 0,
                        "slice {i} of {start}..{end} starts inside a chunk: {s} % {chunk_size} != 0"
                    );
                }
            }
        }
    }

    /// The mamba-state-reuse predicate only fires for a strict extension of
    /// the exact saved history at the cached-prefix boundary.
    #[test]
    fn mamba_state_reusable_requires_exact_extension() {
        let history = [1u32, 2, 3];
        // Exact-owner continuation, prefix hit at the full history length.
        assert!(mamba_state_reusable(&[1, 2, 3, 4], &history, 3));
        // Cold start: never reusable.
        assert!(!mamba_state_reusable(&[1, 2, 3, 4], &history, 0));
        // Partial/foreign hit below the history length.
        assert!(!mamba_state_reusable(&[1, 2, 3, 4], &history, 2));
        // Divergent prompt.
        assert!(!mamba_state_reusable(&[1, 2, 9], &history, 3));
        // Empty history.
        assert!(!mamba_state_reusable(&[1], &[], 0));
    }
    /// Stacking and row-slicing a Mamba2State is an identity round trip.
    #[test]
    fn mamba_state_stack_rows_and_row_round_trip() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let s1 = Mamba2State {
            conv: MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]).unwrap(),
            ssm: MxArray::from_float32(&[0.1, 0.2, 0.3, 0.4], &[1, 2, 1, 2]).unwrap(),
        };
        let s2 = Mamba2State {
            conv: MxArray::from_float32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[1, 3, 2]).unwrap(),
            ssm: MxArray::from_float32(&[0.5, 0.6, 0.7, 0.8], &[1, 2, 1, 2]).unwrap(),
        };
        let stacked = Mamba2State::stack_rows(&[&s1, &s2]).expect("stack");
        assert_eq!(stacked.conv.shape().unwrap().to_vec(), vec![2, 3, 2]);
        assert_eq!(stacked.ssm.shape().unwrap().to_vec(), vec![2, 2, 1, 2]);
        let r1 = stacked.row(0).expect("row 0");
        let r2 = stacked.row(1).expect("row 1");
        assert_eq!(
            r1.conv.to_float32().unwrap().to_vec(),
            s1.conv.to_float32().unwrap().to_vec()
        );
        assert_eq!(
            r1.ssm.to_float32().unwrap().to_vec(),
            s1.ssm.to_float32().unwrap().to_vec()
        );
        assert_eq!(
            r2.conv.to_float32().unwrap().to_vec(),
            s2.conv.to_float32().unwrap().to_vec()
        );
        assert_eq!(
            r2.ssm.to_float32().unwrap().to_vec(),
            s2.ssm.to_float32().unwrap().to_vec()
        );
    }
    /// Batched N=2 decode is T=0-identical to two serial decodes. The batched==serial
    /// equivalence gate for the mamba stack/scatter, the batched paged attention
    /// kernels, and the MoE routing over [N,1,H].
    #[test]
    fn batched_decode_equals_serial_t0() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let cfg = tiny_paged_config();
        let mut inner = NemotronHInner::new(cfg).expect("inner builds");
        assert!(
            inner.paged_adapter.is_some(),
            "gate requires the paged adapter"
        );
        install_dense_moe(&mut inner).expect("install dense experts");

        let prompt_a: Vec<u32> = vec![1, 5, 9, 3];
        let prompt_b: Vec<u32> = vec![2, 7, 11, 4];
        let stream = Stream::default(DeviceType::Gpu);

        // ---- serial oracle: prefill + one decode per request ----
        let prefix_a = inner
            .prime_prefix_state_for(1, &prompt_a, &[], 101, false)
            .expect("prime a");
        let prefix_b = inner
            .prime_prefix_state_for(2, &prompt_b, &[], 102, false)
            .expect("prime b");
        inner.activate_paged_seq(1).expect("activate a");
        let _ = inner
            .paged_prefill(&prompt_a, &prefix_a, stream)
            .expect("prefill a");
        inner.activate_paged_seq(2).expect("activate b");
        let _ = inner
            .paged_prefill(&prompt_b, &prefix_b, stream)
            .expect("prefill b");

        inner.activate_paged_seq(1).expect("activate a2");
        let ser_a = inner
            .run_paged_decode_step(12)
            .expect("serial decode a")
            .to_float32()
            .unwrap()
            .to_vec();
        inner.activate_paged_seq(2).expect("activate b2");
        let ser_b = inner
            .run_paged_decode_step(13)
            .expect("serial decode b")
            .to_float32()
            .unwrap()
            .to_vec();
        let tok_a = argmax(&ser_a);
        let tok_b = argmax(&ser_b);

        // ---- batched: reset both requests and replay the SAME decode ----
        // Cold restart with fresh salts so the second prefill cannot hit the
        // allocator's prefix cache and take a different arithmetic path.
        inner.park_active_scheduled_caches();
        inner.reset_scheduled_caches_for(1);
        inner.reset_scheduled_caches_for(2);
        {
            let adapter = inner.paged_adapter.as_mut().expect("adapter");
            let _ = adapter.release_request_for(1);
            let _ = adapter.release_request_for(2);
        }
        let prefix_a = inner
            .prime_prefix_state_for(1, &prompt_a, &[], 201, false)
            .expect("re-prime a");
        let prefix_b = inner
            .prime_prefix_state_for(2, &prompt_b, &[], 202, false)
            .expect("re-prime b");
        inner.activate_paged_seq(1).expect("activate a3");
        let _ = inner
            .paged_prefill(&prompt_a, &prefix_a, stream)
            .expect("prefill a2");
        inner.activate_paged_seq(2).expect("activate b3");
        let _ = inner
            .paged_prefill(&prompt_b, &prefix_b, stream)
            .expect("prefill b2");

        let batched = inner
            .run_paged_decode_step_batched(&[(1, 12), (2, 13)])
            .expect("batched decode")
            .to_float32()
            .unwrap()
            .to_vec();
        let row_len = ser_a.len();
        assert_eq!(batched.len(), 2 * row_len, "batched logits shape");
        // Numeric probe: how close is the batched row to the serial row?
        let mut max_diff_a = 0.0f32;
        let mut max_diff_b = 0.0f32;
        for i in 0..row_len {
            max_diff_a = max_diff_a.max((batched[i] - ser_a[i]).abs());
            max_diff_b = max_diff_b.max((batched[row_len + i] - ser_b[i]).abs());
        }
        eprintln!("DEBUG batched-vs-serial max|diff| rowA={max_diff_a} rowB={max_diff_b}");
        let got_a = argmax(&batched[..row_len]);
        let got_b = argmax(&batched[row_len..]);
        assert_eq!(
            got_a, tok_a,
            "batched row 0 T=0 token differs from serial decode a"
        );
        assert_eq!(
            got_b, tok_b,
            "batched row 1 T=0 token differs from serial decode b"
        );
    }

    /// The singleton wave delegates to the scalar path and stays identical.
    #[test]
    fn singleton_batched_decode_delegates_to_scalar() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let cfg = tiny_paged_config();
        let mut inner = NemotronHInner::new(cfg).expect("inner builds");
        install_dense_moe(&mut inner).expect("install dense experts");
        let prompt: Vec<u32> = vec![1, 5, 9, 3];
        let stream = Stream::default(DeviceType::Gpu);
        let prefix = inner
            .prime_prefix_state_for(1, &prompt, &[], 1, false)
            .expect("prime");
        inner.activate_paged_seq(1).expect("activate");
        let _ = inner
            .paged_prefill(&prompt, &prefix, stream)
            .expect("prefill");
        let scalar = inner
            .run_paged_decode_step(12)
            .expect("scalar")
            .to_float32()
            .unwrap()
            .to_vec();
        inner.park_active_scheduled_caches();
        inner.reset_scheduled_caches_for(1);
        let _ = inner.paged_adapter.as_mut().unwrap().release_request_for(1);
        let prefix2 = inner
            .prime_prefix_state_for(1, &prompt, &[], 2, false)
            .expect("re-prime");
        inner.activate_paged_seq(1).expect("activate 2");
        let _ = inner
            .paged_prefill(&prompt, &prefix2, stream)
            .expect("prefill 2");
        let via_batched = inner
            .run_paged_decode_step_batched(&[(1, 12)])
            .expect("singleton batched")
            .to_float32()
            .unwrap()
            .to_vec();
        assert_eq!(
            argmax(&via_batched),
            argmax(&scalar),
            "singleton batched T=0 token differs from scalar decode"
        );
    }

    /// The row-exact batched branch (quantized checkpoints) must also be
    /// T=0-identical to serial.
    #[test]
    fn row_exact_batched_decode_equals_serial_t0() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let cfg = tiny_paged_config();
        let mut inner = NemotronHInner::new(cfg).expect("inner builds");
        assert!(
            inner.paged_adapter.is_some(),
            "gate requires the paged adapter"
        );
        install_dense_moe(&mut inner).expect("install dense experts");
        inner.row_exact_decode_projections = true;

        let prompt_a: Vec<u32> = vec![1, 5, 9, 3];
        let prompt_b: Vec<u32> = vec![2, 7, 11, 4];
        let stream = Stream::default(DeviceType::Gpu);
        let prefix_a = inner
            .prime_prefix_state_for(1, &prompt_a, &[], 101, false)
            .expect("prime a");
        let prefix_b = inner
            .prime_prefix_state_for(2, &prompt_b, &[], 102, false)
            .expect("prime b");
        inner.activate_paged_seq(1).expect("act a");
        let _ = inner
            .paged_prefill(&prompt_a, &prefix_a, stream)
            .expect("pf a");
        inner.activate_paged_seq(2).expect("act b");
        let _ = inner
            .paged_prefill(&prompt_b, &prefix_b, stream)
            .expect("pf b");
        inner.activate_paged_seq(1).expect("act a2");
        let ser_a = inner
            .run_paged_decode_step(12)
            .expect("ser a")
            .to_float32()
            .unwrap()
            .to_vec();
        inner.activate_paged_seq(2).expect("act b2");
        let ser_b = inner
            .run_paged_decode_step(13)
            .expect("ser b")
            .to_float32()
            .unwrap()
            .to_vec();
        inner.park_active_scheduled_caches();
        inner.reset_scheduled_caches_for(1);
        inner.reset_scheduled_caches_for(2);
        {
            let adapter = inner.paged_adapter.as_mut().unwrap();
            let _ = adapter.release_request_for(1);
            let _ = adapter.release_request_for(2);
        }
        let prefix_a = inner
            .prime_prefix_state_for(1, &prompt_a, &[], 201, false)
            .expect("reprime a");
        let prefix_b = inner
            .prime_prefix_state_for(2, &prompt_b, &[], 202, false)
            .expect("reprime b");
        inner.activate_paged_seq(1).expect("act a3");
        let _ = inner
            .paged_prefill(&prompt_a, &prefix_a, stream)
            .expect("pf a2");
        inner.activate_paged_seq(2).expect("act b3");
        let _ = inner
            .paged_prefill(&prompt_b, &prefix_b, stream)
            .expect("pf b2");
        let batched = inner
            .run_paged_decode_step_batched(&[(1, 12), (2, 13)])
            .expect("row-exact batched")
            .to_float32()
            .unwrap()
            .to_vec();
        let row_len = ser_a.len();
        assert_eq!(batched.len(), 2 * row_len, "row-exact batched logits shape");
        assert_eq!(
            argmax(&batched[..row_len]),
            argmax(&ser_a),
            "row-exact batched row 0 T=0 token differs from serial decode a"
        );
        assert_eq!(
            argmax(&batched[row_len..]),
            argmax(&ser_b),
            "row-exact batched row 1 T=0 token differs from serial decode b"
        );
    }

    /// `finish_reason == "length"` is NOT a drop signal by itself: the engine leaves
    /// `last_in_cache` at `true` on every length exit, so `rollback_unemitted` tells
    /// the two apart — 0 is the never-forwarded token (drop), nonzero is the mid-cycle
    /// exit where verify already wrote it (keep).
    #[test]
    fn mtp_history_drop_last_separates_the_two_length_exits() {
        // Ordinary length exits still drop the never-forwarded final token.
        assert!(
            mtp_history_drop_last(true, "length", 0),
            "Step-A / clean-boundary length exit must drop the unforwarded token"
        );
        assert!(mtp_history_drop_last(false, "length", 0));
        // Mid-cycle length exit: verify already forwarded the last emitted
        // token, so dropping it would leave history behind the caches.
        assert!(
            !mtp_history_drop_last(true, "length", 1),
            "mid-cycle length exit must keep the verified token"
        );
        assert!(!mtp_history_drop_last(true, "length", 4));
        // An honest `last_in_cache == false` always wins, mid-cycle or not.
        assert!(mtp_history_drop_last(false, "length", 2));
        assert!(mtp_history_drop_last(false, "stop", 2));
        // Every non-length reason rides purely on `last_in_cache`.
        assert!(!mtp_history_drop_last(true, "stop", 0));
        assert!(!mtp_history_drop_last(true, "cancelled", 0));
    }

    /// After a mid-cycle length stop the saved history must be exactly as long as the
    /// token stream the caches forwarded, while the ordinary length stop still ends
    /// one token short.
    #[test]
    fn mid_cycle_length_stop_saves_a_history_matching_the_caches() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        let prompt = [1u32, 5, 9, 3];
        let generated = [11u32, 12, 13];

        // Mid-cycle length stop: all 3 emitted tokens were forwarded by verify.
        inner.save_cache_state_internal(
            true,
            &prompt,
            &generated,
            mtp_history_drop_last(true, "length", 2),
        );
        assert_eq!(
            inner.cached_token_history.len(),
            prompt.len() + generated.len(),
            "mid-cycle length stop must save every forwarded token"
        );
        assert_eq!(inner.cached_token_history.last(), generated.last());

        // Ordinary length stop: the final token was never forwarded.
        inner.save_cache_state_internal(
            true,
            &prompt,
            &generated,
            mtp_history_drop_last(true, "length", 0),
        );
        assert_eq!(
            inner.cached_token_history.len(),
            prompt.len() + generated.len() - 1,
            "ordinary length stop must still drop the unforwarded token"
        );
    }

    /// The scheduler activates a sequence and then activates it AGAIN one call later
    /// inside `prepare_scheduled_prefix`. The second activation takes the already-live
    /// early return, which must leave the survival fact untouched.
    ///
    /// MUTATION: re-adding `active_seq_recurrent_survived = true` to that return.
    #[test]
    fn recurrent_survival_survives_a_second_activation() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        assert!(inner.paged_adapter.is_some(), "gate requires the adapter");

        // Fresh sequence (the shape a preemption-released one also takes).
        inner.activate_paged_seq(4).expect("activate fresh");
        assert!(
            !inner.active_seq_recurrent_survived,
            "fresh zero-state caches must not count as survived"
        );
        inner.activate_paged_seq(4).expect("re-activate live");
        assert!(
            !inner.active_seq_recurrent_survived,
            "re-activating the SAME live sequence must not manufacture survival"
        );

        // A genuinely restored state stays survived across re-activation.
        inner.park_active_scheduled_caches();
        inner.activate_paged_seq(4).expect("activate parked");
        assert!(inner.active_seq_recurrent_survived, "parked state survives");
        inner.activate_paged_seq(4).expect("re-activate live");
        assert!(
            inner.active_seq_recurrent_survived,
            "re-activation must not erase a real survival either"
        );
    }

    /// Adapter over a deliberately tiny block pool so the allocator can be
    /// drained in a handful of calls.
    fn tiny_pool_adapter(config: &NemotronHConfig, num_blocks: u32) -> PagedKVCacheAdapter {
        let block_size = config.paged_block_size.unwrap_or(16);
        let pa_config = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            // The pool validates this floor even though `num_blocks` below
            // is what actually sizes the allocation.
            gpu_memory_mb: 256,
            head_size: config.head_dim as u32,
            num_kv_heads: config.num_key_value_heads as u32,
            num_layers: config.attention_layer_idxs().len() as u32,
            use_fp8_cache: Some(false),
            max_seq_len: Some(config.max_position_embeddings as u32),
            max_batch_size: Some(32),
        };
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            num_blocks, num_blocks, block_size,
        )));
        let pool = mlx_paged_attn::LayerKVPool::new(
            pa_config,
            num_blocks,
            num_blocks,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        )
        .expect("tiny LayerKVPool");
        PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).expect("tiny adapter")
    }

    /// Flatten one sequence's mamba conv + SSM state into comparable f32s. Parks
    /// first so the state is reachable whether or not the sequence is the live one.
    fn mamba_fingerprint(inner: &mut NemotronHInner, seq_id: SeqId) -> Vec<f32> {
        inner.park_active_scheduled_caches();
        let caches = inner
            .scheduled_caches
            .get(&seq_id)
            .expect("sequence has parked caches");
        let mut out = Vec::new();
        for cache in caches {
            if let Some(state) = cache.as_mamba_state() {
                out.extend(state.conv.to_float32().expect("conv f32").to_vec());
                out.extend(state.ssm.to_float32().expect("ssm f32").to_vec());
            }
        }
        assert!(!out.is_empty(), "fixture must have a mamba layer");
        out
    }

    /// A mid-batch allocator squeeze must leave EVERY surviving row exactly where it
    /// was — cursor and mamba state both. The scheduler treats the error as "blocked"
    /// for the whole wave and re-feeds the same token, so a row that already committed
    /// would fold it into its non-invertible mamba state twice. MUTATION: restoring
    /// the per-row activate + `run_paged_decode_step` loop.
    #[test]
    fn row_exact_batched_decode_unwinds_a_mid_batch_allocator_squeeze() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let cfg = tiny_paged_config();
        let mut inner = NemotronHInner::new(cfg).expect("inner builds");
        install_dense_moe(&mut inner).expect("install dense experts");
        inner.row_exact_decode_projections = true;
        let block_size = inner.config.paged_block_size.unwrap_or(16) as usize;
        inner.paged_adapter = Some(tiny_pool_adapter(&inner.config, 8));

        let stream = Stream::default(DeviceType::Gpu);
        // Row A sits mid-block: one more token needs NO new block.
        let prompt_a: Vec<u32> = vec![1, 5, 9, 3];
        // Row B sits exactly on a block boundary: one more token needs a new
        // block, which is the allocation that will fail.
        let prompt_b: Vec<u32> = (0..block_size as u32).map(|i| (i % 30) + 1).collect();

        let prefix_a = inner
            .prime_prefix_state_for(1, &prompt_a, &[], 11, false)
            .expect("prime a");
        inner.activate_paged_seq(1).expect("act a");
        let _ = inner
            .paged_prefill(&prompt_a, &prefix_a, stream)
            .expect("prefill a");
        let prefix_b = inner
            .prime_prefix_state_for(2, &prompt_b, &[], 12, false)
            .expect("prime b");
        inner.activate_paged_seq(2).expect("act b");
        let _ = inner
            .paged_prefill(&prompt_b, &prefix_b, stream)
            .expect("prefill b");

        // Drain every remaining free block through a filler sequence so the
        // wave below hits a REAL allocator exhaustion, not a capacity guard.
        let filler: Vec<u32> = vec![1];
        let _ = inner
            .prime_prefix_state_for(3, &filler, &[], 13, false)
            .expect("prime filler");
        let mut drained = 0usize;
        loop {
            let adapter = inner.paged_adapter.as_mut().expect("adapter");
            if adapter.record_token_for(3, 1).is_err() {
                break;
            }
            drained += 1;
            assert!(drained < 4096, "filler never exhausted the pool");
        }
        assert_eq!(
            inner
                .paged_adapter
                .as_ref()
                .expect("adapter")
                .block_telemetry()
                .expect("telemetry")
                .free_blocks,
            0,
            "the pool must be empty before the wave"
        );

        let cursor_a_before = inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .current_token_count_for(1)
            .expect("row a cursor");
        let cursor_b_before = inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .current_token_count_for(2)
            .expect("row b cursor");
        let mamba_a_before = mamba_fingerprint(&mut inner, 1);

        let error = match inner.run_paged_decode_step_batched(&[(1, 12), (2, 13)]) {
            Ok(_) => panic!("row 1 must fail on the exhausted pool"),
            Err(error) => error,
        };
        assert!(
            crate::engine::scheduler::is_paged_allocation_blocked(&error.reason),
            "the squeeze must stay recognizable as an allocation block, got: {}",
            error.reason
        );

        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        assert_eq!(
            adapter.current_token_count_for(1),
            Some(cursor_a_before),
            "the surviving row's adapter cursor must be unwound"
        );
        assert_eq!(
            adapter.current_token_count_for(2),
            Some(cursor_b_before),
            "the failing row's cursor is untouched by construction"
        );
        let mamba_a_after = mamba_fingerprint(&mut inner, 1);
        assert_eq!(
            mamba_a_after, mamba_a_before,
            "the surviving row's mamba state must NOT have advanced"
        );
    }

    /// Two rows of one wave naming the same sequence would record two tokens against
    /// one cursor; the row-exact branch rejects it before mutating anything.
    #[test]
    fn row_exact_batched_decode_rejects_a_duplicate_sequence() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        install_dense_moe(&mut inner).expect("install dense experts");
        inner.row_exact_decode_projections = true;
        let prompt: Vec<u32> = vec![1, 5, 9, 3];
        let stream = Stream::default(DeviceType::Gpu);
        let prefix = inner
            .prime_prefix_state_for(1, &prompt, &[], 21, false)
            .expect("prime");
        inner.activate_paged_seq(1).expect("act");
        let _ = inner.paged_prefill(&prompt, &prefix, stream).expect("pf");
        let before = inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .current_token_count_for(1)
            .expect("cursor");

        let error = match inner.run_paged_decode_step_batched(&[(1, 12), (1, 13)]) {
            Ok(_) => panic!("duplicate rows must be rejected"),
            Err(error) => error,
        };
        assert!(
            error.reason.contains("duplicate sequence"),
            "unexpected error: {}",
            error.reason
        );
        assert_eq!(
            inner
                .paged_adapter
                .as_ref()
                .expect("adapter")
                .current_token_count_for(1),
            Some(before),
            "a rejected wave must not have advanced the cursor"
        );
    }

    #[derive(Default)]
    struct CollectSink {
        chunks: Mutex<Vec<ChatStreamChunk>>,
        errors: Mutex<Vec<String>>,
    }

    impl ChunkSink for CollectSink {
        fn send(&self, chunk: Result<ChatStreamChunk>) {
            match chunk {
                Ok(chunk) => self.chunks.lock().expect("chunks").push(chunk),
                Err(error) => self
                    .errors
                    .lock()
                    .expect("errors")
                    .push(error.reason.clone()),
            }
        }
    }

    /// Adapter-less fixture WITH an MTP head: the shape that routes an `enable_mtp`
    /// turn to `run_speculative_turn`.
    fn tiny_flat_mtp_config() -> NemotronHConfig {
        NemotronHConfig {
            mtp_layers_block_type: vec!["full_attention".into(), "moe".into()],
            n_mtp_layers: 1,
            use_block_paged_cache: Some(false),
            // No stop ids: random fixture weights hit a real EOS often enough to
            // make a budget assertion flaky. `extra_eos_ids()` unions this list,
            // so emptying it leaves the unreachable `u32::MAX` session EOS as the
            // only stop and the turn always walks the full budget.
            eos_token_ids: Vec::new(),
            ..tiny_paged_config()
        }
    }

    /// A minimal real tokenizer so `decode_stream` / `think_end_id` work
    /// without a checkpoint (mirrors `engine::paged_turn`'s fixture).
    fn tiny_tokenizer() -> Arc<Qwen3Tokenizer> {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": { "a": 0, "b": 1, "c": 2, "<unk>": 3 },
                "unk_token": "<unk>"
            }
        }"#;
        let dir = std::env::temp_dir().join(format!(
            "mlx-node-nemotron-flat-ar-tok-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("fixture dir: {e}"));
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, json).unwrap_or_else(|e| panic!("fixture write: {e}"));
        let tok =
            Qwen3Tokenizer::from_file(&path).unwrap_or_else(|e| panic!("fixture tokenizer: {e}"));
        let _ = std::fs::remove_dir_all(&dir);
        Arc::new(tok)
    }

    /// The turn the engine planner resolves for a text-only request against
    /// this inner's live execution plan. The family no longer holds a second
    /// opinion, so every routing assertion below reads this one.
    fn planned_turn(
        inner: &NemotronHInner,
        speculative_requested: bool,
        streaming: bool,
    ) -> TurnPlan {
        TurnPlan::resolve(
            ChatBackend::execution_plan(inner),
            TurnRequest {
                is_delta: false,
                input_media: crate::engine::plan::MediaCapabilities::NONE,
                context_media: crate::engine::plan::MediaCapabilities::NONE,
                speculative_requested,
                streaming,
            },
        )
    }

    /// A STREAMING MTP request must be refused by the PLANNER, not by the family
    /// mid-turn: the flat MTP core has no streaming arm, so the plan names the
    /// target's own autoregressive lane and that lane really streams the budget.
    ///
    /// MUTATION: flip `supports_streaming` to `true` in `execution_plan` — the
    /// plan resolves `Speculative`/`TurnPath::Speculative` and the two planner
    /// assertions fail.
    #[test]
    fn a_streaming_mtp_request_plans_autoregressive_and_still_streams() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_flat_mtp_config()).expect("inner builds");
        install_dense_moe(&mut inner).expect("install dense experts");
        inner.mtp_weights_loaded = true;
        assert!(inner.has_mtp_weights(), "fixture must carry an MTP head");
        assert!(
            inner.paged_adapter.is_none(),
            "fixture must have NO paged adapter"
        );

        let tokenizer = tiny_tokenizer();
        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(4),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            enable_mtp: Some(true),
            ..Default::default()
        };
        let mut params = crate::engine::params::extract_chat_params(&config);
        params.enable_mtp = true;
        let plan = planned_turn(
            &inner, /* speculative_requested */ true, /* streaming */ true,
        );
        assert_eq!(
            plan.decoder,
            DecoderPlan::Autoregressive,
            "the flat MTP core must be declined on a sink-bearing turn"
        );
        assert_eq!(
            plan.path(),
            TurnPath::Generic,
            "an adapter-less target serves the refused turn on its own AR lane"
        );
        // The same request without a sink still plans MTP.
        assert_eq!(
            planned_turn(&inner, true, false).decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
        );

        let sink = CollectSink::default();
        let cancelled = AtomicBool::new(false);
        let tokens = vec![1u32, 5, 9, 3];
        let mut args = WholeTurnArgs {
            tokens: &tokens,
            tokenizer: &tokenizer,
            // u32::MAX is not a reachable id, so the turn walks to the budget.
            eos_id: u32::MAX,
            config: &config,
            params: &params,
            thinking: ThinkingSetup {
                enabled: false,
                budget: None,
            },
            plan,
            sink: Some(&sink),
            cancelled: Some(&cancelled),
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };

        let out = inner
            .run_flat_ar_turn(&mut args)
            .unwrap_or_else(|e| panic!("streaming AR turn errored: {}", e.reason));
        assert!(
            matches!(out, TurnOutput::Streamed),
            "a sink-bearing turn must report Streamed"
        );
        let errors = sink.errors.lock().expect("errors");
        assert!(errors.is_empty(), "stream carried errors: {errors:?}");
        let chunks = sink.chunks.lock().expect("chunks");
        assert!(!chunks.is_empty(), "the sink received no chunks");
        let terminal = chunks.last().expect("terminal chunk");
        assert!(terminal.done, "the stream must end with a done chunk");
        assert_eq!(
            terminal.num_tokens,
            Some(4),
            "the AR fallback must generate the full budget"
        );
    }

    /// FAIL-CLOSED for the TURN, not for the model: a drafter seed that cannot be
    /// built drops the seed and finishes THIS turn autoregressively, leaving the head
    /// armed so the next turn retries it. The forced failure is real — the config
    /// declares one MTP block while the built module has two — and the mutation
    /// stands, so the SECOND turn proves the retry by reaching the drafter and failing
    /// the same way.
    ///
    /// MUTATION A: re-add `self.mtp_weights_loaded = false` to
    /// `mtp_seed_failed_fallback` — turn 1's still-armed assertions fail.
    /// MUTATION B: drop the `mtp_seed_failures` increment — the count assertions fail.
    /// MUTATION C: propagate the error instead of falling back — the unwrap panics.
    #[test]
    fn a_failed_seed_degrades_only_its_own_turn_and_leaves_the_head_armed() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_flat_mtp_config()).expect("inner builds");
        install_dense_moe(&mut inner).expect("install dense experts");
        inner.mtp_weights_loaded = true;
        assert!(inner.has_mtp_weights());
        assert!(
            inner.execution_plan().speculative.is_some(),
            "the head starts armed"
        );

        // Force the seed to fail: `fresh_caches` follows the config, the
        // module's layers do not.
        inner.config.mtp_layers_block_type = vec!["full_attention".into()];

        let tokenizer = tiny_tokenizer();
        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(4),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            enable_mtp: Some(true),
            ..Default::default()
        };
        let mut params = crate::engine::params::extract_chat_params(&config);
        params.enable_mtp = true;
        assert_eq!(
            planned_turn(&inner, true, false).decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
            "a sync MTP turn must still plan the flat MTP core"
        );

        let tokens = vec![1u32, 5, 9, 3];
        let plan = TurnPlan {
            is_delta: false,
            input_media: crate::engine::plan::MediaCapabilities::NONE,
            context_media: crate::engine::plan::MediaCapabilities::NONE,
            use_paged_attention: false,
            decoder: DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
        };

        for turn in 1..=2u32 {
            let mut args = WholeTurnArgs {
                tokens: &tokens,
                tokenizer: &tokenizer,
                eos_id: u32::MAX,
                config: &config,
                params: &params,
                thinking: ThinkingSetup {
                    enabled: false,
                    budget: None,
                },
                plan,
                sink: None,
                cancelled: None,
                media: MediaInputs {
                    images: &[],
                    audio: &[],
                },
            };

            let out =
                ChatBackend::run_speculative_turn(&mut inner, &mut args).unwrap_or_else(|e| {
                    panic!(
                        "turn {turn}: the seed failure must NOT propagate: {}",
                        e.reason
                    )
                });
            match out {
                TurnOutput::Complete(result) => assert_eq!(
                    result.num_tokens, 4,
                    "turn {turn}: the AR fallback must generate the full budget"
                ),
                _ => panic!("turn {turn}: expected a completed sync turn"),
            }
            assert_eq!(
                inner.mtp_seed_failures, turn,
                "turn {turn}: the fallback arm must count every failed seed"
            );
            assert!(
                inner.mtp_weights_loaded && inner.has_mtp_weights(),
                "turn {turn}: a failed seed must NOT disarm the head"
            );
            assert!(
                inner.pending_mtp_draft_seed.is_none(),
                "turn {turn}: no half-built seed may survive"
            );
            assert!(
                inner.execution_plan().speculative.is_some(),
                "turn {turn}: the speculative plan must stay on offer"
            );
            assert_eq!(
                planned_turn(&inner, true, false).decoder,
                DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
                "turn {turn}: the next MTP request must plan back into the drafter"
            );
        }
    }

    /// A turn CANCELLED mid-prefill is NOT a seeding failure: the distinguished
    /// `PREFILL_CANCELLED` error must propagate instead of being masked as a completed
    /// AR turn. Both arms of `mtp_seed_aborted` leave the head armed and the caches
    /// clean, so `mtp_seed_failures` is the only thing that tells them apart.
    ///
    /// MUTATION: routing the cancellation into `mtp_seed_failed_fallback` — the AR
    /// fallback re-polls the same flag in its own prefill and still rejects, so ONLY
    /// the `mtp_seed_failures` assertion fails.
    #[test]
    fn cancelled_prefill_seed_propagates_without_disarming_the_mtp_head() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_flat_mtp_config()).expect("inner builds");
        install_dense_moe(&mut inner).expect("install dense experts");
        inner.mtp_weights_loaded = true;
        assert!(
            inner.execution_plan().speculative.is_some(),
            "the head starts armed"
        );

        // What `session_start` does before handing the turn to the executor: install the
        // turn's cancel flag, pre-flipped so the first chunk-boundary poll aborts the seed.
        let cancelled = Arc::new(AtomicBool::new(true));
        ChatBackend::set_turn_cancel_flag(&mut inner, Some(Arc::clone(&cancelled)));

        let tokenizer = tiny_tokenizer();
        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(4),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            enable_mtp: Some(true),
            ..Default::default()
        };
        let mut params = crate::engine::params::extract_chat_params(&config);
        params.enable_mtp = true;
        assert_eq!(
            planned_turn(&inner, true, false).decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
            "a sync MTP turn must plan the flat MTP core"
        );

        let tokens = vec![1u32, 5, 9, 3];
        let plan = TurnPlan {
            is_delta: false,
            input_media: crate::engine::plan::MediaCapabilities::NONE,
            context_media: crate::engine::plan::MediaCapabilities::NONE,
            use_paged_attention: false,
            decoder: DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
        };
        let mut args = WholeTurnArgs {
            tokens: &tokens,
            tokenizer: &tokenizer,
            eos_id: u32::MAX,
            config: &config,
            params: &params,
            thinking: ThinkingSetup {
                enabled: false,
                budget: None,
            },
            plan,
            sink: None,
            cancelled: Some(cancelled.as_ref()),
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };

        let err = ChatBackend::run_speculative_turn(&mut inner, &mut args)
            .err()
            .expect("a cancelled turn must not produce a result");
        assert_eq!(
            err.reason, PREFILL_CANCELLED,
            "the cancellation must reach the session layer unmasked"
        );

        assert_eq!(
            inner.mtp_seed_failures, 0,
            "a cancellation must never be counted as a seeding failure"
        );

        ChatBackend::set_turn_cancel_flag(&mut inner, None);
        assert!(
            inner.mtp_weights_loaded,
            "a cancelled turn must leave the head armed"
        );
        assert!(inner.has_mtp_weights());
        assert!(
            inner.execution_plan().speculative.is_some(),
            "later turns on this model must still see the speculative plan"
        );
        assert_eq!(
            planned_turn(&inner, true, false).decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
            "later MTP requests must still plan the MTP core"
        );
        assert!(
            inner.pending_mtp_draft_seed.is_none(),
            "no half-built seed may survive the cancellation"
        );
        assert!(
            inner.cached_token_history.is_empty(),
            "the aborted prefill must leave no committed history behind"
        );
    }

    /// A 40-token prompt: 2 registerable 16-token blocks plus a partial tail.
    fn c1_prompt() -> Vec<u32> {
        (0..40u32).map(|i| (i % 30) + 1).collect()
    }

    /// Cold-prefill `tokens` on `seq_id` with the hash lookup forced OFF. Computed on
    /// the SAME inner — the fixture's weights are random per construction, so a
    /// second inner is not comparable.
    fn cold_prefill_fingerprint(
        inner: &mut NemotronHInner,
        seq_id: SeqId,
        tokens: &[u32],
        cache_salt: u64,
    ) -> Vec<f32> {
        let stream = Stream::default(DeviceType::Gpu);
        let prefix = inner
            .prime_prefix_state_for(seq_id, tokens, &[], cache_salt, /* skip_lookup */ true)
            .expect("cold prime");
        assert_eq!(
            prefix.effective_cached_prefix_len, 0,
            "skip_lookup must force a 0-token prefix"
        );
        inner.activate_paged_seq(seq_id).expect("activate");
        let _ = inner
            .paged_prefill(tokens, &prefix, stream)
            .expect("cold prefill");
        mamba_fingerprint(inner, seq_id)
    }

    /// THE C1 gate. A preempted sequence keeps its owner history AND its registered
    /// K/V blocks, but `handle_preempted` releases its recurrent row, so the resume
    /// must drop the prefix and cold-prefill onto a recurrent state BIT-IDENTICAL to a
    /// never-preempted cold prefill — a reconstruction is not, and that residual flips
    /// greedy argmaxes on the real checkpoint.
    #[test]
    fn preempted_resume_prefills_cold_and_matches_a_fresh_cold_prefill() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        install_dense_moe(&mut inner).expect("install dense experts");
        let stream = Stream::default(DeviceType::Gpu);
        let salt = 77u64;
        let prompt = c1_prompt();
        let replay: Vec<u32> = prompt.iter().copied().chain([7u32, 9u32]).collect();

        // Reference: the same token stream, cold, on an unrelated sequence.
        let cold_reference = cold_prefill_fingerprint(&mut inner, 2, &replay, 5001);

        // Turn 1 on seq 1.
        let prefix = inner
            .prime_prefix_state_for(1, &prompt, &[], salt, false)
            .expect("prime turn 1");
        assert_eq!(prefix.effective_cached_prefix_len, 0, "turn 1 is cold");
        inner.activate_paged_seq(1).expect("activate turn 1");
        let _ = inner
            .paged_prefill(&prompt, &prefix, stream)
            .expect("prefill turn 1");

        // Preempt EXACTLY as `handle_preempted` does: publish the full blocks
        // for reuse, release the adapter request, then drop the recurrent row.
        {
            let adapter = inner.paged_adapter.as_mut().expect("adapter");
            adapter
                .register_full_blocks_for_reuse_for(1, &[], salt, false)
                .expect("register blocks for reuse");
            adapter.release_request_for(1).expect("release request");
        }
        inner.release_scheduled_caches_for(1);

        // Resume, in the scheduler's order: activate the recurrent row FIRST,
        // then prepare the prefix.
        inner.activate_paged_seq(1).expect("activate on resume");
        assert!(
            !inner.active_seq_recurrent_survived,
            "a preempted row must re-activate as NOT survived"
        );
        let admission = HybridSchedulerBackend::prepare_scheduled_prefix(
            &mut inner, 1, &replay, &prompt, true, salt, 16,
        )
        .expect("prepare on resume");
        // `NoRestoreTicket` makes `Waiting` uninhabited, so this is
        // irrefutable: nemotron_h always admits Ready.
        let ScheduledPrefixAdmission::Ready(resumed) = admission;
        assert_eq!(
            resumed.effective_cached_prefix_len, 0,
            "a resume without a live mamba state must drop the K/V prefix"
        );
        assert!(
            !resumed.mamba_state_reusable,
            "the cold path must not claim a reusable mamba state"
        );
        assert_eq!(resumed.suffix_len, replay.len(), "the whole stream re-runs");

        inner.activate_paged_seq(1).expect("activate for prefill");
        let _ = inner
            .paged_prefill(&replay, &resumed, stream)
            .expect("resume prefill");
        assert_eq!(
            mamba_fingerprint(&mut inner, 1),
            cold_reference,
            "the resumed recurrent state must be bit-identical to a cold prefill"
        );
    }

    /// The non-regression gate the C1 fix must NOT break: a warm continuation whose
    /// recurrent state really did survive still keeps its whole K/V prefix.
    ///
    /// NOT caught here: forcing `scheduled_recurrent_state_live` to always-false — the
    /// adapter's `ContinuedLivePrefix` route ignores `skip_lookup` entirely.
    #[test]
    fn warm_continuation_keeps_its_kv_prefix_when_the_mamba_state_survived() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        install_dense_moe(&mut inner).expect("install dense experts");
        let stream = Stream::default(DeviceType::Gpu);
        let salt = 88u64;
        let prompt = c1_prompt();
        let suffix: Vec<u32> = vec![7, 9, 11];
        let tokens: Vec<u32> = prompt
            .iter()
            .copied()
            .chain(suffix.iter().copied())
            .collect();

        let prefix = inner
            .prime_prefix_state_for(1, &prompt, &[], salt, false)
            .expect("prime turn 1");
        inner.activate_paged_seq(1).expect("activate turn 1");
        let _ = inner
            .paged_prefill(&prompt, &prefix, stream)
            .expect("prefill turn 1");
        // The scheduler's success path: keep the request live, park the row.
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .finalize_turn_keep_live(&[], salt)
            .expect("finalize keep-live");
        inner.park_active_scheduled_caches();

        inner.activate_paged_seq(1).expect("activate turn 2");
        assert!(
            inner.active_seq_recurrent_survived,
            "a parked row must re-activate as survived"
        );
        let admission = HybridSchedulerBackend::prepare_scheduled_prefix(
            &mut inner, 1, &tokens, &prompt, true, salt, 16,
        )
        .expect("prepare turn 2");
        // `NoRestoreTicket` makes `Waiting` uninhabited, so this is
        // irrefutable: nemotron_h always admits Ready.
        let ScheduledPrefixAdmission::Ready(warm) = admission;
        assert_eq!(
            warm.effective_cached_prefix_len,
            prompt.len(),
            "a live-state continuation must keep its whole K/V prefix"
        );
        assert!(
            warm.mamba_state_reusable,
            "the surviving mamba state sits exactly on the prefix boundary"
        );
        assert_eq!(warm.suffix_len, suffix.len(), "only the suffix re-runs");
        assert!(
            inner.last_paged_prefill_reused_mamba_state,
            "the perf accounting must report a suffix-only prefill"
        );
    }

    /// The BACKSTOP gate. `ensure_recurrent_slot` evicts an idle recurrent row WITHOUT
    /// releasing that row's adapter request, so the next turn arrives with state=gone
    /// but request=live+already_registered. The adapter's `can_continue` never consults
    /// `skip_lookup`, so the predicate edits alone cannot stop it.
    #[test]
    fn evicted_recurrent_row_with_a_live_adapter_request_reprepares_cold() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        install_dense_moe(&mut inner).expect("install dense experts");
        let stream = Stream::default(DeviceType::Gpu);
        let salt = 99u64;
        let prompt = c1_prompt();
        let tokens: Vec<u32> = prompt.iter().copied().chain([7u32, 9, 11]).collect();

        let prefix = inner
            .prime_prefix_state_for(1, &prompt, &[], salt, false)
            .expect("prime turn 1");
        inner.activate_paged_seq(1).expect("activate turn 1");
        let _ = inner
            .paged_prefill(&prompt, &prefix, stream)
            .expect("prefill turn 1");
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .finalize_turn_keep_live(&[], salt)
            .expect("finalize keep-live");
        assert!(
            inner
                .paged_adapter
                .as_ref()
                .expect("adapter")
                .is_live_for_continue(),
            "the fixture needs a LIVE, already-registered request"
        );
        inner.park_active_scheduled_caches();

        // EXACTLY what `ensure_recurrent_slot` does to an idle victim: drop
        // the recurrent row and nothing else.
        inner.release_scheduled_caches_for(1);

        inner.activate_paged_seq(1).expect("activate turn 2");
        let admission = HybridSchedulerBackend::prepare_scheduled_prefix(
            &mut inner, 1, &tokens, &prompt, true, salt, 16,
        )
        .expect("prepare turn 2");
        // `NoRestoreTicket` makes `Waiting` uninhabited, so this is
        // irrefutable: nemotron_h always admits Ready.
        let ScheduledPrefixAdmission::Ready(after_eviction) = admission;
        assert_eq!(
            after_eviction.effective_cached_prefix_len, 0,
            "an evicted recurrent row must re-prepare cold even on a LIVE request"
        );
        assert!(!after_eviction.mamba_state_reusable);
        assert_eq!(after_eviction.suffix_len, tokens.len());
    }

    /// The invariant is fail-closed at the point of use, not merely upstream: a
    /// nonzero cached prefix with no live mamba state is a hard error, never a silent
    /// full-prefix reconstruction.
    #[test]
    fn paged_prefill_refuses_a_prefix_without_a_live_mamba_state() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        let full = c1_prompt();
        let err = inner
            .run_paged_prefill_chunk(&full, &full[8..], 8, /* skip_reconstruction */ false)
            .err()
            .expect("a prefix without live state must be refused");
        assert!(
            err.reason.contains("without a live mamba state"),
            "unexpected error: {}",
            err.reason
        );
    }

    /// `can_activate_scheduled_recurrent` must actually refuse past the cap — that
    /// refusal is the ONLY thing that makes `ensure_recurrent_slot`'s idle-victim body
    /// run. The expectation is DERIVED from `scheduler_max_num_seqs_for(32)`, never
    /// hard-coded: it reads a process-wide `OnceLock` an earlier test may have forced.
    #[test]
    fn recurrent_slot_cap_refuses_a_new_row_at_capacity() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        let cap = scheduler_max_num_seqs_for(32);
        assert!(cap >= 2, "the cap must leave room for real batching");

        for seq_id in 0..cap as SeqId {
            assert!(
                HybridSchedulerBackend::can_activate_scheduled_recurrent(&inner, seq_id),
                "seq {seq_id} is below the cap and must be admissible"
            );
            inner.activate_paged_seq(seq_id).expect("activate");
        }
        inner.park_active_scheduled_caches();
        assert_eq!(
            inner.scheduled_recurrent_units(),
            cap,
            "every admitted row must be resident"
        );

        let newcomer = cap as SeqId;
        assert!(
            !HybridSchedulerBackend::can_activate_scheduled_recurrent(&inner, newcomer),
            "a new sequence past the cap must be refused so the scheduler evicts"
        );
        // An ALREADY-resident sequence is always admissible, cap or not —
        // otherwise a running turn could not be re-activated.
        assert!(
            HybridSchedulerBackend::can_activate_scheduled_recurrent(&inner, 0),
            "a resident row must stay admissible at the cap"
        );

        // Evicting one idle row (what `ensure_recurrent_slot` does) reopens a slot.
        inner.release_scheduled_caches_for(0);
        assert_eq!(inner.scheduled_recurrent_units(), cap - 1);
        assert!(
            HybridSchedulerBackend::can_activate_scheduled_recurrent(&inner, newcomer),
            "evicting one idle row must admit the newcomer"
        );
    }

    /// `scheduled_recurrent_units` must count the ACTIVE row exactly once.
    /// Double-counting it would silently shrink the effective cap by one.
    #[test]
    fn scheduled_recurrent_units_counts_the_active_row_once() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        assert_eq!(inner.scheduled_recurrent_units(), 0, "fresh model has none");
        inner.activate_paged_seq(1).expect("activate 1");
        assert_eq!(inner.scheduled_recurrent_units(), 1);
        inner.park_active_scheduled_caches();
        assert_eq!(inner.scheduled_recurrent_units(), 1, "parked still counts");
        inner.activate_paged_seq(2).expect("activate 2");
        assert_eq!(inner.scheduled_recurrent_units(), 2);
        // Re-activating an already-parked row must MOVE it, not duplicate it.
        inner.activate_paged_seq(1).expect("re-activate 1");
        assert_eq!(inner.scheduled_recurrent_units(), 2);
        inner.activate_paged_seq(1).expect("re-activate live 1");
        assert_eq!(inner.scheduled_recurrent_units(), 2, "no-op re-activation");
    }

    /// `scheduled_recurrent_state_live` must answer for the sequence being asked
    /// about, BEFORE its activation — the only point at which
    /// `active_seq_recurrent_survived` still describes the PREVIOUS sequence.
    #[test]
    fn scheduled_recurrent_state_live_reads_the_pre_activation_truth() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = NemotronHInner::new(tiny_paged_config()).expect("inner builds");
        assert!(
            !inner.scheduled_recurrent_state_live(1),
            "an unknown sequence has no live state"
        );
        inner.activate_paged_seq(1).expect("activate 1");
        assert!(
            !inner.scheduled_recurrent_state_live(1),
            "a FRESH activation is zero-state, not survived"
        );
        inner.park_active_scheduled_caches();
        assert!(
            inner.scheduled_recurrent_state_live(1),
            "a parked row IS live for its owner"
        );
        inner.activate_paged_seq(1).expect("re-activate 1");
        assert!(
            inner.scheduled_recurrent_state_live(1),
            "the live sequence keeps the survival its activation decided"
        );
        // A different sequence's liveness must not read the active flag.
        assert!(
            !inner.scheduled_recurrent_state_live(2),
            "an unrelated sequence must not inherit the active row's survival"
        );
        inner.release_scheduled_caches_for(1);
        assert!(
            !inner.scheduled_recurrent_state_live(1),
            "a released row is not live"
        );
    }
}
