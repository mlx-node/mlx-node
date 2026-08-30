use std::cell::Cell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::time::Instant;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tracing::{info, warn};

use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, PagedBackend, PagedPrefix, ResetScope, SaveStateArgs,
    ThinkingSetup, TrainBackend, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::{
    ChatCmd, FromChatCmd, FromTrainCmd, TrainCmd, handle_chat_cmd, handle_train_cmd,
};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridSchedulerCommand, pool_tokens_after_recurrent,
    scheduled_turn_context, scheduler_per_seq_context_override,
};
use crate::engine::plan::{
    DecoderPlan, ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan, SpeculativeKind,
    SpeculativePlan,
};
use crate::engine::recurrent_state::{HYBRID_LIVE_STATE_UNITS, RecurrentStateTable};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle};
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::model_thread::{ResponseTx, send_and_await};
#[cfg(test)]
use crate::models::qwen3_5::gdn_checkpoint_store::{
    GDN_PREFIX_CHECKPOINTS_PER_OWNER, GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER,
    compute_paged_prefix_block_hash,
};
use crate::models::qwen3_5::gdn_checkpoint_store::{
    GdnCheckpointLineage, compute_paged_prefix_block_hashes,
    find_longest_valid_gdn_checkpoint_index, gdn_retention_caps, prune_gdn_checkpoints,
    replay_gdn_cache_and_commit,
};
use crate::models::qwen3_5::model::{
    IMAGE_TOKEN_ID, Qwen3_5ContextLimits, VisionCache, VisionCacheInner, async_eval_layer_caches,
    compute_image_token_counts_per_image, constrain_paged_context_params, eval_layer_caches,
    inject_image_placeholders, qwen35_expanded_prompt_token_count, vlm_prepare_vision_features,
};
use crate::models::qwen3_5::processing::Qwen35VLImageProcessor;
use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;
use crate::transformer::paged_kv_cache_adapter::SeqId;

pub(crate) type Qwen35MoeSchedulerState =
    crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35MoeInner>;

use crate::array::MxArray;
use crate::engine;
use crate::engine::backend::{MtpBackend, MtpStepper, MtpTurnSetup, SpecFrontier};
use crate::engine::spec_owner::SpecOwner;
use crate::engine::{
    apply_all_penalties, compute_performance_metrics, extract_chat_params, finalize_chat_result,
    save_cache_state_direct, verify_cache_prefix_direct,
};
use crate::models::qwen3_5::mtp_decode;
use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;
use crate::models::qwen3_5_moe::decoder_layer::DecoderLayer;
use crate::models::qwen3_5_moe::layer_cache::Qwen3_5LayerCache;
use crate::models::qwen3_5_moe::mtp::Qwen3_5MoeMTPModule;
use crate::models::qwen3_5_moe::persistence;
use crate::models::qwen3_5_moe::quantized_linear::LinearProj;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

mod chat_backend;
mod commands;
mod flat_turn;
mod forward;
mod gdn_prefix;
mod lifecycle;
mod mtp;
mod paged_backend;
mod paged_turn;
mod state;
mod training;
mod vision_turn;

#[cfg(test)]
use self::chat_backend::qwen35_moe_speculative_plan;
pub(crate) use self::commands::Qwen35MoeCmd;
pub(crate) use self::forward::PREFILL_STEP_SIZE;
use self::forward::{
    chunked_prefill, eager_verify_step, forward_inner, forward_pre_norm_inner,
    project_logits_from_hidden,
};
pub(crate) use self::mtp::Qwen35MoeDecode;
use self::paged_backend::{Qwen35MoePrefixState, StreamSender};
pub(crate) use self::state::qwen35_moe_vision_active;
use self::state::{
    MoeGdnCheckpointStoreTrace, MoeGdnHistoryCheckpoint, MoeGdnPrefixCheckpoint,
    MoeGdnPrefixPreparation, TokenPrefixMismatchTrace, apply_qwen35_moe_planned_decoder,
    clone_moe_linear_layer_caches, fresh_moe_layer_caches, moe_paged_linear_caches_ready,
    qwen35_moe_media_plan, qwen35_moe_session_media, qwen35_moe_session_media_matches_payloads,
    token_prefix_mismatch_trace,
};

// Import the shared model ID counter from the dense module — dense and MoE
// share the same C++ weight map, so IDs must be globally unique.
use crate::engine::compiled_lock::QWEN35_MODEL_ID_COUNTER;

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership of all inference
/// and training state. Training commands are routed via `TrainingDispatch`.
pub(crate) struct Qwen35MoeInner {
    pub(crate) config: Qwen3_5MoeConfig,
    /// The in-flight turn's cooperative-cancel flag, installed by the sync and
    /// streaming session wrappers via [`ChatBackend::set_turn_cancel_flag`] and
    /// cleared (`None`) in their turn epilogue on every exit path. A set flag
    /// aborts at the next chunk boundary with the distinguished
    /// `"prefill cancelled"` error, riding the engine's fail-closed
    /// prefill-`Err` arms; a single-shot (unchunked) prefill is NOT cancellable.
    pub(crate) turn_cancel: Option<Arc<AtomicBool>>,
    /// Turn-constant layer classification (`Linear` vs `FullAttentionPaged`),
    /// computed once in [`Self::new`] instead of re-derived on every paged
    /// decode step. Pure function of the immutable `config`. Mirrors the
    /// `Gemma4Inner::layer_kinds` caching pattern.
    pub(crate) layer_kinds: Vec<crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind>,
    pub(crate) embedding: Embedding,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) lm_head: Option<LinearProj>,
    pub(crate) caches: Option<Vec<Qwen3_5LayerCache>>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    pub(crate) fa_idx: usize,
    pub(crate) vision_encoder: Option<Arc<Qwen3_5VisionEncoder>>,
    pub(crate) image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    pub(crate) spatial_merge_size: Option<i32>,
    pub(crate) vision_cache: VisionCache,
    pub(crate) cached_token_history: Vec<u32>,
    pub(crate) cached_image_key: Option<u64>,
    /// Absolute expanded-token positions paired with all four per-image digest
    /// words for the live paged request. Retained across text continuations
    /// so image-conditioned blocks keep the same prefix-cache identity.
    pub(crate) cached_paged_image_token_positions: Vec<(u32, u64)>,
    pub(crate) cached_rope_deltas: Option<i32>,
    pub(crate) model_id: u64,
    /// Set when a flat eager-MTP turn stopped mid-cycle leaving `self.caches`
    /// advanced past the emitted token history (GDN state cannot be rewound).
    /// Forces the next turn to discard `self.caches` and re-prefill the full
    /// history into fresh caches. Pure-flat sessions only; the paged path
    /// rolls back its adapter directly.
    pub(crate) flat_mtp_caches_desynced: bool,
    active_cache_owner_id: String,
    gdn_root_cache_owner_id: Option<String>,
    gdn_root_cache_owner_is_explicit: bool,
    gdn_prefix_checkpoints: VecDeque<MoeGdnPrefixCheckpoint>,
    gdn_last_history_checkpoint: Option<MoeGdnHistoryCheckpoint>,
    /// Set when the infallible paged-finalize hook could not register or
    /// release the request. `save_paged_history` consumes this latch instead of
    /// republishing expanded image-placeholder history as a live session.
    paged_finalize_failed: bool,
    /// Engine-computed accepted-but-unemitted tail of the most recent paged
    /// MTP turn.
    pub(crate) paged_mtp_last_rollback_unemitted: usize,
    /// Mid-cycle GDN rewinds this session performed. Pairs with
    /// `paged_mtp_gdn_invalidations`: a rewind that fails increments the
    /// latter instead.
    paged_mtp_gdn_rewinds: u64,
    /// Failed mid-cycle GDN rewinds plus every paged epilogue frontier
    /// disagreement that armed `paged_gdn_state_dirty`.
    paged_mtp_gdn_invalidations: u64,
    /// Refuse-to-persist latch for the GDN half of a paged session: armed when
    /// a paged epilogue found the adapter and the drop-last history at
    /// different frontiers, so the live recurrent state cannot be keyed on that
    /// history. Consumed by `remember_moe_gdn_history_checkpoint` (refuses and
    /// drops the stale checkpoint) and `prepare_moe_gdn_prefix_state` (the
    /// live / last-history fast arms are skipped, so the next turn recomputes).
    /// GDN-only — the adapter's content-addressed K/V is unaffected.
    paged_gdn_state_dirty: bool,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache) for
    /// full-attention layers — same semantics as the dense model.
    /// Enabled by default for compatible checkpoints; explicit false retains
    /// the flat rollback path.
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    /// Packed affine/K-quant projections can select numerically distinct
    /// `B > 1` kernels on Metal. Those checkpoints preserve singleton
    /// projection graphs while paged attention and scheduling stay batched.
    pub(crate) row_exact_decode_projections: bool,
    /// Request-keyed GDN state for the text-only continuous scheduler lane.
    /// Full-attention K/V remains in `paged_adapter`; each entry carries only
    /// the independent recurrent arrays for one cache owner.
    scheduled_recurrent: RecurrentStateTable<Vec<Qwen3_5LayerCache>>,
    active_scheduled_seq: Option<SeqId>,
    /// Multi-Token Prediction head — `Some` when `config.n_mtp_layers > 0`
    /// (the checkpoint shipped MTP weights), `None` otherwise. Owned by
    /// the model thread; the speculative-decode loop reads it directly.
    /// Weight loading happens after construction in `apply_weights_moe_inner`.
    pub(crate) mtp: Option<Qwen3_5MoeMTPModule>,
    /// Set `true` by `apply_weights_moe_inner` ONLY after the MTP
    /// head's required weight set was found COMPLETE. Mirrors the dense
    /// `Qwen35Inner::mtp_weights_loaded`. The module itself is constructed
    /// purely from config (`n_mtp_layers > 0`), so `mtp.is_some()` alone does
    /// NOT prove the head has real weights — a partial/incompatible drafter or
    /// a truncated inline checkpoint would leave the module default-initialized.
    /// `has_mtp_weights()` AND-gates on this flag so speculative decode never
    /// runs against a half-loaded head.
    pub(crate) mtp_weights_loaded: bool,
    /// FIRST-draft acceptance rate (the per-position acceptance at draft
    /// slot 0 — depth-agnostic) of the most recently completed MTP turn,
    /// consulted by the MTP acceptance gate ([`Self::mtp_gate_allows`])
    /// when planning the NEXT turn. `None` = no MTP turn completed yet
    /// (first turn probes), the gate re-probed after
    /// [`mtp_decode::MTP_ACCEPT_GATE_REPROBE_TURNS`] gated turns, or a
    /// full session reset cleared it.
    mtp_draft_accepted: u64,
    mtp_draft_attempted: u64,
    /// Consecutive turns the MTP acceptance gate has blocked; after
    /// [`mtp_decode::MTP_ACCEPT_GATE_REPROBE_TURNS`] gated turns the gate
    /// re-probes (the aggregate resets to zero).
    mtp_gated_turns: u32,
    /// Training state owned by the model thread.
    /// Created when `InitTraining` command is received, destroyed when training ends.
    pub(crate) training_state: Option<crate::training_state::ModelThreadTrainingState>,
    /// Whether the CURRENT generic-flow turn is streaming. Set by the
    /// [`ChatBackend::profiler_label`] hook (the session core calls it
    /// exactly once per generic-flow turn, before `begin_decode`);
    /// consumed by [`ChatBackend::begin_decode`]'s
    /// profiler relabel, which must pick the `moe_chat_*` vs
    /// `moe_chat_stream_*` label family (`TurnSetup` does not carry
    /// streaming-ness). Whole-turn override paths (vision/paged/MTP)
    /// never consult it. Mirrors the dense `turn_is_streaming` field.
    turn_is_streaming: Cell<bool>,
    /// Parsed `generation_config.json` sampling/stop defaults for this
    /// checkpoint. Folded under any explicit per-request value: a request
    /// field wins, else this default applies, else the sampler's builtin.
    /// `eos_token_ids` extends the tokenizer EOS with extra stop ids.
    /// Default (empty) when the checkpoint ships no `generation_config.json`.
    gen_defaults: crate::engine::ModelGenerationDefaults,
}

/// Test-only between-turn snapshot of the MoE paged-MTP GDN bookkeeping, read
/// via [`Qwen35MoeCmd::MtpPagedGdnStateForTest`]. Serialized behind the model
/// thread, so it observes the fully-finalized preceding turn.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct MoeMtpPagedGdnStateForTest {
    /// Whether the block-paged adapter is installed (the paged MTP lane).
    pub paged_active: bool,
    /// `cached_token_history.len()` — the committed drop-last history.
    pub history_len: usize,
    /// Engine-computed accepted-but-unemitted tail of the most recent paged
    /// MTP turn.
    pub last_rollback_unemitted: usize,
    /// Mid-cycle GDN rewinds performed.
    pub gdn_rewinds: u64,
    /// GDN invalidations: failed rewinds + epilogue frontier disagreements.
    pub gdn_invalidations: u64,
    /// Whether the refuse-to-persist latch is currently armed.
    pub state_dirty: bool,
    /// Whether a GDN history checkpoint is currently stored.
    pub has_history_checkpoint: bool,
}

/// Generation configuration for Qwen3.5 MoE
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5MoeGenerationConfig {
    pub max_new_tokens: i32,
    #[napi(ts_type = "number | undefined")]
    pub temperature: Option<f64>,
    #[napi(ts_type = "number | undefined")]
    pub top_k: Option<i32>,
    #[napi(ts_type = "number | undefined")]
    pub top_p: Option<f64>,
    #[napi(ts_type = "number | undefined")]
    pub min_p: Option<f64>,
}

/// Generation result
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5MoeGenerationResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub num_tokens: u32,
    pub finish_reason: String,
}

/// Qwen3.5 MoE Model -- hybrid linear/full attention with Mixture-of-Experts.
///
/// All inference and training state lives on a dedicated OS thread. NAPI methods
/// dispatch commands via channels and await responses. Training commands are
/// routed through `TrainingDispatch` to the model thread.
#[napi]
pub struct Qwen3_5MoeModel {
    /// Dedicated model thread for inference and training.
    pub(crate) thread: crate::model_thread::ModelThread<Qwen35MoeCmd>,
    /// Cloned from inner for pure-getter NAPI methods (no command dispatch needed).
    pub(crate) config: Qwen3_5MoeConfig,
    /// Snapshot of `Qwen35MoeInner::paged_adapter.is_some()` captured at
    /// construction time. Compatible Qwen3.5 MoE checkpoints default the
    /// adapter ON; explicit false and sym8 retain the flat path. VLM
    /// checkpoints can load with the adapter on for text-only inference;
    /// image-bearing chat turns are rejected at runtime by the chat-entry sites.
    /// Surfaced through the `hasBlockPagedCache()` NAPI method.
    pub(crate) paged_active: bool,
    /// Snapshot of `Qwen35MoeInner::has_mtp_weights()` captured at
    /// construction time, mirroring `paged_active`. Surfaced through the
    /// `hasMtpWeights()` NAPI method so the TS ChatSession can auto-default
    /// `enableMtp = true` for checkpoints that ship an MTP head without
    /// round-tripping through the model thread.
    pub(crate) mtp_active: bool,
    /// Snapshot of the fully loaded image execution stack. `true` only when
    /// the vision encoder and image processor were both installed and the
    /// block-paged adapter required by MoE image turns is active. This must be
    /// derived from loaded components rather than `vision_config`, because
    /// sym8 deliberately strips its incompatible vision tower.
    pub(crate) vision_active: bool,
    /// Same loaded processor and merge-size snapshots used by the model thread,
    /// retained for exact CPU-only image-token planning before SSE.
    pub(crate) image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    pub(crate) spatial_merge_size: i32,
    pub(crate) context_limits: Qwen3_5ContextLimits,
    /// RAII: unregisters this model's baseline from the cache-limit
    /// coordinator on drop.
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
    /// RAII debit for the native paged KV pool, whose Metal buffers are not
    /// visible to MLX allocator accounting. Owned for the whole model
    /// lifetime so the coordinator entry (updated in place by the adapter's
    /// growth notifier via `update_pool`) cannot be dropped while the pool
    /// may still grow. Never read directly — retained for its `Drop`.
    #[allow(dead_code)]
    pub(crate) pool_cache_limit_guard: Option<crate::cache_limit::PoolCacheLimitGuard>,
}

#[napi]
impl Qwen3_5MoeModel {
    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Qwen35MoeInner::paged_adapter` was successfully
    /// constructed at load time (driven by
    /// `Qwen3_5MoeConfig::use_block_paged_cache`, default-ON for compatible
    /// checkpoints). On VLM checkpoints the adapter can still be active for text-only
    /// inference; image-bearing chat turns are rejected at runtime by
    /// the chat-entry sites. Surfaced through this NAPI method so
    /// server endpoints can branch on it without round-tripping through
    /// the model thread.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    /// Native admission width for plain text autoregressive turns. MTP and
    /// multimodal turns remain ordered barriers and do not enter the batched
    /// decode lane.
    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        if self.paged_active
            && Qwen35MoeSchedulerState::continuous_batching_enabled()
            && !Qwen35MoeSchedulerState::force_serial()
        {
            crate::engine::hybrid_scheduler::scheduler_max_num_seqs() as u32
        } else {
            1
        }
    }

    /// Snapshot scheduler occupancy plus unified block/recurrent admission.
    #[napi]
    pub async fn scheduler_stats(&self) -> Result<engine::SchedulerStatsJs> {
        send_and_await(&self.thread, |reply| Qwen35MoeCmd::SchedulerStats { reply }).await
    }

    /// Test-only: snapshot the paged-MTP GDN bookkeeping between turns.
    #[doc(hidden)]
    pub async fn mtp_paged_gdn_state_for_test(&self) -> Result<MoeMtpPagedGdnStateForTest> {
        send_and_await(&self.thread, |reply| {
            Qwen35MoeCmd::MtpPagedGdnStateForTest { reply }
        })
        .await
    }

    /// Test-only state oracle: recompute GDN over the persisted history
    /// checkpoint's own token key from fresh caches and bit-compare against the
    /// checkpoint. `Ok(true)` iff the persisted state equals what its key says.
    #[doc(hidden)]
    pub async fn gdn_history_checkpoint_oracle_for_test(&self) -> Result<bool> {
        send_and_await(&self.thread, |reply| {
            Qwen35MoeCmd::GdnHistoryCheckpointOracleForTest { reply }
        })
        .await
    }

    /// Whether this checkpoint shipped an MTP head (module loaded by
    /// `persistence::apply_weights_moe_inner`). Snapshotted at load time from
    /// `Qwen35MoeInner::has_mtp_weights()` so the TS `ChatSession` can
    /// auto-default `enableMtp = true` for MTP-capable checkpoints without
    /// dispatching a command into the model thread. Mirrors
    /// `Qwen3_5Model::has_mtp_weights`.
    ///
    /// Note: this only reports weight availability. Whether the
    /// speculative-decode path actually runs on a given call also requires
    /// the per-request `enableMtp` flag.
    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.mtp_active
    }

    /// Whether this loaded model instance can execute image-bearing turns.
    ///
    /// This is an authoritative load-time snapshot, not a model-family guess:
    /// it requires the loaded vision encoder, image processor, and block-paged
    /// KV adapter used by the MoE vision path.
    #[napi]
    pub fn supports_images(&self) -> bool {
        self.vision_active
    }

    /// Synchronous active-context snapshot shared with the dense wrapper.
    #[napi]
    pub fn context_limits(&self) -> Qwen3_5ContextLimits {
        self.context_limits.clone()
    }

    /// Exact, non-mutating Qwen image-placeholder expansion count for a fully
    /// rendered prompt and complete message history.
    #[napi]
    pub async fn expanded_prompt_token_count(
        &self,
        prompt_tokens: Uint32Array,
        messages: Vec<ChatMessage>,
    ) -> Result<u32> {
        qwen35_expanded_prompt_token_count(
            self.image_processor.clone(),
            self.spatial_merge_size,
            prompt_tokens,
            messages,
        )
        .await
    }

    /// Load a pretrained model from a directory.
    #[napi]
    pub async fn load(path: String) -> Result<Qwen3_5MoeModel> {
        persistence::load_with_thread(&path).await
    }

    /// Generate text from a prompt token sequence.
    #[napi]
    pub async fn generate(
        &self,
        prompt_tokens: &MxArray,
        mut config: Qwen3_5MoeGenerationConfig,
    ) -> Result<Qwen3_5MoeGenerationResult> {
        if config.max_new_tokens <= 0 {
            return Err(Error::from_reason(format!(
                "max_new_tokens must be > 0, got {}",
                config.max_new_tokens
            )));
        }
        let batch_size = prompt_tokens.shape_at(0)?;
        if batch_size != 1 {
            return Err(Error::from_reason(format!(
                "generate() only supports batch_size=1, got batch_size={}",
                batch_size
            )));
        }
        let prompt_len = prompt_tokens.shape_at(1)? as u32;
        let capacity = self.context_limits.effective_window_tokens;
        if prompt_len > capacity {
            return Err(Error::from_reason(format!(
                "context_length_exceeded: prompt has {prompt_len} tokens, effective active \
                 context is {capacity} tokens"
            )));
        }
        let max_output = capacity.saturating_sub(prompt_len).saturating_add(1);
        config.max_new_tokens = config.max_new_tokens.min(max_output as i32);
        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35MoeCmd::Generate {
            prompt_tokens: prompt_tokens.clone(),
            config,
            reply,
        })
        .await
    }

    // Test-only streaming entry points that bypass ThreadsafeFunction and hand
    // back the mpsc receiver, for `crates/mlx-core/tests/qwen3_5_moe_session.rs`.

    /// Test-only entry point that dispatches `ChatStreamSessionStart`
    /// and returns the raw mpsc receiver the model thread writes into.
    /// Callers can iterate the receiver directly rather than going
    /// through a NAPI callback.
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
            .send(Qwen35MoeCmd::Chat(ChatCmd::StreamSessionStart {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            }))?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Test-only entry point that dispatches `ChatStreamSessionContinue`
    /// and returns the raw mpsc receiver the model thread writes into.
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
            .send(Qwen35MoeCmd::Chat(ChatCmd::StreamSessionContinue {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            }))?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Get the number of parameters in the model.
    ///
    /// Pure config computation -- no model-thread dispatch needed.
    #[napi]
    pub fn num_parameters(&self) -> i64 {
        let h = self.config.hidden_size as i64;
        let v = self.config.vocab_size as i64;
        let n = self.config.num_layers as usize;
        let dense_i = self.config.intermediate_size as i64;

        let mut total = v * h;
        if !self.config.tie_word_embeddings {
            total += v * h;
        }

        let num_experts = self.config.num_experts as i64;
        let moe_i = self
            .config
            .moe_intermediate_size
            .unwrap_or(self.config.intermediate_size) as i64;
        let shared_i = self
            .config
            .shared_expert_intermediate_size
            .unwrap_or(self.config.intermediate_size) as i64;

        let kd = self.config.linear_key_dim() as i64;
        let vd = self.config.linear_value_dim() as i64;

        for layer_idx in 0..n {
            let is_linear = self.config.is_linear_layer(layer_idx);
            let is_moe = self.config.is_moe_layer(layer_idx);

            if is_linear {
                let num_vh = self.config.linear_num_value_heads as i64;
                let vhd = self.config.linear_value_head_dim as i64;
                total += h * (kd * 2 + vd * 2)
                    + h * (num_vh * 2)
                    + (kd * 2 + vd) * self.config.linear_conv_kernel_dim as i64
                    + vd * h
                    + num_vh
                    + num_vh
                    + vhd;
            } else {
                let d = self.config.head_dim as i64;
                total += h * h * 2 + h * (self.config.num_kv_heads as i64 * d) * 2 + h * h + d * 2;
            }

            if is_moe {
                total += h * num_experts + num_experts * 3 * h * moe_i + 3 * h * shared_i + h;
            } else {
                total += 3 * h * dense_i;
            }

            total += h * 2;
        }

        total += h;
        total
    }

    /// Save the model weights and configuration to a directory.
    ///
    /// Dispatches to model thread.
    #[napi]
    pub fn save_model<'env>(
        &self,
        env: &'env Env,
        save_path: String,
    ) -> Result<PromiseRaw<'env, ()>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.thread.send(Qwen35MoeCmd::SaveModel {
            save_path,
            reply: tx,
        })?;
        let promise = env.spawn_future(async move {
            rx.await
                .map_err(|_| napi::Error::from_reason("Model thread exited unexpectedly"))?
        })?;
        Ok(promise)
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: Qwen3_5MoeModel,
    thread_cmd: Qwen35MoeCmd,
    thread: direct,
    image_guard: none,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}

#[cfg(test)]
mod paged_construction_tests;

#[cfg(test)]
mod mask_free_full_attention_parity_tests;

#[cfg(test)]
mod eval_teacher_forced_tests;

#[cfg(test)]
mod paged_speculative_routing_tests;
