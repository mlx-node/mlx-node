use std::cell::Cell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::time::Instant;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tracing::{debug, info, warn};

use crate::array::MxArray;
use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, MtpBackend, MtpStepper, MtpTurnSetup, PagedBackend,
    PagedPrefix, ResetScope, SaveStateArgs, SpecFrontier, ThinkingSetup, TrainBackend, TurnOutput,
    TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::{ChatCmd, FromTrainCmd, TrainCmd, handle_chat_cmd, handle_train_cmd};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, pool_tokens_after_recurrent, scheduled_turn_context,
    scheduler_per_seq_context_override,
};
use crate::engine::plan::{
    DecoderPlan, ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan, SpeculativeKind,
    SpeculativePlan,
};
use crate::engine::recurrent_state::{HYBRID_LIVE_STATE_UNITS, RecurrentStateTable};
use crate::engine::spec_owner::SpecOwner;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::model_thread::ResponseTx;
use crate::models::qwen3_5::quantized_linear::LinearProj;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};

use crate::engine;
use crate::engine::vision::VisionMerge;
use crate::engine::{
    apply_all_penalties, compute_performance_metrics, extract_chat_params, finalize_chat_result,
    save_cache_state_direct, verify_cache_prefix_direct,
};
use crate::models::paddleocr_vl::processing::ProcessedImages;
use crate::models::qwen3_5::config::Qwen3_5Config;
use crate::models::qwen3_5::decoder_layer::DecoderLayer;
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
use crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache;
use crate::models::qwen3_5::mtp::Qwen3_5MTPModule;
use crate::models::qwen3_5::mtp_decode;
use crate::models::qwen3_5::persistence;
use crate::models::qwen3_5::processing::{Qwen35VLImageProcessor, merged_image_token_count};
use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;

// This stays a FILE, not `model/mod.rs`: promoting it re-roots every
// `#[path]`-declared child in the family. The seams live in `model/`.
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

// Facade: the names the seams publish back into this hub, so the hub, the
// cousin seams and the `#[cfg(test)]` children keep resolving them unqualified.
#[cfg(test)]
use self::chat_backend::resolve_qwen35_chat_params;
pub(crate) use self::commands::{Qwen35Cmd, handle_qwen35_cmd};
pub(crate) use self::forward::{
    PREFILL_STEP_SIZE, async_eval_layer_caches, eval_layer_caches, forward_dflash2_with_taps,
    partition_prefill_chunks,
};
use self::forward::{
    chunked_prefill, chunked_prefill_with_hidden, eager_verify_step, forward_inner,
    forward_pre_norm_inner, project_logits_from_hidden,
};
#[cfg(test)]
use self::forward::{chunked_prefill_with_hidden_with_size, chunked_prefill_with_size};
#[cfg(test)]
use self::mtp::DenseMtpStepper;
pub(crate) use self::mtp::Qwen35Decode;
use self::paged_backend::{Qwen35PrefixState, StreamSender};
#[cfg(test)]
use self::state::configure_paged_mtp_profiler;
use self::state::{
    ChatDecodeInputs, DenseGdnCheckpointStoreTrace, DenseGdnHistoryCheckpoint,
    DenseGdnPrefixCheckpoint, DenseGdnPrefixPreparation, TokenPrefixMismatchTrace,
    apply_qwen35_dense_planned_decoder, begin_paged_mtp_profiler, clone_dense_linear_layer_caches,
    dense_paged_frontier_skew, dense_paged_linear_caches_ready, fresh_dense_layer_caches,
    qwen35_dense_media_plan, qwen35_dense_session_media,
    qwen35_dense_session_media_matches_payloads, token_prefix_mismatch_trace,
};
pub(crate) use self::state::{
    arrays_bits_equal_for_test, constrain_paged_context_params, qwen35_dense_vision_active,
};
pub(crate) use self::vision_turn::{
    IMAGE_TOKEN_ID, VisionCache, VisionCacheInner, compute_image_token_counts_per_image,
    extract_images_from_messages, inject_image_placeholders, plan_expanded_image_prompt_len,
    vlm_prepare_vision_features,
};
#[cfg(test)]
use self::vision_turn::{
    VISION_CACHE_MAX_BYTES, VISION_GIB, VisionCacheMiss, VisionFeatureCacheKey, VisionImageRequest,
    VisionMemoryCapSource, VisionMemorySnapshot, expanded_image_prompt_len, get_rope_index,
    lookup_vision_feature_cache, partition_vision_cache_misses, plan_vision_image_requests,
    projected_vision_feature_bytes, resolve_vision_memory_budget,
};

pub(crate) type Qwen35SchedulerState =
    crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>;

// The shared model-id counter lives in `crate::engine::compiled_lock` so the
// dense + MoE families draw from one id space (per-instance ids never
// overlap). Re-exported here for unqualified use by this module's
// `Qwen35Inner::new`; MoE imports it from `crate::engine::compiled_lock`
// directly.
pub(crate) use crate::engine::compiled_lock::QWEN35_MODEL_ID_COUNTER;

#[cfg(test)]
mod paged_mtp_profiler_tests;

#[cfg(test)]
mod paged_context_capacity_tests;

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership of all inference
/// and training state. Training commands are routed via `TrainingDispatch`.
pub(crate) struct Qwen35Inner {
    pub(crate) config: Qwen3_5Config,
    /// The in-flight turn's cooperative-cancel flag, installed by the
    /// sync and streaming session wrappers via
    /// [`ChatBackend::set_turn_cancel_flag`] and cleared (`None`) in their
    /// turn epilogue on every exit path.
    /// Threaded into the engine AR prefill chunk loops (flat
    /// `chunked_prefill` from `ChatBackend::prefill`, paged
    /// `run_paged_prefill_chunk_with_size` from
    /// `PagedBackend::paged_prefill`); a set flag aborts at the next
    /// chunk boundary with the distinguished `"prefill cancelled"` error,
    /// riding the engine's fail-closed prefill-`Err` arms. The family's MTP,
    /// vision, hidden-state replay, and GDN replay cores thread the same flag;
    /// single-shot prefills remain the documented residual window.
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
    /// Optional external DFlash2 companion. When installed it takes
    /// precedence over the target checkpoint's inline one-layer MTP head.
    pub(crate) dflash2: Option<crate::models::qwen3_5::dflash2::DFlash2Model>,
    pub(crate) dflash2_context: Option<crate::models::qwen3_5::dflash2::DFlash2ContextCache>,
    pub(crate) dflash2_turn_state: Option<crate::models::qwen3_5::dflash2_decode::DFlash2TurnState>,
    pub(crate) caches: Option<Vec<Qwen3_5LayerCache>>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    pub(crate) vision_encoder: Option<Arc<Qwen3_5VisionEncoder>>,
    pub(crate) image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    pub(crate) spatial_merge_size: Option<i32>,
    pub(crate) vision_cache: VisionCache,
    pub(crate) cached_token_history: Vec<u32>,
    pub(crate) cached_image_key: Option<u64>,
    /// Absolute expanded-token positions paired with all four per-image digest
    /// words for the live paged request. These keys must remain attached to
    /// the image-conditioned prefix when a later text turn extends it and
    /// re-finalizes the request in the shared prefix cache.
    pub(crate) cached_paged_image_token_positions: Vec<(u32, u64)>,
    pub(crate) cached_rope_deltas: Option<i32>,
    pub(crate) model_id: u64,
    active_cache_owner_id: String,
    gdn_root_cache_owner_id: Option<String>,
    gdn_root_cache_owner_is_explicit: bool,
    gdn_prefix_checkpoints: VecDeque<DenseGdnPrefixCheckpoint>,
    gdn_last_history_checkpoint: Option<DenseGdnHistoryCheckpoint>,
    /// Set when the infallible [`PagedBackend::finalize_paged_turn`] hook could
    /// not register or release the adapter request. The engine calls
    /// `save_paged_history` immediately afterwards, so that save must consume
    /// this latch and refuse to republish placeholder-token history as a live
    /// image-derived session.
    paged_finalize_failed: bool,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache) for
    /// full-attention layers.
    ///
    /// **Opt-in via `Qwen3_5Config::use_block_paged_cache`** — see the
    /// flag's rustdoc for the full architectural rationale. When
    /// `Some(...)`, full-attention layers route through this adapter
    /// while linear-attention (GDN) layers stay on
    /// `Qwen3_5LayerCache::Linear` with no cross-request prefix reuse.
    /// Paged turns run the pure-Rust eager paged forward
    /// (`paged_forward::run_paged_prefill_chunk` / `run_paged_decode_step`).
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    /// Request-keyed GDN state for the hybrid scheduler lane. Full-attention
    /// K/V remains in `paged_adapter`; each table value contains only the
    /// corresponding request's two GDN arrays at every linear layer.
    ///
    /// The table has a two-unit cap. One state may be temporarily
    /// activated in `caches` for serial prefill/finalization, so activation
    /// first parks the previous owner and then removes the next owner from the
    /// table. Batched decode parks the active owner and stacks table rows.
    scheduled_recurrent: RecurrentStateTable<Vec<Qwen3_5LayerCache>>,
    active_scheduled_seq: Option<SeqId>,
    /// True when a paged-core turn has populated
    /// the paged adapter's `LayerKVPool` since the last flat full-attention
    /// prefill, so the flat `self.caches` full-attention slots do NOT
    /// reflect the conversation history. The streaming dense-MTP fallback
    /// (`chat_stream_tokens_delta_sync_inner`) consults this to decide
    /// whether it must rebuild the flat caches from the full history before
    /// decoding. Set by the paged cores; cleared after a flat prefill. This
    /// keeps the rebuild a ONE-TIME cost on the paged→dense transition
    /// instead of re-prefilling the whole history on every MTP turn.
    pub(crate) paged_full_attn_caches_dirty: bool,
    /// Set when a flat eager-MTP turn stopped mid-cycle leaving `self.caches`
    /// advanced past the emitted token history (GDN state cannot be rewound).
    /// Forces the next turn to discard `self.caches` and re-prefill the full
    /// history. Pure-flat sessions only; the paged path rolls back its adapter
    /// directly.
    pub(crate) flat_mtp_caches_desynced: bool,
    /// Count of full-history flat re-prefills taken by the streaming delta path
    /// because the caches were desynced (the discard+re-prefill heal at
    /// `chat_stream_tokens_delta_sync_inner`). Monotonic; lets a test confirm a
    /// continue turn actually took the heal path (the streaming chunk's
    /// `prompt_tokens`/`cached_tokens` are reported identically for heal and warm,
    /// so they can't distinguish the two).
    pub(crate) flat_full_reprefill_count: u64,
    /// Engine-observed accepted-token tail passed to the most recent flat MTP
    /// turn's `rollback_unemitted` hook. Unlike
    /// `flat_mtp_caches_desynced`, this value is computed before the family
    /// stepper handles the rollback, so tests can verify that a positive tail
    /// actually arms the family latch.
    pub(crate) flat_mtp_last_rollback_unemitted: usize,
    /// Paged twin of `flat_mtp_last_rollback_unemitted`: the engine-computed
    /// accepted-but-unemitted tail of the most recent PAGED MTP turn. The
    /// paged epilogues record it (instead of discarding the turn outcome) so
    /// tests can verify the mid-cycle GDN rewind actually had a tail to act on.
    pub(crate) paged_mtp_last_rollback_unemitted: usize,
    /// Count of paged MTP mid-cycle stops whose GDN recurrent state was
    /// tape-replayed back to the drop-last-of-emitted frontier
    /// (`DenseMtpStepper::rollback_unemitted`, Paged arm). A rewind path that
    /// silently goes dead shows up as this staying flat while
    /// `paged_mtp_gdn_invalidations` climbs.
    paged_mtp_gdn_rewinds: u64,
    /// Count of paged GDN invalidations: a failed mid-cycle GDN rewind, or an
    /// epilogue frontier disagreement that armed `paged_gdn_state_dirty`.
    paged_mtp_gdn_invalidations: u64,
    /// Refuse-to-persist latch for the paged dense GDN state. Armed when a
    /// paged epilogue's frontier check finds the adapter's recorded token
    /// count disagreeing with the drop-last history it is about to persist.
    /// While armed: `remember_dense_gdn_history_checkpoint` refuses to store
    /// (and drops the stale checkpoint), and `prepare_dense_gdn_prefix_state`
    /// skips every live/history reuse arm, falling to a recompute source
    /// (checkpoint-ladder replay, cold sidecar, or full GDN re-prefill).
    /// Cleared after a recompute arm runs and on release/reset. The heal is
    /// GDN-only — the adapter's content-addressed K/V stays live/registered.
    paged_gdn_state_dirty: bool,
    /// Test-only one-shot: force the next epilogue frontier check to report a
    /// mismatch so the `paged_gdn_state_dirty` refuse-and-heal path can be
    /// exercised deterministically (pattern of `ForceFlatMtpDesyncForTest`).
    paged_gdn_force_mismatch_for_test: bool,
    /// The arm the most recent `prepare_dense_gdn_prefix_state` resolved to
    /// (`"live"`, `"last_history"`, …). Test observability only — the trace
    /// line carrying the same value needs a tracing subscriber to see.
    last_gdn_prefix_prepare_state: &'static str,
    /// Training state owned by the model thread.
    /// Created when `InitTraining` command is received, destroyed when training ends.
    pub(crate) training_state: Option<crate::training_state::ModelThreadTrainingState>,
    /// Optional Multi-Token Prediction head.
    ///
    /// Constructed when `config.n_mtp_layers > 0`. The speculative decode loop
    /// is the only intended caller; the single-token decode path ignores this
    /// field. Weight loading is performed by `persistence::apply_weights_inner`
    /// after the main per-layer weights are loaded.
    pub(crate) mtp: Option<Qwen3_5MTPModule>,
    /// True only after persistence has seen a complete MTP tensor set and
    /// applied it to `mtp`. The module may exist from config alone; this
    /// flag prevents random-init MTP modules from advertising capability.
    pub(crate) mtp_weights_loaded: bool,
    /// Aggregated first-draft acceptance counts (accepted / attempted at
    /// draft slot 0) across completed depth-1 MTP turns, recorded by the
    /// engine's `run_mtp_turn` end-of-turn hook. The MTP acceptance gate
    /// ([`Self::mtp_gate_allows`]) disables speculation for the NEXT
    /// depth-1 turn only when a 95% confidence bound on the aggregate
    /// rate is below the break-even — so a single undersampled turn or a
    /// short bad streak from a healthy head cannot wrongly gate. `None`-
    /// equivalent (attempted == 0) = no history yet (the first turn
    /// probes), the gate re-probed after
    /// [`mtp_decode::MTP_ACCEPT_GATE_REPROBE_TURNS`] gated turns, or a
    /// full session reset cleared it.
    mtp_draft_accepted: u64,
    mtp_draft_attempted: u64,
    /// Consecutive turns the MTP acceptance gate has blocked; after
    /// [`mtp_decode::MTP_ACCEPT_GATE_REPROBE_TURNS`] gated turns the gate
    /// re-probes (the aggregate resets to zero) so a later easier turn
    /// can re-enable speculation.
    mtp_gated_turns: u32,
    /// Whether the CURRENT generic-flow turn is streaming. Set by the
    /// [`ChatBackend::profiler_label`] hook (the session core calls it
    /// exactly once per generic-flow turn, before `begin_decode`);
    /// consumed by [`ChatBackend::begin_decode`]'s
    /// profiler relabel, which must pick the `chat_*` vs `chat_stream_*`
    /// label family (`TurnSetup` does not carry streaming-ness).
    /// Whole-turn override paths (vision/paged/MTP) never consult it.
    turn_is_streaming: Cell<bool>,
    /// Sampling + stop-token defaults parsed from the checkpoint's
    /// `generation_config.json` at load time. Empty for checkpoints that
    /// ship no such file. Consumed by the [`ChatBackend`] sampling/EOS
    /// hooks and the raw `generate` loop.
    gen_defaults: crate::engine::ModelGenerationDefaults,
}

/// Test-only between-turn snapshot of the paged-MTP GDN bookkeeping, read via
/// [`Qwen35Cmd::MtpPagedGdnStateForTest`]. Serialized behind the model thread,
/// so it observes the fully-finalized preceding turn.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct MtpPagedGdnStateForTest {
    /// Whether the block-paged adapter is installed (the paged MTP lane).
    pub paged_active: bool,
    /// `cached_token_history.len()` — the committed drop-last history.
    pub history_len: usize,
    /// Engine-computed accepted-but-unemitted tail of the most recent paged
    /// MTP turn (`paged_mtp_last_rollback_unemitted`).
    pub last_rollback_unemitted: usize,
    /// Mid-cycle GDN rewinds performed (`paged_mtp_gdn_rewinds`).
    pub gdn_rewinds: u64,
    /// GDN invalidations: failed rewinds + epilogue frontier disagreements.
    pub gdn_invalidations: u64,
    /// Whether the refuse-to-persist latch is currently armed.
    pub state_dirty: bool,
    /// Whether a GDN history checkpoint is currently stored.
    pub has_history_checkpoint: bool,
    /// The arm the most recent `prepare_dense_gdn_prefix_state` resolved to.
    pub last_prefix_prepare_state: &'static str,
}

#[cfg(test)]
mod qwen35_param_resolution_tests;

/// Generation configuration for Qwen3.5
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5GenerationConfig {
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

#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct Qwen35LoadOptions {
    /// External z-lab DFlash2 checkpoint directory.
    pub draft_model_path: Option<String>,
}

/// Generation result
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5GenerationResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub num_tokens: u32,
    pub finish_reason: String,
}

/// Trained and physically available active-context limits for one loaded
/// Qwen3.5 model. Values are snapshots taken at load: `effective_window`
/// derives from the pool's MAX capacity (grow-on-demand pools are preflighted
/// against the ceiling they grow toward), while `paged_block_capacity` is the
/// physical pool size actually allocated at load and may lag after a grow.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5ContextLimits {
    pub trained_window_tokens: u32,
    pub effective_window_tokens: u32,
    pub paged_block_capacity: u32,
    pub paged_block_size: u32,
}

impl Qwen3_5ContextLimits {
    pub(crate) fn from_tuple(value: (u32, u32, u32, u32)) -> Self {
        Self {
            trained_window_tokens: value.0,
            effective_window_tokens: value.1,
            paged_block_capacity: value.2,
            paged_block_size: value.3,
        }
    }
}

// Shared chat types live in the model-neutral engine module; import them for
// internal use (no re-export — consumers import from `crate::engine::types`).
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle};

/// Qwen3.5 Model -- hybrid linear/full attention with optional MoE.
///
/// All inference and training state lives on a dedicated OS thread. NAPI methods
/// dispatch commands via channels and await responses. Training commands are
/// routed through `TrainingDispatch` to the model thread.
#[napi]
pub struct Qwen3_5Model {
    /// Dedicated model thread for inference and training.
    pub(crate) thread: crate::model_thread::ModelThread<Qwen35Cmd>,
    /// Cloned from inner for pure-getter NAPI methods (no command dispatch needed).
    pub(crate) config: Qwen3_5Config,
    /// Snapshot of `Qwen35Inner::paged_adapter.is_some()` captured at
    /// construction time. Qwen3.5 checkpoints default the adapter ON; an
    /// explicit false or a structurally unsupported sym8 checkpoint keeps the
    /// flat path. Dense image turns only run on the paged-vision core. Surfaced
    /// through the `hasBlockPagedCache()` NAPI method.
    pub(crate) paged_active: bool,
    /// Snapshot of `Qwen35Inner::has_speculative_weights()` captured
    /// at construction time, mirroring `paged_active`. Surfaced through
    /// the `hasMtpWeights()` NAPI method so the TS ChatSession can
    /// auto-default `enableMtp = true` for inline MTP or an attached DFlash2
    /// companion without round-tripping through the model thread.
    pub(crate) mtp_active: bool,
    /// Snapshot of the fully loaded image execution stack. `true` only when
    /// the vision encoder and image processor were both installed and the
    /// block-paged adapter required by the Qwen3.5 vision core is active.
    /// Config metadata alone is insufficient: sym8 checkpoints can retain a
    /// `vision_config` while deliberately dropping the incompatible vision
    /// weights at load time.
    pub(crate) vision_active: bool,
    /// Loaded image processor retained by the public wrapper for CPU-only
    /// expanded-token planning before a streaming response commits headers.
    /// This is the same `Arc` used by the model thread's real vision prefill.
    pub(crate) image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    /// Actual merge size installed on the loaded inner model, paired with
    /// `image_processor` for exact preflight geometry.
    pub(crate) spatial_merge_size: i32,
    pub(crate) context_limits: Qwen3_5ContextLimits,
    /// Directory containing tokenizer/config assets for this loaded model.
    /// For direct GGUF loads this is the lossless native-packed cache rather
    /// than the source file path.
    pub(crate) model_assets_path: String,
    /// RAII: unregisters this model's baseline from the cache-limit
    /// coordinator on drop, so the global cap can shrink once JS GCs
    /// the wrapper.
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
    /// RAII debit for the native paged KV pool. Owned for the whole model
    /// lifetime so the coordinator entry (updated in place by the adapter's
    /// growth notifier via `update_pool`) cannot be dropped while the pool
    /// may still grow. Never read directly — retained for its `Drop`.
    #[allow(dead_code)]
    pub(crate) pool_cache_limit_guard: Option<crate::cache_limit::PoolCacheLimitGuard>,
}

#[napi]
impl Qwen3_5Model {
    /// Resolved directory containing this model's tokenizer/config assets.
    /// Streaming wrappers use it after a direct GGUF load so chat templating
    /// reads the reconstructed sidecars from the native-packed cache.
    #[napi]
    pub fn model_assets_path(&self) -> String {
        self.model_assets_path.clone()
    }

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Qwen35Inner::paged_adapter` was successfully
    /// constructed at load time (driven by
    /// `Qwen3_5Config::use_block_paged_cache`, default-ON for every compatible
    /// checkpoint). On VLM checkpoints dense image turns ONLY run on the
    /// paged-vision core; a vision turn that reaches a None adapter errors at
    /// dispatch. Surfaced through this NAPI method so
    /// server endpoints can branch on it without round-tripping through
    /// the model thread.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    /// Native admission width for plain text AR turns. Installed vision and
    /// MTP modules do not disable text batching: requests that actually carry
    /// media or set `enable_mtp=true` are routed through the ordered exclusive
    /// lane by the scheduler.
    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        if self.paged_active
            && Qwen35SchedulerState::continuous_batching_enabled()
            && !Qwen35SchedulerState::force_serial()
        {
            crate::engine::hybrid_scheduler::scheduler_max_num_seqs() as u32
        } else {
            1
        }
    }

    /// Whether this instance has an inline MTP head or external DFlash2
    /// companion. Snapshotted at load time so `ChatSession` can auto-default
    /// `enableMtp = true` without dispatching into the model thread.
    ///
    /// Note: this only reports weight availability. Whether the
    /// speculative-decode path actually runs on a given call also requires the
    /// per-request `enableMtp` flag.
    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.mtp_active
    }

    /// Whether this loaded model instance can execute image-bearing turns.
    ///
    /// This is an authoritative load-time snapshot, not a `config.json`
    /// family guess: it requires a loaded vision encoder, image processor,
    /// and the block-paged KV adapter used by the dense vision path.
    #[napi]
    pub fn supports_images(&self) -> bool {
        self.vision_active
    }

    /// Synchronous snapshot used by higher layers to preflight rendered
    /// prompts and clamp output before native cache allocation.
    #[napi]
    pub fn context_limits(&self) -> Qwen3_5ContextLimits {
        self.context_limits.clone()
    }

    /// Compute the exact prompt length after Qwen image-placeholder expansion
    /// without running the vision encoder or touching inference caches.
    ///
    /// `prompt_tokens` is the already-rendered chat-template output. `messages`
    /// supplies the complete image history so both fresh and leased-session
    /// preflights account for every image in template order.
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
    ///
    /// Expects the directory to contain:
    /// - config.json
    /// - model.safetensors (or model-*.safetensors)
    /// - tokenizer.json + tokenizer_config.json
    #[napi]
    pub async fn load(path: String, options: Option<Qwen35LoadOptions>) -> Result<Qwen3_5Model> {
        let draft_model_path = options.and_then(|options| options.draft_model_path);
        let source = std::path::Path::new(&path);
        if source.is_file()
            && source
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            let cache = crate::utils::gguf::prepare_qwen35_native_gguf(source).await?;
            return persistence::load_with_thread(&cache.to_string_lossy(), draft_model_path).await;
        }
        persistence::load_with_thread(&path, draft_model_path).await
    }

    /// Generate text from a prompt token sequence.
    #[napi]
    pub async fn generate(
        &self,
        prompt_tokens: &MxArray,
        mut config: Qwen3_5GenerationConfig,
    ) -> Result<Qwen3_5GenerationResult> {
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
        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35Cmd::Generate {
            prompt_tokens: prompt_tokens.clone(),
            config,
            reply,
        })
        .await
    }

    // Test-only streaming entry points that bypass ThreadsafeFunction and hand
    // back the mpsc receiver, for `crates/mlx-core/tests/qwen3_5_delta_chat.rs`.

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
            .send(Qwen35Cmd::Chat(ChatCmd::StreamSessionStart {
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
            .send(Qwen35Cmd::Chat(ChatCmd::StreamSessionContinue {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            }))?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Test-only snapshot of the flat-MTP cache state, read *between* turns:
    /// `(committed_history_len, flat_mtp_caches_desynced,
    /// full_reprefill_count, rollback_unemitted)`.
    ///
    /// `committed_history_len` is `cached_token_history.len()` — the prompt plus
    /// the committed generation of every completed turn — i.e. exactly how many
    /// tokens a turn committed. Unlike `ChatStreamChunk.prompt_tokens` (hardcoded
    /// to the delta length on the streaming delta path, heal or warm), it is
    /// path-independent and comparable across MTP and AR turns.
    /// `flat_mtp_caches_desynced` reports whether the preceding turn stranded
    /// tokens mid-cycle and armed the heal. `full_reprefill_count` is the
    /// monotonic number of discard+re-prefill heals the streaming delta path has
    /// taken. `rollback_unemitted` is the independently engine-computed accepted
    /// tail sent to the family rollback hook, so a positive value must agree with
    /// the desync flag on flat MTP. Serialized behind the model thread, so it
    /// observes the fully-finalized preceding turn.
    #[doc(hidden)]
    pub async fn mtp_flat_state_for_test(&self) -> Result<(usize, bool, u64, usize)> {
        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35Cmd::MtpFlatStateForTest {
            reply,
        })
        .await
    }

    /// Test-only: arm the flat-MTP desync heal so the NEXT delta turn takes the
    /// discard+re-prefill path. Lets a test exercise the heal deterministically
    /// (the mid-cycle cancel that naturally arms it is host-timing-dependent).
    #[doc(hidden)]
    pub async fn force_flat_mtp_desync_for_test(&self) -> Result<()> {
        crate::model_thread::send_and_await(&self.thread, |reply| {
            Qwen35Cmd::ForceFlatMtpDesyncForTest { reply }
        })
        .await
    }

    /// Test-only between-turn snapshot of the paged-MTP GDN bookkeeping —
    /// the paged twin of [`Self::mtp_flat_state_for_test`]. See
    /// [`MtpPagedGdnStateForTest`].
    #[doc(hidden)]
    pub async fn mtp_paged_gdn_state_for_test(&self) -> Result<MtpPagedGdnStateForTest> {
        crate::model_thread::send_and_await(&self.thread, |reply| {
            Qwen35Cmd::MtpPagedGdnStateForTest { reply }
        })
        .await
    }

    /// Test-only: force the NEXT paged epilogue's frontier check to report a
    /// mismatch, arming the GDN refuse-to-persist latch deterministically.
    #[doc(hidden)]
    pub async fn force_paged_gdn_mismatch_for_test(&self) -> Result<()> {
        crate::model_thread::send_and_await(&self.thread, |reply| {
            Qwen35Cmd::ForcePagedGdnMismatchForTest { reply }
        })
        .await
    }

    /// Test-only state oracle: recompute GDN over the persisted history
    /// checkpoint's own token key from fresh caches and bit-compare against
    /// the checkpoint arrays. `Ok(true)` iff every linear layer matches.
    #[doc(hidden)]
    pub async fn gdn_history_checkpoint_oracle_for_test(&self) -> Result<bool> {
        crate::model_thread::send_and_await(&self.thread, |reply| {
            Qwen35Cmd::GdnHistoryCheckpointOracleForTest { reply }
        })
        .await
    }

    /// Get the number of parameters in the model.
    ///
    /// Pure config computation — no model-thread dispatch needed.
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

        let kd = self.config.linear_key_dim() as i64;
        let vd = self.config.linear_value_dim() as i64;

        for layer_idx in 0..n {
            let is_linear = self.config.is_linear_layer(layer_idx);
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
            total += 3 * h * dense_i;
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
        self.thread.send(Qwen35Cmd::SaveModel {
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

/// Shared dense/MoE NAPI implementation for exact, non-mutating image prompt
/// planning. Inputs are copied before entering the blocking worker so no JS
/// backing-store references cross threads.
pub(crate) async fn qwen35_expanded_prompt_token_count(
    image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    spatial_merge_size: i32,
    prompt_tokens: Uint32Array,
    messages: Vec<ChatMessage>,
) -> Result<u32> {
    let tokens = prompt_tokens.to_vec();
    let images = extract_images_from_messages(&messages);
    if images.is_empty() {
        return u32::try_from(tokens.len())
            .map_err(|_| Error::from_reason("rendered prompt token count exceeds u32"));
    }
    let image_processor = image_processor.ok_or_else(|| {
        Error::from_reason(
            "cannot plan expanded image tokens: Qwen3.5 image processor is not loaded",
        )
    })?;

    napi::bindgen_prelude::spawn_blocking(move || {
        let prompt_len =
            plan_expanded_image_prompt_len(&image_processor, spatial_merge_size, &tokens, &images)?;
        u32::try_from(prompt_len)
            .map_err(|_| Error::from_reason("expanded prompt token count exceeds u32"))
    })
    .await
    .map_err(|join_error| {
        Error::new(
            Status::GenericFailure,
            format!("Expanded prompt planning worker failed: {join_error}"),
        )
    })?
}

impl Qwen3_5Model {
    /// Test-only deterministic teardown for memory-constrained real-weight
    /// integration tests. Requires exclusive command-sender ownership and
    /// leaves production NAPI drops non-blocking.
    #[doc(hidden)]
    pub fn shutdown_for_test(mut self) -> std::result::Result<(), String> {
        self.thread.shutdown_and_join()
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: Qwen3_5Model,
    thread_cmd: Qwen35Cmd,
    thread: direct,
    image_guard: none,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}

#[cfg(test)]
mod vision_feature_cache_tests;

#[cfg(test)]
mod image_placeholder_tests;

#[cfg(test)]
mod rope_index_tests;

#[cfg(test)]
mod paged_construction_tests;

#[cfg(test)]
mod eval_teacher_forced_tests;

#[cfg(test)]
mod layer_kinds_cache_tests;

#[cfg(test)]
mod paged_gdn_frontier_tests;

#[cfg(test)]
mod mtp_gate_state_tests;
