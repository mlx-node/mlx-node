use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;

use crate::array::mask::create_causal_mask;
use crate::array::{DType, MxArray};
use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, FinalizeArgs, PagedBackend, PagedPrefix, ResetScope,
    SaveStateArgs, SpecFrontier, StreamEmitter, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::ChatCmd;
use crate::engine::params::ChatParams;
use crate::engine::plan::{
    DecoderPlan, ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan, SpeculativeKind,
    SpeculativePlan,
};
use crate::engine::spec_paged::SpecPagedCache;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::models::gemma4::quantized_linear::LinearProj;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::transformer::paged_kv_cache_adapter::{
    ColdTierContext, PagedKVCacheAdapter, paged_attention_v2_aux_fits,
};
use crate::transformer::rotating_kv_cache::RotatingKVCacheSnapshot;
use crate::transformer::{
    AttentionKind, KVCacheCoordinator, KVCacheDType, KVCacheGroup, KVCachePhysicalLayout,
    LayerKVCacheRoute, LayerKVCacheSpec, group_layer_kv_cache_specs,
};

use crate::models::gemma4::image_processor::{Gemma4ImageProcessor, ProcessedGemma4Image};
use crate::models::gemma4::vision::{Gemma4MultimodalEmbedder, Gemma4VisionModel};
use crate::models::gemma4::vision_embedder::Gemma4UnifiedVisionEmbedder;
use crate::models::gemma4::vision_mask::apply_bidirectional_vision_overlay;

use crate::engine;
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::models::gemma4::attention::{
    Gemma4PagedPrefillRoutePolicy, gemma4_paged_prefill_route_policy,
    gemma4_paged_prefill_v2_layout_for_chunk,
};
use crate::models::gemma4::config::Gemma4Config;
use crate::models::gemma4::decoder_layer::{Gemma4DecoderLayer, Gemma4LayerKind};
use crate::models::gemma4::dspark::{DsparkContextCache, DsparkTap};
use crate::models::gemma4::layer_cache::Gemma4LayerCache;
use crate::models::gemma4::sliding_sidecar;
use tracing::info;

#[path = "scheduler.rs"]
mod scheduler;
pub(crate) use scheduler::{Gemma4Cmd, Gemma4SchedulerState};

// This stays a FILE, not `model/mod.rs`: a `#[path]` on a non-inline module
// resolves against the DIRECTORY of its own source file, so promoting this to
// `model/mod.rs` would send the `scheduler` declaration above looking for
// `gemma4/model/scheduler.rs`. The seams live in `model/`.
mod backend_impl;
mod construct;
mod forward;
mod kv_cache;
mod multimodal;
mod paged_forward;
mod sliding;
mod sliding_cold;
mod stream;

// Facade: the names each seam publishes back into this hub, so the hub, the
// cousin seams, `model::scheduler` and the `#[cfg(test)]` children keep
// resolving them unqualified. `pub(crate)` entries are the ones a true sibling
// of `model` reaches by path.
use self::backend_impl::Gemma4PrefixState;
#[cfg(test)]
use self::construct::compute_layer_kinds;
pub(crate) use self::forward::{
    AssistantKvSources, Gemma4DsparkPrefillTap, PleComponents, assistant_kv_source_indices,
    assistant_verify_forward, compute_layer_kv_cache_groups, compute_layer_kv_cache_specs,
    compute_ple, dspark_shared_slot_mask, forward_body, forward_inner, lm_head_logits,
    project_paged_hidden_rows, warmup_forward,
};
use self::forward::{
    check_gemma4_repetition_cutoff, gemma4_group_reserved_blocks, init_caches_for_config,
    is_eos_token, layer_kinds_from_routes, make_sampling_config, repetition_cutoff_from_config,
    sample_next_token, validate_paged_tap_layer_ids,
};
#[cfg(test)]
use self::forward::{
    compute_layer_kinds_from_kv_cache_specs, gemma4_default_paged_cache_memory_mb,
    physical_full_attention_layer_count,
};
#[cfg(test)]
use self::kv_cache::PruneOnlySpecPagedCache;
pub(crate) use self::kv_cache::{Gemma4KVCacheCoordinator, Gemma4SpecPagedCache};
#[cfg(test)]
use self::multimodal::expand_image_tokens;
#[cfg(test)]
use self::multimodal::prompt_holds_media_placeholders;
pub(crate) use self::sliding::{
    GEMMA4_PREFILL_STEP_SIZE, eval_gemma4_caches, sliding_mask_offset_for_chunk,
};
#[cfg(test)]
use self::sliding::{
    Gemma4ColdCaptureProbe, Gemma4ColdCaptureSelection, Gemma4SlidingAnchorRungs,
    Gemma4SlidingCheckpointBytes, Gemma4SlidingColdCaptureContext, Gemma4SlidingDecodeBoundary,
    Gemma4SlidingRestoreLimitOverride, Gemma4SlidingRestoreSuppression, PrefixCacheDecision,
    classify_prefix_cache_decision, compute_gemma4_paged_prefix_block_hash,
    gemma4_chunk_cold_restore_tail, gemma4_cold_restore_reachable_boundary,
    gemma4_cold_restore_tail_publish, gemma4_large_sliding_restore_suppression_limit,
    gemma4_large_sliding_restore_suppression_limit_for_override,
    gemma4_paged_prefill_body_chunk_size, gemma4_paged_prefill_chunk_route_is_aux_safe,
    gemma4_paged_prefill_group_chunk_size, gemma4_select_cold_capture_candidate,
    gemma4_sliding_checkpoint_boundaries_crossed, gemma4_sliding_checkpoint_estimated_bytes,
    gemma4_sliding_checkpoint_estimated_bytes_at, gemma4_sliding_chunk_checkpoint_boundaries,
    gemma4_sliding_cold_anchor_rungs, gemma4_sliding_cold_capture_ceiling_blocks,
    gemma4_sliding_decode_boundary_plan, gemma4_sliding_decode_checkpoint_interval,
    gemma4_sliding_decode_publishes_checkpoint,
    gemma4_sliding_prefix_checkpoint_limit_for_override,
    gemma4_sliding_prefix_len_is_on_the_anchor_grid, gemma4_sliding_retention_caps,
    gemma4_sliding_retention_caps_for_override, gemma4_split_body_chunk_plan_at_position,
    parse_gemma4_sliding_restore_limit, trim_gemma4_sliding_prefix_checkpoints,
};
use self::sliding::{
    Gemma4GroupedSlidingColdCheckpoint, Gemma4PagedTurnPreparation,
    Gemma4SlidingCheckpointStoreTrace, Gemma4SlidingPrefixPreparation, Gemma4SlidingRetentionCaps,
    Gemma4VlmTurnPreparation, compute_gemma4_paged_prefix_block_hash_with_keys,
    create_sliding_mask, gemma4_coalesce_single_token_restore_chunks, gemma4_cold_rung_candidates,
    gemma4_paged_prefill_body_chunk_plan, gemma4_paged_prefill_group_max_chunk,
    gemma4_sliding_caches_ready_at, gemma4_sliding_cold_ladder_wanted,
    gemma4_sliding_cold_sidecar_chain_key, gemma4_sliding_retention_caps_for_cold_tier,
    gemma4_vlm_prefill_chunk_end, gemma4_vlm_prefix_policy, materialize_gemma4_sliding_snapshots,
    prefill_body_gemma4, resolve_grouped_sliding_cold_checkpoint, restore_gemma4_sliding_caches,
    snapshot_gemma4_sliding_caches, upsert_gemma4_sliding_prefix_checkpoint,
};
pub(crate) use self::stream::Gemma4Draft;
use self::stream::{
    Gemma4Emitter, Gemma4OwnerMetadata, Gemma4SchedulerOwnerState, Gemma4StreamDispatchState,
    StreamSender, gemma4_carries_image_lineage, gemma4_image_path_loaded, gemma4_media_continuable,
    gemma4_media_plan, gemma4_session_media_matches_payloads, promote_channel_only_output,
};

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership.
pub(crate) struct Gemma4Inner {
    pub(crate) config: Gemma4Config,
    /// The in-flight turn's cooperative-cancel flag, installed by the
    /// sync and streaming session wrappers via
    /// [`ChatBackend::set_turn_cancel_flag`] and cleared (`None`) in their
    /// turn epilogue on every exit path.
    /// Polled at the top of each prefill chunk (flat `prefill_body_gemma4`
    /// and the paged chunk-plan loop in `run_paged_prefill_chunk`);
    /// `true` aborts with the distinguished `"prefill cancelled"` error,
    /// riding the engine's fail-closed prefill-`Err` arms. Speculative
    /// assistant/DSpark and vision paths thread the same flag; single-shot
    /// prefills remain the documented residual window.
    pub(crate) turn_cancel: Option<Arc<AtomicBool>>,
    pub(crate) embed_tokens: Embedding,
    pub(crate) layers: Vec<Gemma4DecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) lm_head: Option<LinearProj>,
    /// Pre-transposed embedding weight for tied lm_head: [hidden_size, vocab_size].
    /// Only populated when tie_word_embeddings=true.
    pub(crate) embed_weight_t: Option<MxArray>,
    pub(crate) ple: Option<PleComponents>,
    // Vision components (None for text-only models)
    pub(crate) vision_tower: Option<Gemma4VisionModel>,
    /// Encoder-free unified vision embedder. `Some` only for the unified
    /// multimodal checkpoint (`unified_vision_config.is_some()`); mutually
    /// exclusive with `vision_tower` (the SigLIP path).
    pub(crate) unified_vision_embedder: Option<Gemma4UnifiedVisionEmbedder>,
    pub(crate) embed_vision: Option<Gemma4MultimodalEmbedder>,
    /// Encoder-free unified AUDIO embedder. `Some` only when the checkpoint
    /// declares an `audio_config` (`config.has_audio`). Structurally identical
    /// to `embed_vision` (RMSNormNoScale + Linear), but projects raw
    /// 640-sample audio windows (`audio_embed_dim` → `hidden_size`).
    pub(crate) embed_audio: Option<Gemma4MultimodalEmbedder>,
    pub(crate) image_processor: Option<Gemma4ImageProcessor>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    /// Lazily-initialized KV caches that persist across chat turns.
    ///
    /// `None` after construction and after `reset_caches_sync`. Populated
    /// by `init_caches_sync`, triggered on the first turn of a session by
    /// the engine's miss-path `reset_caches(ResetScope::PrefixMiss)` (or
    /// defensively inside [`ChatBackend::prefill`] / the vision cores).
    /// Shared across turns by the session API.
    pub(crate) caches: Option<Vec<Gemma4LayerCache>>,
    /// Selects the request-local flat target cache lane used by the ASSISTANT
    /// drafter.
    ///
    /// A loaded draft may coexist with the resident paged pools. The
    /// scheduler installs exactly one owner's flat caches here while an
    /// exclusive assistant-draft command runs, then parks them again. Every
    /// other command — AR, media, and DSpark — leaves this false and uses
    /// `active_paged_seq`.
    active_flat_session: bool,
    /// Tokens (post image-expansion) whose KV state is currently live in
    /// `caches`. Empty when no session is active.
    pub(crate) cached_token_history: Vec<u32>,
    /// Content hash of the live cache's image set; a change mid-session forces a
    /// full restart. Preserved with the ordered image-position sidecar across
    /// warm text saves so later registrations keep the same image-aware lineage.
    pub(crate) cached_image_key: Option<u64>,
    /// Content hash of the audio set associated with the live cache. Audio
    /// counterpart of `cached_image_key`: set after an audio prefill so a
    /// follow-up text delta is rejected (the continue path is text-only) and
    /// a follow-up audio turn cold-restarts. Like the image key, this is
    /// cleared after a warm text save even though the live media KV remains;
    /// `media_session_context` is the persistent source of truth.
    pub(crate) cached_audio_key: Option<u64>,
    /// Ordered absolute image-placeholder positions paired with all four words
    /// of their SHA-256 image digest for the media lineage currently represented
    /// by the live/persisted paged request. Text continuations preserve this
    /// sidecar so every later registration uses the same image-aware per-block
    /// keys instead of republishing image K/V under token-only hashes.
    cached_paged_image_token_positions: Vec<(u32, u64)>,
    /// Runtime KV-cache group coordinator (vLLM-style grouped ownership).
    ///
    /// **Opt-in via `Gemma4Config::use_block_paged_cache`**. Gemma4's
    /// hybrid sliding+global attention, K=V sharing, KV-shared layers
    /// (`forward_shared`), MoE/PLE branches, and per-layer-type head
    /// dimensions are all handled by
    /// `Gemma4DecoderLayer::forward_paged_or_flat`, which routes only
    /// global attention layers through this adapter. Defaults to `None`
    /// when the config flag is unset, in which case the model falls
    /// back to the flat `Gemma4LayerCache` path.
    pub(crate) kv_cache_coordinator: Option<Gemma4KVCacheCoordinator>,
    /// Sequence selected by the scheduler while model-neutral paged and media
    /// hooks run. Ownerless legacy turns use sequence zero.
    pub(crate) active_paged_seq: u32,
    /// Draft model for speculative decoding (`Gemma4LoadOptions::
    /// draft_model_path`), either [`Gemma4Draft`] variant. An assistant draft's
    /// target caches use a per-owner flat lane; DSpark verifies against the
    /// resident grouped paged pools, like every ordinary text/media owner.
    pub(crate) draft: Option<Gemma4Draft>,
    /// Per-turn draft handoff: the whole-turn core builds the variant's
    /// prefill-derived state (DSpark fused-context cache / assistant
    /// last-prompt hidden) during prefill and stashes it here;
    /// `DsparkBackend::begin_dspark_decode` TAKES it into the turn's
    /// stepper. Always `None`
    /// outside a live draft whole-turn.
    pub(crate) draft_turn_state: Option<crate::models::gemma4::dspark_decode::Gemma4DraftTurnState>,
    /// `compute_layer_kinds_from_kv_cache_specs(&config)`, computed once in
    /// `Gemma4Inner::new`. Pure function of the immutable `config`, so it never
    /// changes for the lifetime of this instance. Empty when
    /// `paged_adapter` is `None`: every paged-only call site that reads it
    /// errors out on a `None` adapter before consuming the value.
    pub(crate) layer_kinds: Vec<Gemma4LayerKind>,
    sliding_prefix_checkpoints: VecDeque<Gemma4SlidingPrefixCheckpoint>,
    grouped_sliding_cold_checkpoints: HashMap<u32, VecDeque<Gemma4GroupedSlidingColdCheckpoint>>,
    /// Media kinds causally represented by the current session's live/persisted
    /// prefix. This survives every successful warm text continuation because
    /// those turns extend — rather than replace — the media-derived KV. Cleared
    /// when that session is reset, invalidated, or successfully replaced.
    media_session_context: MediaCapabilities,
    /// Context handed to the currently executing generic paged text turn.
    /// `run_paged_turn` snapshots `TurnPlan::context_media` here so
    /// `save_paged_history` can distinguish a warm media continuation from a
    /// fresh text replacement without widening the model-neutral trait.
    paged_text_turn_context: MediaCapabilities,
    /// True only while a pure image turn left its
    /// global paged KV live AND a sliding history checkpoint remembered at the
    /// full kept-live prefix, so a full-history continuation can reuse
    /// the live media KV causally. Set exclusively by
    /// `finalize_vision_turn_media_state` on the continuable branch; reset to
    /// `false` at every non-continuable point (`clear_reuse_state`, both vision
    /// prefill-start blocks, the non-continuable finalize).
    media_session_continuable: bool,
    /// `PagedBackend::finalize_paged_turn` is infallible at the trait seam, but
    /// Gemma's per-block registration is not. Latch a failure here so the
    /// immediately-following fallible `save_paged_history` refuses to publish
    /// token/sliding history and lets the engine reset the failed session.
    paged_finalize_failed: bool,
    /// True when this turn's rendered prompt ends inside
    /// `<|channel>thought\n`. The generated suffix then begins at the
    /// reasoning body, so both sync and streaming output parsers must start
    /// in `Channel` rather than `Message`. Every render entry point overwrites
    /// the latch before decode; the dedicated model thread serializes turns.
    output_starts_in_reasoning_channel: AtomicBool,
    pub(crate) model_id: u64,
}

/// Gemma 4 dense language model.
///
/// Supports E2B (2.3B), E4B (4.5B), and 31B variants.
/// Features: hybrid attention (sliding + global), GeGLU MLP, logit softcapping,
/// embedding scaling, and optional per-layer embeddings.
///
/// All model state lives on a dedicated OS thread. NAPI methods dispatch
/// commands via channels and await responses.
#[napi]
pub struct Gemma4Model {
    /// Dedicated model thread owning `Gemma4Inner`. `None` when the model
    /// was constructed via `new(config)` without loading weights — in that
    /// uninitialized state every session method returns an error and
    /// only `isInitialized` is meaningful. Mirrors the same `Option<..>`
    /// gate used by the OCR models (`VLModel`, `QianfanOCRModel`).
    ///
    /// Gemma4 is chat-only (no training/generate variants). Its family command
    /// enum adds scheduler telemetry to the model-neutral [`ChatCmd`] surface,
    /// and `Gemma4SchedulerState` owns admission, preemption, and exclusive
    /// draft/media barriers on this thread.
    pub(crate) thread: Option<crate::model_thread::ModelThread<Gemma4Cmd>>,
    pub(crate) model_id: u64,
    /// Whether the loaded config includes `vision_config`. Mirrored here so
    /// the NAPI side can fail fast on image inputs to a text-only model
    /// without round-tripping to the model thread. The actual image
    /// processor lives on `Gemma4Inner` and runs on the model thread.
    pub(crate) has_vision: bool,
    /// Whether the loaded config declares an `audio_config` (unified Gemma 4
    /// audio support, `Gemma4Config::has_audio`). Mirrored here so the NAPI
    /// image-guard can fail fast on audio inputs to a model with no audio
    /// support without round-tripping to the model thread.
    pub(crate) has_audio: bool,
    /// Whether the model was loaded with real weights. `false` for
    /// `new Gemma4Model(config)` calls that never called `load()`.
    /// Session methods check this and refuse to dispatch when false,
    /// since the coordinator was never told about this model's delta
    /// (its guard is `None`) — running inference on that stub would
    /// under-cap the allocator.
    pub(crate) initialized: bool,
    /// Snapshot of the grouped paged coordinator captured at construction.
    /// Gemma4 defaults this on; explicit `use_block_paged_cache: false`
    /// retains the flat single-owner path. Stubs from `new(config)` report
    /// false because no inner was constructed. Surfaced through
    /// `hasBlockPagedCache()` for server-side capacity selection.
    pub(crate) paged_active: bool,
    /// Native scheduler width after intersecting the configured hybrid pool
    /// capacity with the process scheduler cap.
    pub(crate) max_concurrent_sequences: u32,
    /// RAII: unregisters this model's delta from the cache-limit
    /// coordinator on drop. `None` for instances constructed via the
    /// synchronous `new(config)` path that never loaded weights.
    pub(crate) _cache_limit_guard: Option<crate::cache_limit::CacheLimitGuard>,
    /// RAII registration for the grouped paged-KV pools, whose private Metal
    /// buffers are outside MLX allocator accounting but consume the same
    /// unified-memory budget. `None` for flat or uninitialized models.
    pub(crate) _pool_cache_limit_guard: Option<crate::cache_limit::PoolCacheLimitGuard>,
    /// Snapshot of `Gemma4Inner::draft.is_some()` captured at load time
    /// (same mirroring pattern as `paged_active`): whether a draft model —
    /// either [`Gemma4Draft`] variant — was loaded via
    /// `Gemma4LoadOptions::draft_model_path` or discovered in the target's
    /// `draft/` directory. Surfaced through the
    /// `hasMtpWeights()` NAPI method (named for parity with the Qwen3.5
    /// surface) so server endpoints can branch without a model-thread
    /// roundtrip. Stubs from `new(config)` always report `false`.
    pub(crate) draft_active: bool,
}

/// Optional load-time settings for [`Gemma4Model::load`].
#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct Gemma4LoadOptions {
    /// Directory of a draft checkpoint (config.json + safetensors) to load
    /// alongside the target model for speculative decoding — either a
    /// DSpark draft or a Google assistant draft; the kind is probed from
    /// the draft config.json. When omitted, `<model_path>/draft/` is loaded
    /// automatically when present. Draft decoding uses a request-local flat
    /// target-cache lane. The resident target retains its grouped paged pools,
    /// so loading an optional proposer does not disable ordinary batching.
    pub draft_model_path: Option<String>,
}

static MODEL_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

const GEMMA4_SLIDING_PREFIX_CHECKPOINT_MIN_LIMIT: usize = 16;
const GEMMA4_SLIDING_PREFIX_CHECKPOINT_WINDOW_MULTIPLIER: usize = 2;
const GEMMA4_SLIDING_PREFIX_CHECKPOINT_MAX_DEFAULT_LIMIT: usize = 128;
const GEMMA4_IMAGE_PREFIX_CHECKPOINT_LIMIT: usize = 2;
const GEMMA4_SLIDING_CHECKPOINT_MEMORY_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;
const GEMMA4_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB: u32 = 256;
const GEMMA4_PAGED_CACHE_DEFAULT_MEMORY_MB: u32 = 2048;
const GEMMA4_PAGED_DEFAULT_MAX_SEQUENCES: u32 = 8;
const BYTES_PER_MIB: u64 = 1024 * 1024;

/// Spacing of the cold-sidecar anchor rungs, as a multiple of the paged block
/// size: `block_size * RATIO^k`, ascending.
///
/// Anchored at ZERO, not at the prompt end — this is the one place gemma4 must
/// differ from qwen3.5's `gdn_prefill_checkpoint_boundaries`, which walks
/// `deepest / 4^k` DOWN from the prompt. A rung's cold key is the block chain
/// over `tokens[0..b]`, so a grid pinned to 0 makes the SAME sidecar object
/// reusable by every later turn (and every later process) whose prompt shares
/// that prefix. A prompt-anchored ladder would land on 112/496/2032 for one
/// prompt and 128/512/2048 for the next and never dedup — and `mlx agent`,
/// which is what this whole path exists for, sends a slightly different prompt
/// every turn.
const GEMMA4_SLIDING_ANCHOR_RATIO: u32 = 4;

/// How many anchor rungs a ladder may hold, before the byte budget below
/// trims it further. Mirrors `GDN_CHECKPOINT_LADDER_RUNGS`.
///
/// With `block_size = 16` this is `{64, 256, 1024, 4096}` — two rungs BELOW
/// gemma4's 1024-token window (where a payload is `min(b, window)` rows and so
/// nearly free) and two at or above it (a full window each). Without the cap a
/// small checkpoint whose full-window payload is a few MiB would keep admitting
/// rungs until the budget ran out, hundreds of them.
const GEMMA4_SLIDING_ANCHOR_MAX_RUNGS: usize = 4;

/// Byte ceiling for the WHOLE retained set on the ladder arm — the anchors plus
/// the pre-ladder reserve — at the same conservative 4 bytes/element
/// [`gemma4_sliding_checkpoint_estimated_bytes`] uses.
///
/// The 4 bytes/element figure is deliberately conservative: the snapshot type
/// promises no dtype, so the estimate must not assume bf16.
const GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES: u64 = 3 * 1024 * 1024 * 1024;

#[derive(Clone)]
struct Gemma4SlidingPrefixCheckpoint {
    prefix_len: u32,
    block_size: u32,
    final_block_hash: u64,
    protected_image_prompt_boundary: bool,
    /// This entry sits on a [`gemma4_sliding_cold_anchor_rungs`] rung, i.e. it
    /// was published FOR the cold sidecar rather than for the warm in-process
    /// path. Retention under [`Gemma4SlidingRetentionPolicy::Ladder`] evicts
    /// non-anchors first, because an anchor is the only kind of entry a cold
    /// capture can use while the persisted K/V chain still lags the prompt.
    ///
    /// Always `false` on the pre-ladder arm: nothing publishes a rung there,
    /// and `PreLadder` never reads this flag.
    cold_anchor_rung: bool,
    tokens: Vec<u32>,
    snapshots: Vec<Option<RotatingKVCacheSnapshot>>,
}

/// Everything a publishing site honestly knows about a checkpoint it just
/// produced. `cold_anchor_rung` is deliberately NOT among those fields.
///
/// A genuine rung born with the flag clear is the ladder's PREFERRED eviction
/// victim — the exact "born then evicted" failure the ladder exists to fix.
///
/// A draft cannot be pushed into the store, and
/// [`Gemma4SlidingPrefixCheckpointDraft::into_checkpoint`] is the only place in
/// the file that writes the flag, so a publish site cannot mis-mark a rung.
#[derive(Clone)]
struct Gemma4SlidingPrefixCheckpointDraft {
    prefix_len: u32,
    block_size: u32,
    final_block_hash: u64,
    protected_image_prompt_boundary: bool,
    tokens: Vec<u32>,
    snapshots: Vec<Option<RotatingKVCacheSnapshot>>,
}

impl Gemma4SlidingPrefixCheckpointDraft {
    /// Derive the stored entry. The grid that decided this boundary was
    /// published is the same grid that decides retention defers it, so the two
    /// readings come from one `caps` value.
    fn into_checkpoint(self, caps: Gemma4SlidingRetentionCaps) -> Gemma4SlidingPrefixCheckpoint {
        Gemma4SlidingPrefixCheckpoint {
            cold_anchor_rung: caps.wants_ladder() && caps.anchors.contains(self.prefix_len),
            prefix_len: self.prefix_len,
            block_size: self.block_size,
            final_block_hash: self.final_block_hash,
            protected_image_prompt_boundary: self.protected_image_prompt_boundary,
            tokens: self.tokens,
            snapshots: self.snapshots,
        }
    }
}

struct Gemma4SlidingPrefixCheckpointHit {
    prefix_len: u32,
    caches: Vec<Gemma4LayerCache>,
}

#[napi]
impl Gemma4Model {
    /// Create an uninitialized `Gemma4Model` stub from a config.
    ///
    /// **Prefer [`Gemma4Model::load`]** for any real usage — `new(config)`
    /// is a config-only stub that matches the OCR-model pattern
    /// (`VLModel::new(config)`, `QianfanOCRModel::new(config)`) and is
    /// intentionally NOT runnable: the cache-limit coordinator's per-model delta is
    /// registered exclusively on the `load()` path.
    ///
    /// This path does NOT spawn a model thread, NOT materialize any
    /// weights, and NOT register with the cache-limit coordinator. The
    /// returned instance is only useful for config inspection — every
    /// session method (`chatSessionStart` / `chatSessionContinue` /
    /// `chatSessionContinueTool` and their streaming variants) rejects
    /// with a `napi::Error` whose message is exactly
    /// `"Model not initialized. Call Gemma4Model.load() first."` until
    /// `load()` runs and installs the underlying model thread. The
    /// async `resetCaches()` call resolves as a silent no-op on the
    /// stub to keep `ChatSession.reset()` idempotent across both
    /// runnable and stub instances.
    ///
    /// A runnable model requires `await Gemma4Model.load(path)`. The constructor
    /// signature is fixed by NAPI-RS.
    #[napi(constructor)]
    pub fn new(config: Gemma4Config) -> Self {
        let has_vision = config.vision_config.is_some() || config.unified_vision_config.is_some();
        let has_audio = config.has_audio;
        Self {
            thread: None,
            model_id: 0,
            has_vision,
            has_audio,
            initialized: false,
            paged_active: false,
            max_concurrent_sequences: 1,
            _cache_limit_guard: None,
            _pool_cache_limit_guard: None,
            draft_active: false,
        }
    }

    /// Returns true if weights have been loaded via `load()`.
    #[napi(getter)]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Gemma4Inner::paged_adapter` was successfully
    /// constructed at load time (driven by
    /// `Gemma4Config::use_block_paged_cache`). Stubs constructed via
    /// `new(config)` always return `false`. Surfaced
    /// through this NAPI method so server endpoints can branch on it
    /// without a model-thread roundtrip.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        if self.paged_active && !Gemma4SchedulerState::force_serial() {
            self.max_concurrent_sequences.max(1)
        } else {
            1
        }
    }

    #[napi]
    pub async fn scheduler_stats(&self) -> Result<engine::SchedulerStatsJs> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call Gemma4Model.load() first.")
        })?;
        crate::model_thread::send_and_await(thread, |reply| Gemma4Cmd::SchedulerStats { reply })
            .await
    }

    /// Whether this loaded instance can execute image-bearing chat turns.
    /// Config-only stubs and incomplete/non-paged physical paths return false.
    #[napi]
    pub fn supports_images(&self) -> bool {
        self.initialized && self.paged_active && self.has_vision
    }

    #[napi]
    pub fn model_id(&self) -> u32 {
        self.model_id as u32
    }

    /// Whether a draft model — DSpark or Google assistant — is loaded on
    /// this instance (via `Gemma4LoadOptions::draft_model_path`), enabling
    /// the speculative-decode whole-turn path.
    ///
    /// Note: this only reports draft availability. Whether speculative
    /// decoding actually runs on a given call also requires the per-request
    /// `enableMtp` flag. Named `hasMtpWeights` for parity with the Qwen3.5
    /// surface, but it reports an external draft model (either variant),
    /// not in-checkpoint MTP heads. Stubs from `new(config)` always return
    /// `false`.
    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.draft_active
    }

    /// Load a Gemma4 model from a directory.
    #[napi]
    pub async fn load(
        model_path: String,
        options: Option<Gemma4LoadOptions>,
    ) -> Result<Gemma4Model> {
        Self::load_from_dir(&model_path, options).await
    }

    /// Test-only entry point that dispatches `ChatCmd::StreamSessionStart`
    /// and returns the raw mpsc receiver the model thread writes into, so a
    /// pure-Rust integration test can exercise the streaming path without a
    /// NAPI host (same pattern as `Qwen3_5Model::chat_stream_session_start_for_test`).
    #[doc(hidden)]
    pub fn chat_stream_session_start_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        crate::engine::types::ChatStreamHandle,
        tokio::sync::mpsc::Receiver<Result<ChatStreamChunk>>,
    )> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call Gemma4Model.load() first.")
        })?;
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) = crate::model_thread::stream_channel(
            crate::engine::napi_glue::CHAT_STREAM_NATIVE_QUEUE_LIMIT,
        );
        thread.send(Gemma4Cmd::Chat(Box::new(ChatCmd::StreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })))?;
        Ok((
            crate::engine::types::ChatStreamHandle { cancelled },
            stream_rx,
        ))
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: Gemma4Model,
    thread_cmd: crate::models::gemma4::model::Gemma4Cmd,
    thread: { option: "Model not initialized. Call Gemma4Model.load() first." },
    image_guard: { vision: has_vision, audio: has_audio },
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod prefix_cache_decision_tests;

#[cfg(test)]
mod flat_verify_tests;

#[cfg(test)]
mod spec_paged_substrate_tests;

#[cfg(test)]
mod assistant_seam_tests;

#[cfg(test)]
mod reasoning_close_tag_seam_tests;
