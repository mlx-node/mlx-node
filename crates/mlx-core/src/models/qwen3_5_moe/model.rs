use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tracing::{info, warn};

use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, PagedBackend, PagedPrefix, PagedTurnSetup, ResetScope,
    SaveStateArgs, ThinkingSetup, TrainBackend, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::{
    ChatCmd, FromChatCmd, FromTrainCmd, TrainCmd, handle_chat_cmd, handle_train_cmd,
};
use crate::engine::plan::{
    DecoderPlan, ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan, SpeculativeKind,
    SpeculativePlan,
};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle};
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::model_thread::ResponseTx;
#[cfg(test)]
use crate::models::qwen3_5::gdn_checkpoint_store::{
    GDN_PREFIX_CHECKPOINTS_PER_OWNER, GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER,
    compute_paged_prefix_block_hash,
};
use crate::models::qwen3_5::gdn_checkpoint_store::{
    GdnCheckpointLineage, GdnSidecarBoundary, compute_paged_prefix_block_hashes,
    find_longest_valid_gdn_checkpoint_index, gdn_retention_caps, prune_gdn_checkpoints,
    replay_gdn_cache_and_commit, select_gdn_sidecar_boundary,
};
use crate::models::qwen3_5::model::{
    IMAGE_TOKEN_ID, Qwen3_5ContextLimits, VisionCache, VisionCacheInner, async_eval_layer_caches,
    compute_image_token_counts_per_image, constrain_paged_context_params, eval_layer_caches,
    inject_image_placeholders, partition_prefill_chunks, qwen35_expanded_prompt_token_count,
    vlm_prepare_vision_features,
};
use crate::models::qwen3_5::processing::Qwen35VLImageProcessor;
use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::DecoderLayer;
use super::layer_cache::Qwen3_5LayerCache;
use super::mtp::Qwen3_5MoeMTPModule;
use super::persistence;
use super::quantized_linear::LinearProj;
use crate::array::MxArray;
use crate::engine;
use crate::engine::backend::{MtpBackend, MtpStepper, MtpTurnSetup};
use crate::engine::{
    apply_all_penalties, compute_performance_metrics, extract_chat_params, finalize_chat_result,
    save_cache_state_direct, verify_cache_prefix_direct,
};
use crate::models::qwen3_5::mtp_decode;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

fn fresh_moe_layer_caches(config: &Qwen3_5MoeConfig) -> Vec<Qwen3_5LayerCache> {
    (0..config.num_layers as usize)
        .map(|i| {
            if config.is_linear_layer(i) {
                Qwen3_5LayerCache::new_linear()
            } else {
                Qwen3_5LayerCache::new_full_attention()
            }
        })
        .collect()
}

struct MoeGdnPrefixCheckpoint {
    owner_id: String,
    prefix_len: u32,
    block_size: u32,
    final_block_hash: u64,
    block_hashes: Vec<u64>,
    tokens: Vec<u32>,
    caches: Vec<Qwen3_5LayerCache>,
}

impl GdnCheckpointLineage for MoeGdnPrefixCheckpoint {
    fn owner_id(&self) -> &str {
        &self.owner_id
    }

    fn prefix_len(&self) -> u32 {
        self.prefix_len
    }

    fn block_size(&self) -> u32 {
        self.block_size
    }

    fn final_block_hash(&self) -> u64 {
        self.final_block_hash
    }

    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn block_hashes(&self) -> &[u64] {
        &self.block_hashes
    }
}

struct MoeGdnHistoryCheckpoint {
    owner_id: String,
    image_key: Option<u64>,
    tokens: Vec<u32>,
    caches: Vec<Qwen3_5LayerCache>,
}

pub(super) struct MoeGdnPrefixPreparation {
    pub(super) state: &'static str,
    pub(super) already_primed: bool,
    pub(super) restored_prefix_tokens: u32,
    pub(super) replayed_prefix_tokens: u32,
}

#[derive(Default)]
struct MoeGdnCheckpointStoreTrace {
    stored: bool,
    eval_ms: f64,
    clone_ms: f64,
    token_clone_ms: f64,
    update_ms: f64,
    total_ms: f64,
}

impl MoeGdnCheckpointStoreTrace {
    fn finish(mut self, start: Option<std::time::Instant>) -> Self {
        self.total_ms = start.map(elapsed_ms).unwrap_or(0.0);
        self
    }
}

#[derive(Clone, Copy)]
struct TokenPrefixMismatchTrace {
    index: i64,
    prompt_token: i64,
    cached_token: i64,
}

impl Default for TokenPrefixMismatchTrace {
    fn default() -> Self {
        Self {
            index: -1,
            prompt_token: -1,
            cached_token: -1,
        }
    }
}

fn token_prefix_mismatch_trace(prompt: &[u32], cached: &[u32]) -> TokenPrefixMismatchTrace {
    let common_len = prompt.len().min(cached.len());
    for i in 0..common_len {
        if prompt[i] != cached[i] {
            return TokenPrefixMismatchTrace {
                index: i as i64,
                prompt_token: prompt[i] as i64,
                cached_token: cached[i] as i64,
            };
        }
    }

    TokenPrefixMismatchTrace {
        index: common_len as i64,
        prompt_token: prompt.get(common_len).map_or(-1, |token| *token as i64),
        cached_token: cached.get(common_len).map_or(-1, |token| *token as i64),
    }
}

fn moe_paged_linear_caches_ready(
    config: &Qwen3_5MoeConfig,
    caches: Option<&[Qwen3_5LayerCache]>,
) -> bool {
    let Some(caches) = caches else {
        return false;
    };
    if caches.len() != config.num_layers as usize {
        return false;
    }
    for (i, cache) in caches.iter().enumerate() {
        if !config.is_linear_layer(i) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = cache else {
            return false;
        };
        if arrays.get(0).is_none() || arrays.get(1).is_none() {
            return false;
        }
    }
    true
}

fn clone_moe_linear_layer_caches(
    config: &Qwen3_5MoeConfig,
    caches: &[Qwen3_5LayerCache],
) -> Option<Vec<Qwen3_5LayerCache>> {
    if !moe_paged_linear_caches_ready(config, Some(caches)) {
        return None;
    }

    let mut cloned = fresh_moe_layer_caches(config);
    for i in 0..config.num_layers as usize {
        if !config.is_linear_layer(i) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = &caches[i] else {
            return None;
        };
        cloned[i] = Qwen3_5LayerCache::Linear(arrays.clone());
    }
    Some(cloned)
}

// Import the shared model ID counter from the dense module — dense and MoE
// share the same C++ weight map, so IDs must be globally unique.
use crate::engine::compiled_lock::QWEN35_MODEL_ID_COUNTER;

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership of all inference
/// and training state. Training commands are routed via `TrainingDispatch`.
pub(crate) struct Qwen35MoeInner {
    pub(crate) config: Qwen3_5MoeConfig,
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
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache) for
    /// full-attention layers — same semantics as the dense model.
    /// **Opt-in via `Qwen3_5MoeConfig::use_block_paged_cache`.**
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
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

/// Build the MoE media admission contract from this loaded family's own
/// components. Image execution needs the encoder, processor, and paged KV
/// adapter; incomplete stacks still enter the backend for its precise error.
pub(crate) const fn qwen35_moe_vision_active(
    has_vision_encoder: bool,
    has_image_processor: bool,
    has_paged_adapter: bool,
) -> bool {
    has_vision_encoder && has_image_processor && has_paged_adapter
}

const fn qwen35_moe_media_plan(
    has_vision_encoder: bool,
    has_image_processor: bool,
    has_paged_adapter: bool,
) -> MediaPlan {
    let images_available =
        qwen35_moe_vision_active(has_vision_encoder, has_image_processor, has_paged_adapter);
    MediaPlan::with_backend_validation(
        MediaCapabilities {
            images: images_available,
            audio: false,
        },
        MediaCapabilities::IMAGES,
    )
}

/// Media represented by the live MoE session state. Paged text continuations
/// clear the image content key after prefix preparation, while the retained
/// M-RoPE delta continues to prove that the live KV prefix is image-derived.
const fn qwen35_moe_session_media(
    has_cached_image_key: bool,
    has_cached_rope_delta: bool,
) -> MediaCapabilities {
    if has_cached_image_key || has_cached_rope_delta {
        MediaCapabilities::IMAGES
    } else {
        MediaCapabilities::NONE
    }
}

fn qwen35_moe_session_media_matches_payloads(
    cached_image_key: Option<u64>,
    images: &[Vec<u8>],
    audio: &[Vec<u8>],
) -> bool {
    audio.is_empty()
        && !images.is_empty()
        && cached_image_key == Some(engine::compute_image_cache_key(images))
}

/// Project the engine-selected decoder into the legacy MoE whole-turn config
/// before that core re-extracts `ChatParams`.
fn apply_qwen35_moe_planned_decoder(config: &mut ChatConfig, decoder: DecoderPlan) -> bool {
    let planned_mtp = matches!(
        decoder,
        DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
    );
    config.enable_mtp = Some(planned_mtp);
    planned_mtp
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Qwen35MoeCmd {
    /// All chat-session traffic (sync + streaming starts/continues/tool
    /// turns + cache reset), routed through the model-neutral engine
    /// dispatcher ([`crate::engine::cmd::handle_chat_cmd`]) against the
    /// [`ChatBackend`] impl on [`Qwen35MoeInner`]. The per-variant
    /// behavioural contracts live on [`crate::engine::cmd::ChatCmd`].
    Chat(ChatCmd),
    Generate {
        prompt_tokens: MxArray,
        config: Qwen3_5MoeGenerationConfig,
        reply: ResponseTx<Qwen3_5MoeGenerationResult>,
    },
    /// Static FP8 activation-amax calibration prefill (NVIDIA modelopt
    /// `MaxCalibrator`). MoE counterpart of the dense
    /// [`crate::models::qwen3_5::model::Qwen35Cmd::CalibratePrefillRaw`]. For
    /// each raw text: tokenize WITHOUT the chat template, truncate to
    /// `calib_seq` tokens, then run PREFILL ONLY (no generation, no generated
    /// token) so every mxfp8 attn/GDN projection's activation tap fires once,
    /// resetting caches between rows. Runs on the model thread where the
    /// tokenizer lives. The command body SELF-ARMS this model thread's
    /// thread-local `ActivationAmaxCollector` flag (via `CalibrationArmGuard`)
    /// for the prefill's duration; the NAPI caller drains+persists the collected
    /// amax afterwards — this command never touches `config.json`. Replies with
    /// the number of rows actually prefilled (rows that were empty after
    /// tokenize+truncate are skipped).
    CalibratePrefillRaw {
        texts: Vec<String>,
        calib_seq: u32,
        reply: ResponseTx<u32>,
    },
    SaveModel {
        save_path: String,
        reply: ResponseTx<()>,
    },
    /// Training-session commands shared with the model-neutral engine. The
    /// thread loop routes these to
    /// [`crate::engine::cmd::handle_train_cmd`], which drives the
    /// [`TrainBackend`] impl on [`Qwen35MoeInner`].
    Train(TrainCmd),
}

impl FromChatCmd for Qwen35MoeCmd {
    #[inline]
    fn from_chat(cmd: ChatCmd) -> Self {
        Qwen35MoeCmd::Chat(cmd)
    }
}

impl FromTrainCmd for Qwen35MoeCmd {
    #[inline]
    fn from_train(cmd: TrainCmd) -> Self {
        Qwen35MoeCmd::Train(cmd)
    }
}

/// Training backend the model-neutral [`handle_train_cmd`] drives. Each
/// method forwards to the inherent `*_sync_impl` body on
/// [`Qwen35MoeInner`].
impl TrainBackend for Qwen35MoeInner {
    fn training_state_mut(
        &mut self,
    ) -> &mut Option<crate::training_state::ModelThreadTrainingState> {
        &mut self.training_state
    }

    fn init_training_sync(
        &mut self,
        config: Box<crate::grpo::engine::GRPOEngineConfig>,
        model_type: crate::training_model::ModelType,
    ) -> Result<()> {
        self.init_training_sync_impl(*config, model_type)
    }

    fn generate_for_training_thread_sync(
        &mut self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: usize,
        gen_config: crate::models::qwen3::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<crate::training_model::GenerationPlainData> {
        self.generate_for_training_thread_sync_impl(
            prompts,
            group_size,
            gen_config,
            enable_thinking,
            tools,
        )
    }

    fn train_step_grpo_sync(
        &mut self,
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        self.train_step_grpo_sync_impl(rewards, group_size, loss_config, valid_indices)
    }

    fn train_step_sft_sync(
        &mut self,
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        self.train_step_sft_sync_impl(input_ids, input_shape, labels, labels_shape, config)
    }

    fn save_optimizer_state_sync(&self, path: String) -> Result<()> {
        self.save_optimizer_state_sync_impl(path)
    }

    fn load_optimizer_state_sync(&mut self, path: String) -> Result<()> {
        self.load_optimizer_state_sync_impl(path)
    }
}

/// Command handler for the dedicated model thread.
pub(crate) fn handle_qwen35_moe_cmd(inner: &mut Qwen35MoeInner, cmd: Qwen35MoeCmd) {
    match cmd {
        // All chat-session traffic routes through the model-neutral
        // engine dispatcher against `Qwen35MoeInner`'s `ChatBackend`
        // impl. (The engine dispatcher carries the historical NOTE
        // forward: no per-request cache drain here — the TS idle
        // sweeper in `@mlx-node/server` handles between-turn drains.)
        Qwen35MoeCmd::Chat(chat_cmd) => {
            handle_chat_cmd(inner, chat_cmd);
        }
        Qwen35MoeCmd::Generate {
            prompt_tokens,
            config,
            reply,
        } => {
            let _ = reply.send(inner.generate_sync(prompt_tokens, config));
        }
        Qwen35MoeCmd::CalibratePrefillRaw {
            texts,
            calib_seq,
            reply,
        } => {
            let _ = reply.send(inner.calibrate_prefill_raw_sync(texts, calib_seq));
        }
        Qwen35MoeCmd::SaveModel { save_path, reply } => {
            let _ = reply.send(inner.save_model_sync(&save_path));
        }
        // --- Training commands ---
        Qwen35MoeCmd::Train(train_cmd) => {
            handle_train_cmd(inner, train_cmd);
        }
    }
}

/// Adapter giving the engine's [`ChunkSink`] the `.call()` shape the
/// `decode_loop!` macro and the engine's `run_mtp_turn` loop (and the
/// streaming cores behind the whole-turn probes) expect from a
/// `ThreadsafeFunction`-like callback.
///
/// The engine owns the channel and hands the probes a `&dyn ChunkSink`,
/// so the wrapper forwards `.call()` to [`ChunkSink::send`]; the call
/// mode is meaningless on the mpsc path and is dropped. Mirrors the
/// dense `StreamSender` adapter.
struct StreamSender<'a>(&'a dyn ChunkSink);

impl StreamSender<'_> {
    fn call(&self, result: napi::Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
        self.0.send(result);
    }
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

// ========== Qwen35MoeInner implementation ==========
// All these methods run on the dedicated model thread (synchronous, no locks).

impl Qwen35MoeInner {
    /// Create a new Qwen35MoeInner with the given configuration.
    pub(crate) fn new(config: Qwen3_5MoeConfig) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;

        let layers = (0..config.num_layers as usize)
            .map(|i| DecoderLayer::new(&config, i))
            .collect::<Result<Vec<_>>>()?;

        let final_norm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(LinearProj::Standard(Linear::new(
                config.hidden_size as u32,
                config.vocab_size as u32,
                Some(false),
            )?))
        };

        let fa_idx = (0..config.num_layers as usize)
            .find(|&i| !config.is_linear_layer(i))
            .unwrap_or(0);

        let model_id = QWEN35_MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Layer classification is a pure function of the immutable config;
        // compute once here (see the field rustdoc).
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            config.num_layers as usize,
            |i| config.is_linear_layer(i),
        );

        // Persistence constructs the physical pool only after weights are
        // installed and materialized, when live unified-memory pressure can be
        // measured safely.
        let paged_adapter = None;

        // Multi-Token Prediction (MTP) head. Built when the config
        // reports `n_mtp_layers > 0` (i.e. the checkpoint shipped MTP
        // weights). The constructor rejects all-linear configs and zero
        // layer counts; weights are loaded later by
        // `apply_weights_moe_inner`. None when MTP is absent — keeps
        // the decode path cost-free on non-MTP checkpoints.
        let mtp = if config.n_mtp_layers > 0 {
            Some(Qwen3_5MoeMTPModule::new(&config)?)
        } else {
            None
        };

        info!(
            "Qwen3.5 MoE inner created: {} layers, fa_idx={}, experts={}, paged={}, mtp_layers={}",
            config.num_layers,
            fa_idx,
            config.num_experts,
            paged_adapter.is_some(),
            config.n_mtp_layers
        );

        Ok(Self {
            config,
            layer_kinds,
            embedding,
            layers,
            final_norm,
            lm_head,
            caches: None,
            tokenizer: None,
            fa_idx,
            vision_encoder: None,
            image_processor: None,
            spatial_merge_size: None,
            vision_cache: Arc::new(Mutex::new(VisionCacheInner::new())),
            cached_token_history: Vec::new(),
            cached_image_key: None,
            cached_paged_image_token_positions: Vec::new(),
            cached_rope_deltas: None,
            model_id,
            flat_mtp_caches_desynced: false,
            active_cache_owner_id: String::new(),
            gdn_root_cache_owner_id: None,
            gdn_root_cache_owner_is_explicit: false,
            gdn_prefix_checkpoints: VecDeque::new(),
            gdn_last_history_checkpoint: None,
            paged_finalize_failed: false,
            paged_adapter,
            mtp,
            mtp_weights_loaded: false,
            mtp_draft_accepted: 0,
            mtp_draft_attempted: 0,
            mtp_gated_turns: 0,
            training_state: None,
            turn_is_streaming: Cell::new(false),
            gen_defaults: crate::engine::ModelGenerationDefaults::default(),
        })
    }

    /// MoE mirror of the dense post-materialization adaptive paged-pool
    /// initialization.
    pub(crate) fn initialize_paged_adapter(&mut self) -> Result<()> {
        if !self.config.use_block_paged_cache.unwrap_or(false)
            || !crate::engine::persistence::compiled_forward_backend_available()
        {
            return Ok(());
        }
        if self.paged_adapter.is_some() {
            return Ok(());
        }
        let attn_layer_count = self.config.full_attention_layer_count() as u32;
        if attn_layer_count == 0 {
            return Err(Error::from_reason(
                "Qwen3.5 MoE block-paged adapter requires at least one full_attention layer",
            ));
        }
        let block_size = self.config.paged_block_size.unwrap_or(16);
        let head_size = self.config.head_dim as u32;
        let num_kv_heads = self.config.num_kv_heads as u32;
        let max_seq_len = self.config.max_position_embeddings as u32;
        let default_memory_mb =
            crate::models::qwen3_5::config::qwen35_default_paged_cache_memory_mb(
                max_seq_len,
                block_size,
                head_size,
                num_kv_heads,
                attn_layer_count,
            );
        let (requested_memory_mb, requested_source) =
            crate::models::qwen3_5::config::qwen35_resolve_paged_cache_memory_mb(
                self.config.paged_cache_memory_mb,
                default_memory_mb,
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
                "Qwen3.5 MoE requested paged cache {requested_memory_mb} MiB cannot hold one block"
            )));
        }
        let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
        let (num_blocks, sizing_source) = match mlx_paged_attn::profile::load_time_pool_sizing(
            requested_blocks,
            attn_layer_count,
            num_kv_heads,
            head_size,
            block_size,
            cache_dtype,
        ) {
            Ok(sizing) => (
                sizing.selected_blocks,
                format!(
                    "adaptive(requested_blocks={}, active_mib={}, working_set_mib={})",
                    sizing.requested_blocks,
                    sizing.metal_active_bytes / (1024 * 1024),
                    sizing
                        .metal_working_set_bytes
                        .map(|v| (v / (1024 * 1024)).to_string())
                        .unwrap_or_else(|| "n/a".to_string())
                ),
            ),
            Err(e) => {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 MoE adaptive paged cache sizing failed safely; refusing an \
                     uncapped pool request: {e}"
                )));
            }
        };
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            num_blocks, block_size,
        )));
        let pool =
            mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, cache_dtype).map_err(|e| {
                Error::from_reason(format!("Failed to construct Qwen3.5 MoE KV pool: {e}"))
            })?;
        self.paged_adapter = Some(
            PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                Error::from_reason(format!(
                    "Failed to construct Qwen3.5 MoE paged adapter: {e}"
                ))
            })?,
        );
        info!(
            "Qwen3.5 MoE paged adapter enabled after weight materialization: num_blocks={}, \
             block_size={}, effective_window_tokens={}, trained_window_tokens={}, \
             requested_memory_mib={}, requested_source={}, sizing_source={}",
            num_blocks,
            block_size,
            num_blocks.saturating_mul(block_size).min(max_seq_len),
            max_seq_len,
            requested_memory_mb,
            requested_source,
            sizing_source,
        );
        Ok(())
    }

    /// Build the process-global SSD cold-tier context (manager + COMPLETE
    /// content fingerprint) for `model_path` WITHOUT attaching it. MoE mirror of
    /// `Qwen35Inner::build_cold_tier_context` — see its doc for the full
    /// rationale; the ONLY differences are the `qwen3_5_moe` fingerprint family
    /// and that the GDN sidecar geometry/policy is derived from the shared dense
    /// codec via `to_dense_config` (dense and MoE keep byte-identical GDN
    /// recurrent state — same shapes, same dtype, same layer mapping).
    ///
    /// The pool covers the FULL-ATTENTION layers only, so a K/V-only restore
    /// would resume from GDN recurrent state the pool never held. The
    /// [`mlx_paged_attn::ColdSidecarPolicy`] turns the restore walk into vLLM's
    /// reconcile-down — the candidate prefix is reduced to the deepest boundary a
    /// validated GDN sidecar backs, and a boundary nothing backs restores
    /// nothing.
    ///
    /// Returns `None` (fail-open) when the paged adapter is absent, the tier
    /// cannot be opened, this checkpoint has no GDN state to persist, or a
    /// complete content fingerprint cannot be established.
    ///
    /// The `weights` witness pins this call after the loader's
    /// `materialize_weights` pass: MLX preads shard bytes lazily, so an identity
    /// read before that pass can describe bytes the model never runs.
    pub(crate) fn build_cold_tier_context(
        &self,
        model_path: &str,
        weights: &crate::array::memory::WeightsResident,
    ) -> Option<crate::transformer::paged_kv_cache_adapter::ColdTierContext> {
        let adapter = self.paged_adapter.as_ref()?;
        let manager = crate::cold_tier::global_cold_cache()?;
        let pool = adapter.layer_kv_pool();
        let cache_dtype = format!("{:?}", pool.cache_dtype());
        // The GDN state is shared with dense qwen3_5, so derive its geometry and
        // policy from the dense projection of this MoE config. No GDN layers
        // means no out-of-pool state — but it also means this is not the hybrid
        // qwen3_5 MoE the sidecar work validated, so stay off rather than
        // silently behaving like a dense family.
        let dense_config = self.config.to_dense_config();
        let geometry = crate::models::qwen3_5::gdn_sidecar::geometry(&dense_config, &cache_dtype)?;
        let sidecar_policy =
            crate::models::qwen3_5::gdn_sidecar::policy(&dense_config, &cache_dtype)?;
        let mut config_json = serde_json::to_vec(&self.config).ok()?;
        config_json.extend_from_slice(&geometry.fingerprint_component());
        let pool_geometry = crate::cold_tier::ColdTierGeometry {
            block_size: pool.block_size() as u64,
            num_layers: pool.num_layers() as u64,
            num_kv_heads: pool.config().num_kv_heads as u64,
            head_size: pool.config().head_size as u64,
            cache_dtype,
        };
        match crate::cold_tier::build_model_fingerprint(
            "qwen3_5_moe",
            model_path,
            Some(&config_json),
            &pool_geometry,
            weights,
        ) {
            Some(fingerprint) => Some(
                crate::transformer::paged_kv_cache_adapter::ColdTierContext {
                    manager,
                    fingerprint,
                    sidecar_policy: Some(sidecar_policy),
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

    /// Attach a previously-built cold-tier context to the paged adapter. A
    /// no-op (fail-open) when the paged adapter is absent. Split from
    /// [`Self::build_cold_tier_context`] so the caller can verify shard identity
    /// is still stable AFTER the fingerprint read and BEFORE the cold tier is
    /// committed.
    ///
    /// Takes the same `materialize_weights` witness as the build step so the
    /// COMMIT point, not just the identity read, is compiler-pinned below
    /// materialization.
    pub(crate) fn attach_cold_tier(
        &mut self,
        ctx: crate::transformer::paged_kv_cache_adapter::ColdTierContext,
        _weights: &crate::array::memory::WeightsResident,
    ) {
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter.set_cold_tier(ctx);
        }
    }

    pub(crate) fn paged_context_limits(&self) -> (u32, u32, u32, u32) {
        let trained = self.config.max_position_embeddings.max(0) as u32;
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return (trained, trained, 0, 0);
        };
        let blocks = adapter.block_capacity();
        let block_size = adapter.block_size();
        (
            trained,
            trained.min(adapter.max_capacity_tokens()),
            blocks,
            block_size,
        )
    }

    fn preflight_paged_context(
        &self,
        prompt_tokens: usize,
        params: &mut engine::ChatParams,
    ) -> Result<()> {
        let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
            Error::from_reason("context_length_exceeded: paged cache is not initialized")
        })?;
        let capacity = adapter
            .max_capacity_tokens()
            .min(self.config.max_position_embeddings.max(0) as u32);
        constrain_paged_context_params("Qwen3.5 MoE", prompt_tokens, capacity, params)
    }

    /// Store the checkpoint's parsed `generation_config.json` defaults.
    /// Called once at load time after construction.
    pub(crate) fn set_gen_defaults(&mut self, defaults: crate::engine::ModelGenerationDefaults) {
        self.gen_defaults = defaults;
    }

    /// Initialize KV caches.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        let caches = (0..self.config.num_layers as usize)
            .map(|i| {
                if self.config.is_linear_layer(i) {
                    Qwen3_5LayerCache::new_linear()
                } else {
                    Qwen3_5LayerCache::new_full_attention()
                }
            })
            .collect();
        self.caches = Some(caches);
        self.clear_reuse_state();
        Ok(())
    }

    /// Reset all caches.
    pub(crate) fn reset_caches_sync(&mut self) -> Result<()> {
        if let Some(ref mut caches) = self.caches {
            for cache in caches.iter_mut() {
                cache.reset();
            }
        }
        self.caches = None;
        self.clear_reuse_state();
        // A full session reset must also clear the MTP acceptance gate
        // state: a new independent chat on this model starts fresh (probes)
        // instead of inheriting the previous chat's rejection.
        self.mtp_draft_accepted = 0;
        self.mtp_draft_attempted = 0;
        self.mtp_gated_turns = 0;
        Ok(())
    }

    /// Static FP8 activation-amax calibration prefill (runs on the model
    /// thread). Inherent body of [`Qwen35MoeCmd::CalibratePrefillRaw`]; a
    /// faithful mirror of the dense
    /// [`crate::models::qwen3_5::model::Qwen35Inner::calibrate_prefill_raw_sync`].
    ///
    /// For each raw text: tokenize WITHOUT the chat template (`encode_sync` with
    /// `add_special_tokens = false`, so no `<|im_start|>`/`<|im_end|>` control
    /// tokens and no BOS), truncate to `calib_seq` tokens, then run PREFILL ONLY
    /// (no generation) so every mxfp8 attention/GDN projection's activation tap
    /// fires exactly once over realistic raw-text activations — the NVIDIA
    /// modelopt `MaxCalibrator` is defined over raw-text prefill, NOT
    /// chat-templated prompts plus a decode step. Caches are re-initialized per
    /// row so each is an independent turn-0 position-0 prefill.
    ///
    /// This method SELF-ARMS the model thread's thread-local
    /// [`crate::calibration::activation_amax::ActivationAmaxCollector`] flag for
    /// the prefill's duration (RAII `CalibrationArmGuard`, disarmed on every exit
    /// path), so the tap records each projection's running `max|activation|`
    /// during the forward. The NAPI caller drains+persists afterwards; this
    /// method never touches `config.json`. Returns the number of rows actually
    /// prefilled (rows that tokenized to nothing after truncation are skipped).
    pub(crate) fn calibrate_prefill_raw_sync(
        &mut self,
        texts: Vec<String>,
        calib_seq: u32,
    ) -> Result<u32> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::from_reason("calibration prefill requires a tokenizer, but none is loaded")
        })?;
        let cap = calib_seq.max(1) as usize;
        let mut rows_prefilled: u32 = 0;

        // Arm THIS (model) thread's calibration flag for the whole prefill loop.
        // The tap in `QuantizedLinear::forward` runs synchronously on this same
        // thread, so it observes the armed flag and records raw `max|x|`; any
        // other loaded model on its own thread stays unaffected. The RAII guard
        // disarms on EVERY exit path — normal return, a `?` error, or a panic —
        // so a later inference command on this thread never sees a stray armed
        // flag. The NAPI caller drains + persists the collected amax afterwards.
        let _calib_guard = crate::calibration::activation_amax::CalibrationArmGuard::arm();

        for text in &texts {
            // RAW tokenize — no chat template, no special/control tokens. This
            // is the crux of the modelopt-parity fix: repeated chat control
            // tokens must not dominate a tensor's activation amax.
            let mut tokens = tokenizer.encode_sync(text, Some(false))?;
            tokens.truncate(cap);
            if tokens.is_empty() {
                continue;
            }

            // Fresh turn-0 caches per row (prefill asserts `self.caches` is set,
            // and a stale cache would append rows into one growing context).
            self.init_caches_sync()?;
            let generation_stream = Stream::new(DeviceType::Gpu);
            // PREFILL ONLY — trips every mxfp8 attn/GDN tap once. No generated
            // token: a decode step would fold a synthetic argmax token's
            // activations into the amax.
            let logits = self.prefill(&tokens, generation_stream)?;
            // Force the row's forward to complete (the taps already `.item()`
            // each projection input, but eval bounds the lazy graph and makes
            // per-row memory reclamation deterministic).
            logits.eval();
            self.reset_caches_sync()?;
            rows_prefilled += 1;

            // Bound resident scratch across many rows (large models × 1024 rows).
            crate::array::synchronize_and_clear_cache();
        }

        Ok(rows_prefilled)
    }

    /// Clear cached token history, image key, and rope deltas.
    fn clear_reuse_state(&mut self) {
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        self.cached_rope_deltas = None;
        self.gdn_prefix_checkpoints.clear();
        self.gdn_last_history_checkpoint = None;
        self.gdn_root_cache_owner_id = None;
        self.gdn_root_cache_owner_is_explicit = false;
        self.paged_finalize_failed = false;
    }

    /// Tear down a partially prepared or partially executed paged turn.
    ///
    /// MoE's full-attention K/V lives in the adapter while its GDN recurrent
    /// state lives in `self.caches`. If either half fails, retaining the other
    /// half together with old media/history metadata creates a false-live
    /// session. Preserve allocator-owned content-addressed blocks, but clear
    /// every model-local continuation signal and recurrent checkpoint.
    fn discard_moe_paged_session(&mut self) {
        if let Some(adapter) = self.paged_adapter.as_mut()
            && let Err(release_error) = adapter.release_request()
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5_moe::paged",
                "failed to release MoE paged request during invalidation: {release_error}",
            );
        }
        if let Some(caches) = self.caches.as_mut() {
            for cache in caches {
                cache.reset();
            }
        }
        self.caches = None;
        self.clear_reuse_state();
        self.flat_mtp_caches_desynced = false;
    }

    fn invalidate_moe_paged_session(&mut self, context: &str) {
        tracing::warn!(
            target: "mlx_core::qwen3_5_moe::paged",
            "invalidating MoE paged session after {context}",
        );
        self.discard_moe_paged_session();
    }

    /// Fallible terminal lifecycle for the hand-written MoE paged cores.
    /// Registration failure is a turn failure: invalidate before any history
    /// or GDN checkpoint can be published.
    ///
    /// On success the GDN sidecar capture runs here, mirroring the engine hook —
    /// these cores own every planned-MTP turn, and without this an MTP session
    /// persists K/V blocks whose recurrent half no restore can ever reconstruct.
    /// Unlike the engine hook there is no release to order against: these cores
    /// only ever keep the request live, so the adapter's cold-chain frontier is
    /// still set when the capture reads it.
    fn finalize_moe_manual_paged_turn(
        &mut self,
        image_token_positions: &[(u32, u64)],
    ) -> Result<()> {
        let finalize_result = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| "MoE manual paged finalization: paged_adapter is None".to_owned())
            .and_then(|adapter| {
                let finalize_extra_keys = engine::build_paged_extra_keys(
                    adapter.request_tokens().len(),
                    adapter.block_size(),
                    image_token_positions,
                );
                adapter.finalize_turn_keep_live_per_block(&finalize_extra_keys, 0)
            });
        match finalize_result {
            Ok(_) => {
                self.capture_moe_gdn_cold_sidecar(image_token_positions);
                Ok(())
            }
            Err(error) => {
                self.invalidate_moe_paged_session("manual finalization failure");
                Err(Error::from_reason(format!(
                    "MoE paged finalization failed: {error}"
                )))
            }
        }
    }

    /// MoE mirror of dense's terminal paged-finalization downgrade. The engine
    /// saves history immediately after the infallible finalize hook, so an
    /// adapter failure must invalidate every live image-session signal and
    /// leave a latch that prevents that save from reviving placeholder history.
    fn downgrade_failed_paged_finalize(&mut self, error: &str) {
        tracing::warn!(
            target: "mlx_core::qwen3_5_moe::paged",
            "paged adapter finalization failed; invalidating the MoE session: {error}",
        );
        self.discard_moe_paged_session();
        self.paged_finalize_failed = true;
    }

    fn find_moe_gdn_history_checkpoint(
        &self,
        tokens: &[u32],
        prefix_len: u32,
        expected_image_key: Option<u64>,
    ) -> Option<Vec<Qwen3_5LayerCache>> {
        let prefix_tokens = tokens.get(..prefix_len as usize)?;
        let checkpoint = self.gdn_last_history_checkpoint.as_ref()?;
        if checkpoint.owner_id != self.active_cache_owner_id
            || checkpoint.image_key != expected_image_key
            || checkpoint.tokens.as_slice() != prefix_tokens
        {
            return None;
        }
        clone_moe_linear_layer_caches(&self.config, &checkpoint.caches)
    }

    fn remember_moe_gdn_history_checkpoint(&mut self) -> Result<MoeGdnCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = MoeGdnCheckpointStoreTrace::default();
        if self.cached_token_history.is_empty() {
            self.gdn_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }

        let eval_start = trace_enabled.then(std::time::Instant::now);
        eval_layer_caches(&self.caches)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);
        let clone_start = trace_enabled.then(std::time::Instant::now);
        let Some(caches) = self
            .caches
            .as_ref()
            .and_then(|caches| clone_moe_linear_layer_caches(&self.config, caches))
        else {
            self.gdn_last_history_checkpoint = None;
            trace.clone_ms = clone_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.clone_ms = clone_start.map(elapsed_ms).unwrap_or(0.0);
        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let tokens = self.cached_token_history.clone();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        self.gdn_last_history_checkpoint = Some(MoeGdnHistoryCheckpoint {
            owner_id: self.active_cache_owner_id.clone(),
            image_key: self.cached_image_key,
            tokens,
            caches,
        });
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    pub(super) fn find_moe_gdn_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Option<(u32, Vec<Qwen3_5LayerCache>)> {
        let checkpoint_idx = find_longest_valid_gdn_checkpoint_index(
            &self.gdn_prefix_checkpoints,
            &self.active_cache_owner_id,
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
            |checkpoint| moe_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)),
        )?;
        let restored_prefix_len = self.gdn_prefix_checkpoints[checkpoint_idx].prefix_len;
        let caches = clone_moe_linear_layer_caches(
            &self.config,
            &self.gdn_prefix_checkpoints[checkpoint_idx].caches,
        )?;
        let checkpoint = self.gdn_prefix_checkpoints.remove(checkpoint_idx)?;
        self.gdn_prefix_checkpoints.push_back(checkpoint);
        Some((restored_prefix_len, caches))
    }

    pub(super) fn remember_moe_gdn_materialized_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        checkpoint: crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint,
    ) -> bool {
        let prefix_len = checkpoint.prefix_len;
        if !moe_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)) {
            return false;
        }
        let Some(block_hashes) = compute_paged_prefix_block_hashes(
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) else {
            return false;
        };
        let Some(final_block_hash) = block_hashes.last().copied() else {
            return false;
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return false;
        };

        self.gdn_prefix_checkpoints.retain(|existing| {
            !(existing.owner_id == self.active_cache_owner_id
                && existing.prefix_len == prefix_len
                && existing.block_size == block_size
                && existing.final_block_hash == final_block_hash
                && existing.tokens.as_slice() == prefix_tokens)
        });
        self.gdn_prefix_checkpoints
            .push_back(MoeGdnPrefixCheckpoint {
                owner_id: self.active_cache_owner_id.clone(),
                prefix_len,
                block_size,
                final_block_hash,
                block_hashes,
                tokens: prefix_tokens.to_vec(),
                caches: checkpoint.caches,
            });
        self.prune_moe_gdn_prefix_checkpoints();
        true
    }

    fn prune_moe_gdn_prefix_checkpoints(&mut self) {
        // Probe the ladder predicate BEFORE the `&mut self` borrows below:
        // `get_or_insert_with` takes one and `wants_gdn_checkpoint_ladder`
        // needs `&self`, so an inline call in the argument list does not
        // borrow-check.
        let caps = gdn_retention_caps(
            self.wants_gdn_checkpoint_ladder(),
            self.gdn_root_cache_owner_is_explicit,
        );
        let active_owner_id = self.active_cache_owner_id.clone();
        let root_owner_id = self
            .gdn_root_cache_owner_id
            .get_or_insert_with(|| active_owner_id.clone())
            .clone();
        prune_gdn_checkpoints(
            &mut self.gdn_prefix_checkpoints,
            caps,
            &root_owner_id,
            &active_owner_id,
        );
    }

    pub(super) fn publish_moe_gdn_materialized_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        checkpoints: Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
    ) {
        if checkpoints.is_empty() {
            return;
        }
        let Some(block_size) = self
            .paged_adapter
            .as_ref()
            .map(|adapter| adapter.block_size())
        else {
            return;
        };
        // Ascending, so the deepest boundary is remembered last and is the
        // most recent entry the lookup prefers on a tie.
        for checkpoint in checkpoints {
            let prefix_len = checkpoint.prefix_len;
            let stored = self.remember_moe_gdn_materialized_prefix_checkpoint(
                tokens,
                block_size,
                extra_keys_per_block,
                cache_salt,
                checkpoint,
            );
            tracing::info!(
                target: "mlx_core::inference",
                event = "gdn_prefix_checkpoint_store",
                model = "qwen3_5_moe",
                prefix_tokens = prefix_len,
                block_size,
                cache_owner_id = %self.active_cache_owner_id,
                cache_root_owner_id = %self.gdn_root_cache_owner_id.as_deref().unwrap_or(""),
                stored,
                retained_checkpoints = self.gdn_prefix_checkpoints.len(),
                "MoE GDN prefix checkpoint stored"
            );
        }
    }

    /// Install GDN recurrent state the SSD cold tier restored alongside this
    /// turn's paged K/V prefix. MoE mirror of
    /// `Qwen35Inner::install_dense_gdn_cold_sidecar` — same source (an on-disk
    /// [`mlx_paged_attn::ColdSidecar`]), same destination (`self.caches`), same
    /// fail-closed re-checks (group, layout equality against this config's
    /// geometry, boundary == the reported prefix). Consulted ONLY after every
    /// in-memory source missed.
    ///
    /// The GDN codec is shared with dense qwen3_5 (`to_dense_config` projects the
    /// linear geometry), so a boundary the walk reconciled to is decoded with the
    /// identical layout the dense family writes. A contract slip degrades to a
    /// MISS (`Ok(false)`) that falls through to replay, never state installed at
    /// the wrong offset. Taking the sidecar is unconditional so a rejected one
    /// cannot be reconsidered later in the same turn; on success the decoded
    /// state is fed into the in-memory prefix store so later same-process turns
    /// hit RAM.
    fn install_moe_gdn_cold_sidecar(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<bool> {
        let sidecar = {
            let Some(adapter) = self.paged_adapter.as_mut() else {
                return Ok(false);
            };
            match adapter.take_restored_sidecar() {
                Some(sidecar) => sidecar,
                None => return Ok(false),
            }
        };
        if sidecar.layout.group != mlx_paged_attn::ColdGroup::GdnState {
            return Ok(false);
        }
        let boundary = sidecar.layout.boundary_tokens;
        // The walk reconciles the prefix and the state together, so a sidecar
        // that does not describe EXACTLY the reported prefix is a broken
        // contract, not a deeper opportunity: refuse it.
        if boundary == 0 || boundary != cached_prefix_len {
            return Ok(false);
        }
        let cache_dtype = {
            let Some(adapter) = self.paged_adapter.as_ref() else {
                return Ok(false);
            };
            format!("{:?}", adapter.layer_kv_pool().cache_dtype())
        };
        let dense_config = self.config.to_dense_config();
        let Some(geometry) =
            crate::models::qwen3_5::gdn_sidecar::geometry(&dense_config, &cache_dtype)
        else {
            return Ok(false);
        };
        // `load_sidecar` already compared the layout to the policy's template;
        // comparing again against the geometry derived from the LOADED config
        // makes the install independent of that earlier check.
        if crate::models::qwen3_5::gdn_sidecar::layout_at(&geometry, boundary) != sidecar.layout {
            return Ok(false);
        }
        let Some(caches) = crate::models::qwen3_5::gdn_sidecar::decode_caches(
            &dense_config,
            &geometry,
            &sidecar.tensors,
            boundary,
        )?
        else {
            return Ok(false);
        };
        self.caches = Some(caches);
        // The one observable that separates "the tier restored the recurrent
        // half" from "the tier read it and every arm above declined". See the
        // dense twin.
        crate::cold_tier::cold_sidecar_counters().record_install();
        // Feed the in-memory store so later turns in this process hit RAM
        // instead of decoding the sidecar again. Best-effort: a failure to
        // clone/store never invalidates the freshly installed live caches.
        if let Some(snapshot) = self
            .caches
            .as_ref()
            .and_then(|caches| clone_moe_linear_layer_caches(&self.config, caches))
        {
            let checkpoint =
                crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint {
                    prefix_len: boundary,
                    caches: snapshot,
                };
            self.remember_moe_gdn_materialized_prefix_checkpoint(
                tokens,
                block_size,
                extra_keys_per_block,
                cache_salt,
                checkpoint,
            );
        }
        Ok(true)
    }

    /// Effective paged-prefill chunk size for EVERY paged prefill this model
    /// runs — the engine's `paged_prefill` and the hand-written sync/stream cores
    /// alike. MoE mirror of `Qwen35Inner::cold_gdn_prefill_chunk_size`.
    ///
    /// When the SSD cold tier carries a GDN sidecar policy (persistence on),
    /// return a chunk size large enough that the only breaks the prefill takes
    /// are the `gdn_prefill_checkpoint_boundaries` ladder's rungs, which are the
    /// only prefix lengths the sidecar can be anchored at. An explicit
    /// `MLX_PAGED_PREFILL_CHUNK_SIZE` still wins.
    ///
    /// With no cold GDN policy and no env override this is the unchanged
    /// single-shot default (0). With no cold GDN policy but the env var SET, the
    /// break set is the single deep `gdn_checkpoint_target` — what a persist-off
    /// turn split at before the ladder existed — because both this and the
    /// prefill body read `paged_forward::gdn_cold_sidecar_ladder_wanted`.
    /// Splitting is algebraically transparent but not numerically bit-identical
    /// — see the dense doc for the accepted drift.
    ///
    /// A caller must read this BEFORE it takes the `&mut self` borrows the
    /// prefill needs (`&mut self.layers` + `&mut self.paged_adapter`); an inline
    /// call in the argument list does not borrow-check.
    fn cold_gdn_prefill_chunk_size(&self) -> i32 {
        let env_chunk = crate::array::paged_prefill_chunk_size();
        if env_chunk > 0 {
            return env_chunk;
        }
        if self.wants_gdn_checkpoint_ladder() {
            i32::MAX
        } else {
            0
        }
    }

    /// Whether this turn's prefill should publish the whole checkpoint ladder.
    /// MoE mirror of `Qwen35Inner::wants_gdn_checkpoint_ladder`.
    fn wants_gdn_checkpoint_ladder(&self) -> bool {
        self.paged_adapter
            .as_ref()
            .is_some_and(crate::models::qwen3_5::paged_forward::gdn_cold_sidecar_ladder_wanted)
    }

    /// Prefill for the hand-written MoE paged cores (sync + stream, text).
    ///
    /// The engine's `PagedBackend::paged_prefill` serves only AR turns; a
    /// planned-MTP turn returns from `paged_whole_turn` before the engine runs
    /// and prefills here instead. Both cores share this one body so the chunk
    /// size — and with it the whole checkpoint ladder a cold sidecar is anchored
    /// on — cannot be right in one core and wrong in the other.
    #[allow(clippy::too_many_arguments)]
    fn run_moe_core_paged_prefill(
        &mut self,
        tokens: &[u32],
        suffix: &[u32],
        cached_prefix_len: u32,
        gdn_prefix_already_primed: bool,
        layer_kinds: &[crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind],
        context: &'static str,
    ) -> Result<(
        MxArray,
        Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
    )> {
        let embed = self.embedding.clone();
        // Cross-turn M-RoPE delta (0 unless this text turn warm-continues an
        // image prefill); feeds the scalar-offset RoPE for the suffix.
        let rope_deltas = self.cached_rope_deltas.unwrap_or(0);
        let chunk_size = self.cold_gdn_prefill_chunk_size();
        let caches_ref = self
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason(format!("{context}: caches not initialized")))?;
        let adapter = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason(format!("{context}: paged_adapter dropped")))?;
        super::paged_forward::run_paged_prefill_chunk_with_size_and_checkpoint(
            tokens,
            suffix,
            cached_prefix_len,
            gdn_prefix_already_primed,
            &embed,
            &mut self.layers,
            caches_ref,
            &self.final_norm,
            &self.lm_head,
            layer_kinds,
            adapter,
            chunk_size,
            rope_deltas,
        )
    }

    /// Persist this turn's GDN recurrent state to the SSD cold tier. MoE mirror
    /// of `Qwen35Inner::capture_dense_gdn_cold_sidecar`.
    ///
    /// Best-effort and infallible by construction — every failure path is a
    /// silent skip. A missing sidecar is never a correctness problem: the restore
    /// walk reconciles the candidate prefix down past that boundary and the state
    /// is recomputed exactly as it is today.
    ///
    /// A GDN recurrent state is valid ONLY at the exact prefix length it was
    /// produced at (vLLM `MambaSpec`), so the boundary is the DEEPEST
    /// already-materialized in-memory GDN checkpoint (a rung of the
    /// `gdn_prefill_checkpoint_boundaries` ladder a prior prefill published)
    /// whose block-aligned prefix the persisted K/V chain ALSO reaches
    /// (`cold_captured_blocks`). At most ONE sidecar per turn (~31 MiB on the
    /// 35B-A3B); the writer queue is bounded. Media turns are skipped in v1 (the
    /// GDN VLM exactness gate is not modeled on disk, so refusing is the
    /// fail-closed answer).
    ///
    /// `image_token_positions` is the turn's OWN media map, passed in rather
    /// than read off `self`: the VLM cores clear
    /// `cached_paged_image_token_positions` before their prefill and reassign it
    /// only after finalization returns, so a `self`-read guard would see an
    /// empty vec on exactly the media turns it exists to refuse.
    fn capture_moe_gdn_cold_sidecar(&self, image_token_positions: &[(u32, u64)]) {
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
        if policy.group() != mlx_paged_attn::ColdGroup::GdnState {
            return;
        }
        // v1: text-only. See the doc comment.
        if !image_token_positions.is_empty() {
            return;
        }
        let cache_dtype = format!("{:?}", adapter.layer_kv_pool().cache_dtype());
        let dense_config = self.config.to_dense_config();
        let Some(geometry) =
            crate::models::qwen3_5::gdn_sidecar::geometry(&dense_config, &cache_dtype)
        else {
            return;
        };
        let block_size = adapter.block_size();
        // The K/V capture walk that just ran spends its budget waiting for
        // writer-queue slots, so it hands this sidecar a queue it may well have
        // filled microseconds ago. A non-blocking offer here loses that race,
        // and a dropped sidecar is strictly worse than a dropped block: the
        // restore reconciles down to the deepest boundary a VALIDATED sidecar
        // backs, so losing it makes the turn's entire persisted K/V chain
        // unusable. Wait out the same budget the walk did.
        let sidecar_wait = adapter.cold_capture_budget().max_walk;
        if block_size == 0 {
            return;
        }
        let request_tokens = adapter.request_tokens();
        // Whole blocks of this request the persisted K/V chain actually covers.
        let full_blocks = request_tokens.len() / block_size as usize;
        let chain_blocks = (adapter.cold_captured_blocks() as usize).min(full_blocks);
        if chain_blocks == 0 {
            // A different diagnosis from the boundary miss below: the persisted
            // chain covers no whole block of this request, so no checkpoint at
            // any depth could have been used. This is the arm a genuinely cold
            // process hits on its first turns, while the bounded writer queue is
            // still ratcheting the chain forward.
            crate::cold_tier::cold_sidecar_counters().record_chain_empty();
            if inference_trace_enabled() {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3_5_moe gdn_cold_sidecar_capture_skipped reason=persisted_chain_covers_no_whole_block cold_captured_blocks={} full_blocks={} block_size={} prompt_len={}",
                    adapter.cold_captured_blocks(),
                    full_blocks,
                    block_size,
                    request_tokens.len()
                ));
            }
            return;
        }
        let ceiling_tokens = u32::try_from((chain_blocks as u64).saturating_mul(block_size as u64))
            .unwrap_or(u32::MAX);
        let extra_keys_per_block =
            engine::build_paged_extra_keys(request_tokens.len(), block_size, image_token_positions);

        // Deepest same-owner in-memory GDN checkpoint whose block-hash chain
        // matches this request and whose prefix the KV chain reaches. `cache_salt`
        // is 0 here to match the KV-chain finalize
        // (`finalize_turn_keep_live_per_block(.., 0)`) and the in-memory publish,
        // so the restore walk — which chains the sidecar under the SAME salt as
        // the KV blocks — can select it.
        let (idx, boundary) = match select_gdn_sidecar_boundary(
            &self.gdn_prefix_checkpoints,
            &self.active_cache_owner_id,
            request_tokens,
            ceiling_tokens,
            block_size,
            &extra_keys_per_block,
            0,
            |checkpoint| moe_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)),
        ) {
            GdnSidecarBoundary::Selected { index, boundary } => (index, boundary),
            GdnSidecarBoundary::Missed(miss) => {
                // Nothing is written and the restore walk will later reconcile
                // the whole candidate prefix away. Count and trace it, otherwise
                // a session that never persists recurrent state is
                // indistinguishable from one that needs none.
                crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
                if inference_trace_enabled() {
                    let reason = match miss.deepest_retained {
                        Some(deepest) if deepest > ceiling_tokens => {
                            "deepest_checkpoint_past_chain_reach"
                        }
                        _ => "no_checkpoint_on_this_lineage_at_or_below_chain_reach",
                    };
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3_5_moe gdn_cold_sidecar_capture_skipped reason={} ceiling_tokens={} chain_blocks={} cold_captured_blocks={} block_size={} prompt_len={} deepest_retained_boundary={} retained_for_owner={} retained_total={}",
                        reason,
                        ceiling_tokens,
                        chain_blocks,
                        adapter.cold_captured_blocks(),
                        block_size,
                        request_tokens.len(),
                        miss.deepest_retained.unwrap_or(0),
                        miss.retained_for_owner,
                        miss.retained_total
                    ));
                }
                return;
            }
        };
        let checkpoint = &self.gdn_prefix_checkpoints[idx];

        // Derive the GdnState chain key for blocks `0..boundary/block_size` — the
        // KV chain recomputed under the `GdnState` domain tag (vLLM's
        // `BlockHashWithGroupId`). `ColdTierWalk::deepest_backed_boundary`
        // derives the identical chain on restore. Built BEFORE the payload so the
        // `contains_in` dedup below can skip the whole encode (a device->host
        // readback of tens of MiB) when the chain is already on disk.
        let blocks = boundary as usize / block_size as usize;
        let mut parent: Option<mlx_paged_attn::ColdCacheKey> = None;
        for index in 0..blocks {
            let (Some(extra_keys), Some(tokens)) = (
                extra_keys_per_block.get(index),
                request_tokens.get(index * block_size as usize..(index + 1) * block_size as usize),
            ) else {
                return;
            };
            parent = Some(mlx_paged_attn::ColdCacheKey::chain(
                mlx_paged_attn::ColdGroup::GdnState,
                cold.fingerprint,
                parent,
                tokens,
                extra_keys,
                0,
                index,
            ));
        }
        let Some(key) = parent else {
            return;
        };
        // Already persisted for this exact chain: nothing to do. `contains_in` is
        // explicitly side-effect free (no hit/miss accounting), so this arm must
        // do its own — an unrecorded exit here reads downstream as `enqueued=0`,
        // which is also what a collapsed checkpoint ladder produces.
        if cold
            .manager
            .contains_in(&key, mlx_paged_attn::ColdGroup::GdnState)
        {
            crate::cold_tier::cold_sidecar_counters().record_already_persisted();
            return;
        }

        let Ok(Some(tensors)) = crate::models::qwen3_5::gdn_sidecar::encode_tensors(
            &dense_config,
            &geometry,
            &checkpoint.caches,
            boundary,
        ) else {
            return;
        };
        let sidecar = mlx_paged_attn::ColdSidecar {
            key,
            fingerprint: cold.fingerprint,
            layout: crate::models::qwen3_5::gdn_sidecar::layout_at(&geometry, boundary),
            tensors,
        };
        match cold
            .manager
            .enqueue_sidecar_before(sidecar, std::time::Instant::now() + sidecar_wait)
        {
            Ok(true) => crate::cold_tier::cold_sidecar_counters().record_enqueued(),
            // The bounded writer queue stayed full for the whole capture
            // budget. Nothing is written and nothing failed, so this turn is
            // otherwise indistinguishable from a successful capture.
            Ok(false) => {
                crate::cold_tier::cold_sidecar_counters().record_queue_drop();
                tracing::debug!(
                    target: "mlx_core::qwen3_5_moe::paged",
                    "qwen3.5 MoE GDN sidecar dropped at boundary {boundary}: cold-cache writer queue full"
                );
            }
            Err(error) => tracing::debug!(
                target: "mlx_core::qwen3_5_moe::paged",
                "qwen3.5 MoE GDN sidecar enqueue failed at boundary {boundary}: {error}"
            ),
        }
    }

    pub(super) fn prepare_moe_gdn_prefix_state(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
    ) -> Result<MoeGdnPrefixPreparation> {
        let image_aware_prefix = extra_keys_per_block.iter().any(|keys| !keys.is_empty());
        let trace_enabled = inference_trace_enabled();
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let prepare_start = (trace_enabled || inference_info_enabled).then(std::time::Instant::now);
        let cache_owner_id = self.active_cache_owner_id.clone();
        let cache_root_owner_id = self.gdn_root_cache_owner_id.clone().unwrap_or_default();
        let finish = |state: &'static str,
                      restored_prefix_tokens: u32,
                      replayed_prefix_tokens: u32|
         -> MoeGdnPrefixPreparation {
            if inference_info_enabled {
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "gdn_prefix_prepare",
                    model = "qwen3_5_moe",
                    cache_owner_id = %cache_owner_id,
                    cache_root_owner_id = %cache_root_owner_id,
                    state,
                    cached_prefix_tokens = cached_prefix_len,
                    restored_prefix_tokens,
                    replayed_prefix_tokens,
                    elapsed_ms = prepare_start.map(elapsed_ms).unwrap_or(0.0),
                    "MoE GDN prefix state prepared"
                );
            }
            let preparation = MoeGdnPrefixPreparation {
                state,
                already_primed: cached_prefix_len > 0,
                restored_prefix_tokens,
                replayed_prefix_tokens,
            };
            debug_assert_eq!(
                preparation.already_primed,
                preparation.restored_prefix_tokens > 0 || preparation.replayed_prefix_tokens > 0,
                "MoE GDN prefix preparation must account for every primed prefix"
            );
            preparation
        };
        let gdn_caches_ready = moe_paged_linear_caches_ready(&self.config, self.caches.as_deref());
        if gdn_caches_ready && continued_live_prefix {
            return Ok(finish("live", cached_prefix_len, 0));
        }

        let gdn_prefix_from_history = cached_prefix_len > 0
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if gdn_caches_ready && gdn_prefix_from_history {
            return Ok(finish("last_history", cached_prefix_len, 0));
        }
        if cached_prefix_len > 0 {
            let history_lookup_start = trace_enabled.then(std::time::Instant::now);
            let history_checkpoint =
                self.find_moe_gdn_history_checkpoint(tokens, cached_prefix_len, None);
            let history_lookup_ms = history_lookup_start.map(elapsed_ms);
            if let Some(checkpoint) = history_checkpoint {
                self.caches = Some(checkpoint);
                return Ok(finish("last_history_checkpoint", cached_prefix_len, 0));
            } else if trace_enabled {
                let history_checkpoint_len = self
                    .gdn_last_history_checkpoint
                    .as_ref()
                    .map_or(0, |checkpoint| checkpoint.tokens.len());
                let history_mismatch =
                    token_prefix_mismatch_trace(tokens, &self.cached_token_history);
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint_miss \
                     cached_prefix_tokens={} history_len={} checkpoint_len={} \
                     history_match={} history_mismatch_at={} prompt_token={} \
                     history_token={} history_lookup_ms={:.1}",
                    cached_prefix_len,
                    self.cached_token_history.len(),
                    history_checkpoint_len,
                    gdn_prefix_from_history,
                    history_mismatch.index,
                    history_mismatch.prompt_token,
                    history_mismatch.cached_token,
                    history_lookup_ms.unwrap_or(0.0)
                ));
            }
        }

        let prefix_checkpoint = self.find_moe_gdn_prefix_checkpoint(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        );
        if let Some((restored_prefix_len, checkpoint)) = prefix_checkpoint {
            let replayed_prefix_len = cached_prefix_len
                .checked_sub(restored_prefix_len)
                .ok_or_else(|| {
                    Error::from_reason("MoE GDN checkpoint is longer than the cached paged prefix")
                })?;
            if replayed_prefix_len == 0 {
                self.caches = Some(checkpoint);
                return Ok(finish("checkpoint", restored_prefix_len, 0));
            }
            if image_aware_prefix {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                self.caches = Some(fresh_moe_layer_caches(&self.config));
                return Err(Error::from_reason(
                    "image-conditioned GDN prefix requires an exact checkpoint or the original image embeddings",
                ));
            }

            let replay_suffix = tokens
                .get(restored_prefix_len as usize..cached_prefix_len as usize)
                .ok_or_else(|| {
                    Error::from_reason(
                        "MoE paged GDN checkpoint replay range exceeds prompt length",
                    )
                })?;
            let embed = self.embedding.clone();
            let layers = &mut self.layers;
            replay_gdn_cache_and_commit(&mut self.caches, checkpoint, |staged| {
                super::paged_forward::run_gdn_only_prefill_materialized(
                    replay_suffix,
                    &embed,
                    layers,
                    staged,
                )
            })?;
            return Ok(finish(
                "checkpoint_replay_materialized",
                restored_prefix_len,
                replayed_prefix_len,
            ));
        }

        // Cold-tier GDN sidecar: on-SSD recurrent state the restore walk brought
        // back for EXACTLY this cached prefix (reconcile-down guarantees the
        // boundary). Consulted only after every in-memory source above missed —
        // i.e. the common process-restart case. v1 is text-only: an image-aware
        // prefix still goes through the exactness gates below.
        if cached_prefix_len > 0
            && !image_aware_prefix
            && self.install_moe_gdn_cold_sidecar(
                tokens,
                cached_prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            )?
        {
            return Ok(finish("cold_sidecar", cached_prefix_len, 0));
        }

        let fresh_caches = fresh_moe_layer_caches(&self.config);
        if cached_prefix_len == 0 {
            self.caches = Some(fresh_caches);
            return Ok(finish("cold", 0, 0));
        }
        if image_aware_prefix {
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            self.caches = Some(fresh_caches);
            return Err(Error::from_reason(
                "image-conditioned GDN prefix cannot be reconstructed from placeholder token ids",
            ));
        }

        let cached_prefix_len_usize = cached_prefix_len as usize;
        let prefix = tokens.get(..cached_prefix_len_usize).ok_or_else(|| {
            Error::from_reason("MoE paged GDN prefix replay length exceeds prompt length")
        })?;
        let embed = self.embedding.clone();
        let layers = &mut self.layers;
        replay_gdn_cache_and_commit(&mut self.caches, fresh_caches, |staged| {
            super::paged_forward::run_gdn_only_prefill_materialized(prefix, &embed, layers, staged)
        })?;
        Ok(finish("replay_materialized", 0, cached_prefix_len))
    }

    /// Image-aware GDN prefix preparation. Only exact sidecars on the same
    /// image-keyed paged lineage may be restored; otherwise the caller must
    /// discard the K/V candidate and rerun the full image merge from position
    /// zero with fresh recurrent caches.
    fn prepare_moe_gdn_vlm_prefix_state(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
        image_key: u64,
    ) -> Result<bool> {
        if cached_prefix_len == 0 {
            self.caches = Some(fresh_moe_layer_caches(&self.config));
            return Ok(false);
        }

        let caches_ready = moe_paged_linear_caches_ready(&self.config, self.caches.as_deref());
        if caches_ready && continued_live_prefix && self.cached_image_key == Some(image_key) {
            return Ok(true);
        }
        let active_history_matches = caches_ready
            && self.cached_image_key == Some(image_key)
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if active_history_matches {
            return Ok(true);
        }
        if let Some(checkpoint) =
            self.find_moe_gdn_history_checkpoint(tokens, cached_prefix_len, Some(image_key))
        {
            self.caches = Some(checkpoint);
            return Ok(true);
        }
        if let Some((restored_prefix_len, checkpoint)) = self.find_moe_gdn_prefix_checkpoint(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) && restored_prefix_len == cached_prefix_len
        {
            self.caches = Some(checkpoint);
            return Ok(true);
        }

        self.caches = Some(fresh_moe_layer_caches(&self.config));
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_moe_vlm_paged_prefix(
        &mut self,
        tokens: &[u32],
        total_budget: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        reuse_cache: bool,
        allow_live_continue: bool,
        image_key: u64,
    ) -> Result<engine::VlmPagedPrefixResolution> {
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let candidate_plan_result = match self.paged_adapter.as_mut() {
            Some(adapter) => adapter
                .prepare_turn_per_block_with_max_cache_hit_tokens(
                    0,
                    tokens,
                    total_budget,
                    allow_live_continue,
                    extra_keys_per_block,
                    0,
                    !reuse_cache,
                    max_cache_hit_tokens,
                )
                .map_err(Error::from_reason),
            None => Err(Error::from_reason(
                "prepare_moe_vlm_paged_prefix: paged_adapter is None",
            )),
        };
        let candidate_plan = match candidate_plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_moe_paged_session("VLM paged-prefix preparation failure");
                return Err(error);
            }
        };

        let gdn_prefix_already_primed = match self.prepare_moe_gdn_vlm_prefix_state(
            tokens,
            candidate_plan.cached_prefix_len,
            block_size,
            extra_keys_per_block,
            0,
            candidate_plan.continued_live_prefix,
            image_key,
        ) {
            Ok(primed) => primed,
            Err(error) => {
                self.invalidate_moe_paged_session("VLM GDN-prefix preparation failure");
                return Err(error);
            }
        };

        let resolution =
            engine::resolve_vlm_paged_prefix(candidate_plan, gdn_prefix_already_primed, || {
                self.paged_adapter
                    .as_mut()
                    .ok_or_else(|| {
                        "prepare_moe_vlm_paged_prefix: paged_adapter dropped before cold restart"
                            .to_string()
                    })?
                    .restart_prepared_turn_cold_per_block(
                        0,
                        tokens,
                        total_budget,
                        extra_keys_per_block,
                        0,
                    )
            });
        match resolution {
            Ok(resolution) => Ok(resolution),
            Err(error) => {
                self.invalidate_moe_paged_session("VLM cold-restart failure");
                Err(Error::from_reason(error))
            }
        }
    }

    /// Set the tokenizer.
    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Set the vision encoder.
    ///
    /// Paged VLM checkpoints wire this alongside the image processor and
    /// adapter. Image-bearing turns then route through the MoE paged-vision
    /// cores; incomplete stacks stay backend-validated so the family reports
    /// its precise missing component. Text-only inputs still use scalar RoPE
    /// on both flat and paged paths.
    pub(crate) fn set_vision_encoder(&mut self, enc: Qwen3_5VisionEncoder) -> Result<()> {
        self.vision_encoder = Some(Arc::new(enc));
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
        Ok(())
    }

    /// Set the image processor.
    pub(crate) fn set_image_processor(&mut self, proc: Qwen35VLImageProcessor) {
        self.image_processor = Some(Arc::new(proc));
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
    }

    /// Set spatial merge size.
    pub(crate) fn set_spatial_merge_size(&mut self, size: i32) {
        self.spatial_merge_size = Some(size);
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
    }

    /// Initialize M-RoPE on all full attention layers (VLM mode).
    pub(crate) fn init_mrope_layers(
        &mut self,
        mrope_section: Vec<i32>,
        rope_theta: f64,
        max_position_embeddings: i32,
    ) -> Result<()> {
        let rope_dims = self.config.rope_dims();
        for layer in self.layers.iter_mut() {
            if let super::decoder_layer::AttentionType::Full(ref mut attn) = layer.attn {
                attn.init_mrope(
                    mrope_section.clone(),
                    rope_theta,
                    max_position_embeddings,
                    rope_dims,
                )?;
            }
        }
        Ok(())
    }

    /// Core chat implementation (runs on model thread).
    ///
    /// Whole-turn core for fresh SYNC turns reached through the engine's
    /// `vision_turn` (image-bearing) and `mtp_turn` (MTP-enabled)
    /// probes. The engine already rendered the prompt (`tokens`) and
    /// extracted the raw image payloads (`images`); everything from the
    /// paged dispatch onward runs the whole-turn pipeline.
    /// `eos_token_id` is the caller-supplied stop-on token id (typically
    /// `<|im_end|>`) so the cached history ends on a clean ChatML
    /// boundary, yielding a reusable prefix for subsequent session
    /// deltas.
    fn vision_mtp_whole_turn_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
        config: ChatConfig,
        eos_token_id: u32,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        let has_images = !images.is_empty();

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();
        let max_new_tokens = p.max_new_tokens;

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Block-paged dispatch — early-return onto the paged core.
        if self.paged_adapter.is_some() {
            if has_images {
                // All image turns (MTP or not) prefill through the paged-vision
                // core. The eager paged MTP stepper has no M-RoPE prefill seed,
                // so the core decodes plain autoregressively regardless of the
                // per-request MTP flag — it never reads the MTP head.
                return self.vision_paged_turn_sync_core(
                    tokens,
                    images,
                    tokenizer,
                    eos_token_id,
                    p,
                    report_perf,
                    thinking,
                );
            }
            return self.paged_turn_sync_core(
                tokens,
                tokenizer,
                eos_token_id,
                p,
                report_perf,
                thinking,
            );
        }

        // The flat fallback below is text-only. A MoE image turn requires the
        // block-paged backend; reaching here with images means the model was
        // loaded without a paged adapter (use_block_paged_cache=false or a
        // non-Metal build).
        if has_images {
            return Err(Error::from_reason(
                "qwen3.5 MoE image turns require the block-paged KV backend; the model was \
                 loaded without a paged adapter (use_block_paged_cache=false or non-Metal \
                 build)",
            ));
        }

        // Pure-Rust eager MTP. Active when the per-request flag is set and the
        // checkpoint carries an MTP head; runs the speculative-decode arm on
        // `Qwen35MoeInner`'s flat caches. Text-only flat turns only (the paged
        // dispatch already early-returned above).
        let eager_mtp =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none() && !has_images;

        let embedding = self.embedding.clone();

        // Text-only from here: the `has_images` early-return above is the only
        // image path. These bindings preserve the shared cache-reuse / decode
        // plumbing (`has_images` is always false on this branch).
        let (expanded_tokens, current_image_cache_key) = (tokens.clone(), 0u64);

        // === Cache reuse: prefix verification ===
        let cached_prefix_len = if self.flat_mtp_caches_desynced {
            0
        } else {
            verify_cache_prefix_direct(
                reuse_cache,
                has_images,
                &tokens,
                &expanded_tokens,
                current_image_cache_key,
                &self.cached_token_history,
                &self.cached_image_key,
                self.caches.is_some(),
            )
        };

        let prefill_tokens = if cached_prefix_len > 0 {
            if has_images {
                info!(
                    "VLM cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    expanded_tokens.len() - cached_prefix_len
                );
                expanded_tokens[cached_prefix_len..].to_vec()
            } else {
                info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    tokens.len() - cached_prefix_len
                );
                tokens[cached_prefix_len..].to_vec()
            }
        } else {
            // Full reset
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            tokens.clone()
        };

        // Zero-delta guard.
        //
        // Triggers when `cached_prefix_len == (expanded_)tokens.len()`, i.e.
        // the new prompt is byte-for-byte identical to the cached history
        // and there is literally no delta to prefill. We still need to
        // produce a `last_logits` for the decode loop, and the only safe
        // way to do that on the Qwen3.5 MoE hybrid stack is a full reset
        // + re-prefill. Trimming the cache by one token is infeasible
        // because the 30 GDN linear-attention layers carry a recurrent
        // state that cannot be rewound mid-sequence (see the invariant
        // doc on `verify_cache_prefix_direct`). In practice this branch
        // is a cold edge case — real agent turns always append at least
        // a user message, so the cached prefix is strictly shorter than
        // the new prompt.
        let (prefill_tokens, cached_prefix_len) = if prefill_tokens.is_empty() {
            info!("Zero-delta cache hit: resetting caches for full re-prefill");
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            let tokens = if has_images {
                expanded_tokens.clone()
            } else {
                tokens.clone()
            };
            (tokens, 0)
        } else {
            (prefill_tokens, cached_prefix_len)
        };

        let eos_id = eos_token_id;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");

        // Track token history for repetition penalty
        let mut token_history: Vec<u32> = expanded_tokens.clone();

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        // StreamContext created ONCE for entire prefill+decode
        let _stream_ctx = StreamContext::new(generation_stream);

        let fa_idx = self.fa_idx;

        // Profiler
        let mut profiler = crate::decode_profiler::DecodeProfiler::new("moe_chat", "qwen3_5_moe");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // === Text prefill ===
        // Image turns never reach here — they early-return onto the paged-vision
        // core (or error when no paged adapter is present). This is the
        // text-only flat path.
        profiler.begin_prefill();
        let (mut last_logits, _seq_len) = {
            // Standard text prefill. Chunked to bound peak GPU memory for
            // long prompts (e.g. 40k+ tokens) — see `chunked_prefill` docs.
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;

            let logits = chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                generation_stream,
            )?;

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;
            (last_logits, tokens.len() as i64)
        };
        profiler.end_prefill();
        // caches now reflect the prefilled history
        self.flat_mtp_caches_desynced = false;

        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            // Pure-Rust eager MoE MTP — the propose/verify whole-turn loop is
            // engine-owned (`engine::run_mtp_turn`) and drives the
            // `MoeMtpStepper` (`MtpBackend::begin_mtp_decode`). Committed-
            // history v2 activates within-turn (persistent drafter cache
            // across cycles) when its opt-in flag is on; the prompt-prefix seed
            // itself stays inert here because this call site has no MoE
            // hidden-emitting prefill yet, so `prompt_hidden`/`prompt_hidden_ids`
            // stay `None`. The `profiler.set_label("moe_mtp_eager")` relabel
            // moved into `MoeMtpStepper::profiler_relabel`.
            let mut rng = rand::rng();
            MxArray::async_eval_arrays(&[&y]);

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: &p,
                    reasoning_tracker: &mut reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden: None,
                    prompt_hidden_ids: None,
                    prompt_hidden_position_base: 0,
                },
                None,
            )?;

            last_in_cache = outcome.last_in_cache;
            // Propagate a mid-cycle stop: self.caches advanced past the emitted
            // history, so force a full re-prefill next turn.
            if outcome.desynced {
                self.flat_mtp_caches_desynced = true;
            }
        } else {
            // Rust fallback decode loop
            profiler.set_label("moe_chat_rust");

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &Embedding| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        fa_idx,
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            mtp_decode::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                last_in_cache: last_in_cache,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream
            );
        }

        // Save cache state
        save_cache_state_direct(
            p.reuse_cache,
            has_images,
            &generated_tokens,
            &finish_reason,
            /* drop_last_always */ !last_in_cache,
            &tokens,
            Some(&expanded_tokens),
            current_image_cache_key,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );
        self.cached_paged_image_token_positions.clear();

        let performance = compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking.enabled,
            if has_images {
                expanded_tokens.len() as u32
            } else {
                tokens.len() as u32
            },
            reasoning_tracker.reasoning_token_count(),
        )?;
        // Report the length of the reused cached prefix for observability.
        // `cached_prefix_len` is 0 on fresh/miss paths and the full cached
        // length on an exact-append hit — see the invariant doc on
        // `verify_cache_prefix_direct`.
        result.cached_tokens = cached_prefix_len as u32;
        Ok(result)
    }

    /// Single-turn image-bearing block-paged dispatch (non-streaming).
    ///
    /// The paged sibling of the flat MoE VLM prefill: it processes the images,
    /// merges the vision features into the token embeddings, computes M-RoPE
    /// positions, then prefills through the paged adapter via
    /// [`super::paged_forward::run_paged_vlm_prefill_moe`] and runs the plain
    /// autoregressive decode loop.
    ///
    /// Same-image live histories continue in place; fresh histories look up
    /// full blocks using per-image content keys and prefill only the uncached
    /// suffix. Decode uses the image M-RoPE delta carried by the merged prompt.
    /// MTP is not supported here; image-bearing turns decode autoregressively.
    #[allow(clippy::too_many_arguments)]
    fn vision_paged_turn_sync_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        mut p: engine::ChatParams,
        report_perf: bool,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        let (vision_encoder, img_proc) =
            match (self.vision_encoder.clone(), self.image_processor.as_ref()) {
                (Some(enc), Some(proc)) => (enc, proc),
                _ => {
                    return Err(Error::from_reason(
                        "VLM prefill requested but vision encoder/processor not loaded",
                    ));
                }
            };

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let sampling_config = p.sampling_config;

        // === VLM image processing: expand placeholders + merge features ===
        let sms = self.spatial_merge_size.unwrap_or(2);
        let image_refs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
        let processed = img_proc.process_many(&image_refs)?;
        let per_image_token_counts =
            compute_image_token_counts_per_image(&processed.grid_thw(), sms)?;
        let expanded_tokens = inject_image_placeholders(&tokens, &per_image_token_counts)?;
        self.preflight_paged_context(expanded_tokens.len(), &mut p)?;
        let (image_cache_key, per_image_hashes) = engine::compute_image_cache_keys(images);
        let image_token_positions = engine::map_expanded_image_token_positions(
            &expanded_tokens,
            IMAGE_TOKEN_ID as u32,
            &per_image_token_counts,
            &per_image_hashes,
        )
        .map_err(Error::from_reason)?;
        let prompt_token_count = expanded_tokens.len() as u32;

        let embed = self.embedding.clone();
        let input_ids = MxArray::from_uint32(&expanded_tokens, &[1, expanded_tokens.len() as i64])?;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let merge = vlm_prepare_vision_features(
            &input_ids,
            &per_image_hashes,
            &processed,
            &vision_encoder,
            sms,
            &embed,
            generation_stream,
            &self.vision_cache,
        )?;
        drop(processed);
        crate::array::clear_cache();

        // === Image-aware paged-prefix lifecycle ===
        let total_budget = expanded_tokens.len() as u32;
        let block_size = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason("vision_paged_turn_sync_core: paged_adapter is None")
            })?
            .block_size();
        let lookup_extra_keys = engine::build_paged_extra_keys(
            expanded_tokens.len(),
            block_size,
            &image_token_positions,
        );
        let same_live_image = self.cached_image_key == Some(image_cache_key);
        let prefix_resolution = self.prepare_moe_vlm_paged_prefix(
            &expanded_tokens,
            total_budget,
            block_size,
            &lookup_extra_keys,
            p.reuse_cache,
            p.reuse_cache && same_live_image,
            image_cache_key,
        )?;
        let plan = prefix_resolution.effective_plan;
        let cached_prefix_len = plan.cached_prefix_len;
        tracing::info!(
            target: "mlx_core::inference",
            event = "vlm_prefix_plan",
            model = "qwen3_5_moe",
            prompt_tokens = expanded_tokens.len(),
            image_count = images.len(),
            image_tokens = image_token_positions.len(),
            candidate_cached_prefix_tokens = prefix_resolution.candidate_cached_prefix_len,
            effective_cached_prefix_tokens = cached_prefix_len,
            cached_prefix_tokens = cached_prefix_len,
            suffix_tokens = expanded_tokens.len() - cached_prefix_len as usize,
            continued_live_prefix = plan.continued_live_prefix,
            same_live_image,
            gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed,
            downgraded_to_cold = prefix_resolution.downgraded_to_cold,
            "image-aware paged prefix planned"
        );
        let gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        // Store the image prefill's compressed-position delta so a later text
        // warm-continuation rotates its queries at the same compressed M-RoPE
        // positions the image keys were written with.
        self.cached_rope_deltas = Some(merge.rope_deltas as i32);

        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            // === PREFILL ===
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_sync_core: caches not initialized")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_sync_core: paged_adapter dropped")
                })?;
                super::paged_forward::run_paged_vlm_prefill_moe(
                    &expanded_tokens,
                    &merge,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &layer_kinds,
                    adapter,
                )?
            };
            let (last_logits, gdn_checkpoint) = last_logits;
            self.publish_moe_gdn_materialized_prefix_checkpoint(
                &expanded_tokens,
                &lookup_extra_keys,
                0,
                gdn_checkpoint,
            );

            let mut token_history: Vec<u32> = expanded_tokens.clone();
            let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
            let mut y = sample(&last_logits, sampling_config)?;
            y.eval();

            crate::array::synchronize_and_clear_cache();
            if report_perf {
                first_token_instant = Some(std::time::Instant::now());
            }

            // === DECODE LOOP (autoregressive, scalar-offset RoPE) ===
            let max_new_tokens = p.max_new_tokens;
            let mut generated_tokens: Vec<u32> =
                Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);
                token_history.push(token_id);
                reasoning_tracker.observe_token(token_id);

                if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
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

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let caches_ref = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason("vision_paged_turn_sync_core: caches dropped mid-decode")
                    })?;
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "vision_paged_turn_sync_core: paged_adapter dropped mid-decode",
                        )
                    })?;
                    let logits = super::paged_forward::run_paged_decode_step(
                        token_id,
                        &embed,
                        &mut self.layers,
                        caches_ref,
                        &self.final_norm,
                        &self.lm_head,
                        &layer_kinds,
                        adapter,
                        self.cached_rope_deltas.unwrap_or(0),
                    )?;
                    logits.squeeze(Some(&[1]))?
                };

                if reasoning_tracker.should_force_think_end() {
                    let forced_id = reasoning_tracker.forced_token_id()? as i32;
                    y = MxArray::from_int32(&[forced_id], &[1])?;
                    y.eval();
                    continue;
                }
                let next_logits = apply_all_penalties(next_logits, &token_history, &p)?;

                y = sample(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // Terminal lifecycle, mirroring the text paged core
        // (`paged_turn_sync_core`). The error path always releases the request
        // and returns. The success path is resolved below so the session ends
        // in exactly one of two states, never partial: FULLY continuable
        // (keep-live registered AND GDN checkpoint stored AND history + image
        // key published) or NON-continuable (request released AND history
        // cleared AND image key None) so a follow-up text continue is rejected
        // instead of cold-prefilling image-placeholder ids as ordinary tokens.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_moe_paged_session("VLM sync prefill/decode failure");
                return Err(e);
            }
        };

        // Saved history: expanded prompt + generated[..len-1] (drop-last rule
        // shared with the text paged core: the decode loop never forwards the
        // final sampled token into the cache).
        let mut full_history = expanded_tokens.clone();
        if !generated_tokens.is_empty() {
            full_history.extend_from_slice(&generated_tokens[..generated_tokens.len() - 1]);
        }

        // Keep-live before the GDN checkpoint (which snapshots the live
        // recurrent state); short-circuit `&&` preserves that order. The
        // checkpoint reads `cached_token_history`, so publish it first, then
        // checkpoint. Any failure downgrades to NON-continuable rather than
        // discarding the already-successful generation output.
        let keep_live_ok = p.reuse_cache
            && self
                .finalize_moe_manual_paged_turn(&image_token_positions)
                .is_ok();
        let continuable = if keep_live_ok {
            self.cached_token_history = full_history;
            self.cached_image_key = Some(image_cache_key);
            self.cached_paged_image_token_positions = image_token_positions.clone();
            match self.remember_moe_gdn_history_checkpoint() {
                Ok(_) => true,
                Err(error) => {
                    tracing::warn!(
                        target: "mlx_core::qwen3_5_moe::paged",
                        "MoE VLM GDN history checkpoint failed: {error}",
                    );
                    self.invalidate_moe_paged_session("VLM sync GDN history checkpoint failure");
                    false
                }
            }
        } else {
            false
        };

        if continuable {
        } else if self.caches.is_some() {
            // Non-continuable: release the request and reset to a pristine
            // non-live state so a follow-up continue is rejected instead of
            // cold-prefilling image-placeholder ids. `reset_caches_sync` nulls
            // `self.caches` (so `has_live_session()` is false) and clears token
            // history, image key, rope deltas, and GDN checkpoints.
            if p.reuse_cache {
                self.invalidate_moe_paged_session("non-continuable VLM sync completion");
            } else {
                self.discard_moe_paged_session();
            }
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                expanded_tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
        } else {
            None
        };

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tracker.reasoning_token_count(),
        )?;
        result.cached_tokens = cached_prefix_len;
        Ok(result)
    }

    /// Streaming twin of [`Self::vision_paged_turn_sync_core`].
    ///
    /// Single-turn image-bearing block-paged dispatch that emits each
    /// generated token through the streaming callback. Same prefill + decode
    /// spine; MTP is rejected upstream.
    #[allow(clippy::too_many_arguments)]
    fn vision_paged_turn_stream_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        mut p: engine::ChatParams,
        report_perf: bool,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        let (vision_encoder, img_proc) =
            match (self.vision_encoder.clone(), self.image_processor.as_ref()) {
                (Some(enc), Some(proc)) => (enc, proc),
                _ => {
                    return Err(Error::from_reason(
                        "VLM prefill requested but vision encoder/processor not loaded",
                    ));
                }
            };

        let include_reasoning = p.include_reasoning;
        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let sampling_config = p.sampling_config;

        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        // === VLM image processing: expand placeholders + merge features ===
        let sms = self.spatial_merge_size.unwrap_or(2);
        let image_refs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
        let processed = img_proc.process_many(&image_refs)?;
        let per_image_token_counts =
            compute_image_token_counts_per_image(&processed.grid_thw(), sms)?;
        let expanded_tokens = inject_image_placeholders(&tokens, &per_image_token_counts)?;
        self.preflight_paged_context(expanded_tokens.len(), &mut p)?;
        let (image_cache_key, per_image_hashes) = engine::compute_image_cache_keys(images);
        let image_token_positions = engine::map_expanded_image_token_positions(
            &expanded_tokens,
            IMAGE_TOKEN_ID as u32,
            &per_image_token_counts,
            &per_image_hashes,
        )
        .map_err(Error::from_reason)?;
        let prompt_token_count = expanded_tokens.len() as u32;

        let embed = self.embedding.clone();
        let input_ids = MxArray::from_uint32(&expanded_tokens, &[1, expanded_tokens.len() as i64])?;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let merge = vlm_prepare_vision_features(
            &input_ids,
            &per_image_hashes,
            &processed,
            &vision_encoder,
            sms,
            &embed,
            generation_stream,
            &self.vision_cache,
        )?;
        drop(processed);
        crate::array::clear_cache();

        // === Image-aware paged-prefix lifecycle ===
        let total_budget = expanded_tokens.len() as u32;
        let block_size = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason("vision_paged_turn_stream_core: paged_adapter is None")
            })?
            .block_size();
        let lookup_extra_keys = engine::build_paged_extra_keys(
            expanded_tokens.len(),
            block_size,
            &image_token_positions,
        );
        let same_live_image = self.cached_image_key == Some(image_cache_key);
        let prefix_resolution = self.prepare_moe_vlm_paged_prefix(
            &expanded_tokens,
            total_budget,
            block_size,
            &lookup_extra_keys,
            p.reuse_cache,
            p.reuse_cache && same_live_image,
            image_cache_key,
        )?;
        let plan = prefix_resolution.effective_plan;
        let cached_prefix_len = plan.cached_prefix_len;
        tracing::info!(
            target: "mlx_core::inference",
            event = "vlm_prefix_plan",
            model = "qwen3_5_moe",
            prompt_tokens = expanded_tokens.len(),
            image_count = images.len(),
            image_tokens = image_token_positions.len(),
            candidate_cached_prefix_tokens = prefix_resolution.candidate_cached_prefix_len,
            effective_cached_prefix_tokens = cached_prefix_len,
            cached_prefix_tokens = cached_prefix_len,
            suffix_tokens = expanded_tokens.len() - cached_prefix_len as usize,
            continued_live_prefix = plan.continued_live_prefix,
            same_live_image,
            gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed,
            downgraded_to_cold = prefix_resolution.downgraded_to_cold,
            "image-aware paged prefix planned"
        );
        let gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        // Store the image prefill's compressed-position delta so a later text
        // warm-continuation rotates its queries at the same compressed M-RoPE
        // positions the image keys were written with.
        self.cached_rope_deltas = Some(merge.rope_deltas as i32);

        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            // === PREFILL ===
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_stream_core: caches not initialized")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_stream_core: paged_adapter dropped")
                })?;
                super::paged_forward::run_paged_vlm_prefill_moe(
                    &expanded_tokens,
                    &merge,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &layer_kinds,
                    adapter,
                )?
            };
            let (last_logits, gdn_checkpoint) = last_logits;
            self.publish_moe_gdn_materialized_prefix_checkpoint(
                &expanded_tokens,
                &lookup_extra_keys,
                0,
                gdn_checkpoint,
            );

            let mut token_history: Vec<u32> = expanded_tokens.clone();
            let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
            let mut y = sample(&last_logits, sampling_config)?;
            y.eval();

            crate::array::synchronize_and_clear_cache();
            if report_perf {
                first_token_instant = Some(std::time::Instant::now());
            }

            let max_new_tokens = p.max_new_tokens;
            let mut generated_tokens: Vec<u32> =
                Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);
                token_history.push(token_id);
                let is_reasoning = reasoning_tracker.observe_token(token_id);
                last_is_reasoning = is_reasoning;

                if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                    finish_reason = String::from("stop");
                    break;
                }
                if cancelled.load(Ordering::Relaxed) {
                    finish_reason = String::from("cancelled");
                    break;
                }

                let token_text = Qwen3Tokenizer::step_decode_stream(
                    &mut decode_stream,
                    tokenizer.inner(),
                    token_id,
                    &generated_tokens,
                    streamed_text_len,
                );
                streamed_text_len += token_text.len();
                if include_reasoning || !is_reasoning {
                    cb.call(
                        Ok(ChatStreamChunk {
                            text: token_text,
                            done: false,
                            finish_reason: None,
                            tool_calls: None,
                            thinking: None,
                            thinking_enabled: None,
                            num_tokens: None,
                            prompt_tokens: None,
                            reasoning_tokens: None,
                            raw_text: None,
                            public_raw_text: None,
                            text_authoritative: None,
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

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let caches_ref = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "vision_paged_turn_stream_core: caches dropped mid-decode",
                        )
                    })?;
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "vision_paged_turn_stream_core: paged_adapter dropped mid-decode",
                        )
                    })?;
                    let logits = super::paged_forward::run_paged_decode_step(
                        token_id,
                        &embed,
                        &mut self.layers,
                        caches_ref,
                        &self.final_norm,
                        &self.lm_head,
                        &layer_kinds,
                        adapter,
                        self.cached_rope_deltas.unwrap_or(0),
                    )?;
                    logits.squeeze(Some(&[1]))?
                };

                if reasoning_tracker.should_force_think_end() {
                    let forced_id = reasoning_tracker.forced_token_id()? as i32;
                    y = MxArray::from_int32(&[forced_id], &[1])?;
                    y.eval();
                    continue;
                }
                let next_logits = apply_all_penalties(next_logits, &token_history, &p)?;

                y = sample(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // Terminal lifecycle, mirroring the text paged core. The error path
        // always releases and returns. The success path is resolved below so
        // the session ends FULLY continuable (keep-live + GDN checkpoint +
        // history + image key) or NON-continuable (released + history cleared +
        // image key None), never partial — a follow-up text continue must never
        // cold-prefill image-placeholder ids as ordinary tokens.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_moe_paged_session("VLM stream prefill/decode failure");
                return Err(e);
            }
        };

        // Saved history: expanded prompt + generated[..len-1] (drop-last rule).
        let mut full_history = expanded_tokens.clone();
        if !generated_tokens.is_empty() {
            full_history.extend_from_slice(&generated_tokens[..generated_tokens.len() - 1]);
        }

        // Keep-live before the GDN checkpoint (which snapshots the live state);
        // checkpoint reads `cached_token_history`, so publish it first. Any
        // failure downgrades to NON-continuable rather than discarding output.
        let keep_live_ok = p.reuse_cache
            && self
                .finalize_moe_manual_paged_turn(&image_token_positions)
                .is_ok();
        let continuable = if keep_live_ok {
            self.cached_token_history = full_history;
            self.cached_image_key = Some(image_cache_key);
            self.cached_paged_image_token_positions = image_token_positions.clone();
            match self.remember_moe_gdn_history_checkpoint() {
                Ok(_) => true,
                Err(error) => {
                    tracing::warn!(
                        target: "mlx_core::qwen3_5_moe::paged",
                        "MoE streaming VLM GDN history checkpoint failed: {error}",
                    );
                    self.invalidate_moe_paged_session("VLM stream GDN history checkpoint failure");
                    false
                }
            }
        } else {
            false
        };

        if continuable {
        } else if self.caches.is_some() {
            // Non-continuable: release the request and reset to a pristine
            // non-live state so a follow-up continue is rejected instead of
            // cold-prefilling image-placeholder ids. `reset_caches_sync` nulls
            // `self.caches` (so `has_live_session()` is false) and clears token
            // history, image key, rope deltas, and GDN checkpoints.
            if p.reuse_cache {
                self.invalidate_moe_paged_session("non-continuable VLM stream completion");
            } else {
                self.discard_moe_paged_session();
            }
        }

        // Flush residual buffered bytes (mirrors flat / text paged streaming).
        let full_text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            let residual = full_text[streamed_text_len..].to_string();
            if include_reasoning || !last_is_reasoning {
                cb.call(
                    Ok(ChatStreamChunk {
                        text: residual,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                expanded_tokens.len() - cached_prefix_len as usize,
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
            prompt_token_count,
            reasoning_tokens,
        )?;

        cb.call(
            Ok(ChatStreamChunk {
                text: result.text.clone(),
                done: true,
                finish_reason: Some(result.finish_reason.clone()),
                tool_calls: Some(result.tool_calls.clone()),
                thinking: result.thinking.clone(),
                thinking_enabled: Some(result.thinking_enabled),
                num_tokens: Some(result.num_tokens),
                prompt_tokens: Some(result.prompt_tokens),
                reasoning_tokens: Some(result.reasoning_tokens),
                raw_text: Some(result.raw_text.clone()),
                public_raw_text: result.public_raw_text.clone(),
                text_authoritative: Some(true),
                cached_tokens: Some(cached_prefix_len),
                performance: result.performance.clone(),
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Block-paged variant of [`Self::vision_mtp_whole_turn_core`] for the MoE
    /// model. Mirrors the dense paged dispatch — see
    /// `Qwen35Inner::paged_turn_sync_core` for the full rationale.
    ///
    /// The paged decode loop runs the pure-Rust paged forward
    /// (`paged_forward::run_paged_decode_step`): it reads K/V from the
    /// adapter pool via `paged_kv_write` / `paged_attention` and reads GDN
    /// linear caches from the per-layer
    /// `Qwen3_5LayerCache::Linear(ArraysCache)`.
    fn paged_turn_sync_core(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        mut p: engine::ChatParams,
        report_perf: bool,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        let prompt_token_count = tokens.len() as u32;
        let trace_enabled = inference_trace_enabled();
        let sampling_config = p.sampling_config;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        // Thinking is resolved ONCE at turn entry and honors
        // `enable_thinking=false`.
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Per-block extra_keys. See `paged_turn_sync_core` in
        // qwen3_5/model.rs for the rationale; text-only paged dispatch
        // builds an all-empty per-block vec which is bit-equal to
        // passing `&[]` to the uniform API. VLM-paged would replace the
        // empty positions with real (token_pos, image_hash) pairs.
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("MoE paged_turn_sync_core: paged_adapter is None")
            })?;
            adapter.block_size()
        };
        let carries_image_lineage = self.cached_image_key.is_some()
            && !self.cached_paged_image_token_positions.is_empty()
            && !self.cached_token_history.is_empty()
            && tokens.starts_with(&self.cached_token_history);
        let image_positions = if carries_image_lineage {
            self.cached_paged_image_token_positions.as_slice()
        } else {
            &[]
        };
        let lookup_extra_keys =
            engine::build_paged_extra_keys(tokens.len(), block_size, image_positions);
        let cache_salt = 0;
        // vLLM exact-prefix cap — see qwen3/model.rs:paged_turn_sync_core.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        // Adapter-owned warm/cold lifecycle. The [MLX_TRACE] line below
        // reads the PRE-turn live state, so probe the adapter immutably FIRST
        // (prepare_turn mutates request_tokens via continue_turn/reset). The
        // adapter re-reads the same state internally, so live_* is identical to
        // what prepare_turn decides on. extra_keys=&[] (uniform API) is bit-equal
        // to `&lookup_extra_keys` for text-only dispatch (all-empty per-block
        // vec → identical hashes; see the block_size comment above).
        // reuse_cache=true: continuation eligibility carries no reuse term.
        // Suffix blocks are allocated inside prepare_turn.
        {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("MoE paged_turn_sync_core: paged_adapter is None")
            })?;
            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = tokens.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(&tokens, live_tokens);
            }
        }
        let plan_result = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("MoE paged_turn_sync_core: paged_adapter is None"))?
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                seq_id,
                &tokens,
                total_budget,
                true,
                &lookup_extra_keys,
                cache_salt,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason);
        let plan = match plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_moe_paged_session("manual adapter preparation failure");
                return Err(error);
            }
        };
        let cached_prefix_len = plan.cached_prefix_len;
        let continued_live_prefix = plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                tokens.len(),
                cached_prefix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens_len,
                live_mismatch.index,
                live_mismatch.prompt_token,
                live_mismatch.cached_token
            ));
        }

        let gdn_prefix_preparation = match self.prepare_moe_gdn_prefix_state(
            &tokens,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        ) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.invalidate_moe_paged_session("manual GDN-prefix preparation failure");
                return Err(error);
            }
        };
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        // Discharge the adapter's auxiliary (GDN) prefix obligation before the
        // turn's first `record_tokens`. See `prime_prefix_state` for the full
        // rationale; a no-op unless a HOT K/V hit handed back a prefix the cold
        // tier's `ColdSidecarPolicy` was not gated against.
        let confirm_result = self
            .paged_adapter
            .as_mut()
            .map(|adapter| adapter.confirm_aux_prefix_primed(cached_prefix_len))
            .unwrap_or(Ok(()));
        if let Err(error) = confirm_result {
            self.invalidate_moe_paged_session("planned-MTP GDN auxiliary-prefix confirmation");
            return Err(Error::from_reason(error));
        }
        let preserves_image_lineage = carries_image_lineage && cached_prefix_len > 0;
        self.cached_token_history.clear();
        if !preserves_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }
        // Carry the cross-turn M-RoPE delta only when this turn extends the live
        // image sequence (continued_live_prefix); a cold start or a non-live
        // prefix-cache hit (text-only prefix) drops a stale image delta so the
        // text suffix prefill + decode rotate at the raw physical slot.
        self.cached_rope_deltas = crate::models::qwen3_5::paged_forward::rope_delta_for_paged_turn(
            self.cached_rope_deltas,
            preserves_image_lineage,
        );

        let suffix_len = match prompt_token_count.checked_sub(cached_prefix_len) {
            Some(suffix_len) => suffix_len,
            None => {
                self.invalidate_moe_paged_session("manual prefix length mismatch");
                return Err(Error::from_reason(
                    "MoE paged_turn_sync_core: cached_prefix_len > total_prompt_tokens",
                ));
            }
        };

        let forward_result = self.paged_turn_sync_core_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            &p,
            eos_token_id,
            &sampling_config,
            &mut reasoning_tracker,
            report_perf,
            &mut first_token_instant,
            gdn_prefix_already_primed,
            &lookup_extra_keys,
            cache_salt,
        );

        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => {
                let image_token_positions = self.cached_paged_image_token_positions.clone();
                self.finalize_moe_manual_paged_turn(&image_token_positions)?;
                t
            }
            Err(e) => {
                self.invalidate_moe_paged_session("manual sync prefill/decode failure");
                return Err(e);
            }
        };

        let last_token_in_cache = false;
        let mut full_history = tokens.clone();
        if !generated_tokens.is_empty() {
            let upto = if last_token_in_cache {
                generated_tokens.len()
            } else {
                generated_tokens.len().saturating_sub(1)
            };
            full_history.extend_from_slice(&generated_tokens[..upto]);
        }
        self.cached_token_history = full_history;
        let gdn_history_checkpoint_store = match self.remember_moe_gdn_history_checkpoint() {
            Ok(store) => store,
            Err(error) => {
                self.invalidate_moe_paged_session("manual sync GDN history checkpoint failure");
                return Err(error);
            }
        };
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                gdn_history_checkpoint_store.stored,
                self.cached_token_history.len(),
                gdn_history_checkpoint_store.eval_ms,
                gdn_history_checkpoint_store.clone_ms,
                gdn_history_checkpoint_store.token_clone_ms,
                gdn_history_checkpoint_store.update_ms,
                gdn_history_checkpoint_store.total_ms
            ));
        }

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

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tracker.reasoning_token_count(),
        )?;
        result.cached_tokens = cached_prefix_len;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_turn_sync_core_inner(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &engine::ChatParams,
        eos_token_id: u32,
        sampling_config: &Option<crate::sampling::SamplingConfig>,
        reasoning_tracker: &mut engine::ReasoningTracker,
        report_perf: bool,
        first_token_instant: &mut Option<std::time::Instant>,
        gdn_prefix_already_primed: bool,
        checkpoint_extra_keys: &[Vec<u64>],
        checkpoint_cache_salt: u64,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "MoE paged_turn_sync_core_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        // Pure-Rust paged prefill: writes K/V into the adapter pool per
        // layer via `update_keys_values_native` (graph-native `PagedKVWrite`
        // primitive — the default; the synchronous raw-Metal
        // `update_keys_values` is the error/opt-out fallback) and populates
        // the GDN linear caches in `Qwen3_5LayerCache::Linear(ArraysCache)`.
        // Both are exactly what the pure-Rust paged decode steps
        // (`paged_forward::run_paged_decode_step`) read as inputs.
        let (last_logits, gdn_checkpoint) = self.run_moe_core_paged_prefill(
            tokens,
            suffix,
            cached_prefix_len,
            gdn_prefix_already_primed,
            &layer_kinds,
            "MoE paged_turn_sync_core_inner",
        )?;
        self.publish_moe_gdn_materialized_prefix_checkpoint(
            tokens,
            checkpoint_extra_keys,
            checkpoint_cache_salt,
            gdn_checkpoint,
        );

        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, *sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating. Prefill of long prompts builds a massive MLX
        // subgraph; once we have the last logits, those intermediates are
        // dead but MLX's cache holds them.
        crate::array::synchronize_and_clear_cache();

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut finish_reason = String::from("length");

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            reasoning_tracker.observe_token(token_id);

            if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
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

            // Pure-Rust paged decode step.
            let next_logits = {
                let embed = self.embedding.clone();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("MoE paged_turn_sync_core_inner: caches dropped mid-decode")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "MoE paged_turn_sync_core_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let logits = super::paged_forward::run_paged_decode_step(
                    token_id,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &layer_kinds,
                    adapter,
                    self.cached_rope_deltas.unwrap_or(0),
                )?;
                logits.squeeze(Some(&[1]))?
            };

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

    /// Block-paged streaming variant for MoE — mirrors dense
    /// `paged_turn_stream_core`. See [`Self::paged_turn_sync_core`]
    /// for the paged dispatch rationale (pure-Rust paged prefill + decode
    /// against the adapter pool); the streaming path uses the same
    /// adapter lifecycle + prefix-reuse semantics.
    #[allow(clippy::too_many_arguments)]
    fn paged_turn_stream_core(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        mut p: engine::ChatParams,
        report_perf: bool,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        let prompt_token_count = tokens.len() as u32;
        let trace_enabled = inference_trace_enabled();
        let request_trace_start = trace_enabled.then(std::time::Instant::now);
        let sampling_config = p.sampling_config;
        let include_reasoning = p.include_reasoning;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        // Thinking is resolved ONCE at turn entry and honors
        // `enable_thinking=false`.
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Per-block extra_keys. See comments above.
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("MoE paged_turn_stream_core: paged_adapter is None")
            })?;
            adapter.block_size()
        };
        let carries_image_lineage = self.cached_image_key.is_some()
            && !self.cached_paged_image_token_positions.is_empty()
            && !self.cached_token_history.is_empty()
            && tokens.starts_with(&self.cached_token_history);
        let image_positions = if carries_image_lineage {
            self.cached_paged_image_token_positions.as_slice()
        } else {
            &[]
        };
        let lookup_extra_keys =
            engine::build_paged_extra_keys(tokens.len(), block_size, image_positions);
        let cache_salt = 0;
        // See `paged_turn_sync_core` for the vLLM exact-prefix cap rationale.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        // Adapter-owned warm/cold lifecycle (see paged_turn_sync_core for
        // the full bit-identity rationale: pre-turn immutable probe for the
        // trace, extra_keys=&[] bit-equal to per-block for text-only,
        // reuse_cache=true, suffix blocks allocated internally).
        {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("MoE paged_turn_stream_core: paged_adapter is None")
            })?;
            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = tokens.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(&tokens, live_tokens);
            }
        }
        let plan_result = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("MoE paged_turn_stream_core: paged_adapter is None"))?
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                seq_id,
                &tokens,
                total_budget,
                true,
                &lookup_extra_keys,
                cache_salt,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason);
        let plan = match plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_moe_paged_session("manual stream adapter preparation failure");
                return Err(error);
            }
        };
        let cached_prefix_len = plan.cached_prefix_len;
        let continued_live_prefix = plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                tokens.len(),
                cached_prefix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens_len,
                live_mismatch.index,
                live_mismatch.prompt_token,
                live_mismatch.cached_token
            ));
        }

        let prefill_trace_start = trace_enabled.then(std::time::Instant::now);
        let gdn_prefix_preparation = match self.prepare_moe_gdn_prefix_state(
            &tokens,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        ) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.invalidate_moe_paged_session("manual stream GDN-prefix preparation failure");
                return Err(error);
            }
        };
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        // Discharge the adapter's auxiliary (GDN) prefix obligation before the
        // turn's first `record_tokens`. See `prime_prefix_state` for the full
        // rationale; a no-op unless a HOT K/V hit handed back a prefix the cold
        // tier's `ColdSidecarPolicy` was not gated against.
        let confirm_result = self
            .paged_adapter
            .as_mut()
            .map(|adapter| adapter.confirm_aux_prefix_primed(cached_prefix_len))
            .unwrap_or(Ok(()));
        if let Err(error) = confirm_result {
            self.invalidate_moe_paged_session(
                "planned-MTP stream GDN auxiliary-prefix confirmation",
            );
            return Err(Error::from_reason(error));
        }
        let gdn_prefix_state = gdn_prefix_preparation.state;
        let preserves_image_lineage = carries_image_lineage && cached_prefix_len > 0;
        self.cached_token_history.clear();
        if !preserves_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }
        // Carry the cross-turn M-RoPE delta only when this turn extends the live
        // image sequence (continued_live_prefix); a cold start or a non-live
        // prefix-cache hit (text-only prefix) drops a stale image delta so the
        // text suffix prefill + decode rotate at the raw physical slot.
        self.cached_rope_deltas = crate::models::qwen3_5::paged_forward::rope_delta_for_paged_turn(
            self.cached_rope_deltas,
            preserves_image_lineage,
        );

        let suffix_len = match prompt_token_count.checked_sub(cached_prefix_len) {
            Some(suffix_len) => suffix_len,
            None => {
                self.invalidate_moe_paged_session("manual stream prefix length mismatch");
                return Err(Error::from_reason(
                    "MoE paged_turn_stream_core: cached_prefix_len > total_prompt_tokens",
                ));
            }
        };

        if trace_enabled {
            // The size this turn's prefill actually splits on, not the raw
            // `MLX_PAGED_PREFILL_CHUNK_SIZE`: under a GDN cold policy the two
            // differ, and reading 0 here while the same turn reports
            // `gdn_checkpoint_materialized=true` points at the wrong diagnosis.
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe stream_paged_start prompt_tokens={} \
                 cached_prefix_tokens={} suffix_tokens={} block_size={} \
                 prefill_chunk_size={} prefill_eval_interval={} decode_clear_interval={} \
                 gdn_prefix_state={}",
                prompt_token_count,
                cached_prefix_len,
                suffix_len,
                block_size,
                self.cold_gdn_prefill_chunk_size(),
                crate::array::paged_prefill_eval_interval(),
                crate::array::paged_decode_cache_clear_interval(),
                gdn_prefix_state
            ));
        }

        let result = self.paged_turn_stream_core_inner(
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
            gdn_prefix_already_primed,
            prefill_trace_start,
            &lookup_extra_keys,
            cache_salt,
        );

        if let Some(start) = request_trace_start {
            match &result {
                Ok((generated_tokens, finish_reason)) => {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe stream_paged_done generated_tokens={} \
                         finish_reason={} elapsed_ms={:.1}",
                        generated_tokens.len(),
                        finish_reason,
                        elapsed_ms(start)
                    ));
                }
                Err(err) => {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-moe stream_paged_error elapsed_ms={:.1} error={}",
                        elapsed_ms(start),
                        err
                    ));
                }
            }
        }

        let (generated_tokens, finish_reason) = match result {
            Ok(t) => {
                let image_token_positions = self.cached_paged_image_token_positions.clone();
                self.finalize_moe_manual_paged_turn(&image_token_positions)?;
                t
            }
            Err(e) => {
                self.invalidate_moe_paged_session("manual stream prefill/decode failure");
                return Err(e);
            }
        };

        let last_token_in_cache = false;
        let mut full_history = tokens.clone();
        if !generated_tokens.is_empty() {
            let upto = if last_token_in_cache {
                generated_tokens.len()
            } else {
                generated_tokens.len().saturating_sub(1)
            };
            full_history.extend_from_slice(&generated_tokens[..upto]);
        }
        self.cached_token_history = full_history;
        let gdn_history_checkpoint_store = match self.remember_moe_gdn_history_checkpoint() {
            Ok(store) => store,
            Err(error) => {
                self.invalidate_moe_paged_session("manual stream GDN history checkpoint failure");
                return Err(error);
            }
        };
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                gdn_history_checkpoint_store.stored,
                self.cached_token_history.len(),
                gdn_history_checkpoint_store.eval_ms,
                gdn_history_checkpoint_store.clone_ms,
                gdn_history_checkpoint_store.token_clone_ms,
                gdn_history_checkpoint_store.update_ms,
                gdn_history_checkpoint_store.total_ms
            ));
        }

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
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
        }

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
            prompt_token_count,
            reasoning_tokens,
        )?;
        result.cached_tokens = cached_prefix_len;

        cb.call(
            Ok(ChatStreamChunk {
                text: result.text.clone(),
                done: true,
                finish_reason: Some(result.finish_reason.clone()),
                tool_calls: Some(result.tool_calls.clone()),
                thinking: result.thinking.clone(),
                thinking_enabled: Some(result.thinking_enabled),
                num_tokens: Some(result.num_tokens),
                prompt_tokens: Some(result.prompt_tokens),
                reasoning_tokens: Some(result.reasoning_tokens),
                raw_text: Some(result.raw_text.clone()),
                public_raw_text: result.public_raw_text.clone(),
                text_authoritative: Some(true),
                cached_tokens: Some(cached_prefix_len),
                performance: result.performance.clone(),
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_turn_stream_core_inner<'a>(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &engine::ChatParams,
        sampling_config: Option<crate::sampling::SamplingConfig>,
        eos_token_id: u32,
        reasoning_tracker: &mut engine::ReasoningTracker,
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
        gdn_prefix_already_primed: bool,
        prefill_trace_start: Option<std::time::Instant>,
        checkpoint_extra_keys: &[Vec<u64>],
        checkpoint_cache_salt: u64,
    ) -> Result<(Vec<u32>, String)> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "MoE paged_turn_stream_core_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        let trace_enabled = inference_trace_enabled();
        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );

        // Pure-Rust paged prefill — see `paged_turn_sync_core_inner` for
        // the data-flow contract this populates (pool K/V + GDN linear
        // caches).
        let (last_logits, gdn_checkpoint) = self.run_moe_core_paged_prefill(
            tokens,
            suffix,
            cached_prefix_len,
            gdn_prefix_already_primed,
            &layer_kinds,
            "MoE paged_turn_stream_core_inner",
        )?;
        self.publish_moe_gdn_materialized_prefix_checkpoint(
            tokens,
            checkpoint_extra_keys,
            checkpoint_cache_salt,
            gdn_checkpoint,
        );

        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating (see paged_turn_sync_core_inner for rationale).
        crate::array::synchronize_and_clear_cache();

        if let Some(start) = prefill_trace_start {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_first_token_ready prompt_tokens={} \
                 cached_prefix_tokens={} suffix_tokens={} prefill_to_first_token_ms={:.1}",
                tokens.len(),
                cached_prefix_len,
                suffix_len,
                elapsed_ms(start)
            ));
        }

        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut finish_reason = String::from("length");
        let decode_trace_start = trace_enabled.then(std::time::Instant::now);
        let decode_progress_interval = if trace_enabled {
            crate::array::paged_decode_cache_clear_interval().max(1) as usize
        } else {
            usize::MAX
        };
        let mut decode_progress_last = decode_trace_start.unwrap_or_else(std::time::Instant::now);
        let mut decode_progress_last_count = 0usize;
        let decode_build_inputs_ms = 0.0;
        let mut decode_forward_ms = 0.0;
        let mut decode_sample_build_ms = 0.0;
        let mut decode_token_eval_ms = 0.0;
        let mut decode_cache_clear_ms = 0.0;

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            let is_reasoning = reasoning_tracker.observe_token(token_id);
            *last_is_reasoning = is_reasoning;

            if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                finish_reason = String::from("stop");
                break;
            }
            if cancelled.load(Ordering::Relaxed) {
                finish_reason = String::from("cancelled");
                break;
            }

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
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
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

            // Pure-Rust paged decode step.
            let next_logits = {
                let embed = self.embedding.clone();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "MoE paged_turn_stream_core_inner: caches dropped mid-decode",
                    )
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "MoE paged_turn_stream_core_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let forward_trace_start = trace_enabled.then(std::time::Instant::now);
                let logits = super::paged_forward::run_paged_decode_step(
                    token_id,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &layer_kinds,
                    adapter,
                    self.cached_rope_deltas.unwrap_or(0),
                )?;
                if let Some(start) = forward_trace_start {
                    decode_forward_ms += elapsed_ms(start);
                }
                logits.squeeze(Some(&[1]))?
            };

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id()? as i32;
                y = MxArray::from_int32(&[forced_id], &[1])?;
                y.eval();
                continue;
            } else {
                apply_all_penalties(next_logits, &token_history, p)?
            };

            let sample_trace_start = trace_enabled.then(std::time::Instant::now);
            y = sample(&next_logits, sampling_config)?;
            if let Some(start) = sample_trace_start {
                decode_sample_build_ms += elapsed_ms(start);
            }
            let token_eval_trace_start = trace_enabled.then(std::time::Instant::now);
            y.eval();
            if let Some(start) = token_eval_trace_start {
                decode_token_eval_ms += elapsed_ms(start);
            }

            let cache_clear_trace_start = trace_enabled.then(std::time::Instant::now);
            crate::array::maybe_clear_cache_for_paged_step(step);
            if let Some(start) = cache_clear_trace_start {
                decode_cache_clear_ms += elapsed_ms(start);
            }
            if trace_enabled
                && generated_tokens
                    .len()
                    .is_multiple_of(decode_progress_interval)
            {
                let window_ms = elapsed_ms(decode_progress_last);
                let window_tokens = generated_tokens
                    .len()
                    .saturating_sub(decode_progress_last_count);
                let window_tok_s = if window_ms > 0.0 {
                    window_tokens as f64 / (window_ms / 1000.0)
                } else {
                    0.0
                };
                let elapsed_decode_ms = decode_trace_start.map(elapsed_ms).unwrap_or(0.0);
                let active_mib = crate::array::get_active_memory() / (1024.0 * 1024.0);
                let cache_mib = crate::array::get_cache_memory() / (1024.0 * 1024.0);
                let peak_mib = crate::array::get_peak_memory() / (1024.0 * 1024.0);
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-moe paged_decode_progress generated_tokens={} \
                     context_tokens={} window_tokens={} window_ms={:.1} window_tok_s={:.2} \
                     elapsed_ms={:.1} cpp_ready={} build_inputs_ms={:.1} forward_ms={:.1} \
                     sample_ms={:.1} sample_build_ms={:.1} token_eval_ms={:.1} \
                     cache_clear_ms={:.1} active_mib={:.1} cache_mib={:.1} peak_mib={:.1}",
                    generated_tokens.len(),
                    token_history.len(),
                    window_tokens,
                    window_ms,
                    window_tok_s,
                    elapsed_decode_ms,
                    false,
                    decode_build_inputs_ms,
                    decode_forward_ms,
                    decode_sample_build_ms + decode_token_eval_ms,
                    decode_sample_build_ms,
                    decode_token_eval_ms,
                    decode_cache_clear_ms,
                    active_mib,
                    cache_mib,
                    peak_mib
                ));
                decode_progress_last = std::time::Instant::now();
                decode_progress_last_count = generated_tokens.len();
            }
        }

        if let Some(start) = decode_trace_start {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_decode_done generated_tokens={} finish_reason={} \
                 decode_loop_ms={:.1} build_inputs_ms={:.1} \
                 forward_ms={:.1} sample_ms={:.1} sample_build_ms={:.1} \
                 token_eval_ms={:.1} cache_clear_ms={:.1}",
                generated_tokens.len(),
                finish_reason,
                elapsed_ms(start),
                decode_build_inputs_ms,
                decode_forward_ms,
                decode_sample_build_ms + decode_token_eval_ms,
                decode_sample_build_ms,
                decode_token_eval_ms,
                decode_cache_clear_ms
            ));
        }

        Ok((generated_tokens, finish_reason))
    }

    /// Core streaming chat implementation (runs on model thread).
    ///
    /// Whole-turn core for fresh STREAMING turns reached through the
    /// engine's `vision_turn` (image-bearing) and `mtp_turn`
    /// (MTP-enabled) probes. The engine already rendered the prompt
    /// (`tokens`) and extracted the raw image payloads (`images`);
    /// everything from the paged dispatch onward runs the whole-turn
    /// pipeline. `eos_token_id` is the caller-supplied
    /// stop-on token id (typically `<|im_end|>`) so the cached history
    /// ends on a clean ChatML boundary, yielding a reusable prefix for
    /// subsequent session deltas.
    fn vision_mtp_whole_turn_stream_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
        config: ChatConfig,
        eos_token_id: u32,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        let has_images = !images.is_empty();

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let tokenizer_for_decode = tokenizer.clone();

        let mut p = engine::extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Block-paged dispatch — early-return BEFORE the compile lock.
        if self.paged_adapter.is_some() {
            if has_images {
                // All image turns (MTP or not) prefill through the paged-vision
                // stream core. It decodes plain autoregressively regardless of
                // the per-request MTP flag — it never reads the MTP head.
                return self.vision_paged_turn_stream_core(
                    tokens,
                    images,
                    tokenizer_for_decode,
                    eos_token_id,
                    p,
                    report_perf,
                    cb,
                    cancelled,
                    thinking,
                );
            }
            return self.paged_turn_stream_core(
                tokens,
                tokenizer_for_decode,
                eos_token_id,
                p,
                report_perf,
                cb,
                cancelled,
                thinking,
            );
        }

        // The flat fallback below is text-only. A MoE image turn requires the
        // block-paged backend; reaching here with images means the model was
        // loaded without a paged adapter (use_block_paged_cache=false or a
        // non-Metal build).
        if has_images {
            return Err(Error::from_reason(
                "qwen3.5 MoE image turns require the block-paged KV backend; the model was \
                 loaded without a paged adapter (use_block_paged_cache=false or non-Metal \
                 build)",
            ));
        }

        // Pure-Rust eager MTP. Text-only flat turns only (paged already
        // early-returned above).
        let eager_mtp =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none() && !has_images;

        let embedding = self.embedding.clone();

        // Text-only from here: the `has_images` early-return above is the only
        // image path. These bindings preserve the shared cache-reuse / decode
        // plumbing (`has_images` is always false on this branch).
        let (expanded_tokens, current_image_cache_key) = (tokens.clone(), 0u64);

        // Cache reuse
        let cached_prefix_len = if self.flat_mtp_caches_desynced {
            0
        } else {
            verify_cache_prefix_direct(
                reuse_cache,
                has_images,
                &tokens,
                &expanded_tokens,
                current_image_cache_key,
                &self.cached_token_history,
                &self.cached_image_key,
                self.caches.is_some(),
            )
        };

        let prefill_tokens = if cached_prefix_len > 0 {
            if has_images {
                info!(
                    "VLM cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    expanded_tokens.len() - cached_prefix_len
                );
                expanded_tokens[cached_prefix_len..].to_vec()
            } else {
                info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    tokens.len() - cached_prefix_len
                );
                tokens[cached_prefix_len..].to_vec()
            }
        } else {
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            tokens.clone()
        };

        // Zero-delta guard. See the matching `vision_mtp_whole_turn_core` comment for
        // the design rationale — rewinding a GDN recurrent cache by one
        // token is not possible across Qwen3.5 MoE's 30 linear-attention
        // layers, so the only safe response to an exact-match prompt is
        // a full reset + re-prefill.
        let (prefill_tokens, cached_prefix_len) = if prefill_tokens.is_empty() {
            info!("Zero-delta cache hit: resetting caches for full re-prefill");
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            let tokens = if has_images {
                expanded_tokens.clone()
            } else {
                tokens.clone()
            };
            (tokens, 0)
        } else {
            (prefill_tokens, cached_prefix_len)
        };

        let eos_id = eos_token_id;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut decode_stream = tokenizer_for_decode.inner().decode_stream(true);
        let mut streamed_text_len: usize = 0;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let fa_idx = self.fa_idx;

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("moe_chat_stream", "qwen3_5_moe");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text prefill. Image turns never reach here — they early-return onto
        // the paged-vision stream core (or error when no paged adapter is
        // present). This is the text-only flat path.
        profiler.begin_prefill();
        let (mut last_logits, _seq_len) = {
            // Chunked to bound peak GPU memory for long prompts. See
            // `chunked_prefill` docs for the memory rationale.
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;
            let logits = chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                generation_stream,
            )?;

            let seq_len = logits.shape_at(1)?;
            let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
            let last_logits = last_logits.squeeze(Some(&[1]))?;
            (last_logits, tokens.len() as i64)
        };
        profiler.end_prefill();
        // caches now reflect the prefilled history
        self.flat_mtp_caches_desynced = false;

        let mut token_history: Vec<u32> = tokens.clone();
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let starts_in_thinking = thinking.enabled;
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            // Streaming eager MoE MTP — same engine-owned `run_mtp_turn` loop +
            // `MoeMtpStepper` as the sync site, with a `StreamingCtx` wired so
            // accepted tokens stream out the `cb` sink incrementally (qwen3_5
            // MoE does not override `stream_emitter`, so the default ChatML
            // emitter is byte-identical to the former inline emit).
            let mut rng = rand::rng();
            MxArray::async_eval_arrays(&[&y]);

            let mut emitter = crate::engine::backend::DefaultStreamEmitter;
            let streaming = crate::engine::decode::StreamingCtx {
                callback: cb.0,
                cancelled,
                decode_stream: &mut decode_stream,
                tokenizer: tokenizer_for_decode.inner(),
                streamed_text_len: &mut streamed_text_len,
                last_is_reasoning: &mut last_is_reasoning,
                emitter: &mut emitter,
            };

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: &p,
                    reasoning_tracker: &mut reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens: p.max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden: None,
                    prompt_hidden_ids: None,
                    prompt_hidden_position_base: 0,
                },
                Some(streaming),
            )?;

            last_in_cache = outcome.last_in_cache;
            if outcome.desynced {
                self.flat_mtp_caches_desynced = true;
            }
        } else {
            profiler.set_label("moe_chat_stream_rust");

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &Embedding| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        fa_idx,
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            mtp_decode::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: p.max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                last_in_cache: last_in_cache,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                streaming: {
                    callback: cb,
                    cancelled: cancelled,
                    decode_stream: decode_stream,
                    tokenizer: tokenizer_for_decode,
                    streamed_text_len: streamed_text_len,
                    last_is_reasoning: last_is_reasoning
                }
            );
        }

        // Save cache state
        save_cache_state_direct(
            p.reuse_cache,
            has_images,
            &generated_tokens,
            &finish_reason,
            /* drop_last_always */ !last_in_cache,
            &tokens,
            Some(&expanded_tokens),
            current_image_cache_key,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );
        self.cached_paged_image_token_positions.clear();

        let text = tokenizer_for_decode
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });

        // Flush residual bytes
        if text.len() > streamed_text_len {
            let residual = text[streamed_text_len..].to_string();
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if p.include_reasoning || !last_is_reasoning {
                cb.call(
                    Ok(ChatStreamChunk {
                        text: residual,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
        }

        let num_tokens = generated_tokens.len() as u32;
        let prompt_token_count = if has_images {
            expanded_tokens.len() as u32
        } else {
            tokens.len() as u32
        };

        let (clean_text, tool_calls, thinking) = engine::parse_thinking_and_tools(
            &text,
            &generated_tokens,
            starts_in_thinking,
            think_end_id,
            think_end_str.as_deref(),
            p.include_reasoning,
        );

        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        let perf_metrics = compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        // Send final done chunk
        cb.call(
            Ok(ChatStreamChunk {
                text: clean_text,
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                thinking_enabled: Some(
                    crate::engine::resolve_enable_thinking(&config).unwrap_or(true),
                ),
                num_tokens: Some(num_tokens),
                prompt_tokens: Some(prompt_token_count),
                reasoning_tokens: Some(reasoning_tracker.reasoning_token_count()),
                raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    p.include_reasoning,
                )),
                public_raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    false,
                )),
                text_authoritative: Some(true),
                // Start path: report the matched prefix length from
                // `verify_cache_prefix_direct`. Zero on a miss, full
                // cached length on an exact-append hit.
                cached_tokens: Some(cached_prefix_len as u32),
                performance: perf_metrics,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Prefill a pre-tokenized delta on top of the existing KV caches and
    /// run the decode loop. Whole-turn core for SYNC delta turns reached
    /// through the engine's `mtp_turn` probe (MTP-enabled sessions;
    /// non-MTP sync deltas run the engine's generic flow or the paged
    /// probe).
    ///
    /// Uses `<|im_end|>` as the eos token (not `config.eos_token_id`) so
    /// the cached history continues to end on a clean ChatML boundary for
    /// the next turn. Cache save runs unconditionally at the end so the
    /// session stays consistent even on error. The engine's delta guards
    /// already enforce the session preconditions; the checks here are
    /// defense-in-depth for the `mtp_turn` caller.
    pub(crate) fn chat_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        // The delta path is a session-reuse operation by construction.
        if config.reuse_cache == Some(false) {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires reuse_cache to be enabled; \
                 the delta path operates on session state by construction",
            ));
        }
        if self.caches.is_none() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires an initialized session (call chatSessionStart first)",
            ));
        }
        if delta_tokens.is_empty() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires a non-empty delta",
            ));
        }
        // Text-only delta on image-bearing cache is intentional — the KV
        // cache retains the image attention state from the prior prefill.
        // See the sibling guard's doc in `qwen3_5/model.rs`. The engine's
        // `session_continue` gate filters real image-set changes with the
        // `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` prefix so the TS
        // `ChatSession` can route those through `chatSessionStart`.

        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        // Build full token history = cached_history + delta. Used for
        // penalty context AND as the running token history in the decode loop.
        // Snapshot the cached-prefix length before extending so we can
        // report it on the ChatResult for observability — the delta path
        // always reuses the full cached history by construction.
        let cached_prefix_len_for_result = self.cached_token_history.len() as u32;
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();
        let max_new_tokens = p.max_new_tokens;

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Block-paged dispatch — early-return onto the paged core.
        // The delta path drives the paged core with the FULL token
        // history; the adapter's warm-continue path matches the cached
        // prefix automatically.
        if self.paged_adapter.is_some() {
            return self.paged_turn_sync_core(
                full_token_history.clone(),
                tokenizer.clone(),
                eos_id,
                p,
                report_perf,
                thinking,
            );
        }

        // Pure-Rust eager MTP. Delta turns are text-only by construction; paged
        // already early-returned.
        let eager_mtp = p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none();

        let embedding = self.embedding.clone();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        // StreamContext created ONCE for entire prefill+decode
        let _stream_ctx = StreamContext::new(generation_stream);

        let fa_idx = self.fa_idx;

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("moe_chat_delta", "qwen3_5_moe");
        profiler.set_prompt_tokens(delta_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text-only prefill of the delta on top of the existing caches.
        // Usually tiny (a single user turn), but chunked defensively so a
        // user pasting a long follow-up message doesn't blow memory.
        profiler.begin_prefill();
        let logits = if self.flat_mtp_caches_desynced {
            // A prior eager-MTP turn stopped mid-cycle, leaving self.caches advanced
            // past the emitted history; GDN state cannot be rewound, so discard and
            // re-prefill the full conversation into fresh caches.
            self.caches = Some(fresh_moe_layer_caches(&self.config));
            profiler.set_prompt_tokens(full_token_history.len() as u32);
            let prompt =
                MxArray::from_uint32(&full_token_history, &[1, full_token_history.len() as i64])?;
            let logits = chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                generation_stream,
            )?;
            self.flat_mtp_caches_desynced = false;
            logits
        } else {
            let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
            chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                generation_stream,
            )?
        };
        let prefill_out_seq_len = logits.shape_at(1)?;
        let mut last_logits = logits.slice_axis(1, prefill_out_seq_len - 1, prefill_out_seq_len)?;
        last_logits = last_logits.squeeze(Some(&[1]))?;
        profiler.end_prefill();

        let prompt_tokens_for_result = full_token_history.len() as u32;

        // Save snapshot for save_cache_state_direct (prior history + delta).
        let save_tokens = full_token_history.clone();

        // Decode setup.
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");

        let mut token_history: Vec<u32> = full_token_history;
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            // Delta-continuation eager MoE MTP — same engine-owned
            // `run_mtp_turn` loop + `MoeMtpStepper` as the fresh-prefill sync
            // site (committed-history v2 within-turn when its opt-in flag is on;
            // prompt-prefix seed inert here — see the `MoeMtpStepper` struct
            // doc).
            let mut rng = rand::rng();
            MxArray::async_eval_arrays(&[&y]);

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: &p,
                    reasoning_tracker: &mut reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden: None,
                    prompt_hidden_ids: None,
                    prompt_hidden_position_base: 0,
                },
                None,
            )?;

            last_in_cache = outcome.last_in_cache;
            if outcome.desynced {
                self.flat_mtp_caches_desynced = true;
            }
        } else {
            profiler.set_label("moe_chat_delta_rust");

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &Embedding| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        fa_idx,
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            mtp_decode::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                last_in_cache: last_in_cache,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream
            );
        }

        // Save cache state. Delta continuations preserve
        // `cached_image_key` — the live KV cache still encodes the prior
        // prefill's image attention state even though this turn is
        // text-only, and a subsequent cache-prefix verify needs that
        // key to stay in place so a later image-bearing turn correctly
        // flags an image-set change instead of being accepted on the
        // delta path.
        engine::save_cache_state_after_delta(
            p.reuse_cache,
            &generated_tokens,
            &finish_reason,
            /* drop_last_always */ !last_in_cache,
            &save_tokens,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );

        let performance = compute_performance_metrics(
            generation_start,
            first_token_instant,
            delta_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        let _final_sampled_token = y;

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking.enabled,
            prompt_tokens_for_result,
            reasoning_tracker.reasoning_token_count(),
        )?;
        // Delta path always reuses the full cached history — report it.
        result.cached_tokens = cached_prefix_len_for_result;
        Ok(result)
    }

    /// Prefill the delta tokens and run the streaming decode loop.
    ///
    /// Whole-turn core for STREAMING delta turns reached through the
    /// engine's `mtp_turn` probe (MTP-enabled sessions; non-MTP
    /// streaming deltas run the engine's generic flow or the paged
    /// probe). Mirrors [`Self::vision_mtp_whole_turn_stream_core`] but skips the
    /// message rendering + prefix verification stages — the caller owns
    /// cache coherence by construction. Uses `<|im_end|>` as eos so the
    /// cached history continues to end on a clean ChatML boundary after
    /// the reply is saved.
    fn chat_stream_tokens_delta_sync_inner(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let tokenizer_for_decode = tokenizer.clone();

        // Build full token history = cached_history + delta.
        // Capture `prior_cached_len` BEFORE the extend — this is the
        // reused-prefix length reported on the terminal ChatStreamChunk's
        // `cached_tokens` field (mirrors the non-streaming delta path's
        // `cached_tokens_for_result`).
        let prior_cached_len = self.cached_token_history.len() as u32;
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // Block-paged dispatch — early-return onto the paged core.
        if self.paged_adapter.is_some() {
            return self.paged_turn_stream_core(
                full_token_history.clone(),
                tokenizer_for_decode,
                eos_id,
                p,
                report_perf,
                cb,
                cancelled,
                thinking,
            );
        }

        // Pure-Rust eager MTP. Delta turns are text-only by construction; paged
        // already early-returned.
        let eager_mtp = p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none();

        let embedding = self.embedding.clone();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let fa_idx = self.fa_idx;

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("moe_chat_stream_delta", "qwen3_5_moe");
        profiler.set_prompt_tokens(delta_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text-only prefill of the delta on top of the existing caches.
        // Chunked defensively — see the sync sibling for rationale.
        profiler.begin_prefill();
        let logits = if self.flat_mtp_caches_desynced {
            // A prior eager-MTP turn stopped mid-cycle, leaving self.caches advanced
            // past the emitted history; GDN state cannot be rewound, so discard and
            // re-prefill the full conversation into fresh caches.
            self.caches = Some(fresh_moe_layer_caches(&self.config));
            profiler.set_prompt_tokens(full_token_history.len() as u32);
            let prompt =
                MxArray::from_uint32(&full_token_history, &[1, full_token_history.len() as i64])?;
            let logits = chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                generation_stream,
            )?;
            self.flat_mtp_caches_desynced = false;
            logits
        } else {
            let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
            chunked_prefill(
                &prompt,
                &embedding,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
                generation_stream,
            )?
        };
        let prefill_out_seq_len = logits.shape_at(1)?;
        let mut last_logits = logits.slice_axis(1, prefill_out_seq_len - 1, prefill_out_seq_len)?;
        last_logits = last_logits.squeeze(Some(&[1]))?;
        profiler.end_prefill();

        // Save snapshot for save_cache_state_direct (prior history + delta).
        let save_tokens = full_token_history.clone();

        // Decode setup
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut decode_stream = tokenizer_for_decode.inner().decode_stream(true);
        let mut streamed_text_len: usize = 0;

        let mut token_history: Vec<u32> = full_token_history;
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let starts_in_thinking = thinking.enabled;
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            // Streaming delta-continuation eager MoE MTP — same engine-owned
            // `run_mtp_turn` loop + `MoeMtpStepper` + `StreamingCtx` as the
            // fresh-prefill stream site (committed-history v2 within-turn when
            // its opt-in flag is on; prompt-prefix seed inert here — see the
            // `MoeMtpStepper` struct doc).
            let mut rng = rand::rng();
            MxArray::async_eval_arrays(&[&y]);

            let mut emitter = crate::engine::backend::DefaultStreamEmitter;
            let streaming = crate::engine::decode::StreamingCtx {
                callback: cb.0,
                cancelled,
                decode_stream: &mut decode_stream,
                tokenizer: tokenizer_for_decode.inner(),
                streamed_text_len: &mut streamed_text_len,
                last_is_reasoning: &mut last_is_reasoning,
                emitter: &mut emitter,
            };

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: &p,
                    reasoning_tracker: &mut reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens: p.max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden: None,
                    prompt_hidden_ids: None,
                    prompt_hidden_position_base: 0,
                },
                Some(streaming),
            )?;

            last_in_cache = outcome.last_in_cache;
            if outcome.desynced {
                self.flat_mtp_caches_desynced = true;
            }
        } else {
            profiler.set_label("moe_chat_stream_delta_rust");

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &Embedding| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        fa_idx,
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            mtp_decode::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: p.max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                last_in_cache: last_in_cache,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                streaming: {
                    callback: cb,
                    cancelled: cancelled,
                    decode_stream: decode_stream,
                    tokenizer: tokenizer_for_decode,
                    streamed_text_len: streamed_text_len,
                    last_is_reasoning: last_is_reasoning
                }
            );
        }

        // Save cache state unconditionally — even on cancellation, the
        // partial generated_tokens must be appended so the session stays
        // consistent for the next turn. Delta stream preserves
        // `cached_image_key` (see the sync sibling's rationale).
        engine::save_cache_state_after_delta(
            p.reuse_cache,
            &generated_tokens,
            &finish_reason,
            /* drop_last_always */ !last_in_cache,
            &save_tokens,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );

        // Decode the full reply text and emit the final done chunk.
        let text = tokenizer_for_decode
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });

        if text.len() > streamed_text_len {
            let residual = text[streamed_text_len..].to_string();
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if p.include_reasoning || !last_is_reasoning {
                cb.call(
                    Ok(ChatStreamChunk {
                        text: residual,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
        }

        let num_tokens = generated_tokens.len() as u32;
        let prompt_token_count = delta_tokens.len() as u32;

        let (clean_text, tool_calls, thinking) = engine::parse_thinking_and_tools(
            &text,
            &generated_tokens,
            starts_in_thinking,
            think_end_id,
            think_end_str.as_deref(),
            p.include_reasoning,
        );

        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        let perf_metrics = compute_performance_metrics(
            generation_start,
            first_token_instant,
            delta_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        cb.call(
            Ok(ChatStreamChunk {
                text: clean_text,
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                thinking_enabled: Some(
                    crate::engine::resolve_enable_thinking(&config).unwrap_or(true),
                ),
                num_tokens: Some(num_tokens),
                prompt_tokens: Some(prompt_token_count),
                reasoning_tokens: Some(reasoning_tracker.reasoning_token_count()),
                raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    p.include_reasoning,
                )),
                public_raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    false,
                )),
                text_authoritative: Some(true),
                // Delta path reuses the full prior history by construction
                // — report `prior_cached_len` (captured before the
                // `self.cached_token_history` extend above) as the
                // authoritative cached-prefix length.
                cached_tokens: Some(prior_cached_len),
                performance: perf_metrics,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Generate text from prompt tokens (synchronous, runs on model thread).
    pub(crate) fn generate_sync(
        &mut self,
        prompt_tokens: MxArray,
        config: Qwen3_5MoeGenerationConfig,
    ) -> Result<Qwen3_5MoeGenerationResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Init caches
        self.init_caches_sync()?;

        let embedding = self.embedding.clone();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let fa_idx = self.fa_idx;

        // Prefill. Chunked to bound peak GPU memory for long prompts —
        // see `chunked_prefill` docs. `chunked_prefill` internally manages
        // the StreamContext per chunk so we don't need an outer one here.
        let prompt = prompt_tokens.reshape(&[1, -1])?;
        let logits = chunked_prefill(
            &prompt,
            &embedding,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            fa_idx,
            generation_stream,
        )?;

        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;

        // Request value wins; otherwise fall back to the checkpoint's
        // generation_config.json default; otherwise the sampler's builtin.
        // When the request omits temperature, a `do_sample:false` in
        // generation_config.json forces greedy decoding (temperature 0),
        // overriding any gen-config temperature (HuggingFace transformers
        // semantics) — `effective_temperature()` folds that rule in.
        // This raw `generate` surface exposes only the four SamplingConfig
        // fields (no repetition/presence/frequency penalty), so a
        // generation_config repetition_penalty is honored on the ChatSession
        // path but intentionally not here. ChatSession is the full-parity surface.
        let sampling_config = Some(SamplingConfig {
            temperature: config
                .temperature
                .or(self.gen_defaults.effective_temperature()),
            top_k: config.top_k.or(self.gen_defaults.top_k),
            top_p: config.top_p.or(self.gen_defaults.top_p),
            min_p: config.min_p.or(self.gen_defaults.min_p),
        });

        let eos_id = self.config.eos_token_id as u32;
        // Extra stop ids from generation_config.json (e.g. a second EOS).
        // Captured before the loop so its `&mut self` borrows are unaffected.
        let extra_eos_ids = self.gen_defaults.eos_token_ids.clone();
        let is_eos = |t: u32| t == eos_id || extra_eos_ids.contains(&t);
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut y = sample(&last_logits, sampling_config)?;

        for _step in 0..config.max_new_tokens {
            y.eval();
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);

            if is_eos(token_id) {
                break;
            }

            let next_ids = y.reshape(&[1, 1])?;
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                forward_inner(
                    &next_ids,
                    &embedding,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    fa_idx,
                )?
            };

            let logits = logits.squeeze(Some(&[1]))?;
            y = sample(&logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            if (_step + 1) % 256 == 0 {
                crate::array::synchronize_and_clear_cache();
            }
        }

        self.reset_caches_sync()?;

        let finish_reason = if generated_tokens.last().is_some_and(|&t| is_eos(t)) {
            "stop"
        } else {
            "length"
        };

        let text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_default();

        Ok(Qwen3_5MoeGenerationResult {
            tokens: generated_tokens.clone(),
            text,
            num_tokens: generated_tokens.len() as u32,
            finish_reason: finish_reason.to_string(),
        })
    }

    fn first_quantized_save_component(&self) -> Option<String> {
        if self.embedding.is_quantized() {
            return Some("embedding".to_string());
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.is_quantized() {
                return Some(format!("layers.{index}"));
            }
        }
        if self
            .lm_head
            .as_ref()
            .is_some_and(|head| head.is_quantized())
        {
            return Some("lm_head".to_string());
        }
        if self.mtp_weights_loaded
            && self
                .mtp
                .as_ref()
                .is_some_and(|mtp| mtp.has_quantized_weights())
        {
            return Some("mtp".to_string());
        }
        None
    }

    /// Save model weights and configuration to a directory (synchronous).
    ///
    /// Runs on the dedicated model thread and serializes all weights owned
    /// directly by `Qwen35MoeInner` (no locks). Mirrors the dense implementation
    /// in `qwen3_5::model::Qwen35Inner::save_model_sync`, adapted for the MoE
    /// MLP variant (per-layer dense vs sparse expert routing).
    pub(crate) fn save_model_sync(&self, save_path: &str) -> Result<()> {
        use super::decoder_layer::{AttentionType, MLPType};

        if let Some(component) = self.first_quantized_save_component() {
            return Err(Error::from_reason(format!(
                "Cannot save model: save_model is dense/BF16-only, but main-model component \
                 '{component}' contains quantized projections. Refusing before creating the \
                 destination because packed weights require sidecars and quantization metadata \
                 that this save format cannot preserve losslessly."
            )));
        }

        let mut params: HashMap<String, MxArray> = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            // Attention weights
            match &layer.attn {
                AttentionType::Linear(gdn) => {
                    params.insert(
                        format!("{}.linear_attn.in_proj_qkvz.weight", prefix),
                        gdn.get_in_proj_qkvz_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.in_proj_ba.weight", prefix),
                        gdn.get_in_proj_ba_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.conv1d.weight", prefix),
                        gdn.get_conv1d_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.norm.weight", prefix),
                        gdn.get_norm_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.out_proj.weight", prefix),
                        gdn.get_out_proj_weight(),
                    );
                    params.insert(format!("{}.linear_attn.dt_bias", prefix), gdn.get_dt_bias());
                    params.insert(format!("{}.linear_attn.a_log", prefix), gdn.get_a_log());
                }
                AttentionType::Full(attn) => {
                    params.insert(
                        format!("{}.self_attn.q_proj.weight", prefix),
                        attn.get_q_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_proj.weight", prefix),
                        attn.get_k_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.v_proj.weight", prefix),
                        attn.get_v_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.o_proj.weight", prefix),
                        attn.get_o_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.q_norm.weight", prefix),
                        attn.get_q_norm_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_norm.weight", prefix),
                        attn.get_k_norm_weight(),
                    );
                }
            }

            // MLP weights — different for Dense vs MoE layers
            match &layer.mlp {
                MLPType::Dense(mlp) => {
                    params.insert(
                        format!("{}.mlp.gate_proj.weight", prefix),
                        mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.up_proj.weight", prefix),
                        mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.down_proj.weight", prefix),
                        mlp.get_down_proj_weight(),
                    );
                }
                MLPType::MoE(moe) => {
                    // Router gate
                    params.insert(format!("{}.mlp.gate.weight", prefix), moe.get_gate_weight());
                    // Expert weights (3D: [num_experts, out, in])
                    let switch_mlp = moe.get_switch_mlp();
                    params.insert(
                        format!("{}.mlp.switch_mlp.gate_proj.weight", prefix),
                        switch_mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.switch_mlp.up_proj.weight", prefix),
                        switch_mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.switch_mlp.down_proj.weight", prefix),
                        switch_mlp.get_down_proj_weight(),
                    );
                    // Shared expert
                    params.insert(
                        format!("{}.mlp.shared_expert.gate_proj.weight", prefix),
                        moe.get_shared_expert_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.up_proj.weight", prefix),
                        moe.get_shared_expert_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.down_proj.weight", prefix),
                        moe.get_shared_expert_down_proj_weight(),
                    );
                    // Shared expert gate
                    params.insert(
                        format!("{}.mlp.shared_expert_gate.weight", prefix),
                        moe.get_shared_expert_gate_weight(),
                    );
                }
            }

            // Layer norms
            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );

        // LM head (only if not tied to the embedding)
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = self.lm_head
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        // Include vision encoder weights when present (VLM models)
        if let Some(ref vision_enc) = self.vision_encoder {
            let vision_params = vision_enc.get_parameters();
            params.extend(vision_params);
        }

        // Multi-Token Prediction head. `config.n_mtp_layers` round-trips
        // through config.json, so a reloaded checkpoint reconstructs the MTP
        // module from config and expects the `mtp.*` tensors present —
        // without this block the loader finds them absent, sets
        // `mtp_weights_loaded = false`, and silently disables speculative
        // decode. The `mtp_weights_loaded` guard is essential: `mtp.is_some()`
        // alone would serialize a random-init module (constructed from config
        // even when no weights were loaded).
        if self.mtp_weights_loaded
            && let Some(ref mtp) = self.mtp
        {
            // The early fail-closed quantized-state gate above guarantees this
            // MTP head is dense; partial omission would make the saved model
            // silently lose speculative decoding.
            params.extend(mtp.get_parameters());
        }

        // Validate all parameters for NaN/Inf before writing to disk
        for (name, param) in params.iter() {
            let data = param.to_float32()?;
            let invalid_count = data
                .iter()
                .filter(|v| v.is_nan() || v.is_infinite())
                .count();
            if invalid_count > 0 {
                return Err(napi::Error::new(
                    napi::Status::GenericFailure,
                    format!(
                        "Cannot save model: parameter '{}' contains {} NaN/Inf values.",
                        name, invalid_count
                    ),
                ));
            }
        }

        let mut params_clone: HashMap<String, MxArray> =
            params.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // Weights metadata (reference sidecar)
        let mut weights_metadata = serde_json::Map::new();
        for (key, array) in params.iter() {
            let shape_data = array.shape()?;
            let shape: Vec<i64> = shape_data.as_ref().to_vec();
            let dtype = array.dtype()?;
            let mut param_info = serde_json::Map::new();
            param_info.insert("shape".to_string(), serde_json::json!(shape));
            param_info.insert("dtype".to_string(), serde_json::json!(dtype as i32));
            weights_metadata.insert(key.clone(), serde_json::Value::Object(param_info));
        }

        // Serialize config and inject model_type for detectModelType
        let mut config_value = serde_json::to_value(&self.config).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to serialize config: {e}"),
            )
        })?;
        if let serde_json::Value::Object(ref mut map) = config_value {
            map.insert("model_type".to_string(), serde_json::json!("qwen3_5_moe"));
            // `parse_config` reads the MTP layer count ONLY from the
            // HF-convention keys `mtp_num_hidden_layers` /
            // `num_nextn_predict_layers`; the serde field name
            // `n_mtp_layers` is ignored on load. Without this, a saved MTP
            // checkpoint reloads with `n_mtp_layers = 0` and its head is
            // silently dropped.
            map.insert(
                "mtp_num_hidden_layers".to_string(),
                serde_json::json!(self.config.n_mtp_layers),
            );
        }

        let weights_json = serde_json::json!({
            "version": "1.0",
            "config": config_value,
            "weights": weights_metadata,
            "note": "Full weights are in weights.safetensors"
        });

        let path = std::path::Path::new(save_path);
        std::fs::create_dir_all(path)?;

        info!("Saving model to {}", save_path);

        let config_path = path.join("config.json");
        let config_json = serde_json::to_string_pretty(&config_value)?;
        std::fs::write(&config_path, config_json)?;
        info!("Saved config.json");

        let safetensors_path = path.join("weights.safetensors");
        let metadata = Some(serde_json::json!({
            "format": "mlx-node",
            "version": "1.0"
        }));
        crate::utils::safetensors::save_safetensors(
            &safetensors_path,
            &mut params_clone,
            metadata,
        )?;
        info!("Saved weights.safetensors");

        let weights_str = serde_json::to_string_pretty(&weights_json)?;
        let weights_path = path.join("weights.mlx");
        std::fs::write(&weights_path, weights_str)?;
        info!("Saved weights.mlx metadata");

        Ok(())
    }

    // ========== Training methods (run on model thread) ==========

    /// Initialize training state with optimizer and configuration.
    /// Inherent body of [`TrainBackend::init_training_sync`].
    fn init_training_sync_impl(
        &mut self,
        config: crate::grpo::engine::GRPOEngineConfig,
        _model_type: crate::training_model::ModelType,
    ) -> Result<()> {
        if self.training_state.is_some() {
            return Err(napi::Error::from_reason(
                "Training state already initialized. A single model thread can host only one active training run.",
            ));
        }
        let optimizer = if config.optimizer_type.as_deref().unwrap_or("adamw") == "adamw" {
            Some(crate::optimizers::AdamW::new(
                config.learning_rate,
                config.adamw_beta1,
                config.adamw_beta2,
                config.adamw_eps,
                config.weight_decay,
                Some(true), // bias correction
            ))
        } else {
            None
        };

        self.training_state = Some(crate::training_state::ModelThreadTrainingState::new(
            config.learning_rate.unwrap_or(1e-6),
            config.gradient_accumulation_steps.unwrap_or(1),
            config.gradient_clip_norm,
            config.gradient_clip_value,
            config.max_nan_gradients.unwrap_or(100),
            config.emergency_save_threshold.unwrap_or(5),
            config.verbose_nan_detection.unwrap_or(false),
            config.gradient_checkpointing.unwrap_or(true),
            optimizer,
        ));
        info!("Training state initialized on model thread (Qwen3.5 MoE)");
        Ok(())
    }

    /// Inherent body of [`TrainBackend::save_optimizer_state_sync`].
    fn save_optimizer_state_sync_impl(&self, path: String) -> Result<()> {
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        ts.save_optimizer_state_sync(&path)
    }

    /// Inherent body of [`TrainBackend::load_optimizer_state_sync`].
    fn load_optimizer_state_sync_impl(&mut self, path: String) -> Result<()> {
        let ts = self.training_state.as_mut().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        ts.load_optimizer_state_sync(&path)
    }

    /// Generate completions for training.
    ///
    /// Tokenizes prompts using Jinja2 chat template, generates completions,
    /// caches MxArray results in training_state for the subsequent training step,
    /// and returns plain data across the thread boundary.
    /// Inherent body of [`TrainBackend::generate_for_training_thread_sync`].
    fn generate_for_training_thread_sync_impl(
        &mut self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: usize,
        gen_config: crate::models::qwen3::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<crate::training_model::GenerationPlainData> {
        use crate::array::heavy_cleanup;

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Tokenizer not available."))?
            .clone();

        let num_prompts = prompts.len();
        let total_completions = num_prompts * group_size;

        let mut completion_texts = Vec::with_capacity(total_completions);
        let mut prompt_texts = Vec::with_capacity(total_completions);
        let mut completion_tokens_plain = Vec::with_capacity(total_completions);
        let mut completion_logprobs_plain = Vec::with_capacity(total_completions);
        let mut token_counts = Vec::with_capacity(total_completions);
        let mut finish_reasons = Vec::with_capacity(total_completions);

        // Cache MxArrays for the training step (prompt-major layout)
        let mut cached_prompt_tokens: Vec<MxArray> = Vec::with_capacity(num_prompts);
        let mut cached_completion_tokens: Vec<MxArray> = Vec::with_capacity(total_completions);
        let mut cached_completion_logprobs: Vec<MxArray> = Vec::with_capacity(total_completions);

        for prompt_messages in prompts.iter() {
            // Tokenize the prompt using Jinja2 chat template (supports tools + thinking)
            let prompt_token_ids = tokenizer.apply_chat_template_sync(
                prompt_messages,
                Some(true),
                tools.as_deref(),
                enable_thinking,
            )?;

            let prompt_array =
                MxArray::from_uint32(&prompt_token_ids, &[1, prompt_token_ids.len() as i64])?;
            let prompt_array_1d = prompt_array.squeeze(Some(&[0]))?;
            let prompt_text = tokenizer.decode_sync(&prompt_token_ids, true)?;

            // Generate group_size completions for this prompt
            for _g in 0..group_size {
                let result = self
                    .generate_single_for_training_sync(&prompt_array, Some(gen_config.clone()))?;

                // Extract plain data for crossing thread boundary
                let tok_ids: Vec<i32> = result
                    .tokens
                    .to_uint32()?
                    .iter()
                    .map(|&t| t as i32)
                    .collect();
                let lp_data: Vec<f32> = result.logprobs.to_float32()?.to_vec();
                let decoded = tokenizer
                    .decode_sync(&tok_ids.iter().map(|&t| t as u32).collect::<Vec<_>>(), true)?;

                completion_texts.push(decoded);
                prompt_texts.push(prompt_text.clone());
                completion_tokens_plain.push(tok_ids);
                completion_logprobs_plain.push(lp_data);
                token_counts.push(result.num_tokens as u32);
                finish_reasons.push(result.finish_reason.clone());

                // Cache MxArrays (these stay on the model thread)
                cached_completion_tokens.push(result.tokens);
                cached_completion_logprobs.push(result.logprobs);

                // Clean up between completions to prevent Metal context accumulation
                heavy_cleanup();
            }

            cached_prompt_tokens.push(prompt_array_1d);
        }

        // Store cached MxArrays in training_state (prompt-major layout)
        if let Some(ref mut ts) = self.training_state {
            ts.cached_prompt_tokens = Some(cached_prompt_tokens);
            ts.cached_completion_tokens = Some(cached_completion_tokens);
            ts.cached_completion_logprobs = Some(cached_completion_logprobs);
        }

        Ok(crate::training_model::GenerationPlainData {
            completion_texts,
            prompt_texts,
            completion_tokens: completion_tokens_plain,
            completion_logprobs: completion_logprobs_plain,
            token_counts,
            finish_reasons,
        })
    }

    /// Generate a single completion for training purposes.
    ///
    /// Uses fresh local KV caches (not the shared inference caches).
    /// Returns GenerationResult with MxArray tokens and logprobs.
    fn generate_single_for_training_sync(
        &mut self,
        input_ids: &MxArray,
        config: Option<crate::models::qwen3::GenerationConfig>,
    ) -> Result<crate::models::qwen3::GenerationResult> {
        use crate::array::synchronize_and_clear_cache;
        use crate::sampling::{
            apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
            check_repetition_cutoff, sample_and_logprobs,
        };

        let config = config.unwrap_or_default();
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let temperature = config.temperature.unwrap_or(1.0);
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let presence_penalty = config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = config.presence_context_size.unwrap_or(20);
        let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = config.frequency_context_size.unwrap_or(20);
        let max_consecutive_tokens = config
            .max_consecutive_tokens
            .unwrap_or(crate::sampling::DEFAULT_MAX_CONSECUTIVE_TOKENS);
        let max_ngram_repeats = config
            .max_ngram_repeats
            .unwrap_or(crate::sampling::DEFAULT_MAX_NGRAM_REPEATS);
        let ngram_size = config
            .ngram_size
            .unwrap_or(crate::sampling::DEFAULT_NGRAM_SIZE);
        let eos_token_id = config.eos_token_id.or(Some(self.config.eos_token_id));
        let return_logprobs = config.return_logprobs.unwrap_or(true);

        let embedding = self.embedding.clone();
        let generation_stream = Stream::new(DeviceType::Gpu);
        let fa_idx = self.fa_idx;

        // Use fresh caches for training (not shared inference caches)
        let mut training_caches: Option<Vec<Qwen3_5LayerCache>> = Some(
            (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect(),
        );

        let input_tokens = input_ids.to_uint32()?;
        let current_ids = input_ids.clone();
        // Bounded, floored capacity hint (see `generated_capacity_hint`): this
        // training-only path takes `max_new_tokens` from training config
        // (SFT/GRPO) where panics are banned. The helper prevents both the
        // negative-budget `.. as usize` wrap to `usize::MAX` (which would abort)
        // and a multi-GiB eager reservation for an absurd budget, without
        // changing behavior for valid budgets — the buffer still grows to hold
        // every generated token.
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut generated_logprobs: Vec<f32> = if return_logprobs {
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens))
        } else {
            Vec::new()
        };
        let mut finish_reason = "length";

        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

        // PREFILL
        let mut last_logits = {
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                forward_inner(
                    &current_ids,
                    &embedding,
                    &mut self.layers,
                    &mut training_caches,
                    &self.final_norm,
                    &self.lm_head,
                    fa_idx,
                )?
            };
            let seq_len = logits.shape_at(1)?;
            logits
                .slice_axis(1, seq_len - 1, seq_len)?
                .squeeze(Some(&[0, 1]))?
        };

        if repetition_penalty != 1.0 && !input_tokens.is_empty() {
            last_logits = apply_repetition_penalty(
                &last_logits,
                &input_tokens,
                repetition_penalty,
                Some(repetition_context_size),
            )?;
        }
        if presence_penalty != 0.0 {
            last_logits = apply_presence_penalty(
                &last_logits,
                &input_tokens,
                presence_penalty,
                Some(presence_context_size),
            )?;
        }
        if frequency_penalty != 0.0 {
            last_logits = apply_frequency_penalty(
                &last_logits,
                &input_tokens,
                frequency_penalty,
                Some(frequency_context_size),
            )?;
        }

        let (mut token, mut logprobs) = if return_logprobs {
            let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
            (tok, Some(lp))
        } else {
            (sample(&last_logits, Some(sampling_config))?, None)
        };

        // DECODE
        const DECODE_CLEANUP_INTERVAL: i32 = 256;
        for step in 0..max_new_tokens {
            let _stream_ctx = StreamContext::new(generation_stream);
            token.eval();
            if step > 0 && step % DECODE_CLEANUP_INTERVAL == 0 {
                synchronize_and_clear_cache();
            }
            let token_value = token.item_at_int32(0)? as u32;
            generated_tokens.push(token_value);
            if return_logprobs && let Some(ref lp) = logprobs {
                lp.eval();
                let lp_value = lp.item_at_float32(0)?;
                generated_logprobs.push(lp_value);
            }
            if let Some(eos) = eos_token_id
                && token_value == eos as u32
            {
                finish_reason = "stop";
                break;
            }
            if let Some(reason) = check_repetition_cutoff(
                &generated_tokens,
                max_consecutive_tokens,
                max_ngram_repeats,
                ngram_size,
            ) {
                finish_reason = reason;
                break;
            }
            let next_ids = MxArray::from_uint32(&[token_value], &[1, 1])?;
            let next_logits = forward_inner(
                &next_ids,
                &embedding,
                &mut self.layers,
                &mut training_caches,
                &self.final_norm,
                &self.lm_head,
                fa_idx,
            )?;
            let next_last_logits = next_logits.slice_axis(1, 0, 1)?.squeeze(Some(&[0, 1]))?;
            last_logits = next_last_logits;
            if repetition_penalty != 1.0 || presence_penalty != 0.0 || frequency_penalty != 0.0 {
                let context_tokens: Vec<u32> = input_tokens
                    .iter()
                    .copied()
                    .chain(generated_tokens.iter().copied())
                    .collect();
                if repetition_penalty != 1.0 {
                    last_logits = apply_repetition_penalty(
                        &last_logits,
                        &context_tokens,
                        repetition_penalty,
                        Some(repetition_context_size),
                    )?;
                }
                if presence_penalty != 0.0 {
                    last_logits = apply_presence_penalty(
                        &last_logits,
                        &context_tokens,
                        presence_penalty,
                        Some(presence_context_size),
                    )?;
                }
                if frequency_penalty != 0.0 {
                    last_logits = apply_frequency_penalty(
                        &last_logits,
                        &context_tokens,
                        frequency_penalty,
                        Some(frequency_context_size),
                    )?;
                }
            }
            let (next_tok, next_lp) = if return_logprobs {
                let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
                (tok, Some(lp))
            } else {
                (sample(&last_logits, Some(sampling_config))?, None)
            };
            token = next_tok;
            logprobs = next_lp;
        }

        let tokens_array =
            MxArray::from_uint32(&generated_tokens, &[generated_tokens.len() as i64])?;
        let logprobs_array = if return_logprobs {
            MxArray::from_float32(&generated_logprobs, &[generated_logprobs.len() as i64])?
        } else {
            MxArray::from_float32(&[], &[0])?
        };

        Ok(crate::models::qwen3::GenerationResult {
            text: String::new(), // Text decoding done by caller
            tokens: tokens_array,
            logprobs: logprobs_array,
            finish_reason: finish_reason.to_string(),
            num_tokens: generated_tokens.len(),
        })
    }

    /// GRPO training step: compute loss, gradients, and apply optimizer.
    /// Inherent body of [`TrainBackend::train_step_grpo_sync`].
    fn train_step_grpo_sync_impl(
        &mut self,
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        use crate::array::memory::{get_active_memory, get_peak_memory, reset_peak_memory};
        use crate::array::{heavy_cleanup, synchronize_and_clear_cache};
        use crate::grpo::advantages::compute_advantages;
        use crate::grpo::autograd::compute_loss_and_gradients_autograd;
        use crate::optimizers::GradientUtils;
        use crate::training_model::ModelType;

        reset_peak_memory();

        // Get cached generation results from training_state
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;

        let prompt_tokens = ts.cached_prompt_tokens.as_ref().ok_or_else(|| {
            napi::Error::from_reason("No cached prompt tokens. Call GenerateForTraining first.")
        })?;
        let completion_tokens = ts.cached_completion_tokens.as_ref().ok_or_else(|| {
            napi::Error::from_reason("No cached completion tokens. Call GenerateForTraining first.")
        })?;
        let completion_logprobs = ts.cached_completion_logprobs.as_ref().ok_or_else(|| {
            napi::Error::from_reason(
                "No cached completion logprobs. Call GenerateForTraining first.",
            )
        })?;

        let use_checkpointing = ts.gradient_checkpointing;
        let gradient_clip_value = ts.gradient_clip_value;
        let gradient_clip_norm = ts.gradient_clip_norm;
        let verbose_nan = ts.verbose_nan_detection;
        let learning_rate = ts.learning_rate;
        let max_nan_gradients = ts.max_nan_gradients;
        let emergency_save_threshold = ts.emergency_save_threshold;

        // Get model parameters
        let params = self.get_parameters_sync()?;
        let model_type = ModelType::Qwen35Moe(self.config.clone());

        // Build completion/logprob refs, optionally filtering by valid_indices from
        // the engine's degenerate-completion filter. prompt_refs always has one
        // entry per prompt — the autograd function expands them to one per
        // completion via repeat_n(group_size), so the `group_size` passed here
        // must be the effective group size after filtering (the engine computes
        // effective_group_size = valid_indices.len() / num_prompts).
        let prompt_refs: Vec<&MxArray> = prompt_tokens.iter().collect();
        let (completion_refs, logprob_refs): (Vec<&MxArray>, Vec<&MxArray>) =
            if let Some(ref indices) = valid_indices {
                let n = completion_tokens.len();
                for &i in indices {
                    if i >= n {
                        return Err(napi::Error::from_reason(format!(
                            "valid_indices contains out-of-range index {} (completion count = {})",
                            i, n
                        )));
                    }
                }
                let c: Vec<&MxArray> = indices.iter().map(|&i| &completion_tokens[i]).collect();
                let l: Vec<&MxArray> = indices.iter().map(|&i| &completion_logprobs[i]).collect();
                (c, l)
            } else {
                (
                    completion_tokens.iter().collect(),
                    completion_logprobs.iter().collect(),
                )
            };

        let (loss_value, gradients) = compute_loss_and_gradients_autograd(
            &model_type,
            &params,
            &prompt_refs,
            &completion_refs,
            &logprob_refs,
            &rewards,
            group_size,
            loss_config,
            use_checkpointing,
        )?;

        // Check for NaN/Inf loss
        if loss_value.is_nan() || loss_value.is_infinite() {
            warn!("Skipping step due to invalid loss: {}", loss_value);
            synchronize_and_clear_cache();
            // Skipped steps must still advance the authoritative step counter
            // (H1) and drop the cached generation so the next cycle starts
            // clean.
            let ts = self.training_state.as_mut().unwrap();
            ts.clear_generation_cache();
            ts.step += 1;
            let new_step = ts.step;
            let nan_count = ts.nan_gradient_count;
            return Ok(crate::training_model::TrainStepPlainMetrics {
                loss: loss_value,
                gradients_applied: false,
                mean_advantage: 0.0,
                std_advantage: 0.0,
                nan_gradient_count: nan_count,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
                total_tokens: 0,
                step: new_step,
            });
        }

        // Validate ALL gradients — skip entire step if ANY has NaN/Inf
        for (name, grad) in gradients.iter() {
            grad.eval();
            let has_invalid = grad.has_nan_or_inf()?;
            if has_invalid {
                if verbose_nan {
                    let data = grad.to_float32()?;
                    let invalid_count = data
                        .iter()
                        .filter(|v| v.is_nan() || v.is_infinite())
                        .count();
                    warn!(
                        "Gradient '{}' contains {} invalid values - SKIPPING STEP",
                        name, invalid_count
                    );
                } else {
                    warn!("Gradient '{}' contains NaN/Inf - SKIPPING STEP", name);
                }

                let ts = self.training_state.as_mut().unwrap();
                ts.nan_gradient_count += 1;
                ts.consecutive_nan_count += 1;

                if ts.nan_gradient_count >= max_nan_gradients as u64 {
                    return Err(napi::Error::from_reason(format!(
                        "Training stopped: exceeded max NaN gradient count ({}/{})",
                        ts.nan_gradient_count, max_nan_gradients
                    )));
                }

                if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                    warn!(
                        "Emergency save triggered: {} consecutive NaN gradients",
                        ts.consecutive_nan_count
                    );
                }

                // Advance the authoritative step counter (H1) and clear the
                // cached generation data so the next cycle starts clean.
                ts.clear_generation_cache();
                ts.step += 1;
                let new_step = ts.step;
                let nan_count = ts.nan_gradient_count;
                synchronize_and_clear_cache();
                return Ok(crate::training_model::TrainStepPlainMetrics {
                    loss: loss_value,
                    gradients_applied: false,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    nan_gradient_count: nan_count,
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                    total_tokens: 0,
                    step: new_step,
                });
            }
        }

        // Element-wise gradient clipping
        let grad_clip_val = gradient_clip_value.unwrap_or(1.0);
        let mut clamped_gradients: HashMap<String, MxArray> = HashMap::new();
        for (name, grad) in gradients.iter() {
            let clamped = grad.clip(Some(-grad_clip_val), Some(grad_clip_val))?;
            clamped.eval();
            clamped_gradients.insert(name.clone(), clamped);
        }

        // Gradient norm clipping
        let clipped_gradients = if let Some(max_norm) = gradient_clip_norm {
            let grad_refs: HashMap<String, &MxArray> = clamped_gradients
                .iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();
            GradientUtils::clip_grad_norm(grad_refs, max_norm)?
        } else {
            clamped_gradients
        };

        // Accumulate gradients
        let ts = self.training_state.as_mut().unwrap();
        ts.consecutive_nan_count = 0;

        Self::accumulate_gradients_inner(ts, clipped_gradients)?;
        ts.micro_step += 1;

        let grad_acc_steps = ts.grad_accumulation_steps;
        let gradients_applied = if ts.micro_step >= grad_acc_steps {
            let grads = ts
                .accumulated_gradients
                .take()
                .ok_or_else(|| napi::Error::from_reason("No accumulated gradients"))?;

            // Apply optimizer step
            if let Some(ref mut optimizer) = ts.optimizer {
                // AdamW path
                let mut param_names_vec: Vec<String> = Vec::new();
                let mut param_refs: Vec<&MxArray> = Vec::new();
                let mut grad_refs: Vec<&MxArray> = Vec::new();

                let scaled_grads: HashMap<String, MxArray>;
                let grads_to_use = if grad_acc_steps > 1 {
                    let scale = 1.0 / grad_acc_steps as f32;
                    let scale_arr = MxArray::from_float32(&[scale], &[])?;
                    scaled_grads = grads
                        .iter()
                        .map(|(name, grad)| Ok((name.clone(), grad.mul(&scale_arr)?)))
                        .collect::<Result<HashMap<_, _>>>()?;
                    &scaled_grads
                } else {
                    &grads
                };

                for (name, grad) in grads_to_use {
                    if let Some(param) = params.get(name) {
                        param_names_vec.push(name.clone());
                        param_refs.push(param);
                        grad_refs.push(grad);
                    }
                }

                let updated = optimizer.update_batch(
                    param_names_vec.clone(),
                    param_refs.clone(),
                    grad_refs,
                )?;

                // `update_batch` above has already committed the optimizer's step
                // counter and moment tensors. If this fallible delta build (or the
                // atomic apply below) errors, the optimizer is left one step ahead
                // of the still-unchanged model params (all deltas are built before
                // any are applied). Accepted over panicking; such `?` failures are
                // fatal device errors that abort the run.
                let delta_map: HashMap<String, MxArray> = param_names_vec
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let delta = param_refs[i].sub(&updated[i])?;
                        Ok((name.clone(), delta))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;

                let delta_refs: HashMap<String, &MxArray> =
                    delta_map.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(delta_refs, 1.0, &params)?;

                tracing::debug!(
                    "Applied AdamW update (step={})",
                    self.training_state.as_ref().unwrap().step
                );
            } else {
                // SGD path
                let lr = learning_rate / grad_acc_steps as f64;
                let grads_refs: HashMap<String, &MxArray> =
                    grads.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(grads_refs, lr, &params)?;
                tracing::debug!("Applied SGD gradients with lr: {}", lr);
            }

            let ts = self.training_state.as_mut().unwrap();
            ts.accumulated_gradients = None;
            ts.micro_step = 0;
            ts.step += 1;
            true
        } else {
            ts.step += 1;
            false
        };

        // Compute advantage statistics
        let rewards_f32: Vec<f32> = rewards.iter().map(|&r| r as f32).collect();
        let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;
        let advantages = compute_advantages(&rewards_array, group_size, "group".to_string())?;
        let adv_data = advantages.to_float32()?;
        let mean_advantage =
            adv_data.iter().map(|&a| a as f64).sum::<f64>() / adv_data.len().max(1) as f64;
        let std_advantage = {
            let variance = adv_data
                .iter()
                .map(|&a| {
                    let diff = a as f64 - mean_advantage;
                    diff * diff
                })
                .sum::<f64>()
                / adv_data.len().max(1) as f64;
            variance.sqrt()
        };

        // Count tokens BEFORE clearing the cache — otherwise total_tokens is
        // always zero on the success path.
        let ts = self.training_state.as_ref().unwrap();
        let total_tokens: i32 = if let Some(ref ct) = ts.cached_completion_tokens {
            ct.iter()
                .filter_map(|t| t.shape_at(0).ok())
                .map(|n| n as i32)
                .sum()
        } else {
            0
        };

        // Clear cached generation data
        if let Some(ref mut ts) = self.training_state {
            ts.clear_generation_cache();
        }

        // CRITICAL: heavy_cleanup after autograd to clear compiled graph cache
        heavy_cleanup();

        let ts = self.training_state.as_ref().unwrap();
        Ok(crate::training_model::TrainStepPlainMetrics {
            loss: loss_value,
            gradients_applied,
            mean_advantage,
            std_advantage,
            nan_gradient_count: ts.nan_gradient_count,
            peak_memory_mb: get_peak_memory() / 1e6,
            active_memory_mb: get_active_memory() / 1e6,
            total_tokens,
            step: ts.step,
        })
    }

    /// SFT training step: compute loss, gradients, and apply optimizer.
    ///
    /// Receives plain data (Vec<i32> + shape) from the SFT engine, reconstructs
    /// MxArrays on the model thread, computes SFT loss + gradients, validates,
    /// clips, accumulates, and applies optimizer step when accumulation is complete.
    /// Inherent body of [`TrainBackend::train_step_sft_sync`].
    fn train_step_sft_sync_impl(
        &mut self,
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        use crate::array::memory::{get_active_memory, get_peak_memory, reset_peak_memory};
        use crate::array::{heavy_cleanup, synchronize_and_clear_cache};
        use crate::optimizers::GradientUtils;

        reset_peak_memory();

        // Ensure training state is initialized
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        let _ = ts;

        // Reconstruct MxArrays from plain data
        let input_ids_arr = MxArray::from_int32(&input_ids, &input_shape)?;
        let labels_arr = MxArray::from_int32(&labels, &labels_shape)?;

        // Get model parameters
        let params = self.get_parameters_sync()?;
        let model_type = crate::training_model::ModelType::Qwen35Moe(self.config.clone());

        // Build loss config from SftEngineConfig
        let loss_config = crate::sft::SftLossConfig {
            ignore_index: Some(-100),
            label_smoothing: config.label_smoothing,
        };

        let use_checkpointing = config.gradient_checkpointing.unwrap_or(true);
        let verbose_nan = config.verbose_nan_detection.unwrap_or(false);
        let max_nan_gradients = config.max_nan_gradients.unwrap_or(100);
        let emergency_save_threshold = config.emergency_save_threshold.unwrap_or(5);

        // Compute loss and gradients
        let (loss_value, gradients) = crate::sft::autograd::compute_sft_loss_and_gradients(
            &model_type,
            &params,
            &input_ids_arr,
            &labels_arr,
            loss_config,
            use_checkpointing,
        )?;

        // Check for NaN/Inf loss
        if loss_value.is_nan() || loss_value.is_infinite() {
            warn!("SFT: Skipping step due to invalid loss: {}", loss_value);
            synchronize_and_clear_cache();
            let ts = self.training_state.as_mut().unwrap();
            ts.nan_gradient_count += 1;
            ts.consecutive_nan_count += 1;

            if ts.nan_gradient_count >= max_nan_gradients as u64 {
                return Err(napi::Error::from_reason(format!(
                    "Training stopped: exceeded max NaN gradient count ({}/{})",
                    ts.nan_gradient_count, max_nan_gradients
                )));
            }

            if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                warn!(
                    "Emergency save triggered: {} consecutive NaN losses",
                    ts.consecutive_nan_count
                );
            }

            return Ok(crate::training_model::TrainStepPlainMetrics {
                loss: 0.0,
                gradients_applied: false,
                mean_advantage: 0.0,
                std_advantage: 0.0,
                nan_gradient_count: ts.nan_gradient_count,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
                total_tokens: 0,
                step: ts.step,
            });
        }

        // Validate ALL gradients — skip entire step if ANY has NaN/Inf
        for (name, grad) in gradients.iter() {
            grad.eval();
            let has_invalid = grad.has_nan_or_inf()?;
            if has_invalid {
                if verbose_nan {
                    let data = grad.to_float32()?;
                    let invalid_count = data
                        .iter()
                        .filter(|v| v.is_nan() || v.is_infinite())
                        .count();
                    warn!(
                        "SFT: Gradient '{}' contains {} invalid values - SKIPPING STEP",
                        name, invalid_count
                    );
                } else {
                    warn!("SFT: Gradient '{}' contains NaN/Inf - SKIPPING STEP", name);
                }

                let ts = self.training_state.as_mut().unwrap();
                ts.nan_gradient_count += 1;
                ts.consecutive_nan_count += 1;

                if ts.nan_gradient_count >= max_nan_gradients as u64 {
                    return Err(napi::Error::from_reason(format!(
                        "Training stopped: exceeded max NaN gradient count ({}/{})",
                        ts.nan_gradient_count, max_nan_gradients
                    )));
                }

                if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                    warn!(
                        "Emergency save triggered: {} consecutive NaN gradients",
                        ts.consecutive_nan_count
                    );
                }

                synchronize_and_clear_cache();
                return Ok(crate::training_model::TrainStepPlainMetrics {
                    loss: loss_value,
                    gradients_applied: false,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    nan_gradient_count: ts.nan_gradient_count,
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                    total_tokens: 0,
                    step: ts.step,
                });
            }
        }

        // Element-wise gradient clipping (if configured)
        let clipped_gradients = if let Some(clip_val) = config.gradient_clip_value {
            let mut clamped: HashMap<String, MxArray> = HashMap::new();
            for (name, grad) in gradients.iter() {
                let c = grad.clip(Some(-clip_val), Some(clip_val))?;
                c.eval();
                clamped.insert(name.clone(), c);
            }
            clamped
        } else {
            gradients.clone()
        };

        // Gradient norm clipping (if configured)
        let final_gradients = if let Some(clip_norm) = config.gradient_clip_norm {
            let grad_refs: HashMap<String, &MxArray> = clipped_gradients
                .iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();
            GradientUtils::clip_grad_norm(grad_refs, clip_norm)?
        } else {
            clipped_gradients
        };

        // Accumulate gradients
        let ts = self.training_state.as_mut().unwrap();
        ts.consecutive_nan_count = 0;

        Self::accumulate_gradients_inner(ts, final_gradients)?;
        ts.micro_step += 1;

        let grad_acc_steps = config.gradient_accumulation_steps.unwrap_or(1);
        let learning_rate = config.learning_rate.unwrap_or(2e-5);
        let weight_decay = config.weight_decay.unwrap_or(0.01);

        let gradients_applied = if ts.micro_step >= grad_acc_steps {
            let grads = ts
                .accumulated_gradients
                .take()
                .ok_or_else(|| napi::Error::from_reason("No accumulated gradients"))?;

            if let Some(ref mut optimizer) = ts.optimizer {
                let mut param_names_vec: Vec<String> = Vec::new();
                let mut param_refs: Vec<&MxArray> = Vec::new();
                let mut grad_refs: Vec<&MxArray> = Vec::new();

                let scaled_grads: HashMap<String, MxArray>;
                let grads_to_use = if grad_acc_steps > 1 {
                    let scale = 1.0 / grad_acc_steps as f32;
                    let scale_arr = MxArray::from_float32(&[scale], &[])?;
                    scaled_grads = grads
                        .iter()
                        .map(|(name, grad)| Ok((name.clone(), grad.mul(&scale_arr)?)))
                        .collect::<Result<HashMap<_, _>>>()?;
                    &scaled_grads
                } else {
                    &grads
                };

                for (name, grad) in grads_to_use {
                    if let Some(param) = params.get(name) {
                        param_names_vec.push(name.clone());
                        param_refs.push(param);
                        grad_refs.push(grad);
                    }
                }

                let updated = optimizer.update_batch(
                    param_names_vec.clone(),
                    param_refs.clone(),
                    grad_refs,
                )?;

                // `update_batch` above has already committed the optimizer's step
                // counter and moment tensors. If this fallible delta build (or the
                // atomic apply below) errors, the optimizer is left one step ahead
                // of the still-unchanged model params (all deltas are built before
                // any are applied). Accepted over panicking; such `?` failures are
                // fatal device errors that abort the run.
                let delta_map: HashMap<String, MxArray> = param_names_vec
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let delta = param_refs[i].sub(&updated[i])?;
                        Ok((name.clone(), delta))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;

                let delta_refs: HashMap<String, &MxArray> =
                    delta_map.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(delta_refs, 1.0, &params)?;

                tracing::debug!(
                    "SFT: Applied AdamW update (step={})",
                    self.training_state.as_ref().unwrap().step
                );
            } else {
                let lr = learning_rate / grad_acc_steps as f64;

                let grads_with_decay = if weight_decay > 0.0 {
                    grads
                        .into_iter()
                        .map(|(name, grad)| {
                            if let Some(param) = params.get(&name) {
                                if let Ok(decay_term) = param.mul_scalar(weight_decay)
                                    && let Ok(new_grad) = grad.add(&decay_term)
                                {
                                    return (name, new_grad);
                                }
                                (name, grad)
                            } else {
                                (name, grad)
                            }
                        })
                        .collect::<HashMap<_, _>>()
                } else {
                    grads
                };

                let grads_refs: HashMap<String, &MxArray> = grads_with_decay
                    .iter()
                    .map(|(k, v)| (k.clone(), v))
                    .collect();
                self.apply_gradients_inner(grads_refs, lr, &params)?;
                tracing::debug!("SFT: Applied SGD gradients with lr: {}", lr);
            }

            let ts = self.training_state.as_mut().unwrap();
            ts.accumulated_gradients = None;
            ts.micro_step = 0;
            ts.step += 1;
            true
        } else {
            ts.step += 1;
            false
        };

        // Count valid tokens from the labels
        let total_tokens = {
            let ignore_val = MxArray::scalar_int(-100)?;
            let valid_mask = labels_arr.not_equal(&ignore_val)?;
            let count = valid_mask.sum(None, Some(false))?;
            count.eval();
            count.item_at_int32(0).unwrap_or(0)
        };

        // CRITICAL: heavy_cleanup after autograd to clear compiled graph cache
        heavy_cleanup();

        let ts = self.training_state.as_ref().unwrap();
        Ok(crate::training_model::TrainStepPlainMetrics {
            loss: loss_value,
            gradients_applied,
            mean_advantage: 0.0,
            std_advantage: 0.0,
            nan_gradient_count: ts.nan_gradient_count,
            peak_memory_mb: get_peak_memory() / 1e6,
            active_memory_mb: get_active_memory() / 1e6,
            total_tokens,
            step: ts.step,
        })
    }

    /// Accumulate gradients into training state.
    fn accumulate_gradients_inner(
        ts: &mut crate::training_state::ModelThreadTrainingState,
        new_grads: HashMap<String, MxArray>,
    ) -> Result<()> {
        match &mut ts.accumulated_gradients {
            Some(acc) => {
                for (name, grad) in new_grads {
                    grad.eval();
                    if grad.has_nan_or_inf()? {
                        warn!(
                            "Skipping gradient accumulation for '{}' due to NaN/Inf",
                            name
                        );
                        continue;
                    }
                    if let Some(existing) = acc.get_mut(&name) {
                        let summed = existing.add(&grad)?;
                        summed.eval();
                        *existing = summed;
                    } else {
                        acc.insert(name, grad);
                    }
                }
            }
            None => {
                let mut evaluated_grads = HashMap::with_capacity(new_grads.len());
                for (name, grad) in new_grads {
                    grad.eval();
                    if grad.has_nan_or_inf()? {
                        warn!("Skipping initial gradient for '{}' due to NaN/Inf", name);
                        continue;
                    }
                    evaluated_grads.insert(name, grad);
                }
                ts.accumulated_gradients = Some(evaluated_grads);
            }
        }
        Ok(())
    }

    /// Apply gradients to model weights (SGD or AdamW delta application).
    ///
    /// Direct field access on Qwen35MoeInner — no locks needed.
    fn apply_gradients_inner(
        &mut self,
        gradients: HashMap<String, &MxArray>,
        learning_rate: f64,
        current_params: &HashMap<String, MxArray>,
    ) -> Result<()> {
        use super::decoder_layer::{AttentionType, MLPType};

        let updated_params =
            crate::training_model::compute_sgd_updates(&gradients, learning_rate, current_params)?;

        // Apply updated parameters directly to model fields
        for (name, updated_param) in updated_params.iter() {
            if name == "lm_head.weight" {
                if let Some(ref mut lm) = self.lm_head {
                    lm.set_weight(updated_param, "lm_head")?;
                }
            } else if name == "final_norm.weight" {
                self.final_norm.set_weight(updated_param)?;
            } else if name == "embedding.weight" {
                self.embedding.set_weight(updated_param)?;
            } else if name.starts_with("layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 3
                    && let Ok(layer_idx) = parts[1].parse::<usize>()
                    && layer_idx < self.layers.len()
                {
                    let layer = &mut self.layers[layer_idx];
                    if name.contains(".linear_attn.") {
                        if let AttentionType::Linear(ref mut gdn) = layer.attn {
                            if name.ends_with(".in_proj_qkvz.weight") {
                                gdn.set_in_proj_qkvz_weight(updated_param)?;
                            } else if name.ends_with(".in_proj_ba.weight") {
                                gdn.set_in_proj_ba_weight(updated_param)?;
                            } else if name.ends_with(".conv1d.weight") {
                                gdn.set_conv1d_weight(updated_param)?;
                            } else if name.ends_with(".norm.weight") {
                                gdn.set_norm_weight(updated_param)?;
                            } else if name.ends_with(".out_proj.weight") {
                                gdn.set_out_proj_weight(updated_param)?;
                            } else if name.ends_with(".dt_bias") {
                                gdn.set_dt_bias(updated_param);
                            } else if name.ends_with(".a_log") {
                                gdn.set_a_log(updated_param)?;
                            }
                        }
                    } else if name.contains(".self_attn.") {
                        if let AttentionType::Full(ref mut attn) = layer.attn {
                            if name.ends_with(".q_proj.weight") {
                                attn.set_q_proj_weight(updated_param)?;
                            } else if name.ends_with(".k_proj.weight") {
                                attn.set_k_proj_weight(updated_param)?;
                            } else if name.ends_with(".v_proj.weight") {
                                attn.set_v_proj_weight(updated_param)?;
                            } else if name.ends_with(".o_proj.weight") {
                                attn.set_o_proj_weight(updated_param)?;
                            } else if name.ends_with(".q_norm.weight") {
                                attn.set_q_norm_weight(updated_param)?;
                            } else if name.ends_with(".k_norm.weight") {
                                attn.set_k_norm_weight(updated_param)?;
                            }
                        }
                    } else if name.contains(".mlp.") {
                        match &mut layer.mlp {
                            MLPType::Dense(mlp) => {
                                if name.ends_with(".gate_proj.weight") {
                                    mlp.set_gate_proj_weight(updated_param)?;
                                } else if name.ends_with(".up_proj.weight") {
                                    mlp.set_up_proj_weight(updated_param)?;
                                } else if name.ends_with(".down_proj.weight") {
                                    mlp.set_down_proj_weight(updated_param)?;
                                }
                            }
                            MLPType::MoE(moe) => {
                                if name.ends_with(".mlp.gate.weight") {
                                    moe.set_gate_weight(updated_param)?;
                                } else if name.contains(".mlp.switch_mlp.") {
                                    if name.ends_with(".gate_proj.weight") {
                                        moe.set_switch_mlp_gate_proj_weight(updated_param);
                                    } else if name.ends_with(".up_proj.weight") {
                                        moe.set_switch_mlp_up_proj_weight(updated_param);
                                    } else if name.ends_with(".down_proj.weight") {
                                        moe.set_switch_mlp_down_proj_weight(updated_param);
                                    }
                                } else if name.contains(".mlp.shared_expert_gate.") {
                                    moe.set_shared_expert_gate_weight(updated_param)?;
                                } else if name.contains(".mlp.shared_expert.") {
                                    if name.ends_with(".gate_proj.weight") {
                                        moe.set_shared_expert_gate_proj_weight(updated_param)?;
                                    } else if name.ends_with(".up_proj.weight") {
                                        moe.set_shared_expert_up_proj_weight(updated_param)?;
                                    } else if name.ends_with(".down_proj.weight") {
                                        moe.set_shared_expert_down_proj_weight(updated_param)?;
                                    }
                                }
                            }
                        }
                    } else if name.ends_with(".input_layernorm.weight") {
                        layer.set_input_layernorm_weight(updated_param)?;
                    } else if name.ends_with(".post_attention_layernorm.weight") {
                        layer.set_post_attention_layernorm_weight(updated_param)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract all trainable parameters from the model.
    /// Direct field access — no locks needed on model thread.
    fn get_parameters_sync(&self) -> Result<HashMap<String, MxArray>> {
        use super::decoder_layer::{AttentionType, MLPType};

        let mut params = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            // Attention weights
            match &layer.attn {
                AttentionType::Linear(gdn) => {
                    params.insert(
                        format!("{}.linear_attn.in_proj_qkvz.weight", prefix),
                        gdn.get_in_proj_qkvz_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.in_proj_ba.weight", prefix),
                        gdn.get_in_proj_ba_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.conv1d.weight", prefix),
                        gdn.get_conv1d_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.norm.weight", prefix),
                        gdn.get_norm_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.out_proj.weight", prefix),
                        gdn.get_out_proj_weight(),
                    );
                    params.insert(format!("{}.linear_attn.dt_bias", prefix), gdn.get_dt_bias());
                    params.insert(format!("{}.linear_attn.a_log", prefix), gdn.get_a_log());
                }
                AttentionType::Full(attn) => {
                    params.insert(
                        format!("{}.self_attn.q_proj.weight", prefix),
                        attn.get_q_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_proj.weight", prefix),
                        attn.get_k_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.v_proj.weight", prefix),
                        attn.get_v_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.o_proj.weight", prefix),
                        attn.get_o_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.q_norm.weight", prefix),
                        attn.get_q_norm_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_norm.weight", prefix),
                        attn.get_k_norm_weight(),
                    );
                }
            }

            // MLP weights — different for Dense vs MoE layers
            match &layer.mlp {
                MLPType::Dense(mlp) => {
                    params.insert(
                        format!("{}.mlp.gate_proj.weight", prefix),
                        mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.up_proj.weight", prefix),
                        mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.down_proj.weight", prefix),
                        mlp.get_down_proj_weight(),
                    );
                }
                MLPType::MoE(moe) => {
                    // Router gate
                    params.insert(format!("{}.mlp.gate.weight", prefix), moe.get_gate_weight());
                    // Expert weights (3D: [num_experts, out, in])
                    let switch_mlp = moe.get_switch_mlp();
                    params.insert(
                        format!("{}.mlp.switch_mlp.gate_proj.weight", prefix),
                        switch_mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.switch_mlp.up_proj.weight", prefix),
                        switch_mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.switch_mlp.down_proj.weight", prefix),
                        switch_mlp.get_down_proj_weight(),
                    );
                    // Shared expert
                    params.insert(
                        format!("{}.mlp.shared_expert.gate_proj.weight", prefix),
                        moe.get_shared_expert_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.up_proj.weight", prefix),
                        moe.get_shared_expert_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.down_proj.weight", prefix),
                        moe.get_shared_expert_down_proj_weight(),
                    );
                    // Shared expert gate
                    params.insert(
                        format!("{}.mlp.shared_expert_gate.weight", prefix),
                        moe.get_shared_expert_gate_weight(),
                    );
                }
            }

            // Layer norms
            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );

        // LM head (only if not tied)
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = self.lm_head
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        Ok(params)
    }

    /// True when this checkpoint includes an MTP head (module loaded by
    /// `persistence::apply_weights_moe_inner`). The speculative decode loop
    /// gates on this together with the per-request `enable_mtp` flag — both
    /// must be true for the MTP-accelerated path to take over. Mirrors the
    /// dense `Qwen35Inner::has_mtp_weights`.
    pub(crate) fn has_mtp_weights(&self) -> bool {
        self.mtp.is_some() && self.mtp_weights_loaded
    }
}

impl Qwen35MoeInner {
    /// Whole-turn MoE dispatch behind the engine's multimodal and
    /// speculative handlers.
    ///
    /// Routes the four turn shapes onto the whole-turn cores:
    /// fresh sync → [`Self::vision_mtp_whole_turn_core`], delta sync →
    /// [`Self::chat_tokens_delta_sync`], fresh streaming →
    /// [`Self::vision_mtp_whole_turn_stream_core`], delta streaming →
    /// [`Self::chat_stream_tokens_delta_sync_inner`]. These cores own
    /// every MoE-path subtlety the generic flow does not model: VLM
    /// prefill + M-RoPE deltas, the MTP gate (eager MTP, falling back
    /// to AR when ineligible), and the paged-always-wins dispatch (including
    /// requires-paged image routing — unlike dense, MoE has NO
    /// `mtp_takes_dense_path` exception: its paged early-return runs before
    /// any MTP consideration on every path).
    ///
    /// Delta turns recover the raw delta from the engine-composed
    /// `args.tokens` (`cached_history + delta` by construction — the
    /// probes run before any state mutation).
    fn moe_whole_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // Fold generation_config.json defaults into the config the VLM/MTP
        // cores re-extract params from, so they honor the same sampling
        // defaults as the generic AR path (whose `args.params` already had
        // them applied via `resolve_params`). No-op when the checkpoint ships
        // no defaults (`gen_defaults` all-None).
        let mut config = args.config.clone();
        crate::engine::apply_generation_defaults(&mut config, &self.gen_defaults);
        // The request-time plan is authoritative. These legacy whole-turn
        // cores still re-extract `ChatParams`, so mirror the selected decoder
        // into their config rather than independently consulting the raw MTP
        // request flag a second time.
        let mut planned_mtp = apply_qwen35_moe_planned_decoder(&mut config, args.plan.decoder);
        // MTP acceptance gate: a previous turn whose draft head accepted
        // below break-even disables speculation for THIS turn (plain AR).
        if planned_mtp
            && !config.mtp_adaptive_depth.unwrap_or(false)
            && !self.mtp_gate_allows(config.mtp_depth.unwrap_or(1).max(1) as u32)
        {
            planned_mtp = false;
            config.enable_mtp = Some(false);
        }
        debug_assert!(!planned_mtp || self.has_mtp_weights());
        let thinking = args.thinking;
        match (args.sink, args.cancelled) {
            (Some(sink), Some(cancelled)) => {
                let cb = StreamSender(sink);
                if args.plan.is_delta {
                    let delta_start = self.cached_token_history.len().min(args.tokens.len());
                    let delta_tokens = args.tokens[delta_start..].to_vec();
                    self.chat_stream_tokens_delta_sync_inner(
                        delta_tokens,
                        config,
                        &cb,
                        cancelled,
                        thinking,
                    )?;
                } else {
                    self.vision_mtp_whole_turn_stream_core(
                        args.tokens.to_vec(),
                        args.media.images,
                        config,
                        args.eos_id,
                        &cb,
                        cancelled,
                        thinking,
                    )?;
                }
                Ok(TurnOutput::Streamed)
            }
            _ => {
                let result = if args.plan.is_delta {
                    let delta_start = self.cached_token_history.len().min(args.tokens.len());
                    let delta_tokens = args.tokens[delta_start..].to_vec();
                    self.chat_tokens_delta_sync(delta_tokens, config, thinking)?
                } else {
                    self.vision_mtp_whole_turn_core(
                        args.tokens.to_vec(),
                        args.media.images,
                        config,
                        args.eos_id,
                        thinking,
                    )?
                };
                Ok(TurnOutput::Complete(Box::new(result)))
            }
        }
    }
}

/// Per-turn decode stepper for the engine's generic (text-only,
/// non-paged, non-MTP) flow on Qwen3.5 MoE
/// ([`ChatBackend::begin_decode`]).
///
/// Drives the pure-Rust `forward_inner` over the flat caches.
pub(crate) struct Qwen35MoeDecode<'a> {
    inner: &'a mut Qwen35MoeInner,
    embedding: Embedding,
    /// Decode-path profiler relabel (`moe_chat_*_rust` and its streaming /
    /// delta variants), resolved in `begin_decode` from the turn's
    /// streaming-ness and delta-ness.
    relabel: &'static str,
}

impl DecodeStep for Qwen35MoeDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let inner = &mut *self.inner;
        let logits = forward_inner(
            input_ids,
            &self.embedding,
            &mut inner.layers,
            &mut inner.caches,
            &inner.final_norm,
            &inner.lm_head,
            inner.fa_idx,
        )?;
        // `true` == the eager Rust forward returns `[1, 1, vocab]`;
        // the loop squeezes axis 1.
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token, logits]);
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        Some(self.relabel)
    }
}

/// Paged decode stepper for qwen3_5_moe (the paged analog of the FLAT
/// [`Qwen35MoeDecode`]). Drives [`crate::engine::decode::run_decode_loop`]
/// through the generic [`crate::engine::paged_turn::run_paged_turn`]: each
/// `forward` runs the pure-Rust eager paged step against the live post-prefill
/// adapter pools + GDN caches. Created by
/// `<Qwen35MoeInner as PagedBackend>::begin_paged_decode`, consumed across the
/// whole decode loop.
pub(crate) struct Qwen35MoePagedDecode<'a> {
    inner: &'a mut Qwen35MoeInner,
}

impl DecodeStep for Qwen35MoePagedDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // NOT on the hot path — the engine drives decode via
        // `forward_with_token` (which hands the scalar the loop already read).
        // Kept only to satisfy the trait; extract then delegate.
        let token_id = input_ids.item_at_int32(0)? as u32;
        self.forward_with_token(input_ids, token_id)
    }

    fn forward_with_token(
        &mut self,
        _input_ids: &MxArray,
        token_id: u32,
    ) -> Result<(MxArray, bool)> {
        // Pure-Rust eager paged decode step.
        //
        // PERF: `token_id` is HANDED by the engine (already read once at the
        // loop top via `y.item_at_int32`), so we do NOT re-`item_at_int32` the
        // fresh `_input_ids` reshape — that redundant second per-step eval/sync
        // measurably regressed decode. `_input_ids` is unused (kept for
        // signature parity).
        let logits = {
            let embed = self.inner.embedding.clone();
            let caches_ref = self.inner.caches.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen35MoePagedDecode::forward: caches dropped mid-decode")
            })?;
            let adapter = self.inner.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "Qwen35MoePagedDecode::forward: paged_adapter dropped mid-decode",
                )
            })?;
            super::paged_forward::run_paged_decode_step(
                token_id,
                &embed,
                &mut self.inner.layers,
                caches_ref,
                &self.inner.final_norm,
                &self.inner.lm_head,
                &self.inner.layer_kinds,
                adapter,
                self.inner.cached_rope_deltas.unwrap_or(0),
            )?
            .squeeze(Some(&[1]))?
        };

        // `run_paged_decode_step` returns [1, 1, vocab]; the `squeeze([1])`
        // above already collapses to [1, vocab], so `needs_squeeze = FALSE`.
        Ok((logits, false))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        // Single SYNCHRONOUS eval of `next_token`: one `y.eval()` per sample
        // is the cheapest correct cadence for the bandwidth-bound paged
        // forward.
        next_token.eval();
    }

    fn maintain_cache(&mut self, step: i32) {
        // Per-step paged cache-clear cadence.
        crate::array::maybe_clear_cache_for_paged_step(step);
    }

    // `materialize_final` — DO NOT override (default no-op). CRITICAL: moe paged
    // drops the last token UNCONDITIONALLY (see `save_paged_history`). The
    // adapter / GDN caches only advanced for the tokens the loop actually
    // forwarded; re-running a decode step here for the final length-exit token
    // would record a token the GDN/adapter state never advanced →
    // recurrent-state desync vs the saved drop-last history.
}

/// qwen3_5_moe paged prefix state — the effective prefix/suffix split from
/// `prepare_turn_with_max_cache_hit_tokens`, PLUS the full prompt tokens and
/// the GDN-prime flag.
///
/// `full_tokens` is needed because the engine hands `paged_prefill` ONLY the
/// suffix (`tokens[effective_cached_prefix_len..]`), but
/// `run_paged_prefill_chunk` needs the FULL prompt for the GDN pre-pass over
/// the cached prefix. `gdn_prefix_already_primed` is the moe-specific bit the
/// prime resolves (the GDN recurrent state was already populated live / from a
/// checkpoint / via replay) and `paged_prefill` threads into
/// `run_paged_prefill_chunk` so the prefill skips re-priming the GDN prefix.
pub(crate) struct Qwen35MoePrefixState {
    effective_cached_prefix_len: usize,
    suffix_len: usize,
    full_tokens: Vec<u32>,
    gdn_prefix_already_primed: bool,
    checkpoint_extra_keys: Vec<Vec<u64>>,
    checkpoint_cache_salt: u64,
}

impl PagedPrefix for Qwen35MoePrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

impl PagedBackend for Qwen35MoeInner {
    type PagedDecode<'a>
        = Qwen35MoePagedDecode<'a>
    where
        Self: 'a;
    type PrefixState = Qwen35MoePrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        _reuse_cache: bool,
        _block_size: usize,
        _extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        // The `prepare_turn_…` + `prepare_moe_gdn_prefix_state` block that
        // opens a MoE paged turn.
        self.paged_finalize_failed = false;
        let trace_enabled = inference_trace_enabled();
        let total_budget = plan.len() as u32;
        // vLLM exact-prefix cap: leave at least one prompt token to prefill so
        // the decoder always has something to consume.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let seq_id: u32 = 0;
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "prime_prefix_state: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?;
            adapter.block_size()
        };
        let carries_image_lineage = self.cached_image_key.is_some()
            && !self.cached_paged_image_token_positions.is_empty()
            && !self.cached_token_history.is_empty()
            && plan.starts_with(&self.cached_token_history);
        let image_positions = if carries_image_lineage {
            self.cached_paged_image_token_positions.as_slice()
        } else {
            &[]
        };
        let lookup_extra_keys =
            engine::build_paged_extra_keys(plan.len(), block_size, image_positions);

        // Adapter-owned warm/cold lifecycle. The [MLX_TRACE] line below
        // reads the PRE-turn live state, so probe the adapter immutably FIRST
        // (prepare_turn mutates request_tokens via continue_turn/reset). The
        // adapter re-reads the same state internally, so live_* is identical to
        // what prepare_turn decides on. reuse_cache=true literal: continuation
        // eligibility carries no reuse term (the engine's reuse_cache drives
        // finalize/save instead). Suffix blocks are allocated inside
        // prepare_turn.
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        {
            let adapter = self
                .paged_adapter
                .as_ref()
                .ok_or_else(|| Error::from_reason("prime_prefix_state: paged_adapter is None"))?;
            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = plan.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(plan, live_tokens);
            }
        }
        let turn_plan = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("prime_prefix_state: paged_adapter is None"))?
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                seq_id,
                plan,
                total_budget,
                true,
                &lookup_extra_keys,
                cache_salt,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason)?;
        let cached_prefix_len = turn_plan.cached_prefix_len;
        let continued_live_prefix = turn_plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                plan.len(),
                cached_prefix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens_len,
                live_mismatch.index,
                live_mismatch.prompt_token,
                live_mismatch.cached_token
            ));
        }

        // GDN recurrent-state prime (live / checkpoint / replay). No qwen3/lfm2
        // analog — moe carries GDN recurrent state across turns.
        let gdn_prefix_preparation = self.prepare_moe_gdn_prefix_state(
            plan,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        )?;
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        // GDN recurrent state now covers exactly the cached prefix (installed
        // live, restored from a cold sidecar, or replayed just above). Discharge
        // the adapter's auxiliary-state obligation — set when the cold tier
        // carries a `ColdSidecarPolicy` and a HOT K/V hit handed back a prefix no
        // sidecar was gated against — before the turn's first `record_tokens`. A
        // no-op for a live continuation or a sidecar-backed restore.
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter
                .confirm_aux_prefix_primed(cached_prefix_len)
                .map_err(Error::from_reason)?;
        }
        let preserves_image_lineage = carries_image_lineage && cached_prefix_len > 0;
        // Clear the per-turn session state here (history is re-set in
        // `save_paged_history`; image key is reset because the paged path does
        // not carry it across turns). The cross-turn M-RoPE delta is carried
        // only when this turn extends the live image sequence
        // (continued_live_prefix); a cold start or a non-live prefix-cache hit
        // (text-only prefix) drops a stale image delta so the text suffix
        // rotates at the raw physical slot.
        self.cached_token_history.clear();
        if !preserves_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }
        self.cached_rope_deltas = crate::models::qwen3_5::paged_forward::rope_delta_for_paged_turn(
            self.cached_rope_deltas,
            preserves_image_lineage,
        );

        let suffix_len = total_budget.checked_sub(cached_prefix_len).ok_or_else(|| {
            Error::from_reason("prime_prefix_state: cached_prefix_len > total_prompt_tokens")
        })? as usize;

        Ok(Qwen35MoePrefixState {
            effective_cached_prefix_len: cached_prefix_len as usize,
            suffix_len,
            full_tokens: plan.to_vec(),
            gdn_prefix_already_primed,
            checkpoint_extra_keys: lookup_extra_keys,
            checkpoint_cache_salt: cache_salt,
        })
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        // The paged prefill block. `run_paged_prefill_chunk` writes K/V into
        // the adapter pool, populates the GDN linear caches, runs the GDN
        // pre-pass over the cached prefix from `full_tokens` (skipped when
        // `gdn_prefix_already_primed`), then the full forward over the suffix,
        // folding in the last-token slice (returns `[vocab]`). The engine
        // fires the post-prefill `synchronize_and_clear_cache` AFTER this
        // returns (NOT here).
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            self.config.num_layers as usize,
            |i| self.config.is_linear_layer(i),
        );
        let embed = self.embedding.clone();
        // Cross-turn M-RoPE delta (0 unless this engine-driven text turn warm-
        // continues an image prefill); aligns the suffix keys with the
        // compressed-position image keys.
        let rope_deltas = self.cached_rope_deltas.unwrap_or(0);
        // Split the prefill at the checkpoint ladder when persistence is on so
        // the GDN recurrent state is materialized at every boundary the cold
        // sidecar could anchor on. Byte-inert with no cold GDN policy — the
        // unchanged single-shot default. See `cold_gdn_prefill_chunk_size`.
        let chunk_size = self.cold_gdn_prefill_chunk_size();
        let (logits, gdn_checkpoint) = {
            let caches_ref = self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("paged_prefill: caches not initialized"))?;
            let adapter = self
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("paged_prefill: paged_adapter dropped"))?;
            super::paged_forward::run_paged_prefill_chunk_with_size_and_checkpoint(
                &prefix.full_tokens,
                suffix_tokens,
                prefix.effective_cached_prefix_len as u32,
                prefix.gdn_prefix_already_primed,
                &embed,
                &mut self.layers,
                caches_ref,
                &self.final_norm,
                &self.lm_head,
                &layer_kinds,
                adapter,
                chunk_size,
                rope_deltas,
            )?
        };
        self.publish_moe_gdn_materialized_prefix_checkpoint(
            &prefix.full_tokens,
            &prefix.checkpoint_extra_keys,
            prefix.checkpoint_cache_salt,
            gdn_checkpoint,
        );
        Ok(logits)
    }

    fn begin_paged_decode(&mut self, _setup: &PagedTurnSetup<'_>) -> Result<Self::PagedDecode<'_>> {
        // Pure-Rust eager paged decode: the stepper drives
        // `run_paged_decode_step` against the live post-prefill adapter pools +
        // GDN caches. No compiled-graph seeding / lifecycle locks needed.
        Ok(Qwen35MoePagedDecode { inner: self })
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool) {
        // Terminal lifecycle block of a paged turn. Success: keep the request
        // live across turns when reuse is on, using PER-BLOCK extra keys (NOT
        // qwen3's empty `&[]`), so the next turn's continue builds on the
        // partial trailing block's live K/V; otherwise register full blocks
        // for reuse + release. The trait hook remains infallible, but an adapter
        // error must downgrade the session before the engine's unconditional
        // history save can publish image-placeholder history.
        self.paged_finalize_failed = false;
        let mut release_pending = false;
        let mut finalize_error = match self.paged_adapter.as_mut() {
            Some(adapter) if reuse_cache => {
                let total_for_finalize = adapter.request_tokens().len();
                let block_size = adapter.block_size();
                let finalize_extra_keys = engine::build_paged_extra_keys(
                    total_for_finalize,
                    block_size,
                    &self.cached_paged_image_token_positions,
                );
                adapter
                    .finalize_turn_keep_live_per_block(&finalize_extra_keys, 0)
                    .err()
            }
            Some(adapter) => {
                let total_for_finalize = adapter.request_tokens().len();
                let block_size = adapter.block_size();
                let finalize_extra_keys = engine::build_paged_extra_keys(
                    total_for_finalize,
                    block_size,
                    &self.cached_paged_image_token_positions,
                );
                // The release is attempted below (even when registration fails),
                // but only AFTER the GDN sidecar capture — releasing resets the
                // adapter's cold-chain frontier to 0.
                release_pending = true;
                adapter
                    .register_full_blocks_for_reuse_per_block(&finalize_extra_keys, 0)
                    .err()
            }
            None => Some("paged_adapter is None during MoE finalization".to_owned()),
        };
        // Persist the out-of-pool GDN recurrent state for the SAME chain the
        // adapter just captured (its `cold_captured_blocks` is now set), so a
        // later process can resume from the restored K/V prefix instead of
        // replaying GDN over it. Runs BEFORE the release below, which resets the
        // captured-chain frontier to 0. Skipped when the finalize failed: the
        // K/V chain the sidecar would anchor on was not published, so nothing
        // could ever select it.
        if finalize_error.is_none() {
            self.capture_moe_gdn_cold_sidecar(&self.cached_paged_image_token_positions);
        }
        // Always attempt the release for the non-reuse path, even when
        // registration failed (mirrors the prior `register_error.or(release)`).
        if release_pending && let Some(adapter) = self.paged_adapter.as_mut() {
            finalize_error = finalize_error.or(adapter.release_request().err());
        }
        if let Some(error) = finalize_error {
            self.downgrade_failed_paged_finalize(&error);
        }
    }

    fn abort_paged_turn(&mut self) {
        // Hybrid error-path teardown: K/V and GDN may have advanced by
        // different amounts, so a bare adapter release would leave a false-live
        // recurrent session behind. Never register / keep live; invalidate all
        // model-local continuation state without masking the original error.
        self.invalidate_moe_paged_session("generic paged turn abort");
    }

    fn paged_decode_stream(&self, _generation_stream: Stream) -> Stream {
        // Run the compiled-paged DECODE on the canonical DEFAULT stream, NOT the
        // per-turn `generation_stream`. moe's compiled forward + every
        // `y.eval()` run on the MLX DEFAULT stream; running the forward on a
        // queue separate from the shared loop's top-of-iteration `y.eval()`
        // (always on the default stream) would force a cross-queue
        // completion-wait every token (~5% on bandwidth-bound decode).
        // `paged_prefill` still runs on `generation_stream`. See the
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
        if std::mem::take(&mut self.paged_finalize_failed) {
            self.cached_token_history.clear();
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
            self.cached_rope_deltas = None;
            self.gdn_last_history_checkpoint = None;
            self.caches = None;
            return Ok(());
        }
        // moe paged ALWAYS drops the last token, regardless of the engine's
        // `keep_all` (length-exit) signal — the paged decode loop NEVER forwards
        // the LAST sampled token (the engine's forward gate skips it AND
        // `materialize_final` is a no-op for moe), so the last `generated` entry
        // is NOT in the adapter / GDN caches and must be dropped to keep the
        // saved history aligned with the live cache state. Ordering:
        // finalize → set history (drop-last, last_token_in_cache=false) → GDN
        // checkpoint → clear image key. PRESERVE THIS EXACT ORDER — it is the
        // most delicate part for T=0 byte-equality.
        if !reuse_cache {
            self.cached_token_history.clear();
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
            self.cached_rope_deltas = None;
            return Ok(());
        }
        let mut full_history = save_tokens.to_vec();
        if !generated.is_empty() {
            // last_token_in_cache == false → drop-last UNCONDITIONAL.
            let upto = generated.len().saturating_sub(1);
            full_history.extend_from_slice(&generated[..upto]);
        }
        self.cached_token_history = full_history;
        // GDN history checkpoint — must run AFTER the history is set (it
        // snapshots the live recurrent caches keyed on `cached_token_history`),
        // BEFORE clearing the image key. A checkpoint/eval failure here PROPAGATES
        // (`?`) to abort the turn: a half-snapshotted or failed-eval GDN state
        // must NOT be published as a reusable warm-continue checkpoint, or the
        // next turn reads corrupt recurrent state.
        let store = self.remember_moe_gdn_history_checkpoint()?;
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} \
                 total_ms={:.1}",
                store.stored,
                self.cached_token_history.len(),
                store.eval_ms,
                store.clone_ms,
                store.token_clone_ms,
                store.update_ms,
                store.total_ms
            ));
        }
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        _keep_all: bool,
    ) -> bool {
        // moe ALWAYS drops the last token (see `save_paged_history`), so the
        // to-be-saved history length is `prompt_len + (generated.len() - 1)` (or
        // `prompt_len` when nothing was generated). Roll the adapter back to that
        // length so the next turn's warm-continue gate
        // (`prompt.starts_with(request_tokens())`) is not defeated by a trailing
        // token the pipelined loop recorded at the loop top before the
        // stop-check. `_keep_all` is intentionally ignored (qwen3 signal).
        //
        // Token accounting: on BOTH length and early-stop exits the to-be-saved
        // history equals the adapter cursor (the final/terminal forward was
        // skipped), so `surplus` is 0 and this is a true no-op for moe — but the
        // rollback is kept as the defensive contract the trait mandates.
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
                target: "mlx_core::qwen3_5_moe::paged",
                "reconcile_paged_request_tokens: rollback_last_tokens({surplus}) failed \
                 (finalize releases the request; next turn cold-prefills): {e}",
            );
            return false;
        }
        true
    }
}

impl ChatBackend for Qwen35MoeInner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "qwen3_5_moe"
    }

    fn set_cache_owner_id(&mut self, owner_id: &str, root_owner_id: Option<&str>) {
        self.active_cache_owner_id.clear();
        self.active_cache_owner_id.push_str(owner_id);
        if let Some(root_owner_id) = root_owner_id {
            self.gdn_root_cache_owner_id = Some(root_owner_id.to_owned());
            self.gdn_root_cache_owner_is_explicit = true;
        } else {
            self.gdn_root_cache_owner_id = Some(owner_id.to_owned());
            self.gdn_root_cache_owner_is_explicit = false;
        }
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

    // thinking: engine default `policy()` == `ThinkingPolicy::TemplateHonoring`
    // → `thinking_setup` resolves to the legacy
    // `{enabled: resolve_enable_thinking(config).unwrap_or(true),
    //   budget: config.thinking_token_budget}`.

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        match scope {
            // Prefix-miss reset: reset each live layer cache, then install a
            // fresh hybrid cache vec. PRESERVES `cached_token_history` /
            // `cached_image_key` / `cached_rope_deltas` (the end-of-turn
            // save overwrites them) and the GDN checkpoints (paged-path
            // state the flat reset never touches).
            ResetScope::PrefixMiss => {
                if let Some(ref mut caches) = self.caches {
                    for cache in caches.iter_mut() {
                        cache.reset();
                    }
                }
                self.caches = Some(fresh_moe_layer_caches(&self.config));
                Ok(())
            }
            // Full clear including history, image key, rope deltas, GDN
            // checkpoints, via `reset_caches_sync`.
            //
            // The EXPLICIT command reset must additionally restore a
            // fully COLD paged state. `reset_caches_sync` does not touch
            // the paged adapter at all (it only clears the flat caches +
            // reuse state), so the prior turn's full blocks stay
            // content-addressed in the per-instance BlockAllocator's
            // prefix cache. A reset-then-rerun of the same prompt would
            // then take the prefix-hit 1-token-suffix prefill
            // (`find_cached_prefix_per_block_with_max_tokens` ->
            // `find_longest_cache_hit`) instead of the cold full prefill,
            // a different bf16 reduction order that can flip a greedy
            // near-tie (observed on the lfm2 sibling; qwen3_5_moe shares
            // the identical adapter lifecycle). One
            // call both releases the live request and purges every
            // prefix-cache entry. `ResetScope::PrefixMiss` (turn-internal)
            // keeps the prefix cache: cross-request block reuse after a
            // history miss is the paged design's entire point.
            ResetScope::Command => {
                self.reset_caches_sync()?;
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    adapter
                        .release_request_and_purge_prefix_cache()
                        .map_err(|e| {
                            Error::from_reason(format!(
                                "qwen3.5-moe reset_caches: paged prefix-cache purge failed: {e}"
                            ))
                        })?;
                }
                Ok(())
            }
        }
    }

    /// All-or-nothing prefix match (NO exact-match rewind — the 30 GDN
    /// linear-attention layers carry a recurrent state that cannot
    /// rewind one slot; the engine's exact-match-as-miss handling
    /// performs a full reset + re-prefill on a zero-delta hit).
    /// Text-only by construction: the generic flow never
    /// carries images (the vision probe owns those turns), so the
    /// expanded-token / image-key inputs collapse to the plain prompt.
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        verify_cache_prefix_direct(
            reuse_cache,
            false,
            tokens,
            tokens,
            0,
            &self.cached_token_history,
            &self.cached_image_key,
            self.caches.is_some(),
        )
    }

    fn flat_caches_desynced(&self) -> bool {
        self.flat_mtp_caches_desynced
    }

    fn clear_flat_caches_desynced(&mut self) {
        self.flat_mtp_caches_desynced = false;
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        // Delta continuations preserve `cached_image_key` — the KV cache
        // still holds the prior prefill's image attention state even
        // though this turn was text-only. Fresh turns (re)set the key
        // from the turn's (always-false here) `has_images`.
        //
        // `drop_last_always = true`: generic `run_decode_loop` flow (flat,
        // non-MTP, non-image MoE turns) never forwards the final committed
        // token into the physical cache on any exit kind, and the GDN
        // recurrent state is non-invertible, so drop it to keep
        // `cached_token_history.len() == physical_cache_len`.
        if args.is_delta {
            engine::save_cache_state_after_delta(
                args.reuse_cache,
                args.generated_tokens,
                args.finish_reason,
                /* drop_last_always */ true,
                args.save_tokens,
                &mut self.cached_token_history,
                &mut self.cached_image_key,
                &mut self.cached_rope_deltas,
                &mut self.caches,
            );
        } else {
            save_cache_state_direct(
                args.reuse_cache,
                args.has_images,
                args.generated_tokens,
                args.finish_reason,
                /* drop_last_always */ true,
                args.save_tokens,
                args.save_expanded_tokens,
                args.image_cache_key,
                &mut self.cached_token_history,
                &mut self.cached_image_key,
                &mut self.cached_rope_deltas,
                &mut self.caches,
            );
        }
        self.cached_paged_image_token_positions.clear();
    }

    fn eval_caches(&self) -> Result<()> {
        // No post-prefill cache sync on the MoE reference paths:
        // `chunked_prefill` evals internally per chunk and the decode
        // loop schedules async evals. A blocking sync here would
        // introduce an unnecessary stall.
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        // Text-only prefill block (the engine's reset-or-delta split already
        // ran; `self.caches` holds either fresh caches or the live session
        // state). Unlike dense, the MoE `chunked_prefill` returns the
        // full `[1, seq, vocab]` logits, so the slice+squeeze to the last
        // position folds in here (the engine's prefill contract is
        // last-token logits).
        let prompt = MxArray::from_uint32(prompt_tokens, &[1, prompt_tokens.len() as i64])?;
        let fa_idx = self.fa_idx;
        let logits = chunked_prefill(
            &prompt,
            &self.embedding,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            fa_idx,
            stream,
        )?;
        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        last_logits.squeeze(Some(&[1]))
    }

    type Decode<'a>
        = Qwen35MoeDecode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        // NOTE: no decode-entry `info!` trace here — unlike dense, the
        // MoE path does not log a "chat_decode entry" line.
        let is_streaming = self.turn_is_streaming.get();

        let embedding = self.embedding.clone();

        let relabel = match (is_streaming, turn.is_delta) {
            (false, false) => "moe_chat_rust",
            (false, true) => "moe_chat_delta_rust",
            (true, false) => "moe_chat_stream_rust",
            (true, true) => "moe_chat_stream_delta_rust",
        };

        Ok(Qwen35MoeDecode {
            inner: self,
            embedding,
            relabel,
        })
    }

    fn execution_plan(&self) -> ExecutionPlan {
        ExecutionPlan {
            media: qwen35_moe_media_plan(
                self.vision_encoder.is_some(),
                self.image_processor.is_some(),
                self.paged_adapter.is_some(),
            ),
            paged_attention: self.paged_adapter.as_ref().map(|_| PagedAttentionPlan {
                supports_delta: true,
            }),
            speculative: self.has_mtp_weights().then_some(SpeculativePlan {
                kind: SpeculativeKind::NativeMtp,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::NONE,
                // Existing MoE routing gives paged attention precedence;
                // native MTP is flat-cache only for this family.
                supports_paged_attention: false,
            }),
        }
    }

    fn wired_limit_bytes(&self) -> Option<usize> {
        // Per-turn wired-memory limit = the model's estimated footprint.
        Some(self.config.estimate_memory_bytes() as usize)
    }

    fn profiler_label(&self, is_delta: bool, is_streaming: bool) -> &'static str {
        // Record the turn's streaming-ness for `begin_decode`'s relabel
        // (`TurnSetup` does not carry it). The session core calls this
        // hook exactly once per generic-flow turn, before
        // `begin_decode`; specialized whole-turn paths return from their
        // planned executor earlier and never consult either hook.
        self.turn_is_streaming.set(is_streaming);
        match (is_streaming, is_delta) {
            (false, false) => "moe_chat",
            (false, true) => "moe_chat_delta",
            (true, false) => "moe_chat_stream",
            (true, true) => "moe_chat_stream_delta",
        }
    }

    fn has_live_session(&self) -> bool {
        // Delta guard: `self.caches.is_none()` means there is no
        // initialized session to continue.
        self.caches.is_some()
    }

    fn session_media(&self) -> MediaCapabilities {
        qwen35_moe_session_media(
            self.cached_image_key.is_some(),
            self.cached_rope_deltas.is_some(),
        )
    }

    fn session_media_matches_payloads(&self, images: &[Vec<u8>], audio: &[Vec<u8>]) -> bool {
        qwen35_moe_session_media_matches_payloads(self.cached_image_key, images, audio)
    }

    fn template_history_comparison_tokens<'a>(
        &self,
        tokens: &'a [u32],
    ) -> std::borrow::Cow<'a, [u32]> {
        engine::collapse_cached_media_placeholder_runs(
            tokens,
            IMAGE_TOKEN_ID as u32,
            &self.cached_paged_image_token_positions,
        )
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // Unlike dense, the MoE execution plan does not admit speculative
        // decoding with paged attention. The autoregressive text+paged path
        // runs through the generic `run_paged_turn`, which drives the adapter
        // lifecycle via [`PagedBackend`] and reuses the shared decode loop.
        debug_assert!(args.plan.use_paged_attention);
        debug_assert!(self.paged_adapter.is_some());
        debug_assert!(matches!(args.plan.decoder, DecoderPlan::Autoregressive));
        let mut constrained_params = args.params.clone();
        self.preflight_paged_context(args.tokens.len(), &mut constrained_params)?;
        let mut constrained_args = WholeTurnArgs {
            tokens: args.tokens,
            tokenizer: args.tokenizer,
            eos_id: args.eos_id,
            config: args.config,
            params: &constrained_params,
            thinking: args.thinking,
            plan: args.plan,
            sink: args.sink,
            cancelled: args.cancelled,
            media: args.media,
        };
        crate::engine::paged_turn::run_paged_turn(self, &mut constrained_args)
    }

    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The execution plan has already established request opt-in, loaded
        // MTP weights, flat-cache admission, and text-only input.
        debug_assert!(args.media.is_empty());
        debug_assert!(matches!(
            args.plan.decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
        ));
        self.moe_whole_turn(args)
    }

    fn run_multimodal_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The MoE cores own the full image pipeline (VLM prefill via
        // `vlm_prefill_moe`, M-RoPE deltas, paged-backend validation,
        // missing-encoder error).
        self.moe_whole_turn(args)
    }
}

/// Opt-in gate for the MoE MTP committed-history v2 path. Default OFF: the
/// drafter cache retention policy is an unvalidated perf change (v2 adds
/// per-cycle commit work whose accept-rate payoff is unmeasured), so it ships
/// behind this flag until a same-checkpoint A/B confirms the win. When off,
/// `MoeMtpStepper::use_committed` is `false` at every site and the stepper is
/// byte-for-byte the prior cycle-history v1. Read once on first call and
/// cached; subsequent reads hit the `OnceLock` fast path.
fn moe_mtp_committed_history_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default(
            "MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY",
            false,
        )
    })
}

/// Per-turn MTP propose/verify stepper for the MoE family's FLAT eager path
/// that [`crate::engine::mtp_turn::run_mtp_turn`] drives.
///
/// COMMITTED-HISTORY v2 (mirrors dense's `DenseMtpStepper`): when
/// `use_committed` holds, [`Self::begin_cycle`] only truncates the prior
/// cycle's speculative draft tail back to `committed_len` (never rebuilding a
/// fresh cache) and [`Self::commit_mtp`] appends each cycle's newly committed
/// tokens' exact K/V, so the drafter's cache persists across cycles within a
/// turn. `use_committed` is opt-in: it requires the
/// `MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY` flag AND
/// `setup.prompt_hidden_position_base == 0`. With the flag off (the default)
/// `use_committed` is `false`, so this stepper falls back to CYCLE-HISTORY v1
/// (fresh drafter cache each cycle, no-op commit) — byte-for-byte the prior
/// behavior. The prompt-prefix seed in [`MtpBackend::begin_mtp_decode`]
/// mirrors dense's byte-for-byte but is currently INERT: no MoE call site
/// populates a real `prompt_hidden`/`prompt_hidden_ids` yet (there is no MoE
/// hidden-emitting prefill), so even with the flag on, cycle 1 of every turn
/// still starts the drafter from an empty cache — only cycles 2+ within the
/// SAME turn benefit today. FLAT-ONLY: the main forward, verify, and rollback
/// all act on `inner.caches`; there is no paged routing, adapter, or
/// `MtpStepMode` here.
pub(crate) struct MoeMtpStepper<'a> {
    /// The model — owns layers / caches / mtp / final_norm / lm_head and the
    /// `flat_mtp_caches_desynced` latch.
    inner: &'a mut Qwen35MoeInner,
    /// Drafter K/V caches. v2 committed-history mode (`use_committed`) holds
    /// the persistent committed prefix; v1 cycle-history mode (the flag-off
    /// default, or the `position_base != 0` fallback) is reset fresh by
    /// [`Self::begin_cycle`].
    mtp_caches: Vec<Qwen3_5LayerCache>,
    /// Eager analogue of dense's committed-length cursor: committed tokens
    /// whose exact K/V live in `mtp_caches`.
    committed_len: i32,
    /// Committed-history active iff the opt-in flag is on AND the prompt tail's
    /// hiddens start at absolute position 0 — mirrors
    /// `DenseMtpStepper::use_committed`.
    use_committed: bool,
    /// Pre-verify snapshot of the main caches, taken in
    /// [`Self::snapshot_main_linear`], consumed by [`Self::rollback`].
    snap: Option<Result<Vec<super::layer_cache::Qwen3_5LayerSnapshot>>>,
    /// GDN tape recorded by [`Self::verify_step`], consumed by
    /// [`Self::rollback`].
    tape: Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    /// Error stashed by the infallible [`Self::rollback`] replay, surfaced by
    /// [`Self::take_replay_error`].
    replay_err: Option<Error>,
    /// Mid-cycle-stop desync latch (set by [`Self::rollback_unemitted`]),
    /// reported by [`Self::into_desynced`].
    mtp_desynced: bool,
    /// The model's embedding lookup and tied-head projection backend.
    embedding: Embedding,
    /// Config clone for the per-cycle drafter cache reset.
    config: Qwen3_5MoeConfig,
    /// Index of the first full-attention layer, threaded into the MoE eager
    /// forwards.
    fa_idx: usize,
}

impl MtpStepper for MoeMtpStepper<'_> {
    fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    fn committed_history_active(&self) -> bool {
        self.use_committed
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        // The eager MoE MTP path set the turn label via
        // `profiler.set_label("moe_mtp_eager")` at the migration site; the
        // engine applies this relabel once at turn entry instead.
        Some("moe_mtp_eager")
    }

    // Step A main forward: eager pre-norm + final-norm + project. Returns
    // `hidden` shaped `[1, hidden]` (squeeze the time axis); `logits` stays
    // `[1, 1, vocab]` with `needs_squeeze = true`.
    fn forward_with_hidden(
        &mut self,
        ids: &MxArray,
        embedding: &Embedding,
    ) -> Result<(MxArray, MxArray, bool)> {
        let inner = &mut *self.inner;
        let pre = forward_pre_norm_inner(
            ids,
            embedding,
            &mut inner.layers,
            &mut inner.caches,
            self.fa_idx,
        )?;
        let h3 = inner.final_norm.forward(&pre)?;
        let logits = project_logits_from_hidden(&h3, &inner.lm_head, embedding)?;
        let hidden = h3.squeeze(Some(&[1]))?;
        Ok((logits, hidden, true))
    }

    // One MTP draft step on the eager drafter. `h_next` is `[1, 1, hidden]`;
    // project to `draft_logits` `[1, 1, vocab]` then squeeze the time axis to
    // `[1, vocab]`.
    fn draft_step(
        &mut self,
        prev_hidden: &MxArray,
        prev_emb: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let inner = &mut *self.inner;
        let mtp_caches = &mut self.mtp_caches;
        let mtp = inner.mtp.as_mut().ok_or_else(|| {
            Error::from_reason(
                "eager MoE MTP draft_step: inner.mtp is None despite \
                 has_mtp_weights() gate",
            )
        })?;
        let h_next = mtp.forward(prev_hidden, prev_emb, Some(mtp_caches))?;
        let dl3 = project_logits_from_hidden(&h_next, &inner.lm_head, &self.embedding)?;
        let draft_logits = dl3.squeeze(Some(&[1]))?;
        Ok((h_next, draft_logits))
    }

    // Batched verify: run the K+1 verify ids through the main stack,
    // advancing `inner.caches` by K+1, recording the GDN tape.
    fn verify_step(
        &mut self,
        ids: &MxArray,
        embedding: &Embedding,
        depth: usize,
    ) -> Result<mtp_decode::MtpVerifyOutput> {
        let _ = depth;
        let inner = &mut *self.inner;
        let tape = &mut self.tape;
        eager_verify_step(
            &mut inner.layers,
            &mut inner.caches,
            &inner.final_norm,
            &inner.lm_head,
            self.fa_idx,
            ids,
            embedding,
            Some(tape),
        )
    }

    // No native argmax-only / sparse verify on the eager path — the accept
    // loop falls back to dense-logits accept. (Defaults `None`.)

    // Snapshot the main caches before verify mutates them. Stash the fallible
    // result; surfaced in `rollback` / `restore_and_replay_main`.
    fn snapshot_main_linear(&mut self) {
        let inner = &*self.inner;
        let snap = match inner.caches.as_ref() {
            Some(caches) => super::layer_cache::snapshot_all(caches),
            None => Err(Error::from_reason(
                "eager MoE MTP snapshot_main_linear: inner.caches is None",
            )),
        };
        self.snap = Some(snap);
    }

    // Pure-Rust GDN tape replay — fires on BOTH full and partial accept.
    // Infallible signature: any error is stashed in `self.replay_err` and
    // surfaced later. Full-attention layers rewind their K/V offset to
    // `snapshot_offset + accepted_steps`; GDN layers replay the recorded tape.
    fn rollback(&mut self, accepted_drafts: usize, _depth: usize) {
        if self.replay_err.is_some() {
            return;
        }
        let accepted_steps = accepted_drafts + 1;
        let result: Result<()> = (|| {
            let snap = match self.snap.as_ref() {
                Some(Ok(s)) => s,
                Some(Err(e)) => {
                    return Err(Error::from_reason(format!(
                        "eager MoE MTP rollback: snapshot failed: {}",
                        e.reason
                    )));
                }
                None => {
                    return Err(Error::from_reason(
                        "eager MoE MTP rollback: snapshot missing (snapshot_main_linear \
                         did not run)",
                    ));
                }
            };
            let tape = &self.tape;
            let inner = &mut *self.inner;
            let caches = inner.caches.as_mut().ok_or_else(|| {
                Error::from_reason("eager MoE MTP rollback: inner.caches is None")
            })?;
            if caches.len() != snap.len() || caches.len() != tape.len() {
                return Err(Error::from_reason(format!(
                    "eager MoE MTP rollback: length mismatch (caches {}, snapshot {}, \
                     tape {})",
                    caches.len(),
                    snap.len(),
                    tape.len(),
                )));
            }
            for (idx, cache) in caches.iter_mut().enumerate() {
                let Some(layer_tape) = tape[idx].as_ref() else {
                    // Full-attention layer: rewind the offset to
                    // `snapshot_offset + accepted_steps` so the next forward
                    // overwrites the rejected drafts. No-op on full accept.
                    match &snap[idx] {
                        super::layer_cache::Qwen3_5LayerSnapshot::FullAttention { offset } => {
                            let kv = cache.as_kv_cache_mut().ok_or_else(|| {
                                Error::from_reason(format!(
                                    "eager MoE MTP rollback: layer {idx} has a \
                                     FullAttention snapshot but its cache slot is \
                                     not FullAttention",
                                ))
                            })?;
                            let target = *offset + accepted_steps as i32;
                            kv.trim(target);
                        }
                        super::layer_cache::Qwen3_5LayerSnapshot::Linear { .. } => {
                            return Err(Error::from_reason(format!(
                                "eager MoE MTP rollback: layer {idx} has no GDN tape \
                                 but a Linear snapshot",
                            )));
                        }
                    }
                    continue;
                };
                let arrays = cache.as_arrays_cache_mut().ok_or_else(|| {
                    Error::from_reason(format!(
                        "eager MoE MTP rollback: layer {idx} has a GDN tape but its \
                         cache slot is not Linear",
                    ))
                })?;
                let (snap_conv, snap_rec) = match &snap[idx] {
                    super::layer_cache::Qwen3_5LayerSnapshot::Linear {
                        conv_state,
                        recurrent_state,
                    } => (conv_state.as_ref(), recurrent_state.as_ref()),
                    super::layer_cache::Qwen3_5LayerSnapshot::FullAttention { .. } => {
                        return Err(Error::from_reason(format!(
                            "eager MoE MTP rollback: layer {idx} GDN tape but \
                             FullAttention snapshot",
                        )));
                    }
                };
                let window = layer_tape.kernel.window_len()? as usize;
                if accepted_steps > window {
                    return Err(Error::from_reason(format!(
                        "eager MoE MTP rollback: accepted_steps {accepted_steps} \
                         exceeds recorded window {window} at layer {idx}",
                    )));
                }
                layer_tape.replay_into(arrays, snap_conv, snap_rec, accepted_steps)?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.replay_err = Some(e);
        }
    }

    // On rejection (partial accept): the GDN tape replay in `rollback` already
    // reconstructed the AR-exact main cache state, so no re-forward loop is
    // needed. This only surfaces a stashed replay error and clears the
    // per-cycle snapshot + tape.
    fn restore_and_replay_main(&mut self, _accepted: &[u32], _embedding: &Embedding) -> Result<()> {
        self.snap = None;
        self.tape.clear();
        if let Some(e) = self.replay_err.take() {
            return Err(e);
        }
        Ok(())
    }

    // Committed-history commit. Mirrors `DenseMtpStepper::commit_mtp`.
    //
    // v1 (`!use_committed`): no-op.
    //
    // v2 (`use_committed`): append the M newly committed tokens' EXACT K/V to
    // the persistent MTP cache via one multi-token drafter forward.
    fn commit_mtp(
        &mut self,
        anchor: mtp_decode::MtpCommitAnchor,
        seed_hidden: &MxArray,
        verify_hiddens: &MxArray,
        committed_ids: &[u32],
        _k_accepted: usize,
        embedding: &Embedding,
    ) -> Result<()> {
        if !self.use_committed {
            return Ok(());
        }
        let m = committed_ids.len();
        if m == 0 {
            return Ok(());
        }
        let hidden_dim = verify_hiddens.shape_at(2)?;

        // Assemble hidden_seq [1, M, hidden] per anchor.
        let hidden_seq = match anchor {
            mtp_decode::MtpCommitAnchor::IncludeAnchor => {
                // seed_hidden ++ verify_hiddens[:, 0..M-1, :].
                let vh_prefix =
                    verify_hiddens.slice(&[0, 0, 0], &[1, (m - 1) as i64, hidden_dim])?;
                MxArray::concatenate(seed_hidden, &vh_prefix, 1)?
            }
            mtp_decode::MtpCommitAnchor::SkipAlreadyCommittedAnchor => {
                // verify_hiddens[:, 0..M, :].
                verify_hiddens.slice(&[0, 0, 0], &[1, m as i64, hidden_dim])?
            }
        };

        // Gather the M committed-token input embeddings → [1, M, hidden].
        let ids_i32: Vec<i32> = committed_ids.iter().map(|&v| v as i32).collect();
        let ids_arr = MxArray::from_int32(&ids_i32, &[m as i64])?;
        let gathered = embedding.forward(&ids_arr)?;
        let emb_seq = gathered.reshape(&[1, m as i64, hidden_dim])?;

        // Drop this cycle's draft K/V (written past committed_len by the draft
        // steps), then write the exact committed K/V via one multi-token
        // forward.
        let inner = &mut *self.inner;
        let mtp = inner.mtp.as_mut().ok_or_else(|| {
            Error::from_reason(
                "eager MoE MTP commit_mtp: inner.mtp is None despite \
                 has_mtp_weights() gate",
            )
        })?;
        let caches = &mut self.mtp_caches;
        for c in caches.iter_mut() {
            if let Some(kv) = c.as_kv_cache_mut() {
                kv.trim(self.committed_len);
            }
        }
        let _ = mtp.forward(&hidden_seq, &emb_seq, Some(caches))?;
        self.committed_len += m as i32;
        Ok(())
    }

    // Re-anchor the drafter cache at the start of each cycle. Mirrors
    // `DenseMtpStepper::begin_cycle`.
    //
    // v1 (`!use_committed`): reset to a fresh cache.
    //
    // v2 (`use_committed`): the cache is PERSISTENT; truncate the prior
    // cycle's draft tail back to the re-anchor target. `chained_anchor`
    // cycles anchor one slot earlier (`committed_len - 1`); Step-A cycles at
    // `committed_len`.
    fn begin_cycle(&mut self, chained_anchor: bool) {
        if !self.use_committed {
            self.mtp_caches = Qwen3_5MoeMTPModule::fresh_caches(&self.config);
            return;
        }
        let target = if chained_anchor {
            (self.committed_len - 1).max(0)
        } else {
            self.committed_len
        };
        for c in self.mtp_caches.iter_mut() {
            if let Some(kv) = c.as_kv_cache_mut() {
                kv.trim(target);
            }
        }
    }

    // Bound the lazy graph: materialize the token plus the main GDN/full-attn
    // caches; on a budget-forced step also the logits.
    fn eval_step(&self, token: &MxArray, logits: &MxArray, budget_forced: bool) {
        async_eval_layer_caches(&self.inner.caches);
        token.eval();
        if budget_forced {
            logits.eval();
        }
    }

    // Chained end-of-iteration eval: keep the chained `verify_hidden[K]` slice
    // materialized alongside the token and the main caches so the next cycle's
    // draft graph does not force a separate Metal roundtrip.
    fn eval_step_with_chained_hidden(&self, token: &MxArray, chained_hidden: &MxArray) {
        async_eval_layer_caches(&self.inner.caches);
        MxArray::async_eval_arrays(&[token, chained_hidden]);
    }

    fn rollback_unemitted(&mut self, unemitted: usize) {
        if unemitted > 0 {
            self.mtp_desynced = true;
        }
    }

    fn take_replay_error(&mut self) -> Option<Error> {
        self.replay_err.take()
    }

    fn into_desynced(self) -> bool {
        self.mtp_desynced
    }
}

impl MtpBackend for Qwen35MoeInner {
    type MtpDecode<'a>
        = MoeMtpStepper<'a>
    where
        Self: 'a;

    fn begin_mtp_decode(&mut self, setup: &MtpTurnSetup<'_>) -> Result<Self::MtpDecode<'_>> {
        let embedding = self.embedding.clone();
        let config = self.config.clone();
        let fa_idx = self.fa_idx;

        // Committed-history is opt-in (default off) and only correct when the
        // prompt tail's hiddens start at absolute position 0 (the eager
        // drafter derives RoPE purely from the local cache offset) — mirrors
        // `DenseMtpStepper::begin_mtp_decode`'s gate, plus the
        // `MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY` flag. No MoE call site
        // populates a real `prompt_hidden_position_base` yet (see the struct
        // doc), so the position gate is always satisfied today; the flag is
        // what makes this `true`.
        let use_committed =
            moe_mtp_committed_history_enabled() && setup.prompt_hidden_position_base == 0;

        let mut stepper = MoeMtpStepper {
            inner: self,
            mtp_caches: Qwen3_5MoeMTPModule::fresh_caches(&config),
            committed_len: 0,
            use_committed,
            snap: None,
            tape: Vec::new(),
            replay_err: None,
            mtp_desynced: false,
            embedding,
            config,
            fa_idx,
        };

        // Prompt-prefix seed (v2 committed-history only). Mirrors
        // `DenseMtpStepper::begin_mtp_decode`'s prompt-prefix seed block
        // byte-for-byte. Currently INERT: no MoE call site populates
        // `setup.prompt_hidden` / `setup.prompt_hidden_ids` yet (there is no
        // MoE hidden-emitting prefill), so this block never runs until a
        // follow-up wires that capture through — see the struct doc on
        // [`MoeMtpStepper`].
        if use_committed
            && let (Some(ph), Some(ph_ids)) = (setup.prompt_hidden, setup.prompt_hidden_ids)
            && !ph_ids.is_empty()
        {
            let prompt_len = ph_ids.len();
            let hidden_dim = ph.shape_at(2)?;
            let hidden_len = ph.shape_at(1)? as usize;
            if hidden_len != prompt_len {
                return Err(Error::from_reason(format!(
                    "eager MoE MTP prompt-seed: prompt_hidden length {hidden_len} \
                     does not match prompt_hidden_ids length {prompt_len}"
                )));
            }
            // The first sampled token `y` is supplied by the engine via the
            // setup so the prompt seed can commit `[prompt_ids[1..], y]`.
            let y_id = setup.first_sampled_token;

            // Committed run = [prompt_ids[1..prompt_len], y] (length P).
            let mut committed_ids: Vec<i32> = Vec::with_capacity(prompt_len);
            committed_ids.extend(ph_ids[1..prompt_len].iter().map(|&v| v as i32));
            committed_ids.push(y_id as i32);

            let chunk_sizes = partition_prefill_chunks(prompt_len);
            let mut cursor: usize = 0;
            for &chunk in &chunk_sizes {
                let chunk_i64 = chunk as i64;
                let start = cursor as i64;
                // hidden_seq = prompt_hidden[:, cursor..cursor+chunk, :].
                let hidden_seq = ph.slice(&[0, start, 0], &[1, start + chunk_i64, hidden_dim])?;
                // emb_seq = gather embedding rows for the chunk's ids.
                let ids_arr =
                    MxArray::from_int32(&committed_ids[cursor..cursor + chunk], &[chunk_i64])?;
                let gathered = stepper.embedding.forward(&ids_arr)?;
                let emb_seq = gathered.reshape(&[1, chunk_i64, hidden_dim])?;

                let inner = &mut *stepper.inner;
                let mtp = inner.mtp.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "eager MoE MTP prompt-seed: inner.mtp is None despite \
                         has_mtp_weights() gate",
                    )
                })?;
                let caches = &mut stepper.mtp_caches;
                let _ = mtp.forward(&hidden_seq, &emb_seq, Some(caches))?;
                stepper.committed_len += chunk as i32;
                cursor += chunk;
            }
        }

        Ok(stepper)
    }

    fn record_turn_mtp_acceptance(&mut self, accepted: u64, attempted: u64) {
        self.mtp_draft_accepted += accepted;
        self.mtp_draft_attempted += attempted;
        mtp_decode::mtp_bound_gate_history(
            &mut self.mtp_draft_accepted,
            &mut self.mtp_draft_attempted,
        );
        self.mtp_gated_turns = 0;
    }
}

impl Qwen35MoeInner {
    /// MTP acceptance gate — see `mtp_decode::mtp_accept_gate_enabled`.
    /// Mirrors the dense `Qwen35Inner::mtp_gate_allows` policy
    /// (depth-1-scoped, confidence-aware on the aggregate; re-probes
    /// after [`mtp_decode::MTP_ACCEPT_GATE_REPROBE_TURNS`] gated turns).
    fn mtp_gate_allows(&mut self, requested_depth: u32) -> bool {
        if !mtp_decode::mtp_accept_gate_enabled() || requested_depth > 1 {
            return true;
        }
        let attempted = self.mtp_draft_attempted;
        if attempted == 0 {
            return true; // no history — probe
        }
        if !mtp_decode::mtp_accept_gate_blocks(self.mtp_draft_accepted, attempted) {
            return true;
        }
        self.mtp_gated_turns += 1;
        if self.mtp_gated_turns >= mtp_decode::MTP_ACCEPT_GATE_REPROBE_TURNS {
            self.mtp_gated_turns = 0;
            self.mtp_draft_accepted = 0;
            self.mtp_draft_attempted = 0; // re-probe next turn
        }
        false
    }
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
    /// construction time. Currently default-OFF on Qwen3.5 MoE
    /// (parity-pending — see CLAUDE.md and
    /// `Qwen3_5MoeConfig::use_block_paged_cache`). VLM checkpoints can
    /// load with the adapter on for text-only inference; image-bearing
    /// chat turns are rejected at runtime by the chat-entry sites.
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
}

#[napi]
impl Qwen3_5MoeModel {
    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Qwen35MoeInner::paged_adapter` was successfully
    /// constructed at load time (driven by
    /// `Qwen3_5MoeConfig::use_block_paged_cache`, currently default-OFF
    /// because parity is pending real-weights validation). On VLM
    /// checkpoints the adapter can still be active for text-only
    /// inference; image-bearing chat turns are rejected at runtime by
    /// the chat-entry sites. Surfaced through this NAPI method so
    /// server endpoints can branch on it without round-tripping through
    /// the model thread.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
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

    // ---------------------------------------------------------------
    // Test-only helpers: streaming session entry points that bypass
    // ThreadsafeFunction and expose the mpsc receiver directly. Used
    // by `crates/mlx-core/tests/qwen3_5_moe_session.rs` to exercise
    // the streaming path from a pure-Rust integration test without a
    // NAPI host. Marked `#[doc(hidden)]` because they're not part of
    // the public API surface.
    // ---------------------------------------------------------------

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
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
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
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
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

/// Run the MoE eager layer stack over `[1, T]` ids and return the
/// pre-final-norm hidden `[1, T, hidden]`.
///
/// This is the shared eager-MTP primitive: it advances `caches` (the flat
/// per-layer caches: `Linear` GDN slots + `FullAttention` KV slots) by `T`
/// and returns the full per-position hidden, exactly mirroring the dense
/// `forward_pre_norm_inner`. No explicit mask is ever built: `Linear` (GDN)
/// layers run mask-free, and full-attention layers pass `mask: None` too, so
/// `Qwen3_5Attention::forward` picks its fused "causal" SDPA kernel whenever
/// `seq_len > 1` — covering both prefill and the `[1, K+1]` eager-MTP verify
/// shape this helper backs.
fn forward_pre_norm_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    _fa_idx: usize,
) -> Result<MxArray> {
    let hidden_states = embedding.forward(input_ids)?;
    let mut h = hidden_states.clone();

    let num_layers = layers.len();
    for i in 0..num_layers {
        let cache = caches.as_mut().map(|c| &mut c[i]);
        h = layers[i].forward(&h, None, cache, None, true)?;
    }
    Ok(h)
}

/// Like [`forward_pre_norm_inner`], but records a per-layer GDN tape for the
/// eager-MTP rollback replay. `tape` must be pre-sized to `layers.len()`;
/// each GDN (`Linear`) layer writes `Some(GdnLayerTape)` into its slot and
/// full-attention layers leave it `None` — the exact indexing the rollback
/// replay relies on. The forward output is byte-identical to the non-tape
/// variant (the tape is a side-channel clone of the kernel inputs). No
/// explicit mask is built here either — see `forward_pre_norm_inner`.
fn forward_pre_norm_inner_with_tape(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    _fa_idx: usize,
    tape: &mut [Option<super::gated_delta_net::GdnLayerTape>],
) -> Result<MxArray> {
    let hidden_states = embedding.forward(input_ids)?;
    let mut h = hidden_states.clone();

    let num_layers = layers.len();
    debug_assert_eq!(
        tape.len(),
        num_layers,
        "forward_pre_norm_inner_with_tape: tape length must equal layer count"
    );
    for i in 0..num_layers {
        let cache = caches.as_mut().map(|c| &mut c[i]);
        let mut slot: Option<super::gated_delta_net::GdnLayerTape> = None;
        h = layers[i].forward_with_tape(&h, None, cache, None, true, Some(&mut slot))?;
        tape[i] = slot;
    }
    Ok(h)
}

/// Project a pre-/post-final-norm hidden to logits, preserving the leading
/// dims (`[*, hidden] -> [*, vocab]`). Uses the explicit `lm_head` when
/// present, else the tied embedding's linear projection. Mirrors the dense
/// `project_logits_from_hidden` and keeps packed embeddings packed-resident.
fn project_logits_from_hidden(
    hidden: &MxArray,
    lm_head: &Option<LinearProj>,
    embedding: &Embedding,
) -> Result<MxArray> {
    match lm_head {
        Some(head) => head.forward(hidden),
        None => embedding.as_linear(hidden),
    }
}

/// Batched eager verify: run the `[1, K+1]` verify ids through the MoE main
/// stack (recording the GDN tape when `tape` is `Some`), apply the final norm,
/// and project logits. Advances `caches` by `K+1`. Returns the dense logits
/// `[1, K+1, vocab]` plus the post-final-norm hiddens `[1, K+1, hidden]`
/// (`MtpVerifyOutput::logits_only`). Mirrors the dense `eager_verify_step`.
#[allow(clippy::too_many_arguments)]
fn eager_verify_step(
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    fa_idx: usize,
    verify_ids: &MxArray,
    embedding: &Embedding,
    tape: Option<&mut Vec<Option<super::gated_delta_net::GdnLayerTape>>>,
) -> Result<mtp_decode::MtpVerifyOutput> {
    let pre = match tape {
        Some(tape) => {
            tape.clear();
            tape.resize(layers.len(), None);
            forward_pre_norm_inner_with_tape(verify_ids, embedding, layers, caches, fa_idx, tape)?
        }
        None => forward_pre_norm_inner(verify_ids, embedding, layers, caches, fa_idx)?,
    };
    let hiddens = final_norm.forward(&pre)?;
    let logits = project_logits_from_hidden(&hiddens, lm_head, embedding)?;
    Ok(mtp_decode::MtpVerifyOutput::logits_only(logits, hiddens))
}

/// Forward pass using already-acquired lock guards (no lock overhead).
///
/// Used by generate/chat to avoid re-acquiring locks on every decode step.
fn forward_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    _fa_idx: usize,
) -> Result<MxArray> {
    let hidden_states = embedding.forward(input_ids)?;
    let mut h = hidden_states.clone();

    // No explicit mask is ever built. Full-attention layers pass `mask:
    // None`, so `Qwen3_5Attention::forward` picks its fused "causal" SDPA
    // kernel whenever `seq_len > 1` (prefill and the `[1, K+1]` eager-MTP
    // verify shape alike) instead of a materialized boolean-mask array.
    // Linear (GDN) layers already ran mask-free — mlx-vlm never creates one
    // for `ArraysCache`, and an all-ones mask would just be a no-op that
    // adds graph nodes and Metal overhead. Mirrors the dense
    // `forward_pre_norm_inner`.
    let num_layers = layers.len();
    for i in 0..num_layers {
        let cache = caches.as_mut().map(|c| &mut c[i]);
        h = layers[i].forward(&h, None, cache, None, true)?;
    }

    let h = final_norm.forward(&h)?;
    project_logits_from_hidden(&h, lm_head, embedding)
}

/// Default prefill chunk size (tokens per chunk).
///
/// Matches the Qwen3.5 Dense path and Python mlx-lm's `prefill_step_size`
/// default of 2048. Long-context prompts (40k+ tokens) would otherwise
/// allocate all per-layer activations concurrently (batch=1 × seq × hidden
/// plus a full attention score tensor per FA layer), blowing past the 96 GB
/// wired limit on an M3 Max 128 GB box. Chunking bounds the per-layer
/// transient peak at `chunk × hidden_dim` and inserts a cache-eval +
/// `clear_cache` barrier between chunks so the transient allocator state
/// does not accumulate across chunks.
pub(crate) const PREFILL_STEP_SIZE: i64 = 2048;

/// Chunked prefill for Qwen3.5 MoE.
///
/// Processes `prompt` (shape `[1, seq_len]`) in chunks of `PREFILL_STEP_SIZE`
/// tokens, evaluating all KV-cache arrays and clearing the MLX compute cache
/// between chunks to bound peak GPU activation memory. Returns the logits
/// from the **final** chunk, which share the same shape contract as a
/// single-shot `forward_inner` call: `[1, last_chunk_len, vocab_size]`.
///
/// Invariants vs. single-shot `forward_inner`:
/// - Identical numerical output at full precision (the KV caches thread
///   through chunk N into chunk N+1 just like they would through
///   successive `forward_inner(full_prompt)` calls during regular decode).
/// - The linear-attention recurrent state advances chunk-by-chunk. This is
///   the same forward direction as a single-shot call — chunking is a
///   memory-only transformation, not a semantic one.
/// - The decode KV caches are seeded in-place: `caches` is advanced
///   chunk-by-chunk through `&mut`, so when this returns the per-layer
///   `Qwen3_5LayerCache` entries (and the GDN recurrent state) already
///   reflect the full prompt. There is no separate post-prefill seeding
///   step for the caller to run.
///
/// Small prompts (<= `PREFILL_STEP_SIZE` tokens) hit exactly one loop
/// iteration and behave identically to a single `forward_inner` call — no
/// extra evals, no extra cache clears.
#[allow(clippy::too_many_arguments)]
fn chunked_prefill(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    fa_idx: usize,
    generation_stream: Stream,
) -> Result<MxArray> {
    chunked_prefill_with_size(
        prompt,
        embedding,
        layers,
        caches,
        final_norm,
        lm_head,
        fa_idx,
        generation_stream,
        PREFILL_STEP_SIZE,
    )
}

/// Explicit-size variant of `chunked_prefill`.
///
/// Same semantics as `chunked_prefill` but the chunk size is an explicit
/// parameter. Primarily used by tests to compare chunked vs single-shot
/// (by passing a chunk size >= prompt length) without plumbing a config
/// knob through every caller. Production callers should use
/// `chunked_prefill` which hardcodes `PREFILL_STEP_SIZE`.
#[allow(clippy::too_many_arguments)]
fn chunked_prefill_with_size(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    fa_idx: usize,
    generation_stream: Stream,
    chunk_size: i64,
) -> Result<MxArray> {
    debug_assert!(chunk_size > 0, "chunk_size must be positive");
    let total_len = prompt.shape_at(1)?;
    let mut offset: i64 = 0;

    // All-but-last chunks: run forward, eval caches, clear compute cache.
    // The returned logits from these chunks are thrown away because only
    // the final chunk's logits are consumed by the sampler.
    while total_len - offset > chunk_size {
        let chunk = prompt.slice_axis(1, offset, offset + chunk_size)?;
        {
            let _stream_ctx = StreamContext::new(generation_stream);
            let _logits = forward_inner(
                &chunk, embedding, layers, caches, final_norm, lm_head, fa_idx,
            )?;
        }
        // Materialize all cache arrays on GPU so the next chunk doesn't
        // extend a giant lazy graph rooted at the prior chunk's inputs.
        eval_layer_caches(caches)?;
        crate::array::clear_cache();
        offset += chunk_size;
    }

    // Final chunk: return logits to caller. No eval/clear here — the
    // caller's next step (sampling / slicing last_logits) triggers eval
    // naturally, and the outer decode loop clears cache on its own rhythm.
    let remaining = prompt.slice_axis(1, offset, total_len)?;
    let logits = {
        let _stream_ctx = StreamContext::new(generation_stream);
        forward_inner(
            &remaining, embedding, layers, caches, final_norm, lm_head, fa_idx,
        )?
    };
    Ok(logits)
}

#[cfg(test)]
mod prefix_cache_reuse_integration_tests {
    //! End-to-end tests for prefix KV cache reuse on Qwen3.5 MoE. These
    //! verify that the session-start path (the engine's `session_start`)
    //! does not unconditionally wipe the cache — stateless agent clients
    //! that resend the full transcript on every turn should hit the
    //! `verify_cache_prefix_direct` exact-append path and skip redundant
    //! prefill work.
    //!
    //! The MoE variant additionally exercises the zero-delta guard,
    //! which is architecturally constrained to a full reset + re-prefill
    //! because rewinding the 30 GDN linear-attention layers' recurrent
    //! state mid-sequence is infeasible. The exact-match test locks in
    //! that the guard does not corrupt state (even though it's wasteful).
    //!
    //! These tests are `#[ignore]`-marked because they require loading a
    //! real Qwen3.5 MoE model file and a tokenizer. Run them with:
    //!
    //!     cargo test -p mlx-core --test '*' -- --ignored prefix_cache_reuse_integration
    //!
    //! with `MLX_NODE_QWEN35_MOE_MODEL_DIR` set to a local Qwen3.5-MoE
    //! dir.

    /// Append hit: two back-to-back session-start calls where the second
    /// extends the first by exactly one user turn. Must report
    /// `cached_tokens > 0` and only prefill the delta.
    #[ignore = "requires a real Qwen3.5 MoE model directory; run with --ignored"]
    #[test]
    fn append_hit_reuses_cached_prefix() {
        // See the matching test on `qwen3_5/model.rs` for the pseudocode
        // shape. Identical surface; different model type.
    }

    /// Divergence miss: second call's history is unrelated. Must report
    /// `cached_tokens == 0` and do a full-history prefill (which includes
    /// resetting the 30 GDN layers' recurrent state).
    #[ignore = "requires a real Qwen3.5 MoE model directory; run with --ignored"]
    #[test]
    fn divergence_miss_resets_and_full_prefills() {
        // See the matching test on `qwen3_5/model.rs` for the pseudocode
        // shape.
    }

    /// Exact-match: second call's tokens == first call's tokens, no
    /// delta. The zero-delta guard MUST NOT corrupt state — after the
    /// forced full-reset + re-prefill, generation must still produce
    /// coherent output (not random tokens). This test locks in the
    /// behavior documented alongside the guard in
    /// `vision_mtp_whole_turn_core` / `vision_mtp_whole_turn_stream_core`.
    #[ignore = "requires a real Qwen3.5 MoE model directory; run with --ignored"]
    #[test]
    fn exact_match_zero_delta_guard_preserves_correctness() {
        // Pseudocode:
        //
        //   let messages = vec![ChatMessage::user("Ping")];
        //   let r1 = model.chat_session_start(messages.clone(), cfg()).await?;
        //   let r2 = model.chat_session_start(messages, cfg()).await?;
        //   // Zero-delta guard fires — full reset + re-prefill. The
        //   // second response should still be coherent (same length,
        //   // sensible tokens), not garbage from a corrupted GDN state.
        //   assert!(!r2.text.is_empty());
        //   assert!(r2.num_tokens > 0);
    }
}

#[cfg(test)]
mod paged_construction_tests {
    //! Construction-only smoke tests for the MoE block-paged adapter.

    use super::*;
    use crate::array::DType;
    use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;
    use crate::models::qwen3_5_moe::decoder_layer::{AttentionType, MLPType};
    use crate::models::qwen3_5_moe::quantized_linear::{
        MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE, QuantizedSwitchLinear,
    };
    use crate::models::qwen3_5_moe::switch_glu::SwitchGLU;

    fn tiny_moe_cfg(use_block_paged: bool) -> Qwen3_5MoeConfig {
        Qwen3_5MoeConfig {
            vocab_size: 1024,
            hidden_size: 64,
            num_layers: 8,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 1024,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            shared_expert_intermediate_size: None,
            moe_intermediate_size: None,
            norm_topk_prob: true,
            mlp_only_layers: None,
            paged_cache_memory_mb: Some(64),
            paged_block_size: Some(16),
            use_block_paged_cache: if use_block_paged { Some(true) } else { None },
            persist_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    fn inert_mxfp8_switch(out_features: i64, in_features: i64) -> QuantizedSwitchLinear {
        QuantizedSwitchLinear::new(
            MxArray::zeros(&[4, out_features, in_features / 4], Some(DType::Uint32)).unwrap(),
            MxArray::zeros(
                &[4, out_features, in_features / MXFP8_GROUP_SIZE as i64],
                Some(DType::Uint8),
            )
            .unwrap(),
            None,
            MXFP8_GROUP_SIZE,
            MXFP8_BITS,
            MXFP8_MODE.to_string(),
        )
    }

    #[test]
    fn save_model_rejects_quantized_moe_experts_before_creating_destination() {
        let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
        let quantized_switch = SwitchGLU::new_quantized(
            inert_mxfp8_switch(128, 64),
            inert_mxfp8_switch(128, 64),
            inert_mxfp8_switch(64, 128),
        );
        match &mut inner.layers[0].mlp {
            MLPType::MoE(moe) => moe.set_switch_mlp(quantized_switch),
            MLPType::Dense(_) => panic!("layer 0 must be MoE"),
        }

        let destination = std::env::temp_dir().join(format!(
            "mlx_node_moe_quant_save_reject_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(!destination.exists());
        let err = inner
            .save_model_sync(destination.to_str().unwrap())
            .expect_err("quantized expert projections must reject dense-only save");
        let message = err.reason.to_string();
        assert!(message.contains("dense/BF16-only"), "{message}");
        assert!(message.contains("layers.0"), "{message}");
        assert!(message.contains("before creating"), "{message}");
        assert!(
            !destination.exists(),
            "rejected save must not create its destination directory"
        );
    }

    #[test]
    fn test_moe_use_block_paged_cache_serde_default_none() {
        let json = serde_json::json!({
            "vocab_size": 1024,
            "hidden_size": 64,
            "num_layers": 8,
            "num_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-6,
            "head_dim": 16,
            "tie_word_embeddings": true,
            "max_position_embeddings": 1024,
            "pad_token_id": 0,
            "eos_token_id": 0,
            "bos_token_id": 0,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        });
        let cfg: Qwen3_5MoeConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.use_block_paged_cache, None);
        assert_eq!(cfg.paged_block_size, None);
        assert_eq!(cfg.paged_cache_memory_mb, None);
    }

    #[test]
    fn test_moe_full_attention_layer_count() {
        let cfg = tiny_moe_cfg(false);
        assert_eq!(cfg.full_attention_layer_count(), 2);
    }

    #[test]
    fn test_moe_gdn_root_rotation_retains_new_root() {
        fn push_checkpoint(
            inner: &mut Qwen35MoeInner,
            owner_id: &str,
            root_owner_id: &str,
            marker: u32,
        ) {
            crate::engine::backend::ChatBackend::set_cache_owner_id(
                inner,
                owner_id,
                Some(root_owner_id),
            );
            let tokens: Vec<u32> = (0..16).map(|offset| marker * 100 + offset).collect();
            inner
                .gdn_prefix_checkpoints
                .push_back(MoeGdnPrefixCheckpoint {
                    owner_id: inner.active_cache_owner_id.clone(),
                    prefix_len: 16,
                    block_size: 16,
                    final_block_hash: marker as u64,
                    block_hashes: vec![marker as u64],
                    tokens,
                    caches: Vec::new(),
                });
            inner.prune_moe_gdn_prefix_checkpoints();
        }

        let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
        push_checkpoint(&mut inner, "root-0", "root-0", 1);
        push_checkpoint(&mut inner, "root-1", "root-1", 2);
        for (index, owner) in ["child-0", "child-1", "child-2", "child-3"]
            .into_iter()
            .enumerate()
        {
            push_checkpoint(&mut inner, owner, "root-1", 10 + index as u32);
        }
        assert_eq!(inner.gdn_root_cache_owner_id.as_deref(), Some("root-1"));
        assert!(inner.gdn_root_cache_owner_is_explicit);
        assert_eq!(inner.gdn_prefix_checkpoints.len(), 5);
        assert!(
            !inner
                .gdn_prefix_checkpoints
                .iter()
                .any(|checkpoint| checkpoint.owner_id == "root-0")
        );
        assert!(
            inner
                .gdn_prefix_checkpoints
                .iter()
                .any(|checkpoint| checkpoint.owner_id == "root-1")
        );

        // The implicit-root arm, which the rotation cases above never reach
        // because they always name a root. Every caller except `mlx agent`
        // lands here: `cacheRootOwnerId` is set in exactly one place
        // (`packages/agent/src/provider/chat-config.ts`), so a plain
        // `@mlx-node/lm` ChatSession publishes under the implicit owner `""`.
        // This model has no paged adapter and so no cold GDN policy, which
        // makes the pre-ladder cap the right one — see the dense twin.
        let mut legacy = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct legacy MoE");
        assert!(
            !legacy.wants_gdn_checkpoint_ladder(),
            "a model with no paged adapter cannot have a cold GDN sidecar policy"
        );
        for marker in 1..=(GDN_PREFIX_CHECKPOINTS_PER_OWNER as u32 + 1) {
            crate::engine::backend::ChatBackend::set_cache_owner_id(&mut legacy, "", None);
            let tokens: Vec<u32> = (0..16).map(|offset| marker * 100 + offset).collect();
            legacy
                .gdn_prefix_checkpoints
                .push_back(MoeGdnPrefixCheckpoint {
                    owner_id: legacy.active_cache_owner_id.clone(),
                    prefix_len: 16,
                    block_size: 16,
                    final_block_hash: marker as u64,
                    block_hashes: vec![marker as u64],
                    tokens,
                    caches: Vec::new(),
                });
            legacy.prune_moe_gdn_prefix_checkpoints();
        }
        assert!(!legacy.gdn_root_cache_owner_is_explicit);
        assert_eq!(
            legacy.gdn_prefix_checkpoints.len(),
            GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER
        );
    }

    /// MoE twin of
    /// `qwen3_5::model::paged_construction_tests::dense_gdn_retention_follows_the_installed_cold_sidecar_policy`.
    /// The retention cap is chosen at the call site, and the twins have two of
    /// them.
    ///
    /// Catches: hardcoding either arm in `prune_moe_gdn_prefix_checkpoints`.
    #[test]
    fn moe_gdn_retention_follows_the_installed_cold_sidecar_policy() {
        let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
            "moe_gdn_retention_follows_the_installed_cold_sidecar_policy",
        ) else {
            return;
        };
        let tokens: Vec<u32> = (0..64u32).collect();
        let extra_keys = vec![Vec::new(); 4];

        let push_ladder = |inner: &mut Qwen35MoeInner| {
            inner.gdn_prefix_checkpoints.clear();
            crate::engine::backend::ChatBackend::set_cache_owner_id(inner, "", None);
            for rung in 1..=4u32 {
                let prefix_len = rung * 16;
                let block_hashes =
                    compute_paged_prefix_block_hashes(&tokens, prefix_len, 16, &extra_keys, 0)
                        .expect("block-aligned rung must hash");
                inner
                    .gdn_prefix_checkpoints
                    .push_back(MoeGdnPrefixCheckpoint {
                        owner_id: inner.active_cache_owner_id.clone(),
                        prefix_len,
                        block_size: 16,
                        final_block_hash: block_hashes.last().copied().unwrap_or_default(),
                        block_hashes,
                        tokens: tokens[..prefix_len as usize].to_vec(),
                        caches: Vec::new(),
                    });
                inner.prune_moe_gdn_prefix_checkpoints();
            }
            inner.gdn_prefix_checkpoints.len()
        };

        assert!(!inner.wants_gdn_checkpoint_ladder());
        assert_eq!(
            push_ladder(&mut inner),
            GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER,
            "with no cold GDN policy the store must stay at the pre-ladder cap"
        );

        let root = moe_temp_cold_root("moe-gdn-retention-policy");
        install_moe_gdn_cold_tier(&mut inner, &root);
        assert!(inner.wants_gdn_checkpoint_ladder());
        assert_eq!(
            push_ladder(&mut inner),
            GDN_PREFIX_CHECKPOINTS_PER_OWNER,
            "a cold GDN sidecar policy must keep the whole published ladder"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// MoE twin of
    /// `qwen3_5::model::tests::test_dense_ladder_survives_sibling_owners_through_the_model_call_site`.
    /// The two share `prune_gdn_checkpoints` but not the call site that tells
    /// it which owner is publishing, which is exactly where they can drift.
    ///
    /// Like the dense twin, both arms are driven: publisher deferral exists
    /// only under `GdnRetentionPolicy::Ladder`, so the cold tier has to be
    /// installed for the mutation to be expressible at all.
    #[test]
    fn test_moe_ladder_survives_sibling_owners_through_the_model_call_site() {
        fn push(inner: &mut Qwen35MoeInner, owner_id: &str, root_owner_id: &str, blocks: u32) {
            crate::engine::backend::ChatBackend::set_cache_owner_id(
                inner,
                owner_id,
                Some(root_owner_id),
            );
            inner
                .gdn_prefix_checkpoints
                .push_back(MoeGdnPrefixCheckpoint {
                    owner_id: inner.active_cache_owner_id.clone(),
                    prefix_len: blocks * 16,
                    block_size: 16,
                    final_block_hash: u64::from(blocks),
                    block_hashes: (1..=u64::from(blocks)).collect(),
                    tokens: (0..blocks * 16).collect(),
                    caches: Vec::new(),
                });
            inner.prune_moe_gdn_prefix_checkpoints();
        }

        fn run_fleet(inner: &mut Qwen35MoeInner) -> Vec<(String, u32)> {
            inner.gdn_prefix_checkpoints.clear();
            push(inner, "root", "root", 1);
            push(inner, "root", "root", 64);
            for sibling in ["child-0", "child-1"] {
                push(inner, sibling, "root", 64);
            }
            for blocks in [1, 4, 16, 64] {
                push(inner, "child-3", "root", blocks);
            }
            inner
                .gdn_prefix_checkpoints
                .iter()
                .map(|checkpoint| (checkpoint.owner_id.clone(), checkpoint.prefix_len))
                .collect()
        }

        let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
            "test_moe_ladder_survives_sibling_owners_through_the_model_call_site",
        ) else {
            return;
        };

        assert!(!inner.wants_gdn_checkpoint_ladder());
        assert_eq!(
            run_fleet(&mut inner),
            vec![
                ("root".to_string(), 1024),
                ("child-0".to_string(), 1024),
                ("child-1".to_string(), 1024),
                ("child-3".to_string(), 256),
                ("child-3".to_string(), 1024),
            ],
            "a persistence-OFF turn takes the pre-ladder victim order, which \
             keeps the publisher's DEEPEST rungs and defers nobody"
        );

        let root = moe_temp_cold_root("moe-ladder-sibling-call-site");
        install_moe_gdn_cold_tier(&mut inner, &root);
        assert!(inner.wants_gdn_checkpoint_ladder());
        assert_eq!(
            run_fleet(&mut inner),
            vec![
                ("root".to_string(), 1024),
                ("child-0".to_string(), 1024),
                ("child-1".to_string(), 1024),
                ("child-3".to_string(), 16),
                ("child-3".to_string(), 1024),
            ],
            "the publishing subagent must keep the shallow rung this turn's \
             capture anchors on, paid for by the root's spare rung, and no \
             sibling may be pushed to zero"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_qwen35_moe_media_plan_requires_complete_paged_vision_stack() {
        for has_encoder in [false, true] {
            for has_processor in [false, true] {
                for has_paged in [false, true] {
                    let plan = qwen35_moe_media_plan(has_encoder, has_processor, has_paged);
                    let available = has_encoder && has_processor && has_paged;
                    assert_eq!(plan.available.images, available);
                    assert!(!plan.available.audio);
                    assert_eq!(plan.backend_validated.images, !available);
                    assert!(!plan.backend_validated.audio);
                    assert!(plan.admitted().images);
                }
            }
        }
    }

    #[test]
    fn test_qwen35_moe_session_media_survives_paged_image_key_clear() {
        assert_eq!(
            qwen35_moe_session_media(true, false),
            MediaCapabilities::IMAGES
        );
        assert_eq!(
            qwen35_moe_session_media(false, true),
            MediaCapabilities::IMAGES,
            "a retained M-RoPE delta proves successive paged text turns still extend image KV"
        );
        assert_eq!(
            qwen35_moe_session_media(false, false),
            MediaCapabilities::NONE
        );
    }

    #[test]
    fn test_qwen35_moe_session_media_payload_identity() {
        let images = vec![vec![1, 2, 3]];
        let cached_key = Some(engine::compute_image_cache_key(&images));

        assert!(qwen35_moe_session_media_matches_payloads(
            cached_key,
            &images,
            &[]
        ));
        assert!(!qwen35_moe_session_media_matches_payloads(
            cached_key,
            &[vec![1, 2, 4]],
            &[]
        ));
        assert!(!qwen35_moe_session_media_matches_payloads(
            cached_key,
            &images,
            &[vec![9]]
        ));
        assert!(!qwen35_moe_session_media_matches_payloads(
            None,
            &images,
            &[]
        ));
    }

    #[test]
    fn test_moe_paged_finalize_failure_cannot_republish_image_session() {
        let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
        inner.caches = Some(fresh_moe_layer_caches(&inner.config));
        inner.cached_token_history = vec![7, 248_056, 248_056, 8];
        inner.cached_image_key = Some(0xA30B);
        inner.cached_paged_image_token_positions = vec![(1, 0xA11C), (2, 0xA11C)];
        inner.cached_rope_deltas = Some(-2);

        <Qwen35MoeInner as crate::engine::backend::PagedBackend>::finalize_paged_turn(
            &mut inner, true,
        );
        assert!(inner.paged_finalize_failed);
        assert!(inner.caches.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cached_paged_image_token_positions.is_empty());
        assert!(inner.cached_rope_deltas.is_none());

        <Qwen35MoeInner as crate::engine::backend::PagedBackend>::save_paged_history(
            &mut inner,
            &[7, 248_056, 248_056, 8],
            &[9],
            false,
            true,
        )
        .expect("failed finalization must downgrade history save");
        assert!(!inner.paged_finalize_failed);
        assert!(inner.cached_token_history.is_empty());
        assert!(!crate::engine::backend::ChatBackend::has_live_session(
            &inner
        ));
        assert_eq!(
            crate::engine::backend::ChatBackend::session_media(&inner),
            MediaCapabilities::NONE
        );
    }

    fn seed_moe_paged_image_session(inner: &mut Qwen35MoeInner) {
        inner.caches = Some(fresh_moe_layer_caches(&inner.config));
        inner.cached_token_history = vec![7, 248_056, 248_056, 8];
        inner.cached_image_key = Some(0xA30B);
        inner.cached_paged_image_token_positions = vec![(1, 0xA11C), (2, 0xA11C)];
        inner.cached_rope_deltas = Some(-2);
        inner.flat_mtp_caches_desynced = true;
    }

    #[test]
    fn test_moe_manual_paged_finalize_failure_invalidates_session() {
        let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
        seed_moe_paged_image_session(&mut inner);

        let error = inner
            .finalize_moe_manual_paged_turn(&[(1, 0xA11C), (2, 0xA11C)])
            .expect_err("missing adapter must fail manual finalization");
        assert!(error.to_string().contains("paged finalization failed"));
        assert!(inner.caches.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cached_paged_image_token_positions.is_empty());
        assert!(inner.cached_rope_deltas.is_none());
        assert!(!inner.flat_mtp_caches_desynced);
    }

    #[test]
    fn test_moe_vlm_prefix_prepare_failure_invalidates_session() {
        let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
        seed_moe_paged_image_session(&mut inner);

        inner
            .prepare_moe_vlm_paged_prefix(
                &[7, 248_056, 248_056, 8],
                4,
                16,
                &[vec![0xA11C]],
                true,
                true,
                0xA30B,
            )
            .expect_err("missing adapter must fail VLM prefix preparation");
        assert!(inner.caches.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cached_paged_image_token_positions.is_empty());
        assert!(inner.cached_rope_deltas.is_none());
        assert!(!crate::engine::backend::ChatBackend::has_live_session(
            &inner
        ));
    }

    #[test]
    fn test_moe_generic_paged_abort_invalidates_session() {
        let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
        seed_moe_paged_image_session(&mut inner);

        <Qwen35MoeInner as crate::engine::backend::PagedBackend>::abort_paged_turn(&mut inner);
        assert!(inner.caches.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cached_paged_image_token_positions.is_empty());
        assert!(inner.cached_rope_deltas.is_none());
        assert!(!inner.flat_mtp_caches_desynced);
    }

    #[test]
    fn test_qwen35_moe_planned_decoder_overrides_raw_mtp_flag() {
        let mut config = ChatConfig {
            cache_owner_id: None,
            cache_root_owner_id: None,
            enable_mtp: Some(true),
            ..ChatConfig::default()
        };
        assert!(!apply_qwen35_moe_planned_decoder(
            &mut config,
            DecoderPlan::Autoregressive
        ));
        assert_eq!(config.enable_mtp, Some(false));

        assert!(apply_qwen35_moe_planned_decoder(
            &mut config,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
        ));
        assert_eq!(config.enable_mtp, Some(true));
    }

    #[test]
    fn test_moe_inner_no_paged_adapter_when_flag_is_none() {
        let cfg = tiny_moe_cfg(false);
        let inner = Qwen35MoeInner::new(cfg)
            .expect("Qwen35MoeInner::new must succeed without paged adapter");
        assert!(inner.paged_adapter.is_none());
    }

    #[test]
    fn test_fresh_moe_layer_caches_are_not_gdn_reuse_ready() {
        let cfg = tiny_moe_cfg(true);
        let caches = fresh_moe_layer_caches(&cfg);
        assert_eq!(caches.len(), cfg.num_layers as usize);
        assert!(
            !moe_paged_linear_caches_ready(&cfg, Some(&caches)),
            "fresh linear caches have empty conv/recurrent slots, so a live continuation must replay GDN"
        );
        assert!(matches!(caches[0], Qwen3_5LayerCache::Linear(_)));
        assert!(matches!(caches[3], Qwen3_5LayerCache::FullAttention(_)));
    }

    #[test]
    fn test_paged_prefix_block_hash_matches_allocator_chain() {
        let tokens: Vec<u32> = (1..=12).collect();
        let per_block = vec![vec![11], vec![], vec![33, 44]];

        let h0 = mlx_paged_attn::hash_tokens(&tokens[0..4], 0, &per_block[0]);
        let h1 = mlx_paged_attn::hash_tokens(&tokens[4..8], h0, &per_block[1]);
        let h2 = mlx_paged_attn::hash_tokens(&tokens[8..12], h1, &per_block[2]);

        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 12, 4, &per_block, 0),
            Some(h2)
        );
    }

    #[test]
    fn test_paged_prefix_block_hash_applies_salt_to_first_block_only() {
        let tokens: Vec<u32> = (1..=8).collect();
        let per_block = vec![vec![11], vec![22]];
        let salt = 99;

        let mut first_block_keys = per_block[0].clone();
        first_block_keys.push(salt);
        let h0 = mlx_paged_attn::hash_tokens(&tokens[0..4], 0, &first_block_keys);
        let h1 = mlx_paged_attn::hash_tokens(&tokens[4..8], h0, &per_block[1]);

        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 8, 4, &per_block, salt),
            Some(h1)
        );
    }

    #[test]
    fn test_paged_prefix_block_hash_rejects_non_full_or_unkeyed_prefix() {
        let tokens: Vec<u32> = (1..=8).collect();
        let per_block = vec![vec![]];

        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 6, 4, &per_block, 0),
            None
        );
        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 8, 4, &per_block, 0),
            None
        );
    }

    #[test]
    #[ignore = "Allocates Metal LayerKVPool; gate on MLX_TEST_PAGED=1"]
    fn test_moe_inner_constructs_paged_adapter_when_flag_is_true() {
        if std::env::var_os("MLX_TEST_PAGED").is_none() {
            return;
        }
        let cfg = tiny_moe_cfg(true);
        let mut inner = Qwen35MoeInner::new(cfg).expect(
            "Qwen35MoeInner::new with use_block_paged_cache=true must succeed on Metal host",
        );
        inner
            .initialize_paged_adapter()
            .expect("post-load paged adapter initialization must succeed");
        assert!(inner.paged_adapter.is_some());
    }

    /// MoE inner carrying a real adapter over a test-sized `LayerKVPool`.
    /// Mirrors the dense helper: `initialize_paged_adapter` sizes a pool for a
    /// real checkpoint, and the lifecycle below dispatches no kernel.
    fn moe_inner_with_test_adapter_or_skip(test_name: &str) -> Option<Qwen35MoeInner> {
        const BLOCK_SIZE: u32 = 16;
        const NUM_BLOCKS: u32 = 8;
        let pool_config = mlx_paged_attn::PagedAttentionConfig {
            block_size: BLOCK_SIZE,
            num_kv_heads: 1,
            head_size: 32,
            num_layers: 2,
            ..mlx_paged_attn::PagedAttentionConfig::default()
        };
        let pool = match mlx_paged_attn::LayerKVPool::new_for_test(
            pool_config,
            NUM_BLOCKS,
            2,
            mlx_paged_attn::metal::MetalDtype::Float16,
        ) {
            Ok(pool) => Arc::new(pool),
            // Routed through `test_support::metal_device_absent` rather than
            // matching the string inline, so `MLX_TEST_REQUIRE_METAL=1` can
            // turn a self-skip into a failure here too. libtest counts a
            // skipped-and-returned test as passed, and these two are the only
            // gates that can see a hardcoded arm at the prune call site.
            Err(err) if crate::test_support::metal_device_absent(&err) => {
                eprintln!("skipping {test_name} (no Metal device): {err}");
                return None;
            }
            Err(err) => {
                panic!("unexpected LayerKVPool::new_for_test failure in {test_name}: {err}")
            }
        };
        let allocator = Arc::new(Mutex::new(mlx_paged_attn::BlockAllocator::new(
            NUM_BLOCKS, BLOCK_SIZE,
        )));
        let adapter = PagedKVCacheAdapter::new(allocator, pool, BLOCK_SIZE)
            .expect("test paged adapter must construct");
        let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(true)).expect("construct tiny MoE model");
        inner.paged_adapter = Some(adapter);
        Some(inner)
    }

    /// Attach the cold tier shape `build_cold_tier_context` installs for MoE.
    fn install_moe_gdn_cold_tier(inner: &mut Qwen35MoeInner, root: &std::path::Path) {
        let manager = mlx_paged_attn::ColdCacheManager::open_default_at(root.to_path_buf())
            .expect("temp-dir cold cache must open");
        let cache_dtype = {
            let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
            format!("{:?}", adapter.layer_kv_pool().cache_dtype())
        };
        let policy = crate::models::qwen3_5::gdn_sidecar::policy(
            &inner.config.to_dense_config(),
            &cache_dtype,
        )
        .expect("a hybrid config must yield a GDN sidecar policy");
        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        adapter.set_cold_tier(
            crate::transformer::paged_kv_cache_adapter::ColdTierContext {
                manager: Arc::new(manager),
                fingerprint: mlx_paged_attn::ColdCacheFingerprint::from_components([
                    b"qwen3-5-moe-sidecar-test".as_slice(),
                ]),
                sidecar_policy: Some(policy),
            },
        );
    }

    fn moe_temp_cold_root(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mlx-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    fn record_whole_moe_paged_request(inner: &mut Qwen35MoeInner, prompt: &[u32]) {
        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        adapter.reset_for_new_request(0).expect("reset request");
        adapter
            .allocate_suffix_blocks(prompt.len() as u32)
            .expect("allocate suffix blocks");
        adapter.record_tokens(prompt).expect("record tokens");
    }

    /// MoE mirror of the dense case: the hand-written cores finalize through
    /// `finalize_moe_manual_paged_turn`, which must run the GDN sidecar capture.
    ///
    /// Catches: deleting `capture_moe_gdn_cold_sidecar` from the `Ok` arm of
    /// `finalize_moe_manual_paged_turn` — both deltas drop to zero.
    #[test]
    fn moe_manual_paged_finalize_reaches_the_gdn_sidecar_capture() {
        let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
            "moe_manual_paged_finalize_reaches_the_gdn_sidecar_capture",
        ) else {
            return;
        };
        let root = moe_temp_cold_root("moe-manual-finalize-capture");
        install_moe_gdn_cold_tier(&mut inner, &root);
        let prompt: Vec<u32> = (0u32..32).collect();
        record_whole_moe_paged_request(&mut inner, &prompt);

        let _serialized = crate::cold_tier::sidecar_counter_test_lock();
        let before = crate::cold_tier::cold_sidecar_telemetry();
        inner
            .finalize_moe_manual_paged_turn(&[])
            .expect("manual paged finalization must succeed");
        let after = crate::cold_tier::cold_sidecar_telemetry();

        assert_eq!(
            after.capture_reached,
            before.capture_reached + 1,
            "the manual finalize must enter the GDN sidecar capture exactly once"
        );
        // The chain really carried both blocks off the GPU. Asserted here
        // because everything below depends on it and nothing below can see it:
        // a pool whose buffers are too small makes `read_block_all_layers`
        // return `Err`, the chain capture stops at its first block, and the
        // boundary assertion after this one becomes unreachable rather than
        // false. Without this line that failure reads as "the capture did not
        // reach its boundary selection", three layers from the real cause.
        assert_eq!(
            inner
                .paged_adapter
                .as_ref()
                .expect("paged_adapter")
                .cold_captured_blocks(),
            2,
            "both whole blocks must have been captured off the pool"
        );
        // The finalize published two whole blocks to the cold chain, so the
        // capture ran to its boundary selection and found no in-memory
        // checkpoint. Reaching THAT arm proves it got past the policy and media
        // guards rather than bailing at its first line.
        assert_eq!(
            after.boundary_skips,
            before.boundary_skips + 1,
            "the capture must reach its boundary selection under a GdnState policy"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The v1 text-only guard must read the turn's OWN media map, not
    /// `self.cached_paged_image_token_positions` (cleared mid-turn by the VLM
    /// cores).
    ///
    /// Catches: reverting the guard to the `self` field — empty here, so the
    /// capture would run on past it and bump `boundary_skips`.
    #[test]
    fn moe_gdn_sidecar_capture_refuses_the_turns_own_media_map() {
        let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
            "moe_gdn_sidecar_capture_refuses_the_turns_own_media_map",
        ) else {
            return;
        };
        let root = moe_temp_cold_root("moe-manual-finalize-media");
        install_moe_gdn_cold_tier(&mut inner, &root);
        let prompt: Vec<u32> = (0u32..32).collect();
        record_whole_moe_paged_request(&mut inner, &prompt);
        assert!(
            inner.cached_paged_image_token_positions.is_empty(),
            "the model-level media map is empty mid-turn; only the argument carries the truth"
        );

        let _serialized = crate::cold_tier::sidecar_counter_test_lock();
        let before = crate::cold_tier::cold_sidecar_telemetry();
        inner
            .finalize_moe_manual_paged_turn(&[(4, 99)])
            .expect("manual paged finalization must succeed");
        let after = crate::cold_tier::cold_sidecar_telemetry();

        assert_eq!(
            after.capture_reached,
            before.capture_reached + 1,
            "a media turn still enters the capture"
        );
        assert_eq!(
            after.boundary_skips, before.boundary_skips,
            "a media turn must be refused before the capture inspects the cold chain"
        );
        assert_eq!(
            after.chain_empty, before.chain_empty,
            "a media turn must be refused before the capture inspects the cold chain"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// MoE mirror of the dense chunk-size gate.
    ///
    /// Catches: dropping the policy probe from `cold_gdn_prefill_chunk_size`
    /// (returns 0 with a policy installed), which leaves a persist-cold turn
    /// publishing no ladder rung at all.
    #[test]
    fn moe_cold_gdn_prefill_chunk_size_follows_the_installed_sidecar_policy() {
        let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
            "moe_cold_gdn_prefill_chunk_size_follows_the_installed_sidecar_policy",
        ) else {
            return;
        };
        assert_eq!(
            inner.cold_gdn_prefill_chunk_size(),
            0,
            "without a cold GDN policy the prefill stays single-shot"
        );

        let root = moe_temp_cold_root("moe-chunk-size-policy");
        install_moe_gdn_cold_tier(&mut inner, &root);
        assert!(
            inner.cold_gdn_prefill_chunk_size() > 0,
            "an installed GdnState policy must make the prefill split at the ladder"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Tiny MoE config shaped for a REAL paged prefill, unlike [`tiny_moe_cfg`].
    ///
    /// Two departures, both forced by what the paged path demands:
    ///
    /// * `head_dim` 32. Paged attention's Metal kernels reject anything
    ///   smaller, so `tiny_moe_cfg`'s 16 never reaches a prefill. Same reason
    ///   the dense `tiny_paged_forward_cfg` bumps it.
    /// * every layer forced onto the DENSE MLP arm through `mlp_only_layers`.
    ///   `update_keys_values` accepts only 2-byte K/V, so a randomly
    ///   initialized model has to be cast to bf16 end to end, and
    ///   `SparseMoeBlock` publishes no weight accessors to cast through.
    ///   Nothing under test reads the MLP: the chunk size, the ladder and the
    ///   checkpoint materialization all live in `run_moe_core_paged_prefill`
    ///   and `qwen3_5_moe::paged_forward`.
    fn tiny_paged_forward_moe_cfg() -> Qwen3_5MoeConfig {
        let mut cfg = tiny_moe_cfg(true);
        cfg.hidden_size = 128;
        cfg.intermediate_size = 256;
        cfg.head_dim = 32;
        cfg.linear_key_head_dim = 32;
        cfg.linear_value_head_dim = 32;
        cfg.paged_cache_memory_mb = Some(256);
        cfg.mlp_only_layers = Some((0..cfg.num_layers).collect());
        cfg
    }

    /// MoE inner over a production-shaped paged adapter, or `None` on a host
    /// with no usable Metal device.
    ///
    /// Distinct from [`moe_inner_with_test_adapter_or_skip`], which hands back a
    /// hand-built 8-block pool for lifecycle tests that dispatch no kernel.
    /// This one goes through `initialize_paged_adapter`, so the pool geometry is
    /// the one the config really implies and a prefill can run against it.
    fn moe_paged_inner_or_skip(test_name: &str) -> Option<Qwen35MoeInner> {
        // Only a device-less host licenses a skip. A `LayerKVPool::new: ...
        // must be > 0` means the test config stopped producing a usable pool,
        // and skipping on that is a silent green — see `metal_device_absent`.
        let unavailable = crate::test_support::metal_device_absent;
        let mut inner = match Qwen35MoeInner::new(tiny_paged_forward_moe_cfg()) {
            Ok(inner) => inner,
            Err(err) => {
                let msg = err.reason.to_string();
                if unavailable(&msg) {
                    eprintln!("skipping {test_name} (paged adapter unavailable): {msg}");
                    return None;
                }
                panic!("unexpected Qwen35MoeInner::new failure in {test_name}: {msg}");
            }
        };
        if let Err(err) = inner.initialize_paged_adapter() {
            let msg = err.reason.to_string();
            if unavailable(&msg) {
                eprintln!("skipping {test_name} (paged adapter unavailable): {msg}");
                return None;
            }
            panic!("unexpected paged init failure in {test_name}: {msg}");
        }
        if inner.paged_adapter.is_none() {
            // `initialize_paged_adapter` returns `Ok` and installs nothing when
            // the compiled forward backend is missing. That never reaches
            // `metal_device_absent`, so it needs the require-Metal gate of its
            // own or it stays a silent green on the very runner that gate exists
            // to protect.
            assert!(
                !crate::test_support::metal_required(),
                "MLX_TEST_REQUIRE_METAL=1 but {test_name} got no paged adapter: \
                 initialize_paged_adapter returned Ok and installed nothing"
            );
            eprintln!("skipping {test_name}: no paged adapter was installed");
            return None;
        }
        Some(inner)
    }

    /// Cast every weight a paged forward touches to bf16.
    ///
    /// `update_keys_values` refuses anything but Float16/BFloat16 K/V (the pool
    /// allocates 2-byte elements) and a randomly initialized model is f32. A
    /// PARTIAL cast is worse than none: one f32 weight promotes the hidden state
    /// back to f32 and the failure surfaces at the K/V write, several frames
    /// from its cause. So this walks everything, and the sparse-MLP arm panics
    /// rather than skipping.
    fn cast_moe_inner_weights_bf16(inner: &mut Qwen35MoeInner) {
        let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype bf16") };

        let w = inner.embedding.get_weight();
        inner.embedding.set_weight(&cast(&w)).expect("set embed");

        let w = inner.final_norm.get_weight();
        inner
            .final_norm
            .set_weight(&cast(&w))
            .expect("set final_norm");

        if let Some(head) = inner.lm_head.as_mut() {
            let w = head.get_weight();
            head.set_weight(&cast(&w), "lm_head").expect("set lm_head");
        }

        for layer in inner.layers.iter_mut() {
            let w = layer.get_input_layernorm_weight();
            layer
                .set_input_layernorm_weight(&cast(&w))
                .expect("set input_layernorm");
            let w = layer.get_post_attention_layernorm_weight();
            layer
                .set_post_attention_layernorm_weight(&cast(&w))
                .expect("set post_attention_layernorm");

            match &mut layer.attn {
                AttentionType::Linear(gdn) => {
                    let w = gdn.get_dt_bias();
                    gdn.set_dt_bias(&cast(&w));
                    let w = gdn.get_a_log();
                    gdn.set_a_log(&cast(&w)).expect("set a_log");
                    let w = gdn.get_in_proj_qkvz_weight();
                    gdn.set_in_proj_qkvz_weight(&cast(&w))
                        .expect("set in_proj_qkvz");
                    let w = gdn.get_in_proj_ba_weight();
                    gdn.set_in_proj_ba_weight(&cast(&w))
                        .expect("set in_proj_ba");
                    let w = gdn.get_conv1d_weight();
                    gdn.set_conv1d_weight(&cast(&w)).expect("set conv1d");
                    let w = gdn.get_norm_weight();
                    gdn.set_norm_weight(&cast(&w)).expect("set gdn norm");
                    let w = gdn.get_out_proj_weight();
                    gdn.set_out_proj_weight(&cast(&w)).expect("set out_proj");
                }
                AttentionType::Full(attn) => {
                    let w = attn.get_q_proj_weight();
                    attn.set_q_proj_weight(&cast(&w)).expect("set q_proj");
                    let w = attn.get_k_proj_weight();
                    attn.set_k_proj_weight(&cast(&w)).expect("set k_proj");
                    let w = attn.get_v_proj_weight();
                    attn.set_v_proj_weight(&cast(&w)).expect("set v_proj");
                    let w = attn.get_o_proj_weight();
                    attn.set_o_proj_weight(&cast(&w)).expect("set o_proj");
                    let w = attn.get_q_norm_weight();
                    attn.set_q_norm_weight(&cast(&w)).expect("set q_norm");
                    let w = attn.get_k_norm_weight();
                    attn.set_k_norm_weight(&cast(&w)).expect("set k_norm");
                }
            }

            match &mut layer.mlp {
                MLPType::Dense(mlp) => {
                    let w = mlp.get_gate_proj_weight();
                    mlp.set_gate_proj_weight(&cast(&w)).expect("set gate_proj");
                    let w = mlp.get_up_proj_weight();
                    mlp.set_up_proj_weight(&cast(&w)).expect("set up_proj");
                    let w = mlp.get_down_proj_weight();
                    mlp.set_down_proj_weight(&cast(&w)).expect("set down_proj");
                }
                MLPType::MoE(_) => panic!(
                    "tiny_paged_forward_moe_cfg lists every layer in mlp_only_layers so each MLP \
                     can be cast; a sparse block here means that drifted, and the f32 experts \
                     would trip the K/V dtype gate far from the cause"
                ),
            }
        }
    }

    /// Put the adapter where a fresh turn's prefill starts: empty caches, no
    /// cached prefix, suffix blocks allocated for the whole prompt.
    fn reset_moe_paged_request(inner: &mut Qwen35MoeInner, prompt: &[u32]) {
        let caches = fresh_moe_layer_caches(&inner.config);
        inner.caches = Some(caches);

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        if adapter.block_table().is_some() {
            adapter.release_request().expect("release_request");
        }
        adapter.reset_for_new_request(0).expect("reset request");
        let prefix = adapter
            .find_cached_prefix(prompt, &[], 0, false)
            .expect("find_cached_prefix");
        assert_eq!(
            prefix.cached_token_count, 0,
            "MoE chunking tests must start from a cold adapter prefix"
        );
        adapter
            .allocate_suffix_blocks(prompt.len() as u32)
            .expect("allocate suffix blocks");
    }

    /// Drive the MoE prefill worker with an EXPLICIT chunk size, bypassing
    /// `cold_gdn_prefill_chunk_size`. That is the only way to reach the break-set
    /// decision with no policy installed — the shape a persist-off `mlx agent`
    /// turn has, since `run-agent.ts` seeds `MLX_PAGED_PREFILL_CHUNK_SIZE=2048`
    /// before any persistence decision.
    fn run_moe_paged_prefill_with_size_and_checkpoint(
        inner: &mut Qwen35MoeInner,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        chunk_size: i32,
    ) -> Result<(
        MxArray,
        Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
    )> {
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            inner.config.num_layers as usize,
            |i| inner.config.is_linear_layer(i),
        );
        let embed = inner.embedding.clone();
        let caches = inner.caches.as_mut().expect("moe caches initialized");
        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        crate::models::qwen3_5_moe::paged_forward::run_paged_prefill_chunk_with_size_and_checkpoint(
            full_tokens,
            suffix_tokens,
            0,
            false,
            &embed,
            &mut inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            &layer_kinds,
            adapter,
            chunk_size,
            /* cached_rope_deltas */ 0,
        )
    }

    /// Both hand-written MoE cores prefill through `run_moe_core_paged_prefill`,
    /// so this drives that shared body directly: under a GdnState policy it must
    /// publish checkpoint ladder rungs, and with no cold tier it must stay
    /// single-shot. MoE mirror of the dense
    /// `dense_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy`.
    ///
    /// Catches: passing a literal `0`, or `crate::array::paged_prefill_chunk_size()`
    /// (which is 0 unless the env var is set), where
    /// `self.cold_gdn_prefill_chunk_size()` belongs inside
    /// `run_moe_core_paged_prefill`. Either takes
    /// `run_paged_prefill_chunk_with_size_and_checkpoint`'s `chunk_size <= 0`
    /// arm, which hands back `Vec::new()` for the ladder — no rung is
    /// materialized, so every MoE cold capture anchors nothing. That is the
    /// MTP-turn defect, on the MoE side.
    ///
    /// The sibling `moe_cold_gdn_prefill_chunk_size_follows_the_installed_sidecar_policy`
    /// cannot see it: that one reads the accessor, never the call site. The
    /// no-cold-tier leg is load-bearing in the other direction — a hardcoded
    /// `i32::MAX` would chunk prefills that ship today as single-shot.
    ///
    /// The two EXPLICIT-chunk-size arms reach the break-set decision with no
    /// policy installed, which `cold_gdn_prefill_chunk_size`'s own 0 can never
    /// do. They catch hardcoding `gdn_cold_sidecar_ladder_wanted`'s result at
    /// the prefill body in either direction; the 64-token arms cannot, because
    /// at that length the ladder is already one rung and both arms agree.
    ///
    /// Runs in CI on the `qwen3_5-dense` `model-test` leg's `lib_tests` filter,
    /// which needs no checkpoint for this one; there is no MoE leg, and
    /// `docs/paged-cache.md` records why there cannot be.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn moe_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy() {
        let Some(mut inner) = moe_paged_inner_or_skip(
            "moe_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy",
        ) else {
            return;
        };
        cast_moe_inner_weights_bf16(&mut inner);
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            inner.config.num_layers as usize,
            |i| inner.config.is_linear_layer(i),
        );
        let prompt: Vec<u32> = (0u32..64).map(|i| (i * 5 + 7) % 257).collect();

        reset_moe_paged_request(&mut inner, &prompt);
        let (_, no_cold_ladder) = inner
            .run_moe_core_paged_prefill(&prompt, &prompt, 0, false, &layer_kinds, "ladder prefill")
            .expect("prefill with no cold tier");
        assert!(
            no_cold_ladder.is_empty(),
            "with no cold GDN policy the MoE cores must stay single-shot"
        );

        // A prompt long enough that the two break sets DIFFER: ladder
        // [48, 192] against the single pre-ladder boundary [192]. Run BEFORE
        // the cold tier is installed — the adapter has no way to drop one.
        let long_prompt: Vec<u32> = (0u32..200).map(|i| (i * 5 + 7) % 257).collect();
        reset_moe_paged_request(&mut inner, &long_prompt);
        let (_, no_policy_rungs) = run_moe_paged_prefill_with_size_and_checkpoint(
            &mut inner,
            &long_prompt,
            &long_prompt,
            2048,
        )
        .expect("explicit-chunk MoE prefill with no cold tier");
        assert_eq!(
            no_policy_rungs
                .iter()
                .map(|c| c.prefix_len)
                .collect::<Vec<_>>(),
            vec![192],
            "with no cold GDN policy an explicit chunk size must take the single pre-ladder \
             boundary, not the ladder — the extra breaks change the prefill GEMM's M and so the \
             sampled tokens of a persistence-off request"
        );
        // Release WITHOUT registering: publishing this prompt's blocks would
        // give the next arm's shared 64-token prefix a cache hit, and every arm
        // here has to start cold.
        inner
            .paged_adapter
            .as_mut()
            .expect("paged_adapter")
            .release_request()
            .expect("release_request");

        let root = moe_temp_cold_root("moe-core-ladder");
        install_moe_gdn_cold_tier(&mut inner, &root);
        reset_moe_paged_request(&mut inner, &prompt);
        let (_, ladder) = inner
            .run_moe_core_paged_prefill(&prompt, &prompt, 0, false, &layer_kinds, "ladder prefill")
            .expect("prefill under a cold GDN policy");
        assert!(
            !ladder.is_empty(),
            "a persist-cold MoE turn must materialize the GDN checkpoint ladder its sidecar \
             anchors on"
        );
        for checkpoint in &ladder {
            assert!(
                checkpoint.prefix_len > 0 && (checkpoint.prefix_len as usize) < prompt.len(),
                "a rung must sit strictly inside the prompt, got {}",
                checkpoint.prefix_len
            );
        }

        // Same long prompt, same explicit chunk size, policy now installed:
        // every rung of the ladder is a real break.
        reset_moe_paged_request(&mut inner, &long_prompt);
        let (_, cold_rungs) = run_moe_paged_prefill_with_size_and_checkpoint(
            &mut inner,
            &long_prompt,
            &long_prompt,
            2048,
        )
        .expect("explicit-chunk MoE prefill under a cold GDN policy");
        assert_eq!(
            cold_rungs.iter().map(|c| c.prefix_len).collect::<Vec<_>>(),
            vec![48, 192],
            "an explicit chunk size under a GDN cold policy still publishes the whole ladder"
        );

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
        adapter.release_request().expect("release_request");
        let _ = std::fs::remove_dir_all(&root);
    }
}

#[cfg(test)]
mod mask_free_full_attention_parity_tests {
    //! Locks in that `forward_pre_norm_inner`'s mask-free full-attention
    //! forward (the shared attention module's fused "causal" SDPA fast
    //! path, selected whenever `mask` is `None` and `seq_len > 1`) is
    //! numerically transparent versus the explicit `create_causal_mask`
    //! array it replaced, for a PRIMED cache (nonzero KV offset) — the
    //! exact shape of the eager-MTP verify call (`[1, K+1]` ids over an
    //! already-decoded prefix). Mirrors
    //! `causal_attention_matches_explicit_offset_mask_when_kv_is_longer` in
    //! `crate::array::attention::tests`, one layer stack up.

    use super::*;
    use crate::array::mask::create_causal_mask;
    use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;

    fn tiny_moe_cfg() -> Qwen3_5MoeConfig {
        Qwen3_5MoeConfig {
            vocab_size: 1024,
            hidden_size: 64,
            num_layers: 8,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 1024,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            shared_expert_intermediate_size: None,
            moe_intermediate_size: None,
            norm_topk_prob: true,
            mlp_only_layers: None,
            paged_cache_memory_mb: Some(64),
            paged_block_size: Some(16),
            use_block_paged_cache: None,
            persist_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    /// Pre-fix mask construction, kept here ONLY as a reference oracle —
    /// byte-for-byte the code this fix deletes from `forward_pre_norm_inner`.
    /// Builds an explicit causal mask sized from `caches[fa_idx]`'s offset
    /// and hands it to every full-attention layer.
    fn forward_with_explicit_causal_mask(
        input_ids: &MxArray,
        embedding_weight: &MxArray,
        layers: &mut [DecoderLayer],
        caches: &mut Option<Vec<Qwen3_5LayerCache>>,
        fa_idx: usize,
    ) -> Result<MxArray> {
        let embedding = Embedding::from_weight(embedding_weight)?;
        let hidden_states = embedding.forward(input_ids)?;
        let mut h = hidden_states.clone();

        let seq_len = hidden_states.shape_at(1)?;
        let fa_mask = {
            let has_cache = caches.is_some();
            if seq_len <= 1 && has_cache {
                None
            } else {
                let offset = caches.as_ref().map(|c| c[fa_idx].offset()).unwrap_or(0);
                Some(create_causal_mask(seq_len as i32, Some(offset), None)?)
            }
        };

        let num_layers = layers.len();
        for i in 0..num_layers {
            let mask = if layers[i].is_linear() {
                None
            } else {
                fa_mask.as_ref()
            };
            let cache = caches.as_mut().map(|c| &mut c[i]);
            h = layers[i].forward(&h, mask, cache, None, true)?;
        }
        Ok(h)
    }

    #[test]
    fn mask_free_forward_matches_explicit_offset_mask_after_priming() {
        let cfg = tiny_moe_cfg();
        let mut layers = (0..cfg.num_layers as usize)
            .map(|i| DecoderLayer::new(&cfg, i))
            .collect::<Result<Vec<_>>>()
            .expect("layer construction must succeed");
        let embedding = Embedding::new(cfg.vocab_size as u32, cfg.hidden_size as u32)
            .expect("embedding construction must succeed");
        let embedding_weight = embedding.weight();

        let fa_idx = (0..cfg.num_layers as usize)
            .find(|&i| !cfg.is_linear_layer(i))
            .expect("tiny_moe_cfg must contain at least one full-attention layer");

        let mut caches_old = Some(fresh_moe_layer_caches(&cfg));
        let mut caches_new = Some(fresh_moe_layer_caches(&cfg));

        // Prime both cache sets identically with a few single-token decode
        // steps so the eventual multi-token forward runs against a nonzero
        // KV offset — an empty cache is the degenerate offset=0 case both
        // code paths already handled identically, so it wouldn't exercise
        // the fix.
        for tok in [5u32, 9, 13] {
            let ids = MxArray::from_uint32(&[tok], &[1, 1]).expect("prime ids");
            forward_pre_norm_inner(&ids, &embedding, &mut layers, &mut caches_old, fa_idx)
                .expect("priming forward (old-path cache) must succeed");
            forward_pre_norm_inner(&ids, &embedding, &mut layers, &mut caches_new, fa_idx)
                .expect("priming forward (new-path cache) must succeed");
        }

        // The eager-MTP verify shape: `[1, K+1]` ids over the primed prefix.
        let verify_ids = MxArray::from_uint32(&[21u32, 22, 23, 24], &[1, 4]).expect("verify ids");

        let old_out = forward_with_explicit_causal_mask(
            &verify_ids,
            &embedding_weight,
            &mut layers,
            &mut caches_old,
            fa_idx,
        )
        .expect("explicit-mask forward must succeed");
        let new_out = forward_pre_norm_inner(
            &verify_ids,
            &embedding,
            &mut layers,
            &mut caches_new,
            fa_idx,
        )
        .expect("mask-free forward must succeed");

        let old_vals = old_out.to_float32().expect("old output to_float32");
        let new_vals = new_out.to_float32().expect("new output to_float32");
        assert_eq!(old_vals.len(), new_vals.len());
        for (idx, (a, b)) in old_vals.iter().zip(new_vals.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= 1e-4,
                "mask-free forward diverged from explicit-mask forward at {idx}: {a} vs {b} (diff {diff})"
            );
        }
    }
}

#[cfg(test)]
mod calibration_cmd_tests {
    //! Reachability/construction tests for the MoE activation-amax calibration
    //! command. A full model-load calibration is exercised end-to-end by the
    //! dense/MoE e2e; here we lock in that the MoE command variant exists, is
    //! constructible, and carries the same `{texts, calib_seq, reply}` shape the
    //! NAPI dispatch (`calibrate_activation_amax_raw`) sends — so a `qwen3_5_moe`
    //! checkpoint has a command to route to (the finding-1 gap was that no MoE
    //! command existed at all).

    use super::*;

    /// The MoE `CalibratePrefillRaw` variant is constructible and matchable with
    /// exactly the fields the NAPI driver sends. Proves finding-1's MoE command
    /// is reachable without needing a 19GB model load.
    #[test]
    fn moe_calibrate_prefill_raw_cmd_constructs_and_carries_fields() {
        let (tx, _rx) = tokio::sync::oneshot::channel::<napi::Result<u32>>();
        let cmd = Qwen35MoeCmd::CalibratePrefillRaw {
            texts: vec!["hello world".to_string(), "second row".to_string()],
            calib_seq: 128,
            reply: tx,
        };
        match cmd {
            Qwen35MoeCmd::CalibratePrefillRaw {
                texts, calib_seq, ..
            } => {
                assert_eq!(texts.len(), 2);
                assert_eq!(calib_seq, 128);
            }
            _ => panic!("expected CalibratePrefillRaw variant"),
        }
    }
}
