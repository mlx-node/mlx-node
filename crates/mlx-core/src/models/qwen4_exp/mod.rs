//! Qwen3.8-Flash-Next with SSD-streamed experts and hashed PLE rows.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

mod auxiliary;
mod config;
mod decoder;
mod gguf;
mod live;
mod load_plan;
mod math;
mod media;
mod memory;
mod mtp;
mod packed_cache;
mod paged;
mod prefix_state;
mod route_plan;
#[cfg(test)]
mod route_replay;
mod row_fetch;
mod weights;

const MAX_PREFILL_CHUNK: usize = 1024;

pub use load_plan::{Qwen4ExpContextEstimate, estimate_qwen4_exp_context_window};

#[cfg(test)]
pub(crate) use memory::{draft_state_bytes, flash_next_config, prefill_scratch, row_state_bytes};

/// MLX's own `MLX_ENABLE_TF32` (default on). The NAX kernels refuse TF32-off
/// processes in C++; the Rust callers check the same setting first so a
/// refused window is never entered tentatively.
fn tf32_enabled() -> bool {
    std::env::var_os("MLX_ENABLE_TF32").is_none_or(|value| value != "0")
}

use crate::array::MxArray;
use crate::engine::backend::{ChatBackend, DecodeStep, ResetScope, SaveStateArgs, TurnSetup};
use crate::engine::model_command::ModelCommand;
use crate::engine::paged_epilogue::FinalTokenPolicy;
use crate::engine::paged_stepper::{EvalPolicy, PagedStepModel};
use crate::engine::params::{ModelGenerationDefaults, ThinkingPolicy};
use crate::model_thread::ModelThread;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::ChatMessage;
use crate::tokenizer::Qwen3Tokenizer;
use crate::vision::qwen::prompt::IMAGE_TOKEN_ID;
use napi::bindgen_prelude::Uint32Array;
use napi::{Error, Result};
use napi_derive::napi;
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::AtomicBool};

/// Splash server/constraints.py `validate_tokenizer` (d8af1054): refuse a
/// tokenizer that does not encode the reasoning and tool-call tags as one
/// token each, or the turn end as one token. The finalize reads these tags in
/// decoded text, so each must also survive a skip-special decode.
fn check_tokenizer_tags(tokenizer: &Qwen3Tokenizer) -> Result<()> {
    for tag in ["<think>", "</think>", "<tool_call>", "</tool_call>"] {
        let ids = tokenizer.encode_sync(tag, Some(false))?;
        let kept = match ids.as_slice() {
            [id] => tokenizer.decode_sync(&[*id], true)? == tag,
            _ => false,
        };
        if !kept {
            return Err(Error::from_reason(format!(
                "Qwen4Exp tokenizer: {tag:?} must be one token that a skip-special decode keeps \
                 (it encodes to {ids:?}); without it the reasoning split and tool calls are lost"
            )));
        }
    }
    let ids = tokenizer.encode_sync("<|im_end|>", Some(false))?;
    if ids.len() != 1 {
        return Err(Error::from_reason(format!(
            "Qwen4Exp tokenizer: the stop text \"<|im_end|>\" must be one token (it encodes to {ids:?})"
        )));
    }
    Ok(())
}

struct Inner {
    decoder: decoder::Decoder,
    tokenizer: Arc<Qwen3Tokenizer>,
    defaults: ModelGenerationDefaults,
    saved_history: Vec<u32>,
    rows: std::collections::HashMap<u32, decoder::DecoderState>,
    active_seq: Option<u32>,
    mtp: mtp::State,
    has_mtp: bool,
    vision: media::Vision,
    media_prefill: Option<media::Prepared>,
    _pool_guard: Option<crate::cache_limit::PoolCacheLimitGuard>,
    /// The memory plan the load settled on: the pool, the window and the
    /// dynamic budget the scheduled state shares with the pool.
    memory_plan: crate::engine::memory_plan::MemoryPlan,
    /// `--max-context`: the context every turn may hold.
    max_context: Option<u32>,
    /// Scheduled turns admitted and not yet finished; the first to open
    /// after the set emptied starts a live window, every open adds its
    /// turn to the one in progress.
    open_turns: std::collections::HashSet<u32>,
    /// A whole turn of the exclusive lane is open (its cancel flag is
    /// installed); it is one window of its own.
    exclusive_turn: bool,
    wired: Option<Arc<crate::stream::WiredResidency>>,
    /// Stored states for warm prefix reuse and the running turns' points.
    prefix_states: prefix_state::States,
}

impl Inner {
    /// The context every turn may hold: `--max-context`, else the served
    /// window.
    fn context_cap(&self) -> u32 {
        self.max_context
            .unwrap_or_else(|| self.decoder.config.served_context_tokens())
    }
    /// The first turn after an idle gap resumes the banks' direct fill.
    /// Admissions beside an open turn share it.
    fn open_turn(&mut self, first: bool) {
        if first && let Err(error) = self.decoder.weights.begin_turn() {
            tracing::warn!("qwen4 residency: direct fill not resumed: {error}");
        }
        self.decoder.refresh_prefill_window();
    }
}
struct Step<'a>(&'a mut Inner);
impl DecodeStep for Step<'_> {
    fn forward(&mut self, ids: &MxArray) -> Result<(MxArray, bool)> {
        let token = ids.item_at_int32(0)? as u32;
        self.forward_with_token(ids, token)
    }
    fn forward_with_token(&mut self, _ids: &MxArray, token: u32) -> Result<(MxArray, bool)> {
        Ok((self.0.decoder.step(token)?, true))
    }
}
/// Paged lane: [`crate::engine::paged_stepper::PagedStepper`] wraps the
/// same `Step` state (`PagedBackend::PagedDecode` aliases
/// `PagedStepper<Step>`); the FLAT lane keeps the `DecodeStep` impl
/// above. `decoder.step` returns the raw `[1, 1, vocab]` logits the
/// stepper squeezes itself.
impl PagedStepModel for Step<'_> {
    /// `AsyncTokenAndForcedLogits` — the `DecodeStep::eval_step` default
    /// the bare paged impl inherited.
    const EVAL: EvalPolicy = EvalPolicy::AsyncTokenAndForcedLogits;
    /// `AlwaysDrop` — DO NOT re-forward the final token: `decoder.step`
    /// advances the GDN/conv/indexer/PLE recurrent state and
    /// `decoder.history`, so a replayed step would desync the frontier
    /// `save_paged_history` publishes (the exact forwarded-token set,
    /// never an unconsumed sample).
    const FINAL_TOKEN_POLICY: FinalTokenPolicy = FinalTokenPolicy::AlwaysDrop;

    fn paged_step(&mut self, token_id: u32) -> Result<MxArray> {
        self.0.decoder.step(token_id)
    }

    fn maintain_cache(&mut self, step: i32) {
        // The bare paged impl inherited the FLAT every-256-step
        // `clear_cache` default — preserve that cadence exactly (NOT the
        // paged helper's 1024-step cadence).
        if (step + 1) % 256 == 0 {
            crate::array::clear_cache();
        }
    }
}
impl ChatBackend for Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        Ok(Arc::clone(&self.tokenizer))
    }
    fn execution_plan(&self) -> crate::engine::plan::ExecutionPlan {
        use crate::engine::plan::*;
        ExecutionPlan {
            media: MediaPlan::with_backend_validation(
                if self.vision.available() {
                    MediaCapabilities::IMAGES
                } else {
                    MediaCapabilities::NONE
                },
                MediaCapabilities::IMAGES,
            ),
            paged_attention: self.decoder.paged.as_ref().map(|_| PagedAttentionPlan {
                supports_delta: true,
            }),
            speculative: self.has_mtp.then_some(SpeculativePlan {
                kind: SpeculativeKind::NativeMtp,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::NONE,
                supports_paged_attention: true,
                supports_streaming: true,
            }),
        }
    }
    fn run_paged_turn(
        &mut self,
        args: &mut crate::engine::backend::WholeTurnArgs<'_>,
    ) -> Result<crate::engine::backend::TurnOutput> {
        // Native MTP runs only on the scheduled lane; an ownerless MTP request
        // that lands here would otherwise decode AR under `enableMtp: true`.
        if matches!(
            args.plan.decoder,
            crate::engine::plan::DecoderPlan::Speculative(_)
        ) && args.params.cache_owner_id.is_empty()
        {
            return Err(Error::from_reason(
                if crate::engine::hybrid_scheduler::HybridSchedulerState::<Self>::force_serial() {
                    "qwen4_exp runs enableMtp only on its shared scheduler, which MLX_SERVE_FORCE_SERIAL turns off; set enableMtp: false or unset MLX_SERVE_FORCE_SERIAL"
                } else {
                    "qwen4_exp runs enableMtp only on its shared scheduler, and this ownerless request was routed to the exclusive lane; set enableMtp: false"
                },
            ));
        }
        crate::engine::paged_turn::run_paged_turn(self, args)
    }
    fn run_multimodal_turn(
        &mut self,
        args: &mut crate::engine::backend::WholeTurnArgs<'_>,
    ) -> Result<crate::engine::backend::TurnOutput> {
        self.run_media(args)
    }
    fn session_media(&self) -> crate::engine::plan::MediaCapabilities {
        if self.decoder.media_digests.is_empty() {
            crate::engine::plan::MediaCapabilities::NONE
        } else {
            crate::engine::plan::MediaCapabilities::IMAGES
        }
    }
    fn session_media_matches_payloads(&self, images: &[Vec<u8>], audio: &[Vec<u8>]) -> bool {
        audio.is_empty()
            && self.decoder.media_digests
                == crate::engine::cache::compute_image_cache_keys(images).1
    }
    fn template_history_comparison_tokens<'a>(
        &self,
        tokens: &'a [u32],
    ) -> std::borrow::Cow<'a, [u32]> {
        let positions: Vec<(u32, u64)> = if self.decoder.media_digests.is_empty() {
            Vec::new()
        } else {
            self.decoder
                .history
                .iter()
                .take(self.decoder.positions.len())
                .enumerate()
                .filter(|(_, t)| **t == IMAGE_TOKEN_ID as u32)
                .map(|(i, _)| (i as u32, 0))
                .collect()
        };
        crate::engine::cache::collapse_cached_media_placeholder_runs(
            tokens,
            IMAGE_TOKEN_ID as u32,
            &positions,
        )
    }
    fn family_name(&self) -> &'static str {
        "qwen4_exp"
    }
    fn policy(&self) -> ThinkingPolicy {
        ThinkingPolicy::TemplateHonoring
    }
    fn generation_defaults(&self) -> Option<&ModelGenerationDefaults> {
        Some(&self.defaults)
    }
    fn resolve_params(
        &self,
        config: &crate::engine::types::ChatConfig,
    ) -> crate::engine::params::ChatParams {
        let mut config = config.clone();
        crate::engine::params::apply_generation_defaults(&mut config, &self.defaults);
        let mut params = crate::engine::params::extract_chat_params(&config);
        params.mtp_depth = config.mtp_depth.map_or(2, |_| params.mtp_depth.min(3));
        // The shared measured verification policy models greedy acceptance.
        // Stochastic requests retain fixed-depth, distribution-correct MTP.
        if !crate::sampling::is_greedy_temperature(
            params
                .sampling_config
                .and_then(|c| c.temperature)
                .unwrap_or(1.0),
        ) {
            params.mtp_adaptive_depth = false;
        }
        params
    }
    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32> {
        Ok(tok.im_end_id().unwrap_or(self.decoder.config.eos_token_id))
    }
    fn extra_eos_ids(&self) -> Vec<u32> {
        let mut v = self.defaults.eos_token_ids.clone();
        v.push(self.decoder.config.eos_token_id);
        v
    }
    fn cached_token_history(&self) -> &[u32] {
        &self.saved_history
    }
    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        if matches!(scope, ResetScope::Command) {
            if let Some(adapter) = &mut self.decoder.paged {
                for seq in adapter.live_seq_ids() {
                    adapter
                        .release_request_for(seq)
                        .map_err(Error::from_reason)?;
                }
            }
            self.rows.clear();
            self.active_seq = None;
            self.mtp = mtp::State::default();
            self.prefix_states.clear();
            if let Some(adapter) = &mut self.decoder.paged {
                adapter
                    .release_request_and_purge_prefix_cache()
                    .map_err(Error::from_reason)?;
            }
        }
        self.decoder.reset();
        self.saved_history.clear();
        Ok(())
    }
    fn verify_cache_prefix(&self, tokens: &[u32], reuse: bool) -> usize {
        let h = &self.saved_history;
        if reuse
            && !h.is_empty()
            && self.decoder.history == *h
            && tokens.len() > h.len()
            && tokens.starts_with(h)
        {
            h.len()
        } else {
            0
        }
    }
    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        if args.reuse_cache {
            self.saved_history = self.decoder.history.clone();
        } else {
            self.decoder.reset();
            self.saved_history.clear();
        }
    }
    fn eval_caches(&self) -> Result<()> {
        Ok(())
    }
    /// The model's `WiredResidency` holds the lease for its life.
    fn wired_limit_bytes(&self) -> Option<usize> {
        None
    }
    /// The exclusive lane installs its flag once per whole turn and clears
    /// it after; the scheduler installs one per admission and prefill slice
    /// of turns it has already opened (`scheduled_turn_opened`), so those
    /// never start a window here.
    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        if flag.is_none() {
            self.exclusive_turn = false;
        } else if self.open_turns.is_empty() && !self.exclusive_turn {
            self.exclusive_turn = true;
            self.open_turn(true);
        }
        self.decoder.cancelled = flag;
    }
    fn prefill(&mut self, tokens: &[u32], stream: Stream) -> Result<MxArray> {
        let window = self.decoder.context_window;
        if tokens.len().saturating_add(self.decoder.history.len()) > window {
            return Err(Error::from_reason(format!(
                "qwen4_exp prompt exceeds the {window}-token window of the memory plan"
            )));
        }
        let _context = StreamContext::new(stream);
        self.decoder.prefill_prompt(tokens)?.squeeze(Some(&[1]))
    }
    type Decode<'a>
        = Step<'a>
    where
        Self: 'a;
    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Step<'_>> {
        Ok(Step(self))
    }
}

#[napi(object)]
#[derive(Default)]
pub struct Qwen4ExpLoadOptions {
    /// Original matching HF checkpoint containing MTP/vision omitted by GGUF.
    pub auxiliary_model_path: Option<String>,
    /// The Metal memory budget (`--max-memory`) the memory plan sizes the
    /// stores, the pool and the windows within.
    pub max_memory_bytes: Option<i64>,
    /// Context token limit (`--max-context`), 1 to 1,010,000: the KV
    /// allowance the stores leave room for and the most any turn holds; the
    /// load fails when its plan holds fewer. Past the native window the
    /// target's RoPE takes YaRN.
    pub max_context_tokens: Option<i64>,
    /// Element format of the attention K/V pages: `'int8'` (per-token
    /// symmetric int8 rows with one fp32 scale each) or `'bf16'`. Unset:
    /// int8 where the int8 page kernels serve the head dimension (128 or
    /// 256), bf16 elsewhere; an explicit `'int8'` they cannot serve fails
    /// the load.
    #[napi(ts_type = "'int8' | 'bf16' | undefined")]
    pub kv_format: Option<String>,
}

/// Load-time residency decision (the budget is admitted before payload
/// reads) plus the live pinned-bank total.
#[napi(object)]
pub struct Qwen4ExpResidencyInfo {
    pub weight_budget_bytes: f64,
    pub hot_weight_bytes: f64,
    pub prefill_chunk_tokens: u32,
    /// Free plus reclaimable file-backed memory observed at bootstrap.
    pub available_memory_bytes: Option<f64>,
    pub physical_memory_bytes: Option<f64>,
    /// Assembled matrices actually pinned right now: partial projection banks
    /// and the MTP expert banks built at load.
    pub resident_bank_bytes: f64,
    /// Shared partial-slot capacity, or the expert count at full direct
    /// identity residency.
    pub partial_expert_slots: u32,
    /// Every routed expert has an anonymous bank filled directly from GGUF,
    /// with slot id equal to expert id.
    pub direct: bool,
    /// Anonymous bytes allocated to direct expert banks.
    pub direct_bytes: f64,
    /// Routed layers in the direct schedule and layers whose fill completed.
    pub direct_layers: u32,
    pub direct_filled_layers: u32,
    /// A direct layered fill is still running.
    pub direct_fill_in_progress: bool,
    /// Bytes of direct banks the host could hold beside what stays anonymous
    /// when residency was chosen: physical RAM minus a margin, the live
    /// non-reclaimable memory below and the anonymous need.
    pub bank_budget_bytes: f64,
    /// Wired, compressed and other processes' anonymous memory when the
    /// residency was chosen.
    pub live_non_reclaimable_bytes: f64,
    /// Present when the full-bank load gate selected partial direct slots.
    pub fallback_reason: Option<String>,
    /// Token/PLE rows asked of the decoded-row LRU since load, and the ones
    /// it had to import.
    pub decoded_row_lookups: f64,
    pub decoded_row_misses: f64,
    /// Share of decoded-row lookups served without an import.
    pub decoded_row_hit_rate: f64,
    /// Imported rows whose bytes a reader thread had ready, and those the
    /// model thread read or waited for after requesting them.
    pub prefetched_rows: f64,
    pub late_rows: f64,
    /// Decoded rows the LRU bound evicted since load, and the bytes those
    /// evictions released (a window's shared buffer goes with its last row).
    pub decoded_row_evictions: f64,
    pub decoded_row_evicted_bytes: f64,
    /// Routed experts the slots or banks served without a read since load.
    pub slot_hits: f64,
    /// Routed experts a slot fill read from the checkpoint since load.
    pub slot_misses: f64,
    /// Bytes read from the checkpoint files since load (expert fills and row
    /// imports); the page cache may serve part of them.
    pub checkpoint_bytes_read: f64,
    /// Decode tokens and prefill windows replayed after a tentative device
    /// route missed the slots.
    pub route_replays: f64,
}

#[napi(object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4ExpContextLimits {
    pub effective_window_tokens: u32,
    pub paged_block_capacity: u32,
    pub paged_block_size: u32,
    pub trained_window_tokens: u32,
    /// The output budget fills the window: a prompt leaves room for one new
    /// token and the budget is clamped to `window - prompt`.
    pub rest_of_window_output_budget: bool,
    /// An omitted budget on `effective_window_tokens` is the rest of that
    /// window.
    pub omitted_budget_is_rest_of_window: bool,
    /// Format of the K/V every turn writes.
    #[napi(ts_type = "'int8' | 'bf16'")]
    pub kv_format: String,
}

impl Qwen4ExpContextLimits {
    /// The window of the memory plan the load settled on, for every turn.
    fn from_plan(
        config: &config::Config,
        plan: &crate::engine::memory_plan::MemoryPlan,
        cap: u32,
    ) -> Result<Self> {
        let pool = plan
            .pool
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen4Exp memory plan: the load planned no pool"))?;
        Ok(Self {
            effective_window_tokens: pool.lane.window.maximum().min(cap),
            paged_block_capacity: pool.max_blocks,
            paged_block_size: crate::engine::memory_plan::KV_PAGE_TOKENS as u32,
            trained_window_tokens: config.max_position_embeddings.min(u32::MAX as usize) as u32,
            rest_of_window_output_budget: true,
            omitted_budget_is_rest_of_window: true,
            kv_format: config.kv_format.as_str().into(),
        })
    }
}

/// Qwen3.8-Flash-Next with bounded SSD weights, native MTP, image input and
/// paged scheduling. Inference and cache mutations run on one owned thread.
#[napi]
pub struct Qwen4ExpModel {
    thread: ModelThread<ModelCommand>,
    config: config::Config,
    assets_path: String,
    has_mtp: bool,
    context_limits: Qwen4ExpContextLimits,
    residency: memory::Plan,
    live: Arc<live::Live>,
    lookup: Arc<weights::LookupStats>,
    supports_images: bool,
}
#[napi]
impl Qwen4ExpModel {
    /// Frees the weights, caches and state now: the model thread drops them
    /// and exits; every later call fails.
    #[napi]
    pub async fn dispose(&self) -> Result<()> {
        self.thread.dispose().await
    }

    #[napi]
    pub async fn load(model_path: String, options: Option<Qwen4ExpLoadOptions>) -> Result<Self> {
        const LABEL: &str = "Qwen4Exp";
        let options = options.unwrap_or_default();
        let max_memory =
            crate::models::qwen3_5::load_plan::max_memory_option(LABEL, options.max_memory_bytes)?;
        let max_context =
            crate::nn::rope_scaling::max_context_option(LABEL, options.max_context_tokens)?;
        // `gguf::assets` leases the returned sidecar for the model's
        // lifetime; when init fails, the caller must release that reference
        // or the dir is pinned against the cache cap forever.
        let leased_assets = std::sync::Arc::new(std::sync::Mutex::new(None::<PathBuf>));
        let leased_assets_init = std::sync::Arc::clone(&leased_assets);
        let (thread, ready) = ModelThread::spawn_with_scheduler(
            move || {
                let path = PathBuf::from(model_path);
                let mut weights = weights::Store::open(&path)?;
                let mut config = load_plan::checkpoint_config(&weights, &path)?;
                let assets = if weights.gguf {
                    let a = gguf::assets(&path, &weights, &config)?;
                    *leased_assets_init
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner()) = Some(a.clone());
                    a
                } else {
                    path.clone()
                };
                let raw = load_plan::attach_auxiliary(
                    &mut weights,
                    &mut config,
                    &path,
                    options.auxiliary_model_path.as_deref(),
                )?;
                load_plan::serve_context(LABEL, &mut config, max_context)?;
                config.kv_format =
                    load_plan::resolve_kv_format(LABEL, options.kv_format.as_deref(), &config)?;
                let has_mtp = auxiliary::validate_mtp(&weights, &config)?;
                let vision = media::Vision::metadata(&raw, &weights, &config)?;
                let supports_images = vision.available();
                let tokenizer = Arc::new(Qwen3Tokenizer::load_from_file_sync(
                    assets
                        .join("tokenizer.json")
                        .to_str()
                        .ok_or_else(|| Error::from_reason("Non-UTF8 model path"))?,
                )?);
                check_tokenizer_tags(&tokenizer)?;
                let turn_open = match tokenizer.encode_sync("<|im_start|>", Some(false))?[..] {
                    [id] => Some(id),
                    _ => None,
                };
                let defaults =
                    crate::engine::persistence::parse_generation_defaults(Path::new(&assets));
                let live = memory::live_memory().ok_or_else(|| {
                    Error::from_reason(format!(
                        "{LABEL} memory plan: the Metal working set is unreadable; refusing an unplanned load"
                    ))
                })?;
                let inventory = weights.load_inventory(&config, supports_images)?;
                let decision = load_plan::plan_store(
                    LABEL,
                    &config,
                    &inventory,
                    live,
                    max_memory,
                    max_context,
                    has_mtp,
                )?;
                weights.set_store_plan(decision.store, inventory.store_fixed_bytes);
                let mut decoder = decoder::Decoder::new(config.clone(), weights)?;
                decoder.register_cache_limit();
                decoder.weights.select_residency(
                    inventory
                        .store_fixed_bytes
                        .saturating_add(decision.store.row_cache_bytes)
                        .saturating_add(decision.allowance),
                )?;
                decoder.weights.prepare_hot()?;
                decoder.weights.finish_load()?;
                let (pool_guard, memory_plan, pool_bytes) = {
                    let _growth_guard = crate::cache_limit::pool_growth_lock()
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let engine = |held_back: u64| {
                        let live = memory::live_memory().ok_or_else(|| {
                            Error::from_reason(format!(
                                "{LABEL} memory plan: the Metal working set is unreadable"
                            ))
                        })?;
                        load_plan::plan_engine(
                            LABEL,
                            &config,
                            live,
                            held_back.saturating_add(inventory.vision_bytes),
                            max_memory,
                            has_mtp,
                        )
                    };
                    let held_back = decoder.weights.held_back_bytes();
                    let weights = &mut decoder.weights;
                    let plan =
                        load_plan::settle(&engine, held_back, decision.allowance, &mut |bytes| {
                            weights.lower_store_cap(bytes)
                        })?;
                    load_plan::check_context(LABEL, &plan, max_context)?;
                    let pool_plan = plan.pool.ok_or_else(|| {
                        Error::from_reason(format!("{LABEL} memory plan: the load planned no pool"))
                    })?;
                    let adapter = paged::create(&config, &pool_plan, plan.ceiling_bytes())?;
                    let pool_bytes = adapter.pool_allocated_bytes().map_err(Error::from_reason)?;
                    tracing::info!(
                        "{LABEL} pool: kv_format={}, {} B blocks",
                        config.kv_format.as_str(),
                        adapter.bytes_per_block().map_err(Error::from_reason)?
                    );
                    decoder.paged = Some(adapter);
                    let guard = crate::cache_limit::coordinator().register_pool(pool_bytes);
                    (guard, plan, pool_bytes)
                };
                if let Some(adapter) = decoder.paged.as_mut() {
                    let pool_id = pool_guard.id();
                    adapter.set_pool_growth_notifier(Some(Arc::new(move |bytes| {
                        crate::cache_limit::coordinator().update_pool(pool_id, bytes);
                    })));
                }
                tracing::info!("{LABEL} memory plan:\n{}", memory_plan.describe());
                crate::models::qwen3_5::load_plan::audit_qwen35_load(
                    LABEL,
                    &memory_plan,
                    pool_bytes,
                )?;
                let context_limits = Qwen4ExpContextLimits::from_plan(
                    &config,
                    &memory_plan,
                    max_context.unwrap_or_else(|| config.served_context_tokens()),
                )?;
                decoder
                    .weights
                    .set_active_ceiling(memory_plan.ceiling_bytes());
                decoder.weights.publish_live();
                // Register the final budget and the prefill window of the
                // planned context.
                decoder.adopt_plan(
                    context_limits.effective_window_tokens as usize,
                    decision.allowance_tokens,
                );
                let pool_guard = Some(pool_guard);
                let residency = decoder.weights.plan.clone();
                let live = Arc::clone(&decoder.weights.live);
                let lookup = Arc::clone(&decoder.weights.lookup);
                let assets = assets.to_string_lossy().into_owned();
                let mut inner = Inner {
                    decoder,
                    tokenizer,
                    defaults,
                    saved_history: Vec::new(),
                    rows: std::collections::HashMap::new(),
                    active_seq: None,
                    mtp: mtp::State::default(),
                    has_mtp,
                    vision,
                    media_prefill: None,
                    _pool_guard: pool_guard,
                    memory_plan,
                    max_context,
                    open_turns: std::collections::HashSet::new(),
                    exclusive_turn: false,
                    wired: None,
                    prefix_states: prefix_state::States::new(turn_open),
                };
                if let Some(adapter) = inner.decoder.paged.as_ref() {
                    adapter
                        .layer_kv_pool()
                        .hold_resident()
                        .map_err(Error::from_reason)?;
                }
                let wired_bytes = inner.wired_bytes();
                inner.wired = Some(crate::stream::WiredResidency::new(move || wired_bytes));
                Ok((
                    crate::engine::hybrid_scheduler::HybridSchedulerState::new(inner)?,
                    (
                        config,
                        assets,
                        has_mtp,
                        context_limits,
                        residency,
                        live,
                        lookup,
                        supports_images,
                    ),
                ))
            },
            |state, receiver| state.drive(receiver),
        );
        let (
            config,
            assets_path,
            has_mtp,
            context_limits,
            residency,
            live,
            lookup,
            supports_images,
        ) = match ready.await {
            Ok(Ok(init)) => init,
            Ok(Err(error)) => {
                if let Some(dir) = leased_assets
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .take()
                {
                    crate::utils::gguf::native_gguf_unlease_in_use(&dir);
                }
                return Err(error);
            }
            Err(e) => {
                if let Some(dir) = leased_assets
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .take()
                {
                    crate::utils::gguf::native_gguf_unlease_in_use(&dir);
                }
                return Err(Error::from_reason(e.to_string()));
            }
        };
        Ok(Self {
            thread,
            config,
            assets_path,
            has_mtp,
            context_limits,
            residency,
            live,
            lookup,
            supports_images,
        })
    }
    /// Whether the validated checkpoint has a supported image tower.
    #[napi]
    pub fn supports_images(&self) -> bool {
        self.supports_images
    }
    /// Plan the expanded prompt using the same processor and limits as prefill.
    #[napi]
    pub async fn expanded_prompt_token_count(
        &self,
        prompt_tokens: Uint32Array,
        messages: Vec<ChatMessage>,
    ) -> Result<u32> {
        crate::vision::qwen::prompt::expanded_prompt_token_count(
            self.supports_images
                .then(|| Arc::new(media::image_processor())),
            2,
            prompt_tokens,
            messages,
            Some(media::IMAGE_LIMITS),
        )
        .await
    }
    #[napi]
    pub fn residency_info(&self) -> Qwen4ExpResidencyInfo {
        use std::sync::atomic::Ordering;
        let (lookups, misses) = (self.lookup.lookups(), self.lookup.misses());
        let live = &self.live;
        let fallback = live.fallback();
        Qwen4ExpResidencyInfo {
            weight_budget_bytes: live.weight_budget.load(Ordering::Acquire) as f64,
            hot_weight_bytes: self.residency.hot_bytes as f64,
            prefill_chunk_tokens: live.prefill_chunk_tokens.load(Ordering::Acquire) as u32,
            available_memory_bytes: self.residency.available_bytes.map(|n| n as f64),
            physical_memory_bytes: self.residency.physical_bytes.map(|n| n as f64),
            resident_bank_bytes: self
                .residency
                .resident_bank_bytes
                .load(std::sync::atomic::Ordering::Relaxed) as f64,
            partial_expert_slots: live.partial_expert_slots.load(Ordering::Acquire) as u32,
            direct: live.direct.load(Ordering::Acquire),
            direct_bytes: live.direct_bytes.load(Ordering::Acquire) as f64,
            direct_layers: live.direct_layers.load(Ordering::Acquire) as u32,
            direct_filled_layers: live.direct_filled_layers.load(Ordering::Acquire) as u32,
            direct_fill_in_progress: live.direct_fill_in_progress.load(Ordering::Acquire),
            bank_budget_bytes: live.bank_budget.load(Ordering::Acquire) as f64,
            live_non_reclaimable_bytes: live.live_non_reclaimable.load(Ordering::Acquire) as f64,
            fallback_reason: fallback.as_ref().map(ToString::to_string),
            decoded_row_lookups: lookups as f64,
            decoded_row_misses: misses as f64,
            decoded_row_hit_rate: if lookups == 0 {
                0.0
            } else {
                1.0 - misses as f64 / lookups as f64
            },
            prefetched_rows: self.lookup.prefetched() as f64,
            late_rows: self.lookup.late() as f64,
            decoded_row_evictions: self.lookup.evicted_rows() as f64,
            decoded_row_evicted_bytes: self.lookup.evicted_row_bytes() as f64,
            slot_hits: live.slot_hits.load(Ordering::Relaxed) as f64,
            slot_misses: live.slot_misses.load(Ordering::Relaxed) as f64,
            checkpoint_bytes_read: live.bytes_read.load(Ordering::Relaxed) as f64,
            route_replays: live.route_replays.load(Ordering::Relaxed) as f64,
        }
    }
    #[napi]
    pub fn model_assets_path(&self) -> String {
        self.assets_path.clone()
    }
    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.has_mtp
    }
    /// SSD-backed draft and verification work is opt-in; head availability
    /// alone does not establish a latency benefit for the current workload.
    #[napi]
    pub fn mtp_auto_enabled(&self) -> bool {
        false
    }
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        true
    }
    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        scheduler_capacity()
    }
    #[napi]
    pub fn context_limits(&self) -> Qwen4ExpContextLimits {
        self.context_limits.clone()
    }
    #[napi]
    pub fn get_config(&self) -> Result<serde_json::Value> {
        serde_json::to_value(&self.config).map_err(|e| Error::from_reason(e.to_string()))
    }
}
crate::models::chat_napi::chat_napi_surface! {
    class: Qwen4ExpModel,
    thread_cmd: crate::engine::model_command::ModelCommand,
    thread: direct,
    image_guard: none,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}

fn scheduler_capacity() -> u32 {
    if crate::engine::hybrid_scheduler::HybridSchedulerState::<Inner>::force_serial() {
        1
    } else {
        crate::engine::hybrid_scheduler::scheduler_max_num_seqs_for(4) as u32
    }
}

#[cfg(test)]
mod tests;
