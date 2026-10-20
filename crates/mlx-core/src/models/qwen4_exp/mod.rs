//! Qwen3.8-Flash-Next with SSD-streamed experts and hashed PLE rows.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

mod auxiliary;
mod config;
mod decoder;
mod gguf;
mod mapped_pack;
mod math;
mod media;
mod memory;
mod mtp;
mod packed_cache;
mod paged;
mod prefetch;
mod route_plan;
mod row_fetch;
mod tripwire;
mod weights;

const MAX_PREFILL_CHUNK: usize = 1024;

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
    /// Scheduled turns admitted and not yet finished; the first to open
    /// after the set emptied starts a tripwire window, every open adds its
    /// turn to the one in progress.
    open_turns: std::collections::HashSet<u32>,
    /// A whole turn of the exclusive lane is open (its cancel flag is
    /// installed); it is one window of its own.
    exclusive_turn: bool,
}

impl Inner {
    /// A turn opens. A trip whose mid-turn switch failed moves the process to
    /// the slot path here, before any pass of the turn; the first turn of a
    /// window also starts the window's SSD count and slow streak. The prefill
    /// window follows the plan in force.
    fn open_turn(&mut self, first: bool) {
        let applied = if first {
            self.decoder.weights.begin_turn()
        } else {
            self.decoder.weights.apply_fallback()
        };
        match applied {
            Ok(Some(_)) => self.decoder.residency_changed(),
            Ok(None) => self.decoder.refresh_prefill_window(),
            Err(error) => {
                tracing::warn!("qwen4 residency: slot fallback not applied: {error}");
                self.decoder.refresh_prefill_window();
            }
        }
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
    /// Mapped experts are file-backed: a wired limit does not pin them and
    /// only delays their unwiring after the turn (3.2 s vs 1.2 s measured).
    fn wired_limit_bytes(&self) -> Option<usize> {
        if self.decoder.weights.plan.mapped {
            None
        } else {
            Some(usize::MAX)
        }
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
        if tokens.len().saturating_add(self.decoder.history.len())
            > self.decoder.config.effective_context_limit()
        {
            return Err(Error::from_reason(
                "qwen4_exp prompt exceeds the SSD runtime context budget",
            ));
        }
        let _context = StreamContext::new(stream);
        let mut logits = None;
        // Read before every pass: a pass that leaves the mapped path shrinks
        // the window for the passes after it.
        let mut start = 0;
        while start < tokens.len() {
            let window = self.decoder.prefill_slice_size();
            let end = (start + window).min(tokens.len());
            let next_end = (end + window).min(tokens.len());
            logits = Some(self.decoder.prefill_chunk_ahead(
                &tokens[start..end],
                &tokens[end..next_end],
                None,
                end == tokens.len(),
            )?);
            start = end;
        }
        logits
            .ok_or_else(|| Error::from_reason("Empty qwen4_exp prefill"))?
            .squeeze(Some(&[1]))
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
pub struct Qwen4ExpLoadOptions {
    /// Original matching HF checkpoint containing MTP/vision omitted by GGUF.
    pub auxiliary_model_path: Option<String>,
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
    /// and the MTP expert banks built on the first MTP owner.
    pub resident_bank_bytes: f64,
    /// Shared slot capacity for the routed layers at bootstrap (the expert
    /// count in mapped mode).
    pub partial_expert_slots: u32,
    /// Routed expert tensors the slot path reads by range from the mapped
    /// pack; 0 means every expert read is imported from the checkpoint.
    pub mapped_pack_files: u32,
    /// Every routed expert is addressed in place through the mapped pack;
    /// the weight budget then covers dense banks and rows only.
    pub mapped: bool,
    /// Pack bytes backing live arrays in mapped mode (clean page cache the
    /// Metal driver wires while a command buffer binds them).
    pub mapped_bytes: f64,
    /// Bytes of mapped banks the host could hold beside what stays anonymous
    /// when the residency was chosen: physical RAM minus a margin, the live
    /// non-reclaimable memory below and the anonymous need. The pack maps
    /// when this covers every bank.
    pub mapped_bank_budget_bytes: f64,
    /// Wired, compressed and other processes' anonymous memory when the
    /// residency was chosen.
    pub live_non_reclaimable_bytes: f64,
    /// Why the slot path serves the experts although the pack exists: the
    /// load gate, or a mapped turn that read over twice the pack from SSD or
    /// ran eight consecutive passes over ten times their median (over the
    /// fastest larger pass while they have no median of their own).
    pub fallback_reason: Option<String>,
    pub fallback_read_bytes: Option<f64>,
    pub fallback_slow_steps: Option<u32>,
    /// Tokens the request held when it left the mapped path between two of
    /// its passes; absent for the load gate.
    pub fallback_token_index: Option<u32>,
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
}

#[napi(object)]
pub struct Qwen4ExpContextLimits {
    pub effective_window_tokens: u32,
    pub paged_block_capacity: u32,
    pub paged_block_size: u32,
    pub trained_window_tokens: u32,
}

/// Qwen3.8-Flash-Next with bounded SSD weights, native MTP, image input and
/// paged scheduling. Inference and cache mutations run on one owned thread.
#[napi]
pub struct Qwen4ExpModel {
    thread: ModelThread<ModelCommand>,
    config: config::Config,
    assets_path: String,
    has_mtp: bool,
    paged_capacity: u32,
    residency: memory::Plan,
    live: Arc<tripwire::Live>,
    lookup: Arc<weights::LookupStats>,
    supports_images: bool,
}
#[napi]
impl Qwen4ExpModel {
    #[napi]
    pub async fn load(model_path: String, options: Option<Qwen4ExpLoadOptions>) -> Result<Self> {
        // `gguf::assets` leases the returned sidecar for the model's
        // lifetime; when init fails, the caller must release that reference
        // or the dir is pinned against the cache cap forever.
        let leased_assets = std::sync::Arc::new(std::sync::Mutex::new(None::<PathBuf>));
        let leased_assets_init = std::sync::Arc::clone(&leased_assets);
        let (thread, ready) = ModelThread::spawn_with_scheduler(
            move || {
                let path = PathBuf::from(model_path);
                let mut weights = weights::Store::open(&path)?;
                let (mut config, assets) = if weights.gguf {
                    let c = gguf::config(&weights)?;
                    let a = gguf::assets(&path, &weights, &c)?;
                    *leased_assets_init
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner()) = Some(a.clone());
                    (c, a)
                } else {
                    let raw = std::fs::read(path.join("config.json"))
                        .map_err(|e| Error::from_reason(e.to_string()))?;
                    let raw = serde_json::from_slice(&raw)
                        .map_err(|e| Error::from_reason(e.to_string()))?;
                    (config::Config::parse(&raw)?, path.clone())
                };
                let raw = if let Some(aux) = options.and_then(|o| o.auxiliary_model_path) {
                    auxiliary::attach(&mut weights, &mut config, Path::new(&aux))?
                } else if !weights.gguf {
                    serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?
                } else {
                    serde_json::Value::Null
                };
                let has_mtp = auxiliary::validate_mtp(&weights, &config)?;
                let vision = media::Vision::metadata(&raw, &weights, &config)?;
                let supports_images = vision.available();
                let tokenizer = Arc::new(Qwen3Tokenizer::load_from_file_sync(
                    assets
                        .join("tokenizer.json")
                        .to_str()
                        .ok_or_else(|| Error::from_reason("Non-UTF8 model path"))?,
                )?);
                let defaults =
                    crate::engine::persistence::parse_generation_defaults(Path::new(&assets));
                let mut decoder = decoder::Decoder::new(config.clone(), weights)?;
                let adapter = paged::create(&config)?;
                let paged_capacity = adapter.block_capacity();
                let pool_bytes = adapter.bytes_per_block().map_err(Error::from_reason)?
                    * u64::from(paged_capacity);
                decoder.paged = Some(adapter);
                let pool_guard = Some(crate::cache_limit::coordinator().register_pool(pool_bytes));
                decoder.register_cache_limit();
                decoder.weights.prepare_mapped_pack()?;
                decoder.weights.select_mapped_residency(pool_bytes)?;
                decoder.weights.prepare_hot()?;
                decoder.weights.start_pack_stream();
                decoder.weights.publish_live();
                // Register the final reconciled budget, rather than retaining
                // the pre-pool estimate in the process-wide allocator policy.
                decoder.residency_changed();
                let residency = decoder.weights.plan.clone();
                let live = Arc::clone(&decoder.weights.live);
                let lookup = Arc::clone(&decoder.weights.lookup);
                let assets = assets.to_string_lossy().into_owned();
                let inner = Inner {
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
                    open_turns: std::collections::HashSet::new(),
                    exclusive_turn: false,
                };
                Ok((
                    crate::engine::hybrid_scheduler::HybridSchedulerState::new(inner)?,
                    (
                        config,
                        assets,
                        has_mtp,
                        paged_capacity,
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
            paged_capacity,
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
            paged_capacity,
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
            mapped_pack_files: self.residency.mapped_pack_files as u32,
            mapped: live.mapped.load(Ordering::Acquire),
            mapped_bytes: live.mapped_bytes.load(Ordering::Acquire) as f64,
            mapped_bank_budget_bytes: live.mapped_bank_budget.load(Ordering::Acquire) as f64,
            live_non_reclaimable_bytes: live.live_non_reclaimable.load(Ordering::Acquire) as f64,
            fallback_reason: fallback.as_ref().map(ToString::to_string),
            fallback_read_bytes: fallback
                .as_ref()
                .and_then(tripwire::Fallback::read_bytes)
                .map(|n| n as f64),
            fallback_slow_steps: fallback.as_ref().and_then(tripwire::Fallback::slow_steps),
            fallback_token_index: Some(live.fallback_token_index.load(Ordering::Acquire))
                .filter(|&index| index != tripwire::NO_TOKEN_INDEX)
                .map(|index| u32::try_from(index).unwrap_or(u32::MAX)),
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
        Qwen4ExpContextLimits {
            effective_window_tokens: self.config.effective_context_limit() as u32,
            trained_window_tokens: self.config.max_position_embeddings as u32,
            paged_block_capacity: self.paged_capacity,
            paged_block_size: 32,
        }
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
