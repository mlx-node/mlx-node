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
#[cfg(unix)]
mod packed_cache;
mod paged;
mod prefix_disk;
mod prefix_state;
mod route_plan;
#[cfg(test)]
mod route_replay;
mod row_fetch;
mod weights;

const MAX_PREFILL_CHUNK: usize = 1024;
/// Sequences the continuous-batching scheduler runs at once.
const LANES: usize = 4;

pub use load_plan::{Qwen4ExpContextEstimate, estimate_qwen4_exp_context_window};

#[cfg(test)]
pub(crate) use memory::{draft_state_bytes, flash_next_config, prefill_scratch, row_state_bytes};

/// MLX's own `MLX_ENABLE_TF32` (default on). The NAX kernels refuse TF32-off
/// processes in C++; the Rust callers check the same setting first so a
/// refused window is never entered tentatively.
fn tf32_enabled() -> bool {
    std::env::var_os("MLX_ENABLE_TF32").is_none_or(|value| value != "0")
}

#[cfg(test)]
std::thread_local! {
    static CUDA_QWEN4_FAST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Temporarily override [`cuda_qwen4_fast_enabled`] on the current test thread
/// for deterministic in-process A/B parity tests (`MLX_QWEN4_CUDA_FAST=1` vs `0`).
#[cfg(test)]
pub(crate) fn with_cuda_qwen4_fast<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    struct Guard(Option<bool>);
    impl Drop for Guard {
        fn drop(&mut self) {
            CUDA_QWEN4_FAST_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let prev = CUDA_QWEN4_FAST_OVERRIDE.with(|c| c.replace(Some(enabled)));
    let _guard = Guard(prev);
    f()
}

/// Whether Wave A fused CUDA kernels for `qwen4_exp` are enabled.
///
/// Default on whenever CUDA GPU execution is active; set `MLX_QWEN4_CUDA_FAST=0`
/// to fall back to the Phase 0 generic MLX path for A/B parity testing.
#[inline]
pub(super) fn cuda_qwen4_fast_enabled() -> bool {
    if !crate::accelerator::cuda_ops_available() {
        return false;
    }
    #[cfg(test)]
    if let Some(val) = CUDA_QWEN4_FAST_OVERRIDE.with(|c| c.get()) {
        return val;
    }
    #[cfg(test)]
    {
        std::env::var_os("MLX_QWEN4_CUDA_FAST").is_none_or(|v| v != "0")
    }
    #[cfg(not(test))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("MLX_QWEN4_CUDA_FAST").is_none_or(|v| v != "0"))
    }
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
    /// The stored states' disk tier; `None` when it is off.
    disk: Option<prefix_disk::Disk>,
    telemetry: Arc<prefix_disk::Telemetry>,
    /// The DFlash2 draft of the load's `draftModelPath`. Its cycle runs
    /// only in a scheduled turn that `draft_turn` admits at the turn's
    /// start.
    draft: Option<decoder::DraftSlot>,
}

impl Drop for Inner {
    /// A dispose writes the waiting states and drains the copies within one
    /// bound before the tier closes.
    fn drop(&mut self) {
        if let (Some(disk), Some(adapter)) = (self.disk.as_mut(), self.decoder.paged.as_mut()) {
            disk.dispose(&mut self.prefix_states.store, adapter);
        }
    }
}

/// The disk tier a load of `options` plans, under the persistence switch's
/// environment value `env`: its options and whether it is on. On unless
/// `--max-cache-disk 0`; persistent with the switch on and a GGUF identity
/// the stored objects are bound to, else temporary. The load, the tier's
/// open and the context estimate all decide through this one function.
pub(super) fn disk_plan(
    label: &str,
    options: &Qwen4ExpLoadOptions,
    env: Option<&str>,
) -> Result<(crate::disk_tier::DiskTierOptions, bool)> {
    let tier = crate::disk_tier::DiskTierOptions::from_load(
        options.cache_disk.as_deref(),
        crate::disk_tier::resolve_persist(options.persist_paged_cache, env),
        options.cache_dir.clone(),
    )
    .map_err(|message| Error::from_reason(format!("{label} load: {message}")))?;
    let on = tier.cache_disk != crate::disk_tier::CacheDisk::Off;
    Ok((tier, on))
}

impl Inner {
    /// Opens the stored states' tier of `options` and hands its store to the
    /// adapter for the states' KV blocks: persistent, a namespace directory
    /// of the checkpoint `identity` and the math the fingerprint binds, where
    /// the switch asks for it and it opens; else temporary. Fails open: the
    /// states stay in memory.
    fn open_prefix_disk(
        &mut self,
        options: &crate::disk_tier::DiskTierOptions,
        identity: Option<&str>,
        auxiliary: Option<&Path>,
    ) {
        let Some(adapter) = self.decoder.paged.as_ref() else {
            return;
        };
        let window = self.decoder.prefill_slice_size();
        let scratch = self.decoder.host_prefill_scratch;
        let tier = crate::disk_tier::open_disk_tier(
            options,
            || {
                let fingerprint = prefix_disk::fingerprint(
                    identity?,
                    auxiliary,
                    &self.decoder.config,
                    adapter.layer_kv_pool(),
                    window,
                    &self.prefix_states.open,
                    scratch,
                    self.has_mtp,
                )?;
                Some(crate::disk_tier::PersistentIdentity {
                    namespace: fingerprint.to_hex().get(..32)?.to_string(),
                    state: fingerprint,
                    kv: fingerprint,
                })
            },
            memory::encoded_state_bytes(&self.decoder.config, self.has_mtp),
        );
        let Some(tier) = tier else {
            tracing::warn!("qwen4_exp: stored states stay in memory: the SSD cache did not open");
            return;
        };
        let shape = prefix_disk::LaneShape {
            span: memory::encoded_state_bytes(&self.decoder.config, self.has_mtp),
            layers: self.decoder.caches.len(),
            grid: window,
            has_mtp: self.has_mtp,
        };
        match prefix_disk::Disk::open(tier, Arc::clone(&self.telemetry), &shape) {
            Ok(mut disk) => {
                if let Some(adapter) = self.decoder.paged.as_mut() {
                    disk.attach(adapter);
                    disk.adopt(&mut self.prefix_states.store, &self.decoder.config);
                    self.telemetry.sync(&self.prefix_states);
                    self.disk = Some(disk);
                }
            }
            Err(error) => {
                tracing::warn!("qwen4_exp: stored states stay in memory: {error}");
            }
        }
    }
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
    /// The last open turn closed: a store the host gate refused at load
    /// moves up to whole when the host has room now.
    fn close_turns(&mut self) {
        if !self.open_turns.is_empty() || self.exclusive_turn {
            return;
        }
        if let Err(error) = self.promote_store(&mut memory::admit_direct) {
            tracing::warn!("qwen4 residency: the partial store stays: {error}");
        }
    }
    /// The bytes of this model the host holds that its ledgers count: the
    /// store's banks, slots and rows, the KV pool and the stored states.
    fn own_bytes(&self) -> u64 {
        let pool = self
            .decoder
            .paged
            .as_ref()
            .and_then(|adapter| adapter.pool_allocated_bytes().ok())
            .unwrap_or(0);
        self.decoder
            .weights
            .ledgered_bytes()
            .saturating_add(pool)
            .saturating_add(self.prefix_states.bytes())
            .saturating_add(self.draft_bytes_held())
    }
    /// [`decoder::Decoder::promote_whole_with`] through the load's host gate.
    /// A load with the MTP head keeps its store: the whole branch also sizes
    /// the draft banks, which no promotion re-plans.
    fn promote_store(&mut self, admit: &mut impl FnMut(u64) -> Result<()>) -> Result<bool> {
        if self.has_mtp {
            return Ok(false);
        }
        let own = self.own_bytes();
        self.decoder
            .promote_whole_with(own, &mut memory::direct_gate_beside, admit)
    }
}
struct Step<'a>(&'a mut Inner);
impl DecodeStep for Step<'_> {
    fn forward(&mut self, ids: &MxArray) -> Result<(MxArray, bool)> {
        let token = ids.item_at_int32(0)? as u32;
        self.forward_with_token(ids, token)
    }
    fn forward_with_token(&mut self, _ids: &MxArray, token: u32) -> Result<(MxArray, bool)> {
        let lazy = self.0.decoder.paged.is_none() && cuda_qwen4_fast_enabled();
        let prev = std::mem::replace(&mut self.0.decoder.cuda_ar_lazy_logits, lazy);
        let res = self.0.decoder.step(token);
        self.0.decoder.cuda_ar_lazy_logits = prev;
        Ok((res?, true))
    }
    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, budget_forced: bool) {
        if self.0.decoder.paged.is_none() && cuda_qwen4_fast_enabled() {
            self.0
                .decoder
                .flush_cuda_ar_step(next_token, logits, budget_forced);
        } else if budget_forced {
            MxArray::async_eval_arrays(&[next_token, logits]);
        } else {
            MxArray::async_eval_arrays(&[next_token]);
        }
    }
    fn end_decode(&mut self) -> Result<()> {
        if self.0.decoder.paged.is_none() && cuda_qwen4_fast_enabled() {
            self.0.decoder.sync_cuda_pending_states()?;
        }
        Ok(())
    }
    fn maintain_cache(&mut self, step: i32) {
        if (step + 1) % 256 == 0 {
            let should_clear = if crate::accelerator::cuda_ops_available() {
                crate::array::get_cache_memory()
                    > crate::accelerator::cuda_cache_clear_threshold_bytes()
            } else {
                true
            };
            if should_clear {
                crate::array::clear_cache();
            }
        }
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
        self.0.top_up_disk();
    }
}
impl ChatBackend for Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        Ok(Arc::clone(&self.tokenizer))
    }
    fn execution_plan(&self) -> crate::engine::plan::ExecutionPlan {
        use crate::engine::plan::*;
        let speculative = if self.draft.is_some() {
            Some(SpeculativePlan {
                kind: SpeculativeKind::DraftModel,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::NONE,
                supports_paged_attention: self.decoder.paged.is_some(),
                supports_streaming: true,
            })
        } else if self.has_mtp && self.decoder.paged.is_some() {
            Some(SpeculativePlan {
                kind: SpeculativeKind::NativeMtp,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::NONE,
                supports_paged_attention: true,
                supports_streaming: true,
            })
        } else {
            None
        };
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
            speculative,
        }
    }
    fn run_speculative_turn(
        &mut self,
        args: &mut crate::engine::backend::WholeTurnArgs<'_>,
    ) -> Result<crate::engine::backend::TurnOutput> {
        if self.decoder.paged.is_none()
            && matches!(
                args.plan.decoder,
                crate::engine::plan::DecoderPlan::Speculative(
                    crate::engine::plan::SpeculativeKind::DraftModel
                )
            )
        {
            return self.flat_dflash2_turn(args);
        }
        Err(Error::from_reason(
            "qwen4_exp flat speculative turn requires an attached DFlash2 draft model",
        ))
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
        let drafted = self.draft.is_some();
        let unset_depth = if drafted {
            decoder::DRAFT_VERIFY_ROWS - 1
        } else {
            2
        };
        params.mtp_depth = config
            .mtp_depth
            .map_or(unset_depth, |_| params.mtp_depth.min(3));
        // Stochastic requests retain fixed-depth, distribution-correct MTP.
        let greedy = crate::sampling::is_greedy_temperature(
            params
                .sampling_config
                .and_then(|c| c.temperature)
                .unwrap_or(1.0),
        );
        // A draft picks its rows itself unless the request pins them.
        params.mtp_adaptive_depth = if drafted {
            greedy
                && config
                    .mtp_adaptive_depth
                    .unwrap_or(config.mtp_depth.is_none())
        } else {
            greedy && params.mtp_adaptive_depth
        };
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
    fn context_window(&self) -> Option<u32> {
        self.decoder
            .paged
            .is_none()
            .then(|| self.decoder.context_window.min(u32::MAX as usize) as u32)
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
            self.prefix_states.reset(self.disk.is_some());
            if let Some(disk) = &mut self.disk {
                disk.reset();
            }
            self.telemetry.sync(&self.prefix_states);
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
            self.close_turns();
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
    /// A DFlash2 draft checkpoint directory (`DFlash2DraftModel`) for this
    /// target. The load fails when the draft does not serve the model. A
    /// turn that asks for speculation (`enableMtp`) runs the draft's cycle
    /// when it starts alone on a whole store whose banks are filled.
    pub draft_model_path: Option<String>,
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
    /// SSD cache for the stored states (`--max-cache-disk`): `'auto'` (the
    /// default: up to 32 GiB, a quarter of the volume's room), a byte count,
    /// or `'0'`, which turns the SSD cache off.
    pub cache_disk: Option<String>,
    /// The paged KV disk cache switch every family shares (`--no-persist-cache`):
    /// keep the stored states on SSD across restarts in a directory of the
    /// model's own under `cacheDir`; off, the cache is temporary: a directory
    /// under the temp root that takes copies only while memory drops states,
    /// removed at the stop. Unset ⇒ `MLX_PERSIST_PAGED_CACHE`, else off.
    pub persist_paged_cache: Option<bool>,
    /// The persistent cache's root (`--cache-dir`), default
    /// `~/.mlx-node/cache/prefix-cache`; needs `persistPagedCache`.
    pub cache_dir: Option<String>,
}

/// Load-time residency decision (the budget is admitted before payload
/// reads) plus the live pinned-bank total.
#[napi(object)]
#[derive(Default)]
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
    /// The host read behind `liveNonReclaimableBytes`: wired + compressor +
    /// anonymous - purgeable. All 0 without a host read.
    pub host_wired_bytes: f64,
    pub host_compressor_bytes: f64,
    pub host_anonymous_bytes: f64,
    pub host_purgeable_bytes: f64,
    /// What the plan holds beside the experts (fixed weights, draft banks,
    /// row cache, KV allowance, state staging) when residency was chosen.
    pub anonymous_need_bytes: f64,
    /// This process's MLX active plus cache bytes when residency was chosen.
    pub self_mlx_bytes: f64,
    /// Times since load a store the host gate refused at load moved up to
    /// whole banks after a turn, the host having room by then.
    pub residency_promotions: f64,
    /// Times since load the gate admitted such a move but the host could not
    /// take its growth (the whole banks less the partial banks, plus a
    /// layer's staging), so the store stayed partial with every slot kept.
    pub residency_promotion_refusals: f64,
    /// Drafted cycles since load whose rows the rule set to one: the draft
    /// proposed and the verify took the anchor alone. The speculative cycle
    /// counts (`mtpCycles`) leave these out.
    pub draft_one_row_cycles: f64,
    /// Decode steps since load that ran the rows of 2-4 scheduled turns in
    /// one pass (the lane step); a turn decoding alone adds none.
    pub lane_steps: f64,
    /// Where the routed experts live: every one in a direct bank, partial
    /// slots because the host refused the whole bank set or because the
    /// store plan streams them, or per-row imports.
    #[napi(ts_type = "'direct' | 'host-gate' | 'store-plan' | 'rows'")]
    pub residency: String,
    /// Routed experts per layer in the checkpoint.
    pub expert_count: u32,
    /// Present when the experts take partial slots: the host gate's or the
    /// store plan's reason.
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
    /// Replay passes after a tentative device route missed the slots: each
    /// pass a decode token runs again from its missed layer (on device or
    /// host routes; one token can run several) and each prefill window.
    pub route_replays: f64,
    /// Decode tokens that ran a tentative device-route pass, and those of
    /// them whose routes missed the slots.
    pub route_tries: f64,
    pub route_miss_tokens: f64,
    /// Decode layers replays ran again on device routes after reading the
    /// missed experts, and those they ran again on host routes.
    pub replayed_layers: f64,
    pub host_tail_layers: f64,
    /// Times the route guard sent decode tokens to host routes.
    pub route_guard_trips: f64,
    /// Slot fills since load, the wall time inside them in microseconds, and
    /// the bytes they asked of the checkpoint files (each wanted range
    /// rounded out to 16 KiB pages, a page two ranges share read once, no
    /// bytes between two ranges).
    pub fill_calls: f64,
    pub fill_micros: f64,
    pub fill_read_bytes: f64,
    /// The share of `fillCalls`, `fillMicros`, `fillReadBytes`,
    /// `slotMisses`, `checkpointBytesRead` and `routeReplays` that prefill
    /// passes moved since load; decode fills and MTP verification passes are
    /// not in them.
    pub prefill_fill_calls: f64,
    pub prefill_fill_micros: f64,
    pub prefill_fill_read_bytes: f64,
    pub prefill_slot_misses: f64,
    pub prefill_checkpoint_bytes_read: f64,
    pub prefill_route_replays: f64,
    /// Layer-major prefill groups of two or more windows since load, and the
    /// windows they ran.
    pub group_passes: f64,
    pub group_windows: f64,
    /// Stored prefix states with a RAM copy, and the bytes of every stored
    /// state.
    pub prefix_states: f64,
    pub prefix_state_bytes: f64,
    /// States published since load, and turns that resumed from one.
    pub prefix_published: f64,
    pub prefix_restored: f64,
    /// Points not published since load, by reason (a disk read-back the
    /// room or live check refuses counts too): the room the plan leaves for
    /// states had none even after the oldest went; the host's free memory or
    /// MLX's active ceiling had none; the blocks below the point did not
    /// register; a pass a cold prefill does not run came before it; a load
    /// with the MTP head had no draft history there.
    pub prefix_refused_room: f64,
    pub prefix_refused_live: f64,
    pub prefix_refused_register: f64,
    pub prefix_refused_off_grid: f64,
    pub prefix_refused_draft: f64,
    /// Text turns that planned no point since load: the request asked for no
    /// reuse; or no block-aligned row lies past the resume row before the
    /// prompt's last token.
    pub prefix_skipped_reuse_off: f64,
    pub prefix_skipped_no_points: f64,
    /// Turns that ended before their replay point was reached: a pass ran
    /// past it, or the turn was cancelled, failed or reset first.
    pub prefix_missed_replay: f64,
    /// States dropped since load because a block of their chain left the
    /// prefix cache, and because memory gave up their RAM copy.
    pub prefix_dropped_kv: f64,
    pub prefix_dropped_memory: f64,
    /// The stored states' disk tier is open and takes writes.
    pub prefix_disk_enabled: bool,
    /// States waiting for their disk write, the one in flight included.
    pub prefix_disk_waiting: f64,
    /// States written whose whole chain is on disk.
    pub prefix_disk_durable: f64,
    /// Waiting states not written: their conversation went on.
    pub prefix_disk_unneeded: f64,
    /// States not written or dropped (chain incomplete, room, a failed
    /// write, a closed cache), and copies the writer refused.
    pub prefix_disk_refused: f64,
    /// States read back from disk into memory.
    pub prefix_disk_restored: f64,
    /// States a persistent cache took back at load: on disk only until a
    /// turn reads one back.
    pub prefix_disk_adopted: f64,
    /// States removed from disk: a block or draft segment of their chain
    /// was not there, or they did not read back.
    pub prefix_disk_dropped_chain: f64,
    /// Block and draft-segment copies waiting for the writer.
    pub prefix_disk_owed_blocks: f64,
    /// Bytes of states written.
    pub prefix_disk_state_bytes_written: f64,
}

impl Qwen4ExpResidencyInfo {
    /// The stored states' counters, each under its `prefix*` name.
    fn with_prefix_counters(self, c: &prefix_state::Counters) -> Self {
        Self {
            prefix_published: c.published as f64,
            prefix_restored: c.restored as f64,
            prefix_refused_room: c.refused_room as f64,
            prefix_refused_live: c.refused_live as f64,
            prefix_refused_register: c.refused_register as f64,
            prefix_refused_off_grid: c.refused_off_grid as f64,
            prefix_refused_draft: c.refused_draft as f64,
            prefix_skipped_reuse_off: c.skipped_reuse_off as f64,
            prefix_skipped_no_points: c.skipped_no_points as f64,
            prefix_missed_replay: c.missed_replay as f64,
            prefix_dropped_kv: c.dropped_kv as f64,
            prefix_dropped_memory: c.dropped_memory as f64,
            ..self
        }
    }
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
    /// A DFlash2 draft is attached.
    has_draft: bool,
    /// Whether this load owns a Metal block-paged KV pool (`false` on CUDA/CPU
    /// flat BF16 KV execution).
    has_paged_cache: bool,
    context_limits: Qwen4ExpContextLimits,
    residency: memory::Plan,
    live: Arc<live::Live>,
    lookup: Arc<weights::LookupStats>,
    supports_images: bool,
    prefix: Arc<prefix_disk::Telemetry>,
    prefix_closer: Option<std::sync::Weak<prefix_disk::Closer>>,
    state_tier: crate::disk_tier::TierHandle,
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
                load_plan::refuse_head_with_draft(
                    LABEL,
                    has_mtp,
                    options.draft_model_path.as_deref(),
                )?;
                let draft = options
                    .draft_model_path
                    .as_deref()
                    .map(|dir| decoder::load_draft(Path::new(dir), &config))
                    .transpose()?;
                let draft_price = draft
                    .as_ref()
                    .map_or_else(load_plan::DraftPrice::default, |d| {
                        load_plan::DraftPrice::of(&config, &d.config, d.weight_bytes)
                    });
                let env = std::env::var(crate::cold_tier::PERSIST_PAGED_CACHE_ENV).ok();
                let (tier_options, disk) = disk_plan(LABEL, &options, env.as_deref())?;
                let disk = disk && crate::accelerator::host_has_metal();
                let vision = media::Vision::metadata(&raw, &weights, &config)?;
                let supports_images = vision.available();
                let tokenizer = Arc::new(Qwen3Tokenizer::load_from_file_sync(
                    assets
                        .join("tokenizer.json")
                        .to_str()
                        .ok_or_else(|| Error::from_reason("Non-UTF8 model path"))?,
                )?);
                check_tokenizer_tags(&tokenizer)?;
                let open = match tokenizer.encode_sync("<|im_start|>", Some(false))?[..] {
                    [id] => {
                        match tokenizer.encode_sync("<|im_start|>assistant", Some(false))?[..] {
                            [first, role] if first == id => vec![id, role],
                            _ => vec![id],
                        }
                    }
                    _ => Vec::new(),
                };
                let defaults =
                    crate::engine::persistence::parse_generation_defaults(Path::new(&assets));
                let live = memory::live_memory().ok_or_else(|| {
                    Error::from_reason(format!(
                        "{LABEL} memory plan: the device working set is unreadable; refusing an unplanned load"
                    ))
                })?;
                let margin = load_plan::host_margin();
                let inventory = weights.load_inventory(&config, supports_images)?;
                let decision = load_plan::plan_store_with_margin(
                    LABEL,
                    &config,
                    &inventory,
                    live,
                    max_memory,
                    max_context,
                    has_mtp,
                    disk,
                    draft_price,
                    margin,
                )?;
                weights.set_store_plan(decision.store, inventory.store_fixed_bytes);
                let mut decoder = decoder::Decoder::new(config.clone(), weights)?;
                decoder.register_cache_limit();
                decoder.weights.select_residency(
                    inventory
                        .store_fixed_bytes
                        .saturating_add(decision.store.row_cache_bytes)
                        .saturating_add(decision.allowance)
                        .saturating_add(load_plan::state_staging(&config, has_mtp, disk)),
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
                                "{LABEL} memory plan: the device working set is unreadable"
                            ))
                        })?;
                        load_plan::plan_engine_with_margin(
                            LABEL,
                            &config,
                            live,
                            held_back.saturating_add(inventory.vision_bytes),
                            max_memory,
                            has_mtp,
                            disk,
                            draft_price,
                            decision.states,
                            margin,
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
                    if crate::accelerator::host_has_metal() {
                        let adapter = paged::create(&config, &pool_plan, plan.ceiling_bytes())?;
                        let pool_bytes =
                            adapter.pool_allocated_bytes().map_err(Error::from_reason)?;
                        tracing::info!(
                            "{LABEL} pool: kv_format={}, {} B blocks",
                            config.kv_format.as_str(),
                            adapter.bytes_per_block().map_err(Error::from_reason)?
                        );
                        decoder.paged = Some(adapter);
                        let guard = crate::cache_limit::coordinator().register_pool(pool_bytes);
                        (Some(guard), plan, pool_bytes)
                    } else {
                        tracing::info!(
                            "{LABEL} flat KV cache: kv_format={}, window={} tokens",
                            config.kv_format.as_str(),
                            pool_plan.lane.window.maximum()
                        );
                        decoder.paged = None;
                        (None, plan, 0)
                    }
                };
                if let (Some(adapter), Some(guard)) = (decoder.paged.as_mut(), pool_guard.as_ref())
                {
                    let pool_id = guard.id();
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
                let has_paged_cache = decoder.paged.is_some();
                let mut context_limits = Qwen4ExpContextLimits::from_plan(
                    &config,
                    &memory_plan,
                    max_context.unwrap_or_else(|| config.served_context_tokens()),
                )?;
                if !has_paged_cache {
                    context_limits.paged_block_capacity = 0;
                }
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
                let residency = decoder.weights.plan.clone();
                let live = Arc::clone(&decoder.weights.live);
                let lookup = Arc::clone(&decoder.weights.lookup);
                let assets = assets.to_string_lossy().into_owned();
                let telemetry = Arc::new(prefix_disk::Telemetry::default());
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
                    prefix_states: prefix_state::States::new(open),
                    disk: None,
                    telemetry: Arc::clone(&telemetry),
                    draft: None,
                };
                let has_draft = draft.is_some();
                inner.set_draft(draft)?;
                if disk && has_paged_cache {
                    let identity = inner
                        .decoder
                        .weights
                        .checkpoint_identity()
                        .map(str::to_owned);
                    inner.open_prefix_disk(
                        &tier_options,
                        identity.as_deref(),
                        options.auxiliary_model_path.as_deref().map(Path::new),
                    );
                }
                let prefix_closer = inner.disk.as_ref().map(prefix_disk::Disk::closer);
                let state_tier = inner
                    .disk
                    .as_ref()
                    .map(prefix_disk::Disk::tier_handle)
                    .unwrap_or_default();
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
                        has_draft,
                        has_paged_cache,
                        context_limits,
                        residency,
                        live,
                        lookup,
                        supports_images,
                        telemetry,
                        prefix_closer,
                        state_tier,
                    ),
                ))
            },
            |state, receiver| state.drive(receiver),
        );
        let (
            config,
            assets_path,
            has_mtp,
            has_draft,
            has_paged_cache,
            context_limits,
            residency,
            live,
            lookup,
            supports_images,
            prefix,
            prefix_closer,
            state_tier,
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
        // The closer and the disk wake the model thread through a weak
        // sender: they never keep it alive.
        if let (Some(sender), Some(closer)) = (
            thread.cmd_sender(),
            prefix_closer.as_ref().and_then(std::sync::Weak::upgrade),
        ) {
            let sender = sender.downgrade();
            closer.set_wake(Arc::new(move || {
                if let Some(sender) = sender.upgrade() {
                    let _ = sender.send(ModelCommand::MemoryControl);
                }
            }));
        }
        Ok(Self {
            thread,
            config,
            assets_path,
            has_mtp,
            has_draft,
            has_paged_cache,
            context_limits,
            residency,
            live,
            lookup,
            supports_images,
            prefix,
            prefix_closer,
            state_tier,
        })
    }
    /// Closes the stored states' SSD cache at a clean stop: on a persistent
    /// cache it asks the model thread to write the waiting states, the
    /// newest first, then drains every copy, until `deadlineMs`
    /// (milliseconds on `Date.now()`'s clock, so models closed in turn share
    /// one deadline); a temporary cache is removed. The model thread gets
    /// all of the time but the last second (at most half of it): a turn in
    /// flight that holds it longer leaves its waiting states unwritten,
    /// while the copies already queued still land. The cache takes no write
    /// and removes nothing afterwards. Resolves true only when no state was
    /// left waiting and every copy and write landed by the deadline; true
    /// without a cache, or after `dispose`, which closed it. A call while
    /// another close runs gets that close's outcome.
    #[napi]
    pub async fn close_cache(&self, deadline_ms: f64) -> Result<bool> {
        let Some(closer) = self
            .prefix_closer
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
        else {
            return Ok(true);
        };
        let deadline = crate::models::qwen3_5::dflash2_state::disk::instant_at(deadline_ms);
        napi::bindgen_prelude::spawn_blocking(move || closer.close(deadline))
            .await
            .map_err(|error| Error::from_reason(format!("Qwen4Exp closeCache: {error}")))
    }
    /// The stored states' SSD cache: `'persistent'` or `'temp'` while it is
    /// open, `'off'` without one, after `closeCache`, and once a write failed
    /// for anything but room (it takes no write after).
    #[napi(ts_return_type = "'persistent' | 'temp' | 'off'")]
    pub fn state_tier_mode(&self) -> String {
        match self.prefix.mode(&self.state_tier) {
            1 => "persistent",
            2 => "temp",
            _ => "off",
        }
        .to_string()
    }
    /// The directory name of the stored states' open persistent SSD cache
    /// under its cache root, read from the cache itself; null without one,
    /// for a temporary cache, and after `closeCache` or `dispose`.
    #[napi]
    pub fn state_tier_namespace(&self) -> Option<String> {
        self.state_tier.namespace()
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
            host_wired_bytes: live.host_wired.load(Ordering::Acquire) as f64,
            host_compressor_bytes: live.host_compressor.load(Ordering::Acquire) as f64,
            host_anonymous_bytes: live.host_anonymous.load(Ordering::Acquire) as f64,
            host_purgeable_bytes: live.host_purgeable.load(Ordering::Acquire) as f64,
            anonymous_need_bytes: live.anonymous_need.load(Ordering::Acquire) as f64,
            self_mlx_bytes: live.self_mlx_bytes.load(Ordering::Acquire) as f64,
            residency_promotions: live.promotions.load(Ordering::Acquire) as f64,
            residency_promotion_refusals: live.promotion_refusals.load(Ordering::Acquire) as f64,
            draft_one_row_cycles: live.draft_one_row_cycles.load(Ordering::Acquire) as f64,
            lane_steps: live.lane_steps.load(Ordering::Acquire) as f64,
            residency: live.residency().as_str().to_owned(),
            expert_count: self.config.num_experts as u32,
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
            route_tries: live.route_tries.load(Ordering::Relaxed) as f64,
            route_miss_tokens: live.route_miss_tokens.load(Ordering::Relaxed) as f64,
            replayed_layers: live.replayed_layers.load(Ordering::Relaxed) as f64,
            host_tail_layers: live.host_tail_layers.load(Ordering::Relaxed) as f64,
            route_guard_trips: live.route_guard_trips.load(Ordering::Relaxed) as f64,
            fill_calls: live.fill_calls.load(Ordering::Relaxed) as f64,
            fill_micros: live.fill_micros.load(Ordering::Relaxed) as f64,
            fill_read_bytes: live.fill_read_bytes.load(Ordering::Relaxed) as f64,
            prefill_fill_calls: live.prefill_fill_calls.load(Ordering::Relaxed) as f64,
            prefill_fill_micros: live.prefill_fill_micros.load(Ordering::Relaxed) as f64,
            prefill_fill_read_bytes: live.prefill_fill_read_bytes.load(Ordering::Relaxed) as f64,
            prefill_slot_misses: live.prefill_slot_misses.load(Ordering::Relaxed) as f64,
            prefill_checkpoint_bytes_read: live.prefill_bytes_read.load(Ordering::Relaxed) as f64,
            prefill_route_replays: live.prefill_route_replays.load(Ordering::Relaxed) as f64,
            group_passes: live.group_passes.load(Ordering::Relaxed) as f64,
            group_windows: live.group_windows.load(Ordering::Relaxed) as f64,
            prefix_states: self.prefix.states.load(Ordering::Relaxed) as f64,
            prefix_state_bytes: self.prefix.state_bytes.load(Ordering::Relaxed) as f64,
            prefix_disk_enabled: self.prefix.mode(&self.state_tier) != 0,
            prefix_disk_waiting: self.prefix.waiting.load(Ordering::Relaxed) as f64,
            prefix_disk_durable: self.prefix.durable.load(Ordering::Relaxed) as f64,
            prefix_disk_unneeded: self.prefix.unneeded.load(Ordering::Relaxed) as f64,
            prefix_disk_refused: self.prefix.refused.load(Ordering::Relaxed) as f64,
            prefix_disk_restored: self.prefix.disk_restored.load(Ordering::Relaxed) as f64,
            prefix_disk_adopted: self.prefix.adopted.load(Ordering::Relaxed) as f64,
            prefix_disk_dropped_chain: self.prefix.dropped_chain.load(Ordering::Relaxed) as f64,
            prefix_disk_owed_blocks: self.prefix.owed.load(Ordering::Relaxed) as f64,
            prefix_disk_state_bytes_written: self.prefix.state_bytes_written.load(Ordering::Relaxed)
                as f64,
            ..Default::default()
        }
        .with_prefix_counters(&self.prefix.counters())
    }
    #[napi]
    pub fn model_assets_path(&self) -> String {
        self.assets_path.clone()
    }
    /// The MTP head or a DFlash2 draft is attached: a turn that asks for
    /// speculation (`enableMtp`) can run its cycle.
    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.has_mtp || self.has_draft
    }
    /// SSD-backed draft and verification work is opt-in; head availability
    /// alone does not establish a latency benefit for the current workload.
    #[napi]
    pub fn mtp_auto_enabled(&self) -> bool {
        false
    }
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.has_paged_cache
    }
    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        if self.has_paged_cache {
            scheduler_capacity()
        } else {
            1
        }
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
