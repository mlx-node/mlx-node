//! Qwen3.8-Flash-Next with SSD-streamed experts and hashed PLE rows.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

mod auxiliary;
mod config;
mod decoder;
mod gguf;
mod math;
mod media;
mod memory;
mod mtp;
mod packed_cache;
mod paged;
mod route_plan;
mod runtime_flags;
mod weights;

const MAX_PREFILL_CHUNK: usize = 1024;

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
    _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
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
        params.mtp_depth = params.mtp_depth.min(3);
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
    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
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
        let chunk_size = self.decoder.prefill_slice_size();
        for (i, chunk) in tokens.chunks(chunk_size).enumerate() {
            logits = Some(self.decoder.prefill_chunk(
                chunk,
                None,
                (i + 1) * chunk_size >= tokens.len(),
            )?);
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

/// Load-time residency decision; full residency is admitted before payload reads.
#[napi(object)]
pub struct Qwen4ExpResidencyInfo {
    pub policy: String,
    pub weight_budget_bytes: f64,
    pub hot_weight_bytes: f64,
    pub full_hot_residency: bool,
    pub prefill_chunk_tokens: u32,
    /// Free plus reclaimable file-backed memory observed at bootstrap.
    pub available_memory_bytes: Option<f64>,
    pub physical_memory_bytes: Option<f64>,
    /// Assembled matrices actually pinned, including partial projection banks.
    pub resident_bank_bytes: f64,
    /// Complete routed layers whose expert IDs stay on the GPU.
    pub resident_expert_layers: u32,
    /// Shared slot capacity for the remaining partial layers at bootstrap.
    pub partial_expert_slots: u32,
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
    prefill_chunk_size: usize,
    supports_images: bool,
}
#[napi]
impl Qwen4ExpModel {
    #[napi]
    pub async fn load(model_path: String, options: Option<Qwen4ExpLoadOptions>) -> Result<Self> {
        let (thread, ready) = ModelThread::spawn_with_scheduler(
            move || {
                let path = PathBuf::from(model_path);
                let mut weights = weights::Store::open(&path)?;
                let (mut config, assets) = if weights.gguf {
                    let c = gguf::config(&weights)?;
                    let a = gguf::assets(&path, &weights, &c)?;
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
                let mut cache_limit_guard =
                    crate::cache_limit::coordinator().register(decoder.weights.cache_budget());
                decoder.weights.prepare_hot()?;
                // Register the final reconciled budget, rather than retaining
                // the pre-pool estimate in the process-wide allocator policy.
                drop(cache_limit_guard);
                cache_limit_guard =
                    crate::cache_limit::coordinator().register(decoder.weights.cache_budget());
                decoder.prefill_chunk_size =
                    memory::prefill_window(&config, &decoder.weights.plan)?;
                let prefill_chunk_size = decoder.prefill_chunk_size;
                let residency = decoder.weights.plan.clone();
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
                    _cache_limit_guard: cache_limit_guard,
                };
                Ok((
                    crate::engine::hybrid_scheduler::HybridSchedulerState::new(inner)?,
                    (
                        config,
                        assets,
                        has_mtp,
                        paged_capacity,
                        residency,
                        prefill_chunk_size,
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
            prefill_chunk_size,
            supports_images,
        ) = ready
            .await
            .map_err(|e| Error::from_reason(e.to_string()))??;
        Ok(Self {
            thread,
            config,
            assets_path,
            has_mtp,
            paged_capacity,
            residency,
            prefill_chunk_size,
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
        Qwen4ExpResidencyInfo {
            policy: self.residency.policy.clone(),
            weight_budget_bytes: self.residency.budget as f64,
            hot_weight_bytes: self.residency.hot_bytes as f64,
            full_hot_residency: self.residency.resident,
            prefill_chunk_tokens: self.prefill_chunk_size as u32,
            available_memory_bytes: self.residency.available_bytes.map(|n| n as f64),
            physical_memory_bytes: self.residency.physical_bytes.map(|n| n as f64),
            resident_bank_bytes: self.residency.resident_bank_bytes as f64,
            resident_expert_layers: self.residency.resident_expert_layers as u32,
            partial_expert_slots: self.residency.partial_expert_slots as u32,
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
