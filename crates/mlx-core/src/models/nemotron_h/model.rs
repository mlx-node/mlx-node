//! NemotronH model: NemotronHInner + ChatBackend/MtpBackend + NAPI class.
//!
//! Flat-cache backend: Mamba-2 SSM states and attention K/V live in
//! per-layer NemotronHLayerCache slots owned by the model thread. No
//! paged/scheduler hooks yet (a follow-up agent adds concurrency); the
//! speculative MTP head runs the engine's flat run_mtp_turn loop with a
//! depth-1 drafter reading the backbone's final attention layer KV.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::MxArray;
use crate::engine::ThinkingPolicy;
use crate::engine::backend::{
    ChatBackend, DecodeStep, MtpBackend, MtpStepper, MtpTurnSetup, PagedBackend, PagedPrefix,
    ResetScope, SaveStateArgs, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::{ChatCmd, FromChatCmd, handle_chat_cmd};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, HybridSchedulerCommand, HybridSchedulerState, HybridStepExecutor,
    NoRestoreTicket, ScheduledPrefixAdmission, scheduler_max_num_seqs_for,
};
use crate::engine::plan::{
    ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan, SpeculativeKind,
    SpeculativePlan,
};
use crate::engine::types::ChatResult;
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

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum NemotronHCmd {
    Chat(Box<ChatCmd>),
    SchedulerStats {
        reply: ResponseTx<engine::SchedulerStatsJs>,
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

/// Route a thread command to the engine's chat handler.
pub(crate) fn handle_nemotron_h_cmd(inner: &mut NemotronHInner, cmd: NemotronHCmd) {
    match cmd {
        NemotronHCmd::Chat(chat) => handle_chat_cmd(inner, *chat),
        NemotronHCmd::SchedulerStats { reply } => {
            let _ = reply.send(Ok(engine::scheduler::SchedulerStats::default().to_js()));
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
    /// Set true by the loader only after the MTP weight set is COMPLETE.
    pub(crate) mtp_weights_loaded: bool,
    /// Flat MTP mid-cycle-stop desync latch: forces a full re-prefill on the
    /// next AR turn.
    pub(crate) flat_mtp_caches_desynced: bool,
    /// Parsed generation_config.json sampling/stop defaults.
    pub(crate) gen_defaults: crate::engine::ModelGenerationDefaults,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache) covering
    /// the six GQA attention layers. Only `full_attention` layers route
    /// through the adapter (indexed by attention-layer ordinal 0..6); mamba
    /// and moe layers keep their own per-request state / stateless forward.
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    /// Per-request recurrent state for the scheduler lane: one full
    /// `Vec<NemotronHLayerCache>` per live sequence, so the mamba SSM/conv
    /// states are swapped into `caches` for serial prefill/finalize and
    /// stacked into `[N, ...]` rows for batched decode (lfm2 ShortConv
    /// pattern).
    pub(crate) scheduled_caches: HashMap<SeqId, Vec<NemotronHLayerCache>>,
    /// The sequence whose per-request caches currently sit in `caches`.
    pub(crate) active_scheduled_seq: Option<SeqId>,
    /// Whether the most recent `prime_prefix_state_for` decided the live
    /// mamba states already reflected the incoming prompt's cached prefix
    /// (Pass 1 skipped). Read by `paged_perf_prefill_tokens` so telemetry
    /// reports the suffix-scale numerator when Pass 1 did not run.
    pub(crate) last_paged_prefill_reused_mamba_state: bool,
    /// Quantized checkpoints (NVFP4/MXFP8) dispatch different matmul kernels
    /// for M=1 vs M>=2 rows, so a batched `[N,1]` projection differs from the
    /// single-row decode by a few ULP. NemotronH's 23 mamba layers amplify
    /// that into token flips by step ~8, so the batched lane runs the
    /// quantized projections per row (M=1, bit-identical to the scalar path)
    /// while attention stays batched (its projections are dense bf16, which
    /// the kernels compute bit-exactly for any M). Mirrors Qwen3.5 MoE's
    /// `preserve_singleton_projection_graphs` contract.
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
            flat_mtp_caches_desynced: false,
            gen_defaults: crate::engine::ModelGenerationDefaults::default(),
            paged_adapter,
            scheduled_caches: HashMap::new(),
            active_scheduled_seq: None,
            last_paged_prefill_reused_mamba_state: false,
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
    for i in 0..num_layers {
        caches.push(if config.is_mamba_layer(i) {
            let m = match &layers[i].mixer {
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

/// Construct the block-paged KV adapter covering only the GQA attention
/// layers (pool indexed by attention-layer ordinal). Gated on
/// `use_block_paged_cache` (default-on when None) and Metal availability;
/// returns `None` when either gate is closed.
fn build_paged_adapter(config: &NemotronHConfig) -> Result<Option<PagedKVCacheAdapter>> {
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
    let gpu_memory_mb = config.paged_cache_memory_mb.unwrap_or(2048);
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
    let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
        num_blocks, block_size,
    )));
    let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
    let pool =
        mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, cache_dtype).map_err(|e| {
            Error::from_reason(format!(
                "Failed to construct LayerKVPool for NemotronH block-paged adapter: {e}"
            ))
        })?;
    let adapter = PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
        Error::from_reason(format!(
            "Failed to construct NemotronH PagedKVCacheAdapter: {e}"
        ))
    })?;
    Ok(Some(adapter))
}

/// Whether the live mamba states (held in `caches` / `scheduled_caches` for
/// the sequence) are already at the `cached_prefix_len` boundary for this
/// exact token prefix: the turn strictly extends the immediately-preceding
/// saved history byte-for-byte, so the parked per-request states reflect the
/// prefix. Any mismatch (first turn, aborted-since-last-save, foreign/partial
/// prefix hit) falls through to the Pass-1 reconstruction in
/// [`NemotronHInner::run_paged_prefill_chunk`]. The mamba recurrent state is
/// non-invertible, so this is the only legal fast path (lfm2
/// `conv_state_reusable` mirror).
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

    /// MTP-routing predicate for the whole-turn executors: an MTP-requested
    /// turn runs the FLAT speculative core (never the generic paged
    /// executor) exactly when the request opted in, a complete MTP head is
    /// loaded, and the turn is not streaming (the flat MTP core has no
    /// streaming arm; streaming MTP turns keep the paged AR fallback).
    pub(crate) fn mtp_flat_routing_required(
        &self,
        params: &crate::engine::params::ChatParams,
        streaming: bool,
    ) -> bool {
        !streaming && params.enable_mtp && self.has_mtp_weights()
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
    }

    /// Eval every live cache array (post-prefill sync).
    fn eval_caches_internal(&self) -> Result<()> {
        let mut refs = Vec::new();
        for c in self.caches.iter() {
            c.collect_arrays(&mut refs);
        }
        MxArray::eval_arrays(&refs)
    }

    /// Save the session history, aligning it with the physical cache length
    /// (drop-last always - the shared decode loop never forwards the final
    /// committed token, and the Mamba recurrent state is non-invertible).
    fn save_cache_state_internal(
        &mut self,
        reuse_cache: bool,
        tokens: &[u32],
        generated_tokens: &[u32],
        drop_last_always: bool,
        finish_reason: &str,
    ) {
        if reuse_cache {
            let mut full_history = tokens.to_vec();
            let drop_last = drop_last_always || finish_reason == "length";
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
        let mut offset: i64 = 0;
        while total_len - offset > PREFILL_STEP_SIZE {
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
                let _ = self.forward(&chunk)?;
            }
            self.eval_caches_internal()?;
            crate::array::clear_cache();
            offset += PREFILL_STEP_SIZE;
        }
        if offset > 0
            && self
                .turn_cancel
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
        {
            return Err(Error::from_reason("prefill cancelled"));
        }
        let remaining = prompt.slice_axis(1, offset, total_len)?;
        {
            let _stream_ctx = StreamContext::new(generation_stream);
            self.forward(&remaining)
        }
    }
}

impl NemotronHInner {
    // ===================== Paged / scheduler recurrent state =====================

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
            return Ok(());
        }
        self.park_active_scheduled_caches();
        self.caches = self
            .scheduled_caches
            .remove(&seq_id)
            .unwrap_or_else(|| fresh_caches(&self.config, &self.layers).expect("fresh caches"));
        self.active_scheduled_seq = Some(seq_id);
        Ok(())
    }

    fn park_active_scheduled_caches(&mut self) {
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

    fn release_scheduled_caches_for(&mut self, seq_id: SeqId) {
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

    // ===================== Paged prefill =====================

    /// Forward the cached prefix through ALL layers with attention run as a
    /// FLAT causal self-prefill whose K/V are discarded (the prefix K/V already
    /// live in the paged pool). Rebuilds the exact inter-layer residual stream
    /// so every mamba layer's conv/SSM state lands on the `cached_prefix_len`
    /// boundary before the suffix forward continues from it (lfm2 Pass-1
    /// pattern; the mamba state is non-invertible so it must be re-derived from
    /// scratch whenever the live caches are not already known to hold it).
    fn run_mamba_only_prefill(&mut self, prefix_tokens: &[u32]) -> Result<()> {
        if prefix_tokens.is_empty() {
            return Ok(());
        }
        let input_ids = MxArray::from_uint32(prefix_tokens, &[1, prefix_tokens.len() as i64])?;
        let mut hidden = self.embedding.forward(&input_ids)?;
        let num_layers = self.layers.len();
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            let layer: &NemotronHDecoderLayer = unsafe { &*self.layers.as_ptr().add(layer_idx) };
            let normed = layer.norm.forward(&hidden)?;
            let out = match &layer.mixer {
                NemotronHMixer::Mamba(m) => {
                    let cache: &mut NemotronHLayerCache =
                        unsafe { &mut *self.caches.as_mut_ptr().add(layer_idx) };
                    let state = cache.as_mamba_state_mut().ok_or_else(|| {
                        Error::from_reason("run_mamba_only_prefill: mamba cache missing")
                    })?;
                    m.forward(&normed, Some(state))?
                }
                NemotronHMixer::Attention(a) => {
                    // EXACT prefix reconstruction: run attention as a flat
                    // causal self-prefill with NO paged-pool I/O so the
                    // residual feeding downstream mamba layers is identical
                    // to the cold full-prefill arithmetic.
                    a.forward(&normed, None, None)?
                }
                NemotronHMixer::MoE(m) => m.forward(&normed)?,
            };
            hidden = hidden.add(&out)?;
        }
        Ok(())
    }

    /// One paged prefill slice: bring the mamba state to `cached_prefix_len`
    /// (Pass 1 unless already established), record the suffix in the adapter,
    /// and forward the suffix through all layers (attention via the paged
    /// pool). Returns the last-token logits `[vocab]`.
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

        if cached_prefix_len > 0 && !skip_reconstruction {
            let prefix = &full_tokens[..(cached_prefix_len as usize)];
            self.run_mamba_only_prefill(prefix)?;
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
        // Preserve the scalar decode graph for a one-row wave (quantized
        // matrix kernels can round differently when a singleton is forced
        // through a batched graph).
        if let [(seq_id, token_id)] = rows {
            self.activate_paged_seq(*seq_id)?;
            return self.run_paged_decode_step(*token_id);
        }
        // Quantized checkpoints: the batched [N,1] projections (mamba MXFP8,
        // MoE NVFP4, lm_head NVFP4) AND the batched paged-attention kernels
        // round differently from the single-row decode; the 23 mamba layers
        // amplify a few ULP into token flips. Run the full per-row scalar
        // decode (each row records its own token, writes K/V, and advances its
        // mamba state with M=1 kernels), so the batch output is bit-identical
        // to N single-row decodes. Dense checkpoints keep the fused path.
        if self.row_exact_decode_projections {
            let mut logits = Vec::with_capacity(rows.len());
            for &(seq_id, token_id) in rows {
                self.activate_paged_seq(seq_id)?;
                logits.push(self.run_paged_decode_step(token_id)?);
            }
            return MxArray::concatenate_many(logits.iter().collect(), Some(0));
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
        self.config
            .eos_token_ids
            .iter()
            .map(|&v| v as u32)
            .collect()
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
        // The EXPLICIT command reset must restore a fully cold state:
        // release the live request AND purge the allocator prefix cache so a
        // reset-then-rerun of the same prompt replays the cold prefill
        // (lfm2 rationale: the bf16 reduction order of a partial-prefix hit
        // differs from the cold full prefill).
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
            /* drop_last_always */ true,
            args.finish_reason,
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
                // Native MTP is flat-cache only for this family (the draft
                // head reads the FLAT KV). The plan keeps the paged adapter
                // exposed so plain AR turns stay on the paged lane; a sync
                // MTP-requested turn is re-routed to the flat speculative
                // core inside run_paged_turn (the engine also keeps
                // enable_mtp turns off the batched lane).
                supports_paged_attention: false,
            }),
        }
    }

    fn wired_limit_bytes(&self) -> Option<usize> {
        Some(self.config.estimate_memory_bytes() as usize)
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // Native MTP is flat-cache only (the draft head reads the FLAT KV),
        // but a paged adapter still resolves every fresh turn to this path
        // (TurnPlan gives paged attention precedence over the speculative
        // decoder, and this family's SpeculativePlan truthfully declares
        // supports_paged_attention: false). Route sync MTP-requested turns
        // to the flat speculative core so the draft+verify cycle actually
        // runs; streaming MTP turns keep the generic paged AR fallback (the
        // flat core has no streaming arm and would trip
        // whole_turn_outcome's Complete-on-streaming guard).
        if self.mtp_flat_routing_required(args.params, args.sink.is_some()) {
            let result = self.run_mtp_whole_turn(args)?;
            return Ok(TurnOutput::Complete(Box::new(result)));
        }
        crate::engine::paged_turn::run_paged_turn(self, args)
    }

    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        let result = self.run_mtp_whole_turn(args)?;
        Ok(TurnOutput::Complete(Box::new(result)))
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

/// Paged decode stepper for the exclusive/whole-turn lane (the paged analog
/// of NemotronHDecode). Drives the engine-owned run_paged_turn decode
/// loop through PagedBackend.
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

/// NemotronH paged prefix state: the effective prefix/suffix split the
/// adapter resolved, PLUS the full prompt tokens (the mamba Pass-1 rebuild
/// needs full_tokens[..cached_prefix_len], which the engine never hands to
/// paged_prefill).
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
        // The Mamba-2 recurrent state is non-invertible: a cached K/V prefix can
        // only be continued when the mamba state for it is already live in these
        // caches — i.e. when `plan` is a strict token-extension of the history
        // the previous turn left behind. Any other prefix hit (a different
        // owner's shared template head, a same-owner reset) would require
        // reconstructing the prefix mamba state from scratch, and the
        // flat-vs-paged attention reconstruction is not bit-identical to the
        // cold full prefill — the tiny residual differences accumulate through
        // the recurrent decode and flip greedy argmaxes mid-turn (observed as
        // confused rambling on every second session of a model). Force a cold
        // prefill (skip_lookup) in that case: always correct, at the cost of
        // forgoing prefix reuse across owners.
        let continuation_eligible = !owner_history.is_empty()
            && plan.len() > owner_history.len()
            && plan[..owner_history.len()] == owner_history[..];
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
        let slices = chunk_aligned_prefill_slices(cached, end, PREFILL_STEP_SIZE as u32, 128);
        let mut last = None;
        for (s, e) in slices {
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
        // NemotronH paged ALWAYS drops the last token: the decode loop never
        // forwards the final sampled token (the terminal forward is skipped),
        // so it is absent from the adapter and the mamba states. Mirror the
        // flat path's drop-last-always contract.
        self.save_cache_state_internal(
            reuse_cache,
            save_tokens,
            generated,
            /* drop_last_always */ true,
            /* finish_reason */ "",
        );
        Ok(())
    }

    fn paged_perf_prefill_tokens(&self, prompt_token_count: usize, suffix_len: usize) -> usize {
        // A foreign/partial prefix hit re-derives the mamba state over the
        // FULL prompt (Pass 1), so ttft measures full-prompt work; the
        // exact-owner carried-state path prefills only the suffix.
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
        // Same cold-prefill rule as the single-session lane: only a strict
        // token-extension of the live history may reuse a cached K/V prefix
        // (see `prime_prefix_state`); anything else forces skip_lookup.
        let continuation_eligible = !owner_history.is_empty()
            && tokens.len() > owner_history.len()
            && tokens[..owner_history.len()] == owner_history[..];
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
        // The engine's pinned break-set is a 2048-token grid from the
        // effective prefix; re-split every slice at Mamba-2 chunk-128
        // boundaries so no executed prefill forward splits a chunk.
        let slices =
            chunk_aligned_prefill_slices(start as u32, end as u32, PREFILL_STEP_SIZE as u32, 128);
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
        let turn_plan = self
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
        let cached_prefix_len = turn_plan.cached_prefix_len as usize;
        let reused_state = mamba_state_reusable(plan, owner_history, cached_prefix_len);
        self.last_paged_prefill_reused_mamba_state = reused_state;
        // A turn that is NOT a live continuation of the currently live request
        // (fresh owner, or a same-owner continue that had to reset) cannot
        // inherit the previous request's token history. The decode loop applies
        // penalties over this history and labels positions from it, so a stale
        // cross-owner history shifts the penalty context and position
        // accounting — observed as every second session on one model generating
        // confused rambling. Same-owner live continuations
        // (`ContinuedLivePrefix`) keep the history.
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

/// Split the absolute token range [start, end) into sub-slices of at most
/// slice_tokens tokens whose internal boundaries are all multiples of the
/// Mamba-2 chunk_size (128). The first sub-slice starts at start (the
/// effective cached-prefix boundary, possibly unaligned); every later
/// boundary is a chunk multiple, so no executed prefill forward splits a
/// chunk relative to the model's chunk-scan arithmetic.
pub(crate) fn chunk_aligned_prefill_slices(
    start: u32,
    end: u32,
    slice_tokens: u32,
    chunk_size: u32,
) -> Vec<(u32, u32)> {
    let mut slices = Vec::new();
    let mut s = start;
    if s % chunk_size != 0 {
        let a = s.div_ceil(chunk_size).saturating_mul(chunk_size);
        if a < end {
            slices.push((s, a));
            s = a;
        }
    }
    while s < end {
        let e = (s + slice_tokens).min(end);
        slices.push((s, e));
        s = e;
    }
    slices
}

/// Flat MTP propose/verify stepper for the engine-owned run_mtp_turn loop.
///
/// The drafter is STATELESS: its attention reads the backbone's final
/// attention layer KV (read-only), so there is no draft-side cache to
/// rewind on rejection. Rollback restores the main layer caches from the
/// pre-verify snapshot; restore_and_replay_main re-forwards the accepted
/// prefix. The single-step head clamps the requested depth to 1.
pub(crate) struct NemotronHMtpStepper<'a> {
    inner: &'a mut NemotronHInner,
    embedding: Embedding,
    /// Pre-verify snapshot of every layer cache.
    snap: Option<Vec<NemotronHLayerSnapshot>>,
    /// Stashed rollback/replay error (surfaced by restore_and_replay_main).
    replay_err: Option<Error>,
    /// Flat desync latch set by a mid-cycle stop.
    mtp_desynced: bool,
}

impl MtpStepper for NemotronHMtpStepper<'_> {
    fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    fn committed_history_active(&self) -> bool {
        false
    }

    fn forward_with_hidden(
        &mut self,
        ids: &MxArray,
        embedding: &Embedding,
    ) -> Result<(MxArray, MxArray, bool)> {
        // Step A contract (qwen3_5_moe convention): the engine seeds the
        // next cycle with hidden.shape_at(1) as the HIDDEN size, so the
        // returned hidden must be [1, hidden] (time dim 1 squeezed away).
        // The verify step uses forward_with_hidden_3d directly and keeps
        // [1, T, hidden] for its MtpVerifyOutput contract.
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
        let last_attn = self.inner.config.last_attention_idx().ok_or_else(|| {
            Error::from_reason("NemotronH MTP draft_step: config has no attention layer")
        })?;
        let kv = self.inner.caches[last_attn].as_kv_cache().ok_or_else(|| {
            Error::from_reason("NemotronH MTP draft_step: final attention layer cache missing")
        })?;
        let position = kv.get_offset();
        let h_next = mtp.draft_step(prev_hidden, prev_emb, Some(kv), position)?;
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
        let verify_in = MxArray::from_int32(&id_slice, &[1, (depth + 1) as i64])?;
        // The engine slices verify_hiddens[:, K, :], so hiddens must stay
        // [1, depth+1, hidden] - the raw 3D forward (forward_with_hidden
        // itself squeezes the Step-A seed down to [1, hidden]).
        let (logits, hidden) = self.inner.forward_with_hidden_3d(&verify_in, embedding)?;
        Ok(crate::models::qwen3_5::mtp_decode::MtpVerifyOutput {
            logits: Some(logits),
            hiddens: hidden,
            target_argmax: None,
            target_sparse: None,
        })
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

    fn commit_mtp(
        &mut self,
        _anchor: crate::models::qwen3_5::mtp_decode::MtpCommitAnchor,
        _seed_h: &MxArray,
        _verify_hiddens: &MxArray,
        _committed_ids: &[u32],
        _k_accepted: usize,
        _embedding: &Embedding,
    ) -> Result<()> {
        // v1 (no committed-history cache): the drafter is stateless and the
        // main caches are already correct after verify/replay.
        Ok(())
    }

    fn begin_cycle(&mut self, _chained_anchor: bool) {
        // The drafter is stateless; nothing to re-anchor.
    }

    fn eval_step(&self, token: &MxArray, logits: &MxArray, budget_forced: bool) {
        self.inner.async_eval_caches();
        token.eval();
        if budget_forced {
            logits.eval();
        }
    }

    fn eval_step_with_chained_hidden(&self, token: &MxArray, chained_hidden: &MxArray) {
        self.inner.async_eval_caches();
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

impl NemotronHInner {
    /// Whole-turn speculative MTP core (fresh and delta turns).
    ///
    /// Prefills the FULL token stream (re-prefilling the cached history on
    /// warm deltas - correct and simple for the flat path), samples the
    /// first token, then drives the engine-owned run_mtp_turn loop with
    /// the depth-1 NemotronHMtpStepper.
    fn run_mtp_whole_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<ChatResult> {
        if args.tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        let p = args.params;
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
        let mut token_history = self.cached_token_history.clone();
        let mut finish_reason = String::from("stop");
        let mut first_token_instant: Option<std::time::Instant> = None;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        // The flat MTP head reads the backbone KV, so the caches must hold
        // the FULL token stream. Reset + prefill (re-prefills the history
        // on warm deltas - the flat path's simple correctness tradeoff).
        // On a paged model (run_paged_turn routes sync MTP turns here) the
        // flat core takes over self.caches wholesale, so park any
        // adapter-owned scheduled sequence FIRST so its per-request mamba
        // and attention state survives for a later paged turn.
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
        let prefill_logits = self.chunked_prefill(&prompt, generation_stream)?;
        let seq_len = prefill_logits.shape_at(1)?;
        let mut last_logits = prefill_logits
            .slice_axis(1, seq_len - 1, seq_len)?
            .squeeze(Some(&[1]))?;
        profiler.end_prefill();

        last_logits = crate::engine::apply_all_penalties(last_logits, &token_history, p)?;
        let y = crate::sampling::sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let mut reasoning_tracker =
            crate::engine::ReasoningTracker::from_setup(&thinking, think_end_id);
        // last_in_cache is set from the run_mtp_turn outcome.

        let mut rng = rand::rng();
        let outcome = crate::engine::mtp_turn::run_mtp_turn(
            self,
            &mut rng,
            crate::engine::mtp_turn::MtpTurnArgs {
                y: y.clone(),
                // The NemotronH MTP head is a single-step predictor (the
                // drafter attends over the backbone KV without writing its
                // own K/V), so the requested depth is clamped to 1.
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
                prompt_hidden_position_base: 0,
                cancel_flag: turn_cancel.as_deref(),
            },
            None,
        )?;
        let last_in_cache = outcome.last_in_cache;
        if outcome.desynced {
            self.flat_mtp_caches_desynced = true;
        }

        self.save_cache_state_internal(
            p.reuse_cache,
            &tokens,
            &generated_tokens,
            !last_in_cache,
            &finish_reason,
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
        Ok(result)
    }
}

impl MtpBackend for NemotronHInner {
    type MtpDecode<'a>
        = NemotronHMtpStepper<'a>
    where
        Self: 'a;

    fn begin_mtp_decode(&mut self, _setup: &MtpTurnSetup<'_>) -> Result<Self::MtpDecode<'_>> {
        let embedding = self.embedding.clone();
        Ok(NemotronHMtpStepper {
            inner: self,
            embedding,
            snap: None,
            replay_err: None,
            mtp_desynced: false,
        })
    }
}

/// NVIDIA Nemotron 3.5 Lightning language model.
///
/// Hybrid MoE architecture (Mamba-2 SSM + GQA + pure MoE-FFN layers) with
/// an optional in-checkpoint MTP head. All model state lives on a
/// dedicated OS thread; NAPI methods dispatch commands via channels. When
/// the block-paged adapter is active the thread runs the engine-owned
/// `HybridSchedulerState` continuous-batching loop.
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
    /// RAII: unregisters this model's baseline from the cache-limit
    /// coordinator on drop.
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
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

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance (default-on unless `use_block_paged_cache: false`).
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
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
    use super::*;
    use crate::engine::persistence::compiled_forward_backend_available;

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
            rope_theta: 10000.0,
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
            n_routed_experts: 4,
            num_experts_per_tok: 1,
            routed_scaling_factor: 1.0,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            num_logits_to_keep: 1,
            mtp_layers_block_type: Vec::new(),
            n_mtp_layers: 0,
            paged_cache_memory_mb: Some(256),
            paged_block_size: Some(16),
            use_block_paged_cache: Some(true),
        }
    }

    fn argmax(vec: &[f32]) -> usize {
        vec.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// The tiny fixture's MoE layer is constructed with quantized (uint8)
    /// expert backends; install deterministic dense bf16 stacks so the
    /// gather kernels run (mirrors the loader's set_experts).
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
    /// Pure-function gate on the prefill break-set: the executor's slice
    /// splitter must never produce an internal boundary inside a Mamba-2
    /// chunk (every boundary except the range start is a multiple of the
    /// chunk size), and every slice is at most `slice_tokens` long.
    #[test]
    fn chunk_aligned_prefill_slices_never_split_a_chunk() {
        let cases: &[(u32, u32, u32, u32)] = &[
            // Cold start (the common parity case): the 2048 grid from 0
            // lands exactly on chunk multiples.
            (0, 3000, 2048, 128),
            // Warm partial-prefix hit at a non-chunk-multiple boundary.
            (112, 2160, 2048, 128),
            (112, 4096, 2048, 128),
            // Sub-chunk ranges.
            (0, 64, 2048, 128),
            (112, 140, 2048, 128),
            // Exact multiples.
            (128, 1280, 2048, 128),
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
                // Every boundary EXCEPT the very first start (the effective
                // cached-prefix boundary, which the family cannot move) must
                // be a chunk multiple: no break inside a chunk.
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
    /// Batched N=2 decode is T=0-identical to two serial decodes: prefill
    /// both requests cold (distinct cache salts), decode one token per
    /// request serially, reset both, re-prefill cold, then decode the same
    /// two tokens through `run_paged_decode_step_batched`, and compare the
    /// greedy (argmax) tokens row by row. This is the batched==serial
    /// equivalence gate for the mamba stack/scatter, the batched paged
    /// attention kernels, and the MoE routing over [N,1,H].
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
    /// T=0-identical to serial: it runs the per-row scalar decode and stacks
    /// the logits, so the assertion is structural (exercises the branch) plus
    /// token-level.
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
}
