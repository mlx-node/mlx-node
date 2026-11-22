//! Paged forward path, fused Metal kernel wiring, and [`PagedBackend`] / [`ChatBackend`]
//! integration for [`Inkling2Model`].
//!
//! Wires all Phase 3A fused Metal primitives into Inkling-2-Mini's 48-visit looped forward:
//! - `inkling2_loop_blend_rms_norm` (FP32 no-FMA cross-pass blend + sublayer RMSNorm)
//! - `inkling2_sconv_k4` (static SWA + dynamic Global causal `k=4` ShortConv)
//! - `inkling2_situ` (fused SiTU activation with optional per-expert `outer_scale`)
//! - `inkling2_gate_sink` (`O * sigmoid(gate) * sigmoid(lse - sink)`)
//! - `inkling2_paged_prefill_attention` (in-place G=3 D=128 MPP prefill/verify with SWA page skip
//!   and zero-bias `q8q` fast path)
//! - `inkling2_paged_attention` (varlen decode/verify over INT8 32-token KTM and BF16 pools)
//!
//! Supports:
//! - Chunked in-place prefill (up to 2048 rows/chunk, pruning expired SWA blocks after each chunk)
//! - Single-sequence and batched `[N, 1]` decode
//! - Multi-row varlen verify (`[N * K]` rows, for Phase 5 DSpark)

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use mlx_paged_attn::CacheRecency;
use mlx_paged_attn::prefill_plan::{self, Pass};
use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};
use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, FinalizeArgs, PagedBackend, PagedPrefix, ResetScope,
    SaveStateArgs, StreamEmitter, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::hybrid_scheduler::ScheduledPrefillPass;
use crate::engine::paged_epilogue::{FinalTokenPolicy, reconcile_paged_surplus};
use crate::engine::paged_stepper::{EvalPolicy, PagedStepModel, PagedStepper};
use crate::engine::paged_turn::run_paged_turn;
use crate::engine::plan::{
    ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan, SpeculativeKind,
    SpeculativePlan,
};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::models::qwen3_5::dflash2_state::store::StateClass;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::tools::inkling::{
    CONTENT_MODEL_END_SAMPLING_ID, IMAGE_PAD_ID, Inkling2RenderOptions, InklingParserState,
    InklingStreamParser, InklingToolSpec, count_inkling2_reasoning_tokens,
    output_marker_for_token_id, parse_inkling2_output, parse_reasoning_effort,
    render_inkling2_chat_messages_to_ids,
};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, PagedRaggedRow, SeqId};

use super::attention::global_log_scale_tau_for_positions;
use super::attn_res::{ATTN_RES_EPS, apply_sublayer_rms_norm};
use super::config::Inkling2Config;
use super::dspark::Inkling2DsparkModel;
use super::kernels::{
    Inkling2KvDtype, inkling2_block_attn_res, inkling2_gate_sink, inkling2_lm_head_softcap,
    inkling2_paged_attention, inkling2_paged_prefill_attention,
    inkling2_paged_prefill_mpp_supported, inkling2_qr_clip,
};
use super::kv_cache::{
    INKLING2_PAGED_PREFILL_MAX_CHUNK, Inkling2KVCacheCoordinator, Inkling2PagedConfig,
};
use super::model::{AttnInputNorm, Inkling2Inner, Inkling2PhysicalLayer, overwrite_image_pad_rows};
use super::short_conv::{
    Inkling2ShortConvState, Inkling2ShortConvTape, StreamShortConvTape, VisitShortConvState,
    VisitShortConvTape, fused_short_conv_kv_prep_taped, fused_short_conv_ragged_residual_taped,
};
use super::state_disk::{Inkling2StateStore, planned_rolling_checkpoints};
use super::vision::expand_image_tokens;

/// Per-run performance summary returned by [`Inkling2Model::generate_greedy_paged`].
#[derive(Debug, Clone, Copy)]
pub struct Inkling2PagedPerf {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub prefill_tok_per_sec: f64,
    pub decode_tok_per_sec: f64,
    pub swa_blocks_pruned: u32,
}

/// Runtime state owned by [`Inkling2Inner`] for block-paged hybrid inference.
pub struct Inkling2PagedRuntime {
    pub coordinator: Inkling2KVCacheCoordinator,
    /// Per-sequence 48-visit ShortConv rolling states (`k`, `v`, `attn_post`, `mlp_post`).
    pub conv_states: HashMap<SeqId, Inkling2ShortConvState>,
    /// Pre-step ShortConv snapshot for decode steps so `reconcile_paged_request_tokens`
    /// can roll back one trailing stop token without desynchronizing ShortConv state.
    pub(crate) prev_conv_snapshots: HashMap<SeqId, (u32, Inkling2ShortConvState)>,
    /// Exact ShortConv snapshot at the deepest complete `block_size` boundary reached by `seq_id`,
    /// so `StateClass::Ordinary` / preemption capture pairs exact SWA blocks with exact ShortConv.
    pub(crate) block_boundary_conv: HashMap<SeqId, (u32, Inkling2ShortConvState)>,
    /// Qwen 3.8-style hybrid prefix state store (RAM + optional StateDisk SSD cache).
    pub(crate) state_store: Inkling2StateStore,
    pub active_seq: SeqId,
    pub prefill_chunk_size: usize,
    pub(crate) paged_finalize_error: Option<String>,
}

impl Inkling2PagedRuntime {
    pub fn new(config: &Inkling2Config, paged_cfg: Inkling2PagedConfig) -> Result<Self> {
        Self::new_with_recency(config, paged_cfg, Arc::new(CacheRecency::new()))
    }

    pub(crate) fn new_with_recency(
        config: &Inkling2Config,
        paged_cfg: Inkling2PagedConfig,
        recency: Arc<CacheRecency>,
    ) -> Result<Self> {
        let prefill_chunk_size =
            (paged_cfg.max_chunk as usize).clamp(1, INKLING2_PAGED_PREFILL_MAX_CHUNK as usize);
        let coordinator = Inkling2KVCacheCoordinator::new_for_config_with_recency(
            config,
            paged_cfg,
            Arc::clone(&recency),
        )?;
        let block_size = coordinator.block_size();
        Ok(Self {
            coordinator,
            conv_states: HashMap::new(),
            prev_conv_snapshots: HashMap::new(),
            block_boundary_conv: HashMap::new(),
            state_store: Inkling2StateStore::new(block_size, recency),
            active_seq: 0,
            prefill_chunk_size,
            paged_finalize_error: None,
        })
    }

    pub(crate) fn from_coordinator(
        coordinator: Inkling2KVCacheCoordinator,
        max_chunk: u32,
        recency: Arc<CacheRecency>,
    ) -> Self {
        let block_size = coordinator.block_size();
        let prefill_chunk_size =
            (max_chunk as usize).clamp(1, INKLING2_PAGED_PREFILL_MAX_CHUNK as usize);
        Self {
            coordinator,
            conv_states: HashMap::new(),
            prev_conv_snapshots: HashMap::new(),
            block_boundary_conv: HashMap::new(),
            state_store: Inkling2StateStore::new(block_size, recency),
            active_seq: 0,
            prefill_chunk_size,
            paged_finalize_error: None,
        }
    }

    /// Record `seq_id`'s current ShortConv state if `pos_after` is an exact block boundary.
    pub(crate) fn record_block_boundary_conv(&mut self, seq_id: SeqId, pos_after: u32) {
        let bs = self.coordinator.block_size().max(1);
        if pos_after > 0
            && pos_after.is_multiple_of(bs)
            && let Some(conv) = self.conv_states.get(&seq_id).cloned()
        {
            self.block_boundary_conv.insert(seq_id, (pos_after, conv));
        }
    }

    /// Record the deepest `block_size` boundary crossed during a multi-token commit `[pos_after - keep .. pos_after]`
    /// using the cycle's [`Inkling2ShortConvTape`] so speculative multi-token commits that jump over a
    /// `block_size` multiple still snapshot an exact boundary state before SWA blocks prune past it.
    pub(crate) fn record_block_boundary_conv_from_tape(
        &mut self,
        seq_id: SeqId,
        pos_after: u32,
        keep: usize,
        tape: &Inkling2ShortConvTape,
    ) -> Result<()> {
        let bs = self.coordinator.block_size().max(1);
        if pos_after == 0 || keep == 0 {
            return Ok(());
        }
        let prev = pos_after.saturating_sub(u32::try_from(keep).unwrap_or(u32::MAX));
        let boundary = (pos_after / bs) * bs;
        if boundary == 0 || boundary <= prev {
            return Ok(());
        }
        if boundary == pos_after {
            self.record_block_boundary_conv(seq_id, pos_after);
            return Ok(());
        }
        let offset = (boundary - prev) as usize;
        if let Some(mut snap) = self.conv_states.get(&seq_id).cloned() {
            tape.commit_prefix(offset, &mut snap)?;
            self.block_boundary_conv.insert(seq_id, (boundary, snap));
        }
        Ok(())
    }

    pub fn ensure_conv_state(
        &mut self,
        seq_id: SeqId,
        config: &Inkling2Config,
    ) -> Result<&mut Inkling2ShortConvState> {
        match self.conv_states.entry(seq_id) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let state = Inkling2ShortConvState::new(
                    config.num_layer_visits(),
                    config.sconv_kernel_size,
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                    DType::BFloat16,
                )?;
                Ok(entry.insert(state))
            }
        }
    }

    pub fn reset_sequence(&mut self, seq_id: SeqId, config: &Inkling2Config) -> Result<()> {
        self.coordinator
            .reset_scheduled_request(seq_id)
            .map_err(Error::from_reason)?;
        let fresh = Inkling2ShortConvState::new(
            config.num_layer_visits(),
            config.sconv_kernel_size,
            config.num_key_value_heads * config.head_dim,
            config.hidden_size,
            DType::BFloat16,
        )?;
        self.conv_states.insert(seq_id, fresh);
        self.prev_conv_snapshots.remove(&seq_id);
        self.block_boundary_conv.remove(&seq_id);
        Ok(())
    }

    pub fn release_sequence(&mut self, seq_id: SeqId) {
        let _ = self.coordinator.release_request_all(seq_id);
        self.conv_states.remove(&seq_id);
        self.prev_conv_snapshots.remove(&seq_id);
        self.block_boundary_conv.remove(&seq_id);
    }

    pub fn snapshot_conv_state(&self, seq_id: SeqId) -> Option<Inkling2ShortConvState> {
        self.conv_states.get(&seq_id).cloned()
    }

    pub fn restore_conv_state(&mut self, seq_id: SeqId, state: Inkling2ShortConvState) {
        self.conv_states.insert(seq_id, state);
    }
}

/// Precomputed per-group attention metadata shared across all visits in the same group during
/// one forward call (Group 0: 8 Global visits, Group 1: 40 SWA visits).
struct GroupAttentionMeta {
    block_table_1d: Option<MxArray>,
    /// Per-sequence 1D block tables (for per-lane `inkling2_paged_prefill_attention` inside
    /// stacked `[N * 8]` multi-lane DSpark verify waves).
    per_seq_block_tables_1d: Vec<MxArray>,
    block_table_2d: MxArray,
    seq_lens: MxArray,
    cu_seqlens_q: MxArray,
    /// Prebuilt `int64` ragged write slot mapping (`(slot_mapping, total_queries)`) computed ONCE
    /// per KV group before the 48-visit verify loop so KV writes never stall mid-wave.
    ragged_slot_mapping: Option<(MxArray, u32)>,
}

/// How the paged attention primitive writes K/V and dispatches attention for a forward wave.
enum PagedWaveMode<'a> {
    /// Single sequence contiguous chunk (prefill or single-token decode).
    Single {
        seq_id: SeqId,
        first_pos: u32,
        floor: u32,
        rows: u32,
        use_mpp_prefill: bool,
        mpp_pass: Pass,
        pages_per_split: u32,
        int8_q_from: Option<u32>,
    },
    /// Multi-sequence uniform `[N, 1]` decode wave.
    BatchedDecode { planned_rows: &'a [(SeqId, u32)] },
    /// Packed multi-sequence (or single-sequence) varlen verify wave `[N * K]`.
    #[allow(dead_code)]
    RaggedVerify {
        ragged_rows: &'a [PagedRaggedRow],
        use_mpp_verify: bool,
        pages_per_split: u32,
        int8_q_from: Option<u32>,
    },
}

fn device_pages_per_split() -> u32 {
    #[cfg(target_os = "macos")]
    {
        let gpu_gen = unsafe { mlx_sys::mlx_gpu_architecture_gen() };
        let cores = unsafe { mlx_sys::mlx_gpu_core_count() }.max(0) as u32;
        prefill_plan::pages_per_split(gpu_gen, cores)
    }
    #[cfg(not(target_os = "macos"))]
    {
        32
    }
}

pub(crate) fn device_int8_query_from() -> Option<u32> {
    #[cfg(target_os = "macos")]
    {
        let gpu_gen = unsafe { mlx_sys::mlx_gpu_architecture_gen() };
        prefill_plan::int8_query_from(gpu_gen)
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Resolve the active INT8-query prefill threshold for `(kv_format, activations)`.
/// Returns `None` when queries stay BF16 (either BF16 KV cache, `activations == Bf16`,
/// or GPU without INT8-query MPP support).
pub(crate) fn resolved_int8_query_from(
    kv_format: crate::transformer::KvFormat,
    activations: crate::transformer::Activations,
) -> Option<u32> {
    if kv_format != crate::transformer::KvFormat::Int8
        || activations == crate::transformer::Activations::Bf16
    {
        None
    } else {
        device_int8_query_from()
    }
}

/// Build `[num_seqs, max_blocks]` (and optional 1D `[num_blocks]` when `num_seqs == 1`) block-table
/// metadata for one KV cache group across `seqs = [(seq_id, query_len)]`.
fn build_group_attention_meta(
    adapter: &PagedKVCacheAdapter,
    seqs: &[(SeqId, u32)],
) -> Result<GroupAttentionMeta> {
    let num_seqs = seqs.len();
    if num_seqs == 0 {
        return Err(Error::from_reason(
            "build_group_attention_meta: empty sequence list",
        ));
    }
    let mut seq_lens_vec = Vec::with_capacity(num_seqs);
    let mut cu_q_vec = Vec::with_capacity(num_seqs + 1);
    cu_q_vec.push(0i32);
    let mut max_blocks = 0usize;
    let mut tables = Vec::with_capacity(num_seqs);
    let mut running_q = 0i32;

    for &(seq_id, q_len) in seqs {
        let table = adapter.block_table_for(seq_id).ok_or_else(|| {
            Error::from_reason(format!(
                "Inkling2 paged attention: missing block table for seq {seq_id}"
            ))
        })?;
        let ctx_len = i32::try_from(table.num_tokens()).map_err(|_| {
            Error::from_reason(format!(
                "Inkling2 paged attention: seq {seq_id} length exceeds i32"
            ))
        })?;
        seq_lens_vec.push(ctx_len);
        running_q = running_q
            .checked_add(q_len as i32)
            .ok_or_else(|| Error::from_reason("cu_seqlens_q overflow"))?;
        cu_q_vec.push(running_q);
        max_blocks = max_blocks.max(table.num_blocks());
        tables.push(table);
    }
    let max_blocks = max_blocks.max(1);
    let mut bt_flat = vec![0i32; num_seqs * max_blocks];
    let mut per_seq_block_tables_1d = Vec::with_capacity(num_seqs);
    for (s_idx, table) in tables.iter().enumerate() {
        let row_off = s_idx * max_blocks;
        let nb = table.num_blocks().max(1);
        let mut seq_bt = vec![0i32; nb];
        for (b_idx, block) in table.blocks().iter().enumerate() {
            let bid = block.block_id as i32;
            bt_flat[row_off + b_idx] = bid;
            seq_bt[b_idx] = bid;
        }
        per_seq_block_tables_1d.push(MxArray::from_int32(&seq_bt, &[nb as i64])?);
    }
    let block_table_1d = if num_seqs == 1 {
        Some(per_seq_block_tables_1d[0].clone())
    } else {
        None
    };
    let block_table_2d = MxArray::from_int32(&bt_flat, &[num_seqs as i64, max_blocks as i64])?;
    let seq_lens = MxArray::from_int32(&seq_lens_vec, &[num_seqs as i64])?;
    let cu_seqlens_q = MxArray::from_int32(&cu_q_vec, &[(num_seqs + 1) as i64])?;
    Ok(GroupAttentionMeta {
        block_table_1d,
        per_seq_block_tables_1d,
        block_table_2d,
        seq_lens,
        cu_seqlens_q,
        ragged_slot_mapping: None,
    })
}

fn async_eval_conv_state(state: &Inkling2ShortConvState) {
    let mut arrays: Vec<&MxArray> = Vec::with_capacity(state.visits.len() * 4);
    for v in &state.visits {
        arrays.push(&v.k);
        arrays.push(&v.v);
        arrays.push(&v.attn_post);
        arrays.push(&v.mlp_post);
    }
    MxArray::async_eval_arrays(&arrays);
}

/// Queue early and periodic decode/verify visits onto Metal while the CPU builds later visits
/// (mirroring Qwen3.5 `maybe_submit_paged_decode_layer`), hiding CPU graph-build latency behind
/// GPU execution. `partial` carries data dependencies on KV writes, ShortConv updates, and blocks.
#[inline]
fn maybe_submit_paged_visit(visit_idx: usize, num_visits: usize, partial: &MxArray) {
    if visit_idx + 1 >= num_visits {
        return;
    }
    if (visit_idx < 4 || visit_idx % 8 == 3)
        && crate::engine::persistence::compiled_forward_backend_available()
    {
        MxArray::async_eval_arrays(&[partial]);
    }
}

/// Resumable 48-visit prefill unit for 100ms layer-chunked prefill interleaving.
///
/// While a long prefill is paused between visits (`ScheduledPrefillPass::Paused`), active decode
/// lanes run batched `[N, 1]` decode steps on independent sequences without stalling past the
/// 500ms decode-gap cap. If preempted mid-unit, `abort_prefill_run` restores `conv_at_start` and
/// rolls back the uncommitted KV rows `[start..end)` cleanly.
pub(crate) struct Inkling2PrefillRun {
    unit_start: usize,
    unit_end: usize,
    start: usize,
    end: usize,
    first_pos: u32,
    rows: u32,
    compute_last_logits: bool,
    cache_salt: u64,
    next_visit: usize,
    blocks: Vec<MxArray>,
    partial: MxArray,
    prelude_blocks: Vec<MxArray>,
    prelude_partial: Option<MxArray>,
    prelude_block_count: usize,
    current_bank: Option<Vec<MxArray>>,
    group_metas: Vec<GroupAttentionMeta>,
    kv_dtype: Inkling2KvDtype,
    block_size: u32,
    mpp_workspace: Option<MxArray>,
    global_tau: Option<MxArray>,
    use_mpp_prefill: bool,
    mpp_pass: Pass,
    pages_per_split: u32,
    int8_q_from: Option<u32>,
    conv_at_unit_start: Inkling2ShortConvState,
    tap_visits: Vec<usize>,
    captured_taps: Vec<MxArray>,
}

impl Inkling2PrefillRun {
    /// Approximate rows completed so far across the scheduler unit `[unit_start..unit_end)`.
    pub(crate) fn rows_done(&self, total_visits: usize) -> u32 {
        let completed_sub = u32::try_from(self.start.saturating_sub(self.unit_start)).unwrap_or(0);
        if total_visits == 0 {
            return completed_sub;
        }
        let frac =
            (self.next_visit as u64).saturating_mul(u64::from(self.rows)) / (total_visits as u64);
        completed_sub.saturating_add(u32::try_from(frac).unwrap_or(self.rows))
    }
}

/// Internal sub-slice ends within `[start..end)` so rolling checkpoints (every 4096 tokens) and
/// the prompt's deepest block junction `((prompt_len - 1) / bs) * bs` are always hit with exact
/// ShortConv + SWA state even when `MeasuredConfig` chooses arbitrary prefill cut sizes.
fn prefill_sub_boundaries(
    start: usize,
    end: usize,
    prompt_len: usize,
    block_size: u32,
) -> Vec<usize> {
    let bs = block_size.max(1);
    let mut subs = Vec::new();
    if let (Ok(prompt_u32), Ok(start_u32)) = (u32::try_from(prompt_len), u32::try_from(start)) {
        for cp in planned_rolling_checkpoints(prompt_u32, start_u32, bs) {
            let cp_us = cp as usize;
            if cp_us > start && cp_us < end {
                subs.push(cp_us);
            }
        }
        let junction = ((prompt_u32.saturating_sub(1) / bs) * bs) as usize;
        if junction > start && junction < end {
            subs.push(junction);
        }
    }
    subs.push(end);
    subs.sort_unstable();
    subs.dedup();
    subs
}

impl Inkling2Inner {
    /// Access the paged KV coordinator if initialized.
    pub fn paged_coordinator(&self) -> Option<&Inkling2KVCacheCoordinator> {
        self.paged.as_ref().map(|r| &r.coordinator)
    }

    /// Mutably access the paged KV coordinator if initialized.
    pub fn paged_coordinator_mut(&mut self) -> Option<&mut Inkling2KVCacheCoordinator> {
        self.paged.as_mut().map(|r| &mut r.coordinator)
    }

    /// Snapshot `seq_id`'s 48-visit ShortConv state (O(1) MLX array refcount clone; used by Phase 5
    /// DSpark speculative rollback).
    pub fn snapshot_paged_conv_state(&self, seq_id: SeqId) -> Option<Inkling2ShortConvState> {
        self.paged.as_ref()?.snapshot_conv_state(seq_id)
    }

    /// Restore `seq_id`'s 48-visit ShortConv state after a speculative rollback.
    pub fn restore_paged_conv_state(&mut self, seq_id: SeqId, state: Inkling2ShortConvState) {
        if let Some(ref mut rt) = self.paged {
            rt.restore_conv_state(seq_id, state);
        }
    }

    /// Reset a single paged sequence (both KV groups + 48-visit ShortConv state).
    pub fn reset_paged_sequence(&mut self, seq_id: SeqId) -> Result<()> {
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2Model: paged runtime not initialized"))?;
        rt.reset_sequence(seq_id, &self.config)
    }

    /// Evaluate pending pool writes, asynchronously schedule updated ShortConv states,
    /// and prune expired SWA blocks for `seq_id`.
    /// Returns the number of SWA physical blocks reclaimed.
    pub fn settle_paged_step(&mut self, seq_id: SeqId) -> Result<u32> {
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2Model: paged runtime not initialized"))?;
        rt.coordinator
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        if let Some(conv) = rt.conv_states.get(&seq_id) {
            async_eval_conv_state(conv);
        }
        rt.coordinator
            .prune_sliding_all(seq_id)
            .map_err(Error::from_reason)
    }

    /// Evaluate pending pool writes and prune expired SWA blocks anchored at `committed_tokens`
    /// (safe during an open speculative cycle).
    pub fn settle_paged_step_committed(
        &mut self,
        seq_id: SeqId,
        committed_tokens: u32,
    ) -> Result<u32> {
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2Model: paged runtime not initialized"))?;
        rt.coordinator
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        rt.coordinator
            .prune_sliding_all_committed(seq_id, committed_tokens)
            .map_err(Error::from_reason)
    }

    /// Abort an in-flight layer-chunked prefill run for `seq_id`, rolling back its uncommitted
    /// KV rows for the active sub-slice and restoring the ShortConv snapshot captured at that
    /// sub-slice's start.
    pub(crate) fn abort_prefill_run(&mut self, seq_id: SeqId) {
        let Some(run) = self.prefill_runs.remove(&seq_id) else {
            return;
        };
        if let Some(rt) = self.paged.as_mut() {
            let _ = rt.coordinator.eval_pending_pool_writes_all();
            let _ = rt.coordinator.rollback_last_tokens_all(seq_id, run.rows);
            rt.conv_states.insert(seq_id, run.conv_at_unit_start);
        }
    }

    /// Execute one pass (up to `budget_ms`, or to completion when `None`) of the scheduled
    /// prefill unit `source[start..end)` for `seq_id`.
    ///
    /// Returns:
    /// - `ScheduledPrefillPass::Paused` when `budget_ms` expires mid-48-visit pass so active
    ///   decode lanes can interleave batched `[N, 1]` decode steps without exceeding the 500ms gap.
    /// - `ScheduledPrefillPass::Done(logits)` once all 48 visits complete (producing `[vocab]`
    ///   logits when `end == source.len()`, else `None`), settling KV writes, pruning expired SWA
    ///   blocks, and publishing a rolling `StateClass::Checkpoint` when `end` lands on a planned
    ///   checkpoint rung.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_paged_prefill_pass(
        &mut self,
        seq_id: SeqId,
        source: &[u32],
        start: usize,
        end: usize,
        cache_salt: u64,
        generation_stream: Stream,
        budget_ms: Option<f64>,
    ) -> Result<ScheduledPrefillPass> {
        let _ctx = StreamContext::new(generation_stream);
        if start >= end || end > source.len() {
            return Err(Error::from_reason(format!(
                "Inkling2 run_paged_prefill_pass: invalid slice [{start}..{end}) for len {}",
                source.len()
            )));
        }
        let bs = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2Inner: paged runtime missing"))?
            .coordinator
            .block_size();

        // Fast path: unbounded slice with no active run executes sub-slices (splitting at any
        // internal rolling checkpoint or prompt junction so ShortConv + SWA are captured bit-exact).
        if budget_ms.is_none() && !self.prefill_runs.contains_key(&seq_id) {
            let subs = prefill_sub_boundaries(start, end, source.len(), bs);
            let tap_visits = self
                .dspark
                .as_ref()
                .and_then(|d| {
                    self.scheduled_dspark
                        .tap_visits_for(seq_id, &d.config.target_layer_ids)
                })
                .unwrap_or_default();
            let mut sub_start = start;
            let mut final_logits_opt = None;
            for sub_end in subs {
                let chunk = &source[sub_start..sub_end];
                let first_pos = {
                    let rt = self.paged.as_mut().ok_or_else(|| {
                        Error::from_reason("Inkling2Inner: paged runtime missing")
                    })?;
                    rt.ensure_conv_state(seq_id, &self.config)?;
                    let pos = rt
                        .coordinator
                        .request_token_count_all(seq_id)
                        .map_err(Error::from_reason)?;
                    rt.coordinator
                        .record_tokens_all(seq_id, chunk)
                        .map_err(Error::from_reason)?;
                    pos
                };
                let (logits_opt, taps) = self.forward_paged_single_chunk_with_taps(
                    seq_id,
                    chunk,
                    None,
                    first_pos,
                    sub_end == source.len(),
                    &tap_visits,
                )?;
                let _ = self.settle_paged_step(seq_id)?;
                if !tap_visits.is_empty() {
                    self.append_scheduled_dspark_prefill(seq_id, first_pos, &taps, chunk.len())?;
                }
                if let Some(rt) = self.paged.as_mut()
                    && let Ok(sub_end_u32) = u32::try_from(sub_end)
                {
                    rt.record_block_boundary_conv(seq_id, sub_end_u32);
                }
                self.maybe_capture_rolling_checkpoint(seq_id, source, sub_end, cache_salt)?;
                if sub_end == end {
                    final_logits_opt = logits_opt;
                }
                sub_start = sub_end;
            }
            let logits = match final_logits_opt {
                Some(l) => Some(l.squeeze(Some(&[0]))?),
                None => None,
            };
            return Ok(ScheduledPrefillPass::Done(logits));
        }

        // Resumable pass path: either resume an existing run for `[start..end)` or initialize one
        // starting at the first internal sub-boundary.
        let mut run = match self.prefill_runs.remove(&seq_id) {
            Some(existing) if existing.unit_start == start && existing.unit_end == end => existing,
            Some(stale) => {
                // Unit changed (e.g. after replay); roll back stale unit cleanly.
                self.prefill_runs.insert(seq_id, stale);
                self.abort_prefill_run(seq_id);
                let first_sub_end = prefill_sub_boundaries(start, end, source.len(), bs)[0];
                self.init_prefill_run(
                    seq_id,
                    source,
                    start,
                    end,
                    start,
                    first_sub_end,
                    cache_salt,
                    first_sub_end == source.len(),
                    None,
                )?
            }
            None => {
                let first_sub_end = prefill_sub_boundaries(start, end, source.len(), bs)[0];
                self.init_prefill_run(
                    seq_id,
                    source,
                    start,
                    end,
                    start,
                    first_sub_end,
                    cache_salt,
                    first_sub_end == source.len(),
                    None,
                )?
            }
        };

        let pass_started = Instant::now();
        let visits = self.config.visits();
        let num_visits = visits.len();
        let block_size_res = self.config.n_layers_per_attn_res_block;
        let act_dtype = DType::BFloat16;
        let lc = self.config.loop_config.clone();
        let core_len = lc.as_ref().map_or(0, |c| c.core_end - c.core_start);

        let mut visits_this_pass = 0usize;
        loop {
            let cu_seqlens_q = [0usize, run.rows as usize];
            let mode = PagedWaveMode::Single {
                seq_id,
                first_pos: run.first_pos,
                floor: run.first_pos,
                rows: run.rows,
                use_mpp_prefill: run.use_mpp_prefill,
                mpp_pass: run.mpp_pass,
                pages_per_split: run.pages_per_split,
                int8_q_from: run.int8_q_from,
            };

            // Extract `conv` ONCE before the visit loop (zero per-visit HashMap remove/insert!).
            let mut conv_state = {
                let rt = self
                    .paged
                    .as_mut()
                    .ok_or_else(|| Error::from_reason("Inkling2Inner: paged runtime missing"))?;
                rt.conv_states.remove(&seq_id).ok_or_else(|| {
                    Error::from_reason(format!(
                        "Inkling2 prefill pass: missing conv state for seq {seq_id}"
                    ))
                })?
            };

            let mut loop_outcome: Result<bool> = Ok(false); // Ok(true) => paused
            while run.next_visit < num_visits {
                if self
                    .turn_cancel
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::Relaxed))
                {
                    loop_outcome = Err(Error::from_reason("prefill cancelled"));
                    break;
                }

                let visit_idx = run.next_visit;
                let v_meta = visits[visit_idx];

                // Handle Prelude -> Core pass transitions & cross-pass bank resets
                // identically to `forward_paged_wave_inner` (zero normalize/stack dispatches!).
                if let Some(ref lc_ref) = lc {
                    if visit_idx == lc_ref.core_start && run.prelude_partial.is_none() {
                        run.prelude_blocks = run.blocks.clone();
                        run.prelude_partial = Some(run.partial.clone());
                        run.prelude_block_count = run.prelude_blocks.len();
                    } else if v_meta.pass_idx > 0
                        && core_len > 0
                        && visit_idx == lc_ref.core_start + v_meta.pass_idx * core_len
                    {
                        let mut bank = Vec::with_capacity(
                            run.blocks.len().saturating_sub(run.prelude_block_count + 1) + 1,
                        );
                        bank.extend_from_slice(&run.blocks[run.prelude_block_count + 1..]);
                        bank.push(run.partial.clone());
                        run.blocks = run.prelude_blocks.clone();
                        if let Some(ref pp) = run.prelude_partial {
                            run.partial = pp.clone();
                        }
                        run.current_bank = Some(bank);
                    } else if visit_idx == lc_ref.core_start + lc_ref.num_passes * core_len {
                        // Entering Coda: clear cross-pass bank.
                        run.current_bank = None;
                    }
                }

                // Target hidden state taps (`[5, 23, 41]`): capture AFTER any pass restart and
                // BEFORE executing the layer at `visit_idx`!
                if run.tap_visits.contains(&visit_idx) {
                    run.captured_taps.push(run.partial.clone());
                }

                let layer = &self.layers[v_meta.phys_layer];

                let step_res = {
                    let rt = self
                        .paged
                        .as_mut()
                        .ok_or_else(|| Error::from_reason("Inkling2Inner: paged runtime missing"));
                    match rt {
                        Err(e) => Err(e),
                        Ok(rt) => {
                            let route_res = rt
                                .coordinator
                                .route_for_visit(visit_idx)
                                .ok_or_else(|| {
                                    Error::from_reason(format!(
                                        "Inkling2 prefill pass: missing KV route for visit {visit_idx}"
                                    ))
                                })
                                .cloned();
                            match route_res {
                                Err(e) => Err(e),
                                Ok(route) => match rt
                                    .coordinator
                                    .adapter_mut(route.group_id)
                                    .map_err(Error::from_reason)
                                {
                                    Err(e) => Err(e),
                                    Ok(adapter) => {
                                        let group_meta = &run.group_metas[route.group_id];
                                        let mut visit_convs = [&mut conv_state.visits[visit_idx]];
                                        forward_physical_layer_paged(
                                            layer,
                                            &mut run.blocks,
                                            &mut run.partial,
                                            block_size_res,
                                            route.physical_layer_ordinal as u32,
                                            adapter,
                                            group_meta,
                                            run.kv_dtype,
                                            run.block_size,
                                            &mode,
                                            &cu_seqlens_q,
                                            run.global_tau.as_ref(),
                                            &mut visit_convs,
                                            None,
                                            run.current_bank.as_deref(),
                                            run.mpp_workspace.as_mut(),
                                            act_dtype,
                                        )
                                    }
                                },
                            }
                        }
                    }
                };

                if let Err(err) = step_res {
                    loop_outcome = Err(err);
                    break;
                }

                run.next_visit += 1;
                visits_this_pass += 1;

                // Check budget every 4 visits after evaluating the graph so wall-clock timing
                // reflects actual Metal execution.
                if let Some(budget) = budget_ms
                    && run.next_visit < num_visits
                    && visits_this_pass >= 4
                    && run.next_visit.is_multiple_of(4)
                {
                    let mut to_eval: Vec<&MxArray> = Vec::with_capacity(run.blocks.len() + 4);
                    for b in &run.blocks {
                        to_eval.push(b);
                    }
                    if let Some(ref bank) = run.current_bank {
                        for b in bank {
                            to_eval.push(b);
                        }
                    }
                    to_eval.push(&run.partial);
                    if let Err(err) = MxArray::eval_arrays(&to_eval) {
                        loop_outcome = Err(err);
                        break;
                    }
                    if let Some(rt) = self.paged.as_mut()
                        && let Err(err) = rt
                            .coordinator
                            .eval_pending_pool_writes_all()
                            .map_err(Error::from_reason)
                    {
                        loop_outcome = Err(err);
                        break;
                    }
                    if pass_started.elapsed().as_secs_f64() * 1000.0 >= budget {
                        loop_outcome = Ok(true);
                        break;
                    }
                }
            }

            // Always restore `conv_state` before handling pause / error / completion!
            if let Some(rt) = self.paged.as_mut() {
                rt.conv_states.insert(seq_id, conv_state);
            }
            match loop_outcome {
                Err(err) => {
                    self.prefill_runs.insert(seq_id, run);
                    self.abort_prefill_run(seq_id);
                    return Err(err);
                }
                Ok(true) => {
                    self.prefill_runs.insert(seq_id, run);
                    return Ok(ScheduledPrefillPass::Paused);
                }
                Ok(false) => {}
            }

            // All 48 visits finished for `[run.start..run.end)`: settle writes, capture final
            // tap (`visit == 48`) + feed scheduled DSpark context, record exact ShortConv boundary,
            // and capture checkpoint / prompt junction!
            let capture_final_norm = run.tap_visits.contains(&num_visits);
            let post_res: Result<Option<MxArray>> = (|| {
                let (logits_opt, final_norm_opt) = self.compute_final_norm_and_logits(
                    &run.blocks,
                    &run.partial,
                    run.rows as usize,
                    /* last_only */ true,
                    capture_final_norm,
                    run.compute_last_logits,
                    act_dtype,
                )?;
                if let Some(norm_all) = final_norm_opt {
                    run.captured_taps.push(norm_all);
                }
                let logits = match logits_opt {
                    Some(l) => Some(l.squeeze(Some(&[0]))?),
                    None => {
                        run.partial.eval();
                        None
                    }
                };
                let _ = self.settle_paged_step(seq_id)?;
                Ok(logits)
            })();
            let logits = match post_res {
                Ok(l) => l,
                Err(err) => {
                    self.prefill_runs.insert(seq_id, run);
                    self.abort_prefill_run(seq_id);
                    return Err(err);
                }
            };
            if !run.tap_visits.is_empty() {
                self.append_scheduled_dspark_prefill(
                    seq_id,
                    run.first_pos,
                    &run.captured_taps,
                    run.rows as usize,
                )?;
            }
            if let Some(rt) = self.paged.as_mut()
                && let Ok(sub_end_u32) = u32::try_from(run.end)
            {
                rt.record_block_boundary_conv(seq_id, sub_end_u32);
            }
            self.maybe_capture_rolling_checkpoint(seq_id, source, run.end, run.cache_salt)?;

            if run.end >= run.unit_end {
                return Ok(ScheduledPrefillPass::Done(logits));
            }
            // Advance to next internal sub-slice (e.g. tiny `[junction..end)` tail).
            // Sub-slice 1 is already settled (KV pruned, DSpark fed, boundary conv recorded),
            // so pass `None` to capture the exact post-sub-slice ShortConv snapshot for rollback!
            let next_start = run.end;
            let next_end = prefill_sub_boundaries(next_start, run.unit_end, source.len(), bs)[0];
            run = self.init_prefill_run(
                seq_id,
                source,
                run.unit_start,
                run.unit_end,
                next_start,
                next_end,
                run.cache_salt,
                next_end == source.len(),
                None,
            )?;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn init_prefill_run(
        &mut self,
        seq_id: SeqId,
        source: &[u32],
        unit_start: usize,
        unit_end: usize,
        start: usize,
        end: usize,
        cache_salt: u64,
        compute_last_logits: bool,
        preserved_unit_conv: Option<Inkling2ShortConvState>,
    ) -> Result<Inkling2PrefillRun> {
        let chunk = &source[start..end];
        let rows = u32::try_from(chunk.len())
            .map_err(|_| Error::from_reason("Inkling2 prefill slice length overflows u32"))?;
        let act_dtype = DType::BFloat16;

        let (first_pos, conv_at_unit_start) = {
            let rt = self
                .paged
                .as_mut()
                .ok_or_else(|| Error::from_reason("Inkling2Inner: paged runtime missing"))?;
            let conv_snap = match preserved_unit_conv {
                Some(prev) => {
                    let _ = rt.ensure_conv_state(seq_id, &self.config)?;
                    prev
                }
                None => rt.ensure_conv_state(seq_id, &self.config)?.clone(),
            };
            let pos = rt
                .coordinator
                .request_token_count_all(seq_id)
                .map_err(Error::from_reason)?;
            rt.coordinator
                .record_tokens_all(seq_id, chunk)
                .map_err(Error::from_reason)?;
            (pos, conv_snap)
        };

        let ids_i32: Vec<i32> = chunk.iter().map(|&id| id as i32).collect();
        let ids_arr = MxArray::from_int32(&ids_i32, &[rows as i64])?;
        let raw_emb = self.embed_tokens.forward(&ids_arr)?;
        let partial = apply_sublayer_rms_norm(
            &raw_emb,
            &self.embed_norm,
            self.config.rms_norm_eps as f32,
            act_dtype,
        )?;

        let use_mpp_prefill = rows > 1 && inkling2_paged_prefill_mpp_supported();
        let mpp_pass = if rows <= prefill_plan::VERIFY_ROWS {
            Pass::Verify
        } else {
            Pass::Prefill
        };
        let pages_per_split = device_pages_per_split();
        let int8_q_from = if self.activations == crate::transformer::Activations::Bf16 {
            None
        } else {
            device_int8_query_from()
        };

        let seq_q_lens = [(seq_id, rows)];
        let (group_metas, kv_dtype, block_size, mpp_workspace) = {
            let rt = self
                .paged
                .as_ref()
                .ok_or_else(|| Error::from_reason("Inkling2Inner: paged runtime missing"))?;
            let num_groups = rt.coordinator.groups().len();
            let mut metas = Vec::with_capacity(num_groups);
            for gid in 0..num_groups {
                let adapter = rt.coordinator.adapter(gid).map_err(Error::from_reason)?;
                metas.push(build_group_attention_meta(adapter, &seq_q_lens)?);
            }
            let ws = if use_mpp_prefill {
                let splits_needed =
                    prefill_plan::splits(mpp_pass, rows, first_pos, pages_per_split);
                let floats = prefill_plan::workspace_floats(
                    rows,
                    splits_needed,
                    self.config.num_attention_heads as u32,
                    self.config.head_dim as u32,
                );
                Some(MxArray::zeros(&[floats as i64], Some(DType::Float32))?)
            } else {
                None
            };
            (
                metas,
                rt.coordinator.kv_dtype(),
                rt.coordinator.block_size(),
                ws,
            )
        };

        let positions: Vec<u32> = (0..rows).map(|i| first_pos + i).collect();
        let global_tau = global_log_scale_tau_for_positions(
            &positions,
            self.config.log_scaling_n_floor,
            self.config.log_scaling_alpha,
        )?;

        let tap_visits = self
            .dspark
            .as_ref()
            .and_then(|d| {
                self.scheduled_dspark
                    .tap_visits_for(seq_id, &d.config.target_layer_ids)
            })
            .unwrap_or_default();

        Ok(Inkling2PrefillRun {
            unit_start,
            unit_end,
            start,
            end,
            first_pos,
            rows,
            compute_last_logits,
            cache_salt,
            next_visit: 0,
            blocks: Vec::new(),
            partial,
            prelude_blocks: Vec::new(),
            prelude_partial: None,
            prelude_block_count: 0,
            current_bank: None,
            group_metas,
            kv_dtype,
            block_size,
            mpp_workspace,
            global_tau,
            use_mpp_prefill,
            mpp_pass,
            pages_per_split,
            int8_q_from,
            conv_at_unit_start,
            captured_taps: Vec::with_capacity(tap_visits.len()),
            tap_visits,
        })
    }

    fn maybe_capture_rolling_checkpoint(
        &mut self,
        seq_id: SeqId,
        source: &[u32],
        end: usize,
        cache_salt: u64,
    ) -> Result<()> {
        let Some(rt) = self.paged.as_mut() else {
            return Ok(());
        };
        let bs = rt.coordinator.block_size().max(1);
        let Ok(end_u32) = u32::try_from(end) else {
            return Ok(());
        };
        let Ok(prompt_u32) = u32::try_from(source.len()) else {
            return Ok(());
        };
        if end_u32 >= prompt_u32 || !end_u32.is_multiple_of(bs) {
            return Ok(());
        }
        let junction = (prompt_u32.saturating_sub(1) / bs) * bs;
        let class = if junction > 0 && end_u32 == junction {
            StateClass::Ordinary
        } else if planned_rolling_checkpoints(prompt_u32, 0, bs).contains(&end_u32) {
            StateClass::Checkpoint
        } else {
            return Ok(());
        };
        let Some(conv) = rt.conv_states.get(&seq_id).cloned() else {
            return Ok(());
        };
        let _ = rt.state_store.capture_boundary(
            &mut rt.coordinator,
            &conv,
            seq_id,
            source,
            end_u32,
            cache_salt,
            class,
        )?;
        Ok(())
    }

    /// Compute final output AttnRes -> `output_norm` -> optional `lm_head` + softcap.
    ///
    /// When `capture_final_norm` is true (for DSpark tap `visit == 48`), computes `normed_out`
    /// over all `t` rows (`[t, D]` BF16) and returns it as the second tuple element, then slices
    /// only the last row `[t-1..t, D]` for `lm_head` when `last_only && t > 1` so we never
    /// allocate `[2048, 201024]` logits during prefill!
    #[allow(clippy::too_many_arguments)]
    fn compute_final_norm_and_logits(
        &self,
        blocks: &[MxArray],
        partial: &MxArray,
        t: usize,
        last_only: bool,
        capture_final_norm: bool,
        compute_logits: bool,
        act_dtype: DType,
    ) -> Result<(Option<MxArray>, Option<MxArray>)> {
        if !compute_logits && !capture_final_norm {
            return Ok((None, None));
        }
        let d = self.config.hidden_size as i64;
        let eps = self.config.rms_norm_eps as f32;
        let slice_before_norm = !capture_final_norm && last_only && t > 1;

        let (norm_rows, normed_out): (i64, MxArray) = if slice_before_norm {
            let last = (t - 1) as i64;
            let mut sliced = Vec::with_capacity(blocks.len() + 1);
            for b in blocks {
                sliced.push(b.slice(&[last, 0], &[t as i64, d])?);
            }
            sliced.push(partial.slice(&[last, 0], &[t as i64, d])?);
            let refs: Vec<&MxArray> = sliced.iter().collect();
            let normed = inkling2_block_attn_res(
                &refs,
                None,
                &self.output_res_query,
                Some(&self.output_norm),
                ATTN_RES_EPS,
                eps,
                true,
                act_dtype,
            )?;
            (1, normed)
        } else {
            let mut refs: Vec<&MxArray> = Vec::with_capacity(blocks.len() + 1);
            for b in blocks {
                refs.push(b);
            }
            refs.push(partial);
            let normed = inkling2_block_attn_res(
                &refs,
                None,
                &self.output_res_query,
                Some(&self.output_norm),
                ATTN_RES_EPS,
                eps,
                true,
                act_dtype,
            )?;
            (t as i64, normed)
        };
        let captured_norm = capture_final_norm.then(|| normed_out.clone());

        if !compute_logits {
            return Ok((None, captured_norm));
        }

        let lm_in = if !slice_before_norm && last_only && t > 1 {
            let last = (t - 1) as i64;
            normed_out.slice(&[last, 0], &[t as i64, d])?
        } else {
            normed_out
        };
        let _ = norm_rows;

        // Keep FP32 logits directly when `lm_head` already outputs FP32, then fuse vocab slice +
        // `30.0 * tanh(x / 30.0)` softcap in 1 Metal dispatch!
        let logits_full = self.lm_head.forward_logits_keep_f32(&lm_in)?;
        let unpadded = self.config.effective_unpadded_vocab_size();
        let vocab_cols = logits_full.shape()?[1] as usize;
        let out_cols = unpadded.min(vocab_cols);
        let cap = self.config.final_logit_softcapping as f32;
        let logits = inkling2_lm_head_softcap(&logits_full, None, out_cols, cap, false)?;
        Ok((Some(logits), captured_norm))
    }

    /// Run chunked in-place paged prefill (up to `prefill_chunk_size = 2048` rows/chunk) for
    /// `seq_id`, pruning expired SWA blocks after every chunk.
    ///
    /// Returns `(last_token_logits [1, unpadded_vocab_size], total_swa_blocks_pruned)`.
    pub fn paged_prefill_sequence(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        image_embeddings: Option<&MxArray>,
    ) -> Result<(MxArray, u32)> {
        self.paged_prefill_sequence_with_taps(seq_id, tokens, image_embeddings, &[], |_, _, _| {
            Ok(())
        })
    }

    /// Chunked in-place paged prefill with optional target hidden state tap capture (`[5, 23, 41, 48]`)
    /// delivered per chunk to `on_chunk_taps` (used by DSpark speculative decoding).
    pub(crate) fn paged_prefill_sequence_with_taps(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        image_embeddings: Option<&MxArray>,
        tap_visit_indices: &[usize],
        mut on_chunk_taps: impl FnMut(&Inkling2DsparkModel, &[MxArray], usize) -> Result<()>,
    ) -> Result<(MxArray, u32)> {
        if tokens.is_empty() {
            return Err(Error::from_reason(
                "Inkling2Model::paged_prefill_sequence: empty tokens",
            ));
        }
        let chunk_limit = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2Model: paged runtime not initialized"))?
            .prefill_chunk_size;

        // Validate total IMAGE_PAD count up front if image embeddings are supplied.
        let mut img_cursor = 0usize;
        if let Some(img_emb) = image_embeddings {
            let total_pads = tokens.iter().filter(|&&tok| tok == IMAGE_PAD_ID).count();
            let img_rows = img_emb.shape()?[0] as usize;
            if total_pads != img_rows {
                return Err(Error::from_reason(format!(
                    "Inkling2Model::paged_prefill_sequence: prompt has {total_pads} IMAGE_PAD tokens \
                     but image_embeddings has {img_rows} rows"
                )));
            }
        }

        let total_len = tokens.len();
        let mut offset = 0usize;
        let mut total_pruned = 0u32;
        let mut final_logits = None;

        while offset < total_len {
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason("prefill cancelled"));
            }
            let end = (offset + chunk_limit).min(total_len);
            let chunk = &tokens[offset..end];
            let is_final_chunk = end == total_len;

            // Slice image embeddings belonging to this chunk (if any).
            let chunk_img = if let Some(img_emb) = image_embeddings {
                let pad_count = chunk.iter().filter(|&&tok| tok == IMAGE_PAD_ID).count();
                if pad_count > 0 {
                    let d = img_emb.shape()?[1];
                    let slice = img_emb.slice(
                        &[img_cursor as i64, 0],
                        &[(img_cursor + pad_count) as i64, d],
                    )?;
                    img_cursor += pad_count;
                    Some(slice)
                } else {
                    None
                }
            } else {
                None
            };

            // Current sequence cursor BEFORE recording this chunk.
            let first_pos = {
                let rt = self
                    .paged
                    .as_mut()
                    .ok_or_else(|| Error::from_reason("Inkling2Model: paged runtime missing"))?;
                rt.ensure_conv_state(seq_id, &self.config)?;
                let pos = rt
                    .coordinator
                    .request_token_count_all(seq_id)
                    .map_err(Error::from_reason)?;
                rt.coordinator
                    .record_tokens_all(seq_id, chunk)
                    .map_err(Error::from_reason)?;
                pos
            };

            let (logits_opt, taps) = self.forward_paged_single_chunk_with_taps(
                seq_id,
                chunk,
                chunk_img.as_ref(),
                first_pos,
                /* compute_last_logits */ is_final_chunk,
                tap_visit_indices,
            )?;

            // Materialize pending pool writes + ShortConv state and prune expired SWA blocks
            // before launching the next chunk (keeps SWA pool usage bounded to O(window + chunk)).
            let pruned = self.settle_paged_step(seq_id)?;
            total_pruned = total_pruned.saturating_add(pruned);
            if !tap_visit_indices.is_empty()
                && let Some(ref draft) = self.dspark
            {
                on_chunk_taps(draft, &taps, chunk.len())?;
            }
            if let Some(rt) = self.paged.as_mut()
                && let Ok(pos_after) = u32::try_from(first_pos as usize + chunk.len())
            {
                rt.record_block_boundary_conv(seq_id, pos_after);
            }

            if is_final_chunk {
                final_logits = logits_opt;
            } else {
                crate::array::clear_cache();
            }
            offset = end;
        }

        let logits = final_logits.ok_or_else(|| {
            Error::from_reason("Inkling2Model::paged_prefill_sequence produced no logits")
        })?;
        Ok((logits, total_pruned))
    }

    /// Run one single-sequence paged decode step (`[1, 1]`) for `seq_id`.
    /// Records `token_id` atomically across both KV groups and returns `[1, unpadded_vocab_size]`
    /// FP32 softcapped logits.
    pub fn paged_decode_step(&mut self, seq_id: SeqId, token_id: u32) -> Result<MxArray> {
        let first_pos = {
            let rt = self.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2Model: paged runtime not initialized")
            })?;
            rt.ensure_conv_state(seq_id, &self.config)?;
            let pos = rt
                .coordinator
                .request_token_count_all(seq_id)
                .map_err(Error::from_reason)?;
            // Save pre-step ShortConv snapshot so early-stop reconciliation can roll back cleanly.
            if let Some(snap) = rt.conv_states.get(&seq_id).cloned() {
                rt.prev_conv_snapshots.insert(seq_id, (pos, snap));
            }
            rt.coordinator
                .record_tokens_all(seq_id, &[token_id])
                .map_err(Error::from_reason)?;
            pos
        };
        let logits =
            match self.forward_paged_single_chunk(seq_id, &[token_id], None, first_pos, true) {
                Ok(Some(l)) => l,
                Ok(None) => {
                    if let Some(rt) = self.paged.as_mut() {
                        let _ = rt.coordinator.rollback_last_tokens_all(seq_id, 1);
                        if let Some((snap_pos, snap)) = rt.prev_conv_snapshots.remove(&seq_id)
                            && snap_pos == first_pos
                        {
                            rt.conv_states.insert(seq_id, snap);
                        }
                    }
                    return Err(Error::from_reason(
                        "Inkling2Model::paged_decode_step produced no logits",
                    ));
                }
                Err(err) => {
                    if let Some(rt) = self.paged.as_mut() {
                        let _ = rt.coordinator.rollback_last_tokens_all(seq_id, 1);
                        if let Some((snap_pos, snap)) = rt.prev_conv_snapshots.remove(&seq_id)
                            && snap_pos == first_pos
                        {
                            rt.conv_states.insert(seq_id, snap);
                        }
                    }
                    return Err(err);
                }
            };
        if let Some(rt) = self.paged.as_mut() {
            rt.record_block_boundary_conv(seq_id, first_pos.saturating_add(1));
        }
        Ok(logits)
    }

    /// Run one uniform batched `[N, 1]` paged decode wave across `rows = [(seq_id, token_id)]`.
    ///
    /// Atomically checks admission and records one token per sequence across both Global & SWA
    /// groups (rolling back on error), updates each sequence's ShortConv state independently,
    /// and returns `[N, unpadded_vocab_size]` FP32 softcapped logits.
    pub fn paged_decode_step_batched(&mut self, rows: &[(SeqId, u32)]) -> Result<MxArray> {
        if rows.is_empty() {
            return Err(Error::from_reason(
                "Inkling2Model::paged_decode_step_batched requires at least one row",
            ));
        }
        if let &[(seq_id, token_id)] = rows {
            return self.paged_decode_step(seq_id, token_id);
        }

        // 1. Validate uniqueness, gather current positions, check admission across both groups,
        //    and record tokens atomically.
        let planned_rows = {
            let rt = self.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2Model: paged runtime not initialized")
            })?;
            let mut seen = HashSet::with_capacity(rows.len());
            let mut planned = Vec::with_capacity(rows.len());
            for &(seq_id, _) in rows {
                if !seen.insert(seq_id) {
                    return Err(Error::from_reason(format!(
                        "paged_decode_step_batched: duplicate sequence {seq_id}"
                    )));
                }
                rt.ensure_conv_state(seq_id, &self.config)?;
                let pos = rt
                    .coordinator
                    .request_token_count_all(seq_id)
                    .map_err(Error::from_reason)?;
                if let Some(snap) = rt.conv_states.get(&seq_id).cloned() {
                    rt.prev_conv_snapshots.insert(seq_id, (pos, snap));
                }
                planned.push((seq_id, pos));
            }
            let wave: Vec<(SeqId, u32)> = rows.iter().map(|&(seq_id, _)| (seq_id, 1)).collect();
            rt.coordinator
                .check_wave_fits_all(&wave)
                .map_err(|e| Error::from_reason(format!("Inkling2 batched record: {e}")))?;

            let mut recorded = Vec::with_capacity(rows.len());
            for &(seq_id, token_id) in rows {
                if let Err(error) = rt.coordinator.record_tokens_all(seq_id, &[token_id]) {
                    for &prev_seq in recorded.iter().rev() {
                        let _ = rt.coordinator.rollback_last_tokens_all(prev_seq, 1);
                    }
                    return Err(Error::from_reason(format!(
                        "Inkling2 batched record failed for seq {seq_id}: {error}"
                    )));
                }
                recorded.push(seq_id);
            }
            planned
        };

        let token_ids: Vec<u32> = rows.iter().map(|&(_, tok)| tok).collect();
        let seq_ids: Vec<SeqId> = planned_rows.iter().map(|&(sid, _)| sid).collect();
        let positions: Vec<u32> = planned_rows.iter().map(|&(_, pos)| pos).collect();
        let cu_seqlens_q: Vec<usize> = (0..=rows.len()).collect();
        let seq_q_lens: Vec<(SeqId, u32)> = seq_ids.iter().map(|&sid| (sid, 1)).collect();

        let (logits, _) = match self.forward_paged_wave(
            &token_ids,
            None,
            &seq_ids,
            &positions,
            &cu_seqlens_q,
            &seq_q_lens,
            PagedWaveMode::BatchedDecode {
                planned_rows: &planned_rows,
            },
            /* last_only */ false,
            &[],
        ) {
            Ok(res) => res,
            Err(err) => {
                if let Some(rt) = self.paged.as_mut() {
                    for &(seq_id, pos) in &planned_rows {
                        let _ = rt.coordinator.rollback_last_tokens_all(seq_id, 1);
                        if let Some((snap_pos, snap)) = rt.prev_conv_snapshots.remove(&seq_id)
                            && snap_pos == pos
                        {
                            rt.conv_states.insert(seq_id, snap);
                        }
                    }
                }
                return Err(err);
            }
        };
        if let Some(rt) = self.paged.as_mut() {
            for &(seq_id, pos) in &planned_rows {
                rt.record_block_boundary_conv(seq_id, pos.saturating_add(1));
            }
        }
        Ok(logits)
    }

    /// Run one multi-row varlen verify forward (`[N * K]` rows, for Phase 5 DSpark).
    #[allow(dead_code)]
    pub(crate) fn paged_verify_ragged(
        &mut self,
        rows: &[(PagedRaggedRow, Vec<u32>)],
        tap_visit_indices: &[usize],
    ) -> Result<(MxArray, Vec<MxArray>)> {
        let (logits, taps, _tapes) = self.paged_verify_ragged_taped(rows, tap_visit_indices)?;
        Ok((logits, taps))
    }

    /// Run one multi-row varlen verify forward (`[N * K]` rows) WITH per-sequence 48-visit
    /// [`Inkling2ShortConvTape`] rollback recording and exact `[5, 23, 41, 48]` target taps.
    ///
    /// For `1..=4` lanes of `<= 8` tokens each, stacks `[N * 8, 2304]` rows across every dense
    /// Q8_0 and NVFP4/Q8_0 MoE projection while running ShortConv + `inkling2_paged_prefill_attention`
    /// (`Pass::Verify`) per lane so every lane is bit-exact to solo execution!
    pub(crate) fn paged_verify_ragged_taped(
        &mut self,
        rows: &[(PagedRaggedRow, Vec<u32>)],
        tap_visit_indices: &[usize],
    ) -> Result<(MxArray, Vec<MxArray>, Vec<Inkling2ShortConvTape>)> {
        if rows.is_empty() {
            return Err(Error::from_reason(
                "Inkling2Model::paged_verify_ragged_taped requires at least one row",
            ));
        }
        let mut token_ids = Vec::new();
        let mut positions = Vec::new();
        let mut seq_ids = Vec::with_capacity(rows.len());
        let mut seq_q_lens = Vec::with_capacity(rows.len());
        let mut cu_seqlens_q = Vec::with_capacity(rows.len() + 1);
        let mut ragged_rows = Vec::with_capacity(rows.len());
        cu_seqlens_q.push(0usize);

        let mut seen = HashSet::with_capacity(rows.len());
        for (row, tokens) in rows {
            if !seen.insert(row.seq_id) {
                return Err(Error::from_reason(format!(
                    "Inkling2Model::paged_verify_ragged_taped: duplicate seq_id {}",
                    row.seq_id
                )));
            }
            if tokens.is_empty() || tokens.len() != row.query_len as usize {
                return Err(Error::from_reason(
                    "Inkling2Model::paged_verify_ragged_taped: row token length mismatch",
                ));
            }
            ragged_rows.push(*row);
            seq_ids.push(row.seq_id);
            seq_q_lens.push((row.seq_id, row.query_len));
            token_ids.extend_from_slice(tokens);
            for offset in 0..row.query_len {
                positions.push(
                    row.first_logical_position
                        .checked_add(offset)
                        .ok_or_else(|| Error::from_reason("verify position overflow"))?,
                );
            }
            cu_seqlens_q.push(token_ids.len());
        }

        {
            let rt = self.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2Model: paged runtime not initialized")
            })?;
            for &sid in &seq_ids {
                rt.ensure_conv_state(sid, &self.config)?;
            }
        }

        // Enable per-lane MPP verify on long-context Global layers (`first_pos >= 2048`) where
        // split-K + Apple10 `q8q` win; all 40 SWA layers (window 1024 <= 32 pages) and short
        // Global layers run one-shot varlen `inkling2_paged_attention` (faster & AR-identical).
        let use_mpp_verify = ragged_rows.iter().all(|r| {
            r.query_len > 1
                && r.query_len <= prefill_plan::VERIFY_ROWS
                && r.first_logical_position >= 2048
        }) && inkling2_paged_prefill_mpp_supported();
        let pages_per_split = device_pages_per_split();
        let int8_q_from = if self.activations == crate::transformer::Activations::Bf16 {
            None
        } else {
            device_int8_query_from()
        };

        let num_visits = self.config.num_layer_visits();
        let mut tapes: Vec<Inkling2ShortConvTape> = (0..seq_ids.len())
            .map(|_| Inkling2ShortConvTape::new(num_visits))
            .collect();

        let (logits_opt, taps) = self.forward_paged_wave_inner(
            &token_ids,
            None,
            &seq_ids,
            &positions,
            &cu_seqlens_q,
            &seq_q_lens,
            PagedWaveMode::RaggedVerify {
                ragged_rows: &ragged_rows,
                use_mpp_verify,
                pages_per_split,
                int8_q_from,
            },
            /* compute_logits */ true,
            /* last_only */ false,
            tap_visit_indices,
            Some(&mut tapes),
        )?;
        let logits = logits_opt
            .ok_or_else(|| Error::from_reason("paged_verify_ragged_taped produced no logits"))?;
        Ok((logits, taps, tapes))
    }

    /// Greedy autoregressive generation over the paged path (for parity verification & benchmarking).
    pub fn generate_greedy_paged(
        &mut self,
        prompt_ids: &[u32],
        image_embeddings: Option<&MxArray>,
        max_new_tokens: usize,
        stop_token_ids: &[u32],
    ) -> Result<(Vec<u32>, Inkling2PagedPerf)> {
        let seq_id = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2Model: paged runtime not initialized"))?
            .active_seq;
        self.reset_paged_sequence(seq_id)?;

        let t_prefill = Instant::now();
        let (prefill_logits, mut total_pruned) =
            self.paged_prefill_sequence(seq_id, prompt_ids, image_embeddings)?;
        let mut next_tok = prefill_logits.argmax(-1, Some(false))?.item_at_int32(0)? as u32;
        let prefill_secs = t_prefill.elapsed().as_secs_f64();

        let mut generated = Vec::with_capacity(max_new_tokens);
        let t_decode = Instant::now();
        let mut decode_steps = 0usize;
        for step in 0..max_new_tokens {
            generated.push(next_tok);
            if stop_token_ids.contains(&next_tok) || step + 1 == max_new_tokens {
                break;
            }
            let step_logits = self.paged_decode_step(seq_id, next_tok)?;
            next_tok = step_logits.argmax(-1, Some(false))?.item_at_int32(0)? as u32;
            let pruned = self.settle_paged_step(seq_id)?;
            total_pruned = total_pruned.saturating_add(pruned);
            decode_steps += 1;
            crate::array::maybe_clear_cache_for_paged_step(step as i32);
        }
        let decode_secs = t_decode.elapsed().as_secs_f64();
        let perf = Inkling2PagedPerf {
            prompt_tokens: prompt_ids.len(),
            generated_tokens: generated.len(),
            prefill_secs,
            decode_secs,
            prefill_tok_per_sec: if prefill_secs > 0.0 {
                prompt_ids.len() as f64 / prefill_secs
            } else {
                0.0
            },
            decode_tok_per_sec: if decode_secs > 0.0 && decode_steps > 0 {
                decode_steps as f64 / decode_secs
            } else {
                0.0
            },
            swa_blocks_pruned: total_pruned,
        };
        Ok((generated, perf))
    }

    fn forward_paged_single_chunk(
        &mut self,
        seq_id: SeqId,
        chunk: &[u32],
        chunk_img: Option<&MxArray>,
        first_pos: u32,
        compute_last_logits: bool,
    ) -> Result<Option<MxArray>> {
        let (logits_opt, _) = self.forward_paged_single_chunk_with_taps(
            seq_id,
            chunk,
            chunk_img,
            first_pos,
            compute_last_logits,
            &[],
        )?;
        Ok(logits_opt)
    }

    pub(crate) fn forward_paged_single_chunk_with_taps(
        &mut self,
        seq_id: SeqId,
        chunk: &[u32],
        chunk_img: Option<&MxArray>,
        first_pos: u32,
        compute_last_logits: bool,
        tap_visit_indices: &[usize],
    ) -> Result<(Option<MxArray>, Vec<MxArray>)> {
        let rows = u32::try_from(chunk.len())
            .map_err(|_| Error::from_reason("Inkling2 chunk length overflows u32"))?;
        let positions: Vec<u32> = (0..rows).map(|i| first_pos + i).collect();
        let cu_seqlens_q = [0usize, chunk.len()];
        let seq_ids = [seq_id];
        let seq_q_lens = [(seq_id, rows)];

        let use_mpp_prefill = rows > 1 && inkling2_paged_prefill_mpp_supported();
        let mpp_pass = if rows <= prefill_plan::VERIFY_ROWS {
            Pass::Verify
        } else {
            Pass::Prefill
        };
        let pages_per_split = device_pages_per_split();
        let int8_q_from = if self.activations == crate::transformer::Activations::Bf16 {
            None
        } else {
            device_int8_query_from()
        };

        self.forward_paged_wave_inner(
            chunk,
            chunk_img,
            &seq_ids,
            &positions,
            &cu_seqlens_q,
            &seq_q_lens,
            PagedWaveMode::Single {
                seq_id,
                first_pos,
                floor: first_pos,
                rows,
                use_mpp_prefill,
                mpp_pass,
                pages_per_split,
                int8_q_from,
            },
            compute_last_logits,
            /* last_only */ true,
            tap_visit_indices,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_paged_wave(
        &mut self,
        token_ids: &[u32],
        image_embeddings: Option<&MxArray>,
        seq_ids: &[SeqId],
        positions: &[u32],
        cu_seqlens_q: &[usize],
        seq_q_lens: &[(SeqId, u32)],
        mode: PagedWaveMode<'_>,
        last_only: bool,
        tap_visit_indices: &[usize],
    ) -> Result<(MxArray, Vec<MxArray>)> {
        let (logits_opt, taps) = self.forward_paged_wave_inner(
            token_ids,
            image_embeddings,
            seq_ids,
            positions,
            cu_seqlens_q,
            seq_q_lens,
            mode,
            /* compute_logits */ true,
            last_only,
            tap_visit_indices,
            None,
        )?;
        let logits = logits_opt
            .ok_or_else(|| Error::from_reason("forward_paged_wave produced no logits"))?;
        Ok((logits, taps))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_paged_wave_inner(
        &mut self,
        token_ids: &[u32],
        image_embeddings: Option<&MxArray>,
        seq_ids: &[SeqId],
        positions: &[u32],
        cu_seqlens_q: &[usize],
        seq_q_lens: &[(SeqId, u32)],
        mode: PagedWaveMode<'_>,
        compute_logits: bool,
        last_only: bool,
        tap_visit_indices: &[usize],
        mut conv_tapes: Option<&mut [Inkling2ShortConvTape]>,
    ) -> Result<(Option<MxArray>, Vec<MxArray>)> {
        let act_dtype = DType::BFloat16;
        let t = token_ids.len();
        let ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let ids_arr = MxArray::from_int32(&ids_i32, &[t as i64])?;

        // 1. Token embeddings -> `embed_norm` -> BF16
        let raw_emb = self.embed_tokens.forward(&ids_arr)?;
        let mut hidden = apply_sublayer_rms_norm(
            &raw_emb,
            &self.embed_norm,
            self.config.rms_norm_eps as f32,
            act_dtype,
        )?;

        // 2. Overwrite IMAGE_PAD_ID positions AFTER `embed_norm`
        if let Some(img_emb) = image_embeddings {
            let pad_positions: Vec<usize> = token_ids
                .iter()
                .enumerate()
                .filter_map(|(idx, &tok)| (tok == IMAGE_PAD_ID).then_some(idx))
                .collect();
            let img_rows = img_emb.shape()?[0] as usize;
            if pad_positions.len() != img_rows {
                return Err(Error::from_reason(format!(
                    "Inkling2 paged forward: chunk has {} IMAGE_PAD tokens but image_embeddings has {img_rows} rows",
                    pad_positions.len()
                )));
            }
            if !pad_positions.is_empty() {
                hidden = overwrite_image_pad_rows(&hidden, img_emb, &pad_positions, act_dtype)?;
            }
        }

        // 3. Build per-group attention metadata & optional shared MPP workspace ONCE per wave.
        let (group_metas, kv_dtype, block_size, mut mpp_workspace) = {
            let rt = self.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2Model: paged runtime not initialized")
            })?;
            let num_groups = rt.coordinator.groups().len();
            let mut metas = Vec::with_capacity(num_groups);
            for gid in 0..num_groups {
                let adapter = rt
                    .coordinator
                    .adapter_mut(gid)
                    .map_err(Error::from_reason)?;
                let mut meta = build_group_attention_meta(adapter, seq_q_lens)?;
                #[cfg(target_os = "macos")]
                if let PagedWaveMode::RaggedVerify { ragged_rows, .. } = mode {
                    meta.ragged_slot_mapping = Some(
                        adapter
                            .build_ragged_slot_mapping(ragged_rows)
                            .map_err(Error::from_reason)?,
                    );
                }
                metas.push(meta);
            }
            let ws = match mode {
                PagedWaveMode::Single {
                    rows,
                    first_pos,
                    use_mpp_prefill: true,
                    mpp_pass,
                    pages_per_split,
                    ..
                } => {
                    let max_splits_needed =
                        prefill_plan::splits(mpp_pass, rows, first_pos, pages_per_split);
                    let floats = prefill_plan::workspace_floats(
                        rows,
                        max_splits_needed,
                        self.config.num_attention_heads as u32,
                        self.config.head_dim as u32,
                    );
                    Some(MxArray::zeros(&[floats as i64], Some(DType::Float32))?)
                }
                PagedWaveMode::RaggedVerify {
                    ragged_rows,
                    use_mpp_verify: true,
                    pages_per_split,
                    ..
                } => {
                    let mut max_floats = 0u64;
                    for r in ragged_rows {
                        let splits_needed = prefill_plan::splits(
                            Pass::Verify,
                            r.query_len,
                            r.first_logical_position,
                            pages_per_split,
                        );
                        let floats = prefill_plan::workspace_floats(
                            r.query_len,
                            splits_needed,
                            self.config.num_attention_heads as u32,
                            self.config.head_dim as u32,
                        );
                        max_floats = max_floats.max(floats);
                    }
                    Some(MxArray::zeros(&[max_floats as i64], Some(DType::Float32))?)
                }
                _ => None,
            };
            (
                metas,
                rt.coordinator.kv_dtype(),
                rt.coordinator.block_size(),
                ws,
            )
        };

        // Precompute Global `tau(p)` ONCE for the wave (None when all positions < 8192).
        let global_tau = global_log_scale_tau_for_positions(
            positions,
            self.config.log_scaling_n_floor,
            self.config.log_scaling_alpha,
        )?;

        // 4. Execute Prelude -> Core (passes 0..num_passes with raw BF16 blocks & bank) -> Coda
        let block_size_res = self.config.n_layers_per_attn_res_block;
        let num_visits = self.config.num_layer_visits();
        let is_multi_token_prefill = matches!(mode, PagedWaveMode::Single { .. }) && t > 1;
        let mut blocks: Vec<MxArray> = Vec::new();
        let mut partial = hidden;
        let mut captured_taps = Vec::with_capacity(tap_visit_indices.len());

        // Split borrows cleanly: `self.layers` (read-only) vs `self.paged` (mutable).
        let layers = &self.layers;
        let rt = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Inkling2Model: paged runtime not initialized"))?;

        // Extract active sequences' 48-visit ShortConv states ONCE before the 48-visit loop
        // (eliminates 48 x N `HashMap::remove` + `HashMap::insert` per decode/verify step!).
        // Keep lightweight handle clones so a mid-wave error restores pre-wave state cleanly.
        let mut active_convs: Vec<(SeqId, Inkling2ShortConvState)> =
            Vec::with_capacity(seq_ids.len());
        let mut pre_wave_convs: Vec<(SeqId, Inkling2ShortConvState)> =
            Vec::with_capacity(seq_ids.len());
        for &sid in seq_ids {
            match rt.conv_states.remove(&sid) {
                Some(st) => {
                    pre_wave_convs.push((sid, st.clone()));
                    active_convs.push((sid, st));
                }
                None => {
                    for (prev_sid, prev_st) in active_convs {
                        rt.conv_states.insert(prev_sid, prev_st);
                    }
                    return Err(Error::from_reason(format!(
                        "Inkling2 paged forward: missing conv state for seq {sid}"
                    )));
                }
            }
        }

        let wave_res: Result<()> = (|| {
            let mut run_visit = |visit_idx: usize,
                                 layer: &Inkling2PhysicalLayer,
                                 blocks: &mut Vec<MxArray>,
                                 partial: &mut MxArray,
                                 bank_opt: Option<&[MxArray]>|
             -> Result<()> {
                // Target hidden state taps (`[5, 23, 41]`): capture AFTER any pass restart and
                // BEFORE executing the layer at `visit_idx` (matches `vllm/models/inkling2/nvidia/loop.py`).
                if tap_visit_indices.contains(&visit_idx) {
                    captured_taps.push(partial.clone());
                }

                let route = rt
                    .coordinator
                    .route_for_visit(visit_idx)
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "Inkling2 paged forward: missing KV route for visit {visit_idx}"
                        ))
                    })?
                    .clone();
                let adapter = rt
                    .coordinator
                    .adapter_mut(route.group_id)
                    .map_err(Error::from_reason)?;
                let group_meta = &group_metas[route.group_id];

                if active_convs.len() == 1 {
                    let mut single_conv = [&mut active_convs[0].1.visits[visit_idx]];
                    match conv_tapes.as_deref_mut() {
                        Some(all_tapes) => {
                            let mut single_tape = [&mut all_tapes[0].visits[visit_idx]];
                            forward_physical_layer_paged(
                                layer,
                                blocks,
                                partial,
                                block_size_res,
                                route.physical_layer_ordinal as u32,
                                adapter,
                                group_meta,
                                kv_dtype,
                                block_size,
                                &mode,
                                cu_seqlens_q,
                                global_tau.as_ref(),
                                &mut single_conv,
                                Some(&mut single_tape),
                                bank_opt,
                                mpp_workspace.as_mut(),
                                act_dtype,
                            )?;
                        }
                        None => {
                            forward_physical_layer_paged(
                                layer,
                                blocks,
                                partial,
                                block_size_res,
                                route.physical_layer_ordinal as u32,
                                adapter,
                                group_meta,
                                kv_dtype,
                                block_size,
                                &mode,
                                cu_seqlens_q,
                                global_tau.as_ref(),
                                &mut single_conv,
                                None,
                                bank_opt,
                                mpp_workspace.as_mut(),
                                act_dtype,
                            )?;
                        }
                    }
                } else {
                    let mut visit_convs: Vec<&mut VisitShortConvState> = active_convs
                        .iter_mut()
                        .map(|(_, st)| &mut st.visits[visit_idx])
                        .collect();
                    let mut visit_tapes_vec: Option<Vec<&mut VisitShortConvTape>> = conv_tapes
                        .as_deref_mut()
                        .map(|all| all.iter_mut().map(|t| &mut t.visits[visit_idx]).collect());

                    forward_physical_layer_paged(
                        layer,
                        blocks,
                        partial,
                        block_size_res,
                        route.physical_layer_ordinal as u32,
                        adapter,
                        group_meta,
                        kv_dtype,
                        block_size,
                        &mode,
                        cu_seqlens_q,
                        global_tau.as_ref(),
                        &mut visit_convs,
                        visit_tapes_vec.as_deref_mut(),
                        bank_opt,
                        mpp_workspace.as_mut(),
                        act_dtype,
                    )?;
                }

                // Multi-token prefill: smooth peak VRAM every K visits.
                // Decode / verify: pipeline early + periodic async Metal submissions so the GPU
                // executes visits while the CPU finishes building the rest of the 48-visit graph!
                if is_multi_token_prefill {
                    crate::array::maybe_eval_clear_for_paged_prefill_layer(visit_idx, partial)?;
                } else {
                    maybe_submit_paged_visit(visit_idx, num_visits, partial);
                }
                Ok(())
            };

            if let Some(ref lc) = self.config.loop_config {
                let mut visit_idx = 0usize;

                // Prelude: physical `0..core_start`, pass 0
                for layer in &layers[..lc.core_start] {
                    run_visit(visit_idx, layer, &mut blocks, &mut partial, None)?;
                    visit_idx += 1;
                }

                // Save prelude state for resetting before passes 1..num_passes
                let prelude_blocks = blocks.clone();
                let prelude_partial = partial.clone();
                let prelude_block_count = prelude_blocks.len();

                for pass in 0..lc.num_passes {
                    // Build raw BF16 cross-pass bank ONCE per pass (zero normalize/stack dispatches!).
                    let bank: Option<Vec<MxArray>> = if pass > 0 {
                        let mut b = Vec::with_capacity(
                            blocks.len().saturating_sub(prelude_block_count + 1) + 1,
                        );
                        b.extend_from_slice(&blocks[prelude_block_count + 1..]);
                        b.push(partial.clone());
                        blocks = prelude_blocks.clone();
                        partial = prelude_partial.clone();
                        Some(b)
                    } else {
                        None
                    };

                    for layer in &layers[lc.core_start..lc.core_end] {
                        run_visit(visit_idx, layer, &mut blocks, &mut partial, bank.as_deref())?;
                        visit_idx += 1;
                    }
                }

                // Coda: physical `core_end..num_physical_layers`, continues from final core pass
                for layer in &layers[lc.core_end..] {
                    run_visit(visit_idx, layer, &mut blocks, &mut partial, None)?;
                    visit_idx += 1;
                }
            } else {
                for (visit_idx, layer) in layers.iter().enumerate() {
                    run_visit(visit_idx, layer, &mut blocks, &mut partial, None)?;
                }
            }
            Ok(())
        })();

        // Restore pre-wave ShortConv states on error so partial mid-wave mutations never leak;
        // commit updated states only when all 48 visits succeeded.
        if wave_res.is_ok() {
            for (sid, st) in active_convs {
                rt.conv_states.insert(sid, st);
            }
        } else {
            for (sid, st) in pre_wave_convs {
                rt.conv_states.insert(sid, st);
            }
        }
        wave_res?;

        let capture_final_norm = tap_visit_indices.contains(&num_visits);
        let (logits_opt, final_norm_opt) = self.compute_final_norm_and_logits(
            &blocks,
            &partial,
            t,
            last_only,
            capture_final_norm,
            compute_logits,
            act_dtype,
        )?;
        if let Some(norm_all) = final_norm_opt {
            captured_taps.push(norm_all);
        }
        if logits_opt.is_none() {
            partial.eval();
        }
        Ok((logits_opt, captured_taps))
    }
}

/// Execute one logical visit of `layer` on the paged path using all Stage 2 fused Metal kernels.
#[allow(clippy::too_many_arguments)]
fn forward_physical_layer_paged(
    layer: &Inkling2PhysicalLayer,
    blocks: &mut Vec<MxArray>,
    partial: &mut MxArray,
    block_size_res: usize,
    phys_layer_ord: u32,
    adapter: &mut PagedKVCacheAdapter,
    group_meta: &GroupAttentionMeta,
    kv_dtype: Inkling2KvDtype,
    page_block_size: u32,
    mode: &PagedWaveMode<'_>,
    cu_seqlens_q: &[usize],
    global_tau: Option<&MxArray>,
    visit_convs: &mut [&mut VisitShortConvState],
    mut visit_tapes: Option<&mut [&mut VisitShortConvTape]>,
    cross_pass_bank: Option<&[MxArray]>,
    mpp_workspace: Option<&mut MxArray>,
    act_dtype: DType,
) -> Result<()> {
    let t = partial.shape()?[0];
    let d = partial.shape()?[1];

    // Optional cross-pass message read (passes 1 & 2 of looped core layers):
    // 1 fused `inkling2_block_attn_res` dispatch over raw BF16 bank -> `[4, T, D]` FP32!
    let messages = match (cross_pass_bank, &layer.loop_ctrl) {
        (Some(bank), Some(ctrl)) => {
            let bank_refs: Vec<&MxArray> = bank.iter().collect();
            let msgs = inkling2_block_attn_res(
                &bank_refs,
                None,
                &ctrl.msg_queries,
                None,
                ATTN_RES_EPS,
                ATTN_RES_EPS,
                false,
                DType::Float32,
            )?;
            Some((msgs, ctrl))
        }
        _ => None,
    };

    // 1 & 2. Attention Block AttnRes + optional Q/K/V loop controllers + sublayer RMSNorm
    let mut attn_srcs: Vec<&MxArray> = Vec::with_capacity(blocks.len() + 1);
    for b in blocks.iter() {
        attn_srcs.push(b);
    }
    attn_srcs.push(partial);

    let (x_q, x_k_v_normed) = if let Some((ref msgs, ctrl)) = messages {
        // Loop visits (passes 1 & 2): raw FP32 AttnRes mix -> batched Q/K/V loop controllers (4 dispatches!)
        let dest = if let Some(ref q_res) = layer.attn_res_query {
            inkling2_block_attn_res(
                &attn_srcs,
                None,
                q_res,
                None,
                ATTN_RES_EPS,
                layer.rms_norm_eps,
                layer.is_local,
                DType::Float32,
            )?
        } else {
            partial.astype(DType::Float32)?
        };
        let msgs_qkv = msgs.slice(&[0, 0, 0], &[3, t, d])?;
        let norm_w = match &layer.attn_inp_norm {
            AttnInputNorm::Local(w) => w,
            AttnInputNorm::Global { qkv_norm, .. } => qkv_norm,
        };
        let (xq, xk, xv) = ctrl.forward_qkv_fused_rms_norm(
            &dest,
            &msgs_qkv,
            norm_w,
            layer.rms_norm_eps,
            act_dtype,
        )?;
        (xq, Some((xk, xv)))
    } else if let Some(ref q_res) = layer.attn_res_query {
        // Non-loop visits (phys > 0): fuse Block AttnRes + sublayer RMSNorm into 1 SINGLE Metal dispatch!
        match &layer.attn_inp_norm {
            AttnInputNorm::Local(w) => {
                let xq = inkling2_block_attn_res(
                    &attn_srcs,
                    None,
                    q_res,
                    Some(w),
                    ATTN_RES_EPS,
                    layer.rms_norm_eps,
                    true,
                    act_dtype,
                )?;
                (xq, None)
            }
            AttnInputNorm::Global { qkv_norm, .. } => {
                let x_qkv = inkling2_block_attn_res(
                    &attn_srcs,
                    None,
                    q_res,
                    Some(qkv_norm),
                    ATTN_RES_EPS,
                    layer.rms_norm_eps,
                    false,
                    act_dtype,
                )?;
                let xq = x_qkv.slice(&[0, 0, 0], &[1, t, d])?.squeeze(Some(&[0]))?;
                let xk = x_qkv.slice(&[1, 0, 0], &[2, t, d])?.squeeze(Some(&[0]))?;
                let xv = x_qkv.slice(&[2, 0, 0], &[3, t, d])?.squeeze(Some(&[0]))?;
                (xq, Some((xk, xv)))
            }
        }
    } else {
        // Physical layer 0: no `attn_res_query`
        match &layer.attn_inp_norm {
            AttnInputNorm::Local(w) => (
                apply_sublayer_rms_norm(partial, w, layer.rms_norm_eps, act_dtype)?,
                None,
            ),
            AttnInputNorm::Global {
                q_norm,
                k_norm,
                v_norm,
                ..
            } => (
                apply_sublayer_rms_norm(partial, q_norm, layer.rms_norm_eps, act_dtype)?,
                Some((
                    apply_sublayer_rms_norm(partial, k_norm, layer.rms_norm_eps, act_dtype)?,
                    apply_sublayer_rms_norm(partial, v_norm, layer.rms_norm_eps, act_dtype)?,
                )),
            ),
        }
    };

    // 3 & 4. Paged Self-Attention + fused `attn_post_conv` (+ fused FP32 residual epilogue when not a block boundary!)
    let is_block_boundary = layer.phys_idx.is_multiple_of(block_size_res);
    let attn_residual = if is_block_boundary {
        None
    } else {
        Some(&*partial)
    };
    let ffn_inp = forward_attention_paged(
        &layer.self_attn,
        &x_q,
        x_k_v_normed.as_ref().map(|(k, v)| (k, v)),
        attn_residual,
        phys_layer_ord,
        adapter,
        group_meta,
        kv_dtype,
        page_block_size,
        mode,
        cu_seqlens_q,
        global_tau,
        visit_convs,
        visit_tapes.as_deref_mut(),
        mpp_workspace,
    )?;
    if is_block_boundary {
        blocks.push(partial.clone());
    }

    // 5. MLP Block AttnRes + optional MLP loop controller + FFN sublayer RMSNorm
    let mut mlp_srcs: Vec<&MxArray> = Vec::with_capacity(blocks.len() + 1);
    for b in blocks.iter() {
        mlp_srcs.push(b);
    }
    mlp_srcs.push(&ffn_inp);

    let x_mlp = if let Some((ref msgs, ctrl)) = messages {
        let dest_mlp = inkling2_block_attn_res(
            &mlp_srcs,
            None,
            &layer.ffn_res_query,
            None,
            ATTN_RES_EPS,
            layer.rms_norm_eps,
            true,
            DType::Float32,
        )?;
        let msg_mlp = msgs.slice(&[3, 0, 0], &[4, t, d])?.squeeze(Some(&[0]))?;
        ctrl.mlp_ctrl.forward_fused_rms_norm(
            &dest_mlp,
            &msg_mlp,
            &layer.ffn_norm,
            layer.rms_norm_eps,
            act_dtype,
        )?
    } else {
        // Non-loop visit: fuse MLP Block AttnRes + FFN RMSNorm into 1 SINGLE Metal dispatch!
        inkling2_block_attn_res(
            &mlp_srcs,
            None,
            &layer.ffn_res_query,
            Some(&layer.ffn_norm),
            ATTN_RES_EPS,
            layer.rms_norm_eps,
            true,
            act_dtype,
        )?
    };

    // 6 & 7. Sparse MoE (fused route + situ + combine + ragged `ffn_post_conv` + FP32 residual epilogue!)
    let cu_arr = Some(&group_meta.cu_seqlens_q);
    *partial = if visit_convs.len() == 1 {
        let mut hists = [&mut visit_convs[0].mlp_post];
        match visit_tapes {
            Some(ts) => {
                let mut tapes_arr = [&mut ts[0].mlp_post];
                layer.moe.forward_fused_residual_taped(
                    &x_mlp,
                    Some(&ffn_inp),
                    &mut hists,
                    cu_seqlens_q,
                    cu_arr,
                    Some(&mut tapes_arr),
                )?
            }
            None => layer.moe.forward_fused_residual_taped(
                &x_mlp,
                Some(&ffn_inp),
                &mut hists,
                cu_seqlens_q,
                cu_arr,
                None,
            )?,
        }
    } else {
        let mut mlp_post_hists: Vec<&mut MxArray> =
            visit_convs.iter_mut().map(|vc| &mut vc.mlp_post).collect();
        let mut mlp_post_tapes_vec: Option<Vec<&mut Option<StreamShortConvTape>>> =
            visit_tapes.map(|ts| ts.iter_mut().map(|vt| &mut vt.mlp_post).collect());
        layer.moe.forward_fused_residual_taped(
            &x_mlp,
            Some(&ffn_inp),
            &mut mlp_post_hists,
            cu_seqlens_q,
            cu_arr,
            mlp_post_tapes_vec.as_deref_mut(),
        )?
    };
    Ok(())
}

/// Paged self-attention using Stage 2 fused Metal kernels (`inkling2_qr_clip`,
/// `inkling2_sconv_k4_kv_prep`, `inkling2_paged_prefill_attention` / `inkling2_paged_attention`,
/// strided `inkling2_gate_sink`, and `inkling2_sconv_k4_ragged` with optional FP32 residual epilogue).
#[allow(clippy::too_many_arguments)]
fn forward_attention_paged(
    attn: &super::attention::Inkling2Attention,
    x_q: &MxArray,
    x_k_v: Option<(&MxArray, &MxArray)>,
    residual: Option<&MxArray>,
    phys_layer_ord: u32,
    adapter: &mut PagedKVCacheAdapter,
    group_meta: &GroupAttentionMeta,
    kv_dtype: Inkling2KvDtype,
    page_block_size: u32,
    mode: &PagedWaveMode<'_>,
    cu_seqlens_q: &[usize],
    global_tau: Option<&MxArray>,
    visit_convs: &mut [&mut VisitShortConvState],
    mut visit_tapes: Option<&mut [&mut VisitShortConvTape]>,
    mpp_workspace: Option<&mut MxArray>,
) -> Result<MxArray> {
    let t_i64 = x_q.shape()?[0];
    let act_dtype = x_q.dtype()?;
    let q_dim = (attn.num_heads * attn.head_dim) as i64;

    // 1. Projections (using fused `QRGKV` / `QRG` when available — strided views stay zero-copy!)
    let (q_raw, r_raw, gate_raw, k_raw, v_raw, sink_raw) =
        attn.project_qrgkv_and_sink(x_q, x_k_v)?;

    // 2. Global dynamic ShortConv tap projections (only on the 8 Global visits)
    let (x_k_inp, x_v_inp) = x_k_v.unwrap_or((x_q, x_q));
    let dyn_k = if let Some(ref k_cp) = attn.k_conv_proj {
        Some(k_cp.forward(x_k_inp)?.reshape(&[
            t_i64,
            attn.num_kv_heads as i64,
            attn.k_conv.shape()?[1],
        ])?)
    } else {
        None
    };
    let dyn_v = if let Some(ref v_cp) = attn.v_conv_proj {
        Some(v_cp.forward(x_v_inp)?.reshape(&[
            t_i64,
            attn.num_kv_heads as i64,
            attn.v_conv.shape()?[1],
        ])?)
    } else {
        None
    };

    // 3. Fused Q + R L2-clip (+ optional global tau) in 1 Metal dispatch (zero-copy strided views!)
    let tau_opt = if !attn.is_local { global_tau } else { None };
    let (q_cur, r_for_rel) = inkling2_qr_clip(
        &q_raw,
        &r_raw,
        &attn.q_clip,
        &attn.rel_clip,
        tau_opt,
        attn.num_heads,
        attn.head_dim,
        attn.d_rel,
        attn.qk_l2_clip_threshold as f32,
        attn.rel_l2_clip_threshold as f32,
        1e-6,
        DType::BFloat16,
        act_dtype == DType::BFloat16,
    )?;

    // 4. Fused K+V ShortConv + K L2-clip + V RMSNorm in 1 Metal dispatch (single or ragged N lanes!)
    let cu_arr = Some(&group_meta.cu_seqlens_q);
    let (k_cur, v_cur) = if visit_convs.len() == 1 {
        let vc0 = &mut visit_convs[0];
        let mut kh = [&mut vc0.k];
        let mut vh = [&mut vc0.v];
        match visit_tapes.as_deref_mut() {
            Some(ts) => {
                let vt0 = &mut ts[0];
                let mut kt = [&mut vt0.k];
                let mut vt = [&mut vt0.v];
                fused_short_conv_kv_prep_taped(
                    &k_raw,
                    &v_raw,
                    &attn.k_conv,
                    &attn.v_conv,
                    dyn_k.as_ref(),
                    dyn_v.as_ref(),
                    &mut kh,
                    &mut vh,
                    cu_seqlens_q,
                    cu_arr,
                    Some(&mut kt),
                    Some(&mut vt),
                    attn.num_kv_heads,
                    attn.head_dim,
                    attn.qk_l2_clip_threshold as f32,
                    attn.rms_norm_eps,
                )?
            }
            None => fused_short_conv_kv_prep_taped(
                &k_raw,
                &v_raw,
                &attn.k_conv,
                &attn.v_conv,
                dyn_k.as_ref(),
                dyn_v.as_ref(),
                &mut kh,
                &mut vh,
                cu_seqlens_q,
                cu_arr,
                None,
                None,
                attn.num_kv_heads,
                attn.head_dim,
                attn.qk_l2_clip_threshold as f32,
                attn.rms_norm_eps,
            )?,
        }
    } else {
        let mut kh: Vec<&mut MxArray> = Vec::with_capacity(visit_convs.len());
        let mut vh: Vec<&mut MxArray> = Vec::with_capacity(visit_convs.len());
        for vc in visit_convs.iter_mut() {
            kh.push(&mut vc.k);
            vh.push(&mut vc.v);
        }
        match visit_tapes.as_deref_mut() {
            Some(ts) => {
                let mut kt: Vec<&mut Option<StreamShortConvTape>> = Vec::with_capacity(ts.len());
                let mut vt: Vec<&mut Option<StreamShortConvTape>> = Vec::with_capacity(ts.len());
                for t_slot in ts.iter_mut() {
                    kt.push(&mut t_slot.k);
                    vt.push(&mut t_slot.v);
                }
                fused_short_conv_kv_prep_taped(
                    &k_raw,
                    &v_raw,
                    &attn.k_conv,
                    &attn.v_conv,
                    dyn_k.as_ref(),
                    dyn_v.as_ref(),
                    &mut kh,
                    &mut vh,
                    cu_seqlens_q,
                    cu_arr,
                    Some(&mut kt),
                    Some(&mut vt),
                    attn.num_kv_heads,
                    attn.head_dim,
                    attn.qk_l2_clip_threshold as f32,
                    attn.rms_norm_eps,
                )?
            }
            None => fused_short_conv_kv_prep_taped(
                &k_raw,
                &v_raw,
                &attn.k_conv,
                &attn.v_conv,
                dyn_k.as_ref(),
                dyn_v.as_ref(),
                &mut kh,
                &mut vh,
                cu_seqlens_q,
                cu_arr,
                None,
                None,
                attn.num_kv_heads,
                attn.head_dim,
                attn.qk_l2_clip_threshold as f32,
                attn.rms_norm_eps,
            )?,
        }
    };

    // 5. Relative bias `R @ rel_bias_t` -> `[T, H, rel_extent]` FP32
    let rel_bias = r_for_rel.matmul(&attn.rel_bias_t)?;

    // 6. Write K, V into paged Metal pool (INT8 32-token KTM or BF16 vLLM)
    match mode {
        PagedWaveMode::Single {
            seq_id,
            first_pos,
            floor,
            ..
        } => {
            adapter
                .activate_request(*seq_id)
                .map_err(Error::from_reason)?;
            adapter
                .update_keys_values_native_recorded(
                    phys_layer_ord,
                    &k_cur,
                    &v_cur,
                    *first_pos,
                    *floor,
                )
                .map_err(Error::from_reason)?;
        }
        PagedWaveMode::BatchedDecode { planned_rows } => {
            adapter
                .update_keys_values_native_batched(phys_layer_ord, &k_cur, &v_cur, planned_rows)
                .map_err(Error::from_reason)?;
        }
        PagedWaveMode::RaggedVerify { ragged_rows, .. } => {
            #[cfg(target_os = "macos")]
            if let Some((ref slot_mapping, total_queries)) = group_meta.ragged_slot_mapping {
                adapter
                    .update_keys_values_native_with_slot_mapping(
                        phys_layer_ord,
                        &k_cur,
                        &v_cur,
                        slot_mapping,
                        total_queries,
                    )
                    .map_err(Error::from_reason)?;
            } else {
                adapter
                    .update_keys_values_native_ragged(phys_layer_ord, &k_cur, &v_cur, ragged_rows)
                    .map_err(Error::from_reason)?;
            }
            #[cfg(not(target_os = "macos"))]
            adapter
                .update_keys_values_native_ragged(phys_layer_ord, &k_cur, &v_cur, ragged_rows)
                .map_err(Error::from_reason)?;
        }
    }

    // 7. Fetch dependency-carrying pool views and dispatch Inkling2 Paged Attention
    let k_pool = adapter
        .key_pool_array(phys_layer_ord)
        .map_err(Error::from_reason)?;
    let v_pool = adapter
        .value_pool_array(phys_layer_ord)
        .map_err(Error::from_reason)?;
    let (k_scale, v_scale) = adapter
        .scale_pool_arrays(phys_layer_ord)
        .map_err(Error::from_reason)?;

    let scale = 1.0_f32 / (attn.head_dim as f32).sqrt();
    let swa_i32 = if attn.is_local {
        attn.sliding_window as i32
    } else {
        0
    };
    let rel_extent_i32 = attn.rel_extent as i32;
    let bs_i32 = page_block_size as i32;
    let n_q_i32 = attn.num_heads as i32;
    let n_kv_i32 = attn.num_kv_heads as i32;
    let hd_i32 = attn.head_dim as i32;

    let (attn_o, lse) = match (mode, mpp_workspace) {
        (
            PagedWaveMode::Single {
                first_pos,
                rows,
                use_mpp_prefill: true,
                mpp_pass,
                pages_per_split,
                int8_q_from,
                ..
            },
            Some(ws),
        ) => {
            let bt_1d = group_meta.block_table_1d.as_ref().ok_or_else(|| {
                Error::from_reason("Inkling2 paged prefill: missing 1D block_table")
            })?;
            let active_history = if attn.is_local {
                (*first_pos).min((attn.sliding_window as u32).saturating_sub(1))
            } else {
                *first_pos
            };
            let splits = prefill_plan::splits(*mpp_pass, *rows, active_history, *pages_per_split);
            let use_int8_q = !attn.is_local
                && kv_dtype == Inkling2KvDtype::Int8
                && int8_q_from.is_some_and(|from| *first_pos >= from);
            let (out, lse, ws_next) = inkling2_paged_prefill_attention(
                &q_cur,
                &k_pool,
                &v_pool,
                &k_scale,
                &v_scale,
                bt_1d,
                ws,
                Some(&rel_bias),
                *first_pos as i32,
                splits as i32,
                mpp_pass.max_splits() as i32,
                bs_i32,
                n_q_i32,
                n_kv_i32,
                hd_i32,
                swa_i32,
                rel_extent_i32,
                kv_dtype,
                use_int8_q,
                scale,
            )?;
            *ws = ws_next;
            (out, lse)
        }
        (
            PagedWaveMode::RaggedVerify {
                ragged_rows,
                use_mpp_verify: true,
                pages_per_split,
                int8_q_from,
            },
            Some(ws),
        ) if !attn.is_local => {
            // Run `inkling2_paged_prefill_attention` (`Pass::Verify`) PER LANE so every lane in a
            // 2..=4 lane stacked `[N * 8]` wave executes the exact same attention kernel & split
            // schedule as running solo (bit-exact!), while all dense Q8_0 and NVFP4 MoE
            // projections remain stacked `[N * 8, 2304]`.
            if ragged_rows.len() == 1 {
                let bt_1d = group_meta.block_table_1d.as_ref().ok_or_else(|| {
                    Error::from_reason("Inkling2 paged verify: missing 1D block_table")
                })?;
                let r0 = ragged_rows[0];
                let active_history = if attn.is_local {
                    r0.first_logical_position
                        .min((attn.sliding_window as u32).saturating_sub(1))
                } else {
                    r0.first_logical_position
                };
                let splits = prefill_plan::splits(
                    Pass::Verify,
                    r0.query_len,
                    active_history,
                    *pages_per_split,
                );
                let use_int8_q = !attn.is_local
                    && kv_dtype == Inkling2KvDtype::Int8
                    && int8_q_from.is_some_and(|from| r0.first_logical_position >= from);
                let (out, lse, ws_next) = inkling2_paged_prefill_attention(
                    &q_cur,
                    &k_pool,
                    &v_pool,
                    &k_scale,
                    &v_scale,
                    bt_1d,
                    ws,
                    Some(&rel_bias),
                    r0.first_logical_position as i32,
                    splits as i32,
                    Pass::Verify.max_splits() as i32,
                    bs_i32,
                    n_q_i32,
                    n_kv_i32,
                    hd_i32,
                    swa_i32,
                    rel_extent_i32,
                    kv_dtype,
                    use_int8_q,
                    scale,
                )?;
                *ws = ws_next;
                (out, lse)
            } else {
                // Allocate independent per-lane MPP workspaces so all N lanes run without
                // artificial cross-lane write-after-read serialization on a shared workspace!
                let ws_len = ws.shape()?[0];
                let mut lane_outs = Vec::with_capacity(ragged_rows.len());
                let mut lane_lses = Vec::with_capacity(ragged_rows.len());
                let rb_ext = rel_bias.shape()?[2];
                for (lane_idx, r) in ragged_rows.iter().enumerate() {
                    let row_start = cu_seqlens_q[lane_idx] as i64;
                    let row_end = cu_seqlens_q[lane_idx + 1] as i64;
                    let q_lane = q_cur.slice(
                        &[row_start, 0, 0],
                        &[row_end, attn.num_heads as i64, attn.head_dim as i64],
                    )?;
                    let rb_lane = rel_bias.slice(
                        &[row_start, 0, 0],
                        &[row_end, attn.num_heads as i64, rb_ext],
                    )?;
                    let bt_1d = &group_meta.per_seq_block_tables_1d[lane_idx];
                    let active_history = if attn.is_local {
                        r.first_logical_position
                            .min((attn.sliding_window as u32).saturating_sub(1))
                    } else {
                        r.first_logical_position
                    };
                    let splits = prefill_plan::splits(
                        Pass::Verify,
                        r.query_len,
                        active_history,
                        *pages_per_split,
                    );
                    let use_int8_q = !attn.is_local
                        && kv_dtype == Inkling2KvDtype::Int8
                        && int8_q_from.is_some_and(|from| r.first_logical_position >= from);
                    let lane_ws_owned;
                    let lane_ws_ref = if lane_idx == 0 {
                        &*ws
                    } else {
                        lane_ws_owned = MxArray::zeros(&[ws_len], Some(DType::Float32))?;
                        &lane_ws_owned
                    };
                    let (out_l, lse_l, ws_next) = inkling2_paged_prefill_attention(
                        &q_lane,
                        &k_pool,
                        &v_pool,
                        &k_scale,
                        &v_scale,
                        bt_1d,
                        lane_ws_ref,
                        Some(&rb_lane),
                        r.first_logical_position as i32,
                        splits as i32,
                        Pass::Verify.max_splits() as i32,
                        bs_i32,
                        n_q_i32,
                        n_kv_i32,
                        hd_i32,
                        swa_i32,
                        rel_extent_i32,
                        kv_dtype,
                        use_int8_q,
                        scale,
                    )?;
                    if lane_idx == 0 {
                        *ws = ws_next;
                    }
                    lane_outs.push(out_l);
                    lane_lses.push(lse_l);
                }
                let out_refs: Vec<&MxArray> = lane_outs.iter().collect();
                let lse_refs: Vec<&MxArray> = lane_lses.iter().collect();
                (
                    MxArray::concatenate_many(out_refs, Some(0))?,
                    MxArray::concatenate_many(lse_refs, Some(0))?,
                )
            }
        }
        _ => inkling2_paged_attention(
            &q_cur,
            &k_pool,
            &v_pool,
            &group_meta.block_table_2d,
            &group_meta.seq_lens,
            &group_meta.cu_seqlens_q,
            &k_scale,
            &v_scale,
            Some(&rel_bias),
            scale,
            swa_i32,
            rel_extent_i32,
            bs_i32,
            n_q_i32,
            n_kv_i32,
            hd_i32,
            kv_dtype,
            None,
        )?,
    };

    // 8. Fused Gate + Sink epilogue: `Y = O * sigmoid(gate) * sigmoid(lse - sink)`
    let y_3d = inkling2_gate_sink(&attn_o, &gate_raw, &lse, &sink_raw)?;
    let y = y_3d.reshape(&[t_i64, q_dim])?;

    // 9. Output projection + fused static `attn_post_conv` (+ optional FP32 residual epilogue)
    let proj_out = attn.o_proj.forward(&y)?;
    if visit_convs.len() == 1 {
        let mut hists = [&mut visit_convs[0].attn_post];
        match visit_tapes {
            Some(ts) => {
                let mut tapes_arr = [&mut ts[0].attn_post];
                fused_short_conv_ragged_residual_taped(
                    &proj_out,
                    &attn.post_conv,
                    None,
                    residual,
                    None,
                    &mut hists,
                    cu_seqlens_q,
                    cu_arr,
                    Some(&mut tapes_arr),
                )
            }
            None => fused_short_conv_ragged_residual_taped(
                &proj_out,
                &attn.post_conv,
                None,
                residual,
                None,
                &mut hists,
                cu_seqlens_q,
                cu_arr,
                None,
            ),
        }
    } else {
        let mut post_hists: Vec<&mut MxArray> =
            visit_convs.iter_mut().map(|vc| &mut vc.attn_post).collect();
        let mut post_tapes_vec: Option<Vec<&mut Option<StreamShortConvTape>>> =
            visit_tapes.map(|ts| ts.iter_mut().map(|vt| &mut vt.attn_post).collect());
        fused_short_conv_ragged_residual_taped(
            &proj_out,
            &attn.post_conv,
            None,
            residual,
            None,
            &mut post_hists,
            cu_seqlens_q,
            cu_arr,
            post_tapes_vec.as_deref_mut(),
        )
    }
}

// ============================================================================
// Engine Trait Wiring: `ChatBackend` + `PagedBackend` for `Inkling2Inner`
// ============================================================================

/// Flat decode stepper for [`ChatBackend::begin_decode`].
pub(crate) struct Inkling2FlatDecode<'a> {
    model: &'a mut Inkling2Inner,
}

impl DecodeStep for Inkling2FlatDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let tok = input_ids.item_at_int32(0)? as u32;
        let mut cache = self
            .model
            .flat_cache
            .take()
            .ok_or_else(|| Error::from_reason("Inkling2 flat decode: cache missing"))?;
        let res = self.model.forward_tokens(&[tok], None, &mut cache);
        self.model.flat_cache = Some(cache);
        let logits = res?;
        Ok((logits, false))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token]);
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        let mut cache =
            self.model.flat_cache.take().ok_or_else(|| {
                Error::from_reason("Inkling2 flat materialize_final: cache missing")
            })?;
        let res = self.model.forward_tokens(&[token_id], None, &mut cache);
        self.model.flat_cache = Some(cache);
        let _ = res?;
        Ok(())
    }
}

/// Paged decode stepper for [`PagedBackend::begin_paged_decode`].
pub(crate) struct Inkling2PagedDecode<'a> {
    model: &'a mut Inkling2Inner,
    pending_cache_error: Option<String>,
}

impl PagedStepModel for Inkling2PagedDecode<'_> {
    const EVAL: EvalPolicy = EvalPolicy::AsyncToken;
    const FINAL_TOKEN_POLICY: FinalTokenPolicy = FinalTokenPolicy::KeepAllOnLength;

    fn paged_step(&mut self, token_id: u32) -> Result<MxArray> {
        let seq_id = self
            .model
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 paged decode: runtime missing"))?
            .active_seq;
        let logits_2d = self.model.paged_decode_step(seq_id, token_id)?;
        let vocab = logits_2d.shape()?[1];
        logits_2d.reshape(&[1, 1, vocab])
    }

    fn maintain_cache(&mut self, step: i32) {
        if self.pending_cache_error.is_none()
            && let Some(seq_id) = self.model.paged.as_ref().map(|r| r.active_seq)
            && let Err(e) = self.model.settle_paged_step(seq_id)
        {
            self.pending_cache_error = Some(e.reason);
        }
        crate::array::maybe_clear_cache_for_paged_step(step);
    }

    fn end_decode(&mut self) -> Result<()> {
        if self.pending_cache_error.is_none()
            && let Some(seq_id) = self.model.paged.as_ref().map(|r| r.active_seq)
            && let Err(e) = self.model.settle_paged_step(seq_id)
        {
            self.pending_cache_error = Some(e.reason);
        }
        self.pending_cache_error
            .take()
            .map_or(Ok(()), |e| Err(Error::from_reason(e)))
    }

    fn materialize_final_token(&mut self, token_id: u32) -> Result<()> {
        let seq_id = self
            .model
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 paged materialize_final: runtime missing"))?
            .active_seq;
        let _ = self.model.paged_decode_step(seq_id, token_id)?;
        let _ = self.model.settle_paged_step(seq_id)?;
        Ok(())
    }
}

/// Prefix priming metadata returned by [`PagedBackend::prime_prefix_state`].
pub(crate) struct Inkling2PrefixState {
    pub effective_cached_prefix_len: usize,
    pub suffix_len: usize,
    #[allow(dead_code)]
    pub full_tokens: Vec<u32>,
    pub cache_salt: u64,
    /// `true` only when `prepare_prefix_sync` continued a live in-memory sequence on `seq_id`
    /// (never `true` for cold starts or RAM/SSD `StateStore` restores).
    pub live_continuation: bool,
}

impl PagedPrefix for Inkling2PrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }

    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

/// Inkling2's [`StreamEmitter`]: routes committed token text through
/// [`InklingStreamParser`] so reasoning deltas (`is_reasoning = true`, gated by
/// `include_reasoning`) and visible content deltas (`is_reasoning = false`) are
/// separated cleanly, while Inkling2 framing markers (`<|message_model|>`,
/// `<|content_thinking|>`, `<|content_text|>`, `<|content_invoke_tool_json|>`,
/// `<|end_message|>`, `<|content_model_end_sampling|>`), tool-name headers,
/// and raw JSON tool-call payloads are never leaked into streamed chunks.
pub(crate) struct Inkling2StreamEmitter {
    parser: InklingStreamParser,
    /// Latched from the turn's `include_reasoning` flag so [`Self::finish`]
    /// applies the same suppression gate to any reasoning tail flushed at EOS.
    include_reasoning: bool,
}

impl Inkling2StreamEmitter {
    pub(crate) fn new() -> Self {
        Self::with_initial_state(InklingParserState::MessageHeader)
    }

    pub(crate) fn with_initial_state(initial_state: InklingParserState) -> Self {
        Self {
            parser: InklingStreamParser::with_initial_state(initial_state),
            include_reasoning: true,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_prompt_tokens(prompt_tokens: &[u32]) -> Self {
        Self {
            parser: InklingStreamParser::from_prompt_tokens(prompt_tokens),
            include_reasoning: true,
        }
    }

    fn emit_delta(text: String, is_reasoning: bool, sink: &dyn ChunkSink) {
        if text.is_empty() {
            return;
        }
        sink.send(Ok(ChatStreamChunk {
            text,
            done: false,
            finish_reason: None,
            tool_calls: None,
            thinking: None,
            thinking_enabled: None,
            num_tokens: None,
            prompt_tokens: None,
            reasoning_tokens: None,
            reasoning_close: None,
            reasoning_cut_at: None,
            reasoning_cycle_period: None,
            reasoning_stray_closes: None,
            raw_text: None,
            public_raw_text: None,
            text_authoritative: None,
            cached_tokens: None,
            performance: None,
            is_reasoning: Some(is_reasoning),
        }));
    }

    fn emit_deltas(
        &self,
        content_delta: String,
        reasoning_delta: String,
        include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        if include_reasoning && !reasoning_delta.is_empty() {
            Self::emit_delta(reasoning_delta, true, sink);
        }
        if !content_delta.is_empty() {
            Self::emit_delta(content_delta, false, sink);
        }
    }
}

impl StreamEmitter for Inkling2StreamEmitter {
    fn on_token_id(
        &mut self,
        token_id: u32,
        token_text: &str,
        _is_reasoning: bool,
        include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        self.include_reasoning = include_reasoning;
        if let Some(marker) = output_marker_for_token_id(token_id)
            && !token_text.contains(marker)
        {
            if !token_text.is_empty() {
                let (content, reasoning, _) = self.parser.push_text(token_text);
                self.emit_deltas(content, reasoning, include_reasoning, sink);
            }
            let (content, reasoning, _) = self.parser.push_token(token_id, "");
            self.emit_deltas(content, reasoning, include_reasoning, sink);
            return;
        }
        if !token_text.is_empty() {
            let (content, reasoning, _) = self.parser.push_text(token_text);
            self.emit_deltas(content, reasoning, include_reasoning, sink);
        }
    }

    fn on_token_text(
        &mut self,
        token_text: &str,
        _is_reasoning: bool,
        include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        self.include_reasoning = include_reasoning;
        if token_text.is_empty() {
            return;
        }
        let (content, reasoning, _) = self.parser.push_text(token_text);
        self.emit_deltas(content, reasoning, include_reasoning, sink);
    }

    fn on_residual(
        &mut self,
        residual: &str,
        _is_reasoning: bool,
        include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        self.include_reasoning = include_reasoning;
        if residual.is_empty() {
            return;
        }
        let (content, reasoning, _) = self.parser.push_text(residual);
        self.emit_deltas(content, reasoning, include_reasoning, sink);
    }

    fn finish(&mut self, result: &ChatResult, sink: &dyn ChunkSink) {
        let (content_tail, reasoning_tail, _) = self.parser.flush();
        self.emit_deltas(content_tail, reasoning_tail, self.include_reasoning, sink);

        sink.send(Ok(ChatStreamChunk {
            text: result.text.clone(),
            done: true,
            finish_reason: Some(result.finish_reason.clone()),
            tool_calls: Some(result.tool_calls.clone()),
            thinking: result.thinking.clone(),
            thinking_enabled: Some(result.thinking_enabled),
            num_tokens: Some(result.num_tokens),
            prompt_tokens: Some(result.prompt_tokens),
            reasoning_tokens: Some(result.reasoning_tokens),
            reasoning_close: result.reasoning_close.clone(),
            reasoning_cut_at: result.reasoning_cut_at,
            reasoning_cycle_period: result.reasoning_cycle_period,
            reasoning_stray_closes: result.reasoning_stray_closes,
            raw_text: Some(result.raw_text.clone()),
            public_raw_text: result.public_raw_text.clone(),
            text_authoritative: Some(true),
            cached_tokens: Some(result.cached_tokens),
            performance: result.performance.clone(),
            is_reasoning: None,
        }));
    }
}

fn finalize_inkling2_turn(args: FinalizeArgs<'_>) -> Result<ChatResult> {
    let raw_text = args.tokenizer.decode_sync(args.generated_tokens, false)?;
    let parsed = parse_inkling2_output(&raw_text);
    let reasoning_tokens = count_inkling2_reasoning_tokens(args.generated_tokens) as u32;
    Ok(ChatResult {
        text: parsed.text,
        tool_calls: parsed.tool_calls,
        thinking: if args.include_reasoning {
            parsed.thinking
        } else {
            None
        },
        thinking_enabled: args.thinking_enabled,
        num_tokens: args.generated_tokens.len() as u32,
        prompt_tokens: args.prompt_tokens,
        reasoning_tokens,
        reasoning_close: None,
        reasoning_cut_at: None,
        reasoning_cycle_period: None,
        reasoning_stray_closes: None,
        finish_reason: args.finish_reason.to_string(),
        raw_text,
        public_raw_text: None,
        cached_tokens: 0,
        performance: args.performance,
    })
}

impl ChatBackend for Inkling2Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Inkling2 tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "inkling2"
    }

    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.turn_cancel = flag;
    }

    fn wired_limit_bytes(&self) -> Option<usize> {
        if self.wired.is_some() {
            // `WiredResidency` owns the wired memory lease for the model's life.
            None
        } else {
            let max_ws = crate::stream::WiredLimitContext::get_max_working_set_size();
            (max_ws > 0).then_some(max_ws)
        }
    }

    fn has_live_session(&self) -> bool {
        if let Some(ref rt) = self.paged {
            rt.coordinator.is_live_all(rt.active_seq) && rt.conv_states.contains_key(&rt.active_seq)
        } else {
            self.flat_cache.is_some()
        }
    }

    fn session_eos_id(&self, _tok: &Qwen3Tokenizer) -> Result<u32> {
        Ok(CONTENT_MODEL_END_SAMPLING_ID)
    }

    fn policy(&self) -> crate::engine::ThinkingPolicy {
        crate::engine::ThinkingPolicy::None
    }

    fn render_prompt(
        &self,
        tok: &Qwen3Tokenizer,
        messages: &[ChatMessage],
        config: &ChatConfig,
        _preserve_thinking: bool,
    ) -> Result<Vec<u32>> {
        let effort = if let Some(ref raw_effort) = config.reasoning_effort {
            parse_reasoning_effort(raw_effort).map_err(|e| {
                Error::from_reason(format!("Inkling2 invalid reasoning_effort: {e}"))
            })?
        } else if crate::engine::resolve_enable_thinking(config) == Some(false) {
            0.0
        } else {
            crate::tools::inkling::DEFAULT_REASONING_EFFORT
        };
        let tools = match config.tools.as_deref() {
            Some(defs) => defs
                .iter()
                .map(InklingToolSpec::from_tool_definition)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| {
                    Error::from_reason(format!("Inkling2 tool spec conversion failed: {e}"))
                })?,
            None => Vec::new(),
        };
        let opts = Inkling2RenderOptions {
            reasoning_effort: effort,
            tools,
            ..Default::default()
        };
        render_inkling2_chat_messages_to_ids(messages, tok, &opts)
            .map_err(|e| Error::from_reason(format!("Inkling2 prompt render failed: {e}")))
    }

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        self.prefill_runs.clear();
        self.flat_cache = None;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        let _ = self.retract_open_dspark_wave();
        if let Some(ref mut rt) = self.paged {
            if scope == ResetScope::Command {
                self.dspark_turn_contexts.clear();
                self.scheduled_dspark.clear();
                rt.state_store.clear_ram_and_checkpoints();
                rt.coordinator
                    .release_all_and_purge()
                    .map_err(Error::from_reason)?;
                rt.conv_states.clear();
                rt.prev_conv_snapshots.clear();
                rt.block_boundary_conv.clear();
            } else {
                let seq = rt.active_seq;
                self.dspark_turn_contexts.remove(&seq);
                self.scheduled_dspark.seq_lanes.remove(&seq);
                self.scheduled_dspark.peer_tapes.remove(&seq);
                rt.state_store.clear_seq_checkpoint(seq);
                rt.release_sequence(seq);
            }
        } else {
            self.dspark_turn_contexts.clear();
            self.scheduled_dspark.clear();
        }
        Ok(())
    }

    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        if !reuse_cache {
            return 0;
        }
        let cached = &self.cached_token_history;
        if cached.is_empty() || tokens.len() <= cached.len() || tokens[..cached.len()] != cached[..]
        {
            return 0;
        }
        if let Some(ref rt) = self.paged {
            if rt.coordinator.can_continue_all(rt.active_seq, tokens)
                && rt.conv_states.contains_key(&rt.active_seq)
            {
                return cached.len();
            }
            return 0;
        }
        if self.flat_cache.is_some() {
            cached.len()
        } else {
            0
        }
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        crate::models::forward::save_flat_token_history(
            args.save_tokens,
            args.generated_tokens,
            args.finish_reason == "length",
            true,
            FinalTokenPolicy::KeepAllOnLength,
            &mut self.cached_token_history,
        );
    }

    fn eval_caches(&self) -> Result<()> {
        if let Some(ref cache) = self.flat_cache {
            cache.eval_state()?;
        }
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        let _ctx = StreamContext::new(stream);
        let mut cache = match self.flat_cache.take() {
            Some(c) => c,
            None => self.new_cache()?,
        };
        let logits_res = self.forward_tokens(prompt_tokens, None, &mut cache);
        self.flat_cache = Some(cache);
        let logits = logits_res?;
        let t = logits.shape()?[0];
        let vocab = logits.shape()?[1];
        logits.slice(&[t - 1, 0], &[t, vocab])
    }

    type Decode<'a>
        = Inkling2FlatDecode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        Ok(Inkling2FlatDecode { model: self })
    }

    fn finalize_turn(&self, args: FinalizeArgs<'_>) -> Result<ChatResult> {
        finalize_inkling2_turn(args)
    }

    fn resolve_params(&self, config: &ChatConfig) -> crate::engine::params::ChatParams {
        let mut p = crate::engine::extract_chat_params(config);
        if let Some(ref draft) = self.dspark {
            let max_draft = draft.config.num_draft_tokens();
            p.mtp_depth = match config.mtp_depth {
                Some(d) => (d.max(1) as usize).min(max_draft),
                None => max_draft,
            };
            p.mtp_adaptive_depth = match config.mtp_adaptive_depth {
                Some(enabled) => enabled,
                None => config.mtp_depth.is_none(),
            };
        }
        p
    }

    fn execution_plan(&self) -> ExecutionPlan {
        let media = if self.vision_encoder.is_some() {
            MediaPlan::with_backend_validation(MediaCapabilities::IMAGES, MediaCapabilities::IMAGES)
        } else {
            MediaPlan::NONE
        };
        let speculative = if self.dspark.is_some() && self.paged.is_some() {
            Some(SpeculativePlan {
                kind: SpeculativeKind::DraftModel,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::NONE,
                supports_paged_attention: true,
                supports_streaming: true,
            })
        } else {
            None
        };
        ExecutionPlan {
            media,
            paged_attention: self.paged.as_ref().map(|_| PagedAttentionPlan {
                supports_delta: true,
            }),
            speculative,
        }
    }

    fn stream_skip_special_tokens(&self) -> bool {
        false
    }

    fn stream_emitter(&self) -> Box<dyn StreamEmitter> {
        Box::new(Inkling2StreamEmitter::new())
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        match args.plan.decoder {
            crate::engine::plan::DecoderPlan::Speculative(_) => {
                crate::engine::dspark_turn::run_paged_dspark_turn(self, args)
            }
            crate::engine::plan::DecoderPlan::Autoregressive => run_paged_turn(self, args),
        }
    }

    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        crate::engine::dspark_turn::run_paged_dspark_turn(self, args)
    }

    /// Execute an image-bearing turn on the exclusive barrier lane:
    /// encodes all images with the PE-Lang ViT, expands `IMAGE_PAD_ID` placeholders, prefills on
    /// the block-paged hybrid pools (or flat fallback), runs the shared decode loop, and finalizes.
    fn run_multimodal_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        let vision = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 vision encoder is not loaded"))?;
        let mut emb_chunks = Vec::with_capacity(args.media.images.len());
        let mut grids = Vec::with_capacity(args.media.images.len());
        for img_bytes in args.media.images {
            let (emb, grid_rows, grid_cols) = vision.encode_image_bytes(img_bytes)?;
            emb.eval();
            emb_chunks.push(emb);
            grids.push((grid_rows, grid_cols));
        }
        let all_emb = if emb_chunks.is_empty() {
            None
        } else if emb_chunks.len() == 1 {
            emb_chunks.pop()
        } else {
            let refs: Vec<&MxArray> = emb_chunks.iter().collect();
            Some(MxArray::concatenate_many(refs, Some(0))?)
        };
        let image_token_counts: Vec<usize> = grids.iter().map(|(r, c)| r * c).collect();
        let expanded_tokens = expand_image_tokens(args.tokens, &image_token_counts)?;
        let prompt_len = expanded_tokens.len();

        let mut profiler = crate::decode_profiler::DecodeProfiler::new(
            if args.sink.is_some() {
                "chat_vision_stream"
            } else {
                "chat_vision"
            },
            "inkling2",
        );
        let generation_stream = Stream::generation();
        let gen_start = if args.params.report_performance {
            Some(Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<Instant> = None;

        let has_paged = self.paged.is_some();
        let turn_res: Result<TurnOutput> = (|| {
            let first_logits = if let Some(ref mut rt) = self.paged {
                let seq_id = rt.active_seq;
                rt.reset_sequence(seq_id, &self.config)?;
                let _ctx = StreamContext::new(generation_stream);
                let (logits_2d, _) =
                    self.paged_prefill_sequence(seq_id, &expanded_tokens, all_emb.as_ref())?;
                logits_2d.squeeze(Some(&[0]))?
            } else {
                let _ctx = StreamContext::new(generation_stream);
                let mut cache = self.new_cache()?;
                let logits_2d =
                    self.forward_tokens(&expanded_tokens, all_emb.as_ref(), &mut cache)?;
                self.flat_cache = Some(cache);
                let t = logits_2d.shape()?[0];
                let vocab = logits_2d.shape()?[1];
                logits_2d
                    .slice(&[t - 1, 0], &[t, vocab])?
                    .squeeze(Some(&[0]))?
            };

            let y = crate::sampling::sample(&first_logits, args.params.sampling_config)?;
            y.eval();
            crate::array::synchronize_and_clear_cache();

            let close = self.think_end_for_turn(args.config, args.tokenizer);
            let mut reasoning_tracker =
                crate::engine::penalties::ReasoningTracker::from_setup(&args.thinking, &close);
            let mut turn_streaming = args.sink.map(|_| {
                crate::engine::decode::TurnStreaming::new(
                    self,
                    args.tokenizer.inner(),
                    args.thinking.enabled,
                    self.stream_skip_special_tokens(),
                )
            });
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut token_history: Vec<u32> = expanded_tokens.clone();
            let mut finish_reason = String::from("length");
            let extra_eos_ids = self.extra_eos_ids();
            let eos_before_emit = self.eos_before_emit();
            let turn_token_observer = self.turn_token_observer();
            let streaming_ctx = turn_streaming
                .as_mut()
                .and_then(|ts| ts.ctx(args.sink, args.cancelled));

            if self.paged.is_some() {
                let mut stepper = self.begin_paged_decode()?;
                crate::engine::decode::run_decode_loop(
                    &mut stepper,
                    crate::engine::decode::DecodeLoopArgs {
                        y,
                        params: args.params,
                        reasoning_tracker: &mut reasoning_tracker,
                        profiler: &mut profiler,
                        max_new_tokens: args.params.max_new_tokens,
                        eos_id: args.eos_id,
                        extra_eos_ids: &extra_eos_ids,
                        eos_before_emit,
                        generated_tokens: &mut generated_tokens,
                        token_history: &mut token_history,
                        finish_reason: &mut finish_reason,
                        first_token_instant: &mut first_token_instant,
                        report_perf: args.params.report_performance,
                        generation_stream,
                        cancel_flag: args.cancelled,
                        turn_token_observer,
                    },
                    streaming_ctx,
                )?;
                if finish_reason == "length"
                    && let Some(&last_tok) = generated_tokens.last()
                {
                    stepper.materialize_final(last_tok)?;
                }
                stepper.end_decode()?;
            } else {
                let turn_setup = TurnSetup {
                    params: args.params,
                    total_seq_len: prompt_len,
                    is_delta: false,
                };
                let mut stepper = self.begin_decode(&turn_setup)?;
                crate::engine::decode::run_decode_loop(
                    &mut stepper,
                    crate::engine::decode::DecodeLoopArgs {
                        y,
                        params: args.params,
                        reasoning_tracker: &mut reasoning_tracker,
                        profiler: &mut profiler,
                        max_new_tokens: args.params.max_new_tokens,
                        eos_id: args.eos_id,
                        extra_eos_ids: &extra_eos_ids,
                        eos_before_emit,
                        generated_tokens: &mut generated_tokens,
                        token_history: &mut token_history,
                        finish_reason: &mut finish_reason,
                        first_token_instant: &mut first_token_instant,
                        report_perf: args.params.report_performance,
                        generation_stream,
                        cancel_flag: args.cancelled,
                        turn_token_observer,
                    },
                    streaming_ctx,
                )?;
            }

            // Hash image payloads so session media tracking stays distinct from text-only turns.
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};
            for img in args.media.images {
                img.hash(&mut hasher);
            }
            self.cached_image_key = Some(hasher.finish());

            if self.paged.is_some() {
                if let Some(ref rt) = self.paged {
                    let seq_id = rt.active_seq;
                    self.dspark_turn_contexts.remove(&seq_id);
                    self.scheduled_dspark.seq_lanes.remove(&seq_id);
                }
                if args.params.reuse_cache {
                    let finalize_res = if let Some(ref mut rt) = self.paged {
                        let seq_id = rt.active_seq;
                        rt.coordinator
                            .eval_pending_pool_writes_all()
                            .and_then(|()| {
                                rt.coordinator.finalize_keep_live_all(
                                    seq_id,
                                    &[],
                                    args.params.cache_salt,
                                )
                            })
                    } else {
                        Ok(())
                    };
                    if let Err(error) = finalize_res {
                        tracing::warn!(
                            target: "mlx_core::inkling2::paged",
                            "Inkling2 multimodal paged finalize failed: {error}"
                        );
                        self.abort_paged_turn();
                    } else {
                        let hist_tokens = FinalTokenPolicy::KeepAllOnLength
                            .history_tokens(&generated_tokens, finish_reason == "length");
                        self.cached_token_history = expanded_tokens;
                        self.cached_token_history.extend_from_slice(hist_tokens);
                    }
                } else {
                    self.abort_paged_turn();
                }
            } else {
                self.cached_token_history = expanded_tokens;
                self.cached_token_history
                    .extend_from_slice(&generated_tokens);
            }

            if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
                let full_text = args
                    .tokenizer
                    .decode_sync(&generated_tokens, self.stream_skip_special_tokens())
                    .unwrap_or_else(|e| {
                        tracing::warn!("Failed to decode generated tokens: {}", e);
                        String::new()
                    });
                ts.flush_residual(&full_text, args.params.include_reasoning, sink);
            }
            let performance = crate::engine::finalize::compute_performance_metrics(
                gen_start,
                first_token_instant,
                prompt_len,
                generated_tokens.len(),
            );
            let result = self.finalize_turn(FinalizeArgs {
                tokenizer: args.tokenizer,
                generated_tokens: &generated_tokens,
                finish_reason,
                think_end_id: close.think_end_id,
                think_end_str: close.think_end_str.as_deref(),
                think_end_extra_ids: &close.think_end_extra_ids,
                prompt_tokens: prompt_len as u32,
                include_reasoning: args.params.include_reasoning,
                thinking_enabled: args.thinking.enabled,
                performance,
                reasoning: reasoning_tracker.outcome(),
            })?;
            if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
                ts.emitter.finish(&result, sink);
                Ok(TurnOutput::Streamed)
            } else {
                Ok(TurnOutput::Complete(Box::new(result)))
            }
        })();
        if turn_res.is_err() {
            if has_paged {
                self.abort_paged_turn();
            } else {
                self.flat_cache = None;
                self.cached_token_history.clear();
                self.cached_image_key = None;
            }
        }
        turn_res
    }
}

impl PagedBackend for Inkling2Inner {
    type PagedDecode<'a>
        = PagedStepper<Inkling2PagedDecode<'a>>
    where
        Self: 'a;
    type PrefixState = Inkling2PrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        reuse_cache: bool,
        _block_size: usize,
        _extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        let seq_id = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?
            .active_seq;
        self.prepare_prefix_sync(seq_id, plan, reuse_cache, cache_salt)
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        _prefix: &Self::PrefixState,
        stream: Stream,
    ) -> Result<MxArray> {
        let _ctx = StreamContext::new(stream);
        let seq_id = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?
            .active_seq;
        // AR prefill advances target KV without tapping DSpark hidden states; drop any stale
        // draft context on `seq_id` so a future DSpark turn cannot attach to it.
        self.dspark_turn_contexts.remove(&seq_id);
        self.scheduled_dspark.seq_lanes.remove(&seq_id);
        let (logits, _) = self.paged_prefill_sequence(seq_id, suffix_tokens, None)?;
        // `run_paged_turn` expects `[vocab]` (or `[1, vocab]`; `squeeze([0])` matches gemma4).
        logits.squeeze(Some(&[0]))
    }

    fn begin_paged_decode(&mut self) -> Result<Self::PagedDecode<'_>> {
        Ok(PagedStepper(Inkling2PagedDecode {
            model: self,
            pending_cache_error: None,
        }))
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool, cache_salt: u64) {
        let Some(ref mut rt) = self.paged else {
            return;
        };
        rt.paged_finalize_error = None;
        let seq_id = rt.active_seq;
        let res = (|| -> std::result::Result<(), String> {
            rt.coordinator.activate_request_all(seq_id)?;
            rt.coordinator.eval_pending_pool_writes_all()?;
            if reuse_cache {
                rt.coordinator
                    .finalize_keep_live_all(seq_id, &[], cache_salt)?;

                // Capture turn-end ordinary state (`StateClass::Ordinary`) into the hybrid
                // StateStore (40 SWA trailing blocks + 48 ShortConv states) anchored on the exact
                // deepest complete `block_size` boundary where ShortConv was snapshotted!
                if self.cached_image_key.is_none() {
                    let recorded = rt.coordinator.request_token_count_all(seq_id).unwrap_or(0);
                    let bs = rt.coordinator.block_size().max(1);
                    if let Some((boundary, conv)) = rt.block_boundary_conv.get(&seq_id).cloned()
                        && boundary > 0
                        && boundary <= recorded
                        && boundary.is_multiple_of(bs)
                        && let Some(tokens) = rt
                            .coordinator
                            .full_adapter()
                            .request_tokens_for(seq_id)
                            .map(|t| t.to_vec())
                        && tokens.len() >= boundary as usize
                        && let Err(err) = rt.state_store.capture_boundary(
                            &mut rt.coordinator,
                            &conv,
                            seq_id,
                            &tokens,
                            boundary,
                            cache_salt,
                            StateClass::Ordinary,
                        )
                    {
                        tracing::warn!(
                            target: "mlx_core::inkling2::paged",
                            "Inkling2 finalize StateStore capture at boundary={boundary} failed: {err}"
                        );
                    }
                }
            } else {
                rt.state_store.clear_seq_checkpoint(seq_id);
                rt.coordinator.release_request_all(seq_id)?;
                rt.conv_states.remove(&seq_id);
                rt.prev_conv_snapshots.remove(&seq_id);
                rt.block_boundary_conv.remove(&seq_id);
            }
            Ok(())
        })();
        if !reuse_cache {
            self.dspark_turn_contexts.remove(&seq_id);
            self.scheduled_dspark.seq_lanes.remove(&seq_id);
        }
        if let Err(error) = res {
            tracing::warn!(
                target: "mlx_core::inkling2::paged",
                "Inkling2 paged finalize failed: {error}"
            );
            rt.paged_finalize_error = Some(error);
            rt.release_sequence(seq_id);
            self.dspark_turn_contexts.remove(&seq_id);
            self.scheduled_dspark.seq_lanes.remove(&seq_id);
        }
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        if let Some(reason) = self
            .paged
            .as_ref()
            .and_then(|rt| rt.paged_finalize_error.as_deref())
        {
            self.cached_token_history.clear();
            return Err(Error::from_reason(format!(
                "Inkling2 paged finalize failed ({reason}); refusing to publish reusable history"
            )));
        }
        if reuse_cache {
            let mut full_history = save_tokens.to_vec();
            let hist_tokens = FinalTokenPolicy::KeepAllOnLength.history_tokens(generated, keep_all);
            full_history.extend_from_slice(hist_tokens);
            self.cached_token_history = full_history;
        } else {
            self.cached_token_history.clear();
        }
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        keep_all: bool,
    ) -> bool {
        let Some(ref mut rt) = self.paged else {
            return true;
        };
        let seq_id = rt.active_seq;
        if rt.coordinator.activate_request_all(seq_id).is_err() {
            return false;
        }
        let recorded_len = rt.coordinator.full_adapter().request_tokens().len();
        let target_len = prompt_len
            + FinalTokenPolicy::KeepAllOnLength
                .history_tokens(generated, keep_all)
                .len();
        let surplus = recorded_len.saturating_sub(target_len);
        if surplus == 0 {
            return true;
        }
        // Only roll back if we can also restore the exact ShortConv snapshot at `target_len`.
        let conv_snap = rt
            .prev_conv_snapshots
            .get(&seq_id)
            .and_then(|(snap_pos, st)| (*snap_pos as usize == target_len).then(|| st.clone()));
        let Some(snap) = conv_snap else {
            return false;
        };
        let rollback_res = reconcile_paged_surplus(
            recorded_len,
            prompt_len,
            generated.len(),
            keep_all,
            FinalTokenPolicy::KeepAllOnLength,
            |n| rt.coordinator.rollback_last_tokens_all(seq_id, n),
        );
        if rollback_res.is_ok() {
            rt.conv_states.insert(seq_id, snap);
            true
        } else {
            false
        }
    }

    fn abort_paged_turn(&mut self) {
        if let Some(ref mut rt) = self.paged {
            let seq = rt.active_seq;
            rt.state_store.clear_seq_checkpoint(seq);
            rt.release_sequence(seq);
            self.dspark_turn_contexts.remove(&seq);
            self.scheduled_dspark.seq_lanes.remove(&seq);
        }
        self.cached_token_history.clear();
        self.cached_image_key = None;
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::path::Path;

    use crate::models::inkling2::kv_cache::Inkling2PagedConfig;
    use crate::models::inkling2::persistence::load_from_dir_with_paged_config;
    use crate::tools::inkling::{
        CONTENT_MODEL_END_SAMPLING_ID, Inkling2RenderOptions, InklingMessage, InklingStreamParser,
        render_inkling2_messages_to_ids,
    };
    use crate::transformer::kv_cache::KvFormat;
    use crate::transformer::paged_kv_cache_adapter::PagedRaggedRow;

    use super::*;

    const REAL_MODEL_DIR: &str = "/Users/brooklyn/.mlx-node/models/inkling-2-mini-gguf";

    fn decode_and_parse(
        tokenizer: &Qwen3Tokenizer,
        prompt_ids: &[u32],
        gen_ids: &[u32],
    ) -> (String, String, String) {
        let mut parser = InklingStreamParser::from_prompt_tokens(prompt_ids);
        for &tok in gen_ids {
            let piece = tokenizer.decode_sync(&[tok], false).unwrap_or_default();
            let _ = parser.push_token(tok, &piece);
        }
        let _ = parser.flush();
        let raw = tokenizer.decode_sync(gen_ids, false).unwrap_or_default();
        (
            raw,
            parser.thinking().unwrap_or_default(),
            parser.content().to_string(),
        )
    }

    fn token_match_stats(reference: &[u32], candidate: &[u32]) -> (usize, usize, usize) {
        let n = reference.len().min(candidate.len());
        let mut exact = 0usize;
        let mut prefix_match = 0usize;
        let mut prefix_broken = false;
        for i in 0..n {
            if reference[i] == candidate[i] {
                exact += 1;
                if !prefix_broken {
                    prefix_match += 1;
                }
            } else {
                prefix_broken = true;
            }
        }
        (exact, prefix_match, n)
    }

    /// End-to-end real-model parity + throughput gate on `/Users/brooklyn/.mlx-node/models/inkling-2-mini-gguf`:
    /// - Short prompt (< 256 tokens): BF16 KV & INT8 32-token KTM vs flat reference
    /// - Long prompt (> 1,500 tokens, explicitly crossing `sliding_window = 1024`):
    ///   verifies SWA block pruning, relative bias at `d >= 1024`, BF16 parity & INT8 KTM parity
    /// - Batched `[2, 1]` decode parity & 8-row ragged verify parity
    /// - Logs prefill tok/s and decode tok/s on M5 Max!
    #[tokio::test]
    #[ignore = "loads real ~18 GB Inkling-2-Mini GGUF model from disk"]
    async fn inkling2_paged_vs_flat_parity() {
        let model_dir = Path::new(REAL_MODEL_DIR);
        assert!(
            model_dir.exists(),
            "Real model directory missing: {}",
            model_dir.display()
        );

        let t0 = Instant::now();
        // Start with default INT8 32-token KTM config (sized for 4096 tokens so switching formats
        // during the test stays fast and lightweight in memory).
        let mut model = load_from_dir_with_paged_config(
            REAL_MODEL_DIR,
            Some(Inkling2PagedConfig {
                kv_format: KvFormat::Int8,
                block_size: 32,
                max_model_len: 4096,
                max_chunk: 2048,
                max_concurrent_sequences: 2,
                gpu_memory_mb: None,
            }),
        )
        .await
        .expect("load Inkling-2-Mini GGUF");
        eprintln!(
            "[inkling2_paged_vs_flat_parity] Loaded model in {:.2}s (mpp_supported={})",
            t0.elapsed().as_secs_f64(),
            inkling2_paged_prefill_mpp_supported()
        );
        let tokenizer = model
            .tokenizer
            .clone()
            .expect("Expected embedded Inkling2 tokenizer");

        let opts = Inkling2RenderOptions {
            reasoning_effort: 0.1,
            ..Default::default()
        };

        // ------------------------------------------------------------------
        // 1. Short prompt (< 256 tokens)
        // ------------------------------------------------------------------
        let short_msgs = vec![InklingMessage::user("What is 2 + 2? Answer briefly.")];
        let short_prompt = render_inkling2_messages_to_ids(&short_msgs, tokenizer.as_ref(), &opts)
            .expect("render short prompt");
        assert!(
            short_prompt.len() < 256,
            "Expected short prompt < 256 tokens, got {}",
            short_prompt.len()
        );

        let max_new_short = 64usize;
        let flat_short = model
            .generate_greedy(
                &short_prompt,
                None,
                max_new_short,
                &[CONTENT_MODEL_END_SAMPLING_ID],
            )
            .expect("flat short greedy");
        let (flat_short_raw, _, flat_short_content) =
            decode_and_parse(tokenizer.as_ref(), &short_prompt, &flat_short);
        eprintln!(
            "[SHORT flat] prompt={} gen={} content={:?} raw={:?}",
            short_prompt.len(),
            flat_short.len(),
            flat_short_content,
            flat_short_raw
        );

        // Short prompt: INT8 32-token KTM
        let (int8_short, int8_short_perf) = model
            .generate_greedy_paged(
                &short_prompt,
                None,
                max_new_short,
                &[CONTENT_MODEL_END_SAMPLING_ID],
            )
            .expect("paged INT8 short greedy");
        let (int8_short_raw, _, int8_short_content) =
            decode_and_parse(tokenizer.as_ref(), &short_prompt, &int8_short);
        let (int8_s_exact, int8_s_prefix, int8_s_n) = token_match_stats(&flat_short, &int8_short);
        eprintln!(
            "[SHORT INT8-KTM] prompt={} gen={} exact={int8_s_exact}/{int8_s_n} prefix={int8_s_prefix} \
             prefill={:.1} tok/s decode={:.1} tok/s content={:?}",
            short_prompt.len(),
            int8_short.len(),
            int8_short_perf.prefill_tok_per_sec,
            int8_short_perf.decode_tok_per_sec,
            int8_short_content
        );
        assert!(
            int8_short_content.contains('4') || int8_short_raw.contains('4'),
            "INT8 short answer must contain '4': {int8_short_raw:?}"
        );

        // ------------------------------------------------------------------
        // 2. Long prompt (> 1,500 tokens, explicitly crossing sliding_window = 1024!)
        // ------------------------------------------------------------------
        // Place a clear secret code near the beginning (token ~40, well outside SWA=1024
        // by the end of the prompt — ONLY Global attention can retrieve it!) and a second
        // fact near the end (inside SWA=1024).
        let mut long_body =
            String::from("Record the vault key: ALPHA-7391. Below is a long log archive.\n");
        for i in 1..=95 {
            long_body.push_str(&format!(
                "Entry {i:03}: sensor telemetry nominal at sector {s}; temperature 21.{t} C; \
                 pressure 101.{p} kPa; checksum OK.\n",
                s = (i % 12) + 1,
                t = i % 10,
                p = (i * 3) % 10,
            ));
        }
        long_body.push_str(
            "Final instruction: State ONLY the vault key from the very first line of this prompt.",
        );
        let long_msgs = vec![InklingMessage::user(long_body)];
        let long_prompt = render_inkling2_messages_to_ids(&long_msgs, tokenizer.as_ref(), &opts)
            .expect("render long prompt");
        assert!(
            long_prompt.len() > 1500,
            "Expected long prompt > 1500 tokens, got {}",
            long_prompt.len()
        );
        eprintln!(
            "[LONG prompt] {} tokens (crosses sliding_window=1024!)",
            long_prompt.len()
        );

        let max_new_long = 48usize;
        let t_flat_long = Instant::now();
        let flat_long = model
            .generate_greedy(
                &long_prompt,
                None,
                max_new_long,
                &[CONTENT_MODEL_END_SAMPLING_ID],
            )
            .expect("flat long greedy");
        let (flat_long_raw, _, flat_long_content) =
            decode_and_parse(tokenizer.as_ref(), &long_prompt, &flat_long);
        eprintln!(
            "[LONG flat] gen={} in {:.2}s content={:?} raw={:?}",
            flat_long.len(),
            t_flat_long.elapsed().as_secs_f64(),
            flat_long_content,
            flat_long_raw
        );

        // Long prompt: INT8 32-token KTM
        let (int8_long, int8_long_perf) = model
            .generate_greedy_paged(
                &long_prompt,
                None,
                max_new_long,
                &[CONTENT_MODEL_END_SAMPLING_ID],
            )
            .expect("paged INT8 long greedy");
        let (int8_long_raw, _, int8_long_content) =
            decode_and_parse(tokenizer.as_ref(), &long_prompt, &int8_long);
        let (int8_l_exact, int8_l_prefix, int8_l_n) = token_match_stats(&flat_long, &int8_long);
        eprintln!(
            "[LONG INT8-KTM] gen={} exact={int8_l_exact}/{int8_l_n} prefix={int8_l_prefix} \
             swa_pruned={} prefill={:.1} tok/s decode={:.1} tok/s content={:?}",
            int8_long.len(),
            int8_long_perf.swa_blocks_pruned,
            int8_long_perf.prefill_tok_per_sec,
            int8_long_perf.decode_tok_per_sec,
            int8_long_content
        );
        assert!(
            int8_long_perf.swa_blocks_pruned > 0,
            "Expected SWA blocks pruned for >1500 token prompt!"
        );
        assert!(
            int8_long_content.contains("ALPHA-7391") || int8_long_raw.contains("ALPHA-7391"),
            "INT8 long prompt must recall ALPHA-7391 across d>=1024: {int8_long_raw:?}"
        );

        // ------------------------------------------------------------------
        // 3. Switch paged runtime to BF16 (`block_size = 32`) and verify strict parity
        // ------------------------------------------------------------------
        let bf16_cfg = Inkling2PagedConfig {
            kv_format: KvFormat::Bf16,
            block_size: 32,
            max_model_len: 4096,
            max_chunk: 2048,
            max_concurrent_sequences: 2,
            gpu_memory_mb: None,
        };
        model.enable_paged(bf16_cfg).expect("switch to BF16 KV");

        let (bf16_short, bf16_short_perf) = model
            .generate_greedy_paged(
                &short_prompt,
                None,
                max_new_short,
                &[CONTENT_MODEL_END_SAMPLING_ID],
            )
            .expect("paged BF16 short greedy");
        let (bf16_short_raw, _, bf16_short_content) =
            decode_and_parse(tokenizer.as_ref(), &short_prompt, &bf16_short);
        let (bf16_s_exact, bf16_s_prefix, bf16_s_n) = token_match_stats(&flat_short, &bf16_short);
        eprintln!(
            "[SHORT BF16] gen={} exact={bf16_s_exact}/{bf16_s_n} prefix={bf16_s_prefix} \
             prefill={:.1} tok/s decode={:.1} tok/s content={:?}",
            bf16_short.len(),
            bf16_short_perf.prefill_tok_per_sec,
            bf16_short_perf.decode_tok_per_sec,
            bf16_short_content
        );
        assert!(
            bf16_short_content.contains('4') || bf16_short_raw.contains('4'),
            "BF16 short answer must contain '4': {bf16_short_raw:?}"
        );
        // BF16 KV must achieve exact or near-exact greedy token match with flat!
        assert!(
            bf16_s_exact * 100 >= bf16_s_n * 90,
            "BF16 short parity too low: exact={bf16_s_exact}/{bf16_s_n}, flat={flat_short:?}, bf16={bf16_short:?}"
        );

        let (bf16_long, bf16_long_perf) = model
            .generate_greedy_paged(
                &long_prompt,
                None,
                max_new_long,
                &[CONTENT_MODEL_END_SAMPLING_ID],
            )
            .expect("paged BF16 long greedy");
        let (bf16_long_raw, _, bf16_long_content) =
            decode_and_parse(tokenizer.as_ref(), &long_prompt, &bf16_long);
        let (bf16_l_exact, bf16_l_prefix, bf16_l_n) = token_match_stats(&flat_long, &bf16_long);
        eprintln!(
            "[LONG BF16] gen={} exact={bf16_l_exact}/{bf16_l_n} prefix={bf16_l_prefix} \
             swa_pruned={} prefill={:.1} tok/s decode={:.1} tok/s content={:?}",
            bf16_long.len(),
            bf16_long_perf.swa_blocks_pruned,
            bf16_long_perf.prefill_tok_per_sec,
            bf16_long_perf.decode_tok_per_sec,
            bf16_long_content
        );
        assert!(
            bf16_long_perf.swa_blocks_pruned > 0,
            "Expected SWA blocks pruned for BF16 >1500 token prompt!"
        );
        assert!(
            bf16_long_content.contains("ALPHA-7391") || bf16_long_raw.contains("ALPHA-7391"),
            "BF16 long prompt must recall ALPHA-7391 across d>=1024: {bf16_long_raw:?}"
        );
        assert!(
            bf16_l_exact * 100 >= bf16_l_n * 90,
            "BF16 long parity too low: exact={bf16_l_exact}/{bf16_l_n}, flat={flat_long:?}, bf16={bf16_long:?}"
        );

        // ------------------------------------------------------------------
        // 4. Batched `[2, 1]` decode + 8-row ragged verify smoke/parity
        // ------------------------------------------------------------------
        model.reset_paged_sequence(10).expect("reset seq 10");
        model.reset_paged_sequence(11).expect("reset seq 11");
        let (l10, _) = model
            .paged_prefill_sequence(10, &short_prompt, None)
            .expect("prefill seq 10");
        let (l11, _) = model
            .paged_prefill_sequence(11, &short_prompt, None)
            .expect("prefill seq 11");
        let tok10 = l10
            .argmax(-1, Some(false))
            .unwrap()
            .item_at_int32(0)
            .unwrap() as u32;
        let tok11 = l11
            .argmax(-1, Some(false))
            .unwrap()
            .item_at_int32(0)
            .unwrap() as u32;
        assert_eq!(tok10, tok11);

        let batch_logits = model
            .paged_decode_step_batched(&[(10, tok10), (11, tok11)])
            .expect("batched [2, 1] decode");
        assert_eq!(batch_logits.shape().unwrap()[0], 2);
        let batch_argmax = batch_logits
            .argmax(-1, Some(false))
            .unwrap()
            .to_int32()
            .unwrap();
        assert_eq!(
            batch_argmax[0], batch_argmax[1],
            "Identical sequences in batched [2, 1] decode must agree"
        );
        assert_eq!(
            batch_argmax[0] as u32, bf16_short[1],
            "Batched [2, 1] decode token must match single-sequence paged decode"
        );
        let _ = model.settle_paged_step(10);
        let _ = model.settle_paged_step(11);

        // 8-row varlen verify test (using SpecPagedCache!)
        let verify_len = 8usize.min(bf16_short.len().saturating_sub(2));
        if verify_len >= 2 {
            let verify_toks = bf16_short[2..2 + verify_len].to_vec();
            let first_pos = model
                .paged_coordinator()
                .unwrap()
                .request_token_count_all(10)
                .unwrap();
            let conv_snap = model.snapshot_paged_conv_state(10).unwrap();
            let ticket = {
                use crate::engine::spec_paged::SpecPagedCache;
                let coord = model.paged_coordinator_mut().unwrap();
                assert!(coord.reserve_lookahead(10, verify_toks.len()).unwrap());
                coord.record_verify(10, &verify_toks).unwrap()
            };
            let (v_logits, taps) = model
                .paged_verify_ragged(
                    &[(
                        PagedRaggedRow {
                            seq_id: 10,
                            first_logical_position: first_pos,
                            query_len: verify_toks.len() as u32,
                        },
                        verify_toks.clone(),
                    )],
                    &[5, 47],
                )
                .expect("ragged verify");
            assert_eq!(v_logits.shape().unwrap()[0] as usize, verify_toks.len());
            assert_eq!(taps.len(), 2);
            let v_argmax = v_logits
                .argmax(-1, Some(false))
                .unwrap()
                .to_int32()
                .unwrap();
            for i in 0..verify_toks.len().saturating_sub(1) {
                assert_eq!(
                    v_argmax[i] as u32,
                    verify_toks[i + 1],
                    "Verify row {i} greedy argmax should match autoregressive continuation"
                );
            }
            // Rollback verify rows cleanly via SpecPagedCache!
            {
                use crate::engine::spec_paged::SpecPagedCache;
                let coord = model.paged_coordinator_mut().unwrap();
                coord.commit_cycle(10, ticket, 0).unwrap();
                assert_eq!(
                    coord.frontier(10).unwrap().attn_tokens,
                    u64::from(first_pos)
                );
            }
            model.restore_paged_conv_state(10, conv_snap);
            eprintln!(
                "[VERIFY {} rows] matched AR continuation and rolled back cleanly!",
                verify_toks.len()
            );
        }
    }
}

#[cfg(test)]
mod stream_emitter_tests {
    use std::cell::RefCell;

    use serde_json::json;
    use tokenizers::{AddedToken, Tokenizer};

    use super::{Inkling2StreamEmitter, finalize_inkling2_turn};
    use crate::engine::backend::{ChunkSink, FinalizeArgs, StreamEmitter};
    use crate::engine::penalties::ReasoningOutcome;
    use crate::engine::types::{ChatResult, ChatStreamChunk};
    use crate::tokenizer::Qwen3Tokenizer;
    use crate::tools::inkling::{
        CONTENT_INVOKE_TOOL_JSON, CONTENT_INVOKE_TOOL_JSON_ID, CONTENT_MODEL_END_SAMPLING,
        CONTENT_MODEL_END_SAMPLING_ID, CONTENT_TEXT, CONTENT_TEXT_ID, CONTENT_THINKING,
        CONTENT_THINKING_ID, END_MESSAGE, END_MESSAGE_ID, INKLING2_CONTROL_TOKENS,
        InklingParserState, MESSAGE_MODEL, MESSAGE_MODEL_ID, MESSAGE_USER_ID,
        parse_inkling2_output, parse_inkling2_output_with_state,
    };

    #[derive(Default)]
    struct VecSink {
        chunks: RefCell<Vec<ChatStreamChunk>>,
    }

    impl VecSink {
        fn chunks(&self) -> Vec<ChatStreamChunk> {
            self.chunks.borrow().clone()
        }
    }

    impl ChunkSink for VecSink {
        fn send(&self, chunk: napi::bindgen_prelude::Result<ChatStreamChunk>) {
            self.chunks
                .borrow_mut()
                .push(chunk.expect("stream chunk should be Ok"));
        }
    }

    fn build_test_tokenizer() -> Qwen3Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (i, ch) in (32u8..=126u8).enumerate() {
            vocab.insert((ch as char).to_string(), json!(i));
        }
        vocab.insert("\n".to_string(), json!(95));
        let tokenizer_json = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Split",
                "pattern": { "Regex": "." },
                "behavior": "Isolated",
                "invert": false
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "?"
            }
        });
        let mut tok = Tokenizer::from_bytes(tokenizer_json.to_string().as_bytes())
            .expect("build WordLevel tokenizer");
        let added: Vec<AddedToken> = INKLING2_CONTROL_TOKENS
            .iter()
            .map(|&s| AddedToken::from(s, true))
            .collect();
        tok.add_special_tokens(added)
            .expect("register Inkling2 special tokens");
        Qwen3Tokenizer::from_tokenizer(tok, None).expect("wrap Qwen3Tokenizer")
    }

    fn make_final_result(raw_text: &str, include_reasoning: bool) -> ChatResult {
        let parsed = parse_inkling2_output(raw_text);
        ChatResult {
            text: parsed.text,
            tool_calls: parsed.tool_calls,
            thinking: if include_reasoning {
                parsed.thinking
            } else {
                None
            },
            thinking_enabled: false,
            num_tokens: 10,
            prompt_tokens: 5,
            reasoning_tokens: 0,
            reasoning_close: None,
            reasoning_cut_at: None,
            reasoning_cycle_period: None,
            reasoning_stray_closes: None,
            finish_reason: "stop".to_string(),
            raw_text: raw_text.to_string(),
            public_raw_text: None,
            cached_tokens: 0,
            performance: None,
        }
    }

    fn collect_streamed(chunks: &[ChatStreamChunk]) -> (String, String, &ChatStreamChunk) {
        let (done_chunk, deltas) = chunks.split_last().expect("expected terminal chunk");
        assert!(done_chunk.done, "last chunk must be done: true");
        assert_eq!(done_chunk.is_reasoning, None);
        assert_eq!(done_chunk.text_authoritative, Some(true));

        let mut reasoning = String::new();
        let mut content = String::new();
        for c in deltas {
            assert!(!c.done, "non-terminal chunk must have done: false");
            assert!(
                !c.text.is_empty(),
                "emitter must not send empty delta chunks"
            );
            match c.is_reasoning {
                Some(true) => reasoning.push_str(&c.text),
                Some(false) => content.push_str(&c.text),
                None => panic!("delta chunk must tag is_reasoning"),
            }
        }
        (reasoning, content, done_chunk)
    }

    fn assert_no_protocol_leaks(text: &str) {
        for marker in INKLING2_CONTROL_TOKENS {
            assert!(
                !text.contains(marker),
                "leaked Inkling2 marker {marker:?} in {text:?}"
            );
        }
    }

    #[test]
    fn thinking_and_answer_respects_include_reasoning_flag() {
        let raw = concat!(
            "<|message_model|><|content_thinking|>Let me add 2 + 2 = 4.<|end_message|>",
            "<|message_model|><|content_text|>The answer is 4.<|end_message|>",
            "<|content_model_end_sampling|>",
        );
        let pieces = [
            MESSAGE_MODEL,
            CONTENT_THINKING,
            "Let me add ",
            "2 + 2 = 4.",
            END_MESSAGE,
            MESSAGE_MODEL,
            CONTENT_TEXT,
            "The answer ",
            "is 4.",
            END_MESSAGE,
            CONTENT_MODEL_END_SAMPLING,
        ];

        // include_reasoning = true: thinking arrives tagged is_reasoning=true, answer as false
        let sink_with = VecSink::default();
        let mut emitter_with = Inkling2StreamEmitter::new();
        for piece in pieces {
            // Note: engine passes is_reasoning=false because ThinkingPolicy::None
            emitter_with.on_token_text(piece, false, true, &sink_with);
        }
        let result_with = make_final_result(raw, true);
        emitter_with.finish(&result_with, &sink_with);

        let chunks_with = sink_with.chunks();
        let (reasoning_with, content_with, done_with) = collect_streamed(&chunks_with);
        assert_eq!(reasoning_with, "Let me add 2 + 2 = 4.");
        assert_eq!(content_with, "The answer is 4.");
        assert_eq!(done_with.text, result_with.text);
        assert_eq!(
            done_with.thinking,
            Some("Let me add 2 + 2 = 4.".to_string())
        );
        assert_no_protocol_leaks(&reasoning_with);
        assert_no_protocol_leaks(&content_with);

        // include_reasoning = false: zero reasoning deltas emitted, answer still streams cleanly
        let sink_without = VecSink::default();
        let mut emitter_without = Inkling2StreamEmitter::new();
        for piece in pieces {
            emitter_without.on_token_text(piece, false, false, &sink_without);
        }
        let result_without = make_final_result(raw, false);
        emitter_without.finish(&result_without, &sink_without);

        let chunks_without = sink_without.chunks();
        let (reasoning_without, content_without, done_without) = collect_streamed(&chunks_without);
        assert!(
            reasoning_without.is_empty(),
            "include_reasoning=false must emit zero reasoning deltas, got {reasoning_without:?}"
        );
        assert_eq!(content_without, "The answer is 4.");
        assert_eq!(done_without.text, result_without.text);
        assert_eq!(done_without.thinking, None);
        assert_no_protocol_leaks(&content_without);
    }

    #[test]
    fn tool_calls_thinking_and_text_never_leak_markers_headers_or_json() {
        let raw = concat!(
            "<|message_model|><|content_thinking|>Need SF weather.<|end_message|>",
            "<|message_model|>get_weather<|content_invoke_tool_json|>",
            r#"{"name":"get_weather","args":{"city":"SF"}}"#,
            "<|end_message|>",
            "<|message_model|><|content_text|>Checking SF weather now.<|end_message|>",
            "<|content_model_end_sampling|>",
        );
        let pieces = [
            MESSAGE_MODEL,
            CONTENT_THINKING,
            "Need SF ",
            "weather.",
            END_MESSAGE,
            MESSAGE_MODEL,
            "get_",
            "weather",
            CONTENT_INVOKE_TOOL_JSON,
            r#"{"name":"get_weather","#,
            r#""args":{"city":"SF"}}"#,
            END_MESSAGE,
            MESSAGE_MODEL,
            CONTENT_TEXT,
            "Checking SF ",
            "weather now.",
            END_MESSAGE,
            CONTENT_MODEL_END_SAMPLING,
        ];

        for include_reasoning in [true, false] {
            let sink = VecSink::default();
            let mut emitter = Inkling2StreamEmitter::new();
            for piece in pieces {
                emitter.on_token_text(piece, false, include_reasoning, &sink);
            }
            let result = make_final_result(raw, include_reasoning);
            emitter.finish(&result, &sink);

            let chunks = sink.chunks();
            let (reasoning, content, done) = collect_streamed(&chunks);
            if include_reasoning {
                assert_eq!(reasoning, "Need SF weather.");
            } else {
                assert!(reasoning.is_empty());
            }
            assert_eq!(content, "Checking SF weather now.");
            assert!(!content.contains("get_weather"));
            assert!(!content.contains(r#""args""#));
            assert_no_protocol_leaks(&reasoning);
            assert_no_protocol_leaks(&content);

            let tool_calls = done.tool_calls.as_ref().expect("terminal tool_calls");
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].status, "ok");
            assert_eq!(tool_calls[0].name, "get_weather");
            assert_eq!(tool_calls[0].arguments, json!({"city": "SF"}));
        }
    }

    #[test]
    fn partial_protocol_tags_split_across_chunks_never_leak() {
        let raw = concat!(
            "<|message_model|><|content_thinking|>Compare a <| b.<|end_message|>",
            "<|message_model|>lookup<|content_invoke_tool_json|>",
            r#"{"name":"lookup","args":{"q":"<|not_a_tag|>"}}"#,
            "<|end_message|>",
            "<|message_model|><|content_text|>Result: x <| y.<|end_message|>",
            "<|content_model_end_sampling|>",
        );

        let expected = parse_inkling2_output(raw);
        for chunk_size in [1usize, 2, 3, 5, 7, 13] {
            for include_reasoning in [true, false] {
                let sink = VecSink::default();
                let mut emitter = Inkling2StreamEmitter::new();
                let bytes = raw.as_bytes();
                let mut i = 0;
                while i < bytes.len() {
                    let end = (i + chunk_size).min(bytes.len());
                    let piece = std::str::from_utf8(&bytes[i..end]).expect("ascii test input");
                    emitter.on_token_text(piece, false, include_reasoning, &sink);
                    i = end;
                }
                let result = make_final_result(raw, include_reasoning);
                emitter.finish(&result, &sink);

                let chunks = sink.chunks();
                let (reasoning, content, done) = collect_streamed(&chunks);
                if include_reasoning {
                    assert_eq!(
                        Some(reasoning),
                        expected.thinking,
                        "chunk_size={chunk_size}"
                    );
                } else {
                    assert!(reasoning.is_empty(), "chunk_size={chunk_size}");
                }
                assert_eq!(content, expected.text, "chunk_size={chunk_size}");
                assert_no_protocol_leaks(&content);
                assert!(!content.contains("lookup"), "chunk_size={chunk_size}");
                assert!(!content.contains("not_a_tag"), "chunk_size={chunk_size}");
                assert_eq!(
                    done.tool_calls.as_ref().expect("tool_calls").len(),
                    expected.tool_calls.len()
                );
            }
        }
    }

    #[test]
    fn residual_and_finish_flush_buffered_tails_with_reasoning_gate() {
        // Case 1: truncated reasoning ending with a partial marker prefix `<|end`
        let truncated_reasoning = "<|message_model|><|content_thinking|>Still thinking<|end";
        for include_reasoning in [true, false] {
            let sink = VecSink::default();
            let mut emitter = Inkling2StreamEmitter::new();
            emitter.on_token_text(
                "<|message_model|><|content_thinking|>Still ",
                false,
                include_reasoning,
                &sink,
            );
            emitter.on_residual("thinking<|end", false, include_reasoning, &sink);
            let result = make_final_result(truncated_reasoning, include_reasoning);
            emitter.finish(&result, &sink);

            let chunks = sink.chunks();
            let (reasoning, content, done) = collect_streamed(&chunks);
            if include_reasoning {
                assert_eq!(reasoning, "Still thinking<|end");
                assert_eq!(done.thinking.as_deref(), Some("Still thinking<|end"));
            } else {
                assert!(
                    reasoning.is_empty(),
                    "finish() flush must still honor include_reasoning=false"
                );
                assert_eq!(done.thinking, None);
            }
            assert!(content.is_empty());
        }

        // Case 2: truncated content ending with a partial marker prefix `<|content`
        let truncated_content = "<|message_model|><|content_text|>Visible tail<|content";
        let sink = VecSink::default();
        let mut emitter = Inkling2StreamEmitter::new();
        emitter.on_token_text(
            "<|message_model|><|content_text|>Visible ",
            false,
            false,
            &sink,
        );
        emitter.on_residual("tail<|content", false, false, &sink);
        let result = make_final_result(truncated_content, false);
        emitter.finish(&result, &sink);

        let chunks = sink.chunks();
        let (reasoning, content, done) = collect_streamed(&chunks);
        assert!(reasoning.is_empty());
        assert_eq!(content, "Visible tail<|content");
        assert_eq!(done.text, "Visible tail<|content");
    }

    #[test]
    fn continue_final_message_initial_state_matches_parser() {
        let prompt_ids = [
            MESSAGE_USER_ID,
            42,
            END_MESSAGE_ID,
            MESSAGE_MODEL_ID,
            CONTENT_TEXT_ID,
            99,
        ];
        let continuation = " is 42.<|end_message|><|content_model_end_sampling|>";
        let expected = parse_inkling2_output_with_state(continuation, InklingParserState::Content);

        let sink = VecSink::default();
        let mut emitter = Inkling2StreamEmitter::from_prompt_tokens(&prompt_ids);
        emitter.on_token_text(" is ", false, false, &sink);
        emitter.on_token_text("42.<|end_message|>", false, false, &sink);
        emitter.on_token_text(CONTENT_MODEL_END_SAMPLING, false, false, &sink);
        let mut result = make_final_result("", false);
        result.text = expected.text.clone();
        emitter.finish(&result, &sink);

        let chunks = sink.chunks();
        let (reasoning, content, done) = collect_streamed(&chunks);
        assert!(reasoning.is_empty());
        assert_eq!(content, " is 42.");
        assert_eq!(content, expected.text);
        assert_eq!(done.text, expected.text);
    }

    #[test]
    fn on_token_id_handles_empty_marker_text_and_flushed_prefix() {
        let sink = VecSink::default();
        let mut emitter = Inkling2StreamEmitter::new();
        // Simulate skip_special_tokens=true / empty marker text AND a flushed prefix
        emitter.on_token_id(MESSAGE_MODEL_ID, "", false, true, &sink);
        emitter.on_token_id(CONTENT_THINKING_ID, "", false, true, &sink);
        emitter.on_token_id(10, "plan ", false, true, &sink);
        // Flushed trailing text from previous token arrives on END_MESSAGE_ID without marker text
        emitter.on_token_id(END_MESSAGE_ID, "done", false, true, &sink);
        emitter.on_token_id(MESSAGE_MODEL_ID, "", false, true, &sink);
        emitter.on_token_id(11, "get_weather", false, true, &sink);
        emitter.on_token_id(CONTENT_INVOKE_TOOL_JSON_ID, "", false, true, &sink);
        emitter.on_token_id(
            12,
            r#"{"name":"get_weather","args":{}}"#,
            false,
            true,
            &sink,
        );
        emitter.on_token_id(END_MESSAGE_ID, "", false, true, &sink);
        emitter.on_token_id(MESSAGE_MODEL_ID, "", false, true, &sink);
        emitter.on_token_id(CONTENT_TEXT_ID, "", false, true, &sink);
        emitter.on_token_id(13, "Answer", false, true, &sink);
        emitter.on_token_id(END_MESSAGE_ID, "!", false, true, &sink);
        emitter.on_token_id(CONTENT_MODEL_END_SAMPLING_ID, "", false, true, &sink);

        let raw = concat!(
            "<|message_model|><|content_thinking|>plan done<|end_message|>",
            "<|message_model|>get_weather<|content_invoke_tool_json|>",
            r#"{"name":"get_weather","args":{}}"#,
            "<|end_message|>",
            "<|message_model|><|content_text|>Answer!<|end_message|>",
            "<|content_model_end_sampling|>",
        );
        let result = make_final_result(raw, true);
        emitter.finish(&result, &sink);

        let chunks = sink.chunks();
        let (reasoning, content, _) = collect_streamed(&chunks);
        assert_eq!(reasoning, "plan done");
        assert_eq!(content, "Answer!");
        assert!(!content.contains("get_weather"));
    }

    #[test]
    fn streamed_deltas_match_finalize_turn_with_real_decode_stream() {
        let tokenizer = build_test_tokenizer();
        let raw_output = concat!(
            "<|message_model|><|content_thinking|>Step 1: check city.<|end_message|>",
            "<|message_model|>get_weather<|content_invoke_tool_json|>",
            r#"{"name":"get_weather","args":{"city":"Tokyo"}}"#,
            "<|end_message|>",
            "<|message_model|><|content_text|>Calling weather tool.<|end_message|>",
            "<|content_model_end_sampling|>",
        );
        let generated_tokens = tokenizer
            .inner()
            .encode(raw_output, false)
            .expect("encode test output")
            .get_ids()
            .to_vec();

        for include_reasoning in [true, false] {
            let sink = VecSink::default();
            let mut emitter = Inkling2StreamEmitter::new();
            let mut decode_stream = tokenizer.inner().decode_stream(false);
            let mut streamed_text_len = 0usize;
            let mut prefix_tokens = Vec::with_capacity(generated_tokens.len());

            for &tid in &generated_tokens {
                prefix_tokens.push(tid);
                let token_text = Qwen3Tokenizer::step_decode_stream(
                    &mut decode_stream,
                    tokenizer.inner(),
                    tid,
                    &prefix_tokens,
                    streamed_text_len,
                );
                streamed_text_len += token_text.len();
                emitter.on_token_id(tid, &token_text, false, include_reasoning, &sink);
            }

            let full_text = tokenizer
                .decode_sync(&generated_tokens, false)
                .expect("decode_sync");
            if full_text.len() > streamed_text_len {
                emitter.on_residual(
                    &full_text[streamed_text_len..],
                    false,
                    include_reasoning,
                    &sink,
                );
            }

            let finalized = finalize_inkling2_turn(FinalizeArgs {
                tokenizer: &tokenizer,
                generated_tokens: &generated_tokens,
                finish_reason: "stop".to_string(),
                think_end_id: None,
                think_end_str: None,
                think_end_extra_ids: &[],
                performance: None,
                include_reasoning,
                thinking_enabled: false,
                prompt_tokens: 12,
                reasoning: ReasoningOutcome::default(),
            })
            .expect("finalize_inkling2_turn");

            emitter.finish(&finalized, &sink);

            let chunks = sink.chunks();
            let (streamed_reasoning, streamed_content, done) = collect_streamed(&chunks);
            assert_eq!(streamed_content, finalized.text);
            assert_eq!(done.text, finalized.text);
            assert_eq!(done.thinking, finalized.thinking);
            let done_calls = done.tool_calls.as_deref().expect("terminal tool_calls");
            assert_eq!(done_calls.len(), finalized.tool_calls.len());
            for (got, want) in done_calls.iter().zip(finalized.tool_calls.iter()) {
                assert_eq!(got.status, want.status);
                assert_eq!(got.name, want.name);
                assert_eq!(got.arguments, want.arguments);
                assert_eq!(got.raw_content, want.raw_content);
            }
            if include_reasoning {
                assert_eq!(Some(streamed_reasoning), finalized.thinking);
            } else {
                assert!(streamed_reasoning.is_empty());
                assert_eq!(finalized.thinking, None);
            }
            assert_no_protocol_leaks(&streamed_content);
            assert!(!streamed_content.contains("get_weather"));
            assert!(!streamed_content.contains("Tokyo"));
        }
    }
}
