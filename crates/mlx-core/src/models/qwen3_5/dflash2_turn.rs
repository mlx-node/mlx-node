//! Family-generic DFlash2 whole turn shared by the dense and MoE Qwen3.5
//! targets: prefix reuse, tapped prefill, target KV reservation, decode,
//! fail-closed reset and the final-token context append. The speculative
//! steppers stay per family.

use std::borrow::Cow;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, DsparkBackend, FinalizeArgs, ResetScope, TurnOutput, WholeTurnArgs,
};
use crate::engine::decode::TurnStreaming;
use crate::engine::dspark_turn::{DsparkTurnArgs, run_dspark_turn};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::gen_prompt::replay_point;
use crate::engine::params::generated_capacity_hint;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::engine::types::ChatConfig;
use crate::models::forward::PREFILL_CANCELLED;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::KvFormat;

use super::config::Qwen3_5Config;
use super::dflash2::{DFlash2ContextCache, DFlash2Model};
use super::dflash2_decode::{
    constrain_dflash2_context_params, dflash2_decode_cache_limit, dflash2_final_token_fits_context,
    dflash2_kv_budget_of, dflash2_kv_headroom_bytes, dflash2_kv_headroom_for,
    dflash2_kv_need_bytes, dflash2_kv_reserve_rows, dflash2_memory_snapshot,
    dflash2_turn_transient_bytes, kv_row_bytes, min_fa_capacity, reserve_dflash2_target_kv,
    reusable_dflash2_prefix,
};
use super::dflash2_state::flat::{
    FlatAnchor, FlatBacking, FlatStates, flat_attention_kv, join_flat_anchor, join_flat_caches,
    split_flat_caches,
};
use super::dflash2_state::store::{
    Anchor, Backing, EntryParts, EvictReport, HybridStateStore, Lineage, Reservation, StoreHit,
};
use super::dflash2_state::trace::{
    DropKind, Hit, HitKind, LedgerTurn, LiveView, PrefixTracker, StateDesign, StoreView,
    TurnClassification, TurnObservation, design_captures,
};
use super::gated_delta_net::{GdnCapturePlan, GdnCell};
use super::layer_cache::Qwen3_5LayerCache;

pub(crate) struct DFlash2TurnState {
    pub(crate) context: DFlash2ContextCache,
    pub(crate) next_position: i32,
}

/// A capture armed for one prefill at absolute prompt row `row`: the draft
/// window snapshot and every GDN layer's state after rows `[.., row)`. The
/// prefill fills both only when it succeeds and `row` lies in
/// `(position_base, position_base + tokens]`.
pub(crate) struct PrefillCapture {
    pub(crate) row: usize,
    pub(crate) window: Option<DFlash2ContextCache>,
    pub(crate) cell: Option<GdnCell>,
}

impl PrefillCapture {
    pub(crate) fn at(row: usize) -> Self {
        Self {
            row,
            window: None,
            cell: None,
        }
    }

    /// The capture as the anchor of the lineage it was taken on.
    fn into_anchor(self) -> Option<Anchor<FlatAnchor>> {
        let (window, cell) = (self.window?, self.cell?);
        Some(Anchor {
            row: self.row,
            state: FlatAnchor { cell, window },
        })
    }
}

/// First prompt row whose draft K/V a prefill of `[resume, end)` projects:
/// each layer's sliding cache keeps only the newest `window_rows` rows, so
/// older rows would be evicted unread and only advance length and
/// provenance (`record_only`). A capture armed at `capture > resume` needs
/// the rows its snapshot reads, `[capture - window_rows, capture)`.
pub(crate) fn draft_keep_from(
    resume: usize,
    end: usize,
    window_rows: usize,
    capture: Option<usize>,
) -> usize {
    let kept_end = match capture {
        Some(row) if row > resume => row.min(end),
        _ => end,
    };
    kept_end.saturating_sub(window_rows).max(resume)
}

/// A Qwen3.5-family target model that runs DFlash2 turns on its flat caches.
pub(crate) trait DFlash2Target: ChatBackend {
    /// Chunk length of the tapped prefill.
    const PREFILL_STEP_SIZE: i64;

    /// Model name used in error messages.
    fn label(&self) -> &'static str;
    fn dflash2(&self) -> Option<&DFlash2Model>;
    fn caches(&self) -> &Option<Vec<Qwen3_5LayerCache>>;
    fn caches_mut(&mut self) -> &mut Option<Vec<Qwen3_5LayerCache>>;
    fn dflash2_context(&self) -> Option<&DFlash2ContextCache>;
    fn dflash2_context_mut(&mut self) -> &mut Option<DFlash2ContextCache>;
    fn dflash2_turn_state_mut(&mut self) -> &mut Option<DFlash2TurnState>;
    fn prefix_tracker(&self) -> &PrefixTracker;
    fn prefix_tracker_mut(&mut self) -> &mut PrefixTracker;
    fn dflash2_states(&self) -> &FlatStates;
    fn dflash2_states_mut(&mut self) -> &mut FlatStates;
    fn turn_cancel(&self) -> Option<&AtomicBool>;
    fn kv_heads(&self) -> i32;
    fn head_dim(&self) -> i32;
    fn max_position_embeddings(&self) -> i32;
    /// Dense-shaped config the decode-phase cache limit is sized from.
    fn dense_config(&self) -> Cow<'_, Qwen3_5Config>;
    fn set_cached_token_history(&mut self, history: Vec<u32>);
    /// KV row format the config selects for freshly allocated caches.
    fn config_kv_format(&self) -> KvFormat;
    /// Embedding output dtype: what an unallocated BF16 cache will hold.
    fn embedding_dtype(&self) -> Result<DType>;
    /// Prefill-shaped tapped forward: no GDN tape, last-row logits
    /// `[1, 1, vocab]`, plus the residual taps of `tap_layers` and one GDN
    /// cell per `capture` row.
    fn forward_with_taps(
        &mut self,
        input_ids: &MxArray,
        tap_layers: &[usize],
        capture: Option<&GdnCapturePlan<'_>>,
    ) -> Result<(MxArray, Vec<MxArray>, Vec<GdnCell>)>;
    fn init_caches_sync(&mut self) -> Result<()>;
    /// Full cache reset: caches, history, draft context, scheduled recurrent
    /// state, lane flags and the state store.
    fn reset_caches_sync(&mut self) -> Result<()>;
    /// Releases every live paged request and keeps the allocator's prefix
    /// cache.
    fn release_paged_requests(&mut self) -> Result<()>;
    /// Whether the flat caches hold the retained prefix (no paged or MTP
    /// turn has moved past them).
    fn flat_lane_authoritative(&self) -> bool;
    /// Marks the flat caches as holding the retained prefix after a parked
    /// state moved into them.
    fn claim_flat_lane(&mut self);
    /// Runs once the tapped prefill and first sample succeeded: the flat
    /// caches now hold this turn's prompt.
    fn on_flat_prefill_done(&mut self) {}

    fn eval_layer_caches(&self) -> Result<()> {
        crate::models::forward::eval_layer_caches(self.caches())
    }
}

fn turn_cancelled<T: DFlash2Target>(target: &T) -> bool {
    target
        .turn_cancel()
        .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
}

#[cfg(test)]
thread_local! {
    /// Raises the turn's cancel flag when the prefill polls before this
    /// chunk index on this thread.
    pub(crate) static CANCEL_AT_PREFILL_CHUNK: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// When set, collects every `(first row, rows)` append the prefill makes
    /// to the live draft context on this thread.
    pub(crate) static LIVE_DRAFT_APPENDS: std::cell::RefCell<Option<Vec<(usize, usize)>>> =
        const { std::cell::RefCell::new(None) };
}

fn prefill_chunk_cancelled<T: DFlash2Target>(target: &T, chunk: usize) -> bool {
    #[cfg(test)]
    if CANCEL_AT_PREFILL_CHUNK.with(std::cell::Cell::get) == Some(chunk)
        && let Some(flag) = target.turn_cancel()
    {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(test))]
    let _ = chunk;
    turn_cancelled(target)
}

pub(crate) fn dflash2_prefill<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    position_base: i32,
    capture: Option<&mut PrefillCapture>,
    stream: Stream,
) -> Result<(MxArray, DFlash2TurnState)> {
    let label = target.label();
    if tokens.is_empty() {
        return Err(Error::from_reason(format!(
            "{label} DFlash2 requires at least one prefill token"
        )));
    }
    let draft = target
        .dflash2()
        .ok_or_else(|| Error::from_reason(format!("{label} has no loaded DFlash2 companion")))?;
    let tap_layers = draft.config.target_layers.clone();
    let draft_config = draft.config.clone();
    // Polled before the draft context is taken: a cancel here leaves every
    // live buffer as the resume left it.
    if prefill_chunk_cancelled(target, 0) {
        return Err(Error::from_reason(PREFILL_CANCELLED));
    }
    let mut context = match target.dflash2_context_mut().take() {
        Some(context) if context.logical_len() == position_base => context,
        Some(context) => {
            return Err(Error::from_reason(format!(
                "{label} DFlash2 context length {} does not match cached prefix {position_base}",
                context.logical_len()
            )));
        }
        None if position_base == 0 => DFlash2ContextCache::new(&draft_config),
        None => {
            return Err(Error::from_reason(format!(
                "{label} DFlash2 cached prefix {position_base} has no retained draft context"
            )));
        }
    };
    let base_usize = position_base.max(0) as usize;
    let window_rows = draft_config.sliding_window.saturating_sub(1);
    let end_row = base_usize + tokens.len();
    let capture_row = capture
        .as_ref()
        .map(|capture| capture.row)
        .filter(|&row| row > base_usize && row <= end_row);
    // The live context keeps an unarmed prefill's schedule: the append widths
    // set the draft K/V bits on the tiled quantized kernels. When the capture
    // needs rows that schedule only records, a shadow context runs the
    // capture's schedule up to its row and takes the snapshot.
    let keep_from = draft_keep_from(base_usize, end_row, window_rows, None);
    let capture_keep_from = draft_keep_from(base_usize, end_row, window_rows, capture_row);
    let mut shadow = match capture_row {
        Some(_) if capture_keep_from < keep_from => Some(if capture_keep_from == base_usize {
            context.deep_copy()?
        } else {
            let mut shadow = DFlash2ContextCache::new(&draft_config);
            shadow.record_only(context.token_history())?;
            shadow
        }),
        _ => None,
    };
    let mut snapshot = None;
    let mut cell = None;
    let mut offset = 0usize;
    let mut last_logits = None;
    while offset < tokens.len() {
        if offset > 0 && prefill_chunk_cancelled(target, offset / T::PREFILL_STEP_SIZE as usize) {
            return Err(Error::from_reason(PREFILL_CANCELLED));
        }
        let end = (offset + T::PREFILL_STEP_SIZE as usize).min(tokens.len());
        let chunk = tokens[offset..end]
            .iter()
            .map(|&id| id as i32)
            .collect::<Vec<_>>();
        let input = MxArray::from_int32(&chunk, &[1, chunk.len() as i64])?;
        let chunk_end = base_usize + end;
        let chunk_row = capture_row
            .filter(|&row| base_usize + offset < row && row <= chunk_end)
            .map(|row| [row - base_usize - offset]);
        let plan = chunk_row.as_ref().map(|rows| GdnCapturePlan { rows });
        let (logits, taps, cells) = {
            let _stream = StreamContext::new(stream);
            target.forward_with_taps(&input, &tap_layers, plan.as_ref())?
        };
        if plan.is_some() {
            let captured = cells.into_iter().next().ok_or_else(|| {
                Error::from_reason(format!("{label} DFlash2 GDN capture returned no cell"))
            })?;
            // A lazy cell would keep the chunk's graph alive past its eval.
            let arrays: Vec<&MxArray> = captured.conv.iter().chain(&captured.rec).collect();
            if !arrays.is_empty() {
                MxArray::eval_arrays(&arrays)?;
            }
            cell = Some(captured);
        }
        let draft = target
            .dflash2()
            .ok_or_else(|| Error::from_reason(format!("{label} DFlash2 model disappeared")))?;
        let rows = ChunkRows {
            tokens,
            base: base_usize,
            offset,
            end,
        };
        let live_capture = capture_row.filter(|_| shadow.is_none());
        let (window, _appended) =
            advance_draft_context(draft, &mut context, &taps, &rows, keep_from, live_capture)?;
        #[cfg(test)]
        LIVE_DRAFT_APPENDS.with(|log| {
            if let (Some(log), Some(appended)) = (log.borrow_mut().as_mut(), _appended) {
                log.push(appended);
            }
        });
        if window.is_some() {
            snapshot = window;
        }
        if let Some(shadow_context) = shadow.as_mut()
            && let (Some(window), _) = advance_draft_context(
                draft,
                shadow_context,
                &taps,
                &rows,
                capture_keep_from,
                capture_row,
            )?
        {
            snapshot = Some(window);
            shadow = None;
        }
        // `LastRow` span already reduced the logits to the chunk's final
        // row ([1, 1, vocab]); only non-final chunks' rows are dropped.
        let vocab = logits.shape_at(2)?;
        last_logits = Some(logits.reshape(&[vocab])?);
        if end < tokens.len() {
            target.eval_layer_caches()?;
            context.eval()?;
            if let Some(shadow_context) = &shadow {
                shadow_context.eval()?;
            }
            crate::array::clear_cache();
        }
        offset = end;
    }
    if let Some(capture) = capture {
        capture.window = snapshot;
        capture.cell = cell;
    }
    Ok((
        last_logits.ok_or_else(|| Error::from_reason("non-empty DFlash2 prefill"))?,
        DFlash2TurnState {
            context,
            next_position: position_base.saturating_add(tokens.len() as i32),
        },
    ))
}

/// One prefill chunk: prompt rows `[base + offset, base + end)` of a prefill
/// of `tokens` at `base`.
struct ChunkRows<'a> {
    tokens: &'a [u32],
    base: usize,
    offset: usize,
    end: usize,
}

/// Advances `context` over one prefill chunk from its taps: rows below
/// `keep_from` only advance length and provenance, the rest are fused and
/// projected in one append. Returns the window snapshot at `capture` when
/// that row lies in the chunk, and the appended `(first row, rows)`.
fn advance_draft_context(
    draft: &DFlash2Model,
    context: &mut DFlash2ContextCache,
    taps: &[MxArray],
    rows: &ChunkRows<'_>,
    keep_from: usize,
    capture: Option<usize>,
) -> Result<(Option<DFlash2ContextCache>, Option<(usize, usize)>)> {
    let ChunkRows {
        tokens,
        base,
        offset,
        end,
    } = *rows;
    let chunk_end = base + end;
    if chunk_end <= keep_from {
        context.record_only(&tokens[offset..end])?;
        return Ok((None, None));
    }
    let head = keep_from.saturating_sub(base + offset);
    if head > 0 {
        context.record_only(&tokens[offset..offset + head])?;
    }
    // Rows at or past `keep_from` survive in the sliding window: fuse +
    // project only that tail (RoPE keys are per-row at absolute base + row,
    // so a sliced append is identical).
    let fused = if head == 0 {
        draft.fuse_context(taps)?
    } else {
        let tail_taps = taps
            .iter()
            .map(|tap| tap.slice_axis(1, head as i64, (end - offset) as i64))
            .collect::<Result<Vec<_>>>()?;
        draft.fuse_context(&tail_taps)?
    };
    let appended = &tokens[offset + head..end];
    let projected = context.project(draft, &fused, (base + offset + head) as i32)?;
    let snapshot = match capture {
        Some(row) if base + offset < row && row <= chunk_end => {
            Some(context.snapshot_window_at(row as i32, &projected, appended)?)
        }
        _ => None,
    };
    context.append_projected(projected, appended)?;
    Ok((snapshot, Some((base + offset + head, appended.len()))))
}

/// Activation element type the target's full-attention caches hold (BF16
/// format) or quantize from (int8 format): the embedding output dtype,
/// or the allocated BF16 buffer's own.
pub(crate) fn dflash2_target_kv_dtype<T: DFlash2Target>(target: &T) -> Result<DType> {
    let allocated = target
        .caches()
        .iter()
        .flatten()
        .find_map(|cache| match cache {
            Qwen3_5LayerCache::FullAttention(kv) if kv.format() == KvFormat::Bf16 => kv.keys_ref(),
            _ => None,
        });
    match allocated {
        Some(keys) => keys.dtype(),
        None => target.embedding_dtype(),
    }
}

/// Row format of the target's full-attention caches (every FA slot
/// shares the config's format).
pub(crate) fn dflash2_target_kv_format<T: DFlash2Target>(target: &T) -> KvFormat {
    target
        .caches()
        .iter()
        .flatten()
        .find_map(|cache| match cache {
            Qwen3_5LayerCache::FullAttention(kv) => Some(kv.format()),
            Qwen3_5LayerCache::Linear(_) => None,
        })
        .unwrap_or_else(|| target.config_kv_format())
}

#[cfg(test)]
thread_local! {
    /// Stands in for the live KV headroom on this thread.
    pub(crate) static KV_BUDGET_OVERRIDE: std::cell::Cell<Option<u64>> =
        const { std::cell::Cell::new(None) };
}

fn dflash2_kv_headroom_now() -> Option<u64> {
    #[cfg(test)]
    if let Some(bytes) = KV_BUDGET_OVERRIDE.with(std::cell::Cell::get) {
        return Some(bytes);
    }
    dflash2_kv_headroom_bytes(dflash2_memory_snapshot())
}

/// The reservation parked states are evicted for, and the headroom it is
/// measured against: `kv_bytes` of target KV plus `transient_bytes` the turn
/// allocates and frees. With unreadable limits (`headroom` `None`) parked KV
/// has nothing to fit in, so every entry holding KV goes.
pub(crate) fn parked_kv_eviction(
    kv_bytes: u64,
    transient_bytes: u64,
    headroom: Option<u64>,
) -> (Reservation, u64) {
    let kv_bytes = match headroom {
        Some(_) => kv_bytes.saturating_add(transient_bytes),
        None => u64::MAX,
    };
    (
        Reservation {
            state_bytes: 0,
            kv_bytes,
        },
        headroom.unwrap_or(0),
    )
}

/// Evicts parked states until the KV budget read back after the frees covers
/// `need_bytes` and the live headroom also covers `transient_bytes`.
/// `headroom` is the live headroom past the process reserve; with `None`
/// (unreadable limits) every entry holding KV is evicted.
pub(crate) fn evict_for_turn_kv<B: Backing>(
    store: &mut HybridStateStore<B>,
    need_bytes: u64,
    transient_bytes: u64,
    headroom: Option<u64>,
) -> Result<EvictReport> {
    let (need, headroom) = parked_kv_eviction(
        dflash2_kv_headroom_for(need_bytes),
        transient_bytes,
        headroom,
    );
    store.evict_for(need, headroom, None)
}

/// Full-attention K/V bytes of one token row on `target`, and its turn
/// transients for `prompt_len` prompt rows plus `max_new_tokens`.
fn turn_memory<T: DFlash2Target>(
    target: &T,
    prompt_len: usize,
    max_new_tokens: i32,
) -> Result<(u64, u64)> {
    let config = target.dense_config();
    let full_attention_layers = match target.caches() {
        Some(caches) => caches
            .iter()
            .filter(|cache| matches!(cache, Qwen3_5LayerCache::FullAttention(_)))
            .count(),
        None => (0..config.num_layers.max(0) as usize)
            .filter(|&layer| !config.is_linear_layer(layer))
            .count(),
    };
    let format = dflash2_target_kv_format(target);
    let row_bytes = kv_row_bytes(
        full_attention_layers,
        target.kv_heads(),
        target.head_dim(),
        dflash2_target_kv_dtype(target)?,
        format,
    );
    let block_size = target.dflash2().map_or(1, |draft| draft.config.block_size);
    let transient = dflash2_turn_transient_bytes(
        &config,
        block_size,
        prompt_len,
        prompt_len as u64 + max_new_tokens.max(0) as u64,
        format,
    );
    Ok((row_bytes, transient))
}

/// Evicts parked states before a turn of another kind (paged, vision, MTP)
/// takes the model, until the live headroom covers that turn's
/// `prompt + max_new` K/V rows and its transients. The live state stays:
/// its buffers are resident whether it parks or not. A failed eviction
/// drops every parked state.
pub(crate) fn evict_parked_dflash2_for_turn<T: DFlash2Target>(
    target: &mut T,
    prompt_len: usize,
    max_new_tokens: i32,
) {
    if target.dflash2_states().store.bytes().entries == 0 {
        return;
    }
    let evicted =
        turn_memory(target, prompt_len, max_new_tokens).and_then(|(row_bytes, transient)| {
            let rows = prompt_len as u64 + max_new_tokens.max(0) as u64;
            let (need, headroom) = parked_kv_eviction(
                rows.saturating_mul(row_bytes),
                transient,
                dflash2_kv_headroom_now(),
            );
            target
                .dflash2_states_mut()
                .store
                .evict_for(need, headroom, None)
        });
    match evicted {
        Ok(report) if report.freed_kv_bytes > 0 || report.freed_state_bytes > 0 => {
            crate::array::clear_cache();
        }
        Ok(_) => {}
        Err(error) => {
            tracing::warn!(
                target: "mlx_core::inference",
                "DFlash2 parked-state eviction failed, dropping every parked state: {}",
                error.reason
            );
            target.dflash2_states_mut().store.clear();
            crate::array::clear_cache();
        }
    }
}

/// Size every flat full-attention cache for this turn so decode never
/// grows it. Runs before prefill: on a cold turn it only records the
/// size for the first allocation; a warm turn copies its prefix once.
/// Parked states are MLX-active bytes the live budget already excludes, so
/// they are evicted first, until the budget covers the prompt plus the
/// minimum look-ahead and the live headroom also covers the turn's
/// transients.
pub(crate) fn reserve_dflash2_turn_kv<T: DFlash2Target>(
    target: &mut T,
    prompt_len: usize,
    max_new_tokens: i32,
) -> Result<EvictReport> {
    let Some(resident_rows) = min_fa_capacity(target.caches())? else {
        return Ok(EvictReport::default());
    };
    let (row_bytes, transient) = turn_memory(target, prompt_len, max_new_tokens)?;
    let need_bytes = dflash2_kv_need_bytes(prompt_len, max_new_tokens, resident_rows, row_bytes)?;
    let evicted = evict_for_turn_kv(
        &mut target.dflash2_states_mut().store,
        need_bytes,
        transient,
        dflash2_kv_headroom_now(),
    )?;
    if evicted.freed_kv_bytes > 0 || evicted.freed_state_bytes > 0 {
        crate::array::clear_cache();
    }
    let budget_bytes = dflash2_kv_headroom_now().map(dflash2_kv_budget_of);
    let rows = dflash2_kv_reserve_rows(
        prompt_len,
        max_new_tokens,
        resident_rows,
        row_bytes,
        budget_bytes,
    )?;
    tracing::debug!(
        target: "mlx_core::inference",
        prompt_len,
        max_new_tokens,
        resident_rows,
        row_bytes,
        budget_bytes,
        reserved_rows = rows,
        evicted_states = evicted.demoted + evicted.dropped_captures + evicted.dropped_ends,
        freed_kv_bytes = evicted.freed_kv_bytes,
        "DFlash2 target KV reservation"
    );
    if let Some(rows) = rows {
        reserve_dflash2_target_kv(target.caches_mut(), rows)?;
    }
    Ok(evicted)
}

/// Parks the live anchor of a turn cancelled after its prefill started, with
/// the full-attention caches trimmed to its row: the prefill and decode only
/// write rows past the resume point, so an anchor at or below it still
/// resumes exactly. Returns whether the anchor parked; the caller drops the
/// rest of the live state either way.
fn park_cancelled_dflash2_anchor<T: DFlash2Target>(target: &mut T, resume: usize) -> bool {
    let Some(anchor) = target.dflash2_states_mut().live_anchor.take() else {
        return false;
    };
    let row = anchor.row;
    let lineage = anchor.state.window.token_history().to_vec();
    let exact = row > 0
        && row <= resume
        && lineage.len() == row
        && target.flat_lane_authoritative()
        && ChatBackend::cached_token_history(target).get(..row) == Some(lineage.as_slice())
        && flat_attention_frontier(target).is_some_and(|frontier| frontier >= row);
    if !exact {
        return false;
    }
    let Some(caches) = target.caches_mut().take() else {
        return false;
    };
    let mut kv = flat_attention_kv(caches);
    kv.trim(row);
    let salt = target.dflash2_states().live_salt;
    let parts = EntryParts {
        end: None,
        anchor: Some(anchor),
        kv: Some(kv),
    };
    match target
        .dflash2_states_mut()
        .store
        .insert(Lineage::text(&lineage, salt), parts)
    {
        Ok(_) => true,
        Err(error) => {
            tracing::warn!(
                target: "mlx_core::inference",
                "DFlash2 cancel dropped the {row}-token anchor: {error}"
            );
            false
        }
    }
}

/// A turn that failed after its resume was resolved. A user cancel drops
/// only the live state the turn was building, or nothing when the prefill
/// stopped before its first chunk; the live anchor at or below the resume
/// point parks first. Parked states, the paged prefix cache and the paged
/// lane's other requests' blocks stay. Any other error fails closed.
fn dflash2_turn_failed<T: DFlash2Target>(target: &mut T, error: Error, resume: usize) -> Error {
    if error.reason != PREFILL_CANCELLED {
        return dflash2_fail_closed(target, error);
    }
    let untouched = target.dflash2_turn_state_mut().is_none()
        && flat_attention_frontier(target) == Some(resume)
        && target.dflash2_context().is_some() == (resume > 0);
    if untouched {
        return error;
    }
    target.prefix_tracker_mut().take_drop();
    target
        .prefix_tracker_mut()
        .note_drop("dflash2_cancelled", DropKind::Invalidated);
    park_cancelled_dflash2_anchor(target, resume);
    if let Err(drop_error) = drop_live_dflash2_state(target) {
        tracing::warn!(
            target: "mlx_core::inference",
            "DFlash2 cancel could not drop the live state, failing closed: {}",
            drop_error.reason
        );
        return dflash2_fail_closed(target, error);
    }
    error
}

pub(crate) fn dflash2_fail_closed<T: DFlash2Target>(target: &mut T, error: Error) -> Error {
    target.prefix_tracker_mut().take_drop();
    target
        .prefix_tracker_mut()
        .note_drop("dflash2_fail_closed", DropKind::Invalidated);
    let _ = ChatBackend::reset_caches(target, ResetScope::Command);
    *target.dflash2_turn_state_mut() = None;
    error
}

pub(crate) fn dflash2_materialize_final<T: DFlash2Target>(
    target: &mut T,
    token: u32,
    stream: Stream,
) -> Result<()> {
    let label = target.label();
    let tap_layers = target
        .dflash2()
        .ok_or_else(|| Error::from_reason(format!("{label} DFlash2 model disappeared")))?
        .config
        .target_layers
        .clone();
    let input = MxArray::from_int32(&[token as i32], &[1, 1])?;
    let _stream = StreamContext::new(stream);
    let (_, taps, _) = target.forward_with_taps(&input, &tap_layers, None)?;
    let fused = target
        .dflash2()
        .ok_or_else(|| Error::from_reason(format!("{label} DFlash2 model disappeared")))?
        .fuse_context(&taps)?;
    let mut context = target.dflash2_context_mut().take().ok_or_else(|| {
        Error::from_reason(format!(
            "{label} DFlash2 final token has no retained draft context"
        ))
    })?;
    let base = context.logical_len();
    let append_result = target
        .dflash2()
        .ok_or_else(|| Error::from_reason(format!("{label} DFlash2 model disappeared")))
        .and_then(|draft| context.append(draft, &fused, base, &[token]));
    *target.dflash2_context_mut() = Some(context);
    append_result?;
    target.eval_layer_caches()
}

/// Common offset of the flat full-attention caches, or `None` when they
/// disagree (or none are allocated).
pub(crate) fn flat_attention_frontier<T: DFlash2Target>(target: &T) -> Option<usize> {
    let mut frontiers = target.caches().as_ref()?.iter().filter_map(|cache| {
        if matches!(cache, Qwen3_5LayerCache::FullAttention(_)) {
            usize::try_from(cache.offset().max(0)).ok()
        } else {
            None
        }
    });
    let first = frontiers.next()?;
    frontiers.all(|frontier| frontier == first).then_some(first)
}

/// Moves the live flat DFlash2 state out of `target`: every layer cache, the
/// draft window, the token history and the anchor captured on that history.
/// It runs between turns, after the stepper's `finish` handed the window
/// back, so no GDN blob or window writer is alive and the moved buffers are
/// the entry's alone. `None`, with only the live anchor dropped, when the
/// live state is not an exact DFlash2 state of its history.
pub(crate) fn park_flat_dflash2_state<T: DFlash2Target>(
    target: &mut T,
) -> Option<(Vec<u32>, EntryParts<FlatBacking>)> {
    let anchor = target.dflash2_states_mut().live_anchor.take();
    let pending_turn = target.dflash2_turn_state_mut().is_some();
    let history = ChatBackend::cached_token_history(target);
    let parkable = !pending_turn
        && !history.is_empty()
        && target.caches().is_some()
        && target.flat_lane_authoritative()
        && target.dflash2_context().is_some_and(|context| {
            context.token_history() == history
                && usize::try_from(context.logical_len()).ok() == Some(history.len())
        })
        && flat_attention_frontier(target) == Some(history.len());
    if !parkable {
        return None;
    }
    let caches = target.caches_mut().take()?;
    let context = target.dflash2_context_mut().take()?;
    let history = ChatBackend::cached_token_history(target).to_vec();
    target.set_cached_token_history(Vec::new());
    let block_size = target.dflash2_states().store.block_size() as usize;
    let anchor = anchor.filter(|anchor| {
        anchor.row > 0
            && anchor.row <= history.len()
            && anchor.row.is_multiple_of(block_size)
            && anchor.state.window.token_history() == &history[..anchor.row]
    });
    let (kv, end) = split_flat_caches(caches, context);
    Some((
        history,
        EntryParts {
            end: Some(end),
            anchor,
            kv: Some(kv),
        },
    ))
}

/// Parks the live flat DFlash2 state in the model's state store, keyed
/// under the salt of the turn that built it, before a site that would
/// discard or overwrite it. Returns whether the state moved; a state that is
/// not exact stays in place for the caller to drop.
pub(crate) fn park_live_dflash2_state<T: DFlash2Target>(target: &mut T) -> bool {
    let Some((tokens, parts)) = park_flat_dflash2_state(target) else {
        return false;
    };
    let salt = target.dflash2_states().live_salt;
    let states = target.dflash2_states_mut();
    match states.store.insert(Lineage::text(&tokens, salt), parts) {
        Ok(_) => true,
        Err(error) => {
            tracing::warn!(
                target: "mlx_core::inference",
                "DFlash2 state park dropped a {}-token state: {error}",
                tokens.len()
            );
            false
        }
    }
}

/// Moves a parked end-of-turn state back into `target` as its live flat
/// state. The entry's anchor reads rows of this KV, so it becomes the live
/// anchor. The caller parks or drops the live state first, and owns the lane
/// flags: a target whose flat lane is not authoritative still misses on it.
pub(crate) fn install_flat_dflash2_end<T: DFlash2Target>(
    target: &mut T,
    tokens: Vec<u32>,
    parts: EntryParts<FlatBacking>,
) -> Result<()> {
    let label = target.label();
    if target.dflash2_context().is_some() || target.dflash2_turn_state_mut().is_some() {
        return Err(Error::from_reason(format!(
            "{label} DFlash2 parked state would replace a live draft context"
        )));
    }
    let (Some(end), Some(kv)) = (parts.end, parts.kv) else {
        return Err(Error::from_reason(format!(
            "{label} DFlash2 parked entry has no end state to install"
        )));
    };
    let rows = tokens.len();
    if kv
        .layers()
        .iter()
        .any(|(_, cache)| usize::try_from(cache.get_offset()).ok() != Some(rows))
        || end.context().token_history() != tokens.as_slice()
        || usize::try_from(end.next_position()).ok() != Some(rows)
    {
        return Err(Error::from_reason(format!(
            "{label} DFlash2 parked state does not cover its {rows}-token history"
        )));
    }
    let (caches, context) = join_flat_caches(kv, end)?;
    *target.caches_mut() = Some(caches);
    *target.dflash2_context_mut() = Some(context);
    target.set_cached_token_history(tokens);
    target.dflash2_states_mut().live_anchor = parts.anchor;
    Ok(())
}

/// Moves parked entry `X` back into `target` resumed at its anchor C(R): the
/// full-attention caches trimmed to R, the GDN cell and the window snapshot
/// deep-copied in. X's end state is dropped. The anchor stays the live
/// anchor: the rows it reads, `[0, R)`, are the live KV's.
pub(crate) fn install_flat_dflash2_anchor<T: DFlash2Target>(
    target: &mut T,
    tokens: Vec<u32>,
    parts: EntryParts<FlatBacking>,
) -> Result<()> {
    let label = target.label();
    if target.dflash2_context().is_some() || target.dflash2_turn_state_mut().is_some() {
        return Err(Error::from_reason(format!(
            "{label} DFlash2 parked anchor would replace a live draft context"
        )));
    }
    let (Some(anchor), Some(mut kv)) = (parts.anchor, parts.kv) else {
        return Err(Error::from_reason(format!(
            "{label} DFlash2 parked entry has no anchor to install"
        )));
    };
    let row = anchor.row;
    if row == 0
        || row > tokens.len()
        || kv
            .layers()
            .iter()
            .any(|(_, cache)| usize::try_from(cache.get_offset()).map_or(true, |rows| rows < row))
        || anchor.state.window.token_history() != &tokens[..row]
        || usize::try_from(anchor.state.window.logical_len()).ok() != Some(row)
    {
        return Err(Error::from_reason(format!(
            "{label} DFlash2 parked anchor at {row} does not cover its lineage"
        )));
    }
    drop(parts.end);
    kv.trim(row);
    let context = anchor.state.window.deep_copy()?;
    let caches = join_flat_anchor(kv, &anchor.state.cell)?;
    *target.caches_mut() = Some(caches);
    *target.dflash2_context_mut() = Some(context);
    let mut history = tokens;
    history.truncate(row);
    target.set_cached_token_history(history);
    target.dflash2_states_mut().live_anchor = Some(anchor);
    Ok(())
}

/// Drops the live flat state and every live paged request, keeping the state
/// store and the paged prefix cache. A DFlash2 miss replays the cold flat
/// prefill bit for bit without the purge: that prefill reads no paged block.
fn drop_live_dflash2_state<T: DFlash2Target>(target: &mut T) -> Result<()> {
    let store = target.dflash2_states_mut().take_store();
    let reset = target.reset_caches_sync();
    target.dflash2_states_mut().store = store;
    reset?;
    target.release_paged_requests()
}

/// The state a turn resumes from, after the live state was parked or dropped
/// where it does not serve the turn.
struct Resume {
    used: Hit,
    /// The deepest store entry the prompt matched, used or not.
    store: Option<Hit>,
}

/// Resolves the turn's resume point: the live state when it matches,
/// else the deepest exact parked state of `tokens`, moved in, else a miss on
/// fresh caches. A live state the turn does not resume parks first: an
/// earlier turn with reuse on built it. Without reuse nothing is looked up.
/// A parked state that fails to install is dropped and the turn misses.
fn resolve_dflash2_resume<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    live_hit: usize,
    reuse_cache: bool,
    salt: u64,
) -> Result<Resume> {
    let live = (live_hit > 0).then_some(live_hit);
    let miss = |target: &mut T, store: Option<Hit>| -> Result<Resume> {
        drop_live_dflash2_state(target)?;
        target.init_caches_sync()?;
        Ok(Resume {
            used: Hit::MISS,
            store,
        })
    };
    if live.is_none() {
        park_live_dflash2_state(target);
    }
    if !reuse_cache {
        return match live {
            Some(tokens) => Ok(Resume {
                used: Hit {
                    kind: HitKind::Live,
                    tokens,
                },
                store: None,
            }),
            None => miss(target, None),
        };
    }
    let lookup = |target: &T| -> Option<StoreHit> {
        target
            .dflash2_states()
            .store
            .lookup(Lineage::text(tokens, salt), |_| true)
    };
    let found = lookup(target);
    let store = found.map(|hit| Hit {
        kind: hit.kind,
        tokens: hit.depth,
    });
    let Some(hit) = found.filter(|hit| hit.beats_live(live)) else {
        return match live {
            Some(tokens) => Ok(Resume {
                used: Hit {
                    kind: HitKind::Live,
                    tokens,
                },
                store,
            }),
            None => miss(target, store),
        };
    };
    if live.is_some() {
        park_live_dflash2_state(target);
    }
    drop_live_dflash2_state(target)?;
    let Some(entry) = target.dflash2_states_mut().store.take(hit.id) else {
        return miss(target, store);
    };
    let (lineage, parts) = entry.into_parts();
    let installed = match hit.kind {
        HitKind::End => install_flat_dflash2_end(target, lineage, parts),
        _ => install_flat_dflash2_anchor(target, lineage, parts),
    };
    if let Err(error) = installed {
        tracing::warn!(
            target: "mlx_core::inference",
            "DFlash2 parked {}-token state failed to install, prefilling cold: {error}",
            hit.depth
        );
        return miss(target, store);
    }
    target.claim_flat_lane();
    Ok(Resume {
        used: Hit {
            kind: hit.kind,
            tokens: hit.depth,
        },
        store,
    })
}

/// How a turn's prompt met the model's live state, the state store and the
/// turn ledger.
struct PrefixObservation {
    gen_suffix_len: usize,
    replay_point: Option<usize>,
    classification: TurnClassification,
}

/// The live state as the turn found it, before resolution parks or drops it.
struct LiveSeen {
    tokens: Vec<u32>,
    authoritative: bool,
    context_matches: bool,
}

impl LiveSeen {
    fn of<T: DFlash2Target>(target: &T) -> Option<Self> {
        let history = ChatBackend::cached_token_history(target);
        (!history.is_empty() && target.caches().is_some()).then(|| Self {
            tokens: history.to_vec(),
            authoritative: target.flat_lane_authoritative(),
            context_matches: target
                .dflash2_context()
                .is_some_and(|context| context.token_history() == history)
                && flat_attention_frontier(target) == Some(history.len()),
        })
    }
}

struct PrefixSeen {
    gen_suffix_len: usize,
    replay_point: Option<usize>,
    live: Option<LiveSeen>,
    dropped: Option<super::dflash2_state::trace::LiveDrop>,
}

fn see_prefix<T: DFlash2Target>(
    target: &mut T,
    tokenizer: &Qwen3Tokenizer,
    config: &ChatConfig,
    tokens: &[u32],
) -> PrefixSeen {
    let suffix = tokenizer.generation_suffix(config).ok().flatten();
    let replay_point = suffix.as_deref().and_then(|suffix| {
        replay_point(tokens, suffix, target.prefix_tracker().ledger.block_size())
    });
    PrefixSeen {
        gen_suffix_len: suffix.map_or(0, |suffix| suffix.len()),
        replay_point,
        live: LiveSeen::of(target),
        dropped: target.prefix_tracker_mut().take_drop(),
    }
}

fn observe_prefix<T: DFlash2Target>(
    target: &T,
    seen: PrefixSeen,
    tokens: &[u32],
    reuse_cache: bool,
    salt: u64,
    resume: &Resume,
) -> PrefixObservation {
    let evicted_depth = if reuse_cache {
        target
            .dflash2_states()
            .store
            .evicted_depth(Lineage::text(tokens, salt))
    } else {
        0
    };
    let classification = target.prefix_tracker().ledger.classify(&TurnObservation {
        prompt: tokens,
        reuse_cache,
        used: resume.used,
        live: seen.live.as_ref().map(|live| LiveView {
            tokens: &live.tokens,
            authoritative: live.authoritative,
            context_matches: live.context_matches,
        }),
        dropped: seen.dropped,
        store: StoreView {
            deepest: resume.store,
            evicted_depth,
        },
    });
    PrefixObservation {
        gen_suffix_len: seen.gen_suffix_len,
        replay_point: seen.replay_point,
        classification,
    }
}

fn emit_prefix_event<T: DFlash2Target>(
    target: &T,
    observed: &PrefixObservation,
    evicted: &EvictReport,
    prompt_tokens: usize,
    prefilled_tokens: usize,
    ttft_ms: f64,
) {
    let c = &observed.classification;
    let miss_reason = c
        .miss_reason
        .map_or_else(|| String::from("none"), |reason| reason.to_string());
    let store = target.dflash2_states().store.bytes();
    tracing::info!(
        target: "mlx_core::inference",
        event = "dflash2_prefix",
        prompt_tokens,
        cached_tokens = c.hit.tokens,
        prefilled_tokens,
        gen_suffix_len = observed.gen_suffix_len,
        replay_point = observed.replay_point.map_or(-1, |point| point as i64),
        hit_kind = c.hit.kind.as_str(),
        hit_tokens = c.hit.tokens,
        miss_reason = miss_reason.as_str(),
        shadow_hit = c.shadow_hit,
        ideal_tokens = c.ideal,
        diverge_at = c.diverge.index,
        diverge_prompt_token = c.diverge.prompt_token,
        diverge_cached_token = c.diverge.cached_token,
        store_entries = store.entries as u64,
        store_bytes = store.state_bytes.saturating_add(store.kv_bytes),
        evictions = (evicted.demoted + evicted.dropped_captures + evicted.dropped_ends) as u64,
        ttft_ms,
        "DFlash2 prefix"
    );
}

pub(crate) fn dflash2_chat_turn<T: DFlash2Target + DsparkBackend>(
    target: &mut T,
    args: &mut WholeTurnArgs<'_>,
) -> Result<TurnOutput> {
    let turn_start = Instant::now();
    let tokenizer = args.tokenizer.clone();
    let tokens = args.tokens.to_vec();
    let is_delta = args.plan.is_delta;
    let is_streaming = args.sink.is_some();
    let mut params = ChatBackend::resolve_params(target, args.config);
    params.extra_eos_ids = ChatBackend::extra_eos_ids(target);
    let dflash_context_capacity = constrain_dflash2_context_params(
        tokens.len(),
        target.max_position_embeddings(),
        target
            .dflash2()
            .ok_or_else(|| Error::from_reason("loaded DFlash2 companion"))?
            .max_position_embeddings(),
        &mut params,
    )?;
    let prior_cached = if is_delta {
        ChatBackend::cached_token_history(target).len()
    } else {
        0
    };
    let verified_hit = if is_delta {
        0
    } else {
        ChatBackend::verify_cache_prefix(target, &tokens, params.reuse_cache)
    };
    let live_hit = reusable_dflash2_prefix(
        is_delta,
        &tokens,
        prior_cached,
        verified_hit,
        target
            .dflash2_context()
            .map(DFlash2ContextCache::token_history),
        flat_attention_frontier(target),
        target.flat_lane_authoritative() && target.dflash2_states().live_salt == params.cache_salt,
    );
    if turn_cancelled(target) {
        return Err(Error::from_reason(PREFILL_CANCELLED));
    }
    let seen = see_prefix(target, &tokenizer, args.config, &tokens);
    let resume = match resolve_dflash2_resume(
        target,
        &tokens,
        live_hit,
        params.reuse_cache,
        params.cache_salt,
    ) {
        Ok(resume) => resume,
        Err(error) => return Err(dflash2_fail_closed(target, error)),
    };
    let cached_prefix = resume.used.tokens;
    target.dflash2_states_mut().live_salt = params.cache_salt;
    let observed = observe_prefix(
        target,
        seen,
        &tokens,
        params.reuse_cache,
        params.cache_salt,
        &resume,
    );
    let prefill = tokens[cached_prefix..].to_vec();
    let mut capture = observed
        .replay_point
        .filter(|&row| params.reuse_cache && row > cached_prefix)
        .map(PrefillCapture::at);

    let generation_stream = Stream::generation();
    let report_performance = params.report_performance;
    let generation_start = report_performance.then(Instant::now);
    let mut first_token_instant = None;
    let mut profiler = DecodeProfiler::new(
        ChatBackend::profiler_label(target, is_delta, is_streaming),
        ChatBackend::family_name(target),
    );
    profiler.set_prompt_tokens(prefill.len() as u32);
    profiler.snapshot_memory_before();
    let reserved = {
        let _stream = StreamContext::new(generation_stream);
        reserve_dflash2_turn_kv(target, tokens.len(), params.max_new_tokens)
    };
    let evicted = match reserved {
        Ok(evicted) => evicted,
        Err(error) => return Err(dflash2_turn_failed(target, error, cached_prefix)),
    };
    profiler.begin_prefill();
    let (last_logits, state) = match dflash2_prefill(
        target,
        &prefill,
        cached_prefix as i32,
        capture.as_mut(),
        generation_stream,
    ) {
        Ok(value) => value,
        Err(error) => return Err(dflash2_turn_failed(target, error, cached_prefix)),
    };
    profiler.end_prefill();

    let mut token_history = tokens.clone();
    let y = match apply_all_penalties(last_logits, &token_history, &params)
        .and_then(|logits| crate::sampling::sample(&logits, params.sampling_config))
    {
        Ok(token) => token,
        Err(error) => return Err(dflash2_turn_failed(target, error, cached_prefix)),
    };
    y.eval();
    if let Err(error) = target.eval_layer_caches() {
        return Err(dflash2_turn_failed(target, error, cached_prefix));
    }
    let first_token_at = Instant::now();
    if report_performance {
        first_token_instant = Some(first_token_at);
    }
    emit_prefix_event(
        target,
        &observed,
        &evicted,
        tokens.len(),
        prefill.len(),
        first_token_at.duration_since(turn_start).as_secs_f64() * 1e3,
    );
    *target.dflash2_turn_state_mut() = Some(state);
    target.on_flat_prefill_done();

    let mut generated = Vec::with_capacity(generated_capacity_hint(params.max_new_tokens));
    let mut finish_reason = String::from("length");
    let mut reasoning = ReasoningTracker::from_setup(&args.thinking, tokenizer.think_end_id());
    let stream_skip_special = ChatBackend::stream_skip_special_tokens(target);
    // One streaming bundle — detokenizer + cursors + emitter — built
    // only when a sink exists so the sync path never runs the emitter
    // hook.
    let mut turn_streaming = args.sink.map(|_| {
        TurnStreaming::new(
            target,
            tokenizer.inner(),
            args.thinking.enabled,
            stream_skip_special,
        )
    });
    let turn_token_observer = ChatBackend::turn_token_observer(target);
    let block_size = target
        .dflash2()
        .ok_or_else(|| Error::from_reason("loaded DFlash2 companion"))?
        .config
        .block_size;
    let mut rng = rand::rng();
    let outcome = {
        // Decode-phase allocator ceiling: prefill (2048-row chunks) has
        // already run under the load-time cap; the verify/propose loop
        // only needs room to recycle one cycle's transients, so the
        // free-pool is capped for exactly this scope and lifted on drop.
        let _decode_cache_cap =
            crate::cache_limit::coordinator().push_decode_limit(dflash2_decode_cache_limit(
                &target.dense_config(),
                block_size,
                tokens.len() as u64 + params.max_new_tokens.max(0) as u64,
            ));
        let streaming = turn_streaming
            .as_mut()
            .and_then(|ts| ts.ctx(args.sink, args.cancelled));
        run_dspark_turn(
            target,
            &mut rng,
            DsparkTurnArgs {
                y,
                block_size,
                params: &params,
                reasoning_tracker: &mut reasoning,
                profiler: &mut profiler,
                max_new_tokens: params.max_new_tokens,
                eos_id: args.eos_id,
                generated_tokens: &mut generated,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf: report_performance,
                generation_stream,
                cancel_flag: args.cancelled,
                turn_token_observer,
            },
            streaming,
        )
    };
    let mut last_in_cache = match outcome {
        Ok(outcome) => outcome.last_in_cache,
        Err(error) => return Err(dflash2_turn_failed(target, error, cached_prefix)),
    };
    let final_token_fits_context = target.dflash2_context().is_some_and(|context| {
        dflash2_final_token_fits_context(context.logical_len(), dflash_context_capacity)
    });
    if finish_reason == "length"
        && !last_in_cache
        && final_token_fits_context
        && let Some(&last) = generated.last()
    {
        if let Err(error) = dflash2_materialize_final(target, last, generation_stream) {
            return Err(dflash2_turn_failed(target, error, cached_prefix));
        }
        last_in_cache = true;
    }
    let saved_generated = if !last_in_cache && !generated.is_empty() {
        &generated[..generated.len() - 1]
    } else {
        generated.as_slice()
    };
    let mut history = tokens.clone();
    history.extend_from_slice(saved_generated);
    if params.reuse_cache {
        let resume = observed.classification.ideal.max(cached_prefix);
        let tracker = target.prefix_tracker_mut();
        let captures = design_captures(
            StateDesign::Flat,
            tracker.ledger.block_size() as u32,
            observed.replay_point,
            resume,
            resume,
        );
        tracker.ledger.record(LedgerTurn {
            history: &history,
            prompt_len: tokens.len(),
            replay_point: observed.replay_point,
            is_dflash2: true,
            captures: &captures,
        });
    }
    // The resolution above dropped only the state this turn replaced.
    target.prefix_tracker_mut().take_drop();
    target.set_cached_token_history(history);
    let states = target.dflash2_states_mut();
    states.live_salt = params.cache_salt;
    if let Some(anchor) = capture.and_then(PrefillCapture::into_anchor) {
        states.live_anchor = Some(anchor);
    }
    // This specialized path bypasses the generic save hook's non-reuse
    // cleanup. Other sessions' parked states stay.
    if !params.reuse_cache {
        drop_live_dflash2_state(target)?;
    }

    let performance = if report_performance {
        compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill.len(),
            generated.len(),
        )
        .map(|mut metrics| {
            ChatBackend::augment_performance(target, &profiler, &mut metrics);
            metrics
        })
    } else {
        None
    };
    if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
        let decoded = tokenizer
            .decode_sync(&generated, stream_skip_special)
            .unwrap_or_default();
        ts.flush_residual(&decoded, params.include_reasoning, sink);
    }
    let prompt_tokens = if is_delta && is_streaming {
        ChatBackend::stream_delta_prompt_tokens(target, tokens.len(), tokens.len() - prior_cached)
    } else {
        tokens.len() as u32
    };
    let mut result = ChatBackend::finalize_turn(
        target,
        FinalizeArgs {
            tokenizer: &tokenizer,
            generated_tokens: &generated,
            finish_reason,
            think_end_id: tokenizer.think_end_id(),
            think_end_str: tokenizer.think_end_str(),
            think_end_extra_ids: &[],
            performance,
            include_reasoning: params.include_reasoning,
            thinking_enabled: args.thinking.enabled,
            prompt_tokens,
            reasoning_tokens: reasoning.reasoning_token_count(),
        },
    )?;
    result.cached_tokens = cached_prefix as u32;
    if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
        ts.emitter.finish(&result, sink);
        Ok(TurnOutput::Streamed)
    } else {
        Ok(TurnOutput::Complete(Box::new(result)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::models::qwen3_5::dflash2_decode::tests::{
        array_fingerprints, metal_is_default_device, run_tiny_dflash_turn,
        tiny_dflash_inner_with_attention_head_dim, tiny_dflash_turn_config, tiny_int8_dflash_inner,
    };
    use crate::models::qwen3_5::dflash2_state::trace::capture::capture_prefix_events;
    use crate::models::qwen3_5::model::Qwen35Inner;

    type Fingerprints = Vec<(Vec<i64>, Vec<u32>)>;

    const FIXTURES: [&str; 2] = ["bf16_d64", "int8_d256"];
    const SUFFIX: [u32; 2] = [11, 12];
    const MAX_NEW: i32 = 12;

    fn fixture(name: &str) -> Qwen35Inner {
        match name {
            "bf16_d64" => tiny_dflash_inner_with_attention_head_dim(0xDFA5_1701, 64),
            "int8_d256" => tiny_int8_dflash_inner(0xDFA5_1702),
            _ => unreachable!("unknown fixture {name}"),
        }
    }

    fn tokenizer() -> Arc<Qwen3Tokenizer> {
        crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer()
    }

    /// A tokenizer whose generation suffix is [`SUFFIX`], so prompts that end
    /// with it arm C(R).
    fn suffixed_tokenizer() -> Arc<Qwen3Tokenizer> {
        let tokenizer = tokenizer();
        tokenizer.seed_generation_suffix(&tiny_dflash_turn_config(true, MAX_NEW), &SUFFIX);
        tokenizer
    }

    fn body(seed: u32, len: u32) -> Vec<u32> {
        (0..len).map(|i| (seed + i * 7) % 9 + 1).collect()
    }

    fn cat(parts: &[&[u32]]) -> Vec<u32> {
        parts.concat()
    }

    fn turn(inner: &mut Qwen35Inner, tokenizer: &Arc<Qwen3Tokenizer>, tokens: &[u32]) -> u32 {
        run_tiny_dflash_turn(inner, tokenizer, tokens, true, MAX_NEW)
            .expect("tiny DFlash2 turn")
            .cached_tokens
    }

    fn linear_slots(caches: &[Qwen3_5LayerCache]) -> Vec<(&MxArray, &MxArray)> {
        caches
            .iter()
            .filter_map(|cache| match cache {
                Qwen3_5LayerCache::Linear(gdn) => Some((
                    gdn.get(0).expect("conv slot"),
                    gdn.get(1).expect("state slot"),
                )),
                Qwen3_5LayerCache::FullAttention(_) => None,
            })
            .collect()
    }

    /// Full-attention rows `[0, rows)` (keys, values and, for int8, scales)
    /// in layer order.
    fn attention_rows<'a>(
        caches: impl IntoIterator<Item = &'a crate::transformer::KVCache>,
        rows: usize,
    ) -> Result<Fingerprints> {
        let mut arrays = Vec::new();
        for kv in caches {
            for buffer in [
                kv.keys_ref(),
                kv.values_ref(),
                kv.key_scales_ref(),
                kv.value_scales_ref(),
            ]
            .into_iter()
            .flatten()
            {
                arrays.push(buffer.slice_axis(2, 0, rows as i64)?);
            }
        }
        array_fingerprints(arrays.iter().collect())
    }

    fn live_attention(inner: &Qwen35Inner) -> Vec<&crate::transformer::KVCache> {
        inner
            .caches
            .iter()
            .flatten()
            .filter_map(|cache| match cache {
                Qwen3_5LayerCache::FullAttention(kv) => Some(kv),
                Qwen3_5LayerCache::Linear(_) => None,
            })
            .collect()
    }

    /// History, used full-attention rows, GDN slots and draft window of the
    /// live state: equal for states whose buffers differ only in capacity.
    fn live_rows(inner: &Qwen35Inner) -> Result<(Vec<u32>, Fingerprints, Fingerprints)> {
        let used = inner.cached_token_history.len();
        let attention = attention_rows(live_attention(inner), used)?;
        let mut arrays = Vec::new();
        for (conv, rec) in linear_slots(inner.caches.as_ref().expect("live caches")) {
            arrays.extend([conv, rec]);
        }
        inner
            .dflash2_context
            .as_ref()
            .expect("live draft context")
            .collect_arrays(&mut arrays);
        Ok((
            inner.cached_token_history.clone(),
            attention,
            array_fingerprints(arrays)?,
        ))
    }

    fn shares(a: &MxArray, b: &MxArray) -> bool {
        // SAFETY: both handles are live arrays for the call.
        unsafe { mlx_sys::mlx_array_aliases(a.as_raw_ptr(), b.as_raw_ptr()) }
    }

    /// A1, then B1 with reuse off, then A2: B1's miss parks A1's end state
    /// (an earlier reuse-on turn built it) and A2 resumes on it. X1: A2's
    /// cached prefix and its final live state (history, used KV rows, GDN,
    /// window) equal A1 then A2 with nothing in between.
    #[test]
    fn reuse_off_turn_parks_the_state_it_displaces() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let a1 = body(0, 40);
            let mut reference = fixture(name);
            turn(&mut reference, &tokenizer, &a1);
            let a2 = cat(&[&reference.cached_token_history, &[4, 5, 6]]);
            let reference_cached = turn(&mut reference, &tokenizer, &a2);
            assert!(reference_cached > 0, "{name}: reference live hit");

            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &a1);
            let b1 = run_tiny_dflash_turn(&mut inner, &tokenizer, &body(3, 33), false, MAX_NEW)?;
            assert_eq!(b1.cached_tokens, 0, "{name}: B1 is cold");
            assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{name}");
            assert_eq!(
                turn(&mut inner, &tokenizer, &a2),
                reference_cached,
                "{name}"
            );
            assert!(
                live_rows(&inner)? == live_rows(&reference)?,
                "{name}: X1 resumed turn"
            );
        }
        Ok(())
    }

    /// A prompt ending with the generation suffix arms C(R) at the last
    /// block before it. A re-rendered prompt that diverges after R misses the
    /// live and end states and resumes at R from the anchor parked with A1's
    /// end state; R is also the re-rendered prompt's replay point, so no new
    /// capture is armed and the anchor stays the live one.
    #[test]
    fn rerendered_prompt_resumes_at_the_parked_anchor() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = suffixed_tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            let a1 = cat(&[&body(1, 37), &SUFFIX]);
            assert_eq!(turn(&mut inner, &tokenizer, &a1), 0);
            let anchor = inner
                .dflash2_states
                .live_anchor
                .as_ref()
                .expect("A1 captured C(R)");
            assert_eq!(anchor.row, 32, "{name}: R = floor16(37)");
            assert_eq!(anchor.state.window.token_history(), &a1[..32]);
            assert_eq!(anchor.state.window.logical_len(), 32);

            let a2 = cat(&[&a1[..34], &[5, 5, 5], &SUFFIX]);
            assert!(!a2.starts_with(&inner.cached_token_history));
            let (cached, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &a2));
            assert_eq!(cached, 32, "{name}: resume at R");
            assert_eq!(events[0]["hit_kind"], "C", "{name}: {events:?}");
            assert_eq!(events[0]["prefilled_tokens"], (a2.len() - 32).to_string());
            assert_eq!(
                inner
                    .dflash2_states
                    .live_anchor
                    .as_ref()
                    .map(|anchor| anchor.row),
                Some(32),
                "{name}: the anchor stays live"
            );
            assert_eq!(
                flat_attention_frontier(&inner),
                Some(inner.cached_token_history.len()),
                "{name}"
            );
            assert!(inner.cached_token_history.starts_with(&a2));
        }
        Ok(())
    }

    /// X0: installing a parked anchor gives the full-attention rows `[0, R)`
    /// bit for bit, the GDN slots equal to the cell and the draft context
    /// equal to the window snapshot, none of them sharing a buffer with the
    /// anchor, which becomes the live anchor.
    #[test]
    fn anchor_install_copies_the_cell_and_window() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = suffixed_tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            let a1 = cat(&[&body(2, 37), &SUFFIX]);
            turn(&mut inner, &tokenizer, &a1);
            let below_anchor = attention_rows(live_attention(&inner), 32)?;
            let (tokens, parts) = park_flat_dflash2_state(&mut inner).expect("parkable");
            let anchor = parts.anchor.as_ref().expect("parked anchor");
            let row = anchor.row;
            let cell = array_fingerprints(
                anchor
                    .state
                    .cell
                    .conv
                    .iter()
                    .zip(&anchor.state.cell.rec)
                    .flat_map(|(conv, rec)| [conv, rec])
                    .collect(),
            )?;
            let mut window_arrays = Vec::new();
            anchor.state.window.collect_arrays(&mut window_arrays);
            let window = array_fingerprints(window_arrays)?;

            install_flat_dflash2_anchor(&mut inner, tokens.clone(), parts)?;
            assert_eq!(inner.cached_token_history, &tokens[..row], "{name}");
            let attention = live_attention(&inner);
            assert!(attention.iter().all(|kv| kv.get_offset() as usize == row));
            assert!(
                attention_rows(attention, row)? == below_anchor,
                "{name}: KV rows"
            );
            let caches = inner.caches.as_ref().expect("installed caches");
            let slots = linear_slots(caches);
            assert!(
                array_fingerprints(slots.iter().flat_map(|(conv, rec)| [*conv, *rec]).collect())?
                    == cell,
                "{name}: GDN slots"
            );
            let context = inner.dflash2_context.as_ref().expect("installed context");
            let mut live_window = Vec::new();
            context.collect_arrays(&mut live_window);
            assert!(
                array_fingerprints(live_window.clone())? == window,
                "{name}: window"
            );
            assert_eq!(context.token_history(), &tokens[..row]);

            let anchor = inner
                .dflash2_states
                .live_anchor
                .as_ref()
                .expect("live anchor");
            let mut anchor_window = Vec::new();
            anchor.state.window.collect_arrays(&mut anchor_window);
            for (live, kept) in live_window.iter().zip(&anchor_window) {
                assert!(!shares(live, kept), "{name}: window shares a buffer");
            }
            for ((conv, rec), (cell_conv, cell_rec)) in slots
                .iter()
                .zip(anchor.state.cell.conv.iter().zip(&anchor.state.cell.rec))
            {
                assert!(
                    !shares(conv, cell_conv) && !shares(rec, cell_rec),
                    "{name}: GDN slot shares a cell buffer"
                );
            }
        }
        Ok(())
    }

    /// With no KV headroom, the reserve evicts every parked state before it
    /// sizes the turn's KV, and the turn still completes.
    #[test]
    fn kv_pressure_evicts_parked_states_before_the_reserve() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        struct Override;
        impl Drop for Override {
            fn drop(&mut self) {
                KV_BUDGET_OVERRIDE.with(|budget| budget.set(None));
            }
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &body(4, 40));
            turn(&mut inner, &tokenizer, &body(5, 40));
            assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{name}");
            let _override = Override;
            KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
            let (result, events) = capture_prefix_events(|| {
                run_tiny_dflash_turn(&mut inner, &tokenizer, &body(6, 40), true, MAX_NEW)
            });
            let result = result?;
            assert_eq!(result.cached_tokens, 0, "{name}");
            assert!(result.num_tokens > 0, "{name}: the turn completes");
            assert_eq!(events[0]["evictions"], "2", "{name}: {events:?}");
            assert_eq!(events[0]["store_entries"], "0", "{name}");
            assert_eq!(inner.dflash2_states.store.bytes().entries, 0, "{name}");
        }
        Ok(())
    }

    /// Without an armed capture the kept tail is the newest window of the
    /// prompt (today's schedule); a capture past the resume point keeps the
    /// window below it, clamped to the resume point when the gap is shorter
    /// than the window; a capture at or before the resume point, or past the
    /// prompt's end, keeps today's schedule.
    #[test]
    fn keep_from_follows_the_armed_capture() {
        const W: usize = 2047;
        for (resume, end, capture, want, case) in [
            (0, 5000, None, 2953, "no capture"),
            (0, 100, None, 0, "no capture, short prompt"),
            (4000, 5000, None, 4000, "no capture, short tail"),
            (3000, 6000, Some(4000), 3000, "gap < window"),
            (0, 6000, Some(4000), 1953, "gap >= window, cold"),
            (1000, 6000, Some(4000), 1953, "gap >= window, warm"),
            (2000, 6000, Some(4047), 2000, "gap == window"),
            (0, 6000, Some(100), 0, "capture inside the first window"),
            (0, 6000, Some(6000), 3953, "capture at the prompt end"),
            (4000, 9000, Some(4000), 6953, "capture at the resume point"),
            (
                4000,
                9000,
                Some(100),
                6953,
                "capture before the resume point",
            ),
            (0, 6000, Some(7000), 3953, "capture past the prompt end"),
        ] {
            assert_eq!(draft_keep_from(resume, end, W, capture), want, "{case}");
        }
    }
}
