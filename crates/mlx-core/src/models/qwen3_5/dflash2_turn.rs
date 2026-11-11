//! Family-generic DFlash2 whole turn shared by the dense and MoE Qwen3.5
//! targets: prefix reuse, tapped prefill, target KV reservation, decode,
//! fail-closed reset and the final-token context append. The speculative
//! steppers stay per family.

use std::borrow::Cow;
use std::collections::HashSet;
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
use crate::engine::memory_plan::KvLayout;
use crate::engine::params::generated_capacity_hint;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::engine::reclaim::{
    KvNeed, KvReclaim, MetBefore, Reach, Unit, UnitKey, free_for_kv, older_first,
};
use crate::engine::types::ChatConfig;
use crate::models::forward::PREFILL_CANCELLED;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::KvFormat;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::config::Qwen3_5Config;
use super::dflash2::{DFlash2ContextCache, DFlash2Model};
use super::dflash2_decode::{
    constrain_dflash2_context_params, dflash2_cycle_transient_bytes, dflash2_decode_cache_limit,
    dflash2_final_token_fits_context, dflash2_kv_layer_copy_bytes, dflash2_memory_snapshot,
    dflash2_turn_transient_bytes, grow_dflash2_target_kv, kv_row_bytes, min_fa_capacity,
    min_fa_reserved_rows, reusable_dflash2_prefix,
};
use super::dflash2_state::flat::{
    FlatAnchor, FlatBacking, FlatStates, flat_attention_kv, join_flat_anchor, join_flat_caches,
    split_flat_caches,
};
use super::dflash2_state::store::{
    Anchor, EntryParts, EvictReport, Lineage, Reservation, StoreHit,
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
    /// How decode grows the target KV; `None` grows it unchecked.
    pub(crate) kv: Option<DFlash2KvPlan>,
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

/// The prompt a reclaim keeps: its ladder and history entries are no
/// units, and the marked blocks it hits are out of the count. With the
/// prompt's block keys and salt the hit and its ladder rungs are read by
/// hash; without them, by token prefix.
#[derive(Clone, Copy, Default)]
pub(crate) struct Keep<'a> {
    pub tokens: &'a [u32],
    pub keys: Option<(&'a [Vec<u64>], u64)>,
}

impl<'a> Keep<'a> {
    /// The salted, keyed chain hashes the prompt's lookup walks, its last
    /// token excluded (the lookup leaves one token to prefill); `None`
    /// without keys.
    pub(crate) fn hit_chain(&self, block_size: u32) -> Option<Vec<u64>> {
        let (keys, salt) = self.keys?;
        let prompt = &self.tokens[..self.tokens.len().saturating_sub(1)];
        Some(mlx_paged_attn::PrefixKeys::PerBlock(keys).chain_hashes(prompt, block_size, salt))
    }

    /// The replay points this prompt can restore, resolved once for a
    /// reclaim (Splash `RestoredChains`). The token test leaves out the
    /// last token as the lookup does (Splash `Cache::extendMatch`): a point
    /// at the prompt's own end is no replay point of it.
    pub(crate) fn points(self, block_size: Option<u32>) -> KeptPoints<'a> {
        KeptPoints {
            tokens: &self.tokens[..self.tokens.len().saturating_sub(1)],
            chain: block_size.and_then(|block_size| self.hit_chain(block_size)),
        }
    }
}

/// What a [`Keep`] protects from a reclaim. A ladder rung on another salt
/// or image-key chain is no replay point of the prompt even when its
/// tokens prefix it: the lookup rejects it by hash (Splash
/// `Walk::keepsState`).
pub(crate) struct KeptPoints<'a> {
    tokens: &'a [u32],
    chain: Option<Vec<u64>>,
}

impl KeptPoints<'_> {
    /// Whether an entry on `tokens` is kept: by hash chain when the prompt
    /// has keys and the entry sits on blocks (`block_hashes`), else by
    /// token prefix.
    pub(crate) fn keeps(&self, tokens: &[u32], block_hashes: Option<&[u64]>) -> bool {
        match (&self.chain, block_hashes) {
            (Some(chain), Some(hashes)) => !hashes.is_empty() && chain.starts_with(hashes),
            _ => !self.tokens.is_empty() && self.tokens.starts_with(tokens),
        }
    }
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
    /// The context a turn may hold (`context_cap`).
    fn context_tokens(&self) -> u32;
    /// The speculative context window, `None` without a draft.
    fn dflash2_window(&self) -> Option<u32>;
    /// The load's memory plan; `None` when the working set is unreadable.
    fn memory_plan(&self) -> Option<&crate::engine::memory_plan::MemoryPlan>;
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
    fn paged_adapter_ref(
        &self,
    ) -> Option<&crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter>;
    fn paged_adapter_mut_ref(
        &mut self,
    ) -> Option<&mut crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter>;
    /// The image-feature cache (Splash's `embeddings` pool).
    fn vision_cache(&self) -> &crate::vision::qwen::cache::VisionCache;
    /// Whether the flat caches hold the retained prefix (no paged or MTP
    /// turn has moved past them).
    fn flat_lane_authoritative(&self) -> bool;
    /// Marks the flat caches as holding the retained prefix after a parked
    /// state moved into them.
    fn claim_flat_lane(&mut self);
    /// Runs once the tapped prefill and first sample succeeded: the flat
    /// caches now hold this turn's prompt.
    fn on_flat_prefill_done(&mut self) {}

    /// The oldest ladder or history entry a KV need may take, of those a
    /// turn uses (`in_use`) or of the rest.
    fn oldest_gdn_state(
        &self,
        at_cap: bool,
        skip: &HashSet<UnitKey>,
        keep: Keep<'_>,
        in_use: bool,
    ) -> Option<Unit>;
    /// Drops a ladder or history entry; false when it is gone.
    fn drop_gdn_state(&mut self, key: &UnitKey) -> bool;
    fn sync_gdn_state_marks(&self);
    /// Bytes of the GDN entries a reclaim may take, with `in_use` those
    /// turns use too.
    fn gdn_state_bytes(&self, keep: Keep<'_>, in_use: bool) -> u64;
    /// Marked cache-only blocks the kept prompt hits.
    fn marked_hit_blocks(&self, keep: Keep<'_>) -> u32;

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
            crate::engine::progress::with_turn_progress(|p| p.prefill_advanced(end as u32));
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
            kv: None,
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

/// The plan's live headroom on `target`; `None` without a plan or a
/// readable device.
pub(crate) fn dflash2_kv_headroom_now<T: DFlash2Target>(target: &T) -> Option<u64> {
    #[cfg(test)]
    if let Some(bytes) = KV_BUDGET_OVERRIDE.with(std::cell::Cell::get) {
        return Some(bytes);
    }
    target
        .memory_plan()?
        .live_headroom(dflash2_memory_snapshot())
}

/// The target's full-attention KV geometry: its Splash layout and the
/// bytes of one token row across every layer.
fn target_kv_layout<T: DFlash2Target>(target: &T) -> Result<(KvLayout, u64)> {
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
    let dtype = dflash2_target_kv_dtype(target)?;
    let (heads, head_dim) = (target.kv_heads(), target.head_dim());
    Ok((
        KvLayout::of(full_attention_layers, heads, head_dim, dtype, format),
        kv_row_bytes(full_attention_layers, heads, head_dim, dtype, format),
    ))
}

/// Full-attention K/V bytes of one token row on `target`, and its turn
/// transients for `prompt_len` prompt rows and `context_rows` rows of KV.
fn turn_memory<T: DFlash2Target>(
    target: &T,
    prompt_len: usize,
    context_rows: u64,
) -> Result<(u64, u64)> {
    let (_, row_bytes) = target_kv_layout(target)?;
    let (block_size, tap_layers) = target.dflash2().map_or((1, 0), |draft| {
        (draft.config.block_size, draft.config.target_layers.len())
    });
    let transient = dflash2_turn_transient_bytes(
        &target.dense_config(),
        block_size,
        prompt_len,
        context_rows,
        dflash2_target_kv_format(target),
        tap_layers,
    );
    Ok((row_bytes, transient))
}

/// Before a vision turn on a load without a paged pool: parked states give
/// way, one store step at a time, until the live headroom covers what the
/// turn books first (its prompt plus one verify's rows of K/V) and its
/// transients. The live state stays: a vision turn may continue its prefix.
/// Unreadable limits evict nothing.
pub(crate) fn evict_parked_dflash2_for_flat_turn<T: DFlash2Target>(
    target: &mut T,
    prompt_len: usize,
) {
    evict_parked_dflash2_for_flat_turn_with(target, prompt_len, dflash2_kv_headroom_now);
}

pub(crate) fn evict_parked_dflash2_for_flat_turn_with<T: DFlash2Target>(
    target: &mut T,
    prompt_len: usize,
    headroom: impl Fn(&T) -> Option<u64>,
) {
    if target.dflash2_states().store.bytes().entries == 0 {
        return;
    }
    let block_size = target.dflash2().map_or(1, |draft| draft.config.block_size);
    let rows = prompt_len.saturating_add(block_size + 1) as u64;
    let need = match turn_memory(target, prompt_len, rows) {
        Ok((row_bytes, transient)) => rows.saturating_mul(row_bytes).saturating_add(transient),
        Err(error) => {
            tracing::warn!(
                target: "mlx_core::inference",
                "DFlash2 parked-state eviction skipped: {}",
                error.reason
            );
            return;
        }
    };
    while let Some(live) = headroom(target).filter(|&live| live < need) {
        if !evict_one_parked_dflash2(target, live) {
            return;
        }
    }
}

/// What a reclaim took, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GrowReclaim {
    Compaction,
    Store,
    Ladder,
    History,
    Table,
    Vision,
}

#[cfg(test)]
thread_local! {
    /// What reclaims on this thread took, in order.
    pub(crate) static GROW_RECLAIM_LOG: std::cell::RefCell<Vec<GrowReclaim>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn grow_reclaim_log() -> Vec<GrowReclaim> {
    GROW_RECLAIM_LOG.with(|log| log.borrow().clone())
}

#[cfg(test)]
pub(crate) fn clear_grow_reclaim_log() {
    GROW_RECLAIM_LOG.with(|log| log.borrow_mut().clear());
}

fn log_reclaim(_taken: GrowReclaim) {
    #[cfg(test)]
    GROW_RECLAIM_LOG.with(|log| log.borrow_mut().push(_taken));
}

/// One store step of parked-state eviction against `kv_headroom` (which
/// bounds the copy a compaction may run). Returns whether anything was
/// freed; a failed step drops every parked state.
fn evict_one_parked_dflash2<T: DFlash2Target>(target: &mut T, kv_headroom: u64) -> bool {
    let freed = match target
        .dflash2_states_mut()
        .store
        .evict_one(kv_headroom, None)
    {
        Ok(Some(_)) => true,
        Ok(None) => false,
        Err(error) => {
            tracing::warn!(
                target: "mlx_core::inference",
                "DFlash2 parked-state eviction failed, dropping every parked state: {}",
                error.reason
            );
            let states = target.dflash2_states_mut();
            let held = states.store.bytes().entries > 0;
            states.store.clear();
            held
        }
    };
    if freed {
        crate::array::clear_cache();
    }
    freed
}

/// Runs `step`; while its block growth is refused and `reclaim` frees
/// memory, runs it again. A refused paged step has changed nothing:
/// `record_tokens` allocates before any layer runs.
pub(crate) fn retry_after_reclaim<T, R>(
    target: &mut T,
    mut step: impl FnMut(&mut T) -> Result<R>,
    mut reclaim: impl FnMut(&mut T) -> bool,
) -> Result<R> {
    loop {
        match step(target) {
            Err(error)
                if crate::engine::scheduler::is_paged_allocation_blocked(&error.reason)
                    && reclaim(target) => {}
            other => return other,
        }
    }
}

/// Runs `step`; while its reservation is refused and the take order frees
/// what it lacks (or the need is met in place), runs it again (Splash
/// `Engine::allocate`). `keep` is the prompt the step reserves for: its
/// ladder and history entries are no units, and the marked blocks it hits
/// are out of the count.
pub(crate) fn reclaiming<T: DFlash2Target, R>(
    target: &mut T,
    keep: Keep<'_>,
    mut step: impl FnMut(&mut T) -> Result<R>,
) -> Result<R> {
    let mut met_before = MetBefore::default();
    retry_after_reclaim(
        target,
        |target| {
            if let Some(adapter) = target.paged_adapter_ref() {
                adapter.clear_refusals();
            }
            step(target)
        },
        |target| {
            let Some(blocks) = target
                .paged_adapter_ref()
                .and_then(PagedKVCacheAdapter::take_refusal)
            else {
                return false;
            };
            if met_before.seen(&blocks) {
                return false;
            }
            let freed = free_for_kv(
                &mut ModelReclaim {
                    target,
                    keep,
                    park: false,
                },
                KvNeed::in_place(blocks),
            );
            met_before.retry(blocks, freed)
        },
    )
}

/// One serial paged step under [`reclaiming`], keeping no prompt.
pub(crate) fn paged_step_reclaiming<T: DFlash2Target, R>(
    target: &mut T,
    step: impl FnMut(&mut T) -> Result<R>,
) -> Result<R> {
    reclaiming(target, Keep::default(), step)
}

/// The model side of the take order: parked states, ladder and history
/// entries, ordered by their own stamps.
pub(crate) struct ModelReclaim<'a, T: DFlash2Target> {
    pub(crate) target: &'a mut T,
    pub(crate) keep: Keep<'a>,
    /// Park the idle live state before the count (below the cap only).
    pub(crate) park: bool,
}

impl<T: DFlash2Target> KvReclaim for ModelReclaim<'_, T> {
    fn at_cap(&self) -> bool {
        self.target
            .paged_adapter_ref()
            .is_none_or(PagedKVCacheAdapter::at_cap)
    }

    fn takeable(&self) -> u32 {
        self.target
            .paged_adapter_ref()
            .map_or(0, PagedKVCacheAdapter::takeable_blocks)
    }

    fn ungrown(&self) -> u32 {
        self.target
            .paged_adapter_ref()
            .map_or(0, PagedKVCacheAdapter::ungrown_blocks)
    }

    fn bytes_per_block(&self) -> u64 {
        self.target
            .paged_adapter_ref()
            .and_then(|adapter| adapter.bytes_per_block().ok())
            .unwrap_or(0)
    }

    fn met(&mut self, need: KvNeed) -> bool {
        self.target
            .paged_adapter_mut_ref()
            .is_some_and(|adapter| adapter.kv_need_met(need.blocks, need.over_takeable))
    }

    fn grow_room(&self) -> u32 {
        self.target
            .paged_adapter_ref()
            .map_or(0, PagedKVCacheAdapter::last_grow_room)
    }

    fn before_count(&mut self, at_cap: bool) {
        if self.park && !at_cap {
            park_live_dflash2_state(self.target);
        }
    }

    fn sync_marks(&mut self) {
        self.target.sync_gdn_state_marks();
    }

    fn reach(&self, at_cap: bool, in_use: bool) -> Reach {
        let marked = self
            .target
            .paged_adapter_ref()
            .map_or(0, PagedKVCacheAdapter::marked_blocks);
        Reach {
            blocks: u64::from(marked.saturating_sub(self.target.marked_hit_blocks(self.keep))),
            bytes: if at_cap {
                0
            } else {
                state_reach_bytes(&*self.target, self.keep, in_use)
            },
        }
    }

    fn compact_one(&mut self) -> bool {
        compact_one_state(self.target)
    }

    fn oldest_state(&self, at_cap: bool, skip: &HashSet<UnitKey>, in_use: bool) -> Option<Unit> {
        oldest_state_unit(&*self.target, at_cap, skip, self.keep, in_use)
    }

    fn take(&mut self, key: &UnitKey) -> bool {
        take_state_unit(self.target, key).is_some()
    }

    fn rebuilt_bytes(&self) -> u64 {
        rebuilt_bytes(&*self.target)
    }

    fn rebuild_one(&mut self) -> bool {
        release_one_rebuilt(&*self.target)
    }
}

/// Bytes of the image features nothing outside the cache holds.
pub(crate) fn rebuilt_bytes<T: DFlash2Target>(target: &T) -> u64 {
    crate::vision::qwen::cache::unheld_vision_bytes(target.vision_cache())
}

/// Drops the least recently used image feature entry nothing holds.
pub(crate) fn release_one_rebuilt<T: DFlash2Target>(target: &T) -> bool {
    let released = crate::vision::qwen::cache::release_one_unheld_vision(target.vision_cache());
    if released {
        log_reclaim(GrowReclaim::Vision);
    }
    released
}

/// What [`take_state_unit`] took.
pub(crate) enum Taken {
    Store(EvictReport),
    Gdn,
}

/// Bytes every parked state and every GDN entry outside `keep` hold; the
/// entries turns use count only with `in_use`.
pub(crate) fn state_reach_bytes<T: DFlash2Target>(target: &T, keep: Keep<'_>, in_use: bool) -> u64 {
    let store = target.dflash2_states().store.bytes();
    store
        .state_bytes
        .saturating_add(store.kv_bytes)
        .saturating_add(target.gdn_state_bytes(keep, in_use))
}

/// One lossless store compaction; whether it freed anything.
pub(crate) fn compact_one_state<T: DFlash2Target>(target: &mut T) -> bool {
    let headroom = dflash2_kv_headroom_now(&*target).unwrap_or(0);
    match target
        .dflash2_states_mut()
        .store
        .compact_one(headroom, None)
    {
        Ok(Some(_)) => {
            log_reclaim(GrowReclaim::Compaction);
            true
        }
        Ok(None) => false,
        Err(error) => {
            tracing::warn!(
                target: "mlx_core::inference",
                "DFlash2 state compaction failed: {}",
                error.reason
            );
            false
        }
    }
}

/// The oldest state that meets a KV need. At the cap no store step and no
/// history entry is one: they sit on no block. No store step is in use.
pub(crate) fn oldest_state_unit<T: DFlash2Target>(
    target: &T,
    at_cap: bool,
    skip: &HashSet<UnitKey>,
    keep: Keep<'_>,
    in_use: bool,
) -> Option<Unit> {
    let store = if at_cap || in_use {
        None
    } else {
        target
            .dflash2_states()
            .store
            .next_lossy(dflash2_kv_headroom_now(target).unwrap_or(0), None)
            .map(|(stamp, step)| Unit {
                stamp,
                key: UnitKey::Store(step),
            })
            .filter(|unit| !skip.contains(&unit.key))
    };
    older_first(store, target.oldest_gdn_state(at_cap, skip, keep, in_use))
}

/// Takes one state unit; `None` when it freed nothing. A failed store step
/// drops every parked state.
pub(crate) fn take_state_unit<T: DFlash2Target>(target: &mut T, key: &UnitKey) -> Option<Taken> {
    match key {
        UnitKey::Store(step) => {
            let store = &mut target.dflash2_states_mut().store;
            match store.apply_lossy(*step) {
                Ok(report) => {
                    log_reclaim(GrowReclaim::Store);
                    Some(Taken::Store(report))
                }
                Err(error) => {
                    tracing::warn!(
                        target: "mlx_core::inference",
                        "DFlash2 parked-state eviction failed, dropping every parked state: {}",
                        error.reason
                    );
                    let held = store.bytes().entries > 0;
                    store.clear();
                    held.then(|| Taken::Store(EvictReport::default()))
                }
            }
        }
        UnitKey::Ladder { .. } | UnitKey::History => {
            let dropped = target.drop_gdn_state(key);
            if dropped {
                log_reclaim(if matches!(key, UnitKey::History) {
                    GrowReclaim::History
                } else {
                    GrowReclaim::Ladder
                });
            }
            dropped.then_some(Taken::Gdn)
        }
        UnitKey::Owner(_) | UnitKey::Table(_) => None,
    }
}

/// Grows the page pool so it can hand out every block a serial turn's
/// `prompt` takes before the turn's prefix lookup runs: an SSD restore
/// allocates from the pool and cannot grow it, so a runway-sized pool caps
/// the restore. Splash allocates a restored chain's pages through
/// `Engine::allocate` before the restore starts. Blocks the live request
/// holds count as available (the turn continues on them or releases them),
/// and so do the `marked_hit` blocks the prompt hits. The met check grows
/// first and counts cached blocks after (`kv_need_met`); a need it leaves
/// short goes to `reclaim`.
pub(crate) fn grow_paged_pool_for_prompt<T>(
    target: &mut T,
    prompt: &[u32],
    marked_hit: u32,
    adapter: impl Fn(&mut T) -> Option<&mut PagedKVCacheAdapter>,
    mut reclaim: impl FnMut(&mut T, u32) -> bool,
) {
    let Some(pool) = adapter(target) else {
        return;
    };
    let live = pool
        .active_seq_id()
        .and_then(|seq_id| pool.block_table_for(seq_id))
        .map_or(0, |table| table.num_blocks() as u32);
    let needed = u32::try_from((prompt.len() as u64).div_ceil(u64::from(pool.block_size().max(1))))
        .unwrap_or(u32::MAX);
    let blocks = needed.saturating_sub(live).saturating_sub(marked_hit);
    if blocks == 0 || pool.kv_need_met(blocks, false) {
        return;
    }
    reclaim(target, blocks);
}

/// [`grow_paged_pool_for_prompt`] on `target`'s pool, its short need sent
/// through the take order. `keys` and `cache_salt` are the prepare's, so
/// the prompt's own marked hit is read by hash. Runs after the turn parked
/// the live state, so that state is a unit too.
pub(crate) fn ensure_paged_pool_for_prompt<T: DFlash2Target>(
    target: &mut T,
    prompt: &[u32],
    keys: &[Vec<u64>],
    cache_salt: u64,
) {
    let keep = Keep {
        tokens: prompt,
        keys: Some((keys, cache_salt)),
    };
    ensure_paged_pool_keeping(target, prompt, keep);
}

/// [`ensure_paged_pool_for_prompt`] keeping `keep`. A prepare that skips
/// the lookup keeps nothing (Splash keeps the restored chain only when the
/// request does not skip the cache).
pub(crate) fn ensure_paged_pool_keeping<T: DFlash2Target>(
    target: &mut T,
    prompt: &[u32],
    keep: Keep<'_>,
) {
    let marked_hit = target.marked_hit_blocks(keep);
    let mut met_before = MetBefore::default();
    grow_paged_pool_for_prompt(
        target,
        prompt,
        marked_hit,
        |target| target.paged_adapter_mut_ref(),
        |target, blocks| {
            if met_before.seen(&blocks) {
                return false;
            }
            let need = KvNeed {
                blocks,
                over_takeable: false,
                checked: true,
                reach_in_use: false,
                rebuilt: true,
            };
            let freed = free_for_kv(
                &mut ModelReclaim {
                    target,
                    keep,
                    park: false,
                },
                need,
            );
            met_before.retry(blocks, freed)
        },
    );
}

/// How a DFlash2 turn's flat target KV grows: to at most `cap_rows` (the
/// turn's context capacity plus the verify scratch rows), whole Splash KV
/// extents of `extent_rows` at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DFlash2KvPlan {
    pub(crate) cap_rows: i64,
    pub(crate) extent_rows: i64,
}

impl DFlash2KvPlan {
    /// Rows a growth that must hold `needed` rows sizes the KV to.
    pub(crate) fn target_rows(&self, needed: i64) -> i64 {
        let extent = self.extent_rows.max(1);
        let needed = needed.max(0);
        needed
            .saturating_add(extent - 1)
            .saturating_div(extent)
            .saturating_mul(extent)
            .min(self.cap_rows)
            .max(needed)
    }
}

#[cfg(test)]
thread_local! {
    /// KV extent of turns on this thread. Tiny test models' Splash extent
    /// exceeds their trained window, so their turns grow 256 rows at a time
    /// unless a test sets another; `None` uses the model's extent.
    pub(crate) static KV_EXTENT_ROWS_OVERRIDE: std::cell::Cell<Option<i64>> =
        const { std::cell::Cell::new(Some(256)) };
}

/// The KV plan of a turn whose context holds `context_capacity` tokens: the
/// memory plan's extent, else the layout's smallest.
pub(crate) fn dflash2_kv_plan<T: DFlash2Target>(
    target: &T,
    context_capacity: usize,
) -> Result<DFlash2KvPlan> {
    let label = target.label();
    let block_size = target
        .dflash2()
        .ok_or_else(|| Error::from_reason(format!("{label} DFlash2 model disappeared")))?
        .config
        .block_size;
    let cap_rows = i64::try_from(context_capacity.saturating_add(block_size)).unwrap_or(i64::MAX);
    #[cfg(test)]
    if let Some(extent_rows) = KV_EXTENT_ROWS_OVERRIDE.with(std::cell::Cell::get) {
        return Ok(DFlash2KvPlan {
            cap_rows,
            extent_rows,
        });
    }
    let extent_rows = match target
        .memory_plan()
        .and_then(|plan| plan.speculative.as_ref())
        .map(|lane| lane.extent_pages)
        .filter(|&pages| pages > 0)
    {
        Some(pages) => pages * crate::engine::memory_plan::KV_PAGE_TOKENS,
        None => target_kv_layout(target)?.0.extent_rows_for(0),
    };
    Ok(DFlash2KvPlan {
        cap_rows,
        extent_rows: i64::try_from(extent_rows).unwrap_or(i64::MAX),
    })
}

/// When a target KV growth runs, which sets the transients it must leave
/// room for.
#[derive(Clone, Copy, Debug)]
pub(crate) enum KvGrowthPhase {
    /// Before prefill: the whole turn's transients at the grown size.
    TurnStart { prompt_len: usize },
    /// During decode: what the decode transients add at the grown size.
    Decode,
}

/// What [`ensure_dflash2_target_rows`] did.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct KvGrowth {
    pub(crate) grew: bool,
    pub(crate) evicted: EvictReport,
    /// Bytes the paged pool gave back for the growth.
    pub(crate) released_pool_bytes: u64,
    pub(crate) dropped_gdn_states: u32,
    pub(crate) released_tables: u32,
    pub(crate) released_vision_entries: u32,
}

fn add_eviction(total: &mut EvictReport, step: EvictReport) {
    total.compacted += step.compacted;
    total.demoted += step.demoted;
    total.dropped_captures += step.dropped_captures;
    total.dropped_ends += step.dropped_ends;
    total.freed_state_bytes = total
        .freed_state_bytes
        .saturating_add(step.freed_state_bytes);
    total.freed_kv_bytes = total.freed_kv_bytes.saturating_add(step.freed_kv_bytes);
}

/// Splash's `ensureTokens` on the flat target KV: before rows up to
/// `needed` are written, every full-attention cache holds them. A growth
/// takes [`DFlash2KvPlan::target_rows`] and goes ahead once the live
/// headroom's budget covers its KV (the new rows plus one layer's copy) and
/// the headroom also covers the transients of `phase`; until then memory is
/// reclaimed one unit at a time ([`flat_reclaim_one`]), the
/// headroom read again after each. With nothing left the KV alone must fit,
/// else the turn fails with `context_length_exceeded` and the KV is
/// unchanged. Unreadable limits grow unchecked.
pub(crate) fn ensure_dflash2_target_rows<T: DFlash2Target>(
    target: &mut T,
    needed: i64,
    plan: DFlash2KvPlan,
    phase: KvGrowthPhase,
) -> Result<KvGrowth> {
    ensure_dflash2_target_rows_with(target, needed, plan, phase, dflash2_kv_headroom_now)
}

/// One growth [`ensure_dflash2_target_rows`] would make next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvGrowthNeed {
    /// Rows the caches hold now.
    pub(crate) from_rows: i64,
    /// Rows they grow to.
    pub(crate) rows: i64,
    /// New KV bytes plus one layer's copy.
    pub(crate) kv_bytes: u64,
    /// Transient bytes the headroom must hold beside them.
    pub(crate) transient_bytes: u64,
}

/// The growth that makes `target` hold `needed` rows, `None` when it
/// already does or has no full-attention cache.
pub(crate) fn dflash2_kv_growth_need<T: DFlash2Target>(
    target: &T,
    needed: i64,
    plan: DFlash2KvPlan,
    phase: KvGrowthPhase,
) -> Result<Option<KvGrowthNeed>> {
    let Some(from_rows) = min_fa_reserved_rows(target.caches())? else {
        return Ok(None);
    };
    if from_rows >= needed {
        return Ok(None);
    }
    let rows = plan.target_rows(needed);
    let allocated = min_fa_capacity(target.caches())?.unwrap_or(0);
    let (layout, row_bytes) = target_kv_layout(target)?;
    let copy = if allocated > 0 {
        dflash2_kv_layer_copy_bytes(rows as u64, row_bytes, layout.layers)
    } else {
        0
    };
    let kv_bytes = ((rows - allocated).max(0) as u64)
        .saturating_mul(row_bytes)
        .saturating_add(copy);
    let transient_bytes = match phase {
        KvGrowthPhase::TurnStart { prompt_len } => turn_memory(target, prompt_len, rows as u64)?.1,
        KvGrowthPhase::Decode => {
            let config = target.dense_config();
            let block_size = target.dflash2().map_or(1, |draft| draft.config.block_size);
            dflash2_cycle_transient_bytes(&config, block_size, rows as u64).saturating_sub(
                dflash2_cycle_transient_bytes(&config, block_size, from_rows as u64),
            )
        }
    };
    Ok(Some(KvGrowthNeed {
        from_rows,
        rows,
        kv_bytes,
        transient_bytes,
    }))
}

pub(crate) fn ensure_dflash2_target_rows_with<T: DFlash2Target>(
    target: &mut T,
    needed: i64,
    plan: DFlash2KvPlan,
    phase: KvGrowthPhase,
    headroom: impl Fn(&T) -> Option<u64>,
) -> Result<KvGrowth> {
    try_ensure_dflash2_target_rows_with(target, needed, plan, phase, headroom, true)?
}

/// [`ensure_dflash2_target_rows_with`] with a refused growth as the inner
/// `Err`: the KV is unchanged. Memory is reclaimed for it only when
/// `reclaim` is set, and only when the count says the KV itself can be met:
/// the live headroom plus what the pool cut and every state could free. A
/// doomed growth takes nothing (Splash `MemoryLedger::free`).
fn try_ensure_dflash2_target_rows_with<T: DFlash2Target>(
    target: &mut T,
    needed: i64,
    plan: DFlash2KvPlan,
    phase: KvGrowthPhase,
    headroom: impl Fn(&T) -> Option<u64>,
    reclaim: bool,
) -> Result<Result<KvGrowth>> {
    let mut growth = KvGrowth::default();
    let mut in_reach: Option<bool> = None;
    let mut skip = HashSet::new();
    let mut took = false;
    loop {
        let Some(need) = dflash2_kv_growth_need(target, needed, plan, phase)? else {
            if took {
                crate::array::clear_cache();
            }
            return Ok(Ok(growth));
        };
        let live = headroom(target);
        let grows = live.is_none_or(|live| dflash2_growth_fits(live, &need));
        if !grows
            && reclaim
            && let Some(live) = live
        {
            // The count (`flat_kv_in_reach`) gates on the KV alone; a take it
            // lets through goes on toward the KV plus its transients until
            // nothing is left: they have no fixed reservation here, unlike
            // Splash's scratch.
            let counted = *in_reach.get_or_insert_with(|| {
                crate::array::clear_cache();
                flat_kv_in_reach(&*target, live, &need)
            });
            if counted
                && let Some(step) = flat_reclaim_one(
                    target,
                    live,
                    dflash2_kv_growth_shortfall(live, &need),
                    &mut skip,
                )?
            {
                took = true;
                add_reclaimed(&mut growth, step);
                continue;
            }
        }
        if took {
            crate::array::clear_cache();
        }
        if !grows
            && let Some(live) = live
            && live < need.kv_bytes
        {
            return Ok(Err(Error::from_reason(format!(
                "context_length_exceeded: {} DFlash2 target KV could not grow from {} to {} \
                 rows: it needs {} bytes and the memory plan's ceiling of {} bytes leaves {}",
                target.label(),
                need.from_rows,
                need.rows,
                need.kv_bytes,
                target
                    .memory_plan()
                    .map_or(0, crate::engine::memory_plan::MemoryPlan::ceiling_bytes),
                live
            ))));
        }
        grow_dflash2_target_kv(target.caches_mut(), need.rows)?;
        tracing::debug!(
            target: "mlx_core::inference",
            from_rows = need.from_rows,
            rows = need.rows,
            kv_bytes = need.kv_bytes,
            transient_bytes = need.transient_bytes,
            headroom = live,
            evicted_states = growth.evicted.demoted
                + growth.evicted.dropped_captures
                + growth.evicted.dropped_ends,
            dropped_gdn_states = growth.dropped_gdn_states,
            released_tables = growth.released_tables,
            released_vision_entries = growth.released_vision_entries,
            released_pool_bytes = growth.released_pool_bytes,
            "DFlash2 target KV growth"
        );
        growth.grew = true;
        return Ok(Ok(growth));
    }
}

/// Whether `live` headroom holds a growth's KV and its transients.
pub(crate) fn dflash2_growth_fits(live: u64, need: &KvGrowthNeed) -> bool {
    live >= need.kv_bytes.saturating_add(need.transient_bytes)
}

/// Bytes a growth needing `need` is short of in `live` headroom.
pub(crate) fn dflash2_kv_growth_shortfall(live: u64, need: &KvGrowthNeed) -> u64 {
    need.kv_bytes
        .saturating_add(need.transient_bytes)
        .saturating_sub(live)
}

/// Whether `live` headroom plus every byte a reclaim could free reaches the
/// growth's KV: the pool cut with every table released, each parked state,
/// each GDN entry and each image feature entry nothing holds.
fn flat_kv_in_reach<T: DFlash2Target>(target: &T, live: u64, need: &KvGrowthNeed) -> bool {
    let cut = target
        .paged_adapter_ref()
        .map_or(0, PagedKVCacheAdapter::release_reach_bytes);
    live.saturating_add(cut)
        .saturating_add(state_reach_bytes(target, Keep::default(), false))
        .saturating_add(rebuilt_bytes(target))
        >= need.kv_bytes
}

/// What one reclaim step gave back.
enum Reclaimed {
    Pool(u64),
    State(EvictReport),
    Gdn,
    Table,
    Vision,
}

/// One unit of the flat lane's take order for target KV memory the live
/// headroom refused, `shortfall` bytes short (Splash `Cache::takeReclaim`):
/// free pool extents first; a lossless store compaction (mlx-node only);
/// the dead leaves a cut can reach; then one unit by its own stamp, a state
/// on a tie: a store step, a ladder or history entry, or an idle table,
/// offered only while releasing tables brings an extent cut within reach
/// (Splash's `leadsToReach` on leaves). A unit that frees nothing goes into
/// `skip`. With none left, one image feature entry nothing holds (Splash's
/// Rebuild stage). A cut's one-layer copy comes out of the process reserve
/// the live headroom keeps back. `None` when nothing is left.
fn flat_reclaim_one<T: DFlash2Target>(
    target: &mut T,
    live: u64,
    shortfall: u64,
    skip: &mut HashSet<UnitKey>,
) -> Result<Option<Reclaimed>> {
    let pool_cuttable = target
        .paged_adapter_ref()
        .is_some_and(PagedKVCacheAdapter::grows_by_extent);
    if pool_cuttable && let Some(adapter) = target.paged_adapter_mut_ref() {
        let released = adapter.try_release_pool_extents(shortfall, false);
        if released > 0 {
            return Ok(Some(Reclaimed::Pool(released)));
        }
    }
    if let Some(step) = target.dflash2_states_mut().store.compact_one(live, None)? {
        log_reclaim(GrowReclaim::Compaction);
        return Ok(Some(Reclaimed::State(step)));
    }
    if pool_cuttable && let Some(adapter) = target.paged_adapter_mut_ref() {
        let released = adapter.try_release_pool_extents(shortfall, true);
        if released > 0 {
            return Ok(Some(Reclaimed::Pool(released)));
        }
    }
    loop {
        let state = oldest_state_unit(&*target, false, skip, Keep::default(), false);
        let table = target.paged_adapter_ref().and_then(|adapter| {
            if !adapter.extent_within_reach(shortfall) {
                return None;
            }
            adapter
                .live_seq_ids()
                .into_iter()
                .filter(|&seq_id| !skip.contains(&UnitKey::Table(seq_id)))
                .map(|seq_id| (adapter.turn_end_stamp(seq_id).unwrap_or(0), seq_id))
                .min()
                .map(|(stamp, seq_id)| Unit {
                    stamp,
                    key: UnitKey::Table(seq_id),
                })
        });
        let Some(unit) = older_first(state, table) else {
            return Ok(release_one_rebuilt(&*target).then_some(Reclaimed::Vision));
        };
        let taken = match &unit.key {
            UnitKey::Table(seq_id) => {
                let released = target
                    .paged_adapter_mut_ref()
                    .map(|adapter| adapter.release_request_for(*seq_id));
                match released {
                    Some(Ok(_)) => {
                        log_reclaim(GrowReclaim::Table);
                        Some(Reclaimed::Table)
                    }
                    Some(Err(error)) => {
                        tracing::warn!(
                            target: "mlx_core::inference",
                            "DFlash2 paged request {seq_id} release failed: {error}"
                        );
                        None
                    }
                    None => None,
                }
            }
            key => take_state_unit(target, key).map(|taken| match taken {
                Taken::Store(step) => Reclaimed::State(step),
                Taken::Gdn => Reclaimed::Gdn,
            }),
        };
        if let Some(taken) = taken {
            return Ok(Some(taken));
        }
        target.sync_gdn_state_marks();
        skip.insert(unit.key);
    }
}

/// The transient loop's reclaim step: free pool extents, one parked store
/// step, then every idle table and the dead leaves a cut can reach. It
/// never drops a GDN entry.
fn reclaim_one_for_dflash2_transient<T: DFlash2Target>(
    target: &mut T,
    live: u64,
    shortfall: u64,
) -> Result<Option<Reclaimed>> {
    let pool_cuttable = |target: &T| {
        target
            .paged_adapter_ref()
            .is_some_and(|adapter| adapter.grows_by_extent())
    };
    if pool_cuttable(target)
        && let Some(adapter) = target.paged_adapter_mut_ref()
    {
        let released = adapter.try_release_pool_extents(shortfall, false);
        if released > 0 {
            return Ok(Some(Reclaimed::Pool(released)));
        }
    }
    if let Some(step) = target.dflash2_states_mut().store.evict_one(live, None)? {
        return Ok(Some(Reclaimed::State(step)));
    }
    if !pool_cuttable(target) {
        return Ok(None);
    }
    target.release_paged_requests()?;
    let released = target.paged_adapter_mut_ref().map_or(0, |adapter| {
        adapter.try_release_pool_extents(shortfall, true)
    });
    Ok((released > 0).then_some(Reclaimed::Pool(released)))
}

/// Folds one reclaim step into `growth`.
fn add_reclaimed(growth: &mut KvGrowth, step: Reclaimed) {
    match step {
        Reclaimed::Pool(bytes) => {
            growth.released_pool_bytes = growth.released_pool_bytes.saturating_add(bytes);
        }
        Reclaimed::State(step) => add_eviction(&mut growth.evicted, step),
        Reclaimed::Gdn => growth.dropped_gdn_states += 1,
        Reclaimed::Table => growth.released_tables += 1,
        Reclaimed::Vision => growth.released_vision_entries += 1,
    }
}

/// Brings parked states within the state budget. A demotion shrinks the
/// demoted state's KV only when `headroom` holds the copy.
pub(crate) fn evict_over_state_budget(
    store: &mut super::dflash2_state::store::HybridStateStore<FlatBacking>,
    headroom: u64,
) -> Result<EvictReport> {
    store.evict_for(
        Reservation {
            state_bytes: 0,
            kv_bytes: 0,
        },
        headroom,
        None,
    )
}

/// Size every flat full-attention cache for the prompt plus one verify
/// before prefill, as Splash books a turn's first work: a cold turn only
/// records the size for the first allocation, a warm one copies its prefix
/// once. Decode grows the rest through [`DFlash2DecodeKv`]. Parked states
/// first come within the state budget, and afterwards give way to the
/// turn's transients.
pub(crate) fn reserve_dflash2_turn_kv<T: DFlash2Target>(
    target: &mut T,
    prompt_len: usize,
    plan: DFlash2KvPlan,
) -> Result<EvictReport> {
    let block_size = target.dflash2().map_or(1, |draft| draft.config.block_size);
    let needed = i64::try_from(prompt_len.saturating_add(block_size + 1)).unwrap_or(i64::MAX);
    let headroom = dflash2_kv_headroom_now(&*target).unwrap_or(0);
    let over_budget = evict_over_state_budget(&mut target.dflash2_states_mut().store, headroom)?;
    let mut growth = ensure_dflash2_target_rows(
        target,
        needed,
        plan,
        KvGrowthPhase::TurnStart { prompt_len },
    )?;
    add_eviction(&mut growth.evicted, over_budget);
    // The turn's transients have no fixed reservation here, unlike Splash's
    // scratch: reclaimed memory gives way to them, best effort.
    if let Some(rows) = min_fa_reserved_rows(target.caches())? {
        let transient = turn_memory(target, prompt_len, rows.max(0) as u64)?.1;
        while let Some(live) = dflash2_kv_headroom_now(&*target).filter(|&live| live < transient) {
            let Some(step) = reclaim_one_for_dflash2_transient(target, live, transient - live)?
            else {
                break;
            };
            add_reclaimed(&mut growth, step);
        }
    }
    if growth.evicted.freed_kv_bytes > 0 || growth.evicted.freed_state_bytes > 0 {
        crate::array::clear_cache();
    }
    tracing::debug!(
        target: "mlx_core::inference",
        prompt_len,
        cap_rows = plan.cap_rows,
        extent_rows = plan.extent_rows,
        reserved_rows = min_fa_reserved_rows(target.caches())?,
        evicted_states = growth.evicted.demoted
            + growth.evicted.dropped_captures
            + growth.evicted.dropped_ends,
        freed_kv_bytes = growth.evicted.freed_kv_bytes,
        released_pool_bytes = growth.released_pool_bytes,
        "DFlash2 target KV reservation"
    );
    Ok(growth.evicted)
}

/// The decode stepper's side of the target KV: it grows the KV before
/// every verify and keeps the decode-phase allocator ceiling sized to the
/// KV it holds.
pub(crate) struct DFlash2DecodeKv {
    plan: Option<DFlash2KvPlan>,
    /// The live ceiling and the KV rows it was sized for.
    decode_limit: Option<(i64, crate::cache_limit::DecodeCacheLimitGuard)>,
}

impl DFlash2DecodeKv {
    pub(crate) fn begin<T: DFlash2Target>(target: &T, plan: Option<DFlash2KvPlan>) -> Result<Self> {
        let mut kv = Self {
            plan,
            decode_limit: None,
        };
        if plan.is_some() {
            kv.push_decode_limit(target)?;
        }
        Ok(kv)
    }

    /// The new ceiling is pushed before the old guard drops: live guards
    /// compose by `min`, so the old one must not outlive it.
    fn push_decode_limit<T: DFlash2Target>(&mut self, target: &T) -> Result<()> {
        let Some(rows) = min_fa_reserved_rows(target.caches())? else {
            return Ok(());
        };
        let block_size = target.dflash2().map_or(1, |draft| draft.config.block_size);
        let guard = crate::cache_limit::coordinator().push_decode_limit(
            dflash2_decode_cache_limit(&target.dense_config(), block_size, rows.max(0) as u64),
        );
        self.decode_limit = Some((rows, guard));
        Ok(())
    }

    /// KV rows the live decode ceiling was sized for.
    #[cfg(test)]
    pub(crate) fn decode_limit_rows(&self) -> Option<i64> {
        self.decode_limit.as_ref().map(|(rows, _)| *rows)
    }

    /// Grow the target KV for a verify writing `rows` rows past the
    /// frontier.
    pub(crate) fn ensure_ahead<T: DFlash2Target>(
        &mut self,
        target: &mut T,
        rows: usize,
    ) -> Result<()> {
        let Some(plan) = self.plan else {
            return Ok(());
        };
        if min_fa_reserved_rows(target.caches())?.is_none() {
            return Ok(());
        }
        let frontier = flat_attention_frontier(target).ok_or_else(|| {
            Error::from_reason(format!(
                "{} DFlash2 target caches disagree on their frontier",
                target.label()
            ))
        })?;
        let needed = i64::try_from(frontier.saturating_add(rows)).unwrap_or(i64::MAX);
        if ensure_dflash2_target_rows(target, needed, plan, KvGrowthPhase::Decode)?.grew {
            self.push_decode_limit(target)?;
        }
        Ok(())
    }
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

#[cfg(test)]
thread_local! {
    /// Final-token KV growths this thread refuses next.
    pub(crate) static REFUSE_FINAL_TOKEN_GROWTH: std::cell::Cell<u32> =
        const { std::cell::Cell::new(0) };
}

/// Writes the turn's last token into the caches so the next turn resumes
/// past it. Splash gives that terminal token no KV row, so the write evicts
/// no parked state, and a growth the budget refuses skips it (`Ok(false)`)
/// instead of failing the turn.
pub(crate) fn dflash2_materialize_final<T: DFlash2Target>(
    target: &mut T,
    token: u32,
    stream: Stream,
    plan: DFlash2KvPlan,
) -> Result<bool> {
    let label = target.label();
    let tap_layers = target
        .dflash2()
        .ok_or_else(|| Error::from_reason(format!("{label} DFlash2 model disappeared")))?
        .config
        .target_layers
        .clone();
    let input = MxArray::from_int32(&[token as i32], &[1, 1])?;
    let _stream = StreamContext::new(stream);
    if let Some(frontier) = flat_attention_frontier(target) {
        #[cfg(test)]
        if REFUSE_FINAL_TOKEN_GROWTH.with(|left| left.replace(left.get().saturating_sub(1))) > 0 {
            return Ok(false);
        }
        let needed = i64::try_from(frontier.saturating_add(1)).unwrap_or(i64::MAX);
        let growth = try_ensure_dflash2_target_rows_with(
            target,
            needed,
            plan,
            KvGrowthPhase::Decode,
            dflash2_kv_headroom_now,
            false,
        )?;
        if let Err(refused) = growth {
            tracing::debug!(
                target: "mlx_core::inference",
                "DFlash2 final token left out of the cache: {}",
                refused.reason
            );
            return Ok(false);
        }
    }
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
    target.eval_layer_caches()?;
    Ok(true)
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
    let last_used = states.live_last_used;
    match states
        .store
        .insert_at(Lineage::text(&tokens, salt), parts, last_used)
    {
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
    let states = target.dflash2_states_mut();
    states.live_anchor = parts.anchor;
    states.live_last_used = states.store.recency().next();
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
        target.context_tokens(),
        target.dflash2_window(),
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
    let reserved = dflash2_kv_plan(target, dflash_context_capacity).and_then(|plan| {
        let _stream = StreamContext::new(generation_stream);
        reserve_dflash2_turn_kv(target, tokens.len(), plan).map(|evicted| (plan, evicted))
    });
    let (kv_plan, evicted) = match reserved {
        Ok(reserved) => reserved,
        Err(error) => return Err(dflash2_turn_failed(target, error, cached_prefix)),
    };
    let progress = args.sink.and_then(|sink| sink.progress());
    if let Some(progress) = progress {
        progress.begin_prefill(prefill.len() as u32, cached_prefix as u32);
    }
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
    if let Some(progress) = progress {
        progress.first_token();
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
    *target.dflash2_turn_state_mut() = Some(DFlash2TurnState {
        kv: Some(kv_plan),
        ..state
    });
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
        match dflash2_materialize_final(target, last, generation_stream, kv_plan) {
            Ok(materialized) => last_in_cache = materialized,
            Err(error) => return Err(dflash2_turn_failed(target, error, cached_prefix)),
        }
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
    states.live_last_used = states.store.recency().next();
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

    /// A kept prompt protects only the points its lookup can restore, its
    /// last token left out: a point at its own end is not kept, keyed or
    /// not; a shorter one is.
    /// Mutation: pass the whole prompt to the token test in `Keep::points`
    /// → red.
    #[test]
    fn kept_points_leave_the_prompts_last_token() {
        let tokens: Vec<u32> = (1..=32).collect();
        let keyless = Keep {
            tokens: &tokens,
            keys: None,
        }
        .points(Some(16));
        assert!(!keyless.keeps(&tokens, None), "its own end");
        assert!(keyless.keeps(&tokens[..16], None));
        let keys = vec![Vec::new(); 2];
        let keyed = Keep {
            tokens: &tokens,
            keys: Some((&keys, 0)),
        };
        let chain = keyed.hit_chain(16).expect("keys");
        assert_eq!(chain.len(), 1);
        let keyed = keyed.points(Some(16));
        assert!(!keyed.keeps(&tokens, None), "a history entry at its end");
        assert!(keyed.keeps(&tokens[..16], None));
        assert!(keyed.keeps(&tokens[..16], Some(&chain)));
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

    /// With no KV headroom a turn that must grow its KV evicts every parked
    /// state, one step at a time, then fails with `context_length_exceeded`
    /// before prefill, as Splash fails a lone request. Mutation: grow
    /// instead of returning the error in `ensure_dflash2_target_rows_with`
    /// → red (the turn completes).
    #[test]
    fn kv_pressure_evicts_parked_states_then_fails_the_turn() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &body(4, 40));
            turn(&mut inner, &tokenizer, &body(5, 40));
            assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{name}");
            let _override = BudgetOverride;
            KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
            let prompt = body(6, 300);
            let error = run_tiny_dflash_turn(&mut inner, &tokenizer, &prompt, true, MAX_NEW)
                .expect_err("no headroom for the KV");
            assert!(
                error.reason.starts_with("context_length_exceeded: "),
                "{name}: {}",
                error.reason
            );
            assert_eq!(inner.dflash2_states.store.bytes().entries, 0, "{name}");
        }
        Ok(())
    }

    struct BudgetOverride;
    impl Drop for BudgetOverride {
        fn drop(&mut self) {
            KV_BUDGET_OVERRIDE.with(|budget| budget.set(None));
        }
    }

    struct ExtentOverride;
    impl Drop for ExtentOverride {
        fn drop(&mut self) {
            KV_EXTENT_ROWS_OVERRIDE.with(|rows| rows.set(Some(256)));
        }
    }

    /// Rows a growth books: whole extents, at most the turn's cap. On the 27B
    /// (4,096-row extent, a 139,264-row cap) a turn whose prompt plus 8
    /// verify rows is 4,096 books one extent, one row more books two, and the
    /// last extent stops at the cap.
    #[test]
    fn plan_books_whole_extents_up_to_the_cap() {
        let plan = DFlash2KvPlan {
            cap_rows: 139_264,
            extent_rows: 4_096,
        };
        assert_eq!(plan.target_rows(4_088 + 8), 4_096);
        assert_eq!(plan.target_rows(4_089 + 8), 8_192);
        assert_eq!(plan.target_rows(139_000), 139_264);
        assert_eq!(plan.target_rows(139_264), 139_264);
        let short = DFlash2KvPlan {
            cap_rows: 139_257 + 7,
            extent_rows: 4_096,
        };
        assert_eq!(short.target_rows(137_000), 139_264);
    }

    /// A warm turn books its prompt plus one verify of target KV, rounded up
    /// to the extent, not its whole budget; decode grows the rest.
    /// Mutations: book `plan.cap_rows` in `reserve_dflash2_turn_kv` → red
    /// (the first turn already holds 1,026 ≠ 256); grow with the cache's
    /// 256-row step (`self.reserve(rows)` in `KVCache::grow_exact`) → red
    /// (512 ≠ 384).
    #[test]
    fn turn_books_prompt_plus_verify_rows() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &body(4, 40));
            assert_eq!(min_fa_capacity(&inner.caches)?, Some(256), "{name}");
            let _budget = BudgetOverride;
            KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(1 << 40)));
            let _extent = ExtentOverride;
            KV_EXTENT_ROWS_OVERRIDE.with(|rows| rows.set(Some(128)));
            let plan = dflash2_kv_plan(&inner, 1_024)?;
            let block_size = inner.dflash2.as_ref().expect("draft").config.block_size;
            assert_eq!(plan.cap_rows, 1_024 + block_size as i64, "{name}");
            let prompt = 300 - block_size - 1;
            reserve_dflash2_turn_kv(&mut inner, prompt, plan)?;
            assert_eq!(min_fa_capacity(&inner.caches)?, Some(384), "{name}");
        }
        Ok(())
    }

    /// A real turn grows its target KV during decode in the memory plan's
    /// extent, not by the cache's own 256-row step: a warm 300-token prompt
    /// books 320 rows of 160-row extents and 63 decoded tokens grow them to
    /// 480. Mutations: install the turn state with `kv: None` in
    /// `dflash2_chat_turn` → red (574); ignore the plan's extent in
    /// `dflash2_kv_plan` → red (1,026).
    #[test]
    fn turn_grows_decode_kv_in_the_planned_extent() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &body(4, 40));
            let mut prompt = ChatBackend::cached_token_history(&inner).to_vec();
            prompt.extend(body(5, 300 - prompt.len() as u32));
            assert_eq!(min_fa_capacity(&inner.caches)?, Some(256), "{name}");
            let _extent = ExtentOverride;
            KV_EXTENT_ROWS_OVERRIDE.with(|rows| rows.set(None));
            let _budget = BudgetOverride;
            KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(1 << 40)));
            let trained = inner.config.max_position_embeddings as u32;
            inner.memory_plan =
                Some(crate::engine::memory_plan::MemoryPlan::with_speculative_window(trained, 5));
            let cached =
                run_tiny_dflash_turn(&mut inner, &tokenizer, &prompt, true, 64)?.cached_tokens;
            assert!(cached > 0, "{name}: the turn must continue the live state");
            let frontier = flat_attention_frontier(&inner).expect("frontier");
            assert!(frontier >= 363, "{name}: frontier {frontier}");
            assert_eq!(min_fa_capacity(&inner.caches)?, Some(480), "{name}");
        }
        Ok(())
    }

    /// The final token of a length-finished turn whose KV is full and whose
    /// growth the budget refuses stays out of the cache, and the turn
    /// succeeds: Splash gives the terminal token no KV row, so the write
    /// evicts no parked state for it either. Mutations: return the refusal
    /// from `dflash2_materialize_final` as an error → red
    /// (`context_length_exceeded`); pass `reclaim: true` there → red (the
    /// parked state is dropped).
    #[test]
    fn refused_final_token_growth_leaves_it_uncached() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &body(4, 40));
            turn(&mut inner, &tokenizer, &body(6, 40));
            assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{name}");
            let capacity = min_fa_capacity(&inner.caches)?.expect("FA caches");
            let frontier = flat_attention_frontier(&inner).expect("frontier");
            let stream = Stream::generation();
            {
                let _stream = StreamContext::new(stream);
                let fill = (frontier..capacity as usize)
                    .map(|i| (i % 9 + 1) as i32)
                    .collect::<Vec<_>>();
                let input = MxArray::from_int32(&fill, &[1, fill.len() as i64])?;
                let taps = inner
                    .dflash2
                    .as_ref()
                    .expect("draft")
                    .config
                    .target_layers
                    .clone();
                inner.forward_with_taps(&input, &taps, None)?;
                inner.eval_layer_caches()?;
            }
            assert_eq!(flat_attention_frontier(&inner), Some(capacity as usize));
            let plan = DFlash2KvPlan {
                cap_rows: 1_000,
                extent_rows: 256,
            };
            let _budget = BudgetOverride;
            KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
            let materialized = dflash2_materialize_final(&mut inner, 3, stream, plan)?;
            assert!(!materialized, "{name}");
            assert_eq!(min_fa_capacity(&inner.caches)?, Some(capacity), "{name}");
            assert_eq!(flat_attention_frontier(&inner), Some(capacity as usize));
            assert_eq!(
                inner.dflash2_states.store.bytes().entries,
                1,
                "{name}: the final token's write evicted a parked state"
            );
        }
        Ok(())
    }

    /// A length-finished turn whose final token's KV growth is refused saves
    /// its history at the KV frontier, without that token. Mutation: set
    /// `last_in_cache = true` for any `Ok` from `dflash2_materialize_final`
    /// in `dflash2_chat_turn` → red (history one past the frontier).
    #[test]
    fn refused_final_token_turn_saves_history_at_the_frontier() -> Result<()> {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                REFUSE_FINAL_TOKEN_GROWTH.with(|left| left.set(0));
            }
        }
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            let _reset = Reset;
            REFUSE_FINAL_TOKEN_GROWTH.with(|left| left.set(1));
            let prompt = body(4, 40);
            let result = run_tiny_dflash_turn(&mut inner, &tokenizer, &prompt, true, MAX_NEW)?;
            assert_eq!(result.finish_reason, "length", "{name}");
            assert_eq!(
                REFUSE_FINAL_TOKEN_GROWTH.with(std::cell::Cell::get),
                0,
                "{name}: the turn must write its final token"
            );
            let frontier = flat_attention_frontier(&inner).expect("frontier");
            let history = ChatBackend::cached_token_history(&inner).len();
            assert_eq!(history, frontier, "{name}");
            assert_eq!(history, prompt.len() + MAX_NEW as usize - 1, "{name}");
        }
        Ok(())
    }

    /// A turn keeps a parked state when the headroom covers the KV it books
    /// and the transients at that size, not at its whole window. Mutation:
    /// size the turn-start transient at `plan.cap_rows` in
    /// `dflash2_kv_growth_need` → red (state evicted).
    #[test]
    fn long_budget_turn_keeps_the_parked_state() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &body(4, 40));
            turn(&mut inner, &tokenizer, &body(5, 40));
            let before = inner.dflash2_states.store.bytes();
            assert_eq!(before.entries, 1, "{name}");
            let plan = DFlash2KvPlan {
                cap_rows: 100_000,
                extent_rows: 256,
            };
            let prompt = 300usize;
            let phase = KvGrowthPhase::TurnStart { prompt_len: prompt };
            let need = dflash2_kv_growth_need(&inner, prompt as i64 + 3, plan, phase)?
                .expect("the turn grows its KV");
            assert_eq!(need.rows, 512, "{name}");
            let (_, whole_transient) = turn_memory(&inner, prompt, plan.cap_rows as u64)?;
            assert!(whole_transient > need.transient_bytes, "{name}");
            let _override = BudgetOverride;
            KV_BUDGET_OVERRIDE
                .with(|budget| budget.set(Some(need.kv_bytes + need.transient_bytes)));
            reserve_dflash2_turn_kv(&mut inner, prompt, plan)?;
            assert_eq!(inner.dflash2_states.store.bytes(), before, "{name}");
            assert_eq!(min_fa_capacity(&inner.caches)?, Some(512), "{name}");
        }
        Ok(())
    }

    /// Two parked states and a headroom that rises with every byte the
    /// store frees, one byte short of the growth's KV budget: the growth
    /// evicts exactly one step, then grows. Mutations: take every unit at
    /// once (loop `flat_reclaim_one` after the first take in
    /// `try_ensure_dflash2_target_rows_with`) → red (2 steps); skip the
    /// `flat_reclaim_one` call there → red (`context_length_exceeded`).
    #[test]
    fn ensure_evicts_one_parked_state_then_grows() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            for seed in [4, 5, 6] {
                turn(&mut inner, &tokenizer, &body(seed, 40));
            }
            assert_eq!(inner.dflash2_states.store.bytes().entries, 2, "{name}");
            let plan = DFlash2KvPlan {
                cap_rows: 1_000,
                extent_rows: 256,
            };
            let need = dflash2_kv_growth_need(&inner, 300, plan, KvGrowthPhase::Decode)?
                .expect("the KV grows");
            assert_eq!(need.transient_bytes, 0, "{name}");
            let held = |inner: &Qwen35Inner| {
                let bytes = inner.dflash2_states.store.bytes();
                bytes.kv_bytes + bytes.state_bytes
            };
            let start = held(&inner);
            let base = need.kv_bytes - 1;
            let growth = ensure_dflash2_target_rows_with(
                &mut inner,
                300,
                plan,
                KvGrowthPhase::Decode,
                |inner| Some(base + (start - held(inner))),
            )?;
            let report = growth.evicted;
            assert!(growth.grew, "{name}");
            assert_eq!(
                report.compacted + report.demoted + report.dropped_captures + report.dropped_ends,
                1,
                "{name}: {report:?}"
            );
            assert!(inner.dflash2_states.store.bytes().entries >= 1, "{name}");
            assert_eq!(min_fa_capacity(&inner.caches)?, Some(512), "{name}");
        }
        Ok(())
    }

    /// A growth the count can meet only for its KV keeps taking toward the
    /// KV plus its turn transients: the headroom covers the KV after the
    /// first step and never the transients, so every parked state goes and
    /// the KV grows.
    /// Mutation: stop taking once the KV fits (the old `Hard` tier's met
    /// check) in `try_ensure_dflash2_target_rows_with` → red (one step, a
    /// state stays).
    #[test]
    fn kv_only_reach_keeps_taking_toward_the_transients() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            for seed in [4, 5, 6] {
                turn(&mut inner, &tokenizer, &body(seed, 40));
            }
            assert_eq!(inner.dflash2_states.store.bytes().entries, 2, "{name}");
            let plan = DFlash2KvPlan {
                cap_rows: 1_000,
                extent_rows: 256,
            };
            let phase = KvGrowthPhase::TurnStart { prompt_len: 280 };
            let need = dflash2_kv_growth_need(&inner, 300, plan, phase)?.expect("the KV grows");
            assert!(need.transient_bytes > 1, "{name}");
            let held = |inner: &Qwen35Inner| {
                let bytes = inner.dflash2_states.store.bytes();
                bytes.kv_bytes + bytes.state_bytes
            };
            let start = held(&inner);
            let base = need.kv_bytes - 1;
            let growth = ensure_dflash2_target_rows_with(&mut inner, 300, plan, phase, |inner| {
                Some(base + u64::from(held(inner) < start))
            })?;
            let report = growth.evicted;
            assert!(growth.grew, "{name}");
            assert!(
                report.compacted + report.demoted + report.dropped_captures + report.dropped_ends
                    >= 2,
                "{name}: {report:?}"
            );
            assert_eq!(inner.dflash2_states.store.bytes().entries, 0, "{name}");
        }
        Ok(())
    }

    /// A growth whose KV even every state could not cover takes nothing: it
    /// fails with `context_length_exceeded`, the parked state and the live
    /// KV as they were.
    /// Mutation: drop the tier gate in `try_ensure_dflash2_target_rows_with`
    /// → red (0 entries).
    #[test]
    fn doomed_flat_growth_keeps_parked_states() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &body(4, 40));
            turn(&mut inner, &tokenizer, &body(5, 40));
            let plan = DFlash2KvPlan {
                cap_rows: 4_096,
                extent_rows: 256,
            };
            let offset = flat_attention_frontier(&inner);
            let need = dflash2_kv_growth_need(&inner, 4_000, plan, KvGrowthPhase::Decode)?
                .expect("the KV grows");
            assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{name}");
            assert!(
                state_reach_bytes(&inner, Keep::default(), false) < need.kv_bytes,
                "{name}: the state cannot cover the KV"
            );
            let error = ensure_dflash2_target_rows_with(
                &mut inner,
                4_000,
                plan,
                KvGrowthPhase::Decode,
                |_| Some(0),
            )
            .expect_err("no room");
            assert!(
                error.reason.starts_with("context_length_exceeded: "),
                "{name}: {}",
                error.reason
            );
            assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{name}");
            assert_eq!(min_fa_capacity(&inner.caches)?, Some(256), "{name}");
            assert_eq!(flat_attention_frontier(&inner), offset, "{name}");
        }
        Ok(())
    }

    /// Blocks of one pool extent in the AR-growth fixtures.
    const POOL_EXTENT: u32 = 4;

    /// The int8 fixture after `turns` DFlash2 turns, on a draft-loaded
    /// runway pool of `POOL_EXTENT`-block extents that an AR request grew by
    /// one extent and released, its blocks cached as a prefix when `cached`.
    /// Returns the runway's blocks and bytes and the AR tokens.
    fn ar_grown_runway(turns: u32, cached: bool) -> Result<(Qwen35Inner, u32, u64, Vec<u32>)> {
        let tokenizer = tokenizer();
        let mut inner = fixture("int8_d256");
        for seed in 4..4 + turns {
            turn(&mut inner, &tokenizer, &body(seed, 40));
        }
        inner.config.paged_cache_memory_mb = Some(256);
        inner.paged_adapter = None;
        inner.initialize_paged_adapter()?;
        let adapter = inner.paged_adapter.as_mut().expect("paged adapter");
        assert!(
            adapter.grows_by_extent(),
            "a draft-loaded pool starts at the runway"
        );
        adapter.set_extent_growth(POOL_EXTENT);
        adapter.set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        let runway = adapter
            .block_telemetry()
            .map_err(Error::from_reason)?
            .total_blocks;
        let block_size = adapter.block_size();
        let runway_bytes =
            u64::from(runway) * adapter.bytes_per_block().map_err(Error::from_reason)?;
        let tokens: Vec<u32> = (0..(runway + POOL_EXTENT) * block_size)
            .map(|i| i % 13 + 1)
            .collect();
        adapter
            .reset_for_new_request(1)
            .map_err(Error::from_reason)?;
        adapter
            .allocate_suffix_blocks(tokens.len() as u32)
            .map_err(Error::from_reason)?;
        adapter.record_tokens(&tokens).map_err(Error::from_reason)?;
        if cached {
            adapter
                .register_full_blocks_for_reuse(&[], 0)
                .map_err(Error::from_reason)?;
        }
        let blocks = adapter.block_telemetry().map_err(Error::from_reason)?;
        assert_eq!(blocks.total_blocks, runway + POOL_EXTENT);
        Ok((inner, runway, runway_bytes, tokens))
    }

    fn pool_blocks(inner: &Qwen35Inner) -> u32 {
        inner
            .paged_adapter
            .as_ref()
            .expect("paged adapter")
            .layer_kv_pool()
            .num_blocks()
    }

    fn pool_bytes(inner: &Qwen35Inner) -> u64 {
        inner
            .paged_adapter
            .as_ref()
            .map_or(0, |adapter| adapter.layer_kv_pool().total_bytes())
    }

    /// Tokens of `tokens` the pool's prefix cache still hits.
    fn cached_prefix(inner: &Qwen35Inner, tokens: &[u32]) -> usize {
        let adapter = inner.paged_adapter.as_ref().expect("paged adapter");
        let mut allocator = adapter.allocator().lock().expect("allocator");
        let (hit, cached) = allocator.find_longest_cache_hit(tokens, adapter.block_size(), &[], 0);
        for block in hit {
            allocator.free(block);
        }
        cached
    }

    const KV_PLAN: DFlash2KvPlan = DFlash2KvPlan {
        cap_rows: 1_000,
        extent_rows: 256,
    };

    /// An AR turn grew the runway pool by one cached extent and kept its
    /// request; a DFlash2 turn's booking that fits only once that extent is
    /// gone releases the request, evicts the extent's cached leaf blocks,
    /// cuts the pool back to the runway and grows; the AR prefix's head
    /// still hits. Mutation: skip the dead-leaf step and the table unit in
    /// `flat_reclaim_one` → red (pool not cut).
    #[test]
    fn ar_growth_then_dflash2_booking_reclaims_the_pool() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let (mut inner, runway, runway_bytes, tokens) = ar_grown_runway(1, true)?;
        let phase = KvGrowthPhase::TurnStart { prompt_len: 280 };
        let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
        let base = need.kv_bytes + need.transient_bytes + runway_bytes;
        let growth = ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
            Some(base.saturating_sub(pool_bytes(inner)))
        })?;
        assert!(growth.grew);
        assert_eq!(pool_blocks(&inner), runway);
        assert_eq!(pool_bytes(&inner), runway_bytes);
        assert!(growth.released_pool_bytes > 0);
        assert_eq!(min_fa_capacity(&inner.caches)?, Some(512));
        let block_size = inner.paged_adapter.as_ref().expect("adapter").block_size();
        assert_eq!(
            cached_prefix(&inner, &tokens),
            (runway * block_size) as usize,
            "the AR prefix keeps its head"
        );
        Ok(())
    }

    /// [`ar_growth_then_dflash2_booking_reclaims_the_pool`] for the growth a
    /// verify makes (`DFlash2DecodeKv::ensure_ahead`'s phase). Mutation: skip
    /// the dead-leaf step and the table unit in `flat_reclaim_one` → red
    /// (pool not cut).
    #[test]
    fn decode_growth_after_ar_growth_reclaims_the_pool() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let (mut inner, runway, runway_bytes, _) = ar_grown_runway(1, true)?;
        let phase = KvGrowthPhase::Decode;
        let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
        let base = need.kv_bytes + need.transient_bytes + runway_bytes;
        let growth = ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
            Some(base.saturating_sub(pool_bytes(inner)))
        })?;
        assert!(growth.grew);
        assert_eq!(pool_blocks(&inner), runway);
        assert_eq!(min_fa_capacity(&inner.caches)?, Some(512));
        Ok(())
    }

    /// One parked state and a free pool tail that covers the shortfall: the
    /// tail extent goes and the state stays, as Splash releases free
    /// extents before any victim. Mutation: try the store compaction and the
    /// units before the free extents in `flat_reclaim_one` → red (the state
    /// is evicted).
    #[test]
    fn free_pool_extents_go_before_parked_states() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let (mut inner, runway, runway_bytes, _) = ar_grown_runway(2, false)?;
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .release_request()
            .map_err(Error::from_reason)?;
        assert_eq!(inner.dflash2_states.store.bytes().entries, 1);
        let held = |inner: &Qwen35Inner| {
            let bytes = inner.dflash2_states.store.bytes();
            bytes.kv_bytes + bytes.state_bytes
        };
        let start = held(&inner);
        let phase = KvGrowthPhase::Decode;
        let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
        let base = need.kv_bytes + need.transient_bytes + runway_bytes;
        let growth = ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
            Some(base.saturating_sub(pool_bytes(inner)) + (start - held(inner)))
        })?;
        assert!(growth.grew);
        assert_eq!(pool_blocks(&inner), runway);
        assert_eq!(
            inner.dflash2_states.store.bytes().entries,
            1,
            "the state stays"
        );
        Ok(())
    }

    /// A cut back to the runway that failed part way leaves layer bytes
    /// above the pool's block count. A growth whose KV only those bytes and
    /// every state bring within reach is counted as in reach, and its first
    /// take finishes the cut.
    /// Mutations: drop the residual layer bytes from
    /// `PagedKVCacheAdapter::release_reach_bytes` → red
    /// (context_length_exceeded); drop `&& !pending` from the guard in
    /// `try_release_pool_extents` → red (context_length_exceeded).
    #[test]
    fn failed_cut_at_runway_is_reclaimed_for_flat_growth() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let (mut inner, runway, runway_bytes, _) = ar_grown_runway(1, false)?;
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        adapter.release_request().map_err(Error::from_reason)?;
        let extent_bytes =
            u64::from(POOL_EXTENT) * adapter.bytes_per_block().map_err(Error::from_reason)?;
        mlx_paged_attn::set_shrink_fail_at_layer(Some(1));
        adapter.try_release_pool_extents(extent_bytes, false);
        mlx_paged_attn::set_shrink_fail_at_layer(None);
        assert_eq!(pool_blocks(&inner), runway);
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        assert_eq!(
            adapter.allocator().lock().expect("allocator").num_blocks(),
            runway
        );
        assert!(adapter.layer_kv_pool().shrink_pending());
        let residual = pool_bytes(&inner) - runway_bytes;
        assert!(residual > 0);
        let held = |inner: &Qwen35Inner| {
            let bytes = inner.dflash2_states.store.bytes();
            bytes.kv_bytes + bytes.state_bytes
        };
        let start_held = held(&inner);
        let start_pool = pool_bytes(&inner);
        let states = state_reach_bytes(&inner, Keep::default(), false);
        assert_eq!(states, start_held, "the store holds every state byte");
        let phase = KvGrowthPhase::Decode;
        let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
        let base = need
            .kv_bytes
            .checked_sub(residual + states)
            .expect("the residual and states fall short of the KV");
        let growth = ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
            Some(base + (start_pool - pool_bytes(inner)) + (start_held - held(inner)))
        })?;
        assert!(growth.grew);
        assert_eq!(growth.released_pool_bytes, residual);
        assert!(
            !inner
                .paged_adapter
                .as_ref()
                .expect("adapter")
                .layer_kv_pool()
                .shrink_pending()
        );
        assert_eq!(pool_bytes(&inner), runway_bytes);
        Ok(())
    }

    /// An AR grow that headroom held to 2 blocks of a `POOL_EXTENT`-block
    /// extent, released: a growth whose KV only that partial extent and
    /// every state bring within reach is counted as in reach and cuts the
    /// pool back to the runway.
    /// Mutation: floor division of the extents in
    /// `PagedKVCacheAdapter::release_reach_bytes` → red
    /// (context_length_exceeded).
    #[test]
    fn partial_extent_is_reclaimed_for_flat_growth() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        let mut inner = fixture("int8_d256");
        turn(&mut inner, &tokenizer, &body(4, 40));
        inner.config.paged_cache_memory_mb = Some(256);
        inner.paged_adapter = None;
        inner.initialize_paged_adapter()?;
        let adapter = inner.paged_adapter.as_mut().expect("paged adapter");
        assert!(adapter.grows_by_extent());
        adapter.set_extent_growth(POOL_EXTENT);
        let runway = adapter
            .block_telemetry()
            .map_err(Error::from_reason)?
            .total_blocks;
        let partial = runway + 2;
        adapter.set_grow_headroom_probe_override(Some(Arc::new(move |blocks| {
            Ok(blocks.min(partial))
        })));
        let bytes_per_block = adapter.bytes_per_block().map_err(Error::from_reason)?;
        let runway_bytes = u64::from(runway) * bytes_per_block;
        adapter
            .reset_for_new_request(1)
            .map_err(Error::from_reason)?;
        adapter
            .allocate_suffix_blocks(partial * adapter.block_size())
            .map_err(Error::from_reason)?;
        adapter.release_request().map_err(Error::from_reason)?;
        adapter.set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        assert_eq!(pool_blocks(&inner), partial, "headroom holds the grow");
        let partial_bytes = 2 * bytes_per_block;
        assert_eq!(pool_bytes(&inner), runway_bytes + partial_bytes);
        let held = |inner: &Qwen35Inner| {
            let bytes = inner.dflash2_states.store.bytes();
            bytes.kv_bytes + bytes.state_bytes
        };
        let start_held = held(&inner);
        let start_pool = pool_bytes(&inner);
        let states = state_reach_bytes(&inner, Keep::default(), false);
        assert_eq!(states, start_held, "the store holds every state byte");
        let phase = KvGrowthPhase::Decode;
        let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
        let base = need
            .kv_bytes
            .checked_sub(partial_bytes + states)
            .expect("the partial extent and states fall short of the KV");
        let growth = ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
            Some(base + (start_pool - pool_bytes(inner)) + (start_held - held(inner)))
        })?;
        assert!(growth.grew);
        assert_eq!(growth.released_pool_bytes, partial_bytes);
        assert_eq!(pool_blocks(&inner), runway);
        assert_eq!(pool_bytes(&inner), runway_bytes);
        Ok(())
    }

    /// One image feature entry nothing holds, `bytes` long.
    fn insert_unheld_image(inner: &Qwen35Inner, bytes: u64) {
        let features = MxArray::from_float32(&[1.0], &[1]).expect("features");
        features.eval();
        inner
            .vision_cache
            .lock()
            .expect("vision cache")
            .insert_for_test(1, features, usize::try_from(bytes).expect("bytes"));
        assert_eq!(rebuilt_bytes(inner), bytes);
    }

    /// The flat lane's growth with a free pool tail that covers it and one
    /// image entry nothing holds: the tail goes, the entry stays.
    /// Mutation: offer the image entry first in `flat_reclaim_one` → red
    /// (`Vision` in the log, the entry gone).
    #[test]
    fn flat_lane_frees_extents_and_states_before_images() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let (mut inner, runway, runway_bytes, _) = ar_grown_runway(2, false)?;
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .release_request()
            .map_err(Error::from_reason)?;
        insert_unheld_image(&inner, 1 << 20);
        let start = parked_bytes(&inner);
        let phase = KvGrowthPhase::Decode;
        let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
        let base = need.kv_bytes + need.transient_bytes + runway_bytes;
        clear_grow_reclaim_log();
        let growth = ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
            Some(base.saturating_sub(pool_bytes(inner)) + (start - parked_bytes(inner)))
        })?;
        assert!(growth.grew);
        assert_eq!(pool_blocks(&inner), runway, "the free tail is cut");
        assert!(!grow_reclaim_log().contains(&GrowReclaim::Vision));
        assert_eq!(growth.released_vision_entries, 0);
        assert_eq!(rebuilt_bytes(&inner), 1 << 20, "the entry stays");
        Ok(())
    }

    /// No pool, no parked state, no GDN entry: only an image entry nothing
    /// holds, as long as the whole shortfall at zero headroom, can meet the
    /// growth. It is counted and taken, and the KV grows.
    /// Mutation: drop `rebuilt_bytes` from `flat_kv_in_reach` → red
    /// (`context_length_exceeded`).
    #[test]
    fn flat_lane_takes_an_image_entry_when_nothing_else_is_left() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &body(4, 40));
            inner.paged_adapter = None;
            assert_eq!(
                state_reach_bytes(&inner, Keep::default(), false),
                0,
                "{name}"
            );
            let phase = KvGrowthPhase::Decode;
            let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
            let bytes = need.kv_bytes + need.transient_bytes;
            assert!(dflash2_kv_growth_shortfall(0, &need) <= bytes, "{name}");
            insert_unheld_image(&inner, bytes);
            clear_grow_reclaim_log();
            let growth =
                ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
                    Some(bytes - rebuilt_bytes(inner))
                })?;
            assert!(growth.grew, "{name}");
            assert_eq!(growth.released_vision_entries, 1, "{name}");
            assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Vision], "{name}");
            assert_eq!(min_fa_capacity(&inner.caches)?, Some(512), "{name}");
        }
        Ok(())
    }

    /// A headroom the grown extent's cut leaves 10 bytes short of the KV:
    /// the count says no take can meet it, so the growth fails at once and
    /// the extent and its cached prefix stay.
    /// Mutation: drop the tier gate in `try_ensure_dflash2_target_rows_with`
    /// → red (the extent is cut).
    #[test]
    fn doomed_flat_growth_keeps_the_grown_extent() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let (mut inner, runway, runway_bytes, tokens) = ar_grown_runway(1, true)?;
        let phase = KvGrowthPhase::Decode;
        let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
        let base = need.kv_bytes - 10 + runway_bytes;
        let error = ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
            Some(base.saturating_sub(pool_bytes(inner)))
        })
        .expect_err("no room even at the runway");
        assert!(
            error.reason.starts_with("context_length_exceeded: "),
            "{}",
            error.reason
        );
        assert_eq!(pool_blocks(&inner), runway + POOL_EXTENT);
        assert_eq!(min_fa_capacity(&inner.caches)?, Some(256));
        assert_eq!(cached_prefix(&inner, &tokens), tokens.len());
        Ok(())
    }

    /// Compacts every parked state, so a test's first flat-lane unit is not
    /// a compaction.
    fn compact_parked_states(inner: &mut Qwen35Inner) -> Result<()> {
        while inner
            .dflash2_states
            .store
            .compact_one(u64::MAX, None)?
            .is_some()
        {}
        Ok(())
    }

    fn parked_bytes(inner: &Qwen35Inner) -> u64 {
        let bytes = inner.dflash2_states.store.bytes();
        bytes.kv_bytes + bytes.state_bytes
    }

    /// The AR table released by hand leaves the grown extent as dead
    /// leaves: the cut takes them before any parked state.
    /// Mutation: run the stamp-ordered units before the dead-leaf step in
    /// `flat_reclaim_one` → red (a store step).
    #[test]
    fn dead_leaves_cut_an_extent_before_a_parked_state() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let (mut inner, runway, runway_bytes, _) = ar_grown_runway(2, true)?;
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .release_request()
            .map_err(Error::from_reason)?;
        compact_parked_states(&mut inner)?;
        assert_eq!(inner.dflash2_states.store.bytes().entries, 1);
        let start = parked_bytes(&inner);
        let phase = KvGrowthPhase::Decode;
        let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
        let base = need.kv_bytes + need.transient_bytes + runway_bytes;
        clear_grow_reclaim_log();
        let growth = ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
            Some(base.saturating_sub(pool_bytes(inner)) + (start - parked_bytes(inner)))
        })?;
        assert!(growth.grew);
        assert_eq!(pool_blocks(&inner), runway);
        assert_eq!(inner.dflash2_states.store.bytes().entries, 1);
        assert_eq!(parked_bytes(&inner), start);
        assert!(grow_reclaim_log().is_empty(), "{:?}", grow_reclaim_log());
        Ok(())
    }

    /// The AR table is kept and every chain block held past it, so no cut
    /// is in reach: the table is no unit and the parked state goes.
    /// Mutation: offer tables in `flat_reclaim_one` without the
    /// `extent_within_reach` filter → red (the table is released).
    #[test]
    fn kv_victims_wait_when_no_extent_is_in_reach() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let (mut inner, _, _, tokens) = ar_grown_runway(2, true)?;
        compact_parked_states(&mut inner)?;
        let held: Vec<_> = {
            let adapter = inner.paged_adapter.as_ref().expect("adapter");
            let mut allocator = adapter.allocator().lock().expect("allocator");
            allocator
                .find_longest_cache_hit(&tokens, adapter.block_size(), &[], 0)
                .0
        };
        let start = parked_bytes(&inner);
        assert!(start > 0);
        let phase = KvGrowthPhase::Decode;
        let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
        let full = need.kv_bytes + need.transient_bytes;
        clear_grow_reclaim_log();
        let growth = ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
            Some(full - 1 + (start - parked_bytes(inner)))
        })?;
        assert!(growth.grew);
        assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Store]);
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        assert!(adapter.block_table_for(1).is_some(), "the AR table stays");
        {
            let mut allocator = adapter.allocator().lock().expect("allocator");
            for block in held {
                allocator.free(block);
            }
        }
        assert_eq!(cached_prefix(&inner, &tokens), tokens.len());
        Ok(())
    }

    /// The flat lane takes the older of an idle table (by its turn-end
    /// stamp) and a parked state. Case a: the table never ended a turn, so
    /// it goes, then the cut; the state stays. Case b: the table's turn
    /// ended after the park, so the state goes first.
    /// Mutations: release every table before the units (the old order) →
    /// red (case b releases the table first); offer tables only after every
    /// state → red (case a takes the state).
    #[test]
    fn flat_lane_takes_the_older_of_an_idle_table_and_a_parked_state() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        for case in ["a", "b"] {
            let (mut inner, runway, runway_bytes, _) = ar_grown_runway(2, true)?;
            compact_parked_states(&mut inner)?;
            if case == "b" {
                inner
                    .paged_adapter
                    .as_mut()
                    .expect("adapter")
                    .finalize_turn_keep_live_no_prefix()
                    .map_err(Error::from_reason)?;
            }
            let start = parked_bytes(&inner);
            let phase = KvGrowthPhase::Decode;
            let need = dflash2_kv_growth_need(&inner, 300, KV_PLAN, phase)?.expect("the KV grows");
            let base = need.kv_bytes + need.transient_bytes + runway_bytes;
            clear_grow_reclaim_log();
            let growth =
                ensure_dflash2_target_rows_with(&mut inner, 300, KV_PLAN, phase, |inner| {
                    Some(base.saturating_sub(pool_bytes(inner)) + (start - parked_bytes(inner)))
                })?;
            assert!(growth.grew, "{case}");
            let log = grow_reclaim_log();
            if case == "a" {
                assert_eq!(log, vec![GrowReclaim::Table], "{case}");
                assert_eq!(pool_blocks(&inner), runway, "{case}");
                assert_eq!(parked_bytes(&inner), start, "{case}");
            } else {
                assert_eq!(log.first(), Some(&GrowReclaim::Store), "{case}: {log:?}");
            }
        }
        Ok(())
    }

    /// The transient loop drops no GDN entry: only the KV growth's take
    /// order reaches the ladder.
    /// Mutation: call `flat_reclaim_one` in the transient loop of
    /// `reserve_dflash2_turn_kv` → red (the entry is dropped).
    #[test]
    fn transient_loop_drops_no_gdn_state() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        let mut inner = fixture("int8_d256");
        turn(&mut inner, &tokenizer, &body(4, 40));
        inner.push_ladder_entry_for_test(16, vec![11], (0..16).collect(), Vec::new(), 1);
        let _override = BudgetOverride;
        KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
        let rows = min_fa_reserved_rows(&inner.caches)?.expect("reserved rows");
        assert!(turn_memory(&inner, 10, rows.max(0) as u64)?.1 > 0);
        reserve_dflash2_turn_kv(&mut inner, 10, KV_PLAN)?;
        assert_eq!(inner.ladder_entries_for_test().len(), 1);
        Ok(())
    }

    /// A vision turn on a flat load evicts parked states one store step at a
    /// time and stops once the headroom covers its prompt plus one verify's
    /// rows and its transients.
    /// Mutation: evict every parked state at once (`store.clear()` in
    /// `evict_parked_dflash2_for_flat_turn_with`) → red (0 ≠ 1 entries).
    /// Mutation: skip the eviction → red (2 ≠ 1 entries).
    #[test]
    fn flat_vision_turn_evicts_one_parked_state_at_a_time() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            for seed in [4, 5, 6] {
                turn(&mut inner, &tokenizer, &body(seed, 40));
            }
            assert_eq!(inner.dflash2_states.store.bytes().entries, 2, "{name}");
            let prompt = 40usize;
            let block_size = inner.dflash2.as_ref().expect("draft").config.block_size;
            let rows = (prompt + block_size + 1) as u64;
            let (row_bytes, transient) = turn_memory(&inner, prompt, rows)?;
            let need = rows * row_bytes + transient;
            evict_parked_dflash2_for_flat_turn_with(&mut inner, prompt, |inner| {
                Some(if inner.dflash2_states.store.bytes().entries > 1 {
                    need - 1
                } else {
                    need
                })
            });
            assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{name}");
        }
        Ok(())
    }

    /// A refused paged step retries after each reclaim that frees memory,
    /// until it runs or nothing is freed; other errors never reclaim.
    /// Mutation: return the first result without retrying → red.
    /// Mutation: retry once only → red (the third run never happens).
    #[test]
    fn refused_paged_step_retries_after_reclaim() {
        let refused = || {
            Err::<u32, _>(Error::from_reason(
                "context_length_exceeded: paged decode could not reserve 1 more block(s)",
            ))
        };
        let mut calls = 0u32;
        let out = retry_after_reclaim(
            &mut calls,
            |calls| {
                *calls += 1;
                if *calls == 1 { refused() } else { Ok(7) }
            },
            |_| true,
        );
        assert_eq!((out.ok(), calls), (Some(7), 2));

        let mut calls = 0u32;
        let out = retry_after_reclaim(
            &mut calls,
            |calls| {
                *calls += 1;
                if *calls < 3 { refused() } else { Ok(7) }
            },
            |_| true,
        );
        assert_eq!((out.ok(), calls), (Some(7), 3));

        let mut calls = 0u32;
        let out = retry_after_reclaim(
            &mut calls,
            |calls| {
                *calls += 1;
                refused()
            },
            |_| false,
        );
        assert!(out.is_err() && calls == 1);

        let mut reclaimed = false;
        let out = retry_after_reclaim(
            &mut reclaimed,
            |_| Err::<u32, _>(Error::from_reason("other failure")),
            |reclaimed| {
                *reclaimed = true;
                true
            },
        );
        assert!(out.is_err() && !reclaimed);
    }

    /// Real adapter over a `initial`-block pool growing to `max` blocks of
    /// `block_size` tokens; `None` without Metal.
    fn growable_adapter(
        initial: u32,
        max: u32,
        block_size: u32,
    ) -> Option<crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter> {
        let config = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            gpu_memory_mb: 256,
            head_size: 64,
            num_kv_heads: 1,
            num_layers: 2,
            max_seq_len: Some(max * block_size),
            max_batch_size: Some(1),
        };
        let pool = match mlx_paged_attn::LayerKVPool::new(
            config,
            initial,
            max,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        ) {
            Ok(pool) => Arc::new(pool),
            Err(err) if crate::test_support::metal_device_absent(&err) => return None,
            Err(err) => panic!("unexpected LayerKVPool::new failure: {err}"),
        };
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            initial, max, block_size,
        )));
        let mut adapter = crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter::new(
            allocator, pool, block_size,
        )
        .expect("growable adapter");
        adapter.set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        Some(adapter)
    }

    /// A blocked serial paged step evicts one parked state and retries: the
    /// other state stays.
    /// Mutation: drop every parked state at once (`store.clear()` in place
    /// of `apply_lossy` in `take_state_unit`) → red (0 ≠ 1).
    #[test]
    fn blocked_paged_step_evicts_one_parked_state_and_retries() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = tokenizer();
        for name in FIXTURES {
            let mut inner = fixture(name);
            for seed in [4, 5, 6] {
                turn(&mut inner, &tokenizer, &body(seed, 40));
            }
            assert_eq!(inner.dflash2_states.store.bytes().entries, 2, "{name}");
            inner.paged_adapter = growable_adapter(4, 16, 16);
            let adapter = inner.paged_adapter.as_mut().expect("adapter");
            adapter.set_grow_headroom_probe_override(Some(Arc::new(|blocks| {
                Ok(if grow_reclaim_log().is_empty() {
                    4
                } else {
                    blocks
                })
            })));
            let bytes_per_block = adapter.bytes_per_block().map_err(Error::from_reason)?;
            let held = |inner: &Qwen35Inner| {
                let bytes = inner.dflash2_states.store.bytes();
                bytes.kv_bytes + bytes.state_bytes
            };
            let start = held(&inner);
            assert!(start >= bytes_per_block, "{name}: the count must pass");
            let _override = BudgetOverride;
            KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
            clear_grow_reclaim_log();
            let mut runs = 0;
            paged_step_reclaiming(&mut inner, |inner| {
                runs += 1;
                if runs == 1 {
                    inner
                        .paged_adapter
                        .as_ref()
                        .expect("adapter")
                        .refuse_for_test(0, 5);
                    Err(Error::from_reason(
                        "context_length_exceeded: paged decode could not reserve 1 more block(s)",
                    ))
                } else {
                    Ok(())
                }
            })?;
            assert_eq!(runs, 2, "{name}");
            assert_eq!(grow_reclaim_log(), vec![GrowReclaim::Store], "{name}");
            assert!(held(&inner) < start, "{name}");
            assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{name}");
        }
        Ok(())
    }

    /// A serial turn's prompt restores its whole cold chain from a runway
    /// pool: the pool grows for the prompt before the lookup, since a restore
    /// allocates from the pool and cannot grow it.
    /// Mutation: skip `try_grow_pool` in `grow_paged_pool_for_prompt` → red
    /// (16 ≠ 64 restored tokens).
    #[test]
    fn cold_restore_grows_pool_before_prepare() {
        const BLOCK: u32 = 8;
        let tokens: Vec<u32> = (0..64u32).map(|i| 700 + i * 3).collect();
        let fingerprint = mlx_paged_attn::ColdCacheFingerprint::from_components([
            b"qwen3-5-runway-restore".as_slice(),
        ]);
        let root = std::env::temp_dir().join(format!(
            "mlx-runway-cold-restore-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let open = || {
            Arc::new(
                mlx_paged_attn::ColdCacheManager::open_at(root.clone(), 16 * 1024 * 1024, 0, 8)
                    .expect("temp-dir cold cache"),
            )
        };
        let tier = |manager: &Arc<mlx_paged_attn::ColdCacheManager>| {
            crate::transformer::paged_kv_cache_adapter::ColdTierContext {
                manager: Arc::clone(manager),
                fingerprint,
                sidecar_policy: None,
            }
        };
        let Some(mut source) = growable_adapter(8, 8, BLOCK) else {
            return;
        };
        let manager = open();
        source.set_cold_tier(tier(&manager));
        source.reset_for_new_request(0).expect("reset");
        source.allocate_suffix_blocks(64).expect("allocate");
        source.record_tokens(&tokens).expect("record");
        assert_eq!(
            source
                .register_full_blocks_for_reuse(&[], 0)
                .expect("register"),
            8
        );
        assert!(manager.drain(std::time::Duration::from_secs(10)));
        drop(source);
        drop(manager);

        let mut prompt = tokens.clone();
        prompt.extend(1..=8);
        let mut restore = growable_adapter(2, 16, BLOCK).expect("Metal");
        let manager = open();
        restore.set_cold_tier(tier(&manager));
        grow_paged_pool_for_prompt(
            &mut restore,
            &prompt,
            0,
            |adapter| Some(adapter),
            |_, _| false,
        );
        let plan = restore
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                0,
                &prompt,
                prompt.len() as u32,
                true,
                &vec![Vec::new(); prompt.len() / BLOCK as usize],
                0,
                false,
                prompt.len() as u32 - 1,
            )
            .expect("prepare");
        assert_eq!(plan.cached_prefix_len, 64);
        drop(restore);
        drop(manager);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// An extent-growth pool grows for a serial prompt that its free blocks
    /// cannot hold before cached prefix blocks are counted: the cached block
    /// survives. Mutation: count reclaimable blocks as available for an
    /// extent pool in `kv_need_met` → red (pool 4 ≠ 6).
    #[test]
    fn prompt_grow_keeps_cached_blocks_on_an_extent_pool() {
        const BLOCK: u32 = 8;
        let Some(mut adapter) = growable_adapter(4, 16, BLOCK) else {
            return;
        };
        adapter.set_extent_growth(2);
        adapter.reset_for_new_request(0).expect("reset");
        adapter.allocate_suffix_blocks(8).expect("allocate");
        adapter
            .record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8])
            .expect("record");
        assert_eq!(
            adapter
                .register_full_blocks_for_reuse(&[], 0)
                .expect("register"),
            1
        );
        adapter.release_request().expect("release");
        let prompt: Vec<u32> = (100..132).collect();
        grow_paged_pool_for_prompt(
            &mut adapter,
            &prompt,
            0,
            |adapter| Some(adapter),
            |_, _| false,
        );
        let blocks = adapter.block_telemetry().expect("telemetry");
        assert_eq!(blocks.total_blocks, 6);
        assert_eq!(blocks.reclaimable_blocks, 1);
    }

    /// A declined grow on an extent pool whose free plus cached blocks hold
    /// the prompt reclaims nothing: the met check counts them.
    /// Mutation: drop the `takeable >= blocks` arm in `kv_need_met` → red
    /// (the reclaim runs).
    #[test]
    fn prompt_grow_counts_held_and_cached_blocks_before_states() {
        const BLOCK: u32 = 8;
        let Some(mut adapter) = growable_adapter(4, 16, BLOCK) else {
            return;
        };
        adapter.set_extent_growth(2);
        adapter.set_grow_headroom_probe_override(Some(Arc::new(|_| Ok(4))));
        adapter.reset_for_new_request(0).expect("reset");
        adapter.allocate_suffix_blocks(8).expect("allocate");
        adapter
            .record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8])
            .expect("record");
        adapter
            .register_full_blocks_for_reuse(&[], 0)
            .expect("register");
        adapter.release_request().expect("release");
        let prompt: Vec<u32> = (100..132).collect();
        let mut reclaimed = false;
        grow_paged_pool_for_prompt(
            &mut adapter,
            &prompt,
            0,
            |adapter| Some(adapter),
            |_, _| {
                reclaimed = true;
                false
            },
        );
        assert!(!reclaimed);
        let blocks = adapter.block_telemetry().expect("telemetry");
        assert_eq!(blocks.total_blocks, 4);
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
