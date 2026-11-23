//! Block-paged forward dispatch helpers for Qwen3.5 (dense + MoE).
//!
//! These helpers implement the same two-pass prefill / per-step decode
//! pattern as LFM2's paged path, but adapted for Qwen3.5's hybrid layer
//! mix (GDN linear-attention layers + Qwen3_5 full-attention layers).
//!
//! Pass 1: GDN-only prefill over the cached prefix tokens (when
//! `cached_prefix_len > 0`) — brings GDN recurrent state up to position
//! `cached_prefix_len`. Attention layers are skipped on this pass; the
//! adapter pool already holds the prefix K/V from a prior request.
//!
//! Pass 2: full forward (GDN + attention) over the SUFFIX tokens.
//! Attention layers attend over `read_kv_range(0, total_ctx)` to recover
//! cached + new context.
//!
//! The decode step is a single-token forward through every layer,
//! gathering K/V from the paged pool for attention layers.
//!
//! Strategy notes (mirrors LFM2/Qwen3.5-MoE):
//! * Full-attention layers reuse K/V through the paged adapter. GDN
//!   layers can only skip prefix replay when the caller has restored a
//!   matching sidecar checkpoint (`gdn_prefix_already_primed=true`);
//!   otherwise this helper replays the cached prefix through GDN.
//! * The two-pass scheme is approximate for GDN over the cached
//!   prefix: the prefix's GDN forward sees a hidden-state stream
//!   produced by passing through ALL layers (including attention)
//!   in pass 1, but the attention layers can't run during pass 1
//!   without their K/V reaching back into the pool — so pass 1 is
//!   GDN-only, with attention layers acting as identity passthroughs
//!   (their MLP / residual contribution is approximated). This is
//!   the same limitation LFM2 has — pure-cache-hit dispatch is not
//!   bit-equal to a fresh prefill on hybrid models.
//!   For the **no-cache** case (cached_prefix_len = 0), pass 1 is
//!   skipped entirely and the result is exact.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::vision::VisionMerge;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::nn::{Embedding, RMSNorm};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use crate::engine::governor::governor::Reservation;

use super::attention::PagedPass;
use super::decoder_layer::{DecoderLayer, Qwen3_5LayerKind};
use super::dflash2_state::paged::PagedAnchor;
use super::draft_context::DraftContextWindow;
use super::layer_cache::Qwen3_5LayerCache;
use crate::models::quantized_linear::LinearProj;

/// Receives a tapped prefill's per-chunk layer outputs (the DFlash2 draft
/// context; Splash `Qwen3_5Target::addCapture` on the prefill graph).
pub(crate) trait PrefillTapSink {
    /// Decoder layers whose output rows are tapped, in tap order.
    fn tap_layers(&self) -> &[usize];
    /// One chunk: prompt rows `[first_row, first_row + chunk.len())` and each
    /// tapped layer's output over them.
    fn chunk(&mut self, first_row: usize, chunk: &[u32], taps: &[MxArray]) -> Result<()>;
    /// What [`Self::chunk`] queued, evaluated before the prefill clears the
    /// cache between chunks.
    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>);
    fn eval(&self) -> Result<()> {
        let mut arrays = Vec::new();
        self.collect_arrays(&mut arrays);
        if arrays.is_empty() {
            Ok(())
        } else {
            MxArray::eval_arrays(&arrays)
        }
    }
    /// Rows inside the prefill a chunk ends at for a state capture,
    /// ascending (Splash `Engine::reserveCaptures`).
    fn capture_rows(&self) -> &[usize] {
        &[]
    }
    /// Whether the chunk ending at `row` captures: only a capture whose
    /// bytes the ledger granted takes a snapshot.
    fn arm(&mut self, _row: usize, _armed: bool) {}
    /// The GDN slots once the prefill reached armed `row`.
    fn capture_linear(&mut self, _row: usize, _linear: Vec<Qwen3_5LayerCache>) {}
    /// The state captured at `row`, once.
    fn take_captured(&mut self, _row: usize) -> Option<PagedAnchor> {
        None
    }
    /// The lane's context window as it is now, which ends at `row`: a
    /// state the ledger refused before its chunk, taken from the lane after
    /// it (Splash `fromLane`); `None` for a lane that keeps no window (a
    /// load without a draft).
    fn snapshot_lane_window(&mut self, row: usize) -> Result<Option<DraftContextWindow>> {
        Err(Error::from_reason(format!(
            "this prefill keeps no lane window at row {row}"
        )))
    }
    /// Writes the lane's context window, which ends at `row`, into `out`
    /// and returns the first row it holds; `None` when the lane holds none
    /// there.
    fn write_lane_window(&mut self, _row: usize, _out: &mut [u8]) -> Result<Option<u32>> {
        Ok(None)
    }
}

/// The chunk breaks of a prefill: its checkpoint boundaries plus the sink's
/// capture rows inside `(cached_prefix_len, full_len)`, ascending.
pub(crate) fn prefill_breaks(
    checkpoint_boundaries: &[u32],
    capture_rows: &[u32],
    cached_prefix_len: u32,
    full_len: usize,
) -> Vec<u32> {
    let mut breaks = checkpoint_boundaries.to_vec();
    breaks.extend(
        capture_rows
            .iter()
            .copied()
            .filter(|&row| row > cached_prefix_len && (row as usize) < full_len),
    );
    breaks.sort_unstable();
    breaks.dedup();
    breaks
}

/// What the ledger said to a capture before its chunk ran (Splash
/// `Engine::reserveCaptures`): its bytes are reserved until the state is
/// stored (`recycled` when the ledger freed units for them), or it was
/// refused and no snapshot is taken during the chunk.
pub(crate) enum CaptureGrant {
    Reserved(Option<Reservation<'static>>, bool),
    /// Refused while the lane holds a checkpoint it may retire: the state
    /// is taken from the lane once the chunk ran (Splash `fromLane`).
    FromLane,
    Refused,
}

impl CaptureGrant {
    /// The chunk snapshots its state.
    pub(crate) fn armed(&self) -> bool {
        matches!(self, Self::Reserved(..))
    }
}

/// What a chunk of a tapped prefill reached (Splash `Engine::apply`).
pub(crate) enum Reached<'a> {
    /// No capture point: its committed blocks register only.
    NoPoint,
    /// The state captured at the point, with the bytes reserved for it and
    /// whether the ledger freed units for them.
    Captured(PagedAnchor, Option<Reservation<'static>>, bool),
    /// A point whose state the lane still holds: the live GDN cell and the
    /// sink's lane window end at it until the next chunk runs.
    FromLane(&'a mut (dyn PrefillTapSink + 'a)),
    /// A point with no state: the use and dedup steps still run, and the
    /// lane, which ends at it, may write it to the SSD tier.
    Refused(&'a mut (dyn PrefillTapSink + 'a)),
}

/// The GDN checkpoint a prefill breaks at. A tapped (DFlash2) prefill takes
/// none: its states are the turn's own points, one owner per block.
pub(crate) fn prefill_ladder(
    adapter: &PagedKVCacheAdapter,
    full_len: usize,
    cached_prefix_len: u32,
    tapped: bool,
) -> Vec<u32> {
    if tapped {
        return Vec::new();
    }
    prefill_checkpoint_boundaries(full_len, cached_prefix_len, adapter.block_size())
}

pub(crate) fn capture_rows_of(taps: Option<&dyn PrefillTapSink>) -> Vec<u32> {
    taps.map_or_else(Vec::new, |sink| {
        sink.capture_rows()
            .iter()
            .filter_map(|&row| u32::try_from(row).ok())
            .collect()
    })
}

/// Books `chunk`'s rows, then claims the capture at `row` when the chunk
/// ends there; the sink snapshots only a granted one. The KV comes first
/// (Splash `Engine::prepare` runs `growLane` before `reserveCaptures`): an
/// optional state is refused rather than taking the room the chunk needs.
/// A chunk that ends the prompt captures nothing; one that ends a prefill
/// command short of it (a scheduler slice) does (Splash publishes a
/// boundary at a command's end).
pub(crate) fn book_chunk<L>(
    target: &mut dyn PrefillTarget<L>,
    taps: Option<&mut (dyn PrefillTapSink + '_)>,
    capture_rows: &[u32],
    chunk: &[u32],
    row: u32,
    prompt_end: bool,
) -> Result<Option<CaptureGrant>> {
    target.record(chunk)?;
    let Some(sink) = taps else {
        return Ok(None);
    };
    if prompt_end || !capture_rows.contains(&row) {
        return Ok(None);
    }
    let grant = target.prepare_capture(row)?;
    sink.arm(row as usize, grant.armed());
    Ok(Some(grant))
}

/// The live GDN slots, evaluated and snapshotted: the state at the row the
/// last chunk ended at.
pub(crate) fn snapshot_live_linear(
    caches: Option<&[Qwen3_5LayerCache]>,
) -> Result<Vec<Qwen3_5LayerCache>> {
    let caches =
        caches.ok_or_else(|| Error::from_reason("a lane state needs the live GDN cell"))?;
    materialize_linear_layer_caches(caches)?;
    snapshot_materialized_linear_layer_caches(caches)
        .ok_or_else(|| Error::from_reason("the live GDN cell is not materialized"))
}

/// The GDN slots at an armed capture row, into the sink.
pub(crate) fn capture_chunk_linear(
    caches: &mut [Qwen3_5LayerCache],
    sink: &mut (dyn PrefillTapSink + '_),
    row: u32,
    materialized: bool,
) -> Result<()> {
    if !materialized {
        materialize_linear_layer_caches(caches)?;
    }
    if let Some(linear) = snapshot_materialized_linear_layer_caches(caches) {
        sink.capture_linear(row as usize, linear);
    }
    Ok(())
}

/// Hands what a chunk of a tapped prefill reached to the target.
pub(crate) fn reach_chunk<L>(
    target: &mut dyn PrefillTarget<L>,
    taps: Option<&mut (dyn PrefillTapSink + '_)>,
    row: u32,
    grant: Option<CaptureGrant>,
) -> Result<()> {
    let Some(sink) = taps else {
        return Ok(());
    };
    let outcome = match grant {
        None => Reached::NoPoint,
        Some(CaptureGrant::Reserved(reservation, recycled)) => {
            match sink.take_captured(row as usize) {
                Some(anchor) => Reached::Captured(anchor, reservation, recycled),
                None => Reached::Refused(sink),
            }
        }
        Some(CaptureGrant::FromLane) => Reached::FromLane(sink),
        Some(CaptureGrant::Refused) => Reached::Refused(sink),
    };
    target.reached(row, outcome)
}

#[cfg(test)]
thread_local! {
    /// A tapped prefill on this thread is cancelled at the chunk starting
    /// at this row.
    pub(crate) static PREFILL_CANCEL_AT: std::cell::Cell<Option<u32>> =
        const { std::cell::Cell::new(None) };
}

fn cancel_hook(_row: u32) -> bool {
    #[cfg(test)]
    return PREFILL_CANCEL_AT.with(std::cell::Cell::get) == Some(_row);
    #[cfg(not(test))]
    false
}

/// What a paged prefill runs on.
pub(crate) struct PrefillParts<'a, L> {
    pub(crate) layers: &'a mut [L],
    pub(crate) caches: &'a mut [Qwen3_5LayerCache],
    pub(crate) final_norm: &'a RMSNorm,
    pub(crate) lm_head: &'a Option<LinearProj>,
    pub(crate) adapter: &'a mut PagedKVCacheAdapter,
}

/// A paged prefill's target: its parts, and the booking of each chunk's
/// rows before the chunk runs (Splash `Engine::growLane`).
pub(crate) trait PrefillTarget<L> {
    fn parts(&mut self) -> Result<PrefillParts<'_, L>>;
    /// Books `chunk`'s rows and advances the cursor over them.
    fn record(&mut self, chunk: &[u32]) -> Result<()>;
    /// Claims the state the chunk ending at `row` captures, after the
    /// chunk's rows are booked and before the chunk runs (Splash
    /// `Engine::prepare`: `growLane`, then `reserveCaptures`).
    fn prepare_capture(&mut self, _row: u32) -> Result<CaptureGrant> {
        Ok(CaptureGrant::Reserved(None, false))
    }
    /// A non-last chunk of a tapped prefill ended at `row` (Splash
    /// `Engine::publishCommittedBlocks`, `publishReachedStateBoundaries`).
    fn reached(&mut self, _row: u32, _outcome: Reached<'_>) -> Result<()> {
        Ok(())
    }
}

impl<L> PrefillTarget<L> for PrefillParts<'_, L> {
    fn parts(&mut self) -> Result<PrefillParts<'_, L>> {
        Ok(PrefillParts {
            layers: &mut *self.layers,
            caches: &mut *self.caches,
            final_norm: self.final_norm,
            lm_head: self.lm_head,
            adapter: &mut *self.adapter,
        })
    }

    fn record(&mut self, chunk: &[u32]) -> Result<()> {
        self.adapter
            .record_tokens(chunk)
            .map_err(Error::from_reason)
    }
}

#[cfg(test)]
thread_local! {
    /// Block counts the prefill's active request holds right before each
    /// chunk's forward.
    pub(crate) static PREFILL_CHUNK_BLOCKS: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Stands in for the configured paged prefill chunk rows on this thread.
    pub(crate) static PREFILL_CHUNK_ROWS_OVERRIDE: std::cell::Cell<Option<i32>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// The DFlash2 verify forwards (whole turn and scheduled lane) this
    /// thread ran.
    pub(crate) static DFLASH2_VERIFY_FORWARDS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
    /// The verify waves this thread ran, and their lanes.
    pub(crate) static DFLASH2_VERIFY_WAVES: std::cell::Cell<(u64, u64)> =
        const { std::cell::Cell::new((0, 0)) };
}

/// `configured`, or the test override on this thread.
pub(crate) fn prefill_chunk_rows(configured: i32) -> i32 {
    #[cfg(test)]
    if let Some(rows) = PREFILL_CHUNK_ROWS_OVERRIDE.with(std::cell::Cell::get) {
        return rows;
    }
    configured
}

#[cfg(test)]
fn note_prefill_chunk_blocks(adapter: &PagedKVCacheAdapter) {
    PREFILL_CHUNK_BLOCKS.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push(adapter.block_table().map_or(0, |table| table.num_blocks()));
        }
    });
}

fn bytes_to_mib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0)
}

/// Queue complete early decode layers while the host builds later layers.
/// These callers are eager paged AR/MTP forwards; prefill and traced/training
/// graphs keep their existing submission boundaries. Hidden-state dependencies
/// include the layer's attention and cache writes, so no cache is published early.
pub(crate) fn maybe_submit_paged_decode_layer(
    layer_idx: usize,
    num_layers: usize,
    hidden: &MxArray,
) {
    static EARLY_LAYERS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let early_layers = *EARLY_LAYERS.get_or_init(|| {
        std::env::var("MLX_QWEN35_DECODE_EARLY_EVAL_LAYERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4)
    });
    submit_paged_layer(layer_idx, num_layers, early_layers, hidden);
}

/// Which layers of a paged verify forward go to the GPU as they are built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VerifySubmit {
    /// The early layers of [`maybe_submit_paged_decode_layer`].
    Decode,
    /// Every layer but the last.
    EveryLayer,
}

#[cfg(test)]
thread_local! {
    /// Layers this thread queued while building a paged forward.
    pub(crate) static PAGED_LAYER_SUBMITS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
    /// Replaces the [`VerifySubmit`] of the DFlash2 verify entries and the
    /// verify wave on this thread.
    pub(crate) static DFLASH2_VERIFY_SUBMIT_OVERRIDE: std::cell::Cell<Option<VerifySubmit>> =
        const { std::cell::Cell::new(None) };
}

/// `submit`, or the test override on this thread.
fn dflash2_verify_submit(submit: VerifySubmit) -> VerifySubmit {
    #[cfg(test)]
    if let Some(submit) = DFLASH2_VERIFY_SUBMIT_OVERRIDE.with(std::cell::Cell::get) {
        return submit;
    }
    submit
}

fn submit_verify_layer(
    submit: VerifySubmit,
    layer_idx: usize,
    num_layers: usize,
    hidden: &MxArray,
) {
    match submit {
        VerifySubmit::Decode => maybe_submit_paged_decode_layer(layer_idx, num_layers, hidden),
        VerifySubmit::EveryLayer => submit_paged_layer(layer_idx, num_layers, num_layers, hidden),
    }
}

fn submit_paged_layer(layer_idx: usize, num_layers: usize, early_layers: usize, hidden: &MxArray) {
    if layer_idx < early_layers
        && layer_idx + 1 < num_layers
        && crate::engine::persistence::compiled_forward_backend_available()
    {
        #[cfg(test)]
        PAGED_LAYER_SUBMITS.with(|submits| submits.set(submits.get() + 1));
        MxArray::async_eval_arrays(&[hidden]);
    }
}

/// Compute the scalar RoPE rotation offset for a paged forward step, decoupling
/// the rotation position from the physical KV slot.
///
/// Image turns compress their ~hundreds of placeholder tokens into far fewer
/// M-RoPE positions, so the running M-RoPE position trails the physical token
/// count by `cached_rope_deltas` (negative). The paged pool still writes K/V at
/// the PHYSICAL slot, but the query/key rotation must use the compressed
/// position so a warm-continuation query lines up with the image-compressed
/// keys it attends over. Text-only turns carry `cached_rope_deltas == 0`, so
/// the result is the physical position unchanged (byte-identical to the prior
/// behaviour).
///
/// `physical_position` is cast to `i32` BEFORE adding the (possibly negative)
/// delta so the arithmetic never underflows a `u32`.
pub(crate) fn paged_rope_offset(physical_position: u32, cached_rope_deltas: i32) -> i32 {
    physical_position as i32 + cached_rope_deltas
}

/// Decide the cross-turn M-RoPE delta to carry into the next paged turn.
///
/// `cached_rope_deltas` is shared model state: an image prefill bakes in a
/// compressed-position delta (negative) that only aligns with the image's
/// physically-resident K/V. The delta is meaningful only when the caller has
/// validated that the selected live or pooled prefix retains that exact image
/// lineage. Every unrelated outcome must drop it:
/// * a cold/fresh turn carries no cross-turn delta;
/// * a NON-live pure-text prefix-cache hit has no image M-RoPE lineage, so the
///   suffix must rotate at the raw physical slot (delta 0). Image-aware callers
///   that restore an image-keyed prefix pass `true` after validating the saved
///   image identity and retain the matching delta.
///
/// Keying the reset on `cached_prefix_len == 0` is therefore too weak: it leaks
/// a stale image delta into unrelated text requests that merely share a cached
/// text prefix.
pub(crate) fn rope_delta_for_paged_turn(
    cached_rope_deltas: Option<i32>,
    preserve_image_lineage: bool,
) -> Option<i32> {
    if preserve_image_lineage {
        cached_rope_deltas
    } else {
        None
    }
}

fn trace_memory_mib() -> (f64, f64, f64) {
    (
        bytes_to_mib(crate::array::get_active_memory()),
        bytes_to_mib(crate::array::get_cache_memory()),
        bytes_to_mib(crate::array::get_peak_memory()),
    )
}

/// Materialized GDN sidecar state for a block-aligned token prefix.
///
/// Full-attention K/V is owned by the paged adapter, so this snapshot keeps
/// only the linear-attention layers' conv/recurrent arrays. Full-attention
/// entries are empty placeholders that preserve layer indexing.
pub(crate) struct MaterializedGdnPrefixCheckpoint {
    pub(crate) prefix_len: u32,
    pub(crate) caches: Vec<Qwen3_5LayerCache>,
}

pub(crate) fn materialize_linear_layer_caches(caches: &[Qwen3_5LayerCache]) -> Result<()> {
    let mut arrays = Vec::new();
    for cache in caches {
        if matches!(cache, Qwen3_5LayerCache::Linear(_)) {
            cache.collect_arrays(&mut arrays);
        }
    }
    if !arrays.is_empty() {
        MxArray::eval_arrays(&arrays)?;
    }
    Ok(())
}

pub(crate) fn snapshot_materialized_linear_layer_caches(
    caches: &[Qwen3_5LayerCache],
) -> Option<Vec<Qwen3_5LayerCache>> {
    let mut snapshot = Vec::with_capacity(caches.len());
    for cache in caches {
        match cache {
            Qwen3_5LayerCache::Linear(arrays) => {
                if arrays.get(0).is_none() || arrays.get(1).is_none() {
                    return None;
                }
                snapshot.push(Qwen3_5LayerCache::Linear(arrays.clone()));
            }
            Qwen3_5LayerCache::FullAttention(_) => {
                snapshot.push(Qwen3_5LayerCache::new_full_attention());
            }
        }
    }
    Some(snapshot)
}

/// Return the largest complete paged-block boundary strictly before the end of
/// the prompt.
///
/// Production prefix lookup is capped at `prompt.len() - 1` so at least one
/// suffix token remains for prefill. Mirroring that cap here keeps the GDN
/// checkpoint on a boundary the next turn can actually restore. This differs from
/// the final complete block only when the prompt itself is block-aligned; in
/// that case it deliberately backs off one block instead of publishing an
/// unreachable checkpoint.
pub(crate) fn gdn_checkpoint_target(
    full_tokens_len: usize,
    cached_prefix_len: u32,
    block_size: u32,
) -> Option<u32> {
    if block_size == 0 {
        return None;
    }
    let full_tokens_len = u32::try_from(full_tokens_len).ok()?;
    let max_cache_hit_tokens = full_tokens_len.checked_sub(1)?;
    let target = max_cache_hit_tokens / block_size * block_size;
    (target > cached_prefix_len).then_some(target)
}

/// Boundaries a prefill forces a chunk break at, ascending: the one deep
/// [`gdn_checkpoint_target`] a warm turn restores from, or none.
pub(crate) fn prefill_checkpoint_boundaries(
    full_tokens_len: usize,
    cached_prefix_len: u32,
    block_size: u32,
) -> Vec<u32> {
    gdn_checkpoint_target(full_tokens_len, cached_prefix_len, block_size)
        .into_iter()
        .collect()
}

/// Rebase absolute checkpoint boundaries onto the uncached suffix the prefill
/// actually forwards. Boundaries at or below the cached prefix are dropped —
/// this prefill never crosses them, so it cannot snapshot there.
pub(crate) fn checkpoint_suffix_offsets(boundaries: &[u32], cached_prefix_len: u32) -> Vec<usize> {
    boundaries
        .iter()
        .filter_map(|boundary| boundary.checked_sub(cached_prefix_len))
        .filter(|offset| *offset > 0)
        .map(|offset| offset as usize)
        .collect()
}

/// Split the suffix into forward-pass chunks, forcing a break at every
/// checkpoint offset so each boundary is a chunk edge the caller can snapshot
/// at. `checkpoint_suffix_offsets` must be ascending.
pub(crate) fn paged_prefill_ranges(
    suffix_len: usize,
    chunk_size: usize,
    checkpoint_suffix_offsets: &[usize],
) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < suffix_len {
        let mut end = start.saturating_add(chunk_size).min(suffix_len);
        if let Some(&checkpoint) = checkpoint_suffix_offsets
            .iter()
            .find(|&&offset| offset > start && offset < end)
        {
            end = checkpoint;
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
}

/// A media prefill's chunks: at most the step the memory plan prices, with a
/// cut at each checkpoint boundary.
pub(crate) fn media_prefill_ranges(
    suffix_len: usize,
    checkpoint_boundaries: &[u32],
    cached_prefix_len: u32,
) -> Vec<Range<usize>> {
    paged_prefill_ranges(
        suffix_len,
        prefill_chunk_rows(crate::models::forward::paged_prefill_chunk_rows() as i32).max(1)
            as usize,
        &checkpoint_suffix_offsets(checkpoint_boundaries, cached_prefix_len),
    )
}

/// Forward the cached-prefix tokens through GDN layers ONLY. Used as
/// "pass 1" of the paged prefill when there is a non-zero cached
/// prefix.
///
/// Skips full-attention layers — their state is reconstructed from the
/// paged pool's prefix cache during pass 2's `read_kv_range`. The
/// hidden_states stream produced by pass 1 is therefore an
/// approximation that omits attention layers' MLP/residual
/// contribution; this is the same trade-off LFM2 makes (see module
/// rustdoc).
pub(crate) fn run_gdn_only_prefill(
    prefix_tokens: &[u32],
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
) -> Result<()> {
    if prefix_tokens.is_empty() {
        return Ok(());
    }
    let input_ids = MxArray::from_uint32(prefix_tokens, &[1, prefix_tokens.len() as i64])?;
    let hidden_states = embed.forward(&input_ids)?;
    run_gdn_only_prefill_embeddings(&hidden_states, layers, caches)
}

/// Forward an already-embedded prefix through GDN layers only.
pub(crate) fn run_gdn_only_prefill_embeddings(
    prefix_inputs_embeds: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
) -> Result<()> {
    if prefix_inputs_embeds.shape_at(1)? == 0 {
        return Ok(());
    }
    let mut hidden_states = prefix_inputs_embeds.clone();

    let num_layers = layers.len();
    #[allow(clippy::needless_range_loop)]
    for layer_idx in 0..num_layers {
        if !layers[layer_idx].is_linear() {
            // Skip attention layers — pass 2 reads their state from
            // the paged pool. Identity-passthrough on hidden_states.
            continue;
        }
        let cache_slot = unsafe {
            let ptr = caches.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };
        hidden_states =
            layers[layer_idx].forward(&hidden_states, None, Some(cache_slot), None, true)?;
    }
    Ok(())
}

/// Replay a GDN prefix in bounded chunks and materialize recurrent state after
/// each chunk. This is the correctness fallback for a sidecar-checkpoint miss.
/// It remains O(prefix), but it cannot hide an arbitrarily large lazy graph in
/// the first suffix chunk or exhaust the MLX graph allocator.
pub(crate) fn run_gdn_only_prefill_materialized(
    prefix_tokens: &[u32],
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    turn_cancel: Option<&AtomicBool>,
) -> Result<()> {
    let chunk_size = crate::models::forward::paged_prefill_chunk_rows();
    run_gdn_only_prefill_materialized_with_chunk_size(
        prefix_tokens,
        embed,
        layers,
        caches,
        chunk_size,
        turn_cancel,
    )
}

/// Chunk-size-parameterized worker for [`run_gdn_only_prefill_materialized`]
/// (mirrors the MoE split so tests can pin the replay behavior without
/// touching the process environment).
pub(crate) fn run_gdn_only_prefill_materialized_with_chunk_size(
    prefix_tokens: &[u32],
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    chunk_size: usize,
    turn_cancel: Option<&AtomicBool>,
) -> Result<()> {
    if chunk_size == 0 {
        return Err(Error::from_reason(
            "dense GDN materialized replay chunk size must be positive",
        ));
    }
    for (chunk_idx, chunk) in prefix_tokens.chunks(chunk_size).enumerate() {
        // Cooperative-cancel checkpoint (H1b) between replay chunks — an
        // O(prefix) cached-prefix replay must not hold the model thread
        // hostage after a cancel. A one-chunk replay stays single-shot by
        // design. The Err rides the callers' invalidate/abort arms, and
        // `replay_gdn_cache_and_commit` drops the staged cache, so the
        // partial replay is never published.
        if chunk_idx > 0 && turn_cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
            return Err(Error::from_reason("prefill cancelled"));
        }
        run_gdn_only_prefill(chunk, embed, layers, caches)?;
        materialize_linear_layer_caches(caches)?;
        crate::array::synchronize_and_clear_cache();
    }
    Ok(())
}

/// Splash `prefillPlan` refuses a chunk of 0 or more than 2048 rows; the
/// in-place prefill attention launches at most that many.
pub(crate) fn check_prefill_chunk_rows(chunk_size: i32) -> Result<()> {
    if chunk_size <= 0 || chunk_size as u32 > mlx_paged_attn::prefill_plan::PREFILL_MAX_ROWS {
        return Err(Error::from_reason(format!(
            "paged prefill chunk size {chunk_size} is outside 1..={}",
            mlx_paged_attn::prefill_plan::PREFILL_MAX_ROWS
        )));
    }
    Ok(())
}

/// Run a paged prefill over the suffix tokens. Returns the last position's
/// logits squeezed to `[vocab]` plus every GDN recurrent checkpoint the split
/// materialized.
///
/// `cached_prefix_len` is how many tokens the paged adapter has
/// already cached for this request (0 on a fresh prefill). The full
/// prompt is `tokens` (used for the GDN pass-1 prefill of the prefix);
/// the suffix `&tokens[cached_prefix_len..]` is what gets recorded
/// into the paged adapter and fed through the full forward pass.
///
/// `chunk_size` is 1..=2048 rows ([`check_prefill_chunk_rows`]); only the
/// uncached suffix is split. Each chunk writes its K/V into the paged adapter, attends
/// over the cumulative cached range, then clears MLX's transient graph before
/// the next chunk. This matches the Qwen3/Qwen3.5 MoE driver shape and keeps
/// dense Qwen from building one giant prefill graph for 30k+ suffixes.
///
/// The size is the caller's to choose — nothing here reads
/// `MLX_PAGED_PREFILL_CHUNK_SIZE`. Every production caller passes
/// `Qwen35Inner::cold_gdn_prefill_chunk_size()`, so a persist-cold turn also
/// splits at the checkpoint ladder's rungs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_prefill_chunk_with_size(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    target: &mut dyn PrefillTarget<DecoderLayer>,
    layer_kinds: &[Qwen3_5LayerKind],
    chunk_size: i32,
    cached_rope_deltas: i32,
    turn_cancel: Option<&AtomicBool>,
    taps: Option<&mut dyn PrefillTapSink>,
) -> Result<(MxArray, Vec<MaterializedGdnPrefixCheckpoint>)> {
    let (logits, checkpoints) = run_paged_prefill_unit(
        full_tokens,
        suffix_tokens,
        cached_prefix_len,
        gdn_prefix_already_primed,
        embed,
        target,
        layer_kinds,
        chunk_size,
        cached_rope_deltas,
        turn_cancel,
        taps,
        ChunkOutput::Read,
    )?;
    let logits = logits.ok_or_else(|| Error::from_reason("a read paged prefill gave no logits"))?;
    Ok((logits, checkpoints))
}

/// What a scheduler unit's last launch gives: the logits only when the unit
/// ends its prompt; the scheduler drops a shorter unit's.
pub(crate) fn unit_end_output(end: usize, prompt_len: usize) -> ChunkOutput {
    if end < prompt_len {
        ChunkOutput::Unread
    } else {
        ChunkOutput::Read
    }
}

/// [`run_paged_prefill_chunk_with_size`] whose last launch gives
/// `end_output`: no logits when it is [`ChunkOutput::Unread`] and the
/// suffix stops short of `full_tokens`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_prefill_unit(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    target: &mut dyn PrefillTarget<DecoderLayer>,
    layer_kinds: &[Qwen3_5LayerKind],
    chunk_size: i32,
    cached_rope_deltas: i32,
    turn_cancel: Option<&AtomicBool>,
    mut taps: Option<&mut dyn PrefillTapSink>,
    end_output: ChunkOutput,
) -> Result<(Option<MxArray>, Vec<MaterializedGdnPrefixCheckpoint>)> {
    let mut driver = PrefillDriver::new(
        full_tokens,
        suffix_tokens,
        cached_prefix_len,
        gdn_prefix_already_primed,
        target,
        chunk_size,
        cached_rope_deltas,
        taps.as_deref(),
        PassRates::default(),
        end_output,
    )?;
    match driver.run(
        embed,
        target,
        layer_kinds,
        turn_cancel,
        taps.as_mut()
            .map(|sink| &mut **sink as &mut dyn PrefillTapSink),
        None,
    )? {
        PrefillProgress::Done(logits, checkpoints) => Ok((logits, checkpoints)),
        PrefillProgress::Paused => Err(Error::from_reason(
            "a paged prefill without a pass budget paused",
        )),
    }
}

/// What one run of a [`PrefillDriver`] left.
pub(crate) enum PrefillProgress {
    /// Stopped inside the unit; the next run goes on where this one stopped.
    Paused,
    /// The unit ran to its end: its last row's logits (when its last launch
    /// is read) and the GDN checkpoints it materialized.
    Done(Option<MxArray>, Vec<MaterializedGdnPrefixCheckpoint>),
}

/// The measured milliseconds of one linear and one attention layer of a
/// launch, and the launch they were measured on (Splash
/// `Scheduler::passesFor`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PassRates {
    pub(crate) linear_ms: Option<f64>,
    pub(crate) attention_ms: Option<f64>,
    pub(crate) rows: u32,
    pub(crate) pairs: u64,
}

impl PassRates {
    /// The numbers a launch of `rows` rows from `start` begins with: these,
    /// scaled by rows (and attention pairs for the attention layer), up or
    /// down.
    pub(crate) fn for_launch(self, start: u32, rows: u32) -> Self {
        let pairs =
            crate::engine::scheduler::dense_attention_pairs(u64::from(start), u64::from(rows));
        let rows_scale = if self.rows > 0 {
            f64::from(rows) / f64::from(self.rows)
        } else {
            1.0
        };
        let pairs_scale = if self.pairs > 0 {
            pairs as f64 / self.pairs as f64
        } else {
            1.0
        };
        Self {
            linear_ms: self.linear_ms.map(|ms| ms * rows_scale),
            attention_ms: self.attention_ms.map(|ms| ms * pairs_scale),
            rows,
            pairs,
        }
    }

    fn estimate(&self, kind: Qwen3_5LayerKind) -> Option<f64> {
        match kind {
            Qwen3_5LayerKind::Linear => self.linear_ms,
            Qwen3_5LayerKind::FullAttentionPaged { .. } => self.attention_ms,
        }
    }

    /// A pass of `linear` and `attention` layers took `ms`.
    pub(crate) fn observe(&mut self, linear: usize, attention: usize, ms: f64) {
        if attention == 0 {
            if linear > 0 {
                self.linear_ms = Some(ms / linear as f64);
            }
            return;
        }
        let linear_part = self.linear_ms.unwrap_or(0.0) * linear as f64;
        self.attention_ms = Some((ms - linear_part).max(0.0) / attention as f64);
    }
}

/// The layers from `from` one pass runs: at least one; then while each next
/// layer's estimate is known and the sum stays within `budget_ms`. A layer
/// with no estimate runs alone, so it is measured.
pub(crate) fn pass_layers(
    kinds: &[Qwen3_5LayerKind],
    from: usize,
    budget_ms: f64,
    rates: &PassRates,
) -> usize {
    let rest = kinds.len().saturating_sub(from);
    let mut spent = 0.0;
    let mut layers = 0;
    for &kind in &kinds[from.min(kinds.len())..] {
        let Some(ms) = rates.estimate(kind) else {
            break;
        };
        if layers > 0 && spent + ms > budget_ms {
            break;
        }
        spent += ms;
        layers += 1;
        if spent >= budget_ms {
            break;
        }
    }
    layers.max(1).min(rest)
}

/// One launch of a prefill in flight: its rows, the hidden rows after its
/// last layer run, the taps so far, the next layer, and (when it may stop)
/// the GDN cells as they were at its start. A launch whose output no one
/// reads runs its last layer store-only; `tail_writes` then holds that
/// layer's pool writes until an eval names them, since waiting on `hidden`
/// (the layer's input) does not wait for them.
struct Launch {
    range: Range<usize>,
    start: u32,
    grant: Option<CaptureGrant>,
    output: ChunkOutput,
    hidden: MxArray,
    tail_writes: Vec<MxArray>,
    taps: Vec<Option<MxArray>>,
    next_layer: usize,
    cells: Option<Vec<Qwen3_5LayerCache>>,
    trace_start: Option<Instant>,
    info_start: Option<Instant>,
}

/// A paged prefill of one unit that can stop between layers (Splash
/// `PassCommand`): without a budget it runs the unit in one call; with one
/// it runs at most one pass of one launch and stops. Its launches, their
/// rows, state points and publishes are the same either way.
pub(crate) struct PrefillDriver {
    full_tokens: Vec<u32>,
    suffix: Vec<u32>,
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    cached_rope_deltas: i32,
    checkpoint_boundaries: Vec<u32>,
    capture_rows: Vec<u32>,
    tap_layers: Vec<usize>,
    single_shot: bool,
    end_output: ChunkOutput,
    ranges: Vec<Range<usize>>,
    next: usize,
    primed: bool,
    checkpoints: Vec<MaterializedGdnPrefixCheckpoint>,
    last_logits: Option<MxArray>,
    launch: Option<Launch>,
    rates: PassRates,
    trace_enabled: bool,
    info_enabled: bool,
    started: Option<Instant>,
}

impl PrefillDriver {
    /// The unit `suffix_tokens` of `full_tokens` from `cached_prefix_len`;
    /// `rates` are the last launch's measured layer times; `end_output` is
    /// whether the caller reads the unit's logits (a unit that ends the
    /// prompt gives them either way).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
        gdn_prefix_already_primed: bool,
        target: &mut dyn PrefillTarget<DecoderLayer>,
        chunk_size: i32,
        cached_rope_deltas: i32,
        taps: Option<&dyn PrefillTapSink>,
        rates: PassRates,
        end_output: ChunkOutput,
    ) -> Result<Self> {
        if suffix_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_prefill_chunk called with empty suffix",
            ));
        }
        crate::engine::progress::with_turn_progress(|p| {
            p.begin_prefill(suffix_tokens.len() as u32, cached_prefix_len)
        });
        check_prefill_chunk_rows(chunk_size)?;
        let checkpoint_boundaries = prefill_ladder(
            target.parts()?.adapter,
            full_tokens.len(),
            cached_prefix_len,
            taps.is_some(),
        );
        let capture_rows = capture_rows_of(taps);
        let breaks = prefill_breaks(
            &checkpoint_boundaries,
            &capture_rows,
            cached_prefix_len,
            full_tokens.len(),
        );
        let single_shot = breaks.is_empty() && suffix_tokens.len() <= chunk_size as usize;
        let ranges = if single_shot {
            std::iter::once(0..suffix_tokens.len()).collect()
        } else {
            paged_prefill_ranges(
                suffix_tokens.len(),
                chunk_size as usize,
                &checkpoint_suffix_offsets(&breaks, cached_prefix_len),
            )
        };
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let ends_prompt = cached_prefix_len as usize + suffix_tokens.len() >= full_tokens.len();
        Ok(Self {
            full_tokens: full_tokens.to_vec(),
            suffix: suffix_tokens.to_vec(),
            cached_prefix_len,
            gdn_prefix_already_primed,
            cached_rope_deltas,
            checkpoint_boundaries,
            capture_rows,
            tap_layers: taps.map_or_else(Vec::new, |sink| sink.tap_layers().to_vec()),
            single_shot,
            end_output: if ends_prompt {
                ChunkOutput::Read
            } else {
                end_output
            },
            ranges,
            next: 0,
            primed: false,
            checkpoints: Vec::new(),
            last_logits: None,
            launch: None,
            rates,
            trace_enabled: inference_trace_enabled(),
            info_enabled: inference_info_enabled,
            started: (single_shot && inference_info_enabled).then(Instant::now),
        })
    }

    /// The measured layer times, for the unit after this one.
    pub(crate) fn rates(&self) -> PassRates {
        self.rates
    }

    /// The rows of the launch in flight, booked but not all written.
    pub(crate) fn in_flight_rows(&self) -> Option<u32> {
        self.launch.as_ref().map(|launch| launch.range.len() as u32)
    }

    /// What the launch in flight holds between passes: its hidden rows, its
    /// taps and the launch-start cells of the layers it ran. A tap of the
    /// last layer run is the hidden rows, and a layer not run yet shares its
    /// cells with the sequence's: neither is counted twice.
    pub(crate) fn held_bytes(&self) -> u64 {
        self.launch.as_ref().map_or(0, |launch| {
            let taps = launch
                .taps
                .iter()
                .flatten()
                .filter(|tap| !std::sync::Arc::ptr_eq(&tap.handle, &launch.hidden.handle))
                .fold(0u64, |sum, tap| sum.saturating_add(tap.nbytes() as u64));
            let cells = launch.cells.as_deref().map_or(0, |cells| {
                Qwen3_5LayerCache::linear_state_bytes(&cells[..launch.next_layer.min(cells.len())])
            });
            (launch.hidden.nbytes() as u64)
                .saturating_add(taps)
                .saturating_add(cells)
        })
    }

    /// Takes the in-flight launch's GDN cells as they were at its start.
    pub(crate) fn take_launch_start_cells(&mut self) -> Option<Vec<Qwen3_5LayerCache>> {
        self.launch.as_mut().and_then(|launch| launch.cells.take())
    }

    /// Runs the unit to its end (`budget_ms` `None`), or one pass: layers of
    /// one launch whose estimates fit `budget_ms`, at least one; a launch's
    /// end work in the pass that runs its last layer, or alone when the
    /// launch ends at an armed capture. A pass never runs past a launch's
    /// end.
    pub(crate) fn run(
        &mut self,
        embed: &Embedding,
        target: &mut dyn PrefillTarget<DecoderLayer>,
        layer_kinds: &[Qwen3_5LayerKind],
        turn_cancel: Option<&AtomicBool>,
        mut taps: Option<&mut dyn PrefillTapSink>,
        budget_ms: Option<f64>,
    ) -> Result<PrefillProgress> {
        let pass_start = Instant::now();
        let mut measured = true;
        loop {
            if self.launch.is_none() {
                if !self.primed {
                    measured &= !self.prime(embed, target)?;
                }
                self.begin_launch(
                    embed,
                    target,
                    taps.as_deref_mut(),
                    turn_cancel,
                    budget_ms.is_some(),
                )?;
            }
            let launch = self
                .launch
                .as_mut()
                .ok_or_else(|| Error::from_reason("a paged prefill lost its launch"))?;
            let from = launch.next_layer;
            let to = match budget_ms {
                _ if from >= layer_kinds.len() => layer_kinds.len(),
                None => layer_kinds.len(),
                Some(ms) => from + pass_layers(layer_kinds, from, ms, &self.rates),
            };
            {
                let parts = target.parts()?;
                run_paged_prefill_layers(
                    launch,
                    from..to,
                    parts.layers,
                    parts.caches,
                    layer_kinds,
                    parts.adapter,
                    self.cached_rope_deltas,
                    &self.tap_layers,
                )?;
            }
            let ran = &layer_kinds[from..to];
            let pause = budget_ms.is_some()
                && to > from
                && (to < layer_kinds.len()
                    || launch.grant.as_ref().is_some_and(CaptureGrant::armed));
            if pause {
                let evaluated = {
                    let mut arrays: Vec<&MxArray> = vec![&launch.hidden];
                    arrays.extend(&launch.tail_writes);
                    arrays.extend(launch.taps.iter().flatten());
                    MxArray::eval_arrays_with_context(&arrays, "paged_prefill_pass")
                };
                let stored = usize::from(!launch.tail_writes.is_empty());
                // Holding the pool arrays across the decode steps between
                // passes (or until a failed run ends) would make their next
                // writes copy the whole pool.
                launch.tail_writes.clear();
                evaluated?;
                if measured {
                    let attention = ran
                        .iter()
                        .filter(|kind| !matches!(kind, Qwen3_5LayerKind::Linear))
                        .count();
                    self.rates.observe(
                        ran.len() - attention,
                        attention - stored,
                        pass_start.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                note_prefill_pass(ran.len());
                return Ok(PrefillProgress::Paused);
            }
            if self.end_launch(embed, target, layer_kinds, taps.as_deref_mut())? {
                let checkpoints = std::mem::take(&mut self.checkpoints);
                if budget_ms.is_some() {
                    note_prefill_pass(ran.len());
                }
                return Ok(PrefillProgress::Done(self.last_logits.take(), checkpoints));
            }
            if budget_ms.is_some() {
                note_prefill_pass(ran.len());
                return Ok(PrefillProgress::Paused);
            }
        }
    }

    /// Pass 1 over the cached prefix (once, before the first launch), then
    /// the adapter's auxiliary-state acknowledgement. Whether it replayed.
    fn prime(
        &mut self,
        embed: &Embedding,
        target: &mut dyn PrefillTarget<DecoderLayer>,
    ) -> Result<bool> {
        self.primed = true;
        let cached_prefix_len = self.cached_prefix_len;
        let replay = cached_prefix_len > 0 && !self.gdn_prefix_already_primed;
        if replay {
            let gdn_trace_start = (self.trace_enabled && !self.single_shot).then(Instant::now);
            let gdn_info_start = self.info_enabled.then(Instant::now);
            let prefix = &self.full_tokens[..(cached_prefix_len as usize)];
            let parts = target.parts()?;
            run_gdn_only_prefill(prefix, embed, parts.layers, parts.caches)?;
            if let Some(start) = gdn_trace_start {
                let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense paged_prefill_gdn_prefix_done \
                     prefix_tokens={} elapsed_ms={:.1} active_mib={:.1} cache_mib={:.1} peak_mib={:.1}",
                    cached_prefix_len,
                    elapsed_ms(start),
                    active_mib,
                    cache_mib,
                    peak_mib
                ));
            }
            if let Some(start) = gdn_info_start {
                let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "paged_prefill_gdn_prefix_graph_built",
                    prefix_tokens = cached_prefix_len,
                    graph_build_ms = elapsed_ms(start),
                    materialized = false,
                    active_mib,
                    cache_mib,
                    peak_mib,
                    "paged prefill GDN prefix graph built"
                );
            }
        }
        // The recurrent half of the cached prefix is now in hand (either it
        // already was, or the pass above just replayed it from token ids).
        // Clear the adapter's auxiliary-state obligation before the first
        // `record_tokens`.
        target
            .parts()?
            .adapter
            .confirm_aux_prefix_primed(cached_prefix_len)
            .map_err(Error::from_reason)?;
        Ok(replay)
    }

    /// The next launch's cancel check, booking and embedding.
    fn begin_launch(
        &mut self,
        embed: &Embedding,
        target: &mut dyn PrefillTarget<DecoderLayer>,
        taps: Option<&mut (dyn PrefillTapSink + '_)>,
        turn_cancel: Option<&AtomicBool>,
        stops: bool,
    ) -> Result<()> {
        let range = self
            .ranges
            .get(self.next)
            .cloned()
            .ok_or_else(|| Error::from_reason("a paged prefill ran past its last launch"))?;
        let start = self.cached_prefix_len + range.start as u32;
        let chunk = &self.suffix[range.clone()];
        let grant = if self.single_shot {
            target.record(chunk)?;
            None
        } else {
            // Cooperative cancel at the chunk boundary. The Err rides the
            // caller's release arm (`abort_paged_turn` /
            // `invalidate_dense_paged_session`), which releases the live
            // request; only the blocks a tapped prefill's completed chunks
            // registered stay cached (Splash `publishCommittedBlocks`).
            if turn_cancel.is_some_and(|f| f.load(Ordering::Relaxed)) || cancel_hook(start) {
                return Err(Error::from_reason("prefill cancelled"));
            }
            let chunk_end = start + chunk.len() as u32;
            book_chunk(
                target,
                taps,
                &self.capture_rows,
                chunk,
                chunk_end,
                chunk_end as usize >= self.full_tokens.len(),
            )?
        };
        let cells = if stops {
            self.rates = self.rates.for_launch(start, chunk.len() as u32);
            Some(clone_linear_cells(target.parts()?.caches))
        } else {
            None
        };
        #[cfg(test)]
        {
            note_prefill_chunk_blocks(target.parts()?.adapter);
            if !self.single_shot {
                note_timeline("booked", start);
            }
        }
        let output = if self.next + 1 == self.ranges.len() {
            self.end_output
        } else {
            ChunkOutput::Unread
        };
        #[cfg(test)]
        let embed_scope = crate::array::host_waits::scope("prefill_chunk_graph");
        let chunk_len = chunk.len() as i64;
        let input_ids = MxArray::from_uint32(chunk, &[1, chunk_len])?;
        let hidden = embed.forward(&input_ids)?;
        #[cfg(test)]
        drop(embed_scope);
        self.launch = Some(Launch {
            range,
            start,
            grant,
            output,
            hidden,
            tail_writes: Vec::new(),
            taps: vec![None; self.tap_layers.len()],
            next_layer: 0,
            cells,
            trace_start: self.trace_enabled.then(Instant::now),
            info_start: self.info_enabled.then(Instant::now),
        });
        Ok(())
    }

    /// The end work of the launch whose layers all ran: its taps into the
    /// sink, then (last launch) the logits when read, or (any other) the
    /// eval, the GDN snapshot at a checkpoint, the capture, the cache clear
    /// and what it reached. Whether the unit is done.
    fn end_launch(
        &mut self,
        embed: &Embedding,
        target: &mut dyn PrefillTarget<DecoderLayer>,
        _layer_kinds: &[Qwen3_5LayerKind],
        mut taps: Option<&mut (dyn PrefillTapSink + '_)>,
    ) -> Result<bool> {
        let launch = self
            .launch
            .take()
            .ok_or_else(|| Error::from_reason("a paged prefill ended no launch"))?;
        #[cfg(test)]
        if !self.single_shot {
            note_timeline("built", launch.start);
        }
        self.next += 1;
        let chunk_taps = collect_taps(launch.taps, &self.tap_layers)?;
        let chunk = &self.suffix[launch.range.clone()];
        if let Some(sink) = taps.as_deref_mut() {
            sink.chunk(launch.start as usize, chunk, &chunk_taps)?;
        }
        let hidden_states = launch.hidden;
        let parts = target.parts()?;
        if self.single_shot {
            self.last_logits = end_unit(
                launch.output,
                &hidden_states,
                &launch.tail_writes,
                &parts,
                embed,
                _layer_kinds,
            )?;
            if let Some(start) = self.started {
                let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "paged_prefill_single_shot_graph_built",
                    suffix_tokens = self.suffix.len(),
                    cached_prefix_tokens = self.cached_prefix_len,
                    graph_build_ms = elapsed_ms(start),
                    materialized = false,
                    active_mib,
                    cache_mib,
                    peak_mib,
                    "single-shot paged prefill graph built"
                );
            }
            return Ok(true);
        }
        let chunk_idx = self.next - 1;
        let total_chunks = self.ranges.len();
        let chunk_start_position = launch.start;
        let context_after = chunk_start_position + chunk.len() as u32;
        let capture_checkpoint = self.checkpoint_boundaries.contains(&context_after);
        let grant = launch.grant;
        if self.next == total_chunks {
            self.last_logits = end_unit(
                launch.output,
                &hidden_states,
                &launch.tail_writes,
                &parts,
                embed,
                _layer_kinds,
            )?;
            if capture_checkpoint {
                materialize_linear_layer_caches(parts.caches)?;
                if let Some(caches) = snapshot_materialized_linear_layer_caches(parts.caches) {
                    self.checkpoints.push(MaterializedGdnPrefixCheckpoint {
                        prefix_len: context_after,
                        caches,
                    });
                }
            }
            if grant.is_some() {
                if grant.as_ref().is_some_and(CaptureGrant::armed)
                    && let Some(sink) = taps.as_deref_mut()
                {
                    capture_chunk_linear(parts.caches, sink, context_after, capture_checkpoint)?;
                }
                reach_chunk(target, taps.as_deref_mut(), context_after, grant)?;
            }
            if let Some(start) = launch.trace_start {
                let chunk_elapsed_ms = elapsed_ms(start);
                let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense paged_prefill_chunk_final_graph_built \
                     chunk_index={} total_chunks={} chunk_tokens={} context_before={} context_after={} \
                     elapsed_ms={:.1} active_mib={:.1} cache_mib={:.1} peak_mib={:.1}",
                    chunk_idx + 1,
                    total_chunks,
                    chunk.len(),
                    chunk_start_position,
                    context_after,
                    chunk_elapsed_ms,
                    active_mib,
                    cache_mib,
                    peak_mib
                ));
            }
            if let Some(start) = launch.info_start {
                let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "paged_prefill_chunk_done",
                    chunk_index = chunk_idx + 1,
                    total_chunks,
                    chunk_tokens = chunk.len(),
                    context_before = chunk_start_position,
                    context_after,
                    materialized = false,
                    gdn_checkpoint_materialized = capture_checkpoint,
                    graph_build_ms = elapsed_ms(start),
                    active_mib,
                    cache_mib,
                    peak_mib,
                    "final paged prefill chunk graph built"
                );
            }
            return Ok(true);
        }
        {
            let mut roots: Vec<&MxArray> = vec![&hidden_states];
            roots.extend(&launch.tail_writes);
            let merged = !serial_order();
            if merged && let Some(sink) = taps.as_deref() {
                sink.collect_arrays(&mut roots);
            }
            MxArray::eval_arrays_with_context(&roots, "paged_prefill_chunk")?;
            #[cfg(test)]
            note_tail_writes(parts.adapter, _layer_kinds);
            if !merged && let Some(sink) = taps.as_deref() {
                #[cfg(test)]
                PREFILL_SINK_EVALS.with(|count| count.set(count.get() + 1));
                sink.eval()?;
            }
        }
        #[cfg(test)]
        if let Some(sink) = taps.as_deref() {
            let mut arrays = Vec::new();
            sink.collect_arrays(&mut arrays);
            let ready = available(arrays.iter().copied());
            PREFILL_SINK_READY.with(|log| log.borrow_mut().push(ready));
        }
        if capture_checkpoint {
            materialize_linear_layer_caches(parts.caches)?;
            if let Some(caches) = snapshot_materialized_linear_layer_caches(parts.caches) {
                self.checkpoints.push(MaterializedGdnPrefixCheckpoint {
                    prefix_len: context_after,
                    caches,
                });
            }
        }
        if grant.as_ref().is_some_and(CaptureGrant::armed)
            && let Some(sink) = taps.as_deref_mut()
        {
            capture_chunk_linear(parts.caches, sink, context_after, capture_checkpoint)?;
        }
        crate::array::synchronize_and_clear_cache();
        #[cfg(test)]
        note_timeline("settled", context_after);
        reach_chunk(target, taps, context_after, grant)?;
        let cached_prefix_len = self.cached_prefix_len;
        crate::engine::progress::with_turn_progress(|p| {
            p.prefill_advanced(context_after - cached_prefix_len)
        });
        if let Some(start) = launch.trace_start {
            let chunk_elapsed_ms = elapsed_ms(start);
            let chunk_tok_s = if chunk_elapsed_ms > 0.0 {
                chunk.len() as f64 / (chunk_elapsed_ms / 1000.0)
            } else {
                0.0
            };
            let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense paged_prefill_chunk_done \
                 chunk_index={} total_chunks={} chunk_tokens={} context_before={} context_after={} \
                 elapsed_ms={:.1} tok_s={:.2} active_mib={:.1} cache_mib={:.1} peak_mib={:.1}",
                chunk_idx + 1,
                total_chunks,
                chunk.len(),
                chunk_start_position,
                context_after,
                chunk_elapsed_ms,
                chunk_tok_s,
                active_mib,
                cache_mib,
                peak_mib
            ));
        }
        if let Some(start) = launch.info_start {
            let chunk_elapsed_ms = elapsed_ms(start);
            let chunk_tok_s = if chunk_elapsed_ms > 0.0 {
                chunk.len() as f64 / (chunk_elapsed_ms / 1000.0)
            } else {
                0.0
            };
            let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
            tracing::info!(
                target: "mlx_core::inference",
                event = "paged_prefill_chunk_done",
                chunk_index = chunk_idx + 1,
                total_chunks,
                chunk_tokens = chunk.len(),
                context_before = chunk_start_position,
                context_after,
                materialized = true,
                elapsed_ms = chunk_elapsed_ms,
                tok_s = chunk_tok_s,
                active_mib,
                cache_mib,
                peak_mib,
                "paged prefill chunk completed"
            );
        }
        Ok(false)
    }

    /// The rows of the launch done so far: the launches ended, plus the
    /// fraction of the one in flight whose layers ran.
    pub(crate) fn rows_done(&self, layers: usize) -> u32 {
        let ended = self.ranges[..self.next]
            .iter()
            .map(Range::len)
            .sum::<usize>() as u32;
        let partial = self.launch.as_ref().map_or(0, |launch| {
            (launch.range.len() * launch.next_layer / layers.max(1)) as u32
        });
        ended + partial
    }
}

/// The GDN cells as they are, sharing their arrays; attention slots are
/// placeholders (their K/V lives in the paged pool).
pub(crate) fn clone_linear_cells(caches: &[Qwen3_5LayerCache]) -> Vec<Qwen3_5LayerCache> {
    caches
        .iter()
        .map(|cache| match cache {
            Qwen3_5LayerCache::Linear(arrays) => Qwen3_5LayerCache::Linear(arrays.clone()),
            Qwen3_5LayerCache::FullAttention(_) => Qwen3_5LayerCache::new_full_attention(),
        })
        .collect()
}

#[cfg(test)]
thread_local! {
    /// The layers of each pass this thread ran, in order.
    pub(crate) static PREFILL_PASSES: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
    /// Stands in for a scheduled pass's budget on this thread.
    pub(crate) static PREFILL_PASS_BUDGET_OVERRIDE: std::cell::Cell<Option<f64>> =
        const { std::cell::Cell::new(None) };
}

fn note_prefill_pass(_layers: usize) {
    #[cfg(test)]
    PREFILL_PASSES.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push(_layers);
        }
    });
}

/// Layers `layers` of a launch's forward over its hidden rows, each tapped
/// layer's output kept.
#[allow(clippy::too_many_arguments)]
fn run_paged_prefill_layers(
    launch: &mut Launch,
    layers_to_run: Range<usize>,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    cached_rope_deltas: i32,
    tap_layers: &[usize],
) -> Result<()> {
    debug_assert_eq!(layers.len(), caches.len());
    debug_assert_eq!(layers.len(), layer_kinds.len());
    let rope_position_offset = paged_rope_offset(launch.start, cached_rope_deltas);
    let mut mrope_cache: Option<(MxArray, MxArray)> = None;
    for layer_idx in layers_to_run {
        #[cfg(test)]
        let build_scope = crate::array::host_waits::scope("prefill_chunk_graph");
        let kind = layer_kinds[layer_idx];
        if let Some(paged_idx) = store_only_tail(
            launch.output,
            layer_idx,
            layers.len(),
            kind,
            tap_layers,
            false,
        ) {
            layers[layer_idx].store_paged_kv(
                &launch.hidden,
                paged_idx,
                paged_adapter,
                launch.start,
                rope_position_offset,
            )?;
            launch.tail_writes = paged_adapter.pending_pool_write_arrays(paged_idx);
            #[cfg(test)]
            PREFILL_TAILS_STORED.with(|count| count.set(count.get() + 1));
            launch.next_layer = layer_idx + 1;
            continue;
        }
        launch.hidden = if separate_ops() {
            layers[layer_idx].forward_paged_or_flat(
                &launch.hidden,
                kind,
                paged_adapter,
                launch.start,
                PagedPass::Prefill,
                /* mask */ None,
                Some(&mut caches[layer_idx]),
                /* position_ids */ None,
                /* use_kernel */ true,
                rope_position_offset,
                &mut mrope_cache,
            )?
        } else {
            layers[layer_idx].forward_paged_prefill(
                &launch.hidden,
                kind,
                paged_adapter,
                launch.start,
                Some(&mut caches[layer_idx]),
                /* position_ids */ None,
                rope_position_offset,
                &mut mrope_cache,
            )?
        };
        for (slot, &tap_layer) in tap_layers.iter().enumerate() {
            if tap_layer == layer_idx {
                launch.taps[slot] = Some(launch.hidden.clone());
            }
        }
        #[cfg(test)]
        {
            drop(build_scope);
            note_timeline("layer", launch.start + layer_idx as u32);
        }
        crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &launch.hidden)?;
        launch.next_layer = layer_idx + 1;
    }
    Ok(())
}

/// Image-bearing paged prefill with optional cached-prefix reuse.
///
/// Feeds the vision encoder's image-merged token embeddings
/// (`merge.inputs_embeds`) through the paged adapter and applies 3-row M-RoPE
/// over `merge.position_ids` on the full-attention layers, while GDN/linear
/// layers run with neither mask nor positions.
///
/// `expanded_tokens` are the placeholder-expanded prompt tokens (one entry per
/// embedding row). They drive `record_tokens` / the physical slot cursor only;
/// the forward itself consumes the merged embeddings, not re-embedded ids.
///
/// Only `expanded_tokens[cached_prefix_len..]` are recorded and forwarded. The
/// corresponding slices of `merge.inputs_embeds` and `merge.position_ids` are
/// forwarded at their absolute physical positions, so full-attention layers
/// read the cached image-aware K/V prefix and rotate the suffix with the same
/// M-RoPE coordinates as a cold prefill.
///
/// A non-zero cached prefix is accepted only with its matching materialized
/// GDN sidecar. The caller must downgrade a K/V-only candidate to a cold
/// position-zero prefill; replaying only the recurrent layers would omit the
/// intervening attention/MLP residual stream and create a state that never
/// existed in the original image prefill.
///
/// With `taps` the chunks book through `target` and reach the turn's state
/// points as a text prefill's do (Splash `publishReachedStateBoundaries`
/// runs for an image request too).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_vlm_prefill(
    expanded_tokens: &[u32],
    merge: &VisionMerge,
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    target: &mut dyn PrefillTarget<DecoderLayer>,
    layer_kinds: &[Qwen3_5LayerKind],
    turn_cancel: Option<&AtomicBool>,
    mut taps: Option<&mut dyn PrefillTapSink>,
) -> Result<(MxArray, Vec<MaterializedGdnPrefixCheckpoint>)> {
    if expanded_tokens.is_empty() {
        return Err(Error::from_reason(
            "run_paged_vlm_prefill called with empty prompt",
        ));
    }
    let prompt_len = expanded_tokens.len();
    let prompt_len_u32 = u32::try_from(prompt_len)
        .map_err(|_| Error::from_reason("VLM prompt length exceeds u32"))?;
    let cached_prefix_len_us = usize::try_from(cached_prefix_len)
        .map_err(|_| Error::from_reason("VLM cached prefix length does not fit usize"))?;
    if cached_prefix_len >= prompt_len_u32 {
        return Err(Error::from_reason(format!(
            "run_paged_vlm_prefill requires a non-empty suffix: cached_prefix_len={} prompt_len={}",
            cached_prefix_len, prompt_len
        )));
    }
    let embed_len = merge.inputs_embeds.shape_at(1)?;
    let position_len = merge.position_ids.shape_at(2)?;
    if embed_len != prompt_len as i64 || position_len != prompt_len as i64 {
        return Err(Error::from_reason(format!(
            "run_paged_vlm_prefill merge length mismatch: tokens={} inputs_embeds={} position_ids={}",
            prompt_len, embed_len, position_len
        )));
    }
    let adapter_prefix_len = target.parts()?.adapter.current_token_count();
    if adapter_prefix_len != cached_prefix_len {
        return Err(Error::from_reason(format!(
            "run_paged_vlm_prefill adapter prefix mismatch: adapter={} requested={}",
            adapter_prefix_len, cached_prefix_len
        )));
    }

    if cached_prefix_len > 0 && !gdn_prefix_already_primed {
        return Err(Error::from_reason(
            "run_paged_vlm_prefill received a K/V prefix without an exact GDN sidecar; caller must restart cold",
        ));
    }
    // Past the guard the recurrent half is exact by construction (an image
    // prefix is never replayed from token ids — the caller restarts cold
    // instead), so the adapter's obligation is discharged here.
    let block_size = {
        let parts = target.parts()?;
        parts
            .adapter
            .confirm_aux_prefix_primed(cached_prefix_len)
            .map_err(Error::from_reason)?;
        parts.adapter.block_size()
    };

    let suffix_tokens = &expanded_tokens[cached_prefix_len_us..];
    crate::engine::progress::with_turn_progress(|p| {
        p.begin_prefill(suffix_tokens.len() as u32, cached_prefix_len)
    });
    // Image-aware KV reuse also needs an exact recurrent sidecar: split once at
    // the reusable block boundary so a later non-live image-prefix hit remains
    // exact. A media turn keeps the single deepest boundary rather than a
    // ladder: the extra rungs would only cost memory.
    let checkpoint_boundaries: Vec<u32> =
        gdn_checkpoint_target(prompt_len, cached_prefix_len, block_size)
            .into_iter()
            .collect();
    let capture_rows = capture_rows_of(taps.as_deref());
    let breaks = prefill_breaks(
        &checkpoint_boundaries,
        &capture_rows,
        cached_prefix_len,
        prompt_len,
    );
    let chunk_ranges = media_prefill_ranges(suffix_tokens.len(), &breaks, cached_prefix_len);
    let total_chunks = chunk_ranges.len();
    let mut last_logits = None;
    let mut checkpoints = Vec::new();
    let tap_layers = taps
        .as_deref()
        .map_or_else(Vec::new, |sink| sink.tap_layers().to_vec());

    for (chunk_idx, range) in chunk_ranges.into_iter().enumerate() {
        // Cooperative-cancel checkpoint (H1b): abort at the chunk boundary.
        // The Err rides the VLM cores' `invalidate_dense_paged_session` arm —
        // the request is released, never finalized.
        if turn_cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
            return Err(Error::from_reason("prefill cancelled"));
        }
        let absolute_start = cached_prefix_len_us + range.start;
        let absolute_end = cached_prefix_len_us + range.end;
        let chunk_tokens = &expanded_tokens[absolute_start..absolute_end];
        let chunk_embeds =
            merge
                .inputs_embeds
                .slice_axis(1, absolute_start as i64, absolute_end as i64)?;
        let chunk_positions =
            merge
                .position_ids
                .slice_axis(2, absolute_start as i64, absolute_end as i64)?;

        let context_after = absolute_end as u32;
        let is_last_chunk = chunk_idx + 1 == total_chunks;
        let grant = book_chunk(
            target,
            taps.as_deref_mut(),
            &capture_rows,
            chunk_tokens,
            context_after,
            absolute_end >= prompt_len,
        )?;
        let parts = target.parts()?;
        let prefilled = run_paged_prefill_one_chunk_tapped(
            chunk_tokens,
            absolute_start as u32,
            embed,
            parts.layers,
            parts.caches,
            layer_kinds,
            parts.adapter,
            Some(&chunk_embeds),
            Some(&chunk_positions),
            // The explicit M-RoPE grid is authoritative for image prefill.
            0,
            &tap_layers,
            if is_last_chunk {
                ChunkOutput::Read
            } else {
                ChunkOutput::Unread
            },
        )?;
        if let Some(sink) = taps.as_deref_mut() {
            sink.chunk(absolute_start, chunk_tokens, &prefilled.taps)?;
        }

        let capture_checkpoint = checkpoint_boundaries.contains(&context_after);
        if is_last_chunk {
            last_logits = Some(project_last_token_logits(
                prefilled.hidden()?,
                parts.final_norm,
                parts.lm_head,
                embed,
            )?);
        } else {
            prefilled.eval_roots("paged_prefill_chunk")?;
            if let Some(sink) = taps.as_deref() {
                sink.eval()?;
            }
        }
        if capture_checkpoint {
            materialize_linear_layer_caches(parts.caches)?;
            if let Some(caches) = snapshot_materialized_linear_layer_caches(parts.caches) {
                checkpoints.push(MaterializedGdnPrefixCheckpoint {
                    prefix_len: context_after,
                    caches,
                });
            }
        }
        if grant.as_ref().is_some_and(CaptureGrant::armed)
            && let Some(sink) = taps.as_deref_mut()
        {
            capture_chunk_linear(parts.caches, sink, context_after, capture_checkpoint)?;
        }
        if !is_last_chunk {
            crate::array::synchronize_and_clear_cache();
        }
        if !is_last_chunk || grant.is_some() {
            reach_chunk(target, taps.as_deref_mut(), context_after, grant)?;
        }
        if !is_last_chunk {
            crate::engine::progress::with_turn_progress(|p| {
                p.prefill_advanced(context_after - cached_prefix_len)
            });
        }
    }

    let logits = last_logits
        .ok_or_else(|| Error::from_reason("run_paged_vlm_prefill produced no final chunk"))?;
    Ok((logits, checkpoints))
}

/// Paged prefill variant that ALSO returns the post-`final_norm` hidden
/// state for every prompt token, concatenated along the time axis to
/// `[1, prompt_len, hidden]`.
///
/// Mirror of `chunked_prefill_with_hidden` (dense / flat path). The
/// paged-MTP gate inside `paged_turn_sync_core_inner` consumes this so
/// `begin_mtp_decode`'s prompt-prefix seed can commit the full prompt
/// (advancing the stepper's `committed_len` to N) before the
/// first MTP cycle — without it the MTP draft attends over a
/// prompt-less context and parity vs the AR run breaks.
///
/// Caller MUST gate on `cached_prefix_len == 0` (the dense gate uses
/// the same `want_prompt_hidden` predicate). On a cache-reuse turn the
/// prefill only processes the suffix, so the captured hidden would not
/// cover the full prompt and the prompt-prefix seed cannot use it.
///
/// `chunk_size` carries the same contract as the non-hidden worker: 1..=2048
/// rows, and the value comes from the caller, never from `MLX_PAGED_PREFILL_CHUNK_SIZE` read here. An MTP turn is the only
/// consumer, and it is a persist-cold turn like any other — reading the raw env
/// default here is what kept MTP turns from publishing any ladder rung at all.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_prefill_chunk_with_hidden_with_size(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    target: &mut dyn PrefillTarget<DecoderLayer>,
    layer_kinds: &[Qwen3_5LayerKind],
    chunk_size: i32,
    keep_last_hidden: Option<usize>,
    cached_rope_deltas: i32,
    turn_cancel: Option<&AtomicBool>,
    mut taps: Option<&mut dyn PrefillTapSink>,
) -> Result<(MxArray, MxArray, Vec<MaterializedGdnPrefixCheckpoint>)> {
    if suffix_tokens.is_empty() {
        return Err(Error::from_reason(
            "run_paged_prefill_chunk_with_hidden called with empty suffix",
        ));
    }
    crate::engine::progress::with_turn_progress(|p| {
        p.begin_prefill(suffix_tokens.len() as u32, cached_prefix_len)
    });

    check_prefill_chunk_rows(chunk_size)?;

    let checkpoint_boundaries = prefill_ladder(
        target.parts()?.adapter,
        full_tokens.len(),
        cached_prefix_len,
        taps.is_some(),
    );
    let capture_rows = capture_rows_of(taps.as_deref());
    let breaks = prefill_breaks(
        &checkpoint_boundaries,
        &capture_rows,
        cached_prefix_len,
        full_tokens.len(),
    );

    if breaks.is_empty() && suffix_tokens.len() <= chunk_size as usize {
        return run_paged_prefill_single_shot_with_hidden(
            full_tokens,
            suffix_tokens,
            cached_prefix_len,
            gdn_prefix_already_primed,
            embed,
            target,
            layer_kinds,
            keep_last_hidden,
            cached_rope_deltas,
            taps.as_mut()
                .map(|sink| &mut **sink as &mut dyn PrefillTapSink),
        )
        .map(|(logits, hidden)| (logits, hidden, Vec::new()));
    }

    let inference_info_enabled =
        tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
    let prefill_trace_start = inference_info_enabled.then(Instant::now);
    let chunk_size_usize = chunk_size as usize;
    let suffix_offsets = checkpoint_suffix_offsets(&breaks, cached_prefix_len);
    let chunk_ranges = paged_prefill_ranges(suffix_tokens.len(), chunk_size_usize, &suffix_offsets);

    if cached_prefix_len > 0 && !gdn_prefix_already_primed {
        let gdn_trace_start = inference_info_enabled.then(Instant::now);
        let prefix = &full_tokens[..(cached_prefix_len as usize)];
        let parts = target.parts()?;
        run_gdn_only_prefill(prefix, embed, parts.layers, parts.caches)?;
        if let Some(start) = gdn_trace_start {
            let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
            tracing::info!(
                target: "mlx_core::inference",
                event = "paged_mtp_gdn_prefix_graph_built",
                prefix_tokens = cached_prefix_len,
                graph_build_ms = elapsed_ms(start),
                materialized = false,
                active_mib,
                cache_mib,
                peak_mib,
                "paged MTP GDN prefix graph built"
            );
        }
    }
    // See the non-MTP chunked driver: discharge the auxiliary-state obligation
    // before the first `record_tokens` of the turn.
    target
        .parts()?
        .adapter
        .confirm_aux_prefix_primed(cached_prefix_len)
        .map_err(Error::from_reason)?;

    let total_chunks = chunk_ranges.len();
    let mut last_logits: Option<MxArray> = None;
    let mut checkpoints = Vec::new();
    let mut hidden_chunks: Vec<MxArray> = Vec::with_capacity(total_chunks);
    let total_suffix_len = suffix_tokens.len();
    let keep_start = keep_last_hidden
        .map(|keep| total_suffix_len.saturating_sub(keep.max(1)))
        .unwrap_or(0);
    let mut chunk_start_position = cached_prefix_len;
    let mut suffix_offset = 0usize;
    let tap_layers = taps
        .as_deref()
        .map_or_else(Vec::new, |sink| sink.tap_layers().to_vec());

    for (chunk_idx, range) in chunk_ranges.into_iter().enumerate() {
        // Cooperative-cancel checkpoint (H1b): abort at the chunk boundary.
        // The Err rides the caller's `invalidate_dense_paged_session` /
        // `abort_paged_turn` arm — the request is released, never finalized;
        // only the blocks a tapped prefill's completed chunks registered stay
        // cached (Splash `publishCommittedBlocks`).
        if turn_cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
            return Err(Error::from_reason("prefill cancelled"));
        }
        let chunk = &suffix_tokens[range];
        let chunk_trace_start = inference_info_enabled.then(Instant::now);
        let is_last_chunk = chunk_idx + 1 == total_chunks;
        let chunk_start = suffix_offset;
        let chunk_end = chunk_start + chunk.len();
        let overlaps_kept_tail = chunk_end > keep_start;
        let context_after = chunk_start_position + chunk.len() as u32;

        let grant = book_chunk(
            target,
            taps.as_deref_mut(),
            &capture_rows,
            chunk,
            context_after,
            context_after as usize >= full_tokens.len(),
        )?;
        let parts = target.parts()?;
        #[cfg(test)]
        note_prefill_chunk_blocks(parts.adapter);

        let prefilled = run_paged_prefill_one_chunk_tapped(
            chunk,
            chunk_start_position,
            embed,
            parts.layers,
            parts.caches,
            layer_kinds,
            parts.adapter,
            /* inputs_embeds */ None,
            /* position_ids */ None,
            cached_rope_deltas,
            &tap_layers,
            if overlaps_kept_tail || is_last_chunk {
                ChunkOutput::Read
            } else {
                ChunkOutput::Unread
            },
        )?;
        if let Some(sink) = taps.as_deref_mut() {
            sink.chunk(chunk_start_position as usize, chunk, &prefilled.taps)?;
        }
        if prefilled.stored_only() {
            prefilled.eval_roots("paged_prefill_chunk")?;
            #[cfg(test)]
            note_tail_writes(parts.adapter, layer_kinds);
        }

        let chunk_hidden = if overlaps_kept_tail || is_last_chunk {
            // Slice before normalization so a one-row seed cannot retain the
            // full normalized prompt allocation through an MLX view.
            let keep_from = keep_start.max(chunk_start) - chunk_start;
            let kept = prefilled
                .hidden()?
                .slice_axis(1, keep_from as i64, chunk.len() as i64)?;
            Some(parts.final_norm.forward(&kept)?)
        } else {
            None
        };
        let capture_checkpoint = checkpoint_boundaries.contains(&context_after);

        if is_last_chunk {
            // Reuse the already-normed last chunk to project last-token
            // logits — `forward` on a final_norm output is idempotent
            // would be wasteful; slice directly instead.
            let chunk_hidden = chunk_hidden.as_ref().ok_or_else(|| {
                Error::from_reason("run_paged_prefill_chunk_with_hidden: missing last hidden")
            })?;
            let chunk_len = chunk_hidden.shape_at(1)?;
            let last_hidden = chunk_hidden.slice_axis(1, chunk_len - 1, chunk_len)?;
            let logits = if let Some(head) = parts.lm_head {
                head.forward(&last_hidden)?
            } else {
                embed.as_linear(&last_hidden)?
            };
            last_logits = Some(logits.squeeze(Some(&[0, 1]))?);
        }

        if let Some(chunk_hidden) = chunk_hidden
            && overlaps_kept_tail
        {
            let kept_hidden = chunk_hidden;
            // Materialize hidden BEFORE clear_cache; the hidden is a lazy
            // handle into graph nodes that the per-layer cache eviction
            // would otherwise free between chunks.
            kept_hidden.eval();
            hidden_chunks.push(kept_hidden);
        }
        if capture_checkpoint {
            materialize_linear_layer_caches(parts.caches)?;
            if let Some(caches) = snapshot_materialized_linear_layer_caches(parts.caches) {
                checkpoints.push(MaterializedGdnPrefixCheckpoint {
                    prefix_len: context_after,
                    caches,
                });
            }
        }
        if grant.as_ref().is_some_and(CaptureGrant::armed)
            && let Some(sink) = taps.as_deref_mut()
        {
            capture_chunk_linear(parts.caches, sink, context_after, capture_checkpoint)?;
        }
        if !is_last_chunk {
            if let Some(sink) = taps.as_deref() {
                sink.eval()?;
            }
            crate::array::synchronize_and_clear_cache();
        }
        if !is_last_chunk || grant.is_some() {
            reach_chunk(target, taps.as_deref_mut(), context_after, grant)?;
        }
        if !is_last_chunk {
            crate::engine::progress::with_turn_progress(|p| {
                p.prefill_advanced(context_after - cached_prefix_len)
            });
        }
        if let Some(start) = chunk_trace_start {
            // This elapsed value includes the existing materialization point:
            // kept hidden eval for retained MTP history, or the normal
            // inter-chunk synchronize+clear. No synchronization is added for
            // tracing, so the final chunk explicitly reports whether its
            // retained hidden state supplied that barrier.
            let elapsed = elapsed_ms(start);
            let tok_s = if elapsed > 0.0 {
                chunk.len() as f64 / (elapsed / 1000.0)
            } else {
                0.0
            };
            let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
            tracing::info!(
                target: "mlx_core::inference",
                event = "paged_mtp_prefill_chunk_done",
                chunk_index = chunk_idx + 1,
                total_chunks,
                chunk_tokens = chunk.len(),
                context_before = chunk_start_position,
                context_after,
                retained_hidden = overlaps_kept_tail,
                materialized = overlaps_kept_tail || !is_last_chunk || capture_checkpoint,
                gdn_checkpoint_materialized = capture_checkpoint,
                elapsed_ms = elapsed,
                tok_s,
                active_mib,
                cache_mib,
                peak_mib,
                "paged MTP prefill chunk completed"
            );
        }
        chunk_start_position = context_after;
        suffix_offset = chunk_end;
    }

    let last_logits = last_logits.ok_or_else(|| {
        Error::from_reason(
            "chunked prefill (with-hidden) produced no last chunk (unreachable for non-empty suffix)",
        )
    })?;

    let mut retained = hidden_chunks.into_iter();
    let mut prompt_hidden = retained.next().ok_or_else(|| {
        Error::from_reason("run_paged_prefill_chunk_with_hidden: empty hidden chunks")
    })?;
    for chunk in retained {
        prompt_hidden = MxArray::concatenate(&prompt_hidden, &chunk, 1)?;
    }
    // Same contract the single-shot sibling ends on: the caller runs
    // `synchronize_and_clear_cache()` before `begin_mtp_decode` consumes this
    // as its prompt-prefix seed, so the concat must not be left lazy across
    // that sweep.
    prompt_hidden.eval();

    if let Some(start) = prefill_trace_start {
        tracing::info!(
            target: "mlx_core::inference",
            event = "paged_mtp_prefill_done",
            suffix_tokens = suffix_tokens.len(),
            cached_prefix_tokens = cached_prefix_len,
            total_chunks,
            kept_hidden_tokens = keep_last_hidden.unwrap_or(total_suffix_len).min(total_suffix_len),
            elapsed_ms = elapsed_ms(start),
            "paged MTP prefill completed"
        );
    }

    Ok((last_logits, prompt_hidden, checkpoints))
}

#[allow(clippy::too_many_arguments)]
fn run_paged_prefill_single_shot_with_hidden(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    target: &mut dyn PrefillTarget<DecoderLayer>,
    layer_kinds: &[Qwen3_5LayerKind],
    keep_last_hidden: Option<usize>,
    cached_rope_deltas: i32,
    taps: Option<&mut dyn PrefillTapSink>,
) -> Result<(MxArray, MxArray)> {
    // GDN pre-pass before `record_tokens`; see `run_paged_prefill_single_shot`
    // for why the two are order-independent.
    if cached_prefix_len > 0 && !gdn_prefix_already_primed {
        let prefix = &full_tokens[..(cached_prefix_len as usize)];
        let parts = target.parts()?;
        run_gdn_only_prefill(prefix, embed, parts.layers, parts.caches)?;
    }
    target
        .parts()?
        .adapter
        .confirm_aux_prefix_primed(cached_prefix_len)
        .map_err(Error::from_reason)?;

    target.record(suffix_tokens)?;
    let parts = target.parts()?;

    let tap_layers = taps
        .as_deref()
        .map_or_else(Vec::new, |sink| sink.tap_layers().to_vec());
    let prefilled = run_paged_prefill_one_chunk_tapped(
        suffix_tokens,
        cached_prefix_len,
        embed,
        parts.layers,
        parts.caches,
        layer_kinds,
        parts.adapter,
        /* inputs_embeds */ None,
        /* position_ids */ None,
        cached_rope_deltas,
        &tap_layers,
        ChunkOutput::Read,
    )?;
    if let Some(sink) = taps {
        sink.chunk(cached_prefix_len as usize, suffix_tokens, &prefilled.taps)?;
    }

    project_last_token_logits_with_full_hidden(
        prefilled.hidden()?,
        parts.final_norm,
        parts.lm_head,
        embed,
        keep_last_hidden,
    )
}

/// Forward one paged prefill chunk through every layer.
///
/// `inputs_embeds` is the image-merged token embeddings `[1, T, hidden]` for an
/// image-bearing prefill; when `Some` it replaces `embed.forward(chunk_tokens)`,
/// while `chunk_tokens` still drives `record_tokens` / the slot cursor upstream.
/// `position_ids` is the per-chunk M-RoPE slice `[3, 1, T]` (full-attention
/// layers only); both are `None` on the text-only path, which is byte-identical
/// to the prior behaviour.
/// Also returns the output rows of every layer in `tap_layers`, in that
/// order. With [`ChunkOutput::Unread`] a last layer that is full attention,
/// no tap and on the text RoPE path only stores its K/V ([`store_only_tail`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_prefill_one_chunk_tapped(
    chunk_tokens: &[u32],
    chunk_first_position: u32,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    inputs_embeds: Option<&MxArray>,
    position_ids: Option<&MxArray>,
    cached_rope_deltas: i32,
    tap_layers: &[usize],
    output: ChunkOutput,
) -> Result<PrefillChunk> {
    debug_assert_eq!(layers.len(), caches.len());
    debug_assert_eq!(layers.len(), layer_kinds.len());

    #[cfg(test)]
    let embed_scope = crate::array::host_waits::scope("prefill_chunk_graph");
    let mut hidden_states = match inputs_embeds {
        Some(embeds) => embeds.clone(),
        None => {
            let chunk_len = chunk_tokens.len() as i64;
            let input_ids = MxArray::from_uint32(chunk_tokens, &[1, chunk_len])?;
            embed.forward(&input_ids)?
        }
    };

    // Scalar-offset RoPE position for this chunk's queries/keys. For a text
    // suffix that warm-continues an image prefill, the rotation must trail the
    // physical slot by the negative cross-turn delta so the suffix keys stay
    // consistent with the immutable compressed-M-RoPE image keys. Text-only
    // prefill carries `cached_rope_deltas == 0` (offset == physical position),
    // and image prefill uses the M-RoPE arm so this is ignored there.
    let rope_position_offset = paged_rope_offset(chunk_first_position, cached_rope_deltas);

    // Shared per-forward-pass scratch slot for the M-RoPE cos/sin precompute
    // (see `Qwen3_5Attention::forward_paged`'s `mrope_cache` doc comment).
    // Every `FullAttentionPaged` layer in this loop shares one `position_ids`
    // array, so the first such layer computes the selected cos/sin and every
    // later one reuses it instead of recomputing the cos/sin table +
    // `take_along_axis` gather. Stays `None` (untouched) on the text-only
    // path where `position_ids` is `None`.
    let mut mrope_cache: Option<(MxArray, MxArray)> = None;
    let mut taps: Vec<Option<MxArray>> = vec![None; tap_layers.len()];
    let num_layers = layers.len();
    #[cfg(test)]
    drop(embed_scope);

    for (layer_idx, ((layer, cache_slot), kind)) in layers
        .iter_mut()
        .zip(caches.iter_mut())
        .zip(layer_kinds.iter().copied())
        .enumerate()
    {
        #[cfg(test)]
        let build_scope = crate::array::host_waits::scope("prefill_chunk_graph");
        if let Some(paged_idx) = store_only_tail(
            output,
            layer_idx,
            num_layers,
            kind,
            tap_layers,
            position_ids.is_some(),
        ) {
            layer.store_paged_kv(
                &hidden_states,
                paged_idx,
                paged_adapter,
                chunk_first_position,
                rope_position_offset,
            )?;
            let mut roots = vec![hidden_states];
            roots.extend(paged_adapter.pending_pool_write_arrays(paged_idx));
            #[cfg(test)]
            PREFILL_TAILS_STORED.with(|count| count.set(count.get() + 1));
            return Ok(PrefillChunk {
                hidden: None,
                roots,
                taps: collect_taps(taps, tap_layers)?,
            });
        }
        // M-RoPE positions feed full-attention layers only; GDN/linear layers
        // take none (matches the flat VLM prefill policy).
        let layer_positions = match kind {
            Qwen3_5LayerKind::FullAttentionPaged { .. } => position_ids,
            Qwen3_5LayerKind::Linear => None,
        };
        hidden_states = if separate_ops() {
            layer.forward_paged_or_flat(
                &hidden_states,
                kind,
                paged_adapter,
                chunk_first_position,
                PagedPass::Prefill,
                /* mask */ None,
                Some(cache_slot),
                layer_positions,
                /* use_kernel */ true,
                rope_position_offset,
                &mut mrope_cache,
            )?
        } else {
            layer.forward_paged_prefill(
                &hidden_states,
                kind,
                paged_adapter,
                chunk_first_position,
                Some(cache_slot),
                layer_positions,
                rope_position_offset,
                &mut mrope_cache,
            )?
        };
        for (slot, &tap_layer) in tap_layers.iter().enumerate() {
            if tap_layer == layer_idx {
                taps[slot] = Some(hidden_states.clone());
            }
        }
        #[cfg(test)]
        {
            drop(build_scope);
            note_timeline("layer", chunk_first_position + layer_idx as u32);
        }
        crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
    }
    let taps = collect_taps(taps, tap_layers)?;
    Ok(PrefillChunk {
        roots: vec![hidden_states.clone()],
        hidden: Some(hidden_states),
        taps,
    })
}

/// Whether a prefill chunk's caller reads the last layer's output rows
/// (logits, an MTP seed, kept hidden) or only needs the chunk's effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChunkOutput {
    Read,
    Unread,
}

/// One prefill chunk through the layer stack.
pub(crate) struct PrefillChunk {
    /// The last layer's output; `None` when that layer only stored its K/V.
    hidden: Option<MxArray>,
    /// What evaluates every effect of the chunk when nothing reads `hidden`:
    /// the output, or the input of a store-only last layer and its pool
    /// writes.
    roots: Vec<MxArray>,
    taps: Vec<MxArray>,
}

impl PrefillChunk {
    fn hidden(&self) -> Result<&MxArray> {
        self.hidden
            .as_ref()
            .ok_or_else(|| Error::from_reason("a store-only prefill chunk has no output rows"))
    }

    pub(crate) fn stored_only(&self) -> bool {
        self.hidden.is_none()
    }

    pub(crate) fn eval_roots(&self, context: &str) -> Result<()> {
        MxArray::eval_arrays_with_context(&self.roots.iter().collect::<Vec<_>>(), context)
    }
}

/// Runs the prefill layers on the separate ops (no GEMM store epilogues) on
/// this thread: the tests' reference.
fn separate_ops() -> bool {
    #[cfg(test)]
    return PREFILL_SEPARATE_OPS.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

#[cfg(test)]
thread_local! {
    pub(crate) static PREFILL_SEPARATE_OPS: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Evaluates a tapped chunk's sink apart from the chunk on this thread (no
/// merged eval).
fn serial_order() -> bool {
    #[cfg(test)]
    return PREFILL_SERIAL.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

#[cfg(test)]
thread_local! {
    pub(crate) static PREFILL_SERIAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Sink evals the serial order ran apart from the chunk's eval.
    pub(crate) static PREFILL_SINK_EVALS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    /// After each non-final tapped chunk's eval: whether every array its sink
    /// queued was evaluated.
    pub(crate) static PREFILL_SINK_READY: std::cell::RefCell<Vec<bool>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// `(event, row, when)`: `booked`, `built` (row = chunk start),
    /// `settled` (row = chunk end), `layer` (row = chunk start + layer, after
    /// the layer's graph was built).
    pub(crate) static PREFILL_TIMELINE: std::cell::RefCell<Option<PrefillTimeline>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) type PrefillTimeline = Vec<(&'static str, u32, Instant)>;

#[cfg(test)]
fn note_timeline(event: &'static str, row: u32) {
    PREFILL_TIMELINE.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push((event, row, Instant::now()));
        }
    });
}

#[cfg(test)]
fn available<'a>(arrays: impl IntoIterator<Item = &'a MxArray>) -> bool {
    arrays
        .into_iter()
        .all(|array| unsafe { mlx_sys::mlx_array_is_available(array.as_raw_ptr()) })
}

#[cfg(test)]
thread_local! {
    /// Runs every last layer in full on this thread.
    pub(crate) static PREFILL_TAIL_FULL: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Store-only last layers this thread ran.
    pub(crate) static PREFILL_TAILS_STORED: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    /// After each non-final chunk: whether the last full-attention layer's
    /// pool arrays were evaluated.
    pub(crate) static PREFILL_TAIL_WRITES_READY: std::cell::RefCell<Vec<bool>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
#[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
fn note_tail_writes(adapter: &PagedKVCacheAdapter, layer_kinds: &[Qwen3_5LayerKind]) {
    #[cfg(target_os = "macos")]
    if let Some(Qwen3_5LayerKind::FullAttentionPaged { paged_idx }) = layer_kinds.last() {
        let ready = adapter.pool_writes_evaluated(*paged_idx);
        PREFILL_TAIL_WRITES_READY.with(|log| log.borrow_mut().push(ready));
    }
}

/// The pool layer a chunk's layer `layer_idx` stores into when it runs as a
/// store-only tail (Splash `QwenTarget::planOutputRows`): the last layer, full
/// attention, no tap, of a chunk whose output no one reads, on the text RoPE
/// path (an M-RoPE layer may build the pass's cos/sin from its queries).
fn store_only_tail(
    output: ChunkOutput,
    layer_idx: usize,
    num_layers: usize,
    kind: Qwen3_5LayerKind,
    tap_layers: &[usize],
    mrope_positions: bool,
) -> Option<u32> {
    #[cfg(test)]
    if PREFILL_TAIL_FULL.with(std::cell::Cell::get) {
        return None;
    }
    let Qwen3_5LayerKind::FullAttentionPaged { paged_idx } = kind else {
        return None;
    };
    let unread = output == ChunkOutput::Unread;
    let tapped = tap_layers.contains(&layer_idx);
    (unread && layer_idx + 1 == num_layers && !tapped && !mrope_positions).then_some(paged_idx)
}

pub(crate) fn collect_taps(
    taps: Vec<Option<MxArray>>,
    tap_layers: &[usize],
) -> Result<Vec<MxArray>> {
    taps.into_iter()
        .zip(tap_layers)
        .map(|(tap, &layer)| {
            tap.ok_or_else(|| {
                Error::from_reason(format!(
                    "DFlash2 target tap at layer {layer} was not captured"
                ))
            })
        })
        .collect()
}

/// The end of a unit's last launch: its last row's logits when they are
/// read; otherwise the eval the last layer's barrier does on a read launch,
/// naming a store-only last layer's pool writes too.
fn end_unit(
    output: ChunkOutput,
    hidden: &MxArray,
    tail_writes: &[MxArray],
    parts: &PrefillParts<'_, DecoderLayer>,
    embed: &Embedding,
    _layer_kinds: &[Qwen3_5LayerKind],
) -> Result<Option<MxArray>> {
    if output == ChunkOutput::Read {
        return project_last_token_logits(hidden, parts.final_norm, parts.lm_head, embed).map(Some);
    }
    let mut roots = vec![hidden];
    roots.extend(tail_writes);
    MxArray::eval_arrays_with_context(&roots, "paged_prefill_unit_end")?;
    #[cfg(test)]
    note_tail_writes(&*parts.adapter, _layer_kinds);
    Ok(None)
}

fn project_last_token_logits(
    hidden_states: &MxArray,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embed: &Embedding,
) -> Result<MxArray> {
    let seq_len = hidden_states.shape_at(1)?;
    let last_hidden = hidden_states.slice_axis(1, seq_len - 1, seq_len)?;

    let h = final_norm.forward(&last_hidden)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else {
        embed.as_linear(&h)?
    };

    logits.squeeze(Some(&[0, 1]))
}

/// Project the FULL pre-norm hidden chunk through `final_norm` and the LM
/// head, returning `(last_token_logits[vocab], full_chunk_hidden[1, T, hidden])`.
///
/// The paged prefill variant needs every chunk's post-`final_norm` hidden so
/// the MTP committed-history prompt seed (`prompt_hidden`, consumed by
/// `begin_mtp_decode`) gets a contiguous `[1, prompt_len, hidden]` tensor —
/// mirrors `chunked_prefill_with_hidden` on the dense path.
fn project_last_token_logits_with_full_hidden(
    hidden_states: &MxArray,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embed: &Embedding,
    keep_last_hidden: Option<usize>,
) -> Result<(MxArray, MxArray)> {
    let prompt_len = hidden_states.shape_at(1)?;
    let hidden_dim = hidden_states.shape_at(2)?;
    let keep_start = keep_last_hidden
        .map(|keep| prompt_len.saturating_sub(keep.max(1) as i64))
        .unwrap_or(0);
    let kept_hidden = final_norm.forward(&hidden_states.slice_axis(1, keep_start, prompt_len)?)?;
    let kept_len = kept_hidden.shape_at(1)?;
    let last_hidden = kept_hidden.slice_axis(1, kept_len - 1, kept_len)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&last_hidden)?
    } else {
        embed.as_linear(&last_hidden)?
    };

    // The caller runs `synchronize_and_clear_cache()` after prefill, before
    // `begin_mtp_decode` consumes the kept hidden as its prompt-prefix seed
    // — that sweep would otherwise free the lazy graph nodes backing the
    // kept hidden. Materialise before return.
    kept_hidden.eval();
    debug_assert_eq!(kept_hidden.shape_at(0)?, 1);
    debug_assert!(kept_hidden.shape_at(1)? >= 1);
    debug_assert_eq!(kept_hidden.shape_at(2)?, hidden_dim);
    debug_assert_eq!(kept_hidden.dtype()?, hidden_states.dtype()?);

    let logits = logits.squeeze(Some(&[0, 1]))?;
    Ok((logits, kept_hidden))
}

/// Run one paged decode step: feed `[token_id]` through the model.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_decode_step(
    token_id: u32,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    cached_rope_deltas: i32,
) -> Result<MxArray> {
    // Capture logical position BEFORE record_tokens advances the
    // cursor.
    let first_logical_position = paged_adapter.current_token_count();
    paged_adapter
        .record_tokens(&[token_id])
        .map_err(Error::from_reason)?;

    let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
    let mut hidden_states = embed.forward(&input_ids)?;

    // Decode rotates the query at the physical slot plus the cross-turn M-RoPE
    // delta (0 for text turns) while K/V still writes at the physical slot.
    let rope_position_offset = paged_rope_offset(first_logical_position, cached_rope_deltas);

    let num_layers = layers.len();
    #[allow(clippy::needless_range_loop)]
    for layer_idx in 0..num_layers {
        let kind = layer_kinds[layer_idx];
        let layer = unsafe {
            let ptr = layers.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };
        let cache_slot = unsafe {
            let ptr = caches.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };

        hidden_states = layer.forward_paged_or_flat(
            &hidden_states,
            kind,
            paged_adapter,
            first_logical_position,
            PagedPass::Decode,
            /* mask */ None,
            Some(cache_slot),
            /* position_ids */ None,
            /* use_kernel */ true,
            rope_position_offset,
            &mut None,
        )?;
        maybe_submit_paged_decode_layer(layer_idx, num_layers, &hidden_states);
    }

    let h = final_norm.forward(&hidden_states)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else {
        embed.as_linear(&h)?
    };
    Ok(logits)
}

/// Eager paged MTP Step-A forward: a single `[1, 1]` paged forward returning
/// both the verifier logits and the post-`final_norm` hidden.
///
/// Routes full-attention layers through the paged adapter (writing one new K/V
/// slot into the pool, attending over `read_kv_range(0, total_ctx)`) and GDN
/// layers through the flat `Qwen3_5LayerCache::Linear` slots in `caches`, the
/// same split `run_paged_decode_step` uses. The eager analogue of the deleted
/// compiled `forward_with_hidden` closure that called `forward_dense_cpp_paged`
/// + `export_last_hidden_paged`.
///
/// Returns `(logits [1, 1, vocab], hidden [1, hidden])`. The hidden is squeezed
/// on the time axis to match the eager-flat MTP `forward_with_hidden` contract
/// (`needs_squeeze = true`); the caller reshapes it back to `[1, 1, hidden]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_step_with_hidden(
    token_id: u32,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    cached_rope_deltas: i32,
) -> Result<(MxArray, MxArray)> {
    let first_logical_position = paged_adapter.current_token_count();
    paged_adapter
        .record_tokens(&[token_id])
        .map_err(Error::from_reason)?;

    let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
    let mut hidden_states = embed.forward(&input_ids)?;

    // Same cross-turn delta as `run_paged_decode_step`: a text MTP Step-A
    // forward that warm-continues an image prefill must rotate at the
    // compressed position.
    let rope_position_offset = paged_rope_offset(first_logical_position, cached_rope_deltas);

    let num_layers = layers.len();
    #[allow(clippy::needless_range_loop)]
    for layer_idx in 0..num_layers {
        let kind = layer_kinds[layer_idx];
        let layer = unsafe {
            let ptr = layers.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };
        let cache_slot = unsafe {
            let ptr = caches.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };

        hidden_states = layer.forward_paged_or_flat(
            &hidden_states,
            kind,
            paged_adapter,
            first_logical_position,
            PagedPass::Decode,
            /* mask */ None,
            Some(cache_slot),
            /* position_ids */ None,
            /* use_kernel */ true,
            rope_position_offset,
            &mut None,
        )?;
        maybe_submit_paged_decode_layer(layer_idx, num_layers, &hidden_states);
    }

    let h3 = final_norm.forward(&hidden_states)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&h3)?
    } else {
        embed.as_linear(&h3)?
    };
    let hidden = h3.squeeze(Some(&[1]))?;
    Ok((logits, hidden))
}

/// Eager paged MTP batched verify forward: a single `[1, K+1]` paged forward
/// returning the verifier target distribution and the post-`final_norm` hidden
/// at every verify position, recording the per-layer GDN tape for the rollback
/// replay.
///
/// The eager analogue of the deleted compiled `forward_mtp_verify_paged` FFI. The
/// `verify_ids` (`[1, K+1]` int32) are recorded into the adapter in ONE
/// `record_tokens` call (so the new K/V land at logical positions
/// `[ctx, ctx+K]`), then run through every layer: full-attention via the paged
/// adapter (with `is_prefill = true` so the internal causal mask covers all
/// K+1 query positions over the full context), GDN via the flat `Linear`
/// slots while recording a [`GdnLayerTape`] (the bit-exactness keystone the
/// rollback replay consumes).
///
/// Returns `MtpVerifyOutput::logits_only(logits [1, K+1, vocab],
/// hiddens [1, K+1, hidden])`. The `tape` is pre-sized / cleared by this
/// function to `layers.len()` (`Some` for GDN layers, `None` for full-attn).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_verify_step(
    verify_ids: &[u32],
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    tape: &mut Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    cached_rope_deltas: i32,
) -> Result<super::mtp_decode::MtpVerifyOutput> {
    paged_verify_core(
        VerifyInput::Host(verify_ids),
        embed,
        layers,
        caches,
        final_norm,
        lm_head,
        layer_kinds,
        paged_adapter,
        tape,
        cached_rope_deltas,
        &[],
        None,
        None,
        false,
        false,
        VerifySubmit::Decode,
    )
    .map(|(output, _)| output)
}

/// The ids a paged verify embeds.
#[derive(Clone, Copy)]
pub(crate) enum VerifyInput<'a> {
    Host(&'a [u32]),
    /// `[rows]` device ids. The pool books `rows` placeholder records; the
    /// caller patches them with the host ids
    /// (`PagedKVCacheAdapter::patch_last_recorded_tokens`) before the cycle
    /// commits, or rolls the rows back.
    Device {
        ids: &'a MxArray,
        rows: usize,
    },
}

impl VerifyInput<'_> {
    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Host(ids) => ids.len(),
            Self::Device { rows, .. } => *rows,
        }
    }
}

/// Splash `PagedAttention::addVerify` check: a paged verify lane's chunk is
/// exactly the verify rows, its page table covers them, and they stay
/// inside the window plus its speculative scratch rows
/// (`chunkedPrefillValidationError`). Runs before anything is recorded.
pub(crate) fn require_verify_chunk(
    rows: usize,
    adapter: &PagedKVCacheAdapter,
    context_window: u32,
) -> Result<()> {
    use super::dflash2::DFLASH2_VERIFY_ROWS;
    let verify_rows = DFLASH2_VERIFY_ROWS as u64;
    let end = u64::from(adapter.current_token_count()) + verify_rows;
    let block = u64::from(adapter.block_size().max(1));
    let blocks = adapter
        .block_table()
        .map_or(0, |table| table.num_blocks() as u64);
    let cause = if rows != DFLASH2_VERIFY_ROWS {
        format!("{rows} rows, not {DFLASH2_VERIFY_ROWS}")
    } else if blocks < end.div_ceil(block) {
        format!("page table too short: {blocks} blocks for {end} rows of {block}-token blocks")
    } else if end > u64::from(context_window) + verify_rows - 1 {
        format!(
            "context out of range: {end} rows past the {context_window}-token window and its {} scratch rows",
            verify_rows - 1
        )
    } else {
        return Ok(());
    };
    Err(Error::from_reason(format!(
        "a paged verify lane's chunk is not a verify chunk: {cause}"
    )))
}

/// The DFlash2 verify on the pool: [`require_verify_chunk`], then one
/// 8-row forward that stores every row in the pages before attending over
/// them (Splash `PagedAttention::addVerify`), returns the logits of all
/// rows, the output rows of every layer in `tap_layers` (Splash
/// `Qwen3_5Target::addVerify` captures), and folds each linear layer's
/// pending rows into the lane's other GDN cell before its window (Splash
/// `gdn_decode_batch_phase`, see [`super::gdn_blob`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_dflash2_verify_step(
    verify_ids: VerifyInput<'_>,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    tape: &mut Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    cached_rope_deltas: i32,
    tap_layers: &[usize],
    gdn: &super::gdn_blob::GdnStateBlobs,
    row_budget: Option<&MxArray>,
    context_window: u32,
) -> Result<(MxArray, Vec<MxArray>)> {
    require_verify_chunk(verify_ids.rows(), paged_adapter, context_window)?;
    paged_verify_core(
        verify_ids,
        embed,
        layers,
        caches,
        final_norm,
        lm_head,
        layer_kinds,
        paged_adapter,
        tape,
        cached_rope_deltas,
        tap_layers,
        Some(gdn),
        row_budget,
        false,
        true,
        dflash2_verify_submit(VerifySubmit::EveryLayer),
    )
    .map(|(output, taps)| (output.logits, taps))
}

/// [`run_paged_dflash2_verify_step`] over a chunk the pool already recorded
/// (a scheduled lane's ticket records its 8 rows before the forward): the
/// rows end at the cursor, and the chunk was checked before the record.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_recorded_dflash2_verify_step(
    verify_ids: &[u32],
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    tape: &mut Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    cached_rope_deltas: i32,
    tap_layers: &[usize],
    gdn: &super::gdn_blob::GdnStateBlobs,
) -> Result<(MxArray, Vec<MxArray>)> {
    paged_verify_core(
        VerifyInput::Host(verify_ids),
        embed,
        layers,
        caches,
        final_norm,
        lm_head,
        layer_kinds,
        paged_adapter,
        tape,
        cached_rope_deltas,
        tap_layers,
        Some(gdn),
        None,
        true,
        true,
        dflash2_verify_submit(VerifySubmit::EveryLayer),
    )
    .map(|(output, taps)| (output.logits, taps))
}

#[allow(clippy::too_many_arguments)]
fn paged_verify_core(
    verify_ids: VerifyInput<'_>,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    tape: &mut Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    cached_rope_deltas: i32,
    tap_layers: &[usize],
    gdn: Option<&super::gdn_blob::GdnStateBlobs>,
    row_budget: Option<&MxArray>,
    recorded: bool,
    planes_route: bool,
    submit: VerifySubmit,
) -> Result<(super::mtp_decode::MtpVerifyOutput, Vec<MxArray>)> {
    debug_assert_eq!(layers.len(), caches.len());
    debug_assert_eq!(layers.len(), layer_kinds.len());
    #[cfg(test)]
    let _build_scope = crate::array::host_waits::scope("verify_graph");
    #[cfg(test)]
    note_verify_layer(None);

    let verify_len = verify_ids.rows();
    if verify_len == 0 {
        return Err(Error::from_reason("MTP verifier requires token IDs"));
    }
    let n_linear = layer_kinds
        .iter()
        .filter(|kind| matches!(kind, Qwen3_5LayerKind::Linear))
        .count();
    if let Some(gdn) = gdn {
        #[cfg(test)]
        DFLASH2_VERIFY_FORWARDS.with(|forwards| forwards.set(forwards.get() + 1));
        let dst = gdn.destinations();
        if dst.state.len() != n_linear || dst.conv.len() != n_linear {
            return Err(Error::from_reason(format!(
                "DFlash2 paged verify: {} GDN cells for {n_linear} linear layers",
                dst.state.len()
            )));
        }
    }
    let chunk_first_position = if recorded {
        let VerifyInput::Host(ids) = verify_ids else {
            return Err(Error::from_reason(
                "paged verify: a recorded chunk takes host ids",
            ));
        };
        let recorded_ids = paged_adapter.request_tokens();
        let first = recorded_ids
            .len()
            .checked_sub(verify_len)
            .filter(|&first| recorded_ids[first..] == *ids)
            .ok_or_else(|| {
                Error::from_reason("paged verify: the pool's last rows are not the verify ids")
            })?;
        u32::try_from(first).map_err(|_| Error::from_reason("paged verify position overflow"))?
    } else {
        paged_adapter.current_token_count()
    };
    let input_ids = match verify_ids {
        VerifyInput::Host(ids) => {
            if !recorded {
                paged_adapter
                    .record_tokens(ids)
                    .map_err(Error::from_reason)?;
            }
            MxArray::from_uint32(ids, &[1, verify_len as i64])?
        }
        VerifyInput::Device { ids, .. } => {
            if ids.ndim()? != 1 || ids.shape_at(0)? != verify_len as i64 {
                return Err(Error::from_reason(format!(
                    "DFlash2 paged verify: device ids of shape {:?} for {verify_len} rows",
                    ids.shape()?.as_ref()
                )));
            }
            let input_ids = ids
                .astype(crate::array::DType::Uint32)?
                .reshape(&[1, verify_len as i64])?;
            paged_adapter
                .record_placeholder_tokens(verify_len)
                .map_err(Error::from_reason)?;
            input_ids
        }
    };
    let mut hidden_states = embed.forward(&input_ids)?;

    // The K+1 verify ids rotate at the physical context start plus the
    // cross-turn M-RoPE delta (0 for text turns), matching the Step-A forward.
    let rope_position_offset = paged_rope_offset(chunk_first_position, cached_rope_deltas);

    let num_layers = layers.len();
    tape.clear();
    tape.resize(num_layers, None);
    let mut taps: Vec<Option<MxArray>> = vec![None; tap_layers.len()];
    let mut linear_ordinal = 0usize;
    #[allow(clippy::needless_range_loop)]
    for layer_idx in 0..num_layers {
        let kind = layer_kinds[layer_idx];
        let layer = unsafe {
            let ptr = layers.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };
        let cache_slot = unsafe {
            let ptr = caches.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };
        let mut slot: Option<super::gated_delta_net::GdnLayerTape> = None;
        hidden_states = match (kind, gdn) {
            (Qwen3_5LayerKind::Linear, Some(gdn)) => {
                let live = cache_slot.as_arrays_cache_mut().ok_or_else(|| {
                    Error::from_reason("DFlash2 paged verify: linear layer has no GDN slot")
                })?;
                let detached = gdn.verify_cell(live, linear_ordinal)?;
                linear_ordinal += 1;
                let mut detached = Qwen3_5LayerCache::Linear(detached);
                let hidden = if planes_route {
                    layer.forward_verify_with_tape(
                        &hidden_states,
                        kind,
                        paged_adapter,
                        chunk_first_position,
                        Some(&mut detached),
                        Some(&mut slot),
                        rope_position_offset,
                        row_budget,
                    )?
                } else {
                    layer.forward_paged_or_flat_with_tape(
                        &hidden_states,
                        kind,
                        paged_adapter,
                        chunk_first_position,
                        PagedPass::Verify,
                        Some(&mut detached),
                        Some(&mut slot),
                        rope_position_offset,
                        row_budget,
                    )?
                };
                let written = detached.as_arrays_cache_mut().ok_or_else(|| {
                    Error::from_reason("DFlash2 paged verify: detached GDN slot vanished")
                })?;
                super::gdn_blob::adopt_verify_cell(live, written)?;
                hidden
            }
            _ if planes_route => layer.forward_verify_with_tape(
                &hidden_states,
                kind,
                paged_adapter,
                chunk_first_position,
                Some(cache_slot),
                Some(&mut slot),
                rope_position_offset,
                row_budget,
            )?,
            _ => layer.forward_paged_or_flat_with_tape(
                &hidden_states,
                kind,
                paged_adapter,
                chunk_first_position,
                PagedPass::Verify,
                Some(cache_slot),
                Some(&mut slot),
                rope_position_offset,
                row_budget,
            )?,
        };
        tape[layer_idx] = slot;
        for (tap, &tap_layer) in tap_layers.iter().enumerate() {
            if tap_layer == layer_idx {
                taps[tap] = Some(hidden_states.clone());
            }
        }
        submit_verify_layer(submit, layer_idx, num_layers, &hidden_states);
        #[cfg(test)]
        note_verify_layer(Some(layer_idx));
    }

    let hiddens = final_norm.forward(&hidden_states)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&hiddens)?
    } else {
        embed.as_linear(&hiddens)?
    };
    Ok((
        super::mtp_decode::MtpVerifyOutput::logits_only(logits, hiddens),
        collect_taps(taps, tap_layers)?,
    ))
}

/// One lane of [`paged_verify_wave`]: a sequence whose verify ids the pool
/// already recorded, its cache slots (viewing its packed cells), its cells,
/// the tape the verify fills, and its cross-turn RoPE delta.
pub(crate) struct VerifyWaveLane<'a> {
    pub seq: crate::transformer::paged_kv_cache_adapter::SeqId,
    pub ids: &'a [u32],
    pub caches: &'a mut [Qwen3_5LayerCache],
    pub gdn: &'a super::gdn_blob::GdnStateBlobs,
    pub tape: &'a mut Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    pub rope_deltas: i32,
}

/// [`run_recorded_dflash2_verify_step`] for N lanes in one forward: the
/// activations stay `[N, T, H]` through embed, every layer
/// ([`DecoderLayer::forward_wave`]), the final norm and the head
/// (`forward_lanes`), so each projection streams its weights once. Per lane:
/// logits `[1, T, vocab]` and the taps, each the bits the lane kernel gives
/// that lane from the same state; every lane's tape and live slots are left
/// as the lane kernel leaves them. The last lane's sequence stays active.
#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_verify_wave(
    lanes: &mut [VerifyWaveLane<'_>],
    embed: &Embedding,
    layers: &[DecoderLayer],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    tap_layers: &[usize],
) -> Result<Vec<(MxArray, Vec<MxArray>)>> {
    use super::attention::PagedWaveLane;
    use super::gdn_blob::GdnWaveCells;
    debug_assert_eq!(layers.len(), layer_kinds.len());
    let rows = lanes.first().map_or(0, |lane| lane.ids.len());
    if rows == 0 || lanes.iter().any(|lane| lane.ids.len() != rows) {
        return Err(Error::from_reason(
            "DFlash2 verify wave: every lane needs the same nonzero row count",
        ));
    }
    let num_layers = layers.len();
    let submit = dflash2_verify_submit(VerifySubmit::Decode);
    let n_linear = layer_kinds
        .iter()
        .filter(|kind| matches!(kind, Qwen3_5LayerKind::Linear))
        .count();
    #[cfg(test)]
    DFLASH2_VERIFY_WAVES.with(|waves| {
        let (count, lanes_run) = waves.get();
        waves.set((count + 1, lanes_run + lanes.len() as u64));
    });
    let mut attn_lanes = Vec::with_capacity(lanes.len());
    for (i, lane) in lanes.iter().enumerate() {
        let dst = lane.gdn.destinations();
        if dst.state.len() != n_linear || dst.conv.len() != n_linear {
            return Err(Error::from_reason(format!(
                "DFlash2 verify wave: lane {i} has {} GDN cells for {n_linear} linear layers",
                dst.state.len()
            )));
        }
        if lane.caches.len() != num_layers {
            return Err(Error::from_reason(format!(
                "DFlash2 verify wave: lane {i} has {} cache slots for {num_layers} layers",
                lane.caches.len()
            )));
        }
        let recorded = paged_adapter.request_tokens_for(lane.seq).ok_or_else(|| {
            Error::from_reason(format!(
                "DFlash2 verify wave: sequence {} has no pool rows",
                lane.seq
            ))
        })?;
        let first = recorded
            .len()
            .checked_sub(rows)
            .filter(|&first| recorded[first..] == *lane.ids)
            .ok_or_else(|| {
                Error::from_reason("paged verify wave: the pool's last rows are not the verify ids")
            })?;
        let first = u32::try_from(first)
            .map_err(|_| Error::from_reason("paged verify position overflow"))?;
        attn_lanes.push(PagedWaveLane {
            seq: lane.seq,
            first_logical_position: first,
            rope_position_offset: paged_rope_offset(first, lane.rope_deltas),
        });
    }

    let mut ids = Vec::with_capacity(lanes.len() * rows);
    for lane in lanes.iter() {
        ids.extend_from_slice(lane.ids);
    }
    let input_ids = MxArray::from_uint32(&ids, &[lanes.len() as i64, rows as i64])?;
    let mut hidden_states = embed.forward(&input_ids)?;

    for lane in lanes.iter_mut() {
        lane.tape.clear();
        lane.tape.resize(num_layers, None);
    }
    let mut taps: Vec<Vec<Option<MxArray>>> = vec![vec![None; tap_layers.len()]; lanes.len()];
    let mut linear_ordinal = 0usize;
    for (layer_idx, (layer, &kind)) in layers.iter().zip(layer_kinds).enumerate() {
        let mut gdn_lanes: Vec<GdnWaveCells<'_>> = if matches!(kind, Qwen3_5LayerKind::Linear) {
            lanes
                .iter_mut()
                .map(|lane| GdnWaveCells {
                    caches: &mut *lane.caches,
                    blobs: lane.gdn,
                    tape: lane.tape.as_mut_slice(),
                })
                .collect()
        } else {
            Vec::new()
        };
        hidden_states = layer.forward_wave(
            &hidden_states,
            kind,
            layer_idx,
            linear_ordinal,
            paged_adapter,
            &attn_lanes,
            &mut gdn_lanes,
        )?;
        if matches!(kind, Qwen3_5LayerKind::Linear) {
            linear_ordinal += 1;
        }
        for (tap, &tap_layer) in tap_layers.iter().enumerate() {
            if tap_layer == layer_idx {
                for (i, lane_taps) in taps.iter_mut().enumerate() {
                    let from = i as i64;
                    lane_taps[tap] = Some(hidden_states.slice_axis(0, from, from + 1)?);
                }
            }
        }
        submit_verify_layer(submit, layer_idx, num_layers, &hidden_states);
    }

    let hiddens = final_norm.forward(&hidden_states)?;
    let logits = if let Some(head) = lm_head {
        head.forward_lanes(&hiddens)?
    } else {
        let per_lane = (0..lanes.len() as i64)
            .map(|i| embed.as_linear(&hiddens.slice_axis(0, i, i + 1)?))
            .collect::<Result<Vec<_>>>()?;
        MxArray::concatenate_many(per_lane.iter().collect(), Some(0))?
    };
    let mut out = Vec::with_capacity(lanes.len());
    for (i, lane_taps) in taps.into_iter().enumerate() {
        let from = i as i64;
        out.push((
            logits.slice_axis(0, from, from + 1)?,
            collect_taps(lane_taps, tap_layers)?,
        ));
    }
    Ok(out)
}

#[cfg(all(test, target_os = "macos"))]
mod wave_tests;

#[cfg(test)]
thread_local! {
    /// `(layer, when)` for every paged verify graph build on this thread:
    /// `None` at its start, then each layer once its graph (and its early
    /// submit) is built.
    pub(crate) static VERIFY_TIMELINE: std::cell::RefCell<Option<Vec<(Option<usize>, Instant)>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn note_verify_layer(layer: Option<usize>) {
    VERIFY_TIMELINE.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push((layer, Instant::now()));
        }
    });
}

#[cfg(test)]
mod gdn_checkpoint_tests {
    use super::{gdn_checkpoint_target, media_prefill_ranges, paged_prefill_ranges};

    fn assert_contiguous_cover(ranges: &[std::ops::Range<usize>], suffix_len: usize) {
        let mut cursor = 0;
        for range in ranges {
            assert_eq!(range.start, cursor, "range gap or overlap at {cursor}");
            assert!(range.end > range.start, "ranges must be non-empty");
            cursor = range.end;
        }
        assert_eq!(cursor, suffix_len);
    }

    #[test]
    fn checkpoint_target_matches_largest_reusable_block_boundary() {
        assert_eq!(gdn_checkpoint_target(37, 0, 16), Some(32));
        assert_eq!(gdn_checkpoint_target(33, 0, 16), Some(32));
        assert_eq!(gdn_checkpoint_target(32, 0, 16), Some(16));
        assert_eq!(gdn_checkpoint_target(17, 0, 16), Some(16));
        assert_eq!(gdn_checkpoint_target(37, 16, 16), Some(32));
        assert_eq!(gdn_checkpoint_target(37, 32, 16), None);
        assert_eq!(gdn_checkpoint_target(37, 33, 16), None);
        assert_eq!(gdn_checkpoint_target(32, 16, 16), None);
        assert_eq!(gdn_checkpoint_target(16, 16, 16), None);
        assert_eq!(gdn_checkpoint_target(16, 0, 16), None);
        assert_eq!(gdn_checkpoint_target(1, 0, 16), None);
        assert_eq!(gdn_checkpoint_target(0, 0, 16), None);
        assert_eq!(gdn_checkpoint_target(37, 0, 0), None);

        // These block-aligned boundaries are the three costly rollback cases
        // observed in the live agent trace.
        assert_eq!(gdn_checkpoint_target(41_216, 0, 16), Some(41_200));
        assert_eq!(gdn_checkpoint_target(45_360, 0, 16), Some(45_344));
        assert_eq!(gdn_checkpoint_target(47_488, 0, 16), Some(47_472));
    }

    #[test]
    fn prefill_ranges_split_at_checkpoint_inside_chunk() {
        let non_aligned_target = gdn_checkpoint_target(37, 0, 16);
        let ranges = paged_prefill_ranges(
            37,
            2048,
            &non_aligned_target
                .map(|v| v as usize)
                .into_iter()
                .collect::<Vec<_>>(),
        );
        assert_eq!(ranges, vec![0..32, 32..37]);
        assert_contiguous_cover(&ranges, 37);

        let aligned_target = gdn_checkpoint_target(32, 0, 16);
        let ranges = paged_prefill_ranges(
            32,
            2048,
            &aligned_target
                .map(|v| v as usize)
                .into_iter()
                .collect::<Vec<_>>(),
        );
        assert_eq!(ranges, vec![0..16, 16..32]);
        assert_contiguous_cover(&ranges, 32);

        // Cached prefix 16, full prompt 37: checkpoint 32 is suffix offset 16.
        let ranges = paged_prefill_ranges(21, 2048, &[16]);
        assert_eq!(ranges, vec![0..16, 16..21]);
        assert_contiguous_cover(&ranges, 21);
    }

    /// A ladder puts several boundaries inside one chunk. Every rung must end
    /// up a chunk edge, otherwise the snapshot for it is never taken.
    #[test]
    fn prefill_ranges_split_at_every_ladder_rung() {
        let ranges = paged_prefill_ranges(4080, 8192, &[64, 256, 1024, 4080]);
        assert_eq!(ranges, vec![0..64, 64..256, 256..1024, 1024..4080]);
        assert_contiguous_cover(&ranges, 4080);
    }

    #[test]
    fn prefill_ranges_keep_existing_chunk_boundary() {
        let ranges = paged_prefill_ranges(2053, 2048, &[2048]);
        assert_eq!(ranges, vec![0..2048, 2048..2053]);
        assert_contiguous_cover(&ranges, 2053);
    }

    #[test]
    fn prefill_ranges_cover_suffix_without_checkpoint() {
        let ranges = paged_prefill_ranges(5000, 2048, &[]);
        assert_eq!(ranges, vec![0..2048, 2048..4096, 4096..5000]);
        assert_contiguous_cover(&ranges, 5000);
    }

    /// A media prefill runs in chunks of at most the 2,048-row step the
    /// memory plan prices, never the whole suffix. Mutation: the whole suffix
    /// as the chunk in `media_prefill_ranges` → red (2 ranges, one of
    /// 49,152 rows).
    #[test]
    fn media_prefill_runs_in_step_sized_chunks() {
        if crate::array::paged_prefill_chunk_size() != 0 {
            return;
        }
        let ranges = media_prefill_ranges(50_000, &[49_152], 0);
        assert_eq!(ranges.len(), 25);
        assert!(ranges.iter().all(|range| range.len() <= 2048), "{ranges:?}");
        assert!(ranges.iter().any(|range| range.end == 49_152));
        assert_contiguous_cover(&ranges, 50_000);
    }
}

#[cfg(test)]
mod rope_offset_tests {
    //! Model-free coverage for the paged scalar-offset RoPE position helper.
    //!
    //! The cross-turn image-decode fix decouples the rotation position from the
    //! physical KV slot: image prefill compresses ~hundreds of placeholder
    //! tokens into far fewer M-RoPE positions, so a warm-continuation turn must
    //! rotate its queries at `physical_slot + cached_rope_deltas` (the delta is
    //! negative) while K/V still writes at the physical slot. These tests pin
    //! that arithmetic — including the cast-before-add type-safety guard — and
    //! the text-turn identity (`delta == 0`) that keeps text decode
    //! byte-identical. They construct no model, so they run on any host.

    use super::{paged_rope_offset, rope_delta_for_paged_turn};

    #[test]
    fn live_continuation_preserves_image_delta() {
        // A live continuation re-attends the image request's physically-resident
        // compressed-position K/V, so the negative delta MUST survive to keep the
        // text suffix rotating at the compressed position.
        assert_eq!(rope_delta_for_paged_turn(Some(-726), true), Some(-726));
        // Suffix at physical slot 754 then rotates at the compressed position 28.
        let delta = rope_delta_for_paged_turn(Some(-726), true).unwrap_or(0);
        assert_eq!(paged_rope_offset(754, delta), 28);
    }

    #[test]
    fn cold_start_clears_delta() {
        // A fresh/miss turn (no reused prefix) carries no cross-turn delta.
        assert_eq!(rope_delta_for_paged_turn(Some(-726), false), None);
        assert_eq!(rope_delta_for_paged_turn(None, false), None);
    }

    #[test]
    fn non_live_prefix_cache_hit_clears_stale_image_delta() {
        // Regression: a prior image turn leaves a stale negative delta on the
        // shared model. A later TEXT request that merely HITS the cross-request
        // prefix cache (cached_prefix_len > 0) is NOT a live image continuation
        // (continued_live_prefix == false) — its restored blocks can only be the
        // pure-text prefix. The stale delta must be dropped so text rotates at
        // the raw physical slot, NOT at physical + stale_negative_delta.
        let stale_image_delta = Some(-726);
        let after_text_hit = rope_delta_for_paged_turn(stale_image_delta, false);
        assert_eq!(after_text_hit, None);
        assert_eq!(paged_rope_offset(42, after_text_hit.unwrap_or(0)), 42);
    }

    #[test]
    fn text_turn_zero_delta_is_identity() {
        // Text-only turns store delta 0 -> the rotation offset equals the
        // physical KV slot exactly, keeping text decode byte-identical.
        assert_eq!(paged_rope_offset(0, 0), 0);
        assert_eq!(paged_rope_offset(42, 0), 42);
        assert_eq!(paged_rope_offset(1_000_000, 0), 1_000_000);
    }

    #[test]
    fn image_turn_negative_delta_shifts_offset_down() {
        // An image turn compressing ~754 placeholder tokens to ~28 M-RoPE
        // positions stores delta = 28 - 754 = -726. Decode at physical slot
        // 754 must rotate at the compressed position 28, NOT 754; the next
        // physical slot (755) rotates at 29.
        let delta = -726;
        assert_eq!(paged_rope_offset(754, delta), 28);
        assert_eq!(paged_rope_offset(755, delta), 29);
    }

    #[test]
    fn offset_casts_to_i32_before_adding_negative_delta() {
        // Type-safety guard: the cast to i32 happens BEFORE the add, so a small
        // physical position with a large negative delta yields a negative i32
        // rather than wrapping a u32 subtraction. In practice the physical
        // position always exceeds |delta| on a warm continuation, but the
        // helper must not underflow if it ever did not.
        assert_eq!(paged_rope_offset(10, -726), -716);
        // Physical position equal to |delta| collapses to exactly 0.
        assert_eq!(paged_rope_offset(726, -726), 0);
    }

    #[test]
    fn resetting_delta_to_zero_restores_physical_offset() {
        // Round-trip of the stored cross-turn delta: applying a negative delta
        // shifts the offset, and clearing it back to 0 (a fresh text turn /
        // `Option::unwrap_or(0)`) restores the physical position unchanged.
        let physical = 800;
        let with_image_delta = paged_rope_offset(physical, -726);
        assert_eq!(with_image_delta, 74);
        let after_reset = paged_rope_offset(physical, 0);
        assert_eq!(after_reset, physical as i32);
    }
}

#[cfg(test)]
mod last_row_projection_tests {
    use super::*;
    #[test]
    fn retained_prefill_seed_and_logits_match_full_projection() {
        let input = MxArray::from_float32(
            &(0..19 * 128)
                .map(|i| (i as f32 * 0.017).sin())
                .collect::<Vec<_>>(),
            &[1, 19, 128],
        )
        .unwrap()
        .astype(crate::array::DType::BFloat16)
        .unwrap();
        let mut norm = RMSNorm::new(128, Some(1e-6)).unwrap();
        norm.set_weight(
            &norm
                .get_weight()
                .astype(crate::array::DType::BFloat16)
                .unwrap(),
        )
        .unwrap();
        let mut embed = Embedding::new(64, 128).unwrap();
        embed
            .set_weight(
                &embed
                    .get_weight()
                    .astype(crate::array::DType::BFloat16)
                    .unwrap(),
            )
            .unwrap();
        let full_hidden = norm.forward(&input).unwrap();
        let expected_hidden = full_hidden.slice_axis(1, 18, 19).unwrap();
        let (logits, hidden) =
            project_last_token_logits_with_full_hidden(&input, &norm, &None, &embed, Some(1))
                .unwrap();
        let expected_logits = embed
            .as_linear(&expected_hidden)
            .unwrap()
            .squeeze(Some(&[0, 1]))
            .unwrap();
        assert_eq!(hidden.shape().unwrap().as_ref(), [1, 1, 128]);
        assert_eq!(
            hidden
                .astype(crate::array::DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .as_ref(),
            expected_hidden
                .astype(crate::array::DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .as_ref()
        );
        let actual = logits
            .astype(crate::array::DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let expected = expected_logits
            .astype(crate::array::DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let scale = expected.iter().map(|v| v.abs()).fold(1.0_f32, f32::max);
        let error = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            error <= scale * 0.01,
            "prefill head error {error} scale {scale}"
        );
    }
}

#[cfg(test)]
mod pass_sizing_tests {
    use super::*;

    const L: Qwen3_5LayerKind = Qwen3_5LayerKind::Linear;
    const A: Qwen3_5LayerKind = Qwen3_5LayerKind::FullAttentionPaged { paged_idx: 0 };

    fn rates(linear: Option<f64>, attention: Option<f64>) -> PassRates {
        PassRates {
            linear_ms: linear,
            attention_ms: attention,
            rows: 2048,
            pairs: 0,
        }
    }

    /// Layers are added while their estimates fit the budget; a pass runs
    /// at least one, never past the launch's last; a layer with no
    /// estimate runs alone. The 27B's grid at 24K (46 ms linear, 101 ms
    /// attention) in 100 ms passes is [L L] [L] [A].
    /// Mutations, each → red: zero layers when the first does not fit; the
    /// layer that overruns the budget is added.
    #[test]
    fn a_pass_takes_the_layers_that_fit() {
        let kinds = [L, L, L, A, L, L, L, A];
        let known = rates(Some(46.0), Some(101.0));
        assert_eq!(pass_layers(&kinds, 0, 100.0, &known), 2);
        assert_eq!(pass_layers(&kinds, 2, 100.0, &known), 1);
        assert_eq!(pass_layers(&kinds, 3, 100.0, &known), 1);
        assert_eq!(pass_layers(&kinds, 0, 1000.0, &known), 8);
        assert_eq!(pass_layers(&kinds, 7, 1000.0, &known), 1);
        assert_eq!(pass_layers(&kinds, 0, 0.0, &known), 1);
        let unknown = rates(Some(46.0), None);
        assert_eq!(pass_layers(&kinds, 0, 1000.0, &unknown), 3);
        assert_eq!(pass_layers(&kinds, 3, 1000.0, &unknown), 1);
        assert_eq!(pass_layers(&kinds, 0, 1000.0, &rates(None, None)), 1);
    }

    /// The passes a launch takes on the 27B's grid (L L L A x 16) at a
    /// 100 ms budget, from `rates`, when its layers really take `linear`
    /// and `attention` ms.
    fn passes_of_a_launch(mut rates: PassRates, linear: f64, attention: f64) -> Vec<usize> {
        let kinds: Vec<Qwen3_5LayerKind> = (0..64)
            .map(|layer| if (layer + 1) % 4 == 0 { A } else { L })
            .collect();
        let mut passes = Vec::new();
        let mut from = 0;
        while from < kinds.len() {
            let layers = pass_layers(&kinds, from, 100.0, &rates);
            let attention_layers = kinds[from..from + layers]
                .iter()
                .filter(|kind| !matches!(kind, Qwen3_5LayerKind::Linear))
                .count();
            let linear_layers = layers - attention_layers;
            rates.observe(
                linear_layers,
                attention_layers,
                linear_layers as f64 * linear + attention_layers as f64 * attention,
            );
            passes.push(layers);
            from += layers;
        }
        passes
    }

    /// A 20-row launch at 24K whose 64 layers take 40 ms in all runs in one
    /// pass after a 2048-row launch at 22K (46 ms linear, 101 ms
    /// attention): its numbers scale down too.
    /// Mutation: the numbers scale up only → red ([2, 1, 1, 60]).
    #[test]
    fn a_short_launch_after_a_deep_one_runs_in_one_pass() {
        let deep = PassRates {
            linear_ms: Some(46.0),
            attention_ms: Some(101.0),
            rows: 2048,
            pairs: crate::engine::scheduler::dense_attention_pairs(22_000, 2048),
        };
        let passes = passes_of_a_launch(deep.for_launch(24_000, 20), 0.6, 0.7);
        assert_eq!(passes, vec![64]);
        let deeper = passes_of_a_launch(deep.for_launch(24_000, 2048), 46.0, 101.0);
        assert!(deeper.len() > 30, "{deeper:?}");
    }

    /// A launch starts from the last launch's numbers scaled by rows and
    /// by attention pairs; a pass of linear layers sets the linear number,
    /// one with attention the attention number net of its linear layers.
    #[test]
    fn launch_rates_scale_up_and_passes_measure_them() {
        let last = PassRates {
            linear_ms: Some(40.0),
            attention_ms: Some(80.0),
            rows: 1024,
            pairs: crate::engine::scheduler::dense_attention_pairs(0, 1024),
        };
        let next = last.for_launch(1024, 2048);
        assert_eq!(next.linear_ms, Some(80.0));
        let pairs = crate::engine::scheduler::dense_attention_pairs(1024, 2048) as f64
            / crate::engine::scheduler::dense_attention_pairs(0, 1024) as f64;
        assert_eq!(next.attention_ms, Some(80.0 * pairs));
        let smaller = next.for_launch(3072, 16);
        assert_eq!(smaller.linear_ms, Some(80.0 * 16.0 / 2048.0));
        let mut measured = rates(None, None);
        measured.observe(2, 0, 90.0);
        assert_eq!(measured.linear_ms, Some(45.0));
        measured.observe(1, 1, 145.0);
        assert_eq!(measured.attention_ms, Some(100.0));
    }
}
