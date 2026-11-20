//! Scheduled DFlash2 lanes on a dense Qwen3.8 draft load: each owner's
//! cycle is the whole turn's (`Qwen35DFlash2Stepper`) over its own lane, under
//! the shared scheduled-verify transaction. Lanes that reach verify in one
//! step run one verify wave for 2..=4 lanes (each lane's bits are its own
//! `[1, 8]` verify's); a lone lane runs its `[1, 8]` verify.
use napi::bindgen_prelude::{Error, Result};

use super::decoder_layer::{AttentionType, DecoderLayer};
use super::dflash2::DFLASH2_VERIFY_ROWS;
use super::dflash2_decode::{draft_greedy_block, full_verify_ids, propose_block};
use super::dflash2_state::paged::OpenVerify;
use super::gated_delta_net::GdnLayerTape;
use super::gdn_blob::GdnStateBlobs;
use super::model::{LentRecurrent, Qwen35Inner};
use super::paged_forward::{VerifyWaveLane, paged_verify_wave};
use super::scheduled_mtp::MtpCache;
use crate::array::MxArray;
use crate::engine::backend::DsparkProposal;
use crate::engine::dspark_timeline::{self as timeline, Mark};
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, ScheduledVerifyCommit, ScheduledVerifyRow,
};
use crate::engine::params::ChatParams;
use crate::engine::scheduled_verify::{ScheduledDraftVerify, ScheduledVerifyBatch};
use crate::transformer::paged_kv_cache_adapter::PagedRaggedRow;

fn missing_lane(seq: u32) -> Error {
    Error::from_reason(format!("scheduled DFlash2 sequence {seq} has no lane"))
}

fn missing_draft() -> Error {
    Error::from_reason("scheduled DFlash2 has no draft model")
}

fn unpacked_cells() -> Error {
    Error::from_reason("Qwen3.8 DFlash2 needs packed GDN state (bf16 conv, f32 recurrent)")
}

/// Most lanes in one verify wave: the lane tier takes 2..=4 lanes of 8 rows.
const WAVE_LANES: usize = 4;

/// Whether this load's scheduled DFlash2 lanes verify in waves: every
/// layer's MLP projection runs on its attached planes or takes the lane
/// tier on this device, every GDN in-projection takes the lane tier, and
/// every GDN layer meets the complete kernel's contract. Logs one line
/// either way.
pub(crate) fn wave_ready_at_load(inner: &Qwen35Inner) -> bool {
    let (ready, line) = wave_load_line(inner);
    tracing::info!("{line}");
    ready
}

/// [`wave_ready_at_load`]'s answer and the load line it logs.
pub(crate) fn wave_load_line(inner: &Qwen35Inner) -> (bool, String) {
    let reason = match wave_unready(&inner.layers, i64::from(inner.config.hidden_size)) {
        Ok(None) => {
            return (
                true,
                format!(
                    "Qwen3.8 DFlash2 lanes: verify waves on, up to {WAVE_LANES} lanes per wave"
                ),
            );
        }
        Ok(Some(reason)) => reason,
        Err(error) => error.reason,
    };
    (
        false,
        format!("Qwen3.8 DFlash2 lanes: verify waves off, each lane verifies alone ({reason})"),
    )
}

/// Why `layers` on `hidden`-wide lanes cannot verify in waves on this
/// device, naming the first layer that fails; `None` when they can.
pub(crate) fn wave_unready(layers: &[DecoderLayer], hidden: i64) -> Result<Option<String>> {
    // A lane alone then splits K by its mixed projections' joint width; the
    // wave splits each projection by its own, so the lane's FP32 sums differ.
    // SAFETY: a nullary FFI query.
    if unsafe { mlx_sys::mlx_mixed_projection_global_split() } {
        return Ok(Some("MLX_MIXED_PROJECTION_GLOBAL_SPLIT=1".to_string()));
    }
    for (index, layer) in layers.iter().enumerate() {
        if let AttentionType::Linear(gdn) = &layer.attn
            && let Some(reason) = gdn.wave_unready(hidden)?
        {
            return Ok(Some(format!("layer {index}: {reason}")));
        }
        if !layer.mlp.wave_ready(hidden)? {
            return Ok(Some(format!("layer {index}: the MLP's lane tier")));
        }
    }
    Ok(None)
}

/// The rows of `rows` that verify in waves: groups of 2..=4 lanes in row
/// order. None on a load that is not wave-ready; an autoregressive peer's
/// row, and a lane whose cells cannot be lent, run alone beside the waves.
fn wave_groups(inner: &Qwen35Inner, rows: &[(PagedRaggedRow, Vec<u32>)]) -> Vec<Vec<usize>> {
    let lanes = &inner.dflash2_states.seq_lanes;
    if !inner.dflash2_wave_ready {
        return Vec::new();
    }
    let plain: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, (row, ids))| {
            ids.len() == DFLASH2_VERIFY_ROWS
                && lanes
                    .get(&row.seq_id)
                    .is_some_and(|lane| lane.open.is_none())
                && inner.can_lend_scheduled_recurrent(row.seq_id)
        })
        .map(|(index, _)| index)
        .collect();
    plain
        .chunks(WAVE_LANES)
        .filter(|group| group.len() >= 2)
        .map(<[usize]>::to_vec)
        .collect()
}

/// One verify wave over `group` (indexes of `rows`): every lane's cells lent
/// out at once, a lane without packed cells packed, one
/// [`paged_verify_wave`], and every lent set given back on every exit. Each
/// lane's verify is then open; returns each lane's logits `[1, 8, vocab]`.
/// On an error every lane's slots view its current cell again, and the
/// caller's transaction retracts every row of the step.
fn run_wave(
    inner: &mut Qwen35Inner,
    rows: &[(PagedRaggedRow, Vec<u32>)],
    group: &[usize],
    tap_layers: &[usize],
    tiled: bool,
) -> Result<Vec<MxArray>> {
    let mut lent = Vec::with_capacity(group.len());
    for &index in group {
        let seq = rows[index].0.seq_id;
        match inner.lend_scheduled_recurrent(seq) {
            Some(cells) => lent.push(cells),
            None => {
                inner.return_lent_recurrent(lent)?;
                return Err(Error::from_reason(format!(
                    "scheduled DFlash2 wave: sequence {seq} has no recurrent cells"
                )));
            }
        }
    }
    let planes_before = crate::models::quantized_linear::planes_calls();
    let waved = wave_forward(inner, rows, group, &mut lent, tap_layers, tiled);
    let planes_calls = crate::models::quantized_linear::planes_calls() - planes_before;
    let logits = match waved {
        Ok(waved) => open_wave(inner, rows, group, waved, planes_calls),
        Err(error) => {
            // Layers before the failing one adopted every lane's slots.
            for cells in &mut lent {
                if let Some(gdn) = inner
                    .dflash2_states
                    .seq_lanes
                    .get(&cells.seq)
                    .and_then(|lane| lane.gdn.as_ref())
                {
                    let _ = gdn.apply_views(&mut cells.caches);
                }
            }
            Err(error)
        }
    };
    let returned = inner.return_lent_recurrent(lent);
    let logits = logits?;
    returned?;
    Ok(logits)
}

/// One wave lane's result: its logits, its taps and its GDN tape.
type WaveLaneOut = (MxArray, Vec<MxArray>, Vec<Option<GdnLayerTape>>);

/// Each wave lane's verify open with its tape, taps and ids, and the planes
/// calls of the whole wave; returns the lanes' logits.
fn open_wave(
    inner: &mut Qwen35Inner,
    rows: &[(PagedRaggedRow, Vec<u32>)],
    group: &[usize],
    waved: Vec<WaveLaneOut>,
    planes_calls: u64,
) -> Result<Vec<MxArray>> {
    let mut logits = Vec::with_capacity(waved.len());
    for (&index, (lane_logits, tapped, tape)) in group.iter().zip(waved) {
        let (row, ids) = &rows[index];
        let lane = inner
            .dflash2_states
            .seq_lanes
            .get_mut(&row.seq_id)
            .ok_or_else(|| missing_lane(row.seq_id))?;
        lane.open = Some(OpenVerify {
            tape,
            tapped,
            ids: ids.clone(),
            lanes: group.len(),
            planes_calls,
        });
        logits.push(lane_logits);
    }
    Ok(logits)
}

/// [`run_wave`]'s forward over the lent cells (in `group` order).
fn wave_forward(
    inner: &mut Qwen35Inner,
    rows: &[(PagedRaggedRow, Vec<u32>)],
    group: &[usize],
    lent: &mut [LentRecurrent],
    tap_layers: &[usize],
    tiled: bool,
) -> Result<Vec<WaveLaneOut>> {
    for cells in lent.iter_mut() {
        let lane = inner
            .dflash2_states
            .seq_lanes
            .get_mut(&cells.seq)
            .ok_or_else(|| missing_lane(cells.seq))?;
        // The prefill left the sequence's GDN state in its slots: pack it
        // once into the lane's current cell.
        if lane.gdn.is_none() {
            let gdn = GdnStateBlobs::pack(&cells.caches, tiled)?.ok_or_else(unpacked_cells)?;
            gdn.apply_views(&mut cells.caches)?;
            lane.gdn = Some(gdn);
        }
    }
    let rope_deltas = inner.cached_rope_deltas.unwrap_or(0);
    let mut tapes: Vec<Vec<Option<GdnLayerTape>>> = lent.iter().map(|_| Vec::new()).collect();
    let lanes = &inner.dflash2_states.seq_lanes;
    let mut wave = Vec::with_capacity(lent.len());
    for ((cells, tape), &index) in lent.iter_mut().zip(tapes.iter_mut()).zip(group) {
        let gdn = lanes
            .get(&cells.seq)
            .and_then(|lane| lane.gdn.as_ref())
            .ok_or_else(super::dflash2_decode::missing_cells)?;
        wave.push(VerifyWaveLane {
            seq: cells.seq,
            ids: &rows[index].1,
            caches: &mut cells.caches,
            gdn,
            tape,
            rope_deltas,
        });
    }
    let adapter = inner
        .paged_adapter
        .as_mut()
        .ok_or_else(|| Error::from_reason("DFlash2 verify has no paged adapter"))?;
    let outputs = paged_verify_wave(
        &mut wave,
        &inner.embedding,
        &inner.layers,
        &inner.final_norm,
        &inner.lm_head,
        &inner.layer_kinds,
        adapter,
        tap_layers,
    )?;
    drop(wave);
    Ok(outputs
        .into_iter()
        .zip(tapes)
        .map(|((logits, tapped), tape)| (logits, tapped, tape))
        .collect())
}

/// What a scheduled DFlash2 lane holds beside its sequence's GDN cell of
/// `recurrent_state_bytes`: the second cell, the draft ring and context
/// window (`draft_lane_bytes`) and the pending GDN tape between its cycles
/// (Splash `CompositeStateLayout::laneBytes` and `DecodeTensor::GdnTape`).
pub(crate) fn lane_state_bytes(
    config: &super::config::Qwen3_5Config,
    recurrent_state_bytes: u64,
    draft_lane_bytes: u64,
) -> u64 {
    crate::engine::memory_plan::dflash2_lane_state_bytes(recurrent_state_bytes, draft_lane_bytes)
        .saturating_add(super::dflash2_decode::gdn_pending_tape_bytes(config))
}

/// [`lane_state_bytes`] of `inner`'s load: the draft state a scheduled
/// speculative turn on a draft load books, whole, from its admission.
/// `None` without a draft.
pub(crate) fn sequence_lane_bytes(inner: &Qwen35Inner) -> Option<u64> {
    let draft = inner.dflash2.as_ref()?;
    Some(lane_state_bytes(
        &inner.config,
        inner.config.recurrent_state_bytes(),
        draft.config.lane_bytes(),
    ))
}

/// A lane's verify books exactly its 8 rows (Splash `Engine::workEndOf`
/// for a decode command: committed + 8), inside the window and its 7
/// scratch rows; `Ok(false)` when they do not fit. The engine's query
/// count would leave the verify's page table short.
pub(crate) fn reserve_lane(inner: &mut Qwen35Inner, seq: u32) -> Result<bool> {
    timeline::claim_lane(seq);
    let window = inner.pool_context_limit();
    let adapter = inner
        .paged_adapter
        .as_mut()
        .ok_or_else(|| Error::from_reason("DFlash2 reservation has no paged adapter"))?;
    let committed = adapter
        .current_token_count_for(seq)
        .ok_or_else(|| Error::from_reason("DFlash2 reservation has no target frontier"))?;
    let end = u64::from(committed) + DFLASH2_VERIFY_ROWS as u64;
    if end > u64::from(window) + DFLASH2_VERIFY_ROWS as u64 - 1 {
        return Ok(false);
    }
    match adapter.reserve_rows_for(seq, DFLASH2_VERIFY_ROWS as u32) {
        Ok(_) => Ok(true),
        Err(error) if error.starts_with("context_length_exceeded:") => Ok(false),
        Err(error) => Err(Error::from_reason(error)),
    }
}

/// The cycle's whole draft from the lane's ring with the turn's `rng`; the
/// block stays on the lane, the driver gets `cap` proposals.
pub(crate) fn propose(
    inner: &mut Qwen35Inner,
    seq: u32,
    anchor: u32,
    cap: usize,
    params: &ChatParams,
    rng: &mut dyn rand::Rng,
) -> Result<DsparkProposal> {
    let draft = inner.dflash2.as_ref().ok_or_else(missing_draft)?;
    let states = &mut inner.dflash2_states;
    let lane = states
        .seq_lanes
        .get_mut(&seq)
        .ok_or_else(|| missing_lane(seq))?;
    if lane.open.is_some() {
        return Err(Error::from_reason(
            "scheduled DFlash2 propose: prior verify was not committed",
        ));
    }
    let (mut proposal, drafted) = propose_block(
        draft,
        &inner.embedding,
        inner.lm_head.as_ref(),
        &mut lane.ring,
        anchor,
        params,
        rng,
    )?;
    lane.drafted = Some(drafted);
    proposal.truncate(cap);
    Ok(proposal)
}

/// The step's verify: 2..=4 plain lanes per wave, every other row its own
/// `[1, 8]` verify. Each row's ids (anchor, then the driver's
/// proposals as the engine left them, a thinking cut's included) completed
/// with the rest of the lane's draft; an autoregressive peer's row is its
/// one token. Returns each row's `tokens.len()` logits rows, packed on
/// axis 0.
pub(crate) fn verify(inner: &mut Qwen35Inner, rows: &[ScheduledVerifyRow]) -> Result<MxArray> {
    let window = inner.pool_context_limit();
    let mut full_rows = Vec::with_capacity(rows.len());
    for row in rows {
        if row.tokens.is_empty() {
            return Err(Error::from_reason("an empty scheduled DFlash2 row"));
        }
        if !row.speculative {
            full_rows.push(ScheduledVerifyRow {
                seq_id: row.seq_id,
                first_position: row.first_position,
                tokens: row.tokens.clone(),
                speculative: false,
            });
            continue;
        }
        let drafted = inner
            .dflash2_states
            .seq_lanes
            .get_mut(&row.seq_id)
            .ok_or_else(|| missing_lane(row.seq_id))?
            .drafted
            .take();
        let drafted = match drafted {
            Some(drafted) => drafted.host()?,
            None => {
                let draft = inner.dflash2.as_ref().ok_or_else(missing_draft)?;
                let ring = &mut inner
                    .dflash2_states
                    .seq_lanes
                    .get_mut(&row.seq_id)
                    .ok_or_else(|| missing_lane(row.seq_id))?
                    .ring;
                draft_greedy_block(
                    draft,
                    &inner.embedding,
                    inner.lm_head.as_ref(),
                    ring,
                    row.tokens[0],
                )?
            }
        };
        timeline::lane_mark(row.seq_id, Mark::ProposalReadback);
        let ids = full_verify_ids(&row.tokens, &drafted)?;
        timeline::lane_mark(row.seq_id, Mark::VerifyInput);
        // Splash `PagedAttention::addVerify`: the booking covers the 8 rows
        // before anything is recorded.
        let adapter = inner
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("DFlash2 verify has no paged adapter"))?;
        adapter
            .activate_request(row.seq_id)
            .map_err(Error::from_reason)?;
        super::paged_forward::require_verify_chunk(ids.len(), adapter, window)?;
        timeline::lane_mark(row.seq_id, Mark::VerifyReserve);
        full_rows.push(ScheduledVerifyRow {
            seq_id: row.seq_id,
            first_position: row.first_position,
            tokens: ids,
            speculative: true,
        });
    }
    let logits = LaneVerify(inner).verify_scheduled_rows(&full_rows)?;
    let mut parts = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let start = full_rows[..index]
            .iter()
            .map(|row| row.tokens.len())
            .sum::<usize>() as i64;
        parts.push(logits.slice_axis(0, start, start + row.tokens.len() as i64)?);
    }
    let packed = MxArray::concatenate_many(parts.iter().collect(), Some(0))?;
    for row in rows {
        timeline::lane_mark(row.seq_id, Mark::VerifyGraph);
    }
    Ok(packed)
}

/// Closes each lane's verify keeping `keep` rows (cursor, GDN parity, ring
/// append of the kept taps), then queues each kept lane's arrays without a
/// blocking eval (the whole turn's `eval_boundary`).
pub(crate) fn commit(
    inner: &mut Qwen35Inner,
    rows: &[ScheduledVerifyCommit],
) -> Result<Vec<Result<()>>> {
    let results = LaneVerify(inner).commit_scheduled_rows(rows)?;
    for (row, result) in rows.iter().zip(&results) {
        if result.is_err() || row.keep == 0 {
            continue;
        }
        HybridSchedulerBackend::activate_paged_seq(inner, row.seq_id)?;
        let mut arrays: Vec<&MxArray> = Vec::new();
        crate::models::forward::CollectCacheArrays::collect_arrays(&inner.caches, &mut arrays);
        if let Some(lane) = inner.dflash2_states.seq_lanes.get(&row.seq_id) {
            lane.ring.collect_arrays(&mut arrays);
        }
        MxArray::async_eval_arrays(&arrays);
    }
    Ok(results)
}

/// `seq`'s lane ends: an open wave closes (every row keeps 0), the lane's
/// pending rows fold into the sequence's cells while they live, and its ring
/// goes idle. Whatever reads those cells next (the turn end's history
/// checkpoint and parked owner state, or a scheduled AR continuation) reads
/// the frontier.
pub(crate) fn end_lane(inner: &mut Qwen35Inner, seq: u32) {
    timeline::end_lane(seq);
    if !inner.dflash2_states.seq_lanes.contains_key(&seq) {
        return;
    }
    // A sequence the pool already released has no rows left to roll back.
    let _ = retract_open_wave(inner);
    inner.settle_scheduled_lane(seq);
    if let Some(lane) = inner.dflash2_states.seq_lanes.get(&seq) {
        let fields = format!(
            "{} planes_calls={} wave_planes_calls={}",
            lane.verifies.fields(),
            lane.planes_calls,
            lane.wave_planes_calls
        );
        tracing::info!("Qwen3.8 DFlash2 lane end: seq={seq} {fields}");
        crate::inference_trace::write(format_args!(
            "[MLX_TRACE] qwen3.5-dense dflash2_lane_end seq={seq} {fields}"
        ));
    }
    inner.dflash2_states.release_lane(seq);
}

/// Retracts the lanes' open verify wave: every row keeps 0.
pub(crate) fn retract_open_wave(inner: &mut Qwen35Inner) -> Result<()> {
    match inner.dflash2_states.lane_batch.take() {
        Some(batch) => LaneVerify(inner).retract_verify_batch(batch),
        None => Ok(()),
    }
}

/// Points `seq`'s cache slots back at its lane's current cell: a verify
/// left them on the cell it folded into, which a zero keep does not adopt.
fn restore_views(inner: &mut Qwen35Inner, seq: u32) -> Result<()> {
    HybridSchedulerBackend::activate_paged_seq(inner, seq)?;
    let caches = inner
        .caches
        .as_mut()
        .ok_or_else(|| Error::from_reason("scheduled DFlash2 target caches are absent"))?;
    inner
        .dflash2_states
        .seq_lanes
        .get(&seq)
        .and_then(|lane| lane.gdn.as_ref())
        .ok_or_else(super::dflash2_decode::missing_cells)?
        .apply_views(caches)
}

struct LaneVerify<'a>(&'a mut Qwen35Inner);

impl ScheduledDraftVerify for LaneVerify<'_> {
    type Cache<'a>
        = MtpCache<'a>
    where
        Self: 'a;

    fn verify_cache(&mut self) -> Self::Cache<'_> {
        MtpCache(self.0.paged_adapter.as_mut())
    }

    fn take_verify_batch(&mut self) -> Option<ScheduledVerifyBatch> {
        self.0.dflash2_states.lane_batch.take()
    }

    fn store_verify_batch(&mut self, batch: ScheduledVerifyBatch) {
        self.0.dflash2_states.lane_batch = Some(batch);
    }

    fn validate_draft_owner(&self, row: &ScheduledVerifyRow) -> Result<()> {
        if !row.speculative {
            if row.tokens.len() != 1 || self.0.dflash2_states.seq_lanes.contains_key(&row.seq_id) {
                return Err(Error::from_reason(
                    "scheduled DFlash2 peer/verify row mismatch",
                ));
            }
            return Ok(());
        }
        let lane = self
            .0
            .dflash2_states
            .seq_lanes
            .get(&row.seq_id)
            .ok_or_else(|| missing_lane(row.seq_id))?;
        if !row.speculative
            || row.tokens.len() != DFLASH2_VERIFY_ROWS
            || lane.open.is_some()
            || i64::from(lane.next_position) != i64::from(row.first_position)
        {
            return Err(Error::from_reason(
                "scheduled DFlash2 lane/verify row mismatch",
            ));
        }
        Ok(())
    }

    fn run_packed_target(
        &mut self,
        rows: &[(PagedRaggedRow, Vec<u32>)],
    ) -> Result<(MxArray, Vec<MxArray>)> {
        let inner = &mut *self.0;
        let tap_layers = inner
            .dflash2
            .as_ref()
            .ok_or_else(missing_draft)?
            .config
            .target_layers
            .clone();
        let drafting = rows
            .iter()
            .any(|(row, _)| inner.dflash2_states.seq_lanes.contains_key(&row.seq_id));
        // Only the last lane's release drops the limit.
        if drafting && inner.dflash2_states.lane_decode_limit.is_none() {
            let window = inner.pool_context_limit();
            inner.dflash2_states.lane_decode_limit =
                Some(crate::cache_limit::coordinator().push_decode_limit(
                    crate::cache_limit::decode_cache_limit(
                        super::dflash2_decode::pool_verify_cycle_bytes(
                            &inner.config,
                            u64::from(window),
                        ),
                    ),
                ));
        }
        let tiled = inner.config.qwen35_gguf_gdn_layout.as_deref() == Some("tiled");
        let mut outputs: Vec<Option<MxArray>> = rows.iter().map(|_| None).collect();
        for group in wave_groups(inner, rows) {
            let logits = run_wave(inner, rows, &group, &tap_layers, tiled)?;
            for (index, logits) in group.into_iter().zip(logits) {
                outputs[index] = Some(logits.transpose(Some(&[1, 0, 2]))?);
            }
        }
        for ((row, ids), output) in rows.iter().zip(outputs.iter_mut()) {
            if output.is_some() {
                continue;
            }
            let seq = row.seq_id;
            if !inner.dflash2_states.seq_lanes.contains_key(&seq) {
                let (logits, before) =
                    inner.run_recorded_decode_row(seq, ids[0], row.first_logical_position)?;
                inner.dflash2_states.peer_cells.insert(seq, before);
                *output = Some(logits.transpose(Some(&[1, 0, 2]))?);
                continue;
            }
            HybridSchedulerBackend::activate_paged_seq(inner, seq)?;
            let rope_deltas = inner.cached_rope_deltas.unwrap_or(0);
            let caches = inner
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("scheduled DFlash2 target caches are absent"))?;
            let lane = inner
                .dflash2_states
                .seq_lanes
                .get_mut(&seq)
                .ok_or_else(|| missing_lane(seq))?;
            // The prefill left the sequence's GDN state in its slots: pack
            // it once into the lane's current cell.
            if lane.gdn.is_none() {
                let gdn = GdnStateBlobs::pack(caches, tiled)?.ok_or_else(unpacked_cells)?;
                gdn.apply_views(caches)?;
                lane.gdn = Some(gdn);
            }
            let gdn = inner
                .dflash2_states
                .seq_lanes
                .get(&seq)
                .and_then(|lane| lane.gdn.as_ref())
                .ok_or_else(super::dflash2_decode::missing_cells)?;
            let adapter = inner
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("DFlash2 verify has no paged adapter"))?;
            let mut tape = Vec::new();
            let planes_before = crate::models::quantized_linear::planes_calls();
            let (logits, tapped) = super::paged_forward::run_recorded_dflash2_verify_step(
                ids,
                &inner.embedding,
                &mut inner.layers,
                caches,
                &inner.final_norm,
                &inner.lm_head,
                &inner.layer_kinds,
                adapter,
                &mut tape,
                rope_deltas,
                &tap_layers,
                gdn,
            )?;
            let lane = inner
                .dflash2_states
                .seq_lanes
                .get_mut(&seq)
                .ok_or_else(|| missing_lane(seq))?;
            lane.open = Some(OpenVerify {
                tape,
                tapped,
                ids: ids.clone(),
                lanes: 1,
                planes_calls: crate::models::quantized_linear::planes_calls() - planes_before,
            });
            *output = Some(logits.transpose(Some(&[1, 0, 2]))?);
        }
        let outputs = outputs
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::from_reason("scheduled DFlash2 verify left a row unrun"))?;
        Ok((
            MxArray::concatenate_many(outputs.iter().collect(), Some(0))?,
            Vec::new(),
        ))
    }

    fn commit_target_state(&mut self, seq: u32, keep: usize) -> Result<()> {
        timeline::lane_mark(seq, Mark::CommitKv);
        if let Some(before) = self.0.dflash2_states.peer_cells.remove(&seq) {
            if keep == 0 {
                return self.0.reinstate_scheduled_recurrent(seq, before);
            }
            return Ok(());
        }
        if keep == 0 {
            if let Some(lane) = self.0.dflash2_states.seq_lanes.get_mut(&seq) {
                lane.open = None;
            }
            return restore_views(self.0, seq);
        }
        let lane = self
            .0
            .dflash2_states
            .seq_lanes
            .get_mut(&seq)
            .ok_or_else(|| missing_lane(seq))?;
        let tape = lane
            .open
            .as_mut()
            .map(|open| std::mem::take(&mut open.tape))
            .ok_or_else(|| Error::from_reason("scheduled DFlash2 commit has no GDN tape"))?;
        lane.gdn
            .as_mut()
            .ok_or_else(super::dflash2_decode::missing_cells)?
            .retain(&tape, keep)?;
        timeline::lane_mark(seq, Mark::CommitGdn);
        Ok(())
    }

    fn abort_target_state(&mut self, seq: u32) {
        if let Some(before) = self.0.dflash2_states.peer_cells.remove(&seq) {
            // A peer whose cells cannot go back fails its turn with the
            // failure that aborts it.
            let _ = self.0.reinstate_scheduled_recurrent(seq, before);
            return;
        }
        let verified = self
            .0
            .dflash2_states
            .seq_lanes
            .get_mut(&seq)
            .and_then(|lane| lane.open.take())
            .is_some();
        // The failure that aborts this lane fails its turn; the slots only
        // matter to a lane that lives on.
        if verified {
            let _ = restore_views(self.0, seq);
        }
    }

    fn append_committed_taps(&mut self, seq: u32, _taps: &[MxArray], keep: usize) -> Result<()> {
        let open = self
            .0
            .dflash2_states
            .seq_lanes
            .get_mut(&seq)
            .ok_or_else(|| missing_lane(seq))?
            .open
            .take()
            .ok_or_else(|| Error::from_reason("scheduled DFlash2 commit has no target taps"))?;
        let kept = open.ids.get(..keep).ok_or_else(|| {
            Error::from_reason("scheduled DFlash2 keeps more rows than it verified")
        })?;
        let draft = self.0.dflash2.as_ref().ok_or_else(missing_draft)?;
        let lane = self
            .0
            .dflash2_states
            .seq_lanes
            .get_mut(&seq)
            .ok_or_else(|| missing_lane(seq))?;
        lane.append_kept(draft, &open.tapped, kept)?;
        lane.verifies.record(open.lanes);
        lane.planes_calls += open.planes_calls;
        if open.lanes > 1 {
            lane.wave_planes_calls += open.planes_calls;
        }
        timeline::lane_mark(seq, Mark::CommitRing);
        Ok(())
    }

    fn discard_draft_owner(&mut self, seq: u32) {
        if let Some(lane) = self.0.dflash2_states.seq_lanes.get_mut(&seq) {
            lane.open = None;
            lane.drafted = None;
        }
    }
}
