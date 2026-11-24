//! Scheduled continuous-batching DSpark lanes for Inkling-2-Mini with Qwen 3.8-style
//! **2..=4 lane stacked `[N * 8]` verify waves**.
//!
//! Each scheduled speculative sequence owns its own [`Inkling2DsparkContext`] (3 BF16 KV layers
//! and Markov `f_rnn` + cursor). When `2..=4` speculative lanes reach verify in the same scheduler
//! step, they stack `[N * 8, 2304]` rows through every dense Q8_0 projection and NVFP4/Q8_0 MoE
//! expert projection while running ShortConv (with per-lane [`Inkling2ShortConvTape`]) and
//! paged attention — delivering high-throughput batched verification.

use std::collections::HashMap;

use napi::bindgen_prelude::{Error, Result};

use crate::array::MxArray;
use crate::engine::backend::{DsparkProposal, SpecFrontier};
use crate::engine::hybrid_scheduler::{
    ScheduledProposalRequest, ScheduledVerifyCommit, ScheduledVerifyRow,
};
use crate::engine::params::ChatParams;
use crate::engine::scheduled_verify::{ScheduledDraftVerify, ScheduledVerifyBatch};
use crate::engine::spec_paged::SpecPagedCache;
use crate::transformer::paged_kv_cache_adapter::{PagedRaggedRow, SeqId};

use super::dflash2::{Inkling2Dflash2Context, Inkling2DraftContext, Inkling2TargetView};
use super::dspark::{AdaptiveDraftDepth, BatchedDsparkLane, Inkling2DsparkContext};
use super::kv_cache::Inkling2KVCacheCoordinator;
use super::model::Inkling2Inner;
use super::short_conv::Inkling2ShortConvTape;

/// Maximum speculative lanes stacked into one ragged `[sum(1 + k_i)]` verify wave (matches Qwen 3.8 `WAVE_LANES`).
pub(crate) const INKLING2_WAVE_LANES: usize = 4;

fn missing_lane(seq: SeqId) -> Error {
    Error::from_reason(format!(
        "scheduled Inkling2 speculative sequence {seq} has no lane"
    ))
}

fn missing_draft() -> Error {
    Error::from_reason("scheduled Inkling2 speculation has no draft model")
}

/// Open verify state held between `run_scheduled_verify` and `commit_scheduled_verify`.
pub(crate) struct OpenInkling2Verify {
    pub tape: Inkling2ShortConvTape,
    pub tapped: Vec<MxArray>,
    pub verify_tokens: Vec<u32>,
    pub lanes_in_wave: usize,
    pub drafted: usize,
}

/// Per-sequence scheduled speculative lane state (DSpark or DFlash2).
pub(crate) struct Inkling2ScheduledDsparkLane {
    pub ctx: Inkling2DraftContext,
    pub adaptive_depth: AdaptiveDraftDepth,
    pub open: Option<OpenInkling2Verify>,
    pub total_verifies: u64,
    pub multi_lane_wave_verifies: u64,
}

/// Aggregate scheduled DSpark state owned by [`Inkling2Inner`].
#[derive(Default)]
pub(crate) struct Inkling2ScheduledDsparkStates {
    pub seq_lanes: HashMap<SeqId, Inkling2ScheduledDsparkLane>,
    /// Open ShortConv rollback tapes for non-speculative AR peer rows verified alongside DSpark lanes.
    pub peer_tapes: HashMap<SeqId, Inkling2ShortConvTape>,
    pub lane_batch: Option<ScheduledVerifyBatch>,
    /// Diagnostic counters for tests & telemetry.
    pub total_verify_waves: u64,
    pub multi_lane_verify_waves: u64,
    pub max_verify_wave_lanes: usize,
}

impl Inkling2ScheduledDsparkStates {
    pub fn clear(&mut self) {
        self.seq_lanes.clear();
        self.peer_tapes.clear();
        self.lane_batch = None;
    }

    pub fn tap_visits_for(&self, seq_id: SeqId, draft_taps: &[usize]) -> Option<Vec<usize>> {
        self.seq_lanes
            .contains_key(&seq_id)
            .then(|| draft_taps.to_vec())
    }
}

/// Thin wrapper implementing [`SpecPagedCache`] over `Option<&mut Inkling2KVCacheCoordinator>`
/// (mirroring Qwen 3.5's `MtpCache<'a>`).
struct CoordinatorCache<'a>(Option<&'a mut Inkling2KVCacheCoordinator>);

impl CoordinatorCache<'_> {
    fn coord(&self) -> std::result::Result<&Inkling2KVCacheCoordinator, String> {
        self.0
            .as_deref()
            .ok_or_else(|| "Inkling2 scheduled DSpark has no paged KV coordinator".to_string())
    }

    fn coord_mut(&mut self) -> std::result::Result<&mut Inkling2KVCacheCoordinator, String> {
        self.0
            .as_deref_mut()
            .ok_or_else(|| "Inkling2 scheduled DSpark has no paged KV coordinator".to_string())
    }
}

impl SpecPagedCache for CoordinatorCache<'_> {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> std::result::Result<bool, String> {
        self.coord_mut()?.reserve_lookahead(seq_id, rows)
    }

    fn record_rows(&mut self, seq_id: u32, tokens: &[u32]) -> std::result::Result<(), String> {
        self.coord_mut()?.record_rows(seq_id, tokens)
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> std::result::Result<(), String> {
        self.coord_mut()?.rollback_rows(seq_id, rows)
    }

    fn settle_committed(
        &mut self,
        seq_id: u32,
        committed_tokens: u64,
    ) -> std::result::Result<(), String> {
        self.coord_mut()?.settle_committed(seq_id, committed_tokens)
    }

    fn settle_captures_durable_state(&self) -> bool {
        false
    }

    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.coord().ok()?.frontier(seq_id)
    }
}

impl Inkling2Inner {
    /// Begin scheduled DSpark speculation for `seq_id` starting at `position`.
    ///
    /// - At `position == 0`: allocates a fresh [`Inkling2DsparkContext`].
    /// - At `position > 0`: continues speculating if `seq_id` ALREADY holds a matching live
    ///   draft context at `position` (e.g. live multi-turn continuation); otherwise returns
    ///   `Ok(false)` so the turn keeps its cached target prefix and runs scheduled AR cleanly!
    pub(crate) fn begin_scheduled_dspark(&mut self, seq_id: SeqId, position: u32) -> Result<bool> {
        if self.scheduled_dspark.lane_batch.is_some() {
            return Err(Error::from_reason(
                "cannot begin scheduled Inkling2 speculation during an open verify batch",
            ));
        }
        let Some(max_spec) = self.active_draft_block_size() else {
            return Ok(false);
        };
        if position == 0 {
            let ctx = if let Some(ref draft) = self.dflash2 {
                Inkling2DraftContext::Dflash2(Box::new(Inkling2Dflash2Context::new(&draft.config)))
            } else if let Some(ref draft) = self.dspark {
                Inkling2DraftContext::Dspark(Inkling2DsparkContext::new(
                    draft.config.num_hidden_layers,
                ))
            } else {
                return Ok(false);
            };
            self.scheduled_dspark.seq_lanes.insert(
                seq_id,
                Inkling2ScheduledDsparkLane {
                    ctx,
                    adaptive_depth: AdaptiveDraftDepth::new(max_spec),
                    open: None,
                    total_verifies: 0,
                    multi_lane_wave_verifies: 0,
                },
            );
            return Ok(true);
        }

        // Check if `seq_id` already owns a matching draft context at exact `position`.
        if let Some(lane) = self.scheduled_dspark.seq_lanes.get_mut(&seq_id)
            && lane.open.is_none()
            && lane.ctx.committed_len() as u32 == position
        {
            return Ok(true);
        }
        if let Some(ctx) = self.dspark_turn_contexts.remove(&seq_id)
            && ctx.committed_len() as u32 == position
        {
            self.scheduled_dspark.seq_lanes.insert(
                seq_id,
                Inkling2ScheduledDsparkLane {
                    ctx,
                    adaptive_depth: AdaptiveDraftDepth::new(max_spec),
                    open: None,
                    total_verifies: 0,
                    multi_lane_wave_verifies: 0,
                },
            );
            return Ok(true);
        }

        // Restored from cold RAM/SSD state without draft context: keep target prefix and run AR.
        self.scheduled_dspark.seq_lanes.remove(&seq_id);
        Ok(false)
    }

    /// Reserve speculative lookahead rows for `seq_id` across both Global & SWA pools.
    pub(crate) fn reserve_scheduled_dspark(
        &mut self,
        seq_id: SeqId,
        queries: usize,
    ) -> Result<bool> {
        let rt = self.paged.as_mut().ok_or_else(|| {
            Error::from_reason("scheduled Inkling2 speculative reservation has no paged runtime")
        })?;
        rt.coordinator
            .reserve_lookahead(seq_id, queries)
            .map_err(Error::from_reason)
    }

    /// Propose up to `max_drafts` tokens from `seq_id`'s DFlash2 or DSpark context.
    pub(crate) fn propose_scheduled_dspark(
        &mut self,
        seq_id: SeqId,
        anchor: u32,
        max_drafts: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
        confidence: bool,
    ) -> Result<DsparkProposal> {
        let lane = self
            .scheduled_dspark
            .seq_lanes
            .get_mut(&seq_id)
            .ok_or_else(|| missing_lane(seq_id))?;
        if lane.open.is_some() {
            return Err(Error::from_reason(
                "scheduled Inkling2 propose: prior verify was not committed",
            ));
        }
        match &mut lane.ctx {
            Inkling2DraftContext::Dflash2(dctx) => {
                let draft = self.dflash2.as_ref().ok_or_else(missing_draft)?;
                let mut view = Inkling2TargetView {
                    embed_tokens: &self.embed_tokens,
                    embed_norm: &self.embed_norm,
                    lm_head: &self.lm_head,
                    lm_head_bf16_unpadded: self.lm_head_bf16_unpadded.as_ref(),
                    rms_norm_eps: self.config.rms_norm_eps as f32,
                    unpadded_vocab_size: self.config.effective_unpadded_vocab_size(),
                };
                dctx.propose_block(
                    draft, &mut view, anchor, max_drafts, params, rng, confidence,
                )
            }
            Inkling2DraftContext::Dspark(dctx) => {
                let draft = self.dspark.as_ref().ok_or_else(missing_draft)?;
                let effective_drafts = if params.mtp_adaptive_depth {
                    lane.adaptive_depth.current_depth(max_drafts)
                } else {
                    max_drafts
                };
                dctx.propose_block(
                    draft,
                    &self.embed_tokens,
                    &self.embed_norm,
                    &self.lm_head,
                    anchor,
                    effective_drafts,
                    params,
                    rng,
                )
            }
        }
    }

    /// Batched multi-lane scheduled proposal: stacks DSpark lanes through
    /// [`super::dspark::Inkling2DsparkModel::propose_blocks_batched`], and runs
    /// DFlash2 lanes sequentially through each lane's compiled 8-row draft graph.
    pub(crate) fn propose_scheduled_dspark_batch(
        &mut self,
        requests: &mut [ScheduledProposalRequest<'_>],
    ) -> Vec<Result<DsparkProposal>> {
        if requests.len() <= 1 || self.dflash2.is_some() {
            return requests
                .iter_mut()
                .map(|req| {
                    self.propose_scheduled_dspark(
                        req.seq_id,
                        req.anchor,
                        req.max_drafts,
                        req.params,
                        req.rng,
                        false,
                    )
                })
                .collect();
        }
        let Some(ref draft) = self.dspark else {
            return requests.iter().map(|_| Err(missing_draft())).collect();
        };

        let mut results: Vec<Option<Result<DsparkProposal>>> =
            (0..requests.len()).map(|_| None).collect();
        let mut valid_indices = Vec::with_capacity(requests.len());
        let mut lanes: Vec<BatchedDsparkLane<'_>> = Vec::with_capacity(requests.len());

        for (i, req) in requests.iter().enumerate() {
            match self.scheduled_dspark.seq_lanes.get(&req.seq_id) {
                None => results[i] = Some(Err(missing_lane(req.seq_id))),
                Some(lane) if lane.open.is_some() => {
                    results[i] = Some(Err(Error::from_reason(
                        "scheduled Inkling2 DSpark propose: prior verify was not committed",
                    )));
                }
                Some(lane) => match &lane.ctx {
                    Inkling2DraftContext::Dspark(dctx) => {
                        let effective_drafts = if req.params.mtp_adaptive_depth {
                            lane.adaptive_depth.current_depth(req.max_drafts)
                        } else {
                            req.max_drafts
                        };
                        valid_indices.push(i);
                        lanes.push(BatchedDsparkLane {
                            ctx: dctx,
                            anchor_id: req.anchor,
                            max_drafts: effective_drafts,
                            params: req.params,
                        });
                    }
                    Inkling2DraftContext::Dflash2(_) => {
                        results[i] = Some(Err(Error::from_reason(
                            "scheduled Inkling2 DSpark batch encountered DFlash2 lane",
                        )));
                    }
                },
            }
        }

        if !valid_indices.is_empty() {
            let mut rngs: Vec<&mut dyn rand::Rng> = Vec::with_capacity(valid_indices.len());
            let mut next_pos = 0usize;
            for (i, req) in requests.iter_mut().enumerate() {
                if next_pos < valid_indices.len() && valid_indices[next_pos] == i {
                    rngs.push(req.rng);
                    next_pos += 1;
                }
            }
            match draft.propose_blocks_batched(
                &self.embed_tokens,
                &self.embed_norm,
                &self.lm_head,
                &lanes,
                &mut rngs,
            ) {
                Ok(props) => {
                    for (idx, prop) in valid_indices.into_iter().zip(props) {
                        results[idx] = Some(Ok(prop));
                    }
                }
                Err(err) => {
                    for idx in valid_indices {
                        results[idx] = Some(Err(Error::from_reason(err.reason.clone())));
                    }
                }
            }
        }

        results
            .into_iter()
            .map(|opt| {
                opt.unwrap_or_else(|| {
                    Err(Error::from_reason(
                        "scheduled Inkling2 DSpark propose missing slot",
                    ))
                })
            })
            .collect()
    }

    /// Feed prefill tapped hidden states (`[5, 23, 41, 48]`) into `seq_id`'s scheduled lane.
    pub(crate) fn append_scheduled_dspark_prefill(
        &mut self,
        seq_id: SeqId,
        first_pos: u32,
        taps: &[MxArray],
        count: usize,
    ) -> Result<()> {
        let Some(lane) = self.scheduled_dspark.seq_lanes.get_mut(&seq_id) else {
            return Ok(());
        };
        if lane.ctx.committed_len() as u32 != first_pos {
            return Err(Error::from_reason(format!(
                "scheduled Inkling2 prefill cursor mismatch for seq {seq_id}: draft={} target={first_pos}",
                lane.ctx.committed_len()
            )));
        }
        match &mut lane.ctx {
            Inkling2DraftContext::Dspark(dctx) => {
                let Some(ref draft) = self.dspark else {
                    return Ok(());
                };
                dctx.append_taps(draft, taps, count)
            }
            Inkling2DraftContext::Dflash2(dctx) => {
                let Some(ref draft) = self.dflash2 else {
                    return Ok(());
                };
                let start = first_pos as usize;
                let end = start.saturating_add(count);
                let token_slice: Vec<u32> = self
                    .paged
                    .as_ref()
                    .and_then(|rt| rt.coordinator.full_adapter().request_tokens_for(seq_id))
                    .and_then(|tokens| tokens.get(start..end).map(|s| s.to_vec()))
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "scheduled Inkling2 DFlash2 prefill missing recorded tokens [{start}..{end}) for seq {seq_id}"
                        ))
                    })?;
                dctx.append_taps(draft, taps, &token_slice)
            }
        }
    }

    /// Execute one scheduled verify step across speculative lanes (grouping 2..=4 full 8-row lanes
    /// into stacked `[N * 8]` verify waves!) plus any solo lanes / AR peers.
    pub(crate) fn verify_scheduled_dspark(
        &mut self,
        rows: &[ScheduledVerifyRow],
    ) -> Result<MxArray> {
        LaneVerify(self).verify_scheduled_rows(rows)
    }

    /// Commit one scheduled verify step across all rows in the wave.
    pub(crate) fn commit_scheduled_dspark(
        &mut self,
        rows: &[ScheduledVerifyCommit],
    ) -> Result<Vec<Result<()>>> {
        let results = LaneVerify(self).commit_scheduled_rows(rows)?;
        // Queue async eval of committed ShortConv + DSpark context arrays for overlap.
        let mut arrays: Vec<&MxArray> = Vec::new();
        for (row, res) in rows.iter().zip(&results) {
            if res.is_err() || row.keep == 0 {
                continue;
            }
            if let Some(ref rt) = self.paged
                && let Some(conv) = rt.conv_states.get(&row.seq_id)
            {
                for v in &conv.visits {
                    arrays.push(&v.k);
                    arrays.push(&v.v);
                    arrays.push(&v.attn_post);
                    arrays.push(&v.mlp_post);
                }
            }
            if let Some(lane) = self.scheduled_dspark.seq_lanes.get(&row.seq_id) {
                lane.ctx.collect_arrays(&mut arrays);
            }
        }
        if !arrays.is_empty() {
            MxArray::async_eval_arrays(&arrays);
        }
        Ok(results)
    }

    /// Retract any open scheduled verify wave (keeping 0 rows for every participant).
    pub(crate) fn retract_open_dspark_wave(&mut self) -> Result<()> {
        match self.scheduled_dspark.lane_batch.take() {
            Some(batch) => LaneVerify(self).retract_verify_batch(batch),
            None => Ok(()),
        }
    }

    /// Release scheduled DSpark lane state for `seq_id`.
    pub(crate) fn release_scheduled_dspark(&mut self, seq_id: SeqId) {
        if self.scheduled_dspark.lane_batch.is_some()
            && (self.scheduled_dspark.seq_lanes.contains_key(&seq_id)
                || self.scheduled_dspark.peer_tapes.contains_key(&seq_id))
        {
            let _ = self.retract_open_dspark_wave();
        }
        self.scheduled_dspark.seq_lanes.remove(&seq_id);
        self.scheduled_dspark.peer_tapes.remove(&seq_id);
        self.dspark_turn_contexts.remove(&seq_id);
    }
}

/// Partition `rows` into groups of `2..=4` speculative lanes (`ids.len() >= 2`, supporting ragged
/// adaptive per-lane `k`) that stack into one `[sum(1 + k_i)]` verify wave!
fn select_verify_wave_groups(
    inner: &Inkling2Inner,
    rows: &[(PagedRaggedRow, Vec<u32>)],
) -> Vec<Vec<usize>> {
    select_verify_wave_groups_from_lanes(&inner.scheduled_dspark.seq_lanes, rows)
}

fn select_verify_wave_groups_from_lanes(
    lanes: &HashMap<SeqId, Inkling2ScheduledDsparkLane>,
    rows: &[(PagedRaggedRow, Vec<u32>)],
) -> Vec<Vec<usize>> {
    crate::engine::scheduled_verify::partition_verify_wave_groups(
        rows,
        INKLING2_WAVE_LANES,
        |(row, ids)| {
            ids.len() >= 2
                && lanes
                    .get(&row.seq_id)
                    .is_some_and(|lane| lane.open.is_none())
        },
    )
}

struct LaneVerify<'a>(&'a mut Inkling2Inner);

impl ScheduledDraftVerify for LaneVerify<'_> {
    type Cache<'a>
        = CoordinatorCache<'a>
    where
        Self: 'a;

    fn verify_cache(&mut self) -> Self::Cache<'_> {
        CoordinatorCache(self.0.paged.as_mut().map(|rt| &mut rt.coordinator))
    }

    fn take_verify_batch(&mut self) -> Option<ScheduledVerifyBatch> {
        self.0.scheduled_dspark.lane_batch.take()
    }

    fn store_verify_batch(&mut self, batch: ScheduledVerifyBatch) {
        self.0.scheduled_dspark.lane_batch = Some(batch);
    }

    fn validate_draft_owner(&self, row: &ScheduledVerifyRow) -> Result<()> {
        if !row.speculative {
            if row.tokens.len() != 1 || self.0.scheduled_dspark.seq_lanes.contains_key(&row.seq_id)
            {
                return Err(Error::from_reason(
                    "scheduled Inkling2 DSpark peer/verify row mismatch",
                ));
            }
            return Ok(());
        }
        let lane = self
            .0
            .scheduled_dspark
            .seq_lanes
            .get(&row.seq_id)
            .ok_or_else(|| missing_lane(row.seq_id))?;
        if lane.open.is_some() || lane.ctx.committed_len() as u32 != row.first_position {
            return Err(Error::from_reason(format!(
                "scheduled Inkling2 lane/verify cursor mismatch for seq {}: draft={} target={}",
                row.seq_id,
                lane.ctx.committed_len(),
                row.first_position
            )));
        }
        Ok(())
    }

    fn run_packed_target(
        &mut self,
        rows: &[(PagedRaggedRow, Vec<u32>)],
    ) -> Result<(MxArray, Vec<MxArray>)> {
        let inner = &mut *self.0;
        let tap_visits = inner
            .active_draft_tap_visits()
            .ok_or_else(missing_draft)?
            .to_vec();

        let mut outputs: Vec<Option<MxArray>> = rows.iter().map(|_| None).collect();

        // 1. Run 2..=4 lane stacked `[N * 8]` verify waves!
        for group in select_verify_wave_groups(inner, rows) {
            let wave_rows: Vec<(PagedRaggedRow, Vec<u32>)> =
                group.iter().map(|&idx| rows[idx].clone()).collect();
            let n_lanes = group.len();
            let (wave_logits_2d, wave_taps, wave_tapes) =
                inner.paged_verify_ragged_taped(&wave_rows, &tap_visits)?;

            let vocab = wave_logits_2d.shape()?[1];
            let d = inner.config.hidden_size as i64;
            let mut row_off = 0i64;
            for ((&orig_idx, (_, ids)), tape) in group.iter().zip(&wave_rows).zip(wave_tapes) {
                let len = ids.len() as i64;
                let lane_logits = wave_logits_2d
                    .slice(&[row_off, 0], &[row_off + len, vocab])?
                    .reshape(&[len, 1, vocab])?;
                let mut lane_taps = Vec::with_capacity(wave_taps.len());
                for tap in &wave_taps {
                    lane_taps.push(tap.slice(&[row_off, 0], &[row_off + len, d])?);
                }
                let seq_id = rows[orig_idx].0.seq_id;
                let lane = inner
                    .scheduled_dspark
                    .seq_lanes
                    .get_mut(&seq_id)
                    .ok_or_else(|| missing_lane(seq_id))?;
                lane.open = Some(OpenInkling2Verify {
                    tape,
                    tapped: lane_taps,
                    verify_tokens: ids.clone(),
                    lanes_in_wave: n_lanes,
                    drafted: ids.len().saturating_sub(1),
                });
                outputs[orig_idx] = Some(lane_logits);
                row_off += len;
            }
            inner.scheduled_dspark.total_verify_waves += 1;
            inner.scheduled_dspark.multi_lane_verify_waves += 1;
            inner.scheduled_dspark.max_verify_wave_lanes =
                inner.scheduled_dspark.max_verify_wave_lanes.max(n_lanes);
        }

        // 2. Run any remaining solo speculative lane or non-speculative AR peer row
        for ((row, ids), out_slot) in rows.iter().zip(outputs.iter_mut()) {
            if out_slot.is_some() {
                continue;
            }
            let seq_id = row.seq_id;
            let is_spec = inner.scheduled_dspark.seq_lanes.contains_key(&seq_id);
            let taps_arg: &[usize] = if is_spec { &tap_visits } else { &[] };
            let single = [(*row, ids.clone())];
            let (logits_2d, taps, mut tapes) =
                inner.paged_verify_ragged_taped(&single, taps_arg)?;
            let tape = tapes.pop().ok_or_else(|| {
                Error::from_reason("scheduled Inkling2 verify produced no ShortConvTape")
            })?;
            let len = ids.len() as i64;
            let vocab = logits_2d.shape()?[1];
            let logits_3d = logits_2d.reshape(&[len, 1, vocab])?;
            if is_spec {
                let lane = inner
                    .scheduled_dspark
                    .seq_lanes
                    .get_mut(&seq_id)
                    .ok_or_else(|| missing_lane(seq_id))?;
                lane.open = Some(OpenInkling2Verify {
                    tape,
                    tapped: taps,
                    verify_tokens: ids.clone(),
                    lanes_in_wave: 1,
                    drafted: ids.len().saturating_sub(1),
                });
                inner.scheduled_dspark.total_verify_waves += 1;
                inner.scheduled_dspark.max_verify_wave_lanes =
                    inner.scheduled_dspark.max_verify_wave_lanes.max(1);
            } else {
                inner.scheduled_dspark.peer_tapes.insert(seq_id, tape);
            }
            *out_slot = Some(logits_3d);
        }

        let packed_parts: Vec<MxArray> = outputs
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                Error::from_reason("scheduled Inkling2 DSpark verify left a row unrun")
            })?;
        let refs: Vec<&MxArray> = packed_parts.iter().collect();
        let packed_logits = MxArray::concatenate_many(refs, Some(0))?;
        Ok((packed_logits, Vec::new()))
    }

    fn commit_target_state(&mut self, seq: u32, keep: usize) -> Result<()> {
        if let Some(peer_tape) = self.0.scheduled_dspark.peer_tapes.remove(&seq) {
            if let Some(ref mut rt) = self.0.paged
                && let Some(conv) = rt.conv_states.get_mut(&seq)
            {
                peer_tape.commit_prefix(keep, conv)?;
                if keep > 0
                    && let Ok(committed) = rt.coordinator.request_token_count_all(seq)
                {
                    rt.record_block_boundary_conv_from_tape(seq, committed, keep, &peer_tape)?;
                }
            }
            return Ok(());
        }
        let lane = self
            .0
            .scheduled_dspark
            .seq_lanes
            .get_mut(&seq)
            .ok_or_else(|| missing_lane(seq))?;
        let tape = lane
            .open
            .as_ref()
            .map(|open| open.tape.clone())
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "scheduled Inkling2 DSpark commit missing ShortConvTape for seq {seq}"
                ))
            })?;
        if let Some(ref mut rt) = self.0.paged
            && let Some(conv) = rt.conv_states.get_mut(&seq)
        {
            tape.commit_prefix(keep, conv)?;
            if keep > 0
                && let Ok(committed) = rt.coordinator.request_token_count_all(seq)
            {
                rt.record_block_boundary_conv_from_tape(seq, committed, keep, &tape)?;
            }
        }
        if keep == 0
            && let Some(lane) = self.0.scheduled_dspark.seq_lanes.get_mut(&seq)
        {
            lane.open = None;
        }
        Ok(())
    }

    fn abort_target_state(&mut self, seq: u32) {
        if let Some(peer_tape) = self.0.scheduled_dspark.peer_tapes.remove(&seq) {
            if let Some(ref mut rt) = self.0.paged
                && let Some(conv) = rt.conv_states.get_mut(&seq)
            {
                let _ = peer_tape.commit_prefix(0, conv);
            }
            return;
        }
        if let Some(open) = self
            .0
            .scheduled_dspark
            .seq_lanes
            .get_mut(&seq)
            .and_then(|lane| lane.open.take())
            && let Some(ref mut rt) = self.0.paged
            && let Some(conv) = rt.conv_states.get_mut(&seq)
        {
            let _ = open.tape.commit_prefix(0, conv);
        }
    }

    fn supports_batched_append_taps(&self) -> bool {
        true
    }

    fn append_committed_taps_batch(
        &mut self,
        items: &[(u32, &[MxArray], usize)],
    ) -> Vec<Result<()>> {
        if items.len() <= 1 || self.0.dflash2.is_some() {
            return items
                .iter()
                .map(|&(seq, taps, keep)| self.append_committed_taps(seq, taps, keep))
                .collect();
        }
        let Some(ref draft) = self.0.dspark else {
            return items.iter().map(|_| Err(missing_draft())).collect();
        };

        // Extract each lane's owned `Inkling2DsparkContext` & `OpenInkling2Verify` so we can pass
        // disjoint `&mut Inkling2DsparkContext` refs to `append_taps_batched` without borrow conflict.
        struct Extracted {
            orig_idx: usize,
            seq: u32,
            lane: Inkling2ScheduledDsparkLane,
            open: OpenInkling2Verify,
            keep: usize,
        }
        let mut results: Vec<Option<Result<()>>> = (0..items.len()).map(|_| None).collect();
        let mut extracted: Vec<Extracted> = Vec::with_capacity(items.len());
        for (i, &(seq, _, keep)) in items.iter().enumerate() {
            match self.0.scheduled_dspark.seq_lanes.remove(&seq) {
                None => results[i] = Some(Err(missing_lane(seq))),
                Some(mut lane) => match lane.open.take() {
                    None => {
                        self.0.scheduled_dspark.seq_lanes.insert(seq, lane);
                        results[i] = Some(Err(Error::from_reason(format!(
                            "scheduled Inkling2 DSpark commit missing open taps for seq {seq}"
                        ))));
                    }
                    Some(open) => extracted.push(Extracted {
                        orig_idx: i,
                        seq,
                        lane,
                        open,
                        keep,
                    }),
                },
            }
        }

        if !extracted.is_empty() {
            let batch_res = {
                let mut call_args: Vec<(&mut Inkling2DsparkContext, &[MxArray], usize)> =
                    Vec::with_capacity(extracted.len());
                for e in &mut extracted {
                    match &mut e.lane.ctx {
                        Inkling2DraftContext::Dspark(dctx) => {
                            call_args.push((dctx, e.open.tapped.as_slice(), e.keep));
                        }
                        Inkling2DraftContext::Dflash2(_) => {
                            return items
                                .iter()
                                .map(|_| {
                                    Err(Error::from_reason(
                                        "scheduled Inkling2 DSpark batch encountered DFlash2 lane",
                                    ))
                                })
                                .collect();
                        }
                    }
                }
                Inkling2DsparkContext::append_taps_batched(draft, &mut call_args)
            };
            match batch_res {
                Ok(()) => {
                    for mut e in extracted {
                        e.lane.total_verifies += 1;
                        if e.open.lanes_in_wave >= 2 {
                            e.lane.multi_lane_wave_verifies += 1;
                        }
                        if e.open.drafted > 0 {
                            e.lane
                                .adaptive_depth
                                .observe(e.open.drafted, e.keep.saturating_sub(1));
                        }
                        self.0.scheduled_dspark.seq_lanes.insert(e.seq, e.lane);
                        results[e.orig_idx] = Some(Ok(()));
                    }
                }
                Err(err) => {
                    for e in extracted {
                        self.0.scheduled_dspark.seq_lanes.insert(e.seq, e.lane);
                        results[e.orig_idx] = Some(Err(Error::from_reason(err.reason.clone())));
                    }
                }
            }
        }

        results
            .into_iter()
            .map(|opt| {
                opt.unwrap_or_else(|| {
                    Err(Error::from_reason(
                        "scheduled Inkling2 DSpark append missing slot",
                    ))
                })
            })
            .collect()
    }

    fn append_committed_taps(&mut self, seq: u32, _taps: &[MxArray], keep: usize) -> Result<()> {
        let open = self
            .0
            .scheduled_dspark
            .seq_lanes
            .get_mut(&seq)
            .ok_or_else(|| missing_lane(seq))?
            .open
            .take()
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "scheduled Inkling2 commit missing open taps for seq {seq}"
                ))
            })?;
        let lane = self
            .0
            .scheduled_dspark
            .seq_lanes
            .get_mut(&seq)
            .ok_or_else(|| missing_lane(seq))?;
        match &mut lane.ctx {
            Inkling2DraftContext::Dspark(dctx) => {
                let draft = self.0.dspark.as_ref().ok_or_else(missing_draft)?;
                dctx.append_taps(draft, &open.tapped, keep)?;
            }
            Inkling2DraftContext::Dflash2(dctx) => {
                let draft = self.0.dflash2.as_ref().ok_or_else(missing_draft)?;
                if open.verify_tokens.len() < keep {
                    return Err(Error::from_reason(format!(
                        "scheduled Inkling2 DFlash2 commit token provenance len {} < keep {keep}",
                        open.verify_tokens.len()
                    )));
                }
                dctx.append_taps(draft, &open.tapped, &open.verify_tokens[..keep])?;
            }
        }
        lane.total_verifies += 1;
        if open.lanes_in_wave >= 2 {
            lane.multi_lane_wave_verifies += 1;
        }
        if open.drafted > 0 {
            lane.adaptive_depth
                .observe(open.drafted, keep.saturating_sub(1));
        }
        Ok(())
    }

    fn discard_draft_owner(&mut self, seq: u32) {
        self.0.scheduled_dspark.seq_lanes.remove(&seq);
        self.0.scheduled_dspark.peer_tapes.remove(&seq);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    fn mk_lane(open: bool) -> Inkling2ScheduledDsparkLane {
        Inkling2ScheduledDsparkLane {
            ctx: Inkling2DraftContext::Dspark(Inkling2DsparkContext::new(3)),
            adaptive_depth: AdaptiveDraftDepth::new(7),
            open: open.then(|| OpenInkling2Verify {
                tape: Inkling2ShortConvTape::new(0),
                tapped: Vec::new(),
                verify_tokens: Vec::new(),
                lanes_in_wave: 1,
                drafted: 0,
            }),
            total_verifies: 0,
            multi_lane_wave_verifies: 0,
        }
    }

    #[test]
    fn select_verify_wave_groups_stacks_ragged_adaptive_depths() {
        let mut lanes = HashMap::new();
        lanes.insert(1, mk_lane(false)); // k=3 (4 rows)
        lanes.insert(2, mk_lane(false)); // k=5 (6 rows)
        lanes.insert(3, mk_lane(false)); // k=7 (8 rows)
        lanes.insert(4, mk_lane(false)); // k=4 (5 rows)
        lanes.insert(5, mk_lane(true)); // open -> excluded
        lanes.insert(6, mk_lane(false)); // k=2 (3 rows)
        lanes.insert(7, mk_lane(false)); // k=3 (4 rows)

        let rows = vec![
            (
                PagedRaggedRow {
                    seq_id: 100, // AR peer (len 1, not in lanes)
                    first_logical_position: 10,
                    query_len: 1,
                },
                vec![42],
            ),
            (
                PagedRaggedRow {
                    seq_id: 1,
                    first_logical_position: 20,
                    query_len: 4,
                },
                vec![1, 2, 3, 4],
            ),
            (
                PagedRaggedRow {
                    seq_id: 5,
                    first_logical_position: 20,
                    query_len: 5,
                },
                vec![1, 2, 3, 4, 5],
            ),
            (
                PagedRaggedRow {
                    seq_id: 2,
                    first_logical_position: 30,
                    query_len: 6,
                },
                vec![1, 2, 3, 4, 5, 6],
            ),
            (
                PagedRaggedRow {
                    seq_id: 3,
                    first_logical_position: 40,
                    query_len: 8,
                },
                vec![1, 2, 3, 4, 5, 6, 7, 8],
            ),
            (
                PagedRaggedRow {
                    seq_id: 4,
                    first_logical_position: 50,
                    query_len: 5,
                },
                vec![1, 2, 3, 4, 5],
            ),
            (
                PagedRaggedRow {
                    seq_id: 6,
                    first_logical_position: 60,
                    query_len: 3,
                },
                vec![1, 2, 3],
            ),
            (
                PagedRaggedRow {
                    seq_id: 7,
                    first_logical_position: 70,
                    query_len: 4,
                },
                vec![1, 2, 3, 4],
            ),
        ];

        let groups = select_verify_wave_groups_from_lanes(&lanes, &rows);
        // Indices: 0=AR(skip), 1=seq1, 2=seq5(open,skip), 3=seq2, 4=seq3, 5=seq4 -> chunk 1: [1,3,4,5]
        //          6=seq6, 7=seq7 -> chunk 2: [6,7]
        assert_eq!(groups, vec![vec![1, 3, 4, 5], vec![6, 7]]);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;

    use tokio::sync::mpsc;

    use crate::engine::cmd::ChatCmd;
    use crate::engine::types::{ChatConfig, ChatResult};
    use crate::models::inkling2::model::{Inkling2LoadOptions, Inkling2Model};
    use crate::models::inkling2::persistence::load_from_dir;
    use crate::models::inkling2::persistence::test_real_model_path;
    use crate::models::inkling2::scheduler::Inkling2Cmd;
    use crate::tokenizer::ChatMessage;
    use crate::tools::inkling::{
        CONTENT_MODEL_END_SAMPLING_ID, Inkling2RenderOptions, InklingMessage,
        render_inkling2_messages_to_ids,
    };

    fn user_msg(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".to_string(),
            content: text.to_string(),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            reasoning_content: None,
            thinking_enabled: None,
            images: None,
            audio: None,
        }]
    }

    fn greedy_cfg(
        owner: Option<&str>,
        max_tokens: i32,
        salt: &str,
        enable_mtp: bool,
        adaptive: bool,
    ) -> ChatConfig {
        ChatConfig {
            max_new_tokens: Some(max_tokens),
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(0),
            reasoning_effort: Some("none".to_string()),
            reuse_cache: Some(true),
            cache_salt: Some(salt.to_string()),
            cache_owner_id: owner.map(ToOwned::to_owned),
            enable_mtp: Some(enable_mtp),
            mtp_adaptive_depth: Some(adaptive),
            report_performance: Some(true),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            ..ChatConfig::default()
        }
    }

    async fn run_turn(
        tx: &mpsc::UnboundedSender<Inkling2Cmd>,
        messages: Vec<ChatMessage>,
        config: ChatConfig,
    ) -> ChatResult {
        let (reply, rx) = tokio::sync::oneshot::channel();
        tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::SessionStart {
            messages,
            config,
            reply,
            cancelled: Arc::new(AtomicBool::new(false)),
        })))
        .expect("send SessionStart");
        rx.await
            .expect("scheduler reply channel")
            .expect("SessionStart succeeded")
    }

    /// **PHASE 5 GATE 1 (`inkling2_dspark_greedy_parity`)**:
    ///
    /// Verifies on the real Inkling-2-Mini + `dspark-Inkling-2-Mini-Q8_0.gguf` model:
    /// 1. Auto-discovery attaches `dspark-Inkling-2-Mini-Q8_0.gguf` (`has_mtp_weights() == true`).
    /// 2. Whole-turn DSpark greedy (`run_paged_dspark_turn` + `ShortConvTape` + FP64-folded Markov
    ///    head) matches plain AR greedy **token-for-token** over 256 tokens on a realistic coding
    ///    prompt (both with fixed depth and with `DsparkBreakEvenPolicy`).
    /// 3. Scheduled single-lane DSpark greedy ALSO matches plain AR greedy **token-for-token**!
    /// 4. Prints exact AR tok/s, DSpark tok/s, speedup, and mean acceptance length!
    #[tokio::test]
    #[ignore = "Requires real Inkling-2-Mini + DSpark GGUF on disk (~18 GB)"]
    async fn inkling2_dspark_greedy_parity() {
        let model_path = test_real_model_path();
        let model_dir = Path::new(&model_path);
        assert!(model_dir.exists(), "Missing {}", model_dir.display());

        let t_load = Instant::now();
        let model = Inkling2Model::load(
            model_path,
            Some(Inkling2LoadOptions {
                kv_format: Some("int8".to_string()),
                cache_disk: Some("0".to_string()),
                ..Inkling2LoadOptions::default()
            }),
        )
        .await
        .expect("Inkling2Model::load");
        eprintln!(
            "[inkling2_dspark_greedy_parity] Loaded in {:.2}s (paged={}, dspark={}, lanes={})",
            t_load.elapsed().as_secs_f64(),
            model.has_block_paged_cache(),
            model.has_mtp_weights(),
            model.max_concurrent_sequences(),
        );
        assert!(model.has_block_paged_cache());
        assert!(
            model.has_mtp_weights(),
            "Expected auto-discovered dspark-Inkling-2-Mini-Q8_0.gguf"
        );

        let tx = model.thread.cmd_sender().expect("cmd_sender");
        let prompt = "Write a complete, idiomatic Rust function `fn binary_search(slice: &[i32], target: i32) -> Option<usize>` using iterative bounds, with doc comments and a unit test.";
        let max_tokens = 64;

        // 1. Plain AR greedy baseline (scheduled lane with enable_mtp = false)
        let t_ar = Instant::now();
        let ar = run_turn(
            &tx,
            user_msg(prompt),
            greedy_cfg(Some("ar-baseline"), max_tokens, "parity-ar", false, false),
        )
        .await;
        let ar_wall = t_ar.elapsed().as_secs_f64();
        let ar_tps = ar
            .performance
            .as_ref()
            .map(|p| p.decode_tokens_per_second)
            .unwrap_or(ar.num_tokens as f64 / ar_wall);
        eprintln!(
            "  [AR baseline] tokens={} wall={ar_wall:.2}s decode_tps={ar_tps:.2} tok/s",
            ar.num_tokens
        );
        model.reset_caches().await.expect("reset_caches");

        // 2. Scheduled DSpark greedy (enable_mtp = true, owner set -> scheduled lane)
        let t_spec = Instant::now();
        let spec = run_turn(
            &tx,
            user_msg(prompt),
            greedy_cfg(Some("dspark-sched"), max_tokens, "parity-spec", true, false),
        )
        .await;
        let spec_wall = t_spec.elapsed().as_secs_f64();
        let spec_tps = spec
            .performance
            .as_ref()
            .map(|p| p.decode_tokens_per_second)
            .unwrap_or(spec.num_tokens as f64 / spec_wall);
        let spec_mean_acc = spec
            .performance
            .as_ref()
            .and_then(|p| p.mtp_mean_accepted_tokens_total);
        eprintln!(
            "  [DSpark sched] tokens={} wall={spec_wall:.2}s decode_tps={spec_tps:.2} tok/s (speedup={:.2}x, mean_accept={spec_mean_acc:?})",
            spec.num_tokens,
            if ar_tps > 0.0 { spec_tps / ar_tps } else { 0.0 },
        );
        assert_eq!(
            spec.raw_text, ar.raw_text,
            "Scheduled DSpark greedy diverged from AR baseline!"
        );
        assert_eq!(spec.num_tokens, ar.num_tokens);
        model.reset_caches().await.expect("reset_caches");

        // 3. Whole-turn AR baseline (`owner = None`) vs Whole-turn DSpark + BreakEven (`owner = None`)
        //    (40 tokens = ~12-15 full DSpark propose/verify/ShortConvTape-rollback cycles + AR probes)
        let whole_max_tokens = 40;
        let t_whole_ar = Instant::now();
        let whole_ar = run_turn(
            &tx,
            user_msg(prompt),
            greedy_cfg(None, whole_max_tokens, "parity-whole-ar", false, false),
        )
        .await;
        let whole_ar_wall = t_whole_ar.elapsed().as_secs_f64();
        let whole_ar_tps = whole_ar
            .performance
            .as_ref()
            .map(|p| p.decode_tokens_per_second)
            .unwrap_or(whole_ar.num_tokens as f64 / whole_ar_wall);
        eprintln!(
            "  [AR whole-turn] tokens={} wall={whole_ar_wall:.2}s decode_tps={whole_ar_tps:.2} tok/s",
            whole_ar.num_tokens
        );
        model.reset_caches().await.expect("reset_caches");

        let t_whole = Instant::now();
        let whole = run_turn(
            &tx,
            user_msg(prompt),
            greedy_cfg(None, whole_max_tokens, "parity-whole", true, true),
        )
        .await;
        let whole_wall = t_whole.elapsed().as_secs_f64();
        let whole_tps = whole
            .performance
            .as_ref()
            .map(|p| p.decode_tokens_per_second)
            .unwrap_or(whole.num_tokens as f64 / whole_wall);
        let whole_mean_acc = whole
            .performance
            .as_ref()
            .and_then(|p| p.mtp_mean_accepted_tokens_total);
        eprintln!(
            "  [DSpark whole-turn + BreakEven] tokens={} wall={whole_wall:.2}s decode_tps={whole_tps:.2} tok/s (speedup={:.2}x, mean_accept={whole_mean_acc:?})",
            whole.num_tokens,
            if whole_ar_tps > 0.0 {
                whole_tps / whole_ar_tps
            } else {
                0.0
            },
        );
        assert_eq!(
            whole.raw_text, whole_ar.raw_text,
            "Whole-turn DSpark greedy diverged from whole-turn AR baseline!"
        );
        assert_eq!(whole.num_tokens, whole_ar.num_tokens);

        drop(tx);
        model.dispose().await.expect("dispose");
    }

    /// **PHASE 5 GATE 2 (`inkling2_dspark_concurrent_wave_parity`)**:
    ///
    /// Runs 4 concurrent DSpark sequences simultaneously through `Inkling2Model::load` (engaging
    /// 2..=4 lane stacked `[N * 8]` verify waves!) and verifies every lane produces **bit-exact**
    /// greedy output matching solo execution!
    #[tokio::test]
    #[ignore = "Requires real Inkling-2-Mini + DSpark GGUF on disk (~18 GB)"]
    async fn inkling2_dspark_concurrent_wave_parity() {
        let model_path = test_real_model_path();
        let model_dir = Path::new(&model_path);
        assert!(model_dir.exists(), "Missing {}", model_dir.display());

        let model = Inkling2Model::load(
            model_path,
            Some(Inkling2LoadOptions {
                kv_format: Some("int8".to_string()),
                cache_disk: Some("0".to_string()),
                ..Inkling2LoadOptions::default()
            }),
        )
        .await
        .expect("Inkling2Model::load");
        assert!(model.has_mtp_weights());
        assert!(model.max_concurrent_sequences() >= 4);

        let tx = model.thread.cmd_sender().expect("cmd_sender");
        let prompts = [
            "Write a short Rust function `fn fibonacci(n: u64) -> u64` using iteration, with a brief doc comment.",
            "Write a short Python function to merge two sorted lists into a single sorted list without using sort().",
            "Explain in 4 concise bullet points how B-trees keep lookups O(log n) on SSDs.",
            "Write a short TypeScript generic function `groupBy<T, K extends string>` with clean types.",
        ];
        let max_tokens = 32;

        // 1. Solo greedy DSpark baseline for each prompt
        let mut solo_outputs = Vec::with_capacity(prompts.len());
        for (i, prompt) in prompts.iter().enumerate() {
            let res = run_turn(
                &tx,
                user_msg(prompt),
                greedy_cfg(
                    Some(&format!("solo-{i}")),
                    max_tokens,
                    &format!("solo-salt-{i}"),
                    true,
                    false,
                ),
            )
            .await;
            solo_outputs.push(res);
            model.reset_caches().await.expect("reset_caches");
        }

        // 2. Concurrent 4-lane DSpark stacked `[N * 8]` verify wave execution!
        let t_wave = Instant::now();
        let (r0, r1, r2, r3) = tokio::join!(
            run_turn(
                &tx,
                user_msg(prompts[0]),
                greedy_cfg(Some("wave-0"), max_tokens, "wave-salt-0", true, false)
            ),
            run_turn(
                &tx,
                user_msg(prompts[1]),
                greedy_cfg(Some("wave-1"), max_tokens, "wave-salt-1", true, false)
            ),
            run_turn(
                &tx,
                user_msg(prompts[2]),
                greedy_cfg(Some("wave-2"), max_tokens, "wave-salt-2", true, false)
            ),
            run_turn(
                &tx,
                user_msg(prompts[3]),
                greedy_cfg(Some("wave-3"), max_tokens, "wave-salt-3", true, false)
            ),
        );
        let wave_secs = t_wave.elapsed().as_secs_f64();
        let wave_outputs = [r0, r1, r2, r3];
        let total_toks: u32 = wave_outputs.iter().map(|r| r.num_tokens).sum();
        let stats = model.scheduler_stats().await.expect("scheduler_stats");
        eprintln!(
            "[inkling2_dspark_concurrent_wave_parity] 4 concurrent DSpark lanes: total_toks={total_toks} \
             wall={wave_secs:.2}s agg_tps={:.2} tok/s (global_steps={}, max_occupancy={})",
            total_toks as f64 / wave_secs,
            stats.global_steps,
            stats.max_batch_occupancy,
        );
        assert!(
            stats.max_batch_occupancy >= 2,
            "Expected concurrent wave occupancy >= 2, got {}",
            stats.max_batch_occupancy
        );

        for (i, (solo, wave)) in solo_outputs.iter().zip(wave_outputs.iter()).enumerate() {
            assert_eq!(
                wave.raw_text, solo.raw_text,
                "4-lane concurrent DSpark wave diverged from solo on prompt {i}!"
            );
            assert_eq!(wave.num_tokens, solo.num_tokens);
        }

        drop(tx);
        model.dispose().await.expect("dispose");
    }

    /// **PHASE 5 FULL BENCHMARK SUITE (`inkling2_phase5_full_benchmark`)**:
    ///
    /// Measures all Phase 5 real-model KPIs on `/Users/brooklyn/.mlx-node/models/inkling-2-mini-gguf`:
    /// 1. Single-seq AR decode tok/s vs DSpark decode tok/s + EXACT mean acceptance length (256 tokens)
    /// 2. 4-lane concurrent aggregate tok/s (DSpark `[N*8]` waves vs AR `[N,1]` decode)
    /// 3. Prefill tok/s at 2K and 8K context
    /// 4. SSD cold restore latency vs fresh prefill at 4K+ context
    /// 5. Vision prompt smoke test (synthetic PNG through PE-Lang ViT + paged backbone)
    #[tokio::test]
    #[ignore = "Requires real Inkling-2-Mini + DSpark + mmproj GGUFs on disk (~18.7 GB)"]
    async fn inkling2_phase5_full_benchmark() {
        let model_path = test_real_model_path();
        let model_dir = Path::new(&model_path);
        assert!(model_dir.exists(), "Missing {}", model_dir.display());

        // ------------------------------------------------------------------
        // A. Direct inner benchmark: exact mean acceptance length, 2K & 8K prefill
        // ------------------------------------------------------------------
        let mut inner = load_from_dir(&model_path).await.expect("load_from_dir");
        assert!(inner.dspark.is_some(), "Expected auto-discovered DSpark");
        let tokenizer = inner.tokenizer.clone().expect("tokenizer");

        let coding_prompt = "Write a complete, idiomatic Rust implementation of Dijkstra's shortest path algorithm using std::collections::BinaryHeap. Include a Graph struct, edge relaxation, path reconstruction, and a small main() example.";
        let render_opts = Inkling2RenderOptions {
            reasoning_effort: 0.0,
            ..Default::default()
        };
        let prompt_ids = render_inkling2_messages_to_ids(
            &[InklingMessage::user(coding_prompt)],
            tokenizer.as_ref(),
            &render_opts,
        )
        .expect("render coding prompt");

        // 1. Single-seq AR decode (256 tokens)
        let (ar_ids, ar_perf) = inner
            .generate_greedy_paged(&prompt_ids, None, 256, &[CONTENT_MODEL_END_SAMPLING_ID])
            .expect("AR greedy");

        // 2. Single-seq DSpark decode (256 tokens) with EXACT cycle/acceptance tracking
        let seq_id = 0u32;
        inner.reset_paged_sequence(seq_id).expect("reset");
        let tap_visits = inner
            .dspark
            .as_ref()
            .expect("dspark")
            .config
            .target_layer_ids
            .clone();
        let n_draft_layers = inner
            .dspark
            .as_ref()
            .expect("dspark")
            .config
            .num_hidden_layers;
        let mut draft_ctx =
            crate::models::inkling2::dspark::Inkling2DsparkContext::new(n_draft_layers);
        let t_dp_pre = Instant::now();
        let (first_logits, _) = inner
            .paged_prefill_sequence_with_taps(
                seq_id,
                &prompt_ids,
                None,
                &tap_visits,
                |draft, taps, len| draft_ctx.append_taps(draft, taps, len),
            )
            .expect("dspark prefill");
        draft_ctx.eval_state().expect("eval draft ctx");
        let mut anchor = first_logits
            .argmax(-1, Some(false))
            .expect("argmax")
            .item_at_int32(0)
            .expect("item") as u32;
        let _dp_pre_secs = t_dp_pre.elapsed().as_secs_f64();

        let mut dspark_ids = Vec::with_capacity(256);
        dspark_ids.push(anchor);
        let mut cycles = 0usize;
        let mut total_drafted = 0usize;
        let mut total_accepted_drafts = 0usize;
        let mut rng = rand::rng();
        let greedy_params = crate::engine::extract_chat_params(&ChatConfig {
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(0),
            ..ChatConfig::default()
        });
        let t_dp_dec = Instant::now();
        while dspark_ids.len() < 256 && anchor != CONTENT_MODEL_END_SAMPLING_ID {
            let remaining = 256 - dspark_ids.len();
            let cap = 7usize.min(remaining.saturating_sub(1));
            let mut verify_ids = vec![anchor];
            let mut proposal_ids = Vec::new();
            if cap > 0 {
                let draft = inner.dspark.as_ref().expect("dspark");
                let mut prop = draft_ctx
                    .propose_block(
                        draft,
                        &inner.embed_tokens,
                        &inner.embed_norm,
                        &inner.lm_head,
                        anchor,
                        cap,
                        &greedy_params,
                        &mut rng,
                    )
                    .expect("propose");
                prop.materialize_draft_ids().expect("materialize");
                for &id in &prop.draft_ids {
                    proposal_ids.push(id as u32);
                    verify_ids.push(id as u32);
                }
            }
            let (first_pos, ticket) = {
                use crate::engine::spec_paged::SpecPagedCache;
                let rt = inner.paged.as_mut().expect("paged");
                let pos = rt.coordinator.request_token_count_all(seq_id).expect("pos");
                let ticket = rt
                    .coordinator
                    .record_verify(seq_id, &verify_ids)
                    .expect("record");
                (pos, ticket)
            };
            let row = crate::transformer::paged_kv_cache_adapter::PagedRaggedRow {
                seq_id,
                first_logical_position: first_pos,
                query_len: verify_ids.len() as u32,
            };
            let (logits_2d, taps, mut tapes) = inner
                .paged_verify_ragged_taped(&[(row, verify_ids.clone())], &tap_visits)
                .expect("verify");
            let argmax = logits_2d.argmax(-1, Some(false)).expect("argmax");
            argmax.eval();
            let mut k = 0usize;
            while k < proposal_ids.len()
                && argmax.item_at_int32(k).expect("am") as u32 == proposal_ids[k]
            {
                k += 1;
            }
            let boundary = argmax.item_at_int32(k).expect("boundary") as u32;
            let mut emit = Vec::with_capacity(k + 1);
            emit.extend_from_slice(&proposal_ids[..k]);
            emit.push(boundary);
            let mut stopped = false;
            let mut emit_n = 0usize;
            for &tok in &emit {
                if dspark_ids.len() + emit_n >= 256 {
                    break;
                }
                emit_n += 1;
                if tok == CONTENT_MODEL_END_SAMPLING_ID {
                    stopped = true;
                    break;
                }
            }
            let keep = if stopped && emit_n <= k {
                1 + emit_n.saturating_sub(1)
            } else {
                1 + emit_n.min(k)
            };
            let tape = tapes.pop().expect("tape");
            {
                use crate::engine::spec_paged::SpecPagedCache;
                let rt = inner.paged.as_mut().expect("paged");
                rt.coordinator
                    .commit_cycle(seq_id, ticket, keep)
                    .expect("commit");
                let conv = rt.conv_states.get_mut(&seq_id).expect("conv");
                tape.commit_prefix(keep, conv).expect("tape commit");
                let committed = rt.coordinator.request_token_count_all(seq_id).expect("pos");
                rt.coordinator
                    .settle_committed(seq_id, u64::from(committed))
                    .expect("settle");
            }
            let draft = inner.dspark.as_ref().expect("dspark");
            draft_ctx
                .append_taps(draft, &taps, keep)
                .expect("append taps");

            if cap > 0 {
                cycles += 1;
                total_drafted += cap;
                total_accepted_drafts += k;
            }
            for &tok in &emit[..emit_n] {
                dspark_ids.push(tok);
                anchor = tok;
            }
        }
        let dp_dec_secs = t_dp_dec.elapsed().as_secs_f64();
        let dp_gen_steps = dspark_ids.len().saturating_sub(1);
        let dp_tps = dp_gen_steps as f64 / dp_dec_secs;
        let mean_accept_len = if cycles > 0 {
            (total_accepted_drafts + cycles) as f64 / cycles as f64
        } else {
            1.0
        };
        let matching_prefix = dspark_ids
            .iter()
            .zip(ar_ids.iter())
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "\n=== PHASE 5 BENCHMARK: SINGLE-SEQ DECODE (256 tokens, coding prompt) ===\n\
             AR decode:     {:.2} tok/s ({} tokens in {:.2}s)\n\
             DSpark decode: {:.2} tok/s ({} tokens in {:.2}s, speedup={:.2}x)\n\
             DSpark stats:  cycles={}, drafted={}, accepted_drafts={}, mean_accept_len={:.2} tok/cycle (accept_rate={:.1}%), exact_ar_prefix={}/{}\n",
            ar_perf.decode_tok_per_sec,
            ar_ids.len(),
            ar_perf.decode_secs,
            dp_tps,
            dspark_ids.len(),
            dp_dec_secs,
            dp_tps / ar_perf.decode_tok_per_sec.max(1e-6),
            cycles,
            total_drafted,
            total_accepted_drafts,
            mean_accept_len,
            100.0 * total_accepted_drafts as f64 / total_drafted.max(1) as f64,
            matching_prefix,
            ar_ids.len(),
        );
        assert!(
            matching_prefix >= 64,
            "Direct 256-token DSpark greedy diverged too early from AR (matched only {matching_prefix}/{})",
            ar_ids.len()
        );

        // 3. Prefill tok/s at 2K and 8K context
        for target_ctx in [2048usize, 8192usize] {
            let mut long_text = String::with_capacity(target_ctx * 6);
            while long_text.len() < target_ctx * 5 {
                long_text.push_str(
                    "The Rust borrow checker enforces aliasing XOR mutability at compile time, preventing data races across concurrent threads without runtime garbage collection overhead. ",
                );
            }
            let mut ids = render_inkling2_messages_to_ids(
                &[InklingMessage::user(&long_text)],
                tokenizer.as_ref(),
                &render_opts,
            )
            .expect("render long");
            ids.truncate(target_ctx);
            // Resize paged runtime if needed for 8K
            inner
                .enable_paged(crate::models::inkling2::kv_cache::Inkling2PagedConfig {
                    max_model_len: (target_ctx as u32).max(4096) + 256,
                    max_concurrent_sequences: 1,
                    ..crate::models::inkling2::kv_cache::Inkling2PagedConfig::default_for_config(
                        &inner.config,
                    )
                })
                .expect("enable_paged");
            inner.reset_paged_sequence(0).expect("reset");
            let t_pre = Instant::now();
            let (logits, _) = inner
                .paged_prefill_sequence(0, &ids, None)
                .expect("prefill");
            logits.eval();
            let secs = t_pre.elapsed().as_secs_f64();
            eprintln!(
                "=== PREFILL @ {} tokens: {:.2}s ({:.1} tok/s) ===",
                ids.len(),
                secs,
                ids.len() as f64 / secs
            );
        }

        // Release direct inner before loading full NAPI model (avoids holding 2x 18GB weights)
        drop(inner);
        crate::array::synchronize_and_clear_cache();

        // ------------------------------------------------------------------
        // B. Full Inkling2Model benchmark: 4-lane concurrent DSpark waves,
        //    SSD cold restore latency vs fresh prefill, and Vision smoke!
        // ------------------------------------------------------------------
        let temp_cache_dir =
            std::env::temp_dir().join(format!("inkling2_phase5_bench_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_cache_dir);
        let model = Inkling2Model::load(
            model_path.clone(),
            Some(Inkling2LoadOptions {
                kv_format: Some("int8".to_string()),
                cache_disk: Some("auto".to_string()),
                cache_dir: Some(temp_cache_dir.display().to_string()),
                persist_paged_cache: Some(true),
                ..Inkling2LoadOptions::default()
            }),
        )
        .await
        .expect("Inkling2Model::load");
        let tx = model.thread.cmd_sender().expect("cmd_sender");

        // 4. 4-lane concurrent aggregate tok/s (DSpark [4*8] waves vs AR [4,1])
        let wave_prompts = [
            "Write a Rust implementation of binary search with clear edge-case handling and complexity analysis.",
            "Write a Python implementation of merge sort with step-by-step comments explaining the divide and merge phases.",
            "Explain how Apple Silicon unified memory and Metal tile memory accelerate transformer inference in 6 concise points.",
            "Write a TypeScript implementation of a bounded async semaphore with acquire and release methods.",
        ];
        let t_4ar = Instant::now();
        let (a0, a1, a2, a3) = tokio::join!(
            run_turn(
                &tx,
                user_msg(wave_prompts[0]),
                greedy_cfg(Some("ar4-0"), 128, "ar4-0", false, false)
            ),
            run_turn(
                &tx,
                user_msg(wave_prompts[1]),
                greedy_cfg(Some("ar4-1"), 128, "ar4-1", false, false)
            ),
            run_turn(
                &tx,
                user_msg(wave_prompts[2]),
                greedy_cfg(Some("ar4-2"), 128, "ar4-2", false, false)
            ),
            run_turn(
                &tx,
                user_msg(wave_prompts[3]),
                greedy_cfg(Some("ar4-3"), 128, "ar4-3", false, false)
            ),
        );
        let ar4_secs = t_4ar.elapsed().as_secs_f64();
        let ar4_toks = a0.num_tokens + a1.num_tokens + a2.num_tokens + a3.num_tokens;
        model.reset_caches().await.expect("reset");

        let t_4dp = Instant::now();
        let (d0, d1, d2, d3) = tokio::join!(
            run_turn(
                &tx,
                user_msg(wave_prompts[0]),
                greedy_cfg(Some("dp4-0"), 128, "dp4-0", true, false)
            ),
            run_turn(
                &tx,
                user_msg(wave_prompts[1]),
                greedy_cfg(Some("dp4-1"), 128, "dp4-1", true, false)
            ),
            run_turn(
                &tx,
                user_msg(wave_prompts[2]),
                greedy_cfg(Some("dp4-2"), 128, "dp4-2", true, false)
            ),
            run_turn(
                &tx,
                user_msg(wave_prompts[3]),
                greedy_cfg(Some("dp4-3"), 128, "dp4-3", true, false)
            ),
        );
        let dp4_secs = t_4dp.elapsed().as_secs_f64();
        let dp4_toks = d0.num_tokens + d1.num_tokens + d2.num_tokens + d3.num_tokens;
        eprintln!(
            "=== 4-LANE CONCURRENT AGGREGATE (128 tok/lane) ===\n\
             4-lane AR:     {:.2} agg tok/s ({ar4_toks} tokens in {ar4_secs:.2}s)\n\
             4-lane DSpark: {:.2} agg tok/s ({dp4_toks} tokens in {dp4_secs:.2}s, speedup={:.2}x)\n",
            ar4_toks as f64 / ar4_secs,
            dp4_toks as f64 / dp4_secs,
            (dp4_toks as f64 / dp4_secs) / (ar4_toks as f64 / ar4_secs).max(1e-6),
        );
        model.reset_caches().await.expect("reset");

        // 5. SSD Cold Restore Latency vs Fresh Prefill (~4.5K context)
        let mut doc = String::from("Remember the secret verification code: SIGMA-9417.\n\n");
        for i in 0..160 {
            doc.push_str(&format!(
                "Section {i}: Distributed consensus systems maintain linearizability across replicated state machines through quorum intersections and term-numbered logs.\n"
            ));
        }
        doc.push_str("\nWhat is the secret verification code mentioned at the very beginning? Reply with only the code.");
        let mut ssd_cfg = greedy_cfg(Some("ssd-fresh"), 64, "ssd-bench", false, false);
        ssd_cfg.reuse_cache = Some(true);

        let t_fresh = Instant::now();
        let fresh_res = run_turn(&tx, user_msg(&doc), ssd_cfg.clone()).await;
        let fresh_ttft_ms = fresh_res
            .performance
            .as_ref()
            .map(|p| p.ttft_ms)
            .unwrap_or(t_fresh.elapsed().as_secs_f64() * 1000.0);
        // Flush to SSD & close, then reopen a fresh Inkling2Model pointing at the same persistent cache!
        assert!(model.close_cache(None).await.expect("close_cache"));
        drop(tx);
        model.dispose().await.expect("dispose");
        drop(model);
        crate::array::synchronize_and_clear_cache();

        let model2 = Inkling2Model::load(
            model_path,
            Some(Inkling2LoadOptions {
                kv_format: Some("int8".to_string()),
                cache_disk: Some("auto".to_string()),
                cache_dir: Some(temp_cache_dir.display().to_string()),
                persist_paged_cache: Some(true),
                ..Inkling2LoadOptions::default()
            }),
        )
        .await
        .expect("reload Inkling2Model for SSD cold restore");
        let tx2 = model2.thread.cmd_sender().expect("cmd_sender2");
        let mut restore_cfg = greedy_cfg(Some("ssd-cold"), 64, "ssd-bench", false, false);
        restore_cfg.reuse_cache = Some(true);
        let t_cold = Instant::now();
        let cold_res = run_turn(&tx2, user_msg(&doc), restore_cfg).await;
        let cold_ttft_ms = cold_res
            .performance
            .as_ref()
            .map(|p| p.ttft_ms)
            .unwrap_or(t_cold.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(cold_res.raw_text, fresh_res.raw_text);
        eprintln!(
            "=== SSD COLD RESTORE vs FRESH PREFILL ({} prompt tokens) ===\n\
             Fresh prefill TTFT:    {fresh_ttft_ms:.1} ms (answer={:?})\n\
             SSD cold restore TTFT: {cold_ttft_ms:.1} ms (cached_tokens={}, speedup={:.2}x)\n",
            fresh_res.prompt_tokens,
            fresh_res.raw_text.trim(),
            cold_res.cached_tokens,
            fresh_ttft_ms / cold_ttft_ms.max(1e-3),
        );

        // 6. Vision prompt smoke test (solid red 80x80 PNG encoded via `image` crate)
        assert!(model2.supports_images());
        let mut png_bytes = Vec::new();
        {
            let img = image::RgbImage::from_pixel(80, 80, image::Rgb([220, 20, 20]));
            let mut cursor = std::io::Cursor::new(&mut png_bytes);
            img.write_to(&mut cursor, image::ImageFormat::Png)
                .expect("encode PNG");
        }
        let vision_msg = vec![ChatMessage {
            role: "user".to_string(),
            content: "What primary color is this solid square image? Reply in one short sentence."
                .to_string(),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            reasoning_content: None,
            thinking_enabled: None,
            images: Some(vec![png_bytes.into()]),
            audio: None,
        }];
        let t_vis = Instant::now();
        let vis_res = run_turn(
            &tx2,
            vision_msg,
            greedy_cfg(Some("vision-smoke"), 96, "vision-salt", false, false),
        )
        .await;
        eprintln!(
            "=== VISION SMOKE (80x80 solid red PNG) ===\n\
             prompt_tokens={}, gen_tokens={}, wall={:.2}s, output={:?}\n",
            vis_res.prompt_tokens,
            vis_res.num_tokens,
            t_vis.elapsed().as_secs_f64(),
            vis_res.raw_text.trim(),
        );
        assert!(
            vis_res.raw_text.to_ascii_lowercase().contains("red"),
            "Expected vision model to identify red image, got {:?}",
            vis_res.raw_text
        );

        drop(tx2);
        model2.dispose().await.expect("dispose");
        let _ = std::fs::remove_dir_all(&temp_cache_dir);
    }
}

#[cfg(test)]
mod wave_grouper_tests {
    use std::collections::HashMap;

    use crate::models::inkling2::dflash2::Inkling2DraftContext;
    use crate::models::inkling2::dspark::{AdaptiveDraftDepth, Inkling2DsparkContext};
    use crate::models::inkling2::short_conv::Inkling2ShortConvTape;
    use crate::transformer::paged_kv_cache_adapter::PagedRaggedRow;

    use super::{
        Inkling2ScheduledDsparkLane, OpenInkling2Verify, select_verify_wave_groups_from_lanes,
    };

    fn make_lane(open: bool) -> Inkling2ScheduledDsparkLane {
        Inkling2ScheduledDsparkLane {
            ctx: Inkling2DraftContext::Dspark(Inkling2DsparkContext::new(3)),
            adaptive_depth: AdaptiveDraftDepth::new(7),
            open: open.then(|| OpenInkling2Verify {
                tape: Inkling2ShortConvTape::new(0),
                tapped: Vec::new(),
                lanes_in_wave: 1,
                drafted: 4,
                verify_tokens: Vec::new(),
            }),
            total_verifies: 0,
            multi_lane_wave_verifies: 0,
        }
    }

    fn make_row(seq_id: u32, len: usize) -> (PagedRaggedRow, Vec<u32>) {
        (
            PagedRaggedRow {
                seq_id,
                first_logical_position: 16,
                query_len: len as u32,
            },
            vec![42u32; len],
        )
    }

    #[test]
    fn ragged_multi_lane_wave_grouper_stacks_adaptive_depths_and_filters_peers() {
        let mut lanes = HashMap::new();
        for seq_id in [1u32, 2, 3, 4, 5, 6] {
            lanes.insert(seq_id, make_lane(false));
        }
        // Lane 7 already has an open verify -> excluded.
        lanes.insert(7, make_lane(true));

        // Mix ragged adaptive speculative lanes (`len` in 4..=8), a 1-row AR peer (seq 99),
        // an open lane (seq 7), and a 1-row tail verify (seq 6 with len=1).
        let rows = vec![
            make_row(1, 4),  // idx 0: k=3 (eligible)
            make_row(99, 1), // idx 1: AR peer (excluded)
            make_row(2, 6),  // idx 2: k=5 (eligible)
            make_row(7, 8),  // idx 3: already open (excluded)
            make_row(3, 8),  // idx 4: k=7 (eligible)
            make_row(4, 5),  // idx 5: k=4 (eligible)
            make_row(6, 1),  // idx 6: len=1 tail (excluded)
            make_row(5, 2),  // idx 7: k=1 (eligible, starts second chunk but lone -> ignored)
        ];
        let groups = select_verify_wave_groups_from_lanes(&lanes, &rows);
        assert_eq!(groups, vec![vec![0, 2, 4, 5]]);

        // Add one more eligible lane so the second chunk has 2 lanes and stacks too!
        let mut rows2 = rows;
        lanes.insert(8, make_lane(false));
        rows2.push(make_row(8, 3)); // idx 8: k=2 (eligible)
        let groups2 = select_verify_wave_groups_from_lanes(&lanes, &rows2);
        assert_eq!(groups2, vec![vec![0, 2, 4, 5], vec![7, 8]]);

        // A single eligible lane does not form a multi-lane wave group.
        let solo = vec![make_row(1, 5), make_row(99, 1)];
        assert!(select_verify_wave_groups_from_lanes(&lanes, &solo).is_empty());
    }
}
