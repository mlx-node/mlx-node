//! Speculative decode with the MTP head: the engine's `run_mtp_turn`
//! drives `Stepper` (flat: one `Lagged` sequence; paged: the active owner
//! of an exclusive turn, `Paged`); `run_turn` is the flat whole turn around
//! it (verify, chunked prefill, first sample, MTP loop, save, finalize,
//! stream). Scheduled turns use the propose / verify / commit hooks of
//! `sched.rs` instead.
//!
//! ```text
//! Step A  forward y at p        draft row p-1 closed first when missing (seed, y)
//! cycle   draft rows p.. (row p exact, later rows drafts) ─► verify depth+1 trunk rows
//!         commit: drafts dropped, exact rows (h, next id) from the verify hiddens
//!         rollback: trunk cut to snapshot + accepted + 1 (append-only state, no replay)
//! stop    unemitted drafts cut from the trunk; the draft settles to trunk - 1 at save
//! ```
//!
//! Every trunk state is a set of lengths with `WINDOW_SLACK` rows of window
//! and indexer slack, so a rejected draft is a truncation. In paged mode a
//! wave writes the MTP rows its trunk rows close, so a commit only checks
//! and the rollback's cut leaves the exact rows.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::sync::Arc;
use std::time::Instant;

use napi::{Error, Result};

use super::cache::WINDOW_SLACK;
use super::chat::Inner;
use super::model::{Logits, Model, Want};
use super::moe::RowKind;
use super::mtp::{Draft, Lagged, MtpLayer, Seeds};
use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, FinalizeArgs, MtpBackend, MtpStepper, ResetScope, SaveStateArgs, SpecFrontier,
    TurnOutput, WholeTurnArgs,
};
use crate::engine::decode::TurnStreaming;
use crate::engine::mtp_turn::{MtpTurnArgs, run_mtp_turn};
use crate::engine::params::{generated_capacity_hint, resolve_enable_thinking};
use crate::engine::plan::{MediaCapabilities, SpeculativeKind, SpeculativePlan};
use crate::engine::{ReasoningTracker, apply_all_penalties, compute_performance_metrics};
use crate::models::qwen3_5::mtp_decode::{MtpCommitAnchor, MtpVerifyOutput};
use crate::nn::Embedding;
use crate::stream::{Stream, WiredLimitContext};
use crate::transformer::paged_kv_cache_adapter::SeqId;

/// Most trunk rows one cycle may add past the snapshot: the anchor plus
/// `WINDOW_SLACK` drafts, all of which a rejection must be able to cut.
pub const LOOKAHEAD_MAX: usize = WINDOW_SLACK + 1;

/// The flat MTP plan: no paged pool, a streaming arm.
pub(crate) const PLAN: SpeculativePlan = SpeculativePlan {
    kind: SpeculativeKind::NativeMtp,
    supported_input_media: MediaCapabilities::NONE,
    supported_context_media: MediaCapabilities::NONE,
    supports_paged_attention: false,
    supports_streaming: true,
};

/// The paged MTP plan: verify on the pool, so the scheduled lane.
pub(crate) const PAGED_PLAN: SpeculativePlan = SpeculativePlan {
    supports_paged_attention: true,
    ..PLAN
};

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 MTP decode: {msg}"))
}

/// The draft row a cycle's commit starts at, for a verify that began at
/// trunk length `snap`: the Step-A anchor's row `snap - 1`, or `snap` on a
/// chained cycle (its anchor row is already committed).
pub(crate) fn commit_row(anchor: MtpCommitAnchor, snap: usize) -> Option<usize> {
    match anchor {
        MtpCommitAnchor::IncludeAnchor => snap.checked_sub(1),
        MtpCommitAnchor::SkipAlreadyCommittedAnchor => Some(snap),
    }
}

/// Draft rows kept when a cycle starts: every committed row, or one fewer
/// on a chained cycle, whose first draft rewrites its anchor's row.
pub fn cycle_rows(committed: usize, chained: bool) -> Option<usize> {
    if chained {
        committed.checked_sub(1)
    } else {
        Some(committed)
    }
}

/// Trunk rows kept after a verify from `snap` with `accepted` drafts: the
/// anchor and the accepted drafts.
pub fn verify_keep(snap: usize, accepted: usize) -> usize {
    snap + accepted + 1
}

/// Trunk rows kept when `unemitted` (at most the accepted count) cycle
/// tokens were not emitted: the bonus was never a trunk row, the other
/// unemitted ones are accepted drafts. The last emitted token stays a
/// trunk row, so the draft cut stays within `WINDOW_SLACK`.
pub fn unemitted_keep(trunk: usize, unemitted: usize) -> usize {
    trunk.saturating_sub(unemitted.saturating_sub(1))
}

/// The `[m, d]` trunk outputs a commit of `m` ids pairs them with: the
/// cycle seed then the verify rows (Step A), or the verify rows alone
/// (chained).
fn commit_hiddens(
    anchor: MtpCommitAnchor,
    seed: &MxArray,
    verify: &MxArray,
    m: usize,
) -> Result<MxArray> {
    let d = verify.shape_at(2)?;
    let n = verify.shape_at(1)? as usize;
    let vh = verify.reshape(&[n as i64, d])?;
    match anchor {
        MtpCommitAnchor::IncludeAnchor => {
            let s = seed.reshape(&[1, d])?;
            match m {
                0 => Err(bad("a commit of no ids")),
                1 => Ok(s),
                _ if m - 1 <= n => {
                    MxArray::concatenate(&s, &vh.slice(&[0, 0], &[m as i64 - 1, d])?, 0)
                }
                _ => Err(bad(format!("{m} ids for {n} verify rows"))),
            }
        }
        MtpCommitAnchor::SkipAlreadyCommittedAnchor if m > 0 && m <= n => {
            vh.slice(&[0, 0], &[m as i64, d])
        }
        MtpCommitAnchor::SkipAlreadyCommittedAnchor => {
            Err(bad(format!("{m} ids for {n} verify rows")))
        }
    }
}

/// The per-turn MTP stepper over one `Lagged` sequence. `committed` counts
/// the draft rows that pair trunk outputs with committed tokens; rows past
/// it are drafts of this cycle.
pub(crate) struct Flat<'a> {
    model: &'a Model,
    mtp: &'a MtpLayer,
    seq: &'a mut Lagged,
    embedding: Embedding,
    committed: usize,
    snap: Option<usize>,
    draft_scope: Option<super::stats::DeferScope>,
    err: Option<Error>,
}

impl<'a> Flat<'a> {
    /// A stepper on a settled sequence (`trunk - 1` draft rows).
    fn begin(model: &'a Model, mtp: &'a MtpLayer, seq: &'a mut Lagged) -> Result<Self> {
        let trunk = seq.len();
        let draft = seq
            .draft()
            .ok_or_else(|| bad("the sequence has no draft"))?;
        if trunk == 0 || draft.len() + 1 != trunk || draft.seed(trunk - 1).is_none() {
            return Err(bad(format!(
                "a turn on a trunk of {trunk} with {} draft rows",
                draft.len()
            )));
        }
        Ok(Self {
            model,
            mtp,
            committed: draft.len(),
            embedding: mtp.embedding().clone(),
            seq,
            snap: None,
            draft_scope: None,
            err: None,
        })
    }

    fn draft_mut(&mut self) -> Result<&mut Draft> {
        self.seq
            .parts()
            .1
            .ok_or_else(|| bad("the sequence has no draft"))
    }

    fn stash(&mut self, r: Result<()>) {
        if let Err(e) = r
            && self.err.is_none()
        {
            self.err = Some(e);
        }
    }

    fn cut_trunk(&mut self, keep: usize) -> Result<()> {
        let (trunk, draft) = self.seq.parts();
        if keep > trunk.len() {
            return Err(bad(format!("keep {keep} of a trunk of {}", trunk.len())));
        }
        trunk.truncate(keep)?;
        if let Some(d) = draft {
            d.seeds_mut().cut(keep);
        }
        Ok(())
    }
}

impl MtpStepper for Flat<'_> {
    fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    fn committed_history_active(&self) -> bool {
        true
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        Some("hy4_mtp")
    }

    fn forward_with_hidden(
        &mut self,
        ids: &MxArray,
        _embedding: &Embedding,
    ) -> Result<(MxArray, MxArray, bool)> {
        let token = ids.item_at_int32(0)? as u32;
        if self.draft_mut()?.len() != self.committed {
            return Err(bad("a main forward with draft rows held"));
        }
        let (logits, hidden) = self.seq.step(self.model, Some(self.mtp), token)?;
        let rows = self.draft_mut()?.len();
        self.committed = rows;
        let hidden = hidden.ok_or_else(|| bad("the trunk returned no hidden row"))?;
        Ok((logits, hidden, false))
    }

    fn draft_step(&mut self, prev_h: &MxArray, prev_emb: &MxArray) -> Result<(MxArray, MxArray)> {
        self.draft_scope
            .get_or_insert_with(super::stats::DeferScope::enter);
        let (model, mtp) = (self.model, self.mtp);
        let d = self.draft_mut()?;
        mtp.draft_lazy(model, d, prev_h, prev_emb)
    }

    fn verify_step(
        &mut self,
        ids: &[u32],
        _embedding: &Embedding,
        depth: usize,
    ) -> Result<MtpVerifyOutput> {
        if ids.len() != depth + 1 || ids.len() > LOOKAHEAD_MAX {
            self.draft_scope = None;
            return Err(bad(format!("{} verify ids at depth {depth}", ids.len())));
        }
        let model = self.model;
        let (trunk, draft) = self.seq.parts();
        let d = draft.ok_or_else(|| bad("the sequence has no draft"))?;
        let start = trunk.len();
        let want = Want {
            logits: Logits::All,
            hidden: true,
        };
        let out = model.forward(trunk, ids, RowKind::Decode, want, None, None);
        self.draft_scope = None;
        let out = out?;
        let n = ids.len() as i64;
        let shaped = out
            .logits
            .zip(out.hidden)
            .ok_or_else(|| bad("the verify returned no logits or hiddens"))
            .and_then(|(l, h)| {
                let seeds = Seeds::copies(&h, start)?;
                let (v, dim) = (l.shape_at(1)?, h.shape_at(1)?);
                Ok((l.reshape(&[1, n, v])?, h.reshape(&[1, n, dim])?, seeds))
            });
        match shaped {
            Ok((logits, hiddens, seeds)) => {
                d.seeds_mut().extend(seeds);
                Ok(MtpVerifyOutput { logits, hiddens })
            }
            Err(e) => {
                trunk.truncate(start)?;
                Err(e)
            }
        }
    }

    fn snapshot_main_linear(&mut self) {
        self.snap = Some(self.seq.len());
    }

    fn rollback(&mut self, accepted_drafts: usize, depth: usize) {
        if self.err.is_some() {
            return;
        }
        let r = match self.snap {
            Some(snap) if accepted_drafts <= depth => {
                self.cut_trunk(verify_keep(snap, accepted_drafts))
            }
            Some(_) => Err(bad(format!("{accepted_drafts} accepted of {depth} drafts"))),
            None => Err(bad("rollback without a snapshot")),
        };
        self.stash(r);
    }

    fn restore_and_replay_main(&mut self, _accepted: &[u32], _embedding: &Embedding) -> Result<()> {
        match self.err.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn commit_mtp(
        &mut self,
        anchor: MtpCommitAnchor,
        seed_h: &MxArray,
        verify_hiddens: &MxArray,
        committed_ids: &[u32],
        _k_accepted: usize,
        _embedding: &Embedding,
    ) -> Result<()> {
        let snap = self
            .snap
            .ok_or_else(|| bad("a commit without a snapshot"))?;
        let at = commit_row(anchor, snap);
        if at != Some(self.committed) {
            return Err(bad(format!(
                "a commit at row {at:?} after {} committed rows",
                self.committed
            )));
        }
        let hs = commit_hiddens(anchor, seed_h, verify_hiddens, committed_ids.len())?;
        let (model, mtp, keep) = (self.model, self.mtp, self.committed);
        let d = self.draft_mut()?;
        d.cut(keep)?;
        let committed = {
            let _scope = super::stats::DeferScope::enter();
            mtp.commit(model, d, &hs, committed_ids)
                .and_then(|()| super::stats::flush("hy_v4_flash MTP commit"))
        };
        if let Err(e) = committed {
            let _ = d.cut(keep);
            return Err(e);
        }
        let rows = d.len();
        self.committed = rows;
        Ok(())
    }

    fn begin_cycle(&mut self, chained_anchor: bool) {
        self.draft_scope = None;
        let r = match cycle_rows(self.committed, chained_anchor) {
            Some(keep) => self.draft_mut().and_then(|d| d.cut(keep)),
            None => Err(bad("a chained cycle with no committed row")),
        };
        self.stash(r);
    }

    /// `Ok(false)` past the slack or the context: the engine decodes the
    /// token AR instead.
    fn reserve_cycle_lookahead(&mut self, rows: usize) -> Result<bool> {
        let trunk = self.seq.trunk();
        Ok(rows <= LOOKAHEAD_MAX && trunk.len() + rows <= trunk.cap())
    }

    fn eval_step(&self, token: &MxArray, logits: &MxArray, budget_forced: bool) {
        if budget_forced {
            MxArray::async_eval_arrays(&[token, logits]);
        } else {
            MxArray::async_eval_arrays(&[token]);
        }
    }

    fn eval_step_with_chained_hidden(&self, token: &MxArray, chained_h: &MxArray) {
        MxArray::async_eval_arrays(&[token, chained_h]);
    }

    fn rollback_unemitted(&mut self, unemitted: usize) {
        let keep = unemitted_keep(self.seq.len(), unemitted);
        let r = self.cut_trunk(keep).and_then(|()| {
            let d = self.draft_mut()?;
            let rows = d.len().min(keep.saturating_sub(1));
            d.cut(rows)?;
            Ok(rows)
        });
        match r {
            Ok(rows) => self.committed = rows,
            Err(e) => self.stash(Err(e)),
        }
    }

    fn frontier(&self) -> Option<SpecFrontier> {
        Some(SpecFrontier {
            attn_tokens: self.seq.len() as u64,
            recurrent_tokens: None,
        })
    }

    fn take_replay_error(&mut self) -> Option<Error> {
        self.err.take()
    }

    fn into_desynced(self) -> bool {
        false
    }
}

/// The per-turn MTP stepper of an exclusive paged turn over the active
/// owner. Each trunk forward is a one-owner wave, which also writes the MTP
/// rows it closes; draft rows past the trunk's last row are this cycle's
/// and go before the next wave.
pub(crate) struct Paged<'a> {
    inner: &'a mut Inner,
    seq: SeqId,
    embedding: Embedding,
    snap: Option<usize>,
    draft_scope: Option<super::stats::DeferScope>,
    err: Option<Error>,
}

impl<'a> Paged<'a> {
    fn begin(inner: &'a mut Inner) -> Result<Self> {
        let embedding = inner
            .mtp
            .as_ref()
            .ok_or_else(|| bad("an MTP turn without the MTP layer"))?
            .embedding()
            .clone();
        let seq = inner
            .paged()?
            .owners
            .active()
            .ok_or_else(|| bad("a paged MTP turn without an active owner"))?;
        if inner.owner_len(seq) == 0 {
            return Err(bad("a paged MTP turn on an empty cell"));
        }
        inner.settle(seq)?;
        Ok(Self {
            inner,
            seq,
            embedding,
            snap: None,
            draft_scope: None,
            err: None,
        })
    }

    fn stash(&mut self, r: Result<()>) {
        if let Err(e) = r
            && self.err.is_none()
        {
            self.err = Some(e);
        }
    }

    /// One wave of `ids` with `logits`: `(logits, hidden)` of its rows.
    fn wave(&mut self, ids: &[u32], logits: Logits) -> Result<(MxArray, MxArray)> {
        let out = self
            .inner
            .mtp_wave(&[(self.seq, ids)], logits)?
            .into_iter()
            .next()
            .ok_or_else(|| bad("the wave returned no owner"))?;
        out.logits
            .zip(out.hidden)
            .ok_or_else(|| bad("the wave returned no logits or hiddens"))
    }
}

impl MtpStepper for Paged<'_> {
    fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    fn committed_history_active(&self) -> bool {
        true
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        Some("hy4_mtp")
    }

    fn forward_with_hidden(
        &mut self,
        ids: &MxArray,
        _embedding: &Embedding,
    ) -> Result<(MxArray, MxArray, bool)> {
        let token = ids.item_at_int32(0)? as u32;
        let (logits, hidden) = self.wave(&[token], Logits::Last)?;
        Ok((logits, hidden, false))
    }

    fn draft_step(&mut self, prev_h: &MxArray, prev_emb: &MxArray) -> Result<(MxArray, MxArray)> {
        self.draft_scope
            .get_or_insert_with(super::stats::DeferScope::enter);
        self.inner.draft_row_lazy(self.seq, prev_h, prev_emb)
    }

    fn verify_step(
        &mut self,
        ids: &[u32],
        _embedding: &Embedding,
        depth: usize,
    ) -> Result<MtpVerifyOutput> {
        if ids.len() != depth + 1 || ids.len() > LOOKAHEAD_MAX {
            self.draft_scope = None;
            return Err(bad(format!("{} verify ids at depth {depth}", ids.len())));
        }
        let start = self.inner.owner_len(self.seq);
        let wave = self.wave(ids, Logits::All);
        self.draft_scope = None;
        let (logits, hidden) = wave?;
        let shaped = verify_output(&logits, &hidden, ids.len() as i64);
        if shaped.is_err() {
            let _ = self.inner.cut_owner(self.seq, start);
        }
        shaped
    }

    fn snapshot_main_linear(&mut self) {
        self.snap = Some(self.inner.owner_len(self.seq));
    }

    fn rollback(&mut self, accepted_drafts: usize, depth: usize) {
        if self.err.is_some() {
            return;
        }
        let r = match self.snap {
            Some(snap) if accepted_drafts <= depth => self
                .inner
                .cut_owner(self.seq, verify_keep(snap, accepted_drafts)),
            Some(_) => Err(bad(format!("{accepted_drafts} accepted of {depth} drafts"))),
            None => Err(bad("rollback without a snapshot")),
        };
        self.stash(r);
    }

    fn restore_and_replay_main(&mut self, _accepted: &[u32], _embedding: &Embedding) -> Result<()> {
        match self.err.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The verify wave already wrote the exact MTP rows of every verify
    /// row; the rollback's cut keeps the committed ones.
    fn commit_mtp(
        &mut self,
        _anchor: MtpCommitAnchor,
        _seed_h: &MxArray,
        _verify_hiddens: &MxArray,
        committed_ids: &[u32],
        _k_accepted: usize,
        _embedding: &Embedding,
    ) -> Result<()> {
        if self.snap.is_none() || committed_ids.is_empty() {
            return Err(bad("a commit without a snapshot or ids"));
        }
        Ok(())
    }

    fn begin_cycle(&mut self, _chained_anchor: bool) {
        self.draft_scope = None;
        let r = self.inner.settle(self.seq);
        self.stash(r);
    }

    /// `Ok(false)` past the slack, the row cap or the context: the engine
    /// decodes the token AR instead.
    fn reserve_cycle_lookahead(&mut self, rows: usize) -> Result<bool> {
        self.inner.reserve_rows(self.seq, rows)
    }

    fn eval_step(&self, token: &MxArray, logits: &MxArray, budget_forced: bool) {
        if budget_forced {
            MxArray::async_eval_arrays(&[token, logits]);
        } else {
            MxArray::async_eval_arrays(&[token]);
        }
    }

    fn eval_step_with_chained_hidden(&self, token: &MxArray, chained_h: &MxArray) {
        MxArray::async_eval_arrays(&[token, chained_h]);
    }

    fn rollback_unemitted(&mut self, unemitted: usize) {
        let keep = unemitted_keep(self.inner.owner_len(self.seq), unemitted);
        let r = self.inner.cut_owner(self.seq, keep);
        self.stash(r);
    }

    fn frontier(&self) -> Option<SpecFrontier> {
        Some(SpecFrontier {
            attn_tokens: self.inner.owner_len(self.seq) as u64,
            recurrent_tokens: None,
        })
    }

    fn take_replay_error(&mut self) -> Option<Error> {
        self.err.take()
    }

    fn into_desynced(self) -> bool {
        false
    }
}

/// A verify's `logits [n, vocab]` and `hidden [n, d]` as one owner's
/// `[1, n, ..]`.
fn verify_output(logits: &MxArray, hidden: &MxArray, n: i64) -> Result<MtpVerifyOutput> {
    let (v, d) = (logits.shape_at(1)?, hidden.shape_at(1)?);
    Ok(MtpVerifyOutput {
        logits: logits.reshape(&[1, n, v])?,
        hiddens: hidden.reshape(&[1, n, d])?,
    })
}

/// The MTP stepper of a turn: flat or paged, by the load's mode.
pub(crate) enum Stepper<'a> {
    Flat(Flat<'a>),
    Paged(Paged<'a>),
}

impl<'a> Stepper<'a> {
    /// A flat stepper on a settled sequence (`trunk - 1` draft rows).
    pub(crate) fn begin(model: &'a Model, mtp: &'a MtpLayer, seq: &'a mut Lagged) -> Result<Self> {
        Flat::begin(model, mtp, seq).map(Self::Flat)
    }

    /// A paged stepper on `inner`'s active owner.
    pub(crate) fn paged(inner: &'a mut Inner) -> Result<Self> {
        Paged::begin(inner).map(Self::Paged)
    }
}

impl MtpStepper for Stepper<'_> {
    fn embedding(&self) -> &Embedding {
        match self {
            Self::Flat(s) => s.embedding(),
            Self::Paged(s) => s.embedding(),
        }
    }

    fn committed_history_active(&self) -> bool {
        match self {
            Self::Flat(s) => s.committed_history_active(),
            Self::Paged(s) => s.committed_history_active(),
        }
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        match self {
            Self::Flat(s) => s.profiler_relabel(),
            Self::Paged(s) => s.profiler_relabel(),
        }
    }

    fn forward_with_hidden(
        &mut self,
        ids: &MxArray,
        embedding: &Embedding,
    ) -> Result<(MxArray, MxArray, bool)> {
        match self {
            Self::Flat(s) => s.forward_with_hidden(ids, embedding),
            Self::Paged(s) => s.forward_with_hidden(ids, embedding),
        }
    }

    fn draft_step(&mut self, prev_h: &MxArray, prev_emb: &MxArray) -> Result<(MxArray, MxArray)> {
        match self {
            Self::Flat(s) => s.draft_step(prev_h, prev_emb),
            Self::Paged(s) => s.draft_step(prev_h, prev_emb),
        }
    }

    fn verify_step(
        &mut self,
        ids: &[u32],
        embedding: &Embedding,
        depth: usize,
    ) -> Result<MtpVerifyOutput> {
        match self {
            Self::Flat(s) => s.verify_step(ids, embedding, depth),
            Self::Paged(s) => s.verify_step(ids, embedding, depth),
        }
    }

    fn snapshot_main_linear(&mut self) {
        match self {
            Self::Flat(s) => s.snapshot_main_linear(),
            Self::Paged(s) => s.snapshot_main_linear(),
        }
    }

    fn rollback(&mut self, accepted_drafts: usize, depth: usize) {
        match self {
            Self::Flat(s) => s.rollback(accepted_drafts, depth),
            Self::Paged(s) => s.rollback(accepted_drafts, depth),
        }
    }

    fn restore_and_replay_main(&mut self, accepted: &[u32], embedding: &Embedding) -> Result<()> {
        match self {
            Self::Flat(s) => s.restore_and_replay_main(accepted, embedding),
            Self::Paged(s) => s.restore_and_replay_main(accepted, embedding),
        }
    }

    fn commit_mtp(
        &mut self,
        anchor: MtpCommitAnchor,
        seed_h: &MxArray,
        verify_hiddens: &MxArray,
        committed_ids: &[u32],
        k_accepted: usize,
        embedding: &Embedding,
    ) -> Result<()> {
        match self {
            Self::Flat(s) => s.commit_mtp(
                anchor,
                seed_h,
                verify_hiddens,
                committed_ids,
                k_accepted,
                embedding,
            ),
            Self::Paged(s) => s.commit_mtp(
                anchor,
                seed_h,
                verify_hiddens,
                committed_ids,
                k_accepted,
                embedding,
            ),
        }
    }

    fn begin_cycle(&mut self, chained_anchor: bool) {
        match self {
            Self::Flat(s) => s.begin_cycle(chained_anchor),
            Self::Paged(s) => s.begin_cycle(chained_anchor),
        }
    }

    fn reserve_cycle_lookahead(&mut self, rows: usize) -> Result<bool> {
        match self {
            Self::Flat(s) => s.reserve_cycle_lookahead(rows),
            Self::Paged(s) => s.reserve_cycle_lookahead(rows),
        }
    }

    fn eval_step(&self, token: &MxArray, logits: &MxArray, budget_forced: bool) {
        match self {
            Self::Flat(s) => s.eval_step(token, logits, budget_forced),
            Self::Paged(s) => s.eval_step(token, logits, budget_forced),
        }
    }

    fn eval_step_with_chained_hidden(&self, token: &MxArray, chained_h: &MxArray) {
        match self {
            Self::Flat(s) => s.eval_step_with_chained_hidden(token, chained_h),
            Self::Paged(s) => s.eval_step_with_chained_hidden(token, chained_h),
        }
    }

    fn rollback_unemitted(&mut self, unemitted: usize) {
        match self {
            Self::Flat(s) => s.rollback_unemitted(unemitted),
            Self::Paged(s) => s.rollback_unemitted(unemitted),
        }
    }

    fn frontier(&self) -> Option<SpecFrontier> {
        match self {
            Self::Flat(s) => s.frontier(),
            Self::Paged(s) => s.frontier(),
        }
    }

    fn take_replay_error(&mut self) -> Option<Error> {
        match self {
            Self::Flat(s) => s.take_replay_error(),
            Self::Paged(s) => s.take_replay_error(),
        }
    }

    fn into_desynced(self) -> bool {
        match self {
            Self::Flat(s) => s.into_desynced(),
            Self::Paged(s) => s.into_desynced(),
        }
    }
}

/// Puts `backend` back to an empty conversation and returns `e`.
fn fail_closed<B: ChatBackend, T>(backend: &mut B, e: Error) -> Result<T> {
    if let Err(r) = backend.reset_caches(ResetScope::Command) {
        tracing::error!(
            "hy_v4_flash MTP turn failed ({}) and the reset failed too ({})",
            e.reason,
            r.reason
        );
    }
    Err(e)
}

/// One whole MTP turn: the generic flat flow with `run_mtp_turn` as the
/// decode loop.
pub(crate) fn run_turn<B: MtpBackend>(
    backend: &mut B,
    args: &mut WholeTurnArgs<'_>,
) -> Result<TurnOutput> {
    let tokens = args.tokens;
    if tokens.is_empty() {
        return Err(bad("empty prompt"));
    }
    let tokenizer = Arc::clone(args.tokenizer);
    let mut p = args.params.clone();
    p.extra_eos_ids = backend.extra_eos_ids();
    let max_new = match backend.context_token_limit() {
        Some(limit) => p
            .max_new_tokens
            .min(i32::try_from((limit + 1).saturating_sub(tokens.len())).unwrap_or(i32::MAX)),
        None => p.max_new_tokens,
    };
    let report = p.report_performance;
    let started = report.then(Instant::now);
    let hit = backend.verify_cache_prefix(tokens, p.reuse_cache);
    let (prefill, cached) = if hit > 0 && hit < tokens.len() {
        (&tokens[hit..], hit)
    } else {
        if let Err(e) = backend.reset_caches(ResetScope::PrefixMiss) {
            return fail_closed(backend, e);
        }
        (tokens, 0)
    };
    let stream = Stream::generation();
    let _wired = backend
        .wired_limit_bytes()
        .map(|b| WiredLimitContext::new(b, vec![stream]));
    let mut profiler = DecodeProfiler::new(
        backend.profiler_label(args.plan.is_delta, args.sink.is_some()),
        backend.family_name(),
    );
    profiler.set_prompt_tokens(prefill.len() as u32);
    profiler.snapshot_memory_before();
    let close = backend.think_end_for_turn(args.config, &tokenizer);
    let mut tracker = ReasoningTracker::from_setup(&args.thinking, &close);
    let skip_special = backend.stream_skip_special_tokens();
    let mut ts = args.sink.map(|_| {
        TurnStreaming::new(
            &*backend,
            tokenizer.inner(),
            args.thinking.enabled,
            skip_special,
        )
    });
    let mut history = tokens.to_vec();
    let mut generated = Vec::with_capacity(generated_capacity_hint(max_new));
    let mut finish = String::from("length");
    let mut first: Option<Instant> = None;

    let decoded: Result<()> = (|| {
        let progress = args.sink.and_then(|s| s.progress());
        if let Some(pr) = progress {
            pr.begin_prefill(prefill.len() as u32, cached as u32);
        }
        profiler.begin_prefill();
        let logits = backend.prefill(prefill, stream)?;
        profiler.end_prefill();
        let logits = apply_all_penalties(logits, &history, &p)?;
        let y = crate::sampling::sample(&logits, p.sampling_config)?;
        y.eval();
        if let Some(pr) = progress {
            pr.first_token();
        }
        backend.eval_caches()?;
        if report {
            first = Some(Instant::now());
        }
        let ctx = match (args.sink, ts.as_mut()) {
            (Some(s), Some(t)) => t.ctx(Some(s), args.cancelled),
            _ => None,
        };
        let mut rng = rand::rng();
        run_mtp_turn(
            &mut *backend,
            &mut rng,
            MtpTurnArgs {
                y,
                depth: p.mtp_depth,
                params: &p,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens: max_new,
                eos_id: args.eos_id,
                generated_tokens: &mut generated,
                token_history: &mut history,
                finish_reason: &mut finish,
                first_token_instant: &mut first,
                report_perf: report,
                generation_stream: stream,
                prompt_hidden: None,
                prompt_hidden_ids: None,
                cancel_flag: args.cancelled,
            },
            ctx,
        )?;
        Ok(())
    })();
    if let Err(e) = decoded {
        return fail_closed(backend, e);
    }

    backend.save_cache_state(SaveStateArgs {
        reuse_cache: p.reuse_cache,
        is_delta: args.plan.is_delta,
        has_images: false,
        generated_tokens: &generated,
        finish_reason: &finish,
        save_tokens: tokens,
        save_expanded_tokens: None,
        image_cache_key: 0,
    });

    let performance = if report {
        compute_performance_metrics(started, first, prefill.len(), generated.len()).map(|mut m| {
            backend.augment_performance(&profiler, &mut m);
            m
        })
    } else {
        None
    };
    if let (Some(s), Some(t)) = (args.sink, ts.as_mut()) {
        let full = tokenizer
            .decode_sync(&generated, skip_special)
            .unwrap_or_else(|e| {
                tracing::warn!("hy_v4_flash: failed to decode generated tokens: {e}");
                String::new()
            });
        t.flush_residual(&full, p.include_reasoning, s);
    }
    let finalized = backend.finalize_turn(FinalizeArgs {
        tokenizer: &tokenizer,
        generated_tokens: &generated,
        finish_reason: finish,
        think_end_id: close.think_end_id,
        think_end_str: close.think_end_str.as_deref(),
        think_end_extra_ids: &close.think_end_extra_ids,
        performance,
        include_reasoning: p.include_reasoning,
        thinking_enabled: args.thinking.enabled,
        prompt_tokens: tokens.len() as u32,
        reasoning: tracker.outcome(),
    });
    let mut result = match finalized {
        Ok(r) => r,
        Err(e) => return fail_closed(backend, e),
    };
    result.thinking_enabled = resolve_enable_thinking(args.config).unwrap_or(true);
    result.cached_tokens = cached as u32;
    if let (Some(s), Some(t)) = (args.sink, ts.as_mut()) {
        t.emitter.finish(&result, s);
        return Ok(TurnOutput::Streamed);
    }
    Ok(TurnOutput::Complete(Box::new(result)))
}
