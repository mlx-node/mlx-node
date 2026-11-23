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

use napi::{Error, Result};

use super::cache::WINDOW_SLACK;
use super::chat::Inner;
use super::model::{Logits, Model, Want};
use super::moe::RowKind;
use super::mtp::{Draft, Lagged, MtpLayer, Seeds};
use crate::array::MxArray;
use crate::engine::backend::{MtpStepper, SpecFrontier, TurnOutput, WholeTurnArgs};
use crate::engine::mtp_decode::{MtpCommitAnchor, MtpVerifyOutput};
use crate::engine::mtp_turn::{FlatMtpBackend, run_flat_mtp_turn};
use crate::engine::plan::{MediaCapabilities, SpeculativeKind, SpeculativePlan};
use crate::nn::Embedding;
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
        self.draft_step_with_persist(prev_h, prev_emb, true)
    }

    fn draft_step_with_persist(
        &mut self,
        prev_h: &MxArray,
        prev_emb: &MxArray,
        persist_state: bool,
    ) -> Result<(MxArray, MxArray)> {
        self.draft_scope
            .get_or_insert_with(super::stats::DeferScope::enter);
        let (model, mtp, committed) = (self.model, self.mtp, self.committed);
        let d = self.draft_mut()?;
        // On a chained cycle `begin_cycle` cuts one row below `committed` and
        // `commit_mtp` reuses the row written by slot 0, so rows below
        // `committed` must persist both KV and indexer state.
        let persist = persist_state || d.len() < committed;
        mtp.draft_lazy_with_persist(model, d, prev_h, prev_emb, persist)
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
        let committed = self.committed;
        let r = self.cut_trunk(keep).and_then(|()| {
            let d = self.draft_mut()?;
            let rows = committed
                .min(d.len())
                .min(d.index().len())
                .min(keep.saturating_sub(1));
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
        self.draft_step_with_persist(prev_h, prev_emb, true)
    }

    fn draft_step_with_persist(
        &mut self,
        prev_h: &MxArray,
        prev_emb: &MxArray,
        persist_state: bool,
    ) -> Result<(MxArray, MxArray)> {
        self.draft_scope
            .get_or_insert_with(super::stats::DeferScope::enter);
        self.inner
            .draft_row_lazy_with_persist(self.seq, prev_h, prev_emb, persist_state)
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

    fn draft_step_with_persist(
        &mut self,
        prev_h: &MxArray,
        prev_emb: &MxArray,
        persist_state: bool,
    ) -> Result<(MxArray, MxArray)> {
        match self {
            Self::Flat(s) => s.draft_step_with_persist(prev_h, prev_emb, persist_state),
            Self::Paged(s) => s.draft_step_with_persist(prev_h, prev_emb, persist_state),
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

impl FlatMtpBackend for Inner {}

/// One whole MTP turn: delegates to the engine-owned [`run_flat_mtp_turn`].
pub(crate) fn run_turn<B: FlatMtpBackend>(
    backend: &mut B,
    args: &mut WholeTurnArgs<'_>,
) -> Result<TurnOutput> {
    run_flat_mtp_turn(backend, args)
}
