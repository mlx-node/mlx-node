//! Native HC MTP with trusted target frontiers and transactional verification.
//! Only accepted target hidden rows advance the persistent draft cache;
//! rejected speculative hiddens never do. Accepted rows wait for the next
//! proposal, whose first draft pass covers them and the anchor row together.
use super::{
    Inner,
    decoder::{DecoderState, LayerCache},
};
use crate::array::{DType, MxArray};
use crate::engine::backend::DsparkProposal;
use crate::engine::hybrid_scheduler::{
    HybridSchedulerBackend, ScheduledVerifyCommit, ScheduledVerifyRow,
};
use crate::engine::params::ChatParams;
use napi::{Error, Result};
use std::collections::HashMap;

#[derive(Default)]
struct Owner {
    cache: LayerCache,
    hidden: Option<MxArray>,
    /// Trusted (target hidden, next token) rows not yet in `cache`, ending at
    /// position `frontier - 2`.
    rows: Vec<(MxArray, u32)>,
    frontier: usize,
    tracking: bool,
    /// The first draft passes, cut out of `cache` once a state takes them:
    /// segment `k` holds rows `[kW, (k + 1)W)`.
    segments: Vec<super::prefix_state::Segment>,
    /// Every row of `cache` came from a draft pass of `W` rows starting at a
    /// multiple of `W`, as a cold prefill's observes write it.
    aligned: bool,
    /// The scheduler began speculation on this owner for its turn.
    speculative: bool,
}
impl Owner {
    /// An owner before row 0 that records what it observes.
    fn fresh() -> Self {
        Self {
            tracking: true,
            aligned: true,
            ..Default::default()
        }
    }
}
struct Pending {
    base: DecoderState,
    after: Vec<DecoderState>,
    hidden: Vec<MxArray>,
    tokens: Vec<u32>,
}
#[derive(Default)]
pub struct State {
    owners: HashMap<u32, Owner>,
    pending: HashMap<u32, Pending>,
    pub budget: crate::engine::verification_budget::ScheduledVerificationBudget,
}
impl State {
    pub fn release(&mut self, seq: u32) {
        self.owners.remove(&seq);
        self.pending.remove(&seq);
        self.budget.clear();
    }
    /// Drops `seq`'s owner: its draft cache holds a history the sequence no
    /// longer has.
    pub fn forget(&mut self, seq: u32) {
        self.owners.remove(&seq);
        self.pending.remove(&seq);
    }
    /// (rows in the draft cache, trusted rows waiting for the next pass).
    #[cfg(test)]
    pub(super) fn draft_rows(&self, seq: u32) -> Option<(usize, usize)> {
        self.owners
            .get(&seq)
            .map(|owner| (owner.cache.history_shape().0, owner.rows.len()))
    }
    /// (draft cache arrays, waiting (hidden, token) rows, frontier) of an
    /// owner: what a prefill's observes leave behind.
    #[cfg(test)]
    pub(super) fn draft_state(
        &self,
        seq: u32,
    ) -> Option<(Vec<MxArray>, Vec<(MxArray, u32)>, usize)> {
        self.owners.get(&seq).map(|owner| {
            (
                owner.cache.arrays().cloned().collect(),
                owner.rows.clone(),
                owner.frontier,
            )
        })
    }
    /// (cached draft K/V, V and indexer key rows, waiting hidden rows and
    /// their tokens, seed, frontier) of an owner, bit for bit.
    #[cfg(test)]
    pub(super) fn draft_bits(&self, seq: u32) -> Option<DraftBits> {
        let owner = self.owners.get(&seq)?;
        let bits = |a: &MxArray| {
            a.astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec()
        };
        let rows = owner.cache.draft_row_count();
        let mut cached = Vec::new();
        if let Some((rows, keys, values)) = owner.cache.draft_kv() {
            for buffer in [keys, values] {
                cached.extend(bits(&buffer.slice_axis(2, 0, rows as i64).unwrap()));
            }
        }
        cached.extend(owner.cache.draft_index().map(bits).unwrap_or_default());
        Some(DraftBits {
            rows,
            cached,
            waiting: owner.rows.iter().flat_map(|(h, _)| bits(h)).collect(),
            tokens: owner.rows.iter().map(|(_, t)| *t).collect(),
            seed: owner.hidden.as_ref().map(bits),
            frontier: owner.frontier,
        })
    }
    /// The owner of `seq` no longer holds the rows a cold prefill's draft
    /// passes write.
    #[cfg(test)]
    pub(super) fn misalign(&mut self, seq: u32) {
        if let Some(owner) = self.owners.get_mut(&seq) {
            owner.aligned = false;
        }
    }
    /// Whether the scheduler began speculation on `seq`'s owner.
    pub(super) fn speculative(&self, seq: u32) -> bool {
        self.owners.get(&seq).is_some_and(|owner| owner.speculative)
    }
    pub fn owner_frontier(&self, seq: u32) -> Option<usize> {
        self.owners.get(&seq).map(|owner| owner.frontier)
    }
    /// A cold text prefill on a load with the MTP head records its draft
    /// history from row 0: an owner already there stays, any other is
    /// replaced.
    pub(super) fn start(&mut self, seq: u32) {
        if self
            .owners
            .get(&seq)
            .is_some_and(|owner| owner.frontier == 0)
        {
            return;
        }
        self.pending.remove(&seq);
        self.owners.insert(seq, Owner::fresh());
    }
    /// The draft history of a state at row `lineage.len()`: the owner a cold
    /// prefill of `lineage` leaves there, replacing any other.
    pub(super) fn restore(
        &mut self,
        seq: u32,
        draft: &super::prefix_state::DraftCell,
        lineage: &[u32],
        window: usize,
    ) -> Result<()> {
        let row = lineage.len();
        let cached = super::memory::cached_draft_rows(row as u64, window as u64) as usize;
        let cache = LayerCache::from_draft_segments(&draft.segments)?;
        let waiting = row.saturating_sub(1 + cached);
        let held = match &draft.waiting {
            Some(rows) => rows.shape_at(1)? as usize,
            None => 0,
        };
        if row == 0 || cache.draft_row_count() != cached || held != waiting {
            return Err(Error::from_reason(
                "Qwen4 stored draft history does not match its row",
            ));
        }
        let rows = match &draft.waiting {
            Some(hidden) => (0..waiting)
                .map(|j| {
                    Ok((
                        hidden.slice_axis(1, j as i64, j as i64 + 1)?,
                        lineage[cached + 1 + j],
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
            None => Vec::new(),
        };
        self.pending.remove(&seq);
        self.owners.insert(
            seq,
            Owner {
                cache,
                hidden: Some(draft.seed.clone()),
                rows,
                frontier: row,
                tracking: true,
                segments: draft.segments.clone(),
                aligned: true,
                speculative: false,
            },
        );
        Ok(())
    }
    /// The draft part of a state at `row`: the owner's draft passes cut into
    /// segments (shared with the states this owner published before), and
    /// copies of its waiting rows and seed. `None` unless every row of the
    /// owner's cache came from the cold prefill's draft passes and the owner
    /// stands at `row`.
    pub(super) fn capture(
        &mut self,
        seq: u32,
        row: usize,
        window: usize,
        ratio: usize,
    ) -> Result<Option<super::prefix_state::DraftCell>> {
        let Some(owner) = self.owners.get_mut(&seq) else {
            return Ok(None);
        };
        let cached = super::memory::cached_draft_rows(row as u64, window as u64) as usize;
        if !owner.tracking
            || !owner.aligned
            || owner.frontier != row
            || window == 0
            || !window.is_multiple_of(ratio.max(1))
            || owner.cache.draft_row_count() != cached
            || owner.cache.index_pending()
            || owner.rows.len() + cached + 1 != row
        {
            return Ok(None);
        }
        let Some(seed) = &owner.hidden else {
            return Ok(None);
        };
        while owner.segments.len() * window < cached {
            let from = owner.segments.len() * window;
            let Some(segment) = owner.cache.draft_segment(from, from + window, ratio)? else {
                owner.aligned = false;
                owner.segments.clear();
                return Ok(None);
            };
            owner.segments.push(segment);
        }
        let waiting = if owner.rows.is_empty() {
            None
        } else {
            Some(
                MxArray::concatenate_many(owner.rows.iter().map(|(h, _)| h).collect(), Some(1))?
                    .deep_copy()?,
            )
        };
        let seed = seed.deep_copy()?;
        let mut ready = vec![&seed];
        ready.extend(waiting.iter());
        MxArray::eval_arrays(&ready)?;
        Ok(Some(super::prefix_state::DraftCell {
            segments: owner.segments.clone(),
            waiting,
            seed,
        }))
    }
    /// What an owner holds: draft K/V and indexer keys for its cached rows,
    /// the hidden rows waiting for the next first draft pass (trusted rows
    /// and the seed), the segments cut out of its cache (copies, shared with
    /// the states it published), plus its verify snapshots inside an MTP
    /// turn.
    fn owner_bytes(c: &super::config::Config, owner: &Owner) -> u64 {
        let hidden_rows = owner.rows.len() as u64 + u64::from(owner.hidden.is_some());
        let cached_rows = (owner.frontier as u64).saturating_sub(hidden_rows);
        let segments: u64 = owner
            .segments
            .iter()
            .map(|segment| {
                (segment.keys.nbytes() + segment.values.nbytes() + segment.index.nbytes()) as u64
            })
            .sum();
        super::memory::draft_owner_bytes(c, cached_rows, hidden_rows, owner.speculative) + segments
    }
    pub fn held_bytes(&self, c: &super::config::Config) -> u64 {
        self.owners
            .values()
            .map(|owner| Self::owner_bytes(c, owner))
            .sum()
    }
    pub fn owner_held_bytes(&self, c: &super::config::Config, seq: u32) -> u64 {
        self.owners
            .get(&seq)
            .map_or(0, |owner| Self::owner_bytes(c, owner))
    }
    pub fn finish(&mut self, seq: u32, frontier: Option<usize>) {
        self.pending.remove(&seq);
        if !self
            .owners
            .get(&seq)
            .is_some_and(|owner| Some(owner.frontier) == frontier)
        {
            self.owners.remove(&seq);
        } else if let Some(owner) = self.owners.get_mut(&seq) {
            // A completed MTP turn can resume this exact frontier. If the
            // scheduler instead falls back to AR, its next target observation
            // drops the head without doing further draft work.
            owner.tracking = false;
            owner.speculative = false;
        }
    }
}

/// An owner's draft history, compared bit for bit by the tests.
#[cfg(test)]
#[derive(Debug, PartialEq)]
pub(super) struct DraftBits {
    pub rows: usize,
    pub cached: Vec<f32>,
    pub waiting: Vec<f32>,
    pub tokens: Vec<u32>,
    pub seed: Option<Vec<f32>>,
    pub frontier: usize,
}
impl Inner {
    /// An owner at `position` (the one a prefill's prime left there, or one
    /// a finished MTP turn left at the row's frontier) begins speculation;
    /// at row 0 a fresh one does.
    pub fn begin_mtp(&mut self, seq: u32, position: u32) -> Result<bool> {
        if let Some(owner) = self.mtp.owners.get_mut(&seq)
            && owner.frontier == position as usize
        {
            owner.tracking = true;
            owner.speculative = true;
            return Ok(true);
        }
        if position == 0 {
            self.mtp.owners.insert(
                seq,
                Owner {
                    speculative: true,
                    ..Owner::fresh()
                },
            );
            return Ok(true);
        }
        if let Some(owner) = self.mtp.owners.get_mut(&seq) {
            owner.tracking = false;
        }
        Ok(false)
    }
    /// Whether `observe_mtp` for `seq` would record the hidden rows it is
    /// given: a prefill builds its per-token views only then.
    pub(super) fn mtp_tracking(&self, seq: u32) -> bool {
        self.mtp
            .owners
            .get(&seq)
            .is_some_and(|owner| owner.tracking)
    }
    pub fn observe_mtp(&mut self, seq: u32, tokens: &[u32], hidden: &[MxArray]) -> Result<()> {
        let Some(mut owner) = self.mtp.owners.remove(&seq) else {
            return Ok(());
        };
        if !owner.tracking {
            return Ok(());
        }
        if tokens.len() != hidden.len() {
            return Err(Error::from_reason(
                "Qwen4 MTP target hidden alignment mismatch",
            ));
        }
        // Prefill hands over lazy views of its hidden window, and a verifier
        // restore must not reinstate a frontier that still holds them.
        let lazy: Vec<_> = hidden
            .iter()
            .filter(|h| !unsafe { mlx_sys::mlx_array_is_available(h.as_raw_ptr()) })
            .collect();
        MxArray::eval_arrays_with_context(&lazy, "qwen4::mtp::target_rows")?;
        for (&token, target_hidden) in tokens.iter().zip(hidden) {
            if let Some(previous) = owner.hidden.replace(target_hidden.clone()) {
                owner.rows.push((previous, token));
            }
            owner.frontier += 1;
        }
        let width = self.decoder.prefill_chunk_size;
        let result = (|| {
            // A full window cannot also carry the next proposal's anchor row.
            while owner.rows.len() >= width {
                let position = owner.frontier - 1 - owner.rows.len();
                if !position.is_multiple_of(width) {
                    owner.aligned = false;
                    owner.segments.clear();
                }
                let (h, t): (Vec<_>, Vec<_>) = owner.rows.drain(..width).unzip();
                self.decoder
                    .draft_prefill(&h, &t, position, &mut owner.cache)?;
            }
            Ok(())
        })();
        if result.is_ok() {
            self.mtp.owners.insert(seq, owner);
        }
        result
    }
    pub fn propose_mtp(
        &mut self,
        seq: u32,
        anchor: u32,
        cap: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
        confidence: bool,
    ) -> Result<DsparkProposal> {
        let owner = self
            .mtp
            .owners
            .get(&seq)
            .ok_or_else(|| Error::from_reason("Qwen4 MTP missing owner"))?;
        let seed = owner
            .hidden
            .clone()
            .ok_or_else(|| Error::from_reason("Qwen4 MTP missing target HC seed"))?;
        let start = owner
            .frontier
            .checked_sub(1)
            .ok_or_else(|| Error::from_reason("Qwen4 MTP empty frontier"))?;
        let mut cache = owner.cache.clone();
        let (mut hidden, mut tokens): (Vec<_>, Vec<_>) = owner.rows.iter().cloned().unzip();
        hidden.push(seed);
        tokens.push(anchor);
        let mut position = start + 1 - tokens.len();
        let mut draft_ids = Vec::new();
        let mut draft_dists = Vec::new();
        let mut keep_probabilities = Vec::new();
        let cfg = params.sampling_config.unwrap_or_default();
        let greedy = crate::sampling::is_greedy_temperature(cfg.temperature.unwrap_or(1.0));
        let cfg = crate::models::qwen3_5::mtp_decode::mtp_draft_sampling_config(cfg);
        for i in 0..cap.min(3) {
            self.decoder.check_cancelled()?;
            let (next, logits, draft) = self
                .decoder
                .draft_step(&hidden, &tokens, position, &mut cache)?;
            let logits = logits.reshape(&[-1])?;
            let id = if greedy {
                logits.argmax(-1, None)?.astype(DType::Int32)?
            } else {
                let dist = crate::sampling::sampling_distribution(&logits, Some(cfg))?
                    .astype(DType::Float32)?;
                let id = crate::sampling::sample_dense_distribution_array(&dist, rng)?;
                draft_dists.push(dist);
                id
            };
            let probabilities = if confidence {
                Some(crate::nn::Activations::softmax_precise(
                    &logits.astype(DType::Float32)?,
                    Some(-1),
                )?)
            } else {
                None
            };
            let mut ready = vec![&id, &next];
            ready.extend(probabilities.iter());
            if i == 0 {
                ready.extend(cache.arrays());
            }
            MxArray::eval_arrays_with_context(&ready, "qwen4::mtp::draft")?;
            if i == 0 {
                // The first pass appended exactly the owner's trusted rows.
                let owner = self
                    .mtp
                    .owners
                    .get_mut(&seq)
                    .ok_or_else(|| Error::from_reason("Qwen4 MTP missing owner"))?;
                owner.cache = cache;
                owner.rows.clear();
                // A proposal's first pass wrote rows no prefill writes: no
                // state is taken from this owner again.
                owner.aligned = false;
                owner.segments.clear();
            }
            let token = id.item_at_int32(0)? as u32;
            if let Some(probabilities) = &probabilities {
                keep_probabilities.push(probabilities.item_at_float32(token as usize)?);
            }
            draft_ids.push(token as i32);
            hidden = vec![next];
            tokens = vec![token];
            position = start + i + 1;
            cache = draft;
        }
        Ok(DsparkProposal {
            draft_ids,
            device_draft_ids: None,
            draft_dists,
            draft_sparse_dists: Vec::new(),
            keep_probabilities: confidence.then_some(keep_probabilities),
        })
    }
    pub fn verify_mtp(&mut self, rows: &[ScheduledVerifyRow]) -> Result<MxArray> {
        if !self.mtp.pending.is_empty() {
            return Err(Error::from_reason(
                "Qwen4 verifier transaction already open",
            ));
        }
        let mut logits = Vec::new();
        for row in rows {
            self.activate_paged_seq(row.seq_id)?;
            if self.decoder.history.len() != row.first_position as usize
                || row.tokens.is_empty()
                || row.tokens.len() > 4
            {
                return Err(Error::from_reason(
                    "Qwen4 verifier frontier or width mismatch",
                ));
            }
            let mut pending = Pending {
                base: self.decoder.snapshot(),
                after: Vec::new(),
                hidden: Vec::new(),
                tokens: row.tokens.clone(),
            };
            let result = self.decoder.verify_chunk(&row.tokens);
            let (batch_logits, states) = match result {
                Ok(value) => value,
                Err(error) => {
                    self.mtp.release(row.seq_id);
                    return Err(error);
                }
            };
            // Shared speculative sampling expects [queries, 1, vocabulary]
            // and transposes each owner's slice to [1, queries, vocabulary].
            logits.push(batch_logits.expand_dims(1)?);
            pending.hidden = self.decoder.last_chunk_hidden.clone();
            pending.after = states;
            self.mtp.pending.insert(row.seq_id, pending);
        }
        MxArray::concatenate_many(logits.iter().collect(), Some(0))
    }
    pub fn commit_mtp(&mut self, rows: &[ScheduledVerifyCommit]) -> Result<Vec<Result<()>>> {
        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            results.push((|| {
                let pending = self.mtp.pending.remove(&row.seq_id).ok_or_else(|| {
                    Error::from_reason("Qwen4 verifier commit has no transaction")
                })?;
                if row.keep > pending.tokens.len() {
                    return Err(Error::from_reason(
                        "Qwen4 verifier commit exceeds verified rows",
                    ));
                }
                self.activate_paged_seq(row.seq_id)?;
                self.decoder
                    .paged
                    .as_mut()
                    .ok_or_else(|| Error::from_reason("Qwen4 verifier commit has no paged cache"))?
                    .rollback_last_tokens((pending.tokens.len() - row.keep) as u32)
                    .map_err(Error::from_reason)?;
                let state = if row.keep == 0 {
                    pending.base
                } else {
                    pending.after[row.keep - 1].clone()
                };
                self.decoder.restore_state(state);
                self.saved_history = self.decoder.history.clone();
                self.observe_mtp(
                    row.seq_id,
                    &pending.tokens[..row.keep],
                    &pending.hidden[..row.keep],
                )
            })());
        }
        Ok(results)
    }
}
