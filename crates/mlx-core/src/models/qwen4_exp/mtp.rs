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
    /// (rows in the draft cache, trusted rows waiting for the next pass).
    #[cfg(test)]
    pub(super) fn draft_rows(&self, seq: u32) -> Option<(usize, usize)> {
        self.owners
            .get(&seq)
            .map(|owner| (owner.cache.history_shape().0, owner.rows.len()))
    }
    pub fn owner_frontier(&self, seq: u32) -> Option<usize> {
        self.owners.get(&seq).map(|owner| owner.frontier)
    }
    /// What an owner holds: draft K/V for its cached rows, the hidden rows
    /// waiting for the next first draft pass (trusted rows and the seed),
    /// plus its verify snapshots inside an MTP turn.
    fn owner_bytes(c: &super::config::Config, owner: &Owner) -> u64 {
        let hidden_rows = owner.rows.len() as u64 + u64::from(owner.hidden.is_some());
        let cached_rows = (owner.frontier as u64).saturating_sub(hidden_rows);
        super::memory::draft_owner_bytes(c, cached_rows, hidden_rows, owner.tracking)
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
        }
    }
}
impl Inner {
    pub fn begin_mtp(&mut self, seq: u32, position: u32) -> Result<bool> {
        if position == 0 {
            self.mtp.owners.insert(
                seq,
                Owner {
                    tracking: true,
                    ..Default::default()
                },
            );
            return Ok(true);
        }
        if let Some(owner) = self.mtp.owners.get_mut(&seq) {
            owner.tracking = owner.frontier == position as usize;
            return Ok(owner.tracking);
        }
        Ok(false)
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
