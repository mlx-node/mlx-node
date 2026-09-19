//! Native HC MTP with trusted target frontiers and transactional verification.
//! Proposals use a private draft-cache clone. Only accepted target hidden rows
//! advance the persistent draft cache; rejected speculative hiddens never do.
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
use crate::models::qwen4_exp::runtime_flags;
use napi::{Error, Result};
use std::collections::HashMap;

#[derive(Default)]
struct Owner {
    cache: LayerCache,
    hidden: Option<MxArray>,
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
        let result = (|| {
            if tokens.len() > 1 && !runtime_flags::is_zero(c"MLX_QWEN4_MTP_PREFILL") {
                let skip = usize::from(owner.hidden.is_none());
                let mut previous = Vec::with_capacity(tokens.len());
                if let Some(h) = &owner.hidden {
                    previous.push(h.clone());
                }
                previous.extend_from_slice(&hidden[..hidden.len() - 1]);
                let position = (owner.frontier + skip).saturating_sub(1);
                let width = self.decoder.prefill_chunk_size;
                for (chunk, (h, t)) in previous
                    .chunks(width)
                    .zip(tokens[skip..].chunks(width))
                    .enumerate()
                {
                    self.decoder
                        .draft_prefill(h, t, position + chunk * width, &mut owner.cache)?;
                }
                owner.hidden = hidden.last().cloned();
                owner.frontier += tokens.len();
                return Ok(());
            }
            for (&token, target_hidden) in tokens.iter().zip(hidden) {
                if let Some(previous) = &owner.hidden {
                    self.decoder.draft_prefill(
                        std::slice::from_ref(previous),
                        &[token],
                        owner.frontier - 1,
                        &mut owner.cache,
                    )?;
                }
                owner.hidden = Some(target_hidden.clone());
                owner.frontier += 1;
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
        let mut hidden = owner
            .hidden
            .clone()
            .ok_or_else(|| Error::from_reason("Qwen4 MTP missing target HC seed"))?;
        let mut cache = owner.cache.clone();
        let start = owner
            .frontier
            .checked_sub(1)
            .ok_or_else(|| Error::from_reason("Qwen4 MTP empty frontier"))?;
        let mut token = anchor;
        let mut draft_ids = Vec::new();
        let mut draft_dists = Vec::new();
        let mut keep_probabilities = Vec::new();
        let cfg = params.sampling_config.unwrap_or_default();
        let greedy = crate::sampling::is_greedy_temperature(cfg.temperature.unwrap_or(1.0));
        let cfg = crate::models::qwen3_5::mtp_decode::mtp_draft_sampling_config(cfg);
        for i in 0..cap.min(3) {
            self.decoder.check_cancelled()?;
            let (next, logits) =
                self.decoder
                    .draft_step(&hidden, token, start + i, &mut cache, true)?;
            hidden = next;
            let logits = logits
                .ok_or_else(|| Error::from_reason("Qwen4 MTP draft step produced no logits"))?
                .reshape(&[-1])?;
            let id = if greedy {
                logits.argmax(-1, None)?.astype(DType::Int32)?
            } else {
                let dist = crate::sampling::sampling_distribution(&logits, Some(cfg))?
                    .astype(DType::Float32)?;
                let id = crate::sampling::sample_dense_distribution_array(&dist, rng)?;
                draft_dists.push(dist);
                id
            };
            token = id.item_at_int32(0)? as u32;
            if confidence {
                let probabilities = crate::nn::Activations::softmax_precise(
                    &logits.astype(DType::Float32)?,
                    Some(-1),
                )?;
                keep_probabilities.push(probabilities.item_at_float32(token as usize)?);
            }
            draft_ids.push(token as i32);
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
