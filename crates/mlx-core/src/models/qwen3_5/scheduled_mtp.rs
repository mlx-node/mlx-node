//! Shared native-MTP scheduling for dense and MoE Qwen targets.
//! Target families provide projection/decoder math; request state, proposal
//! chaining, scratch verification and accepted-prefix replay live here once.
use super::arrays_cache::ArraysCache;
use super::gated_delta_net::GdnLayerTape;
use super::layer_cache::{Qwen3_5LayerCache, Qwen3_5LayerSnapshot, replay_mtp_snapshot_to};
use crate::array::{DType, MxArray};
use crate::engine::backend::{DsparkProposal, SpecFrontier};
use crate::engine::hybrid_scheduler::{ScheduledVerifyCommit, ScheduledVerifyRow};
use crate::engine::params::ChatParams;
use crate::engine::scheduled_verify::{ScheduledDraftVerify, ScheduledVerifyBatch};
use crate::engine::spec_paged::SpecPagedCache;
use crate::sampling::{is_greedy_temperature, sampling_distribution};
use crate::sampling::{materialize_draft_tokens, sample_dense_distribution_array};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, PagedRaggedRow};
use napi::bindgen_prelude::{Error, Result};
use rand::Rng;
use std::collections::{BTreeMap, HashMap};

pub(crate) const MAX_DRAFTS: usize = 7;

pub(crate) struct MtpOwner {
    pub frontier: u32,
    pub hidden: Option<MxArray>,
    draft_caches: Vec<Qwen3_5LayerCache>,
    committed_frontier: Option<usize>,
}

impl MtpOwner {
    fn validate_committed_history(&self) -> Result<()> {
        if let Some(frontier) = self.committed_frontier
            && (self.frontier.checked_sub(1).map(|value| value as usize) != Some(frontier)
                || self.draft_caches.iter().any(|cache| match cache {
                    Qwen3_5LayerCache::FullAttention(cache) => {
                        cache.get_offset() as usize != frontier
                    }
                    _ => true,
                }))
        {
            return Err(Error::from_reason(
                "MTP committed history disagrees with target frontier",
            ));
        }
        Ok(())
    }
}

struct PendingOwner {
    first_position: u32,
    tokens: Vec<u32>,
    snapshot: Vec<Qwen3_5LayerSnapshot>,
    tape: Vec<Option<GdnLayerTape>>,
    hidden: MxArray,
}

#[derive(Default)]
pub(crate) struct ScheduledMtpState {
    pub owners: HashMap<u32, MtpOwner>,
    pending: HashMap<u32, PendingOwner>,
    batch: Option<ScheduledVerifyBatch>,
}

impl ScheduledMtpState {
    pub fn release(&mut self, seq_id: u32) {
        self.owners.remove(&seq_id);
        self.pending.remove(&seq_id);
    }

    pub fn reservation_bytes(config: &super::config::Qwen3_5Config, total_tokens: u32) -> u64 {
        let linear_layers = (0..config.num_layers.max(0) as usize)
            .filter(|&i| config.is_linear_layer(i))
            .count() as u64;
        let heads = config.linear_num_value_heads.max(0) as u64;
        let tape_row = heads
            .saturating_mul(
                2 * config.linear_key_head_dim.max(0) as u64
                    + config.linear_value_head_dim.max(0) as u64
                    + 2,
            )
            .saturating_add(config.linear_conv_dim().max(0) as u64);
        let tapes = linear_layers
            .saturating_mul((MAX_DRAFTS + 1) as u64)
            .saturating_mul(tape_row)
            .saturating_mul(4);
        let draft_kv = (config.n_mtp_layers.max(0) as u64)
            .saturating_mul(config.num_kv_heads.max(0) as u64)
            .saturating_mul(config.head_dim.max(0) as u64)
            .saturating_mul(u64::from(total_tokens).saturating_add(256 + MAX_DRAFTS as u64))
            .saturating_mul(8);
        config
            .recurrent_state_bytes()
            .saturating_mul(8)
            .saturating_add(tapes)
            .saturating_add(draft_kv)
            .saturating_add(
                (config.hidden_size.max(0) as u64)
                    .saturating_mul(u64::from(total_tokens).max((MAX_DRAFTS + 3) as u64))
                    .saturating_mul(8),
            )
    }
}

pub(crate) trait ScheduledMtpTarget: Sized {
    fn mtp_state(&self) -> &ScheduledMtpState;
    fn mtp_state_mut(&mut self) -> &mut ScheduledMtpState;
    fn mtp_adapter(&self) -> Option<&PagedKVCacheAdapter>;
    fn mtp_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter>;
    fn park_mtp_target(&mut self) -> Result<()>;
    fn mtp_target_caches(&self, seq_id: u32) -> Result<&[Qwen3_5LayerCache]>;
    fn replace_mtp_target_caches(
        &mut self,
        seq_id: u32,
        caches: Vec<Qwen3_5LayerCache>,
    ) -> Result<()>;
    fn fresh_mtp_draft_caches(&self) -> Vec<Qwen3_5LayerCache>;
    fn embed_mtp_token(&self, ids: &MxArray) -> Result<MxArray>;
    fn run_mtp_draft_hidden(
        &mut self,
        hidden: &MxArray,
        embedding: &MxArray,
        caches: &mut [Qwen3_5LayerCache],
    ) -> Result<MxArray>;
    fn project_mtp_logits(&self, hidden: &MxArray) -> Result<MxArray>;
    fn run_mtp_draft(
        &mut self,
        hidden: &MxArray,
        embedding: &MxArray,
        caches: &mut [Qwen3_5LayerCache],
    ) -> Result<(MxArray, MxArray)> {
        let hidden = self.run_mtp_draft_hidden(hidden, embedding, caches)?;
        let logits = self.project_mtp_logits(&hidden)?;
        Ok((hidden, logits))
    }
    fn append_mtp_history(
        &mut self,
        hidden: &MxArray,
        ids: &[u32],
        caches: &mut [Qwen3_5LayerCache],
    ) -> Result<()> {
        if hidden.shape_at(1)? != ids.len() as i64 {
            return Err(Error::from_reason(
                "MTP history hidden/token length mismatch",
            ));
        }
        for start in (0..ids.len()).step_by(7) {
            let end = (start + 7).min(ids.len());
            let hidden = hidden.slice_axis(1, start as i64, end as i64)?;
            let ids = MxArray::from_uint32(&ids[start..end], &[1, (end - start) as i64])?;
            let embedding = self.embed_mtp_token(&ids)?;
            self.run_mtp_draft_hidden(&hidden, &embedding, caches)?;
        }
        Ok(())
    }
    fn run_mtp_target_bucket(
        &mut self,
        rows: &[(PagedRaggedRow, Vec<u32>)],
        caches: &mut [Qwen3_5LayerCache],
        tape: &mut [Option<GdnLayerTape>],
    ) -> Result<(MxArray, MxArray)>;

    #[cfg(test)]
    fn run_reference_mtp_target(
        &mut self,
        seq: u32,
        ids: &[u32],
        caches: &mut [Qwen3_5LayerCache],
        tape: &mut Vec<Option<GdnLayerTape>>,
    ) -> Result<MxArray>;

    fn begin_scheduled_mtp(&mut self, seq_id: u32, position: u32) -> Result<bool> {
        if self.mtp_state().batch.is_some() {
            return Err(Error::from_reason(
                "cannot seed MTP during an open verifier wave",
            ));
        }
        // Target KV/GDN prefix state does not contain the shifted hidden
        // history needed by the drafter. An empty draft cache would discard
        // both that prefix and every later suffix chunk. Keep the target's
        // prefix reuse and decline speculation instead of rebuilding it cold.
        if position != 0 {
            self.mtp_state_mut().release(seq_id);
            return Ok(false);
        }
        let draft_caches = self.fresh_mtp_draft_caches();
        let committed_frontier = (!draft_caches.is_empty()).then_some(0);
        self.mtp_state_mut().owners.insert(
            seq_id,
            MtpOwner {
                frontier: position,
                hidden: None,
                draft_caches,
                committed_frontier,
            },
        );
        Ok(true)
    }

    fn prefill_scheduled_mtp(
        &mut self,
        seq: u32,
        frontier: u32,
        hidden: MxArray,
        ids: &[u32],
    ) -> Result<()> {
        let mut owner = self
            .mtp_state_mut()
            .owners
            .remove(&seq)
            .ok_or_else(|| Error::from_reason("MTP prefill has no draft owner"))?;
        let result = (|| {
            if ids.is_empty()
                || hidden.shape_at(0)? != 1
                || hidden.shape_at(1)? != ids.len() as i64
                || owner.frontier.checked_add(
                    u32::try_from(ids.len())
                        .map_err(|_| Error::from_reason("MTP prefill width overflow"))?,
                ) != Some(frontier)
            {
                return Err(Error::from_reason(
                    "MTP prefill seed frontier/shape mismatch",
                ));
            }
            if let Some(committed) = owner.committed_frontier {
                let before_last = hidden.slice_axis(1, 0, ids.len() as i64 - 1)?;
                let (history, history_ids) = if let Some(previous) = owner.hidden.as_ref() {
                    (MxArray::concatenate(previous, &before_last, 1)?, ids)
                } else {
                    (before_last, &ids[1..])
                };
                self.append_mtp_history(&history, history_ids, &mut owner.draft_caches)?;
                owner.committed_frontier = Some(committed + history_ids.len());
            }
            // A compact GPU copy of one row releases the full prompt buffer.
            owner.hidden = Some(
                hidden
                    .slice_axis(1, ids.len() as i64 - 1, ids.len() as i64)?
                    .copy()?,
            );
            owner.frontier = frontier;
            owner.validate_committed_history()
        })();
        self.mtp_state_mut().owners.insert(seq, owner);
        result
    }

    #[cfg(test)]
    fn seed_scheduled_mtp(&mut self, seq_id: u32, frontier: u32, hidden: MxArray) -> Result<()> {
        let owner = self
            .mtp_state_mut()
            .owners
            .get_mut(&seq_id)
            .ok_or_else(|| Error::from_reason("MTP prefill has no draft owner"))?;
        if hidden.shape_at(0)? != 1 || hidden.shape_at(1)? != 1 || frontier < owner.frontier {
            return Err(Error::from_reason("invalid MTP prefill seed"));
        }
        owner.committed_frontier = None;
        owner.frontier = frontier;
        owner.hidden = Some(hidden);
        Ok(())
    }

    fn propose_scheduled_mtp(
        &mut self,
        seq_id: u32,
        anchor: u32,
        cap: usize,
        params: &ChatParams,
        rng: &mut dyn Rng,
    ) -> Result<DsparkProposal> {
        let mut owner = self
            .mtp_state_mut()
            .owners
            .remove(&seq_id)
            .ok_or_else(|| Error::from_reason("scheduled MTP has no draft owner"))?;
        let result = (|| {
            let mut hidden = owner
                .hidden
                .clone()
                .ok_or_else(|| Error::from_reason("scheduled MTP has no normalized target seed"))?;
            let mut ids = MxArray::from_uint32(&[anchor], &[1, 1])?;
            if let Some(frontier) = owner.committed_frontier {
                if frontier + 1 != owner.frontier as usize {
                    return Err(Error::from_reason("MTP committed draft frontier mismatch"));
                }
                for cache in &mut owner.draft_caches {
                    if let Some(kv) = cache.as_kv_cache_mut() {
                        kv.trim(frontier as i32);
                    }
                }
            } else {
                owner.draft_caches = self.fresh_mtp_draft_caches();
            }
            let config = params.sampling_config.unwrap_or_default();
            let greedy = is_greedy_temperature(config.temperature.unwrap_or(1.0));
            let config = super::mtp_decode::mtp_draft_sampling_config(config);
            let mut device_ids = Vec::new();
            let mut distributions = Vec::new();
            for _ in 0..cap.min(MAX_DRAFTS) {
                let embedding = self.embed_mtp_token(&ids)?;
                let (next_hidden, logits) =
                    self.run_mtp_draft(&hidden, &embedding, &mut owner.draft_caches)?;
                hidden = next_hidden;
                let logits = logits.reshape(&[-1])?;
                let token = if greedy {
                    logits.argmax(-1, None)?.astype(DType::Int32)?
                } else {
                    let distribution =
                        sampling_distribution(&logits, Some(config))?.astype(DType::Float32)?;
                    let token = sample_dense_distribution_array(&distribution, rng)?;
                    distributions.push(distribution);
                    token
                };
                ids = token.reshape(&[1, 1])?;
                device_ids.push(token);
            }
            let draft_ids = materialize_draft_tokens(&device_ids)?;
            Ok(DsparkProposal {
                draft_ids,
                draft_dists: distributions,
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            })
        })();
        self.mtp_state_mut().owners.insert(seq_id, owner);
        result
    }

    fn verify_scheduled_mtp(&mut self, rows: &[ScheduledVerifyRow]) -> Result<MxArray> {
        MtpVerify(self).verify_scheduled_rows(rows)
    }
    fn commit_scheduled_mtp(&mut self, rows: &[ScheduledVerifyCommit]) -> Result<Vec<Result<()>>> {
        MtpVerify(self).commit_scheduled_rows(rows)
    }
}

/// Attention metadata is transactional here; recurrent state is committed by
/// MtpVerify only after all attention tickets close. Durable GDN/SSD snapshots
/// remain in the existing turn epilogue, as on the whole-turn native MTP path.
struct MtpCache<'a>(Option<&'a mut PagedKVCacheAdapter>);
impl MtpCache<'_> {
    fn adapter(&mut self) -> std::result::Result<&mut PagedKVCacheAdapter, String> {
        self.0
            .as_deref_mut()
            .ok_or_else(|| "scheduled MTP has no paged adapter".into())
    }
}
impl SpecPagedCache for MtpCache<'_> {
    fn reserve_lookahead(&mut self, seq: u32, rows: usize) -> std::result::Result<bool, String> {
        match self.adapter()?.reserve_rows_for(
            seq,
            u32::try_from(rows).map_err(|_| "MTP reservation overflow")?,
        ) {
            Ok(_) => Ok(true),
            Err(error) if error.starts_with("context_length_exceeded:") => Ok(false),
            Err(error) => Err(error),
        }
    }
    fn record_rows(&mut self, seq: u32, tokens: &[u32]) -> std::result::Result<(), String> {
        self.adapter()?.record_tokens_for(seq, tokens)
    }
    fn rollback_rows(&mut self, seq: u32, rows: usize) -> std::result::Result<(), String> {
        self.adapter()?.rollback_last_tokens_for(
            seq,
            u32::try_from(rows).map_err(|_| "MTP rollback overflow")?,
        )
    }
    fn frontier(&self, seq: u32) -> Option<SpecFrontier> {
        self.0
            .as_deref()?
            .current_token_count_for(seq)
            .map(|n| SpecFrontier {
                attn_tokens: u64::from(n),
                recurrent_tokens: None,
            })
    }
    fn settle_committed(&mut self, seq: u32, tokens: u64) -> std::result::Result<(), String> {
        if self
            .frontier(seq)
            .is_none_or(|frontier| frontier.attn_tokens != tokens)
        {
            return Err("scheduled MTP settlement frontier mismatch".into());
        }
        Ok(())
    }
    fn settle_captures_durable_state(&self) -> bool {
        false
    }
}

struct MtpVerify<'a, T>(&'a mut T);
impl<T: ScheduledMtpTarget> ScheduledDraftVerify for MtpVerify<'_, T> {
    type Cache<'a>
        = MtpCache<'a>
    where
        Self: 'a;
    fn verify_cache(&mut self) -> Self::Cache<'_> {
        MtpCache(self.0.mtp_adapter_mut())
    }
    fn take_verify_batch(&mut self) -> Option<ScheduledVerifyBatch> {
        self.0.mtp_state_mut().batch.take()
    }
    fn store_verify_batch(&mut self, batch: ScheduledVerifyBatch) {
        self.0.mtp_state_mut().batch = Some(batch);
    }
    fn validate_draft_owner(&self, row: &ScheduledVerifyRow) -> Result<()> {
        let owner = self.0.mtp_state().owners.get(&row.seq_id);
        if row.speculative != owner.is_some()
            || owner
                .is_some_and(|owner| owner.frontier != row.first_position || owner.hidden.is_none())
        {
            return Err(Error::from_reason(
                "scheduled MTP draft owner/frontier mismatch",
            ));
        }
        if row.tokens.len() > MAX_DRAFTS + 1 {
            return Err(Error::from_reason(
                "scheduled MTP verifier width exceeds its tape reservation",
            ));
        }
        Ok(())
    }
    fn run_packed_target(
        &mut self,
        rows: &[(PagedRaggedRow, Vec<u32>)],
    ) -> Result<(MxArray, Vec<MxArray>)> {
        self.0.park_mtp_target()?;
        let mut buckets = BTreeMap::<usize, Vec<usize>>::new();
        for (index, (_, ids)) in rows.iter().enumerate() {
            buckets.entry(ids.len()).or_default().push(index);
        }
        let mut outputs = (0..rows.len())
            .map(|_| None)
            .collect::<Vec<Option<MxArray>>>();
        for indices in buckets.values() {
            let selected = indices
                .iter()
                .map(|&index| rows[index].clone())
                .collect::<Vec<_>>();
            let snapshots = selected
                .iter()
                .map(|(row, _)| {
                    self.0.mtp_target_caches(row.seq_id).map(|caches| {
                        caches
                            .iter()
                            .map(|cache| match cache {
                                Qwen3_5LayerCache::FullAttention(_) => {
                                    Qwen3_5LayerSnapshot::FullAttention { offset: 0 }
                                }
                                // GDN replaces cache handles; it never mutates input arrays.
                                // Verify uses separate stacked caches. Retain graph references
                                // to live input state, avoiding a redundant snapshot copy.
                                Qwen3_5LayerCache::Linear(cache) => Qwen3_5LayerSnapshot::Linear {
                                    conv_state: cache.get(0).cloned(),
                                    recurrent_state: cache.get(1).cloned(),
                                },
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut scratch = Vec::new();
            for (layer, snapshot) in snapshots[0].iter().enumerate() {
                match snapshot {
                    Qwen3_5LayerSnapshot::FullAttention { .. } => {
                        scratch.push(Qwen3_5LayerCache::new_full_attention())
                    }
                    Qwen3_5LayerSnapshot::Linear { .. } => {
                        let caches = selected
                            .iter()
                            .map(|(row, _)| {
                                self.0
                                    .mtp_target_caches(row.seq_id)?
                                    .get(layer)
                                    .and_then(|cache| match cache {
                                        Qwen3_5LayerCache::Linear(cache) => Some(cache),
                                        _ => None,
                                    })
                                    .ok_or_else(|| {
                                        Error::from_reason("MTP owner cache layouts disagree")
                                    })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        scratch.push(Qwen3_5LayerCache::Linear(ArraysCache::stack_rows(&caches)?));
                    }
                }
            }
            let mut tapes = (0..scratch.len()).map(|_| None).collect::<Vec<_>>();
            let (logits, hidden) =
                self.0
                    .run_mtp_target_bucket(&selected, &mut scratch, &mut tapes)?;
            for (batch_row, ((row, ids), snapshot)) in selected.iter().zip(snapshots).enumerate() {
                let tape = tapes
                    .iter()
                    .map(|tape| {
                        tape.as_ref()
                            .map(|tape| tape.row(batch_row, selected.len()))
                            .transpose()
                    })
                    .collect::<Result<Vec<_>>>()?;
                let hidden = hidden.slice_axis(0, batch_row as i64, batch_row as i64 + 1)?;
                outputs[indices[batch_row]] = Some(
                    logits
                        .slice_axis(0, batch_row as i64, batch_row as i64 + 1)?
                        .transpose(Some(&[1, 0, 2]))?,
                );
                self.0.mtp_state_mut().pending.insert(
                    row.seq_id,
                    PendingOwner {
                        first_position: row.first_logical_position,
                        tokens: ids.clone(),
                        snapshot,
                        tape,
                        hidden,
                    },
                );
            }
        }
        let outputs = outputs
            .into_iter()
            .map(|value| value.ok_or_else(|| Error::from_reason("MTP verifier lost an output row")))
            .collect::<Result<Vec<_>>>()?;
        Ok((
            MxArray::concatenate_many(outputs.iter().collect(), Some(0))?,
            Vec::new(),
        ))
    }
    fn commit_target_state(&mut self, seq: u32, keep: usize) -> Result<()> {
        let pending = self
            .0
            .mtp_state_mut()
            .pending
            .remove(&seq)
            .ok_or_else(|| Error::from_reason("MTP commit has no pending target state"))?;
        if keep == 0 {
            return Ok(());
        }
        let frontier = pending
            .first_position
            .checked_add(keep as u32)
            .ok_or_else(|| Error::from_reason("MTP frontier overflow"))?;
        if self
            .0
            .mtp_adapter()
            .and_then(|adapter| adapter.current_token_count_for(seq))
            != Some(frontier)
        {
            return Err(Error::from_reason(
                "MTP attention/recurrent commit mismatch",
            ));
        }
        let mut caches = pending
            .snapshot
            .iter()
            .map(|state| match state {
                Qwen3_5LayerSnapshot::Linear { .. } => Qwen3_5LayerCache::new_linear(),
                _ => Qwen3_5LayerCache::new_full_attention(),
            })
            .collect::<Vec<_>>();
        replay_mtp_snapshot_to(
            &mut caches,
            &pending.snapshot,
            &pending.tape,
            keep,
            true,
            "scheduled MTP commit",
        )?;
        self.0.replace_mtp_target_caches(seq, caches)?;
        if let Some(mut owner) = self.0.mtp_state_mut().owners.remove(&seq) {
            if let Some(committed) = owner.committed_frontier {
                for cache in &mut owner.draft_caches {
                    if let Some(kv) = cache.as_kv_cache_mut() {
                        kv.trim(committed as i32);
                    }
                }
                let previous = owner
                    .hidden
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("MTP history commit has no seed"))?;
                let accepted = pending.hidden.slice_axis(1, 0, keep as i64 - 1)?;
                let hidden = MxArray::concatenate(previous, &accepted, 1)?;
                self.0.append_mtp_history(
                    &hidden,
                    &pending.tokens[..keep],
                    &mut owner.draft_caches,
                )?;
                owner.committed_frontier = Some(committed + keep);
            }
            owner.hidden = Some(pending.hidden.slice_axis(1, keep as i64 - 1, keep as i64)?);
            owner.frontier = frontier;
            owner.validate_committed_history()?;
            self.0.mtp_state_mut().owners.insert(seq, owner);
        }
        Ok(())
    }
    fn complete_target_states(&mut self, seq_ids: &[u32]) -> Result<()> {
        let mut arrays = Vec::new();
        for &seq in seq_ids {
            for cache in self.0.mtp_target_caches(seq)? {
                cache.collect_arrays(&mut arrays);
            }
            if let Some(owner) = self.0.mtp_state().owners.get(&seq) {
                if let Some(hidden) = &owner.hidden {
                    arrays.push(hidden);
                }
                for cache in &owner.draft_caches {
                    cache.collect_arrays(&mut arrays);
                }
            }
        }
        if arrays.is_empty() {
            Ok(())
        } else {
            MxArray::eval_arrays(&arrays)
        }
    }
    fn append_committed_taps(&mut self, _: u32, _: &[MxArray], _: usize) -> Result<()> {
        Ok(())
    }
    fn abort_target_state(&mut self, seq: u32) {
        self.0.mtp_state_mut().pending.remove(&seq);
    }
    fn discard_draft_owner(&mut self, seq: u32) {
        self.0.mtp_state_mut().owners.remove(&seq);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn copy_caches(snapshot: &[Qwen3_5LayerSnapshot]) -> Vec<Qwen3_5LayerCache> {
        snapshot
            .iter()
            .map(|state| match state {
                Qwen3_5LayerSnapshot::FullAttention { .. } => {
                    Qwen3_5LayerCache::new_full_attention()
                }
                Qwen3_5LayerSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                } => {
                    let mut cache = ArraysCache::new(2);
                    if let Some(array) = conv_state {
                        cache.set(0, array.clone()).unwrap();
                    }
                    if let Some(array) = recurrent_state {
                        cache.set(1, array.clone()).unwrap();
                    }
                    Qwen3_5LayerCache::Linear(cache)
                }
            })
            .collect()
    }
    fn values(caches: &[Qwen3_5LayerCache]) -> Vec<Vec<f32>> {
        let mut arrays = Vec::new();
        for cache in caches {
            cache.collect_arrays(&mut arrays);
        }
        arrays
            .iter()
            .map(|array| {
                array
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec()
            })
            .collect()
    }
    fn close(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}");
        let max = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let scale = expected.iter().map(|v| v.abs()).fold(1.0_f32, f32::max);
        assert!(
            actual.iter().chain(expected).all(|v| v.is_finite()) && max <= scale * 0.025,
            "{label}: max error {max} scale {scale}"
        );
    }

    /// Compare owner-packed verification and accepted-prefix state with
    /// independent singleton target calls from the same materialized inputs.
    /// Includes equal/unequal widths, partial/full/zero commits and an AR peer.
    pub(crate) fn owner_replay_matches_independent<T: ScheduledMtpTarget>(target: &mut T) {
        target.park_mtp_target().unwrap();
        let seqs = [101, 202];
        let positions = seqs.map(|seq| target.mtp_state().owners[&seq].frontier);
        let snapshots = seqs
            .iter()
            .map(|&seq| {
                super::super::layer_cache::snapshot_all_mtp(
                    target.mtp_target_caches(seq).unwrap(),
                    true,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let initial = snapshots
            .iter()
            .map(|snapshot| values(&copy_caches(snapshot)))
            .collect::<Vec<_>>();
        let seeds = seqs
            .iter()
            .map(|seq| {
                target.mtp_state().owners[seq]
                    .hidden
                    .as_ref()
                    .unwrap()
                    .clone()
            })
            .collect::<Vec<_>>();
        for (widths, keeps, mixed) in [
            ([4, 4], [1, 3], false),
            ([4, 4], [4, 0], false),
            ([4, 1], [2, 1], true),
        ] {
            for (index, &seq) in seqs.iter().enumerate() {
                let adapter = target.mtp_adapter_mut().unwrap();
                let count = adapter.current_token_count_for(seq).unwrap();
                adapter
                    .rollback_last_tokens_for(seq, count - positions[index])
                    .unwrap();
                target
                    .replace_mtp_target_caches(seq, copy_caches(&snapshots[index]))
                    .unwrap();
                target.mtp_state_mut().release(seq);
                if !(mixed && index == 1) {
                    // This target-only oracle injects a seed without testing
                    // draft history. Production warm starts decline MTP.
                    assert!(target.begin_scheduled_mtp(seq, 0).unwrap());
                    target
                        .seed_scheduled_mtp(seq, positions[index], seeds[index].clone())
                        .unwrap();
                }
            }
            let rows = seqs
                .iter()
                .enumerate()
                .map(|(index, &seq_id)| ScheduledVerifyRow {
                    seq_id,
                    first_position: positions[index],
                    tokens: (0..widths[index])
                        .map(|offset| 19 + index as u32 * 8 + offset as u32)
                        .collect(),
                    speculative: !(mixed && index == 1),
                })
                .collect::<Vec<_>>();
            let batched = target
                .verify_scheduled_mtp(&rows)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec();
            for (index, &seq) in seqs.iter().enumerate() {
                assert_eq!(
                    values(target.mtp_target_caches(seq).unwrap()),
                    initial[index],
                    "verification mutated a live GDN input"
                );
            }
            let commits = seqs
                .iter()
                .enumerate()
                .map(|(index, &seq_id)| ScheduledVerifyCommit {
                    seq_id,
                    keep: keeps[index],
                })
                .collect::<Vec<_>>();
            assert!(
                target
                    .commit_scheduled_mtp(&commits)
                    .unwrap()
                    .iter()
                    .all(Result::is_ok)
            );
            let committed = seqs
                .iter()
                .map(|&seq| values(target.mtp_target_caches(seq).unwrap()))
                .collect::<Vec<_>>();
            let vocab = batched.len() / widths.iter().sum::<usize>();
            let mut offset = 0;
            for (index, row) in rows.iter().enumerate() {
                assert_eq!(
                    target
                        .mtp_adapter()
                        .unwrap()
                        .current_token_count_for(row.seq_id),
                    Some(positions[index] + keeps[index] as u32)
                );
                let mut scratch = copy_caches(&snapshots[index]);
                let mut tape = (0..scratch.len()).map(|_| None).collect::<Vec<_>>();
                // Keep the original allocation but restore metadata for the
                // independent full-width forward before inspecting its state.
                let adapter = target.mtp_adapter_mut().unwrap();
                adapter
                    .rollback_last_tokens_for(row.seq_id, keeps[index] as u32)
                    .unwrap();
                let logits = target
                    .run_reference_mtp_target(row.seq_id, &row.tokens, &mut scratch, &mut tape)
                    .unwrap();
                let logits = logits
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec();
                close(
                    &batched[offset..offset + logits.len()],
                    &logits,
                    "batched target logits",
                );
                offset += logits.len();
                if keeps[index] > 0 {
                    replay_mtp_snapshot_to(
                        &mut scratch,
                        &snapshots[index],
                        &tape,
                        keeps[index],
                        true,
                        "independent MTP test",
                    )
                    .unwrap();
                } else {
                    scratch = copy_caches(&snapshots[index]);
                }
                for (actual, expected) in committed[index].iter().zip(values(&scratch)) {
                    close(actual, &expected, "owner recurrent prefix");
                }
                assert_eq!(logits.len(), widths[index] * vocab);
            }
        }
    }
}
