//! Request-owned recurrent state for hybrid continuous batching.
//!
//! Unlike attention K/V, a GDN/Mamba state or a rotating-window snapshot is
//! one constant-size unit per live request.  Keeping those units in a table
//! keyed by scheduler sequence prevents the model's historical singleton
//! cache from leaking between interleaved rows.  Prefix checkpoints retain one
//! complete unit at a token boundary; lookup walks deepest-first and refuses a
//! state published by a peer in the current scheduler step.

use std::collections::{BTreeMap, VecDeque};

use crate::transformer::paged_kv_cache_adapter::SeqId;

/// Maximum live recurrent units for the Stage-2 hybrid lane.
pub(crate) const HYBRID_LIVE_STATE_UNITS: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RecurrentStateIdentity {
    pub owner_id: u64,
    pub boundary: u32,
    pub final_block_hash: u64,
}

struct LiveState<S> {
    bytes: u64,
    state: S,
}

struct Checkpoint<S> {
    identity: RecurrentStateIdentity,
    published_step: u64,
    bytes: u64,
    state: S,
}

/// Bounded request state plus a deepest-boundary prefix side table.
pub(crate) struct RecurrentStateTable<S> {
    live: BTreeMap<SeqId, LiveState<S>>,
    checkpoints: VecDeque<Checkpoint<S>>,
    max_live_units: usize,
    max_checkpoint_units: usize,
}

impl<S> RecurrentStateTable<S> {
    pub(crate) fn new(max_live_units: usize, max_checkpoint_units: usize) -> Result<Self, String> {
        if max_live_units == 0 {
            return Err("recurrent-state live-unit cap must be positive".to_string());
        }
        Ok(Self {
            live: BTreeMap::new(),
            checkpoints: VecDeque::new(),
            max_live_units,
            max_checkpoint_units,
        })
    }

    pub(crate) fn stage2() -> Self {
        Self::new(HYBRID_LIVE_STATE_UNITS, HYBRID_LIVE_STATE_UNITS)
            .expect("Stage-2 recurrent-state limits are valid")
    }

    pub(crate) fn live_len(&self) -> usize {
        self.live.len()
    }

    pub(crate) fn live_bytes(&self) -> u64 {
        self.live
            .values()
            .fold(0u64, |total, entry| total.saturating_add(entry.bytes))
    }

    pub(crate) fn contains_live(&self, seq_id: SeqId) -> bool {
        self.live.contains_key(&seq_id)
    }

    pub(crate) fn can_insert_live(&self, seq_id: SeqId) -> bool {
        self.contains_live(seq_id) || self.live.len() < self.max_live_units
    }

    pub(crate) fn insert_live(
        &mut self,
        seq_id: SeqId,
        bytes: u64,
        state: S,
    ) -> Result<Option<S>, String> {
        if bytes == 0 {
            return Err(format!(
                "sequence {seq_id}: recurrent-state reservation must be positive"
            ));
        }
        let replacing = self.live.contains_key(&seq_id);
        if !replacing && self.live.len() >= self.max_live_units {
            return Err(format!(
                "sequence {seq_id}: recurrent-state live-unit cap {} reached",
                self.max_live_units
            ));
        }
        Ok(self
            .live
            .insert(seq_id, LiveState { bytes, state })
            .map(|entry| entry.state))
    }

    pub(crate) fn live(&self, seq_id: SeqId) -> Option<&S> {
        self.live.get(&seq_id).map(|entry| &entry.state)
    }

    pub(crate) fn live_mut(&mut self, seq_id: SeqId) -> Option<&mut S> {
        self.live.get_mut(&seq_id).map(|entry| &mut entry.state)
    }

    pub(crate) fn take_live(&mut self, seq_id: SeqId) -> Option<S> {
        self.live.remove(&seq_id).map(|entry| entry.state)
    }

    pub(crate) fn remove_live(&mut self, seq_id: SeqId) -> bool {
        self.live.remove(&seq_id).is_some()
    }

    /// Publish one complete state unit. A boundary has exactly one state: a
    /// newer publication with the same identity replaces the older one.
    ///
    /// Qwen3.5 and Gemma4 currently publish their durable tensors through the
    /// family sidecar codecs; this in-memory form is the model-neutral contract
    /// pinned by the table tests and becomes live when those stores converge.
    #[allow(dead_code)]
    pub(crate) fn publish_checkpoint(
        &mut self,
        identity: RecurrentStateIdentity,
        published_step: u64,
        bytes: u64,
        state: S,
    ) {
        if self.max_checkpoint_units == 0 || bytes == 0 || identity.boundary == 0 {
            return;
        }
        if let Some(index) = self
            .checkpoints
            .iter()
            .position(|entry| entry.identity == identity)
        {
            self.checkpoints.remove(index);
        }
        self.checkpoints.push_back(Checkpoint {
            identity,
            published_step,
            bytes,
            state,
        });
        while self.checkpoints.len() > self.max_checkpoint_units {
            self.checkpoints.pop_front();
        }
    }

    /// Find the deepest exact lineage boundary, walking right-to-left.
    /// Checkpoints created in `scheduler_step` are deliberately invisible so
    /// one request cannot consume a peer's just-produced recurrent state in
    /// the same model forward.
    #[allow(dead_code)]
    pub(crate) fn deepest_checkpoint(
        &self,
        owner_id: u64,
        candidates: &[(u32, u64)],
        scheduler_step: u64,
    ) -> Option<(RecurrentStateIdentity, &S)> {
        candidates
            .iter()
            .rev()
            .find_map(|&(boundary, final_block_hash)| {
                self.checkpoints
                    .iter()
                    .rev()
                    .find(|entry| {
                        entry.identity.owner_id == owner_id
                            && entry.identity.boundary == boundary
                            && entry.identity.final_block_hash == final_block_hash
                            && entry.published_step < scheduler_step
                    })
                    .map(|entry| (entry.identity, &entry.state))
            })
    }

    #[allow(dead_code)]
    pub(crate) fn checkpoint_bytes(&self) -> u64 {
        self.checkpoints
            .iter()
            .fold(0u64, |total, entry| total.saturating_add(entry.bytes))
    }
}

/// A hybrid prefix is usable only through the deepest boundary every cache
/// group backs. Empty input means no reusable hybrid prefix.
pub(crate) fn reconcile_hybrid_prefix(group_hits: &[u32]) -> u32 {
    group_hits.iter().copied().min().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_table_caps_two_units_and_keeps_rows_isolated() {
        let mut table = RecurrentStateTable::stage2();
        table.insert_live(1, 10, vec![11]).unwrap();
        table.insert_live(2, 20, vec![22]).unwrap();
        assert_eq!(table.live_len(), 2);
        assert_eq!(table.live_bytes(), 30);
        assert_eq!(table.live(1), Some(&vec![11]));
        assert_eq!(table.live(2), Some(&vec![22]));
        assert!(table.insert_live(3, 30, vec![33]).is_err());
        table.live_mut(1).unwrap().push(12);
        assert_eq!(table.live(1), Some(&vec![11, 12]));
        assert_eq!(table.live(2), Some(&vec![22]));
    }

    #[test]
    fn deepest_checkpoint_is_exact_and_not_visible_in_the_same_step() {
        let mut table = RecurrentStateTable::new(2, 4).unwrap();
        table.publish_checkpoint(
            RecurrentStateIdentity {
                owner_id: 7,
                boundary: 16,
                final_block_hash: 0x16,
            },
            4,
            8,
            "sixteen",
        );
        table.publish_checkpoint(
            RecurrentStateIdentity {
                owner_id: 7,
                boundary: 64,
                final_block_hash: 0x64,
            },
            5,
            8,
            "sixty-four",
        );
        let candidates = [(16, 0x16), (64, 0x64)];
        let (identity, state) = table.deepest_checkpoint(7, &candidates, 6).unwrap();
        assert_eq!(identity.boundary, 64);
        assert_eq!(*state, "sixty-four");

        let (identity, _) = table.deepest_checkpoint(7, &candidates, 5).unwrap();
        assert_eq!(
            identity.boundary, 16,
            "step-5 publication is not peer-visible"
        );
        assert!(table.deepest_checkpoint(8, &candidates, 6).is_none());
        assert!(
            table
                .deepest_checkpoint(7, &[(16, 0xdead), (64, 0xbeef)], 6)
                .is_none()
        );
    }

    #[test]
    fn hybrid_hit_is_the_minimum_group_agreement() {
        assert_eq!(reconcile_hybrid_prefix(&[4096, 1024, 2048]), 1024);
        assert_eq!(reconcile_hybrid_prefix(&[]), 0);
    }
}
