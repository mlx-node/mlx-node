//! Request-owned recurrent state for hybrid continuous batching.
//!
//! Unlike attention K/V, a GDN/Mamba state or a rotating-window snapshot is
//! one constant-size unit per live request.  Keeping those units in a table
//! keyed by scheduler sequence prevents the model's historical singleton
//! cache from leaking between interleaved rows. Durable prefix checkpoints are
//! owned by each family's sidecar codec rather than duplicated in this table.

use std::collections::BTreeMap;

use crate::transformer::paged_kv_cache_adapter::SeqId;

/// Maximum live recurrent units for the Stage-2 hybrid lane.
pub(crate) const HYBRID_LIVE_STATE_UNITS: usize = 2;

struct LiveState<S> {
    bytes: u64,
    state: S,
}

/// Bounded request-owned recurrent state.
pub(crate) struct RecurrentStateTable<S> {
    live: BTreeMap<SeqId, LiveState<S>>,
    max_live_units: usize,
}

impl<S> RecurrentStateTable<S> {
    pub(crate) fn stage2() -> Self {
        Self {
            live: BTreeMap::new(),
            max_live_units: HYBRID_LIVE_STATE_UNITS,
        }
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
    fn hybrid_hit_is_the_minimum_group_agreement() {
        assert_eq!(reconcile_hybrid_prefix(&[4096, 1024, 2048]), 1024);
        assert_eq!(reconcile_hybrid_prefix(&[]), 0);
    }
}
