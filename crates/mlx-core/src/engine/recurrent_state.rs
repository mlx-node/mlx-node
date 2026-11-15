//! Request-owned recurrent state for hybrid continuous batching.
//!
//! Unlike attention K/V, a GDN/Mamba state or a rotating-window snapshot is
//! one constant-size unit per live request.  Keeping those units in a table
//! keyed by scheduler sequence prevents the model's historical singleton
//! cache from leaking between interleaved rows. Durable prefix checkpoints are
//! owned by each family's sidecar codec rather than duplicated in this table.

use std::collections::BTreeMap;

use crate::transformer::paged_kv_cache_adapter::SeqId;

/// Most live recurrent units, and so decode lanes, a hybrid load holds.
/// The load's memory plan grants 1 to this many from the device's memory.
pub(crate) const HYBRID_LIVE_STATE_UNITS: usize = 8;

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
    pub(crate) fn new(max_live_units: usize) -> Self {
        Self {
            live: BTreeMap::new(),
            max_live_units: max_live_units.max(1),
        }
    }

    pub(crate) fn max_live_units(&self) -> usize {
        self.max_live_units
    }

    /// Lowers or raises the cap; rows already live stay.
    pub(crate) fn set_max_live_units(&mut self, max_live_units: usize) {
        self.max_live_units = max_live_units.max(1);
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
    fn live_table_caps_eight_units_and_keeps_rows_isolated() {
        assert_eq!(HYBRID_LIVE_STATE_UNITS, 8);
        let mut table = RecurrentStateTable::new(HYBRID_LIVE_STATE_UNITS);
        for seq in 1..=8u32 {
            assert!(table.can_insert_live(seq));
            table
                .insert_live(seq, u64::from(seq) * 10, vec![seq * 11])
                .unwrap();
        }
        assert_eq!(table.live_len(), 8);
        assert_eq!(table.live_bytes(), 360);
        for seq in 1..=8u32 {
            assert_eq!(table.live(seq), Some(&vec![seq * 11]));
        }
        assert!(!table.can_insert_live(9));
        assert!(table.insert_live(9, 90, vec![99]).is_err());
        assert!(table.can_insert_live(8), "a live row replaces itself");
        table.live_mut(1).unwrap().push(12);
        assert_eq!(table.live(1), Some(&vec![11, 12]));
        assert_eq!(table.live(2), Some(&vec![22]));
    }

    /// A finished or cancelled row frees its unit for the next one.
    #[test]
    fn a_released_unit_admits_the_next_row() {
        let mut table = RecurrentStateTable::new(HYBRID_LIVE_STATE_UNITS);
        for seq in 1..=8u32 {
            table.insert_live(seq, 10, seq).unwrap();
        }
        assert!(table.insert_live(9, 10, 9).is_err());
        assert!(table.remove_live(2));
        assert!(table.can_insert_live(9));
        table.insert_live(9, 10, 9).unwrap();
        assert!(table.insert_live(10, 10, 10).is_err());
        assert_eq!(table.take_live(3), Some(3));
        table.insert_live(10, 10, 10).unwrap();
        assert_eq!(table.live_len(), 8);
    }

    /// A plan that grants fewer lanes caps the table there.
    #[test]
    fn a_lowered_cap_refuses_rows_past_it() {
        let mut table = RecurrentStateTable::new(HYBRID_LIVE_STATE_UNITS);
        table.set_max_live_units(2);
        table.insert_live(1, 10, 1).unwrap();
        table.insert_live(2, 10, 2).unwrap();
        assert!(table.insert_live(3, 10, 3).is_err());
        assert_eq!(RecurrentStateTable::<u32>::new(0).max_live_units(), 1);
    }

    #[test]
    fn hybrid_hit_is_the_minimum_group_agreement() {
        assert_eq!(reconcile_hybrid_prefix(&[4096, 1024, 2048]), 1024);
        assert_eq!(reconcile_hybrid_prefix(&[]), 0);
    }
}
