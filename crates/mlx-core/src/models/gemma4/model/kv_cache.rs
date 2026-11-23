//! Grouped block-paged KV cache: re-exports the shared [`HybridKVCacheCoordinator`]
//! and [`PruneOnlySpecPagedCache`] plus Gemma4's family-scope [`Gemma4SpecPagedCache`].

use crate::engine::backend::SpecFrontier;
use crate::engine::spec_paged::SpecPagedCache;

use super::Gemma4Inner;

pub(crate) use crate::transformer::{
    HybridKVCacheCoordinator as Gemma4KVCacheCoordinator, PruneOnlySpecPagedCache,
};

/// The paged speculative cache contract (`engine::spec_paged`) at FAMILY scope,
/// and the one a gemma4 paged speculative driver holds.
///
/// The four cycle primitives are the coordinator's, delegated to
/// [`PruneOnlySpecPagedCache`] so the cycle arithmetic stays single-sourced;
/// the settle is the family's ([`Gemma4Inner::settle_grouped_kv_step_at`]),
/// which walks the cold-checkpoint rungs the coordinator cannot reach before
/// running that same committed-basis prune. The rung walk is what makes this
/// settle irreversible, so [`SpecPagedCache::settle_captures_durable_state`]
/// answers `true` and L-SETTLE keeps it out of an open cycle entirely.
pub(crate) struct Gemma4SpecPagedCache<'a>(&'a mut Gemma4Inner);

#[allow(dead_code)]
impl<'a> Gemma4SpecPagedCache<'a> {
    pub(crate) fn new(inner: &'a mut Gemma4Inner) -> Self {
        Self(inner)
    }

    pub(crate) fn model(&self) -> &Gemma4Inner {
        self.0
    }

    fn coordinator_cache(&mut self) -> std::result::Result<PruneOnlySpecPagedCache<'_>, String> {
        self.0
            .kv_cache_coordinator
            .as_mut()
            .map(PruneOnlySpecPagedCache::new)
            .ok_or_else(|| "Gemma4 hybrid KV coordinator missing".to_string())
    }
}

impl SpecPagedCache for Gemma4SpecPagedCache<'_> {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> std::result::Result<bool, String> {
        self.coordinator_cache()?.reserve_lookahead(seq_id, rows)
    }

    fn record_rows(&mut self, seq_id: u32, tokens: &[u32]) -> std::result::Result<(), String> {
        self.coordinator_cache()?.record_rows(seq_id, tokens)
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> std::result::Result<(), String> {
        self.coordinator_cache()?.rollback_rows(seq_id, rows)
    }

    fn settle_committed(
        &mut self,
        seq_id: u32,
        committed_tokens: u64,
    ) -> std::result::Result<(), String> {
        let committed = u32::try_from(committed_tokens).map_err(|_| {
            format!("Gemma4 committed frontier {committed_tokens} does not fit u32")
        })?;
        self.0
            .settle_grouped_kv_step_at(seq_id, committed)
            .map_err(|error| error.reason.clone())
    }

    fn settle_captures_durable_state(&self) -> bool {
        // The family settle walks the cold-checkpoint rungs, and a captured
        // checkpoint is not retractable by a rollback.
        true
    }

    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.0.kv_cache_coordinator.as_ref()?.spec_frontier(seq_id)
    }
}
