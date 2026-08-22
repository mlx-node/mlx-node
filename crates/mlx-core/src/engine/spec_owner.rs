//! Owner addressing for per-turn speculative state.
//!
//! # I6 — speculative state dies with its owner
//!
//! A speculative turn writes rows into a cache the MODEL owns: the paged
//! adapter, the drafter's K/V, the GDN tape. That cache outlives the turn and
//! can be re-pointed at another request between turns. So a turn must address
//! it by the sequence it OPENED on, never by "whatever the cache currently
//! says is active" — the two agree exactly until they do not, and the moment
//! they disagree the cache silently accepts a dead owner's writes.
//!
//! [`SpecOwner`] is that address. It is claimed once at turn entry, it is the
//! only way the turn reaches the cache afterwards, and every resolution
//! re-checks the cache still belongs to it. A stepper therefore does not need
//! to MOVE the cache out of the model to keep it safe for the turn's
//! duration: refusal at the seam is what makes the borrow-back sound.
//!
//! The refusals here are fail-closed: `Err` at the call boundary, never a
//! silent fallback to the cache's current owner.

use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};

/// The sequence a speculative turn's state belongs to, claimed at turn entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpecOwner(SeqId);

impl SpecOwner {
    /// Claim the cache's currently active sequence as this turn's owner.
    ///
    /// `None` is refused rather than deferred: a paged speculative turn with
    /// no addressable sequence has nowhere to write, and failing here leaves
    /// the cache untouched instead of half-advanced by a Step-A forward that
    /// only explodes at the verify write.
    pub(crate) fn claim(active: Option<SeqId>, context: &str) -> Result<Self, String> {
        match active {
            Some(seq_id) => Ok(Self(seq_id)),
            None => Err(format!(
                "{context}: the paged cache names no active request"
            )),
        }
    }

    pub(crate) fn seq_id(self) -> SeqId {
        self.0
    }

    /// Refuse a call addressed to any sequence but this owner's.
    pub(crate) fn accepts(self, seq_id: SeqId, context: &str) -> Result<(), String> {
        if seq_id == self.0 {
            return Ok(());
        }
        Err(format!(
            "{context}: sequence {seq_id} is not this turn's owner ({})",
            self.0
        ))
    }

    /// Refuse a cache whose active request is no longer this owner's. The
    /// single predicate both resolve paths apply, so the read and write
    /// borrows can never drift apart on which cache they accept.
    pub(crate) fn matches(self, active: Option<SeqId>, context: &str) -> Result<(), String> {
        if active == Some(self.0) {
            return Ok(());
        }
        Err(format!(
            "{context}: sequence {} is no longer the paged cache's active request \
             ({active:?})",
            self.0
        ))
    }

    /// Borrow the owner's paged adapter back out of the model's slot.
    ///
    /// Takes the slot rather than the model so a caller can hold it alongside
    /// disjoint `&mut` borrows of the model's other fields — the reason the
    /// adapter never has to be moved into the stepper.
    pub(crate) fn resolve<'a>(
        self,
        slot: &'a mut Option<PagedKVCacheAdapter>,
        context: &str,
    ) -> Result<&'a mut PagedKVCacheAdapter, String> {
        let adapter = slot
            .as_mut()
            .ok_or_else(|| format!("{context}: sequence {}'s paged cache is gone", self.0))?;
        self.matches(adapter.active_seq_id(), context)?;
        Ok(adapter)
    }

    /// Read-only twin of [`Self::resolve`].
    pub(crate) fn resolve_ref<'a>(
        self,
        slot: &'a Option<PagedKVCacheAdapter>,
        context: &str,
    ) -> Result<&'a PagedKVCacheAdapter, String> {
        let adapter = slot
            .as_ref()
            .ok_or_else(|| format!("{context}: sequence {}'s paged cache is gone", self.0))?;
        self.matches(adapter.active_seq_id(), context)?;
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::SpecOwner;

    #[test]
    fn claim_refuses_a_cache_with_no_active_request() {
        let err = SpecOwner::claim(None, "ctx").expect_err("no active request must refuse");
        assert!(err.contains("no active request"), "{err}");
        assert_eq!(
            SpecOwner::claim(Some(7), "ctx").expect("claimed").seq_id(),
            7
        );
    }

    #[test]
    fn accepts_only_the_claimed_sequence() {
        let owner = SpecOwner::claim(Some(7), "ctx").expect("claimed");
        owner
            .accepts(7, "ctx")
            .expect("the owner's own id is accepted");
        let err = owner
            .accepts(8, "ctx")
            .expect_err("another sequence must be refused");
        assert!(err.contains("not this turn's owner"), "{err}");
    }

    /// The cache-side half of I6: a cache that has moved on to another
    /// request — or released the one it had — is refused, never written
    /// through. Both resolve paths apply exactly this predicate.
    #[test]
    fn matches_refuses_a_cache_that_moved_on() {
        let owner = SpecOwner::claim(Some(7), "ctx").expect("claimed");
        owner
            .matches(Some(7), "ctx")
            .expect("the owner's own cache is accepted");
        for moved in [None, Some(8)] {
            let err = owner
                .matches(moved, "ctx")
                .expect_err("a cache that moved on must be refused");
            assert!(
                err.contains("no longer the paged cache's active request"),
                "{err}"
            );
        }
    }

    #[test]
    fn resolve_refuses_an_empty_slot() {
        let owner = SpecOwner::claim(Some(7), "ctx").expect("claimed");
        let mut slot = None;
        let err = owner
            .resolve(&mut slot, "ctx")
            .err()
            .expect("a vacated slot must refuse");
        assert!(err.contains("paged cache is gone"), "{err}");
        let err = owner
            .resolve_ref(&slot, "ctx")
            .err()
            .expect("a vacated slot must refuse the read path too");
        assert!(err.contains("paged cache is gone"), "{err}");
    }
}
