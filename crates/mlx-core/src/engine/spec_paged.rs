//! Family-neutral contract for paged speculative KV bookkeeping.
//!
//! One speculative verify cycle touches the paged cache in exactly four
//! moves — reserve the lookahead region, write the verify rows, commit the
//! accepted prefix, settle durable state — and [`SpecPagedCache`] names them
//! once, so every family (dense/MoE native MTP, gemma4 DSpark, muse DFlash,
//! gemma4 assistant) lands on the same call order instead of re-deriving it
//! against raw adapter/coordinator surfaces.
//!
//! # Not ported: vLLM's per-group metadata capture
//!
//! vLLM snapshots per-kv-cache-group attention metadata and swaps drafter
//! block tables around its verify pass (`gpu_model_runner.py:2662-2670`,
//! `gemma4.py:62-103`) because its verify runs outside the module that owns
//! the group routing. That capture is unnecessary here **by construction**:
//! our verify forward runs inside the family's grouped layer loop
//! (`gemma4::run_paged_prefill_layer_loop`, `muse_glimmer::run_paged_layer_loop`,
//! the qwen3_5 paged verify cores), which routes each layer's cache group
//! through the coordinator internally — no out-of-band per-group view ever
//! exists that could go stale or need capturing.

use crate::engine::backend::SpecFrontier;

/// Facade over the paged-KV bookkeeping one speculative verify cycle needs.
///
/// Implemented by the per-family paged cache owners (the qwen3_5 dense/MoE
/// adapter-backed steppers, `Gemma4KVCacheCoordinator` for gemma4 and muse).
/// The engine-side speculative loops (`run_mtp_turn`, `run_dspark_turn`)
/// stay on their existing stepper traits; this contract is what those
/// steppers' paged modes are written against, so the ordering laws below
/// hold for every family by conformance rather than by convention.
///
/// # L-SETTLE (I9) — settle at the committed frontier, never the cursor
///
/// Between a sequence's [`Self::record_verify`] and its
/// [`Self::commit_cycle`], NO settle work may run for that sequence: no
/// sliding-window prune, no cold/durable checkpoint, no sidecar capture, no
/// prefix-block registration. In that gap the write cursor sits up to the
/// whole lookahead ahead of the committed frontier, so any of those
/// consumers would persist or prune against rows a rollback can still
/// retract. Settle runs only through [`Self::settle_committed`], after the
/// commit, at the committed length the commit landed on. [`Self::frontier`]
/// (and the I10 rollback-as-cursor-arithmetic it certifies) is valid iff
/// this law holds.
///
/// # L-EPILOGUE (I11) — one epilogue
///
/// Every paged speculative turn exits only through the family's
/// `finish_paged_turn` shape (reconcile → finalize → save,
/// `engine/paged_turn.rs`); prefix registration hard-requires
/// `request_tokens.len() == num_tokens`, which keeps the
/// never-persist-the-unverified cap (I3) single-sourced. Neither an
/// implementation of this trait nor a caller may fork a private epilogue —
/// forking one re-opens the GDN-seam bug class Stage A closed.
#[allow(dead_code)]
pub(crate) trait SpecPagedCache {
    /// Reserve block capacity for `rows` rows past `seq_id`'s current
    /// committed frontier WITHOUT advancing any token cursor — the
    /// speculative lookahead region. Called once per cycle: every committed
    /// token moves the frontier, so a turn-entry reservation only covers the
    /// first cycle.
    ///
    /// `Ok(true)` = the coming verify write is covered. `Ok(false)` = pool
    /// exhaustion with untouched state; the caller skips the cycle and
    /// decodes autoregressively instead of erroring the turn. Non-capacity
    /// failures are `Err`.
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> Result<bool, String>;

    /// Record the cycle's verify write `[anchor, drafts..]` at the
    /// optimistic write cursor, in pre-reserved blocks. Atomic per slice: on
    /// failure no partial suffix of `tokens` stays recorded.
    fn record_verify(&mut self, seq_id: u32, tokens: &[u32]) -> Result<(), String>;

    /// Land the cycle on its committed frontier: of the `total` rows the
    /// verify wrote, keep the accepted prefix of `keep` and roll back
    /// exactly [`commit_rollback_rows`]`(keep, total)` rows. Rollback is
    /// cursor arithmetic (I10): paged positions are absolute and sliding
    /// groups null-prune instead of wrapping, so no block is freed and the
    /// next record overwrites in place.
    fn commit_cycle(&mut self, seq_id: u32, keep: usize, total: usize) -> Result<(), String>;

    /// Settle durable/derived cache state at the COMMITTED frontier the
    /// last commit landed on: sliding-window prune, cold checkpoints,
    /// pending-write eval — the family's existing per-step settle,
    /// re-anchored from the write cursor to `committed_tokens`. The only
    /// legal settle entry point during a speculative turn (L-SETTLE).
    fn settle_committed(&mut self, seq_id: u32, committed_tokens: u64) -> Result<(), String>;

    /// The one frontier every cache group agrees on for `seq_id`, or `None`
    /// when the cache cannot name one (unknown sequence, or groups
    /// disagreeing mid-flight). This is the paged survival of the flat
    /// snapshot/commit transaction's all-groups-advanced-exactly assert;
    /// the engine's post-rollback debug asserts consume it through the
    /// stepper `frontier()` hooks.
    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier>;
}

/// Rows a commit must roll back: the verify wrote `total` rows, acceptance
/// kept `keep`, so exactly `total - keep` come off the cursor. The single
/// source for the commit arithmetic — implementations of
/// [`SpecPagedCache::commit_cycle`] call this rather than re-deriving it.
#[allow(dead_code)]
pub(crate) fn commit_rollback_rows(keep: usize, total: usize) -> usize {
    assert!(
        keep <= total,
        "commit_cycle kept {keep} of only {total} verify rows — keep may never exceed total"
    );
    total - keep
}

/// Executable form of L-SETTLE: wraps any [`SpecPagedCache`] and refuses a
/// `settle_committed` issued between a sequence's `record_verify` and its
/// `commit_cycle`, instead of letting the early settle silently prune or
/// checkpoint at the optimistic cursor. Conformance harness for facade
/// implementations and their drivers (the cross-module gates run the real
/// coordinator through it with the mock-verified call order).
#[allow(dead_code)]
pub(crate) struct SettleOrderChecked<C> {
    inner: C,
    /// Sequences with a recorded verify write whose commit has not run yet.
    open_cycles: std::collections::HashSet<u32>,
}

#[allow(dead_code)]
impl<C> SettleOrderChecked<C> {
    pub(crate) fn new(inner: C) -> Self {
        Self {
            inner,
            open_cycles: std::collections::HashSet::new(),
        }
    }

    pub(crate) fn inner(&self) -> &C {
        &self.inner
    }

    pub(crate) fn into_inner(self) -> C {
        self.inner
    }
}

impl<C: SpecPagedCache> SpecPagedCache for SettleOrderChecked<C> {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> Result<bool, String> {
        self.inner.reserve_lookahead(seq_id, rows)
    }

    fn record_verify(&mut self, seq_id: u32, tokens: &[u32]) -> Result<(), String> {
        self.inner.record_verify(seq_id, tokens)?;
        self.open_cycles.insert(seq_id);
        Ok(())
    }

    fn commit_cycle(&mut self, seq_id: u32, keep: usize, total: usize) -> Result<(), String> {
        self.inner.commit_cycle(seq_id, keep, total)?;
        self.open_cycles.remove(&seq_id);
        Ok(())
    }

    fn settle_committed(&mut self, seq_id: u32, committed_tokens: u64) -> Result<(), String> {
        if self.open_cycles.contains(&seq_id) {
            return Err(format!(
                "L-SETTLE violation: settle_committed(seq={seq_id}) between record_verify and \
                 commit_cycle — settle may only run post-commit at the committed frontier"
            ));
        }
        self.inner.settle_committed(seq_id, committed_tokens)
    }

    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.inner.frontier(seq_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference conformer: a bare recorded-row cursor per the I10 model
    /// (absolute positions, rollback = cursor subtraction), plus a call log
    /// so ordering tests can assert what reached the cache.
    #[derive(Default)]
    struct MockCursorCache {
        cursor: u64,
        calls: Vec<String>,
    }

    impl SpecPagedCache for MockCursorCache {
        fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> Result<bool, String> {
            self.calls.push(format!("reserve({seq_id},{rows})"));
            Ok(true)
        }

        fn record_verify(&mut self, seq_id: u32, tokens: &[u32]) -> Result<(), String> {
            self.calls
                .push(format!("record({seq_id},{})", tokens.len()));
            self.cursor += tokens.len() as u64;
            Ok(())
        }

        fn commit_cycle(&mut self, seq_id: u32, keep: usize, total: usize) -> Result<(), String> {
            self.calls.push(format!("commit({seq_id},{keep},{total})"));
            let rolled_back = commit_rollback_rows(keep, total) as u64;
            self.cursor = self
                .cursor
                .checked_sub(rolled_back)
                .ok_or_else(|| format!("rollback of {rolled_back} rows underflows the cursor"))?;
            Ok(())
        }

        fn settle_committed(&mut self, seq_id: u32, committed_tokens: u64) -> Result<(), String> {
            self.calls
                .push(format!("settle({seq_id},{committed_tokens})"));
            Ok(())
        }

        fn frontier(&self, _seq_id: u32) -> Option<SpecFrontier> {
            Some(SpecFrontier {
                attn_tokens: self.cursor,
                recurrent_tokens: None,
            })
        }
    }

    #[test]
    fn spec_paged_commit_is_exact_cursor_arithmetic() {
        let mut cache = MockCursorCache::default();

        // Partial accept: 5 verify rows (anchor + 4 drafts), keep 3.
        cache.record_verify(7, &[10, 11, 12, 13, 14]).unwrap();
        assert_eq!(cache.cursor, 5);
        cache.commit_cycle(7, 3, 5).unwrap();
        assert_eq!(cache.cursor, 3, "commit(3,5) must roll back exactly 2 rows");
        assert_eq!(
            cache.frontier(7),
            Some(SpecFrontier {
                attn_tokens: 3,
                recurrent_tokens: None
            })
        );

        // Full accept: keep == total rolls back exactly zero rows.
        cache.record_verify(7, &[20, 21]).unwrap();
        cache.commit_cycle(7, 2, 2).unwrap();
        assert_eq!(cache.cursor, 5, "commit(2,2) must roll back exactly 0 rows");

        // Reject-all: keep == 0 retracts the whole verify write.
        cache.record_verify(7, &[30]).unwrap();
        cache.commit_cycle(7, 0, 1).unwrap();
        assert_eq!(cache.cursor, 5, "commit(0,1) must roll back exactly 1 row");
    }

    #[test]
    fn spec_paged_settle_never_runs_inside_a_cycle() {
        let mut cache = SettleOrderChecked::new(MockCursorCache::default());

        cache.reserve_lookahead(3, 2).unwrap();
        cache.record_verify(3, &[1, 2]).unwrap();

        // Settle between record_verify and commit_cycle must trip, and the
        // wrapped cache must never see the illegal call.
        let err = cache.settle_committed(3, 0).unwrap_err();
        assert!(err.contains("L-SETTLE"), "unexpected error text: {err}");
        assert!(
            !cache.inner().calls.iter().any(|c| c.starts_with("settle")),
            "an in-cycle settle reached the cache: {:?}",
            cache.inner().calls
        );

        // The lawful order flows through untouched.
        cache.commit_cycle(3, 1, 2).unwrap();
        cache.settle_committed(3, 1).unwrap();
        assert_eq!(
            cache.inner().calls,
            [
                "reserve(3,2)",
                "record(3,2)",
                "commit(3,1,2)",
                "settle(3,1)"
            ]
        );

        // The check is per sequence: another sequence's open cycle neither
        // blocks this one nor is unblocked by it.
        cache.record_verify(9, &[5]).unwrap();
        cache.settle_committed(3, 1).unwrap();
        cache.settle_committed(9, 0).unwrap_err();
    }
}
