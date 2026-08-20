//! Family-neutral contract for paged speculative KV bookkeeping.
//!
//! One speculative verify cycle touches the paged cache in exactly four
//! moves — reserve the lookahead region, write the verify rows, commit the
//! accepted prefix, settle durable state — and [`SpecPagedCache`] names them
//! once, so every family (dense/MoE native MTP, gemma4 DSpark, muse DFlash,
//! gemma4 assistant) lands on the same call order instead of re-deriving it
//! against raw adapter/coordinator surfaces.
//!
//! The middle two moves are one TRANSACTION: the write opens a cycle and
//! hands back a [`VerifyTicket`] carrying the pre-cycle frontier and the row
//! width, and only that ticket can close it. A driver therefore cannot hand
//! the commit two independently-derived counts that disagree — the shape
//! that lets an oversized rollback eat pre-cycle history.
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

/// An OPEN speculative verify cycle for one sequence.
///
/// Minted where the cycle's rows are written and consumed by
/// [`SpecPagedCache::commit_cycle`], which is the only way to close a cycle.
/// It carries the two facts the commit would otherwise have to trust a
/// caller for — which sequence the cycle belongs to, and how far the cursor
/// sat before the write — so the commit needs exactly ONE caller-supplied
/// number (the accepted row count) and derives the rollback itself.
///
/// Not `Clone`/`Copy`: one cycle, one close.
#[allow(dead_code)]
#[must_use = "an open verify cycle must be closed by SpecPagedCache::commit_cycle"]
#[derive(Debug)]
pub(crate) struct VerifyTicket {
    seq_id: u32,
    pre_attn_tokens: u64,
    rows: usize,
}

#[allow(dead_code)]
impl VerifyTicket {
    /// Mint a ticket for a verify write this facade did NOT perform.
    ///
    /// Some families record the verify rows inside their verify core rather
    /// than through [`SpecPagedCache::record_verify`] — the qwen3_5 dense
    /// paged verify writes its `[anchor, drafts..]` slice as part of the
    /// forward, and routing that write back out through the facade would
    /// mean recording the rows twice. Those drivers open the cycle around
    /// the core write instead, and this is the mint that keeps such a cycle
    /// expressible while still yielding a CHECKED commit: the commit
    /// verifies the cursor actually moved by `rows` and lands on
    /// `pre_attn_tokens + keep`, so an out-of-band write that wrote a
    /// different number of rows is caught before anything is rolled back.
    ///
    /// Prefer [`SpecPagedCache::open_core_write_cycle`], which reads
    /// `pre_attn_tokens` off the cache instead of trusting a caller to
    /// remember it.
    pub(crate) fn from_core_write(seq_id: u32, pre_attn_tokens: u64, rows: usize) -> Self {
        Self {
            seq_id,
            pre_attn_tokens,
            rows,
        }
    }

    pub(crate) fn seq_id(&self) -> u32 {
        self.seq_id
    }

    /// The attention frontier the cycle opened at — where a reject-all
    /// commit returns the cursor to.
    pub(crate) fn pre_attn_tokens(&self) -> u64 {
        self.pre_attn_tokens
    }

    /// Rows the verify write added. The commit's `keep` may not exceed it.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
}

/// Facade over the paged-KV bookkeeping one speculative verify cycle needs.
///
/// Implemented by the per-family paged cache owners (the qwen3_5 dense/MoE
/// adapter-backed steppers, `Gemma4KVCacheCoordinator` for gemma4 and muse).
/// The engine-side speculative loops (`run_mtp_turn`, `run_dspark_turn`)
/// stay on their existing stepper traits; this contract is what those
/// steppers' paged modes are written against, so the ordering laws below
/// hold for every family by conformance rather than by convention.
///
/// Implementations supply the four primitives ([`Self::reserve_lookahead`],
/// [`Self::record_rows`], [`Self::rollback_rows`], [`Self::settle_committed`])
/// plus [`Self::frontier`] and the [`Self::settle_captures_durable_state`]
/// declaration. The cycle transaction ([`Self::record_verify`],
/// [`Self::open_core_write_cycle`], [`Self::commit_cycle`]) is PROVIDED and
/// must not be re-derived per family — that is what keeps the commit
/// arithmetic single-sourced. A wrapper that enforces call order overrides
/// the transaction methods; it must override BOTH cycle openers and the
/// commit, or an out-of-band cycle escapes its bookkeeping.
///
/// # L-SETTLE (I9) — settle at the committed frontier, never the cursor
///
/// Between a sequence's cycle open ([`Self::record_verify`] or
/// [`Self::open_core_write_cycle`]) and its [`Self::commit_cycle`], the
/// write cursor sits up to the whole lookahead ahead of the committed
/// frontier. In that gap the following is FORBIDDEN for that sequence,
/// because it is IRREVERSIBLE — it publishes or destroys state a rollback
/// can still retract:
///
/// - cold / durable checkpoint capture,
/// - sidecar capture,
/// - prefix-block registration,
/// - freeing any block at or above the committed cutoff.
///
/// The following IS PERMITTED in the gap, and real drivers depend on it:
///
/// - evaluating pending pool writes (it materializes rows already written,
///   publishes nothing, and frees nothing),
/// - the committed-basis sliding-window prune.
///
/// Why the committed-basis prune is safe, in full: its cutoff is
/// `committed - window`, and `committed - window <= committed <= p` for
/// every speculative row position `p` in the open cycle, so the pruned
/// range and the rollback range are disjoint — a committed-basis prune can
/// never reach a row the commit may retract. What it must never take is the
/// CURSOR basis (`cursor - window`), which can exceed `committed` and null
/// the block the rollback returns the live window into. Two gates hold this
/// down, and both would be illegal under a blanket no-settle-in-cycle
/// reading: `prune_committed_cutoff_never_nulls_rollback_range`
/// (`transformer::paged_kv_cache_adapter`) at adapter scope, and
/// `settle_committed_never_nulls_the_rollback_range_at_coordinator_level`
/// (`models::gemma4::model`) at coordinator scope. Muse's per-chunk settle
/// on the unified paged layer loop (`MusePagedSettle`) is legal for exactly
/// this reason, which is what lets a DFlash driver settle inside a cycle at
/// all.
///
/// [`NoDurableSettleInCycle`] is the executable form of this law.
/// [`NoSettleInCycle`] enforces the stricter shape — no settle of any kind
/// in the gap — which fits a driver that settles only post-commit, but by
/// construction cannot wrap one that settles per chunk.
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
///
/// This law has NO executable enforcement here: nothing in this module can
/// observe a turn epilogue, so it is a documented obligation on the driver.
/// The first driver that will execute it is Stage D1 (gemma4 DSpark paged),
/// with Stage D2 (dense/MoE native MTP) next; whichever lands first owns
/// making it checkable.
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

    /// Append `tokens` at the optimistic write cursor, in pre-reserved
    /// blocks. Atomic per slice: on failure no partial suffix stays
    /// recorded. Drivers call [`Self::record_verify`], which wraps this in a
    /// cycle; this primitive exists so the wrapping lives once.
    fn record_rows(&mut self, seq_id: u32, tokens: &[u32]) -> Result<(), String>;

    /// Retract the last `rows` recorded rows of `seq_id`. Cursor arithmetic
    /// (I10): paged positions are absolute and sliding groups null-prune
    /// instead of wrapping, so no block is freed and the next record
    /// overwrites in place. Called only by [`Self::commit_cycle`], with a
    /// count derived from the cycle's ticket and never zero.
    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> Result<(), String>;

    /// Settle durable/derived cache state at the COMMITTED frontier: the
    /// family's per-step settle re-anchored from the write cursor to
    /// `committed_tokens`. The only legal settle entry point during a
    /// speculative turn (L-SETTLE).
    fn settle_committed(&mut self, seq_id: u32, committed_tokens: u64) -> Result<(), String>;

    /// Whether [`Self::settle_committed`] can do IRREVERSIBLE work — cold /
    /// durable checkpoint capture, sidecar capture, prefix-block
    /// registration, or freeing a block at or above the committed cutoff.
    ///
    /// This is a claim about the implementation, and it is what lets
    /// [`NoDurableSettleInCycle`] admit a per-chunk settle from one cache
    /// while refusing it from another: a `true` cache may not be settled
    /// inside an open cycle at ALL. Required rather than defaulted so a new
    /// implementation cannot inherit permission it did not think about.
    fn settle_captures_durable_state(&self) -> bool;

    /// The one frontier every cache group agrees on for `seq_id`, or `None`
    /// when the cache cannot name one (unknown sequence, or groups
    /// disagreeing mid-flight). This is the paged survival of the flat
    /// snapshot/commit transaction's all-groups-advanced-exactly assert;
    /// the engine's post-rollback debug asserts consume it through the
    /// stepper `frontier()` hooks, and [`Self::commit_cycle`] checks the
    /// cycle against it.
    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier>;

    /// Record the cycle's verify write `[anchor, drafts..]` and OPEN the
    /// cycle it belongs to.
    fn record_verify(&mut self, seq_id: u32, tokens: &[u32]) -> Result<VerifyTicket, String> {
        let pre_attn_tokens = committed_attn_tokens(self, seq_id)?;
        self.record_rows(seq_id, tokens)?;
        Ok(VerifyTicket::from_core_write(
            seq_id,
            pre_attn_tokens,
            tokens.len(),
        ))
    }

    /// Open a cycle for a verify write the family's own core performs (see
    /// [`VerifyTicket::from_core_write`]). Call this BEFORE the core write —
    /// it reads the pre-cycle frontier off the cache — and pass the exact
    /// row width the core is about to write.
    fn open_core_write_cycle(&mut self, seq_id: u32, rows: usize) -> Result<VerifyTicket, String> {
        let pre_attn_tokens = committed_attn_tokens(self, seq_id)?;
        Ok(VerifyTicket::from_core_write(seq_id, pre_attn_tokens, rows))
    }

    /// Close the cycle `ticket` opened: of the rows the verify wrote, keep
    /// the accepted prefix of `keep` and roll back the rest.
    ///
    /// Checked in this order, so a rejection mutates nothing:
    ///
    /// 1. the ticket names `seq_id` — checked FIRST, before the full-accept
    ///    path can return, so a wrong sequence cannot slip through a commit
    ///    that happens to roll back zero rows;
    /// 2. `keep <= ticket.rows()` — `Err`, never a panic;
    /// 3. the cursor sits exactly `ticket.rows()` past the frontier the
    ///    ticket opened at, i.e. the ticket describes THIS cycle. An
    ///    oversized or stale ticket is refused here, before any rollback,
    ///    which is what makes it impossible for one to eat pre-cycle
    ///    history.
    ///
    /// Then the rollback runs, and the post-commit frontier must be exactly
    /// `ticket.pre_attn_tokens() + keep`.
    fn commit_cycle(
        &mut self,
        seq_id: u32,
        ticket: VerifyTicket,
        keep: usize,
    ) -> Result<(), String> {
        if ticket.seq_id != seq_id {
            return Err(format!(
                "commit_cycle(seq={seq_id}) was handed sequence {}'s verify ticket",
                ticket.seq_id
            ));
        }
        let Some(rollback) = ticket.rows.checked_sub(keep) else {
            return Err(format!(
                "commit_cycle(seq={seq_id}) kept {keep} of only {} verify rows — keep may \
                 never exceed the rows the cycle wrote",
                ticket.rows
            ));
        };
        let open_at = committed_attn_tokens(self, seq_id)?;
        let expected_open = ticket.pre_attn_tokens + ticket.rows as u64;
        if open_at != expected_open {
            return Err(format!(
                "commit_cycle(seq={seq_id}): the cache sits at {open_at} attention tokens \
                 but the ticket describes a cycle of {} rows opened at {} (expected \
                 {expected_open}) — stale or oversized ticket, nothing rolled back",
                ticket.rows, ticket.pre_attn_tokens
            ));
        }
        if rollback > 0 {
            self.rollback_rows(seq_id, rollback)?;
        }
        let landed = committed_attn_tokens(self, seq_id)?;
        let expected = ticket.pre_attn_tokens + keep as u64;
        if landed != expected {
            return Err(format!(
                "commit_cycle(seq={seq_id}) landed on {landed} attention tokens, not the \
                 pre-cycle frontier {} plus the {keep} kept rows ({expected})",
                ticket.pre_attn_tokens
            ));
        }
        Ok(())
    }
}

/// The attention frontier `seq_id` sits at, or the error a cycle boundary
/// reports when the cache cannot name one.
#[allow(dead_code)]
fn committed_attn_tokens<C: SpecPagedCache + ?Sized>(
    cache: &C,
    seq_id: u32,
) -> Result<u64, String> {
    cache
        .frontier(seq_id)
        .map(|frontier| frontier.attn_tokens)
        .ok_or_else(|| format!("the paged cache cannot name a frontier for sequence {seq_id}"))
}

/// Executable form of the STRICT L-SETTLE shape: wraps any
/// [`SpecPagedCache`] and refuses a `settle_committed` issued inside an open
/// cycle, whatever its basis. Conformance harness for a driver that settles
/// only post-commit; a driver that settles per chunk at the committed basis
/// is lawful (see L-SETTLE) and belongs in [`NoDurableSettleInCycle`].
#[allow(dead_code)]
pub(crate) struct NoSettleInCycle<C> {
    inner: C,
    /// Sequences with an open verify cycle.
    open_cycles: std::collections::HashSet<u32>,
}

#[allow(dead_code)]
impl<C> NoSettleInCycle<C> {
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

impl<C: SpecPagedCache> SpecPagedCache for NoSettleInCycle<C> {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> Result<bool, String> {
        self.inner.reserve_lookahead(seq_id, rows)
    }

    fn record_rows(&mut self, seq_id: u32, tokens: &[u32]) -> Result<(), String> {
        self.inner.record_rows(seq_id, tokens)
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> Result<(), String> {
        self.inner.rollback_rows(seq_id, rows)
    }

    fn settle_committed(&mut self, seq_id: u32, committed_tokens: u64) -> Result<(), String> {
        if self.open_cycles.contains(&seq_id) {
            return Err(format!(
                "L-SETTLE violation: settle_committed(seq={seq_id}) inside an open verify \
                 cycle — this cache is wrapped in the strict no-settle-in-cycle checker"
            ));
        }
        self.inner.settle_committed(seq_id, committed_tokens)
    }

    fn settle_captures_durable_state(&self) -> bool {
        self.inner.settle_captures_durable_state()
    }

    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.inner.frontier(seq_id)
    }

    fn record_verify(&mut self, seq_id: u32, tokens: &[u32]) -> Result<VerifyTicket, String> {
        let ticket = self.inner.record_verify(seq_id, tokens)?;
        self.open_cycles.insert(seq_id);
        Ok(ticket)
    }

    fn open_core_write_cycle(&mut self, seq_id: u32, rows: usize) -> Result<VerifyTicket, String> {
        let ticket = self.inner.open_core_write_cycle(seq_id, rows)?;
        self.open_cycles.insert(seq_id);
        Ok(ticket)
    }

    fn commit_cycle(
        &mut self,
        seq_id: u32,
        ticket: VerifyTicket,
        keep: usize,
    ) -> Result<(), String> {
        self.inner.commit_cycle(seq_id, ticket, keep)?;
        self.open_cycles.remove(&seq_id);
        Ok(())
    }
}

/// Executable form of L-SETTLE as written: wraps any [`SpecPagedCache`] and
/// refuses only the IRREVERSIBLE half inside an open cycle, so a real driver
/// — one whose layer loop settles per chunk while speculative rows are
/// pending — can be wrapped in it.
///
/// Inside an open cycle a settle is refused when either
///
/// - the wrapped cache declares [`SpecPagedCache::settle_captures_durable_state`]
///   (its settle can checkpoint, capture a sidecar, or register prefix
///   blocks — none of which a rollback can retract), or
/// - the requested basis is past the frontier the cycle opened at, i.e. the
///   caller is settling at the WRITE CURSOR and would prune against rows the
///   commit can still retract.
///
/// A committed-basis settle from a non-durable cache passes through, with
/// the L-SETTLE proof behind it.
#[allow(dead_code)]
pub(crate) struct NoDurableSettleInCycle<C> {
    inner: C,
    /// Open cycles, each mapped to the committed frontier it opened at —
    /// the highest basis a settle may name while the cycle is open.
    open_cycles: std::collections::HashMap<u32, u64>,
}

#[allow(dead_code)]
impl<C> NoDurableSettleInCycle<C> {
    pub(crate) fn new(inner: C) -> Self {
        Self {
            inner,
            open_cycles: std::collections::HashMap::new(),
        }
    }

    pub(crate) fn inner(&self) -> &C {
        &self.inner
    }

    pub(crate) fn into_inner(self) -> C {
        self.inner
    }
}

impl<C: SpecPagedCache> SpecPagedCache for NoDurableSettleInCycle<C> {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> Result<bool, String> {
        self.inner.reserve_lookahead(seq_id, rows)
    }

    fn record_rows(&mut self, seq_id: u32, tokens: &[u32]) -> Result<(), String> {
        self.inner.record_rows(seq_id, tokens)
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> Result<(), String> {
        self.inner.rollback_rows(seq_id, rows)
    }

    fn settle_committed(&mut self, seq_id: u32, committed_tokens: u64) -> Result<(), String> {
        if let Some(&committed_basis) = self.open_cycles.get(&seq_id) {
            if self.inner.settle_captures_durable_state() {
                return Err(format!(
                    "L-SETTLE violation: settle_committed(seq={seq_id}) inside an open \
                     verify cycle on a cache whose settle captures durable state — a \
                     rollback cannot retract a checkpoint, a sidecar, or a registered \
                     prefix block"
                ));
            }
            if committed_tokens > committed_basis {
                return Err(format!(
                    "L-SETTLE violation: settle_committed(seq={seq_id}) at basis \
                     {committed_tokens} inside a cycle opened at committed frontier \
                     {committed_basis} — that is the write cursor, not the committed \
                     frontier, and it prunes against rows the commit can still retract"
                ));
            }
        }
        self.inner.settle_committed(seq_id, committed_tokens)
    }

    fn settle_captures_durable_state(&self) -> bool {
        self.inner.settle_captures_durable_state()
    }

    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.inner.frontier(seq_id)
    }

    fn record_verify(&mut self, seq_id: u32, tokens: &[u32]) -> Result<VerifyTicket, String> {
        let ticket = self.inner.record_verify(seq_id, tokens)?;
        self.open_cycles.insert(seq_id, ticket.pre_attn_tokens());
        Ok(ticket)
    }

    fn open_core_write_cycle(&mut self, seq_id: u32, rows: usize) -> Result<VerifyTicket, String> {
        let ticket = self.inner.open_core_write_cycle(seq_id, rows)?;
        self.open_cycles.insert(seq_id, ticket.pre_attn_tokens());
        Ok(ticket)
    }

    fn commit_cycle(
        &mut self,
        seq_id: u32,
        ticket: VerifyTicket,
        keep: usize,
    ) -> Result<(), String> {
        self.inner.commit_cycle(seq_id, ticket, keep)?;
        self.open_cycles.remove(&seq_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Reference conformer: a per-sequence recorded-row cursor per the I10
    /// model (absolute positions, rollback = cursor subtraction), plus a
    /// call log so ordering tests can assert what reached the cache.
    #[derive(Default)]
    struct MockCursorCache {
        cursors: HashMap<u32, u64>,
        calls: Vec<String>,
        durable_settle: bool,
    }

    impl MockCursorCache {
        /// Make `seq_id` known to the cache, as a real family's
        /// `reset_scheduled_request` does, and seed its committed history.
        fn open_seq(&mut self, seq_id: u32, history: u64) -> &mut Self {
            self.cursors.insert(seq_id, history);
            self
        }

        fn cursor(&self, seq_id: u32) -> Option<u64> {
            self.cursors.get(&seq_id).copied()
        }
    }

    impl SpecPagedCache for MockCursorCache {
        fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> Result<bool, String> {
            self.calls.push(format!("reserve({seq_id},{rows})"));
            Ok(true)
        }

        fn record_rows(&mut self, seq_id: u32, tokens: &[u32]) -> Result<(), String> {
            self.calls
                .push(format!("record({seq_id},{})", tokens.len()));
            let cursor = self
                .cursors
                .get_mut(&seq_id)
                .ok_or_else(|| format!("record_rows: sequence {seq_id} is not open"))?;
            *cursor += tokens.len() as u64;
            Ok(())
        }

        fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> Result<(), String> {
            self.calls.push(format!("rollback({seq_id},{rows})"));
            let cursor = self
                .cursors
                .get_mut(&seq_id)
                .ok_or_else(|| format!("rollback_rows: sequence {seq_id} is not open"))?;
            *cursor = cursor
                .checked_sub(rows as u64)
                .ok_or_else(|| format!("rollback of {rows} rows underflows the cursor"))?;
            Ok(())
        }

        fn settle_committed(&mut self, seq_id: u32, committed_tokens: u64) -> Result<(), String> {
            self.calls
                .push(format!("settle({seq_id},{committed_tokens})"));
            Ok(())
        }

        fn settle_captures_durable_state(&self) -> bool {
            self.durable_settle
        }

        fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
            self.cursors.get(&seq_id).map(|&attn_tokens| SpecFrontier {
                attn_tokens,
                recurrent_tokens: None,
            })
        }
    }

    #[test]
    fn spec_paged_commit_is_exact_cursor_arithmetic() {
        let mut cache = MockCursorCache::default();
        cache.open_seq(7, 0);

        // Partial accept: 5 verify rows (anchor + 4 drafts), keep 3.
        let ticket = cache.record_verify(7, &[10, 11, 12, 13, 14]).unwrap();
        assert_eq!(cache.cursor(7), Some(5));
        assert_eq!((ticket.pre_attn_tokens(), ticket.rows()), (0, 5));
        cache.commit_cycle(7, ticket, 3).unwrap();
        assert_eq!(cache.cursor(7), Some(3), "commit(keep=3 of 5) rolls back 2");

        // Full accept: keep == rows rolls back exactly zero rows.
        let ticket = cache.record_verify(7, &[20, 21]).unwrap();
        cache.commit_cycle(7, ticket, 2).unwrap();
        assert_eq!(cache.cursor(7), Some(5), "commit(keep=2 of 2) rolls back 0");

        // Reject-all: keep == 0 retracts the whole verify write.
        let ticket = cache.record_verify(7, &[30]).unwrap();
        cache.commit_cycle(7, ticket, 0).unwrap();
        assert_eq!(cache.cursor(7), Some(5), "commit(keep=0 of 1) rolls back 1");

        // A full accept never touches the rollback primitive.
        assert_eq!(
            cache.calls,
            [
                "record(7,5)",
                "rollback(7,2)",
                "record(7,2)",
                "record(7,1)",
                "rollback(7,1)",
            ]
        );
    }

    /// The cycle is a transaction for ONE sequence: a ticket minted on
    /// sequence A may not close a cycle on sequence B. Both commit shapes
    /// are covered — the partial accept AND the full accept, whose rollback
    /// is zero and which therefore has no other reason to touch the cache.
    ///
    /// Mutation this catches: validating the ticket's sequence AFTER the
    /// zero-rollback early return (the pre-transaction shape, where a
    /// full-accept commit returned `Ok` before looking at the sequence id).
    #[test]
    fn commit_cycle_rejects_a_ticket_minted_for_another_sequence() {
        let mut cache = MockCursorCache::default();
        cache.open_seq(3, 4).open_seq(9, 4);

        // Full accept (rollback == 0) with a foreign ticket.
        let ticket_three = cache.record_verify(3, &[1, 2]).unwrap();
        let err = cache
            .commit_cycle(9, ticket_three, 2)
            .expect_err("sequence 9 may not be committed with sequence 3's ticket");
        assert!(
            err.contains("sequence 3's verify ticket"),
            "unexpected error text: {err}"
        );
        assert_eq!(cache.cursor(3), Some(6), "the refused commit moved 3");
        assert_eq!(cache.cursor(9), Some(4), "the refused commit moved 9");

        // Partial accept (rollback > 0) with a foreign ticket.
        let ticket_nine = cache.record_verify(9, &[5, 6, 7]).unwrap();
        cache
            .commit_cycle(3, ticket_nine, 1)
            .expect_err("sequence 3 may not be committed with sequence 9's ticket");
        assert_eq!(cache.cursor(3), Some(6));
        assert_eq!(cache.cursor(9), Some(7));
        assert!(
            !cache.calls.iter().any(|call| call.starts_with("rollback")),
            "a refused commit must reach no primitive: {:?}",
            cache.calls
        );
    }

    /// `keep` past the cycle's row count is a caller bug the facade REPORTS
    /// — the pre-transaction shape asserted, i.e. panicked, inside the
    /// commit arithmetic. Nothing may be mutated on the way out.
    ///
    /// Mutation this catches: restoring the `assert!(keep <= total)` /
    /// panicking form.
    #[test]
    fn commit_cycle_rejects_keep_past_the_cycle_rows_without_mutating() {
        let mut cache = MockCursorCache::default();
        cache.open_seq(7, 12);

        let ticket = cache.record_verify(7, &[1, 2, 3]).unwrap();
        let err = cache
            .commit_cycle(7, ticket, 4)
            .expect_err("keep may never exceed the rows the cycle wrote");
        assert!(
            err.contains("kept 4 of only 3 verify rows"),
            "unexpected error text: {err}"
        );
        assert_eq!(
            cache.cursor(7),
            Some(15),
            "a refused commit must leave the cursor exactly where the write left it"
        );
        assert!(
            !cache.calls.iter().any(|call| call.starts_with("rollback")),
            "a refused commit must reach no primitive: {:?}",
            cache.calls
        );
    }

    /// The commit's rollback comes from the TICKET, never from the cursor —
    /// the cursor also counts pre-cycle history, so deriving from it turns a
    /// partial accept into a truncation of committed rows. And a ticket that
    /// does not describe the open cycle (oversized, or left over from an
    /// earlier one) is refused BEFORE any rollback, so it cannot reach that
    /// history either.
    ///
    /// Mutations this catches: (a) deriving the rollback from the current
    /// cursor instead of `ticket.rows() - keep`; (b) dropping the
    /// ticket-describes-this-cycle check, which lets an oversized ticket
    /// roll back past the rows the cycle actually wrote.
    #[test]
    fn commit_cycle_cannot_truncate_pre_cycle_history() {
        const HISTORY: u64 = 10;
        let mut cache = MockCursorCache::default();
        cache.open_seq(7, HISTORY);

        // A healthy partial accept over non-empty history.
        let ticket = cache.record_verify(7, &[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(cache.cursor(7), Some(HISTORY + 5));
        cache
            .commit_cycle(7, ticket, 3)
            .expect("keep 3 of the cycle's 5 rows");
        assert_eq!(
            cache.cursor(7),
            Some(HISTORY + 3),
            "the commit must land on the pre-cycle frontier plus the kept rows, \
             not on cursor - keep"
        );

        // An oversized ticket over a shorter cycle: refused before it can
        // roll back past what the cycle wrote.
        cache.record_rows(7, &[6, 7]).expect("out-of-band write");
        let cursor_before = cache.cursor(7);
        let oversized = VerifyTicket::from_core_write(7, HISTORY + 3, 9);
        let err = cache
            .commit_cycle(7, oversized, 0)
            .expect_err("a ticket that does not describe the open cycle must be refused");
        assert!(
            err.contains("stale or oversized ticket"),
            "unexpected error text: {err}"
        );
        assert_eq!(
            cache.cursor(7),
            cursor_before,
            "the refused commit must roll back nothing at all"
        );

        // The honest ticket for that same out-of-band write commits.
        let honest = VerifyTicket::from_core_write(7, HISTORY + 3, 2);
        cache.commit_cycle(7, honest, 1).expect("keep 1 of 2");
        assert_eq!(cache.cursor(7), Some(HISTORY + 4));
    }

    #[test]
    fn spec_paged_settle_never_runs_inside_a_cycle() {
        let mut inner = MockCursorCache::default();
        inner.open_seq(3, 0).open_seq(9, 0);
        let mut cache = NoSettleInCycle::new(inner);

        cache.reserve_lookahead(3, 2).unwrap();
        let ticket = cache.record_verify(3, &[1, 2]).unwrap();

        // Settle between the write and the commit must trip, and the
        // wrapped cache must never see the illegal call.
        let err = cache.settle_committed(3, 0).unwrap_err();
        assert!(err.contains("L-SETTLE"), "unexpected error text: {err}");
        assert!(
            !cache.inner().calls.iter().any(|c| c.starts_with("settle")),
            "an in-cycle settle reached the cache: {:?}",
            cache.inner().calls
        );

        // The lawful order flows through untouched.
        cache.commit_cycle(3, ticket, 1).unwrap();
        cache.settle_committed(3, 1).unwrap();
        assert_eq!(
            cache.inner().calls,
            [
                "reserve(3,2)",
                "record(3,2)",
                "rollback(3,1)",
                "settle(3,1)"
            ]
        );

        // The check is per sequence: another sequence's open cycle neither
        // blocks this one nor is unblocked by it.
        let nine = cache.record_verify(9, &[5]).unwrap();
        cache.settle_committed(3, 1).unwrap();
        cache.settle_committed(9, 0).unwrap_err();
        cache.commit_cycle(9, nine, 1).unwrap();
    }

    /// The permissive checker is the one a real driver fits in: a
    /// committed-basis settle inside an open cycle is LAWFUL (muse's
    /// per-chunk settle, gemma4's coordinator settle) and must pass through,
    /// while the two irreversible shapes must not — a cache whose settle
    /// captures durable state, and a settle at the write cursor.
    ///
    /// Mutations this catches: (a) letting a durable-capture cache settle
    /// in-cycle; (b) letting a cursor-basis settle through; (c) refusing the
    /// committed-basis settle (the strict shape, which no per-chunk driver
    /// could be wrapped in).
    #[test]
    fn permissive_checker_admits_the_committed_basis_prune_only() {
        let mut inner = MockCursorCache::default();
        inner.open_seq(7, 36);
        let mut cache = NoDurableSettleInCycle::new(inner);

        assert!(cache.reserve_lookahead(7, 5).unwrap());
        let ticket = cache.record_verify(7, &[100, 101, 102, 103, 104]).unwrap();
        assert_eq!(cache.frontier(7).unwrap().attn_tokens, 41);

        // Committed basis (at or below the frontier the cycle opened at):
        // lawful, and it must actually reach the cache.
        cache
            .settle_committed(7, 36)
            .expect("a committed-basis settle is lawful inside a cycle");
        cache
            .settle_committed(7, 30)
            .expect("a basis below the committed frontier is lawful too");
        assert_eq!(
            cache
                .inner()
                .calls
                .iter()
                .filter(|call| call.starts_with("settle"))
                .count(),
            2,
            "the lawful settles must reach the cache: {:?}",
            cache.inner().calls
        );

        // Write cursor basis: refused, and it must not reach the cache.
        let err = cache
            .settle_committed(7, 41)
            .expect_err("a cursor-basis settle inside a cycle must trip");
        assert!(err.contains("L-SETTLE"), "unexpected error text: {err}");
        assert!(
            !cache.inner().calls.iter().any(|c| c == "settle(7,41)"),
            "a cursor-basis settle reached the cache: {:?}",
            cache.inner().calls
        );

        cache.commit_cycle(7, ticket, 1).expect("keep the anchor");
        // Post-commit the basis is whatever the commit landed on.
        cache.settle_committed(7, 37).expect("post-commit settle");

        // Same call order against a cache whose settle captures durable
        // state: the committed basis is no longer enough.
        let mut durable = MockCursorCache::default();
        durable.open_seq(7, 36);
        durable.durable_settle = true;
        let mut cache = NoDurableSettleInCycle::new(durable);
        let ticket = cache.record_verify(7, &[100, 101, 102, 103, 104]).unwrap();
        let err = cache
            .settle_committed(7, 36)
            .expect_err("a durable-capture settle may not run inside a cycle at any basis");
        assert!(
            err.contains("captures durable state"),
            "unexpected error text: {err}"
        );
        assert!(
            !cache.inner().calls.iter().any(|c| c.starts_with("settle")),
            "a durable in-cycle settle reached the cache: {:?}",
            cache.inner().calls
        );
        cache.commit_cycle(7, ticket, 1).expect("keep the anchor");
        cache
            .settle_committed(7, 37)
            .expect("post-commit the durable capture is lawful");
    }
}
