//! Dense MTP stepper, `MtpBackend`, and the generic decode stepper.

use super::*;

/// Per-turn decode stepper for the engine's generic (text-only,
/// non-paged, non-MTP) flow on Qwen3.5 dense
/// ([`ChatBackend::begin_decode`]).
///
/// Drives the pure-Rust eager `forward_inner` over the flat caches.
pub(crate) struct Qwen35Decode<'a> {
    pub(super) inner: &'a mut Qwen35Inner,
    pub(super) embedding: Embedding,
    /// Decode-path profiler relabel (`chat_rust` and its
    /// `chat_stream[_delta]_*` streaming variants), resolved in
    /// `begin_decode` from the turn's streaming-ness.
    pub(super) relabel: &'static str,
}

impl DecodeStep for Qwen35Decode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let inner = &mut *self.inner;
        let logits = forward_inner(
            input_ids,
            &self.embedding,
            &mut inner.layers,
            &mut inner.caches,
            &inner.final_norm,
            &inner.lm_head,
        )?;
        // `true` == the eager Rust forward returns `[1, 1, vocab]`;
        // the loop squeezes axis 1.
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token, logits]);
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        Some(self.relabel)
    }
}

/// Per-turn pure-Rust ("eager") dense MTP stepper the engine-owned
/// [`crate::engine::mtp_turn::run_mtp_turn`] drives. The per-cycle scratch
/// (the GDN tape, the pre-verify snapshot, the stashed replay error, the
/// desync latch, the committed-history bookkeeping) are plain fields: the
/// engine calls the methods strictly sequentially and non-nested.
///
/// Drives BOTH the FLAT and the block-PAGED dense MTP turn, selected by
/// [`Self::owner`]: with no owner the eager pre-norm forwards run against
/// `inner.caches`; with one, the main Step-A / verify forwards route through
/// `inner.paged_adapter`
/// ([`crate::models::qwen3_5::paged_forward::run_paged_step_with_hidden`] /
/// [`crate::models::qwen3_5::paged_forward::run_paged_verify_step`]) while the GDN recurrent
/// state stays FLAT in `inner.caches` Linear slots. Only four methods
/// (`forward_with_hidden`, `verify_step`, `rollback`, `rollback_unemitted`)
/// branch on it; every other method is path-identical (the drafter and the
/// committed-history commit are paged-agnostic).
///
/// The adapter stays ON THE MODEL for the whole turn: each paged touch
/// borrows it back through [`SpecOwner::resolve`], which refuses once the
/// adapter's active request is no longer this turn's. That refusal is what
/// makes the borrow-back sound, so nothing has to be moved out and restored.
pub(crate) struct DenseMtpStepper<'a> {
    /// The model — owns layers / caches / mtp / final_norm / lm_head and the
    /// `flat_mtp_caches_desynced` latch.
    inner: &'a mut Qwen35Inner,
    /// Drafter K/V caches. Committed-history mode holds the persistent
    /// committed prefix; cycle-history mode is reset fresh by `begin_cycle`.
    mtp_caches: Vec<Qwen3_5LayerCache>,
    /// Committed tokens whose exact K/V live in `mtp_caches`.
    committed_len: i32,
    /// Committed-history active iff the prompt tail's hiddens start at
    /// absolute position 0.
    use_committed: bool,
    /// Chained verify-hidden reuse is only numerically stable when this turn
    /// seeded the drafter with the full prompt. Warm suffix-only prefills have
    /// no such seed and must retain Step A between cycles.
    chained_cycles_supported: bool,
    /// Pre-verify snapshot of the main caches, taken in
    /// `snapshot_main_linear`, consumed by `rollback`.
    snap: Option<Result<Vec<crate::models::qwen3_5::layer_cache::Qwen3_5LayerSnapshot>>>,
    /// GDN tape recorded by `verify_step`, consumed by `rollback`.
    tape: Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>>,
    /// Number of tape steps the retained snapshot + tape currently represent
    /// as the cycle's committed GDN frontier: set to the recorded step count
    /// (`depth + 1`) by `verify_step`, overwritten to `accepted_steps`
    /// (`accepted_drafts + 1`) by `rollback`. `rollback_unemitted` subtracts
    /// `unemitted` in these SAME units, so the mid-cycle rewind target
    /// (`last_cycle_steps - unemitted`) is immune to tape-step-unit skew.
    last_cycle_steps: usize,
    /// The paged verify cycle [`MtpStepper::verify_step`] opened around the
    /// verify core's own row write, consumed by [`MtpStepper::rollback`]'s
    /// facade commit — the one place the adapter's speculative rows are
    /// retracted, so the commit arithmetic cannot drift from the
    /// conformance surface. `None` in flat mode, between cycles, and when
    /// the adapter names no active sequence (the verify core then fails on
    /// its own `record_tokens` and the turn never reaches a rollback).
    pub(super) open_cycle: Option<crate::engine::spec_paged::VerifyTicket>,
    /// Absolute tokens the GDN recurrent state has consumed, reported by
    /// [`MtpStepper::frontier`] against the ATTENTION side's ground truth
    /// (adapter recorded rows / flat full-attention offset). Seeded from the
    /// attention frontier at construction (the two sides are aligned at turn
    /// entry) and moved ONLY where the recurrent state actually moves:
    /// `forward_with_hidden` +1, `verify_step` +depth+1, replay-driven
    /// rollbacks to `recurrent_snapshot_base + steps`. `None` after a failed
    /// replay — the count is then unknown and the stashed error fail-closes
    /// the turn.
    recurrent_frontier: Option<u64>,
    /// [`Self::recurrent_frontier`] captured by `snapshot_main_linear` — the
    /// base the replay-driven rollbacks land relative to.
    recurrent_snapshot_base: Option<u64>,
    /// Error stashed by the infallible `rollback` replay, surfaced by
    /// `take_replay_error`.
    replay_err: Option<Error>,
    /// Mid-cycle-stop desync latch (set by `rollback_unemitted`), reported by
    /// `into_desynced`.
    mtp_desynced: bool,
    /// The model's embedding lookup and tied-head projection backend.
    embedding: Embedding,
    /// Config clone for the per-cycle drafter cache reset/fresh build.
    config: Qwen3_5Config,
    /// The sequence this turn's paged main-forwards belong to, claimed at
    /// `begin_mtp_decode`. `None` runs the flat main path.
    owner: Option<SpecOwner>,
    /// Per-layer attention/linear classification consumed by the paged
    /// forwards. Empty on the flat path (unused there).
    layer_kinds: Vec<crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind>,
}

impl Drop for DenseMtpStepper<'_> {
    fn drop(&mut self) {
        // A turn that failed between the verify write and its rollback (any
        // `?` in the engine's accept path) leaves the cycle open; close it
        // here so the ticket's abandoned-cycle guard does not turn that error
        // return into a debug-build panic. Firing in `Drop` covers EVERY exit
        // path of `run_mtp_turn` — the `Ok` tail, the `take_replay_error`
        // early return, and any mid-loop `?` propagation.
        self.close_abandoned_cycle();
    }
}

impl DenseMtpStepper<'_> {
    /// Restore the pre-verify snapshot and replay the first `steps` recorded
    /// tape steps into the live main caches — the shared GDN replay both
    /// `rollback` (to `accepted_steps`) and the paged `rollback_unemitted`
    /// (to `last_cycle_steps - unemitted`) drive. Pure over
    /// `(snapshot, tape, steps)`: `steps == 0` degenerates to a bare snapshot
    /// restore. On the flat path full-attention layers rewind via `kv.trim`;
    /// on the paged path their K/V lives in the pool (rewound by the adapter)
    /// and the flat shells are skipped.
    fn replay_main_linear_to(&mut self, steps: usize) -> Result<()> {
        let paged = self.owner.is_some();
        let snap = match self.snap.as_ref() {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                return Err(Error::from_reason(format!(
                    "eager MTP replay: snapshot failed: {}",
                    e.reason
                )));
            }
            None => {
                return Err(Error::from_reason(
                    "eager MTP replay: snapshot missing (snapshot_main_linear \
                     did not run)",
                ));
            }
        };
        let tape = &self.tape;
        let inner = &mut *self.inner;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("eager MTP replay: inner.caches is None"))?;
        crate::models::qwen3_5::layer_cache::replay_mtp_snapshot_to(
            caches,
            snap,
            tape,
            steps,
            paged,
            "eager MTP replay",
        )
    }

    /// The attention side's ground-truth frontier: the paged adapter's
    /// recorded rows, or the first flat FullAttention cache's offset.
    fn attention_frontier(&self) -> Option<u64> {
        match self.owner {
            Some(owner) => Some(
                owner
                    .resolve_ref(&self.inner.paged_adapter, DENSE_PAGED_MTP)
                    .ok()?
                    .request_tokens()
                    .len() as u64,
            ),
            None => {
                let caches = self.inner.caches.as_ref()?;
                caches
                    .iter()
                    .find(|c| matches!(c, Qwen3_5LayerCache::FullAttention(_)))
                    .map(|c| c.offset().max(0) as u64)
            }
        }
    }

    /// The paged speculative cache the facade (`engine::spec_paged`)
    /// addresses: the model's adapter, borrowed back through the turn's
    /// owner, iff `seq_id` is that owner. Any other id is refused — a facade
    /// call must never re-activate a different request mid-turn — and a flat
    /// turn has no paged cache to hand out.
    fn paged_cache_for(
        &mut self,
        seq_id: u32,
    ) -> std::result::Result<&mut PagedKVCacheAdapter, String> {
        let owner = self.owner.ok_or_else(|| {
            format!("{DENSE_PAGED_MTP}: a flat turn has no paged speculative cache")
        })?;
        owner.accepts(seq_id, DENSE_PAGED_MTP)?;
        owner.resolve(&mut self.inner.paged_adapter, DENSE_PAGED_MTP)
    }

    /// The paged half of [`MtpStepper::verify_step`]: slice `ids` to exactly
    /// the `depth + 1` rows the core records, OPEN the facade cycle around
    /// that write, then run it. The open is pure — it reads the pre-write
    /// frontier off the adapter and mints the ticket — so a verify that
    /// fails afterwards leaves the adapter exactly where the un-ticketed
    /// path left it.
    ///
    /// The cycle is opened AFTER the fallible id work so a malformed `ids`
    /// never mints a ticket, and BEFORE the write so the ticket's basis is
    /// the pre-write cursor: the commit checks the cursor moved by exactly
    /// the promised rows, which is what refuses a core that wrote a
    /// different width.
    fn paged_verify_step(
        &mut self,
        ids: &[u32],
        depth: usize,
    ) -> Result<mtp_decode::MtpVerifyOutput> {
        if ids.len() != depth + 1 {
            return Err(Error::from_reason(
                "MTP verifier token count does not match depth",
            ));
        }
        let owner = self
            .owner
            .expect("paged_verify_step is only reached on a paged turn");
        self.open_verify_cycle(owner, depth + 1);
        let inner = &mut *self.inner;
        let adapter = owner
            .resolve(&mut inner.paged_adapter, "eager paged MTP verify_step")
            .map_err(Error::from_reason)?;
        // Cross-turn M-RoPE delta carried by a text turn that warm-
        // continues an image prefill; 0 for pure-text sessions.
        let rope_deltas = inner.cached_rope_deltas.unwrap_or(0);
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("eager paged MTP verify_step: caches is None"))?;
        let tape = &mut self.tape;
        crate::models::qwen3_5::paged_forward::run_paged_verify_step(
            ids,
            &inner.embedding,
            &mut inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            &self.layer_kinds,
            adapter,
            tape,
            rope_deltas,
        )
    }

    /// Mint the cycle ticket the verify core's own row write belongs to.
    ///
    /// Failure is not fatal and not silent: the only way to reach it is an
    /// adapter that no longer answers to this turn's owner, whose
    /// `record_tokens` fails the verify a few lines later, so the pre-facade
    /// error semantics stand.
    fn open_verify_cycle(&mut self, owner: SpecOwner, rows: usize) {
        self.close_abandoned_cycle();
        let seq_id = owner.seq_id();
        match crate::engine::spec_paged::SpecPagedCache::open_core_write_cycle(self, seq_id, rows) {
            Ok(ticket) => self.open_cycle = Some(ticket),
            Err(e) => tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "eager MTP-paged verify cycle ({rows} rows) could not be opened \
                 (the rollback falls back to a direct adapter retraction): {e}",
            ),
        }
    }

    /// Close a cycle abandoned between its verify write and its rollback.
    /// Keeping every written row is what the un-ticketed path did — they
    /// stay recorded past the emitted frontier and the epilogue's
    /// length-agreement check refuses to persist the turn — so a full keep
    /// retracts nothing and this consumes the ticket instead of letting its
    /// abandoned-cycle guard fire.
    /// The turn's paged adapter, borrowed back through the owner. Tests
    /// assert on adapter state across a whole turn; production reaches it
    /// through `paged_cache_for` or an inline `owner.resolve`.
    #[cfg(test)]
    pub(super) fn owned_adapter(&self) -> &PagedKVCacheAdapter {
        self.owner
            .expect("the turn must be paged")
            .resolve_ref(&self.inner.paged_adapter, DENSE_PAGED_MTP)
            .expect("the paged adapter must still answer to the turn's owner")
    }

    #[cfg(test)]
    pub(super) fn owned_adapter_mut(&mut self) -> &mut PagedKVCacheAdapter {
        self.owner
            .expect("the turn must be paged")
            .resolve(&mut self.inner.paged_adapter, DENSE_PAGED_MTP)
            .expect("the paged adapter must still answer to the turn's owner")
    }

    fn close_abandoned_cycle(&mut self) {
        let Some(ticket) = self.open_cycle.take() else {
            return;
        };
        let (seq_id, rows) = (ticket.seq_id(), ticket.rows());
        if let Err(e) =
            crate::engine::spec_paged::SpecPagedCache::commit_cycle(self, seq_id, ticket, rows)
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "eager MTP-paged abandoned verify cycle on sequence {seq_id} \
                 ({rows} rows) could not be closed cleanly (ignored): {e}",
            );
        }
    }
}

impl MtpStepper for DenseMtpStepper<'_> {
    fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    fn committed_history_active(&self) -> bool {
        self.use_committed
    }

    fn chained_cycles_supported(&self) -> bool {
        self.chained_cycles_supported
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        Some("mtp_eager")
    }

    // Step A main forward: eager pre-norm + final-norm + project. Returns
    // `hidden` shaped `[1, hidden]` (squeeze the time axis) to match the
    // [`MtpStepper::forward_with_hidden`] contract; `logits` stays
    // `[1, 1, vocab]` with `needs_squeeze = true`.
    fn forward_with_hidden(
        &mut self,
        ids: &MxArray,
        embedding: &Embedding,
    ) -> Result<(MxArray, MxArray, bool)> {
        let output = match self.owner {
            None => {
                let inner = &mut *self.inner;
                let pre =
                    forward_pre_norm_inner(ids, embedding, &mut inner.layers, &mut inner.caches)?;
                let h3 = inner.final_norm.forward(&pre)?;
                let logits = project_logits_from_hidden(&h3, &inner.lm_head, embedding)?;
                let hidden = h3.squeeze(Some(&[1]))?;
                Ok((logits, hidden, true))
            }
            Some(owner) => {
                ids.eval();
                let token_id = ids.item_at_int32(0)? as u32;
                let inner = &mut *self.inner;
                let adapter = owner
                    .resolve(
                        &mut inner.paged_adapter,
                        "eager paged MTP forward_with_hidden",
                    )
                    .map_err(Error::from_reason)?;
                // Cross-turn M-RoPE delta carried by a text turn that warm-
                // continues an image prefill; 0 for pure-text sessions.
                let rope_deltas = inner.cached_rope_deltas.unwrap_or(0);
                let caches = inner.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("eager paged MTP forward_with_hidden: caches is None")
                })?;
                let (logits, hidden) =
                    crate::models::qwen3_5::paged_forward::run_paged_step_with_hidden(
                        token_id,
                        &inner.embedding,
                        &mut inner.layers,
                        caches,
                        &inner.final_norm,
                        &inner.lm_head,
                        &self.layer_kinds,
                        adapter,
                        rope_deltas,
                    )?;
                Ok((logits, hidden, true))
            }
        };
        if output.is_ok()
            && let Some(frontier) = self.recurrent_frontier.as_mut()
        {
            // The main forward consumed one token on the GDN side.
            *frontier += 1;
        }
        output
    }

    // One MTP draft step on the eager drafter. `h_next` is `[1, 1, hidden]`;
    // project to `draft_logits` `[1, 1, vocab]` then squeeze the time axis to
    // `[1, vocab]`.
    fn draft_step(
        &mut self,
        prev_hidden: &MxArray,
        prev_emb: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let inner = &mut *self.inner;
        let mtp_caches = &mut self.mtp_caches;
        let mtp = inner.mtp.as_mut().ok_or_else(|| {
            Error::from_reason(
                "eager MTP draft_step: inner.mtp is None despite \
                 has_mtp_weights() gate",
            )
        })?;
        let h_next = mtp.forward(prev_hidden, prev_emb, Some(mtp_caches))?;
        let dl3 = project_logits_from_hidden(&h_next, &inner.lm_head, &self.embedding)?;
        let draft_logits = dl3.squeeze(Some(&[1]))?;
        Ok((h_next, draft_logits))
    }

    // Batched verify: run the K+1 verify ids through the main stack,
    // advancing `inner.caches` by K+1, recording the GDN tape. The paged
    // half also opens the facade cycle its core write belongs to
    // ([`DenseMtpStepper::paged_verify_step`]); `rollback` closes it.
    fn verify_step(
        &mut self,
        ids: &[u32],
        embedding: &Embedding,
        depth: usize,
    ) -> Result<mtp_decode::MtpVerifyOutput> {
        if ids.len() != depth + 1 {
            return Err(Error::from_reason(
                "MTP verifier token count does not match depth",
            ));
        }
        let output = if self.owner.is_some() {
            self.paged_verify_step(ids, depth)
        } else {
            let ids = MxArray::from_uint32(ids, &[1, ids.len() as i64])?;
            let inner = &mut *self.inner;
            let tape = &mut self.tape;
            eager_verify_step(
                &mut inner.layers,
                &mut inner.caches,
                &inner.final_norm,
                &inner.lm_head,
                &ids,
                embedding,
                Some(tape),
            )
        };
        if output.is_ok() {
            // The recorded tape step count for this cycle: the anchor plus D
            // drafts. `rollback` overwrites it with the accepted count.
            self.last_cycle_steps = depth + 1;
            if let Some(frontier) = self.recurrent_frontier.as_mut() {
                // Verify consumed the anchor plus D drafts on the GDN side.
                *frontier += depth as u64 + 1;
            }
        }
        output
    }

    // No native argmax-only / sparse verify on the eager path — the accept
    // loop falls back to dense-logits accept. (Defaults `None`.)

    // Snapshot the main caches before verify mutates them. Stash the fallible
    // result; surfaced in `rollback` / `restore_and_replay_main`.
    fn snapshot_main_linear(&mut self) {
        // On the paged backend the FullAttention K/V lives in the paged pool,
        // not `inner.caches`, so its flat slot is an empty shell. Snapshot
        // paged-aware so we capture only the GDN (Linear) state and skip the
        // shells — `rollback` rewinds those via the adapter and never reads
        // their snapshot.
        let paged = self.owner.is_some();
        let inner = &*self.inner;
        let snap = match inner.caches.as_ref() {
            Some(caches) => crate::models::qwen3_5::layer_cache::snapshot_all_mtp(caches, paged),
            None => Err(Error::from_reason(
                "eager MTP snapshot_main_linear: inner.caches is None",
            )),
        };
        self.snap = Some(snap);
        self.recurrent_snapshot_base = self.recurrent_frontier;
    }

    // Pure-Rust GDN tape replay — the correctness keystone. Fires on BOTH
    // full and partial accept. Infallible signature: any error is stashed in
    // `self.replay_err` and surfaced later.
    fn rollback(&mut self, accepted_drafts: usize, depth: usize) {
        if self.replay_err.is_some() {
            return;
        }
        // Paged path rewinds the full-attention K/V (which lives in the paged
        // pool, not `inner.caches`) by `rejected` tokens before the shared GDN
        // tape replay. On full accept `rejected == 0` (no-op). Flat layers keep
        // their full-attention K/V in `inner.caches` and rewind it via `kv.trim`
        // inside the replay loop instead.
        //
        // The retraction goes through the cycle `verify_step` opened, so the
        // adapter half of this rollback and the facade's conformance surface
        // are the same arithmetic: the commit derives its rollback as
        // `rows - keep` = `(depth + 1) - (accepted_drafts + 1)`, which is the
        // `depth - accepted_drafts` this path always applied.
        if let Some(owner) = self.owner {
            match self.open_cycle.take() {
                Some(ticket) => {
                    let seq_id = ticket.seq_id();
                    // `keep` is CLAMPED to the rows the cycle wrote, exactly
                    // as the `saturating_sub` it replaces clamped the rejected
                    // count. `accepted_drafts <= depth` is an engine
                    // invariant; were it ever broken, the clamp keeps this on
                    // the commit's checked path — retracting zero rows, as the
                    // `saturating_sub` did — instead of leaving through an Err
                    // and a log line.
                    let keep = (accepted_drafts + 1).min(ticket.rows());
                    if let Err(e) = crate::engine::spec_paged::SpecPagedCache::commit_cycle(
                        self, seq_id, ticket, keep,
                    ) {
                        tracing::warn!(
                            target: "mlx_core::qwen3_5::paged",
                            "eager MTP-paged verify commit (keep {keep} of the cycle's \
                             rows) failed (ignored): {e}",
                        );
                    }
                }
                None => {
                    // No cycle to close: `verify_step` could not open one,
                    // which also fails the verify itself, so production never
                    // lands here. Retract directly rather than leave rejected
                    // rows recorded.
                    let rejected = depth.saturating_sub(accepted_drafts);
                    if rejected > 0
                        && let Err(e) = owner
                            .resolve(&mut self.inner.paged_adapter, DENSE_PAGED_MTP)
                            .map_err(Error::from_reason)
                            .and_then(|adapter| {
                                adapter.rollback_last_tokens(rejected as u32).map_err(|e| {
                                    Error::from_reason(format!("adapter rollback: {e}"))
                                })
                            })
                    {
                        tracing::warn!(
                            target: "mlx_core::qwen3_5::paged",
                            "eager MTP-paged rollback_last_tokens({rejected}) outside a \
                             verify cycle failed (ignored): {e}",
                        );
                    }
                }
            }
        }
        let accepted_steps = accepted_drafts + 1;
        self.last_cycle_steps = accepted_steps;
        match self.replay_main_linear_to(accepted_steps) {
            Ok(()) => {
                self.recurrent_frontier = self
                    .recurrent_snapshot_base
                    .map(|base| base + accepted_steps as u64);
            }
            Err(e) => {
                self.recurrent_frontier = None;
                self.replay_err = Some(e);
            }
        }
    }

    // On rejection (partial accept): the GDN tape replay in `rollback`
    // already reconstructed the AR-exact main cache state, so no re-forward
    // loop is needed. This only surfaces a stashed replay error.
    //
    // The per-cycle snapshot + tape are deliberately RETAINED: both are
    // re-armed by the next cycle anyway (`snapshot_main_linear` overwrites
    // `snap`; the verify cores clear + re-record `tape` at record time), and
    // a mid-cycle stop after THIS cycle still needs them — the paged
    // `rollback_unemitted` replays the GDN state back to the emitted frontier
    // from exactly this snapshot + tape.
    fn restore_and_replay_main(&mut self, _accepted: &[u32], _embedding: &Embedding) -> Result<()> {
        if let Some(e) = self.replay_err.take() {
            return Err(e);
        }
        Ok(())
    }

    // Committed-history commit.
    //
    // v1 (`!use_committed`): no-op.
    //
    // v2 (`use_committed`): append the M newly committed tokens' EXACT K/V to
    // the persistent MTP cache via one multi-token drafter forward.
    fn commit_mtp(
        &mut self,
        anchor: mtp_decode::MtpCommitAnchor,
        seed_hidden: &MxArray,
        verify_hiddens: &MxArray,
        committed_ids: &[u32],
        _k_accepted: usize,
        embedding: &Embedding,
    ) -> Result<()> {
        if !self.use_committed {
            return Ok(());
        }
        let m = committed_ids.len();
        if m == 0 {
            return Ok(());
        }
        let hidden_dim = verify_hiddens.shape_at(2)?;

        // Assemble hidden_seq [1, M, hidden] per anchor.
        let hidden_seq = match anchor {
            mtp_decode::MtpCommitAnchor::IncludeAnchor => {
                // seed_hidden ++ verify_hiddens[:, 0..M-1, :].
                let vh_prefix =
                    verify_hiddens.slice(&[0, 0, 0], &[1, (m - 1) as i64, hidden_dim])?;
                MxArray::concatenate(seed_hidden, &vh_prefix, 1)?
            }
            mtp_decode::MtpCommitAnchor::SkipAlreadyCommittedAnchor => {
                // verify_hiddens[:, 0..M, :].
                verify_hiddens.slice(&[0, 0, 0], &[1, m as i64, hidden_dim])?
            }
        };

        // Gather the M committed-token input embeddings → [1, M, hidden].
        let ids_i32: Vec<i32> = committed_ids.iter().map(|&v| v as i32).collect();
        let ids_arr = MxArray::from_int32(&ids_i32, &[m as i64])?;
        let gathered = embedding.forward(&ids_arr)?;
        let emb_seq = gathered.reshape(&[1, m as i64, hidden_dim])?;

        // Drop this cycle's draft K/V (written past committed_len by the draft
        // steps), then write the exact committed K/V via one multi-token
        // forward.
        let inner = &mut *self.inner;
        let mtp = inner.mtp.as_mut().ok_or_else(|| {
            Error::from_reason(
                "eager MTP commit_mtp: inner.mtp is None despite \
                 has_mtp_weights() gate",
            )
        })?;
        let caches = &mut self.mtp_caches;
        for c in caches.iter_mut() {
            if let Some(kv) = c.as_kv_cache_mut() {
                kv.trim(self.committed_len);
            }
        }
        let _ = mtp.forward(&hidden_seq, &emb_seq, Some(caches))?;
        self.committed_len += m as i32;
        Ok(())
    }

    // Re-anchor the drafter cache at the start of each cycle.
    //
    // v1 (`!use_committed`): reset to a fresh cache.
    //
    // v2 (`use_committed`): the cache is PERSISTENT; truncate the prior
    // cycle's draft tail back to the re-anchor target. `chained_anchor`
    // cycles anchor one slot earlier (`committed_len - 1`); Step-A cycles at
    // `committed_len`.
    fn begin_cycle(&mut self, chained_anchor: bool) {
        if !self.use_committed {
            self.mtp_caches = Qwen3_5MTPModule::fresh_caches(&self.config);
            return;
        }
        let target = if chained_anchor {
            (self.committed_len - 1).max(0)
        } else {
            self.committed_len
        };
        for c in self.mtp_caches.iter_mut() {
            if let Some(kv) = c.as_kv_cache_mut() {
                kv.trim(target);
            }
        }
    }

    // Per-cycle paged twin of `reserve_paged_mtp_lookahead`: re-reserve the
    // lookahead region past the adapter's CURRENT cursor so this cycle's
    // verify writes land in pre-allocated blocks (the turn-entry reservation
    // only covered the first cycle's cursor). The mechanism — adapter
    // `reserve_rows` plus the capacity-exhaustion mapping — lives once, in
    // the stepper's `SpecPagedCache::reserve_lookahead`; this hook only
    // names the turn's active sequence for it. Exhaustion reports AR
    // fallback with untouched adapter state; the flat mode has no
    // reservation semantics and is always covered.
    fn reserve_cycle_lookahead(&mut self, rows: usize) -> Result<bool> {
        let Some(owner) = self.owner else {
            return Ok(true);
        };
        let seq_id = owner.seq_id();
        crate::engine::spec_paged::SpecPagedCache::reserve_lookahead(self, seq_id, rows).map_err(
            |e| {
                Error::from_reason(format!(
                    "eager paged MTP per-cycle lookahead reservation ({rows} rows): {e}"
                ))
            },
        )
    }

    // Bound the lazy graph: materialize the token plus the main GDN/full-attn
    // caches; on a budget-forced step also the logits.
    fn eval_step(&self, token: &MxArray, logits: &MxArray, budget_forced: bool) {
        async_eval_layer_caches(&self.inner.caches);
        token.eval();
        if budget_forced {
            logits.eval();
        }
    }

    // Chained end-of-iteration eval: keep the chained `verify_hidden[K]` slice
    // materialized alongside the token and the main caches so the next cycle's
    // draft graph does not force a separate Metal roundtrip.
    fn eval_step_with_chained_hidden(&self, token: &MxArray, chained_hidden: &MxArray) {
        async_eval_layer_caches(&self.inner.caches);
        MxArray::async_eval_arrays(&[token, chained_hidden]);
    }

    fn rollback_unemitted(&mut self, unemitted: usize) {
        let Some(owner) = self.owner else {
            if unemitted > 0 {
                self.mtp_desynced = true;
            }
            return;
        };
        // Truncate the live paged adapter by the accepted-but-unemitted
        // tokens; the paged path never sets the FLAT desync latch. An adapter
        // truncate failure is not fatal here: the epilogue's frontier check
        // sees the skew and refuses to persist.
        if let Err(e) = owner
            .resolve(&mut self.inner.paged_adapter, DENSE_PAGED_MTP)
            .map_err(Error::from_reason)
            .and_then(|adapter| {
                adapter
                    .rollback_last_tokens(unemitted as u32)
                    .map_err(|e| Error::from_reason(format!("adapter rollback: {e}")))
            })
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "eager MTP-paged rollback_unemitted({unemitted}) adapter \
                 truncate failed (epilogue frontier check refuses to \
                 persist): {e}",
            );
        }
        // Paged GDN rewind twin of the adapter truncate above: replay the
        // retained snapshot + tape to `last_cycle_steps - unemitted` steps so
        // the recurrent state lands on the SAME drop-last-of-emitted frontier
        // as the adapter and the to-be-saved history. The skew vs the saved
        // history is exactly `unemitted` tokens (the history also drops the
        // last emitted token, which the adapter/GDN never consumed);
        // `unemitted == last_cycle_steps` degenerates to a pure snapshot
        // restore (a stop before any cycle token was emitted). A replay
        // failure is stashed in `replay_err` — the engine polls
        // `take_replay_error` right after this hook and fail-closes the turn
        // through `invalidate_dense_paged_session`.
        if self.replay_err.is_some() {
            // The turn is already failing; the stashed error aborts it and
            // invalidates the session, so a second (snapshot-less) replay
            // attempt would only shadow the root cause.
            return;
        }
        let Some(target) = self.last_cycle_steps.checked_sub(unemitted) else {
            self.inner.paged_mtp_gdn_invalidations += 1;
            self.recurrent_frontier = None;
            self.replay_err = Some(Error::from_reason(format!(
                "eager MTP-paged rollback_unemitted: unemitted {unemitted} exceeds \
                 the cycle's committed steps {}",
                self.last_cycle_steps
            )));
            return;
        };
        match self.replay_main_linear_to(target) {
            Ok(()) => {
                self.recurrent_frontier = self
                    .recurrent_snapshot_base
                    .map(|base| base + target as u64);
                self.inner.paged_mtp_gdn_rewinds += 1;
            }
            Err(e) => {
                self.recurrent_frontier = None;
                self.inner.paged_mtp_gdn_invalidations += 1;
                self.replay_err = Some(e);
            }
        }
    }

    fn frontier(&self) -> Option<SpecFrontier> {
        Some(SpecFrontier {
            attn_tokens: self.attention_frontier()?,
            recurrent_tokens: self.recurrent_frontier,
        })
    }

    fn take_replay_error(&mut self) -> Option<Error> {
        self.replay_err.take()
    }

    fn into_desynced(self) -> bool {
        // Paged rewinds BOTH sides of a mid-cycle stop in `rollback_unemitted`
        // — the adapter cursor by truncation and the GDN recurrent state by
        // tape replay — so every paged target-state kind already sits at the
        // drop-last-of-emitted frontier and reporting `false` is honest; it
        // never touches the FLAT desync latch. A rewind failure routes through
        // `take_replay_error` → session invalidation instead. (`self` is
        // consumed by value rather than destructured because the `Drop` impl
        // forbids moving fields out.)
        self.owner.is_none() && self.mtp_desynced
    }
}

/// Context prefix for every owner-addressed refusal on this family's paged
/// speculative path.
const DENSE_PAGED_MTP: &str = "dense paged MTP facade";

/// The refusal both facade writers return for this family; see the
/// `SpecPagedCache` impl docs below.
fn dense_core_writes_its_own_rows(seq_id: u32) -> String {
    format!(
        "dense paged MTP facade: the verify core records sequence {seq_id}'s rows \
         itself — open the cycle with open_core_write_cycle around that write \
         instead of record_verify/record_rows"
    )
}

/// Facade conformance for the dense paged MTP turn (`engine::spec_paged`):
/// the cache is the model's paged adapter, borrowed back through the turn's
/// [`SpecOwner`] ([`DenseMtpStepper::paged_cache_for`] refuses any other id).
///
/// # What production routes through here, and what it does not
///
/// PRODUCTION-ROUTED — these are the turn's only path, so conformance and
/// production cannot drift:
///
/// * `reserve_lookahead` — [`MtpStepper::reserve_cycle_lookahead`] delegates
///   here for the mechanism ([`PagedKVCacheAdapter::reserve_rows`] plus the
///   capacity-exhaustion mapping) and only names the active sequence.
/// * `open_core_write_cycle` + `commit_cycle` — the dense verify core
///   records its `[anchor, drafts..]` slice as part of the forward, so
///   [`MtpStepper::verify_step`] opens the cycle AROUND that write and
///   [`MtpStepper::rollback`] closes it. The commit is the adapter half of
///   that rollback; the GDN tape replay and the recurrent frontier update
///   are layered on top of it and stay the stepper's.
///
/// REFUSED — the verify core writes this family's rows:
///
/// * `record_verify` and the `record_rows` primitive under it would write
///   rows the verify forward never wrote. Paired with `verify_step` they
///   record the cycle's rows twice; on their own, every row a commit KEEPS
///   advances the adapter while the recurrent state stands still, and the
///   two frontiers desync — the skew
///   [`Qwen35Inner::check_dense_paged_frontier`] cannot see, since it
///   compares the adapter against the history and never against the GDN
///   count. Both return `Err`, so `open_core_write_cycle` is the only way to
///   open a cycle here and no facade row can outlive one.
///
/// NOT production-routed — conformance surface only:
///
/// * `settle_committed` / `settle_captures_durable_state` — see below.
///
/// `rollback_unemitted` retracts COMMITTED rows after a mid-cycle stop, past
/// the end of the cycle the commit already closed, so it stays a direct
/// [`PagedKVCacheAdapter::rollback_last_tokens`] and is outside this
/// contract.
///
/// `settle_committed` is the IDENTITY for this family, by construction: the
/// dense adapter is full-attention-only (`sliding_window == 0`, so no
/// per-step prune exists), and every durable surface — GDN history
/// checkpoints, cold sidecars, the paged-history save, prefix registration —
/// runs in the turn epilogue at the committed frontier (I3, enforced by the
/// `paged_gdn_state_dirty` latch and the epilogue length agreement), never
/// per step. There is no settle work to re-anchor, so the method only
/// validates its arguments and touches nothing. Conformance tests for this
/// family therefore MUST NOT gate on settle side effects — they would pass
/// vacuously — and assert the reservation/commit block accounting instead.
impl crate::engine::spec_paged::SpecPagedCache for DenseMtpStepper<'_> {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> std::result::Result<bool, String> {
        let rows = u32::try_from(rows).unwrap_or(u32::MAX);
        match self.paged_cache_for(seq_id)?.reserve_rows(rows) {
            Ok(_) => Ok(true),
            Err(e) if e.starts_with("context_length_exceeded:") => {
                tracing::warn!(
                    target: "mlx_core::qwen3_5::paged",
                    lookahead_rows = rows,
                    "dense paged MTP lookahead reservation exhausted the paged \
                     pool; the cycle degrades to autoregressive decode: {e}"
                );
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    fn record_rows(&mut self, seq_id: u32, _tokens: &[u32]) -> std::result::Result<(), String> {
        Err(dense_core_writes_its_own_rows(seq_id))
    }

    fn record_verify(
        &mut self,
        seq_id: u32,
        _tokens: &[u32],
    ) -> std::result::Result<crate::engine::spec_paged::VerifyTicket, String> {
        Err(dense_core_writes_its_own_rows(seq_id))
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> std::result::Result<(), String> {
        let rows = u32::try_from(rows).map_err(|_| {
            format!("dense paged MTP commit rollback of {rows} rows does not fit u32")
        })?;
        self.paged_cache_for(seq_id)?.rollback_last_tokens(rows)
    }

    fn settle_committed(
        &mut self,
        seq_id: u32,
        committed_tokens: u64,
    ) -> std::result::Result<(), String> {
        // The identity (see the impl docs): validate only, touch nothing.
        let recorded = self.paged_cache_for(seq_id)?.request_tokens().len() as u64;
        if committed_tokens > recorded {
            return Err(format!(
                "dense paged MTP settle_committed: committed frontier \
                 {committed_tokens} exceeds recorded token count {recorded}"
            ));
        }
        Ok(())
    }

    fn settle_captures_durable_state(&self) -> bool {
        // The identity settle (see the impl docs) touches nothing at all.
        false
    }

    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.owner
            .filter(|owner| owner.seq_id() == seq_id)
            .and_then(|_| MtpStepper::frontier(self))
    }
}

impl MtpBackend for Qwen35Inner {
    type MtpDecode<'a>
        = DenseMtpStepper<'a>
    where
        Self: 'a;

    fn begin_mtp_decode(&mut self, setup: &MtpTurnSetup<'_>) -> Result<Self::MtpDecode<'_>> {
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let seed_trace_start = inference_info_enabled.then(std::time::Instant::now);
        let mut seed_tokens = 0usize;
        let mut seed_chunks = 0usize;
        // Turn-constant captures the packed-aware embedding backend and a
        // config clone for the per-cycle drafter cache reset.
        let embedding = self.embedding.clone();
        let config = self.config.clone();

        // Committed history needs the prompt seed itself, not just a prompt.
        // Cache-reuse continuations do not capture full-prompt hiddens;
        // advertising an empty drafter cache as prompt-committed would let
        // chained cycles skip the main-model anchor against a nonexistent
        // history and diverge from the full-reprefill heal. Seedless turns
        // fall back to cycle history.
        let has_prompt_seed = setup.prompt_hidden.is_some()
            && setup.prompt_hidden_ids.is_some_and(|ids| !ids.is_empty());
        let use_committed = has_prompt_seed;

        // Auto-select the main-forward routing: the paged cores leave a paged
        // adapter on `self`, so the turn claims its active sequence as the
        // owner and borrows the adapter back per touch; the flat cores have
        // none and run flat. The paged forwards need the per-layer kind
        // classification (unused flat).
        let (owner, layer_kinds) = match self.paged_adapter.as_mut() {
            Some(adapter) => {
                let owner = SpecOwner::claim(adapter.active_seq_id(), "eager paged MTP turn entry")
                    .map_err(Error::from_reason)?;
                // Reserve the speculative lookahead region before any cycle
                // writes (I1: `setup.lookahead_rows` comes from the
                // `SpeculativePlan` property — never a local `depth + 1`).
                // The paged cores reserved this same margin at their
                // AR-fallback gate, so this normally takes the covered no-op
                // branch; a caller that skips that gate still fails HERE,
                // pre-cycle with untouched state, instead of mid-verify.
                // Later cycles are covered by the engine loop's per-cycle
                // `reserve_cycle_lookahead` call on the stepper.
                if setup.lookahead_rows > 0 {
                    let rows = u32::try_from(setup.lookahead_rows).unwrap_or(u32::MAX);
                    adapter.reserve_rows(rows).map_err(|e| {
                        Error::from_reason(format!(
                            "eager paged MTP lookahead reservation ({rows} rows): {e}"
                        ))
                    })?;
                }
                // Cached once at construction (see the field rustdoc); clone is
                // a copy of the turn-constant classification.
                (Some(owner), self.layer_kinds.clone())
            }
            None => (None, Vec::new()),
        };

        let mut stepper = DenseMtpStepper {
            inner: self,
            mtp_caches: Qwen3_5MTPModule::fresh_caches(&config),
            committed_len: 0,
            use_committed,
            chained_cycles_supported: has_prompt_seed,
            snap: None,
            tape: Vec::new(),
            last_cycle_steps: 0,
            open_cycle: None,
            recurrent_frontier: None,
            recurrent_snapshot_base: None,
            replay_err: None,
            mtp_desynced: false,
            embedding,
            config,
            owner,
            layer_kinds,
        };
        // The GDN recurrent state is aligned with the attention state at turn
        // entry (both consumed exactly the prefilled history), so the
        // recurrent bookkeeping seeds from the attention side's ground truth;
        // cross-turn skew is the epilogue length-agreement checks' job.
        stepper.recurrent_frontier = stepper.attention_frontier();

        // Prompt-prefix seed (v2 committed-history only): commit the
        // contiguous run
        // `[prompt_hidden_ids[1..], y]` (length P, token 0 skipped, the first
        // sampled token `y` appended) into the persistent MTP cache so the
        // drafter attends the prompt from cycle 1. Each committed token `x` is
        // paired with `prompt_hidden[:, idx, :]` = h(token before `x`). Chunk
        // into pieces of size <= 7 and run the SAME multi-token eager KV-writer
        // as `commit_mtp` per chunk. `position_base == 0` is guaranteed by
        // `use_committed`, so RoPE (= local cache offset) aligns with absolute
        // position.
        if use_committed
            && let (Some(ph), Some(ph_ids)) = (setup.prompt_hidden, setup.prompt_hidden_ids)
            && !ph_ids.is_empty()
        {
            let prompt_len = ph_ids.len();
            let hidden_dim = ph.shape_at(2)?;
            let hidden_len = ph.shape_at(1)? as usize;
            if hidden_len != prompt_len {
                return Err(Error::from_reason(format!(
                    "eager MTP prompt-seed: prompt_hidden length {hidden_len} \
                     does not match prompt_hidden_ids length {prompt_len}"
                )));
            }
            // The first sampled token `y` is supplied by the engine via the
            // setup so the prompt seed can commit `[prompt_ids[1..], y]`.
            let y_id = setup.first_sampled_token;

            // Committed run = [prompt_ids[1..prompt_len], y] (length P).
            let mut committed_ids: Vec<i32> = Vec::with_capacity(prompt_len);
            committed_ids.extend(ph_ids[1..prompt_len].iter().map(|&v| v as i32));
            committed_ids.push(y_id as i32);

            let chunk_sizes = partition_prefill_chunks(prompt_len);
            seed_tokens = prompt_len;
            seed_chunks = chunk_sizes.len();
            let mut cursor: usize = 0;
            for &chunk in &chunk_sizes {
                let chunk_i64 = chunk as i64;
                let start = cursor as i64;
                // hidden_seq = prompt_hidden[:, cursor..cursor+chunk, :].
                let hidden_seq = ph.slice(&[0, start, 0], &[1, start + chunk_i64, hidden_dim])?;
                // emb_seq = gather embedding rows for the chunk's ids.
                let ids_arr =
                    MxArray::from_int32(&committed_ids[cursor..cursor + chunk], &[chunk_i64])?;
                let gathered = stepper.embedding.forward(&ids_arr)?;
                let emb_seq = gathered.reshape(&[1, chunk_i64, hidden_dim])?;

                let inner = &mut *stepper.inner;
                let mtp = inner.mtp.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "eager MTP prompt-seed: inner.mtp is None despite \
                         has_mtp_weights() gate",
                    )
                })?;
                let caches = &mut stepper.mtp_caches;
                let _ = mtp.forward(&hidden_seq, &emb_seq, Some(caches))?;
                stepper.committed_len += chunk as i32;
                cursor += chunk;
            }
        }

        if inference_info_enabled {
            tracing::info!(
                target: "mlx_core::inference",
                event = "mtp_prompt_seed_done",
                mode = if use_committed {
                    "committed_history"
                } else {
                    "cycle_history"
                },
                seed_tokens,
                chunks = seed_chunks,
                max_chunk_tokens = 7,
                setup_elapsed_ms = seed_trace_start.map(elapsed_ms).unwrap_or(0.0),
                "MTP prompt seed completed"
            );
        }

        Ok(stepper)
    }

    fn record_turn_mtp_acceptance(&mut self, accepted: u64, attempted: u64) {
        // Aggregate across turns so the confidence-aware gate decision has
        // a growing sample — but bound it so a long healthy phase cannot
        // drown out a later degradation (see `mtp_bound_gate_history`).
        self.mtp_draft_accepted += accepted;
        self.mtp_draft_attempted += attempted;
        mtp_decode::mtp_bound_gate_history(
            &mut self.mtp_draft_accepted,
            &mut self.mtp_draft_attempted,
        );
        self.mtp_gated_turns = 0;
    }
}

impl Qwen35Inner {
    pub(super) fn mtp_gate_allows(&mut self, requested_depth: u32) -> bool {
        mtp_decode::mtp_gate_allows(
            &mut self.mtp_draft_accepted,
            &mut self.mtp_draft_attempted,
            &mut self.mtp_gated_turns,
            requested_depth,
        )
    }
}
