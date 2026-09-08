//! Speculative (MTP) decode surface for Qwen3.5 MoE: the flat decode
//! stepper, the propose/verify stepper driving both lanes, and the
//! `MtpBackend` admission gate.

use super::*;

/// Per-turn decode stepper for the engine's generic (text-only,
/// non-paged, non-MTP) flow on Qwen3.5 MoE
/// ([`ChatBackend::begin_decode`]).
///
/// Drives the pure-Rust `forward_inner` over the flat caches.
pub(crate) struct Qwen35MoeDecode<'a> {
    pub(super) inner: &'a mut Qwen35MoeInner,
    pub(super) embedding: Embedding,
    /// Decode-path profiler relabel (`moe_chat_*_rust` and its streaming /
    /// delta variants), resolved in `begin_decode` from the turn's
    /// streaming-ness and delta-ness.
    pub(super) relabel: &'static str,
}

impl DecodeStep for Qwen35MoeDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let inner = &mut *self.inner;
        let logits = forward_inner(
            input_ids,
            &self.embedding,
            &mut inner.layers,
            &mut inner.caches,
            &inner.final_norm,
            &inner.lm_head,
            inner.fa_idx,
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

/// Per-turn MTP propose/verify stepper for the MoE family, driving BOTH the
/// FLAT and the block-PAGED lane of [`crate::engine::mtp_turn::run_mtp_turn`],
/// selected by [`Self::owner`].
///
/// With no owner the eager pre-norm forwards run against `inner.caches`; with
/// one, the main Step-A / verify forwards route through `inner.paged_adapter`
/// ([`crate::models::qwen3_5_moe::paged_forward::run_paged_step_with_hidden`] /
/// [`crate::models::qwen3_5_moe::paged_forward::run_paged_verify_step`]) while the GDN recurrent
/// state stays FLAT in `inner.caches` Linear slots. Only four methods
/// (`forward_with_hidden`, `verify_step`, `rollback`, `rollback_unemitted`)
/// branch on it.
///
/// The adapter stays ON THE MODEL for the whole turn: each paged touch
/// borrows it back through [`SpecOwner::resolve`], which refuses once the
/// adapter's active request is no longer this turn's.
///
/// History policy is CYCLE history: [`Self::begin_cycle`] rebuilds a fresh
/// drafter cache every cycle and [`Self::commit_mtp`] is a no-op. The dense
/// family's committed-history mode is gated on a prompt-hidden seed
/// (`MtpTurnSetup::prompt_hidden`), and no MoE call site can produce one —
/// there is no MoE hidden-emitting prefill — so the mode's own precondition is
/// unsatisfiable here.
pub(crate) struct MoeMtpStepper<'a> {
    /// The model — owns layers / caches / mtp / final_norm / lm_head and the
    /// `flat_mtp_caches_desynced` latch.
    inner: &'a mut Qwen35MoeInner,
    /// Drafter K/V caches, rebuilt fresh by [`Self::begin_cycle`].
    mtp_caches: Vec<Qwen3_5LayerCache>,
    /// Pre-verify snapshot of the main caches, taken in
    /// [`Self::snapshot_main_linear`], consumed by [`Self::rollback`] and the
    /// paged [`Self::rollback_unemitted`].
    snap: Option<Result<Vec<crate::models::qwen3_5_moe::layer_cache::Qwen3_5LayerSnapshot>>>,
    /// GDN tape recorded by [`Self::verify_step`], consumed by the same two
    /// rewinds.
    tape: Vec<Option<crate::models::qwen3_5_moe::gated_delta_net::GdnLayerTape>>,
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
    /// retracted, so the commit arithmetic cannot drift from the conformance
    /// surface. `None` in flat mode and between cycles; a paged cycle that
    /// cannot be opened refuses the verify rather than writing without one.
    pub(super) open_cycle: Option<crate::engine::spec_paged::VerifyTicket>,
    /// Absolute tokens the GDN recurrent state has consumed, reported by
    /// [`MtpStepper::frontier`] against the ATTENTION side's ground truth
    /// (adapter recorded rows / flat full-attention offset). Seeded from that
    /// frontier at construction and moved ONLY where the recurrent state
    /// actually moves: `forward_with_hidden` +1, `verify_step` +depth+1,
    /// replay-driven rollbacks to `recurrent_snapshot_base + steps`. `None`
    /// after a failed replay — the count is then unknown and the stashed error
    /// fail-closes the turn.
    recurrent_frontier: Option<u64>,
    /// [`Self::recurrent_frontier`] captured by `snapshot_main_linear` — the
    /// base the replay-driven rollbacks land relative to.
    recurrent_snapshot_base: Option<u64>,
    /// Error stashed by the infallible [`Self::rollback`] replay, surfaced by
    /// [`Self::take_replay_error`].
    replay_err: Option<Error>,
    /// Mid-cycle-stop desync latch (set by the FLAT [`Self::rollback_unemitted`]),
    /// reported by [`Self::into_desynced`].
    mtp_desynced: bool,
    /// The model's embedding lookup and tied-head projection backend.
    embedding: Embedding,
    /// Config clone for the per-cycle drafter cache reset.
    config: Qwen3_5MoeConfig,
    /// Index of the first full-attention layer, threaded into the MoE eager
    /// flat forwards.
    fa_idx: usize,
    /// The sequence this turn's paged main-forwards belong to, claimed at
    /// `begin_mtp_decode`. `None` runs the flat main path.
    pub(super) owner: Option<SpecOwner>,
    /// Per-layer attention/linear classification consumed by the paged
    /// forwards. Empty on the flat path (unused there).
    layer_kinds: Vec<crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind>,
}

impl Drop for MoeMtpStepper<'_> {
    fn drop(&mut self) {
        // A turn that failed between the verify write and its rollback (any
        // `?` in the engine's accept path) leaves the cycle open; close it here
        // so the ticket's abandoned-cycle guard does not turn that error return
        // into a debug-build panic.
        self.close_abandoned_cycle();
    }
}

/// Context prefix for every owner-addressed refusal on this family's paged
/// speculative path.
const MOE_PAGED_MTP: &str = "MoE paged MTP facade";

/// The refusal both facade writers return for this family; see the
/// `SpecPagedCache` impl docs below.
fn moe_core_writes_its_own_rows(seq_id: u32) -> String {
    format!(
        "MoE paged MTP facade: the verify core records sequence {seq_id}'s rows \
         itself — open the cycle with open_core_write_cycle around that write \
         instead of record_verify/record_rows"
    )
}

impl MoeMtpStepper<'_> {
    /// Restore the pre-verify snapshot and replay the first `steps` recorded
    /// tape steps into the live main caches — the shared GDN replay both
    /// `rollback` (to `accepted_steps`) and the paged `rollback_unemitted`
    /// (to `last_cycle_steps - unemitted`) drive.
    fn replay_main_linear_to(&mut self, steps: usize) -> Result<()> {
        let paged = self.owner.is_some();
        let snap = match self.snap.as_ref() {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                return Err(Error::from_reason(format!(
                    "eager MoE MTP replay: snapshot failed: {}",
                    e.reason
                )));
            }
            None => {
                return Err(Error::from_reason(
                    "eager MoE MTP replay: snapshot missing (snapshot_main_linear \
                     did not run)",
                ));
            }
        };
        let tape = &self.tape;
        let inner = &mut *self.inner;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("eager MoE MTP replay: inner.caches is None"))?;
        crate::models::qwen3_5_moe::layer_cache::replay_mtp_snapshot_to(
            caches,
            snap,
            tape,
            steps,
            paged,
            "eager MoE MTP replay",
        )
    }

    /// The attention side's ground-truth frontier: the paged adapter's
    /// recorded rows, or the flat full-attention cache's offset.
    fn attention_frontier(&self) -> Option<u64> {
        match self.owner {
            Some(owner) => Some(
                owner
                    .resolve_ref(&self.inner.paged_adapter, MOE_PAGED_MTP)
                    .ok()?
                    .request_tokens()
                    .len() as u64,
            ),
            None => self
                .inner
                .caches
                .as_ref()?
                .get(self.fa_idx)
                .map(|cache| cache.offset().max(0) as u64),
        }
    }

    /// The paged speculative cache the facade (`engine::spec_paged`)
    /// addresses: the model's adapter, borrowed back through the turn's
    /// owner, iff `seq_id` is that owner.
    fn paged_cache_for(
        &mut self,
        seq_id: u32,
    ) -> std::result::Result<&mut PagedKVCacheAdapter, String> {
        let owner = self.owner.ok_or_else(|| {
            format!("{MOE_PAGED_MTP}: a flat turn has no paged speculative cache")
        })?;
        owner.accepts(seq_id, MOE_PAGED_MTP)?;
        owner.resolve(&mut self.inner.paged_adapter, MOE_PAGED_MTP)
    }

    /// The paged half of [`MtpStepper::verify_step`]: slice `ids` to exactly
    /// the `depth + 1` rows the core records, OPEN the facade cycle around
    /// that write, then run it. The cycle is opened AFTER the fallible id work
    /// so a malformed `ids` never mints a ticket, and BEFORE the write so the
    /// ticket's basis is the pre-write cursor.
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
        self.open_verify_cycle(owner, depth + 1)?;
        let inner = &mut *self.inner;
        let adapter = owner
            .resolve(&mut inner.paged_adapter, "eager paged MoE MTP verify_step")
            .map_err(Error::from_reason)?;
        // Cross-turn M-RoPE delta carried by a text turn that warm-continues an
        // image prefill; 0 for pure-text sessions.
        let rope_deltas = inner.cached_rope_deltas.unwrap_or(0);
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("eager paged MoE MTP verify_step: caches is None"))?;
        let tape = &mut self.tape;
        crate::models::qwen3_5_moe::paged_forward::run_paged_verify_step(
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
    /// A mint that fails REFUSES the verify. The only way to reach it is an
    /// adapter that no longer answers to this turn's owner — whose row write
    /// would fail a few lines later anyway — and proceeding without a ticket
    /// would leave the retraction to something other than
    /// [`crate::engine::spec_paged::SpecPagedCache::commit_cycle`], which is
    /// where the commit arithmetic lives.
    pub(super) fn open_verify_cycle(&mut self, owner: SpecOwner, rows: usize) -> Result<()> {
        self.close_abandoned_cycle();
        let seq_id = owner.seq_id();
        let ticket =
            crate::engine::spec_paged::SpecPagedCache::open_core_write_cycle(self, seq_id, rows)
                .map_err(|e| {
                    Error::from_reason(format!(
                        "eager MoE MTP-paged verify cycle ({rows} rows) on sequence \
                         {seq_id} could not be opened: {e}"
                    ))
                })?;
        self.open_cycle = Some(ticket);
        Ok(())
    }

    /// Close a cycle abandoned between its verify write and its rollback.
    /// Keeping every written row is the fail-closed answer: they stay recorded
    /// past the emitted frontier and the epilogue's frontier check refuses to
    /// persist the turn, so a full keep retracts nothing and this consumes the
    /// ticket instead of letting its abandoned-cycle guard fire.
    fn close_abandoned_cycle(&mut self) {
        let Some(ticket) = self.open_cycle.take() else {
            return;
        };
        let (seq_id, rows) = (ticket.seq_id(), ticket.rows());
        if let Err(e) =
            crate::engine::spec_paged::SpecPagedCache::commit_cycle(self, seq_id, ticket, rows)
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5_moe::paged",
                "eager MoE MTP-paged abandoned verify cycle on sequence {seq_id} \
                 ({rows} rows) could not be closed cleanly (ignored): {e}",
            );
        }
    }
}

impl MtpStepper for MoeMtpStepper<'_> {
    fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    fn committed_history_active(&self) -> bool {
        // Cycle history: the drafter cache is rebuilt every cycle and
        // `commit_mtp` is a no-op (see the struct doc).
        false
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        Some("moe_mtp_eager")
    }

    // Step A main forward: eager pre-norm + final-norm + project. Returns
    // `hidden` shaped `[1, hidden]` (squeeze the time axis); `logits` stays
    // `[1, 1, vocab]` with `needs_squeeze = true`.
    fn forward_with_hidden(
        &mut self,
        ids: &MxArray,
        embedding: &Embedding,
    ) -> Result<(MxArray, MxArray, bool)> {
        let output = match self.owner {
            None => {
                let inner = &mut *self.inner;
                let pre = forward_pre_norm_inner(
                    ids,
                    embedding,
                    &mut inner.layers,
                    &mut inner.caches,
                    self.fa_idx,
                )?;
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
                        "eager paged MoE MTP forward_with_hidden",
                    )
                    .map_err(Error::from_reason)?;
                // Cross-turn M-RoPE delta carried by a text turn that warm-
                // continues an image prefill; 0 for pure-text sessions.
                let rope_deltas = inner.cached_rope_deltas.unwrap_or(0);
                let caches = inner.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("eager paged MoE MTP forward_with_hidden: caches is None")
                })?;
                let (logits, hidden) =
                    crate::models::qwen3_5_moe::paged_forward::run_paged_step_with_hidden(
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
                "eager MoE MTP draft_step: inner.mtp is None despite \
                 has_mtp_weights() gate",
            )
        })?;
        let h_next = mtp.forward(prev_hidden, prev_emb, Some(mtp_caches))?;
        let dl3 = project_logits_from_hidden(&h_next, &inner.lm_head, &self.embedding)?;
        let draft_logits = dl3.squeeze(Some(&[1]))?;
        Ok((h_next, draft_logits))
    }

    // Batched verify: run the K+1 verify ids through the main stack,
    // advancing the caches by K+1, recording the GDN tape. The paged half
    // also opens the facade cycle its core write belongs to
    // ([`MoeMtpStepper::paged_verify_step`]); `rollback` closes it.
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
                self.fa_idx,
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
        // shells — `rollback` rewinds those via the adapter.
        let paged = self.owner.is_some();
        let inner = &*self.inner;
        let snap = match inner.caches.as_ref() {
            Some(caches) => {
                crate::models::qwen3_5_moe::layer_cache::snapshot_all_mtp(caches, paged)
            }
            None => Err(Error::from_reason(
                "eager MoE MTP snapshot_main_linear: inner.caches is None",
            )),
        };
        self.snap = Some(snap);
        self.recurrent_snapshot_base = self.recurrent_frontier;
    }

    // Pure-Rust GDN tape replay — fires on BOTH full and partial accept.
    // Infallible signature: any error is stashed in `self.replay_err` and
    // surfaced later.
    fn rollback(&mut self, accepted_drafts: usize, depth: usize) {
        if self.replay_err.is_some() {
            return;
        }
        // The paged path rewinds the full-attention K/V (which lives in the
        // paged pool, not `inner.caches`) before the shared GDN tape replay,
        // and only through the cycle `verify_step` opened — so the adapter
        // half of this rollback and the facade's conformance surface are the
        // same arithmetic: the commit derives its rollback as `rows - keep` =
        // `(depth + 1) - (accepted_drafts + 1)`.
        if let Some(owner) = self.owner {
            match self.open_cycle.take() {
                Some(ticket) => {
                    let seq_id = ticket.seq_id();
                    // `keep` is CLAMPED to the rows the cycle wrote.
                    // `accepted_drafts <= depth` is an engine invariant; were
                    // it ever broken, the clamp keeps this on the commit's
                    // checked path — retracting zero rows — instead of leaving
                    // through an Err and a log line.
                    let keep = (accepted_drafts + 1).min(ticket.rows());
                    if let Err(e) = crate::engine::spec_paged::SpecPagedCache::commit_cycle(
                        self, seq_id, ticket, keep,
                    ) {
                        tracing::warn!(
                            target: "mlx_core::qwen3_5_moe::paged",
                            "eager MoE MTP-paged verify commit (keep {keep} of the cycle's \
                             rows) failed (ignored): {e}",
                        );
                    }
                }
                None => {
                    // `verify_step` refuses to write rows it could not open a
                    // cycle for, so the rows this would retract cannot exist.
                    // A stepper that lands here disagrees with its own cache:
                    // fail the turn closed rather than retract by hand, which
                    // would be a second copy of the arithmetic `commit_cycle`
                    // is here to single-source.
                    self.replay_err = Some(Error::from_reason(format!(
                        "eager MoE MTP-paged rollback on sequence {} found no open verify \
                         cycle ({accepted_drafts} of {depth} drafts accepted); refusing to \
                         retract the adapter outside the facade",
                        owner.seq_id()
                    )));
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
                // A refusal stashed above is the root cause; a replay that
                // also fails on top of it must not shadow it.
                self.replay_err.get_or_insert(e);
            }
        }
    }

    // On rejection (partial accept): the GDN tape replay in `rollback` already
    // reconstructed the AR-exact main cache state, so no re-forward loop is
    // needed. This only surfaces a stashed replay error.
    //
    // The per-cycle snapshot + tape are deliberately RETAINED: both are
    // re-armed by the next cycle anyway (`snapshot_main_linear` overwrites
    // `snap`; the verify cores clear + re-record `tape` at record time), and a
    // mid-cycle stop after THIS cycle still needs them — the paged
    // `rollback_unemitted` replays the GDN state back to the emitted frontier
    // from exactly this snapshot + tape.
    fn restore_and_replay_main(&mut self, _accepted: &[u32], _embedding: &Embedding) -> Result<()> {
        if let Some(e) = self.replay_err.take() {
            return Err(e);
        }
        Ok(())
    }

    // Cycle history: nothing to commit into the drafter cache.
    fn commit_mtp(
        &mut self,
        _anchor: mtp_decode::MtpCommitAnchor,
        _seed_hidden: &MxArray,
        _verify_hiddens: &MxArray,
        _committed_ids: &[u32],
        _k_accepted: usize,
        _embedding: &Embedding,
    ) -> Result<()> {
        Ok(())
    }

    // Re-anchor the drafter cache at the start of each cycle: a fresh cache.
    fn begin_cycle(&mut self, _chained_anchor: bool) {
        self.mtp_caches = Qwen3_5MoeMTPModule::fresh_caches(&self.config);
    }

    // Per-cycle paged twin of the turn-entry lookahead reservation: re-reserve
    // the lookahead region past the adapter's CURRENT cursor so this cycle's
    // verify writes land in pre-allocated blocks. The mechanism lives once, in
    // the stepper's `SpecPagedCache::reserve_lookahead`; this hook only names
    // the turn's active sequence for it. Exhaustion reports AR fallback with
    // untouched adapter state; the flat mode has no reservation semantics and
    // is always covered.
    fn reserve_cycle_lookahead(&mut self, rows: usize) -> Result<bool> {
        let Some(owner) = self.owner else {
            return Ok(true);
        };
        let seq_id = owner.seq_id();
        crate::engine::spec_paged::SpecPagedCache::reserve_lookahead(self, seq_id, rows).map_err(
            |e| {
                Error::from_reason(format!(
                    "eager paged MoE MTP per-cycle lookahead reservation ({rows} rows): {e}"
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
            .resolve(&mut self.inner.paged_adapter, MOE_PAGED_MTP)
            .map_err(Error::from_reason)
            .and_then(|adapter| {
                adapter
                    .rollback_last_tokens(unemitted as u32)
                    .map_err(|e| Error::from_reason(format!("adapter rollback: {e}")))
            })
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5_moe::paged",
                "eager MoE MTP-paged rollback_unemitted({unemitted}) adapter \
                 truncate failed (epilogue frontier check refuses to persist): {e}",
            );
        }
        // Paged GDN rewind twin of the adapter truncate above: replay the
        // retained snapshot + tape to `last_cycle_steps - unemitted` steps so
        // the recurrent state lands on the SAME drop-last-of-emitted frontier
        // as the adapter and the to-be-saved history. `unemitted ==
        // last_cycle_steps` degenerates to a pure snapshot restore (a stop
        // before any cycle token was emitted). A replay failure is stashed in
        // `replay_err` — the engine polls `take_replay_error` right after this
        // hook and fail-closes the turn.
        if self.replay_err.is_some() {
            // The turn is already failing; the stashed error aborts it, so a
            // second (snapshot-less) replay attempt would only shadow the root
            // cause.
            return;
        }
        let Some(target) = self.last_cycle_steps.checked_sub(unemitted) else {
            self.inner.paged_mtp_gdn_invalidations += 1;
            self.recurrent_frontier = None;
            self.replay_err = Some(Error::from_reason(format!(
                "eager MoE MTP-paged rollback_unemitted: unemitted {unemitted} exceeds \
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
        // drop-last-of-emitted frontier and reporting `false` is honest. A
        // rewind failure routes through `take_replay_error` → session
        // invalidation instead. (`self` is consumed by value rather than
        // destructured because the `Drop` impl forbids moving fields out.)
        self.owner.is_none() && self.mtp_desynced
    }
}

/// Facade conformance for the MoE paged MTP turn (`engine::spec_paged`): the
/// cache is the model's paged adapter, borrowed back through the turn's
/// [`SpecOwner`] ([`MoeMtpStepper::paged_cache_for`] refuses any other id).
///
/// PRODUCTION-ROUTED — these are the turn's only path, so conformance and
/// production cannot drift:
///
/// * `reserve_lookahead` — [`MtpStepper::reserve_cycle_lookahead`] delegates
///   here for the mechanism ([`PagedKVCacheAdapter::reserve_rows`] plus the
///   capacity-exhaustion mapping) and only names the active sequence.
/// * `open_core_write_cycle` + `commit_cycle` — the MoE verify core records
///   its `[anchor, drafts..]` slice as part of the forward, so
///   [`MtpStepper::verify_step`] opens the cycle AROUND that write and
///   [`MtpStepper::rollback`] closes it.
///
/// REFUSED — the verify core writes this family's rows:
///
/// * `record_verify` and the `record_rows` primitive under it would write rows
///   the verify forward never wrote. Paired with `verify_step` they record the
///   cycle's rows twice; on their own, every row a commit KEEPS advances the
///   adapter while the recurrent state stands still, and the two frontiers
///   desync.
///
/// NOT production-routed — conformance surface only:
///
/// * `settle_committed` / `settle_captures_durable_state`. Settle is the
///   IDENTITY for this family, by construction: the MoE adapter is
///   full-attention-only (`sliding_window == 0`, so no per-step prune exists),
///   and every durable surface — GDN history checkpoints, cold sidecars, the
///   paged-history save, prefix registration — runs in the turn epilogue at
///   the committed frontier (I3), never per step. Conformance tests for this
///   family therefore MUST NOT gate on settle side effects.
impl crate::engine::spec_paged::SpecPagedCache for MoeMtpStepper<'_> {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> std::result::Result<bool, String> {
        let rows = u32::try_from(rows).unwrap_or(u32::MAX);
        match self.paged_cache_for(seq_id)?.reserve_rows(rows) {
            Ok(_) => Ok(true),
            Err(e) if e.starts_with("context_length_exceeded:") => {
                tracing::warn!(
                    target: "mlx_core::qwen3_5_moe::paged",
                    lookahead_rows = rows,
                    "MoE paged MTP lookahead reservation exhausted the paged pool; \
                     the cycle degrades to autoregressive decode: {e}"
                );
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    fn record_rows(&mut self, seq_id: u32, _tokens: &[u32]) -> std::result::Result<(), String> {
        Err(moe_core_writes_its_own_rows(seq_id))
    }

    fn record_verify(
        &mut self,
        seq_id: u32,
        _tokens: &[u32],
    ) -> std::result::Result<crate::engine::spec_paged::VerifyTicket, String> {
        Err(moe_core_writes_its_own_rows(seq_id))
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> std::result::Result<(), String> {
        let rows = u32::try_from(rows).map_err(|_| {
            format!("MoE paged MTP commit rollback of {rows} rows does not fit u32")
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
                "MoE paged MTP settle_committed: committed frontier {committed_tokens} \
                 exceeds recorded token count {recorded}"
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

impl MtpBackend for Qwen35MoeInner {
    type MtpDecode<'a>
        = MoeMtpStepper<'a>
    where
        Self: 'a;

    fn begin_mtp_decode(&mut self, setup: &MtpTurnSetup<'_>) -> Result<Self::MtpDecode<'_>> {
        let embedding = self.embedding.clone();
        let config = self.config.clone();
        let fa_idx = self.fa_idx;

        // Auto-select the main-forward routing: the paged cores leave a paged
        // adapter on `self`, so the turn claims its active sequence as the
        // owner and borrows the adapter back per touch; the flat cores have
        // none and run flat. The paged forwards need the per-layer kind
        // classification (unused flat).
        let (owner, layer_kinds) = match self.paged_adapter.as_mut() {
            Some(adapter) => {
                let owner =
                    SpecOwner::claim(adapter.active_seq_id(), "eager paged MoE MTP turn entry")
                        .map_err(Error::from_reason)?;
                // Reserve the speculative lookahead region before any cycle
                // writes (I1: `setup.lookahead_rows` comes from the
                // `SpeculativePlan` property — never a local `depth + 1`). The
                // paged core reserved this same margin at its AR-fallback
                // gate, so this normally takes the covered no-op branch; a
                // caller that skips that gate still fails HERE, pre-cycle with
                // untouched state, instead of mid-verify. Later cycles are
                // covered by the engine loop's per-cycle
                // `reserve_cycle_lookahead` call on the stepper.
                if setup.lookahead_rows > 0 {
                    let rows = u32::try_from(setup.lookahead_rows).unwrap_or(u32::MAX);
                    adapter.reserve_rows(rows).map_err(|e| {
                        Error::from_reason(format!(
                            "eager paged MoE MTP lookahead reservation ({rows} rows): {e}"
                        ))
                    })?;
                }
                (Some(owner), self.layer_kinds.clone())
            }
            None => (None, Vec::new()),
        };

        let mut stepper = MoeMtpStepper {
            inner: self,
            mtp_caches: Qwen3_5MoeMTPModule::fresh_caches(&config),
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
            fa_idx,
            owner,
            layer_kinds,
        };
        // The GDN recurrent state is aligned with the attention state at turn
        // entry (both consumed exactly the prefilled history), so the recurrent
        // bookkeeping seeds from the attention side's ground truth.
        stepper.recurrent_frontier = stepper.attention_frontier();

        Ok(stepper)
    }

    fn record_turn_mtp_acceptance(&mut self, accepted: u64, attempted: u64) {
        self.mtp_draft_accepted += accepted;
        self.mtp_draft_attempted += attempted;
        mtp_decode::mtp_bound_gate_history(
            &mut self.mtp_draft_accepted,
            &mut self.mtp_draft_attempted,
        );
        self.mtp_gated_turns = 0;
    }
}

impl Qwen35MoeInner {
    pub(super) fn mtp_gate_allows(&mut self, requested_depth: u32) -> bool {
        mtp_decode::mtp_gate_allows(
            &mut self.mtp_draft_accepted,
            &mut self.mtp_draft_attempted,
            &mut self.mtp_gated_turns,
            requested_depth,
        )
    }
}
