//! Engine-owned MTP (Multi-Token Prediction) propose/verify whole-turn
//! path — the MTP analog of [`crate::engine::paged_turn`]. Families opt in
//! via [`crate::engine::backend::MtpBackend`]; their
//! `ChatBackend::mtp_turn` body becomes `Some(run_mtp_turn(self, args))`.
//!
//! SCAFFOLD STEP: the relocated `decode_loop_mtp!` outer body
//! (`run_mtp_turn`) and the relocated `run_mtp_cycle_inner` (`run_mtp_cycle`)
//! land in later steps. Today this module carries ONLY the
//! [`MtpStepper`](crate::engine::backend::MtpStepper) contract's test
//! harness — a scripted [`MockMtpStepper`] double + call-ledger unit tests
//! that PROVE the trait + GAT lifetimes + the strictly-sequential
//! `&mut self` borrow model compile and are usable. Nothing in production
//! calls this module yet, so the families' MTP behavior is byte-identical.

#[cfg(test)]
mod tests {
    //! `MtpStepper`-contract tests over a scripted mock — NO model, NO
    //! Metal. The UNIQUE value here is proving the trait's GAT lifetimes
    //! and the strictly-sequential `&mut self` borrow model: the harness
    //! drives a short scripted propose/verify/commit/rollback sequence and
    //! asserts the recorded call ledger, exactly as the
    //! `run_paged_turn` mock asserts the paged lifecycle sequence.

    use std::cell::RefCell;

    use napi::bindgen_prelude::*;

    use crate::array::MxArray;
    use crate::engine::backend::MtpStepper;
    use crate::models::qwen3_5::mtp_decode::{MtpCommitAnchor, MtpVerifyOutput};
    use crate::sampling::SamplingConfig;

    /// One recorded `MtpStepper` call, tagged so a test can assert the
    /// exact propose/verify/commit/rollback ORDER (the analog of the paged
    /// harness's `Vec<&'static str>` ledger, enum-typed so per-call payload
    /// — depths, accept counts — rides along).
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Call {
        EmbeddingWeight,
        CommittedHistoryActive,
        ProfilerRelabel,
        ForwardWithHidden,
        DraftStep,
        VerifyStep { depth: usize },
        VerifyStepArgmaxOnly { depth: usize },
        VerifyStepSparse { depth: usize },
        SnapshotMainLinear,
        Rollback { accepted: usize, depth: usize },
        RestoreAndReplayMain { accepted: usize },
        CommitMtp { anchor: MtpCommitAnchor, k: usize },
        BeginCycle { chained: bool },
        EvalStep { budget_forced: bool },
        EvalStepWithChainedHidden,
        RollbackUnemitted { unemitted: usize },
        TakeReplayError,
        IntoDesynced,
    }

    /// Tiny lazy `[1, 1]` array — fabricated WITHOUT Metal (mlx arrays are
    /// lazy, so construction never touches the GPU). The mock hands these
    /// back wherever the contract returns an [`MxArray`]; the engine never
    /// evals them in S0 (no loop yet), and the borrow-model proof needs
    /// only that the handles thread cleanly between calls.
    fn lazy_scalar(v: f32) -> MxArray {
        MxArray::from_float32(&[v], &[1, 1]).expect("lazy [1,1] array construction is infallible")
    }

    /// Scripted [`MtpStepper`] double. Records every call into an ordered
    /// ledger (interior-mutable so the `&self` `eval_step*` /
    /// `profiler_relabel` / `embedding_weight` methods can record too) and
    /// returns canned lazy arrays / values.
    ///
    /// `committed_history` toggles [`MtpStepper::committed_history_active`]
    /// (dense=true / MoE=false); `relabel` is the canned
    /// [`MtpStepper::profiler_relabel`]; `desynced` is the canned
    /// [`MtpStepper::into_desynced`] terminal value (paged MUST be false);
    /// `replay_error` lets a test script a stashed rollback error the
    /// engine would surface via [`MtpStepper::take_replay_error`].
    ///
    /// `has_argmax_only` / `has_sparse` gate the optional verify fast paths
    /// (default-`None` on every eager family today) so a test can prove
    /// both the "fast path present" and "fall back to verify_step" arms
    /// compile and dispatch.
    struct MockMtpStepper {
        ledger: RefCell<Vec<Call>>,
        emb: MxArray,
        committed_history: bool,
        relabel: Option<&'static str>,
        desynced: bool,
        replay_error: RefCell<Option<Error>>,
        has_argmax_only: bool,
        has_sparse: bool,
    }

    impl MockMtpStepper {
        fn new() -> Self {
            Self {
                ledger: RefCell::new(Vec::new()),
                emb: lazy_scalar(1.0),
                committed_history: true,
                relabel: None,
                desynced: false,
                replay_error: RefCell::new(None),
                has_argmax_only: false,
                has_sparse: false,
            }
        }

        fn record(&self, c: Call) {
            self.ledger.borrow_mut().push(c);
        }

        fn snapshot(&self) -> Vec<Call> {
            self.ledger.borrow().clone()
        }
    }

    impl MtpStepper for MockMtpStepper {
        fn embedding_weight(&self) -> &MxArray {
            self.record(Call::EmbeddingWeight);
            &self.emb
        }

        fn committed_history_active(&self) -> bool {
            self.record(Call::CommittedHistoryActive);
            self.committed_history
        }

        fn profiler_relabel(&self) -> Option<&'static str> {
            self.record(Call::ProfilerRelabel);
            self.relabel
        }

        fn forward_with_hidden(
            &mut self,
            _ids: &MxArray,
            _emb: &MxArray,
        ) -> Result<(MxArray, MxArray, bool)> {
            self.record(Call::ForwardWithHidden);
            // (logits [1,1], hidden [1,1], needs_squeeze) — eager shape.
            Ok((lazy_scalar(0.0), lazy_scalar(0.0), true))
        }

        fn draft_step(
            &mut self,
            _prev_h: &MxArray,
            _prev_emb: &MxArray,
        ) -> Result<(MxArray, MxArray)> {
            self.record(Call::DraftStep);
            Ok((lazy_scalar(0.0), lazy_scalar(0.0)))
        }

        fn verify_step(
            &mut self,
            _ids: &MxArray,
            _emb: &MxArray,
            depth: usize,
        ) -> Result<MtpVerifyOutput> {
            self.record(Call::VerifyStep { depth });
            Ok(MtpVerifyOutput::logits_only(
                lazy_scalar(0.0),
                lazy_scalar(0.0),
            ))
        }

        fn verify_step_argmax_only(
            &mut self,
            _ids: &MxArray,
            _emb: &MxArray,
            depth: usize,
        ) -> Option<Result<MtpVerifyOutput>> {
            self.record(Call::VerifyStepArgmaxOnly { depth });
            if self.has_argmax_only {
                Some(Ok(MtpVerifyOutput::logits_only(
                    lazy_scalar(0.0),
                    lazy_scalar(0.0),
                )))
            } else {
                None
            }
        }

        fn verify_step_sparse(
            &mut self,
            _ids: &MxArray,
            _emb: &MxArray,
            depth: usize,
            _cfg: &SamplingConfig,
        ) -> Option<Result<MtpVerifyOutput>> {
            self.record(Call::VerifyStepSparse { depth });
            if self.has_sparse {
                Some(Ok(MtpVerifyOutput::logits_only(
                    lazy_scalar(0.0),
                    lazy_scalar(0.0),
                )))
            } else {
                None
            }
        }

        fn snapshot_main_linear(&mut self) {
            self.record(Call::SnapshotMainLinear);
        }

        fn rollback(&mut self, accepted_drafts: usize, depth: usize) {
            self.record(Call::Rollback {
                accepted: accepted_drafts,
                depth,
            });
        }

        fn restore_and_replay_main(&mut self, accepted: &[u32], _emb: &MxArray) -> Result<()> {
            self.record(Call::RestoreAndReplayMain {
                accepted: accepted.len(),
            });
            Ok(())
        }

        fn commit_mtp(
            &mut self,
            anchor: MtpCommitAnchor,
            _seed_h: &MxArray,
            _verify_hiddens: &MxArray,
            committed_ids: &[u32],
            k_accepted: usize,
            _emb: &MxArray,
        ) -> Result<()> {
            self.record(Call::CommitMtp {
                anchor,
                k: k_accepted,
            });
            // The K+2 committed-sequence shape the real commit consumes.
            assert_eq!(
                committed_ids.len(),
                k_accepted + 2,
                "committed_ids is [last_committed, d_0..d_{{K-1}}, boundary] (K+2)"
            );
            Ok(())
        }

        fn begin_cycle(&mut self, chained_anchor: bool) {
            self.record(Call::BeginCycle {
                chained: chained_anchor,
            });
        }

        fn eval_step(&self, _token: &MxArray, _logits: &MxArray, budget_forced: bool) {
            self.record(Call::EvalStep { budget_forced });
        }

        fn eval_step_with_chained_hidden(&self, _token: &MxArray, _chained_h: &MxArray) {
            self.record(Call::EvalStepWithChainedHidden);
        }

        fn rollback_unemitted(&mut self, unemitted: usize) {
            self.record(Call::RollbackUnemitted { unemitted });
        }

        fn take_replay_error(&mut self) -> Option<Error> {
            self.record(Call::TakeReplayError);
            self.replay_error.borrow_mut().take()
        }

        fn into_desynced(self) -> bool {
            self.record(Call::IntoDesynced);
            self.desynced
        }
    }

    /// Drive a short scripted propose/verify/commit/rollback sequence
    /// through the trait, EXACTLY in the order `run_mtp_cycle_inner` calls
    /// `ops.*` today (Step A forward → begin_cycle → D draft steps →
    /// snapshot → verify → commit → rollback → restore/replay on reject →
    /// eval), then the iteration-boundary fused chained eval. Proves the
    /// strictly-sequential `&mut self` borrow model + GAT-free dyn-less
    /// dispatch compile and run — no Metal, no model.
    fn drive_one_reject_cycle(step: &mut MockMtpStepper, depth: usize, accepted: usize) {
        // Turn-entry reads (the engine pulls these once before the loop).
        let _relabel = step.profiler_relabel();
        let _committed = step.committed_history_active();

        // Step A: main-path forward → seed hidden/emb. `emb` is read
        // through `&self` then re-borrowed into the `&mut self` forward —
        // the clone breaks the borrow overlap the real loop also avoids.
        let emb = step.embedding_weight().clone();
        let (_logits, _hidden, _sq) = step
            .forward_with_hidden(&lazy_scalar(0.0), &emb)
            .expect("mock forward never fails");

        // Re-anchor, then D draft steps threading (h_next, emb) forward.
        step.begin_cycle(false);
        let mut prev_h = lazy_scalar(0.0);
        let mut prev_emb = emb.clone();
        for _ in 0..depth {
            let (h_next, _draft_logits) = step
                .draft_step(&prev_h, &prev_emb)
                .expect("mock draft never fails");
            prev_h = h_next;
            prev_emb = lazy_scalar(0.0);
        }

        // Snapshot → verify (fast-path probe falls through to verify_step).
        step.snapshot_main_linear();
        let _argmax = step.verify_step_argmax_only(&lazy_scalar(0.0), &emb, depth);
        let _verify = step
            .verify_step(&lazy_scalar(0.0), &emb, depth)
            .expect("mock verify never fails");

        // Commit the K+2 committed sequence, then rollback + replay on the
        // partial-accept (reject) arm.
        let committed_ids: Vec<u32> = std::iter::repeat_n(7u32, accepted + 2).collect();
        step.commit_mtp(
            MtpCommitAnchor::IncludeAnchor,
            &lazy_scalar(0.0),
            &lazy_scalar(0.0),
            &committed_ids,
            accepted,
            &emb,
        )
        .expect("mock commit never fails");
        step.rollback(accepted, depth);
        if accepted < depth {
            let accepted_ids: Vec<u32> = std::iter::repeat_n(7u32, accepted).collect();
            step.restore_and_replay_main(&accepted_ids, &emb)
                .expect("mock replay never fails");
        }
        // The engine surfaces any stashed replay error after the
        // (infallible) rollback.
        let _stashed = step.take_replay_error();

        // Per-token eval + the iteration-boundary fused chained eval.
        step.eval_step(&lazy_scalar(0.0), &lazy_scalar(0.0), false);
        step.eval_step_with_chained_hidden(&lazy_scalar(0.0), &lazy_scalar(0.0));
    }

    #[test]
    fn mtp_stepper_reject_cycle_call_sequence() {
        let mut step = MockMtpStepper::new();
        drive_one_reject_cycle(&mut step, 3, 1);
        let desynced = step.snapshot();
        // `into_desynced` consumes `self`; capture the terminal value
        // separately after snapshotting the ledger.
        let mut step2 = MockMtpStepper::new();
        drive_one_reject_cycle(&mut step2, 3, 1);
        let terminal_desynced = step2.into_desynced();

        assert_eq!(
            desynced,
            vec![
                Call::ProfilerRelabel,
                Call::CommittedHistoryActive,
                Call::EmbeddingWeight,
                Call::ForwardWithHidden,
                Call::BeginCycle { chained: false },
                Call::DraftStep,
                Call::DraftStep,
                Call::DraftStep,
                Call::SnapshotMainLinear,
                Call::VerifyStepArgmaxOnly { depth: 3 },
                Call::VerifyStep { depth: 3 },
                Call::CommitMtp {
                    anchor: MtpCommitAnchor::IncludeAnchor,
                    k: 1,
                },
                Call::Rollback {
                    accepted: 1,
                    depth: 3,
                },
                Call::RestoreAndReplayMain { accepted: 1 },
                Call::TakeReplayError,
                Call::EvalStep {
                    budget_forced: false,
                },
                Call::EvalStepWithChainedHidden,
            ],
            "the engine must drive the MTP propose/verify/commit/rollback \
             sequence in the order run_mtp_cycle_inner calls ops.* today"
        );
        // Paged MUST report not-desynced; the mock default mirrors that.
        assert!(
            !terminal_desynced,
            "into_desynced default (paged contract) is false"
        );
    }

    #[test]
    fn mtp_stepper_full_accept_skips_restore_and_replay() {
        // K == depth (full accept) → the engine SKIPS restore_and_replay_main
        // (verify already advanced the linear state through all D drafts).
        let mut step = MockMtpStepper::new();
        drive_one_reject_cycle(&mut step, 2, 2);
        let seq = step.snapshot();
        assert!(
            !seq.contains(&Call::RestoreAndReplayMain { accepted: 2 }),
            "full accept must NOT replay"
        );
        assert!(
            seq.contains(&Call::Rollback {
                accepted: 2,
                depth: 2,
            }),
            "full accept still calls rollback(accepted=depth, depth) for GDN normalization"
        );
    }

    #[test]
    fn mtp_stepper_verify_fast_paths_dispatch() {
        // argmax-only fast path present → returns Some, engine uses it.
        let mut argmax = MockMtpStepper::new();
        argmax.has_argmax_only = true;
        let r = argmax.verify_step_argmax_only(&lazy_scalar(0.0), &lazy_scalar(0.0), 4);
        assert!(r.is_some(), "argmax-only present must return Some");
        assert!(r.expect("present").is_ok());

        // sparse fast path present → Some; absent default → None (fall back
        // to verify_step, the eager-family shape).
        let mut sparse = MockMtpStepper::new();
        sparse.has_sparse = true;
        let cfg = SamplingConfig::default();
        let s = sparse.verify_step_sparse(&lazy_scalar(0.0), &lazy_scalar(0.0), 4, &cfg);
        assert!(s.is_some(), "sparse present must return Some");

        let mut none = MockMtpStepper::new();
        assert!(
            none.verify_step_argmax_only(&lazy_scalar(0.0), &lazy_scalar(0.0), 4)
                .is_none(),
            "absent argmax-only default is None"
        );
        assert!(
            none.verify_step_sparse(&lazy_scalar(0.0), &lazy_scalar(0.0), 4, &cfg)
                .is_none(),
            "absent sparse default is None"
        );
    }

    #[test]
    fn mtp_stepper_surfaces_stashed_replay_error() {
        // A stashed rollback-replay error is surfaced by take_replay_error
        // (the engine then `?`-propagates it AFTER the infallible rollback).
        let mut step = MockMtpStepper::new();
        *step.replay_error.borrow_mut() = Some(Error::from_reason("scripted replay failure"));
        step.rollback(1, 3);
        let err = step.take_replay_error();
        assert!(err.is_some(), "stashed replay error must surface");
        assert_eq!(err.expect("present").reason, "scripted replay failure");
        // Drained: a second take yields None.
        assert!(
            step.take_replay_error().is_none(),
            "take_replay_error drains the stash"
        );
    }

    #[test]
    fn mtp_stepper_into_desynced_paged_vs_flat() {
        // Flat/MoE may set the desync flag on a mid-cycle stop; paged MUST
        // return false. Both arms compile through the consuming `self`
        // signature.
        let flat = {
            let mut s = MockMtpStepper::new();
            s.desynced = true;
            s.rollback_unemitted(2); // mid-cycle stop left 2 unemitted
            s
        };
        assert!(flat.into_desynced(), "flat mid-cycle stop reports desynced");

        let paged = {
            let mut s = MockMtpStepper::new();
            s.desynced = false; // paged truncates the adapter, never flat-desyncs
            s.rollback_unemitted(2);
            s
        };
        assert!(
            !paged.into_desynced(),
            "paged into_desynced contract is false"
        );
    }
}
