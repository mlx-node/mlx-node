//! Synthetic end-to-end gate for the eager MoE MTP path in its DEFAULT
//! (flag-off, cycle-history v1) state.
//!
//! Pins the flag-OFF path: `MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY` is
//! defensively REMOVED at startup — the stepper reads it once per process,
//! and an inherited environment (e.g. a shell that exported the flag for a
//! committed-history run) must not flip the v1 baseline ON. Its sibling
//! `qwen3_5_moe_mtp_synthetic_v2.rs` pins the flag-on
//! path. See `tests/common/mod.rs` for the shared harness and the rationale
//! (all local real MoE-MTP checkpoints are VLM → forced block-paged → the
//! eager stepper is unreachable, so a tiny random checkpoint is the only
//! always-on route through it).
//!
//! Asserts the four deterministic conditions documented in the harness
//! header: MTP head engages after reload, AR baseline and MTP decode both
//! complete the full token budget crash-free with `mtp_cycles > 1` and
//! populated acceptance metrics, and a repeat MTP decode is byte-identical
//! to the first. MTP==AR byte-identity is NOT asserted on random weights —
//! measured ~10-15% of fresh checkpoint draws flip a late token via greedy
//! argmax near-ties, at the SAME rate in this flag-off (pre-existing v1)
//! binary as in the v2 sibling, so the flakiness is kernel rounding, not the
//! committed-history port. That byte-identity gate lives in the real-weights
//! deep test `qwen3_5_moe_mtp_committed_history.rs`.

mod common;

#[test]
fn synthetic_moe_mtp_gate_v1_cycle_history() {
    // Defensively strip the committed-history flag before building the
    // runtime: this binary pins the DEFAULT (flag-off) path and the stepper
    // reads the flag once per process, so an inherited environment must not
    // flip it ON for the v1 baseline.
    // SAFETY: `remove_var` requires no concurrent access to the process
    // environment. This is the binary's only `#[test]`, so libtest runs just
    // this one test thread; the tokio workers and model threads — the only
    // in-process readers of this flag — are spawned below, after the remove.
    unsafe { std::env::remove_var("MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY") };

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(common::run_synthetic_mtp_gate(
            "v1 cycle-history, flag off/default",
        ));
}
