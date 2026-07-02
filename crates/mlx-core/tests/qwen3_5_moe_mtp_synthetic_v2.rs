//! Synthetic end-to-end gate for the eager MoE MTP path with
//! COMMITTED-HISTORY v2 forced ON (`MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY=1`).
//!
//! This exercises the persistent-drafter-cache mechanics every cycle —
//! `commit_mtp`'s multi-token drafter forward and `begin_cycle`'s trim back
//! to the committed cursor. Separate binary from the v1 sibling because the
//! stepper reads the flag once per process (`OnceLock`); the env var is set
//! before the tokio runtime (and thus any model work) starts. See
//! `tests/common/mod.rs` for the shared harness.
//!
//! Asserts the four deterministic conditions documented in the harness
//! header: MTP head engages after reload, AR baseline and MTP decode both
//! complete the full token budget crash-free with `mtp_cycles > 1` and
//! populated acceptance metrics, and a repeat MTP decode is byte-identical
//! to the first — every cycle runs the v2 trim/commit path, so a shape or
//! cache-length bug in the port fails here loudly. MTP==AR byte-identity is
//! NOT asserted on random weights — measured ~10-15% of fresh checkpoint
//! draws flip a late token via greedy argmax near-ties, at the SAME rate as
//! the flag-off v1 sibling (pre-existing code), so the flakiness is kernel
//! rounding, not the committed-history port. That byte-identity gate lives
//! in the real-weights deep test `qwen3_5_moe_mtp_committed_history.rs`.

mod common;

#[test]
fn synthetic_moe_mtp_gate_v2_committed_history() {
    // Set the flag before building the runtime, and before any model code can
    // hit the stepper's once-per-process flag read.
    // SAFETY: `set_var` requires no concurrent access to the process
    // environment. This is the binary's only `#[test]`, so libtest runs just
    // this one test thread; the tokio workers and model threads — the only
    // in-process readers of this flag — are spawned below, after the set.
    unsafe { std::env::set_var("MLX_QWEN35_MOE_MTP_COMMITTED_HISTORY", "1") };

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(common::run_synthetic_mtp_gate(
            "v2 committed-history, flag on",
        ));
}
