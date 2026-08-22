//! Synthetic end-to-end gate for the eager MoE MTP path on the FLAT lane.
//!
//! The tiny random checkpoint this drives sets `use_block_paged_cache: false`
//! (its 16-wide attention head is not valid PagedAttention geometry), so it is
//! the always-on route through `MoeMtpStepper`'s flat mode. The PAGED mode's
//! gates need a real checkpoint and live in
//! `qwen3_5_moe_paged_mtp_parity.rs` / `qwen3_5_moe_paged_mtp_midcycle.rs`.
//!
//! Asserts the four deterministic conditions documented in the harness header:
//! the MTP head engages after reload, the AR baseline and the MTP decode both
//! complete the full token budget crash-free with `mtp_cycles > 1` and
//! populated acceptance metrics, and a repeat MTP decode is byte-identical to
//! the first. MTP==AR byte-identity is NOT asserted on random weights —
//! measured ~10-15% of fresh checkpoint draws flip a late token via greedy
//! argmax near-ties, so the assert would measure kernel rounding rather than
//! stepper correctness. That byte-identity gate lives in the real-weights
//! parity test.

mod common;

#[test]
fn synthetic_moe_mtp_gate() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(common::run_synthetic_mtp_gate("flat lane"));
}
