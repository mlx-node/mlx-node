//! Stress test for the C++-side paged-attention dispatch.
//!
//! `PagedKVWrite::eval_gpu` and `PagedAttention::eval_gpu` encode their
//! kernels onto MLX's own `metal::CommandEncoder` (via
//! `crates/mlx-sys/src/mlx_paged_dispatch.cpp`) rather than the extern-C
//! shim into the mlx-paged-attn Rust crate's separate command queue. This
//! test stresses that integration:
//!
//! - Build a graph that interleaves paged ops with standard MLX ops:
//!   `q' = q + bias` (non-paged op),
//!   `(k', v') = paged_kv_write(...)` (paged in-place write),
//!   `attn = paged_attention(q', k', v', ...)` (paged read of
//!   just-written data — exercises the encoder's `set_output_array` →
//!   `set_input_array` fence), then `out = attn + attn` (non-paged op
//!   consuming `attn`).
//!
//! - Run the graph N times with IDENTICAL inputs.
//!
//! - Assert each run is byte-equal to a SYNCHRONOUS REFERENCE output
//!   computed with an explicit `eval()` between the write and the
//!   read. The eval boundary is a hard sync barrier — Metal's queues
//!   drain to completion before host code continues, so the
//!   reference's read necessarily sees the fully-written K/V. This is
//!   the known-good answer; it can't be tainted by a missing fence.
//!
//! - Assert each run DIFFERS from a NO-WRITE BASELINE computed by
//!   running the same attention graph against a zero-initialized
//!   pool with no preceding write. A broken write→read fence might
//!   deterministically schedule the read before the write completes
//!   and read all-zero K/V on every iteration — that pattern would
//!   pass a naive determinism check while being silently wrong. The
//!   non-equality assertion against the no-write baseline rejects it.
//!
//! - Cover BOTH the V1 and V2 paged_attention kernel paths. The
//!   underlying dispatcher picks V1 for `max_context_len <= 512` and
//!   V2 (partitioning + reduce pass + auxiliary-buffer allocation)
//!   for longer sequences. V2's two-kernel chain has a strictly
//!   larger fence surface than V1's single dispatch — both must
//!   honor the write→read fence and the partition-output→reduce-input
//!   fence.
//!
//! ## Why we use byte-equal-to-reference rather than `MLX_GPU_VERIFY=1`
//!
//! Searching MLX for `GPU_VERIFY` returns no hits — there is no env
//! var of that name in this MLX version. The "byte-equal to
//! synchronously-computed reference" check is a strict superset of
//! what `MLX_GPU_VERIFY` would verify: any race-induced divergence
//! (corrupted kernel output, stale read, mid-kernel write tear)
//! surfaces as differing bytes between the run and the reference.
//!
//! ## Skip behavior
//!
//! Skips on hosts without Metal (the underlying C++ dispatch checks
//! `mlx::core::metal::is_available()`).

#![cfg(target_os = "macos")]

const SEQ_LEN_V1: i32 = 8;
// Pick a context length strictly greater than the V1/V2 partition
// cutoff (`PARTITION_SIZE = 512` in
// `crates/mlx-paged-attn/src/metal/paged_attention.rs` and
// `crates/mlx-sys/src/mlx_paged_dispatch.cpp`). Using a value that
// crosses one full partition boundary (768 = 1.5 * 512) means
// `max_num_partitions = ceil(768 / 512) = 2`, so the V2 reduce pass
// has to combine two partitions — that's the minimum config that
// actually exercises the partition-output → reduce-input fence as
// well as the write → partition-attention fence.
const SEQ_LEN_V2: i32 = 768;

fn handle_rc(rc: i32, iterations: i32, seq_len: i32, label: &str) {
    match rc {
        0 => {
            // Success: every run was byte-equal to the synchronous
            // reference and differed from the no-write baseline.
        }
        -3 => {
            eprintln!("{label}: Metal not available; skipped.");
        }
        -2 => panic!(
            "Phase 2 dispatch race detected ({label}, seq_len={seq_len}, \
             iterations={iterations}): a stress run diverged from the \
             synchronous reference. The synchronous reference uses an \
             explicit eval() between write and read so its output is \
             provably correct; a stress-run divergence means MLX's \
             command-encoder dependency tracking is NOT handling the \
             paged write → paged read fence correctly."
        ),
        -4 => panic!(
            "Phase 2 dispatch race detected ({label}, seq_len={seq_len}, \
             iterations={iterations}): a stress run was byte-equal to \
             the no-write baseline. The write→read fence either failed \
             or never ran; the read saw zeros instead of the \
             just-written K/V. (Without the no-write baseline check, a \
             deterministically stale read would have passed the \
             determinism check by accident.)"
        ),
        -1 => panic!(
            "{label} (seq_len={seq_len}): internal/setup error in C++ \
             helper (see stderr)"
        ),
        other => panic!("{label} (seq_len={seq_len}): unexpected rc {other} from C++ helper"),
    }
}

/// V1 path (max_context_len = 8, no partitioning), 1000 runs.
#[test]
fn phase2_stress_mixed_graph_v1_byte_deterministic_across_1000_runs() {
    let iterations: i32 = 1000;
    let rc = unsafe { mlx_sys::mlx_paged_phase2_stress_mixed_graph_v(iterations, SEQ_LEN_V1) };
    handle_rc(
        rc,
        iterations,
        SEQ_LEN_V1,
        "phase2_stress_mixed_graph_v1_byte_deterministic_across_1000_runs",
    );
}

/// V2 path (max_context_len = 768 > PARTITION_SIZE = 512, with
/// partitioning + reduce pass), 100 runs. V2 dispatches twice as many
/// kernels per call (phase 1 partitioned attention + phase 2 reduce)
/// AND allocates auxiliary buffers (`exp_sums`, `max_logits`,
/// `tmp_out`), so each run is meaningfully more expensive than V1 —
/// dropped the iteration count proportionally to keep total wall-time
/// reasonable while still hammering the V2 fence path.
#[test]
fn phase2_stress_mixed_graph_v2_byte_deterministic_across_100_runs() {
    let iterations: i32 = 100;
    let rc = unsafe { mlx_sys::mlx_paged_phase2_stress_mixed_graph_v(iterations, SEQ_LEN_V2) };
    handle_rc(
        rc,
        iterations,
        SEQ_LEN_V2,
        "phase2_stress_mixed_graph_v2_byte_deterministic_across_100_runs",
    );
}

/// Faster smoke version: V1, 10 runs only. Runs in the standard test
/// pass and serves as a quick integration check; the 1000-run version
/// stresses the encoder fencing harder but takes longer.
#[test]
fn phase2_stress_mixed_graph_v1_byte_deterministic_smoke() {
    let iterations: i32 = 10;
    let rc = unsafe { mlx_sys::mlx_paged_phase2_stress_mixed_graph_v(iterations, SEQ_LEN_V1) };
    handle_rc(
        rc,
        iterations,
        SEQ_LEN_V1,
        "phase2_stress_mixed_graph_v1_byte_deterministic_smoke",
    );
}

/// Faster smoke version: V2, 10 runs only.
#[test]
fn phase2_stress_mixed_graph_v2_byte_deterministic_smoke() {
    let iterations: i32 = 10;
    let rc = unsafe { mlx_sys::mlx_paged_phase2_stress_mixed_graph_v(iterations, SEQ_LEN_V2) };
    handle_rc(
        rc,
        iterations,
        SEQ_LEN_V2,
        "phase2_stress_mixed_graph_v2_byte_deterministic_smoke",
    );
}
