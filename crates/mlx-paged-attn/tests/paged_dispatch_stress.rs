//! Phase 2 stress test for the C++-side paged-attention dispatch.
//!
//! After Phase 2, `PagedKVWrite::eval_gpu` and `PagedAttention::eval_gpu`
//! encode their kernels onto MLX's own `metal::CommandEncoder` (via
//! `crates/mlx-sys/src/mlx_paged_dispatch.cpp`) instead of the Phase 1
//! extern-C shim into the mlx-paged-attn Rust crate's separate command
//! queue. This test stresses that integration:
//!
//! - Build a graph that interleaves paged ops with standard MLX ops:
//!     1. q' = q + bias                    (non-paged op)
//!     2. (k', v') = paged_kv_write(...)   (paged in-place write)
//!     3. attn = paged_attention(q', k', v', ...)
//!                                         (paged read of just-written
//!                                          data — exercises the
//!                                          encoder's
//!                                          set_output_array →
//!                                          set_input_array fence)
//!     4. out = attn + attn                (non-paged op consuming attn)
//!
//! - Run the graph N=1000 times with IDENTICAL inputs.
//!
//! - Assert the output is byte-equal across every run.
//!
//! A race between paged_kv_write's writes and the subsequent paged
//! attention read (Phase 1's weakness, since the kv_write ran on a
//! different command queue MLX wasn't tracking) would surface as
//! nondeterminism: some runs would see stale K/V, others wouldn't,
//! producing different attention outputs across runs.
//!
//! ## Why we use byte-equal-determinism rather than `MLX_GPU_VERIFY=1`
//!
//! Searching MLX for `GPU_VERIFY` returns no hits — there is no env
//! var of that name in this MLX version. The byte-equal-determinism
//! check is a strict superset of what `MLX_GPU_VERIFY` would verify:
//! any race-induced divergence (corrupted kernel output, stale read,
//! mid-kernel write tear) surfaces as differing bytes between runs.
//!
//! ## Skip behavior
//!
//! Skips on hosts without Metal (the underlying C++ dispatch checks
//! `mlx::core::metal::is_available()`).

#![cfg(target_os = "macos")]

#[test]
fn phase2_stress_mixed_graph_byte_deterministic_across_1000_runs() {
    let iterations: i32 = 1000;
    let rc = unsafe { mlx_sys::mlx_paged_phase2_stress_mixed_graph(iterations) };
    match rc {
        0 => {
            // Success: all runs byte-identical.
        }
        -3 => {
            eprintln!(
                "phase2_stress_mixed_graph_byte_deterministic_across_1000_runs: \
                 Metal not available; skipped."
            );
        }
        -2 => panic!(
            "Phase 2 dispatch race detected: outputs diverged across {iterations} \
             runs of an identical-input mixed paged + non-paged graph. \
             This means MLX's command-encoder dependency tracking is NOT \
             handling the paged write → paged read fence correctly, OR the \
             paged attention output is being read before the kernel completes."
        ),
        -1 => panic!(
            "phase2_stress_mixed_graph_byte_deterministic_across_1000_runs: \
             internal/setup error in C++ helper (see stderr)"
        ),
        other => panic!(
            "phase2_stress_mixed_graph_byte_deterministic_across_1000_runs: \
             unexpected rc {other} from C++ helper"
        ),
    }
}

/// Faster smoke version of the determinism test: 10 runs only. Runs in
/// the standard test pass and serves as a quick integration check; the
/// 1000-run version above stresses the encoder fencing harder but
/// takes longer.
#[test]
fn phase2_stress_mixed_graph_byte_deterministic_smoke() {
    let iterations: i32 = 10;
    let rc = unsafe { mlx_sys::mlx_paged_phase2_stress_mixed_graph(iterations) };
    match rc {
        0 => {}
        -3 => {
            eprintln!(
                "phase2_stress_mixed_graph_byte_deterministic_smoke: \
                 Metal not available; skipped."
            );
        }
        -2 => panic!(
            "Phase 2 dispatch race detected at the smoke level (10 runs). \
             See `phase2_stress_mixed_graph_byte_deterministic_across_1000_runs` \
             for the harder version."
        ),
        -1 => panic!(
            "phase2_stress_mixed_graph_byte_deterministic_smoke: \
             internal/setup error in C++ helper (see stderr)"
        ),
        other => panic!(
            "phase2_stress_mixed_graph_byte_deterministic_smoke: \
             unexpected rc {other} from C++ helper"
        ),
    }
}
