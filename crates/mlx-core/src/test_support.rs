//! Test-only helpers for detecting host-level numeric-environment issues
//! that make certain assertions meaningless.

use std::sync::OnceLock;

use crate::array::{DType, MxArray};

/// Returns `true` when this host's half-precision GEMM produces results
/// that deviate from an f32 host reference by more than `0.1` on a small
/// `[8, 64] x [64, 64]` bf16 canary.
///
/// The vendored MLX pin's NAX steel GEMM (dispatched for every non-f32
/// matmul on gen>=17 GPUs under macOS 26.2+, see
/// `mlx/backend/metal/matmul.cpp::steel_matmul_regular_axpby`) mishandles
/// unaligned-K tiles: K < 128 produces garbage for every output element,
/// and K remainders in `[128, 256)` corrupt N-tiles past the first. The
/// wrong results are deterministic functions of the inputs (padding the
/// operands does not change them) and are NOT reproduced by the f32 path,
/// by the M=1 GEMV path, or by stock pre-NAX MLX wheels on the same
/// hardware.
///
/// Tiny-config chunked-vs-single-shot parity tests (hidden 64-128,
/// head_dim 32) compute almost entirely inside that broken regime, and
/// chunking changes which kernel class each token's math takes (a 1-token
/// tail chunk dispatches the CORRECT GEMV while the single-shot rows go
/// through the broken GEMM; per-chunk context lengths change the
/// score/out matmul shapes) — so the two paths deterministically diverge
/// O(1) without any chunk-bookkeeping bug. Parity assertions are gated on
/// this canary so they resume automatically once an MLX bump repairs the
/// kernel.
///
/// Returns `false` (trustworthy) if the canary cannot run at all (e.g. no
/// Metal device) — callers are expected to have their own
/// Metal-availability skips.
///
/// Set `MLX_TEST_FORCE_HALF_PARITY=1` to bypass the canary and force the
/// gated assertions to run anyway (for measuring the divergence on a
/// broken host, or for re-validating after an MLX pin bump before the
/// canary is removed).
pub(crate) fn half_gemm_untrustworthy() -> bool {
    static RESULT: OnceLock<bool> = OnceLock::new();
    *RESULT.get_or_init(|| {
        if std::env::var_os("MLX_TEST_FORCE_HALF_PARITY").is_some_and(|v| v == "1") {
            eprintln!(
                "half_gemm_untrustworthy: MLX_TEST_FORCE_HALF_PARITY=1 — bypassing \
                 the canary; gated parity assertions will run even if this host's \
                 half-precision GEMM is broken"
            );
            return false;
        }
        let run = || -> Result<bool, napi::Error> {
            let m = 8usize;
            let k_dim = 64usize;
            let n_dim = 64usize;
            let x_vals: Vec<f32> = (0..(m * k_dim))
                .map(|i| ((i as f32 * 0.9173 + 0.37).sin()) * 1.5)
                .collect();
            let w_vals: Vec<f32> = (0..(k_dim * n_dim))
                .map(|i| ((i as f32 * 0.5711 + 0.71).sin()) * 0.5)
                .collect();
            let x = MxArray::from_float32(&x_vals, &[m as i64, k_dim as i64])?
                .astype(DType::BFloat16)?;
            let w = MxArray::from_float32(&w_vals, &[k_dim as i64, n_dim as i64])?
                .astype(DType::BFloat16)?;

            let flat = |a: &MxArray| -> Result<Vec<f32>, napi::Error> {
                let n: i64 = a.shape()?.iter().product();
                let f = a.reshape(&[n])?.astype(DType::Float32)?;
                f.eval();
                (0..n as usize).map(|i| f.item_at_float32(i)).collect()
            };

            let y = flat(&x.matmul(&w)?)?;
            let xb = flat(&x)?;
            let wb = flat(&w)?;
            let mut max_err = 0.0f32;
            for r in 0..m {
                for n in 0..n_dim {
                    let mut acc = 0.0f32;
                    for k in 0..k_dim {
                        acc += xb[r * k_dim + k] * wb[k * n_dim + n];
                    }
                    max_err = max_err.max((y[r * n_dim + n] - acc).abs());
                }
            }
            Ok(max_err > 0.1)
        };
        match run() {
            Ok(untrustworthy) => {
                if untrustworthy {
                    eprintln!(
                        "half_gemm_untrustworthy: this host's bf16 GEMM fails the \
                         K=64/N=64 canary (vendored-MLX NAX unaligned-K bug on \
                         gen>=17 GPUs); half-precision parity assertions on tiny \
                         configs are gated off"
                    );
                }
                untrustworthy
            }
            Err(_) => false,
        }
    })
}
