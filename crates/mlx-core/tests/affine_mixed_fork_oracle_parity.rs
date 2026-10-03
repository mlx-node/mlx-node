//! TEST-ONLY oracle, deleted with the move to upstream MLX: the bridge
//! BF16-x / F32-sidecar affine matmul (`mlx_quantized_matmul_affine_bf16`)
//! must equal the MLX fork's own mixed `QuantizedMatmul` (fork commit
//! ce81e3b19, pin 053e43fec) bit for bit, eagerly and after a shapeless
//! compile replay.
//!
//! The fork falls back to the promoted F32 graph for shapes its kernel does
//! not take, and so does ours, so bit-identity alone cannot show the native
//! kernel ran: the per-thread family counters must also match the expected
//! routing. Ours takes the native kernel for 2..=8 rows only; from 9 rows the
//! fork still ran its kernel (up to its batch limit), which equals the
//! promoted graph bit for bit, so the bits must match either way.
//!
//! ```text
//! cargo test -p mlx-core --release --test affine_mixed_fork_oracle_parity -- --nocapture
//! MLX_METAL_GPU_ARCH=applegpu_g14s cargo test ...   # gen 14: no qmv_wide, all promoted
//! ```

mod affine_mixed_support;
mod kquant_support;

use affine_mixed_support::*;
use kquant_support::{assert_identical, family_count, gpu_gen, start_counting, stop_counting};
use mlx_core::array::MxArray;

/// Group sizes and bits with F32 sidecars at load that the entry point accepts:
/// GGUF Q8_0 (32/8), Q4_0 and Q4_1 (32/4), and MLX affine checkpoints whose
/// F16 sidecars are promoted under BF16 compute. Q5_1 (5 bits) is rejected and
/// takes the caller's promoted path.
const LAYOUTS: [(i32, i32); 5] = [(32, 8), (32, 4), (64, 4), (64, 8), (128, 4)];

/// N below and at/above the 2048 tile-cap switch; K of 64 and 128 route the
/// F32 path to qmv_quad; N=1000 leaves a partial row tile.
const SHAPES: [(i64, i64); 7] = [
    (136, 768),
    (1000, 1024),
    (2048, 768),
    (2560, 1024),
    (96, 64),
    (96, 128),
    (512, 5120),
];

const ROWS: [i64; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17, 33];

fn fork(x: &mlx_core::array::MxArray, w: &Weights, gs: i32, bits: i32) -> *mut mlx_sys::mlx_array {
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_test_fork_affine_mixed_qmm(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            gs,
            bits,
        )
    }
}

/// Whether a contiguous x must take the native kernel on this GPU: gs 32,
/// 8 bits, K not 64/128 and 2 <= M <= 8. Everything else, rows 9 and up
/// included, takes the promoted graph.
fn route(arch_gen: i32, gs: i32, bits: i32, k: i64, m: i64) -> bool {
    arch_gen >= 15 && gs == 32 && bits == 8 && k != 64 && k != 128 && (2..=8).contains(&m)
}

struct Tally {
    cases: u64,
    promoted_above_8: u64,
}

impl Tally {
    /// Bit-identity with the fork, then the counter delta proves the path.
    fn check(&mut self, ctx: &str, x: &MxArray, w: &Weights, gs: i32, bits: i32, expect: bool) {
        let wide_before = family_count("affine_mixed_qmv_wide");
        let promoted_before = family_count("affine_mixed_promoted");
        assert_identical(ctx, ours(x, w, gs, bits), fork(x, w, gs, bits));
        let wide = family_count("affine_mixed_qmv_wide") - wide_before;
        let promoted = family_count("affine_mixed_promoted") - promoted_before;
        assert_eq!(wide + promoted, 1, "{ctx}: one path per case");
        assert_eq!(
            wide == 1,
            expect,
            "{ctx}: native kernel ran = {}",
            wide == 1
        );
        let rows = x.size().unwrap() as i64 / x.shape_at(x.ndim().unwrap() - 1).unwrap();
        if rows > 8 {
            self.promoted_above_8 += promoted;
        }
        self.cases += 1;
    }
}

#[cfg(target_os = "macos")]
#[test]
fn mixed_affine_matches_fork_bitwise() {
    assert!(metal_available(), "this oracle needs Metal");
    let arch_gen = gpu_gen();
    start_counting();
    let mut t = Tally {
        cases: 0,
        promoted_above_8: 0,
    };
    for (li, &(gs, bits)) in LAYOUTS.iter().enumerate() {
        for (si, &(n, k)) in SHAPES.iter().enumerate() {
            if k % gs as i64 != 0 {
                continue;
            }
            let seed = (li * 31 + si) as u32;
            let w = weights(n, k, gs, bits, 0x5000 + seed);
            let at = |m| route(arch_gen, gs, bits, k, m);
            for (ri, &m) in ROWS.iter().enumerate() {
                for amp in AMPLITUDES {
                    let x = activation(&[m, k], 0x6000 + seed * 64 + ri as u32, amp);
                    t.check(
                        &format!("gs{gs} b{bits} N{n} K{k} M{m} amp{amp}"),
                        &x,
                        &w,
                        gs,
                        bits,
                        at(m),
                    );
                }
                if (2..=8).contains(&m) {
                    let x3 = activation(&[1, m, k], 0x7000 + seed * 64 + ri as u32, 1.0);
                    t.check(
                        &format!("gs{gs} b{bits} N{n} K{k} [1,{m},K]"),
                        &x3,
                        &w,
                        gs,
                        bits,
                        at(m),
                    );
                }
            }
            let xt = activation(&[k, 6], 0x8000 + seed, 1.0)
                .transpose(Some(&[1, 0]))
                .unwrap();
            t.check(
                &format!("gs{gs} b{bits} N{n} K{k} transposed x"),
                &xt,
                &w,
                gs,
                bits,
                false,
            );
            let xb = activation(&[2, 4, k], 0x9000 + seed, 1.0);
            t.check(
                &format!("gs{gs} b{bits} N{n} K{k} [2,4,K]"),
                &xb,
                &w,
                gs,
                bits,
                at(8),
            );
        }
    }
    let got_wide = family_count("affine_mixed_qmv_wide");
    let got_promoted = family_count("affine_mixed_promoted");
    let per_nv: Vec<u64> = (2..=8)
        .map(|nv| family_count(&format!("affine_mixed_qmv_wide_nv{nv}")))
        .collect();
    stop_counting();
    eprintln!(
        "gen {arch_gen}: {} cases bit-identical to the fork; qmv_wide {got_wide}, \
         promoted {got_promoted} ({} with M > 8), qmv_wide nv2..8 {per_nv:?}",
        t.cases, t.promoted_above_8
    );
    assert_eq!(
        got_wide + got_promoted,
        t.cases,
        "every case takes exactly one path"
    );
    if arch_gen >= 15 {
        assert!(
            per_nv.iter().all(|&c| c > 0),
            "every vecs_per_tg instantiation must run"
        );
    } else {
        assert_eq!(got_wide, 0, "no qmv_wide below gen 15");
    }
}

/// A shapeless compile traced at one row count and replayed at another must
/// route by the replay's rows: same bits as the fork's eager result.
#[cfg(target_os = "macos")]
#[test]
fn mixed_affine_shapeless_replay_matches_fork_bitwise() {
    assert!(metal_available(), "this oracle needs Metal");
    let arch_gen = gpu_gen();
    start_counting();
    let mut cases = 0;
    for (si, &(n, k)) in SHAPES.iter().enumerate() {
        let w = weights(n, k, 32, 8, 0xA000 + si as u32);
        for (trace_m, m) in [
            (3, 3),
            (3, 8),
            (8, 2),
            (5, 1),
            (2, 6),
            (1, 6),
            (2, 12),
            (12, 5),
        ] {
            let trace_x = activation(&[1, trace_m, k], 0xB000 + si as u32, 1.0);
            let x = activation(&[1, m, k], 0xC000 + si as u32 * 16 + m as u32, 1.0);
            let ctx = format!("N{n} K{k} M {trace_m} -> {m}");
            let before = family_count("affine_mixed_qmv_wide");
            let (replayed, eager) = shapeless_replay(&trace_x, &x, &w, 32, 8);
            // trace eval at trace_m, then the replay and the eager run at m.
            let wide = |m| u64::from(route(arch_gen, 32, 8, k, m));
            assert_eq!(
                family_count("affine_mixed_qmv_wide") - before,
                wide(trace_m) + 2 * wide(m),
                "{ctx}: the replay must route by its own rows"
            );
            assert_identical(&format!("{ctx} replay"), replayed, fork(&x, &w, 32, 8));
            assert_identical(&format!("{ctx} eager"), eager, fork(&x, &w, 32, 8));
            cases += 1;
        }
    }
    stop_counting();
    eprintln!("{cases} shapeless replays bit-identical to the fork");
}
