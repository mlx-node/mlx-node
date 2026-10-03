//! Pin-bump gate for the bridge BF16-x / F32-sidecar affine matmul
//! (`mlx_quantized_matmul_affine_bf16`), eager and after a shapeless compile
//! replay: every case must reproduce the output digest in `tests/golden`. The
//! digests belong to the capture machine (M5 Max, applegpu_g17s); anywhere
//! else the gate fails.
//!
//! Each fixture line carries the case's route, and the per-thread family
//! counters must agree with it. Ours takes the native kernel for 2..=8 rows
//! only; everything else runs MLX's promoted F32 graph, which follows MLX's
//! own generation, NAX and 'd'-class batch limits. The two kinds of line have
//! different provenance:
//!
//! - `native`: our kernel, captured while bit-identical to the MLX fork's own
//!   mixed `QuantizedMatmul` (fork commit ce81e3b19, pin 053e43fec). An MLX
//!   pin change must leave these unchanged.
//! - `promoted`: MLX's own graph. A pin change may move these bits; re-capture
//!   them only with accuracy evidence against an f64 reference.
//!
//! Run every captured route after an MLX pin change:
//!
//! ```text
//! cargo test -p mlx-core --release --test affine_mixed_golden_gate -- --ignored --nocapture
//! MLX_METAL_GPU_ARCH=applegpu_g17d cargo test ...   # 'd' class qmv batch limits
//! MLX_METAL_GPU_ARCH=applegpu_g17g cargo test ...   # base class
//! MLX_METAL_GPU_ARCH=applegpu_g17p cargo test ...   # gen 17 without NAX
//! MLX_METAL_GPU_ARCH=applegpu_g16s cargo test ...   # gen 16: no NAX
//! MLX_METAL_GPU_ARCH=applegpu_g14s cargo test ...   # gen 14: no qmv_wide, all promoted
//! MLX_METAL_GPU_ARCH=applegpu_g14d cargo test ...   # gen 14, 'd' class
//! ```

mod affine_mixed_support;
mod golden_support;
mod kquant_support;

use affine_mixed_support::*;
use golden_support::{Golden, digest};
use kquant_support::{family_count, gpu_gen, read_output, start_counting, stop_counting};
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

/// Whether a contiguous x must take the native kernel on this GPU: gs 32,
/// 8 bits, K not 64/128 and 2 <= M <= 8. Everything else, rows 9 and up
/// included, takes the promoted graph.
fn route(arch_gen: i32, gs: i32, bits: i32, k: i64, m: i64) -> bool {
    arch_gen >= 15 && gs == 32 && bits == 8 && k != 64 && k != 128 && (2..=8).contains(&m)
}

/// A fixture line: the output digest, then the route the counters proved.
fn routed(digest: String, native: bool) -> String {
    format!("{digest} {}", if native { "native" } else { "promoted" })
}

struct Tally {
    cases: u64,
    promoted_above_8: u64,
    golden: Golden,
}

impl Tally {
    /// The golden digest, then the counter delta proves the path.
    fn check(&mut self, ctx: &str, x: &MxArray, w: &Weights, gs: i32, bits: i32, expect: bool) {
        let wide_before = family_count("affine_mixed_qmv_wide");
        let promoted_before = family_count("affine_mixed_promoted");
        let (shape, dtype, out) = read_output(ctx, ours(x, w, gs, bits));
        let wide = family_count("affine_mixed_qmv_wide") - wide_before;
        let promoted = family_count("affine_mixed_promoted") - promoted_before;
        assert_eq!(wide + promoted, 1, "{ctx}: one path per case");
        assert_eq!(
            wide == 1,
            expect,
            "{ctx}: native kernel ran = {}",
            wide == 1
        );
        self.golden
            .record_value(ctx, &routed(digest(&shape, dtype, &out), expect));
        let rows = x.size().unwrap() as i64 / x.shape_at(x.ndim().unwrap() - 1).unwrap();
        if rows > 8 {
            self.promoted_above_8 += promoted;
        }
        self.cases += 1;
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "pin-bump gate with hardware-specific goldens; see the module docs"]
fn mixed_affine_matches_golden() {
    assert!(metal_available(), "this gate needs Metal");
    let arch_gen = gpu_gen();
    start_counting();
    let mut t = Tally {
        cases: 0,
        promoted_above_8: 0,
        golden: Golden::metal("affine_mixed_matmul"),
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
        "gen {arch_gen}: {} cases; qmv_wide {got_wide}, \
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
    t.golden.finish();
}

/// A shapeless compile traced at one row count and replayed at another must
/// route by the replay's rows; replay and eager must both equal the golden.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "pin-bump gate with hardware-specific goldens; see the module docs"]
fn mixed_affine_shapeless_replay_matches_golden() {
    assert!(metal_available(), "this gate needs Metal");
    let arch_gen = gpu_gen();
    let mut g = Golden::metal("affine_mixed_shapeless");
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
            for (tag, handle) in [("replay", replayed), ("eager", eager)] {
                let what = format!("{ctx} {tag}");
                let (shape, dtype, out) = read_output(&what, handle);
                g.record_value(&what, &routed(digest(&shape, dtype, &out), wide(m) == 1));
            }
            cases += 1;
        }
    }
    stop_counting();
    eprintln!("{cases} shapeless replays");
    g.finish();
}
