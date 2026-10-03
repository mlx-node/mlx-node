//! The BF16-x / F32-sidecar affine matmul runs inside the shapeless DFlash2
//! verify compile. Traced at one row count and replayed at another, it must
//! route by the replay's rows and equal both its eager result and the promoted
//! F32 graph bit for bit.

mod affine_mixed_support;
mod kquant_support;

use affine_mixed_support::*;
use kquant_support::assert_identical;

#[cfg(target_os = "macos")]
#[test]
fn mixed_affine_shapeless_replay_matches_eager_and_promoted() {
    assert!(metal_available(), "this test needs Metal");
    let mut cases = 0;
    for (si, &(n, k)) in [(136i64, 768i64), (2048, 768), (96, 128), (512, 5120)]
        .iter()
        .enumerate()
    {
        for &(gs, bits) in &[(32, 8), (32, 4)] {
            let w = weights(n, k, gs, bits, 0xD000 + si as u32);
            for (trace_m, m) in [(3, 3), (3, 8), (8, 2), (5, 1), (2, 12), (1, 6)] {
                let trace_x = activation(&[1, trace_m, k], 0xE000 + si as u32, 1.0);
                let x = activation(&[1, m, k], 0xF000 + si as u32 * 16 + m as u32, 30.0);
                let ctx = format!("gs{gs} b{bits} N{n} K{k} M {trace_m} -> {m}");
                let (replayed, eager) = shapeless_replay(&trace_x, &x, &w, gs, bits);
                assert_identical(&format!("{ctx} replay vs eager"), replayed, eager);
                let (replayed, _) = shapeless_replay(&trace_x, &x, &w, gs, bits);
                assert_identical(
                    &format!("{ctx} replay vs promoted"),
                    replayed,
                    promoted(&x, &w, gs, bits),
                );
                cases += 1;
            }
        }
    }
    eprintln!("{cases} shapeless replays bit-identical to eager and to the promoted graph");
}
