//! A K-quant matmul traced by a shapeless `compile` must replay at other batch
//! shapes. The batch broadcast between x and a singleton-batched weight is
//! traced as `BroadcastAxes`, so the replay re-derives it from the new inputs
//! instead of reusing the shape seen at trace time.

mod kquant_support;

use kquant_support::*;
use mlx_core::array::MxArray;

struct Case {
    name: &'static str,
    w_batch: i64,
    trace_x: [i64; 3],
    x: [i64; 3],
}

const K: i64 = 512;
const N: i64 = 136;

const CASES: [Case; 4] = [
    Case {
        name: "x batch 1 -> 2, w batch 1",
        w_batch: 1,
        trace_x: [1, 1, K],
        x: [2, 1, K],
    },
    Case {
        name: "x batch 2 -> 3, w batch 1",
        w_batch: 1,
        trace_x: [2, 1, K],
        x: [3, 1, K],
    },
    Case {
        name: "x batch 1 -> 2, w batch 1, M 3 -> 40",
        w_batch: 1,
        trace_x: [1, 3, K],
        x: [2, 40, K],
    },
    Case {
        name: "x batch 1, M 3 -> 5, w batch 2",
        w_batch: 2,
        trace_x: [1, 3, K],
        x: [1, 5, K],
    },
];

fn replay(device: i32) {
    let mut n = 0;
    for (ki, kq) in KQUANTS.iter().enumerate() {
        for case in &CASES {
            let w = weights(kq, &[case.w_batch, N], K, 0x4000 + ki as u32);
            for dtype in DTYPES {
                let trace_x = activation(&case.trace_x, 0x4100 + ki as u32, dtype);
                let x = activation(&case.x, 0x4200 + ki as u32, dtype);
                let (replayed, eager) = run(&trace_x, &x, &w, kq, device);
                assert_identical(
                    &format!(
                        "{} {} {dtype:?} {}",
                        if device == GPU { "gpu" } else { "cpu" },
                        kq.mode,
                        case.name
                    ),
                    replayed,
                    eager,
                );
                n += 1;
            }
        }
    }
    println!("{n} shapeless replays bit-identical to eager");
}

fn run(
    trace_x: &MxArray,
    x: &MxArray,
    w: &Weights,
    kq: &KQuant,
    device: i32,
) -> (*mut mlx_sys::mlx_array, *mut mlx_sys::mlx_array) {
    let mode = mode_cstr(kq);
    let mut replayed = std::ptr::null_mut();
    let mut eager = std::ptr::null_mut();
    // SAFETY: every handle outlives the call; both outputs are owned on success.
    let ok = unsafe {
        mlx_sys::mlx_test_kquant_shapeless_replay(
            trace_x.as_raw_ptr(),
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            true,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
            device,
            &mut replayed,
            &mut eager,
        )
    };
    assert!(ok, "{}: shapeless replay failed", kq.mode);
    (replayed, eager)
}

#[test]
fn cpu_shapeless_replay_matches_eager() {
    replay(CPU);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_shapeless_replay_matches_eager() {
    assert!(gpu_gen() > 0, "no Metal device: this macOS test needs one");
    replay(GPU);
}
