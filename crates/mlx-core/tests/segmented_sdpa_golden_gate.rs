//! Pin-bump gate for the bridge's segmented SDPA kernels
//! (`metal/segmented_sdpa/sdpa_segmented.metal`, prebuilt) and planners
//! (`mlx_segmented_sdpa_plan.h`): every case must reproduce the digest in
//! `tests/golden`, captured while the same cases were bit-identical to the
//! MLX fork's segmented kernels and planners (fork commits 03914b9b3 and
//! a6189690e, pin 053e43fec). Kernel digests belong to the capture machine
//! (M5 Max, applegpu_g17s); anywhere else the gate fails. The planner digests
//! are pure and run by default on any machine.
//!
//! The thread's kernel-family counters prove every kernel and verify route
//! ran. Run the native class and the other policy classes after an MLX pin
//! change:
//!
//! ```text
//! cargo test -p mlx-core --release --test segmented_sdpa_golden_gate -- --include-ignored --nocapture
//! MLX_METAL_GPU_ARCH=applegpu_g17d cargo test ...   # class 'd' policy
//! MLX_METAL_GPU_ARCH=applegpu_g17g cargo test ...   # base-class policy
//! ```

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

mod golden_support;
mod kquant_support;

use golden_support::Golden;
use kquant_support::{family_count, read_output, start_counting, stop_counting};
use mlx_core::array::MxArray;

const D: i64 = 256;
const SCALE: f32 = 0.0625;

const KERNELS: [&str; 4] = [
    "segmented_sdpa_one_pass",
    "segmented_sdpa_2pass_1",
    "segmented_sdpa_verify_2pass_1",
    "segmented_sdpa_2pass_2",
];
const ROUTES: [&str; 4] = [
    "segmented_sdpa_route_single",
    "segmented_sdpa_route_one_pass",
    "segmented_sdpa_route_unified",
    "segmented_sdpa_route_split",
];

fn require_metal() {
    // SAFETY: nullary predicate.
    let available = unsafe { mlx_sys::mlx_metal_is_available() };
    assert!(available, "this gate needs a Metal device");
}

fn bf16_bits(len: usize, salt: u32) -> Vec<u16> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(salt);
            let sign = ((x >> 23) as u16 & 1) << 15;
            sign | 0x3f00 | ((x >> 16) as u16 & 0x7f)
        })
        .collect()
}

fn bf16(shape: &[i64], salt: u32) -> MxArray {
    let len = shape.iter().product::<i64>() as usize;
    MxArray::from_bfloat16(&bf16_bits(len, salt), shape).expect("bf16 array")
}

fn device_class() -> u8 {
    let mut class: std::ffi::c_char = 0;
    unsafe { mlx_sys::mlx_segmented_sdpa_test_device_verify_route(1, 1, 1, 1, &mut class) };
    class as u8
}

fn max_query_length(gqa: i64) -> i64 {
    i64::from(unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(gqa as i32) })
}

fn counts(names: &[&str]) -> Vec<u64> {
    names.iter().map(|n| family_count(n)).collect()
}

/// Rows of the leading chunk, as `segmented_verify_head_rows`.
fn head_rows(rows: i64, max_q: i64) -> Option<i64> {
    if max_q < 1 || !(1..=8).contains(&rows) {
        return None;
    }
    if rows <= max_q {
        return Some(0);
    }
    let head = rows - max_q.min(rows - 1);
    (head <= max_q).then_some(head)
}

struct Tally {
    cases: usize,
    kernels: [u64; 4],
    routes: [u64; 4],
    golden: Golden,
}

impl Tally {
    fn check(&mut self, ctx: &str, q: &MxArray, segments: [&MxArray; 4], causal: bool) {
        let [pk, pv, nk, nv] = segments;
        let ours = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_forward(
                q.as_raw_ptr(),
                pk.as_raw_ptr(),
                pv.as_raw_ptr(),
                nk.as_raw_ptr(),
                nv.as_raw_ptr(),
                SCALE,
                causal,
            )
        };
        let kernels_before = counts(&KERNELS);
        let routes_before = counts(&ROUTES);
        let (ours_shape, ours_dtype, ours) = read_output(ctx, ours);
        let kernels: Vec<u64> = counts(&KERNELS)
            .iter()
            .zip(&kernels_before)
            .map(|(a, b)| a - b)
            .collect();
        let routes: Vec<u64> = counts(&ROUTES)
            .iter()
            .zip(&routes_before)
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(routes.iter().sum::<u64>(), 1, "{ctx}: one route per call");
        assert!(
            kernels[0] + kernels[1] + kernels[2] >= 1,
            "{ctx}: no segmented kernel ran"
        );
        self.golden.record_bits(ctx, &ours_shape, ours_dtype, &ours);
        for (total, delta) in self.kernels.iter_mut().zip(&kernels) {
            *total += delta;
        }
        for (total, delta) in self.routes.iter_mut().zip(&routes) {
            *total += delta;
        }
        self.cases += 1;
    }
}

/// Pure planners and vector-SDPA policy of every device class.
#[test]
fn planners_and_policy_match_golden() {
    for class in *b"sdg" {
        let mut digests = [0u64; 6];
        // SAFETY: `digests` holds the 6 outputs.
        let inputs = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_plan_digests(
                class as std::ffi::c_char,
                0,
                digests.as_mut_ptr(),
            )
        };
        let mut g = Golden::pure(&format!("segmented_sdpa_plan.class_{}", class as char));
        g.record_value("inputs", &inputs.to_string());
        for (name, d) in PLAN_DIGESTS.iter().zip(digests) {
            g.record_value(name, &format!("{d:016x}"));
        }
        g.finish();
    }
}

const PLAN_DIGESTS: [&str; 6] = [
    "sdpa_vector_uses_two_pass",
    "sdpa_vector_partition_count",
    "segmented_verify_head_rows",
    "plan_segmented_sdpa_launch",
    "plan_segmented_verify_launch",
    "select_segmented_verify_route",
];

#[cfg(target_os = "macos")]
#[test]
#[ignore = "pin-bump gate with hardware-specific goldens; see the module docs"]
fn max_query_length_matches_golden() {
    require_metal();
    let mut g = Golden::metal("segmented_sdpa_max_query_length");
    let mut widths = Vec::new();
    for gqa in 0..=33 {
        let width = max_query_length(gqa);
        g.record_value(&format!("gqa {gqa}"), &width.to_string());
        widths.push(width);
    }
    eprintln!("max query length by gqa 0..=33: {widths:?}");
    assert!(widths[1..=32].iter().all(|&w| w >= 1));
    g.finish();
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "pin-bump gate with hardware-specific goldens; see the module docs"]
fn kernels_match_golden() {
    require_metal();
    const KV: i64 = 4;
    const CAPACITY: i64 = 65_600;
    // A cache slice, as the compiled verify reads it: row stride D, head
    // stride CAPACITY * D. Fewer KV heads slice the head axis too.
    let cache_k = bf16(&[1, KV, CAPACITY, D], 0x1234_5678);
    let cache_v = bf16(&[1, KV, CAPACITY, D], 0x8765_4321);
    cache_k.eval();
    cache_v.eval();
    start_counting();
    let mut tally = Tally {
        cases: 0,
        kernels: [0; 4],
        routes: [0; 4],
        golden: Golden::metal("segmented_sdpa_kernels"),
    };
    // (q heads, kv heads): GQA 1..32 including the Qwen3.5/3.8 head layouts.
    let layouts: [(i64, i64); 9] = [
        (1, 1),
        (2, 1),
        (6, 1),
        (8, 2),
        (24, 4),
        (16, 2),
        (32, 2),
        (12, 4),
        (32, 1),
    ];
    // Totals on both sides of every vector-SDPA policy boundary of every
    // device class (two-pass at 1024 / 4096, partition steps up to 65536).
    let boundaries = [
        1024_i64, 1025, 4096, 8192, 8193, 16384, 32768, 32769, 65536, 65537,
    ];
    let mut index = 0u32;
    for (q_heads, kv_heads) in layouts {
        let gqa = q_heads / kv_heads;
        let max_q = max_query_length(gqa);
        let pk_all = cache_k
            .slice(&[0, 0, 0, 0], &[1, kv_heads, CAPACITY, D])
            .unwrap();
        let pv_all = cache_v
            .slice(&[0, 0, 0, 0], &[1, kv_heads, CAPACITY, D])
            .unwrap();
        for rows in 1_i64..=8 {
            let Some(head) = head_rows(rows, max_q) else {
                continue;
            };
            let mut prefixes = vec![1_i64, 31, 32, 33, 87, 1000];
            for b in boundaries {
                prefixes.extend([b - rows - 1, b - rows]);
            }
            for prefix in prefixes {
                index += 1;
                let sharp = index.is_multiple_of(2);
                let mut q_bits = bf16_bits((q_heads * rows * D) as usize, 0x1357_9bdf ^ index);
                if sharp {
                    // x8: exact in BF16, sharper softmax.
                    q_bits.iter_mut().for_each(|b| *b += 0x0180);
                }
                let q = MxArray::from_bfloat16(&q_bits, &[1, q_heads, rows, D]).unwrap();
                let pk = pk_all
                    .slice(&[0, 0, 0, 0], &[1, kv_heads, prefix, D])
                    .unwrap();
                let pv = pv_all
                    .slice(&[0, 0, 0, 0], &[1, kv_heads, prefix, D])
                    .unwrap();
                let mut variants = vec![(rows, rows > 1)];
                if head == 0 {
                    if rows > 1 {
                        variants.push((rows, false));
                    }
                    variants.push((rows + 3, true));
                }
                for (new_n, causal) in variants {
                    let nk = bf16(&[1, kv_heads, new_n, D], 0x2468_ace0 ^ index);
                    let nv = bf16(&[1, kv_heads, new_n, D], 0xfdb9_7531 ^ index);
                    let ctx = format!(
                        "#{index} gqa {gqa} ({q_heads}/{kv_heads}) rows {rows} prefix {prefix} \
                         new {new_n} causal {causal} sharp {sharp}"
                    );
                    tally.check(&ctx, &q, [&pk, &pv, &nk, &nv], causal);
                }
            }
        }
    }

    // Production layout: projections leave [B, T, H, D] and are transposed
    // into [B, H, T, D] views; batch 2; the prefix from a [B, P, H, D] cache
    // has row stride H * D.
    for prefix in [87_i64, 1017, 4096, 8189, 32766] {
        for rows in [1_i64, 5, 8] {
            let q = bf16(&[2, rows, 12, D], 0x90a0_b0c0)
                .transpose(Some(&[0, 2, 1, 3]))
                .unwrap();
            let nk = bf16(&[2, rows, 2, D], 0xd0e0_f001)
                .transpose(Some(&[0, 2, 1, 3]))
                .unwrap();
            let nv = bf16(&[2, rows, 2, D], 0x1234_abcd)
                .transpose(Some(&[0, 2, 1, 3]))
                .unwrap();
            let pk = bf16(&[2, prefix, 2, D], 0x1020_3040)
                .transpose(Some(&[0, 2, 1, 3]))
                .unwrap();
            let pv = bf16(&[2, prefix, 2, D], 0x5060_7080)
                .transpose(Some(&[0, 2, 1, 3]))
                .unwrap();
            if head_rows(rows, max_query_length(6)).is_none() {
                continue;
            }
            let ctx = format!("production layout rows {rows} prefix {prefix}");
            tally.check(&ctx, &q, [&pk, &pv, &nk, &nv], rows > 1);
        }
    }
    stop_counting();

    eprintln!("class '{}': {} cases", device_class() as char, tally.cases);
    for (name, n) in KERNELS.iter().zip(tally.kernels) {
        eprintln!("  {name} {n}");
    }
    for (name, n) in ROUTES.iter().zip(tally.routes) {
        eprintln!("  {name} {n}");
    }
    for (name, n) in KERNELS.iter().zip(tally.kernels) {
        assert!(n > 0, "kernel family {name} never ran");
    }
    for (name, n) in ROUTES.iter().zip(tally.routes) {
        assert!(n > 0, "verify route {name} never ran");
    }
    tally.golden.finish();
}
