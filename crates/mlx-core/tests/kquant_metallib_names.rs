//! The K-quant kernels run only from the prebuilt `paged_attn.metallib`; there
//! is no JIT fallback. These tests take every kernel name the Metal dispatcher
//! can build (`kquant::metal_kernel_names()`, from the same builders and
//! parameter tables the dispatcher uses) and check the loaded metallib both
//! ways: every name the dispatcher can request on this device is in it, and it
//! holds no K-quant function the dispatcher cannot name.
//!
//!   cargo test -p mlx-core --release --test kquant_metallib_names
//!
//! The name-set check is the every-PR gate. Building a pipeline for each of
//! the ~2800 names is Metal compiler time (~2 min on an M-series laptop, far
//! longer on a hosted runner) and catches a different defect class: a
//! function that is present but refuses to load on this OS, which changes on
//! a metallib / MLX pin bump, not per PR. That test is `#[ignore]`d and runs
//! in CI's `model-test` lane (`.github/workflows/ci.yml`, the `qwen3` leg):
//!
//!   cargo test -p mlx-core --test kquant_metallib_names -- --ignored
//!
//! Custom JIT kernels consume a generated copy of the same K-quant header.
//! A separate compile check covers its inlined local helpers, which the
//! prebuilt library's name set cannot validate.

#![cfg(target_os = "macos")]

use std::ffi::{CStr, c_char};

/// The modes kquant.metal instantiates: q6k, q4k, q5k, q3k, q2k, iq4nl, iq4xs,
/// the legacy iq3s8 and the seven grid formats iq2xxs, iq2xs, iq2s, iq3xxs,
/// iq1s, iq1m, iq3s.
const MODES: i64 = 15;
/// The MLX affine modes a4g64 and a8g64: dequantize and the Tiled64 families
/// only.
const AFFINE_MODES: i64 = 2;
/// 15 modes x 3 dtypes x (39 row-major families + 15 Tiled64 "_t64" families:
/// qmv_t64 at 8 and 16 k-splits per 32-, 16- and 8-row threadgroup, qmv_wide
/// nv 2..8, qmm_t, qmm_t_splitk), plus 4 bfloat16 qmv_sg8 kernels and the 2
/// sg8 prep kernels (group sizes 16 and 32), plus each affine mode's
/// dequantize and 15 Tiled64 families.
// Q4_K and affine4/64 each add a tiled register-MMA kernel; the six
// register-MMA formats each add a split-K kernel; three exact M1 gathers
// consume Q4_K/Q5_K/Q6_K T64 expert stacks.
const BASE_NAMES: i64 = MODES * 3 * (39 + 15) + 4 + 2 + AFFINE_MODES * 3 * (1 + 15) + 2 + 6 + 3;
/// 15 modes x 3 dtypes x (qmm_t_nax {aligned, unaligned} x {batched, single}
/// plus the aligned, single Tiled64 qmm_t_nax_t64, plus gather_qmm_rhs_nax_nt
/// at bm 32 and 64), plus the bfloat16 tensor-op kernels: the three Tiled64
/// row tiers (qmm_m8/m16/m32_nax_t64) per mode and the 10 row-major
/// qmm_m8_nax (q3k, q2k, iq4nl and the seven grid formats); each affine mode
/// adds qmm_t_nax_t64 per dtype and its three Tiled64 row tiers.
// The Q6_K tiled M8 head adds one FP32-output NAX kernel; seven bfloat16
// Tiled64 modes add the residual and swiglu store-epilogue GEMMs.
const NAX_NAMES: i64 =
    MODES * 3 * (4 + 1 + 2) + MODES * 3 + 10 + AFFINE_MODES * 3 + AFFINE_MODES * 3 + 1 + 7 * 2;
/// The GGUF planes kernels (`mlx_gguf_planes.cpp`, Splash's sources in
/// `src/metal/splash_gguf`): 8 formats x 3 row tiers (8, 16, 32) x {a, r, g,
/// gate_up}, the fused kernel at 3 row tiers and `gguf_repack`.
const PLANES_NAMES: i64 = 8 * 3 * 4 + 3 + 1;
/// The planes' test-only kernels, in `gguf_planes_test.metallib` alone:
/// `gguf_dequant` of both builds and the two Q4_K 8-row fp32 epilogue kernels.
const PLANES_TEST_NAMES: i64 = 2 + 2;

struct Check {
    base: i64,
    nax: i64,
    in_library: i64,
    built: i64,
    planes: i64,
    report: String,
}

fn check(build_pipelines: bool) -> Check {
    let mut counts = [0i64; 5];
    let mut report = vec![0 as c_char; 1 << 20];
    // SAFETY: `counts` has the 5 slots the hook writes; `report` is a
    // writable buffer of the length passed.
    let ok = unsafe {
        mlx_sys::mlx_test_kquant_metallib_check(
            build_pipelines,
            counts.as_mut_ptr(),
            report.as_mut_ptr(),
            report.len(),
        )
    };
    assert!(
        ok,
        "mlx_test_kquant_metallib_check failed (no Metal device, or the metallib did not load)"
    );
    // SAFETY: the hook wrote a NUL-terminated string into `report`.
    let report = unsafe { CStr::from_ptr(report.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Check {
        base: counts[0],
        nax: counts[1],
        in_library: counts[2],
        built: counts[3],
        planes: counts[4],
        report,
    }
}

fn nax_available() -> bool {
    // SAFETY: nullary predicate that catches internally.
    unsafe { mlx_sys::mlx_metal_is_nax_available() }
}

#[test]
fn every_dispatchable_kernel_is_in_the_metallib() {
    let c = check(false);
    assert_eq!(c.base, BASE_NAMES, "dispatcher base kernel names");
    assert_eq!(c.nax, NAX_NAMES, "dispatcher NAX kernel names");
    assert_eq!(c.planes, PLANES_NAMES, "GGUF planes kernel names");
    assert!(
        c.report.is_empty(),
        "paged_attn.metallib does not match the dispatcher:\n{}",
        c.report
    );
    // NAX kernels are built under MLX's own NAX condition; a host that cannot
    // dispatch them may still carry them.
    let expected = if nax_available() {
        vec![BASE_NAMES + NAX_NAMES + PLANES_NAMES]
    } else {
        vec![
            BASE_NAMES + PLANES_NAMES,
            BASE_NAMES + NAX_NAMES + PLANES_NAMES,
        ]
    };
    assert!(
        expected.contains(&c.in_library),
        "{} K-quant and gguf_ functions in the metallib, expected one of {expected:?}",
        c.in_library
    );
    println!(
        "{} K-quant and gguf_ functions in paged_attn.metallib (nax available: {})",
        c.in_library,
        nax_available()
    );
}

/// A planes kernel the library lacks is reported by name: the check is run
/// with `gguf_decode_iq4xs_m8_r` taken out of the library's set.
#[test]
fn a_missing_planes_kernel_is_named() {
    // SAFETY: test-only thread-local setter.
    unsafe { mlx_sys::mlx_test_gguf_planes_mutation(5) };
    let c = check(false);
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_gguf_planes_mutation(0) };
    assert_eq!(
        c.report.trim(),
        "missing gguf_decode_iq4xs_m8_r",
        "the check must name the one missing planes kernel"
    );
}

/// A test-only planes kernel in the shipped library is reported by name: the
/// check is run with `gguf_dequant` put into the library's set.
#[test]
fn a_test_kernel_in_the_shipped_metallib_is_named() {
    // SAFETY: test-only thread-local setter.
    unsafe { mlx_sys::mlx_test_gguf_planes_mutation(9) };
    let c = check(false);
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_gguf_planes_mutation(0) };
    assert_eq!(
        c.report.trim(),
        "unexpected gguf_dequant",
        "the check must name the test kernel in paged_attn.metallib"
    );
}

/// `gguf_planes_test.metallib` holds the planes' test-only kernels and
/// nothing else.
#[test]
fn the_test_metallib_holds_the_test_kernels_only() {
    mlx_sys::gguf_planes_use_test_library();
    let mut count = 0i64;
    let mut report = vec![0 as c_char; 1 << 16];
    // SAFETY: `report` is a writable buffer of the length passed.
    let ok = unsafe {
        mlx_sys::mlx_test_gguf_planes_test_metallib_check(
            &mut count,
            report.as_mut_ptr(),
            report.len(),
        )
    };
    assert!(ok, "gguf_planes_test.metallib did not load");
    // SAFETY: the hook wrote a NUL-terminated string into `report`.
    let report = unsafe { CStr::from_ptr(report.as_ptr()) }.to_string_lossy();
    assert!(
        report.is_empty(),
        "gguf_planes_test.metallib does not match the test kernel names:\n{report}"
    );
    assert_eq!(
        count, PLANES_TEST_NAMES,
        "functions in gguf_planes_test.metallib"
    );
    println!("{count} functions in gguf_planes_test.metallib");
}

#[test]
fn jit_preamble_compiles_with_local_helpers() {
    // SAFETY: nullary predicate that catches Metal compiler errors internally.
    assert!(
        unsafe { mlx_sys::mlx_test_kquant_jit_preamble_check() },
        "K-quant JIT preamble failed to compile with its SG8 and split helpers"
    );
}

/// Builds a pipeline for every kernel this device can request, so a function
/// that is present but refuses to load on this OS fails here, not mid-request.
/// Pin-bump gate, not a per-PR one: see the module docs.
#[test]
#[ignore = "builds every pipeline (~minutes); run in the model-e2e lane: cargo test -p mlx-core --test kquant_metallib_names -- --ignored"]
fn every_dispatchable_kernel_builds_a_pipeline() {
    let c = check(true);
    assert!(c.report.is_empty(), "{}", c.report);
    let expected = BASE_NAMES
        + if nax_available() {
            NAX_NAMES + PLANES_NAMES
        } else {
            0
        };
    assert_eq!(c.built, expected, "pipelines built");
}
