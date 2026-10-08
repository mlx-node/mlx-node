//! The K-quant kernels run only from the prebuilt `paged_attn.metallib`; there
//! is no JIT fallback. These tests take every kernel name the Metal dispatcher
//! can build (`kquant::metal_kernel_names()`, from the same builders and
//! parameter tables the dispatcher uses) and check the loaded metallib both
//! ways: every name the dispatcher can request on this device is in it, and it
//! holds no K-quant function the dispatcher cannot name.
//!
//!   cargo test -p mlx-core --release --test kquant_metallib_names

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
const BASE_NAMES: i64 = MODES * 3 * (39 + 15) + 4 + 2 + AFFINE_MODES * 3 * (1 + 15);
/// 15 modes x 3 dtypes x (qmm_t_nax {aligned, unaligned} x {batched, single}
/// plus the aligned, single Tiled64 qmm_t_nax_t64, plus gather_qmm_rhs_nax_nt
/// at bm 32 and 64), plus the bfloat16 tensor-op kernels: the three Tiled64
/// row tiers (qmm_m8/m16/m32_nax_t64) per mode and the 10 row-major
/// qmm_m8_nax (q3k, q2k, iq4nl and the seven grid formats); each affine mode
/// adds qmm_t_nax_t64 per dtype and its three Tiled64 row tiers.
const NAX_NAMES: i64 =
    MODES * 3 * (4 + 1 + 2) + MODES * 3 + 10 + AFFINE_MODES * 3 + AFFINE_MODES * 3;

struct Check {
    base: i64,
    nax: i64,
    in_library: i64,
    built: i64,
    report: String,
}

fn check(build_pipelines: bool) -> Check {
    let mut counts = [0i64; 4];
    let mut report = vec![0 as c_char; 1 << 20];
    // SAFETY: `counts` has the 4 slots the hook writes; `report` is a
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
    assert!(
        c.report.is_empty(),
        "paged_attn.metallib does not match the dispatcher:\n{}",
        c.report
    );
    // NAX kernels are built under MLX's own NAX condition; a host that cannot
    // dispatch them may still carry them.
    let expected = if nax_available() {
        vec![BASE_NAMES + NAX_NAMES]
    } else {
        vec![BASE_NAMES, BASE_NAMES + NAX_NAMES]
    };
    assert!(
        expected.contains(&c.in_library),
        "{} K-quant functions in the metallib, expected one of {expected:?}",
        c.in_library
    );
    println!(
        "{} K-quant functions in paged_attn.metallib (nax available: {})",
        c.in_library,
        nax_available()
    );
}

/// Builds a pipeline for every kernel this device can request, so a function
/// that is present but refuses to load on this OS fails here, not mid-request.
#[test]
fn every_dispatchable_kernel_builds_a_pipeline() {
    let c = check(true);
    assert!(c.report.is_empty(), "{}", c.report);
    let expected = BASE_NAMES + if nax_available() { NAX_NAMES } else { 0 };
    assert_eq!(c.built, expected, "pipelines built");
}
