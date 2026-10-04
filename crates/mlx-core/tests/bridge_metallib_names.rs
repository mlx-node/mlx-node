//! The segmented SDPA and mixed-affine `qmv_wide` kernels run only from the
//! prebuilt `paged_attn.metallib`; there is no JIT fallback. These tests take
//! every kernel name each dispatcher can request (from the same tables and
//! name builders it uses) and check the loaded metallib both ways: every name
//! is in it, and it holds no function of the family the dispatcher cannot
//! name. The pipeline tests build every pipeline through the dispatcher's own
//! loader, each segmented function-constant specialization included.
//!
//!   cargo test -p mlx-core --release --test bridge_metallib_names

#![cfg(target_os = "macos")]

use std::ffi::{CStr, CString, c_char};

/// one_pass, 2pass_1, verify_2pass_1.
const SEGMENTED_NAMES: i64 = 3;
/// Partition counts the vector-SDPA policy returns: 32, 64, 128, 256, 512, 1024.
const PARTITIONS: i64 = 6;
/// one_pass x causal {false, true}; 2pass_1 x causal x partitions; causal
/// verify x partitions x gqa 1..=32 x rows 2..=8.
const SEGMENTED_PIPELINES: i64 = 2 + 2 * PARTITIONS + PARTITIONS * 32 * 7;
/// qmv_wide tile widths 2..=8.
const AFFINE_MIXED_NAMES: i64 = 7;

struct Check {
    names: i64,
    in_library: i64,
    built: i64,
    report: String,
}

fn check(family: &str, build_pipelines: bool) -> Check {
    let family = CString::new(family).expect("family name");
    let mut counts = [0i64; 3];
    let mut report = vec![0 as c_char; 1 << 16];
    // SAFETY: `family` is NUL-terminated, `counts` has the 3 slots the hook
    // writes and `report` is a writable buffer of the length passed.
    let ok = unsafe {
        mlx_sys::mlx_test_bridge_metallib_check(
            family.as_ptr(),
            build_pipelines,
            counts.as_mut_ptr(),
            report.as_mut_ptr(),
            report.len(),
        )
    };
    assert!(
        ok,
        "mlx_test_bridge_metallib_check failed (no Metal device, a kernel did not load, or the metallib did not load)"
    );
    // SAFETY: the hook wrote a NUL-terminated string into `report`.
    let report = unsafe { CStr::from_ptr(report.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Check {
        names: counts[0],
        in_library: counts[1],
        built: counts[2],
        report,
    }
}

fn assert_names(family: &str, expected: i64) {
    let c = check(family, false);
    assert_eq!(c.names, expected, "{family} dispatcher kernel names");
    assert!(
        c.report.is_empty(),
        "paged_attn.metallib does not match the {family} dispatcher:\n{}",
        c.report
    );
    assert_eq!(c.in_library, expected, "{family} functions in the metallib");
}

#[test]
fn every_segmented_sdpa_kernel_is_in_the_metallib() {
    assert_names("segmented_sdpa", SEGMENTED_NAMES);
}

#[test]
fn every_affine_mixed_kernel_is_in_the_metallib() {
    assert_names("affine_mixed", AFFINE_MIXED_NAMES);
}

/// A function that is present but refuses to load or specialize on this OS
/// fails here, not mid-request.
#[test]
fn every_segmented_sdpa_specialization_builds_a_pipeline() {
    let c = check("segmented_sdpa", true);
    assert!(c.report.is_empty(), "{}", c.report);
    assert_eq!(c.built, SEGMENTED_PIPELINES, "pipelines built");
}

#[test]
fn every_affine_mixed_kernel_builds_a_pipeline() {
    let c = check("affine_mixed", true);
    assert!(c.report.is_empty(), "{}", c.report);
    assert_eq!(c.built, AFFINE_MIXED_NAMES, "pipelines built");
}
