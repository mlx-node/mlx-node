//! Phase 10 integration tests for FP8 K/V scale calibration determinism.
//!
//! Phase 10 wires `KvScaleManager` into `PagedKVCacheAdapter` so per-layer
//! FP8 scales (currently `1.0` placeholders) flow through the paged decode
//! path. The full 50-token-FP8-vs-bf16 end-to-end test from the spec
//! requires a model checkpoint that runs with `use_fp8_cache: Some(true)`,
//! which no production caller wires today — those tests are deferred to
//! the FP8-cache enablement work.
//!
//! What this test pins NOW: the GPU-backed calibration primitives
//! (`KvScaleManager::calibrate_layer` and `update_layer_ema`) are
//! deterministic in the sense the spec requires — same input arrays
//! produce byte-identical scales across runs. That's the load-bearing
//! contract for any future calibration runner: a warmup pass with the
//! same prompts MUST converge to the same per-layer scales every time
//! (otherwise users can't reproduce inference behavior across launches).
//!
//! Each test allocates its own MLX arrays via `mlx_array_zeros` /
//! `mlx_array_from_float32`, runs calibration, deletes the arrays, then
//! runs calibration again with freshly-allocated arrays carrying the same
//! values. Asserts byte-identical (`==`) scale values.
//!
//! These tests require a working Metal backend (the calibration kernels
//! dispatch through MLX → Metal). They are gated behind `#[ignore]` so
//! `cargo test -p mlx-paged-attn` on a non-Metal host (or any CI without
//! GPU access) reports them as "ignored" rather than green-on-skip — the
//! previous "green when Metal unavailable" pattern silently masked
//! determinism regressions on those hosts. Run on Metal-capable machines
//! with `cargo test -p mlx-paged-attn -- --ignored`.

#![cfg(target_os = "macos")]

use mlx_paged_attn::metal::KvScaleManager;

const FLOAT32: i32 = 0;

/// Assert that the host has a working Metal backend, panicking with a
/// clear message otherwise. The tests below are `#[ignore]`-gated so they
/// only run when the operator opts in via `--ignored`; in that mode we
/// fail loudly rather than silently skipping if Metal is unavailable.
fn require_metal() {
    let available = unsafe { mlx_sys::mlx_metal_is_available() };
    assert!(
        available,
        "Metal backend is required for kv_scale_calibration tests; \
         run on an Apple Silicon host with Metal enabled"
    );
}

/// Allocate a contiguous fp32 array carrying `values` with the given shape.
/// Caller deletes via `mlx_array_delete`. MUST only be called after
/// `require_metal()` (i.e. inside an `#[ignore]`-gated `#[test]` body
/// that has already verified Metal is available).
fn f32_arr(values: &[f32], shape: &[i64]) -> *mut mlx_sys::mlx_array {
    unsafe { mlx_sys::mlx_array_from_float32(values.as_ptr(), shape.as_ptr(), shape.len()) }
}

/// Allocate a fp32 zero-array of the given shape (used for the
/// "all zeros" calibration determinism case below).
fn f32_zeros(shape: &[i64]) -> *mut mlx_sys::mlx_array {
    unsafe { mlx_sys::mlx_array_zeros(shape.as_ptr(), shape.len(), FLOAT32) }
}

unsafe fn delete(handle: *mut mlx_sys::mlx_array) {
    if !handle.is_null() {
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
}

/// Same calibration inputs MUST produce byte-identical scales across
/// independent runs. This is the determinism contract a future
/// calibration runner depends on: a warmup pass run twice over the same
/// prompts (e.g. an inference server replaying its calibration set on
/// boot vs. on a fresh launch) MUST converge to the same per-layer
/// scales. Without this guarantee, users would see token streams diverge
/// from one launch to the next under nominally identical configs.
///
/// Test approach:
///  1. Build K and V arrays with known absolute-value distributions.
///  2. Run `calibrate_layer` on a fresh manager → record (k_scale, v_scale).
///  3. Delete the arrays, allocate fresh copies with the same values.
///  4. Run `calibrate_layer` on a fresh manager → assert `==` to step 2.
///
/// We reuse a deterministic data pattern (linear ramp from -3.0 to 3.0
/// over 16 elements) so the max-abs is exactly 3.0; the resulting scale
/// is `448 / 3 ≈ 149.33333…`. Floating-point bit equality across MLX
/// invocations of `abs` + `max` over the same inputs is the contract
/// being pinned — if MLX ever introduces non-determinism here (e.g.
/// different reduction order between launches), this test catches it.
#[test]
#[ignore = "Metal-required calibration determinism test; run with `--ignored` on Metal hosts"]
fn calibrate_layer_is_deterministic_across_runs() {
    require_metal();

    // Deterministic ramp: -3.0, -2.6, ..., 2.6, 3.0 (16 elements).
    let values: Vec<f32> = (0..16).map(|i| -3.0 + (i as f32) * (6.0 / 15.0)).collect();
    let shape: [i64; 2] = [4, 4];

    // First run.
    let (k1, v1) = {
        let mut manager = KvScaleManager::new(2);
        let k_arr = f32_arr(&values, &shape);
        let v_arr = f32_arr(&values, &shape);
        // SAFETY: arrays are freshly allocated and live for the call;
        // calibrate_layer internally evals + extracts via mlx_sys.
        let (k_scale, v_scale) = unsafe {
            manager
                .calibrate_layer(0, k_arr, v_arr)
                .expect("calibrate_layer (run 1) must succeed")
        };
        unsafe {
            delete(k_arr);
            delete(v_arr);
        }
        (k_scale, v_scale)
    };

    // Second run with freshly-allocated arrays carrying the same values.
    let (k2, v2) = {
        let mut manager = KvScaleManager::new(2);
        let k_arr = f32_arr(&values, &shape);
        let v_arr = f32_arr(&values, &shape);
        let (k_scale, v_scale) = unsafe {
            manager
                .calibrate_layer(0, k_arr, v_arr)
                .expect("calibrate_layer (run 2) must succeed")
        };
        unsafe {
            delete(k_arr);
            delete(v_arr);
        }
        (k_scale, v_scale)
    };

    assert_eq!(
        k1, k2,
        "k_scale must be byte-identical across calibration runs (run1={k1}, run2={k2})"
    );
    assert_eq!(
        v1, v2,
        "v_scale must be byte-identical across calibration runs (run1={v1}, run2={v2})"
    );

    // Sanity: max-abs of the ramp is 3.0, so scale should be 448/3.0.
    // We compute this through `KvScaleManager::set_scales` round-trip
    // semantics — the test would still pin run1 == run2 even if the
    // formula changes, but pinning the actual value catches accidental
    // changes to the FP8_E4M3_MAX constant or compute_scale logic.
    let expected = 448.0_f32 / 3.0;
    let abs_diff = (k1 - expected).abs();
    assert!(
        abs_diff < 1e-3,
        "k_scale should be approximately 448/3.0 = {expected}; got {k1} (|diff|={abs_diff})"
    );
}

/// All-zero K/V tensors must produce the documented "default scale = 1.0"
/// behavior (see `KvScaleManager::compute_scale`: when max_abs is below
/// MIN_SCALE = 1e-12 the scale is forced to 1.0). This test pins the
/// edge case at the GPU boundary — `mlx_array_max` over an all-zero
/// tensor returns 0.0, which feeds into `compute_scale` and hits the
/// MIN_SCALE branch. Without the early return, the kernel would divide
/// FP8_E4M3_MAX by 0 and produce inf, then the FP8 quantizer would write
/// NaNs into the cache.
#[test]
#[ignore = "Metal-required calibration determinism test; run with `--ignored` on Metal hosts"]
fn calibrate_layer_returns_unit_scale_for_zero_tensor() {
    require_metal();

    let mut manager = KvScaleManager::new(1);
    let shape: [i64; 2] = [4, 4];
    let k_arr = f32_zeros(&shape);
    let v_arr = f32_zeros(&shape);

    // SAFETY: arrays are freshly allocated and live for the call.
    let (k_scale, v_scale) = unsafe {
        manager
            .calibrate_layer(0, k_arr, v_arr)
            .expect("calibrate_layer must succeed for zero tensor")
    };
    unsafe {
        delete(k_arr);
        delete(v_arr);
    }

    assert_eq!(
        k_scale, 1.0_f32,
        "k_scale must default to 1.0 for an all-zero K tensor (got {k_scale})"
    );
    assert_eq!(
        v_scale, 1.0_f32,
        "v_scale must default to 1.0 for an all-zero V tensor (got {v_scale})"
    );
    assert!(
        manager.is_calibrated(),
        "manager must mark itself calibrated even when the tensor is all-zero"
    );
    assert_eq!(manager.k_scale(0), 1.0_f32);
    assert_eq!(manager.v_scale(0), 1.0_f32);
}

/// `update_layer_ema` over the same inputs must yield byte-identical
/// scales across runs. Same contract as `calibrate_layer_is_deterministic`
/// but exercises the EMA path (which carries running max state across
/// calls). A future online calibration runner driven by `update_layer_ema`
/// MUST be reproducible across launches when fed the same warmup prompts;
/// this test pins that property.
#[test]
#[ignore = "Metal-required calibration determinism test; run with `--ignored` on Metal hosts"]
fn update_layer_ema_is_deterministic_across_runs() {
    require_metal();

    // Two distinct ramps so the EMA blend is nontrivial.
    let ramp_a: Vec<f32> = (0..16).map(|i| -3.0 + (i as f32) * (6.0 / 15.0)).collect();
    let ramp_b: Vec<f32> = (0..16).map(|i| -1.5 + (i as f32) * (3.0 / 15.0)).collect();
    let shape: [i64; 2] = [4, 4];

    let run = || {
        let mut manager = KvScaleManager::new(1);

        // First EMA observation.
        let k1 = f32_arr(&ramp_a, &shape);
        let v1 = f32_arr(&ramp_a, &shape);
        let _ = unsafe {
            manager
                .update_layer_ema(0, k1, v1)
                .expect("update_layer_ema (call 1) must succeed")
        };
        unsafe {
            delete(k1);
            delete(v1);
        }

        // Second EMA observation with different distribution; the EMA
        // running max should now be a blend of (ramp_a, ramp_b).
        let k2 = f32_arr(&ramp_b, &shape);
        let v2 = f32_arr(&ramp_b, &shape);
        let (k_scale, v_scale) = unsafe {
            manager
                .update_layer_ema(0, k2, v2)
                .expect("update_layer_ema (call 2) must succeed")
        };
        unsafe {
            delete(k2);
            delete(v2);
        }
        (k_scale, v_scale)
    };

    let (k_run1, v_run1) = run();
    let (k_run2, v_run2) = run();

    assert_eq!(
        k_run1, k_run2,
        "EMA k_scale must be byte-identical across runs (run1={k_run1}, run2={k_run2})"
    );
    assert_eq!(
        v_run1, v_run2,
        "EMA v_scale must be byte-identical across runs (run1={v_run1}, run2={v_run2})"
    );
}

/// `KvScaleManager` round-trip via `get_all_scales` → `load_scales` is
/// byte-identical. This is the persistence contract a future scale-from-
/// disk loader depends on: scales saved in one process MUST load
/// bit-identically in another. The `test_serialization` unit test in
/// `kv_scale.rs` already covers this for the `set_scales` path; this
/// test extends the guarantee to scales that came from a real GPU-backed
/// calibration call (where floating-point representation could in
/// principle drift through `mlx_array_max`'s reduction).
#[test]
#[ignore = "Metal-required calibration determinism test; run with `--ignored` on Metal hosts"]
fn calibrated_scales_round_trip_through_serialization() {
    require_metal();

    let mut source = KvScaleManager::new(3);
    let shape: [i64; 2] = [4, 4];

    // Calibrate three layers with different distributions.
    for (layer_idx, max_abs) in [(0u32, 0.5_f32), (1, 1.5), (2, 32.0)] {
        let values: Vec<f32> = (0..16)
            .map(|i| -max_abs + (i as f32) * (2.0 * max_abs / 15.0))
            .collect();
        let k_arr = f32_arr(&values, &shape);
        let v_arr = f32_arr(&values, &shape);
        unsafe {
            source
                .calibrate_layer(layer_idx, k_arr, v_arr)
                .expect("calibrate_layer must succeed");
        }
        unsafe {
            delete(k_arr);
            delete(v_arr);
        }
    }

    let (k_scales, v_scales) = source.get_all_scales();
    assert_eq!(k_scales.len(), 3);
    assert_eq!(v_scales.len(), 3);

    // Load into a fresh manager and assert bit-equal scale lookup.
    let mut sink = KvScaleManager::new(3);
    sink.load_scales(&k_scales, &v_scales);
    for layer_idx in 0..3u32 {
        assert_eq!(
            source.k_scale(layer_idx),
            sink.k_scale(layer_idx),
            "k_scale must round-trip bit-equal at layer {layer_idx}"
        );
        assert_eq!(
            source.v_scale(layer_idx),
            sink.v_scale(layer_idx),
            "v_scale must round-trip bit-equal at layer {layer_idx}"
        );
    }
    assert!(
        sink.is_calibrated(),
        "load_scales must mark sink as calibrated"
    );
}
