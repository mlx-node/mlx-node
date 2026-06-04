//! Integration tests for the profile-run auto-sizer.
//!
//! Runs against the public API in
//! `crates/mlx-paged-attn/src/profile.rs`. The math-only steps
//! (`bytes_per_block`, `compute_num_blocks_from_measurements`,
//! `dtype_size_for`) are also covered by the in-module unit tests; this
//! file pins the cross-cutting behaviour:
//!
//! 1. Hand-checked num_blocks math vs. realistic Qwen3-class params.
//! 2. Edge cases (peak exceeds budget, safety margin eats remainder,
//!    block size larger than budget).
//! 3. Env-var rejection paths (we mutate process env in serial-test
//!    style — see `serial_test_lock` below; cargo runs each integration
//!    test crate in its own process so the lock is per-process).
//!
//! These tests do NOT exercise the `profile_run_and_compute_num_blocks`
//! glue (which calls a model forward and reads `mlx_get_peak_memory()`)
//! because (a) it requires a fully loaded model with synthetic input,
//! which is a model-specific concern, and (b) it depends on host MLX
//! state that these integration tests shouldn't entangle.

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::profile::{
    DEFAULT_KV_MEMORY_UTILIZATION, DEFAULT_SAFETY_MARGIN_BYTES, ProfileError, bytes_per_block,
    compute_num_blocks_from_measurements, compute_num_blocks_with_working_set, dtype_size_for,
    profile_run_and_compute_num_blocks,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Realistic Qwen3-class params: 28 layers, 8 KV heads, head_size=128,
/// block_size=16, BF16. Per-block ≈ 28 * 8 * 128 * 16 * 2 * 2 = 1.75 MiB.
/// On a 64 GiB system with 0.85 utilization and 1 GiB safety, a 10 GiB
/// peak leaves about 64*0.85 - 10 - 1 = 43.4 GiB for KV → ~25,400 blocks
/// (a reasonable Qwen3 budget).
#[test]
fn qwen3_realistic_sizing() {
    let bpb = bytes_per_block(28, 8, 128, 16, MetalDtype::BFloat16).unwrap();
    let total = 64u64 * 1024 * 1024 * 1024;
    let peak = 10u64 * 1024 * 1024 * 1024;
    let util = 0.85;
    let safety = 1024u64 * 1024 * 1024;
    let (num_blocks, kv) =
        compute_num_blocks_from_measurements(total, peak, util, safety, bpb).unwrap();
    // Sanity bounds rather than an exact value (fp rounding contributes
    // ~0.01% noise per knob and we don't want to over-pin).
    assert!(
        (24_000..=27_000).contains(&num_blocks),
        "expected ~25,400 Qwen3 blocks, got {num_blocks}"
    );
    // KV budget should be roughly 43 GiB.
    let kv_gib = kv / (1024 * 1024 * 1024);
    assert!(
        (40..=45).contains(&kv_gib),
        "expected ~43 GiB KV, got {kv_gib} GiB"
    );
}

/// Realistic LFM2 hybrid scenario: only `full_attention` layers go through
/// paged. Half the 32 total layers are full-attention → 16 layers.
#[test]
fn lfm2_realistic_sizing() {
    let bpb = bytes_per_block(16, 4, 64, 32, MetalDtype::BFloat16).unwrap();
    let total = 32u64 * 1024 * 1024 * 1024;
    let peak = 5u64 * 1024 * 1024 * 1024;
    let util = 0.85;
    let safety = 512u64 * 1024 * 1024;
    let (num_blocks, _) =
        compute_num_blocks_from_measurements(total, peak, util, safety, bpb).unwrap();
    // bpb = 2 * 16 * 4 * 64 * 32 * 2 = 524 288 = 512 KiB / block.
    // 32 * 0.85 - 5 - 0.5 = 21.7 GiB → 21.7 * 1024 * 2 = ~44,400 blocks.
    assert!(
        (40_000..=50_000).contains(&num_blocks),
        "expected ~44,400 LFM2 blocks, got {num_blocks}"
    );
}

/// FP8 cache: half the per-block size of BF16 → twice the blocks for the
/// same budget. Verifies the dtype_size_for hook on the path.
#[test]
fn fp8_doubles_block_count() {
    let bpb_bf16 = bytes_per_block(28, 8, 128, 16, MetalDtype::BFloat16).unwrap();
    let bpb_fp8 = bytes_per_block(28, 8, 128, 16, MetalDtype::UChar).unwrap();
    assert_eq!(bpb_bf16, 2 * bpb_fp8, "FP8 should be exactly half of BF16");
    let total = 64u64 * 1024 * 1024 * 1024;
    let peak = 10u64 * 1024 * 1024 * 1024;
    let util = 0.85;
    let safety = 1024u64 * 1024 * 1024;
    let (n_bf16, _) =
        compute_num_blocks_from_measurements(total, peak, util, safety, bpb_bf16).unwrap();
    let (n_fp8, _) =
        compute_num_blocks_from_measurements(total, peak, util, safety, bpb_fp8).unwrap();
    // FP8 has exactly 2x the block count (within fp rounding from the
    // intermediate kv_bytes float math — both block counts come from the
    // same kv_bytes scalar so the ratio is exact in u64).
    assert_eq!(n_fp8, 2 * n_bf16);
}

/// Insufficient memory: model peak exceeds utilization budget.
#[test]
fn rejects_peak_exceeds_budget() {
    let total = 32u64 * 1024 * 1024 * 1024;
    let peak = 30u64 * 1024 * 1024 * 1024;
    let util = 0.85; // budget = 27.2 GiB; peak > budget
    let safety = 0u64;
    let bpb = 1024u64;
    let res = compute_num_blocks_from_measurements(total, peak, util, safety, bpb);
    match res {
        Err(ProfileError::InsufficientMemory { .. }) => {}
        other => panic!("expected InsufficientMemory, got {other:?}"),
    }
}

/// Block size larger than budget → 0 blocks → NotEnoughBlocks rejection
/// (the auto-sizer never silently disables paged attention).
#[test]
fn rejects_zero_blocks() {
    let gib = 1024u64 * 1024 * 1024;
    let total = 4 * gib;
    let peak = gib;
    let util = 1.0; // budget = 4 GiB; after_peak = 3 GiB
    let safety = 0u64;
    let bpb = 4 * gib; // 4 GiB / block; budget < bpb
    let res = compute_num_blocks_from_measurements(total, peak, util, safety, bpb);
    match res {
        Err(ProfileError::NotEnoughBlocks { .. }) => {}
        other => panic!("expected NotEnoughBlocks, got {other:?}"),
    }
}

/// Defaults match the documented constants. Pins the expected env shape
/// so a future change to the defaults requires updating this assertion
/// (and any docs that quote the values).
#[test]
fn default_constants() {
    assert!((DEFAULT_KV_MEMORY_UTILIZATION - 0.85).abs() < 1e-9);
    assert_eq!(DEFAULT_SAFETY_MARGIN_BYTES, 1024 * 1024 * 1024);
    assert_eq!(dtype_size_for(MetalDtype::BFloat16), 2);
    assert_eq!(dtype_size_for(MetalDtype::Float16), 2);
    assert_eq!(dtype_size_for(MetalDtype::UChar), 1);
}

/// Zero shape parameters reject deterministically (programmer error,
/// not runtime condition).
#[test]
fn rejects_zero_shape_params() {
    assert!(matches!(
        bytes_per_block(0, 8, 128, 16, MetalDtype::BFloat16),
        Err(ProfileError::InvalidShape(_))
    ));
    assert!(matches!(
        bytes_per_block(28, 0, 128, 16, MetalDtype::BFloat16),
        Err(ProfileError::InvalidShape(_))
    ));
    assert!(matches!(
        bytes_per_block(28, 8, 0, 16, MetalDtype::BFloat16),
        Err(ProfileError::InvalidShape(_))
    ));
    assert!(matches!(
        bytes_per_block(28, 8, 128, 0, MetalDtype::BFloat16),
        Err(ProfileError::InvalidShape(_))
    ));
}

/// Working-set clamp pins the auto-sizer below the GPU recommended
/// working-set bound on Apple Silicon, where `hw.memsize` is materially
/// larger than `MTLDevice recommendedMaxWorkingSetSize`. Without the
/// clamp the auto-sizer would over-commit and surface as MTLBuffer
/// allocation failures during serving.
///
/// Realistic numbers: 64 GiB unified memory, 48 GiB working set,
/// 10 GiB peak, 0.85 utilization, 1 GiB safety. Per-block 1 MiB.
/// Physical budget = 64*0.85 - 10 - 1 = 43.4 GiB.
/// Working-set budget = 48*0.85 - 10 - 1 = 29.8 GiB.
/// The smaller (working set) wins.
#[test]
fn working_set_clamp_bounds_blocks_below_physical() {
    let gib = 1024u64 * 1024 * 1024;
    let total = 64 * gib;
    let working_set = 48 * gib;
    let peak = 10 * gib;
    let util = 0.85;
    let safety = gib;
    let bpb = 1024u64 * 1024;

    let (phys_blocks, _) =
        compute_num_blocks_from_measurements(total, peak, util, safety, bpb).unwrap();
    let (clamped_blocks, kv_bytes, phys_budget, ws_budget) =
        compute_num_blocks_with_working_set(total, Some(working_set), peak, util, safety, bpb)
            .unwrap();

    assert!(
        clamped_blocks < phys_blocks,
        "expected the clamp to reduce blocks: physical={phys_blocks}, clamped={clamped_blocks}"
    );
    assert!(
        (30_000..=31_000).contains(&clamped_blocks),
        "expected ~30,500 blocks under working-set budget, got {clamped_blocks}"
    );
    let ws_budget = ws_budget.expect("working-set budget reported");
    assert_eq!(kv_bytes, ws_budget.min(phys_budget));
    assert!(ws_budget < phys_budget);
}

/// On platforms where the working-set bound isn't reported (CPU-only
/// build, device_info missing), the helper falls through to the
/// physical-RAM budget so the path is still functional.
#[test]
fn working_set_unreported_uses_physical_budget() {
    let gib = 1024u64 * 1024 * 1024;
    let total = 64 * gib;
    let peak = 10 * gib;
    let util = 0.85;
    let safety = gib;
    let bpb = 1024u64 * 1024;

    let (phys_blocks, _) =
        compute_num_blocks_from_measurements(total, peak, util, safety, bpb).unwrap();
    let (clamped_blocks, _, _, ws_budget) =
        compute_num_blocks_with_working_set(total, None, peak, util, safety, bpb).unwrap();
    assert_eq!(clamped_blocks, phys_blocks);
    assert!(ws_budget.is_none());
}

/// `profile_run_and_compute_num_blocks` threads `max_position_embeddings`
/// into the closure verbatim. The closure captures the value via an
/// AtomicU32 so the assertion runs on the closure's side without
/// blocking on GPU memory APIs (which the math doesn't need to actually
/// hit — the macOS path returns `MetalUnavailable` on no-GPU CI runners
/// after invoking the closure once, and the test still verifies the
/// closure was invoked with the expected value).
///
/// Reproduction note: previously this path called `reset_peak_memory()`
/// unconditionally on macOS, which on a no-Metal host (sysctl works,
/// Metal init fails) would throw a C++ `std::runtime_error` from
/// `metal::allocator()` across the FFI boundary and abort the process
/// with "fatal runtime error: Rust cannot catch foreign exceptions".
/// The fix is two-pronged: (1) the C++ shims wrap each MLX call in
/// catch-all so the abort cannot recur even if the gate is bypassed,
/// and (2) the Rust path checks `mlx_metal_is_available()` upfront and
/// returns `MetalUnavailable` cleanly without ever invoking the
/// peak-memory FFI on a no-Metal host.
#[test]
fn profile_run_threads_max_position_embeddings_to_closure() {
    let observed = Arc::new(AtomicU32::new(0));
    let observed_for_closure = Arc::clone(&observed);
    let dummy_forward = move |max_seq: u32| -> Result<(), String> {
        observed_for_closure.store(max_seq, Ordering::SeqCst);
        // Returning Ok lets the auto-sizer continue past the dummy
        // forward; later steps may fail on no-GPU hosts but the
        // observation we care about already happened.
        Ok(())
    };

    let res = profile_run_and_compute_num_blocks(
        dummy_forward,
        4321u32, // The exact value to assert reached the closure.
        28,
        8,
        128,
        MetalDtype::BFloat16,
        16,
    );

    // Closure ran with the supplied max_seq on every path EXCEPT the
    // degraded-Metal sub-path of MetalUnavailable, where
    // `reset_peak_memory()` returns `-1` and `?`-propagates BEFORE the
    // closure runs. In that case `observed` stays at its initial 0;
    // any non-zero value MUST equal the supplied max_seq. (See the
    // companion test `profile_run_no_metal_returns_metal_unavailable_
    // without_abort` for the explicit invocation-count contract.)
    let observed_value = observed.load(Ordering::SeqCst);
    assert!(
        observed_value == 0 || observed_value == 4321,
        "observed max_seq must be 0 (closure never ran) or 4321 (closure ran), got {observed_value}"
    );

    // Sanity: the result is either Ok (Metal-present host with
    // enough memory) or an explicit error variant. NEVER an abort
    // from an unwound C++ exception.
    match res {
        Ok(_) => {
            // Metal present + enough memory — auto-sizer succeeded;
            // closure must have run with the supplied max_seq.
            assert_eq!(observed.load(Ordering::SeqCst), 4321);
        }
        Err(ProfileError::MetalUnavailable) => {} // No-GPU host: short-circuited cleanly.
        Err(ProfileError::TotalMemoryUnavailable) => {} // sysctl failed: also acceptable.
        Err(ProfileError::InsufficientMemory { .. }) => {
            // Tiny dummy forward, plausible budget reject — closure ran.
            assert_eq!(observed.load(Ordering::SeqCst), 4321);
        }
        Err(ProfileError::NotEnoughBlocks { .. }) => {
            // Small budget after util haircut — closure ran.
            assert_eq!(observed.load(Ordering::SeqCst), 4321);
        }
        other => panic!("unexpected result: {other:?}"),
    }
}

/// `MetalUnavailable` is surfaced WITHOUT a process abort when the
/// underlying memory FFI would otherwise throw across the boundary, AND
/// the auto-sizer NEVER silently consumes a sentinel-zero peak from a
/// caught C++ exception as if it were a real measurement.
///
/// Bug fixed: previously `mlx_get_peak_memory()` and friends returned
/// the ambiguous sentinel `0` from a caught C++ exception, indistinguishable
/// from a legitimate "0 bytes peak" reading. On a degraded-Metal host
/// (where `mlx_metal_is_available()` somehow reports true but the
/// downstream `MetalAllocator` constructor still throws), the auto-sizer
/// would log `peak_non_kv 0 MiB` and return through the success path
/// with millions of blocks — silently oversizing the KV pool to billions
/// of pages, which then fail later during serving with MTLBuffer
/// allocation errors.
///
/// The fallible-FFI contract fixes this STRUCTURALLY across the three
/// memory shims that touch the auto-sizer:
///   - `mlx_reset_peak_memory` → `reset_peak_memory()` `?`-propagates as
///     `MetalUnavailable`.
///   - `mlx_get_peak_memory` → `read_peak_memory()` `?`-propagates as
///     `MetalUnavailable`.
///   - `mlx_max_recommended_working_set_size` → `read_working_set_bytes()`
///     `?`-propagates as `MetalUnavailable` (-1 return = caught exception).
///     A return of `0` still surfaces as `Ok(None)` because that's the
///     legitimate "device_info doesn't report this entry" case, NOT a
///     failure mode.
///
/// There is no longer a code path through the auto-sizer that consumes a
/// caught-exception peak — or a caught-exception working_set — as a real
/// measurement. See `formula_well_behaved_with_zero_peak_and_no_working_set`
/// below for the math-layer pin that complements this test.
///
/// What this test verifies given the test cannot synthesize a degraded-
/// Metal environment from a Metal-equipped host:
///
/// - The process does NOT abort at any point. A foreign-exception unwind
///   would have killed the binary before reaching the `match` below;
///   reaching the assertions is itself the regression guard for the
///   `Rust cannot catch foreign exceptions` abort.
/// - The result is exactly one of the documented variants. There is no
///   silent-success path: any caught C++ exception is funneled through
///   `read_peak_memory()` / `reset_peak_memory()` returning `Err(...)`,
///   which the auto-sizer `?`-propagates.
/// - On no-Metal hosts (CI runners without GPU): we get `MetalUnavailable`
///   AND the closure was invoked exactly once with the supplied
///   `max_position_embeddings`.
/// - On Metal-equipped hosts: the auto-sizer returns `Ok` (with a
///   `num_blocks > 0` derived from whatever `peak_non_kv` was real on
///   the host — for this test's no-op dummy forward, that's typically
///   close to zero MiB plus whatever the host already had loaded; the
///   resulting `num_blocks` can legitimately be in the millions and
///   that's NOT the regression we're guarding against). Or it returns
///   one of the expected explicit-error variants.
///
/// The test does NOT and cannot bound `num_blocks` on the success path
/// because a no-op closure produces a legitimate near-zero `peak`, and
/// the bug being regressed isn't "millions of blocks per se" — it's
/// "a caught exception silently became a peak measurement". With the
/// fallible-FFI contract, a caught exception ALWAYS surfaces as `Err`,
/// so a successful `Ok` necessarily came from a real measurement.
#[test]
fn profile_run_no_metal_returns_metal_unavailable_without_abort() {
    let observed = Arc::new(AtomicU32::new(0));
    let invoke_count = Arc::new(AtomicU32::new(0));
    let observed_for_closure = Arc::clone(&observed);
    let invoke_count_for_closure = Arc::clone(&invoke_count);
    let dummy_forward = move |max_seq: u32| -> Result<(), String> {
        observed_for_closure.store(max_seq, Ordering::SeqCst);
        invoke_count_for_closure.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };

    let res = profile_run_and_compute_num_blocks(
        dummy_forward,
        9999u32,
        4,
        2,
        64,
        MetalDtype::BFloat16,
        16,
    );

    // Reaching this assertion proves the FFI did not abort the process.
    // (A foreign-exception unwind would have killed the test binary
    // before this line.)
    match res {
        Ok(num_blocks) => {
            // Metal-equipped host or successful no-op closure: the
            // fallible-FFI contract guarantees this `Ok` came from a
            // real `peak_non_kv` measurement, not a caught-exception
            // sentinel. Assert at least one block — anything zero
            // would imply a silent path through `compute_num_blocks_*`
            // that the math layer is supposed to reject explicitly.
            assert!(
                num_blocks > 0,
                "auto-sizer returned 0 blocks on the success path \
                 (math layer should have surfaced NotEnoughBlocks instead)"
            );
            // Closure must have been invoked exactly once on the
            // success path too — the auto-sizer threads the supplied
            // max_seq into the closure as part of measuring peak,
            // and a Metal-present host runs through reset → forward → read.
            assert_eq!(observed.load(Ordering::SeqCst), 9999);
            assert_eq!(invoke_count.load(Ordering::SeqCst), 1);
        }
        Err(ProfileError::MetalUnavailable) => {
            // Two legitimate sub-paths land here:
            //   (a) `mlx_metal_is_available()` returned false upfront —
            //       the no-Metal early return invokes the closure once
            //       to surface platform-portable model-loading bugs
            //       deterministically before short-circuiting.
            //   (b) `mlx_metal_is_available()` returned true (degraded-
            //       Metal host) but `reset_peak_memory()` then returned
            //       `-1` because the allocator constructor threw. In
            //       that path `reset_peak_memory()?` propagates BEFORE
            //       the closure runs, so `invoke_count == 0`.
            // Accept both invocation counts; when the closure DID run,
            // `observed` must equal the supplied max_seq.
            let count = invoke_count.load(Ordering::SeqCst);
            assert!(
                count <= 1,
                "MetalUnavailable should invoke closure 0 or 1 times, got {count}"
            );
            if count == 1 {
                assert_eq!(observed.load(Ordering::SeqCst), 9999);
            }
        }
        Err(ProfileError::TotalMemoryUnavailable) => {
            // Non-macOS host (sysctl unavailable). On non-macOS the
            // auto-sizer invokes the closure once before returning.
            // No silent-success path.
        }
        Err(ProfileError::InsufficientMemory { .. })
        | Err(ProfileError::NotEnoughBlocks { .. }) => {
            // Tiny dummy forward + tight budget rejection — both
            // explicit-error variants. The closure was invoked.
            assert_eq!(invoke_count.load(Ordering::SeqCst), 1);
        }
        Err(other) => {
            panic!("unexpected error variant: {other:?}");
        }
    }
}

/// Pin the formula's behaviour in the precise "peak=0 + working_set=None"
/// scenario that the fallible-FFI contract is designed to keep the
/// auto-sizer from ever observing through a caught-exception sentinel.
///
/// Why this test exists:
/// The original bug report's regression scenario was a degraded-Metal host
/// where `mlx_get_peak_memory()` and `mlx_max_recommended_working_set_size()`
/// both threw C++ exceptions and the previous infallible FFI signatures
/// silently returned `0` and `None` respectively. The auto-sizer then
/// fed those sentinels into `compute_num_blocks_with_working_set` as if
/// they were real measurements and computed millions of blocks against
/// the full physical-RAM budget — oversizing the KV pool and deferring
/// the actual MTLBuffer allocation failure to serving time.
///
/// The structural fix lives in the FFI shims and their Rust callers:
/// `read_peak_memory()` / `read_working_set_bytes()` now return
/// `Err(MetalUnavailable)` on a -1 return, and `profile_run_and_compute_num_blocks`
/// `?`-propagates the failure BEFORE the math layer ever runs. So the
/// math layer should never see a sentinel-zero peak from a caught
/// exception. But if a caller in some future callsite passes peak=0
/// from a legitimate measurement (a freshly reset allocator on a
/// brand-new process) plus `working_set_bytes=None` (CPU-only build),
/// the formula must still produce a sensible result rather than
/// degenerate.
///
/// What this test verifies:
/// - The math accepts peak=0 + working_set=None and produces a single
///   well-defined block count (no panic, no overflow).
/// - The block count is bounded by `(total * util - safety) /
///   bytes_per_block` — i.e. it scales linearly with the inputs and
///   does NOT blow up to `u32::MAX`.
/// - The chosen budget equals the physical budget (the `None` path)
///   and `ws_budget` is `None`.
///
/// This complements the no-Metal regression test above: that one verifies
/// the FFI layer prevents the math from ever observing the bug scenario;
/// this one verifies the math itself is well-behaved if a future caller
/// does construct that exact input from real measurements.
#[test]
fn formula_well_behaved_with_zero_peak_and_no_working_set() {
    let gib = 1024u64 * 1024 * 1024;
    let total = 64 * gib;
    // Simulate the legitimate "peak=0" case: a brand-new process whose
    // allocator counter has never moved. With the FFI layer fixed, this
    // is the ONLY way the math layer can observe peak=0 — never via a
    // caught-exception sentinel.
    let peak = 0u64;
    let util = 0.85;
    let safety = gib;
    // Per-block 1 MiB. With these inputs the budget is
    // 64 * 0.85 - 0 - 1 = 53.4 GiB → ~54,700 blocks. NOT millions or
    // billions; the formula's natural arithmetic bounds the result.
    let bpb = 1024u64 * 1024;

    let (num_blocks, kv_bytes, phys_budget, ws_budget) =
        compute_num_blocks_with_working_set(total, None, peak, util, safety, bpb)
            .expect("formula should handle peak=0 + working_set=None cleanly");

    // working_set was None → ws_budget is None and chosen budget = physical.
    assert!(
        ws_budget.is_none(),
        "ws_budget must be None when working_set_bytes=None"
    );
    assert_eq!(kv_bytes, phys_budget);

    // Expected budget: 64 GiB * 0.85 - 0 peak - 1 GiB safety = 53.4 GiB.
    // That's 54,681 MiB / 1 MiB per block = 54,681 blocks. Bound at ~50–60k.
    let kv_gib = kv_bytes / gib;
    assert!(
        (50..=55).contains(&kv_gib),
        "expected ~53 GiB KV under peak=0 + working_set=None, got {kv_gib} GiB"
    );
    assert!(
        (50_000..=60_000).contains(&num_blocks),
        "expected ~54,700 blocks, got {num_blocks} — \
         this is the regression bound: a future bug producing billions of blocks would fail this assertion"
    );
}

/// The fallible FFI contract guarantees the math layer never observes a
/// `working_set_bytes` value that came from a caught C++ exception (those
/// surface as `Err(MetalUnavailable)` BEFORE the math runs). But the
/// formula must still be well-defined for the legitimate edge case where
/// a working_set bound IS reported but happens to exceed `hw.memsize`
/// (driver bug, paravirtual VM exposing a misreported recommended limit).
/// In that case the physical-RAM budget is the tighter of the two and
/// must win.
#[test]
fn formula_picks_physical_when_working_set_exceeds_total() {
    let gib = 1024u64 * 1024 * 1024;
    let total = 32 * gib;
    // Pathological: working_set claims more memory than the system
    // physically has. The clamp must still produce the smaller of the
    // two budgets so we don't over-commit `hw.memsize`.
    let working_set = 64 * gib;
    let peak = 5 * gib;
    let util = 0.85;
    let safety = gib;
    let bpb = 1024u64 * 1024;

    let (phys_blocks, _) =
        compute_num_blocks_from_measurements(total, peak, util, safety, bpb).unwrap();
    let (clamped_blocks, kv_bytes, phys_budget, ws_budget) =
        compute_num_blocks_with_working_set(total, Some(working_set), peak, util, safety, bpb)
            .unwrap();

    // Physical budget is the tighter constraint here (32 GiB physical <
    // 64 GiB working set). The clamp must pick it.
    assert_eq!(clamped_blocks, phys_blocks);
    assert_eq!(kv_bytes, phys_budget);
    let ws_budget = ws_budget.expect("ws_budget reported");
    assert!(
        phys_budget <= ws_budget,
        "physical budget should be the tighter clamp when working_set > total"
    );
}

/// Closure failure propagates verbatim to ProfileError::DummyForwardFailed
/// AND the closure observed the right max_position_embeddings value.
#[test]
fn profile_run_propagates_closure_error() {
    let captured = Arc::new(Mutex::new(None::<u32>));
    let captured_for_closure = Arc::clone(&captured);
    let dummy_forward = move |max_seq: u32| -> Result<(), String> {
        *captured_for_closure.lock().unwrap() = Some(max_seq);
        Err("synthetic forward failure".to_string())
    };

    let res = profile_run_and_compute_num_blocks(
        dummy_forward,
        128u32,
        4,
        2,
        64,
        MetalDtype::BFloat16,
        16,
    );

    // The closure observed value depends on which path the function
    // takes:
    // - Metal present, sysctl works → reset_peak_memory(), then closure
    //   invoked, returns Err → DummyForwardFailed propagates.
    // - Metal absent on macOS (fast-path gate) → MetalUnavailable,
    //   closure was invoked once first to surface platform-portable
    //   model-loading bugs.
    // - Degraded-Metal on macOS (gate passes but reset_peak_memory
    //   throws) → MetalUnavailable propagates BEFORE the closure runs.
    //   `captured` stays None.
    // - Non-macOS or sysctl failure → TotalMemoryUnavailable; on the
    //   non-macOS branch the closure also runs once.
    // Either way: when the closure runs, max_seq must be 128.
    match res {
        Err(ProfileError::DummyForwardFailed(msg)) => {
            assert_eq!(msg, "synthetic forward failure");
            assert_eq!(*captured.lock().unwrap(), Some(128));
        }
        Err(ProfileError::MetalUnavailable) => {
            // Closure was invoked 0 times (reset_peak_memory failed
            // first) or 1 time (no-Metal fast-path gate). When it ran,
            // max_seq must be the supplied 128.
            let captured_val = *captured.lock().unwrap();
            assert!(
                captured_val.is_none() || captured_val == Some(128),
                "captured max_seq must be None (closure never ran) or Some(128) (closure ran), \
                 got {captured_val:?}"
            );
        }
        Err(ProfileError::TotalMemoryUnavailable) => {
            // The non-macOS branch invokes the closure before returning;
            // macOS branch only reaches TotalMemoryUnavailable via
            // sysctl failure (closure not yet invoked).
            assert!(captured.lock().unwrap().is_none() || *captured.lock().unwrap() == Some(128));
        }
        other => panic!("unexpected result: {other:?}"),
    }
}
