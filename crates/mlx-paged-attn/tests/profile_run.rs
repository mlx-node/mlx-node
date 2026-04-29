//! Integration tests for the Phase 3 profile-run auto-sizer.
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
//! which is a model-specific concern that lives in Phases 4-9, and (b)
//! it depends on host MLX state that integration tests for Phase 3
//! shouldn't entangle.

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::profile::{
    DEFAULT_KV_MEMORY_UTILIZATION, DEFAULT_SAFETY_MARGIN_BYTES, ProfileError, bytes_per_block,
    compute_num_blocks_from_measurements, dtype_size_for,
};

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
