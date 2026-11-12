//! Integration tests for the load-time pool sizer on its public API
//! (`crates/mlx-paged-attn/src/profile.rs`): the pool is what a hard budget
//! holds after MLX active memory and the caller's reservations.

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::profile::{
    ProfileError, bytes_per_block, compute_load_time_pool_sizing, dtype_size_for,
};

const GIB: u64 = 1 << 30;

/// Qwen3-class pool (28 layers, 8 KV heads, head 128, block 16, bf16:
/// 1.75 MiB a block) on a 64 GiB Mac's hard budget (48 GiB working set less
/// 15% and 1 GiB = 39.8 GiB) with 10 GiB of weights: 29.8 GiB / 1.75 MiB.
#[test]
fn qwen3_realistic_sizing() {
    let bpb = bytes_per_block(28, 8, 128, 16, MetalDtype::BFloat16).unwrap();
    assert_eq!(bpb, 1_835_008);
    let working_set = 48 * GIB;
    let hard = working_set / 100 * 85 - GIB;
    let sizing = compute_load_time_pool_sizing(u32::MAX, hard, 10 * GIB, bpb, 0).unwrap();
    assert_eq!(u64::from(sizing.selected_blocks), (hard - 10 * GIB) / bpb);
    assert_eq!(sizing.selected_blocks, 17_437);
}

/// FP8 cache: half the per-block size of BF16, twice the blocks on the
/// same budget (the budget is a whole number of BF16 blocks).
#[test]
fn fp8_doubles_block_count() {
    let bpb_bf16 = bytes_per_block(28, 8, 128, 16, MetalDtype::BFloat16).unwrap();
    let bpb_fp8 = bytes_per_block(28, 8, 128, 16, MetalDtype::UChar).unwrap();
    assert_eq!(bpb_bf16, 2 * bpb_fp8);
    let hard = 10 * GIB + 1_000 * bpb_bf16;
    let bf16 = compute_load_time_pool_sizing(u32::MAX, hard, 10 * GIB, bpb_bf16, 0).unwrap();
    let fp8 = compute_load_time_pool_sizing(u32::MAX, hard, 10 * GIB, bpb_fp8, 0).unwrap();
    assert_eq!(bf16.selected_blocks, 1_000);
    assert_eq!(fp8.selected_blocks, 2_000);
}

/// Active memory above the hard budget leaves no block.
#[test]
fn rejects_active_above_the_budget() {
    let res = compute_load_time_pool_sizing(16, 20 * GIB, 22 * GIB, 1024, 0);
    assert!(
        matches!(
            res,
            Err(ProfileError::LoadTimeBudgetExhausted {
                available_bytes: 0,
                ..
            })
        ),
        "{res:?}"
    );
}

/// A block larger than the budget left is refused; the sizer never returns
/// zero blocks.
#[test]
fn rejects_zero_blocks() {
    let res = compute_load_time_pool_sizing(16, 4 * GIB, GIB, 4 * GIB, 0);
    assert!(matches!(
        res,
        Err(ProfileError::LoadTimeBudgetExhausted { .. })
    ));
}

#[test]
fn dtype_sizes() {
    assert_eq!(dtype_size_for(MetalDtype::BFloat16), 2);
    assert_eq!(dtype_size_for(MetalDtype::Float16), 2);
    assert_eq!(dtype_size_for(MetalDtype::UChar), 1);
}

/// Zero shape parameters reject deterministically.
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
    assert!(matches!(
        compute_load_time_pool_sizing(0, 4 * GIB, 0, 1024, 0),
        Err(ProfileError::InvalidShape(_))
    ));
}
