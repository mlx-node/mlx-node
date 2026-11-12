//! Load-time sizing of a block pool against a hard memory budget.
//!
//! The caller owns the budget: the hard Metal budget its memory plan derives
//! from the recommended working set (Splash `EngineMemoryPolicy::
//! hardBudgetBytes`). The sizer only reads MLX's live active bytes, subtracts
//! them, the caller's other reservations and this pool's restore staging,
//! and caps the requested pool at the whole blocks the rest holds.

use crate::metal::MetalDtype;

/// Errors the sizer can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    /// Metal is unavailable, or a memory probe's C++ call threw.
    MetalUnavailable,
    /// The configured layer/head/block parameters multiply to zero.
    InvalidShape(String),
    /// The budget left after active memory and the reservations holds no
    /// block.
    LoadTimeBudgetExhausted {
        hard_budget_bytes: u64,
        active_bytes: u64,
        reserved_bytes: u64,
        available_bytes: u64,
        bytes_per_block: u64,
    },
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileError::MetalUnavailable => {
                write!(f, "Metal device unavailable; cannot read MLX active memory")
            }
            ProfileError::InvalidShape(s) => write!(f, "invalid shape parameter: {s}"),
            ProfileError::LoadTimeBudgetExhausted {
                hard_budget_bytes,
                active_bytes,
                reserved_bytes,
                available_bytes,
                bytes_per_block,
            } => write!(
                f,
                "pool budget {available_bytes} B is smaller than per-block size \
                 {bytes_per_block} B; would yield 0 blocks (hard budget={hard_budget_bytes} B, \
                 MLX active={active_bytes} B, other pools/restore reserve={reserved_bytes} B)"
            ),
        }
    }
}

impl std::error::Error for ProfileError {}

/// Auditable result of the load-time pool cap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadTimePoolSizing {
    pub requested_blocks: u32,
    pub selected_blocks: u32,
    pub bytes_per_block: u64,
    pub selected_bytes: u64,
    pub hard_budget_bytes: u64,
    pub metal_active_bytes: u64,
}

/// Bytes-per-element for the KV cache storage dtype. Mirrors the kernel
/// instantiation list — see `crates/mlx-paged-attn/src/metal/state.rs`'s
/// `MetalDtype::size`.
pub fn dtype_size_for(dtype: MetalDtype) -> u32 {
    match dtype {
        // Non-FP8 caches are 2 bytes/element regardless of input
        // (kernels only instantiate (half, half) and (bfloat16_t, bfloat16_t)).
        MetalDtype::Float16 | MetalDtype::BFloat16 => 2,
        // FP8 E4M3 cache: 1 byte/element.
        MetalDtype::UChar => 1,
        // Float32 caches are not supported (no kernel instantiation). The
        // profile-run path treats them as 4 bytes for consistency, but
        // `LayerKVPool::new` rejects them at construction so we never
        // actually return a num_blocks for an F32 pool.
        MetalDtype::Float32 => 4,
    }
}

/// Compute per-block bytes following vLLM's KV pool layout:
/// `2 * num_layers * num_kv_heads * head_size * block_size * dtype_size`
/// (factor of 2 covers K and V).
///
/// Returns `Err(InvalidShape)` if any factor is zero.
pub fn bytes_per_block(
    num_layers: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    dtype: MetalDtype,
) -> Result<u64, ProfileError> {
    if num_layers == 0 || num_kv_heads == 0 || head_size == 0 || block_size == 0 {
        return Err(ProfileError::InvalidShape(format!(
            "num_layers={num_layers}, num_kv_heads={num_kv_heads}, \
             head_size={head_size}, block_size={block_size}; all must be > 0"
        )));
    }
    let dt = dtype_size_for(dtype) as u64;
    Ok(2u64 * num_layers as u64 * num_kv_heads as u64 * head_size as u64 * block_size as u64 * dt)
}

/// `min(requested, (hard - active - reserved) / bytes_per_block)` whole
/// blocks. `requested_blocks` is always an upper bound; the result is never
/// zero.
pub fn compute_load_time_pool_sizing(
    requested_blocks: u32,
    hard_budget_bytes: u64,
    metal_active_bytes: u64,
    bytes_per_block: u64,
    reserved_bytes: u64,
) -> Result<LoadTimePoolSizing, ProfileError> {
    if requested_blocks == 0 || bytes_per_block == 0 {
        return Err(ProfileError::InvalidShape(format!(
            "requested_blocks={requested_blocks}, bytes_per_block={bytes_per_block}; both must be > 0"
        )));
    }
    let safe_bytes = hard_budget_bytes
        .saturating_sub(metal_active_bytes)
        .saturating_sub(reserved_bytes);
    let selected_bytes = (u64::from(requested_blocks) * bytes_per_block).min(safe_bytes);
    let selected_blocks = selected_bytes / bytes_per_block;
    if selected_blocks == 0 {
        return Err(ProfileError::LoadTimeBudgetExhausted {
            hard_budget_bytes,
            active_bytes: metal_active_bytes,
            reserved_bytes,
            available_bytes: selected_bytes,
            bytes_per_block,
        });
    }
    let selected_blocks = u32::try_from(selected_blocks).unwrap_or(u32::MAX);
    Ok(LoadTimePoolSizing {
        requested_blocks,
        selected_blocks,
        bytes_per_block,
        selected_bytes: u64::from(selected_blocks) * bytes_per_block,
        hard_budget_bytes,
        metal_active_bytes,
    })
}

/// [`compute_load_time_pool_sizing`] on MLX's live active bytes, with this
/// pool's restore staging (direct Metal allocations MLX does not count)
/// added to `extra_reserved_bytes` (other pools' private buffers). Callers
/// must fail closed on an error.
#[allow(clippy::too_many_arguments)]
pub fn load_time_pool_sizing(
    requested_blocks: u32,
    num_layers: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    dtype: MetalDtype,
    hard_budget_bytes: u64,
    extra_reserved_bytes: u64,
) -> Result<LoadTimePoolSizing, ProfileError> {
    let bpb = bytes_per_block(num_layers, num_kv_heads, head_size, block_size, dtype)?;
    let reserved = extra_reserved_bytes.saturating_add(crate::layer_kv_pool::RESTORE_STAGING_BYTES);

    #[cfg(target_os = "macos")]
    {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Err(ProfileError::MetalUnavailable);
        }
        let mut active = 0u64;
        if unsafe { mlx_sys::mlx_get_active_memory(&mut active) } != 0 {
            return Err(ProfileError::MetalUnavailable);
        }
        compute_load_time_pool_sizing(requested_blocks, hard_budget_bytes, active, bpb, reserved)
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (requested_blocks, hard_budget_bytes, bpb, reserved);
        Err(ProfileError::MetalUnavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_size_table() {
        assert_eq!(dtype_size_for(MetalDtype::Float16), 2);
        assert_eq!(dtype_size_for(MetalDtype::BFloat16), 2);
        assert_eq!(dtype_size_for(MetalDtype::UChar), 1);
        assert_eq!(dtype_size_for(MetalDtype::Float32), 4);
    }

    #[test]
    fn bytes_per_block_formula() {
        // num_layers=32, num_kv_heads=8, head_size=64, block_size=16,
        // bf16 -> 2 bytes/element. Per-block bytes = 2 * 32 * 8 * 64 * 16
        // * 2 = 1,048,576 = 1 MiB.
        let bpb = bytes_per_block(32, 8, 64, 16, MetalDtype::BFloat16).unwrap();
        assert_eq!(bpb, 2 * 32 * 8 * 64 * 16 * 2);
        assert_eq!(bpb, 1024 * 1024);
    }

    #[test]
    fn bytes_per_block_rejects_zero() {
        assert!(matches!(
            bytes_per_block(0, 8, 64, 16, MetalDtype::BFloat16),
            Err(ProfileError::InvalidShape(_))
        ));
        assert!(matches!(
            bytes_per_block(32, 0, 64, 16, MetalDtype::BFloat16),
            Err(ProfileError::InvalidShape(_))
        ));
        assert!(matches!(
            bytes_per_block(32, 8, 0, 16, MetalDtype::BFloat16),
            Err(ProfileError::InvalidShape(_))
        ));
        assert!(matches!(
            bytes_per_block(32, 8, 64, 0, MetalDtype::BFloat16),
            Err(ProfileError::InvalidShape(_))
        ));
    }

    /// Mutation: subtract another 1 GiB in `compute_load_time_pool_sizing`
    /// → the 4,096-block case selects 3,072 blocks.
    #[test]
    fn load_time_sizing_on_the_hard_budget() {
        let gib = 1u64 << 30;
        let mib = 1u64 << 20;
        // hard 20 GiB - active 14 GiB - reserved 2 GiB = 4 GiB = 4,096 blocks.
        let sizing =
            compute_load_time_pool_sizing(u32::MAX, 20 * gib, 14 * gib, mib, 2 * gib).unwrap();
        assert_eq!(sizing.selected_blocks, 4_096);
        assert_eq!(sizing.selected_bytes, 4 * gib);
        assert_eq!(sizing.hard_budget_bytes, 20 * gib);
        // The request stays the maximum.
        let capped =
            compute_load_time_pool_sizing(2_048, 20 * gib, 14 * gib, mib, 2 * gib).unwrap();
        assert_eq!(capped.selected_blocks, 2_048);
        // Whole blocks, rounded down.
        let odd =
            compute_load_time_pool_sizing(u32::MAX, 20 * gib, 14 * gib, 3 * mib, 2 * gib).unwrap();
        assert_eq!(odd.selected_blocks, 1_365);
        assert_eq!(odd.selected_bytes, 1_365 * 3 * mib);
    }

    #[test]
    fn load_time_sizing_rejects_less_than_one_block() {
        let gib = 1u64 << 30;
        let mib = 1u64 << 20;
        let error = compute_load_time_pool_sizing(8_192, 20 * gib, 19 * gib, 2 * gib, 64 * mib)
            .unwrap_err();
        assert_eq!(
            error,
            ProfileError::LoadTimeBudgetExhausted {
                hard_budget_bytes: 20 * gib,
                active_bytes: 19 * gib,
                reserved_bytes: 64 * mib,
                available_bytes: gib - 64 * mib,
                bytes_per_block: 2 * gib,
            }
        );
        let message = error.to_string();
        assert!(message.contains(&format!("hard budget={} B", 20 * gib)));
        assert!(message.contains(&format!("other pools/restore reserve={} B", 64 * mib)));
    }
}
