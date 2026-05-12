//! copy_blocks Metal kernel dispatch
//!
//! This kernel copies blocks for copy-on-write semantics (used in beam search).

use super::state::{MetalDtype, MetalState};

/// Parameters for copy_blocks kernel
#[derive(Debug, Clone)]
pub struct CopyBlocksParams {
    /// Number of block pairs to copy
    pub num_pairs: u32,
    /// Number of elements per block (num_kv_heads * head_size * block_size)
    pub numel_per_block: u32,
}

impl CopyBlocksParams {
    /// Create params from cache config
    pub fn from_config(num_pairs: u32, num_kv_heads: u32, head_size: u32, block_size: u32) -> Self {
        Self {
            num_pairs,
            numel_per_block: num_kv_heads * head_size * block_size,
        }
    }
}

impl MetalState {
    /// Get the copy_blocks kernel name for a dtype
    ///
    /// # Arguments
    /// * `dtype` - Data type for cache elements
    pub fn copy_blocks_kernel_name(dtype: MetalDtype) -> String {
        let type_str = dtype.type_string();
        format!("copy_blocks_{}", type_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_blocks_params() {
        let params = CopyBlocksParams::from_config(
            2,   // num_pairs
            4,   // num_kv_heads
            128, // head_size
            16,  // block_size
        );
        assert_eq!(params.num_pairs, 2);
        assert_eq!(params.numel_per_block, 4 * 128 * 16);
    }

    #[test]
    fn test_copy_blocks_kernel_name() {
        assert_eq!(
            MetalState::copy_blocks_kernel_name(MetalDtype::Float16),
            "copy_blocks_half"
        );
        assert_eq!(
            MetalState::copy_blocks_kernel_name(MetalDtype::Float32),
            "copy_blocks_float"
        );
        assert_eq!(
            MetalState::copy_blocks_kernel_name(MetalDtype::BFloat16),
            "copy_blocks_bfloat16_t"
        );
    }

    #[test]
    fn test_get_copy_blocks_pipeline() {
        // Graceful skip on no-Metal hosts (CI VMs, sandboxes). Mirrors the
        // pattern in `LayerKVPool::test_new_allocates_per_layer_buffers`:
        // probe `MetalState::get()` first and bail with a `skipping`
        // notice if the device isn't available, rather than panicking.
        let state = match MetalState::get() {
            Ok(s) => s,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_get_copy_blocks_pipeline: {e}");
                return;
            }
            Err(e) => panic!("unexpected MetalState::get failure: {e}"),
        };
        let kernel_name = MetalState::copy_blocks_kernel_name(MetalDtype::Float16);
        let pipeline = state.get_pipeline(&kernel_name);
        assert!(
            pipeline.is_ok(),
            "Failed to get copy_blocks pipeline: {:?}",
            pipeline.err()
        );
    }
}
