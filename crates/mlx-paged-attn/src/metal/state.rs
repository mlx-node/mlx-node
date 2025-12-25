//! Metal state management - device, library, pipeline states
//!
//! FORKED KERNEL APPROACH:
//! The paged attention kernels have been forked from the HuggingFace kernels-community
//! implementation to use template parameters instead of Metal function constants.
//! This allows us to call kernels by name without needing to set function constants,
//! which MLX's metal_kernel() API doesn't support.
//!
//! Kernel naming convention:
//! - reshape_and_cache: reshape_and_cache_kv_{type}_cache_{cache_type}[_fp8]
//! - paged_attention V1: paged_attention_{type}_cache_{cache_type}_hs{head}_bs{block}_nt256_nsl32_ps0[_alibi]
//! - paged_attention V2: paged_attention_{type}_cache_{cache_type}_hs{head}_bs{block}_nt256_nsl32_ps512[_alibi]
//! - paged_attention V2 reduce: paged_attention_v2_reduce_{type}_hs{head}_nt256_nsl32_ps512

use metal::{ComputePipelineState, Device, Library};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Embedded metallib bytes (compiled at build time)
const METALLIB_BYTES: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/paged_attn.metallib"));

/// Global Metal state singleton
static METAL_STATE: OnceLock<Result<MetalState, String>> = OnceLock::new();

/// Cached Metal resources for paged attention
pub struct MetalState {
    /// Metal device
    pub device: Device,
    /// Compiled Metal library
    pub library: Library,
    /// Pipeline states keyed by kernel name (uses RwLock for interior mutability)
    pipelines: RwLock<HashMap<String, ComputePipelineState>>,
}

impl MetalState {
    /// Get the global Metal state singleton
    pub fn get() -> Result<&'static MetalState, String> {
        METAL_STATE
            .get_or_init(Self::init)
            .as_ref()
            .map_err(|e| e.clone())
    }

    /// Initialize Metal state
    fn init() -> Result<MetalState, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        // Write metallib to temp file and load
        // (Metal requires loading from file path, not memory)
        let temp_path = std::env::temp_dir().join("mlx_paged_attn.metallib");
        std::fs::write(&temp_path, METALLIB_BYTES)
            .map_err(|e| format!("Failed to write metallib to temp: {}", e))?;

        let library = device
            .new_library_with_file(&temp_path)
            .map_err(|e| format!("Failed to load metallib: {}", e))?;

        Ok(MetalState {
            device,
            library,
            pipelines: RwLock::new(HashMap::new()),
        })
    }

    /// Get or create a compute pipeline for a kernel
    ///
    /// FORKED: Kernels are now template-specialized, so we just load by name
    /// without needing to set function constants. The kernel name encodes
    /// all the options (e.g., _fp8, _alibi suffixes).
    ///
    /// Pipelines are cached for performance - subsequent calls with the same
    /// kernel name return the cached pipeline.
    pub fn get_pipeline(&self, kernel_name: &str) -> Result<ComputePipelineState, String> {
        // Check cache first (read lock)
        {
            let cache = self
                .pipelines
                .read()
                .map_err(|e| format!("Lock poisoned: {}", e))?;
            if let Some(pipeline) = cache.get(kernel_name) {
                return Ok(pipeline.clone());
            }
        }

        // Cache miss - create new pipeline (write lock)
        let mut cache = self
            .pipelines
            .write()
            .map_err(|e| format!("Lock poisoned: {}", e))?;

        // Double-check after acquiring write lock (another thread may have inserted)
        if let Some(pipeline) = cache.get(kernel_name) {
            return Ok(pipeline.clone());
        }

        // FORKED: No function constants needed - kernels are template-specialized
        let function = self
            .library
            .get_function(kernel_name, None)
            .map_err(|e| format!("Kernel '{}' not found: {}", kernel_name, e))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create pipeline for '{}': {}", kernel_name, e))?;

        // Insert into cache
        cache.insert(kernel_name.to_string(), pipeline.clone());

        Ok(pipeline)
    }

    /// Get the reshape_and_cache kernel name for a dtype
    ///
    /// # Arguments
    /// * `cache_dtype` - Data type for cache storage (UChar for FP8, Float16 otherwise)
    /// * `use_fp8` - Whether to use FP8 scaling
    ///
    /// Note: Input KV tensors are always float16 from the model.
    pub fn reshape_and_cache_kernel_name(cache_dtype: MetalDtype, use_fp8: bool) -> String {
        // Input is always float16 (from model), cache dtype varies
        let input_type = MetalDtype::Float16.type_string();
        let cache_type = cache_dtype.type_string();
        let suffix = if use_fp8 { "_fp8" } else { "" };
        format!(
            "reshape_and_cache_kv_{}_cache_{}{}",
            input_type, cache_type, suffix
        )
    }

    /// Get the paged_attention V1 kernel name (no partitioning)
    ///
    /// # Arguments
    /// * `cache_dtype` - Data type for cache (UChar for FP8, Float16 otherwise)
    /// * `head_size` - Head dimension (64 or 128)
    /// * `block_size` - Block size (16 or 32)
    /// * `use_alibi` - Whether to use ALiBi positional encoding
    ///
    /// Note: Input queries are always float16, output is always float16.
    pub fn paged_attention_v1_kernel_name(
        cache_dtype: MetalDtype,
        head_size: u32,
        block_size: u32,
        use_alibi: bool,
    ) -> String {
        // Input/output are always float16, cache dtype varies
        let io_type = MetalDtype::Float16.type_string();
        let cache_type = cache_dtype.type_string();
        let suffix = if use_alibi { "_alibi" } else { "" };
        format!(
            "paged_attention_{}_cache_{}_hs{}_bs{}_nt256_nsl32_ps0{}",
            io_type, cache_type, head_size, block_size, suffix
        )
    }

    /// Get the paged_attention V2 kernel name (with partitioning)
    ///
    /// # Arguments
    /// * `cache_dtype` - Data type for cache (UChar for FP8, Float16 otherwise)
    /// * `head_size` - Head dimension (64 or 128)
    /// * `block_size` - Block size (16 or 32)
    /// * `use_alibi` - Whether to use ALiBi positional encoding
    ///
    /// Note: Input queries are always float16, output is always float16.
    pub fn paged_attention_v2_kernel_name(
        cache_dtype: MetalDtype,
        head_size: u32,
        block_size: u32,
        use_alibi: bool,
    ) -> String {
        // Input/output are always float16, cache dtype varies
        let io_type = MetalDtype::Float16.type_string();
        let cache_type = cache_dtype.type_string();
        let suffix = if use_alibi { "_alibi" } else { "" };
        format!(
            "paged_attention_{}_cache_{}_hs{}_bs{}_nt256_nsl32_ps512{}",
            io_type, cache_type, head_size, block_size, suffix
        )
    }

    /// Get the paged_attention V2 reduce kernel name
    ///
    /// Note: Reduce kernel always uses float16 since partitioned outputs are float16.
    pub fn paged_attention_v2_reduce_kernel_name(
        _cache_dtype: MetalDtype,
        head_size: u32,
    ) -> String {
        // Reduce kernel always works with float16 outputs from partitions
        let io_type = MetalDtype::Float16.type_string();
        format!(
            "paged_attention_v2_reduce_{}_hs{}_nt256_nsl32_ps512",
            io_type, head_size
        )
    }
}

/// Data types supported by Metal kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalDtype {
    Float32,
    Float16,
    BFloat16,
    /// FP8 E4M3 format (1 byte per element)
    UChar,
}

impl MetalDtype {
    /// Get the Metal kernel type string
    pub fn type_string(&self) -> &'static str {
        match self {
            MetalDtype::Float32 => "float",
            MetalDtype::Float16 => "half",
            MetalDtype::BFloat16 => "bfloat16_t",
            MetalDtype::UChar => "uchar",
        }
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            MetalDtype::Float32 => 4,
            MetalDtype::Float16 | MetalDtype::BFloat16 => 2,
            MetalDtype::UChar => 1,
        }
    }

    /// Check if this is an FP8 type
    pub fn is_fp8(&self) -> bool {
        matches!(self, MetalDtype::UChar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_state_init() {
        let state = MetalState::get();
        assert!(
            state.is_ok(),
            "Failed to init Metal state: {:?}",
            state.err()
        );
    }

    #[test]
    fn test_kernel_names() {
        // FORKED: Updated tests for new API with use_fp8/use_alibi parameters
        // Float16 cache mode
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(MetalDtype::Float16, false),
            "reshape_and_cache_kv_half_cache_half"
        );
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(MetalDtype::Float16, true),
            "reshape_and_cache_kv_half_cache_half_fp8"
        );
        assert_eq!(
            MetalState::paged_attention_v1_kernel_name(MetalDtype::Float16, 128, 16, false),
            "paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps0"
        );
        assert_eq!(
            MetalState::paged_attention_v1_kernel_name(MetalDtype::Float16, 128, 16, true),
            "paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps0_alibi"
        );
        assert_eq!(
            MetalState::paged_attention_v2_kernel_name(MetalDtype::Float16, 128, 16, false),
            "paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps512"
        );
        assert_eq!(
            MetalState::paged_attention_v2_kernel_name(MetalDtype::Float16, 128, 16, true),
            "paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps512_alibi"
        );

        // FP8 cache mode (UChar cache, but input/output still float16)
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(MetalDtype::UChar, true),
            "reshape_and_cache_kv_half_cache_uchar_fp8"
        );
        assert_eq!(
            MetalState::paged_attention_v1_kernel_name(MetalDtype::UChar, 128, 16, false),
            "paged_attention_half_cache_uchar_hs128_bs16_nt256_nsl32_ps0"
        );
        assert_eq!(
            MetalState::paged_attention_v2_kernel_name(MetalDtype::UChar, 128, 16, false),
            "paged_attention_half_cache_uchar_hs128_bs16_nt256_nsl32_ps512"
        );
        // Reduce kernel always uses float16 regardless of cache dtype
        assert_eq!(
            MetalState::paged_attention_v2_reduce_kernel_name(MetalDtype::UChar, 128),
            "paged_attention_v2_reduce_half_hs128_nt256_nsl32_ps512"
        );
    }

    #[test]
    fn test_get_reshape_and_cache_pipeline() {
        let state = MetalState::get().expect("Failed to init Metal state");
        let kernel_name = MetalState::reshape_and_cache_kernel_name(MetalDtype::Float16, false);
        let pipeline = state.get_pipeline(&kernel_name);
        assert!(
            pipeline.is_ok(),
            "Failed to get reshape_and_cache pipeline: {:?}",
            pipeline.err()
        );
    }

    #[test]
    fn test_get_paged_attention_pipeline() {
        let state = MetalState::get().expect("Failed to init Metal state");
        // Test V1 kernel for Qwen3 config: head_size=128, block_size=16
        let kernel_name =
            MetalState::paged_attention_v1_kernel_name(MetalDtype::Float16, 128, 16, false);
        let pipeline = state.get_pipeline(&kernel_name);
        assert!(
            pipeline.is_ok(),
            "Failed to get paged_attention V1 pipeline: {:?}",
            pipeline.err()
        );
    }
}
