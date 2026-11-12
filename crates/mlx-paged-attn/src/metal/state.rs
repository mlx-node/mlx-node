//! Metal state management - device, library, pipeline states
//!
//! FORKED KERNEL APPROACH:
//! The paged attention kernels have been forked from the HuggingFace kernels-community
//! implementation to use template parameters instead of Metal function constants.
//! This allows us to call kernels by name without needing to set function constants,
//! which MLX's metal_kernel() API doesn't support.
//!
//! Kernel naming convention:
//! - reshape_and_cache: reshape_and_cache_kv_{type}_cache_{cache_type}
//! - paged_attention V1: paged_attention_{type}_cache_{cache_type}_hs{head}_bs{block}_nt256_nsl32_ps0[_alibi]
//! - paged_attention V2: paged_attention_{type}_cache_{cache_type}_hs{head}_bs{block}_nt256_nsl32_ps512[_alibi]
//! - paged_attention V2 reduce: paged_attention_v2_reduce_{type}_hs{head}_nt256_nsl32_ps512

use metal::{ComputePipelineState, Device, Library};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Embedded metallib bytes (compiled at build time)
#[cfg(mlx_node_metal_enabled)]
const METALLIB_BYTES: &[u8] = include_bytes!(env!("PAGED_ATTN_METALLIB"));

/// Global Metal state singleton
static METAL_STATE: OnceLock<Result<MetalState, String>> = OnceLock::new();

/// Cached Metal resources for paged attention
pub struct MetalState {
    /// Metal device
    pub device: Device,
    /// Compiled Metal library
    pub library: Library,
    /// Reusable command queue (avoids per-dispatch allocation)
    pub command_queue: metal::CommandQueue,
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
        #[cfg(not(mlx_node_metal_enabled))]
        return Err("No Metal device found: Metal backend disabled at build time".to_string());

        #[cfg(mlx_node_metal_enabled)]
        {
            let device = Device::system_default().ok_or("No Metal device found")?;

            // Write metallib to temp file and load
            // (Metal requires loading from file path, not memory)
            let temp_path = std::env::temp_dir().join("mlx_paged_attn.metallib");
            std::fs::write(&temp_path, METALLIB_BYTES)
                .map_err(|e| format!("Failed to write metallib to temp: {}", e))?;

            let library = device
                .new_library_with_file(&temp_path)
                .map_err(|e| format!("Failed to load metallib: {}", e))?;

            let command_queue = device
                .try_new_command_queue()
                .ok_or_else(|| "Metal device returned no command queue".to_string())?;

            Ok(MetalState {
                device,
                library,
                command_queue,
                pipelines: RwLock::new(HashMap::new()),
            })
        }
    }

    /// Get or create a compute pipeline for a kernel
    ///
    /// FORKED: Kernels are now template-specialized, so we just load by name
    /// without needing to set function constants. The kernel name encodes
    /// all the options (e.g. the _alibi suffix).
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

    /// Get the `reshape_and_cache` kernel name for an `(input, cache)` dtype
    /// pair. The metal source instantiates `(float, float)`, `(half, half)`
    /// and `(bfloat16_t, bfloat16_t)`; any other pair fails the lookup.
    pub fn reshape_and_cache_kernel_name(
        input_dtype: MetalDtype,
        cache_dtype: MetalDtype,
    ) -> String {
        format!(
            "reshape_and_cache_kv_{}_cache_{}",
            input_dtype.type_string(),
            cache_dtype.type_string()
        )
    }

    /// Get the paged_attention V1 kernel name (no partitioning).
    ///
    /// The metal source instantiates these `(io_type, cache_type)` pairs (see
    /// `crates/mlx-paged-attn/metal/attention/paged_attention.metal`):
    /// `(float, float)`, `(half, half)`, `(bfloat16_t, bfloat16_t)` and the
    /// int8 `(bfloat16_t, char)`; the metallib lookup fails loudly on any
    /// pair it never instantiated.
    ///
    /// # Arguments
    /// * `io_dtype` - Data type for queries + output (`cache_dtype.io_dtype()`)
    /// * `cache_dtype` - Data type for cache storage
    /// * `head_size` - Head dimension (32, 64, 80, 96, 112, 120, 128, 192, 256)
    /// * `block_size` - Block size (8, 16, 32)
    /// * `use_alibi` - Whether to use ALiBi positional encoding
    pub fn paged_attention_v1_kernel_name(
        io_dtype: MetalDtype,
        cache_dtype: MetalDtype,
        head_size: u32,
        block_size: u32,
        use_alibi: bool,
    ) -> String {
        let io_type = io_dtype.type_string();
        let cache_type = cache_dtype.type_string();
        let suffix = if use_alibi { "_alibi" } else { "" };
        format!(
            "paged_attention_{}_cache_{}_hs{}_bs{}_nt256_nsl32_ps0{}",
            io_type, cache_type, head_size, block_size, suffix
        )
    }

    /// Get the paged_attention V2 kernel name (with partitioning).
    ///
    /// See [`Self::paged_attention_v1_kernel_name`] for the instantiation
    /// list. Same `(io_type, cache_type)` pairs are instantiated for V2.
    ///
    /// # Arguments
    /// * `io_dtype` - Data type for queries + partition output (see V1 docs)
    /// * `cache_dtype` - Data type for cache storage (see V1 docs)
    /// * `head_size` - Head dimension
    /// * `block_size` - Block size
    /// * `use_alibi` - Whether to use ALiBi positional encoding
    pub fn paged_attention_v2_kernel_name(
        io_dtype: MetalDtype,
        cache_dtype: MetalDtype,
        head_size: u32,
        block_size: u32,
        use_alibi: bool,
    ) -> String {
        let io_type = io_dtype.type_string();
        let cache_type = cache_dtype.type_string();
        let suffix = if use_alibi { "_alibi" } else { "" };
        format!(
            "paged_attention_{}_cache_{}_hs{}_bs{}_nt256_nsl32_ps512{}",
            io_type, cache_type, head_size, block_size, suffix
        )
    }

    /// Get the paged_attention V2 reduce kernel name.
    ///
    /// The reduce kernel is templated on a single io type (the partitioned
    /// outputs from phase 1 share the same dtype as the final output). The
    /// metal source instantiates the reduce kernel for `float`, `half`, and
    /// `bfloat16_t` — pick whichever matches the V2 dispatch's `io_dtype`.
    pub fn paged_attention_v2_reduce_kernel_name(io_dtype: MetalDtype, head_size: u32) -> String {
        let io_type = io_dtype.type_string();
        format!(
            "paged_attention_v2_reduce_{}_hs{}_nt256_nsl32_ps512",
            io_type, head_size
        )
    }

    /// Concrete long-context grouped-GQA stage-1 specialization used by
    /// Qwen3.5/3.6 dense and MoE decode (BF16, D256, block size 16,
    /// 24Q/4KV or 16Q/2KV).
    pub fn paged_attention_grouped_qwen35_kernel_name() -> &'static str {
        "paged_attention_grouped_bfloat16_hs256_bs16_striped"
    }

    /// Dedicated MLX-style second pass for the grouped strided specialization.
    pub fn paged_attention_grouped_qwen35_reduce_kernel_name() -> &'static str {
        "paged_attention_grouped_bfloat16_hs256_striped_reduce"
    }

    /// Geometry-based grouped paged kernel for BF16, D512, block size 16, and
    /// the supported GQA layouts. The production specialization reads K/V
    /// directly from device memory without threadgroup staging.
    pub fn paged_attention_grouped_d512_kernel_name() -> &'static str {
        "paged_attention_grouped_bfloat16_hs512_bs16_striped"
    }

    /// Cooperative-staging D512 variant retained for manual parity,
    /// performance, and rollback comparisons. Production capability probing
    /// does not select this pipeline.
    pub fn paged_attention_grouped_d512_staged_kernel_name() -> &'static str {
        "paged_attention_grouped_bfloat16_hs512_bs16_striped_staged"
    }

    /// D512 reducer paired with `paged_attention_grouped_d512_kernel_name`.
    pub fn paged_attention_grouped_d512_reduce_kernel_name() -> &'static str {
        "paged_attention_grouped_bfloat16_hs512_striped_reduce"
    }

    /// Varlen counterparts to the V1/V2/reduce kernel name helpers above.
    /// The naming convention mirrors the single-row helpers
    /// 1:1 except for the `varlen` infix so the metallib lookup is
    /// unambiguous; this also means a missing-kernel typo here fails
    /// loudly at pipeline-build time rather than silently selecting the
    /// single-row kernel.
    pub fn paged_attention_varlen_v1_kernel_name(
        io_dtype: MetalDtype,
        cache_dtype: MetalDtype,
        head_size: u32,
        block_size: u32,
        use_alibi: bool,
    ) -> String {
        let io_type = io_dtype.type_string();
        let cache_type = cache_dtype.type_string();
        let suffix = if use_alibi { "_alibi" } else { "" };
        format!(
            "paged_attention_varlen_{}_cache_{}_hs{}_bs{}_nt256_nsl32_ps0{}",
            io_type, cache_type, head_size, block_size, suffix
        )
    }

    pub fn paged_attention_varlen_v2_kernel_name(
        io_dtype: MetalDtype,
        cache_dtype: MetalDtype,
        head_size: u32,
        block_size: u32,
        use_alibi: bool,
    ) -> String {
        let io_type = io_dtype.type_string();
        let cache_type = cache_dtype.type_string();
        let suffix = if use_alibi { "_alibi" } else { "" };
        format!(
            "paged_attention_varlen_{}_cache_{}_hs{}_bs{}_nt256_nsl32_ps512{}",
            io_type, cache_type, head_size, block_size, suffix
        )
    }

    pub fn paged_attention_varlen_v2_reduce_kernel_name(
        io_dtype: MetalDtype,
        head_size: u32,
    ) -> String {
        let io_type = io_dtype.type_string();
        format!(
            "paged_attention_varlen_v2_reduce_{}_hs{}_nt256_nsl32_ps512",
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
    /// Splash `kv::Format::Int8`: symmetric int8 with one f32 scale per
    /// token, head and side (`splash_store_kv_row`).
    Int8,
}

impl MetalDtype {
    /// Get the Metal kernel type string
    pub fn type_string(&self) -> &'static str {
        match self {
            MetalDtype::Float32 => "float",
            MetalDtype::Float16 => "half",
            MetalDtype::BFloat16 => "bfloat16_t",
            MetalDtype::Int8 => "char",
        }
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            MetalDtype::Float32 => 4,
            MetalDtype::Float16 | MetalDtype::BFloat16 => 2,
            MetalDtype::Int8 => 1,
        }
    }

    /// Dtype of the rows written into and read out of a cache of this
    /// dtype. Splash io is always bf16 for an int8 cache.
    pub fn io_dtype(&self) -> MetalDtype {
        match self {
            MetalDtype::Int8 => MetalDtype::BFloat16,
            other => *other,
        }
    }

    /// Scale bytes per token, head and side (Splash `kv::Layout`
    /// `bytesPerLayerPage`: an f32 per row for int8, none otherwise).
    pub fn scale_bytes(&self) -> usize {
        match self {
            MetalDtype::Int8 => 4,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_state_init() {
        match MetalState::get() {
            Ok(_) => {}
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_metal_state_init: {e}");
            }
            Err(e) => panic!("unexpected MetalState::get failure: {e}"),
        }
    }

    #[test]
    fn test_kernel_names() {
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(MetalDtype::Float16, MetalDtype::Float16),
            "reshape_and_cache_kv_half_cache_half"
        );
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(MetalDtype::BFloat16, MetalDtype::BFloat16),
            "reshape_and_cache_kv_bfloat16_t_cache_bfloat16_t"
        );
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(MetalDtype::Float32, MetalDtype::Float32),
            "reshape_and_cache_kv_float_cache_float"
        );

        // Paged attention V1/V2 — io_dtype split from cache_dtype.
        // (half, half) — Float16 model, Float16 cache (existing default).
        assert_eq!(
            MetalState::paged_attention_v1_kernel_name(
                MetalDtype::Float16,
                MetalDtype::Float16,
                128,
                16,
                false,
            ),
            "paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps0"
        );
        assert_eq!(
            MetalState::paged_attention_v1_kernel_name(
                MetalDtype::Float16,
                MetalDtype::Float16,
                128,
                16,
                true,
            ),
            "paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps0_alibi"
        );
        assert_eq!(
            MetalState::paged_attention_v2_kernel_name(
                MetalDtype::Float16,
                MetalDtype::Float16,
                128,
                16,
                false,
            ),
            "paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps512"
        );
        assert_eq!(
            MetalState::paged_attention_v2_kernel_name(
                MetalDtype::Float16,
                MetalDtype::Float16,
                128,
                16,
                true,
            ),
            "paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps512_alibi"
        );
        // (bfloat16_t, bfloat16_t) — BF16 model + BF16 cache (Qwen3.5
        // production path; this is the kernel name `gather_attention` MUST
        // dispatch when the cache_dtype field is BFloat16).
        assert_eq!(
            MetalState::paged_attention_v1_kernel_name(
                MetalDtype::BFloat16,
                MetalDtype::BFloat16,
                128,
                16,
                false,
            ),
            "paged_attention_bfloat16_t_cache_bfloat16_t_hs128_bs16_nt256_nsl32_ps0"
        );
        assert_eq!(
            MetalState::paged_attention_v2_kernel_name(
                MetalDtype::BFloat16,
                MetalDtype::BFloat16,
                128,
                16,
                false,
            ),
            "paged_attention_bfloat16_t_cache_bfloat16_t_hs128_bs16_nt256_nsl32_ps512"
        );
        // Reduce kernel uses io_dtype only (the partition outputs are the
        // io type; cache_dtype is irrelevant for the reduce phase).
        assert_eq!(
            MetalState::paged_attention_v2_reduce_kernel_name(MetalDtype::Float16, 128),
            "paged_attention_v2_reduce_half_hs128_nt256_nsl32_ps512"
        );
        assert_eq!(
            MetalState::paged_attention_v2_reduce_kernel_name(MetalDtype::BFloat16, 128),
            "paged_attention_v2_reduce_bfloat16_t_hs128_nt256_nsl32_ps512"
        );
    }

    #[test]
    fn test_get_reshape_and_cache_pipeline() {
        // Graceful skip on no-Metal hosts (CI VMs, sandboxes). See
        // `LayerKVPool::test_new_allocates_per_layer_buffers` for the
        // canonical pattern.
        let state = match MetalState::get() {
            Ok(s) => s,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_get_reshape_and_cache_pipeline: {e}");
                return;
            }
            Err(e) => panic!("unexpected MetalState::get failure: {e}"),
        };
        let kernel_name =
            MetalState::reshape_and_cache_kernel_name(MetalDtype::Float16, MetalDtype::Float16);
        let pipeline = state.get_pipeline(&kernel_name);
        assert!(
            pipeline.is_ok(),
            "Failed to get reshape_and_cache pipeline: {:?}",
            pipeline.err()
        );
    }

    #[test]
    fn test_get_paged_attention_pipeline() {
        // Graceful skip on no-Metal hosts (CI VMs, sandboxes).
        let state = match MetalState::get() {
            Ok(s) => s,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_get_paged_attention_pipeline: {e}");
                return;
            }
            Err(e) => panic!("unexpected MetalState::get failure: {e}"),
        };
        // Test V1 kernel for Qwen3 config: head_size=128, block_size=16
        let kernel_name = MetalState::paged_attention_v1_kernel_name(
            MetalDtype::Float16,
            MetalDtype::Float16,
            128,
            16,
            false,
        );
        let pipeline = state.get_pipeline(&kernel_name);
        assert!(
            pipeline.is_ok(),
            "Failed to get paged_attention V1 pipeline: {:?}",
            pipeline.err()
        );
    }
}
