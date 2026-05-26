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
        let device = Device::system_default().ok_or("No Metal device found")?;

        // Write metallib to temp file and load
        // (Metal requires loading from file path, not memory)
        let temp_path = std::env::temp_dir().join("mlx_paged_attn.metallib");
        std::fs::write(&temp_path, METALLIB_BYTES)
            .map_err(|e| format!("Failed to write metallib to temp: {}", e))?;

        let library = device
            .new_library_with_file(&temp_path)
            .map_err(|e| format!("Failed to load metallib: {}", e))?;

        let command_queue = device.new_command_queue();

        Ok(MetalState {
            device,
            library,
            command_queue,
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

    /// Get the `reshape_and_cache` kernel name for an `(input, cache)` dtype pair.
    ///
    /// The metal source instantiates these combinations (see
    /// `crates/mlx-paged-attn/metal/cache/reshape_and_cache.metal`):
    /// - Non-FP8: `(float, float)`, `(half, half)`, `(bfloat16_t, bfloat16_t)`
    /// - FP8 (cache = `uchar`): `(float, uchar)`, `(half, uchar)`, `(bfloat16_t, uchar)`
    ///
    /// Hard-coding the input dtype as `half` (the previous behavior) silently
    /// dispatched the wrong kernel when callers passed BF16 or F32 K/V — and
    /// for FP8 with BF16 input, even routed BF16 bytes through a kernel
    /// expecting `half`, corrupting the cache. Splitting the two dtypes
    /// explicitly forces every caller to identify the input dtype, and the
    /// Metal source's kernel-name lookup will fail loudly if the requested
    /// pair was never instantiated.
    ///
    /// # Arguments
    /// * `input_dtype` - Data type of the K/V input arrays handed to the kernel
    /// * `cache_dtype` - Data type for cache storage (UChar for FP8, otherwise
    ///   the same as `input_dtype`)
    /// * `use_fp8` - Whether to use the `_fp8` (scale-divided) variant
    pub fn reshape_and_cache_kernel_name(
        input_dtype: MetalDtype,
        cache_dtype: MetalDtype,
        use_fp8: bool,
    ) -> String {
        let input_type = input_dtype.type_string();
        let cache_type = cache_dtype.type_string();
        let suffix = if use_fp8 { "_fp8" } else { "" };
        format!(
            "reshape_and_cache_kv_{}_cache_{}{}",
            input_type, cache_type, suffix
        )
    }

    /// Get the paged_attention V1 kernel name (no partitioning).
    ///
    /// The metal source instantiates these `(io_type, cache_type)` pairs (see
    /// `crates/mlx-paged-attn/metal/attention/paged_attention.metal`):
    /// - Non-FP8: `(float, float)`, `(half, half)`, `(bfloat16_t, bfloat16_t)`
    /// - FP8 (`cache_type = uchar`): `(float, uchar)`, `(half, uchar)`,
    ///   `(bfloat16_t, uchar)`
    ///
    /// Hard-coding the io dtype as `half` (the previous behavior) silently
    /// dispatched the wrong kernel when the cache dtype was BFloat16 — and
    /// for non-FP8 caches there is no `(half, bfloat16_t)` instantiation, so
    /// the kernel-name lookup would fall back to corrupt routing through
    /// `(half, half)` (or fail loudly, depending on the cache_dtype value).
    /// Splitting `io_dtype` from `cache_dtype` forces every caller to
    /// identify the io dtype, and the metallib lookup now fails loudly if
    /// the requested pair was never instantiated.
    ///
    /// # Arguments
    /// * `io_dtype` - Data type for queries + output (must equal `cache_dtype`
    ///   for non-FP8; for FP8 cache the io dtype can independently be Float16,
    ///   BFloat16, or Float32)
    /// * `cache_dtype` - Data type for cache storage (UChar for FP8,
    ///   otherwise Float16 / BFloat16 / Float32 to match the cache buffers)
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

    /// Varlen counterparts to the V1/V2/reduce kernel name helpers above
    /// (Phase 4a). The naming convention mirrors the single-row helpers
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
        // Non-FP8: every (input, cache) pair instantiated by the metal source.
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(
                MetalDtype::Float16,
                MetalDtype::Float16,
                false,
            ),
            "reshape_and_cache_kv_half_cache_half"
        );
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(
                MetalDtype::BFloat16,
                MetalDtype::BFloat16,
                false,
            ),
            "reshape_and_cache_kv_bfloat16_t_cache_bfloat16_t"
        );
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(
                MetalDtype::Float32,
                MetalDtype::Float32,
                false,
            ),
            "reshape_and_cache_kv_float_cache_float"
        );

        // FP8: every (input, uchar) pair instantiated with the `_fp8` suffix.
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(MetalDtype::Float16, MetalDtype::UChar, true,),
            "reshape_and_cache_kv_half_cache_uchar_fp8"
        );
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(
                MetalDtype::BFloat16,
                MetalDtype::UChar,
                true,
            ),
            "reshape_and_cache_kv_bfloat16_t_cache_uchar_fp8"
        );
        assert_eq!(
            MetalState::reshape_and_cache_kernel_name(MetalDtype::Float32, MetalDtype::UChar, true,),
            "reshape_and_cache_kv_float_cache_uchar_fp8"
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
        // (half, uchar) — FP8 cache, half io.
        assert_eq!(
            MetalState::paged_attention_v1_kernel_name(
                MetalDtype::Float16,
                MetalDtype::UChar,
                128,
                16,
                false,
            ),
            "paged_attention_half_cache_uchar_hs128_bs16_nt256_nsl32_ps0"
        );
        assert_eq!(
            MetalState::paged_attention_v2_kernel_name(
                MetalDtype::Float16,
                MetalDtype::UChar,
                128,
                16,
                false,
            ),
            "paged_attention_half_cache_uchar_hs128_bs16_nt256_nsl32_ps512"
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
        let kernel_name = MetalState::reshape_and_cache_kernel_name(
            MetalDtype::Float16,
            MetalDtype::Float16,
            false,
        );
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
