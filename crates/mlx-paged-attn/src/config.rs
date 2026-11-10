//! Configuration types for PagedAttention

use crate::metal::MetalDtype;
use crate::profile::bytes_per_block;

const BYTES_PER_MIB: u64 = 1024 * 1024;
/// The smallest `gpu_memory_mb` that [`PagedAttentionConfig::validate`] accepts.
pub const MIN_GPU_MEMORY_MB: u32 = 256;

/// Configuration for PagedAttention
#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    /// Block size in tokens (8, 16, or 32)
    /// Default: 32
    pub block_size: u32,

    /// Total GPU memory for KV cache in MB
    /// Default: 4096 (4GB)
    pub gpu_memory_mb: u32,

    /// Head size (must match model: 64, 80, 96, 112, 120, 128, 192, or 256)
    pub head_size: u32,

    /// Number of KV heads (for GQA, typically < num_query_heads)
    pub num_kv_heads: u32,

    /// Number of transformer layers
    pub num_layers: u32,

    /// Whether to use FP8 quantization for cache
    /// Reduces memory by ~50% with minimal quality loss
    /// Default: false
    pub use_fp8_cache: Option<bool>,

    /// Maximum sequence length supported
    /// Default: 8192
    pub max_seq_len: Option<u32>,

    /// Maximum batch size for continuous batching
    /// Default: 256
    pub max_batch_size: Option<u32>,
}

impl PagedAttentionConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate block size
        if ![8, 16, 32].contains(&self.block_size) {
            return Err(format!(
                "Invalid block_size: {}. Must be 8, 16, or 32.",
                self.block_size
            ));
        }

        // Validate head size
        let valid_head_sizes = [32, 64, 80, 96, 112, 120, 128, 192, 256, 512];
        if !valid_head_sizes.contains(&self.head_size) {
            return Err(format!(
                "Invalid head_size: {}. Must be one of {:?}.",
                self.head_size, valid_head_sizes
            ));
        }

        // Validate memory
        if self.gpu_memory_mb < MIN_GPU_MEMORY_MB {
            return Err(format!(
                "gpu_memory_mb too small: {}. Minimum is {MIN_GPU_MEMORY_MB} MB.",
                self.gpu_memory_mb
            ));
        }

        // Validate layer count
        if self.num_layers == 0 {
            return Err("num_layers must be > 0".to_string());
        }

        // Validate KV heads
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be > 0".to_string());
        }

        // FP8 cache requires block_size 16 or 32 for optimal memory alignment
        if self.use_fp8_cache.unwrap_or(false) && self.block_size == 8 {
            return Err(
                "FP8 cache requires block_size 16 or 32 for optimal alignment.".to_string(),
            );
        }

        Ok(())
    }

    /// MiB needed to cache one sequence of `max_seq_len` tokens in a
    /// `cache_dtype` pool ([`crate::profile::bytes_per_block`]).
    ///
    /// `num_layers` is the physical full-attention layer count, not the
    /// model's total mixer count. Zero inputs return the 256 MiB floor.
    pub fn memory_mb_for_one_full_sequence(
        max_seq_len: u32,
        block_size: u32,
        head_size: u32,
        num_kv_heads: u32,
        num_layers: u32,
        cache_dtype: MetalDtype,
    ) -> u32 {
        let Ok(bytes_per_block) =
            bytes_per_block(num_layers, num_kv_heads, head_size, block_size, cache_dtype)
        else {
            return MIN_GPU_MEMORY_MB;
        };
        if max_seq_len == 0 {
            return MIN_GPU_MEMORY_MB;
        }
        let max_blocks = u64::from(max_seq_len.div_ceil(block_size));
        let required_mb = bytes_per_block
            .saturating_mul(max_blocks)
            .div_ceil(BYTES_PER_MIB)
            .max(u64::from(MIN_GPU_MEMORY_MB));
        u32::try_from(required_mb).unwrap_or(u32::MAX)
    }

    /// Whole blocks of a `cache_dtype` pool in `gpu_memory_mb`
    /// ([`crate::profile::bytes_per_block`], the bytes Splash
    /// `kv::Layout::bytesPerModelPage` divides a budget by). 0 when a shape
    /// factor is 0.
    pub fn calculate_num_blocks(&self, cache_dtype: MetalDtype) -> u32 {
        let Ok(bytes_per_block) = bytes_per_block(
            self.num_layers,
            self.num_kv_heads,
            self.head_size,
            self.block_size,
            cache_dtype,
        ) else {
            return 0;
        };
        let total_bytes = u64::from(self.gpu_memory_mb) * BYTES_PER_MIB;
        u32::try_from(total_bytes / bytes_per_block).unwrap_or(u32::MAX)
    }

    /// Tokens a `cache_dtype` pool of [`Self::calculate_num_blocks`] holds.
    pub fn max_cached_tokens(&self, cache_dtype: MetalDtype) -> u64 {
        u64::from(self.calculate_num_blocks(cache_dtype)) * u64::from(self.block_size)
    }

    /// Get the use_fp8_cache setting with default
    pub fn use_fp8(&self) -> bool {
        self.use_fp8_cache.unwrap_or(false)
    }

    /// Get the max_seq_len setting with default
    pub fn max_seq_len(&self) -> u32 {
        self.max_seq_len.unwrap_or(8192)
    }

    /// Get the max_batch_size setting with default
    pub fn max_batch_size(&self) -> u32 {
        self.max_batch_size.unwrap_or(256)
    }
}

impl Default for PagedAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 32,
            gpu_memory_mb: 4096,
            head_size: 128, // Qwen3 default
            num_kv_heads: 4,
            num_layers: 28,
            use_fp8_cache: Some(false),
            max_seq_len: Some(8192),
            max_batch_size: Some(256),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let mut config = PagedAttentionConfig::default();
        assert!(config.validate().is_ok());

        // Invalid block size
        config.block_size = 64;
        assert!(config.validate().is_err());
        config.block_size = 32;

        // Invalid head size
        config.head_size = 100;
        assert!(config.validate().is_err());
        config.head_size = 128;

        // Valid again
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_block_calculation() {
        let config = PagedAttentionConfig {
            block_size: 32,
            gpu_memory_mb: 1024, // 1 GB
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 28,
            use_fp8_cache: Some(false),
            ..Default::default()
        };

        let num_blocks = config.calculate_num_blocks(MetalDtype::BFloat16);
        // 1 GB / (2 * 4 * 128 * 32 * 2 * 28) = 1024*1024*1024 / (2 * 4 * 128 * 32 * 2 * 28)
        // = 1073741824 / 1835008 ≈ 585
        assert!(
            num_blocks > 500 && num_blocks < 700,
            "Got {} blocks",
            num_blocks
        );
    }

    #[test]
    fn test_fp8_doubles_capacity() {
        let config_fp16 = PagedAttentionConfig {
            block_size: 32,
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 28,
            use_fp8_cache: Some(false),
            ..Default::default()
        };

        let config_fp8 = PagedAttentionConfig {
            use_fp8_cache: Some(true),
            ..config_fp16.clone()
        };

        let blocks_fp16 = config_fp16.calculate_num_blocks(MetalDtype::Float16);
        let blocks_fp8 = config_fp8.calculate_num_blocks(MetalDtype::UChar);

        // FP8 should give ~2x the blocks (calculation still works, just validation rejects)
        assert!(blocks_fp8 >= blocks_fp16 * 2 - 1 && blocks_fp8 <= blocks_fp16 * 2 + 1);
    }

    #[test]
    fn test_fp8_validation() {
        // FP8 with block_size 32 should work
        let config = PagedAttentionConfig {
            use_fp8_cache: Some(true),
            block_size: 32,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        // FP8 with block_size 16 should work
        let config = PagedAttentionConfig {
            use_fp8_cache: Some(true),
            block_size: 16,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        // FP8 with block_size 8 should fail (alignment)
        let config = PagedAttentionConfig {
            use_fp8_cache: Some(true),
            block_size: 8,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("block_size 16 or 32"));
    }

    #[test]
    fn one_full_sequence_mb_for_nemotron_lightning() {
        // 1_048_576 tokens, block 16, head 128, 2 kv, 6 attn layers, bf16 K+V
        let mb = PagedAttentionConfig::memory_mb_for_one_full_sequence(
            1_048_576,
            16,
            128,
            2,
            6,
            MetalDtype::BFloat16,
        );
        assert_eq!(mb, 6144);
        let cfg = PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: mb,
            head_size: 128,
            num_kv_heads: 2,
            num_layers: 6,
            use_fp8_cache: Some(false),
            max_seq_len: Some(1_048_576),
            max_batch_size: Some(32),
        };
        assert_eq!(cfg.max_cached_tokens(MetalDtype::BFloat16), 1_048_576);
    }

    #[test]
    fn one_full_sequence_mb_matches_qwen35_35b_a3b() {
        let mb = PagedAttentionConfig::memory_mb_for_one_full_sequence(
            262_144,
            16,
            256,
            2,
            10,
            MetalDtype::BFloat16,
        );
        assert_eq!(mb, 5_120);
    }

    /// Int8 sizing is `2·L·H·bs·(D+4)` per block in both directions (Splash
    /// `kv::Layout::bytesPerModelPage`). 27B: L 16, H 4, D 256, bs 16.
    /// Mutation M8: `calculate_num_blocks` sizes at bf16 whatever the dtype.
    #[test]
    fn int8_sizing_is_one_formula() {
        let config = PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 5_051,
            head_size: 256,
            num_kv_heads: 4,
            num_layers: 16,
            use_fp8_cache: Some(false),
            ..Default::default()
        };
        assert_eq!(config.calculate_num_blocks(MetalDtype::Int8), 9_946);
        assert_eq!(config.calculate_num_blocks(MetalDtype::BFloat16), 5_051);
        assert_eq!(config.max_cached_tokens(MetalDtype::Int8), 9_946 * 16);

        let mb = PagedAttentionConfig::memory_mb_for_one_full_sequence(
            262_144,
            16,
            256,
            4,
            16,
            MetalDtype::Int8,
        );
        assert_eq!(mb, 8_320);
        let full = PagedAttentionConfig {
            gpu_memory_mb: mb,
            ..config
        };
        assert_eq!(full.calculate_num_blocks(MetalDtype::Int8), 16_384);
        assert_eq!(full.max_cached_tokens(MetalDtype::Int8), 262_144);
    }
}
