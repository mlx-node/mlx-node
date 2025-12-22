//! Configuration types for PagedAttention

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
        let valid_head_sizes = [32, 64, 80, 96, 112, 120, 128, 192, 256];
        if !valid_head_sizes.contains(&self.head_size) {
            return Err(format!(
                "Invalid head_size: {}. Must be one of {:?}.",
                self.head_size, valid_head_sizes
            ));
        }

        // Validate memory
        if self.gpu_memory_mb < 256 {
            return Err(format!(
                "gpu_memory_mb too small: {}. Minimum is 256 MB.",
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

        // FP8 cache is not yet implemented - reject early with clear error
        if self.use_fp8_cache.unwrap_or(false) {
            return Err(
                "FP8 cache is not yet implemented. Set use_fp8_cache to false or None.".to_string(),
            );
        }

        Ok(())
    }

    /// Calculate the number of blocks that can be allocated
    pub fn calculate_num_blocks(&self) -> u32 {
        let use_fp8 = self.use_fp8_cache.unwrap_or(false);
        let element_size = if use_fp8 { 1 } else { 2 }; // FP8 = 1 byte, FP16 = 2 bytes

        // Memory per block per layer: 2 * num_kv_heads * head_size * block_size * element_size
        // (factor of 2 for K and V)
        let bytes_per_block_per_layer = 2
            * self.num_kv_heads as u64
            * self.head_size as u64
            * self.block_size as u64
            * element_size as u64;

        let bytes_per_block = bytes_per_block_per_layer * self.num_layers as u64;
        let total_bytes = self.gpu_memory_mb as u64 * 1024 * 1024;

        (total_bytes / bytes_per_block) as u32
    }

    /// Calculate maximum tokens that can be cached
    pub fn max_cached_tokens(&self) -> u64 {
        self.calculate_num_blocks() as u64 * self.block_size as u64
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

        let num_blocks = config.calculate_num_blocks();
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

        let blocks_fp16 = config_fp16.calculate_num_blocks();
        let blocks_fp8 = config_fp8.calculate_num_blocks();

        // FP8 should give ~2x the blocks (calculation still works, just validation rejects)
        assert!(blocks_fp8 >= blocks_fp16 * 2 - 1 && blocks_fp8 <= blocks_fp16 * 2 + 1);
    }

    #[test]
    fn test_fp8_validation_rejected() {
        let config = PagedAttentionConfig {
            use_fp8_cache: Some(true),
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("FP8 cache is not yet implemented")
        );
    }
}
