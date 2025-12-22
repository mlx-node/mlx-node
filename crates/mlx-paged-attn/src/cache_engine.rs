//! Cache engine for per-layer KV cache management
//!
//! Each layer has its own CacheEngine that manages the physical
//! key and value cache tensors.

use crate::block_allocator::BlockAllocator;
use crate::config::PagedAttentionConfig;

/// Cache engine for a single transformer layer
pub struct CacheEngine {
    /// Layer index
    layer_idx: u32,

    /// Configuration (stored for future Metal buffer management)
    #[allow(dead_code)]
    config: PagedAttentionConfig,

    /// Whether the cache has been initialized
    initialized: bool,
}

impl CacheEngine {
    /// Create a new cache engine
    pub fn new(layer_idx: u32, config: PagedAttentionConfig) -> Self {
        Self {
            layer_idx,
            config,
            initialized: false,
        }
    }

    /// Get the layer index
    pub fn layer_idx(&self) -> u32 {
        self.layer_idx
    }

    /// Check if the cache is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    // TODO: Implement Metal buffer management
    // - Initialize key/value cache buffers
    // - Call reshape_and_cache kernel
    // - Call paged_attention kernel
}

/// Manager for all layer cache engines
pub struct CacheEngineManager {
    /// Per-layer cache engines
    engines: Vec<CacheEngine>,

    /// Shared block allocator
    allocator: BlockAllocator,

    /// Configuration
    config: PagedAttentionConfig,
}

impl CacheEngineManager {
    /// Create a new cache engine manager
    pub fn new(config: PagedAttentionConfig) -> Result<Self, String> {
        config.validate()?;

        let num_blocks = config.calculate_num_blocks();
        let allocator = BlockAllocator::new(num_blocks, config.block_size);

        let engines: Vec<CacheEngine> = (0..config.num_layers)
            .map(|i| CacheEngine::new(i, config.clone()))
            .collect();

        Ok(Self {
            engines,
            allocator,
            config,
        })
    }

    /// Get the allocator
    pub fn allocator(&self) -> &BlockAllocator {
        &self.allocator
    }

    /// Get mutable access to the allocator
    pub fn allocator_mut(&mut self) -> &mut BlockAllocator {
        &mut self.allocator
    }

    /// Get a layer's cache engine
    pub fn get_engine(&self, layer_idx: u32) -> Option<&CacheEngine> {
        self.engines.get(layer_idx as usize)
    }

    /// Get mutable access to a layer's cache engine
    pub fn get_engine_mut(&mut self, layer_idx: u32) -> Option<&mut CacheEngine> {
        self.engines.get_mut(layer_idx as usize)
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> u32 {
        self.engines.len() as u32
    }

    /// Get the configuration
    pub fn config(&self) -> &PagedAttentionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_engine_manager() {
        let config = PagedAttentionConfig {
            block_size: 32,
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 28,
            ..Default::default()
        };

        let manager = CacheEngineManager::new(config).unwrap();
        assert_eq!(manager.num_layers(), 28);
        assert!(manager.allocator().num_free_blocks() > 0);
    }
}
