//! Cache engine for per-layer KV cache management
//!
//! Each layer has its own CacheEngine that manages the physical
//! key and value cache tensors.

use crate::block_allocator::BlockAllocator;
use crate::config::PagedAttentionConfig;

#[cfg(target_os = "macos")]
use metal::Buffer;

/// Cache engine for a single transformer layer
///
/// Manages the GPU memory for KV cache of a single transformer layer.
/// The cache layout follows vLLM conventions:
/// - Key cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
/// - Value cache: [num_blocks, num_kv_heads, head_size, block_size]
///
/// Where x is 16 / sizeof(dtype) for vectorized memory access.
pub struct CacheEngine {
    /// Layer index
    layer_idx: u32,

    /// Configuration
    config: PagedAttentionConfig,

    /// Key cache buffer (only on macOS with Metal)
    #[cfg(target_os = "macos")]
    key_cache: Option<Buffer>,

    /// Value cache buffer (only on macOS with Metal)
    #[cfg(target_os = "macos")]
    value_cache: Option<Buffer>,

    /// Whether the cache has been initialized
    initialized: bool,
}

impl CacheEngine {
    /// Create a new cache engine (uninitialized)
    pub fn new(layer_idx: u32, config: PagedAttentionConfig) -> Self {
        Self {
            layer_idx,
            config,
            #[cfg(target_os = "macos")]
            key_cache: None,
            #[cfg(target_os = "macos")]
            value_cache: None,
            initialized: false,
        }
    }

    /// Initialize the Metal cache buffers
    ///
    /// Must be called before using update() or attention().
    /// Allocates GPU memory for key and value caches.
    #[cfg(target_os = "macos")]
    pub fn initialize(&mut self, num_blocks: u32) -> Result<(), String> {
        use crate::metal::MetalState;
        use metal::MTLResourceOptions;

        if self.initialized {
            return Ok(());
        }

        let state = MetalState::get()?;

        // Calculate buffer sizes based on FP8 or FP16
        let use_fp8 = self.config.use_fp8();
        let element_size = if use_fp8 { 1u64 } else { 2u64 }; // FP8 = 1 byte, FP16 = 2 bytes

        // Key cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        // where x = 16 / sizeof(dtype)
        // For FP16: x = 16/2 = 8
        // For FP8:  x = 16/1 = 16
        let x = if use_fp8 { 16u32 } else { 8u32 };

        let key_cache_size = num_blocks as u64
            * self.config.num_kv_heads as u64
            * (self.config.head_size as u64 / x as u64)
            * self.config.block_size as u64
            * x as u64
            * element_size;

        // Value cache: [num_blocks, num_kv_heads, head_size, block_size]
        let value_cache_size = num_blocks as u64
            * self.config.num_kv_heads as u64
            * self.config.head_size as u64
            * self.config.block_size as u64
            * element_size;

        // Allocate GPU buffers
        let key_cache = state
            .device
            .new_buffer(key_cache_size, MTLResourceOptions::StorageModePrivate);

        let value_cache = state
            .device
            .new_buffer(value_cache_size, MTLResourceOptions::StorageModePrivate);

        self.key_cache = Some(key_cache);
        self.value_cache = Some(value_cache);
        self.initialized = true;

        Ok(())
    }

    /// Get the layer index
    pub fn layer_idx(&self) -> u32 {
        self.layer_idx
    }

    /// Check if the cache is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the key cache buffer
    #[cfg(target_os = "macos")]
    pub fn key_cache(&self) -> Option<&Buffer> {
        self.key_cache.as_ref()
    }

    /// Get the value cache buffer
    #[cfg(target_os = "macos")]
    pub fn value_cache(&self) -> Option<&Buffer> {
        self.value_cache.as_ref()
    }

    /// Get the configuration
    pub fn config(&self) -> &PagedAttentionConfig {
        &self.config
    }
}

/// Manager for all layer cache engines
pub struct CacheEngineManager {
    /// Per-layer cache engines
    engines: Vec<CacheEngine>,

    /// Shared block allocator
    allocator: BlockAllocator,

    /// Configuration
    config: PagedAttentionConfig,

    /// Number of blocks allocated
    num_blocks: u32,
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
            num_blocks,
        })
    }

    /// Initialize all cache engines (allocates GPU memory)
    ///
    /// Must be called before using update() or attention().
    #[cfg(target_os = "macos")]
    pub fn initialize(&mut self) -> Result<(), String> {
        for engine in &mut self.engines {
            engine.initialize(self.num_blocks)?;
        }
        Ok(())
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
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
