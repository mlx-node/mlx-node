//! PagedKVCache - Main type for paged attention
//!
//! This is the primary interface for using PagedAttention.

use crate::block_table::BlockTable;
use crate::cache_engine::CacheEngineManager;
use crate::config::PagedAttentionConfig;

/// PagedKVCache for efficient KV cache management
///
/// Uses block-based memory allocation inspired by OS virtual memory,
/// achieving near-zero memory waste compared to traditional pre-allocated caches.
///
/// ## Example
/// ```typescript
/// const cache = new PagedKVCache({
///   blockSize: 32,
///   gpuMemoryMb: 4096,
///   headSize: 128,
///   numKvHeads: 4,
///   numLayers: 28,
/// });
///
/// // Add a sequence
/// const seqId = cache.addSequence(100); // 100-token prompt
///
/// // Generate tokens
/// for (let i = 0; i < maxTokens; i++) {
///   // Update cache with new K/V
///   cache.update(layerIdx, keys, values);
///
///   // Run paged attention
///   const output = cache.attention(layerIdx, queries, scale);
/// }
///
/// // Remove when done
/// cache.remove_sequence(seq_id);
/// ```
pub struct PagedKVCache {
    /// Block table for sequence management
    block_table: BlockTable,

    /// Cache engine manager
    engine_manager: CacheEngineManager,

    /// Configuration
    config: PagedAttentionConfig,
}

impl PagedKVCache {
    /// Create a new PagedKVCache
    ///
    /// # Arguments
    /// * `config` - Configuration for the paged attention system
    pub fn new(config: PagedAttentionConfig) -> Result<Self, String> {
        let engine_manager = CacheEngineManager::new(config.clone())?;

        let block_table = BlockTable::new(config.block_size);

        Ok(Self {
            block_table,
            engine_manager,
            config,
        })
    }

    /// Add a new sequence to the cache
    ///
    /// # Arguments
    /// * `prompt_len` - Number of tokens in the prompt
    ///
    /// # Returns
    /// * Sequence ID for future operations
    pub fn add_sequence(&mut self, prompt_len: u32) -> Result<u32, String> {
        // Calculate blocks needed
        let blocks_needed = prompt_len.div_ceil(self.config.block_size);

        // Check if we can allocate
        if !self.engine_manager.allocator().can_allocate(blocks_needed) {
            return Err(format!(
                "Not enough memory to allocate {} blocks for {} tokens",
                blocks_needed, prompt_len
            ));
        }

        // Add sequence to block table
        let seq_id = self.block_table.add_sequence();

        // Allocate blocks for the sequence
        let table = self.block_table.get_mut(seq_id).unwrap();
        for _ in 0..blocks_needed {
            let block = self
                .engine_manager
                .allocator_mut()
                .allocate()
                .ok_or_else(|| "Block allocation failed".to_string())?;
            table.add_block(block);
        }
        table.set_num_tokens(prompt_len);

        Ok(seq_id)
    }

    /// Remove a sequence from the cache
    ///
    /// Frees all blocks associated with the sequence.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID to remove
    pub fn remove_sequence(&mut self, seq_id: u32) -> Result<(), String> {
        let table = self
            .block_table
            .remove_sequence(seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        // Free all blocks
        for block in table.blocks().iter() {
            self.engine_manager.allocator_mut().free(block.clone());
        }

        Ok(())
    }

    /// Check if we can allocate blocks for a new sequence
    ///
    /// # Arguments
    /// * `num_blocks` - Number of blocks needed
    ///
    /// # Returns
    /// * true if allocation is possible
    pub fn can_allocate(&self, num_blocks: u32) -> bool {
        self.engine_manager.allocator().can_allocate(num_blocks)
    }

    /// Get the number of active sequences
    pub fn num_sequences(&self) -> u32 {
        self.block_table.num_sequences() as u32
    }

    /// Get the number of free blocks
    pub fn num_free_blocks(&self) -> u32 {
        self.engine_manager.allocator().num_free_blocks()
    }

    /// Get the total number of blocks
    pub fn total_blocks(&self) -> u32 {
        self.engine_manager.allocator().total_blocks()
    }

    /// Get context lengths for all sequences
    pub fn get_context_lens(&self) -> Vec<u32> {
        self.block_table.build_context_lens_array()
    }

    /// Get sequence IDs
    pub fn get_seq_ids(&self) -> Vec<u32> {
        self.block_table.seq_ids()
    }

    /// Get the block size
    pub fn get_block_size(&self) -> u32 {
        self.config.block_size
    }

    /// Get the maximum blocks per sequence (for kernel dispatch)
    /// Internal: used by Rust code, not exposed to JS
    pub fn get_max_blocks_per_seq(&self) -> u32 {
        let max_seq_len = self.config.max_seq_len();
        max_seq_len.div_ceil(self.config.block_size)
    }

    /// Build block tables array for kernel dispatch
    /// Internal: used by Rust attention code
    pub fn build_block_tables(&self) -> Vec<u32> {
        let max_blocks_per_seq = self.get_max_blocks_per_seq() as usize;
        self.block_table
            .build_block_tables_array(max_blocks_per_seq)
    }

    /// Extend a sequence with additional tokens
    /// Allocates new blocks if needed
    pub fn extend_sequence(&mut self, seq_id: u32, num_new_tokens: u32) -> Result<(), String> {
        let table = self
            .block_table
            .get_mut(seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        let blocks_needed = table.blocks_needed(num_new_tokens);

        for _ in 0..blocks_needed {
            let block = self
                .engine_manager
                .allocator_mut()
                .allocate()
                .ok_or_else(|| "Block allocation failed".to_string())?;
            table.add_block(block);
        }

        let current_tokens = table.num_tokens();
        table.set_num_tokens(current_tokens + num_new_tokens);

        Ok(())
    }

    /// Get slot mapping for new tokens being added
    /// Internal: used by reshape_and_cache kernel
    pub fn get_slot_mapping(
        &self,
        seq_id: u32,
        start_pos: u32,
        num_tokens: u32,
    ) -> Result<Vec<i64>, String> {
        let table = self
            .block_table
            .get(seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        let mut slots = Vec::with_capacity(num_tokens as usize);
        for i in 0..num_tokens {
            let token_pos = start_pos + i;
            let slot = table.absolute_slot_index(token_pos).ok_or_else(|| {
                format!(
                    "Token position {} out of range for sequence {}",
                    token_pos, seq_id
                )
            })?;
            slots.push(slot);
        }
        Ok(slots)
    }

    /// Get slot mapping for multiple sequences (batch operation)
    ///
    /// For each sequence, returns slot indices for the tokens being processed.
    /// This is used for batch prefill or decode steps.
    ///
    /// # Arguments
    /// * `seq_ids` - Sequence IDs
    /// * `context_lens` - Current context length for each sequence (tokens already in cache)
    /// * `is_prefill` - Whether each sequence is in prefill phase
    /// * `input_lens` - Number of input tokens for each sequence (prefill_len or 1 for decode)
    ///
    /// # Returns
    /// * Flat vector of slot indices (int64) for all tokens across all sequences
    pub fn get_slot_mapping_batch(
        &self,
        seq_ids: &[u32],
        context_lens: &[u32],
        is_prefill: &[bool],
        input_lens: &[u32],
    ) -> Result<Vec<i64>, String> {
        let mut all_slots = Vec::new();
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let context_len = context_lens.get(i).copied().unwrap_or(0);
            let is_pf = is_prefill.get(i).copied().unwrap_or(false);
            let input_len = input_lens.get(i).copied().unwrap_or(1);

            if is_pf {
                // Prefill: get slots for positions 0..input_len
                let slots = self.get_slot_mapping(seq_id, 0, input_len)?;
                all_slots.extend(slots);
            } else {
                // Decode: get slot for the token at position context_len - 1
                // The scheduler increments seq.position AFTER each output, so:
                // - context_len = prompt_len + generated_count (already includes last generated)
                // - The token being processed is at position context_len - 1 (0-indexed)
                // - We need to store its K/V at that slot
                if context_len == 0 {
                    return Err(format!(
                        "Decode sequence {} has context_len=0. This indicates prefill was not completed. \
                        Prefill must run before decode.",
                        seq_id
                    ));
                }
                let slots = self.get_slot_mapping(seq_id, context_len - 1, 1)?;
                all_slots.extend(slots);
            }
        }
        Ok(all_slots)
    }

    /// Get slot mapping for decode-only (simplified API for single-token generation)
    ///
    /// # Arguments
    /// * `seq_ids` - Sequence IDs
    /// * `context_lens` - Current context length for each sequence
    ///
    /// # Returns
    /// * Vector of slot indices (int64), one per sequence
    pub fn get_slot_mapping_decode(
        &self,
        seq_ids: &[u32],
        context_lens: &[u32],
    ) -> Result<Vec<i64>, String> {
        let is_prefill = vec![false; seq_ids.len()];
        let input_lens = vec![1u32; seq_ids.len()];
        self.get_slot_mapping_batch(seq_ids, context_lens, &is_prefill, &input_lens)
    }

    /// Build block tables for multiple sequences (batch operation)
    ///
    /// Returns block tables for each sequence in the batch.
    ///
    /// # Arguments
    /// * `seq_ids` - Sequence IDs to include
    ///
    /// # Returns
    /// * Vector of block tables, one per sequence
    pub fn build_block_tables_batch(&self, seq_ids: &[u32]) -> Result<Vec<Vec<u32>>, String> {
        let mut tables = Vec::with_capacity(seq_ids.len());
        for &seq_id in seq_ids {
            let table = self
                .block_table
                .get(seq_id)
                .ok_or_else(|| format!("Sequence {} not found", seq_id))?;
            tables.push(table.block_ids());
        }
        Ok(tables)
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let total_blocks = self.engine_manager.allocator().total_blocks();
        let free_blocks = self.engine_manager.allocator().num_free_blocks();
        let allocated_blocks = self.engine_manager.allocator().num_allocated_blocks();

        let bytes_per_block = 2
            * self.config.num_kv_heads as u64
            * self.config.head_size as u64
            * self.config.block_size as u64
            * self.config.num_layers as u64
            * if self.config.use_fp8() { 1 } else { 2 };

        MemoryStats {
            total_blocks,
            free_blocks,
            allocated_blocks,
            total_memory_mb: (total_blocks as u64 * bytes_per_block / 1024 / 1024) as u32,
            used_memory_mb: (allocated_blocks as u64 * bytes_per_block / 1024 / 1024) as u32,
            utilization_percent: if total_blocks > 0 {
                allocated_blocks as f64 / total_blocks as f64 * 100.0
            } else {
                0.0
            },
        }
    }

    // TODO: Implement Metal kernel integration
    // - update() - calls reshape_and_cache kernel
    // - attention() - calls paged_attention kernel
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total number of blocks in the pool
    pub total_blocks: u32,
    /// Number of free blocks
    pub free_blocks: u32,
    /// Number of allocated blocks
    pub allocated_blocks: u32,
    /// Total memory in MB
    pub total_memory_mb: u32,
    /// Used memory in MB
    pub used_memory_mb: u32,
    /// Utilization percentage
    pub utilization_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_kv_cache() {
        let config = PagedAttentionConfig {
            block_size: 32,
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 28,
            ..Default::default()
        };

        let mut cache = PagedKVCache::new(config).unwrap();

        // Add a sequence
        let seq_id = cache.add_sequence(100).unwrap();
        assert_eq!(cache.num_sequences(), 1);

        // Check blocks allocated (100 tokens / 32 block_size = 4 blocks)
        let stats = cache.get_memory_stats();
        assert_eq!(stats.allocated_blocks, 4);

        // Remove sequence
        cache.remove_sequence(seq_id).unwrap();
        assert_eq!(cache.num_sequences(), 0);

        // Blocks should be freed
        let stats = cache.get_memory_stats();
        assert_eq!(stats.allocated_blocks, 0);
    }

    #[test]
    fn test_extend_sequence() {
        let config = PagedAttentionConfig {
            block_size: 32,
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 28,
            ..Default::default()
        };

        let mut cache = PagedKVCache::new(config).unwrap();

        // Add a sequence with 30 tokens (fits in 1 block)
        let seq_id = cache.add_sequence(30).unwrap();
        assert_eq!(cache.get_memory_stats().allocated_blocks, 1);

        // Extend by 2 tokens (still fits: 30+2=32 = exactly 1 block)
        cache.extend_sequence(seq_id, 2).unwrap();
        assert_eq!(cache.get_memory_stats().allocated_blocks, 1);

        // Extend by 1 token (now needs 2nd block: 32+1=33 > 32)
        cache.extend_sequence(seq_id, 1).unwrap();
        assert_eq!(cache.get_memory_stats().allocated_blocks, 2);

        // Extend by 30 more tokens (needs 3rd block: 33+30=63 > 64)
        cache.extend_sequence(seq_id, 30).unwrap();
        assert_eq!(cache.get_memory_stats().allocated_blocks, 2); // 63 fits in 2 blocks

        // Extend by 2 more (65 tokens needs 3 blocks)
        cache.extend_sequence(seq_id, 2).unwrap();
        assert_eq!(cache.get_memory_stats().allocated_blocks, 3);

        // Get context lens
        let context_lens = cache.get_context_lens();
        assert_eq!(context_lens, vec![65]);

        cache.remove_sequence(seq_id).unwrap();
    }

    #[test]
    fn test_slot_mapping() {
        let config = PagedAttentionConfig {
            block_size: 32,
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 28,
            ..Default::default()
        };

        let mut cache = PagedKVCache::new(config).unwrap();

        // Add a sequence with 64 tokens (needs 2 blocks)
        let seq_id = cache.add_sequence(64).unwrap();

        // Get slot mapping for tokens 0-4
        let slots = cache.get_slot_mapping(seq_id, 0, 5).unwrap();
        assert_eq!(slots.len(), 5);

        // First block starts at slot 0
        assert_eq!(slots[0], 0);
        assert_eq!(slots[1], 1);
        assert_eq!(slots[4], 4);

        // Get slot mapping for tokens 32-34 (second block)
        let slots = cache.get_slot_mapping(seq_id, 32, 3).unwrap();
        assert_eq!(slots.len(), 3);
        // Second block should have different base offset
        assert!(slots[0] >= 32); // In second block

        cache.remove_sequence(seq_id).unwrap();
    }

    #[test]
    fn test_block_tables() {
        let config = PagedAttentionConfig {
            block_size: 32,
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 28,
            max_seq_len: Some(256), // 8 blocks max
            ..Default::default()
        };

        let mut cache = PagedKVCache::new(config).unwrap();

        // Add two sequences
        let seq1 = cache.add_sequence(64).unwrap(); // 2 blocks
        let seq2 = cache.add_sequence(96).unwrap(); // 3 blocks

        assert_eq!(cache.num_sequences(), 2);
        assert_eq!(cache.get_memory_stats().allocated_blocks, 5);

        // Build block tables
        let block_tables = cache.build_block_tables();
        let max_blocks = cache.get_max_blocks_per_seq() as usize;

        // Should have num_seqs * max_blocks entries
        assert_eq!(block_tables.len(), 2 * max_blocks);

        cache.remove_sequence(seq1).unwrap();
        cache.remove_sequence(seq2).unwrap();
    }
}
