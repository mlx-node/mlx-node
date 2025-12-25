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

    /// KV scale manager for FP8 quantization (only used when use_fp8 is enabled)
    #[cfg(target_os = "macos")]
    scale_manager: Option<crate::metal::KvScaleManager>,

    /// Token tracker for logical block management
    token_tracker: crate::token_tracker::TokenTracker,
}

impl PagedKVCache {
    /// Create a new PagedKVCache
    ///
    /// # Arguments
    /// * `config` - Configuration for the paged attention system
    pub fn new(config: PagedAttentionConfig) -> Result<Self, String> {
        let engine_manager = CacheEngineManager::new(config.clone())?;

        let block_table = BlockTable::new(config.block_size);

        // Create scale manager for FP8 mode
        #[cfg(target_os = "macos")]
        let scale_manager = if config.use_fp8() {
            let mut manager = crate::metal::KvScaleManager::new(config.num_layers);
            manager.init_default_scales();
            Some(manager)
        } else {
            None
        };

        // Create token tracker for prefix caching
        let token_tracker = crate::token_tracker::TokenTracker::new(config.block_size);

        Ok(Self {
            block_table,
            engine_manager,
            config,
            #[cfg(target_os = "macos")]
            scale_manager,
            token_tracker,
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
    /// Frees all blocks associated with the sequence and cleans up token tracking data.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID to remove
    pub fn remove_sequence(&mut self, seq_id: u32) -> Result<(), String> {
        let table = self
            .block_table
            .remove_sequence(seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        // Free all blocks and clean up token tracking
        for block in table.blocks().iter() {
            // Only remove token tracking if this is the last reference to the block.
            // For shared blocks (e.g., from fork_sequence), keep tracking for remaining sequences.
            if block.get_ref_count() == 1 {
                self.token_tracker.remove_block(block.block_id);
            }
            // Free the physical block (decrements ref_count, only returns to pool if reaches 0)
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

    // ========== FP8 Scale Calibration Methods ==========

    /// Calibrate FP8 scales for a specific layer from observed KV tensors
    ///
    /// This computes optimal quantization scales based on the maximum absolute
    /// values in the key and value tensors. Should be called during a calibration
    /// pass before FP8 inference.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index (0 to num_layers-1)
    /// * `keys` - Key tensor for scale computation
    /// * `values` - Value tensor for scale computation
    ///
    /// # Returns
    /// * (k_scale, v_scale) tuple on success
    ///
    /// # Safety
    /// The key and value pointers must be valid MLX array handles.
    #[cfg(target_os = "macos")]
    pub unsafe fn calibrate_layer_scales(
        &mut self,
        layer_idx: u32,
        keys: *mut mlx_sys::mlx_array,
        values: *mut mlx_sys::mlx_array,
    ) -> Result<(f32, f32), String> {
        let manager = self.scale_manager.as_mut().ok_or_else(|| {
            "FP8 scale calibration requires use_fp8 to be enabled in config".to_string()
        })?;

        unsafe { manager.calibrate_layer(layer_idx, keys, values) }
    }

    /// Update FP8 scales using exponential moving average
    ///
    /// This is useful for online calibration during inference. The scale is
    /// updated as: new_scale = alpha * observed_scale + (1-alpha) * old_scale
    ///
    /// # Safety
    /// The key and value pointers must be valid MLX array handles.
    #[cfg(target_os = "macos")]
    pub unsafe fn update_layer_scales_ema(
        &mut self,
        layer_idx: u32,
        keys: *mut mlx_sys::mlx_array,
        values: *mut mlx_sys::mlx_array,
    ) -> Result<(f32, f32), String> {
        let manager = self.scale_manager.as_mut().ok_or_else(|| {
            "FP8 scale update requires use_fp8 to be enabled in config".to_string()
        })?;

        unsafe { manager.update_layer_ema(layer_idx, keys, values) }
    }

    /// Get the FP8 scales for a specific layer
    ///
    /// Returns (k_scale, v_scale) for use in kernel dispatch.
    /// Returns (1.0, 1.0) if FP8 is not enabled or layer not calibrated.
    #[cfg(target_os = "macos")]
    pub fn get_layer_scales(&self, layer_idx: u32) -> (f32, f32) {
        match &self.scale_manager {
            Some(manager) => (manager.k_scale(layer_idx), manager.v_scale(layer_idx)),
            None => (1.0, 1.0),
        }
    }

    /// Check if FP8 calibration has been performed
    #[cfg(target_os = "macos")]
    pub fn is_fp8_calibrated(&self) -> bool {
        self.scale_manager
            .as_ref()
            .map(|m| m.is_calibrated())
            .unwrap_or(false)
    }

    /// Get FP8 scale statistics
    #[cfg(target_os = "macos")]
    pub fn get_scale_stats(&self) -> Option<crate::metal::KvScaleStats> {
        self.scale_manager.as_ref().map(|m| m.stats())
    }

    /// Set FP8 scales directly (useful for loading pre-calibrated values)
    #[cfg(target_os = "macos")]
    pub fn set_layer_scales(&mut self, layer_idx: u32, k_scale: f32, v_scale: f32) {
        if let Some(manager) = &mut self.scale_manager {
            manager.set_scales(layer_idx, k_scale, v_scale);
        }
    }

    /// Load all FP8 scales from vectors (for checkpoint loading)
    #[cfg(target_os = "macos")]
    pub fn load_scales(&mut self, k_scales: &[f32], v_scales: &[f32]) {
        if let Some(manager) = &mut self.scale_manager {
            manager.load_scales(k_scales, v_scales);
        }
    }

    /// Get all FP8 scales as vectors (for checkpoint saving)
    #[cfg(target_os = "macos")]
    pub fn get_all_scales(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        self.scale_manager.as_ref().map(|m| m.get_all_scales())
    }

    // ========== End FP8 Scale Methods ==========

    // ========== Token Tracking Methods ==========

    /// Track tokens being stored in a block
    ///
    /// This should be called when new tokens are added to the cache to enable
    /// prefix caching and debugging capabilities.
    ///
    /// # Arguments
    /// * `block_id` - Physical block ID
    /// * `tokens` - Token IDs being stored
    /// * `start_offset` - Starting position within the block
    ///
    /// # Returns
    /// * Hash of the block's tokens (for prefix cache integration)
    pub fn track_tokens(&mut self, block_id: u32, tokens: &[u32], start_offset: u32) -> u64 {
        self.token_tracker
            .track_tokens(block_id, tokens, start_offset)
    }

    /// Track tokens for a sequence at the current position
    ///
    /// Convenience method that computes block assignments automatically.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID
    /// * `tokens` - Token IDs being added
    /// * `start_pos` - Starting position in the sequence
    pub fn track_sequence_tokens(
        &mut self,
        seq_id: u32,
        tokens: &[u32],
        start_pos: u32,
    ) -> Result<(), String> {
        let table = self
            .block_table
            .get(seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        let block_size = self.config.block_size;
        let blocks = table.blocks();

        for (i, &token) in tokens.iter().enumerate() {
            let pos = start_pos + i as u32;
            let block_idx = (pos / block_size) as usize;
            let offset_in_block = pos % block_size;

            if block_idx < blocks.len() {
                let block_id = blocks[block_idx].block_id;
                self.token_tracker
                    .track_tokens(block_id, &[token], offset_in_block);
            }
        }

        Ok(())
    }

    /// Get tokens stored in a specific block
    ///
    /// # Arguments
    /// * `block_id` - Physical block ID
    ///
    /// # Returns
    /// * Token IDs in the block
    pub fn get_block_tokens(&self, block_id: u32) -> &[u32] {
        self.token_tracker.get_block_tokens(block_id)
    }

    /// Get all tokens for a sequence
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID
    ///
    /// # Returns
    /// * Vector of all token IDs in the sequence's blocks
    pub fn get_sequence_tokens(&self, seq_id: u32) -> Result<Vec<u32>, String> {
        let table = self
            .block_table
            .get(seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        let num_tokens = table.num_tokens() as usize;
        let mut tokens = Vec::with_capacity(num_tokens);

        for block in table.blocks() {
            let block_tokens = self.token_tracker.get_block_tokens(block.block_id);
            tokens.extend_from_slice(block_tokens);
        }

        // Truncate to actual token count (last block may not be full)
        tokens.truncate(num_tokens);

        Ok(tokens)
    }

    /// Find blocks matching a token prefix
    ///
    /// Searches for cached blocks that contain the given token sequence.
    /// Useful for prefix caching - reuse KV cache for shared prompts.
    ///
    /// # Arguments
    /// * `tokens` - Token sequence to match
    ///
    /// # Returns
    /// * Vector of (block_id, match_length) for matching blocks
    pub fn find_prefix_matches(&self, tokens: &[u32]) -> Vec<(u32, u32)> {
        self.token_tracker.find_prefix_matches(tokens)
    }

    /// Find the longest matching prefix in the cache
    ///
    /// # Arguments
    /// * `tokens` - Token sequence to match
    ///
    /// # Returns
    /// * (matched_block_ids, total_matched_tokens)
    pub fn find_longest_prefix(&self, tokens: &[u32]) -> (Vec<u32>, u32) {
        self.token_tracker.find_longest_prefix(tokens)
    }

    /// Remove token tracking for a block
    ///
    /// Call this when a block is freed.
    pub fn untrack_block(&mut self, block_id: u32) {
        self.token_tracker.remove_block(block_id);
    }

    /// Get token tracking statistics
    pub fn get_token_tracker_stats(&self) -> crate::token_tracker::TokenTrackerStats {
        self.token_tracker.stats()
    }

    /// Get memory usage of token tracker in bytes
    pub fn get_token_tracker_memory(&self) -> usize {
        self.token_tracker.memory_usage()
    }

    /// Check if token tracking is enabled
    pub fn is_token_tracking_enabled(&self) -> bool {
        self.token_tracker.is_enabled()
    }

    // ========== End Token Tracking Methods ==========

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

    /// Get the configuration
    pub fn config(&self) -> &PagedAttentionConfig {
        &self.config
    }

    /// Get access to the engine manager (for PagedAttentionLayer)
    pub fn engine_manager(&self) -> &CacheEngineManager {
        &self.engine_manager
    }

    /// Build PagedAttentionInputMetadata for a batch of sequences
    ///
    /// This creates all the metadata needed for paged attention operations:
    /// block tables, context lengths, slot mappings, etc.
    ///
    /// # Arguments
    /// * `seq_ids` - Sequence IDs in the batch
    /// * `input_lens` - Number of input tokens per sequence (prompt len for prefill, 1 for decode)
    /// * `is_prefill` - Whether each sequence is in prefill phase
    ///
    /// # Returns
    /// * `PagedAttentionInputMetadata` ready for use with PagedAttentionLayer
    pub fn build_input_metadata(
        &self,
        seq_ids: &[u32],
        input_lens: &[u32],
        is_prefill: &[bool],
    ) -> Result<crate::input_metadata::PagedAttentionInputMetadata, String> {
        use crate::input_metadata::PagedAttentionInputMetadata;

        // Get context lengths
        let context_lens: Vec<i32> = seq_ids
            .iter()
            .map(|&id| {
                self.block_table
                    .get(id)
                    .map(|t| t.num_tokens() as i32)
                    .unwrap_or(0)
            })
            .collect();

        let max_context_len = context_lens.iter().copied().max().unwrap_or(0) as u32;

        // Build slot mappings
        let slot_mappings = self.get_slot_mapping_batch(
            seq_ids,
            &context_lens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            is_prefill,
            input_lens,
        )?;

        // Build block tables
        let block_tables = self.build_block_tables_batch(seq_ids)?;
        let max_blocks_per_seq = self.get_max_blocks_per_seq() as usize;

        // Flatten block tables
        let mut flat_block_tables: Vec<i32> =
            Vec::with_capacity(seq_ids.len() * max_blocks_per_seq);
        for table in &block_tables {
            for &block_id in table {
                flat_block_tables.push(block_id as i32);
            }
            // Pad to max_blocks_per_seq
            let pad_count = max_blocks_per_seq - table.len();
            flat_block_tables.extend(std::iter::repeat_n(0i32, pad_count));
        }

        // Determine if this is a prefill batch (all sequences in prefill)
        let batch_is_prefill = is_prefill.iter().all(|&p| p);

        Ok(PagedAttentionInputMetadata::new(
            flat_block_tables,
            context_lens,
            slot_mappings,
            max_context_len,
            batch_is_prefill,
            max_blocks_per_seq as u32,
        ))
    }

    /// Build input metadata for decode-only (simplified API)
    ///
    /// All sequences are assumed to be in decode phase (1 new token each).
    pub fn build_decode_metadata(
        &self,
        seq_ids: &[u32],
    ) -> Result<crate::input_metadata::PagedAttentionInputMetadata, String> {
        let input_lens = vec![1u32; seq_ids.len()];
        let is_prefill = vec![false; seq_ids.len()];
        self.build_input_metadata(seq_ids, &input_lens, &is_prefill)
    }

    /// Build input metadata for prefill (simplified API)
    ///
    /// All sequences are assumed to be in prefill phase.
    pub fn build_prefill_metadata(
        &self,
        seq_ids: &[u32],
        prompt_lens: &[u32],
    ) -> Result<crate::input_metadata::PagedAttentionInputMetadata, String> {
        let is_prefill = vec![true; seq_ids.len()];
        self.build_input_metadata(seq_ids, prompt_lens, &is_prefill)
    }

    /// Initialize the GPU cache buffers
    ///
    /// Must be called before using update() or attention().
    /// This allocates GPU memory for all layers' KV caches.
    #[cfg(target_os = "macos")]
    pub fn initialize(&mut self) -> Result<(), String> {
        self.engine_manager.initialize()
    }

    /// Update the KV cache with new keys and values
    ///
    /// This is the core operation that writes new K/V pairs into the paged cache.
    /// Uses the `reshape_and_cache` Metal kernel for GPU-accelerated updates.
    ///
    /// # Arguments
    /// * `layer_idx` - The transformer layer index (0 to num_layers-1)
    /// * `keys` - MLX array handle for keys [num_tokens, num_kv_heads, head_size]
    /// * `values` - MLX array handle for values [num_tokens, num_kv_heads, head_size]
    /// * `slot_mapping` - Slot indices for each token [num_tokens], i64
    ///
    /// # Safety
    /// - keys, values, slot_mapping must be valid MLX array pointers
    /// - Arrays must remain valid until this function returns
    /// - Cache must be initialized via initialize() before calling
    #[cfg(target_os = "macos")]
    pub unsafe fn update(
        &self,
        layer_idx: u32,
        keys: *mut mlx_sys::mlx_array,
        values: *mut mlx_sys::mlx_array,
        slot_mapping: *mut mlx_sys::mlx_array,
    ) -> Result<(), String> {
        use crate::metal::{
            MetalDtype, MlxMetalBuffer, RawBufferInfo, ReshapeAndCacheParams,
            dispatch_reshape_and_cache_raw, is_metal_extraction_supported, synchronize_mlx,
        };

        // Check Metal availability
        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }

        // Get the cache engine for this layer
        let engine = self
            .engine_manager
            .get_engine(layer_idx)
            .ok_or_else(|| format!("Layer {} not found", layer_idx))?;

        if !engine.is_initialized() {
            return Err(format!(
                "Layer {} cache not initialized. Call initialize() first.",
                layer_idx
            ));
        }

        let key_cache = engine
            .key_cache()
            .ok_or_else(|| "Key cache buffer not initialized".to_string())?;
        let value_cache = engine
            .value_cache()
            .ok_or_else(|| "Value cache buffer not initialized".to_string())?;

        // Synchronize MLX to ensure all operations are complete
        synchronize_mlx();

        // Extract Metal buffer info from MLX arrays
        // SAFETY: Caller guarantees these are valid MLX array pointers
        let key_info = unsafe { MlxMetalBuffer::from_mlx_array(keys) }
            .ok_or_else(|| "Failed to extract Metal buffer from keys".to_string())?;
        let value_info = unsafe { MlxMetalBuffer::from_mlx_array(values) }
            .ok_or_else(|| "Failed to extract Metal buffer from values".to_string())?;
        let slot_info = unsafe { MlxMetalBuffer::from_mlx_array(slot_mapping) }
            .ok_or_else(|| "Failed to extract Metal buffer from slot_mapping".to_string())?;

        // Calculate number of tokens from the key array
        let num_tokens = key_info.data_size
            / (self.config.num_kv_heads as usize * self.config.head_size as usize);

        // Prepare kernel parameters
        // x = 16 / sizeof(dtype): 8 for FP16, 16 for FP8
        let use_fp8 = self.config.use_fp8();
        let x = if use_fp8 { 16i32 } else { 8i32 };
        let stride = (self.config.num_kv_heads * self.config.head_size) as i32;

        // Get FP8 scales from scale_manager if enabled
        let (k_scale, v_scale) = if use_fp8 {
            self.scale_manager
                .as_ref()
                .map(|sm| (sm.k_scale(layer_idx), sm.v_scale(layer_idx)))
                .unwrap_or((1.0, 1.0))
        } else {
            (1.0, 1.0)
        };

        let params = ReshapeAndCacheParams {
            num_tokens: num_tokens as u32,
            num_heads: self.config.num_kv_heads,
            head_size: self.config.head_size,
            block_size: self.config.block_size,
            key_stride: stride,
            value_stride: stride,
            x,
            k_scale,
            v_scale,
        };

        // Prepare raw buffer info
        let key_raw = RawBufferInfo {
            ptr: key_info.buffer_ptr,
            offset: key_info.offset,
        };
        let value_raw = RawBufferInfo {
            ptr: value_info.buffer_ptr,
            offset: value_info.offset,
        };
        let slot_raw = RawBufferInfo {
            ptr: slot_info.buffer_ptr,
            offset: slot_info.offset,
        };

        // Determine dtype based on FP8 mode
        let dtype = if use_fp8 {
            MetalDtype::UChar
        } else {
            MetalDtype::Float16
        };

        // Dispatch the kernel
        // SAFETY: Buffer pointers are valid (extracted from MLX arrays above)
        unsafe {
            dispatch_reshape_and_cache_raw(
                &key_raw,
                &value_raw,
                key_cache,
                value_cache,
                &slot_raw,
                &params,
                dtype,
            )?;
        }

        Ok(())
    }

    /// Update the KV cache using slot mapping slice (convenience method)
    ///
    /// This variant creates the slot_mapping MLX array internally from a Rust slice.
    ///
    /// # Arguments
    /// * `layer_idx` - The transformer layer index
    /// * `keys` - MLX array handle for keys [num_tokens, num_kv_heads, head_size]
    /// * `values` - MLX array handle for values [num_tokens, num_kv_heads, head_size]
    /// * `slot_mapping` - Slot indices as a Rust slice
    ///
    /// # Safety
    /// - `keys` and `values` must be valid MLX array pointers
    /// - Arrays must remain valid until this function returns
    /// - Cache must be initialized via `initialize()` before calling
    #[cfg(target_os = "macos")]
    pub unsafe fn update_with_slots(
        &self,
        layer_idx: u32,
        keys: *mut mlx_sys::mlx_array,
        values: *mut mlx_sys::mlx_array,
        slot_mapping: &[i64],
    ) -> Result<(), String> {
        use crate::metal::{
            MetalDtype, MetalState, MlxMetalBuffer, RawBufferInfo, ReshapeAndCacheParams,
            dispatch_reshape_and_cache_raw, is_metal_extraction_supported, synchronize_mlx,
        };
        use metal::MTLResourceOptions;

        // Check Metal availability
        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }

        // Get the cache engine for this layer
        let engine = self
            .engine_manager
            .get_engine(layer_idx)
            .ok_or_else(|| format!("Layer {} not found", layer_idx))?;

        if !engine.is_initialized() {
            return Err(format!(
                "Layer {} cache not initialized. Call initialize() first.",
                layer_idx
            ));
        }

        let key_cache = engine
            .key_cache()
            .ok_or_else(|| "Key cache buffer not initialized".to_string())?;
        let value_cache = engine
            .value_cache()
            .ok_or_else(|| "Value cache buffer not initialized".to_string())?;

        // Synchronize MLX
        synchronize_mlx();

        // Extract Metal buffer info from MLX arrays
        // SAFETY: Caller guarantees these are valid MLX array pointers
        let key_info = unsafe { MlxMetalBuffer::from_mlx_array(keys) }
            .ok_or_else(|| "Failed to extract Metal buffer from keys".to_string())?;
        let value_info = unsafe { MlxMetalBuffer::from_mlx_array(values) }
            .ok_or_else(|| "Failed to extract Metal buffer from values".to_string())?;

        // Create slot_mapping Metal buffer from slice
        let state = MetalState::get()?;
        let slot_buffer = state.device.new_buffer_with_data(
            slot_mapping.as_ptr() as *const _,
            std::mem::size_of_val(slot_mapping) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let num_tokens = slot_mapping.len();

        // Prepare kernel parameters
        // x = 16 / sizeof(dtype): 8 for FP16, 16 for FP8
        let use_fp8 = self.config.use_fp8();
        let x = if use_fp8 { 16i32 } else { 8i32 };
        let stride = (self.config.num_kv_heads * self.config.head_size) as i32;

        // Get FP8 scales from scale_manager if enabled
        let (k_scale, v_scale) = if use_fp8 {
            self.scale_manager
                .as_ref()
                .map(|sm| (sm.k_scale(layer_idx), sm.v_scale(layer_idx)))
                .unwrap_or((1.0, 1.0))
        } else {
            (1.0, 1.0)
        };

        let params = ReshapeAndCacheParams {
            num_tokens: num_tokens as u32,
            num_heads: self.config.num_kv_heads,
            head_size: self.config.head_size,
            block_size: self.config.block_size,
            key_stride: stride,
            value_stride: stride,
            x,
            k_scale,
            v_scale,
        };

        // Prepare raw buffer info
        let key_raw = RawBufferInfo {
            ptr: key_info.buffer_ptr,
            offset: key_info.offset,
        };
        let value_raw = RawBufferInfo {
            ptr: value_info.buffer_ptr,
            offset: value_info.offset,
        };

        // For slot buffer, we need to get the raw pointer
        use metal::foreign_types::ForeignType;
        let slot_raw = RawBufferInfo {
            ptr: slot_buffer.as_ptr() as *mut _,
            offset: 0,
        };

        // Determine dtype based on FP8 mode
        let dtype = if use_fp8 {
            MetalDtype::UChar
        } else {
            MetalDtype::Float16
        };

        // Dispatch the kernel
        // SAFETY: Buffer pointers are valid (extracted from MLX arrays and created above)
        unsafe {
            dispatch_reshape_and_cache_raw(
                &key_raw,
                &value_raw,
                key_cache,
                value_cache,
                &slot_raw,
                &params,
                dtype,
            )?;
        }

        Ok(())
    }

    /// Run paged attention forward pass
    ///
    /// Computes attention using the paged KV cache. Automatically selects V1 or V2
    /// kernel based on the maximum context length.
    ///
    /// # Arguments
    /// * `layer_idx` - The transformer layer index (0 to num_layers-1)
    /// * `queries` - MLX array handle for queries [num_seqs, num_heads, head_size]
    /// * `seq_ids` - Sequence IDs in the batch
    /// * `num_query_heads` - Number of query heads (for GQA, >= num_kv_heads)
    /// * `scale` - Attention scale factor (typically 1/sqrt(head_size))
    ///
    /// # Returns
    /// * `PagedAttentionOutput` containing the output buffer and metadata
    ///
    /// # Safety
    /// - queries must be a valid MLX array pointer
    /// - Cache must be initialized via initialize() before calling
    /// - All seq_ids must be valid (added via add_sequence)
    #[cfg(target_os = "macos")]
    pub unsafe fn attention(
        &self,
        layer_idx: u32,
        queries: *mut mlx_sys::mlx_array,
        seq_ids: &[u32],
        num_query_heads: u32,
        scale: f32,
    ) -> Result<crate::metal::PagedAttentionOutput, String> {
        use crate::metal::{
            MetalDtype, MetalState, MlxMetalBuffer, PagedAttentionParams, RawBufferInfo,
            dispatch_paged_attention_auto, is_metal_extraction_supported, synchronize_mlx,
        };
        use metal::MTLResourceOptions;

        // Check Metal availability
        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }

        // Get the cache engine for this layer
        let engine = self
            .engine_manager
            .get_engine(layer_idx)
            .ok_or_else(|| format!("Layer {} not found", layer_idx))?;

        if !engine.is_initialized() {
            return Err(format!(
                "Layer {} cache not initialized. Call initialize() first.",
                layer_idx
            ));
        }

        let key_cache = engine
            .key_cache()
            .ok_or_else(|| "Key cache buffer not initialized".to_string())?;
        let value_cache = engine
            .value_cache()
            .ok_or_else(|| "Value cache buffer not initialized".to_string())?;

        // Synchronize MLX
        synchronize_mlx();

        // Extract query buffer
        let query_info = unsafe { MlxMetalBuffer::from_mlx_array(queries) }
            .ok_or_else(|| "Failed to extract Metal buffer from queries".to_string())?;

        // Get context lengths for the batch
        let context_lens: Vec<u32> = seq_ids
            .iter()
            .map(|&id| {
                self.block_table
                    .get(id)
                    .map(|t| t.num_tokens())
                    .unwrap_or(0)
            })
            .collect();

        let max_context_len = context_lens.iter().copied().max().unwrap_or(0);

        // Build block tables for the batch
        let block_tables = self.build_block_tables_batch(seq_ids)?;
        let max_blocks_per_seq = self.get_max_blocks_per_seq() as usize;

        // Flatten block tables into 2D array [num_seqs, max_blocks_per_seq]
        let mut flat_block_tables: Vec<i32> =
            Vec::with_capacity(seq_ids.len() * max_blocks_per_seq);
        for table in &block_tables {
            for &block_id in table {
                flat_block_tables.push(block_id as i32);
            }
            // Pad to max_blocks_per_seq
            let pad_count = max_blocks_per_seq - table.len();
            flat_block_tables.extend(std::iter::repeat_n(0i32, pad_count));
        }

        // Convert context_lens to i32
        let context_lens_i32: Vec<i32> = context_lens.iter().map(|&x| x as i32).collect();

        // Create Metal buffers for block_tables and context_lens
        let state = MetalState::get()?;

        let block_tables_buffer = state.device.new_buffer_with_data(
            flat_block_tables.as_ptr() as *const _,
            (flat_block_tables.len() * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let context_lens_buffer = state.device.new_buffer_with_data(
            context_lens_i32.as_ptr() as *const _,
            (context_lens_i32.len() * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Calculate strides for kernel
        let num_seqs = seq_ids.len() as u32;
        let q_stride = (num_query_heads * self.config.head_size) as i32;

        // KV cache layout: [num_blocks, num_kv_heads, head_size/x, block_size, x] for keys
        // kv_block_stride = num_kv_heads * head_size * block_size (elements per block per layer)
        // x = 16 / sizeof(half) = 8 for FP16
        let kv_block_stride =
            (self.config.num_kv_heads * self.config.head_size * self.config.block_size) as i32;
        let kv_head_stride = (self.config.head_size * self.config.block_size) as i32;

        // Get FP8 scales from scale_manager if enabled
        let use_fp8 = self.config.use_fp8();
        let (k_scale, v_scale) = if use_fp8 {
            self.scale_manager
                .as_ref()
                .map(|sm| (sm.k_scale(layer_idx), sm.v_scale(layer_idx)))
                .unwrap_or((1.0, 1.0))
        } else {
            (1.0, 1.0)
        };

        let params = PagedAttentionParams {
            num_seqs,
            num_heads: num_query_heads,
            num_kv_heads: self.config.num_kv_heads,
            head_size: self.config.head_size,
            block_size: self.config.block_size,
            max_seq_len: max_context_len,
            max_num_blocks_per_seq: max_blocks_per_seq as u32,
            scale,
            softcapping: 1.0, // No softcapping
            q_stride,
            kv_block_stride,
            kv_head_stride,
            k_scale,
            v_scale,
        };

        let query_raw = RawBufferInfo {
            ptr: query_info.buffer_ptr,
            offset: query_info.offset,
        };

        // Determine dtype based on config
        let dtype = if self.config.use_fp8() {
            MetalDtype::UChar
        } else {
            MetalDtype::Float16
        };

        // Dispatch the kernel
        let output = unsafe {
            dispatch_paged_attention_auto(
                &query_raw,
                key_cache,
                value_cache,
                &block_tables_buffer,
                &context_lens_buffer,
                max_context_len,
                &params,
                dtype,
            )?
        };

        Ok(output)
    }

    /// Run paged attention with explicit context lengths (convenience method)
    ///
    /// Use this when you already have context lengths computed.
    ///
    /// # Safety
    /// - `queries` must be a valid MLX array pointer
    /// - Cache must be initialized via `initialize()` before calling
    /// - All `seq_ids` must be valid (added via `add_sequence`)
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_with_context(
        &self,
        layer_idx: u32,
        queries: *mut mlx_sys::mlx_array,
        seq_ids: &[u32],
        context_lens: &[u32],
        num_query_heads: u32,
        scale: f32,
    ) -> Result<crate::metal::PagedAttentionOutput, String> {
        use crate::metal::{
            MetalDtype, MetalState, MlxMetalBuffer, PagedAttentionParams, RawBufferInfo,
            dispatch_paged_attention_auto, is_metal_extraction_supported, synchronize_mlx,
        };
        use metal::MTLResourceOptions;

        // Check Metal availability
        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }

        // Get the cache engine for this layer
        let engine = self
            .engine_manager
            .get_engine(layer_idx)
            .ok_or_else(|| format!("Layer {} not found", layer_idx))?;

        if !engine.is_initialized() {
            return Err(format!(
                "Layer {} cache not initialized. Call initialize() first.",
                layer_idx
            ));
        }

        let key_cache = engine
            .key_cache()
            .ok_or_else(|| "Key cache buffer not initialized".to_string())?;
        let value_cache = engine
            .value_cache()
            .ok_or_else(|| "Value cache buffer not initialized".to_string())?;

        // Synchronize MLX
        synchronize_mlx();

        // Extract query buffer
        let query_info = unsafe { MlxMetalBuffer::from_mlx_array(queries) }
            .ok_or_else(|| "Failed to extract Metal buffer from queries".to_string())?;

        let max_context_len = context_lens.iter().copied().max().unwrap_or(0);

        // Build block tables for the batch
        let block_tables = self.build_block_tables_batch(seq_ids)?;
        let max_blocks_per_seq = self.get_max_blocks_per_seq() as usize;

        // Flatten block tables
        let mut flat_block_tables: Vec<i32> =
            Vec::with_capacity(seq_ids.len() * max_blocks_per_seq);
        for table in &block_tables {
            for &block_id in table {
                flat_block_tables.push(block_id as i32);
            }
            let pad_count = max_blocks_per_seq - table.len();
            flat_block_tables.extend(std::iter::repeat_n(0i32, pad_count));
        }

        let context_lens_i32: Vec<i32> = context_lens.iter().map(|&x| x as i32).collect();

        // Create Metal buffers
        let state = MetalState::get()?;

        let block_tables_buffer = state.device.new_buffer_with_data(
            flat_block_tables.as_ptr() as *const _,
            (flat_block_tables.len() * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let context_lens_buffer = state.device.new_buffer_with_data(
            context_lens_i32.as_ptr() as *const _,
            (context_lens_i32.len() * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Calculate strides
        let num_seqs = seq_ids.len() as u32;
        let q_stride = (num_query_heads * self.config.head_size) as i32;
        let kv_block_stride =
            (self.config.num_kv_heads * self.config.head_size * self.config.block_size) as i32;
        let kv_head_stride = (self.config.head_size * self.config.block_size) as i32;

        // Get FP8 scales from scale_manager if enabled
        let use_fp8 = self.config.use_fp8();
        let (k_scale, v_scale) = if use_fp8 {
            self.scale_manager
                .as_ref()
                .map(|sm| (sm.k_scale(layer_idx), sm.v_scale(layer_idx)))
                .unwrap_or((1.0, 1.0))
        } else {
            (1.0, 1.0)
        };

        let params = PagedAttentionParams {
            num_seqs,
            num_heads: num_query_heads,
            num_kv_heads: self.config.num_kv_heads,
            head_size: self.config.head_size,
            block_size: self.config.block_size,
            max_seq_len: max_context_len,
            max_num_blocks_per_seq: max_blocks_per_seq as u32,
            scale,
            softcapping: 1.0,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            k_scale,
            v_scale,
        };

        let query_raw = RawBufferInfo {
            ptr: query_info.buffer_ptr,
            offset: query_info.offset,
        };

        // Determine dtype based on config
        let dtype = if self.config.use_fp8() {
            MetalDtype::UChar
        } else {
            MetalDtype::Float16
        };

        let output = unsafe {
            dispatch_paged_attention_auto(
                &query_raw,
                key_cache,
                value_cache,
                &block_tables_buffer,
                &context_lens_buffer,
                max_context_len,
                &params,
                dtype,
            )?
        };

        Ok(output)
    }

    /// Copy blocks for copy-on-write semantics (used in beam search)
    ///
    /// Copies data from source blocks to destination blocks using the Metal
    /// copy_blocks kernel. This is used during beam search when a sequence
    /// branches and we need to make independent copies of shared blocks.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index to copy blocks for
    /// * `block_mapping` - Pairs of (src_block_id, dst_block_id)
    ///
    /// # Example
    /// ```rust,ignore
    /// // Copy block 5 to block 10, and block 6 to block 11
    /// cache.copy_blocks(0, &[(5, 10), (6, 11)])?;
    /// ```
    #[cfg(target_os = "macos")]
    pub fn copy_blocks(&self, layer_idx: u32, block_mapping: &[(u32, u32)]) -> Result<(), String> {
        use crate::metal::{CopyBlocksParams, MetalDtype, MetalState, dispatch_copy_blocks};
        use metal::MTLResourceOptions;

        if block_mapping.is_empty() {
            return Ok(());
        }

        let engine = self
            .engine_manager
            .get_engine(layer_idx)
            .ok_or_else(|| format!("Layer {} not found", layer_idx))?;

        if !engine.is_initialized() {
            return Err(format!("Layer {} cache not initialized", layer_idx));
        }

        let key_cache = engine
            .key_cache()
            .ok_or_else(|| "Key cache not initialized".to_string())?;
        let value_cache = engine
            .value_cache()
            .ok_or_else(|| "Value cache not initialized".to_string())?;

        // Create block mapping buffer (flattened pairs as i64)
        let state = MetalState::get()?;
        let mapping_flat: Vec<i64> = block_mapping
            .iter()
            .flat_map(|(src, dst)| [*src as i64, *dst as i64])
            .collect();

        let mapping_buffer = state.device.new_buffer_with_data(
            mapping_flat.as_ptr() as *const _,
            (mapping_flat.len() * std::mem::size_of::<i64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params = CopyBlocksParams::from_config(
            block_mapping.len() as u32,
            self.config.num_kv_heads,
            self.config.head_size,
            self.config.block_size,
        );

        // Determine dtype based on FP8 config
        let dtype = if self.config.use_fp8() {
            MetalDtype::UChar
        } else {
            MetalDtype::Float16
        };

        dispatch_copy_blocks(key_cache, value_cache, &mapping_buffer, &params, dtype)
    }

    /// Fork a sequence for beam search
    ///
    /// Creates a new sequence that shares all blocks with the parent sequence.
    /// When either sequence is modified, copy-on-write semantics are used to
    /// create independent copies of modified blocks.
    ///
    /// # Arguments
    /// * `parent_seq_id` - Sequence ID to fork from
    ///
    /// # Returns
    /// * New sequence ID for the forked sequence
    ///
    /// # Example
    /// ```rust,ignore
    /// // During beam search, fork the best sequence
    /// let beam1_id = cache.fork_sequence(parent_id)?;
    /// let beam2_id = cache.fork_sequence(parent_id)?;
    /// ```
    pub fn fork_sequence(&mut self, parent_seq_id: u32) -> Result<u32, String> {
        // Get parent's block info - extract data before mutating
        let (parent_blocks, parent_tokens) = {
            let parent_table = self
                .block_table
                .get(parent_seq_id)
                .ok_or_else(|| format!("Parent sequence {} not found", parent_seq_id))?;

            (parent_table.blocks().to_vec(), parent_table.num_tokens())
        };

        // Create new sequence
        let child_id = self.block_table.add_sequence();

        // Share all blocks (increment reference counts)
        let child_table = self.block_table.get_mut(child_id).unwrap();
        for block in parent_blocks.iter() {
            block.incref();
            child_table.add_block(block.clone());
        }
        child_table.set_num_tokens(parent_tokens);

        Ok(child_id)
    }

    /// Handle copy-on-write for a sequence before writing to a block
    ///
    /// If the block at the given position is shared, allocates a new block
    /// and copies the data. Returns the block mapping for the Metal kernel.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID
    /// * `layer_idx` - Layer to copy
    /// * `block_idx` - Index of block in the sequence's block table
    ///
    /// # Returns
    /// * Some((src_block_id, dst_block_id)) if copy was needed
    /// * None if the block is not shared
    #[cfg(target_os = "macos")]
    pub fn handle_cow(
        &mut self,
        seq_id: u32,
        layer_idx: u32,
        block_idx: usize,
    ) -> Result<Option<(u32, u32)>, String> {
        // Check if block is shared
        let table = self
            .block_table
            .get(seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        let blocks = table.blocks();
        if block_idx >= blocks.len() {
            return Err(format!("Block index {} out of range", block_idx));
        }

        let block = &blocks[block_idx];
        if !block.is_shared() {
            return Ok(None);
        }

        // Block is shared, need to copy
        let src_block_id = block.block_id;

        // Allocate new block via copy_on_write
        let new_block = self
            .engine_manager
            .allocator_mut()
            .copy_on_write(block)
            .ok_or_else(|| "Failed to allocate block for copy-on-write".to_string())?;

        let dst_block_id = new_block.block_id;

        // Copy the data using Metal kernel
        self.copy_blocks(layer_idx, &[(src_block_id, dst_block_id)])?;

        // Update block table with new block
        let table = self.block_table.get_mut(seq_id).unwrap();
        if !table.replace_block(block_idx, new_block) {
            return Err(format!(
                "Failed to replace block at index {} for sequence {} (out of bounds)",
                block_idx, seq_id
            ));
        }

        Ok(Some((src_block_id, dst_block_id)))
    }

    /// Swap blocks out to CPU memory
    ///
    /// Copies the specified blocks from GPU to CPU memory to free up
    /// GPU memory for other sequences. The blocks can be swapped back
    /// in later with `swap_in`.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer to swap blocks from
    /// * `block_ids` - Block IDs to swap out
    /// * `cpu_cache` - CPU cache to store swapped blocks
    ///
    /// # Example
    /// ```rust,ignore
    /// // Swap out blocks 10, 11, 12 from layer 0
    /// cache.swap_out(0, &[10, 11, 12], &mut cpu_cache)?;
    /// ```
    #[cfg(target_os = "macos")]
    pub fn swap_out(
        &self,
        layer_idx: u32,
        block_ids: &[u32],
        cpu_cache: &mut crate::metal::CpuBlockCache,
    ) -> Result<(), String> {
        use crate::metal::{SwapBlocksParams, swap_out as metal_swap_out};

        if block_ids.is_empty() {
            return Ok(());
        }

        let engine = self
            .engine_manager
            .get_engine(layer_idx)
            .ok_or_else(|| format!("Layer {} not found", layer_idx))?;

        if !engine.is_initialized() {
            return Err(format!("Layer {} cache not initialized", layer_idx));
        }

        let key_cache = engine
            .key_cache()
            .ok_or_else(|| "Key cache not initialized".to_string())?;
        let value_cache = engine
            .value_cache()
            .ok_or_else(|| "Value cache not initialized".to_string())?;

        let params = SwapBlocksParams {
            num_kv_heads: self.config.num_kv_heads,
            head_size: self.config.head_size,
            block_size: self.config.block_size,
            use_fp8: self.config.use_fp8(),
        };

        metal_swap_out(key_cache, value_cache, block_ids, cpu_cache, &params)
    }

    /// Swap blocks back in from CPU memory
    ///
    /// Copies the specified blocks from CPU back to GPU memory.
    /// The blocks are removed from the cpu_cache after swap-in.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer to swap blocks into
    /// * `block_ids` - Block IDs to swap in
    /// * `cpu_cache` - CPU cache containing swapped blocks
    ///
    /// # Example
    /// ```rust,ignore
    /// // Swap in blocks 10, 11, 12 to layer 0
    /// cache.swap_in(0, &[10, 11, 12], &mut cpu_cache)?;
    /// ```
    #[cfg(target_os = "macos")]
    pub fn swap_in(
        &self,
        layer_idx: u32,
        block_ids: &[u32],
        cpu_cache: &mut crate::metal::CpuBlockCache,
    ) -> Result<(), String> {
        use crate::metal::{SwapBlocksParams, swap_in as metal_swap_in};

        if block_ids.is_empty() {
            return Ok(());
        }

        let engine = self
            .engine_manager
            .get_engine(layer_idx)
            .ok_or_else(|| format!("Layer {} not found", layer_idx))?;

        if !engine.is_initialized() {
            return Err(format!("Layer {} cache not initialized", layer_idx));
        }

        let key_cache = engine
            .key_cache()
            .ok_or_else(|| "Key cache not initialized".to_string())?;
        let value_cache = engine
            .value_cache()
            .ok_or_else(|| "Value cache not initialized".to_string())?;

        let params = SwapBlocksParams {
            num_kv_heads: self.config.num_kv_heads,
            head_size: self.config.head_size,
            block_size: self.config.block_size,
            use_fp8: self.config.use_fp8(),
        };

        metal_swap_in(key_cache, value_cache, block_ids, cpu_cache, &params)
    }

    /// Create a new CPU block cache for swap operations
    ///
    /// # Returns
    /// * A new `CpuBlockCache` configured for this cache's block size
    #[cfg(target_os = "macos")]
    pub fn create_cpu_cache(&self) -> crate::metal::CpuBlockCache {
        crate::metal::CpuBlockCache::new(
            self.config.num_kv_heads,
            self.config.head_size,
            self.config.block_size,
        )
    }
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

    #[test]
    fn test_fork_sequence() {
        let config = PagedAttentionConfig {
            block_size: 32,
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 28,
            ..Default::default()
        };

        let mut cache = PagedKVCache::new(config).unwrap();

        // Add a parent sequence with 64 tokens (2 blocks)
        let parent_id = cache.add_sequence(64).unwrap();
        assert_eq!(cache.num_sequences(), 1);
        assert_eq!(cache.get_memory_stats().allocated_blocks, 2);

        // Fork the sequence
        let child_id = cache.fork_sequence(parent_id).unwrap();
        assert_eq!(cache.num_sequences(), 2);

        // Still only 2 blocks allocated (shared)
        assert_eq!(cache.get_memory_stats().allocated_blocks, 2);

        // Both sequences should have same context length
        let context_lens = cache.get_context_lens();
        assert_eq!(context_lens.len(), 2);
        assert_eq!(context_lens[0], context_lens[1]);

        // Remove parent - blocks should still be allocated (child has refs)
        cache.remove_sequence(parent_id).unwrap();
        assert_eq!(cache.num_sequences(), 1);
        assert_eq!(cache.get_memory_stats().allocated_blocks, 2);

        // Remove child - blocks should be freed
        cache.remove_sequence(child_id).unwrap();
        assert_eq!(cache.num_sequences(), 0);
        assert_eq!(cache.get_memory_stats().allocated_blocks, 0);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_fp8_scale_integration() {
        // Test with FP8 disabled - scale manager should be None
        let config_fp16 = PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 4,
            use_fp8_cache: Some(false),
            ..Default::default()
        };

        let cache_fp16 = PagedKVCache::new(config_fp16).unwrap();
        assert!(!cache_fp16.is_fp8_calibrated());
        assert_eq!(cache_fp16.get_layer_scales(0), (1.0, 1.0));
        assert!(cache_fp16.get_scale_stats().is_none());

        // Test with FP8 enabled - scale manager should be initialized
        let config_fp8 = PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 4,
            use_fp8_cache: Some(true),
            ..Default::default()
        };

        let mut cache_fp8 = PagedKVCache::new(config_fp8).unwrap();

        // Default scales should be 1.0
        assert_eq!(cache_fp8.get_layer_scales(0), (1.0, 1.0));
        assert_eq!(cache_fp8.get_layer_scales(1), (1.0, 1.0));
        assert_eq!(cache_fp8.get_layer_scales(2), (1.0, 1.0));
        assert_eq!(cache_fp8.get_layer_scales(3), (1.0, 1.0));

        // Set scales manually
        cache_fp8.set_layer_scales(0, 0.5, 0.25);
        cache_fp8.set_layer_scales(1, 2.0, 1.5);

        assert_eq!(cache_fp8.get_layer_scales(0), (0.5, 0.25));
        assert_eq!(cache_fp8.get_layer_scales(1), (2.0, 1.5));

        // Check stats - all 4 layers have scales (init_default_scales + 2 overwritten)
        let stats = cache_fp8.get_scale_stats().unwrap();
        assert_eq!(stats.num_layers_calibrated, 4);

        // Test load/save scales
        let (k_scales, v_scales) = cache_fp8.get_all_scales().unwrap();
        assert_eq!(k_scales.len(), 4);
        assert_eq!(v_scales.len(), 4);
        assert_eq!(k_scales[0], 0.5);
        assert_eq!(v_scales[0], 0.25);
        assert_eq!(k_scales[1], 2.0);
        assert_eq!(v_scales[1], 1.5);
    }

    #[test]
    fn test_token_tracking_integration() {
        let config = PagedAttentionConfig {
            block_size: 8, // Must be 8, 16, or 32
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 2,
            ..Default::default()
        };

        let mut cache = PagedKVCache::new(config).unwrap();
        assert!(cache.is_token_tracking_enabled());

        // Add a sequence with 24 tokens (allocates 3 blocks of size 8)
        let tokens: Vec<u32> = vec![
            100, 101, 102, 103, 104, 105, 106, 107, // Block 0
            200, 201, 202, 203, 204, 205, 206, 207, // Block 1
            300, 301, 302, 303, 304, 305, 306, 307, // Block 2
        ];
        let seq_id = cache.add_sequence(tokens.len() as u32).unwrap();

        // Track tokens for the sequence
        cache.track_sequence_tokens(seq_id, &tokens, 0).unwrap();

        // Verify tokens are tracked
        let seq_tokens = cache.get_sequence_tokens(seq_id).unwrap();
        assert_eq!(seq_tokens, tokens);

        // Add another sequence with 16 tokens and same first block
        let tokens2: Vec<u32> = vec![
            100, 101, 102, 103, 104, 105, 106, 107, // Same as seq1 block 0
            500, 501, 502, 503, 504, 505, 506, 507, // Different block
        ];
        let seq_id2 = cache.add_sequence(tokens2.len() as u32).unwrap();

        // Track tokens with same first block
        cache.track_sequence_tokens(seq_id2, &tokens2, 0).unwrap();

        // Find prefix matches for the common prefix (first block)
        // Note: prefix_index stores only one block per hash, so we get the most recent one
        let matches = cache.find_prefix_matches(&[100, 101, 102, 103, 104, 105, 106, 107]);
        assert_eq!(matches.len(), 1); // Returns the most recently tracked block with this hash

        // Find longest prefix for a new sequence
        let test_tokens: Vec<u32> = vec![
            100, 101, 102, 103, 104, 105, 106, 107, // Match block 0
            200, 201, 202, 203, 204, 205, 206, 207, // Match block 1
            999, 999, 999, 999, 999, 999, 999, 999, // No match
        ];
        let (matching_blocks, matched_tokens) = cache.find_longest_prefix(&test_tokens);
        assert_eq!(matched_tokens, 16); // First 2 blocks match seq1
        assert_eq!(matching_blocks.len(), 2);

        // Check memory usage
        let memory = cache.get_token_tracker_memory();
        assert!(memory > 0);

        // Check stats
        let stats = cache.get_token_tracker_stats();
        assert!(stats.num_tracked_blocks > 0);
        assert!(stats.total_tracked_tokens >= 40); // 24 + 16 tokens

        // Remove a sequence and verify untracking works
        cache.remove_sequence(seq_id).unwrap();

        // seq1's unique blocks should be untracked, but shared prefix block remains
        let seq2_tokens = cache.get_sequence_tokens(seq_id2).unwrap();
        assert_eq!(seq2_tokens, tokens2);
    }

    #[test]
    fn test_token_tracking_prefix_caching_workflow() {
        let config = PagedAttentionConfig {
            block_size: 8, // Must be 8, 16, or 32
            gpu_memory_mb: 1024,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 2,
            ..Default::default()
        };

        let mut cache = PagedKVCache::new(config).unwrap();

        // Simulate a system prompt that many requests will share (24 tokens = 3 blocks)
        let system_prompt_tokens: Vec<u32> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, // Block 0
            9, 10, 11, 12, 13, 14, 15, 16, // Block 1
            17, 18, 19, 20, 21, 22, 23, 24, // Block 2
        ];

        // First request with system prompt + user message (32 tokens = 4 blocks)
        let request1_tokens: Vec<u32> = system_prompt_tokens
            .iter()
            .chain(&[100, 101, 102, 103, 104, 105, 106, 107]) // User message (8 tokens)
            .copied()
            .collect();
        let seq1 = cache.add_sequence(request1_tokens.len() as u32).unwrap();
        cache
            .track_sequence_tokens(seq1, &request1_tokens, 0)
            .unwrap();

        // Second request comes in - check for prefix cache hit
        let request2_prefix = &system_prompt_tokens;
        let (matching_blocks, num_matched_tokens) = cache.find_longest_prefix(request2_prefix);

        // Should find all 3 system prompt blocks
        assert_eq!(num_matched_tokens, 24);
        assert_eq!(matching_blocks.len(), 3);

        // In a real implementation, we could now:
        // 1. Copy the matching blocks to the new sequence (using copy_blocks)
        // 2. Only compute attention for the non-matched tokens

        // Verify the matching blocks contain the right tokens
        for &block_id in &matching_blocks {
            let block_tokens = cache.get_block_tokens(block_id);
            assert!(!block_tokens.is_empty());
        }
    }
}
