//! Input metadata for paged attention operations
//!
//! This module provides the `PagedAttentionInputMetadata` struct which encapsulates
//! all the information needed to perform paged attention: block tables, context lengths,
//! slot mappings, and phase information (prefill vs decode).

#[cfg(target_os = "macos")]
use metal::Buffer;

/// Input metadata for paged attention operations
///
/// Contains all the information needed by the paged attention kernel to locate
/// KV cache blocks and compute attention correctly.
///
/// # Usage
///
/// ```typescript
/// // Build metadata for a batch of sequences
/// const metadata = pagedKVCache.buildInputMetadata(
///   seqIds,
///   inputLens,
///   isPrefill
/// );
///
/// // Use for attention
/// const output = pagedAttnLayer.forward(queries, keys, values, metadata);
/// ```
#[derive(Clone, Debug)]
pub struct PagedAttentionInputMetadata {
    /// Block tables for each sequence: [num_seqs, max_blocks_per_seq]
    /// Maps sequence positions to physical cache block IDs.
    pub block_tables: Vec<i32>,

    /// Context length for each sequence (total tokens in KV cache)
    pub context_lens: Vec<i32>,

    /// Slot mapping: maps each input token to its cache slot
    /// Shape: [total_num_tokens]
    pub slot_mappings: Vec<i64>,

    /// Maximum context length across all sequences in the batch
    pub max_context_len: u32,

    /// Whether this is the first chunk of a prompt (prefill phase)
    /// During prefill, we use standard attention; during decode, we use paged attention.
    pub is_prefill: bool,

    /// Number of sequences in the batch
    pub num_seqs: u32,

    /// Maximum blocks per sequence (for array indexing)
    pub max_blocks_per_seq: u32,

    // Metal buffers (cached for kernel dispatch)
    #[cfg(target_os = "macos")]
    block_tables_buffer: Option<Buffer>,

    #[cfg(target_os = "macos")]
    context_lens_buffer: Option<Buffer>,

    #[cfg(target_os = "macos")]
    slot_mappings_buffer: Option<Buffer>,
}

impl PagedAttentionInputMetadata {
    /// Create new input metadata
    ///
    /// # Arguments
    /// * `block_tables` - Flattened block tables [num_seqs * max_blocks_per_seq]
    /// * `context_lens` - Context length for each sequence
    /// * `slot_mappings` - Slot indices for input tokens
    /// * `max_context_len` - Maximum context length
    /// * `is_prefill` - Whether in prefill phase
    /// * `max_blocks_per_seq` - Maximum blocks per sequence
    pub fn new(
        block_tables: Vec<i32>,
        context_lens: Vec<i32>,
        slot_mappings: Vec<i64>,
        max_context_len: u32,
        is_prefill: bool,
        max_blocks_per_seq: u32,
    ) -> Self {
        let num_seqs = context_lens.len() as u32;
        Self {
            block_tables,
            context_lens,
            slot_mappings,
            max_context_len,
            is_prefill,
            num_seqs,
            max_blocks_per_seq,
            #[cfg(target_os = "macos")]
            block_tables_buffer: None,
            #[cfg(target_os = "macos")]
            context_lens_buffer: None,
            #[cfg(target_os = "macos")]
            slot_mappings_buffer: None,
        }
    }

    /// Create a dummy metadata for testing/profiling
    ///
    /// This creates minimal metadata that won't be used for actual decoding.
    pub fn dummy() -> Self {
        Self {
            block_tables: vec![0],
            context_lens: vec![1],
            slot_mappings: vec![0],
            max_context_len: 1,
            is_prefill: true,
            num_seqs: 1,
            max_blocks_per_seq: 1,
            #[cfg(target_os = "macos")]
            block_tables_buffer: None,
            #[cfg(target_os = "macos")]
            context_lens_buffer: None,
            #[cfg(target_os = "macos")]
            slot_mappings_buffer: None,
        }
    }

    /// Get the number of sequences
    pub fn num_seqs(&self) -> u32 {
        self.num_seqs
    }

    /// Get the total number of input tokens
    pub fn num_tokens(&self) -> usize {
        self.slot_mappings.len()
    }

    /// Check if this is a prefill phase
    pub fn is_prefill(&self) -> bool {
        self.is_prefill
    }

    /// Get block tables as slice
    pub fn block_tables(&self) -> &[i32] {
        &self.block_tables
    }

    /// Get context lengths as slice
    pub fn context_lens(&self) -> &[i32] {
        &self.context_lens
    }

    /// Get slot mappings as slice
    pub fn slot_mappings(&self) -> &[i64] {
        &self.slot_mappings
    }

    /// Ensure Metal buffers are created and return them
    ///
    /// This lazily creates Metal buffers from the CPU data.
    /// The buffers are cached for reuse.
    #[cfg(target_os = "macos")]
    pub fn get_metal_buffers(&mut self) -> Result<(&Buffer, &Buffer, &Buffer), String> {
        use crate::metal::MetalState;
        use metal::MTLResourceOptions;

        // Create block tables buffer if not exists
        if self.block_tables_buffer.is_none() {
            let state = MetalState::get()?;
            let buffer = state.device.new_buffer_with_data(
                self.block_tables.as_ptr() as *const _,
                (self.block_tables.len() * std::mem::size_of::<i32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.block_tables_buffer = Some(buffer);
        }

        // Create context lens buffer if not exists
        if self.context_lens_buffer.is_none() {
            let state = MetalState::get()?;
            let buffer = state.device.new_buffer_with_data(
                self.context_lens.as_ptr() as *const _,
                (self.context_lens.len() * std::mem::size_of::<i32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.context_lens_buffer = Some(buffer);
        }

        // Create slot mappings buffer if not exists
        if self.slot_mappings_buffer.is_none() {
            let state = MetalState::get()?;
            let buffer = state.device.new_buffer_with_data(
                self.slot_mappings.as_ptr() as *const _,
                (self.slot_mappings.len() * std::mem::size_of::<i64>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.slot_mappings_buffer = Some(buffer);
        }

        Ok((
            self.block_tables_buffer.as_ref().unwrap(),
            self.context_lens_buffer.as_ref().unwrap(),
            self.slot_mappings_buffer.as_ref().unwrap(),
        ))
    }

    /// Invalidate cached Metal buffers
    ///
    /// Call this when the underlying data changes.
    #[cfg(target_os = "macos")]
    pub fn invalidate_buffers(&mut self) {
        self.block_tables_buffer = None;
        self.context_lens_buffer = None;
        self.slot_mappings_buffer = None;
    }
}

/// Builder for PagedAttentionInputMetadata
///
/// Provides a fluent API for constructing input metadata.
#[derive(Default)]
pub struct PagedAttentionInputMetadataBuilder {
    block_tables: Vec<i32>,
    context_lens: Vec<i32>,
    slot_mappings: Vec<i64>,
    max_context_len: u32,
    is_prefill: bool,
    max_blocks_per_seq: u32,
}

impl PagedAttentionInputMetadataBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set block tables
    pub fn block_tables(mut self, tables: Vec<i32>) -> Self {
        self.block_tables = tables;
        self
    }

    /// Set context lengths
    pub fn context_lens(mut self, lens: Vec<i32>) -> Self {
        self.context_lens = lens;
        self
    }

    /// Set slot mappings
    pub fn slot_mappings(mut self, mappings: Vec<i64>) -> Self {
        self.slot_mappings = mappings;
        self
    }

    /// Set maximum context length
    pub fn max_context_len(mut self, len: u32) -> Self {
        self.max_context_len = len;
        self
    }

    /// Set prefill flag
    pub fn is_prefill(mut self, is_prefill: bool) -> Self {
        self.is_prefill = is_prefill;
        self
    }

    /// Set maximum blocks per sequence
    pub fn max_blocks_per_seq(mut self, max_blocks: u32) -> Self {
        self.max_blocks_per_seq = max_blocks;
        self
    }

    /// Build the metadata
    pub fn build(self) -> PagedAttentionInputMetadata {
        PagedAttentionInputMetadata::new(
            self.block_tables,
            self.context_lens,
            self.slot_mappings,
            self.max_context_len,
            self.is_prefill,
            self.max_blocks_per_seq,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_metadata_creation() {
        let metadata = PagedAttentionInputMetadata::new(
            vec![0, 1, 2, 3], // block tables
            vec![32, 64],     // context lens
            vec![0, 1, 2],    // slot mappings
            64,               // max context len
            false,            // is_prefill
            2,                // max_blocks_per_seq
        );

        assert_eq!(metadata.num_seqs(), 2);
        assert_eq!(metadata.num_tokens(), 3);
        assert!(!metadata.is_prefill());
        assert_eq!(metadata.max_context_len, 64);
    }

    #[test]
    fn test_builder() {
        let metadata = PagedAttentionInputMetadataBuilder::new()
            .block_tables(vec![0, 1])
            .context_lens(vec![16])
            .slot_mappings(vec![0])
            .max_context_len(16)
            .is_prefill(true)
            .max_blocks_per_seq(2)
            .build();

        assert_eq!(metadata.num_seqs(), 1);
        assert!(metadata.is_prefill());
    }

    #[test]
    fn test_dummy() {
        let metadata = PagedAttentionInputMetadata::dummy();
        assert_eq!(metadata.num_seqs(), 1);
        assert!(metadata.is_prefill());
    }
}
