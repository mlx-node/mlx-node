//! Block table for mapping logical blocks to physical blocks
//!
//! Each sequence maintains a list of logical blocks (0, 1, 2, ...)
//! that map to physical blocks in the KV cache.

use crate::block_allocator::PhysicalBlock;
use std::sync::Arc;

/// Block table entry for a single sequence
#[derive(Debug)]
pub struct SequenceBlockTable {
    /// Sequence ID
    pub seq_id: u32,

    /// List of physical blocks in order
    /// blocks[i] contains KV cache for tokens [i*block_size, (i+1)*block_size)
    blocks: Vec<Arc<PhysicalBlock>>,

    /// Number of tokens currently in the sequence
    num_tokens: u32,

    /// Block size (tokens per block)
    block_size: u32,
}

impl SequenceBlockTable {
    /// Create a new block table for a sequence
    pub fn new(seq_id: u32, block_size: u32) -> Self {
        Self {
            seq_id,
            blocks: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    /// Add a block to the sequence
    pub fn add_block(&mut self, block: Arc<PhysicalBlock>) {
        self.blocks.push(block);
    }

    /// Replace a block at the given index (for copy-on-write)
    ///
    /// # Returns
    /// * `true` if the block was successfully replaced
    /// * `false` if the index was out of bounds
    pub fn replace_block(&mut self, index: usize, block: Arc<PhysicalBlock>) -> bool {
        if index < self.blocks.len() {
            self.blocks[index] = block;
            true
        } else {
            false
        }
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the number of tokens
    pub fn num_tokens(&self) -> u32 {
        self.num_tokens
    }

    /// Set the number of tokens
    pub fn set_num_tokens(&mut self, num_tokens: u32) {
        self.num_tokens = num_tokens;
    }

    /// Get the blocks
    pub fn blocks(&self) -> &[Arc<PhysicalBlock>] {
        &self.blocks
    }

    /// Get mutable access to blocks
    pub fn blocks_mut(&mut self) -> &mut Vec<Arc<PhysicalBlock>> {
        &mut self.blocks
    }

    /// Get the block IDs as a vector (for kernel dispatch)
    pub fn block_ids(&self) -> Vec<u32> {
        self.blocks.iter().map(|b| b.block_id).collect()
    }

    /// Get the last block (for appending new tokens)
    pub fn last_block(&self) -> Option<&Arc<PhysicalBlock>> {
        self.blocks.last()
    }

    /// Get mutable access to the last block
    pub fn last_block_mut(&mut self) -> Option<&mut Arc<PhysicalBlock>> {
        self.blocks.last_mut()
    }

    /// Check if the last block is full
    pub fn is_last_block_full(&self) -> bool {
        let tokens_in_last_block = self.num_tokens % self.block_size;
        tokens_in_last_block == 0 && self.num_tokens > 0
    }

    /// Get the number of free slots in the last block
    pub fn free_slots_in_last_block(&self) -> u32 {
        if self.blocks.is_empty() {
            0
        } else {
            let tokens_in_last_block = self.num_tokens % self.block_size;
            if tokens_in_last_block == 0 && self.num_tokens > 0 {
                0 // Block is full
            } else {
                self.block_size - tokens_in_last_block
            }
        }
    }

    /// Calculate how many new blocks are needed for additional tokens
    pub fn blocks_needed(&self, new_tokens: u32) -> u32 {
        let free_slots = self.free_slots_in_last_block();
        if new_tokens <= free_slots {
            0
        } else {
            let remaining = new_tokens - free_slots;
            remaining.div_ceil(self.block_size)
        }
    }

    /// Calculate the slot index for a given token position
    pub fn slot_index(&self, token_pos: u32) -> (u32, u32) {
        let block_idx = token_pos / self.block_size;
        let offset_in_block = token_pos % self.block_size;
        (block_idx, offset_in_block)
    }

    /// Calculate the absolute slot index for kernel dispatch
    pub fn absolute_slot_index(&self, token_pos: u32) -> Option<i64> {
        let (block_idx, offset_in_block) = self.slot_index(token_pos);
        if (block_idx as usize) < self.blocks.len() {
            let block_id = self.blocks[block_idx as usize].block_id;
            Some(block_id as i64 * self.block_size as i64 + offset_in_block as i64)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_allocator::BlockAllocator;

    #[test]
    fn test_sequence_block_table() {
        let mut allocator = BlockAllocator::new(10, 32);
        let mut table = SequenceBlockTable::new(0, 32);

        // Initially empty
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.num_tokens(), 0);

        // Add first block
        let block = allocator.allocate().unwrap();
        table.add_block(block);
        assert_eq!(table.num_blocks(), 1);

        // Add tokens
        table.set_num_tokens(20);
        assert!(!table.is_last_block_full());
        assert_eq!(table.free_slots_in_last_block(), 12);
        assert_eq!(table.blocks_needed(10), 0); // Fits in current block
        assert_eq!(table.blocks_needed(15), 1); // Needs one more block

        // Fill the block
        table.set_num_tokens(32);
        assert!(table.is_last_block_full());
        assert_eq!(table.free_slots_in_last_block(), 0);
    }

    #[test]
    fn test_slot_index() {
        let table = SequenceBlockTable::new(0, 32);

        assert_eq!(table.slot_index(0), (0, 0));
        assert_eq!(table.slot_index(31), (0, 31));
        assert_eq!(table.slot_index(32), (1, 0));
        assert_eq!(table.slot_index(50), (1, 18));
    }
}
