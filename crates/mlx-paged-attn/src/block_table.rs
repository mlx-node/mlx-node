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

/// Global block table managing all sequences
pub struct BlockTable {
    /// Per-sequence block tables
    tables: Vec<SequenceBlockTable>,

    /// Mapping from sequence ID to table index
    seq_id_to_idx: std::collections::HashMap<u32, usize>,

    /// Next sequence ID to assign
    next_seq_id: u32,

    /// Block size
    block_size: u32,
}

impl BlockTable {
    /// Create a new block table
    pub fn new(block_size: u32) -> Self {
        Self {
            tables: Vec::new(),
            seq_id_to_idx: std::collections::HashMap::new(),
            next_seq_id: 0,
            block_size,
        }
    }

    /// Add a new sequence and return its ID
    pub fn add_sequence(&mut self) -> u32 {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let table = SequenceBlockTable::new(seq_id, self.block_size);
        let idx = self.tables.len();
        self.tables.push(table);
        self.seq_id_to_idx.insert(seq_id, idx);

        seq_id
    }

    /// Remove a sequence
    pub fn remove_sequence(&mut self, seq_id: u32) -> Option<SequenceBlockTable> {
        if let Some(&idx) = self.seq_id_to_idx.get(&seq_id) {
            self.seq_id_to_idx.remove(&seq_id);

            // Swap remove and update the swapped element's index
            let table = self.tables.swap_remove(idx);

            // Update index of the swapped element (if any)
            if idx < self.tables.len() {
                let swapped_seq_id = self.tables[idx].seq_id;
                self.seq_id_to_idx.insert(swapped_seq_id, idx);
            }

            Some(table)
        } else {
            None
        }
    }

    /// Get a sequence's block table
    pub fn get(&self, seq_id: u32) -> Option<&SequenceBlockTable> {
        self.seq_id_to_idx
            .get(&seq_id)
            .map(|&idx| &self.tables[idx])
    }

    /// Get mutable access to a sequence's block table
    pub fn get_mut(&mut self, seq_id: u32) -> Option<&mut SequenceBlockTable> {
        if let Some(&idx) = self.seq_id_to_idx.get(&seq_id) {
            Some(&mut self.tables[idx])
        } else {
            None
        }
    }

    /// Get all sequence IDs
    pub fn seq_ids(&self) -> Vec<u32> {
        self.tables.iter().map(|t| t.seq_id).collect()
    }

    /// Get the number of sequences
    pub fn num_sequences(&self) -> usize {
        self.tables.len()
    }

    /// Get all block tables
    pub fn all_tables(&self) -> &[SequenceBlockTable] {
        &self.tables
    }

    /// Build the block tables tensor for kernel dispatch
    /// Returns a 2D array [num_seqs, max_blocks_per_seq]
    pub fn build_block_tables_array(&self, max_blocks_per_seq: usize) -> Vec<u32> {
        let num_seqs = self.tables.len();
        let mut result = vec![0u32; num_seqs * max_blocks_per_seq];

        for (seq_idx, table) in self.tables.iter().enumerate() {
            for (block_idx, block) in table.blocks().iter().enumerate() {
                if block_idx < max_blocks_per_seq {
                    result[seq_idx * max_blocks_per_seq + block_idx] = block.block_id;
                }
            }
        }

        result
    }

    /// Build context lengths array for kernel dispatch
    pub fn build_context_lens_array(&self) -> Vec<u32> {
        self.tables.iter().map(|t| t.num_tokens()).collect()
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

    #[test]
    fn test_block_table() {
        let mut table = BlockTable::new(32);

        let seq1 = table.add_sequence();
        let seq2 = table.add_sequence();

        assert_eq!(table.num_sequences(), 2);
        assert!(table.get(seq1).is_some());
        assert!(table.get(seq2).is_some());

        table.remove_sequence(seq1);
        assert_eq!(table.num_sequences(), 1);
        assert!(table.get(seq1).is_none());
        assert!(table.get(seq2).is_some());
    }
}
