//! Token tracking for logical block management
//!
//! This module tracks which tokens are stored in each block of the KV cache.
//! This enables:
//! - **Prefix Caching**: Match incoming prompts against cached token sequences
//! - **Debugging**: Inspect what tokens are in each block
//! - **Eviction Decisions**: Make informed decisions based on block content
//!
//! ## Architecture
//!
//! The `TokenTracker` maintains:
//! - Per-block token ID storage
//! - Hash-based prefix index for O(1) lookups
//! - Sequence-to-block mapping for efficient queries
//!
//! ## Memory Overhead
//!
//! For each block, we store `block_size` token IDs (4 bytes each).
//! Example: 32 block size × 10K blocks = 1.28 MB overhead.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Token tracker for logical block management
///
/// Tracks token IDs stored in each physical block and provides
/// efficient prefix matching for cache reuse.
#[derive(Debug)]
pub struct TokenTracker {
    /// Token IDs per block: block_id -> Vec<token_id>
    block_tokens: HashMap<u32, Vec<u32>>,

    /// Prefix hash index: hash -> (block_id, num_tokens)
    /// Maps a hash of tokens to the block containing them
    prefix_index: HashMap<u64, (u32, u32)>,

    /// Block size (max tokens per block)
    block_size: u32,

    /// Whether tracking is enabled
    enabled: bool,
}

impl TokenTracker {
    /// Create a new token tracker
    ///
    /// # Arguments
    /// * `block_size` - Maximum tokens per block
    pub fn new(block_size: u32) -> Self {
        Self {
            block_tokens: HashMap::new(),
            prefix_index: HashMap::new(),
            block_size,
            enabled: true,
        }
    }

    /// Create a disabled token tracker (no-op)
    pub fn disabled() -> Self {
        Self {
            block_tokens: HashMap::new(),
            prefix_index: HashMap::new(),
            block_size: 0,
            enabled: false,
        }
    }

    /// Check if tracking is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record tokens being stored in a block
    ///
    /// # Arguments
    /// * `block_id` - Physical block ID
    /// * `tokens` - Token IDs being stored
    /// * `start_offset` - Starting position within the block
    ///
    /// # Returns
    /// * Hash of the block's tokens (for prefix caching integration)
    pub fn track_tokens(&mut self, block_id: u32, tokens: &[u32], start_offset: u32) -> u64 {
        if !self.enabled {
            return 0;
        }

        // Compute old hash before modifying (to remove stale entry from prefix_index)
        let old_hash = self
            .block_tokens
            .get(&block_id)
            .filter(|t| !t.is_empty())
            .map(|t| Self::hash_tokens(t));

        // Get or create token vector for this block
        let block_tokens = self
            .block_tokens
            .entry(block_id)
            .or_insert_with(|| Vec::with_capacity(self.block_size as usize));

        // Ensure vector is long enough for the offset
        if block_tokens.len() < start_offset as usize {
            block_tokens.resize(start_offset as usize, 0);
        }

        // Insert or extend tokens
        for (i, &token) in tokens.iter().enumerate() {
            let pos = start_offset as usize + i;
            if pos < block_tokens.len() {
                block_tokens[pos] = token;
            } else {
                block_tokens.push(token);
            }
        }

        // Compute hash and num_tokens while we have the borrow
        let hash = Self::hash_tokens(block_tokens);
        let num_tokens = block_tokens.len() as u32;

        // Remove stale prefix_index entry if hash changed and belongs to this block
        if let Some(old) = old_hash
            && old != hash
            && self.prefix_index.get(&old).map(|(id, _)| *id) == Some(block_id)
        {
            self.prefix_index.remove(&old);
        }

        // Update prefix index with new hash
        self.prefix_index.insert(hash, (block_id, num_tokens));

        hash
    }

    /// Get tokens stored in a block
    ///
    /// # Arguments
    /// * `block_id` - Physical block ID
    ///
    /// # Returns
    /// * Token IDs in the block, or empty slice if not tracked
    pub fn get_block_tokens(&self, block_id: u32) -> &[u32] {
        self.block_tokens
            .get(&block_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get the number of tokens in a block
    pub fn get_block_token_count(&self, block_id: u32) -> u32 {
        self.block_tokens
            .get(&block_id)
            .map(|v| v.len() as u32)
            .unwrap_or(0)
    }

    /// Remove tracking for a block
    ///
    /// Call this when a block is freed.
    pub fn remove_block(&mut self, block_id: u32) {
        if !self.enabled {
            return;
        }

        // Compute hash before removing (for prefix index cleanup)
        // Only remove from prefix_index if the entry belongs to this block
        if self.block_tokens.contains_key(&block_id) {
            let hash = self.compute_block_hash(block_id);
            if self.prefix_index.get(&hash).map(|(id, _)| *id) == Some(block_id) {
                self.prefix_index.remove(&hash);
            }
        }

        self.block_tokens.remove(&block_id);
    }

    /// Copy tokens from one block to another
    ///
    /// Used during copy-on-write operations.
    pub fn copy_block(&mut self, src_block_id: u32, dst_block_id: u32) {
        if !self.enabled {
            return;
        }

        // Remove stale prefix_index entry for destination block if it exists and belongs to it
        if let Some(dst_tokens) = self.block_tokens.get(&dst_block_id)
            && !dst_tokens.is_empty()
        {
            let old_hash = Self::hash_tokens(dst_tokens);
            if self.prefix_index.get(&old_hash).map(|(id, _)| *id) == Some(dst_block_id) {
                self.prefix_index.remove(&old_hash);
            }
        }

        if let Some(src_tokens) = self.block_tokens.get(&src_block_id) {
            let tokens = src_tokens.clone();
            let hash = Self::hash_tokens(&tokens);
            let num_tokens = tokens.len() as u32;

            self.block_tokens.insert(dst_block_id, tokens);
            self.prefix_index.insert(hash, (dst_block_id, num_tokens));
        }
    }

    /// Find blocks that match a token prefix
    ///
    /// # Arguments
    /// * `tokens` - Token sequence to match
    ///
    /// # Returns
    /// * Vector of (block_id, match_length) for blocks matching the prefix
    pub fn find_prefix_matches(&self, tokens: &[u32]) -> Vec<(u32, u32)> {
        if !self.enabled || tokens.is_empty() {
            return Vec::new();
        }

        let mut matches = Vec::new();

        // Try matching full blocks first
        let full_blocks = tokens.len() / self.block_size as usize;

        for block_idx in 0..=full_blocks {
            let start = block_idx * self.block_size as usize;
            let end = std::cmp::min(start + self.block_size as usize, tokens.len());

            if start >= tokens.len() {
                break;
            }

            let block_tokens = &tokens[start..end];
            let hash = Self::hash_tokens(block_tokens);

            if let Some(&(block_id, num_tokens)) = self.prefix_index.get(&hash) {
                // Verify the match (hash collision protection)
                if let Some(stored) = self.block_tokens.get(&block_id)
                    && stored.len() >= block_tokens.len()
                    && &stored[..block_tokens.len()] == block_tokens
                {
                    matches.push((block_id, num_tokens));
                }
            }
        }

        matches
    }

    /// Find the longest matching prefix
    ///
    /// # Arguments
    /// * `tokens` - Token sequence to match
    ///
    /// # Returns
    /// * (matched_blocks, total_matched_tokens) - Blocks and total tokens matched
    pub fn find_longest_prefix(&self, tokens: &[u32]) -> (Vec<u32>, u32) {
        if !self.enabled || tokens.is_empty() {
            return (Vec::new(), 0);
        }

        let mut matched_blocks = Vec::new();
        let mut total_matched = 0u32;
        let mut pos = 0usize;

        while pos < tokens.len() {
            let remaining = tokens.len() - pos;
            let block_len = std::cmp::min(self.block_size as usize, remaining);
            let block_tokens = &tokens[pos..pos + block_len];
            let hash = Self::hash_tokens(block_tokens);

            match self.prefix_index.get(&hash) {
                Some(&(block_id, num_tokens)) => {
                    // Verify match
                    if let Some(stored) = self.block_tokens.get(&block_id)
                        && stored.len() >= block_tokens.len()
                        && &stored[..block_tokens.len()] == block_tokens
                    {
                        matched_blocks.push(block_id);
                        total_matched += num_tokens;
                        pos += num_tokens as usize;
                        continue;
                    }
                    // No match, stop searching
                    break;
                }
                None => break,
            }
        }

        (matched_blocks, total_matched)
    }

    /// Compute hash for a block's tokens
    fn compute_block_hash(&self, block_id: u32) -> u64 {
        match self.block_tokens.get(&block_id) {
            Some(tokens) => Self::hash_tokens(tokens),
            None => 0,
        }
    }

    /// Hash a token sequence using FNV-1a
    fn hash_tokens(tokens: &[u32]) -> u64 {
        let mut hasher = FnvHasher::new();
        for &token in tokens {
            token.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Get statistics about tracked blocks
    pub fn stats(&self) -> TokenTrackerStats {
        let num_blocks = self.block_tokens.len() as u32;
        let total_tokens: usize = self.block_tokens.values().map(|v| v.len()).sum();
        let avg_tokens_per_block = if num_blocks > 0 {
            total_tokens as f32 / num_blocks as f32
        } else {
            0.0
        };

        TokenTrackerStats {
            num_tracked_blocks: num_blocks,
            total_tracked_tokens: total_tokens as u32,
            avg_tokens_per_block,
            prefix_index_size: self.prefix_index.len() as u32,
        }
    }

    /// Clear all tracking data
    pub fn clear(&mut self) {
        self.block_tokens.clear();
        self.prefix_index.clear();
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let tokens_mem: usize = self
            .block_tokens
            .values()
            .map(|v| v.capacity() * std::mem::size_of::<u32>())
            .sum();
        let index_mem = self.prefix_index.capacity()
            * (std::mem::size_of::<u64>() + std::mem::size_of::<(u32, u32)>());

        tokens_mem + index_mem
    }
}

/// Statistics about token tracking
#[derive(Debug, Clone)]
pub struct TokenTrackerStats {
    /// Number of blocks being tracked
    pub num_tracked_blocks: u32,
    /// Total tokens tracked across all blocks
    pub total_tracked_tokens: u32,
    /// Average tokens per block
    pub avg_tokens_per_block: f32,
    /// Size of prefix index
    pub prefix_index_size: u32,
}

/// FNV-1a hasher for token sequences
///
/// Chosen for its simplicity and good distribution for integer sequences.
struct FnvHasher {
    state: u64,
}

impl FnvHasher {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self {
            state: Self::FNV_OFFSET,
        }
    }
}

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(Self::FNV_PRIME);
        }
    }
}

/// Compute hash for a sequence of token IDs
///
/// This is a standalone function for use with the existing BlockAllocator prefix cache.
pub fn compute_token_hash(tokens: &[u32]) -> u64 {
    let mut hasher = FnvHasher::new();
    for &token in tokens {
        token.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_tracker_creation() {
        let tracker = TokenTracker::new(32);
        assert!(tracker.is_enabled());
        assert_eq!(tracker.block_size, 32);

        let disabled = TokenTracker::disabled();
        assert!(!disabled.is_enabled());
    }

    #[test]
    fn test_track_tokens() {
        let mut tracker = TokenTracker::new(32);

        // Track some tokens
        let tokens = vec![1, 2, 3, 4, 5];
        let hash = tracker.track_tokens(0, &tokens, 0);

        assert!(hash != 0);
        assert_eq!(tracker.get_block_tokens(0), &[1, 2, 3, 4, 5]);
        assert_eq!(tracker.get_block_token_count(0), 5);
    }

    #[test]
    fn test_track_tokens_with_offset() {
        let mut tracker = TokenTracker::new(32);

        // Track initial tokens
        tracker.track_tokens(0, &[1, 2, 3], 0);

        // Track more tokens at offset
        tracker.track_tokens(0, &[4, 5], 3);

        assert_eq!(tracker.get_block_tokens(0), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_remove_block() {
        let mut tracker = TokenTracker::new(32);

        tracker.track_tokens(0, &[1, 2, 3], 0);
        tracker.track_tokens(1, &[4, 5, 6], 0);

        assert_eq!(tracker.get_block_token_count(0), 3);
        assert_eq!(tracker.get_block_token_count(1), 3);

        tracker.remove_block(0);

        assert_eq!(tracker.get_block_token_count(0), 0);
        assert_eq!(tracker.get_block_token_count(1), 3);
    }

    #[test]
    fn test_copy_block() {
        let mut tracker = TokenTracker::new(32);

        tracker.track_tokens(0, &[1, 2, 3, 4, 5], 0);
        tracker.copy_block(0, 1);

        assert_eq!(tracker.get_block_tokens(0), &[1, 2, 3, 4, 5]);
        assert_eq!(tracker.get_block_tokens(1), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_find_prefix_matches() {
        let mut tracker = TokenTracker::new(4);

        // Track blocks with different tokens
        tracker.track_tokens(0, &[1, 2, 3, 4], 0);
        tracker.track_tokens(1, &[5, 6, 7, 8], 0);
        tracker.track_tokens(2, &[1, 2, 3, 4], 0); // Duplicate content

        // Find match for first block's content
        let matches = tracker.find_prefix_matches(&[1, 2, 3, 4]);
        // Should find block 2 (last one with this content in prefix_index)
        assert!(!matches.is_empty());
        assert_eq!(matches[0].1, 4); // 4 tokens matched
    }

    #[test]
    fn test_find_longest_prefix() {
        let mut tracker = TokenTracker::new(4);

        // Track a sequence of blocks
        tracker.track_tokens(0, &[1, 2, 3, 4], 0);
        tracker.track_tokens(1, &[5, 6, 7, 8], 0);

        // Find longest prefix for a sequence that matches both blocks
        let (blocks, total) = tracker.find_longest_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // Should match both blocks (8 tokens total)
        assert_eq!(blocks.len(), 2);
        assert_eq!(total, 8);

        // Test partial match - only first block matches
        let (blocks2, total2) = tracker.find_longest_prefix(&[1, 2, 3, 4, 99, 99, 99, 99]);
        assert_eq!(blocks2.len(), 1);
        assert_eq!(total2, 4);

        // Test no match
        let (blocks3, total3) = tracker.find_longest_prefix(&[99, 98, 97, 96]);
        assert_eq!(blocks3.len(), 0);
        assert_eq!(total3, 0);
    }

    #[test]
    fn test_stats() {
        let mut tracker = TokenTracker::new(32);

        tracker.track_tokens(0, &[1, 2, 3], 0);
        tracker.track_tokens(1, &[4, 5, 6, 7], 0);

        let stats = tracker.stats();
        assert_eq!(stats.num_tracked_blocks, 2);
        assert_eq!(stats.total_tracked_tokens, 7);
        assert!((stats.avg_tokens_per_block - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_disabled_tracker() {
        let mut tracker = TokenTracker::disabled();

        // All operations should be no-ops
        let hash = tracker.track_tokens(0, &[1, 2, 3], 0);
        assert_eq!(hash, 0);
        assert_eq!(tracker.get_block_token_count(0), 0);

        let matches = tracker.find_prefix_matches(&[1, 2, 3]);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_hash_consistency() {
        // Same tokens should produce same hash
        let tokens = vec![1, 2, 3, 4, 5];
        let hash1 = compute_token_hash(&tokens);
        let hash2 = compute_token_hash(&tokens);
        assert_eq!(hash1, hash2);

        // Different tokens should produce different hash
        let tokens2 = vec![1, 2, 3, 4, 6];
        let hash3 = compute_token_hash(&tokens2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_memory_usage() {
        let mut tracker = TokenTracker::new(32);

        let initial = tracker.memory_usage();
        tracker.track_tokens(0, &[1, 2, 3, 4, 5], 0);
        let after = tracker.memory_usage();

        assert!(after >= initial);
    }

    #[test]
    fn test_clear() {
        let mut tracker = TokenTracker::new(32);

        tracker.track_tokens(0, &[1, 2, 3], 0);
        tracker.track_tokens(1, &[4, 5, 6], 0);

        assert_eq!(tracker.stats().num_tracked_blocks, 2);

        tracker.clear();

        assert_eq!(tracker.stats().num_tracked_blocks, 0);
    }

    #[test]
    fn test_shared_hash_after_copy_block_track_tokens() {
        // Regression test: When blocks share the same hash after copy_block,
        // modifying one block should not remove the other's prefix_index entry.
        let mut tracker = TokenTracker::new(4);

        // Block 0 has tokens [1, 2, 3, 4]
        tracker.track_tokens(0, &[1, 2, 3, 4], 0);

        // Copy block 0 to block 1 - now both have same tokens and hash
        // prefix_index now points to block 1 (last writer wins)
        tracker.copy_block(0, 1);

        // Verify block 1 is in the prefix_index
        let matches_before = tracker.find_prefix_matches(&[1, 2, 3, 4]);
        assert!(!matches_before.is_empty());
        assert_eq!(matches_before[0].0, 1); // Block 1 should be indexed

        // Now modify block 0 with different tokens
        // This should NOT remove block 1's entry from prefix_index
        tracker.track_tokens(0, &[5, 6, 7, 8], 0);

        // Block 1's prefix_index entry should still exist
        let matches_after = tracker.find_prefix_matches(&[1, 2, 3, 4]);
        assert!(
            !matches_after.is_empty(),
            "Block 1's prefix_index entry was incorrectly removed"
        );
        assert_eq!(matches_after[0].0, 1); // Block 1 should still be indexed
    }

    #[test]
    fn test_shared_hash_after_copy_block_remove_block() {
        // Regression test: When blocks share the same hash after copy_block,
        // removing one block should not remove the other's prefix_index entry.
        let mut tracker = TokenTracker::new(4);

        // Block 0 has tokens [1, 2, 3, 4]
        tracker.track_tokens(0, &[1, 2, 3, 4], 0);

        // Copy block 0 to block 1
        // prefix_index now points to block 1
        tracker.copy_block(0, 1);

        // Verify block 1 is indexed
        let matches_before = tracker.find_prefix_matches(&[1, 2, 3, 4]);
        assert!(!matches_before.is_empty());
        assert_eq!(matches_before[0].0, 1);

        // Remove block 0 - this should NOT remove block 1's entry
        tracker.remove_block(0);

        // Block 1's prefix_index entry should still exist
        let matches_after = tracker.find_prefix_matches(&[1, 2, 3, 4]);
        assert!(
            !matches_after.is_empty(),
            "Block 1's prefix_index entry was incorrectly removed"
        );
        assert_eq!(matches_after[0].0, 1);
    }

    #[test]
    fn test_shared_hash_copy_block_overwrites() {
        // When copying to a block that has different tokens, the old entry
        // for the destination block should be removed (if it owns the entry).
        let mut tracker = TokenTracker::new(4);

        // Block 0 has tokens [1, 2, 3, 4]
        tracker.track_tokens(0, &[1, 2, 3, 4], 0);

        // Block 1 has tokens [5, 6, 7, 8]
        tracker.track_tokens(1, &[5, 6, 7, 8], 0);

        // Both blocks should be findable
        let matches0 = tracker.find_prefix_matches(&[1, 2, 3, 4]);
        let matches1 = tracker.find_prefix_matches(&[5, 6, 7, 8]);
        assert!(!matches0.is_empty());
        assert!(!matches1.is_empty());

        // Copy block 0 to block 1 - block 1's old entry should be removed
        tracker.copy_block(0, 1);

        // [5, 6, 7, 8] should no longer be findable
        let matches_old = tracker.find_prefix_matches(&[5, 6, 7, 8]);
        assert!(
            matches_old.is_empty(),
            "Block 1's old entry should have been removed"
        );

        // [1, 2, 3, 4] should now point to block 1
        let matches_new = tracker.find_prefix_matches(&[1, 2, 3, 4]);
        assert!(!matches_new.is_empty());
        assert_eq!(matches_new[0].0, 1);
    }
}
