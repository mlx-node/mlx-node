import { describe, it, expect } from 'vite-plus/test';
import { MxArray } from '@mlx-node/core';
import { shape, assertShape, int32 } from '../test-utils';

// TODO: PagedKVCache and PagedAttentionConfig are not yet exported from @mlx-node/core
// These types are currently internal to the mlx-paged-attn Rust crate.
// Once NAPI bindings are added, uncomment the import below:
// import { PagedKVCache, PagedAttentionConfig } from '@mlx-node/core';

/**
 * Test suite for PagedKVCache
 *
 * These tests are currently skipped because PagedKVCache is not yet exposed
 * via NAPI bindings. The tests demonstrate the expected API once bindings are added.
 *
 * To enable these tests:
 * 1. Add #[napi] annotations to PagedKVCache and PagedAttentionConfig in mlx-core
 * 2. Rebuild with `yarn build:native`
 * 3. Export types from packages/core/src/index.ts
 * 4. Remove the .skip from describe.skip below
 */

describe.skip('PagedKVCache', () => {
  describe('Constructor and Configuration', () => {
    it('should create cache with valid config', () => {
      // Example usage once NAPI bindings are added:
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      // expect(cache).toBeDefined();

      expect(true).toBe(true);
    });

    it('should validate block size (must be 8, 16, or 32)', () => {
      // Invalid block size should throw
      // const config = {
      //   blockSize: 64, // Invalid
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // expect(() => new PagedKVCache(config)).toThrow(/block_size/i);

      expect(true).toBe(true);
    });

    it('should validate head size (must be valid Metal kernel size)', () => {
      // Valid head sizes: 32, 64, 80, 96, 112, 120, 128, 192, 256
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 100, // Invalid
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // expect(() => new PagedKVCache(config)).toThrow(/head_size/i);

      expect(true).toBe(true);
    });

    it('should calculate number of blocks from memory budget', () => {
      // const config = {
      //   blockSize: 32,
      //   gpuMemoryMb: 1024, // 1 GB
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 28,
      // };
      // const cache = new PagedKVCache(config);
      //
      // Expected calculation:
      // bytes_per_block_per_layer = 2 * 4 * 128 * 32 * 2 = 65,536 bytes
      // bytes_per_block = 65,536 * 28 = 1,835,008 bytes
      // num_blocks = 1024 * 1024 * 1024 / 1,835,008 ≈ 585 blocks
      //
      // const stats = cache.getMemoryStats();
      // expect(stats.totalBlocks).toBeGreaterThan(500);
      // expect(stats.totalBlocks).toBeLessThan(700);

      expect(true).toBe(true);
    });

    it('should support FP8 cache for 2x memory efficiency', () => {
      // const configFp16 = {
      //   blockSize: 32,
      //   gpuMemoryMb: 1024,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 28,
      //   useFp8Cache: false,
      // };
      //
      // const configFp8 = {
      //   ...configFp16,
      //   useFp8Cache: true,
      // };
      //
      // const cacheFp16 = new PagedKVCache(configFp16);
      // const cacheFp8 = new PagedKVCache(configFp8);
      //
      // const statsFp16 = cacheFp16.getMemoryStats();
      // const statsFp8 = cacheFp8.getMemoryStats();
      //
      // // FP8 should provide ~2x blocks (1 byte vs 2 bytes per element)
      // expect(statsFp8.totalBlocks).toBeGreaterThanOrEqual(statsFp16.totalBlocks * 2 - 1);

      expect(true).toBe(true);
    });
  });

  describe('Sequence Management', () => {
    it('should add and remove sequences', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // // Add sequence with 100-token prompt
      // const seqId = cache.addSequence(100);
      // expect(seqId).toBeGreaterThanOrEqual(0);
      //
      // // Verify sequence was added
      // const stats = cache.getMemoryStats();
      // expect(stats.numFreeBlocks).toBeLessThan(stats.totalBlocks);
      //
      // // Remove sequence
      // cache.removeSequence(seqId);
      //
      // // Verify blocks were freed
      // const statsAfter = cache.getMemoryStats();
      // expect(statsAfter.numFreeBlocks).toBe(stats.totalBlocks);

      expect(true).toBe(true);
    });

    it('should allocate correct number of blocks for sequence', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // // 100 tokens with blockSize=16 needs ceil(100/16) = 7 blocks
      // const initialStats = cache.getMemoryStats();
      // const seqId = cache.addSequence(100);
      // const afterStats = cache.getMemoryStats();
      //
      // expect(initialStats.numFreeBlocks - afterStats.numFreeBlocks).toBe(7);

      expect(true).toBe(true);
    });

    it('should fail when insufficient memory', () => {
      // Small cache with limited blocks
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 32, // Very small
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // // Try to allocate more tokens than available
      // expect(() => cache.addSequence(10000)).toThrow(/not enough memory/i);

      expect(true).toBe(true);
    });

    it('should support multiple concurrent sequences', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // const seqId1 = cache.addSequence(50);
      // const seqId2 = cache.addSequence(75);
      // const seqId3 = cache.addSequence(100);
      //
      // expect(seqId1).not.toBe(seqId2);
      // expect(seqId2).not.toBe(seqId3);
      //
      // // Remove middle sequence
      // cache.removeSequence(seqId2);
      //
      // // Should still be able to use other sequences
      // expect(() => cache.removeSequence(seqId1)).not.toThrow();
      // expect(() => cache.removeSequence(seqId3)).not.toThrow();

      expect(true).toBe(true);
    });

    it('should check can_allocate before adding sequence', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // // Check if we can allocate 100 tokens (7 blocks)
      // const blocksNeeded = Math.ceil(100 / config.blockSize);
      // expect(cache.canAllocate(blocksNeeded)).toBe(true);
      //
      // // Fill cache
      // while (cache.canAllocate(blocksNeeded)) {
      //   cache.addSequence(100);
      // }
      //
      // // Should no longer be able to allocate
      // expect(cache.canAllocate(blocksNeeded)).toBe(false);

      expect(true).toBe(true);
    });
  });

  describe('Block Management', () => {
    it('should get slot mapping for sequences', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // const seqId = cache.addSequence(100);
      //
      // // Slot mapping maps token positions to physical block slots
      // const slotMapping = cache.getSlotMapping([seqId]);
      // expect(slotMapping).toBeInstanceOf(Int32Array);
      // expect(slotMapping.length).toBeGreaterThan(0);

      expect(true).toBe(true);
    });

    it('should build block tables for batch', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // const seqId1 = cache.addSequence(50);
      // const seqId2 = cache.addSequence(75);
      //
      // // Block table: [batch_size, max_blocks_per_seq]
      // const blockTables = cache.buildBlockTables([seqId1, seqId2]);
      // expect(blockTables).toBeInstanceOf(Int32Array);
      //
      // // Should have entries for both sequences
      // const maxBlocks = Math.max(
      //   Math.ceil(50 / config.blockSize),
      //   Math.ceil(75 / config.blockSize)
      // );
      // expect(blockTables.length).toBe(2 * maxBlocks);

      expect(true).toBe(true);
    });

    it('should extend sequence when generating new tokens', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // // Start with 16 tokens (exactly 1 block)
      // const seqId = cache.addSequence(16);
      // const initialStats = cache.getMemoryStats();
      //
      // // Extend by 1 token (should allocate new block)
      // cache.extendSequence(seqId, 1);
      // const afterStats = cache.getMemoryStats();
      //
      // // Should have allocated 1 more block
      // expect(initialStats.numFreeBlocks - afterStats.numFreeBlocks).toBe(1);

      expect(true).toBe(true);
    });
  });

  describe('Memory Statistics', () => {
    it('should track total blocks correctly', () => {
      // const config = {
      //   blockSize: 32,
      //   gpuMemoryMb: 1024,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 28,
      // };
      // const cache = new PagedKVCache(config);
      //
      // const stats = cache.getMemoryStats();
      // expect(stats.totalBlocks).toBeGreaterThan(0);
      // expect(stats.numFreeBlocks).toBe(stats.totalBlocks);
      // expect(stats.numUsedBlocks).toBe(0);

      expect(true).toBe(true);
    });

    it('should update used/free blocks when allocating', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // const initialStats = cache.getMemoryStats();
      // const seqId = cache.addSequence(100);
      // const afterStats = cache.getMemoryStats();
      //
      // const blocksAllocated = Math.ceil(100 / config.blockSize);
      // expect(afterStats.numUsedBlocks).toBe(blocksAllocated);
      // expect(afterStats.numFreeBlocks).toBe(initialStats.numFreeBlocks - blocksAllocated);
      // expect(afterStats.totalBlocks).toBe(initialStats.totalBlocks);

      expect(true).toBe(true);
    });

    it('should restore free blocks when removing sequence', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // const initialStats = cache.getMemoryStats();
      // const seqId = cache.addSequence(100);
      // cache.removeSequence(seqId);
      // const finalStats = cache.getMemoryStats();
      //
      // expect(finalStats.numFreeBlocks).toBe(initialStats.numFreeBlocks);
      // expect(finalStats.numUsedBlocks).toBe(0);

      expect(true).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('should handle single-token sequences', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // // Single token still needs 1 block
      // const seqId = cache.addSequence(1);
      // const stats = cache.getMemoryStats();
      // expect(stats.numUsedBlocks).toBe(1);

      expect(true).toBe(true);
    });

    it('should handle sequences exactly matching block size', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // // Exactly 3 blocks worth
      // const seqId = cache.addSequence(48);
      // const stats = cache.getMemoryStats();
      // expect(stats.numUsedBlocks).toBe(3);

      expect(true).toBe(true);
    });

    it('should handle very long sequences', () => {
      // const config = {
      //   blockSize: 32,
      //   gpuMemoryMb: 2048, // Larger cache
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // // 4096 tokens = 128 blocks
      // const seqId = cache.addSequence(4096);
      // const stats = cache.getMemoryStats();
      // expect(stats.numUsedBlocks).toBe(128);

      expect(true).toBe(true);
    });

    it('should throw on invalid sequence ID', () => {
      // const config = {
      //   blockSize: 16,
      //   gpuMemoryMb: 512,
      //   headSize: 128,
      //   numKvHeads: 4,
      //   numLayers: 2,
      // };
      // const cache = new PagedKVCache(config);
      //
      // expect(() => cache.removeSequence(999)).toThrow(/not found/i);

      expect(true).toBe(true);
    });
  });
});
