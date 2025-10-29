import { describe, it, expect } from 'vitest';
import { BatchKVCache, MxArray } from '@mlx-node/core';
import { shape, assertShape, int32 } from '../test-utils';

describe('BatchKVCache', () => {
  describe('Constructor and Basic Operations', () => {
    it('should create cache with left padding', () => {
      const cache = new BatchKVCache(int32(1, 3, 0));
      expect(cache).toBeDefined();
      expect(cache.getIdx()).toBe(0);
      expect(Array.from(cache.getLeftPadding())).toEqual([1, 3, 0]);
      expect(cache.getOffsets()).toEqual([-1, -3, 0]); // Negative due to padding
    });

    it('should create cache with uniform padding', () => {
      const cache = new BatchKVCache(int32(2, 2, 2));
      expect(Array.from(cache.getLeftPadding())).toEqual([2, 2, 2]);
      expect(cache.getOffsets()).toEqual([-2, -2, -2]);
    });

    it('should create cache with no padding', () => {
      const cache = new BatchKVCache(int32(0, 0, 0));
      expect(Array.from(cache.getLeftPadding())).toEqual([0, 0, 0]);
      expect(cache.getOffsets()).toEqual([0, 0, 0]);
    });
  });

  describe('Update and Fetch Operations', () => {
    it('should handle first update with left padding', () => {
      const cache = new BatchKVCache(int32(1, 2, 0));

      // Create keys and values: (batch=3, n_kv_heads=2, seq_len=4, head_dim=8)
      const keys = MxArray.randomNormal(shape(3, 2, 4, 8), 0, 1);
      const values = MxArray.randomNormal(shape(3, 2, 4, 8), 0, 1);

      const result = cache.updateAndFetch(keys, values);

      expect(result).toHaveLength(2);
      expect(cache.getIdx()).toBe(4);

      // Offsets account for left padding and new tokens
      expect(cache.getOffsets()).toEqual([3, 2, 4]); // -1+4, -2+4, 0+4

      // Verify shapes
      assertShape(result[0], [3, 2, 4, 8]);
      assertShape(result[1], [3, 2, 4, 8]);
    });

    it('should concatenate subsequent updates', () => {
      const cache = new BatchKVCache(int32(0, 1, 2));

      // First update: 5 tokens
      const keys1 = MxArray.randomNormal(shape(3, 4, 5, 16), 0, 1);
      const values1 = MxArray.randomNormal(shape(3, 4, 5, 16), 0, 1);
      cache.updateAndFetch(keys1, values1);

      expect(cache.getIdx()).toBe(5);

      // Second update: 3 more tokens
      const keys2 = MxArray.randomNormal(shape(3, 4, 3, 16), 0, 1);
      const values2 = MxArray.randomNormal(shape(3, 4, 3, 16), 0, 1);
      const result = cache.updateAndFetch(keys2, values2);

      expect(cache.getIdx()).toBe(8); // 5 + 3
      expect(cache.getOffsets()).toEqual([8, 7, 6]); // 0+8, -1+8, -2+8

      // Verify concatenated shapes
      assertShape(result[0], [3, 4, 8, 16]);
      assertShape(result[1], [3, 4, 8, 16]);
    });

    it('should handle single-token updates (autoregressive generation)', () => {
      const cache = new BatchKVCache(int32(0, 0));

      // Initial prompt: 10 tokens
      const keys1 = MxArray.randomNormal(shape(2, 8, 10, 32), 0, 1);
      const values1 = MxArray.randomNormal(shape(2, 8, 10, 32), 0, 1);
      cache.updateAndFetch(keys1, values1);

      // Generate 5 tokens one by one
      for (let i = 0; i < 5; i++) {
        const key_token = MxArray.randomNormal(shape(2, 8, 1, 32), 0, 1);
        const value_token = MxArray.randomNormal(shape(2, 8, 1, 32), 0, 1);
        const result = cache.updateAndFetch(key_token, value_token);

        expect(cache.getIdx()).toBe(10 + i + 1);
        assertShape(result[0], [2, 8, 10 + i + 1, 32]);
        assertShape(result[1], [2, 8, 10 + i + 1, 32]);
      }

      expect(cache.getIdx()).toBe(15);
      expect(cache.getOffsets()).toEqual([15, 15]);
    });

    it('should handle large batch sizes', () => {
      const leftPadding = new Int32Array(
        Array(16)
          .fill(0)
          .map((_, i) => i % 4),
      ); // [0,1,2,3,0,1,2,3,...]
      const cache = new BatchKVCache(leftPadding);

      const keys = MxArray.randomNormal(shape(16, 4, 8, 16), 0, 1);
      const values = MxArray.randomNormal(shape(16, 4, 8, 16), 0, 1);

      const result = cache.updateAndFetch(keys, values);

      expect(result).toHaveLength(2);
      expect(cache.getIdx()).toBe(8);
      assertShape(result[0], [16, 4, 8, 16]);
      assertShape(result[1], [16, 4, 8, 16]);
    });

    it('should allocate cache in 256-step increments', () => {
      const cache = new BatchKVCache(int32(0));

      // First update with 10 tokens should allocate 256 steps
      const keys = MxArray.randomNormal(shape(1, 2, 10, 8), 0, 1);
      const values = MxArray.randomNormal(shape(1, 2, 10, 8), 0, 1);
      cache.updateAndFetch(keys, values);

      expect(cache.getIdx()).toBe(10);

      // Should reuse allocated space for small updates
      for (let i = 0; i < 10; i++) {
        const key = MxArray.randomNormal(shape(1, 2, 1, 8), 0, 1);
        const value = MxArray.randomNormal(shape(1, 2, 1, 8), 0, 1);
        cache.updateAndFetch(key, value);
      }

      expect(cache.getIdx()).toBe(20);
    });
  });

  describe('Left Padding Correctness', () => {
    it('should correctly track offsets with varying padding', () => {
      const cache = new BatchKVCache(int32(5, 0, 3));

      const keys = MxArray.randomNormal(shape(3, 2, 10, 8), 0, 1);
      const values = MxArray.randomNormal(shape(3, 2, 10, 8), 0, 1);
      cache.updateAndFetch(keys, values);

      // Offset = initial_padding + tokens_added
      expect(cache.getOffsets()).toEqual([5, 10, 7]); // -5+10, 0+10, -3+10
      expect(cache.getIdx()).toBe(10);
    });

    it('should maintain padding after multiple updates', () => {
      const cache = new BatchKVCache(int32(2, 4, 1));

      // First update
      const keys1 = MxArray.randomNormal(shape(3, 2, 5, 8), 0, 1);
      const values1 = MxArray.randomNormal(shape(3, 2, 5, 8), 0, 1);
      cache.updateAndFetch(keys1, values1);

      expect(cache.getOffsets()).toEqual([3, 1, 4]); // -2+5, -4+5, -1+5

      // Second update
      const keys2 = MxArray.randomNormal(shape(3, 2, 3, 8), 0, 1);
      const values2 = MxArray.randomNormal(shape(3, 2, 3, 8), 0, 1);
      cache.updateAndFetch(keys2, values2);

      expect(cache.getOffsets()).toEqual([6, 4, 7]); // 3+3, 1+3, 4+3
      expect(cache.getIdx()).toBe(8); // 5 + 3
    });

    it('should handle extreme padding differences', () => {
      const cache = new BatchKVCache(int32(0, 50, 25));

      const keys = MxArray.randomNormal(shape(3, 2, 10, 8), 0, 1);
      const values = MxArray.randomNormal(shape(3, 2, 10, 8), 0, 1);
      cache.updateAndFetch(keys, values);

      expect(cache.getOffsets()).toEqual([10, -40, -15]);
      expect(Array.from(cache.getLeftPadding())).toEqual([0, 50, 25]);
    });
  });

  describe('Filter Operation', () => {
    it('should filter batch elements by indices', () => {
      const cache = new BatchKVCache(int32(1, 2, 0, 3));

      // Add data for 4 batch elements
      const keys = MxArray.randomNormal(shape(4, 2, 5, 8), 0, 1);
      const values = MxArray.randomNormal(shape(4, 2, 5, 8), 0, 1);
      cache.updateAndFetch(keys, values);

      // Filter to keep only elements 0 and 2
      cache.filter(int32(0, 2));

      expect(Array.from(cache.getLeftPadding())).toEqual([1, 0]); // Kept elements 0 and 2
      expect(cache.getOffsets()).toEqual([4, 5]); // -1+5, 0+5
      expect(cache.getIdx()).toBe(5);
    });

    it('should optimize padding after filtering', () => {
      const cache = new BatchKVCache(int32(5, 5, 5));

      const keys = MxArray.randomNormal(shape(3, 2, 10, 8), 0, 1);
      const values = MxArray.randomNormal(shape(3, 2, 10, 8), 0, 1);
      cache.updateAndFetch(keys, values);

      expect(Array.from(cache.getLeftPadding())).toEqual([5, 5, 5]);

      // Filter to keep all elements (triggers padding optimization)
      cache.filter(int32(0, 1, 2));

      // Minimum padding is 5, so shift left by 5
      expect(Array.from(cache.getLeftPadding())).toEqual([0, 0, 0]);
      expect(cache.getIdx()).toBe(5); // 10 - 5
    });

    it('should handle filtering to single element', () => {
      const cache = new BatchKVCache(int32(1, 2, 3));

      const keys = MxArray.randomNormal(shape(3, 4, 8, 16), 0, 1);
      const values = MxArray.randomNormal(shape(3, 4, 8, 16), 0, 1);
      cache.updateAndFetch(keys, values);

      // Keep only element 1 (has padding=2)
      cache.filter(int32(1));

      // After filtering, min_padding=2, so optimization shifts left by 2
      expect(Array.from(cache.getLeftPadding())).toEqual([0]); // 2-2=0
      expect(cache.getOffsets()).toEqual([4]); // (6-2)=4, where 6=-2+8
      expect(cache.getIdx()).toBe(6); // 8-2=6
    });

    it('should handle empty cache filter gracefully', () => {
      const cache = new BatchKVCache(int32(1, 2));

      // Filter without any data
      expect(() => cache.filter(int32(0))).not.toThrow();

      expect(Array.from(cache.getLeftPadding())).toEqual([1]);
      expect(cache.getOffsets()).toEqual([-1]);
    });

    it('should filter with non-sequential indices', () => {
      const cache = new BatchKVCache(int32(0, 1, 2, 3, 4));

      const keys = MxArray.randomNormal(shape(5, 2, 6, 8), 0, 1);
      const values = MxArray.randomNormal(shape(5, 2, 6, 8), 0, 1);
      cache.updateAndFetch(keys, values);

      // Keep elements 4, 1, 3 (out of order, padding=[4,1,3])
      cache.filter(int32(4, 1, 3));

      // After filtering, min_padding=1, so optimization shifts left by 1
      expect(Array.from(cache.getLeftPadding())).toEqual([3, 0, 2]); // [4-1, 1-1, 3-1]
      expect(cache.getOffsets()).toEqual([1, 4, 2]); // [2-1, 5-1, 3-1], where original offsets were -4+6, -1+6, -3+6
      expect(cache.getIdx()).toBe(5); // 6-1=5
    });
  });

  describe('Extend Operation', () => {
    it('should extend with another cache', () => {
      const cache1 = new BatchKVCache(int32(0, 1));
      const cache2 = new BatchKVCache(int32(2, 0));

      // Add data to both caches
      const keys1 = MxArray.randomNormal(shape(2, 2, 5, 8), 0, 1);
      const values1 = MxArray.randomNormal(shape(2, 2, 5, 8), 0, 1);
      cache1.updateAndFetch(keys1, values1);

      const keys2 = MxArray.randomNormal(shape(2, 2, 5, 8), 0, 1);
      const values2 = MxArray.randomNormal(shape(2, 2, 5, 8), 0, 1);
      cache2.updateAndFetch(keys2, values2);

      // Extend cache1 with cache2
      cache1.extend(cache2);

      // Should have 4 batch elements now (2 + 2)
      expect(cache1.getOffsets()).toHaveLength(4);
      expect(Array.from(cache1.getLeftPadding())).toHaveLength(4);
      expect(cache1.getIdx()).toBe(5); // Max of both
    });

    it('should align caches with different idx values', () => {
      const cache1 = new BatchKVCache(int32(0));
      const cache2 = new BatchKVCache(int32(0));

      // Cache1: 10 tokens
      const keys1 = MxArray.randomNormal(shape(1, 2, 10, 8), 0, 1);
      const values1 = MxArray.randomNormal(shape(1, 2, 10, 8), 0, 1);
      cache1.updateAndFetch(keys1, values1);

      // Cache2: 5 tokens
      const keys2 = MxArray.randomNormal(shape(1, 2, 5, 8), 0, 1);
      const values2 = MxArray.randomNormal(shape(1, 2, 5, 8), 0, 1);
      cache2.updateAndFetch(keys2, values2);

      cache1.extend(cache2);

      // Both should align to max idx (10)
      expect(cache1.getIdx()).toBe(10);
      expect(cache1.getOffsets()).toHaveLength(2);
    });

    it('should handle extending with different padding', () => {
      const cache1 = new BatchKVCache(int32(1, 2));
      const cache2 = new BatchKVCache(int32(0, 3));

      const keys1 = MxArray.randomNormal(shape(2, 2, 8, 8), 0, 1);
      const values1 = MxArray.randomNormal(shape(2, 2, 8, 8), 0, 1);
      cache1.updateAndFetch(keys1, values1);

      const keys2 = MxArray.randomNormal(shape(2, 2, 8, 8), 0, 1);
      const values2 = MxArray.randomNormal(shape(2, 2, 8, 8), 0, 1);
      cache2.updateAndFetch(keys2, values2);

      cache1.extend(cache2);

      expect(cache1.getOffsets()).toHaveLength(4);
      expect(Array.from(cache1.getLeftPadding())).toHaveLength(4);
      expect(cache1.getIdx()).toBe(8);
    });

    it('should throw error when extending empty caches', () => {
      const cache1 = new BatchKVCache(int32(0));
      const cache2 = new BatchKVCache(int32(0));

      expect(() => cache1.extend(cache2)).toThrow(/empty/i);
    });
  });

  describe('Reset Operation', () => {
    it('should reset cache to initial state', () => {
      const cache = new BatchKVCache(int32(1, 2, 3));

      // Add data
      const keys = MxArray.randomNormal(shape(3, 4, 10, 16), 0, 1);
      const values = MxArray.randomNormal(shape(3, 4, 10, 16), 0, 1);
      cache.updateAndFetch(keys, values);

      expect(cache.getIdx()).toBe(10);

      // Reset
      cache.reset();

      expect(cache.getIdx()).toBe(0);
      expect(cache.getOffsets()).toEqual([-1, -2, -3]); // Back to -padding
      expect(Array.from(cache.getLeftPadding())).toEqual([1, 2, 3]); // Padding preserved
    });

    it('should work correctly after reset', () => {
      const cache = new BatchKVCache(int32(0, 1));

      // First use
      const keys1 = MxArray.randomNormal(shape(2, 2, 5, 8), 0, 1);
      const values1 = MxArray.randomNormal(shape(2, 2, 5, 8), 0, 1);
      cache.updateAndFetch(keys1, values1);

      cache.reset();

      // Second use after reset
      const keys2 = MxArray.randomNormal(shape(2, 2, 3, 8), 0, 1);
      const values2 = MxArray.randomNormal(shape(2, 2, 3, 8), 0, 1);
      const result = cache.updateAndFetch(keys2, values2);

      expect(cache.getIdx()).toBe(3);
      expect(cache.getOffsets()).toEqual([3, 2]);
      assertShape(result[0], [2, 2, 3, 8]);
      assertShape(result[1], [2, 2, 3, 8]);
    });
  });

  describe('Edge Cases', () => {
    it('should handle single batch element', () => {
      const cache = new BatchKVCache(int32(2));

      const keys = MxArray.randomNormal(shape(1, 4, 10, 16), 0, 1);
      const values = MxArray.randomNormal(shape(1, 4, 10, 16), 0, 1);
      const result = cache.updateAndFetch(keys, values);

      expect(cache.getIdx()).toBe(10);
      expect(cache.getOffsets()).toEqual([8]); // -2+10
      assertShape(result[0], [1, 4, 10, 16]);
    });

    it('should handle maximum padding scenario', () => {
      const cache = new BatchKVCache(int32(100, 100));

      const keys = MxArray.randomNormal(shape(2, 2, 50, 8), 0, 1);
      const values = MxArray.randomNormal(shape(2, 2, 50, 8), 0, 1);
      cache.updateAndFetch(keys, values);

      expect(cache.getOffsets()).toEqual([-50, -50]);
      expect(cache.getIdx()).toBe(50);
    });

    it('should handle very long sequences', () => {
      const cache = new BatchKVCache(int32(0, 0));

      // Add 500 tokens
      const keys = MxArray.randomNormal(shape(2, 4, 500, 16), 0, 1);
      const values = MxArray.randomNormal(shape(2, 4, 500, 16), 0, 1);
      const result = cache.updateAndFetch(keys, values);

      expect(cache.getIdx()).toBe(500);
      assertShape(result[0], [2, 4, 500, 16]);
      assertShape(result[1], [2, 4, 500, 16]);
    });

    it('should handle zero-length updates gracefully', () => {
      const cache = new BatchKVCache(int32(1, 2));

      const keys = MxArray.randomNormal(shape(2, 2, 0, 8), 0, 1);
      const values = MxArray.randomNormal(shape(2, 2, 0, 8), 0, 1);

      // This might not be practical but shouldn't crash
      expect(() => cache.updateAndFetch(keys, values)).not.toThrow();
    });
  });

  describe('Complex Scenarios', () => {
    it('should handle dynamic batch reduction (filter after generation)', () => {
      const cache = new BatchKVCache(int32(0, 1, 2, 3));

      // Initial prompt for 4 sequences
      const keys1 = MxArray.randomNormal(shape(4, 4, 10, 16), 0, 1);
      const values1 = MxArray.randomNormal(shape(4, 4, 10, 16), 0, 1);
      cache.updateAndFetch(keys1, values1);

      // Generate a few tokens
      const keys2 = MxArray.randomNormal(shape(4, 4, 5, 16), 0, 1);
      const values2 = MxArray.randomNormal(shape(4, 4, 5, 16), 0, 1);
      cache.updateAndFetch(keys2, values2);

      expect(cache.getIdx()).toBe(15);

      // Filter to keep only 2 sequences (e.g., beam search)
      cache.filter(int32(0, 2));

      expect(cache.getOffsets()).toHaveLength(2);
      expect(cache.getIdx()).toBe(15);

      // Continue generation with reduced batch
      const keys3 = MxArray.randomNormal(shape(2, 4, 3, 16), 0, 1);
      const values3 = MxArray.randomNormal(shape(2, 4, 3, 16), 0, 1);
      const result = cache.updateAndFetch(keys3, values3);

      expect(cache.getIdx()).toBe(18);
      assertShape(result[0], [2, 4, 18, 16]);
    });

    it('should handle multiple filter operations', () => {
      const cache = new BatchKVCache(int32(0, 1, 2, 3, 4));

      const keys = MxArray.randomNormal(shape(5, 2, 8, 8), 0, 1);
      const values = MxArray.randomNormal(shape(5, 2, 8, 8), 0, 1);
      cache.updateAndFetch(keys, values);

      // First filter
      cache.filter(int32(0, 2, 4));
      expect(cache.getOffsets()).toHaveLength(3);

      // Second filter
      cache.filter(int32(0, 2));
      expect(cache.getOffsets()).toHaveLength(2);

      // Third filter
      cache.filter(int32(1));
      expect(cache.getOffsets()).toHaveLength(1);
    });

    it('should work in typical batch generation pipeline', () => {
      // Simulate real batch generation: 3 prompts of different lengths
      const leftPadding = int32(2, 0, 1); // Padded to align
      const cache = new BatchKVCache(leftPadding);

      // Step 1: Process prompts (lengths 6, 8, 7 after padding to 8)
      const promptKeys = MxArray.randomNormal(shape(3, 8, 8, 32), 0, 1);
      const promptValues = MxArray.randomNormal(shape(3, 8, 8, 32), 0, 1);
      let result = cache.updateAndFetch(promptKeys, promptValues);

      expect(cache.getIdx()).toBe(8);
      expect(cache.getOffsets()).toEqual([6, 8, 7]);

      // Step 2: Generate 10 tokens autoregressively
      for (let i = 0; i < 10; i++) {
        const tokenKeys = MxArray.randomNormal(shape(3, 8, 1, 32), 0, 1);
        const tokenValues = MxArray.randomNormal(shape(3, 8, 1, 32), 0, 1);
        result = cache.updateAndFetch(tokenKeys, tokenValues);

        expect(cache.getIdx()).toBe(8 + i + 1);
      }

      // Final state
      expect(cache.getIdx()).toBe(18);
      expect(cache.getOffsets()).toEqual([16, 18, 17]);
      assertShape(result[0], [3, 8, 18, 32]);
      assertShape(result[1], [3, 8, 18, 32]);
    });
  });
});
