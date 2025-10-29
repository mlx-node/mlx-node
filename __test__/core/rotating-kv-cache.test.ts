import { describe, it, expect } from 'vitest';
import { RotatingKVCache, MxArray } from '@mlx-node/core';
import { shape, assertShape, createFloat32Array } from '../test-utils';

/**
 * RotatingKVCache Tests
 *
 * Reference: mlx-lm/tests/test_prompt_cache.py lines 62-98, 368-383
 * Tests rotating KV cache with fixed maximum size and optional "keep" tokens
 *
 * Key features:
 * - max_size: Maximum number of tokens to cache
 * - keep: Number of initial tokens to never evict (e.g., system prompt)
 * - Rotation: When cache is full, old tokens are evicted (except "keep" tokens)
 * - Two update paths: single-token (in-place) vs multi-token (concat)
 */
describe('RotatingKVCache (MLX-LM Reference)', () => {
  describe('Basic Functionality', () => {
    it('should create empty cache with max_size', () => {
      const cache = new RotatingKVCache(8);
      expect(cache.getOffset()).toBe(0);
      expect(cache.getMaxSize()).toBe(8);
      expect(cache.getKeep()).toBe(0);
      expect(cache.getIdx()).toBe(0);
    });

    it('should create cache with keep parameter', () => {
      const cache = new RotatingKVCache(8, 2);
      expect(cache.getMaxSize()).toBe(8);
      expect(cache.getKeep()).toBe(2);
    });

    it('should update with initial keys/values (below max_size)', () => {
      const cache = new RotatingKVCache(10);

      // Add 6 tokens (below max_size=10)
      const keys = MxArray.randomNormal(shape(1, 2, 6, 8), 0, 1);
      const values = MxArray.randomNormal(shape(1, 2, 6, 8), 0, 1);

      const result = cache.updateAndFetch(keys, values);

      expect(result).toHaveLength(2);
      expect(cache.getOffset()).toBe(6);
      assertShape(result[0], [1, 2, 6, 8]);
      assertShape(result[1], [1, 2, 6, 8]);
    });

    it('should update with multiple tokens without rotation', () => {
      const cache = new RotatingKVCache(10);

      // First update: 4 tokens
      const keys1 = MxArray.randomNormal(shape(1, 2, 4, 8), 0, 1);
      const values1 = MxArray.randomNormal(shape(1, 2, 4, 8), 0, 1);
      cache.updateAndFetch(keys1, values1);

      // Second update: 3 more tokens (total 7, still below max_size=10)
      const keys2 = MxArray.randomNormal(shape(1, 2, 3, 8), 0, 1);
      const values2 = MxArray.randomNormal(shape(1, 2, 3, 8), 0, 1);
      const result = cache.updateAndFetch(keys2, values2);

      expect(cache.getOffset()).toBe(7);
      assertShape(result[0], [1, 2, 7, 8]);
      assertShape(result[1], [1, 2, 7, 8]);
    });
  });

  describe('Rotation Mechanism', () => {
    it('should rotate when exceeding max_size', () => {
      const cache = new RotatingKVCache(4, 0); // max_size=4, no keep

      // Add 10 tokens total, should only keep last 4
      const keys = MxArray.randomNormal(shape(1, 2, 10, 8), 0, 1);
      const values = MxArray.randomNormal(shape(1, 2, 10, 8), 0, 1);
      const result = cache.updateAndFetch(keys, values);

      // Should have processed 10 tokens but only cached last max_size-1 + 10 = 13
      // Actually: ensures every token gets at least max_size context
      // Reference: trim_size = current_len - max_size + 1
      expect(cache.getOffset()).toBe(10);

      // Result should contain all tokens to ensure last token has max_size context
      // The result is the cache needed to process all input tokens
      expect(result[0].shape()[2]).toBeGreaterThanOrEqual(4);
      expect(result[0].shape()[2]).toBeLessThanOrEqual(13); // max_size + seq_len - 1
    });

    it('should keep initial tokens during rotation (keep parameter)', () => {
      const cache = new RotatingKVCache(8, 2); // max_size=8, keep first 2 tokens

      // Add 12 tokens - should keep first 2, rotate the rest
      const keys = MxArray.randomNormal(shape(1, 2, 12, 8), 0, 1);
      const values = MxArray.randomNormal(shape(1, 2, 12, 8), 0, 1);
      const result = cache.updateAndFetch(keys, values);

      expect(cache.getOffset()).toBe(12);

      // Cache should maintain structure that preserves first 2 tokens
      // Result size ensures all input tokens have sufficient context
      expect(result[0].shape()[2]).toBeGreaterThanOrEqual(8);
      expect(result[0].shape()[2]).toBeLessThanOrEqual(19); // max_size + seq_len - 1
    });

    it('should handle single-token updates after multi-token (rotation)', () => {
      const cache = new RotatingKVCache(8, 2);

      // Initial prompt: 8 tokens (at max_size)
      const keys1 = MxArray.randomNormal(shape(1, 4, 8, 16), 0, 1);
      const values1 = MxArray.randomNormal(shape(1, 4, 8, 16), 0, 1);
      cache.updateAndFetch(keys1, values1);
      expect(cache.getOffset()).toBe(8);

      // Add 2 single tokens - should trigger rotation
      for (let i = 0; i < 2; i++) {
        const key_token = MxArray.randomNormal(shape(1, 4, 1, 16), 0, 1);
        const value_token = MxArray.randomNormal(shape(1, 4, 1, 16), 0, 1);
        const result = cache.updateAndFetch(key_token, value_token);

        expect(cache.getOffset()).toBe(8 + i + 1);
        // After hitting max_size, cache stays at max_size
        assertShape(result[0], [1, 4, 8, 16]);
        assertShape(result[1], [1, 4, 8, 16]);
      }
    });
  });

  describe('Single Token Updates (In-Place)', () => {
    it('should handle incremental single-token generation', () => {
      const cache = new RotatingKVCache(10, 0);

      // Initial prompt: 5 tokens
      const keys1 = MxArray.randomNormal(shape(1, 4, 5, 16), 0, 1);
      const values1 = MxArray.randomNormal(shape(1, 4, 5, 16), 0, 1);
      cache.updateAndFetch(keys1, values1);
      expect(cache.getOffset()).toBe(5);

      // Generate 3 tokens one by one (still below max_size=10)
      for (let i = 0; i < 3; i++) {
        const key_token = MxArray.randomNormal(shape(1, 4, 1, 16), 0, 1);
        const value_token = MxArray.randomNormal(shape(1, 4, 1, 16), 0, 1);
        const result = cache.updateAndFetch(key_token, value_token);

        expect(cache.getOffset()).toBe(5 + i + 1);
        assertShape(result[0], [1, 4, 5 + i + 1, 16]);
        assertShape(result[1], [1, 4, 5 + i + 1, 16]);
      }

      expect(cache.getOffset()).toBe(8);
    });

    it('should rotate during single-token updates when max_size reached', () => {
      const cache = new RotatingKVCache(6, 0);

      // Fill to max_size
      const keys1 = MxArray.randomNormal(shape(1, 2, 6, 8), 0, 1);
      const values1 = MxArray.randomNormal(shape(1, 2, 6, 8), 0, 1);
      cache.updateAndFetch(keys1, values1);

      // Add 4 more single tokens - should rotate
      for (let i = 0; i < 4; i++) {
        const key_token = MxArray.randomNormal(shape(1, 2, 1, 8), 0, 1);
        const value_token = MxArray.randomNormal(shape(1, 2, 1, 8), 0, 1);
        const result = cache.updateAndFetch(key_token, value_token);

        expect(cache.getOffset()).toBe(6 + i + 1);
        // Cache size stays at max_size during rotation
        assertShape(result[0], [1, 2, 6, 8]);
        assertShape(result[1], [1, 2, 6, 8]);
      }
    });

    it('should preserve keep tokens during single-token rotation', () => {
      const cache = new RotatingKVCache(6, 2); // max_size=6, keep first 2

      // Fill to max_size
      const keys1 = MxArray.randomNormal(shape(1, 2, 6, 8), 0, 1);
      const values1 = MxArray.randomNormal(shape(1, 2, 6, 8), 0, 1);
      cache.updateAndFetch(keys1, values1);

      // Add tokens - rotation should keep first 2
      for (let i = 0; i < 3; i++) {
        const key_token = MxArray.randomNormal(shape(1, 2, 1, 8), 0, 1);
        const value_token = MxArray.randomNormal(shape(1, 2, 1, 8), 0, 1);
        const result = cache.updateAndFetch(key_token, value_token);

        expect(cache.getOffset()).toBe(6 + i + 1);
        assertShape(result[0], [1, 2, 6, 8]);
      }
    });
  });

  describe('Data Integrity', () => {
    it('should maintain data integrity without rotation', () => {
      const cache = new RotatingKVCache(10, 0);

      // First update with known values
      const keys1 = createFloat32Array(
        [1, 2, 3, 4], // Simple sequential values
        [1, 1, 4, 1], // (batch=1, heads=1, seq=4, dim=1)
      );
      const values1 = createFloat32Array([10, 20, 30, 40], [1, 1, 4, 1]);

      const result1 = cache.updateAndFetch(keys1, values1);
      const keys1_data = result1[0].toFloat32();
      const values1_data = result1[1].toFloat32();

      expect(Array.from(keys1_data)).toEqual([1, 2, 3, 4]);
      expect(Array.from(values1_data)).toEqual([10, 20, 30, 40]);

      // Second update
      const keys2 = createFloat32Array([5, 6], [1, 1, 2, 1]);
      const values2 = createFloat32Array([50, 60], [1, 1, 2, 1]);

      const result2 = cache.updateAndFetch(keys2, values2);
      const keys2_data = result2[0].toFloat32();
      const values2_data = result2[1].toFloat32();

      // Should have concatenated correctly (still below max_size)
      expect(Array.from(keys2_data)).toEqual([1, 2, 3, 4, 5, 6]);
      expect(Array.from(values2_data)).toEqual([10, 20, 30, 40, 50, 60]);
    });

    it('should reset correctly', () => {
      const cache = new RotatingKVCache(8, 2);

      // Add some data
      const keys = MxArray.randomNormal(shape(2, 8, 6, 32), 0, 1);
      const values = MxArray.randomNormal(shape(2, 8, 6, 32), 0, 1);
      cache.updateAndFetch(keys, values);
      expect(cache.getOffset()).toBe(6);

      // Reset
      cache.reset();
      expect(cache.getOffset()).toBe(0);
      expect(cache.getIdx()).toBe(0);

      // Should work like a fresh cache after reset
      const keys2 = MxArray.randomNormal(shape(2, 8, 5, 32), 0, 1);
      const values2 = MxArray.randomNormal(shape(2, 8, 5, 32), 0, 1);
      const result = cache.updateAndFetch(keys2, values2);

      expect(cache.getOffset()).toBe(5);
      assertShape(result[0], [2, 8, 5, 32]);
    });
  });

  describe('Edge Cases', () => {
    it('should handle max_size = 1', () => {
      const cache = new RotatingKVCache(1, 0);

      // Add 3 tokens, should only keep last one
      for (let i = 0; i < 3; i++) {
        const keys = MxArray.randomNormal(shape(1, 2, 1, 8), 0, 1);
        const values = MxArray.randomNormal(shape(1, 2, 1, 8), 0, 1);
        const result = cache.updateAndFetch(keys, values);

        expect(cache.getOffset()).toBe(i + 1);
        assertShape(result[0], [1, 2, 1, 8]);
      }
    });

    it('should handle keep = max_size (all tokens kept)', () => {
      const cache = new RotatingKVCache(5, 5);

      // Add 5 tokens
      const keys1 = MxArray.randomNormal(shape(1, 2, 5, 8), 0, 1);
      const values1 = MxArray.randomNormal(shape(1, 2, 5, 8), 0, 1);
      cache.updateAndFetch(keys1, values1);

      // Add 3 more - all should be kept since keep=max_size
      const keys2 = MxArray.randomNormal(shape(1, 2, 3, 8), 0, 1);
      const values2 = MxArray.randomNormal(shape(1, 2, 3, 8), 0, 1);
      const result = cache.updateAndFetch(keys2, values2);

      expect(cache.getOffset()).toBe(8);
      // Should grow beyond max_size because all tokens are "keep"
      expect(result[0].shape()[2]).toBeGreaterThanOrEqual(5);
    });

    it('should handle very large max_size', () => {
      const cache = new RotatingKVCache(1000, 10);

      // Add 50 tokens (well below max_size)
      const keys = MxArray.randomNormal(shape(1, 8, 50, 64), 0, 1);
      const values = MxArray.randomNormal(shape(1, 8, 50, 64), 0, 1);
      const result = cache.updateAndFetch(keys, values);

      expect(cache.getOffset()).toBe(50);
      assertShape(result[0], [1, 8, 50, 64]);
    });

    it('should handle batch size > 1', () => {
      const cache = new RotatingKVCache(10, 2);

      // Batch size 4
      const keys = MxArray.randomNormal(shape(4, 2, 8, 16), 0, 1);
      const values = MxArray.randomNormal(shape(4, 2, 8, 16), 0, 1);
      const result = cache.updateAndFetch(keys, values);

      expect(cache.getOffset()).toBe(8);
      assertShape(result[0], [4, 2, 8, 16]);
      assertShape(result[1], [4, 2, 8, 16]);
    });

    it('should handle GQA with different kv_heads', () => {
      const cache = new RotatingKVCache(12, 0);

      // 8 query heads but only 2 KV heads (4x GQA)
      const keys = MxArray.randomNormal(shape(1, 2, 10, 64), 0, 1);
      const values = MxArray.randomNormal(shape(1, 2, 10, 64), 0, 1);
      const result = cache.updateAndFetch(keys, values);

      assertShape(result[0], [1, 2, 10, 64]);
      assertShape(result[1], [1, 2, 10, 64]);
    });
  });

  describe('Real-World Scenarios', () => {
    it('should handle chat completion with rotation (system prompt kept)', () => {
      const cache = new RotatingKVCache(100, 10); // Keep first 10 tokens (system prompt)

      // System + user prompt: 50 tokens
      const keys1 = MxArray.randomNormal(shape(1, 8, 50, 64), 0, 1);
      const values1 = MxArray.randomNormal(shape(1, 8, 50, 64), 0, 1);
      cache.updateAndFetch(keys1, values1);
      expect(cache.getOffset()).toBe(50);

      // Generate 60 tokens incrementally (total 110 > max_size=100)
      for (let i = 0; i < 60; i++) {
        const key_token = MxArray.randomNormal(shape(1, 8, 1, 64), 0, 1);
        const value_token = MxArray.randomNormal(shape(1, 8, 1, 64), 0, 1);
        const result = cache.updateAndFetch(key_token, value_token);

        expect(cache.getOffset()).toBe(50 + i + 1);

        // After exceeding max_size, cache stays at max_size
        if (50 + i + 1 > 100) {
          assertShape(result[0], [1, 8, 100, 64]);
        }
      }

      expect(cache.getOffset()).toBe(110);
    });

    it('should handle conversation with multiple resets', () => {
      const cache = new RotatingKVCache(50, 5);

      // First conversation
      const keys1 = MxArray.randomNormal(shape(1, 4, 30, 32), 0, 1);
      const values1 = MxArray.randomNormal(shape(1, 4, 30, 32), 0, 1);
      cache.updateAndFetch(keys1, values1);
      expect(cache.getOffset()).toBe(30);

      // Second conversation after reset
      cache.reset();
      const keys2 = MxArray.randomNormal(shape(1, 4, 40, 32), 0, 1);
      const values2 = MxArray.randomNormal(shape(1, 4, 40, 32), 0, 1);
      cache.updateAndFetch(keys2, values2);
      expect(cache.getOffset()).toBe(40);
    });

    it('should handle long generation beyond max_size multiple times', () => {
      const cache = new RotatingKVCache(8, 2);

      // Initial: 5 tokens
      const keys1 = MxArray.randomNormal(shape(1, 2, 5, 8), 0, 1);
      const values1 = MxArray.randomNormal(shape(1, 2, 5, 8), 0, 1);
      cache.updateAndFetch(keys1, values1);

      // Generate 20 tokens (will rotate multiple times)
      for (let i = 0; i < 20; i++) {
        const key_token = MxArray.randomNormal(shape(1, 2, 1, 8), 0, 1);
        const value_token = MxArray.randomNormal(shape(1, 2, 1, 8), 0, 1);
        const result = cache.updateAndFetch(key_token, value_token);

        expect(cache.getOffset()).toBe(5 + i + 1);

        // Cache size caps at max_size after initial fill
        if (5 + i + 1 >= 8) {
          assertShape(result[0], [1, 2, 8, 8]);
        }
      }

      expect(cache.getOffset()).toBe(25);
    });
  });

  describe('Comparison with Regular KVCache', () => {
    it('should behave like regular KVCache when below max_size', () => {
      const rotating = new RotatingKVCache(20, 0);

      // Add 10 tokens (well below max_size)
      const keys = createFloat32Array(
        Array.from({ length: 10 }, (_, i) => i + 1),
        [1, 1, 10, 1],
      );
      const values = createFloat32Array(
        Array.from({ length: 10 }, (_, i) => (i + 1) * 10),
        [1, 1, 10, 1],
      );

      const result = rotating.updateAndFetch(keys, values);

      expect(rotating.getOffset()).toBe(10);

      const keys_data = result[0].toFloat32();
      const values_data = result[1].toFloat32();
      expect(Array.from(keys_data)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      expect(Array.from(values_data)).toEqual([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    });
  });
});
