import { describe, expect, it } from 'vitest';
import { getHighEntropyMask } from '@mlx-node/core';
import { MxArray } from '@mlx-node/core';
import { createFloat32Array, createInt32Array, float32, int32 } from '../test-utils';

// Helper to assert shape equality
function assertShapeEqual(arr: MxArray, expectedShape: number[]) {
  const actualShape = Array.from(arr.shape());
  const expected = expectedShape.map((d) => BigInt(d));
  expect(actualShape).toEqual(expected);
}

/**
 * Entropy Filtering Tests
 *
 * Reference: trl/tests/test_grpo_trainer.py lines 58-128
 * Tests the get_high_entropy_mask functionality for selective GRPO training
 *
 * The entropy filtering mechanism allows training on only high-uncertainty tokens,
 * which is a core optimization in GRPO. The quantile threshold determines what
 * percentage of tokens to train on (e.g., 0.8 = top 20% highest entropy tokens).
 */
describe('Entropy Filtering (TRL Reference)', () => {
  describe('Basic Entropy Mask Computation', () => {
    it('threshold=0.8: top 20% high-entropy tokens unmasked', () => {
      // Reference: test_compute_entropy_mask_0
      // We have 12 tokens total, 10 non-pad tokens
      // threshold=0.8 means top 20% (2 tokens) should be unmasked
      // The highest entropies are 0.9 and 1.0 (1.1 and 1.2 are padding)

      const entropies = createFloat32Array(float32(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2), [2, 6]);

      const mask = createInt32Array(int32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0), [2, 6]);

      const entropyMask = getHighEntropyMask(entropies, mask, 0.8);

      // Expected: only tokens at positions [1,2] and [1,3] with entropies 0.9, 1.0
      const expected = createInt32Array(int32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0), [2, 6]);

      assertShapeEqual(entropyMask, [2, 6]);

      const entropyMaskData = entropyMask.toInt32();
      const expectedData = expected.toInt32();

      for (let i = 0; i < expectedData.length; i++) {
        expect(entropyMaskData[i]).toBe(expectedData[i]);
      }
    });

    it('threshold=0.8: different entropy distribution', () => {
      // Reference: test_compute_entropy_mask_1
      // 8 non-pad tokens, threshold=0.8 means top 20% ≈ 2 tokens
      // Highest entropies: 1.4 and 0.8 (or 1.0)

      const entropies = createFloat32Array(
        float32(0.1, 0.2, 0.3, 1.4, 0.5, 0.14, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        [2, 6],
      );

      const mask = createInt32Array(int32(1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0), [2, 6]);

      const entropyMask = getHighEntropyMask(entropies, mask, 0.8);

      // Expected: positions [0,3] (1.4) and [1,3] (0.8) are top 2
      const expected = createInt32Array(int32(0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0), [2, 6]);

      const entropyMaskData = entropyMask.toInt32();
      const expectedData = expected.toInt32();

      for (let i = 0; i < expectedData.length; i++) {
        expect(entropyMaskData[i]).toBe(expectedData[i]);
      }
    });

    it('threshold=0.5: top 50% tokens unmasked', () => {
      // Reference: test_compute_entropy_mask_lower_threshold
      // 10 non-pad tokens, threshold=0.5 means top 50% = 5 tokens

      const entropies = createFloat32Array(float32(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2), [2, 6]);

      const mask = createInt32Array(int32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0), [2, 6]);

      const entropyMask = getHighEntropyMask(entropies, mask, 0.5);

      // Expected: top 5 entropies (0.6, 0.7, 0.8, 0.9, 1.0)
      const expected = createInt32Array(int32(0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0), [2, 6]);

      const entropyMaskData = entropyMask.toInt32();
      const expectedData = expected.toInt32();

      for (let i = 0; i < expectedData.length; i++) {
        expect(entropyMaskData[i]).toBe(expectedData[i]);
      }
    });
  });

  describe('Edge Cases: Threshold Boundaries', () => {
    it('threshold=0.0: all non-pad tokens unmasked', () => {
      // Reference: test_compute_entropy_threshold_0
      // threshold=0.0 means include all tokens (0th quantile)

      const entropies = createFloat32Array(float32(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2), [2, 6]);

      const mask = createInt32Array(int32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0), [2, 6]);

      const entropyMask = getHighEntropyMask(entropies, mask, 0.0);

      // Expected: same as input mask (all non-pad tokens)
      const entropyMaskData = entropyMask.toInt32();
      const maskData = mask.toInt32();

      for (let i = 0; i < maskData.length; i++) {
        expect(entropyMaskData[i]).toBe(maskData[i]);
      }
    });

    it('threshold=1.0: only highest entropy token unmasked', () => {
      // Reference: test_compute_entropy_threshold_1
      // threshold=1.0 means only the single highest entropy token

      const entropies = createFloat32Array(float32(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2), [2, 6]);

      const mask = createInt32Array(int32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0), [2, 6]);

      const entropyMask = getHighEntropyMask(entropies, mask, 1.0);

      // Expected: only position [1,3] with entropy 1.0 (highest among non-pad)
      const expected = createInt32Array(int32(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0), [2, 6]);

      const entropyMaskData = entropyMask.toInt32();
      const expectedData = expected.toInt32();

      for (let i = 0; i < expectedData.length; i++) {
        expect(entropyMaskData[i]).toBe(expectedData[i]);
      }
    });

    it('all tokens masked (no non-pad tokens)', () => {
      // Reference: test_compute_entropy_all_masked
      // When all tokens are padding, mask should be all zeros

      const entropies = createFloat32Array(float32(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2), [2, 6]);

      const mask = createInt32Array(int32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), [2, 6]);

      const entropyMask = getHighEntropyMask(entropies, mask, 0.5);

      // Expected: all zeros (no tokens to train on)
      const entropyMaskData = entropyMask.toInt32();

      for (let i = 0; i < entropyMaskData.length; i++) {
        expect(entropyMaskData[i]).toBe(0);
      }
    });
  });

  describe('Shape and Type Validation', () => {
    it('should handle single batch', () => {
      const entropies = createFloat32Array(float32(0.1, 0.5, 0.9, 0.3, 0.7), [1, 5]);

      const mask = createInt32Array(int32(1, 1, 1, 1, 0), [1, 5]);

      const entropyMask = getHighEntropyMask(entropies, mask, 0.5);

      assertShapeEqual(entropyMask, [1, 5]);

      // Top 50% of 4 non-pad tokens = top 2 (0.9, 0.5)
      // Non-pad entropies: [0.1, 0.5, 0.9, 0.3], sorted: [0.1, 0.3, 0.5, 0.9]
      // 0.5 quantile = interpolate(0.3, 0.5) = 0.4
      // Selected: values >= 0.4 → [0.5, 0.9]
      const entropyMaskData = entropyMask.toInt32();
      expect(entropyMaskData[0]).toBe(0); // 0.1 < 0.4
      expect(entropyMaskData[1]).toBe(1); // 0.5 >= 0.4
      expect(entropyMaskData[2]).toBe(1); // 0.9 >= 0.4
      expect(entropyMaskData[3]).toBe(0); // 0.3 < 0.4
      expect(entropyMaskData[4]).toBe(0); // padding
    });

    it('should handle large batch sizes', () => {
      const batchSize = 16;
      const seqLen = 32;

      const entropies = createFloat32Array(
        new Float32Array(batchSize * seqLen).map(() => Math.random()),
        [batchSize, seqLen],
      );

      const mask = createInt32Array(
        new Int32Array(batchSize * seqLen).map(() => (Math.random() > 0.1 ? 1 : 0)),
        [batchSize, seqLen],
      );

      const entropyMask = getHighEntropyMask(entropies, mask, 0.8);

      assertShapeEqual(entropyMask, [batchSize, seqLen]);

      // Verify padding tokens are always masked
      const entropyMaskData = entropyMask.toInt32();
      const maskData = mask.toInt32();

      for (let i = 0; i < maskData.length; i++) {
        if (maskData[i] === 0) {
          expect(entropyMaskData[i]).toBe(0);
        }
      }
    });

    it('should preserve entropy ordering', () => {
      // Verify that higher entropy values are more likely to be selected
      const entropies = createFloat32Array(float32(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), [1, 8]);

      const mask = createInt32Array(int32(1, 1, 1, 1, 1, 1, 1, 1), [1, 8]);

      const entropyMask = getHighEntropyMask(entropies, mask, 0.75);

      // Top 25% of 8 tokens = top 2 (0.8, 0.7)
      const entropyMaskData = entropyMask.toInt32();

      expect(entropyMaskData[7]).toBe(1); // 0.8 - highest
      expect(entropyMaskData[6]).toBe(1); // 0.7 - second highest

      // Lower entropy tokens should be masked
      expect(entropyMaskData[0]).toBe(0); // 0.1
      expect(entropyMaskData[1]).toBe(0); // 0.2
    });
  });

  describe('Numerical Edge Cases', () => {
    it('should handle all equal entropies', () => {
      const entropies = createFloat32Array(float32(0.5, 0.5, 0.5, 0.5, 0.5, 0.5), [1, 6]);

      const mask = createInt32Array(int32(1, 1, 1, 1, 0, 0), [1, 6]);

      const entropyMask = getHighEntropyMask(entropies, mask, 0.5);

      // When all entropies are equal, the quantile threshold equals the value
      // All non-pad tokens >= threshold should be selected
      const entropyMaskData = entropyMask.toInt32();

      // All non-pad tokens should be selected (or all/none depending on >= vs >)
      const selectedCount = entropyMaskData.slice(0, 4).reduce((sum, val) => sum + val, 0);
      expect(selectedCount).toBeGreaterThan(0); // At least some selected
    });

    it('should handle very small entropies', () => {
      const entropies = createFloat32Array(float32(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3), [1, 6]);

      const mask = createInt32Array(int32(1, 1, 1, 1, 1, 1), [1, 6]);

      const entropyMask = getHighEntropyMask(entropies, mask, 0.8);

      // Top 20% should still work with very small values
      assertShapeEqual(entropyMask, [1, 6]);

      const entropyMaskData = entropyMask.toInt32();
      const selectedCount = entropyMaskData.reduce((sum, val) => sum + val, 0);

      // Should select approximately top 20% (1-2 tokens out of 6)
      expect(selectedCount).toBeGreaterThanOrEqual(1);
      expect(selectedCount).toBeLessThanOrEqual(2);
    });

    it('should handle very large entropies', () => {
      const entropies = createFloat32Array(float32(100, 200, 300, 400, 500, 600), [1, 6]);

      const mask = createInt32Array(int32(1, 1, 1, 1, 1, 1), [1, 6]);

      const entropyMask = getHighEntropyMask(entropies, mask, 0.8);

      // Top 20% = top 1-2 tokens (highest values)
      const entropyMaskData = entropyMask.toInt32();

      expect(entropyMaskData[5]).toBe(1); // 600 - highest
      expect(entropyMaskData[0]).toBe(0); // 100 - lowest
    });
  });
});
