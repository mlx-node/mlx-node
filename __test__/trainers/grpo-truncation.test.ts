/**
 * Tests for GRPO truncation masking
 *
 * Based on TRL's truncation handling:
 * - test_training_with_mask_truncated_completions()
 * - test_training_with_mask_truncated_completions_all_masked()
 *
 * Truncated completions are sequences that reach max_length without
 * generating an EOS token. These should be masked out from training
 * to avoid rewarding incomplete responses.
 */

import { describe, it, expect } from 'vitest';
import { MxArray } from '@mlx-node/core';
import { shape, int32 } from '../test-utils.js';

/**
 * Detect truncated sequences
 *
 * A sequence is truncated if:
 * 1. It reached max_length without EOS token
 * 2. The last non-padding token is not EOS
 *
 * @param tokens - Token sequences [B, T]
 * @param eosTokenId - EOS token ID
 * @param padTokenId - Padding token ID (optional)
 * @returns Boolean mask [B] where true = truncated
 */
export function detectTruncatedSequences(tokens: MxArray, eosTokenId: number, padTokenId?: number): boolean[] {
  const tokensData = tokens.toInt32();
  const tokensShape = Array.from(tokens.shape());

  const batchSize = Number(tokensShape[0]);
  const seqLen = Number(tokensShape[1]);

  const truncated: boolean[] = Array.from({ length: batchSize }, () => false);

  for (let b = 0; b < batchSize; b++) {
    // Find last non-padding token
    let lastTokenPos = seqLen - 1;

    if (padTokenId !== undefined) {
      for (let t = seqLen - 1; t >= 0; t--) {
        if (tokensData[b * seqLen + t] !== padTokenId) {
          lastTokenPos = t;
          break;
        }
      }
    }

    // Check if last token is EOS
    const lastToken = tokensData[b * seqLen + lastTokenPos];
    truncated[b] = lastToken !== eosTokenId;
  }

  return truncated;
}

/**
 * Create mask for truncated completions
 *
 * Returns a mask [B, T] where:
 * - 1.0 = keep this token in loss
 * - 0.0 = mask out this token
 *
 * Truncated sequences get all zeros.
 *
 * @param tokens - Token sequences [B, T]
 * @param eosTokenId - EOS token ID
 * @param padTokenId - Padding token ID (optional)
 * @returns Mask array [B, T]
 */
export function createTruncationMask(tokens: MxArray, eosTokenId: number, padTokenId?: number): MxArray {
  const tokensShape = Array.from(tokens.shape());
  const batchSize = Number(tokensShape[0]);
  const seqLen = Number(tokensShape[1]);

  const truncated = detectTruncatedSequences(tokens, eosTokenId, padTokenId);

  // Create mask: 1.0 for non-truncated, 0.0 for truncated
  const maskData = new Float32Array(batchSize * seqLen);

  for (let b = 0; b < batchSize; b++) {
    const maskValue = truncated[b] ? 0.0 : 1.0;
    for (let t = 0; t < seqLen; t++) {
      maskData[b * seqLen + t] = maskValue;
    }
  }

  return MxArray.fromFloat32(maskData, shape(batchSize, seqLen));
}

describe('GRPO Truncation Masking', () => {
  const EOS_TOKEN = 151645; // Qwen3 EOS token
  const PAD_TOKEN = 151643; // Qwen3 PAD token

  describe('detectTruncatedSequences', () => {
    it('should detect sequence without EOS as truncated', () => {
      // Sequence: [1, 2, 3, 4, 5] (no EOS)
      const tokens = MxArray.fromInt32(int32(1, 2, 3, 4, 5), shape(1, 5));

      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN);

      expect(truncated).toEqual([true]);
    });

    it('should detect sequence with EOS as not truncated', () => {
      // Sequence: [1, 2, 3, EOS, PAD]
      const tokens = MxArray.fromInt32(int32(1, 2, 3, EOS_TOKEN, PAD_TOKEN), shape(1, 5));

      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN, PAD_TOKEN);

      expect(truncated).toEqual([false]);
    });

    it('should handle multiple sequences', () => {
      // Batch of 3 sequences:
      // Seq 0: [1, 2, EOS, PAD, PAD] - not truncated
      // Seq 1: [1, 2, 3, 4, 5]       - truncated
      // Seq 2: [1, EOS, PAD, PAD, PAD] - not truncated
      const tokens = MxArray.fromInt32(
        int32(1, 2, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, 1, 2, 3, 4, 5, 1, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN),
        shape(3, 5),
      );

      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN, PAD_TOKEN);

      expect(truncated).toEqual([false, true, false]);
    });

    it('should handle all truncated', () => {
      // All sequences without EOS
      const tokens = MxArray.fromInt32(int32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), shape(2, 5));

      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN);

      expect(truncated).toEqual([true, true]);
    });

    it('should handle all non-truncated', () => {
      // All sequences with EOS
      const tokens = MxArray.fromInt32(
        int32(1, 2, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, 1, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN),
        shape(2, 5),
      );

      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN, PAD_TOKEN);

      expect(truncated).toEqual([false, false]);
    });

    it('should detect EOS at end of sequence', () => {
      // Sequence ends with EOS (no padding)
      const tokens = MxArray.fromInt32(int32(1, 2, 3, 4, EOS_TOKEN), shape(1, 5));

      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN);

      expect(truncated).toEqual([false]);
    });

    it('should handle single-token sequences', () => {
      // Just EOS
      const tokensWithEos = MxArray.fromInt32(int32(EOS_TOKEN), shape(1, 1));
      const truncatedWithEos = detectTruncatedSequences(tokensWithEos, EOS_TOKEN);
      expect(truncatedWithEos).toEqual([false]);

      // No EOS
      const tokensWithoutEos = MxArray.fromInt32(int32(123), shape(1, 1));
      const truncatedWithoutEos = detectTruncatedSequences(tokensWithoutEos, EOS_TOKEN);
      expect(truncatedWithoutEos).toEqual([true]);
    });

    it('should ignore padding when finding last token', () => {
      // Sequence: [1, 2, 3, PAD, PAD]
      // Last non-pad token is 3 (not EOS) → truncated
      const tokens = MxArray.fromInt32(int32(1, 2, 3, PAD_TOKEN, PAD_TOKEN), shape(1, 5));

      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN, PAD_TOKEN);

      expect(truncated).toEqual([true]);
    });
  });

  describe('createTruncationMask', () => {
    it('should create zero mask for truncated sequence', () => {
      const tokens = MxArray.fromInt32(int32(1, 2, 3, 4, 5), shape(1, 5));

      const mask = createTruncationMask(tokens, EOS_TOKEN);
      const maskData = mask.toFloat32();

      // All zeros for truncated sequence
      expect(maskData).toEqual(new Float32Array([0, 0, 0, 0, 0]));
    });

    it('should create ones mask for non-truncated sequence', () => {
      const tokens = MxArray.fromInt32(int32(1, 2, 3, EOS_TOKEN, PAD_TOKEN), shape(1, 5));

      const mask = createTruncationMask(tokens, EOS_TOKEN, PAD_TOKEN);
      const maskData = mask.toFloat32();

      // All ones for non-truncated sequence
      expect(maskData).toEqual(new Float32Array([1, 1, 1, 1, 1]));
    });

    it('should create mixed mask for batch', () => {
      // Batch:
      // Seq 0: [1, 2, EOS, PAD, PAD] - not truncated → all 1s
      // Seq 1: [1, 2, 3, 4, 5]       - truncated → all 0s
      const tokens = MxArray.fromInt32(int32(1, 2, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, 1, 2, 3, 4, 5), shape(2, 5));

      const mask = createTruncationMask(tokens, EOS_TOKEN, PAD_TOKEN);
      const maskData = mask.toFloat32();

      expect(maskData).toEqual(
        new Float32Array([
          1,
          1,
          1,
          1,
          1, // Seq 0: not truncated
          0,
          0,
          0,
          0,
          0, // Seq 1: truncated
        ]),
      );
    });

    it('should handle all-truncated batch', () => {
      const tokens = MxArray.fromInt32(int32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), shape(2, 5));

      const mask = createTruncationMask(tokens, EOS_TOKEN);
      const maskData = mask.toFloat32();

      // All zeros
      expect(maskData).toEqual(new Float32Array(10).fill(0));
    });
  });

  describe('Integration with loss computation', () => {
    it('should mask out truncated sequences in loss', () => {
      // Simulate a batch with 1 truncated, 1 not truncated
      const tokens = MxArray.fromInt32(
        int32(
          1,
          2,
          EOS_TOKEN,
          PAD_TOKEN,
          PAD_TOKEN, // Good completion
          1,
          2,
          3,
          4,
          5, // Truncated
        ),
        shape(2, 5),
      );

      const mask = createTruncationMask(tokens, EOS_TOKEN, PAD_TOKEN);

      // Simulate loss per token (all same value)
      const lossPerToken = MxArray.ones(shape(2, 5));

      // Apply mask
      const maskedLoss = lossPerToken.mul(mask);
      const maskedLossData = maskedLoss.toFloat32();

      // First sequence: all 1s (kept)
      expect(maskedLossData[0]).toBe(1);
      expect(maskedLossData[4]).toBe(1);

      // Second sequence: all 0s (masked)
      expect(maskedLossData[5]).toBe(0);
      expect(maskedLossData[9]).toBe(0);

      // Compute mean loss (only over unmasked tokens)
      const totalLoss = maskedLoss.sum(undefined, false).toFloat32()[0];
      const numUnmasked = mask.sum(undefined, false).toFloat32()[0];
      const meanLoss = totalLoss / numUnmasked;

      expect(meanLoss).toBeCloseTo(1.0, 5);
    });

    it('should return zero loss for all-truncated batch', () => {
      // All truncated
      const tokens = MxArray.fromInt32(int32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), shape(2, 5));

      const mask = createTruncationMask(tokens, EOS_TOKEN);

      // Check that mask sum is zero
      const maskSum = mask.sum(undefined, false).toFloat32()[0];
      expect(maskSum).toBe(0);

      // This would trigger a "no valid samples" warning in TRL
    });

    it('should combine with completion mask', () => {
      // Simulate: prompt=[1,2], completion=[3,4,5]
      // Completion mask: [0, 0, 1, 1, 1]
      const completionMask = MxArray.fromFloat32(new Float32Array([0, 0, 1, 1, 1]), shape(1, 5));

      // Tokens without EOS (truncated)
      const tokens = MxArray.fromInt32(int32(1, 2, 3, 4, 5), shape(1, 5));

      const truncationMask = createTruncationMask(tokens, EOS_TOKEN);

      // Combined mask: truncation AND completion
      const combinedMask = completionMask.mul(truncationMask);
      const combinedData = combinedMask.toFloat32();

      // All zeros because sequence is truncated
      expect(combinedData).toEqual(new Float32Array([0, 0, 0, 0, 0]));
    });
  });

  describe('Edge cases', () => {
    it('should handle empty sequences gracefully', () => {
      // This shouldn't happen in practice, but test defensive coding
      const tokens = MxArray.fromInt32(int32(), shape(0, 0));

      // Should not crash
      expect(() => {
        detectTruncatedSequences(tokens, EOS_TOKEN);
      }).not.toThrow();
    });

    it('should handle very long sequences', () => {
      // Sequence of 1000 tokens without EOS
      const longTokens = new Int32Array(1000).fill(123);
      const tokens = MxArray.fromInt32(longTokens, shape(1, 1000));

      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN);

      expect(truncated).toEqual([true]);
    });

    it('should handle EOS in middle of sequence', () => {
      // Sequence: [1, 2, EOS, 4, 5]
      // Last token is 5 (not EOS) → truncated
      const tokens = MxArray.fromInt32(int32(1, 2, EOS_TOKEN, 4, 5), shape(1, 5));

      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN);

      // Truncated because last token isn't EOS
      expect(truncated).toEqual([true]);
    });

    it('should handle multiple EOS tokens', () => {
      // Sequence: [1, EOS, EOS, EOS, PAD]
      // Last non-pad token is EOS → not truncated
      const tokens = MxArray.fromInt32(int32(1, EOS_TOKEN, EOS_TOKEN, EOS_TOKEN, PAD_TOKEN), shape(1, 5));

      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN, PAD_TOKEN);

      expect(truncated).toEqual([false]);
    });
  });

  describe('Performance characteristics', () => {
    it('should handle large batches efficiently', () => {
      const batchSize = 128;
      const seqLen = 512;

      // Create random tokens (no EOS, all truncated)
      const tokensData = new Int32Array(batchSize * seqLen);
      for (let i = 0; i < tokensData.length; i++) {
        tokensData[i] = Math.floor(Math.random() * 50000) + 1; // Avoid 0 and EOS
      }

      const tokens = MxArray.fromInt32(tokensData, shape(batchSize, seqLen));

      const startTime = Date.now();
      const truncated = detectTruncatedSequences(tokens, EOS_TOKEN);
      const endTime = Date.now();

      expect(truncated.length).toBe(batchSize);
      expect(truncated.every((t) => t === true)).toBe(true);

      // Should complete in reasonable time (< 100ms for this size)
      const duration = endTime - startTime;
      expect(duration).toBeLessThan(100);
    });
  });
});
