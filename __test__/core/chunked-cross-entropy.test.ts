/**
 * Tests for chunked cross-entropy implementation
 *
 * The chunked_logsumexp and efficient_selective_log_softmax functions are internal
 * Rust implementations used to handle large vocabulary sizes (e.g., Qwen3's 151,936 tokens).
 *
 * While these functions aren't directly exported to TypeScript, we test their behavior
 * through the components that use them:
 * - Cross-entropy loss with large vocab
 * - GRPO training with Qwen3-sized vocabularies
 *
 * Background:
 * - Standard logsumexp fails for vocab_size > 65536 due to int32 overflow in indexing
 * - Chunked implementation splits computation into chunks of 65536 or less
 * - Used automatically in efficient_selective_log_softmax when vocab_size > 65536
 */

import { describe, it, expect } from 'vite-plus/test';
import { MxArray, Losses, Activations } from '@mlx-node/core';
import { createFloat32Array, assertFinite, assertShape } from '../test-utils';

describe('Chunked Cross-Entropy (Large Vocabulary)', () => {
  describe('Small vocabulary (baseline)', () => {
    it('should compute cross-entropy correctly for small vocab', () => {
      // Test baseline: vocab size well below chunking threshold
      const batchSize = 2;
      const vocabSize = 1000;

      const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);
      const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize)]), 0, vocabSize);

      const loss = Losses.crossEntropy(logits, targets);

      // Verify shape (scalar)
      assertShape(loss, []);

      // Verify loss is positive and finite
      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeGreaterThan(0);
      expect(isFinite(lossValue)).toBe(true);
    });
  });

  describe('Medium vocabulary (near threshold)', () => {
    it('should handle vocab size just below chunking threshold (65536)', () => {
      // Test edge case: just below the threshold where chunking kicks in
      const batchSize = 2;
      const vocabSize = 65535; // Just below threshold

      const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);
      const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize)]), 0, vocabSize);

      const loss = Losses.crossEntropy(logits, targets);

      assertShape(loss, []);
      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeGreaterThan(0);
      expect(isFinite(lossValue)).toBe(true);
    });

    it('should handle vocab size at chunking threshold (65536)', () => {
      // Test exact threshold where chunking starts
      const batchSize = 2;
      const vocabSize = 65536; // Exact threshold

      const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);
      const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize)]), 0, vocabSize);

      const loss = Losses.crossEntropy(logits, targets);

      assertShape(loss, []);
      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeGreaterThan(0);
      expect(isFinite(lossValue)).toBe(true);
    });
  });

  describe('Large vocabulary (Qwen3 scale)', () => {
    it('should handle vocab size just above chunking threshold', () => {
      // Test: just above threshold, requires chunking
      const batchSize = 2;
      const vocabSize = 70000; // Requires chunking into 2 chunks

      const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);
      const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize)]), 0, vocabSize);

      const loss = Losses.crossEntropy(logits, targets);

      assertShape(loss, []);
      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeGreaterThan(0);
      expect(isFinite(lossValue)).toBe(true);
    });

    it('should handle Qwen3 vocabulary size (151936)', () => {
      // Test realistic case: full Qwen3 vocabulary
      // Use small batch to keep memory reasonable
      const batchSize = 2;
      const vocabSize = 151936; // Qwen3 vocab size

      const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);
      const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize)]), 0, vocabSize);

      const loss = Losses.crossEntropy(logits, targets);

      // Verify output shape and properties
      assertShape(loss, []);
      const lossValue = loss.toFloat32()[0];

      // For uniform random logits with vocab_size=151936:
      // Expected loss ≈ log(151936) ≈ 11.93
      // Allow reasonable tolerance for random variation
      expect(lossValue).toBeGreaterThan(10.0);
      expect(lossValue).toBeLessThan(15.0);
      expect(isFinite(lossValue)).toBe(true);
    });

    it('should handle multiple tokens with large vocab', () => {
      // Test sequence of tokens (more realistic for language modeling)
      const batchSize = 4;
      const seqLen = 8;
      const vocabSize = 151936;

      // Create 3D logits: [batch, seq_len, vocab]
      const logits = MxArray.randomNormal(
        BigInt64Array.from([BigInt(batchSize), BigInt(seqLen), BigInt(vocabSize)]),
        0,
        1,
      );

      // Reshape for cross-entropy: [batch * seq_len, vocab]
      const logitsReshaped = logits.reshape(BigInt64Array.from([BigInt(batchSize * seqLen), BigInt(vocabSize)]));

      const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize * seqLen)]), 0, vocabSize);

      const loss = Losses.crossEntropy(logitsReshaped, targets);

      assertShape(loss, []);
      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeGreaterThan(10.0);
      expect(lossValue).toBeLessThan(15.0);
      expect(isFinite(lossValue)).toBe(true);
    });
  });

  describe('Numerical stability', () => {
    it('should handle extreme logit values with large vocab', () => {
      // Test numerical stability with extreme values
      const batchSize = 2;
      const vocabSize = 100000;

      // Create logits with extreme values
      const logits = MxArray.randomNormal(
        BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]),
        0,
        10, // Large std dev creates extreme values
      );
      const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize)]), 0, vocabSize);

      const loss = Losses.crossEntropy(logits, targets);

      assertShape(loss, []);
      assertFinite(loss);

      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeGreaterThan(0);
    });

    it('should handle uniform logits with large vocab', () => {
      // When all logits are uniform, loss should be -log(1/vocab_size) = log(vocab_size)
      const batchSize = 2;
      const vocabSize = 100000;

      const logits = createFloat32Array(new Float32Array(batchSize * vocabSize).fill(0.0), [batchSize, vocabSize]);
      const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize)]), 0, vocabSize);

      const loss = Losses.crossEntropy(logits, targets);

      const lossValue = loss.toFloat32()[0];
      const expectedLoss = Math.log(vocabSize);

      // Should be close to log(vocab_size)
      expect(Math.abs(lossValue - expectedLoss)).toBeLessThan(0.01);
    });

    it('should handle very confident predictions with large vocab', () => {
      // Test case where model is very confident (high logit for target)
      const batchSize = 2;
      const vocabSize = 100000;

      const logitsData = new Float32Array(batchSize * vocabSize).fill(-10.0);

      // Set high values for target positions
      logitsData[0] = 10.0; // Target 0 for batch 0
      logitsData[vocabSize + 1] = 10.0; // Target 1 for batch 1

      const logits = createFloat32Array(logitsData, [batchSize, vocabSize]);
      const targets = MxArray.fromInt32(new Int32Array([0, 1]), BigInt64Array.from([BigInt(batchSize)]));

      const loss = Losses.crossEntropy(logits, targets);

      const lossValue = loss.toFloat32()[0];

      // With very confident predictions, loss should be very small
      expect(lossValue).toBeLessThan(0.1);
      expect(lossValue).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Comparison with manual computation', () => {
    it('should match manual log-softmax for manageable vocab size', () => {
      // For smaller vocab, we can verify chunked matches non-chunked
      const batchSize = 2;
      const vocabSize = 5000; // Manageable for manual computation

      const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);
      const targets = MxArray.fromInt32(new Int32Array([10, 20]), BigInt64Array.from([BigInt(batchSize)]));

      // Compute using cross-entropy loss (uses efficient_selective_log_softmax internally)
      const loss = Losses.crossEntropy(logits, targets);

      // Manual computation using log-softmax + gather
      const logSoftmax = Activations.logSoftmax(logits);
      const logSoftmaxData = logSoftmax.toFloat32();
      const targetsData = targets.toInt32();

      // Manually gather log probabilities for targets
      let manualLoss = 0;
      for (let i = 0; i < batchSize; i++) {
        const targetIdx = targetsData[i];
        const logProb = logSoftmaxData[i * vocabSize + targetIdx];
        manualLoss += -logProb;
      }
      manualLoss /= batchSize;

      const lossValue = loss.toFloat32()[0];

      // Should match within floating point tolerance
      expect(Math.abs(lossValue - manualLoss)).toBeLessThan(1e-4);
    });
  });

  describe('Edge cases', () => {
    it('should handle single batch element with large vocab', () => {
      const vocabSize = 151936;

      const logits = MxArray.randomNormal(BigInt64Array.from([1n, BigInt(vocabSize)]), 0, 1);
      const targets = MxArray.randint(BigInt64Array.from([1n]), 0, vocabSize);

      const loss = Losses.crossEntropy(logits, targets);

      assertShape(loss, []);
      assertFinite(loss);
    });

    it('should handle large batch with large vocab', () => {
      // Test larger batch (but not too large to avoid memory issues)
      const batchSize = 16;
      const vocabSize = 151936;

      const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);
      const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize)]), 0, vocabSize);

      const loss = Losses.crossEntropy(logits, targets);

      assertShape(loss, []);
      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeGreaterThan(0);
      expect(isFinite(lossValue)).toBe(true);
    });

    it('should handle targets at vocabulary boundaries', () => {
      // Test targets at the first and last vocab indices
      const batchSize = 4;
      const vocabSize = 100000;

      const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);

      // Test boundary cases: first, last, and middle indices
      const targets = MxArray.fromInt32(
        new Int32Array([0, vocabSize - 1, 0, vocabSize - 1]),
        BigInt64Array.from([BigInt(batchSize)]),
      );

      const loss = Losses.crossEntropy(logits, targets);

      assertShape(loss, []);
      assertFinite(loss);
    });
  });

  describe('Ignore index functionality', () => {
    it('should handle ignore index with large vocab', () => {
      // Test that ignore_index works correctly with large vocabularies
      const batchSize = 4;
      const vocabSize = 100000;

      const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);

      // Mix of real targets and ignored indices
      const targets = MxArray.fromInt32(
        new Int32Array([100, -1, 200, -1]), // -1 should be ignored
        BigInt64Array.from([BigInt(batchSize)]),
      );

      const loss = Losses.crossEntropy(logits, targets, undefined, -1);

      assertShape(loss, []);
      assertFinite(loss);

      // Loss should only consider non-ignored samples
      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeGreaterThan(0);
    });
  });

  describe('Performance characteristics', () => {
    it('should handle multiple chunked computations efficiently', () => {
      // Test that multiple independent chunked computations work
      const batchSize = 4;
      const vocabSize = 151936;

      const losses: number[] = [];

      // Run multiple times to verify consistency
      for (let i = 0; i < 3; i++) {
        const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);
        const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize)]), 0, vocabSize);

        const loss = Losses.crossEntropy(logits, targets);
        losses.push(loss.toFloat32()[0]);
      }

      // All losses should be reasonable and finite
      losses.forEach((loss, idx) => {
        expect(loss).toBeGreaterThan(10.0);
        expect(loss).toBeLessThan(15.0);
        expect(isFinite(loss)).toBe(true);
      });
    });
  });
});

describe('Log-Softmax with Large Vocabulary', () => {
  it('should compute log-softmax correctly for large vocab', () => {
    // Verify log-softmax works correctly with large vocabularies
    const batchSize = 2;
    const vocabSize = 100000;

    const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);

    const logSoftmax = Activations.logSoftmax(logits);

    // Verify shape
    assertShape(logSoftmax, [batchSize, vocabSize]);

    // Verify all values are <= 0 (property of log probabilities)
    assertFinite(logSoftmax);
    const values = logSoftmax.toFloat32();
    for (let i = 0; i < values.length; i += 10000) {
      // Sample values
      expect(values[i]).toBeLessThanOrEqual(0);
    }
  });

  it('should sum to 1 in probability space (exp of log-softmax)', () => {
    // Verify that exp(log_softmax) sums to 1
    const batchSize = 2;
    const vocabSize = 1000; // Use smaller vocab for this test to avoid numerical issues

    const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);

    const logSoftmax = Activations.logSoftmax(logits);
    const probs = logSoftmax.exp();

    // Sum along vocab dimension
    const probSum = probs.sum(Int32Array.from([-1]), false);
    const sumValues = probSum.toFloat32();

    // Each batch element should sum to ~1
    for (let i = 0; i < batchSize; i++) {
      expect(Math.abs(sumValues[i] - 1.0)).toBeLessThan(1e-5);
    }
  });
});
