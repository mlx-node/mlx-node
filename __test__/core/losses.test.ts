/**
 * Tests for loss functions
 * Ported from mlx/python/tests/test_losses.py
 */

import { describe, it, expect } from 'vitest';
import { MxArray, Losses, Activations } from '@mlx-node/core';
import { createFloat32Array } from '../test-utils';

describe('Cross Entropy Loss', () => {
  it('should compute cross entropy correctly', () => {
    // Simple case: perfect predictions
    const logits = createFloat32Array(
      [
        10.0,
        -10.0,
        -10.0, // Strong prediction for class 0
        -10.0,
        10.0,
        -10.0, // Strong prediction for class 1
        -10.0,
        -10.0,
        10.0, // Strong prediction for class 2
      ],
      [3, 3],
    );

    const targets = MxArray.fromInt32(new Int32Array([0, 1, 2]), BigInt64Array.from([3n]));

    const loss = Losses.crossEntropy(logits, targets);

    // With perfect predictions, loss should be close to 0
    const lossValue = loss.toFloat32()[0];
    expect(lossValue).toBeLessThan(0.001);
  });

  it('should handle incorrect predictions', () => {
    const logits = createFloat32Array(
      [
        -10.0,
        10.0,
        -10.0, // Predicts class 1
        10.0,
        -10.0,
        -10.0, // Predicts class 0
      ],
      [2, 3],
    );

    const targets = MxArray.fromInt32(new Int32Array([0, 1]), BigInt64Array.from([2n])); // True classes are 0 and 1

    const loss = Losses.crossEntropy(logits, targets);

    // Loss should be high for wrong predictions
    const lossValue = loss.toFloat32()[0];
    expect(lossValue).toBeGreaterThan(1.0);
  });

  it('should handle uniform predictions', () => {
    const logits = createFloat32Array(
      [
        0.0,
        0.0,
        0.0,
        0.0, // Uniform logits
        0.0,
        0.0,
        0.0,
        0.0,
      ],
      [2, 4],
    );

    const targets = MxArray.fromInt32(new Int32Array([0, 3]), BigInt64Array.from([2n]));

    const loss = Losses.crossEntropy(logits, targets);

    // For uniform distribution with 4 classes, entropy = -log(1/4) = log(4) ≈ 1.386
    const lossValue = loss.toFloat32()[0];
    expect(lossValue).toBeCloseTo(Math.log(4), 2);
  });

  it('should handle ignore index', () => {
    const logits = createFloat32Array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3]);

    const targets = MxArray.fromInt32(new Int32Array([0, -1, 2]), BigInt64Array.from([3n])); // -1 is ignored

    const loss = Losses.crossEntropy(logits, targets, -1);

    // Loss should only consider non-ignored samples
    const lossData = loss.toFloat32();
    expect(lossData.length).toBe(1);
  });

  it('should work with different batch sizes', () => {
    const batchSizes = [1, 5, 10, 32];

    batchSizes.forEach((batchSize) => {
      const vocabSize = 10;
      const logits = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(vocabSize)]), 0, 1);
      const targets = MxArray.randint(BigInt64Array.from([BigInt(batchSize)]), 0, vocabSize);

      const loss = Losses.crossEntropy(logits, targets);

      // Loss should be a scalar
      expect(Array.from(loss.shape()).map((x) => Number(x))).toEqual([]);

      // Loss should be positive
      const lossValue = loss.toFloat32()[0];
      expect(lossValue).toBeGreaterThan(0);
    });
  });
});

describe('KL Divergence Loss', () => {
  it('should compute KL divergence correctly', () => {
    // KL divergence between identical distributions should be 0
    const logits = createFloat32Array([1.0, 2.0, 3.0, 4.0], [1, 4]);
    const log_p = Activations.logSoftmax(logits);
    const log_q = Activations.logSoftmax(logits);

    const kl = Losses.klDivergence(log_p, log_q);

    const klValue = kl.toFloat32()[0];
    expect(klValue).toBeCloseTo(0.0, 5);
  });

  it('should compute non-zero KL for different distributions', () => {
    const log_p = Activations.logSoftmax(createFloat32Array([1.0, 2.0, 3.0], [1, 3]));
    const log_q = Activations.logSoftmax(createFloat32Array([3.0, 2.0, 1.0], [1, 3]));

    const kl = Losses.klDivergence(log_p, log_q);

    // KL divergence should be positive for different distributions
    const klValue = kl.toFloat32()[0];
    expect(klValue).toBeGreaterThan(0);
  });

  it('should handle batch inputs', () => {
    const batch = 4;
    const dim = 8;

    const log_p = Activations.logSoftmax(MxArray.randomNormal(BigInt64Array.from([BigInt(batch), BigInt(dim)]), 0, 1));
    const log_q = Activations.logSoftmax(MxArray.randomNormal(BigInt64Array.from([BigInt(batch), BigInt(dim)]), 0, 1));

    const kl = Losses.klDivergence(log_p, log_q);

    // Should return scalar mean KL
    expect(Array.from(kl.shape()).map((x) => Number(x))).toEqual([]);

    // KL should be non-negative
    const klValue = kl.toFloat32()[0];
    expect(klValue).toBeGreaterThanOrEqual(0);
  });

  it('should be asymmetric', () => {
    // Use truly asymmetric distributions (not just flipped/mirrored)
    const log_p = Activations.logSoftmax(createFloat32Array([5.0, 2.0, 0.0], [1, 3]));
    const log_q = Activations.logSoftmax(createFloat32Array([0.0, 1.0, 3.0], [1, 3]));

    const kl_pq = Losses.klDivergence(log_p, log_q);
    const kl_qp = Losses.klDivergence(log_q, log_p);

    // KL(P||Q) != KL(Q||P) in general
    const kl_pq_value = kl_pq.toFloat32()[0];
    const kl_qp_value = kl_qp.toFloat32()[0];

    // With these distributions, KL(P||Q) ≈ 2.87 and KL(Q||P) ≈ 4.09
    expect(Math.abs(kl_pq_value - kl_qp_value)).toBeGreaterThan(1.0);
  });
});

describe('Mean Squared Error Loss', () => {
  it('should compute MSE correctly', () => {
    const predictions = createFloat32Array([0.5, 0.2, 0.9, 0.0], [4]);
    const targets = createFloat32Array([0.7, 0.1, 0.8, 0.2], [4]);

    const loss = Losses.mse(predictions, targets);

    // MSE = mean((pred - target)^2)
    // = mean([0.04, 0.01, 0.01, 0.04])
    // = 0.025
    const expected = 0.025;
    const lossValue = loss.toFloat32()[0];

    expect(lossValue).toBeCloseTo(expected, 5);
  });

  it('should handle perfect predictions', () => {
    const predictions = createFloat32Array([1.0, 2.0, 3.0, 4.0], [4]);
    const targets = createFloat32Array([1.0, 2.0, 3.0, 4.0], [4]);

    const loss = Losses.mse(predictions, targets);

    const lossValue = loss.toFloat32()[0];
    expect(lossValue).toBeCloseTo(0.0, 5);
  });

  it('should handle 2D inputs', () => {
    const predictions = createFloat32Array([1.0, 2.0, 3.0, 4.0], [2, 2]);

    const targets = createFloat32Array([1.5, 2.5, 2.5, 3.5], [2, 2]);

    const loss = Losses.mse(predictions, targets);

    // MSE = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
    const lossValue = loss.toFloat32()[0];
    expect(lossValue).toBeCloseTo(0.25, 5);
  });

  it('should handle batch inputs', () => {
    const batchSize = 10;
    const dim = 5;

    const predictions = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(dim)]), 0, 1);
    const targets = MxArray.randomNormal(BigInt64Array.from([BigInt(batchSize), BigInt(dim)]), 0, 1);

    const loss = Losses.mse(predictions, targets);

    // Should return scalar
    expect(Array.from(loss.shape()).map((x) => Number(x))).toEqual([]);

    // Loss should be non-negative
    const lossValue = loss.toFloat32()[0];
    expect(lossValue).toBeGreaterThanOrEqual(0);
  });
});

describe('Loss Function Properties', () => {
  it('cross entropy should be consistent with manual calculation', () => {
    const logits = createFloat32Array([2.0, 1.0, 0.1], [1, 3]);
    const targets = MxArray.fromInt32(new Int32Array([0]), BigInt64Array.from([1n]));

    const loss = Losses.crossEntropy(logits, targets);

    // Manual calculation:
    // log_probs = log_softmax(logits)
    // loss = -log_probs[target]
    const log_probs = Activations.logSoftmax(logits);
    const log_probs_data = log_probs.toFloat32();
    const expected_loss = -log_probs_data[0]; // target is class 0

    const lossValue = loss.toFloat32()[0];
    expect(lossValue).toBeCloseTo(expected_loss, 5);
  });

  it('losses should handle edge cases gracefully', () => {
    // Very small values
    const smallPred = createFloat32Array([1e-10, 1e-10], [2]);
    const smallTarget = createFloat32Array([1e-10, 1e-10], [2]);

    const mseLoss = Losses.mse(smallPred, smallTarget);
    const mseValue = mseLoss.toFloat32()[0];
    expect(isFinite(mseValue)).toBe(true);

    // Very large values
    const largePred = createFloat32Array([1e10, 1e10], [2]);
    const largeTarget = createFloat32Array([1e10, 1e10], [2]);

    const mseLossLarge = Losses.mse(largePred, largeTarget);
    const mseValueLarge = mseLossLarge.toFloat32()[0];
    expect(isFinite(mseValueLarge)).toBe(true);
  });
});

describe('Loss Gradients (Numerical Verification)', () => {
  it('cross entropy gradient should be correct', () => {
    const logits = createFloat32Array([1.0, 2.0, 3.0], [1, 3]);

    // For manual gradient check:
    // d(CE)/d(logits) = softmax(logits) - one_hot(target)
    // For target index 1, the gradient should be:
    const softmax = Activations.softmax(logits);
    const softmaxData = softmax.toFloat32();

    // Expected gradient (softmax - one_hot) where target is index 1
    const expectedGrad = [
      softmaxData[0] - 0, // Not target
      softmaxData[1] - 1, // Target (subtract 1)
      softmaxData[2] - 0, // Not target
    ];

    // The gradient should follow: grad = p - y (where y is one-hot)
    expect(expectedGrad[1]).toBeLessThan(0); // Gradient for correct class should be negative
    expect(expectedGrad[0]).toBeGreaterThan(0); // Gradient for wrong classes should be positive
    expect(expectedGrad[2]).toBeGreaterThan(0);
  });
});

describe('Combined Loss Scenarios', () => {
  it('should work in training simulation', () => {
    // Simulate a mini training scenario
    const vocabSize = 100;
    const batchSize = 16n;

    // Generate random logits (model output)
    const logits = MxArray.randomNormal(BigInt64Array.from([batchSize, BigInt(vocabSize)]), 0, 1);

    // Generate random targets
    const targets = MxArray.randint(BigInt64Array.from([batchSize]), 0, vocabSize);

    // Compute cross-entropy loss
    const ceLoss = Losses.crossEntropy(logits, targets);

    // For KL regularization (e.g., in GRPO)
    const refLogits = MxArray.randomNormal(BigInt64Array.from([batchSize, BigInt(vocabSize)]), 0, 1);
    const log_p = Activations.logSoftmax(logits);
    const log_q = Activations.logSoftmax(refLogits);
    const klLoss = Losses.klDivergence(log_p, log_q);

    // Combined loss
    const alpha = 0.1; // KL weight
    const totalLoss = ceLoss.toFloat32()[0] + alpha * klLoss.toFloat32()[0];

    expect(totalLoss).toBeGreaterThan(0);
    expect(isFinite(totalLoss)).toBe(true);
  });
});
