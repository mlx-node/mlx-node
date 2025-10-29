/**
 * Numerical Consistency Tests (Metal GPU Validation)
 *
 * These tests validate that MLX operations on Metal GPU produce:
 * 1. Deterministic results (same inputs → same outputs)
 * 2. Numerically stable gradients
 * 3. Consistent behavior across multiple runs
 * 4. Precise computations within acceptable tolerances
 *
 * Based on MLX-LM's CPU vs GPU consistency tests, but adapted for
 * Metal-only testing until stream switching is exposed.
 *
 * Future work: Add explicit CPU stream testing when stream API is exposed.
 */

import { describe, it, expect } from 'vitest';
import { MxArray, Linear, RMSNorm, Attention, Losses, Activations, Gradients, Adam } from '@mlx-node/core';
import { shape, float32 } from '../test-utils.js';

// MLX-LM standard tolerance
function assertClose(actual: MxArray, expected: MxArray, rtol: number = 1e-4, atol: number = 1e-6) {
  const actualData = actual.toFloat32();
  const expectedData = expected.toFloat32();

  expect(actualData.length).toBe(expectedData.length);

  for (let i = 0; i < actualData.length; i++) {
    const diff = Math.abs(actualData[i] - expectedData[i]);
    const tolerance = atol + rtol * Math.abs(expectedData[i]);

    if (diff > tolerance) {
      throw new Error(
        `Arrays not close at index ${i}: actual=${actualData[i]}, expected=${expectedData[i]}, diff=${diff}, tolerance=${tolerance}`,
      );
    }
  }
}

describe('Numerical Consistency (Metal GPU Validation)', () => {
  describe('Deterministic Operations', () => {
    it('should produce identical results for same inputs (basic ops)', () => {
      const x = MxArray.fromFloat32(float32(1, 2, 3, 4), shape(2, 2));
      const y = MxArray.fromFloat32(float32(5, 6, 7, 8), shape(2, 2));

      // Run operation multiple times
      const result1 = x.add(y).mul(x).exp();
      result1.eval();

      const result2 = x.add(y).mul(x).exp();
      result2.eval();

      const result3 = x.add(y).mul(x).exp();
      result3.eval();

      // All results should be identical
      assertClose(result1, result2, 1e-10, 1e-10);
      assertClose(result2, result3, 1e-10, 1e-10);
    });

    it('should produce identical results for matmul operations', () => {
      const a = MxArray.fromFloat32(float32(1, 2, 3, 4, 5, 6), shape(2, 3));
      const b = MxArray.fromFloat32(float32(7, 8, 9, 10, 11, 12), shape(3, 2));

      // Run matmul multiple times
      const result1 = a.matmul(b);
      result1.eval();

      const result2 = a.matmul(b);
      result2.eval();

      // Should be identical
      assertClose(result1, result2, 1e-10, 1e-10);
    });

    it('should produce identical results for reduction operations', () => {
      const x = MxArray.fromFloat32(float32(1, 2, 3, 4, 5, 6, 7, 8), shape(2, 4));

      // Test multiple reductions
      const sum1 = x.sum(undefined, false);
      sum1.eval();
      const sum2 = x.sum(undefined, false);
      sum2.eval();

      const mean1 = x.mean(undefined, false);
      mean1.eval();
      const mean2 = x.mean(undefined, false);
      mean2.eval();

      const max1 = x.max(undefined, false);
      max1.eval();
      const max2 = x.max(undefined, false);
      max2.eval();

      assertClose(sum1, sum2, 1e-10, 1e-10);
      assertClose(mean1, mean2, 1e-10, 1e-10);
      assertClose(max1, max2, 1e-10, 1e-10);
    });
  });

  describe('Loss Function Consistency', () => {
    it('should compute MSE loss consistently', () => {
      const pred = MxArray.fromFloat32(float32(1.0, 2.0, 3.0, 4.0), shape(2, 2));
      const target = MxArray.fromFloat32(float32(1.5, 2.5, 2.5, 3.5), shape(2, 2));

      // Compute loss multiple times
      const loss1 = Losses.mse(pred, target);
      loss1.eval();

      const loss2 = Losses.mse(pred, target);
      loss2.eval();

      const loss3 = Losses.mse(pred, target);
      loss3.eval();

      // Should be identical
      assertClose(loss1, loss2, 1e-10, 1e-10);
      assertClose(loss2, loss3, 1e-10, 1e-10);

      // Verify correct value: MSE = mean((pred - target)^2)
      // Differences: [-0.5, -0.5, 0.5, 0.5]
      // Squared: [0.25, 0.25, 0.25, 0.25]
      // Mean: 0.25
      const expectedLoss = MxArray.fromFloat32(float32(0.25), shape(1));
      assertClose(loss1, expectedLoss, 1e-6, 1e-6);
    });

    it('should compute cross-entropy loss consistently', () => {
      const logits = MxArray.fromFloat32(float32(2.0, 1.0, 0.1, 3.0, 1.0, 0.5), shape(2, 3));
      const targets = MxArray.fromInt32(new Int32Array([0, 2]), shape(2));

      // Compute loss multiple times
      const loss1 = Losses.crossEntropy(logits, targets);
      loss1.eval();

      const loss2 = Losses.crossEntropy(logits, targets);
      loss2.eval();

      // Should be identical
      assertClose(loss1, loss2, 1e-6, 1e-6);
    });

    it('should compute KL divergence loss consistently', () => {
      const logitsP = MxArray.fromFloat32(float32(1.0, 2.0, 3.0, 2.0, 1.0, 0.5), shape(2, 3));
      const logitsQ = MxArray.fromFloat32(float32(1.5, 1.5, 2.5, 1.8, 1.2, 0.7), shape(2, 3));

      // Compute KL divergence multiple times
      const kl1 = Losses.klDivergence(logitsP, logitsQ);
      kl1.eval();

      const kl2 = Losses.klDivergence(logitsP, logitsQ);
      kl2.eval();

      const kl3 = Losses.klDivergence(logitsP, logitsQ);
      kl3.eval();

      // Should be identical
      assertClose(kl1, kl2, 1e-6, 1e-6);
      assertClose(kl2, kl3, 1e-6, 1e-6);
    });
  });

  describe('Activation Function Consistency', () => {
    it('should compute SiLU consistently', () => {
      const x = MxArray.fromFloat32(float32(-2.0, -1.0, 0.0, 1.0, 2.0), shape(5));

      const result1 = Activations.silu(x);
      result1.eval();

      const result2 = Activations.silu(x);
      result2.eval();

      assertClose(result1, result2, 1e-10, 1e-10);
    });

    it('should compute softmax consistently', () => {
      const x = MxArray.fromFloat32(float32(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), shape(2, 3));

      const result1 = Activations.softmax(x);
      result1.eval();

      const result2 = Activations.softmax(x);
      result2.eval();

      assertClose(result1, result2, 1e-6, 1e-6);
    });

    it('should compute GELU consistently', () => {
      const x = MxArray.fromFloat32(float32(-1.0, -0.5, 0.0, 0.5, 1.0), shape(5));

      const result1 = Activations.gelu(x);
      result1.eval();

      const result2 = Activations.gelu(x);
      result2.eval();

      assertClose(result1, result2, 1e-10, 1e-10);
    });
  });

  describe('Gradient Consistency', () => {
    it('should compute linear backward consistently', () => {
      const input = MxArray.fromFloat32(float32(1, 2, 3, 4), shape(2, 2));
      const weight = MxArray.fromFloat32(float32(0.5, -0.5, 0.3, -0.3, 0.2, -0.2), shape(3, 2));
      const gradOutput = MxArray.fromFloat32(float32(1, 0.5, 0.3, 1, 0.5, 0.3), shape(2, 3));

      // Compute gradients multiple times
      const grads1 = Gradients.linearBackward(input, weight, gradOutput, false);
      grads1[0].eval();
      grads1[1].eval();

      const grads2 = Gradients.linearBackward(input, weight, gradOutput, false);
      grads2[0].eval();
      grads2[1].eval();

      // Should be identical
      assertClose(grads1[0], grads2[0], 1e-6, 1e-6);
      assertClose(grads1[1], grads2[1], 1e-6, 1e-6);
    });

    it('should compute cross-entropy backward consistently', () => {
      const logits = MxArray.fromFloat32(float32(2.0, 1.0, 0.1, 3.0, 1.0, 0.5), shape(2, 3));
      const targets = MxArray.fromInt32(new Int32Array([0, 2]), shape(2));

      // Compute gradient multiple times (third param is num_classes)
      const grad1 = Gradients.crossEntropyBackward(logits, targets, 3);
      grad1.eval();

      const grad2 = Gradients.crossEntropyBackward(logits, targets, 3);
      grad2.eval();

      // Should be identical
      assertClose(grad1, grad2, 1e-6, 1e-6);
    });

    it('should compute SiLU backward consistently', () => {
      const x = MxArray.fromFloat32(float32(-1.0, 0.0, 1.0, 2.0), shape(4));
      const gradOutput = MxArray.ones(shape(4));

      const grad1 = Gradients.siluBackward(x, gradOutput);
      grad1.eval();

      const grad2 = Gradients.siluBackward(x, gradOutput);
      grad2.eval();

      assertClose(grad1, grad2, 1e-6, 1e-6);
    });
  });

  describe('Layer Consistency', () => {
    it('should produce consistent Linear layer outputs', () => {
      const layer = new Linear(4, 3, true);

      const input = MxArray.fromFloat32(float32(1, 2, 3, 4), shape(1, 4));

      // Forward pass multiple times
      const output1 = layer.forward(input);
      output1.eval();

      const output2 = layer.forward(input);
      output2.eval();

      const output3 = layer.forward(input);
      output3.eval();

      // Should be identical
      assertClose(output1, output2, 1e-10, 1e-10);
      assertClose(output2, output3, 1e-10, 1e-10);
    });

    it('should produce consistent RMSNorm outputs', () => {
      const norm = new RMSNorm(4);

      const input = MxArray.fromFloat32(float32(1, 2, 3, 4), shape(1, 4));

      // Forward pass multiple times
      const output1 = norm.forward(input);
      output1.eval();

      const output2 = norm.forward(input);
      output2.eval();

      assertClose(output1, output2, 1e-6, 1e-6);
    });

    it('should produce consistent Attention outputs', () => {
      const attention = new Attention(
        128, // hidden_size
        4, // num_heads
        4, // num_kv_heads
        undefined, // head_dim (auto)
        10000, // rope_theta
        false, // use_qk_norm
      );

      const input = MxArray.randomNormal(shape(1, 8, 128), 0, 0.02);

      // Forward pass multiple times
      const output1 = attention.forward(input, null, null);
      output1.eval();

      const output2 = attention.forward(input, null, null);
      output2.eval();

      // Should be identical
      assertClose(output1, output2, 1e-6, 1e-6);
    });
  });

  describe('Training Step Consistency', () => {
    it('should produce consistent gradient updates', () => {
      // Create two identical models
      const model1 = new Linear(4, 2, false);
      const model2 = new Linear(4, 2, false);

      // Set same weights
      model2.setWeight(model1.getWeight());

      // Same input and target
      const input = MxArray.fromFloat32(float32(1, 2, 3, 4), shape(1, 4));
      const target = MxArray.fromFloat32(float32(1, 0), shape(1, 2));

      // Training step 1
      const output1 = model1.forward(input);
      output1.eval();
      const gradOutput1 = output1.sub(target);
      gradOutput1.eval();
      const grads1 = Gradients.linearBackward(input, model1.getWeight(), gradOutput1, false);
      grads1[1].eval();

      const optimizer1 = new Adam(0.01);
      const updated1 = optimizer1.updateSingle('weight', model1.getWeight(), grads1[1]);
      updated1.eval();

      // Training step 2
      const output2 = model2.forward(input);
      output2.eval();
      const gradOutput2 = output2.sub(target);
      gradOutput2.eval();
      const grads2 = Gradients.linearBackward(input, model2.getWeight(), gradOutput2, false);
      grads2[1].eval();

      const optimizer2 = new Adam(0.01);
      const updated2 = optimizer2.updateSingle('weight', model2.getWeight(), grads2[1]);
      updated2.eval();

      // Outputs should be identical
      assertClose(output1, output2, 1e-6, 1e-6);

      // Gradients should be identical
      assertClose(grads1[1], grads2[1], 1e-6, 1e-6);

      // Updated weights should be identical
      assertClose(updated1, updated2, 1e-6, 1e-6);
    });

    it('should converge to same solution with same initialization', () => {
      // Create two identical models
      const model1 = new Linear(2, 1, false);
      const model2 = new Linear(2, 1, false);

      // Set same weights
      model2.setWeight(model1.getWeight());

      // Same data
      const X = MxArray.fromFloat32(float32(1, 2, 3, 4, 5, 6), shape(3, 2));
      const y = MxArray.fromFloat32(float32(1, 2, 3), shape(3, 1));

      // Train both models
      const optimizer1 = new Adam(0.1);
      const optimizer2 = new Adam(0.1);

      for (let step = 0; step < 10; step++) {
        // Model 1
        const pred1 = model1.forward(X);
        const gradOut1 = pred1.sub(y).mul(MxArray.fromFloat32(float32(2.0 / 3), shape(1)));
        const grads1 = Gradients.linearBackward(X, model1.getWeight(), gradOut1, false);
        const updated1 = optimizer1.updateSingle('weight', model1.getWeight(), grads1[1]);
        model1.setWeight(updated1);

        // Model 2
        const pred2 = model2.forward(X);
        const gradOut2 = pred2.sub(y).mul(MxArray.fromFloat32(float32(2.0 / 3), shape(1)));
        const grads2 = Gradients.linearBackward(X, model2.getWeight(), gradOut2, false);
        const updated2 = optimizer2.updateSingle('weight', model2.getWeight(), grads2[1]);
        model2.setWeight(updated2);
      }

      // Final weights should be identical
      assertClose(model1.getWeight(), model2.getWeight(), 1e-6, 1e-6);
    });
  });

  describe('Numerical Stability', () => {
    it('should handle large values without overflow in softmax', () => {
      const x = MxArray.fromFloat32(float32(100, 200, 300), shape(3));

      const result = Activations.softmax(x);
      result.eval();

      const data = result.toFloat32();

      // Should not have NaN or Inf
      for (const val of data) {
        expect(isFinite(val)).toBe(true);
        expect(isNaN(val)).toBe(false);
      }

      // Sum should be 1
      const sum = data.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    });

    it('should handle small values in log without underflow', () => {
      const x = MxArray.fromFloat32(float32(1e-10, 1e-8, 1e-6, 1e-4), shape(4));

      const result = x.log();
      result.eval();

      const data = result.toFloat32();

      // Should not have NaN
      for (const val of data) {
        expect(isNaN(val)).toBe(false);
      }

      // Should be negative (log of small positive numbers)
      for (const val of data) {
        expect(val).toBeLessThan(0);
      }
    });

    it('should compute logsumexp stably for large values', () => {
      const x = MxArray.fromFloat32(float32(100, 200, 300, 400), shape(4));

      const lse = x.logsumexp(undefined, false);
      lse.eval();

      const data = lse.toFloat32();

      // Should be finite
      expect(isFinite(data[0])).toBe(true);

      // Should be close to max value (dominated by exp(400))
      expect(data[0]).toBeGreaterThan(399.9);
      expect(data[0]).toBeLessThan(401.0);
    });

    it('should handle division by very small numbers', () => {
      const x = MxArray.fromFloat32(float32(1.0), shape(1));
      const y = MxArray.fromFloat32(float32(1e-10), shape(1));

      const result = x.div(y);
      result.eval();

      const data = result.toFloat32();

      // Should be very large but finite
      expect(isFinite(data[0])).toBe(true);
      expect(data[0]).toBeGreaterThan(1e9);
    });
  });

  describe('Precision Tests', () => {
    it('should maintain precision in long computation chains', () => {
      // Start with known values
      let x = MxArray.fromFloat32(float32(1.0, 2.0, 3.0, 4.0), shape(2, 2));

      // Long chain of operations
      for (let i = 0; i < 5; i++) {
        x = x.add(MxArray.fromFloat32(float32(0.1), shape(1)));
        x = x.mul(MxArray.fromFloat32(float32(0.99), shape(1)));
        x.eval();
      }

      // Compute expected manually
      const expected = [1.0, 2.0, 3.0, 4.0].map((v) => {
        let result = v;
        for (let i = 0; i < 5; i++) {
          result = (result + 0.1) * 0.99;
        }
        return result;
      });

      const actual = x.toFloat32();

      for (let i = 0; i < actual.length; i++) {
        expect(actual[i]).toBeCloseTo(expected[i], 5);
      }
    });

    it('should compute matrix products with high precision', () => {
      // Identity matrix test
      const I = MxArray.fromFloat32(float32(1, 0, 0, 1), shape(2, 2));
      const A = MxArray.fromFloat32(float32(3, 4, 5, 6), shape(2, 2));

      const result = I.matmul(A);
      result.eval();

      // Should equal A exactly
      assertClose(result, A, 1e-10, 1e-10);
    });

    it('should handle mixed operations with consistent precision', () => {
      const a = MxArray.fromFloat32(float32(1.23456789, 2.3456789), shape(2));
      const b = MxArray.fromFloat32(float32(3.456789, 4.56789), shape(2));

      // Complex expression: (a + b) * a - b / 2
      const result = a
        .add(b)
        .mul(a)
        .sub(b.div(MxArray.fromFloat32(float32(2.0), shape(1))));
      result.eval();

      // Compute same expression again
      const result2 = a
        .add(b)
        .mul(a)
        .sub(b.div(MxArray.fromFloat32(float32(2.0), shape(1))));
      result2.eval();

      // Should be identical
      assertClose(result, result2, 1e-10, 1e-10);
    });
  });
});
