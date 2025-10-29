/**
 * Tests for activation functions
 * Ported from mlx/python/tests/test_nn.py
 */

import { describe, it, expect } from 'vitest';
import { MxArray, Activations } from '@mlx-node/core';
import { createFloat32Array, assertArrayClose } from '../test-utils';

// Helper function for backward compatibility
function assertArrayEqual(actual: MxArray, expected: MxArray, atol: number = 1e-5): void {
  assertArrayClose(actual, expected, atol, atol);
}

describe('Activation Functions', () => {
  describe('ReLU', () => {
    it('should compute ReLU correctly', () => {
      const x = createFloat32Array([1.0, -1.0, 0.0, 2.5, -0.5], [5]);
      const y = Activations.relu(x);
      const expected = createFloat32Array([1.0, 0.0, 0.0, 2.5, 0.0], [5]);

      assertArrayEqual(y, expected);
      const shape = Array.from(y.shape()).map((x) => Number(x));
      expect(shape).toEqual([5]);
    });

    it('should handle negative values', () => {
      const x = createFloat32Array([-1, -2, -3, -4], [4]);
      const y = Activations.relu(x);
      const expected = createFloat32Array([0, 0, 0, 0], [4]);

      assertArrayEqual(y, expected);
    });

    it('should handle 2D arrays', () => {
      const x = createFloat32Array([1, -1, 0, 2, -2, 3], [2, 3]);
      const y = Activations.relu(x);
      const expected = createFloat32Array([1, 0, 0, 2, 0, 3], [2, 3]);

      assertArrayEqual(y, expected);
      expect(Array.from(y.shape()).map((x) => Number(x))).toEqual([2, 3]);
    });
  });

  describe('Sigmoid', () => {
    it('should compute sigmoid correctly', () => {
      const x = createFloat32Array([0.0, 1.0, -1.0], [3]);
      const y = Activations.sigmoid(x);

      // Expected values computed from: 1 / (1 + exp(-x))
      const expected = createFloat32Array([0.5, 0.7310586, 0.26894143], [3]);

      assertArrayEqual(y, expected, 1e-5);
    });

    it('should handle extreme values', () => {
      const x = createFloat32Array([10.0, -10.0], [2]);
      const y = Activations.sigmoid(x);

      // For large positive x, sigmoid -> 1
      // For large negative x, sigmoid -> 0
      const expected = createFloat32Array([0.9999546, 0.0000454], [2]);

      assertArrayEqual(y, expected, 1e-4);
    });
  });

  describe('SiLU (Swish)', () => {
    it('should compute SiLU correctly', () => {
      const x = createFloat32Array([-2, -1, 0, 1, 2], [5]);
      const y = Activations.silu(x);

      // SiLU(x) = x * sigmoid(x)
      // Expected values from MLX Python tests
      const expected = createFloat32Array([-0.23840584, -0.26894143, 0.0, 0.7310586, 1.7615942], [5]);

      assertArrayEqual(y, expected, 1e-5);
    });
  });

  describe('GELU', () => {
    it('should compute GELU approximation correctly', () => {
      // Test values from MLX Python tests
      const inputs = [1.15286231, -0.81037411, 0.35816911, 0.77484438, 0.66276414];
      const x = createFloat32Array(inputs, [5]);
      const y = Activations.gelu(x);

      // Expected values from JAX/MLX
      const expected = createFloat32Array([1.0093501, -0.16925684, 0.22918941, 0.60498625, 0.49459383], [5]);

      assertArrayEqual(y, expected, 1e-3); // GELU approximation has higher tolerance
    });

    it('should handle edge cases', () => {
      const x = createFloat32Array([0, -3, 3], [3]);
      const y = Activations.gelu(x);

      // At x=0, GELU ≈ 0
      // For large negative x, GELU ≈ 0
      // For large positive x, GELU ≈ x
      const yData = y.toFloat32();
      expect(Math.abs(yData[0])).toBeLessThan(0.01);
      expect(Math.abs(yData[1])).toBeLessThan(0.01);
      expect(Math.abs(yData[2] - 3)).toBeLessThan(0.01);
    });
  });

  describe('Softmax', () => {
    it('should compute softmax correctly', () => {
      const x = createFloat32Array([1.0, -1.0, 0.0], [3]);
      const y = Activations.softmax(x);

      // Softmax should sum to 1
      const sum = y.toFloat32().reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1.0)).toBeLessThan(1e-5);

      // Expected values
      const expected = createFloat32Array([0.6652, 0.09, 0.2447], [3]);
      assertArrayEqual(y, expected, 1e-3);
    });

    it('should handle 2D arrays with axis', () => {
      const x = createFloat32Array([1, 2, 3, 4], [2, 2]);
      const y = Activations.softmax(x, -1); // Along last axis

      // Each row should sum to 1
      const yData = y.toFloat32();
      const row1Sum = yData[0] + yData[1];
      const row2Sum = yData[2] + yData[3];

      expect(Math.abs(row1Sum - 1.0)).toBeLessThan(1e-5);
      expect(Math.abs(row2Sum - 1.0)).toBeLessThan(1e-5);
    });

    it('should handle numerical stability with large values', () => {
      const x = createFloat32Array([1000, 1001, 999], [3]);
      const y = Activations.softmax(x);

      // Should not produce NaN or Inf
      const yData = y.toFloat32();
      yData.forEach((val) => {
        expect(isNaN(val)).toBe(false);
        expect(isFinite(val)).toBe(true);
      });

      // Should still sum to 1
      const sum = yData.reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1.0)).toBeLessThan(1e-5);
    });
  });

  describe('LogSoftmax', () => {
    it('should compute log_softmax correctly', () => {
      const x = createFloat32Array([1.0, 2.0, 3.0], [3]);
      const y = Activations.logSoftmax(x);

      // log_softmax should be log of softmax
      const softmax = Activations.softmax(x);
      const logSoftmaxData = y.toFloat32();
      const softmaxData = softmax.toFloat32();

      for (let i = 0; i < softmaxData.length; i++) {
        const expected = Math.log(softmaxData[i]);
        expect(Math.abs(logSoftmaxData[i] - expected)).toBeLessThan(1e-5);
      }
    });

    it('should handle numerical stability', () => {
      const x = createFloat32Array([1000, 1001, 999], [3]);
      const y = Activations.logSoftmax(x);

      // Should not produce NaN or Inf
      const yData = y.toFloat32();
      yData.forEach((val) => {
        expect(isNaN(val)).toBe(false);
        expect(isFinite(val)).toBe(true);
      });
    });
  });

  describe('SwiGLU', () => {
    it('should compute SwiGLU correctly', () => {
      const gate = createFloat32Array([1, 2, 3, 4], [4]);
      const up = createFloat32Array([2, 3, 4, 5], [4]);
      const y = Activations.swiglu(gate, up);

      // SwiGLU(gate, up) = SiLU(gate) * up
      const silu_gate = Activations.silu(gate);
      const expected = silu_gate.mul(up);

      assertArrayEqual(y, expected, 1e-5);
    });

    it('should handle 2D arrays', () => {
      const gate = createFloat32Array([1, 2, 3, 4], [2, 2]);
      const up = createFloat32Array([2, 3, 4, 5], [2, 2]);
      const y = Activations.swiglu(gate, up);

      expect(Array.from(y.shape()).map((x) => Number(x))).toEqual([2, 2]);

      // Verify non-zero output
      const yData = y.toFloat32();
      const hasNonZero = yData.some((val) => val !== 0);
      expect(hasNonZero).toBe(true);
    });
  });
});

describe('Edge Cases and Boundary Conditions', () => {
  it('should handle empty-like inputs gracefully', () => {
    const x = MxArray.zeros(BigInt64Array.from([0n]), null);
    // These should not crash
    expect(() => Activations.relu(x)).not.toThrow();
  });

  it('should handle single element arrays', () => {
    const x = createFloat32Array([2.0], [1]);

    const relu = Activations.relu(x);
    expect(relu.toFloat32()[0]).toBeCloseTo(2.0, 5);

    const sigmoid = Activations.sigmoid(x);
    expect(sigmoid.toFloat32()[0]).toBeCloseTo(0.8807971, 5);
  });

  it('should preserve input shape', () => {
    const shapes: number[][] = [[5], [2, 3], [2, 3, 4], [1, 1, 1, 1]];

    shapes.forEach((shape) => {
      const size = shape.reduce((a, b) => a * b, 1);
      const data = Array.from({ length: size }, (_, i) => i - size / 2);
      const x = createFloat32Array(data, shape);

      const relu = Activations.relu(x);
      const sigmoid = Activations.sigmoid(x);
      const silu = Activations.silu(x);

      expect(Array.from(relu.shape()).map((x) => Number(x))).toEqual(shape);
      expect(Array.from(sigmoid.shape()).map((x) => Number(x))).toEqual(shape);
      expect(Array.from(silu.shape()).map((x) => Number(x))).toEqual(shape);
    });
  });
});

describe('Activation Approximations', () => {
  it('should have GELU approximation within tolerance', () => {
    // Test approximation accuracy as in MLX tests
    const x = MxArray.arange(-6.0, 6.0, 0.12);
    const y = Activations.gelu(x);

    // Since we're using the tanh approximation, tolerance is higher
    const xData = x.toFloat32();
    const yData = y.toFloat32();

    // GELU is not strictly monotonic - it has a small dip around x=-0.7
    // Check monotonicity only in regions where it should be monotonic
    // For x > 0, GELU should be strictly increasing
    const zeroIdx = xData.findIndex((val) => val >= 0);
    for (let i = zeroIdx + 1; i < yData.length; i++) {
      expect(yData[i]).toBeGreaterThanOrEqual(yData[i - 1]);
    }

    // At x=0, GELU should be approximately 0
    expect(Math.abs(yData[zeroIdx])).toBeLessThan(0.1);

    // For large negative values, GELU should approach 0
    expect(Math.abs(yData[0])).toBeLessThan(0.01);

    // For large positive values, GELU should approach x
    expect(Math.abs(yData[yData.length - 1] - xData[xData.length - 1])).toBeLessThan(0.01);
  });
});
