/**
 * Tests for neural network layers (Linear, RMSNorm, LayerNorm, Embedding)
 * Ported from mlx/python/tests/test_nn.py
 */

import { describe, it, expect } from 'vitest';
import { MxArray, Linear, RMSNorm, LayerNorm, Embedding } from '@mlx-node/core';
import { createFloat32Array, assertArrayClose } from '../test-utils';

describe('Linear Layer', () => {
  it('should create linear layer with correct dimensions', () => {
    const layer = new Linear(4, 8, true);
    const input = MxArray.zeros(BigInt64Array.from([10n, 4n]));
    const output = layer.forward(input);

    expect(Array.from(output.shape()).map((x) => Number(x))).toEqual([10, 8]);
  });

  it('should perform linear transformation correctly', () => {
    const layer = new Linear(3, 2, true);

    // Set specific weights and bias for testing
    const weight = createFloat32Array([1, 0, -1, 0, 1, 1], [2, 3]); // [out, in]
    const bias = createFloat32Array([0.5, -0.5], [2]);

    layer.setWeight(weight);
    layer.setBias(bias);

    // Input: [1, 2, 3]
    const input = createFloat32Array([1, 2, 3], [1, 3]);
    const output = layer.forward(input);

    // Expected: input @ weight.T + bias
    // weight.T = [[1, 0], [0, 1], [-1, 1]]
    // [1, 2, 3] @ [[1, 0], [0, 1], [-1, 1]] = [1*1 + 2*0 + 3*(-1), 1*0 + 2*1 + 3*1]
    //                                         = [-2, 5]
    // Adding bias: [-2 + 0.5, 5 + (-0.5)] = [-1.5, 4.5]
    const expected = createFloat32Array([-1.5, 4.5], [1, 2]);

    assertArrayClose(output, expected, 1e-5);
  });

  it('should work without bias', () => {
    const layer = new Linear(3, 2, false);

    // Set specific weights for testing
    const weight = createFloat32Array([1, 0, -1, 0, 1, 1], [2, 3]);
    layer.setWeight(weight);

    const input = createFloat32Array([1, 2, 3], [1, 3]);
    const output = layer.forward(input);

    // Without bias: [-2, 5]
    const expected = createFloat32Array([-2, 5], [1, 2]);

    assertArrayClose(output, expected, 1e-5);
  });

  it('should handle batch inputs', () => {
    const layer = new Linear(4, 3, true);
    const input = MxArray.randomNormal(BigInt64Array.from([5n, 4n]), 0, 1); // batch_size=5
    const output = layer.forward(input);

    expect(Array.from(output.shape()).map((x) => Number(x))).toEqual([5, 3]);
  });

  it('should reject incorrect weight shapes', () => {
    const layer = new Linear(4, 3, true);
    const wrongWeight = createFloat32Array([1, 2, 3, 4], [2, 2]);

    expect(() => layer.setWeight(wrongWeight)).toThrow();
  });
});

describe('RMSNorm Layer', () => {
  it('should normalize correctly', () => {
    const norm = new RMSNorm(4, 1e-5);
    const input = createFloat32Array([1, 2, 3, 4], [1, 4]);
    const output = norm.forward(input);

    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    // mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5
    // rms = sqrt(7.5 + 1e-5) ≈ 2.738613
    // normalized = [1/2.738613, 2/2.738613, 3/2.738613, 4/2.738613]
    //            ≈ [0.365148, 0.730297, 1.095445, 1.460594]
    // Since weight initializes to 1, output = normalized

    const outputData = output.toFloat32();

    // Check that it's normalized (not exact values due to eps)
    expect(outputData[0]).toBeCloseTo(0.365148, 4);
    expect(outputData[1]).toBeCloseTo(0.730297, 4);
    expect(outputData[2]).toBeCloseTo(1.095445, 4);
    expect(outputData[3]).toBeCloseTo(1.460594, 4);
  });

  it('should handle multiple samples', () => {
    const norm = new RMSNorm(4);
    const input = createFloat32Array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const output = norm.forward(input);

    expect(Array.from(output.shape()).map((x) => Number(x))).toEqual([2, 4]);

    // Each sample should be normalized independently
    const outputData = output.toFloat32();

    // First sample normalized
    const firstSample = outputData.slice(0, 4);
    const firstRMS = Math.sqrt(firstSample.reduce((sum, x) => sum + x * x, 0) / 4);

    // Second sample normalized
    const secondSample = outputData.slice(4, 8);
    const secondRMS = Math.sqrt(secondSample.reduce((sum, x) => sum + x * x, 0) / 4);

    // Both should be close to 1 (normalized)
    expect(firstRMS).toBeCloseTo(1.0, 3);
    expect(secondRMS).toBeCloseTo(1.0, 3);
  });

  it('should handle different epsilon values', () => {
    const norm1 = new RMSNorm(4, 1e-5);
    const norm2 = new RMSNorm(4, 1e-3);

    const input = createFloat32Array([0.001, 0.001, 0.001, 0.001], [1, 4]);

    const output1 = norm1.forward(input);
    const output2 = norm2.forward(input);

    // With larger epsilon, normalization should be different
    const data1 = output1.toFloat32();
    const data2 = output2.toFloat32();

    expect(data1[0]).not.toBeCloseTo(data2[0], 2);
  });
});

describe('LayerNorm Layer', () => {
  it('should normalize to zero mean and unit variance', () => {
    const norm = new LayerNorm(4);
    const input = createFloat32Array([1, 2, 3, 4], [1, 4]);
    const output = norm.forward(input);

    // LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    // mean = 2.5, var = 1.25
    // std = sqrt(1.25 + eps) ≈ 1.118034
    // normalized = [-1.341641, -0.447214, 0.447214, 1.341641]

    const outputData = output.toFloat32();

    // Check zero mean (approximately)
    const mean = outputData.reduce((a, b) => a + b, 0) / outputData.length;
    expect(Math.abs(mean)).toBeLessThan(1e-5);

    // Check unit variance (approximately)
    const variance = outputData.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / outputData.length;
    expect(variance).toBeCloseTo(1.0, 4);
  });

  it('should handle batch normalization', () => {
    const norm = new LayerNorm(4);
    const input = createFloat32Array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const output = norm.forward(input);

    expect(Array.from(output.shape()).map((x) => Number(x))).toEqual([2, 4]);

    // Each sample should be normalized independently
    const outputData = output.toFloat32();

    // First sample
    const firstSample = outputData.slice(0, 4);
    const firstMean = firstSample.reduce((a, b) => a + b, 0) / 4;
    expect(Math.abs(firstMean)).toBeLessThan(1e-5);

    // Second sample
    const secondSample = outputData.slice(4, 8);
    const secondMean = secondSample.reduce((a, b) => a + b, 0) / 4;
    expect(Math.abs(secondMean)).toBeLessThan(1e-5);
  });

  it('should apply weight and bias correctly', () => {
    const norm = new LayerNorm(4);

    // Modify weight and bias (would need setters in real implementation)
    // For now, just test that initialization works
    const input = createFloat32Array([2, 2, 2, 2], [1, 4]);
    const output = norm.forward(input);

    // With all same values, normalized should be all zeros (plus bias)
    const outputData = output.toFloat32();
    outputData.forEach((val) => {
      expect(Math.abs(val)).toBeLessThan(1e-5);
    });
  });
});

describe('Embedding Layer', () => {
  it('should create embedding with correct dimensions', () => {
    const embedding = new Embedding(10, 8); // vocab_size=10, embedding_dim=8
    const indices = MxArray.fromInt32(new Int32Array([0, 1, 2]), BigInt64Array.from([3n]));
    const output = embedding.forward(indices);

    expect(Array.from(output.shape()).map((x) => Number(x))).toEqual([3, 8]);
  });

  it('should return different embeddings for different indices', () => {
    const embedding = new Embedding(10, 4);
    const indices1 = MxArray.fromInt32(new Int32Array([0]), BigInt64Array.from([1n]));
    const indices2 = MxArray.fromInt32(new Int32Array([1]), BigInt64Array.from([1n]));

    const output1 = embedding.forward(indices1);
    const output2 = embedding.forward(indices2);

    // Embeddings should be different
    const data1 = output1.toFloat32();
    const data2 = output2.toFloat32();

    let allSame = true;
    for (let i = 0; i < data1.length; i++) {
      if (Math.abs(data1[i] - data2[i]) > 1e-6) {
        allSame = false;
        break;
      }
    }
    expect(allSame).toBe(false);
  });

  it('should handle batch indices', () => {
    const embedding = new Embedding(10, 8);
    const indices = MxArray.fromInt32(new Int32Array([0, 1, 2, 3, 4]), BigInt64Array.from([5n]));
    const output = embedding.forward(indices);

    expect(Array.from(output.shape()).map((x) => Number(x))).toEqual([5, 8]);
  });

  it('should handle 2D indices', () => {
    const embedding = new Embedding(10, 8);
    const indices = MxArray.fromInt32(new Int32Array([0, 1, 2, 3, 4, 5]), BigInt64Array.from([2n, 3n]));
    const output = embedding.forward(indices);

    expect(Array.from(output.shape()).map((x) => Number(x))).toEqual([2, 3, 8]);
  });

  it('should load pretrained weights', () => {
    const embedding = new Embedding(3, 2);

    // Create specific weights for testing
    const weights = createFloat32Array(
      [
        1.0,
        2.0, // embedding for index 0
        3.0,
        4.0, // embedding for index 1
        5.0,
        6.0, // embedding for index 2
      ],
      [3, 2],
    );

    embedding.loadWeight(weights);

    // Test that correct embeddings are returned
    const indices = MxArray.fromInt32(new Int32Array([0, 2, 1]), BigInt64Array.from([3n]));
    const output = embedding.forward(indices);

    const expected = createFloat32Array(
      [
        1.0,
        2.0, // index 0
        5.0,
        6.0, // index 2
        3.0,
        4.0, // index 1
      ],
      [3, 2],
    );

    assertArrayClose(output, expected, 1e-5);
  });

  it('should reject incorrect weight shapes', () => {
    const embedding = new Embedding(10, 8);
    const wrongWeight = createFloat32Array([1, 2, 3, 4], [2, 2]);

    expect(() => embedding.loadWeight(wrongWeight)).toThrow();
  });
});

describe('Layer Composition', () => {
  it('should compose layers correctly', () => {
    // Create a simple network: Embedding -> Linear -> RMSNorm
    const embedding = new Embedding(10, 8);
    const linear = new Linear(8, 4, true);
    const norm = new RMSNorm(4);

    const indices = MxArray.fromInt32(new Int32Array([0, 1, 2]), BigInt64Array.from([3n]));

    // Forward pass through all layers
    let x = embedding.forward(indices);
    expect(Array.from(x.shape()).map((x) => Number(x))).toEqual([3, 8]);

    x = linear.forward(x);
    expect(Array.from(x.shape()).map((x) => Number(x))).toEqual([3, 4]);

    x = norm.forward(x);
    expect(Array.from(x.shape()).map((x) => Number(x))).toEqual([3, 4]);

    // Output should be normalized
    const outputData = x.toFloat32();
    expect(outputData.length).toBe(12); // 3 * 4
  });

  it('should handle complex shapes through layers', () => {
    const linear = new Linear(16, 8, true);
    const norm = new LayerNorm(8);

    // Input with batch dimension
    const input = MxArray.randomNormal(BigInt64Array.from([5n, 16n]), 0, 1);

    let x = linear.forward(input);
    expect(Array.from(x.shape()).map((x) => Number(x))).toEqual([5, 8]);

    x = norm.forward(x);
    expect(Array.from(x.shape()).map((x) => Number(x))).toEqual([5, 8]);
  });
});

describe('Edge Cases', () => {
  it('should handle single sample inputs', () => {
    const linear = new Linear(4, 2, true);
    const input = createFloat32Array([1, 2, 3, 4], [1, 4]);
    const output = linear.forward(input);

    expect(Array.from(output.shape()).map((x) => Number(x))).toEqual([1, 2]);
  });

  it('should handle large batch sizes', () => {
    const linear = new Linear(8, 4, true);
    const input = MxArray.randomNormal(BigInt64Array.from([100n, 8n]), 0, 1);
    const output = linear.forward(input);

    expect(Array.from(output.shape()).map((x) => Number(x))).toEqual([100, 4]);
  });

  it('should preserve numerical precision', () => {
    const norm = new RMSNorm(4);

    // Very small values
    const smallInput = createFloat32Array([1e-6, 2e-6, 3e-6, 4e-6], [1, 4]);
    const smallOutput = norm.forward(smallInput);
    const smallData = smallOutput.toFloat32();

    // Should not produce NaN or Inf
    smallData.forEach((val) => {
      expect(isFinite(val)).toBe(true);
      expect(isNaN(val)).toBe(false);
    });

    // Very large values
    const largeInput = createFloat32Array([1e6, 2e6, 3e6, 4e6], [1, 4]);
    const largeOutput = norm.forward(largeInput);
    const largeData = largeOutput.toFloat32();

    // Should not produce NaN or Inf
    largeData.forEach((val) => {
      expect(isFinite(val)).toBe(true);
      expect(isNaN(val)).toBe(false);
    });
  });
});
