import { describe, expect, it } from 'vitest';
import { MxArray, scaledDotProductAttention } from '@mlx-node/core';
import { shape } from '../test-utils';

describe('Scaled Dot-Product Attention', () => {
  it('should compute basic attention without mask', () => {
    const batch = 2;
    const numHeads = 4;
    const seqLen = 8;
    const headDim = 16;

    const queries = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.5);
    const keys = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.5);
    const values = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.5);

    const scale = 1.0 / Math.sqrt(headDim);

    const output = scaledDotProductAttention(queries, keys, values, scale);

    // Output should have same shape as values
    expect(Array.from(output.shape()).map(Number)).toEqual([batch, numHeads, seqLen, headDim]);
  });

  it('should handle single head attention', () => {
    const batch = 1;
    const numHeads = 1;
    const seqLen = 4;
    const headDim = 8;

    const queries = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);
    const keys = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);
    const values = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);

    const scale = 1.0 / Math.sqrt(headDim);

    const output = scaledDotProductAttention(queries, keys, values, scale);

    expect(Array.from(output.shape()).map(Number)).toEqual([batch, numHeads, seqLen, headDim]);
  });

  it('should apply causal mask correctly', () => {
    const batch = 1;
    const numHeads = 1;
    const seqLen = 4;
    const headDim = 8;

    const queries = MxArray.ones(shape(batch, numHeads, seqLen, headDim));
    const keys = MxArray.ones(shape(batch, numHeads, seqLen, headDim));
    const values = MxArray.ones(shape(batch, numHeads, seqLen, headDim));

    // Create causal mask: lower triangular matrix
    // For seqLen=4, mask should be:
    // [[0, -inf, -inf, -inf],
    //  [0,    0, -inf, -inf],
    //  [0,    0,    0, -inf],
    //  [0,    0,    0,    0]]
    const maskData = new Float32Array(seqLen * seqLen);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        maskData[i * seqLen + j] = j > i ? -1e9 : 0;
      }
    }
    const mask = MxArray.fromFloat32(maskData, shape(1, 1, seqLen, seqLen));

    const scale = 1.0 / Math.sqrt(headDim);

    const output = scaledDotProductAttention(queries, keys, values, scale, mask);

    expect(Array.from(output.shape()).map(Number)).toEqual([batch, numHeads, seqLen, headDim]);

    // With causal mask, the attention should be properly masked
    // (we can't easily verify the exact values without manual computation)
  });

  it('should handle different query and key sequence lengths', () => {
    const batch = 2;
    const numHeads = 4;
    const querySeqLen = 4;
    const kvSeqLen = 8;
    const headDim = 16;

    const queries = MxArray.randomNormal(shape(batch, numHeads, querySeqLen, headDim), 0, 1);
    const keys = MxArray.randomNormal(shape(batch, numHeads, kvSeqLen, headDim), 0, 1);
    const values = MxArray.randomNormal(shape(batch, numHeads, kvSeqLen, headDim), 0, 1);

    const scale = 1.0 / Math.sqrt(headDim);

    const output = scaledDotProductAttention(queries, keys, values, scale);

    // Output should have query sequence length
    expect(Array.from(output.shape()).map(Number)).toEqual([batch, numHeads, querySeqLen, headDim]);
  });

  it('should produce normalized output (attention weights sum to 1)', () => {
    // Use simple test case to verify softmax normalization
    const batch = 1;
    const numHeads = 1;
    const seqLen = 3;
    const headDim = 4;

    // Create queries and keys that will produce predictable attention scores
    const queries = MxArray.ones(shape(batch, numHeads, seqLen, headDim));
    const keys = MxArray.ones(shape(batch, numHeads, seqLen, headDim));
    const values = MxArray.ones(shape(batch, numHeads, seqLen, headDim));

    const scale = 1.0;

    const output = scaledDotProductAttention(queries, keys, values, scale);

    // With uniform queries, keys, and values, output should be close to values
    // (since attention weights are uniform)
    const outputMean = output.mean().toFloat32()[0];
    expect(Math.abs(outputMean - 1.0)).toBeLessThan(0.1);
  });

  it('should handle large head dimensions', () => {
    const batch = 2;
    const numHeads = 8;
    const seqLen = 16;
    const headDim = 128;

    const queries = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.02);
    const keys = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.02);
    const values = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.02);

    const scale = 1.0 / Math.sqrt(headDim);

    const output = scaledDotProductAttention(queries, keys, values, scale);

    expect(Array.from(output.shape()).map(Number)).toEqual([batch, numHeads, seqLen, headDim]);
  });

  it('should handle long sequences', () => {
    const batch = 1;
    const numHeads = 4;
    const seqLen = 512;
    const headDim = 64;

    const queries = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.1);
    const keys = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.1);
    const values = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.1);

    const scale = 1.0 / Math.sqrt(headDim);

    const output = scaledDotProductAttention(queries, keys, values, scale);

    expect(Array.from(output.shape()).map(Number)).toEqual([batch, numHeads, seqLen, headDim]);
  });

  it('should be deterministic with same inputs', () => {
    const batch = 1;
    const numHeads = 2;
    const seqLen = 4;
    const headDim = 8;

    const queries = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);
    const keys = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);
    const values = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);

    const scale = 1.0 / Math.sqrt(headDim);

    const output1 = scaledDotProductAttention(queries, keys, values, scale);
    const output2 = scaledDotProductAttention(queries, keys, values, scale);

    // Outputs should be identical
    const diff = output1.sub(output2).abs().sum().toFloat32()[0];
    expect(diff).toBeLessThan(1e-6);
  });

  it('should respect scaling factor', () => {
    const batch = 1;
    const numHeads = 1;
    const seqLen = 4;
    const headDim = 16;

    const queries = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);
    const keys = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);
    const values = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);

    const scale1 = 0.125; // 1/sqrt(64)
    const scale2 = 0.0625; // 1/sqrt(256)

    const output1 = scaledDotProductAttention(queries, keys, values, scale1);
    const output2 = scaledDotProductAttention(queries, keys, values, scale2);

    // Different scales should produce different outputs
    const diff = output1.sub(output2).abs().sum().toFloat32()[0];
    expect(diff).toBeGreaterThan(0.01);
  });

  it('should handle multi-batch attention', () => {
    const batch = 16;
    const numHeads = 8;
    const seqLen = 32;
    const headDim = 64;

    const queries = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.1);
    const keys = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.1);
    const values = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 0.1);

    const scale = 1.0 / Math.sqrt(headDim);

    const output = scaledDotProductAttention(queries, keys, values, scale);

    expect(Array.from(output.shape()).map(Number)).toEqual([batch, numHeads, seqLen, headDim]);
  });

  it('should work with zero values', () => {
    const batch = 1;
    const numHeads = 1;
    const seqLen = 4;
    const headDim = 8;

    const queries = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);
    const keys = MxArray.randomNormal(shape(batch, numHeads, seqLen, headDim), 0, 1);
    const values = MxArray.zeros(shape(batch, numHeads, seqLen, headDim));

    const scale = 1.0 / Math.sqrt(headDim);

    const output = scaledDotProductAttention(queries, keys, values, scale);

    // With zero values, output should also be close to zero
    const outputMean = output.abs().mean().toFloat32()[0];
    expect(outputMean).toBeLessThan(1e-5);
  });

  it('should handle typical transformer dimensions', () => {
    // Test with common transformer configurations
    const configs = [
      { batch: 32, heads: 8, seqLen: 128, headDim: 64 }, // Small model
      { batch: 16, heads: 12, seqLen: 256, headDim: 64 }, // Medium model
      { batch: 8, heads: 16, seqLen: 512, headDim: 64 }, // Large model
    ];

    for (const { batch, heads, seqLen, headDim } of configs) {
      const queries = MxArray.randomNormal(shape(batch, heads, seqLen, headDim), 0, 0.02);
      const keys = MxArray.randomNormal(shape(batch, heads, seqLen, headDim), 0, 0.02);
      const values = MxArray.randomNormal(shape(batch, heads, seqLen, headDim), 0, 0.02);

      const scale = 1.0 / Math.sqrt(headDim);

      const output = scaledDotProductAttention(queries, keys, values, scale);

      expect(Array.from(output.shape()).map(Number)).toEqual([batch, heads, seqLen, headDim]);
    }
  });
});
