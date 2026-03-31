import { HarrierModel, MxArray } from '@mlx-node/core';
import { describe, it, expect } from 'vite-plus/test';

import { shape } from '../test-utils';

// Tiny config for fast unit tests (not real model sizes)
const TINY_CONFIG = {
  vocabSize: 1000,
  hiddenSize: 128,
  numLayers: 2,
  numHeads: 4,
  numKeyValueHeads: 2,
  headDim: 32,
  intermediateSize: 512,
  rmsNormEps: 1e-6,
  ropeTheta: 1_000_000.0,
  maxPositionEmbeddings: 512,
  useQkNorm: true,
};

// Harrier 0.6B config (for config correctness tests)
const HARRIER_0_6B_CONFIG = {
  vocabSize: 151936,
  hiddenSize: 1024,
  numLayers: 28,
  numHeads: 16,
  numKeyValueHeads: 8,
  headDim: 128,
  intermediateSize: 3072,
  rmsNormEps: 1e-6,
  ropeTheta: 1_000_000.0,
  maxPositionEmbeddings: 32768,
  useQkNorm: true,
};

describe.sequential('HarrierModel', () => {
  describe('Model Instantiation', () => {
    it('should create model from tiny config', () => {
      const model = new HarrierModel(TINY_CONFIG);
      expect(model).toBeDefined();
      expect(model.getConfig().hiddenSize).toBe(128);
      expect(model.getConfig().numLayers).toBe(2);
    });

    it('should create model from 0.6B config', () => {
      const model = new HarrierModel(HARRIER_0_6B_CONFIG);
      expect(model).toBeDefined();
      expect(model.getConfig().hiddenSize).toBe(1024);
      expect(model.getConfig().numLayers).toBe(28);
      expect(model.getConfig().numKeyValueHeads).toBe(8);
    });

    it('should report correct number of parameters', () => {
      const model = new HarrierModel(TINY_CONFIG);
      const numParams = model.numParameters();
      expect(numParams).toBeGreaterThan(0);
      expect(numParams).toBe(620288);
    });

    it('should default useQkNorm to true when omitted', () => {
      // useQkNorm is optional — Qwen3 always uses QK normalization
      const { useQkNorm: _, ...configWithoutQkNorm } = TINY_CONFIG;
      const model = new HarrierModel(configWithoutQkNorm as any);
      expect(model.getConfig().useQkNorm).toBe(true);
    });

    it('should accept explicit useQkNorm value', () => {
      const model = new HarrierModel({ ...TINY_CONFIG, useQkNorm: true });
      expect(model.getConfig().useQkNorm).toBe(true);
    });

    it('should return empty prompts map for programmatically created model', () => {
      const model = new HarrierModel(TINY_CONFIG);
      const prompts = model.getPrompts();
      expect(Object.keys(prompts)).toHaveLength(0);
    });
  });

  describe('Forward Pass', () => {
    it('should return hidden states with shape [batch, seq_len, hidden_size]', () => {
      const model = new HarrierModel(TINY_CONFIG);

      const batchSize = 1;
      const seqLen = 5;
      const inputIds = MxArray.randint(shape(batchSize, seqLen), 0, TINY_CONFIG.vocabSize);

      const hidden = model.forward(inputIds);
      expect(hidden).toBeDefined();

      const outputShape = hidden.shape();
      expect(outputShape[0]).toBe(BigInt(batchSize));
      expect(outputShape[1]).toBe(BigInt(seqLen));
      expect(outputShape[2]).toBe(BigInt(TINY_CONFIG.hiddenSize));
    });

    it('should return hidden_size dim, NOT vocab_size dim (not logits)', () => {
      const model = new HarrierModel(TINY_CONFIG);

      const inputIds = MxArray.randint(shape(1, 3), 0, TINY_CONFIG.vocabSize);
      const hidden = model.forward(inputIds);

      const lastDim = hidden.shape()[2];
      expect(lastDim).toBe(BigInt(TINY_CONFIG.hiddenSize));
      expect(lastDim).not.toBe(BigInt(TINY_CONFIG.vocabSize));
    });

    it('should handle different sequence lengths', () => {
      const model = new HarrierModel(TINY_CONFIG);

      for (const seqLen of [1, 10, 50]) {
        const inputIds = MxArray.randint(shape(1, seqLen), 0, TINY_CONFIG.vocabSize);
        const hidden = model.forward(inputIds);
        expect(hidden.shape()[1]).toBe(BigInt(seqLen));
      }
    });
  });

  describe('L2 Normalization (via forward + manual pooling)', () => {
    it('should produce non-zero hidden states suitable for normalization', () => {
      const model = new HarrierModel(TINY_CONFIG);
      const inputIds = MxArray.randint(shape(1, 3), 0, TINY_CONFIG.vocabSize);
      const hidden = model.forward(inputIds);

      // Extract last token hidden state: [1, 1, hidden_size]
      const lastToken = hidden.slice(
        BigInt64Array.from([0n, 2n, 0n]),
        BigInt64Array.from([1n, 3n, BigInt(TINY_CONFIG.hiddenSize)]),
      );

      // Compute L2 norm manually: sqrt(sum(x^2))
      const sq = lastToken.square();
      const sumSq = sq.sum(Int32Array.from([-1]), true);
      const norm = sumSq.sqrt();

      // Norm should be positive (model produces non-trivial output)
      const normVals = norm.toFloat32();
      expect(normVals[0]).toBeGreaterThan(0);
    });
  });

  describe('Tokenizer requirement', () => {
    it('should reject encode() without tokenizer', async () => {
      const model = new HarrierModel(TINY_CONFIG);
      await expect(model.encode('hello')).rejects.toThrow('Tokenizer not loaded');
    });

    it('should reject encodeBatch() without tokenizer', async () => {
      const model = new HarrierModel(TINY_CONFIG);
      await expect(model.encodeBatch(['hello'])).rejects.toThrow('Tokenizer not loaded');
    });
  });
});
