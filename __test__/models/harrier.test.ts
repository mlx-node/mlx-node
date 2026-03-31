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

      // Rough calculation for tiny config:
      // embedding: 1000 * 128 = 128,000
      // Per layer: attn(4*32*128 + 2*32*128 + 2*32*128 + 128*4*32) + mlp(512*128*3) + norms(128*2) + qk_norms(32*2)
      // = 16384 + 8192 + 8192 + 16384 + 196608 + 256 + 64 = 246,080 per layer
      // final_norm: 128
      // Total: 128000 + 2*246080 + 128 = 620,288
      expect(numParams).toBe(620288);
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
      // Should be hidden_size (128), not vocab_size (1000)
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
});
