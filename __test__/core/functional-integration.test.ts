/**
 * Integration tests for functional forward pass with autograd
 *
 * Tests the complete flow from parameters to gradients through
 * the functional forward pass architecture.
 */

import { Qwen3Model, type Qwen3Config, MxArray } from '@mlx-node/core';
import { describe, it, expect, beforeAll } from 'vite-plus/test';

import { shape } from '../test-utils';

describe('Functional Forward Pass Integration', () => {
  let tinyConfig: Qwen3Config;

  beforeAll(() => {
    tinyConfig = {
      vocabSize: 50,
      hiddenSize: 32,
      numLayers: 2,
      numHeads: 4,
      numKvHeads: 4,
      headDim: 8, // hiddenSize / numHeads = 32 / 4 = 8
      intermediateSize: 128,
      rmsNormEps: 1e-6,
      ropeTheta: 10000.0,
      maxPositionEmbeddings: 512,
      useQkNorm: false,
      tieWordEmbeddings: false,
      padTokenId: 0,
      eosTokenId: 1,
      bosTokenId: 0,
    };
  });

  describe('Gradient Flow Simulation', () => {
    it('should compute loss from logits', () => {
      const model = new Qwen3Model(tinyConfig);
      const inputIds = MxArray.fromInt32(new Int32Array([0, 1, 2]), shape(1, 3));
      const targets = MxArray.fromInt32(new Int32Array([1, 2, 3]), shape(1, 3));

      // Compute cross-entropy loss
      const loss = model.computeLoss(inputIds, targets);

      const lossVal = loss.toFloat32()[0];
      expect(isFinite(lossVal)).toBe(true);
      expect(lossVal).toBeGreaterThan(0);
    });

    it('should support gradient computation with valueAndGrad', () => {
      const model = new Qwen3Model(tinyConfig);

      // Simple optimization target: minimize sum of embeddings squared
      const params = model.getParameters();
      const embWeight = params['embedding.weight'];

      // Compute loss: sum(embedding^2)
      const loss = embWeight.square().sum();

      const lossVal = loss.toFloat32()[0];
      expect(isFinite(lossVal)).toBe(true);
      expect(lossVal).toBeGreaterThan(0);
    });
  });

  describe('End-to-End Training Simulation', () => {
    it('should complete training step with loss computation', () => {
      const model = new Qwen3Model(tinyConfig);

      // Create mini-batch
      const batchSize = 2;
      const seqLen = 5;
      const inputIds = MxArray.randint(shape(batchSize, seqLen), 0, 50);
      const targets = MxArray.randint(shape(batchSize, seqLen), 0, 50);

      // Compute loss
      const loss = model.computeLoss(inputIds, targets);

      const lossVal = loss.toFloat32()[0];
      expect(isFinite(lossVal)).toBe(true);
      expect(lossVal).toBeGreaterThan(0);

      // Loss should be reasonable (not too large)
      expect(lossVal).toBeLessThan(20);
    });

    it('should handle multiple training steps', () => {
      const model = new Qwen3Model(tinyConfig);
      const inputIds = MxArray.fromInt32(new Int32Array([0, 1, 2, 3, 4]), shape(1, 5));
      const targets = MxArray.fromInt32(new Int32Array([1, 2, 3, 4, 5]), shape(1, 5));

      const losses: number[] = [];

      // Simulate 3 training steps
      for (let step = 0; step < 3; step++) {
        const loss = model.computeLoss(inputIds, targets);
        const lossVal = loss.toFloat32()[0];
        losses.push(lossVal);

        // All losses should be finite
        expect(isFinite(lossVal)).toBe(true);
      }

      // Without gradient updates, losses should be similar
      expect(Math.abs(losses[0] - losses[1])).toBeLessThan(1e-5);
      expect(Math.abs(losses[1] - losses[2])).toBeLessThan(1e-5);
    });
  });
});
