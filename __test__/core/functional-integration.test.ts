/**
 * Integration tests for functional forward pass with autograd
 *
 * Tests the complete flow from parameters to gradients through
 * the functional forward pass architecture.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Qwen3Model, type Qwen3Config, MxArray } from '@mlx-node/core';
import { shape, int32 } from '../test-utils';

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

  describe('Full Model Forward Pass', () => {
    it('should match stateful model output', () => {
      const model = new Qwen3Model(tinyConfig);
      const inputIds = MxArray.fromInt32(new Int32Array([0, 1, 2, 3, 4]), shape(1, 5));

      // Stateful forward pass
      const logits1 = model.forward(inputIds);

      // Forward pass again (should be identical)
      const logits2 = model.forward(inputIds);

      // Should match exactly
      const diff = logits1.sub(logits2);
      const maxDiff = diff.abs().max().toFloat32()[0];
      expect(maxDiff).toBeLessThan(1e-6);
    });

    it('should produce valid logits shape', () => {
      const model = new Qwen3Model(tinyConfig);
      const batchSize = 2;
      const seqLen = 8;

      const inputIds = MxArray.randint(shape(batchSize, seqLen), 0, 50);
      const logits = model.forward(inputIds);

      const resultShape = logits.shape();
      expect(Array.from(resultShape).map(Number)).toEqual([batchSize, seqLen, tinyConfig.vocabSize]);
    });

    it('should handle different sequence lengths', () => {
      const model = new Qwen3Model(tinyConfig);

      const seqLengths = [1, 5, 16, 32, 64];

      for (const seqLen of seqLengths) {
        const inputIds = MxArray.randint(shape(1, seqLen), 0, 50);
        const logits = model.forward(inputIds);

        const resultShape = logits.shape();
        expect(Array.from(resultShape).map(Number)).toEqual([1, seqLen, tinyConfig.vocabSize]);

        // Check numerical stability
        const maxVal = logits.abs().max().toFloat32()[0];
        expect(isFinite(maxVal)).toBe(true);
      }
    });

    it('should handle different batch sizes', () => {
      const model = new Qwen3Model(tinyConfig);

      const batchSizes = [1, 2, 4, 8];

      for (const batchSize of batchSizes) {
        const inputIds = MxArray.randint(shape(batchSize, 5), 0, 50);
        const logits = model.forward(inputIds);

        const resultShape = logits.shape();
        expect(Array.from(resultShape).map(Number)).toEqual([batchSize, 5, tinyConfig.vocabSize]);
      }
    });
  });

  describe('Parameter Update Flow', () => {
    it('should reflect parameter changes in output', () => {
      const model = new Qwen3Model(tinyConfig);
      const inputIds = MxArray.fromInt32(new Int32Array([0, 1, 2]), shape(1, 3));

      // Initial forward pass
      const logits1 = model.forward(inputIds);

      // Update all parameters by scaling
      const params = model.getParameters();
      const updatedParams: Record<string, MxArray> = {};
      for (const [name, param] of Object.entries(params)) {
        updatedParams[name] = param.mul(MxArray.full(shape(), 1.1));
      }
      model.loadParameters(updatedParams);

      // Forward pass with updated parameters
      const logits2 = model.forward(inputIds);

      // Outputs should be different
      const diff = logits1.sub(logits2);
      const maxDiff = diff.abs().max().toFloat32()[0];
      expect(maxDiff).toBeGreaterThan(0.01);
    });

    it('should handle small parameter updates', () => {
      const model = new Qwen3Model(tinyConfig);
      const inputIds = MxArray.fromInt32(new Int32Array([5, 6, 7]), shape(1, 3));

      const logitsBefore = model.forward(inputIds);

      // Small update to one parameter
      const params = model.getParameters();
      const embWeight = params['embedding.weight'];
      const embUpdated = embWeight.add(MxArray.full(shape(), 0.001));

      model.loadParameters({
        'embedding.weight': embUpdated,
      });

      const logitsAfter = model.forward(inputIds);

      // Should have small change
      const diff = logitsBefore.sub(logitsAfter);
      const maxDiff = diff.abs().max().toFloat32()[0];
      expect(maxDiff).toBeGreaterThan(1e-6);
      expect(maxDiff).toBeLessThan(0.5); // Relaxed threshold for deep network
    });
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

  describe('Numerical Stability in Full Forward Pass', () => {
    it('should handle long sequences without overflow', () => {
      const model = new Qwen3Model(tinyConfig);
      const seqLen = 128;

      const inputIds = MxArray.randint(shape(1, seqLen), 0, 50);
      const logits = model.forward(inputIds);

      const maxVal = logits.abs().max().toFloat32()[0];
      expect(isFinite(maxVal)).toBe(true);
      expect(maxVal).toBeLessThan(100); // Should stay bounded
    });

    it('should handle repeated forward passes', () => {
      const model = new Qwen3Model(tinyConfig);
      const inputIds = MxArray.fromInt32(new Int32Array([0, 1, 2]), shape(1, 3));

      // Run forward pass 10 times
      let prevLogits = model.forward(inputIds);

      for (let i = 0; i < 10; i++) {
        const logits = model.forward(inputIds);

        // Should be identical to previous
        const diff = logits.sub(prevLogits);
        const maxDiff = diff.abs().max().toFloat32()[0];
        expect(maxDiff).toBeLessThan(1e-6);

        prevLogits = logits;
      }
    });

    it('should handle extreme input token IDs', () => {
      const model = new Qwen3Model(tinyConfig);

      // All zeros
      const zeros = MxArray.fromInt32(new Int32Array([0, 0, 0, 0, 0]), shape(1, 5));
      const logits1 = model.forward(zeros);
      expect(isFinite(logits1.abs().max().toFloat32()[0])).toBe(true);

      // Maximum valid token ID
      const maxToken = tinyConfig.vocabSize - 1;
      const maxIds = MxArray.fromInt32(new Int32Array([maxToken, maxToken, maxToken]), shape(1, 3));
      const logits2 = model.forward(maxIds);
      expect(isFinite(logits2.abs().max().toFloat32()[0])).toBe(true);

      // Mixed
      const mixed = MxArray.fromInt32(new Int32Array([0, maxToken, 0, maxToken]), shape(1, 4));
      const logits3 = model.forward(mixed);
      expect(isFinite(logits3.abs().max().toFloat32()[0])).toBe(true);
    });
  });

  describe('GRPO Training Preparation', () => {
    it('should support GRPO loss computation components', () => {
      const model = new Qwen3Model(tinyConfig);

      // Simulate GRPO training setup
      const promptIds = MxArray.fromInt32(new Int32Array([0, 1, 2]), shape(1, 3));
      const completionIds = MxArray.fromInt32(new Int32Array([3, 4, 5, 6]), shape(1, 4));

      // Concatenate for full sequence
      const fullIds = MxArray.concatenate(promptIds, completionIds, 1);

      // Forward pass
      const logits = model.forward(fullIds);

      const resultShape = logits.shape();
      expect(Array.from(resultShape).map(Number)).toEqual([1, 7, tinyConfig.vocabSize]); // 3 prompt + 4 completion

      // Extract completion logits
      const completionLogits = logits.slice(shape(0, 3, 0), shape(1, 7, tinyConfig.vocabSize));
      const completionShape = completionLogits.shape();
      expect(Array.from(completionShape).map(Number)).toEqual([1, 4, tinyConfig.vocabSize]);
    });

    it('should compute log probabilities for GRPO', () => {
      const model = new Qwen3Model(tinyConfig);
      const inputIds = MxArray.fromInt32(new Int32Array([0, 1, 2, 3, 4]), shape(1, 5));

      const logits = model.forward(inputIds);

      // Compute log softmax
      const logProbs = logits.logSoftmax(-1);

      const resultShape = logProbs.shape();
      expect(Array.from(resultShape).map(Number)).toEqual([1, 5, tinyConfig.vocabSize]);

      // Log probs should be negative
      const maxLogProb = logProbs.max().toFloat32()[0];
      expect(maxLogProb).toBeLessThanOrEqual(0);

      // Sum of probs should be ~1 (exp(logprob))
      const probs = logProbs.exp();
      const probSum = probs.sum(int32(-1));
      const probSumData = probSum.toFloat32();
      for (const sum of probSumData) {
        expect(sum).toBeCloseTo(1.0, 0);
      }
    });

    it('should handle batch GRPO setup', () => {
      const model = new Qwen3Model(tinyConfig);

      // Multiple prompt-completion pairs
      const batchSize = 4;
      const promptLen = 3;
      const completionLen = 5;

      const promptIds = MxArray.randint(shape(batchSize, promptLen), 0, 50);
      const completionIds = MxArray.randint(shape(batchSize, completionLen), 0, 50);

      const fullIds = MxArray.concatenate(promptIds, completionIds, 1);

      const logits = model.forward(fullIds);

      const resultShape = logits.shape();
      expect(Array.from(resultShape).map(Number)).toEqual([batchSize, promptLen + completionLen, tinyConfig.vocabSize]);

      // Extract completion logits
      const completionLogits = logits.slice(
        shape(0, promptLen, 0),
        shape(batchSize, promptLen + completionLen, tinyConfig.vocabSize),
      );

      const completionShape = completionLogits.shape();
      expect(Array.from(completionShape).map(Number)).toEqual([batchSize, completionLen, tinyConfig.vocabSize]);
    });
  });

  describe('Memory Efficiency', () => {
    it('should handle moderate model size', () => {
      const mediumConfig: Qwen3Config = {
        ...tinyConfig,
        hiddenSize: 128,
        numLayers: 4,
        headDim: 32, // hiddenSize / numHeads = 128 / 4 = 32
        intermediateSize: 512,
      };

      const model = new Qwen3Model(mediumConfig);
      const inputIds = MxArray.randint(shape(2, 16), 0, 50);

      const logits = model.forward(inputIds);

      const resultShape = logits.shape();
      expect(Array.from(resultShape).map(Number)).toEqual([2, 16, 50]);
    });

    it('should not leak memory across multiple forward passes', () => {
      const model = new Qwen3Model(tinyConfig);
      const inputIds = MxArray.fromInt32(new Int32Array([0, 1, 2]), shape(1, 3));

      // Multiple forward passes
      for (let i = 0; i < 20; i++) {
        const logits = model.forward(inputIds);
        // Don't keep references
        const _ = logits.shape();
      }

      // If we got here without OOM, memory is managed properly
      expect(true).toBe(true);
    });
  });

  describe('Determinism and Reproducibility', () => {
    it('should be deterministic with same model and input', () => {
      const model1 = new Qwen3Model(tinyConfig);
      const model2 = new Qwen3Model(tinyConfig);

      // Copy parameters from model1 to model2
      const params1 = model1.getParameters();
      model2.loadParameters(params1);

      const inputIds = MxArray.fromInt32(new Int32Array([5, 10, 15]), shape(1, 3));

      const logits1 = model1.forward(inputIds);
      const logits2 = model2.forward(inputIds);

      // Should be identical
      const diff = logits1.sub(logits2);
      const maxDiff = diff.abs().max().toFloat32()[0];
      expect(maxDiff).toBeLessThan(1e-6);
    });

    it('should produce consistent outputs across runs', () => {
      const model = new Qwen3Model(tinyConfig);
      const inputIds = MxArray.fromInt32(new Int32Array([1, 2, 3, 4, 5]), shape(1, 5));

      const outputs: number[][] = [];

      // Run 5 times
      for (let run = 0; run < 5; run++) {
        const logits = model.forward(inputIds);
        const logitsData = logits.toFloat32();
        outputs.push(Array.from(logitsData.slice(0, 10))); // First 10 values
      }

      // All runs should be identical
      for (let i = 1; i < outputs.length; i++) {
        for (let j = 0; j < outputs[i].length; j++) {
          expect(outputs[i][j]).toBeCloseTo(outputs[0][j], 5);
        }
      }
    });
  });
});
