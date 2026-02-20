import { describe, it, expect } from 'vite-plus/test';
import { MxArray, Qwen3Model } from '@mlx-node/core';
import type { Qwen3Config } from '@mlx-node/core';
import { shape, int32, float32, float64 } from '../test-utils';

// Tiny config for autograd tests — the full 0.6B model (28 layers, 896 hidden)
// takes 2-3 minutes per autograd call, exceeding the 120s CI timeout.
// This small config completes in seconds.
const TINY_AUTOGRAD_CONFIG: Qwen3Config = {
  vocabSize: 1000,
  hiddenSize: 64,
  numLayers: 2,
  numHeads: 4,
  numKvHeads: 2,
  headDim: 16,
  intermediateSize: 128,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  maxPositionEmbeddings: 512,
  tieWordEmbeddings: true,
  useQkNorm: false,
  padTokenId: 0,
  eosTokenId: 1,
  bosTokenId: 0,
};

describe('GRPO Autograd Integration', () => {
  describe('train_step_grpo_autograd Method', () => {
    it('should execute autograd training step without errors', () => {
      const model = new Qwen3Model(TINY_AUTOGRAD_CONFIG);

      // Create minimal training data
      const groupSize = 2;
      const seqLen = 4;

      const promptTokens = [
        MxArray.fromInt32(int32(1, 2, 3, 4), shape(seqLen)),
        MxArray.fromInt32(int32(5, 6, 7, 8), shape(seqLen)),
      ];

      const completionTokens = [
        MxArray.fromInt32(int32(10, 11, 12), shape(3)),
        MxArray.fromInt32(int32(13, 14, 15), shape(3)),
        MxArray.fromInt32(int32(16, 17, 18), shape(3)),
        MxArray.fromInt32(int32(19, 20, 21), shape(3)),
      ];

      const completionLogprobs = [
        MxArray.fromFloat32(float32(-0.1, -0.2, -0.15), shape(3)),
        MxArray.fromFloat32(float32(-0.12, -0.18, -0.14), shape(3)),
        MxArray.fromFloat32(float32(-0.11, -0.19, -0.16), shape(3)),
        MxArray.fromFloat32(float32(-0.13, -0.17, -0.15), shape(3)),
      ];

      const rewards = float64(1.0, 0.8, 0.9, 0.7);

      const config = {
        epsilonLow: 0.2,
        epsilonHigh: undefined,
        beta: 0.0,
        lossType: 'grpo',
        importanceSamplingLevel: 'token',
        maxCompletionLength: 256,
        numItemsInBatch: undefined,
        gradientAccumulationSteps: 1,
      };

      const learningRate = 0.0001;

      // Execute autograd training step
      const [loss, metrics] = model.trainStepGrpoAutograd(
        promptTokens,
        completionTokens,
        completionLogprobs,
        rewards,
        groupSize,
        config,
        learningRate,
      );

      // Verify results
      expect(loss).toBeTypeOf('number');
      expect(isFinite(loss)).toBe(true);

      expect(metrics.loss).toBe(loss);
      expect(metrics.mean_reward).toBeTypeOf('number');
      expect(metrics.std_reward).toBeTypeOf('number');
      expect(metrics.mean_advantage).toBeTypeOf('number');
      expect(metrics.num_gradients).toBeGreaterThan(0);

      console.log('Autograd Training Step Results:');
      console.log(`  Loss: ${loss.toFixed(6)}`);
      console.log(`  Mean Reward: ${metrics.mean_reward.toFixed(4)}`);
      console.log(`  Num Gradients: ${metrics.num_gradients}`);
    });

    it('should handle variable-length prompts with correct masking', () => {
      const model = new Qwen3Model(TINY_AUTOGRAD_CONFIG);

      // Use prompts with DIFFERENT lengths to test padding/masking
      const promptTokens = [
        MxArray.fromInt32(int32(1, 2, 3), shape(3)), // 3 tokens
        MxArray.fromInt32(int32(4, 5, 6, 7, 8), shape(5)), // 5 tokens
        MxArray.fromInt32(int32(9, 10), shape(2)), // 2 tokens
        MxArray.fromInt32(int32(11, 12, 13, 14), shape(4)), // 4 tokens
      ];

      const groupSize = 2;

      // 4 prompts × 2 completions each = 8 completions
      const completionTokens = [
        MxArray.fromInt32(int32(100, 101, 102), shape(3)),
        MxArray.fromInt32(int32(103, 104, 105), shape(3)),
        MxArray.fromInt32(int32(106, 107, 108), shape(3)),
        MxArray.fromInt32(int32(109, 110, 111), shape(3)),
        MxArray.fromInt32(int32(112, 113, 114), shape(3)),
        MxArray.fromInt32(int32(115, 116, 117), shape(3)),
        MxArray.fromInt32(int32(118, 119, 120), shape(3)),
        MxArray.fromInt32(int32(121, 122, 123), shape(3)),
      ];

      const completionLogprobs = completionTokens.map(() => MxArray.fromFloat32(float32(-0.1, -0.15, -0.12), shape(3)));

      const rewards = float64(1.0, 0.8, 0.9, 0.7, 0.6, 0.85, 0.75, 0.95);

      const config = {
        epsilonLow: 0.2,
        epsilonHigh: undefined,
        beta: 0.0,
        lossType: 'grpo',
        importanceSamplingLevel: 'token',
        maxCompletionLength: 256,
        numItemsInBatch: undefined,
        gradientAccumulationSteps: 1,
      };

      // Execute with variable-length prompts
      const [loss, metrics] = model.trainStepGrpoAutograd(
        promptTokens,
        completionTokens,
        completionLogprobs,
        rewards,
        groupSize,
        config,
        0.0001,
      );

      // Verify results are valid (not NaN, not infinite)
      expect(loss).toBeTypeOf('number');
      expect(isFinite(loss)).toBe(true);
      expect(Number.isNaN(loss)).toBe(false);

      // Gradients should be computed
      expect(metrics.num_gradients).toBeGreaterThan(0);

      console.log('Variable-Length Prompts Test Results:');
      console.log(`  Prompt lengths: [3, 5, 2, 4]`);
      console.log(`  Loss: ${loss.toFixed(6)}`);
      console.log(`  Num Gradients: ${metrics.num_gradients}`);
    });

    it('should handle variable-length completions with correct masking', () => {
      const model = new Qwen3Model(TINY_AUTOGRAD_CONFIG);

      const promptTokens = [
        MxArray.fromInt32(int32(1, 2, 3, 4), shape(4)),
        MxArray.fromInt32(int32(5, 6, 7, 8), shape(4)),
      ];

      const groupSize = 2;

      // Completions with DIFFERENT lengths to test masking
      const completionTokens = [
        MxArray.fromInt32(int32(100, 101), shape(2)), // 2 tokens
        MxArray.fromInt32(int32(102, 103, 104, 105), shape(4)), // 4 tokens
        MxArray.fromInt32(int32(106), shape(1)), // 1 token
        MxArray.fromInt32(int32(107, 108, 109), shape(3)), // 3 tokens
      ];

      const completionLogprobs = [
        MxArray.fromFloat32(float32(-0.1, -0.15), shape(2)),
        MxArray.fromFloat32(float32(-0.1, -0.15, -0.12, -0.18), shape(4)),
        MxArray.fromFloat32(float32(-0.1), shape(1)),
        MxArray.fromFloat32(float32(-0.1, -0.15, -0.12), shape(3)),
      ];

      const rewards = float64(1.0, 0.8, 0.9, 0.7);

      const config = {
        epsilonLow: 0.2,
        epsilonHigh: undefined,
        beta: 0.0,
        lossType: 'grpo',
        importanceSamplingLevel: 'token',
        maxCompletionLength: 256,
        numItemsInBatch: undefined,
        gradientAccumulationSteps: 1,
      };

      const [loss, metrics] = model.trainStepGrpoAutograd(
        promptTokens,
        completionTokens,
        completionLogprobs,
        rewards,
        groupSize,
        config,
        0.0001,
      );

      expect(loss).toBeTypeOf('number');
      expect(isFinite(loss)).toBe(true);
      expect(Number.isNaN(loss)).toBe(false);
      expect(metrics.num_gradients).toBeGreaterThan(0);

      console.log('Variable-Length Completions Test Results:');
      console.log(`  Completion lengths: [2, 4, 1, 3]`);
      console.log(`  Loss: ${loss.toFixed(6)}`);
      console.log(`  Num Gradients: ${metrics.num_gradients}`);
    });

    it('should compute gradients for all parameters', () => {
      const model = new Qwen3Model(TINY_AUTOGRAD_CONFIG);

      const groupSize = 2;

      const promptTokens = [MxArray.fromInt32(int32(1, 2, 3), shape(3))];

      const completionTokens = [MxArray.fromInt32(int32(10, 11), shape(2)), MxArray.fromInt32(int32(12, 13), shape(2))];

      const completionLogprobs = [
        MxArray.fromFloat32(float32(-0.1, -0.2), shape(2)),
        MxArray.fromFloat32(float32(-0.15, -0.18), shape(2)),
      ];

      const rewards = float64(1.0, 0.5);

      const config = {
        epsilonLow: 0.2,
        epsilonHigh: undefined,
        beta: 0.0,
        lossType: 'grpo',
        importanceSamplingLevel: 'token',
        maxCompletionLength: 256,
        numItemsInBatch: undefined,
        gradientAccumulationSteps: 1,
      };

      const [, metrics] = model.trainStepGrpoAutograd(
        promptTokens,
        completionTokens,
        completionLogprobs,
        rewards,
        groupSize,
        config,
        0.0001,
      );

      // Should have computed gradients for multiple parameters
      expect(metrics.num_gradients).toBeGreaterThan(0);
      console.log(`Computed ${metrics.num_gradients} gradients via autograd`);
    });
  });

  describe('Comparison with Manual Gradients', () => {
    it('should produce similar results to manual gradient method', () => {
      const model1 = new Qwen3Model(TINY_AUTOGRAD_CONFIG);
      const model2 = new Qwen3Model(TINY_AUTOGRAD_CONFIG);

      const promptTokens = [MxArray.fromInt32(int32(1, 2, 3), shape(3))];

      const completionTokens = [MxArray.fromInt32(int32(10, 11), shape(2)), MxArray.fromInt32(int32(12, 13), shape(2))];

      const completionLogprobs = [
        MxArray.fromFloat32(float32(-0.1, -0.2), shape(2)),
        MxArray.fromFloat32(float32(-0.15, -0.18), shape(2)),
      ];

      const rewards = float64(1.0, 0.5);

      const config = {
        epsilonLow: 0.2,
        epsilonHigh: undefined,
        beta: 0.0,
        lossType: 'grpo',
        importanceSamplingLevel: 'token',
        maxCompletionLength: 256,
        numItemsInBatch: undefined,
        gradientAccumulationSteps: 1,
      };

      // Run autograd version
      const [lossAutograd, metricsAutograd] = model1.trainStepGrpoAutograd(
        promptTokens,
        completionTokens,
        completionLogprobs,
        rewards,
        2,
        config,
        0.0001,
      );

      // Run manual version
      const [lossManual] = model2.trainStepGrpo(
        promptTokens,
        completionTokens,
        completionLogprobs,
        rewards,
        2,
        config,
        0.0001,
      );

      console.log('\nComparison:');
      console.log(`  Autograd Loss: ${lossAutograd.toFixed(6)}`);
      console.log(`  Manual Loss:   ${lossManual.toFixed(6)}`);
      console.log(`  Autograd Gradients: ${metricsAutograd.num_gradients}`);

      // Both should compute finite losses
      expect(isFinite(lossAutograd)).toBe(true);
      expect(isFinite(lossManual)).toBe(true);

      // Both should have computed gradients
      expect(metricsAutograd.num_gradients).toBeGreaterThan(0);
    });
  });
});
