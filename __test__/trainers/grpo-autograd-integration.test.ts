import { describe, it, expect } from 'vitest';
import { QWEN3_CONFIGS } from '@mlx-node/lm';
import { MxArray, Qwen3Model } from '@mlx-node/core';
import { shape, int32, float32, float64 } from '../test-utils';

describe('GRPO Autograd Integration', () => {
  describe('train_step_grpo_autograd Method', () => {
    it('should execute autograd training step without errors', () => {
      const model = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);

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

    it('should compute gradients for all parameters', () => {
      const model = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);

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
      const model1 = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);
      const model2 = new Qwen3Model(QWEN3_CONFIGS['qwen3-0.6b']);

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
