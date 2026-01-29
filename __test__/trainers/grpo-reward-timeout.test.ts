/**
 * Tests for reward function timeout in GRPO training
 *
 * These tests verify that the reward function timeout mechanism works correctly,
 * preventing training hangs when reward functions take too long.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vite-plus/test';
import { GRPOTrainer, RewardTimeoutError, type RewardOutput } from '@mlx-node/trl';
import { createTempModel } from '../test-model-utils.js';

describe('GRPO Reward Function Timeout', () => {
  let tempModel: { modelPath: string; cleanup: () => void };

  beforeAll(async () => {
    tempModel = await createTempModel();
  });

  afterAll(() => {
    tempModel?.cleanup();
  });

  describe('RewardTimeoutError', () => {
    it('should have correct properties', () => {
      const error = new RewardTimeoutError('Test timeout', 5000);
      expect(error.name).toBe('RewardTimeoutError');
      expect(error.message).toBe('Test timeout');
      expect(error.timeoutMs).toBe(5000);
      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(RewardTimeoutError);
    });
  });

  describe('scoreGenerations timeout', () => {
    it('should timeout when reward function takes too long', async () => {
      // Reward function that hangs longer than timeout
      const slowRewardFn = async (outputs: RewardOutput[]): Promise<Float32Array> => {
        // Wait longer than the timeout
        await new Promise((resolve) => setTimeout(resolve, 2000));
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
        rewardFunction: slowRewardFn,
        rewardTimeout: 100, // 100ms timeout
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const genResult = await trainer.generateBatch(promptMessages);

      // Should throw RewardTimeoutError
      await expect(trainer.scoreGenerations(promptMessages, genResult.completionTexts, [null], 2)).rejects.toThrow(
        RewardTimeoutError,
      );

      // Verify error message contains timeout info
      await expect(trainer.scoreGenerations(promptMessages, genResult.completionTexts, [null], 2)).rejects.toThrow(
        /timed out after 100ms/,
      );
    });

    it('should work normally when reward function completes within timeout', async () => {
      // Fast reward function
      const fastRewardFn = async (outputs: RewardOutput[]): Promise<Float32Array> => {
        await new Promise((resolve) => setTimeout(resolve, 10));
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
        rewardFunction: fastRewardFn,
        rewardTimeout: 5000, // 5 second timeout
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const genResult = await trainer.generateBatch(promptMessages);

      const rewards = await trainer.scoreGenerations(promptMessages, genResult.completionTexts, [null], 2);

      expect(rewards).toBeInstanceOf(Float32Array);
      expect(rewards.length).toBe(2);
    });

    it('should allow disabling timeout with rewardTimeout=0', async () => {
      // Reward function with delay (but not too long for test)
      const delayedRewardFn = async (outputs: RewardOutput[]): Promise<Float32Array> => {
        await new Promise((resolve) => setTimeout(resolve, 50));
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
        rewardFunction: delayedRewardFn,
        rewardTimeout: 0, // Disable timeout
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const genResult = await trainer.generateBatch(promptMessages);

      // Should complete without timeout
      const rewards = await trainer.scoreGenerations(promptMessages, genResult.completionTexts, [null], 2);

      expect(rewards).toBeInstanceOf(Float32Array);
      expect(rewards.length).toBe(2);
    });
  });

  describe('trainStep timeout', () => {
    it('should timeout when reward function takes too long during training', async () => {
      // Reward function that hangs longer than timeout
      const slowRewardFn = async (outputs: RewardOutput[]): Promise<Float32Array> => {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 3,
        rewardFunction: slowRewardFn,
        rewardTimeout: 100, // 100ms timeout
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      // Should throw error containing timeout message
      // Note: When going through Rust FFI, the error gets wrapped in a GenericFailure
      // but the original timeout message is preserved
      await expect(trainer.trainStep(promptMessages)).rejects.toThrow(/timed out after 100ms/);
    });

    it('should complete training step when reward function finishes in time', async () => {
      // Fast reward function
      const fastRewardFn = async (outputs: RewardOutput[]): Promise<Float32Array> => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 3,
        rewardFunction: fastRewardFn,
        rewardTimeout: 5000, // 5 second timeout
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const metrics = await trainer.trainStep(promptMessages);

      expect(metrics).toBeDefined();
      expect(metrics.loss).toBeTypeOf('number');
      expect(metrics.meanReward).toBeCloseTo(1.0);
    });
  });

  describe('default timeout behavior', () => {
    it('should use default 60 second timeout when not specified', async () => {
      // Reward function that completes quickly
      const fastRewardFn = async (outputs: RewardOutput[]): Promise<Float32Array> => {
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
        rewardFunction: fastRewardFn,
        // No rewardTimeout specified - should use default of 60000ms
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const genResult = await trainer.generateBatch(promptMessages);

      // Should work fine with default timeout
      const rewards = await trainer.scoreGenerations(promptMessages, genResult.completionTexts, [null], 2);

      expect(rewards).toBeInstanceOf(Float32Array);
      expect(rewards.length).toBe(2);
    });
  });

  describe('synchronous reward function timeout', () => {
    it('should handle sync reward functions that complete immediately', async () => {
      // Synchronous reward function (no await)
      const syncRewardFn = (outputs: RewardOutput[]): Float32Array => {
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
        rewardFunction: syncRewardFn,
        rewardTimeout: 100, // 100ms timeout
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const genResult = await trainer.generateBatch(promptMessages);

      // Should work fine - sync functions resolve immediately
      const rewards = await trainer.scoreGenerations(promptMessages, genResult.completionTexts, [null], 2);

      expect(rewards).toBeInstanceOf(Float32Array);
      expect(rewards.length).toBe(2);
    });
  });
});
