/**
 * Tests for async reward functions in GRPO training
 *
 * These tests verify that the rewardFunction in GRPOConfig can be async
 * and that the trainer correctly awaits the result.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { GRPOTrainer } from '@mlx-node/trl';
import { createTempModel } from '../test-model-utils.js';

describe('GRPO Async Reward Functions', () => {
  let tempModel: { modelPath: string; cleanup: () => void };

  beforeAll(async () => {
    tempModel = await createTempModel();
  });

  afterAll(() => {
    tempModel?.cleanup();
  });

  describe('Async Reward Function Support', () => {
    it('should work with async reward function', async () => {
      // Async reward function that simulates API call
      const asyncRewardFn = async (
        _prompts: string[],
        completions: string[],
        _answers: (string | null)[],
      ): Promise<Float32Array> => {
        // Simulate async operation (e.g., API call)
        await new Promise((resolve) => setTimeout(resolve, 10));
        return Float32Array.from(completions.map((c) => c.length / 100));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
        rewardFunction: asyncRewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const genResult = await trainer.generateBatch(promptMessages);

      const rewards = await trainer.scoreGenerations(promptMessages, genResult.completionTexts, [null], 2);

      expect(rewards).toBeInstanceOf(Float32Array);
      expect(rewards.length).toBe(2);
    });

    it('should work with synchronous reward function (backward compatibility)', async () => {
      // Synchronous reward function
      const syncRewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]): Float32Array => {
        return Float32Array.from(completions.map((c) => c.length / 100));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
        rewardFunction: syncRewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const genResult = await trainer.generateBatch(promptMessages);

      const rewards = await trainer.scoreGenerations(promptMessages, genResult.completionTexts, [null], 2);

      expect(rewards).toBeInstanceOf(Float32Array);
      expect(rewards.length).toBe(2);
    });

    it('should handle async reward function with parallel processing', async () => {
      // Async reward function that processes in parallel
      const parallelRewardFn = async (
        _prompts: string[],
        completions: string[],
        _answers: (string | null)[],
      ): Promise<Float32Array> => {
        // Simulate parallel processing
        const rewards = await Promise.all(
          completions.map(async (completion) => {
            await new Promise((resolve) => setTimeout(resolve, 5));
            return completion.length / 100;
          }),
        );
        return Float32Array.from(rewards);
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        maxNewTokens: 5,
        rewardFunction: parallelRewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const genResult = await trainer.generateBatch(promptMessages);

      const rewards = await trainer.scoreGenerations(promptMessages, genResult.completionTexts, [null], 4);

      expect(rewards).toBeInstanceOf(Float32Array);
      expect(rewards.length).toBe(4);
    });

    it('should execute training step with async reward function', async () => {
      // Async reward function
      const asyncRewardFn = async (
        _prompts: string[],
        completions: string[],
        _answers: (string | null)[],
      ): Promise<Float32Array> => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        return Float32Array.from(completions.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 3,
        rewardFunction: asyncRewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const answers = [null];

      const metrics = await trainer.trainStep(promptMessages, answers);

      expect(metrics).toBeDefined();
      expect(metrics.loss).toBeTypeOf('number');
      expect(metrics.meanReward).toBeCloseTo(1.0);
      expect(metrics.step).toBe(1);
    });

    it('should handle async reward function that accesses answers', async () => {
      // Async reward function that uses answers
      const answerAwareRewardFn = async (
        _prompts: string[],
        completions: string[],
        answers: (string | null)[],
      ): Promise<Float32Array> => {
        await new Promise((resolve) => setTimeout(resolve, 5));

        const rewards = completions.map((completion, i) => {
          const answer = answers[i];
          if (answer && completion.includes(answer)) {
            return 1.0;
          }
          return 0.5;
        });

        return Float32Array.from(rewards);
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
        rewardFunction: answerAwareRewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const genResult = await trainer.generateBatch(promptMessages);

      const rewards = await trainer.scoreGenerations(
        promptMessages,
        genResult.completionTexts,
        ['answer1', 'answer2'],
        2,
      );

      expect(rewards).toBeInstanceOf(Float32Array);
      expect(rewards.length).toBe(2);
      // All rewards should be either 0.5 or 1.0
      for (const reward of rewards) {
        expect(reward).toBeGreaterThanOrEqual(0.5);
        expect(reward).toBeLessThanOrEqual(1.0);
      }
    });
  });
});
