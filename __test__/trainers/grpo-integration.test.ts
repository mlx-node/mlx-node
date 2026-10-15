/**
 * Integration Tests for GRPO Training Pipeline
 *
 * These tests validate the entire training workflow end-to-end
 */

import { existsSync, rmSync } from 'node:fs';

import { GRPOTrainer, type RewardOutput } from '@mlx-node/trl';
import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from 'vite-plus/test';

import { createTempModel } from '../test-model-utils';

const TEST_OUTPUT_DIR = './test-integration-output';

// Shared temp model for all tests
let tempModel: { modelPath: string; cleanup: () => void };

describe('GRPO Integration Tests', () => {
  beforeAll(async () => {
    tempModel = await createTempModel();
  });

  afterAll(() => {
    tempModel?.cleanup();
  });

  beforeEach(() => {
    // Clean up test output before each test
    if (existsSync(TEST_OUTPUT_DIR)) {
      rmSync(TEST_OUTPUT_DIR, { recursive: true, force: true });
    }
  });

  afterEach(() => {
    // Clean up test output after each test
    if (existsSync(TEST_OUTPUT_DIR)) {
      rmSync(TEST_OUTPUT_DIR, { recursive: true, force: true });
    }
  });

  // `train()` loop coverage (epochs × batches, partial batch, batch > dataset,
  // empty dataset) lives in `grpo-trainer.test.ts` → 'GRPOTrainer - train()',
  // which asserts the exact reward-call sizes instead of "did not throw".

  describe('Generation and Scoring Pipeline', () => {
    it('should generate, score, and compute loss correctly', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 5,
        temperature: 0.8,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (outputs: RewardOutput[]) => {
          // Simple length-based reward
          return Float32Array.from(outputs.map((o) => o.completion.rawText.length / 10));
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      // Generate completions (groupSize=3 from config)
      const genResult = await trainer.generateBatch(promptMessages);
      expect(genResult.completionTexts.length).toBe(3);

      // Score completions
      const rewards = await trainer.scoreGenerations(promptMessages, genResult.completionTexts, [], 3);
      expect(rewards.length).toBe(3);

      // Execute training step
      const metrics = await trainer.trainStep(promptMessages);
      expect(metrics.loss).toBeDefined();
      expect(isFinite(metrics.loss)).toBe(true);
      expect(metrics.meanReward).toBeGreaterThan(0);
    });

    it('should maintain consistency across multiple steps', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 4,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (outputs: RewardOutput[]) => {
          return Float32Array.from(outputs.map(() => 1.0));
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      // Run 5 training steps
      const metrics: any[] = [];
      for (let i = 0; i < 5; i++) {
        const m = await trainer.trainStep(promptMessages);
        metrics.push(m);
      }

      // Steps should increment
      expect(metrics[0].step).toBe(1);
      expect(metrics[4].step).toBe(5);

      // Most steps should have valid losses
      // Note: Occasional NaN can occur due to GPU numerical precision issues,
      // which the engine handles gracefully by skipping gradient updates
      let finiteCount = 0;
      for (const m of metrics) {
        if (isFinite(m.loss)) finiteCount++;
        expect(m.totalTokens).toBeGreaterThan(0);
      }
      // At least 4 out of 5 should be finite
      expect(finiteCount).toBeGreaterThanOrEqual(4);
    });
  });

  // Loss math per type and clip-epsilon handling: `crates/mlx-core/src/grpo/loss.rs` →
  // `test_grpo_loss_basic`, `test_bnpo_loss_computation`, `test_clip_epsilon_symmetric`,
  // `test_clip_epsilon_asymmetric`. Only the `lossType` plumbing is exercised here.
  describe('Loss Computation Variants', () => {
    it('should work with different loss types', async () => {
      const lossTypes: Array<'grpo' | 'bnpo'> = ['grpo', 'bnpo'];

      for (const lossType of lossTypes) {
        const trainer = await GRPOTrainer.create({
          modelPath: tempModel.modelPath,
          modelName: 'qwen3-0.6b',
          groupSize: 3,
          maxCompletionLength: 5,
          lossType,
          logInterval: 1000,
          saveInterval: 1000,
          outputDir: TEST_OUTPUT_DIR,
          logConsole: false,
          rewardFunction: (outputs: RewardOutput[]) => {
            return Float32Array.from(outputs.map((_, i) => i + 1.0));
          },
        });

        const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

        const metrics = await trainer.trainStep(promptMessages);

        expect(isFinite(metrics.loss)).toBe(true);
        expect(metrics.step).toBe(1);
      }
    });
  });

  // Group-normalised advantages (zero mean, uniform → 0) and reward mean/std:
  // `crates/mlx-core/src/grpo/advantages.rs` → `test_compute_advantages_basic`,
  // `test_uniform_rewards_zero_advantage`; `crates/mlx-core/src/grpo/engine.rs` →
  // `test_compute_reward_stats`. Metric plumbing is checked in 'Metrics Tracking' below
  // and in `grpo-trainer.test.ts` → 'Integration'.

  describe('Sampling Configuration', () => {
    it('should respect temperature settings', async () => {
      const temperatures = [0.5, 1.0, 1.5];

      for (const temp of temperatures) {
        const trainer = await GRPOTrainer.create({
          modelPath: tempModel.modelPath,
          modelName: 'qwen3-0.6b',
          groupSize: 3,
          maxCompletionLength: 6,
          temperature: temp,
          logInterval: 1000,
          saveInterval: 1000,
          outputDir: TEST_OUTPUT_DIR,
          logConsole: false,
          rewardFunction: (outputs: RewardOutput[]) => {
            return Float32Array.from(outputs.map(() => 1.0));
          },
        });

        const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

        const result = await trainer.generateBatch(promptMessages);

        expect(result.completionTexts.length).toBe(3);
        expect(result.nativeResult.completionLengths.length).toBe(3);
      }
    });

    it('should apply top-p and top-k filtering', async () => {
      const configs = [
        { topP: 0.9, topK: undefined },
        { topP: undefined, topK: 50 },
        { topP: 0.95, topK: 100 },
      ];

      for (const config of configs) {
        const trainer = await GRPOTrainer.create({
          modelPath: tempModel.modelPath,
          modelName: 'qwen3-0.6b',
          groupSize: 2,
          maxCompletionLength: 5,
          temperature: 0.8,
          ...config,
          logInterval: 1000,
          saveInterval: 1000,
          outputDir: TEST_OUTPUT_DIR,
          logConsole: false,
          rewardFunction: (outputs: RewardOutput[]) => {
            return Float32Array.from(outputs.map(() => 1.0));
          },
        });

        const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

        const result = await trainer.generateBatch(promptMessages);

        expect(result.completionTexts.length).toBe(2);
      }
    }, 30000); // 30s timeout - creates 3 trainers in sequence
  });

  describe('Error Handling', () => {
    it('should throw error when reward function returns wrong count', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 5,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (_outputs: RewardOutput[]) => {
          // Return wrong number of rewards
          return Float32Array.from([1.0, 2.0]); // Should be 3
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      // Generate completions (groupSize=3 from config)
      const genResult = await trainer.generateBatch(promptMessages);

      await expect(trainer.scoreGenerations(promptMessages, genResult.completionTexts, [], 3)).rejects.toThrow(
        /Reward function returned 2 rewards.*expected 3/,
      );
    });

    it('should throw error when no reward function configured', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        // No reward function
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const completions = ['completion1', 'completion2'];

      await expect(trainer.scoreGenerations(promptMessages, completions, [], 2)).rejects.toThrow(
        /No reward function configured/,
      );
    });
  });

  describe('Metrics Tracking', () => {
    it('should track all metrics correctly', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 4,
        maxCompletionLength: 6,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (outputs: RewardOutput[]) => {
          return Float32Array.from(outputs.map((_, i) => i + 1.0));
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages);

      // Validate all metrics are present and valid
      expect(metrics.step).toBe(1);
      expect(metrics.loss).toBeDefined();
      expect(isFinite(metrics.loss)).toBe(true);
      // meanReward may differ from the raw mean (2.5) because the engine
      // filters out degenerate completions (those hitting ≥90% of maxCompletionLength).
      // After filtering, only surviving completions' rewards are averaged, making
      // the exact value nondeterministic. Assert it's a valid number within the
      // reward range [1, 4] instead.
      expect(metrics.meanReward).toBeGreaterThanOrEqual(1);
      expect(metrics.meanReward).toBeLessThanOrEqual(4);
      // stdReward can be 0 when only 1 completion survives degenerate filtering
      expect(metrics.stdReward).toBeGreaterThanOrEqual(0);
      expect(metrics.meanAdvantage).toBeDefined();
      expect(metrics.totalTokens).toBeGreaterThan(0);
    });

    it('should track token counts accurately', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 10,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (outputs: RewardOutput[]) => {
          return Float32Array.from(outputs.map(() => 1.0));
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages);

      // Should have generated some tokens (3 completions × up to 10 tokens)
      expect(metrics.totalTokens).toBeGreaterThan(0);
      expect(metrics.totalTokens).toBeLessThanOrEqual(30);
    });
  });
});
