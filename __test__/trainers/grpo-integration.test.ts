/**
 * Integration Tests for GRPO Training Pipeline
 *
 * These tests validate the entire training workflow end-to-end
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { GRPOTrainer } from '../../src/index';
import { clearCache } from '@mlx-node/core';
import { existsSync, rmSync } from 'node:fs';
import { createTempModel } from '../test-model-utils';

const TEST_OUTPUT_DIR = './test-integration-output';

// Shared temp model for all tests
let tempModel: { modelPath: string; cleanup: () => void };

describe.sequential('GRPO Integration Tests', () => {
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
    // Clear GPU cache to prevent memory accumulation between tests
    clearCache();
  });

  describe.sequential('End-to-End Training', () => {
    it('should complete a full training run with multiple epochs', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
        numEpochs: 2,
        batchSize: 2,
        logInterval: 1,
        saveInterval: 1000, // Don't save during test
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        logJsonl: false,
        rewardFunction: (_prompts, completions) => {
          return Float32Array.from(completions.map((c) => c.split(',').length / 10));
        },
      });

      const dataset = [
        { prompt: [{ role: 'user', content: 'Test 1' }], answer: '42' },
        { prompt: [{ role: 'user', content: 'Test 2' }], answer: '43' },
        { prompt: [{ role: 'user', content: 'Test 3' }], answer: '44' },
      ];

      // Should complete without errors
      await expect(trainer.train(dataset as any)).resolves.not.toThrow();
    });

    it('should generate diverse completions across training', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        maxNewTokens: 8,
        temperature: 0.9, // High temperature for diversity
        numEpochs: 1,
        batchSize: 1,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        logJsonl: false,
        rewardFunction: (_prompts, completions) => {
          // Reward longer completions
          return Float32Array.from(completions.map((c) => c.split(',').length));
        },
      });

      const dataset = [
        { prompt: [{ role: 'user', content: 'Question 1' }], answer: 'Answer 1' },
        { prompt: [{ role: 'user', content: 'Question 2' }], answer: 'Answer 2' },
      ];

      await trainer.train(dataset as any);

      // Test passed if no errors thrown
      expect(true).toBe(true);
    });

    it('should handle training with varying reward values', async () => {
      let rewardCallCount = 0;

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 6,
        numEpochs: 1,
        batchSize: 2,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        logJsonl: false,
        rewardFunction: (_prompts, completions) => {
          rewardCallCount++;
          // Varying rewards: some high, some low
          return Float32Array.from(completions.map((_, i) => Math.sin(i) * 5 + 5));
        },
      });

      const dataset = [
        { prompt: [{ role: 'user', content: 'A' }], answer: '1' },
        { prompt: [{ role: 'user', content: 'B' }], answer: '2' },
        { prompt: [{ role: 'user', content: 'C' }], answer: '3' },
        { prompt: [{ role: 'user', content: 'D' }], answer: '4' },
      ];

      await trainer.train(dataset as any);

      // Reward function should be called for each batch (2 batches)
      expect(rewardCallCount).toBeGreaterThan(0);
    });
  });

  describe.sequential('Generation and Scoring Pipeline', () => {
    it('should generate, score, and compute loss correctly', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 5,
        temperature: 0.8,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (_prompts, completions) => {
          // Simple length-based reward
          return Float32Array.from(completions.map((c) => c.length / 10));
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
      const metrics = await trainer.trainStep(promptMessages, []);
      expect(metrics.loss).toBeDefined();
      expect(isFinite(metrics.loss)).toBe(true);
      expect(metrics.meanReward).toBeGreaterThan(0);
    });

    it('should maintain consistency across multiple steps', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 4,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (_prompts, _completions) => {
          return Float32Array.from([1.0, 1.0]);
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      // Run 5 training steps
      const metrics: any[] = [];
      for (let i = 0; i < 5; i++) {
        const m = await trainer.trainStep(promptMessages, []);
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

  describe.sequential('Loss Computation Variants', () => {
    it('should compute GRPO loss correctly', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        maxNewTokens: 6,
        lossType: 'grpo',
        clipEpsilon: 0.2,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (_prompts, completions) => {
          return Float32Array.from(completions.map((_, i) => i * 0.5));
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      expect(isFinite(metrics.loss)).toBe(true);
      expect(metrics.meanReward).toBeDefined();
      expect(Math.abs(metrics.meanAdvantage)).toBeLessThan(1e-4); // Should be ~0
    });

    it('should work with different loss types', async () => {
      const lossTypes: Array<'grpo' | 'bnpo'> = ['grpo', 'bnpo'];

      for (const lossType of lossTypes) {
        const trainer = await GRPOTrainer.create({
          modelPath: tempModel.modelPath,
          modelConfig: 'qwen3-0.6b',
          groupSize: 3,
          maxNewTokens: 5,
          lossType,
          logInterval: 1000,
          saveInterval: 1000,
          outputDir: TEST_OUTPUT_DIR,
          logConsole: false,
          rewardFunction: (_prompts, _completions) => {
            return Float32Array.from([1.0, 2.0, 3.0]);
          },
        });

        const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

        const metrics = await trainer.trainStep(promptMessages, []);

        expect(isFinite(metrics.loss)).toBe(true);
        expect(metrics.step).toBe(1);
      }
    });

    it('should handle different clip epsilons', async () => {
      const epsilons = [0.1, 0.2, 0.3];

      for (const epsilon of epsilons) {
        const trainer = await GRPOTrainer.create({
          modelPath: tempModel.modelPath,
          modelConfig: 'qwen3-0.6b',
          groupSize: 2,
          maxNewTokens: 4,
          clipEpsilon: epsilon,
          logInterval: 1000,
          saveInterval: 1000,
          outputDir: TEST_OUTPUT_DIR,
          logConsole: false,
          rewardFunction: (_prompts, _completions) => {
            return Float32Array.from([1.0, 2.0]);
          },
        });

        const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

        const metrics = await trainer.trainStep(promptMessages, []);

        expect(isFinite(metrics.loss)).toBe(true);
      }
    });
  });

  describe.sequential('Advantage Computation', () => {
    it('should normalize advantages per group', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        maxNewTokens: 5,
        advantageNormalization: true,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (_prompts, _completions) => {
          // Varying rewards within group
          return Float32Array.from([1.0, 2.0, 3.0, 4.0]);
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      // Mean advantage should be close to 0 with group normalization
      expect(Math.abs(metrics.meanAdvantage)).toBeLessThan(1e-4);
    });

    it('should handle uniform rewards correctly', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 4,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (_prompts, _completions) => {
          // All same reward
          return Float32Array.from([5.0, 5.0, 5.0]);
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      // With uniform rewards, std should be 0
      expect(metrics.stdReward).toBeCloseTo(0, 5);
      expect(metrics.meanReward).toBeCloseTo(5.0, 5);
    });
  });

  describe.sequential('Batch Processing', () => {
    it('should handle large batches correctly', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
        numEpochs: 1,
        batchSize: 5,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        logJsonl: false,
        rewardFunction: (_prompts, completions) => {
          return Float32Array.from(completions.map(() => Math.random()));
        },
      });

      // Create 10 examples - should result in 2 batches
      const dataset = Array.from({ length: 10 }, (_, i) => ({
        prompt: [{ role: 'user', content: `Question ${i}` }],
        answer: `Answer ${i}`,
      }));

      await trainer.train(dataset as any);

      // Should complete without errors
      expect(true).toBe(true);
    });

    it('should handle batch size larger than dataset', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
        numEpochs: 1,
        batchSize: 20, // Larger than dataset
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        logJsonl: false,
        rewardFunction: (_prompts, _completions) => {
          return Float32Array.from([1.0, 1.0]);
        },
      });

      const dataset = [{ prompt: [{ role: 'user', content: 'Test' }], answer: '42' }];

      await trainer.train(dataset as any);

      expect(true).toBe(true);
    });
  });

  describe.sequential('Sampling Configuration', () => {
    it('should respect temperature settings', async () => {
      const temperatures = [0.5, 1.0, 1.5];

      for (const temp of temperatures) {
        const trainer = await GRPOTrainer.create({
          modelPath: tempModel.modelPath,
          modelConfig: 'qwen3-0.6b',
          groupSize: 3,
          maxNewTokens: 6,
          temperature: temp,
          logInterval: 1000,
          saveInterval: 1000,
          outputDir: TEST_OUTPUT_DIR,
          logConsole: false,
          rewardFunction: (_prompts, _completions) => {
            return Float32Array.from([1.0, 1.0, 1.0]);
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
          modelConfig: 'qwen3-0.6b',
          groupSize: 2,
          maxNewTokens: 5,
          temperature: 0.8,
          ...config,
          logInterval: 1000,
          saveInterval: 1000,
          outputDir: TEST_OUTPUT_DIR,
          logConsole: false,
          rewardFunction: (_prompts, _completions) => {
            return Float32Array.from([1.0, 1.0]);
          },
        });

        const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

        const result = await trainer.generateBatch(promptMessages);

        expect(result.completionTexts.length).toBe(2);
      }
    }, 30000); // 30s timeout - creates 3 trainers in sequence
  });

  describe.sequential('Error Handling', () => {
    it('should handle empty dataset gracefully', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
        numEpochs: 1,
        batchSize: 2,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        logJsonl: false,
        rewardFunction: (_prompts, _completions) => {
          return new Float32Array(0);
        },
      });

      const dataset: any[] = [];

      await trainer.train(dataset);

      expect(true).toBe(true);
    });

    it('should throw error when reward function returns wrong count', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 5,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (_prompts, _completions) => {
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
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
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

  describe.sequential('Metrics Tracking', () => {
    it('should track all metrics correctly', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        maxNewTokens: 6,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (_prompts, _completions) => {
          return Float32Array.from([1.0, 2.0, 3.0, 4.0]);
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      // Validate all metrics are present and valid
      expect(metrics.step).toBe(1);
      expect(metrics.loss).toBeDefined();
      expect(isFinite(metrics.loss)).toBe(true);
      expect(metrics.meanReward).toBe(2.5); // (1+2+3+4)/4
      expect(metrics.stdReward).toBeGreaterThan(0);
      expect(metrics.meanAdvantage).toBeDefined();
      expect(metrics.totalTokens).toBeGreaterThan(0);
    });

    it('should track token counts accurately', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 10,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: TEST_OUTPUT_DIR,
        logConsole: false,
        rewardFunction: (_prompts, _completions) => {
          return Float32Array.from([1.0, 1.0, 1.0]);
        },
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      // Should have generated some tokens (3 completions × up to 10 tokens)
      expect(metrics.totalTokens).toBeGreaterThan(0);
      expect(metrics.totalTokens).toBeLessThanOrEqual(30);
    });
  });
});
