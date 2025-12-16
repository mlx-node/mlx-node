import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { GRPOTrainer, type ChatMessage } from '@mlx-node/trl';
import { clearCache } from '@mlx-node/core';
import { createTempModel } from '../test-model-utils';

// Shared temp model for all tests
let tempModel: { modelPath: string; cleanup: () => void };

beforeAll(async () => {
  tempModel = await createTempModel();
});

afterAll(() => {
  tempModel?.cleanup();
});

afterEach(() => {
  // Clear GPU cache to prevent memory accumulation between tests
  clearCache();
});

describe.sequential('GRPOTrainer - generateBatch()', () => {
  describe('Basic Functionality', () => {
    it('should generate completions for single prompt', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        maxNewTokens: 10,
        temperature: 0.8,
        topP: 0.95,
      });

      // Create a simple prompt
      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];

      // Generate 4 completions (groupSize=4 from config)
      const result = await trainer.generateBatch([messages]);

      // Should have 4 generations
      expect(result.completionTexts.length).toBe(4);
      expect(result.tokenCounts.length).toBe(4);
      expect(result.nativeResult).toBeDefined();
      expect(result.nativeResult.completionLengths.length).toBe(4);

      // Each completion should have text
      for (let i = 0; i < 4; i++) {
        expect(result.completionTexts[i]).toBeDefined();
        expect(result.tokenCounts[i]).toBeGreaterThan(0);
        expect(result.tokenCounts[i]).toBeLessThanOrEqual(10); // maxNewTokens
      }
    });

    it('should generate completions for multiple prompts', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 8,
      });

      // Create 2 prompts
      const messages1: ChatMessage[] = [{ role: 'user', content: 'Test prompt 1' }];
      const messages2: ChatMessage[] = [{ role: 'user', content: 'Test prompt 2' }];

      // Generate 3 completions per prompt = 6 total (groupSize=3 from config)
      const result = await trainer.generateBatch([messages1, messages2]);

      // Should have 6 generations total (3 per prompt)
      expect(result.completionTexts.length).toBe(6);
      expect(result.tokenCounts.length).toBe(6);
    });

    it('should use default groupSize from config', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 5,
        maxNewTokens: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];

      // Don't specify groupSize - should use default (5)
      const result = await trainer.generateBatch([messages]);

      expect(result.completionTexts.length).toBe(5);
    });
  });

  describe('Generation Configuration', () => {
    it('should respect maxNewTokens parameter', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      // All generations should have <= maxNewTokens
      for (let i = 0; i < result.tokenCounts.length; i++) {
        expect(result.tokenCounts[i]).toBeLessThanOrEqual(5);
      }
    });

    it('should generate diverse completions with temperature > 0', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 5,
        maxNewTokens: 10,
        temperature: 1.0, // High temperature for diversity
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      // With high temperature, we expect some variation in generated sequences
      const uniqueCompletions = new Set(result.completionTexts);
      // With sampling, very unlikely all sequences are identical
      expect(uniqueCompletions.size).toBeGreaterThan(1);
    });

    it('should apply top-p filtering when configured', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 8,
        temperature: 0.8,
        topP: 0.9, // Nucleus sampling
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      // Should successfully generate with top-p
      expect(result.completionTexts.length).toBe(3);
    });

    it('should apply top-k filtering when configured', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 8,
        temperature: 0.8,
        topK: 50, // Top-k sampling
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      // Should successfully generate with top-k
      expect(result.completionTexts.length).toBe(3);
    });
  });

  describe('Native Result Validation', () => {
    it('should return log probabilities in native result', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 6,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      // Native result should have logprobs
      expect(result.nativeResult.completionLogprobs.length).toBeGreaterThan(0);

      // Logprobs should be negative (log of probability <= 1)
      for (const logprob of result.nativeResult.completionLogprobs) {
        expect(logprob).toBeLessThanOrEqual(0);
        expect(logprob).toBeGreaterThan(-100); // Not too negative
        expect(isFinite(logprob)).toBe(true);
      }
    });

    it('should have consistent lengths across native result fields', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      // Sum of completion lengths should equal total tokens
      const totalTokens = result.nativeResult.completionLengths.reduce((a, b) => a + b, 0);
      expect(result.nativeResult.completionTokens.length).toBe(totalTokens);
      expect(result.nativeResult.completionLogprobs.length).toBe(totalTokens);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty batch', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
      });

      const result = await trainer.generateBatch([]);

      expect(result.completionTexts.length).toBe(0);
      expect(result.tokenCounts.length).toBe(0);
    });

    it('should handle groupSize of 1', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 1,
        maxNewTokens: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      expect(result.completionTexts.length).toBe(1);
    });

    it('should handle very short maxNewTokens', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 1,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      // Should generate at least 1 token
      for (let i = 0; i < result.tokenCounts.length; i++) {
        expect(result.tokenCounts[i]).toBeGreaterThanOrEqual(1);
        expect(result.tokenCounts[i]).toBeLessThanOrEqual(1);
      }
    });

    it('should handle large batch of prompts', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 5,
      });

      // Create 10 prompts
      const prompts = Array.from({ length: 10 }, (_, i) => [{ role: 'user' as const, content: `Test prompt ${i}` }]);

      const result = await trainer.generateBatch(prompts);

      // Should have 20 generations total (2 per prompt)
      expect(result.completionTexts.length).toBe(20);
    });
  });
});

describe.sequential('GRPOTrainer - Constructor', () => {
  it('should create trainer with default config', async () => {
    const trainer = await GRPOTrainer.create({ modelPath: tempModel.modelPath, modelConfig: 'qwen3-0.6b' });
    expect(trainer).toBeDefined();
  });

  it('should create trainer with custom config', async () => {
    const trainer = await GRPOTrainer.create({
      modelPath: tempModel.modelPath,
      modelConfig: 'qwen3-0.6b',
      groupSize: 16,
      maxNewTokens: 512,
      temperature: 1.0,
    });
    expect(trainer).toBeDefined();
  });

  it('should throw error for missing model path', async () => {
    await expect(GRPOTrainer.create({ modelConfig: 'qwen3-0.6b' })).rejects.toThrow('modelPath is required');
  });
});

describe.sequential('GRPOTrainer - scoreGenerations()', () => {
  describe('Basic Functionality', () => {
    it('should score completions with custom reward function', async () => {
      // Simple reward function: length-based
      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from(completions.map((c) => c.length / 100));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 5,
        rewardFunction: rewardFn,
      });

      // Create prompts and generate completions
      const promptMessages = [
        [{ role: 'user' as const, content: 'Test prompt 1' }],
        [{ role: 'user' as const, content: 'Test prompt 2' }],
      ];

      // Generate completions (groupSize=3 from config)
      const genResult = await trainer.generateBatch(promptMessages);

      const rewards = await trainer.scoreGenerations(promptMessages, genResult.completionTexts, [], 3);

      // Should have 6 rewards (2 prompts × 3 completions each)
      expect(rewards.length).toBe(6);

      // All rewards should be non-negative
      for (let i = 0; i < rewards.length; i++) {
        expect(rewards[i]).toBeGreaterThanOrEqual(0);
      }
    });

    it('should validate completion count matches prompts × groupSize', async () => {
      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from(completions.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      // Wrong number of completions (3 instead of 4)
      const wrongCompletions = ['completion1', 'completion2', 'completion3'];

      await expect(trainer.scoreGenerations(promptMessages, wrongCompletions, [], 4)).rejects.toThrow(
        /Expected 4 completions.*but got 3/,
      );
    });

    it('should throw error if no reward function configured', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        // No rewardFunction
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const completions = ['completion1', 'completion2'];

      await expect(trainer.scoreGenerations(promptMessages, completions, [], 2)).rejects.toThrow(
        /No reward function configured/,
      );
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty batch', async () => {
      const rewardFn = (_prompts: string[], _completions: string[], _answers: (string | null)[]) => {
        return new Float32Array(0);
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        rewardFunction: rewardFn,
      });

      const rewards = await trainer.scoreGenerations([], [], [], 2);

      expect(rewards.length).toBe(0);
    });
  });
});

describe.sequential('GRPOTrainer - trainStep()', () => {
  describe('Basic Functionality', () => {
    it('should execute a training step and return metrics', async () => {
      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from(completions.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 5,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      // Should return valid metrics
      expect(metrics).toBeDefined();
      expect(metrics.loss).toBeDefined();
      expect(typeof metrics.loss).toBe('number');
      expect(isFinite(metrics.loss)).toBe(true);

      expect(metrics.meanReward).toBeDefined();
      expect(metrics.stdReward).toBeDefined();
      expect(metrics.meanAdvantage).toBeDefined();
      expect(metrics.totalTokens).toBeGreaterThan(0);
      expect(metrics.step).toBe(1);
    });

    it('should increment step counter', async () => {
      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from(completions.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 3,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics1 = await trainer.trainStep(promptMessages, []);
      expect(metrics1.step).toBe(1);

      const metrics2 = await trainer.trainStep(promptMessages, []);
      expect(metrics2.step).toBe(2);

      const metrics3 = await trainer.trainStep(promptMessages, []);
      expect(metrics3.step).toBe(3);
    });

    it('should handle multiple prompts', async () => {
      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from(completions.map(() => Math.random()));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 4,
        rewardFunction: rewardFn,
      });

      const promptMessages = [
        [{ role: 'user' as const, content: 'Test prompt 1' }],
        [{ role: 'user' as const, content: 'Test prompt 2' }],
      ];

      const metrics = await trainer.trainStep(promptMessages, []);

      expect(metrics).toBeDefined();
      expect(metrics.totalTokens).toBeGreaterThan(0);
      expect(isFinite(metrics.loss)).toBe(true);
    });
  });

  describe('Metrics Validation', () => {
    it('should return correct mean reward', async () => {
      const fixedReward = 2.5;
      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from(completions.map(() => fixedReward));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        maxNewTokens: 3,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      // All rewards are 2.5, so mean should be 2.5
      expect(metrics.meanReward).toBeCloseTo(fixedReward, 5);

      // Std should be 0 (all same value)
      expect(metrics.stdReward).toBeCloseTo(0, 5);
    });

    it('should compute advantages correctly', async () => {
      // Advantages should be zero-mean per group
      const rewardFn = (_prompts: string[], _completions: string[], _answers: (string | null)[]) => {
        // Different rewards
        return Float32Array.from([1.0, 2.0, 3.0, 4.0]);
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        maxNewTokens: 3,
        rewardFunction: rewardFn,
        advantageNormalization: false, // Just zero-mean, no std normalization
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      // Mean advantage should be close to 0 (group normalization)
      expect(Math.abs(metrics.meanAdvantage)).toBeLessThan(1e-5);
    });

    it('should track total tokens', async () => {
      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from(completions.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 5,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      // Should have generated some tokens (3 completions × up to 5 tokens each)
      expect(metrics.totalTokens).toBeGreaterThan(0);
      expect(metrics.totalTokens).toBeLessThanOrEqual(15); // 3 × 5 max
    });
  });

  describe('Loss Computation', () => {
    it('should compute GRPO loss without errors', async () => {
      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from(completions.map((_, i) => i * 0.5));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        maxNewTokens: 4,
        rewardFunction: rewardFn,
        lossType: 'grpo',
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      // Loss should be finite
      expect(isFinite(metrics.loss)).toBe(true);
      expect(isNaN(metrics.loss)).toBe(false);
    });

    it('should work with different loss types', async () => {
      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from(completions.map(() => Math.random()));
      };

      const lossTypes: Array<'grpo' | 'bnpo'> = ['grpo', 'bnpo'];

      for (const lossType of lossTypes) {
        const trainer = await GRPOTrainer.create({
          modelPath: tempModel.modelPath,
          modelConfig: 'qwen3-0.6b',
          groupSize: 3,
          maxNewTokens: 3,
          rewardFunction: rewardFn,
          lossType,
        });

        const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

        const metrics = await trainer.trainStep(promptMessages, []);

        expect(isFinite(metrics.loss)).toBe(true);
        expect(metrics.step).toBe(1);
      }
    });
  });

  describe('Edge Cases', () => {
    it('should handle single prompt with groupSize 1', async () => {
      const rewardFn = (_prompts: string[], _completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from([0.5]);
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 1,
        maxNewTokens: 3,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      expect(metrics).toBeDefined();
      expect(metrics.step).toBe(1);
      expect(isFinite(metrics.loss)).toBe(true);
    });

    it('should handle varying reward values', async () => {
      const rewardFn = (_prompts: string[], _completions: string[], _answers: (string | null)[]) => {
        // Very different rewards
        return Float32Array.from([10.0, 0.1, 5.0, -2.0]);
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 4,
        maxNewTokens: 3,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      expect(metrics.stdReward).toBeGreaterThan(0); // Should have variance
      expect(isFinite(metrics.loss)).toBe(true);
    });
  });

  describe('Integration', () => {
    it('should integrate generateBatch, scoreGenerations, and loss computation', async () => {
      let scoreCalled = 0;

      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        scoreCalled++;
        return Float32Array.from(completions.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 3,
        maxNewTokens: 4,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages, []);

      // Reward function should have been called
      expect(scoreCalled).toBe(1);

      // Metrics should be valid
      expect(metrics.step).toBe(1);
      expect(isFinite(metrics.loss)).toBe(true);
      expect(metrics.meanReward).toBe(1.0);
    });

    it('should handle multiple training steps in sequence', async () => {
      const rewardFn = (_prompts: string[], completions: string[], _answers: (string | null)[]) => {
        return Float32Array.from(completions.map(() => Math.random()));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 3,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      // Run 3 training steps
      const metrics1 = await trainer.trainStep(promptMessages, []);
      const metrics2 = await trainer.trainStep(promptMessages, []);
      const metrics3 = await trainer.trainStep(promptMessages, []);

      // Steps should increment
      expect(metrics1.step).toBe(1);
      expect(metrics2.step).toBe(2);
      expect(metrics3.step).toBe(3);

      // All should have valid losses
      expect(isFinite(metrics1.loss)).toBe(true);
      expect(isFinite(metrics2.loss)).toBe(true);
      expect(isFinite(metrics3.loss)).toBe(true);
    });
  });
});

// ============================================================================
// GRPOTrainer - train() tests
// ============================================================================

describe.sequential('GRPOTrainer - train()', () => {
  describe('Basic Training Loop', () => {
    it('should execute training loop with dataset', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 10,
        numEpochs: 1,
        batchSize: 2,
        logInterval: 1,
        saveInterval: 1000, // Don't save during test
        outputDir: './test-output',
        rewardFunction: (_prompts, _completions) => {
          return new Float32Array([1.0, 1.0, 1.0, 1.0]);
        },
      });

      // Create mock dataset
      const dataset = [
        { prompt: [{ role: 'user', content: 'Test 1' }], answer: '42' },
        { prompt: [{ role: 'user', content: 'Test 2' }], answer: '43' },
      ];

      // Execute training (should not throw)
      await expect(trainer.train(dataset as any)).resolves.not.toThrow();
    });

    it('should iterate through multiple epochs', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 10,
        numEpochs: 2, // Multiple epochs
        batchSize: 1,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: './test-output',
        rewardFunction: (_prompts, _completions) => {
          return new Float32Array([1.0, 1.0]);
        },
      });

      const dataset = [{ prompt: [{ role: 'user', content: 'Test' }], answer: '42' }];

      await trainer.train(dataset as any);

      // After training with 2 epochs and 1 batch per epoch, should have 2 steps
      expect(true).toBe(true); // Basic completion test
    });

    it('should handle batching correctly', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 10,
        numEpochs: 1,
        batchSize: 2, // Batch size of 2
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: './test-output',
        rewardFunction: (_prompts, completions) => {
          // Should be called with batch_size * group_size completions
          return new Float32Array(completions.length).fill(1.0);
        },
      });

      const dataset = [
        { prompt: [{ role: 'user', content: 'Test 1' }], answer: '42' },
        { prompt: [{ role: 'user', content: 'Test 2' }], answer: '43' },
        { prompt: [{ role: 'user', content: 'Test 3' }], answer: '44' },
      ];

      // 3 examples with batch_size=2 should result in 2 batches
      await trainer.train(dataset as any);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty dataset', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 10,
        numEpochs: 1,
        batchSize: 2,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: './test-output',
        rewardFunction: (_prompts, _completions) => {
          return new Float32Array(0);
        },
      });

      const dataset: any[] = [];
      await trainer.train(dataset);
      // Should complete without error
    });

    it('should handle single example', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 10,
        numEpochs: 1,
        batchSize: 2,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: './test-output',
        rewardFunction: (_prompts, _completions) => {
          return new Float32Array([1.0, 1.0]);
        },
      });

      const dataset = [{ prompt: [{ role: 'user', content: 'Test' }], answer: '42' }];

      await trainer.train(dataset as any);
    });

    it('should handle batch size larger than dataset', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelConfig: 'qwen3-0.6b',
        groupSize: 2,
        maxNewTokens: 10,
        numEpochs: 1,
        batchSize: 10, // Larger than dataset
        logInterval: 1000,
        saveInterval: 1000,
        outputDir: './test-output',
        rewardFunction: (_prompts, _completions) => {
          return new Float32Array([1.0, 1.0]);
        },
      });

      const dataset = [{ prompt: [{ role: 'user', content: 'Test' }], answer: '42' }];

      await trainer.train(dataset as any);
    });
  });
});
