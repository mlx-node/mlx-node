import { GRPOTrainer, type ChatMessage, type RewardOutput } from '@mlx-node/trl';
import { describe, it, expect, beforeAll, afterAll } from 'vite-plus/test';

import { createTempModel } from '../test-model-utils';

// Shared temp model for all tests
let tempModel: { modelPath: string; cleanup: () => void };

beforeAll(async () => {
  tempModel = await createTempModel();
});

afterAll(() => {
  tempModel?.cleanup();
});

describe('GRPOTrainer - generateBatch()', () => {
  describe('Basic Functionality', () => {
    it('should generate completions for single prompt', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 4,
        maxCompletionLength: 10,
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
        expect(result.tokenCounts[i]).toBeLessThanOrEqual(10); // maxCompletionLength
      }
    });

    it('should generate completions for multiple prompts', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 8,
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
        modelName: 'qwen3-0.6b',
        groupSize: 5,
        maxCompletionLength: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];

      // Don't specify groupSize - should use default (5)
      const result = await trainer.generateBatch([messages]);

      expect(result.completionTexts.length).toBe(5);
    });
  });

  describe('Generation Configuration', () => {
    it('should respect maxCompletionLength parameter', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      // All generations should have <= maxCompletionLength
      for (let i = 0; i < result.tokenCounts.length; i++) {
        expect(result.tokenCounts[i]).toBeLessThanOrEqual(5);
      }
    });

    it('should generate diverse completions with temperature > 0', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 5,
        maxCompletionLength: 10,
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
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 8,
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
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 8,
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
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 6,
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
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 5,
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
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
      });

      const result = await trainer.generateBatch([]);

      expect(result.completionTexts.length).toBe(0);
      expect(result.tokenCounts.length).toBe(0);
    });

    it('should handle groupSize of 1', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 1,
        maxCompletionLength: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      expect(result.completionTexts.length).toBe(1);
    });

    it('should handle very short maxCompletionLength', async () => {
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 1,
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
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
      });

      // Create 10 prompts
      const prompts = Array.from({ length: 10 }, (_, i) => [{ role: 'user' as const, content: `Test prompt ${i}` }]);

      const result = await trainer.generateBatch(prompts);

      // Should have 20 generations total (2 per prompt)
      expect(result.completionTexts.length).toBe(20);
    });
  });
});

describe('GRPOTrainer - Constructor', () => {
  it('should create trainer with default config', async () => {
    const trainer = await GRPOTrainer.create({ modelPath: tempModel.modelPath, modelName: 'qwen3-0.6b' });
    expect(trainer).toBeDefined();
  });

  it('should create trainer with custom config', async () => {
    const trainer = await GRPOTrainer.create({
      modelPath: tempModel.modelPath,
      modelName: 'qwen3-0.6b',
      groupSize: 16,
      maxCompletionLength: 512,
      temperature: 1.0,
    });
    expect(trainer).toBeDefined();
  });

  it('should throw error for missing model path', async () => {
    await expect(GRPOTrainer.create({ modelName: 'qwen3-0.6b' })).rejects.toThrow('modelPath is required');
  });

  it('should throw error for nonpositive maxCompletionLength (fail fast, no model load)', async () => {
    // A 0/negative budget makes every completion an empty "length"-capped
    // sample → the GRPO filter drops them all → training advances bookkeeping
    // without applying gradients (a silent no-op). Reject up front, before the
    // (slow) model load. The `modelPath` is present-but-irrelevant: the
    // maxCompletionLength check throws before any disk access.
    await expect(
      GRPOTrainer.create({ modelPath: '/nonexistent', modelName: 'qwen3-0.6b', maxCompletionLength: 0 }),
    ).rejects.toThrow('maxCompletionLength must be a positive integer');
    await expect(
      GRPOTrainer.create({ modelPath: '/nonexistent', modelName: 'qwen3-0.6b', maxCompletionLength: -1 }),
    ).rejects.toThrow('maxCompletionLength must be a positive integer');
  });
});

describe('GRPOTrainer - scoreGenerations()', () => {
  describe('Basic Functionality', () => {
    it('should score completions with custom reward function', async () => {
      // Simple reward function: length-based
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map((o) => o.completion.rawText.length / 100));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 5,
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
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
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
        modelName: 'qwen3-0.6b',
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
      const rewardFn = (_outputs: RewardOutput[]) => {
        return new Float32Array(0);
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        rewardFunction: rewardFn,
      });

      const rewards = await trainer.scoreGenerations([], [], [], 2);

      expect(rewards.length).toBe(0);
    });
  });
});

describe('GRPOTrainer - trainStep()', () => {
  describe('Basic Functionality', () => {
    it('should execute a training step and return metrics', async () => {
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 5,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages);

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
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 3,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics1 = await trainer.trainStep(promptMessages);
      expect(metrics1.step).toBe(1);

      const metrics2 = await trainer.trainStep(promptMessages);
      expect(metrics2.step).toBe(2);

      const metrics3 = await trainer.trainStep(promptMessages);
      expect(metrics3.step).toBe(3);
    });

    it('should handle multiple prompts', async () => {
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map(() => Math.random()));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 4,
        rewardFunction: rewardFn,
      });

      const promptMessages = [
        [{ role: 'user' as const, content: 'Test prompt 1' }],
        [{ role: 'user' as const, content: 'Test prompt 2' }],
      ];

      const metrics = await trainer.trainStep(promptMessages);

      expect(metrics).toBeDefined();
      expect(metrics.totalTokens).toBeGreaterThan(0);
      expect(isFinite(metrics.loss)).toBe(true);

      // Second trainStep with DIFFERENT prompts. This catches the B1 regression
      // where the generation cache would silently hold the wrong prompts after
      // the per-prompt dispatch loop overwrote it each iteration.
      const otherPromptMessages = [
        [{ role: 'user' as const, content: 'Another question A' }],
        [{ role: 'user' as const, content: 'Another question B' }],
      ];

      const metrics2 = await trainer.trainStep(otherPromptMessages);

      expect(metrics2).toBeDefined();
      expect(metrics2.totalTokens).toBeGreaterThan(0);
      expect(isFinite(metrics2.loss)).toBe(true);
      expect(metrics2.step).toBe(metrics.step + 1);
    });
  });

  // Reward mean/std math: `crates/mlx-core/src/grpo/engine.rs` →
  // `test_compute_reward_stats` / `test_compute_reward_stats_empty`.
  // Group-normalised advantages sum to zero: `crates/mlx-core/src/grpo/advantages.rs` →
  // `test_compute_advantages_basic` / `test_uniform_rewards_zero_advantage`.
  // The TS side only checks the values reach `metrics` (see 'Integration' below).
  describe('Metrics Validation', () => {
    it('should track total tokens', async () => {
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 5,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages);

      // Should have generated some tokens (3 completions × up to 5 tokens each)
      expect(metrics.totalTokens).toBeGreaterThan(0);
      expect(metrics.totalTokens).toBeLessThanOrEqual(15); // 3 × 5 max
    });
  });

  // Per-loss-type math (grpo/bnpo/dr_grpo/dapo): `crates/mlx-core/src/grpo/loss.rs` →
  // `test_grpo_loss_basic`, `test_bnpo_loss_computation`, …. One end-to-end
  // `lossType` plumbing check stays in `grpo-integration.test.ts`.
  describe('Loss Computation', () => {
    it('should compute GRPO loss without errors', async () => {
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map((_, i) => i * 0.5));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 4,
        maxCompletionLength: 4,
        rewardFunction: rewardFn,
        lossType: 'grpo',
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages);

      // Loss should be finite
      expect(isFinite(metrics.loss)).toBe(true);
      expect(isNaN(metrics.loss)).toBe(false);
    });
  });

  describe('Edge Cases', () => {
    it('should handle single prompt with groupSize 1', async () => {
      const rewardFn = (_outputs: RewardOutput[]) => {
        return Float32Array.from([0.5]);
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 1,
        maxCompletionLength: 3,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages);

      expect(metrics).toBeDefined();
      expect(metrics.step).toBe(1);
      expect(isFinite(metrics.loss)).toBe(true);
    });

    it('should handle varying reward values', async () => {
      const rewardFn = (_outputs: RewardOutput[]) => {
        // Very different rewards
        return Float32Array.from([10.0, 0.1, 5.0, -2.0]);
      };

      // `stdReward` is computed over the completions that SURVIVE the
      // degenerate-length filter (`finish_reason == "length"` and tokens >=
      // 0.9 * maxCompletionLength — `compute_balanced_valid_indices` in
      // `crates/mlx-core/src/grpo/engine.rs`), falling back to all rewards
      // only when nothing survives. With sampling on random weights, exactly
      // one of the four can hit EOS early and survive alone → std 0 (seen in
      // CI). Greedy decoding makes the four identical-prompt completions
      // identical, so they all survive or all get filtered — either way the
      // std is over all four distinct rewards.
      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 4,
        maxCompletionLength: 3,
        temperature: 0,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages);

      expect(metrics.stdReward).toBeGreaterThan(0); // Should have variance
      expect(isFinite(metrics.loss)).toBe(true);
    });
  });

  describe('Integration', () => {
    it('should integrate generateBatch, scoreGenerations, and loss computation', async () => {
      let scoreCalled = 0;

      const rewardFn = (outputs: RewardOutput[]) => {
        scoreCalled++;
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 4,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics = await trainer.trainStep(promptMessages);

      // Reward function should have been called
      expect(scoreCalled).toBe(1);

      // Metrics should be valid
      expect(metrics.step).toBe(1);
      expect(isFinite(metrics.loss)).toBe(true);
      expect(metrics.meanReward).toBe(1.0);
    });

    it('should handle multiple training steps in sequence', async () => {
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map(() => Math.random()));
      };

      const trainer = await GRPOTrainer.create({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 3,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      // Run 3 training steps
      const metrics1 = await trainer.trainStep(promptMessages);
      const metrics2 = await trainer.trainStep(promptMessages);
      const metrics3 = await trainer.trainStep(promptMessages);

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

// `train()` is the TS epoch × batch loop in `packages/trl/src/trainers/grpo-trainer.ts`.
// Each batch is one `trainStep`, which calls the reward function exactly once with
// `batch.length * groupSize` outputs — so the sequence of reward-call sizes IS the
// batching contract. (These used to be "doesn't throw" tests; the integration
// file's copies of the same scenarios were folded in here.)
describe('GRPOTrainer - train()', () => {
  const example = (content: string) => ({ prompt: [{ role: 'user' as const, content }], answer: '42' });

  async function runTrain(
    options: { numEpochs: number; batchSize: number; groupSize: number },
    dataset: ReturnType<typeof example>[],
  ): Promise<number[]> {
    const rewardCallSizes: number[] = [];
    const trainer = await GRPOTrainer.create({
      modelPath: tempModel.modelPath,
      modelName: 'qwen3-0.6b',
      maxCompletionLength: 10,
      logInterval: 1000,
      saveInterval: 1000, // Don't checkpoint mid-run
      outputDir: './test-output',
      logConsole: false,
      logJsonl: false,
      ...options,
      rewardFunction: (outputs) => {
        rewardCallSizes.push(outputs.length);
        return new Float32Array(outputs.length).fill(1.0);
      },
    });
    await trainer.train(dataset as any);
    return rewardCallSizes;
  }

  describe('Basic Training Loop', () => {
    it('runs one step per batch', async () => {
      const sizes = await runTrain({ numEpochs: 1, batchSize: 2, groupSize: 2 }, [
        example('Test 1'),
        example('Test 2'),
      ]);
      // 2 examples / batchSize 2 → 1 batch of 2 prompts × 2 completions.
      expect(sizes).toEqual([4]);
    });

    it('iterates through multiple epochs', async () => {
      const sizes = await runTrain({ numEpochs: 2, batchSize: 1, groupSize: 2 }, [example('Test')]);
      // 1 batch per epoch × 2 epochs.
      expect(sizes).toEqual([2, 2]);
    });

    it('fills a trailing partial batch', async () => {
      const sizes = await runTrain({ numEpochs: 1, batchSize: 2, groupSize: 2 }, [
        example('Test 1'),
        example('Test 2'),
        example('Test 3'),
      ]);
      // 3 examples / batchSize 2 → [2 prompts, 1 prompt] × 2 completions.
      expect(sizes).toEqual([4, 2]);
    });

    it('clears the processed-batch set at an epoch boundary', async () => {
      const sizes = await runTrain({ numEpochs: 2, batchSize: 2, groupSize: 2 }, [
        example('A'),
        example('B'),
        example('C'),
        example('D'),
      ]);
      // If `processedBatchIndices` leaked across epochs, epoch 2 would skip both batches.
      expect(sizes).toEqual([4, 4, 4, 4]);
    });
  });

  describe('Edge Cases', () => {
    it('returns without a step on an empty dataset', async () => {
      const sizes = await runTrain({ numEpochs: 1, batchSize: 2, groupSize: 2 }, []);
      expect(sizes).toEqual([]);
    });

    it('caps a batch larger than the dataset to the dataset', async () => {
      const sizes = await runTrain({ numEpochs: 1, batchSize: 10, groupSize: 2 }, [example('Test')]);
      expect(sizes).toEqual([2]);
    });
  });
});
