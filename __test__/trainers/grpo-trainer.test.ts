import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { loadModel, type TrainableModel } from '@mlx-node/lm';
import { GRPOTrainer, type ChatMessage, type GRPOTrainerConfig, type RewardOutput } from '@mlx-node/trl';
import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vite-plus/test';

import { createTempModel } from '../test-model-utils';

// Shared temp model and pre-loaded read-only instance for generation/scoring
// tests. Weight-mutating trainStep/train tests reload a fresh model from the
// shared temp checkpoint via `createTrainTrainer` so no gradient update leaks
// between tests.
let tempModel: { modelPath: string; cleanup: () => void };
let evalModel: TrainableModel;
const activeTrainers: GRPOTrainer[] = [];

beforeAll(async () => {
  tempModel = await createTempModel();
  evalModel = (await loadModel(tempModel.modelPath)) as TrainableModel;
}, 60_000);

afterEach(() => {
  for (const trainer of activeTrainers.splice(0)) {
    trainer.reset();
  }
});

afterAll(() => {
  tempModel?.cleanup();
});

function createTrainer(model: TrainableModel, config: Partial<GRPOTrainerConfig>): GRPOTrainer {
  const trainer = new GRPOTrainer(model, config);
  activeTrainers.push(trainer);
  return trainer;
}

async function createTrainTrainer(config: Partial<GRPOTrainerConfig>): Promise<GRPOTrainer> {
  const model = (await loadModel(tempModel.modelPath)) as TrainableModel;
  return createTrainer(model, config);
}

describe('GRPOTrainer - generateBatch()', () => {
  describe('Basic Functionality', () => {
    it('should generate completions for single prompt', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 4,
        maxCompletionLength: 10,
        temperature: 0.8,
        topP: 0.95,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      expect(result.completionTexts.length).toBe(4);
      expect(result.tokenCounts.length).toBe(4);
      expect(result.nativeResult).toBeDefined();
      expect(result.nativeResult.completionLengths.length).toBe(4);

      for (let i = 0; i < 4; i++) {
        expect(result.completionTexts[i]).toBeDefined();
        expect(result.tokenCounts[i]).toBeGreaterThan(0);
        expect(result.tokenCounts[i]).toBeLessThanOrEqual(10);
      }
    });

    it('should generate completions for multiple prompts', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 8,
      });

      const messages1: ChatMessage[] = [{ role: 'user', content: 'Test prompt 1' }];
      const messages2: ChatMessage[] = [{ role: 'user', content: 'Test prompt 2' }];
      const result = await trainer.generateBatch([messages1, messages2]);

      expect(result.completionTexts.length).toBe(6);
      expect(result.tokenCounts.length).toBe(6);
    });

    it('should use default groupSize from config', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 5,
        maxCompletionLength: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      expect(result.completionTexts.length).toBe(5);
    });
  });

  describe('Generation Configuration', () => {
    it('should respect maxCompletionLength parameter', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      for (let i = 0; i < result.tokenCounts.length; i++) {
        expect(result.tokenCounts[i]).toBeLessThanOrEqual(5);
      }
    });

    it('should generate diverse completions with temperature > 0', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 5,
        maxCompletionLength: 10,
        temperature: 1.0,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      const uniqueCompletions = new Set(result.completionTexts);
      expect(uniqueCompletions.size).toBeGreaterThan(1);
    });

    it('should apply top-p filtering when configured', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 8,
        temperature: 0.8,
        topP: 0.9,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      expect(result.completionTexts.length).toBe(3);
    });

    it('should apply top-k filtering when configured', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 8,
        temperature: 0.8,
        topK: 50,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      expect(result.completionTexts.length).toBe(3);
    });
  });

  describe('Native Result Validation', () => {
    it('should return log probabilities in native result', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 6,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      expect(result.nativeResult.completionLogprobs.length).toBeGreaterThan(0);

      for (const logprob of result.nativeResult.completionLogprobs) {
        expect(logprob).toBeLessThanOrEqual(0);
        expect(logprob).toBeGreaterThan(-100);
        expect(isFinite(logprob)).toBe(true);
      }
    });

    it('should have consistent lengths across native result fields', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      const totalTokens = result.nativeResult.completionLengths.reduce((a, b) => a + b, 0);
      expect(result.nativeResult.completionTokens.length).toBe(totalTokens);
      expect(result.nativeResult.completionLogprobs.length).toBe(totalTokens);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty batch', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
      });

      const result = await trainer.generateBatch([]);

      expect(result.completionTexts.length).toBe(0);
      expect(result.tokenCounts.length).toBe(0);
    });

    it('should handle groupSize of 1', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 1,
        maxCompletionLength: 5,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      expect(result.completionTexts.length).toBe(1);
    });

    it('should handle very short maxCompletionLength', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 1,
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Test prompt' }];
      const result = await trainer.generateBatch([messages]);

      for (let i = 0; i < result.tokenCounts.length; i++) {
        expect(result.tokenCounts[i]).toBeGreaterThanOrEqual(1);
        expect(result.tokenCounts[i]).toBeLessThanOrEqual(1);
      }
    });

    it('should handle large batch of prompts', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 5,
      });

      const prompts = Array.from({ length: 10 }, (_, i) => [{ role: 'user' as const, content: `Test prompt ${i}` }]);
      const result = await trainer.generateBatch(prompts);

      expect(result.completionTexts.length).toBe(20);
    });
  });
});

describe('GRPOTrainer - Constructor', () => {
  it('should create trainer with default config', async () => {
    const trainer = await GRPOTrainer.create({ modelPath: tempModel.modelPath, modelName: 'qwen3-0.6b' });
    activeTrainers.push(trainer);
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
    activeTrainers.push(trainer);
    expect(trainer).toBeDefined();
  });

  it('should throw error for missing model path', async () => {
    await expect(GRPOTrainer.create({ modelName: 'qwen3-0.6b' })).rejects.toThrow('modelPath is required');
  });

  it('should throw error for nonpositive maxCompletionLength (fail fast, no model load)', async () => {
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
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map((o) => o.completion.rawText.length / 100));
      };

      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 5,
        rewardFunction: rewardFn,
      });

      const promptMessages = [
        [{ role: 'user' as const, content: 'Test prompt 1' }],
        [{ role: 'user' as const, content: 'Test prompt 2' }],
      ];

      const genResult = await trainer.generateBatch(promptMessages);
      const rewards = await trainer.scoreGenerations(promptMessages, genResult.completionTexts, [], 3);

      expect(rewards.length).toBe(6);

      for (let i = 0; i < rewards.length; i++) {
        expect(rewards[i]).toBeGreaterThanOrEqual(0);
      }
    });

    it('should validate completion count matches prompts × groupSize', async () => {
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map(() => 1.0));
      };

      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 4,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const wrongCompletions = ['completion1', 'completion2', 'completion3'];

      await expect(trainer.scoreGenerations(promptMessages, wrongCompletions, [], 4)).rejects.toThrow(
        /Expected 4 completions.*but got 3/,
      );
    });

    it('should throw error if no reward function configured', async () => {
      const trainer = createTrainer(evalModel, {
        modelName: 'qwen3-0.6b',
        groupSize: 2,
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

      const trainer = createTrainer(evalModel, {
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

      const trainer = await createTrainTrainer({
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 5,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const metrics = await trainer.trainStep(promptMessages);

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

      const trainer = await createTrainTrainer({
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

      const trainer = await createTrainTrainer({
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

      const trainer = await createTrainTrainer({
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 5,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const metrics = await trainer.trainStep(promptMessages);

      expect(metrics.totalTokens).toBeGreaterThan(0);
      expect(metrics.totalTokens).toBeLessThanOrEqual(15);
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

      const trainer = await createTrainTrainer({
        modelName: 'qwen3-0.6b',
        groupSize: 4,
        maxCompletionLength: 4,
        rewardFunction: rewardFn,
        lossType: 'grpo',
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const metrics = await trainer.trainStep(promptMessages);

      expect(isFinite(metrics.loss)).toBe(true);
      expect(isNaN(metrics.loss)).toBe(false);
    });
  });

  describe('Edge Cases', () => {
    it('should handle single prompt with groupSize 1', async () => {
      const rewardFn = (_outputs: RewardOutput[]) => {
        return Float32Array.from([0.5]);
      };

      const trainer = await createTrainTrainer({
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
        return Float32Array.from([10.0, 0.1, 5.0, -2.0]);
      };

      const trainer = await createTrainTrainer({
        modelName: 'qwen3-0.6b',
        groupSize: 4,
        maxCompletionLength: 3,
        temperature: 0,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const metrics = await trainer.trainStep(promptMessages);

      expect(metrics.stdReward).toBeGreaterThan(0);
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

      const trainer = await createTrainTrainer({
        modelName: 'qwen3-0.6b',
        groupSize: 3,
        maxCompletionLength: 4,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];
      const metrics = await trainer.trainStep(promptMessages);

      expect(scoreCalled).toBe(1);
      expect(metrics.step).toBe(1);
      expect(isFinite(metrics.loss)).toBe(true);
      expect(metrics.meanReward).toBe(1.0);
    });

    it('should handle multiple training steps in sequence', async () => {
      const rewardFn = (outputs: RewardOutput[]) => {
        return Float32Array.from(outputs.map(() => Math.random()));
      };

      const trainer = await createTrainTrainer({
        modelName: 'qwen3-0.6b',
        groupSize: 2,
        maxCompletionLength: 3,
        rewardFunction: rewardFn,
      });

      const promptMessages = [[{ role: 'user' as const, content: 'Test prompt' }]];

      const metrics1 = await trainer.trainStep(promptMessages);
      const metrics2 = await trainer.trainStep(promptMessages);
      const metrics3 = await trainer.trainStep(promptMessages);

      expect(metrics1.step).toBe(1);
      expect(metrics2.step).toBe(2);
      expect(metrics3.step).toBe(3);

      expect(isFinite(metrics1.loss)).toBe(true);
      expect(isFinite(metrics2.loss)).toBe(true);
      expect(isFinite(metrics3.loss)).toBe(true);
    });
  });
});

// ============================================================================
// GRPOTrainer - train() tests
// ============================================================================

describe('GRPOTrainer - train()', () => {
  const example = (content: string) => ({ prompt: [{ role: 'user' as const, content }], answer: '42' });

  async function runTrain(
    options: { numEpochs: number; batchSize: number; groupSize: number; verifyFinalCheckpoint?: boolean },
    dataset: ReturnType<typeof example>[],
  ): Promise<number[]> {
    const rewardCallSizes: number[] = [];
    const outputDir = mkdtempSync(join(tmpdir(), 'grpo-train-'));
    try {
      const trainer = await createTrainTrainer({
        modelPath: tempModel.modelPath,
        modelName: 'qwen3-0.6b',
        maxCompletionLength: 10,
        logInterval: 1000,
        saveInterval: 1000,
        outputDir,
        logConsole: false,
        logJsonl: false,
        numEpochs: options.numEpochs,
        batchSize: options.batchSize,
        groupSize: options.groupSize,
        rewardFunction: (outputs) => {
          rewardCallSizes.push(outputs.length);
          return new Float32Array(outputs.length).fill(1.0);
        },
      });
      await trainer.train(dataset as any);
      if (options.verifyFinalCheckpoint) {
        const finalDir = join(outputDir, 'final');
        expect(existsSync(join(finalDir, 'tokenizer.json'))).toBe(true);
        const reloaded = (await loadModel(finalDir)) as TrainableModel;
        const reloadedTrainer = createTrainer(reloaded, {
          modelName: 'qwen3-0.6b',
          groupSize: 1,
          maxCompletionLength: 2,
        });
        const sample = await reloadedTrainer.generateBatch([[{ role: 'user', content: 'Hi' }]]);
        expect(sample.completionTexts).toHaveLength(1);
      }
      return rewardCallSizes;
    } finally {
      rmSync(outputDir, { recursive: true, force: true });
    }
  }

  describe('Basic Training Loop', () => {
    it('runs one step per batch', async () => {
      const sizes = await runTrain({ numEpochs: 1, batchSize: 2, groupSize: 2, verifyFinalCheckpoint: true }, [
        example('Test 1'),
        example('Test 2'),
      ]);
      expect(sizes).toEqual([4]);
    });

    it('iterates through multiple epochs', async () => {
      const sizes = await runTrain({ numEpochs: 2, batchSize: 1, groupSize: 2 }, [example('Test')]);
      expect(sizes).toEqual([2, 2]);
    });

    it('fills a trailing partial batch', async () => {
      const sizes = await runTrain({ numEpochs: 1, batchSize: 2, groupSize: 2 }, [
        example('Test 1'),
        example('Test 2'),
        example('Test 3'),
      ]);
      expect(sizes).toEqual([4, 2]);
    });

    it('clears the processed-batch set at an epoch boundary', async () => {
      const sizes = await runTrain({ numEpochs: 2, batchSize: 2, groupSize: 2 }, [
        example('A'),
        example('B'),
        example('C'),
        example('D'),
      ]);
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
