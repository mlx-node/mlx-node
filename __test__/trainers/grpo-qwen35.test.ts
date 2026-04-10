/**
 * GRPO trainer smoke tests for Qwen3.5 Dense and Qwen3.5 MoE.
 *
 * Exercises the dedicated model-thread training paths:
 * - `GrpoTrainingEngine::fromQwen35` / `fromQwen35Moe`
 * - `InitTraining` / `GenerateForTraining` / `TrainStepGRPO` command handlers
 *
 * Each variant: create a temp random-weight model on disk, load it via
 * `loadModel`, construct a `GRPOTrainer`, run a single `trainStep`, and
 * assert the returned loss is a finite number.
 */

import { Qwen35Model, Qwen35MoeModel } from '@mlx-node/core';
import { loadModel } from '@mlx-node/lm';
import { GRPOTrainer, type RewardOutput } from '@mlx-node/trl';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { createTempQwen35Model, createTempQwen35MoeModel, type TempModel } from '../test-model-utils';

const constantReward = (outputs: RewardOutput[]): Float32Array => Float32Array.from(outputs.map(() => 0.5));

describe.sequential('GRPOTrainer - Qwen3.5 Dense smoke', () => {
  let tempModel: TempModel;

  beforeAll(async () => {
    tempModel = await createTempQwen35Model();
  });

  afterAll(() => {
    tempModel?.cleanup();
  });

  it('runs one train step and returns finite loss', async () => {
    const loaded = await loadModel(tempModel.modelPath);
    expect(loaded).toBeInstanceOf(Qwen35Model);
    const model = loaded as unknown as Qwen35Model;

    const trainer = new GRPOTrainer(model, {
      modelName: 'qwen3_5-tiny-dense',
      groupSize: 2,
      maxCompletionLength: 4,
      temperature: 0.8,
      topP: 0.95,
      learningRate: 1e-5,
      rewardFunction: constantReward,
      logConsole: false,
    });

    const metrics = await trainer.trainStep([[{ role: 'user', content: 'hi' }]]);

    expect(metrics).toBeDefined();
    expect(typeof metrics.loss).toBe('number');
    expect(Number.isFinite(metrics.loss)).toBe(true);
    expect(typeof metrics.meanAdvantage).toBe('number');
    expect(metrics.totalTokens).toBeGreaterThan(0);
    expect(metrics.step).toBe(1);
  });
});

describe.sequential('GRPOTrainer - Qwen3.5 MoE smoke', () => {
  let tempModel: TempModel;

  beforeAll(async () => {
    tempModel = await createTempQwen35MoeModel();
  });

  afterAll(() => {
    tempModel?.cleanup();
  });

  it('runs one train step and returns finite loss', async () => {
    const loaded = await loadModel(tempModel.modelPath);
    expect(loaded).toBeInstanceOf(Qwen35MoeModel);
    const model = loaded as unknown as Qwen35MoeModel;

    const trainer = new GRPOTrainer(model, {
      modelName: 'qwen3_5-tiny-moe',
      groupSize: 2,
      maxCompletionLength: 4,
      temperature: 0.8,
      topP: 0.95,
      learningRate: 1e-5,
      rewardFunction: constantReward,
      logConsole: false,
    });

    const metrics = await trainer.trainStep([[{ role: 'user', content: 'hi' }]]);

    expect(metrics).toBeDefined();
    expect(typeof metrics.loss).toBe('number');
    expect(Number.isFinite(metrics.loss)).toBe(true);
    expect(typeof metrics.meanAdvantage).toBe('number');
    expect(metrics.totalTokens).toBeGreaterThan(0);
    expect(metrics.step).toBe(1);
  });
});
