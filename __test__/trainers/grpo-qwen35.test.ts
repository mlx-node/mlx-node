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

import { copyFileSync, existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { Qwen35Model, Qwen35MoeModel } from '@mlx-node/core';
import type { Qwen35Config, Qwen35MoeConfig } from '@mlx-node/core';
import { loadModel } from '@mlx-node/lm';
import { GRPOTrainer, type RewardOutput } from '@mlx-node/trl';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

// Tiny Qwen3.5 dense config — kept small so construction and a single train step
// stay under a few seconds. Mirrors the shape used in
// `__test__/models/qwen3_5-generation.test.ts`.
const TINY_DENSE_CONFIG: Qwen35Config = {
  vocabSize: 1000,
  hiddenSize: 128,
  numLayers: 4,
  numHeads: 4,
  numKvHeads: 2,
  intermediateSize: 256,
  rmsNormEps: 1e-6,
  headDim: 32,
  tieWordEmbeddings: true,
  attentionBias: false,
  maxPositionEmbeddings: 512,
  padTokenId: 0,
  eosTokenId: 1,
  bosTokenId: 0,
  linearNumValueHeads: 8,
  linearNumKeyHeads: 4,
  linearKeyHeadDim: 32,
  linearValueHeadDim: 16,
  linearConvKernelDim: 4,
  fullAttentionInterval: 4,
  partialRotaryFactor: 0.25,
  ropeTheta: 10000.0,
};

// Tiny Qwen3.5 MoE config — extends dense with minimal expert routing.
const TINY_MOE_CONFIG: Qwen35MoeConfig = {
  ...TINY_DENSE_CONFIG,
  numExperts: 4,
  numExpertsPerTok: 2,
  decoderSparseStep: 1,
  sharedExpertIntermediateSize: 128,
  moeIntermediateSize: 64,
  normTopkProb: true,
};

interface TempModel {
  modelPath: string;
  cleanup: () => void;
}

/**
 * Locate a Qwen3.5-compatible `tokenizer.json` in the local `.cache`.
 * Falls back to the Qwen3 tokenizer (compatible vocab) so tests still run
 * on machines that only have the smaller model downloaded.
 */
function findTokenizerPath(): string {
  const candidates = [
    join(process.cwd(), '.cache/models/qwen3.5-0.8b-mlx-bf16/tokenizer.json'),
    join(process.cwd(), '.cache/models/qwen3.5-0.8b/tokenizer.json'),
    join(process.cwd(), '.cache/models/qwen3-0.6b-mlx-bf16/tokenizer.json'),
    join(process.cwd(), '.cache/models/qwen3-0.6b/tokenizer.json'),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  throw new Error(
    'Could not find a Qwen tokenizer.json in .cache/models. ' +
      'Run `yarn download:qwen3` or download a Qwen3.5 model first.',
  );
}

async function createTempDenseModel(): Promise<TempModel> {
  const tempDir = mkdtempSync(join(tmpdir(), 'mlx-test-qwen35-dense-'));
  const model = new Qwen35Model(TINY_DENSE_CONFIG);
  await model.saveModel(tempDir);
  copyFileSync(findTokenizerPath(), join(tempDir, 'tokenizer.json'));
  return {
    modelPath: tempDir,
    cleanup: () => {
      try {
        rmSync(tempDir, { recursive: true, force: true });
      } catch (err) {
        console.warn(`Failed to cleanup temp dense model at ${tempDir}:`, err);
      }
    },
  };
}

async function createTempMoeModel(): Promise<TempModel> {
  const tempDir = mkdtempSync(join(tmpdir(), 'mlx-test-qwen35-moe-'));
  const model = new Qwen35MoeModel(TINY_MOE_CONFIG);
  await model.saveModel(tempDir);
  copyFileSync(findTokenizerPath(), join(tempDir, 'tokenizer.json'));
  return {
    modelPath: tempDir,
    cleanup: () => {
      try {
        rmSync(tempDir, { recursive: true, force: true });
      } catch (err) {
        console.warn(`Failed to cleanup temp MoE model at ${tempDir}:`, err);
      }
    },
  };
}

const constantReward = (outputs: RewardOutput[]): Float32Array => Float32Array.from(outputs.map(() => 0.5));

describe.sequential('GRPOTrainer - Qwen3.5 Dense smoke', () => {
  let tempModel: TempModel;

  beforeAll(async () => {
    tempModel = await createTempDenseModel();
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
    tempModel = await createTempMoeModel();
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
