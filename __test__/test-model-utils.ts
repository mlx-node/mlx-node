/**
 * Test utilities for creating temporary models with random weights.
 *
 * All temp-model helpers go through the Phase 1 NAPI checkpoint builders
 * (`createRandomQwen3Checkpoint`, `createRandomQwen35Checkpoint`,
 * `createRandomQwen35MoeCheckpoint`) instead of holding a JS-side model
 * instance. This keeps test fixtures aligned with the Phase 6 goal of
 * removing the direct `Qwen3Model` / `Qwen35Model` / `Qwen35MoeModel`
 * constructors from the public NAPI surface.
 */

import { copyFileSync, existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  createRandomQwen35Checkpoint,
  createRandomQwen35MoeCheckpoint,
  createRandomQwen3Checkpoint,
} from '@mlx-node/core';
import type { Qwen35Config, Qwen35MoeConfig, Qwen3Config } from '@mlx-node/lm';

/**
 * Tiny Qwen3 test configuration for fast model creation.
 */
export const TINY_TEST_CONFIG: Qwen3Config = {
  vocabSize: 1000,
  hiddenSize: 64,
  numLayers: 2,
  numHeads: 4,
  numKvHeads: 2,
  headDim: 16,
  intermediateSize: 128,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  maxPositionEmbeddings: 512,
  useQkNorm: false,
  tieWordEmbeddings: false,
  padTokenId: 0,
  eosTokenId: 1,
  bosTokenId: 0,
};

/**
 * Tiny Qwen3.5 dense test configuration — mirrors the inline config used in
 * `__test__/trainers/grpo-qwen35.test.ts`. Kept small so construction stays
 * under a few seconds.
 */
export const TINY_QWEN35_CONFIG: Qwen35Config = {
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

/**
 * Tiny Qwen3.5 MoE test configuration — extends the dense config with
 * minimal expert routing.
 */
export const TINY_QWEN35_MOE_CONFIG: Qwen35MoeConfig = {
  ...TINY_QWEN35_CONFIG,
  numExperts: 4,
  numExpertsPerTok: 2,
  decoderSparseStep: 1,
  sharedExpertIntermediateSize: 128,
  moeIntermediateSize: 64,
  normTopkProb: true,
};

/**
 * Options accepted by every `createTemp*Model` helper.
 */
export interface CreateTempModelOptions {
  /**
   * When `true`, skip copying a `tokenizer.json` into the temp directory.
   * Useful for unit tests that only validate config parsing / parameter
   * handling and never actually run the tokenizer. Defaults to `false`,
   * which preserves the original behavior: a tokenizer is required and the
   * helper throws if none can be located in the local `.cache`.
   */
  skipTokenizer?: boolean;
}

/**
 * Return shape of every `createTemp*Model` helper.
 */
export interface TempModel {
  modelPath: string;
  cleanup: () => void;
}

/**
 * Locate a Qwen tokenizer.json in the local `.cache`. Prefers Qwen3.5
 * tokenizers first (which are a superset), falling back to the Qwen3
 * tokenizer (compatible vocab) so tests still run on machines that only
 * have the smaller model downloaded. Canonical tokenizer search used by
 * all `createTemp*Model` helpers here. (A duplicate still exists in
 * `__test__/trainers/grpo-qwen35.test.ts` and will be removed in Phase 3.)
 */
export function findTokenizerPath(): string {
  const candidates = [
    join(process.cwd(), '.cache/models/qwen3.5-0.8b-mlx-bf16/tokenizer.json'),
    join(process.cwd(), '.cache/models/qwen3.5-0.8b/tokenizer.json'),
    join(process.cwd(), '.cache/models/qwen3-0.6b-mlx-bf16/tokenizer.json'),
    join(process.cwd(), '.cache/models/qwen3-0.6b/tokenizer.json'),
    join(process.cwd(), '.cache/assets/tokenizers/qwen3_tokenizer.json'),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  const searched = candidates.map((c) => `  - ${c}`).join('\n');
  throw new Error(
    `Could not find a Qwen tokenizer.json. Searched:\n${searched}\nDownload a Qwen model first, e.g.:\n  yarn mlx download model -m Qwen/Qwen3-0.6B -o .cache/models/qwen3-0.6b`,
  );
}

function makeCleanup(tempDir: string, label: string): () => void {
  return () => {
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch (error) {
      console.warn(`Failed to cleanup temp ${label} model at ${tempDir}:`, error);
    }
  };
}

function maybeCopyTokenizer(tempDir: string, options: CreateTempModelOptions | undefined): void {
  if (options?.skipTokenizer) return;
  copyFileSync(findTokenizerPath(), join(tempDir, 'tokenizer.json'));
}

/**
 * Create a temporary Qwen3 model with random weights via
 * `createRandomQwen3Checkpoint`. The JS side never holds a `Qwen3Model`
 * instance — the NAPI helper builds, saves, and drops the model internally.
 *
 * @param config - Model configuration (defaults to `TINY_TEST_CONFIG`).
 * @param options - Optional behavior flags (see `CreateTempModelOptions`).
 * @returns Object with model path and cleanup function.
 */
export async function createTempModel(
  config: Qwen3Config = TINY_TEST_CONFIG,
  options?: CreateTempModelOptions,
): Promise<TempModel> {
  const tempDir = mkdtempSync(join(tmpdir(), 'mlx-test-model-'));
  await createRandomQwen3Checkpoint(config, tempDir);
  maybeCopyTokenizer(tempDir, options);
  return {
    modelPath: tempDir,
    cleanup: makeCleanup(tempDir, 'Qwen3'),
  };
}

/**
 * Create a temporary Qwen3.5 dense model with random weights via
 * `createRandomQwen35Checkpoint`.
 *
 * @param config - Model configuration (defaults to `TINY_QWEN35_CONFIG`).
 * @param options - Optional behavior flags (see `CreateTempModelOptions`).
 * @returns Object with model path and cleanup function.
 */
export async function createTempQwen35Model(
  config: Qwen35Config = TINY_QWEN35_CONFIG,
  options?: CreateTempModelOptions,
): Promise<TempModel> {
  const tempDir = mkdtempSync(join(tmpdir(), 'mlx-test-qwen35-model-'));
  await createRandomQwen35Checkpoint(config, tempDir);
  maybeCopyTokenizer(tempDir, options);
  return {
    modelPath: tempDir,
    cleanup: makeCleanup(tempDir, 'Qwen3.5'),
  };
}

/**
 * Create a temporary Qwen3.5 MoE model with random weights via
 * `createRandomQwen35MoeCheckpoint`.
 *
 * @param config - Model configuration (defaults to `TINY_QWEN35_MOE_CONFIG`).
 * @param options - Optional behavior flags (see `CreateTempModelOptions`).
 * @returns Object with model path and cleanup function.
 */
export async function createTempQwen35MoeModel(
  config: Qwen35MoeConfig = TINY_QWEN35_MOE_CONFIG,
  options?: CreateTempModelOptions,
): Promise<TempModel> {
  const tempDir = mkdtempSync(join(tmpdir(), 'mlx-test-qwen35-moe-model-'));
  await createRandomQwen35MoeCheckpoint(config, tempDir);
  maybeCopyTokenizer(tempDir, options);
  return {
    modelPath: tempDir,
    cleanup: makeCleanup(tempDir, 'Qwen3.5 MoE'),
  };
}
