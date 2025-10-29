/**
 * Test utilities for creating temporary models with random weights
 */

import { mkdtempSync, rmSync, copyFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Qwen3Model, type Qwen3Config } from '../packages/training/src/index';

/**
 * Tiny test configuration for fast model creation
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
 * Create a temporary model with random weights
 *
 * @param config - Model configuration (defaults to TINY_TEST_CONFIG)
 * @returns Object with model path and cleanup function
 */
export async function createTempModel(config: Qwen3Config = TINY_TEST_CONFIG): Promise<{
  modelPath: string;
  cleanup: () => void;
}> {
  // Create temporary directory
  const tempDir = mkdtempSync(join(tmpdir(), 'mlx-test-model-'));

  // Create model with random weights
  const model = new Qwen3Model(config);

  // Save to temp directory
  await model.saveModel(tempDir);

  // Copy tokenizer to temp directory
  // Try multiple possible tokenizer locations
  const possibleTokenizerPaths = [
    join(process.cwd(), '.cache/models/qwen3-0.6b-mlx-bf16/tokenizer.json'),
    join(process.cwd(), '.cache/models/qwen3-0.6b/tokenizer.json'),
    join(process.cwd(), '.cache/assets/tokenizers/qwen3_tokenizer.json'),
  ];

  let tokenizerCopied = false;
  for (const tokenizerPath of possibleTokenizerPaths) {
    if (existsSync(tokenizerPath)) {
      copyFileSync(tokenizerPath, join(tempDir, 'tokenizer.json'));
      tokenizerCopied = true;
      break;
    }
  }

  if (!tokenizerCopied) {
    throw new Error(
      'Could not find tokenizer file. Please ensure a Qwen3 model is downloaded with: yarn download:qwen3',
    );
  }

  return {
    modelPath: tempDir,
    cleanup: () => {
      try {
        rmSync(tempDir, { recursive: true, force: true });
      } catch (error) {
        console.warn(`Failed to cleanup temp model at ${tempDir}:`, error);
      }
    },
  };
}
