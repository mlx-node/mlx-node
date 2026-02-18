/**
 * Qwen3.5 Model Configurations and Type Definitions
 *
 * Supports both dense and MoE variants. MoE fields are optional -
 * when `numExperts` is undefined, the model uses dense MLP layers.
 */

import type {
  Qwen35Config as RustQwen35Config,
  Qwen35GenerationConfig as RustQwen35GenerationConfig,
  Qwen35GenerationResult as RustQwen35GenerationResult,
} from '@mlx-node/core';

export type Qwen35Config = RustQwen35Config;
export type Qwen35GenerationConfig = RustQwen35GenerationConfig;
export type Qwen35GenerationResult = RustQwen35GenerationResult;

/**
 * Default configurations for common Qwen3.5 models
 */
export const QWEN35_CONFIGS: { [key: string]: Qwen35Config } = {
  'qwen3.5-0.6b': {
    vocabSize: 151936,
    hiddenSize: 1024,
    numLayers: 28,
    numHeads: 16,
    numKvHeads: 8,
    intermediateSize: 3072,
    rmsNormEps: 1e-6,
    headDim: 64,
    tieWordEmbeddings: true,
    attentionBias: false,
    maxPositionEmbeddings: 131072,
    padTokenId: 151643,
    eosTokenId: 151645,
    bosTokenId: 151643,
    linearNumValueHeads: 64,
    linearNumKeyHeads: 16,
    linearKeyHeadDim: 192,
    linearValueHeadDim: 128,
    linearConvKernelDim: 4,
    fullAttentionInterval: 4,
    partialRotaryFactor: 0.25,
    ropeTheta: 100000.0,
  },
};

/**
 * Get a Qwen3.5 configuration by name
 *
 * @param name - Model name (e.g., "qwen3.5-0.6b")
 * @returns Model configuration
 * @throws Error if model name is not recognized
 */
export function getQwen35Config(name: string): Qwen35Config {
  const config = QWEN35_CONFIGS[name];
  if (!config) {
    throw new Error(`Unknown Qwen3.5 config: ${name}. Available: ${Object.keys(QWEN35_CONFIGS).join(', ')}`);
  }
  return config;
}
