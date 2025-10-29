/**
 * Qwen3 Model Configurations and Type Definitions
 *
 * This module provides:
 * - Default configurations for common Qwen3 model sizes
 * - Type re-exports from Rust with enhanced documentation
 * - Helper functions for config management
 */

import type {
  Qwen3Config as RustQwen3Config,
  GenerationConfig as RustGenerationConfig,
  GenerationResult as RustGenerationResult,
} from '@mlx-node/core';

/**
 * Configuration for Qwen3 models
 *
 * All fields are required when creating a model directly.
 * Use QWEN3_CONFIGS for pre-configured model sizes.
 */
export type Qwen3Config = RustQwen3Config;

/**
 * Configuration for text generation
 *
 * Controls sampling behavior, temperature, and stopping criteria.
 */
export type GenerationConfig = RustGenerationConfig;

/**
 * Result from text generation with detailed metadata
 *
 * Includes generated tokens, log probabilities, finish reason, and token count.
 */
export type GenerationResult = RustGenerationResult;

/**
 * Default configurations for common Qwen3 models
 *
 * Includes optimized hyperparameters for:
 * - qwen3-0.6b: Smallest model (1024 hidden size, 28 layers)
 * - qwen3-1.8b: Medium model (1536 hidden size, 28 layers)
 * - qwen3-7b: Large model (3072 hidden size, 32 layers)
 */
export const QWEN3_CONFIGS: { [key: string]: Qwen3Config } = {
  'qwen3-0.6b': {
    vocabSize: 151936,
    hiddenSize: 1024,
    numLayers: 28,
    numHeads: 16,
    numKvHeads: 8, // GQA with 2:1 ratio
    headDim: 64, // hiddenSize / numHeads = 1024 / 16 = 64
    intermediateSize: 3072,
    rmsNormEps: 1e-6,
    ropeTheta: 1000000.0,
    maxPositionEmbeddings: 40960,
    useQkNorm: true, // Qwen3 always uses QK normalization (core feature)
    tieWordEmbeddings: true,
    padTokenId: 151643,
    eosTokenId: 151645,
    bosTokenId: 151643,
  },
  'qwen3-1.8b': {
    vocabSize: 151936,
    hiddenSize: 1536,
    numLayers: 28,
    numHeads: 12,
    numKvHeads: 2, // GQA with 6:1 ratio
    headDim: 128, // hiddenSize / numHeads = 1536 / 12 = 128
    intermediateSize: 8960,
    rmsNormEps: 1e-6,
    ropeTheta: 1000000.0,
    maxPositionEmbeddings: 131072,
    useQkNorm: true, // Qwen3 always uses QK normalization (core feature)
    tieWordEmbeddings: false,
    padTokenId: 151643,
    eosTokenId: 151645,
    bosTokenId: 151643,
  },
  'qwen3-7b': {
    vocabSize: 151936,
    hiddenSize: 3072,
    numLayers: 32,
    numHeads: 24,
    numKvHeads: 4, // GQA with 6:1 ratio
    headDim: 128, // hiddenSize / numHeads = 3072 / 24 = 128
    intermediateSize: 18944,
    rmsNormEps: 1e-6,
    ropeTheta: 1000000.0,
    maxPositionEmbeddings: 131072,
    useQkNorm: true, // Qwen3 always uses QK normalization (core feature)
    tieWordEmbeddings: false,
    padTokenId: 151643,
    eosTokenId: 151645,
    bosTokenId: 151643,
  },
};

/**
 * Get a Qwen3 configuration by name
 *
 * @param name - Model name (e.g., "qwen3-0.6b", "qwen3-1.8b", "qwen3-7b")
 * @returns Model configuration
 * @throws Error if model name is not recognized
 */
export function getQwen3Config(name: string): Qwen3Config {
  const config = QWEN3_CONFIGS[name];
  if (!config) {
    throw new Error(`Unknown model configuration: ${name}. Available models: ${Object.keys(QWEN3_CONFIGS).join(', ')}`);
  }
  return config;
}
