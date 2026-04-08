import type {
  Gemma4Model,
  HarrierModel,
  Qwen3Model,
  Qwen35Model,
  Qwen35MoeModel,
  QianfanOCRModel,
} from '@mlx-node/core';

/**
 * Union of all model classes that can be used with training engines.
 * Uses the native (core) types so trainers can pass instances directly
 * to Rust engine factory methods without type conflicts.
 */
export type TrainableModel = Qwen3Model | Qwen35Model | Qwen35MoeModel;

/**
 * Union of all embedding model classes.
 * These models encode text into fixed-size embedding vectors.
 */
export type EmbeddingModel = HarrierModel;

/**
 * Union of all model classes that loadModel can return.
 * Includes trainable models, inference-only models, and embedding models.
 */
export type LoadableModel = TrainableModel | QianfanOCRModel | EmbeddingModel | Gemma4Model;
