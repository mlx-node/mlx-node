import type { Qwen3Model, Qwen35Model, Qwen35MoeModel } from '@mlx-node/core';

/**
 * Union of all model classes that can be used with training engines.
 * Uses the native (core) types so trainers can pass instances directly
 * to Rust engine factory methods without type conflicts.
 */
export type TrainableModel = Qwen3Model | Qwen35Model | Qwen35MoeModel;
