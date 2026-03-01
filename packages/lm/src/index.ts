/**
 * @mlx-node/lm - High-level inference API for MLX models
 *
 * This package provides everything needed for model loading and inference,
 * aligned with Python's mlx-lm library.
 *
 * @example
 * ```typescript
 * import { Qwen3Model, ModelLoader, QWEN3_CONFIGS } from '@mlx-node/lm';
 *
 * const model = await ModelLoader.loadPretrained('./models/qwen3-0.6b');
 * const result = await model.generate([{ role: 'user', content: 'Hello!' }]);
 * ```
 */

// Model classes (for inference)
export { Qwen3Model, Qwen3Tokenizer } from '@mlx-node/core';
export { Qwen35Model, Qwen35Model as Qwen3_5Model } from '@mlx-node/core';
export type {
  Qwen35Config,
  Qwen35ChatConfig,
  Qwen35ChatResult,
  Qwen35GenerationConfig,
  Qwen35GenerationResult,
} from '@mlx-node/core';

// MoE variant
export { Qwen35MoeModel, Qwen35MoeModel as Qwen3_5MoeModel } from '@mlx-node/core';
export type {
  Qwen35MoeConfig,
  Qwen35MoeChatConfig,
  Qwen35MoeChatResult,
  Qwen35MoeGenerationConfig,
  Qwen35MoeGenerationResult,
} from '@mlx-node/core';

// Note: Memory management is handled internally by Rust - not exposed to JS

// Types
export type { DType } from '@mlx-node/core';
export type { SamplingConfig, BatchGenerationResult } from '@mlx-node/core';

// Chat API types from core (for model.chat() API)
export type { ToolCallResult, ChatResult, ChatConfig, ChatMessage } from '@mlx-node/core';

// Model utilities (TypeScript-only)
export {
  type Qwen3Config,
  QWEN3_CONFIGS,
  type GenerationResult,
  type GenerationConfig,
  getQwen3Config,
} from './models/qwen3-configs';

export { ModelLoader } from './models/model-loader';

export { QWEN35_CONFIGS, getQwen35Config } from './models/qwen3_5-configs';

// Tool calling utilities
export * from './tools';
