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

// Memory management
export {
  getActiveMemory,
  getCacheMemory,
  getPeakMemory,
  resetPeakMemory,
  clearCache,
  synchronizeAndClearCache,
  heavyCleanup,
  getMemoryLimit,
  setMemoryLimit,
} from '@mlx-node/core';

// Types
export type { DType } from '@mlx-node/core';
export type { SamplingConfig, BatchGenerationConfig, BatchGenerationResult } from '@mlx-node/core';

// Model utilities (TypeScript-only)
export {
  type Qwen3Config,
  QWEN3_CONFIGS,
  type GenerationResult,
  type GenerationConfig,
  getQwen3Config,
} from './models/qwen3-configs';

export { ModelLoader } from './models/model-loader';
