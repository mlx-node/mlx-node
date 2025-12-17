/**
 * @mlx-node/trl - Training utilities for MLX models
 *
 * This package provides everything needed for training ML models,
 * aligned with Python's TRL (Transformer Reinforcement Learning) library.
 *
 * For model loading and inference, import from @mlx-node/lm.
 *
 * @example
 * ```typescript
 * import { GRPOTrainer, GRPOConfig, Adam, loadLocalGsm8kDataset } from '@mlx-node/trl';
 * import { ModelLoader } from '@mlx-node/lm';
 *
 * const model = await ModelLoader.loadPretrained('./models/qwen3-0.6b');
 * const trainer = await GRPOTrainer.create({ modelPath: './models/qwen3-0.6b' });
 * ```
 */

// =============================================================================
// Re-exports from @mlx-node/core for training
// =============================================================================

// Optimizers
export { Adam, AdamW, SGD, RMSprop, LRScheduler } from '@mlx-node/core';

// Gradient utilities
export { Gradients, GradientUtils, clipGradientsByGlobalNorm, clipGradientsByValue } from '@mlx-node/core';

// Core tensor (for custom rewards/models)
export { MxArray } from '@mlx-node/core';

// Low-level layers (for custom architectures)
export { Linear, RMSNorm, LayerNorm, Embedding, Activations, Losses } from '@mlx-node/core';

// Transformer components
export { Attention, FusedAttention, TransformerBlock, MLP, RoPE } from '@mlx-node/core';
export { KVCache, BatchKVCache, RotatingKVCache } from '@mlx-node/core';

// GRPO utilities
export { computeAdvantages, computeEntropy, getHighEntropyMask, selectiveLogSoftmax } from '@mlx-node/core';

// Utility functions
export { padSequences, padFloatSequences, createAttentionMaskForTransformer, PaddedSequences } from '@mlx-node/core';

// Model conversion
export { convertModel, convertParquetToJsonl } from '@mlx-node/core';
export type { ConversionOptions, ConversionResult } from '@mlx-node/core';

// =============================================================================
// TRL-specific exports
// =============================================================================

// Trainers
export {
  type MLXGRPOConfig,
  ConfigError,
  getDefaultConfig,
  mergeConfig,
  loadTomlConfig,
  applyOverrides,
} from './trainers/grpo-config';

export {
  GRPOTrainer,
  type GRPOTrainerConfig,
  type GRPOConfig,
  DEFAULT_GRPO_CONFIG,
  createRewardRegistry,
  type RewardFn,
  type GenerateBatchResult,
  type TrainStepMetrics,
  type TrainingMetrics,
  type TrainingState,
  // Re-export native types from trainer
  GrpoTrainingEngine,
  NativeRewardRegistry,
  type GrpoEngineConfig,
  type EngineStepMetrics,
  type EngineEpochMetrics,
  type BuiltinRewardConfig,
} from './trainers/grpo-trainer';

// Unified Training Logger (recommended)
export {
  TrainingLogger,
  createTrainingLogger,
  type TrainingLoggerConfig,
  type TrainingMetrics as TrainingLoggerMetrics,
  type GenerationSample,
  type TrainingConfigFields,
  type TuiMessage,
  type LogEvent,
} from './trainers/training-logger';

// Legacy Logger (deprecated - use TrainingLogger instead)
export {
  GRPOLogger,
  createLogger,
  MetricsAggregator,
  type LoggerConfig,
  type TrainingConfigFields as GRPOLoggerConfigFields,
  type LogEvent as GRPOLogEvent,
} from './trainers/grpo-logger';

// Entropy configuration
export { type EntropyFilteringConfig, DEFAULT_ENTROPY_CONFIG } from './trainers/grpo-entropy';

// Data
export * from './data/dataset';

// Utils
export * from './utils/xml-parser';

// Rewards
export * from './rewards';

// Types
export type {
  ChatRole,
  ChatMessage,
  CompletionMessage,
  Completion,
  DatasetSplit,
  DatasetExample,
  XmlParseResult,
  RewardComputationInput,
  PromptFormatterOptions,
  PromptTemplate,
  DatasetLoader,
  RewardFunction,
  PromptFormatter,
  // Reward function types
  CompletionInfo,
  RewardOutput,
} from './types';
