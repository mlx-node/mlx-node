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

// Tool calling types (for tool-use training)
export type { ToolDefinition, FunctionDefinition, FunctionParameters } from '@mlx-node/core';

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
export { computeAdvantages, computeEntropy, getHighEntropyMask } from '@mlx-node/core';

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
  DEFAULT_GRPO_CONFIG,
  createRewardRegistry,
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
  type PromptChoice,
  type PromptOptions,
} from './trainers/training-logger';

// Entropy configuration
export { type EntropyFilteringConfig, DEFAULT_ENTROPY_CONFIG } from './trainers/grpo-entropy';

// SFT Trainer
export {
  SFTTrainer,
  SftTrainingEngine,
  type SFTTrainStepResult,
  type SFTTrainingState,
  type SftEngineConfig,
  type SftStepMetrics,
  type SftEpochMetrics,
} from './trainers/sft-trainer';

export {
  type SFTTrainerConfig,
  SFTConfigError,
  getDefaultSFTConfig,
  mergeSFTConfig,
  loadSFTTomlConfig,
  applySFTOverrides,
  DEFAULT_SFT_CONFIG,
} from './trainers/sft-config';

// Data
export * from './data/dataset';
export {
  SFTDataset,
  loadSFTDataset,
  createSFTDataset,
  type SFTExample,
  type SFTPromptCompletionExample,
  type SFTConversationExample,
  type SFTBatch,
  type SFTDatasetConfig,
} from './data/sft-dataset';

// Utils
export * from './utils/xml-parser';

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
