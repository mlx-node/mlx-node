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
 * import { GRPOTrainer, GRPOConfig, loadLocalGsm8kDataset } from '@mlx-node/trl';
 * import { loadModel } from '@mlx-node/lm';
 *
 * const model = await loadModel('./models/qwen3-0.6b');
 * const trainer = await GRPOTrainer.create({ modelPath: './models/qwen3-0.6b' });
 * ```
 */

// =============================================================================
// Re-exports from @mlx-node/core for training
// =============================================================================

// Tool calling types (for tool-use training)
export type { ToolDefinition, FunctionDefinition, FunctionParameters } from '@mlx-node/core';

// Note: MxArray, convertModel, convertParquetToJsonl are available from @mlx-node/core directly.
// They are not re-exported here to avoid multiple-package export confusion.

// =============================================================================
// TRL-specific exports
// =============================================================================

// Trainers
export {
  GRPOTrainer,
  type GRPOTrainerConfig,
  DEFAULT_GRPO_CONFIG,
  createRewardRegistry,
  computeDatasetHash,
  RewardTimeoutError,
  type GenerateBatchResult,
  type TrainStepMetrics,
  type TrainingMetrics,
  type TrainingState,
  type DatasetMetadata,
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
export {
  loadLocalGsm8kDataset,
  LocalGsm8kDatasetLoader,
  createDatasetExample,
  extractGsm8kAnswer,
  validateDatasetExample,
  type LocalDatasetOptions,
} from './data/dataset';
export {
  SFTDataset,
  loadSFTDataset,
  createSFTDataset,
  type SFTExample,
  type SFTPromptCompletionExample,
  type SFTConversationExample,
  type SFTBatch,
  type SFTDatasetConfig,
  type SpecialTokenIds,
} from './data/sft-dataset';

// Utils
export { parseXmlCot, extractXmlAnswer, extractXmlReasoning, extractHashAnswer } from './utils/xml-parser';
export {
  validatePathContainment,
  resolveAndValidatePath,
  getAllowedRoot,
  PathTraversalError,
  type PathValidationOptions,
} from './utils/path-security';

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
