/**
 * @mlx-node/lm - High-level inference API for MLX models
 *
 * This package provides everything needed for model loading and inference,
 * aligned with Python's mlx-lm library.
 *
 * @example
 * ```typescript
 * import { loadModel, Qwen3Model } from '@mlx-node/lm';
 *
 * const model = await loadModel('./models/qwen3-0.6b');
 * const result = await model.generate([{ role: 'user', content: 'Hello!' }]);
 * ```
 */

// Model classes (for inference)
export { Qwen3Tokenizer } from '@mlx-node/core';
export { Qwen3Model } from './stream.js';

// Gemma4 models
export { Gemma4Model, Gemma4Model as Gemma4_Model } from './stream.js';
export type { Gemma4Config } from '@mlx-node/core';

// Embedding models
export { HarrierModel } from '@mlx-node/core';
export type { HarrierConfig } from '@mlx-node/core';
export { Qwen35Model, Qwen35Model as Qwen3_5Model } from './stream.js';
export type { Qwen35Config, Qwen35GenerationConfig, Qwen35GenerationResult } from '@mlx-node/core';

// LFM2 models
export { Lfm2Model } from './stream.js';
export type { Lfm2Config } from '@mlx-node/core';
export { LFM2_CONFIGS, getLfm2Config } from './models/lfm2-configs.js';

// MoE variant
export { Qwen35MoeModel, Qwen35MoeModel as Qwen3_5MoeModel } from './stream.js';
export type { Qwen35MoeConfig, Qwen35MoeGenerationConfig, Qwen35MoeGenerationResult } from '@mlx-node/core';

// Note: Memory management is handled internally by Rust - not exposed to JS

// Types
export type { DType } from '@mlx-node/core';
export type { SamplingConfig, BatchGenerationResult } from '@mlx-node/core';

// Unified Chat API types (shared by Qwen3, Qwen3.5, Qwen3.5 MoE)
export type { ChatConfig, ChatResult, ChatMessage, ToolCallResult, PerformanceMetrics } from '@mlx-node/core';

// Streaming chat API
export type { ChatStreamDelta, ChatStreamFinal, ChatStreamEvent } from './stream.js';
export type { ChatStreamChunk, ChatStreamHandle } from '@mlx-node/core';
// Internal: exported for testing the callback-to-AsyncGenerator bridge
// Not part of the public API — may change without notice.
// `_runChatStream` is the generic adapter used by every model wrapper
// (and the VLM package's QianfanOCR wrapper) to turn a callback-based
// native stream into an `AsyncGenerator<ChatStreamEvent>`.
export { _runChatStream } from './stream.js';
// Cross-model chat session wrapper (see chat-session.ts for design notes).
// `SessionCapableModel` is the structural interface matched by every
// generative model wrapper and used as the upper-bound for
// `ChatSession<M>`; exported so the VLM wrapper can pin a compile-time
// conformance assertion.
export { ChatSession } from './chat-session.js';
export type { ChatSessionOptions, SendOptions, SessionCapableModel } from './chat-session.js';

// Model utilities (TypeScript-only)
export {
  type Qwen3Config,
  QWEN3_CONFIGS,
  type GenerationResult,
  type GenerationConfig,
  getQwen3Config,
} from './models/qwen3-configs.js';

// Model loading
export { loadModel, loadSession, detectModelType, type ModelType } from './models/model-loader.js';

// Interfaces
export type { TrainableModel, LoadableModel, EmbeddingModel } from './interfaces.js';

export { QWEN35_CONFIGS, getQwen35Config } from './models/qwen3_5-configs.js';

// Tool calling utilities
export * from './tools/index.js';

// Profiling API
export { enableProfiling, disableProfiling } from './profiling.js';
