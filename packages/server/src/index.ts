/**
 * @mlx-node/server -- OpenAI Responses API server for MLX models
 *
 * Exposes loaded models via `POST /v1/responses` (OpenAI Responses API)
 * and `GET /v1/models`. Supports both streaming (SSE) and non-streaming
 * response modes.
 *
 * @example
 * ```typescript
 * import { createServer } from '@mlx-node/server';
 * import { Qwen35Model } from '@mlx-node/lm';
 *
 * const { registry, close } = await createServer({ port: 8080 });
 * const model = await Qwen35Model.load('./models/qwen3.5-3b');
 * registry.register('qwen3.5-3b', model);
 * ```
 *
 * @example
 * ```typescript
 * // Composable handler (no server lifecycle)
 * import { createHandler, ModelRegistry } from '@mlx-node/server';
 * import http from 'node:http';
 *
 * const registry = new ModelRegistry();
 * const handler = createHandler(registry);
 * const server = http.createServer(handler);
 * server.listen(3000);
 * ```
 */

// Server lifecycle
export { createServer } from './server.js';
export type { ServerConfig, ServerInstance } from './server.js';

// Composable handler
export { createHandler } from './handler.js';
export type { HandlerOptions } from './handler.js';

// Model registry
export { ModelRegistry } from './registry.js';
export type { ServableModel, ModelEntry } from './registry.js';

// Types (Responses API)
export type {
  ResponsesAPIRequest,
  ResponseObject,
  ResponseUsage,
  ResponseError,
  InputItem,
  InputMessage,
  InputFunctionCall,
  InputFunctionCallOutput,
  OutputItem,
  MessageOutputItem,
  ReasoningOutputItem,
  FunctionCallOutputItem,
  OutputTextPart,
  SummaryTextPart,
  ResponsesToolDefinition,
  ContentPart,
  InputTextPart,
  StreamEvent,
} from './types.js';

// Streaming utilities
export { writeSSEEvent, beginSSE, endSSE } from './streaming.js';
