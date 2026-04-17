/**
 * @mlx-node/server -- OpenAI Responses + Anthropic Messages server for MLX models.
 *
 * Exposes loaded models via `POST /v1/responses`, `POST /v1/messages`, and
 * `GET /v1/models`, in both streaming (SSE) and non-streaming modes.
 */

export { createServer } from './server.js';
export type { ServerConfig, ServerInstance } from './server.js';

/**
 * Internal helpers re-exported for unit testing only. Not part of the
 * supported public API — names may change without notice.
 */
export { parseEnvSeconds as __parseEnvSeconds, parseEnvPositiveInt as __parseEnvPositiveInt } from './server.js';

export { createHandler } from './handler.js';
export type { HandlerOptions } from './handler.js';

export { ModelRegistry } from './registry.js';
export type { ServableModel, ModelEntry, ModelRegistryOptions } from './registry.js';

export { QueueFullError, SessionRegistry } from './session-registry.js';
export type { SessionLookupResult, SessionRegistryOptions } from './session-registry.js';

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

export type {
  AnthropicMessagesRequest,
  AnthropicMessagesResponse,
  AnthropicMessage,
  AnthropicContentBlock,
  AnthropicTextContentBlock,
  AnthropicImageContentBlock,
  AnthropicToolResultContentBlock,
  AnthropicToolUseContentBlock,
  AnthropicThinkingContentBlock,
  AnthropicToolDefinition,
  AnthropicToolChoice,
  AnthropicResponseContent,
  AnthropicResponseTextBlock,
  AnthropicResponseThinkingBlock,
  AnthropicResponseToolUseBlock,
  AnthropicUsage,
  AnthropicStreamEvent,
  AnthropicMessageStartEvent,
  AnthropicContentBlockStartEvent,
  AnthropicContentBlockDeltaEvent,
  AnthropicContentBlockStopEvent,
  AnthropicMessageDeltaEvent,
  AnthropicMessageStopEvent,
  AnthropicDelta,
  AnthropicTextDelta,
  AnthropicThinkingDelta,
  AnthropicInputJsonDelta,
  SystemBlock,
} from './types-anthropic.js';

export { writeSSEEvent, beginSSE, endSSE } from './streaming.js';
