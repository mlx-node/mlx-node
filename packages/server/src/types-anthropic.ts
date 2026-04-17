/** Anthropic Messages API types: request/response shapes and SSE streaming events for POST /v1/messages. */

export interface AnthropicTextContentBlock {
  type: 'text';
  text: string;
}

export interface AnthropicImageContentBlock {
  type: 'image';
  source: {
    type: 'base64';
    media_type: string;
    data: string;
  };
}

export interface AnthropicToolResultContentBlock {
  type: 'tool_result';
  tool_use_id: string;
  /** May be a string, text-block array, or mix of text and image blocks. Image-mixed shapes are rejected by the mapper; see `resolveToolResultContent`. */
  content?: string | (AnthropicTextContentBlock | AnthropicImageContentBlock)[];
  is_error?: boolean;
}

export interface AnthropicToolUseContentBlock {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface AnthropicThinkingContentBlock {
  type: 'thinking';
  thinking: string;
}

export type AnthropicContentBlock =
  | AnthropicTextContentBlock
  | AnthropicImageContentBlock
  | AnthropicToolResultContentBlock
  | AnthropicToolUseContentBlock
  | AnthropicThinkingContentBlock;

// ---------------------------------------------------------------------------
// System block
// ---------------------------------------------------------------------------

export interface SystemBlock {
  type: 'text';
  text: string;
  cache_control?: { type: string };
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string | AnthropicContentBlock[];
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export interface AnthropicToolDefinition {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
}

export interface AnthropicToolChoice {
  type: 'auto' | 'any' | 'tool';
  name?: string;
  disable_parallel_tool_use?: boolean;
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

export interface AnthropicMessagesRequest {
  model: string;
  messages: AnthropicMessage[];
  max_tokens: number;
  system?: string | SystemBlock[];
  temperature?: number;
  top_p?: number;
  top_k?: number;
  tools?: AnthropicToolDefinition[];
  tool_choice?: AnthropicToolChoice;
  stream?: boolean;
  stop_sequences?: string[];
  metadata?: { user_id?: string };
}

// ---------------------------------------------------------------------------
// Response content blocks
// ---------------------------------------------------------------------------

export interface AnthropicResponseTextBlock {
  type: 'text';
  text: string;
}

export interface AnthropicResponseThinkingBlock {
  type: 'thinking';
  thinking: string;
}

export interface AnthropicResponseToolUseBlock {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export type AnthropicResponseContent =
  | AnthropicResponseTextBlock
  | AnthropicResponseThinkingBlock
  | AnthropicResponseToolUseBlock;

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

export interface AnthropicUsage {
  input_tokens: number;
  output_tokens: number;
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

export interface AnthropicMessagesResponse {
  id: string;
  type: 'message';
  role: 'assistant';
  model: string;
  content: AnthropicResponseContent[];
  stop_reason: 'end_turn' | 'max_tokens' | 'stop_sequence' | 'tool_use' | null;
  stop_sequence: string | null;
  usage: AnthropicUsage;
}

// ---------------------------------------------------------------------------
// Streaming delta types
// ---------------------------------------------------------------------------

export interface AnthropicTextDelta {
  type: 'text_delta';
  text: string;
}

export interface AnthropicThinkingDelta {
  type: 'thinking_delta';
  thinking: string;
}

export interface AnthropicInputJsonDelta {
  type: 'input_json_delta';
  partial_json: string;
}

export type AnthropicDelta = AnthropicTextDelta | AnthropicThinkingDelta | AnthropicInputJsonDelta;

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

export interface AnthropicMessageStartEvent {
  type: 'message_start';
  message: AnthropicMessagesResponse;
}

export interface AnthropicContentBlockStartEvent {
  type: 'content_block_start';
  index: number;
  content_block: AnthropicResponseContent;
}

export interface AnthropicContentBlockDeltaEvent {
  type: 'content_block_delta';
  index: number;
  delta: AnthropicDelta;
}

export interface AnthropicContentBlockStopEvent {
  type: 'content_block_stop';
  index: number;
}

export interface AnthropicMessageDeltaEvent {
  type: 'message_delta';
  delta: {
    stop_reason: string | null;
    stop_sequence: string | null;
  };
  usage: { input_tokens?: number; output_tokens: number };
}

export interface AnthropicMessageStopEvent {
  type: 'message_stop';
}

export type AnthropicStreamEvent =
  | AnthropicMessageStartEvent
  | AnthropicContentBlockStartEvent
  | AnthropicContentBlockDeltaEvent
  | AnthropicContentBlockStopEvent
  | AnthropicMessageDeltaEvent
  | AnthropicMessageStopEvent;
