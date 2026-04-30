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
  // NOTE: `prompt_cache_key` is intentionally NOT advertised on this
  // endpoint. KV-cache reuse on `/v1/messages` is delivered via the
  // server-side `getOrCreateWarmAny` warm-slot mechanism keyed on the
  // mapped system/instructions string and a per-model sentinel id —
  // see the block comment on `endpoints/messages.ts`. The
  // `prompt_cache_key` field is therefore unnecessary on this surface
  // and re-adding it without a per-key tier-2 path on the handler
  // would be a no-op that silently misleads clients. The equivalent
  // field on `/v1/responses` is still honoured for that endpoint's
  // tier-2 lookup.
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

/**
 * Anthropic Messages API `usage` block.
 *
 * Per the Anthropic spec the cache accounting fields are OPTIONAL and
 * carry distinct semantics that drive client-side cost / billing
 * displays (Claude Code reads `cache_read_input_tokens` directly):
 *
 *   * `input_tokens` — prompt tokens that were processed at full cost
 *     this turn. When a cached prefix is reused, this MUST be reduced
 *     to the unsuffixed remainder (`promptTokens - cachedTokens`) so
 *     the client doesn't double-count the cached prefix.
 *   * `cache_read_input_tokens` — prompt tokens served from a cached
 *     prefix on this turn. Emitted only when reuse genuinely happened
 *     (`cachedTokens > 0`); omitted on a cold turn so the wire matches
 *     other Anthropic-compatible servers that elide the field on cache
 *     misses.
 *   * `cache_creation_input_tokens` — prompt tokens written to a NEW
 *     cache entry. Currently always omitted: this server's KV reuse
 *     is implicit and has no `cache_control` breakpoints, so a client
 *     that did not request explicit caching should never see a
 *     non-zero creation count.
 *
 * The `time_to_first_token_ms`, `prefill_tokens_per_second`, and
 * `decode_tokens_per_second` fields are NON-Anthropic extension
 * fields surfaced for the launcher's verbose log
 * (`requests.ndjson`) so per-turn decode-rate / TTFT / prefill-rate
 * telemetry rides the same response envelope. They are emitted only
 * when the underlying native dispatch produced a finite, positive
 * value — missing or non-finite metrics are elided rather than
 * surfaced as zero / null. Anthropic-compatible clients (Claude
 * Code, official Anthropic SDKs) ignore unknown fields, so the
 * extension is wire-safe — it parallels how `cache_read_input_tokens`
 * is treated above.
 */
export interface AnthropicUsage {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
  /** Server-extension: time-to-first-token in milliseconds. */
  time_to_first_token_ms?: number;
  /** Server-extension: prompt-token throughput during prefill. */
  prefill_tokens_per_second?: number;
  /** Server-extension: generated-token throughput during decode. */
  decode_tokens_per_second?: number;
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
  /**
   * Streaming `message_delta` carries the SAME cache-accounting
   * semantics as the non-streaming response `usage` block (see
   * `AnthropicUsage` above) — `input_tokens` MUST be net of any reused
   * prefix, and `cache_read_input_tokens` is emitted only on a true
   * reuse turn.
   *
   * `time_to_first_token_ms`, `prefill_tokens_per_second`, and
   * `decode_tokens_per_second` are server-extension fields (not in
   * Anthropic's spec) surfaced for the launcher's verbose log; see
   * the docstring on `AnthropicUsage`. Clients that do not recognize
   * them ignore them.
   */
  usage: {
    input_tokens?: number;
    output_tokens: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
    /** Server-extension: time-to-first-token in milliseconds. */
    time_to_first_token_ms?: number;
    /** Server-extension: prompt-token throughput during prefill. */
    prefill_tokens_per_second?: number;
    /** Server-extension: generated-token throughput during decode. */
    decode_tokens_per_second?: number;
  };
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
