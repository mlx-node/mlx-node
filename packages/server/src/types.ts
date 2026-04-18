/** OpenAI Responses API types: request/response shapes and SSE streaming events for POST /v1/responses. */

export interface InputTextPart {
  type: 'input_text';
  text: string;
}

/**
 * An image attached to a user message. `image_url` may be an `http(s)://` URL
 * (not fetched by the mapper) or a `data:<mime>;base64,<payload>` URL
 * (decoded and forwarded to the model as raw bytes).
 */
export interface InputImagePart {
  type: 'input_image';
  image_url?: string;
  file_id?: string;
  detail?: 'auto' | 'low' | 'high';
}

/**
 * Echoed back by clients (Codex, pi-ai, etc.) when they replay prior
 * assistant turns in `input[]` instead of using `previous_response_id`.
 * Equivalent to `InputTextPart` for mapping purposes — we only need the text.
 */
export interface InputAssistantTextPart {
  type: 'output_text';
  text: string;
  annotations?: never[];
}

/** Echoed back by clients when replaying a refusal block from a prior turn. */
export interface InputRefusalPart {
  type: 'refusal';
  refusal: string;
}

/**
 * Some clients inline a reasoning summary as a content part on an assistant
 * message instead of as a top-level `reasoning` input item.
 */
export interface InputSummaryTextPart {
  type: 'summary_text';
  text: string;
}

export type ContentPart =
  | InputTextPart
  | InputImagePart
  | InputAssistantTextPart
  | InputRefusalPart
  | InputSummaryTextPart;

// ---------------------------------------------------------------------------
// Input items
// ---------------------------------------------------------------------------

export interface InputMessage {
  type?: 'message';
  role: 'user' | 'assistant' | 'system' | 'developer';
  content: string | ContentPart[];
}

export interface InputFunctionCall {
  type: 'function_call';
  id: string;
  call_id: string;
  name: string;
  arguments: string;
}

export interface InputFunctionCallOutput {
  type: 'function_call_output';
  call_id: string;
  output: string;
}

/**
 * Top-level reasoning item replayed by clients that echo prior assistant
 * reasoning summaries instead of using `previous_response_id`. The summary
 * is coalesced onto the next assistant `ChatMessage` as `reasoningContent`.
 */
export interface InputReasoningItem {
  type: 'reasoning';
  id?: string;
  summary?: { type?: 'summary_text'; text: string }[];
  encrypted_content?: string;
}

export type InputItem = InputMessage | InputFunctionCall | InputFunctionCallOutput | InputReasoningItem;

// ---------------------------------------------------------------------------
// Tool definitions (Responses API shape)
// ---------------------------------------------------------------------------

export interface ResponsesToolDefinition {
  type: 'function';
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  strict?: boolean;
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

export interface ResponsesAPIRequest {
  model: string;
  input: string | InputItem[];
  instructions?: string;
  tools?: ResponsesToolDefinition[];
  tool_choice?: 'auto' | 'required' | 'none' | { type: 'function'; name: string };
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  max_output_tokens?: number;
  reasoning?: { effort?: string; summary?: string };
  previous_response_id?: string;
  store?: boolean;
  /**
   * OpenAI-reserved `metadata` slot, repurposed here to carry MLX-Node
   * extensions. Today this only exposes `retention_seconds` as a
   * per-request override of `ServerConfig.responseRetentionSec`;
   * unrelated keys are accepted and ignored (additive, forward-compat).
   */
  metadata?: {
    /**
     * Per-request retention override for the stored response row, in
     * seconds. Must be a finite positive integer in `[60, 90 * 86400]`
     * (1 minute … 90 days). Out-of-range or non-integer values return
     * 400. When omitted, the server-wide default applies.
     */
    retention_seconds?: number;
  };
}

// ---------------------------------------------------------------------------
// Output items
// ---------------------------------------------------------------------------

export interface OutputTextPart {
  type: 'output_text';
  text: string;
  annotations: never[];
}

export interface SummaryTextPart {
  type: 'summary_text';
  text: string;
}

export interface MessageOutputItem {
  id: string;
  type: 'message';
  role: 'assistant';
  status: 'completed' | 'incomplete' | 'in_progress';
  content: OutputTextPart[];
}

export interface ReasoningOutputItem {
  id: string;
  type: 'reasoning';
  summary: SummaryTextPart[];
}

export interface FunctionCallOutputItem {
  id: string;
  type: 'function_call';
  call_id: string;
  name: string;
  arguments: string;
  status: 'completed' | 'incomplete';
}

export type OutputItem = MessageOutputItem | ReasoningOutputItem | FunctionCallOutputItem;

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

export interface ResponseUsage {
  input_tokens: number;
  output_tokens: number;
  output_tokens_details: { reasoning_tokens: number };
  total_tokens: number;
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

export interface ResponseError {
  type: string;
  message: string;
  code: string | null;
  param: string | null;
}

// ---------------------------------------------------------------------------
// Response object
// ---------------------------------------------------------------------------

export interface ResponseObject {
  id: string;
  object: 'response';
  created_at: number;
  status: 'completed' | 'incomplete' | 'in_progress' | 'failed';
  model: string;
  output: OutputItem[];
  output_text: string;
  error: ResponseError | null;
  incomplete_details: { reason: string } | null;
  usage: ResponseUsage;
  instructions: string | null;
  temperature: number | null;
  top_p: number | null;
  max_output_tokens: number | null;
  tools: ResponsesToolDefinition[];
  tool_choice: 'auto' | 'required' | 'none' | { type: 'function'; name: string } | null;
  reasoning: { effort?: string; summary?: string } | null;
  previous_response_id: string | null;
}

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

export interface ResponseCreatedEvent {
  type: 'response.created';
  response: ResponseObject;
}

export interface ResponseInProgressEvent {
  type: 'response.in_progress';
  response: ResponseObject;
}

export interface ResponseCompletedEvent {
  type: 'response.completed';
  response: ResponseObject;
}

export interface ResponseFailedEvent {
  type: 'response.failed';
  response: ResponseObject;
}

export interface OutputItemAddedEvent {
  type: 'response.output_item.added';
  output_index: number;
  item: OutputItem;
}

export interface OutputItemDoneEvent {
  type: 'response.output_item.done';
  output_index: number;
  item: OutputItem;
}

export interface ContentPartAddedEvent {
  type: 'response.content_part.added';
  item_id: string;
  output_index: number;
  content_index: number;
  part: OutputTextPart;
}

export interface ContentPartDoneEvent {
  type: 'response.content_part.done';
  item_id: string;
  output_index: number;
  content_index: number;
  part: OutputTextPart;
}

export interface OutputTextDeltaEvent {
  type: 'response.output_text.delta';
  item_id: string;
  output_index: number;
  content_index: number;
  delta: string;
}

export interface OutputTextDoneEvent {
  type: 'response.output_text.done';
  item_id: string;
  output_index: number;
  content_index: number;
  text: string;
}

export interface ReasoningSummaryTextDeltaEvent {
  type: 'response.reasoning_summary_text.delta';
  item_id: string;
  output_index: number;
  summary_index: number;
  delta: string;
}

export interface ReasoningSummaryTextDoneEvent {
  type: 'response.reasoning_summary_text.done';
  item_id: string;
  output_index: number;
  summary_index: number;
  text: string;
}

export interface FunctionCallArgumentsDeltaEvent {
  type: 'response.function_call_arguments.delta';
  item_id: string;
  output_index: number;
  delta: string;
}

export interface FunctionCallArgumentsDoneEvent {
  type: 'response.function_call_arguments.done';
  item_id: string;
  output_index: number;
  arguments: string;
}

export type StreamEvent =
  | ResponseCreatedEvent
  | ResponseInProgressEvent
  | ResponseCompletedEvent
  | ResponseFailedEvent
  | OutputItemAddedEvent
  | OutputItemDoneEvent
  | ContentPartAddedEvent
  | ContentPartDoneEvent
  | OutputTextDeltaEvent
  | OutputTextDoneEvent
  | ReasoningSummaryTextDeltaEvent
  | ReasoningSummaryTextDoneEvent
  | FunctionCallArgumentsDeltaEvent
  | FunctionCallArgumentsDoneEvent;
