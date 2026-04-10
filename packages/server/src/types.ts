/**
 * OpenAI Responses API type definitions
 *
 * Covers the request/response shapes for POST /v1/responses
 * and the SSE streaming event protocol.
 */

// ---------------------------------------------------------------------------
// Content parts
// ---------------------------------------------------------------------------

export interface InputTextPart {
  type: 'input_text';
  text: string;
}

export type ContentPart = InputTextPart;

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

export type InputItem = InputMessage | InputFunctionCall | InputFunctionCallOutput;

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
  status: 'completed';
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
