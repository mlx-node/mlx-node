/** ChatResult / ChatStreamEvent → OpenAI Responses API output. */

import { randomUUID } from 'node:crypto';

import type { ChatResult } from '@mlx-node/core';

import type {
  FunctionCallOutputItem,
  MessageOutputItem,
  OutputItem,
  ReasoningOutputItem,
  ResponseObject,
  ResponsesAPIRequest,
  ResponseUsage,
} from '../types.js';

export function genId(prefix: string): string {
  return `${prefix}${randomUUID().replaceAll('-', '')}`;
}

export function mapFinishReasonToStatus(finishReason: string): 'completed' | 'incomplete' {
  switch (finishReason) {
    case 'length':
      return 'incomplete';
    default:
      return 'completed';
  }
}

export function buildOutputItems(result: ChatResult): OutputItem[] {
  const items: OutputItem[] = [];

  if (result.thinking) {
    const reasoningItem: ReasoningOutputItem = {
      id: genId('rs_'),
      type: 'reasoning',
      summary: [{ type: 'summary_text', text: result.thinking }],
    };
    items.push(reasoningItem);
  }

  const okToolCalls = result.toolCalls.filter((t) => t.status === 'ok');

  // Always emit a message item (possibly with empty text) unless there are tool calls and no text.
  if (result.text || okToolCalls.length === 0) {
    const messageItem: MessageOutputItem = {
      id: genId('msg_'),
      type: 'message',
      role: 'assistant',
      status: mapFinishReasonToStatus(result.finishReason),
      content: [{ type: 'output_text', text: result.text, annotations: [] as never[] }],
    };
    items.push(messageItem);
  }

  for (const tc of okToolCalls) {
    const callId = tc.id ?? genId('call_');
    const fcItem: FunctionCallOutputItem = {
      id: genId('fc_'),
      type: 'function_call',
      call_id: callId,
      name: tc.name,
      arguments: typeof tc.arguments === 'string' ? tc.arguments : JSON.stringify(tc.arguments),
      status: 'completed',
    };
    items.push(fcItem);
  }

  return items;
}

export function buildUsage(result: ChatResult): ResponseUsage {
  return {
    input_tokens: result.promptTokens,
    output_tokens: result.numTokens,
    output_tokens_details: { reasoning_tokens: result.reasoningTokens },
    total_tokens: result.promptTokens + result.numTokens,
  };
}

/** Concatenate all `output_text` parts from message items. */
export function computeOutputText(items: OutputItem[]): string {
  const parts: string[] = [];
  for (const item of items) {
    if (item.type === 'message') {
      for (const c of item.content) {
        parts.push(c.text);
      }
    }
  }
  return parts.join('');
}

export function buildResponseObject(
  result: ChatResult,
  req: ResponsesAPIRequest,
  responseId: string,
  previousResponseId?: string,
): ResponseObject {
  const output = buildOutputItems(result);
  const status = mapFinishReasonToStatus(result.finishReason);

  return {
    id: responseId,
    object: 'response',
    created_at: Math.floor(Date.now() / 1000),
    status,
    model: req.model,
    output,
    output_text: computeOutputText(output),
    error: null,
    incomplete_details: status === 'incomplete' ? { reason: 'max_output_tokens' } : null,
    usage: buildUsage(result),
    instructions: req.instructions ?? null,
    temperature: req.temperature ?? null,
    top_p: req.top_p ?? null,
    max_output_tokens: req.max_output_tokens ?? null,
    tools: req.tools ?? [],
    tool_choice: req.tool_choice ?? null,
    reasoning: req.reasoning ?? null,
    previous_response_id: previousResponseId ?? null,
  };
}

/** Build an in-progress ResponseObject for `response.created` / `response.in_progress`, before any output exists. */
export function buildPartialResponse(
  req: ResponsesAPIRequest,
  responseId: string,
  previousResponseId?: string,
): ResponseObject {
  return {
    id: responseId,
    object: 'response',
    created_at: Math.floor(Date.now() / 1000),
    status: 'in_progress',
    model: req.model,
    output: [],
    output_text: '',
    error: null,
    incomplete_details: null,
    usage: { input_tokens: 0, output_tokens: 0, output_tokens_details: { reasoning_tokens: 0 }, total_tokens: 0 },
    instructions: req.instructions ?? null,
    temperature: req.temperature ?? null,
    top_p: req.top_p ?? null,
    max_output_tokens: req.max_output_tokens ?? null,
    tools: req.tools ?? [],
    tool_choice: req.tool_choice ?? null,
    reasoning: req.reasoning ?? null,
    previous_response_id: previousResponseId ?? null,
  };
}
