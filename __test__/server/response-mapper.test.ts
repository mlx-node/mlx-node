import { describe, expect, it } from 'vite-plus/test';

import {
  buildOutputItems,
  buildResponseObject,
  buildUsage,
  computeOutputText,
  mapFinishReasonToStatus,
} from '../../packages/server/src/mappers/response.js';

function makeChatResult(overrides: Record<string, unknown> = {}) {
  return {
    text: 'Hello!',
    toolCalls: [] as {
      id: string;
      name: string;
      arguments: Record<string, unknown> | string;
      status: string;
      rawContent: string;
      error?: string;
    }[],
    thinking: undefined as string | undefined,
    numTokens: 5,
    promptTokens: 10,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'Hello!',
    performance: undefined,
    ...overrides,
  };
}

describe('mapFinishReasonToStatus', () => {
  it('maps "stop" to "completed"', () => {
    expect(mapFinishReasonToStatus('stop')).toBe('completed');
  });

  it('maps "length" to "incomplete"', () => {
    expect(mapFinishReasonToStatus('length')).toBe('incomplete');
  });

  it('maps "tool_calls" to "completed"', () => {
    expect(mapFinishReasonToStatus('tool_calls')).toBe('completed');
  });

  it('maps unknown reason to "completed" (default case)', () => {
    expect(mapFinishReasonToStatus('unknown_reason')).toBe('completed');
  });
});

describe('buildOutputItems', () => {
  it('produces a single message item for text-only result', () => {
    const result = makeChatResult();
    const items = buildOutputItems(result);

    expect(items).toHaveLength(1);
    expect(items[0].type).toBe('message');
    const msg = items[0] as { type: 'message'; role: string; status: string; content: { text: string }[] };
    expect(msg.role).toBe('assistant');
    expect(msg.status).toBe('completed');
    expect(msg.content).toHaveLength(1);
    expect(msg.content[0].text).toBe('Hello!');
  });

  it('produces reasoning item before message when thinking is present', () => {
    const result = makeChatResult({ thinking: 'Let me think...' });
    const items = buildOutputItems(result);

    expect(items).toHaveLength(2);
    expect(items[0].type).toBe('reasoning');
    const reasoning = items[0] as { type: 'reasoning'; summary: { text: string }[] };
    expect(reasoning.summary).toHaveLength(1);
    expect(reasoning.summary[0].text).toBe('Let me think...');
    expect(items[1].type).toBe('message');
  });

  it('produces function_call items after message when tool calls are present', () => {
    const result = makeChatResult({
      text: 'Let me check the weather.',
      toolCalls: [{ id: 'call_123', name: 'get_weather', arguments: '{"city":"SF"}', status: 'ok', rawContent: '' }],
    });
    const items = buildOutputItems(result);

    expect(items).toHaveLength(2);
    expect(items[0].type).toBe('message');
    expect(items[1].type).toBe('function_call');
    const fc = items[1] as { type: 'function_call'; name: string; arguments: string; call_id: string };
    expect(fc.name).toBe('get_weather');
    expect(fc.arguments).toBe('{"city":"SF"}');
    expect(fc.call_id).toBe('call_123');
  });

  it('omits message item when text is empty and tool calls are present', () => {
    const result = makeChatResult({
      text: '',
      toolCalls: [{ id: 'call_1', name: 'get_weather', arguments: '{}', status: 'ok', rawContent: '' }],
    });
    const items = buildOutputItems(result);

    expect(items).toHaveLength(1);
    expect(items[0].type).toBe('function_call');
  });

  it('includes message item with empty text when no tool calls', () => {
    const result = makeChatResult({ text: '' });
    const items = buildOutputItems(result);

    expect(items).toHaveLength(1);
    expect(items[0].type).toBe('message');
    const msg = items[0] as { type: 'message'; content: { text: string }[] };
    expect(msg.content[0].text).toBe('');
  });

  it('stringifies non-string tool call arguments', () => {
    const result = makeChatResult({
      text: '',
      toolCalls: [{ id: 'call_2', name: 'fn', arguments: { key: 'value' }, status: 'ok', rawContent: '' }],
    });
    const items = buildOutputItems(result);

    const fc = items[0] as { arguments: string };
    expect(fc.arguments).toBe('{"key":"value"}');
  });

  it('generates unique IDs for each item', () => {
    const result = makeChatResult({
      thinking: 'hmm',
      toolCalls: [{ id: 'call_3', name: 'fn', arguments: '{}', status: 'ok', rawContent: '' }],
    });
    const items = buildOutputItems(result);

    const ids = items.map((i) => i.id);
    const uniqueIds = new Set(ids);
    expect(uniqueIds.size).toBe(ids.length);
  });

  it('sets status to incomplete when finish reason is length', () => {
    const result = makeChatResult({ finishReason: 'length' });
    const items = buildOutputItems(result);

    const msg = items[0] as { status: string };
    expect(msg.status).toBe('incomplete');
  });
});

describe('buildUsage', () => {
  it('correctly maps promptTokens and numTokens', () => {
    const result = makeChatResult({ promptTokens: 15, numTokens: 25 });
    const usage = buildUsage(result);

    expect(usage.input_tokens).toBe(15);
    expect(usage.output_tokens).toBe(25);
    expect(usage.total_tokens).toBe(40);
    expect(usage.output_tokens_details.reasoning_tokens).toBe(0);
  });

  it('reports reasoning_tokens from ChatResult', () => {
    const result = makeChatResult({ promptTokens: 15, numTokens: 25, reasoningTokens: 12 });
    const usage = buildUsage(result);

    expect(usage.input_tokens).toBe(15);
    expect(usage.output_tokens).toBe(25);
    expect(usage.total_tokens).toBe(40);
    expect(usage.output_tokens_details.reasoning_tokens).toBe(12);
  });

  it('handles zero tokens', () => {
    const result = makeChatResult({ promptTokens: 0, numTokens: 0 });
    const usage = buildUsage(result);

    expect(usage.input_tokens).toBe(0);
    expect(usage.output_tokens).toBe(0);
    expect(usage.total_tokens).toBe(0);
  });
});

describe('computeOutputText', () => {
  it('concatenates text from message items only', () => {
    const items = [
      { id: 'rs_1', type: 'reasoning' as const, summary: [{ type: 'summary_text' as const, text: 'thinking' }] },
      {
        id: 'msg_1',
        type: 'message' as const,
        role: 'assistant' as const,
        status: 'completed' as const,
        content: [{ type: 'output_text' as const, text: 'Hello', annotations: [] as never[] }],
      },
      {
        id: 'fc_1',
        type: 'function_call' as const,
        call_id: 'call_1',
        name: 'fn',
        arguments: '{}',
        status: 'completed' as const,
      },
    ];

    expect(computeOutputText(items)).toBe('Hello');
  });

  it('returns empty string when no message items', () => {
    expect(computeOutputText([])).toBe('');
  });
});

describe('buildResponseObject', () => {
  it('builds a complete response object', () => {
    const result = makeChatResult();
    const req = {
      model: 'test-model',
      input: 'Hello',
      instructions: 'Be brief',
      temperature: 0.5,
      top_p: 0.9,
      max_output_tokens: 100,
      tools: [] as any[],
      tool_choice: 'auto' as const,
      reasoning: { effort: 'high' },
    };

    const response = buildResponseObject(result, req, 'resp_123', 'resp_prev');

    expect(response.id).toBe('resp_123');
    expect(response.object).toBe('response');
    expect(response.status).toBe('completed');
    expect(response.model).toBe('test-model');
    expect(response.output).toHaveLength(1);
    expect(response.output_text).toBe('Hello!');
    expect(response.error).toBeNull();
    expect(response.incomplete_details).toBeNull();
    expect(response.instructions).toBe('Be brief');
    expect(response.temperature).toBe(0.5);
    expect(response.top_p).toBe(0.9);
    expect(response.max_output_tokens).toBe(100);
    expect(response.tool_choice).toBe('auto');
    expect(response.reasoning).toEqual({ effort: 'high' });
    expect(response.previous_response_id).toBe('resp_prev');
    expect(response.usage.input_tokens).toBe(10);
    expect(response.usage.output_tokens).toBe(5);
  });

  it('sets incomplete_details when finish reason is length', () => {
    const result = makeChatResult({ finishReason: 'length' });
    const req = { model: 'test-model', input: 'Hello' };

    const response = buildResponseObject(result, req, 'resp_1');

    expect(response.status).toBe('incomplete');
    expect(response.incomplete_details).toEqual({ reason: 'max_output_tokens' });
  });

  it('defaults optional fields to null', () => {
    const result = makeChatResult();
    const req = { model: 'test-model', input: 'Hello' };

    const response = buildResponseObject(result, req, 'resp_1');

    expect(response.instructions).toBeNull();
    expect(response.temperature).toBeNull();
    expect(response.top_p).toBeNull();
    expect(response.max_output_tokens).toBeNull();
    expect(response.tool_choice).toBeNull();
    expect(response.reasoning).toBeNull();
    expect(response.previous_response_id).toBeNull();
  });

  it('uses empty array for tools when not specified', () => {
    const result = makeChatResult();
    const req = { model: 'test-model', input: 'Hello' };

    const response = buildResponseObject(result, req, 'resp_1');
    expect(response.tools).toEqual([]);
  });
});
