import { describe, expect, it } from 'vite-plus/test';

import {
  buildAnthropicResponse,
  buildContentBlockDelta,
  buildContentBlockStart,
  buildContentBlockStop,
  buildMessageDelta,
  buildMessageStartEvent,
  buildMessageStop,
  mapStopReason,
} from '../../packages/server/src/mappers/anthropic-response.js';

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
    numTokens: 10,
    promptTokens: 5,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'Hello!',
    performance: undefined,
    ...overrides,
  };
}

const baseReq = {
  model: 'claude-3-5-sonnet-20241022',
  messages: [],
  max_tokens: 1024,
};

describe('mapStopReason', () => {
  it('maps "stop" with no tool calls to "end_turn"', () => {
    expect(mapStopReason('stop', false)).toBe('end_turn');
  });

  it('maps "stop" with tool calls to "tool_use"', () => {
    expect(mapStopReason('stop', true)).toBe('tool_use');
  });

  it('maps "length" to "max_tokens" regardless of tool calls', () => {
    expect(mapStopReason('length', false)).toBe('max_tokens');
    expect(mapStopReason('length', true)).toBe('max_tokens');
  });

  it('maps unknown reason with no tool calls to "end_turn"', () => {
    expect(mapStopReason('unknown', false)).toBe('end_turn');
  });

  it('maps unknown reason with tool calls to "tool_use"', () => {
    expect(mapStopReason('unknown', true)).toBe('tool_use');
  });
});

describe('buildAnthropicResponse', () => {
  it('text-only response produces a single text content block', () => {
    const result = makeChatResult();
    const response = buildAnthropicResponse(result, baseReq, 'msg_abc123');

    expect(response.id).toBe('msg_abc123');
    expect(response.type).toBe('message');
    expect(response.role).toBe('assistant');
    expect(response.model).toBe('claude-3-5-sonnet-20241022');
    expect(response.content).toHaveLength(1);
    expect(response.content[0]).toEqual({ type: 'text', text: 'Hello!' });
    expect(response.stop_reason).toBe('end_turn');
    expect(response.stop_sequence).toBeNull();
  });

  it('thinking + text produces thinking block then text block', () => {
    const result = makeChatResult({ thinking: 'Let me reason through this.' });
    const response = buildAnthropicResponse(result, baseReq, 'msg_thinking');

    expect(response.content).toHaveLength(2);
    expect(response.content[0]).toEqual({ type: 'thinking', thinking: 'Let me reason through this.' });
    expect(response.content[1]).toEqual({ type: 'text', text: 'Hello!' });
  });

  it('tool use response produces tool_use blocks with parsed input', () => {
    const result = makeChatResult({
      text: '',
      toolCalls: [{ id: 'toolu_01', name: 'get_weather', arguments: '{"city":"SF"}', status: 'ok', rawContent: '' }],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_tool');

    expect(response.content).toHaveLength(1);
    expect(response.content[0]).toEqual({
      type: 'tool_use',
      id: 'toolu_01',
      name: 'get_weather',
      input: { city: 'SF' },
    });
    expect(response.stop_reason).toBe('tool_use');
  });

  it('tool use with object arguments uses them directly', () => {
    const result = makeChatResult({
      text: '',
      toolCalls: [{ id: 'toolu_02', name: 'search', arguments: { query: 'MLX' }, status: 'ok', rawContent: '' }],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_obj_args');

    expect(response.content[0]).toEqual({
      type: 'tool_use',
      id: 'toolu_02',
      name: 'search',
      input: { query: 'MLX' },
    });
  });

  it('mixed thinking + text + tool_use → all three blocks in order', () => {
    const result = makeChatResult({
      thinking: 'I should call a tool.',
      text: 'Let me look that up.',
      toolCalls: [{ id: 'toolu_03', name: 'lookup', arguments: '{"term":"foo"}', status: 'ok', rawContent: '' }],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_mixed');

    expect(response.content).toHaveLength(3);
    expect(response.content[0].type).toBe('thinking');
    expect(response.content[1].type).toBe('text');
    expect(response.content[2].type).toBe('tool_use');
  });

  it('stop_reason is "max_tokens" when finishReason is "length"', () => {
    const result = makeChatResult({ finishReason: 'length' });
    const response = buildAnthropicResponse(result, baseReq, 'msg_len');

    expect(response.stop_reason).toBe('max_tokens');
  });

  it('usage maps promptTokens → input_tokens and numTokens → output_tokens', () => {
    const result = makeChatResult({ promptTokens: 42, numTokens: 7 });
    const response = buildAnthropicResponse(result, baseReq, 'msg_usage');

    expect(response.usage.input_tokens).toBe(42);
    expect(response.usage.output_tokens).toBe(7);
  });

  it('empty text with tool calls produces no text block, only tool_use blocks', () => {
    const result = makeChatResult({
      text: '',
      toolCalls: [{ id: 'toolu_04', name: 'fn', arguments: '{}', status: 'ok', rawContent: '' }],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_no_text');

    expect(response.content.every((b) => b.type !== 'text')).toBe(true);
    expect(response.content).toHaveLength(1);
    expect(response.content[0].type).toBe('tool_use');
  });

  it('skips tool calls with status !== "ok"', () => {
    const result = makeChatResult({
      text: 'Done.',
      toolCalls: [{ id: 'toolu_err', name: 'broken', arguments: '{}', status: 'error', rawContent: '' }],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_err_tool');

    expect(response.content).toHaveLength(1);
    expect(response.content[0].type).toBe('text');
  });

  it('generates a toolu_ prefixed id when tool call id is missing', () => {
    const result = makeChatResult({
      text: '',
      toolCalls: [{ id: undefined as unknown as string, name: 'fn', arguments: '{}', status: 'ok', rawContent: '' }],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_gen_id');

    const toolBlock = response.content[0] as { type: 'tool_use'; id: string };
    expect(toolBlock.id).toMatch(/^toolu_/);
  });
});

describe('buildMessageStartEvent', () => {
  it('returns correct structure with empty content and zero output tokens', () => {
    const event = buildMessageStartEvent(baseReq, 'msg_start_01', 20);

    expect(event.type).toBe('message_start');
    expect(event.message.id).toBe('msg_start_01');
    expect(event.message.type).toBe('message');
    expect(event.message.role).toBe('assistant');
    expect(event.message.model).toBe('claude-3-5-sonnet-20241022');
    expect(event.message.content).toEqual([]);
    expect(event.message.stop_reason).toBeNull();
    expect(event.message.stop_sequence).toBeNull();
    expect(event.message.usage.input_tokens).toBe(20);
    expect(event.message.usage.output_tokens).toBe(0);
  });
});

describe('buildContentBlockStart', () => {
  it('returns a content_block_start event with the given index and block', () => {
    const block = { type: 'text' as const, text: '' };
    const event = buildContentBlockStart(0, block);

    expect(event.type).toBe('content_block_start');
    expect(event.index).toBe(0);
    expect(event.content_block).toEqual(block);
  });
});

describe('buildContentBlockDelta', () => {
  it('returns correct structure for a text_delta', () => {
    const delta = { type: 'text_delta' as const, text: 'Hello' };
    const event = buildContentBlockDelta(0, delta);

    expect(event.type).toBe('content_block_delta');
    expect(event.index).toBe(0);
    expect(event.delta).toEqual(delta);
  });

  it('returns correct structure for a thinking_delta', () => {
    const delta = { type: 'thinking_delta' as const, thinking: 'hmm' };
    const event = buildContentBlockDelta(0, delta);

    expect(event.delta).toEqual(delta);
  });

  it('returns correct structure for an input_json_delta', () => {
    const delta = { type: 'input_json_delta' as const, partial_json: '{"foo"' };
    const event = buildContentBlockDelta(1, delta);

    expect(event.index).toBe(1);
    expect(event.delta).toEqual(delta);
  });
});

describe('buildContentBlockStop', () => {
  it('returns a content_block_stop event with the given index', () => {
    const event = buildContentBlockStop(2);

    expect(event.type).toBe('content_block_stop');
    expect(event.index).toBe(2);
  });
});

describe('buildMessageDelta', () => {
  it('returns correct structure with stop_reason and output_tokens', () => {
    const event = buildMessageDelta('end_turn', 42);

    expect(event.type).toBe('message_delta');
    expect(event.delta.stop_reason).toBe('end_turn');
    expect(event.delta.stop_sequence).toBeNull();
    expect(event.usage.output_tokens).toBe(42);
  });
});

describe('buildMessageStop', () => {
  it('returns a message_stop event', () => {
    const event = buildMessageStop();

    expect(event.type).toBe('message_stop');
  });
});
