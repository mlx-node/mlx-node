import { describe, expect, it } from 'vite-plus/test';

import {
  anthropicToolUseIdToInternal,
  buildAnthropicResponse,
  buildContentBlockDelta,
  buildContentBlockStart,
  buildContentBlockStop,
  buildMessageDelta,
  buildMessageStartEvent,
  buildMessageStop,
  internalToolCallIdToAnthropic,
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
    cachedTokens: 0,
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
    expect(response.content[0]).toEqual({
      type: 'thinking',
      thinking: 'Let me reason through this.',
    });
    expect(response.content[1]).toEqual({ type: 'text', text: 'Hello!' });
  });

  it('tool use response produces tool_use blocks with parsed input', () => {
    const result = makeChatResult({
      text: '',
      toolCalls: [
        {
          id: 'toolu_01',
          name: 'get_weather',
          arguments: '{"city":"SF"}',
          status: 'ok',
          rawContent: '',
        },
      ],
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

  it('suppresses Gemma4 parsed tool-call text without duplicating reasoning when tools are not allowed', () => {
    const result = makeChatResult({
      text: '',
      thinking: 'I should inspect files.',
      toolCalls: [
        {
          id: 'toolu_01',
          name: 'read_file',
          arguments: '{"path":"Cargo.toml"}',
          status: 'ok',
          rawContent: '',
        },
      ],
      rawText:
        '<|channel>thought\nI should inspect files.\n<channel|><|tool_call>call:read_file{path:<|"|>Cargo.toml<|"|>}<tool_call|><turn|>',
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_no_tools', undefined, false);

    expect(response.stop_reason).toBe('end_turn');
    expect(response.content).toHaveLength(2);
    expect(response.content[0]).toEqual({
      type: 'thinking',
      thinking: 'I should inspect files.',
    });
    expect(response.content[1]).toEqual({
      type: 'text',
      text: '',
    });
  });

  it('tool use with object arguments uses them directly', () => {
    const result = makeChatResult({
      text: '',
      toolCalls: [
        {
          id: 'toolu_02',
          name: 'search',
          arguments: { query: 'MLX' },
          status: 'ok',
          rawContent: '',
        },
      ],
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
      toolCalls: [
        {
          id: 'toolu_03',
          name: 'lookup',
          arguments: '{"term":"foo"}',
          status: 'ok',
          rawContent: '',
        },
      ],
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

  it('omits cache fields when cachedTokens === 0', () => {
    // Anthropic spec leaves the cache fields OPTIONAL and other
    // Anthropic-compatible servers omit them on misses — so a wire
    // emitting `cache_read_input_tokens: 0` would diverge.
    const result = makeChatResult({
      promptTokens: 11,
      numTokens: 4,
      cachedTokens: 0,
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_no_cache');

    expect(response.usage.input_tokens).toBe(11);
    expect(response.usage.output_tokens).toBe(4);
    expect(response.usage).not.toHaveProperty('cache_read_input_tokens');
    expect(response.usage).not.toHaveProperty('cache_creation_input_tokens');
  });

  it('emits cache_read_input_tokens with reduced input_tokens when cachedTokens > 0', () => {
    // The spec says `input_tokens` is the count processed at full
    // cost on the turn, so on a cache HIT it MUST be the unsuffixed
    // remainder (`promptTokens - cachedTokens`) — billing UIs read
    // these fields directly.
    const result = makeChatResult({
      promptTokens: 20,
      numTokens: 5,
      cachedTokens: 7,
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_cache_hit');

    expect(response.usage.cache_read_input_tokens).toBe(7);
    expect(response.usage.input_tokens).toBe(13);
    expect(response.usage.output_tokens).toBe(5);
    // Implicit-prefix-cache server: never emits a non-zero
    // `cache_creation_input_tokens` because it does not honour
    // explicit `cache_control` breakpoints.
    expect(response.usage).not.toHaveProperty('cache_creation_input_tokens');
  });

  it('emits server-extension perf fields when performance is supplied', () => {
    // Non-Anthropic extension fields surfaced for the launcher's
    // verbose log. Anthropic-compatible clients ignore unknown
    // fields, so these stay wire-safe.
    const result = makeChatResult();
    const response = buildAnthropicResponse(result, baseReq, 'msg_perf', {
      ttftMs: 1234,
      prefillTokensPerSecond: 800,
      decodeTokensPerSecond: 73.5,
    });

    expect(response.usage.time_to_first_token_ms).toBe(1234);
    expect(response.usage.prefill_tokens_per_second).toBe(800);
    expect(response.usage.decode_tokens_per_second).toBe(73.5);
    expect(response.usage.server_time_to_first_token_ms).toBe(1234);
    expect(response.usage.server_prefill_tokens_per_second).toBe(800);
    expect(response.usage.server_decode_tokens_per_second).toBe(73.5);
    expect(response.usage.prefill_input_tokens).toBe(5);
    expect(response.usage).not.toHaveProperty('cached_prefix_tokens');
  });

  it('adds cached-prefix context for perf fields on cache hits', () => {
    const result = makeChatResult({
      promptTokens: 20,
      numTokens: 5,
      cachedTokens: 15,
    });
    const response = buildAnthropicResponse(
      result,
      baseReq,
      'msg_cache_perf',
      {
        ttftMs: 250,
        prefillTokensPerSecond: 20,
        decodeTokensPerSecond: 40,
      },
      true,
      {
        server_model_resolve_ms: 57,
        server_queue_ms: 5,
        server_pre_inference_ms: 62,
        server_paged_prefill_chunk_size: 4096,
        server_paged_prefill_eval_interval: 8,
        server_paged_decode_cache_clear_interval: 64,
      },
    );

    expect(response.usage.input_tokens).toBe(5);
    expect(response.usage.cache_read_input_tokens).toBe(15);
    expect(response.usage.prefill_input_tokens).toBe(5);
    expect(response.usage.cached_prefix_tokens).toBe(15);
    expect(response.usage.server_inference_elapsed_ms).toBe(350);
    expect(response.usage.server_total_time_to_first_token_ms).toBe(312);
    expect(response.usage.server_model_resolve_ms).toBe(57);
    expect(response.usage.server_queue_ms).toBe(5);
    expect(response.usage.server_pre_inference_ms).toBe(62);
    expect(response.usage.server_paged_prefill_chunk_size).toBe(4096);
    expect(response.usage.server_paged_prefill_eval_interval).toBe(8);
    expect(response.usage.server_paged_decode_cache_clear_interval).toBe(64);
  });

  it('elides perf fields when performance is undefined', () => {
    const result = makeChatResult();
    const response = buildAnthropicResponse(result, baseReq, 'msg_no_perf', undefined);

    expect(response.usage).not.toHaveProperty('time_to_first_token_ms');
    expect(response.usage).not.toHaveProperty('prefill_tokens_per_second');
    expect(response.usage).not.toHaveProperty('decode_tokens_per_second');
  });

  it('elides invalid / zero perf metrics, keeps finite > 0 values', () => {
    // Same gating pattern as `cache_read_input_tokens`: a
    // partially-plumbed driver (or a bf16 NaN slipping through the
    // native path) must NOT surface a zero / NaN as if it were a
    // real measurement. Only the finite, positive value lands.
    const result = makeChatResult();
    const response = buildAnthropicResponse(result, baseReq, 'msg_partial_perf', {
      ttftMs: Number.NaN,
      prefillTokensPerSecond: 0,
      decodeTokensPerSecond: 73.5,
    });

    expect(response.usage).not.toHaveProperty('time_to_first_token_ms');
    expect(response.usage).not.toHaveProperty('prefill_tokens_per_second');
    expect(response.usage.decode_tokens_per_second).toBe(73.5);
  });

  it('empty text with tool calls produces no text block, only tool_use blocks', () => {
    const result = makeChatResult({
      text: '',
      toolCalls: [
        {
          id: 'toolu_04',
          name: 'fn',
          arguments: '{}',
          status: 'ok',
          rawContent: '',
        },
      ],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_no_text');

    expect(response.content.every((b) => b.type !== 'text')).toBe(true);
    expect(response.content).toHaveLength(1);
    expect(response.content[0].type).toBe('tool_use');
  });

  it('skips tool calls with status !== "ok"', () => {
    const result = makeChatResult({
      text: 'Done.',
      toolCalls: [
        {
          id: 'toolu_err',
          name: 'broken',
          arguments: '{}',
          status: 'error',
          rawContent: '',
        },
      ],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_err_tool');

    expect(response.content).toHaveLength(1);
    expect(response.content[0].type).toBe('text');
  });

  it('translates native call_<uuid> ids to Anthropic toolu_<uuid> on the wire', () => {
    // The native parser mints `call_<uuid>` (the OpenAI Responses
    // convention, kept stable on the OpenAI endpoint). The Anthropic
    // wire spec uses `toolu_<uuid>`. Translation happens at the wire
    // boundary only — the uuid body is preserved verbatim so a client
    // that echoes the id back via `tool_result.tool_use_id` round-trips
    // losslessly through `anthropicToolUseIdToInternal`.
    const result = makeChatResult({
      text: '',
      toolCalls: [
        {
          id: 'call_abc123def456',
          name: 'get_weather',
          arguments: '{"city":"SF"}',
          status: 'ok',
          rawContent: '',
        },
      ],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_translate');

    const toolBlock = response.content[0] as { type: 'tool_use'; id: string };
    expect(toolBlock.id).toBe('toolu_abc123def456');
  });

  it('passes through tool_use ids without the call_ prefix unchanged', () => {
    // Defensive: an in-process driver or legacy bridge that already
    // emits a non-`call_*` id must not have it mangled. The translator
    // is a no-op on any id that does not match the expected prefix.
    const result = makeChatResult({
      text: '',
      toolCalls: [
        {
          id: 'custom_xyz',
          name: 'fn',
          arguments: '{}',
          status: 'ok',
          rawContent: '',
        },
      ],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_passthrough');

    const toolBlock = response.content[0] as { type: 'tool_use'; id: string };
    expect(toolBlock.id).toBe('custom_xyz');
  });

  it('generates a toolu_ prefixed id when tool call id is missing', () => {
    const result = makeChatResult({
      text: '',
      toolCalls: [
        {
          id: undefined as unknown as string,
          name: 'fn',
          arguments: '{}',
          status: 'ok',
          rawContent: '',
        },
      ],
    });
    const response = buildAnthropicResponse(result, baseReq, 'msg_gen_id');

    const toolBlock = response.content[0] as { type: 'tool_use'; id: string };
    expect(toolBlock.id).toMatch(/^toolu_/);
  });
});

describe('tool-call id translation helpers', () => {
  it('rewrites internal call_<uuid> ids to Anthropic toolu_<uuid>', () => {
    expect(internalToolCallIdToAnthropic('call_abc123')).toBe('toolu_abc123');
  });

  it('rewrites Anthropic toolu_<uuid> ids back to internal call_<uuid>', () => {
    expect(anthropicToolUseIdToInternal('toolu_abc123')).toBe('call_abc123');
  });

  it('round-trips losslessly across the wire boundary', () => {
    const internal = 'call_0123456789abcdef';
    expect(anthropicToolUseIdToInternal(internalToolCallIdToAnthropic(internal))).toBe(internal);
  });

  it('passes unrecognised ids through unchanged in both directions', () => {
    // A legacy caller that ships raw `call_*` over the Anthropic wire,
    // or any in-process driver constructing its own id, must not have
    // the value mangled. Both translators are no-ops outside the
    // expected prefix.
    expect(internalToolCallIdToAnthropic('legacy_xyz')).toBe('legacy_xyz');
    expect(anthropicToolUseIdToInternal('legacy_xyz')).toBe('legacy_xyz');
    expect(internalToolCallIdToAnthropic('toolu_already')).toBe('toolu_already');
    expect(anthropicToolUseIdToInternal('call_already')).toBe('call_already');
    expect(internalToolCallIdToAnthropic('')).toBe('');
    expect(anthropicToolUseIdToInternal('')).toBe('');
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

  it('passes through input_tokens when supplied without cachedTokens', () => {
    const event = buildMessageDelta('end_turn', 5, 11);

    expect(event.usage.input_tokens).toBe(11);
    expect(event.usage.output_tokens).toBe(5);
    expect(event.usage).not.toHaveProperty('cache_read_input_tokens');
    expect(event.usage).not.toHaveProperty('cache_creation_input_tokens');
  });

  it('omits cache fields when cachedTokens is 0', () => {
    // Mirrors the response mapper's miss-shape — a streaming turn
    // with no native reuse must look the same as a cold turn.
    const event = buildMessageDelta('end_turn', 5, 11, 0);

    expect(event.usage.input_tokens).toBe(11);
    expect(event.usage).not.toHaveProperty('cache_read_input_tokens');
    expect(event.usage).not.toHaveProperty('cache_creation_input_tokens');
  });

  it('emits cache_read_input_tokens with reduced input_tokens when cachedTokens > 0', () => {
    // Streaming variant of the response mapper hit-shape: the
    // `usage` block on `message_delta` carries the same body-level
    // cache accounting as the non-streaming response.
    const event = buildMessageDelta('end_turn', 5, 20, 7);

    expect(event.usage.cache_read_input_tokens).toBe(7);
    expect(event.usage.input_tokens).toBe(13);
    expect(event.usage.output_tokens).toBe(5);
    expect(event.usage).not.toHaveProperty('cache_creation_input_tokens');
  });

  it('emits cache_read_input_tokens without input_tokens when inputTokens is omitted', () => {
    // Edge case: a driver that never plumbs `inputTokens` but does
    // surface `cachedTokens > 0` still gets a useful streaming delta
    // — `cache_read_input_tokens` lands on the wire and
    // `input_tokens` simply stays absent (cannot be derived without
    // the prompt count).
    const event = buildMessageDelta('end_turn', 5, undefined, 7);

    expect(event.usage.cache_read_input_tokens).toBe(7);
    expect(event.usage).not.toHaveProperty('input_tokens');
    expect(event.usage.output_tokens).toBe(5);
  });

  it('emits server-extension perf fields when performance is supplied', () => {
    // Non-Anthropic extension fields surfaced for the launcher's
    // verbose log (`requests.ndjson`). Mirrors the non-streaming
    // `buildAnthropicResponse` perf path.
    const event = buildMessageDelta('end_turn', 5, undefined, undefined, {
      ttftMs: 1234,
      prefillTokensPerSecond: 800,
      decodeTokensPerSecond: 73.5,
    });

    expect(event.usage.time_to_first_token_ms).toBe(1234);
    expect(event.usage.prefill_tokens_per_second).toBe(800);
    expect(event.usage.decode_tokens_per_second).toBe(73.5);
    expect(event.usage.server_time_to_first_token_ms).toBe(1234);
    expect(event.usage.server_prefill_tokens_per_second).toBe(800);
    expect(event.usage.server_decode_tokens_per_second).toBe(73.5);
  });

  it('adds cached-prefix context to streaming perf usage', () => {
    const event = buildMessageDelta(
      'end_turn',
      5,
      20,
      15,
      {
        ttftMs: 250,
        prefillTokensPerSecond: 20,
        decodeTokensPerSecond: 40,
      },
      {
        server_model_resolve_ms: 57,
        server_queue_ms: 5,
        server_pre_inference_ms: 62,
        server_paged_prefill_chunk_size: 4096,
        server_paged_prefill_eval_interval: 8,
        server_paged_decode_cache_clear_interval: 64,
      },
    );

    expect(event.usage.input_tokens).toBe(5);
    expect(event.usage.cache_read_input_tokens).toBe(15);
    expect(event.usage.prefill_input_tokens).toBe(5);
    expect(event.usage.cached_prefix_tokens).toBe(15);
    expect(event.usage.server_inference_elapsed_ms).toBe(350);
    expect(event.usage.server_total_time_to_first_token_ms).toBe(312);
    expect(event.usage.server_model_resolve_ms).toBe(57);
    expect(event.usage.server_queue_ms).toBe(5);
    expect(event.usage.server_pre_inference_ms).toBe(62);
    expect(event.usage.server_paged_prefill_chunk_size).toBe(4096);
    expect(event.usage.server_paged_prefill_eval_interval).toBe(8);
    expect(event.usage.server_paged_decode_cache_clear_interval).toBe(64);
  });

  it('elides perf fields when performance is undefined', () => {
    const event = buildMessageDelta('end_turn', 5, 11, 0, undefined);

    expect(event.usage).not.toHaveProperty('time_to_first_token_ms');
    expect(event.usage).not.toHaveProperty('prefill_tokens_per_second');
    expect(event.usage).not.toHaveProperty('decode_tokens_per_second');
  });

  it('elides invalid / zero perf metrics, keeps finite > 0 values', () => {
    // Filtering: a partially-plumbed dispatch returning NaN / 0
    // must NOT surface those as wire metrics — the launcher reads
    // absence as "not plumbed", a literal zero would look like a
    // real measurement.
    const event = buildMessageDelta('end_turn', 5, undefined, undefined, {
      ttftMs: Number.NaN,
      prefillTokensPerSecond: 0,
      decodeTokensPerSecond: 73.5,
    });

    expect(event.usage).not.toHaveProperty('time_to_first_token_ms');
    expect(event.usage).not.toHaveProperty('prefill_tokens_per_second');
    expect(event.usage.decode_tokens_per_second).toBe(73.5);
  });
});

describe('buildMessageStop', () => {
  it('returns a message_stop event', () => {
    const event = buildMessageStop();

    expect(event.type).toBe('message_stop');
  });
});
