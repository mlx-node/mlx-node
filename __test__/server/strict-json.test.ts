/**
 * BUG 2 regression: /v1 JSON responses must be strictly-parseable.
 *
 * Two leak vectors covered here:
 *
 *   1. Raw C0 control characters inside JSON string literals. Model
 *      output routinely embeds literal \x00-\x1f bytes; every emitted
 *      document must escape them. `JSON.stringify` already does — the
 *      defence-in-depth `stringifyStrictJson` guarantees no emission
 *      path can regress.
 *
 *   2. Verbatim inner-JSON payloads. `FunctionCallOutputItem.arguments`
 *      and the Anthropic `partial_json` delta ship a model-generated JSON
 *      string verbatim — a raw control char inside it made that INNER
 *      document strictly un-parseable, and on /v1/messages a bare
 *      `JSON.parse` of such arguments threw the whole turn into a 500.
 *      `sanitizeJsonString` (port of the Rust chatml sanitizer) repairs
 *      them byte-identically for compliant input.
 */

import type { ServerResponse } from 'node:http';

import type { ChatResult, ToolCallResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { handleCreateResponse } from '../../packages/server/src/endpoints/responses.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';
import { sanitizeJsonString, stringifyStrictJson } from '../../packages/server/src/strict-json.js';

// ---------------------------------------------------------------------------
// Mock helpers (mirrors messages-handler.test.ts)
// ---------------------------------------------------------------------------

function createMockRes(): {
  res: ServerResponse;
  getStatus: () => number;
  getBody: () => string;
} {
  const { Writable } = require('node:stream');
  let status = 200;
  let body = '';

  const writable = new Writable({
    write(chunk: Uint8Array | string, _encoding: string, callback: () => void) {
      body += chunk.toString();
      callback();
    },
  });
  writable.on('error', () => {});

  writable.writeHead = (s: number, _h?: Record<string, string>) => {
    status = s;
    writable.headersSent = true;
    return writable;
  };
  writable.setHeader = () => {};
  writable.headersSent = false;

  const origEnd = writable.end.bind(writable);
  writable.end = (chunkArg?: unknown, encodingArg?: unknown, cbArg?: unknown) => {
    let chunk: string | Uint8Array | undefined;
    let encoding: unknown;
    let cb: ((err?: Error | null) => void) | undefined;
    if (typeof chunkArg === 'function') {
      cb = chunkArg as (err?: Error | null) => void;
    } else {
      chunk = chunkArg as string | Uint8Array | undefined;
      if (typeof encodingArg === 'function') {
        cb = encodingArg as (err?: Error | null) => void;
      } else {
        encoding = encodingArg;
        if (typeof cbArg === 'function') {
          cb = cbArg as (err?: Error | null) => void;
        }
      }
    }
    if (chunk != null) body += chunk.toString();
    writable.headersSent = true;
    origEnd(undefined, encoding, (err?: Error | null) => {
      if (cb) cb(err ?? null);
    });
    return writable;
  };

  return {
    res: writable as unknown as ServerResponse,
    getStatus: () => status,
    getBody: () => body,
  };
}

function makeChatResult(overrides: Partial<ChatResult> = {}): ChatResult {
  return {
    text: 'Hello!',
    toolCalls: [] as ToolCallResult[],
    thinkingEnabled: true,
    numTokens: 10,
    promptTokens: 5,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'Hello!',
    cachedTokens: 0,
    performance: undefined,
    ...overrides,
  } as ChatResult;
}

function createMockModel(result: ChatResult = makeChatResult()): SessionCapableModel {
  async function* fallbackStream() {
    yield {
      done: true,
      text: result.text,
      finishReason: result.finishReason,
      toolCalls: result.toolCalls,
      thinking: result.thinking ?? null,
      numTokens: result.numTokens,
      promptTokens: result.promptTokens,
      reasoningTokens: result.reasoningTokens,
      rawText: result.rawText,
    };
  }
  return {
    chatSessionStart: vi.fn().mockResolvedValue(result),
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path not expected')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path not expected')),
    chatStreamSessionStart: vi.fn(() => fallbackStream()),
    chatStreamSessionContinue: vi.fn(() => fallbackStream()),
    chatStreamSessionContinueTool: vi.fn(() => fallbackStream()),
    resetCaches: vi.fn().mockResolvedValue(undefined),
  } as unknown as SessionCapableModel;
}

function createMockStreamModel(streamEvents: Array<Record<string, unknown>>): SessionCapableModel {
  async function* makeStream() {
    for (const event of streamEvents) {
      yield event;
    }
  }
  return {
    chatSessionStart: vi.fn().mockRejectedValue(new Error('Should use chatStreamSessionStart')),
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path not expected')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path not expected')),
    chatStreamSessionStart: vi.fn(() => makeStream()),
    chatStreamSessionContinue: vi.fn(() => makeStream()),
    chatStreamSessionContinueTool: vi.fn(() => makeStream()),
    resetCaches: vi.fn().mockResolvedValue(undefined),
  } as unknown as SessionCapableModel;
}

/** Parse SSE body into { event, data } pairs, strictly parsing each data payload. */
function parseSSE(body: string): Array<{ event: string; data: Record<string, unknown>; rawData: string }> {
  const results: Array<{ event: string; data: Record<string, unknown>; rawData: string }> = [];
  let currentEvent = '';
  for (const line of body.split('\n')) {
    if (line.startsWith('event: ')) {
      currentEvent = line.slice(7);
    } else if (line.startsWith('data: ')) {
      const rawData = line.slice(6);
      results.push({ event: currentEvent, data: JSON.parse(rawData) as Record<string, unknown>, rawData });
    }
  }
  return results;
}

/** True when the raw payload text contains any unescaped C0/DEL byte. */
function hasRawControlChar(text: string): boolean {
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code <= 0x1f || code === 0x7f) return true;
  }
  return false;
}

const TOOLS = [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }];

// ---------------------------------------------------------------------------
// sanitizeJsonString / stringifyStrictJson unit tests
// ---------------------------------------------------------------------------

describe('sanitizeJsonString', () => {
  it('escapes raw C0 control characters inside string literals', () => {
    expect(sanitizeJsonString('{"a":"x\x01y"}')).toBe('{"a":"x\\u0001y"}');
    expect(sanitizeJsonString('{"a":"x\x07y\x1bz\x1f"}')).toBe('{"a":"x\\u0007y\\u001bz\\u001f"}');
    expect(sanitizeJsonString('{"a":"x\x00y"}')).toBe('{"a":"x\\u0000y"}');
  });

  it('uses shorthand escapes for \\n \\r \\t \\b \\f', () => {
    expect(sanitizeJsonString('{"a":"l1\nl2"}')).toBe('{"a":"l1\\nl2"}');
    expect(sanitizeJsonString('{"a":"l1\rl2"}')).toBe('{"a":"l1\\rl2"}');
    expect(sanitizeJsonString('{"a":"l1\tl2"}')).toBe('{"a":"l1\\tl2"}');
    expect(sanitizeJsonString('{"a":"l1\bl2"}')).toBe('{"a":"l1\\bl2"}');
    expect(sanitizeJsonString('{"a":"l1\fl2"}')).toBe('{"a":"l1\\fl2"}');
  });

  it('leaves whitespace between tokens and structural chars untouched', () => {
    // A newline OUTSIDE string literals is legal JSON whitespace and must
    // not be escaped (it is what pretty-printed documents look like).
    const doc = '{\n  "a": 1,\n  "b": [\n    2\n  ]\n}';
    expect(sanitizeJsonString(doc)).toBe(doc);
  });

  it('is byte-identical for compliant input (idempotent)', () => {
    const compliant = JSON.stringify({ a: 'x\ny\tz', b: ['\\n', 'esc"aped', 'back\\slash'], c: null });
    expect(sanitizeJsonString(compliant)).toBe(compliant);
    // Existing escape sequences are copied verbatim, never double-escaped.
    expect(sanitizeJsonString('{"a":"\\n\\u0041\\t"}')).toBe('{"a":"\\n\\u0041\\t"}');
  });

  it('leaves non-ASCII and UTF-8 content untouched', () => {
    const doc = '{"a":"héllo wörld — 你好 😀"}';
    expect(sanitizeJsonString(doc)).toBe(doc);
  });

  it('does not treat a quote inside an escaped sequence as a string boundary', () => {
    // The `\\"` keeps the scanner inside the string, so the raw \n after it
    // is still escaped.
    expect(sanitizeJsonString('{"a":"q\\"uote\nx"}')).toBe('{"a":"q\\"uote\\nx"}');
  });

  it('produces strictly-parseable output for a raw-C0 document', () => {
    const repaired = sanitizeJsonString('{"query":"line1\nline2","n":1}');
    expect(JSON.parse(repaired)).toEqual({ query: 'line1\nline2', n: 1 });
  });

  it('repairs a control char directly after a backslash instead of copying it raw', () => {
    // Malformed input `"a\<LF>b"` (backslash then raw newline inside the
    // string) must not leak the raw C0 through the escape-copy path.
    const repaired = sanitizeJsonString('{"a":"x\\\ny"}');
    expect(hasRawControlChar(repaired)).toBe(false);
    expect(JSON.parse(repaired)).toEqual({ a: 'x\\\ny' });
  });
});

describe('stringifyStrictJson', () => {
  it('is byte-identical to JSON.stringify for compliant payloads', () => {
    const value = { a: 'x\ny', b: [1, 'two', null], c: { d: 0x1f } };
    expect(stringifyStrictJson(value)).toBe(JSON.stringify(value));
  });

  it('round-trips control characters through strict JSON.parse', () => {
    const value = { text: 'a\x00b\x07c\x1fd' };
    const out = stringifyStrictJson(value);
    expect(hasRawControlChar(out)).toBe(false);
    expect(JSON.parse(out)).toEqual(value);
  });
});

// ---------------------------------------------------------------------------
// /v1/messages
// ---------------------------------------------------------------------------

describe('BUG 2: /v1/messages strict JSON', () => {
  it('non-streaming: control characters in text/thinking are escaped on the wire', async () => {
    const registry = new ModelRegistry();
    registry.register(
      'test-model',
      createMockModel(
        makeChatResult({
          text: 'line1\x01line2\x07\nline3\x1f',
          thinking: 'think\x00ing\x1b!',
          rawText: 'line1\x01line2\x07\nline3\x1f',
        }),
      ),
    );
    const { res, getStatus, getBody } = createMockRes();

    await handleCreateMessage(
      res,
      { model: 'test-model', messages: [{ role: 'user', content: 'hi' }], max_tokens: 100 } as never,
      registry,
    );

    expect(getStatus()).toBe(200);
    const body = getBody();
    // Strict parse of the raw bytes — the bug's original failure mode.
    const parsed = JSON.parse(body) as { content: Array<{ type: string; text?: string; thinking?: string }> };
    const textBlock = parsed.content.find((b) => b.type === 'text')!;
    const thinkBlock = parsed.content.find((b) => b.type === 'thinking')!;
    expect(textBlock.text).toBe('line1\x01line2\x07\nline3\x1f');
    expect(thinkBlock.thinking).toBe('think\x00ing\x1b!');
  });

  it('non-streaming: tool_use arguments with raw control chars do not 500 and carry sanitized input', async () => {
    const registry = new ModelRegistry();
    registry.register(
      'test-model',
      createMockModel(
        makeChatResult({
          text: '',
          // Raw newline inside the model's arguments JSON — previously a
          // bare JSON.parse threw and the turn became a 500 api_error.
          toolCalls: [
            {
              status: 'ok',
              id: 'call_ctrl1',
              name: 'get_weather',
              arguments: '{"query":"line1\nline2"}',
            } as ToolCallResult,
          ],
          finishReason: 'stop',
          rawText: '',
        }),
      ),
    );
    const { res, getStatus, getBody } = createMockRes();

    await handleCreateMessage(
      res,
      {
        model: 'test-model',
        messages: [{ role: 'user', content: 'weather?' }],
        max_tokens: 100,
        tools: TOOLS,
      } as never,
      registry,
    );

    expect(getStatus()).toBe(200);
    const parsed = JSON.parse(getBody()) as { content: Array<{ type: string; input?: unknown }> };
    const toolBlock = parsed.content.find((b) => b.type === 'tool_use');
    expect(toolBlock).toBeDefined();
    // The inner input round-trips semantically: the raw newline survives
    // as a real newline inside the parsed value.
    expect(toolBlock!.input).toEqual({ query: 'line1\nline2' });
  });

  it('streaming: every data: payload strictly parses and partial_json is independently strict', async () => {
    const registry = new ModelRegistry();
    const streamEvents = [
      { text: 'reason\x01ing', done: false, isReasoning: true },
      { text: 'pre\x07 ', done: false, isReasoning: false },
      {
        text: 'pre\x07 ',
        done: true,
        finishReason: 'tool_calls',
        toolCalls: [
          {
            status: 'ok',
            id: 'call_ctrl2',
            name: 'get_weather',
            arguments: '{"q":"a\ta\x1fb"}',
          },
        ],
        thinking: 'reason\x01ing',
        numTokens: 9,
        promptTokens: 4,
        reasoningTokens: 2,
        rawText: '',
      },
    ];
    registry.register('test-model', createMockStreamModel(streamEvents));
    const { res, getBody } = createMockRes();

    await handleCreateMessage(
      res,
      {
        model: 'test-model',
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 100,
        stream: true,
        tools: TOOLS,
      } as never,
      registry,
    );

    const events = parseSSE(getBody());
    expect(events.length).toBeGreaterThan(0);
    // No raw control byte may appear inside any data: payload.
    for (const e of events) {
      expect(hasRawControlChar(e.rawData)).toBe(false);
    }
    // The input_json_delta's partial_json is itself a JSON document a
    // client parses incrementally — it must be strict too.
    const jsonDelta = events.find(
      (e) => e.event === 'content_block_delta' && (e.data['delta'] as { type?: string }).type === 'input_json_delta',
    );
    expect(jsonDelta).toBeDefined();
    const partialJson = (jsonDelta!.data['delta'] as { partial_json: string }).partial_json;
    expect(hasRawControlChar(partialJson)).toBe(false);
    expect(JSON.parse(partialJson)).toEqual({ q: 'a\ta\x1fb' });
    expect(events.some((e) => e.event === 'message_stop')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// /v1/responses
// ---------------------------------------------------------------------------

describe('BUG 2: /v1/responses strict JSON', () => {
  it('non-streaming: verbatim arguments string is sanitized for strict clients', async () => {
    const registry = new ModelRegistry();
    registry.register(
      'test-model',
      createMockModel(
        makeChatResult({
          text: '',
          toolCalls: [
            {
              status: 'ok',
              id: 'call_ctrl3',
              name: 'get_weather',
              // Raw control chars inside a verbatim arguments JSON string.
              arguments: '{"city":"SF\x01X","q":"l1\nl2"}',
            } as ToolCallResult,
          ],
          finishReason: 'stop',
          rawText: '',
        }),
      ),
    );
    const { res, getStatus, getBody } = createMockRes();

    await handleCreateResponse(
      res,
      { model: 'test-model', input: 'weather?' } as never,
      registry,
      null,
    );

    expect(getStatus()).toBe(200);
    const parsed = JSON.parse(getBody()) as {
      output: Array<{ type: string; arguments?: string }>;
    };
    const fc = parsed.output.find((i) => i.type === 'function_call')!;
    // The INNER arguments payload a client parses must itself be strict.
    expect(hasRawControlChar(fc.arguments!)).toBe(false);
    expect(JSON.parse(fc.arguments!)).toEqual({ city: 'SF\x01X', q: 'l1\nl2' });
  });

  it('non-streaming: control characters in output_text/reasoning are escaped on the wire', async () => {
    const registry = new ModelRegistry();
    registry.register(
      'test-model',
      createMockModel(
        makeChatResult({
          text: 'out\x00put\x1f',
          thinking: 'sum\x07mary',
          rawText: 'out\x00put\x1f',
        }),
      ),
    );
    const { res, getStatus, getBody } = createMockRes();

    await handleCreateResponse(
      res,
      { model: 'test-model', input: 'hi' } as never,
      registry,
      null,
    );

    expect(getStatus()).toBe(200);
    const parsed = JSON.parse(getBody()) as {
      output_text: string;
      output: Array<{ type: string; content?: Array<{ text: string }>; summary?: Array<{ text: string }> }>;
    };
    expect(parsed.output_text).toBe('out\x00put\x1f');
    const reasoning = parsed.output.find((i) => i.type === 'reasoning')!;
    expect(reasoning.summary![0].text).toBe('sum\x07mary');
  });

  it('streaming: every data: payload strictly parses and function_call_arguments inner JSON is strict', async () => {
    const registry = new ModelRegistry();
    const streamEvents = [
      { text: 'rsn\x02', done: false, isReasoning: true },
      { text: 'delta\x01text', done: false, isReasoning: false },
      {
        text: 'delta\x01text',
        done: true,
        finishReason: 'tool_calls',
        toolCalls: [
          {
            status: 'ok',
            id: 'call_ctrl4',
            name: 'get_weather',
            arguments: '{"a":"x\x03y"}',
          },
        ],
        thinking: 'rsn\x02',
        numTokens: 7,
        promptTokens: 3,
        reasoningTokens: 1,
        rawText: '',
      },
    ];
    registry.register('test-model', createMockStreamModel(streamEvents));
    const { res, getBody } = createMockRes();

    await handleCreateResponse(
      res,
      { model: 'test-model', input: 'hi', stream: true } as never,
      registry,
      null,
    );

    const events = parseSSE(getBody());
    expect(events.length).toBeGreaterThan(0);
    for (const e of events) {
      expect(hasRawControlChar(e.rawData)).toBe(false);
    }
    const argsDelta = events.find((e) => e.event === 'response.function_call_arguments.delta');
    expect(argsDelta).toBeDefined();
    const innerArgs = (argsDelta!.data['delta'] as string | undefined) ?? '';
    expect(hasRawControlChar(innerArgs)).toBe(false);
    expect(JSON.parse(innerArgs)).toEqual({ a: 'x\x03y' });
    const argsDone = events.find((e) => e.event === 'response.function_call_arguments.done');
    expect(argsDone).toBeDefined();
    expect(JSON.parse(argsDone!.data['arguments'] as string)).toEqual({ a: 'x\x03y' });
    expect(events.some((e) => e.event === 'response.completed')).toBe(true);
  });
});
