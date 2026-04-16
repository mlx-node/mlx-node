import type { ServerResponse } from 'node:http';

import type { ChatResult, ToolCallResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';

// ---------------------------------------------------------------------------
// Mock helpers
// ---------------------------------------------------------------------------

/**
 * Capture writes to a ServerResponse via a simple writable mock.
 */
function createMockRes(): {
  res: ServerResponse;
  getStatus: () => number;
  getBody: () => string;
  getHeaders: () => Record<string, string | string[]>;
  wasDestroyed: () => boolean;
} {
  const { Writable } = require('node:stream');
  let status = 200;
  let body = '';
  const headers: Record<string, string | string[]> = {};
  let destroyed = false;

  const writable = new Writable({
    write(chunk: Uint8Array | string, _encoding: string, callback: () => void) {
      body += chunk.toString();
      callback();
    },
  });
  // Swallow any `'error'` emitted by the underlying Writable's
  // destroy path — the mock has no error listeners, so a real
  // `writable.destroy(err)` would blow up as an uncaught error.
  writable.on('error', () => {});

  writable.writeHead = (s: number, h?: Record<string, string>) => {
    status = s;
    if (h) {
      for (const [k, v] of Object.entries(h)) {
        headers[k.toLowerCase()] = v;
      }
    }
    writable.headersSent = true;
    return writable;
  };

  writable.setHeader = (name: string, value: string) => {
    headers[name.toLowerCase()] = value;
  };

  writable.getHeader = (name: string) => {
    return headers[name.toLowerCase()];
  };

  writable.headersSent = false;

  const origEnd = writable.end.bind(writable);
  // Mirror Node's overloaded `end()` signature (chunk?, encoding? |
  // cb?, cb?). Iter-33 regression tests call `res.end(body, cb)`
  // via `endJson`, so the mock MUST hoist the callback out of the
  // `encoding` slot when it is a function — otherwise the cb never
  // fires and `endJson` hangs.
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

  const origDestroy = writable.destroy.bind(writable);
  writable.destroy = (err?: Error) => {
    destroyed = true;
    writable.headersSent = true;
    try {
      origDestroy(err);
    } catch {
      // Already torn down.
    }
    return writable;
  };

  return {
    res: writable as unknown as ServerResponse,
    getStatus: () => status,
    getBody: () => body,
    getHeaders: () => headers,
    wasDestroyed: () => destroyed,
  };
}

/**
 * Synthesize a ChatResult. Tests override only the fields they care about.
 */
function makeChatResult(overrides: Partial<ChatResult> = {}): ChatResult {
  return {
    text: 'Hello!',
    toolCalls: [] as ToolCallResult[],
    numTokens: 10,
    promptTokens: 5,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'Hello!',
    performance: undefined,
    ...overrides,
  };
}

/**
 * Build a session-capable mock model that resolves `chatSessionStart` with
 * the supplied `ChatResult`. The Anthropic endpoint is stateless so only the
 * cold-path entry points (`chatSessionStart` / `chatStreamSessionStart`) are
 * ever invoked; `chatSessionContinue` and friends are filled in with
 * rejecting stubs so a mistaken hot-path call surfaces immediately.
 */
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
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: chatSessionContinue not expected')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: chatSessionContinueTool not expected')),
    chatStreamSessionStart: vi.fn(() => fallbackStream()),
    chatStreamSessionContinue: vi.fn(() => fallbackStream()),
    chatStreamSessionContinueTool: vi.fn(() => fallbackStream()),
    resetCaches: vi.fn(),
  } as unknown as SessionCapableModel;
}

/**
 * Session-capable mock whose `chatStreamSessionStart` yields the supplied
 * stream events. `chatSessionStart` rejects so accidental non-streaming
 * routing is caught immediately.
 */
function createMockStreamModel(streamEvents: Array<Record<string, unknown>>): SessionCapableModel {
  async function* makeStream() {
    for (const event of streamEvents) {
      yield event;
    }
  }
  return {
    chatSessionStart: vi.fn().mockRejectedValue(new Error('Should use chatStreamSessionStart')),
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: chatSessionContinue not expected')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: chatSessionContinueTool not expected')),
    chatStreamSessionStart: vi.fn(() => makeStream()),
    chatStreamSessionContinue: vi.fn(() => makeStream()),
    chatStreamSessionContinueTool: vi.fn(() => makeStream()),
    resetCaches: vi.fn(),
  } as unknown as SessionCapableModel;
}

/** Parse SSE body into an array of { event, data } objects. */
function parseSSE(body: string): Array<{ event: string; data: Record<string, unknown> }> {
  const results: Array<{ event: string; data: Record<string, unknown> }> = [];
  const lines = body.split('\n');
  let currentEvent = '';
  for (const line of lines) {
    if (line.startsWith('event: ')) {
      currentEvent = line.slice(7);
    } else if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6)) as Record<string, unknown>;
      results.push({ event: currentEvent, data });
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('handleCreateMessage', () => {
  // -----------------------------------------------------------------------
  // Validation
  // -----------------------------------------------------------------------

  describe('validation', () => {
    it('returns 400 for missing model', async () => {
      const registry = new ModelRegistry();
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(res, { messages: [{ role: 'user', content: 'hi' }], max_tokens: 100 } as any, registry);

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('invalid_request_error');
      expect(parsed.error.message).toContain('model');
    });

    it('returns 400 for missing messages', async () => {
      const registry = new ModelRegistry();
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(res, { model: 'test', max_tokens: 100 } as any, registry);

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.message).toContain('messages');
    });

    it('returns 400 for empty messages array', async () => {
      const registry = new ModelRegistry();
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(res, { model: 'test', messages: [], max_tokens: 100 } as any, registry);

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.message).toContain('messages');
    });

    it('returns 400 for missing max_tokens', async () => {
      const registry = new ModelRegistry();
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(res, { model: 'test', messages: [{ role: 'user', content: 'hi' }] } as any, registry);

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.message).toContain('max_tokens');
    });

    it('returns 400 for non-positive max_tokens', async () => {
      const registry = new ModelRegistry();
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        { model: 'test', messages: [{ role: 'user', content: 'hi' }], max_tokens: 0 } as any,
        registry,
      );

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.message).toContain('max_tokens');
    });

    it('returns 400 for null message items', async () => {
      const registry = new ModelRegistry();
      registry.register('test', createMockModel());
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(res, { model: 'test', messages: [null as any], max_tokens: 100 }, registry);

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.message).toContain('non-null object');
    });

    it('returns 404 for unknown model', async () => {
      const registry = new ModelRegistry();
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        { model: 'nonexistent', messages: [{ role: 'user', content: 'hi' }], max_tokens: 100 },
        registry,
      );

      expect(getStatus()).toBe(404);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('not_found_error');
      expect(parsed.error.message).toContain('nonexistent');
    });
  });

  // -----------------------------------------------------------------------
  // Non-streaming
  // -----------------------------------------------------------------------

  describe('non-streaming', () => {
    it('returns 200 with correct Anthropic response format (text only)', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Hello' }],
          max_tokens: 100,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('message');
      expect(parsed.role).toBe('assistant');
      expect(parsed.model).toBe('test-model');
      expect(parsed.content).toHaveLength(1);
      expect(parsed.content[0].type).toBe('text');
      expect(parsed.content[0].text).toBe('Hello!');
      expect(parsed.stop_reason).toBe('end_turn');
      expect(parsed.usage.input_tokens).toBe(5);
      expect(parsed.usage.output_tokens).toBe(10);
    });

    it('returns thinking + text content blocks', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: 'The answer is 42.',
          toolCalls: [],
          thinking: 'Let me think about this...',
          numTokens: 15,
          promptTokens: 8,
          reasoningTokens: 5,
          finishReason: 'stop',
          rawText: 'The answer is 42.',
        }),
      );
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'What is the meaning of life?' }],
          max_tokens: 200,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.content).toHaveLength(2);
      expect(parsed.content[0].type).toBe('thinking');
      expect(parsed.content[0].thinking).toBe('Let me think about this...');
      expect(parsed.content[1].type).toBe('text');
      expect(parsed.content[1].text).toBe('The answer is 42.');
    });

    it('returns tool_use content blocks', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: '',
          toolCalls: [
            {
              status: 'ok',
              id: 'toolu_abc123',
              name: 'get_weather',
              arguments: '{"location":"San Francisco"}',
            } as ToolCallResult,
          ],
          numTokens: 20,
          promptTokens: 10,
          reasoningTokens: 0,
          finishReason: 'stop',
          rawText: '',
        }),
      );
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'What is the weather?' }],
          max_tokens: 100,
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }],
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.stop_reason).toBe('tool_use');
      // Should have tool_use block (no text block since text is empty with tool calls)
      const toolBlock = parsed.content.find((b: any) => b.type === 'tool_use');
      expect(toolBlock).toBeDefined();
      expect(toolBlock.name).toBe('get_weather');
      expect(toolBlock.input).toEqual({ location: 'San Francisco' });
    });
  });

  // -----------------------------------------------------------------------
  // Streaming (native chatStream)
  // -----------------------------------------------------------------------

  describe('streaming (native)', () => {
    it('emits correct SSE event sequence for text-only streaming', async () => {
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'Hello', done: false, isReasoning: false },
        { text: ' world', done: false, isReasoning: false },
        {
          text: 'Hello world',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 5,
          promptTokens: 3,
          reasoningTokens: 0,
          rawText: 'Hello world',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Hi' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      // message_start
      expect(events[0].event).toBe('message_start');
      expect(events[0].data['message']).toBeDefined();

      // content_block_start for text
      expect(events[1].event).toBe('content_block_start');
      expect((events[1].data['content_block'] as any).type).toBe('text');

      // text deltas
      const deltas = events.filter((e) => e.event === 'content_block_delta');
      expect(deltas.length).toBeGreaterThanOrEqual(2);
      expect((deltas[0].data['delta'] as any).text).toBe('Hello');
      expect((deltas[1].data['delta'] as any).text).toBe(' world');

      // content_block_stop
      const stops = events.filter((e) => e.event === 'content_block_stop');
      expect(stops.length).toBeGreaterThanOrEqual(1);

      // message_delta
      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeDefined();
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('end_turn');
      expect((msgDelta!.data['usage'] as any).output_tokens).toBe(5);

      // message_stop
      const msgStop = events.find((e) => e.event === 'message_stop');
      expect(msgStop).toBeDefined();
    });

    it('emits thinking + text with correct content block indices', async () => {
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'Let me think...', done: false, isReasoning: true },
        { text: 'More thought', done: false, isReasoning: true },
        { text: 'The answer', done: false, isReasoning: false },
        {
          text: 'The answer',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: 'Let me think...More thought',
          numTokens: 8,
          promptTokens: 4,
          reasoningTokens: 3,
          rawText: 'The answer',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Think about this' }],
          max_tokens: 200,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      // message_start
      expect(events[0].event).toBe('message_start');

      // content_block_start for thinking (index 0)
      expect(events[1].event).toBe('content_block_start');
      expect(events[1].data['index']).toBe(0);
      expect((events[1].data['content_block'] as any).type).toBe('thinking');

      // thinking deltas at index 0
      const thinkingDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'thinking_delta',
      );
      expect(thinkingDeltas.length).toBe(2);
      for (const d of thinkingDeltas) {
        expect(d.data['index']).toBe(0);
      }

      // content_block_stop for thinking (index 0)
      const thinkingStop = events.find((e) => e.event === 'content_block_stop' && e.data['index'] === 0);
      expect(thinkingStop).toBeDefined();

      // content_block_start for text (index 1)
      const textStart = events.find(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStart).toBeDefined();
      expect(textStart!.data['index']).toBe(1);

      // text delta at index 1
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      expect(textDeltas.length).toBeGreaterThanOrEqual(1);
      for (const d of textDeltas) {
        expect(d.data['index']).toBe(1);
      }
    });

    it('handles tool call streaming with tag suppression', async () => {
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'Let me check. ', done: false, isReasoning: false },
        { text: '<tool_call>', done: false, isReasoning: false },
        { text: '{"name":"get_weather"}', done: false, isReasoning: false },
        {
          text: '',
          done: true,
          finishReason: 'stop',
          toolCalls: [
            {
              status: 'ok',
              id: 'toolu_test1',
              name: 'get_weather',
              arguments: '{"location":"NYC"}',
            },
          ],
          thinking: null,
          numTokens: 12,
          promptTokens: 6,
          reasoningTokens: 0,
          rawText: 'Let me check. <tool_call>{"name":"get_weather"}',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Weather?' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      // Should have text block with "Let me check. " before suppression
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const textContent = textDeltas.map((d) => (d.data['delta'] as any).text).join('');
      expect(textContent).toBe('Let me check. ');

      // Should have tool_use block
      const toolStart = events.find(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolStart).toBeDefined();
      expect((toolStart!.data['content_block'] as any).name).toBe('get_weather');

      // Should have input_json_delta
      const jsonDelta = events.find(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'input_json_delta',
      );
      expect(jsonDelta).toBeDefined();

      // message_delta should have tool_use stop_reason
      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('tool_use');
    });

    it('suppresses tool_call tag and skips text block when empty', async () => {
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: '<tool_call>', done: false, isReasoning: false },
        { text: '{"name":"search"}', done: false, isReasoning: false },
        {
          text: '',
          done: true,
          finishReason: 'stop',
          toolCalls: [
            {
              status: 'ok',
              id: 'toolu_xyz',
              name: 'search',
              arguments: '{"query":"test"}',
            },
          ],
          thinking: null,
          numTokens: 8,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: '<tool_call>{"name":"search"}',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Search' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      // Should NOT have any text content_block_start
      const textStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStarts).toHaveLength(0);

      // Should have tool_use block
      const toolStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolStarts).toHaveLength(1);
    });

    it('recovers suppressed text after false-alarm tool_call tag when text was already emitted', async () => {
      // The model streams "Hello " then "<tool_call>" which triggers suppression,
      // but the final event has no actual tool calls — only plain text.
      // The client should receive ALL of "Hello <tool_call>world", not just "Hello ".
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'Hello ', done: false, isReasoning: false },
        { text: '<tool_call>', done: false, isReasoning: false },
        { text: 'world', done: false, isReasoning: false },
        {
          text: 'Hello <tool_call>world',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 10,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: 'Hello <tool_call>world',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Say hi' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      // Should have exactly one text content_block_start
      const textStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStarts).toHaveLength(1);

      // The combined text deltas should reconstruct the full text
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const combined = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(combined).toBe('Hello <tool_call>world');

      // Should NOT have any tool_use block
      const toolStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolStarts).toHaveLength(0);

      // stop reason should be end_turn (no tool calls)
      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('end_turn');
    });

    it('recovers full text after false-alarm tool_call tag when no text was emitted yet', async () => {
      // The model immediately outputs "<tool_call>" with no prior text,
      // but the final event has no actual tool calls.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: '<tool_call>', done: false, isReasoning: false },
        { text: 'just text', done: false, isReasoning: false },
        {
          text: '<tool_call>just text',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 8,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: '<tool_call>just text',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Say something' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      // Should have exactly one text content_block_start
      const textStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStarts).toHaveLength(1);

      // All of finalText should be in the text deltas
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const combined = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(combined).toBe('<tool_call>just text');
    });
  });

  // -----------------------------------------------------------------------
  // Error handling
  // -----------------------------------------------------------------------

  describe('error handling', () => {
    it('returns 500 when the session throws during non-streaming start', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      (mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('Model crashed'));
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );

      expect(getStatus()).toBe(500);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('api_error');
      expect(parsed.error.message).toContain('Model crashed');
    });

    it('emits error SSE event when the stream throws after headers are sent', async () => {
      const registry = new ModelRegistry();
      async function* crashingStream() {
        yield { text: 'partial', done: false, isReasoning: false };
        throw new Error('Stream crashed');
      }
      const mockModel = {
        chatSessionStart: vi.fn().mockRejectedValue(new Error('should use stream')),
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: vi.fn(() => crashingStream()),
        chatStreamSessionContinue: vi.fn(() => crashingStream()),
        chatStreamSessionContinueTool: vi.fn(() => crashingStream()),
        resetCaches: vi.fn(),
      } as unknown as SessionCapableModel;
      registry.register('test-model', mockModel);
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());
      const errorEvent = events.find((e) => e.event === 'error');
      expect(errorEvent).toBeDefined();
      expect((errorEvent!.data['error'] as any).message).toContain('Stream crashed');
    });

    it('routes a mid-decode throw through the failure epilogue without adopting the session', async () => {
      // Iter-28 finding 2 regression (Anthropic half): before the
      // fix, a mid-decode throw from the underlying async
      // generator escaped out of the `for await` loop in
      // `handleStreamingNative` straight into the outer generic
      // catch in `handleCreateMessage`. The outer catch emitted a
      // single Anthropic `error` event AFTER SSE headers had been
      // flushed, but did so WITHOUT consulting the commit gate
      // or running the failure epilogue. The consequences were:
      //
      //   1. `message_stop` was sometimes still emitted by the
      //      pre-break write path (labelling a crashed turn as
      //      a clean completion).
      //   2. The `content_block_stop` frame that pairs with any
      //      open `content_block_start` was never emitted, so
      //      clients tracking content_block_index state saw a
      //      dangling in-progress block.
      //   3. There was no signal the session had failed — the
      //      commit gate's `wasCommitted()` was never read, so
      //      the happy path's post-loop flow ran uninspected.
      //
      // The fix wraps the loop in a try/catch that captures the
      // throw into a sticky `thrownError` flag and routes the
      // post-loop block through the failure epilogue. Every
      // invariant below pins that epilogue:
      //
      //   * `message_stop` / `message_delta` MUST NOT appear.
      //   * A single Anthropic `error` event MUST be present
      //     with `type: 'api_error'` and a message that cites
      //     the thrown error.
      //   * Any content block that was opened before the throw
      //     MUST be closed with `content_block_stop` before the
      //     error frame (so clients aren't left with dangling
      //     state).
      //   * The session registry stays empty (the Anthropic
      //     endpoint never adopts, and the commit gate skips the
      //     success branch on a throw).
      async function* throwingStream() {
        yield { text: 'par', done: false, isReasoning: false };
        yield { text: 'tial', done: false, isReasoning: false };
        throw new Error('native decode crashed mid-flight');
      }
      const mockModel = {
        chatSessionStart: vi.fn().mockRejectedValue(new Error('should use stream')),
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: vi.fn(() => throwingStream()),
        chatStreamSessionContinue: vi.fn(() => throwingStream()),
        chatStreamSessionContinueTool: vi.fn(() => throwingStream()),
        resetCaches: vi.fn(),
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      // No clean terminal events.
      expect(events.find((e) => e.event === 'message_stop')).toBeUndefined();
      expect(events.find((e) => e.event === 'message_delta')).toBeUndefined();

      // Exactly one `error` event, with the thrown error's
      // message on the envelope.
      const errorEvents = events.filter((e) => e.event === 'error');
      expect(errorEvents).toHaveLength(1);
      const errorBody = errorEvents[0].data['error'] as { type: string; message: string };
      expect(errorBody.type).toBe('api_error');
      expect(errorBody.message).toContain('native decode crashed mid-flight');

      // Any content block that was opened before the throw must
      // be closed BEFORE the error frame.
      const orderedEvents = events.map((e) => e.event);
      const errorIdx = orderedEvents.indexOf('error');
      const anyBlockStopBeforeError = orderedEvents.slice(0, errorIdx).some((e) => e === 'content_block_stop');
      const anyBlockStartBeforeError = orderedEvents.slice(0, errorIdx).some((e) => e === 'content_block_start');
      if (anyBlockStartBeforeError) {
        expect(anyBlockStopBeforeError).toBe(true);
      }

      // Session registry stays empty — the Anthropic endpoint
      // never adopts, and no leak ever landed on the throw path.
      const sessionReg = registry.getSessionRegistry('test-model');
      expect(sessionReg!.size).toBe(0);
    });

    it('routes a client disconnect through the failure epilogue without adopting the session', async () => {
      // Iter-28 finding 2 regression (Anthropic half, client
      // abort path). Before the fix, a client disconnect while
      // streaming had no signal inside the helper: the
      // `for await` loop kept pulling deltas and the writer kept
      // calling `writeSSEEvent` into a dead socket until the
      // native generator drained a done chunk, at which point
      // the commit gate ran the success branch and the writer
      // emitted `message_delta` + `message_stop` into the void.
      //
      // The fix installs `close`/`error` listeners on the HTTP
      // request that flip a `clientAborted` flag checked at
      // loop-top. When the flag flips, the helper `break`s out
      // of the loop and routes through the failure epilogue. We
      // cannot physically close a socket in this unit test, so
      // we emit a synthetic `close` event on a lightweight
      // IncomingMessage-shaped mock after the generator yields
      // its first delta. The helper's loop-top guard must fire
      // on the next iteration and break out.
      let proceedResolve: (() => void) | undefined;
      const proceed = new Promise<void>((r) => {
        proceedResolve = r;
      });
      async function* abortingStream() {
        yield { text: 'partial', done: false, isReasoning: false };
        await proceed;
        // These events should not reach the client: the loop-top
        // `if (clientAborted) break;` trips on the next iter.
        yield { text: 'should-not-arrive', done: false, isReasoning: false };
        yield {
          text: 'should-not-arrive',
          done: true,
          finishReason: 'stop',
          toolCalls: [] as ToolCallResult[],
          thinking: null,
          numTokens: 1,
          promptTokens: 1,
          reasoningTokens: 0,
          rawText: 'should-not-arrive',
        };
      }
      const mockModel = {
        chatSessionStart: vi.fn().mockRejectedValue(new Error('should use stream')),
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: vi.fn(() => abortingStream()),
        chatStreamSessionContinue: vi.fn(() => abortingStream()),
        chatStreamSessionContinueTool: vi.fn(() => abortingStream()),
        resetCaches: vi.fn(),
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);
      const { res, getBody } = createMockRes();

      // Build a minimal IncomingMessage-shaped mock. Only
      // `once('close'|'error', fn)` and `off(...)` need to work
      // for the fault plumbing — the helper does not read body
      // or headers from the `httpReq` argument.
      const { EventEmitter } = require('node:events');
      const reqEmitter = new EventEmitter();
      const httpReqMock = Object.assign(reqEmitter, {
        method: 'POST',
        url: '/v1/messages',
        headers: { 'content-type': 'application/json', host: 'localhost' },
      });

      const inflight = handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
        httpReqMock as any,
      );
      await new Promise((r) => setImmediate(r));
      httpReqMock.emit('close');
      proceedResolve?.();
      await inflight;

      const events = parseSSE(getBody());

      // No clean terminal events.
      expect(events.find((e) => e.event === 'message_stop')).toBeUndefined();
      expect(events.find((e) => e.event === 'message_delta')).toBeUndefined();

      // Exactly one `error` event, citing the client disconnect.
      const errorEvents = events.filter((e) => e.event === 'error');
      expect(errorEvents).toHaveLength(1);
      const errorBody = errorEvents[0].data['error'] as { type: string; message: string };
      expect(errorBody.type).toBe('api_error');
      expect(errorBody.message).toMatch(/client disconnected/i);

      // The session registry stays empty on a client abort,
      // mirroring the other failure paths.
      const sessionReg = registry.getSessionRegistry('test-model');
      expect(sessionReg!.size).toBe(0);
    });
  });

  // -----------------------------------------------------------------------
  // SessionRegistry integration (findings 1-3 regressions)
  // -----------------------------------------------------------------------

  describe('session registry integration', () => {
    it('forwards a top-level system string into the mapped chatSessionStart history', async () => {
      // The Anthropic endpoint is stateless: every request allocates a
      // fresh ChatSession via `getOrCreate(null, systemString)`. The
      // registry parameter is unused on `null`, but the system prompt
      // still needs to land in the primed history. Guard against a
      // regression where the endpoint forgets to wire it through.
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      registry.register('test-model', mockModel);
      const { res, getStatus } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          system: 'You are terse.',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      // oxlint-disable-next-line @typescript-eslint/unbound-method
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(startSpy).toHaveBeenCalledTimes(1);
      const [messages] = startSpy.mock.calls[0] as [Array<{ role: string; content: string }>];
      const systemMsg = messages.find((m) => m.role === 'system');
      expect(systemMsg?.content).toBe('You are terse.');

      // The Anthropic endpoint never adopts a session, so the registry
      // stays empty regardless of the request outcome.
      const sessionReg = registry.getSessionRegistry('test-model');
      expect(sessionReg!.size).toBe(0);
    });

    it('handles a structured system field (array of content blocks)', async () => {
      // Anthropic `system` may be an array of SystemBlocks. The
      // mapper concatenates the text blocks into a single system
      // message, and the endpoint stringifies the array for the
      // registry identity check. Both paths must leave the request
      // working end-to-end.
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      registry.register('test-model', mockModel);
      const { res, getStatus } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          system: [{ type: 'text', text: 'Be concise.' } as any],
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      // oxlint-disable-next-line @typescript-eslint/unbound-method
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      const [messages] = startSpy.mock.calls[0] as [Array<{ role: string; content: string }>];
      const systemMsg = messages.find((m) => m.role === 'system');
      expect(systemMsg?.content).toBe('Be concise.');
    });

    it('emits a streaming error event (not message_stop) when the final chunk reports finishReason=error', async () => {
      // Iter-27 finding 3 regression: the previous implementation
      // emitted `message_stop` unconditionally as soon as a `done`
      // event arrived, even when `finishReason: 'error'`. That
      // reported a failed generation as a clean completion to the
      // client — no `error` SSE frame, no way to distinguish a
      // real success from a mid-decode failure. Mirror the
      // `/v1/responses` commit gate: on an uncommitted terminal
      // (ChatSession's `sawFinal` filter rolls back `turnCount`
      // whenever `finishReason === 'error'`, so the post-drain
      // `wasCommitted()` reads false), emit a single `error` SSE
      // event in the Anthropic shape and omit `message_stop`.
      const streamEvents = [
        { text: 'partial', done: false, isReasoning: false },
        {
          text: 'partial',
          done: true,
          finishReason: 'error',
          toolCalls: [] as ToolCallResult[],
          thinking: null,
          numTokens: 1,
          promptTokens: 3,
          reasoningTokens: 0,
          rawText: 'partial',
        },
      ];
      const registry = new ModelRegistry();
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      // Primary assertion: `message_stop` MUST NOT appear. Pairing
      // `message_stop` with a failed turn is the bug we are fixing.
      const msgStop = events.find((e) => e.event === 'message_stop');
      expect(msgStop).toBeUndefined();

      // And a `message_delta` with a non-error stop_reason must
      // NOT appear either — `message_delta` is the
      // usage/stop-reason announcement paired with `message_stop`.
      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeUndefined();

      // Primary assertion: an Anthropic streaming `error` event
      // MUST be present with the conventional envelope shape.
      const errorEvent = events.find((e) => e.event === 'error');
      expect(errorEvent).toBeDefined();
      expect(errorEvent!.data['type']).toBe('error');
      const errorBody = errorEvent!.data['error'] as { type: string; message: string };
      expect(errorBody.type).toBe('api_error');
      expect(errorBody.message).toMatch(/finishReason=error|did not commit/i);

      // The registry stayed empty — no adopt call leaked. The
      // Anthropic endpoint never adopts, but this pins the
      // invariant: no cached session ever reaches the registry on
      // a failed streaming turn.
      const sessionReg = registry.getSessionRegistry('test-model');
      expect(sessionReg!.size).toBe(0);
    });

    it('emits a streaming error event when the underlying async iterator exhausts without a done event', async () => {
      // Iter-27 finding 3 regression: if the native stream
      // exhausts mid-flight (producer throws, native crash, etc.)
      // the session's `sawFinal` finally runs without a commit,
      // so `wasCommitted()` reads false. Before the fix, the
      // post-loop safety net emitted `message_delta('end_turn')`
      // + `message_stop` unconditionally — again labelling a
      // failed turn as a clean completion. The new code emits an
      // Anthropic `error` SSE event instead.
      //
      // Use a stream that yields one delta and then exits cleanly
      // without a `done: true` chunk (the iterator exhausts with
      // no final frame). This simulates the producer terminating
      // early without signalling completion.
      const streamEvents = [{ text: 'partial', done: false, isReasoning: false }];
      const registry = new ModelRegistry();
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      // No clean terminal events.
      expect(events.find((e) => e.event === 'message_stop')).toBeUndefined();
      expect(events.find((e) => e.event === 'message_delta')).toBeUndefined();

      // Exactly one `error` event — the uncommitted terminal.
      const errorEvents = events.filter((e) => e.event === 'error');
      expect(errorEvents).toHaveLength(1);
      expect(errorEvents[0].data['type']).toBe('error');
      const errorBody = errorEvents[0].data['error'] as { type: string; message: string };
      expect(errorBody.type).toBe('api_error');
      expect(errorBody.message).toMatch(/without a done event|stream ended/i);

      // Content block that was opened mid-stream must be closed
      // before the error frame so the client isn't left with a
      // dangling in-progress text block.
      const blockStops = events.filter((e) => e.event === 'content_block_stop');
      expect(blockStops.length).toBeGreaterThanOrEqual(1);
    });

    it('still emits message_delta + message_stop on a clean streaming completion (counter-test)', async () => {
      // Positive counter-test for iter-27 finding 3: a happy-path
      // streaming turn — one that commits with a non-error
      // `finishReason` — still produces `message_delta` +
      // `message_stop` and NOT an `error` event. Pins the
      // commit-gate's success branch so the new uncommitted
      // handling can't accidentally swallow clean completions.
      const streamEvents = [
        { text: 'hello', done: false, isReasoning: false },
        { text: ' world', done: false, isReasoning: false },
        {
          text: 'hello world',
          done: true,
          finishReason: 'stop',
          toolCalls: [] as ToolCallResult[],
          thinking: null,
          numTokens: 7,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: 'hello world',
        },
      ];
      const registry = new ModelRegistry();
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeDefined();
      expect((msgDelta!.data['delta'] as { stop_reason: string }).stop_reason).toBe('end_turn');

      const msgStop = events.find((e) => e.event === 'message_stop');
      expect(msgStop).toBeDefined();

      // No `error` SSE event on the happy path.
      expect(events.find((e) => e.event === 'error')).toBeUndefined();
    });
  });

  // -----------------------------------------------------------------------
  // Iter-19 finding 2: canonicalize stateless fan-out tool order
  // -----------------------------------------------------------------------

  describe('stateless fan-out tool order', () => {
    it('canonicalizes reversed sibling tool_result order to match the assistant fan-out', async () => {
      // Iteration-19 finding 2 regression: the `/v1/messages`
      // endpoint is ALWAYS a stateless cold-start. The caller
      // ships a full conversation in `req.messages`, including
      // `tool_use`/`tool_result` blocks that the Anthropic mapper
      // folds into assistant fan-outs + subsequent `tool`
      // ChatMessages. Without the new
      // `validateAndCanonicalizeHistoryToolOrder` gate, caller-
      // supplied tool_result ordering flowed straight into
      // `primeHistory()`, and several native backends pair tool
      // results to fan-out calls POSITIONALLY — so reversing two
      // sibling tool_result blocks silently bound each output to
      // the WRONG sibling call.
      //
      // Construct an assistant fan-out with tool_use ids
      // [call_a, call_b], then submit reversed tool_result blocks
      // [call_b, call_a] in the follow-up user turn. Spy on
      // `chatSessionStart` to assert the handler reordered the
      // tool messages to match the sibling declaration order
      // BEFORE dispatching the primed history.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ text: 'both fetched' }));
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [
            { role: 'user', content: 'get weather and news' },
            {
              role: 'assistant',
              content: [
                { type: 'tool_use', id: 'call_a', name: 'get_weather', input: { city: 'SF' } },
                { type: 'tool_use', id: 'call_b', name: 'get_news', input: { q: 'tech' } },
              ],
            },
            {
              role: 'user',
              content: [
                // Intentionally reversed order — the handler must
                // canonicalize to [call_a, call_b] before dispatch.
                { type: 'tool_result', tool_use_id: 'call_b', content: '{"headlines":[]}' },
                { type: 'tool_result', tool_use_id: 'call_a', content: '{"temp":68}' },
              ],
            },
          ],
          max_tokens: 100,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.content[0].text).toBe('both fetched');

      // Inspect the messages primed into chatSessionStart. The two
      // tool messages must appear in canonical sibling order
      // [call_a, call_b], with their contents moved along with the
      // ids so each output is bound to the correct call.
      // eslint-disable-next-line @typescript-eslint/unbound-method
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(startSpy).toHaveBeenCalledTimes(1);
      const [primedMessages] = startSpy.mock.calls[0] as [
        Array<{ role: string; content: string; toolCallId?: string }>,
      ];
      const toolMessages = primedMessages.filter((m) => m.role === 'tool');
      expect(toolMessages).toHaveLength(2);
      expect(toolMessages[0]!.toolCallId).toBe('call_a');
      expect(toolMessages[1]!.toolCallId).toBe('call_b');
      expect(toolMessages[0]!.content).toBe('{"temp":68}');
      expect(toolMessages[1]!.content).toBe('{"headlines":[]}');
    });

    it('canonicalizes a reversed tool-result block when an earlier fan-out is already resolved', async () => {
      // Iteration-20 regression (finding 1), Anthropic variant:
      // before the fix `canonicalizeToolMessageOrder` scanned to
      // `messages.length`, so when the full-history walker invoked
      // it for the first fan-out in a history with multiple
      // resolved fan-outs, the helper would see tool messages from
      // every later block, fail its count gate
      // (`toolPositions.length !== expectedOrder.length`), and
      // silently leave a reversed first block uncorrected. The
      // `/v1/messages` endpoint is the same user-facing risk
      // surface as `/v1/responses` — both run the walker over a
      // full self-contained history on every request — so the
      // regression needs a twin assertion here.
      //
      // Build a two-fan-out Anthropic history where the FIRST
      // fan-out's `tool_result` blocks are reversed and the
      // SECOND fan-out is canonical. Assert that
      // `chatSessionStart` is primed with both blocks in sibling
      // order and that each output's content stayed bound to its
      // call id through the swap.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ text: 'all fetched' }));
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [
            { role: 'user', content: 'call fn' },
            {
              role: 'assistant',
              content: [
                { type: 'tool_use', id: 'call_1', name: 'get_a', input: { k: 'a' } },
                { type: 'tool_use', id: 'call_2', name: 'get_b', input: { k: 'b' } },
              ],
            },
            {
              role: 'user',
              content: [
                // First fan-out's tool_result blocks reversed.
                { type: 'tool_result', tool_use_id: 'call_2', content: '{"v":"b-result"}' },
                { type: 'tool_result', tool_use_id: 'call_1', content: '{"v":"a-result"}' },
              ],
            },
            {
              role: 'assistant',
              content: [
                { type: 'tool_use', id: 'call_3', name: 'get_c', input: { k: 'c' } },
                { type: 'tool_use', id: 'call_4', name: 'get_d', input: { k: 'd' } },
              ],
            },
            {
              role: 'user',
              content: [
                // Second fan-out already canonical.
                { type: 'tool_result', tool_use_id: 'call_3', content: '{"v":"c-result"}' },
                { type: 'tool_result', tool_use_id: 'call_4', content: '{"v":"d-result"}' },
              ],
            },
          ],
          max_tokens: 100,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.content[0].text).toBe('all fetched');

      // eslint-disable-next-line @typescript-eslint/unbound-method
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(startSpy).toHaveBeenCalledTimes(1);
      const [primedMessages] = startSpy.mock.calls[0] as [
        Array<{ role: string; content: string; toolCallId?: string }>,
      ];
      const toolMessages = primedMessages.filter((m) => m.role === 'tool');
      expect(toolMessages).toHaveLength(4);
      // First fan-out's tool block must now be in canonical sibling
      // order [call_1, call_2] and each content must track its id.
      expect(toolMessages[0]!.toolCallId).toBe('call_1');
      expect(toolMessages[0]!.content).toBe('{"v":"a-result"}');
      expect(toolMessages[1]!.toolCallId).toBe('call_2');
      expect(toolMessages[1]!.content).toBe('{"v":"b-result"}');
      // Second fan-out's tool block was already canonical.
      expect(toolMessages[2]!.toolCallId).toBe('call_3');
      expect(toolMessages[2]!.content).toBe('{"v":"c-result"}');
      expect(toolMessages[3]!.toolCallId).toBe('call_4');
      expect(toolMessages[3]!.content).toBe('{"v":"d-result"}');
    });

    it('passes a well-formed fan-out history through unchanged (canonicalization no-op)', async () => {
      // Happy-path sibling of the reversed-order test. A
      // well-formed fan-out with tool_result blocks already in
      // sibling order must flow through without reordering.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ text: 'both fetched' }));
      registry.register('test-model', mockModel);
      const { res, getStatus } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [
            { role: 'user', content: 'get weather and news' },
            {
              role: 'assistant',
              content: [
                { type: 'tool_use', id: 'call_a', name: 'get_weather', input: { city: 'SF' } },
                { type: 'tool_use', id: 'call_b', name: 'get_news', input: { q: 'tech' } },
              ],
            },
            {
              role: 'user',
              content: [
                { type: 'tool_result', tool_use_id: 'call_a', content: '{"temp":68}' },
                { type: 'tool_result', tool_use_id: 'call_b', content: '{"headlines":[]}' },
              ],
            },
          ],
          max_tokens: 100,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(200);
      // eslint-disable-next-line @typescript-eslint/unbound-method
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(startSpy).toHaveBeenCalledTimes(1);
      const [primedMessages] = startSpy.mock.calls[0] as [
        Array<{ role: string; content: string; toolCallId?: string }>,
      ];
      const toolMessages = primedMessages.filter((m) => m.role === 'tool');
      expect(toolMessages.map((m) => m.toolCallId)).toEqual(['call_a', 'call_b']);
      // Contents must line up with their original ids — a naive
      // swap that only moved ids without content would fail here.
      expect(toolMessages[0]!.content).toBe('{"temp":68}');
      expect(toolMessages[1]!.content).toBe('{"headlines":[]}');
    });

    it('returns 400 on a malformed fan-out missing a sibling tool_result', async () => {
      // The helper must reject a history with a declared
      // sibling that has no matching tool_result. Submitting only
      // `call_a`'s result when the assistant fanned out to both
      // [call_a, call_b] would orphan `call_b`. Reject with 400.
      //
      // The follow-up user turn carries a second `user` message
      // with plain text AFTER the tool_result turn — a legal
      // iter-23 shape (no mixing inside a single user turn) that
      // still trips the validator because the assistant fan-out
      // is never fully resolved.
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [
            { role: 'user', content: 'get both' },
            {
              role: 'assistant',
              content: [
                { type: 'tool_use', id: 'call_a', name: 'get_weather', input: { city: 'SF' } },
                { type: 'tool_use', id: 'call_b', name: 'get_news', input: { q: 'tech' } },
              ],
            },
            {
              role: 'user',
              content: [
                // Only call_a is resolved — call_b is missing.
                { type: 'tool_result', tool_use_id: 'call_a', content: '{"temp":68}' },
              ],
            },
            {
              role: 'user',
              content: 'any updates?',
            },
          ],
          max_tokens: 100,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('invalid_request_error');
      // Iter-23 finding 4: error vocabulary is Anthropic-flavored
      // on `/v1/messages`, so the text references `tool_result`
      // and `assistant turn with tool_use blocks`, not
      // `function_call_output` or `assistant fan-out`.
      expect(parsed.error.message).toMatch(/unresolved sibling tool calls|tool_result/);
      expect(parsed.error.message).not.toMatch(/function_call_output|\bcall_id\b/);
      // eslint-disable-next-line @typescript-eslint/unbound-method
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(startSpy).not.toHaveBeenCalled();
    });

    it('returns 400 on a tool_result referencing an unknown tool_use_id', async () => {
      // Binding a tool_result to a call id not declared by the
      // preceding assistant fan-out would silently flow the output
      // to the wrong place (or to nothing at all). Reject.
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [
            { role: 'user', content: 'get weather' },
            {
              role: 'assistant',
              content: [{ type: 'tool_use', id: 'call_a', name: 'get_weather', input: { city: 'SF' } }],
            },
            {
              role: 'user',
              content: [{ type: 'tool_result', tool_use_id: 'call_ghost', content: '{"temp":68}' }],
            },
          ],
          max_tokens: 100,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      // Iter-23 finding 4: Anthropic vocabulary — the validator
      // returns "assistant turn with tool_use blocks" and
      // `tool_use_id` for /v1/messages callers.
      expect(parsed.error.message).toMatch(/not declared by the preceding assistant turn with tool_use blocks/);
      expect(parsed.error.message).toMatch(/tool_use_id/);
      expect(parsed.error.message).not.toMatch(/function_call_output|\bcall_id\b/);
    });

    it('rejects mixed text + tool_result in a single user turn with 400', async () => {
      // Iter-23 finding 3: the iter-22 mapper silently hoisted
      // tool_result blocks to the front of a mixed turn and
      // emitted residual text/image content as a synthetic
      // trailing user message — lossy reordering of
      // caller-supplied blocks. The mapper now rejects the
      // mixed shape; this test pins the 400 + guarantees no
      // dispatch happens.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ text: 'should not fire' }));
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [
            { role: 'user', content: 'run both tools' },
            {
              role: 'assistant',
              content: [
                { type: 'tool_use', id: 'call_a', name: 'get_weather', input: { city: 'SF' } },
                { type: 'tool_use', id: 'call_b', name: 'get_news', input: { q: 'tech' } },
              ],
            },
            {
              role: 'user',
              content: [
                { type: 'text', text: 'here are outputs' },
                { type: 'tool_result', tool_use_id: 'call_b', content: '{"v":"b"}' },
                { type: 'tool_result', tool_use_id: 'call_a', content: '{"v":"a"}' },
              ],
            },
          ],
          max_tokens: 100,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('invalid_request_error');
      // Iter-26 finding 2: the mapper now accepts tool_result blocks
      // as a contiguous PREFIX of a user turn (followed by trailing
      // text/image), but still rejects the inverse shape where a
      // text/image block precedes a tool_result. This test pins the
      // non-prefix rejection.
      expect(parsed.error.message).toMatch(/tool_result blocks must appear as a contiguous prefix/i);
      // eslint-disable-next-line @typescript-eslint/unbound-method
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(startSpy).not.toHaveBeenCalled();
    });

    it('accepts split turns: tool_result-only user turn followed by a separate text user turn', async () => {
      // Positive counter-test for iter-23 finding 3: a caller
      // that splits the mixed turn into two legal user turns
      // (one carrying ONLY tool_result blocks, one carrying the
      // follow-up text) must dispatch cleanly. Reversed tool
      // order on the first turn verifies the canonicalization
      // pass still runs end-to-end.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ text: 'followup ok' }));
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [
            { role: 'user', content: 'run both tools' },
            {
              role: 'assistant',
              content: [
                { type: 'tool_use', id: 'call_a', name: 'get_weather', input: { city: 'SF' } },
                { type: 'tool_use', id: 'call_b', name: 'get_news', input: { q: 'tech' } },
              ],
            },
            {
              role: 'user',
              content: [
                // Reversed — canonicalization will reorder to [call_a, call_b].
                { type: 'tool_result', tool_use_id: 'call_b', content: '{"v":"b"}' },
                { type: 'tool_result', tool_use_id: 'call_a', content: '{"v":"a"}' },
              ],
            },
            { role: 'user', content: 'here are outputs' },
          ],
          max_tokens: 100,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.content[0].text).toBe('followup ok');

      // eslint-disable-next-line @typescript-eslint/unbound-method
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(startSpy).toHaveBeenCalledTimes(1);
      const [primedMessages] = startSpy.mock.calls[0] as [
        Array<{ role: string; content: string; toolCallId?: string; toolCalls?: Array<{ id: string }> }>,
      ];

      const assistantIdx = primedMessages.findIndex((m) => m.role === 'assistant');
      expect(assistantIdx).toBeGreaterThanOrEqual(0);

      // Canonicalization reordered [call_b, call_a] → [call_a, call_b].
      const afterAssistant1 = primedMessages[assistantIdx + 1]!;
      const afterAssistant2 = primedMessages[assistantIdx + 2]!;
      expect(afterAssistant1.role).toBe('tool');
      expect(afterAssistant2.role).toBe('tool');
      expect(afterAssistant1.toolCallId).toBe('call_a');
      expect(afterAssistant1.content).toBe('{"v":"a"}');
      expect(afterAssistant2.toolCallId).toBe('call_b');
      expect(afterAssistant2.content).toBe('{"v":"b"}');

      const afterTool = primedMessages[assistantIdx + 3]!;
      expect(afterTool.role).toBe('user');
      expect(afterTool.content).toContain('here are outputs');
    });

    it('primes tool_result.is_error=true content with a JSON envelope through the full /v1/messages dispatch', async () => {
      // Iter-24 finding 2 smoke test: the Anthropic mapper now
      // wraps `tool_result.is_error === true` content in a JSON
      // envelope — `{"is_error":true,"content":<original>}` —
      // instead of the iter-23 `[tool error] ` prefix. Exercise
      // the full /v1/messages handler end-to-end so the primed
      // history passed to `chatSessionStart` reflects the
      // envelope. Without the fix `ChatMessage.content` would
      // carry either the raw tool output (losing the flag) or
      // the ambiguous prefix (corrupting JSON payloads).
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ text: 'ack' }));
      registry.register('test-model', mockModel);
      const { res, getStatus } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [
            { role: 'user', content: 'call the tool' },
            {
              role: 'assistant',
              content: [{ type: 'tool_use', id: 'call_fail', name: 'get_weather', input: { city: 'SF' } }],
            },
            {
              role: 'user',
              content: [
                {
                  type: 'tool_result',
                  tool_use_id: 'call_fail',
                  content: 'boom: connection refused',
                  is_error: true,
                },
              ],
            },
          ],
          max_tokens: 100,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(200);

      // eslint-disable-next-line @typescript-eslint/unbound-method
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(startSpy).toHaveBeenCalledTimes(1);
      const [primedMessages] = startSpy.mock.calls[0] as [
        Array<{ role: string; content: string; toolCallId?: string }>,
      ];
      const toolMsg = primedMessages.find((m) => m.role === 'tool' && m.toolCallId === 'call_fail');
      expect(toolMsg).toBeDefined();
      expect(toolMsg!.content).toBe(JSON.stringify({ is_error: true, content: 'boom: connection refused' }));
      // Envelope content is valid JSON and round-trips cleanly.
      const parsed = JSON.parse(toolMsg!.content) as { is_error: boolean; content: string };
      expect(parsed.is_error).toBe(true);
      expect(parsed.content).toBe('boom: connection refused');
    });
  });

  // -----------------------------------------------------------------------
  // Hot-swap race guard inside the per-model execution mutex
  // -----------------------------------------------------------------------

  describe('in-mutex binding re-read (iter-25 finding 2)', () => {
    it('rejects a queued request when the binding is re-registered while the mutex holds a prior dispatch', async () => {
      // Iter-25 finding 2 regression: the Anthropic handler
      // captured `sessionReg = registry.getSessionRegistry(body.model)`
      // once and then waited on
      // `sessionReg.withExclusive(...)` with NO post-acquisition
      // binding check. A request queued behind a long decode on
      // the old registry would still execute through that stale
      // registry even if `registry.register(body.model, newModel)`
      // had already rebound the name mid-wait. Unlike
      // `/v1/responses`, the Anthropic path has no stored
      // identity check later to catch the mismatch — two
      // requests for the same model name could silently be
      // serviced by different underlying models based purely on
      // queue timing. The in-mutex re-read added for this
      // finding catches the drift and rejects 400.
      const registry = new ModelRegistry();
      const originalModel = createMockModel(makeChatResult({ text: 'original' }));
      const swappedModel = createMockModel(makeChatResult({ text: 'swapped' }));

      // Pin the blocker's `chatSessionStart` on an externally
      // controlled gate so we can choose exactly when it
      // resolves.
      let releaseBlocker!: () => void;
      const blockerGate = new Promise<void>((resolve) => {
        releaseBlocker = resolve;
      });
      (originalModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>).mockImplementationOnce(async () => {
        await blockerGate;
        return makeChatResult({ text: 'original' });
      });

      registry.register('race-model', originalModel);

      // Kick off the blocker. It acquires the mutex, calls
      // `chatSessionStart`, and parks on `blockerGate`.
      const { res: res1 } = createMockRes();
      const blockerDone = handleCreateMessage(
        res1,
        {
          model: 'race-model',
          messages: [{ role: 'user', content: 'blocking turn' }],
          max_tokens: 100,
        },
        registry,
      );

      // Yield so the blocker enters `withExclusive` and awaits
      // `chatSessionStart`.
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();

      // Fire the queued request. It will enter
      // `withExclusive` and park on the chain's `prev` promise
      // until the blocker releases the lock.
      const { res: res2, getStatus: getStatus2, getBody: getBody2 } = createMockRes();
      const queuedDone = handleCreateMessage(
        res2,
        {
          model: 'race-model',
          messages: [{ role: 'user', content: 'queued turn' }],
          max_tokens: 100,
        },
        registry,
      );

      // Yield so the queued request reaches the mutex await.
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();

      // Hot-swap the binding STRICTLY between the queued
      // request's pre-lock snapshot and the moment it wins the
      // mutex.
      registry.register('race-model', swappedModel);

      // Release the blocker so the mutex falls through to the
      // queued request.
      releaseBlocker();
      await blockerDone;
      await queuedDone;

      // The queued request was rejected 400 by the in-lock
      // guard, in the Anthropic error envelope.
      expect(getStatus2()).toBe(400);
      const err = JSON.parse(getBody2());
      expect(err.type).toBe('error');
      expect(err.error.type).toBe('invalid_request_error');
      expect(err.error.message).toContain('race-model');
      expect(err.error.message).toMatch(/binding changed/i);
      expect(err.error.message).toMatch(/queued behind the per-model execution mutex/i);

      // The swapped model must NOT have been dispatched — the
      // queued request's closure aborted before `getOrCreate`.
      // eslint-disable-next-line @typescript-eslint/unbound-method
      const swappedStart = swappedModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(swappedStart).not.toHaveBeenCalled();

      // Original model serviced the blocker exactly once and
      // was not re-invoked by the queued request.
      // eslint-disable-next-line @typescript-eslint/unbound-method
      const originalStart = originalModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(originalStart).toHaveBeenCalledTimes(1);
    });
  });

  // -----------------------------------------------------------------------
  // Iter-33 finding 3: wire-contract / visibility fixes on /v1/messages
  // -----------------------------------------------------------------------

  describe('transport visibility (iter-33 finding 3)', () => {
    it('non-streaming: JSON-mode async end-callback failure destroys the socket instead of emitting SSE', async () => {
      // Iter-33 finding 3 regression. The Anthropic handler is
      // stateless (no session to adopt) but its outer catch still
      // keyed "JSON vs SSE fallback" on `res.headersSent`. A JSON
      // request whose `res.end()` reported an async callback
      // failure — happens when the underlying socket breaks AFTER
      // Node queued the payload but BEFORE the flush completes —
      // would then emit an SSE frame INTO a `Content-Type:
      // application/json` body.
      //
      // The fix commits the wire format in `endJson` via
      // `responseMode = 'json'` and switches the outer catch to
      // branch on that. On a JSON-mode failure we destroy the
      // socket so the client sees a truncated JSON body rather
      // than a malformed document with unexpected MIME frames.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ text: 'hi' }));
      registry.register('test-model', mockModel);
      const { res, getBody, wasDestroyed } = createMockRes();

      // Poison the end callback. First call: report an async error
      // to the callback; the handler must see that as
      // responseBodyWritten = false and route to the outer catch.
      const originalEnd = res.end.bind(res);
      let endCallCount = 0;
      (res as unknown as { end: (...args: unknown[]) => unknown }).end = (
        chunk?: unknown,
        encodingOrCb?: unknown,
        maybeCb?: unknown,
      ) => {
        endCallCount++;
        if (endCallCount === 1) {
          const cb = typeof encodingOrCb === 'function' ? encodingOrCb : maybeCb;
          if (typeof cb === 'function') {
            queueMicrotask(() => (cb as (err: Error) => void)(new Error('simulated late socket failure')));
          }
          return res;
        }
        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
        return originalEnd(chunk as any, encodingOrCb as any, maybeCb as any);
      };

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        } as any,
        registry,
      );
      // Allow the queued microtask to fire.
      await new Promise((r) => setTimeout(r, 0));

      // Wire contract: socket destroyed, body contains NO SSE
      // frame. `event: error` or `data: ` in a JSON response is
      // exactly the corruption finding 3 prevents.
      expect(wasDestroyed()).toBe(true);
      const body = getBody();
      expect(body).not.toContain('event: error');
      expect(body).not.toMatch(/^data: /m);
    });

    it('streaming: early SSE write crash before any terminal emits an error frame and does not hang', async () => {
      // Iter-33 finding 3 regression on the streaming path. A
      // mid-decode `res.write` crash BEFORE any terminal
      // (`message_stop` / streaming `error`) used to land in the
      // old outer-catch branch that keyed on `headersSent` — which
      // would STILL try to emit an SSE error frame but would NOT
      // know whether a terminal had already arrived, so a race
      // could leak a double terminal. The fix gates
      // `terminalEmitted` on the actual terminal write callback
      // and branches the outer catch on `responseMode` +
      // `terminalEmitted`, so the fallback frame only fires when
      // the client has not already observed a terminal.
      const registry = new ModelRegistry();
      async function* stream() {
        yield { done: false, text: 'never emitted', isReasoning: false };
      }
      const mockModel = {
        chatSessionStart: vi.fn().mockRejectedValue(new Error('should not be called')),
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('should not be called')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('should not be called')),
        chatStreamSessionStart: vi.fn(() => stream()),
        chatStreamSessionContinue: vi.fn(() => stream()),
        chatStreamSessionContinueTool: vi.fn(() => stream()),
        resetCaches: vi.fn(),
      } as unknown as SessionCapableModel;
      registry.register('test-model', mockModel);

      const { res, getBody } = createMockRes();

      // First `res.write` throws (the `message_start` SSE frame
      // never lands). Subsequent writes succeed so the fallback
      // error frame from the outer catch can be emitted.
      let writeCallCount = 0;
      const originalWrite = res.write.bind(res);
      res.write = ((chunk: Uint8Array | string, ...rest: unknown[]) => {
        writeCallCount++;
        if (writeCallCount === 1) {
          throw new Error('simulated early SSE write crash');
        }
        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
        return (originalWrite as unknown as (...a: unknown[]) => boolean)(chunk, ...rest);
      }) as ServerResponse['write'];

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          stream: true,
        } as any,
        registry,
      );

      // The outer catch ran: a best-effort `error` SSE frame
      // landed on the wire AFTER the first write crashed. The
      // exact payload follows Anthropic's error envelope.
      const body = getBody();
      expect(body).toContain('event: error');
      expect(body).toContain('api_error');
    });
  });
});
