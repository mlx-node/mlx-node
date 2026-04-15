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
} {
  const { Writable } = require('node:stream');
  let status = 200;
  let body = '';
  const headers: Record<string, string | string[]> = {};

  const writable = new Writable({
    write(chunk: Uint8Array | string, _encoding: string, callback: () => void) {
      body += chunk.toString();
      callback();
    },
  });

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
  writable.end = (chunk?: string | Uint8Array, ...args: any[]) => {
    if (chunk) body += chunk.toString();
    writable.headersSent = true;
    origEnd(undefined, ...args);
    return writable;
  };

  return {
    res: writable as unknown as ServerResponse,
    getStatus: () => status,
    getBody: () => body,
    getHeaders: () => headers,
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

    it('does not cache a session when a streaming turn emits a done event with finishReason error', async () => {
      // Finding 3 parity: the streaming commit path must not leak a
      // resumable cached session entry when the final chunk reports
      // `finishReason: 'error'`. The Anthropic endpoint is stateless
      // and never adopts, but this test pins that invariant and also
      // verifies the endpoint emits SSE terminal events cleanly under
      // an error finish instead of throwing.
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

      // The endpoint drained the stream without throwing. Minimum SSE
      // expectation: at least one terminal event was written.
      const events = parseSSE(getBody());
      const terminal = events.find((e) => e.event === 'message_stop' || e.event === 'error');
      expect(terminal).toBeDefined();

      // And the registry stayed empty — no adopt call leaked.
      const sessionReg = registry.getSessionRegistry('test-model');
      expect(sessionReg!.size).toBe(0);
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
      expect(parsed.error.message).toMatch(/cannot mix tool_result blocks with text or image blocks/i);
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
});
