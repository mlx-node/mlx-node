import type { ServerResponse } from 'node:http';

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

function createMockModel() {
  return {
    chat: vi.fn().mockResolvedValue({
      text: 'Hello!',
      toolCalls: [],
      thinking: null,
      numTokens: 10,
      promptTokens: 5,
      reasoningTokens: 0,
      finishReason: 'stop',
      rawText: 'Hello!',
      performance: undefined,
    }),
  };
}

function createMockStreamModel(streamEvents: Array<Record<string, unknown>>) {
  return {
    chat: vi.fn().mockRejectedValue(new Error('Should use chatStream')),
    async *chatStream(_messages: unknown, _config: unknown) {
      for (const event of streamEvents) {
        yield event;
      }
    },
  };
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
      const mockModel = {
        chat: vi.fn().mockResolvedValue({
          text: 'The answer is 42.',
          toolCalls: [],
          thinking: 'Let me think about this...',
          numTokens: 15,
          promptTokens: 8,
          reasoningTokens: 5,
          finishReason: 'stop',
          rawText: 'The answer is 42.',
          performance: undefined,
        }),
      };
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
      const mockModel = {
        chat: vi.fn().mockResolvedValue({
          text: '',
          toolCalls: [
            {
              status: 'ok',
              id: 'toolu_abc123',
              name: 'get_weather',
              arguments: '{"location":"San Francisco"}',
            },
          ],
          thinking: null,
          numTokens: 20,
          promptTokens: 10,
          reasoningTokens: 0,
          finishReason: 'stop',
          rawText: '',
          performance: undefined,
        }),
      };
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
  // Streaming (simulated fallback)
  // -----------------------------------------------------------------------

  describe('streaming (simulated)', () => {
    it('simulates streaming from chat() when chatStream is not available', async () => {
      const registry = new ModelRegistry();
      const mockModel = {
        chat: vi.fn().mockResolvedValue({
          text: 'Simulated response',
          toolCalls: [],
          thinking: null,
          numTokens: 8,
          promptTokens: 4,
          reasoningTokens: 0,
          finishReason: 'stop',
          rawText: 'Simulated response',
          performance: undefined,
        }),
      };
      // No chatStream method -> simulated streaming
      registry.register('test-model', mockModel);
      const { res, getBody, getHeaders } = createMockRes();

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

      expect(getHeaders()['content-type']).toBe('text/event-stream');

      const events = parseSSE(getBody());

      // message_start
      expect(events[0].event).toBe('message_start');

      // content_block_start for text
      const textStart = events.find(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStart).toBeDefined();

      // text delta
      const textDelta = events.find(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      expect(textDelta).toBeDefined();
      expect((textDelta!.data['delta'] as any).text).toBe('Simulated response');

      // content_block_stop
      const cbStop = events.find((e) => e.event === 'content_block_stop');
      expect(cbStop).toBeDefined();

      // message_delta + message_stop
      expect(events.find((e) => e.event === 'message_delta')).toBeDefined();
      expect(events.find((e) => e.event === 'message_stop')).toBeDefined();
    });

    it('simulates streaming with thinking + text + tool_use', async () => {
      const registry = new ModelRegistry();
      const mockModel = {
        chat: vi.fn().mockResolvedValue({
          text: 'I will look that up.',
          toolCalls: [
            {
              status: 'ok',
              id: 'toolu_sim',
              name: 'lookup',
              arguments: '{"key":"value"}',
            },
          ],
          thinking: 'Reasoning here',
          numTokens: 20,
          promptTokens: 10,
          reasoningTokens: 5,
          finishReason: 'stop',
          rawText: 'I will look that up.',
          performance: undefined,
        }),
      };
      registry.register('test-model', mockModel);
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Look up X' }],
          max_tokens: 200,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());

      // Check block types in order
      const blockStarts = events.filter((e) => e.event === 'content_block_start');
      expect(blockStarts).toHaveLength(3);
      expect((blockStarts[0].data['content_block'] as any).type).toBe('thinking');
      expect(blockStarts[0].data['index']).toBe(0);
      expect((blockStarts[1].data['content_block'] as any).type).toBe('text');
      expect(blockStarts[1].data['index']).toBe(1);
      expect((blockStarts[2].data['content_block'] as any).type).toBe('tool_use');
      expect(blockStarts[2].data['index']).toBe(2);

      // Verify stop_reason is tool_use
      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('tool_use');
    });
  });

  // -----------------------------------------------------------------------
  // Error handling
  // -----------------------------------------------------------------------

  describe('error handling', () => {
    it('returns 500 on model.chat() error (non-streaming)', async () => {
      const registry = new ModelRegistry();
      const mockModel = {
        chat: vi.fn().mockRejectedValue(new Error('Model crashed')),
      };
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

    it('emits error SSE event on streaming error after headers sent', async () => {
      const registry = new ModelRegistry();
      const mockModel = {
        chat: vi.fn(),
        async *chatStream() {
          yield { text: 'partial', done: false, isReasoning: false };
          throw new Error('Stream crashed');
        },
      };
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
});
