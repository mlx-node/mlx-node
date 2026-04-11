import type { IncomingMessage, ServerResponse } from 'node:http';
import { Writable } from 'node:stream';

import { createHandler, ModelRegistry } from '@mlx-node/server';
import { describe, expect, it, vi } from 'vite-plus/test';

// ---------------------------------------------------------------------------
// Mock helpers
// ---------------------------------------------------------------------------

/**
 * Create a minimal mock IncomingMessage that emits a JSON body.
 */
function createMockReq(method: string, url: string, body?: object): IncomingMessage {
  const { Readable } = require('node:stream');
  const req = new Readable({
    read() {
      if (body) {
        this.push(JSON.stringify(body));
      }
      this.push(null);
    },
  }) as IncomingMessage;
  req.method = method;
  req.url = url;
  req.headers = { 'content-type': 'application/json', host: 'localhost:3000' };
  (req as any).httpVersion = '1.1';
  (req as any).httpVersionMajor = 1;
  (req as any).httpVersionMinor = 1;
  return req;
}

class MockServerResponse extends Writable {
  headersSent = true;

  writeHead(_s: number, _h?: Record<string, string>) {}
  setHeader(_name: string, _value: string) {}
  getHeader(_name: string) {}
}

/**
 * Capture writes to a ServerResponse via a simple writable mock.
 */
function createMockRes(): {
  res: ServerResponse;
  getStatus: () => number;
  getBody: () => string;
  getHeaders: () => Record<string, string | string[]>;
  waitForEnd: () => Promise<void>;
} {
  let status = 200;
  let body = '';
  const headers: Record<string, string | string[]> = {};
  let endResolve: () => void;
  const endPromise = new Promise<void>((resolve) => {
    endResolve = resolve;
  });

  const writable = new MockServerResponse({
    write(chunk: Uint8Array | string, _encoding: string, callback: () => void) {
      body += chunk.toString();
      callback();
    },
  });

  // Attach ServerResponse-like methods
  writable.writeHead = (s: number, h?: Record<string, string>) => {
    status = s;
    if (h) {
      for (const [k, v] of Object.entries(h)) {
        headers[k.toLowerCase()] = v;
      }
    }
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
  // @ts-expect-error
  writable.end = (chunk: string | Uint8Array, encoding: BufferEncoding, cb?: () => void) => {
    if (chunk) body += chunk.toString();
    writable.headersSent = true;
    origEnd(undefined, encoding, cb);
    endResolve();
    return writable;
  };

  return {
    res: writable as unknown as ServerResponse,
    getStatus: () => status,
    getBody: () => body,
    getHeaders: () => headers,
    waitForEnd: () => endPromise,
  };
}

function createMockModel() {
  return {
    chat: vi.fn().mockResolvedValue({
      text: 'Hello!',
      toolCalls: [],
      thinking: undefined,
      numTokens: 5,
      promptTokens: 10,
      reasoningTokens: 0,
      finishReason: 'stop',
      rawText: 'Hello!',
      performance: undefined,
    }),
  };
}

/**
 * Create a mock model that supports chatStream() with configurable events.
 * Each event in `streamEvents` is yielded by the async generator.
 */
function createMockStreamModel(streamEvents: Array<Record<string, unknown>>) {
  return {
    chat: vi.fn().mockResolvedValue({
      text: '',
      toolCalls: [],
      thinking: undefined,
      numTokens: 0,
      promptTokens: 0,
      reasoningTokens: 0,
      finishReason: 'stop',
      rawText: '',
      performance: undefined,
    }),
    async *chatStream() {
      for (const event of streamEvents) {
        yield event;
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('createHandler', () => {
  describe('POST /v1/responses', () => {
    it('returns 200 JSON response with simple input', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      registry.register('test-model', mockModel);

      const handler = createHandler(registry);
      const req = createMockReq('POST', '/v1/responses', {
        model: 'test-model',
        input: 'Hello',
      });
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.object).toBe('response');
      expect(parsed.status).toBe('completed');
      expect(parsed.model).toBe('test-model');
      expect(parsed.output_text).toBe('Hello!');
      expect(parsed.output).toHaveLength(1);
      expect(parsed.output[0].type).toBe('message');
    });

    it('returns 400 when model is missing', async () => {
      const registry = new ModelRegistry();
      const handler = createHandler(registry);
      const req = createMockReq('POST', '/v1/responses', {
        input: 'Hello',
      });
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.type).toBe('invalid_request_error');
      expect(parsed.error.message).toContain('model');
    });

    it('returns 400 when input is missing', async () => {
      const registry = new ModelRegistry();
      const handler = createHandler(registry);
      const req = createMockReq('POST', '/v1/responses', {
        model: 'test-model',
      });
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.message).toContain('input');
    });

    it('returns 400 when input is not a string or array', async () => {
      const registry = new ModelRegistry();
      registry.register('test-model', createMockModel());
      const handler = createHandler(registry);
      const req = createMockReq('POST', '/v1/responses', {
        model: 'test-model',
        input: 42,
      });
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.type).toBe('invalid_request_error');
      expect(parsed.error.message).toContain('string or an array');
    });

    it('returns 400 when input array contains null items', async () => {
      const registry = new ModelRegistry();
      registry.register('test-model', createMockModel());
      const handler = createHandler(registry);
      const req = createMockReq('POST', '/v1/responses', {
        model: 'test-model',
        input: [null],
      });
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.type).toBe('invalid_request_error');
      expect(parsed.error.message).toContain('non-null object');
    });

    it('returns 404 when model is not found', async () => {
      const registry = new ModelRegistry();
      const handler = createHandler(registry);
      const req = createMockReq('POST', '/v1/responses', {
        model: 'nonexistent',
        input: 'Hello',
      });
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(404);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.type).toBe('not_found_error');
      expect(parsed.error.message).toContain('nonexistent');
    });

    it('returns 404 when previous_response_id is not found in store', async () => {
      const registry = new ModelRegistry();
      registry.register('test-model', createMockModel());

      const mockStore = {
        getChain: vi.fn().mockRejectedValue(new Error('not found')),
        save: vi.fn(),
        cleanup: vi.fn(),
      };

      const handler = createHandler(registry, { store: mockStore as any });
      const req = createMockReq('POST', '/v1/responses', {
        model: 'test-model',
        input: 'Hello',
        previous_response_id: 'resp_missing',
      });
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(404);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.type).toBe('not_found_error');
      expect(parsed.error.message).toContain('resp_missing');
      expect(parsed.error.message).toContain('not found or expired');
    });

    it('does not persist instructions as input messages in store', async () => {
      const registry = new ModelRegistry();
      registry.register('test-model', createMockModel());

      let storedRecord: any = null;
      const mockStore = {
        getChain: vi.fn(),
        store: vi.fn().mockImplementation((record: any) => {
          storedRecord = record;
          return Promise.resolve();
        }),
        cleanupExpired: vi.fn(),
      };

      const handler = createHandler(registry, { store: mockStore as any });
      const req = createMockReq('POST', '/v1/responses', {
        model: 'test-model',
        input: 'Hello',
        instructions: 'Be brief',
      });
      const { res, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(mockStore.store).toHaveBeenCalledTimes(1);
      const inputMessages = JSON.parse(storedRecord.inputJson);
      // Instructions should NOT be in the stored input messages
      expect(inputMessages).toHaveLength(1);
      expect(inputMessages[0].role).toBe('user');
      expect(inputMessages[0].content).toBe('Hello');
    });

    it('passes mapped messages and config to model.chat', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      registry.register('test-model', mockModel);

      const handler = createHandler(registry);
      const req = createMockReq('POST', '/v1/responses', {
        model: 'test-model',
        input: 'Hello',
        temperature: 0.7,
        max_output_tokens: 100,
      });
      const { res, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(mockModel.chat).toHaveBeenCalledTimes(1);
      const [messages, config] = mockModel.chat.mock.calls[0];
      expect(messages).toEqual([{ role: 'user', content: 'Hello' }]);
      expect(config.temperature).toBe(0.7);
      expect(config.maxNewTokens).toBe(100);
    });
  });

  describe('GET /v1/models', () => {
    it('returns model list', async () => {
      const registry = new ModelRegistry();
      registry.register('model-a', createMockModel());
      registry.register('model-b', createMockModel());

      const handler = createHandler(registry);
      const req = createMockReq('GET', '/v1/models');
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.object).toBe('list');
      expect(parsed.data).toHaveLength(2);
      expect(parsed.data[0].id).toBe('model-a');
      expect(parsed.data[1].id).toBe('model-b');
    });
  });

  describe('routing', () => {
    it('returns 404 for unknown path', async () => {
      const registry = new ModelRegistry();
      const handler = createHandler(registry);
      const req = createMockReq('POST', '/v1/unknown');
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(404);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.type).toBe('not_found_error');
    });

    it('returns 405 for GET /v1/responses', async () => {
      const registry = new ModelRegistry();
      const handler = createHandler(registry);
      const req = createMockReq('GET', '/v1/responses');
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(405);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.message).toBe('Method not allowed');
    });

    it('returns 405 for POST /v1/models', async () => {
      const registry = new ModelRegistry();
      const handler = createHandler(registry);
      const req = createMockReq('POST', '/v1/models');
      const { res, getStatus, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(405);
    });
  });

  describe('CORS', () => {
    it('handles OPTIONS preflight with 204 and correct headers', async () => {
      const registry = new ModelRegistry();
      const handler = createHandler(registry);
      const req = createMockReq('OPTIONS', '/v1/responses');
      const { res, getStatus, getHeaders, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(204);
      expect(getHeaders()['access-control-allow-origin']).toBe('*');
      expect(getHeaders()['access-control-allow-methods']).toBe('GET, POST, OPTIONS');
      expect(getHeaders()['access-control-allow-headers']).toBe(
        'Content-Type, Authorization, x-api-key, anthropic-version',
      );
    });

    it('includes CORS headers on normal responses', async () => {
      const registry = new ModelRegistry();
      registry.register('test-model', createMockModel());

      const handler = createHandler(registry);
      const req = createMockReq('POST', '/v1/responses', {
        model: 'test-model',
        input: 'Hello',
      });
      const { res, getHeaders, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getHeaders()['access-control-allow-origin']).toBe('*');
    });

    it('does not include CORS headers when cors is disabled', async () => {
      const registry = new ModelRegistry();
      registry.register('test-model', createMockModel());

      const handler = createHandler(registry, { cors: false });
      const req = createMockReq('POST', '/v1/responses', {
        model: 'test-model',
        input: 'Hello',
      });
      const { res, getHeaders, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getHeaders()['access-control-allow-origin']).toBeUndefined();
    });
  });

  describe('health check', () => {
    it('returns 200 ok for /health', async () => {
      const registry = new ModelRegistry();
      const handler = createHandler(registry);
      const req = createMockReq('GET', '/health');
      const { res, getStatus, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.status).toBe('ok');
    });
  });

  describe('streaming with tool calls', () => {
    it('does not leak <tool_call> markup in text deltas', async () => {
      // Simulate a model that streams normal text, then tool-call markup, then final event
      const streamEvents = [
        { done: false, text: 'Let me ', isReasoning: false },
        { done: false, text: 'look that up.', isReasoning: false },
        // Tool-call markup starts leaking
        { done: false, text: '\n<tool_call>\n', isReasoning: false },
        { done: false, text: '{"name": "get_weather",', isReasoning: false },
        { done: false, text: ' "arguments": {"city": "SF"}}', isReasoning: false },
        { done: false, text: '\n</tool_call>', isReasoning: false },
        // Final event with parsed results
        {
          done: true,
          text: 'Let me look that up.',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_123',
              name: 'get_weather',
              arguments: '{"city": "SF"}',
              status: 'ok',
              rawContent: '',
            },
          ],
          thinking: null,
          numTokens: 20,
          promptTokens: 10,
          reasoningTokens: 0,
          rawText:
            'Let me look that up.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>',
        },
      ];

      const registry = new ModelRegistry();
      const mockModel = createMockStreamModel(streamEvents);
      registry.register('stream-model', mockModel);

      const handler = createHandler(registry);
      const reqBody = {
        model: 'stream-model',
        input: 'What is the weather in SF?',
        stream: true,
      };
      const req = createMockReq('POST', '/v1/responses', reqBody);
      const { res, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      const body = getBody();

      // Parse all SSE events
      const events: Array<{ event: string; data: Record<string, unknown> }> = [];
      for (const line of body.split('\n')) {
        if (line.startsWith('event: ')) {
          events.push({ event: line.slice(7), data: {} });
        } else if (line.startsWith('data: ') && events.length > 0) {
          events[events.length - 1].data = JSON.parse(line.slice(6));
        }
      }

      // Collect all text deltas
      const textDeltas = events
        .filter((e) => e.event === 'response.output_text.delta')
        .map((e) => e.data.delta as string);

      // Text deltas should NOT contain tool-call markup
      const allDeltaText = textDeltas.join('');
      expect(allDeltaText).not.toContain('<tool_call>');
      expect(allDeltaText).not.toContain('</tool_call>');
      expect(allDeltaText).not.toContain('get_weather');

      // The clean text deltas should be present
      expect(allDeltaText).toContain('Let me ');
      expect(allDeltaText).toContain('look that up.');

      // There should be a function_call item in the completed response
      const completedEvent = events.find((e) => e.event === 'response.completed');
      expect(completedEvent).toBeDefined();
      const response = completedEvent!.data.response as Record<string, unknown>;
      const output = response.output as Array<Record<string, unknown>>;

      // Should have a message item and a function_call item
      const messageItems = output.filter((i) => i.type === 'message');
      const fcItems = output.filter((i) => i.type === 'function_call');
      expect(messageItems).toHaveLength(1);
      expect(fcItems).toHaveLength(1);
      expect(fcItems[0].name).toBe('get_weather');

      // The message content should be clean (no markup)
      const msgContent = (messageItems[0].content as Array<Record<string, unknown>>)[0];
      expect(msgContent.text).toBe('Let me look that up.');
    });

    it('skips message item when final text is empty and tool calls are present', async () => {
      // Model immediately produces tool-call markup, no visible text
      const streamEvents = [
        { done: false, text: '<tool_call>\n', isReasoning: false },
        { done: false, text: '{"name": "search", "arguments": {"q": "test"}}', isReasoning: false },
        { done: false, text: '\n</tool_call>', isReasoning: false },
        {
          done: true,
          text: '', // No clean text
          finishReason: 'tool_calls',
          toolCalls: [{ id: 'call_456', name: 'search', arguments: '{"q": "test"}', status: 'ok', rawContent: '' }],
          thinking: null,
          numTokens: 15,
          promptTokens: 8,
          reasoningTokens: 0,
          rawText: '<tool_call>\n{"name": "search", "arguments": {"q": "test"}}\n</tool_call>',
        },
      ];

      const registry = new ModelRegistry();
      const mockModel = createMockStreamModel(streamEvents);
      registry.register('stream-model', mockModel);

      const handler = createHandler(registry);
      const reqBody = {
        model: 'stream-model',
        input: 'Search for test',
        stream: true,
      };
      const req = createMockReq('POST', '/v1/responses', reqBody);
      const { res, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      const body = getBody();

      // Parse SSE events
      const events: Array<{ event: string; data: Record<string, unknown> }> = [];
      for (const line of body.split('\n')) {
        if (line.startsWith('event: ')) {
          events.push({ event: line.slice(7), data: {} });
        } else if (line.startsWith('data: ') && events.length > 0) {
          events[events.length - 1].data = JSON.parse(line.slice(6));
        }
      }

      // No text deltas should have been emitted at all
      const textDeltas = events.filter((e) => e.event === 'response.output_text.delta');
      expect(textDeltas).toHaveLength(0);

      // Completed response should have only function_call items, no message items
      const completedEvent = events.find((e) => e.event === 'response.completed');
      expect(completedEvent).toBeDefined();
      const response = completedEvent!.data.response as Record<string, unknown>;
      const output = response.output as Array<Record<string, unknown>>;

      const messageItems = output.filter((i) => i.type === 'message');
      const fcItems = output.filter((i) => i.type === 'function_call');
      expect(messageItems).toHaveLength(0);
      expect(fcItems).toHaveLength(1);
      expect(fcItems[0].name).toBe('search');
    });

    it('does not emit whitespace-only prefix delta when whitespace and <tool_call> arrive in same chunk', async () => {
      // Model emits "\n<tool_call>\n..." in a single chunk — a common pattern where the
      // model puts a newline before the tool-call markup. The cleanPrefix ("\n") is
      // whitespace-only and must not create a dangling message item.
      const streamEvents = [
        // Single chunk: newline immediately followed by the tool-call opening tag
        { done: false, text: '\n<tool_call>\n', isReasoning: false },
        { done: false, text: '{"name": "get_time", "arguments": {}}', isReasoning: false },
        { done: false, text: '\n</tool_call>', isReasoning: false },
        // Final event: empty parsed text (only tool call output)
        {
          done: true,
          text: '',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_ws',
              name: 'get_time',
              arguments: '{}',
              status: 'ok',
              rawContent: '',
            },
          ],
          thinking: null,
          numTokens: 12,
          promptTokens: 8,
          reasoningTokens: 0,
          rawText: '\n<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>',
        },
      ];

      const registry = new ModelRegistry();
      const mockModel = createMockStreamModel(streamEvents);
      registry.register('stream-model', mockModel);

      const handler = createHandler(registry);
      const reqBody = {
        model: 'stream-model',
        input: 'What time is it?',
        stream: true,
      };
      const req = createMockReq('POST', '/v1/responses', reqBody);
      const { res, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      const body = getBody();

      // Parse all SSE events
      const events: Array<{ event: string; data: Record<string, unknown> }> = [];
      for (const line of body.split('\n')) {
        if (line.startsWith('event: ')) {
          events.push({ event: line.slice(7), data: {} });
        } else if (line.startsWith('data: ') && events.length > 0) {
          events[events.length - 1].data = JSON.parse(line.slice(6));
        }
      }

      // 1. No text deltas at all (the "\n" prefix is whitespace-only, must not be emitted)
      const textDeltas = events.filter((e) => e.event === 'response.output_text.delta');
      expect(textDeltas).toHaveLength(0);

      // 2. Completed response must have only function_call items, no message items
      const completedEvent = events.find((e) => e.event === 'response.completed');
      expect(completedEvent).toBeDefined();
      const response = completedEvent!.data.response as Record<string, unknown>;
      const output = response.output as Array<Record<string, unknown>>;

      const messageItems = output.filter((i) => i.type === 'message');
      const fcItems = output.filter((i) => i.type === 'function_call');
      expect(messageItems).toHaveLength(0);
      expect(fcItems).toHaveLength(1);
      expect(fcItems[0].name).toBe('get_time');

      // 3. Every output_item.added event must have a corresponding output_item.done event
      const addedItemIds = events
        .filter((e) => e.event === 'response.output_item.added')
        .map((e) => (e.data.item as Record<string, unknown>).id as string);
      const doneItemIds = events
        .filter((e) => e.event === 'response.output_item.done')
        .map((e) => (e.data.item as Record<string, unknown>).id as string);
      for (const id of addedItemIds) {
        expect(doneItemIds).toContain(id);
      }
    });

    it('gracefully closes dangling message item when whitespace arrives in separate chunk before <tool_call>', async () => {
      // Model emits "\n" in one chunk, then "<tool_call>..." in the next. The "\n" chunk
      // gets emitted as a delta (we cannot suppress it without look-ahead). When the tool
      // call tag arrives in the next chunk, suppressTextDeltas is set. At finalization
      // the skipMessageItem branch must send done events to close the dangling item so
      // clients do not see it stuck in-progress, AND the completed response must not
      // contain that message item.
      const streamEvents = [
        // First chunk is just a newline — arrives before the tool-call tag
        { done: false, text: '\n', isReasoning: false },
        // Second chunk contains the tool-call opening tag
        { done: false, text: '<tool_call>\n', isReasoning: false },
        { done: false, text: '{"name": "get_time", "arguments": {}}', isReasoning: false },
        { done: false, text: '\n</tool_call>', isReasoning: false },
        // Final event: empty parsed text (only tool call output)
        {
          done: true,
          text: '',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_ws2',
              name: 'get_time',
              arguments: '{}',
              status: 'ok',
              rawContent: '',
            },
          ],
          thinking: null,
          numTokens: 13,
          promptTokens: 8,
          reasoningTokens: 0,
          rawText: '\n<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>',
        },
      ];

      const registry = new ModelRegistry();
      const mockModel = createMockStreamModel(streamEvents);
      registry.register('stream-model', mockModel);

      const handler = createHandler(registry);
      const reqBody = {
        model: 'stream-model',
        input: 'What time is it?',
        stream: true,
      };
      const req = createMockReq('POST', '/v1/responses', reqBody);
      const { res, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      const body = getBody();

      // Parse all SSE events
      const events: Array<{ event: string; data: Record<string, unknown> }> = [];
      for (const line of body.split('\n')) {
        if (line.startsWith('event: ')) {
          events.push({ event: line.slice(7), data: {} });
        } else if (line.startsWith('data: ') && events.length > 0) {
          events[events.length - 1].data = JSON.parse(line.slice(6));
        }
      }

      // 1. Completed response must have only function_call items, no message items
      const completedEvent = events.find((e) => e.event === 'response.completed');
      expect(completedEvent).toBeDefined();
      const response = completedEvent!.data.response as Record<string, unknown>;
      const output = response.output as Array<Record<string, unknown>>;

      const messageItems = output.filter((i) => i.type === 'message');
      const fcItems = output.filter((i) => i.type === 'function_call');
      expect(messageItems).toHaveLength(0);
      expect(fcItems).toHaveLength(1);
      expect(fcItems[0].name).toBe('get_time');

      // 2. Every output_item.added event must have a corresponding output_item.done event
      //    (no dangling items stuck in-progress)
      const addedItemIds = events
        .filter((e) => e.event === 'response.output_item.added')
        .map((e) => (e.data.item as Record<string, unknown>).id as string);
      const doneItemIds = events
        .filter((e) => e.event === 'response.output_item.done')
        .map((e) => (e.data.item as Record<string, unknown>).id as string);
      for (const id of addedItemIds) {
        expect(doneItemIds).toContain(id);
      }
    });

    it('streams text deltas normally when no tool calls are present', async () => {
      const streamEvents = [
        { done: false, text: 'Hello', isReasoning: false },
        { done: false, text: ' world!', isReasoning: false },
        {
          done: true,
          text: 'Hello world!',
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 3,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: 'Hello world!',
        },
      ];

      const registry = new ModelRegistry();
      const mockModel = createMockStreamModel(streamEvents);
      registry.register('stream-model', mockModel);

      const handler = createHandler(registry);
      const reqBody = {
        model: 'stream-model',
        input: 'Say hello',
        stream: true,
      };
      const req = createMockReq('POST', '/v1/responses', reqBody);
      const { res, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      const body = getBody();

      // Parse SSE events
      const events: Array<{ event: string; data: Record<string, unknown> }> = [];
      for (const line of body.split('\n')) {
        if (line.startsWith('event: ')) {
          events.push({ event: line.slice(7), data: {} });
        } else if (line.startsWith('data: ') && events.length > 0) {
          events[events.length - 1].data = JSON.parse(line.slice(6));
        }
      }

      // All text deltas should be present
      const textDeltas = events
        .filter((e) => e.event === 'response.output_text.delta')
        .map((e) => e.data.delta as string);
      expect(textDeltas).toEqual(['Hello', ' world!']);

      // output_text.done should have the final text
      const textDone = events.find((e) => e.event === 'response.output_text.done');
      expect(textDone).toBeDefined();
      expect(textDone!.data.text).toBe('Hello world!');
    });

    it('does not leak markup when <tool_call> is split across chunks', async () => {
      // The tag '<tool_call>' is split: first chunk ends with '<tool', second starts with '_call>'
      const streamEvents = [
        { done: false, text: 'Looking up', isReasoning: false },
        { done: false, text: '.\n<tool', isReasoning: false },
        { done: false, text: '_call>\n{"name": "get_weather"', isReasoning: false },
        { done: false, text: ', "arguments": {"city": "SF"}}', isReasoning: false },
        { done: false, text: '\n</tool_call>', isReasoning: false },
        {
          done: true,
          text: 'Looking up.',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_split',
              name: 'get_weather',
              arguments: '{"city": "SF"}',
              status: 'ok',
              rawContent: '',
            },
          ],
          thinking: null,
          numTokens: 18,
          promptTokens: 8,
          reasoningTokens: 0,
          rawText: 'Looking up.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>',
        },
      ];

      const registry = new ModelRegistry();
      const mockModel = createMockStreamModel(streamEvents);
      registry.register('stream-model', mockModel);

      const handler = createHandler(registry);
      const reqBody = {
        model: 'stream-model',
        input: 'Weather in SF?',
        stream: true,
      };
      const req = createMockReq('POST', '/v1/responses', reqBody);
      const { res, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      const body = getBody();

      // Parse all SSE events
      const events: Array<{ event: string; data: Record<string, unknown> }> = [];
      for (const line of body.split('\n')) {
        if (line.startsWith('event: ')) {
          events.push({ event: line.slice(7), data: {} });
        } else if (line.startsWith('data: ') && events.length > 0) {
          events[events.length - 1].data = JSON.parse(line.slice(6));
        }
      }

      // Collect all text deltas
      const textDeltas = events
        .filter((e) => e.event === 'response.output_text.delta')
        .map((e) => e.data.delta as string);

      const allDeltaText = textDeltas.join('');

      // No raw markup should appear
      expect(allDeltaText).not.toContain('<tool_call>');
      expect(allDeltaText).not.toContain('<tool');
      expect(allDeltaText).not.toContain('get_weather');

      // Clean text should be emitted
      expect(allDeltaText).toContain('Looking up');

      // Function call should still be present in output
      const completedEvent = events.find((e) => e.event === 'response.completed');
      expect(completedEvent).toBeDefined();
      const response = completedEvent!.data.response as Record<string, unknown>;
      const output = response.output as Array<Record<string, unknown>>;
      const fcItems = output.filter((i) => i.type === 'function_call');
      expect(fcItems).toHaveLength(1);
      expect(fcItems[0].name).toBe('get_weather');
    });

    it('flushes pending text as delta when stream ends without tool calls', async () => {
      // Text that ends with a partial prefix of '<tool_call>' (e.g., ends with '<')
      // but the stream finishes without any actual tool call
      const streamEvents = [
        { done: false, text: 'Value is 5 <', isReasoning: false },
        { done: false, text: ' 10', isReasoning: false },
        {
          done: true,
          text: 'Value is 5 < 10',
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 6,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: 'Value is 5 < 10',
        },
      ];

      const registry = new ModelRegistry();
      const mockModel = createMockStreamModel(streamEvents);
      registry.register('stream-model', mockModel);

      const handler = createHandler(registry);
      const reqBody = {
        model: 'stream-model',
        input: 'Compare values',
        stream: true,
      };
      const req = createMockReq('POST', '/v1/responses', reqBody);
      const { res, getBody, waitForEnd } = createMockRes();

      handler(req, res);
      await waitForEnd();

      const body = getBody();

      // Parse SSE events
      const events: Array<{ event: string; data: Record<string, unknown> }> = [];
      for (const line of body.split('\n')) {
        if (line.startsWith('event: ')) {
          events.push({ event: line.slice(7), data: {} });
        } else if (line.startsWith('data: ') && events.length > 0) {
          events[events.length - 1].data = JSON.parse(line.slice(6));
        }
      }

      // The text with '<' should eventually be flushed
      const textDeltas = events
        .filter((e) => e.event === 'response.output_text.delta')
        .map((e) => e.data.delta as string);
      const allDeltaText = textDeltas.join('');
      expect(allDeltaText).toContain('Value is 5');
      expect(allDeltaText).toContain('< 10');

      // output_text.done should have the final text
      const textDone = events.find((e) => e.event === 'response.output_text.done');
      expect(textDone).toBeDefined();
      expect(textDone!.data.text).toBe('Value is 5 < 10');
    });
  });
});
