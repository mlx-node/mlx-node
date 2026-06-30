import type { ServerResponse } from 'node:http';

import type { ChatMessage, ChatResult, ToolCallResult } from '@mlx-node/core';
import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';
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
  // Mirror Node's overloaded `end()` signature (chunk?, encoding? | cb?, cb?). The
  // `endJson` helper calls `res.end(body, cb)`, so the mock MUST hoist the callback
  // out of the `encoding` slot when it is a function — otherwise cb never fires.
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
    cachedTokens: 0,
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
        {
          model: 'test',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 0,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.message).toContain('max_tokens');
    });

    it('returns 400 when max_tokens exceeds i32::MAX (2^31)', async () => {
      // NAPI truncates a JS integer above i32::MAX to a NEGATIVE i32, which
      // the core clamp turns into 0 (silent empty completion). Reject at the
      // edge instead.
      const registry = new ModelRegistry();
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 2147483648,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.error.message).toContain('max_tokens');
    });

    it('returns 400 when max_tokens is Number.MAX_SAFE_INTEGER', async () => {
      const registry = new ModelRegistry();
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: Number.MAX_SAFE_INTEGER,
        } as any,
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
        {
          model: 'nonexistent',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );

      expect(getStatus()).toBe(404);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('not_found_error');
      expect(parsed.error.message).toContain('nonexistent');
    });

    it('does NOT call resolveModel when mapAnthropicRequest will reject the body as 400', async () => {
      // Regression: previously `resolveModel` ran AFTER shallow validation but
      // BEFORE `mapAnthropicRequest`, so a malformed-but-shallow-valid request
      // (e.g. an unsupported content block type) triggered a multi-second
      // model load — and possibly evicted the currently-resident model — just
      // to return 400 a moment later.
      //
      // Fix: hoist the (pure-transform) `mapAnthropicRequest` above the
      // resolveModel hook so mapping errors short-circuit before any load.
      const registry = new ModelRegistry();
      const resolveModel = vi.fn().mockResolvedValue(undefined);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [
            {
              role: 'user',
              content: [
                // Unsupported block type — mapAnthropicRequest throws.
                { type: 'mystery_block_type', text: 'oops' } as any,
              ],
            },
          ],
          max_tokens: 100,
        },
        registry,
        undefined,
        null,
        resolveModel,
      );

      // The critical assertion: the lazy-load hook MUST NOT have fired for a
      // request that was destined to 400 on mapping. Asserted first because
      // it's the load-bearing claim of this regression test.
      expect(resolveModel).not.toHaveBeenCalled();
      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('invalid_request_error');
    });

    it('returns 400 when tool_choice names a tool not in the tools list (no warm slot touched)', async () => {
      // End-to-end: the mapper rejection bubbles through the handler's outer
      // try/catch into an Anthropic-shape 400. Crucially, the warm slot must
      // NOT be created on the failure path (the handler short-circuits before
      // any session work).
      const registry = new ModelRegistry();
      registry.register('test-model', createMockModel());
      const sessionReg = registry.getSessionRegistry('test-model')!;
      const sizeBefore = sessionReg.size;
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          tool_choice: { type: 'tool', name: 'nonexistent' },
          tools: [
            { name: 'A', input_schema: {} },
            { name: 'B', input_schema: {} },
          ],
        },
        registry,
      );

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('invalid_request_error');
      expect(parsed.error.message).toContain('nonexistent');
      // Failure path MUST NOT have touched the warm slot.
      expect(sessionReg.size).toBe(sizeBefore);
    });

    it('accepts non-empty stop_sequences instead of rejecting with 400', async () => {
      // End-to-end: the mapper no longer rejects `stop_sequences`; it threads
      // the stop strings out so a downstream consumer can honour them. The
      // request must proceed normally rather than 400.
      const registry = new ModelRegistry();
      registry.register('test-model', createMockModel());
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('message');
    });
  });

  // -----------------------------------------------------------------------
  // Idle sweeper bracketing of resolveModel
  // -----------------------------------------------------------------------

  describe('resolveModel idle-sweeper bracketing', () => {
    it('wraps resolveModel in withSuspendedDrains BEFORE beginRequest fires', async () => {
      // Regression: in `mlx launch claude` mode `resolveModel` may run a
      // 30s `loadModel()` on first sight of an unknown name. The previous
      // request's `endRequest()` arms a 30s drain timer that fires
      // `clearCache()` on expiry. If we did NOT bracket the lazy-load
      // call, the timer could fire MID-LOAD while weight materialization
      // was still allocating through the Metal free pool — exactly the
      // hot-load race that `withSuspendedDrains` exists to prevent.
      //
      // The fix wraps the `resolveModel(...)` invocation in
      // `idleSweeper.withSuspendedDrains(...)` so the drain is suspended
      // for the full duration of the load. The bracket MUST land BEFORE
      // `beginRequest()` (which itself only cancels an already-armed
      // timer; it does not protect against a timer that fires *during*
      // the await on resolveModel before beginRequest runs).
      const registry = new ModelRegistry();
      const { res } = createMockRes();

      // Side-effect: register the model so the post-resolveModel
      // `registry.get(...)` lookup succeeds and the handler proceeds
      // through dispatch. The mock model simply returns a canned
      // `ChatResult`.
      const mockModel = createMockModel();
      const resolveOrder: string[] = [];
      const resolveModel = vi.fn(async (_name: string) => {
        resolveOrder.push('resolveModel:enter');
        // Force at least one event-loop turn so the await is real.
        await new Promise<void>((resolve) => setTimeout(resolve, 10));
        registry.register('test-model', mockModel);
        resolveOrder.push('resolveModel:exit');
      });

      // Fake idle sweeper. `withSuspendedDrains` simply runs the supplied
      // function; the assertion is purely on call ordering.
      const withSuspendedDrains = vi.fn(async <T>(fn: () => T | Promise<T>): Promise<T> => {
        return await fn();
      });
      const beginRequest = vi.fn();
      const endRequest = vi.fn();
      const fakeSweeper = {
        withSuspendedDrains,
        beginRequest,
        endRequest,
        suspendDrains: vi.fn(() => () => {}),
        close: vi.fn(),
        isPending: false,
        inFlight: 0,
      };

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
        undefined,
        fakeSweeper as any,
        resolveModel,
      );

      // Load-bearing assertion #1: resolveModel was actually invoked.
      expect(resolveModel).toHaveBeenCalledTimes(1);
      expect(resolveModel).toHaveBeenCalledWith('test-model');

      // Load-bearing assertion #2: the bracket wraps resolveModel.
      expect(withSuspendedDrains).toHaveBeenCalledTimes(1);
      expect(resolveOrder).toEqual(['resolveModel:enter', 'resolveModel:exit']);

      // Load-bearing assertion #3: withSuspendedDrains runs BEFORE
      // beginRequest. Without this ordering the previous request's
      // armed timer could fire mid-load.
      expect(beginRequest).toHaveBeenCalledTimes(1);
      expect(withSuspendedDrains.mock.invocationCallOrder[0]).toBeLessThan(beginRequest.mock.invocationCallOrder[0]);

      // Sanity: the matching endRequest fires.
      expect(endRequest).toHaveBeenCalledTimes(1);
    });

    it('returns 500 with an Anthropic-shape error envelope when resolveModel rejects', async () => {
      // Regression: previously `resolveModel` ran outside any try/catch in
      // this handler, so a rejection (bad model path, corrupt weights,
      // native loader error in `mlx launch claude` mode) bubbled up to the
      // outer `createHandler` catch which emits the OpenAI-shape
      // `{ "error": { type, message } }` body via `sendInternalError`.
      // This endpoint is `/v1/messages` (Anthropic) — clients parse the
      // `{ "type": "error", "error": { "type": "api_error", "message": ... } }`
      // envelope, so the OpenAI-shape body could not be parsed.
      //
      // Fix: wrap the `resolveModel(...)` call in a try/catch and route
      // failures through `sendAnthropicInternalError`. Mirrors the
      // `mapAnthropicRequest` try/catch a few lines above in the same handler.
      const registry = new ModelRegistry();
      const resolveModel = vi.fn().mockRejectedValue(new Error('bad model path'));
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
        undefined,
        null,
        resolveModel,
      );

      expect(resolveModel).toHaveBeenCalledTimes(1);
      expect(getStatus()).toBe(500);
      const parsed = JSON.parse(getBody());
      // Anthropic-shape envelope: `{ type: 'error', error: { type, message } }`
      // (see `sendAnthropicInternalError` in `packages/server/src/errors.ts`).
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('api_error');
      expect(parsed.error.message).toContain('bad model path');
    });

    it('falls back to a direct resolveModel call when no idle sweeper is provided', async () => {
      // The bracket is conditional on `idleSweeper != null` so a host
      // that opted out (or hasn't wired one) still gets the lazy-load
      // hook fired. Asserts the fallback branch.
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      const resolveModel = vi.fn(async (_name: string) => {
        registry.register('test-model', mockModel);
      });
      const { res, getStatus } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
        undefined,
        // No sweeper → direct call.
        null,
        resolveModel,
      );

      expect(resolveModel).toHaveBeenCalledTimes(1);
      expect(getStatus()).toBe(200);
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

    it('clamps max_tokens to the registered output cap before dispatch', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      registry.register('test-model', mockModel, { maxOutputTokens: 16 });
      const { res, getStatus } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Hello' }],
          max_tokens: 128000,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      const config = startSpy.mock.calls[0]?.[1] as { maxNewTokens?: number };
      expect(config.maxNewTokens).toBe(16);
    });

    it('leaves max_tokens unchanged when no registered output cap exists', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel();
      registry.register('test-model', mockModel);
      const { res, getStatus } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Hello' }],
          max_tokens: 128000,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      const config = startSpy.mock.calls[0]?.[1] as { maxNewTokens?: number };
      expect(config.maxNewTokens).toBe(128000);
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

    it('streaming suppresses Gemma4 parsed tool-call markup when tools are absent', async () => {
      const rawText =
        '<|channel>thought\nI should inspect files.\n<channel|><|tool_call>call:read_file{path:<|"|>Cargo.toml<|"|>}<tool_call|><turn|>';
      const registry = new ModelRegistry();
      registry.register(
        'test-model',
        createMockStreamModel([
          { text: 'I should inspect files.', done: false, isReasoning: true },
          {
            text: '',
            done: true,
            finishReason: 'tool_calls',
            toolCalls: [
              {
                status: 'ok',
                id: 'toolu_abc123',
                name: 'read_file',
                arguments: '{"path":"Cargo.toml"}',
              } as ToolCallResult,
            ],
            thinking: 'I should inspect files.',
            numTokens: 20,
            promptTokens: 10,
            reasoningTokens: 0,
            rawText,
          },
        ]),
      );
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'summarize Cargo.toml' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());
      const text = events
        .filter((e) => e.event === 'content_block_delta')
        .map((e) => e.data['delta'] as { type?: string; text?: string })
        .filter((delta) => delta.type === 'text_delta')
        .map((delta) => delta.text ?? '')
        .join('');

      expect(text).toBe('');
      expect(text).not.toContain('<|channel>');
      expect(text).not.toContain('<channel|>');
      expect(text).not.toContain('<|tool_call>');
      expect(text).not.toContain('<tool_call|>');

      const stop = events.find((e) => e.event === 'message_delta')?.data['delta'] as { stop_reason?: string };
      expect(stop.stop_reason).toBe('end_turn');
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
          tools: [
            {
              name: 'get_weather',
              input_schema: { type: 'object', properties: {} },
            },
          ],
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

    it('does not emit tool_use blocks or parsed tool markup when request has no tools', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: '',
          toolCalls: [
            {
              status: 'ok',
              id: 'toolu_no_tools',
              name: 'get_weather',
              arguments: '{"location":"San Francisco"}',
              rawContent: '{"name":"get_weather","arguments":{"location":"San Francisco"}}',
            } as ToolCallResult,
          ],
          numTokens: 20,
          promptTokens: 10,
          reasoningTokens: 0,
          finishReason: 'stop',
          rawText: '<tool_call>{"name":"get_weather","arguments":{"location":"San Francisco"}}</tool_call>',
        }),
      );
      registry.register('test-model', mockModel);
      const noToolsSessionReg = registry.getSessionRegistry('test-model')!;
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'What is the weather?' }],
          max_tokens: 100,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.stop_reason).toBe('end_turn');
      expect(parsed.content.some((b: any) => b.type === 'tool_use')).toBe(false);
      expect(parsed.content).toEqual([
        {
          type: 'text',
          text: '',
        },
      ]);
      expect(noToolsSessionReg.size).toBe(0);
    });

    it('drops the warm slot when chatSessionStart resolves with finishReason="error"', async () => {
      // Defensive-hardening gate: today every native failure throws,
      // but `runSessionNonStreaming` enforces the invariant locally so
      // a future Rust change that resolves `chat_session_start_sync`
      // with `Ok(finish_reason="error")` cannot silently poison the
      // warm slot. Mirrors the streaming-side dual-gate
      // (`streamResult.ok && outcome.wasCommitted()`) and the sibling
      // `/v1/responses` adopt gate.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ finishReason: 'error' }));
      registry.register('test-model', mockModel);
      const sessionReg = registry.getSessionRegistry('test-model')!;
      const { res, getStatus } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Hello' }],
          max_tokens: 100,
        },
        registry,
      );

      // Native promise resolved cleanly, so the request returns 200
      // and the body still flushes — only the warm-slot adoption is
      // gated. The next request that should hit the slot must miss.
      expect(getStatus()).toBe(200);
      expect(sessionReg.size).toBe(0);
    });

    it('adopts the warm slot on a clean finishReason="stop" (regression)', async () => {
      // Companion to the `finishReason="error"` test above: pin the
      // happy-path branch of the same gate so a regression that
      // inverts the predicate fails loudly instead of silently
      // suppressing every warm-slot adoption.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ finishReason: 'stop' }));
      registry.register('test-model', mockModel);
      const sessionReg = registry.getSessionRegistry('test-model')!;
      const { res, getStatus } = createMockRes();

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
      expect(sessionReg.size).toBe(1);
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

    it('clamps max_tokens to the registered output cap before streaming dispatch', async () => {
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'Hello', done: false, isReasoning: false },
        {
          text: 'Hello',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 1,
          promptTokens: 1,
          reasoningTokens: 0,
          rawText: 'Hello',
        },
      ];
      const mockModel = createMockStreamModel(streamEvents);
      registry.register('test-model', mockModel, { maxOutputTokens: 16 });
      const { res, getStatus } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Hi' }],
          max_tokens: 128000,
          stream: true,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const streamSpy = mockModel.chatStreamSessionStart as unknown as ReturnType<typeof vi.fn>;
      const config = streamSpy.mock.calls[0]?.[1] as { maxNewTokens?: number };
      expect(config.maxNewTokens).toBe(16);
    });

    it('uses a short non-thinking config for Claude Code title generation requests', async () => {
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: '{', done: false, isReasoning: true },
        { text: '{"title": "Analyze codebase architecture"}', done: false, isReasoning: false },
        {
          text: '{"title": "Analyze codebase architecture"}',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 8,
          promptTokens: 32,
          reasoningTokens: 0,
          rawText: '{"title": "Analyze codebase architecture"}',
        },
      ];
      const mockModel = createMockStreamModel(streamEvents);
      registry.register('test-model', mockModel, { maxOutputTokens: 81920 });
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Deepresearch the whole codebase, describe the architecture' }],
          system: [
            { type: 'text', text: 'x-anthropic-billing-header: cc_version=2.1.138; cch=d2ad0;' },
            { type: 'text', text: "You are Claude Code, Anthropic's official CLI for Claude." },
            {
              type: 'text',
              text:
                'Generate a concise, sentence-case title (3-7 words) that captures the main topic. ' +
                'Return JSON with a single "title" field.',
            },
          ],
          tools: [],
          max_tokens: 64000,
          output_config: {
            format: {
              type: 'json_schema',
              schema: {
                type: 'object',
                properties: { title: { type: 'string' } },
                required: ['title'],
                additionalProperties: false,
              },
            },
          },
          stream: true,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const streamSpy = mockModel.chatStreamSessionStart as unknown as ReturnType<typeof vi.fn>;
      const config = streamSpy.mock.calls[0]?.[1] as {
        maxNewTokens?: number;
        reasoningEffort?: string;
        thinkingTokenBudget?: number;
        includeReasoning?: boolean;
      };
      expect(config.maxNewTokens).toBe(128);
      expect(config.reasoningEffort).toBe('none');
      expect(config.thinkingTokenBudget).toBe(0);
      expect(config.includeReasoning).toBe(false);

      const events = parseSSE(getBody());
      const thinkingStart = events.find(
        (event) =>
          event.event === 'content_block_start' &&
          (event.data['content_block'] as Record<string, unknown> | undefined)?.['type'] === 'thinking',
      );
      expect(thinkingStart).toBeUndefined();
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
          tools: [
            {
              name: 'get_weather',
              input_schema: { type: 'object', properties: {} },
            },
          ],
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
          tools: [
            {
              name: 'search',
              input_schema: { type: 'object', properties: {} },
            },
          ],
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

    it('suppresses parsed tool_call markup when request has no tools', async () => {
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
              id: 'toolu_no_tools',
              name: 'search',
              arguments: '{"query":"test"}',
            },
          ],
          thinking: null,
          numTokens: 8,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: '<tool_call>{"name":"search"}</tool_call>',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const noToolsStreamSessionReg = registry.getSessionRegistry('test-model')!;
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
      const textStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStarts).toHaveLength(0);

      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const combined = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(combined).toBe('');

      const toolStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolStarts).toHaveLength(0);

      const jsonDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'input_json_delta',
      );
      expect(jsonDeltas).toHaveLength(0);

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('end_turn');
      expect(noToolsStreamSessionReg.size).toBe(0);
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

    it('recovers full malformed tool_call when finalText is post-</think>-trim cleaned', async () => {
      // Real bug 2026-04-28: model decode-loops past max_tokens with an unclosed
      // <tool_call>. Streamed chunks include "\n\n" (post-</think> whitespace)
      // before the <tool_call> suppression triggers. Native finalText is the
      // cleaned text starting with "<tool_call>" (split_at_think_end trims the
      // leading "\n\n"). The recovery branch must NOT slice into <t — it must
      // emit a complete "<tool_call>" tag.
      const registry = new ModelRegistry();
      const streamEvents = [
        // Thinking tokens
        { text: 'reasoning here', done: false, isReasoning: true },
        // Reasoning closing fence (server filters '</think>' out via replace)
        { text: '</think>', done: false, isReasoning: true },
        // Post-</think> whitespace flowing as a non-reasoning text delta —
        // this advances emittedTextLength to 2 BEFORE suppression triggers.
        { text: '\n\n', done: false, isReasoning: false },
        // Unclosed tool_call — tagBuffer suppresses everything from here on.
        {
          text: '<tool_call>\n<function=Agent>{"q":"x"}',
          done: false,
          isReasoning: false,
        },
        {
          // Native side trims the leading "\n\n" (split_at_think_end), so
          // finalText starts with "<tool_call>" — NOT "\n\n<tool_call>".
          // toolCalls is empty because the parser failed (no </tool_call>).
          text: '<tool_call>\n<function=Agent>{"q":"x"}',
          done: true,
          finishReason: 'length',
          toolCalls: [],
          thinking: 'reasoning here',
          numTokens: 30,
          promptTokens: 10,
          reasoningTokens: 5,
          rawText: '<think>reasoning here</think>\n\n<tool_call>\n<function=Agent>{"q":"x"}',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Use a tool' }],
          max_tokens: 30,
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

      // The combined text deltas must contain a complete "<tool_call>" tag.
      // The bug emitted "ool_call>\n<function=..." (leading "<t" sliced off
      // by `finalText.slice(emittedTextLength)` where emittedTextLength=2
      // counted the streamed "\n\n" but finalText had been post-</think>-trim
      // cleaned to start at "<tool_call>"). Check that every "ool_call>"
      // occurrence is preceded by "<t" — i.e. no orphaned, half-sliced tag.
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const combined = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(combined).toContain('<tool_call>');
      // Guard against the bug's specific signature: an "ool_call>" with no
      // "<t" immediately before it. Under the fix every "ool_call>" is part
      // of a complete "<tool_call>".
      const orphanMatch = /(^|[^t])ool_call>/.exec(combined);
      expect(orphanMatch, `combined text contains a half-sliced tool_call tag: ${combined!}`).toBeNull();

      // Should NOT have any tool_use block (parser failed → empty toolCalls)
      const toolStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolStarts).toHaveLength(0);
    });

    it('does not duplicate preamble when closed tool_call parses non-ok and finalText is trimmed', async () => {
      // Codex-found regression 2026-04-29 (branch #2 path): when a CLOSED
      // <tool_call>...</tool_call> block parses to a non-ok status (e.g.
      // invalid JSON inside <function>), `okToolCalls` is empty, so the
      // suppression-recovery branch fires. With a streamed preamble that
      // had trailing whitespace and a finalText that the native side
      // trimmed via `parse_tool_calls`, the overlap math returns 0 and
      // emits finalText whole — duplicating the preamble.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'Let me check. ', done: false, isReasoning: false },
        {
          text: '<tool_call>\n<function=lookup>\n<parameter=q>\n{not valid json\n</parameter>\n</function>\n</tool_call>',
          done: false,
          isReasoning: false,
        },
        {
          // Native parse_tool_calls strips the closed block AND trims →
          // finalText is the trimmed preamble.
          text: 'Let me check.',
          done: true,
          finishReason: 'stop',
          // Non-ok tool call — okToolCalls filter returns [].
          toolCalls: [
            {
              id: 'tool_1',
              name: 'lookup',
              arguments: '',
              status: 'invalid_json',
            },
          ],
          thinking: null,
          numTokens: 25,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText:
            'Let me check. <tool_call>\n<function=lookup>\n<parameter=q>\n{not valid json\n</parameter>\n</function>\n</tool_call>',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'use a tool' }],
          max_tokens: 30,
          stream: true,
          tools: [
            {
              name: 'lookup',
              input_schema: { type: 'object', properties: {} },
            },
          ],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const combined = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');

      // Preamble must appear exactly once. Bug would emit "Let me check. Let me check."
      const occurrences = combined.match(/Let me check/g) ?? [];
      expect(occurrences.length, `preamble duplicated; combined text deltas: ${JSON.stringify(combined)}`).toBe(1);
    });

    it('recovers malformed tool_call when streamed leading-whitespace exceeds finalText length', async () => {
      // Codex-found regression 2026-04-29 (third pass): a `finalText.length >
      // emittedText.length` length guard was wrong when streamed leading
      // whitespace (e.g. many newlines after `</think>`) is LONGER than the
      // malformed tool_call body in finalText. With the length guard, branch
      // #2 was skipped → client received only whitespace, losing the malformed
      // `<tool_call>...` text entirely. The fix uses `!emittedText.includes(
      // finalText)` instead, which correctly distinguishes "trimmed substring
      // of streamed" (duplicate case → skip) from "different content"
      // (recovery case → emit).
      const longWhitespace = '\n'.repeat(80); // 80 chars > finalText length below
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: longWhitespace, done: false, isReasoning: false },
        {
          text: '<tool_call>\n<function=Agent>{"q":"x"}',
          done: false,
          isReasoning: false,
        },
        {
          // Native split_at_think_end trims the leading whitespace → finalText
          // starts with "<tool_call>" (length ~38, less than emittedText's 80
          // chars of whitespace).
          text: '<tool_call>\n<function=Agent>{"q":"x"}',
          done: true,
          finishReason: 'length',
          toolCalls: [],
          thinking: null,
          numTokens: 25,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: `${longWhitespace}<tool_call>\n<function=Agent>{"q":"x"}`,
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'x' }],
          max_tokens: 30,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const combined = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');

      // The malformed `<tool_call>` text MUST surface to the client (not be
      // swallowed by the length guard).
      expect(combined).toContain('<tool_call>');
      expect(combined).toContain('<function=Agent>');
    });

    it('does not duplicate preamble when finalText is shorter than streamed (native trim)', async () => {
      // Codex-found regression 2026-04-29: when the model emits a closed
      // tool_call with leading text, the streaming path emits the preamble
      // verbatim ("Let me check. " — note trailing space), but the native
      // side trims trailing whitespace from result.text via parse_tool_calls
      // (`.trim()` at the end of parse_generation_output / split_at_think_end).
      // So `emittedText = "Let me check. "` (length 15) but `finalText =
      // "Let me check."` (length 14). The terminal "tail catch-up" branch
      // must NOT re-emit finalText whole — that would duplicate the preamble.
      const registry = new ModelRegistry();
      const streamEvents = [
        // Pre-tool text streamed verbatim through tagBuffer.
        { text: 'Let me check. ', done: false, isReasoning: false },
        // Closed tool_call — tagBuffer suppresses, then parse extracts.
        {
          text: '<tool_call>\n<function=lookup>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>',
          done: false,
          isReasoning: false,
        },
        {
          // finalText: native parse_tool_calls strips the closed <tool_call>
          // block AND calls .trim() on the cleaned text → trailing space gone.
          text: 'Let me check.',
          done: true,
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'tool_1',
              name: 'lookup',
              arguments: { q: 'x' },
              status: 'ok',
            },
          ],
          thinking: null,
          numTokens: 20,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText:
            'Let me check. <tool_call>\n<function=lookup>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'use a tool' }],
          max_tokens: 30,
          stream: true,
          tools: [
            {
              name: 'lookup',
              input_schema: { type: 'object', properties: {} },
            },
          ],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const combined = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');

      // The preamble must appear exactly once. The bug would have produced
      // "Let me check. Let me check." (streamed prefix + duplicated finalText).
      const occurrences = combined.match(/Let me check/g) ?? [];
      expect(occurrences.length, `preamble duplicated; combined text deltas: ${JSON.stringify(combined)}`).toBe(1);

      // The valid tool_use block must still be emitted.
      const toolStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolStarts).toHaveLength(1);
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

    it('drops whitespace-only text emitted right before a tool_call tag (no stray text block)', async () => {
      // Symptom from the production log: when the model emits `</think>\n\n<tool_call>...`,
      // the streaming buffer flushed the `\n\n` as `safeText` BEFORE the
      // `<tool_call>` tag opened, ratifying a content_block_start(text) /
      // content_block_delta(\n\n) / content_block_stop triple right before
      // the tool_use frame — which clients render as visible whitespace.
      // The fix holds back leading whitespace-only text until either real
      // content arrives (flushed together) or a non-text block opens
      // (dropped silently).
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: '\n\n', done: false, isReasoning: false },
        {
          text: '',
          done: true,
          finishReason: 'tool_calls',
          toolCalls: [
            {
              status: 'ok',
              id: 'call_weather',
              name: 'get_weather',
              arguments: '{"city":"SF"}',
            } as ToolCallResult,
          ],
          thinking: null,
          numTokens: 8,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: '\n\n',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'weather?' }],
          max_tokens: 100,
          stream: true,
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }],
        },
        registry,
      );

      const events = parseSSE(getBody());

      // No text content_block_start at all — the leading `\n\n` was held
      // back and dropped when the tool_use opened.
      const textStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStarts).toHaveLength(0);

      // The tool_use frame opens at index 0 (no text block was emitted
      // ahead of it). Its id is rewritten to the Anthropic `toolu_*` shape.
      const toolStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolStarts).toHaveLength(1);
      expect((toolStarts[0].data['content_block'] as any).id).toBe('toolu_weather');

      // No stray whitespace text_delta on the wire.
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      expect(textDeltas).toHaveLength(0);
    });

    it('flushes buffered leading whitespace alongside the first real text delta', async () => {
      // The opposite end of the spectrum: when whitespace IS followed by
      // real content, the buffered prefix must be flushed as part of the
      // first text block so the wire-content matches what the model
      // produced. Exactly one text block opens, with combined `"\n\nhello"`.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: '\n\n', done: false, isReasoning: false },
        { text: 'hello', done: false, isReasoning: false },
        {
          text: '\n\nhello',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 3,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: '\n\nhello',
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
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStarts).toHaveLength(1);

      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const combined = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(combined).toBe('\n\nhello');
    });

    it('preserves buffered leading whitespace when tool_call is preceded by visible text in the same chunk', async () => {
      // Regression: when the stream is `["\n\n", "Let me check.<tool_call>..."]`,
      // the buffered `\n\n` (held back because the first delta is whitespace-only)
      // semantically belongs to the visible `Let me check.` prefix that arrives
      // in the same chunk as the `<tool_call>` tag. The previous implementation
      // unconditionally cleared `pendingLeadingWhitespace` in the `tagFound`
      // branch before deciding whether to emit `cleanPrefix`, so the wire
      // received `"Let me check."` instead of `"\n\nLet me check."` and the
      // leading whitespace was silently dropped (or duplicated downstream
      // depending on `</think>` trimming).
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: '\n\n', done: false, isReasoning: false },
        {
          text: 'Let me check.<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>',
          done: false,
          isReasoning: false,
        },
        {
          text: '\n\nLet me check.',
          done: true,
          finishReason: 'tool_calls',
          toolCalls: [
            {
              status: 'ok',
              id: 'call_weather',
              name: 'get_weather',
              arguments: '{"city":"SF"}',
            } as ToolCallResult,
          ],
          thinking: null,
          numTokens: 12,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: '\n\nLet me check.<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'weather?' }],
          max_tokens: 100,
          stream: true,
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }],
        },
        registry,
      );

      const events = parseSSE(getBody());

      // The text block opens BEFORE the tool_use frame and its deltas join to
      // `"\n\nLet me check."` — the leading `\n\n` survives the tag transition.
      const textStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStarts).toHaveLength(1);
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const combined = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(combined).toBe('\n\nLet me check.');

      // The tool_use frame still follows and carries the parsed call.
      const toolStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolStarts).toHaveLength(1);
    });

    it('does not double leading whitespace when only whitespace deltas precede the done event', async () => {
      // Regression: `finalText` on the terminal done chunk is the FULL
      // accumulated text from the native side, not a delta. When the only
      // intermediate delta is whitespace (`"\n\n"`) and gets buffered into
      // `pendingLeadingWhitespace`, the same whitespace is already part of
      // `finalText`. The previous implementation prepended the buffer onto
      // `finalText` and emitted `"\n\n\n\nhello"` (4 newlines) instead of
      // `"\n\nhello"` (2 newlines, the byte-accurate model output).
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: '\n\n', done: false, isReasoning: false },
        {
          text: '\n\nhello',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 3,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: '\n\nhello',
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
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStarts).toHaveLength(1);
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const combined = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(combined).toBe('\n\nhello');
    });
  });

  // -----------------------------------------------------------------------
  // Stop sequences (honoring Anthropic `stop_sequences`)
  // -----------------------------------------------------------------------

  describe('stop sequences', () => {
    it('non-streaming: truncates text at the stop sequence and reports stop_sequence', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: 'keep this HALT drop this',
          rawText: 'keep this HALT drop this',
          finishReason: 'stop',
          numTokens: 12,
          promptTokens: 6,
        }),
      );
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.type).toBe('message');
      expect(parsed.content).toHaveLength(1);
      expect(parsed.content[0].type).toBe('text');
      expect(parsed.content[0].text).toBe('keep this ');
      expect(parsed.stop_reason).toBe('stop_sequence');
      expect(parsed.stop_sequence).toBe('HALT');
    });

    it('streaming: suppresses the stop sequence + tail and reports stop_sequence', async () => {
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'keep this HALT drop this', done: false, isReasoning: false },
        {
          text: 'keep this HALT drop this',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 12,
          promptTokens: 6,
          reasoningTokens: 0,
          rawText: 'keep this HALT drop this',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).toBe('keep this ');
      expect(streamedText).not.toContain('HALT');
      expect(streamedText).not.toContain('drop this');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeDefined();
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('HALT');

      const msgStop = events.find((e) => e.event === 'message_stop');
      expect(msgStop).toBeDefined();
    });

    it('streaming: detects a stop sequence split across two deltas', async () => {
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'keep HA', done: false, isReasoning: false },
        { text: 'LTdrop', done: false, isReasoning: false },
        {
          text: 'keep HALTdrop',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 10,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: 'keep HALTdrop',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).toBe('keep ');
      expect(streamedText).not.toContain('HALT');
      expect(streamedText).not.toContain('drop');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('HALT');
    });

    it('non-streaming: leaves text and stop_reason untouched when no stop sequence matches', async () => {
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: 'keep this whole thing',
          rawText: 'keep this whole thing',
          finishReason: 'stop',
          numTokens: 9,
          promptTokens: 4,
        }),
      );
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.content[0].text).toBe('keep this whole thing');
      expect(parsed.stop_reason).toBe('end_turn');
      expect(parsed.stop_sequence).toBe(null);
    });

    it('streaming: leaves text and stop_reason untouched when no stop sequence matches', async () => {
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'keep ', done: false, isReasoning: false },
        { text: 'this whole thing', done: false, isReasoning: false },
        {
          text: 'keep this whole thing',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 9,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: 'keep this whole thing',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).toBe('keep this whole thing');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('end_turn');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe(null);
    });

    it('streaming: honors a stop sequence that lands in the visible text before a tool-call tag', async () => {
      // Finding 1: the `tagFound` emit path used to write the visible text
      // before a structural marker (`cleanPrefix`) straight to the wire
      // without consulting the stop-sequence detector, so a stop string in
      // that prefix leaked and `stop_reason` stayed `tool_use`. The model
      // emits "keep this HALT " immediately followed by a `<tool_call>` in
      // the SAME delta, so the tag buffer reports `tagFound` with
      // `cleanPrefix === "keep this HALT "`.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'keep this HALT <tool_call>{"name":"get_weather"}', done: false, isReasoning: false },
        {
          text: '',
          done: true,
          finishReason: 'stop',
          toolCalls: [{ status: 'ok', id: 'toolu_w1', name: 'get_weather', arguments: '{"location":"NYC"}' }],
          thinking: null,
          numTokens: 12,
          promptTokens: 6,
          reasoningTokens: 0,
          rawText: 'keep this HALT <tool_call>{"name":"get_weather"}',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).toBe('keep this ');
      expect(streamedText).not.toContain('HALT');
      expect(streamedText).not.toContain('<tool_call>');

      // The stop match wins over the tool call: `stop_reason` is
      // `stop_sequence`, not `tool_use`.
      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeDefined();
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('HALT');

      // The tool call's tag came AFTER the stop boundary, so no tool_use
      // content block may be emitted alongside the stop_sequence terminal.
      const toolBlock = events.find(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolBlock).toBeUndefined();

      const msgStop = events.find((e) => e.event === 'message_stop');
      expect(msgStop).toBeDefined();
    });

    it('non-streaming: suppresses the tool_use block when a stop matched in the visible text', async () => {
      // Non-streaming sibling of the stop-before-tool case: the visible text
      // carries a stop string and the native result also parsed a tool call.
      // The truncated text wins and the tool_use block must be dropped.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: 'keep this HALT drop this',
          rawText: 'keep this HALT <tool_call>{"name":"get_weather"}</tool_call>',
          finishReason: 'stop',
          toolCalls: [
            {
              status: 'ok',
              id: 'toolu_stop_ns',
              name: 'get_weather',
              arguments: '{"location":"NYC"}',
            } as ToolCallResult,
          ],
          numTokens: 12,
          promptTokens: 6,
        }),
      );
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stop_sequences: ['HALT'],
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }],
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.stop_reason).toBe('stop_sequence');
      expect(parsed.stop_sequence).toBe('HALT');
      expect(parsed.content.some((b: any) => b.type === 'tool_use')).toBe(false);
      expect(parsed.content).toEqual([{ type: 'text', text: 'keep this ' }]);
    });

    it('non-streaming: resolves a held overlapping stop at flush', async () => {
      // BUG A: `push()` holds a complete stop ('HALT') because a longer
      // overlapping stop ('HALTED') is still viable, so it returns
      // `matched:null`. Without a follow-up `flush()` the held stop leaks as
      // normal text with `stop_reason: end_turn`. The non-streaming path must
      // push+flush like the streaming done-path.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: 'keep HALT',
          rawText: 'keep HALT',
          finishReason: 'stop',
          numTokens: 6,
          promptTokens: 3,
        }),
      );
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stop_sequences: ['HALT', 'HALTED'],
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.content).toHaveLength(1);
      expect(parsed.content[0].type).toBe('text');
      expect(parsed.content[0].text).toBe('keep ');
      expect(parsed.stop_reason).toBe('stop_sequence');
      expect(parsed.stop_sequence).toBe('HALT');
    });

    it('non-streaming: honors a stop sequence inside recovered suppressed-tool text', async () => {
      // BUG B: the request disallows tools, but the native parser still
      // produced a tool call and `result.text` is empty, so
      // `buildAnthropicContent` emits `recoverSuppressedToolCallText(rawText)`
      // as the visible text. The stop scan must run over THAT recovered text,
      // not the empty `result.text` — otherwise a stop inside it leaks with
      // `stop_reason: end_turn`.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: '',
          rawText: 'Sure STOP <tool_call>{"name":"get_weather","arguments":{}}</tool_call>',
          finishReason: 'stop',
          toolCalls: [
            {
              status: 'ok',
              id: 'call_x',
              name: 'get_weather',
              arguments: '{}',
            } as ToolCallResult,
          ],
          numTokens: 12,
          promptTokens: 6,
        }),
      );
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stop_sequences: ['STOP'],
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.content.some((b: any) => b.type === 'tool_use')).toBe(false);
      const textBlock = parsed.content.find((b: any) => b.type === 'text');
      expect(textBlock).toBeDefined();
      expect(textBlock.text).toBe('Sure ');
      expect(parsed.stop_reason).toBe('stop_sequence');
      expect(parsed.stop_sequence).toBe('STOP');
    });

    it('non-streaming: keeps a trailing incomplete partial when no stop completes', async () => {
      // Control for the flush() addition: 'keep HAL' ends in a prefix of
      // 'HALT' but never completes it. `flush()` must release the held partial
      // as normal text — nothing may be dropped and no false truncation may
      // occur.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: 'keep HAL',
          rawText: 'keep HAL',
          finishReason: 'stop',
          numTokens: 6,
          promptTokens: 3,
        }),
      );
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.content[0].text).toBe('keep HAL');
      expect(parsed.stop_reason).toBe('end_turn');
      expect(parsed.stop_sequence).toBe(null);
    });

    it('streaming: honors a stop sequence hidden in suppressed/recovered text on a no-tools turn', async () => {
      // Finding 1: the done-path recovery branch re-emits native final text
      // that the tag buffer suppressed (here the visible bytes around a
      // `<tool_call>` on a request with NO tools). A stop string living in
      // that recovered tail was never routed through the detector, so it
      // leaked and the terminal kept the native reason. The recovered text
      // must run through the SAME detector before emission.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'Sure <tool_call>{"name":"x"}</tool_call> STOP now', done: false, isReasoning: false },
        {
          text: 'Sure  STOP now',
          done: true,
          finishReason: 'stop',
          toolCalls: [{ status: 'ok', id: 'call_x', name: 'x', arguments: '{}' }],
          thinking: null,
          numTokens: 12,
          promptTokens: 6,
          reasoningTokens: 0,
          rawText: 'Sure <tool_call>{"name":"x"}</tool_call> STOP now',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['STOP'],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).not.toContain('STOP');
      expect(streamedText).not.toContain('now');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeDefined();
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('STOP');

      const msgStop = events.find((e) => e.event === 'message_stop');
      expect(msgStop).toBeDefined();
    });

    it('streaming: flushes parked pre-stop whitespace as the truncated prefix', async () => {
      // Finding 5: when the same push that cleared a whitespace-only safe
      // prefix also matched the stop, that whitespace was parked in
      // `pendingLeadingWhitespace` and then dropped at done, so streaming
      // returned no text while non-streaming returned the leading whitespace.
      // The parked safe bytes must be flushed as the truncated prefix.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: ' HALTtail', done: false, isReasoning: false },
        {
          text: ' HALTtail',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 8,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: ' HALTtail',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      // Exactly the safe pre-stop prefix " ", matching the non-streaming result.
      expect(streamedText).toBe(' ');
      expect(streamedText).not.toContain('HALT');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('HALT');
    });

    it('streaming: opens an empty text block when a stop sequence consumes all visible output', async () => {
      // Parity with the non-streaming path: when the stop matches at the very
      // start so the visible text collapses to '', the reconstructed content
      // must be [{type:'text', text:''}] — a text block opened and closed with
      // no body, exactly like buildAnthropicContent. The old `body.length > 0`
      // guard skipped the block entirely, so a client rebuilding from SSE saw
      // `content: []` while the non-streaming sibling returned an empty block.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'STOPhello', done: false, isReasoning: false },
        {
          text: 'STOPhello',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 5,
          promptTokens: 3,
          reasoningTokens: 0,
          rawText: 'STOPhello',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['STOP'],
        },
        registry,
      );

      const events = parseSSE(getBody());

      // Exactly one text block, opened with empty text — the empty-collapse block.
      const textStarts = events.filter(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'text',
      );
      expect(textStarts).toHaveLength(1);
      expect((textStarts[0].data['content_block'] as any).text).toBe('');

      // The stop ate everything: no visible text leaked and no empty text_delta.
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      expect(textDeltas).toHaveLength(0);

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('STOP');
    });

    it('non-streaming: returns the leading whitespace prefix when a stop matches right after it', async () => {
      // Non-streaming counterpart of the parked-whitespace streaming case:
      // the result text " HALTtail" truncates to " " (the safe prefix).
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: ' HALTtail',
          rawText: ' HALTtail',
          finishReason: 'stop',
          numTokens: 8,
          promptTokens: 4,
        }),
      );
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.content).toEqual([{ type: 'text', text: ' ' }]);
      expect(parsed.stop_reason).toBe('stop_sequence');
      expect(parsed.stop_sequence).toBe('HALT');
    });

    it('non-streaming: returns an empty text block when a stop sequence consumes all visible output', async () => {
      // Sibling/parity anchor of the streaming empty-collapse case: the result
      // text "STOPhello" truncates to '' and the content is a single empty text
      // block — the target the streaming path now matches.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(
        makeChatResult({
          text: 'STOPhello',
          rawText: 'STOPhello',
          finishReason: 'stop',
          numTokens: 5,
          promptTokens: 3,
        }),
      );
      registry.register('test-model', mockModel);
      const { res, getStatus, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stop_sequences: ['STOP'],
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      const parsed = JSON.parse(getBody());
      expect(parsed.content).toEqual([{ type: 'text', text: '' }]);
      expect(parsed.stop_reason).toBe('stop_sequence');
      expect(parsed.stop_sequence).toBe('STOP');
    });

    it('streaming: releases a benign held partial when a tool-call tag interrupts it', async () => {
      // Finding 1 (consequence 2): when the detector holds a benign partial
      // (a prefix of a stop sequence, e.g. "H" of "HALT") and the next delta
      // is a structural marker, the partial used to be stranded in the
      // detector and dropped by the suppressed-terminal residue gate. The
      // `tagFound` path now flushes the detector, releasing the partial as
      // visible text. Here "keep H" holds "H"; the next delta is a bare
      // `<tool_call>` (cleanPrefix === ""), and "H" can never become "HALT".
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'keep H', done: false, isReasoning: false },
        { text: '<tool_call>{"name":"get_weather"}', done: false, isReasoning: false },
        {
          text: '',
          done: true,
          finishReason: 'stop',
          toolCalls: [{ status: 'ok', id: 'toolu_w2', name: 'get_weather', arguments: '{"location":"NYC"}' }],
          thinking: null,
          numTokens: 10,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: 'keep H<tool_call>{"name":"get_weather"}',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      // The benign "H" is preserved, not dropped.
      expect(streamedText).toBe('keep H');

      // No stop matched, so the tool call drives the terminal as before.
      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('tool_use');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe(null);
    });

    it('streaming: preserves order of buffered leading whitespace and a held stop-prefix before a tool-call tag', async () => {
      // Re-review of the `tagFound` fix: a turn-leading whitespace-only delta
      // (" ") and a stop-prefix ("H" of "HALT") arrive together as " H", then a
      // bare `<tool_call>` follows. The detector clears the leading " " (visible)
      // and holds "H"; the " " parks in `pendingLeadingWhitespace` because no
      // text block is open yet. At the tag, `pendingLeadingWhitespace` is text
      // that ALREADY passed the detector, so it must be prepended OUTSIDE the
      // buffer. Re-pushing it would queue it after the held "H" and emit "H "
      // (order inverted). The visible run is " H" and "H" can never become
      // "HALT", so no stop matches and the tool call drives the terminal.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: ' H', done: false, isReasoning: false },
        { text: '<tool_call>{"name":"get_weather"}', done: false, isReasoning: false },
        {
          text: '',
          done: true,
          finishReason: 'stop',
          toolCalls: [{ status: 'ok', id: 'toolu_w4', name: 'get_weather', arguments: '{"location":"NYC"}' }],
          thinking: null,
          numTokens: 9,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: ' H<tool_call>{"name":"get_weather"}',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      // Order preserved (" H", not "H "), nothing leaked, nothing dropped.
      expect(streamedText).toBe(' H');

      // No stop matched, so the tool call drives the terminal.
      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('tool_use');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe(null);
    });

    it('streaming: leaves the pre-tag visible text untouched when no stop_sequences are configured', async () => {
      // Finding 2: with `stop_sequences` absent the detector is a pass-through,
      // so the `tagFound` path stays byte-identical — the full visible prefix
      // (here "keep this HALT ", stop string and all) reaches the wire and
      // `stop_sequence` is null. Same stream as the honor test above, minus
      // `stop_sequences`.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'keep this HALT <tool_call>{"name":"get_weather"}', done: false, isReasoning: false },
        {
          text: '',
          done: true,
          finishReason: 'stop',
          toolCalls: [{ status: 'ok', id: 'toolu_w3', name: 'get_weather', arguments: '{"location":"NYC"}' }],
          thinking: null,
          numTokens: 12,
          promptTokens: 6,
          reasoningTokens: 0,
          rawText: 'keep this HALT <tool_call>{"name":"get_weather"}',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).toBe('keep this HALT ');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('tool_use');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe(null);
    });

    it('streaming: detects a stop split across a held partial and recovered terminal text', async () => {
      // Finding B: the streamed delta "HA" is held by the detector (prefix of
      // "HALT"); the native done text "HALTtail" completes the stop. The held
      // partial must stay in the buffer while the terminal/recovered text is
      // scanned so "HALT" is caught across the boundary instead of leaking.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'HA', done: false, isReasoning: false },
        {
          text: 'HALTtail',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 8,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: 'HALTtail',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).not.toContain('HALT');
      expect(streamedText).not.toContain('tail');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeDefined();
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('HALT');
    });

    it('streaming: detects a stop split across a held partial and tag-suppressed recovered text', async () => {
      // Finding B (tag-suppression variant): the model emits "HA", then a
      // suppressed <tool_call> on a no-tools turn, then "LTtail". The native
      // cleaned done text "HALTtail" reconstitutes the visible text. The held
      // "HA" must still be buffered when the recovered tail is scanned, so the
      // already-streamed bytes never include any part of the matched stop.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'HA<tool_call>{}</tool_call>LTtail', done: false, isReasoning: false },
        {
          text: 'HALTtail',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 8,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: 'HA<tool_call>{}</tool_call>LTtail',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).not.toContain('HALT');
      expect(streamedText).not.toContain('tail');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeDefined();
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('HALT');
    });

    it('streaming: suppresses the tool_use block when a stop matches in recovered terminal text', async () => {
      // Finding C: tools enabled, streamed "prefix ", native done text
      // "prefix HALT tail" plus one ok tool call. The stop is found only in
      // the recovered terminal text, so the tool call must be suppressed and
      // stop_reason must be stop_sequence — a streamed response must never
      // carry both a tool_use block and stop_sequence. Matches non-streaming.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'prefix ', done: false, isReasoning: false },
        {
          text: 'prefix HALT tail',
          done: true,
          finishReason: 'stop',
          toolCalls: [{ status: 'ok', id: 'call_w', name: 'get_weather', arguments: '{"location":"NYC"}' }],
          thinking: null,
          numTokens: 10,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: 'prefix HALT tail',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).not.toContain('HALT');

      const toolBlock = events.find(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolBlock).toBeUndefined();

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeDefined();
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('HALT');
    });

    it('streaming: suppresses the tool_use block when a stop matches in tag-suppressed recovered text', async () => {
      // Finding C (tag-suppression variant): the model emits "prefix ", a
      // suppressed <tool_call>, then "HALT tail". The native cleaned done text
      // "prefix HALT tail" surfaces the stop. Tools must be suppressed and the
      // terminal must be stop_sequence — never both.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: 'prefix <tool_call>{"name":"get_weather"}</tool_call>HALT tail', done: false, isReasoning: false },
        {
          text: 'prefix HALT tail',
          done: true,
          finishReason: 'stop',
          toolCalls: [{ status: 'ok', id: 'call_w', name: 'get_weather', arguments: '{"location":"NYC"}' }],
          thinking: null,
          numTokens: 10,
          promptTokens: 5,
          reasoningTokens: 0,
          rawText: 'prefix <tool_call>{"name":"get_weather"}</tool_call>HALT tail',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
          tools: [{ name: 'get_weather', input_schema: { type: 'object', properties: {} } }],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).not.toContain('HALT');

      const toolBlock = events.find(
        (e) => e.event === 'content_block_start' && (e.data['content_block'] as any).type === 'tool_use',
      );
      expect(toolBlock).toBeUndefined();

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeDefined();
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('HALT');
    });

    it('streaming: counts parked leading whitespace in the terminal overlap so a held stop never replays it', async () => {
      // Wave #3 regression: the streamed delta " H" parks " " in
      // `pendingLeadingWhitespace` (whitespace-only, no block open yet) and holds
      // "H" in the stop detector (prefix of "HALT"). The native done text
      // " HALTtail" carries the full visible text. The terminal overlap basis must
      // include the parked " " so the recovered tail is only "ALTtail" (which
      // completes the held "HALT") instead of the whole " HALTtail" — otherwise the
      // already-received " H" gets replayed and the "H" of the stop leaks. Correct:
      // emit exactly the pre-stop whitespace " ", matching the non-streaming result.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: ' H', done: false, isReasoning: false },
        {
          text: ' HALTtail',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 8,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: ' HALTtail',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
          stop_sequences: ['HALT'],
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).toBe(' ');
      expect(streamedText).not.toContain('H');
      expect(streamedText).not.toContain('HALT');
      expect(streamedText).not.toContain('tail');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('stop_sequence');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe('HALT');
    });

    it('streaming: no-op with no stop_sequences never replays parked whitespace + tag residue', async () => {
      // Wave #3 regression (C8 no-op): with NO stop_sequences the streamed delta
      // " <" parks " " in `pendingLeadingWhitespace` and holds "<" in the tag
      // buffer (possible tag start). The native done text " <x" carries the full
      // visible text. The terminal overlap basis must include the parked " " and
      // the held "<" so the recovered tail is only "x" — otherwise " <x" replays
      // and the wire shows the duplicated " < <x". Correct: emit " <x" exactly once.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: ' <', done: false, isReasoning: false },
        {
          text: ' <x',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 4,
          promptTokens: 2,
          reasoningTokens: 0,
          rawText: ' <x',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).toBe(' <x');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('end_turn');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe(null);
    });

    it('streaming: no-op control recovers only the unsent suffix after parked whitespace (no stop_sequences)', async () => {
      // Wave #3 control: with NO stop_sequences the streamed delta " H" opens a
      // text block immediately (non-whitespace), and the native done text " Htail"
      // recovers only the unsent "tail". The combined stream is exactly " Htail" —
      // byte-identical to the no-stop behavior, no duplication, no drop.
      const registry = new ModelRegistry();
      const streamEvents = [
        { text: ' H', done: false, isReasoning: false },
        {
          text: ' Htail',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 6,
          promptTokens: 3,
          reasoningTokens: 0,
          rawText: ' Htail',
        },
      ];
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getBody } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'go' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );

      const events = parseSSE(getBody());
      const textDeltas = events.filter(
        (e) => e.event === 'content_block_delta' && (e.data['delta'] as any).type === 'text_delta',
      );
      const streamedText = textDeltas.map((d) => (d.data['delta'] as any).text as string).join('');
      expect(streamedText).toBe(' Htail');

      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect((msgDelta!.data['delta'] as any).stop_reason).toBe('end_turn');
      expect((msgDelta!.data['delta'] as any).stop_sequence).toBe(null);
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
      // A mid-decode throw from the native async generator must route through the
      // failure epilogue, not the generic outer catch. Invariants pinned below:
      //   * `message_stop` / `message_delta` MUST NOT appear.
      //   * Exactly one `error` event with `type: 'api_error'` citing the thrown error.
      //   * Any `content_block_start` opened before the throw MUST be closed with
      //     `content_block_stop` before the error frame (no dangling block state).
      //   * The session registry stays empty — Anthropic never adopts on failure.
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
      const errorBody = errorEvents[0].data['error'] as {
        type: string;
        message: string;
      };
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
      // A client disconnect mid-stream must flip `clientAborted` via close/error
      // listeners on `httpReq`; the loop-top guard then `break`s into the failure
      // epilogue. We simulate the disconnect by emitting a synthetic `close` event
      // on a lightweight IncomingMessage-shaped mock after the first delta.
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
      const errorBody = errorEvents[0].data['error'] as {
        type: string;
        message: string;
      };
      expect(errorBody.type).toBe('api_error');
      expect(errorBody.message).toMatch(/client disconnected/i);

      // The session registry stays empty on a client abort,
      // mirroring the other failure paths.
      const sessionReg = registry.getSessionRegistry('test-model');
      expect(sessionReg!.size).toBe(0);
    });

    it('iter-35 finding 1: AbortSignal is propagated through the session to the streaming entry point on client disconnect', async () => {
      // The outer handler installs an `AbortController` on `res`/`httpReq` close
      // events and plumbs the signal through `ChatSession.startFromHistoryStream` →
      // `chatStreamSessionStart` → `_runChatStream`. The streaming entry point must
      // therefore receive an AbortSignal whose `aborted` flag flips the moment
      // `httpReq` fires `'close'`. The mock observes the signal and completes on
      // abort — modelling `_runChatStream`'s fast-abort without the native addon.
      let observedSignal: AbortSignal | undefined;
      let resolveAbortSeen: (() => void) | undefined;
      const abortSeen = new Promise<void>((r) => {
        resolveAbortSeen = r;
      });
      async function* signalAwareStream(
        _messages: unknown,
        _config: unknown,
        signal: AbortSignal | undefined,
      ): AsyncGenerator<Record<string, unknown>> {
        observedSignal = signal;
        yield { done: false, text: 'first', isReasoning: false };
        await new Promise<void>((resolve) => {
          if (signal?.aborted) {
            resolve();
            return;
          }
          signal?.addEventListener('abort', () => resolve(), { once: true });
        });
        resolveAbortSeen?.();
        // Synthetic failure terminal: the handler writes a
        // streaming `error` event, not `message_stop`, which is
        // exactly the failure-epilogue shape the real
        // `_runChatStream` abort path produces.
        yield {
          done: true,
          text: '',
          finishReason: 'error',
          toolCalls: [] as ToolCallResult[],
          thinking: null,
          numTokens: 0,
          promptTokens: 0,
          reasoningTokens: 0,
          rawText: '',
        };
      }
      const mockModel = {
        chatSessionStart: vi.fn().mockRejectedValue(new Error('should use stream')),
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: vi.fn(signalAwareStream),
        chatStreamSessionContinue: vi.fn(signalAwareStream),
        chatStreamSessionContinueTool: vi.fn(signalAwareStream),
        resetCaches: vi.fn(),
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('stall-model', mockModel);
      const { res } = createMockRes();

      const { EventEmitter } = require('node:events');
      const reqEmitter = new EventEmitter();
      const httpReqMock = Object.assign(reqEmitter, {
        method: 'POST',
        url: '/v1/messages',
        headers: { 'content-type': 'application/json', host: 'localhost' },
      });

      const start = Date.now();
      const inflight = handleCreateMessage(
        res,
        {
          model: 'stall-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
        httpReqMock as any,
      );
      await new Promise((r) => setImmediate(r));
      await new Promise((r) => setImmediate(r));
      httpReqMock.emit('close');
      await abortSeen;
      await inflight;
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(500);
      expect(observedSignal).toBeDefined();
      expect(observedSignal?.aborted).toBe(true);
    });
  });

  // -----------------------------------------------------------------------
  // SessionRegistry integration (findings 1-3 regressions)
  // -----------------------------------------------------------------------

  describe('session registry integration', () => {
    it('forwards a top-level system string into the mapped chatSessionStart history', async () => {
      // The Anthropic endpoint is stateless on the wire: every request
      // looks up the per-model warm slot via `getOrCreateWarmAny`. On
      // turn 1 (cold start) the lookup misses, so the handler runs a
      // full `session.reset()` (which in turn calls `model.resetCaches()`)
      // before priming the history. The system prompt still needs to
      // land in the primed history so the native side sees it; this
      // test pins that wiring and the always-resets-via-one-of-two-paths
      // invariant that landed with the warm-slot feature.
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

      // Post Task 1: the warm-slot feature adopts the session on a
      // successful turn under the sentinel id `'__msg_warm__'` so the
      // NEXT request with byte-equal `system` can lease it. Exactly
      // one of (`reset()`, `resetPreservingNativeCacheForWarmReuse`)
      // fires per turn — turn 1 is a miss, so the cold-reset path
      // ran and `resetCaches` was called at least once.
      const sessionReg = registry.getSessionRegistry('test-model');
      expect(sessionReg!.size).toBe(1);
      // oxlint-disable-next-line @typescript-eslint/unbound-method
      const resetCachesSpy = mockModel.resetCaches as unknown as ReturnType<typeof vi.fn>;
      expect(resetCachesSpy).toHaveBeenCalled();
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
      // On an uncommitted terminal (`ChatSession.sawFinal` rolls back `turnCount`
      // when `finishReason === 'error'`, so `wasCommitted()` reads false), emit a
      // single Anthropic-shaped `error` SSE event and omit `message_stop` — mirroring
      // the `/v1/responses` commit gate so a failed turn isn't labelled clean.
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

      // `message_stop` and its paired `message_delta` MUST NOT appear on a failed turn.
      const msgStop = events.find((e) => e.event === 'message_stop');
      expect(msgStop).toBeUndefined();
      const msgDelta = events.find((e) => e.event === 'message_delta');
      expect(msgDelta).toBeUndefined();

      // A single Anthropic streaming `error` event with the conventional envelope.
      const errorEvent = events.find((e) => e.event === 'error');
      expect(errorEvent).toBeDefined();
      expect(errorEvent!.data['type']).toBe('error');
      const errorBody = errorEvent!.data['error'] as {
        type: string;
        message: string;
      };
      expect(errorBody.type).toBe('api_error');
      expect(errorBody.message).toMatch(/finishReason=error|did not commit/i);

      // No cached session ever reaches the registry on a failed streaming turn.
      const sessionReg = registry.getSessionRegistry('test-model');
      expect(sessionReg!.size).toBe(0);
    });

    it('emits a streaming error event when the underlying async iterator exhausts without a done event', async () => {
      // If the native iterator exhausts mid-flight without a `done: true` frame, the
      // session's `sawFinal` runs without commit and `wasCommitted()` reads false.
      // The handler emits an Anthropic `error` SSE event instead of `message_stop`.
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
      const errorBody = errorEvents[0].data['error'] as {
        type: string;
        message: string;
      };
      expect(errorBody.type).toBe('api_error');
      expect(errorBody.message).toMatch(/without a done event|stream ended/i);

      // Content block that was opened mid-stream must be closed
      // before the error frame so the client isn't left with a
      // dangling in-progress text block.
      const blockStops = events.filter((e) => e.event === 'content_block_stop');
      expect(blockStops.length).toBeGreaterThanOrEqual(1);
    });

    it('still emits message_delta + message_stop on a clean streaming completion (counter-test)', async () => {
      // A happy-path streaming turn (commits with a non-error `finishReason`) must
      // still produce `message_delta` + `message_stop` and NOT an `error` event.
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
  // Stateless fan-out tool order canonicalization
  // -----------------------------------------------------------------------

  describe('stateless fan-out tool order', () => {
    it('canonicalizes reversed sibling tool_result order to match the assistant fan-out', async () => {
      // `/v1/messages` is ALWAYS a stateless cold-start; the caller ships a full
      // conversation including tool_use/tool_result blocks. Several native backends
      // pair tool results to fan-out calls POSITIONALLY, so
      // `validateAndCanonicalizeHistoryToolOrder` must reorder caller-supplied
      // tool_result messages to match the assistant's declared sibling order before
      // they reach `primeHistory()` — otherwise each output binds to the wrong call.
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
                {
                  type: 'tool_use',
                  id: 'call_a',
                  name: 'get_weather',
                  input: { city: 'SF' },
                },
                {
                  type: 'tool_use',
                  id: 'call_b',
                  name: 'get_news',
                  input: { q: 'tech' },
                },
              ],
            },
            {
              role: 'user',
              content: [
                // Intentionally reversed order — the handler must
                // canonicalize to [call_a, call_b] before dispatch.
                {
                  type: 'tool_result',
                  tool_use_id: 'call_b',
                  content: '{"headlines":[]}',
                },
                {
                  type: 'tool_result',
                  tool_use_id: 'call_a',
                  content: '{"temp":68}',
                },
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
      // `canonicalizeToolMessageOrder` must scan only to the next assistant boundary;
      // scanning to `messages.length` lets it see tool messages from later blocks,
      // trip its count gate, and silently leave a reversed earlier block uncorrected.
      // Two fan-outs: the first's tool_result blocks are reversed, the second's are
      // canonical — both must end up in sibling order with contents tracking ids.
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
                {
                  type: 'tool_use',
                  id: 'call_1',
                  name: 'get_a',
                  input: { k: 'a' },
                },
                {
                  type: 'tool_use',
                  id: 'call_2',
                  name: 'get_b',
                  input: { k: 'b' },
                },
              ],
            },
            {
              role: 'user',
              content: [
                // First fan-out's tool_result blocks reversed.
                {
                  type: 'tool_result',
                  tool_use_id: 'call_2',
                  content: '{"v":"b-result"}',
                },
                {
                  type: 'tool_result',
                  tool_use_id: 'call_1',
                  content: '{"v":"a-result"}',
                },
              ],
            },
            {
              role: 'assistant',
              content: [
                {
                  type: 'tool_use',
                  id: 'call_3',
                  name: 'get_c',
                  input: { k: 'c' },
                },
                {
                  type: 'tool_use',
                  id: 'call_4',
                  name: 'get_d',
                  input: { k: 'd' },
                },
              ],
            },
            {
              role: 'user',
              content: [
                // Second fan-out already canonical.
                {
                  type: 'tool_result',
                  tool_use_id: 'call_3',
                  content: '{"v":"c-result"}',
                },
                {
                  type: 'tool_result',
                  tool_use_id: 'call_4',
                  content: '{"v":"d-result"}',
                },
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
                {
                  type: 'tool_use',
                  id: 'call_a',
                  name: 'get_weather',
                  input: { city: 'SF' },
                },
                {
                  type: 'tool_use',
                  id: 'call_b',
                  name: 'get_news',
                  input: { q: 'tech' },
                },
              ],
            },
            {
              role: 'user',
              content: [
                {
                  type: 'tool_result',
                  tool_use_id: 'call_a',
                  content: '{"temp":68}',
                },
                {
                  type: 'tool_result',
                  tool_use_id: 'call_b',
                  content: '{"headlines":[]}',
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
      const toolMessages = primedMessages.filter((m) => m.role === 'tool');
      expect(toolMessages.map((m) => m.toolCallId)).toEqual(['call_a', 'call_b']);
      // Contents must line up with their original ids — a naive
      // swap that only moved ids without content would fail here.
      expect(toolMessages[0]!.content).toBe('{"temp":68}');
      expect(toolMessages[1]!.content).toBe('{"headlines":[]}');
    });

    it('returns 400 on a malformed fan-out missing a sibling tool_result', async () => {
      // A declared sibling with no matching tool_result must be rejected — submitting
      // only `call_a`'s result when the fan-out was [call_a, call_b] orphans call_b.
      // The follow-up user turn adds a plain-text user message after the tool_result
      // turn (a legal non-mixed shape) that still trips the validator because the
      // assistant fan-out is never fully resolved.
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
                {
                  type: 'tool_use',
                  id: 'call_a',
                  name: 'get_weather',
                  input: { city: 'SF' },
                },
                {
                  type: 'tool_use',
                  id: 'call_b',
                  name: 'get_news',
                  input: { q: 'tech' },
                },
              ],
            },
            {
              role: 'user',
              content: [
                // Only call_a is resolved — call_b is missing.
                {
                  type: 'tool_result',
                  tool_use_id: 'call_a',
                  content: '{"temp":68}',
                },
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
      // Error vocabulary on `/v1/messages` is Anthropic-flavoured: `tool_result` and
      // `assistant turn with tool_use blocks`, NOT `function_call_output`/`fan-out`.
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
              content: [
                {
                  type: 'tool_use',
                  id: 'call_a',
                  name: 'get_weather',
                  input: { city: 'SF' },
                },
              ],
            },
            {
              role: 'user',
              content: [
                {
                  type: 'tool_result',
                  tool_use_id: 'call_ghost',
                  content: '{"temp":68}',
                },
              ],
            },
          ],
          max_tokens: 100,
        } as any,
        registry,
      );

      expect(getStatus()).toBe(400);
      const parsed = JSON.parse(getBody());
      // Anthropic vocabulary: `assistant turn with tool_use blocks` and `tool_use_id`.
      expect(parsed.error.message).toMatch(/not declared by the preceding assistant turn with tool_use blocks/);
      expect(parsed.error.message).toMatch(/tool_use_id/);
      expect(parsed.error.message).not.toMatch(/function_call_output|\bcall_id\b/);
    });

    it('rejects mixed text + tool_result in a single user turn with 400', async () => {
      // The mapper rejects the mixed-shape turn outright rather than silently
      // hoisting tool_result blocks and emitting residual text as a synthetic
      // trailing user message — that earlier behaviour was lossy reordering.
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
                {
                  type: 'tool_use',
                  id: 'call_a',
                  name: 'get_weather',
                  input: { city: 'SF' },
                },
                {
                  type: 'tool_use',
                  id: 'call_b',
                  name: 'get_news',
                  input: { q: 'tech' },
                },
              ],
            },
            {
              role: 'user',
              content: [
                { type: 'text', text: 'here are outputs' },
                {
                  type: 'tool_result',
                  tool_use_id: 'call_b',
                  content: '{"v":"b"}',
                },
                {
                  type: 'tool_result',
                  tool_use_id: 'call_a',
                  content: '{"v":"a"}',
                },
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
      // The mapper accepts tool_result blocks as a contiguous PREFIX (followed by
      // trailing text/image) but rejects the inverse shape where text/image precedes
      // a tool_result. This pins the non-prefix rejection.
      expect(parsed.error.message).toMatch(/tool_result blocks must appear as a contiguous prefix/i);
      // eslint-disable-next-line @typescript-eslint/unbound-method
      const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
      expect(startSpy).not.toHaveBeenCalled();
    });

    it('accepts split turns: tool_result-only user turn followed by a separate text user turn', async () => {
      // A caller that splits the mixed turn into two legal user turns (one carrying
      // ONLY tool_result blocks, one carrying the follow-up text) must dispatch
      // cleanly. Reversed tool order on the first turn verifies canonicalization
      // still runs end-to-end.
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
                {
                  type: 'tool_use',
                  id: 'call_a',
                  name: 'get_weather',
                  input: { city: 'SF' },
                },
                {
                  type: 'tool_use',
                  id: 'call_b',
                  name: 'get_news',
                  input: { q: 'tech' },
                },
              ],
            },
            {
              role: 'user',
              content: [
                // Reversed — canonicalization will reorder to [call_a, call_b].
                {
                  type: 'tool_result',
                  tool_use_id: 'call_b',
                  content: '{"v":"b"}',
                },
                {
                  type: 'tool_result',
                  tool_use_id: 'call_a',
                  content: '{"v":"a"}',
                },
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
        Array<{
          role: string;
          content: string;
          toolCallId?: string;
          toolCalls?: Array<{ id: string }>;
        }>,
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

    it('primes tool_result.is_error=true via the structured isError field through the full /v1/messages dispatch', async () => {
      // End-to-end smoke test: the Anthropic mapper translates
      // `tool_result.is_error === true` into the structured `isError`
      // field on the internal `ChatMessage` (mirroring `toolCallId`) and
      // leaves `content` byte-for-byte equal to the original payload.
      // The Rust-side wire renderer injects the model-facing
      // `[tool error]` cue at serialization time, but the primed history
      // visible to the test never contains an in-band marker — replay
      // and audit paths see the unmodified bytes plus the structured
      // failure signal.
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
              content: [
                {
                  type: 'tool_use',
                  id: 'call_fail',
                  name: 'get_weather',
                  input: { city: 'SF' },
                },
              ],
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
        Array<{ role: string; content: string; toolCallId?: string; isError?: boolean }>,
      ];
      const toolMsg = primedMessages.find((m) => m.role === 'tool' && m.toolCallId === 'call_fail');
      expect(toolMsg).toBeDefined();
      // Structured field carries the failure signal; content is verbatim.
      expect(toolMsg!.isError).toBe(true);
      expect(toolMsg!.content).toBe('boom: connection refused');
    });
  });

  // -----------------------------------------------------------------------
  // Hot-swap race guard inside the per-model execution mutex
  // -----------------------------------------------------------------------

  describe('in-mutex binding re-read (iter-25 finding 2)', () => {
    it('rejects a queued request when the binding is re-registered while the mutex holds a prior dispatch', async () => {
      // The Anthropic handler must re-read the binding INSIDE `withExclusive`: a
      // request queued behind a long decode would otherwise execute through a stale
      // `SessionRegistry` captured pre-lock if `register()` rebinds the name mid-wait.
      // Unlike `/v1/responses`, the Anthropic path has no later stored-identity check,
      // so the in-mutex re-read is the only line of defence — it rejects 400 on drift.
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
  // Wire-contract / visibility fixes on /v1/messages
  // -----------------------------------------------------------------------

  describe('transport visibility (iter-33 finding 3)', () => {
    it('non-streaming: JSON-mode async end-callback failure destroys the socket instead of emitting SSE', async () => {
      // The outer catch must NOT emit an SSE frame into a `Content-Type:
      // application/json` body when `res.end()` reports an async callback failure
      // (socket breaks after Node queued the payload). `endJson` commits the wire
      // format via `responseMode = 'json'`; on a JSON-mode failure we destroy the
      // socket so the client sees a truncated JSON body, never mixed MIME frames.
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

      // Wire contract: socket destroyed, body contains NO SSE frame. `event: error`
      // or `data: ` in a JSON response is exactly the corruption this test prevents.
      expect(wasDestroyed()).toBe(true);
      const body = getBody();
      expect(body).not.toContain('event: error');
      expect(body).not.toMatch(/^data: /m);
    });

    it('streaming: early SSE write crash before any terminal emits an error frame and does not hang', async () => {
      // `terminalEmitted` gates off the actual terminal write callback; the outer
      // catch branches on `responseMode` + `terminalEmitted`, so a mid-decode
      // `res.write` crash before any terminal emits a fallback `error` SSE frame
      // without risking a double terminal when a real one was already delivered.
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

    it('non-streaming: destroyed socket before end rejects endJson and does not hang', async () => {
      // `res.end(payload, cb)` does not invoke the callback on an already-destroyed
      // socket — awaiting it would pin the per-model `withExclusive` mutex on a
      // dead client. `endJson` pre-checks `res.destroyed || res.socket?.destroyed`
      // and rejects synchronously.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ text: 'hi' }));
      registry.register('test-model', mockModel);
      const { res, getBody, wasDestroyed } = createMockRes();

      Object.defineProperty(res, 'socket', {
        configurable: true,
        get: () => ({
          destroyed: true,
          once: () => {},
          removeListener: () => {},
          off: () => {},
        }),
      });

      const handlerPromise = handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        } as any,
        registry,
      );
      await Promise.race([
        handlerPromise,
        new Promise<void>((_, reject) =>
          setTimeout(() => reject(new Error('handler hung waiting for destroyed-socket endJson callback')), 1000),
        ),
      ]);

      // JSON-mode failure destroyed the socket and emitted no SSE
      // frame into the JSON body.
      expect(wasDestroyed()).toBe(true);
      const body = getBody();
      expect(body).not.toContain('event: ');
      expect(body).not.toMatch(/^data: /m);
    });

    it('non-streaming: close event during end rejects endJson and does not hang', async () => {
      // If the peer disconnects AFTER `res.end()` returns but BEFORE the kernel acks,
      // Node emits `'close'` on the response (or socket) and the end callback never
      // fires. `endJson` attaches `res.once('close', …)` so peer disconnect rejects
      // the helper's promise.
      const registry = new ModelRegistry();
      const mockModel = createMockModel(makeChatResult({ text: 'hi' }));
      registry.register('test-model', mockModel);
      const { res, getBody, wasDestroyed } = createMockRes();

      let endCallCount = 0;
      const originalEnd = res.end.bind(res);
      (res as unknown as { end: (...args: unknown[]) => unknown }).end = (
        chunkArg?: unknown,
        encodingOrCbArg?: unknown,
        maybeCbArg?: unknown,
      ) => {
        endCallCount++;
        if (endCallCount === 1) {
          setTimeout(() => {
            res.emit('close');
          }, 0);
          return res;
        }
        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
        return originalEnd(chunkArg as any, encodingOrCbArg as any, maybeCbArg as any);
      };

      const handlerPromise = handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        } as any,
        registry,
      );
      await Promise.race([
        handlerPromise,
        new Promise<void>((_, reject) =>
          setTimeout(() => reject(new Error('handler hung waiting for close-driven endJson rejection')), 1000),
        ),
      ]);

      expect(wasDestroyed()).toBe(true);
      const body = getBody();
      expect(body).not.toContain('event: ');
      expect(body).not.toMatch(/^data: /m);
    });
  });

  describe('X-Session-Cache observability header', () => {
    // `/v1/messages` is stateless — every request calls
    // `sessionReg.getOrCreate(null, …)` — so the header is always
    // `fresh`. The assertion exists to keep the header contract
    // uniform with `/v1/responses` so operator tooling can pin on
    // it regardless of endpoint.

    it('messages endpoint always emits X-Session-Cache: fresh on non-streaming', async () => {
      const registry = new ModelRegistry();
      registry.register('test-model', createMockModel());
      const { res, getStatus, getHeaders } = createMockRes();

      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      expect(getHeaders()['x-session-cache']).toBe('fresh');
    });

    it('messages endpoint emits X-Session-Cache: streaming on streaming responses', async () => {
      const streamEvents = [
        { done: false, text: 'hi', isReasoning: false },
        {
          done: true,
          text: 'hi',
          finishReason: 'stop',
          toolCalls: [] as ToolCallResult[],
          thinking: null,
          numTokens: 1,
          promptTokens: 1,
          reasoningTokens: 0,
          rawText: 'hi',
        },
      ];
      const registry = new ModelRegistry();
      registry.register('test-model', createMockStreamModel(streamEvents));
      const { res, getHeaders } = createMockRes();

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

      // Streaming responses emit the literal `'streaming'` value to
      // signal that the authoritative cache classification is on the
      // SSE stream (`message_delta.usage.cache_read_input_tokens`)
      // and HTTP `X-Cached-Tokens` trailer rather than this header.
      // SSE headers are committed by `beginSSE`; the observability
      // header lands alongside them because it was set BEFORE
      // `writeHead` fired.
      expect(getHeaders()['x-session-cache']).toBe('streaming');
      expect(getHeaders()['content-type']).toBe('text/event-stream');
    });

    it('messages endpoint can lease an instructions-equal warm slot via getOrCreateWarmAny (single-warm cross-endpoint reuse)', async () => {
      // Post-Task 1 contract: `/v1/messages` no longer passes
      // `prompt_cache_key` (the field is not on the type) and no longer
      // routes through tier-1 / tier-2 lookups at all. Instead it calls
      // `SessionRegistry.getOrCreateWarmAny(requestedSystem)`, which
      // walks the registry's at-most-one warm entry and leases it on
      // BYTE-EQUAL `instructions`, IGNORING the entry's prior
      // `previousResponseId` keying and `promptCacheKey`.
      //
      // The behaviour-change this test pins:
      //   * A pre-seeded entry adopted by /v1/responses with
      //     `instructions === requestedSystem` IS a valid warm slot for
      //     /v1/messages — `getOrCreateWarmAny` will lease it.
      //   * On HIT the handler runs the JS-only warm-reuse helper
      //     (`resetPreservingNativeCacheForWarmReuse`) instead of a
      //     full `session.reset()`. The proxy: `model.resetCaches()`
      //     fires on a full reset and DOES NOT fire on the warm-reuse
      //     helper (see `chat-session-warm-reuse.ts`).
      //   * The pre-seeded warm wrapper IS the one running the turn
      //     (verified via `primeHistory` spy on the same instance).
      //   * After commit the slot is re-adopted under the sentinel
      //     `'__msg_warm__'`, NOT under the original `'resp_pre_seed'`
      //     — that is the expected single-warm cross-endpoint
      //     eviction documented on `getOrCreateWarmAny`.
      const mockModel = createMockModel();
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);

      const sessionReg = registry.getSessionRegistry('test-model')!;
      const warmSession = new ChatSession(mockModel);
      const primeSpy = vi.spyOn(warmSession, 'primeHistory');
      const resetSpy = vi.spyOn(warmSession, 'reset');
      // Adopt under the byte-equal `instructions = "sysA"` so
      // `getOrCreateWarmAny('sysA')` will hit. The `chain-xyz`
      // prompt-cache-key is deliberately left in place as a sanity
      // check — `getOrCreateWarmAny` ignores it.
      sessionReg.adopt('resp_pre_seed', warmSession, 'sysA', 'chain-xyz');
      // oxlint-disable-next-line @typescript-eslint/unbound-method
      const resetCachesSpy = mockModel.resetCaches as unknown as ReturnType<typeof vi.fn>;
      const resetCachesBefore = resetCachesSpy.mock.calls.length;

      const { res, getStatus } = createMockRes();
      await handleCreateMessage(
        res,
        {
          model: 'test-model',
          system: 'sysA',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );

      expect(getStatus()).toBe(200);
      // Warm-reuse helper fired (NOT a full reset): the pre-seeded
      // session's `reset()` method was NOT called — the helper writes
      // to private fields directly — and `model.resetCaches()` did
      // not advance.
      expect(resetSpy).not.toHaveBeenCalled();
      expect(resetCachesSpy.mock.calls.length).toBe(resetCachesBefore);
      // The pre-seeded warm wrapper was the one running the turn.
      expect(primeSpy).toHaveBeenCalledTimes(1);
      // After commit the slot is re-adopted under the sentinel id
      // `'__msg_warm__'`. The `entries.clear()` inside
      // `getOrCreateWarmAny` clobbered the prior 'resp_pre_seed' key,
      // and `adopt` re-keyed the same session under the sentinel.
      expect(sessionReg.size).toBe(1);
    });
  });

  // -----------------------------------------------------------------------
  // Queue-depth backpressure (HTTP 429)
  // -----------------------------------------------------------------------

  describe('queue cap', () => {
    it('POST /v1/messages returns 429 Anthropic-format when queue is full', async () => {
      // Stateless Anthropic endpoint still serializes per-model through
      // `SessionRegistry.withExclusive`, so the admission-control 429
      // path applies here too. Shape matches Anthropic: 429 status,
      // `Retry-After: 1` header, body `{ type: 'error', error: { type:
      // 'rate_limit_error', message: '...' } }`.
      let releaseFirst!: () => void;
      const firstHold = new Promise<void>((r) => {
        releaseFirst = r;
      });
      const model = {
        chatSessionStart: vi.fn(async () => {
          await firstHold;
          return makeChatResult({ text: 'ok' });
        }),
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path not expected')),
        chatStreamSessionStart: vi.fn(),
        chatStreamSessionContinue: vi.fn(),
        chatStreamSessionContinueTool: vi.fn(),
        resetCaches: vi.fn(),
      } as unknown as SessionCapableModel;

      const registry = new ModelRegistry({ maxQueueDepth: 1 });
      registry.register('cap-model', model);

      // A: active holder.
      const mockA = createMockRes();
      const promiseA = handleCreateMessage(
        mockA.res,
        {
          model: 'cap-model',
          messages: [{ role: 'user', content: 'A' }],
          max_tokens: 32,
        },
        registry,
      );
      await new Promise((r) => setImmediate(r));

      // B: first waiter (within cap).
      const mockB = createMockRes();
      const promiseB = handleCreateMessage(
        mockB.res,
        {
          model: 'cap-model',
          messages: [{ role: 'user', content: 'B' }],
          max_tokens: 32,
        },
        registry,
      );
      await new Promise((r) => setImmediate(r));

      // C: second waiter, cap exceeded -> 429 Anthropic shape.
      const mockC = createMockRes();
      await handleCreateMessage(
        mockC.res,
        {
          model: 'cap-model',
          messages: [{ role: 'user', content: 'C' }],
          max_tokens: 32,
        },
        registry,
      );

      expect(mockC.getStatus()).toBe(429);
      expect(mockC.getHeaders()['retry-after']).toBe('1');
      const parsed = JSON.parse(mockC.getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('rate_limit_error');
      expect(parsed.error.message).toContain('Model queue full');

      // Drain pending so the test tears down cleanly.
      releaseFirst();
      await promiseA;
      await promiseB;
    });
  });

  // -----------------------------------------------------------------------
  // Warm-slot KV reuse via getOrCreateWarmAny (Task 1)
  //
  // Coverage matrix for the feature that landed in commits 7424ac8..f9759bf:
  //   * Three-turn replay (non-streaming) reuses the warm slot.
  //   * Three-turn replay (streaming) reuses the warm slot, but the
  //     `X-Session-Cache` header stays at `'fresh'` (intentional —
  //     SSE flushes headers pre-dispatch so there is no pre-flush
  //     proof of native KV reuse to advertise).
  //   * Instructions drift between turns invalidates the slot.
  //   * Mid-decode failure does NOT poison the slot for the next turn.
  //   * Model hot-swap (`unregister` + `register`) drops the slot.
  //   * Cross-endpoint single-warm trade-off: a /v1/messages turn that
  //     leases an instructions-equal entry pre-seeded under any prior
  //     id (e.g. a /v1/responses `'resp_xyz'`) clobbers the original
  //     keying when it adopts under `'__msg_warm__'`.
  //
  // Warm-reuse witness: `model.resetCaches()` is called by
  // `ChatSession.reset()` but NOT by
  // `resetPreservingNativeCacheForWarmReuse(session)` (see
  // `packages/server/src/chat-session-warm-reuse.ts`). Comparing the
  // mock's `resetCaches.mock.calls.length` between turns is therefore
  // a behavioural witness for which reset path fired.
  // -----------------------------------------------------------------------

  describe('warm-slot KV reuse via getOrCreateWarmAny', () => {
    it('three-turn non-streaming replay reuses the warm slot (resetCaches stays flat after turn 1)', async () => {
      // Turn 1: cold start (empty registry). `getOrCreateWarmAny` misses,
      //   handler runs full `session.reset()` -> `resetCaches` fires.
      //   `cachedTokens: 0`, header `fresh`, no `x-cached-tokens`.
      // Turn 2 / Turn 3: warm slot leased. Handler runs the JS-only
      //   `resetPreservingNativeCacheForWarmReuse` (NO resetCaches call).
      //   `cachedTokens > 0`, header `prefix_hit`, `x-cached-tokens` present.
      const startResults = [
        makeChatResult({ text: 'A1', cachedTokens: 0 }),
        makeChatResult({ text: 'A2', cachedTokens: 12 }),
        makeChatResult({ text: 'A3', cachedTokens: 24 }),
      ];
      const chatSessionStart = vi
        .fn()
        .mockResolvedValueOnce(startResults[0])
        .mockResolvedValueOnce(startResults[1])
        .mockResolvedValueOnce(startResults[2]);
      const resetCaches = vi.fn();
      const mockModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        chatStreamSessionContinue: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        chatStreamSessionContinueTool: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        resetCaches,
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);
      const sessionReg = registry.getSessionRegistry('test-model')!;

      // Identity witness: spy on `ChatSession.prototype.primeHistory`.
      // `mock.contexts[i]` records the `this` value of call i. After
      // three turns we assert all three contexts are reference-equal —
      // proves the warm slot is the SAME ChatSession instance leased
      // across turns 1..3, not three fresh sessions that
      // `getOrCreateWarmAny` falsely reported as warm hits.
      const primeHistorySpy = vi.spyOn(ChatSession.prototype, 'primeHistory');
      try {
        // ---- Turn 1 ----
        const r1 = createMockRes();
        await handleCreateMessage(
          r1.res,
          {
            model: 'test-model',
            messages: [{ role: 'user', content: 'A' }],
            max_tokens: 100,
          },
          registry,
        );
        expect(r1.getStatus()).toBe(200);
        expect(r1.getHeaders()['x-session-cache']).toBe('fresh');
        expect(r1.getHeaders()['x-cached-tokens']).toBeUndefined();
        expect(sessionReg.size).toBe(1);
        // Turn 1 cold reset fired.
        const resetCachesAfterT1 = resetCaches.mock.calls.length;
        expect(resetCachesAfterT1).toBeGreaterThanOrEqual(1);

        // ---- Turn 2 ----
        const r2 = createMockRes();
        await handleCreateMessage(
          r2.res,
          {
            model: 'test-model',
            messages: [
              { role: 'user', content: 'A' },
              { role: 'assistant', content: 'A1' },
              { role: 'user', content: 'B' },
            ],
            max_tokens: 100,
          },
          registry,
        );
        expect(r2.getStatus()).toBe(200);
        // Warm slot leased and native reuse confirmed (cachedTokens > 0).
        expect(r2.getHeaders()['x-session-cache']).toBe('prefix_hit');
        expect(r2.getHeaders()['x-cached-tokens']).toBe('12');
        expect(sessionReg.size).toBe(1);
        // CRITICAL: `resetCaches` was NOT called between turn 1 and
        // turn 2 — the warm-reuse helper wipes JS state only.
        expect(resetCaches.mock.calls.length).toBe(resetCachesAfterT1);

        // ---- Turn 3 ----
        const r3 = createMockRes();
        await handleCreateMessage(
          r3.res,
          {
            model: 'test-model',
            messages: [
              { role: 'user', content: 'A' },
              { role: 'assistant', content: 'A1' },
              { role: 'user', content: 'B' },
              { role: 'assistant', content: 'A2' },
              { role: 'user', content: 'C' },
            ],
            max_tokens: 100,
          },
          registry,
        );
        expect(r3.getStatus()).toBe(200);
        expect(r3.getHeaders()['x-session-cache']).toBe('prefix_hit');
        expect(r3.getHeaders()['x-cached-tokens']).toBe('24');
        expect(sessionReg.size).toBe(1);
        // Warm-reuse helper still wins — no further resetCaches call.
        expect(resetCaches.mock.calls.length).toBe(resetCachesAfterT1);

        // All three turns went through the cold-start native entry
        // point because `/v1/messages` always replays the full history
        // (the chat-session delta API cannot splice a multi-message
        // tail). The `chatSessionContinue` rejecting stub guarantees
        // we never accidentally took the hot path.
        expect(chatSessionStart).toHaveBeenCalledTimes(3);

        // Transcript witness: each turn must dispatch the FULL mapped
        // history through `chatSessionStart`, not just the trailing
        // user message. `ChatSession.startFromHistory` calls
        // `model.chatSessionStart(this.history.slice(), config)` after
        // `primeHistory(messages)` deep-copies the messages array, so
        // `mock.calls[i][0]` is the exact history seeded on turn i. A
        // regression where the handler primed only `[lastUser]` (or
        // dropped intermediate assistant turns, or otherwise corrupted
        // the full-history append) would still pass identity / reset /
        // cachedTokens checks but break this transcript gate.
        const turn1Messages = chatSessionStart.mock.calls[0]![0] as ChatMessage[];
        expect(turn1Messages).toEqual([{ role: 'user', content: 'A' }]);
        const turn2Messages = chatSessionStart.mock.calls[1]![0] as ChatMessage[];
        expect(turn2Messages).toEqual([
          { role: 'user', content: 'A' },
          { role: 'assistant', content: 'A1' },
          { role: 'user', content: 'B' },
        ]);
        const turn3Messages = chatSessionStart.mock.calls[2]![0] as ChatMessage[];
        expect(turn3Messages).toEqual([
          { role: 'user', content: 'A' },
          { role: 'assistant', content: 'A1' },
          { role: 'user', content: 'B' },
          { role: 'assistant', content: 'A2' },
          { role: 'user', content: 'C' },
        ]);

        // Identity witness check: `primeHistory` ran three times, all
        // bound to the SAME ChatSession instance. A regression where
        // `getOrCreateWarmAny` returned `{ session: <fresh>, hit: true }`
        // would yield distinct `this` values here — the registry +
        // resetCaches assertions above could not catch that.
        expect(primeHistorySpy).toHaveBeenCalledTimes(3);
        const ctx0 = primeHistorySpy.mock.contexts[0];
        const ctx1 = primeHistorySpy.mock.contexts[1];
        const ctx2 = primeHistorySpy.mock.contexts[2];
        expect(ctx0).toBeInstanceOf(ChatSession);
        expect(ctx1).toBe(ctx0);
        expect(ctx2).toBe(ctx0);
      } finally {
        primeHistorySpy.mockRestore();
      }
    });

    it('rotating x-anthropic-billing-header does not bust the warm slot', async () => {
      // Pins Task 5 — the request mapper drops Anthropic's per-request
      // billing/attribution header (Claude Code injects it as the first
      // system block with a `cch=<rotating-token>;` segment that changes
      // every turn). Without the strip:
      //
      //   1. The warm-slot gate (`getOrCreateWarmAny`) compares
      //      `requestedSystem` byte-equally — `JSON.stringify(body.system)`
      //      with the rotating cch= token would always differ between
      //      turns, missing the warm slot every request.
      //   2. Even if the gate matched, the native token-prefix verifier
      //      would still bust the cached prefix because the rotating ~60
      //      token header is part of the prompt.
      //
      // After the strip, both views are stable across turns. The model
      // never sees the billing header (assertion below on the first arg
      // of `chatSessionStart`), and turn 2 hits the warm slot exactly
      // like the simpler three-turn test above — but with a rotating
      // billing block prepended.
      const startResults = [
        makeChatResult({
          text: 'A1',
          cachedTokens: 0,
          promptTokens: 5,
          numTokens: 3,
        }),
        makeChatResult({
          text: 'A2',
          cachedTokens: 12,
          promptTokens: 20,
          numTokens: 5,
        }),
      ];
      const chatSessionStart = vi.fn().mockResolvedValueOnce(startResults[0]).mockResolvedValueOnce(startResults[1]);
      const resetCaches = vi.fn();
      const mockModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        chatStreamSessionContinue: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        chatStreamSessionContinueTool: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        resetCaches,
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);
      const sessionReg = registry.getSessionRegistry('test-model')!;

      // ---- Turn 1 ----
      const r1 = createMockRes();
      await handleCreateMessage(
        r1.res,
        {
          model: 'test-model',
          system: [
            {
              type: 'text',
              text: 'x-anthropic-billing-header: cc_version=2.1.119.806; cch=AAAA;',
            },
            { type: 'text', text: 'You are Claude.' },
          ],
          messages: [{ role: 'user', content: 'A' }],
          max_tokens: 100,
        },
        registry,
      );
      expect(r1.getStatus()).toBe(200);
      expect(r1.getHeaders()['x-session-cache']).toBe('fresh');
      expect(sessionReg.size).toBe(1);
      const resetCachesAfterT1 = resetCaches.mock.calls.length;
      expect(resetCachesAfterT1).toBeGreaterThanOrEqual(1);

      // ---- Turn 2: rotated cch= token, otherwise identical conversation ----
      const r2 = createMockRes();
      await handleCreateMessage(
        r2.res,
        {
          model: 'test-model',
          system: [
            // cch= rotated. Without the strip this whole request would
            // miss the warm slot at the gate level.
            {
              type: 'text',
              text: 'x-anthropic-billing-header: cc_version=2.1.119.806; cch=BBBB;',
            },
            { type: 'text', text: 'You are Claude.' },
          ],
          messages: [
            { role: 'user', content: 'A' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'B' },
          ],
          max_tokens: 100,
        },
        registry,
      );

      // Warm-slot hit assertions — the rotating header MUST NOT bust the
      // gate. These are the gates that fail under a regression.
      expect(r2.getStatus()).toBe(200);
      expect(r2.getHeaders()['x-session-cache']).toBe('prefix_hit');
      expect(r2.getHeaders()['x-cached-tokens']).toBe('12');
      const t2Body = JSON.parse(r2.getBody()) as {
        usage: Record<string, number>;
      };
      expect(t2Body.usage.cache_read_input_tokens).toBe(12);
      expect(t2Body.usage.input_tokens).toBe(8); // 20 - 12
      expect(sessionReg.size).toBe(1);
      // Warm-reuse helper fired (NOT a full reset) — model.resetCaches
      // count is unchanged from turn 1.
      expect(resetCaches.mock.calls.length).toBe(resetCachesAfterT1);

      // Model-input witness: BOTH turns dispatch a system message with
      // the billing header DROPPED. The model only ever sees the stable
      // `"You are Claude."` content.
      expect(chatSessionStart).toHaveBeenCalledTimes(2);
      const turn1Messages = chatSessionStart.mock.calls[0]![0] as ChatMessage[];
      const turn2Messages = chatSessionStart.mock.calls[1]![0] as ChatMessage[];
      expect(turn1Messages[0]).toEqual({
        role: 'system',
        content: 'You are Claude.',
      });
      expect(turn2Messages[0]).toEqual({
        role: 'system',
        content: 'You are Claude.',
      });
    });

    it('three-turn streaming replay reuses the warm slot (header reports streaming)', async () => {
      // Streaming counterpart of the previous test. Behaviour mirror:
      //   * Each turn flushes a clean SSE wire (`message_start` ...
      //     `message_stop`).
      //   * `wasCommitted() && streamResult.ok` lets adopt fire on
      //     each successful turn -> `sessionReg.size === 1`.
      //   * Turn 2 / Turn 3 hit the warm slot -> warm-reuse helper
      //     fires (no resetCaches call advance).
      //   * Header is ALWAYS `'fresh'` on streaming — Task 1 commit
      //     f9759bf documents that streaming intentionally lacks a
      //     pre-flush proof of native KV reuse, so it cannot promote
      //     to `prefix_hit` without acting as a presence side-channel.
      function makeStream(text: string) {
        return async function* () {
          yield { text, done: false, isReasoning: false };
          yield {
            text,
            done: true,
            finishReason: 'stop',
            toolCalls: [] as ToolCallResult[],
            thinking: null,
            numTokens: 3,
            promptTokens: 5,
            reasoningTokens: 0,
            rawText: text,
          };
        };
      }
      const stream1 = vi.fn().mockImplementationOnce(makeStream('A1'));
      stream1.mockImplementationOnce(makeStream('A2')).mockImplementationOnce(makeStream('A3'));
      const resetCaches = vi.fn();
      const mockModel = {
        chatSessionStart: vi.fn().mockRejectedValue(new Error('streaming test')),
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: stream1,
        chatStreamSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        resetCaches,
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('stream-model', mockModel);
      const sessionReg = registry.getSessionRegistry('stream-model')!;

      async function runTurn(messages: Array<{ role: 'user' | 'assistant'; content: string }>) {
        const mock = createMockRes();
        await handleCreateMessage(
          mock.res,
          {
            model: 'stream-model',
            messages,
            max_tokens: 100,
            stream: true,
          },
          registry,
        );
        return { events: parseSSE(mock.getBody()), headers: mock.getHeaders() };
      }

      // Identity witness: see the non-streaming sibling test for the
      // rationale. The streaming handler also calls `primeHistory` once
      // per turn before dispatching `startFromHistoryStream`, so the
      // `mock.contexts` snapshot is the cleanest proof that the warm
      // slot leases the SAME session across turns 2/3.
      const primeHistorySpy = vi.spyOn(ChatSession.prototype, 'primeHistory');
      try {
        // ---- Turn 1 ----
        const t1 = await runTurn([{ role: 'user', content: 'A' }]);
        expect(t1.headers['x-session-cache']).toBe('streaming');
        expect(t1.events[0].event).toBe('message_start');
        expect(t1.events.find((e) => e.event === 'message_stop')).toBeDefined();
        expect(sessionReg.size).toBe(1);
        const resetCachesAfterT1 = resetCaches.mock.calls.length;
        expect(resetCachesAfterT1).toBeGreaterThanOrEqual(1);

        // ---- Turn 2 ----
        const t2 = await runTurn([
          { role: 'user', content: 'A' },
          { role: 'assistant', content: 'A1' },
          { role: 'user', content: 'B' },
        ]);
        // Header reports `streaming` on streaming responses — the
        // authoritative cache classification lives in the SSE
        // `message_delta.usage.cache_read_input_tokens` field and
        // the `X-Cached-Tokens` HTTP trailer.
        expect(t2.headers['x-session-cache']).toBe('streaming');
        expect(t2.events.find((e) => e.event === 'message_stop')).toBeDefined();
        expect(sessionReg.size).toBe(1);
        // Warm-reuse helper fired — no resetCaches call advance.
        expect(resetCaches.mock.calls.length).toBe(resetCachesAfterT1);

        // ---- Turn 3 ----
        const t3 = await runTurn([
          { role: 'user', content: 'A' },
          { role: 'assistant', content: 'A1' },
          { role: 'user', content: 'B' },
          { role: 'assistant', content: 'A2' },
          { role: 'user', content: 'C' },
        ]);
        expect(t3.headers['x-session-cache']).toBe('streaming');
        expect(t3.events.find((e) => e.event === 'message_stop')).toBeDefined();
        expect(sessionReg.size).toBe(1);
        expect(resetCaches.mock.calls.length).toBe(resetCachesAfterT1);

        // All three turns dispatched through the streaming cold-start
        // entry point.
        expect(stream1).toHaveBeenCalledTimes(3);

        // Transcript witness: same as the non-streaming sibling — each
        // turn must dispatch the FULL mapped history through
        // `chatStreamSessionStart`, not just the trailing user message.
        // `ChatSession.startFromHistoryStream` calls
        // `model.chatStreamSessionStart(historySnapshot, ...)` against
        // a slice of the primed history, so `mock.calls[i][0]` is the
        // history seeded on turn i. A regression where the handler
        // primed only `[lastUser]` (or dropped intermediate assistant
        // turns) would slip past the identity / reset gates above but
        // fail this transcript gate.
        const turn1Messages = stream1.mock.calls[0]![0] as ChatMessage[];
        expect(turn1Messages).toEqual([{ role: 'user', content: 'A' }]);
        const turn2Messages = stream1.mock.calls[1]![0] as ChatMessage[];
        expect(turn2Messages).toEqual([
          { role: 'user', content: 'A' },
          { role: 'assistant', content: 'A1' },
          { role: 'user', content: 'B' },
        ]);
        const turn3Messages = stream1.mock.calls[2]![0] as ChatMessage[];
        expect(turn3Messages).toEqual([
          { role: 'user', content: 'A' },
          { role: 'assistant', content: 'A1' },
          { role: 'user', content: 'B' },
          { role: 'assistant', content: 'A2' },
          { role: 'user', content: 'C' },
        ]);

        // Identity witness check: same ChatSession instance leased on
        // turns 1..3.
        expect(primeHistorySpy).toHaveBeenCalledTimes(3);
        const ctx0 = primeHistorySpy.mock.contexts[0];
        const ctx1 = primeHistorySpy.mock.contexts[1];
        const ctx2 = primeHistorySpy.mock.contexts[2];
        expect(ctx0).toBeInstanceOf(ChatSession);
        expect(ctx1).toBe(ctx0);
        expect(ctx2).toBe(ctx0);
      } finally {
        primeHistorySpy.mockRestore();
      }
    });

    it('instructions mismatch invalidates the warm slot — full reset fires on turn 2', async () => {
      // Turn 1 with `system: "S1"` adopts the slot under the sentinel.
      // Turn 2 with `system: "S2"` — `getOrCreateWarmAny` returns a
      // miss (instructions drift), `entries.clear()` runs in the miss
      // branch, and the handler runs a full `session.reset()` so the
      // shared native KV cache is wiped before the new system prompt
      // is primed (the cross-request cache-affinity guard documented
      // in messages.ts). After turn 2 the new session is itself
      // adopted under the sentinel, so `size === 1` again.
      const chatSessionStart = vi
        .fn()
        .mockResolvedValueOnce(makeChatResult({ text: 'first', cachedTokens: 0 }))
        .mockResolvedValueOnce(makeChatResult({ text: 'second', cachedTokens: 0 }));
      const resetCaches = vi.fn();
      const mockModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: vi.fn(),
        chatStreamSessionContinue: vi.fn(),
        chatStreamSessionContinueTool: vi.fn(),
        resetCaches,
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);
      const sessionReg = registry.getSessionRegistry('test-model')!;

      // Turn 1: system "S1".
      const r1 = createMockRes();
      await handleCreateMessage(
        r1.res,
        {
          model: 'test-model',
          system: 'S1',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );
      expect(r1.getStatus()).toBe(200);
      expect(sessionReg.size).toBe(1);
      const resetCachesAfterT1 = resetCaches.mock.calls.length;
      expect(resetCachesAfterT1).toBeGreaterThanOrEqual(1);

      // Turn 2: system "S2" — same conversation otherwise. Slot from
      // turn 1 is invalidated by the byte-equal compare in
      // `getOrCreateWarmAny`.
      const r2 = createMockRes();
      await handleCreateMessage(
        r2.res,
        {
          model: 'test-model',
          system: 'S2',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );
      expect(r2.getStatus()).toBe(200);
      // Header is `fresh` — the lookup missed so no `prefix_hit`
      // promotion, and `cachedTokens === 0` would have demoted it
      // anyway.
      expect(r2.getHeaders()['x-session-cache']).toBe('fresh');
      // Full `session.reset()` fired — `model.resetCaches()` was
      // called at least one more time after turn 1.
      expect(resetCaches.mock.calls.length).toBeGreaterThan(resetCachesAfterT1);
      // The new session was adopted under the sentinel.
      expect(sessionReg.size).toBe(1);
    });

    it('mid-decode failure (streaming finishReason=error) does NOT poison the warm slot', async () => {
      // The `streamResult.ok && wasCommitted()` adopt gate must keep
      // the slot empty when the turn signalled failure. We use the
      // `finishReason: 'error'` variant because it routes through the
      // chat-session uncommitted path AND through the streaming-mock
      // harness already used for the analogous regression in this
      // file (line ~1361). The thrown-mid-decode variant is exercised
      // separately by the existing "routes a mid-decode throw"
      // regression at line ~1048, which already pins
      // `sessionReg.size === 0` post-failure.
      const failingStream = async function* () {
        yield { text: 'partial', done: false, isReasoning: false };
        yield {
          text: 'partial',
          done: true,
          finishReason: 'error',
          toolCalls: [] as ToolCallResult[],
          thinking: null,
          numTokens: 1,
          promptTokens: 3,
          reasoningTokens: 0,
          rawText: 'partial',
        };
      };
      const successfulStream = async function* () {
        yield { text: 'ok', done: false, isReasoning: false };
        yield {
          text: 'ok',
          done: true,
          finishReason: 'stop',
          toolCalls: [] as ToolCallResult[],
          thinking: null,
          numTokens: 1,
          promptTokens: 3,
          reasoningTokens: 0,
          rawText: 'ok',
        };
      };
      const streamSpy = vi.fn().mockImplementationOnce(failingStream).mockImplementationOnce(successfulStream);
      const resetCaches = vi.fn();
      const mockModel = {
        chatSessionStart: vi.fn().mockRejectedValue(new Error('streaming test')),
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: streamSpy,
        chatStreamSessionContinue: vi.fn(),
        chatStreamSessionContinueTool: vi.fn(),
        resetCaches,
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);
      const sessionReg = registry.getSessionRegistry('test-model')!;

      // Turn 1: failing stream. The handler emits a streaming `error`
      // event in place of `message_stop` and the gate denies adopt.
      const r1 = createMockRes();
      await handleCreateMessage(
        r1.res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );
      const t1Events = parseSSE(r1.getBody());
      expect(t1Events.find((e) => e.event === 'message_stop')).toBeUndefined();
      expect(t1Events.find((e) => e.event === 'error')).toBeDefined();
      // Slot dropped — NOT adopted.
      expect(sessionReg.size).toBe(0);
      const resetCachesAfterT1 = resetCaches.mock.calls.length;

      // Turn 2: a fresh request. The slot is empty, so
      // `getOrCreateWarmAny` MISSES — handler runs the full
      // `session.reset()` path (NOT the warm-reuse helper). That
      // means `model.resetCaches()` advances at least one more time.
      const r2 = createMockRes();
      await handleCreateMessage(
        r2.res,
        {
          model: 'test-model',
          messages: [{ role: 'user', content: 'hi again' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );
      const t2Events = parseSSE(r2.getBody());
      expect(t2Events.find((e) => e.event === 'message_stop')).toBeDefined();
      expect(resetCaches.mock.calls.length).toBeGreaterThan(resetCachesAfterT1);
      // Successful turn 2 adopts.
      expect(sessionReg.size).toBe(1);
    });

    it('warm-hit committed-producer failure drops the leased slot (next request cold-starts)', async () => {
      // Companion to the cold-failure test above. The DANGEROUS path
      // — and the one this test must pin — is `wasCommitted() === true
      // && streamResult.ok === false`. A regression that adopts on
      // `wasCommitted()` alone (ignoring `streamResult.ok`) would
      // silently leak the failed session to the next request.
      //
      // Trigger choice: a clean done event whose `toolCalls[].arguments`
      // is malformed JSON. Routing in `handleStreamingNative`
      // (messages.ts:295-317):
      //
      //   1. The done event yields with `finishReason: 'stop'` and
      //      `status: 'ok'` tool calls — `sawDone = true`,
      //      `terminalErrorMessage = null`.
      //   2. The producer (`ChatSession.startFromHistoryStream`) sets
      //      `sawFinal = true` BEFORE yielding (chat-session.ts:806-815),
      //      so the moment control passes to the consumer's body the
      //      producer is already poised to commit on iterator cleanup.
      //   3. The okToolCalls writer loop runs `JSON.parse(tc.arguments)`
      //      on the malformed string, which throws.
      //   4. The throw propagates out of the for-await body. The spec
      //      requires `for-await` to call `iterator.return()` before
      //      re-throwing, which runs the producer's `finally` block —
      //      `sawFinal` is true, so `turnCount++` fires.
      //   5. The consumer's outer `catch` sets `thrownError`. The
      //      post-loop `successful` gate fails on `thrownError != null`
      //      and the writer emits a streaming `error` event, returning
      //      `{ ok: false }`.
      //
      // At the adopt gate in `handleCreateMessage` (messages.ts:880):
      // `streamResult.ok === false` AND `outcome.wasCommitted() === true`
      // (because turnCount advanced in step 4). The gate denies adopt
      // and the handler drops the slot. A regression that gates on
      // `wasCommitted()` alone would re-adopt the failed session, and
      // the next same-instructions request would lease a session whose
      // `turnCount > 0` — `primeHistory()` would then throw, breaking
      // the cold-start replay path.
      //
      // Coverage:
      //   * Identity witness: `primeHistory` runs on the SAME
      //     pre-seeded session instance on turn 1 — proves the warm
      //     hit actually leased it.
      //   * Committed-producer witness: `warmSession.turns === 1` after
      //     the failure — names the exact path under test (commit
      //     fired despite the writer-side failure).
      //   * Failed wire: `error` event without a `message_stop`.
      //   * Slot drop: `sessionReg.size === 0` after the failure —
      //     the leased session was NOT re-adopted under the sentinel.
      //   * Round-trip: a follow-up streaming turn cold-starts
      //     (warm slot is empty, full reset fires on a fresh
      //     session, NOT on the dropped failed one), and adopts.
      const malformedToolCalls: ToolCallResult[] = [
        {
          id: 'toolu_01',
          name: 'do_thing',
          // Strings hit the `typeof tc.arguments === 'string'` branch in
          // messages.ts which calls JSON.parse. Malformed JSON throws.
          arguments: '{not valid json',
          status: 'ok',
          rawContent: '{not valid json',
        },
      ];
      const committedFailingStream = async function* () {
        yield { text: '', done: false, isReasoning: false };
        // Clean done event — `finishReason: 'stop'`, NOT 'error'. This
        // is what makes the producer's `sawFinal` flip true and lets
        // the `finally` commit on cleanup. The malformed arguments
        // payload is what derails the writer loop AFTER commit.
        yield {
          text: '',
          done: true,
          finishReason: 'stop',
          toolCalls: malformedToolCalls,
          thinking: null,
          numTokens: 1,
          promptTokens: 3,
          reasoningTokens: 0,
          rawText: '',
        };
      };
      const successfulStream = async function* () {
        yield { text: 'ok', done: false, isReasoning: false };
        yield {
          text: 'ok',
          done: true,
          finishReason: 'stop',
          toolCalls: [] as ToolCallResult[],
          thinking: null,
          numTokens: 1,
          promptTokens: 3,
          reasoningTokens: 0,
          rawText: 'ok',
        };
      };
      const streamSpy = vi.fn().mockImplementationOnce(committedFailingStream).mockImplementationOnce(successfulStream);
      const resetCaches = vi.fn();
      const mockModel = {
        chatSessionStart: vi.fn().mockRejectedValue(new Error('streaming test')),
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: streamSpy,
        chatStreamSessionContinue: vi.fn(),
        chatStreamSessionContinueTool: vi.fn(),
        resetCaches,
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);
      const sessionReg = registry.getSessionRegistry('test-model')!;

      // Pre-seed the warm slot the way Test 6 does — adopt under
      // a non-sentinel id so we can also confirm the
      // `entries.clear()` in `getOrCreateWarmAny` clobbered the
      // original key (covered indirectly via `sessionReg.size`).
      const warmSession = new ChatSession(mockModel);
      sessionReg.adopt('warm-prefix', warmSession, 'sysA', null);
      expect(sessionReg.size).toBe(1);

      // Identity witness: spy on the prototype (matches the
      // 3-turn streaming test pattern). The streaming dispatcher
      // calls `primeHistory` once on the leased session before
      // `startFromHistoryStream` — `mock.contexts[0]` MUST be
      // `warmSession` to prove the warm hit happened.
      const primeHistorySpy = vi.spyOn(ChatSession.prototype, 'primeHistory');
      try {
        // Turn 1: warm-hit, committed-producer + writer-side failure.
        // Lookup leases `warmSession`; the producer commits on its
        // `finally` (turnCount: 0 -> 1); the writer's JSON.parse on
        // the malformed tool-call args throws; the dual-gate denies
        // adopt; the handler drops the slot.
        const r1 = createMockRes();
        await handleCreateMessage(
          r1.res,
          {
            model: 'test-model',
            system: 'sysA',
            messages: [{ role: 'user', content: 'hi' }],
            max_tokens: 100,
            stream: true,
            tools: [
              {
                name: 'do_thing',
                input_schema: { type: 'object', properties: {} },
              },
            ],
          },
          registry,
        );
        const t1Events = parseSSE(r1.getBody());
        // Wire: error terminal, no message_stop.
        expect(t1Events.find((e) => e.event === 'message_stop')).toBeUndefined();
        expect(t1Events.find((e) => e.event === 'error')).toBeDefined();
        // Identity witness: `primeHistory` ran on the warm
        // session (NOT a fresh one). Rules out the regression
        // where the warm hit is silently bypassed.
        expect(primeHistorySpy).toHaveBeenCalledTimes(1);
        expect(primeHistorySpy.mock.contexts[0]).toBe(warmSession);
        // Committed-producer witness: turnCount advanced on the warm
        // session even though the wire emitted `error`. This is the
        // exact `wasCommitted() === true && streamResult.ok === false`
        // scenario the dual-gate exists to drop. A regression that
        // adopts on `wasCommitted()` alone would slip past this turn.
        expect(warmSession.turns).toBe(1);
        // Slot dropped — the committed-but-failed warm session was
        // NOT re-adopted under the sentinel.
        expect(sessionReg.size).toBe(0);
        const resetCachesAfterT1 = resetCaches.mock.calls.length;

        // Turn 2: fresh request, same instructions. The slot is
        // empty after the drop, so `getOrCreateWarmAny` MISSES
        // and the handler runs a full `session.reset()` on a
        // BRAND-NEW session (NOT the dropped failed one). Lease-
        // poisoning regression test: if turn 1 had falsely re-adopted
        // the dropped session, turn 2 would lease a session whose
        // `turnCount === 1` and `primeHistory()` would throw on the
        // turn-0 invariant. The proxy: `model.resetCaches()` advances
        // at least once.
        const r2 = createMockRes();
        await handleCreateMessage(
          r2.res,
          {
            model: 'test-model',
            system: 'sysA',
            messages: [{ role: 'user', content: 'hi again' }],
            max_tokens: 100,
            stream: true,
          },
          registry,
        );
        const t2Events = parseSSE(r2.getBody());
        // Successful wire.
        expect(t2Events.find((e) => e.event === 'message_stop')).toBeDefined();
        expect(resetCaches.mock.calls.length).toBeGreaterThan(resetCachesAfterT1);
        // The cold-start session is adopted under the sentinel —
        // exactly one entry, and it is NOT the failed `warmSession`.
        expect(sessionReg.size).toBe(1);
        // Turn 2's `primeHistory` was the SECOND call (turn 1 was
        // the first), and its context MUST be a fresh wrapper —
        // NOT the dropped `warmSession`.
        expect(primeHistorySpy).toHaveBeenCalledTimes(2);
        expect(primeHistorySpy.mock.contexts[1]).not.toBe(warmSession);
      } finally {
        primeHistorySpy.mockRestore();
      }
    });

    it('model hot-swap (unregister + register) drops the warm slot', async () => {
      // `ModelRegistry.register` rebuilds the per-model
      // `SessionRegistry`, so the old warm slot is torn down with
      // the old registry. After re-registering, a follow-up turn
      // takes the cold-start path (resetCaches fires on the NEW
      // model) and the new slot is then adopted under the sentinel.
      const modelA = createMockModel(makeChatResult({ text: 'A' }));
      const registry = new ModelRegistry();
      registry.register('m', modelA);

      // Turn 1 against modelA — adopts the warm slot.
      const r1 = createMockRes();
      await handleCreateMessage(
        r1.res,
        {
          model: 'm',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );
      expect(r1.getStatus()).toBe(200);
      const sessionRegA = registry.getSessionRegistry('m')!;
      expect(sessionRegA.size).toBe(1);

      // Hot swap: unregister + register a brand-new model. The
      // SessionRegistry binding is rebuilt; the old slot is gone.
      registry.unregister('m');
      const modelB = createMockModel(makeChatResult({ text: 'B' }));
      registry.register('m', modelB);
      const sessionRegB = registry.getSessionRegistry('m')!;
      // It IS a different SessionRegistry instance.
      expect(sessionRegB).not.toBe(sessionRegA);
      // And it starts empty.
      expect(sessionRegB.size).toBe(0);

      // Turn 2 against the new binding — cold start on modelB, warm
      // slot is empty so resetCaches fires on the NEW model.
      const r2 = createMockRes();
      await handleCreateMessage(
        r2.res,
        {
          model: 'm',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );
      expect(r2.getStatus()).toBe(200);
      // Header `fresh` (cold start, cachedTokens === 0).
      expect(r2.getHeaders()['x-session-cache']).toBe('fresh');
      // Full reset fired on the new model.
      // oxlint-disable-next-line @typescript-eslint/unbound-method
      const resetCachesB = modelB.resetCaches as unknown as ReturnType<typeof vi.fn>;
      expect(resetCachesB).toHaveBeenCalled();
      // The old model's resetCaches did NOT fire on this turn
      // (it is detached from the live binding).
      // The new slot is adopted.
      expect(sessionRegB.size).toBe(1);
    });

    it('cross-endpoint single-warm trade-off: a /v1/messages turn evicts a pre-seeded responses-style entry', async () => {
      // Pre-seed an entry the way `/v1/responses` would (under a
      // `'resp_xyz'` id, a `'sysA'` instructions string, and a
      // `'chain-key'` prompt-cache key). `getOrCreateWarmAny`
      // IGNORES `responseId` and `promptCacheKey` and matches solely
      // on byte-equal `instructions`, so the /v1/messages turn
      // leases the slot on a `system: 'sysA'` request.
      //
      // After the turn commits, the slot is re-adopted under the
      // sentinel `'__msg_warm__'` — a subsequent `/v1/responses`
      // request that tries to resume `'resp_xyz'` via tier-1 will
      // MISS, because `entries.clear()` clobbered the original key
      // when the messages handler leased the slot. This is the
      // explicit single-warm cross-endpoint trade-off documented on
      // `SessionRegistry.getOrCreateWarmAny`'s docstring (and the
      // long block comment in messages.ts around the call site).
      const mockModel = createMockModel(makeChatResult({ text: 'reused', cachedTokens: 7 }));
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);
      const sessionReg = registry.getSessionRegistry('test-model')!;

      // Pre-seed the way /v1/responses would: commit a real turn
      // BEFORE adopting so the warm slot reflects the post-commit
      // state of a real request (`turns === 1`). A freshly
      // constructed `ChatSession` has `turns === 0`, which would
      // make `primeHistory()` succeed even if the warm-reuse helper
      // failed to wipe `turnCount` — silencing the regression where
      // `resetPreservingNativeCacheForWarmReuse` no longer clears
      // JS-side state on a previously committed session. Running a
      // turn through the same mock model that drives the rest of
      // the test pins the realistic fixture and exercises the
      // helper's load-bearing wipe path.
      const warmSession = new ChatSession(mockModel);
      warmSession.primeHistory([{ role: 'user', content: 'seed' }]);
      await warmSession.startFromHistory();
      expect(warmSession.turns).toBe(1);
      const primeSpy = vi.spyOn(warmSession, 'primeHistory');
      const resetSpy = vi.spyOn(warmSession, 'reset');
      sessionReg.adopt('resp_xyz', warmSession, 'sysA', 'chain-key');
      // oxlint-disable-next-line @typescript-eslint/unbound-method
      const resetCaches = mockModel.resetCaches as unknown as ReturnType<typeof vi.fn>;
      const resetCachesBefore = resetCaches.mock.calls.length;

      // /v1/messages turn with `system: 'sysA'`. Lookup leases the
      // pre-seeded session on byte-equal instructions.
      const r1 = createMockRes();
      await handleCreateMessage(
        r1.res,
        {
          model: 'test-model',
          system: 'sysA',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      );
      expect(r1.getStatus()).toBe(200);
      // Warm-reuse helper fired (NOT a full reset on the leased
      // session): `warmSession.reset()` was NOT called and the model's
      // `resetCaches` did not advance.
      expect(resetSpy).not.toHaveBeenCalled();
      expect(resetCaches.mock.calls.length).toBe(resetCachesBefore);
      // Native cache reuse confirmed (`cachedTokens > 0`) so the
      // header promotes to `prefix_hit`.
      expect(r1.getHeaders()['x-session-cache']).toBe('prefix_hit');
      expect(r1.getHeaders()['x-cached-tokens']).toBe('7');
      // The pre-seeded session WAS the one running the turn.
      expect(primeSpy).toHaveBeenCalledTimes(1);
      // Slot is re-adopted under the sentinel — exactly one entry.
      expect(sessionReg.size).toBe(1);

      // Sentinel re-key invariant: the surviving entry MUST be keyed
      // under the literal `'__msg_warm__'` sentinel — that is the
      // contract that makes `/v1/responses` tier-1 lookup of the
      // sentinel impossible by construction (no Anthropic client can
      // produce a `previous_response_id === '__msg_warm__'`). Lease
      // the slot via `getOrCreate('__msg_warm__', 'sysA', null)`:
      // a hit with the SAME `warmSession` reference proves both the
      // sentinel keying AND that the slot still wraps the original
      // pre-seeded ChatSession. NB: `getOrCreate` is leasing —
      // `entries.clear()` runs on every call — so this assertion
      // CONSUMES the slot. The follow-up `'resp_xyz'` lookup below
      // therefore reads an empty registry, which is exactly what the
      // single-warm cross-endpoint trade-off pins.
      const sentinelLookup = sessionReg.getOrCreate('__msg_warm__', 'sysA', null);
      expect(sentinelLookup.hit).toBe(true);
      expect(sentinelLookup.session).toBe(warmSession);

      // Subsequent /v1/responses-style tier-1 lookup against the
      // original `'resp_xyz'` id MISSES — the prior key was
      // clobbered by `entries.clear()` inside `getOrCreateWarmAny`
      // and re-keyed under the sentinel. Operators who need
      // stronger isolation should run separate model bindings or
      // front the server with a tenant-aware proxy (per the
      // single-warm trade-off documented in the registry).
      const probe = sessionReg.getOrCreate('resp_xyz', 'sysA');
      expect(probe.hit).toBe(false);
    });

    it('non-streaming warm hit emits Anthropic cache_read_input_tokens with reduced input_tokens', async () => {
      // Pins the body-level cache-accounting contract on /v1/messages
      // (Task 4 Part 1). Anthropic clients (Claude Code in particular)
      // read `response.usage.cache_read_input_tokens` directly for
      // cost / billing display — a header-only signal is invisible to
      // them, so the response body MUST carry the spec fields whenever
      // the warm slot delivered actual native KV reuse.
      //
      //   Turn 1 — cold (cachedTokens === 0): usage carries
      //     `input_tokens: promptTokens` and NO cache fields.
      //   Turn 2 — warm hit (`cachedTokens: 7`, promptTokens: 20):
      //     usage carries `cache_read_input_tokens: 7` AND
      //     `input_tokens: 13` (= 20 - 7). The legacy
      //     `X-Cached-Tokens` header remains as a redundant ops
      //     signal; `X-Session-Cache: prefix_hit` mirrors the
      //     existing classification.
      const chatSessionStart = vi
        .fn()
        .mockResolvedValueOnce(
          makeChatResult({
            text: 'A1',
            numTokens: 3,
            promptTokens: 5,
            cachedTokens: 0,
          }),
        )
        .mockResolvedValueOnce(
          makeChatResult({
            text: 'A2',
            numTokens: 5,
            promptTokens: 20,
            cachedTokens: 7,
          }),
        );
      const mockModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        chatStreamSessionContinue: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        chatStreamSessionContinueTool: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        resetCaches: vi.fn(),
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);

      // Turn 1: cold — cache fields are OMITTED, not zeroed.
      const r1 = createMockRes();
      await handleCreateMessage(
        r1.res,
        {
          model: 'test-model',
          system: 'S',
          messages: [{ role: 'user', content: 'A' }],
          max_tokens: 100,
        },
        registry,
      );
      expect(r1.getStatus()).toBe(200);
      const t1Body = JSON.parse(r1.getBody()) as {
        usage: Record<string, number>;
      };
      expect(t1Body.usage.input_tokens).toBe(5);
      expect(t1Body.usage.output_tokens).toBe(3);
      expect(t1Body.usage).not.toHaveProperty('cache_read_input_tokens');
      expect(t1Body.usage).not.toHaveProperty('cache_creation_input_tokens');
      expect(r1.getHeaders()['x-cached-tokens']).toBeUndefined();
      expect(r1.getHeaders()['x-session-cache']).toBe('fresh');

      // Turn 2: warm hit — `cache_read_input_tokens` lands in the
      // body, `input_tokens` is the unsuffixed remainder, and
      // `cache_creation_input_tokens` stays OFF the wire (we don't
      // expose explicit cache_control breakpoints).
      const r2 = createMockRes();
      await handleCreateMessage(
        r2.res,
        {
          model: 'test-model',
          system: 'S',
          messages: [
            { role: 'user', content: 'A' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'B' },
          ],
          max_tokens: 100,
        },
        registry,
      );
      expect(r2.getStatus()).toBe(200);
      const t2Body = JSON.parse(r2.getBody()) as {
        usage: Record<string, number>;
      };
      expect(t2Body.usage.cache_read_input_tokens).toBe(7);
      expect(t2Body.usage.input_tokens).toBe(13);
      expect(t2Body.usage.output_tokens).toBe(5);
      expect(t2Body.usage).not.toHaveProperty('cache_creation_input_tokens');
      // Header behaviour unchanged — `X-Cached-Tokens` and
      // `X-Session-Cache` still carry the same redundant signal.
      expect(r2.getHeaders()['x-cached-tokens']).toBe('7');
      expect(r2.getHeaders()['x-session-cache']).toBe('prefix_hit');
    });

    it('streaming message_delta emits cache_read_input_tokens with reduced input_tokens on warm hit', async () => {
      // Streaming counterpart to the non-streaming Anthropic cache
      // accounting test above. The body-level fields ride the
      // `message_delta` event's `usage` block — flushed AFTER the
      // SSE headers are committed — so they are independent of the
      // pre-flush `X-Session-Cache` classification (which stays
      // `'fresh'` on streaming by design; see the existing 3-turn
      // streaming replay test for the full rationale).
      function makeStream(text: string, promptTokens: number, cachedTokens: number, numTokens: number) {
        return async function* () {
          yield { text, done: false, isReasoning: false };
          yield {
            text,
            done: true,
            finishReason: 'stop',
            toolCalls: [] as ToolCallResult[],
            thinking: null,
            numTokens,
            promptTokens,
            reasoningTokens: 0,
            rawText: text,
            cachedTokens,
          };
        };
      }
      const stream = vi
        .fn()
        .mockImplementationOnce(makeStream('A1', 5, 0, 3))
        .mockImplementationOnce(makeStream('A2', 20, 7, 5));
      const mockModel = {
        chatSessionStart: vi.fn().mockRejectedValue(new Error('streaming test')),
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: stream,
        chatStreamSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        resetCaches: vi.fn(),
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('stream-model', mockModel);

      // Turn 1: cold streaming turn — `message_delta.usage` carries
      // only the bare two fields (`input_tokens` + `output_tokens`).
      const r1 = createMockRes();
      await handleCreateMessage(
        r1.res,
        {
          model: 'stream-model',
          system: 'S',
          messages: [{ role: 'user', content: 'A' }],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );
      const t1Events = parseSSE(r1.getBody());
      const t1Delta = t1Events.find((e) => e.event === 'message_delta');
      expect(t1Delta).toBeDefined();
      const t1Usage = t1Delta!.data['usage'] as Record<string, number>;
      expect(t1Usage.input_tokens).toBe(5);
      expect(t1Usage.output_tokens).toBe(3);
      expect(t1Usage).not.toHaveProperty('cache_read_input_tokens');
      expect(t1Usage).not.toHaveProperty('cache_creation_input_tokens');

      // Turn 2: warm hit — `cache_read_input_tokens: 7` lands in
      // the streaming usage block, `input_tokens` is the
      // unsuffixed remainder. `X-Session-Cache` STAYS `'fresh'`
      // on streaming per the existing pre-flush design — that is
      // independent of the body fields.
      const r2 = createMockRes();
      await handleCreateMessage(
        r2.res,
        {
          model: 'stream-model',
          system: 'S',
          messages: [
            { role: 'user', content: 'A' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'B' },
          ],
          max_tokens: 100,
          stream: true,
        },
        registry,
      );
      const t2Events = parseSSE(r2.getBody());
      const t2Delta = t2Events.find((e) => e.event === 'message_delta');
      expect(t2Delta).toBeDefined();
      const t2Usage = t2Delta!.data['usage'] as Record<string, number>;
      expect(t2Usage.cache_read_input_tokens).toBe(7);
      expect(t2Usage.input_tokens).toBe(13);
      expect(t2Usage.output_tokens).toBe(5);
      expect(t2Usage).not.toHaveProperty('cache_creation_input_tokens');
      // Streaming header reports `streaming` — the cache-hit count is
      // carried by the SSE `usage.cache_read_input_tokens` field
      // (asserted above) and the `X-Cached-Tokens` HTTP trailer.
      expect(r2.getHeaders()['x-session-cache']).toBe('streaming');
    });

    it('warm-slot lease where the native verifier rejects (cachedTokens=0) demotes X-Session-Cache to fresh and omits cache fields', async () => {
      // Defense-in-depth gate for the trust-the-native-verifier
      // strategy. `getOrCreateWarmAny` matches solely on byte-equal
      // `instructions` — but the actual prompt prefix that the
      // native side compares (`verify_cache_prefix_direct`) sees the
      // FULL token stream, including image tokens, tool-manifest
      // tokens, and any chat-template revision baked into the
      // tokenizer. When those drift between turns despite a same
      // `system`, the JS-side warm hit fires but the native verifier
      // rejects on token-prefix mismatch and reports
      // `cachedTokens === 0`. The handler MUST then:
      //   * Demote `X-Session-Cache` from the optimistic `prefix_hit`
      //     back to `fresh` (the post-dispatch overwrite at
      //     messages.ts ~line 897).
      //   * NOT emit `X-Cached-Tokens`.
      //   * NOT emit `cache_read_input_tokens` /
      //     `cache_creation_input_tokens` in the response body.
      // The mock controls `cachedTokens` directly — that is exactly
      // what the native verifier returns — so the test is mocking
      // the native contract under image-set / tool-manifest /
      // template-revision drift WITHOUT actually wiring up images
      // or tools. Removing the post-dispatch demote in messages.ts
      // (the `if (lookup.hit && result.cachedTokens === 0) { ... }`
      // block) breaks this test by leaving the header at
      // `'prefix_hit'`.
      const chatSessionStart = vi
        .fn()
        .mockResolvedValueOnce(
          makeChatResult({
            text: 'first',
            numTokens: 4,
            promptTokens: 6,
            cachedTokens: 0,
          }),
        )
        // Second turn: instructions match (same `system: 'S'`) so
        // `getOrCreateWarmAny` returns a hit, BUT the mocked native
        // result reports `cachedTokens === 0` — exactly what
        // `verify_cache_prefix_direct` returns when the prefix
        // mismatched after a hidden change (images, tools, template).
        .mockResolvedValueOnce(
          makeChatResult({
            text: 'second',
            numTokens: 6,
            promptTokens: 11,
            cachedTokens: 0,
          }),
        );
      const resetCaches = vi.fn();
      const mockModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: not expected')),
        chatStreamSessionStart: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        chatStreamSessionContinue: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        chatStreamSessionContinueTool: vi.fn().mockRejectedValue(new Error('non-streaming test')),
        resetCaches,
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('test-model', mockModel);

      // Turn 1: adopts the warm slot under the sentinel.
      const r1 = createMockRes();
      await handleCreateMessage(
        r1.res,
        {
          model: 'test-model',
          system: 'S',
          messages: [{ role: 'user', content: 'A' }],
          max_tokens: 100,
        },
        registry,
      );
      expect(r1.getStatus()).toBe(200);
      const resetCachesAfterT1 = resetCaches.mock.calls.length;
      expect(resetCachesAfterT1).toBeGreaterThanOrEqual(1);

      // Identity witness: spy on `primeHistory` so we can prove the
      // warm-reuse helper actually leased the same session for
      // turn 2 — i.e. `getOrCreateWarmAny` HIT — even though the
      // native side then rejected on token-prefix mismatch. Without
      // this witness a regression that quietly skipped the warm
      // lease (and therefore never had a chance to demote) would
      // pass `expect(...x-session-cache).toBe('fresh')` for the
      // wrong reason.
      const primeHistorySpy = vi.spyOn(ChatSession.prototype, 'primeHistory');
      try {
        const r2 = createMockRes();
        await handleCreateMessage(
          r2.res,
          {
            model: 'test-model',
            system: 'S',
            messages: [
              { role: 'user', content: 'A' },
              { role: 'assistant', content: 'first' },
              { role: 'user', content: 'B' },
            ],
            max_tokens: 100,
          },
          registry,
        );
        expect(r2.getStatus()).toBe(200);

        // Warm-reuse helper fired (NOT `session.reset()`): the
        // model's `resetCaches` did NOT advance — proves the lookup
        // hit. If it had been a miss, the cold-start branch would
        // have called `session.reset()` and bumped `resetCaches`.
        expect(resetCaches.mock.calls.length).toBe(resetCachesAfterT1);

        // Header demoted from optimistic `prefix_hit` to `fresh`.
        // This is the contract under test.
        expect(r2.getHeaders()['x-session-cache']).toBe('fresh');
        // No `X-Cached-Tokens` header — the redundant ops signal
        // also stays off the wire when no tokens were actually
        // reused.
        expect(r2.getHeaders()['x-cached-tokens']).toBeUndefined();

        // Body cache fields stay OFF the wire — no
        // `cache_read_input_tokens`, no `cache_creation_input_tokens`.
        const t2Body = JSON.parse(r2.getBody()) as {
          usage: Record<string, number>;
        };
        expect(t2Body.usage).not.toHaveProperty('cache_read_input_tokens');
        expect(t2Body.usage).not.toHaveProperty('cache_creation_input_tokens');
        expect(t2Body.usage.input_tokens).toBe(11);
        expect(t2Body.usage.output_tokens).toBe(6);

        // Identity witness: turn 2's `primeHistory` ran on the SAME
        // ChatSession that turn 1 adopted under the sentinel. Proves
        // the warm-reuse helper actually leased it (i.e. the demote
        // branch was the one that fired, not a quiet miss).
        expect(primeHistorySpy).toHaveBeenCalled();
        const lastCtx = primeHistorySpy.mock.contexts[primeHistorySpy.mock.contexts.length - 1];
        expect(lastCtx).toBeInstanceOf(ChatSession);
      } finally {
        primeHistorySpy.mockRestore();
      }
    });
  });
});
