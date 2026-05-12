import type { IncomingMessage, ServerResponse } from 'node:http';
import { Readable, Writable } from 'node:stream';

import type { ChatResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { createHandler, ModelRegistry } from '@mlx-node/server';
import { describe, expect, it, vi } from 'vite-plus/test';

function createMockReq(method: string, url: string, body?: object): IncomingMessage {
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
  return req;
}

function createMockRes(): {
  res: ServerResponse;
  getStatus: () => number;
  getBody: () => string;
  waitForEnd: () => Promise<void>;
} {
  let status = 200;
  let body = '';
  let endResolve: () => void;
  const endPromise = new Promise<void>((resolve) => {
    endResolve = resolve;
  });

  const writable = new Writable({
    write(chunk: Uint8Array | string, _encoding: string, callback: () => void) {
      body += chunk.toString();
      callback();
    },
  }) as Writable & {
    headersSent: boolean;
    writeHead: (status: number) => Writable;
    setHeader: () => Writable;
  };

  writable.writeHead = (s: number) => {
    status = s;
    writable.headersSent = true;
    return writable;
  };
  writable.setHeader = () => writable;
  writable.headersSent = false;

  const origEnd = writable.end.bind(writable);
  writable.end = (chunkArg?: unknown, encodingArg?: unknown, cbArg?: unknown) => {
    let chunk: string | Uint8Array | undefined;
    let cb: ((err?: Error | null) => void) | undefined;
    if (typeof chunkArg === 'function') {
      cb = chunkArg as (err?: Error | null) => void;
    } else {
      chunk = chunkArg as string | Uint8Array | undefined;
      if (typeof encodingArg === 'function') {
        cb = encodingArg as (err?: Error | null) => void;
      } else if (typeof cbArg === 'function') {
        cb = cbArg as (err?: Error | null) => void;
      }
    }
    if (chunk != null) body += chunk.toString();
    writable.headersSent = true;
    origEnd(undefined, (err?: Error | null) => {
      if (cb) cb(err ?? null);
      endResolve();
    });
    return writable;
  };

  return {
    res: writable as unknown as ServerResponse,
    getStatus: () => status,
    getBody: () => body,
    waitForEnd: () => endPromise,
  };
}

function createCountingModel(tokenCount: number): SessionCapableModel & {
  applyChatTemplate: ReturnType<typeof vi.fn>;
  spies: {
    chatSessionStart: ReturnType<typeof vi.fn>;
    chatStreamSessionStart: ReturnType<typeof vi.fn>;
  };
} {
  const result: ChatResult = {
    text: 'unexpected',
    toolCalls: [],
    numTokens: 0,
    promptTokens: 0,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'unexpected',
    cachedTokens: 0,
  };
  async function* stream() {
    yield { done: true, text: '', finishReason: 'stop', toolCalls: [], numTokens: 0, promptTokens: 0, rawText: '' };
  }
  const chatSessionStart = vi.fn().mockResolvedValue(result);
  const chatStreamSessionStart = vi.fn(() => stream());
  return {
    applyChatTemplate: vi.fn(async () => new Uint32Array(tokenCount)),
    chatSessionStart,
    chatSessionContinue: vi.fn().mockResolvedValue(result),
    chatSessionContinueTool: vi.fn().mockResolvedValue(result),
    chatStreamSessionStart,
    chatStreamSessionContinue: vi.fn(() => stream()),
    chatStreamSessionContinueTool: vi.fn(() => stream()),
    resetCaches: vi.fn(),
    spies: { chatSessionStart, chatStreamSessionStart },
  } as unknown as SessionCapableModel & {
    applyChatTemplate: ReturnType<typeof vi.fn>;
    spies: {
      chatSessionStart: ReturnType<typeof vi.fn>;
      chatStreamSessionStart: ReturnType<typeof vi.fn>;
    };
  };
}

function createSessionOnlyModel(): SessionCapableModel {
  const model = createCountingModel(0);
  delete (model as Partial<{ applyChatTemplate: unknown }>).applyChatTemplate;
  return model;
}

describe('POST /v1/messages/count_tokens', () => {
  it('counts chat-template tokens without requiring max_tokens or running inference', async () => {
    const registry = new ModelRegistry();
    const model = createCountingModel(17);
    registry.register('test-model', model);
    const handler = createHandler(registry);
    const req = createMockReq('POST', '/v1/messages/count_tokens', {
      model: 'test-model',
      system: 'You are concise.',
      messages: [{ role: 'user', content: 'Hello' }],
    });
    const { res, getStatus, getBody, waitForEnd } = createMockRes();

    await handler(req, res);
    await waitForEnd();

    expect(getStatus()).toBe(200);
    expect(JSON.parse(getBody())).toEqual({ input_tokens: 17 });
    expect(model.applyChatTemplate).toHaveBeenCalledWith(
      [
        { role: 'system', content: 'You are concise.' },
        { role: 'user', content: 'Hello' },
      ],
      true,
      null,
    );
    expect(model.spies.chatSessionStart).not.toHaveBeenCalled();
    expect(model.spies.chatStreamSessionStart).not.toHaveBeenCalled();
  });

  it('uses the Anthropic mapper for tools before counting', async () => {
    const registry = new ModelRegistry();
    const model = createCountingModel(23);
    registry.register('tool-model', model);
    const handler = createHandler(registry);
    const req = createMockReq('POST', '/v1/messages/count_tokens', {
      model: 'tool-model',
      tools: [
        {
          name: 'lookup',
          description: 'Lookup a value',
          input_schema: {
            type: 'object',
            properties: { id: { type: 'string' } },
            required: ['id'],
          },
        },
      ],
      messages: [{ role: 'user', content: 'Find abc' }],
    });
    const { res, getStatus, getBody, waitForEnd } = createMockRes();

    await handler(req, res);
    await waitForEnd();

    expect(getStatus()).toBe(200);
    expect(JSON.parse(getBody())).toEqual({ input_tokens: 23 });
    const [, , tools] = model.applyChatTemplate.mock.calls[0];
    expect(tools).toEqual([
      {
        type: 'function',
        function: {
          name: 'lookup',
          description: 'Lookup a value',
          parameters: {
            type: 'object',
            properties: JSON.stringify({ id: { type: 'string' } }),
            required: ['id'],
          },
        },
      },
    ]);
  });

  it('returns 501 when the registered model lacks a non-generating chat-template tokenizer API', async () => {
    const registry = new ModelRegistry();
    registry.register('session-only', createSessionOnlyModel());
    const handler = createHandler(registry);
    const req = createMockReq('POST', '/v1/messages/count_tokens', {
      model: 'session-only',
      messages: [{ role: 'user', content: 'Hello' }],
    });
    const { res, getStatus, getBody, waitForEnd } = createMockRes();

    await handler(req, res);
    await waitForEnd();

    expect(getStatus()).toBe(501);
    const parsed = JSON.parse(getBody());
    expect(parsed.type).toBe('error');
    expect(parsed.error.type).toBe('not_supported_error');
    expect(parsed.error.message).toContain('applyChatTemplate');
  });

  it('rejects a token-count result when the model binding changes while counting is running', async () => {
    const registry = new ModelRegistry();
    const originalModel = createCountingModel(17);
    const swappedModel = createCountingModel(29);
    let resolveApply!: (tokens: Uint32Array) => void;
    const applyGate = new Promise<Uint32Array>((resolve) => {
      resolveApply = resolve;
    });
    let markApplyStarted!: () => void;
    const applyStarted = new Promise<void>((resolve) => {
      markApplyStarted = resolve;
    });
    originalModel.applyChatTemplate = vi.fn(async () => {
      markApplyStarted();
      return applyGate;
    });
    registry.register('race-model', originalModel);

    const handler = createHandler(registry);
    const req = createMockReq('POST', '/v1/messages/count_tokens', {
      model: 'race-model',
      messages: [{ role: 'user', content: 'running count' }],
    });
    const { res, getStatus, getBody, waitForEnd } = createMockRes();
    const countDone = (async () => {
      await handler(req, res);
      await waitForEnd();
    })();

    await applyStarted;
    registry.register('race-model', swappedModel);
    resolveApply(new Uint32Array(17));
    await countDone;

    expect(getStatus()).toBe(400);
    const parsed = JSON.parse(getBody());
    expect(parsed.type).toBe('error');
    expect(parsed.error.type).toBe('invalid_request_error');
    expect(parsed.error.message).toContain('race-model');
    expect(parsed.error.message).toMatch(/binding changed/i);
    expect(parsed.error.message).toMatch(/running/i);
    expect(originalModel.applyChatTemplate).toHaveBeenCalledTimes(1);
    expect(swappedModel.applyChatTemplate).not.toHaveBeenCalled();
  });
});
