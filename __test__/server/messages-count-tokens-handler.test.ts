import type { IncomingMessage, ServerResponse } from 'node:http';
import { Readable, Writable } from 'node:stream';

import type { ChatResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { createHandler, ModelRegistry, ModelWorkCoordinator } from '@mlx-node/server';
import { describe, expect, it, vi } from 'vite-plus/test';

import { __setImageValidatorForTests } from '../../packages/lm/src/chat-session.js';

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
    thinkingEnabled: true,
    numTokens: 0,
    promptTokens: 0,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'unexpected',
    cachedTokens: 0,
  };
  async function* stream() {
    yield {
      done: true,
      text: '',
      finishReason: 'stop',
      toolCalls: [],
      thinkingEnabled: true,
      numTokens: 0,
      promptTokens: 0,
      rawText: '',
    };
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
    resetCaches: vi.fn().mockResolvedValue(undefined),
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
      null,
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

  // Splash `Frontend.count_tokens`: the count generation prefills, images
  // widened and the request's effort applied. Mutation: count the bare
  // template length (old `applyChatTemplate(messages, true, tools)`) → red.
  it('counts through the context clamp: images widened, effort passed', async () => {
    const validPng = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const registry = new ModelRegistry();
    const model = createCountingModel(4);
    const expandedPromptTokenCount = vi.fn(() => 300);
    Object.assign(model, { expandedPromptTokenCount, supportsImages: () => true });
    registry.register('vision-model', model);
    const handler = createHandler(registry);
    const req = createMockReq('POST', '/v1/messages/count_tokens', {
      model: 'vision-model',
      extra_body: { reasoning_effort: 'low' },
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'describe' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: validPng } },
          ],
        },
      ],
    });
    const { res, getStatus, getBody, waitForEnd } = createMockRes();

    await handler(req, res);
    await waitForEnd();

    expect(getStatus()).toBe(200);
    expect(JSON.parse(getBody())).toEqual({ input_tokens: 300 });
    expect(expandedPromptTokenCount).toHaveBeenCalledOnce();
    const call = model.applyChatTemplate.mock.calls[0] as unknown[];
    expect(call.slice(1)).toEqual([true, null, true, 'low']);
  });

  it('rejects images on text-only models, corrupt/unsupported images, and >64 images with 400', async () => {
    const validPng = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const pngBytes = Buffer.from(validPng, 'base64');
    const halfPng = pngBytes.subarray(0, Math.floor(pngBytes.length / 2)).toString('base64');
    const bmpB64 = Buffer.from([
      0x42, 0x4d, 0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
    ]).toString('base64');
    const tiffB64 = Buffer.from([0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00]).toString('base64');
    const heicB64 = Buffer.from([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70, 0x68, 0x65, 0x69, 0x63]).toString(
      'base64',
    );
    const garbageB64 = Buffer.from('not-an-image-at-all').toString('base64');

    // Text-only model rejects valid image with 400
    const textRegistry = new ModelRegistry();
    const textModel = createCountingModel(4);
    textRegistry.register('text-model', textModel);
    const textHandler = createHandler(textRegistry);
    {
      const req = createMockReq('POST', '/v1/messages/count_tokens', {
        model: 'text-model',
        messages: [
          {
            role: 'user',
            content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: validPng } }],
          },
        ],
      });
      const { res, getStatus, getBody, waitForEnd } = createMockRes();
      await textHandler(req, res);
      await waitForEnd();
      expect(getStatus()).toBe(400);
      expect(JSON.parse(getBody()).error.message).toContain('Could not process image');
      expect(textModel.applyChatTemplate).not.toHaveBeenCalled();
    }

    // Vision model rejects corrupt / BMP / TIFF / HEIC / garbage with 400
    const visionRegistry = new ModelRegistry();
    const visionModel = Object.assign(createCountingModel(4), { supportsImages: () => true });
    visionRegistry.register('vision-model', visionModel);
    const visionHandler = createHandler(visionRegistry);
    for (const badData of [halfPng, bmpB64, tiffB64, heicB64, garbageB64]) {
      const req = createMockReq('POST', '/v1/messages/count_tokens', {
        model: 'vision-model',
        messages: [
          {
            role: 'user',
            content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: badData } }],
          },
        ],
      });
      const { res, getStatus, getBody, waitForEnd } = createMockRes();
      await visionHandler(req, res);
      await waitForEnd();
      expect(getStatus()).toBe(400);
      expect(JSON.parse(getBody()).error).toMatchObject({
        type: 'invalid_request_error',
        message: 'Could not process image',
      });
      expect(visionModel.applyChatTemplate).not.toHaveBeenCalled();
    }

    // 65 images -> 400 naming oldest block path
    {
      const req = createMockReq('POST', '/v1/messages/count_tokens', {
        model: 'vision-model',
        messages: [
          {
            role: 'user',
            content: Array.from({ length: 65 }, () => ({
              type: 'image',
              source: { type: 'base64', media_type: 'image/png', data: validPng },
            })),
          },
        ],
      });
      const { res, getStatus, getBody, waitForEnd } = createMockRes();
      await visionHandler(req, res);
      await waitForEnd();
      expect(getStatus()).toBe(400);
      expect(JSON.parse(getBody()).error.message).toBe(
        'messages.0.content.0.image: requests support at most 64 images',
      );
    }
  });

  it('passes reasoningEffort none when thinking.type is disabled', async () => {
    const registry = new ModelRegistry();
    const model = createCountingModel(11);
    registry.register('test-model', model);
    const handler = createHandler(registry);
    const req = createMockReq('POST', '/v1/messages/count_tokens', {
      model: 'test-model',
      thinking: { type: 'disabled' },
      messages: [{ role: 'user', content: 'Hello' }],
    });
    const { res, getStatus, getBody, waitForEnd } = createMockRes();

    await handler(req, res);
    await waitForEnd();

    expect(getStatus()).toBe(200);
    expect(JSON.parse(getBody())).toEqual({ input_tokens: 11 });
    const call = model.applyChatTemplate.mock.calls[0] as unknown[];
    expect(call.slice(1)).toEqual([true, null, false, 'none']);
  });

  it('routes the handler thinking budget default to /v1/messages', async () => {
    const registry = new ModelRegistry();
    const model = createCountingModel(4);
    registry.register('test-model', model);
    const handler = createHandler(registry, { thinkingTokenBudget: 3000 });
    const req = createMockReq('POST', '/v1/messages', {
      model: 'test-model',
      max_tokens: 64,
      messages: [{ role: 'user', content: 'Hello' }],
    });
    const { res, getStatus, waitForEnd } = createMockRes();

    await handler(req, res);
    await waitForEnd();

    expect(getStatus()).toBe(200);
    expect(model.spies.chatSessionStart.mock.calls[0]?.[1]?.thinkingTokenBudget).toBe(3000);
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

describe('POST /v1/messages/count_tokens beside a running answer', () => {
  function holdAnswer(coordinator: ModelWorkCoordinator): () => Promise<void> {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const held = coordinator.withInference(() => gate);
    return async () => {
      release();
      await held;
    };
  }

  function countRequest(model: string): IncomingMessage {
    return createMockReq('POST', '/v1/messages/count_tokens', {
      model,
      messages: [{ role: 'user', content: '.' }],
    });
  }

  it('answers for a resident model while an answer holds a reader slot', async () => {
    const registry = new ModelRegistry();
    const model = createCountingModel(5);
    registry.register('resident-model', model);
    const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
    const resolveModel = vi.fn(async () => {});
    const handler = createHandler(registry, { resolveModel, modelWorkCoordinator: coordinator });
    const endAnswer = holdAnswer(coordinator);
    const { res, getStatus, getBody, waitForEnd } = createMockRes();
    let ended = false;
    const countDone = (async () => {
      await handler(countRequest('resident-model'), res);
      await waitForEnd();
      ended = true;
    })();
    try {
      await vi.waitFor(() => expect(ended).toBe(true));
      expect(coordinator.readerCount).toBe(1);
      expect(getStatus()).toBe(200);
      expect(JSON.parse(getBody())).toEqual({ input_tokens: 5 });
      expect(resolveModel).not.toHaveBeenCalled();
      expect(coordinator.lastLoad).toBeNull();
    } finally {
      await endAnswer();
      await countDone;
    }
  });

  it('takes the writer bracket for a name with no resident binding', async () => {
    const registry = new ModelRegistry();
    const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
    const resolveModel = vi.fn(async (name: string) => {
      registry.register(name, createCountingModel(9));
    });
    const handler = createHandler(registry, { resolveModel, modelWorkCoordinator: coordinator });
    const endAnswer = holdAnswer(coordinator);
    const { res, getStatus, getBody, waitForEnd } = createMockRes();
    const countDone = (async () => {
      await handler(countRequest('cold-model'), res);
      await waitForEnd();
    })();

    await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(1));
    expect(resolveModel).not.toHaveBeenCalled();
    expect(res.headersSent).toBe(false);

    await endAnswer();
    await countDone;
    expect(resolveModel).toHaveBeenCalledTimes(1);
    expect(resolveModel).toHaveBeenCalledWith('cold-model');
    expect(getStatus()).toBe(200);
    expect(JSON.parse(getBody())).toEqual({ input_tokens: 9 });
  });

  it('still answers 400 when the binding changes while the count runs', async () => {
    const registry = new ModelRegistry();
    const originalModel = createCountingModel(3);
    let resolveApply!: (tokens: Uint32Array) => void;
    const applyGate = new Promise<Uint32Array>((resolve) => {
      resolveApply = resolve;
    });
    originalModel.applyChatTemplate = vi.fn(() => applyGate);
    registry.register('swap-model', originalModel);
    const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
    const handler = createHandler(registry, { resolveModel: vi.fn(async () => {}), modelWorkCoordinator: coordinator });
    const endAnswer = holdAnswer(coordinator);
    const { res, getStatus, getBody, waitForEnd } = createMockRes();
    const countDone = (async () => {
      await handler(countRequest('swap-model'), res);
      await waitForEnd();
    })();

    await vi.waitFor(() => expect(originalModel.applyChatTemplate).toHaveBeenCalledTimes(1));
    registry.register('swap-model', createCountingModel(4));
    resolveApply(new Uint32Array(3));
    await countDone;
    await endAnswer();

    expect(getStatus()).toBe(400);
    expect(JSON.parse(getBody()).error.message).toContain('binding changed');
  });

  it('validates images asynchronously without blocking timers and caches repeat payloads', async () => {
    const validPng = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const registry = new ModelRegistry();
    const model = Object.assign(createCountingModel(7), { supportsImages: () => true });
    registry.register('vision', model);
    const handler = createHandler(registry);

    let entered!: () => void;
    let release!: () => void;
    const enteredPromise = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const gatePromise = new Promise<void>((resolve) => {
      release = resolve;
    });
    const validator = vi.fn(async () => {
      if (validator.mock.calls.length === 1) {
        entered();
        await gatePromise;
      }
    });

    __setImageValidatorForTests(validator);
    try {
      const req1 = createMockReq('POST', '/v1/messages/count_tokens', {
        model: 'vision',
        messages: [
          {
            role: 'user',
            content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: validPng } }],
          },
        ],
      });
      const r1 = createMockRes();
      const firstPromise = handler(req1, r1.res);
      await enteredPromise;

      let timerRan = false;
      await new Promise<void>((resolve) => {
        setTimeout(() => {
          timerRan = true;
          resolve();
        }, 0);
      });
      expect(timerRan).toBe(true);

      release();
      await firstPromise;
      await r1.waitForEnd();
      expect(r1.getStatus()).toBe(200);
      expect(JSON.parse(r1.getBody())).toEqual({ input_tokens: 7 });
      expect(validator).toHaveBeenCalledTimes(1);

      // Second count_tokens call with identical image hits digest cache
      const req2 = createMockReq('POST', '/v1/messages/count_tokens', {
        model: 'vision',
        messages: [
          {
            role: 'user',
            content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: validPng } }],
          },
        ],
      });
      const r2 = createMockRes();
      await handler(req2, r2.res);
      await r2.waitForEnd();
      expect(r2.getStatus()).toBe(200);
      expect(validator).toHaveBeenCalledTimes(1);
    } finally {
      __setImageValidatorForTests(null);
    }
  });

  it('bounds concurrent count_tokens image validations globally and cancels remaining work on disconnect', async () => {
    const registry = new ModelRegistry();
    const model = Object.assign(createCountingModel(5), { supportsImages: () => true });
    registry.register('vision', model);
    const handler = createHandler(registry);

    // 1. Concurrent count_tokens requests share the global concurrency ceiling of 4.
    let inFlight = 0;
    let maxInFlight = 0;
    const boundedValidator = vi.fn(async () => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 5));
      inFlight--;
    });
    __setImageValidatorForTests(boundedValidator);
    try {
      await Promise.all(
        Array.from({ length: 3 }, (_, rIdx) => {
          const req = createMockReq('POST', '/v1/messages/count_tokens', {
            model: 'vision',
            messages: [
              {
                role: 'user',
                content: Array.from({ length: 4 }, (_, iIdx) => ({
                  type: 'image',
                  source: {
                    type: 'base64',
                    media_type: 'image/png',
                    data: Buffer.from(`ct-global-${rIdx}-${iIdx}`).toString('base64'),
                  },
                })),
              },
            ],
          });
          const resBox = createMockRes();
          return handler(req, resBox.res).then(async () => {
            await resBox.waitForEnd();
            expect(resBox.getStatus()).toBe(200);
          });
        }),
      );
      expect(boundedValidator).toHaveBeenCalledTimes(12);
      expect(maxInFlight).toBe(4);
    } finally {
      __setImageValidatorForTests(null);
    }

    // 2. Client disconnect during count_tokens image preflight stops remaining images and skips applyChatTemplate.
    model.applyChatTemplate.mockClear();
    let entered = 0;
    let fourEntered!: () => void;
    const fourEnteredPromise = new Promise<void>((resolve) => {
      fourEntered = resolve;
    });
    let releaseGate!: () => void;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const gatedValidator = vi.fn(async () => {
      entered++;
      if (entered === 4) fourEntered();
      await gate;
    });

    __setImageValidatorForTests(gatedValidator);
    try {
      const req = createMockReq('POST', '/v1/messages/count_tokens', {
        model: 'vision',
        messages: [
          {
            role: 'user',
            content: Array.from({ length: 10 }, (_, i) => ({
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/png',
                data: Buffer.from(`ct-abort-${i}`).toString('base64'),
              },
            })),
          },
        ],
      });
      const r1 = createMockRes();
      const pending = handler(req, r1.res);
      await fourEnteredPromise;
      r1.res.emit('close');
      await pending;

      expect(r1.getBody()).toBe('');
      expect(model.applyChatTemplate).not.toHaveBeenCalled();

      releaseGate();
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(gatedValidator).toHaveBeenCalledTimes(4);

      // Follow-up text count_tokens succeeds cleanly (lease and global slots intact).
      const req2 = createMockReq('POST', '/v1/messages/count_tokens', {
        model: 'vision',
        messages: [{ role: 'user', content: 'hello' }],
      });
      const r2 = createMockRes();
      await handler(req2, r2.res);
      await r2.waitForEnd();
      expect(r2.getStatus()).toBe(200);
      expect(JSON.parse(r2.getBody())).toEqual({ input_tokens: 5 });
    } finally {
      releaseGate();
      __setImageValidatorForTests(null);
    }
  });

  it('aborts acquireRead and releases the dispatch lease immediately on disconnect while parked behind withModelLoad', async () => {
    const registry = new ModelRegistry();
    const model = createCountingModel(9);
    registry.register('vision', model);
    const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
    const releaseSpy = vi.spyOn(registry, 'releaseDispatchLease');
    const handler = createHandler(registry, { modelWorkCoordinator: coordinator });

    let writerEntered!: () => void;
    const writerEnteredPromise = new Promise<void>((resolve) => {
      writerEntered = resolve;
    });
    let releaseWriter!: () => void;
    const writerGate = new Promise<void>((resolve) => {
      releaseWriter = resolve;
    });
    const activeLoad = coordinator.withModelLoad(async () => {
      writerEntered();
      await writerGate;
    });
    await writerEnteredPromise;
    expect(coordinator.writerActive).toBe(true);

    try {
      const req = createMockReq('POST', '/v1/messages/count_tokens', {
        model: 'vision',
        messages: [{ role: 'user', content: 'hello' }],
      });
      const r1 = createMockRes();
      const pending = handler(req, r1.res);

      // Wait until the handler has acquired the dispatch lease and parked in withInference -> acquireRead.
      await vi.waitFor(() => {
        expect(r1.res.listenerCount('close')).toBeGreaterThan(0);
      });
      expect(releaseSpy).not.toHaveBeenCalled();

      // Disconnect while withModelLoad writer is still held.
      r1.res.emit('close');
      await pending;

      // Request must settle and release its dispatch lease BEFORE the writer completes!
      expect(coordinator.writerActive).toBe(true);
      expect(releaseSpy).toHaveBeenCalledTimes(1);
      expect(releaseSpy).toHaveBeenCalledWith(model);
      expect(r1.getBody()).toBe('');
      expect(model.applyChatTemplate).not.toHaveBeenCalled();
    } finally {
      releaseWriter();
      await activeLoad;
    }

    // Even after the writer completes, the aborted request must never wake up and run applyChatTemplate.
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(model.applyChatTemplate).not.toHaveBeenCalled();
  });
});
