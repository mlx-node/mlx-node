/**
 * A streaming request whose client disconnects while it waits behind another
 * turn in the per-model queue must never be dispatched to the model.
 */

import { EventEmitter } from 'node:events';
import type { ServerResponse } from 'node:http';

import { describe, expect, it, vi } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { handleCreateResponse } from '../../packages/server/src/endpoints/responses.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';

function createMockRes(): { res: ServerResponse; disconnect: () => void; getBody: () => string } {
  const emitter = new EventEmitter();
  const res = emitter as unknown as ServerResponse & {
    writableEnded: boolean;
    writableFinished: boolean;
    headersSent: boolean;
    destroyed: boolean;
  };
  let body = '';
  let socketDestroyed = false;
  const socket = {
    get destroyed() {
      return socketDestroyed;
    },
    once: (event: string, cb: () => void): void => {
      emitter.once(event, cb);
    },
    on: (event: string, cb: () => void): void => {
      emitter.on(event, cb);
    },
    removeListener: (event: string, cb: () => void): void => {
      emitter.removeListener(event, cb);
    },
  };
  res.writableEnded = false;
  res.writableFinished = false;
  res.headersSent = false;
  res.destroyed = false;
  Object.defineProperty(res, 'socket', { value: socket, configurable: true });

  const takeCallback = (args: unknown[]): ((err?: Error | null) => void) | undefined =>
    args.find((a): a is (err?: Error | null) => void => typeof a === 'function');
  res.writeHead = (() => {
    res.headersSent = true;
    return res;
  }) as unknown as ServerResponse['writeHead'];
  res.setHeader = (() => res) as unknown as ServerResponse['setHeader'];
  res.getHeader = (() => undefined) as unknown as ServerResponse['getHeader'];
  res.removeHeader = (() => {}) as ServerResponse['removeHeader'];
  res.write = ((chunk: unknown, ...rest: unknown[]) => {
    if (typeof chunk === 'string') body += chunk;
    const cb = takeCallback(rest);
    if (cb) queueMicrotask(() => cb(null));
    return true;
  }) as ServerResponse['write'];
  res.end = ((chunk: unknown, ...rest: unknown[]) => {
    if (typeof chunk === 'string') body += chunk;
    res.headersSent = true;
    res.writableEnded = true;
    const cb = takeCallback(typeof chunk === 'function' ? [chunk] : rest);
    if (cb) queueMicrotask(() => cb(null));
    return res;
  }) as ServerResponse['end'];
  res.destroy = (() => {
    socketDestroyed = true;
    res.destroyed = true;
    emitter.emit('close');
    return res;
  }) as ServerResponse['destroy'];

  return {
    res,
    disconnect(): void {
      socketDestroyed = true;
      res.destroyed = true;
      emitter.emit('close');
    },
    getBody: () => body,
  };
}

const terminalChunk = {
  done: true,
  text: 'ok',
  finishReason: 'stop',
  toolCalls: [],
  thinking: null,
  thinkingEnabled: false,
  numTokens: 1,
  promptTokens: 1,
  reasoningTokens: 0,
  rawText: 'ok',
};

/** Non-streaming turns park on `hold` so a second request queues behind them. */
function createHeldModel(hold: Promise<void>) {
  async function* stream() {
    yield terminalChunk;
  }
  return {
    chatSessionStart: vi.fn(async () => {
      await hold;
      return { ...terminalChunk, cachedTokens: 0, performance: undefined };
    }),
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path not expected')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path not expected')),
    chatStreamSessionStart: vi.fn(() => stream()),
    chatStreamSessionContinue: vi.fn(() => stream()),
    chatStreamSessionContinueTool: vi.fn(() => stream()),
    resetCaches: vi.fn().mockResolvedValue(undefined),
  };
}

async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
}

describe('streaming request aborted while queued is never dispatched', () => {
  it('/v1/messages', async () => {
    let release!: () => void;
    const hold = new Promise<void>((r) => {
      release = r;
    });
    const model = createHeldModel(hold);
    const registry = new ModelRegistry();
    registry.register('m', model as never);

    const holder = createMockRes();
    const holderDone = handleCreateMessage(
      holder.res,
      { model: 'm', messages: [{ role: 'user', content: 'A' }], max_tokens: 8 },
      registry,
    );
    await settle();
    expect(model.chatSessionStart).toHaveBeenCalledTimes(1);

    const queued = createMockRes();
    const queuedDone = handleCreateMessage(
      queued.res,
      { model: 'm', messages: [{ role: 'user', content: 'B' }], max_tokens: 8, stream: true },
      registry,
    );
    await settle();
    queued.disconnect();
    const resetsBeforeRelease = model.resetCaches.mock.calls.length;

    release();
    await holderDone;
    await queuedDone;

    expect(model.chatStreamSessionStart).not.toHaveBeenCalled();
    expect(model.resetCaches.mock.calls.length).toBe(resetsBeforeRelease);
    expect(queued.getBody()).toBe('');
  });

  it('/v1/responses', async () => {
    let release!: () => void;
    const hold = new Promise<void>((r) => {
      release = r;
    });
    const model = createHeldModel(hold);
    const registry = new ModelRegistry();
    registry.register('m', model as never);

    const holder = createMockRes();
    const holderDone = handleCreateResponse(holder.res, { model: 'm', input: 'A', store: false }, registry, null);
    await settle();
    expect(model.chatSessionStart).toHaveBeenCalledTimes(1);

    const queued = createMockRes();
    const queuedDone = handleCreateResponse(
      queued.res,
      { model: 'm', input: 'B', stream: true, store: false },
      registry,
      null,
    );
    await settle();
    queued.disconnect();
    const resetsBeforeRelease = model.resetCaches.mock.calls.length;

    release();
    await holderDone;
    await queuedDone;

    expect(model.chatStreamSessionStart).not.toHaveBeenCalled();
    expect(model.resetCaches.mock.calls.length).toBe(resetsBeforeRelease);
    expect(queued.getBody()).toBe('');
  });
});
