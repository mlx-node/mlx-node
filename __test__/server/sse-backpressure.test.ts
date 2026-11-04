import { once } from 'node:events';
import { request as httpRequest, type IncomingMessage, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { Writable } from 'node:stream';

import type { ChatMessage, ChatResult, ChatStreamProgress } from '@mlx-node/core';
import type { ChatStreamProgressRef, SessionCapableModel } from '@mlx-node/lm';
import { createServer, type ServerInstance } from '@mlx-node/server';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';
import * as streaming from '../../packages/server/src/streaming.js';

const TOTAL_CHUNKS = 4_096;
const DELTA = 'x'.repeat(4 * 1_024);
const MAX_CONSUMED_AT_PARK = 1_024;
const MAX_WRITABLE_OVERSHOOT = DELTA.length * 4;

let servers: ServerInstance[] = [];

afterEach(async () => {
  const pending = servers;
  servers = [];
  for (const instance of pending) {
    await instance.close({ timeoutMs: 1_000 });
  }
});

function deferred<T = void>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function withTimeout<T>(promise: Promise<T>, label: string, timeoutMs = 2_000): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(label)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

async function waitUntil(predicate: () => boolean, label: string, timeoutMs = 5_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error(label);
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
}

async function waitForBackpressurePark(
  response: ServerResponse,
  consumed: () => number,
  label: string,
  timeoutMs = 5_000,
): Promise<number> {
  const deadline = Date.now() + timeoutMs;
  let previous = -1;
  let stableTurns = 0;
  while (Date.now() < deadline) {
    await new Promise<void>((resolve) => setImmediate(resolve));
    const current = consumed();
    if (response.writableNeedDrain && current === previous) {
      stableTurns += 1;
      if (stableTurns >= 4) return current;
    } else {
      stableTurns = 0;
    }
    previous = current;
    if (current === TOTAL_CHUNKS) break;
  }
  throw new Error(`${label}: consumed=${consumed()} needDrain=${response.writableNeedDrain}`);
}

function drainHelper(): (res: ServerResponse) => Promise<void> {
  const helper = (
    streaming as typeof streaming & {
      awaitDrainOrClose?: (res: ServerResponse) => Promise<void>;
    }
  ).awaitDrainOrClose;
  expect(helper, 'streaming.ts must export awaitDrainOrClose').toBeTypeOf('function');
  return helper!;
}

function createHeldWritable(): {
  res: ServerResponse;
  release: () => void;
} {
  let release: (() => void) | undefined;
  const writable = new Writable({
    highWaterMark: 1,
    write(_chunk, _encoding, callback) {
      release = callback;
    },
  });
  return {
    res: writable as unknown as ServerResponse,
    release: () => {
      if (release === undefined) throw new Error('write callback was not captured');
      release();
    },
  };
}

describe('SSE writable contract', () => {
  it('returns res.write backpressure and waits for a real Writable drain', async () => {
    const { res, release } = createHeldWritable();

    const ok = streaming.writeSSEEvent(res, 'delta', { text: 'x' });
    expect(ok).toBe(false);

    const drain = drainHelper()(res);
    let settled = false;
    void drain.then(() => {
      settled = true;
    });
    await Promise.resolve();
    expect(settled).toBe(false);

    release();
    await withTimeout(drain, 'awaitDrainOrClose hung after Writable emitted drain');
    expect(settled).toBe(true);
  });

  it('settles on close and removes every temporary listener', async () => {
    const { res } = createHeldWritable();
    expect(streaming.writeSSEEvent(res, 'delta', { text: 'x' })).toBe(false);

    const baseline = {
      drain: res.listenerCount('drain'),
      close: res.listenerCount('close'),
      error: res.listenerCount('error'),
    };
    const drain = drainHelper()(res);
    res.destroy();
    await withTimeout(drain, 'awaitDrainOrClose hung after response close');

    expect(res.listenerCount('drain')).toBe(baseline.drain);
    expect(res.listenerCount('close')).toBe(baseline.close);
    expect(res.listenerCount('error')).toBe(baseline.error);
  });

  it('times out a connected non-draining peer, marks abort, and destroys the transport', async () => {
    const { res } = createHeldWritable();
    expect(streaming.writeSSEEvent(res, 'delta', { text: 'x' })).toBe(false);

    const baseline = {
      drain: res.listenerCount('drain'),
      close: res.listenerCount('close'),
      error: res.listenerCount('error'),
    };
    let timedOut = false;
    await withTimeout(
      streaming.awaitDrainOrClose(res, {
        timeoutMs: 10,
        onTimeout: () => {
          timedOut = true;
        },
      }),
      'awaitDrainOrClose ignored its bounded timeout',
    );

    expect(timedOut).toBe(true);
    expect(res.destroyed).toBe(true);
    expect(res.listenerCount('drain')).toBe(baseline.drain);
    expect(res.listenerCount('close')).toBe(baseline.close);
    expect(res.listenerCount('error')).toBe(baseline.error);
  });

  it('settles through the asynchronous ERR_STREAM_WRITE_AFTER_END error event', async () => {
    const writable = new Writable({
      write(_chunk, _encoding, callback) {
        callback();
      },
    });
    const res = writable as unknown as ServerResponse;
    const errorEvent = once(writable, 'error') as Promise<[NodeJS.ErrnoException]>;

    writable.end();
    const ok = streaming.writeSSEEvent(res, 'too_late', { text: 'x' });
    expect(ok).toBe(false);

    // Node reports write-after-end on a later tick. The drain helper must arm
    // its error listener immediately after the false return rather than rely on
    // a synchronous throw from writeSSEEvent.
    const [error] = await withTimeout(
      Promise.all([errorEvent, drainHelper()(res)]).then(([args]) => args),
      'awaitDrainOrClose missed ERR_STREAM_WRITE_AFTER_END',
    );
    expect(error.code).toBe('ERR_STREAM_WRITE_AFTER_END');
  });
});

type EndpointCase = {
  name: string;
  path: string;
  body: (model: string) => object;
  successorBody: (model: string) => object;
  residualMarker: string;
  successTerminal: string;
  failureTerminal: string;
  abortEvent: 'close' | 'error';
};

const endpointCases: EndpointCase[] = [
  {
    name: 'Responses API',
    path: '/v1/responses',
    body: (model) => ({ model, input: 'hi', stream: true, max_output_tokens: TOTAL_CHUNKS + 1 }),
    successorBody: (model) => ({ model, input: 'successor', max_output_tokens: 8 }),
    residualMarker: 'event: response.output_item.added',
    successTerminal: 'event: response.completed',
    failureTerminal: 'event: response.failed',
    abortEvent: 'close',
  },
  {
    name: 'Messages API',
    path: '/v1/messages',
    body: (model) => ({
      model,
      messages: [{ role: 'user', content: 'hi' }],
      max_tokens: TOTAL_CHUNKS + 1,
      stream: true,
    }),
    successorBody: (model) => ({
      model,
      messages: [{ role: 'user', content: 'successor' }],
      max_tokens: 8,
    }),
    residualMarker: 'event: message_delta',
    successTerminal: 'event: message_stop',
    failureTerminal: 'event: error',
    abortEvent: 'error',
  },
];

function quickResult(): ChatResult {
  return {
    text: 'successor-ok',
    toolCalls: [],
    thinkingEnabled: false,
    numTokens: 1,
    promptTokens: 1,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'successor-ok',
    cachedTokens: 0,
  };
}

function createFloodModel(): {
  model: SessionCapableModel;
  consumed: () => number;
  generatorClosed: Promise<void>;
} {
  let consumed = 0;
  const closed = deferred();

  async function* flood(): AsyncGenerator<Record<string, unknown>> {
    try {
      for (let i = 0; i < TOTAL_CHUNKS; i += 1) {
        consumed += 1;
        yield { done: false, text: DELTA, isReasoning: false };
      }
      yield {
        done: true,
        text: '',
        finishReason: 'stop',
        toolCalls: [],
        thinking: null,
        numTokens: TOTAL_CHUNKS,
        promptTokens: 1,
        reasoningTokens: 0,
        rawText: '',
      };
    } finally {
      closed.resolve();
    }
  }

  const model = {
    chatSessionStart: vi.fn(async () => quickResult()),
    chatSessionContinue: vi.fn(async () => quickResult()),
    chatSessionContinueTool: vi.fn(async () => quickResult()),
    chatStreamSessionStart: vi.fn(flood),
    chatStreamSessionContinue: vi.fn(flood),
    chatStreamSessionContinueTool: vi.fn(flood),
    resetCaches: vi.fn().mockResolvedValue(undefined),
  } as unknown as SessionCapableModel;

  return { model, consumed: () => consumed, generatorClosed: closed.promise };
}

async function postJson(
  instance: ServerInstance,
  path: string,
  body: object,
): Promise<{ status: number; body: string }> {
  const { port } = instance.server.address() as AddressInfo;
  const payload = JSON.stringify(body);
  return withTimeout(
    new Promise<{ status: number; body: string }>((resolve, reject) => {
      const req = httpRequest(
        {
          host: '127.0.0.1',
          port,
          path,
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            'content-length': Buffer.byteLength(payload),
          },
        },
        (res) => {
          let responseBody = '';
          res.setEncoding('utf8');
          res.on('data', (chunk: string) => {
            responseBody += chunk;
          });
          res.once('end', () => resolve({ status: res.statusCode ?? 0, body: responseBody }));
          res.once('error', reject);
        },
      );
      req.once('error', reject);
      req.end(payload);
    }),
    `${path} successor request hung behind the disconnected stream`,
    5_000,
  );
}

async function openPausedStream(
  instance: ServerInstance,
  endpoint: EndpointCase,
  model: string,
): Promise<{ clientResponse: IncomingMessage; serverResponse: ServerResponse }> {
  const { port } = instance.server.address() as AddressInfo;
  const capturedResponse = new Promise<ServerResponse>((resolve) => {
    instance.server.once('request', (_req, res) => resolve(res));
  });
  const payload = JSON.stringify(endpoint.body(model));
  const clientResponse = new Promise<IncomingMessage>((resolve, reject) => {
    const req = httpRequest(
      {
        host: '127.0.0.1',
        port,
        path: endpoint.path,
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'content-length': Buffer.byteLength(payload),
        },
      },
      (res) => {
        res.pause();
        resolve(res);
      },
    );
    req.once('error', reject);
    req.end(payload);
  });

  return {
    clientResponse: await withTimeout(clientResponse, `${endpoint.name} never returned SSE headers`, 5_000),
    serverResponse: await withTimeout(capturedResponse, `${endpoint.name} never reached the HTTP server`, 5_000),
  };
}

describe('paused SSE clients', () => {
  for (const endpoint of endpointCases) {
    it(`${endpoint.name} stops consuming native chunks at HTTP backpressure and close unwinds the wait`, async () => {
      const instance = await createServer({ port: 0, host: '127.0.0.1', disableStore: true });
      servers.push(instance);
      const modelName = `slow-reader-${endpoint.name.toLowerCase().replaceAll(' ', '-')}`;
      const flood = createFloodModel();
      instance.registry.register(modelName, flood.model);

      const { clientResponse, serverResponse } = await openPausedStream(instance, endpoint, modelName);
      const parkedAt = await waitForBackpressurePark(
        serverResponse,
        flood.consumed,
        `${endpoint.name} never parked at persistent HTTP backpressure`,
      );
      expect(serverResponse.writableNeedDrain).toBe(true);
      expect(parkedAt).toBeLessThanOrEqual(MAX_CONSUMED_AT_PARK);
      expect(serverResponse.writableLength).toBeLessThanOrEqual(
        serverResponse.writableHighWaterMark + MAX_WRITABLE_OVERSHOOT,
      );
      await new Promise<void>((resolve) => setImmediate(resolve));
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(flood.consumed()).toBe(parkedAt);

      clientResponse.socket.destroy();
      await withTimeout(flood.generatorClosed, `${endpoint.name} handler hung after the paused socket was destroyed`);
      const successor = await postJson(instance, endpoint.path, endpoint.successorBody(modelName));
      expect(successor.status).toBe(200);
      expect(successor.body).toContain('successor-ok');
      await waitUntil(() => {
        const sessionRegistry = instance.registry.getSessionRegistry(modelName);
        return sessionRegistry?.queueDepth === 0 && sessionRegistry.preDispatchAdmitCount === 0;
      }, `${endpoint.name} admission footprint did not drain after the same-model successor`);
      const sessionRegistry = instance.registry.getSessionRegistry(modelName)!;
      expect(sessionRegistry.queueDepth).toBe(0);
      expect(sessionRegistry.preDispatchAdmitCount).toBe(0);
    });
  }
});

function createResidualModel(endpoint: EndpointCase): SessionCapableModel {
  async function* residual(): AsyncGenerator<Record<string, unknown>> {
    yield {
      done: true,
      text: endpoint.name === 'Messages API' ? 'complete text' : '',
      finishReason: endpoint.name === 'Responses API' ? 'tool_calls' : 'stop',
      toolCalls:
        endpoint.name === 'Responses API'
          ? [
              {
                id: 'call_residual',
                name: 'lookup',
                arguments: '{"q":"residual"}',
                status: 'ok',
                rawContent: '',
              },
            ]
          : [],
      thinking: null,
      numTokens: 1,
      promptTokens: 1,
      reasoningTokens: 0,
      rawText: '',
    };
  }

  return {
    chatSessionStart: vi.fn(async () => quickResult()),
    chatSessionContinue: vi.fn(async () => quickResult()),
    chatSessionContinueTool: vi.fn(async () => quickResult()),
    chatStreamSessionStart: vi.fn(residual),
    chatStreamSessionContinue: vi.fn(residual),
    chatStreamSessionContinueTool: vi.fn(residual),
    resetCaches: vi.fn().mockResolvedValue(undefined),
  } as unknown as SessionCapableModel;
}

describe('post-loop residual backpressure', () => {
  for (const endpoint of endpointCases) {
    it(`${endpoint.name} keeps abort tracking through residual drain and never emits/adopts success`, async () => {
      const instance = await createServer({ port: 0, host: '127.0.0.1', disableStore: true });
      servers.push(instance);
      const modelName = `residual-${endpoint.name.toLowerCase().replaceAll(' ', '-')}`;
      instance.registry.register(modelName, createResidualModel(endpoint));

      let residualTriggered = false;
      instance.server.once('request', (_req, res) => {
        const originalWrite = res.write.bind(res);
        res.write = ((chunk: string | Uint8Array, ...args: unknown[]): boolean => {
          const ok = Reflect.apply(originalWrite, res, [chunk, ...args]) as boolean;
          if (!residualTriggered && chunk.toString().includes(endpoint.residualMarker)) {
            residualTriggered = true;
            queueMicrotask(() => {
              if (endpoint.abortEvent === 'error') {
                res.emit('error', new Error('simulated residual write error'));
              } else {
                res.emit('close');
              }
            });
            return false;
          }
          return ok;
        }) as ServerResponse['write'];
      });

      const response = await postJson(instance, endpoint.path, {
        ...endpoint.body(modelName),
        max_output_tokens: 8,
        max_tokens: 8,
      });
      expect(response.status).toBe(200);
      expect(residualTriggered).toBe(true);
      expect(response.body).toContain(endpoint.failureTerminal);
      expect(response.body).not.toContain(endpoint.successTerminal);
      expect(instance.registry.getSessionRegistry(modelName)?.size).toBe(0);
    });
  }
});

/**
 * Response whose writes succeed but report backpressure for the first frame
 * containing `holdMarker`, until the test emits `drain`.
 */
function createBackpressureRes(holdMarker: string): { res: ServerResponse; body: () => string; held: () => boolean } {
  let body = '';
  let held = false;
  const headers: Record<string, string | string[]> = {};
  const writable = new Writable({
    write(chunk: Uint8Array | string, _encoding, callback) {
      body += chunk.toString();
      callback();
    },
  }) as Writable & Record<string, unknown>;
  writable.on('error', () => {});
  const write = writable.write.bind(writable) as (chunk: string, ...rest: unknown[]) => boolean;
  writable.write = ((chunk: string, ...rest: unknown[]) => {
    write(chunk, ...rest);
    if (!held && chunk.includes(holdMarker)) {
      held = true;
      return false;
    }
    return true;
  }) as Writable['write'];
  writable.writeHead = (_status: number, h?: Record<string, string>) => {
    for (const [k, v] of Object.entries(h ?? {})) headers[k.toLowerCase()] = v;
    writable.headersSent = true;
    return writable;
  };
  writable.setHeader = (name: string, value: string) => {
    headers[name.toLowerCase()] = value;
  };
  writable.getHeader = (name: string) => headers[name.toLowerCase()];
  writable.headersSent = false;
  return { res: writable as unknown as ServerResponse, body: () => body, held: () => held };
}

describe('progress pings under backpressure', () => {
  it('Messages API skips ping ticks while a write is waiting for drain', async () => {
    const original = process.env.MLX_SSE_PROGRESS_INTERVAL_MS;
    process.env.MLX_SSE_PROGRESS_INTERVAL_MS = '1000';
    vi.useFakeTimers();
    try {
      const snapshot: ChatStreamProgress = { phase: 'decode', generatedTokens: 1, decodeElapsedMs: 100 };
      const nextEvent = deferred<Record<string, unknown>>();
      let started = false;
      async function* stream(ref?: ChatStreamProgressRef): AsyncGenerator<Record<string, unknown>> {
        started = true;
        if (ref !== undefined) ref.read = () => snapshot;
        try {
          yield { done: false, text: 'held', isReasoning: false };
          yield await nextEvent.promise;
        } finally {
          if (ref !== undefined) ref.read = null;
        }
      }
      const start = vi.fn(
        (_messages: ChatMessage[], _config: unknown, _signal?: AbortSignal, ref?: ChatStreamProgressRef) => stream(ref),
      );
      const model = {
        chatSessionStart: vi.fn(async () => quickResult()),
        chatSessionContinue: vi.fn(async () => quickResult()),
        chatSessionContinueTool: vi.fn(async () => quickResult()),
        chatStreamSessionStart: start,
        chatStreamSessionContinue: start,
        chatStreamSessionContinueTool: start,
        resetCaches: vi.fn().mockResolvedValue(undefined),
      } as unknown as SessionCapableModel;
      const registry = new ModelRegistry();
      registry.register('ping-backpressure', model);
      const { res, body, held } = createBackpressureRes('event: content_block_delta');
      const pingCount = () => body().split('event: ping\n').length - 1;

      const done = handleCreateMessage(
        res,
        { model: 'ping-backpressure', messages: [{ role: 'user', content: 'hi' }], max_tokens: 8, stream: true },
        registry,
      );
      while (!started || !held()) {
        await vi.advanceTimersByTimeAsync(0);
      }

      await vi.advanceTimersByTimeAsync(5_000);
      expect(pingCount()).toBe(0);

      res.emit('drain');
      await vi.advanceTimersByTimeAsync(1_000);
      expect(pingCount()).toBe(1);

      nextEvent.resolve({
        done: true,
        text: 'held',
        finishReason: 'stop',
        toolCalls: [],
        thinking: null,
        numTokens: 1,
        promptTokens: 1,
        reasoningTokens: 0,
        rawText: 'held',
      });
      await vi.advanceTimersByTimeAsync(0);
      await done;
      expect(body()).toContain('event: message_stop');
    } finally {
      vi.useRealTimers();
      if (original === undefined) delete process.env.MLX_SSE_PROGRESS_INTERVAL_MS;
      else process.env.MLX_SSE_PROGRESS_INTERVAL_MS = original;
    }
  });
});
