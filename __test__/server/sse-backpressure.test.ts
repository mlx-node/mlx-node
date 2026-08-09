import { once } from 'node:events';
import { request as httpRequest, type IncomingMessage, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { Writable } from 'node:stream';

import type { SessionCapableModel } from '@mlx-node/lm';
import { createServer, type ServerInstance } from '@mlx-node/server';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import * as streaming from '../../packages/server/src/streaming.js';

const TOTAL_CHUNKS = 4_096;
const DELTA = 'x'.repeat(4 * 1_024);

let servers: ServerInstance[] = [];

afterEach(async () => {
  const pending = servers;
  servers = [];
  for (const instance of pending) {
    await instance.close({ timeoutMs: 250 }).catch(() => {});
  }
});

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
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
};

const endpointCases: EndpointCase[] = [
  {
    name: 'Responses API',
    path: '/v1/responses',
    body: (model) => ({ model, input: 'hi', stream: true, max_output_tokens: TOTAL_CHUNKS + 1 }),
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
  },
];

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
    chatSessionStart: vi.fn().mockRejectedValue(new Error('should use streaming dispatch')),
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('should use streaming dispatch')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('should use streaming dispatch')),
    chatStreamSessionStart: vi.fn(flood),
    chatStreamSessionContinue: vi.fn(flood),
    chatStreamSessionContinueTool: vi.fn(flood),
    resetCaches: vi.fn().mockResolvedValue(undefined),
  } as unknown as SessionCapableModel;

  return { model, consumed: () => consumed, generatorClosed: closed.promise };
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
      expect(parkedAt).toBeLessThan(TOTAL_CHUNKS);
      await new Promise<void>((resolve) => setImmediate(resolve));
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(flood.consumed()).toBe(parkedAt);

      clientResponse.socket.destroy();
      await withTimeout(flood.generatorClosed, `${endpoint.name} handler hung after the paused socket was destroyed`);
      await waitUntil(
        () => instance.registry.getSessionRegistry(modelName)?.queueDepth === 0,
        `${endpoint.name} queue did not drain after client disconnect`,
      );
    });
  }
});
