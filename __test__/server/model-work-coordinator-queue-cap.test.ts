import type { ServerResponse } from 'node:http';
import { Writable } from 'node:stream';

import type { ChatResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { handleCreateResponse } from '../../packages/server/src/endpoints/responses.js';
import { ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';
import { DEFAULT_MAX_QUEUE_DEPTH_PER_MODEL } from '../../packages/server/src/server.js';

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
const timeout = (ms: number) => new Promise<'timeout'>((resolve) => setTimeout(() => resolve('timeout'), ms));

function makeChatResult(overrides: Partial<ChatResult> = {}): ChatResult {
  return {
    text: 'ok',
    toolCalls: [],
    thinkingEnabled: true,
    numTokens: 1,
    promptTokens: 1,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'ok',
    cachedTokens: 0,
    ...overrides,
  };
}

function createModel(): SessionCapableModel {
  return {
    chatSessionStart: vi.fn(async () => makeChatResult()),
    chatSessionContinue: vi.fn(async () => makeChatResult()),
    chatSessionContinueTool: vi.fn(async () => makeChatResult()),
    chatStreamSessionStart: vi.fn(),
    chatStreamSessionContinue: vi.fn(),
    chatStreamSessionContinueTool: vi.fn(),
    resetCaches: vi.fn(),
  } as unknown as SessionCapableModel;
}

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
  let endResolve!: () => void;
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
    writeHead: (statusCode: number, headers?: Record<string, string>) => Writable;
    setHeader: (name: string, value: string) => void;
  };

  writable.headersSent = false;
  writable.writeHead = (statusCode: number, responseHeaders?: Record<string, string>) => {
    status = statusCode;
    if (responseHeaders) {
      for (const [key, value] of Object.entries(responseHeaders)) {
        headers[key.toLowerCase()] = value;
      }
    }
    writable.headersSent = true;
    return writable;
  };
  writable.setHeader = (name: string, value: string) => {
    headers[name.toLowerCase()] = value;
  };

  const originalEnd = writable.end.bind(writable);
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
    originalEnd(undefined, (err?: Error | null) => {
      if (cb) cb(err ?? null);
      endResolve();
    });
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

async function waitForQueueDepth(registry: ModelRegistry, modelName: string, expected: number): Promise<void> {
  for (let i = 0; i < 10; i += 1) {
    if (registry.getSessionRegistry(modelName)?.queueDepth === expected) return;
    await tick();
  }
  expect(registry.getSessionRegistry(modelName)?.queueDepth).toBe(expected);
}

describe('ModelWorkCoordinator + SessionRegistry queue cap', () => {
  it('POST /v1/messages enforces the per-model queue cap before waiting behind a model-load writer', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    registry.register('cap-model', createModel());
    const coordinator = new ModelWorkCoordinator();
    const writerStarted = deferred();
    const releaseWriter = deferred();
    let requestA: Promise<void> | undefined;
    let requestB: Promise<void> | undefined;
    const writer = coordinator.withModelLoad(async () => {
      writerStarted.resolve(undefined);
      await releaseWriter.promise;
    });

    await writerStarted.promise;

    try {
      const mockA = createMockRes();
      requestA = handleCreateMessage(
        mockA.res,
        {
          model: 'cap-model',
          messages: [{ role: 'user', content: 'A' }],
          max_tokens: 16,
        },
        registry,
        undefined,
        null,
        undefined,
        coordinator,
      );
      await tick();

      const mockB = createMockRes();
      requestB = handleCreateMessage(
        mockB.res,
        {
          model: 'cap-model',
          messages: [{ role: 'user', content: 'B' }],
          max_tokens: 16,
        },
        registry,
        undefined,
        null,
        undefined,
        coordinator,
      );
      await waitForQueueDepth(registry, 'cap-model', 1);

      const mockC = createMockRes();
      const requestC = handleCreateMessage(
        mockC.res,
        {
          model: 'cap-model',
          messages: [{ role: 'user', content: 'C' }],
          max_tokens: 16,
        },
        registry,
        undefined,
        null,
        undefined,
        coordinator,
      );
      const cOutcome = await Promise.race([requestC.then(() => 'done' as const), timeout(50)]);

      expect(cOutcome).toBe('done');
      await mockC.waitForEnd();
      expect(mockC.getStatus()).toBe(429);
      expect(mockC.getHeaders()['retry-after']).toBe('1');
      const parsed = JSON.parse(mockC.getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('rate_limit_error');
      expect(parsed.error.message).toContain('Model queue full');
    } finally {
      releaseWriter.resolve(undefined);
      await writer;
      if (requestA) await requestA;
      if (requestB) await requestB;
    }
  });

  it('POST /v1/responses enforces the per-model queue cap before waiting behind a model-load writer', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    registry.register('cap-model', createModel());
    const coordinator = new ModelWorkCoordinator();
    const writerStarted = deferred();
    const releaseWriter = deferred();
    let requestA: Promise<void> | undefined;
    let requestB: Promise<void> | undefined;
    const writer = coordinator.withModelLoad(async () => {
      writerStarted.resolve(undefined);
      await releaseWriter.promise;
    });

    await writerStarted.promise;

    try {
      const mockA = createMockRes();
      requestA = handleCreateResponse(
        mockA.res,
        { model: 'cap-model', input: 'A' },
        registry,
        null,
        undefined,
        undefined,
        null,
        coordinator,
      );
      await tick();

      const mockB = createMockRes();
      requestB = handleCreateResponse(
        mockB.res,
        { model: 'cap-model', input: 'B' },
        registry,
        null,
        undefined,
        undefined,
        null,
        coordinator,
      );
      await waitForQueueDepth(registry, 'cap-model', 1);

      const mockC = createMockRes();
      const requestC = handleCreateResponse(
        mockC.res,
        { model: 'cap-model', input: 'C' },
        registry,
        null,
        undefined,
        undefined,
        null,
        coordinator,
      );
      const cOutcome = await Promise.race([requestC.then(() => 'done' as const), timeout(50)]);

      expect(cOutcome).toBe('done');
      await mockC.waitForEnd();
      expect(mockC.getStatus()).toBe(429);
      expect(mockC.getHeaders()['retry-after']).toBe('1');
      const parsed = JSON.parse(mockC.getBody());
      expect(parsed.error.type).toBe('rate_limit_error');
      expect(parsed.error.code).toBe('queue_full');
      expect(parsed.error.message).toContain('Model queue full');
    } finally {
      releaseWriter.resolve(undefined);
      await writer;
      if (requestA) await requestA;
      if (requestB) await requestB;
    }
  });
});

/**
 * Gated model for host-mode admission tests: the FIRST `chatSessionStart`
 * parks on `hold` and signals `onEnter`, so the test can pin one
 * inference open — holding the coordinator READER inside `withExclusive`
 * — while concurrent arrivals park in the `resolveModel` writer queue.
 * Later calls return immediately so the backlog drains. Exposes the
 * `chatSessionStart` mock so a test can gate one more turn via
 * `mockImplementationOnce`.
 */
function createGatedModel(
  onEnter: () => void,
  hold: Promise<void>,
): { model: SessionCapableModel; chatSessionStart: ReturnType<typeof vi.fn> } {
  let first = true;
  const chatSessionStart = vi.fn(async () => {
    if (first) {
      first = false;
      onEnter();
      await hold;
    }
    return makeChatResult();
  });
  const model = {
    chatSessionStart,
    chatSessionContinue: vi.fn(async () => makeChatResult()),
    chatSessionContinueTool: vi.fn(async () => makeChatResult()),
    chatStreamSessionStart: vi.fn(),
    chatStreamSessionContinue: vi.fn(),
    chatStreamSessionContinueTool: vi.fn(),
    resetCaches: vi.fn(),
  } as unknown as SessionCapableModel;
  return { model, chatSessionStart };
}

/**
 * H3 host-mode regression: with `resolveModel` wired (as
 * `createInferenceHost` wires it), every request enters the
 * `ModelWorkCoordinator` writer bracket BEFORE `withExclusive`. While a
 * turn holds the coordinator's reader, arrivals park as writer waiters
 * where `SessionRegistry.queueDepth` stays 0 — so without a pre-dispatch
 * gate the per-model cap could never fire and the parked backlog grew
 * without bound. These tests pin the gate: over-cap arrivals get the
 * same 429 envelope, BEFORE the active turn completes.
 */
describe('pre-dispatch admission gate (resolveModel + coordinator wiring)', () => {
  it('POST /v1/messages rejects over-cap arrivals with 429 while the active turn still holds the reader', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: DEFAULT_MAX_QUEUE_DEPTH_PER_MODEL });
    const coordinator = new ModelWorkCoordinator();
    const holderEntered = deferred();
    const releaseHolder = deferred();
    const { model } = createGatedModel(() => holderEntered.resolve(undefined), releaseHolder.promise);
    registry.register('gated-model', model);
    // Resident fast path: the host's resolveModel is a no-op once the
    // model is registered, but it still runs inside the writer bracket.
    const resolveModel = vi.fn(async () => {});

    const sendOne = () => {
      const mock = createMockRes();
      const done = handleCreateMessage(
        mock.res,
        {
          model: 'gated-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 16,
        },
        registry,
        undefined,
        null,
        resolveModel,
        coordinator,
      );
      return { mock, done };
    };

    const holder = sendOne();
    await holderEntered.promise;

    // 18 concurrent arrivals against cap 16: the first 16 are admitted
    // (they park behind the coordinator writer gate), #17 and #18 are
    // rejected by the pre-dispatch gate.
    const arrivals = Array.from({ length: 18 }, () => sendOne());
    await tick();

    for (const overflow of arrivals.slice(16)) {
      const outcome = await Promise.race([overflow.done.then(() => 'done' as const), timeout(200)]);
      // The 429 must land BEFORE the holder releases — this is the whole
      // point of the gate; without it the overflow would park unbounded
      // for the length of the active turn.
      expect(outcome).toBe('done');
      await overflow.mock.waitForEnd();
      expect(overflow.mock.getStatus()).toBe(429);
      expect(overflow.mock.getHeaders()['retry-after']).toBe('1');
      const parsed = JSON.parse(overflow.mock.getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('rate_limit_error');
      expect(parsed.error.message).toContain('Model queue full');
    }
    // The finding's mechanism, pinned: the admitted arrivals are parked
    // at the COORDINATOR, not in the SessionRegistry FIFO — queueDepth
    // never needs to exceed (or even reach) the cap for the gate to fire.
    expect(registry.getSessionRegistry('gated-model')!.queueDepth).toBe(0);

    releaseHolder.resolve(undefined);
    await holder.done;
    await holder.mock.waitForEnd();
    expect(holder.mock.getStatus()).toBe(200);
    for (const admitted of arrivals.slice(0, 16)) {
      await admitted.done;
      await admitted.mock.waitForEnd();
      expect(admitted.mock.getStatus()).toBe(200);
    }
  }, 15000);

  it('POST /v1/responses rejects over-cap arrivals with 429 while the active turn still holds the reader', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: DEFAULT_MAX_QUEUE_DEPTH_PER_MODEL });
    const coordinator = new ModelWorkCoordinator();
    const holderEntered = deferred();
    const releaseHolder = deferred();
    const { model } = createGatedModel(() => holderEntered.resolve(undefined), releaseHolder.promise);
    registry.register('gated-model', model);
    const resolveModel = vi.fn(async () => {});

    const sendOne = () => {
      const mock = createMockRes();
      const done = handleCreateResponse(
        mock.res,
        { model: 'gated-model', input: 'hi' },
        registry,
        null,
        undefined,
        undefined,
        null,
        coordinator,
        resolveModel,
      );
      return { mock, done };
    };

    const holder = sendOne();
    await holderEntered.promise;

    const arrivals = Array.from({ length: 18 }, () => sendOne());
    await tick();

    for (const overflow of arrivals.slice(16)) {
      const outcome = await Promise.race([overflow.done.then(() => 'done' as const), timeout(200)]);
      expect(outcome).toBe('done');
      await overflow.mock.waitForEnd();
      expect(overflow.mock.getStatus()).toBe(429);
      expect(overflow.mock.getHeaders()['retry-after']).toBe('1');
      const parsed = JSON.parse(overflow.mock.getBody());
      expect(parsed.error.type).toBe('rate_limit_error');
      expect(parsed.error.code).toBe('queue_full');
      expect(parsed.error.message).toContain('Model queue full');
    }
    expect(registry.getSessionRegistry('gated-model')!.queueDepth).toBe(0);

    releaseHolder.resolve(undefined);
    await holder.done;
    await holder.mock.waitForEnd();
    expect(holder.mock.getStatus()).toBe(200);
    for (const admitted of arrivals.slice(0, 16)) {
      await admitted.done;
      await admitted.mock.waitForEnd();
      expect(admitted.mock.getStatus()).toBe(200);
    }
  }, 15000);

  it('cold-load burst bypasses the early gate (not resident yet); the cap engages once the model is resident', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 2 });
    const coordinator = new ModelWorkCoordinator();
    const holderEntered = deferred();
    const releaseHolder = deferred();
    const { model, chatSessionStart } = createGatedModel(() => holderEntered.resolve(undefined), releaseHolder.promise);
    // Cold-load hook: registers the model on first resolve, exactly like
    // the host's lazy loader. Until then `getSessionRegistry` is
    // undefined, so the pre-dispatch gate cannot (and must not) fire.
    const resolveModel = vi.fn(async () => {
      registry.register('cold-model', model);
    });

    const sendOne = () => {
      const mock = createMockRes();
      const done = handleCreateMessage(
        mock.res,
        {
          model: 'cold-model',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 16,
        },
        registry,
        undefined,
        null,
        resolveModel,
        coordinator,
      );
      return { mock, done };
    };

    // Phase 1 — cold burst: 3 concurrent requests at a NON-resident
    // model with cap 2. All bypass the early gate; the first parks the
    // turn open (gated model), so this also proves the burst was not
    // early-rejected while the load + first turn were in flight.
    const cold = Array.from({ length: 3 }, () => sendOne());
    await holderEntered.promise;
    releaseHolder.resolve(undefined);
    for (const req of cold) {
      await req.done;
      await req.mock.waitForEnd();
      // Documented residual: non-resident requests are never 429d by the
      // early gate; 3 requests fit the post-load withExclusive budget
      // (runner + 2 waiters), so every one completes.
      expect(req.mock.getStatus()).toBe(200);
    }
    expect(resolveModel).toHaveBeenCalled();

    // Phase 2 — the model is now resident, so the same burst shape hits
    // the pre-dispatch gate: holder + 2 admitted (cap 2), 3rd arrival
    // rejected while the holder still runs.
    const holderEntered2 = deferred();
    const releaseHolder2 = deferred();
    chatSessionStart.mockImplementationOnce(async () => {
      holderEntered2.resolve(undefined);
      await releaseHolder2.promise;
      return makeChatResult();
    });
    const holder2 = sendOne();
    await holderEntered2.promise;
    const arrivals = Array.from({ length: 3 }, () => sendOne());
    await tick();

    const overflow = arrivals[2]!;
    const outcome = await Promise.race([overflow.done.then(() => 'done' as const), timeout(200)]);
    expect(outcome).toBe('done');
    await overflow.mock.waitForEnd();
    expect(overflow.mock.getStatus()).toBe(429);
    expect(overflow.mock.getHeaders()['retry-after']).toBe('1');

    releaseHolder2.resolve(undefined);
    await holder2.done;
    for (const admitted of arrivals.slice(0, 2)) {
      await admitted.done;
      await admitted.mock.waitForEnd();
      expect(admitted.mock.getStatus()).toBe(200);
    }
  }, 15000);
});
