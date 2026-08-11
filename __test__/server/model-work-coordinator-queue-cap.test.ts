import type { ServerResponse } from 'node:http';
import { Writable } from 'node:stream';

import type { ChatResult, ResponseStore, StoredResponseRecord } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { handleCreateResponse } from '../../packages/server/src/endpoints/responses.js';
import { ModelLoadQueueFullError, ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';
import { DEFAULT_MAX_QUEUE_DEPTH_PER_MODEL } from '../../packages/server/src/server.js';
import { QueueFullError } from '../../packages/server/src/session-registry.js';

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
    resetCaches: vi.fn().mockResolvedValue(undefined),
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
  it('keys cold-load admission independently by requested model', () => {
    const coordinator = new ModelWorkCoordinator(1);
    const a1 = coordinator.beginRequestLoadAdmission('model-a');
    const a2 = coordinator.beginRequestLoadAdmission('model-a');
    expect(() => coordinator.beginRequestLoadAdmission('model-a')).toThrow(ModelLoadQueueFullError);

    const b1 = coordinator.beginRequestLoadAdmission('model-b');
    expect(coordinator.requestLoadAdmissionCount).toBe(3);

    a1.release();
    a2.release();
    b1.release();
    expect(coordinator.requestLoadAdmissionCount).toBe(0);
  });

  it('transfers every cold permit into the resident registry at registration', async () => {
    const coordinator = new ModelWorkCoordinator(1);
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    registry.setModelLoadAdmissionCoordinator(coordinator);
    const coldA = coordinator.beginRequestLoadAdmission('cold-model');
    const coldB = coordinator.beginRequestLoadAdmission('cold-model');

    registry.register('cold-model', createModel());
    const sessionReg = registry.getSessionRegistry('cold-model')!;
    expect(coordinator.requestLoadAdmissionCount).toBe(0);
    expect(sessionReg.preDispatchAdmitCount).toBe(2);
    expect(() => sessionReg.beginPreDispatchAdmission()).toThrow(QueueFullError);

    const releaseHolder = deferred();
    const holder = sessionReg.withExclusive(async () => releaseHolder.promise, coldA.transferToResident(sessionReg));
    const waiter = sessionReg.withExclusive(async () => {}, coldB.transferToResident(sessionReg));
    coldA.release();
    coldB.release();
    expect(sessionReg.preDispatchAdmitCount).toBe(0);
    expect(sessionReg.queueDepth).toBe(1);

    releaseHolder.resolve(undefined);
    await Promise.all([holder, waiter]);
    expect(sessionReg.queueDepth).toBe(0);
  });

  it('fails excess cold-to-resident transfers closed and keeps that request rejected', () => {
    const coordinator = new ModelWorkCoordinator();
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    registry.setModelLoadAdmissionCoordinator(coordinator);
    const first = coordinator.beginRequestLoadAdmission('cold-model');
    const second = coordinator.beginRequestLoadAdmission('cold-model');
    const rejected = coordinator.beginRequestLoadAdmission('cold-model');

    registry.register('cold-model', createModel());
    const sessionReg = registry.getSessionRegistry('cold-model')!;
    expect(sessionReg.preDispatchAdmitCount).toBe(2);
    expect(coordinator.requestLoadAdmissionCount).toBe(0);
    expect(() => rejected.transferToResident(sessionReg)).toThrow(QueueFullError);

    first.release();
    expect(sessionReg.preDispatchAdmitCount).toBe(1);
    // The rejected request does not opportunistically re-enter after another
    // caller frees capacity; its endpoint returns the stored 429 and the client
    // must retry as a new request.
    expect(() => rejected.transferToResident(sessionReg)).toThrow(QueueFullError);

    second.release();
    rejected.release();
    expect(sessionReg.preDispatchAdmitCount).toBe(0);
  });

  it('moves a live cold admission to a replacement resident registry on hot swap', () => {
    const coordinator = new ModelWorkCoordinator();
    const modelA = createModel();
    const modelB = createModel();
    const registryA = new ModelRegistry({ maxQueueDepth: 1 });
    const registryB = new ModelRegistry({ maxQueueDepth: 1 });
    registryA.register('model-a', modelA);
    registryB.register('model-b', modelB);
    const laneA = registryA.getSessionRegistry('model-a')!;
    const laneB = registryB.getSessionRegistry('model-b')!;
    const admission = coordinator.beginRequestLoadAdmission('logical-model');

    coordinator.bindRequestLoadAdmissions('logical-model', laneA);
    expect(laneA.preDispatchAdmitCount).toBe(1);
    expect(laneB.preDispatchAdmitCount).toBe(0);

    coordinator.bindRequestLoadAdmissions('logical-model', laneB);
    expect(laneA.preDispatchAdmitCount).toBe(0);
    expect(laneB.preDispatchAdmitCount).toBe(1);
    expect(admission.transferToResident(laneB)).toBeDefined();

    admission.release();
    expect(laneB.preDispatchAdmitCount).toBe(0);
  });

  it('maps a resident-transfer failure to the exact Responses 429 envelope', async () => {
    const coordinator = new ModelWorkCoordinator();
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    registry.setModelLoadAdmissionCoordinator(coordinator);
    registry.register('cap-model', createModel());
    const sessionReg = registry.getSessionRegistry('cap-model')!;
    const first = sessionReg.beginPreDispatchAdmission();
    const second = sessionReg.beginPreDispatchAdmission();

    try {
      const mock = createMockRes();
      await handleCreateResponse(
        mock.res,
        { model: 'cap-model', input: 'over transferred capacity' },
        registry,
        null,
        undefined,
        undefined,
        null,
        coordinator,
      );
      await mock.waitForEnd();

      expect(mock.getStatus()).toBe(429);
      expect(mock.getHeaders()['retry-after']).toBe('1');
      const parsed = JSON.parse(mock.getBody());
      expect(parsed.error.type).toBe('rate_limit_error');
      expect(parsed.error.code).toBe('queue_full');
    } finally {
      first.release();
      second.release();
    }
  });

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
 * parks on `hold` and signals `onEnter`, so the test can pin one inference
 * open while concurrent resident arrivals move into the SessionRegistry
 * lane. Later calls return immediately so the backlog drains. Exposes the
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
    resetCaches: vi.fn().mockResolvedValue(undefined),
  } as unknown as SessionCapableModel;
  return { model, chatSessionStart };
}

/**
 * H3 host-mode regression with `resolveModel` wired exactly as
 * `createInferenceHost` wires it. A resident request must bypass the
 * model-load writer: otherwise an active inference reader serializes every
 * arrival before it reaches the continuous-batching admission lane. These
 * tests prove arrivals reach the SessionRegistry queue, stay bounded by its
 * pre-dispatch gate, and return the same 429 envelope before the holder ends.
 */
describe('pre-dispatch admission gate (resolveModel + coordinator wiring)', () => {
  it('POST /v1/messages rejects over-cap arrivals with 429 while the active turn still holds the reader', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: DEFAULT_MAX_QUEUE_DEPTH_PER_MODEL });
    const coordinator = new ModelWorkCoordinator();
    const holderEntered = deferred();
    const releaseHolder = deferred();
    const { model } = createGatedModel(() => holderEntered.resolve(undefined), releaseHolder.promise);
    registry.register('gated-model', model);
    // Resident fast path: endpoints must recognize the existing registry and
    // avoid invoking even this no-op resolver inside the writer bracket.
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

    // 18 concurrent arrivals against cap 16: the first 16 reach the resident
    // queue, while #17 and #18 are rejected by the pre-dispatch gate.
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
    // The admitted arrivals reached the resident lane instead of parking as
    // coordinator writers. The fixed waiter bound is fully occupied.
    expect(resolveModel).not.toHaveBeenCalled();
    expect(coordinator.waitingWriters).toBe(0);
    expect(registry.getSessionRegistry('gated-model')!.queueDepth).toBe(16);

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
    expect(resolveModel).not.toHaveBeenCalled();
    expect(coordinator.waitingWriters).toBe(0);
    expect(registry.getSessionRegistry('gated-model')!.queueDepth).toBe(16);

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

  it('POST /v1/messages bounds a cold-load burst while the load promise remains unresolved', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 2 });
    const coordinator = new ModelWorkCoordinator(2);
    const model = createModel();
    const releaseLoad = deferred();
    const resolveModel = vi.fn(async () => {
      await releaseLoad.promise;
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

    const cold = Array.from({ length: 4 }, () => sendOne());
    await tick();

    const overflow = cold[3]!;
    expect(await Promise.race([overflow.done.then(() => 'done' as const), timeout(200)])).toBe('done');
    await overflow.mock.waitForEnd();
    expect(overflow.mock.getStatus()).toBe(429);
    expect(overflow.mock.getHeaders()['retry-after']).toBe('1');
    expect(JSON.parse(overflow.mock.getBody()).error.type).toBe('rate_limit_error');
    expect(registry.getSessionRegistry('cold-model')).toBeUndefined();
    expect(coordinator.requestLoadAdmissionCount).toBe(3);

    releaseLoad.resolve(undefined);
    for (const req of cold.slice(0, 3)) {
      await req.done;
      await req.mock.waitForEnd();
      expect(req.mock.getStatus()).toBe(200);
    }
    expect(coordinator.requestLoadAdmissionCount).toBe(0);
  }, 15000);

  it('POST /v1/responses bounds a cold-load burst while the load promise remains unresolved', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 2 });
    const coordinator = new ModelWorkCoordinator(2);
    const model = createModel();
    const releaseLoad = deferred();
    const resolveModel = vi.fn(async () => {
      await releaseLoad.promise;
      registry.register('cold-model', model);
    });

    const sendOne = () => {
      const mock = createMockRes();
      const done = handleCreateResponse(
        mock.res,
        { model: 'cold-model', input: 'hi' },
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

    const cold = Array.from({ length: 4 }, () => sendOne());
    await tick();

    const overflow = cold[3]!;
    expect(await Promise.race([overflow.done.then(() => 'done' as const), timeout(200)])).toBe('done');
    await overflow.mock.waitForEnd();
    expect(overflow.mock.getStatus()).toBe(429);
    expect(overflow.mock.getHeaders()['retry-after']).toBe('1');
    const parsed = JSON.parse(overflow.mock.getBody());
    expect(parsed.error.type).toBe('rate_limit_error');
    expect(parsed.error.code).toBe('queue_full');
    expect(registry.getSessionRegistry('cold-model')).toBeUndefined();
    expect(coordinator.requestLoadAdmissionCount).toBe(3);

    releaseLoad.resolve(undefined);
    for (const req of cold.slice(0, 3)) {
      await req.done;
      await req.mock.waitForEnd();
      expect(req.mock.getStatus()).toBe(200);
    }
    expect(coordinator.requestLoadAdmissionCount).toBe(0);
  }, 15000);
});

/**
 * Permit lifetime across pre-lock asynchronous work. A continuation can
 * block at `await store.getChain(...)` (and its retry paths) between the
 * pre-dispatch gate and the `withExclusive` placement. The permit must
 * stay held for that whole interval and convert ATOMICALLY into the
 * `withExclusive` admission — one budget, one token per request. If the
 * permit were released early, the request would be counted by NEITHER
 * `preDispatchAdmits` NOR `queuedCount` while parked at the store, other
 * arrivals would refill the budget, and `withExclusive` would then admit
 * a second full waiter budget on top — unbounded accumulation of
 * slow-store continuations ahead of the cap.
 */
describe('pre-dispatch permit lifetime (pre-lock async work)', () => {
  it('holds the permit across a blocked store.getChain — the budget never double-spends', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    const holderEntered = deferred();
    const releaseHolder = deferred();
    const { model } = createGatedModel(() => holderEntered.resolve(undefined), releaseHolder.promise);
    registry.register('m', model);
    const sessReg = registry.getSessionRegistry('m')!;

    let rejectGetChain!: (err: Error) => void;
    const getChainGate = new Promise<StoredResponseRecord[]>((_, reject) => {
      rejectGetChain = reject;
    });
    const store = {
      getChain: vi.fn(() => getChainGate),
      store: vi.fn(async () => {}),
    } as unknown as ResponseStore;

    const sendOne = (body: { model: string; input: string; previous_response_id?: string }) => {
      const mock = createMockRes();
      const done = handleCreateResponse(mock.res, body, registry, store, undefined, undefined, null, undefined);
      return { mock, done };
    };

    // A: continuation parked inside `store.getChain` — pre-lock, holding
    // its admission permit the whole time.
    const a = sendOne({ model: 'm', input: 'a', previous_response_id: 'resp_1' });
    await tick();
    expect(sessReg.queueDepth).toBe(0);
    expect(sessReg.preDispatchAdmitCount).toBe(1);

    // B: stateless request; its permit converts into the runner slot at
    // `withExclusive` placement, then the turn parks inside the model.
    const b = sendOne({ model: 'm', input: 'b' });
    await holderEntered.promise;
    // Probe shape from the review: queueDepth + outstanding permits must
    // never exceed the cap. A's permit is the whole budget (cap 1).
    expect(sessReg.queueDepth).toBe(0);
    expect(sessReg.preDispatchAdmitCount).toBe(1);
    expect(sessReg.queueDepth + sessReg.preDispatchAdmitCount).toBeLessThanOrEqual(1);

    // C: must be rejected at the gate NOW — under the early-release bug
    // it would be admitted on the refilled budget and park as a second
    // in-flight waiter on a cap-1 model.
    const c = sendOne({ model: 'm', input: 'c' });
    const cOutcome = await Promise.race([c.done.then(() => 'done' as const), timeout(200)]);
    expect(cOutcome).toBe('done');
    await c.mock.waitForEnd();
    expect(c.mock.getStatus()).toBe(429);
    expect(c.mock.getHeaders()['retry-after']).toBe('1');
    expect(sessReg.queueDepth + sessReg.preDispatchAdmitCount).toBeLessThanOrEqual(1);

    // Unblock A's getChain with the native miss shape: A settles as a
    // 404 and its permit is released by the handler's outer finally.
    rejectGetChain(new Error('Response not found: resp_1'));
    await a.done;
    await a.mock.waitForEnd();
    expect(a.mock.getStatus()).toBe(404);
    expect(sessReg.preDispatchAdmitCount).toBe(0);

    releaseHolder.resolve(undefined);
    await b.done;
    await b.mock.waitForEnd();
    expect(b.mock.getStatus()).toBe(200);
  }, 15000);

  it('a permitless cold-load alias cannot double-spend a budget slot held by an outstanding permit', async () => {
    // Mixed admission, endpoint-reachable: cold-load requests skip the
    // gate (no SessionRegistry at gate time) and reach `withExclusive`
    // with NO permit. If the permitless admission checked only
    // `queuedCount`, it would spend the slot an outstanding permit
    // (here: a continuation parked in getChain) already owns —
    // breaching the cap instead of returning 429.
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    const holderEntered = deferred();
    const releaseHolder = deferred();
    const { model } = createGatedModel(() => holderEntered.resolve(undefined), releaseHolder.promise);
    registry.register('m', model);
    const sessReg = registry.getSessionRegistry('m')!;

    let rejectGetChain!: (err: Error) => void;
    const getChainGate = new Promise<StoredResponseRecord[]>((_, reject) => {
      rejectGetChain = reject;
    });
    const store = {
      getChain: vi.fn(() => getChainGate),
      store: vi.fn(async () => {}),
    } as unknown as ResponseStore;
    // The lazy loader aliases the SAME model object under the cold
    // name — the alias resolves to the SAME SessionRegistry. Only the
    // request that actually ASKS for the alias registers it, so the
    // earlier 'm' requests cannot make it resident ahead of C's gate.
    const resolveModel = vi.fn(async (name: string) => {
      if (name === 'm-alias') {
        registry.register('m-alias', model);
      }
    });

    const sendOne = (body: { model: string; input: string; previous_response_id?: string }) => {
      const mock = createMockRes();
      const done = handleCreateResponse(
        mock.res,
        body,
        registry,
        store,
        undefined,
        undefined,
        null,
        undefined,
        resolveModel,
      );
      return { mock, done };
    };

    // B holds the turn (runner slot — outside the waiter budget).
    const b = sendOne({ model: 'm', input: 'b' });
    await holderEntered.promise;

    // A: continuation parked inside getChain, holding the single waiter
    // slot as an outstanding permit.
    const a = sendOne({ model: 'm', input: 'a', previous_response_id: 'resp_1' });
    await tick();
    expect(sessReg.queueDepth).toBe(0);
    expect(sessReg.preDispatchAdmitCount).toBe(1);

    // C: cold name — the gate is skipped (no SessionRegistry for
    // 'm-alias' yet); resolveModel aliases it to the same model, and
    // C reaches withExclusive permitless. It must get the same 429,
    // not a seat on top of A's outstanding permit.
    const c = sendOne({ model: 'm-alias', input: 'c' });
    const cOutcome = await Promise.race([c.done.then(() => 'done' as const), timeout(200)]);
    expect(cOutcome).toBe('done');
    await c.mock.waitForEnd();
    expect(c.mock.getStatus()).toBe(429);
    expect(c.mock.getHeaders()['retry-after']).toBe('1');
    const parsed = JSON.parse(c.mock.getBody());
    expect(parsed.error.type).toBe('rate_limit_error');
    expect(parsed.error.code).toBe('queue_full');
    // Combined footprint never exceeds the cap.
    expect(sessReg.queueDepth + sessReg.preDispatchAdmitCount).toBeLessThanOrEqual(1);

    // Drain: A's chain misses (404, permit released), holder finishes,
    // and the invariant still holds at rest.
    rejectGetChain(new Error('Response not found: resp_1'));
    await a.done;
    await a.mock.waitForEnd();
    expect(a.mock.getStatus()).toBe(404);
    expect(sessReg.preDispatchAdmitCount).toBe(0);
    releaseHolder.resolve(undefined);
    await b.done;
    await b.mock.waitForEnd();
    expect(b.mock.getStatus()).toBe(200);
  }, 15000);

  it('an idle-chain cold-load alias cannot double-spend permits parked in store preflight', async () => {
    // The variant above starts with an ACTIVE holder. This one starts with the
    // exec chain IDLE: the gate has legitimately lent the whole
    // runner-plus-waiter capacity (cap 1 + runner entitlement = 2 permits) to
    // two resident continuations parked in store.getChain(), with
    // withExclusive untouched. A cold-alias request reaches the SAME still-idle
    // registry and must not spend a runner seat one of those permits owns.
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    const model = createModel();
    registry.register('m', model);
    const sessReg = registry.getSessionRegistry('m')!;

    const resolveModel = vi.fn(async (name: string) => {
      if (name === 'm-alias') {
        // Same model object -> same SessionRegistry (alias binding).
        registry.register('m-alias', model);
      }
    });

    let rejectGetChain!: (error: Error) => void;
    const getChainGate = new Promise<StoredResponseRecord[]>((_, reject) => {
      rejectGetChain = reject;
    });
    const store = {
      getChain: vi.fn(() => getChainGate),
      store: vi.fn(async () => {}),
    } as unknown as ResponseStore;

    const sendOne = (body: { model: string; input: string; previous_response_id?: string }) => {
      const mock = createMockRes();
      const done = handleCreateResponse(
        mock.res,
        body,
        registry,
        store,
        undefined,
        undefined,
        null,
        undefined,
        resolveModel,
      );
      return { mock, done };
    };

    // A1 + A2: resident requests; each takes a permit at the gate (an idle
    // chain admits cap + 1 = 2) and parks in store preflight.
    const a1 = sendOne({ model: 'm', input: 'a1', previous_response_id: 'resp_1' });
    const a2 = sendOne({ model: 'm', input: 'a2', previous_response_id: 'resp_2' });
    await tick();
    expect(sessReg.queueDepth).toBe(0);
    expect(sessReg.preDispatchAdmitCount).toBe(2);

    // C: cold alias — the gate is skipped ('m-alias' is not resident
    // yet); resolveModel registers the alias and C reaches
    // `withExclusive` permitless at the still-idle chain. It must get
    // the 429, not the runner seat.
    const c = sendOne({ model: 'm-alias', input: 'c' });
    const cOutcome = await Promise.race([c.done.then(() => 'done' as const), timeout(200)]);
    expect(cOutcome).toBe('done');
    await c.mock.waitForEnd();
    expect(c.mock.getStatus()).toBe(429);
    expect(c.mock.getHeaders()['retry-after']).toBe('1');
    const parsed = JSON.parse(c.mock.getBody());
    expect(parsed.error.code).toBe('queue_full');
    // The reject seated nothing: both permits are still outstanding
    // and nothing is queued.
    expect(sessReg.queueDepth).toBe(0);
    expect(sessReg.preDispatchAdmitCount).toBe(2);

    // Unblock both parked requests with the native miss shape. Neither ever
    // reaches placement, and both permits return to the idle registry.
    rejectGetChain(new Error('Response not found'));
    await Promise.all([a1.done, a2.done]);
    await a1.mock.waitForEnd();
    await a2.mock.waitForEnd();
    expect(a1.mock.getStatus()).toBe(404);
    expect(a2.mock.getStatus()).toBe(404);
    expect(sessReg.queueDepth).toBe(0);
    expect(sessReg.preDispatchAdmitCount).toBe(0);
  }, 15000);

  it('releases the permit when the storage lookup throws', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    registry.register('m', createModel());
    const sessReg = registry.getSessionRegistry('m')!;
    const store = {
      // Non-"not found" message: a real infrastructure error that must
      // bubble to the handler's error path, not the retry path.
      getChain: vi.fn(async () => {
        throw new Error('disk exploded');
      }),
      store: vi.fn(async () => {}),
    } as unknown as ResponseStore;

    const mock = createMockRes();
    await handleCreateResponse(
      mock.res,
      { model: 'm', input: 'a', previous_response_id: 'resp_1' },
      registry,
      store,
      undefined,
      undefined,
      null,
      undefined,
    );
    await mock.waitForEnd();
    expect(mock.getStatus()).toBeGreaterThanOrEqual(500);
    // The failed continuation must not leak its budget slot...
    expect(sessReg.preDispatchAdmitCount).toBe(0);

    // ...so the next request against the same model is admitted cleanly.
    const followUp = createMockRes();
    await handleCreateResponse(
      followUp.res,
      { model: 'm', input: 'b' },
      registry,
      store,
      undefined,
      undefined,
      null,
      undefined,
    );
    await followUp.waitForEnd();
    expect(followUp.getStatus()).toBe(200);
  });

  it('releases the permit on the binding-changed 400 exit', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    registry.register('m', createModel());
    const oldReg = registry.getSessionRegistry('m')!;

    let resolveGetChain!: (chain: StoredResponseRecord[]) => void;
    const getChainGate = new Promise<StoredResponseRecord[]>((resolve) => {
      resolveGetChain = resolve;
    });
    const store = {
      getChain: vi.fn(() => getChainGate),
      store: vi.fn(async () => {}),
    } as unknown as ResponseStore;

    const mock = createMockRes();
    const done = handleCreateResponse(
      mock.res,
      { model: 'm', input: 'a', previous_response_id: 'resp_1' },
      registry,
      store,
      undefined,
      undefined,
      null,
      undefined,
    );
    await tick();
    expect(oldReg.preDispatchAdmitCount).toBe(1);

    // Hot-swap while the continuation is parked at getChain: the name now
    // points at a DIFFERENT model instance, so the post-await binding
    // guard must 400 — and the permit (held against the OLD registry)
    // must be released on that exit.
    registry.register('m', createModel());
    resolveGetChain([{} as StoredResponseRecord]);
    await done;
    await mock.waitForEnd();
    expect(mock.getStatus()).toBe(400);
    expect(mock.getBody()).toContain('binding changed');
    expect(oldReg.preDispatchAdmitCount).toBe(0);
  });
});
