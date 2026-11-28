/**
 * Disconnect cancellation while a chat endpoint request is parked in
 * `ModelWorkCoordinator.withModelLoad*` behind another model load (or while
 * its own `resolveModel` is running).
 *
 * Regression for the Portato scenario where a client stops a second model
 * selection queued behind an in-flight load: the queued request must dequeue
 * immediately, never invoke `resolveModel`, release its load-admission slot,
 * unwind `idleSweeper.withSuspendedDrains`, detach abort listeners, and write
 * no HTTP error body.
 */

import { EventEmitter } from 'node:events';
import { createServer, request as httpRequest, type IncomingMessage, type ServerResponse } from 'node:http';
import { connect as netConnect, type AddressInfo } from 'node:net';
import { Readable } from 'node:stream';

import type { ChatResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

vi.mock('@mlx-node/core', () => ({ __internal__: { clearCache: () => {} } }));

import { handleCountMessageTokens } from '../../packages/server/src/endpoints/messages-count-tokens.js';
import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { handleCreateResponse } from '../../packages/server/src/endpoints/responses.js';
import { createHandler } from '../../packages/server/src/handler.js';
import { createHealthReporter } from '../../packages/server/src/health.js';
import type { IdleSweeper } from '../../packages/server/src/idle-sweeper.js';
import { ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';
import { stringifyStrictJson } from '../../packages/server/src/strict-json.js';

const testCleanups: Array<() => Promise<void> | void> = [];

afterEach(async () => {
  while (testCleanups.length > 0) {
    const fn = testCleanups.pop();
    try {
      await fn?.();
    } catch {
      // Best-effort cleanup so earlier assertion failures stay visible.
    }
  }
});

function withDeadline<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Timed out after ${ms}ms waiting for ${label}`)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      },
    );
  });
}

async function listenLoopback(server: ReturnType<typeof createServer>): Promise<number> {
  testCleanups.push(async () => {
    server.closeAllConnections?.();
    if (!server.listening) return;
    await withDeadline(new Promise<void>((resolve) => server.close(() => resolve())), 2_000, 'loopback server close');
  });
  await withDeadline(
    new Promise<void>((resolve, reject) => {
      const onError = (err: Error) => {
        server.off('listening', onListening);
        reject(err);
      };
      const onListening = () => {
        server.off('error', onError);
        resolve();
      };
      server.once('error', onError);
      server.once('listening', onListening);
      server.listen(0, '127.0.0.1');
    }),
    5_000,
    'loopback server listen',
  );
  return (server.address() as AddressInfo).port;
}

async function connectLoopback(port: number): Promise<ReturnType<typeof netConnect>> {
  const socket = netConnect(port, '127.0.0.1');
  testCleanups.push(() => {
    socket.destroy();
  });
  await withDeadline(
    new Promise<void>((resolve, reject) => {
      const onError = (err: Error) => {
        socket.off('connect', onConnect);
        reject(err);
      };
      const onConnect = () => {
        socket.off('error', onError);
        socket.on('error', () => {});
        resolve();
      };
      socket.once('error', onError);
      socket.once('connect', onConnect);
    }),
    5_000,
    'loopback socket connect',
  );
  return socket;
}

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createTrackingSweeper(): IdleSweeper & { readonly suspendDepth: number; readonly suspendCalls: number } {
  let inFlight = 0;
  let suspendDepth = 0;
  let suspendCalls = 0;
  function withSuspendedDrains<T>(fn: () => Promise<T>): Promise<T>;
  function withSuspendedDrains<T>(fn: () => T): T;
  function withSuspendedDrains<T>(fn: () => T | Promise<T>): T | Promise<T> {
    suspendCalls += 1;
    suspendDepth += 1;
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      suspendDepth -= 1;
    };
    try {
      const result = fn();
      if (result && typeof (result as PromiseLike<T>).then === 'function') {
        return (result as Promise<T>).finally(release);
      }
      release();
      return result;
    } catch (err) {
      release();
      throw err;
    }
  }
  return {
    beginRequest(): void {
      inFlight += 1;
    },
    endRequest(): void {
      inFlight -= 1;
    },
    close(): void {},
    withSuspendedDrains,
    suspendDrains(): () => void {
      suspendCalls += 1;
      suspendDepth += 1;
      let released = false;
      return () => {
        if (released) return;
        released = true;
        suspendDepth -= 1;
      };
    },
    get isPending(): boolean {
      return false;
    },
    get inFlight(): number {
      return inFlight;
    },
    get suspendDepth(): number {
      return suspendDepth;
    },
    get suspendCalls(): number {
      return suspendCalls;
    },
  };
}

function createMockRes(req?: IncomingMessage): {
  res: ServerResponse;
  socket: EventEmitter & { destroyed: boolean };
  disconnect: () => void;
  emitCloseOnly: () => void;
  getStatus: () => number;
  getBody: () => string;
} {
  const emitter = new EventEmitter();
  const socketEmitter = new EventEmitter() as EventEmitter & { destroyed: boolean };
  socketEmitter.destroyed = false;

  const res = emitter as unknown as ServerResponse & {
    writableEnded: boolean;
    writableFinished: boolean;
    headersSent: boolean;
    destroyed: boolean;
    req?: IncomingMessage;
  };
  let status = 200;
  let body = '';

  res.writableEnded = false;
  res.writableFinished = false;
  res.headersSent = false;
  res.destroyed = false;
  if (req) res.req = req;
  Object.defineProperty(res, 'socket', { value: socketEmitter, configurable: true });

  const takeCallback = (args: unknown[]): ((err?: Error | null) => void) | undefined =>
    args.find((a): a is (err?: Error | null) => void => typeof a === 'function');
  res.writeHead = ((s: number) => {
    status = s;
    res.headersSent = true;
    return res;
  }) as unknown as ServerResponse['writeHead'];
  res.setHeader = (() => res) as unknown as ServerResponse['setHeader'];
  res.getHeader = (() => undefined) as unknown as ServerResponse['getHeader'];
  res.removeHeader = (() => {}) as unknown as ServerResponse['removeHeader'];
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
    socketEmitter.destroyed = true;
    res.destroyed = true;
    emitter.emit('close');
    return res;
  }) as ServerResponse['destroy'];

  return {
    res,
    socket: socketEmitter,
    disconnect(): void {
      socketEmitter.destroyed = true;
      res.destroyed = true;
      emitter.emit('close');
      socketEmitter.emit('close');
    },
    emitCloseOnly(): void {
      emitter.emit('close');
    },
    getStatus: () => status,
    getBody: () => body,
  };
}

function createMockModel(tokenCount = 7): SessionCapableModel & {
  applyChatTemplate: ReturnType<typeof vi.fn>;
  chatSessionStart: ReturnType<typeof vi.fn>;
  chatStreamSessionStart: ReturnType<typeof vi.fn>;
} {
  const result: ChatResult = {
    text: 'hello',
    toolCalls: [],
    thinkingEnabled: false,
    numTokens: 1,
    promptTokens: 3,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'hello',
    cachedTokens: 0,
  };
  async function* stream() {
    yield {
      done: true,
      text: 'hello',
      finishReason: 'stop',
      toolCalls: [],
      thinking: null,
      thinkingEnabled: false,
      numTokens: 1,
      promptTokens: 3,
      reasoningTokens: 0,
      rawText: 'hello',
    };
  }
  return {
    applyChatTemplate: vi.fn(async () => new Uint32Array(tokenCount)),
    chatSessionStart: vi.fn().mockResolvedValue(result),
    chatSessionContinue: vi.fn().mockResolvedValue(result),
    chatSessionContinueTool: vi.fn().mockResolvedValue(result),
    chatStreamSessionStart: vi.fn(() => stream()),
    chatStreamSessionContinue: vi.fn(() => stream()),
    chatStreamSessionContinueTool: vi.fn(() => stream()),
    resetCaches: vi.fn().mockResolvedValue(undefined),
  } as unknown as SessionCapableModel & {
    applyChatTemplate: ReturnType<typeof vi.fn>;
    chatSessionStart: ReturnType<typeof vi.fn>;
    chatStreamSessionStart: ReturnType<typeof vi.fn>;
  };
}

async function holdActiveLoad(coordinator: ModelWorkCoordinator, label = 'model-a') {
  const entered = deferred();
  const release = deferred();
  const activeLoad = coordinator.withModelLoad(async () => {
    entered.resolve();
    await release.promise;
  }, label);
  testCleanups.push(async () => {
    release.resolve();
    await withDeadline(activeLoad, 2_000, `active load ${label} cleanup`);
  });
  await withDeadline(entered.promise, 2_000, `active load ${label} entry`);
  return {
    release: async () => {
      release.resolve();
      await withDeadline(activeLoad, 2_000, `active load ${label} release`);
    },
  };
}

function postRaw(
  port: number,
  path: string,
  body: object,
): {
  req: ReturnType<typeof httpRequest>;
  response: Promise<{ status: number; body: string }>;
} {
  const payload = JSON.stringify(body);
  let resolveResp!: (v: { status: number; body: string }) => void;
  let rejectResp!: (err: unknown) => void;
  const response = new Promise<{ status: number; body: string }>((res, rej) => {
    resolveResp = res;
    rejectResp = rej;
  });
  const req = httpRequest(
    {
      hostname: '127.0.0.1',
      port,
      path,
      method: 'POST',
      agent: false,
      headers: {
        'content-type': 'application/json',
        'content-length': Buffer.byteLength(payload),
      },
    },
    (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (c: Buffer) => chunks.push(c));
      res.on('end', () => resolveResp({ status: res.statusCode ?? 0, body: Buffer.concat(chunks).toString('utf8') }));
      res.on('error', rejectResp);
    },
  );
  req.on('error', rejectResp);
  req.end(payload);
  return { req, response };
}

describe('queued model load cancellation on client disconnect', () => {
  it('/v1/messages/count_tokens dequeues immediately on disconnect while parked behind an active load', async () => {
    const registry = new ModelRegistry();
    const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
    const sweeper = createTrackingSweeper();
    const resolveModel = vi.fn(async (name: string) => {
      registry.register(name, createMockModel(11));
    });

    const active = await holdActiveLoad(coordinator, 'bf16-model');
    try {
      const mock = createMockRes();
      const pending = handleCountMessageTokens(
        mock.res,
        { model: 'q4-model', messages: [{ role: 'user', content: '.' }] },
        registry,
        sweeper,
        resolveModel,
        coordinator,
      );

      await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(1));
      expect(resolveModel).not.toHaveBeenCalled();

      mock.emitCloseOnly();
      await withDeadline(pending, 2_000, 'queued load abort');

      expect(coordinator.writerActive).toBe(true);
      expect(coordinator.waitingWriters).toBe(0);
      expect(coordinator.lastLoad).toBeNull();
      expect(resolveModel).not.toHaveBeenCalled();
      expect(mock.getBody()).toBe('');
      expect(mock.res.headersSent).toBe(false);
      expect(mock.res.listenerCount('close')).toBe(0);
      expect(mock.socket.listenerCount('close')).toBe(0);
      expect(sweeper.suspendDepth).toBe(0);
    } finally {
      await active.release();
    }

    await new Promise((r) => setTimeout(r, 10));
    expect(resolveModel).not.toHaveBeenCalled();
    expect(coordinator.lastLoad).toMatchObject({ label: 'bf16-model', ok: true });
  });

  it('/v1/messages dequeues immediately on disconnect while parked behind an active load and releases admission', async () => {
    const registry = new ModelRegistry();
    const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
    const sweeper = createTrackingSweeper();
    const resolveModel = vi.fn(async (name: string) => {
      registry.register(name, createMockModel());
    });

    const active = await holdActiveLoad(coordinator, 'bf16-model');
    try {
      const mock = createMockRes();
      const pending = handleCreateMessage(
        mock.res,
        { model: 'q4-model', messages: [{ role: 'user', content: 'hi' }], max_tokens: 16 },
        registry,
        undefined,
        sweeper,
        resolveModel,
        coordinator,
      );

      await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(1));
      expect(coordinator.requestLoadAdmissionCount).toBe(1);
      expect(resolveModel).not.toHaveBeenCalled();

      mock.emitCloseOnly();
      await withDeadline(pending, 2_000, 'queued load abort');

      expect(coordinator.writerActive).toBe(true);
      expect(coordinator.waitingWriters).toBe(0);
      expect(coordinator.requestLoadAdmissionCount).toBe(0);
      expect(coordinator.lastLoad).toBeNull();
      expect(resolveModel).not.toHaveBeenCalled();
      expect(mock.getBody()).toBe('');
      expect(mock.res.headersSent).toBe(false);
      expect(mock.res.listenerCount('close')).toBe(0);
      expect(mock.socket.listenerCount('close')).toBe(0);
      expect(sweeper.inFlight).toBe(0);
      expect(sweeper.suspendDepth).toBe(0);
    } finally {
      await active.release();
    }

    await new Promise((r) => setTimeout(r, 10));
    expect(resolveModel).not.toHaveBeenCalled();
  });

  it('/v1/responses dequeues immediately on disconnect while parked behind an active load and unwinds drain suspension', async () => {
    const registry = new ModelRegistry();
    const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
    const sweeper = createTrackingSweeper();
    const resolveModel = vi.fn(async (name: string) => {
      registry.register(name, createMockModel());
    });

    const active = await holdActiveLoad(coordinator, 'bf16-model');
    try {
      const mock = createMockRes();
      const pending = handleCreateResponse(
        mock.res,
        { model: 'q4-model', input: 'hi', store: false },
        registry,
        null,
        undefined,
        undefined,
        sweeper,
        coordinator,
        resolveModel,
      );

      await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(1));
      expect(coordinator.requestLoadAdmissionCount).toBe(1);
      expect(sweeper.suspendDepth).toBe(1);
      expect(resolveModel).not.toHaveBeenCalled();

      mock.emitCloseOnly();
      await withDeadline(pending, 2_000, 'queued load abort');

      expect(coordinator.writerActive).toBe(true);
      expect(coordinator.waitingWriters).toBe(0);
      expect(coordinator.requestLoadAdmissionCount).toBe(0);
      expect(coordinator.lastLoad).toBeNull();
      expect(resolveModel).not.toHaveBeenCalled();
      expect(mock.getBody()).toBe('');
      expect(mock.res.headersSent).toBe(false);
      expect(mock.res.listenerCount('close')).toBe(0);
      expect(mock.socket.listenerCount('close')).toBe(0);
      expect(sweeper.inFlight).toBe(0);
      expect(sweeper.suspendDepth).toBe(0);
    } finally {
      await active.release();
    }

    await new Promise((r) => setTimeout(r, 10));
    expect(resolveModel).not.toHaveBeenCalled();
  });

  it('skips inference and response writes when the client disconnects during its own active resolveModel', async () => {
    for (const kind of ['count_tokens', 'messages', 'responses'] as const) {
      const registry = new ModelRegistry();
      const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
      const sweeper = createTrackingSweeper();
      const model = createMockModel(13);
      const loadEntered = deferred();
      const loadGate = deferred();
      const resolveModel = vi.fn(async (name: string) => {
        loadEntered.resolve();
        await loadGate.promise;
        registry.register(name, model);
      });
      const leaseSpy = vi.spyOn(registry, 'acquireDispatchLease');
      const mock = createMockRes();

      const pending =
        kind === 'count_tokens'
          ? handleCountMessageTokens(
              mock.res,
              { model: 'm', messages: [{ role: 'user', content: '.' }] },
              registry,
              sweeper,
              resolveModel,
              coordinator,
            )
          : kind === 'messages'
            ? handleCreateMessage(
                mock.res,
                { model: 'm', messages: [{ role: 'user', content: 'hi' }], max_tokens: 16 },
                registry,
                undefined,
                sweeper,
                resolveModel,
                coordinator,
              )
            : handleCreateResponse(
                mock.res,
                { model: 'm', input: 'hi', store: false },
                registry,
                null,
                undefined,
                undefined,
                sweeper,
                coordinator,
                resolveModel,
              );

      try {
        await withDeadline(loadEntered.promise, 2_000, `${kind} active resolveModel entry`);
        mock.disconnect();
      } finally {
        loadGate.resolve();
      }
      await withDeadline(pending, 2_000, `${kind} active resolveModel disconnect`);

      expect(resolveModel).toHaveBeenCalledTimes(1);
      expect(leaseSpy).not.toHaveBeenCalled();
      expect(model.applyChatTemplate).not.toHaveBeenCalled();
      expect(model.chatSessionStart).not.toHaveBeenCalled();
      expect(coordinator.requestLoadAdmissionCount).toBe(0);
      expect(registry.getSessionRegistry('m')?.preDispatchAdmitCount ?? 0).toBe(0);
      expect(mock.getBody()).toBe('');
      expect(mock.res.headersSent).toBe(false);
      expect(mock.res.listenerCount('close')).toBe(0);
      expect(sweeper.inFlight).toBe(0);
      expect(sweeper.suspendDepth).toBe(0);
    }
  });

  it('does not write a 500 when resolveModel rejects after client disconnect', async () => {
    for (const kind of ['count_tokens', 'messages', 'responses'] as const) {
      const registry = new ModelRegistry();
      const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
      const sweeper = createTrackingSweeper();
      const loadEntered = deferred();
      const loadGate = deferred();
      const resolveModel = vi.fn(async () => {
        loadEntered.resolve();
        await loadGate.promise;
        throw new Error('corrupt weights');
      });
      const mock = createMockRes();

      const pending =
        kind === 'count_tokens'
          ? handleCountMessageTokens(
              mock.res,
              { model: 'broken', messages: [{ role: 'user', content: '.' }] },
              registry,
              sweeper,
              resolveModel,
              coordinator,
            )
          : kind === 'messages'
            ? handleCreateMessage(
                mock.res,
                { model: 'broken', messages: [{ role: 'user', content: 'hi' }], max_tokens: 16 },
                registry,
                undefined,
                sweeper,
                resolveModel,
                coordinator,
              )
            : handleCreateResponse(
                mock.res,
                { model: 'broken', input: 'hi', store: false },
                registry,
                null,
                undefined,
                undefined,
                sweeper,
                coordinator,
                resolveModel,
              );

      try {
        await withDeadline(loadEntered.promise, 2_000, `${kind} rejecting resolveModel entry`);
        mock.emitCloseOnly();
      } finally {
        loadGate.resolve();
      }
      await withDeadline(pending, 2_000, `${kind} rejecting resolveModel disconnect`);

      expect(mock.getBody()).toBe('');
      expect(mock.res.headersSent).toBe(false);
      expect(coordinator.requestLoadAdmissionCount).toBe(0);
      expect(mock.res.listenerCount('close')).toBe(0);
      expect(sweeper.suspendDepth).toBe(0);
    }
  });

  it('does not spuriously abort a queued load when req emits close with complete=true', async () => {
    const registry = new ModelRegistry();
    const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
    const resolveModel = vi.fn(async (name: string) => {
      registry.register(name, createMockModel(9));
    });

    const active = await holdActiveLoad(coordinator, 'model-a');
    const req = new Readable({ read() {} }) as IncomingMessage;
    req.complete = true;
    const mock = createMockRes(req);

    const pending = handleCreateMessage(
      mock.res,
      { model: 'model-b', messages: [{ role: 'user', content: 'hi' }], max_tokens: 16 },
      registry,
      req,
      undefined,
      resolveModel,
      coordinator,
    );

    await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(1));
    // Normal HTTP/1.1 post-body teardown emits 'close' on req with complete=true.
    req.emit('close');
    await new Promise((r) => setTimeout(r, 10));
    expect(coordinator.waitingWriters).toBe(1);

    await active.release();
    await withDeadline(pending, 2_000, 'complete=true queued load settlement');

    expect(resolveModel).toHaveBeenCalledWith('model-b');
    expect(mock.getStatus()).toBe(200);
    expect(req.listenerCount('close')).toBe(0);
    expect(req.listenerCount('aborted')).toBe(0);
  });

  it('real HTTP loopback: destroying a queued select request cancels it without blocking live follow-up requests', async () => {
    const registry = new ModelRegistry();
    const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
    const modelAEntered = deferred();
    const modelAGate = deferred();
    const loadedNames: string[] = [];

    const resolveModel = vi.fn(async (name: string) => {
      loadedNames.push(name);
      if (name === 'model-a') {
        modelAEntered.resolve();
        await modelAGate.promise;
      }
      registry.register(name, createMockModel(15));
    });

    const handler = createHandler(registry, { resolveModel, modelWorkCoordinator: coordinator });
    const server = createServer((req, res) => {
      void handler(req, res);
    });
    testCleanups.push(() => {
      modelAGate.resolve();
    });
    const port = await listenLoopback(server);

    // 1. First request starts loading model-a.
    const reqA = postRaw(port, '/v1/messages/count_tokens', {
      model: 'model-a',
      messages: [{ role: 'user', content: '.' }],
    });
    testCleanups.push(() => {
      reqA.req.destroy();
    });
    reqA.response.catch(() => {});
    await withDeadline(modelAEntered.promise, 5_000, 'model-a load entry');
    expect(coordinator.writerActive).toBe(true);

    // 2. Second request queues model-b behind model-a, then disconnects via req.destroy()
    //    (matching Portato's select-model.ts Stop path).
    const reqB = postRaw(port, '/v1/messages/count_tokens', {
      model: 'model-b',
      messages: [{ role: 'user', content: '.' }],
    });
    testCleanups.push(() => {
      reqB.req.destroy();
    });
    reqB.response.catch(() => {});
    await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(1));

    reqB.req.destroy();
    await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(0));

    // 3. Third & fourth requests stay connected for /v1/messages + /v1/responses on model-c/d
    //    (verifies real HTTP/1.1 post-body teardown does NOT spuriously abort live waiters).
    const reqC = postRaw(port, '/v1/messages', {
      model: 'model-c',
      messages: [{ role: 'user', content: 'hi' }],
      max_tokens: 16,
    });
    testCleanups.push(() => {
      reqC.req.destroy();
    });
    reqC.response.catch(() => {});
    const reqD = postRaw(port, '/v1/responses', {
      model: 'model-d',
      input: 'hi',
      store: false,
    });
    testCleanups.push(() => {
      reqD.req.destroy();
    });
    reqD.response.catch(() => {});
    await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(2));

    // Release model-a: model-b must NEVER load, while model-a/c/d all complete 200.
    modelAGate.resolve();
    const [respA, respC, respD] = await withDeadline(
      Promise.all([reqA.response, reqC.response, reqD.response]),
      5_000,
      'loopback responses',
    );

    expect(respA.status).toBe(200);
    expect(respC.status).toBe(200);
    expect(respD.status).toBe(200);
    expect(loadedNames[0]).toBe('model-a');
    expect([...loadedNames].sort()).toEqual(['model-a', 'model-c', 'model-d']);
    expect(coordinator.waitingWriters).toBe(0);
    expect(coordinator.requestLoadAdmissionCount).toBe(0);
  });

  it('real HTTP/1.1 pipelined disconnect cancels queued cold loads even when res.socket is null', async () => {
    const cases: Array<{ path: string; bodyFor: (model: string) => object }> = [
      {
        path: '/v1/messages/count_tokens',
        bodyFor: (model) => ({ model, messages: [{ role: 'user', content: '.' }] }),
      },
      {
        path: '/v1/messages',
        bodyFor: (model) => ({ model, messages: [{ role: 'user', content: 'hi' }], max_tokens: 16 }),
      },
      {
        path: '/v1/responses',
        bodyFor: (model) => ({ model, input: 'hi', store: false }),
      },
    ];

    const encodePost = (path: string, body: object): string => {
      const payload = JSON.stringify(body);
      return (
        `POST ${path} HTTP/1.1\r\n` +
        `Host: 127.0.0.1\r\n` +
        `Content-Type: application/json\r\n` +
        `Content-Length: ${Buffer.byteLength(payload)}\r\n` +
        `Connection: keep-alive\r\n\r\n` +
        payload
      );
    };

    for (const testCase of cases) {
      const registry = new ModelRegistry();
      const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
      const modelAEntered = deferred();
      const modelAGate = deferred();
      const loadedNames: string[] = [];

      const resolveModel = vi.fn(async (name: string) => {
        loadedNames.push(name);
        if (name === 'model-a') {
          modelAEntered.resolve();
          await modelAGate.promise;
        }
        registry.register(name, createMockModel(15));
      });

      const handler = createHandler(registry, { resolveModel, modelWorkCoordinator: coordinator });
      const server = createServer((req, res) => {
        void handler(req, res);
      });
      testCleanups.push(() => {
        modelAGate.resolve();
      });
      const port = await listenLoopback(server);
      const rawSocket = await connectLoopback(port);

      // Pipeline two POSTs on one TCP socket: request 1 enters model-a load while
      // request 2's ServerResponse sits detached (res.socket === null) behind it.
      rawSocket.write(
        encodePost(testCase.path, testCase.bodyFor('model-a')) + encodePost(testCase.path, testCase.bodyFor('model-b')),
      );

      await withDeadline(modelAEntered.promise, 5_000, `${testCase.path} model-a load entry`);
      await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(1));

      // Drop the pipelined TCP socket; queued model-b must cancel before model-a finishes.
      rawSocket.destroy();
      await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(0));

      modelAGate.resolve();
      await vi.waitFor(() => expect(coordinator.writerActive).toBe(false));
      await new Promise((r) => setTimeout(r, 10));

      expect(loadedNames).toEqual(['model-a']);
      expect(coordinator.requestLoadAdmissionCount).toBe(0);
    }
  });

  it('skips resolveModel and leaves lastLoad untouched if socket.destroyed flips synchronously before close fires at handoff', async () => {
    for (const kind of ['count_tokens', 'messages', 'responses'] as const) {
      const registry = new ModelRegistry();
      const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
      const sweeper = createTrackingSweeper();
      const resolveModel = vi.fn(async (name: string) => {
        registry.register(name, createMockModel());
      });

      const active = await holdActiveLoad(coordinator, 'prior-model');
      const mock = createMockRes();
      const pending =
        kind === 'count_tokens'
          ? handleCountMessageTokens(
              mock.res,
              { model: 'q4-model', messages: [{ role: 'user', content: '.' }] },
              registry,
              sweeper,
              resolveModel,
              coordinator,
            )
          : kind === 'messages'
            ? handleCreateMessage(
                mock.res,
                { model: 'q4-model', messages: [{ role: 'user', content: 'hi' }], max_tokens: 16 },
                registry,
                undefined,
                sweeper,
                resolveModel,
                coordinator,
              )
            : handleCreateResponse(
                mock.res,
                { model: 'q4-model', input: 'hi', store: false },
                registry,
                null,
                undefined,
                undefined,
                sweeper,
                coordinator,
                resolveModel,
              );

      await vi.waitFor(() => expect(coordinator.waitingWriters).toBe(1));

      // Simulate socket.destroy() flipping socket.destroyed synchronously while
      // its 'close' event is still pending on nextTick, right as active releases.
      mock.socket.destroyed = true;
      await active.release();
      await withDeadline(pending, 2_000, `${kind} socket-destroyed handoff`);

      expect(resolveModel).not.toHaveBeenCalled();
      expect(coordinator.lastLoad).toMatchObject({ label: 'prior-model', ok: true, error: null });
      expect(coordinator.waitingWriters).toBe(0);
      expect(coordinator.requestLoadAdmissionCount).toBe(0);
      expect(mock.getBody()).toBe('');
      expect(mock.res.headersSent).toBe(false);
      expect(sweeper.suspendDepth).toBe(0);
    }
  });

  it('does not store raw malformed model objects into lastLoad or break /health JSON serialization', async () => {
    // Deeply nested object that would exceed strict JSON depth limits if copied into lastLoad.label.
    let nested: Record<string, unknown> = { leaf: true };
    for (let i = 0; i < 120; i++) nested = { next: nested };

    for (const kind of ['count_tokens', 'messages'] as const) {
      const registry = new ModelRegistry();
      const coordinator = new ModelWorkCoordinator(registry.queueDepthLimit);
      const resolveModel = vi.fn(async () => {});
      const mock = createMockRes();

      if (kind === 'count_tokens') {
        await withDeadline(
          handleCountMessageTokens(
            mock.res,
            { model: nested as unknown as string, messages: [{ role: 'user', content: '.' }] },
            registry,
            undefined,
            resolveModel,
            coordinator,
          ),
          2_000,
          'malformed model count_tokens',
        );
      } else {
        await withDeadline(
          handleCreateMessage(
            mock.res,
            { model: nested as unknown as string, messages: [{ role: 'user', content: 'hi' }], max_tokens: 16 },
            registry,
            undefined,
            undefined,
            resolveModel,
            coordinator,
          ),
          2_000,
          'malformed model messages',
        );
      }

      expect(coordinator.lastLoad?.label).toBeNull();
      const health = createHealthReporter({
        registry,
        startedAt: Date.now(),
        modelWorkCoordinator: coordinator,
      })();
      expect(() => stringifyStrictJson(health)).not.toThrow();
    }
  });
});
