/**
 * Abort-path coverage for the idle-sweeper bracket inside
 * `handleCreateResponse` and `handleCreateMessage`.
 *
 * Round-4 regression: the older code attached `finish` / `close` /
 * `error` listeners LAZILY from the outer `finally` block. A fast
 * terminal socket event (e.g. a streaming handler that awaited
 * `res.end()` after the socket had already been destroyed) could
 * fire BEFORE the outer `finally` ran, so the `writableEnded`
 * `writableFinished` probe returned `false`, the listeners were
 * attached on a socket whose terminal event had already been
 * emitted, and the matching `idleSweeper.endRequest()` was never
 * called. `inFlight` latched above zero forever.
 *
 * The fix attaches listeners EAGERLY at `beginRequest()` time and
 * calls an idempotent `finalize()` unconditionally from the outer
 * `finally`. This test verifies the invariant: after any terminal
 * socket event — including a mid-stream `close` simulating a client
 * disconnect — `inFlight` returns to zero exactly once.
 */

import { EventEmitter } from 'node:events';
import type { ServerResponse } from 'node:http';

import { describe, expect, it, vi } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { handleCreateResponse } from '../../packages/server/src/endpoints/responses.js';
import type { IdleSweeper } from '../../packages/server/src/idle-sweeper.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** A counting sweeper mock that exposes `inFlight` so tests can assert on it. */
function createCountingSweeper(): IdleSweeper {
  let inFlight = 0;
  // Unused in the abort-path tests but required by the interface
  // (round-8/9 added hot-load bracketing); pass-through / no-op
  // keeps the mock minimal.
  function withSuspendedDrains<T>(fn: () => Promise<T>): Promise<T>;
  function withSuspendedDrains<T>(fn: () => T): T;
  function withSuspendedDrains<T>(fn: () => T | Promise<T>): T | Promise<T> {
    return fn();
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
      return (): void => {};
    },
    get isPending(): boolean {
      return false;
    },
    get inFlight(): number {
      return inFlight;
    },
  };
}

/**
 * `ServerResponse` mock that satisfies `endJson` / `flushTerminalSSE`
 * — both of which call `res.end(body, cb)` / `res.write(chunk, cb)`
 * and await the callback. The mock invokes the callback
 * asynchronously (on `queueMicrotask`) to mirror real Node
 * behaviour.
 *
 * Crucially, this mock can fire `close` at any time via
 * `simulateAbort()` — enabling the round-4 regression test where
 * the socket closes mid-request while the native model is still
 * "running". When `simulateAbort` is called BEFORE `end()`, the
 * handler's transport-visibility layer sees
 * `socket.destroyed === true` in the pre-check and rejects
 * synchronously — which is the socket-gone branch the eager-attach
 * fix is meant to keep from leaking `inFlight`.
 */
function createMockRes(): {
  res: ServerResponse;
  simulateAbort: () => void;
  completeEnd: () => void;
  getStatus: () => number;
  getBody: () => string;
} {
  const emitter = new EventEmitter();
  const writable = emitter as unknown as ServerResponse & {
    writableEnded: boolean;
    writableFinished: boolean;
    headersSent: boolean;
    socket: {
      destroyed: boolean;
      once: (event: string, cb: () => void) => void;
      removeListener: (event: string, cb: () => void) => void;
    } | null;
  };
  let status = 200;
  let body = '';
  let socketDestroyed = false;

  // Minimal socket that the transport-visibility layer inspects via
  // `isSocketGone(res)`. Shares `destroyed` with the response-level
  // close path, and forwards its own `close` listener registration
  // to the response emitter so a single `close` event wakes both.
  const socket = {
    get destroyed() {
      return socketDestroyed;
    },
    once: (event: string, cb: () => void): void => {
      emitter.once(event, cb);
    },
    removeListener: (event: string, cb: () => void): void => {
      emitter.removeListener(event, cb);
    },
  };
  writable.writableEnded = false;
  writable.writableFinished = false;
  writable.headersSent = false;
  // The real socket is a `net.Socket`; tests only care about
  // `destroyed` + `once('close', …)` + `removeListener('close', …)`.
  // Cast through `unknown` so the minimal mock satisfies the handler's
  // `res.socket` usage without pulling in the full socket surface.
  writable.socket = socket as unknown as typeof writable.socket;

  writable.writeHead = ((s: number, _h?: Record<string, string>) => {
    status = s;
    writable.headersSent = true;
    return writable;
  }) as ServerResponse['writeHead'];
  writable.setHeader = (() => writable) as unknown as ServerResponse['setHeader'];
  writable.getHeader = (() => undefined) as unknown as ServerResponse['getHeader'];
  writable.removeHeader = (() => {}) as ServerResponse['removeHeader'];

  writable.write = ((chunkArg?: unknown, ...rest: unknown[]) => {
    let chunk: string | Uint8Array | undefined;
    let cb: ((err?: Error | null) => void) | undefined;
    if (typeof chunkArg === 'function') {
      cb = chunkArg as (err?: Error | null) => void;
    } else {
      chunk = chunkArg as string | Uint8Array | undefined;
      for (const arg of rest) {
        if (typeof arg === 'function') {
          cb = arg as (err?: Error | null) => void;
          break;
        }
      }
    }
    if (chunk != null) body += typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8');
    if (cb) queueMicrotask(() => cb!(null));
    return true;
  }) as ServerResponse['write'];

  writable.end = ((chunkArg?: unknown, ...rest: unknown[]) => {
    let chunk: string | Uint8Array | undefined;
    let cb: ((err?: Error | null) => void) | undefined;
    if (typeof chunkArg === 'function') {
      cb = chunkArg as (err?: Error | null) => void;
    } else {
      chunk = chunkArg as string | Uint8Array | undefined;
      for (const arg of rest) {
        if (typeof arg === 'function') {
          cb = arg as (err?: Error | null) => void;
          break;
        }
      }
    }
    if (chunk != null) body += typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8');
    writable.headersSent = true;
    writable.writableEnded = true;
    if (cb) queueMicrotask(() => cb!(null));
    return writable;
  }) as ServerResponse['end'];

  writable.destroy = ((err?: Error) => {
    socketDestroyed = true;
    if (err) emitter.emit('error', err);
    emitter.emit('close');
    writable.writableEnded = true;
    return writable;
  }) as ServerResponse['destroy'];

  return {
    res: writable,
    simulateAbort(): void {
      // Mirror a mid-request client disconnect: flip `socket.destroyed`
      // AND emit `close`. The transport-visibility layer checks
      // `isSocketGone(res)` before every write; a `close` that
      // arrives between eager-attach and the outer finally exercises
      // the listener path; a pre-end destroy exercises the
      // socket-gone early-reject path.
      socketDestroyed = true;
      emitter.emit('close');
    },
    completeEnd(): void {
      writable.writableFinished = true;
      emitter.emit('finish');
    },
    getStatus: () => status,
    getBody: () => body,
  };
}

/**
 * Session-capable mock whose `chatStreamSessionStart` yields the
 * caller-supplied event sequence. Abort tests pass a stream that
 * throws to simulate native-side cancellation on `AbortSignal.abort()`.
 */
function createMockStreamModelAbortable(streamFactory: () => AsyncGenerator<Record<string, unknown>>): unknown {
  return {
    chatSessionStart: vi.fn().mockRejectedValue(new Error('should use chatStreamSessionStart')),
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('should use chatStreamSessionContinue')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('should use chatStreamSessionContinueTool')),
    chatStreamSessionStart: vi.fn(() => streamFactory()),
    chatStreamSessionContinue: vi.fn(() => streamFactory()),
    chatStreamSessionContinueTool: vi.fn(() => streamFactory()),
    resetCaches: vi.fn(),
  };
}

// ---------------------------------------------------------------------------
// /v1/responses
// ---------------------------------------------------------------------------

describe('handleCreateResponse idle-sweeper abort path', () => {
  it('decrements inFlight exactly once when the socket aborts while the stream is still running', async () => {
    const registry = new ModelRegistry();
    const sweeper = createCountingSweeper();

    // Stream that yields one delta then never terminates — the
    // handler keeps awaiting until the AbortController fires (via
    // the eagerly-attached `close` listener) or the socket-gone
    // pre-check trips inside `flushTerminalSSE`. The test simulates
    // a client disconnect while the stream is in flight; the
    // handler's finally block must still end up firing
    // `endRequest()` exactly once.
    async function* streamEvents() {
      yield { done: false, text: 'hi' };
      // Yield a terminal AFTER a macrotask tick so the test has
      // time to call `simulateAbort()` while the handler is parked
      // on the next-iter await.
      await new Promise((r) => setTimeout(r, 0));
      yield {
        done: true,
        text: 'hi',
        finishReason: 'stop',
        toolCalls: [],
        thinking: null,
        numTokens: 1,
        promptTokens: 1,
        reasoningTokens: 0,
        rawText: 'hi',
      };
    }

    const model = createMockStreamModelAbortable(() => streamEvents());
    registry.register('abort-test-model', model as never);

    const { res, simulateAbort } = createMockRes();

    const handlerPromise = handleCreateResponse(
      res,
      {
        model: 'abort-test-model',
        input: 'hello',
        stream: true,
      } as never,
      registry,
      null,
      undefined,
      undefined,
      sweeper,
    );

    // Let the handler reach the inner stream loop.
    await new Promise((r) => setImmediate(r));
    await new Promise((r) => setImmediate(r));
    simulateAbort();

    await handlerPromise;

    // `inFlight` must be exactly zero. With the round-3 lazy-attach
    // design this would often be stuck at 1 — the `close` fired
    // before the outer `finally` attached listeners, and the
    // `writableEnded`/`writableFinished` probe saw `false` (the
    // handler hadn't called `res.end()` yet because it errored
    // out), so the attach happened on an already-closed emitter
    // and `endRequest()` never fired. The round-4 fix attaches
    // listeners at `beginRequest()` time and the outer `finally`
    // unconditionally calls `finalize()` — so whichever path fires,
    // the `done` flag guarantees exactly one decrement.
    expect(sweeper.inFlight).toBe(0);
  });

  it('decrements inFlight exactly once on the happy path', async () => {
    const registry = new ModelRegistry();
    const sweeper = createCountingSweeper();

    async function* streamEvents() {
      yield {
        done: true,
        text: 'ok',
        finishReason: 'stop',
        toolCalls: [],
        thinking: null,
        numTokens: 1,
        promptTokens: 1,
        reasoningTokens: 0,
        rawText: 'ok',
      };
    }

    const model = createMockStreamModelAbortable(() => streamEvents());
    registry.register('happy-path-model', model as never);

    const { res, completeEnd } = createMockRes();

    const handlerPromise = handleCreateResponse(
      res,
      {
        model: 'happy-path-model',
        input: 'hello',
        stream: true,
      } as never,
      registry,
      null,
      undefined,
      undefined,
      sweeper,
    );

    await handlerPromise;
    // Fire `finish` AFTER the handler returns — mirroring the
    // real-world case where the TCP buffer drains after `end()` and
    // Node emits `finish` asynchronously. The eagerly-attached
    // listener catches this event; the outer `finally` already
    // called `finalize()` with the `done` flag set, so this `finish`
    // is a no-op (the listener has been detached during
    // `finalize()`'s post-call cleanup, but even if a stray
    // `finalize` landed it would trip the `idleRequestEnded` guard).
    completeEnd();

    expect(sweeper.inFlight).toBe(0);
  });

  it('does not decrement inFlight for pre-dispatch validation failures (early return)', async () => {
    const registry = new ModelRegistry();
    const sweeper = createCountingSweeper();

    const { res } = createMockRes();

    // Missing `model` field — `handleCreateResponse` short-circuits
    // before calling `idleSweeper.beginRequest()`. The counter must
    // stay at zero; an unconditional `finalize()` in the outer
    // `finally` would otherwise underflow to -1 (or trip the clamp
    // inside the real sweeper).
    await handleCreateResponse(res, { input: 'hello' } as never, registry, null, undefined, undefined, sweeper);

    expect(sweeper.inFlight).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// /v1/messages
// ---------------------------------------------------------------------------

describe('handleCreateMessage idle-sweeper abort path', () => {
  it('decrements inFlight exactly once when the socket fires close mid-stream', async () => {
    const registry = new ModelRegistry();
    const sweeper = createCountingSweeper();

    async function* streamEvents() {
      yield { done: false, text: 'hi' };
      yield {
        done: true,
        text: 'hi',
        finishReason: 'stop',
        toolCalls: [],
        thinking: null,
        numTokens: 1,
        promptTokens: 1,
        reasoningTokens: 0,
        rawText: 'hi',
      };
    }

    const model = createMockStreamModelAbortable(() => streamEvents());
    registry.register('abort-msg-model', model as never);

    const { res, simulateAbort } = createMockRes();

    const handlerPromise = handleCreateMessage(
      res,
      {
        model: 'abort-msg-model',
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 16,
        stream: true,
      },
      registry,
      undefined,
      sweeper,
    );

    await new Promise((r) => setImmediate(r));
    simulateAbort();

    await handlerPromise;

    expect(sweeper.inFlight).toBe(0);
  });

  it('decrements inFlight exactly once on the happy path when finish lands on the eager listener', async () => {
    const registry = new ModelRegistry();
    const sweeper = createCountingSweeper();

    async function* streamEvents() {
      yield {
        done: true,
        text: 'ok',
        finishReason: 'stop',
        toolCalls: [],
        thinking: null,
        numTokens: 1,
        promptTokens: 1,
        reasoningTokens: 0,
        rawText: 'ok',
      };
    }

    const model = createMockStreamModelAbortable(() => streamEvents());
    registry.register('happy-msg-model', model as never);

    const { res, completeEnd } = createMockRes();

    const handlerPromise = handleCreateMessage(
      res,
      {
        model: 'happy-msg-model',
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 16,
        stream: true,
      },
      registry,
      undefined,
      sweeper,
    );

    await handlerPromise;
    completeEnd();

    expect(sweeper.inFlight).toBe(0);
  });

  it('does not decrement inFlight for pre-dispatch validation failures (early return)', async () => {
    const registry = new ModelRegistry();
    const sweeper = createCountingSweeper();

    const { res } = createMockRes();

    // Missing `model` — short-circuits before `beginRequest()`.
    await handleCreateMessage(
      res,
      { messages: [{ role: 'user', content: 'hi' }], max_tokens: 16 } as never,
      registry,
      undefined,
      sweeper,
    );

    expect(sweeper.inFlight).toBe(0);
  });
});
