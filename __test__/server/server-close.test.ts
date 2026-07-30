import { once } from 'node:events';
import { request as httpRequest } from 'node:http';
import type { AddressInfo } from 'node:net';

import type { SessionCapableModel } from '@mlx-node/lm';
import { activeSSEStreamCount, createServer, type ServerInstance } from '@mlx-node/server';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

/**
 * Bounded, idempotent shutdown.
 *
 * Before this change `close()` was `clearInterval` + `idleSweeper.close()` +
 * `server.close()`. `server.close()` stops accepting NEW connections but waits
 * for every existing one to end — and an open SSE stream keeps its connection
 * active for the whole remaining generation. Shutdown latency was therefore
 * unbounded and controlled by whoever held the longest stream. A second
 * `close()` also rejected with `ERR_SERVER_NOT_RUNNING`.
 *
 * The forced path leans on machinery that already exists: `/v1/messages`
 * attaches `res.once('close', onAbortClose)` → `AbortController` → the
 * signal plumbed through `@mlx-node/lm`'s `_runChatStream` → native
 * `ChatStreamHandle.cancel()`. `server.closeAllConnections()` destroys the
 * socket, which fires exactly that `'close'` event. The stall model below
 * parks on the signal precisely so the test fails if that chain is broken.
 */

let servers: ServerInstance[] = [];

async function start(config: Parameters<typeof createServer>[0] = {}): Promise<{
  instance: ServerInstance;
  base: string;
}> {
  const instance = await createServer({ port: 0, host: '127.0.0.1', disableStore: true, ...config });
  servers.push(instance);
  const { port } = instance.server.address() as AddressInfo;
  return { instance, base: `http://127.0.0.1:${port}` };
}

afterEach(async () => {
  const pending = servers;
  servers = [];
  for (const instance of pending) {
    await instance.close({ timeoutMs: 250 }).catch(() => {});
  }
});

/**
 * A session-capable model whose stream emits one delta and then parks on the
 * abort signal — the same shape a real decode loop has while sitting in
 * `waitForItem()`. `entered` resolves once the generator body has actually
 * started, which is strictly AFTER `beginSSE(res)` has registered the
 * response as an active SSE stream.
 */
function createStallModel(): { model: SessionCapableModel; entered: Promise<void>; sawAbort: Promise<void> } {
  let resolveEntered: (() => void) | undefined;
  const entered = new Promise<void>((r) => {
    resolveEntered = r;
  });
  let resolveAbort: (() => void) | undefined;
  const sawAbort = new Promise<void>((r) => {
    resolveAbort = r;
  });

  async function* stall(
    _messages: unknown,
    _config: unknown,
    signal: AbortSignal | undefined,
  ): AsyncGenerator<Record<string, unknown>> {
    resolveEntered?.();
    yield { done: false, text: 'thinking', isReasoning: false };
    await new Promise<void>((resolve) => {
      if (signal?.aborted) {
        resolve();
        return;
      }
      signal?.addEventListener('abort', () => resolve(), { once: true });
    });
    resolveAbort?.();
    yield {
      done: true,
      text: '',
      finishReason: 'error',
      toolCalls: [],
      thinking: null,
      numTokens: 0,
      promptTokens: 0,
      reasoningTokens: 0,
      rawText: '',
    };
  }

  const model = {
    chatSessionStart: vi.fn().mockRejectedValue(new Error('should use the streaming path')),
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('should use the streaming path')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('should use the streaming path')),
    chatStreamSessionStart: vi.fn(stall),
    chatStreamSessionContinue: vi.fn(stall),
    chatStreamSessionContinueTool: vi.fn(stall),
    resetCaches: vi.fn(),
  } as unknown as SessionCapableModel;

  return { model, entered, sawAbort };
}

/** Kick off a streaming `/v1/messages` request; never awaited to completion. */
function openStream(base: string, model: string): Promise<Response> {
  return fetch(`${base}/v1/messages`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: 'hi' }],
      max_tokens: 64,
      stream: true,
    }),
  });
}

/** A model whose stream terminates immediately, so the SSE response ends normally. */
function createQuickModel(): SessionCapableModel {
  async function* quick(): AsyncGenerator<Record<string, unknown>> {
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
  return {
    chatSessionStart: vi.fn().mockRejectedValue(new Error('should use the streaming path')),
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('should use the streaming path')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('should use the streaming path')),
    chatStreamSessionStart: vi.fn(quick),
    chatStreamSessionContinue: vi.fn(quick),
    chatStreamSessionContinueTool: vi.fn(quick),
    resetCaches: vi.fn(),
  } as unknown as SessionCapableModel;
}

describe('active SSE accounting', () => {
  // The set backing `activeSSEStreamCount()` is module-scoped and therefore
  // process-wide. If a completed stream were left behind, `streamsAborted`
  // would over-report on every later shutdown — and grow without bound in a
  // long-lived server.

  it('returns to zero after a stream ends normally', async () => {
    const { instance, base } = await start();
    instance.registry.register('quick-model', createQuickModel());

    const res = await openStream(base, 'quick-model');
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain('message_stop');
    expect(activeSSEStreamCount()).toBe(0);
  });

  it('does not count completed streams towards a later forced close', async () => {
    const { instance, base } = await start();
    instance.registry.register('quick-model', createQuickModel());
    const { model, entered } = createStallModel();
    instance.registry.register('stall-model', model);

    await (await openStream(base, 'quick-model')).text();
    const stalled = await openStream(base, 'stall-model');
    await entered;

    const result = await instance.close({ timeoutMs: 200 });
    expect(result.streamsAborted).toBe(1);
    await stalled.body?.cancel().catch(() => {});
  });

  it('does not count a stream owned by another server', async () => {
    const { instance: jsonServer, base: jsonBase } = await start();
    const { instance: streamServer, base: streamBase } = await start();
    const stalled = createStallModel();
    streamServer.registry.register('stall-model', stalled.model);

    // Server A has one incomplete JSON upload but no SSE. Keeping the request
    // body open makes A's close hit its force deadline without involving a
    // model or another application-level promise.
    const requestArrived = once(jsonServer.server, 'request');
    const jsonRequest = httpRequest(`${jsonBase}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
    });
    const jsonRequestClosed = new Promise<void>((resolve) => {
      jsonRequest.once('error', () => resolve());
      jsonRequest.once('close', () => resolve());
    });
    jsonRequest.write('{"model":');
    await requestArrived;

    // Server B owns the only process-wide SSE stream and remains live while A
    // is forced closed.
    const streamResponse = await openStream(streamBase, 'stall-model');
    await stalled.entered;
    expect(activeSSEStreamCount()).toBe(1);

    const jsonClose = await jsonServer.close({ timeoutMs: 200 });
    expect(jsonClose).toMatchObject({ forced: true, streamsAborted: 0 });
    expect(activeSSEStreamCount()).toBe(1);

    await jsonRequestClosed;

    // B still owns and aborts its own stream later; A did not merely suppress
    // the global count.
    const streamClose = await streamServer.close({ timeoutMs: 200 });
    expect(streamClose).toMatchObject({ forced: true, streamsAborted: 1 });
    await stalled.sawAbort;
    await streamResponse.body?.cancel().catch(() => {});
    expect(activeSSEStreamCount()).toBe(0);
  });
});

describe('close() idempotency', () => {
  it('memoizes the promise so a second call is the same object', async () => {
    const { instance } = await start();
    const first = instance.close();
    const second = instance.close();
    expect(second).toBe(first);
    await expect(first).resolves.toMatchObject({ forced: false });
  });

  it('does not reject with ERR_SERVER_NOT_RUNNING on a second call', async () => {
    const { instance } = await start();
    const first = await instance.close();
    // Sequential (not concurrent) second call — this is the shape that used
    // to blow up: `server.close()` on an already-closed server invokes its
    // callback with ERR_SERVER_NOT_RUNNING.
    const second = await instance.close();
    expect(second).toEqual(first);
  });

  it('survives a third call long after the first settled', async () => {
    const { instance } = await start();
    await instance.close();
    await instance.close();
    await expect(instance.close()).resolves.toBeDefined();
  });

  it('ignores the timeout of a later call (first call owns the deadline)', async () => {
    const { instance } = await start();
    const first = instance.close({ timeoutMs: 4_000 });
    const second = instance.close({ timeoutMs: 1 });
    expect(second).toBe(first);
    await first;
  });
});

describe('close() on a quiet server', () => {
  it('reports forced:false with nothing aborted', async () => {
    const { instance } = await start();
    const result = await instance.close({ timeoutMs: 5_000 });
    expect(result.forced).toBe(false);
    expect(result.streamsAborted).toBe(0);
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
    expect(result.durationMs).toBeLessThan(5_000);
  });

  it('does not wait on an idle keep-alive connection', async () => {
    // undici parks the socket in its pool after a completed request. Without
    // `closeIdleConnections()` the shutdown would sit on it until the pool's
    // own keep-alive timeout.
    const { instance, base } = await start();
    expect((await fetch(`${base}/health`)).status).toBe(200);

    const startedAt = Date.now();
    const result = await instance.close({ timeoutMs: 5_000 });
    expect(result.forced).toBe(false);
    expect(Date.now() - startedAt).toBeLessThan(2_000);
  });

  it('stops the server from accepting new requests', async () => {
    const { instance, base } = await start();
    await instance.close({ timeoutMs: 1_000 });
    await expect(fetch(`${base}/health`)).rejects.toThrow();
  });
});

describe('close() with an in-flight SSE stream', () => {
  it('aborts the stream at the deadline and reports forced:true', async () => {
    const { instance, base } = await start();
    const { model, entered, sawAbort } = createStallModel();
    instance.registry.register('stall-model', model);

    const streamRes = await openStream(base, 'stall-model');
    expect(streamRes.status).toBe(200);
    // The generator body only runs on the first `next()`, which happens
    // strictly after `beginSSE(res)` registered the active stream.
    await entered;

    const startedAt = Date.now();
    const result = await instance.close({ timeoutMs: 200 });
    const elapsed = Date.now() - startedAt;

    expect(result.forced).toBe(true);
    expect(result.streamsAborted).toBe(1);
    // Bounded: without the forced path this would hang until the generation
    // finished (here: forever).
    expect(elapsed).toBeLessThan(5_000);
    expect(result.durationMs).toBeGreaterThanOrEqual(190);

    // The destroyed socket must reach the model's AbortSignal — this is the
    // whole point of forcing rather than just dropping the socket.
    await sawAbort;

    // Draining the client side after a forced close errors; that is expected.
    await streamRes.body?.cancel().catch(() => {});
  });

  it('waits for the deadline rather than killing the stream immediately', async () => {
    const { instance, base } = await start();
    const { model, entered } = createStallModel();
    instance.registry.register('stall-model', model);

    const streamRes = await openStream(base, 'stall-model');
    await entered;

    const startedAt = Date.now();
    const result = await instance.close({ timeoutMs: 700 });
    expect(Date.now() - startedAt).toBeGreaterThanOrEqual(650);
    expect(result.forced).toBe(true);
    await streamRes.body?.cancel().catch(() => {});
  });

  it('counts every open stream', async () => {
    const { instance, base } = await start();
    const a = createStallModel();
    const b = createStallModel();
    instance.registry.register('stall-a', a.model);
    instance.registry.register('stall-b', b.model);

    const resA = await openStream(base, 'stall-a');
    await a.entered;
    const resB = await openStream(base, 'stall-b');
    await b.entered;

    const result = await instance.close({ timeoutMs: 200 });
    expect(result.forced).toBe(true);
    expect(result.streamsAborted).toBe(2);
    await resA.body?.cancel().catch(() => {});
    await resB.body?.cancel().catch(() => {});
  });

  it('keeps the memoized result stable after a forced close', async () => {
    const { instance, base } = await start();
    const { model, entered } = createStallModel();
    instance.registry.register('stall-model', model);
    const streamRes = await openStream(base, 'stall-model');
    await entered;

    const first = await instance.close({ timeoutMs: 200 });
    const second = await instance.close({ timeoutMs: 200 });
    expect(second).toEqual(first);
    expect(second.forced).toBe(true);
    await streamRes.body?.cancel().catch(() => {});
  });
});
