/**
 * The dashboard RPC, driven over a real `node:worker_threads` `MessageChannel`:
 * two real ports, real structured clone, no Electron. That is the whole reason
 * the protocol is a separate module from the Electron shell — every rule it has
 * to keep (no caller hangs, subscriptions really stop, envelopes survive the
 * clone) is provable here.
 */

import { MessageChannel, type MessagePort } from 'node:worker_threads';

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { failure, type ApiResponse } from '../src/api/errors.js';
import type { DownloadEvent } from '../src/download.js';
import { createRpcClient, type RpcClient } from '../src/rpc/client.js';
import { serveRuntimeOverPort, type RpcRuntime } from '../src/rpc/host.js';
import {
  bindEventEmitterPort,
  bindEventTargetPort,
  type EventEmitterPort,
  type EventTargetPort,
  type RpcPortEvents,
} from '../src/rpc/port.js';
import type { RpcRequest } from '../src/rpc/protocol.js';
import type { ApiCall } from '../src/runtime.js';

const cleanups: Array<() => void> = [];

afterEach(() => {
  while (cleanups.length > 0) {
    try {
      cleanups.pop()?.();
    } catch {
      // A port already torn down by the test itself: nothing to do.
    }
  }
});

/** Let queued port messages cross. One turn per hop, with slack. */
async function flush(turns = 4): Promise<void> {
  for (let i = 0; i < turns; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * Settle `promise`, or resolve to `'HUNG'` after `ms`.
 *
 * Every "must not hang" assertion goes through this rather than relying on the
 * suite timeout: a test that proves its point by taking 120 s to fail is one
 * nobody will run the mutation against.
 */
async function within<T>(promise: Promise<T>, ms = 2000): Promise<T | 'HUNG'> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const sentinel = new Promise<'HUNG'>((resolve) => {
    timer = setTimeout(() => resolve('HUNG'), ms);
  });
  try {
    return await Promise.race([promise, sentinel]);
  } finally {
    clearTimeout(timer);
  }
}

interface Stub {
  runtime: RpcRuntime;
  /** Fire a progress event at every live subscriber of `jobId`. */
  emit(jobId: string, event: DownloadEvent): void;
  /** How many `runtime.subscribe` registrations are still live. */
  liveSubscriptions(): number;
  calls: ApiCall[];
  signals: Array<AbortSignal | undefined>;
}

function stubRuntime(answer?: (call: ApiCall, signal?: AbortSignal) => ApiResponse | Promise<ApiResponse>): Stub {
  const subscribers = new Map<number, { jobId: string; listener: (event: DownloadEvent) => void }>();
  let nextKey = 1;
  const calls: ApiCall[] = [];
  const signals: Array<AbortSignal | undefined> = [];
  return {
    calls,
    signals,
    runtime: {
      async call(call: ApiCall, signal?: AbortSignal): Promise<ApiResponse> {
        calls.push(call);
        signals.push(signal);
        return (await answer?.(call, signal)) ?? { ok: true, status: 200, body: { echo: call.path } };
      },
      subscribe(jobId: string, listener: (event: DownloadEvent) => void): () => void {
        const key = nextKey++;
        subscribers.set(key, { jobId, listener });
        return () => {
          subscribers.delete(key);
        };
      },
    },
    emit(jobId: string, event: DownloadEvent): void {
      for (const s of subscribers.values()) if (s.jobId === jobId) s.listener(event);
    },
    liveSubscriptions(): number {
      return subscribers.size;
    },
  };
}

/** A live client/host pair over a real MessageChannel. */
function connected(
  stub: Stub,
  opts: { timeoutMs?: number; cancellationGraceMs?: number; onUnresponsive?(reason: string): void } = {},
): { client: RpcClient; dispose: () => void; hostPort: MessagePort; clientPort: MessagePort } {
  const { port1, port2 } = new MessageChannel();
  // Never hold the event loop open on a test channel's account.
  port1.unref();
  port2.unref();
  // No cast: `node:worker_threads`' MessagePort satisfies `EventTargetPort`
  // structurally, which is the point of keeping that interface to four members.
  const dispose = serveRuntimeOverPort(stub.runtime, bindEventTargetPort(port2));
  const client = createRpcClient(bindEventTargetPort(port1), {
    ...opts,
    onUnresponsive(reason): void {
      if (opts.onUnresponsive === undefined) port2.close();
      else opts.onUnresponsive(reason);
    },
  });
  cleanups.push(() => {
    client.close();
    dispose();
  });
  return { client, dispose, hostPort: port2, clientPort: port1 };
}

const PROGRESS = {
  type: 'progress',
  id: 'job',
  file: 'model.safetensors',
  receivedBytes: 10,
  jobReceivedBytes: 10,
  totalBytes: 100,
  fileIndex: 0,
  fileCount: 1,
} satisfies DownloadEvent;

describe('rpc port adapters', () => {
  it('unwraps a MessageEvent, starts delivery, and detaches on demand (EventTarget port)', () => {
    const listeners = new Map<string, Array<(event: unknown) => void>>();
    let started = false;
    let closed = false;
    const fake: EventTargetPort = {
      postMessage: () => {},
      addEventListener(type, listener) {
        const list = listeners.get(type) ?? [];
        list.push(listener);
        listeners.set(type, list);
      },
      removeEventListener(type, listener) {
        listeners.set(
          type,
          (listeners.get(type) ?? []).filter((l) => l !== listener),
        );
      },
      start() {
        started = true;
      },
      close() {
        closed = true;
      },
    };

    const seen: unknown[] = [];
    let closes = 0;
    const port = bindEventTargetPort(fake);
    const detach = port.listen({ onMessage: (d) => seen.push(d), onClose: () => closes++ });

    // Delivery is started by the binder: a listener added with `addEventListener`
    // (rather than by assigning `onmessage`) does not start a web MessagePort.
    expect(started).toBe(true);

    for (const l of listeners.get('message') ?? []) l({ data: { hello: 1 } });
    for (const l of listeners.get('close') ?? []) l({});
    expect(seen).toEqual([{ hello: 1 }]);
    expect(closes).toBe(1);

    detach();
    for (const l of listeners.get('message') ?? []) l({ data: { hello: 2 } });
    for (const l of listeners.get('close') ?? []) l({});
    expect(seen).toEqual([{ hello: 1 }]);
    expect(closes).toBe(1);

    port.close();
    expect(closed).toBe(true);
  });

  it('unwraps a MessageEvent, starts delivery, and detaches on demand (EventEmitter port)', () => {
    // Mirrors Electron's `MessagePortMain`: `.on`/`.off`, a `{ data }` wrapper on
    // 'message', and NOTHING delivered until `start()`.
    const listeners = new Map<string, Array<(payload: unknown) => void>>();
    let started = false;
    const deliver = (type: string, payload: unknown): void => {
      if (!started) return;
      for (const l of listeners.get(type) ?? []) l(payload);
    };
    const fake: EventEmitterPort = {
      postMessage: () => {},
      on(event, listener) {
        const list = listeners.get(event) ?? [];
        list.push(listener);
        listeners.set(event, list);
      },
      off(event, listener) {
        listeners.set(
          event,
          (listeners.get(event) ?? []).filter((l) => l !== listener),
        );
      },
      start() {
        started = true;
      },
      close() {},
    };

    const seen: unknown[] = [];
    let closes = 0;
    const detach = bindEventEmitterPort(fake).listen({ onMessage: (d) => seen.push(d), onClose: () => closes++ });

    deliver('message', { data: { hello: 1 } });
    deliver('close', undefined);
    expect(seen).toEqual([{ hello: 1 }]);
    expect(closes).toBe(1);

    detach();
    deliver('message', { data: { hello: 2 } });
    expect(seen).toEqual([{ hello: 1 }]);
  });
});

describe('rpc request/response', () => {
  it('round-trips a success envelope through a real structured clone', async () => {
    const stub = stubRuntime(() => ({ ok: true, status: 202, body: { id: 'job-1', nested: [1, 2, { deep: true }] } }));
    const { client } = connected(stub);

    const res = await within(client.call({ method: 'POST', path: '/api/downloads', body: { repo: 'org/repo' } }));
    expect(res).not.toBe('HUNG');
    expect(res).toEqual({ ok: true, status: 202, body: { id: 'job-1', nested: [1, 2, { deep: true }] } });
    // The call reached the runtime verbatim, query string and all.
    expect(stub.calls).toEqual([{ method: 'POST', path: '/api/downloads', body: { repo: 'org/repo' } }]);
  });

  it('carries a path with its query string as text (URLSearchParams cannot be cloned)', async () => {
    const stub = stubRuntime();
    const { client } = connected(stub);
    expect(await within(client.call({ method: 'GET', path: '/api/sessions?limit=1&offset=2' }))).not.toBe('HUNG');
    expect(stub.calls[0].path).toBe('/api/sessions?limit=1&offset=2');
  });

  it('correlates concurrent replies that come back out of order', async () => {
    const gates = new Map<string, (response: ApiResponse) => void>();
    const stub = stubRuntime(
      (call) =>
        new Promise<ApiResponse>((resolve) => {
          gates.set(call.path, resolve);
        }),
    );
    const { client } = connected(stub);

    const a = client.call({ method: 'GET', path: '/a' });
    const b = client.call({ method: 'GET', path: '/b' });
    const c = client.call({ method: 'GET', path: '/c' });
    await flush();

    // Answer in reverse order; each caller must still get ITS answer.
    gates.get('/c')!({ ok: true, status: 200, body: 'C' });
    gates.get('/a')!({ ok: true, status: 200, body: 'A' });
    gates.get('/b')!({ ok: true, status: 200, body: 'B' });

    expect(await within(Promise.all([a, b, c]))).toEqual([
      { ok: true, status: 200, body: 'A' },
      { ok: true, status: 200, body: 'B' },
      { ok: true, status: 200, body: 'C' },
    ]);
  });

  it('round-trips a failure envelope with its code AND its derived status', async () => {
    const stub = stubRuntime(() => failure('E_CONFLICT', 'Session is being written'));
    const { client } = connected(stub);

    const res = await within(client.call({ method: 'PATCH', path: '/api/sessions/x' }));
    expect(res).toEqual({ ok: false, code: 'E_CONFLICT', message: 'Session is being written', status: 409 });
  });

  it('drops a malformed reply instead of settling a real call with it', async () => {
    let answer: ((response: ApiResponse) => void) | undefined;
    const stub = stubRuntime(
      () =>
        new Promise<ApiResponse>((resolve) => {
          answer = resolve;
        }),
    );
    const { client, hostPort } = connected(stub, { timeoutMs: 60_000 });

    // Posting on the HOST's port is what the CLIENT receives — a port delivers to
    // its peer. Ids are handed out from 1, so this junk claims the in-flight
    // call's id, which is the dangerous shape: accepting it settles a real caller
    // with a payload the runtime never produced.
    const inflight = client.call({ method: 'GET', path: '/api/models' });
    await flush();
    hostPort.postMessage({ kind: 'response', id: 1 }); // no envelope
    hostPort.postMessage({ kind: 'nope', id: 1, response: { ok: true, status: 200, body: 'JUNK' } });
    hostPort.postMessage({ kind: 'response', id: '1', response: { ok: true, status: 200, body: 'JUNK' } });
    await flush();

    answer!({ ok: true, status: 200, body: 'REAL' });
    expect(await within(inflight, 1000)).toEqual({ ok: true, status: 200, body: 'REAL' });
  });

  it('drops a malformed request instead of dispatching it to the runtime', async () => {
    const stub = stubRuntime();
    const { client, clientPort } = connected(stub);

    // Posting on the CLIENT's port is what the HOST receives.
    clientPort.postMessage({ kind: 'call', id: 90, call: 'not-an-object' });
    clientPort.postMessage({ kind: 'call', call: { method: 'GET', path: '/hijack' } }); // no id
    clientPort.postMessage({ kind: 'subscribe', id: 91 }); // no jobId
    await flush();

    // Nothing malformed reached the runtime, and the channel still serves.
    expect(await within(client.call({ method: 'GET', path: '/api/models' }))).not.toBe('HUNG');
    expect(stub.calls).toEqual([{ method: 'GET', path: '/api/models' }]);
    expect(stub.liveSubscriptions()).toBe(0);
  });

  it('survives non-object junk on the wire in both directions', async () => {
    const stub = stubRuntime();
    const { client, hostPort, clientPort } = connected(stub);

    for (const junk of [null, 42, 'hello', undefined, []]) {
      clientPort.postMessage(junk);
      hostPort.postMessage(junk);
    }
    await flush();

    const res = await within(client.call({ method: 'GET', path: '/api/models' }));
    expect(res).toEqual({ ok: true, status: 200, body: { echo: '/api/models' } });
  });
});

describe('rpc never hangs a caller', () => {
  it('takes down a transport that stays silent after cancellation without claiming the mutation was cancelled', async () => {
    vi.useFakeTimers();
    try {
      const order: string[] = [];
      let closes = 0;
      let events: RpcPortEvents | undefined;
      const client = createRpcClient(
        {
          postMessage(message: unknown): void {
            order.push((message as { kind: string }).kind);
          },
          listen(next): () => void {
            events = next;
            return () => {
              order.push('detached');
            };
          },
          close(): void {
            closes += 1;
            order.push('closed');
          },
        },
        {
          timeoutMs: 100,
          cancellationGraceMs: 50,
          onUnresponsive: () => order.push('recover'),
        },
      );

      const response = client.call({ method: 'DELETE', path: '/api/sessions/queued' }).then((result) => {
        order.push('settled');
        return result;
      });
      await vi.advanceTimersByTimeAsync(100);
      expect(order).toEqual(['call', 'cancel']);

      // The first deadline only requests withdrawal. The original response is
      // still authoritative throughout the grace window.
      await vi.advanceTimersByTimeAsync(49);
      expect(order).toEqual(['call', 'cancel']);

      await vi.advanceTimersByTimeAsync(1);
      expect(order).toEqual(['call', 'cancel', 'recover']);
      expect(closes).toBe(0);

      // Merely posting the recovery request is not proof that the queued
      // mutation died. The old process's port close is that witness.
      events!.onClose();
      const result = await response;
      expect(result).toMatchObject({ ok: false, code: 'E_UNAVAILABLE', status: 503 });
      expect((result as { message: string }).message).toMatch(/outcome is unknown/i);
      expect((result as { message: string }).message).not.toMatch(/cancellation acknowledged/i);
      // Recovery is requested before the peer-death witness exposes the unknown
      // outcome, and the suspect port is then retired.
      expect(order).toEqual(['call', 'cancel', 'recover', 'detached', 'closed', 'settled']);
      expect(closes).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('bounds graceful close when its cancellation acknowledgement never arrives', async () => {
    vi.useFakeTimers();
    try {
      const posted: RpcRequest[] = [];
      let closes = 0;
      let recoveries = 0;
      let events: RpcPortEvents | undefined;
      let settled = false;
      const client = createRpcClient(
        {
          postMessage: (message) => posted.push(message as RpcRequest),
          listen(next): () => void {
            events = next;
            return () => {};
          },
          close: () => {
            closes += 1;
          },
        },
        {
          timeoutMs: 60_000,
          cancellationGraceMs: 25,
          onUnresponsive: () => {
            recoveries += 1;
          },
        },
      );

      const pending = client.call({ method: 'POST', path: '/api/models/queued' }).then((result) => {
        settled = true;
        return result;
      });
      client.close();
      expect(posted.map((message) => message.kind)).toEqual(['call', 'cancel']);
      await vi.advanceTimersByTimeAsync(24);
      expect(closes).toBe(0);

      await vi.advanceTimersByTimeAsync(1);
      expect(recoveries).toBe(1);
      expect(closes).toBe(0);
      expect(settled).toBe(false);

      events!.onClose();
      const result = await pending;
      expect(result).toMatchObject({ ok: false, code: 'E_UNAVAILABLE' });
      expect((result as { message: string }).message).toMatch(/outcome is unknown/i);
      expect(closes).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('keeps a timed-out call pending until cancellation is acknowledged', async () => {
    const order: string[] = [];
    let events: RpcPortEvents | undefined;
    const client = createRpcClient(
      {
        postMessage(message: unknown): void {
          order.push((message as { kind: string }).kind);
        },
        listen(next): () => void {
          events = next;
          return () => {};
        },
        close: () => {},
      },
      { timeoutMs: 0, onUnresponsive: () => events?.onClose() },
    );

    try {
      let settled = false;
      const response = client.call({ method: 'DELETE', path: '/api/sessions/doomed' }).then((res) => {
        settled = true;
        order.push('settled');
        return res;
      });
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(order).toEqual(['call', 'cancel']);
      expect(settled).toBe(false);

      events!.onMessage({
        kind: 'response',
        id: 1,
        response: failure('E_UNAVAILABLE', 'Cancellation acknowledged'),
      });
      expect(await response).toMatchObject({ ok: false, code: 'E_UNAVAILABLE' });
      expect(order).toEqual(['call', 'cancel', 'settled']);
    } finally {
      client.close();
    }
  });

  it('settles after the host acknowledges cancellation of an unanswered request', async () => {
    const stub = stubRuntime(
      (_call, signal) =>
        new Promise<ApiResponse>((resolve) => {
          signal?.addEventListener('abort', () => resolve(failure('E_UNAVAILABLE', 'Cancellation acknowledged')), {
            once: true,
          });
        }),
    );
    const { client } = connected(stub, { timeoutMs: 50 });

    const res = await within(client.call({ method: 'GET', path: '/api/models' }));
    expect(res).toMatchObject({ ok: false, code: 'E_UNAVAILABLE', status: 503 });
    expect(res).not.toBe('HUNG');
    await flush();
    // The host correlated `cancel` to this connection's call and propagated it
    // as an AbortSignal instead of leaving the runtime operation orphaned.
    expect(stub.signals).toHaveLength(1);
    expect(stub.signals[0]?.aborted).toBe(true);
  });

  it('returns the real reply when cancellation loses the race to started work', async () => {
    let answer: ((response: ApiResponse) => void) | undefined;
    const stub = stubRuntime(
      () =>
        new Promise<ApiResponse>((resolve) => {
          answer = resolve;
        }),
    );
    const { client } = connected(stub, { timeoutMs: 50 });

    const first = client.call({ method: 'GET', path: '/api/models' });
    expect(await within(first, 100)).toBe('HUNG');

    // The deadline requested cancellation, but this stub models a handler that
    // had already started and therefore returns its authoritative result.
    answer!({ ok: true, status: 200, body: 'completed' });
    expect(await within(first, 1000)).toEqual({ ok: true, status: 200, body: 'completed' });
  });

  it('settles every in-flight request when the peer dies, well before any deadline', async () => {
    const stub = stubRuntime(() => new Promise<ApiResponse>(() => {}));
    // A deadline far longer than the assertion window: only the close handling can
    // settle these in time.
    const { client, hostPort } = connected(stub, { timeoutMs: 60_000 });

    const a = client.call({ method: 'GET', path: '/a' });
    const b = client.call({ method: 'GET', path: '/b' });
    await flush();

    hostPort.close();

    const settled = await within(Promise.all([a, b]), 1000);
    expect(settled).not.toBe('HUNG');
    for (const res of settled as ApiResponse[]) {
      expect(res).toMatchObject({ ok: false, code: 'E_UNAVAILABLE', status: 503 });
      expect((res as { message: string }).message).toMatch(/call outcome is unknown/i);
    }
  });

  it('fails a request issued after the peer died, without waiting out the deadline', async () => {
    const stub = stubRuntime();
    const { client, hostPort } = connected(stub, { timeoutMs: 60_000 });
    hostPort.close();
    await flush();

    const res = await within(client.call({ method: 'GET', path: '/api/models' }), 1000);
    expect(res).toMatchObject({ ok: false, code: 'E_UNAVAILABLE', status: 503 });
  });

  it('keeps a closing client call pending until the runtime acknowledges cancellation', async () => {
    let acknowledge: ((response: ApiResponse) => void) | undefined;
    const stub = stubRuntime(
      () =>
        new Promise<ApiResponse>((resolve) => {
          acknowledge = resolve;
        }),
    );
    const { client } = connected(stub, { timeoutMs: 60_000 });

    const inflight = client.call({ method: 'GET', path: '/a' });
    await flush();
    client.close();
    await flush();

    expect(stub.signals[0]?.aborted).toBe(true);
    expect(await within(inflight, 100)).toBe('HUNG');
    expect(await client.call({ method: 'GET', path: '/new' })).toMatchObject({
      ok: false,
      code: 'E_UNAVAILABLE',
    });

    acknowledge!(failure('E_UNAVAILABLE', 'Cancellation acknowledged'));
    expect(await within(inflight, 1000)).toMatchObject({
      ok: false,
      code: 'E_UNAVAILABLE',
      message: 'Cancellation acknowledged',
    });
  });

  it('returns the real result when close cancellation loses the race to started work', async () => {
    let finish: ((response: ApiResponse) => void) | undefined;
    const stub = stubRuntime(
      () =>
        new Promise<ApiResponse>((resolve) => {
          finish = resolve;
        }),
    );
    const { client } = connected(stub, { timeoutMs: 60_000 });

    const inflight = client.call({ method: 'DELETE', path: '/api/sessions/already-started' });
    await flush();
    client.close();
    await flush();

    expect(stub.signals[0]?.aborted).toBe(true);
    expect(await within(inflight, 100)).toBe('HUNG');

    finish!({ ok: true, status: 204, body: null });
    expect(await within(inflight, 1000)).toEqual({ ok: true, status: 204, body: null });
  });

  it('closes immediately and idempotently when no calls are pending', async () => {
    const posted: RpcRequest[] = [];
    let detaches = 0;
    let closes = 0;
    const client = createRpcClient(
      {
        postMessage(message: unknown): void {
          posted.push(message as RpcRequest);
        },
        listen: () => () => {
          detaches += 1;
        },
        close(): void {
          closes += 1;
        },
      },
      { onUnresponsive: () => {} },
    );

    const unsubscribe = client.subscribe('job', () => {});
    client.close();
    client.close();
    unsubscribe();

    expect(posted).toEqual([
      { kind: 'subscribe', id: 1, jobId: 'job' },
      { kind: 'unsubscribe', id: 1 },
    ]);
    expect(detaches).toBe(1);
    expect(closes).toBe(1);
    expect(await client.call({ method: 'GET', path: '/new' })).toMatchObject({
      ok: false,
      code: 'E_UNAVAILABLE',
    });
  });

  it('settles a request whose body cannot be cloned, instead of waiting out the deadline', async () => {
    const stub = stubRuntime();
    const { client } = connected(stub, { timeoutMs: 60_000 });

    // A function is the canonical structured-clone refusal; `postMessage` throws
    // synchronously and the request never leaves this side.
    const res = await within(client.call({ method: 'POST', path: '/api/ingest', body: () => 'nope' }), 1000);
    expect(res).toMatchObject({ ok: false, code: 'E_BAD_REQUEST', status: 400 });
    expect((res as { message: string }).message).toMatch(/cannot cross the message port/i);
    // Nothing was delivered, so the runtime never saw it.
    expect(stub.calls).toHaveLength(0);
  });

  it('answers with a failure envelope when a handler body cannot be cloned', async () => {
    const stub = stubRuntime(() => ({ ok: true, status: 200, body: { render: () => 'nope' } }));
    const { client } = connected(stub, { timeoutMs: 60_000 });

    const res = await within(client.call({ method: 'GET', path: '/api/models' }), 1000);
    expect(res).toMatchObject({ ok: false, code: 'E_INTERNAL', status: 500 });
    expect((res as { message: string }).message).toMatch(/cannot cross the message port/i);
  });
});

describe('rpc subscriptions', () => {
  it('delivers each job’s events to its own subscriber', async () => {
    const stub = stubRuntime();
    const { client } = connected(stub);

    const a: DownloadEvent[] = [];
    const b: DownloadEvent[] = [];
    client.subscribe('job-a', (e) => a.push(e));
    client.subscribe('job-b', (e) => b.push(e));
    await flush();

    stub.emit('job-a', { ...PROGRESS, id: 'job-a' });
    stub.emit('job-b', { type: 'done', id: 'job-b', outputDir: '/models/b' });
    await flush();

    expect(a).toEqual([{ ...PROGRESS, id: 'job-a' }]);
    expect(b).toEqual([{ type: 'done', id: 'job-b', outputDir: '/models/b' }]);
  });

  it('stops delivering an event already in flight when the caller unsubscribes', async () => {
    const stub = stubRuntime();
    const { client } = connected(stub);

    const seen: DownloadEvent[] = [];
    const off = client.subscribe('job', (e) => seen.push(e));
    await flush();
    stub.emit('job', PROGRESS);
    await flush();
    expect(seen).toHaveLength(1);

    // The host has not seen the unsubscribe yet, so it still posts this event.
    // Only dropping the LOCAL listener can stop it being delivered.
    off();
    stub.emit('job', { ...PROGRESS, receivedBytes: 99 });
    await flush();
    expect(seen).toHaveLength(1);
  });

  it('releases the runtime subscription on unsubscribe', async () => {
    const stub = stubRuntime();
    const { client } = connected(stub);

    const off = client.subscribe('job', () => {});
    await flush();
    expect(stub.liveSubscriptions()).toBe(1);

    off();
    await flush();
    expect(stub.liveSubscriptions()).toBe(0);
  });

  it('releases every runtime subscription when the peer closes', async () => {
    const stub = stubRuntime();
    const { client, clientPort } = connected(stub);

    client.subscribe('job-a', () => {});
    client.subscribe('job-b', () => {});
    await flush();
    expect(stub.liveSubscriptions()).toBe(2);

    // The window went away without unsubscribing — the download manager outlives
    // it, so a listener left behind leaks for the life of the process.
    clientPort.close();
    await flush();
    expect(stub.liveSubscriptions()).toBe(0);
  });

  it('releases every runtime subscription when the host is disposed', async () => {
    const stub = stubRuntime();
    const { client, dispose } = connected(stub);

    client.subscribe('job-a', () => {});
    await flush();
    expect(stub.liveSubscriptions()).toBe(1);

    dispose();
    expect(stub.liveSubscriptions()).toBe(0);
  });

  it('does not deliver events to a subscriber after the peer died', async () => {
    const stub = stubRuntime();
    const { client, hostPort } = connected(stub);

    const seen: DownloadEvent[] = [];
    client.subscribe('job', (e) => seen.push(e));
    await flush();

    hostPort.close();
    await flush();
    // A late subscribe on a dead port is inert rather than a listener that would
    // never be released.
    client.subscribe('job', (e) => seen.push(e));
    stub.emit('job', PROGRESS);
    await flush();

    expect(seen).toEqual([]);
  });
});
