/**
 * CONTROL PANEL's one rule: a new port replaces the old one.
 *
 * The runtime outlives every port and the download manager outlives the window,
 * so a subscription opened by a page that has since reloaded stays registered
 * forever — one leaked listener per reload, each still receiving progress for a
 * job nobody is watching, and each holding whatever the closure captured.
 *
 * Driven over a REAL `node:worker_threads` MessageChannel with the REAL
 * `serveRuntimeOverPort`, because the leak is a property of that function's
 * dispose, not of anything this module could assert about itself. The RPC layer
 * is transport-agnostic on purpose (`rpc/port.ts`), which is what makes this
 * reachable without Electron at all.
 */

import { MessageChannel, type MessagePort } from 'node:worker_threads';

import { bindEventTargetPort, type ApiCall, type ApiResponse, type DownloadEvent } from '@mlx-node/dashboard';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { createControlPanelSession, type ControlPanelSessionRuntime } from '../src/control-panel/session.js';

const open: MessagePort[] = [];

afterEach(() => {
  for (const port of open.splice(0)) port.close();
});

function channel(): { host: MessagePort; peer: MessagePort } {
  const { port1, port2 } = new MessageChannel();
  open.push(port1, port2);
  return { host: port1, peer: port2 };
}

interface StubRuntime extends ControlPanelSessionRuntime {
  calls: ApiCall[];
  /** One entry per `subscribe`, flipped to true when its unsubscribe runs. */
  released: boolean[];
  emit(index: number, event: DownloadEvent): void;
  closed: number;
}

function stubRuntime(): StubRuntime {
  const listeners: ((event: DownloadEvent) => void)[] = [];
  const runtime: StubRuntime = {
    calls: [],
    released: [],
    closed: 0,
    call(call: ApiCall): Promise<ApiResponse> {
      runtime.calls.push(call);
      return Promise.resolve({ ok: true, status: 200, body: { path: call.path } });
    },
    subscribe(_jobId: string, listener: (event: DownloadEvent) => void): () => void {
      const index = listeners.push(listener) - 1;
      runtime.released.push(false);
      return () => {
        runtime.released[index] = true;
      };
    },
    emit(index: number, event: DownloadEvent): void {
      listeners[index](event);
    },
    close(): Promise<void> {
      runtime.closed += 1;
      return Promise.resolve();
    },
  };
  return runtime;
}

/** Send one request and wait for the reply, or give up. */
function ask(port: MessagePort, request: unknown, timeoutMs = 500): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('no reply')), timeoutMs);
    port.once('message', (data: unknown) => {
      clearTimeout(timer);
      resolve(data);
    });
    port.postMessage(request);
  });
}

/** Collect everything that arrives in `ms`. */
function drain(port: MessagePort, ms = 120): Promise<unknown[]> {
  const seen: unknown[] = [];
  port.on('message', (data: unknown) => seen.push(data));
  return new Promise((resolve) => setTimeout(() => resolve(seen), ms));
}

describe('attach', () => {
  it('serves calls on the attached port', async () => {
    const runtime = stubRuntime();
    const session = createControlPanelSession({ runtime });
    const { host, peer } = channel();
    session.attach(bindEventTargetPort(host as never));

    const reply = await ask(peer, { kind: 'call', id: 1, call: { method: 'GET', path: '/api/models' } });
    expect(reply).toEqual({
      kind: 'response',
      id: 1,
      response: { ok: true, status: 200, body: { path: '/api/models' } },
    });
    expect(runtime.calls).toEqual([{ method: 'GET', path: '/api/models' }]);
  });

  it('releases the previous port’s subscriptions when a reload brings a new one', async () => {
    const runtime = stubRuntime();
    const session = createControlPanelSession({ runtime });

    const first = channel();
    session.attach(bindEventTargetPort(first.host as never));
    first.peer.postMessage({ kind: 'subscribe', id: 9, jobId: 'job-a' });
    await settle();
    expect(runtime.released).toEqual([false]);

    // The reload. MAIN mints a fresh `MessageChannelMain` because a transferred
    // port is consumed once — see `src/main/broker.ts`.
    const second = channel();
    session.attach(bindEventTargetPort(second.host as never));

    // The old subscription is gone from the download manager, which outlives the
    // window. Without this, every reload adds one listener that nothing can ever
    // remove.
    expect(runtime.released).toEqual([true]);
    expect(session.generation()).toBe(2);

    // …and the old channel is genuinely dead, not merely unsubscribed.
    const stale = drain(first.peer);
    first.peer.postMessage({ kind: 'call', id: 2, call: { method: 'GET', path: '/api/models' } });
    expect(await stale).toEqual([]);

    // The new one works.
    const reply = await ask(second.peer, { kind: 'call', id: 3, call: { method: 'GET', path: '/api/sessions' } });
    expect(reply).toMatchObject({ kind: 'response', id: 3 });
  });

  it('keeps a replaced port alive until its in-flight call has an authoritative result', async () => {
    const runtime = stubRuntime();
    let oldSignal: AbortSignal | undefined;
    let finishOld: ((response: ApiResponse) => void) | undefined;
    runtime.call = (call: ApiCall, signal?: AbortSignal): Promise<ApiResponse> => {
      runtime.calls.push(call);
      if (call.path === '/api/sessions/already-started') {
        oldSignal = signal;
        return new Promise<ApiResponse>((resolve) => {
          finishOld = resolve;
        });
      }
      return Promise.resolve({ ok: true, status: 200, body: { path: call.path } });
    };

    const session = createControlPanelSession({ runtime });
    const first = channel();
    session.attach(bindEventTargetPort(first.host as never));
    const oldReply = ask(first.peer, {
      kind: 'call',
      id: 1,
      call: { method: 'DELETE', path: '/api/sessions/already-started' },
    });
    await settle();

    const second = channel();
    session.attach(bindEventTargetPort(second.host as never));

    expect(oldSignal?.aborted).toBe(true);
    expect(await Promise.race([oldReply.then(() => 'settled'), settle().then(() => 'pending')])).toBe('pending');

    // Replacing the connection must not hold up unrelated work on the new one.
    expect(
      await ask(second.peer, { kind: 'call', id: 2, call: { method: 'GET', path: '/api/models' } }),
    ).toMatchObject({ kind: 'response', id: 2 });

    // Cancellation lost the race to a mutation that had already started. The
    // retired connection stays alive long enough to report that real result.
    finishOld!({ ok: true, status: 204, body: null });
    expect(await oldReply).toEqual({
      kind: 'response',
      id: 1,
      response: { ok: true, status: 204, body: null },
    });
  });

  it('stops delivering events to a replaced port', async () => {
    const runtime = stubRuntime();
    const session = createControlPanelSession({ runtime });

    const first = channel();
    session.attach(bindEventTargetPort(first.host as never));
    first.peer.postMessage({ kind: 'subscribe', id: 5, jobId: 'job-a' });
    await settle();

    const second = channel();
    session.attach(bindEventTargetPort(second.host as never));

    const stale = drain(first.peer);
    // Progress on a job the old page subscribed to. It must reach nobody: the
    // page is gone, and a `postMessage` into a closed port is silently dropped
    // rather than throwing (see `RpcPort.postMessage`), so this is the only way
    // the leak is observable.
    expect(() => runtime.emit(0, { type: 'progress' } as unknown as DownloadEvent)).not.toThrow();
    expect(await stale).toEqual([]);
  });
});

describe('close', () => {
  it('releases the port and closes the runtime, once', async () => {
    const runtime = stubRuntime();
    const session = createControlPanelSession({ runtime });
    const { host, peer } = channel();
    session.attach(bindEventTargetPort(host as never));
    peer.postMessage({ kind: 'subscribe', id: 1, jobId: 'job-a' });
    await settle();

    await session.close();
    await session.close();

    expect(runtime.closed).toBe(1);
    expect(runtime.released).toEqual([true]);
  });

  it('refuses a port that arrives during shutdown', async () => {
    const runtime = stubRuntime();
    const session = createControlPanelSession({ runtime });
    const closing = session.close();

    const { host, peer } = channel();
    // MAIN could broker a window that opened as the app was quitting. Serving it
    // would register subscriptions on a runtime that is being drained and answer
    // calls against a database that is closing underneath them.
    session.attach(bindEventTargetPort(host as never));
    const stale = drain(peer);
    peer.postMessage({ kind: 'call', id: 1, call: { method: 'GET', path: '/api/models' } });

    expect(await stale).toEqual([]);
    expect(session.generation()).toBe(0);
    await closing;
  });

  it('survives a serve whose dispose throws', async () => {
    const runtime = stubRuntime();
    const session = createControlPanelSession({
      runtime,
      serve: () => () => {
        throw new Error('port already gone');
      },
    });
    const { host } = channel();
    session.attach(bindEventTargetPort(host as never));

    // A dispose that throws must not stop the next port from being installed,
    // nor take the process down with it — an unhandled throw in CONTROL PANEL is exactly
    // what crash isolation is meant to contain, not to cause.
    expect(() => session.attach(bindEventTargetPort(channel().host as never))).not.toThrow();
    await expect(session.close()).resolves.toBeUndefined();
    expect(runtime.closed).toBe(1);
  });
});

const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 20));
