/**
 * The SPA's client (`ui/src/lib/api.ts`) driven over a real port pair.
 *
 * What matters here is the boundary the pages actually see: `getJson`/`mutate`
 * hand back the body, and a failure envelope becomes a thrown `DashboardApiError`
 * that still carries the runtime's `code` and its derived `status`. Every page
 * renders `err.message`; the cache and models pages branch on the rest.
 */

import { MessageChannel } from 'node:worker_threads';

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { failure, type ApiResponse } from '../src/api/errors.js';
import type { DownloadEvent } from '../src/download.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort, type RpcPort, type RpcPortEvents } from '../src/rpc/port.js';
import type { ApiCall } from '../src/runtime.js';
import {
  connectDashboardApi,
  DashboardApiError,
  disconnectDashboardApi,
  getJson,
  mutate,
  subscribeDownload,
} from '../ui/src/lib/api.js';

let teardown: (() => void) | null = null;

afterEach(() => {
  disconnectDashboardApi();
  teardown?.();
  teardown = null;
});

async function flush(turns = 4): Promise<void> {
  for (let i = 0; i < turns; i++) await new Promise((r) => setTimeout(r, 0));
}

interface Harness {
  calls: ApiCall[];
  emit(jobId: string, event: DownloadEvent): void;
  liveSubscriptions(): number;
}

/** Connect the SPA client to a stub runtime over a real MessageChannel. */
function connect(answer?: (call: ApiCall) => ApiResponse): Harness {
  const calls: ApiCall[] = [];
  const subscribers = new Map<number, { jobId: string; listener: (event: DownloadEvent) => void }>();
  let nextKey = 1;

  const { port1, port2 } = new MessageChannel();
  port1.unref();
  port2.unref();
  const dispose = serveRuntimeOverPort(
    {
      call: async (call: ApiCall) => {
        calls.push(call);
        return answer?.(call) ?? { ok: true, status: 200, body: { echo: call.path } };
      },
      subscribe: (jobId: string, listener: (event: DownloadEvent) => void) => {
        const key = nextKey++;
        subscribers.set(key, { jobId, listener });
        return () => {
          subscribers.delete(key);
        };
      },
    },
    bindEventTargetPort(port2),
  );
  connectDashboardApi(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });
  teardown = dispose;

  return {
    calls,
    emit(jobId, event) {
      for (const s of subscribers.values()) if (s.jobId === jobId) s.listener(event);
    },
    liveSubscriptions: () => subscribers.size,
  };
}

describe('SPA api client', () => {
  it('returns the body of a success envelope and prefixes the path with /api', async () => {
    const h = connect();
    await expect(getJson('/models')).resolves.toEqual({ echo: '/api/models' });
    expect(h.calls[0]).toEqual({ method: 'GET', path: '/api/models' });
  });

  it('does not double-prefix an already-qualified path', async () => {
    const h = connect();
    await getJson('/api/sessions?limit=1');
    expect(h.calls[0].path).toBe('/api/sessions?limit=1');
  });

  it('sends the method and body a mutation carries', async () => {
    const h = connect();
    await mutate('PATCH', '/sessions/fix-1', { name: 'Renamed' });
    expect(h.calls[0]).toEqual({ method: 'PATCH', path: '/api/sessions/fix-1', body: { name: 'Renamed' } });
  });

  it('rethrows a failure envelope as a DashboardApiError keeping code, status and message', async () => {
    connect(() => failure('E_CONFLICT', 'Session is being written'));

    const err = await getJson('/sessions/fix-1').then(
      () => null,
      (e: unknown) => e,
    );
    expect(err).toBeInstanceOf(DashboardApiError);
    const api = err as DashboardApiError;
    // `code` is the discriminator the runtime raised; `status` is what it derives
    // to. Losing either turns every failure into an indistinguishable red toast.
    expect(api.code).toBe('E_CONFLICT');
    expect(api.status).toBe(409);
    expect(api.message).toBe('Session is being written');
  });

  it('distinguishes a 404 from a 400 by status, not just by message', async () => {
    connect((call) =>
      call.path.includes('ghost') ? failure('E_NOT_FOUND', 'no such model') : failure('E_BAD_REQUEST', 'not a model'),
    );

    const notFound = (await mutate('DELETE', '/models/ghost').catch((e: DashboardApiError) => e)) as DashboardApiError;
    const badRequest = (await mutate('DELETE', '/models/other').catch(
      (e: DashboardApiError) => e,
    )) as DashboardApiError;
    expect([notFound.status, notFound.code]).toEqual([404, 'E_NOT_FOUND']);
    expect([badRequest.status, badRequest.code]).toEqual([400, 'E_BAD_REQUEST']);
  });

  it('fails fast with E_UNAVAILABLE when no port has been connected', async () => {
    // A page that renders before the preload hop must show an error, not hang on
    // a promise nothing will ever settle.
    disconnectDashboardApi();
    const err = (await getJson('/models').catch((e: unknown) => e)) as DashboardApiError;
    expect(err).toBeInstanceOf(DashboardApiError);
    expect(err.status).toBe(503);
    expect(err.code).toBe('E_UNAVAILABLE');
  });

  it('delivers download events and stops on close', async () => {
    const h = connect();
    const seen: DownloadEvent[] = [];
    const sub = subscribeDownload('job-1', (e) => seen.push(e));
    await flush();

    h.emit('job-1', { type: 'done', id: 'job-1', outputDir: '/models/job-1' });
    await flush();
    expect(seen).toHaveLength(1);

    sub.close();
    await flush();
    expect(h.liveSubscriptions()).toBe(0);
    h.emit('job-1', { type: 'error', id: 'job-1', message: 'late' });
    await flush();
    expect(seen).toHaveLength(1);
  });

  it('closes the previous connection when a new port is connected', async () => {
    const h = connect();
    subscribeDownload('job-1', () => {});
    await flush();
    expect(h.liveSubscriptions()).toBe(1);

    // A reconnect that left the old client attached would keep its subscriptions
    // registered on a runtime that outlives the window.
    const { port1, port2 } = new MessageChannel();
    port1.unref();
    port2.unref();
    connectDashboardApi(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });
    port2.close();
    await flush();
    expect(h.liveSubscriptions()).toBe(0);
  });

  it('does not let a retired connection restart the runtime serving its replacement', async () => {
    vi.useFakeTimers();
    try {
      let oldEvents: RpcPortEvents | undefined;
      const oldPort: RpcPort = {
        postMessage: () => {},
        listen(events): () => void {
          oldEvents = events;
          return () => {};
        },
        close: () => {},
      };
      let recoveries = 0;
      connectDashboardApi(oldPort, {
        timeoutMs: 60_000,
        cancellationGraceMs: 10,
        onUnresponsive: () => {
          recoveries += 1;
        },
      });
      const oldMutation = mutate('DELETE', '/sessions/old').catch((error: unknown) => error);

      // Reconnection asks the old client to cancel. If that old port is wedged,
      // its grace timer must not later kill the runtime behind this new port.
      connectDashboardApi(
        {
          postMessage: () => {},
          listen: () => () => {},
          close: () => {},
        },
        { onUnresponsive: () => {} },
      );
      await vi.advanceTimersByTimeAsync(10);
      expect(recoveries).toBe(0);

      oldEvents!.onClose();
      await oldMutation;
    } finally {
      vi.useRealTimers();
    }
  });
});
