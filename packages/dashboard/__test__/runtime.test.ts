/**
 * The transport-independent runtime: bootstrap, `call`, the thread split, and
 * shutdown ordering.
 */

import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { MessageChannel } from 'node:worker_threads';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import type { MainApiContext, WorkerApiContext } from '../src/api/context.js';
import { dispatchMain, dispatchWorker, routeThreadFor } from '../src/api/dispatch.js';
import { ROUTES } from '../src/api/routes.js';
import { createRpcClient } from '../src/rpc/client.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import { createDashboardRuntime, type DashboardRuntime, type RuntimeLifecycleEvent } from '../src/runtime.js';
import { startDbWorker } from '../src/worker/client.js';

let base: string;
let sessionsRoot: string;
let tracesDir: string;
let modelsDir: string;
let runtime: DashboardRuntime;
/**
 * Shutdown witnesses in the order they happened. The database now lives on the
 * worker, so its close is observed through the lifecycle notification the worker
 * posts AFTER closing the handle; main-thread steps push into the same array.
 */
let order: string[];

/** Enough real session files that an ingest spans several I/O turns. */
const SESSION_COUNT = 120;

function seedSessions(root: string, count: number): void {
  const dir = join(root, '--w--');
  mkdirSync(dir, { recursive: true });
  for (let i = 0; i < count; i++) {
    const id = `rt-${i}`;
    const lines = [
      { type: 'session', version: 3, id, timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: `question ${i}` },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: `answer ${i}` }],
          model: 'qwen3_5',
          usage: { input: 5, output: 6 },
        },
      },
    ];
    writeFileSync(join(dir, `2026-07-01T10-00-00_${id}.jsonl`), `${lines.map((l) => JSON.stringify(l)).join('\n')}\n`);
  }
}

/** Settle `promise`, or resolve to `'HUNG'` — a hang must fail fast, not at the 2 min test timeout. */
async function withDeadline<T>(promise: Promise<T>, ms: number): Promise<T | 'HUNG'> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<'HUNG'>((resolve) => {
    timer = setTimeout(() => resolve('HUNG'), ms);
  });
  try {
    return await Promise.race([promise, deadline]);
  } finally {
    clearTimeout(timer);
  }
}

beforeEach(() => {
  base = mkdtempSync(join(tmpdir(), 'dash-runtime-'));
  sessionsRoot = join(base, 'sessions');
  seedSessions(sessionsRoot, SESSION_COUNT);
  tracesDir = join(base, 'traces');
  mkdirSync(tracesDir, { recursive: true });
  modelsDir = join(base, 'models');
  mkdirSync(modelsDir, { recursive: true });

  order = [];
  runtime = createDashboardRuntime({
    dbPath: ':memory:',
    sessionsRoot,
    tracesDir,
    modelsDir,
    onLifecycle: (event: RuntimeLifecycleEvent) => order.push(event.type),
  });
});

afterEach(async () => {
  await runtime.close();
  rmSync(base, { recursive: true, force: true });
});

describe('dashboard runtime — call', () => {
  /**
   * `call` documents that it never rejects, and route matching decodes each
   * `:param` with `decodeURIComponent` — which throws on a malformed escape.
   * The ownership check that triggers that decode used to run BEFORE the try,
   * so a caller got a bare `URIError` instead of the envelope everything else
   * produces.
   *
   * Both threads are asserted deliberately: fixing only the main-thread branch
   * leaves the worker-owned leg rejecting, and this suite has a habit of going
   * green for the wrong reason. So is the `%25` control — a well-formed escape
   * must stay an ordinary miss, or a blanket "reject anything with a %" would
   * pass the first two assertions.
   */
  it('returns a failure envelope, never a rejection, for a malformed percent-escape', async () => {
    // Worker-owned route.
    await expect(runtime.call({ method: 'GET', path: '/api/sessions/%' })).resolves.toMatchObject({
      ok: false,
      code: 'E_BAD_REQUEST',
      status: 400,
    });
    // Main-owned route, same shape.
    await expect(runtime.call({ method: 'DELETE', path: '/api/downloads/%' })).resolves.toMatchObject({
      ok: false,
      code: 'E_BAD_REQUEST',
      status: 400,
    });
    // Control: `%25` decodes cleanly to `%`, so this is a plain miss.
    await expect(runtime.call({ method: 'GET', path: '/api/sessions/%25' })).resolves.toMatchObject({
      ok: false,
      code: 'E_NOT_FOUND',
      status: 404,
    });
  });

  it('answers an API call with the route body, no socket involved', async () => {
    const res = await runtime.call({ method: 'GET', path: '/api/health' });
    expect(res.status).toBe(200);
    expect(res.ok ? res.body : null).toMatchObject({ status: 'ok', sessionsRoot, tracesDir, modelsDir });
  });

  it('parses the query string off the path', async () => {
    await runtime.ingestNow();
    const res = await runtime.call({ method: 'GET', path: '/api/sessions?limit=3' });
    expect(res.ok).toBe(true);
    const body = res.ok ? (res.body as { sessions: unknown[]; total: number }) : { sessions: [], total: 0 };
    expect(body.sessions).toHaveLength(3);
    expect(body.total).toBe(SESSION_COUNT);
  });

  it('returns a failure envelope (never rejects) for an unknown path', async () => {
    const res = await runtime.call({ method: 'GET', path: '/not-an-api-path' });
    expect(res.ok).toBe(false);
    expect(res.status).toBe(404);
    expect(res.ok ? '' : res.code).toBe('E_NOT_FOUND');
  });

  it('refuses the SSE route over a transport that cannot stream', async () => {
    const res = await runtime.call({ method: 'GET', path: '/api/downloads/job-1/events' });
    expect(res.status).toBe(503);
    expect(res.ok ? '' : res.code).toBe('E_UNAVAILABLE');
  });
});

// The whole point of the split: a synchronous query on the worker must not stall
// the thread that owns the transport, the download manager and its progress
// events. Asserting SETTLE ORDER rather than a duration keeps it deterministic —
// a transport-thread route is answered in microtasks, so it cannot lose to a
// cross-thread round trip, let alone one queued behind a 120-session ingest.
describe('dashboard runtime — thread split', () => {
  it('answers a transport-thread route while the worker is busy ingesting', async () => {
    const settled: string[] = [];
    const heavy = runtime.call({ method: 'POST', path: '/api/ingest' }).then((res) => {
      settled.push('worker');
      return res;
    });
    const light = runtime.call({ method: 'GET', path: '/api/downloads' }).then((res) => {
      settled.push('main');
      return res;
    });

    const [heavyRes, lightRes] = await Promise.all([heavy, light]);
    expect(lightRes.status).toBe(200);
    expect(heavyRes.status).toBe(200);
    expect(settled).toEqual(['main', 'worker']);
  });

  it('answers a transport-thread route before the worker has even booted', async () => {
    // A cancel click during startup must not wait for the database to open.
    const fresh = createDashboardRuntime({ dbPath: ':memory:', sessionsRoot, tracesDir, modelsDir });
    try {
      const settled: string[] = [];
      const boot = fresh.call({ method: 'GET', path: '/api/health' }).then(() => settled.push('worker'));
      const cancel = fresh.call({ method: 'DELETE', path: '/api/downloads/ghost' }).then(() => settled.push('main'));
      await Promise.all([boot, cancel]);
      expect(settled).toEqual(['main', 'worker']);
    } finally {
      await fresh.close();
    }
  });
});

// Ownership is data on the route, so it can be asserted as data. A route added
// later lands on the thread its table entry names — this locks WHICH entries name
// the transport thread, since everything else is the worker by construction.
describe('dashboard runtime — route ownership', () => {
  it('keeps exactly the download routes on the transport thread', () => {
    const main = ROUTES.filter((r) => r.thread === 'main').map((r) => `${r.method} /${r.segments.join('/')}`);
    expect(main).toEqual([
      'GET /api/downloads',
      'POST /api/downloads',
      'DELETE /api/downloads/:id',
      'GET /api/downloads/:id/events',
    ]);
  });

  it('reports the owning thread per request, and none for an unmatched one', () => {
    expect(routeThreadFor('GET', '/api/sessions')).toBe('worker');
    expect(routeThreadFor('DELETE', '/api/downloads/job-1')).toBe('main');
    // Wrong method on a known path, and an unknown path: no owner, answered
    // wherever the request landed (405/404 needs no context).
    expect(routeThreadFor('PUT', '/api/models')).toBe(null);
    expect(routeThreadFor('GET', '/api/nope')).toBe(null);
  });

  it('refuses to run a route the other thread owns', async () => {
    const mainCtx = { modelsDir, sessionsRoot, tracesDir, cacheRoot: undefined } as unknown as MainApiContext;
    const workerCtx = { modelsDir, sessionsRoot, tracesDir, cacheRoot: undefined } as unknown as WorkerApiContext;
    const query = new URLSearchParams();

    const dbOnMain = await dispatchMain(mainCtx, { method: 'GET', pathname: '/api/sessions', query });
    expect(dbOnMain.status).toBe(500);
    expect(dbOnMain.ok ? '' : dbOnMain.message).toContain('owned by the worker thread');

    const downloadOnWorker = await dispatchWorker(workerCtx, { method: 'GET', pathname: '/api/downloads', query });
    expect(downloadOnWorker.status).toBe(500);
    expect(downloadOnWorker.ok ? '' : downloadOnWorker.message).toContain('owned by the main thread');
  });
});

describe('dashboard runtime — shutdown', () => {
  // `clearInterval` only stops FUTURE rescans. A tick already in flight kept
  // running against the database while `close()` shut it, and the failure was
  // invisible because the ingest catch swallowed "database is not open" into a
  // warning nobody read. `close()` must await the ingest chain instead.
  it('awaits an in-flight ingest before closing the database', async () => {
    // `ingestSessions`/`ingestTraces` have synchronous bodies, so ONE queued
    // rescan can finish inside the handful of turns `downloads.shutdown()` costs
    // — a single-ingest assertion would pass even with the await removed. Queue a
    // deep chain instead: it takes far more turns than the rest of the shutdown,
    // so only a `close()` that actually awaits the chain can observe it settled.
    const QUEUED = 40;
    let pending = runtime.ingestNow();
    for (let i = 1; i < QUEUED; i++) pending = runtime.ingestNow();

    let settled = false;
    const tail = pending.then((s) => {
      settled = true;
      return s;
    });
    // Guard the guard: if the chain had already drained, the assertion below
    // would pass for the wrong reason.
    expect(settled).toBe(false);

    await runtime.close();

    expect(settled).toBe(true);
    const summary = await tail;
    // It ran to completion against an OPEN database: no swallowed
    // "database is not open", no worker torn down under it, and every seeded
    // session was indexed.
    expect(summary.sessions.warnings).toEqual([]);
    expect(summary.sessions.scanned).toBe(SESSION_COUNT);
  });

  // Downloads must drain BEFORE anything else tears down: a shutdown that killed
  // the process mid-write would orphan a partial, potentially multi-GB `.staging`
  // tree, and a job publishing after the database closed would write into a dead
  // handle. The database is the LAST thing to go — and it now goes on another
  // thread, so the witness is the worker's own post-close notification.
  it('drains downloads before it closes the database', async () => {
    const realShutdown = runtime.context.downloads.shutdown.bind(runtime.context.downloads);
    runtime.context.downloads.shutdown = async () => {
      order.push('downloads.shutdown');
      await realShutdown();
    };

    await runtime.close();
    expect(order).toEqual(['downloads.shutdown', 'db-closed']);
  });

  it('is idempotent: a second close (and a drain after close) is a no-op', async () => {
    await runtime.drain();
    await runtime.close();
    await runtime.close();
    await runtime.drain();
    // One database close, not one per call.
    expect(order).toEqual(['db-closed']);
  });
});

// A dead worker must never leave a caller waiting: the control panel UI would show a
// spinner forever and the supervisor's per-RPC timeouts would be the only thing
// that noticed. Every unanswerable call comes back as a failure ENVELOPE.
describe('dashboard runtime — worker death', () => {
  it('settles an in-flight call when the worker dies under it', async () => {
    const crashed = createDashboardRuntime({
      dbPath: ':memory:',
      sessionsRoot,
      tracesDir,
      modelsDir,
      workerUrl: new URL('./helpers/crashing-db-worker.ts', import.meta.url),
      onLifecycle: (event) => order.push(event.type),
    });
    try {
      const inFlight = await withDeadline(crashed.call({ method: 'GET', path: '/api/sessions' }), 10_000);
      expect(inFlight).not.toBe('HUNG');
      const res = inFlight as Exclude<typeof inFlight, 'HUNG'>;
      expect(res.ok).toBe(false);
      expect(res.ok ? '' : res.code).toBe('E_UNAVAILABLE');

      // And a call issued after the death fails fast rather than queueing onto a
      // thread that will never read it.
      const after = await withDeadline(crashed.call({ method: 'GET', path: '/api/models' }), 10_000);
      expect(after).not.toBe('HUNG');
      expect((after as Exclude<typeof after, 'HUNG'>).status).toBe(503);

      expect(order).toContain('worker-down');
      // A transport-thread route is untouched by the worker's death.
      const downloads = await crashed.call({ method: 'GET', path: '/api/downloads' });
      expect(downloads.status).toBe(200);
    } finally {
      expect(await withDeadline(crashed.close(), 10_000)).not.toBe('HUNG');
    }
  });

  it('rejects an unconfirmed drain and still closes on a bounded timeout', async () => {
    const wedged = createDashboardRuntime({
      dbPath: ':memory:',
      sessionsRoot,
      tracesDir,
      modelsDir,
      workerUrl: new URL('./helpers/wedged-db-worker.ts', import.meta.url),
      shutdownTimeoutMs: 200,
    });
    // The public drain promise must reject rather than claim the still-live
    // worker reached its ingest fixpoint. This is the exact contract an
    // in-process caller relies on before proceeding with its own shutdown.
    const drained = await withDeadline(
      wedged.drain().then(
        () => 'RESOLVED' as const,
        (error: unknown) => error,
      ),
      1_000,
    );
    expect(drained).not.toBe('HUNG');
    expect(drained).not.toBe('RESOLVED');
    expect(drained).toBeInstanceOf(Error);
    expect((drained as Error).message).toContain('did not confirm drain');
    expect((drained as Error).message).toContain('timed out after 200ms');

    // `close()` reuses that rejected idempotent drain, but must still execute the
    // worker's bounded close/terminate path. Generous cap, far below the point
    // where "it hung" is indistinguishable from "it was slow".
    const close = await withDeadline(
      wedged.close().then(
        () => 'RESOLVED' as const,
        (error: unknown) => error,
      ),
      10_000,
    );
    expect(close).not.toBe('HUNG');
    expect(close).not.toBe('RESOLVED');
    // No cleanup failure: close preserves the original barrier error exactly.
    expect(close).toBe(drained);

    // `DashboardRuntime.close()` must still run the worker's bounded close path
    // after the truthful drain failure. A later worker-owned call therefore
    // fails fast rather than reaching the silent thread or waiting 60 seconds.
    const after = await withDeadline(wedged.call({ method: 'GET', path: '/api/models' }), 1_000);
    expect(after).not.toBe('HUNG');
    expect((after as Exclude<typeof after, 'HUNG'>).status).toBe(503);
  });

  // The other half of "wedged": the thread stays ALIVE and simply never answers,
  // so nothing emits `error` or `exit` and `goDown` never runs. Bounding only the
  // shutdown steps left every ORDINARY request pending forever — the caller's
  // promise never settled, its `pending` entry was never removed, and each one
  // held `worker.ref()`, so a 30s periodic ingest accumulated one leak per tick
  // and kept the process alive on a worker that would never reply.
  //
  // Driven against `startDbWorker` rather than the runtime because the deadline
  // being asserted is the client's, and the runtime does not surface it.
  it('terminates a silent worker before settling an unacknowledged call deadline', async () => {
    const lifecycle: RuntimeLifecycleEvent[] = [];
    const client = startDbWorker({
      workerUrl: new URL('./helpers/wedged-db-worker.ts', import.meta.url),
      shutdownTimeoutMs: 200,
      requestTimeoutMs: 150,
      onLifecycle: (event) => lifecycle.push(event),
      dbPath: ':memory:',
      modelsDir,
      sessionsRoot,
      tracesDir,
      cacheRoot: undefined,
    });

    try {
      const call = await withDeadline(client.call({ method: 'GET', path: '/api/sessions' }), 5_000);
      expect(call).not.toBe('HUNG');
      const res = call as Exclude<typeof call, 'HUNG'>;
      expect(res.status).toBe(503);
      expect(res.ok ? '' : res.code).toBe('E_UNAVAILABLE');

      // The periodic rescan is the leak's engine, so it needs the same bound.
      const ingest = await withDeadline(client.ingest(), 5_000);
      expect(ingest).not.toBe('HUNG');
      expect((ingest as Exclude<typeof ingest, 'HUNG'>).sessions.warnings).toHaveLength(1);

      // No cancellation acknowledgement and no real response means the worker
      // cannot remain alive after the caller is failed: it could otherwise
      // execute the queued mutation later.
      const down = lifecycle.find((event) => event.type === 'worker-down');
      expect(down).toMatchObject({ type: 'worker-down' });
      expect(down?.type === 'worker-down' ? down.reason : '').toContain('cancellation outcome');
      expect(down?.type === 'worker-down' ? down.reason : '').toContain('remained unknown');
      expect(await withDeadline(client.call({ method: 'GET', path: '/api/models' }), 5_000)).not.toBe('HUNG');
    } finally {
      expect(await withDeadline(client.close(), 10_000)).not.toBe('HUNG');
    }
  });
});

/**
 * A deadline starts cancellation, but the request was posted the moment `send`
 * ran and may be sitting in the worker's message queue. Without an acknowledged
 * retraction, a timed-out `DELETE` answered the UI `503 E_UNAVAILABLE` and then
 * performed the deletion anyway — the user is told nothing happened, retries or
 * gives up, and the model / session / cache entry is gone regardless.
 *
 * Asserted on the on-disk transcript rather than a database row: it is the part
 * of the mutation that outlives the worker, and it is visible from this thread
 * without asking the worker anything.
 */
describe('dashboard database worker — withdrawing an abandoned request', () => {
  const transcriptOf = (id: string): string => join(sessionsRoot, '--w--', `2026-07-01T10-00-00_${id}.jsonl`);

  function client(requestTimeoutMs: number): ReturnType<typeof startDbWorker> {
    return startDbWorker({
      // The SOURCE entry, not `defaultWorkerUrl()`: the check under test lives
      // in `db-worker.ts`, and a stale `dist` beside it would test nothing.
      workerUrl: new URL('../src/worker/ts-bootstrap.ts', import.meta.url),
      shutdownTimeoutMs: 10_000,
      requestTimeoutMs,
      onLifecycle: (): void => {},
      dbPath: ':memory:',
      modelsDir,
      sessionsRoot,
      tracesDir,
      cacheRoot: undefined,
    });
  }

  it('does not delete a session whose caller was already told the call failed', async () => {
    const doomed = transcriptOf('rt-0');
    expect(existsSync(doomed)).toBe(true);

    // 20 ms cannot survive worker startup, so the deadline is guaranteed to
    // expire while the request is still queued — the exact window the
    // withdrawal covers.
    const timing = client(20);
    const res = await withDeadline(timing.call({ method: 'DELETE', path: '/api/sessions/rt-0' }), 10_000);
    expect(res).not.toBe('HUNG');
    const failed = res as Exclude<typeof res, 'HUNG'>;
    expect(failed.status).toBe(503);
    expect(failed.ok ? '' : failed.code).toBe('E_UNAVAILABLE');

    // `close` is posted AFTER the DELETE and the worker answers in order, so
    // once it resolves the DELETE has provably been reached. No sleep, and no
    // way for the assertion below to pass merely by running early.
    await withDeadline(timing.close(), 10_000);
    expect(existsSync(doomed)).toBe(true);
  });

  it('still deletes when the caller waits for the answer', async () => {
    // The control. Without it the test above would pass just as well against a
    // DELETE route that had stopped working altogether.
    const doomed = transcriptOf('rt-1');
    expect(existsSync(doomed)).toBe(true);

    const patient = client(30_000);
    const res = await withDeadline(patient.call({ method: 'DELETE', path: '/api/sessions/rt-1' }), 10_000);
    expect(res).not.toBe('HUNG');
    expect((res as Exclude<typeof res, 'HUNG'>).status).toBe(200);

    await withDeadline(patient.close(), 10_000);
    expect(existsSync(doomed)).toBe(false);
  });

  it('does not post a pre-aborted call or poison the next worker request id', async () => {
    const spared = transcriptOf('rt-3');
    const control = transcriptOf('rt-4');
    expect(existsSync(spared)).toBe(true);
    expect(existsSync(control)).toBe(true);

    const worker = client(30_000);
    const controller = new AbortController();
    controller.abort();
    const cancelled = await withDeadline(
      worker.call({ method: 'DELETE', path: '/api/sessions/rt-3' }, controller.signal),
      10_000,
    );
    expect(cancelled).not.toBe('HUNG');
    const failed = cancelled as Exclude<typeof cancelled, 'HUNG'>;
    expect(failed.status).toBe(503);
    expect(failed.ok ? '' : failed.code).toBe('E_UNAVAILABLE');

    // A normal call immediately after the pre-aborted one proves both that the
    // first was never posted and that cancellation did not occupy/corrupt a
    // request id in the worker's ordered protocol.
    const deleted = await withDeadline(worker.call({ method: 'DELETE', path: '/api/sessions/rt-4' }), 10_000);
    expect(deleted).not.toBe('HUNG');
    expect((deleted as Exclude<typeof deleted, 'HUNG'>).status).toBe(200);

    await withDeadline(worker.close(), 10_000);
    expect(existsSync(spared)).toBe(true);
    expect(existsSync(control)).toBe(false);
  });

  it('returns the real result when the worker sampled just before withdrawal arrived', async () => {
    const target = join(sessionsRoot, 'late-withdraw-target');
    writeFileSync(target, 'alive');
    const gate = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT * 2));
    const controller = new AbortController();

    const worker = startDbWorker({
      workerUrl: new URL('./helpers/late-withdraw-db-worker.ts', import.meta.url),
      // This is only a hang backstop. The explicit abort below is the signal the
      // renderer's 500 ms deadline drives through the RPC host, without making
      // this worker-level race test depend on timer scheduling.
      requestTimeoutMs: 30_000,
      shutdownTimeoutMs: 2_000,
      onLifecycle: (): void => {},
      dbPath: ':memory:',
      modelsDir,
      sessionsRoot,
      tracesDir,
      cacheRoot: undefined,
    });

    try {
      const result = worker.call({ method: 'DELETE', path: '/race', body: gate.buffer }, controller.signal);
      const waitForGateChange = async (expected: number): Promise<number> => {
        const waiting = Atomics.waitAsync(gate, 0, expected);
        if (waiting.async) await waiting.value;
        return Atomics.load(gate, 0);
      };

      // Causal ordering, not elapsed-time luck:
      //   worker sampled empty → renderer cancellation crosses withdrawal port
      //   → parent releases worker → mutation returns its real result.
      expect(await withDeadline(waitForGateChange(0), 5_000)).toBe(1);
      // Abort dispatch is synchronous in the worker client: by the time this
      // returns, the withdrawal has been posted on the independent port.
      controller.abort();
      expect(await withDeadline(waitForGateChange(1), 5_000)).toBe(2);
      Atomics.store(gate, 1, 1);
      Atomics.notify(gate, 1);

      const settled = await withDeadline(result, 5_000);
      expect(settled).not.toBe('HUNG');
      expect(settled).toEqual({
        ok: true,
        status: 200,
        body: { mutated: true, withdrawalArrivedAfterSample: true },
      });
      expect(existsSync(target)).toBe(false);
    } finally {
      // Always unblock the worker if an assertion above failed while it was at
      // either gate before closing it. Abort first so its evidence loop can reach
      // state 2 even when the assertion failed before the normal abort.
      controller.abort();
      Atomics.store(gate, 1, 1);
      Atomics.notify(gate, 1);
      expect(await withDeadline(worker.close(), 10_000)).not.toBe('HUNG');
    }
  });

  it('does not execute a deletion after the renderer RPC deadline reports failure', async () => {
    const doomed = transcriptOf('rt-2');
    expect(existsSync(doomed)).toBe(true);

    const { port1, port2 } = new MessageChannel();
    port1.unref();
    port2.unref();
    const dispose = serveRuntimeOverPort(runtime, bindEventTargetPort(port2));
    // The call is posted before this zero-delay timer can run, but the timer
    // expires before the fresh worker can finish its boot ingest and reach the
    // queued DELETE. This is the renderer's shorter deadline racing the worker's
    // independent 60 s deadline — the production defect, end to end.
    const renderer = createRpcClient(bindEventTargetPort(port1), {
      timeoutMs: 0,
      onUnresponsive: () => port2.close(),
    });

    try {
      const res = await withDeadline(renderer.call({ method: 'DELETE', path: '/api/sessions/rt-2' }), 10_000);
      expect(res).not.toBe('HUNG');
      const failed = res as Exclude<typeof res, 'HUNG'>;
      expect(failed.status).toBe(503);
      expect(failed.ok ? '' : failed.code).toBe('E_UNAVAILABLE');

      // The failure itself is now the worker's acknowledgement that its exact
      // pre-handler gate skipped the mutation; no event-loop delay or
      // cross-channel ordering assumption is part of the proof.
      expect(existsSync(doomed)).toBe(true);
      await runtime.close();
    } finally {
      renderer.close();
      dispose();
    }
  });
});
