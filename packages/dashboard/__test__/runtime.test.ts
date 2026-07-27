/**
 * The transport-independent runtime: bootstrap, `call`, and shutdown ordering.
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { createDashboardRuntime, type DashboardRuntime } from '../src/runtime.js';

let base: string;
let sessionsRoot: string;
let tracesDir: string;
let modelsDir: string;
let runtime: DashboardRuntime;

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

beforeEach(() => {
  base = mkdtempSync(join(tmpdir(), 'dash-runtime-'));
  sessionsRoot = join(base, 'sessions');
  seedSessions(sessionsRoot, SESSION_COUNT);
  tracesDir = join(base, 'traces');
  mkdirSync(tracesDir, { recursive: true });
  modelsDir = join(base, 'models');
  mkdirSync(modelsDir, { recursive: true });

  runtime = createDashboardRuntime({ dbPath: ':memory:', sessionsRoot, tracesDir, modelsDir });
});

afterEach(async () => {
  await runtime.close();
  rmSync(base, { recursive: true, force: true });
});

describe('dashboard runtime — call', () => {
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

describe('dashboard runtime — shutdown', () => {
  // `clearInterval` only stops FUTURE rescans. A tick already in flight kept
  // running against the database while `close()` shut it, and the failure was
  // invisible because `doIngest`'s catch swallowed "database is not open" into a
  // warning nobody read. `close()` must await the ingest chain instead.
  it('awaits an in-flight ingest before closing the database', async () => {
    // `ingestSessions`/`ingestTraces` have synchronous bodies, so ONE queued
    // rescan can finish inside the handful of microtasks `downloads.shutdown()`
    // costs — a single-ingest assertion would pass even with the await removed.
    // Queue a deep chain instead: it takes far more turns than the rest of the
    // shutdown, so only a `close()` that actually awaits the chain can observe
    // it settled.
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
    // "database is not open", and every seeded session was indexed.
    expect(summary.sessions.warnings).toEqual([]);
    expect(summary.sessions.scanned).toBe(SESSION_COUNT);
  });

  // Downloads must drain BEFORE anything else tears down: a shutdown that killed
  // the process mid-write would orphan a partial, potentially multi-GB `.staging`
  // tree, and a job publishing after the database closed would write into a dead
  // handle. The database is the LAST thing to go.
  it('drains downloads before it closes the database', async () => {
    const order: string[] = [];
    const realShutdown = runtime.context.downloads.shutdown.bind(runtime.context.downloads);
    runtime.context.downloads.shutdown = async () => {
      order.push('downloads.shutdown');
      await realShutdown();
    };
    const realClose = runtime.context.dash.close.bind(runtime.context.dash);
    runtime.context.dash.close = () => {
      order.push('dash.close');
      realClose();
    };

    await runtime.close();
    expect(order).toEqual(['downloads.shutdown', 'dash.close']);
  });

  it('is idempotent: a second close (and a drain after close) is a no-op', async () => {
    await runtime.drain();
    await runtime.close();
    await runtime.close();
    await runtime.drain();
  });
});
