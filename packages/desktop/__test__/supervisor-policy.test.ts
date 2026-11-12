/**
 * The supervisor's restart timing against Splash `RestartPolicy.after`, on a
 * fake clock: the real numbers (at once, then 5 s doubling, stop at the sixth
 * short failure, 60 s probation) take minutes of wall time with real children.
 * A fake transport stands in for the process and a stubbed `fetch` for
 * `/health`; real-process exits are covered in `supervisor.test.ts`.
 */

import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { createSupervisor, type Supervisor } from '../src/main/supervisor/index.js';
import type { ChildEvents, ChildTransport, SupervisorEvent } from '../src/main/supervisor/types.js';

// Any file that exists: the fake transport never runs it.
const ENTRY = fileURLToPath(import.meta.url);

interface FakeChild {
  events: ChildEvents;
  spawnedAtMs: number;
}

let dir = '';
let healthStatus = 'ok';
let supervisor: Supervisor | null = null;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
  dir = mkdtempSync(join(tmpdir(), 'mlx-supervisor-policy-'));
  healthStatus = 'ok';
  vi.stubGlobal('fetch', async () => new Response(JSON.stringify({ status: healthStatus }), { status: 200 }));
});

afterEach(async () => {
  const live = supervisor;
  supervisor = null;
  if (live !== null) {
    const disposing = live.dispose();
    await vi.advanceTimersByTimeAsync(60_000);
    await disposing;
  }
  vi.unstubAllGlobals();
  vi.useRealTimers();
  rmSync(dir, { recursive: true, force: true });
});

function harness(): { supervisor: Supervisor; children: FakeChild[]; events: SupervisorEvent[] } {
  const children: FakeChild[] = [];
  const transport: ChildTransport = (_spec, events) => {
    children.push({ events, spawnedAtMs: Date.now() });
    return {
      pid: 40_000 + children.length,
      stdout: null,
      stderr: null,
      postMessage: () => {},
      kill: () => queueMicrotask(() => events.onExit({ code: 0, signal: null })),
      forceKill: () => queueMicrotask(() => events.onExit({ code: null, signal: 'SIGKILL' })),
    };
  };
  const created = createSupervisor({
    entry: ENTRY,
    traceDir: join(dir, 'trace'),
    transport,
    baseEnv: {},
    timings: { readyTimeoutMs: 600_000, healthPollMs: 100, healthIntervalMs: 1_000 },
  });
  supervisor = created;
  const events: SupervisorEvent[] = [];
  created.on((event) => events.push(event));
  return { supervisor: created, children, events };
}

/** Lets real fs I/O (the trace watcher's last scan) complete between fake ticks. */
async function until(predicate: () => boolean, label: string): Promise<void> {
  for (let i = 0; i < 500; i += 1) {
    if (predicate()) return;
    await vi.advanceTimersByTimeAsync(0);
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  throw new Error(`never reached: ${label}`);
}

/** Advances the clock in 1 s steps (every delay here is a whole second) until `predicate`. */
async function advanceUntil(predicate: () => boolean, label: string, maxMs = 120_000): Promise<void> {
  await until(() => true, label);
  for (let waited = 0; !predicate(); waited += 1_000) {
    if (waited > maxMs) throw new Error(`never reached within ${maxMs} ms: ${label}`);
    await vi.advanceTimersByTimeAsync(1_000);
    await until(() => true, label);
  }
}

async function becomeReady(sup: Supervisor, child: FakeChild): Promise<void> {
  child.events.onMessage({ kind: 'mlx:ready', url: 'http://127.0.0.1:9', authToken: 'fake' });
  await until(() => sup.snapshot().state === 'running', 'running');
}

/** Gaps from each exit to the next spawn, over `failures` short-lived children. */
async function shortFailures(
  sup: Supervisor,
  children: FakeChild[],
  failures: number,
  serve: (child: FakeChild) => Promise<void>,
): Promise<number[]> {
  const gaps: number[] = [];
  for (let i = 0; i < failures; i += 1) {
    const child = children.at(-1);
    if (child === undefined) throw new Error('no child');
    await serve(child);
    const exitAt = Date.now();
    const before = children.length;
    child.events.onExit({ code: 1, signal: null });
    await until(() => sup.snapshot().lastExit?.atMs === exitAt, 'the exit verdict');
    if (sup.snapshot().state === 'failed') break;
    await advanceUntil(() => children.length > before, 'the relaunch');
    gaps.push((children.at(-1)?.spawnedAtMs ?? 0) - exitAt);
  }
  return gaps;
}

describe('restart timing (Splash RestartPolicy.after, real numbers)', () => {
  it('restarts at once, then after 5 s doubling, and stops at the sixth short failure', async () => {
    const { supervisor: sup, children, events } = harness();
    const starting = sup.start();
    starting.catch(() => {});

    const gaps = await shortFailures(sup, children, 6, async (child) => {
      await becomeReady(sup, child);
      await vi.advanceTimersByTimeAsync(1_000);
    });

    expect(gaps).toEqual([0, 5_000, 10_000, 20_000, 40_000]);
    expect(sup.snapshot()).toMatchObject({ state: 'failed', generation: 6, shortLivedFailures: 6 });
    expect(events.filter((event) => event.type === 'gave-up')).toEqual([
      {
        type: 'gave-up',
        message:
          'stopped restarting inference: it failed 6 times in a row, each within 60 s of starting (exited 1). Start inference again after fixing the cause.',
      },
    ]);
    // The first child's Ready resolved it; the give-up rejects only later waiters.
    await expect(starting).resolves.toBeUndefined();
  });

  // Splash's Ready follows the weights' load; probation is service after it.
  it('counts service from Ready, not from the spawn', async () => {
    const { supervisor: sup, children } = harness();
    sup.start().catch(() => {});

    const gaps = await shortFailures(sup, children, 6, async (child) => {
      healthStatus = 'loading';
      child.events.onMessage({ kind: 'mlx:ready', url: 'http://127.0.0.1:9', authToken: 'fake' });
      await vi.advanceTimersByTimeAsync(50_000);
      expect(sup.snapshot().state).toBe('starting');
      healthStatus = 'ok';
      await vi.advanceTimersByTimeAsync(100);
      await until(() => sup.snapshot().state === 'running', 'running');
      await vi.advanceTimersByTimeAsync(20_000);
    });

    expect(gaps).toEqual([0, 5_000, 10_000, 20_000, 40_000]);
    expect(sup.snapshot().state).toBe('failed');
  });

  // A lazy model load after Ready is the load Splash's start holds: not service.
  it('does not count a model load as service', async () => {
    const { supervisor: sup, children } = harness();
    sup.start().catch(() => {});

    const gaps = await shortFailures(sup, children, 6, async (child) => {
      await becomeReady(sup, child);
      healthStatus = 'loading';
      await vi.advanceTimersByTimeAsync(70_000);
      healthStatus = 'ok';
      await vi.advanceTimersByTimeAsync(2_000);
    });

    expect(gaps).toEqual([0, 5_000, 10_000, 20_000, 40_000]);
    expect(sup.snapshot().state).toBe('failed');
  });

  it('a user start or restart starts the count anew', async () => {
    const { supervisor: sup, children } = harness();
    sup.start().catch(() => {});
    const serve = async (child: FakeChild): Promise<void> => {
      await becomeReady(sup, child);
      await vi.advanceTimersByTimeAsync(1_000);
    };

    expect(await shortFailures(sup, children, 3, serve)).toEqual([0, 5_000, 10_000]);
    const restarting = sup.restart();
    await until(() => children.length === 5, 'the restart spawn');
    await becomeReady(sup, children[4]);
    await restarting;
    expect(sup.snapshot()).toMatchObject({ streak: 0, shortLivedFailures: 0 });
    expect(await shortFailures(sup, children, 6, serve)).toEqual([0, 5_000, 10_000, 20_000, 40_000]);
    expect(sup.snapshot().state).toBe('failed');

    sup.start().catch(() => {});
    expect(await shortFailures(sup, children, 2, serve)).toEqual([0, 5_000]);
  });
});
