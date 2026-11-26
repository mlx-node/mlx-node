/**
 * A stuck or dead database worker restarts the worker only: its open calls fail typed, the next call is answered by a
 * new thread, and the runtime (with its agents and their turns) is never closed. Driven by events, not by time: the
 * deadlines are injected and every wait is for a lifecycle event or a reply.
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { MemorySecretStore } from '../src/chat/providers/secrets.js';
import type { ConversationEvent } from '../src/chat/types.js';
import { createDashboardRuntime, type RuntimeLifecycleEvent } from '../src/runtime.js';
import { startDbWorker, type DbWorkerLifecycle } from '../src/worker/client.js';
import { endpointModel, FAKE_AGENT, INFERENCE_TOKEN, readScenario, until } from './helpers/chat-fake.js';
import { startFakeEngine } from './helpers/fake-engine.js';

const HELD = new URL('./helpers/held-db-worker.ts', import.meta.url);
const CRASHING = new URL('./helpers/crashing-db-worker.ts', import.meta.url);

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function base(): string {
  const dir = mkdtempSync(join(tmpdir(), 'dash-worker-restart-'));
  dirs.push(dir);
  mkdirSync(join(dir, 'traces'), { recursive: true });
  mkdirSync(join(dir, 'models'), { recursive: true });
  return dir;
}

/** Resolves with the first lifecycle event of `type`. */
function eventOf<T extends RuntimeLifecycleEvent['type']>(
  seen: RuntimeLifecycleEvent[],
  type: T,
): Promise<Extract<RuntimeLifecycleEvent, { type: T }>> {
  return new Promise((resolve) => {
    const check = (): void => {
      const found = seen.find((event) => event.type === type);
      if (found !== undefined) resolve(found as Extract<RuntimeLifecycleEvent, { type: T }>);
      else setImmediate(check);
    };
    check();
  });
}

function threadOf(response: { ok: boolean; body?: unknown }): number | null {
  return response.ok ? ((response.body as { threadId?: number }).threadId ?? null) : null;
}

describe('a held worker call', () => {
  it('restarts the worker alone: the held caller gets E_UNAVAILABLE, the next call a new thread, the runtime stays', async () => {
    const dir = base();
    const seen: RuntimeLifecycleEvent[] = [];
    const runtime = createDashboardRuntime({
      chat: { root: join(dir, 'chat'), tempRoots: [] },
      agentHome: join(dir, 'agent'),
      dbPath: ':memory:',
      tracesDir: join(dir, 'traces'),
      modelsDir: join(dir, 'models'),
      supervisorPath: join(dir, 'supervisor.json'),
      workerUrl: HELD,
      shutdownTimeoutMs: 100,
      workerRequestTimeoutMs: 200,
      onLifecycle: (event) => seen.push(event),
    });
    let closed = false;
    const close = runtime.close.bind(runtime);
    runtime.close = async (): Promise<void> => {
      closed = true;
      await close();
    };
    try {
      const before = await runtime.call({ method: 'GET', path: '/api/models' });
      expect(threadOf(before)).not.toBeNull();

      const held = runtime.call({ method: 'GET', path: '/api/metrics/turns?traceId=x' });
      const controller = new AbortController();
      // The page's deadline withdraws it; the held handler can acknowledge nothing.
      const second = runtime.call({ method: 'GET', path: '/api/metrics/turns?traceId=y' }, controller.signal);
      controller.abort();

      const restarted = await eventOf(seen, 'worker-restarted');
      expect(restarted).toMatchObject({ type: 'worker-restarted', reason: 'no-ack', attempt: 1 });
      expect(restarted.route).toBe('GET /api/metrics/turns');
      expect(restarted.ageMs).toBeGreaterThanOrEqual(0);

      for (const response of [await held, await second]) {
        expect(response).toMatchObject({
          ok: false,
          code: 'E_UNAVAILABLE',
          message: 'Dashboard database restarted; the request was not answered',
        });
      }
      const after = await runtime.call({ method: 'GET', path: '/api/models' });
      expect(threadOf(after)).not.toBeNull();
      expect(threadOf(after)).not.toBe(threadOf(before));
      // A main-thread route never noticed.
      expect((await runtime.call({ method: 'GET', path: '/api/downloads' })).status).toBe(200);
      expect(closed).toBe(false);
      expect(seen.map((event) => event.type)).toEqual(['worker-restarted']);
    } finally {
      await runtime.close();
    }
  }, 30_000);
});

describe('a held worker call while a Work turn runs', () => {
  it('restarts the worker only: the agent and its turn keep running and the turn completes', async () => {
    const dir = base();
    const scenarioPath = join(dir, 'scenario.json');
    writeFileSync(
      scenarioPath,
      JSON.stringify({ ...readScenario('work-permission'), recordTo: join(dir, 'record.jsonl') }),
    );
    const engine = await startFakeEngine();
    const home = join(dir, 'home');
    const project = join(home, 'code', 'app');
    mkdirSync(project, { recursive: true });
    const seen: RuntimeLifecycleEvent[] = [];
    const runtime = createDashboardRuntime({
      chat: {
        root: join(dir, 'chat'),
        tempRoots: [],
        agentCommand: { command: process.execPath, args: [FAKE_AGENT, scenarioPath] },
        readEndpoint: () =>
          Promise.resolve({
            version: 1,
            pid: process.pid,
            url: engine.url,
            token: INFERENCE_TOKEN,
            models: [endpointModel('qwen3.5-4b', '/models/qwen3.5-4b')],
          }),
        secrets: new MemorySecretStore(),
        homeDir: home,
        baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home },
        timings: { stopEofMs: 300, stopTermMs: 300 },
      },
      agentHome: join(dir, 'agent-home'),
      dbPath: ':memory:',
      tracesDir: join(dir, 'traces'),
      modelsDir: join(dir, 'models'),
      supervisorPath: join(dir, 'supervisor.json'),
      workerUrl: HELD,
      shutdownTimeoutMs: 100,
      workerRequestTimeoutMs: 200,
      onLifecycle: (event) => seen.push(event),
    });
    const chat = runtime.context.chat!;
    const window = chat.watch({ scope: 'host' }, () => {});
    try {
      const { conversation } = await chat.create({ mode: 'work', text: 'ls', cwd: project });
      const events: ConversationEvent[] = [];
      const off = chat.watchConversation(conversation.id, (event) => events.push(event));
      const card = (): string | null => {
        for (const event of events) {
          if (event.type === 'interaction') return event.interaction.id;
          if (event.type === 'sync' && event.interactions.length > 0) return event.interactions[0]!.id;
        }
        return null;
      };
      await until(() => card() !== null, 5_000);

      const held = await runtime.call({ method: 'GET', path: '/api/metrics/turns?traceId=x' });
      expect(held).toMatchObject({ ok: false, code: 'E_UNAVAILABLE' });
      expect(seen.map((event) => event.type)).toEqual(['worker-restarted']);
      expect(runtime.busyConversations()).toEqual([conversation.id]);

      await chat.answer(conversation.id, card()!, { optionId: 'allow-once' });
      const ended = (): string | undefined => {
        for (const event of events) if (event.type === 'turn' && event.turn === null) return event.outcome;
        return undefined;
      };
      await until(() => ended() !== undefined, 5_000);
      off();
      expect(ended()).toBe('completed');
    } finally {
      window();
      await runtime.close();
      await engine.close();
    }
  }, 30_000);
});

describe('a worker that keeps dying', () => {
  it('waits longer each time and gives up after six quick deaths; a later call starts one new worker', async () => {
    const dir = base();
    const seen: DbWorkerLifecycle[] = [];
    let now = 0;
    const client = startDbWorker({
      workerUrl: CRASHING,
      shutdownTimeoutMs: 100,
      requestTimeoutMs: 5_000,
      onLifecycle: (event) => seen.push(event),
      restartDelaysMs: [0, 0, 0, 0, 0],
      now: () => now,
      dbPath: ':memory:',
      modelsDir: join(dir, 'models'),
      tracesDir: join(dir, 'traces'),
      runtimeDir: join(dir, 'runtime'),
      cacheRoot: undefined,
      prefixCacheRoot: undefined,
      agentHome: join(dir, 'agent'),
      chatAgentHome: join(dir, 'chat-agent'),
      chatIndexPath: join(dir, 'conversations.json'),
      scratchRoot: join(dir, 'Portato'),
    });
    try {
      for (let death = 1; death <= 6; death += 1) {
        const response = await client.call({ method: 'GET', path: '/api/models', route: 'GET /api/models' });
        expect(response).toMatchObject({ ok: false, code: 'E_UNAVAILABLE' });
      }
      expect(seen.map((event) => event.type)).toEqual([
        'worker-restarted',
        'worker-restarted',
        'worker-restarted',
        'worker-restarted',
        'worker-restarted',
        'worker-gave-up',
      ]);
      expect(seen.at(-1)).toMatchObject({ reason: 'exit', attempt: 6 });

      // Given up: worker routes fail at once.
      expect(await client.call({ method: 'GET', path: '/api/models' })).toMatchObject({
        ok: false,
        message: 'Dashboard database is unavailable: the database worker stopped restarting',
      });
      // A minute later, one call starts one new worker; it dies quickly too, and the client gives up again.
      now += 60_000;
      await client.call({ method: 'GET', path: '/api/models' });
      expect(seen.map((event) => event.type).at(-1)).toBe('worker-gave-up');
      expect(seen).toHaveLength(7);
    } finally {
      await client.close();
    }
  }, 30_000);

  it('a worker that hangs soon after its start is a quick death, though it dies a whole deadline later', async () => {
    const dir = base();
    const seen: DbWorkerLifecycle[] = [];
    let now = 0;
    const client = startDbWorker({
      workerUrl: HELD,
      shutdownTimeoutMs: 100,
      requestTimeoutMs: 200,
      onLifecycle: (event) => seen.push(event),
      restartDelaysMs: [0, 0, 0, 0, 0],
      now: () => now,
      dbPath: ':memory:',
      modelsDir: join(dir, 'models'),
      tracesDir: join(dir, 'traces'),
      runtimeDir: join(dir, 'runtime'),
      cacheRoot: undefined,
      prefixCacheRoot: undefined,
      agentHome: join(dir, 'agent'),
      chatAgentHome: join(dir, 'chat-agent'),
      chatIndexPath: join(dir, 'conversations.json'),
      scratchRoot: join(dir, 'Portato'),
    });
    try {
      // The first worker served 100 s before it hung: a fresh run of deaths.
      now = 100_000;
      for (let death = 1; death <= 6; death += 1) {
        const held = client.call({ method: 'GET', path: '/api/metrics/turns', route: 'GET /api/metrics/turns' });
        // Each hang is found a whole deadline (60 s + 4 s in the app) after the call was posted.
        now += 70_000;
        expect(await held).toMatchObject({ ok: false, code: 'E_UNAVAILABLE' });
      }
      expect(seen.map((event) => event.type)).toEqual([
        'worker-restarted',
        'worker-restarted',
        'worker-restarted',
        'worker-restarted',
        'worker-restarted',
        'worker-gave-up',
      ]);
      expect(seen.at(-1)).toMatchObject({ reason: 'no-ack', route: 'GET /api/metrics/turns', attempt: 6 });
    } finally {
      await client.close();
    }
  }, 30_000);

  it('uses 0, 5, 10, 20 and 40 s between quick deaths by default', async () => {
    const { WORKER_RESTART_DELAYS_MS, WORKER_GIVE_UP_DEATHS, WORKER_QUICK_DEATH_MS } =
      await import('../src/worker/client.js');
    expect([WORKER_RESTART_DELAYS_MS, WORKER_GIVE_UP_DEATHS, WORKER_QUICK_DEATH_MS]).toEqual([
      [0, 5_000, 10_000, 20_000, 40_000],
      6,
      60_000,
    ]);
  });
});
