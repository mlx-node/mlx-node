import type { IncomingMessage, ServerResponse } from 'node:http';
import { Writable } from 'node:stream';

import { describe, expect, it } from 'vite-plus/test';

import {
  createHealthReporter,
  deriveHealthStatus,
  toMinimalHealth,
  type HealthStatusInputs,
  type ModelLoadRecord,
} from '../../packages/server/src/health.js';
import { ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';
import { routeRequest } from '../../packages/server/src/router.js';

/**
 * `deriveHealthStatus` is deliberately a PURE function over a plain input
 * record so the supervisor-facing status ladder can be pinned without
 * standing up an HTTP server, a registry, or the native addon. Every case
 * below is a fixture, not an observation.
 *
 * The ladder is ordered — a higher rung wins even when a lower rung's
 * condition also holds:
 *
 *   1. `writerActive`                              → 'loading'
 *   2. `lastLoad.ok === false` AND no resident     → 'error'
 *   3. queue saturated, OR (writers waiting AND    → 'degraded'
 *      inference in flight)
 *   4. otherwise                                   → 'ok'
 */

function inputs(overrides: Partial<HealthStatusInputs> = {}): HealthStatusInputs {
  return {
    writerActive: false,
    waitingWriters: 0,
    inFlight: 0,
    residentModelCount: 0,
    queueSaturated: false,
    lastLoad: null,
    ...overrides,
  };
}

function loadRecord(overrides: Partial<ModelLoadRecord> = {}): ModelLoadRecord {
  return {
    label: 'qwen3.5-3b',
    startedAt: 1_000,
    finishedAt: 2_000,
    ok: true,
    error: null,
    ...overrides,
  };
}

describe('deriveHealthStatus', () => {
  it("reports 'ok' for a freshly started server with nothing loaded", () => {
    // "Up, nothing loaded" is a legitimate steady state for a supervised
    // server that lazy-loads on first request — it must NOT read as an error.
    expect(deriveHealthStatus(inputs())).toBe('ok');
  });

  it("reports 'ok' with a model resident and no work in flight", () => {
    expect(deriveHealthStatus(inputs({ residentModelCount: 1, lastLoad: loadRecord() }))).toBe('ok');
  });

  it("reports 'loading' while the coordinator's writer slot is held", () => {
    expect(deriveHealthStatus(inputs({ writerActive: true }))).toBe('loading');
  });

  it("prefers 'loading' over 'error' when a retry load is already in flight", () => {
    // A previous load failed and nothing is resident, but a new load has the
    // writer slot. The supervisor should wait, not restart the process.
    expect(
      deriveHealthStatus(
        inputs({
          writerActive: true,
          residentModelCount: 0,
          lastLoad: loadRecord({ ok: false, error: 'no such file' }),
        }),
      ),
    ).toBe('loading');
  });

  it("prefers 'loading' over 'degraded' when the queue is saturated mid-load", () => {
    expect(deriveHealthStatus(inputs({ writerActive: true, queueSaturated: true }))).toBe('loading');
  });

  it("reports 'error' when the last load failed and nothing is resident", () => {
    expect(
      deriveHealthStatus(
        inputs({ residentModelCount: 0, lastLoad: loadRecord({ ok: false, error: 'weights missing' }) }),
      ),
    ).toBe('error');
  });

  it("does NOT report 'error' when a failed load left an earlier model resident", () => {
    // A failed hot-swap that rolled back to the previous resident is still a
    // serving process. Reporting 'error' here would make a supervisor kill a
    // server that is answering requests correctly.
    expect(
      deriveHealthStatus(inputs({ residentModelCount: 1, lastLoad: loadRecord({ ok: false, error: 'oom' }) })),
    ).toBe('ok');
  });

  it("does NOT report 'error' for a successful load with nothing resident", () => {
    // `withModelLoad` brackets every resolve attempt, including no-op fast
    // paths, so `ok: true` with an empty registry is possible and benign.
    expect(deriveHealthStatus(inputs({ residentModelCount: 0, lastLoad: loadRecord({ ok: true }) }))).toBe('ok');
  });

  it("reports 'degraded' when a per-model queue is at its configured max", () => {
    expect(deriveHealthStatus(inputs({ residentModelCount: 1, queueSaturated: true }))).toBe('degraded');
  });

  it("reports 'degraded' when a load is queued behind live inference", () => {
    // waitingWriters > 0 AND inFlight > 0 means a swap is parked behind
    // readers: new work on the incoming model will stall until they drain.
    expect(deriveHealthStatus(inputs({ residentModelCount: 1, waitingWriters: 1, inFlight: 2 }))).toBe('degraded');
  });

  it("stays 'ok' when writers are waiting but nothing is in flight", () => {
    // A transient state between `acquireWrite()` and the queue draining;
    // it resolves on the next tick and is not worth alarming on.
    expect(deriveHealthStatus(inputs({ waitingWriters: 1, inFlight: 0 }))).toBe('ok');
  });

  it("stays 'ok' when inference is in flight with no writer waiting", () => {
    expect(deriveHealthStatus(inputs({ residentModelCount: 1, inFlight: 4 }))).toBe('ok');
  });

  it("prefers 'error' over 'degraded'", () => {
    expect(
      deriveHealthStatus(
        inputs({ residentModelCount: 0, queueSaturated: true, lastLoad: loadRecord({ ok: false, error: 'boom' }) }),
      ),
    ).toBe('error');
  });
});

describe('toMinimalHealth', () => {
  it('projects exactly the three fields safe to serve without a token', () => {
    const full = createHealthReporter({ registry: new ModelRegistry() })();
    const minimal = toMinimalHealth(full);
    expect(Object.keys(minimal).sort()).toEqual(['pid', 'status', 'uptimeMs']);
    expect(minimal.status).toBe(full.status);
    expect(minimal.pid).toBe(full.pid);
  });

  it('never leaks resident model names', () => {
    const registry = new ModelRegistry();
    registry.register('secret-project-model', { chatStreamSessionStart(): void {} } as never);
    const minimal = toMinimalHealth(createHealthReporter({ registry })());
    expect(JSON.stringify(minimal)).not.toContain('secret-project-model');
  });
});

describe('createHealthReporter', () => {
  it('lists resident model names and count', () => {
    const registry = new ModelRegistry();
    const model = { chatStreamSessionStart(): void {} } as never;
    registry.register('m-a', model);
    registry.register('m-b', model);
    const health = createHealthReporter({ registry })();
    expect(health.models.resident.sort()).toEqual(['m-a', 'm-b']);
    expect(health.models.count).toBe(2);
  });

  it('keeps `{ status: "ok" }` a strict subset of the full body', () => {
    // Pre-existing consumers poll `/health` and check `body.status === 'ok'`.
    // The richer body must not move or rename that field.
    const health = createHealthReporter({ registry: new ModelRegistry() })();
    expect(health.status).toBe('ok');
    expect(JSON.parse(JSON.stringify(health)).status).toBe('ok');
  });

  it('surfaces the coordinator writer state', async () => {
    const modelWorkCoordinator = new ModelWorkCoordinator();
    const report = createHealthReporter({ registry: new ModelRegistry(), modelWorkCoordinator });

    let release: (() => void) | undefined;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const load = modelWorkCoordinator.withModelLoad(() => held, 'slow-model');
    await Promise.resolve();

    const during = report();
    expect(during.work.writerActive).toBe(true);
    expect(during.status).toBe('loading');

    release?.();
    await load;

    const after = report();
    expect(after.work.writerActive).toBe(false);
    expect(after.lastLoad?.label).toBe('slow-model');
    expect(after.lastLoad?.ok).toBe(true);
  });

  it("reports 'error' after a failed load with an empty registry", async () => {
    const modelWorkCoordinator = new ModelWorkCoordinator();
    const report = createHealthReporter({ registry: new ModelRegistry(), modelWorkCoordinator });
    await expect(
      modelWorkCoordinator.withModelLoad(() => Promise.reject(new Error('weights not found')), 'broken'),
    ).rejects.toThrow('weights not found');

    const health = report();
    expect(health.status).toBe('error');
    expect(health.lastLoad?.ok).toBe(false);
    expect(health.lastLoad?.error).toMatch(/weights not found/);
    expect(health.lastLoad?.label).toBe('broken');
  });

  it('reports a monotonic uptime measured from the reporter start', () => {
    let clock = 5_000;
    const report = createHealthReporter({
      registry: new ModelRegistry(),
      startedAt: 5_000,
      now: () => clock,
    });
    expect(report().uptimeMs).toBe(0);
    clock = 8_500;
    expect(report().uptimeMs).toBe(3_500);
  });

  it('never returns a negative uptime if the clock steps backwards', () => {
    let clock = 5_000;
    const report = createHealthReporter({ registry: new ModelRegistry(), startedAt: 5_000, now: () => clock });
    clock = 4_000;
    expect(report().uptimeMs).toBe(0);
  });
});

/**
 * `routeRequest` is a module-level export inside `@mlx-node/server`.
 * `createHandler` always hands it a reporter, but a direct caller (or a
 * future refactor that forgets the extras bag) must still get a parseable
 * body — `JSON.stringify(undefined)` returns `undefined`, so `res.end()`
 * would otherwise emit an EMPTY 200 that every consumer's `JSON.parse`
 * chokes on. The fallback keeps the historical constant instead.
 */
describe('routeRequest /health without a reporter', () => {
  function mockRes(): { res: ServerResponse; status: () => number; body: () => string; done: Promise<void> } {
    let status = 0;
    let body = '';
    let resolveDone: (() => void) | undefined;
    const done = new Promise<void>((r) => {
      resolveDone = r;
    });
    const writable = new Writable({
      write(chunk: Uint8Array | string, _enc: string, cb: () => void) {
        body += chunk.toString();
        cb();
      },
    });
    (writable as unknown as { writeHead: (s: number) => void }).writeHead = (s: number) => {
      status = s;
    };
    const originalEnd = writable.end.bind(writable);
    (writable as unknown as { end: (c?: unknown) => unknown }).end = (chunk?: unknown) => {
      if (typeof chunk === 'string') body += chunk;
      resolveDone?.();
      return originalEnd();
    };
    return { res: writable as unknown as ServerResponse, status: () => status, body: () => body, done };
  }

  function mockReq(path: string): IncomingMessage {
    return { method: 'GET', url: path, headers: { host: 'localhost' } } as unknown as IncomingMessage;
  }

  it('falls back to the legacy constant body', async () => {
    const { res, status, body, done } = mockRes();
    await routeRequest(mockReq('/health'), res, new ModelRegistry(), null);
    await done;
    expect(status()).toBe(200);
    expect(JSON.parse(body())).toEqual({ status: 'ok' });
  });

  it('serves the full body once a reporter is supplied', async () => {
    const registry = new ModelRegistry();
    const { res, body, done } = mockRes();
    await routeRequest(mockReq('/v1/health'), res, registry, null, undefined, null, undefined, undefined, undefined, {
      health: createHealthReporter({ registry }),
      authenticated: true,
    });
    await done;
    const parsed = JSON.parse(body()) as Record<string, unknown>;
    expect(parsed.status).toBe('ok');
    expect(parsed.models).toEqual({ resident: [], count: 0 });
  });

  it('scrubs the body for an unauthenticated caller', async () => {
    const registry = new ModelRegistry();
    registry.register('hidden', { chatStreamSessionStart(): void {} } as never);
    const { res, body, done } = mockRes();
    await routeRequest(mockReq('/health'), res, registry, null, undefined, null, undefined, undefined, undefined, {
      health: createHealthReporter({ registry }),
      authenticated: false,
    });
    await done;
    expect(Object.keys(JSON.parse(body()) as object).sort()).toEqual(['pid', 'status', 'uptimeMs']);
    expect(body()).not.toContain('hidden');
  });
});
