import type { IncomingMessage, ServerResponse } from 'node:http';
import { Writable } from 'node:stream';

import type { MemoryGovernorStatus } from '@mlx-node/core';
import { describe, expect, it } from 'vite-plus/test';

import {
  createHealthReporter,
  deriveHealthStatus,
  toMemoryGovernorHealth,
  toMinimalHealth,
  type HealthStatusInputs,
  type ModelLoadRecord,
} from '../../packages/server/src/health.js';
import { ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';
import { routeRequest } from '../../packages/server/src/router.js';
import { createWeightResidency } from '../../packages/server/src/weight-residency.js';

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
    // A successfully loaded binding can be removed later, leaving a successful
    // historical record and an empty registry. That state is benign.
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

  it('reports the weights and their armed release, and stays ok once they are released', async () => {
    const registry = new ModelRegistry();
    const weights = createWeightResidency({
      idleReleaseMs: 600_000,
      release: async () => undefined,
      drain: () => undefined,
      exclusive: (fn) => fn(),
      busy: () => false,
      resident: () => false,
      log: () => undefined,
    });
    const report = createHealthReporter({ registry, weights });
    expect(report().work.releasePending).toBe(true);
    expect(report().weights).toEqual({
      idleReleaseSeconds: 600,
      released: false,
      restoring: false,
      restores: 0,
      restoreFailures: 0,
    });
    await weights.release('request');
    expect(report()).toMatchObject({ status: 'ok', weights: { released: true } });
    weights.close();
    expect(report().work.releasePending).toBe(false);
    expect('weights' in createHealthReporter({ registry })()).toBe(false);
  });

  // Mutation: map int8 to `none`/`none` → red.
  // Mutation: copy `kv` into the minimal body → red.
  it("kv lists resident models' format and window", () => {
    const registry = new ModelRegistry();
    const model = (window: number, kvFormat?: 'int8' | 'bf16') =>
      ({
        chatStreamSessionStart(): void {},
        contextLimits: () => ({
          trainedWindowTokens: 262144,
          effectiveWindowTokens: window,
          pagedBlockCapacity: 0,
          pagedBlockSize: 16,
          ...(kvFormat !== undefined ? { kvFormat } : {}),
        }),
      }) as never;
    registry.register('int8', model(139257, 'int8'));
    registry.register('bf16', model(73696, 'bf16'));
    registry.register('unknown', model(4096));
    registry.register('none', { chatStreamSessionStart(): void {} } as never);
    const health = createHealthReporter({ registry })();
    expect(health.kv).toEqual({
      int8: { maximum_context_tokens: 139257, format: 'int8', quantization: 'symmetric_int8', scale_type: 'float32' },
      bf16: { maximum_context_tokens: 73696, format: 'bf16', quantization: 'none', scale_type: 'none' },
      unknown: { maximum_context_tokens: 4096 },
    });
    expect(toMinimalHealth(health)).not.toHaveProperty('kv');
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

  // Mutation TS3: copy `memory_governor` into the minimal body → red.
  it('memory_governor is in the authenticated body only', () => {
    const cause = { gaps: 0, ms: 0 };
    const status: MemoryGovernorStatus = {
      limit: 16,
      charged: 7,
      headroom: 9,
      pressure: 'normal',
      deniedReservations: 0,
      host: {
        measured: true,
        available: 20,
        spare: 18,
        spareReserve: 3,
        headroom: 15,
        systemPressure: 'normal',
        growthAllowed: true,
      },
      accounts: [
        { name: 'fixed', current: 3 },
        { name: 'kv', current: 4 },
      ],
      owed: { bytes: 0, transfers: 0 },
      claims: { waiting: 0, bytes: 0, oldestMs: 0 },
      resourceWaits: { memory: 1, concurrency: 0, suspended: 0, draining: false, heldBehindRefusal: 2, oldestMs: 40 },
      counters: {
        replans: 1,
        resourceSuspensions: 0,
        resourceResumptions: 0,
        resourceTimeouts: 0,
        capacityExhausted: 0,
        metalOutOfMemory: 0,
        publicationsRefused: 3,
      },
      ttftPartsMs: { ownPrefill: 10, claimWait: 5, decode: 0, otherPrefill: 0 },
      decodeGapCauses: {
        prefillUnit: { gaps: 1, ms: 600 },
        claimWait: cause,
        ssdRead: cause,
        poolRelease: cause,
        replan: cause,
        other: cause,
        prefillMs: 200,
      },
      decodeGapMs: { p50: 1, p99: 2, max: 600 },
      decisions: ['Memory: start claim needs 1.0 MiB: short'],
      scheduler: [],
    };
    const governor = toMemoryGovernorHealth(status);
    const health = createHealthReporter({ registry: new ModelRegistry(), memoryGovernor: () => governor })();
    expect(health.memory_governor).toMatchObject({
      charged: 7,
      accounts: [
        { name: 'fixed', current: 3 },
        { name: 'kv', current: 4 },
      ],
      claims: { oldest_ms: 0 },
      resource_waits: { held_behind_refusal: 2, oldest_ms: 40 },
      counters: { replans: 1, publications_refused: 3 },
      ttft_parts_ms: { own_prefill: 10, claim_wait: 5 },
      decode_gap_causes: { prefill_unit: { gaps: 1, ms: 600 }, prefill_ms: 200 },
      decode_gap_ms: { max: 600 },
      owed: { bytes: 0 },
    });
    expect(toMinimalHealth(health)).not.toHaveProperty('memory_governor');
    const none = createHealthReporter({ registry: new ModelRegistry(), memoryGovernor: () => null })();
    expect('memory_governor' in none).toBe(false);
  });

  // Splash `/status` `scheduler` per engine: each entry is named after the
  // resident model whose context limits carry its governor id.
  // Mutation: skip the name join → red (no `model`).
  it('memory_governor.scheduler names each model it serves', () => {
    const latency = { p50: 512, p95: 640, p99: 640, max: 640, total: 1152, samples: 2 };
    const scheduler = {
      modelId: 7,
      queued: 1,
      waitingResources: 0,
      waitingPrefix: 0,
      prefilling: 1,
      decoding: 1,
      prefillBatches: 3,
      prefillRows: 1536,
      decodeBatches: 5,
      decodeBatchesByWidth: [4, 1],
      decodeGapTargetMs: 500,
      contendedPrefillMs: latency,
      prefillUnits: 3,
      boundCuts: 2,
      replans: 0,
    };
    const cause = { gaps: 0, ms: 0 };
    const governor = toMemoryGovernorHealth({
      limit: 16,
      charged: 7,
      headroom: 9,
      pressure: 'normal',
      deniedReservations: 0,
      host: {
        measured: true,
        available: 20,
        spare: 18,
        spareReserve: 3,
        headroom: 15,
        systemPressure: 'normal',
        growthAllowed: true,
      },
      accounts: [],
      owed: { bytes: 0, transfers: 0 },
      claims: { waiting: 0, bytes: 0, oldestMs: 0 },
      resourceWaits: { memory: 0, concurrency: 0, suspended: 0, draining: false, heldBehindRefusal: 0, oldestMs: 0 },
      counters: {
        replans: 0,
        resourceSuspensions: 0,
        resourceResumptions: 0,
        resourceTimeouts: 0,
        capacityExhausted: 0,
        metalOutOfMemory: 0,
        publicationsRefused: 0,
      },
      ttftPartsMs: { ownPrefill: 0, claimWait: 0, decode: 0, otherPrefill: 0 },
      decodeGapCauses: {
        prefillUnit: cause,
        claimWait: cause,
        ssdRead: cause,
        poolRelease: cause,
        replan: cause,
        other: cause,
        prefillMs: 0,
      },
      decodeGapMs: { p50: 0, p99: 0, max: 0 },
      decisions: [],
      scheduler: [scheduler, { ...scheduler, modelId: 8 }],
    });
    const registry = new ModelRegistry();
    registry.register('qwen', {
      chatStreamSessionStart(): void {},
      contextLimits: () => ({
        trainedWindowTokens: 262144,
        effectiveWindowTokens: 139257,
        pagedBlockCapacity: 0,
        pagedBlockSize: 16,
        governorModelId: 7,
      }),
    } as never);
    const health = createHealthReporter({ registry, memoryGovernor: () => governor })();
    expect(health.memory_governor?.scheduler).toEqual([
      {
        model: 'qwen',
        model_id: 7,
        queued: 1,
        waiting_resources: 0,
        waiting_prefix: 0,
        prefilling: 1,
        decoding: 1,
        prefill_batches: 3,
        prefill_rows: 1536,
        decode_batches: 5,
        decode_batches_by_width: { b1: 4, b2: 1 },
        decode_gap_target_ms: 500,
        contended_prefill_ms: latency,
        prefill_units: 3,
        bound_cuts: 2,
        replans: 0,
      },
      expect.not.objectContaining({ model: expect.anything() }),
    ]);
    expect(health.memory_governor?.scheduler[1]?.model_id).toBe(8);
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
