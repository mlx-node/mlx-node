/**
 * Unit tests for Task B: BUFFER_RELEASE_BATCH fire-and-forget (F&F).
 *
 * Verifies that:
 *   1. flushPendingReleases writes the expected cmd-SAB state and returns
 *      WITHOUT an Atomics.wait round-trip (the core perf win).
 *   2. The next cmd-SAB-touching RPC drains the prior F&F before clobbering
 *      cmd-SAB state — proven by running the drain inside a Worker (main
 *      thread cannot call Atomics.wait). See release-batch-faf-worker.ts.
 *
 * These are white-box tests against the public `createBridgeStub` API —
 * we observe SAB bytes and bridge stats, never reach into closures.
 */

import { describe, expect, it } from 'vitest';

import { createBridgeStub } from '../webgpu-bridge-stub';
import { CMD_OFFSET, STATUS, STATUS_INDEX, RpcFn, MAX_RELEASE_BATCH } from '../rpc-protocol';

/** Allocate a fresh cmd SAB + a minimal wasmMemory for the bridge. */
function makeBridgeHarness() {
  const cmdBuffer = new SharedArrayBuffer(CMD_OFFSET.TOTAL);
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const stub = createBridgeStub(cmdBuffer, wasmMemory, {
    instanceHandle: 1,
    adapterHandle: 2,
    deviceHandle: 3,
    queueHandle: 4,
  });
  const cmdU32 = new Uint32Array(cmdBuffer);
  const cmdI32 = new Int32Array(cmdBuffer);
  return { cmdBuffer, cmdU32, cmdI32, stub };
}

describe('BUFFER_RELEASE_BATCH fire-and-forget (Task B)', () => {
  it('fires F&F synchronously when pendingReleases hits MAX_RELEASE_BATCH', () => {
    const { cmdU32, cmdI32, stub } = makeBridgeHarness();
    const release = stub.imports.wgpuBufferRelease as (h: number) => void;

    // Drive 64 unknown handles through wgpuBufferRelease. Unknown handles
    // (no setBufferMeta entry, no mappedAtCreation entry, no deferred entry)
    // fall through to queueRelease. When the queue hits MAX_RELEASE_BATCH,
    // flushPendingReleases fires F&F.
    const handles: number[] = [];
    for (let i = 0; i < MAX_RELEASE_BATCH; i++) {
      // Use handles in the 1..1000 range so they are NOT classified as fake
      // (>= 0x7d000000) — though for an unknown handle this doesn't matter;
      // it will route through queueRelease either way.
      const h = 1000 + i;
      handles.push(h);
      release(h);
    }

    // After exactly MAX_RELEASE_BATCH releases, the auto-flush must have
    // fired F&F. Observable state:
    expect(cmdU32[CMD_OFFSET.FN_ID >>> 2]).toBe(RpcFn.BUFFER_RELEASE_BATCH);
    expect(cmdU32[CMD_OFFSET.ARG0 >>> 2]).toBe(MAX_RELEASE_BATCH);
    expect(cmdI32[STATUS_INDEX]).toBe(STATUS.PENDING);
    // All 64 handles were packed into the UNIFORM_DATA region.
    const uniformBase = CMD_OFFSET.UNIFORM_DATA >>> 2;
    for (let i = 0; i < MAX_RELEASE_BATCH; i++) {
      expect(cmdU32[uniformBase + i]).toBe(handles[i]);
    }
    // CALLBACK_COUNT was cleared (matches rpcCall's invariant).
    expect(cmdU32[CMD_OFFSET.CALLBACK_COUNT >>> 2]).toBe(0);
    // Bridge histogram records the batched release (parity with the pre-F&F
    // rpcCall(BUFFER_RELEASE_BATCH) path so `?profile=1` diagnostics keep
    // their counts).
    const stats = stub.getBridgeStats();
    expect(stats.byFn[RpcFn.BUFFER_RELEASE_BATCH]).toBe(1);
    expect(stats.rpcCount).toBe(1);
  });

  it('does not block on Atomics.wait (returns within 5 ms)', () => {
    const { stub } = makeBridgeHarness();
    const release = stub.imports.wgpuBufferRelease as (h: number) => void;

    // If the implementation regressed and Atomics.wait were somehow invoked
    // on the main thread, it would either throw (TypeError in browsers) OR
    // block until timeout. Measure wall clock of the auto-flush to catch
    // either failure mode. 5 ms is generous — the fire-and-forget path is
    // a handful of stores.
    const t0 = performance.now();
    for (let i = 0; i < MAX_RELEASE_BATCH; i++) {
      release(2000 + i);
    }
    const elapsed = performance.now() - t0;
    expect(elapsed).toBeLessThan(5);
  });

  it('leaves STATUS=IDLE when there are zero pending releases', () => {
    const { cmdI32, stub } = makeBridgeHarness();
    const release = stub.imports.wgpuBufferRelease as (h: number) => void;

    // Drive 32 releases — half of MAX_RELEASE_BATCH — so the queue has entries
    // but no auto-flush fires. STATUS should remain IDLE, no F&F pending.
    for (let i = 0; i < 32; i++) {
      release(3000 + i);
    }
    expect(cmdI32[STATUS_INDEX]).toBe(STATUS.IDLE);
    // No BUFFER_RELEASE_BATCH in the bridge histogram yet.
    const stats = stub.getBridgeStats();
    expect(stats.byFn[RpcFn.BUFFER_RELEASE_BATCH] ?? 0).toBe(0);
  });

  it('drain interlock: next F&F drains prior F&F before clobbering UNIFORM_DATA', async () => {
    // Atomics.wait is only allowed inside a Worker, so delegate the whole
    // scenario to release-batch-faf-worker.ts. The helper runs:
    //   1. Fire F&F #1 with handles 1..64.
    //   2. Snapshot UNIFORM_DATA (proves the #1 handles are staged).
    //   3. Flip STATUS=DONE manually — simulates the gpu-worker finishing.
    //   4. Fire F&F #2 (handles 100..163). flushPendingReleases at the top
    //      calls drainFireAndForget(), which Atomics.wait's — but STATUS is
    //      DONE so wait returns 'not-equal' immediately, then the drain
    //      resets STATUS to IDLE and clears ffPending.
    //   5. After the drain, #2 stages its own UNIFORM_DATA, flips STATUS to
    //      PENDING.
    // Assertions: the pre-drain snapshot shows the #1 handles (no clobber
    // before drain) AND #2 fires correctly afterward.
    const worker = new Worker(new URL('./release-batch-faf-worker.ts', import.meta.url), { type: 'module' });
    try {
      const result = await new Promise<{ ok: boolean; error?: string; [k: string]: unknown }>((resolve, reject) => {
        const t = setTimeout(() => reject(new Error('drain-interlock worker timeout (10s)')), 10_000);
        worker.onmessage = (e) => {
          clearTimeout(t);
          resolve(e.data);
        };
        worker.onerror = (err) => {
          clearTimeout(t);
          reject(new Error(`worker error: ${err.message}`));
        };
        worker.postMessage({ type: 'run' });
      });
      expect(result.ok, `worker result: ${JSON.stringify(result)}`).toBe(true);
    } finally {
      worker.terminate();
    }
  });
});
