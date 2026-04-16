/**
 * Worker helper for release-batch-faf.test.ts.
 *
 * Runs the drain-interlock scenario entirely inside a Worker (where
 * Atomics.wait is permitted, unlike the browser main thread). See the test
 * file for the scenario description.
 */

import { createBridgeStub } from '../webgpu-bridge-stub';
import { CMD_OFFSET, STATUS, STATUS_INDEX, RpcFn, MAX_RELEASE_BATCH } from '../rpc-protocol';

self.onmessage = (e: MessageEvent) => {
  const { type } = e.data;
  if (type !== 'run') return;
  try {
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
    const uniformBase = CMD_OFFSET.UNIFORM_DATA >>> 2;
    const release = stub.imports.wgpuBufferRelease as (h: number) => void;

    // Fire F&F #1: 64 handles starting at 1.
    for (let i = 0; i < MAX_RELEASE_BATCH; i++) release(1 + i);

    if (cmdI32[STATUS_INDEX] !== STATUS.PENDING) {
      (self as any).postMessage({ ok: false, error: 'F&F #1 did not flip STATUS to PENDING' });
      return;
    }

    // Snapshot UNIFORM_DATA as it looks while F&F #1 is "in flight". A
    // real gpu-worker would be reading these bytes right now. If the
    // upcoming F&F #2 skips the drain, it will overwrite these slots
    // before the gpu-worker is done — the snapshot proves this does NOT
    // happen by showing the pre-drain contents are the #1 handles.
    const observedBeforeDrain = Array.from<number>({ length: MAX_RELEASE_BATCH });
    for (let i = 0; i < MAX_RELEASE_BATCH; i++) {
      observedBeforeDrain[i] = cmdU32[uniformBase + i]!;
    }

    // Flip STATUS to DONE synchronously — as if the gpu-worker finished
    // processing F&F #1. The drain's Atomics.wait will observe STATUS is
    // no longer PENDING and return 'not-equal' immediately. This tests
    // the drain *mechanics* without needing a second worker to ack.
    Atomics.store(cmdI32, STATUS_INDEX, STATUS.DONE);

    // Fire F&F #2. The auto-flush calls flushPendingReleases, which calls
    // drainFireAndForget() at the top. Correct drain behavior:
    //   1. ffPending is true → enter wait
    //   2. Atomics.wait(STATUS=PENDING) → 'not-equal' (STATUS is DONE now)
    //   3. Atomics.store(STATUS=IDLE), clear ffPending
    //   4. proceed to stage the next 64 handles into UNIFORM_DATA
    for (let i = 0; i < MAX_RELEASE_BATCH; i++) release(100 + i);

    // Verify F&F #2 landed correctly.
    const ok2 =
      cmdI32[STATUS_INDEX] === STATUS.PENDING &&
      cmdU32[CMD_OFFSET.FN_ID >>> 2] === RpcFn.BUFFER_RELEASE_BATCH &&
      cmdU32[CMD_OFFSET.ARG0 >>> 2] === MAX_RELEASE_BATCH &&
      cmdU32[uniformBase] === 100 &&
      cmdU32[uniformBase + 63] === 163;

    // Before-drain observation must be the #1 handles 1..64.
    let priorOk = true;
    for (let i = 0; i < MAX_RELEASE_BATCH; i++) {
      if (observedBeforeDrain[i] !== 1 + i) {
        priorOk = false;
        break;
      }
    }

    const stats = stub.getBridgeStats();
    const batchCount = stats.byFn[RpcFn.BUFFER_RELEASE_BATCH] ?? 0;

    (self as any).postMessage({
      ok: ok2 && priorOk && batchCount === 2,
      ok2,
      priorOk,
      batchCount,
    });
  } catch (err) {
    const msg = err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err);
    (self as any).postMessage({ ok: false, error: msg });
  }
};
