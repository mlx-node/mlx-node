/**
 * Worker helper for release-batch-faf.test.ts.
 *
 * Runs the drain-interlock scenario entirely inside a Worker (where
 * Atomics.wait is permitted, unlike the browser main thread).
 *
 * Scenario (proves drainFireAndForget() actually blocks):
 *   1. Fire F&F #1 with 64 handles (1..64). STATUS flips to PENDING.
 *      Record T1 = performance.now() right after this returns.
 *   2. Signal the main thread we are about to fire F&F #2. The main
 *      thread starts a ~50ms timer that, when it fires, writes STATUS=DONE
 *      and Atomics.notify's the worker.
 *   3. Immediately fire F&F #2 (64 more handles, 100..163). This triggers
 *      flushPendingReleases → drainFireAndForget, which Atomics.wait's for
 *      STATUS to leave PENDING. The worker must BLOCK here until the main
 *      thread flips STATUS to DONE.
 *   4. Record T2 = performance.now() after F&F #2 returns.
 *   5. elapsed = T2 - T1 must be >= the delay (proving the drain blocked).
 *      If drainFireAndForget() is removed from production, F&F #2 returns
 *      in microseconds and elapsed would be < 1 ms.
 *
 * The test file asserts elapsed >= 40 ms (allowing slack below the 50 ms
 * delay for scheduler / timer jitter) AND that F&F #2 landed correctly
 * in the SAB.
 */

import { CMD_OFFSET, STATUS, STATUS_INDEX, RpcFn, MAX_RELEASE_BATCH } from '../rpc-protocol';
import { createBridgeStub } from '../webgpu-bridge-stub';

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

    // ---- Step 1: Fire F&F #1 (handles 1..64), leaves STATUS=PENDING. ----
    for (let i = 0; i < MAX_RELEASE_BATCH; i++) release(1 + i);

    if (cmdI32[STATUS_INDEX] !== STATUS.PENDING) {
      // Terminal failure path: tag with type: 'done' so the harness fails
      // fast instead of waiting for its 10s outer timeout.
      (self as any).postMessage({ type: 'done', ok: false, error: 'F&F #1 did not flip STATUS to PENDING' });
      return;
    }

    // Handoff the cmdBuffer to the main thread so it can schedule the
    // delayed STATUS=DONE write while we are blocked in Atomics.wait.
    // The SAB is shared memory: the main thread sees writes immediately
    // without transferring. We just need to tell it "go now".
    (self as any).postMessage({ type: 'arm', cmdBuffer });

    // Small busy yield to let the main thread receive the 'arm' message
    // and start its setTimeout BEFORE we enter the drain. Atomics.wait
    // on STATUS=PENDING will return immediately if STATUS is already not
    // PENDING, so we do NOT want the main thread to race ahead and flip
    // STATUS=DONE before we even enter drainFireAndForget.
    //
    // Use a short tight loop to give the main thread one event-loop tick.
    // 2 ms busy-wait is enough for any reasonable scheduler — and because
    // the main thread's setTimeout will fire ~50 ms AFTER it processes
    // the 'arm' message (not NOW), this brief delay only pushes out the
    // start of that 50 ms window. The assertion tolerates up to 50 ms.
    const yieldUntil = performance.now() + 2;
    while (performance.now() < yieldUntil) {
      // spin
    }

    // ---- Step 2: Fire F&F #2 and time how long it blocks. ----
    // At this instant STATUS is still PENDING (we set it above, main
    // thread's setTimeout has NOT fired yet). flushPendingReleases
    // entering drainFireAndForget will Atomics.wait until the main
    // thread flips STATUS to something other than PENDING.
    const t1 = performance.now();
    for (let i = 0; i < MAX_RELEASE_BATCH; i++) release(100 + i);
    const t2 = performance.now();
    const elapsedMs = t2 - t1;

    // ---- Step 3: Verify F&F #2 landed correctly. ----
    // After F&F #2: STATUS is PENDING again (the new batch is live),
    // FN_ID = BUFFER_RELEASE_BATCH, ARG0 = 64, UNIFORM_DATA[0..63] =
    // 100..163. If the drain actually ran, UNIFORM_DATA was rewritten
    // ONLY after the main thread flipped STATUS to DONE.
    const post = {
      status: cmdI32[STATUS_INDEX],
      fnId: cmdU32[CMD_OFFSET.FN_ID >>> 2],
      arg0: cmdU32[CMD_OFFSET.ARG0 >>> 2],
      first: cmdU32[uniformBase],
      last: cmdU32[uniformBase + 63],
    };

    const stats = stub.getBridgeStats();
    const batchCount = stats.byFn[RpcFn.BUFFER_RELEASE_BATCH] ?? 0;

    (self as any).postMessage({
      type: 'done',
      ok:
        post.status === STATUS.PENDING &&
        post.fnId === RpcFn.BUFFER_RELEASE_BATCH &&
        post.arg0 === MAX_RELEASE_BATCH &&
        post.first === 100 &&
        post.last === 163 &&
        batchCount === 2,
      elapsedMs,
      post,
      batchCount,
    });
  } catch (err) {
    const msg = err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err);
    (self as any).postMessage({ type: 'done', ok: false, error: msg });
  }
};
