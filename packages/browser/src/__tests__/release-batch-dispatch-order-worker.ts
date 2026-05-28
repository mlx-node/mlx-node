/**
 * Worker helper for the dispatch-batch-ordering test in
 * release-batch-faf.test.ts.
 *
 * Scenario (proves flushPendingReleases drains any staged DISPATCH_BATCH
 * BEFORE publishing the BUFFER_RELEASE_BATCH F&F):
 *
 *   1. Build a bridge WITH dispatch batching enabled (dispatchBatchBuffer +
 *      batchEnabled=true). The drain-interlock worker uses a
 *      batching-disabled stub, so it cannot cover the dispatch-flush branch
 *      added for Task B's ordering fix.
 *   2. Wire up a minimal compute pass state so one FUSED_FULL_DISPATCH
 *      record stages into the dispatch batch (batchCount becomes 1).
 *      This uses 0-entry bind group descriptors so stageDispatchBatchRecord
 *      never touches any real GPU resources.
 *   3. Queue MAX_RELEASE_BATCH unknown handles through wgpuBufferRelease.
 *      The final push trips queueRelease's MAX_RELEASE_BATCH guard and
 *      invokes flushPendingReleases. flushPendingReleases sees
 *      batchCount > 0 and calls flushDispatchBatchInner FIRST, which does
 *      rpcCall(DISPATCH_BATCH) — a synchronous round-trip. Only after
 *      that returns does flushPendingReleases publish the F&F
 *      BUFFER_RELEASE_BATCH.
 *   4. While the worker runs the sequence the main thread acts as a fake
 *      gpu-worker: Atomics.waitAsync on STATUS, each time STATUS flips to
 *      PENDING it records the FN_ID and flips STATUS back to DONE +
 *      notify. The worker's blocking DISPATCH_BATCH rpcCall unblocks on
 *      that flip.
 *   5. After the sequence the worker reports the final cmd state (the F&F
 *      BUFFER_RELEASE_BATCH) so the test can double-check that the second
 *      PENDING the main thread saw was the release batch with exactly
 *      MAX_RELEASE_BATCH handles.
 *
 * Critical assertion: the main thread observes [DISPATCH_BATCH,
 * BUFFER_RELEASE_BATCH] in that order. If the ordering flush in
 * flushPendingReleases regresses, only BUFFER_RELEASE_BATCH appears (the
 * staged dispatch is silently dropped or reordered).
 *
 * Handshake: 'run' -> 'arm' (worker hands cmdBuffer to main) ->
 * 'go' (main has installed its waitAsync loop) -> 'done' (worker finished).
 * The worker stashes state across the 'run'/'go' boundary on `self`.
 */

import { CMD_OFFSET, STATUS_INDEX, MAX_RELEASE_BATCH, DISPATCH_BATCH_BUFFER_SIZE } from '../rpc-protocol';
import { createBridgeStub, type BridgeStub } from '../webgpu-bridge-stub';

interface WorkerState {
  cmdBuffer: SharedArrayBuffer;
  stub: BridgeStub;
  wasmMemory: WebAssembly.Memory;
}

let state: WorkerState | null = null;

self.onmessage = (e: MessageEvent) => {
  const { type } = e.data;
  try {
    if (type === 'run') {
      const cmdBuffer = new SharedArrayBuffer(CMD_OFFSET.TOTAL);
      const dispatchBatchBuffer = new SharedArrayBuffer(DISPATCH_BATCH_BUFFER_SIZE);
      const wasmMemory = new WebAssembly.Memory({ initial: 1 });
      const stub = createBridgeStub(
        cmdBuffer,
        wasmMemory,
        {
          instanceHandle: 1,
          adapterHandle: 2,
          deviceHandle: 3,
          queueHandle: 4,
        },
        /* readbackBuffer */ undefined,
        /* gpuFeatures */ undefined,
        /* poolStatsBuffer */ undefined,
        /* dispatchBatchBuffer */ dispatchBatchBuffer,
        /* batchEnabled */ true,
      );
      state = { cmdBuffer, stub, wasmMemory };
      // Hand the cmdBuffer to the main thread so it can install its fake
      // gpu-worker waitAsync loop. We do NOT proceed until main sends 'go'.
      (self as any).postMessage({ type: 'arm', cmdBuffer });
      return;
    }
    if (type === 'go' && state) {
      const { cmdBuffer, stub, wasmMemory } = state;
      // ---- Step 1: stage one FUSED_FULL_DISPATCH record into the batch SAB.
      //
      // wgpuDeviceCreateBindGroup reads the descriptor from wasmMemory.
      // Layout (from webgpu-bridge-stub.ts wgpuDeviceCreateBindGroup):
      //   +8  u32 layoutHandle
      //   +12 u32 entryCount
      //   +16 u32 entriesPtr
      // We use entryCount=0 so entriesPtr is irrelevant and no bind group
      // entry parsing runs. This keeps the stage path from triggering any
      // writeBuffer / deferred-create RPCs — we want a pure "stage & done".
      const heapView = new DataView(wasmMemory.buffer);
      const descPtr = 64; // arbitrary, well inside the 64 KiB initial page
      heapView.setUint32(descPtr + 8, 42 /* layoutHandle */, true);
      heapView.setUint32(descPtr + 12, 0 /* entryCount */, true);
      heapView.setUint32(descPtr + 16, 0 /* entriesPtr */, true);

      const bgHandle = (stub.imports.wgpuDeviceCreateBindGroup as (device: number, desc: number) => number)(3, descPtr);

      // Set up the pending compute-pass state so the dispatch takes the
      // FUSED_FULL_DISPATCH staging path.
      const passHandle = 0x5000; // arbitrary non-zero pass handle
      const pipelineHandle = 0x6000;
      (stub.imports.wgpuComputePassEncoderSetPipeline as (pass: number, pipe: number) => void)(
        passHandle,
        pipelineHandle,
      );
      (
        stub.imports.wgpuComputePassEncoderSetBindGroup as (
          pass: number,
          grpIdx: number,
          bg: number,
          dynOffCount: number,
          dynOffPtr: number,
        ) => void
      )(passHandle, 0, bgHandle, 0, 0);
      // Dispatch: stages one record. No RPC is issued yet because the
      // staging path returned true. batchCount is now 1 inside the stub.
      (
        stub.imports.wgpuComputePassEncoderDispatchWorkgroups as (pass: number, x: number, y: number, z: number) => void
      )(passHandle, 1, 1, 1);

      // ---- Step 2: trigger flushPendingReleases via queueRelease's
      // MAX_RELEASE_BATCH auto-flush. The release path for an unknown
      // handle (no size/usage entry in bufMetaSab) funnels straight into
      // queueRelease. Handles in the 1..1000 range are NOT treated as fake.
      const release = stub.imports.wgpuBufferRelease as (h: number) => void;
      for (let i = 0; i < MAX_RELEASE_BATCH; i++) {
        release(500 + i);
      }

      // ---- Step 3: report done. The main thread has been recording the
      // sequence of FN_IDs it observed. Send it the final SAB state so it
      // can double-check the F&F landing.
      const cmdU32 = new Uint32Array(cmdBuffer);
      const cmdI32 = new Int32Array(cmdBuffer);
      (self as any).postMessage({
        type: 'done',
        ok: true,
        finalFnId: cmdU32[CMD_OFFSET.FN_ID >>> 2],
        finalArg0: cmdU32[CMD_OFFSET.ARG0 >>> 2],
        finalStatus: cmdI32[STATUS_INDEX],
      });
      state = null;
      return;
    }
  } catch (err) {
    const msg = err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err);
    (self as any).postMessage({ type: 'done', ok: false, error: msg });
    state = null;
  }
};
