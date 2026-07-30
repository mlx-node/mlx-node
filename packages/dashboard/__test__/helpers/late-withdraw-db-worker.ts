/**
 * A worker that deterministically loses the withdrawal race.
 *
 * `/race` receives a two-cell SharedArrayBuffer from the parent. It samples the
 * independent withdrawal port exactly once and moves cell 0 from 0 → 1 after
 * that actionable sample was empty. The parent then aborts the request (the exact
 * signal the renderer deadline drives through the RPC host). The fixture observes
 * that late withdrawal as evidence only and moves cell 0 from 1 → 2; only then
 * does the parent release cell 1. The fixture mutates an on-disk witness and
 * returns success — the precise cross-port interleaving the real client must
 * treat as "already started", never as a cancelled failure.
 */

import { rmSync } from 'node:fs';
import { join } from 'node:path';
import { parentPort, receiveMessageOnPort, workerData, type MessagePort } from 'node:worker_threads';

interface Boot {
  sessionsRoot: string;
  withdrawPort?: MessagePort;
}

interface Request {
  kind: 'call' | 'ingest' | 'drain' | 'close';
  id: number;
  path?: string;
  body?: unknown;
}

const port = parentPort;
if (port === null) throw new Error('late-withdraw-db-worker.ts must run as a worker');
const { sessionsRoot, withdrawPort } = workerData as Boot;

function sampledWithdrawal(id: number): boolean {
  if (withdrawPort === undefined) return false;
  let received = receiveMessageOnPort(withdrawPort);
  while (received !== undefined) {
    if (received.message === id) return true;
    received = receiveMessageOnPort(withdrawPort);
  }
  return false;
}

function requireRaceGate(body: unknown): Int32Array {
  if (!(body instanceof SharedArrayBuffer)) {
    throw new Error('/race requires a SharedArrayBuffer synchronization gate');
  }
  const gate = new Int32Array(body);
  if (gate.length < 2) throw new Error('/race synchronization gate requires two cells');
  return gate;
}

port.on('message', (message: Request) => {
  switch (message.kind) {
    case 'call': {
      if (message.path === '/ready') {
        port.postMessage({ kind: 'response', id: message.id, response: { ok: true, status: 200, body: 'ready' } });
        return;
      }
      const gate = requireRaceGate(message.body);
      const withdrawn = sampledWithdrawal(message.id);
      if (withdrawn) {
        // Impossible in the gated test: cancellation is not requested until the
        // parent observes state 1. Keep the fixture honest if that ordering ever
        // regresses rather than silently exercising the wrong branch.
        Atomics.store(gate, 0, -1);
        Atomics.notify(gate, 0);
        port.postMessage({ kind: 'withdrawn', id: message.id });
        return;
      }

      // Tell the parent the one actionable pre-handler sample completed empty.
      Atomics.store(gate, 0, 1);
      Atomics.notify(gate, 0);

      // Evidence-only second observation. The real worker never samples after
      // its pre-handler gate, but the fixture must prove the independently
      // delivered withdrawal arrived before the parent releases the mutation.
      while (!sampledWithdrawal(message.id)) {
        // A short timed wait yields this worker while retaining synchronous access
        // to the otherwise-unstarted withdrawal port.
        Atomics.wait(gate, 1, 0, 10);
      }
      Atomics.store(gate, 0, 2);
      Atomics.notify(gate, 0);

      while (Atomics.load(gate, 1) === 0) Atomics.wait(gate, 1, 0);
      rmSync(join(sessionsRoot, 'late-withdraw-target'), { force: true });
      port.postMessage({
        kind: 'response',
        id: message.id,
        response: { ok: true, status: 200, body: { mutated: true, withdrawalArrivedAfterSample: true } },
      });
      return;
    }
    case 'ingest':
      port.postMessage({
        kind: 'ingested',
        id: message.id,
        summary: {
          sessions: { scanned: 0, updated: 0, removed: 0, warnings: [] },
          traces: { files: 0, records: 0, pruned: 0, warnings: [] },
        },
      });
      return;
    case 'drain':
      port.postMessage({ kind: 'drained', id: message.id });
      return;
    case 'close':
      port.postMessage({ kind: 'closed', id: message.id });
      port.close();
      withdrawPort?.close();
      return;
  }
});
