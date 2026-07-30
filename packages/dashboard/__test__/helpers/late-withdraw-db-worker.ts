/**
 * A worker that deterministically loses the withdrawal race.
 *
 * `/ready` warms the thread first. `/race` then samples the independent
 * withdrawal port exactly once, blocks past the client's deadline, mutates an
 * on-disk witness, and returns success. Cancellation arrives while it is
 * blocked, just after the only sample — the precise cross-port interleaving the
 * real client must treat as "already started", never as a cancelled failure.
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

port.on('message', (message: Request) => {
  switch (message.kind) {
    case 'call': {
      if (message.path === '/ready') {
        port.postMessage({ kind: 'response', id: message.id, response: { ok: true, status: 200, body: 'ready' } });
        return;
      }
      const withdrawn = sampledWithdrawal(message.id);
      // Block this worker's event loop long enough for the transport-thread
      // deadline to post withdrawal on the other port.
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 1_000);
      if (withdrawn) {
        port.postMessage({ kind: 'withdrawn', id: message.id });
        return;
      }
      rmSync(join(sessionsRoot, 'late-withdraw-target'), { force: true });
      port.postMessage({
        kind: 'response',
        id: message.id,
        response: { ok: true, status: 200, body: { mutated: true } },
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
