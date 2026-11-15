/**
 * A database worker that holds one route forever: a call to `GET /api/metrics/turns` blocks the thread inside its
 * handler, so it can answer neither the call nor a withdrawal. Every other call is answered at once with this
 * thread's id, so a test can tell a replacement worker from the one it replaced.
 * Imports nothing relative, so it needs no `.js`→`.ts` resolve hook.
 */

import { parentPort, threadId } from 'node:worker_threads';

interface Request {
  kind: 'call' | 'ingest' | 'drain' | 'close';
  id: number;
  path?: string;
}

const port = parentPort;
if (port === null) throw new Error('held-db-worker.ts must run as a worker');

port.on('message', (message: Request) => {
  switch (message.kind) {
    case 'call':
      if (message.path?.startsWith('/api/metrics/turns') === true) {
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0);
        return;
      }
      port.postMessage({ kind: 'response', id: message.id, response: { ok: true, status: 200, body: { threadId } } });
      return;
    case 'ingest':
      port.postMessage({
        kind: 'ingested',
        id: message.id,
        summary: { traces: { files: 0, records: 0, pruned: 0, warnings: [] } },
      });
      return;
    case 'drain':
      port.postMessage({ kind: 'drained', id: message.id });
      return;
    case 'close':
      port.postMessage({ kind: 'closed', id: message.id });
      port.close();
      return;
  }
});
