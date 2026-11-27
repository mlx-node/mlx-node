/**
 * The dashboard's database worker.
 *
 * It owns the SQLite handle exclusively, plus every route whose handler does
 * synchronous work — `node:sqlite` and drizzle are synchronous, and the model /
 * trace / cache walks are synchronous `fs` too. All of that used to run on the
 * thread that also holds the transport and the download manager, where a single
 * heavy query froze download progress for its whole duration (measured: a trivial
 * RPC issued during a 1.5 s synchronous call waited 1449 ms) and delayed a cancel
 * click from even being RECEIVED.
 *
 * Blocking THIS thread is fine and expected: nothing time-critical lives here.
 */

import { parentPort, receiveMessageOnPort, workerData, type MessagePort } from 'node:worker_threads';

import { failure } from '../api/errors.js';
import { createDbWorkerEngine } from './engine.js';
import type { DbWorkerBoot, MainToWorker, WorkerToMain } from './protocol.js';

function requireParentPort(): MessagePort {
  if (parentPort === null) throw new Error('db-worker.ts must be started as a worker thread');
  return parentPort;
}
const port = requireParentPort();

const boot = workerData as DbWorkerBoot;
const { withdrawPort } = boot;

/**
 * Requests the transport thread stopped waiting for before this thread reached
 * them. Their callers have already been answered `E_UNAVAILABLE`, so RUNNING
 * one would perform a mutation the UI reported as failed — the delete/rename
 * routes are exactly the ones that matter.
 */
const withdrawn = new Set<number>();
/**
 * Highest id handled so far. Ids are allocated and posted in one synchronous
 * step on the transport thread, so this thread sees them strictly increasing —
 * a withdrawal for anything at or below the mark names a request already
 * answered, and dropping those is what keeps the set bounded by queue depth
 * rather than growing for the life of the process.
 */
let highestHandled = 0;

/**
 * Drain the withdrawal port and report whether THIS request was retracted.
 *
 * Drained synchronously, right as the request is picked up, because that is the
 * only moment the answer is actionable: a `port.on('message')` listener would
 * not run until the current turn ends, by which point the mutation has already
 * happened. `receiveMessageOnPort` never starts the port, so it also never
 * becomes a handle keeping this thread alive.
 */
function isWithdrawn(id: number): boolean {
  if (id > highestHandled) highestHandled = id;
  if (withdrawPort !== undefined) {
    let received = receiveMessageOnPort(withdrawPort);
    while (received !== undefined) {
      const abandoned = received.message as number;
      if (abandoned >= highestHandled) withdrawn.add(abandoned);
      received = receiveMessageOnPort(withdrawPort);
    }
  }
  return withdrawn.delete(id);
}

const engine = createDbWorkerEngine(boot);

/**
 * Post a reply, or — when the payload defeats the structured clone algorithm (a
 * function, a class instance) — an `E_INTERNAL` in its place. `postMessage`
 * throws SYNCHRONOUSLY in that case, so without this the caller would wait for a
 * reply that was never sent.
 */
function post(message: WorkerToMain): void {
  try {
    port.postMessage(message);
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    port.postMessage({
      kind: 'response',
      id: message.id,
      response: failure('E_INTERNAL', `Response cannot cross the worker boundary: ${reason}`),
    } satisfies WorkerToMain);
  }
}

async function handle(message: MainToWorker): Promise<void> {
  switch (message.kind) {
    case 'call':
      // Checked BEFORE the handler runs, and only for `call`: `drain` / `close`
      // are shutdown steps whose whole point is to run even when nobody is
      // listening, and `ingest` is idempotent bookkeeping. The acknowledgement
      // is posted only when this exact gate skipped the handler; if cancellation
      // lost the cross-port race, the real response remains authoritative.
      if (isWithdrawn(message.id)) {
        post({ kind: 'withdrawn', id: message.id });
        return;
      }
      post({ kind: 'response', id: message.id, response: await engine.call(message) });
      return;
    case 'ingest':
      post({ kind: 'ingested', id: message.id, summary: await engine.ingest() });
      return;
    case 'drain':
      await engine.drain();
      post({ kind: 'drained', id: message.id });
      return;
    case 'close':
      await engine.close();
      post({ kind: 'closed', id: message.id });
      // Ending the message loop is what lets the thread exit on its own; the
      // runtime still terminates it if this never happens.
      port.close();
      withdrawPort?.close();
      return;
  }
}

port.on('message', (message: MainToWorker) => {
  void handle(message);
});

// Boot ingest, owned here so the transport thread never waits on it.
void engine.ingest();
