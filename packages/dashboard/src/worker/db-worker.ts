/**
 * The dashboard's database worker.
 *
 * It owns the SQLite handle exclusively, plus every route whose handler does
 * synchronous work — `node:sqlite` and drizzle are synchronous, and the model /
 * session / cache walks are synchronous `fs` too. All of that used to run on the
 * thread that also holds the transport and the download manager, where a single
 * heavy query froze download progress for its whole duration (measured: a trivial
 * RPC issued during a 1.5 s synchronous call waited 1449 ms) and delayed a cancel
 * click from even being RECEIVED.
 *
 * Blocking THIS thread is fine and expected: nothing time-critical lives here.
 */

import { parentPort, receiveMessageOnPort, workerData, type MessagePort } from 'node:worker_threads';

import type { IngestSummary, WorkerApiContext } from '../api/context.js';
import { dispatchWorker } from '../api/dispatch.js';
import { failure, toFailure, type ApiResponse } from '../api/errors.js';
import { openDashboardDb } from '../db/open.js';
import { ingestSessions } from '../ingest/sessions.js';
import { ingestTraces } from '../ingest/traces.js';
import type { DbWorkerBoot, MainToWorker, WorkerToMain } from './protocol.js';

/** Trace-file retention passed to the periodic + boot ingest. */
const RETENTION_DAYS = 30;

function requireParentPort(): MessagePort {
  if (parentPort === null) throw new Error('db-worker.ts must be started as a worker thread');
  return parentPort;
}
const port = requireParentPort();

const { dbPath, modelsDir, sessionsRoot, tracesDir, cacheRoot, withdrawPort } = workerData as DbWorkerBoot;

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

const dash = openDashboardDb(dbPath);

// Serialize ingests so overlapping timer/boot/manual runs never interleave
// SQLite transactions. `doIngest` never throws, so a failed run cannot poison
// the chain for later callers (a rejected promise would skip every queued
// `.then`).
const doIngest = async (): Promise<IngestSummary> => {
  try {
    const sessionsResult = await ingestSessions(dash, sessionsRoot);
    const tracesResult = await ingestTraces(dash, tracesDir, { retentionDays: RETENTION_DAYS });
    return { sessions: sessionsResult, traces: tracesResult };
  } catch (err) {
    return {
      sessions: { scanned: 0, updated: 0, removed: 0, warnings: [String(err)] },
      traces: { files: 0, records: 0, pruned: 0, warnings: [] },
    };
  }
};
let ingestChain: Promise<IngestSummary> = Promise.resolve({
  sessions: { scanned: 0, updated: 0, removed: 0, warnings: [] },
  traces: { files: 0, records: 0, pruned: 0, warnings: [] },
});
const runIngest = (): Promise<IngestSummary> => {
  ingestChain = ingestChain.then(doIngest);
  return ingestChain;
};

/**
 * Await the ingest chain to its FIXPOINT, re-reading after each await: a caller
 * can queue another rescan while we wait, and closing the database under an
 * in-flight one only ever looked harmless because `doIngest`'s catch swallowed
 * the resulting "database is not open".
 *
 * `ingestSessions`/`ingestTraces` have fully synchronous bodies today, so the
 * chain advances on microtasks and no `close` message can be DELIVERED while it
 * runs — which makes this await invisible in a test right now. It is what keeps
 * the invariant true the day one of them awaits real I/O.
 */
const drainIngest = async (): Promise<void> => {
  let awaited: Promise<IngestSummary> | null = null;
  while (awaited !== ingestChain) {
    awaited = ingestChain;
    await awaited;
  }
};

const ctx: WorkerApiContext = { dash, modelsDir, sessionsRoot, tracesDir, cacheRoot, runIngest };

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

async function answerCall(message: Extract<MainToWorker, { kind: 'call' }>): Promise<ApiResponse> {
  try {
    // A constant, trusted base: only the path + query of `message.path` are used.
    // Rebuilding the params HERE is why the wire carries a string — a
    // `URLSearchParams` is not structured-clonable.
    const url = new URL(message.path, 'http://localhost');
    return await dispatchWorker(ctx, {
      method: message.method,
      pathname: url.pathname,
      query: url.searchParams,
      body: message.body,
      ...(message.bodyError !== undefined ? { bodyError: message.bodyError } : {}),
    });
  } catch (err) {
    // `dispatchWorker` converts handler throws itself; this catches a fault in
    // matching (e.g. `decodeURIComponent` on a malformed `%` escape) so every
    // request gets exactly one reply.
    return toFailure(err);
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
      post({ kind: 'response', id: message.id, response: await answerCall(message) });
      return;
    case 'ingest':
      post({ kind: 'ingested', id: message.id, summary: await runIngest() });
      return;
    case 'drain':
      await drainIngest();
      post({ kind: 'drained', id: message.id });
      return;
    case 'close':
      // Drain again rather than trusting the earlier `drain`: a call answered in
      // between can queue a rescan of its own (the rename/delete handlers do), and
      // that must finish against an OPEN database.
      await drainIngest();
      dash.close();
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
void runIngest();
