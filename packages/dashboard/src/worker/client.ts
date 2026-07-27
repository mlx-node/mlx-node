/**
 * The transport thread's side of the database worker.
 *
 * Its whole job is that no caller can hang. Every request is registered before
 * it is posted and settles exactly once — on the worker's reply, on a
 * structured-clone refusal (which `postMessage` throws synchronously), or on the
 * worker dying. A `call` that cannot be answered comes back as a failure
 * ENVELOPE rather than a rejection, so `runtime.call`'s "never rejects" contract
 * holds whatever happens to the thread.
 */

import { Worker } from 'node:worker_threads';

import type { IngestSummary } from '../api/context.js';
import { failure, type ApiResponse } from '../api/errors.js';
import type { DbWorkerBootstrap, MainToWorker, WorkerToMain } from './protocol.js';

/**
 * A reply, or why none is coming: `undeliverable` means the request never left
 * this thread (the structured clone algorithm refused it — a caller-side fault),
 * `down` means the worker cannot answer.
 */
type SendResult = { ok: true; reply: WorkerToMain } | { ok: false; why: 'undeliverable' | 'down'; reason: string };

export type DbWorkerLifecycle =
  /** The worker closed the SQLite handle. Ordered AFTER `downloads.shutdown()`. */
  | { type: 'db-closed' }
  /** The worker died outside of shutdown; every route it owns now fails. */
  | { type: 'worker-down'; reason: string };

export interface DbWorkerOptions extends DbWorkerBootstrap {
  /** Entry module. Packaging overrides it: under Electron the worker must load
   *  from an unpacked path outside the asar archive. */
  workerUrl: URL | string;
  /** Budget for one shutdown step before the thread is terminated outright. */
  shutdownTimeoutMs: number;
  onLifecycle: (event: DbWorkerLifecycle) => void;
}

export interface DbWorkerCall {
  method: string;
  /** Path, optionally with a query string — the worker re-parses it. */
  path: string;
  body?: unknown;
  bodyError?: string;
}

export interface DbWorkerClient {
  call(request: DbWorkerCall): Promise<ApiResponse>;
  ingest(): Promise<IngestSummary>;
  /** Await the worker's ingest chain; the database stays open. */
  drain(): Promise<void>;
  /** Drain, close the database, end the thread. Idempotent and bounded. */
  close(): Promise<void>;
}

/** Mirrors `doIngest`'s never-throw contract: a failed rescan is a warning, not an error. */
function unavailableSummary(reason: string): IngestSummary {
  return {
    sessions: { scanned: 0, updated: 0, removed: 0, warnings: [reason] },
    traces: { files: 0, records: 0, pruned: 0, warnings: [] },
  };
}

export function startDbWorker(opts: DbWorkerOptions): DbWorkerClient {
  const worker = new Worker(opts.workerUrl, {
    workerData: {
      dbPath: opts.dbPath,
      modelsDir: opts.modelsDir,
      sessionsRoot: opts.sessionsRoot,
      tracesDir: opts.tracesDir,
      cacheRoot: opts.cacheRoot,
    } satisfies DbWorkerBootstrap,
  });

  const pending = new Map<number, (result: SendResult) => void>();
  let nextId = 1;
  /** Set once no further reply can arrive; every later send fails fast. */
  let down: string | null = null;
  let closing = false;

  // Never hold the process open on the worker's account — same reasoning as the
  // unref'd rescan timer: the transport decides how long the process lives. Ref
  // again while a request is outstanding, or an otherwise idle loop would exit
  // before its reply arrived.
  const syncRef = (): void => {
    if (pending.size > 0) worker.ref();
    else worker.unref();
  };
  syncRef();

  const goDown = (reason: string): void => {
    if (down === null) {
      down = reason;
      if (!closing) opts.onLifecycle({ type: 'worker-down', reason });
    }
    const waiting = [...pending.values()];
    pending.clear();
    syncRef();
    for (const settle of waiting) settle({ ok: false, why: 'down', reason: down });
  };

  worker.on('error', (err: Error) => goDown(`dashboard database worker failed: ${err.message}`));
  worker.on('exit', (code: number) => goDown(`dashboard database worker exited (code ${code})`));

  worker.on('message', (message: WorkerToMain) => {
    // The db-closed witness is reported even if nobody is waiting on the reply,
    // so a supervisor sees the handle go down in the right order.
    if (message.kind === 'closed') opts.onLifecycle({ type: 'db-closed' });
    const settle = pending.get(message.id);
    if (settle === undefined) return;
    pending.delete(message.id);
    syncRef();
    settle({ ok: true, reply: message });
  });

  const send = (build: (id: number) => MainToWorker): Promise<SendResult> => {
    if (down !== null) return Promise.resolve({ ok: false, why: 'down', reason: down });
    const id = nextId++;
    return new Promise<SendResult>((resolve) => {
      pending.set(id, resolve);
      syncRef();
      try {
        worker.postMessage(build(id));
      } catch (err) {
        // A payload the structured clone algorithm refuses throws HERE, before
        // the worker ever sees it. Settling now is what keeps the caller from
        // waiting on a request that was never delivered.
        pending.delete(id);
        syncRef();
        resolve({ ok: false, why: 'undeliverable', reason: err instanceof Error ? err.message : String(err) });
      }
    });
  };

  /** `send` with a deadline, for the shutdown steps a wedged worker must not stall. */
  const sendBounded = async (build: (id: number) => MainToWorker, ms: number): Promise<SendResult> => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const deadline = new Promise<SendResult>((resolve) => {
      timer = setTimeout(() => resolve({ ok: false, why: 'down', reason: `timed out after ${ms}ms` }), ms);
    });
    try {
      return await Promise.race([send(build), deadline]);
    } finally {
      clearTimeout(timer);
    }
  };

  let closed: Promise<void> | null = null;

  return {
    async call(request: DbWorkerCall): Promise<ApiResponse> {
      const result = await send((id) => ({
        kind: 'call',
        id,
        method: request.method,
        path: request.path,
        body: request.body,
        ...(request.bodyError !== undefined ? { bodyError: request.bodyError } : {}),
      }));
      if (!result.ok) {
        return result.why === 'undeliverable'
          ? failure('E_BAD_REQUEST', `Request cannot cross the worker boundary: ${result.reason}`)
          : failure('E_UNAVAILABLE', `Dashboard database is unavailable: ${result.reason}`);
      }
      if (result.reply.kind !== 'response') {
        return failure('E_INTERNAL', `Unexpected worker reply "${result.reply.kind}" for a call`);
      }
      return result.reply.response;
    },
    async ingest(): Promise<IngestSummary> {
      const result = await send((id) => ({ kind: 'ingest', id }));
      if (!result.ok) return unavailableSummary(result.reason);
      if (result.reply.kind !== 'ingested') {
        return unavailableSummary(`Unexpected worker reply "${result.reply.kind}" for an ingest`);
      }
      return result.reply.summary;
    },
    async drain(): Promise<void> {
      await sendBounded((id) => ({ kind: 'drain', id }), opts.shutdownTimeoutMs);
    },
    close(): Promise<void> {
      closed ??= (async () => {
        closing = true;
        await sendBounded((id) => ({ kind: 'close', id }), opts.shutdownTimeoutMs);
        // Unconditional: after the ack the thread is already ending (it closed
        // its port), and a worker that never acked must not outlive the runtime.
        await worker.terminate();
        goDown('dashboard database worker is closed');
      })();
      return closed;
    },
  };
}
