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

import { MessageChannel, Worker } from 'node:worker_threads';

import type { IngestSummary } from '../api/context.js';
import { failure, type ApiResponse } from '../api/errors.js';
import type { DbWorkerBoot, DbWorkerBootstrap, MainToWorker, WorkerToMain } from './protocol.js';

/**
 * A reply, or why none is coming: `undeliverable` means the request never left
 * this thread (the structured clone algorithm refused it — a caller-side fault),
 * `down` means the worker cannot answer, and `aborted` means an upstream caller
 * withdrew the request before this client got its reply.
 */
type SendResult =
  | { ok: true; reply: WorkerToMain }
  | { ok: false; why: 'undeliverable' | 'down' | 'aborted'; reason: string };

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
  /**
   * Budget for one ordinary request. Defaults to {@link DEFAULT_REQUEST_TIMEOUT_MS}.
   *
   * Its job is boundedness, not responsiveness: a worker that is ALIVE but
   * blocked emits neither `error` nor `exit`, so nothing else can ever settle a
   * caller. Set it above the slowest legitimate query rather than near it.
   */
  requestTimeoutMs?: number;
  onLifecycle: (event: DbWorkerLifecycle) => void;
}

/**
 * Long on purpose. This is the backstop for a thread that will never answer, not
 * a latency target — a boot ingest over a large sessions tree can hold the
 * worker for tens of seconds, and killing that caller would turn a slow scan into
 * a visible failure.
 */
export const DEFAULT_REQUEST_TIMEOUT_MS = 60_000;

export interface DbWorkerCall {
  method: string;
  /** Path, optionally with a query string — the worker re-parses it. */
  path: string;
  body?: unknown;
  bodyError?: string;
}

export interface DbWorkerClient {
  /**
   * Withdraw a call that has not started when `signal` aborts. A call already
   * executing gets the opportunity to return its real result; if the worker
   * produces neither result nor withdrawal acknowledgement by the hard bound,
   * it is terminated before the outcome-unknown failure is returned.
   */
  call(request: DbWorkerCall, signal?: AbortSignal): Promise<ApiResponse>;
  ingest(): Promise<IngestSummary>;
  /**
   * Await the worker's ingest chain; the database stays open.
   * Rejects when a still-live worker cannot confirm the barrier within the
   * shutdown budget — resolving without that acknowledgement would falsely
   * report that queued ingest work had finished. A worker already known dead is
   * vacuously drained because it can execute no future work.
   */
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
  /**
   * The withdrawal channel. A deadline starts cancellation, but the request
   * itself was posted the moment `send` ran and may be sitting in the worker's
   * message queue — so without this a timed-out `DELETE /api/models/:name`
   * answered the UI with `E_UNAVAILABLE` and then deleted the model anyway.
   *
   * It has to be a SECOND port. A cancellation posted on the main channel
   * queues BEHIND the request it is cancelling — the worker is blocked, which
   * is why the deadline fired at all — so it could only ever arrive too late.
   * A dedicated port can be drained synchronously with `receiveMessageOnPort`
   * at the instant the worker picks the request up, which is the one moment
   * the answer matters.
   */
  const withdrawals = new MessageChannel();
  // Never a reason the process stays alive: the request's own `worker.ref()`
  // already bounds that, and this port carries nothing anyone waits for. The
  // worker's end needs no equivalent — `receiveMessageOnPort` never starts it,
  // so it holds no handle on that side either.
  withdrawals.port1.unref();

  const worker = new Worker(opts.workerUrl, {
    workerData: {
      dbPath: opts.dbPath,
      modelsDir: opts.modelsDir,
      sessionsRoot: opts.sessionsRoot,
      tracesDir: opts.tracesDir,
      cacheRoot: opts.cacheRoot,
      withdrawPort: withdrawals.port2,
    } satisfies DbWorkerBoot,
    transferList: [withdrawals.port2],
  });

  const pending = new Map<number, (result: SendResult) => void>();
  let nextId = 1;
  /** Set once no further reply can arrive; every later send fails fast. */
  let down: string | null = null;
  let closing = false;
  /** Specific safety reason to preserve across the terminate → exit event race. */
  let forcedDownReason: string | null = null;

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

  worker.on('error', (err: Error) => {
    // Once forced termination starts, only confirmed exit/terminate completion
    // proves the worker can perform no future work. An error alone is not that
    // witness, so preserve the outcome-unknown reason and wait for exit.
    if (forcedDownReason === null) goDown(`dashboard database worker failed: ${err.message}`);
  });
  worker.on('exit', (code: number) => goDown(forcedDownReason ?? `dashboard database worker exited (code ${code})`));

  worker.on('message', (message: WorkerToMain) => {
    // The db-closed witness is reported even if nobody is waiting on the reply,
    // so a supervisor sees the handle go down in the right order.
    if (message.kind === 'closed') opts.onLifecycle({ type: 'db-closed' });
    // Undefined means the request already settled or the worker was declared
    // down. Dropping the late reply is the only correct move.
    // `settle` owns the map and ref bookkeeping, so nothing is done here.
    pending.get(message.id)?.({ ok: true, reply: message });
  });

  /**
   * Post one request and settle exactly once — on the reply, on a clone refusal,
   * or once the worker is known down.
   *
   * Non-call deadlines (ingest/drain/close) retain the old bounded settlement.
   * A call deadline instead starts withdrawal and keeps the pending entry until
   * the worker acknowledges the skipped handler, returns its real response, or
   * is terminated and confirmed down after acknowledging neither.
   *
   * `withdrawOnTimeout` turns the deadline into a cancellation handshake for
   * `call`. The worker answers either with `withdrawn` from the exact pre-handler
   * gate (the handler never began) or with the call's real response (withdrawal
   * lost the cross-port race). Only that acknowledgement/result settles the
   * caller.
   *
   * A worker that acknowledges neither is terminated before callers are failed:
   * reporting a timeout while leaving a queued destructive mutation alive is
   * precisely the split-brain outcome this protocol exists to prevent.
   */
  const requestTimeoutMs = opts.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  const send = (
    build: (id: number) => MainToWorker,
    ms: number = requestTimeoutMs,
    withdrawOnTimeout = false,
    signal?: AbortSignal,
  ): Promise<SendResult> => {
    if (down !== null) return Promise.resolve({ ok: false, why: 'down', reason: down });
    if (signal?.aborted) {
      return Promise.resolve({ ok: false, why: 'aborted', reason: 'request was cancelled before it was posted' });
    }
    const id = nextId++;
    return new Promise<SendResult>((resolve) => {
      let onAbort: (() => void) | undefined;
      let posted = false;
      let withdrawalReason: string | null = null;
      let terminating = false;
      let timer: ReturnType<typeof setTimeout>;
      // `goDown` clears the whole map before invoking the settlers, so this
      // cannot be a `pending.delete(id)` check — it would swallow that path.
      let settled = false;
      const settle = (result: SendResult): void => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (onAbort !== undefined) signal?.removeEventListener('abort', onAbort);
        pending.delete(id);
        syncRef();
        resolve(result);
      };
      const terminateForMissingAck = (): void => {
        if (settled || terminating) return;
        terminating = true;
        const reason = `worker terminated because the cancellation outcome for request ${id} remained unknown`;
        forcedDownReason ??= reason;
        // Wait for termination/exit before settling through goDown. Until then a
        // live worker could still reach the queued mutation we are retracting.
        void worker.terminate().then(() => goDown(forcedDownReason!)).catch(() => {
          // A rejected terminate is not proof the worker stopped. Keep waiting
          // for its real response or eventual error/exit rather than report a
          // failure while it may still execute the request later.
        });
      };
      const requestWithdrawal = (reason: string): void => {
        if (settled || withdrawalReason !== null) return;
        withdrawalReason = reason;
        try {
          // The worker's pre-handler gate either samples this and acknowledges
          // `withdrawn`, or misses it and returns the real call response.
          withdrawals.port1.postMessage(id);
        } catch {
          terminateForMissingAck();
        }
      };
      // Never the reason the process stays up: `syncRef` already refs the worker
      // for exactly as long as this request is outstanding.
      timer = setTimeout(() => {
        if (!withdrawOnTimeout) {
          settle({ ok: false, why: 'down', reason: `timed out after ${ms}ms` });
          return;
        }
        if (withdrawalReason !== null) {
          // An upstream deadline requested cancellation earlier. The worker has
          // now consumed this request's full ordinary budget without producing
          // either a response or the withdrawal acknowledgement.
          terminateForMissingAck();
          return;
        }
        requestWithdrawal(`timed out after ${ms}ms`);
        if (settled || terminating) return;
        // The ordinary deadline itself initiated withdrawal. Give its
        // acknowledgement one bounded shutdown-sized grace period, then remove
        // the worker before reporting failure.
        timer = setTimeout(terminateForMissingAck, opts.shutdownTimeoutMs);
        timer.unref?.();
      }, ms);
      timer.unref?.();
      pending.set(id, settle);
      syncRef();
      if (signal !== undefined) {
        onAbort = () => {
          // A signal caught before the original post needs no handshake: there
          // is no worker request to withdraw. Once posted, keep the caller
          // pending until the worker acknowledges a skip or returns the real
          // result of work that had already started.
          if (!posted) {
            settle({ ok: false, why: 'aborted', reason: 'request was cancelled before it was posted' });
            return;
          }
          requestWithdrawal('request was cancelled');
        };
        signal.addEventListener('abort', onAbort, { once: true });
        // Close the check→listen race. In particular, do not post below when a
        // signal became aborted while its listener was being installed.
        if (signal.aborted) onAbort();
      }
      if (settled) return;
      try {
        worker.postMessage(build(id));
        posted = true;
      } catch (err) {
        // A payload the structured clone algorithm refuses throws HERE, before
        // the worker ever sees it. Settling now is what keeps the caller from
        // waiting on a request that was never delivered.
        settle({ ok: false, why: 'undeliverable', reason: err instanceof Error ? err.message : String(err) });
      }
    });
  };

  let closed: Promise<void> | null = null;

  return {
    async call(request: DbWorkerCall, signal?: AbortSignal): Promise<ApiResponse> {
      const result = await send(
        (id) => ({
          kind: 'call',
          id,
          method: request.method,
          path: request.path,
          body: request.body,
          ...(request.bodyError !== undefined ? { bodyError: request.bodyError } : {}),
        }),
        requestTimeoutMs,
        true,
        signal,
      );
      if (!result.ok) {
        return result.why === 'undeliverable'
          ? failure('E_BAD_REQUEST', `Request cannot cross the worker boundary: ${result.reason}`)
          : failure('E_UNAVAILABLE', `Dashboard database is unavailable: ${result.reason}`);
      }
      if (result.reply.kind === 'withdrawn') {
        return failure('E_UNAVAILABLE', 'Dashboard database request was cancelled before it started');
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
      const result = await send((id) => ({ kind: 'drain', id }), opts.shutdownTimeoutMs);
      if (!result.ok) {
        // `send` also uses `why: "down"` for its own deadline. Only the global
        // latch proves exit/error already made future work impossible; a timeout
        // while `down` is still null leaves the queued drain live and must reject.
        if (result.why === 'down' && down !== null) return;
        throw new Error(`Dashboard database worker did not confirm drain: ${result.reason}`);
      }
      if (result.reply.kind !== 'drained') {
        throw new Error(`Unexpected worker reply "${result.reply.kind}" for a drain`);
      }
    },
    close(): Promise<void> {
      closed ??= (async () => {
        closing = true;
        await send((id) => ({ kind: 'close', id }), opts.shutdownTimeoutMs);
        // Unconditional: after the ack the thread is already ending (it closed
        // its port), and a worker that never acked must not outlive the runtime.
        await worker.terminate();
        withdrawals.port1.close();
        goDown('dashboard database worker is closed');
      })();
      return closed;
    },
  };
}
