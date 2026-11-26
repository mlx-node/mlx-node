/**
 * The transport thread's side of the database worker.
 *
 * Its whole job is that no caller can hang. Every request is registered before
 * it is posted and settles exactly once — on the worker's reply, on a
 * structured-clone refusal (which `postMessage` throws synchronously), or on the
 * worker dying. A `call` that cannot be answered comes back as a failure
 * ENVELOPE rather than a rejection, so `runtime.call`'s "never rejects" contract
 * holds whatever happens to the thread.
 *
 * A worker that dies, or that answers a call neither with a reply nor with a
 * withdrawal acknowledgement in time, is replaced by a new one once its exit is
 * confirmed. Only the worker restarts: the process, its agents and their turns
 * do not notice beyond the calls that worker had open.
 */

import { MessageChannel, Worker } from 'node:worker_threads';

import { failedIngestSummary, type IngestSummary } from '../api/context.js';
import { failure, type ApiResponse } from '../api/errors.js';
import type { DbWorkerBoot, DbWorkerBootstrap, MainToWorker, WorkerToMain } from './protocol.js';

/**
 * A reply, or why none is coming: `undeliverable` means the request never left
 * this thread (the structured clone algorithm refused it — a caller-side fault),
 * `down` means no worker can answer, `restarted` means the worker it was sent to
 * died and a new one took its place, and `aborted` means an upstream caller
 * withdrew the request before this client got its reply.
 */
type SendResult =
  | { ok: true; reply: WorkerToMain }
  | { ok: false; why: 'undeliverable' | 'down' | 'restarted' | 'aborted'; reason: string };

/** Why a worker was replaced: it threw, it exited, or a call got neither a reply nor a withdrawal ack in time. */
export type WorkerRestartReason = 'error' | 'exit' | 'no-ack';

export type DbWorkerLifecycle =
  /** The worker closed the SQLite handle. Ordered AFTER `downloads.shutdown()`. */
  | { type: 'db-closed' }
  /**
   * A worker died outside of shutdown and a new one was started (`attempt` counts the restarts). `route` is the
   * pattern of the call whose deadline ended it, for `no-ack`; `ageMs` that call's age.
   */
  | {
      type: 'worker-restarted';
      reason: WorkerRestartReason;
      route: string | null;
      ageMs: number | null;
      attempt: number;
    }
  /** Six quick deaths in a row: worker routes fail at once until a call arrives a minute later. */
  | {
      type: 'worker-gave-up';
      reason: WorkerRestartReason;
      route: string | null;
      ageMs: number | null;
      attempt: number;
    };

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
  /** Delays before each restart in a run of quick deaths. Default {@link WORKER_RESTART_DELAYS_MS}. */
  restartDelaysMs?: readonly number[];
  now?: () => number;
}

/**
 * Long on purpose. This is the backstop for a thread that will never answer, not
 * a latency target — a boot ingest over a large trace dir can hold the
 * worker for tens of seconds, and killing that caller would turn a slow scan into
 * a visible failure.
 */
export const DEFAULT_REQUEST_TIMEOUT_MS = 60_000;
/** Restart delays: the first death restarts at once; each later death within a minute of its start waits longer. */
export const WORKER_RESTART_DELAYS_MS: readonly number[] = [0, 5_000, 10_000, 20_000, 40_000];
/** A worker that dies within this long of its start counts toward giving up. */
export const WORKER_QUICK_DEATH_MS = 60_000;
/** Quick deaths in a row after which no timer restarts the worker. */
export const WORKER_GIVE_UP_DEATHS = 6;
/** After giving up, a call this long later starts one new worker. */
export const WORKER_RETRY_AFTER_GIVE_UP_MS = 60_000;

export interface DbWorkerCall {
  method: string;
  /** Path, optionally with a query string — the worker re-parses it. */
  path: string;
  body?: unknown;
  bodyError?: string;
  /** The route pattern, for the restart record and the oldest-call report. Never a concrete path. */
  route?: string;
}

/** The worker's queue right now: how many requests wait, how old the oldest is, and its route pattern. */
export interface WorkerQueue {
  wq: number;
  wbusyMs: number;
  wroute: string | null;
}

export interface DbWorkerClient {
  /**
   * Withdraw a call that has not started when `signal` aborts. A call already
   * executing gets the opportunity to return its real result; if the worker
   * produces neither result nor withdrawal acknowledgement by the hard bound,
   * it is terminated and replaced before the outcome-unknown failure is returned.
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
  /** Drain, close the database, end the thread. Idempotent and bounded. No restart follows. */
  close(): Promise<void>;
  /** The live worker's queue, or null when nothing waits. */
  queue(): WorkerQueue | null;
}

/** Mirrors `doIngest`'s never-throw contract: a failed rescan is a warning, not an error. */
function unavailableSummary(reason: string): IngestSummary {
  return failedIngestSummary(reason);
}

interface PendingRequest {
  settle: (result: SendResult) => void;
  route: string | null;
  startedAt: number;
}

interface Cause {
  reason: WorkerRestartReason;
  route: string | null;
  ageMs: number | null;
}

/** One worker thread and everything that dies with it. */
interface Instance {
  worker: Worker;
  /**
   * The withdrawal channel. A deadline starts cancellation, but the request
   * itself was posted the moment `send` ran and may be sitting in the worker's
   * message queue — so without this a timed-out `DELETE /api/models/:name`
   * answered the UI with `E_UNAVAILABLE` and then deleted the model anyway.
   *
   * It has to be a SECOND port. A cancellation posted on the main channel
   * queues BEHIND the request it is cancelling — the worker is blocked, which
   * is why the deadline fired at all — so it could only ever arrive too late.
   */
  withdrawals: MessageChannel;
  pending: Map<number, PendingRequest>;
  startedAt: number;
  /** Set once no further reply can arrive from this thread. */
  down: string | null;
  /** Why it died: set by `error`, or by a forced termination, before the exit that confirms it. */
  cause: Cause | null;
  forced: boolean;
  /** For a forced termination: when the call it hung on was posted, the end of its useful life. */
  stuckSince: number | null;
}

export function startDbWorker(opts: DbWorkerOptions): DbWorkerClient {
  const now = opts.now ?? Date.now;
  const delays = opts.restartDelaysMs ?? WORKER_RESTART_DELAYS_MS;
  const requestTimeoutMs = opts.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  let current: Instance | null = null;
  let closing = false;
  /** Deaths in a row, each within a minute of its start after the first. */
  let deaths = 0;
  let attempt = 0;
  let gaveUpAt: number | null = null;
  let respawn: ReturnType<typeof setTimeout> | null = null;
  let nextId = 1;

  const syncRef = (inst: Instance): void => {
    if (inst.pending.size > 0 && inst.down === null) inst.worker.ref();
    else inst.worker.unref();
  };

  const spawn = (): Instance => {
    const withdrawals = new MessageChannel();
    // Never a reason the process stays alive: the request's own `worker.ref()`
    // already bounds that, and this port carries nothing anyone waits for.
    withdrawals.port1.unref();
    const worker = new Worker(opts.workerUrl, {
      workerData: {
        dbPath: opts.dbPath,
        modelsDir: opts.modelsDir,
        tracesDir: opts.tracesDir,
        runtimeDir: opts.runtimeDir,
        cacheRoot: opts.cacheRoot,
        prefixCacheRoot: opts.prefixCacheRoot,
        agentHome: opts.agentHome,
        chatAgentHome: opts.chatAgentHome,
        chatIndexPath: opts.chatIndexPath,
        scratchRoot: opts.scratchRoot,
        withdrawPort: withdrawals.port2,
      } satisfies DbWorkerBoot,
      transferList: [withdrawals.port2],
    });
    const inst: Instance = {
      worker,
      withdrawals,
      pending: new Map(),
      startedAt: now(),
      down: null,
      cause: null,
      forced: false,
      stuckSince: null,
    };
    syncRef(inst);
    worker.on('error', () => {
      // `exit` always follows `error`; only the exit proves the thread can do no more work.
      inst.cause ??= { reason: 'error', route: null, ageMs: null };
    });
    worker.on('exit', () => {
      instanceDown(inst, inst.cause ?? { reason: 'exit', route: null, ageMs: null });
    });
    worker.on('message', (message: WorkerToMain) => {
      // The db-closed witness is reported even if nobody is waiting on the reply,
      // so a supervisor sees the handle go down in the right order.
      if (message.kind === 'closed') opts.onLifecycle({ type: 'db-closed' });
      // Undefined means the request already settled or the worker was declared
      // down. Dropping the late reply is the only correct move.
      inst.pending.get(message.id)?.settle({ ok: true, reply: message });
    });
    return inst;
  };

  /**
   * The thread is confirmed gone: settle its callers, then start a new one unless the client is closing. A queued
   * mutation died with the thread, so the callers are told it was not answered.
   */
  const instanceDown = (inst: Instance, cause: Cause): void => {
    if (inst.down !== null) return;
    const reason = closing ? 'dashboard database worker is closed' : 'the database worker restarted';
    inst.down = reason;
    const waiting = [...inst.pending.values()];
    inst.pending.clear();
    syncRef(inst);
    try {
      inst.withdrawals.port1.close();
    } catch {
      // Already closed.
    }
    for (const entry of waiting) entry.settle({ ok: false, why: closing ? 'down' : 'restarted', reason });
    if (current !== inst) return;
    current = null;
    if (closing) return;
    const at = now();
    // A hung worker dies a whole deadline after the call it hung on: its life ended when that call was posted.
    deaths = (inst.stuckSince ?? at) - inst.startedAt >= WORKER_QUICK_DEATH_MS ? 1 : deaths + 1;
    attempt += 1;
    if (deaths >= WORKER_GIVE_UP_DEATHS) {
      gaveUpAt = at;
      opts.onLifecycle({ type: 'worker-gave-up', ...cause, attempt });
      return;
    }
    const delay = delays[Math.min(deaths - 1, delays.length - 1)] ?? 0;
    if (delay <= 0) current = spawn();
    else {
      respawn = setTimeout(() => {
        respawn = null;
        if (!closing && current === null) current = spawn();
      }, delay);
      respawn.unref();
    }
    opts.onLifecycle({ type: 'worker-restarted', ...cause, attempt });
  };

  /** The live worker; after a give-up, one new worker for a call that comes a minute later. */
  const live = (): Instance | null => {
    if (current !== null || closing) return current;
    if (gaveUpAt !== null && now() - gaveUpAt >= WORKER_RETRY_AFTER_GIVE_UP_MS) {
      gaveUpAt = null;
      deaths = WORKER_GIVE_UP_DEATHS - 1;
      current = spawn();
    }
    return current;
  };

  const unavailable = (): string =>
    closing
      ? 'dashboard database worker is closed'
      : gaveUpAt !== null
        ? 'the database worker stopped restarting'
        : 'the database worker is restarting';

  /**
   * Post one request and settle exactly once — on the reply, on a clone refusal,
   * or once its worker is known down.
   *
   * Non-call deadlines (ingest/drain/close) retain the old bounded settlement.
   * A call deadline instead starts withdrawal and keeps the pending entry until
   * the worker acknowledges the skipped handler, returns its real response, or
   * is terminated and confirmed down after acknowledging neither.
   *
   * A worker that acknowledges neither is terminated before callers are failed:
   * reporting a timeout while leaving a queued destructive mutation alive is
   * precisely the split-brain outcome this protocol exists to prevent.
   */
  const send = (
    build: (id: number) => MainToWorker,
    ms: number = requestTimeoutMs,
    withdrawOnTimeout = false,
    signal?: AbortSignal,
    route: string | null = null,
  ): Promise<SendResult> => {
    const inst = live();
    if (inst === null || inst.down !== null) {
      return Promise.resolve({ ok: false, why: 'down', reason: inst?.down ?? unavailable() });
    }
    if (signal?.aborted) {
      return Promise.resolve({ ok: false, why: 'aborted', reason: 'request was cancelled before it was posted' });
    }
    const id = nextId++;
    const startedAt = now();
    return new Promise<SendResult>((resolve) => {
      let onAbort: (() => void) | undefined;
      let posted = false;
      let withdrawalReason: string | null = null;
      let terminating = false;
      let timer: ReturnType<typeof setTimeout>;
      // `instanceDown` clears the whole map before invoking the settlers, so this
      // cannot be a `pending.delete(id)` check — it would swallow that path.
      let settled = false;
      const settle = (result: SendResult): void => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (onAbort !== undefined) signal?.removeEventListener('abort', onAbort);
        inst.pending.delete(id);
        syncRef(inst);
        resolve(result);
      };
      const terminateForMissingAck = (): void => {
        if (settled || terminating) return;
        terminating = true;
        if (!inst.forced) {
          inst.forced = true;
          inst.cause = { reason: 'no-ack', route, ageMs: now() - startedAt };
          inst.stuckSince = startedAt;
        }
        // Wait for termination before settling: until then a live worker could
        // still reach the queued mutation we are retracting.
        void inst.worker
          .terminate()
          .then(() => instanceDown(inst, inst.cause!))
          .catch(() => {
            // A rejected terminate is not proof the worker stopped. Keep waiting
            // for its real response or eventual exit rather than report a
            // failure while it may still execute the request later.
          });
      };
      const requestWithdrawal = (reason: string): void => {
        if (settled || withdrawalReason !== null) return;
        withdrawalReason = reason;
        try {
          // The worker's pre-handler gate either samples this and acknowledges
          // `withdrawn`, or misses it and returns the real call response.
          inst.withdrawals.port1.postMessage(id);
        } catch {
          terminateForMissingAck();
        }
      };
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
        // acknowledgement one bounded shutdown-sized grace period, then replace
        // the worker before reporting failure.
        timer = setTimeout(terminateForMissingAck, opts.shutdownTimeoutMs);
        timer.unref?.();
      }, ms);
      timer.unref?.();
      inst.pending.set(id, { settle, route, startedAt });
      syncRef(inst);
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
        inst.worker.postMessage(build(id));
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
  current = spawn();

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
        request.route ?? null,
      );
      if (!result.ok) {
        if (result.why === 'undeliverable') {
          return failure('E_BAD_REQUEST', `Request cannot cross the worker boundary: ${result.reason}`);
        }
        if (result.why === 'restarted') {
          return failure('E_UNAVAILABLE', 'Dashboard database restarted; the request was not answered');
        }
        return failure('E_UNAVAILABLE', `Dashboard database is unavailable: ${result.reason}`);
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
      const inst = current;
      const result = await send((id) => ({ kind: 'drain', id }), opts.shutdownTimeoutMs);
      if (!result.ok) {
        // `send` also uses `why: "down"` for its own deadline. Only a worker
        // confirmed gone proves future work is impossible; a timeout while it
        // still lives leaves the queued drain live and must reject.
        if (inst === null || inst.down !== null) return;
        throw new Error(`Dashboard database worker did not confirm drain: ${result.reason}`);
      }
      if (result.reply.kind !== 'drained') {
        throw new Error(`Unexpected worker reply "${result.reply.kind}" for a drain`);
      }
    },
    close(): Promise<void> {
      closed ??= (async () => {
        closing = true;
        if (respawn !== null) {
          clearTimeout(respawn);
          respawn = null;
        }
        const inst = current;
        if (inst === null) return;
        await send((id) => ({ kind: 'close', id }), opts.shutdownTimeoutMs);
        // Unconditional: after the ack the thread is already ending (it closed
        // its port), and a worker that never acked must not outlive the runtime.
        await inst.worker.terminate();
        instanceDown(inst, { reason: 'exit', route: null, ageMs: null });
      })();
      return closed;
    },
    queue(): WorkerQueue | null {
      const inst = current;
      if (inst === null || inst.pending.size === 0) return null;
      let oldest: PendingRequest | null = null;
      for (const entry of inst.pending.values())
        if (oldest === null || entry.startedAt < oldest.startedAt) oldest = entry;
      return { wq: inst.pending.size, wbusyMs: now() - oldest!.startedAt, wroute: oldest!.route };
    },
  };
}
