/**
 * The message contract between the dashboard runtime (transport thread) and its
 * database worker.
 *
 * Every field here crosses a `postMessage`, so it must survive the structured
 * clone algorithm. `ApiResponse` already is a plain envelope, which is what makes
 * this boundary cheap. `URLSearchParams` is NOT clonable, so a call carries the
 * raw path-with-query and the worker re-parses it — the same string the caller
 * handed `runtime.call`.
 *
 * Every request carries an `id` and gets exactly one reply bearing it; that
 * pairing is what lets a dead worker settle every in-flight caller instead of
 * leaving them hanging.
 */

import type { MessagePort } from 'node:worker_threads';

import type { IngestSummary } from '../api/context.js';
import type { ApiResponse } from '../api/errors.js';

/** Everything the worker needs to open the database and answer its routes. */
export interface DbWorkerBootstrap {
  dbPath: string;
  modelsDir: string;
  sessionsRoot: string;
  tracesDir: string;
  cacheRoot: string | undefined;
}

/**
 * The actual `workerData`: the caller's bootstrap plus the one thing the client
 * owns rather than receives.
 *
 * `withdrawPort` is a second, transferred `MessagePort` carrying nothing but
 * ids the transport thread has given up waiting for. It is separate from the
 * request channel because a withdrawal sent down the request channel would
 * queue behind the very request it withdraws — see `client.ts`. Optional, so a
 * test fixture may be started with a plain {@link DbWorkerBootstrap}; the
 * worker then simply has nothing to withdraw.
 */
export interface DbWorkerBoot extends DbWorkerBootstrap {
  withdrawPort?: MessagePort;
}

export type MainToWorker =
  | { kind: 'call'; id: number; method: string; path: string; body?: unknown; bodyError?: string }
  | { kind: 'ingest'; id: number }
  /** Await the ingest chain to its fixpoint; the database stays OPEN. */
  | { kind: 'drain'; id: number }
  /** Drain, then close the database and end the message loop. */
  | { kind: 'close'; id: number };

export type WorkerToMain =
  | { kind: 'response'; id: number; response: ApiResponse }
  /** The call was skipped before its handler began; this is the withdrawal acknowledgement. */
  | { kind: 'withdrawn'; id: number }
  | { kind: 'ingested'; id: number; summary: IngestSummary }
  | { kind: 'drained'; id: number }
  /** Posted AFTER the SQLite handle is closed — the shutdown-ordering witness. */
  | { kind: 'closed'; id: number };
