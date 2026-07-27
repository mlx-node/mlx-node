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

export type MainToWorker =
  | { kind: 'call'; id: number; method: string; path: string; body?: unknown; bodyError?: string }
  | { kind: 'ingest'; id: number }
  /** Await the ingest chain to its fixpoint; the database stays OPEN. */
  | { kind: 'drain'; id: number }
  /** Drain, then close the database and end the message loop. */
  | { kind: 'close'; id: number };

export type WorkerToMain =
  | { kind: 'response'; id: number; response: ApiResponse }
  | { kind: 'ingested'; id: number; summary: IngestSummary }
  | { kind: 'drained'; id: number }
  /** Posted AFTER the SQLite handle is closed — the shutdown-ordering witness. */
  | { kind: 'closed'; id: number };
