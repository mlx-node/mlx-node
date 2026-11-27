import { failedIngestSummary, type IngestSummary } from '../api/context.js';
import { failure, type ApiResponse } from '../api/errors.js';
import type { DbWorkerCall, DbWorkerClient, DbWorkerLifecycle, WorkerQueue } from './client.js';
import { createDbWorkerEngine } from './engine.js';
import type { DbWorkerBootstrap } from './protocol.js';

export interface InProcessDbWorkerOptions extends DbWorkerBootstrap {
  onLifecycle: (event: DbWorkerLifecycle) => void;
}

/**
 * In-process {@link DbWorkerClient} backed directly by {@link createDbWorkerEngine}.
 *
 * Used by tests (`inProcessWorker: true`) to avoid spawning a fresh TS-stripping
 * Node worker thread per test case while keeping identical route dispatch,
 * ingest serialization, and lifecycle semantics.
 */
export function startInProcessDbWorker(opts: InProcessDbWorkerOptions): DbWorkerClient {
  const engine = createDbWorkerEngine(opts);
  // Mirror `db-worker.ts`'s boot ingest; serialize subsequent calls behind it
  // just like the single-threaded worker event loop does.
  let queueChain: Promise<void> = engine.ingest().then(
    () => undefined,
    () => undefined,
  );
  let closed = false;
  let closing: Promise<void> | null = null;
  const pending = new Set<{ route: string | null; startedAt: number }>();

  return {
    async call(request: DbWorkerCall, signal?: AbortSignal): Promise<ApiResponse> {
      if (closed || closing !== null) return failure('E_UNAVAILABLE', 'Database worker is closed');
      if (signal?.aborted) return failure('E_UNAVAILABLE', 'Request aborted before dispatch');
      const entry = { route: request.route ?? null, startedAt: Date.now() };
      pending.add(entry);
      const run = queueChain.then(async (): Promise<ApiResponse> => {
        try {
          if (closed) return failure('E_UNAVAILABLE', 'Database worker is closed');
          if (signal?.aborted) return failure('E_UNAVAILABLE', 'Request aborted before dispatch');
          return await engine.call(request);
        } finally {
          pending.delete(entry);
        }
      });
      queueChain = run.then(
        () => undefined,
        () => undefined,
      );
      return await run;
    },

    async ingest(): Promise<IngestSummary> {
      if (closed || closing !== null) return failedIngestSummary('Database worker is closed');
      const run = queueChain.then(() => engine.ingest());
      queueChain = run.then(
        () => undefined,
        () => undefined,
      );
      return await run;
    },

    async drain(): Promise<void> {
      if (closed) return;
      await queueChain;
      await engine.drain();
    },

    async close(): Promise<void> {
      if (closing !== null) return await closing;
      closing = (async () => {
        await queueChain;
        await engine.close();
        closed = true;
        opts.onLifecycle({ type: 'db-closed' });
      })();
      return await closing;
    },

    queue(): WorkerQueue | null {
      const oldest = pending.values().next().value;
      if (oldest === undefined) return null;
      return {
        wq: pending.size,
        wbusyMs: Math.max(0, Date.now() - oldest.startedAt),
        wroute: oldest.route,
      };
    },
  };
}
