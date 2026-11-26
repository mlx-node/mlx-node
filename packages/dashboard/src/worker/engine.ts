import { AgentSessionsReader } from '../agent-sessions/reader.js';
import type { IngestPassResult, IngestSummary, WorkerApiContext } from '../api/context.js';
import { dispatchWorker } from '../api/dispatch.js';
import { toFailure, type ApiResponse } from '../api/errors.js';
import { readWorkIndexSync } from '../chat/index-file.js';
import { openDashboardDb } from '../db/open.js';
import { ingestRuntime } from '../ingest/runtime.js';
import { ingestTraces } from '../ingest/traces.js';
import { closePrefixCacheWalks } from '../prefix-cache.js';
import type { DbWorkerBootstrap } from './protocol.js';

/** Trace-file retention passed to the periodic + boot ingest. */
const RETENTION_DAYS = 30;

export interface DbWorkerEngineCall {
  method: string;
  path: string;
  body?: unknown;
  bodyError?: string;
}

export interface DbWorkerEngine {
  call(message: DbWorkerEngineCall): Promise<ApiResponse>;
  ingest(): Promise<IngestSummary>;
  drain(): Promise<void>;
  close(): Promise<void>;
}

/**
 * Core SQLite + synchronous-FS dispatch engine owned by the database worker
 * thread in production and optionally run in-process by API tests.
 */
export function createDbWorkerEngine(boot: DbWorkerBootstrap): DbWorkerEngine {
  const {
    dbPath,
    modelsDir,
    tracesDir,
    runtimeDir,
    cacheRoot,
    prefixCacheRoot,
    agentHome,
    chatAgentHome,
    chatIndexPath,
    scratchFolders,
  } = boot;

  const dash = openDashboardDb(dbPath);

  // Serialize ingests so overlapping timer/boot/manual runs never interleave
  // SQLite transactions. `doIngest` never throws, so a failed run cannot poison
  // the chain for later callers (a rejected promise would skip every queued
  // `.then`).
  const doIngest = async (): Promise<IngestSummary> => {
    let traces: IngestPassResult;
    let runtime: IngestPassResult;
    try {
      traces = await ingestTraces(dash, tracesDir, { retentionDays: RETENTION_DAYS });
    } catch (err) {
      traces = { files: 0, records: 0, pruned: 0, warnings: [String(err)] };
    }
    try {
      runtime = await ingestRuntime(dash, runtimeDir);
    } catch (err) {
      runtime = { files: 0, records: 0, pruned: 0, warnings: [String(err)] };
    }
    return { traces, runtime };
  };

  let ingestChain: Promise<IngestSummary> = Promise.resolve({
    traces: { files: 0, records: 0, pruned: 0, warnings: [] },
    runtime: { files: 0, records: 0, pruned: 0, warnings: [] },
  });

  const runIngest = (): Promise<IngestSummary> => {
    ingestChain = ingestChain.then(doIngest);
    return ingestChain;
  };

  /**
   * Await the ingest chain to its FIXPOINT, re-reading after each await: a caller
   * can queue another rescan while we wait.
   */
  const drainIngest = async (): Promise<void> => {
    let awaited: Promise<IngestSummary> | null = null;
    while (awaited !== ingestChain) {
      awaited = ingestChain;
      await awaited;
    }
  };

  const ctx: WorkerApiContext = {
    dash,
    modelsDir,
    tracesDir,
    runtimeDir,
    cacheRoot,
    prefixCacheRoot,
    agentHome,
    runIngest,
    agentSessions: new AgentSessionsReader({ agentHome }),
    chatSessions: new AgentSessionsReader({ agentHome: chatAgentHome, origin: 'chat' }),
    chatIndex: () => readWorkIndexSync(chatIndexPath),
    scratchFolders,
  };

  let closed = false;

  return {
    async call(message: DbWorkerEngineCall): Promise<ApiResponse> {
      try {
        // A constant, trusted base: only the path + query of `message.path` are used.
        const url = new URL(message.path, 'http://localhost');
        return await dispatchWorker(ctx, {
          method: message.method,
          pathname: url.pathname,
          query: url.searchParams,
          body: message.body,
          ...(message.bodyError !== undefined ? { bodyError: message.bodyError } : {}),
        });
      } catch (err) {
        return toFailure(err);
      }
    },
    ingest: runIngest,
    drain: drainIngest,
    async close(): Promise<void> {
      if (closed) return;
      closed = true;
      await drainIngest();
      dash.close();
      closePrefixCacheWalks();
    },
  };
}
