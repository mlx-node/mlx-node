/**
 * Dashboard HTTP server: static SPA + JSON `/api` + SSE, over plain `node:http`.
 *
 * Lifecycle and routing live here; the route handlers are in `api.ts` and the
 * SPA file serving in `static.ts`. The server binds `127.0.0.1` by default and
 * guards every mutating (non-GET) request with a local-origin check to block
 * drive-by CSRF / DNS-rebinding against the loopback port. It never links the
 * native addon — all data comes from disk via the C1–C4 modules.
 */

import { mkdirSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { handleApiRequest, type ApiDeps, type IngestSummary, type SseClient } from './api.js';
import { openDashboardDb, type DashboardDb } from './db/open.js';
import { DownloadManager } from './download.js';
import { ingestSessions } from './ingest/sessions.js';
import { ingestTraces } from './ingest/traces.js';
import { defaultModelsDir } from './models.js';
import { agentSessionsRoot, dashboardDbPath, metricsTraceDir } from './paths.js';
import { serveStatic } from './static.js';

export interface DashboardServerOptions {
  port?: number;
  host?: string;
  dbPath?: string;
  modelsDir?: string;
  sessionsRoot?: string;
  tracesDir?: string;
  cacheRoot?: string;
  webRoot?: string;
}

export interface DashboardServer {
  url: string;
  port: number;
  close(): Promise<void>;
}

/** Default port; matches `mlx dashboard --port 6590`. */
const DEFAULT_PORT = 6590;
/** Trace-file retention passed to the periodic + boot ingest. */
const RETENTION_DAYS = 30;
/** Incremental rescan cadence while the server runs. */
const INGEST_INTERVAL_MS = 30_000;

const LOCAL_HOSTNAMES = new Set(['localhost', '127.0.0.1', '::1']);

/** Split a `Host`/URL authority into hostname + optional port, handling `[::1]`. */
function splitHostPort(authority: string): { hostname: string; port: string } {
  if (authority.startsWith('[')) {
    const end = authority.indexOf(']');
    if (end === -1) return { hostname: authority, port: '' };
    const hostname = authority.slice(1, end);
    const rest = authority.slice(end + 1);
    return { hostname, port: rest.startsWith(':') ? rest.slice(1) : '' };
  }
  const colon = authority.indexOf(':');
  // A bare IPv6 literal (multiple colons, unbracketed) has no port component.
  if (colon === -1 || authority.indexOf(':', colon + 1) !== -1) return { hostname: authority, port: '' };
  return { hostname: authority.slice(0, colon), port: authority.slice(colon + 1) };
}

/** Whether an authority names a loopback host, optionally carrying the server port. */
function isLocalAuthority(authority: string | undefined, port: number): boolean {
  if (authority === undefined || authority === '') return false;
  const { hostname, port: hostPort } = splitHostPort(authority);
  if (!LOCAL_HOSTNAMES.has(hostname)) return false;
  return hostPort === '' || hostPort === String(port);
}

/**
 * Local-origin guard for mutating requests. The `Host` header must name a
 * loopback host (optionally with the server's port); when an `Origin` header is
 * present it must additionally be an `http` origin of a loopback host. GET/HEAD
 * are unguarded (static assets + read-only endpoints).
 */
function isRequestAllowed(req: IncomingMessage, port: number): boolean {
  if (!isLocalAuthority(req.headers.host, port)) return false;
  const origin = req.headers.origin;
  if (origin === undefined || origin === '' || origin === 'null') {
    // No Origin (curl-style / same-origin GET-less clients) is allowed once Host passed.
    return origin !== 'null';
  }
  let parsed: URL;
  try {
    parsed = new URL(origin);
  } catch {
    return false;
  }
  if (parsed.protocol !== 'http:') return false;
  return isLocalAuthority(parsed.host, port);
}

function defaultWebRoot(): string {
  // dist/server.js → ../web (the built SPA shipped in package `files`).
  return fileURLToPath(new URL('../web', import.meta.url));
}

export async function startDashboardServer(opts: DashboardServerOptions = {}): Promise<DashboardServer> {
  const host = opts.host ?? '127.0.0.1';
  const requestedPort = opts.port ?? DEFAULT_PORT;
  const dbPath = opts.dbPath ?? dashboardDbPath();
  const modelsDir = opts.modelsDir ?? defaultModelsDir();
  const sessionsRoot = opts.sessionsRoot ?? agentSessionsRoot();
  const tracesDir = opts.tracesDir ?? metricsTraceDir();
  const cacheRoot = opts.cacheRoot;
  const webRoot = opts.webRoot ?? defaultWebRoot();

  if (host !== '127.0.0.1' && host !== 'localhost' && host !== '::1') {
    console.warn(
      `[mlx-dashboard] binding to non-loopback host "${host}"; the dashboard has no auth and exposes local models, sessions, and cache to anyone who can reach this address.`,
    );
  }

  if (dbPath !== ':memory:') {
    mkdirSync(dirname(dbPath), { recursive: true });
  }
  const dash: DashboardDb = openDashboardDb(dbPath);
  const downloads = new DownloadManager({ modelsDir });
  const sseClients = new Set<SseClient>();

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
        traces: { files: 0, records: 0, pruned: 0 },
      };
    }
  };
  let ingestChain: Promise<IngestSummary> = Promise.resolve({
    sessions: { scanned: 0, updated: 0, removed: 0, warnings: [] },
    traces: { files: 0, records: 0, pruned: 0 },
  });
  const runIngest = (): Promise<IngestSummary> => {
    ingestChain = ingestChain.then(doIngest);
    return ingestChain;
  };

  const deps: ApiDeps = { dash, modelsDir, sessionsRoot, tracesDir, cacheRoot, downloads, runIngest, sseClients };

  // The actual bound port (may differ from requested when 0 is used).
  let boundPort = requestedPort;

  const server: Server = createServer((req: IncomingMessage, res: ServerResponse) => {
    void handle(req, res);
  });

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const method = req.method ?? 'GET';
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);

    // Guard every mutating request; reads (GET/HEAD) are unguarded.
    if (method !== 'GET' && method !== 'HEAD' && !isRequestAllowed(req, boundPort)) {
      res.writeHead(403, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: 'Forbidden: request must originate from a local origin' }));
      return;
    }

    try {
      const handled = await handleApiRequest(req, res, url, deps);
      if (!handled) serveStatic(req, res, webRoot, url.pathname);
    } catch (err) {
      if (!res.headersSent) {
        res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : 'Internal server error' }));
      } else {
        res.end();
      }
    }
  }

  await new Promise<void>((resolveListen, reject) => {
    const onError = (err: Error): void => {
      server.removeListener('error', onError);
      reject(err);
    };
    server.on('error', onError);
    server.listen(requestedPort, host, () => {
      server.removeListener('error', onError);
      const address = server.address();
      if (address !== null && typeof address === 'object') boundPort = address.port;
      resolveListen();
    });
  });

  // Ingest on boot (non-blocking) plus a periodic rescan.
  runIngest().catch(() => {});
  const ingestTimer = setInterval(() => {
    runIngest().catch(() => {});
  }, INGEST_INTERVAL_MS);
  ingestTimer.unref();

  const displayHost = host === '::1' ? '[::1]' : host;
  const url = `http://${displayHost}:${boundPort}`;

  return {
    url,
    port: boundPort,
    async close() {
      clearInterval(ingestTimer);
      for (const client of sseClients) {
        try {
          client.cleanup();
          client.res.end();
        } catch {
          // A client already torn down: ignore.
        }
      }
      sseClients.clear();
      await new Promise<void>((resolveClose, reject) => {
        server.close((err) => {
          if (err) reject(err);
          else resolveClose();
        });
      });
      dash.close();
    },
  };
}
