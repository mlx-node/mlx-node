/**
 * Dashboard HTTP server: static SPA + JSON `/api` + SSE, over plain `node:http`.
 *
 * Lifecycle and routing live here; the route handlers are in `api.ts` and the
 * SPA file serving in `static.ts`. The server binds `127.0.0.1` by default and
 * validates a loopback `Host` on EVERY request (reads included) to block
 * DNS-rebinding against the loopback port, additionally requiring a local
 * `Origin` on mutations to block drive-by CSRF. It never links the native
 * addon — all data comes from disk via the C1–C4 modules.
 */

import { mkdirSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { networkInterfaces } from 'node:os';
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

/**
 * Wrap a bare IPv6 literal in `[...]` so it is safe inside a URL authority. Any
 * host carrying a `:` that is not already bracketed is an IPv6 literal (`::1`,
 * `2001:db8::1`, or a scoped `fe80::1%en0`); an unbracketed one makes the trailing
 * `:<port>` ambiguous and `new URL()` reject it. IPv4/hostnames (no `:`) and
 * already-bracketed literals pass through unchanged. A scoped literal is bracketed
 * too: `new URL` still rejects the `%`, but the printed string stays well-formed
 * rather than crashing the caller.
 */
export function bracketHost(host: string): string {
  return host.includes(':') && !host.startsWith('[') ? `[${host}]` : host;
}

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

/**
 * Rewrite a bind host into the exact spelling a request carries by the time it is
 * compared, i.e. after the same `new URL(...)` authority parse `classifyHost` runs.
 * That parser does more than ASCII-lowercase a hostname: it also compresses IPv6
 * literals and re-spells IPv4-mapped ones, so a host kept verbatim here
 * (`2001:0db8:0000:0000:0000:0000:0000:0001`, `::ffff:127.0.0.1`) can never equal the
 * `2001:db8::1` / `::ffff:7f00:1` a browser sends, and a server that bound fine
 * answers 403 to every request.
 *
 * A host the parser rejects (a scoped `fe80::1%en0`) or one carrying anything beyond
 * a bare host keeps its lowercased spelling: no request authority normalizes to those
 * either, so the fallback can only leave the allowlist narrower, never wider — the
 * `href` check is what stops a `--host evil.example/x` from being lifted into the
 * usable entry `evil.example`.
 */
function canonicalHost(host: string): string {
  try {
    const url = new URL(`http://${bracketHost(host)}`);
    if (url.port === '' && url.href === `http://${url.host}/`) return splitHostPort(url.host).hostname;
  } catch {
    // Not a parseable authority; the raw spelling is the best available.
  }
  return host.toLowerCase();
}

/**
 * Build the Host/Origin allowlist for a bind. A concrete host allows the
 * loopback names plus that exact host. A wildcard bind (`0.0.0.0`, `::`, or the
 * empty string) cannot allowlist the literal wildcard — no browser ever sends it
 * as `Host` — so it enumerates the machine's real interface addresses instead: a
 * LAN client's `Host: <iface-ip>:<port>` then matches while a rebound
 * `Host: evil.example` (no matching local IP) is still rejected, preserving the
 * DNS-rebinding defense. `networkInterfaces()` is a boot snapshot, adequate for a
 * local tool; IPv6 zone ids (`fe80::1%en0`) never appear in a browser Host, so
 * link-local entries are inert. Configured and enumerated hosts alike go through
 * `canonicalHost`, so neither side of the comparison can drift from the other.
 */
export function bindAllowedHosts(host: string): Set<string> {
  const wildcard = host === '0.0.0.0' || host === '::' || host === '';
  if (!wildcard) return new Set([...LOCAL_HOSTNAMES, canonicalHost(host)]);
  const hosts = new Set<string>(LOCAL_HOSTNAMES);
  for (const addrs of Object.values(networkInterfaces())) {
    for (const a of addrs ?? []) hosts.add(canonicalHost(a.address));
  }
  return hosts;
}

/**
 * Whether an authority names an allowed host on the server's own port. An
 * omitted port is `http`'s default 80, so it only matches when the server binds
 * 80 — otherwise an `Origin: http://127.0.0.1` (a page on loopback port 80) would
 * be treated as same-origin with a dashboard on 6590 and could drive-by a
 * mutating POST. `new URL(...).host` already strips a redundant `:80`, so a
 * port-80 bind still sees the omitted-port form here.
 */
function isAllowedAuthority(authority: string | undefined, port: number, allowedHosts: Set<string>): boolean {
  if (authority === undefined || authority === '') return false;
  // Normalize HERE, not at one caller: the allowlist holds canonical spellings, so
  // any caller comparing a raw header against it drifts from the ones that parse
  // first. That split let a `Host: LOCALHOST:6590` or an expanded `[0:0:0:0:0:0:0:1]`
  // pass every read and 403 every mutation from the same authority. Idempotent for an
  // authority already through `new URL`, so the Origin and `classifyHost` paths are
  // unchanged.
  let normalized: string;
  try {
    normalized = new URL(`http://${authority}`).host;
  } catch {
    return false;
  }
  const { hostname, port: hostPort } = splitHostPort(normalized);
  if (!allowedHosts.has(hostname)) return false;
  return hostPort === String(port) || (hostPort === '' && port === 80);
}

type HostVerdict = 'allowed' | 'foreign' | 'malformed';

/**
 * Classify the untrusted `Host` header against the allowlist. A syntactically
 * invalid authority (e.g. `[`) is `malformed` (→ 400) rather than throwing; a
 * well-formed but non-loopback host is `foreign` (→ 403). Validating this on
 * EVERY request — including reads — blocks DNS-rebinding: an attacker page
 * rebound to the loopback port still carries its own hostname in `Host`.
 */
function classifyHost(authority: string | undefined, port: number, allowedHosts: Set<string>): HostVerdict {
  if (authority === undefined || authority === '') return 'foreign';
  let normalizedHost: string;
  try {
    // Reject a malformed authority here instead of letting an unguarded
    // `new URL` throw an unhandled rejection later; `.host` also strips any
    // userinfo/path smuggled into the header so only the real host is matched.
    normalizedHost = new URL(`http://${authority}`).host;
  } catch {
    return 'malformed';
  }
  return isAllowedAuthority(normalizedHost, port, allowedHosts) ? 'allowed' : 'foreign';
}

/**
 * Cross-origin guard for mutating requests, layered on top of the per-request
 * `Host` check. When an `Origin` header is present it must be an `http` origin
 * of an allowed (loopback) host; a missing Origin (curl-style clients) passes.
 */
function isRequestAllowed(req: IncomingMessage, port: number, allowedHosts: Set<string>): boolean {
  if (!isAllowedAuthority(req.headers.host, port, allowedHosts)) return false;
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
  return isAllowedAuthority(parsed.host, port, allowedHosts);
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

  const deps: ApiDeps = { dash, modelsDir, sessionsRoot, tracesDir, cacheRoot, downloads, runIngest, sseClients };

  // The actual bound port (may differ from requested when 0 is used).
  let boundPort = requestedPort;

  // Hosts accepted in the `Host`/`Origin` allowlist: the loopback names plus the
  // explicitly configured bind host (or, for a wildcard bind, the machine's real
  // interface addresses), so an intentional (warned) non-loopback bind stays
  // reachable without weakening the loopback default.
  const allowedHosts = bindAllowedHosts(host);

  const server: Server = createServer((req: IncomingMessage, res: ServerResponse) => {
    handle(req, res).catch((err: unknown) => {
      // Terminal safety net: no request handler may reject unhandled (which
      // would crash the process). Respond 500 if nothing was sent yet.
      console.error('[mlx-dashboard] unhandled request error', err);
      if (!res.headersSent) {
        res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: 'Internal server error' }));
      } else {
        try {
          res.end();
        } catch {
          // Socket already destroyed: nothing to do.
        }
      }
    });
  });

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const method = req.method ?? 'GET';

    // Validate the untrusted Host on EVERY request (reads included) before
    // routing, so DNS-rebinding cannot reach read handlers.
    const verdict = classifyHost(req.headers.host, boundPort, allowedHosts);
    if (verdict === 'malformed') {
      res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: 'Bad Request: malformed Host header' }));
      return;
    }
    if (verdict === 'foreign') {
      res.writeHead(403, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: 'Forbidden: request must originate from a local origin' }));
      return;
    }

    // Parse the request target against a constant, trusted base so a malformed
    // target yields 400 instead of throwing; the Host is never used here.
    let url: URL;
    try {
      url = new URL(req.url ?? '/', 'http://localhost');
    } catch {
      res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: 'Bad Request: malformed request target' }));
      return;
    }

    // Mutations additionally require a local Origin (CSRF defense).
    if (method !== 'GET' && method !== 'HEAD' && !isRequestAllowed(req, boundPort, allowedHosts)) {
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

  // A wildcard bind has no connectable literal host: advertise a loopback the
  // Host allowlist accepts. `::` binds IPv6 → `[::1]`; `0.0.0.0`/'' bind IPv4 →
  // `127.0.0.1`. Both are in LOCAL_HOSTNAMES, so the printed/opened URL passes
  // classifyHost, unlike the raw wildcard (`0.0.0.0` → 403, `::`/'' → malformed).
  // A concrete host is advertised as-is, bracketing any IPv6 literal (`::1`,
  // `2001:db8::1`, …) so the URL parses.
  const wildcard = host === '0.0.0.0' || host === '::' || host === '';
  const displayHost = wildcard ? (host === '::' ? '[::1]' : '127.0.0.1') : bracketHost(host);
  const url = `http://${displayHost}:${boundPort}`;

  return {
    url,
    port: boundPort,
    async close() {
      clearInterval(ingestTimer);
      // Abort and drain in-flight downloads BEFORE anything else so a shutdown
      // (SIGINT/SIGTERM → `process.exit`) can't kill the process mid-write and
      // orphan a partial, potentially multi-GB `.staging` tree, nor let a
      // background job publish a model after the server is considered closed.
      await downloads.shutdown();
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
