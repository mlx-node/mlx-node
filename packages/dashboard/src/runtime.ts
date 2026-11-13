/**
 * The dashboard runtime: everything the API needs to answer a call, with no
 * transport attached.
 *
 * It spans TWO threads. This one owns the download manager, the rescan timer and
 * whatever transport is attached (`server.ts`'s HTTP adapter, or the Electron
 * MessagePort bridge). A `node:worker_threads` worker owns the SQLite index, the
 * serialized incremental ingest, and every route whose handler does synchronous
 * work. `node:sqlite` and drizzle are synchronous, so with everything on one
 * thread a single heavy query froze download progress for its whole duration
 * (measured: a trivial RPC issued during a 1.5 s synchronous call waited
 * 1449 ms), delayed a cancel click from even being RECEIVED, and false-positived
 * the supervisor's per-RPC timeouts.
 *
 * Which side answers a route is declared on the route itself (`routes.ts`), not
 * decided here.
 */

import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { probeAgentInstall } from '@mlx-node/agent/binary';
import { agentConfigDir } from '@mlx-node/agent/paths';
import { readDesktopEndpoint } from '@mlx-node/server/host/desktop-endpoint';

import type { SessionDetailResponse } from './agent-sessions/types.js';
import type { IngestSummary, MainApiContext } from './api/context.js';
import { dispatchMain, isApiPath, routeThreadFor, splitSegments } from './api/dispatch.js';
import { failure, toFailure, type ApiResponse } from './api/errors.js';
import { launchFromProbe, type HostTimings } from './chat/agent/agent-host.js';
import type { EndpointReader } from './chat/agent/endpoint.js';
import type { AllowedRoot } from './chat/agent/roots.js';
import { chatPaths, defaultChatRoot } from './chat/paths.js';
import { KeyringSecretStore, type SecretStore } from './chat/providers/secrets.js';
import { ProvidersService } from './chat/providers/service.js';
import type { ProviderId } from './chat/providers/types.js';
import { redactDeep, redactEntry } from './chat/redact.js';
import { ChatService, webServerScript, type AgentResolution } from './chat/service.js';
import type { ChatHostEvent, ConversationEvent } from './chat/types.js';
import { CodingAgentsService } from './coding-agents.js';
import type { CodingAgentsOptions } from './coding-agents.js';
import { DownloadManager, type DownloadEvent } from './download.js';
import { createLiveWatcher, type LiveEvent } from './live.js';
import { defaultModelsDir } from './models.js';
import { dashboardDbPath, desktopSupervisorStatePath, metricsRuntimeDir, metricsTraceDir } from './paths.js';
import type { ChatScope } from './rpc/protocol.js';
import { startDbWorker, type DbWorkerLifecycle } from './worker/client.js';

/** Incremental rescan cadence while the runtime is alive. */
const INGEST_INTERVAL_MS = 30_000;
/** Default budget for one shutdown step before the worker is terminated. */
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 5_000;

export type RuntimeLifecycleEvent = DbWorkerLifecycle;

/** Chat and Work. Tests and the dev harness set the agent command, the root and the endpoint reader. */
export interface ChatRuntimeOptions {
  /** App data folder. Default `~/.mlx-node/chat`. */
  root?: string;
  /** The inference endpoint. Default: `~/.mlx-node/desktop/inference.json`. */
  readEndpoint?: EndpointReader;
  /** The size on disk of a model path. Default: the files on disk. The dev harness passes fixture sizes. */
  modelSize?: (path: string) => Promise<number | null>;
  /** Replaces the agent binary and its version check (the scripted fake agent). */
  agentCommand?: { command: string; args: string[]; env?: Record<string, string> };
  /** What the agent runs the web tool server with. Default: this process's executable. */
  node?: { command: string };
  /** Where provider keys are kept. Default: the macOS Keychain. Tests and the dev harness pass a memory store. */
  secrets?: SecretStore;
  /** Provider origins for key checks. Default: the fixed provider hosts. The dev harness passes loopback fakes. */
  providerBaseUrls?: Partial<Record<ProviderId, string>>;
  /** Runs `mlx agent update` for the install action. */
  installAgent?: () => Promise<void>;
  /** Work: the process cwd (the user's home). */
  homeDir?: string;
  /** Work: the environment the agent inherits. Default: this process's. Tests and the dev harness pass a throwaway home. */
  baseEnv?: NodeJS.ProcessEnv;
  /** Work: the temp roots Auto covers. Default: `/tmp` when the resolver accepts it. Tests pass `[]` or a throwaway root. */
  tempRoots?: readonly AllowedRoot[];
  /** With no window attached for this long, open Work requests are denied. */
  pageGoneMs?: number;
  timings?: Partial<HostTimings>;
  createBudgetMs?: number;
  flushMs?: number;
  stateCoalesceMs?: number;
}

export interface DashboardRuntimeOptions {
  prepareDelegation?: CodingAgentsOptions['prepareCommand'];
  dbPath?: string;
  modelsDir?: string;
  tracesDir?: string;
  /** Inference runtime series and live heartbeats. Default: `runtime` beside `tracesDir`, else `metricsRuntimeDir()`. */
  runtimeDir?: string;
  /** Desktop supervisor state file. Default `desktopSupervisorStatePath()`. */
  supervisorPath?: string;
  cacheRoot?: string;
  /** grok-build (`mlx agent`) config home for the Sessions API. Default `agentConfigDir()`. */
  agentHome?: string;
  /**
   * Override the database worker's entry module. Packaging needs this: under
   * Electron the worker must load from an unpacked path outside the asar archive.
   */
  workerUrl?: URL | string;
  /**
   * How long each shutdown step waits for the worker before it is terminated
   * outright. A supervisor with its own quit deadline (Electron's `before-quit`)
   * should set this below that deadline.
   */
  shutdownTimeoutMs?: number;
  /** Periodic rescan cadence. Default 30 s. */
  ingestIntervalMs?: number;
  /** Worker lifecycle notifications, for a supervisor that must react to a dead worker. */
  onLifecycle?: (event: RuntimeLifecycleEvent) => void;
  /** Where coding-agent instruction files are looked up. Tests and the dev harness pass a sandbox. */
  codingAgents?: Pick<CodingAgentsOptions, 'home' | 'env'>;
  chat?: ChatRuntimeOptions;
}

/** One API call, addressed the way `fetch` would address it. */
export interface ApiCall {
  method: string;
  /** Path, optionally with a query string (`/api/models?limit=1`). */
  path: string;
  /** Already-parsed payload; omit when the call carries none. */
  body?: unknown;
  /** Set instead of {@link body} when the transport could not parse the payload. */
  bodyError?: string;
}

export interface DashboardRuntime {
  readonly modelsDir: string;
  readonly tracesDir: string;
  readonly runtimeDir: string;
  readonly cacheRoot: string | undefined;
  readonly agentHome: string;
  /**
   * What THIS thread owns, for a transport that dispatches its own main-thread
   * routes. The database is deliberately absent — it lives on the worker and is
   * reachable only through {@link call}.
   */
  readonly context: MainApiContext;
  /**
   * Answer one API call. Never rejects for an API failure — see `ApiResponse`.
   * An optional signal withdraws worker work that has not started yet; a handler
   * already running is allowed to finish rather than being interrupted halfway
   * through a mutation.
   */
  call(call: ApiCall, signal?: AbortSignal): Promise<ApiResponse>;
  /** Subscribe to a download job's progress events; returns the unsubscribe. */
  subscribe(jobId: string, listener: (event: DownloadEvent) => void): () => void;
  /**
   * Subscribe to live engine state and data-change events. The files are polled
   * only while at least one listener is attached. Returns the unsubscribe.
   */
  watchLive(listener: (event: LiveEvent) => void): () => void;
  /** Open a Chat stream: the host state, or one conversation. Returns the unsubscribe. */
  watchChat(scope: ChatScope, send: (event: ChatHostEvent | ConversationEvent) => void): () => void;
  /** Run (and await) an incremental rescan, serialized with every other ingest. */
  ingestNow(): Promise<IngestSummary>;
  /**
   * Stop the periodic rescan, drain in-flight downloads and await any in-flight
   * ingest — everything that must finish while the database is still open.
   * Idempotent; `close()` calls it for you. Rejects if the database worker cannot
   * confirm that its ingest chain reached a fixpoint within the shutdown budget.
   */
  drain(): Promise<void>;
  /**
   * {@link drain}, then close the SQLite handle and end the worker. Idempotent.
   * A drain failure is reported only after the bounded worker teardown still runs.
   */
  close(): Promise<void>;
}

/**
 * The shipped worker entry, or the source-run bootstrap when there is no build
 * beside us (vitest, `oxnode scripts/dev-dashboard.ts`). Checking for the built
 * file rather than sniffing the environment keeps a real `dist` authoritative —
 * a stale build can never be silently papered over by the `.ts`.
 */
function dashboardVersion(): string {
  try {
    const manifest = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as {
      version?: unknown;
    };
    return typeof manifest.version === 'string' ? manifest.version : '0.0.0';
  } catch {
    return '0.0.0';
  }
}

/**
 * What the agent runs the web tool server with: in the desktop app the app executable (as Node, through
 * `ELECTRON_RUN_AS_NODE` in the server env), named by the `MLX_DESKTOP_CLI` the app hands this process; else this
 * process's own executable.
 */
function webServerCommand(): string {
  const raw = process.env.MLX_DESKTOP_CLI;
  if (raw === undefined || raw === '') return process.execPath;
  try {
    const config = JSON.parse(raw) as { executable?: unknown };
    if (typeof config.executable === 'string' && isAbsolute(config.executable)) return config.executable;
  } catch {
    // Not the app's launcher config: fall back to this process.
  }
  return process.execPath;
}

function resolveAgent(command: ChatRuntimeOptions['agentCommand']): AgentResolution {
  if (command !== undefined) {
    return {
      launch: { ...command, source: 'test', tag: null, versionVerified: false },
      source: 'test',
      tag: null,
    };
  }
  try {
    const probe = probeAgentInstall();
    return { launch: launchFromProbe(probe), source: probe.source, tag: probe.effectiveTag ?? null };
  } catch {
    return { launch: { failure: 'agent-missing' }, source: 'none', tag: null };
  }
}

/**
 * The worker routes that send session text: the worker holds no needles, and a Work transcript may hold a provider
 * key or the inference token that a command printed. `detail` carries a transcript.
 */
function sessionTextRoute(pathname: string): 'detail' | 'other' | null {
  const s = splitSegments(pathname);
  if (s[0] !== 'api') return null;
  if (s[1] === 'chat') return s.length === 3 && s[2] === 'conversations' ? 'other' : null;
  if (s[1] !== 'sessions') return null;
  if (s.length === 2) return 'other';
  if (s.length === 3) return 'detail';
  return s.length === 4 && s[3] === 'timeline' ? 'other' : null;
}

function redactSessionText(body: unknown, detail: boolean, needles: readonly string[]): unknown {
  if (needles.length === 0) return body;
  if (!detail) return redactDeep(body, needles);
  const { transcript, ...rest } = body as SessionDetailResponse;
  return { ...redactDeep(rest, needles), transcript: transcript.map((entry) => redactEntry(entry, needles)) };
}

function defaultWorkerUrl(): URL {
  const built = new URL('./worker/db-worker.js', import.meta.url);
  return existsSync(fileURLToPath(built)) ? built : new URL('./worker/ts-bootstrap.ts', import.meta.url);
}

export function createDashboardRuntime(opts: DashboardRuntimeOptions = {}): DashboardRuntime {
  const dbPath = opts.dbPath ?? dashboardDbPath();
  const modelsDir = opts.modelsDir ?? defaultModelsDir();
  const tracesDir = opts.tracesDir ?? metricsTraceDir();
  const runtimeDir =
    opts.runtimeDir ?? (opts.tracesDir !== undefined ? join(dirname(tracesDir), 'runtime') : metricsRuntimeDir());
  const cacheRoot = opts.cacheRoot;
  // Absolute: the resume command `cd`s into the session cwd before setting
  // GROK_HOME, and grok uses it verbatim, so a relative home would point elsewhere.
  const agentHome = resolve(opts.agentHome ?? agentConfigDir());
  const chatOpts = opts.chat ?? {};
  const paths = chatPaths(resolve(chatOpts.root ?? defaultChatRoot()));

  // Kept on this thread: one `mkdir` at construction is not the blocking problem,
  // and an unwritable location stays a synchronous, actionable throw from the
  // factory instead of an asynchronous worker death.
  if (dbPath !== ':memory:') {
    mkdirSync(dirname(dbPath), { recursive: true });
  }

  // Spawning is asynchronous but `new Worker` is not, and messages queue on the
  // port until the thread is up — so the factory stays synchronous and no caller
  // has to await readiness.
  const worker = startDbWorker({
    workerUrl: opts.workerUrl ?? defaultWorkerUrl(),
    shutdownTimeoutMs: opts.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS,
    onLifecycle: opts.onLifecycle ?? ((): void => {}),
    dbPath,
    modelsDir,
    tracesDir,
    runtimeDir,
    cacheRoot,
    agentHome,
    chatAgentHome: paths.chatAgent,
    chatIndexPath: paths.conversations,
  });
  const downloads = new DownloadManager({ modelsDir });

  const codingAgents = new CodingAgentsService({
    ...opts.codingAgents,
    prepareCommand: opts.prepareDelegation,
    listModels: async () => {
      const response = await worker.call({ method: 'GET', path: '/api/coding-agents/models' });
      if (!response.ok) throw new Error(response.message);
      return (response.body as { models: { name: string }[] }).models.map((model) => model.name);
    },
  });
  const live = createLiveWatcher({
    runtimeDir,
    supervisorPath: opts.supervisorPath ?? desktopSupervisorStatePath(),
    ingest: () => worker.ingest(),
  });
  const providers = new ProvidersService({
    paths,
    secrets: chatOpts.secrets ?? new KeyringSecretStore(),
    baseUrls: chatOpts.providerBaseUrls,
  });
  const chat = new ChatService({
    paths,
    readEndpoint: chatOpts.readEndpoint ?? (() => readDesktopEndpoint()),
    modelSize: chatOpts.modelSize,
    resolveAgent: () => resolveAgent(chatOpts.agentCommand),
    readSession: async (id: string): Promise<SessionDetailResponse | null> => {
      const response = await worker.call({ method: 'GET', path: `/api/sessions/${encodeURIComponent(id)}` });
      if (response.ok) return response.body as SessionDetailResponse;
      if (response.code === 'E_NOT_FOUND' || response.code === 'E_BAD_REQUEST') return null;
      throw new Error('session read failed');
    },
    providers,
    live,
    clientVersion: dashboardVersion(),
    webServer: { command: chatOpts.node?.command ?? webServerCommand(), script: webServerScript() },
    installAgent: chatOpts.installAgent,
    extraCaCerts: process.env.NODE_EXTRA_CA_CERTS,
    agentHome: agentHome === resolve(agentConfigDir()) ? null : agentHome,
    workAgentHome: agentHome,
    baseEnv: chatOpts.baseEnv,
    tempRoots: chatOpts.tempRoots,
    pageGoneMs: chatOpts.pageGoneMs,
    homeDir: chatOpts.homeDir,
    timings: chatOpts.timings,
    createBudgetMs: chatOpts.createBudgetMs,
    flushMs: chatOpts.flushMs,
    stateCoalesceMs: chatOpts.stateCoalesceMs,
  });
  const context: MainApiContext = {
    modelsDir,
    tracesDir,
    runtimeDir,
    cacheRoot,
    downloads,
    codingAgents,
    live,
    chat,
    providers,
  };

  // The boot ingest runs inside the worker; only the periodic rescan is driven
  // from here. It is the backstop of the live watcher's ingest, and reports to it.
  const ingestTimer = setInterval(() => {
    void worker.ingest().then((summary) => live.notifyIngested(summary));
  }, opts.ingestIntervalMs ?? INGEST_INTERVAL_MS);
  ingestTimer.unref();

  let draining: Promise<void> | null = null;

  const drain = (): Promise<void> => {
    draining ??= (async () => {
      clearInterval(ingestTimer);
      live.close();
      // Abort and drain in-flight downloads BEFORE anything else so a shutdown
      // (SIGINT/SIGTERM → `process.exit`) can't kill the process mid-write and
      // orphan a partial, potentially multi-GB `.staging` tree, nor let a
      // background job publish a model after the runtime is considered closed.
      await Promise.all([downloads.shutdown(), codingAgents.close(), chat.close()]);
      // `clearInterval` only stops FUTURE ticks: a rescan already in flight kept
      // running while the database closed underneath it, which only ever looked
      // harmless because the ingest catch swallowed the resulting "database is
      // not open". The worker awaits its chain to a fixpoint for us.
      await worker.drain();
    })();
    return draining;
  };

  return {
    modelsDir,
    tracesDir,
    runtimeDir,
    cacheRoot,
    agentHome,
    context,
    async call(c: ApiCall, signal?: AbortSignal): Promise<ApiResponse> {
      if (signal?.aborted) {
        return failure('E_UNAVAILABLE', 'Dashboard request was cancelled before it started');
      }
      let url: URL;
      try {
        // A constant, trusted base: only the path + query of `c.path` are used.
        url = new URL(c.path, 'http://localhost');
      } catch {
        return failure('E_BAD_REQUEST', `Malformed request path ${c.path}`);
      }
      if (!isApiPath(url.pathname)) {
        return failure('E_NOT_FOUND', `No route matches ${c.method} ${url.pathname}`);
      }
      try {
        // Ownership is data on the route, never a path list here. An unmatched
        // request has no owner and is answered locally — its 405/404 needs no
        // context, and a garbage path never costs a round trip.
        //
        // INSIDE the try: deciding the owner runs the same route matcher, which
        // decodes each `:param`. A malformed escape threw from here, above the
        // catch, so `call` rejected instead of answering — breaking the contract
        // the catch below exists to keep. `worker.call` returns envelopes rather
        // than throwing, so nesting it changes nothing else.
        if (routeThreadFor(c.method, url.pathname) === 'worker') {
          const text = c.method === 'GET' ? sessionTextRoute(url.pathname) : null;
          const needles = text === null ? null : chat.redactionNeedles();
          const response = await worker.call(
            {
              method: c.method,
              path: c.path,
              body: c.body,
              ...(c.bodyError !== undefined ? { bodyError: c.bodyError } : {}),
            },
            signal,
          );
          if (needles === null || !response.ok) return response;
          return { ...response, body: redactSessionText(response.body, text === 'detail', await needles) };
        }
        if (signal?.aborted) {
          return failure('E_UNAVAILABLE', 'Dashboard request was cancelled before it started');
        }
        return await dispatchMain(context, {
          method: c.method,
          pathname: url.pathname,
          query: url.searchParams,
          body: c.body,
          ...(c.bodyError !== undefined ? { bodyError: c.bodyError } : {}),
        });
      } catch (err) {
        // `dispatchMain` converts handler throws itself; this catches a fault in
        // matching (e.g. `decodeURIComponent` on a malformed `%` escape) so the
        // documented "never rejects" contract holds for every caller.
        return toFailure(err);
      }
    },
    subscribe(jobId: string, listener: (event: DownloadEvent) => void): () => void {
      return downloads.subscribe(jobId, listener);
    },
    watchLive(listener: (event: LiveEvent) => void): () => void {
      return live.subscribe(listener);
    },
    watchChat(scope: ChatScope, send: (event: ChatHostEvent | ConversationEvent) => void): () => void {
      return chat.watch(scope, send);
    },
    ingestNow(): Promise<IngestSummary> {
      return worker.ingest();
    },
    drain,
    async close(): Promise<void> {
      // Even a failed drain must not leave the database worker alive. The drain
      // rejection remains the caller-visible proof that shutdown was not clean,
      // while close keeps its separate bounded teardown guarantee. Preserve both
      // errors if teardown itself also fails; otherwise a terminate failure would
      // hide the earlier unconfirmed ingest barrier.
      let drainFailed = false;
      let drainError: unknown;
      try {
        await drain();
      } catch (error) {
        drainFailed = true;
        drainError = error;
      }
      try {
        await worker.close();
      } catch (closeError) {
        if (drainFailed) {
          throw new AggregateError([drainError, closeError], 'Dashboard runtime failed to drain and close cleanly');
        }
        throw closeError;
      }
      if (drainFailed) throw drainError;
    },
  };
}
