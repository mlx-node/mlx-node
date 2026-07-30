/**
 * `@mlx-node/server/host` — the reusable inference-host bootstrap.
 *
 * Everything a process needs to go from "a directory of downloaded models" to
 * "a listening Anthropic/OpenAI-compatible endpoint with one resident model
 * and a working `/model` swap", with none of the opinions about WHO supervises
 * the process.
 *
 * That last part is why this is a module rather than a CLI flag. The two known
 * front-ends invert the supervision relationship:
 *
 *   - `mlx launch claude` starts the host, then spawns and supervises a child
 *     (`claude`), and exits when the child does.
 *   - The desktop app's Electron `utilityProcess` IS the child; the host is
 *     supervised, gets told when to shut down, and may be SIGKILLed.
 *
 * `mlx serve` is the third, degenerate case — nothing above, nothing below —
 * and doubles as the terminal-visible reproduction of the sidecar when the
 * utilityProcess wedges.
 *
 * A subpath export rather than part of `@mlx-node/server`'s main entry, so
 * importing the plain HTTP handler does not drag in `@mlx-node/lm` and model
 * loading.
 *
 * ## Ownership
 *
 * The returned host owns, and disposes on {@link InferenceHost.close}:
 *   1. the HTTP server,
 *   2. the verbose request logger (when `logDir` is set),
 *   3. the `PagedConfigOverrideManager`'s temp root.
 *
 * The server goes first because the logger records a request from that
 * request's own `finish`/`close` handler. Ending the log streams while a
 * request is still draining loses exactly the completion lines a verbose
 * shutdown exists to capture. The temp root goes last because a
 * still-draining request may still be reading a cloned config. Each step is
 * independently guarded — one failing disposer must not strand the ones
 * after it.
 */

import type { Server } from 'node:http';

import { loadModel as loadModelNative, PagedConfigOverrideManager, type LoadableModel } from '@mlx-node/lm';

import type { ServerHealth } from '../health.js';
import { createServer, resolveAuthToken, type CloseOptions, type ServerInstance } from '../server.js';
import { discoverModels, type DiscoveredModel } from './discover.js';
import { applyEnginePolicy, type EnginePolicy } from './env-policy.js';
import { attachLogger as defaultAttachLogger, type Logger } from './logger.js';
import { hostUrl, isLoopbackBindHost, normalizeLoopbackBindHost, pickFreePort } from './net.js';
import { resolveModelsDir } from './paths.js';
import { makeSwapController } from './swap.js';
import { hostTempDirPrefix, sweepOrphanHostTempRoots } from './temp-root.js';

/** Families `mlx launch claude` has historically forced onto the paged path. */
export const DEFAULT_PAGED_MODEL_TYPES = ['qwen3_5', 'qwen3_5_moe'] as const;

/** Thrown when `modelsDir` holds nothing servable. Carries the dir for the caller's message. */
export class NoModelsDiscoveredError extends Error {
  constructor(readonly modelsDir: string) {
    super(`No models discovered under ${modelsDir}.`);
    this.name = 'NoModelsDiscoveredError';
  }
}

/**
 * Thrown when a bind reachable from the network was asked for with no shared
 * secret to gate it.
 *
 * There is no safe way to serve that: every route but the `/health` liveness
 * carve-out runs inference, so an unauthenticated LAN-reachable bind hands
 * anyone who can route to this machine the GPU, the RAM, and the list of
 * models on disk. Failing at startup is the only outcome the operator can act
 * on — serving-with-a-warning is a warning nobody reads scrolling past a
 * model load.
 */
export class InsecureBindError extends Error {
  constructor(readonly host: string) {
    super(
      `Refusing to bind ${host} without an auth token: every route except /health runs inference. ` +
        `Pass an auth token (mlx serve --auth-token, or MLX_SERVER_AUTH_TOKEN), or bind 127.0.0.1.`,
    );
    this.name = 'InsecureBindError';
  }
}

/** Thrown when a requested model name is not among the discovered ones. */
export class ModelNotFoundError extends Error {
  constructor(
    readonly requested: string,
    readonly modelsDir: string,
    readonly available: string[],
  ) {
    super(`Model "${requested}" not found under ${modelsDir}.`);
    this.name = 'ModelNotFoundError';
  }
}

/** Thrown when a model load is requested after host shutdown reaches its admission boundary. */
export class InferenceHostClosedError extends Error {
  constructor() {
    super('Inference host is closing or closed; model loads are no longer accepted.');
    this.name = 'InferenceHostClosedError';
  }
}

export interface InferenceHostOptions {
  /**
   * Port to bind. Omitted ⇒ a free port is picked up-front (so the URL is
   * known before the server starts, which `mlx launch claude` needs in order
   * to bake `ANTHROPIC_BASE_URL` into the child's env). `0` binds an
   * ephemeral port and the real one is read back off the socket.
   */
  port?: number;
  /**
   * Host to bind. Default `127.0.0.1`.
   *
   * A non-loopback value (including the wildcards `0.0.0.0` / `::`) requires
   * an auth token — from {@link InferenceHostOptions.authToken} or
   * `MLX_SERVER_AUTH_TOKEN` — or the call throws {@link InsecureBindError}
   * instead of listening.
   */
  host?: string;
  /** Model discovery root. Default: {@link resolveModelsDir}'s resolution order. */
  modelsDir?: string;
  /**
   * Which discovered model is the bound/default one. Precedence, highest
   * first: this option, `ANTHROPIC_MODEL`, `discovered[0]` (alphabetical).
   * A name that matches nothing discovered throws {@link ModelNotFoundError}
   * rather than silently falling back.
   */
  model?: string;
  /** Shared secret for every route except `/health`. See `ServerConfig.authToken`. */
  authToken?: string;
  /** When set, every HTTP turn is captured under this directory. */
  logDir?: string;
  /**
   * Engine env policy to apply to `process.env` BEFORE anything can load a
   * model. Omitted ⇒ nothing is written and the engine's own defaults stand.
   * Out-of-process hosts must pass `engineEnvFor(policy)` to `fork({ env })`
   * instead and leave this unset — the vars latch via `OnceLock` on first
   * read, so mutating them in the child is too late.
   */
  enginePolicy?: EnginePolicy;
  /** Model types forced onto the block-paged KV cache. Default {@link DEFAULT_PAGED_MODEL_TYPES}. */
  pagedModelTypes?: readonly string[];
  /** Path to the SQLite response store. See `ServerConfig.storePath`. */
  storePath?: string;
  /** Disable response persistence entirely. See `ServerConfig.disableStore`. */
  disableStore?: boolean;
  /**
   * Remove temp roots left behind by hosts that were killed without running
   * `close()`. Default `true`; a supervisor that runs several hosts under one
   * pid namespace and wants to control the timing can turn it off and call
   * {@link sweepOrphanHostTempRoots} itself.
   */
  sweepOrphanTempRoots?: boolean;

  /** Test seam: replaces the native model loader. */
  loadModel?: (path: string) => Promise<LoadableModel>;
  /** Test seam: replaces the verbose request logger. */
  attachLogger?: (server: Server, logDir: string) => Logger;
}

export interface InferenceHost {
  /** Connectable base URL, e.g. `http://127.0.0.1:51234`. */
  url: string;
  /** The port actually bound (resolved, never `0`). */
  port: number;
  /** The host actually bound — the requested value, not the advertised one. */
  host: string;
  /** The resolved model discovery root. */
  modelsDir: string;
  /** Everything discovered under `modelsDir`, alphabetical. */
  models: DiscoveredModel[];
  /** Name of the default/bound model. Always a member of {@link models}. */
  boundModel: string;
  /** Verbose log directory, or `null` when logging is off. */
  logDir: string | null;
  /** Escape hatch for callers that need the registry, store, or raw `http.Server`. */
  server: ServerInstance;
  /** Current readiness snapshot — the same body `GET /health` returns. */
  health(): ServerHealth;
  /**
   * Make `name` the resident model, out of band (no HTTP request needed).
   *
   * Runs under the SAME brackets `ServerInstance.loadModel` uses — drains
   * suspended OUTSIDE the coordinator's exclusive writer slot — so it cannot
   * race the process-wide Metal allocator against in-flight inference, and
   * `/health` reports `loading` plus a `lastLoad` record labelled with `name`.
   *
   * It deliberately does NOT call `ServerInstance.loadModel` itself. That
   * helper only ever registers, never unregisters, so calling it per swap
   * would accumulate every model the user has ever picked in memory. The
   * single-resident swap controller is the thing that makes a swap a swap;
   * this method just wraps it in the right brackets.
   *
   * Rejects with {@link ModelNotFoundError} for an unknown name — the swap
   * controller would otherwise treat it as an alias for the current resident,
   * which is right for Claude Code's hardcoded `claude-haiku-*` but wrong for
   * a supervisor that asked for a specific model.
   *
   * Calls admitted before {@link close} begins are allowed to finish. Calls
   * made after shutdown begins reject with {@link InferenceHostClosedError}.
   */
  loadModel(name: string): Promise<void>;
  /**
   * Stop accepting new HTTP and out-of-band load work, wait for every
   * out-of-band load already admitted, then dispose the logger and temp root.
   * Idempotent and memoized; disposal steps are individually guarded so one
   * failure cannot strand the remaining steps.
   */
  close(opts?: CloseOptions): Promise<void>;
}

/**
 * Start a local inference host.
 *
 * Throws {@link NoModelsDiscoveredError} / {@link ModelNotFoundError} /
 * {@link InsecureBindError} rather than exiting — a library cannot know
 * whether its caller is a CLI, a test, or an Electron main process. Front-ends
 * render those into their own messages.
 */
export async function createInferenceHost(opts: InferenceHostOptions = {}): Promise<InferenceHost> {
  const host = opts.host ?? '127.0.0.1';
  // BEFORE the engine policy, the temp sweep and discovery — all of which
  // mutate state outside this function — so a refused start leaves nothing
  // behind. Resolved through `resolveAuthToken` rather than reading
  // `opts.authToken`, or `MLX_SERVER_AUTH_TOKEN=… mlx serve --host 0.0.0.0`
  // would be refused despite being fully protected.
  if (!isLoopbackBindHost(host) && resolveAuthToken(opts.authToken) === undefined) {
    throw new InsecureBindError(host);
  }
  // `[::1]` is valid URL-authority spelling and the security predicate accepts
  // it as loopback, but Node's listen host must be the bare literal. Normalize
  // only after the reachability gate, and only the exact safe form.
  const bindHost = normalizeLoopbackBindHost(host);

  // FIRST, before anything can touch the engine: the native side latches
  // these via `OnceLock` on first read, so a policy applied after a load has
  // silently done nothing.
  if (opts.enginePolicy !== undefined) applyEnginePolicy(opts.enginePolicy);

  // Reclaim roots left by hosts that never got to run `close()`. Best effort
  // and awaited only so a test can observe it; a failure here is not a reason
  // to refuse to start.
  if (opts.sweepOrphanTempRoots !== false) {
    await sweepOrphanHostTempRoots().catch(() => []);
  }

  const modelsDir = resolveModelsDir(opts.modelsDir);
  const models = await discoverModels(modelsDir);
  if (models.length === 0) throw new NoModelsDiscoveredError(modelsDir);

  // Precedence: explicit option > ANTHROPIC_MODEL > discovered[0].
  const requestedModel = opts.model ?? process.env.ANTHROPIC_MODEL;
  const requestedEntry = requestedModel != null ? models.find((m) => m.name === requestedModel) : undefined;
  if (requestedModel != null && requestedEntry === undefined) {
    throw new ModelNotFoundError(
      requestedModel,
      modelsDir,
      models.map((m) => m.name),
    );
  }
  const boundEntry = requestedEntry ?? models[0];

  // `undefined` means "you pick"; `0` means "the kernel picks and I will read
  // it back". Only the former needs the up-front probe.
  const requestedPort = opts.port ?? (await pickFreePort());

  // The swap controller needs the registry from the server instance, but the
  // server needs the controller's callbacks at construction. Bridge via a
  // late-bound holder: the callbacks capture `ctrlRef.current` by closure.
  const ctrlRef: { current: ReturnType<typeof makeSwapController> | null } = { current: null };
  let acceptingHttpModelLoads = true;
  const activeHttpModelLoads = new Set<Promise<void>>();

  const resolveHttpModel = (name: string): Promise<void> => {
    if (!acceptingHttpModelLoads) return Promise.reject(new InferenceHostClosedError());
    const operation = ctrlRef.current!.resolveModel(name);
    activeHttpModelLoads.add(operation);
    void operation.then(
      () => activeHttpModelLoads.delete(operation),
      () => activeHttpModelLoads.delete(operation),
    );
    return operation;
  };

  const serverConfig = {
    port: requestedPort,
    host: bindHost,
    resolveModel: resolveHttpModel,
    listModels: () => ctrlRef.current!.listModels(),
    ...(opts.authToken !== undefined ? { authToken: opts.authToken } : {}),
    ...(opts.storePath !== undefined ? { storePath: opts.storePath } : {}),
    ...(opts.disableStore !== undefined ? { disableStore: opts.disableStore } : {}),
  };
  const server = await createServer(serverConfig);

  // Wrap the loader so managed families get `use_block_paged_cache: true`
  // injected via a temp-dir clone with a patched config.json. The temp root is
  // named after our pid so a SIGKILLed host's root can be reclaimed by the
  // next host's startup sweep — see `temp-root.ts`.
  const pagedConfigOverrides = new PagedConfigOverrideManager({
    modelTypes: opts.pagedModelTypes ?? DEFAULT_PAGED_MODEL_TYPES,
    tempDirPrefix: hostTempDirPrefix(),
  });
  const loadModelFn = opts.loadModel ?? loadModelNative;
  const loadModelPagedAware = async (path: string): Promise<LoadableModel> =>
    loadModelFn(await pagedConfigOverrides.resolve(path));
  const controller = makeSwapController(models, server.registry, loadModelPagedAware, boundEntry.name);
  ctrlRef.current = controller;

  // Attach AFTER `createServer` so the wrapper sees every incoming request,
  // including the `GET /v1/models` a client fires on startup.
  //
  // By this point the socket is bound and `ctrlRef.current` is wired, so the
  // endpoint already answers real requests. `attachLogger` opens with a
  // synchronous `mkdirSync`, which throws on an unwritable `--log-dir`; without
  // this rollback the rejection would strand a fully working inference server
  // with no handle left to close it. Every caller in this repo exits the
  // process on failure, so the leak is only reachable by an in-process
  // embedder — which is exactly who this module is for.
  let logger: Logger | null = null;
  if (opts.logDir !== undefined) {
    try {
      logger = (opts.attachLogger ?? defaultAttachLogger)(server.server, opts.logDir);
    } catch (err) {
      // Mirror `close()`'s order and its independent guards, and rethrow the
      // ORIGINAL failure — a secondary close error must not mask the EACCES
      // that actually explains what went wrong.
      try {
        await server.close();
      } catch {
        /* already down, or a socket refused to die; fall through */
      }
      // A forced close destroys sockets, not the async handler promises
      // `http.createServer` discarded. Close resolver admission only after
      // the server has finished its graceful window, then drain any lazy load
      // that was already inside the controller before removing its temp files.
      acceptingHttpModelLoads = false;
      await Promise.allSettled(activeHttpModelLoads);
      try {
        await pagedConfigOverrides.cleanup();
      } catch {
        /* the startup sweep of the next host will reclaim it */
      }
      throw err;
    }
  }

  const address = server.server.address();
  const boundPort = address !== null && typeof address === 'object' ? address.port : requestedPort;

  const byName = new Map(models.map((m) => [m.name, m]));

  // `closeStarted` is an admission latch, checked and flipped synchronously:
  // once close() returns its promise no later out-of-band load can join this
  // set. The operations themselves retain the normal coordinator/controller
  // ordering; shutdown only observes their promises and never takes a lock
  // they need, so queued swaps can drain without deadlocking against close.
  let closeStarted = false;
  const activeHostLoads = new Set<Promise<void>>();
  let closePromise: Promise<void> | null = null;

  return {
    url: hostUrl(host, boundPort),
    port: boundPort,
    host,
    modelsDir,
    models,
    boundModel: boundEntry.name,
    logDir: logger?.logDir ?? null,
    server,
    health: () => server.health(),

    loadModel(name: string): Promise<void> {
      if (closeStarted) return Promise.reject(new InferenceHostClosedError());
      if (!byName.has(name)) {
        return Promise.reject(
          new ModelNotFoundError(
            name,
            modelsDir,
            models.map((m) => m.name),
          ),
        );
      }
      // Same nesting as `runGuardedModelLoad`: the drain suspension must be
      // OUTSIDE the writer lock so the armed `clearCache()` timer cannot fire
      // while we are parked waiting for the lock. The HTTP path takes the
      // locks in this same order, so the two can never deadlock against each
      // other on the controller's internal serialization.
      const operation = server.withSuspendedDrains(async () => {
        await server.modelWork.withModelLoad(() => controller.resolveModel(name), name);
      });
      activeHostLoads.add(operation);
      // Use a two-arm `then`, rather than an ignored `finally()` promise:
      // cleanup must not manufacture an unhandled rejection when the caller
      // legitimately observes a failed load through `operation`.
      void operation.then(
        () => activeHostLoads.delete(operation),
        () => activeHostLoads.delete(operation),
      );
      return operation;
    },

    close(closeOpts?: CloseOptions): Promise<void> {
      if (closePromise !== null) return closePromise;

      // Flip the latch and snapshot in the same synchronous turn. Every call
      // in the snapshot was fully admitted before shutdown; no later call can
      // race into the set after this point.
      closeStarted = true;
      const admittedHostLoads = [...activeHostLoads];

      closePromise = (async (): Promise<void> => {
        // Every step is independently guarded: a server that fails to close
        // must still leave the log streams ended and the temp root
        // reclaimed. The temp root is the only one of the three that leaks
        // OUTSIDE this process.
        try {
          await server.close(closeOpts);
        } catch {
          /* already down, or a socket refused to die; fall through */
        }
        // A forced server close destroys request sockets but does not await
        // the async handler promises. Stop any handler that has not reached
        // the resolver from starting a late load, then snapshot resolver calls
        // already inside the controller. Direct calls were closed and
        // snapshotted synchronously above; both groups can now drain without
        // shutdown holding a coordinator/controller lock they need.
        acceptingHttpModelLoads = false;
        const admittedHttpModelLoads = [...activeHttpModelLoads];
        // `allSettled` preserves cleanup when either kind of loader rejects;
        // the original operation still carries that error to its caller.
        await Promise.allSettled([...admittedHostLoads, ...admittedHttpModelLoads]);
        // Only once nothing is still serving. The logger writes a request's
        // record from that request's `finish` handler, so ending the streams
        // first both drops the record and — with no `error` listener — makes
        // the write fatal. `attachLogger` guards the second half; this
        // ordering is what preserves the first.
        if (logger !== null) {
          try {
            await logger.close();
          } catch {
            /* logging is best effort; never block shutdown */
          }
        }
        try {
          await pagedConfigOverrides.cleanup();
        } catch {
          /* the startup sweep of the next host will reclaim it */
        }
      })();
      return closePromise;
    },
  };
}

export { discoverModels, type DiscoveredModel } from './discover.js';
export {
  applyEnginePolicy,
  engineEnvFor,
  ENGINE_POLICY_ENV_VARS,
  LAUNCHER_ENGINE_POLICY,
  type EnginePolicy,
} from './env-policy.js';
export { attachLogger, resolveLogDir, type Logger } from './logger.js';
export { bracketHost, hostUrl, isLoopbackBindHost, pickFreePort } from './net.js';
export { resolveMlxNodeHome, resolveModelsDir } from './paths.js';
export { makeSwapController, type SwapController } from './swap.js';
export {
  hostTempDirPrefix,
  isProcessAlive,
  sweepOrphanHostTempRoots,
  HOST_TEMP_DIR_STEM,
  type SweepOrphanTempRootsOptions,
} from './temp-root.js';
