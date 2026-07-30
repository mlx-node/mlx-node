/**
 * Everything the INFERENCE sidecar decides, with no `@mlx-node/lm` and no
 * Electron in it.
 *
 * `index.ts` beside this file is the entry the supervisor forks, and importing
 * it maps the 233 MB native addon into the importing process — which puts it out
 * of reach of every plain Node test, and out of reach of any test that is allowed
 * to run on a machine with no GPU. So the entry is reduced to "construct the real
 * host, hand it to `runSidecar`" and every rule that could be wrong lives here,
 * against an interface a stub can satisfy. Same split as `www.ts` vs
 * `protocol.ts` and `state.ts` vs `child-utility.ts`.
 *
 * The contract implemented here is the one documented on `ChildToSupervisor` in
 * `src/main/supervisor/types.ts` and demonstrated by
 * `__test__/fixtures/sidecar.mjs`.
 */

import { randomBytes } from 'node:crypto';

/**
 * Where the sidecar binds. See {@link sidecarHostOptions} for what the entry
 * actually passes.
 *
 * `port: 0` is the load-bearing part. `createInferenceHost` treats an omitted
 * port as "pick one for me" and calls `pickFreePort()`, which binds, reads the
 * port, closes, and hands it back for the real listen to use — racy by its own
 * docstring, and a lost race here is an EADDRINUSE at launch that no user can
 * act on. `0` binds an ephemeral port and the host reads the real one back off
 * the socket, so the URL it reports is the socket's own answer rather than a
 * prediction.
 *
 * `host` is pinned rather than left to the default for a different reason: the
 * supervisor refuses a handshake URL that is not loopback (`isLoopbackHttpUrl`),
 * because that is the cheapest way to make it poll something that answers `ok`
 * while the sidecar is dead. Binding loopback is what makes that check pass by
 * construction instead of by luck.
 */
export const SIDECAR_HOST_OPTIONS = Object.freeze({ port: 0, host: '127.0.0.1' });

/**
 * How long the inference host may drain HTTP requests before aborting them.
 *
 * Kept explicit at the process seam rather than relying on the server package's
 * default: MAIN's supervisor must allow this whole interval PLUS time for
 * `InferenceHost.close()` to flush its logger and remove the paged-config temp
 * root before escalating SIGTERM to SIGKILL.
 */
export const SIDECAR_HOST_CLOSE_TIMEOUT_MS = 5_000;

/**
 * A fresh secret for one sidecar process.
 *
 * 32 bytes from the CSPRNG. Not derived from the pid, the port or the clock:
 * all three are observable by any local process, and a token an attacker can
 * recompute is worse than no token because it also switches CORS off and makes
 * the surface look protected.
 */
export function newSidecarAuthToken(): string {
  return randomBytes(32).toString('base64url');
}

/**
 * What the entry passes to `createInferenceHost`.
 *
 * A loopback bind is NOT a closed door. Every local process on the machine can
 * reach the port, and so can JavaScript on any page the user's browser happens
 * to be showing — `fetch('http://127.0.0.1:<port>/v1/messages')` from a hostile
 * page is an ordinary cross-origin request, and the pre-auth default of
 * `Access-Control-Allow-Origin: *` made the reply readable to it. The token is
 * what makes that request a 401; the server's own CORS default then follows the
 * auth posture and stops sending the wildcard at all.
 *
 * A NEW token per call, per process. It is never persisted and never leaves
 * this process: the Control Panel window talks to CONTROL PANEL over a transferred
 * `MessagePort` and the supervisor polls `/health`, which is deliberately
 * carved out of the gate, so nothing outside needs it. Adding it to the `info`
 * reply would put a live secret into the supervisor's per-generation trace file
 * on disk — if a consumer ever genuinely needs it, that is the hazard to solve
 * first.
 */
export function sidecarHostOptions(): { port: number; host: string; authToken: string } {
  return { ...SIDECAR_HOST_OPTIONS, authToken: newSidecarAuthToken() };
}

/**
 * The slice of `InferenceHost` this module uses.
 *
 * Restated rather than imported: the type lives in `@mlx-node/server/host`,
 * whose module graph value-imports `@mlx-node/lm`. Importing it even for a type
 * would put an `import` of that specifier in this file, one character away from
 * becoming a value import — and `__test__/child-entries.test.ts` reads these
 * files and forbids exactly that.
 */
export interface SidecarHost {
  /** Connectable base URL, from the bound socket. */
  url: string;
  port: number;
  modelsDir: string;
  boundModel: string;
  models: readonly { name: string; path: string; modelType: string }[];
  logDir: string | null;
  health(): unknown;
  loadModel(name: string): Promise<void>;
  close(opts?: { timeoutMs?: number }): Promise<void>;
}

/**
 * The parent end of the control channel, normalised across the two transports
 * `ChildTransport` can be built on.
 */
export interface ParentChannel {
  post(message: unknown): void;
  onMessage(listener: (message: unknown) => void): void;
}

/** The `process` members {@link resolveParentChannel} reads. See there for why both exist. */
export interface ChannelProcess {
  /** Electron `utilityProcess` only. Its `message` listener receives `{ data }`. */
  parentPort?:
    | {
        postMessage(message: unknown): void;
        on(event: 'message', listener: (event: { data: unknown }) => void): void;
      }
    | undefined;
  /** `child_process.fork` only. Its `message` listener receives the raw value. */
  send?: ((message: unknown) => void) | undefined;
  on(event: 'message', listener: (message: unknown) => void): void;
}

/**
 * Pick the channel this process was actually started with.
 *
 * Both transports are real: `utilityProcess` is production and `child_process`
 * is the escape hatch if the GPU watchdog (ml-explore/mlx#3267) turns out to
 * fire inside a utilityProcess after all — Phase 0 did not reproduce it, which
 * is not the same as ruling it out. The sidecar has to run unchanged under
 * either, so the choice is made here rather than baked in.
 *
 * `parentPort` wins when both are present. It is the one that exists only in a
 * utilityProcess, so preferring it can never mis-detect; a preference the other
 * way would depend on `process.send` being absent under Electron, which is a
 * property of Electron's internals rather than of anything documented.
 */
export function resolveParentChannel(proc: ChannelProcess): ParentChannel {
  const parentPort = proc.parentPort;
  if (parentPort !== undefined) {
    return {
      post: (message) => {
        parentPort.postMessage(message);
      },
      onMessage: (listener) => {
        parentPort.on('message', (event) => {
          listener(event.data);
        });
      },
    };
  }
  if (typeof proc.send === 'function') {
    return {
      // Called as a method, never detached: `process.send` reads `this` for the
      // IPC channel, and a captured reference invoked bare throws.
      post: (message) => {
        proc.send?.(message);
      },
      onMessage: (listener) => {
        proc.on('message', listener);
      },
    };
  }
  // Running the sidecar by hand (`node dist/inference/index.js`) is a legitimate
  // way to reproduce a wedge in a terminal, and it must not crash on the first
  // `post`. It simply has nobody to talk to.
  throw new NoParentChannelError();
}

/** No IPC channel: this process was not forked by anything. */
export class NoParentChannelError extends Error {
  constructor() {
    super('no parent IPC channel (neither process.parentPort nor process.send)');
    this.name = 'NoParentChannelError';
  }
}

/*
 * ---------------------------------------------------------------------------
 * Request handling
 * ---------------------------------------------------------------------------
 */

/**
 * What the supervisor may ask the sidecar over the control channel.
 *
 * Deliberately not "anything the HTTP API can do". The sidecar's HTTP port is
 * the inference surface and the Control Panel window reaches it directly; this channel
 * exists for the things a *supervisor* needs and an HTTP client cannot get —
 * what was discovered on disk, and an out-of-band model swap that runs under the
 * same drain/writer brackets the HTTP path uses.
 */
export type SidecarRequest =
  /** Static facts settled at startup: url, port, models dir, bound model, log dir. */
  | { op: 'info' }
  /** The same body `GET /health` returns, without a round trip through the socket. */
  | { op: 'health' }
  /** Everything discovered under the models dir, alphabetical. */
  | { op: 'models' }
  /** Make `name` the resident model. Rejects for a name that was not discovered. */
  | { op: 'load'; name: string };

/** Answer one request. Rejects for a malformed or unknown payload — never guesses. */
export async function handleSidecarRequest(host: SidecarHost, payload: unknown): Promise<unknown> {
  if (typeof payload !== 'object' || payload === null) {
    throw new Error(`malformed request payload (${payload === null ? 'null' : typeof payload})`);
  }
  const request = payload as { op?: unknown; name?: unknown };
  switch (request.op) {
    case 'info':
      return {
        url: host.url,
        port: host.port,
        modelsDir: host.modelsDir,
        boundModel: host.boundModel,
        logDir: host.logDir,
      };
    case 'health':
      return host.health();
    case 'models':
      // Projected field by field rather than passed through. Every reply crosses
      // a structured clone, and `DiscoveredModel` carries a `preset` whose shape
      // is the server package's business — a non-clonable member added there
      // would break this channel with an error naming neither side.
      return host.models.map((model) => ({ name: model.name, path: model.path, modelType: model.modelType }));
    case 'load': {
      if (typeof request.name !== 'string') throw new Error('load requires a model name');
      await host.loadModel(request.name);
      return { boundModel: request.name };
    }
    default:
      throw new Error(`unknown op ${JSON.stringify(request.op)}`);
  }
}

/*
 * ---------------------------------------------------------------------------
 * The run loop
 * ---------------------------------------------------------------------------
 */

export interface SidecarDeps {
  channel: ParentChannel;
  /** Build the real host. Rejecting here is fatal — see {@link runSidecar}. */
  createHost(): Promise<SidecarHost>;
  /**
   * The bearer token `createHost` was configured with.
   *
   * It travels with the ready handshake because the URL alone is not a
   * capability any more: every inference route is gated, so a client handed
   * only the URL gets 401 for everything. MAIN keeps it in memory to hand to
   * whoever the user points at this server — it is never written to disk, never
   * logged, and deliberately still absent from the `info` reply, which is the
   * one sidecar response the supervisor stores per generation.
   */
  authToken: string;
  /**
   * Register the graceful-stop handler. `utilityProcess.kill()` is SIGTERM on
   * POSIX and takes no argument, so SIGTERM is the only stop signal that reaches
   * a sidecar in production; the entry also wires SIGINT so a hand-run
   * reproduction answers Ctrl-C the same way.
   */
  onStopSignal(handler: () => void): void;
  /**
   * One line to stderr, written SYNCHRONOUSLY.
   *
   * stdout/stderr are async pipes under both transports and an exit right after a
   * write truncates it. For the failures below the line is the only diagnostic
   * that reaches the supervisor's crash report at all.
   */
  logError(line: string): void;
  exit(code: number): void;
}

/** Startup failed. Distinct from 1 so a crash report can tell it from a throw. */
export const EXIT_STARTUP_FAILED = 78;

/**
 * Run the sidecar until it is told to stop.
 *
 * Resolves once the host is up and the handshake has been posted; the process
 * stays alive on the host's own handles after that. It does NOT resolve on
 * shutdown — `deps.exit` ends the process.
 *
 * A `createHost` rejection exits immediately instead of waiting out the
 * supervisor's 60 s readiness budget. The common cause is
 * `NoModelsDiscoveredError`, which no amount of waiting fixes, and a fast exit
 * turns it into a visible `failed` state with the reason in the crash report
 * rather than a minute of "starting" per restart.
 */
export async function runSidecar(deps: SidecarDeps): Promise<void> {
  const { channel } = deps;
  let host: SidecarHost | null = null;
  let stopping = false;

  const stop = (): void => {
    if (stopping) return;
    stopping = true;
    // `close()` disposes the verbose logger, the HTTP server and the paged-config
    // temp root. The temp root is the only one of the three that outlives this
    // process, so skipping it leaves work for the next host's startup sweep.
    const closing =
      host === null ? Promise.resolve() : host.close({ timeoutMs: SIDECAR_HOST_CLOSE_TIMEOUT_MS });
    void closing
      .catch((error: unknown) => {
        deps.logError(`[mlx] inference host did not close cleanly: ${describe(error)}`);
      })
      .then(() => {
        // 0 on purpose, and it is NOT read as success: the supervisor decides
        // clean-vs-crash from the intent it recorded before the kill, precisely
        // because `utilityProcess.kill()` reports 0 either way.
        deps.exit(0);
      });
  };
  deps.onStopSignal(stop);

  // Installed BEFORE the host exists. The supervisor does not send a request
  // until it has seen a serving `/health`, so this cannot normally fire early —
  // but a request that arrives anyway gets an honest failure instead of being
  // dropped into a promise the caller waits out.
  channel.onMessage((raw: unknown) => {
    const request = parseSupervisorMessage(raw);
    if (request === null) return;
    const { id, payload } = request;
    if (host === null) {
      channel.post({ kind: 'mlx:response', id, ok: false, error: 'inference host is not up yet' });
      return;
    }
    void handleSidecarRequest(host, payload).then(
      (value) => {
        channel.post({ kind: 'mlx:response', id, ok: true, value });
      },
      (error: unknown) => {
        channel.post({ kind: 'mlx:response', id, ok: false, error: describe(error) });
      },
    );
  });

  let started: SidecarHost;
  try {
    started = await deps.createHost();
  } catch (error) {
    deps.logError(`[mlx] inference host failed to start: ${describe(error)}`);
    deps.exit(EXIT_STARTUP_FAILED);
    return;
  }
  if (stopping) {
    // The stop arrived while the host was still coming up, so `stop()` found
    // nothing to close and has already exited. Close what appeared anyway: a
    // `deps.exit` that does not end the process (a test, or a future caller)
    // would otherwise be left holding a listening socket it can no longer reach,
    // and announcing readiness for a process on its way out is worse.
    await started.close({ timeoutMs: SIDECAR_HOST_CLOSE_TIMEOUT_MS }).catch((error: unknown) => {
      deps.logError(`[mlx] inference host did not close cleanly: ${describe(error)}`);
    });
    return;
  }
  host = started;

  // `started.url` and nothing derived from it. The host built this string from
  // `server.address().port` — the socket's own answer — and the supervisor polls
  // `/health` on exactly what arrives here, so any adjustment made in passing
  // would point the readiness probe at a port nothing is listening on.
  channel.post({ kind: 'mlx:ready', url: started.url, authToken: deps.authToken });
}

/** `SupervisorToChild`, parsed rather than cast. Anything else is ignored. */
function parseSupervisorMessage(raw: unknown): { id: number; payload: unknown } | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const message = raw as { kind?: unknown; id?: unknown; payload?: unknown };
  if (message.kind !== 'mlx:request' || typeof message.id !== 'number') return null;
  return { id: message.id, payload: message.payload };
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
