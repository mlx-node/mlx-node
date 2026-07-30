/**
 * The seam between the supervisor and whatever actually holds the OS process.
 *
 * Phase 0's GPU-watchdog A/B did not reproduce ml-explore/mlx#3267 inside an
 * Electron `utilityProcess`, but a did-not-reproduce is not a clean bill of
 * health — the aggravators we could not test (dragging/resizing the window
 * mid-decode, an external display) are exactly the ones that make the watchdog
 * fire. So moving INFERENCE out to a fully separate OS process has to stay a
 * configuration change, not a rewrite. That is what {@link ChildTransport} buys.
 *
 * It is also what makes the supervisor testable at all: `import { utilityProcess }
 * from 'electron'` cannot resolve outside an Electron process, so anything that
 * touches it is unreachable from a plain Node test. Same discipline as
 * `www.ts` vs `protocol.ts` — the decisions live where they can be proven, and
 * `child-utility.ts` is reduced to a fork call.
 */

/** Everything a transport needs to start the sidecar. */
export interface ChildSpec {
  /**
   * Absolute path to a **JavaScript** module. Both `utilityProcess.fork` and
   * `child_process.fork` hand the path to a Node runtime with no TS loader, so
   * this is always build output, never a `.ts` source.
   */
  modulePath: string;
  args: readonly string[];
  /**
   * The child's complete environment. Complete, not a patch: every engine knob
   * the sidecar reads latches through a Rust `OnceLock` on first read, so a
   * value that is not present at fork time can never be supplied later. See
   * `env.ts`.
   */
  env: Readonly<Record<string, string>>;
  cwd?: string;
}

/** How a child process ended, normalised across the two transports. */
export interface ChildExit {
  /** Exit status. `null` when the OS reported a signal instead. */
  code: number | null;
  /**
   * POSIX signal name, or `null`.
   *
   * **Always `null` under Electron's `utilityProcess`** — its `exit` event
   * carries a bare `code` and nothing else. The field exists so the
   * `child_process` transport can put its extra information somewhere useful,
   * and it is DIAGNOSTIC ONLY: no classification may read it, or the
   * production path would decide differently from the path the tests drive.
   * `assessExit` in `state.ts` uses it for the human-facing reason string and
   * never for the verdict.
   */
  signal: string | null;
}

/**
 * Callbacks handed to the transport at spawn time rather than attached
 * afterwards. An EventEmitter shape would leave a window between `fork()` and
 * `.on('exit')` in which a child that died instantly is missed — and a sidecar
 * that dies instantly is the exact case the restart budget exists for.
 */
export interface ChildEvents {
  onMessage(message: unknown): void;
  onExit(exit: ChildExit): void;
  /**
   * Spawn failed, or the runtime hit a non-continuable fault.
   *
   * The two transports differ and the supervisor must tolerate both:
   * `child_process` may emit this with NO following `exit` (the process never
   * existed), while Electron documents that `exit` always follows. `index.ts`
   * therefore treats this as advisory and arms a short deadline for the `exit`
   * that may never come.
   */
  onError(error: Error): void;
}

/** The live child. Deliberately not an EventEmitter — see {@link ChildEvents}. */
export interface ChildHandle {
  /** `undefined` before spawn completes and after exit, in both transports. */
  readonly pid: number | undefined;
  /**
   * Present when the transport was asked for piped stdio. The supervisor keeps
   * a small tail of both: for a class-(c) failure — a C++ throw unwinding
   * across an `extern "C"` shim, which makes Rust `abort()` — the stderr line
   * is the *only* diagnostic that exists. The exit code alone says nothing.
   */
  readonly stdout: NodeJS.ReadableStream | null;
  readonly stderr: NodeJS.ReadableStream | null;
  /** Never throws; a send on a dead channel is dropped. */
  postMessage(message: unknown): void;
  /**
   * Graceful stop. Takes no signal argument because `utilityProcess.kill()`
   * takes none — it is SIGTERM on POSIX, always. Widening this to accept a
   * signal would be a promise only one of the two transports could keep.
   */
  kill(): void;
  /**
   * Last resort, after the grace period. Separate from {@link kill} for the
   * same reason: `utilityProcess` cannot escalate, so its implementation has to
   * go around the handle via `process.kill(pid, 'SIGKILL')`.
   */
  forceKill(): void;
}

export type ChildTransport = (spec: ChildSpec, events: ChildEvents) => ChildHandle;

/*
 * ---------------------------------------------------------------------------
 * Wire protocol
 * ---------------------------------------------------------------------------
 */

/** Supervisor → sidecar. */
export type SupervisorToChild = { kind: 'mlx:request'; id: number; payload: unknown };

/** Sidecar → supervisor. */
export type ChildToSupervisor =
  | { kind: 'mlx:ready'; url: string; authToken: string }
  | { kind: 'mlx:response'; id: number; ok: true; value: unknown }
  | { kind: 'mlx:response'; id: number; ok: false; error: string };

/**
 * The contract the sidecar entry module must satisfy:
 *
 *  1. post `{ kind: 'mlx:ready', url, authToken }` once its HTTP server is
 *     listening, with
 *     the port it ACTUALLY bound — the supervisor does not pre-assign one,
 *     because `pickFreePort` is racy by its own documentation and the child's
 *     bound socket is the only authority;
 *  2. answer every `mlx:request` with the matching `mlx:response`;
 *  3. install a SIGTERM handler that runs `InferenceHost.close()` and exits 0.
 *     Without it the host's temp root is left for the next start-up sweep to
 *     reclaim and the verbose log's tail never reaches disk.
 *
 * Anything else the sidecar posts is ignored, which is why this parses rather
 * than casts.
 */
export function parseChildMessage(message: unknown): ChildToSupervisor | null {
  if (typeof message !== 'object' || message === null) return null;
  const record = message as Record<string, unknown>;
  if (record.kind === 'mlx:ready') {
    if (typeof record.url !== 'string' || typeof record.authToken !== 'string') return null;
    return { kind: 'mlx:ready', url: record.url, authToken: record.authToken };
  }
  if (record.kind === 'mlx:response') {
    if (typeof record.id !== 'number') return null;
    if (record.ok === true) return { kind: 'mlx:response', id: record.id, ok: true, value: record.value };
    if (record.ok === false) {
      // Only a string is trustworthy here: stringifying an object would put
      // `[object Object]` in front of the user as the reason their request
      // failed, which is worse than admitting we do not know.
      const error = typeof record.error === 'string' ? record.error : 'unknown error';
      return { kind: 'mlx:response', id: record.id, ok: false, error };
    }
  }
  return null;
}

/*
 * ---------------------------------------------------------------------------
 * Supervisor surface
 * ---------------------------------------------------------------------------
 */

/**
 * Where the child is in its life. `crashed` is deliberately absent: a crash is
 * an event, and the resting state it leads to is `restarting` or `failed`.
 * Parking in a `crashed` state would mean the supervisor reports something that
 * is not true a moment later.
 */
export type Lifecycle = 'stopped' | 'starting' | 'running' | 'restarting' | 'failed';

/**
 * What the tray and Control Panel show. `lying` is `running` plus the one fact that
 * changes what the user should believe:
 *
 * A class-(b) failure — `mlx_array_eval` / `mlx_synchronize` / `mlx_clear_cache`
 * catching the C++ exception and calling `mlx_trace_native_error` — produces no
 * JS error, no HTTP error, and a `/health` that says `ok`. The output is simply
 * wrong. `MLX_INFERENCE_TRACE=1` plus `MLX_INFERENCE_TRACE_FILE` is the only
 * way that is observable at all, and both must be set at fork time.
 */
export type SupervisorState = Lifecycle | 'lying';

/** One class-(b) native error, parsed out of the trace file. */
export interface NativeError {
  /** The `extern "C"` shim that swallowed it, e.g. `array_eval`. */
  context: string;
  /** `e.what()`, with newlines and quotes flattened to spaces by the writer. */
  detail: string;
  /** When the supervisor observed the line, not when the child wrote it. */
  observedAtMs: number;
}

/** How the last child ended, as reported to the UI. */
export interface ExitRecord {
  verdict: 'clean' | 'crash';
  reason: string;
  code: number | null;
  signal: string | null;
  atMs: number;
  /** Last lines the child wrote before dying, newest last. Best effort. */
  stderrTail: readonly string[];
}

/** Everything the tray and Control Panel need, in one object that cannot get out of sync. */
export interface SupervisorSnapshot {
  state: SupervisorState;
  pid: number | undefined;
  /** Base URL the sidecar reported, or `null` before it is ready. */
  url: string | null;
  /** How many times we have forked. 1 on the first start. */
  generation: number;
  /** Crashes since the last child that stayed up long enough to earn a fresh budget. */
  consecutiveCrashes: number;
  lastExit: ExitRecord | null;
  /** Most recent `/health` body, or `null` if the last poll failed or none has run. */
  health: HealthProbe | null;
  /** Class-(b) errors seen in this generation's trace file, capped and oldest first. */
  nativeErrors: readonly NativeError[];
  /** Path this generation's child was told to write its trace to. */
  traceFile: string | null;
}

/**
 * The subset of `ServerHealth` (packages/server/src/health.ts) the supervisor
 * consumes.
 *
 * Restated rather than imported: `@mlx-node/server/host` does not re-export the
 * type, and the main entry that does pulls `@mlx-node/lm` — which loads the
 * 233 MB native addon into whatever process imports it. Measured: importing
 * `@mlx-node/server/host` maps `mlx-core.darwin-arm64.node` and adds ~29 MB
 * RSS on the spot. The whole point of the three-process split is that MAIN
 * never does that.
 */
export interface HealthProbe {
  status: string;
  [key: string]: unknown;
}

export type SupervisorEvent =
  | { type: 'state'; snapshot: SupervisorSnapshot }
  | { type: 'ready'; url: string }
  | { type: 'crashed'; exit: ExitRecord }
  | { type: 'gave-up'; consecutiveCrashes: number }
  | { type: 'lying'; error: NativeError }
  | { type: 'log'; stream: 'stdout' | 'stderr'; line: string };

export type SupervisorListener = (event: SupervisorEvent) => void;

/** Why a `request()` could not be answered. Every rejection carries one. */
export type RequestFailure = 'not-running' | 'child-exited' | 'timeout';

/**
 * In-flight RPC promises hang forever when the child dies — nothing in either
 * transport rejects them. Every pending call is settled by the supervisor with
 * one of these instead.
 *
 * A rejection, not a `{ ok: false }` envelope. An envelope needs every caller
 * to check a discriminant, and the one caller that forgets renders a dead
 * sidecar as an empty model list or a blank metric — which is the same silent
 * wrongness the `lying` state exists to catch. A rejection cannot be ignored by
 * accident.
 */
export class SidecarRequestError extends Error {
  constructor(
    readonly failure: RequestFailure,
    message: string,
  ) {
    super(message);
    this.name = 'SidecarRequestError';
  }
}
