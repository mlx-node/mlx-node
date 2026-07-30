/**
 * The supervisor for INFERENCE.
 *
 * Fork it, know whether it is alive, know how it died, restart it, and never
 * lie about its state. Nothing in this file imports an Electron symbol — the
 * one that does is `child-utility.ts`, which is a fork call and no decisions —
 * and nothing imports `@mlx-node/server` either: its `/host` entry statically
 * pulls `@mlx-node/lm`, which maps the 233 MB native addon into whatever
 * process evaluates it. MAIN staying free of that IS the three-process split.
 *
 * Three things here are easy to get wrong and are handled explicitly rather
 * than inferred:
 *
 *   INTENT.   `utilityProcess`'s `kill()` reports exit code 0, which is
 *             byte-identical to a sidecar that returned on its own. Intent is
 *             tracked as a flag set before the kill; `assessExit` never reads
 *             the code to decide. See `state.ts`.
 *
 *   SETTLING. Neither transport rejects an in-flight message when the child
 *             dies — the promise simply never settles. Every pending request
 *             is rejected here, on exit and on its own timeout.
 *
 *   LYING.    A class-(b) failure leaves `/health` saying `ok` and the output
 *             wrong. `trace.ts` watches the two channels that make it visible,
 *             and `running` becomes `lying`. Readiness alone is never enough
 *             to report health.
 */

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

import { buildChildEnv } from './env.js';
import {
  assessExit,
  DEFAULT_RESTART_POLICY,
  isLoopbackHttpUrl,
  isServingStatus,
  nextCrashCount,
  planRestart,
  projectState,
  type ChildIntent,
  type RestartPolicy,
} from './state.js';
import { watchTraceFile, parseSwallowedStderrLine, type SwallowedError, type TraceWatcher } from './trace.js';
import {
  parseChildMessage,
  SidecarRequestError,
  type ChildExit,
  type ChildHandle,
  type ChildTransport,
  type ExitRecord,
  type HealthProbe,
  type Lifecycle,
  type NativeError,
  type SupervisorEvent,
  type SupervisorListener,
  type SupervisorSnapshot,
} from './types.js';

export interface SupervisorTimings {
  /**
   * Budget for the whole of start-up: fork, `mlx:ready` handshake, and the
   * first `/health` that reports a serving status. One deadline rather than
   * three, because a sidecar that is stuck has not told us which step it is
   * stuck in.
   *
   * A model load is NOT inside this budget — the host answers `/health` as
   * soon as its socket is listening, and reports `loading` while a load holds
   * the writer slot.
   */
  readyTimeoutMs: number;
  /** Gap between readiness `/health` polls. */
  healthPollMs: number;
  /** Gap between background `/health` polls once running. */
  healthIntervalMs: number;
  /** Per-`/health` request timeout. */
  healthRequestTimeoutMs: number;
  /** Default per-request RPC timeout; overridable per call. */
  requestTimeoutMs: number;
  /**
   * SIGTERM → SIGKILL escalation window.
   *
   * The default must remain longer than the sidecar host's 5 s HTTP drain
   * budget: after forced connection teardown the child still has to flush its
   * request logs and remove its paged-config temp root before it exits.
   */
  killGraceMs: number;
  /**
   * How long to wait for stdout/stderr EOF after `exit` before reporting the
   * crash. `exit` fires when the process dies, not when its pipes have
   * drained, and for a class-(c) abort the stderr line is the only diagnostic
   * that exists. Electron's `utilityProcess` has no `close` event, so waiting
   * for the streams themselves is the only thing portable across the seam.
   */
  stdioFlushMs: number;
  /** Trace-file scan cadence. */
  tracePollMs: number;
}

export const DEFAULT_TIMINGS: SupervisorTimings = Object.freeze({
  readyTimeoutMs: 60_000,
  healthPollMs: 250,
  healthIntervalMs: 5_000,
  healthRequestTimeoutMs: 2_000,
  requestTimeoutMs: 60_000,
  // 5 s for the sidecar host's HTTP drain, then 3 s for log flush + temp-root
  // cleanup. Still below MAIN's independent whole-app quit deadline.
  killGraceMs: 8_000,
  stdioFlushMs: 50,
  tracePollMs: 1_000,
});

export interface SupervisorOptions {
  /** Absolute path to the sidecar's built JS entry. See `ChildSpec.modulePath`. */
  entry: string;
  args?: readonly string[];
  cwd?: string;
  /** Where per-generation trace files are written. Created if missing. */
  traceDir: string;
  /**
   * Required, with no default. The `utilityProcess` transport is production and
   * the `child_process` one is the fallback; defaulting to either would let the
   * wrong one ship by omission.
   */
  transport: ChildTransport;
  /** Normally `engineEnvFor(LAUNCHER_ENGINE_POLICY)`. See `env.ts` for why it is passed in. */
  enginePolicyEnv?: Readonly<Record<string, string>>;
  /** Supervisor-chosen env that must beat the user's shell. */
  env?: Readonly<Record<string, string>>;
  /** Defaults to `process.env`. Read and copied — never written. */
  baseEnv?: NodeJS.ProcessEnv;
  restart?: Partial<RestartPolicy>;
  timings?: Partial<SupervisorTimings>;
  /** Lines of the child's stderr kept for the crash report. Default 50. */
  stderrTailLines?: number;
}

export interface RequestOptions {
  timeoutMs?: number;
}

export interface Supervisor {
  /**
   * Start, and resolve when the sidecar is serving.
   *
   * Rejects if the supervisor gives up, or if it is stopped before it gets
   * there. It does NOT reject on a single crash — that is what the restart
   * budget is for.
   */
  start(): Promise<void>;
  /** Stop deliberately. Any exit that follows is clean, whatever its code. */
  stop(): Promise<void>;
  /** Stop, clear the crash budget, start. A user-initiated restart is not a crash. */
  restart(): Promise<void>;
  /** Send one request. Never hangs: rejects with {@link SidecarRequestError} instead. */
  request<T = unknown>(payload: unknown, opts?: RequestOptions): Promise<T>;
  snapshot(): SupervisorSnapshot;
  /**
   * The running child's bearer token, or `null` when nothing is serving.
   *
   * Every inference route is gated, so the advertised URL on its own is not a
   * capability — a client given only the URL gets 401 for everything. This is
   * the one way out of the process, kept off {@link SupervisorSnapshot} on
   * purpose so a live credential is never broadcast to listeners, drawn into
   * the tray, or stringified into a diagnostic.
   */
  connectionToken(): string | null;
  /** Subscribe. Returns the unsubscribe. A throwing listener cannot break the supervisor. */
  on(listener: SupervisorListener): () => void;
  /** Stop and release everything. Idempotent. */
  dispose(): Promise<void>;
}

/** Class-(b) errors kept per generation. See {@link recordSwallowed}. */
const MAX_NATIVE_ERRORS = 20;

export function createSupervisor(opts: SupervisorOptions): Supervisor {
  const timings: SupervisorTimings = { ...DEFAULT_TIMINGS, ...opts.timings };
  const policy: RestartPolicy = { ...DEFAULT_RESTART_POLICY, ...opts.restart };
  const baseEnv = opts.baseEnv ?? process.env;
  const stderrTailLines = opts.stderrTailLines ?? 50;

  const listeners = new Set<SupervisorListener>();

  let lifecycle: Lifecycle = 'stopped';
  let generation = 0;
  let child: ChildHandle | null = null;
  let disposed = false;

  // Per generation.
  let intent: ChildIntent = 'run';
  let abortReason: string | null = null;
  let becameReady = false;
  let readyAtMs = 0;
  let url: string | null = null;
  /**
   * The current child's bearer token, in memory only.
   *
   * Deliberately NOT on `SupervisorSnapshot`. The snapshot is broadcast to every
   * listener, rendered into the tray, and stringified whole in diagnostics — any
   * of which would put a live credential somewhere it outlives the process. It
   * is reachable only through {@link Supervisor.connectionToken}, whose one
   * caller hands it straight to the clipboard.
   */
  let authToken: string | null = null;
  let health: HealthProbe | null = null;
  let nativeErrors: NativeError[] = [];
  let traceFile: string | null = null;
  let watcher: TraceWatcher | null = null;
  let stderrTail: string[] = [];
  let exitHandled = false;
  // True between the `exit` event and the end of `finishExit`. The child is
  // already gone, but the verdict has not been reached, so `stop()` can still
  // claim the exit as requested rather than racing it.
  let exitPending = false;
  let stdoutEnded = true;
  let stderrEnded = true;
  const genTimers = new Set<NodeJS.Timeout>();

  // Across generations.
  let consecutiveCrashes = 0;
  let lastExit: ExitRecord | null = null;
  let restartTimer: NodeJS.Timeout | null = null;
  let nextRequestId = 0;
  const pending = new Map<
    number,
    { reject(error: Error): void; resolve(value: unknown): void; timer: NodeJS.Timeout }
  >();
  let startWaiters: { resolve(): void; reject(error: Error): void }[] = [];
  let stopWaiters: (() => void)[] = [];

  function emit(event: SupervisorEvent): void {
    for (const listener of listeners) {
      try {
        listener(event);
      } catch {
        // A subscriber's bug must never propagate into the supervisor: this
        // runs on MAIN's event loop, which must never block or die.
      }
    }
  }

  function snapshot(): SupervisorSnapshot {
    return {
      state: projectState(lifecycle, nativeErrors.length),
      pid: child?.pid,
      url,
      generation,
      consecutiveCrashes,
      lastExit,
      health,
      nativeErrors: [...nativeErrors],
      traceFile,
    };
  }

  function emitState(): void {
    emit({ type: 'state', snapshot: snapshot() });
  }

  function setLifecycle(next: Lifecycle): void {
    lifecycle = next;
    emitState();
  }

  function genTimer(fn: () => void, ms: number): NodeJS.Timeout {
    const timer = setTimeout(() => {
      genTimers.delete(timer);
      fn();
    }, ms);
    genTimers.add(timer);
    return timer;
  }

  function clearGenTimers(): void {
    for (const timer of genTimers) clearTimeout(timer);
    genTimers.clear();
  }

  function settlePending(message: string): void {
    for (const [, entry] of pending) {
      clearTimeout(entry.timer);
      entry.reject(new SidecarRequestError('child-exited', message));
    }
    pending.clear();
  }

  /**
   * Record a swallowed native error from either channel.
   *
   * Capped by keeping the FIRST errors and dropping later ones: once
   * `mlx_array_eval` has swallowed one exception every array downstream of it
   * is suspect, so the later entries are cascade and the first is the one that
   * explains what happened.
   */
  function recordSwallowed(error: SwallowedError): void {
    if (nativeErrors.length >= MAX_NATIVE_ERRORS) return;
    const entry: NativeError = { context: error.context, detail: error.detail, observedAtMs: Date.now() };
    nativeErrors.push(entry);
    emit({ type: 'lying', error: entry });
    emitState();
  }

  function attachLineReader(
    stream: NodeJS.ReadableStream | null,
    streamName: 'stdout' | 'stderr',
    streamGeneration: number,
  ): void {
    if (stream === null) return;
    if (streamName === 'stdout') stdoutEnded = false;
    else stderrEnded = false;
    let buffered = '';
    stream.on('data', (chunk: Buffer | string) => {
      // Same-generation data remains valuable after `exit` while
      // waitForStdioDrain is collecting the crash tail. Once a replacement
      // exists, however, this listener belongs to an old pipe: do not append
      // even a partial line to its buffer, emit it, or mutate the replacement's
      // stderr/native-error state.
      if (generation !== streamGeneration) return;
      buffered += typeof chunk === 'string' ? chunk : chunk.toString('utf8');
      const lines = buffered.split('\n');
      buffered = lines.pop() ?? '';
      for (const line of lines) {
        emit({ type: 'log', stream: streamName, line });
        if (streamName !== 'stderr') continue;
        stderrTail.push(line);
        if (stderrTail.length > stderrTailLines) stderrTail.shift();
        const swallowed = parseSwallowedStderrLine(line);
        if (swallowed !== null) recordSwallowed(swallowed);
      }
    });
    stream.on('end', () => {
      if (generation !== streamGeneration) return;
      if (streamName === 'stdout') stdoutEnded = true;
      else stderrEnded = true;
    });
    stream.on('error', () => {
      // A pipe torn down under a dying child is not an event worth reporting.
      if (generation !== streamGeneration) return;
      if (streamName === 'stdout') stdoutEnded = true;
      else stderrEnded = true;
    });
  }

  async function probeHealth(base: string): Promise<HealthProbe | null> {
    try {
      const response = await fetch(new URL('/health', base), {
        signal: AbortSignal.timeout(timings.healthRequestTimeoutMs),
      });
      if (!response.ok) return null;
      const body: unknown = await response.json();
      if (typeof body !== 'object' || body === null) return null;
      const status = (body as Record<string, unknown>).status;
      return typeof status === 'string' ? (body as HealthProbe) : null;
    } catch {
      // Connection refused, DNS, timeout, non-JSON: all "not serving yet".
      return null;
    }
  }

  function markReady(): void {
    becameReady = true;
    readyAtMs = Date.now();
    clearGenTimers();
    setLifecycle('running');
    if (url !== null) emit({ type: 'ready', url });
    const waiters = startWaiters;
    startWaiters = [];
    for (const waiter of waiters) waiter.resolve();
    scheduleBackgroundHealth();
  }

  function scheduleBackgroundHealth(): void {
    const gen = generation;
    const tick = (): void => {
      if (generation !== gen || child === null || url === null) return;
      void probeHealth(url).then((probe) => {
        if (generation !== gen || child === null) return;
        const changed = probe?.status !== health?.status;
        health = probe;
        // A `/health` that stops answering while the process is alive is a
        // hang, not a crash, and restarting on it would kill a legitimately
        // slow model load — `deriveHealthStatus` reports `loading` precisely so
        // a supervisor waits instead. It is surfaced and left to the user.
        if (changed) emitState();
        genTimer(tick, timings.healthIntervalMs);
      });
    };
    genTimer(tick, timings.healthIntervalMs);
  }

  /**
   * `intent !== 'run'` abandons the handshake, and it is load-bearing rather
   * than tidiness.
   *
   * `markReady` clears the WHOLE generation timer set, and that set is also
   * where termination lives: `killWithEscalation`'s SIGTERM→SIGKILL timer and
   * `stop`'s completion cap. A probe still in flight when `stop()` (or an
   * `abort()`) lands would otherwise resolve `ok`, disarm both, restore
   * `running`, and resolve `start()` — so a child that ignores SIGTERM is never
   * force-killed and `stop()` never settles, hanging restart and app quit.
   *
   * Keyed on `intent` and not on `lifecycle` because `intent` is per-generation
   * and `spawn()` resets it to `'run'`: a restart (user-requested or after a
   * crash) re-arms readiness for the NEW child, while the dying one stays
   * abandoned. It also covers `abort`, which arms the same escalation timer.
   */
  function pollUntilServing(): void {
    const gen = generation;
    if (becameReady || child === null || url === null || intent !== 'run') return;
    void probeHealth(url).then((probe) => {
      if (generation !== gen || becameReady || child === null || intent !== 'run') return;
      health = probe;
      if (probe !== null && isServingStatus(probe.status)) {
        markReady();
        return;
      }
      genTimer(pollUntilServing, timings.healthPollMs);
    });
  }

  function abort(reason: string): void {
    if (child === null || exitHandled || intent === 'stop') return;
    intent = 'abort';
    abortReason = reason;
    killWithEscalation();
  }

  function killWithEscalation(): void {
    const target = child;
    if (target === null) return;
    target.kill();
    genTimer(() => {
      // `killed` on a child flips the moment the signal is SENT, so it cannot
      // answer "is it gone yet"; the exit handler's own flag can.
      if (!exitHandled) target.forceKill();
    }, timings.killGraceMs);
  }

  function onMessage(raw: unknown): void {
    const message = parseChildMessage(raw);
    if (message === null) return;

    if (message.kind === 'mlx:ready') {
      if (becameReady || url !== null) return;
      if (!isLoopbackHttpUrl(message.url)) {
        abort(`sidecar reported a non-loopback url (${message.url})`);
        return;
      }
      url = message.url;
      authToken = message.authToken;
      emitState();
      pollUntilServing();
      return;
    }

    const entry = pending.get(message.id);
    // No entry means the request already timed out and was rejected. Dropping
    // the late reply is the only correct move — resolving a settled promise is
    // a no-op, but re-entering the caller's code is not what it asked for.
    if (entry === undefined) return;
    pending.delete(message.id);
    clearTimeout(entry.timer);
    if (message.ok) entry.resolve(message.value);
    else entry.reject(new SidecarRequestError('child-exited', message.error));
  }

  function onError(error: Error): void {
    if (exitHandled) return;
    // The transports disagree about what follows: Electron documents that
    // `exit` always comes after `error`, while `child_process` may emit `error`
    // for a process that never existed and never emit `exit` at all. Prefer the
    // real exit when it arrives, and synthesise one if it does not, so a failed
    // spawn cannot leave the supervisor parked in `starting` forever.
    abortReason = `failed to spawn: ${error.message}`;
    if (child === null) {
      // `assessExit` only reads `abortReason` when the intent is `abort`, so
      // without this the synthesised exit is judged as an ordinary early death
      // and the crash report loses the only line explaining what happened.
      intent = 'abort';
      onExit({ code: null, signal: null });
      return;
    }
    intent = 'abort';
    genTimer(() => {
      if (!exitHandled) onExit({ code: null, signal: null });
    }, timings.stdioFlushMs);
  }

  function onExit(exit: ChildExit): void {
    if (exitHandled) return;
    exitHandled = true;
    exitPending = true;
    void finishExit(exit);
  }

  /**
   * `exit` fires when the process dies, not when its pipes have drained.
   * Bounded wait so the crash report carries the lines that explain it — for a
   * class-(c) abort they are the only diagnostic there is.
   */
  async function waitForStdioDrain(): Promise<void> {
    if (stdoutEnded && stderrEnded) return;
    await new Promise<void>((resolve) => {
      let settled = false;
      const done = (): void => {
        if (settled) return;
        settled = true;
        clearInterval(poller);
        clearTimeout(cap);
        resolve();
      };
      const poller = setInterval(() => {
        if (stdoutEnded && stderrEnded) done();
      }, 5);
      const cap = setTimeout(done, timings.stdioFlushMs);
      poller.unref();
      cap.unref();
    });
  }

  async function finishExit(exit: ChildExit): Promise<void> {
    // Dropped FIRST, before any await. Holding the handle across the stdio
    // drain would leave `snapshot().pid` naming a reaped process and `request()`
    // accepting work for it, for as long as the drain takes — a small window,
    // but reporting a child that is already gone as running is exactly the kind
    // of lie this component exists to avoid.
    child = null;
    // The credential dies with the child. Dropped here rather than only on the
    // next spawn so it is not sitting in memory for however long the supervisor
    // stays stopped.
    authToken = null;
    await waitForStdioDrain();

    clearGenTimers();
    // One last scan before closing: a swallowed error written milliseconds
    // before the child died would otherwise fall between two polls, and the
    // crash would be reported without the thing that caused it.
    const lastWatcher = watcher;
    watcher = null;
    if (lastWatcher !== null) {
      await lastWatcher.poll().catch(() => {
        /* the file may already be gone */
      });
      lastWatcher.close();
    }
    // Everything from here is synchronous, so `stop()` can no longer
    // interleave and the exit is no longer "pending".
    exitPending = false;

    const assessment = assessExit({
      intent,
      becameReady,
      exit,
      ...(abortReason !== null ? { abortReason } : {}),
    });
    lastExit = {
      verdict: assessment.verdict,
      reason: assessment.reason,
      code: exit.code,
      signal: exit.signal,
      atMs: Date.now(),
      stderrTail: [...stderrTail],
    };

    settlePending(`inference sidecar exited (${assessment.reason})`);
    // The child is gone, which is all `stop()` promised — resolve its waiters
    // in every branch, including the one that schedules a restart. Anything
    // conditional here leaves an app-shutdown await hanging on a crash.
    resolveStopWaiters();

    if (assessment.verdict === 'clean') {
      setLifecycle('stopped');
      rejectStartWaiters(new Error(`inference sidecar ${assessment.reason}`));
      return;
    }

    emit({ type: 'crashed', exit: lastExit });
    const readyForMs = becameReady ? Date.now() - readyAtMs : 0;
    consecutiveCrashes = nextCrashCount(policy, consecutiveCrashes, readyForMs);
    const decision = planRestart(policy, consecutiveCrashes);
    if (decision.action === 'give-up') {
      setLifecycle('failed');
      emit({ type: 'gave-up', consecutiveCrashes });
      rejectStartWaiters(
        new Error(`inference sidecar gave up after ${consecutiveCrashes} crashes: ${assessment.reason}`),
      );
      return;
    }

    setLifecycle('restarting');
    restartTimer = setTimeout(() => {
      restartTimer = null;
      if (disposed || lifecycle !== 'restarting') return;
      spawn();
    }, decision.delayMs);
  }

  function resolveStopWaiters(): void {
    const waiters = stopWaiters;
    stopWaiters = [];
    for (const waiter of waiters) waiter();
  }

  function rejectStartWaiters(error: Error): void {
    const waiters = startWaiters;
    startWaiters = [];
    for (const waiter of waiters) waiter.reject(error);
  }

  function spawn(): void {
    generation += 1;
    const spawnedGeneration = generation;
    intent = 'run';
    abortReason = null;
    becameReady = false;
    readyAtMs = 0;
    url = null;
    // A token belongs to one child. Clearing it here means a copied connect
    // command cannot silently keep pointing at a generation that is gone.
    authToken = null;
    health = null;
    nativeErrors = [];
    stderrTail = [];
    exitHandled = false;
    stdoutEnded = true;
    stderrEnded = true;

    // Everything from here to the transport call can throw SYNCHRONOUSLY, and
    // an escape leaves the supervisor wedged in `starting` with no child, no
    // readiness timer (it is armed below, after the call) and a start waiter
    // nothing can settle — the Start action is then dead for the process's
    // lifetime. Worse, `spawn()` is also called from the restart timer, where
    // there is no caller at all and the throw becomes an uncaughtException on
    // MAIN's event loop.
    //
    // `mkdirSync` is the reachable one today: EACCES, EROFS, ENOSPC, or a plain
    // file sitting at `traceDir` all throw. `utilityProcess.fork` cannot throw
    // in the shipped configuration, but `nodeChildTransport` — the documented
    // escape hatch — does, on an ENOTDIR `cwd` or a NUL byte in the env.
    try {
      // A fresh trace file per generation, so `lying` clears when the child does
      // and a restart is not judged by the previous child's corruption.
      mkdirSync(opts.traceDir, { recursive: true });
      traceFile = join(opts.traceDir, `inference-${generation}.trace.log`);
      watcher = watchTraceFile(traceFile, {
        intervalMs: timings.tracePollMs,
        onError: recordSwallowed,
      });

      setLifecycle('starting');

      child = opts.transport(
        {
          modulePath: opts.entry,
          args: opts.args ?? [],
          env: buildChildEnv({
            baseEnv,
            enginePolicyEnv: opts.enginePolicyEnv ?? {},
            ...(opts.env !== undefined ? { overrides: opts.env } : {}),
            traceFile,
          }),
          ...(opts.cwd !== undefined ? { cwd: opts.cwd } : {}),
        },
        {
          // A transport error can be followed by no exit at all, so onError
          // synthesizes one after a deadline. Electron may still deliver the
          // real exit later — even after that synthetic exit has restarted the
          // supervisor and `exitHandled` belongs to a replacement child.
          // Bind every callback to the generation that owns the transport so
          // a stale event can never mutate the replacement's global state.
          onMessage: (message) => {
            if (generation !== spawnedGeneration || exitHandled) return;
            onMessage(message);
          },
          onExit: (exit) => {
            if (generation !== spawnedGeneration) return;
            onExit(exit);
          },
          onError: (error) => {
            if (generation !== spawnedGeneration) return;
            onError(error);
          },
        },
      );
    } catch (error) {
      // Route it through the same path a failed spawn already takes, so the
      // restart budget, the crash report and the start waiters all see it.
      child = null;
      onError(error instanceof Error ? error : new Error(String(error)));
      return;
    }

    attachLineReader(child.stdout, 'stdout', spawnedGeneration);
    attachLineReader(child.stderr, 'stderr', spawnedGeneration);

    genTimer(() => {
      abort(
        becameReady
          ? `did not settle within ${timings.readyTimeoutMs}ms`
          : url === null
            ? `never announced itself within ${timings.readyTimeoutMs}ms`
            : `announced ${url} but never reported a serving /health within ${timings.readyTimeoutMs}ms`,
      );
    }, timings.readyTimeoutMs);

    emitState();
  }

  function start(): Promise<void> {
    if (disposed) return Promise.reject(new Error('supervisor disposed'));
    if (lifecycle === 'running') return Promise.resolve();
    const waiter = new Promise<void>((resolve, reject) => {
      startWaiters.push({ resolve, reject });
    });
    if (lifecycle === 'stopped' || lifecycle === 'failed') {
      consecutiveCrashes = 0;
      spawn();
    }
    return waiter;
  }

  function stop(): Promise<void> {
    if (restartTimer !== null) {
      clearTimeout(restartTimer);
      restartTimer = null;
    }
    if (child === null) {
      if (exitPending) {
        // It died on its own a moment ago and the verdict is still being
        // reached. Recording the intent now makes that exit a requested one,
        // so an app quit that lands in this window cannot trigger a respawn.
        intent = 'stop';
        return new Promise<void>((resolve) => {
          stopWaiters.push(resolve);
        });
      }
      // Nothing to kill: either never started, already dead, or parked in the
      // restart backoff whose timer we just cancelled.
      if (lifecycle !== 'stopped') setLifecycle('stopped');
      // A `start()` from before the backoff is still pending here, and the
      // contract on `start()` says it rejects when stopped before it becomes
      // ready. Without this it never settles: the caller hangs forever, and if
      // a later generation does reach ready, `markReady` resolves the
      // cancelled start too — reporting success for a start the user stopped.
      // `restart()` is safe because it awaits `stop()` before creating its own
      // waiter.
      rejectStartWaiters(new Error('inference sidecar stopped before it became ready'));
      return Promise.resolve();
    }
    const waiter = new Promise<void>((resolve) => {
      stopWaiters.push(resolve);
    });
    if (intent !== 'stop') {
      intent = 'stop';
      killWithEscalation();
      // SIGKILL is not refusable, so an exit always follows; the cap exists so
      // a caller awaiting shutdown cannot be held hostage by a transport that
      // loses the event. It resolves the promise WITHOUT claiming the child is
      // gone — the state is whatever the exit handler last set.
      genTimer(() => {
        if (!exitHandled) resolveStopWaiters();
      }, timings.killGraceMs * 2);
    }
    return waiter;
  }

  return {
    start,
    stop,
    async restart(): Promise<void> {
      await stop();
      // A restart the user asked for is not evidence the sidecar is unstable.
      consecutiveCrashes = 0;
      await start();
    },
    request<T = unknown>(payload: unknown, requestOpts?: RequestOptions): Promise<T> {
      // `lying` is not refused: it is still `running` underneath, and blocking
      // RPC would break the very UI that has to tell the user their output is
      // suspect.
      if (lifecycle !== 'running' || child === null) {
        return Promise.reject(new SidecarRequestError('not-running', `inference sidecar is ${lifecycle}`));
      }
      const id = (nextRequestId += 1);
      const timeoutMs = requestOpts?.timeoutMs ?? timings.requestTimeoutMs;
      return new Promise<T>((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id);
          reject(new SidecarRequestError('timeout', `inference request ${id} timed out after ${timeoutMs}ms`));
        }, timeoutMs);
        pending.set(id, { resolve: resolve as unknown as (value: unknown) => void, reject, timer });
        child?.postMessage({ kind: 'mlx:request', id, payload });
      });
    },
    snapshot,
    // Gated on the lifecycle as well as the field: `running` is the only state
    // in which there is something on the other end to authenticate against, and
    // handing back a token for anything else invites a connect command that
    // looks valid and is not.
    connectionToken: () => (lifecycle === 'running' ? authToken : null),
    on(listener: SupervisorListener): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    async dispose(): Promise<void> {
      if (disposed) return;
      disposed = true;
      await stop();
      watcher?.close();
      watcher = null;
      clearGenTimers();
      settlePending('supervisor disposed');
      listeners.clear();
    },
  };
}

export { nodeChildTransport } from './child-node.js';
export { buildChildEnv } from './env.js';
export * from './state.js';
export * from './trace.js';
export * from './types.js';
