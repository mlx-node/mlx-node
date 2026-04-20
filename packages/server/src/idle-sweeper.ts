/**
 * Idle cache-pool sweeper.
 *
 * The MLX Metal allocator holds a process-wide free pool. Without an
 * explicit drain the pool can sit near the wired ceiling (tens of GB on
 * an M3 Max) for the entire server uptime. `@mlx-node/core`'s decode
 * loop already calls `mlx_clear_cache()` every 256 steps, but that
 * cadence only fires while a generation is in progress — idle periods
 * between HTTP requests never get a drain.
 *
 * An earlier iteration put a `ClearCacheOnDrop` RAII guard around every
 * session command so the pool was flushed after every turn. That is
 * wrong on a multi-model server: the free pool is shared across model
 * instances, so flushing after a request on model A discards blocks
 * that model B's next turn is about to reuse.
 *
 * The round-1 replacement was a debounced `touch()`-based sweeper:
 * every request start / end reset a 30s timer that fired `clearCache()`
 * when the timer expired. That design had a correctness bug flagged in
 * round-2 review: a long-running request (streaming decode of 60+
 * seconds) had its 30s timer armed at ARRIVAL, and the timer fired
 * mid-request at t=30s — exactly when the generation thread still
 * needed those blocks. Worse, `clearCache()` routes through
 * `mlx_synchronize()` WITHOUT a stream argument, which only drains the
 * default stream (see `crates/mlx-sys/mlx/mlx/scheduler.cpp`), but
 * decode runs on custom streams — so the drain could race with live
 * command buffers, risking buffer use-after-free.
 *
 * This version fixes both by tracking `inFlight` explicitly. The timer
 * is ONLY armed when the counter returns to zero, and is cancelled the
 * instant any new request arrives — mid-request drains are impossible
 * by construction.
 *
 * # Scope: inference endpoints only
 *
 * The counter is bumped by `/v1/responses` and `/v1/messages` ONLY.
 * Non-inference traffic — `/v1/models`, `/v1/health`, CORS `OPTIONS`
 * preflights, 404s for unknown routes — deliberately does NOT touch
 * the sweeper. Counting those would let a `GET /v1/models` on every
 * client startup, a CORS preflight from a browser, or a cron health
 * probe keep the allocator pinned forever.
 *
 * # Drain is post-request only
 *
 * The only drain path is post-request: `endRequest()` decrements the
 * in-flight counter and, on the 1 -> 0 transition, arms the timer. A
 * cold-idle server that loaded models but never served a request
 * never drains. That is deliberate:
 *
 * - Earlier iterations armed a cold-start drain from either
 *   `createServer()` (regressed on slow model loads > `idleClearCacheMs`,
 *   firing mid-load) or from `ModelRegistry.register()` (round-7 review
 *   surfaced that it still raced sequential multi-model loads — the
 *   timer armed by `register(A)` could fire while `await load(B)` was
 *   still resolving). Loader-bracketing would require API changes on
 *   the user-facing load path, so we remove the cold-start path
 *   entirely instead.
 * - Load-time allocator growth is bounded (a few GB of scratch), not
 *   the tens-of-GB problem the post-request drain targets. macOS
 *   handles truly-idle memory pressure via compression / swap.
 * - Python `mlx-lm` has no cold-start drain either, so this matches
 *   that baseline for servers that never receive a request.
 *
 * # Hot-load bracketing: `withSuspendedDrains`
 *
 * Hot-load flows — a `Model::load()` invoked AFTER the server has
 * already served at least one request — race the post-request drain
 * timer: the t+delayMs timer armed by the previous `endRequest()` can
 * fire MID-LOAD while weight materialization is still allocating
 * through the Metal free pool. The canonical fix is to bracket the
 * load with `withSuspendedDrains(fn)`:
 *
 * ```ts
 * await server.withSuspendedDrains(async () => {
 *   const model = await Qwen35MoeModel.load(modelPath);
 *   server.registry.register('qwen', model);
 * });
 * ```
 *
 * `withSuspendedDrains` handles try/finally bracketing itself, so a
 * thrown load never leaks the internal suspend counter — a footgun the
 * earlier raw `suspendDrains()` / `resumeDrains()` pair exposed
 * (round-9 MEDIUM). The low-level `suspendDrains()` entry point
 * remains available for callers that genuinely need manual control;
 * its returned disposer is token-scoped and idempotent.
 *
 * `withSuspendedDrains` is also thenable-safe: non-Promise thenables
 * (plain objects or even callables that expose a `.then` method) are
 * normalized via `Promise.resolve()` so callers always get a real
 * Promise<T> back and the suspend-release fires on settle. A throwing
 * `.then` getter is treated as non-thenable (the getter access is
 * guarded); the value is returned directly and the suspend releases
 * synchronously — we deliberately do NOT propagate a getter throw, on
 * the theory that leaking the suspend is a far worse failure mode than
 * losing a pathological value (round-10 MEDIUM).
 *
 * # Tuning
 *
 * - The default (30_000 ms) balances "give models a chance to reuse
 *   hot blocks across back-to-back requests" against "don't hold
 *   tens of GB hostage while the process is truly idle".
 * - `idleClearCacheMs: 0` disables the sweeper entirely — useful for
 *   benchmarks or single-model workloads where the only memory
 *   pressure is the decode-loop cadence already in place.
 *   `withSuspendedDrains(fn)` on the disabled sweeper is a pass-through
 *   so call sites can unconditionally bracket.
 * - `MLX_IDLE_CLEAR_CACHE_MS` env var overrides the server's
 *   constructor value; explicit constructor value wins over env.
 *
 * Drain cost is a single `mlx_synchronize` + `mlx_clear_cache` —
 * constant time relative to generation length.
 */

// `clearCache` lives under the `__internal__` NAPI namespace — it is
// deliberately NOT re-exported on the root `@mlx-node/core` object to
// keep the process-wide, custom-stream drain out of the public
// surface. User code that deep-imports from `@mlx-node/core` must
// acknowledge the namespace (`core.__internal__.clearCache`) and
// read the `@internal` caveat there. See
// `crates/mlx-core/src/cache_limit.rs`.
//
// We import the module namespace (NOT a destructured `__internal__`)
// so that a stale `.node` binary missing the `__internal__` namespace
// does NOT hard-fail the import of `@mlx-node/server` itself. The
// `clearCache` symbol is resolved LAZILY inside `createIdleSweeper`
// (see round-5 Finding A): if the namespace / function is absent on
// the loaded binding we warn once and fall back to a no-op drain,
// keeping the server runnable on partial upgrades / downgrades. The
// expected resolution path is `core.__internal__.clearCache` — the
// sweeper never touches the root namespace so there is no ambiguity
// with the deliberately-omitted root-level `clearCache` export.
import * as core from '@mlx-node/core';

/** Default idle window before draining the allocator's free pool (ms). */
export const DEFAULT_IDLE_CLEAR_CACHE_MS = 30_000;

/**
 * Promise/A+ thenable probe that is robust against:
 *
 * - Function-typed thenables (awkward but legal — a callable that
 *   also exposes a `.then` method). The earlier probe only checked
 *   `typeof === 'object'`, so those values fell through to the
 *   synchronous branch and released the suspend before the async
 *   work had even started (round-10 MEDIUM #2).
 * - A throwing `.then` getter. Accessing the property is guarded by
 *   try/catch so a pathological value can't escape before the
 *   caller's `release()` runs. We deliberately prefer "return false
 *   on getter throw" over "propagate" here: the whole point of this
 *   helper is to avoid leaking the suspend counter, and the caller
 *   treats a non-thenable result as immediately-released.
 */
function isThenable(value: unknown): value is PromiseLike<unknown> {
  if (value == null) return false;
  const kind = typeof value;
  if (kind !== 'object' && kind !== 'function') return false;
  try {
    return typeof (value as { then?: unknown }).then === 'function';
  } catch {
    return false;
  }
}

/**
 * Parse `MLX_IDLE_CLEAR_CACHE_MS`. Same semantics as the other env
 * knobs in `server.ts`: finite non-negative integer or fall through
 * to the caller's default. A value of `0` explicitly disables the
 * sweeper; negative / non-integer / unparseable values are ignored.
 */
export function parseIdleClearCacheEnv(): number | undefined {
  const raw = process.env.MLX_IDLE_CLEAR_CACHE_MS;
  if (raw == null || raw === '') return undefined;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 0) return undefined;
  if (!Number.isInteger(parsed)) return undefined;
  return parsed;
}

/**
 * In-flight counter-based idle-drain scheduler.
 *
 * - `beginRequest()` is called ONCE per inference request BEFORE the
 *   native model is dispatched. It increments the in-flight counter
 *   and cancels any pending drain timer — eliminating the mid-request
 *   drain race the debounced-`touch()` design had. Only the inference
 *   endpoints (`/v1/responses` + `/v1/messages`) should call this;
 *   `/v1/models`, `/v1/health`, and CORS preflights MUST NOT, or they
 *   would keep the allocator pinned forever on purely observational
 *   traffic.
 * - `endRequest()` is called ONCE per request in a `finally` block
 *   after the model stream has fully ended (covering success, error,
 *   and client-abort paths). It decrements the counter; when the
 *   counter reaches zero it arms a `delayMs` timer that calls
 *   `onDrain()` on expiry. This is the ONLY path that ever schedules
 *   a drain — see the module-level "Drain is post-request only"
 *   note for why cold-start arming was removed.
 * - `close()` cancels any pending drain — used during graceful
 *   shutdown so the timer does not keep Node alive after
 *   `server.close()`. It intentionally does NOT reset the counter:
 *   draining partway through in-flight requests is the exact failure
 *   mode we're avoiding.
 *
 * Thread-safety is provided by Node's single-threaded event loop —
 * `beginRequest` / `endRequest` are only ever invoked from the HTTP
 * handler (or tests). Every `beginRequest()` MUST be paired with
 * exactly one `endRequest()` on every exit path — a missed
 * `endRequest()` would leave the counter pinned above zero and the
 * drain would never fire.
 */
export interface IdleSweeper {
  /** Mark a request as arrived. Cancels any pending drain. */
  beginRequest(): void;
  /**
   * Mark a request as completed. When the in-flight counter reaches
   * zero, arms a drain timer for `delayMs`.
   */
  endRequest(): void;
  /** Cancel the pending drain. Idempotent. Does NOT reset the counter. */
  close(): void;
  /**
   * Run `fn` with drains suspended. Handles try/finally bracketing so
   * a thrown load never leaks the internal suspend counter. This is
   * the canonical entry point for hot-load flows — a `Model::load()`
   * invoked AFTER the server has already served at least one request.
   * In that scenario the post-request drain timer armed by
   * `endRequest()` (t+delayMs) can otherwise fire MID-LOAD while
   * weight materialization is still allocating through the Metal free
   * pool, racing the allocator state.
   *
   * Accepts both sync and async functions; returns `fn`'s own return
   * value (or resolved promise). On exit (normal or thrown), the
   * suspend token is disposed exactly once — the drain timer is
   * re-armed if `inFlight === 0` and no other suspend is active.
   *
   * Thenable-safe: non-Promise thenables (including function-typed
   * values that expose a `.then` method) are normalized via
   * `Promise.resolve()` so callers always receive a real `Promise<T>`
   * on the async branch; a throwing `.then` getter is guarded and the
   * value is treated as non-thenable rather than leaking the suspend.
   *
   * The common `serve.ts` pattern — load all models before
   * `createServer(...)`, i.e. before any request is served — has no
   * armed timer in the first place and therefore does NOT need to
   * bracket.
   *
   * Pass-through on the disabled sweeper (`delayMs <= 0`): `fn` is
   * invoked directly and its return value is returned unchanged, so
   * call sites can unconditionally bracket without branching.
   *
   * @example
   * ```ts
   * await server.withSuspendedDrains(async () => {
   *   const model = await Qwen35MoeModel.load(modelPath);
   *   server.registry.register('qwen', model);
   * });
   * ```
   */
  withSuspendedDrains<T>(fn: () => Promise<T>): Promise<T>;
  withSuspendedDrains<T>(fn: () => T): T;
  /**
   * Low-level: suspend drains and return an idempotent disposer.
   * Prefer {@link withSuspendedDrains} unless you need manual
   * control over when the suspend is released (for example across
   * async boundaries the caller wants to manage explicitly).
   *
   * Semantics: cancels any pending drain and increments an internal
   * "load" counter. Returns a disposer — calling the disposer
   * decrements the counter. When the counter reaches zero AND
   * `inFlight === 0`, a fresh drain timer is armed so a subsequent
   * idle window still drains.
   *
   * Safe to nest: N suspends require N dispose calls. The returned
   * disposer is token-scoped and idempotent — calling it more than
   * once is a no-op, so it cannot over-decrement and mysteriously
   * shift the idle window.
   *
   * No-op on the disabled sweeper (`delayMs <= 0`): returns a no-op
   * disposer so call sites can unconditionally bracket without
   * branching on whether the sweeper is enabled.
   */
  suspendDrains(): () => void;
  /** Observability hook — `true` while a drain is scheduled. */
  readonly isPending: boolean;
  /** Observability hook — current in-flight request count. */
  readonly inFlight: number;
}

/** Set to `true` after the first missing-binding warn so we don't spam stderr. */
let __warnedMissingClearCache = false;

/**
 * Resolve `__internal__.clearCache` from the loaded `@mlx-node/core`
 * binding, guarded against stale / partial / downgraded `.node` files.
 *
 * Round-5 Finding A: the previous design dereferenced
 * `__internal__.clearCache` at MODULE scope, so the import of
 * `@mlx-node/server` itself threw `TypeError: Cannot read properties
 * of undefined (reading 'clearCache')` if the loaded native binding
 * lacked the namespace. That was unrecoverable from user code —
 * there was no escape hatch, not even `idleClearCacheMs: 0`, because
 * the throw fired before `createServer()` was reached.
 *
 * This resolver probes the namespace defensively at sweeper-creation
 * time (and ONLY when the sweeper is actually enabled, so the
 * `delayMs <= 0` opt-out short-circuits before we even look). Missing
 * namespace / missing function / wrong-type function all route to a
 * one-time `console.warn` + no-op fallback so the server stays up.
 */
function resolveClearCache(): () => void {
  // Read through the module namespace — `core.__internal__` returns
  // `undefined` when the loaded binding predates the namespace, WITHOUT
  // throwing. Using an optional chain on the function keeps the probe
  // safe even if `__internal__` exists but is some exotic shape.
  const fn: unknown = core.__internal__?.clearCache;
  if (typeof fn === 'function') {
    return fn as () => void;
  }
  if (!__warnedMissingClearCache) {
    __warnedMissingClearCache = true;
    // Intentionally a `console.warn`, NOT a throw — the server must
    // stay up on a stale native binding; the user's error surface is
    // stderr, not an uncaught exception on a module-import side-effect.
    console.warn(
      '[@mlx-node/server] __internal__.clearCache not found on @mlx-node/core; idle cache drains disabled. Rebuild @mlx-node/core.',
    );
  }
  return (): void => {};
}

/**
 * Create an idle sweeper. Pass `0` or a non-positive value to opt out —
 * the returned object becomes a no-op that still satisfies the
 * interface. Callers can therefore unconditionally wire
 * `beginRequest()` / `endRequest()` without branching on whether the
 * sweeper is enabled.
 *
 * When `onDrain` is omitted, the sweeper resolves
 * `__internal__.clearCache` on `@mlx-node/core` at creation time and
 * caches the result in the returned closure. A missing namespace /
 * function triggers a one-time `console.warn` and a no-op fallback —
 * see `resolveClearCache()`. The `delayMs <= 0` path skips the
 * resolution entirely so the opt-out remains purely passive.
 */
export function createIdleSweeper(delayMs: number, onDrain?: () => void): IdleSweeper {
  if (!Number.isFinite(delayMs) || delayMs <= 0) {
    // No-op sweeper: no timer to cancel, no counter that could latch.
    // `withSuspendedDrains` becomes a pass-through so call sites can
    // unconditionally bracket without branching on whether the
    // sweeper is enabled. `suspendDrains` returns a no-op disposer
    // for the same reason.
    function passthrough<T>(fn: () => Promise<T>): Promise<T>;
    function passthrough<T>(fn: () => T): T;
    function passthrough<T>(fn: () => T | Promise<T>): T | Promise<T> {
      return fn();
    }
    return {
      beginRequest(): void {},
      endRequest(): void {},
      close(): void {},
      withSuspendedDrains: passthrough,
      suspendDrains(): () => void {
        return (): void => {};
      },
      get isPending(): boolean {
        return false;
      },
      get inFlight(): number {
        return 0;
      },
    };
  }

  // Resolve the drain callback exactly once per sweeper — either the
  // caller-supplied `onDrain` (used by tests for observability) or the
  // guarded `__internal__.clearCache` lookup. Caching in the closure
  // means the namespace probe runs ONCE per sweeper, not once per
  // timer firing.
  const drainFn: () => void = onDrain ?? resolveClearCache();

  let timer: ReturnType<typeof setTimeout> | null = null;
  let inFlight = 0;
  // `loadCounter` tracks active `suspendDrains()` brackets. Any value
  // > 0 means some caller has declared "do not fire a drain right now
  // — a long-running, unbracketed allocator-heavy operation is in
  // progress". `scheduleDrain()` bails when it's positive, and the
  // drain-fire callback re-checks it defensively before calling
  // `drainFn()` in case a suspend arrived between arming and firing
  // on the same event-loop tick.
  let loadCounter = 0;

  const cancelTimer = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const onTimerFire = (): void => {
    timer = null;
    // Double-check: if a request arrived between the timer being
    // armed and the callback firing, `inFlight` is non-zero and
    // `beginRequest()` should already have cancelled us. This guard
    // is belt-and-suspenders — shouldn't trip in practice but
    // protects against a timer racing a synchronous
    // `beginRequest()` in adversarial tests.
    if (inFlight !== 0) return;
    // Equivalent defensive re-check for the suspend path: a
    // `suspendDrains()` that landed between arming and firing
    // (possible if the arming `setTimeout` callback is queued
    // alongside a synchronous suspend in the same tick) should
    // reschedule rather than drain mid-load. The microtask ordering
    // of `setTimeout` + `clearTimeout` makes this nearly impossible
    // in practice on Node's single-threaded loop — but the check is
    // cheap and future-proofs against timer-pool refactors.
    if (loadCounter > 0) {
      // Re-arm for another window. We intentionally do NOT call
      // `drainFn()` here. If the suspend clears before the next
      // `delayMs` elapses, the disposer returned by `suspendDrains()`
      // will cancel this timer and start fresh anyway.
      timer = setTimeout(onTimerFire, delayMs);
      timer.unref();
      return;
    }
    try {
      drainFn();
    } catch {
      // Swallow — drain is best-effort and must not crash the
      // server loop. The underlying FFI call can't currently fail
      // but we defend against future refactors growing fallibility.
    }
  };

  const scheduleDrain = (): void => {
    // Caller must already have confirmed `inFlight === 0`. We
    // defensively null-out the existing timer first so overlapping
    // begin→end→begin→end patterns can't leak a timer.
    cancelTimer();
    // Skip arming while a suspend bracket is active — the matching
    // disposer that takes the counter back to zero will arm a fresh
    // timer on the transition (provided `inFlight === 0`).
    if (loadCounter > 0) return;
    timer = setTimeout(onTimerFire, delayMs);
    // Do not keep the event loop alive on this timer alone. If the
    // server would otherwise exit (e.g. the HTTP listener closed),
    // we do not want to force a last-ditch drain.
    timer.unref();
  };

  /**
   * Suspend drains and return an idempotent, token-scoped disposer.
   * Each call allocates a fresh token; the returned disposer checks
   * the token on every invocation so re-calling the same disposer is
   * a guaranteed no-op. Critically, a *stale* disposer — one that
   * survived its matching release via `withSuspendedDrains` — also
   * cannot re-decrement the counter and mysteriously shift the idle
   * window (round-9 MEDIUM). The `Math.max(0, …)` clamp is retained
   * as defense-in-depth for the impossible case where two different
   * disposers both land after cross-thread reordering.
   */
  const suspendDrains = (): (() => void) => {
    cancelTimer();
    loadCounter += 1;
    const token = { disposed: false };
    return (): void => {
      if (token.disposed) return;
      token.disposed = true;
      loadCounter = Math.max(0, loadCounter - 1);
      if (loadCounter === 0 && inFlight === 0) {
        scheduleDrain();
      }
    };
  };

  /**
   * Bracket `fn` with `suspendDrains()`. Works for both sync and
   * async functions: the sync return path releases the suspend and
   * returns `fn`'s value directly; the promise path releases the
   * suspend once the promise settles. Either way, a throw / rejection
   * releases the suspend before propagating.
   *
   * Round-10 MEDIUM hardening:
   *
   * - Thenable detection also admits `typeof === 'function'` so a
   *   callable that doubles as a thenable (awkward but legal per
   *   Promise/A+) is not treated as a synchronous return.
   * - The `.then` property access is wrapped in try/catch so a
   *   throwing getter does not escape before `release()` runs; such
   *   a value is treated as non-thenable and the suspend releases
   *   synchronously.
   * - Thenables are normalized via `Promise.resolve(result)` so a
   *   plain-object thenable that resolves without returning a Promise
   *   from `.then()` still produces a real `Promise<T>` for the
   *   caller (adopted via the Promise resolution procedure). A
   *   throw from inside `.then()` is captured by `Promise.resolve`
   *   and surfaces as a rejection, so the suspend still releases
   *   via `.finally()`.
   */
  function withSuspendedDrains<T>(fn: () => Promise<T>): Promise<T>;
  function withSuspendedDrains<T>(fn: () => T): T;
  function withSuspendedDrains<T>(fn: () => T | Promise<T>): T | Promise<T> {
    const release = suspendDrains();
    let released = false;
    const releaseOnce = (): void => {
      if (released) return;
      released = true;
      release();
    };
    let result: T | Promise<T>;
    try {
      result = fn();
    } catch (err) {
      releaseOnce();
      throw err;
    }
    if (isThenable(result)) {
      // `Promise.resolve` adopts any thenable via the Promise
      // resolution procedure, so a non-Promise thenable (plain
      // object or callable) still hands the caller back a real
      // `Promise<T>`. `.finally` runs on both fulfil and reject.
      return Promise.resolve(result).finally(releaseOnce) as Promise<T>;
    }
    releaseOnce();
    return result as T;
  }

  return {
    beginRequest(): void {
      inFlight += 1;
      cancelTimer();
    },
    endRequest(): void {
      if (inFlight === 0) {
        // Defensive: an `endRequest()` without a matching
        // `beginRequest()` is a caller bug. Clamp at zero rather
        // than letting the counter go negative and latch the drain
        // off forever.
        return;
      }
      inFlight -= 1;
      if (inFlight === 0) {
        scheduleDrain();
      }
    },
    close(): void {
      cancelTimer();
    },
    withSuspendedDrains,
    suspendDrains,
    get isPending(): boolean {
      return timer !== null;
    },
    get inFlight(): number {
      return inFlight;
    },
  };
}
