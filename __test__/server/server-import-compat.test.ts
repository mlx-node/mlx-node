/**
 * Import-compatibility regression coverage for `@mlx-node/server`
 * against a stale / partial `@mlx-node/core` native binding.
 *
 * # Round-5 Finding A — "server hard-fails at import on stale .node binary"
 *
 * The idle-sweeper used to do `import { __internal__ } from '@mlx-node/core'`
 * at the top of `packages/server/src/idle-sweeper.ts` and dereference
 * `__internal__.clearCache` at MODULE scope. When the loaded
 * `@mlx-node/core` binding lacked the `__internal__` namespace (stale
 * `.node` from a previous branch, partial upgrade, downgrade after a
 * cache-purge), the side-effect threw
 * `TypeError: Cannot read properties of undefined (reading 'clearCache')`
 * from the SERVER's module init. That was unrecoverable from user code:
 * the throw preceded `createServer()`, so not even `idleClearCacheMs: 0`
 * could route around it.
 *
 * The fix (round-5) resolves `__internal__.clearCache` LAZILY inside
 * `createIdleSweeper(...)` and, when the namespace / function is
 * missing, emits a single `console.warn` and installs a no-op drain.
 * These tests guard that contract:
 *
 *   1. Importing `@mlx-node/server` must NOT throw when
 *      `@mlx-node/core.__internal__` is `undefined`.
 *   2. A sweeper created with `delayMs > 0` must still be a usable
 *      object — `beginRequest` / `endRequest` / `close` all no-op,
 *      and `isPending` / `inFlight` stay defined.
 *   3. Exactly ONE `console.warn` fires across the sweeper's lifetime
 *      even if the drain path is exercised multiple times, so a
 *      downgraded binary does not spam stderr on every idle tick.
 *   4. `delayMs <= 0` still short-circuits to the no-op sweeper
 *      WITHOUT triggering the namespace resolution — the opt-out is
 *      purely passive.
 *
 * The test uses `vi.mock('@mlx-node/core', ...)` before dynamically
 * importing `@mlx-node/server` so the mock is observed by the server's
 * idle-sweeper module. Every test block hoists the `vi.mock` call to
 * the top of the module via Vitest's factory pattern.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

/**
 * Shared scratch for the mocked core binding. The factory below reads
 * this object at import time, so tests can flip the shape (presence /
 * absence of `__internal__`) before `vi.resetModules()` + dynamic
 * `import('@mlx-node/server')` re-loads the server.
 */
type MockCoreInternal = { clearCache?: () => void };

/**
 * Mutable override for the `__internal__` export. `undefined` means
 * "the binding lacks the namespace" (the stale-`.node` case Finding A
 * guards). Tests assign into this before their dynamic
 * `import('@mlx-node/server')` to drive the sweeper down each branch.
 *
 * We delegate every OTHER export to the real `@mlx-node/core` via
 * `vi.importActual(...)` because the `@mlx-node/server` dependency
 * chain (server.ts → @mlx-node/lm → many value-level imports like
 * `Qwen35ModelNative`, `ResponseStore`, etc.) would break under a
 * strict hand-rolled mock — and we only care about exercising the
 * `__internal__` shape here anyway.
 */
let mockInternalOverride: { value: MockCoreInternal | undefined } = { value: undefined };
let useOverride = false;

vi.mock('@mlx-node/core', async () => {
  // `importActual` loads the REAL binding so all unrelated exports
  // (every model class, utility, etc.) continue to resolve normally
  // — the server's transitive dep graph stays intact. We wrap the
  // result in an object whose `__internal__` getter forwards to
  // `mockInternalOverride` when the test has opted in, so one test
  // can simulate "stale binary, no namespace" while the next simulates
  // "namespace present with a controlled clearCache spy".
  const actual = (await vi.importActual<Record<string, unknown>>('@mlx-node/core')) as Record<string, unknown>;
  return new Proxy(actual, {
    get(target, prop, receiver): unknown {
      if (prop === '__internal__' && useOverride) {
        return mockInternalOverride.value;
      }
      return Reflect.get(target, prop, receiver);
    },
    has(target, prop): boolean {
      // Always report `__internal__` as a present export — vitest
      // enforces "export was declared" semantics on the module
      // namespace proxy it builds around our factory output. Even
      // when the test simulates a stale binding (override value =
      // `undefined`), the exported slot must EXIST; its VALUE being
      // `undefined` is what the lazy resolver is supposed to handle.
      if (prop === '__internal__') return true;
      return Reflect.has(target, prop);
    },
  });
});

describe('@mlx-node/server import-compat with missing __internal__ (round-5 Finding A)', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // Force a fresh server module graph so the mocked `@mlx-node/core`
    // shape is observed anew. Without this, module caching across
    // tests leaves the sweeper holding a cached `drainFn` closure
    // from whichever test loaded first, AND the module-level
    // `__warnedMissingClearCache` flag in `idle-sweeper.ts` stays
    // `true` across tests, defeating the "warns exactly once per
    // sweeper graph" assertion below.
    vi.resetModules();
    // Silence + observe `console.warn`. The lazy resolver emits a
    // one-time warning when `__internal__.clearCache` is absent.
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // Disable the override by default — individual tests flip it on
    // via `useOverride = true` to exercise a specific shape.
    useOverride = false;
    mockInternalOverride = { value: undefined };
  });

  afterEach(() => {
    warnSpy.mockRestore();
    useOverride = false;
    mockInternalOverride = { value: undefined };
  });

  it('does NOT throw on import when @mlx-node/core lacks __internal__ entirely', async () => {
    // Force the Proxy to return `undefined` for `__internal__`,
    // simulating a downgraded native binding that predates the
    // namespace.
    useOverride = true;
    mockInternalOverride.value = undefined;

    // Dynamic import after the mock shape is set. `resetModules()` in
    // `beforeEach` guarantees we get a freshly-evaluated server module.
    await expect(import('@mlx-node/server')).resolves.toBeDefined();
    // No warning at import time — the resolver is lazy, so the warn
    // fires only when `createIdleSweeper` is actually called with a
    // positive `delayMs`.
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('createIdleSweeper(positive) falls back to a no-op and warns exactly once', async () => {
    useOverride = true;
    mockInternalOverride.value = undefined; // `__internal__` missing
    const { __createIdleSweeper } = await import('@mlx-node/server');
    const sweeper = __createIdleSweeper(10);

    // Sweeper must present the full interface even on the fallback
    // path — callers don't branch on whether the binding is stale.
    expect(typeof sweeper.beginRequest).toBe('function');
    expect(typeof sweeper.endRequest).toBe('function');
    expect(typeof sweeper.close).toBe('function');
    expect(sweeper.isPending).toBe(false);
    expect(sweeper.inFlight).toBe(0);

    // Exactly one warn across the sweeper's lifetime — check the
    // message content so we don't accidentally swallow an unrelated
    // warning as "the" missing-clearCache probe.
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0]?.[0]).toMatch(/__internal__\.clearCache not found on @mlx-node\/core/);

    // Creating a SECOND sweeper in the same process must NOT emit a
    // second warn — the one-time flag persists across sweeper
    // instances, preventing log-spam on a downgraded binary.
    __createIdleSweeper(10);
    expect(warnSpy).toHaveBeenCalledTimes(1);

    sweeper.close();
  });

  it('createIdleSweeper(0) short-circuits without probing __internal__ or warning', async () => {
    // `__internal__` is intentionally absent. If the no-op path
    // resolved the drain callback anyway, the lazy resolver would
    // hit the fallback branch and emit a warn — so asserting zero
    // warns is sufficient proof that the short-circuit runs BEFORE
    // `resolveClearCache()`. (The positive-resolution test below
    // uses a callable `__internal__.clearCache` to confirm the other
    // direction.)
    useOverride = true;
    mockInternalOverride.value = undefined;

    const { __createIdleSweeper } = await import('@mlx-node/server');
    const sweeper = __createIdleSweeper(0);

    expect(sweeper.inFlight).toBe(0);
    expect(sweeper.isPending).toBe(false);
    expect(warnSpy).not.toHaveBeenCalled();
    sweeper.close();
  });

  it('createIdleSweeper(positive) uses the resolved clearCache when __internal__ IS present', async () => {
    const clearCacheSpy = vi.fn();
    useOverride = true;
    mockInternalOverride.value = { clearCache: clearCacheSpy };

    const { __createIdleSweeper } = await import('@mlx-node/server');
    // A very short `delayMs` so the drain fires within the test
    // window. Node timers backed by `unref` do not keep the event
    // loop alive but DO fire on schedule while the loop is active.
    const sweeper = __createIdleSweeper(5);
    // The only path that schedules a drain is the counter transition
    // `inFlight: 1 -> 0`, which is the `endRequest()` that matches a
    // prior `beginRequest()`. Synthesise that transition to drive the
    // resolved clearCache through the closure.
    sweeper.beginRequest();
    sweeper.endRequest();

    // Give the timer a chance to fire. A 30ms wait is comfortably
    // larger than the 5ms drain delay and small enough that the
    // test suite stays fast.
    await new Promise((resolve) => setTimeout(resolve, 30));

    expect(clearCacheSpy).toHaveBeenCalledTimes(1);
    // The resolver-found path must NOT warn — the warn is reserved
    // for the missing-namespace fallback.
    expect(warnSpy).not.toHaveBeenCalled();
    sweeper.close();
  });
});
