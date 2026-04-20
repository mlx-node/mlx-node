/**
 * Round-8/9 regression coverage for the hot-load bracketing API on
 * `IdleSweeper`: `withSuspendedDrains(fn)` (canonical) and the
 * low-level, token-scoped `suspendDrains()` disposer.
 *
 * # Scenario
 *
 * Even with the round-7 "post-request drain only" design (no cold-start
 * arming from `createServer()` or `ModelRegistry.register()`), an
 * unbracketed hot-load still races the drain timer:
 *
 *   request served → endRequest() arms timer (t+delayMs)
 *   user calls Model.load('/path/to/B') (takes > delayMs)
 *   t+delayMs: drain fires MID-LOAD, corrupting Metal allocator state
 *     during weight materialization
 *
 * The fix is the suspend API: `withSuspendedDrains(fn)` brackets the
 * load (cancels the armed timer before entering the load, re-arms on
 * the way out, and releases the suspend even if `fn` throws). The
 * low-level `suspendDrains()` returns a token-scoped disposer for
 * callers that need manual control.
 *
 * These tests use `vi.useFakeTimers()` to drive the underlying
 * `setTimeout` deterministically. They cover:
 *
 *   1. Base race: without suspend, drain fires at t+delayMs.
 *   2. Fix (low-level): suspend between endRequest and timer expiry
 *      prevents the drain; disposer re-arms without needing a new
 *      request.
 *   3. Double-dispose is a no-op (token-scoped idempotency).
 *   4. Nested suspends: N suspends require N dispose calls before the
 *      drain re-arms.
 *   5. Suspend during active request: endRequest does NOT arm while
 *      suspended; disposer (at counter 0 + inFlight 0) arms.
 *   6. `withSuspendedDrains` happy path: suspend prevents drain,
 *      resolve re-arms.
 *   7. `withSuspendedDrains` error safety: fn throws, counter still
 *      decrements; drain re-arms if idle.
 *   8. `withSuspendedDrains` with async fn: await works correctly.
 *   9. Nested `withSuspendedDrains`: inner throws, outer catches;
 *      counter decrements by exactly 2.
 *   10. Sync fn via `withSuspendedDrains`: returns the synchronous
 *       value without awaiting.
 *   11. No-op sweeper: `createIdleSweeper(0)` returns a sweeper whose
 *       `withSuspendedDrains` is pass-through and `suspendDrains()`
 *       returns a no-op disposer; neither throws.
 */

import { __createIdleSweeper } from '@mlx-node/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

describe('IdleSweeper suspendDrains disposer (round-8/9 hot-load race)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('base race: without suspend, the drain fires at t+delayMs after endRequest', () => {
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();
    expect(sweeper.isPending).toBe(true);

    vi.advanceTimersByTime(30);

    // Demonstrates the race: if `Model.load()` were running right
    // now with no bracket, `drainFn()` would be invoked mid-load —
    // which is exactly the problem `withSuspendedDrains()` fixes.
    expect(drainFn).toHaveBeenCalledTimes(1);
    expect(sweeper.isPending).toBe(false);
  });

  it('suspend between endRequest and timer expiry prevents the drain; disposer re-arms', () => {
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();
    expect(sweeper.isPending).toBe(true);

    // Simulate the user calling `Model.load()` right after the
    // request returned — they bracket with `suspendDrains()`.
    const dispose = sweeper.suspendDrains();
    expect(sweeper.isPending).toBe(false);

    // Even if the full `delayMs` would have elapsed during the load,
    // the drain must NOT fire.
    vi.advanceTimersByTime(30);
    expect(drainFn).not.toHaveBeenCalled();
    expect(sweeper.isPending).toBe(false);

    // Load finishes — disposer re-arms. No fresh `beginRequest()` is
    // required: the suspend → release transition (loadCounter 1 → 0
    // while inFlight === 0) is itself an arm trigger, so the next
    // idle window is still covered.
    dispose();
    expect(sweeper.isPending).toBe(true);

    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
    expect(sweeper.isPending).toBe(false);
  });

  it('disposer is idempotent: calling it twice does not double-decrement the suspend counter', () => {
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();

    const disposeA = sweeper.suspendDrains();
    const disposeB = sweeper.suspendDrains();

    // Call `disposeA` twice — the second call must be a no-op.
    // Without per-disposer token tracking, the second call would
    // decrement past `disposeB`'s bracket, arming the drain while the
    // second bracket is still conceptually active.
    disposeA();
    disposeA();
    expect(sweeper.isPending).toBe(false);

    // Only `disposeB`'s decrement should bring the counter to zero.
    disposeB();
    expect(sweeper.isPending).toBe(true);

    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('nested suspends: two suspends + one dispose keeps drains suspended; second dispose arms', () => {
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();

    const releaseA = sweeper.suspendDrains();
    const releaseB = sweeper.suspendDrains();
    expect(sweeper.isPending).toBe(false);

    releaseA();
    // Still suspended — one bracket is active.
    expect(sweeper.isPending).toBe(false);
    vi.advanceTimersByTime(30);
    expect(drainFn).not.toHaveBeenCalled();

    releaseB();
    // Now the outermost bracket unwinds — arm.
    expect(sweeper.isPending).toBe(true);
    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('suspend during active request: endRequest does not arm while suspended, disposer arms', () => {
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    expect(sweeper.inFlight).toBe(1);

    // Suspend mid-request — common in a flow where the admin
    // endpoint that triggers a hot-load coexists with an in-flight
    // inference request. The suspend cancels nothing (no timer
    // armed yet; `beginRequest` already cleared any pending timer)
    // but bumps the internal load counter.
    const release = sweeper.suspendDrains();

    sweeper.endRequest();
    expect(sweeper.inFlight).toBe(0);
    // `inFlight` just transitioned to zero — WITHOUT the loadCounter
    // check, this would arm a drain timer that fires mid-load. With
    // the fix, `scheduleDrain()` bails because `loadCounter > 0`.
    expect(sweeper.isPending).toBe(false);

    vi.advanceTimersByTime(30);
    expect(drainFn).not.toHaveBeenCalled();

    // Load finishes. `loadCounter` → 0 AND `inFlight === 0`, so
    // the disposer arms a drain.
    release();
    expect(sweeper.isPending).toBe(true);

    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('withSuspendedDrains happy path: brackets fn, suspends drain, resolve re-arms', async () => {
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();
    expect(sweeper.isPending).toBe(true);

    // Simulate an async `Model.load()` that runs for 50ms — longer
    // than the 30ms drain window. Without the bracket, drainFn would
    // fire at t+30ms while the load is still allocating.
    const fn = vi.fn(async () => {
      await vi.advanceTimersByTimeAsync(50);
      return 'loaded';
    });

    const pending = sweeper.withSuspendedDrains(fn);
    expect(sweeper.isPending).toBe(false);

    const value = await pending;
    expect(value).toBe('loaded');
    expect(fn).toHaveBeenCalledTimes(1);

    // After the bracket unwinds, fresh drain timer is armed because
    // inFlight === 0.
    expect(sweeper.isPending).toBe(true);
    expect(drainFn).not.toHaveBeenCalled();

    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('withSuspendedDrains error safety: fn throws, counter still decrements and drain re-arms', async () => {
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();
    expect(sweeper.isPending).toBe(true);

    const boom = new Error('load failed');
    await expect(
      sweeper.withSuspendedDrains(async () => {
        await vi.advanceTimersByTimeAsync(5);
        throw boom;
      }),
    ).rejects.toBe(boom);

    // The thrown load MUST release the suspend — otherwise
    // `loadCounter` stays > 0 forever and every future
    // `endRequest()` → `scheduleDrain()` bails out, silently killing
    // all future drains (round-9 MEDIUM #2).
    expect(sweeper.isPending).toBe(true);
    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('withSuspendedDrains handles a sync-throwing fn too', () => {
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();

    const boom = new Error('sync boom');
    expect(() =>
      sweeper.withSuspendedDrains((): string => {
        throw boom;
      }),
    ).toThrow(boom);

    // Sync throw path must also release the suspend.
    expect(sweeper.isPending).toBe(true);
    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('withSuspendedDrains returns sync fn value directly without awaiting', () => {
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();

    // A sync `fn` that returns a non-promise must take the
    // immediate-release branch and return its value unchanged —
    // callers should not have to `await` a trivially-sync helper.
    const value: number = sweeper.withSuspendedDrains(() => 42);
    expect(value).toBe(42);

    // After the sync bracket, drain re-arms.
    expect(sweeper.isPending).toBe(true);
    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('nested withSuspendedDrains: inner throws, outer propagates; counter decrements by exactly 2', async () => {
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();

    const boom = new Error('inner boom');

    await expect(
      sweeper.withSuspendedDrains(async () => {
        // Inside the outer bracket, loadCounter === 1.
        expect(sweeper.isPending).toBe(false);
        await sweeper.withSuspendedDrains(async () => {
          // Now loadCounter === 2.
          expect(sweeper.isPending).toBe(false);
          throw boom;
        });
      }),
    ).rejects.toBe(boom);

    // Both brackets must have released (inner → 1, outer → 0) even
    // though the inner threw. If either disposer leaked, the drain
    // would stay suppressed forever.
    expect(sweeper.isPending).toBe(true);
    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('withSuspendedDrains accepts a plain-object thenable and normalizes it to a real Promise<T>', async () => {
    // Round-10 MEDIUM #1: a non-Promise thenable whose `.then` never
    // returns anything used to break the awaitable contract (the
    // earlier impl called `result.then(...)` directly, so its own
    // return value — often `undefined` — became what the caller
    // awaited). The fix wraps the return in `Promise.resolve(result)`
    // so the adopt path always produces a real `Promise<T>`.
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();

    const thenable = {
      // oxlint-disable-next-line unicorn/no-thenable
      then(resolve: (value: string) => void): void {
        // A void-returning `.then` that resolves asynchronously —
        // the classic shape for e.g. legacy jQuery Deferreds.
        setTimeout(() => {
          resolve('x');
        }, 10);
      },
    };

    const pending = sweeper.withSuspendedDrains(() => thenable as unknown as Promise<string>);
    // Must be suspended — the thenable is async.
    expect(sweeper.isPending).toBe(false);

    // `pending` MUST be awaitable and MUST resolve to `'x'`.
    await vi.advanceTimersByTimeAsync(10);
    const value = await pending;
    expect(value).toBe('x');

    // Counter correctly decremented after settlement.
    expect(sweeper.isPending).toBe(true);
    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('withSuspendedDrains treats a function-typed thenable as async and defers release until settle', async () => {
    // Round-10 MEDIUM #2: a callable that also has a `.then` method
    // used to be treated as synchronous because only
    // `typeof === 'object'` was checked. With the fix,
    // `typeof === 'function'` is also admitted so the suspend does
    // not release until the thenable settles.
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();

    // Build a callable-thenable: a function with a `.then` property.
    // Typed loosely as `unknown` because the `.then` shape here is
    // intentionally non-standard (ignores the reject callback) —
    // that's the whole point of the round-10 robustness fix.
    const callableThenable: unknown = Object.assign(
      (): void => {
        // deliberately unused — the callable-side should NOT be
        // invoked synchronously by `withSuspendedDrains`.
      },
      {
        // oxlint-disable-next-line unicorn/no-thenable
        then(resolve: (value: string) => void): void {
          setTimeout(() => {
            resolve('callable'); // resolves async
          }, 20);
        },
      },
    );

    // NOTE: `fn` must *return* the callable-thenable (not call it).
    const pending = sweeper.withSuspendedDrains(() => callableThenable as unknown as Promise<string>);
    // Crucial: the suspend MUST still be active. The earlier buggy
    // check would have released here synchronously.
    expect(sweeper.isPending).toBe(false);

    await vi.advanceTimersByTimeAsync(20);
    await expect(pending).resolves.toBe('callable');

    // Release only fires after the async settle.
    expect(sweeper.isPending).toBe(true);
    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('withSuspendedDrains handles a throwing `.then` getter by treating the value as non-thenable', () => {
    // Round-10 MEDIUM #3: an exotic value whose `.then` getter
    // throws used to escape before `release()` ran, leaking the
    // suspend counter forever. The fix guards the property access
    // inside `isThenable` with try/catch; the value is treated as
    // non-thenable and the suspend releases synchronously. We
    // intentionally do NOT propagate the getter error — preserving
    // the release is strictly more important than surfacing a
    // pathological value.
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();

    // oxlint-disable-next-line unicorn/no-thenable
    const mischief = Object.defineProperty({} as { then?: unknown }, 'then', {
      get(): never {
        throw new Error('boom');
      },
    });

    // Must not throw — the mischief value is returned directly and
    // the suspend releases synchronously.
    const value = sweeper.withSuspendedDrains(() => mischief);
    expect(value).toBe(mischief);

    // Suspend released → timer armed → next window drains.
    expect(sweeper.isPending).toBe(true);
    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('withSuspendedDrains surfaces a rejection when `.then()` itself throws synchronously', async () => {
    // Round-10 MEDIUM #5: a thenable whose `.then(...)` invocation
    // throws synchronously (as opposed to calling its reject
    // callback) would previously bypass the `.catch` branch — the
    // synchronous throw escaped the `(result as Promise<T>).then(...)`
    // call before the handlers were wired up. The fix normalizes
    // via `Promise.resolve(result)`, which captures any throw from
    // inside `.then()` and surfaces it as a rejection. The suspend
    // still releases via `.finally()`.
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();

    const thrower = {
      // oxlint-disable-next-line unicorn/no-thenable
      then(): never {
        throw new Error('then-throw');
      },
    };

    const pending = sweeper.withSuspendedDrains(() => thrower as unknown as Promise<never>);
    expect(sweeper.isPending).toBe(false);

    // Caller awaits and receives the rejection.
    await expect(pending).rejects.toThrow('then-throw');

    // Counter decrements via `.finally()` → drain re-arms.
    expect(sweeper.isPending).toBe(true);
    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('withSuspendedDrains releases the suspend and re-throws when fn throws synchronously (idempotent with round-9 coverage)', () => {
    // Round-10 MEDIUM #4: explicit coverage that a synchronous
    // throw from `fn()` itself is caught, the suspend is released
    // ONCE, and the error propagates unchanged. Overlaps with the
    // round-9 "sync-throwing fn" test intentionally — this variant
    // additionally asserts that the `inFlight` counter is untouched
    // and that the re-armed timer still fires exactly once.
    const drainFn = vi.fn();
    const sweeper = __createIdleSweeper(30, drainFn);

    sweeper.beginRequest();
    sweeper.endRequest();

    const boom = new Error('sync');
    expect(() =>
      sweeper.withSuspendedDrains((): never => {
        throw boom;
      }),
    ).toThrow(boom);

    // Counter back to zero (no lingering suspend).
    expect(sweeper.inFlight).toBe(0);
    expect(sweeper.isPending).toBe(true);
    vi.advanceTimersByTime(30);
    expect(drainFn).toHaveBeenCalledTimes(1);
  });

  it('no-op sweeper (delayMs = 0): withSuspendedDrains is pass-through; suspendDrains returns a no-op disposer', async () => {
    const sweeper = __createIdleSweeper(0);

    // `suspendDrains()` must return a callable that is safe to
    // invoke multiple times and throws nothing.
    const dispose = sweeper.suspendDrains();
    expect(typeof dispose).toBe('function');
    expect(() => dispose()).not.toThrow();
    expect(() => dispose()).not.toThrow();
    expect(() => sweeper.suspendDrains()()).not.toThrow();

    // `withSuspendedDrains` must invoke `fn` directly and return its
    // value unchanged — sync form.
    expect(sweeper.withSuspendedDrains(() => 99)).toBe(99);

    // Async form is awaited and resolves to `fn`'s resolved value.
    await expect(sweeper.withSuspendedDrains(async () => 'ok')).resolves.toBe('ok');

    // Thrown `fn` still propagates even on the no-op sweeper.
    const boom = new Error('noop boom');
    await expect(
      sweeper.withSuspendedDrains(async () => {
        throw boom;
      }),
    ).rejects.toBe(boom);

    // And the disabled sweeper still presents the full interface.
    expect(sweeper.inFlight).toBe(0);
    expect(sweeper.isPending).toBe(false);
  });
});
