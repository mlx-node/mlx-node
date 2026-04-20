/**
 * Export-gate assertions for `@mlx-node/core`.
 *
 * Round-5 Codex follow-up: the Metal cache-pool drain function
 * (`clear_cache` in `crates/mlx-core/src/cache_limit.rs`) is annotated
 * `#[napi(namespace = "__internal__")]` on purpose — the drain is a
 * process-wide `mlx_synchronize` routed through the default stream,
 * which does NOT wait on the custom generation streams that per-model
 * threads run on. Calling it while a decode is in flight risks racing
 * live Metal command buffers.
 *
 * The namespace prefix is a deliberate speed-bump: the ONLY safe
 * caller today is `@mlx-node/server`'s idle sweeper (which only
 * triggers after the in-flight counter returns to zero). Every other
 * caller has to opt in by acknowledging the `__internal__` prefix and
 * reading the `@internal` caveat.
 *
 * These tests guard that gate:
 *
 *   1. The root module must NOT expose `clearCache` directly — a
 *      regression where someone drops the `namespace = "__internal__"`
 *      attribute would silently surface the footgun on the public API.
 *   2. The namespace'd form must BE present and callable — a broken
 *      native build (e.g. a `#[napi]` attribute error that drops the
 *      export) would leave `@mlx-node/server`'s sweeper with no drain
 *      path.
 *   3. `memoryStats` stays exposed on the root as a cheap smoke test
 *      that the broader binding is loading at all — if the native
 *      addon is mis-linked both this AND the `__internal__` probe
 *      fail, which localizes the problem.
 */

import { describe, expect, it } from 'vite-plus/test';

// Intentionally read through `require` as well as ESM `import` so the
// test catches both consumption patterns. The `vite.config.ts` alias
// points `@mlx-node/core` at `packages/core/index.cjs`, so both
// resolutions land on the same module instance in the test runtime.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const coreRequire: Record<string, unknown> = require('@mlx-node/core');

describe('@mlx-node/core public export surface', () => {
  it('does NOT expose `clearCache` on the root namespace', () => {
    // A regression here means someone dropped the
    // `#[napi(namespace = "__internal__")]` attribute from
    // `clear_cache` in `crates/mlx-core/src/cache_limit.rs`. That
    // would make the process-wide Metal drain callable from user
    // code without the `__internal__.` speed-bump — exactly the
    // footgun the gate was designed to prevent.
    expect(coreRequire.clearCache).toBeUndefined();
  });

  it('exposes `__internal__.clearCache` as a callable function', () => {
    const internal = coreRequire.__internal__ as { clearCache?: unknown } | undefined;
    expect(internal).toBeDefined();
    expect(typeof internal?.clearCache).toBe('function');
  });

  it('exposes `memoryStats` on the root namespace as a callable function', () => {
    // Acts as a "did the binding load at all" smoke test alongside
    // the `__internal__` check above — a broken build drops BOTH,
    // which is easier to diagnose than one failing in isolation.
    expect(typeof coreRequire.memoryStats).toBe('function');
  });
});
