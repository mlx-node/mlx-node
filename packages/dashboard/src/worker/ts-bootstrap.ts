/**
 * Worker entry used when the dashboard runs straight from TypeScript source —
 * `vitest`, and `oxnode scripts/dev-dashboard.ts`. A built `dist` never reaches
 * here: `defaultWorkerUrl()` picks the real `db-worker.js` whenever it exists.
 *
 * Node type-strips a `.ts` worker entry natively, but it does NOT remap the
 * `./x.js` specifiers `tsc` NodeNext requires us to write, and a worker thread
 * runs outside Vitest's module runner, so Vitest's resolution does not apply
 * either. Without the hook below the worker cannot import a single source file.
 *
 * `module.registerHooks` (Node 22.15+/24) is synchronous and thread-local, so
 * this affects nothing but this worker. Keeping the hook here rather than in
 * `db-worker.ts` leaves the shipped entry with no source-only seam in it.
 *
 * Two details are load-bearing, both verified by breaking them:
 *  - The `.ts` is used only when it actually EXISTS. Rewriting unconditionally
 *    also hits the `./x.js` specifiers inside `node_modules` (drizzle, the pi
 *    SDK), which have no `.ts` sibling — the worker then dies on startup. It also
 *    keeps a genuine built artifact authoritative, so this can never mask a
 *    stale-dist problem.
 *  - `format` must be `module-typescript`, the format that means "ESM, strip the
 *    types". Omitting it works under plain Node but `oxnode` rejects the hook
 *    result outright ("Missing field `format`"); saying `module` satisfies
 *    `oxnode` but then Node parses the TypeScript as JavaScript ("Unexpected
 *    token '{'"). Only `module-typescript` runs under both.
 */

import { existsSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { fileURLToPath } from 'node:url';

registerHooks({
  resolve(spec, ctx, next) {
    if (spec.startsWith('.') && spec.endsWith('.js') && ctx.parentURL !== undefined) {
      const asTs = new URL(`${spec.slice(0, -3)}.ts`, ctx.parentURL);
      if (existsSync(fileURLToPath(asTs))) return { url: asTs.href, format: 'module-typescript', shortCircuit: true };
    }
    return next(spec, ctx);
  },
});

await import(new URL('./db-worker.ts', import.meta.url).href);
