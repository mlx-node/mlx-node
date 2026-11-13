/**
 * Entry of the `mlx_web` server when the dashboard runs from TypeScript source (tests, the dev harness): Node strips
 * the types but does not map the `./x.js` specifiers to their `.ts` files, so this hook does, exactly as
 * `worker/ts-bootstrap.ts` does for the database worker. A built `dist/chat/web/server.js` never comes here.
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

await import(new URL('./server.ts', import.meta.url).href);
