import type { ApiContext } from '../context.js';

export function handleHealth(ctx: ApiContext): unknown {
  return {
    status: 'ok',
    modelsDir: ctx.modelsDir,
    sessionsRoot: ctx.sessionsRoot,
    tracesDir: ctx.tracesDir,
  };
}
