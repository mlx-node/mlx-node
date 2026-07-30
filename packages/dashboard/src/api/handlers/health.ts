import type { ApiPaths } from '../context.js';

export function handleHealth(ctx: ApiPaths): unknown {
  return {
    status: 'ok',
    modelsDir: ctx.modelsDir,
    sessionsRoot: ctx.sessionsRoot,
    tracesDir: ctx.tracesDir,
  };
}
