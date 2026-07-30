import type { WorkerApiContext } from '../context.js';

export async function handleIngest(ctx: WorkerApiContext): Promise<unknown> {
  return await ctx.runIngest();
}
