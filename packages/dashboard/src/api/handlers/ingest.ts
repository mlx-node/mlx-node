import type { ApiContext } from '../context.js';

export async function handleIngest(ctx: ApiContext): Promise<unknown> {
  return await ctx.runIngest();
}
