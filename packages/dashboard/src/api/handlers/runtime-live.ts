import type { RuntimeLive } from '../../live.js';
import type { MainApiContext } from '../context.js';

/** First paint before the first `live` push. Small files, async fs, no database. */
export function handleRuntimeLive(ctx: MainApiContext): Promise<RuntimeLive> {
  return ctx.live.readNow();
}
