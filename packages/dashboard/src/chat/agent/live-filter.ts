/**
 * Live update lines that must not reach the reducer. The agent sends a `model_changed` during every `session/new`
 * (before the reply) and after every `session/load`, and never stores it: passed on, it would put "Model switched"
 * at the top of every conversation and make the live entry list differ from the disk list.
 */

import type { UpdateLine } from '../../agent-sessions/transcript.js';

export interface LiveFilterContext {
  /** A turn of this conversation is running. */
  turnRunning: boolean;
  /** The conversation's current model id. */
  modelId: string;
}

export function liveFilter(line: UpdateLine, ctx: LiveFilterContext): 'pass' | 'drop' {
  const updateMeta = line.update._meta;
  const replayInUpdate =
    typeof updateMeta === 'object' && updateMeta !== null && (updateMeta as Record<string, unknown>).isReplay === true;
  if (line.meta?.isReplay === true || replayInUpdate) return 'drop';
  if (line.xai && line.update.sessionUpdate === 'model_changed') {
    if (!ctx.turnRunning) return 'drop';
    if (line.update.model_id === ctx.modelId) return 'drop';
  }
  return 'pass';
}
