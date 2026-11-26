/**
 * The totals of a conversation, summed from its stored turn metrics (`turn_completed.usage` on disk), never from the
 * agent's `x.ai/session/usage`: those reset with the agent process. A rewound turn is not counted (its line is gone).
 */

import type { TurnMetric } from './types';

export interface SessionUsage {
  turns: number;
  /** The rest are null when no turn reported usage. */
  modelCalls: number | null;
  /** Prompt tokens of every model call, cache reads included. */
  readTokens: number | null;
  cachedTokens: number | null;
  writtenTokens: number | null;
  /** Null when no turn reported its time. */
  elapsedMs: number | null;
}

/** A turn with no usage counts as a turn, not as a model call that read and wrote nothing. */
export function sessionUsage(turns: readonly TurnMetric[]): SessionUsage {
  let reported = 0;
  let modelCalls = 0;
  let read = 0;
  let cached = 0;
  let written = 0;
  let timed = 0;
  let elapsed = 0;
  for (const turn of turns) {
    if (turn.elapsedMs !== null) {
      timed += 1;
      elapsed += turn.elapsedMs;
    }
    // `modelCalls` is null when the turn's end carried no `usage` (or counted no call): no tokens to add.
    if (turn.modelCalls === null) continue;
    reported += 1;
    modelCalls += turn.modelCalls;
    read += turn.inputTokens;
    cached += turn.cachedTokens;
    written += turn.outputTokens;
  }
  const some = reported > 0;
  return {
    turns: turns.length,
    modelCalls: some ? modelCalls : null,
    readTokens: some ? read : null,
    cachedTokens: some ? cached : null,
    writtenTokens: some ? written : null,
    elapsedMs: timed > 0 ? elapsed : null,
  };
}
