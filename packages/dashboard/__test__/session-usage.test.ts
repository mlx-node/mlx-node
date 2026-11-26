/**
 * The Context panel's totals: summed from the stored turn metrics only.
 */

import { describe, expect, it } from 'vite-plus/test';

import { TranscriptReducer } from '../src/agent-sessions/transcript.js';
import { sessionUsage } from '../ui/src/lib/session-usage.js';
import type { TurnMetric } from '../ui/src/lib/types.js';

function turn(extra: Partial<TurnMetric> = {}): TurnMetric {
  return {
    promptId: 'p',
    ts: 1,
    inputTokens: 0,
    outputTokens: 0,
    cachedTokens: 0,
    reasoningTokens: 0,
    elapsedMs: null,
    ttftMs: null,
    prefillMs: null,
    prefillTokens: null,
    prefillCalls: null,
    modelCalls: null,
    decodeTps: null,
    model: null,
    ...extra,
  };
}

describe('sessionUsage', () => {
  it('sums the turns: prompt tokens with their cache reads, written tokens, model calls and turn time', () => {
    expect(
      sessionUsage([
        turn({ inputTokens: 1_000, cachedTokens: 800, outputTokens: 40, modelCalls: 3, elapsedMs: 2_000 }),
        turn({ inputTokens: 5_000, cachedTokens: 4_000, outputTokens: 60, modelCalls: 2, elapsedMs: 500 }),
      ]),
    ).toEqual({
      turns: 2,
      modelCalls: 5,
      readTokens: 6_000,
      cachedTokens: 4_800,
      writtenTokens: 100,
      elapsedMs: 2_500,
    });
  });

  it('a turn with no usage counts as a turn, not as a call that read and wrote nothing', () => {
    expect(
      sessionUsage([turn({ inputTokens: 100, outputTokens: 5, modelCalls: 1 }), turn({ elapsedMs: 900 })]),
    ).toEqual({ turns: 2, modelCalls: 1, readTokens: 100, cachedTokens: 0, writtenTokens: 5, elapsedMs: 900 });
    expect(sessionUsage([turn(), turn()])).toEqual({
      turns: 2,
      modelCalls: null,
      readTokens: null,
      cachedTokens: null,
      writtenTokens: null,
      elapsedMs: null,
    });
  });

  it('an empty list is no turns and nothing known', () => {
    expect(sessionUsage([])).toEqual({
      turns: 0,
      modelCalls: null,
      readTokens: null,
      cachedTokens: null,
      writtenTokens: null,
      elapsedMs: null,
    });
  });

  it('reads the metrics the reducer makes of stored `turn_completed` lines, with and without usage', () => {
    const reducer = new TranscriptReducer({ now: () => 1 });
    const line = (update: Record<string, unknown>, ts: number) =>
      reducer.apply({ xai: true, update, meta: { agentTimestampMs: ts } });
    line(
      {
        sessionUpdate: 'turn_completed',
        prompt_id: 'a',
        elapsed_ms: 1_200,
        usage: { inputTokens: 900, outputTokens: 30, cachedReadTokens: 700, reasoningTokens: 0, modelCalls: 2 },
      },
      1,
    );
    line({ sessionUpdate: 'turn_completed', prompt_id: 'b', elapsed_ms: 300 }, 2);
    expect(sessionUsage(reducer.snapshot().turns)).toEqual({
      turns: 2,
      modelCalls: 2,
      readTokens: 900,
      cachedTokens: 700,
      writtenTokens: 30,
      elapsedMs: 1_500,
    });
  });
});
