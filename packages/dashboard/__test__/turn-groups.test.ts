import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates } from '../src/agent-sessions/transcript.js';
import { groupTurns, hasTurnWork, openTurnGroup, turnTimeParts, turnTitle } from '../ui/src/lib/sessions.js';
import type { TranscriptEntry, TurnMetric, WakeCause } from '../ui/src/lib/types.js';
import { wokenParentText } from './helpers/woken-turn.js';

const T0 = 1_700_000_000_000;

function metric(overrides: Partial<TurnMetric> = {}): TurnMetric {
  return {
    promptId: 'p',
    ts: T0,
    inputTokens: 1000,
    outputTokens: 200,
    cachedTokens: 0,
    reasoningTokens: 0,
    elapsedMs: 10_000,
    ttftMs: 1200,
    prefillMs: null,
    prefillTokens: null,
    prefillCalls: null,
    modelCalls: null,
    decodeTps: 50,
    model: 'qwen3.8-27b',
    ...overrides,
  };
}

describe('turnTimeParts', () => {
  it('parts sum to the total', () => {
    const parts = turnTimeParts({ elapsedMs: 10_000, ttftMs: 1500, decodeTps: 50, outputTokens: 200 });
    expect(parts).toEqual({ firstTokenMs: 1500, decodeMs: 4000, otherMs: 4500, totalMs: 10_000 });
  });

  it('all null without a duration', () => {
    expect(turnTimeParts({ elapsedMs: null, ttftMs: 1500, decodeTps: 50, outputTokens: 200 })).toEqual({
      firstTokenMs: null,
      decodeMs: null,
      otherMs: null,
      totalMs: null,
    });
  });

  it('decode is clamped to what is left', () => {
    const parts = turnTimeParts({ elapsedMs: 5000, ttftMs: 3000, decodeTps: 10, outputTokens: 100 });
    expect(parts).toEqual({ firstTokenMs: 3000, decodeMs: 2000, otherMs: 0, totalMs: 5000 });
  });
});

describe('groupTurns', () => {
  const user = (ts: number, text: string): TranscriptEntry => ({ kind: 'user', ts, text, promptIndex: null });
  const end = (ts: number): TranscriptEntry => ({ kind: 'turn-end', ts, stopReason: 'end_turn' });

  it('pairs the k-th turn-end with the k-th metric', () => {
    const turns = [
      metric({ promptId: 'a', elapsedMs: 2000 }),
      metric({ promptId: 'b', elapsedMs: null }),
      metric({ promptId: 'c' }),
    ];
    const groups = groupTurns([user(T0, 'one'), end(T0 + 3000), user(T0 + 4000, 'two'), end(T0 + 6000)], turns);
    expect(groups.map((g) => [g.number, g.metric?.promptId, g.entries.length, g.startTs, g.endTs])).toEqual([
      [1, 'a', 1, T0 + 1000, T0 + 3000],
      [2, 'b', 1, T0 + 4000, T0 + 6000],
    ]);
  });

  it('keeps an open trailing group', () => {
    const groups = groupTurns([user(T0, 'one'), end(T0 + 3000), user(T0 + 4000, 'two')], [metric()]);
    expect(groups).toHaveLength(2);
    expect(groups[1]).toMatchObject({ number: 2, end: null, metric: null, startTs: T0 + 4000, endTs: null });
  });

  const spawn = (ts: number): TranscriptEntry => ({
    kind: 'subagent',
    ts,
    childId: 'c1',
    type: 'general',
    description: 'research',
  });

  it('an open woken group starts at its wake, not at a spawn line from the gap before it', () => {
    const wake: TranscriptEntry = { kind: 'wake', ts: T0 + 5000, cause: 'workflow', text: 'hidden' };
    const groups = groupTurns(
      [user(T0, 'one'), end(T0 + 3000), spawn(T0 + 4000), wake, { kind: 'thinking', ts: T0 + 6000, text: 't' }],
      [metric({ elapsedMs: 3000 })],
    );
    expect(groups[1]).toMatchObject({ number: 2, end: null, startTs: T0 + 5000 });
  });

  it('an open group without a timed prompt starts at its own work before its spawn lines', () => {
    const groups = groupTurns(
      [user(T0, 'one'), end(T0 + 3000), spawn(T0 + 4000), { kind: 'assistant', ts: T0 + 7000, text: 'a' }],
      [metric({ elapsedMs: 3000 })],
    );
    expect(groups[1]!.startTs).toBe(T0 + 7000);
  });

  it('an open group with an interjection before its wake starts at the wake', () => {
    const groups = groupTurns(
      [
        user(T0, 'one'),
        end(T0 + 3000),
        { kind: 'user', ts: T0 + 3500, text: 'inter', promptIndex: 1, interjection: true },
        { kind: 'wake', ts: T0 + 5000, cause: 'task', text: 'hidden' },
      ],
      [metric({ elapsedMs: 3000 })],
    );
    expect(groups[1]!.startTs).toBe(T0 + 5000);
  });

  it('an open group that starts with an interjection starts at it, as the server window does', () => {
    const interjection: TranscriptEntry = {
      kind: 'user',
      ts: T0 + 4000,
      text: 'inter',
      promptIndex: 1,
      interjection: true,
    };
    const only = groupTurns([user(T0, 'one'), end(T0 + 3000), interjection], [metric({ elapsedMs: 3000 })]);
    expect(only[1]!.startTs).toBe(T0 + 4000);
    const then = groupTurns(
      [user(T0, 'one'), end(T0 + 3000), interjection, { kind: 'assistant', ts: T0 + 7000, text: 'a' }],
      [metric({ elapsedMs: 3000 })],
    );
    expect(then[1]!.startTs).toBe(T0 + 4000);
    const model: TranscriptEntry = { kind: 'model', ts: T0 + 3500, modelId: 'm' };
    const spawned = groupTurns(
      [user(T0, 'one'), end(T0 + 3000), model, spawn(T0 + 3800)],
      [metric({ elapsedMs: 3000 })],
    );
    expect(spawned[1]!.startTs).toBe(T0 + 3800);
  });

  it('the open woken turn of a real file shape starts at the hidden prompt', () => {
    const parsed = parseUpdates(wokenParentText({ open: true }));
    const groups = groupTurns(parsed.transcript, parsed.turns);
    const open = openTurnGroup(groups);
    const wake = open?.entries.find((entry) => entry.kind === 'wake');
    expect(open?.number).toBe(2);
    expect(wake).toBeDefined();
    expect(open!.entries[0]!.kind).toBe('subagent');
    expect(open!.startTs).toBe(wake!.ts);
  });
});

describe('openTurnGroup', () => {
  const user = (ts: number): TranscriptEntry => ({ kind: 'user', ts, text: 'go', promptIndex: null });
  const end = (ts: number): TranscriptEntry => ({ kind: 'turn-end', ts, stopReason: 'end_turn' });
  const model = (ts: number): TranscriptEntry => ({ kind: 'model', ts, modelId: 'qwen3.8-27b' });

  it('is the trailing group with work, numbered after the finished turns', () => {
    const groups = groupTurns([user(T0), end(T0 + 1000), user(T0 + 2000)], [metric()]);
    expect(openTurnGroup(groups)?.number).toBe(2);
  });

  it('is null when the transcript ends with a turn end', () => {
    expect(openTurnGroup(groupTurns([user(T0), end(T0 + 1000)], [metric()]))).toBeNull();
  });

  it('is null for a trailing model switch, plan or late subagent line', () => {
    const groups = groupTurns([user(T0), end(T0 + 1000), model(T0 + 2000)], [metric()]);
    expect(groups).toHaveLength(2);
    expect(hasTurnWork(groups[1]!.entries)).toBe(false);
    expect(openTurnGroup(groups)).toBeNull();
  });
});

describe('turnTitle', () => {
  type Wake = Extract<TranscriptEntry, { kind: 'wake' }>;
  const wake = (cause: WakeCause, extra: Partial<Wake> = {}): Wake => ({
    kind: 'wake',
    ts: T0,
    cause,
    text: 'hidden',
    ...extra,
  });
  const title = (entry: TranscriptEntry): string => turnTitle([{ kind: 'model', ts: T0, modelId: 'm' }, entry]);

  it('is the first line of a typed prompt, or (no prompt) without one', () => {
    expect(turnTitle([{ kind: 'user', ts: 1, text: '  Fix it\nplease', promptIndex: 0 }])).toBe('Fix it');
    expect(turnTitle([{ kind: 'user', ts: 1, text: '  \n ', promptIndex: 0 }])).toBe('(no prompt)');
    expect(turnTitle([{ kind: 'model', ts: 1, modelId: 'm' }])).toBe('(no prompt)');
    expect(turnTitle([])).toBe('(no prompt)');
  });

  it('says what woke the agent when no prompt was typed', () => {
    const run = (status: string) => ({ workflow: { name: 'deep-research', status } });
    expect(title(wake('workflow', run('complete')))).toBe('Woken by workflow deep-research');
    expect(title(wake('workflow', run('failed')))).toBe('Woken by workflow deep-research · failed');
    expect(title(wake('workflow'))).toBe('Woken by a background workflow');
    expect(title(wake('subagent', { subagent: { childId: 'c1', description: 'Check the build\nthen report' } }))).toBe(
      'Woken by subagent: Check the build',
    );
    expect(title(wake('subagent', { subagent: { childId: 'c1', description: '' } }))).toBe(
      'Woken by a background subagent',
    );
    expect(title(wake('subagent'))).toBe('Woken by a background subagent');
    expect(title(wake('task'))).toBe('Woken by a background task');
    expect(title(wake('notifications'))).toBe('Woken by background notifications');
    expect(title(wake('goal'))).toBe('Started by the goal loop');
    expect(title(wake('unknown'))).toBe('Woken by a background event');
  });

  it('a turn woken by a background workflow is titled by the workflow', () => {
    const parsed = parseUpdates(wokenParentText());
    const groups = groupTurns(parsed.transcript, parsed.turns);
    expect(groups.map((group) => turnTitle(group.entries))).toEqual([
      'Research the topic in depth',
      'Woken by workflow deep-research',
    ]);
  });

  it('a wake that opens the transcript is turn 1, also when a typed interjection follows it', () => {
    const groups = groupTurns(
      [
        { kind: 'model', ts: T0, modelId: 'm' },
        wake('task', { ts: T0 + 10 }),
        { kind: 'assistant', ts: T0 + 100, text: 'a' },
        { kind: 'user', ts: T0 + 145, text: 'stop and check the tests', promptIndex: 1, interjection: true },
        { kind: 'assistant', ts: T0 + 200, text: 'b' },
        { kind: 'turn-end', ts: T0 + 300, stopReason: 'end_turn' },
      ],
      [metric()],
    );
    expect(groups.map((group) => group.number)).toEqual([1]);
    expect(turnTitle(groups[0]!.entries)).toBe('Woken by a background task');
  });

  it('skips an interjection that precedes the wake, and falls back to an interjection only without a prompt', () => {
    const inter: TranscriptEntry = { kind: 'user', ts: T0, text: 'inter', promptIndex: 1, interjection: true };
    expect(turnTitle([inter, wake('workflow', { workflow: { name: 'deep-research', status: 'complete' } })])).toBe(
      'Woken by workflow deep-research',
    );
    expect(turnTitle([inter, { kind: 'user', ts: T0 + 1, text: 'typed', promptIndex: 2 }])).toBe('typed');
    expect(turnTitle([inter, { kind: 'assistant', ts: T0 + 1, text: 'a' }])).toBe('inter');
  });

  it('a group with only a wake entry is a turn and the open one', () => {
    const groups = groupTurns(
      [
        { kind: 'user', ts: T0, text: 'go', promptIndex: 0 },
        { kind: 'turn-end', ts: T0 + 1000, stopReason: 'end_turn' },
        wake('unknown', { ts: T0 + 2000 }),
      ],
      [metric()],
    );
    expect(hasTurnWork(groups[1]!.entries)).toBe(true);
    expect(openTurnGroup(groups)?.number).toBe(2);
  });
});
