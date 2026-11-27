import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates } from '../src/agent-sessions/transcript.js';
import { readPdfCommand } from '../src/chat/agent/pdf-tool.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import {
  CONTRACT_RETRY_TITLE,
  groupTurns,
  hasTurnWork,
  openTurnGroup,
  turnTitle,
  WAKE_ENDED,
} from '../ui/src/lib/sessions.js';
import type { TranscriptEntry, TurnMetric, WakeCause } from '../ui/src/lib/types.js';
import { answerLine, eventsText, turnLine, userLine } from './helpers/agent-session-files.js';
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

  // Mutation: the Read it now command left as the title (its JSON shows) → red.
  it('reads a Read it now command as the row it draws, by the name of its PDF', () => {
    const command = readPdfCommand('laz3a7wn-40', 'What are the findings?', 'report.pdf');
    const turn: TranscriptEntry[] = [{ kind: 'user', ts: 1, text: command, promptIndex: 0 }];
    expect(turnTitle(turn, new Map([['laz3a7wn-40', 'report.pdf']]))).toBe('You started reading report.pdf');
    expect(turnTitle(turn)).toBe(CHAT.pdf.hostStartBare);
  });

  it('says what woke the agent when no prompt was typed', () => {
    const run = (status: string) => ({
      workflow: {
        runId: 'wf_1',
        revision: 9,
        name: 'deep-research',
        snapshot: { status, agentsUsed: 3, agentBudget: 128, agentsFailed: 0, elapsedMs: 1_000 },
      },
    });
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

  it('a Send now that missed its answer ran as its own turn (#3)', () => {
    expect(title(wake('message'))).toBe('Your message ran as its own turn');
  });

  // Mutation: name only the first label of several → the joined labels are missing.
  it('names the commands and monitors whose end woke the agent (H-b)', () => {
    const cmd = (id: string, label: string) => ({ id, kind: 'command' as const, label });
    const mon = (id: string, label: string) => ({ id, kind: 'monitor' as const, label });
    const named = (what: string, labels: string) => fill(WAKE_ENDED.named, { what, labels });
    expect(title(wake('task', { task: cmd('t1', 'npm test') }))).toBe(named(WAKE_ENDED.command, 'npm test'));
    expect(title(wake('task', { task: mon('t1', 'Watch the build log') }))).toBe(
      named(WAKE_ENDED.monitor, 'Watch the build log'),
    );
    expect(title(wake('task', { task: cmd('t1', '') }))).toBe(WAKE_ENDED.command);
    expect(title(wake('task', { task: cmd('t1', 'echo \u202Egood\nrm -rf x') }))).toBe(
      named(WAKE_ENDED.command, 'echo \\u{202E}good'),
    );
    expect(title(wake('notifications', { tasks: [cmd('t1', 'npm test')] }))).toBe(
      named(WAKE_ENDED.command, 'npm test'),
    );
    expect(title(wake('notifications', { tasks: [cmd('t1', 'npm test'), cmd('t2', 'cargo build')] }))).toBe(
      named(fill(WAKE_ENDED.commands, { n: 2 }), 'npm test, cargo build'),
    );
    expect(title(wake('notifications', { tasks: [mon('t1', 'a'), mon('t2', 'b')] }))).toBe(
      named(fill(WAKE_ENDED.monitors, { n: 2 }), 'a, b'),
    );
    expect(title(wake('notifications', { tasks: [cmd('t1', 'a'), mon('t2', '')] }))).toBe(
      named(fill(WAKE_ENDED.mixed, { n: 2 }), 'a'),
    );
    expect(title(wake('notifications', { tasks: [] }))).toBe('Woken by background notifications');
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
    const workflow = { runId: 'wf_1', revision: 9, name: 'deep-research', snapshot: null };
    expect(turnTitle([inter, wake('workflow', { workflow })])).toBe('Woken by workflow deep-research');
    expect(turnTitle([inter, { kind: 'user', ts: T0 + 1, text: 'typed', promptIndex: 2 }])).toBe('typed');
    expect(turnTitle([inter, { kind: 'assistant', ts: T0 + 1, text: 'a' }])).toBe('inter');
  });

  /** The workflow host's retry prompt, as a resumed helper attempt stores it. */
  const REASON =
    'final message did not contain valid JSON (expected a ```json fenced block): control character (\\u0000-\\u001F) found while parsing a string at line 15 column 0';
  const RETRY_TEXT = `Your final message did not satisfy the output contract: ${REASON}\nReply with a single \`\`\`json fenced block containing one JSON value conforming to the schema from <output-contract>, and nothing else.`;
  const retryTitle = fill(CONTRACT_RETRY_TITLE, { reason: REASON });

  // Mutation: no lead check → the raw line. Mutation: slice off a wrong length → the reason loses or keeps letters.
  // Mutation: fill the reason from the whole prompt → the second line rides along.
  it("a workflow host's output-contract retry is titled as one, with its reason", () => {
    expect(turnTitle([{ kind: 'user', ts: T0, text: RETRY_TEXT, promptIndex: 1 }])).toBe(retryTitle);
    expect(retryTitle).toBe(`Output contract retry: ${REASON}`);
  });

  // Mutation: `includes` in place of `startsWith` → a prompt that quotes the lead is titled a retry.
  it('a prompt that only quotes the lead keeps its own first line', () => {
    const quoted = `Why: ${RETRY_TEXT}`;
    expect(turnTitle([{ kind: 'user', ts: T0, text: quoted, promptIndex: 0 }])).toBe(quoted.split('\n')[0]);
  });

  // Mutation: fold a new prompt index into the open turn → one turn titled by the task.
  it("a resumed attempt's file: the inherited task turn, then the retry turn", () => {
    const parsed = parseUpdates(
      eventsText('01a21688-13f5-7b32-873d-247e85ca5c3f', [
        userLine(T0, 'Track 4 task\nYou work in the local napi-rs checkout.', 0),
        answerLine(T0 + 180_000, 'Here is the report.'),
        turnLine(T0 + 187_523, 187_523),
        userLine(T0 + 187_977, RETRY_TEXT, 1),
        answerLine(T0 + 300_000, '```json\n{}\n```'),
        turnLine(T0 + 308_282, 120_305),
      ]),
    );
    const groups = groupTurns(parsed.transcript, parsed.turns);
    expect(groups.map((group) => turnTitle(group.entries))).toEqual(['Track 4 task', retryTitle]);
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
