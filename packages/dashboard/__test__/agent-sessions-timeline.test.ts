import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import {
  buildTimeline,
  type ChildEvents,
  type DigestEntry,
  digestUpdates,
  hasTurnWork,
  type LaneSource,
  type LaneSpan,
  MAX_LANE_SPANS,
  placeRetries,
  type RetryLine,
  stack,
  type TimelineChildInfo,
  timelineChildReads,
  type TimelineInput,
  turnWindows,
} from '../src/agent-sessions/timeline.js';
import { parseUpdates } from '../src/agent-sessions/transcript.js';
import type { SessionTimelineResponse, TimelineSpan, TurnMetric } from '../src/agent-sessions/types.js';
import { groupTurns } from '../ui/src/lib/sessions.js';
import { answerLine, eventsText, type EventLine } from './helpers/agent-session-files.js';
import { WOKEN_CHILDREN, WOKEN_T, wokenChildLines, wokenParentText, type WokenOptions } from './helpers/woken-turn.js';

const SESSIONS = fileURLToPath(new URL('./fixtures/grok-sessions/home/sessions', import.meta.url));
const PARENT = `${SESSIONS}/%2FUsers%2Fdev%2Fproj/01a20000-0000-7000-8000-00000000000a/updates.jsonl`;
const CHILD_B = `${SESSIONS}/%2FUsers%2Fdev%2Fproj/01a20000-0000-7000-8000-00000000000b/updates.jsonl`;
const MANUAL = `${SESSIONS}/%2FUsers%2Fdev%2Fother/01a20000-0000-7000-8000-000000000011/updates.jsonl`;
const B = '01a20000-0000-7000-8000-00000000000b';
const C = '01a20000-0000-7000-8000-00000000000c';

const T = 1_800_000_000_000;

const user = (at: number, interjection = false): DigestEntry => ({ kind: 'user', ts: T + at, interjection });
const wake = (at: number): DigestEntry => ({ kind: 'wake', ts: T + at });
const think = (at: number, end: number): DigestEntry => ({ kind: 'thinking', ts: T + at, endTs: T + end });
const answer = (at: number, end: number): DigestEntry => ({ kind: 'assistant', ts: T + at, endTs: T + end });
const plan = (at: number): DigestEntry => ({ kind: 'plan', ts: T + at });
const model = (at: number): DigestEntry => ({ kind: 'model', ts: T + at });
function tool(at: number, end: number | null, name = 'read_file', failed = false): DigestEntry {
  return { kind: 'tool', ts: T + at, endTs: end === null ? null : T + end, name, failed };
}
function spawn(
  at: number,
  childId: string,
  end: number | null = null,
  extra: Partial<Extract<DigestEntry, { kind: 'subagent' }>> = {},
): DigestEntry {
  return {
    kind: 'subagent',
    ts: T + at,
    endTs: end === null ? null : T + end,
    childId,
    type: 'explore',
    description: `Work ${childId}`,
    status: end === null ? null : 'completed',
    failed: false,
    durationMs: null,
    noSpawn: false,
    ...extra,
  };
}
function finishOnly(at: number, childId: string, durationMs: number | null): DigestEntry {
  return spawn(at, childId, at, { noSpawn: true, durationMs, type: '', description: '' });
}
function turnEnd(at: number, elapsedMs: number | null = null, tokens = 0, stopReason = 'end_turn'): DigestEntry {
  return { kind: 'turn-end', ts: T + at, stopReason, elapsedMs, tokens, model: null };
}
const retry = (at: number, attempt = 1, max = 3): RetryLine => ({ ts: T + at, attempt, max });

function source(entries: DigestEntry[], retries: RetryLine[] = [], lastTs?: number): LaneSource {
  let last = 0;
  for (const e of entries) last = Math.max(last, e.ts, 'endTs' in e && e.endTs !== null ? e.endTs : 0);
  return { entries, retries, lastTs: lastTs ?? last };
}

interface Options {
  turn?: number;
  live?: boolean;
  retries?: RetryLine[];
  lastTs?: number;
  children?: Record<string, Partial<TimelineChildInfo>>;
  files?: Record<string, ChildEvents | DigestEntry[]>;
}

function input(entries: DigestEntry[], options: Options = {}): TimelineInput {
  const children = new Map<string, TimelineChildInfo>();
  for (const e of entries) {
    if (e.kind !== 'subagent' || children.has(e.childId)) continue;
    children.set(e.childId, {
      status: 'completed',
      type: '',
      description: '',
      model: null,
      exists: true,
      subagents: 0,
      ...options.children?.[e.childId],
    });
  }
  return {
    turn: options.turn ?? 1,
    parent: source(entries, options.retries ?? [], options.lastTs),
    live: options.live ?? false,
    asOfMs: T + 1_000_000,
    main: { name: 'Main agent', type: null, model: 'fallback-model', sessionId: 'parent' },
    children,
  };
}

/** Builds like the reader does (only the files the builder asks for) and checks that no lane overlaps itself. */
function build(entries: DigestEntry[], options: Options = {}): SessionTimelineResponse {
  const inp = input(entries, options);
  const reads = timelineChildReads(inp);
  if (reads === 'no-turn') throw new Error('no-turn');
  const events = new Map<string, ChildEvents>();
  for (const id of reads) {
    const file = options.files?.[id];
    events.set(
      id,
      file === undefined
        ? { ok: false, noSteps: 'empty', fileBytes: null }
        : Array.isArray(file)
          ? { ok: true, source: source(file) }
          : file,
    );
  }
  const result = buildTimeline(inp, events);
  if (result === 'no-turn') throw new Error('no-turn');
  for (const lane of result.lanes) {
    const spans = result.spans.filter((s) => s.lane === lane.id);
    for (let i = 1; i < spans.length; i += 1) {
      const prev = spans[i - 1]!;
      expect(prev.endMs, `${lane.id} span ${prev.id} runs on`).not.toBeNull();
      expect(spans[i]!.startMs, `${lane.id} spans overlap at ${spans[i]!.id}`).toBeGreaterThanOrEqual(prev.endMs!);
    }
    for (const span of spans) if (span.endMs !== null) expect(span.endMs).toBeGreaterThanOrEqual(span.startMs);
  }
  return result;
}

function label(span: TimelineSpan): string {
  switch (span.kind) {
    case 'generate':
      return span.phase;
    case 'tool':
      return `tool ${span.tools.join(',')}`;
    case 'wait':
      return span.reason === 'model' ? `wait ${span.until}` : `wait subagent ${String(span.calls)}`;
    case 'run':
      return `run ${span.noSteps}`;
    case 'idle':
      return 'idle';
  }
}

/** `label start-end` per span of a lane; `null` ends print as `…`. */
function lane(result: SessionTimelineResponse, id = 'main'): string[] {
  return result.spans
    .filter((s) => s.lane === id)
    .map((s) => `${label(s)} ${String(s.startMs)}-${s.endMs === null ? '…' : String(s.endMs)}`);
}

function spansOf(result: SessionTimelineResponse, id = 'main'): TimelineSpan[] {
  return result.spans.filter((s) => s.lane === id);
}

function fixtureParent(): LaneSource {
  return digestUpdates(parseUpdates(readFileSync(PARENT, 'utf8')));
}

function fixtureInput(turn: number): TimelineInput {
  return {
    turn,
    parent: fixtureParent(),
    live: false,
    asOfMs: 0,
    main: { name: 'Main agent', type: null, model: null, sessionId: 'p' },
    children: new Map([
      [B, { status: 'completed', type: 'general-purpose', description: '', model: 'm', exists: true, subagents: 0 }],
      [C, { status: 'running', type: 'codex', description: '', model: 'm', exists: true, subagents: 0 }],
    ]),
  };
}

function fixtureTimeline(turn: number): SessionTimelineResponse {
  const inp = fixtureInput(turn);
  const events = new Map<string, ChildEvents>([
    [B, { ok: true, source: digestUpdates(parseUpdates(readFileSync(CHILD_B, 'utf8'))) }],
    [C, { ok: false, noSteps: 'empty', fileBytes: null }],
  ]);
  const result = buildTimeline(inp, events);
  if (result === 'no-turn') throw new Error('no-turn');
  return result;
}

describe('digestUpdates', () => {
  it('names a tool by its _meta name only, never by its title', () => {
    const line = (at: number, update: object): string =>
      JSON.stringify({
        method: 'session/update',
        params: { sessionId: 's', update, _meta: { agentTimestampMs: T + at } },
      });
    const title = 'cat /Users/dev/secret-notes.txt';
    const text = [
      line(0, { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: 'hello' } }),
      line(100, { sessionUpdate: 'tool_call', toolCallId: 't1', title, status: 'pending' }),
      line(200, { sessionUpdate: 'tool_call_update', toolCallId: 't1', status: 'completed' }),
      line(300, { sessionUpdate: 'tool_call', toolCallId: 't2', title: 'grep TODO src', status: 'pending' }),
      line(400, {
        sessionUpdate: 'tool_call_update',
        toolCallId: 't2',
        status: 'completed',
        _meta: { 'x.ai/tool': { name: 'grep' } },
      }),
    ]
      .map((l) => `${l}\n`)
      .join('');
    const digest = digestUpdates(parseUpdates(text));
    expect(digest.entries.filter((e) => e.kind === 'tool').map((e) => e.name)).toEqual(['tool', 'grep']);
    const result = buildTimeline(
      {
        turn: 1,
        parent: digest,
        live: false,
        asOfMs: T + 1_000,
        main: { name: 'Main agent', type: null, model: null, sessionId: 's' },
        children: new Map(),
      },
      new Map(),
    );
    const strings: string[] = [];
    const walk = (value: unknown): void => {
      if (typeof value === 'string') strings.push(value);
      else if (Array.isArray(value)) value.forEach(walk);
      else if (typeof value === 'object' && value !== null) Object.values(value).forEach(walk);
    };
    walk(result);
    expect(strings).toContain('tool');
    expect(strings.some((value) => value.includes('secret-notes') || value.includes('TODO'))).toBe(false);
  });
});

describe('turnWindows', () => {
  const partition = (transcript: Parameters<typeof groupTurns>[0], turns: TurnMetric[]) => ({
    groups: groupTurns(transcript, turns).map((g) => ({ number: g.number, entries: g.entries, end: g.end })),
    windows: turnWindows(transcript),
  });

  it('partitions exactly as groupTurns on the fixture transcripts', () => {
    for (const path of [PARENT, CHILD_B, MANUAL]) {
      const parsed = parseUpdates(readFileSync(path, 'utf8'));
      const { groups, windows } = partition(parsed.transcript, parsed.turns);
      expect(windows).toEqual(groups);
    }
  });

  it('partitions exactly as groupTurns with leading entries, an open turn and extra metrics', () => {
    const transcript = [
      { kind: 'model' as const, ts: 1, modelId: 'm' },
      { kind: 'plan' as const, ts: 2, entries: [] },
      { kind: 'user' as const, ts: 3, text: 'a', promptIndex: 0 },
      { kind: 'turn-end' as const, ts: 4, stopReason: 'end_turn' },
      { kind: 'model' as const, ts: 5, modelId: 'n' },
      { kind: 'user' as const, ts: 6, text: 'b', promptIndex: 1 },
      { kind: 'assistant' as const, ts: 7, text: 'c', endTs: 7 },
    ];
    const metric = (ts: number): TurnMetric => ({
      promptId: '',
      ts,
      inputTokens: 1,
      outputTokens: 1,
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
    });
    const { groups, windows } = partition(transcript, [metric(4), metric(8), metric(9)]);
    expect(windows).toEqual(groups);
    expect(windows.map((w) => w.number)).toEqual([0, 1, 2]);
  });

  it('trailing entries without work are not a turn', () => {
    expect(hasTurnWork([model(1), plan(2), spawn(3, 'x', 4)])).toBe(false);
    expect(hasTurnWork([model(1), tool(2, 3)])).toBe(true);
    const done = [user(0), answer(1, 2), turnEnd(3)];
    expect(timelineChildReads(input([...done, model(10)], { turn: 2 }))).toBe('no-turn');
    expect(timelineChildReads(input([...done, plan(10), spawn(11, 'x', 12)], { turn: 2 }))).toBe('no-turn');
    expect(timelineChildReads(input([...done, model(10), user(11)], { turn: 2 }))).toEqual([]);
    expect(timelineChildReads(input(done, { turn: 3 }))).toBe('no-turn');
  });
});

describe('buildTimeline: the window', () => {
  it('starts at the prompt when it precedes the measured start (fixture)', () => {
    const one = fixtureTimeline(1);
    expect(one.range).toMatchObject({ startMs: 1791643279022, endMs: 1791643300080, startExact: false, leadMs: 14976 });
    const two = fixtureTimeline(2);
    expect(two.range).toMatchObject({ startMs: 1791643400090, startExact: false, leadMs: 1004 });
    expect(two.stopReason).toBe('error');
  });

  it('starts at end − elapsed when nothing precedes it', () => {
    const result = build([user(100), answer(200, 300), turnEnd(1000, 950)]);
    expect(result.range).toMatchObject({ startMs: T + 50, endMs: T + 1000, startExact: true, leadMs: 0 });
    expect(lane(result)).toEqual(['wait text 50-150', 'answer 150-250']);
    expect(spansOf(result)[0]!.inferredStart).toBe(false);
  });

  it('a model entry before the prompt does not move the start; no elapsedMs → the prompt', () => {
    const result = build([model(10), user(20), answer(30, 40), turnEnd(50)]);
    expect(result.range).toMatchObject({ startMs: T + 20, endMs: T + 50, startExact: false, leadMs: 0 });
    expect(lane(result)).toEqual(['wait text 0-10', 'answer 10-20']);
  });

  it('no timed entry → no range; untimed entries are counted', () => {
    const untimed: DigestEntry[] = [
      { kind: 'user', ts: 0, interjection: false },
      { kind: 'assistant', ts: 0, endTs: 0 },
      { kind: 'turn-end', ts: 0, stopReason: 'end_turn', elapsedMs: null, tokens: 0, model: null },
    ];
    const result = build(untimed);
    expect(result.range).toBeNull();
    expect(result.lanes).toEqual([]);
    expect(result.spans).toEqual([]);
    expect(result.notes.untimedEvents).toBe(2);
  });

  it('a turn end before the first entry with no elapsedMs → no range', () => {
    expect(build([user(100), answer(110, 120), turnEnd(50)]).range).toBeNull();
  });

  it('first entry and turn end in the same ms → a range of length 0', () => {
    const result = build([user(100), answer(100, 100), turnEnd(100)]);
    expect(result.range).toMatchObject({ startMs: T + 100, endMs: T + 100 });
    expect(lane(result)).toEqual(['answer 0-0']);
  });

  it('counts untimed entries of a timed turn and leaves them out', () => {
    const result = build([
      user(0),
      { kind: 'tool', ts: 0, endTs: null, name: 'x', failed: false },
      answer(5, 6),
      turnEnd(10),
    ]);
    expect(result.notes.untimedEvents).toBe(1);
    expect(lane(result)).toEqual(['wait text 0-5', 'answer 5-6']);
  });
});

describe('buildTimeline: the cursor', () => {
  it('the turn’s own prompt makes the first wait exact in a finished, an open and an unmeasured turn', () => {
    expect(spansOf(fixtureTimeline(1))[0]).toMatchObject({ kind: 'wait', until: 'text', inferredStart: false });
    const open = build([user(0), think(10, 20)], { turn: 1 });
    expect(spansOf(open)[0]).toMatchObject({
      kind: 'wait',
      until: 'text',
      startMs: 0,
      endMs: 10,
      inferredStart: false,
    });
    const unmeasured = build([user(0), answer(10, 20), turnEnd(30)]);
    expect(spansOf(unmeasured)[0]).toMatchObject({ kind: 'wait', until: 'text', inferredStart: false });
  });

  it('without a prompt the first wait starts from an inferred cursor', () => {
    const result = build([answer(100, 110), turnEnd(200, 150)]);
    expect(result.range).toMatchObject({ startMs: T + 50, startExact: true });
    expect(spansOf(result)[0]).toMatchObject({ until: 'text', inferredStart: false });
    const noMeasure = build([model(5), tool(10, 20), answer(30, 40), turnEnd(50)]);
    expect(lane(noMeasure)).toEqual(['tool read_file 0-10', 'wait text 10-20', 'answer 20-30']);
  });

  it('tool end → next tool start is a wait for the model; reasoning → tool call is a gap', () => {
    const result = build([user(0), think(10, 20), tool(30, 40), tool(50, 60, 'grep'), turnEnd(70)]);
    expect(lane(result)).toEqual([
      'wait text 0-10',
      'reasoning 10-20',
      'tool read_file 30-40',
      'wait tool 40-50',
      'tool grep 50-60',
    ]);
  });

  it('reasoning → plan → answer gives two generate spans and no wait', () => {
    const result = build([user(0), think(10, 20), plan(25), answer(30, 40), turnEnd(50)]);
    expect(lane(result)).toEqual(['wait text 0-10', 'reasoning 10-20', 'answer 30-40']);
  });

  it('an interjection during a tool does not move the cursor; one after it does', () => {
    const during = build([user(0), tool(10, 50), user(20, true), think(60, 70), turnEnd(80)]);
    expect(lane(during)).toEqual(['wait tool 0-10', 'tool read_file 10-50', 'wait text 50-60', 'reasoning 60-70']);
    const later = build([user(0), answer(5, 10), user(30, true), answer(40, 50), turnEnd(60)]);
    expect(lane(later)).toEqual(['wait text 0-5', 'answer 5-10', 'wait text 30-40', 'answer 40-50']);
  });

  it('a trailing request in a live open turn waits until now', () => {
    const result = build([user(0), tool(10, 20)], { live: true });
    expect(lane(result)).toEqual(['wait tool 0-10', 'tool read_file 10-20', 'wait open 20-…']);
    expect(spansOf(result)[2]).toMatchObject({ status: 'running', endMs: null });
    const streaming = build([user(0), think(10, 20)], { live: true });
    expect(lane(streaming)).toEqual(['wait text 0-10', 'reasoning 10-…']);
    expect(spansOf(streaming)[1]!.status).toBe('running');
  });
});

describe('buildTimeline: tools', () => {
  it('end from endTs; no end → the next non-tool event, inferred', () => {
    const result = build([user(0), tool(10, null), plan(30), answer(40, 50), turnEnd(60)]);
    expect(lane(result)).toEqual(['wait tool 0-10', 'tool read_file 10-30', 'wait text 30-40', 'answer 40-50']);
    expect(spansOf(result)[1]).toMatchObject({ inferredEnd: true, status: 'unfinished' });
    expect(spansOf(result)[2]!.inferredStart).toBe(true);
    // The block's end is inferred, so the wait after it starts from an inferred cursor.
    const next = build([user(0), tool(10, null), user(30, true), tool(40, 50), turnEnd(60)]);
    expect(lane(next)).toEqual(['wait tool 0-10', 'tool read_file 10-30', 'wait tool 30-40', 'tool read_file 40-50']);
  });

  it('missing ends: finished → unfinished, open live → running, open not live → the last line', () => {
    const finished = build([user(0), tool(10, null), turnEnd(40)]);
    expect(spansOf(finished)[1]).toMatchObject({ endMs: 40, status: 'unfinished', inferredEnd: true });
    const live = build([user(0), tool(10, null)], { live: true });
    expect(spansOf(live)[1]).toMatchObject({ endMs: null, status: 'running', inferredEnd: false });
    const stopped = build([user(0), tool(10, null)], { lastTs: T + 25 });
    expect(spansOf(stopped)[1]).toMatchObject({ endMs: 25, status: 'unfinished', inferredEnd: true });
    expect(stopped.range).toMatchObject({ endMs: null, lastEventMs: T + 25 });
  });

  it('overlapping tools merge into one span', () => {
    const result = build([
      user(0),
      tool(10, 30, 'read_file'),
      tool(15, 20, 'grep', true),
      tool(20, 40, 'read_file'),
      tool(35, 36, 'edit'),
      tool(36, 38, 'run'),
      turnEnd(50),
    ]);
    expect(spansOf(result)[1]).toMatchObject({
      kind: 'tool',
      startMs: 10,
      endMs: 40,
      calls: 5,
      failedCalls: 1,
      tools: ['read_file', 'grep', 'edit'],
      status: 'failed',
      inferredEnd: false,
    });
  });

  it('a merged tool span has an inferred end only when its latest end is inferred', () => {
    const early = build([user(0), tool(10, 50), tool(15, null), plan(30), answer(55, 58), turnEnd(60)]);
    expect(spansOf(early).find((s) => s.kind === 'tool')).toMatchObject({
      startMs: 10,
      endMs: 50,
      inferredEnd: false,
      status: 'ok',
    });
    const late = build([user(0), tool(10, 20), tool(15, null), answer(40, 45), turnEnd(60)]);
    expect(spansOf(late).find((s) => s.kind === 'tool')).toMatchObject({
      endMs: 40,
      inferredEnd: true,
      status: 'unfinished',
    });
  });

  it('status order: running, failed, unfinished, ok', () => {
    const running = build([user(0), tool(10, 20, 'a', true), tool(15, null, 'b')], { live: true });
    expect(spansOf(running)[1]).toMatchObject({ status: 'running', endMs: null, failedCalls: 1 });
    const failed = build([user(0), tool(10, 20, 'a', true), tool(15, null, 'b'), turnEnd(40)]);
    expect(spansOf(failed)[1]).toMatchObject({ status: 'failed', inferredEnd: true });
  });

  it('a spawn tool is a wait for subagents; parallel spawn tools merge', () => {
    const one = build([user(0), tool(10, 50, 'task'), turnEnd(60)]);
    expect(spansOf(one)[1]).toMatchObject({ kind: 'wait', reason: 'subagent', calls: 1, tools: ['task'] });
    const three = build([
      user(0),
      tool(10, 50, 'task'),
      tool(11, 40, 'Task'),
      tool(12, 55, 'spawn_subagent'),
      turnEnd(60),
    ]);
    expect(lane(three)).toEqual(['wait tool 0-10', 'wait subagent 3 10-55']);
    expect(spansOf(three)[1]).toMatchObject({ tools: ['task', 'Task', 'spawn_subagent'] });
  });
});

describe('buildTimeline: spawns without a spawn tool', () => {
  it('spawn → return with nothing of the lane between is a wait, then a wait for the model', () => {
    const result = build([user(0), spawn(10, 'a', 50), answer(60, 70), turnEnd(80)]);
    expect(lane(result)).toEqual(['wait tool 0-10', 'wait subagent 0 10-50', 'wait text 50-60', 'answer 60-70']);
  });

  it('the lane works before the return → no wait', () => {
    const result = build([user(0), spawn(10, 'a', 50), answer(20, 30), turnEnd(80)]);
    expect(lane(result)).toEqual(['wait text 0-20', 'answer 20-30']);
  });

  it('neither a return nor a next event → a wait to the turn end', () => {
    const result = build([user(0), spawn(10, 'a'), turnEnd(80)]);
    expect(lane(result)).toEqual(['wait tool 0-10', 'wait subagent 0 10-80']);
    expect(spansOf(result)[1]).toMatchObject({ status: 'ok', inferredEnd: false });
    const stopped = build([user(0), spawn(10, 'a')], { lastTs: T + 30 });
    expect(spansOf(stopped)[1]).toMatchObject({ endMs: 30, status: 'unfinished', inferredEnd: true });
    const live = build([user(0), spawn(10, 'a')], { live: true, children: { a: { status: 'running' } } });
    expect(spansOf(live)[1]).toMatchObject({ endMs: null, status: 'running' });
  });

  it('two parallel spawns merge, and a spawn line merges with a spawn tool it touches', () => {
    const result = build([user(0), spawn(10, 'a', 40), spawn(12, 'b', 50), answer(60, 70), turnEnd(80)]);
    expect(lane(result)).toEqual(['wait tool 0-10', 'wait subagent 0 10-50', 'wait text 50-60', 'answer 60-70']);
    const mixed = build([user(0), spawn(3, 'a', 5), tool(5, 40, 'task'), answer(60, 70), turnEnd(80)]);
    expect(lane(mixed)).toEqual(['wait tool 0-3', 'wait subagent 1 3-40', 'wait text 40-60', 'answer 60-70']);
  });

  it('a spawn inside a spawn tool call makes no wait of its own', () => {
    const result = build([user(0), tool(10, 20, 'task'), spawn(15, 'a', 90), answer(30, 40), turnEnd(100)]);
    expect(lane(result)).toEqual(['wait tool 0-10', 'wait subagent 1 10-20', 'wait text 20-30', 'answer 30-40']);
  });

  it('a running spawn tool stays running past the spawn line inside it', () => {
    const live = build([user(0), tool(10, null, 'task'), spawn(20, 'a')], {
      live: true,
      children: { a: { status: 'running' } },
    });
    expect(lane(live)).toEqual(['wait tool 0-10', 'wait subagent 1 10-…']);
    expect(spansOf(live)[1]).toMatchObject({ status: 'running' });
  });

  it('a spawn line ends a regular tool that has no recorded end', () => {
    const result = build([user(0), tool(100, null, 'grep'), spawn(200, 'a', 3000), answer(3100, 3200), turnEnd(4000)]);
    expect(lane(result)).toEqual([
      'wait tool 0-100',
      'tool grep 100-200',
      'wait subagent 0 200-3000',
      'wait text 3000-3100',
      'answer 3100-3200',
    ]);
    expect(spansOf(result)[1]).toMatchObject({ status: 'unfinished', inferredEnd: true });
  });

  it('a child that returns after the next turn’s prompt was not waited for', () => {
    const entries = [
      user(0),
      answer(100, 200),
      spawn(300, 'a', 5000),
      turnEnd(1000),
      user(2000),
      answer(2100, 2200),
      turnEnd(6000),
    ];
    const result = build(entries);
    expect(lane(result)).toEqual(['wait text 0-100', 'answer 100-200']);
    expect(result.lanes[1]).toMatchObject({ id: 'a', carriedOut: true });
    expect(result.range!.lastEventMs).toBeLessThanOrEqual(result.range!.endMs!);
    const before = build([...entries.slice(0, 2), spawn(300, 'a', 1500), ...entries.slice(3)]);
    expect(lane(before)).toEqual(['wait text 0-100', 'answer 100-200', 'wait subagent 0 300-1000']);
  });

  it('a carried-in child makes no wait', () => {
    const entries = [
      user(0),
      spawn(10, 'a', 150),
      answer(20, 30),
      turnEnd(40),
      user(100),
      answer(160, 170),
      turnEnd(200),
    ];
    const result = build(entries, { turn: 2 });
    expect(lane(result)).toEqual(['wait text 0-60', 'answer 60-70']);
    expect(result.lanes[1]).toMatchObject({ id: 'a', carriedIn: true });
  });
});

describe('buildTimeline: precedence', () => {
  it('merges first, then cuts a lower span where a higher one stands; pieces keep their facts', () => {
    const result = build([
      user(0),
      tool(10, null, 'task', true),
      tool(20, 30, 'read_file'),
      answer(100, 110),
      turnEnd(120),
    ]);
    expect(lane(result)).toEqual([
      'wait tool 0-10',
      'wait subagent 1 10-20',
      'tool read_file 20-30',
      'wait subagent 1 30-100',
      'answer 100-110',
    ]);
    const [, first, , last] = spansOf(result);
    expect(first).toMatchObject({ calls: 1, failedCalls: 1, status: 'failed', inferredEnd: false });
    expect(last).toMatchObject({ calls: 1, failedCalls: 1, status: 'failed', inferredEnd: true });
  });

  it('drops pieces under 1 ms and keeps a span recorded with zero length', () => {
    const result = build([
      { kind: 'user', ts: T, interjection: false },
      { kind: 'tool', ts: T + 10, endTs: T + 20.5, name: 'task', failed: false },
      tool(10, 20, 'read_file'),
      answer(30, 30),
      turnEnd(40),
    ]);
    expect(lane(result)).toEqual(['wait tool 0-10', 'tool read_file 10-20', 'wait text 20.5-30', 'answer 30-30']);
  });
});

describe('buildTimeline: the clock going backward', () => {
  it('an end before its start reads as the start; out-of-order events are sorted', () => {
    const result = build([user(0), answer(50, 60), think(10, 20), tool(30, 25), turnEnd(70)]);
    expect(lane(result)).toEqual([
      'wait text 0-10',
      'reasoning 10-20',
      'tool read_file 30-30',
      'wait text 30-50',
      'answer 50-60',
    ]);
  });
});

describe('buildTimeline: retries', () => {
  it('a retry lands on the model span that holds it, else on the next one, never on a tool', () => {
    const result = build([user(0), think(10, 20), tool(30, 40), answer(50, 60), turnEnd(70)], {
      retries: [retry(5), retry(35, 2), retry(15, 1, 0)],
    });
    const spans = spansOf(result);
    expect(spans.map((s) => s.retry)).toEqual([
      { attempt: 1, max: 3 },
      { attempt: 1, max: 0 },
      null,
      { attempt: 2, max: 3 },
      null,
    ]);
    expect(spans[3]).toMatchObject({ kind: 'wait', reason: 'model' });
  });

  it('two retries on one span keep the higher attempt; none at or after → unplaced', () => {
    const result = build([user(0), answer(10, 20), tool(30, 40, 'task'), turnEnd(50)], {
      retries: [retry(2, 2), retry(3, 1), retry(35)],
    });
    expect(spansOf(result)[0]!.retry).toEqual({ attempt: 2, max: 3 });
    expect(result.notes.unplacedRetries).toBe(1);
    expect(fixtureTimeline(1).notes.unplacedRetries).toBe(1);
  });
});

describe('buildTimeline: handoffs and runs', () => {
  it('one spawn and one return per run; failed from the run; a finish before its spawn is ignored', () => {
    const result = build(
      [
        user(0),
        spawn(10, 'a', 30),
        spawn(11, 'b', 40, { failed: true, status: 'failed' }),
        spawn(12, 'c'),
        spawn(50, 'd', 45),
        answer(60, 70),
        turnEnd(80),
      ],
      { files: { a: [user(10)], b: [user(11)], c: [user(12)], d: [user(50)] } },
    );
    expect(result.handoffs.map((h) => [h.lane, h.kind, h.atMs, h.failed])).toEqual([
      ['a', 'spawn', 10, false],
      ['a', 'return', 30, false],
      ['b', 'spawn', 11, false],
      ['b', 'return', 40, true],
      ['c', 'spawn', 12, false],
      ['d', 'spawn', 50, false],
    ]);
    expect(result.handoffs.map((h) => h.id)).toContain('b:return:0');
    expect(result.lanes.map((l) => [l.id, l.status])).toEqual([
      ['main', 'completed'],
      ['a', 'completed'],
      ['b', 'failed'],
      ['c', 'unknown'],
      ['d', 'completed'],
    ]);
  });

  it('the return is the finish line’s time, not spawn + durationMs (fixture)', () => {
    const result = fixtureTimeline(1);
    expect(result.handoffs.filter((h) => h.lane === B).map((h) => [h.kind, h.atMs])).toEqual([
      ['spawn', 48],
      ['return', 50],
    ]);
  });

  it('a finish with no spawn → a no-spawn bar that ends at the finish, a return and no spawn handoff', () => {
    const result = build([user(0), finishOnly(50, 'a', 20), answer(60, 70), turnEnd(80)]);
    expect(lane(result, 'a')).toEqual(['run no-spawn 30-50']);
    expect(spansOf(result, 'a')[0]).toMatchObject({ inferredStart: true, inferredEnd: false, status: 'ok' });
    expect(result.lanes[1]).toMatchObject({ noSteps: 'no-spawn', name: 'Subagent' });
    expect(result.handoffs.map((h) => h.kind)).toEqual(['return']);
    expect(timelineChildReads(input([user(0), finishOnly(50, 'a', 20), turnEnd(80)]))).toEqual([]);
    const early = build([user(0), finishOnly(5, 'a', 20), turnEnd(80)]);
    expect(lane(early, 'a')).toEqual(['run no-spawn 0-5']);
    const zero = build([user(0), finishOnly(5, 'a', null), turnEnd(80)]);
    expect(lane(zero, 'a')).toEqual(['run no-spawn 5-5']);
  });

  it('a child spawned twice in one window: one lane, two runs, four handoffs, each run with its own events', () => {
    const child = [
      user(10),
      tool(12, 18, 'one'),
      turnEnd(20),
      user(25),
      tool(27, 28, 'between'),
      user(40),
      tool(42, 48, 'two'),
      turnEnd(50),
    ];
    const result = build(
      [user(0), spawn(10, 'a', 20), answer(21, 22), spawn(40, 'a', 50), answer(51, 52), turnEnd(60)],
      {
        files: { a: child },
      },
    );
    expect(result.lanes.map((l) => l.id)).toEqual(['main', 'a']);
    expect(result.handoffs.map((h) => h.id)).toEqual(['a:spawn:0', 'a:return:0', 'a:spawn:1', 'a:return:1']);
    expect(lane(result, 'a')).toEqual(['wait tool 10-12', 'tool one 12-18', 'wait tool 40-42', 'tool two 42-48']);
  });

  it('a child spawned in two windows shows its own run in each', () => {
    const child = [user(10), tool(12, 18, 'one'), turnEnd(20), user(110), tool(112, 118, 'two'), turnEnd(120)];
    const entries = [
      user(0),
      spawn(10, 'a', 20),
      answer(21, 22),
      turnEnd(30),
      user(100),
      spawn(110, 'a', 120),
      turnEnd(130),
    ];
    expect(lane(build(entries, { files: { a: child } }), 'a')).toEqual(['wait tool 10-12', 'tool one 12-18']);
    expect(lane(build(entries, { turn: 2, files: { a: child } }), 'a')).toEqual(['wait tool 10-12', 'tool two 12-18']);
  });

  it('no child event inside the run → an out-of-range bar (fixture child b)', () => {
    const result = fixtureTimeline(1);
    expect(result.lanes.find((l) => l.id === B)).toMatchObject({ noSteps: 'out-of-range', status: 'completed' });
    expect(lane(result, B)).toEqual(['run out-of-range 48-50']);
  });

  it('a step that crosses the return is cut there and the lane is clamped; no event moves', () => {
    const child = [user(10), tool(15, 40, 'slow')];
    const result = build([user(0), spawn(10, 'a', 30), answer(31, 32), turnEnd(50)], { files: { a: child } });
    expect(lane(result, 'a')).toEqual(['wait tool 10-15', 'tool slow 15-30']);
    expect(result.lanes[1]!.clamped).toBe(true);
    expect(spansOf(result, 'a')[1]).toMatchObject({ cutStart: false, cutEnd: false });
    const inside = build([user(0), spawn(10, 'a', 30), answer(31, 32), turnEnd(50)], {
      files: { a: [user(10), tool(15, 20)] },
    });
    expect(inside.lanes[1]!.clamped).toBe(false);
  });

  it('a child that outlives the turn: clipped and carried out, then carried in with its return', () => {
    const child = [user(10), tool(15, 140, 'long'), turnEnd(150, null, 500)];
    const entries = [
      user(0),
      spawn(10, 'a', 150),
      answer(20, 30),
      turnEnd(40),
      user(100),
      answer(160, 170),
      turnEnd(200),
    ];
    const one = build(entries, { files: { a: child } });
    expect(one.lanes[1]).toMatchObject({ carriedOut: true, carriedIn: false, status: 'completed' });
    expect(lane(one, 'a')).toEqual(['wait tool 10-15', 'tool long 15-40']);
    expect(spansOf(one, 'a').map((s) => [s.cutStart, s.cutEnd])).toEqual([
      [false, false],
      [false, true],
    ]);
    expect(spansOf(one).some((s) => s.cutStart || s.cutEnd)).toBe(false);
    expect(one.handoffs.map((h) => h.kind)).toEqual(['spawn']);
    const two = build(entries, { turn: 2, files: { a: child } });
    expect(two.lanes[1]).toMatchObject({ carriedOut: false, carriedIn: true });
    expect(lane(two, 'a')).toEqual(['tool long 0-40']);
    expect(spansOf(two, 'a')[0]).toMatchObject({ inferredStart: false, cutStart: true, cutEnd: false });
    expect(two.handoffs.map((h) => [h.kind, h.atMs])).toEqual([['return', 50]]);
    expect(two.lanes[1]!.tokenReports).toEqual([{ atMs: 50, tokens: 500 }]);
  });

  it('a child with no finish line is carried only into the open window of a live session while running', () => {
    const entries = [user(0), spawn(10, 'a'), answer(20, 30), turnEnd(40), user(100), tool(110, null)];
    const files = { a: [user(10), tool(15, null, 'long')] };
    const one = build(entries, { files, live: true, children: { a: { status: 'running' } } });
    expect(one.lanes[1]).toMatchObject({ status: 'unknown', carriedOut: false });
    expect(spansOf(one, 'a').at(-1)).toMatchObject({ endMs: 40, status: 'unfinished', inferredEnd: true });
    const live = build(entries, { turn: 2, files, live: true, children: { a: { status: 'running' } } });
    expect(live.lanes[1]).toMatchObject({ id: 'a', status: 'running', carriedIn: true });
    expect(spansOf(live, 'a').at(-1)).toMatchObject({ endMs: null, status: 'running' });
    expect(build(entries, { turn: 2, files, children: { a: { status: 'running' } } }).lanes).toHaveLength(1);
    expect(build(entries, { turn: 2, files, live: true, children: { a: { status: 'completed' } } }).lanes).toHaveLength(
      1,
    );
  });

  it('a running child whose own turn ended has nothing under way after its turn end', () => {
    const entries = [user(0), spawn(100, 'a'), answer(110, 120)];
    const children = { a: { status: 'running' } };
    const answered = build(entries, {
      live: true,
      children,
      files: { a: [user(101), answer(105, 140), turnEnd(150)] },
    });
    expect(answered.lanes[1]!.status).toBe('running');
    expect(lane(answered, 'a')).toEqual(['wait text 101-105', 'answer 105-140']);
    expect(spansOf(answered, 'a')[1]).toMatchObject({ status: 'ok', endMs: 140 });
    const tooled = build(entries, { live: true, children, files: { a: [user(101), tool(105, 130), turnEnd(150)] } });
    expect(lane(tooled, 'a')).toEqual(['wait tool 101-105', 'tool read_file 105-130']);
    const working = build(entries, { live: true, children, files: { a: [user(101), tool(105, 130)] } });
    expect(lane(working, 'a')).toEqual(['wait tool 101-105', 'tool read_file 105-130', 'wait open 130-…']);
  });

  it('a child without files: missing folder, refused file, unreadable file', () => {
    const entries = [user(0), spawn(10, 'a', 30), spawn(11, 'b', 30), spawn(12, 'c', 30), answer(40, 50), turnEnd(60)];
    const result = build(entries, {
      children: { a: { exists: false } },
      files: {
        b: { ok: false, noSteps: 'too-large', fileBytes: 9_000_000 },
        c: { ok: false, noSteps: 'unreadable', fileBytes: null },
      },
    });
    expect(timelineChildReads(input(entries, { children: { a: { exists: false } } }))).toEqual(['b', 'c']);
    expect(result.lanes.slice(1).map((l) => [l.id, l.noSteps, l.fileBytes, l.sessionId])).toEqual([
      ['a', 'missing', null, null],
      ['b', 'too-large', 9_000_000, 'b'],
      ['c', 'unreadable', null, 'c'],
    ]);
    expect(lane(result, 'b')).toEqual(['run too-large 11-30']);
  });
});

describe('buildTimeline: lanes', () => {
  const many = (n: number, at = 10): DigestEntry[] =>
    Array.from({ length: n }, (_, i) => spawn(at + i, `c${String(i)}`, 90));

  it('orders lanes by spawn, folds past eight, reads no folded file', () => {
    const entries = [user(0), ...many(10).reverse(), answer(95, 96), turnEnd(100)];
    const inp = input(entries);
    const reads = timelineChildReads(inp);
    expect(reads).toEqual(['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']);
    const result = build(entries);
    expect(result.lanes.map((l) => l.id)).toEqual(['main', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'folded']);
    expect(result.lanes.at(-1)).toMatchObject({
      role: 'folded',
      name: '2 more subagents',
      subagents: 2,
      noSteps: 'folded',
    });
    expect(lane(result, 'folded')).toEqual(['run folded 18-90']);
    expect(result.handoffs.some((h) => h.lane === 'folded' || h.lane === 'c8' || h.lane === 'c9')).toBe(false);
    expect(result.notes.foldedSubagents).toBe(2);
  });

  it('slot is the ordinal over the whole transcript mod 8; a clash takes the next free slot', () => {
    const entries = [
      user(0),
      ...many(9),
      answer(95, 96),
      turnEnd(100),
      user(200),
      spawn(210, 'c8', 290),
      spawn(211, 'c0', 290),
      turnEnd(300),
    ];
    const two = build(entries, { turn: 2 });
    expect(two.lanes.map((l) => [l.id, l.slot])).toEqual([
      ['main', null],
      ['c8', 0],
      ['c0', 1],
    ]);
  });

  it('names a lane by description, else type, else Subagent; uses the meta for model and nesting', () => {
    const result = build(
      [
        user(0),
        spawn(10, 'a', 20),
        spawn(11, 'b', 20, { description: '' }),
        spawn(12, 'c', 20, { description: '', type: '' }),
        turnEnd(30),
      ],
      { children: { a: { model: 'qwen', subagents: 2 } } },
    );
    expect(result.lanes.slice(1).map((l) => [l.name, l.type, l.model, l.subagents])).toEqual([
      ['Work a', 'explore', 'qwen', 2],
      ['explore', 'explore', null, 0],
      ['Subagent', null, null, 0],
    ]);
    expect(result.lanes[0]).toMatchObject({ name: 'Main agent', model: 'fallback-model', slot: null, subagents: 3 });
  });
});

describe('buildTimeline: tokens', () => {
  it('main report at the turn end; child reports at its own turn ends inside its runs', () => {
    const child = [user(5), turnEnd(8, null, 99), user(10), answer(12, 14), turnEnd(15, null, 300)];
    const result = build([user(0), spawn(10, 'a', 20), answer(30, 40), turnEnd(50, null, 1234)], {
      files: { a: child },
    });
    expect(result.lanes[0]!.tokenReports).toEqual([{ atMs: 50, tokens: 1234 }]);
    expect(result.lanes[1]!.tokenReports).toEqual([{ atMs: 15, tokens: 300 }]);
    expect(build([user(0), spawn(10, 'a', 20), turnEnd(50)]).lanes[1]!.tokenReports).toEqual([]);
    expect(build([user(0), answer(5, 6)]).lanes[0]!.tokenReports).toEqual([]);
  });
});

describe('buildTimeline: the cap', () => {
  it('keeps the longest spans of a lane and reports the rest', () => {
    const entries: DigestEntry[] = [user(0)];
    for (let i = 0; i < 1100; i += 1) entries.push(tool(i * 10 + 2, i * 10 + 2 + (i % 7) + 1, `t${String(i)}`));
    entries.push(turnEnd(20_000));
    const result = build(entries);
    const spans = spansOf(result);
    expect(spans).toHaveLength(MAX_LANE_SPANS);
    expect(result.notes.droppedSpans).toBe(2200 - MAX_LANE_SPANS);
    expect(spans.map((s) => s.id)).toEqual(spans.map((_, i) => `main:${String(i)}`));
    const shortest = Math.min(...spans.map((s) => s.endMs! - s.startMs));
    expect(shortest).toBeGreaterThanOrEqual(1);
  });
});

describe('buildTimeline: fixture turn 1, whole', () => {
  it('lanes, main-lane spans, handoffs and the unplaced retry', () => {
    const result = fixtureTimeline(1);
    expect(result.lanes.map((l) => [l.id, l.noSteps, l.status])).toEqual([
      ['main', null, 'completed'],
      [B, 'out-of-range', 'completed'],
      [C, 'empty', 'unknown'],
    ]);
    expect(lane(result)).toEqual([
      'wait text 0-8',
      'reasoning 8-9',
      'answer 18-19',
      'tool todo_write 28-30',
      'wait tool 30-38',
      'tool read_file 38-39',
      'wait tool 39-40',
      'tool tool 40-40',
      'wait tool 40-48',
      'wait subagent 0 48-21058',
    ]);
    expect(spansOf(result)[7]).toMatchObject({ status: 'failed', failedCalls: 1 });
    expect(lane(result, C)).toEqual(['run empty 49-21058']);
    expect(spansOf(result, C)[0]).toMatchObject({ status: 'unfinished', inferredEnd: true });
    expect(result.handoffs.map((h) => [h.lane, h.kind, h.atMs])).toEqual([
      [B, 'spawn', 48],
      [B, 'return', 50],
      [C, 'spawn', 49],
    ]);
    expect(result.notes).toEqual({ untimedEvents: 0, foldedSubagents: 0, droppedSpans: 0, unplacedRetries: 1 });
    expect(result.lanes[0]!.tokenReports).toEqual([{ atMs: 21058, tokens: 38827 }]);
    expect(timelineChildReads(fixtureInput(1))).toEqual([B, C]);
  });
});

describe('buildTimeline: impossible times', () => {
  const SID = '01a20000-0000-7000-8000-00000000000a';
  const line = (ts: string, method: string, update: object): string =>
    `{"timestamp":${ts},"method":"${method}","params":{"sessionId":"${SID}","update":${JSON.stringify(update)},"_meta":{}}}`;
  const prompt = (ts: string, i: number): string =>
    line(ts, 'session/update', {
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: 'hi' },
      _meta: { promptIndex: i },
    });
  const thought = (ts: string): string =>
    line(ts, 'session/update', { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: 'x' } });
  const done = (ts: string): string =>
    line(ts, '_x.ai/session/update', { sessionUpdate: 'turn_completed', stop_reason: 'end_turn', elapsed_ms: 500 });

  function fromLines(lines: string[], turn: number, live: boolean): SessionTimelineResponse {
    const parent = digestUpdates(parseUpdates(`${lines.join('\n')}\n`));
    const result = buildTimeline(
      {
        turn,
        parent,
        live,
        asOfMs: T,
        main: { name: 'Main agent', type: null, model: null, sessionId: SID },
        children: new Map(),
      },
      new Map(),
    );
    if (result === 'no-turn') throw new Error('no-turn');
    return structuredClone(result);
  }

  function expectFinite(result: SessionTimelineResponse): void {
    const numbers: number[] = [];
    if (result.range !== null) {
      numbers.push(result.range.startMs, result.range.lastEventMs, result.range.leadMs);
      if (result.range.endMs !== null) numbers.push(result.range.endMs);
    }
    for (const span of result.spans) {
      numbers.push(span.startMs);
      if (span.endMs !== null) numbers.push(span.endMs);
    }
    for (const handoff of result.handoffs) numbers.push(handoff.atMs);
    for (const lane of result.lanes) for (const report of lane.tokenReports) numbers.push(report.atMs);
    for (const n of numbers) expect(Number.isFinite(n), String(n)).toBe(true);
  }

  it('a time that overflows to Infinity leaves the turn without times, recorded or live', () => {
    const shapes: Array<[string[], number]> = [
      [[prompt('1791643279', 0), thought('1791643280'), done('1791643281'), prompt('1e308', 1), thought('1e308')], 2],
      [[prompt('1e308', 0), thought('1e308'), done('1e308')], 1],
    ];
    for (const [lines, turn] of shapes) {
      for (const live of [false, true]) {
        const result = fromLines(lines, turn, live);
        expect(result.range).toBeNull();
        expectFinite(result);
      }
    }
  });

  it('an end that overflows reads as an end before the start: no range', () => {
    const result = fromLines([prompt('1791643279', 0), thought('1791643280'), done('1e308')], 1, false);
    expect(result.range).toBeNull();
    expectFinite(result);
  });
});

/** The quadratic subtraction `stack` replaced: the reference its output must equal. */
const TIER_REF: Record<LaneSpan['kind'], number> = {
  generate: 0,
  tool: 1,
  'wait-subagent': 2,
  'wait-model': 3,
  run: 4,
  idle: 5,
};
function byStartRef(a: LaneSpan, b: LaneSpan): number {
  return a.start - b.start || a.end - b.end || TIER_REF[a.kind] - TIER_REF[b.kind];
}
function stackRef(spans: readonly LaneSpan[]): LaneSpan[] {
  const order = [...spans].sort((a, b) => TIER_REF[a.kind] - TIER_REF[b.kind] || byStartRef(a, b));
  const taken: Array<[number, number]> = [];
  const out: LaneSpan[] = [];
  for (const span of order) {
    if (span.zero) {
      if (taken.some(([a, b]) => a < span.start && span.start < b)) continue;
      out.push(span);
      taken.push([span.start, span.start]);
      continue;
    }
    let pieces: Array<[number, number]> = [[span.start, span.end]];
    for (const [a, b] of taken) {
      const next: Array<[number, number]> = [];
      for (const [p0, p1] of pieces) {
        const overlaps = a === b ? p0 < a && a < p1 : a < p1 && b > p0;
        if (!overlaps) {
          next.push([p0, p1]);
          continue;
        }
        if (a > p0) next.push([p0, a]);
        if (b < p1) next.push([b, p1]);
      }
      pieces = next;
    }
    for (const [p0, p1] of pieces) {
      if (p1 - p0 < 1) continue;
      const piece: LaneSpan = {
        ...span,
        start: p0,
        end: p1,
        inferredStart: span.inferredStart && p0 === span.start,
        inferredEnd: span.inferredEnd && p1 === span.end,
        cutStart: span.cutStart && p0 === span.start,
        cutEnd: span.cutEnd && p1 === span.end,
        tools: [...span.tools],
      };
      if (piece.status === 'running' && piece.end !== Infinity) piece.status = 'ok';
      out.push(piece);
      taken.push([p0, p1]);
    }
  }
  return out.sort(byStartRef);
}
function placeRetriesRef(spans: readonly LaneSpan[], retries: readonly RetryLine[]): number {
  const model = spans.filter((s) => s.kind === 'generate' || s.kind === 'wait-model').sort(byStartRef);
  let unplaced = 0;
  for (const r of [...retries].sort((a, b) => a.ts - b.ts)) {
    const target = model.find((s) => s.start <= r.ts && r.ts <= s.end) ?? model.find((s) => s.start >= r.ts);
    if (target === undefined) {
      unplaced += 1;
      continue;
    }
    if (target.retry === null || r.attempt > target.retry.attempt) target.retry = { attempt: r.attempt, max: r.max };
  }
  return unplaced;
}

function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const KINDS: ReadonlyArray<LaneSpan['kind']> = ['generate', 'tool', 'wait-subagent', 'wait-model', 'run'];
function randomLane(random: () => number): LaneSpan[] {
  const count = 1 + Math.floor(random() * 40);
  const out: LaneSpan[] = [];
  for (let i = 0; i < count; i += 1) {
    const start = Math.floor(random() * 60) + (random() < 0.2 ? 0.5 : 0);
    const shape = random();
    const end =
      shape < 0.15
        ? start
        : shape < 0.25
          ? Infinity
          : shape < 0.35
            ? start + 0.4
            : start + 1 + Math.floor(random() * 20);
    out.push({
      kind: KINDS[Math.floor(random() * KINDS.length)]!,
      start,
      end,
      inferredStart: random() < 0.5,
      inferredEnd: random() < 0.5,
      cutStart: random() < 0.3,
      cutEnd: random() < 0.3,
      status: end === Infinity ? 'running' : random() < 0.2 ? 'failed' : 'ok',
      zero: start === end,
      phase: random() < 0.5 ? 'reasoning' : 'answer',
      until: 'tool',
      tools: [`t${i}`],
      calls: 1,
      failedCalls: 0,
      noSteps: 'missing',
      retry: null,
    });
  }
  return out;
}

describe('buildTimeline: work per lane', () => {
  it('stacking equals subtracting every higher span in turn, on random lanes', () => {
    const random = seeded(7);
    for (let i = 0; i < 3000; i += 1) {
      const spans = randomLane(random);
      expect(stack(structuredClone(spans)), `lane ${i}`).toEqual(stackRef(structuredClone(spans)));
    }
  });

  it('retries land where a scan over every model span puts them, on random lanes', () => {
    const random = seeded(11);
    for (let i = 0; i < 2000; i += 1) {
      const stacked = stackRef(randomLane(random));
      const retries: RetryLine[] = Array.from({ length: Math.floor(random() * 12) }, () => ({
        ts: Math.floor(random() * 90) - 5 + (random() < 0.3 ? 0.5 : 0),
        attempt: 1 + Math.floor(random() * 3),
        max: 3,
      }));
      const mine = structuredClone(stacked);
      const ref = structuredClone(stacked);
      expect(placeRetries(mine, retries), `lane ${i}`).toBe(placeRetriesRef(ref, retries));
      expect(mine, `lane ${i}`).toEqual(ref);
    }
  });

  it('a child lane of 16,000 tool calls stays well under a second and keeps the longest spans', () => {
    const n = 16_000;
    const child: DigestEntry[] = [];
    for (let i = 0; i < n; i += 1) child.push(tool(20 + i * 600, 120 + i * 600));
    const runEnd = 10 + n * 600 + 1000;
    const started = performance.now();
    const lastTool = 20 + (n - 1) * 600;
    const retries = Array.from({ length: 20_000 }, (_, i) => retry(lastTool + 1 + i * 0.004));
    const result = build([user(0), spawn(10, B, runEnd), turnEnd(runEnd + 10)], {
      files: { [B]: { ok: true, source: source(child, retries) } },
    });
    const elapsed = performance.now() - started;
    expect(elapsed).toBeLessThan(1000);
    expect(spansOf(result, B)).toHaveLength(MAX_LANE_SPANS);
    expect(result.notes.droppedSpans).toBe(2 * n - MAX_LANE_SPANS);
    expect(result.notes.unplacedRetries).toBe(20_000);
  }, 60_000);

  it('a lane of 30,000 spawn lines inside 30,000 spawn tools checks each line once', () => {
    const n = 30_000;
    const entries: DigestEntry[] = [user(0)];
    for (let i = 0; i < n; i += 1) {
      entries.push(tool(20 + i * 600, 120 + i * 600, 'task'));
      entries.push(spawn(50 + i * 600, B, 60 + i * 600));
    }
    entries.push(turnEnd(10 + n * 600 + 1000));
    const started = performance.now();
    const result = build(entries);
    expect(performance.now() - started).toBeLessThan(1000);
    expect(result.notes.droppedSpans).toBe(2 * n - MAX_LANE_SPANS + (n - MAX_LANE_SPANS));
  }, 60_000);
});

describe('buildTimeline: a woken turn', () => {
  const digest = (text: string): DigestEntry[] => digestUpdates(parseUpdates(text)).entries;
  function woken(turn: number, options: WokenOptions = {}): SessionTimelineResponse {
    const files: Record<string, DigestEntry[]> = {};
    for (const child of WOKEN_CHILDREN) files[child.id] = digest(eventsText(child.id, wokenChildLines(child)));
    return build(digest(wokenParentText(options)), { turn, files });
  }
  const at = (offset: number): number => WOKEN_T + offset;

  it('woken turn 2: the axis starts at turn 1 end, the turn itself at its measured start', () => {
    const r = woken(2);
    expect(r.range).toMatchObject({
      startMs: at(649_557),
      turnStartMs: at(2_680_145),
      endMs: at(2_932_482),
      startExact: true,
      leadMs: 0,
    });
    expect(r.range!.endMs! - r.range!.turnStartMs).toBe(252_337);
    const main = spansOf(r);
    expect(main[0]).toMatchObject({ kind: 'idle', startMs: 0, endMs: 2_030_588, status: 'ok' });
    expect(main[0]).toMatchObject({ inferredStart: false, inferredEnd: false, cutStart: false, cutEnd: false });
    expect(main[1]).toMatchObject({
      kind: 'wait',
      reason: 'model',
      until: 'text',
      startMs: 2_030_651,
      endMs: 2_050_025,
      inferredStart: false,
    });
    expect(main.filter((s) => s.kind === 'idle')).toHaveLength(1);
    expect(main.some((s) => s.kind === 'wait' && s.reason === 'subagent')).toBe(false);
    expect(r.lanes).toHaveLength(8);
    expect(r.lanes.filter((l) => l.carriedIn)).toHaveLength(4);
    expect(r.handoffs.filter((h) => h.kind === 'spawn')).toHaveLength(3);
    expect(
      r.handoffs
        .filter((h) => h.kind === 'return')
        .map((h) => h.atMs)
        .sort((a, b) => a - b),
    ).toEqual([220_685, 458_900, 569_983, 632_444, 1_429_723, 1_528_251, 2_030_561]);
    expect(r.lanes[0]!.tokenReports).toEqual([{ atMs: 2_282_925, tokens: 1_502_141 }]);
    const childReports = r.lanes.slice(1).flatMap((l) => l.tokenReports);
    expect(childReports).toHaveLength(7);
    expect(childReports.every((report) => report.atMs < 2_030_588)).toBe(true);
    expect(childReports.reduce((sum, report) => sum + report.tokens, 0)).toBe(1_697_987);
  });

  it('turn 1 before a woken turn is unchanged: no stretch, research lanes run on past its end', () => {
    const r = woken(1);
    expect(r.range!.startMs).toBe(r.range!.turnStartMs);
    expect(spansOf(r).some((s) => s.kind === 'idle')).toBe(false);
    expect(r.lanes).toHaveLength(6);
    expect(r.lanes.filter((l) => l.carriedOut)).toHaveLength(4);
  });

  it('the server turn windows and the UI turn groups put every line of a woken session in the same turn', () => {
    const parsed = parseUpdates(wokenParentText());
    const windows = turnWindows(digestUpdates(parsed).entries);
    const groups = groupTurns(parsed.transcript, parsed.turns);
    expect(windows.map((w) => [w.number, w.entries.map((e) => e.kind)])).toEqual(
      groups.map((g) => [g.number, g.entries.map((e) => e.kind)]),
    );
    expect(groups.map((g) => g.number)).toEqual([1, 2]);
    expect(groups[1]!.entries.filter((e) => e.kind === 'wake')).toHaveLength(1);
  });

  it('a typed prompt in place of the hidden one gets no stretch', () => {
    const r = woken(2, { typed: true });
    expect(r.range!.startMs).toBe(r.range!.turnStartMs);
    expect(r.lanes).toHaveLength(1);
    expect(spansOf(r).some((s) => s.kind === 'idle')).toBe(false);
  });

  it('no stretch without a run active in the gap, or without a timed previous turn end', () => {
    const quiet = build([user(0), turnEnd(100, 100), wake(200), think(210, 220), turnEnd(300, 100)], { turn: 2 });
    expect(quiet.range).toMatchObject({ startMs: T + 200, turnStartMs: T + 200 });
    expect(quiet.lanes).toHaveLength(1);
    expect(spansOf(quiet).some((s) => s.kind === 'idle')).toBe(false);
    const untimedEnd: DigestEntry = {
      kind: 'turn-end',
      ts: 0,
      stopReason: 'end_turn',
      elapsedMs: null,
      tokens: 0,
      model: null,
    };
    const noEnd = build([user(0), spawn(50, 'x', 250), untimedEnd, wake(300), think(310, 320), turnEnd(400, 100)], {
      turn: 2,
    });
    expect(noEnd.range!.startMs).toBe(noEnd.range!.turnStartMs);
  });

  it('a run with no return spawned in the gap starts a stretch; one spawned before the gap does not', () => {
    const inGap = build([user(0), turnEnd(100, 100), spawn(150, 'x'), wake(200), think(210, 220), turnEnd(300, 100)], {
      turn: 2,
    });
    expect(inGap.range).toMatchObject({ startMs: T + 150, turnStartMs: T + 200 });
    expect(lane(inGap)[0]).toBe('idle 0-50');
    const before = build([user(0), spawn(50, 'y'), turnEnd(100, 100), wake(200), think(210, 220), turnEnd(300, 100)], {
      turn: 2,
    });
    expect(before.range!.startMs).toBe(before.range!.turnStartMs);
  });

  it('a spawn line no longer sets the start of a turn that has work of its own', () => {
    const r = build(
      [user(0), turnEnd(1_000, 1_000), spawn(2_000, 'z', 2_500), answer(102_000, 102_500), turnEnd(103_000)],
      {
        turn: 2,
      },
    );
    expect(r.range).toMatchObject({ startMs: T + 102_000, turnStartMs: T + 102_000 });
  });

  it('a wake after a typed prompt in one window is a prompt for the cursor, not a stretch', () => {
    const r = build([user(0), think(10, 20), wake(500), answer(600, 700), turnEnd(800, 800)]);
    expect(lane(r)).toEqual(['wait text 0-10', 'reasoning 10-20', 'wait text 500-600', 'answer 600-700']);
  });

  it('a hidden first prompt of the file, then a typed interjection: turn 1 is woken and starts at the wake', () => {
    const lines: EventLine[] = [
      {
        at: T,
        update: {
          sessionUpdate: 'user_message_chunk',
          content: { type: 'text', text: 'A background task stopped.' },
          _meta: { promptIndex: 0, hideFromScrollback: true },
        },
      },
      answerLine(T + 100, 'Reading the result'),
      {
        at: T + 145,
        update: {
          sessionUpdate: 'user_message_chunk',
          content: { type: 'text', text: 'stop and check the tests' },
          _meta: { promptIndex: 1, interjection: true },
        },
      },
      answerLine(T + 200, 'Checking'),
      { at: T + 300, xai: true, update: { sessionUpdate: 'turn_completed', stop_reason: 'end_turn', elapsed_ms: 300 } },
    ];
    const parsed = parseUpdates(eventsText('parent', lines));
    const entries = digestUpdates(parsed).entries;
    const kinds = ['wake', 'assistant', 'user', 'assistant'];
    expect(turnWindows(entries).map((w) => [w.number, w.entries.map((e) => e.kind)])).toEqual([[1, kinds]]);
    expect(groupTurns(parsed.transcript, parsed.turns).map((g) => [g.number, g.entries.map((e) => e.kind)])).toEqual([
      [1, kinds],
    ]);
    const r = build(entries);
    expect(r.range).toMatchObject({ startMs: T, turnStartMs: T });
    expect(lane(r)[0]).toBe('wait text 0-100');
  });

  it('runs that meet the turn take the lanes before runs of the stretch alone; lanes keep spawn order', () => {
    const gap = Array.from({ length: 8 }, (_, i) => spawn(200 + i, `bg${String(i)}`, 1_000 + i));
    const entries = [
      user(0),
      answer(10, 20),
      turnEnd(100, 100),
      ...gap,
      wake(2_000),
      answer(2_100, 2_150),
      spawn(2_200, 'own', 2_500),
      answer(2_600, 2_700),
      turnEnd(3_000, 1_000),
    ];
    const r = build(entries, { turn: 2, files: { own: [think(2_210, 2_300), answer(2_300, 2_400), turnEnd(2_450)] } });
    expect(r.range).toMatchObject({ startMs: T + 200, turnStartMs: T + 2_000 });
    expect(r.lanes.map((l) => l.id)).toEqual([
      'main',
      'bg0',
      'bg1',
      'bg2',
      'bg3',
      'bg4',
      'bg5',
      'bg6',
      'own',
      'folded',
    ]);
    expect(r.notes.foldedSubagents).toBe(1);
    expect(lane(r, 'own')).toEqual(['wait text 2000-2010', 'reasoning 2010-2100', 'answer 2100-2200']);
    expect(r.handoffs.filter((h) => h.lane === 'own').map((h) => [h.kind, h.atMs])).toEqual([
      ['spawn', 2_000],
      ['return', 2_300],
    ]);
    expect(lane(r)).toContain('wait subagent 0 2000-2300');
  });

  it('an open window that starts with an interjection starts there', () => {
    const only = build([user(0), turnEnd(100, 100), user(1_000, true)], { turn: 2, live: true, lastTs: T + 1_000 });
    expect(only.range).toMatchObject({ startMs: T + 1_000, turnStartMs: T + 1_000 });
    const then = build([user(0), turnEnd(100, 100), user(1_000, true), answer(4_000, 4_500)], {
      turn: 2,
      live: true,
      lastTs: T + 4_500,
    });
    expect(then.range).toMatchObject({ startMs: T + 1_000, turnStartMs: T + 1_000 });
    expect(lane(then)).toEqual(['wait text 0-3000', 'answer 3000-…']);
  });

  it('a trailing group with only a wake entry is the open turn', () => {
    expect(hasTurnWork([wake(1)])).toBe(true);
    const r = build([user(0), turnEnd(100, 100), wake(200)], { turn: 2, live: true, lastTs: T + 200 });
    expect(r.open).toBe(true);
  });
});
