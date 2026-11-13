import { describe, expect, it } from 'vite-plus/test';

import {
  activeLanes,
  axisTicks,
  cardPlace,
  drawnSpans,
  liveAxisEnd,
  liveNow,
  moveFocus,
  replayRate,
  scrubStep,
  selectionPath,
  tabStop,
  tokensAt,
  type AgentHandoff,
  type AgentSpan,
} from '../ui/src/components/arc-x/agent-timeline/agent-timeline-lib.js';
import { expectedLanes, timelineAxis, timelineModel } from '../ui/src/lib/agent-timeline.js';
import { boxInView } from '../ui/src/lib/float-box.js';
import { formatDuration } from '../ui/src/lib/format.js';
import type { SessionTimelineResponse, TimelineLane, TimelineSpan } from '../ui/src/lib/types.js';
import { wokenTimeline } from './helpers/woken-turn.js';

const MIN = 60_000;

function span(
  id: string,
  laneId: string,
  startMs: number,
  endMs: number | null,
  extra: Partial<AgentSpan> = {},
): AgentSpan {
  return { id, laneId, kind: 'tool', startMs, endMs, status: endMs === null ? 'running' : 'ok', label: id, ...extra };
}

describe('replayRate', () => {
  it('is a true multiplier that keeps a replay within 20 s', () => {
    expect(replayRate(15_000)).toBe(1);
    expect(replayRate(20_000)).toBe(1);
    expect(replayRate(72_000)).toBe(5);
    expect(replayRate(12 * MIN)).toBe(60);
    expect(replayRate(10 * 60 * MIN)).toBe(1800);
    expect(replayRate(30 * 60 * MIN)).toBe(5400);
  });
});

describe('scrubStep', () => {
  it('is about a sixtieth of the record on a round value', () => {
    expect(scrubStep(26_400)).toBe(500);
    expect(scrubStep(12 * MIN)).toBe(15_000);
    expect(scrubStep(1)).toBe(100);
  });
});

describe('liveAxisEnd', () => {
  it('climbs the 1-2-5 ladder with a tenth of room', () => {
    expect(liveAxisEnd(0)).toBe(10_000);
    expect(liveAxisEnd(9_000)).toBe(10_000);
    expect(liveAxisEnd(9_001)).toBe(20_000);
    expect(liveAxisEnd(42_000)).toBe(50_000);
    expect(liveAxisEnd(100_000)).toBe(2 * MIN);
    expect(liveAxisEnd(95 * MIN)).toBe(200 * MIN);
    expect(liveAxisEnd(400 * MIN)).toBe(500 * MIN);
  });

  it('returns a finite length for a clock offset that is not finite', () => {
    for (const x of [Number.NaN, Infinity, -Infinity]) expect(Number.isFinite(liveAxisEnd(x)), String(x)).toBe(true);
    expect(liveAxisEnd(-Infinity)).toBe(10_000);
  });

  it('every step has at least three ticks in the unit it prints', () => {
    let x = 0;
    const seen = new Set<number>();
    while (x < 2000 * MIN) {
      const end = liveAxisEnd(x);
      seen.add(end);
      expect(axisTicks(end, timelineAxis(end).unitMs).length, String(end)).toBeGreaterThanOrEqual(3);
      x = end * 0.9 + 1;
    }
    expect(seen.size).toBeGreaterThan(10);
  });
});

describe('axisTicks', () => {
  it('is round in the printed unit and never passes the axis', () => {
    expect(axisTicks(72_000, 1000)).toEqual([0, 20_000, 40_000, 60_000]);
    expect(axisTicks(12 * MIN, MIN)).toEqual([0, 5 * MIN, 10 * MIN]);
    expect(axisTicks(9_000, 1000)).toEqual([0, 5_000]);
    const tiny = axisTicks(1, 1);
    expect(tiny.every((tick) => tick <= 1)).toBe(true);
    expect(tiny[0]).toBe(0);
  });
});

describe('liveNow', () => {
  it('is the elapsed time, or whole seconds under static motion', () => {
    expect(liveNow(3_400, false)).toBe(3_400);
    expect(liveNow(3_400, true)).toBe(3_000);
    expect(liveNow(-5, false)).toBe(0);
  });
});

describe('tabStop', () => {
  const spans = [span('a', 'main', 0, 100), span('b', 'main', 200, 300), span('c', 'kid', 50, 400)];

  it('is the active span when it is shown', () => {
    expect(tabStop('b', spans, 250, false)).toBe('b');
    expect(tabStop('b', spans, 0, true)).toBe('b');
  });

  it('is the last shown span before a pending active span on its lane, else the first shown span', () => {
    expect(tabStop('b', spans, 150, false)).toBe('a');
    expect(tabStop('c', spans, 10, false)).toBe('a');
  });

  it('is the plot itself when no span is shown', () => {
    expect(tabStop('b', spans, 0, false)).toBeNull();
    expect(tabStop(null, [], 0, true)).toBeNull();
  });
});

describe('tokensAt', () => {
  const reports = [
    { laneId: 'kid', atMs: 500, tokens: 100 },
    { laneId: 'other', atMs: 700, tokens: 50 },
    { laneId: 'main', atMs: 1000, tokens: 900 },
  ];

  it('holds nothing before the first report and keeps the main and subagent figures apart', () => {
    const none = { before: null, beforeLanes: 0 };
    expect(tokensAt(reports, 'main', 499)).toEqual({ main: null, sub: null, subLanes: 0, ...none });
    expect(tokensAt(reports, 'main', 500)).toEqual({ main: null, sub: 100, subLanes: 1, ...none });
    expect(tokensAt(reports, 'main', 999)).toEqual({ main: null, sub: 150, subLanes: 2, ...none });
    expect(tokensAt(reports, 'main', 1000)).toEqual({ main: 900, sub: 150, subLanes: 2, ...none });
  });

  it('splits the subagent reports at the turn start; the main figure is never split', () => {
    expect(tokensAt(reports, 'main', 1000, 600)).toEqual({
      main: 900,
      sub: 50,
      subLanes: 1,
      before: 100,
      beforeLanes: 1,
    });
    expect(tokensAt(reports, 'main', 550, 600)).toEqual({
      main: null,
      sub: null,
      subLanes: 0,
      before: 100,
      beforeLanes: 1,
    });
  });
});

describe('activeLanes', () => {
  it('counts lanes with a span under way', () => {
    const spans = [span('a', 'main', 0, 100), span('b', 'kid', 50, null), span('c', 'kid', 0, 10)];
    expect(activeLanes(spans, 60)).toBe(2);
    expect(activeLanes(spans, 200)).toBe(1);
  });

  it('an idle agent is not working', () => {
    const spans = [span('i', 'main', 0, 100, { kind: 'idle' }), span('b', 'kid', 50, null)];
    expect(activeLanes(spans, 60)).toBe(1);
  });
});

describe('selectionPath', () => {
  const spans: AgentSpan[] = [
    span('m1', 'main', 0, 100, { kind: 'generate' }),
    span('w', 'main', 100, 1000, { kind: 'wait', ownsHandoffs: true }),
    span('m2', 'main', 1000, 1200, { kind: 'generate' }),
    span('k1', 'kid', 110, 300),
    span('k2', 'kid', 400, 950),
    span('o1', 'other', 120, 500),
  ];
  const handoffs: AgentHandoff[] = [
    { id: 'kid:spawn:0', laneId: 'kid', kind: 'spawn', atMs: 105 },
    { id: 'kid:return:0', laneId: 'kid', kind: 'return', atMs: 960 },
    { id: 'other:spawn:0', laneId: 'other', kind: 'spawn', atMs: 110 },
    { id: 'other:return:0', laneId: 'other', kind: 'return', atMs: 1100 },
  ];

  it('a subagent span keeps its lane, the main waits over its runs and its handoffs', () => {
    const path = selectionPath('k1', spans, handoffs, 'main')!;
    expect([...path.spans].sort()).toEqual(['k1', 'k2', 'w']);
    expect([...path.handoffs].sort()).toEqual(['kid:return:0', 'kid:spawn:0']);
  });

  it('a main wait on subagents keeps the handoffs inside it and the edge spans of those children', () => {
    const path = selectionPath('w', spans, handoffs, 'main')!;
    expect([...path.spans].sort()).toEqual(['k1', 'k2', 'o1', 'w']);
    expect([...path.handoffs].sort()).toEqual(['kid:return:0', 'kid:spawn:0', 'other:spawn:0']);
  });

  it('any other main span keeps only itself', () => {
    const path = selectionPath('m1', spans, handoffs, 'main')!;
    expect([...path.spans]).toEqual(['m1']);
    expect(path.handoffs.size).toBe(0);
    expect(selectionPath(null, spans, handoffs, 'main')).toBeNull();
  });
});

describe('drawnSpans', () => {
  it('keeps the longest spans per lane and reports the rest', () => {
    const spans = [
      ...Array.from({ length: 300 }, (_, i) => span(`a${String(i)}`, 'a', i, i + 1 + (i % 3))),
      ...Array.from({ length: 300 }, (_, i) => span(`b${String(i)}`, 'b', i, i + 1)),
    ];
    const { drawn, hidden } = drawnSpans(spans, 400);
    expect(drawn).toHaveLength(400);
    expect(hidden).toBe(200);
    const a = drawn.filter((s) => s.laneId === 'a');
    expect(a).toHaveLength(200);
    expect(a.every((s) => s.endMs! - s.startMs >= 2)).toBe(true);
    expect(drawn.map((s) => s.id)).toEqual(spans.filter((s) => drawn.includes(s)).map((s) => s.id));
    expect(drawnSpans(spans.slice(0, 10)).hidden).toBe(0);
  });
});

describe('moveFocus', () => {
  const shown = [
    span('a1', 'a', 0, 10),
    span('a2', 'a', 20, 30),
    span('a3', 'a', 40, 50),
    span('c1', 'c', 22, 25),
    span('c2', 'c', 45, 48),
  ];
  const lanes = ['a', 'b', 'c'];

  it('arrows move in the lane and to the nearest start in the next lane with spans', () => {
    expect(moveFocus('a2', 'ArrowRight', shown, lanes)).toBe('a3');
    expect(moveFocus('a3', 'ArrowRight', shown, lanes)).toBe('a3');
    expect(moveFocus('a1', 'ArrowLeft', shown, lanes)).toBe('a1');
    expect(moveFocus('a2', 'ArrowDown', shown, lanes)).toBe('c1');
    expect(moveFocus('c2', 'ArrowUp', shown, lanes)).toBe('a3');
    expect(moveFocus('a2', 'ArrowUp', shown, lanes)).toBe('a2');
  });

  it('Home and End stay in the lane; with the modifier they reach the whole timeline', () => {
    expect(moveFocus('c2', 'Home', shown, lanes)).toBe('c1');
    expect(moveFocus('a1', 'End', shown, lanes)).toBe('a3');
    expect(moveFocus('a2', 'End', shown, lanes, true)).toBe('c2');
    expect(moveFocus('c2', 'Home', shown, lanes, true)).toBe('a1');
  });
});

describe('cardPlace', () => {
  const card = { width: 264, height: 80 };
  const window = { left: 0, top: 0, right: 1180, bottom: 800 };

  it('sits above its span with a 6 px gap, 12 px left of its start', () => {
    expect(cardPlace({ left: 200, top: 150, bottom: 164 }, card, window)).toEqual({ x: 188, y: 64, below: false });
  });

  it('goes below when the whole card does not fit above', () => {
    expect(cardPlace({ left: 200, top: 90, bottom: 104 }, card, window)).toEqual({ x: 188, y: 110, below: true });
  });

  it('counts the top limit (the page area under the top bar), not the window top', () => {
    const page = { ...window, top: 40 };
    expect(cardPlace({ left: 200, top: 130, bottom: 144 }, card, page).below).toBe(true);
    expect(cardPlace({ left: 200, top: 134, bottom: 148 }, card, page)).toEqual({ x: 188, y: 48, below: false });
  });

  it('goes above near the window bottom', () => {
    expect(cardPlace({ left: 200, top: 760, bottom: 774 }, card, window)).toEqual({ x: 188, y: 674, below: false });
  });

  it('with no room on either side takes the larger side and stays inside the limits', () => {
    const short = { left: 0, top: 0, right: 1180, bottom: 200 };
    const tall = { width: 264, height: 150 };
    const low = cardPlace({ left: 200, top: 120, bottom: 134 }, tall, short);
    expect(low).toEqual({ x: 188, y: 8, below: false });
    const high = cardPlace({ left: 200, top: 60, bottom: 74 }, tall, short);
    expect(high).toEqual({ x: 188, y: 200 - 8 - 150, below: true });
    for (const place of [low, high]) {
      expect(place.y).toBeGreaterThanOrEqual(8);
      expect(place.y + tall.height).toBeLessThanOrEqual(200 - 8);
    }
  });

  it('is clamped 8 px inside the limits on both sides', () => {
    const page = { left: 208, top: 40, right: 900, bottom: 800 };
    expect(cardPlace({ left: 210, top: 150, bottom: 164 }, card, page).x).toBe(216);
    expect(cardPlace({ left: 890, top: 150, bottom: 164 }, card, page).x).toBe(900 - 264 - 8);
  });
});

describe('boxInView', () => {
  const view = { left: 100, top: 40, right: 900, bottom: 800 };

  it('gives the visible part of a span', () => {
    expect(boxInView({ left: 60, top: 300, right: 160, bottom: 314 }, view)).toEqual({
      left: 100,
      top: 300,
      right: 160,
      bottom: 314,
    });
  });

  it('is null for a span scrolled out of view, sideways or up', () => {
    expect(boxInView({ left: 20, top: 300, right: 100, bottom: 314 }, view)).toBeNull();
    expect(boxInView({ left: 200, top: 10, right: 260, bottom: 40 }, view)).toBeNull();
  });
});

const T0 = 1_700_000_000_000;

function lane(overrides: Partial<TimelineLane>): TimelineLane {
  return {
    id: 'main',
    role: 'main',
    name: 'Main agent',
    type: null,
    model: 'Qwen3.5-27B-4bit',
    slot: null,
    sessionId: 's1',
    status: 'completed',
    carriedIn: false,
    carriedOut: false,
    noSteps: null,
    fileBytes: null,
    clamped: false,
    subagents: 0,
    tokenReports: [],
    ...overrides,
  };
}

function base(overrides: Partial<TimelineSpan> & Pick<TimelineSpan, 'kind'>): TimelineSpan {
  return {
    id: 'x',
    lane: 'main',
    startMs: 0,
    endMs: 100,
    status: 'ok',
    inferredStart: false,
    inferredEnd: false,
    cutStart: false,
    cutEnd: false,
    retry: null,
    ...overrides,
  } as TimelineSpan;
}

function response(overrides: Partial<SessionTimelineResponse> = {}): SessionTimelineResponse {
  return {
    turn: 3,
    turns: 3,
    open: false,
    live: false,
    stopReason: 'end_turn',
    asOfMs: T0 + 70_000,
    range: { startMs: T0, turnStartMs: T0, endMs: T0 + 72_000, lastEventMs: T0 + 72_000, startExact: true, leadMs: 0 },
    lanes: [
      lane({ tokenReports: [{ atMs: 72_000, tokens: 38_827 }] }),
      lane({
        id: 'kid',
        role: 'subagent',
        name: 'Map the scheduler queues',
        type: 'explore',
        slot: 2,
        sessionId: 'kid-session',
        subagents: 2,
        tokenReports: [{ atMs: 30_000, tokens: 9_000 }],
      }),
      lane({
        id: 'gone',
        role: 'subagent',
        name: 'Check the docs',
        slot: 3,
        sessionId: null,
        status: 'failed',
        noSteps: 'missing',
      }),
    ],
    spans: [
      base({ id: 'main:0', kind: 'wait', reason: 'model', until: 'text', endMs: 500 }),
      base({ id: 'main:1', kind: 'generate', phase: 'reasoning', startMs: 500, endMs: 900 }),
      base({
        id: 'main:2',
        kind: 'wait',
        reason: 'model',
        until: 'tool',
        startMs: 900,
        endMs: 1000,
        retry: { attempt: 1, max: 3 },
      }),
      base({
        id: 'main:3',
        kind: 'tool',
        tools: ['read_file', 'grep', 'ls'],
        calls: 4,
        failedCalls: 1,
        startMs: 1000,
        endMs: 2000,
        status: 'failed',
      }),
      base({
        id: 'main:4',
        kind: 'wait',
        reason: 'subagent',
        tools: [],
        calls: 0,
        failedCalls: 0,
        startMs: 2000,
        endMs: 60_000,
      }),
      base({
        id: 'main:5',
        kind: 'tool',
        tools: ['run_terminal_command'],
        calls: 1,
        failedCalls: 0,
        startMs: 60_000,
        endMs: 61_000,
        inferredEnd: true,
        status: 'unfinished',
      }),
      base({ id: 'main:6', kind: 'generate', phase: 'answer', startMs: 61_000, endMs: 72_000 }),
      base({
        id: 'kid:0',
        lane: 'kid',
        kind: 'tool',
        tools: ['grep'],
        calls: 1,
        failedCalls: 0,
        startMs: 2100,
        endMs: 3000,
      }),
      base({
        id: 'gone:0',
        lane: 'gone',
        kind: 'run',
        noSteps: 'missing',
        startMs: 2000,
        endMs: 50_000,
        inferredStart: true,
      }),
    ],
    handoffs: [
      { id: 'kid:spawn:0', lane: 'kid', kind: 'spawn', atMs: 2000, failed: false },
      { id: 'gone:return:0', lane: 'gone', kind: 'return', atMs: 50_000, failed: true },
    ],
    notes: { untimedEvents: 0, foldedSubagents: 0, droppedSpans: 0, unplacedRetries: 0 },
    ...overrides,
  };
}

describe('timelineModel', () => {
  it('follows the clock only for a fresh live body, stands at asOfMs when stale, else is a record', () => {
    const open = response({ open: true, live: true, stopReason: null });
    expect(timelineModel(open, { fresh: true }).mode).toEqual({ kind: 'live', originMs: T0 });
    const stale = timelineModel(open, { fresh: false }).mode;
    expect(stale.kind).toBe('stale');
    expect(stale.kind === 'stale' ? stale.atMs : null).toBe(70_000);
    expect(timelineModel(response(), { fresh: true }).mode).toEqual({ kind: 'recorded' });
    expect(timelineModel(response({ open: true, live: false }), { fresh: true }).mode).toEqual({ kind: 'recorded' });
  });

  it('never gives a length under 1 ms', () => {
    const flat = response({
      range: { startMs: T0, turnStartMs: T0, endMs: T0, lastEventMs: T0, startExact: true, leadMs: 0 },
    });
    expect(timelineModel(flat, { fresh: false }).durationMs).toBe(1);
    expect(timelineModel(response(), { fresh: false }).durationMs).toBe(72_000);
    const open = response({
      open: true,
      range: { startMs: T0, turnStartMs: T0, endMs: null, lastEventMs: T0 + 5_000, startExact: false, leadMs: 0 },
    });
    expect(timelineModel(open, { fresh: false }).durationMs).toBe(5_000);
  });

  it('names every span kind and its facts and notes', () => {
    const spans = timelineModel(response(), { fresh: false }).spans;
    const byId = new Map(spans.map((s) => [s.id, s]));
    expect(byId.get('main:0')!.label).toBe('Waiting for the first token');
    expect(byId.get('main:1')!.label).toBe('Reasoning');
    expect(byId.get('main:2')).toMatchObject({ label: 'Waiting for the model', retry: 'Retry 1 of 3' });
    expect(byId.get('main:3')).toMatchObject({ label: '4 tool calls · read_file, grep +1', facts: ['1 failed'] });
    expect(byId.get('main:4')).toMatchObject({
      label: 'Waiting for subagents',
      ownsHandoffs: true,
      note: 'Read from the order of events; the spawn call itself was not recorded.',
    });
    expect(byId.get('main:5')).toMatchObject({
      label: 'Tool call · run_terminal_command',
      open: 'end',
      note: 'End not recorded. Drawn to the next recorded event.',
    });
    expect(byId.get('main:6')!.label).toBe('Writing the answer');
    expect(byId.get('gone:0')).toMatchObject({
      label: 'Ran, no step record',
      open: 'start',
      note: "No step record: this subagent's session folder is gone.",
    });
    expect(byId.get('main:0')!.cut).toBeUndefined();
  });

  it('marks the edges the turn cut and says the step went on past them', () => {
    const cut = (cutStart: boolean, cutEnd: boolean) =>
      timelineModel(
        response({
          spans: [
            base({
              id: 'kid:0',
              lane: 'kid',
              kind: 'tool',
              tools: ['grep'],
              calls: 1,
              failedCalls: 0,
              cutStart,
              cutEnd,
            }),
          ],
        }),
        { fresh: false },
      ).spans[0]!;
    expect(cut(false, true)).toMatchObject({ cut: 'end', note: 'Went on after this turn; drawn to the turn end.' });
    expect(cut(true, false)).toMatchObject({
      cut: 'start',
      note: 'Began before this turn; drawn from the start of the axis.',
    });
    expect(cut(true, true).cut).toBe('both');
    expect(cut(false, false).cut).toBeUndefined();
    expect(cut(false, true).open).toBeUndefined();
  });

  it('colors lanes by slot, links only a subagent with a folder, and words their outcome', () => {
    const lanes = timelineModel(response(), { fresh: false }).lanes;
    expect(lanes.map((l) => l.color)).toEqual(['var(--lane-main)', 'var(--lane-3)', 'var(--lane-4)']);
    expect(lanes.map((l) => l.to)).toEqual([undefined, '/sessions/kid-session', undefined]);
    expect(lanes[1]!.extra).toBe('+2 subagents');
    expect(lanes[2]!.outcome).toEqual({ text: 'Failed', tone: 'danger' });
  });

  it('words the outcome per stop reason', () => {
    const word = (stopReason: string | null, extra: Partial<SessionTimelineResponse> = {}): string =>
      timelineModel(response({ stopReason, ...extra }), { fresh: false }).outcome.word;
    expect(word('end_turn')).toBe('Finished');
    expect(word('error')).toBe('Failed');
    expect(word('cancelled')).toBe('Cancelled');
    expect(word('max_tokens')).toBe('Ended · max_tokens');
    expect(word(null, { open: true })).toBe('Not finished');
  });

  it('writes the footnote sentences that apply', () => {
    expect(timelineModel(response(), { fresh: false }).footnote).toBe(
      'Positions come from recorded event times. Empty stretches have no recorded event. Faded ends were not recorded and are drawn to the nearest recorded event. Replay runs 5× faster than the turn did.',
    );
    const busy = response({
      range: {
        startMs: T0,
        turnStartMs: T0,
        endMs: T0 + 9_000,
        lastEventMs: T0 + 9_000,
        startExact: false,
        leadMs: 1_004,
      },
      spans: [],
      lanes: [lane({ clamped: true })],
      notes: { untimedEvents: 1, foldedSubagents: 2, droppedSpans: 3, unplacedRetries: 2 },
    });
    expect(timelineModel(busy, { fresh: false, formatDuration }).footnote).toBe(
      'Positions come from recorded event times. Empty stretches have no recorded event. The axis starts at the prompt, 1.00 s before the start the agent measured for this turn. Some subagent steps crossed their handoff times and are cut there. 2 model request retries are not shown on a span. 1 event had no time and is not drawn. Only the first 8 subagents have a lane. 3 very short spans are left out.',
    );
  });

  it('gives every lane its token reports', () => {
    expect(timelineModel(response(), { fresh: false }).reports).toEqual([
      { laneId: 'main', atMs: 72_000, tokens: 38_827 },
      { laneId: 'kid', atMs: 30_000, tokens: 9_000 },
    ]);
  });
});

describe('timelineAxis, expectedLanes', () => {
  it('prints one unit per axis', () => {
    expect(timelineAxis(1_500).format(500)).toBe('500 ms');
    expect(timelineAxis(72_000).format(20_000)).toBe('20 s');
    expect(timelineAxis(12 * MIN).format(5 * MIN)).toBe('5 min');
  });

  it('expects the main lane, one lane per distinct child up to 8, and one folded lane past 8', () => {
    const child = (id: string) =>
      ({ kind: 'subagent', ts: 1, childId: id, type: 't', description: 'd', status: 'completed' }) as const;
    expect(expectedLanes([child('a'), child('a'), child('b')])).toBe(3);
    expect(expectedLanes(Array.from({ length: 8 }, (_, i) => child(String(i))))).toBe(9);
    expect(expectedLanes(Array.from({ length: 9 }, (_, i) => child(String(i))))).toBe(10);
    expect(expectedLanes(Array.from({ length: 12 }, (_, i) => child(String(i))))).toBe(10);
  });
});

describe('timelineModel: a woken turn', () => {
  const model = timelineModel(wokenTimeline(2), { fresh: false });

  it('places the turn start on the axis and words the idle stretch', () => {
    expect(model.turnStartMs).toBe(2_030_588);
    expect(model.durationMs).toBe(2_282_925);
    const idle = model.spans.find((s) => s.kind === 'idle')!;
    expect(idle).toMatchObject({
      laneId: 'main',
      startMs: 0,
      endMs: 2_030_588,
      label: 'Idle between turns',
      note: 'No turn was open. Handoffs here are background work, not steps of this agent. The turn starts where this line ends.',
      ownsHandoffs: true,
    });
    expect(model.spans.filter((s) => s.ownsHandoffs === true)).toEqual([idle]);
  });

  it('says why the axis begins before the turn, and drops the lead sentence', () => {
    expect(model.footnote).toContain(
      'A background event started this turn; no prompt was typed. The axis begins 33m 51s before the turn, where background work ran while the main agent was idle.',
    );
    expect(model.footnote).not.toContain('The axis starts at the prompt');
    expect(model.footnote).toContain('Replay runs 120× faster than real time.');
    expect(model.footnote).not.toContain('than the turn did');
  });

  it('a typed turn has its turn start at 0', () => {
    expect(timelineModel(wokenTimeline(1), { fresh: false }).turnStartMs).toBe(0);
  });

  it('selecting the idle span lights the handoffs inside it and the child ends they touch', () => {
    const path = selectionPath(model.spans.find((s) => s.kind === 'idle')!.id, model.spans, model.handoffs, 'main')!;
    expect(path.handoffs.size).toBe(10);
    expect([...path.spans].filter((id) => id.startsWith('main:'))).toEqual(['main:0']);
    const child = model.spans.find((s) => s.laneId !== 'main')!;
    const fromChild = selectionPath(child.id, model.spans, model.handoffs, 'main')!;
    expect(fromChild.spans.has('main:0')).toBe(false);
  });
});
