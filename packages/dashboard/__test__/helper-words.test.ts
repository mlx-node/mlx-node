/**
 * The words of one agent row: the state word and at most three facts in its head, the counts on a second line, a
 * failed row's reason under it. The same rule draws a Task subagent and a workflow agent on every surface.
 */

import { describe, expect, it } from 'vite-plus/test';

import { progressLine } from '../ui/src/components/arc-x/activity-rail/progress-line.js';
import type { AgentEnd, AgentNow, AgentRow } from '../ui/src/lib/agents.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { HelperProgress } from '../ui/src/lib/chat-types.js';
import { formatDuration } from '../ui/src/lib/format.js';
import { CONTEXT_WARN_PCT, nowWords, readerWords, rowWords } from '../ui/src/lib/helper-words.js';
import { pageOpened, readerFact } from '../ui/src/lib/pdf-run.js';

const T0 = 1_795_674_158_000;
const RUN = 'wf_01a2168534c27fd1a477edb6fad7b0a4';
const F = CHAT.agent.facts;

const now = (state: AgentEnd, over: Partial<AgentRow> = {}): AgentNow => ({
  row: {
    childId: 'child-1',
    label: 'track:holes',
    run: RUN,
    phase: 'Recon',
    entry: {
      kind: 'subagent',
      ts: T0,
      childId: 'child-1',
      type: 'general-purpose',
      description: 'track:holes',
      turns: 3,
      toolCalls: 14,
    },
    attempts: ['child-1'],
    at: [7],
    roster: null,
    rosterTs: null,
    continues: null,
    tokens: null,
    durationMs: 0,
    ...over,
  },
  state,
  until: null,
});

const tick = (over: Partial<HelperProgress> = {}): HelperProgress => ({
  childId: 'child-1',
  contextWindow: null,
  now: null,
  turns: 2,
  toolCalls: 12,
  tokens: 18_200,
  contextPct: 41,
  errors: 0,
  durationMs: 134_000,
  at: T0 + 134_000,
  ...over,
});

describe('rowWords', () => {
  // Mutation: put the tick's tool calls in the head → the head has four facts and the second line loses them.
  it('running: what it does now, its time and its context in the head; its turn and tool calls under it', () => {
    const words = rowWords(now('running'), { progress: tick(), now: T0 + 136_000 });
    expect(words).toEqual({
      mark: 'running',
      word: CHAT.agent.now.starting,
      tone: 'success',
      facts: ['2:16', fill(F.contextPct, { pct: 41 })],
      sub: [fill(CHAT.activity.progressTurn, { n: 2 }), fill(CHAT.subagent.toolCalls, { n: 12 })],
      reason: null,
    });
  });

  // Mutation: keep the word Running → a row never says what its agent does.
  it('what it does now: thinking, writing, its newest open call, compacting; Starting before any', () => {
    const word = (doing: HelperProgress['now']) =>
      rowWords(now('running'), { progress: tick({ now: doing }), now: T0 + 136_000 }).word;
    expect([
      word(null),
      word({ kind: 'thinking' }),
      word({ kind: 'writing' }),
      word({ kind: 'tool', title: 'grep -rn "External" crates/' }),
      word({ kind: 'tool', title: '' }),
      word({ kind: 'compacting' }),
    ]).toEqual(['Starting', 'Thinking', 'Writing', 'Running: grep -rn "External" crates/', 'Running', 'Compacting']);
    expect(nowWords({ kind: 'writing' })).toBe(CHAT.agent.now.writing);
  });

  // Mutation: compare with `>` → a context at exactly 90% does not warn.
  it(`a context ${CONTEXT_WARN_PCT}% full warns in place; one under it does not`, () => {
    const at = (pct: number) => rowWords(now('running'), { progress: tick({ contextPct: pct }), now: T0 + 134_000 });
    expect(at(CONTEXT_WARN_PCT).tone).toBe('warning');
    expect(at(CONTEXT_WARN_PCT - 1).tone).toBe('success');
  });

  // Mutation: check the ask after the stop → a stopping agent with an open card reads Asking.
  it('asking: the request and how long it waits; stopping wins over an ask; quiet after 20 s with no tick', () => {
    const asking = rowWords(now('running'), {
      progress: tick(),
      now: T0 + 140_000,
      ask: { title: 'cargo test', since: T0 + 125_000 },
    });
    expect(asking).toMatchObject({ mark: 'asking', tone: 'warning', facts: ['cargo test', '0:15'] });
    const stopping = rowWords(now('running'), {
      progress: tick(),
      now: T0 + 140_000,
      ask: { title: 'cargo test', since: T0 + 125_000 },
      stopping: true,
    });
    expect(stopping).toMatchObject({ mark: 'stopping', word: CHAT.agent.state.stopping });
    const quiet = rowWords(now('running'), { progress: tick(), now: T0 + 154_000 });
    expect(quiet).toMatchObject({ mark: 'quiet', word: fill(CHAT.activity.progressQuiet, { time: '0:20' }) });
  });

  // Mutation: read the entry's turns as the live `turn n` → a done row says "turn 3" for three turns.
  it('done: a workflow agent its step, time and total use; a subagent its time and its context at its end', () => {
    const done = rowWords(now('done', { durationMs: 62_000, tokens: { n: 8_140, kind: 'used' } }), {
      progress: null,
      now: T0,
    });
    expect(done).toMatchObject({
      mark: 'done',
      facts: ['Recon', '1m 02s', fill(F.tokensUsed, { n: '8,140' })],
      sub: [fill(CHAT.subagent.turns, { n: 3 }), fill(CHAT.subagent.toolCalls, { n: 14 })],
    });
    const task = rowWords(
      now('done', { run: null, phase: null, durationMs: 4_000, tokens: { n: 1_200, kind: 'context' } }),
      { progress: null, now: T0 },
    );
    expect(task.facts).toEqual(['4.00 s', fill(F.contextAtEnd, { n: '1,200' })]);
  });

  // Mutation: drop the reason → a failed row says Failed and not why.
  it('failed: its reason on its own line; stopped by you says so; a retried row counts its attempt', () => {
    const failed = rowWords(
      now('failed', {
        durationMs: 41_000,
        attempts: ['child-0', 'child-1'],
        entry: { ...now('done').row.entry!, error: 'It broke.' },
      }),
      { progress: null, now: T0 },
    );
    expect(failed).toMatchObject({ mark: 'failed', tone: 'danger', facts: ['Recon', '41.0 s'], reason: 'It broke.' });
    expect(failed.sub).toContain(fill(F.attempt, { n: 2 }));
    expect(rowWords(now('stopped'), { progress: null, now: T0, stoppedByYou: true }).word).toBe(
      CHAT.agent.state.stoppedByYou,
    );
  });
});

describe("the rail's live line of an agent", () => {
  const tick = (over: Partial<HelperProgress> = {}): HelperProgress => ({
    childId: 'child-1',
    turns: 1,
    toolCalls: 4,
    tokens: 13_000,
    contextPct: 10,
    errors: 0,
    durationMs: 5_000,
    contextWindow: 131_072,
    at: T0,
    now: { kind: 'tool', title: 'rg -n "Ref::new" crates/napi/src' },
    ...over,
  });

  // Mutation: push the now word while quiet too → a quiet agent still reads as busy first.
  it('leads with what it does now, once its own lines said; quiet takes that place', () => {
    const a = CHAT.activity;
    expect(progressLine(tick(), T0).text).toBe(
      [
        fill(CHAT.agent.now.running, { title: 'rg -n "Ref::new" crates/napi/src' }),
        fill(a.progressCalls, { n: 4 }),
        fill(a.progressContext, { pct: 10 }),
        fill(a.progressTokens, { n: '13K' }),
      ].join(' · '),
    );
    expect(progressLine(tick({ now: null }), T0).text.startsWith(fill(a.progressCalls, { n: 4 }))).toBe(true);
    const quiet = progressLine(tick(), T0 + 25_000).text;
    expect(quiet.startsWith(`${fill(a.progressQuiet, { time: '0:25' })} · ${fill(a.progressCalls, { n: 4 })}`)).toBe(
      true,
    );
  });
});

describe('readerWords', () => {
  const P = CHAT.pdf;
  const reader = { pages: [41, 42, 43, 44], again: false };
  const request = (phase: 'queued' | 'prefill' | 'decode', done: number | null = null) => ({
    session: 'child-1',
    modelId: 'm',
    phase,
    startedAt: T0,
    prefillDone: done,
    prefillTotal: done === null ? null : 4_300,
    generated: null,
  });

  it("running: the engine's words for its session, then its time; the page it opened on the second line", () => {
    const live = { progress: tick(), now: T0 + 136_000 };
    const words = readerWords(now('running'), live, {
      reader,
      engine: readerFact(request('prefill', 2_100)),
      opened: 43,
    });
    expect(words.word).toBe(fill(P.readerFacts.readingOf, { done: '2,100', total: '4,300' }));
    expect(words.facts).toEqual(['2:16']);
    expect(words.sub.at(-1)).toBe(fill(P.readerOpened, { n: 43 }));
    expect(readerFact(request('prefill', 0))).toBe(P.readerFacts.reading);
    expect(readerFact(request('queued'))).toBe(P.readerFacts.waiting);
    expect(readerFact(request('decode'))).toBe(P.readerFacts.writing);
    expect(readerWords(now('running'), live, { reader, engine: null, opened: null }).word).toBe(
      rowWords(now('running'), live).word,
    );
  });

  it('a second try says so; a failed reader says what becomes of its pages; an ended one drops its step', () => {
    const live = { progress: null, now: T0 };
    const failed = readerWords(now('failed', { durationMs: 48_000 }), live, { reader, engine: null, opened: 43 });
    expect(failed.reason).toBe(P.readerFailed);
    expect(failed.facts).toEqual([formatDuration(48_000)]);
    expect(failed.sub.some((line) => line === fill(P.readerOpened, { n: 43 }))).toBe(false);
    const twice = readerWords(now('failed', { durationMs: 48_000 }), live, {
      reader: { ...reader, again: true },
      engine: null,
      opened: null,
    });
    expect(twice.reason).toBe(P.readerFailedTwice);
    expect(twice.facts[0]).toBe(P.readerFacts.again);
  });

  it('the page a read names: page-NNN.jpg at the end of its path, else none', () => {
    expect(pageOpened('Read `/x/work-pdf/abcdefgh-40/page-043.jpg`')).toBe(43);
    expect(pageOpened('Read `/x/page-04.jpg`')).toBeNull();
    expect(pageOpened('Read `/x/page-000.jpg`')).toBeNull();
    expect(pageOpened('Read `/x/notes.md`')).toBeNull();
  });
});
