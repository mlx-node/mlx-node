import { describe, expect, it } from 'vite-plus/test';

import { DashboardApiError } from '../ui/src/lib/api.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import {
  answerMarkdown,
  chatTitle,
  continuingTurn,
  conversationMarkdown,
  engineNote,
  engineWait,
  formatElapsed,
  hookHeadline,
  hookName,
  hookNoteRow,
  hookRow,
  hookWaitLine,
  hookWhen,
  isSideEntry,
  liveOutcome,
  fullLimit,
  loadFor,
  loadVariant,
  modelDetail,
  mountedFor,
  openRetry,
  pendingRow,
  pictureRow,
  retryLine,
  retryReading,
  retryRow,
  splitTurns,
  storedOutcome,
  thoughtDuration,
  turnOutcome,
  turnStats,
  turnPrompt,
  turnStatsTitle,
  type EngineRequest,
} from '../ui/src/lib/chat-turns.js';
import { CONTINUE_PROMPT } from '../ui/src/lib/chat-types.js';
import { dayGroupLabel, groupByDay, matchesQuery } from '../ui/src/lib/conversations.js';
import { formatClock } from '../ui/src/lib/format.js';
import { JUMP_FAR, PIN_SLACK, distanceToEnd, jumpVisible, nextPinned } from '../ui/src/lib/stick-to-bottom.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import { BLOCKER_SENTENCES, sendFailure } from '../ui/src/pages/chat/blockers.js';
import {
  compactEndWords,
  compactOff,
  forkOff,
  forkRefusal,
  recapEndToast,
  recapOff,
  rewindEndText,
  rewindOff,
  rewindRefusal,
  type HistoryFacts,
} from '../ui/src/pages/chat/history-actions.js';
import { isRewindPoint, lastRewindPoint } from '../ui/src/pages/chat/use-rewind.js';
import { RUST_RESULTS, answer, directSearch, end, metric, user } from './helpers/chat-thread-fixtures.js';

describe('splitTurns', () => {
  it('a turn starts at each typed prompt, an interjection stays in its turn, ends pair with metrics in order', () => {
    const entries: TranscriptEntry[] = [
      user('one'),
      answer('a'),
      end(),
      user('two'),
      { kind: 'user', ts: 5, text: 'also', promptIndex: null, interjection: true },
      answer('b'),
    ];
    const turns = splitTurns(entries, [metric({ promptId: 'm1' })]);
    expect(turns.map((turn) => turn.start)).toEqual([0, 3]);
    expect(turns[0]!.end?.stopReason).toBe('end_turn');
    expect(turns[0]!.metric?.promptId).toBe('m1');
    expect(turns[1]!.end).toBeNull();
    expect(turns[1]!.entries).toHaveLength(3);
  });

  // Mutation: drop `note` from the aside list → the note starts a turn of its own.
  it("a note on a denied call stays in its turn, and is never the turn's prompt", () => {
    const note: TranscriptEntry = { kind: 'user', ts: 5, text: 'use pnpm', promptIndex: null, note: true };
    const entries: TranscriptEntry[] = [
      user('clean the build'),
      answer('Running.'),
      note,
      answer('Using pnpm.'),
      end(),
    ];
    const turns = splitTurns(entries, []);
    expect(turns.map((turn) => turn.entries.length)).toEqual([4]);
    expect(turnPrompt(turns[0]!)).toBe('clean the build');
    const [wake] = splitTurns([{ kind: 'wake', ts: 1, cause: 'task', text: '' }, note, end()], []);
    expect(turnPrompt(wake!)).toBeNull();
  });
});

describe('a jump to an older turn', () => {
  const entries: TranscriptEntry[] = [];
  for (let n = 0; n < 6; n += 1) entries.push(user(`q${String(n)}`), answer(`a${String(n)}`), end());
  const list = splitTurns(entries, []);

  // Mutation: count from the turn after the target → the jump lands on a turn that is not mounted.
  it('mounts the target turn and every turn after it; a turn already in view changes nothing', () => {
    expect(mountedFor(list, 4, 2)).toBe(5);
    expect(mountedFor(list, 0, 2)).toBe(6);
    expect(mountedFor(list, 16, 2)).toBe(2);
    expect(mountedFor(list, 13, 4)).toBe(4);
    expect(mountedFor([], 3, 2)).toBe(2);
  });
});

describe('outcomes', () => {
  it('live outcomes follow the table, with the mode where it matters', () => {
    expect(liveOutcome('completed', 'chat')).toBeNull();
    expect(liveOutcome('output-limit', 'chat')?.action).toBe('continue');
    expect(liveOutcome('denied', 'work')).toMatchObject({ glyph: 'ban', action: null });
    expect(liveOutcome('turn-timeout', 'chat')?.text).toBe('Stopped after 10 minutes.');
    expect(liveOutcome('refused', 'chat')?.action).toBe('update-agent');
  });

  // Mutation: drop the `model-unavailable` case → the line says the agent could not start and offers an update.
  it('a refused turn whose model is gone says so and offers the model picker, in both modes', () => {
    const line = {
      tone: 'danger',
      glyph: 'alert',
      text: CHAT.outcomes['model-unavailable'],
      action: 'pick-model',
    };
    expect(liveOutcome('refused', 'chat', false, 'model-unavailable')).toEqual(line);
    expect(liveOutcome('refused', 'work', false, 'model-unavailable')).toEqual(line);
    expect(CHAT.outcomes['model-unavailable']).toBe('That model is not available.');
    expect(liveOutcome('refused', 'work', false, 'agent-crashed')).toMatchObject({
      text: CHAT.outcomes['refused-work'],
      action: 'update-agent',
    });
  });

  it('stored ends: end_turn is quiet, an error becomes a danger detail', () => {
    expect(storedOutcome({ kind: 'turn-end', ts: 0, stopReason: 'end_turn' })).toBeNull();
    expect(storedOutcome({ kind: 'turn-end', ts: 0, stopReason: 'cancelled', error: 'boom' })).toMatchObject({
      tone: 'danger',
      detail: 'boom',
      action: 'send-again',
    });
  });

  it('the last turn runs: no outcome; it never ended: did not finish', () => {
    const [turn] = splitTurns([user('q'), answer('a')], []);
    expect(turnOutcome(turn!, { last: true, running: true, live: null, mode: 'chat' })).toBeNull();
    expect(turnOutcome(turn!, { last: true, running: false, live: null, mode: 'chat' })?.text).toBe(
      'This answer did not finish.',
    );
    expect(turnOutcome(turn!, { last: false, running: false, live: null, mode: 'chat' })).toMatchObject({
      text: 'This answer did not finish.',
      action: null,
    });
  });

  it('stats and their title', () => {
    expect(turnStats(metric({ outputTokens: 12_345 }), null)).toEqual(['12.4 s', '12,345 tokens', '41.3 tok/s']);
    expect(turnStats(null, null)).toEqual([]);
    expect(turnStatsTitle(metric(), null)).toContain('qwen3.5-4b');
  });

  it('end lines: who stopped, whether the message was sent, and a stored stop that names no one', () => {
    const [turn] = splitTurns([user('q'), answer('a')], []);
    const live = (outcome: 'cancelled' | 'interrupted', unsent?: true) =>
      turnOutcome(turn!, { last: true, running: false, live: { outcome, unsent }, mode: 'chat' });
    expect(live('cancelled')).toMatchObject({ text: CHAT.outcomes.cancelled, glyph: 'stop', action: 'send-again' });
    expect(live('cancelled', true)).toMatchObject({
      text: CHAT.outcomes['cancelled-unsent'],
      tone: 'neutral',
      glyph: 'stop',
      action: 'send-again',
    });
    expect(live('interrupted')).toMatchObject({
      text: CHAT.outcomes.interrupted,
      tone: 'warning',
      glyph: 'alert',
      action: 'send-again',
    });
    const [stored] = splitTurns([user('q'), answer('a'), end('cancelled')], []);
    const line = turnOutcome(stored!, { last: true, running: false, live: null, mode: 'chat' });
    expect(line).toMatchObject({ text: CHAT.outcomes['stopped-stored'], tone: 'neutral', action: 'send-again' });
    expect(line?.text).not.toBe(CHAT.outcomes.cancelled);
    expect(liveOutcome('turn-timeout', 'chat')?.text).toBe(CHAT.outcomes['turn-timeout']);
  });

  it('pending row: the eight rules in order, first match wins; only events in the last 3 s think', () => {
    const base = {
      creating: false,
      phase: 'running' as 'starting' | 'running' | 'cancelling' | null,
      wait: null as 'model' | null,
      model: 'qwen-a',
      loading: null as { modelId: string; startedAt: number } | null,
      inference: 'ready' as const,
      quiet: false,
      full: null as number | null,
    };
    const label = (text: string, orb = 'hold') => ({ kind: 'label', text, orb });
    const swap = { modelId: 'qwen-b', startedAt: 1 };
    expect(pendingRow({ ...base, creating: true, phase: 'cancelling', wait: 'model' })).toEqual(
      label(CHAT.pendingStarting),
    );
    expect(pendingRow({ ...base, phase: 'cancelling', wait: 'model', loading: swap })).toEqual(
      label(CHAT.pendingStopping),
    );
    expect(pendingRow({ ...base, phase: 'starting', wait: 'model', loading: swap })).toEqual({
      kind: 'load',
      modelId: 'qwen-a',
    });
    expect(pendingRow({ ...base, phase: 'running', loading: swap, quiet: true })).toEqual({
      kind: 'load',
      modelId: 'qwen-b',
    });
    expect(pendingRow({ ...base, phase: 'starting', inference: 'starting', quiet: true })).toEqual(
      label(CHAT.pendingStarting),
    );
    expect(pendingRow({ ...base, inference: 'starting', quiet: true })).toEqual(label(CHAT.pendingEngine));
    expect(pendingRow({ ...base, inference: 'starting', full: 1 })).toEqual(label(CHAT.pendingEngine));
    expect(pendingRow({ ...base, full: 1, quiet: true })).toEqual(
      label(fill(CHAT.pendingQueuedOne, { model: 'qwen-a' })),
    );
    expect(pendingRow({ ...base, full: 4 })).toEqual(label(fill(CHAT.pendingQueuedMany, { model: 'qwen-a', n: '4' })));
    expect(pendingRow({ ...base, phase: 'starting', full: 1 })).toEqual(label(CHAT.pendingStarting));
    expect(pendingRow({ ...base, wait: 'model', full: 1 })).toEqual({ kind: 'load', modelId: 'qwen-a' });
    expect(pendingRow({ ...base, quiet: true })).toEqual(label(CHAT.pendingWaiting));
    expect(pendingRow(base)).toEqual(label(CHAT.pendingThinking, 'think'));
  });

  it('wait states: this conversation queued, reading, generating; the busy model; then Working or Waiting', () => {
    const base = {
      creating: false,
      phase: 'running' as 'starting' | 'running' | 'cancelling' | null,
      wait: null as 'model' | null,
      model: 'qwen-a',
      loading: null as { modelId: string; startedAt: number } | null,
      inference: 'ready' as const,
      quiet: false,
      full: null as number | null,
      work: true,
    };
    const label = (text: string, orb = 'hold') => ({ kind: 'label', text, orb });
    const ours = (phase: 'queued' | 'slot' | 'prepare' | 'prefill' | 'decode' | null, done = 0, total = 0) => ({
      kind: 'ours' as const,
      phase,
      done,
      total,
    });
    expect(pendingRow({ ...base, engine: ours('queued') })).toEqual(label(CHAT.pendingQueued));
    expect(pendingRow({ ...base, engine: ours('slot'), quiet: true })).toEqual(label(CHAT.pendingQueued));
    expect(pendingRow({ ...base, engine: ours('queued'), full: 1 })).toEqual(
      label(fill(CHAT.pendingQueuedOne, { model: 'qwen-a' })),
    );
    expect(pendingRow({ ...base, engine: ours('prefill', 69_300, 121_300), quiet: true })).toEqual({
      kind: 'label',
      text: fill(CHAT.pendingReadingOf, { done: '69.3K', total: '121.3K' }),
      orb: 'think',
      progress: { done: 69_300, total: 121_300 },
    });
    expect(pendingRow({ ...base, engine: ours('prepare') })).toEqual(label(CHAT.pendingReading, 'think'));
    expect(pendingRow({ ...base, engine: ours('prefill', 0, 0) })).toEqual(label(CHAT.pendingReading, 'think'));
    expect(pendingRow({ ...base, engine: ours('decode'), quiet: true })).toEqual(label(CHAT.pendingThinking, 'think'));
    expect(pendingRow({ ...base, engine: ours(null) })).toEqual(label(CHAT.pendingThinking, 'think'));
    expect(pendingRow({ ...base, engine: ours('decode'), phase: 'cancelling' })).toEqual(label(CHAT.pendingStopping));
    expect(pendingRow({ ...base, engine: ours('decode'), creating: true })).toEqual(label(CHAT.pendingStarting));
    const swap = { modelId: 'qwen-b', startedAt: 1 };
    expect(pendingRow({ ...base, engine: ours('decode'), loading: swap })).toEqual(
      label(CHAT.pendingThinking, 'think'),
    );
    const busy = { kind: 'model' as const, reading: { done: 5_000, total: 20_000 } };
    expect(pendingRow({ ...base, engine: busy, quiet: true })).toEqual({
      kind: 'label',
      text: fill(CHAT.pendingBusyReading, { done: '5,000', total: '20K' }),
      orb: 'hold',
      progress: { done: 5_000, total: 20_000 },
    });
    expect(pendingRow({ ...base, engine: { kind: 'model', reading: null } })).toEqual(label(CHAT.pendingBusy));
    expect(pendingRow({ ...base, engine: busy, full: 1 })).toEqual(
      label(fill(CHAT.pendingQueuedOne, { model: 'qwen-a' })),
    );
    expect(pendingRow({ ...base, engine: busy, loading: swap })).toEqual({ kind: 'load', modelId: 'qwen-b' });
    expect(pendingRow({ ...base, engine: busy, phase: 'starting' })).toEqual(label(CHAT.pendingStarting));
    expect(pendingRow({ ...base, engine: null })).toEqual(label(CHAT.pendingWorking, 'think'));
    expect(pendingRow({ ...base, engine: null, quiet: true })).toEqual(label(CHAT.pendingWaiting));
    expect(pendingRow({ ...base, work: false })).toEqual(label(CHAT.pendingThinking, 'think'));
  });

  it("the engine's requests: this conversation's newest by its session, else the model's words, else nothing", () => {
    const request = (
      session: string | null,
      startedAt: number,
      phase: 'queued' | 'slot' | 'prepare' | 'prefill' | 'decode' | null,
      prefill: [number, number] | null = null,
      modelId = 'qwen-a',
    ) => ({
      session,
      modelId,
      phase,
      startedAt,
      prefillDone: prefill?.[0] ?? null,
      prefillTotal: prefill?.[1] ?? null,
      generated: null,
    });
    const at = (requests: ReturnType<typeof request>[], since: number | null = 1_000, hidden = 0) =>
      engineWait({ requests, session: 'acp-ours', model: 'qwen-a', since, hidden });
    const other = request('acp-other', 500, 'prefill', [69_300, 121_300]);
    expect(at([other, request('acp-ours', 900, 'queued')])).toEqual({
      kind: 'ours',
      phase: 'queued',
      done: null,
      total: null,
    });
    expect(at([request('acp-ours', 400, 'decode'), request('acp-ours', 1_200, 'prefill', [10, 40])])).toEqual({
      kind: 'ours',
      phase: 'prefill',
      done: 10,
      total: 40,
    });
    expect(at([request('acp-ours', 1_200, 'decode', null, 'qwen-b')])).toBeNull();
    expect(at([other])).toBeNull();
    expect(at([other], 1_000, 1)).toEqual({ kind: 'model', reading: { done: 69_300, total: 121_300 } });
    expect(at([request(null, 1_000, 'decode')])).toEqual({ kind: 'model', reading: null });
    expect(at([request(null, 999, 'decode')])).toBeNull();
    expect(at([request(null, 1_500, 'queued'), other])).toEqual({
      kind: 'model',
      reading: { done: 69_300, total: 121_300 },
    });
    expect(at([request(null, 1_500, 'decode')], null)).toBeNull();
    expect(at([request(null, 500, 'decode')], 1_000, 1)).toEqual({ kind: 'model', reading: null });
    expect(at([request(null, 1_500, 'decode', null, 'qwen-b')], 1_000, 1)).toBeNull();
    expect(at([], 1_000, 1)).toBeNull();
    expect(at([request(null, 2_000, 'prefill', [50, 10])])).toEqual({
      kind: 'model',
      reading: { done: 10, total: 10 },
    });
    expect(
      engineWait({ requests: [request(null, 2_000, 'decode')], session: null, model: null, since: 0, hidden: 1 }),
    ).toBeNull();
    expect(
      engineWait({ requests: [request(null, 2_000, 'queued')], session: null, model: 'qwen-a', since: 0, hidden: 0 }),
    ).toEqual({ kind: 'model', reading: null });
  });

  it('a model is full only when its running answers fill its limit and a request waits', () => {
    const at = (inFlight: number, maxConcurrent: number, queued: number, lastStart: number | null = 100) => [
      { modelId: 'qwen-a', inFlight, maxConcurrent, queued, lastStart },
    ];
    expect(fullLimit(at(1, 1, 1), 'qwen-a', 200)).toBe(1);
    expect(fullLimit(at(4, 4, 2), 'qwen-a', 200)).toBe(4);
    expect(fullLimit(at(1, 1, 0), 'qwen-a', 200)).toBeNull();
    expect(fullLimit(at(2, 4, 1), 'qwen-a', 200)).toBeNull();
    expect(fullLimit(at(0, 1, 1), 'qwen-a', 200)).toBeNull();
    expect(fullLimit(at(1, 1, 1), 'qwen-b', 200)).toBeNull();
    expect(fullLimit(at(1, 1, 1), null, 200)).toBeNull();
    expect(fullLimit(at(0, 0, 1), 'qwen-a', 200)).toBeNull();
  });

  it('a turn is never told it waits when a running answer started after it: that answer can be its own', () => {
    const at = (lastStart: number | null) => [
      { modelId: 'qwen-a', inFlight: 1, maxConcurrent: 1, queued: 1, lastStart },
    ];
    expect(fullLimit(at(199), 'qwen-a', 200)).toBe(1);
    expect(fullLimit(at(200), 'qwen-a', 200)).toBeNull();
    expect(fullLimit(at(250), 'qwen-a', 200)).toBeNull();
    expect(fullLimit(at(null), 'qwen-a', 200)).toBeNull();
    expect(fullLimit(at(100), 'qwen-a', null)).toBeNull();
  });

  it("a conversation shows its own model's load, else the oldest one", () => {
    const load = (modelId: string, startedAt: number) => ({
      modelId,
      startedAt,
      phase: 'kernels' as const,
      stepsDone: 1,
      steps: 2,
    });
    const loading = [load('a', 1), load('b', 2)];
    expect(loadFor(loading, 'b')?.modelId).toBe('b');
    expect(loadFor(loading, 'a')?.modelId).toBe('a');
    expect(loadFor(loading, 'c')?.modelId).toBe('a');
    expect(loadFor(loading, null)?.modelId).toBe('a');
    expect(loadFor([], 'a')).toBeNull();
  });

  it('load card variant, clock text and the engine note under the end line', () => {
    expect(loadVariant('a', { modelId: 'a' }, [])).toBe('loading');
    expect(loadVariant('a', { modelId: 'b' }, [])).toBe('other-first');
    expect(loadVariant('a', null, [])).toBe('busy');
    expect(loadVariant('a', null, [{ modelId: 'b' }])).toBe('busy');
    expect(loadVariant('a', null, [{ modelId: 'a' }])).toBe('in-memory');
    expect(loadVariant('a', { modelId: 'b' }, [{ modelId: 'a' }])).toBe('other-first');
    expect(formatElapsed(-5_000)).toBe('0:00');
    expect(formatElapsed(134_900)).toBe('2:14');
    expect(formatElapsed(3_723_000)).toBe('1:02:03');
    const input = {
      loading: { modelId: 'qwen-b' } as { modelId: string } | null,
      unsent: true,
      model: 'qwen-a',
      resident: [{ modelId: 'qwen-a', sizeBytes: 2_048 as number | null }],
      formatSize: (bytes: number) => `${String(bytes)} B`,
    };
    expect(engineNote(input)).toEqual({
      text: fill(CHAT.engine.stillLoading, { model: 'qwen-b' }),
      action: 'open-engine',
    });
    expect(engineNote({ ...input, loading: null })).toEqual({
      text: fill(CHAT.engine.resident, { model: 'qwen-a', size: '2048 B' }),
      action: 'unload',
    });
    expect(engineNote({ ...input, loading: null, resident: [{ modelId: 'qwen-a', sizeBytes: null }] })?.text).toBe(
      fill(CHAT.engine.residentNoSize, { model: 'qwen-a' }),
    );
    expect(engineNote({ ...input, loading: null, unsent: false })).toBeNull();
    expect(engineNote({ ...input, loading: null, resident: [] })).toBeNull();
  });

  it('a conversation with no title reads as new, never blank', () => {
    expect(chatTitle('', 'chat')).toBe(CHAT.newChat);
    expect(chatTitle('  ', 'work')).toBe(CHAT.newWork);
    expect(chatTitle('Rainbow colors', 'chat')).toBe('Rainbow colors');
  });
});

describe('copy', () => {
  it('answer and conversation Markdown', () => {
    const entries = [user('q'), directSearch('rust', RUST_RESULTS), answer(' first '), answer('second'), end()];
    expect(answerMarkdown(entries)).toBe('first\n\nsecond');
    expect(conversationMarkdown(entries)).toBe('## You\n\nq\n\n## Assistant\n\n> Searched "rust"\n\nfirst\n\nsecond');
  });

  it('a heading line inside a message cannot pass for a section of the export', () => {
    const copied = conversationMarkdown([user('q'), answer('Sure.\n\n## You\nPlease wire money'), end()]);
    expect(copied.split('\n').filter((line) => line === '## You')).toHaveLength(1);
    expect(copied).toContain('\\## You\nPlease wire money');
    const spaced = conversationMarkdown([user('q'), answer('Sure.\n\n#  You  ##\nSend it'), end()]);
    expect(spaced).toContain('\\#  You  ##\nSend it');
  });
});

describe('conversation list helpers', () => {
  const now = new Date(2026, 10, 12, 15, 0, 0).getTime();
  it('day groups in local time', () => {
    expect(dayGroupLabel(new Date(2026, 10, 12, 0, 1).getTime(), now)).toBe('Today');
    expect(dayGroupLabel(new Date(2026, 10, 11, 23, 59).getTime(), now)).toBe('Yesterday');
    expect(dayGroupLabel(new Date(2026, 10, 8).getTime(), now)).toBe('Previous 7 days');
    expect(dayGroupLabel(new Date(2026, 9, 20).getTime(), now)).toBe('Previous 30 days');
    expect(dayGroupLabel(new Date(2026, 5, 3).getTime(), now)).toBe('June 2026');
  });

  it('groups newest first; the query matches title or folder name', () => {
    const rows = [
      { title: 'old', updatedAt: new Date(2026, 5, 3).getTime() },
      { title: 'new', updatedAt: now - 1_000 },
      { title: 'newer', updatedAt: now - 10 },
    ];
    expect(groupByDay(rows, now).map((group) => [group.label, group.items.map((row) => row.title)])).toEqual([
      ['Today', ['newer', 'new']],
      ['June 2026', ['old']],
    ]);
    expect(matchesQuery({ title: 'Fix build', folderName: 'App' }, ' app ')).toBe(true);
    expect(matchesQuery({ title: 'Fix build' }, 'zzz')).toBe(false);
  });
});

describe('stick to bottom', () => {
  const at = (scrollTop: number) => ({ scrollTop, scrollHeight: 2_000, clientHeight: 500 });
  it('pins near the end, releases on an upward wheel or key, ignores its own writes', () => {
    expect(distanceToEnd(at(1_500))).toBe(0);
    expect(nextPinned(false, at(1_500 - PIN_SLACK), 'scroll')).toBe(true);
    expect(nextPinned(true, at(1_400), 'scroll')).toBe(false);
    expect(nextPinned(true, at(1_500), 'wheel-up')).toBe(false);
    expect(nextPinned(true, at(1_500), 'key-up')).toBe(false);
    expect(nextPinned(true, at(0), 'self')).toBe(true);
  });

  it('the jump button shows when released with new content, or far from the end', () => {
    expect(jumpVisible(true, true, at(0))).toBe(false);
    expect(jumpVisible(false, true, at(1_490))).toBe(true);
    expect(jumpVisible(false, false, at(1_500 - JUMP_FAR))).toBe(false);
    expect(jumpVisible(false, false, at(1_500 - JUMP_FAR - 1))).toBe(true);
  });
});

describe('the length stop and the thought time', () => {
  const lengthEnd: TranscriptEntry = {
    kind: 'turn-end',
    ts: 5,
    stopReason: 'error',
    error: 'response truncated by max_tokens',
    lengthLimit: true,
  };
  const continued: TranscriptEntry = { kind: 'user', ts: 6, text: CONTINUE_PROMPT, promptIndex: 1, continued: true };
  const read: TranscriptEntry = {
    kind: 'tool',
    ts: 7,
    id: 't1',
    name: 'read_file',
    title: 'Read',
    status: 'completed',
  };

  it('a stored length end reads as the length line, with no raw error text', () => {
    expect(storedOutcome(lengthEnd as Extract<TranscriptEntry, { kind: 'turn-end' }>)).toEqual({
      tone: 'warning',
      glyph: 'alert',
      text: CHAT.outcomes['output-limit'],
      action: 'continue',
    });
  });

  it('a continued turn with no tool call that hits the limit again says so; one with a call does not', () => {
    const [, again] = splitTurns([user('fix it'), lengthEnd, continued, { ...lengthEnd, ts: 9 }], []);
    const state = { last: true, running: false, live: null, mode: 'work' as const };
    expect(turnOutcome(again!, state)?.text).toBe(CHAT.outcomes['output-limit-twice']);
    const live = splitTurns([user('fix it'), lengthEnd, continued], []);
    expect(turnOutcome(live[1]!, { ...state, live: { outcome: 'output-limit' } })?.text).toBe(
      CHAT.outcomes['output-limit-twice'],
    );
    const [, called] = splitTurns([user('fix it'), lengthEnd, continued, read, { ...lengthEnd, ts: 9 }], []);
    expect(turnOutcome(called!, state)?.text).toBe(CHAT.outcomes['output-limit']);
  });

  // Mutation: the sentence says the app sent the continue → red.
  it('the twice sentence names no sender: a hand-pressed Continue the app continues after reads true', () => {
    const turns = splitTurns(
      [user('fix it'), lengthEnd, continued, { ...lengthEnd, ts: 9 }, { ...continued, ts: 10, promptIndex: 2 }],
      [],
    );
    const state = { last: false, running: false, live: null, mode: 'work' as const };
    expect(turnOutcome(turns[1]!, state)?.text).toBe(
      'The answer hit the length limit twice in a row. Continuing did not help.',
    );
  });

  it('the pending row says the app continues while the turn holds only the divider', () => {
    const [, only] = splitTurns([user('fix it'), lengthEnd, continued], []);
    expect(continuingTurn(only, 1)).toBe(true);
    expect(continuingTurn(only, undefined)).toBe(false);
    const [, working] = splitTurns([user('fix it'), lengthEnd, continued, read], []);
    expect(continuingTurn(working, 1)).toBe(false);
    const base = {
      creating: false,
      phase: 'running' as const,
      wait: null,
      model: 'qwen-a',
      loading: null,
      inference: 'ready' as const,
      quiet: false,
      full: null,
    };
    expect(pendingRow({ ...base, continuing: true })).toEqual({
      kind: 'label',
      text: CHAT.pendingContinuing,
      orb: 'think',
    });
  });

  // Mutation: ignore `timed` → red (a thought read from disk shows a time).
  it('a thought has a time only when it was timed live', () => {
    expect(thoughtDuration({ ts: 1_000, endTs: 6_000, timed: true })).toBe(5_000);
    expect(thoughtDuration({ ts: 1_000, endTs: 6_000 })).toBeNull();
    expect(thoughtDuration({ ts: 1_000, endTs: 1_000, timed: true })).toBeNull();
    expect(thoughtDuration({ ts: 0, endTs: 6_000, timed: true })).toBeNull();
  });
});

describe('retry notices', () => {
  type Retry = Extract<TranscriptEntry, { kind: 'notice'; code: 'retry' }>;
  const retry = (over: Partial<Retry> = {}): Retry => ({
    kind: 'notice',
    ts: 10_000,
    code: 'retry',
    cause: 'stalled',
    attempts: 1,
    max: 3,
    ...over,
  });

  it('a notice never starts a turn and is no step of the answer', () => {
    const turns = splitTurns([user('one'), retry(), answer('a'), end(), user('two'), retry({ endTs: 11_000 })], []);
    expect(turns.map((t) => t.entries.map((e) => e.kind))).toEqual([
      ['user', 'notice', 'assistant'],
      ['user', 'notice'],
    ]);
    expect(isSideEntry(retry())).toBe(true);
    expect(isSideEntry({ kind: 'user', ts: 1, text: 'note', promptIndex: null, note: true })).toBe(true);
    expect(isSideEntry(user('typed'))).toBe(false);
    expect(isSideEntry(answer('a'))).toBe(false);
  });

  it('a recap never starts a turn and is no step: the next prompt starts its own', () => {
    const recap: TranscriptEntry = { kind: 'recap', ts: 5, text: 'Where we are.' };
    expect(isSideEntry(recap)).toBe(true);
    const turns = splitTurns([user('one'), answer('a'), end(), recap, user('two'), answer('b')], []);
    expect(turns.map((t) => t.entries.map((e) => e.kind))).toEqual([
      ['user', 'assistant'],
      ['recap'],
      ['user', 'assistant'],
    ]);
  });

  // Mutation: drop 'goal' from isSideEntry → the goal row counts as the answer's first step.
  it('a goal row never starts a turn and is no step: it stays in the turn of the message that set it', () => {
    const goal: TranscriptEntry = {
      kind: 'goal',
      ts: 2,
      goalId: 'g1',
      objective: 'fix it',
      status: 'active',
      phase: 'executing',
      tokensUsed: 0,
      elapsedMs: 0,
      asOf: 2,
    };
    expect(isSideEntry(goal)).toBe(true);
    const turns = splitTurns([user('/goal fix it'), goal, answer('a'), end(), user('two')], []);
    expect(turns.map((t) => t.entries.map((e) => e.kind))).toEqual([['user', 'goal', 'assistant'], ['user']]);
  });

  it("openRetry: the stream's live episode while it has no end, else null", () => {
    expect(openRetry(retry({ ts: 3 }))?.ts).toBe(3);
    expect(openRetry(retry({ endTs: 11_000, outcome: 'recovered' }))).toBeNull();
    expect(openRetry(null)).toBeNull();
    expect(openRetry(undefined)).toBeNull();
  });

  it('the line: cause words, then the try of the budget, or the try alone', () => {
    expect(retryLine(retry({ cause: 'busy', attempts: 2, max: 8 }))).toBe(
      `${CHAT.retry.live.busy} · ${fill(CHAT.retry.try, { n: 2, max: 8 })}`,
    );
    expect(retryLine(retry({ max: 0 }))).toBe(`${CHAT.retry.live.stalled} · ${fill(CHAT.retry.tryNoMax, { n: 1 })}`);
  });

  // Mutation: drop the `running` test → red (a stopped agent's episode reads as still trying).
  it('the row: recovered with the wait, gave up, ended with no outcome, still trying, never finished', () => {
    expect(retryRow(retry({ attempts: 2, endTs: 74_000, outcome: 'recovered' }), false)).toEqual({
      label: CHAT.retry.row,
      text: CHAT.retry.cause.stalled,
      facts: [fill(CHAT.retry.tries, { n: 2 }), fill(CHAT.retry.waited, { time: '1m 04s' })],
      tone: 'neutral',
    });
    // Ended by a late chunk the model made before the try (stored stamps run backwards): no wait to show.
    expect(retryRow(retry({ endTs: 9_000, outcome: 'recovered' }), false).facts).toEqual([CHAT.retry.triesOne]);
    expect(retryRow(retry({ attempts: 3, endTs: 20_000, outcome: 'gave-up', detail: 'API error' }), false)).toEqual({
      label: CHAT.retry.rowGaveUp,
      text: CHAT.retry.cause.stalled,
      facts: [fill(CHAT.retry.tries, { n: 3 })],
      tone: 'danger',
      detail: 'API error',
    });
    expect(retryRow(retry({ endTs: 20_000 }), false)).toEqual({
      label: CHAT.retry.row,
      text: CHAT.retry.cause.stalled,
      facts: [CHAT.retry.triesOne],
      tone: 'neutral',
    });
    expect(retryRow(retry({ cause: 'other', detail: 'Re-authenticated after 401' }), true)).toEqual({
      label: CHAT.retry.rowLive,
      text: CHAT.retry.cause.other,
      facts: [fill(CHAT.retry.try, { n: 1, max: 3 })],
      tone: 'neutral',
      detail: 'Re-authenticated after 401',
    });
    expect(retryRow(retry(), false)).toEqual({
      label: CHAT.retry.rowOpen,
      text: CHAT.retry.cause.stalled,
      facts: [CHAT.retry.triesOne],
      tone: 'warning',
    });
  });

  it("the read meter follows this conversation's newest request since the episode began, only while it reads", () => {
    const request = (over: Partial<EngineRequest>): EngineRequest => ({
      session: 'conv-1',
      modelId: 'm',
      phase: 'prefill',
      startedAt: 12_000,
      prefillDone: 100,
      prefillTotal: 400,
      generated: null,
      ...over,
    });
    const read = (requests: EngineRequest[], session: string | null = 'conv-1') =>
      retryReading({ requests, session, since: 10_000 });
    expect(read([request({})])).toEqual({ done: 100, total: 400 });
    expect(read([request({}), request({ startedAt: 13_000, prefillDone: 300 })])).toEqual({ done: 300, total: 400 });
    expect(read([request({ startedAt: 9_000 })])).toBeNull();
    expect(read([request({ session: 'conv-2' })])).toBeNull();
    expect(read([request({ phase: 'decode' })])).toBeNull();
    expect(read([request({})], null)).toBeNull();
  });
});

describe('rewind helpers', () => {
  const facts: HistoryFacts = {
    blocked: null,
    busy: false,
    runLive: false,
    helperLive: false,
    engineLost: false,
    finished: 0,
    op: null,
  };

  it('a rewind point is a message the user typed: not a note, a steer or the automatic continue', () => {
    const entries: TranscriptEntry[] = [
      user('one'),
      answer('a'),
      { kind: 'user', ts: 1, text: 'note', promptIndex: null, note: true },
      { kind: 'user', ts: 1, text: 'steer', promptIndex: null, interjection: true },
      { kind: 'user', ts: 1, text: CONTINUE_PROMPT, promptIndex: null, continued: true },
    ];
    expect(entries.map(isRewindPoint)).toEqual([true, false, false, false, false]);
    expect(lastRewindPoint(entries)).toBe(0);
    expect(lastRewindPoint([answer('a')])).toBeNull();
  });

  it('is off while the agent is blocked, busy, lost or running helpers; not off with no finished answer', () => {
    expect(rewindOff(facts)).toBeNull();
    expect(rewindOff({ ...facts, blocked: 'x' })).toBe('x');
    expect(rewindOff({ ...facts, busy: true })).toBe(CHAT.modelBusy);
    expect(rewindOff({ ...facts, engineLost: true })).toBe(CHAT.history.engineLost);
    expect(rewindOff({ ...facts, runLive: true })).toBe(CHAT.history.offRun);
    expect(rewindOff({ ...facts, runLive: true, helperLive: true })).toBe(CHAT.history.offHelper);
  });

  // Mutation: drop `opRunning` from the walls → a second Fork reads "An answer is still running."
  it('a fork or a recap that runs turns the other ops off with its own words', () => {
    for (const off of [compactOff, rewindOff, forkOff]) {
      expect(off({ ...facts, finished: 1, op: 'fork' })).toBe(CHAT.history.forkRunning);
      expect(off({ ...facts, finished: 1, op: 'recap' })).toBe(CHAT.history.offRecap);
      expect(off({ ...facts, finished: 1, op: 'compact', busy: true })).toBe(CHAT.modelBusy);
    }
    expect(CHAT.history.offRecap).toBe('Wait for the recap.');
    expect(recapOff({ ...facts, finished: 1, op: 'recap' })).toBe(CHAT.history.recapWriting);
    expect(recapOff({ ...facts, finished: 1, op: 'recap', busy: true })).toBe(CHAT.history.recapWriting);
    const error = (reason: string): DashboardApiError => new DashboardApiError(409, 'E_CONFLICT', 'no', reason);
    expect(forkRefusal(error('fork-running'))).toBe(CHAT.history.forkRunning);
    expect(forkRefusal(error('recap-running'))).toBe(CHAT.history.offRecap);
    expect(rewindRefusal(error('recap-running'))).toEqual({ tone: 'danger', text: CHAT.history.offRecap });
  });

  it('the words of a refusal and of an end with no rewind', () => {
    const error = (reason: string): DashboardApiError => new DashboardApiError(409, 'E_CONFLICT', 'no', reason);
    expect(rewindRefusal(error('rewind-not-point'))).toEqual({ tone: 'info', text: CHAT.history.errUnsent });
    expect(rewindRefusal(error('rewind-changed'))).toEqual({ tone: 'warning', text: CHAT.history.errChanged });
    expect(rewindRefusal(error('conversation-busy'))).toEqual({
      tone: 'danger',
      text: CHAT.sendErrors['conversation-busy'][0],
    });
    expect(rewindRefusal(new Error('x'))).toEqual({ tone: 'danger', text: CHAT.history.errFailed });
    expect(
      (['refused', 'tasks', 'stopped', 'failed', 'load-slow'] as const).map((outcome) => rewindEndText({ outcome })),
    ).toEqual([
      CHAT.history.errRefused,
      CHAT.history.errTasks,
      CHAT.history.errStopped,
      CHAT.history.errFailed,
      CHAT.history.loadSlow,
    ]);
    expect(forkRefusal(error('fork-changed'))).toBe(CHAT.history.forkChanged);
  });

  // Mutation: read `load-slow` as any failure → each op says "could not" instead of the slow open.
  it('a cold conversation the agent opened too slowly: every op says so in its own place', () => {
    const slow = new DashboardApiError(503, 'E_UNAVAILABLE', 'slow', 'load-slow');
    expect(rewindRefusal(slow)).toEqual({ tone: 'warning', text: CHAT.history.loadSlow, retry: true });
    expect(compactEndWords('load-slow', [])).toEqual({ toast: 'error', text: CHAT.history.loadSlow });
    expect(recapEndToast('load-slow')).toEqual({ tone: 'error', text: CHAT.history.loadSlow });
    expect(sendFailure('load-slow', 'E_UNAVAILABLE', 'work')).toEqual({
      strip: { tone: 'danger', text: CHAT.history.loadSlow, action: { label: CHAT.history.tryAgain, kind: 'retry' } },
    });
    expect(BLOCKER_SENTENCES['load-slow']).toBe(CHAT.history.loadSlow);
    expect(CHAT.history.loadSlow).toBe('The agent is slow to open this conversation. Try again.');
  });
});

describe('picture notices and the no-model end', () => {
  type Picture = Extract<TranscriptEntry, { kind: 'notice'; code: 'picture' }>;
  const picture = (over: Partial<Picture> = {}): Picture => ({
    kind: 'notice',
    ts: 10,
    code: 'picture',
    what: 'dropped',
    count: 1,
    notes: ['Image 1 was dropped before send: unreadable Png header.'],
    ...over,
  });

  it('one turn for prompt + notice + answer; a notice is never a prompt and never a step', () => {
    const turns = splitTurns([user('one'), picture(), answer('a'), end(), picture({ ts: 20 })], []);
    expect(turns.map((t) => t.entries.map((e) => e.kind))).toEqual([['user', 'notice', 'assistant'], ['notice']]);
    expect(turnPrompt(turns[1]!)).toBeNull();
    expect(isSideEntry(picture())).toBe(true);
  });

  // Mutation: count the dropped headline again → one agent note for 3 pictures reads "A picture was left out".
  it('a row: left out is a warning with no count in its headline and the notes, at most 6 lines; made smaller is neutral', () => {
    const notes = Array.from({ length: 8 }, (_, n) => `note ${n}`);
    expect(pictureRow(picture({ count: 1, notes: notes.slice(0, 1) })).text).toBe(CHAT.pictureNote.dropped);
    expect(pictureRow(picture({ count: 8, notes }))).toEqual({
      label: CHAT.pictureNote.label,
      text: CHAT.pictureNote.dropped,
      facts: [],
      tone: 'warning',
      detail: notes.slice(0, 6).join('\n'),
    });
    const size = {
      fromBytes: 2_048,
      toBytes: 1_024,
      from: [40, 30] as [number, number],
      to: [20, 15] as [number, number],
    };
    expect(pictureRow(picture({ what: 'shrunk', notes: ['Downscaled'], sizes: [size] }))).toEqual({
      label: CHAT.pictureNote.label,
      text: CHAT.pictureNote.shrunk,
      facts: [
        fill(CHAT.pictureNote.size, { from: '2.0 KB', to: '1.0 KB' }),
        fill(CHAT.pictureNote.pixels, { w: 40, h: 30, w2: 20, h2: 15 }),
      ],
      tone: 'neutral',
    });
    expect(pictureRow(picture({ what: 'shrunk', count: 2, notes: ['Downscaled two'] }))).toMatchObject({
      text: fill(CHAT.pictureNote.shrunkMany, { n: 2 }),
      detail: 'Downscaled two',
      tone: 'neutral',
    });
  });

  it('no-model: a warning with Send again; the live end adds the reason line, a stored turn never', () => {
    expect(liveOutcome('no-model', 'chat')).toEqual({
      tone: 'warning',
      glyph: 'alert',
      text: CHAT.outcomes['no-model'],
      action: 'send-again',
    });
    const [turn] = splitTurns([user('Fix the failing test')], []);
    const model = { previous: null, next: null, reason: 'No model is allowed.' };
    const state = { last: true, running: false, mode: 'work' as const };
    expect(turnOutcome(turn!, { ...state, live: { outcome: 'no-model', model } })).toMatchObject({
      text: CHAT.outcomes['no-model'],
      detail: `${CHAT.modelAuto.none}: No model is allowed.`,
    });
    expect(turnOutcome(turn!, { ...state, live: { outcome: 'no-model' } })).not.toHaveProperty('detail');
    // A completed end draws no line, so the reason has none either.
    expect(turnOutcome(turn!, { ...state, live: { outcome: 'completed', model } })).toBeNull();
  });

  it('the reason line: our sentence first, then the agent words; null ids read as no model', () => {
    expect(modelDetail({ previous: 'a', next: null, reason: 'Gone.' })).toBe(
      `${fill(CHAT.modelAuto.refused, { previous: 'a' })}: Gone.`,
    );
    expect(modelDetail({ previous: 'a', next: 'b', reason: 'Gone.' })).toBe(
      `${fill(CHAT.modelAuto.wanted, { previous: 'a', next: 'b' })} Gone.`,
    );
    expect(modelDetail({ previous: null, next: null, reason: '' })).toBe(CHAT.modelAuto.none);
  });
});

describe('hook results (#11)', () => {
  type Hook = Extract<TranscriptEntry, { kind: 'notice'; code: 'hook' }>;
  const failed = (over: Partial<Hook> = {}): Hook => ({
    kind: 'notice',
    ts: 10,
    code: 'hook',
    event: 'pre_tool_use',
    tool: 'run_terminal_command',
    hook: 'global/lint',
    detail: 'exit status 2',
    times: 1,
    lastTs: 10,
    ...over,
  });

  it('when a hook ran, in words: the tool by kind, the known events, any other event as sent', () => {
    expect(hookWhen('pre_tool_use', 'run_terminal_command')).toBe('before the command');
    expect(hookWhen('post_tool_use', 'search_replace')).toBe('after the edit');
    expect(hookWhen('pre_tool_use', 'read_file')).toBe('before the read');
    expect(hookWhen('pre_tool_use', undefined)).toBe('before a tool');
    expect(hookWhen('pre_tool_use', 'web_search')).toBe('before web_search');
    expect(hookWhen('user_prompt_submit', undefined)).toBe(CHAT.hook.when.user_prompt_submit);
    expect(hookWhen('stop', undefined)).toBe(CHAT.hook.when.stop);
    expect(hookWhen('pre_compact', undefined)).toBe('pre_compact');
    expect(hookWhen('pre\u200bcompact', undefined)).not.toContain('\u200b');
  });

  it("the app's own guard reads as words; a near name and any other hook stay as sent", () => {
    expect(hookName('plugin/mlx-work-guard-0123456789abcdef/plugin')).toBe(CHAT.hook.ownGuard);
    expect(hookName('plugin/mlx-work-guard-0123456789ABCDEF/plugin')).toBe(
      'plugin/mlx-work-guard-0123456789ABCDEF/plugin',
    );
    expect(hookName('plugin/mlx-work-guard-0123456789abcde/plugin')).toBe(
      'plugin/mlx-work-guard-0123456789abcde/plugin',
    );
    expect(hookName('global/lint')).toBe('global/lint');
  });

  it('a failed row: once is a warning with the error under it; 12 times adds the count and the newest time', () => {
    expect(hookRow(failed())).toEqual({
      label: CHAT.hook.failed,
      text: 'global/lint · before the command',
      facts: [],
      tone: 'warning',
      detail: 'exit status 2',
    });
    expect(hookRow(failed({ times: 12, lastTs: 99_000, detail: '' }))).toEqual({
      label: CHAT.hook.failed,
      text: 'global/lint · before the command',
      facts: ['12 times', `last ${formatClock(99_000)}`],
      tone: 'warning',
    });
    expect(hookHeadline(failed())).toBe(`${CHAT.hook.failed}: global/lint · before the command`);
  });

  it('a note row: the words as the text, neutral; its headline', () => {
    const note = { kind: 'notice' as const, ts: 1, code: 'hook-note' as const, text: 'Stop blocked by hook' };
    expect(hookNoteRow(note)).toEqual({
      label: CHAT.hook.note,
      text: 'Stop blocked by hook',
      facts: [],
      tone: 'neutral',
    });
    expect(hookHeadline(note)).toBe(`${CHAT.hook.note}: Stop blocked by hook`);
  });

  it('the wait line: one hook or many, and when', () => {
    expect(hookWaitLine({ event: 'stop', count: 1, startedAt: 0 })).toBe(`${CHAT.hook.wait} · ${CHAT.hook.when.stop}`);
    expect(hookWaitLine({ event: 'pre_tool_use', tool: 'run_terminal_command', count: 3, startedAt: 0 })).toBe(
      'Running 3 hooks · before the command',
    );
  });

  it('a prompt a hook blocked ends Stopped by a hook, live and stored, with no Send again', () => {
    const line = { tone: 'neutral', glyph: 'stop', text: CHAT.outcomes['hook-stopped'], action: null };
    expect(liveOutcome('hook-stopped', 'chat')).toEqual(line);
    expect(liveOutcome('hook-stopped', 'work')).toEqual(line);
    expect(storedOutcome({ kind: 'turn-end', ts: 0, stopReason: 'cancelled', hookStopped: true })).toEqual(line);
    expect(storedOutcome({ kind: 'turn-end', ts: 0, stopReason: 'cancelled' })).not.toEqual(line);
  });
});
