/**
 * One conversation's stream: `sync` first, `seq` without gaps across listeners, the 33 ms op batching with merged
 * text, hold-back redaction of a secret split over chunks (current and retired needles), and a fresh `sync` after a
 * hot → cold switch. The page's reducer refuses a gap.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { parseRunState } from '../src/agent-sessions/run-state.js';
import type { UpdateLine } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { CHAT_FLUSH_MS, Conversation, mergeTextOps, type HostView } from '../src/chat/conversation.js';
import type { ConversationEvent, ConversationInfo } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';

const INFO: ConversationInfo = {
  id: 'c1',
  mode: 'chat',
  title: '',
  cwd: null,
  modelId: 'qwen3.5-4b',
  createdAt: 1,
  updatedAt: 1,
  hot: true,
  permissionMode: 'ask',
};
const HOST: HostView = {
  turn: null,
  interactions: [],
  compaction: null,
  hookRun: null,
  recap: null,
  runs: [],
  live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
  commands: null,
};

function chunk(text: string, kind = 'agent_message_chunk'): UpdateLine {
  return { xai: false, update: { sessionUpdate: kind, content: { type: 'text', text } }, meta: undefined };
}

function fold(events: ConversationEvent[]): ConversationState {
  let state: ConversationState | null = null;
  for (const event of events) {
    const next = applyConversationEvent(state, event);
    if (next === 'resync') throw new Error(`gap at ${event.type} ${event.seq}`);
    state = next;
  }
  return state!;
}

function texts(state: ConversationState): string[] {
  return state.entries.flatMap((e) => (e.kind === 'assistant' || e.kind === 'thinking' ? [e.text] : []));
}

describe('a field the transcript cut', () => {
  it('never ends in part of a key', () => {
    const key = 'fc-0123456789abcdef0123456789abcdef';
    const cut = `${'x'.repeat(100)} ${key.slice(0, 14)}`;
    const conv = new Conversation(INFO, { needles: () => [key] });
    const disk: TranscriptEntry[] = [
      {
        kind: 'tool',
        ts: 1,
        id: 't1',
        name: 'run_terminal_command',
        title: 'cat .env',
        status: 'completed',
        output: cut,
        truncated: true,
      },
    ];
    conv.seed(disk, [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    const sync = a[0] as Extract<ConversationEvent, { type: 'sync' }>;
    const output = (sync.entries[0] as { output: string }).output;
    expect(output).toBe(`${'x'.repeat(100)} `);
    expect(output).not.toContain('fc-');
  });
});

describe('the model a turn could not use (#16)', () => {
  // Mutation: drop the model copy in turnEnded → the page never sees the reason.
  it('rides the turn event and the sync to the page store, redacted, a cut end that could start a key too', () => {
    const key = 'fc-0123456789abcdef0123456789abcdef';
    const conv = new Conversation(INFO, { needles: () => [key] });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    const reason = `${'r'.repeat(280)} ${key.slice(0, 14)}…`;
    conv.turnEnded({ turnId: 't', outcome: 'no-model', model: { previous: key, next: null, reason } });
    const model = { previous: '[redacted]', next: null, reason: `${'r'.repeat(280)}…` };
    expect(fold(a).lastEnd).toEqual({ turnId: 't', outcome: 'no-model', model });
    const b: ConversationEvent[] = [];
    conv.attach((e) => b.push(e), HOST);
    expect(fold(b).lastEnd).toEqual({ turnId: 't', outcome: 'no-model', model });
  });
});

describe('conversation stream', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('starts with a complete sync and numbers every later event by one', () => {
    const conv = new Conversation(INFO, { needles: () => [] });
    const disk: TranscriptEntry[] = [{ kind: 'user', ts: 1, text: 'earlier', promptIndex: 0 }];
    conv.seed(disk, [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    expect(a[0]).toMatchObject({ type: 'sync', seq: 0, entries: disk, turn: null });

    conv.userPrompt('now');
    conv.applyLine(chunk('Hel'));
    conv.applyLine(chunk('lo'));
    expect(a).toHaveLength(1);
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    expect(a).toHaveLength(2);
    const ops = a[1]!;
    expect(ops.type).toBe('ops');
    if (ops.type === 'ops') {
      expect(ops.seq).toBe(1);
      expect(ops.ops.map((o) => o.op)).toEqual(['push', 'push', 'text']);
    }

    // A late page gets the view so far with the current seq, then both pages see the same numbers.
    const b: ConversationEvent[] = [];
    conv.attach((e) => b.push(e), HOST);
    expect(b[0]).toMatchObject({ type: 'sync', seq: 1 });
    conv.turnChanged({ turnId: 't', phase: 'running', startedAt: 2, wait: null });
    conv.applyLine(chunk('!'));
    conv.turnEnded({ turnId: 't', outcome: 'completed' });
    expect(fold(a)).toMatchObject({ seq: 4, lastEnd: { outcome: 'completed' } });
    expect(fold(b).seq).toBe(4);
    expect(texts(fold(a))).toEqual(['Hello!']);
    expect(texts(fold(b))).toEqual(['Hello!']);
  });

  it('flushes the queue before a turn event, and posts nothing with no page attached', () => {
    const conv = new Conversation(INFO, { needles: () => [] });
    conv.seed([], [], null);
    conv.applyLine(chunk('quiet'));
    expect(vi.getTimerCount()).toBe(0);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    expect(texts(fold(a))).toEqual(['quiet']);
    conv.applyLine(chunk(' more'));
    conv.turnEnded({ turnId: 't', outcome: 'cancelled' });
    expect(a.map((e) => e.type)).toEqual(['sync', 'ops', 'turn']);
    expect(texts(fold(a))).toEqual(['quiet more']);
  });

  it('merges consecutive text ops on one entry', () => {
    expect(
      mergeTextOps([
        { op: 'text', index: 1, append: 'a' },
        { op: 'text', index: 1, append: 'b' },
        { op: 'text', index: 2, append: 'c' },
        { op: 'text', index: 1, append: 'd' },
      ]),
    ).toEqual([
      { op: 'text', index: 1, append: 'ab' },
      { op: 'text', index: 2, append: 'c' },
      { op: 'text', index: 1, append: 'd' },
    ]);
  });

  it('holds back text so a secret split over two chunks is redacted, for current and retired needles', () => {
    const needles = ['sk-current-KEY-0001', 'sk-retired-KEY-0002'];
    const conv = new Conversation(INFO, { needles: () => needles });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    conv.applyLine(chunk('key one sk-curr'));
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    conv.applyLine(chunk('ent-KEY-0001 and two sk-retired-'));
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    conv.applyLine(chunk('KEY-0002 end'));
    conv.turnEnded({ turnId: 't', outcome: 'completed' });
    const wire = JSON.stringify(a);
    for (const n of needles) expect(wire).not.toContain(n);
    expect(wire).not.toContain('sk-curr"');
    expect(texts(fold(a))).toEqual(['key one [redacted] and two [redacted] end']);
  });

  it('a needle that arrives after the seed still redacts the next sync', () => {
    const needles: string[] = [];
    const conv = new Conversation(INFO, { needles: () => needles });
    conv.seed([{ kind: 'assistant', ts: 1, text: 'key sk-late-KEY-0003 end' }], [], null);
    needles.push('sk-late-KEY-0003');
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    expect(JSON.stringify(a)).not.toContain('sk-late-KEY-0003');
    expect(texts(fold(a))).toEqual(['key [redacted] end']);
  });

  it('a key that becomes known while a text entry streams is redacted in what follows, the held-back text too', () => {
    const late = 'exa-late-KEY-0123456789';
    const needles = ['tok-0000000000'];
    const conv = new Conversation(INFO, { needles: () => needles });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    conv.userPrompt('q');
    conv.applyLine(chunk('start exa-late-'));
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    needles.push(late);
    conv.resync(HOST);
    conv.applyLine(chunk(`KEY-0123456789 then ${late} end`));
    conv.turnEnded({ turnId: 't', outcome: 'completed' });
    expect(JSON.stringify(a)).not.toContain(late);
    expect(texts(fold(a))).toEqual(['start [redacted] then [redacted] end']);
  });

  it('a key that becomes known mid-entry is redacted wherever it was split, with or without a needle before it', () => {
    const late = 'exa-late-KEY-0123456789';
    const leaks: string[] = [];
    for (const before of [['tok-0000000000'], []]) {
      for (const resync of [true, false]) {
        for (let s = 0; s <= late.length; s++) {
          const needles = [...before];
          const conv = new Conversation(INFO, { needles: () => needles });
          conv.seed([], [], null);
          const a: ConversationEvent[] = [];
          conv.attach((e) => a.push(e), HOST);
          conv.userPrompt('q');
          conv.applyLine(chunk('start '));
          conv.applyLine(chunk(late.slice(0, s)));
          conv.flush();
          needles.push(late);
          if (resync) {
            conv.resync(HOST);
            const mid = texts(fold(a)).join('|');
            if (s > 0 && mid.includes(late.slice(0, s)))
              leaks.push(`needles=${before.length} split=${s} at resync: ${mid}`);
          }
          conv.applyLine(chunk(`${late.slice(s)} then ${late} end`));
          conv.turnEnded({ turnId: 't', outcome: 'completed' });
          const shown = texts(fold(a));
          if (shown.join('|') !== 'start [redacted] then [redacted] end') {
            leaks.push(`needles=${before.length} resync=${resync} split=${s}: ${shown.join('|')}`);
          }
        }
      }
    }
    expect(leaks).toEqual([]);
  });

  it('a page that attaches after a key became known mid-entry gets no part of it in its sync', () => {
    const late = 'exa-late-KEY-0123456789';
    const shown: string[] = [];
    for (let s = 4; s < late.length; s++) {
      const needles: string[] = [];
      const conv = new Conversation(INFO, { needles: () => needles });
      conv.seed([], [], null);
      conv.userPrompt('q');
      conv.applyLine(chunk(`start ${late.slice(0, s)}`));
      needles.push(late);
      const a: ConversationEvent[] = [];
      conv.attach((e) => a.push(e), HOST);
      if (JSON.stringify(a).includes(late.slice(0, s))) shown.push(`split=${s}`);
      conv.applyLine(chunk(`${late.slice(s)} end`));
      conv.turnEnded({ turnId: 't', outcome: 'completed' });
      expect(texts(fold(a))).toEqual(['start [redacted] end']);
    }
    expect(shown).toEqual([]);
  });

  it('a needle that is gone after a rekey stays redacted in text already sent', () => {
    const needles = ['tok-OLD-0000000000'];
    const conv = new Conversation(INFO, { needles: () => needles });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    conv.userPrompt('q');
    conv.applyLine(chunk(`was tok-OLD-0000000000 and ${'x'.repeat(40)}`));
    conv.flush();
    needles.splice(0, 1, 'tok-NEW-0000000000');
    conv.resync(HOST);
    conv.applyLine(chunk(' end'));
    conv.turnEnded({ turnId: 't', outcome: 'completed' });
    expect(JSON.stringify(a)).not.toContain('tok-OLD-0000000000');
    expect(texts(fold(a))).toEqual([`was [redacted] and ${'x'.repeat(40)} end`]);
  });

  it('a gap in seq makes the page ask for a fresh sync', () => {
    const conv = new Conversation(INFO, { needles: () => [] });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    conv.turnChanged({ turnId: 't', phase: 'starting', startedAt: 2, wait: null });
    conv.turnChanged({ turnId: 't', phase: 'running', startedAt: 2, wait: null });
    const state = applyConversationEvent(null, a[0]!) as ConversationState;
    expect(applyConversationEvent(state, a[2]!)).toBe('resync');
    expect(applyConversationEvent(null, a[1]!)).toBe('resync');
  });

  it('sends every page a fresh sync after a hot → cold switch re-read the disk', () => {
    const conv = new Conversation(INFO, { needles: () => [] });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    conv.userPrompt('q');
    conv.applyLine(chunk('partial'));
    conv.info = { ...conv.info, hot: false };
    conv.seed(
      [
        { kind: 'user', ts: 1, text: 'q', promptIndex: 0 },
        { kind: 'assistant', ts: 2, text: 'partial on disk' },
      ],
      [],
      null,
    );
    conv.resync(HOST);
    const state = fold(a);
    expect(state.conversation.hot).toBe(false);
    expect(texts(state)).toEqual(['partial on disk']);
    // The ops queued before the re-read were never sent: the fresh sync replaces them.
    expect(a.map((e) => [e.type, e.seq])).toEqual([
      ['sync', 0],
      ['sync', 1],
    ]);
  });

  it('a page that attaches after a turn ended gets how it ended in its sync, as the live event gave it', () => {
    const conv = new Conversation(INFO, { needles: () => [] });
    conv.seed([], [], null);
    conv.userPrompt('read that');
    conv.turnEnded({ turnId: 't9', outcome: 'tool-refused' });
    const later: ConversationEvent[] = [];
    conv.attach((e) => later.push(e), HOST);
    expect(later[0]).toMatchObject({ type: 'sync', lastEnd: { turnId: 't9', outcome: 'tool-refused' } });
    expect(fold(later).lastEnd).toEqual({ turnId: 't9', outcome: 'tool-refused' });

    const live: ConversationEvent[] = [];
    conv.attach((e) => live.push(e), HOST);
    conv.turnEnded({ turnId: 't10', failure: { code: 'agent-crashed', message: 'The agent stopped.' } });
    const failed = { turnId: 't10', failure: { code: 'agent-crashed', message: 'The agent stopped.' } };
    expect(fold(live).lastEnd).toEqual(failed);
    const after: ConversationEvent[] = [];
    conv.attach((e) => after.push(e), HOST);
    expect(fold(after).lastEnd).toEqual(failed);

    // A new turn: a page that attaches while it runs gets no end of the turn before.
    conv.userPrompt('again');
    const running: ConversationEvent[] = [];
    conv.attach((e) => running.push(e), {
      turn: { turnId: 't11', phase: 'running', startedAt: 3, wait: null },
      interactions: [],
      compaction: null,
      hookRun: null,
      recap: null,
      runs: [],
      live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
      commands: null,
    });
    expect(running[0]).not.toHaveProperty('lastEnd');
    expect(fold(running).lastEnd).toBeNull();
  });

  it('the context use: seeded from disk, moved by _meta.totalTokens, kept by a line without it, sent to every page', () => {
    const counted = (text: string, totalTokens?: number): UpdateLine => ({
      ...chunk(text),
      meta: totalTokens === undefined ? undefined : { totalTokens },
    });
    const conv = new Conversation(INFO, { needles: () => [] });
    conv.seed([{ kind: 'user', ts: 1, text: 'earlier', promptIndex: 0 }], [], null, 4475);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    expect(a[0]).toMatchObject({ type: 'sync', contextUsed: 4475 });
    expect(fold(a).contextUsed).toBe(4475);

    conv.userPrompt('next');
    conv.applyLine(counted('a', 4872));
    conv.applyLine(counted('b', 5833));
    conv.applyLine(counted('c'));
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    const ops = a[1]!;
    // One batch: the text ops still merge, and only the last count goes out.
    expect(ops.type === 'ops' && ops.ops.map((o) => o.op)).toEqual(['push', 'push', 'text', 'context']);
    expect(fold(a).contextUsed).toBe(5833);
    expect(texts(fold(a))).toEqual(['abc']);

    const late: ConversationEvent[] = [];
    conv.attach((e) => late.push(e), HOST);
    expect(fold(late).contextUsed).toBe(5833);

    // Loaded again from disk (hot → cold, or an app restart): the stored count comes back with the thread.
    conv.seed([{ kind: 'user', ts: 1, text: 'earlier', promptIndex: 0 }], [], null, 6262);
    conv.resync(HOST);
    expect(fold(a).contextUsed).toBe(6262);
    conv.seed([], [], null);
    conv.resync(HOST);
    expect(fold(a).contextUsed).toBeNull();
  });

  it('a compaction: its state on the stream and in a late sync, its marker as an op, the state gone at its end', () => {
    const conv = new Conversation(INFO, { needles: () => [] });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    expect(a[0]).not.toHaveProperty('compaction');
    expect(fold(a).compaction).toBeNull();
    const running = { phase: 'running' as const, startedAt: 7, tokensBefore: 222_822 };
    conv.compactionChanged(running);
    expect(fold(a).compaction).toEqual(running);
    const late: ConversationEvent[] = [];
    conv.attach((e) => late.push(e), { ...HOST, compaction: running });
    expect(late[0]).toMatchObject({ type: 'sync', compaction: running });
    expect(fold(late).compaction).toEqual(running);

    conv.applyLine({
      xai: true,
      update: { sessionUpdate: 'auto_compact_completed', tokens_before: 222_822, tokens_after: 31_204 },
      meta: { agentTimestampMs: 9 },
    });
    conv.compactionChanged(null);
    // The marker op goes out before the state ends: the page never shows neither.
    expect(a.slice(-2).map((e) => e.type)).toEqual(['ops', 'compaction']);
    expect(fold(a).compaction).toBeNull();
    expect(fold(a).entries).toEqual([
      { kind: 'compaction', ts: 9, outcome: 'done', tokensBefore: 222_822, tokensAfter: 31_204 },
    ]);
  });

  it('a compaction that starts shows the text held back for redaction at once, and later text still redacts', () => {
    const key = 'sk-current-KEY-0001';
    const conv = new Conversation(INFO, { needles: () => [key] });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    conv.userPrompt('compact two');
    conv.applyLine(chunk('Let me look. sk-cur'));
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    expect(texts(fold(a))).toEqual(['Let me look. ']);
    conv.compactionChanged({ phase: 'running', startedAt: 7 });
    expect(texts(fold(a))).toEqual(['Let me look. sk-cur']);
    expect(a.at(-1)!.type).toBe('compaction');

    conv.compactionChanged(null);
    conv.applyLine(chunk(` Now ${key} end`));
    conv.turnEnded({ turnId: 't', outcome: 'completed' });
    expect(JSON.stringify(a)).not.toContain(key);
    expect(texts(fold(a))).toEqual(['Let me look. sk-cur Now [redacted] end']);
  });

  it('a hook batch: its state on the stream and in a late sync, redacted, held-back text shown at its start, gone at its end', () => {
    const key = 'sk-current-KEY-0001';
    const conv = new Conversation(INFO, { needles: () => [key] });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    expect(a[0]).not.toHaveProperty('hookRun');
    expect(fold(a).hookRun).toBeNull();
    conv.userPrompt('run it');
    conv.applyLine(chunk('Let me look. sk-cur'));
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    expect(texts(fold(a))).toEqual(['Let me look. ']);
    const run = { event: 'pre_tool_use', tool: `run ${key}`, count: 2, startedAt: 7 };
    conv.hookRunChanged(run);
    expect(texts(fold(a))).toEqual(['Let me look. sk-cur']);
    expect(a.at(-1)).toMatchObject({ type: 'hook-run' });
    expect(JSON.stringify(a)).not.toContain(key);
    expect(fold(a).hookRun).toEqual({ ...run, tool: 'run [redacted]' });

    const late: ConversationEvent[] = [];
    conv.attach((e) => late.push(e), { ...HOST, hookRun: run });
    expect(late[0]).toMatchObject({ type: 'sync', hookRun: { tool: 'run [redacted]' } });
    expect(JSON.stringify(late)).not.toContain(key);
    expect(fold(late).hookRun).toEqual({ ...run, tool: 'run [redacted]' });

    conv.hookRunChanged(null);
    expect(fold(a).hookRun).toBeNull();
    expect(fold(late).hookRun).toBeNull();
  });

  it('a notice clears on every page: the event drops it, a later sync no longer has it, a second clear sends nothing', () => {
    const conv = new Conversation(INFO, { needles: () => [] });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    conv.notice({ code: 'key-printed', tone: 'danger', provider: 'exa' });
    conv.notice({ code: 'key-printed', tone: 'danger', provider: 'brave' });
    expect(fold(a).notices.map((n) => n.provider)).toEqual(['exa', 'brave']);
    conv.clearNotice('key-printed');
    conv.clearNotice('key-printed');
    expect(a.map((e) => e.type)).toEqual(['sync', 'notice', 'notice', 'notice-cleared']);
    expect(fold(a).notices).toEqual([]);
    const b: ConversationEvent[] = [];
    conv.attach((e) => b.push(e), HOST);
    expect(fold(b).notices).toEqual([]);
  });

  it('ends the stream with gone', () => {
    const conv = new Conversation(INFO, { needles: () => [] });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    conv.gone('deleted');
    conv.applyLine(chunk('late'));
    conv.turnEnded({ turnId: 't', outcome: 'completed' });
    vi.advanceTimersByTime(CHAT_FLUSH_MS * 2);
    expect(a.map((e) => e.type)).toEqual(['sync', 'gone']);
    expect(fold(a).gone).toBe('deleted');
  });
});

describe("a workflow run's live updates on the page", () => {
  const run = (revision: number, status: string, extra: Record<string, unknown> = {}): UpdateLine => ({
    xai: true,
    update: {
      sessionUpdate: 'workflow_updated',
      run_id: 'wf_1',
      revision,
      name: 'deep-research',
      status,
      phases: [
        { title: 'Plan', state: 'done' },
        { title: 'Research', state: 'active' },
      ],
      current_phase: 'Research',
      ...extra,
    },
    meta: { agentTimestampMs: 1_000 + revision },
  });
  const runOf = (state: ConversationState) => state.entries.find((e) => e.kind === 'workflow');

  // Mutation: flush the queued ops as their own event before the runs event → for one page step the run reads done
  // with its report while the host's word that a wake is due has not come, so the report draws in the block.
  it('a runs event carries the ops queued before it: the run and its due wake reach the page in one event', () => {
    vi.useFakeTimers();
    try {
      const conv = new Conversation({ ...INFO, mode: 'work' }, { needles: () => [] });
      conv.seed([], [], null);
      const a: ConversationEvent[] = [];
      conv.attach((e) => a.push(e), HOST);
      conv.applyLine(run(1, 'active'));
      vi.advanceTimersByTime(CHAT_FLUSH_MS);
      conv.applyLine(run(2, 'complete', { result_summary: '# Findings' }));
      const view = { runId: 'wf_1', name: 'deep-research', status: 'complete', live: false } as const;
      conv.runsChanged([{ ...view, stopping: false, by: null, waiting: false, wakeDue: true }]);
      expect(a.map((e) => e.type)).toEqual(['sync', 'ops', 'runs']);
      const sent = a[2]!;
      expect(sent.type === 'runs' && sent.ops?.map((op) => op.op)).toEqual(['set']);
      const page = fold(a);
      expect(runOf(page)).toMatchObject({ status: 'complete', report: '# Findings' });
      expect(page.runs[0]?.wakeDue).toBe(true);
      vi.advanceTimersByTime(CHAT_FLUSH_MS);
      expect(a).toHaveLength(3);
    } finally {
      vi.useRealTimers();
    }
  });

  // Mutation: drop the live `state.json` merge → a run of a live thread has no log, no now line and no step details.
  it("a run's state.json read live gives the page its history and step details; the same read again sends nothing", () => {
    const file = readFileSync(fileURLToPath(new URL('./fixtures/workflow-run/state.json', import.meta.url)), 'utf8');
    const doc = JSON.parse(file) as { state: Record<string, unknown> };
    doc.state.run_id = 'wf_1';
    doc.state.revision = 1;
    const state = parseRunState(JSON.stringify(doc))!;
    const conv = new Conversation({ ...INFO, mode: 'work' }, { needles: () => [] });
    conv.seed([], [], null);
    const a: ConversationEvent[] = [];
    conv.attach((e) => a.push(e), HOST);
    conv.applyLine(run(3, 'active'));
    conv.runState(state, 2_000);
    conv.flush();
    const page = fold(a);
    const entry = runOf(page);
    expect(entry?.kind === 'workflow' && entry.history?.length).toBe(state.history.length);
    expect(entry?.kind === 'workflow' && entry.revision).toBe(3);
    expect(entry?.kind === 'workflow' && entry.phases.map((phase) => phase.detail)).toEqual([
      state.details.get('Plan'),
      state.details.get('Research'),
    ]);
    const count = a.length;
    conv.runState(state, 2_000);
    conv.flush();
    expect(a).toHaveLength(count);
    // A later update keeps what the file gave.
    conv.applyLine(run(4, 'active'));
    conv.flush();
    const later = runOf(fold(a));
    expect(later?.kind === 'workflow' && later.history?.length).toBe(state.history.length);
    expect(later?.kind === 'workflow' && later.phases[1]?.detail).toBe(state.details.get('Research'));
  });
});

describe('a live thought keeps its time on the page', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const T = 1_794_411_000_000;
  const thought = (text: string, at: number): UpdateLine => ({
    xai: false,
    update: { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text } },
    meta: { agentTimestampMs: at },
  });

  // Mutation: drop `endTs` from the `text` op → red (the page's thought ends where it started).
  it('three deltas at t, t+2 s, t+5 s reach the page timed, 5 s long, also through a held-back key', () => {
    const key = 'exa-KEY-0123456789abcdef';
    const conv = new Conversation(INFO, { needles: () => [key] });
    conv.seed([], [], null);
    const events: ConversationEvent[] = [];
    conv.attach((e) => events.push(e), HOST);
    conv.userPrompt('go');
    conv.applyLine(thought('First ', T));
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    conv.applyLine(thought('then exa-KEY', T + 2_000));
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    conv.applyLine(thought('-0123456789abcdef', T + 5_000));
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    const entry = fold(events).entries.find((e) => e.kind === 'thinking');
    expect(entry).toMatchObject({ kind: 'thinking', ts: T, endTs: T + 5_000, timed: true });
    expect(entry!.kind === 'thinking' && entry!.text).not.toContain(key);
  });

  // Mutation: hold back a fixed `longest needle − 1` characters → red (the thought reads '' until the next entry).
  it('a short thought shows its text while the model streams a tool call, with a 43-character token', () => {
    const token = 'Zq3'.repeat(14) + 'x';
    const conv = new Conversation(INFO, { needles: () => [token] });
    conv.seed([], [], null);
    const events: ConversationEvent[] = [];
    conv.attach((e) => events.push(e), HOST);
    conv.userPrompt('go');
    for (const [n, part] of ['Now', ' write', ' the', ' workflow', ' file', '.'].entries()) {
      conv.applyLine(thought(part, T + n * 20));
    }
    for (let n = 0; n < 10; n += 1) {
      conv.applyLine({
        xai: true,
        update: { sessionUpdate: 'tool_call_delta_chunk', tool_call_id: 't1', tool_index: 0, arguments_delta: 'x' },
        meta: undefined,
      });
    }
    vi.advanceTimersByTime(CHAT_FLUSH_MS);
    expect(fold(events).entries.find((e) => e.kind === 'thinking')).toMatchObject({
      text: 'Now write the workflow file.',
      ts: T,
      endTs: T + 100,
    });
  });

  it('merged text ops keep the later stamp', () => {
    expect(
      mergeTextOps([
        { op: 'text', index: 0, append: 'a', endTs: 1 },
        { op: 'text', index: 0, append: 'b', endTs: 2 },
        { op: 'text', index: 0, append: 'c' },
      ]),
    ).toEqual([{ op: 'text', index: 0, append: 'abc', endTs: 2 }]);
  });
});
