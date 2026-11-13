/**
 * One conversation's stream: `sync` first, `seq` without gaps across listeners, the 33 ms op batching with merged
 * text, hold-back redaction of a secret split over chunks (current and retired needles), and a fresh `sync` after a
 * hot → cold switch. The page's reducer refuses a gap.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import type { UpdateLine } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { CHAT_FLUSH_MS, Conversation, mergeTextOps } from '../src/chat/conversation.js';
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
const HOST = { turn: null, interactions: [] };

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
    });
    expect(running[0]).not.toHaveProperty('lastEnd');
    expect(fold(running).lastEnd).toBeNull();
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
