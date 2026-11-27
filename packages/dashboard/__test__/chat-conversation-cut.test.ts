/**
 * `Conversation.cut`: a rewind's cut of the host's own entries. It keeps the entries before the index and the metrics
 * of the turns whose end is kept, sends one `sync` with the next `seq`, keeps redaction, and a later live line
 * indexes after the cut.
 */

import { describe, expect, it } from 'vite-plus/test';

import type { TranscriptEntry, TurnMetric } from '../src/agent-sessions/types.js';
import { Conversation, type HostView } from '../src/chat/conversation.js';
import type { ConversationEvent, ConversationInfo } from '../src/chat/types.js';

const SECRET = 'sk-SECRET-0123456789';

const info: ConversationInfo = {
  id: 'c-cut',
  mode: 'chat',
  title: 'Cut',
  cwd: null,
  modelId: 'm',
  createdAt: 1,
  updatedAt: 1,
  hot: true,
  permissionMode: 'ask',
};

const host: HostView = {
  turn: null,
  interactions: [],
  compaction: null,
  hookRun: null,
  recap: null,
  runs: [],
  live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
  commands: null,
};

function metric(promptId: string): TurnMetric {
  return {
    promptId,
    ts: 1,
    inputTokens: 1,
    outputTokens: 1,
    cachedTokens: 0,
    reasoningTokens: 0,
    elapsedMs: 1,
    ttftMs: null,
    prefillMs: null,
    prefillTokens: null,
    prefillCalls: null,
    modelCalls: 1,
    decodeTps: null,
    model: 'm',
  };
}

const entries: TranscriptEntry[] = [
  { kind: 'user', ts: 1, text: `first ${SECRET}`, promptIndex: 0 },
  { kind: 'assistant', ts: 2, text: 'one' },
  { kind: 'turn-end', ts: 3, stopReason: 'end_turn' },
  { kind: 'user', ts: 4, text: 'second', promptIndex: 1 },
  { kind: 'assistant', ts: 5, text: 'two' },
  { kind: 'turn-end', ts: 6, stopReason: 'end_turn' },
];

describe('Conversation.cut', () => {
  it('keeps [0, i) and the first K metrics; one sync with the next seq; redacted; live lines index after it', () => {
    const conversation = new Conversation(info, { needles: () => [SECRET], flushMs: 1 });
    conversation.seed(entries, [metric('p0'), metric('p1')], null, 900);
    const events: ConversationEvent[] = [];
    conversation.attach((event) => events.push(event), host);
    const before = conversation.baseVersion;
    const kept = conversation.cut(3, host);
    expect(kept.map((e) => e.kind)).toEqual(['user', 'assistant', 'turn-end']);
    expect(kept[0]).toMatchObject({ text: `first ${SECRET}` });
    expect(conversation.baseVersion).toBeGreaterThan(before);
    expect(events.map((e) => [e.type, e.seq])).toEqual([
      ['sync', 0],
      ['sync', 1],
    ]);
    const sync = events[1] as Extract<ConversationEvent, { type: 'sync' }>;
    expect(sync.entries).toHaveLength(3);
    expect(JSON.stringify(sync.entries)).not.toContain(SECRET);
    expect(sync.turns.map((t) => t.promptId)).toEqual(['p0']);
    expect(sync.contextUsed).toBeUndefined();
    expect(sync.lastEnd).toBeUndefined();
    expect(conversation.shownEntries()).toHaveLength(3);

    conversation.userPrompt('third');
    conversation.flush();
    const ops = events[2] as Extract<ConversationEvent, { type: 'ops' }>;
    expect(ops).toMatchObject({ type: 'ops', seq: 2 });
    expect(ops.ops[0]).toMatchObject({ op: 'push', entry: { kind: 'user', text: 'third' } });
    expect(conversation.shownEntries()).toHaveLength(4);

    conversation.contextRead(123);
    conversation.flush();
    expect(events[3]).toMatchObject({ type: 'ops', ops: [{ op: 'context', used: 123 }] });
    expect(conversation.contextUsed()).toBe(123);
  });

  it('a cut to the start keeps nothing; with no base it is a no-op', () => {
    const conversation = new Conversation(info, { needles: () => [], flushMs: 1 });
    expect(conversation.cut(0, host)).toEqual([]);
    conversation.seed(entries, [metric('p0'), metric('p1')], null);
    expect(conversation.cut(0, host)).toEqual([]);
    expect(conversation.shownEntries()).toEqual([]);
  });

  it('a rewind end goes to the page after the cut', () => {
    const conversation = new Conversation(info, { needles: () => [], flushMs: 1 });
    conversation.seed(entries, [], null);
    const events: ConversationEvent[] = [];
    conversation.attach((event) => events.push(event), host);
    conversation.cut(3, host);
    conversation.rewound({ outcome: 'done', mode: 'conversation', restored: 0, entry: 3 });
    expect(events.map((e) => e.type)).toEqual(['sync', 'sync', 'rewind']);
    expect(events[2]).toEqual({
      type: 'rewind',
      seq: 2,
      n: 1,
      end: { outcome: 'done', mode: 'conversation', restored: 0, entry: 3 },
    });
  });

  // Mutation: leave `rewindEnd` out of the sync → a page that resynced after the cut never sees the end.
  it('the sync carries the last rewind end with its number, so a page that resyncs after the cut still gets it', () => {
    const conversation = new Conversation(info, { needles: () => [], flushMs: 1 });
    conversation.seed(entries, [], null);
    const events: ConversationEvent[] = [];
    conversation.attach((event) => events.push(event), host);
    const syncs = () => events.filter((e) => e.type === 'sync');
    expect(syncs()[0]).not.toHaveProperty('rewindEnd');
    conversation.cut(3, host);
    conversation.rewound({ outcome: 'done', mode: 'conversation', restored: 0, entry: 3 });
    conversation.resync(host);
    expect(syncs().at(-1)).toMatchObject({
      rewindEnd: { n: 1, end: { outcome: 'done', mode: 'conversation', restored: 0, entry: 3 } },
    });
    conversation.rewound({ outcome: 'stopped' });
    conversation.resync(host);
    expect(events.filter((e) => e.type === 'rewind').map((e) => e.n)).toEqual([1, 2]);
    expect(syncs().at(-1)).toMatchObject({ rewindEnd: { n: 2, end: { outcome: 'stopped' } } });
  });
});
