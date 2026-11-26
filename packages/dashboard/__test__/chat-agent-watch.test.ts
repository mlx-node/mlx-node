/**
 * The Agent view's live data (`agent-watch.ts`): the seam between an agent's file and its kept live lines, the
 * redaction of both, the end swap, the window cap, and the kept lines' limits.
 */

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, type UpdateLine } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import {
  AGENT_LINES_BYTES,
  AGENT_WATCHES_PER_WINDOW,
  AgentWatches,
  type AgentEntriesEvent,
  type AgentWatchStart,
} from '../src/chat/agent/agent-watch.js';
import { applyOp } from '../src/chat/conversation.js';
import { REDACTED } from '../src/chat/redact.js';
import { until, type Json } from './helpers/chat-fake.js';

const KEY = 'marker-KEY-12345678';
const T0 = 1_795_674_158_000;

interface Line {
  n: number | string;
  update: Json;
  xai?: boolean;
}

function idOf(child: string, n: number | string): string {
  return typeof n === 'string' ? n : `${child}-${n}`;
}

function live(child: string, l: Line): UpdateLine {
  const n = typeof l.n === 'number' ? l.n : 0;
  return { xai: l.xai === true, update: l.update, meta: { eventId: idOf(child, l.n), agentTimestampMs: T0 + n } };
}

function fileOf(child: string, lines: readonly Line[]): string {
  return lines
    .map((l) => {
      const n = typeof l.n === 'number' ? l.n : 0;
      return `${JSON.stringify({
        timestamp: Math.floor((T0 + n) / 1000),
        method: l.xai === true ? '_x.ai/session/update' : 'session/update',
        params: { sessionId: child, update: l.update, _meta: { eventId: idOf(child, l.n), agentTimestampMs: T0 + n } },
      })}\n`;
    })
    .join('');
}

const prompt = (n: number, text = 'Find the flaky test'): Line => ({
  n,
  update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text } },
});
const thought = (n: number | string, text: string): Line => ({
  n,
  update: { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text } },
});
const message = (n: number | string, text: string): Line => ({
  n,
  update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text } },
});
const call = (n: number | string, id: string, title = 'read'): Line => ({
  n,
  update: { sessionUpdate: 'tool_call', toolCallId: id, title, status: 'pending' },
});
const callUpdate = (n: number | string, id: string, fields: Json): Line => ({
  n,
  update: { sessionUpdate: 'tool_call_update', toolCallId: id, ...fields },
});
const hook = (n: number): Line => ({
  n,
  xai: true,
  update: { sessionUpdate: 'hook_execution', event_name: 'pre_tool_use', runs: [] },
});
const turnEnd = (n: number): Line => ({
  n,
  xai: true,
  update: { sessionUpdate: 'turn_completed', stop_reason: 'end_turn' },
});

interface Rig {
  watches: AgentWatches;
  files: Map<string, string>;
  events: AgentEntriesEvent[];
  needles: string[];
  running: Set<string>;
  feed(child: string, lines: readonly Line[]): void;
  start(child: string, window?: string): Promise<AgentWatchStart | 'too-many'>;
}

function rig(timings: { keepMs?: number; swapEveryMs?: number; swapMaxMs?: number } = {}): Rig {
  const files = new Map<string, string>();
  const events: AgentEntriesEvent[] = [];
  const needles: string[] = [];
  const running = new Set<string>();
  const watches = new AgentWatches({
    read: async (child) => files.get(child) ?? '',
    needles: () => needles,
    send: (_sessionId, event) => events.push(event),
    timings: { keepMs: 60, swapEveryMs: 10, swapMaxMs: 400, flushMs: 1, ...timings },
  });
  return {
    watches,
    files,
    events,
    needles,
    running,
    feed: (child, lines) => {
      for (const l of lines) watches.line('parent', child, live(child, l), running.has(child));
    },
    start: (child, window = 'w1') => watches.watch('parent', child, window, () => running.has(child)),
  };
}

function started(start: AgentWatchStart | 'too-many'): AgentWatchStart {
  if (start === 'too-many') throw new Error('refused');
  return start;
}

/** The list a page holds: the start, then every op of the window's events for that agent. */
function pageList(start: AgentWatchStart, events: readonly AgentEntriesEvent[], child: string): TranscriptEntry[] {
  let entries = [...start.entries];
  for (const event of events) {
    if (event.childId !== child) continue;
    if (event.stored !== undefined) entries = [...event.stored];
    for (const op of event.ops) applyOp(entries, [], op);
  }
  return entries;
}

function texts(entries: readonly TranscriptEntry[], kind: 'thinking' | 'assistant'): string[] {
  return entries.flatMap((e) => (e.kind === kind ? [e.text] : []));
}

describe('the seam between the file and the kept lines (T36)', () => {
  // Mutation: dedupe by id alone → the chunk before the merged line's id is applied again (a merged thought shows twice).
  it('a line the file holds is skipped, and so is a text chunk before the last ACP line it holds', async () => {
    const r = rig();
    r.running.add('c1');
    r.feed('c1', [
      prompt(0),
      thought(1, 'Hel'),
      thought(2, 'lo'),
      call(3, 't1'),
      callUpdate(4, 't1', { status: 'completed' }),
      call(5, 't2'),
    ]);
    // The store merges the two chunks into one line that takes the newest chunk's meta, and still holds `t2`.
    r.files.set(
      'c1',
      fileOf('c1', [prompt(0), thought(2, 'Hello'), call(3, 't1'), callUpdate(4, 't1', { status: 'completed' })]),
    );
    const start = started(await r.start('c1'));
    expect(start.live).toBe(true);
    expect(texts(start.entries, 'thinking')).toEqual(['Hello']);
    expect(start.entries.filter((e) => e.kind === 'tool').map((e) => e.kind === 'tool' && [e.id, e.status])).toEqual([
      ['t1', 'completed'],
      ['t2', 'pending'],
    ]);
  });

  // Mutation: drop every line before the last match → the unstored title of `t1` is lost.
  it('a line the file does not hold that is not a text chunk is kept, even before the last match', async () => {
    const r = rig();
    r.running.add('c1');
    r.feed('c1', [
      prompt(0),
      call(3, 't1'),
      callUpdate('live-only-1', 't1', { title: 'Read src/a.ts' }),
      call(4, 't2'),
      call(5, 't3'),
    ]);
    r.files.set('c1', fileOf('c1', [prompt(0), call(3, 't1'), call(4, 't2')]));
    const start = started(await r.start('c1'));
    const t1 = start.entries.find((e) => e.kind === 'tool' && e.id === 't1');
    expect(t1?.kind === 'tool' && t1.title).toBe('Read src/a.ts');
  });

  // Mutation: the seam at the last match of any kind → a thought the store still holds is skipped and lost.
  it('a text chunk before a stored xAI line stays while the store still holds it', async () => {
    const r = rig();
    r.running.add('c1');
    r.feed('c1', [prompt(0), thought(1, 'Thinking hard'), hook(2)]);
    // The xAI line is written at once; the thought waits for the next ACP line.
    r.files.set('c1', fileOf('c1', [prompt(0), hook(2)]));
    const start = started(await r.start('c1'));
    expect(texts(start.entries, 'thinking')).toEqual(['Thinking hard']);
  });

  it('later lines go to the window as ops on the list it started with', async () => {
    const r = rig();
    r.running.add('c1');
    r.files.set('c1', fileOf('c1', [prompt(0)]));
    const start = started(await r.start('c1'));
    r.feed('c1', [thought(1, 'Look'), thought(2, 'ing'), call(3, 't1')]);
    await until(() => r.events.length > 0);
    expect(r.events.every((e) => e.window === 'w1' && e.childId === 'c1')).toBe(true);
    const list = pageList(start, r.events, 'c1');
    expect(texts(list, 'thinking')).toEqual(['Looking']);
    expect(list.at(-1)?.kind).toBe('tool');
  });
});

describe('redaction of file and live lines (T37)', () => {
  // Mutation: redact only the live lines → the key in the file's message reaches the page.
  // Mutation: no hold-back on a live text → the key's first half goes out before its second half comes.
  it('a key in a file entry and a key split over two live chunks are both redacted', async () => {
    const r = rig();
    r.needles.push(KEY);
    r.running.add('c2');
    r.files.set('c2', fileOf('c2', [prompt(0), message(1, `The key is ${KEY}.`), call(2, 't1')]));
    const start = started(await r.start('c2'));
    expect(JSON.stringify(start.entries)).not.toContain(KEY);
    expect(texts(start.entries, 'assistant')).toEqual([`The key is ${REDACTED}.`]);

    r.feed('c2', [message(3, `Again ${KEY.slice(0, 8)}`)]);
    await new Promise((resolve) => setTimeout(resolve, 20));
    r.feed('c2', [message(4, `${KEY.slice(8)} done`)]);
    await until(() => texts(pageList(start, r.events, 'c2'), 'assistant').length === 2);
    await new Promise((resolve) => setTimeout(resolve, 20));
    for (const event of r.events) expect(JSON.stringify(event)).not.toContain(KEY.slice(0, 8));
    expect(texts(pageList(start, r.events, 'c2'), 'assistant')).toEqual([
      `The key is ${REDACTED}.`,
      `Again ${REDACTED} done`,
    ]);
  });
});

describe('the end swap and the window cap (T38)', () => {
  // Mutation: swap at the finish line → the list is the file of that moment, which lacks the last message.
  it('the end swap waits until the file holds the turn end, then sends the stored list', async () => {
    const r = rig();
    r.running.add('c3');
    const early = [prompt(0), thought(1, 'Checking')];
    r.files.set('c3', fileOf('c3', early));
    const start = started(await r.start('c3'));
    r.feed('c3', [call(2, 't1'), message(3, 'Final '), message(4, 'answer'), turnEnd(5)]);
    r.running.delete('c3');
    r.watches.ended('c3');
    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(r.events.some((e) => e.stored !== undefined)).toBe(false);

    const final = fileOf('c3', [...early, call(2, 't1'), message(4, 'Final answer'), turnEnd(5)]);
    r.files.set('c3', final);
    await until(() => r.events.some((e) => e.stored !== undefined));
    const stored = r.events.find((e) => e.stored !== undefined)!;
    expect(stored).toMatchObject({ window: 'w1', childId: 'c3', ops: [] });
    expect(stored.stored).toEqual(parseUpdates(final, { helper: true, sessionId: 'c3' }).transcript);
    expect(texts(pageList(start, r.events, 'c3'), 'assistant')).toEqual(['Final answer']);
    const after = r.events.length;
    r.feed('c3', [message(6, 'late')]);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(r.events.length).toBe(after);
  });

  // Mutation: the late list from the file alone → the message the file never got is gone.
  it('a file that never holds the turn end gives, past the wait, the file and the kept lines past the seam', async () => {
    const r = rig({ swapMaxMs: 50 });
    r.running.add('c3');
    r.files.set('c3', fileOf('c3', [prompt(0)]));
    r.feed('c3', [prompt(0)]);
    await r.start('c3');
    r.feed('c3', [message(1, 'Partial answer'), turnEnd(2)]);
    r.running.delete('c3');
    r.watches.ended('c3');
    await until(() => r.events.some((e) => e.stored !== undefined));
    const stored = r.events.find((e) => e.stored !== undefined)!.stored!;
    expect(texts(stored, 'assistant')).toEqual(['Partial answer']);
  });

  // Mutation: no cap → a third live view of one window is accepted.
  it(`a window holds ${AGENT_WATCHES_PER_WINDOW} live views; a third is refused, a view again or another window's is not`, async () => {
    const r = rig();
    for (const child of ['a', 'b', 'c']) {
      r.running.add(child);
      r.files.set(child, fileOf(child, [prompt(0)]));
    }
    expect(started(await r.start('a')).live).toBe(true);
    expect(started(await r.start('b')).live).toBe(true);
    expect(await r.start('c')).toBe('too-many');
    expect(started(await r.start('a')).live).toBe(true);
    expect(started(await r.start('c', 'w2')).live).toBe(true);
    r.files.set('d', fileOf('d', [prompt(0)]));
    expect(started(await r.start('d')).live).toBe(false);
    expect(r.watches.unwatch('b', 'w1')).toBe(true);
    expect(started(await r.start('c')).live).toBe(true);
  });

  // Mutation: register a watch for an agent that ended → its answer reads live.
  it('a view of an agent that ended is one read and gets nothing later', async () => {
    const r = rig();
    r.files.set('e', fileOf('e', [prompt(0), message(1, 'Done.'), turnEnd(2)]));
    const start = started(await r.start('e'));
    expect(start.live).toBe(false);
    expect(texts(start.entries, 'assistant')).toEqual(['Done.']);
    r.feed('e', [message(3, 'never shown')]);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(r.events).toEqual([]);
  });
});

describe('the kept lines', () => {
  // Mutation: keep the lines past the end for good → a view long after the end still shows a line the file lacks.
  it('stay for the keep time past the end, then go', async () => {
    const r = rig({ keepMs: 40 });
    r.running.add('k');
    r.files.set('k', fileOf('k', [prompt(0)]));
    r.feed('k', [prompt(0), message(1, 'Not stored yet')]);
    r.running.delete('k');
    r.watches.settle('parent', (child) => r.running.has(child));
    expect(texts(started(await r.start('k')).entries, 'assistant')).toEqual(['Not stored yet']);
    await new Promise((resolve) => setTimeout(resolve, 80));
    expect(texts(started(await r.start('k')).entries, 'assistant')).toEqual([]);
  });

  // Mutation: no byte limit → the oldest line is still kept.
  // Mutation: keep a line with no event id → a line the file never holds shows.
  it(`keep the newest ${AGENT_LINES_BYTES / 1024} KiB of lines, and skip a line with no event id`, async () => {
    const r = rig();
    r.running.add('big');
    r.files.set('big', fileOf('big', [prompt(0)]));
    const block = 'x'.repeat(64 * 1024);
    const lines: Line[] = [];
    for (let i = 1; i <= 10; i++) lines.push(call(i, `t${i}`, `${i} ${block}`));
    r.feed('big', lines);
    r.watches.line('parent', 'big', { xai: false, update: call(0, 'no-id').update, meta: undefined }, true);
    const ids = started(await r.start('big')).entries.flatMap((e) => (e.kind === 'tool' ? [e.id] : []));
    expect(ids).not.toContain('t1');
    expect(ids).not.toContain('no-id');
    expect(ids.at(-1)).toBe('t10');
  });

  // Mutation: keep the lines of an agent that does not run → a stray line shows in a later view.
  it('a line of an agent that does not run is not kept', async () => {
    const r = rig();
    r.files.set('n', fileOf('n', [prompt(0)]));
    r.feed('n', [message(1, 'stray')]);
    r.running.add('n');
    expect(texts(started(await r.start('n')).entries, 'assistant')).toEqual([]);
  });
});
