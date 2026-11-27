/**
 * The Agent view's rules: where it stands at a width, how a watch's answer and the stream's ops build its list, what
 * stands where the message field goes, the agent before and after it, its permission counts, the copied lines of a
 * retry, its context words, and the message size words.
 */

import { describe, expect, it } from 'vite-plus/test';

import { COUNT_FROM_BYTES, canSendMessage, kbWords, messageBytes, messageCount } from '../ui/src/lib/agent-message.js';
import {
  AGENT_VIEW_COLUMN_WIDTH,
  AGENT_VIEW_DRAWER_WIDTH,
  type AgentFeed,
  agentField,
  agentMarkCounts,
  agentViewPlace,
  contextWords,
  feedAnswer,
  feedEvent,
  feedFailed,
  feedStart,
  markCountWords,
  neighbourAgent,
  splitCopied,
} from '../ui/src/lib/agent-view.js';
import type { AgentRow } from '../ui/src/lib/agents.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import {
  type ConversationEvent,
  type HelperProgress,
  type PermissionMark,
  SUBAGENT_MESSAGE_MAX_BYTES,
} from '../ui/src/lib/chat-types.js';
import { retryTitleOf } from '../ui/src/lib/sessions.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';

const WINDOW = 'window-a';

function text(value: string, copied = false): TranscriptEntry {
  return { kind: 'assistant', ts: 1, text: value, ...(copied ? { copied: true } : {}) };
}

function user(value: string, copied = false): TranscriptEntry {
  return { kind: 'user', ts: 1, text: value, promptIndex: 0, ...(copied ? { copied: true } : {}) };
}

const words = (entries: readonly TranscriptEntry[]): string[] =>
  entries.map((entry) => ('text' in entry && typeof entry.text === 'string' ? entry.text : entry.kind));

const push = (entry: TranscriptEntry) => ({ op: 'push' as const, entry });

function row(childId: string, attempts: string[] = [childId], run: string | null = null): AgentRow {
  return {
    childId,
    label: childId,
    run,
    phase: null,
    attempts,
    at: [],
    roster: null,
    rosterTs: null,
    continues: null,
    tokens: null,
    durationMs: 0,
  };
}

type EntriesEvent = Extract<ConversationEvent, { type: 'agent-entries' }>;

function event(fields: Partial<EntriesEvent>): EntriesEvent {
  return { type: 'agent-entries', seq: 1, window: WINDOW, childId: 'c1', ops: [], ...fields };
}

function ready(entries: TranscriptEntry[], live = true): AgentFeed {
  return feedAnswer(feedStart(null, 'c1', 1), 1, { childId: 'c1', attempts: ['c1'], entries, live });
}

describe('agentViewPlace', () => {
  // Mutation: `width >= AGENT_VIEW_COLUMN_WIDTH` to `>` puts the view in a drawer at exactly 1448 px.
  it('stands in the column from 1448 px, a drawer from 1296 px, else full', () => {
    expect(AGENT_VIEW_COLUMN_WIDTH).toBe(1448);
    expect(AGENT_VIEW_DRAWER_WIDTH).toBe(1296);
    expect(agentViewPlace(1448)).toBe('column');
    expect(agentViewPlace(1447)).toBe('drawer');
    expect(agentViewPlace(1296)).toBe('drawer');
    expect(agentViewPlace(1295)).toBe('full');
  });

  // Mutation: dropping `width === 0 ||` opens a page with no layout yet in a full drawer.
  it('counts a width of 0 as wide', () => {
    expect(agentViewPlace(0)).toBe('column');
  });
});

describe('the feed', () => {
  // Mutation: `request !== feed.request` dropped lets an older watch's answer replace the newer one's list.
  it("drops an older watch's answer", () => {
    const first = feedStart(null, 'c1', 1);
    const second = feedStart(first, 'c1', 2);
    const old = feedAnswer(second, 1, { childId: 'c1', attempts: ['c1'], entries: [text('old')], live: true });
    expect(old.status).toBe('loading');
    expect(old.entries).toEqual([]);
    const fresh = feedAnswer(second, 2, { childId: 'c1', attempts: ['c0', 'c1'], entries: [text('new')], live: true });
    expect(fresh.status).toBe('ready');
    expect(words(fresh.entries)).toEqual(['new']);
    expect(fresh.attempts).toEqual(['c0', 'c1']);
  });

  // Mutation: `entries: same ? previous.entries : []` to `previous?.entries ?? []` shows one attempt's lines under
  // another's head while the new watch loads.
  it('keeps the list over a watch of the same attempt, not of another', () => {
    const shown = ready([text('kept')]);
    expect(feedStart(shown, 'c1', 2).entries).toHaveLength(1);
    expect(feedStart(shown, 'c2', 2).entries).toEqual([]);
    expect(feedStart(shown, 'c1', 2).live).toBe(false);
  });

  // Mutation: `|| !feed.live` dropped applies ops that belong to the watch a new one replaces.
  it('applies no op before the answer of the watch that replaces it', () => {
    const loading = feedStart(ready([text('kept')]), 'c1', 2);
    const after = feedEvent(loading, event({ ops: [push(text('early'))] }), WINDOW);
    expect(after).toBe(loading);
  });

  // Mutation: `event.window !== window` dropped applies another window's watch ops twice in this one.
  it("applies only this window's ops of the attempt shown", () => {
    const feed = ready([]);
    const op = push(text('line'));
    expect(feedEvent(feed, event({ window: 'other', ops: [op] }), WINDOW)).toBe(feed);
    expect(feedEvent(feed, event({ childId: 'c2', ops: [op] }), WINDOW)).toBe(feed);
    expect(feedEvent(feed, event({ ops: [op] }), WINDOW).entries).toHaveLength(1);
  });

  // Mutation: `live: false` to `live: feed.live` on the stored list lets later ops add to the file's list twice.
  it('takes the stored list at the end and stops', () => {
    const feed = ready([text('live')]);
    const stored = feedEvent(feed, event({ stored: [text('stored'), user('sent')] }), WINDOW);
    expect(stored.entries).toHaveLength(2);
    expect(stored.live).toBe(false);
    const late = feedEvent(stored, event({ ops: [push(text('late'))] }), WINDOW);
    expect(late).toBe(stored);
  });

  // Mutation: `request !== feed.request ? feed :` dropped marks the newer watch failed when an older one fails.
  it('fails only the newest watch', () => {
    const second = feedStart(feedStart(null, 'c1', 1), 'c1', 2);
    expect(feedFailed(second, 1)).toBe(second);
    expect(feedFailed(second, 2).status).toBe('failed');
  });
});

describe('agentField', () => {
  const held = { stopping: false, messages: 'on' as const };

  // Mutation: the `input.workflow` line moved after the address checks tells of a missing address, not of the build.
  it('gives a workflow agent a line, not a field', () => {
    const note = { kind: 'note', text: CHAT.agent.noMessages };
    expect(agentField({ workflow: true, state: 'running', held, waits: false })).toEqual(note);
    expect(agentField({ workflow: true, state: 'running', held: null, waits: false })).toEqual(note);
    expect(
      agentField({ workflow: true, state: 'running', held: { stopping: false, messages: 'no-address' }, waits: false }),
    ).toEqual(note);
  });

  // Mutation: `held?.stopping === true` dropped keeps the field of an agent a stop was sent to.
  // Mutation: the stopping line gives the ended note → the field says "It ended." while the stop is on its way.
  it('says an ended or stopping agent cannot get a message, each in its own words', () => {
    const ended = { kind: 'note', text: CHAT.agent.ended };
    const stopping = { kind: 'note', text: CHAT.agent.stoppingNote };
    expect(agentField({ workflow: false, state: 'done', held, waits: false })).toEqual(ended);
    expect(agentField({ workflow: false, state: 'running', held: { ...held, stopping: true }, waits: false })).toEqual(
      stopping,
    );
    expect(agentField({ workflow: true, state: 'failed', held: null, waits: false })).toEqual(ended);
  });

  // Mutation: `held === null ||` dropped crashes on, or gives a field to, an agent the host does not hold.
  it('says why messages are off', () => {
    expect(agentField({ workflow: false, state: 'running', held: null, waits: false })).toEqual({
      kind: 'note',
      text: CHAT.activity.messageNoAddress,
    });
    expect(
      agentField({ workflow: false, state: 'running', held: { stopping: false, messages: 'off' }, waits: false }),
    ).toEqual({ kind: 'note', text: CHAT.activity.messageOff });
  });

  // Mutation: `input.waits ? ... : null` to `null` drops the notice of an agent that waits on a request.
  it('gives a running subagent the field, with a notice while it waits', () => {
    expect(agentField({ workflow: false, state: 'running', held, waits: false })).toEqual({
      kind: 'field',
      notice: null,
    });
    expect(agentField({ workflow: false, state: 'running', held, waits: true })).toEqual({
      kind: 'field',
      notice: CHAT.activity.messageWaits,
    });
  });
});

describe('neighbourAgent', () => {
  const list = [row('a'), row('b2', ['b1', 'b2']), row('c')];

  // Mutation: `agentRowOf(list, childId)` to `list.find((r) => r.childId === childId)` loses the place of a row
  // opened by its first attempt's id.
  it('steps from a row named by any of its attempts', () => {
    expect(neighbourAgent(list, 'b1', 1)?.childId).toBe('c');
    expect(neighbourAgent(list, 'b1', -1)?.childId).toBe('a');
  });

  // Mutation: `?? null` to `?? list[0]` wraps around at the end.
  it('stops at the ends and off the list', () => {
    expect(neighbourAgent(list, 'c', 1)).toBeNull();
    expect(neighbourAgent(list, 'a', -1)).toBeNull();
    expect(neighbourAgent(list, 'z', 1)).toBeNull();
  });
});

describe('permission counts', () => {
  const marks: PermissionMark[] = [
    { kind: 'allowed', at: 1, mode: 'auto', toolCallId: 't1', access: 'bash', card: null, child: 'b1' },
    { kind: 'allowed', at: 2, mode: 'auto', toolCallId: 't2', access: 'edit', card: null, child: 'b2' },
    { kind: 'allowed', at: 3, mode: 'auto', toolCallId: 't3', access: 'bash', card: null },
    { kind: 'denied', at: 4, toolCallId: 't4', access: 'bash', by: 'note', card: null, child: 'b2' },
    { kind: 'denied', at: 5, toolCallId: 't5', access: 'bash', by: 'note', card: null, child: 'other' },
  ];

  // Mutation: `attempts.includes(mark.child)` to `mark.child === attempts.at(-1)` drops the first attempt's calls.
  it("counts the calls of every attempt of the row, not the conversation's own", () => {
    expect(agentMarkCounts(marks, ['b1', 'b2'])).toEqual({ allowed: 2, denied: 1 });
    expect(agentMarkCounts(undefined, ['b1'])).toEqual({ allowed: 0, denied: 0 });
  });

  // Mutation: the `allowed > 0 && denied > 0` line dropped reads "2 allowed" with a denied call too.
  it('reads both counts, either one, or nothing', () => {
    expect(markCountWords({ allowed: 2, denied: 1 })).toBe('2 allowed, 1 denied');
    expect(markCountWords({ allowed: 2, denied: 0 })).toBe('2 allowed');
    expect(markCountWords({ allowed: 0, denied: 1 })).toBe('1 denied');
    expect(markCountWords({ allowed: 0, denied: 0 })).toBeNull();
  });
});

describe('a retry', () => {
  // Mutation: `entry.copied === true ? copied : own` to `entry.copied === true ? own : copied` shows the first
  // attempt's lines as this one's.
  it('keeps the copied lines apart from its own, in order', () => {
    const { copied, own } = splitCopied([
      user('first task', true),
      text('first try', true),
      user('Your final message did not satisfy the output contract: missing field'),
      text('second try'),
    ]);
    expect(words(copied)).toEqual(['first task', 'first try']);
    expect(words(own)).toEqual(['Your final message did not satisfy the output contract: missing field', 'second try']);
  });

  // Mutation: `line.startsWith(CONTRACT_RETRY_LEAD)` to `line.includes('output contract')` titles a plain prompt that
  // names the contract as a retry.
  it('titles the fix-up prompt by its reason', () => {
    expect(retryTitleOf('Your final message did not satisfy the output contract: missing field\nmore')).toBe(
      'Output contract retry: missing field',
    );
    expect(retryTitleOf('Check the output contract: it is fine')).toBeNull();
  });
});

describe('contextWords', () => {
  const tick = (contextWindow: number | null): HelperProgress => ({
    childId: 'c1',
    turns: 1,
    toolCalls: 1,
    tokens: 41_000,
    contextPct: 32,
    errors: 0,
    durationMs: 1,
    contextWindow,
    at: 1,
    now: null,
  });

  // Mutation: `progress.contextWindow <= 0` dropped reads "41K of 0 context" from a tick with a zero window.
  it('names the window when the tick has one, else the share', () => {
    expect(contextWords(tick(128_000))).toBe('41K of 128K context');
    expect(contextWords(tick(null))).toBe(fill(CHAT.agent.facts.contextPct, { pct: 32 }));
    expect(contextWords(tick(0))).toBe(fill(CHAT.agent.facts.contextPct, { pct: 32 }));
  });
});

describe('the message field', () => {
  // Mutation: `messageBytes` as `text.length` lets a 32 KB text of two-byte letters through.
  it('counts UTF-8 bytes as the agent does', () => {
    expect(messageBytes('é')).toBe(2);
    expect(canSendMessage('é'.repeat(SUBAGENT_MESSAGE_MAX_BYTES / 2))).toBe(true);
    expect(canSendMessage(`${'é'.repeat(SUBAGENT_MESSAGE_MAX_BYTES / 2)}a`)).toBe(false);
  });

  // Mutation: `text.trim() !== ''` to `text !== ''` sends a message of spaces.
  it('sends no text of only spaces', () => {
    expect(canSendMessage('  \n ')).toBe(false);
    expect(canSendMessage(' go ')).toBe(true);
  });

  // Mutation: `bytes < COUNT_FROM_BYTES` to `bytes <= ...` hides the count at exactly 28 KB.
  it('shows the count from 28 KB and the too-long words past 32 KB', () => {
    expect(messageCount('a'.repeat(COUNT_FROM_BYTES - 1))).toBeNull();
    expect(messageCount('a'.repeat(COUNT_FROM_BYTES))).toEqual({
      text: fill(CHAT.activity.messageCount, { n: '28.0' }),
      tooLong: false,
    });
    expect(messageCount('a'.repeat(SUBAGENT_MESSAGE_MAX_BYTES))?.tooLong).toBe(false);
    const past = messageCount('a'.repeat(SUBAGENT_MESSAGE_MAX_BYTES + 1));
    expect(past?.tooLong).toBe(true);
    expect(past?.text).toBe(fill(CHAT.activity.messageTooLong, { n: '32.1' }));
  });

  // Mutation: `Math.ceil` past the limit to `Math.floor` reads one byte past 32 KB as "32.0".
  it('never reads a text past the limit as 32.0', () => {
    expect(kbWords(SUBAGENT_MESSAGE_MAX_BYTES + 1)).toBe('32.1');
    expect(kbWords(SUBAGENT_MESSAGE_MAX_BYTES)).toBe('32.0');
    expect(kbWords(SUBAGENT_MESSAGE_MAX_BYTES - 1)).toBe('31.9');
  });
});
