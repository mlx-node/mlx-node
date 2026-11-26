/**
 * A call a hook denied, in the shared reducer: the agent fails the call with "Hook denied: <reason>" and sends its
 * words on the verdict as a `hook_annotation` of kind `tool_outcome` (mlx-agent v0.3.4 `session/acp_session/hooks.rs`
 * `block_tool_call`). It sends those words at once and the call's lines through its buffer, so either can come first:
 * they pair in order, and the call's own row carries the words as `blocked`. Live and stored alike.
 */

import { describe, expect, it } from 'vite-plus/test';

import { HOOK_DENIED_TEXT, parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry, TurnMetric } from '../src/agent-sessions/types.js';
import { applyOp } from '../src/chat/conversation.js';
import { redactEntry } from '../src/chat/redact.js';

const SID = '01a20000-0000-7000-8000-0000000000f2';
const T0 = 1_791_643_279_000;
const at = (n: number): number => T0 + n * 1_000;

type Update = Record<string, unknown>;
type Tool = Extract<TranscriptEntry, { kind: 'tool' }>;

interface Line {
  xai: boolean;
  update: Update;
  /** The typed prompt: the host pushes it live. */
  prompt?: string;
}

const prompt = (text: string, promptIndex = 0): Line => ({
  xai: false,
  update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text }, _meta: { promptIndex } },
  prompt: text,
});
const call = (id: string, command: string): Line => ({
  xai: false,
  update: {
    sessionUpdate: 'tool_call',
    toolCallId: id,
    title: 'run_terminal_command',
    rawInput: { command },
    _meta: { 'x.ai/tool': { version: 1, name: 'run_terminal_command', kind: 'execute', namespace: 'grok_build' } },
  },
});
const denied = (id: string, reason: string): Line => ({
  xai: false,
  update: {
    sessionUpdate: 'tool_call_update',
    toolCallId: id,
    status: 'failed',
    content: [{ type: 'content', content: { type: 'text', text: `${HOOK_DENIED_TEXT}${reason}` } }],
  },
});
const verdict = (message: string): Line => ({
  xai: true,
  update: { sessionUpdate: 'hook_annotation', message, kind: 'tool_outcome' },
});
const text = (body: string): Line => ({
  xai: false,
  update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: body } },
});
const done = (): Line => ({
  xai: true,
  update: { sessionUpdate: 'turn_completed', prompt_id: 'p-1', stop_reason: 'end_turn' },
});

const RM = '`run_terminal_command` blocked by global/no-rm: deleting build output is not allowed here';
const PUSH = '`run_terminal_command` blocked by global/no-push: pushes need a review';

function diskText(lines: readonly Line[]): string {
  return lines
    .map((line, n) =>
      JSON.stringify({
        timestamp: Math.floor(at(n) / 1000),
        method: line.xai ? '_x.ai/session/update' : 'session/update',
        params: { sessionId: SID, update: line.update, _meta: { eventId: `${SID}-${n}`, agentTimestampMs: at(n) } },
      }),
    )
    .map((l) => `${l}\n`)
    .join('');
}

function live(lines: readonly Line[]): TranscriptEntry[] {
  const reducer = new TranscriptReducer();
  const entries: TranscriptEntry[] = [];
  const turns: TurnMetric[] = [];
  lines.forEach((line, n) => {
    const ops =
      line.prompt !== undefined
        ? reducer.userPrompt(line.prompt, [], at(n))
        : reducer.apply({ xai: line.xai, update: line.update, meta: { agentTimestampMs: at(n) } });
    for (const op of ops) applyOp(entries, turns, op);
  });
  return entries;
}

/** The stored list, after checking the live ops give the same one (a live prompt has no index). */
function both(lines: readonly Line[]): TranscriptEntry[] {
  const stored = parseUpdates(diskText(lines)).transcript;
  const plain = (entries: readonly TranscriptEntry[]): unknown[] =>
    entries.map((entry) => (entry.kind === 'user' ? { ...entry, promptIndex: null } : entry));
  expect(plain(live(lines))).toEqual(plain(stored));
  return stored;
}

const tools = (entries: readonly TranscriptEntry[]): Tool[] =>
  entries.filter((entry): entry is Tool => entry.kind === 'tool');
const kinds = (entries: readonly TranscriptEntry[]): string[] => entries.map((entry) => entry.kind);

describe('a call a hook denied', () => {
  it('the failed update, then the verdict: the call carries the words, and no notice is made', () => {
    const entries = both([prompt('Clean'), call('c1', 'rm -rf build'), denied('c1', 'no'), verdict(RM), done()]);
    expect(kinds(entries)).toEqual(['user', 'tool', 'turn-end']);
    expect(tools(entries)[0]).toMatchObject({ id: 'c1', status: 'failed', blocked: RM });
  });

  it('the verdict before the failed update (and before the call itself) gives the same list', () => {
    const late = both([prompt('Clean'), verdict(RM), call('c1', 'rm -rf build'), denied('c1', 'no'), done()]);
    const first = both([prompt('Clean'), call('c1', 'rm -rf build'), verdict(RM), denied('c1', 'no'), done()]);
    expect(kinds(late)).toEqual(['user', 'tool', 'turn-end']);
    expect(tools(late)[0]).toMatchObject({ id: 'c1', blocked: RM });
    expect(tools(first)[0]).toMatchObject({ id: 'c1', blocked: RM });
  });

  it('live, the verdict lands on the row the page already has', () => {
    const reducer = new TranscriptReducer();
    const entries: TranscriptEntry[] = [];
    const turns: TurnMetric[] = [];
    const step = (line: Line, n: number) => {
      const ops =
        line.prompt !== undefined
          ? reducer.userPrompt(line.prompt, [], at(n))
          : reducer.apply({ xai: line.xai, update: line.update, meta: { agentTimestampMs: at(n) } });
      for (const op of ops) applyOp(entries, turns, op);
      return ops;
    };
    step(prompt('Clean'), 0);
    step(call('c1', 'rm -rf build'), 1);
    step(denied('c1', 'no'), 2);
    expect(tools(entries)[0]).not.toHaveProperty('blocked');
    expect(step(verdict(RM), 3)).toEqual([{ op: 'set', index: 1, entry: expect.objectContaining({ blocked: RM }) }]);
    expect(tools(entries)[0]).toMatchObject({ blocked: RM });
  });

  it('two denies in one turn pair in order, whatever comes first', () => {
    const lines = [
      prompt('Clean and push'),
      call('c1', 'rm -rf build'),
      call('c2', 'git push'),
      verdict(RM),
      denied('c1', 'no'),
      verdict(PUSH),
      denied('c2', 'review'),
      done(),
    ];
    expect(tools(both(lines)).map((t) => [t.id, t.blocked])).toEqual([
      ['c1', RM],
      ['c2', PUSH],
    ]);
    const swapped = [
      prompt('Clean and push'),
      call('c1', 'rm -rf build'),
      call('c2', 'git push'),
      denied('c1', 'no'),
      denied('c2', 'review'),
      verdict(RM),
      verdict(PUSH),
      done(),
    ];
    expect(tools(both(swapped)).map((t) => [t.id, t.blocked])).toEqual([
      ['c1', RM],
      ['c2', PUSH],
    ]);
  });

  it('a verdict no call took is a note at the turn end; a denied call with no verdict stays a failed call', () => {
    const lone = both([prompt('Clean'), text('ok'), verdict(RM), done()]);
    expect(kinds(lone)).toEqual(['user', 'assistant', 'notice', 'turn-end']);
    expect(lone[2]).toMatchObject({ kind: 'notice', code: 'hook-note', text: RM, ts: at(2) });
    const quiet = both([prompt('Clean'), call('c1', 'rm -rf build'), denied('c1', 'no'), done()]);
    expect(tools(quiet)[0]).not.toHaveProperty('blocked');
    expect(tools(quiet)[0]).toMatchObject({ status: 'failed', output: `${HOOK_DENIED_TEXT}no` });
  });

  it('a verdict never pairs with a call of the next turn; a turn with no end gives it up at the next prompt', () => {
    const lines = [
      prompt('one'),
      verdict(RM),
      done(),
      prompt('two', 1),
      call('c2', 'rm -rf build'),
      denied('c2', 'no'),
      done(),
    ];
    const entries = both(lines);
    expect(tools(entries)[0]).not.toHaveProperty('blocked');
    expect(entries.filter((e) => e.kind === 'notice')).toEqual([
      expect.objectContaining({ code: 'hook-note', text: RM }),
    ]);
    const noEnd = parseUpdates(
      diskText([prompt('one'), verdict(RM), prompt('two', 1), call('c2', 'x'), denied('c2', 'no')]),
    );
    expect(kinds(noEnd.transcript)).toEqual(['user', 'notice', 'user', 'tool']);
    expect(tools(noEnd.transcript)[0]).not.toHaveProperty('blocked');
  });

  it('a failure with other words is no deny; a later update of a denied call does not queue it twice', () => {
    const other: Line = {
      xai: false,
      update: {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'c1',
        status: 'failed',
        content: [{ type: 'content', content: { type: 'text', text: 'Denied by permission policy' } }],
      },
    };
    const entries = both([prompt('Clean'), call('c1', 'rm -rf build'), other, verdict(RM), done()]);
    expect(tools(entries)[0]).not.toHaveProperty('blocked');
    const twice = both([
      prompt('Clean'),
      call('c1', 'rm -rf build'),
      denied('c1', 'no'),
      denied('c1', 'no'),
      call('c2', 'git push'),
      denied('c2', 'review'),
      verdict(RM),
      verdict(PUSH),
      done(),
    ]);
    expect(tools(twice).map((t) => [t.id, t.blocked])).toEqual([
      ['c1', RM],
      ['c2', PUSH],
    ]);
  });

  it('the words are cut at 600 characters; a cut end that could start a key is redacted', () => {
    const long = `${'w'.repeat(596)}sk-live-123456`;
    const entries = both([prompt('Clean'), call('c1', 'rm'), denied('c1', 'no'), verdict(long), done()]);
    const row = tools(entries)[0]!;
    expect(row.blocked).toBe(`${long.slice(0, 600)}…`);
    expect((redactEntry(row, ['sk-live-123456']) as Tool).blocked).toBe(`${'w'.repeat(596)}…`);
  });
});
