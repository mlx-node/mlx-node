import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import {
  filterRewindLines,
  MAX_FIELD_CHARS,
  parseUpdates,
  splitCompleteLines,
  truncateField,
} from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';

const SESSIONS = fileURLToPath(new URL('./fixtures/grok-sessions/home/sessions', import.meta.url));
const PARENT = `${SESSIONS}/%2FUsers%2Fdev%2Fproj/01a20000-0000-7000-8000-00000000000a/updates.jsonl`;
const MANUAL = `${SESSIONS}/%2FUsers%2Fdev%2Fother/01a20000-0000-7000-8000-000000000011/updates.jsonl`;
const SID = '01a20000-0000-7000-8000-00000000000a';

/** Real on-disk envelope shapes (grok `acp_envelope` / `xai_envelope`). */
function acp(update: object, n = 1): string {
  return JSON.stringify({
    timestamp: 1791643279,
    method: 'session/update',
    params: { sessionId: SID, update, _meta: { eventId: `${SID}-${n}`, agentTimestampMs: 1791643279000 + n } },
  });
}
function xai(update: object, n = 1): string {
  return JSON.stringify({
    timestamp: 1791643279,
    method: '_x.ai/session/update',
    params: { sessionId: SID, update, _meta: { eventId: `${SID}-${n}`, agentTimestampMs: 1791643279000 + n } },
  });
}
const user = (text: string, promptIndex?: number) =>
  acp({
    sessionUpdate: 'user_message_chunk',
    content: { type: 'text', text },
    ...(promptIndex === undefined ? {} : { _meta: { promptIndex } }),
  });
const agent = (text: string) => acp({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text } });
const rewind = (target: number) =>
  xai({ sessionUpdate: 'rewind_marker', target_prompt_index: target, created_at: '2026-10-20T09:00:00Z' });

/** Which test texts survived, in order. */
function texts(lines: string[]): string[] {
  return lines.map((l) => {
    const update = (JSON.parse(l) as { params?: { update?: { content?: { text?: string } } } }).params?.update;
    return update?.content?.text ?? '?';
  });
}

describe('splitCompleteLines', () => {
  it('drops a torn last line and blank lines, trims ASCII whitespace', () => {
    expect(splitCompleteLines('a\n\n  b \r\n{"torn":')).toEqual(['a', 'b']);
    expect(splitCompleteLines('{"only":"torn"')).toEqual([]);
    expect(splitCompleteLines('a\nb\n')).toEqual(['a', 'b']);
  });
});

describe('filterRewindLines (grok filter_rewind_lines)', () => {
  it('is a no-op without markers', () => {
    const lines = [user('p0'), agent('r0')];
    expect(filterRewindLines(lines)).toBe(lines);
  });

  it('removes the dead branch', () => {
    const out = filterRewindLines([user('first'), agent('resp1'), user('second'), agent('resp2'), rewind(1)]);
    expect(texts(out)).toEqual(['first', 'resp1']);
  });

  it('target is an ordinal of counted runs: unmarked prefix, then marked prompts', () => {
    const out = filterRewindLines([
      user('old0'),
      agent('A0'),
      user('old1'),
      agent('A1'),
      user('new2', 2),
      agent('A2'),
      user('new3', 3),
      rewind(2),
      user('after', 2),
    ]);
    expect(texts(out)).toEqual(['old0', 'A0', 'old1', 'A1', 'after']);
  });

  it('ignores unmarked phantom runs once markers exist', () => {
    const out = filterRewindLines([
      user('P0', 0),
      agent('A0'),
      user('!pwd'),
      user('P1', 1),
      agent('A1'),
      user('P2', 2),
      rewind(2),
      user('after', 2),
    ]);
    expect(texts(out)).toEqual(['P0', 'A0', '!pwd', 'P1', 'A1', 'after']);
  });

  it('keeps a malformed line as a run breaker', () => {
    const torn = '{ torn, unparseable jsonl line';
    const out = filterRewindLines([user('first'), agent('resp1'), torn, user('second'), agent('resp2'), rewind(1)]);
    expect(out).toEqual([user('first'), agent('resp1'), torn]);
  });

  it('stacks markers, and out-of-range targets keep everything', () => {
    const out = filterRewindLines([
      user('p1'),
      agent('r1'),
      user('p2'),
      agent('r2'),
      user('p3'),
      agent('r3'),
      rewind(2),
      user('p4'),
      agent('r4'),
      rewind(1),
      user('p5'),
      rewind(9),
    ]);
    expect(texts(out)).toEqual(['p1', 'r1', 'p5']);
  });

  it('a multi-block prompt is one run; hostTurn echoes and interjections break runs without counting', () => {
    const host = acp({
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: 'host' },
      _meta: { hostTurn: true },
    });
    const interjection = acp({
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: 'interj' },
      _meta: { interjection: true, modelId: 'm' },
    });
    const out = filterRewindLines([
      user('P0a', 0),
      user('P0b', 0),
      host,
      agent('A0'),
      interjection,
      user('P1', 1),
      rewind(1),
    ]);
    expect(texts(out)).toEqual(['P0a', 'P0b', 'host', 'A0', 'interj']);
  });

  it('honors a marker only under the xAI method', () => {
    const acpMarker = acp({ sessionUpdate: 'rewind_marker', target_prompt_index: 0 });
    const out = filterRewindLines([user('p0'), agent('r0'), acpMarker]);
    expect(out).toHaveLength(3);
  });
});

describe('truncateField', () => {
  it('caps without splitting a surrogate pair', () => {
    const text = `${'a'.repeat(9)}😀tail`;
    expect(truncateField(text, 10)).toEqual({ text: 'a'.repeat(9), truncated: true });
    expect(truncateField('short', 10)).toEqual({ text: 'short', truncated: false });
  });
});

describe('parseUpdates', () => {
  it('builds the parent transcript from real line shapes', () => {
    const parsed = parseUpdates(readFileSync(PARENT, 'utf8'));
    const kinds = parsed.transcript.map((e) => e.kind);
    expect(kinds).toEqual([
      'user',
      'thinking',
      'assistant',
      'tool',
      'plan',
      'tool',
      'tool',
      'subagent',
      'subagent',
      'turn-end',
      'user',
      'model',
      'assistant',
      'turn-end',
    ]);
    const [u0, thinking, assistant, todo, plan, read, orphan, sub1, sub2, end0, u1, model, , end1] =
      parsed.transcript as TranscriptEntry[];
    expect(u0).toMatchObject({ kind: 'user', text: 'Research this project', promptIndex: 0, ts: 1791643279022 });
    // Consecutive chunks of one kind merge.
    expect(thinking).toMatchObject({ text: 'Let me plan.' });
    expect(assistant).toMatchObject({ text: "I'll start by planning." });
    // tool_call + updates merged by toolCallId; the later title wins, x.ai/tool names it.
    expect(todo).toMatchObject({
      kind: 'tool',
      id: 'toolu_todo',
      name: 'todo_write',
      title: 'Updating plan',
      status: 'completed',
    });
    expect(todo.kind === 'tool' && JSON.parse(todo.input!)).toMatchObject({ variant: 'TodoWrite' });
    expect(todo.kind === 'tool' && JSON.parse(todo.output!)).toMatchObject({ type: 'Todo' });
    expect(plan).toMatchObject({ entries: [{ content: 'Map project structure', status: 'in_progress' }, {}] });
    // Content text is preferred over rawOutput and capped.
    expect(read).toMatchObject({ kind: 'tool', name: 'read_file', status: 'completed', truncated: true });
    expect(read.kind === 'tool' && read.output!.length).toBe(MAX_FIELD_CHARS);
    expect(read.kind === 'tool' && read.output!.startsWith('0→line 0')).toBe(true);
    // An update with no preceding tool_call still shows.
    expect(orphan).toMatchObject({ kind: 'tool', id: 'toolu_orphan', status: 'failed', name: 'tool' });
    expect(orphan.kind === 'tool' && orphan.output).toContain('Failed to parse arguments');
    expect(sub1).toMatchObject({
      kind: 'subagent',
      childId: '01a20000-0000-7000-8000-00000000000b',
      type: 'general-purpose',
      status: 'completed',
      durationMs: 5000,
      tokens: 1200,
    });
    expect(sub2).toMatchObject({
      childId: '01a20000-0000-7000-8000-00000000000c',
      description: 'Trace paged attention',
    });
    expect(sub2).not.toHaveProperty('status');
    expect(end0).toEqual({ kind: 'turn-end', ts: 1791643300080, stopReason: 'end_turn', elapsedMs: 6082 });
    // Two content blocks of one prompt merge; the image becomes a placeholder.
    expect(u1).toMatchObject({ kind: 'user', text: 'Look at this screenshot\n[image]', promptIndex: 1 });
    expect(model).toMatchObject({ kind: 'model', modelId: 'qwen3.5-0.8b-mlx-bf16' });
    expect(end1).toMatchObject({
      stopReason: 'error',
      elapsedMs: 600000,
      error: 'The model stopped responding after 600s.',
    });

    expect(parsed.turns).toEqual([
      {
        promptId: '5e4f8f96-fc17-4bf8-ba28-105a52e184f2',
        ts: 1791643300080,
        inputTokens: 38723,
        outputTokens: 104,
        cachedTokens: 28182,
        reasoningTokens: 7,
        elapsedMs: 6082,
        ttftMs: 75.738875,
        prefillTps: 290.47170293987074,
        decodeTps: 80.37156392244144,
        model: 'qwen3.5-0.8b-mlx-bf16',
      },
      {
        promptId: '6a1d',
        ts: 1791644001094,
        inputTokens: 0,
        outputTokens: 0,
        cachedTokens: 0,
        reasoningTokens: 0,
        elapsedMs: 600000,
        ttftMs: null,
        prefillTps: null,
        decodeTps: null,
        model: 'qwen3.5-0.8b-mlx-bf16',
      },
    ]);
    expect(parsed.turnsWithUsage).toBe(1);
    expect(parsed.openTurn).toBe(false);
  });

  it('handles legacy lines, bad lines, rewinds, displayText, hidden echoes and interjections', () => {
    const parsed = parseUpdates(readFileSync(MANUAL, 'utf8'));
    expect(parsed.transcript).toEqual([
      { kind: 'user', ts: 0, text: 'legacy hello', promptIndex: null },
      { kind: 'assistant', ts: 0, text: 'legacy reply' },
      { kind: 'user', ts: 1791643279008, text: '/review live', promptIndex: 0 },
      { kind: 'user', ts: 1791643279009, text: 'host echo', promptIndex: null },
      { kind: 'assistant', ts: 1791643279011, text: 'live reply' },
      { kind: 'user', ts: 1791643279012, text: 'also check tests', promptIndex: null, interjection: true },
      { kind: 'assistant', ts: 1791643279013, text: 'after interjection' },
      { kind: 'turn-end', ts: 1791643279015, stopReason: 'end_turn', elapsedMs: 1200 },
    ]);
  });

  it('ignores a torn last line', () => {
    const text = readFileSync(PARENT, 'utf8');
    const torn = `${text}${xai({ sessionUpdate: 'turn_completed', prompt_id: 'x', stop_reason: 'end_turn' }).slice(0, -3)}`;
    expect(parseUpdates(torn).transcript).toEqual(parseUpdates(text).transcript);
    // A complete version of the same line is read.
    const whole = `${text}${xai({ sessionUpdate: 'turn_completed', prompt_id: 'x', stop_reason: 'end_turn' })}\n`;
    expect(parseUpdates(whole).turns).toHaveLength(3);
  });

  it('reports an unterminated turn', () => {
    const parsed = parseUpdates(
      [
        user('run it', 0),
        acp({ sessionUpdate: 'tool_call', toolCallId: 't1', title: 'run_terminal_command', rawInput: 'sleep 600' }),
        '',
      ].join('\n'),
    );
    expect(parsed.openTurn).toBe(true);
    expect(parsed.transcript[1]).toMatchObject({ kind: 'tool', input: 'sleep 600', status: 'pending' });
  });

  it('shows a slash command echoed as a host turn above its output, as the grok pager does', () => {
    const parsed = parseUpdates(
      [
        acp({
          sessionUpdate: 'user_message_chunk',
          content: { type: 'text', text: '/context' },
          _meta: { hostTurn: true },
        }),
        acp({
          sessionUpdate: 'agent_message_chunk',
          content: { type: 'text', text: 'Context: 12k tokens' },
          _meta: { hostTurn: true },
        }),
        xai({ sessionUpdate: 'turn_completed', prompt_id: 'h1', stop_reason: 'end_turn' }),
        '',
      ].join('\n'),
    );
    expect(parsed.transcript.map((e) => [e.kind, 'text' in e ? e.text : null])).toEqual([
      ['user', '/context'],
      ['assistant', 'Context: 12k tokens'],
      ['turn-end', null],
    ]);
  });

  it('a hidden or host-turn prompt opens a turn before any tool call', () => {
    for (const meta of [{ hostTurn: true }, { hideFromScrollback: true, promptIndex: 0 }]) {
      const parsed = parseUpdates(
        `${acp({ sessionUpdate: 'user_message_chunk', content: { type: 'text', text: 'go' }, _meta: meta })}\n`,
      );
      expect(parsed.openTurn).toBe(true);
    }
  });

  it('chips binary tool output instead of dumping it', () => {
    const parsed = parseUpdates(
      [
        acp({ sessionUpdate: 'tool_call', toolCallId: 't1', title: 'read_file' }),
        acp({
          sessionUpdate: 'tool_call_update',
          toolCallId: 't1',
          status: 'completed',
          content: [{ type: 'content', content: { type: 'text', text: '%PDF-1.7\u0000\u0001binary' } }],
        }),
        '',
      ].join('\n'),
    );
    expect(parsed.transcript[0]).toMatchObject({ output: '[PDF · 16 B]' });
  });
});
