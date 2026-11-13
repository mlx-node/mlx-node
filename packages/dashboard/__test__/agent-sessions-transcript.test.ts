import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it, vi } from 'vite-plus/test';

import {
  filterRewindLines,
  MAX_FIELD_CHARS,
  parseUpdates,
  splitCompleteLines,
  TranscriptReducer,
  truncateField,
  wakeCause,
} from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { redactEntry } from '../src/chat/redact.js';
import { HIDDEN_TEXT, WOKEN_T, wokenParentText } from './helpers/woken-turn.js';

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
        prefillMs: null,
        prefillTokens: null,
        prefillCalls: null,
        modelCalls: 2,
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
        prefillMs: null,
        prefillTokens: null,
        prefillCalls: null,
        modelCalls: null,
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
      { kind: 'assistant', ts: 0, text: 'legacy reply', endTs: 0 },
      { kind: 'user', ts: 1791643279008, text: '/review live', promptIndex: 0 },
      { kind: 'user', ts: 1791643279009, text: 'host echo', promptIndex: null },
      { kind: 'wake', ts: 1791643279010, cause: 'unknown', text: 'hidden echo' },
      { kind: 'assistant', ts: 1791643279011, text: 'live reply', endTs: 1791643279011 },
      { kind: 'user', ts: 1791643279012, text: 'also check tests', promptIndex: null, interjection: true },
      { kind: 'assistant', ts: 1791643279013, text: 'after interjection', endTs: 1791643279013 },
      { kind: 'turn-end', ts: 1791643279015, stopReason: 'end_turn', elapsedMs: 1200 },
    ]);
  });

  it('tool and text entries carry endTs', () => {
    const toolCall = (n: number, status: string, extra: object = {}) =>
      acp(
        {
          sessionUpdate: n === 1 ? 'tool_call' : 'tool_call_update',
          toolCallId: 'call-1',
          title: 'Read',
          status,
          ...extra,
        },
        n,
      );
    const thought = (text: string, n: number) =>
      acp({ sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text } }, n);
    const parsed = parseUpdates(
      [
        thought('one ', 1),
        thought('two', 3),
        toolCall(1, 'pending'),
        toolCall(5, 'in_progress'),
        toolCall(8, 'completed'),
        toolCall(9, 'in_progress'),
        acp({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'done' } }, 12),
        acp({ sessionUpdate: 'tool_call', toolCallId: 'call-2', title: 'Run', status: 'pending' }, 13),
      ]
        .map((l) => `${l}\n`)
        .join(''),
    );
    const byKind = (kind: string): Array<Record<string, unknown>> =>
      parsed.transcript.filter((e) => e.kind === kind) as unknown as Array<Record<string, unknown>>;
    expect(byKind('thinking')).toEqual([
      { kind: 'thinking', ts: 1791643279001, text: 'one two', endTs: 1791643279003 },
    ]);
    expect(byKind('assistant')).toEqual([{ kind: 'assistant', ts: 1791643279012, text: 'done', endTs: 1791643279012 }]);
    const tools = byKind('tool');
    expect(tools[0]).toMatchObject({ id: 'call-1', ts: 1791643279001, endTs: 1791643279008, status: 'in_progress' });
    expect(tools[1].id).toBe('call-2');
    expect('endTs' in tools[1]).toBe(false);
  });

  it('a subagent entry carries the finish time; a finish line with no spawn line still makes an entry', () => {
    const spawned = (child: string, n: number) =>
      xai(
        { sessionUpdate: 'subagent_spawned', child_session_id: child, subagent_type: 'explore', description: 'd' },
        n,
      );
    const finished = (child: string, n: number) =>
      xai({ sessionUpdate: 'subagent_finished', child_session_id: child, status: 'completed', duration_ms: 5 }, n);
    const parsed = parseUpdates(
      [spawned('a', 10), finished('a', 20), spawned('a', 30), spawned('b', 31), finished('a', 40), finished('c', 50)]
        .map((l) => `${l}\n`)
        .join(''),
    );
    const subagents = parsed.transcript.filter((e) => e.kind === 'subagent');
    const at = (ms: number | undefined) => (ms === undefined ? null : ms - 1791643279000);
    expect(subagents.map((e) => [e.childId, at(e.ts), at(e.endTs)])).toEqual([
      ['a', 10, 20],
      ['a', 30, 40],
      ['b', 31, null],
      ['c', 50, 50],
    ]);
    expect(subagents.map((e) => e.noSpawn === true)).toEqual([false, false, false, true]);
    expect(subagents[2]).not.toHaveProperty('endTs');
  });

  it('collects model request retries and the time of the last line', () => {
    const retry = (type: string, n: number, extra: object = {}) =>
      xai({ sessionUpdate: 'retry_state', type, reason: 'r', ...extra }, n);
    const parsed = parseUpdates(
      [
        retry('retrying', 3, { attempt: 1, max_retries: 3 }),
        retry('retrying', 4, { attempt: 2 }),
        retry('retrying', 5, { attempt: 'x' }),
        retry('exhausted', 6, { attempt: 3, max_retries: 3 }),
        retry('failed', 7),
        acp({ sessionUpdate: 'tool_call', toolCallId: 'c', title: 'Run', status: 'pending' }, 8),
        acp({ sessionUpdate: 'tool_call_update', toolCallId: 'c', status: 'in_progress' }, 12),
      ]
        .map((l) => `${l}\n`)
        .join(''),
    );
    expect(parsed.retries).toEqual([
      { ts: 1791643279003, attempt: 1, max: 3 },
      { ts: 1791643279004, attempt: 2, max: 0 },
    ]);
    expect(parsed.lastTs).toBe(1791643279012);
    expect(parseUpdates('').lastTs).toBe(0);
    const fixture = parseUpdates(readFileSync(PARENT, 'utf8'));
    expect(fixture.retries).toEqual([{ ts: 1791643279073, attempt: 1, max: 3 }]);
    expect(fixture.lastTs).toBe(1791644001094);
  });

  it('ignores a torn last line', () => {
    const text = readFileSync(PARENT, 'utf8');
    const torn = `${text}${xai({ sessionUpdate: 'turn_completed', prompt_id: 'x', stop_reason: 'end_turn' }).slice(0, -3)}`;
    expect(parseUpdates(torn).transcript).toEqual(parseUpdates(text).transcript);
    // A complete version of the same line is read.
    const whole = `${text}${xai({ sessionUpdate: 'turn_completed', prompt_id: 'x', stop_reason: 'end_turn' })}\n`;
    expect(parseUpdates(whole).turns).toHaveLength(3);
  });

  it('reads the measured prefill sums as a pair, and nothing from a first-call rate', () => {
    const completed = (perf: object, usage?: object) =>
      xai({ sessionUpdate: 'turn_completed', prompt_id: 'p', stop_reason: 'end_turn', perf, usage });
    const parsed = parseUpdates(
      [
        completed(
          {
            ttft_ms: 400,
            prefill_tokens_per_second: 571.2,
            decode_tokens_per_second: 40,
            prefill_input_tokens: 471,
            cached_prefix_tokens: 110_000,
          },
          { inputTokens: 1, outputTokens: 1, modelCalls: 5 },
        ),
        completed(
          { ttft_ms: 400, prefill_ms: 36_250, prefill_tokens: 20_700, prefill_calls: 2 },
          { inputTokens: 1, outputTokens: 1, modelCalls: 5 },
        ),
        completed({ ttft_ms: 400, prefill_ms: 36_250 }),
        '',
      ].join('\n'),
    );
    expect(parsed.turns.map((t) => [t.ttftMs, t.prefillMs, t.prefillTokens, t.prefillCalls, t.modelCalls])).toEqual([
      [400, null, null, null, 5],
      [400, 36_250, 20_700, 2, 5],
      [400, null, null, null, null],
    ]);
    for (const t of parsed.turns) expect('prefillTps' in t).toBe(false);
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

describe('parseUpdates: shell command rows', () => {
  const BASH_ROWS = fileURLToPath(new URL('./fixtures/agent-updates/bash-rows.jsonl', import.meta.url));

  it('a command shows its own output, none when it printed nothing, and how it ended apart from it', () => {
    const tools = parseUpdates(readFileSync(BASH_ROWS, 'utf8')).transcript.filter((e) => e.kind === 'tool');
    const ended = tools.map((e) =>
      e.kind === 'tool'
        ? [e.id, e.status, e.output ?? null, e.exitCode ?? null, e.signal ?? null, e.timedOut ?? null]
        : null,
    );
    expect(ended).toEqual([
      ['toolu_real_ls', 'completed', 'a.txt\n', null, null, null],
      ['toolu_empty', 'completed', null, null, null, null],
      ['toolu_exit2', 'completed', 'ls: missing: No such file or directory\n', 2, null, null],
      ['toolu_timeout', 'completed', null, null, 'timeout', true],
    ]);
    for (const e of tools) expect(JSON.stringify(e)).not.toContain('current_dir');
  });

  it('live, a command that printed nothing drops the description it showed before it ran; a long output is cut and its key end dropped', () => {
    const bash = (output: string, exit: number) => ({
      type: 'Bash',
      output: [],
      output_for_prompt: output,
      exit_code: exit,
      command: 'true',
      truncated: false,
      signal: null,
      timed_out: false,
      current_dir: '/Users/dev/project',
    });
    const text = (t: string) => [{ type: 'content', content: { type: 'text', text: t } }];
    const run = (id: string, last: string, exit: number) => {
      const reducer = new TranscriptReducer({ now: () => 1 });
      const ops = [
        { sessionUpdate: 'tool_call', toolCallId: id, title: 'run_terminal_command' },
        { sessionUpdate: 'tool_call_update', toolCallId: id, title: 'Execute `true`', content: text('Run true') },
        {
          sessionUpdate: 'tool_call_update',
          toolCallId: id,
          status: 'in_progress',
          content: text(''),
          rawOutput: bash('', 0),
        },
        {
          sessionUpdate: 'tool_call_update',
          toolCallId: id,
          status: 'completed',
          content: text(last),
          rawOutput: bash(last, exit),
        },
      ].flatMap((update) => reducer.apply({ xai: false, update, meta: undefined }));
      const end = ops[ops.length - 1]!;
      if (end.op !== 'set') throw new Error(`last op is ${end.op}`);
      return { final: end.entry, snapshot: reducer.snapshot().entries[0]! };
    };
    const empty = run('toolu_live_empty', '', 1);
    for (const e of [empty.final, empty.snapshot]) {
      expect(e).toMatchObject({ kind: 'tool', status: 'completed', exitCode: 1 });
      expect(e).not.toHaveProperty('output');
    }
    const key = 'sk-live-KEY-0123456789';
    const long = `${'x'.repeat(MAX_FIELD_CHARS - 6)}${key}`;
    const cut = run('toolu_live_long', long, 0).final;
    expect(cut).toMatchObject({ kind: 'tool', truncated: true });
    expect(cut.kind === 'tool' && cut.output!.length).toBe(MAX_FIELD_CHARS);
    const shown = redactEntry(cut, [key]);
    expect(shown.kind === 'tool' && shown.output).toBe('x'.repeat(MAX_FIELD_CHARS - 6));
  });

  it('only the Bash shape changes: another tool with no content text still shows its rawOutput', () => {
    const lines = [
      acp({ sessionUpdate: 'tool_call', toolCallId: 'toolu_other', title: 'todo_write', rawInput: {} }, 1),
      acp(
        {
          sessionUpdate: 'tool_call_update',
          toolCallId: 'toolu_other',
          status: 'completed',
          content: [],
          rawOutput: { type: 'Todo', exit_code: 3 },
        },
        2,
      ),
    ];
    const entry = parseUpdates(`${lines.join('\n')}\n`).transcript.find((e) => e.kind === 'tool');
    expect(entry).toMatchObject({ kind: 'tool', id: 'toolu_other' });
    expect(entry!.kind === 'tool' && JSON.parse(entry!.output!)).toEqual({ type: 'Todo', exit_code: 3 });
    expect(entry).not.toHaveProperty('exitCode');
  });
});

describe('parseUpdates: impossible times', () => {
  const at = (timestamp: string, meta: object): string =>
    `{"timestamp":${timestamp},"method":"session/update","params":{"sessionId":"${SID}","update":{"sessionUpdate":"user_message_chunk","content":{"type":"text","text":"hi"}},"_meta":${JSON.stringify(meta)}}}`;
  const tsOf = (line: string): { ts: number[]; lastTs: number } => {
    const parsed = parseUpdates(`${line}\n`);
    return { ts: parsed.transcript.map((e) => e.ts), lastTs: parsed.lastTs };
  };

  it('seconds that overflow to Infinity in ms read as untimed', () => {
    expect(tsOf(at('1e308', {}))).toEqual({ ts: [0], lastTs: 0 });
  });

  it('an agent time outside the Date range or not after the epoch reads as untimed', () => {
    for (const ms of [1e300, -5, 8_640_000_000_000_001]) {
      expect(tsOf(at('1791643279', { agentTimestampMs: ms })), String(ms)).toEqual({ ts: [0], lastTs: 0 });
    }
  });

  it('real times are kept', () => {
    expect(tsOf(at('1791643279', {}))).toEqual({ ts: [1791643279000], lastTs: 1791643279000 });
    expect(tsOf(at('1', { agentTimestampMs: 1791643279123 }))).toEqual({
      ts: [1791643279123],
      lastTs: 1791643279123,
    });
    expect(tsOf(at('1', { agentTimestampMs: 8_640_000_000_000_000 })).ts).toEqual([8_640_000_000_000_000]);
  });
});

describe('parseUpdates: wake prompts', () => {
  const hidden = (text: string, n: number, extra: object = {}) =>
    acp(
      {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text },
        _meta: { promptIndex: 1, hideFromScrollback: true, ...extra },
      },
      n,
    );
  const hook = (promptId: string, n: number) =>
    xai({ sessionUpdate: 'hook_execution', event_name: 'user_prompt_submit', prompt_id: promptId, runs: [] }, n);
  const done = (promptId: string, n: number) =>
    xai({ sessionUpdate: 'turn_completed', prompt_id: promptId, stop_reason: 'end_turn' }, n);
  const workflow = (runId: string, status: string, n: number, name = 'deep-research') =>
    xai({ sessionUpdate: 'workflow_updated', run_id: runId, name, status, revision: n, foreground: false }, n);
  const wakes = (lines: string[]) =>
    parseUpdates(`${lines.join('\n')}\n`).transcript.filter(
      (e): e is Extract<TranscriptEntry, { kind: 'wake' }> => e.kind === 'wake',
    );

  it('a turn woken by a background workflow: the hidden prompt is one wake entry woken by the workflow, never a user entry', () => {
    const parsed = parseUpdates(wokenParentText());
    expect(parsed.transcript.filter((e) => e.kind === 'user')).toHaveLength(1);
    const wake = parsed.transcript.filter((e) => e.kind === 'wake');
    expect(wake).toEqual([
      {
        kind: 'wake',
        ts: WOKEN_T + 2_680_208,
        cause: 'workflow',
        workflow: { name: 'deep-research', status: 'complete' },
        text: HIDDEN_TEXT,
      },
    ]);
    const end = parsed.transcript.findIndex((e) => e.kind === 'turn-end');
    // Between the two turns: the spawn lines of the gap only.
    expect(parsed.transcript.slice(end + 1, parsed.transcript.indexOf(wake[0]!)).map((e) => e.kind)).toEqual([
      'subagent',
      'subagent',
      'subagent',
    ]);
  });

  it('reads the cause from the hook line while the turn is open, and from turn_completed at its end', () => {
    expect(wakes([workflow('wf_1', 'complete', 1), hook('workflow-completed-wf_1-1', 2), hidden('go', 3)])).toEqual([
      {
        kind: 'wake',
        ts: 1791643279003,
        cause: 'workflow',
        workflow: { name: 'deep-research', status: 'complete' },
        text: 'go',
      },
    ]);
    expect(wakes([workflow('wf_1', 'complete', 1), hidden('go', 3)])[0]).toEqual({
      kind: 'wake',
      ts: 1791643279003,
      cause: 'unknown',
      text: 'go',
    });
    expect(
      wakes([workflow('wf_1', 'complete', 1), hidden('go', 3), done('workflow-completed-wf_1-1', 4)])[0],
    ).toMatchObject({ cause: 'workflow', workflow: { name: 'deep-research', status: 'complete' } });
  });

  it('names the child whose finish woke the agent', () => {
    const spawn = xai(
      {
        sessionUpdate: 'subagent_spawned',
        subagent_id: 's1',
        child_session_id: 'c1',
        subagent_type: 'general',
        description: 'Check the build\nthen report',
      },
      1,
    );
    const finish = xai(
      { sessionUpdate: 'subagent_finished', subagent_id: 's1', child_session_id: 'c1', status: 'completed' },
      2,
    );
    expect(wakes([spawn, finish, hook('subagent-completed-s1', 3), hidden('done', 4)])[0]).toMatchObject({
      cause: 'subagent',
      subagent: { childId: 'c1', description: 'Check the build\nthen report' },
    });
    expect(wakes([hook('subagent-completed-s9', 3), hidden('done', 4)])[0]).toEqual({
      kind: 'wake',
      ts: 1791643279004,
      cause: 'subagent',
      text: 'done',
    });
  });

  it('maps every prompt id prefix of the agent to its cause', () => {
    expect(wakeCause('workflow-completed-wf_1-30')).toBe('workflow');
    expect(wakeCause('subagent-completed-s1')).toBe('subagent');
    expect(wakeCause('task-completed-t1')).toBe('task');
    expect(wakeCause('notifications-1f2e')).toBe('notifications');
    expect(wakeCause('goal-summary-3')).toBe('goal');
    expect(wakeCause('goal-classifier-nudge-3')).toBe('goal');
    expect(wakeCause('2e6f2f41-a474-4869')).toBe('unknown');
    expect(wakeCause(null)).toBe('unknown');
    for (const [id, cause] of [
      ['task-completed-t1', 'task'],
      ['notifications-1f2e', 'notifications'],
      ['goal-summary-3', 'goal'],
      ['goal-classifier-nudge-3', 'goal'],
    ] as const) {
      expect(wakes([hook(id, 1), hidden('x', 2)])[0]?.cause, id).toBe(cause);
    }
  });

  it('takes the run id up to the last dash, and leaves the workflow out for a run no line names', () => {
    expect(
      wakes([workflow('wf-1', 'complete', 1), hook('workflow-completed-wf-1-9', 2), hidden('x', 3)])[0],
    ).toMatchObject({
      workflow: { name: 'deep-research', status: 'complete' },
    });
    expect(wakes([workflow('wf-2', 'complete', 1), hook('workflow-completed-wf-1-9', 2), hidden('x', 3)])[0]).toEqual({
      kind: 'wake',
      ts: 1791643279003,
      cause: 'workflow',
      text: 'x',
    });
    expect(
      wakes([workflow('wf-1', 'complete', 1, ''), hook('workflow-completed-wf-1-9', 2), hidden('x', 3)])[0],
    ).not.toHaveProperty('workflow');
  });

  it('keeps the status the run had when it woke the agent', () => {
    const lines = [
      workflow('wf_1', 'paused', 1),
      hidden('x', 2),
      workflow('wf_1', 'active', 3),
      done('workflow-completed-wf_1-1', 4),
    ];
    expect(wakes(lines)[0]).toMatchObject({ cause: 'workflow', workflow: { name: 'deep-research', status: 'paused' } });
  });

  it('a woken turn keeps only the runs updated inside it, never a copy of every run', () => {
    const runs = 2000;
    const lines: string[] = [];
    for (let i = 0; i < runs; i += 1) lines.push(workflow(`wf_${i}`, 'paused', i + 1));
    for (let i = 0; i < runs; i += 1) {
      const n = runs + 3 * i;
      lines.push(
        hidden('x', n, { promptIndex: i }),
        workflow(`wf_${i}`, 'archived', n + 1),
        done(`workflow-completed-wf_${i}-1`, n + 2),
      );
    }
    const text = `${lines.join('\n')}\n`;
    const set = vi.spyOn(Map.prototype, 'set');
    let parsed: ReturnType<typeof parseUpdates>;
    let sets: number;
    try {
      parsed = parseUpdates(text);
      sets = set.mock.calls.length;
    } finally {
      set.mockRestore();
    }
    // Per line at most the live run, its state at the wake and the wake text: a copy per wake makes runs² / 2.
    expect(sets).toBeLessThanOrEqual(lines.length * 2);
    const woken = parsed.transcript.filter((e) => e.kind === 'wake');
    expect(woken).toHaveLength(runs);
    expect(woken.at(-1)).toMatchObject({ cause: 'workflow', workflow: { status: 'paused' } });
  });

  it('joins chunks of one hidden prompt, also with hostTurn, and cuts a long one', () => {
    expect(wakes([hidden('first', 1), hidden('second', 2)])).toEqual([
      { kind: 'wake', ts: 1791643279001, cause: 'unknown', text: 'first\nsecond' },
    ]);
    expect(wakes([hidden('first', 1), hidden('other', 2, { promptIndex: 2 })])).toHaveLength(2);
    expect(wakes([hidden('host', 1, { hostTurn: true })])).toHaveLength(1);
    const long = wakes([hidden('x'.repeat(MAX_FIELD_CHARS + 10), 1)])[0]!;
    expect(long.text).toHaveLength(MAX_FIELD_CHARS);
    expect(long.truncated).toBe(true);
    expect(wakes([hidden('', 1)])).toEqual([{ kind: 'wake', ts: 1791643279001, cause: 'unknown', text: '' }]);
  });

  it("a typed turn's prompt id never changes a later wake; a visible chunk stays a user entry", () => {
    const lines = [
      hook('workflow-completed-wf_1-1', 1),
      user('typed', 0),
      done('workflow-completed-wf_1-1', 3),
      workflow('wf_1', 'complete', 4),
      hidden('woke', 5),
    ];
    const parsed = parseUpdates(`${lines.join('\n')}\n`);
    expect(parsed.transcript.map((e) => e.kind)).toEqual(['user', 'turn-end', 'wake']);
    expect(parsed.transcript[2]).toMatchObject({ cause: 'unknown' });
    expect(parsed.transcript[0]).toMatchObject({ kind: 'user', text: 'typed' });
  });
});

describe('the length stop and the automatic continue', () => {
  const thoughtLine = (text: string, n: number) =>
    acp({ sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text } }, n);
  const lengthEnd = (n: number) =>
    xai(
      {
        sessionUpdate: 'turn_completed',
        prompt_id: 'p1',
        stop_reason: 'error',
        agent_result: 'response truncated by max_tokens',
        error_kind: 'max_tokens_truncation',
      },
      n,
    );
  const normalEnd = (n: number) =>
    xai({ sessionUpdate: 'turn_completed', prompt_id: 'p2', stop_reason: 'end_turn' }, n);
  const lines = [
    user('fix it', 0),
    thoughtLine('Let me ', 2),
    thoughtLine('think.', 3),
    lengthEnd(4),
    user('continue', 1),
    thoughtLine('Again.', 6),
    agent('Done.'),
    normalEnd(8),
  ];

  /** The lines as the app sees them live: its own prompts pushed, every other line applied. */
  function live(input: string[]): { entries: TranscriptEntry[]; ops: unknown[] } {
    const reducer = new TranscriptReducer();
    const ops: unknown[] = [];
    for (const text of input) {
      const params = (
        JSON.parse(text) as {
          method: string;
          params: { update: Record<string, unknown>; _meta: Record<string, unknown> };
        }
      ).params;
      const content = params.update.content as { text?: string } | undefined;
      if (params.update.sessionUpdate === 'user_message_chunk') {
        ops.push(...reducer.userPrompt(content?.text ?? '', params._meta.agentTimestampMs as number));
      } else {
        const isXai = (JSON.parse(text) as { method: string }).method === '_x.ai/session/update';
        ops.push(...reducer.apply({ xai: isXai, update: params.update, meta: params._meta }));
      }
    }
    return { entries: reducer.snapshot().entries, ops };
  }

  /** What may differ live and from disk: the thought's time, and the prompt index the agent stores. */
  function comparable(entries: readonly TranscriptEntry[]): unknown[] {
    return entries.map((entry) => {
      if (entry.kind === 'thinking') return { kind: entry.kind, text: entry.text };
      if (entry.kind === 'user') return { ...entry, promptIndex: null };
      return entry;
    });
  }

  // Mutation: mark `continued` by the text alone → red (a continue after a normal end).
  it('live and from disk give the same entries, with continued and lengthLimit', () => {
    const disk = parseUpdates(`${lines.join('\n')}\n`).transcript;
    const hot = live(lines).entries;
    expect(comparable(hot)).toEqual(comparable(disk));
    const users = disk.filter((e) => e.kind === 'user');
    expect(users.map((e) => e.kind === 'user' && e.continued === true)).toEqual([false, true]);
    const ends = disk.filter((e) => e.kind === 'turn-end');
    expect(ends.map((e) => e.kind === 'turn-end' && e.lengthLimit === true)).toEqual([true, false]);

    const after = parseUpdates(`${[user('hi', 0), agent('ok'), normalEnd(3), user('continue', 1)].join('\n')}\n`);
    const last = after.transcript.at(-1)!;
    expect(last).toMatchObject({ kind: 'user', text: 'continue' });
    expect(last).not.toHaveProperty('continued');
  });

  // Mutation: stamp `timed` on a thought read from disk → red.
  it('only a thought made from live lines is timed; the text op carries the end stamp', () => {
    const disk = parseUpdates(`${lines.join('\n')}\n`).transcript.filter((e) => e.kind === 'thinking');
    expect(disk.every((e) => !('timed' in e))).toBe(true);
    const { entries, ops } = live(lines);
    const thoughts = entries.filter((e) => e.kind === 'thinking');
    expect(thoughts.map((e) => e.kind === 'thinking' && e.timed === true)).toEqual([true, true]);
    expect(ops).toContainEqual({ op: 'text', index: 1, append: 'think.', endTs: 1791643279003 });

    const split = parseUpdates(
      `${[user('go', 0), thoughtLine('a', 2), acp({ sessionUpdate: 'available_commands_update', availableCommands: [] }, 3), thoughtLine('b', 4)].join('\n')}\n`,
    ).transcript.filter((e) => e.kind === 'thinking');
    expect(split).toHaveLength(1);
    expect(split[0]).toMatchObject({ kind: 'thinking', text: 'ab' });
    expect(split[0]).not.toHaveProperty('timed');
  });
});
