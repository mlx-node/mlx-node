import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it, vi } from 'vite-plus/test';

import {
  filterRewindLines,
  MAX_FIELD_CHARS,
  MAX_REPORT_CHARS,
  parseUpdates,
  splitCompleteLines,
  stripAnsi,
  TranscriptReducer,
  truncateField,
  wakeCause,
} from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { promptParams } from '../src/chat/agent/profiles.js';
import { attachmentView, pdfNote, type AttachmentInput } from '../src/chat/attachment-rule.js';
import { applyOp, Conversation, type HostView } from '../src/chat/conversation.js';
import { redactEntry } from '../src/chat/redact.js';
import type { ConversationEvent } from '../src/chat/types.js';
import { HIDDEN_TEXT, WOKEN_PHASES, WOKEN_REPORT, WOKEN_RUN, WOKEN_T, wokenParentText } from './helpers/woken-turn.js';

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

describe('a rewind: the disk read equals the live cut', () => {
  const done = (promptId: string, n: number) =>
    xai(
      {
        sessionUpdate: 'turn_completed',
        prompt_id: promptId,
        stop_reason: 'end_turn',
        usage: { inputTokens: 10, outputTokens: 2, modelCalls: 1 },
      },
      n,
    );
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

  // Mutation: keep every turn metric in `Conversation.cut` → the live turns hold the cut turn's metric.
  it('the entries and turn metrics the host keeps are the ones a read of the file with the marker gives', () => {
    const lines = [user('first', 0), agent('one'), done('p0', 2), user('second', 1), agent('two'), done('p1', 4)];
    const before = parseUpdates(`${lines.join('\n')}\n`);
    const conversation = new Conversation(
      {
        id: 'c',
        mode: 'chat',
        title: 't',
        cwd: null,
        modelId: 'm',
        createdAt: 1,
        updatedAt: 1,
        hot: true,
        permissionMode: 'ask',
      },
      { needles: () => [], flushMs: 1 },
    );
    conversation.seed(before.transcript, before.turns, null);
    const events: ConversationEvent[] = [];
    conversation.attach((event) => events.push(event), host);
    const at = before.transcript.findIndex((e) => e.kind === 'user' && e.text === 'second');
    const kept = conversation.cut(at, host);
    const disk = parseUpdates(`${[...lines, rewind(1)].join('\n')}\n`);
    expect(kept).toEqual(disk.transcript);
    const sync = events.at(-1) as Extract<ConversationEvent, { type: 'sync' }>;
    expect(sync.entries).toEqual(disk.transcript);
    expect(sync.turns).toEqual(disk.turns);
    expect(disk.turns.map((t) => t.promptId)).toEqual(['p0']);
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
      'notice',
      'turn-end',
      'user',
      'model',
      'assistant',
      'turn-end',
    ]);
    const [u0, thinking, assistant, todo, plan, read, orphan, sub1, sub2, stall, end0, u1, model, , end1] =
      parsed.transcript as TranscriptEntry[];
    // The stored stall is one retry episode; the turn end closed it with no outcome.
    expect(stall).toEqual({
      kind: 'notice',
      ts: 1791643279073,
      code: 'retry',
      cause: 'stalled',
      attempts: 1,
      max: 3,
      endTs: 1791643300080,
    });
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
    // Two content blocks of one prompt merge; the image is an attachment of the message, not a text line.
    expect(u1).toEqual({
      kind: 'user',
      ts: expect.any(Number),
      text: 'Look at this screenshot',
      promptIndex: 1,
      attachments: [{ kind: 'image', mediaType: 'image/png', bytes: 8 }],
    });
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

describe('parseUpdates: colored command output', () => {
  const stored = (text: string): TranscriptEntry | undefined =>
    parseUpdates(
      `${[
        acp({ sessionUpdate: 'tool_call', toolCallId: 'toolu_color', title: 'run_terminal_command' }, 1),
        acp(
          {
            sessionUpdate: 'tool_call_update',
            toolCallId: 'toolu_color',
            status: 'completed',
            content: [{ type: 'content', content: { type: 'text', text } }],
            rawOutput: {
              type: 'Bash',
              output: [...Buffer.from(text)],
              output_for_prompt: 'exit: 0\n',
              exit_code: 0,
              command: 'yarn test',
              truncated: text.includes('... (output truncated) ...'),
              signal: null,
              timed_out: false,
              description: 'Run the tests',
              current_dir: '/Users/dev/proj',
              output_file: '/Users/dev/.mlx-agent/sessions/proj/s/terminal/toolu_color.log',
              total_bytes: Buffer.byteLength(text),
            },
          },
          2,
        ),
      ].join('\n')}\n`,
    ).transcript.find((e) => e.kind === 'tool');

  // Mutation: drop stripAnsi from toolContentText → '[binary · 1 KB]'.
  it('a stored output with color codes reads as text, not [binary …]', () => {
    const line =
      '\u001b[1m\u001b[32m PASS \u001b[39m\u001b[22m __test__/line-diff.test.ts \u001b[2m(12 tests)\u001b[22m\n';
    expect(stored(line.repeat(20))).toMatchObject({
      output: ' PASS  __test__/line-diff.test.ts (12 tests)\n'.repeat(20),
    });
  });

  it('a real binary blob still reads [binary …], also with a color code in it', () => {
    expect(stored('\u001b[0m\u0000\u0001\u0002\u0003raw')).toMatchObject({ output: '[binary · 7 B]' });
  });

  // Mutation: strip every ESC → 'abcdef'.
  it('a lone ESC with no sequence still counts as binary', () => {
    expect(stored('abc\u001bdef')).toMatchObject({ output: '[binary · 7 B]' });
    expect(stored('abc\u001b(Bdef')).toMatchObject({ output: '[binary · 9 B]' });
  });

  // Mutation: keep an ESC the agent's head-and-tail cut left alone → '[binary · …]'.
  it('output the agent cut inside a color code reads as text: ESC alone, ESC [, ESC [ with parameters', () => {
    const cut = '\n\n... (output truncated) ...\n\n';
    for (const head of ['\u001b', '\u001b[', '\u001b[1;3']) {
      expect(stored(`\u001b[32mok\u001b[0m ${head}${cut}2m tail\n`), JSON.stringify(head)).toMatchObject({
        output: `ok ${cut}2m tail\n`,
      });
    }
  });
});

describe('stripAnsi', () => {
  it('removes CSI and OSC sequences only, and keeps a \\r redraw as text', () => {
    expect(stripAnsi('plain\ttext\r\n')).toBe('plain\ttext\r\n');
    expect(stripAnsi('\u001b[38;5;196mred\u001b[0m \u001b[?25lhidden\u001b[?25h')).toBe('red hidden');
    expect(stripAnsi('\u001b]0;title\u0007a \u001b]8;;https://x.dev\u001b\\link\u001b]8;;\u001b\\ b')).toBe('a link b');
    expect(stripAnsi('50%\r\u001b[2K100%\n')).toBe('50%\r100%\n');
  });

  it('a CSI cut before its final byte loses only ESC [ and its parameters; an OSC with no end stays', () => {
    expect(stripAnsi('ok \u001b[3\n\n... (output truncated) ...\n\ntail')).toBe(
      'ok \n\n... (output truncated) ...\n\ntail',
    );
    expect(stripAnsi('done \u001b[38;5')).toBe('done ');
    expect(stripAnsi('a \u001b]0;title with no end')).toBe('a \u001b]0;title with no end');
  });

  it('an ESC a cut left alone goes: the last character, or right before the cut marker; any other lone ESC stays', () => {
    expect(stripAnsi('x\u001b')).toBe('x');
    expect(stripAnsi('x\u001b\n\n... (output truncated) ...\n\ny')).toBe('x\n\n... (output truncated) ...\n\ny');
    expect(stripAnsi('x\u001b\ny')).toBe('x\u001b\ny');
    expect(stripAnsi('x\u001b y')).toBe('x\u001b y');
    expect(stripAnsi('x\u001b\\y')).toBe('x\u001b\\y');
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
        workflow: {
          runId: WOKEN_RUN,
          revision: 30,
          name: 'deep-research',
          snapshot: {
            status: 'complete',
            report: WOKEN_REPORT,
            agentsUsed: 8,
            agentBudget: 128,
            agentsFailed: 0,
            elapsedMs: 2_680_142 - 60_702,
          },
        },
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
    const snapshot = { status: 'complete', agentsUsed: 0, agentBudget: null, agentsFailed: 0, elapsedMs: 0 };
    expect(wakes([workflow('wf_1', 'complete', 1), hook('workflow-completed-wf_1-1', 2), hidden('go', 3)])).toEqual([
      {
        kind: 'wake',
        ts: 1791643279003,
        cause: 'workflow',
        workflow: { runId: 'wf_1', revision: 1, name: 'deep-research', snapshot },
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
    ).toMatchObject({ cause: 'workflow', workflow: { name: 'deep-research', snapshot } });
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
      workflow: { runId: 'wf-1', revision: 9, name: 'deep-research', snapshot: null },
    });
    expect(
      wakes([workflow('wf-1', 'complete', 1), hook('workflow-completed-wf-1-x', 2), hidden('x', 3)])[0],
    ).not.toHaveProperty('workflow');
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

  it('keeps the update of the revision its prompt id names, never a later one', () => {
    const lines = [
      workflow('wf_1', 'failed', 1),
      hidden('x', 2),
      workflow('wf_1', 'active', 3),
      done('workflow-completed-wf_1-1', 4),
    ];
    expect(wakes(lines)[0]).toMatchObject({
      cause: 'workflow',
      workflow: { revision: 1, snapshot: { status: 'failed' } },
    });
  });

  it('a woken turn keeps only the runs updated inside it, never a copy of every run', () => {
    const runs = 2000;
    const lines: string[] = [];
    for (let i = 0; i < runs; i += 1) lines.push(workflow(`wf_${i}`, 'failed', i + 1));
    for (let i = 0; i < runs; i += 1) {
      const n = runs + 3 * i;
      lines.push(
        hidden('x', n, { promptIndex: i }),
        workflow(`wf_${i}`, 'active', n + 1),
        done(`workflow-completed-wf_${i}-${i + 1}`, n + 2),
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
    // Per line at most the run's name, entry and kept update, and the wake text: a copy per wake makes runs² / 2.
    expect(sets).toBeLessThanOrEqual(lines.length * 2);
    const woken = parsed.transcript.filter((e) => e.kind === 'wake');
    expect(woken).toHaveLength(runs);
    expect(woken.at(-1)).toMatchObject({ cause: 'workflow', workflow: { snapshot: { status: 'failed' } } });
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
    expect(parsed.transcript.map((e) => e.kind)).toEqual(['user', 'turn-end', 'workflow', 'wake']);
    expect(parsed.transcript[3]).toMatchObject({ cause: 'unknown' });
    expect(parsed.transcript[0]).toMatchObject({ kind: 'user', text: 'typed' });
  });
});

describe('parseUpdates: workflow runs', () => {
  type Run = Extract<TranscriptEntry, { kind: 'workflow' }>;
  const runs = (entries: readonly TranscriptEntry[]): Run[] => entries.filter((e): e is Run => e.kind === 'workflow');
  const line = (runId: string, revision: number, status: string, n: number, extra: object = {}) =>
    xai({ sessionUpdate: 'workflow_updated', run_id: runId, name: 'deep-research', status, revision, ...extra }, n);

  it('a woken session holds one entry for its run, in the turn that started it, with the last revision and the report', () => {
    const parsed = parseUpdates(wokenParentText());
    const found = runs(parsed.transcript);
    expect(found).toHaveLength(1);
    const run = found[0]!;
    expect(parsed.transcript.indexOf(run)).toBeLessThan(parsed.transcript.findIndex((e) => e.kind === 'turn-end'));
    expect(run).toMatchObject({
      kind: 'workflow',
      ts: WOKEN_T + 60_702,
      updatedTs: WOKEN_T + 2_680_142,
      runId: WOKEN_RUN,
      revision: 30,
      name: 'deep-research',
      status: 'complete',
      currentPhase: null,
      agentBudget: 128,
      agentsUsed: 8,
      report: WOKEN_REPORT,
    });
    expect(run.phases).toEqual(WOKEN_PHASES.map((title) => ({ title, state: 'done' })));
    expect(run.agents).toHaveLength(8);
    expect(run.agents[0]).toEqual({
      agentId: 'sa-plan',
      label: 'Plan',
      phase: null,
      state: 'done',
      tokens: 40_000,
      durationMs: 227_364 - 60_750,
    });
    expect(run.truncated).toBeUndefined();
    expect(run.pauseMessage).toBeUndefined();
  });

  it('a lower or equal revision changes nothing; each run has its own entry', () => {
    const parsed = parseUpdates(
      `${[
        line('wf_a', 2, 'active', 1, { pause_message: 'held' }),
        line('wf_b', 1, 'active', 2),
        line('wf_a', 1, 'complete', 3, { result_summary: 'old' }),
        line('wf_a', 2, 'failed', 4),
        line('wf_a', 5, 'user_paused', 5, { pause_message: 'Paused by the user' }),
        line('wf_a', 6, 'cancelled', 6),
        line('wf_x', Number.NaN, 'active', 7),
      ].join('\n')}\n`,
    );
    const found = runs(parsed.transcript);
    expect(found.map((r) => [r.runId, r.revision, r.status])).toEqual([
      ['wf_a', 6, 'cancelled'],
      ['wf_b', 1, 'active'],
    ]);
    expect(found[0]!.report).toBeUndefined();
    expect(found[0]!.pauseMessage).toBeUndefined();
    expect(found[0]!.ts).toBe(1791643279001);
    expect(found[0]!.updatedTs).toBe(1791643279006);
  });

  it('live: the first update pushes the entry, a higher revision sets it in place, a lower one sends nothing', () => {
    const reducer = new TranscriptReducer({ now: () => 1 });
    const live = (revision: number, status: string, extra: object = {}) =>
      reducer.apply({
        xai: true,
        update: { sessionUpdate: 'workflow_updated', run_id: 'wf_1', name: 'n', status, revision, ...extra },
        meta: undefined,
      });
    expect(live(1, 'active').map((o) => o.op)).toEqual(['push']);
    const set = live(3, 'complete', { result_summary: 'Done.' });
    expect(set).toMatchObject([{ op: 'set', index: 0, entry: { kind: 'workflow', revision: 3, report: 'Done.' } }]);
    expect(live(2, 'active')).toEqual([]);
  });

  it('seeded: a run read from disk is set in place by a higher revision, a lower one sends nothing', () => {
    const disk = parseUpdates(
      `${[user('go', 0), line('wf_1', 1, 'user_paused', 1, { pause_message: 'Paused by the user' })].join('\n')}\n`,
    );
    const reducer = new TranscriptReducer({ now: () => 1 });
    reducer.seed(disk.transcript, disk.turns);
    const live = (revision: number, status: string, extra: object = {}) =>
      reducer.apply({
        xai: true,
        update: {
          sessionUpdate: 'workflow_updated',
          run_id: 'wf_1',
          name: 'deep-research',
          status,
          revision,
          ...extra,
        },
        meta: undefined,
      });
    const set = live(2, 'complete', { result_summary: 'Done.' });
    expect(set).toMatchObject([{ op: 'set', index: 1, entry: { kind: 'workflow', revision: 2, report: 'Done.' } }]);
    expect(live(1, 'active')).toEqual([]);
    const found = runs(reducer.snapshot().entries);
    expect(found.map((r) => [r.runId, r.revision, r.status, r.report])).toEqual([['wf_1', 2, 'complete', 'Done.']]);
    expect(found[0]!.pauseMessage).toBeUndefined();
  });

  it('a report longer than 20 KB is cut and marked; redaction drops a key cut at its end', () => {
    const key = 'sk-SECRET-0123456789';
    const report = `${'r'.repeat(MAX_REPORT_CHARS - 5)}${key}`;
    const run = runs(parseUpdates(`${line('wf_1', 1, 'complete', 1, { result_summary: report })}\n`).transcript)[0]!;
    expect(run.report).toHaveLength(MAX_REPORT_CHARS);
    expect(run.truncated).toBe(true);
    const shown = redactEntry(run, [key]) as Run;
    expect(shown.report).not.toContain(key.slice(0, 5));
  });

  // Mutations: drop `run` → the thread draws the helper twice; drop `retryOf` → a retried helper is listed twice in its
  // card; keep an empty `retry_of` → a spawn folds onto no row.
  it("a helper's spawn names its run, also before the run's entry, what it resumed and the attempt it retries", () => {
    const spawn = (child: string, n: number, extra: object) =>
      xai(
        {
          sessionUpdate: 'subagent_spawned',
          subagent_id: child,
          parent_session_id: SID,
          child_session_id: child,
          subagent_type: 'general-purpose',
          description: 'track:process',
          ...extra,
        },
        n,
      );
    const parsed = parseUpdates(
      `${[
        spawn('c0', 1, { workflow_run_id: 'wf_1' }),
        line('wf_1', 1, 'active', 2),
        spawn('c1', 3, { workflow_run_id: 'wf_1' }),
        spawn('c2', 4, { workflow_run_id: 'wf_1', resumed_from: 'c1', retry_of: 'c1' }),
        spawn('c3', 5, { resumed_from: 'c2', retry_of: '' }),
      ].join('\n')}\n`,
    );
    const helpers = parsed.transcript.filter((e) => e.kind === 'subagent');
    expect(helpers.map((e) => [e.childId, e.run, e.resumedFrom, 'retryOf' in e ? e.retryOf : 'absent'])).toEqual([
      ['c0', 'wf_1', undefined, 'absent'],
      ['c1', 'wf_1', undefined, 'absent'],
      ['c2', 'wf_1', 'c1', 'c1'],
      ['c3', undefined, 'c2', 'absent'],
    ]);
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
        ops.push(...reducer.userPrompt(content?.text ?? '', [], params._meta.agentTimestampMs as number));
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

describe('parseUpdates: attachments of a user message', () => {
  const PNG = 'iVBORw0KGgoAAAANSUhEUg==';
  const FILES: AttachmentInput[] = [
    { kind: 'file', name: 'notes#1.md', text: 'línea uno\nline two\n' },
    { kind: 'image', mediaType: 'image/png', data: PNG },
  ];
  const block = (content: object, promptIndex: number | null, n: number, extra: object = {}) =>
    acp(
      {
        sessionUpdate: 'user_message_chunk',
        content,
        ...(promptIndex === null ? {} : { _meta: { promptIndex, ...extra } }),
      },
      n,
    );
  /** The prompt as the agent stores it: one user chunk per block the app sent, all of one prompt index. */
  const stored = (text: string, attachments: AttachmentInput[], promptIndex = 0) =>
    (promptParams('sid', text, attachments).prompt as object[]).map((b, i) => block(b, promptIndex, i + 1));
  const users = (lines: string[]) => parseUpdates(`${lines.join('\n')}\n`).transcript.filter((e) => e.kind === 'user');

  // Mutation: the reader drops `resource` / `image` chunks of a stored user message → red.
  it('a stored text, file and picture of one prompt are one user entry with the views, and no placeholder text', () => {
    expect(users(stored('look at these', FILES))).toEqual([
      {
        kind: 'user',
        ts: expect.any(Number),
        text: 'look at these',
        promptIndex: 0,
        attachments: [
          { kind: 'file', name: 'notes_1.md', bytes: 20 },
          { kind: 'image', mediaType: 'image/png', bytes: 16 },
        ],
      },
    ]);
  });

  it('a resource with a blob counts its bytes; one with neither text nor blob stays a placeholder line', () => {
    const lines = [
      user('two files', 2),
      block({ type: 'resource', resource: { uri: 'file:///attached/a.bin', blob: PNG } }, 2, 2),
      block({ type: 'resource', resource: { uri: 'file:///attached/b.txt' } }, 2, 3),
    ];
    expect(users(lines)).toEqual([
      {
        kind: 'user',
        ts: expect.any(Number),
        text: 'two files\n[resource]',
        promptIndex: 2,
        attachments: [{ kind: 'file', name: 'a.bin', bytes: 16 }],
      },
    ]);
  });

  it('a picture with no text before it is an entry with empty text; the text after it joins without a blank line', () => {
    const lines = [block({ type: 'image', mimeType: 'image/jpeg', data: PNG }, 4, 1), user('what is this', 4)];
    expect(users(lines)).toEqual([
      {
        kind: 'user',
        ts: expect.any(Number),
        text: 'what is this',
        promptIndex: 4,
        attachments: [{ kind: 'image', mediaType: 'image/jpeg', bytes: 16 }],
      },
    ]);
    const alone = users([block({ type: 'image', data: PNG }, null, 1)]);
    expect(alone).toEqual([
      {
        kind: 'user',
        ts: expect.any(Number),
        text: '',
        promptIndex: null,
        attachments: [{ kind: 'image', mediaType: 'image', bytes: 16 }],
      },
    ]);
  });

  it('a hidden picture chunk is still the placeholder text of its wake entry', () => {
    const lines = [
      block({ type: 'text', text: 'woke' }, 1, 1, { hideFromScrollback: true }),
      block({ type: 'image', mimeType: 'image/png', data: PNG }, 1, 2, { hideFromScrollback: true }),
    ];
    const parsed = parseUpdates(`${lines.join('\n')}\n`).transcript;
    expect(parsed).toEqual([{ kind: 'wake', ts: expect.any(Number), cause: 'unknown', text: 'woke\n[image]' }]);
  });

  const PAGE_A = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3, 4, 5]).toString('base64');
  const PAGE_B = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3, 4, 5, 6, 7, 8]).toString('base64');
  const WITH_PDF: AttachmentInput[] = [
    { kind: 'file', name: 'notes.md', text: 'a\n' },
    { kind: 'pdf', name: 'paper.pdf', images: [PAGE_A, PAGE_B] },
    { kind: 'image', mediaType: 'image/png', data: PNG },
  ];

  // Mutation: every image after a note taken as one of its pages → the picture before the pages joins the PDF → red.
  it("a PDF's note and its page images are one PDF view; a picture of the same message stays a picture", () => {
    expect(users(stored('what is the claim?', WITH_PDF))).toEqual([
      {
        kind: 'user',
        ts: expect.any(Number),
        text: 'what is the claim?',
        promptIndex: 0,
        attachments: [
          { kind: 'file', name: 'notes.md', bytes: 2 },
          { kind: 'pdf', name: 'paper.pdf', pages: 2, way: 'message', bytes: 9 + 12, chars: 12 + 16 },
          { kind: 'image', mediaType: 'image/png', bytes: 16 },
        ],
      },
    ]);
  });

  // Mutation: the note's session left off the view → a fork's copy of the original's message reads as its own.
  it('a tool note is a PDF view with its id, its session and no pages; a later picture stays a picture', () => {
    const session = '01a20000-0000-7000-8000-000000000010';
    const toolNote = pdfNote({ name: 'report.pdf', pages: 212, way: 'tool', id: 'k3f9a2c1-212', session });
    const lines = [
      user('summarize', 3),
      block({ type: 'resource', resource: { uri: 'file:///attached/report.pdf', text: toolNote } }, 3, 2),
      block({ type: 'image', mimeType: 'image/png', data: PNG }, 3, 3),
    ];
    expect(users(lines)).toEqual([
      {
        kind: 'user',
        ts: expect.any(Number),
        text: 'summarize',
        promptIndex: 3,
        attachments: [
          { kind: 'pdf', name: 'report.pdf', pages: 212, way: 'tool', bytes: 0, chars: 0, id: 'k3f9a2c1-212', session },
          { kind: 'image', mediaType: 'image/png', bytes: 16 },
        ],
      },
    ]);
  });

  // Mutation: a text file named like a PDF read as a note → red.
  it('a text file named x.pdf without the note line stays a file, and its pictures stay pictures', () => {
    const lines = [
      user('two', 5),
      block({ type: 'resource', resource: { uri: 'file:///attached/x.pdf', text: 'images: 1-1\n' } }, 5, 2),
      block({ type: 'image', mimeType: 'image/jpeg', data: PAGE_A }, 5, 3),
    ];
    expect(users(lines)).toEqual([
      {
        kind: 'user',
        ts: expect.any(Number),
        text: 'two',
        promptIndex: 5,
        attachments: [
          { kind: 'file', name: 'x.pdf', bytes: 12 },
          { kind: 'image', mediaType: 'image/jpeg', bytes: 9 },
        ],
      },
    ]);
  });

  it('live, a PDF message carries the same views the reader makes from its stored blocks', () => {
    const reducer = new TranscriptReducer();
    reducer.userPrompt('what is the claim?', WITH_PDF.map(attachmentView), 5);
    const [entry] = reducer.snapshot().entries;
    const [read] = users(stored('what is the claim?', WITH_PDF));
    expect(entry!.kind === 'user' && entry.attachments).toEqual(read!.kind === 'user' && read.attachments);
  });

  it('live, the pushed prompt carries the same views the reader makes from the stored blocks, in the ops the page gets', () => {
    const reducer = new TranscriptReducer();
    const ops = reducer.userPrompt('look at these', FILES.map(attachmentView), 5);
    const [entry] = reducer.snapshot().entries;
    const [read] = users(stored('look at these', FILES));
    expect(entry).toEqual({
      kind: 'user',
      ts: 5,
      text: 'look at these',
      promptIndex: null,
      attachments: read!.kind === 'user' ? read.attachments : [],
    });
    expect(ops.at(-1)).toEqual({ op: 'set', index: 0, entry });
    expect(new TranscriptReducer().userPrompt('plain', [], 5)).toEqual([
      { op: 'push', entry: { kind: 'user', ts: 5, text: 'plain', promptIndex: null } },
    ]);
  });
});

describe('mode lines (`current_mode_update`)', () => {
  type Step = { line: string; live: 'prompt' | 'apply' | 'none' };
  const MS = 1791643279000;
  const ms = (n: number) => MS + n;
  const stamp = (update: object, n: number, method: string): string =>
    JSON.stringify({
      timestamp: 1791643279,
      method,
      params: { sessionId: SID, update, _meta: { eventId: `${SID}-${n}`, agentTimestampMs: ms(n) } },
    });
  const mode = (id: string) => ({ sessionUpdate: 'current_mode_update', currentModeId: id });
  const typed = (text: string, promptIndex: number) => ({
    sessionUpdate: 'user_message_chunk',
    content: { type: 'text', text },
    _meta: { promptIndex },
  });
  const said = (text: string) => ({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text } });
  const done = { sessionUpdate: 'turn_completed', prompt_id: 'p', stop_reason: 'end_turn' };

  /** The stored file of `steps`, and the page's list from the live lines (the host pushes each typed prompt itself). */
  function play(steps: ReadonlyArray<[object, 'acp' | 'xai', Step['live']]>): {
    disk: TranscriptEntry[];
    live: TranscriptEntry[];
  } {
    const text = steps
      .map(([update, kind], n) => stamp(update, n, kind === 'xai' ? '_x.ai/session/update' : 'session/update'))
      .map((line) => `${line}\n`)
      .join('');
    const reducer = new TranscriptReducer();
    const live: TranscriptEntry[] = [];
    steps.forEach(([update, kind, how], n) => {
      const u = update as { content?: { text?: string } };
      const ops =
        how === 'prompt'
          ? reducer.userPrompt(u.content?.text ?? '', [], ms(n))
          : how === 'apply'
            ? reducer.apply({
                xai: kind === 'xai',
                update: update as Record<string, unknown>,
                meta: { agentTimestampMs: ms(n) },
              })
            : [];
      for (const op of ops) applyOp(live, [], op);
    });
    return { disk: parseUpdates(text).transcript, live };
  }
  const shape = (entries: readonly TranscriptEntry[]) =>
    entries.map((entry) =>
      entry.kind === 'plan-mode' ? `mode:${entry.mode}` : entry.kind === 'user' ? `user:${entry.text}` : entry.kind,
    );
  const noIndex = (entries: readonly TranscriptEntry[]) =>
    entries.map((entry) => (entry.kind === 'user' ? { ...entry, promptIndex: null } : entry));

  it('plan, then default, gives two lines; a repeat of the mode gives none', () => {
    const { disk } = play([
      [mode('plan'), 'acp', 'apply'],
      [typed('Plan the cache', 0), 'acp', 'prompt'],
      [said('Here is the plan.'), 'acp', 'apply'],
      [mode('plan'), 'acp', 'apply'],
      [done, 'xai', 'apply'],
      [mode('default'), 'acp', 'apply'],
      [typed('Build it', 1), 'acp', 'prompt'],
      [said('Done.'), 'acp', 'apply'],
      [mode('default'), 'acp', 'apply'],
      [done, 'xai', 'apply'],
    ]);
    expect(shape(disk)).toEqual([
      'user:Plan the cache',
      'mode:plan',
      'assistant',
      'turn-end',
      'user:Build it',
      'mode:default',
      'assistant',
      'turn-end',
    ]);
    expect(disk[1]).toEqual({ kind: 'plan-mode', ts: ms(0), mode: 'plan' });
  });

  it('a mode line stored before its prompt chunk and one stored after it give the list the live lines give', () => {
    // The agent applies a prompt's `_meta.mode` before it stores the chunk; live, the host pushes the prompt first.
    const before = play([
      [mode('plan'), 'acp', 'none'],
      [typed('Plan the cache', 0), 'acp', 'prompt'],
      [typed('with the notes', 0), 'acp', 'none'],
      [said('Plan.'), 'acp', 'apply'],
      [done, 'xai', 'apply'],
    ]);
    const after = play([
      [typed('Plan the cache', 0), 'acp', 'prompt'],
      [typed('with the notes', 0), 'acp', 'none'],
      [mode('plan'), 'acp', 'apply'],
      [said('Plan.'), 'acp', 'apply'],
      [done, 'xai', 'apply'],
    ]);
    // On disk a mode line between two chunks of one prompt waits, so the chunks still merge.
    const between = play([
      [typed('Plan the cache', 0), 'acp', 'prompt'],
      [mode('plan'), 'acp', 'apply'],
      [typed('with the notes', 0), 'acp', 'none'],
      [said('Plan.'), 'acp', 'apply'],
      [done, 'xai', 'apply'],
    ]);
    expect(shape(before.disk)).toEqual(['user:Plan the cache\nwith the notes', 'mode:plan', 'assistant', 'turn-end']);
    expect(shape(after.disk)).toEqual(shape(before.disk));
    expect(shape(between.disk)).toEqual(shape(before.disk));
    expect(shape(between.live)).toEqual(['user:Plan the cache', 'mode:plan', 'assistant', 'turn-end']);
    // Live the mode line arrives after the host's prompt: the same list, the prompt as one text.
    expect(shape(after.live)).toEqual(['user:Plan the cache', 'mode:plan', 'assistant', 'turn-end']);
    const one = play([
      [mode('plan'), 'acp', 'none'],
      [typed('Plan it', 0), 'acp', 'prompt'],
      [mode('plan'), 'acp', 'apply'],
      [said('Plan.'), 'acp', 'apply'],
      [done, 'xai', 'apply'],
    ]);
    expect(shape(one.disk)).toEqual(shape(one.live));
    expect(noIndex(one.live).map((e) => e.kind)).toEqual(noIndex(one.disk).map((e) => e.kind));
  });

  it('the model enters plan mode in a turn and the approved plan leaves it: lines in place, live equals disk', () => {
    const steps: Array<[object, 'acp' | 'xai', Step['live']]> = [
      [typed('Fix the cache', 0), 'acp', 'prompt'],
      [said('I will plan first.'), 'acp', 'apply'],
      [mode('plan'), 'acp', 'apply'],
      [{ sessionUpdate: 'tool_call', toolCallId: 'x1', title: 'exit_plan_mode', status: 'pending' }, 'acp', 'apply'],
      [mode('default'), 'acp', 'apply'],
      [{ sessionUpdate: 'tool_call_update', toolCallId: 'x1', status: 'completed' }, 'acp', 'apply'],
      [said('Building.'), 'acp', 'apply'],
      [done, 'xai', 'apply'],
    ];
    const { disk, live } = play(steps);
    expect(shape(disk)).toEqual([
      'user:Fix the cache',
      'assistant',
      'mode:plan',
      'tool',
      'mode:default',
      'assistant',
      'turn-end',
    ]);
    expect(noIndex(live)).toEqual(noIndex(disk));
  });

  it('an id the agent does not know is its default mode; a line with no id makes nothing; ask has its own line', () => {
    const { disk } = play([
      [typed('Go', 0), 'acp', 'prompt'],
      [mode('plan'), 'acp', 'apply'],
      [mode('bypass'), 'acp', 'apply'],
      [{ sessionUpdate: 'current_mode_update' }, 'acp', 'apply'],
      [mode('ask'), 'acp', 'apply'],
      [done, 'xai', 'apply'],
    ]);
    expect(shape(disk)).toEqual(['user:Go', 'mode:plan', 'mode:default', 'mode:ask', 'turn-end']);
  });

  it('a mode line after the last turn end waits for the next prompt, and none shows without one', () => {
    const lines: Array<[object, 'acp' | 'xai', Step['live']]> = [
      [typed('Plan it', 0), 'acp', 'prompt'],
      [mode('plan'), 'acp', 'apply'],
      [done, 'xai', 'apply'],
      [mode('default'), 'acp', 'apply'],
    ];
    expect(shape(play(lines).disk)).toEqual(['user:Plan it', 'mode:plan', 'turn-end']);
    const next = play([...lines, [typed('Now build', 1), 'acp', 'prompt'], [done, 'xai', 'apply']]);
    expect(shape(next.disk)).toEqual([
      'user:Plan it',
      'mode:plan',
      'turn-end',
      'user:Now build',
      'mode:default',
      'turn-end',
    ]);
    expect(shape(next.live)).toEqual(shape(next.disk));
  });

  it('a reducer seeded from disk knows the mode: a repeat makes no line, and modeState counts only moves', () => {
    const { disk } = play([
      [mode('plan'), 'acp', 'none'],
      [typed('Plan it', 0), 'acp', 'prompt'],
      [done, 'xai', 'apply'],
    ]);
    const seeded = new TranscriptReducer();
    seeded.seed(disk, []);
    expect(seeded.modeState()).toEqual({ mode: 'plan', moves: 0 });
    seeded.userPrompt('Keep planning', [], ms(10));
    expect(seeded.apply({ xai: false, update: mode('plan'), meta: { agentTimestampMs: ms(11) } })).toEqual([]);
    const ops = seeded.apply({ xai: false, update: mode('default'), meta: { agentTimestampMs: ms(12) } });
    expect(ops).toEqual([{ op: 'push', entry: { kind: 'plan-mode', ts: ms(12), mode: 'default' } }]);
    expect(seeded.modeState()).toEqual({ mode: 'default', moves: 1 });
    expect(new TranscriptReducer().modeState()).toEqual({ mode: 'default', moves: 0 });
  });

  it('the prompt body names its mode on the request only; no block carries `_meta`', () => {
    const params = promptParams('s1', 'Plan it', [{ kind: 'file', name: 'a.md', text: 'x' }], { mode: 'plan' });
    expect(params._meta).toEqual({ mode: 'plan' });
    expect((params.prompt as Array<Record<string, unknown>>).every((block) => !('_meta' in block))).toBe(true);
    expect('_meta' in promptParams('s1', 'Go')).toBe(false);
  });
});

describe('Send now rows (`_meta.interjection`)', () => {
  const MS = 1791643279000;
  const at = (n: number) => MS + n;
  const stamp = (update: object, n: number, method = 'session/update'): string =>
    JSON.stringify({
      timestamp: 1791643279,
      method,
      params: { sessionId: SID, update, _meta: { eventId: `${SID}-${n}`, agentTimestampMs: at(n) } },
    });
  const typed = {
    sessionUpdate: 'user_message_chunk',
    content: { type: 'text', text: 'Fix it' },
    _meta: { promptIndex: 0 },
  };
  const said = (text: string) => ({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text } });
  // As the agent stores a steer once the model read it (`interjection.rs`): the wrapped text, the typed one in
  // `displayText`, no live copy.
  const stored = {
    sessionUpdate: 'user_message_chunk',
    content: {
      type: 'text',
      text: 'The user sent a message while you were working:\n<user_query>\nuse pnpm\n</user_query>\nIf the user is asking for a response, address the user first. After replying, complete any unfinished tasks from previous turns.',
      _meta: { displayText: 'use pnpm' },
    },
    _meta: { modelId: 'm', interjection: true },
  };
  const done = { sessionUpdate: 'turn_completed', prompt_id: 'p', stop_reason: 'end_turn' };

  it("the host's steer line and the stored chunk give the same entry, inside the answer", () => {
    const disk = parseUpdates(
      [
        stamp(typed, 0),
        stamp(said('Working'), 1),
        stamp(stored, 2),
        stamp(said('Using pnpm'), 3),
        stamp(done, 4, '_x.ai/session/update'),
      ]
        .map((line) => `${line}\n`)
        .join(''),
    ).transcript;
    const reducer = new TranscriptReducer();
    const live: TranscriptEntry[] = [];
    const push = (ops: ReturnType<TranscriptReducer['apply']>): void => {
      for (const op of ops) applyOp(live, [], op);
    };
    push(reducer.userPrompt('Fix it', [], at(0)));
    push(reducer.apply({ xai: false, update: said('Working'), meta: { agentTimestampMs: at(1) } }));
    push(reducer.steer('use pnpm', at(2)));
    push(reducer.apply({ xai: false, update: said('Using pnpm'), meta: { agentTimestampMs: at(3) } }));
    push(reducer.apply({ xai: true, update: done, meta: { agentTimestampMs: at(4) } }));

    const steer = { kind: 'user', ts: at(2), text: 'use pnpm', promptIndex: null, interjection: true };
    expect(disk.find((e) => e.kind === 'user' && e.interjection === true)).toEqual(steer);
    expect(live.find((e) => e.kind === 'user' && e.interjection === true)).toEqual(steer);
    const kinds = (entries: readonly TranscriptEntry[]) =>
      entries.map((e) => (e.kind === 'user' ? `user:${e.text}` : e.kind === 'assistant' ? `said:${e.text}` : e.kind));
    expect(kinds(live)).toEqual(kinds(disk));
    // A steer starts no turn: one prompt, one end.
    expect(kinds(disk)).toEqual(['user:Fix it', 'said:Working', 'user:use pnpm', 'said:Using pnpm', 'turn-end']);
  });

  it('`interject-fallback-` names a turn the agent ran for a steer that missed its answer: cause `message`', () => {
    expect(wakeCause('interject-fallback-0195d3a0-7c1e-7000-8000-000000000001')).toBe('message');
    const reducer = new TranscriptReducer();
    reducer.apply({
      xai: false,
      update: {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text: '' },
        _meta: { hideFromScrollback: true },
      },
      meta: { agentTimestampMs: at(0) },
    });
    reducer.apply({ xai: false, update: said('Read it'), meta: { agentTimestampMs: at(1) } });
    reducer.apply({
      xai: true,
      update: { sessionUpdate: 'turn_completed', prompt_id: 'interject-fallback-0195', stop_reason: 'end_turn' },
      meta: { agentTimestampMs: at(2) },
    });
    expect(reducer.snapshot().entries[0]).toMatchObject({ kind: 'wake', cause: 'message' });
  });
});

describe('a recap (`session_recap`)', () => {
  const recapLine = (summary: unknown, n: number) => xai({ sessionUpdate: 'session_recap', summary, auto: false }, n);
  const none = (n: number) => xai({ sessionUpdate: 'session_recap_unavailable' }, n);
  const done = (n: number) => xai({ sessionUpdate: 'turn_completed', prompt_id: `p${n}`, stop_reason: 'end_turn' }, n);

  // Mutation: hold the recap like a notice (`side`) → it waits for the next prompt and the live list differs.
  it('one entry after the answer it follows, the same from disk and live; the unavailable line and a blank one make none', () => {
    const lines = [
      user('Fix the upload', 0),
      agent('Done.'),
      done(3),
      recapLine('  The upload retry is in place. Next: the docs.  ', 4),
      none(5),
      recapLine('   ', 6),
    ];
    const disk = parseUpdates(`${lines.join('\n')}\n`).transcript;
    expect(disk.map((e) => e.kind)).toEqual(['user', 'assistant', 'turn-end', 'recap']);
    expect(disk[3]).toEqual({
      kind: 'recap',
      ts: 1791643279004,
      text: 'The upload retry is in place. Next: the docs.',
    });
    const reducer = new TranscriptReducer();
    const live: TranscriptEntry[] = [];
    for (const op of reducer.userPrompt('Fix the upload', [], 1791643279001)) applyOp(live, [], op);
    for (const line of lines.slice(1)) {
      const params = (
        JSON.parse(line) as {
          method: string;
          params: { update: Record<string, unknown>; _meta: Record<string, unknown> };
        }
      ).params;
      const xaiLine = (JSON.parse(line) as { method: string }).method !== 'session/update';
      for (const op of reducer.apply({ xai: xaiLine, update: params.update, meta: params._meta }))
        applyOp(live, [], op);
    }
    expect(live.map((e) => (e.kind === 'user' ? { ...e, promptIndex: 0 } : e))).toEqual(disk);
  });

  it('a long recap is cut at 2,000 characters with an ellipsis', () => {
    const text = `${'a'.repeat(1_999)}bc`;
    const [entry] = parseUpdates(`${recapLine(text, 1)}\n`).transcript;
    expect(entry).toEqual({ kind: 'recap', ts: 1791643279001, text: `${'a'.repeat(1_999)}b…` });
  });

  // Mutation: keep every line in `filterRewindLines` → the recap of the removed part stays.
  it('a rewind to before it removes it; one made before the rewind point stays', () => {
    const lines = [
      user('first', 0),
      agent('one'),
      done(3),
      recapLine('Kept recap.', 4),
      user('second', 1),
      agent('two'),
      done(7),
      recapLine('Cut recap.', 8),
    ];
    const kept = parseUpdates(`${[...lines, rewind(1)].join('\n')}\n`).transcript;
    expect(kept.filter((e) => e.kind === 'recap').map((e) => (e.kind === 'recap' ? e.text : ''))).toEqual([
      'Kept recap.',
    ]);
  });

  // Mutation: drop the `recap` branch in `redactEntry` → the start of the key at the cut end shows.
  it('its text passes the redactor, a key cut at its end too', () => {
    const key = 'sk-live-0123456789abcdef';
    const [whole] = parseUpdates(`${recapLine(`The key ${key} was printed.`, 1)}\n`).transcript;
    expect((redactEntry(whole!, [key]) as { text: string }).text).not.toContain(key);
    const [cut] = parseUpdates(`${recapLine(`${'x'.repeat(1_990)} ${key}`, 2)}\n`).transcript;
    const shown = (redactEntry(cut!, [key]) as { text: string }).text;
    expect(shown.endsWith('…')).toBe(true);
    expect(shown).not.toContain('sk-live');
  });
});

describe('a goal (`goal_updated`)', () => {
  const MS = 1791643279000;
  const ms = (n: number) => MS + n;
  /** `stored`: on disk and live; `tick`: live only; `disk`: stored only; `typed`: stored as given, live the typed text. */
  type How = 'stored' | 'tick' | 'disk' | { typed: string };
  type Goal = Extract<TranscriptEntry, { kind: 'goal' }>;
  const OBJECTIVE = 'Fix the flaky cache tests';
  const goal = (fields: object, id = 'g1', objective = OBJECTIVE) => ({
    sessionUpdate: 'goal_updated',
    goal_id: id,
    objective,
    status: 'active',
    phase: 'executing',
    tokens_used: 0,
    elapsed_ms: 0,
    total_deliverables: 0,
    completed_deliverables: 0,
    total_worker_rounds: 0,
    total_verify_rounds: 0,
    ...fields,
  });
  const cleared = {
    sessionUpdate: 'goal_updated',
    goal_id: '',
    objective: '',
    status: 'cleared',
    phase: 'idle',
    tokens_used: 0,
    elapsed_ms: 0,
  };
  // The agent's goal rules (`goal_rules.md`) as it stores them for `/goal <objective>` and a resume that runs.
  const rules = (objective: string, resume: boolean) => {
    const body = `A goal has been set: ${objective}\n\nYou are working directly on this goal across multiple turns.`;
    return resume
      ? `<system-reminder>\n${body}\nContinue working now.\n</system-reminder>`
      : `<system-reminder>\n${body}\nStart now.\n</system-reminder>\n\n`;
  };
  const prompt = (text: string, promptIndex: number) => ({
    sessionUpdate: 'user_message_chunk',
    content: { type: 'text', text },
    _meta: { promptIndex },
  });
  const said = (text: string) => ({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text } });
  const ended = (stop: string) => ({ sessionUpdate: 'turn_completed', prompt_id: 'p', stop_reason: stop });
  const XAI = new Set(['goal_updated', 'turn_completed']);

  /** Plays `steps`; after each stored goal line, the disk read of the lines so far next to the live list. */
  function play(steps: ReadonlyArray<[Record<string, unknown>, How]>) {
    const reducer = new TranscriptReducer();
    const live: TranscriptEntry[] = [];
    const stored: string[] = [];
    const checks: Array<{ disk: TranscriptEntry[]; live: TranscriptEntry[] }> = [];
    const opsAt: number[] = [];
    steps.forEach(([update, how], n) => {
      const xaiLine = XAI.has(String(update.sessionUpdate));
      if (how !== 'tick') {
        stored.push(
          JSON.stringify({
            timestamp: 1791643279,
            method: xaiLine ? '_x.ai/session/update' : 'session/update',
            params: { sessionId: SID, update, _meta: { eventId: `${SID}-${n}`, agentTimestampMs: ms(n) } },
          }),
        );
      }
      const ops =
        how === 'disk'
          ? []
          : typeof how === 'object'
            ? reducer.userPrompt(how.typed, [], ms(n))
            : reducer.apply({ xai: xaiLine, update, meta: { agentTimestampMs: ms(n) } });
      opsAt.push(ops.length);
      for (const op of ops) applyOp(live, [], op);
      if (how === 'stored' && update.sessionUpdate === 'goal_updated') {
        checks.push({ disk: parseUpdates(`${stored.join('\n')}\n`).transcript, live: structuredClone(live) });
      }
    });
    return { live, disk: parseUpdates(`${stored.join('\n')}\n`).transcript, checks, opsAt };
  }
  const goals = (entries: readonly TranscriptEntry[]): Goal[] => entries.filter((e): e is Goal => e.kind === 'goal');
  /** What the page shows of each goal, and its clock: the seconds at `asOf`, counted on while it runs. */
  const shown = (entries: readonly TranscriptEntry[]) =>
    goals(entries).map((g) => ({
      ...g,
      tokensUsed: Math.floor(g.tokensUsed / 1_000),
      clock: g.elapsedMs - (g.asOf - MS),
      elapsedMs: 0,
      asOf: 0,
    }));
  const typedTexts = (entries: readonly TranscriptEntry[]) =>
    entries.flatMap((e) => (e.kind === 'user' && e.interjection !== true ? [e.text] : []));

  // Mutation: drop the `goalShown` compare (op at every line) → a tick that moved no shown field sends an op.
  // Mutation: register no entry by id (`goalEntries`) → the pause and the resume make entries of their own.
  it('set → ticks → Stop → resume → done is one entry; a line that moves no shown field sends nothing', () => {
    const { live, disk, checks, opsAt } = play([
      [goal({ last_event: 'goal_created' }), 'stored'],
      [prompt(rules(OBJECTIVE, false), 0), { typed: `/goal ${OBJECTIVE}` }],
      [goal({ tokens_used: 400, elapsed_ms: 2, live_phase: 'worker' }), 'tick'],
      [goal({ tokens_used: 1_500, elapsed_ms: 3 }), 'tick'],
      [said('Working.'), 'stored'],
      [goal({ tokens_used: 1_700, elapsed_ms: 5, last_event: 'worker_completed' }), 'stored'],
      [ended('cancelled'), 'stored'],
      [goal({ status: 'user_paused', phase: 'idle', tokens_used: 1_700, elapsed_ms: 7 }), 'stored'],
      [goal({ tokens_used: 1_700, elapsed_ms: 7, last_event: 'goal_resumed' }), 'stored'],
      [prompt(rules(OBJECTIVE, true), 1), { typed: '/goal resume' }],
      [goal({ tokens_used: 4_200, elapsed_ms: 9_000, verifying_completion: true }), 'stored'],
      [goal({ status: 'complete', phase: 'idle', tokens_used: 4_400, elapsed_ms: 9_500 }), 'stored'],
      [ended('end_turn'), 'stored'],
    ]);
    expect(opsAt[2]).toBe(0);
    expect(opsAt[3]).toBe(1);
    expect(opsAt[5]).toBe(0);
    expect(goals(live)).toEqual([
      {
        kind: 'goal',
        ts: ms(0),
        goalId: 'g1',
        objective: OBJECTIVE,
        status: 'complete',
        phase: 'idle',
        tokensUsed: 4_400,
        elapsedMs: 9_500,
        asOf: ms(11),
      },
    ]);
    // The stored rules read as what was typed, and the row sits after the message that set the goal.
    expect(typedTexts(disk)).toEqual([`/goal ${OBJECTIVE}`, '/goal resume']);
    expect(disk.slice(0, 2).map((e) => e.kind)).toEqual(['user', 'goal']);
    expect(live.slice(0, 2).map((e) => e.kind)).toEqual(['user', 'goal']);
    // The page's copy moves only with what it shows; the disk read shows the same at every stored line.
    for (const { disk: d, live: l } of checks) expect(shown(d)).toEqual(shown(l));
    expect(checks).toHaveLength(6);
  });

  // Mutation: compare the objective without its line end in `goalPrompt` → a longer objective's rules read as this one.
  it('a budget reads back as `--budget`; rules of another objective, or with no active goal, stay as stored', () => {
    const { disk } = play([
      [goal({ token_budget: 200_000 }, 'g2', 'Ship it'), 'stored'],
      [prompt(rules('Ship it', false), 0), { typed: '/goal Ship it --budget 200000' }],
      [prompt(rules('Ship it now', false), 1), { typed: 'x' }],
      [goal({ status: 'user_paused', phase: 'idle', token_budget: 200_000 }, 'g2', 'Ship it'), 'stored'],
      [prompt(rules('Ship it', true), 2), { typed: 'y' }],
    ]);
    expect(typedTexts(disk)).toEqual([
      '/goal Ship it --budget 200000',
      rules('Ship it now', false),
      rules('Ship it', true),
    ]);
  });

  // Mutation: let a cleared line close any last goal (no `complete` check) → the done goal reads cleared.
  it('cleared closes the open goal; a second goal is a second entry; a done goal stays done', () => {
    const { live, checks } = play([
      [goal({ status: 'complete', phase: 'idle', tokens_used: 900 }), 'stored'],
      [cleared, 'stored'],
      [goal({}, 'g2', 'Ship it'), 'stored'],
      [prompt(rules('Ship it', false), 1), { typed: '/goal Ship it' }],
      [goal({ status: 'user_paused', phase: 'idle' }, 'g2', 'Ship it'), 'stored'],
      [prompt('/goal clear', 2), 'disk'],
      [cleared, 'stored'],
      [said('Goal cleared.'), 'stored'],
    ]);
    expect(goals(live).map((g) => [g.goalId, g.status, g.phase])).toEqual([
      ['g1', 'complete', 'idle'],
      ['g2', 'cleared', 'idle'],
    ]);
    for (const { disk: d, live: l } of checks) expect(shown(d)).toEqual(shown(l));
  });

  it('the agent’s reason is cut at 300 characters; the cut passes the redactor', () => {
    const key = 'sk-live-0123456789abcdef';
    const reason = `${'x'.repeat(290)} ${key}`;
    const { disk } = play([
      [prompt('Fix it', 0), { typed: 'Fix it' }],
      [goal({ status: 'blocked', phase: 'idle', pause_message: reason }), 'stored'],
    ]);
    const [entry] = goals(disk);
    expect(entry!.pauseMessage).toBe(`${reason.slice(0, 300)}…`);
    const shown = (redactEntry(entry!, [key]) as Goal).pauseMessage!;
    expect(shown.endsWith('…')).toBe(true);
    expect(shown).not.toContain('sk-li');
  });
});
