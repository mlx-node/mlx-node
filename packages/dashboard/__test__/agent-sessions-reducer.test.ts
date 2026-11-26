/**
 * The shared transcript reducer as the live stream uses it: ops replayed onto an empty list equal the snapshot, MCP
 * results become plain output text, and recorded new / load traffic yields no `model` entry.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import { mcpOutputText } from '../src/agent-sessions/mcp-output.js';
import {
  filterRewindLines,
  parseUpdates,
  splitCompleteLines,
  TranscriptReducer,
  WAKE_PROMPT_ID,
  wakeCause,
  type UpdateLine,
} from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry, TurnMetric } from '../src/agent-sessions/types.js';
import { liveFilter } from '../src/chat/agent/live-filter.js';
import type { TranscriptOp } from '../src/chat/types.js';
import { splitTurns } from '../ui/src/lib/chat-turns.js';

const FIXTURES = fileURLToPath(new URL('./fixtures/', import.meta.url));
const RECORDED = join(FIXTURES, 'chat-scenarios/recorded');

type Json = Record<string, unknown>;

function replay(ops: readonly TranscriptOp[]): {
  entries: TranscriptEntry[];
  turns: TurnMetric[];
  contextUsed: number | null;
} {
  const entries: TranscriptEntry[] = [];
  const turns: TurnMetric[] = [];
  let contextUsed: number | null = null;
  for (const op of ops) {
    if (op.op === 'push') entries.push(op.entry);
    else if (op.op === 'set') entries[op.index] = op.entry;
    else if (op.op === 'text') {
      const e = entries[op.index]!;
      if (e.kind !== 'assistant' && e.kind !== 'thinking') throw new Error(`text op on ${e.kind}`);
      entries[op.index] = { ...e, text: e.text + op.append };
    } else if (op.op === 'context') contextUsed = op.used;
    else turns.push(op.metric);
  }
  return { entries, turns, contextUsed };
}

/** `endTs` of a text entry moves with every chunk; `text` ops do not carry it. */
function withoutTextEnd(entries: readonly TranscriptEntry[]): unknown[] {
  return entries.map((e) => {
    if (e.kind !== 'assistant' && e.kind !== 'thinking') return e;
    const { endTs: _endTs, ...rest } = e as typeof e & { endTs?: number };
    return rest;
  });
}

function diskLine(line: string): UpdateLine | null {
  let obj: unknown;
  try {
    obj = JSON.parse(line);
  } catch {
    return null;
  }
  if (typeof obj !== 'object' || obj === null) return null;
  const o = obj as Json;
  const params = ('params' in o ? o.params : o) as Json | null;
  if (typeof params !== 'object' || params === null || typeof params.update !== 'object' || params.update === null)
    return null;
  const update = params.update as Json;
  if (typeof update.sessionUpdate !== 'string') return null;
  return {
    xai: o.method === '_x.ai/session/update',
    update,
    meta: typeof params._meta === 'object' && params._meta !== null ? (params._meta as Json) : undefined,
  };
}

function updatesFiles(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) out.push(...updatesFiles(path));
    else if (name === 'updates.jsonl') out.push(path);
  }
  return out;
}

interface Recorded {
  dir: 'in' | 'out';
  msg: Json;
}

function recorded(name: string): Recorded[] {
  return readFileSync(join(RECORDED, name), 'utf8')
    .split('\n')
    .filter((l) => l !== '')
    .map((l) => JSON.parse(l) as Recorded);
}

/** Drives the recorded traffic through the live filter and the reducer as the Chat host and stream do. */
function live(traffic: Recorded[]): { reducer: TranscriptReducer; ops: TranscriptOp[] } {
  const reducer = new TranscriptReducer({ now: () => 1 });
  const ops: TranscriptOp[] = [];
  let promptId: unknown = null;
  let modelId = 'qwen3.5-4b-nvfp4-mtp';
  for (const { dir, msg } of traffic) {
    if (dir === 'out' && msg.method === 'session/prompt') {
      promptId = msg.id;
      const block = ((msg.params as Json).prompt as Json[])[0]!;
      ops.push(...reducer.userPrompt(block.text as string, [], 1));
      continue;
    }
    if (dir === 'in' && msg.id === promptId && msg.method === undefined) {
      promptId = null;
      continue;
    }
    const method = msg.method;
    if (dir !== 'in' || typeof method !== 'string') continue;
    if (method !== 'session/update' && method !== '_x.ai/session_notification') continue;
    const params = msg.params as Json;
    const line: UpdateLine = {
      xai: method !== 'session/update',
      update: params.update as Json,
      meta: params._meta as Json | undefined,
    };
    if (line.update.sessionUpdate === 'session_info_update') continue;
    if (liveFilter(line, { turnRunning: promptId !== null, modelId }) === 'drop') continue;
    if (line.xai && line.update.sessionUpdate === 'model_changed') modelId = line.update.model_id as string;
    ops.push(...reducer.apply(line));
  }
  return { reducer, ops };
}

describe('mcpOutputText', () => {
  it('reads exactly the Okay shape', () => {
    expect(
      mcpOutputText({
        type: 'MCP',
        tool_name: 'web_search',
        server_name: 'mlx_web',
        output: { OkayOutput: 'WEB_SEARCH x' },
      }),
    ).toEqual({ server: 'mlx_web', tool: 'web_search', text: 'WEB_SEARCH x' });
    for (const bad of [
      null,
      'text',
      { type: 'Bash', tool_name: 't', server_name: 's', output: { OkayOutput: 'x' } },
      { type: 'MCP', tool_name: 't', server_name: 's', output: { ErrorOutput: 'x' } },
      { type: 'MCP', tool_name: 't', output: { OkayOutput: 'x' } },
      { type: 'MCP', tool_name: 1, server_name: 's', output: { OkayOutput: 'x' } },
      { type: 'MCP', tool_name: 't', server_name: 's', output: { OkayOutput: 3 } },
    ]) {
      expect(mcpOutputText(bad)).toBeNull();
    }
  });
});

describe('TranscriptReducer, MCP output rule', () => {
  const update = (fields: Json): UpdateLine => ({
    xai: false,
    update: { sessionUpdate: 'tool_call_update', toolCallId: 't1', ...fields },
    meta: undefined,
  });

  it('stores an MCP result as its plain text, not JSON, even past the field cap', () => {
    const long = `WEB_PAGE provider=exa url=https://a.com\ntitle: t\n\n${'"x'.repeat(20_000)}`;
    const r = new TranscriptReducer({ now: () => 5 });
    r.apply({
      xai: false,
      update: { sessionUpdate: 'tool_call', toolCallId: 't1', title: 'use_tool' },
      meta: undefined,
    });
    r.apply(
      update({
        status: 'completed',
        rawOutput: { type: 'MCP', tool_name: 'web_fetch', server_name: 'mlx_web', output: { OkayOutput: long } },
      }),
    );
    const tool = r.snapshot().entries[0] as Extract<TranscriptEntry, { kind: 'tool' }>;
    expect(tool.output?.startsWith('WEB_PAGE provider=exa url=https://a.com\ntitle: t\n\n"x"x')).toBe(true);
    expect(tool.output).toBe(long.slice(0, 16 * 1024));
    expect(tool.truncated).toBe(true);
  });

  it('content text wins over rawOutput; a non-Okay rawOutput stays JSON text', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    r.apply({ xai: false, update: { sessionUpdate: 'tool_call', toolCallId: 't1', title: 'x' }, meta: undefined });
    r.apply(
      update({
        content: [{ type: 'content', content: { type: 'text', text: 'denied' } }],
        rawOutput: { type: 'MCP', tool_name: 'a', server_name: 'b', output: { OkayOutput: 'ok' } },
      }),
    );
    expect((r.snapshot().entries[0] as { output?: string }).output).toBe('denied');
    const r2 = new TranscriptReducer({ now: () => 5 });
    r2.apply({ xai: false, update: { sessionUpdate: 'tool_call', toolCallId: 't1', title: 'x' }, meta: undefined });
    r2.apply(update({ rawOutput: { type: 'MCP', tool_name: 'a', server_name: 'b', output: { ErrorOutput: 'e' } } }));
    expect((r2.snapshot().entries[0] as { output?: string }).output).toBe(
      JSON.stringify({ type: 'MCP', tool_name: 'a', server_name: 'b', output: { ErrorOutput: 'e' } }, null, 2),
    );
  });
});

describe('recorded Chat traffic', () => {
  for (const [file, shape] of [
    ['f-chat-use-tool.jsonl', 'use_tool'],
    ['f-chat-direct.jsonl', 'mlx_web__web_search'],
  ] as const) {
    it(`${file}: the web result arrives as plain tool output, no model entry, ops equal the snapshot`, () => {
      const { reducer, ops } = live(recorded(file));
      const { entries, turns } = reducer.snapshot();
      expect(entries.some((e) => e.kind === 'model')).toBe(false);
      expect(entries[0]).toMatchObject({ kind: 'user' });
      const tools = entries.filter((e): e is Extract<TranscriptEntry, { kind: 'tool' }> => e.kind === 'tool');
      const web = tools.filter((t) => t.output?.startsWith('WEB_SEARCH provider=exa results=2\n[1] FCHK-RESULT-ONE'));
      expect(web.length).toBeGreaterThan(0);
      expect(web[0]!.name).toBe(shape);
      expect(web[0]!.status).toBe('completed');
      expect(entries.some((e) => e.kind === 'turn-end')).toBe(true);
      expect(turns.length).toBeGreaterThan(0);
      const replayed = replay(ops);
      expect(withoutTextEnd(replayed.entries)).toEqual(withoutTextEnd(entries));
      expect(replayed.turns).toEqual(turns);
    });
  }

  it('a new and a loaded conversation start with no model entry (f-u12)', () => {
    const traffic = recorded('f-u12-new-and-load.jsonl');
    const modelLines = traffic.filter(
      (t) =>
        (t.msg.params as Json | undefined)?.update !== undefined &&
        ((t.msg.params as Json).update as Json).sessionUpdate === 'model_changed',
    );
    expect(modelLines.length).toBe(2);
    const { reducer } = live(traffic);
    expect(reducer.snapshot().entries.filter((e) => e.kind === 'model')).toEqual([]);
  });
});

describe('op stream equals the snapshot for every stored session fixture', () => {
  for (const file of updatesFiles(join(FIXTURES, 'grok-sessions'))) {
    it(file.slice(FIXTURES.length), () => {
      const reducer = new TranscriptReducer({ now: () => 7 });
      const ops: TranscriptOp[] = [];
      for (const raw of filterRewindLines(splitCompleteLines(readFileSync(file, 'utf8')))) {
        const line = diskLine(raw);
        if (line !== null) ops.push(...reducer.apply(line));
      }
      const snap = reducer.snapshot();
      const replayed = replay(ops);
      const parsed = parseUpdates(readFileSync(file, 'utf8'));
      expect(withoutTextEnd(replayed.entries)).toEqual(withoutTextEnd(snap.entries));
      expect(replayed.turns).toEqual(snap.turns);
      expect(snap.entries.map((e) => e.kind)).toEqual(parsed.transcript.map((e) => e.kind));
      expect(replayed.contextUsed).toBe(reducer.context());
      expect(parsed.contextUsed).toBe(reducer.context());
    });
  }
});

describe('seed', () => {
  it('live ops index after the disk entries of a cold conversation', () => {
    const r = new TranscriptReducer({ now: () => 9 });
    r.seed(
      [
        { kind: 'user', ts: 1, text: 'old', promptIndex: 0 },
        { kind: 'assistant', ts: 2, text: 'old answer' },
      ],
      [],
    );
    const ops = [
      ...r.userPrompt('new', [], 9),
      ...r.apply({
        xai: false,
        update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'a' } },
        meta: undefined,
      }),
      ...r.apply({
        xai: false,
        update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'b' } },
        meta: undefined,
      }),
    ];
    expect(ops).toEqual([
      { op: 'push', entry: { kind: 'user', ts: 9, text: 'new', promptIndex: null } },
      { op: 'push', entry: { kind: 'assistant', ts: 9, text: 'a', endTs: 9 } },
      { op: 'text', index: 3, append: 'b', endTs: 9 },
    ]);
    expect(r.snapshot().entries.map((e) => e.kind)).toEqual(['user', 'assistant', 'user', 'assistant']);
  });
});

describe('context use: params._meta.totalTokens', () => {
  /** The shape of a real mlx-agent v0.3.3 record; the counts are those of one real turn. */
  function line(sessionUpdate: string, totalTokens?: number): UpdateLine {
    const update: Json = sessionUpdate.endsWith('_chunk')
      ? { sessionUpdate, content: { type: 'text', text: 'x' } }
      : { sessionUpdate, toolCallId: 't1', title: 'use_tool', status: 'pending' };
    return {
      xai: false,
      update,
      meta: totalTokens === undefined ? { agentTimestampMs: 5 } : { totalTokens, agentTimestampMs: 5 },
    };
  }
  const TURN: Array<[string, number | undefined]> = [
    ['user_message_chunk', undefined],
    ['agent_thought_chunk', 535],
    ['tool_call', 1929],
    ['tool_call_update', 1929],
    ['tool_call_update', 3602],
    ['agent_thought_chunk', 11971],
    ['agent_message_chunk', 11971],
  ];

  it('a live update with totalTokens sets the used count; one without it keeps the last value', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    const seen: Array<number | null> = [];
    const contextOps: TranscriptOp[] = [];
    for (const [kind, total] of TURN) {
      contextOps.push(...r.apply(line(kind, total)).filter((op) => op.op === 'context'));
      seen.push(r.context());
    }
    expect(seen).toEqual([null, 535, 1929, 1929, 3602, 11971, 11971]);
    expect(contextOps).toEqual([
      { op: 'context', used: 535 },
      { op: 'context', used: 1929 },
      { op: 'context', used: 3602 },
      { op: 'context', used: 11971 },
    ]);
    r.apply({ xai: true, update: { sessionUpdate: 'turn_completed', stop_reason: 'end_turn' }, meta: undefined });
    r.userPrompt('next', [], 6);
    r.apply(line('user_message_chunk'));
    expect(r.context()).toBe(11971);
  });

  it('the disk reader reads the same value from the stored lines', () => {
    const text = TURN.map(([kind, total]) => {
      const l = line(kind, total);
      return JSON.stringify({ timestamp: 1, method: 'session/update', params: { update: l.update, _meta: l.meta } });
    }).join('\n');
    expect(parseUpdates(`${text}\n`).contextUsed).toBe(11971);
    expect(
      parseUpdates(`${JSON.stringify({ params: { update: line('user_message_chunk').update } })}\n`).contextUsed,
    ).toBe(null);
  });

  it('a seeded value stays until a line carries a new one, also a smaller one', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    r.seed([], [], 11971);
    expect(r.apply(line('agent_message_chunk'))).not.toContainEqual(expect.objectContaining({ op: 'context' }));
    expect(r.context()).toBe(11971);
    expect(r.apply(line('agent_message_chunk', 900))).toContainEqual({ op: 'context', used: 900 });
    expect(r.context()).toBe(900);
  });
});

describe('compaction marker: auto_compact_* lines', () => {
  /** The four updates as mlx-agent v0.3.3 sends them (`extensions/notification.rs`), with its notification meta. */
  const xai = (update: Json, at: number): UpdateLine => ({
    xai: true,
    update,
    meta: { eventId: `e${at}`, agentTimestampMs: at },
  });
  const STARTED = {
    sessionUpdate: 'auto_compact_started',
    tokens_used: 222_822,
    context_window: 262_144,
    percentage: 85,
    reason: 'Context window 85% full',
  };
  const COMPLETED = {
    sessionUpdate: 'auto_compact_completed',
    tokens_before: 222_822,
    tokens_after: 31_204,
    elapsed_ms: 1_221_000,
    summary_preview: null,
  };
  const FAILED = {
    sessionUpdate: 'auto_compact_failed',
    error: "it'll retry on the next turn, or start a new session using /new.\nstream idle timeout after 600s",
  };
  const CANCELLED = { sessionUpdate: 'auto_compact_cancelled', reason: 'user_cancelled' };

  it('started makes no entry; each end makes one marker where it happened', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    expect(r.apply(xai(STARTED, 100))).toEqual([]);
    expect(r.apply(xai(COMPLETED, 200))).toEqual([
      {
        op: 'push',
        entry: { kind: 'compaction', ts: 200, outcome: 'done', tokensBefore: 222_822, tokensAfter: 31_204 },
      },
    ]);
    expect(r.apply(xai(FAILED, 300))).toEqual([
      { op: 'push', entry: { kind: 'compaction', ts: 300, outcome: 'failed', retry: true } },
    ]);
    expect(
      r.apply(xai({ sessionUpdate: 'auto_compact_failed', error: 'this conversation is too large to compact.' }, 350)),
    ).toEqual([{ op: 'push', entry: { kind: 'compaction', ts: 350, outcome: 'failed' } }]);
    expect(r.apply(xai(CANCELLED, 400))).toEqual([
      { op: 'push', entry: { kind: 'compaction', ts: 400, outcome: 'stopped' } },
    ]);
    // Manual `/compact` sends only the end, with no time.
    expect(r.apply(xai({ sessionUpdate: 'auto_compact_completed', tokens_after: 9_000 }, 500))).toEqual([
      { op: 'push', entry: { kind: 'compaction', ts: 500, outcome: 'done', tokensAfter: 9_000 } },
    ]);
    // On the ACP method it is not the agent's update.
    expect(r.apply({ xai: false, update: COMPLETED, meta: undefined })).toEqual([]);
  });

  it('the marker is rebuilt from updates.jsonl, at its place in the turn', () => {
    const stored = (method: string, update: Json, at: number, meta: Json = {}): string =>
      JSON.stringify({
        timestamp: Math.floor(at / 1000),
        method,
        params: { sessionId: 's', update, _meta: { eventId: `e${at}`, agentTimestampMs: at, ...meta } },
      });
    const text = [
      stored('session/update', { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: 'go' } }, 1_000),
      stored('session/update', { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'a' } }, 2_000, {
        totalTokens: 222_822,
      }),
      stored('_x.ai/session/update', STARTED, 3_000),
      stored('_x.ai/session/update', COMPLETED, 4_000),
      stored('session/update', { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'b' } }, 5_000, {
        totalTokens: 31_204,
      }),
    ].join('\n');
    const parsed = parseUpdates(`${text}\n`);
    expect(parsed.transcript.map((e) => e.kind)).toEqual(['user', 'assistant', 'compaction', 'assistant']);
    expect(parsed.transcript[2]).toEqual({
      kind: 'compaction',
      ts: 4_000,
      outcome: 'done',
      tokensBefore: 222_822,
      tokensAfter: 31_204,
    });
    // The context ring drops through the existing totalTokens path: the next line after the compaction carries it.
    expect(parsed.contextUsed).toBe(31_204);
  });

  // Mutation: drop the `tokensAfter < contextUsed` step in the reducer's auto_compact_completed case → stays 222,822.
  it('live, the context takes a lower tokens_after at the compaction end, before any later line', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    r.seed([], [], 222_822);
    r.apply(xai(STARTED, 100));
    expect(r.apply(xai(COMPLETED, 200))).toEqual([
      { op: 'context', used: 31_204 },
      {
        op: 'push',
        entry: { kind: 'compaction', ts: 200, outcome: 'done', tokensBefore: 222_822, tokensAfter: 31_204 },
      },
    ]);
    expect(r.context()).toBe(31_204);
  });

  it('a tokens_after not below the known count, or with no count known, leaves the context as it is', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    r.seed([], [], 235_939);
    const same = { ...COMPLETED, tokens_before: 235_952, tokens_after: 235_939 };
    expect(r.apply(xai(same, 200)).filter((op) => op.op === 'context')).toEqual([]);
    expect(r.apply(xai({ ...same, tokens_after: 240_000 }, 300)).filter((op) => op.op === 'context')).toEqual([]);
    expect(r.context()).toBe(235_939);
    const fresh = new TranscriptReducer({ now: () => 5 });
    expect(fresh.apply(xai(COMPLETED, 200)).filter((op) => op.op === 'context')).toEqual([]);
    expect(fresh.context()).toBeNull();
  });

  it('live, a later line with totalTokens sets the context again', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    r.seed([], [], 222_822);
    r.apply(xai(STARTED, 100));
    r.apply(xai(COMPLETED, 200));
    const ops = r.apply({
      xai: false,
      update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'x' } },
      meta: { totalTokens: 31_580, agentTimestampMs: 300 },
    });
    expect(ops).toContainEqual({ op: 'context', used: 31_580 });
    expect(r.context()).toBe(31_580);
  });
});

describe('wake rows: the host line and the agent prompt', () => {
  const xai = (update: Json): UpdateLine => ({ xai: true, update, meta: undefined });
  const hidden = (text: string, meta: Json): UpdateLine => ({
    xai: false,
    update: {
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text },
      _meta: { hideFromScrollback: true, ...meta },
    },
    meta: undefined,
  });
  const RUN = {
    sessionUpdate: 'workflow_updated',
    run_id: 'wf_1',
    name: 'deep-research',
    status: 'complete',
    revision: 5,
  };
  const wakes = (r: TranscriptReducer) => r.snapshot().entries.filter((e) => e.kind === 'wake');
  const WOKEN = {
    runId: 'wf_1',
    revision: 5,
    name: 'deep-research',
    snapshot: { status: 'complete', agentsUsed: 0, agentBudget: null, agentsFailed: 0, elapsedMs: 0 },
  };

  // Mutation: read no `mlxWakePromptId` in the hidden-chunk case → cause 'unknown' until turn_completed.
  it("the host line's prompt id names the run at once", () => {
    const r = new TranscriptReducer({ now: () => 5 });
    r.apply(xai(RUN));
    r.apply(hidden('', { [WAKE_PROMPT_ID]: 'workflow-completed-wf_1-5' }));
    expect(wakes(r)).toEqual([{ kind: 'wake', ts: 5, cause: 'workflow', workflow: WOKEN, text: '' }]);
  });

  // Mutation: drop the merge into the host's empty row → two wake rows.
  it("the agent's own prompt after the host line fills that row; a later wake is a new row", () => {
    const r = new TranscriptReducer({ now: () => 5 });
    r.apply(xai(RUN));
    r.apply(hidden('', { [WAKE_PROMPT_ID]: 'workflow-completed-wf_1-5' }));
    r.apply(hidden('A workflow finished.', { promptIndex: 1 }));
    r.apply(hidden('Read its report.', { promptIndex: 1 }));
    expect(wakes(r)).toEqual([
      {
        kind: 'wake',
        ts: 5,
        cause: 'workflow',
        workflow: WOKEN,
        text: 'A workflow finished.\nRead its report.',
      },
    ]);
    r.apply(xai({ sessionUpdate: 'turn_completed', prompt_id: 'workflow-completed-wf_1-5', stop_reason: 'end_turn' }));
    r.apply(hidden('', {}));
    r.apply(hidden('A task finished.', { promptIndex: 2 }));
    expect(wakes(r).map((e) => e.text)).toEqual(['A workflow finished.\nRead its report.', 'A task finished.']);
  });
});

describe('wake cause (H-b): what ended names the wake row', () => {
  const plain = (update: Json, meta?: Json): UpdateLine => ({ xai: false, update, meta });
  const xai = (update: Json): UpdateLine => ({ xai: true, update, meta: undefined });
  /** A command (or a monitor) the model started in the background, as the agent sends it (fake-agent `backgroundStep`). */
  const started = (n: number, label: string, monitor = false): UpdateLine[] => [
    plain({
      sessionUpdate: 'tool_call',
      toolCallId: `call-${n}`,
      title: monitor ? 'monitor' : 'run_terminal_command',
      rawInput: { command: `cmd-${n}`, description: label, ...(monitor ? {} : { is_background: true }) },
    }),
    xai({
      sessionUpdate: 'task_backgrounded',
      tool_call_id: `call-${n}`,
      task_id: `task-${n}`,
      command: `cmd-${n}`,
      ...(monitor ? { monitor_description: label } : { description: label }),
    }),
    plain({ sessionUpdate: 'tool_call_update', toolCallId: `call-${n}`, status: 'completed' }),
  ];
  /** grok `TaskCompleted { task_snapshot, will_wake }`. */
  const ended = (n: number, willWake: boolean, snapshot: Json = {}): UpdateLine =>
    xai({
      sessionUpdate: 'task_completed',
      task_snapshot: {
        task_id: `task-${n}`,
        command: `cmd-${n}`,
        kind: 'bash',
        completed: true,
        block_waited: false,
        explicitly_killed: false,
        kill_result_delivered: false,
        ...snapshot,
      },
      will_wake: willWake,
    });
  const typed = (index: number, text: string): UpdateLine =>
    plain({ sessionUpdate: 'user_message_chunk', content: { type: 'text', text }, _meta: { promptIndex: index } });
  const end = (promptId: string): UpdateLine =>
    xai({ sessionUpdate: 'turn_completed', prompt_id: promptId, stop_reason: 'end_turn' });
  /** The host's empty wake line (live only), with the id it expects for the wake. */
  const hostLine = (promptId: string | null): UpdateLine =>
    plain({
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: '' },
      _meta: { hideFromScrollback: true, ...(promptId === null ? {} : { [WAKE_PROMPT_ID]: promptId }) },
    });
  /** The agent's own wake prompt: stored, never sent live. */
  const agentPrompt = (index: number): UpdateLine =>
    plain({
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: 'A background task finished.' },
      _meta: { hideFromScrollback: true, promptIndex: index },
    });
  const answer = plain({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Done.' } });
  const wakes = (entries: readonly TranscriptEntry[]) =>
    entries.flatMap((e) => (e.kind === 'wake' ? [{ ...e, text: '' }] : []));
  const stored = (lines: readonly UpdateLine[]): TranscriptEntry[] =>
    parseUpdates(
      lines
        .map((l) =>
          JSON.stringify({
            method: l.xai ? '_x.ai/session/update' : 'session/update',
            params: { sessionId: 's', update: l.update, ...(l.meta === undefined ? {} : { _meta: l.meta }) },
          }),
        )
        .join('\n') + '\n',
    ).transcript;

  it('every prefix of the table; monitor-completed- is a task, not a monitor report', () => {
    expect(wakeCause('task-completed-t1')).toBe('task');
    expect(wakeCause('bash-completed-t1')).toBe('task');
    expect(wakeCause('monitor-completed-t1')).toBe('task');
    expect(wakeCause('monitor-t1-0193f1e2')).toBe('notifications');
    expect(wakeCause('monitor-user-start-drain-0193f1e2')).toBe('notifications');
    expect(wakeCause('notifications-0193f1e2')).toBe('notifications');
    expect(wakeCause('plan-resume-1764000000000')).toBe('unknown');
  });

  // Mutation: drop the 'task' branch of applyWake → no `task` on the row.
  it('a task-completed- wake names the command, or the monitor, from its first line; live and disk agree', () => {
    for (const monitor of [false, true]) {
      const before = [typed(0, 'start it'), ...started(1, 'Start the dev server', monitor), end('p0')];
      const kind = monitor ? 'monitor' : 'command';
      const r = new TranscriptReducer({ now: () => 7 });
      const ops: TranscriptOp[] = [];
      for (const line of [
        ...before,
        ended(1, true, { kind: monitor ? 'monitor' : 'bash' }),
        hostLine('task-completed-task-1'),
      ])
        ops.push(...r.apply(line));
      const task = { id: 'task-1', kind, label: 'Start the dev server' };
      expect(wakes(r.snapshot().entries)).toEqual([{ kind: 'wake', ts: 7, cause: 'task', task, text: '' }]);
      for (const line of [answer, end('task-completed-task-1')]) ops.push(...r.apply(line));
      expect(replay(ops).entries).toEqual(r.snapshot().entries);
      const disk = stored([
        ...before,
        ended(1, true, { kind: monitor ? 'monitor' : 'bash' }),
        agentPrompt(1),
        answer,
        end('task-completed-task-1'),
      ]);
      expect(wakes(disk).map(({ ts: _ts, ...e }) => e)).toEqual(
        wakes(r.snapshot().entries).map(({ ts: _ts, ...e }) => e),
      );
    }
  });

  it('the queued names bash-completed- and monitor-completed- link the task too; an unlinked task is named by its end line', () => {
    const r = new TranscriptReducer({ now: () => 7 });
    for (const line of [...started(1, 'Run the tests'), ended(1, false), hostLine('bash-completed-task-1')])
      r.apply(line);
    expect(wakes(r.snapshot().entries)).toMatchObject([
      { cause: 'task', task: { id: 'task-1', label: 'Run the tests' } },
    ]);
    const lone = new TranscriptReducer({ now: () => 7 });
    lone.apply(ended(9, false, { kind: 'monitor', description: 'Watch the build log' }));
    lone.apply(hostLine('monitor-completed-task-9'));
    expect(wakes(lone.snapshot().entries)).toMatchObject([
      { cause: 'task', task: { id: 'task-9', kind: 'monitor', label: 'Watch the build log' } },
    ]);
    const none = new TranscriptReducer({ now: () => 7 });
    none.apply(hostLine('task-completed-task-404'));
    expect(wakes(none.snapshot().entries)).toEqual([{ kind: 'wake', ts: 7, cause: 'task', text: '' }]);
  });

  // Mutation: keep the quiet ends across a turn end → the first end shows in the drain too.
  // Mutation: count a will_wake end as quiet → task-3 is listed.
  it('a notifications- wake lists what ended since the last turn end with no wake of its own; live and disk agree', () => {
    const lines = [
      ...started(1, 'Old build'),
      ...started(2, 'npm test'),
      ...started(3, 'Admitted one'),
      ...started(4, 'Waited one'),
      ...started(5, 'Stopped one'),
      ...started(6, 'Watch the log', true),
      ended(1, false),
      end('p0'),
      ended(2, false),
      ended(3, true),
      ended(4, false, { block_waited: true }),
      ended(5, false, { explicitly_killed: true, kill_result_delivered: true }),
      ended(6, false, { kind: 'monitor' }),
    ];
    const id = 'notifications-0193f1e2-7c3a';
    const r = new TranscriptReducer({ now: () => 7 });
    const ops: TranscriptOp[] = [];
    for (const line of [typed(0, 'go'), ...lines, hostLine(null)]) ops.push(...r.apply(line));
    expect(wakes(r.snapshot().entries)).toEqual([{ kind: 'wake', ts: 7, cause: 'unknown', text: '' }]);
    for (const line of [answer, end(id)]) ops.push(...r.apply(line));
    const tasks = [
      { id: 'task-2', kind: 'command', label: 'npm test' },
      { id: 'task-6', kind: 'monitor', label: 'Watch the log' },
    ];
    expect(wakes(r.snapshot().entries)).toEqual([{ kind: 'wake', ts: 7, cause: 'notifications', tasks, text: '' }]);
    expect(replay(ops).entries).toEqual(r.snapshot().entries);
    const disk = stored([typed(0, 'go'), ...lines, agentPrompt(1), answer, end(id)]);
    expect(wakes(disk).map(({ ts: _ts, ...e }) => e)).toEqual([
      { kind: 'wake', cause: 'notifications', tasks, text: '' },
    ]);
  });

  it('a notifications- wake names at most 8 ends; with none it names nothing', () => {
    const r = new TranscriptReducer({ now: () => 7 });
    for (let n = 1; n <= 10; n += 1) r.apply(ended(n, false, { description: `job ${n}` }));
    r.apply(agentPrompt(0));
    r.apply(end('notifications-a'));
    const [wake] = wakes(r.snapshot().entries);
    expect(wake?.tasks?.map((t) => t.label)).toEqual([
      'job 1',
      'job 2',
      'job 3',
      'job 4',
      'job 5',
      'job 6',
      'job 7',
      'job 8',
    ]);
    r.apply(agentPrompt(1));
    r.apply(end('notifications-b'));
    expect(wakes(r.snapshot().entries)[1]).toEqual({ kind: 'wake', ts: 7, cause: 'notifications', text: '' });
  });

  // Mutation: re-apply only a known cause at turn_completed → the live row keeps the host's guess.
  it("the wake's own turn_completed id wins over the host's guess, as on disk", () => {
    const r = new TranscriptReducer({ now: () => 7 });
    for (const line of [...started(1, 'npm test'), ended(1, true), hostLine('task-completed-task-1')]) r.apply(line);
    expect(wakes(r.snapshot().entries)).toMatchObject([{ cause: 'task' }]);
    r.apply(end('plan-resume-1764000000000'));
    expect(wakes(r.snapshot().entries)).toEqual([{ kind: 'wake', ts: 7, cause: 'unknown', text: '' }]);
  });
});

describe('shell output ticks with x.ai/incrementalBashOutput', () => {
  const ID = 'toolu_ticks';
  const bytes = (text: string): number[] => [...Buffer.from(text)];
  const bash = (fields: Json): Json => ({
    type: 'Bash',
    output_for_prompt: '',
    exit_code: 0,
    command: 'yarn test',
    truncated: false,
    signal: null,
    timed_out: false,
    description: null,
    current_dir: '/Users/dev/proj',
    output_file: '',
    total_bytes: 0,
    ...fields,
  });
  const line = (fields: Json): UpdateLine => ({
    xai: false,
    update: { sessionUpdate: 'tool_call_update', toolCallId: ID, ...fields },
    meta: undefined,
  });
  /** A tick as the agent sends it with the flag: the whole output as text, no byte copy, the new bytes apart. */
  const tick = (whole: string, delta: string): UpdateLine =>
    line({
      status: 'in_progress',
      content: [{ type: 'content', content: { type: 'text', text: whole } }],
      rawOutput: bash({ output: [], output_delta: bytes(delta), total_bytes: Buffer.byteLength(whole) }),
    });
  const toolOf = (entries: readonly TranscriptEntry[]): Extract<TranscriptEntry, { kind: 'tool' }> => {
    const tool = entries.find((e) => e.kind === 'tool');
    if (tool?.kind !== 'tool') throw new Error('no tool entry');
    return tool;
  };
  const start = (): { reducer: TranscriptReducer; ops: TranscriptOp[] } => {
    const reducer = new TranscriptReducer({ now: () => 5 });
    const ops = reducer.apply({
      xai: false,
      update: { sessionUpdate: 'tool_call', toolCallId: ID, title: 'run_terminal_command', rawInput: { command: 'x' } },
      meta: undefined,
    });
    return { reducer, ops };
  };

  // Mutation: read `output_delta` (or the empty `output` array) instead of the text block → 'c\n' or ''.
  it("three ticks: the entry's output is the last tick's whole text, live and in the snapshot", () => {
    const { reducer, ops } = start();
    ops.push(...reducer.apply(tick('a\n', 'a\n')));
    ops.push(...reducer.apply(tick('a\nb\n', 'b\n')));
    ops.push(...reducer.apply(tick('a\nb\nc\n', 'c\n')));
    expect(toolOf(replay(ops).entries).output).toBe('a\nb\nc\n');
    expect(toolOf(reducer.snapshot().entries)).toMatchObject({ status: 'in_progress', output: 'a\nb\nc\n' });
    expect(toolOf(reducer.snapshot().entries)).not.toHaveProperty('exitCode');
  });

  // Mutation: treat an empty `output_delta` as the agent's "clear" → the output is gone.
  it('a tick with an empty output_delta changes nothing', () => {
    const { reducer } = start();
    reducer.apply(tick('a\nb\n', 'a\nb\n'));
    const before = toolOf(reducer.snapshot().entries);
    reducer.apply(tick('a\nb\n', ''));
    expect(toolOf(reducer.snapshot().entries)).toEqual(before);
  });

  it('colored ticks read as text, and the final result line still sets the exit code and the output', () => {
    const { reducer, ops } = start();
    const red = '\u001b[31mFAIL\u001b[39m a.test.ts\n';
    ops.push(...reducer.apply(tick(red, red)));
    expect(toolOf(replay(ops).entries).output).toBe('FAIL a.test.ts\n');
    const whole = `${red}1 failed\n`;
    ops.push(
      ...reducer.apply(
        line({
          status: 'completed',
          content: [{ type: 'content', content: { type: 'text', text: whole } }],
          rawOutput: bash({
            output: bytes(whole),
            output_for_prompt: 'exit: 1\nFAIL a.test.ts\n1 failed\n',
            exit_code: 1,
            description: 'run',
            output_file: '/Users/dev/.mlx-agent/sessions/proj/s/terminal/toolu_ticks.log',
            total_bytes: Buffer.byteLength(whole),
          }),
        }),
      ),
    );
    const shown = toolOf(replay(ops).entries);
    expect(shown).toMatchObject({ status: 'completed', exitCode: 1, output: 'FAIL a.test.ts\n1 failed\n' });
    expect(toolOf(reducer.snapshot().entries)).toEqual(shown);
  });
});

describe('notes: the user note on a call denied with one', () => {
  const NOTE = 'use pnpm, not npm';
  const disk = (update: Json, xai = false): string =>
    JSON.stringify({
      timestamp: 1,
      method: xai ? '_x.ai/session/update' : 'session/update',
      params: { sessionId: 's', update, _meta: { agentTimestampMs: 1_000 } },
    });
  const chunk = (text: string, meta: Json): string =>
    disk({ sessionUpdate: 'user_message_chunk', content: { type: 'text', text }, _meta: { modelId: 'm', ...meta } });
  const typed = (text: string, promptIndex: number): string => chunk(text, { promptIndex });
  /** As the agent stores a note: no `promptIndex`, only the model id. */
  const bare = (text: string): string => chunk(text, {});
  const said = (text: string): string =>
    disk({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text } });
  const done = disk({ sessionUpdate: 'turn_completed', prompt_id: 'p', stop_reason: 'end_turn' }, true);
  const users = (text: string, helper = false) =>
    parseUpdates(text, { helper })
      .transcript.filter((e) => e.kind === 'user')
      .map((e) => (e.kind === 'user' ? [e.text, e.note === true] : null));

  // Mutation: read no `mlxNote` → the host's line is a typed message.
  it("the host's own line is a note: a user entry that is not the turn's prompt; ops equal the snapshot", () => {
    const r = new TranscriptReducer({ now: () => 5 });
    const ops = [...r.userPrompt('clean the build', [], 4)];
    ops.push(...r.note(NOTE));
    const note = { kind: 'user', ts: 5, text: NOTE, promptIndex: null, note: true };
    expect(r.snapshot().entries).toEqual([{ kind: 'user', ts: 4, text: 'clean the build', promptIndex: null }, note]);
    expect(replay(ops).entries).toEqual(r.snapshot().entries);
  });

  // Mutation: drop the pair rule → two note rows.
  it('disk: two equal bare chunks of one open turn are one note, four are two; the entry equals the live one', () => {
    const two = [typed('clean', 0), said('Running.'), bare(NOTE), said('Using pnpm.'), bare(NOTE), done];
    expect(users(two.join('\n') + '\n')).toEqual([
      ['clean', false],
      [NOTE, true],
    ]);
    const four = [typed('clean', 0), bare(NOTE), bare(NOTE), said('Again.'), bare(NOTE), bare(NOTE), done];
    expect(users(four.join('\n') + '\n')).toEqual([
      ['clean', false],
      [NOTE, true],
      [NOTE, true],
    ]);
    const diskNote = parseUpdates(two.join('\n') + '\n').transcript.find((e) => e.kind === 'user' && e.note === true)!;
    const live = new TranscriptReducer({ now: () => 1_000 });
    live.userPrompt('clean');
    live.note(NOTE);
    expect(diskNote).toEqual(live.snapshot().entries[1]);
  });

  it('a new turn starts the pair count again', () => {
    const text = [typed('one', 0), bare(NOTE), done, typed('two', 1), bare(NOTE), bare(NOTE), done].join('\n') + '\n';
    expect(users(text)).toEqual([
      ['one', false],
      [NOTE, true],
      ['two', false],
      [NOTE, true],
    ]);
  });

  it("never a note: the host's typed prompt, a hostTurn chunk, a chunk with no turn open", () => {
    const r = new TranscriptReducer({ now: () => 5 });
    r.apply({
      xai: false,
      update: {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text: 'one' },
        _meta: { promptIndex: 0 },
      },
      meta: undefined,
    });
    r.userPrompt('two');
    expect(r.snapshot().entries.map((e) => e.kind === 'user' && e.note === true)).toEqual([false, false]);
    const hostTurn =
      [typed('one', 0), said('Working.'), chunk('/goal pause', { hostTurn: true }), done].join('\n') + '\n';
    expect(users(hostTurn)).toEqual([
      ['one', false],
      ['/goal pause', false],
    ]);
    expect(users([typed('one', 0), said('ok'), done, bare('a message after the end')].join('\n') + '\n')).toEqual([
      ['one', false],
      ['a message after the end', false],
    ]);
  });

  it('an old build with no promptIndex and no turn_completed: every chunk stays a message and turns split as today', () => {
    const text = [bare('first'), said('one'), bare('second'), said('two')].join('\n') + '\n';
    const parsed = parseUpdates(text);
    expect(users(text)).toEqual([
      ['first', false],
      ['second', false],
    ]);
    expect(splitTurns(parsed.transcript, parsed.turns).map((t) => t.entries.length)).toEqual([2, 2]);
  });

  // Mutation: ignore `helper` → the parent's message reads as a note.
  it("a helper's own session: its parent's bare message stays a message", () => {
    const text = [typed('task', 0), said('Working.'), bare('also check the docs'), said('ok'), done].join('\n') + '\n';
    expect(users(text, true)).toEqual([
      ['task', false],
      ['also check the docs', false],
    ]);
    expect(users(text)).toEqual([
      ['task', false],
      ['also check the docs', true],
    ]);
  });

  it('filterRewindLines cuts at the same prompt with note lines in the turn', () => {
    const rewind = disk({ sessionUpdate: 'rewind_marker', target_prompt_index: 1 }, true);
    const plain = [typed('one', 0), said('A1'), done, typed('two', 1), said('A2'), done, rewind];
    const noted = [typed('one', 0), said('A1'), bare(NOTE), said('A1b'), bare(NOTE), done, ...plain.slice(3)];
    const kept = filterRewindLines(noted);
    expect(kept.filter((l) => !l.includes(NOTE) && !l.includes('A1b'))).toEqual(filterRewindLines(plain));
    expect(kept).toEqual(noted.slice(0, 6));
  });
});

describe('ask_user_question: the entry carries the questions and what the stored sentence says', () => {
  const META = {
    'x.ai/tool': { version: 1, name: 'ask_user_question', kind: 'ask_user', namespace: 'grok_build', read_only: true },
  };
  const Q = 'Which checks should run before a merge?';
  const options = [
    { label: 'Unit tests', description: 'Fast.' },
    { label: 'Lint', description: '' },
  ];
  const SENTENCE = `User has answered your questions: "${Q}"="Unit tests, Lint" user notes: also e2e. You can now continue with the user's answers in mind.`;
  /** As the agent sends and stores them: the model's call, its own copy of the input, then the result. */
  const updates: Json[] = [
    {
      sessionUpdate: 'tool_call',
      toolCallId: 'call_q',
      title: 'ask_user_question',
      rawInput: { questions: [{ question: Q, options, multi_select: true }] },
      _meta: META,
    },
    {
      sessionUpdate: 'tool_call_update',
      toolCallId: 'call_q',
      kind: 'other',
      title: `Ask: ${Q}`,
      locations: [],
      rawInput: { variant: 'AskUserQuestion', questions: [{ question: Q, options, multiSelect: true }] },
      _meta: META,
    },
    {
      sessionUpdate: 'tool_call_update',
      toolCallId: 'call_q',
      status: 'completed',
      content: [{ type: 'content', content: { type: 'text', text: SENTENCE } }],
      rawOutput: { type: 'AskUserQuestion', UserAnswered: { message: SENTENCE } },
    },
  ];

  // Mutation: read `ask` only on disk (snapshot) → the live entry has none.
  it('live ops and disk text give the same entry; multi_select of the model input is read before the copy lands', () => {
    const live = new TranscriptReducer({ now: () => 1_000 });
    const ops: TranscriptOp[] = [];
    ops.push(...live.apply({ xai: false, update: updates[0]!, meta: { agentTimestampMs: 1_000 } }));
    const first = live.snapshot().entries[0];
    expect(first?.kind === 'tool' ? first.ask : null).toEqual({
      questions: [{ text: Q, multi: true, options: ['Unit tests', 'Lint'] }],
      result: null,
    });
    for (const update of updates.slice(1))
      ops.push(...live.apply({ xai: false, update, meta: { agentTimestampMs: 1_000 } }));
    const entries = live.snapshot().entries;
    expect(entries).toHaveLength(1);
    const tool = entries[0]!;
    expect(tool.kind === 'tool' ? tool.ask : null).toEqual({
      questions: [{ text: Q, multi: true, options: ['Unit tests', 'Lint'] }],
      result: { kind: 'answered', answers: [{ picks: ['Unit tests', 'Lint'], other: 'also e2e' }] },
    });
    expect(replay(ops).entries).toEqual(entries);
    const disk = updates
      .map((update) =>
        JSON.stringify({
          timestamp: 1,
          method: 'session/update',
          params: { sessionId: 's', update, _meta: { agentTimestampMs: 1_000 } },
        }),
      )
      .join('\n');
    expect(parseUpdates(`${disk}\n`).transcript).toEqual(entries);
  });

  it('another tool never gets ask', () => {
    const r = new TranscriptReducer({ now: () => 1 });
    r.apply({
      xai: false,
      update: { ...updates[0]!, _meta: { 'x.ai/tool': { name: 'read_file' } } },
      meta: undefined,
    });
    const [entry] = r.snapshot().entries;
    expect(entry?.kind === 'tool' && entry.ask === undefined).toBe(true);
  });
});

describe('helper progress (#10): live ticks make nothing, the end stores turns and tool calls', () => {
  const xai = (update: Json): UpdateLine => ({ xai: true, update, meta: undefined });
  const spawned = xai({
    sessionUpdate: 'subagent_spawned',
    subagent_id: 'child-1',
    parent_session_id: 's',
    child_session_id: 'child-1',
    subagent_type: 'explore',
    description: 'Find the flaky test',
  });
  /** grok `SubagentProgress`: never stored. */
  const tick = xai({
    sessionUpdate: 'subagent_progress',
    subagent_id: 'child-1',
    parent_session_id: 's',
    child_session_id: 'child-1',
    duration_ms: 2_000,
    turn_count: 1,
    tool_call_count: 4,
    tokens_used: 9_000,
    context_window_tokens: 32_768,
    context_usage_pct: 27,
    tools_used: ['grep'],
    error_count: 0,
  });
  /** grok `SubagentFinished`: stored. */
  const finished = xai({
    sessionUpdate: 'subagent_finished',
    subagent_id: 'child-1',
    child_session_id: 'child-1',
    status: 'completed',
    tool_calls: 14,
    turns: 3,
    duration_ms: 134_000,
    tokens_used: 21_480,
    output: 'Found it.',
    will_wake: false,
  });

  it('a tick makes no entry and no op', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    r.apply(spawned);
    const before = r.snapshot().entries;
    expect(r.apply(tick)).toEqual([]);
    expect(r.snapshot().entries).toEqual(before);
  });

  // Mutation: drop `entry.turns = turns` → the row has no turns.
  it('subagent_finished stores turns and tool calls; ops, snapshot and the disk read agree', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    const ops: TranscriptOp[] = [];
    for (const line of [spawned, tick, finished]) ops.push(...r.apply(line));
    const row = r.snapshot().entries.find((e) => e.kind === 'subagent');
    expect(row).toMatchObject({ status: 'completed', turns: 3, toolCalls: 14, tokens: 21_480, durationMs: 134_000 });
    expect(replay(ops).entries).toEqual(r.snapshot().entries);
    const disk = parseUpdates(
      [spawned, finished]
        .map((l) => JSON.stringify({ method: '_x.ai/session/update', params: { sessionId: 's', update: l.update } }))
        .join('\n') + '\n',
    ).transcript;
    const shape = (entries: readonly TranscriptEntry[]) =>
      entries.flatMap((e) => (e.kind === 'subagent' ? [{ ...e, ts: 0, endTs: 0 }] : []));
    expect(shape(disk)).toEqual(shape(r.snapshot().entries));
  });
});

describe('a stopped task (T1): `killed` from the end line', () => {
  const plain = (update: Json): UpdateLine => ({ xai: false, update, meta: undefined });
  const xai = (update: Json): UpdateLine => ({ xai: true, update, meta: undefined });
  const started = (n: number, monitor = false): UpdateLine[] => [
    plain({
      sessionUpdate: 'tool_call',
      toolCallId: `call-${n}`,
      title: monitor ? 'monitor' : 'run_terminal_command',
      rawInput: { command: `cmd-${n}`, description: `Task ${n}`, ...(monitor ? {} : { is_background: true }) },
    }),
    xai({
      sessionUpdate: 'task_backgrounded',
      tool_call_id: `call-${n}`,
      task_id: `task-${n}`,
      command: `cmd-${n}`,
      ...(monitor ? { monitor_description: `Task ${n}` } : { description: `Task ${n}` }),
    }),
    plain({ sessionUpdate: 'tool_call_update', toolCallId: `call-${n}`, status: 'completed' }),
  ];
  /** grok `TaskCompleted`: a kill leaves exit code none and signal `killed` (`terminal.rs` finalize_process). */
  const ended = (n: number, killed: boolean): UpdateLine =>
    xai({
      sessionUpdate: 'task_completed',
      task_snapshot: {
        task_id: `task-${n}`,
        command: `cmd-${n}`,
        kind: 'bash',
        completed: true,
        exit_code: killed ? null : 0,
        signal: killed ? 'killed' : null,
        end_time: { secs_since_epoch: 1_800_000_000, nanos_since_epoch: 0 },
        block_waited: false,
        explicitly_killed: killed,
        kill_result_delivered: false,
      },
      will_wake: true,
    });
  const emptyList = xai({ sessionUpdate: 'background_tasks', tasks: [] });
  const stored = (lines: readonly UpdateLine[]): TranscriptEntry[] =>
    parseUpdates(
      lines
        .map((l) =>
          JSON.stringify({
            method: l.xai ? '_x.ai/session/update' : 'session/update',
            params: { sessionId: 's', update: l.update },
          }),
        )
        .join('\n') + '\n',
    ).transcript;
  const tasks = (entries: readonly TranscriptEntry[]) =>
    entries.flatMap((e) => (e.kind === 'tool' && e.task !== undefined ? [e.task] : []));
  function live(lines: readonly UpdateLine[]): { entries: TranscriptEntry[]; ops: TranscriptOp[] } {
    const r = new TranscriptReducer({ now: () => 7 });
    const ops: TranscriptOp[] = [];
    for (const line of lines) ops.push(...r.apply(line));
    return { entries: r.snapshot().entries, ops };
  }

  // Mutation: drop `explicitly_killed` from the task_completed case → no "Stopped in the background".
  it('a killed end marks the task; one that ended by itself does not; ops, snapshot and disk agree', () => {
    const lines = [...started(1), ...started(2, true), ended(1, true), ended(2, false)];
    const { entries, ops } = live(lines);
    expect(tasks(entries)).toEqual([
      { id: 'task-1', kind: 'command', label: 'Task 1', endTs: 1_800_000_000_000, killed: true },
      { id: 'task-2', kind: 'monitor', label: 'Task 2', endTs: 1_800_000_000_000 },
    ]);
    expect(replay(ops).entries).toEqual(entries);
    expect(tasks(stored(lines))).toEqual(tasks(entries));
  });

  // Mutation: return early on an ended task in endTask → the list's end hides the kill.
  it('the list ended the task first: the kill still lands on it', () => {
    const lines = [...started(1), emptyList, ended(1, true)];
    const { entries, ops } = live(lines);
    expect(tasks(entries)).toEqual([
      { id: 'task-1', kind: 'command', label: 'Task 1', endTs: expect.any(Number), killed: true },
    ]);
    expect(replay(ops).entries).toEqual(entries);
    expect(tasks(stored(lines)).map((t) => t.killed)).toEqual([true]);
  });

  it('the end came before the line that names the task: the task starts ended and killed', () => {
    const [call, backgrounded, result] = started(1);
    const lines = [call!, ended(1, true), backgrounded!, result!];
    const { entries, ops } = live(lines);
    expect(tasks(entries)).toEqual([
      { id: 'task-1', kind: 'command', label: 'Task 1', endTs: 1_800_000_000_000, killed: true },
    ]);
    expect(replay(ops).entries).toEqual(entries);
    expect(tasks(stored(lines))).toEqual(tasks(entries));
  });
});

describe('model_auto_switched (#16): live only, never an entry', () => {
  /** grok `SessionUpdate::ModelAutoSwitched` as `send_model_auto_switched` sends it: no `_meta`, never stored. */
  const switched: UpdateLine = {
    xai: true,
    update: {
      sessionUpdate: 'model_auto_switched',
      previous_model_id: '',
      new_model_id: '',
      reason: 'Your previous model is no longer available. Please start a new session.',
    },
    meta: undefined,
  };

  it('a line fed to the reducer makes no entry and no op, in a turn and outside one', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    expect(r.apply(switched)).toEqual([]);
    r.userPrompt('Fix the failing test', [], 6);
    const before = r.snapshot().entries;
    expect(r.apply(switched)).toEqual([]);
    expect(r.snapshot().entries).toEqual(before);
    const disk = `${JSON.stringify({ method: '_x.ai/session/update', params: { sessionId: 's', update: switched.update } })}\n`;
    expect(parseUpdates(disk).transcript).toEqual([]);
  });
});

describe('monitor_event (T2): live only, never an entry', () => {
  /** grok `SessionUpdate::MonitorEvent` as the bridge sends it (`_x.ai/monitor_event`): no `_meta`, never stored. */
  const event: UpdateLine = {
    xai: true,
    update: {
      sessionUpdate: 'monitor_event',
      task_id: '01a20df5-8562-75b2-aaf5-d3c1e2ec9a01',
      description: 'Watch the build log',
      event_text: "error TS2345: Argument of type 'string' is not assignable",
    },
    meta: undefined,
  };

  it('a line fed to the reducer makes no entry and no op, in a turn and outside one', () => {
    const r = new TranscriptReducer({ now: () => 5 });
    expect(r.apply(event)).toEqual([]);
    r.userPrompt('Watch the build', [], 6);
    const before = r.snapshot().entries;
    expect(r.apply(event)).toEqual([]);
    expect(r.snapshot().entries).toEqual(before);
    const disk = `${JSON.stringify({ method: '_x.ai/session/update', params: { sessionId: 's', update: event.update } })}\n`;
    expect(parseUpdates(disk).transcript).toEqual([]);
  });
});

describe('workflow runs: what the reducer keeps of a run, its calls, its agents and its wake', () => {
  /** A real run's parent lines and its retried agent's own lines, prose replaced (`fixtures/workflow-run`). */
  const RUN_DIR = join(FIXTURES, 'workflow-run');
  const RUN = 'wf_01a2168534c27fd1a477edb6fad7b0a4';
  const RETRY_CHILD = '01a21688-13f5-7b32-873d-247e85ca5c3f';
  const parentLines = (): string[] => splitCompleteLines(readFileSync(join(RUN_DIR, 'parent.jsonl'), 'utf8'));
  const text = (lines: readonly string[]): string => `${lines.join('\n')}\n`;
  const updateOf = (line: string): Json => ((JSON.parse(line) as Json).params as Json).update as Json;
  const withUpdate = (line: string, update: Json): string => {
    const obj = JSON.parse(line) as Json;
    return JSON.stringify({ ...obj, params: { ...(obj.params as Json), update } });
  };
  const xaiAt = (n: number, update: Json): string =>
    JSON.stringify({
      timestamp: Math.floor(n / 1000),
      method: '_x.ai/session/update',
      params: { sessionId: 's', update, _meta: { agentTimestampMs: n } },
    });
  const acpAt = (n: number, update: Json): string =>
    JSON.stringify({
      timestamp: Math.floor(n / 1000),
      method: 'session/update',
      params: { sessionId: 's', update, _meta: { agentTimestampMs: n } },
    });
  type Tool = Extract<TranscriptEntry, { kind: 'tool' }>;
  type Wake = Extract<TranscriptEntry, { kind: 'wake' }>;
  type Run = Extract<TranscriptEntry, { kind: 'workflow' }>;
  const tools = (entries: readonly TranscriptEntry[]): Tool[] =>
    entries.filter((e): e is Tool => e.kind === 'tool' && e.name === 'workflow');

  // Mutation: put back `&& this.runEntries.has(run)` → a spawn read before (or without) its run's entry loses it.
  it('a spawn names its run and model with no run entry in the transcript', () => {
    const spawn = parentLines().find((l) => updateOf(l).sessionUpdate === 'subagent_spawned')!;
    const [entry] = parseUpdates(text([spawn])).transcript;
    expect(entry).toMatchObject({
      kind: 'subagent',
      childId: '01a21685-34f1-7d63-8879-909f581e5a5b',
      description: 'track:process',
      run: RUN,
      model: 'Inkling-2-Mini-UD-Q4_K_M',
    });
  });

  // Mutation: class a call by `rawOutput.run_id` → the pause, stop and resume results read as launches.
  it('a workflow call names its kind by source.type and its run by the id that came back, else by its name', () => {
    const lines = parentLines();
    const real = tools(parseUpdates(text(lines)).transcript);
    expect(real.map((e) => [e.status, e.run])).toEqual([
      ['failed', undefined],
      ['failed', { id: null, kind: 'check' }],
      ['completed', { id: null, kind: 'check' }],
      ['completed', { id: RUN, kind: 'launch' }],
    ]);
    const launch = lines.filter((l) => updateOf(l).toolCallId === 'toolu_71e89f72f5d44bbd88745919cabf3fec');
    const call = (id: string, n: number, source: Json, rawOutput: Json | null, status = 'completed'): string[] => [
      acpAt(n, {
        sessionUpdate: 'tool_call',
        toolCallId: id,
        title: 'workflow',
        rawInput: { source },
        _meta: { 'x.ai/tool': { name: 'workflow', kind: 'workflow' } },
      }),
      acpAt(n + 1, {
        sessionUpdate: 'tool_call_update',
        toolCallId: id,
        status,
        ...(rawOutput === null ? {} : { rawOutput }),
      }),
    ];
    const controlled = (verb: string) => ({
      type: 'Workflow',
      run_id: RUN,
      task_id: RUN,
      name: 'research-advisory',
      message: `${verb} workflow 'research-advisory'; its child agents were cancelled.`,
    });
    const refused = { error: 'workflow_resolve_failed', message: 'no such workflow' };
    const t = 1795676600000;
    const later = [
      ...call('pause-1', t, { type: 'pause', run_id: 'research-advisory' }, controlled('Paused')),
      ...call(
        'resume-1',
        t + 10,
        { type: 'resume', resume_from_run_id: RUN },
        { ...controlled('Resumed'), message: '' },
      ),
      ...call('stop-1', t + 20, { type: 'stop', run_id: 'research-advisory' }, null, 'failed'),
      ...call('stop-2', t + 30, { type: 'stop', run_id: 'other-run' }, null, 'failed'),
      ...call('launch-2', t + 40, { type: 'name', name: 'deep-research' }, refused, 'failed'),
    ];
    const all = tools(parseUpdates(text([...lines.slice(0, lines.indexOf(launch.at(-1)!) + 1), ...later])).transcript);
    expect(all.slice(4).map((e) => [e.id, e.run])).toEqual([
      ['pause-1', { id: RUN, kind: 'pause' }],
      ['resume-1', { id: RUN, kind: 'resume' }],
      ['stop-1', { id: RUN, kind: 'stop' }],
      ['stop-2', { id: null, kind: 'stop' }],
      ['launch-2', { id: null, kind: 'launch' }],
    ]);
  });

  // Mutation: read the run's newest state for the snapshot → a resume in the wake turn empties the first row.
  it('a wake keeps the stored update of the revision its prompt id names, also after a resume', () => {
    const real = parseUpdates(text(parentLines())).transcript.filter((e): e is Wake => e.kind === 'wake');
    expect(real).toHaveLength(1);
    expect(real[0]!.workflow).toEqual({
      runId: RUN,
      revision: 26,
      name: 'research-advisory',
      snapshot: {
        status: 'complete',
        report: '# Advisory research\n\nThe fix holds for the reported path; one more path needs a guard.\n',
        agentsUsed: 7,
        agentBudget: 128,
        agentsFailed: 0,
        elapsedMs: 2_323_877,
      },
    });
    const run = (n: number, revision: number, status: string, extra: Json = {}) =>
      xaiAt(n, {
        sessionUpdate: 'workflow_updated',
        run_id: 'wf_2',
        name: 'deep-research',
        status,
        revision,
        agent_budget: 4,
        agents_used: 4,
        elapsed_ms: n,
        agents: [
          { agent_id: 'a', label: 'a', state: 'done', tokens_used: 1, duration_ms: 1 },
          { agent_id: 'b', label: 'b', state: 'failed', tokens_used: 1, duration_ms: 1 },
        ],
        ...extra,
      });
    const wake = (n: number, revision: number): string[] => [
      xaiAt(n, {
        sessionUpdate: 'hook_execution',
        event_name: 'user_prompt_submit',
        prompt_id: `workflow-completed-wf_2-${revision}`,
        runs: [],
      }),
      acpAt(n + 1, {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text: 'A background workflow stopped.' },
        _meta: { promptIndex: revision, hideFromScrollback: true },
      }),
    ];
    const end = (n: number, revision: number) =>
      xaiAt(n, {
        sessionUpdate: 'turn_completed',
        prompt_id: `workflow-completed-wf_2-${revision}`,
        stop_reason: 'end_turn',
      });
    const lines = [
      run(1_000, 10, 'budget_limited', { pause_message: 'Out of agents.' }),
      ...wake(2_000, 10),
      run(3_000, 11, 'active'),
      end(4_000, 10),
      run(5_000, 20, 'complete', { result_summary: 'All four agree.' }),
      ...wake(6_000, 20),
      run(7_000, 21, 'complete', { result_summary: 'All four agree.' }),
      end(8_000, 20),
    ];
    const wakes = parseUpdates(text(lines)).transcript.filter((e): e is Wake => e.kind === 'wake');
    expect(wakes.map((e) => e.workflow)).toEqual([
      {
        runId: 'wf_2',
        revision: 10,
        name: 'deep-research',
        snapshot: {
          status: 'budget_limited',
          pauseMessage: 'Out of agents.',
          agentsUsed: 4,
          agentBudget: 4,
          agentsFailed: 1,
          elapsedMs: 1_000,
        },
      },
      {
        runId: 'wf_2',
        revision: 20,
        name: 'deep-research',
        snapshot: {
          status: 'complete',
          report: 'All four agree.',
          agentsUsed: 4,
          agentBudget: 4,
          agentsFailed: 1,
          elapsedMs: 5_000,
        },
      },
    ]);
  });

  // Mutation: mark a copy by its outer `timestamp` (the copy time) → the copied answer and turn end read as own.
  it("a retried agent's copied lines are marked by their eventId prefix, and keep the first attempt's times", () => {
    const childText = readFileSync(join(RUN_DIR, 'retry-child.jsonl'), 'utf8');
    const lines = splitCompleteLines(childText);
    const own = (line: string) =>
      ((((JSON.parse(line) as Json).params as Json)._meta as Json).eventId as string).startsWith(`${RETRY_CHILD}-`);
    expect(lines.findIndex(own)).toBe(56);
    expect(lines.slice(56).every(own)).toBe(true);
    const parsed = parseUpdates(childText, { helper: true, sessionId: RETRY_CHILD });
    const turns = parsed.transcript.findIndex((e) => e.kind === 'turn-end');
    expect(turns).toBeGreaterThan(0);
    const firstAttempt = parsed.transcript.slice(0, turns + 1);
    const secondAttempt = parsed.transcript.slice(turns + 1);
    expect(firstAttempt.every((e) => e.copied === true)).toBe(true);
    expect(secondAttempt.length).toBeGreaterThan(0);
    expect(secondAttempt.every((e) => e.copied === undefined)).toBe(true);
    expect(firstAttempt.filter((e) => e.kind === 'assistant')).toHaveLength(1);
    expect(firstAttempt[0]).toMatchObject({ kind: 'user', ts: 1795674158731 });
    expect(parsed.turns.map((t) => [t.copied, t.ts])).toEqual([
      [true, 1795674346254],
      [undefined, 1795674467013],
    ]);
    expect(parseUpdates(childText, { helper: true }).transcript.some((e) => e.copied !== undefined)).toBe(false);
  });

  // Mutations: link only tool calls → a typed control turn names no run; take the first of two prefix matches.
  it('a typed /workflow pause|resume|stop host turn names the run its name picks, none for two matches', () => {
    const launch = parentLines().filter((l) => {
      const u = updateOf(l);
      return u.sessionUpdate === 'workflow_updated' && u.revision === 1;
    });
    const second = withUpdate(launch[0]!, {
      ...updateOf(launch[0]!),
      run_id: 'wf_01a2168534c27fd1a477edb6fad7b0a5',
      name: 'research-advisory-2',
    });
    const hostTurn = (n: number, typed: string, said: string): string[] => [
      acpAt(n, {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text: typed },
        _meta: { hostTurn: true },
      }),
      acpAt(n + 1, {
        sessionUpdate: 'agent_message_chunk',
        content: { type: 'text', text: said },
        _meta: { hostTurn: true },
      }),
      xaiAt(n + 2, { sessionUpdate: 'turn_completed', prompt_id: `p${n}`, stop_reason: 'end_turn', elapsed_ms: 5 }),
    ];
    const t = 1795676700000;
    const lines = [
      launch[0]!,
      second,
      ...hostTurn(t, '/workflow resume research-advisory', 'Resumed research-advisory from its journal.'),
      ...hostTurn(t + 10, '/workflow stop research', "Several runs could be 'stop'."),
      ...hostTurn(t + 20, '/workflow research-advisory-2 PAUSE', 'Paused research-advisory-2.'),
      ...hostTurn(t + 30, '/workflow stop wf_01a2168534c27fd1a477edb6fad7b0a4', 'Stopped research-advisory.'),
      ...hostTurn(t + 40, '/workflow save research-advisory', 'Saved.'),
      ...hostTurn(t + 50, '/workflowstop research-advisory', 'Unknown command.'),
    ];
    const users = parseUpdates(text(lines)).transcript.filter((e) => e.kind === 'user');
    expect(users.map((e) => (e.kind === 'user' ? e.runControl : null))).toEqual([
      { verb: 'resume', runId: RUN },
      undefined,
      { verb: 'pause', runId: 'wf_01a2168534c27fd1a477edb6fad7b0a5' },
      { verb: 'stop', runId: RUN },
      undefined,
      undefined,
    ]);
    const typedOnly = parseUpdates(
      text([launch[0]!, ...hostTurn(t, '/workflow resume research-advisory', 'Ran.').filter((_, i) => i !== 1)]),
    ).transcript;
    expect(typedOnly.find((e) => e.kind === 'user')).not.toHaveProperty('runControl');
    // Live: the host pushes the typed prompt, and the agent's hostTurn answer names the run.
    const r = new TranscriptReducer({ now: () => t });
    r.apply({ xai: true, update: updateOf(launch[0]!), meta: undefined });
    r.userPrompt('/workflow pause research-advisory', [], t);
    const ops = r.apply({
      xai: false,
      update: {
        sessionUpdate: 'agent_message_chunk',
        content: { type: 'text', text: 'Paused.' },
        _meta: { hostTurn: true },
      },
      meta: undefined,
    });
    expect(ops[0]).toMatchObject({
      op: 'set',
      index: 1,
      entry: { kind: 'user', runControl: { verb: 'pause', runId: RUN } },
    });
  });

  // Mutation: keep only the newest step start (`phasesEntered = [that one]`) → the first step's start is lost.
  it('a run entry keeps the objective, run time, model of a roster row and each stored step start', () => {
    const lines = parentLines();
    const usage = lines.findIndex((l) => updateOf(l).revision === 14);
    const flagged = withUpdate(lines[usage]!, {
      ...updateOf(lines[usage]!),
      agent_usage_incomplete: true,
      agents: [{ ...((updateOf(lines[usage]!).agents as Json[])[0] as Json), model: 'small-model' }],
    });
    const [run] = parseUpdates(text([...lines.slice(0, usage), flagged])).transcript.filter(
      (e): e is Run => e.kind === 'workflow',
    );
    expect(run).toMatchObject({
      revision: 14,
      objective:
        "Research a GitHub security advisory against the local checkout: verify the fix commit, the advisory's claims, and remaining holes",
      elapsedMs: 845_501,
      usageIncomplete: true,
      phasesEntered: [
        { title: 'Recon', ts: 1795674158302 },
        { title: 'Verify', ts: 1795675003800 },
      ],
    });
    expect(run!.agents[0]).toMatchObject({ label: 'track:fix-reality', model: 'small-model' });
    const done = parseUpdates(text(lines)).transcript.find((e): e is Run => e.kind === 'workflow')!;
    expect(done.phasesEntered.map((p) => p.title)).toEqual(['Recon', 'Verify', 'Synthesize']);
    expect(done.usageIncomplete).toBeUndefined();
  });
});
