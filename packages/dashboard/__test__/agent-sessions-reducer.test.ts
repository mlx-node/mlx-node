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
  type UpdateLine,
} from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry, TurnMetric } from '../src/agent-sessions/types.js';
import { liveFilter } from '../src/chat/agent/live-filter.js';
import type { TranscriptOp } from '../src/chat/types.js';

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
      ops.push(...reducer.userPrompt(block.text as string, 1));
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
      ...r.userPrompt('new', 9),
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
    r.userPrompt('next', 6);
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

  // Mutation: read no `mlxWakePromptId` in the hidden-chunk case → cause 'unknown' until turn_completed.
  it("the host line's prompt id names the run at once", () => {
    const r = new TranscriptReducer({ now: () => 5 });
    r.apply(xai(RUN));
    r.apply(hidden('', { [WAKE_PROMPT_ID]: 'workflow-completed-wf_1-5' }));
    expect(wakes(r)).toEqual([
      { kind: 'wake', ts: 5, cause: 'workflow', workflow: { name: 'deep-research', status: 'complete' }, text: '' },
    ]);
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
        workflow: { name: 'deep-research', status: 'complete' },
        text: 'A workflow finished.\nRead its report.',
      },
    ]);
    r.apply(xai({ sessionUpdate: 'turn_completed', prompt_id: 'workflow-completed-wf_1-5', stop_reason: 'end_turn' }));
    r.apply(hidden('', {}));
    r.apply(hidden('A task finished.', { promptIndex: 2 }));
    expect(wakes(r).map((e) => e.text)).toEqual(['A workflow finished.\nRead its report.', 'A task finished.']);
  });
});
