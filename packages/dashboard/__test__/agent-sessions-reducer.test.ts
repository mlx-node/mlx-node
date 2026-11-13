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
  type UpdateLine,
} from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry, TurnMetric } from '../src/agent-sessions/types.js';
import { liveFilter } from '../src/chat/agent/live-filter.js';
import type { TranscriptOp } from '../src/chat/types.js';

const FIXTURES = fileURLToPath(new URL('./fixtures/', import.meta.url));
const RECORDED = join(FIXTURES, 'chat-scenarios/recorded');

type Json = Record<string, unknown>;

function replay(ops: readonly TranscriptOp[]): { entries: TranscriptEntry[]; turns: TurnMetric[] } {
  const entries: TranscriptEntry[] = [];
  const turns: TurnMetric[] = [];
  for (const op of ops) {
    if (op.op === 'push') entries.push(op.entry);
    else if (op.op === 'set') entries[op.index] = op.entry;
    else if (op.op === 'text') {
      const e = entries[op.index]!;
      if (e.kind !== 'assistant' && e.kind !== 'thinking') throw new Error(`text op on ${e.kind}`);
      entries[op.index] = { ...e, text: e.text + op.append };
    } else turns.push(op.metric);
  }
  return { entries, turns };
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
      expect(withoutTextEnd(replayed.entries)).toEqual(withoutTextEnd(snap.entries));
      expect(replayed.turns).toEqual(snap.turns);
      expect(snap.entries.map((e) => e.kind)).toEqual(
        parseUpdates(readFileSync(file, 'utf8')).transcript.map((e) => e.kind),
      );
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
      { op: 'text', index: 3, append: 'b' },
    ]);
    expect(r.snapshot().entries.map((e) => e.kind)).toEqual(['user', 'assistant', 'user', 'assistant']);
  });
});
