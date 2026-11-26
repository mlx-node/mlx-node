/**
 * Edit diffs in the shared reducer: the agent's `diff` blocks (start and result, shapes of mlx-agent v0.3.4
 * `tool_calls.rs` and `acp_conversion.rs`) become `edits` on the tool entry, the same from live lines and from disk.
 */

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, slimRawOutput, TranscriptReducer, type UpdateLine } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';

const SID = '01a20000-0000-7000-8000-0000000000e1';
const T0 = 1_791_643_279_000;
const PATH = '/w/src/app.ts';

type Update = Record<string, unknown>;
type Tool = Extract<TranscriptEntry, { kind: 'tool' }>;

function lines(updates: readonly Update[]): { disk: string; live: UpdateLine[] } {
  const disk = updates
    .map((update, n) =>
      JSON.stringify({
        timestamp: Math.floor(T0 / 1000),
        method: 'session/update',
        params: { sessionId: SID, update, _meta: { eventId: `${SID}-${n}`, agentTimestampMs: T0 + n } },
      }),
    )
    .join('\n');
  const live = updates.map((update, n) => ({ xai: false, update, meta: { agentTimestampMs: T0 + n } }));
  return { disk: `${disk}\n`, live };
}

function both(updates: readonly Update[]): { stored: Tool; live: Tool } {
  const { disk, live } = lines(updates);
  const stored = parseUpdates(disk).transcript.find((e): e is Tool => e.kind === 'tool')!;
  const reducer = new TranscriptReducer();
  for (const line of live) reducer.apply(line);
  const entry = reducer.snapshot().entries.find((e): e is Tool => e.kind === 'tool')!;
  return { stored, live: entry };
}

const tool = (name: string, rawInput: unknown): Update => ({
  sessionUpdate: 'tool_call',
  toolCallId: 'call_1',
  title: name,
  kind: 'other',
  status: 'pending',
  rawInput,
  _meta: { 'x.ai/tool': { version: 1, name, kind: name, namespace: 'grok_build', label: name, read_only: false } },
});

const start = (oldText: string, newText: string, meta?: object): Update => ({
  sessionUpdate: 'tool_call_update',
  toolCallId: 'call_1',
  title: 'Edit `src/app.ts`',
  kind: 'edit',
  locations: [{ path: 'src/app.ts' }],
  content: [{ type: 'diff', path: PATH, oldText, newText, ...(meta === undefined ? {} : { _meta: meta }) }],
});

interface Detail {
  old_string: string;
  old_line: number;
  new_string: string;
  new_line: number;
  context_before: string;
  context_after: string;
  line_prefix: string;
}

const result = (oldText: string, newText: string, details: Detail[]): Update => ({
  sessionUpdate: 'tool_call_update',
  toolCallId: 'call_1',
  status: 'completed',
  content: [{ type: 'diff', path: PATH, oldText, newText, _meta: { details } }],
  rawOutput: {
    type: 'SearchReplace',
    EditsApplied: {
      old_string: oldText,
      new_string: newText,
      tool_output_for_prompt: 'The file was edited.',
      absolute_path: PATH,
      edits: { details },
    },
  },
});

const detail = (line: number, overrides: Partial<Detail> = {}): Detail => ({
  old_string: 'port = 3000',
  old_line: line,
  new_string: 'port = 8080',
  new_line: line,
  context_before: 'a\nb\n',
  context_after: '\nc\n',
  line_prefix: 'export const ',
  ...overrides,
});

describe('edit diffs in the transcript reducer', () => {
  it('a start line asks for a change at its _meta.old_line, with no output', () => {
    const { stored, live } = both([
      tool('search_replace', { file_path: 'src/app.ts', old_string: 'a\nb\n', new_string: 'a\nc\n' }),
      start('a\nb\n', 'a\nc\n', { old_line: 41, new_line: 41 }),
    ]);
    expect(stored.edits).toEqual([
      { path: PATH, added: 1, removed: 1, hunks: [{ start: [41, 41], body: ' a\n-b\n+c' }] },
    ]);
    expect(stored).not.toHaveProperty('output');
    expect(live).toEqual(stored);
  });

  it('a start with no line named has hunks with no numbers', () => {
    const { stored } = both([tool('search_replace', {}), start('x\n', 'y\n')]);
    expect(stored.edits?.[0]?.hunks).toEqual([{ start: null, body: '-x\n+y' }]);
  });

  it('a result with two matches has a hunk at each, with the context lines the agent sent', () => {
    const { stored, live } = both([
      tool('search_replace', { file_path: 'src/app.ts', replace_all: true }),
      start('port = 3000', 'port = 8080', { old_line: 3, new_line: 3 }),
      result('port = 3000', 'port = 8080', [detail(3), detail(20)]),
    ]);
    const body = ' a\n b\n-export const port = 3000\n+export const port = 8080\n \n c';
    expect(stored.status).toBe('completed');
    expect(stored.edits).toEqual([
      {
        path: PATH,
        added: 2,
        removed: 2,
        hunks: [
          { start: [1, 1], body },
          { start: [18, 18], body },
        ],
      },
    ]);
    expect(stored).not.toHaveProperty('output');
    expect(JSON.stringify(stored)).not.toContain('EditsApplied');
    expect(live).toEqual(stored);
  });

  it('a write before its result adds every line; its result diffs the whole old file against the new one', () => {
    const old = 'one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n';
    const next = old.replace('nine', 'NINE');
    const asked = both([tool('write', { file_path: 'src/app.ts', content: next }), start('', next)]);
    expect(asked.stored.edits).toEqual([
      {
        path: PATH,
        added: 10,
        removed: 0,
        hunks: [
          {
            start: [0, 1],
            body: next
              .trimEnd()
              .split('\n')
              .map((l) => `+${l}`)
              .join('\n'),
          },
        ],
      },
    ]);
    const done = both([
      tool('write', { file_path: 'src/app.ts', content: next }),
      start('', next),
      result(old, next, [
        detail(1, { old_string: old, new_string: next, context_before: '', context_after: '', line_prefix: '' }),
      ]),
    ]);
    expect(done.stored.edits).toEqual([
      {
        path: PATH,
        added: 1,
        removed: 1,
        hunks: [{ start: [6, 6], body: ' six\n seven\n eight\n-nine\n+NINE\n ten' }],
      },
    ]);
    expect(done.live).toEqual(done.stored);
  });

  it('a result with no old text is a new file; a write start is not', () => {
    const detailNew = detail(1, {
      old_string: '',
      new_string: 'hi\n',
      context_before: '',
      context_after: '',
      line_prefix: '',
    });
    const created = both([
      tool('write', { file_path: 'src/app.ts', content: 'hi\n' }),
      start('', 'hi\n'),
      result('', 'hi\n', [detailNew]),
    ]);
    expect(created.stored.edits).toEqual([
      { path: PATH, create: true, added: 1, removed: 0, hunks: [{ start: [0, 1], body: '+hi' }] },
    ]);
    const asked = both([tool('write', {}), start('', 'hi\n')]);
    expect(asked.stored.edits?.[0]).not.toHaveProperty('create');
  });

  it('a failed update keeps the asked-for change and shows the agent’s error as output', () => {
    const { stored, live } = both([
      tool('search_replace', { file_path: 'src/app.ts' }),
      start('a\n', 'b\n', { old_line: 7, new_line: 7 }),
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'call_1',
        status: 'failed',
        content: [{ type: 'content', content: { type: 'text', text: 'No matches found for old_string' } }],
        rawOutput: { type: 'SearchReplace', NoMatchesFound: { message: 'No matches found', file_path: PATH } },
      },
    ]);
    expect(stored.status).toBe('failed');
    expect(stored.edits).toEqual([{ path: PATH, added: 1, removed: 1, hunks: [{ start: [7, 7], body: '-a\n+b' }] }]);
    expect(stored.output).toBe('No matches found for old_string');
    expect(live).toEqual(stored);
  });

  it('live ops carry the asked-for change, then the made one, and end equal to the stored entry', () => {
    const updates = [
      tool('search_replace', {}),
      start('port = 3000', 'port = 8080', { old_line: 3, new_line: 3 }),
      result('port = 3000', 'port = 8080', [detail(3)]),
    ];
    const reducer = new TranscriptReducer();
    const seen = lines(updates).live.flatMap((line) => reducer.apply(line));
    const edits = seen.map((op) =>
      (op.op === 'push' || op.op === 'set') && op.entry.kind === 'tool' ? op.entry.edits : null,
    );
    expect(edits[0]).toBeUndefined();
    expect(edits[1]?.[0]?.hunks[0]?.start).toEqual([3, 3]);
    expect(edits[2]?.[0]?.hunks[0]?.start).toEqual([1, 1]);
    expect(reducer.snapshot().entries).toEqual(parseUpdates(lines(updates).disk).transcript);
  });

  it('several files of one call keep their order', () => {
    const { stored } = both([
      tool('apply_patch', {}),
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'call_1',
        status: 'completed',
        content: [
          { type: 'diff', path: '/w/a.ts', oldText: 'a\n', newText: 'b\n', _meta: { details: [] } },
          { type: 'diff', path: '/w/new.ts', oldText: null, newText: 'n\n' },
        ],
      },
    ]);
    expect(stored.edits?.map((e) => [e.path, e.create === true, e.added, e.removed])).toEqual([
      ['/w/a.ts', false, 1, 1],
      ['/w/new.ts', true, 1, 0],
    ]);
  });
  it('the reducer keeps no whole file of a result: EditsApplied goes, every other result stays as it is', () => {
    const raw = result('a', 'b', [detail(1)]).rawOutput;
    expect(slimRawOutput(raw)).toEqual({ type: 'SearchReplace' });
    const failed = { type: 'SearchReplace', NoMatchesFound: { message: 'No match', file_path: PATH } };
    expect(slimRawOutput(failed)).toBe(failed);
    const bash = { type: 'Bash', exit_code: 0, EditsApplied: 'not an edit' };
    expect(slimRawOutput(bash)).toBe(bash);
  });
});
