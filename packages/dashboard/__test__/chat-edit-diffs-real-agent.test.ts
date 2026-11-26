/**
 * Edit diffs on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: the five edits of the
 * harness word "diff" run on the real agent and on the fake one. Both send the same `diff` blocks, and the reducer gives
 * the same `edits` from the live lines and from `updates.jsonl`. Skipped without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { fakeHost, noProviders, until, type FakeHost, type Json } from './helpers/chat-fake.js';
import { hasToolResult, lastMessage, startScriptedEngine, type ModelReply } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;

type Tool = Extract<TranscriptEntry, { kind: 'tool' }>;

interface EditStep {
  toolName: 'search_replace' | 'write';
  file_path: string;
  before: string | null;
  old_string?: string;
  new_string?: string;
  replace_all?: boolean;
  content?: string;
}

/** The edit steps of the harness word "diff", in order. */
function diffSteps(): EditStep[] {
  const scenario = JSON.parse(
    readFileSync(join(import.meta.dirname, 'fixtures/chat-scenarios/chat-harness.json'), 'utf8'),
  ) as { work: { prompts: Array<{ match: string; steps: Array<{ edit?: EditStep }> }> } };
  const prompt = scenario.work.prompts.find((p) => p.match === 'diff')!;
  return prompt.steps.flatMap((step) => (step.edit === undefined ? [] : [step.edit]));
}

/** Real agent only: two replace_all matches one line apart, then a deleted block. */
const EXTRA: EditStep[] = [
  {
    toolName: 'search_replace',
    file_path: 'src/list.ts',
    before: 'a\nb\nc\nx = 1;\nd\nx = 1;\ne\nf\ng\nh\n',
    old_string: 'x = 1;',
    new_string: 'x = 2;',
    replace_all: true,
  },
  {
    toolName: 'search_replace',
    file_path: 'src/drop.ts',
    before: 'a\nb\nc\nx\ny\nd\ne\nf\ng\n',
    old_string: 'x\ny\n',
    new_string: '',
  },
];

/** The model's call for one step, as the agent's tool schema takes it. */
function callOf(step: EditStep): ModelReply {
  const input: Json =
    step.toolName === 'write'
      ? { file_path: step.file_path, content: step.content }
      : {
          file_path: step.file_path,
          old_string: step.old_string,
          new_string: step.new_string,
          ...(step.replace_all === true ? { replace_all: true } : {}),
        };
  return { blocks: [{ type: 'tool_use', name: step.toolName, input }] };
}

/** Each tool call's `diff` blocks on the wire, in order, paths relative to the project. */
function diffBlocks(fake: FakeHost, sessionId: string, project: string): Json[][] {
  const byCall = new Map<string, Json[]>();
  for (const u of fake.events.updates) {
    if (u.sessionId !== sessionId) continue;
    const update = u.line.update as Json;
    if (update.sessionUpdate !== 'tool_call' && update.sessionUpdate !== 'tool_call_update') continue;
    const id = update.toolCallId as string;
    if (!byCall.has(id)) byCall.set(id, []);
    if (!Array.isArray(update.content)) continue;
    for (const item of update.content as Json[]) {
      if (item.type !== 'diff') continue;
      const path = String(item.path);
      byCall.get(id)!.push({ ...item, path: path.startsWith(`${project}/`) ? path.slice(project.length + 1) : path });
    }
  }
  return [...byCall.values()].filter((blocks) => blocks.length > 0);
}

function storedTools(agentHome: string, marker: string): Tool[] {
  const files = readdirSync(agentHome, { recursive: true, encoding: 'utf8' })
    .filter((f) => f.endsWith('updates.jsonl'))
    .map((f) => readFileSync(join(agentHome, f), 'utf8'))
    .filter((text) => text.includes(marker));
  expect(files).toHaveLength(1);
  return parseUpdates(files[0]!).transcript.filter((e): e is Tool => e.kind === 'tool');
}

function liveTools(fake: FakeHost, sessionId: string): Tool[] {
  const reducer = new TranscriptReducer();
  for (const u of fake.events.updates) if (u.sessionId === sessionId) reducer.apply(u.line);
  return reducer.snapshot().entries.filter((e): e is Tool => e.kind === 'tool');
}

/** Runs one turn and allows every request it makes, until it ends. */
async function runAllowing(fake: FakeHost, sessionId: string, text: string): Promise<string> {
  const turn = fake.host.startTurn(sessionId, text, { modelId: MODEL, cwd: fake.project });
  let answered = 0;
  const ended: { outcome: string | null } = { outcome: null };
  void turn.done.then((end) => {
    ended.outcome = end.outcome ?? 'no outcome';
  });
  for (;;) {
    await until(() => ended.outcome !== null || fake.events.interactions.length > answered, AGENT_WAIT_MS);
    for (; answered < fake.events.interactions.length; answered += 1) {
      fake.host.answer(sessionId, fake.events.interactions[answered]!.interaction.id, 'allow-once');
    }
    if (ended.outcome !== null) return ended.outcome;
  }
}

/** What the page draws of each edit: status, per file counts and hunks, and the output. */
function shapes(tools: readonly Tool[]): unknown[] {
  return tools.map((t) => [
    t.name,
    t.status,
    t.edits?.map((d) => [d.create === true, d.added, d.removed, d.hunks]),
    t.output ?? null,
  ]);
}

describe.skipIf(BIN === undefined)('Edit diffs on the real agent', () => {
  it('"diff" on both agents gives the same diff blocks; live and stored give the same edits; near matches and a deletion', async () => {
    const steps = diffSteps();
    expect(steps).toHaveLength(5);
    const script = [...steps, ...EXTRA];

    const fakeRun = await fakeHost('chat-harness', { mode: 'work' });
    let fakeBlocks: Json[][];
    let fakeShapes: unknown[];
    try {
      const { sessionId } = await fakeRun.host.createSession({ modelId: MODEL, cwd: fakeRun.project });
      expect(await runAllowing(fakeRun, sessionId, 'diff')).toBe('completed');
      fakeBlocks = diffBlocks(fakeRun, sessionId, realpathSync(fakeRun.project));
      fakeShapes = shapes(liveTools(fakeRun, sessionId));
      expect(fakeBlocks).toHaveLength(5);
    } finally {
      await fakeRun.close();
    }

    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-diff-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    let offered: string[] = [];
    let calls = 0;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Diff probe' } }] };
          }
          offered = q.tools;
          if (calls > 0 && !hasToolResult(lastMessage(q))) return { blocks: [{ type: 'text', text: 'Done.' }] };
          const step = script[calls];
          calls += 1;
          return step === undefined ? { blocks: [{ type: 'text', text: 'Done.' }] } : callOf(step);
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const project = realpathSync(real.project);
      for (const step of script) {
        if (step.before === null) continue;
        mkdirSync(dirname(join(project, step.file_path)), { recursive: true });
        writeFileSync(join(project, step.file_path), step.before);
      }
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd: real.project });
      expect(await runAllowing(real, sessionId, 'Make the five edits.')).toBe('completed');
      expect(offered).toEqual(expect.arrayContaining(['search_replace', 'write']));

      expect(diffBlocks(real, sessionId, project).slice(0, 5)).toEqual(fakeBlocks);

      const live = liveTools(real, sessionId);
      expect(live).toHaveLength(7);
      expect(storedTools(real.agentHome, 'docs/notes.md')).toEqual(live);
      expect(shapes(live.slice(0, 5)).map((s) => (s as unknown[]).slice(0, 3))).toEqual(
        fakeShapes.map((s) => (s as unknown[]).slice(0, 3)),
      );
      expect(JSON.stringify(live)).not.toContain('EditsApplied');
      // Two matches whose context lines meet: one hunk, every line once.
      expect(live[5]!.edits?.[0]?.hunks).toEqual([
        {
          start: [1, 1],
          body: [' a', ' b', ' c', '-x = 1;', '+x = 2;', ' d', '-x = 1;', '+x = 2;', ' e', ' f', ' g'].join('\n'),
        },
      ]);
      // A deleted block: the agent's after context skips the line that took its place, so none is shown.
      expect(live[6]!.edits?.[0]?.hunks).toEqual([{ start: [1, 1], body: ' a\n b\n c\n-x\n-y' }]);
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  });
});
