/**
 * History ops on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model. Compact now with no
 * turn: the summary call is held, `session/cancel` stops it, the run ends `stopped` with no mark, no wake starts, the
 * live lines and `updates.jsonl` give the same transcript, and the next prompt runs. Rewind (conversation only): the
 * agent's file maps a live message to its prompt index (`targetOf`), and the file with the agent's marker gives the
 * live part before the removed prompt. Fork: the copy on disk is a fork of the source, cut inclusively at the target.
 * Rewind files: the dry run names the files the agent's own edit tool wrote, flags a hand edit
 * and a hand delete, writes nothing, and names the same files after an agent restart; "files only" puts them back and
 * keeps the conversation; "all" puts back the rest and writes the marker; a command's change and a helper's edit stay;
 * a fork's copy has no files to put back. Skipped without the variable.
 */

import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { AgentSessionsReader, MAX_LIST_LIMIT } from '../src/agent-sessions/reader.js';
import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { targetOf } from '../src/chat/rewind-target.js';
import { fakeHost, noProviders, until, type FakeHost, type Json } from './helpers/chat-fake.js';
import {
  hasToolResult,
  lastMessage,
  messageText,
  startScriptedEngine,
  type ModelReply,
  type ScriptedEngine,
} from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const MARKER = 'Compact probe: say hello.';
// The agent retries a summary under 500 characters as degenerate.
const SUMMARY = `<summary>\n1. Primary request: ${'the user asked the assistant to say hello. '.repeat(16)}\n</summary>`;

function stored(agentHome: string): TranscriptEntry[] {
  const files = readdirSync(agentHome, { recursive: true, encoding: 'utf8' })
    .filter((f) => f.endsWith('updates.jsonl'))
    .map((f) => readFileSync(join(agentHome, f), 'utf8'))
    .filter((text) => text.includes(MARKER));
  expect(files).toHaveLength(1);
  return parseUpdates(files[0]!).transcript;
}

/** Runs one turn and allows every request it makes (the helper's too), until it ends. */
async function runAllowing(fake: FakeHost, sessionId: string, text: string): Promise<string> {
  const turn = fake.host.startTurn(sessionId, text, { modelId: MODEL, cwd: fake.project });
  let answered = fake.events.interactions.length;
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

const tool = (name: string, input: Json): ModelReply => ({ blocks: [{ type: 'tool_use', name, input }] });

function live(fake: FakeHost, sessionId: string): TranscriptEntry[] {
  const reducer = new TranscriptReducer();
  for (const u of fake.events.updates) if (u.sessionId === sessionId) reducer.apply(u.line);
  return reducer.snapshot().entries;
}

describe.skipIf(BIN === undefined)('History ops on the real agent', () => {
  it('a manual compact with no turn is stopped by session/cancel', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-compact-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    let compacting = false;
    let engine: ScriptedEngine | null = null;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: async () => {
        engine = await startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Compact probe' } }] };
          }
          if (compacting) return { blocks: [{ type: 'text', text: 'Summary.' }], holdMs: 60_000 };
          return { blocks: [{ type: 'text', text: 'Hello.' }], inputTokens: 4_000 };
        });
        return engine;
      },
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const load = { modelId: MODEL, cwd: real.project };
      const { sessionId } = await real.host.createSession(load);
      expect((await real.host.startTurn(sessionId, MARKER, load).done).outcome).toBe('completed');
      const calls = engine!.modelRequests.length;
      const turnsBefore = real.events.turns.length;

      compacting = true;
      const run = real.host.compact(sessionId, load, null);
      await until(
        () => engine!.modelRequests.some((q) => q.index >= calls && q.tools[0] !== 'session_title'),
        AGENT_WAIT_MS,
      );
      const summary = engine!.modelRequests.find((q) => q.index >= calls && q.tools[0] !== 'session_title')!;
      expect(real.host.isBusy(sessionId)).toBe(true);
      expect(real.events.compactions.at(-1)).toMatchObject({ sessionId, compaction: { phase: 'running' } });

      real.host.cancel(sessionId);
      expect(await run.done).toBe('stopped');
      await until(() => summary.closedEarly || summary.doneAt !== null, AGENT_WAIT_MS);
      expect(summary.closedEarly).toBe(true);
      expect(
        real.events.compactions
          .filter((c) => c.sessionId === sessionId)
          .map((c) => [c.compaction?.phase ?? null, c.end]),
      ).toEqual([
        ['running', undefined],
        ['stopping', undefined],
        [null, 'stopped'],
      ]);
      expect(live(real, sessionId).some((e) => e.kind === 'compaction')).toBe(false);
      expect(real.host.isBusy(sessionId)).toBe(false);
      expect(real.host.engineBusy(sessionId)).toBe(false);

      await new Promise((resolve) => setTimeout(resolve, 1_000));
      expect(real.events.turns.slice(turnsBefore)).toEqual([]);
      // The agent echoes no prompt on the live stream; the page draws its own.
      const disk = stored(real.agentHome);
      expect(disk.filter((e) => e.kind !== 'user')).toEqual(live(real, sessionId));
      expect(disk.some((e) => e.kind === 'compaction')).toBe(false);

      compacting = false;
      expect((await real.host.startTurn(sessionId, 'Again.', load).done).outcome).toBe('completed');
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  });

  it('a manual compact that ends well: its line before the reply, one mark, the same entry live and from disk', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-compact-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    let compacting = false;
    let engine: ScriptedEngine | null = null;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: async () => {
        engine = await startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Compact probe' } }] };
          }
          if (compacting) return { blocks: [{ type: 'text', text: SUMMARY }] };
          return { blocks: [{ type: 'text', text: 'Hello.' }], inputTokens: 4_000 };
        });
        return engine;
      },
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const load = { modelId: MODEL, cwd: real.project };
      const { sessionId } = await real.host.createSession(load);
      expect((await real.host.startTurn(sessionId, MARKER, load).done).outcome).toBe('completed');
      const turnsBefore = real.events.turns.length;

      compacting = true;
      expect(await real.host.compact(sessionId, load, null).done).toBe('compacted');
      expect(
        real.events.compactions
          .filter((c) => c.sessionId === sessionId)
          .map((c) => [c.compaction?.phase ?? null, c.end]),
      ).toEqual([
        ['running', undefined],
        [null, 'compacted'],
      ]);
      expect(real.events.timeline.slice(-2)).toEqual(['auto_compact_completed', 'compaction end']);
      const marks = live(real, sessionId).filter((e) => e.kind === 'compaction');
      expect(marks).toEqual([
        {
          kind: 'compaction',
          ts: expect.any(Number),
          outcome: 'done',
          tokensBefore: expect.any(Number),
          tokensAfter: expect.any(Number),
        },
      ]);
      expect(real.host.isBusy(sessionId)).toBe(false);
      expect(real.events.turns.slice(turnsBefore)).toEqual([]);

      // The line rides the agent's merge buffer: the file catches up after the next turn, in its own time.
      compacting = false;
      expect((await real.host.startTurn(sessionId, 'Again.', load).done).outcome).toBe('completed');
      const spoken = live(real, sessionId);
      await until(() => stored(real.agentHome).filter((e) => e.kind !== 'user').length >= spoken.length, AGENT_WAIT_MS);
      const disk = stored(real.agentHome);
      expect(disk.filter((e) => e.kind === 'compaction')).toEqual(marks);
      expect(disk.filter((e) => e.kind !== 'user')).toEqual(spoken);
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  });

  // Rewind to prompt 1 (conversation only): the agent writes its marker, the file read gives the live part before the
  // second prompt, and the next prompt runs without the removed one in the model's messages.
  it('a rewind to the second prompt: the marker on disk, the disk equals the live cut, the next prompt runs', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-rewind-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    let engine: ScriptedEngine | null = null;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: async () => {
        engine = await startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Rewind probe' } }] };
          }
          return { blocks: [{ type: 'text', text: 'Hello.' }], inputTokens: 4_000 };
        });
        return engine;
      },
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const load = { modelId: MODEL, cwd: real.project };
      const { sessionId } = await real.host.createSession(load);
      expect((await real.host.startTurn(sessionId, MARKER, load).done).outcome).toBe('completed');
      const second = 'Second prompt to remove.';
      expect((await real.host.startTurn(sessionId, second, load).done).outcome).toBe('completed');
      const spoken = live(real, sessionId);
      const firstEnd = spoken.findIndex((e) => e.kind === 'turn-end');
      expect(firstEnd).toBeGreaterThan(0);
      const cut = spoken.slice(0, firstEnd + 1);
      // The page's view as the host pushes it (live user rows carry no prompt index), mapped against the agent's file.
      const view: TranscriptEntry[] = [
        { kind: 'user', ts: 0, text: MARKER, promptIndex: null },
        ...cut,
        { kind: 'user', ts: 0, text: second, promptIndex: null },
        ...spoken.slice(firstEnd + 1),
      ];
      const typed = (entries: TranscriptEntry[]): string[] =>
        entries.flatMap((e) => (e.kind === 'user' ? [e.text] : []));
      await until(() => typed(stored(real.agentHome)).length === 2, AGENT_WAIT_MS);
      const fileBefore = stored(real.agentHome);
      expect(fileBefore.flatMap((e) => (e.kind === 'user' ? [e.promptIndex] : []))).toEqual([0, 1]);
      expect(targetOf(view, fileBefore, cut.length + 1)).toEqual({ target: 1 });
      expect(targetOf(view, fileBefore, 0)).toEqual({ target: 0 });

      const run = real.host.rewind(sessionId, load, {
        target: 1,
        mode: 'conversation',
        tasks: [],
        schedules: [],
        runs: [],
        files: [],
      });
      expect(await run.done).toEqual({ outcome: 'done', restored: 0 });
      expect(real.host.isBusy(sessionId)).toBe(false);
      const file = (): string =>
        readdirSync(real.agentHome, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(real.agentHome, f), 'utf8'))
          .find((text) => text.includes(MARKER)) ?? '';
      await until(() => file().includes('rewind_marker'), AGENT_WAIT_MS);
      const disk = stored(real.agentHome);
      expect(disk.filter((e) => e.kind === 'user').map((e) => (e.kind === 'user' ? e.text : ''))).toEqual([MARKER]);
      expect(disk.filter((e) => e.kind !== 'user')).toEqual(cut);

      const before = engine!.modelRequests.length;
      expect((await real.host.startTurn(sessionId, 'Third.', load).done).outcome).toBe('completed');
      const asked = engine!.modelRequests.filter((q) => q.index >= before && q.tools[0] !== 'session_title');
      expect(asked.length).toBeGreaterThan(0);
      const sentText = JSON.stringify(asked[0]!.messages);
      expect(sentText).toContain('Third.');
      expect(sentText).not.toContain(second);
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  });

  // Fork: the agent's reply names the copy; its summary says `fork` of the source; `targetPromptIndex` is inclusive
  // (prompt 0 and its answer are kept); the copy is not loaded, and its first prompt loads it without the cut prompt.
  it('a fork: the whole copy and one up to the first answer, both on disk as forks; the cut copy answers', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-fork-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    let engine: ScriptedEngine | null = null;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: async () => {
        engine = await startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Fork probe' } }] };
          }
          return { blocks: [{ type: 'text', text: 'Hello.' }], inputTokens: 4_000 };
        });
        return engine;
      },
      env: { TMPDIR: `${agentTmp}/` },
    });
    const folderOf = (id: string): string => {
      const found = readdirSync(real.agentHome, { recursive: true, encoding: 'utf8' }).find((f) =>
        f.endsWith(join(id, 'summary.json')),
      );
      expect(found).toBeDefined();
      return join(real.agentHome, found!, '..');
    };
    try {
      const load = { modelId: MODEL, cwd: real.project };
      const { sessionId } = await real.host.createSession(load);
      expect((await real.host.startTurn(sessionId, MARKER, load).done).outcome).toBe('completed');
      const second = 'Second prompt to leave out.';
      expect((await real.host.startTurn(sessionId, second, load).done).outcome).toBe('completed');
      await until(
        () => parseUpdates(readFileSync(join(folderOf(sessionId), 'updates.jsonl'), 'utf8')).transcript.length > 4,
        AGENT_WAIT_MS,
      );

      const whole = await real.host.fork(sessionId, load, null);
      const cut = await real.host.fork(sessionId, load, 0);
      expect(real.host.isBusy(sessionId)).toBe(false);
      const typed = (id: string): string[] =>
        parseUpdates(readFileSync(join(folderOf(id), 'updates.jsonl'), 'utf8')).transcript.flatMap((e) =>
          e.kind === 'user' ? [e.text] : [],
        );
      expect(typed(whole)).toEqual([MARKER, second]);
      expect(typed(cut)).toEqual([MARKER]);
      const cutEntries = parseUpdates(readFileSync(join(folderOf(cut), 'updates.jsonl'), 'utf8')).transcript;
      expect(cutEntries.some((e) => e.kind === 'assistant')).toBe(true);
      for (const id of [whole, cut]) {
        const summary = JSON.parse(readFileSync(join(folderOf(id), 'summary.json'), 'utf8')) as Record<string, unknown>;
        expect(summary).toMatchObject({ session_kind: 'fork', parent_session_id: sessionId });
        expect((summary.info as { cwd: string }).cwd).toBe(realpathSync(real.project));
      }
      // The app's reader over the agent's own copy: the list row and the detail name the source.
      const reader = new AgentSessionsReader({ agentHome: real.agentHome });
      const rows = reader.list({ limit: MAX_LIST_LIMIT }).sessions;
      const source = rows.find((r) => r.id === sessionId);
      expect(source).toMatchObject({ kind: null, parentId: null });
      expect(source!.title).not.toBe('');
      for (const id of [whole, cut]) {
        expect(rows.find((r) => r.id === id)).toMatchObject({
          kind: 'fork',
          parentId: sessionId,
          title: source!.title,
        });
        expect(reader.detail(id)!.session.parent).toEqual({ id: sessionId, title: source!.title });
      }
      expect(real.host.isHot(cut)).toBe(false);

      const before = engine!.modelRequests.length;
      expect((await real.host.startTurn(cut, 'Third.', load).done).outcome).toBe('completed');
      const asked = engine!.modelRequests.filter((q) => q.index >= before && q.tools[0] !== 'session_title');
      const sentText = JSON.stringify(asked[0]!.messages);
      expect(sentText).toContain(MARKER);
      expect(sentText).toContain('Third.');
      expect(sentText).not.toContain(second);

      await expect(real.host.fork('00000000-0000-7000-8000-000000000000', load, null)).rejects.toMatchObject({
        code: 'agent-protocol-error',
      });
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  });

  // The gate of the file choices. Prompt 0 edits `kept.txt`; prompt 1 edits `a.txt` and `hand.txt`, writes the
  // new `new.txt`, changes `shell.txt` with a command and has a helper edit `helper.txt`; then `hand.txt` is changed by
  // hand and `new.txt` deleted by hand. Only the files the agent's own edit tool wrote come back; a hand edit or delete
  // is a conflict that a restore overwrites.
  it('rewind files: the dry run, files only, then all; a command change and a helper edit stay; a fork has none', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-rewind-files-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const PARENT = 'FILEPROBE';
    const CHILD = 'CHILDPROBE: change h1 to h2 in helper.txt, then say done.';
    let steps: ModelReply[] = [];
    let helperAsked = 0;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Files probe' } }] };
          }
          const last = lastMessage(q);
          const all = JSON.stringify(q.messages);
          if (!all.includes(PARENT)) {
            if (!all.includes(CHILD) || hasToolResult(last)) return { blocks: [{ type: 'text', text: 'Done.' }] };
            helperAsked += 1;
            return tool('search_replace', { file_path: 'helper.txt', old_string: 'h1', new_string: 'h2' });
          }
          if (!q.tools.includes('search_replace')) return { blocks: [{ type: 'text', text: 'Done.' }] };
          if (hasToolResult(last) || messageText(last).includes(PARENT)) {
            const next = steps.shift();
            if (next !== undefined) return next;
          }
          return { blocks: [{ type: 'text', text: 'Done.' }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const project = realpathSync(real.project);
      const at = (name: string): string => join(project, name);
      const read = (name: string): string | null => (existsSync(at(name)) ? readFileSync(at(name), 'utf8') : null);
      for (const [name, text] of [
        ['kept.txt', 'k1\n'],
        ['a.txt', 'a1\n'],
        ['hand.txt', 'x1\n'],
        ['shell.txt', 's1\n'],
        ['helper.txt', 'h1\n'],
      ] as const)
        writeFileSync(at(name), text);
      const load = { modelId: MODEL, cwd: real.project };
      const { sessionId } = await real.host.createSession(load);

      steps = [tool('search_replace', { file_path: 'kept.txt', old_string: 'k1', new_string: 'k2' })];
      expect(await runAllowing(real, sessionId, `${PARENT} 0: edit kept.txt.`)).toBe('completed');
      steps = [
        tool('search_replace', { file_path: 'a.txt', old_string: 'a1', new_string: 'a2' }),
        tool('search_replace', { file_path: 'hand.txt', old_string: 'x1', new_string: 'x2' }),
        tool('write', { file_path: 'new.txt', content: 'n1\n' }),
        tool('run_terminal_command', { command: "printf 's2\\n' > shell.txt", description: 'shell' }),
        tool('spawn_subagent', {
          description: 'helper',
          prompt: CHILD,
          subagent_type: 'general-purpose',
          run_in_background: false,
        }),
      ];
      expect(await runAllowing(real, sessionId, `${PARENT} 1: edit, write, run, delegate.`)).toBe('completed');
      expect(steps).toEqual([]);
      expect(helperAsked).toBeGreaterThan(0);
      expect([
        read('kept.txt'),
        read('a.txt'),
        read('hand.txt'),
        read('new.txt'),
        read('shell.txt'),
        read('helper.txt'),
      ]).toEqual(['k2\n', 'a2\n', 'x2\n', 'n1\n', 's2\n', 'h2\n']);
      writeFileSync(at('hand.txt'), 'x3\n');
      rmSync(at('new.txt'));
      // The helper's end reaches the host after the answer's.
      await until(() => !real.host.engineBusy(sessionId), AGENT_WAIT_MS);

      // A fork's copy holds the prompts but no file of them: its dry run names none (the dialog offers the
      // conversation only).
      const copy = await real.host.fork(sessionId, load, null);
      expect(await real.host.rewindPreview(copy, load, 1)).toEqual({ files: [] });
      expect(await real.host.rewindPreview(copy, load, 5)).toEqual({ refused: true });

      const byPath = (files: ReadonlyArray<{ path: string; state: string }>) =>
        [...files].sort((x, y) => (x.path < y.path ? -1 : 1));
      const preview = await real.host.rewindPreview(sessionId, load, 1);
      expect(real.host.isBusy(sessionId)).toBe(false);
      expect('files' in preview && byPath(preview.files)).toEqual([
        { path: 'a.txt', state: 'clean' },
        { path: 'hand.txt', state: 'modified' },
        { path: 'new.txt', state: 'deleted' },
      ]);
      expect(await real.host.rewindPreview(sessionId, load, 2)).toEqual({ refused: true });
      const whole = await real.host.rewindPreview(sessionId, load, 0);
      expect('files' in whole && byPath(whole.files)).toEqual([
        { path: 'a.txt', state: 'clean' },
        { path: 'hand.txt', state: 'modified' },
        { path: 'kept.txt', state: 'clean' },
        { path: 'new.txt', state: 'deleted' },
      ]);
      // A dry run writes nothing.
      expect([read('kept.txt'), read('a.txt'), read('hand.txt'), read('new.txt')]).toEqual([
        'k2\n',
        'a2\n',
        'x3\n',
        null,
      ]);
      // The points outlive the agent: a new process loads the cold session and names the same files.
      await real.host.restart();
      expect(real.host.isHot(sessionId)).toBe(false);
      const cold = await real.host.rewindPreview(sessionId, load, 0);
      expect('files' in cold && byPath(cold.files)).toEqual('files' in whole && byPath(whole.files));

      // Files only: the three files back as they were before prompt 1 (the hand edit overwritten, the new file gone);
      // the command's change and the helper's edit stay; the conversation stays, with no marker.
      const updates = (): string => {
        const found = readdirSync(real.agentHome, { recursive: true, encoding: 'utf8' }).find((f) =>
          f.endsWith(join(sessionId, 'updates.jsonl')),
        );
        return found === undefined ? '' : readFileSync(join(real.agentHome, found), 'utf8');
      };
      const files = await real.host.rewind(sessionId, load, {
        target: 1,
        mode: 'files',
        tasks: [],
        schedules: [],
        runs: [],
        files: 'files' in preview ? preview.files : [],
      }).done;
      expect(files).toEqual({ outcome: 'done', restored: 3 });
      expect([
        read('kept.txt'),
        read('a.txt'),
        read('hand.txt'),
        read('new.txt'),
        read('shell.txt'),
        read('helper.txt'),
      ]).toEqual(['k2\n', 'a1\n', 'x1\n', null, 's2\n', 'h2\n']);
      expect(updates()).not.toContain('rewind_marker');
      expect(await real.host.rewindPreview(sessionId, load, 1)).toEqual({ files: [] });
      const typed = (): string[] =>
        parseUpdates(updates()).transcript.flatMap((e) => (e.kind === 'user' ? [e.text] : []));
      expect(typed()).toHaveLength(2);

      // All, to the first prompt: kept.txt back too, and the marker written.
      const first = await real.host.rewindPreview(sessionId, load, 0);
      expect(first).toEqual({ files: [{ path: 'kept.txt', state: 'clean' }] });
      const all = await real.host.rewind(sessionId, load, {
        target: 0,
        mode: 'all',
        tasks: [],
        schedules: [],
        runs: [],
        files: 'files' in first ? first.files : [],
      }).done;
      expect(all).toEqual({ outcome: 'done', restored: 1 });
      expect([read('kept.txt'), read('a.txt'), read('shell.txt'), read('helper.txt')]).toEqual([
        'k1\n',
        'a1\n',
        's2\n',
        'h2\n',
      ]);
      await until(() => updates().includes('rewind_marker'), AGENT_WAIT_MS);
      expect(typed()).toEqual([]);
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);

  // The agent's commit overwrites any file it finds changed, and answers success when a file could not be written (and
  // drops the points it would need to try again). A file changed after the plan is no file the user agreed to
  // overwrite: the rewind stops before it changes anything. A file the restore could not write is named.
  it('rewind files: a hand edit after the plan stops the restore; a file that cannot be written is named', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-rewind-commit-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const PARENT = 'COMMITPROBE';
    let steps: ModelReply[] = [];
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Commit probe' } }] };
          }
          const last = lastMessage(q);
          if (!q.tools.includes('search_replace') || !JSON.stringify(q.messages).includes(PARENT)) {
            return { blocks: [{ type: 'text', text: 'Done.' }] };
          }
          if (hasToolResult(last) || messageText(last).includes(PARENT)) {
            const next = steps.shift();
            if (next !== undefined) return next;
          }
          return { blocks: [{ type: 'text', text: 'Done.' }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    const project = realpathSync(real.project);
    const at = (name: string): string => join(project, name);
    try {
      const read = (name: string): string => readFileSync(at(name), 'utf8');
      for (const [name, text] of [
        ['a.txt', 'a1\n'],
        ['b.txt', 'b1\n'],
      ] as const)
        writeFileSync(at(name), text);
      const load = { modelId: MODEL, cwd: real.project };
      const { sessionId } = await real.host.createSession(load);
      expect(await runAllowing(real, sessionId, `${PARENT} 0: nothing.`)).toBe('completed');
      steps = [
        tool('search_replace', { file_path: 'a.txt', old_string: 'a1', new_string: 'a2' }),
        tool('search_replace', { file_path: 'b.txt', old_string: 'b1', new_string: 'b2' }),
      ];
      expect(await runAllowing(real, sessionId, `${PARENT} 1: edit a and b.`)).toBe('completed');
      expect([read('a.txt'), read('b.txt')]).toEqual(['a2\n', 'b2\n']);

      const byPath = (files: ReadonlyArray<{ path: string; state: string }>) =>
        [...files].sort((x, y) => (x.path < y.path ? -1 : 1));
      const plan = await real.host.rewindPreview(sessionId, load, 1);
      expect('files' in plan && byPath(plan.files)).toEqual([
        { path: 'a.txt', state: 'clean' },
        { path: 'b.txt', state: 'clean' },
      ]);
      const files = 'files' in plan ? plan.files : [];

      // b.txt is edited by hand after the plan showed it clean: nothing is put back, and the conversation stays.
      writeFileSync(at('b.txt'), 'b3\n');
      for (const mode of ['files', 'all'] as const) {
        const changed = await real.host.rewind(sessionId, load, {
          target: 1,
          mode,
          tasks: [],
          schedules: [],
          runs: [],
          files,
        }).done;
        expect(changed).toEqual({ outcome: 'changed' });
        expect([read('a.txt'), read('b.txt')]).toEqual(['a2\n', 'b3\n']);
      }
      expect(real.host.isBusy(sessionId)).toBe(false);

      // The plan again shows the hand edit; the user agrees to overwrite it. a.txt cannot be written: the agent skips
      // it and still answers success; the host names it.
      const again = await real.host.rewindPreview(sessionId, load, 1);
      expect('files' in again && byPath(again.files)).toEqual([
        { path: 'a.txt', state: 'clean' },
        { path: 'b.txt', state: 'modified' },
      ]);
      chmodSync(at('a.txt'), 0o444);
      const partial = await real.host.rewind(sessionId, load, {
        target: 1,
        mode: 'files',
        tasks: [],
        schedules: [],
        runs: [],
        files: 'files' in again ? again.files : [],
      }).done;
      expect(partial).toEqual({ outcome: 'partial', restored: 1, failed: ['a.txt'] });
      expect([read('a.txt'), read('b.txt')]).toEqual(['a2\n', 'b1\n']);
    } finally {
      await real.close();
      if (existsSync(at('a.txt'))) chmodSync(at('a.txt'), 0o644);
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});
