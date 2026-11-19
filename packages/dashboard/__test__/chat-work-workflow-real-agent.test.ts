/**
 * Work workflows on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: the child's
 * tools and prompt, its guarded ask with the requester, the report and the wake, Stop run, the Chat switch, the
 * session header, the context count, and an auto compaction. Skipped without the variable.
 * `MLX_AGENT_E2E_RECORD` (a folder) keeps each case's raw wire, engine requests and host events there.
 */

import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { TranscriptReducer, type UpdateLine } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { spawnAgent, type SpawnAgent } from '../src/chat/acp/process.js';
import { AGENT_ARGS, WORK_RULES, workGuardHookName } from '../src/chat/agent/profiles.js';
import type { PermissionInteraction } from '../src/chat/types.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { compactionText } from '../ui/src/lib/chat-turns.js';
import { formatCount } from '../ui/src/lib/format.js';
import { fakeHost, noProviders, until, type FakeHost, type FakeHostOptions, type Json } from './helpers/chat-fake.js';
import {
  DEFAULT_INPUT_TOKENS,
  DEFAULT_OUTPUT_TOKENS,
  hasToolResult,
  lastMessage,
  messageText,
  startScriptedEngine,
  type ModelReply,
  type ModelRequest,
  type ModelScript,
  type ScriptedEngine,
} from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const RECORD = process.env.MLX_AGENT_E2E_RECORD;
const MODEL = 'qwen3.5-4b';
const WINDOW = 262_144;
const PROMPT = 'PROBE: start the probe workflow.';
const SCRIPT = [
  'let meta = #{ name: "probe", description: "probe" };',
  'let r = agent("probe", #{ label: "probe-0" });',
  'complete(r.output);',
].join('\n');
const REPORT = 'Probe report: the folder was listed.';
const LISTING = 'The following workflows are available';
const SESSION_HEADER = 'x-grok-session-id';
const AGENT_WAIT_MS = 30_000;

/** The agent's stdout lines and the host's stdin writes, parsed, in order. */
interface Wire {
  out: Json[];
  in: Json[];
  pids: number[];
  stderr: string[];
}

function teeSpawn(wire: Wire): SpawnAgent {
  return (command) => {
    const child = spawnAgent(command);
    if (child.pid !== undefined) wire.pids.push(child.pid);
    const lines = (into: Json[]): ((chunk: Buffer | string) => void) => {
      let partial = '';
      return (chunk) => {
        partial += chunk.toString();
        let nl = partial.indexOf('\n');
        while (nl !== -1) {
          const line = partial.slice(0, nl);
          partial = partial.slice(nl + 1);
          if (line.trim() !== '') {
            try {
              into.push(JSON.parse(line) as Json);
            } catch {
              into.push({ unparsed: line });
            }
          }
          nl = partial.indexOf('\n');
        }
      };
    };
    child.stdout?.on('data', lines(wire.out));
    child.stderr?.on('data', (chunk: Buffer) => wire.stderr.push(chunk.toString()));
    const stdin = child.stdin;
    if (stdin !== null) {
      const write = stdin.write.bind(stdin) as (chunk: unknown, ...rest: unknown[]) => boolean;
      const record = lines(wire.in);
      stdin.write = ((chunk: unknown, ...rest: unknown[]) => {
        if (typeof chunk === 'string' || Buffer.isBuffer(chunk)) record(chunk);
        return write(chunk, ...rest);
      }) as typeof stdin.write;
    }
    return child;
  };
}

interface Rig {
  fake: FakeHost;
  engine: ScriptedEngine;
  wire: Wire;
  /** The probe provider skill's `SKILL.md`. */
  skill: string;
  outer: string;
}

async function rig(
  reply: ModelScript,
  options: { env?: Record<string, string>; window?: number; host?: FakeHostOptions } = {},
): Promise<Rig> {
  const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-')));
  const pluginDir = join(outer, 'probe-plugin');
  const skillDir = join(pluginDir, 'skills', 'probe-web');
  mkdirSync(skillDir, { recursive: true });
  const skill = join(skillDir, 'SKILL.md');
  writeFileSync(
    skill,
    '---\nname: probe-web\ndescription: Probe skill of the workflow test. Search the web with it.\n---\n\n# Probe web\n\nNothing to do.\n',
  );
  const agentTmp = join(outer, 'tmp');
  mkdirSync(agentTmp);
  const wire: Wire = { out: [], in: [], pids: [], stderr: [] };
  let engine: ScriptedEngine | null = null;
  const fake = await fakeHost('work-workflow-run', {
    mode: 'work',
    providers: noProviders({ workPluginDirs: () => [pluginDir] }),
    launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
    startEngine: async () => {
      engine = await startScriptedEngine([{ id: MODEL, contextWindow: options.window ?? WINDOW }], reply);
      return engine;
    },
    env: { TMPDIR: `${agentTmp}/`, ...options.env },
    spawn: teeSpawn(wire),
    ...options.host,
  });
  return { fake, engine: engine!, wire, skill, outer };
}

function listFiles(dir: string, base = dir): string[] {
  const out: string[] = [];
  let names: string[] = [];
  try {
    names = readdirSync(dir);
  } catch {
    return out;
  }
  for (const name of names) {
    const path = join(dir, name);
    let sub: string[] | null = null;
    try {
      sub = readdirSync(path);
    } catch {
      sub = null;
    }
    if (sub === null) out.push(path.slice(base.length + 1));
    else out.push(...listFiles(path, base));
  }
  return out;
}

async function finish(name: string, r: Rig, extra: Json = {}): Promise<void> {
  if (RECORD !== undefined) {
    mkdirSync(RECORD, { recursive: true });
    const record = {
      engine: {
        modelPath: r.engine.modelPath(),
        strays: r.engine.strays.map((s) => ({ method: s.method, path: s.path })),
        other: r.engine.requests
          .filter((q) => q.path !== r.engine.modelPath())
          .map((q) => ({ method: q.method, path: q.path, headers: q.headers })),
        model: r.engine.modelRequests,
      },
      wire: { out: r.wire.out, in: r.wire.in, pids: r.wire.pids },
      stderr: r.wire.stderr.join('').slice(-20_000),
      events: {
        updates: r.fake.events.updates,
        interactions: r.fake.events.interactions,
        turns: r.fake.events.turns,
        ends: r.fake.events.ends,
        runs: r.fake.events.runs,
        compactions: r.fake.events.compactions,
        timeline: r.fake.events.timeline,
        closed: r.fake.events.closed,
      },
      agentHomeFiles: listFiles(r.fake.agentHome),
      ...extra,
    };
    writeFileSync(join(RECORD, `${name}.json`), `${JSON.stringify(record, null, 2)}\n`);
  }
  await r.fake.close();
  rmSync(r.outer, { recursive: true, force: true });
}

async function session(fake: FakeHost): Promise<string> {
  return (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
}

function start(fake: FakeHost, id: string, text: string) {
  return fake.host.startTurn(id, text, { modelId: MODEL, cwd: fake.project });
}

async function nthCard(fake: FakeHost, n: number): Promise<PermissionInteraction> {
  await until(() => fake.events.interactions.length >= n, AGENT_WAIT_MS);
  const interaction = fake.events.interactions[n - 1]!.interaction;
  expect(interaction.kind).toBe('permission');
  return interaction as PermissionInteraction;
}

function guardHook(fake: FakeHost): string {
  const manifest = JSON.parse(readFileSync(join(fake.paths.workGuard, 'plugin.json'), 'utf8')) as { name: string };
  return workGuardHookName(manifest.name);
}

/** The `x.ai/requester` asks the agent wrote. */
function childAsks(wire: Wire): Json[] {
  return wire.out.filter((m) => {
    if (m.method !== 'session/request_permission') return false;
    const meta = (m.params as Json)._meta as Json | undefined;
    return meta?.['x.ai/requester'] !== undefined;
  });
}

function xaiUpdates(wire: Wire, kind: string): Json[] {
  return wire.out
    .filter((m) => m.method === '_x.ai/session_notification')
    .map((m) => (m.params as Json).update as Json)
    .filter((u) => u.sessionUpdate === kind);
}

function linesOf(fake: FakeHost, id: string): UpdateLine[] {
  return fake.events.updates.filter((u) => u.sessionId === id).map((u) => u.line);
}

function reduce(lines: readonly UpdateLine[]): TranscriptReducer {
  const reducer = new TranscriptReducer();
  for (const line of lines) reducer.apply(line);
  return reducer;
}

function entriesOf(fake: FakeHost, id: string): TranscriptEntry[] {
  return reduce(linesOf(fake, id)).snapshot().entries;
}

/** The agent's side call that names the session: its only tool is `session_title`, and it must call it. */
const isTitle = (q: ModelRequest): boolean => q.tools.length === 1 && q.tools[0] === 'session_title';
const TITLE_REPLY: ModelReply = {
  blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Probe session' } }],
};
const isParent = (q: ModelRequest): boolean => q.tools.includes('workflow');
const isChild = (q: ModelRequest): boolean => !isTitle(q) && !isParent(q);

/** Parent: the probe prompt starts the workflow, its result ends the turn, the wake reads the report. */
function parentReply(q: ModelRequest): ModelReply {
  const last = lastMessage(q);
  if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'The probe runs on its own.' }] };
  if (messageText(last).includes(PROMPT)) {
    return { blocks: [{ type: 'tool_use', name: 'workflow', input: { source: { type: 'script', script: SCRIPT } } }] };
  }
  return { blocks: [{ type: 'text', text: 'Report read.' }] };
}

function stripCacheControl(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripCacheControl);
  if (typeof value !== 'object' || value === null) return value;
  const out: Json = {};
  for (const [k, v] of Object.entries(value)) if (k !== 'cache_control') out[k] = stripCacheControl(v);
  return out;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

describe.skipIf(BIN === undefined)('Work workflows on the real agent', () => {
  it('a run: the child tools, prompt and skill; its guarded ask with the requester; report, wake, idle stop', async () => {
    const r = await rig(
      (q) => {
        if (isTitle(q)) return TITLE_REPLY;
        if (isParent(q)) return parentReply(q);
        if (hasToolResult(lastMessage(q))) return { blocks: [{ type: 'text', text: REPORT }] };
        return {
          holdMs: 1_000,
          blocks: [{ type: 'tool_use', name: 'run_terminal_command', input: { command: 'ls', description: 'list' } }],
        };
      },
      { host: { timings: { idleStopMs: 300 } } },
    );
    const notes: Json = {};
    try {
      const { fake, engine, wire } = r;
      const id = await session(fake);
      const turn = start(fake, id, PROMPT);
      const startCard = await nthCard(fake, 1);
      expect(startCard.toolName).toBe('workflow');
      fake.host.answer(id, startCard.id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
      const typedEnd = Date.now();

      const helperCard = await nthCard(fake, 2);
      notes.askAfterTypedEndMs = Date.now() - typedEnd;
      expect(fake.host.turnState(id)).toBeNull();
      expect(fake.host.modeState().state).toBe('ready');
      expect(() => process.kill(wire.pids[0]!, 0)).not.toThrow();
      expect(helperCard.from?.label).toBe('probe-0');

      const asks = childAsks(wire);
      expect(asks).toHaveLength(1);
      const params = asks[0]!.params as Json;
      const meta = params._meta as Json;
      const requester = meta['x.ai/requester'] as Json;
      const runs = xaiUpdates(wire, 'workflow_updated');
      const runId = runs[0]!.run_id as string;
      expect(params.sessionId).toBe(id);
      expect((meta.hookAsk as Json).hookName).toBe(guardHook(fake));
      expect(requester).toEqual({
        sessionId: requester.sessionId,
        label: 'probe-0',
        runId,
        cwd: realpathSync(fake.project),
      });
      expect(requester.sessionId).not.toBe(id);
      const childId = requester.sessionId as string;
      const firstChildAsk = wire.out.indexOf(asks[0]!);
      const firstActive = wire.out.findIndex(
        (m) =>
          m.method === '_x.ai/session_notification' &&
          ((m.params as Json).update as Json).sessionUpdate === 'workflow_updated' &&
          ((m.params as Json).update as Json).status === 'active',
      );
      expect(firstActive).toBeGreaterThanOrEqual(0);
      expect(firstActive).toBeLessThan(firstChildAsk);

      const child = engine.modelRequests.find(isChild)!;
      notes.childTools = child.tools;
      for (const tool of ['run_terminal_command', 'search_replace', 'write', 'monitor', 'search_tool', 'use_tool']) {
        expect(child.tools).toContain(tool);
      }
      for (const tool of ['workflow', 'spawn_subagent', 'scheduler_create']) expect(child.tools).not.toContain(tool);
      expect(child.system).toContain('<human_rules>');
      expect(child.system).toContain(WORK_RULES.split('\n')[0]!);
      notes.skillInChildFirstRequest = {
        system: child.system.includes(r.skill),
        messages: JSON.stringify(child.messages).includes(r.skill),
      };

      const parents = engine.modelRequests.filter(isParent);
      notes.parentHeaders = parents.map((q) => q.headers[SESSION_HEADER] ?? null);
      notes.childHeaders = engine.modelRequests.filter(isChild).map((q) => q.headers[SESSION_HEADER] ?? null);
      notes.titleHeaders = engine.modelRequests
        .filter(isTitle)
        .map((q) => ({ header: q.headers[SESSION_HEADER] ?? null, model: q.body.model }));
      notes.ids = { parent: id, child: childId };
      for (const q of parents) expect(q.headers[SESSION_HEADER]).toBe(id);
      notes.listingIn = {
        system: parents[0]!.system.includes(LISTING),
        body: JSON.stringify(parents[0]!.body).includes(LISTING),
      };
      expect(JSON.stringify(parents[0]!.body)).toContain(LISTING);

      const contexts = fake.events.updates
        .filter((u) => u.sessionId === id)
        .map((u) => ({ kind: String(u.line.update.sessionUpdate), totalTokens: u.line.meta?.totalTokens }));
      notes.contexts = contexts;
      const answerTotal = DEFAULT_INPUT_TOKENS + DEFAULT_OUTPUT_TOKENS;
      const firstAnswer = contexts.findIndex((c) => c.kind === 'response_started');
      expect(firstAnswer).toBeGreaterThanOrEqual(0);
      expect(contexts.slice(firstAnswer).find((c) => typeof c.totalTokens === 'number')?.totalTokens).toBe(answerTotal);
      expect(reduce(linesOf(fake, id)).context()).toBe(answerTotal);

      fake.host.answer(id, helperCard.id, 'allow-once');
      await until(() => xaiUpdates(wire, 'workflow_updated').some((u) => u.status === 'complete'), AGENT_WAIT_MS);
      notes.skillInChildRequests = engine.modelRequests.filter(isChild).map((q) => ({
        index: q.index,
        inSystem: q.system.includes(r.skill),
        inMessages: JSON.stringify(q.messages).includes(r.skill),
      }));
      const run = entriesOf(fake, id).find((e) => e.kind === 'workflow');
      expect(run).toMatchObject({ kind: 'workflow', runId, status: 'complete' });
      expect(run?.kind === 'workflow' ? run.report : undefined).toContain(REPORT);
      await until(() => fake.events.turns.some((t) => t.turn?.origin === 'wake'), AGENT_WAIT_MS);
      await until(() => fake.events.ends.length === 2, AGENT_WAIT_MS);
      expect(fake.events.ends.map((e) => e.result.outcome)).toEqual(['completed', 'completed']);
      await until(() => fake.host.modeState().state === 'stopped', 5_000);
      const entries = entriesOf(fake, id);
      const wakeAt = entries.findIndex((e) => e.kind === 'wake');
      expect(entries[wakeAt]).toMatchObject({ kind: 'wake', cause: 'workflow', workflow: { name: 'probe' } });
      expect(entries.slice(wakeAt + 1).find((e) => e.kind === 'assistant')).toMatchObject({ text: 'Report read.' });
      const parentLines = linesOf(fake, id);
      const wakeEnd = parentLines.findIndex(
        (l) =>
          l.xai &&
          l.update.sessionUpdate === 'turn_completed' &&
          String(l.update.prompt_id).startsWith('workflow-completed-'),
      );
      expect(wakeEnd).toBeGreaterThan(0);
      expect(
        reduce(parentLines.slice(0, wakeEnd))
          .snapshot()
          .entries.filter((e) => e.kind === 'wake'),
      ).toMatchObject([{ kind: 'wake', cause: 'workflow', workflow: { name: 'probe', status: 'complete' } }]);
      const stored = listFiles(fake.agentHome).find((f) => f.includes(id) && f.endsWith('updates.jsonl'));
      notes.storedWakePrompts =
        stored === undefined
          ? null
          : readFileSync(join(fake.agentHome, stored), 'utf8')
              .split('\n')
              .filter((l) => l.includes('hideFromScrollback'));
      notes.wakeEchoes = wire.out
        .filter(
          (m) =>
            m.method === 'session/update' && ((m.params as Json).update as Json).sessionUpdate === 'user_message_chunk',
        )
        .map((m) => m.params);
      notes.turnCompleted = xaiUpdates(wire, 'turn_completed');
      // Last: the rest of the case is evidence even while this fails.
      expect(`${child.system}\n${JSON.stringify(child.messages)}`).toContain(r.skill);
    } finally {
      await finish('step1-2-run', r, { notes });
    }
  }, 120_000);

  it('Stop run: stopped, the terminal update is cancelled, and no wake follows', async () => {
    const marker = `e2e-stop-${Math.random().toString(16).slice(2, 10)}`;
    const command = `sleep 30 # ${marker}`;
    const r = await rig((q) => {
      if (isTitle(q)) return TITLE_REPLY;
      if (isParent(q)) return parentReply(q);
      if (hasToolResult(lastMessage(q))) return { blocks: [{ type: 'text', text: REPORT }] };
      return { blocks: [{ type: 'tool_use', name: 'run_terminal_command', input: { command, description: 'wait' } }] };
    });
    const notes: Json = {};
    const sleepers = (): string[] =>
      execFileSync('ps', ['-axo', 'pid=,ppid=,command='], { encoding: 'utf8' })
        .split('\n')
        .filter((l) => l.includes(marker) && !l.includes('ps -axo'));
    try {
      const { fake, wire } = r;
      const id = await session(fake);
      const turn = start(fake, id, PROMPT);
      const startCard = await nthCard(fake, 1);
      fake.host.answer(id, startCard.id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
      const helperCard = await nthCard(fake, 2);
      fake.host.answer(id, helperCard.id, 'allow-once');
      const runId = xaiUpdates(wire, 'workflow_updated')[0]!.run_id as string;
      await until(() => sleepers().length > 0, 10_000).catch(() => {});
      notes.sleepersBeforeStop = sleepers();
      expect(await fake.host.stopRun(id, runId)).toBe(true);
      const request = wire.in.find((m) => m.method === '_x.ai/workflows/stop')!;
      await until(() => wire.out.some((m) => m.id === request.id && m.method === undefined), 10_000);
      const reply = wire.out.find((m) => m.id === request.id && m.method === undefined)!;
      notes.stopRequest = request;
      notes.stopReply = reply;
      expect(reply.result).toMatchObject({ outcome: 'stopped', runId });
      await until(() => xaiUpdates(wire, 'workflow_updated').some((u) => u.status === 'cancelled'), 10_000);
      const stoppedAt = wire.out.length;
      const parentCalls = r.engine.modelRequests.filter(isParent).length;
      await sleep(2_000);
      const after = wire.out.slice(stoppedAt);
      notes.afterStop = after;
      const parentLine = (m: Json, kinds: string[]): boolean =>
        (m.params as Json | undefined)?.sessionId === id &&
        kinds.includes(String(((m.params as Json).update as Json | undefined)?.sessionUpdate));
      expect(
        after.filter((m) => parentLine(m, ['user_message_chunk', 'agent_message_chunk', 'turn_completed'])),
      ).toEqual([]);
      expect(r.engine.modelRequests.filter(isParent)).toHaveLength(parentCalls);
      expect(fake.events.turns.some((t) => t.turn?.origin === 'wake')).toBe(false);
      expect(fake.host.runs(id)).toEqual([]);
      notes.terminalUpdate = xaiUpdates(wire, 'workflow_updated').find((u) => u.status === 'cancelled');
      notes.sleepersAfterStop = sleepers();
    } finally {
      await finish('step3-stop', r, { notes });
      const left = sleepers();
      notes.sleepersAfterClose = left;
      for (const line of left) {
        const pid = Number(line.trim().split(' ')[0]);
        if (Number.isSafeInteger(pid) && pid > 0) {
          try {
            process.kill(pid, 'SIGKILL');
          } catch {
            // gone
          }
        }
      }
      if (RECORD !== undefined)
        writeFileSync(join(RECORD, 'step3-sleepers.json'), `${JSON.stringify(notes, null, 2)}\n`);
    }
  }, 120_000);

  it('GROK_WORKFLOWS=0: the first request has no workflow listing and no workflow tool', async () => {
    const r = await rig(() => ({ blocks: [{ type: 'text', text: 'ok' }] }), { env: { GROK_WORKFLOWS: '0' } });
    const notes: Json = {};
    try {
      const { fake, engine } = r;
      const id = await session(fake);
      expect((await start(fake, id, 'CHAT-SWITCH: say ok.').done).outcome).toBe('completed');
      const first = engine.modelRequests.find((q) => !isTitle(q))!;
      notes.tools = first.tools;
      expect(JSON.stringify(first.body)).not.toContain(LISTING);
      expect(first.tools).not.toContain('workflow');
      expect(first.tools).toContain('run_terminal_command');
    } finally {
      await finish('step4-no-workflows', r, { notes });
    }
  }, 60_000);

  it('an auto compaction: its updates, the line and the marker, the request prefix and thinking, no prefire', async () => {
    const MARK = 'COMPACT';
    /** 90% of the window: over the agent's 85% auto compaction line. */
    const USED = Math.floor(WINDOW * 0.9);
    /**
     * The agent's count after a compaction is its bytes/4 estimate of the new history times (last reported usage /
     * estimate of the old history), capped at the old count. With a tiny first prompt the summary history estimates
     * larger than the old one and the count stays at the cap, so the first prompt carries ~30K tokens of text.
     */
    const PAD = ' filler'.repeat(17_000);
    /** The agent refuses a summary under 500 characters. */
    const SUMMARY = [
      'Summary of the session so far.',
      'The user sent the first probe message and asked for the single word one; the assistant answered one.',
      'No files were read or changed, no command was run, and no tool was used in this session.',
      'The user has now sent a second probe message and expects the single word two as the answer.',
      'Nothing else is pending; continue by answering the second probe message with the word two.',
      'There are no open questions, no errors, no pending tool calls and no files of interest in the workspace.',
      'The workspace folder is an empty test project; nothing in it was inspected or needs to be inspected.',
    ].join('\n');
    /** The agent's summary prompt opens with these words (seen on the real agent). */
    const summary = (q: ModelRequest): boolean =>
      !isTitle(q) && messageText(lastMessage(q)).startsWith('Your task is to produce a faithful, concise summary');
    const r = await rig(
      (q) => {
        if (isTitle(q)) return TITLE_REPLY;
        if (summary(q)) return { holdMs: 3_000, inputTokens: 900, blocks: [{ type: 'text', text: SUMMARY }] };
        const text = messageText(lastMessage(q));
        if (text.includes(`${MARK} one`)) return { inputTokens: USED, blocks: [{ type: 'text', text: 'one' }] };
        return { inputTokens: 1_200, blocks: [{ type: 'text', text: 'two' }] };
      },
      // The agent keeps 32,768 tokens of the window for the summary: a small window cuts the history it summarizes.
      { window: WINDOW },
    );
    const notes: Json = {};
    try {
      const { fake, engine, wire } = r;
      const id = await session(fake);
      expect((await start(fake, id, `${MARK} one: answer one.${PAD}`).done).outcome).toBe('completed');
      notes.contextAfterFirst = fake.events.updates
        .filter((u) => u.sessionId === id && typeof u.line.meta?.totalTokens === 'number')
        .map((u) => ({ kind: u.line.update.sessionUpdate, totalTokens: u.line.meta!.totalTokens }));
      expect(reduce(linesOf(fake, id)).context()).toBe(USED + DEFAULT_OUTPUT_TOKENS);
      const second = await start(fake, id, `${MARK} two: answer two.`).done;
      notes.secondOutcome = second.outcome;
      await until(
        () =>
          wire.out.some(
            (m) =>
              m.method === '_x.ai/session_notification' &&
              String(((m.params as Json).update as Json).sessionUpdate).startsWith('auto_compact_') &&
              ((m.params as Json).update as Json).sessionUpdate !== 'auto_compact_started',
          ),
        AGENT_WAIT_MS,
      );
      const compactLines = wire.out
        .filter((m) => m.method === '_x.ai/session_notification')
        .map((m) => (m.params as Json).update as Json)
        .filter(
          (u) => String(u.sessionUpdate).startsWith('auto_compact_') || String(u.sessionUpdate).startsWith('memory_'),
        );
      notes.compactLines = compactLines;
      expect(compactLines.map((u) => u.sessionUpdate)).toEqual(['auto_compact_started', 'auto_compact_completed']);

      const summaries = engine.modelRequests.filter(summary);
      notes.requests = engine.modelRequests.map((q) => ({
        index: q.index,
        summary: summary(q),
        messages: q.messages.length,
        thinking: q.body.thinking ?? null,
        at: q.at,
        doneAt: q.doneAt,
        closedEarly: q.closedEarly,
        header: q.headers[SESSION_HEADER] ?? null,
      }));
      expect(summaries).toHaveLength(1);
      const s = summaries[0]!;
      expect(s.closedEarly).toBe(false);
      expect(s.doneAt).not.toBeNull();
      expect(s.body.thinking).toEqual({ type: 'enabled', budget_tokens: 4096 });
      const previous = engine.modelRequests.filter((q) => q.index < s.index && !summary(q) && !isTitle(q)).at(-1)!;
      expect(stripCacheControl(s.messages.slice(0, previous.messages.length))).toEqual(
        stripCacheControl(previous.messages),
      );
      expect(stripCacheControl(s.body.system)).toEqual(stripCacheControl(previous.body.system));
      expect(stripCacheControl(s.body.tools)).toEqual(stripCacheControl(previous.body.tools));
      expect(s.tools.length).toBeGreaterThan(0);
      notes.prefix = { previous: previous.messages.length, summary: s.messages.length };

      notes.contextAll = fake.events.updates
        .filter((u) => u.sessionId === id && typeof u.line.meta?.totalTokens === 'number')
        .map((u) => ({ kind: u.line.update.sessionUpdate, totalTokens: u.line.meta!.totalTokens }));
      notes.compactions = fake.events.compactions;
      const started = fake.events.compactions.findIndex((c) => c.compaction !== null);
      expect(started).toBeGreaterThanOrEqual(0);
      expect(fake.events.compactions.slice(started + 1).some((c) => c.compaction === null)).toBe(true);
      const tl = fake.events.timeline;
      expect(tl.indexOf('compaction start')).toBeLessThan(tl.indexOf('auto_compact_completed'));
      const [startedLine, completedLine] = compactLines as [Json, Json];
      const tokensBefore = completedLine.tokens_before as number;
      const tokensAfter = completedLine.tokens_after as number;
      expect(tokensBefore).toBe(startedLine.tokens_used);
      const markers = entriesOf(fake, id).filter((e) => e.kind === 'compaction');
      expect(markers).toEqual([{ kind: 'compaction', ts: markers[0]?.ts, outcome: 'done', tokensBefore, tokensAfter }]);
      // The agent reports the new history's count: below the count before, and the conversation's context drops to it.
      expect(tokensAfter).toBeLessThan(tokensBefore);
      expect(compactionText(markers[0]!)).toBe(
        fill(CHAT.compactedTokens, { before: formatCount(tokensBefore), after: formatCount(tokensAfter) }),
      );
      const parentLines = linesOf(fake, id);
      const completedAt = parentLines.findIndex((l) => l.xai && l.update.sessionUpdate === 'auto_compact_completed');
      const contextBefore = reduce(parentLines.slice(0, completedAt)).context();
      const contextAfter = reduce(parentLines.slice(0, completedAt + 1)).context();
      notes.contextAtCompaction = { before: contextBefore, after: contextAfter };
      expect(contextBefore).not.toBeNull();
      expect(contextAfter).not.toBeNull();
      expect(contextAfter!).toBeLessThan(contextBefore!);
      expect(contextAfter!).toBeLessThanOrEqual(tokensAfter);
    } finally {
      await finish('step7-compaction', r, { notes });
    }
  }, 120_000);
});
