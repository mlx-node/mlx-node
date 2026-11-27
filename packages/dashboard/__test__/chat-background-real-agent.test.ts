/**
 * Background work on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: the host's
 * tracker reads the lines v0.3.4 really sends for a background command and a background subagent, holds the process
 * through a settings change and the idle stop, reads the agent's own `_x.ai/subagent/list_running` reply, and takes a
 * helper's `subagent_progress` ticks into the live facts, stops a command through `_x.ai/task/kill`, and takes
 * a monitor's `_x.ai/monitor_event` lines into the live facts, naming the wake that drains them, and deletes a
 * paused schedule through a load and `_x.ai/scheduler/delete`, and reads a fire's run as its `firing`. Skipped
 * without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, rfc3339Ms, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { readRunningSubagents, readSubagentProgress } from '../src/chat/acp/messages.js';
import { spawnAgent, type SpawnAgent } from '../src/chat/acp/process.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { wakeReportOf, wakeReportsOf } from '../ui/src/lib/activity.js';
import { fill } from '../ui/src/lib/chat-strings.js';
import { WAKE_ENDED, WAKE_REPORTED, wakeTitle } from '../ui/src/lib/sessions.js';
import { fakeHost, noProviders, until, type FakeHost, type FakeHostOptions, type Json } from './helpers/chat-fake.js';
import {
  hasToolResult,
  lastMessage,
  messageText,
  startScriptedEngine,
  type ModelRequest,
  type ModelScript,
} from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;

/** The agent's stdout lines and the host's stdin writes, parsed, in order. */
interface Wire {
  out: Json[];
  in: Json[];
}

function teeSpawn(wire: Wire): SpawnAgent {
  return (command) => {
    const child = spawnAgent(command);
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

const isTitle = (q: ModelRequest): boolean => q.tools.length === 1 && q.tools[0] === 'session_title';
const TITLE = { blocks: [{ type: 'tool_use' as const, name: 'session_title', input: { session_title: 'Probe' } }] };

async function rig(
  reply: ModelScript,
  host: FakeHostOptions = {},
): Promise<{ fake: FakeHost; wire: Wire; close: () => Promise<void> }> {
  const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-bg-')));
  const agentTmp = join(outer, 'tmp');
  mkdirSync(agentTmp);
  const wire: Wire = { out: [], in: [] };
  const fake = await fakeHost('chat-harness', {
    mode: 'work',
    providers: noProviders(),
    launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
    startEngine: () =>
      startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => (isTitle(q) ? TITLE : reply(q))),
    env: { TMPDIR: `${agentTmp}/` },
    spawn: teeSpawn(wire),
    ...host,
  });
  return {
    fake,
    wire,
    close: async () => {
      await fake.close();
      rmSync(outer, { recursive: true, force: true });
    },
  };
}

/** Runs one turn and allows every request it makes, until it ends. */
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

/** The xAI updates of the session the host passed on, in order. */
function xaiLines(fake: FakeHost, sessionId: string, kind: string): Json[] {
  return fake.events.updates
    .filter((u) => u.sessionId === sessionId && u.line.xai && u.line.update.sessionUpdate === kind)
    .map((u) => u.line.update as Json);
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// The agent confirms a new schedule before it writes it (its state is saved on a 500 ms tick), and a stop before that
// tick loses the schedule.
function scheduleOnDisk(root: string, scheduleId: string): boolean {
  return readdirSync(root, { recursive: true, encoding: 'utf8' })
    .filter((f) => f.endsWith('resources_state.json'))
    .some((f) => readFileSync(join(root, f), 'utf8').includes(scheduleId));
}

describe.skipIf(BIN === undefined)('Background work on the real agent (H-a)', () => {
  it('a background command: tracked from the real lines, it holds a settings change until it ends', async () => {
    const PROMPT = 'BGPROBE: start the waiter.';
    const r = await rig((q) => {
      const last = lastMessage(q);
      if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'Started.' }] };
      if (messageText(last).includes(PROMPT) && q.tools.includes('run_terminal_command')) {
        return {
          blocks: [
            {
              type: 'tool_use',
              name: 'run_terminal_command',
              input: { command: 'sleep 3', description: 'wait', block_until_ms: 0 },
            },
          ],
        };
      }
      return { blocks: [{ type: 'text', text: 'The waiter ended.' }] };
    });
    try {
      const { fake } = r;
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      expect(await runAllowing(fake, id, PROMPT)).toBe('completed');
      const started = xaiLines(fake, id, 'task_backgrounded');
      expect(started).toHaveLength(1);
      const taskId = started[0]!.task_id as string;
      expect(fake.host.isWorking(id)).toBe(true);
      expect(fake.host.engineBusy(id)).toBe(false);
      expect(fake.host.modeState()).toMatchObject({ state: 'ready', tasks: 1, background: 0 });

      fake.host.markStale();
      await sleep(300);
      expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: true, tasks: 1 });
      await until(() => fake.host.modeState().state === 'stopped', AGENT_WAIT_MS);
      const ended = xaiLines(fake, id, 'task_completed');
      expect(ended.map((u) => (u.task_snapshot as Json).task_id)).toEqual([taskId]);
      // The completion announces a wake, and the process stays until that wake ended.
      expect(ended[0]!.will_wake).toBe(true);
      expect(fake.events.turns.some((t) => t.turn?.origin === 'wake')).toBe(true);
      expect(fake.events.ends.map((e) => e.result.outcome)).toEqual(['completed', 'completed']);
      // H-b: the host's wake line names the id the agent gives that wake, so the row names the command at once.
      const wakeId = `task-completed-${taskId}`;
      const hostLines = fake.events.updates.filter(
        (u) => u.sessionId === id && u.line.update.sessionUpdate === 'user_message_chunk',
      );
      expect(hostLines.map((u) => u.line.update._meta)).toEqual([
        { hideFromScrollback: true, mlxWakePromptId: wakeId },
      ]);
      expect(xaiLines(fake, id, 'turn_completed').map((u) => u.prompt_id)).toContain(wakeId);
      const reducer = new TranscriptReducer();
      reducer.userPrompt(PROMPT);
      for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
      const wakes = (entries: readonly TranscriptEntry[]) =>
        entries.flatMap((e) => (e.kind === 'wake' ? [{ cause: e.cause, task: e.task }] : []));
      const live = wakes(reducer.snapshot().entries);
      expect(live).toEqual([{ cause: 'task', task: { id: taskId, kind: 'command', label: 'wait' } }]);
      const files = readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
        .filter((f) => f.endsWith('updates.jsonl'))
        .map((f) => readFileSync(join(fake.root, f), 'utf8'))
        .filter((text) => text.includes(wakeId));
      expect(files).toHaveLength(1);
      expect(wakes(parseUpdates(files[0]!).transcript)).toEqual(live);
    } finally {
      await r.close();
    }
  }, 90_000);

  it('a background subagent: tracked from the real lines, list_running names it, the idle stop waits for its end', async () => {
    const PROMPT = 'BGPROBE: start the helper.';
    const CHILD = 'CHILDPROBE: say done.';
    const r = await rig(
      (q) => {
        const all = JSON.stringify(q.messages);
        const last = lastMessage(q);
        if (!all.includes(PROMPT)) return { holdMs: 4_000, blocks: [{ type: 'text', text: 'Done.' }] };
        if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'Started.' }] };
        if (messageText(last).includes(PROMPT) && q.tools.includes('spawn_subagent')) {
          return {
            blocks: [
              {
                type: 'tool_use',
                name: 'spawn_subagent',
                input: {
                  description: 'probe',
                  prompt: CHILD,
                  subagent_type: 'general-purpose',
                  run_in_background: true,
                },
              },
            ],
          };
        }
        return { blocks: [{ type: 'text', text: 'The helper is done.' }] };
      },
      { timings: { idleStopMs: 300 } },
    );
    try {
      const { fake, wire } = r;
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      expect(await runAllowing(fake, id, PROMPT)).toBe('completed');
      const spawned = xaiLines(fake, id, 'subagent_spawned');
      expect(spawned).toHaveLength(1);
      const child = spawned[0]!.child_session_id as string;
      expect(fake.host.engineBusy(id)).toBe(true);

      const asked = (): Json[] => wire.in.filter((m) => m.method === '_x.ai/subagent/list_running');
      await until(() => asked().length > 0 && wire.out.some((m) => m.id === asked()[0]!.id), AGENT_WAIT_MS);
      expect(asked()[0]!.params).toEqual({ sessionId: id });
      const reply = wire.out.find((m) => m.id === asked()[0]!.id && m.method === undefined)!;
      // What the client's request resolves with: the reply's `result`.
      const listed = readRunningSubagents(reply.result);
      expect(listed).not.toBeNull();
      expect(listed!.has(child)).toBe(true);
      expect(fake.host.modeState().state).toBe('ready');
      expect(fake.host.engineBusy(id)).toBe(true);

      await until(() => fake.host.modeState().state === 'stopped', AGENT_WAIT_MS);
      expect(xaiLines(fake, id, 'subagent_finished').map((u) => u.child_session_id)).toEqual([child]);
      // H-b: the helper's end announces a wake; the host's wake line names the id the agent gives it.
      const finished = xaiLines(fake, id, 'subagent_finished')[0]!;
      expect(finished.will_wake).toBe(true);
      const wakeId = `subagent-completed-${finished.subagent_id as string}`;
      const hostLines = fake.events.updates.filter(
        (u) => u.sessionId === id && u.line.update.sessionUpdate === 'user_message_chunk',
      );
      expect(hostLines.map((u) => u.line.update._meta)).toEqual([
        { hideFromScrollback: true, mlxWakePromptId: wakeId },
      ]);
      expect(xaiLines(fake, id, 'turn_completed').map((u) => u.prompt_id)).toContain(wakeId);
      const reducer = new TranscriptReducer();
      reducer.userPrompt(PROMPT);
      for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
      expect(reducer.snapshot().entries.filter((e) => e.kind === 'wake')).toMatchObject([
        { cause: 'subagent', subagent: { childId: child, description: 'probe' } },
      ]);
    } finally {
      await r.close();
    }
  }, 90_000);

  it('a background subagent ticks: its real subagent_progress lines reach the live facts and never the reducer (#10)', async () => {
    const PROMPT = 'BGPROBE: start the ticking helper.';
    const CHILD = 'CHILDPROBE: think, then say done.';
    const r = await rig(
      (q) => {
        const all = JSON.stringify(q.messages);
        const last = lastMessage(q);
        if (!all.includes(PROMPT)) return { holdMs: 7_000, blocks: [{ type: 'text', text: 'Done.' }] };
        if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'Started.' }] };
        if (messageText(last).includes(PROMPT) && q.tools.includes('spawn_subagent')) {
          return {
            blocks: [
              {
                type: 'tool_use',
                name: 'spawn_subagent',
                input: {
                  description: 'probe',
                  prompt: CHILD,
                  subagent_type: 'general-purpose',
                  run_in_background: true,
                },
              },
            ],
          };
        }
        return { blocks: [{ type: 'text', text: 'The helper is done.' }] };
      },
      { timings: { activityEveryMs: 100 } },
    );
    try {
      const { fake, wire } = r;
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      expect(await runAllowing(fake, id, PROMPT)).toBe('completed');
      const child = xaiLines(fake, id, 'subagent_spawned')[0]!.child_session_id as string;
      await until(
        () => fake.events.activity.some((e) => e.live.helpers.some((h) => h.childId === child)),
        AGENT_WAIT_MS,
      );
      await until(() => xaiLines(fake, id, 'subagent_finished').length === 1, AGENT_WAIT_MS);
      await until(() => fake.events.activity.at(-1)?.live.helpers.length === 0, AGENT_WAIT_MS);
      const ticks = wire.out
        .filter((m) => m.method === '_x.ai/session_notification' && (m.params as Json).update !== undefined)
        .map((m) => m.params as Json)
        .filter((p) => (p.update as Json).sessionUpdate === 'subagent_progress');
      expect(ticks.length).toBeGreaterThan(0);
      for (const tick of ticks) {
        // The parent session names the line, the agent sends no `_meta`, and every count reads.
        expect(tick.sessionId).toBe(id);
        expect(tick._meta).toBeUndefined();
        expect(readSubagentProgress(tick.update as Json)).toMatchObject({ childId: child });
      }
      expect(fake.events.activity.every((e) => e.sessionId === id)).toBe(true);
      expect(fake.events.updates.some((u) => u.line.update.sessionUpdate === 'subagent_progress')).toBe(false);
      const finished = xaiLines(fake, id, 'subagent_finished')[0]!;
      const reducer = new TranscriptReducer();
      reducer.userPrompt(PROMPT);
      for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
      expect(reducer.snapshot().entries.find((e) => e.kind === 'subagent')).toMatchObject({
        childId: child,
        turns: finished.turns,
        toolCalls: finished.tool_calls,
      });
      expect(typeof finished.turns).toBe('number');
      expect(typeof finished.tool_calls).toBe('number');
      // The agent's own log of the parent session gives the same row, and holds no tick.
      const stored = (): string[] =>
        readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(fake.root, f), 'utf8'))
          .filter((text) => text.includes('"subagent_finished"') && text.includes(child));
      await until(() => stored().length === 1, AGENT_WAIT_MS);
      expect(stored()[0]).not.toContain('subagent_progress');
      const row = (entries: readonly TranscriptEntry[]) => {
        const found = entries.find((e) => e.kind === 'subagent');
        if (found?.kind !== 'subagent') throw new Error('no subagent row');
        const { childId, status, turns, toolCalls, durationMs, tokens } = found;
        return { childId, status, turns, toolCalls, durationMs, tokens };
      };
      expect(row(parseUpdates(stored()[0]!).transcript)).toEqual(row(reducer.snapshot().entries));
    } finally {
      await r.close();
    }
  }, 90_000);
});

describe.skipIf(BIN === undefined)('Stop a background command on the real agent (T1)', () => {
  it('Stop kills it, the model is told in a wake, the entry reads killed live and on disk; a teardown stop wakes nothing', async () => {
    const FIRST = 'BGSTOP: start the first waiter.';
    const SECOND = 'BGSTOP: start the second waiter.';
    const r = await rig((q) => {
      const last = lastMessage(q);
      if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'Started.' }] };
      const text = messageText(last);
      if ((text.includes(FIRST) || text.includes(SECOND)) && q.tools.includes('run_terminal_command')) {
        return {
          blocks: [
            {
              type: 'tool_use',
              name: 'run_terminal_command',
              input: { command: 'sleep 60', description: text.includes(FIRST) ? 'first' : 'second', block_until_ms: 0 },
            },
          ],
        };
      }
      return { blocks: [{ type: 'text', text: 'Noted.' }] };
    });
    try {
      const { fake, wire } = r;
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      expect(await runAllowing(fake, id, FIRST)).toBe('completed');
      const first = xaiLines(fake, id, 'task_backgrounded')[0]!.task_id as string;
      expect(fake.host.liveActivity(id).tasks).toEqual([
        { id: first, kind: 'command', stopping: false, events: 0, recent: [] },
      ]);

      expect(await fake.host.stopTask(id, first)).toBe('stopped');
      const kill = wire.in.find((m) => m.method === '_x.ai/task/kill')!;
      expect(kill.params).toEqual({ sessionId: id, taskId: first, source: 'clientUi' });
      // What the client's request resolves with: the reply's `result`, the agent's `ExtMethodResult`.
      expect(wire.out.find((m) => m.id === kill.id && m.method === undefined)?.result).toEqual({
        result: { taskId: first, outcome: 'killed' },
      });
      await until(() => xaiLines(fake, id, 'task_completed').length === 1, AGENT_WAIT_MS);
      const end = xaiLines(fake, id, 'task_completed')[0]!;
      expect(end.will_wake).toBe(true);
      expect(end.task_snapshot).toMatchObject({
        task_id: first,
        explicitly_killed: true,
        kill_result_delivered: false,
        exit_code: null,
        signal: 'killed',
      });
      await until(() => fake.host.liveActivity(id).tasks.length === 0, AGENT_WAIT_MS);
      // The model is told: a wake turn runs and ends.
      await until(() => fake.events.ends.length === 2, AGENT_WAIT_MS);
      expect(fake.events.ends.map((e) => e.result.outcome)).toEqual(['completed', 'completed']);

      expect(await runAllowing(fake, id, SECOND)).toBe('completed');
      const second = xaiLines(fake, id, 'task_backgrounded')[1]!.task_id as string;
      expect(await fake.host.stopTask(id, second, 'teardown')).toBe('stopped');
      await until(() => xaiLines(fake, id, 'task_completed').length === 2, AGENT_WAIT_MS);
      expect(xaiLines(fake, id, 'task_completed')[1]).toMatchObject({
        will_wake: false,
        task_snapshot: { task_id: second, explicitly_killed: true, kill_result_delivered: true },
      });
      await sleep(2_000);
      expect(fake.events.ends).toHaveLength(3);
      expect(fake.host.isWorking(id)).toBe(false);

      // A task that ended is not live: the host sends no kill for it.
      expect(await fake.host.stopTask(id, first)).toBe('not-live');
      expect(wire.in.filter((m) => m.method === '_x.ai/task/kill')).toHaveLength(2);

      const reducer = new TranscriptReducer();
      reducer.userPrompt(FIRST);
      for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
      const tasks = (entries: readonly TranscriptEntry[]) =>
        entries.flatMap((e) =>
          e.kind === 'tool' && e.task !== undefined
            ? [{ id: e.task.id, killed: e.task.killed, ended: e.task.endTs !== undefined }]
            : [],
        );
      const live = tasks(reducer.snapshot().entries);
      expect(live).toEqual([
        { id: first, killed: true, ended: true },
        { id: second, killed: true, ended: true },
      ]);
      const stored = (): string[] =>
        readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(fake.root, f), 'utf8'))
          .filter((text) => text.includes(second) && text.includes('"task_completed"'));
      await until(() => stored().length === 1, AGENT_WAIT_MS);
      expect(tasks(parseUpdates(stored()[0]!).transcript)).toEqual(live);
    } finally {
      await r.close();
    }
  }, 120_000);
});

describe.skipIf(BIN === undefined)('Monitor events on the real agent (T2)', () => {
  it('a monitor’s line reaches the live facts and never the reducer; the drain wake is named by its report', async () => {
    const PROMPT = 'MONPROBE: watch the probe.';
    const r = await rig((q) => {
      const last = lastMessage(q);
      if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'Watching.' }] };
      if (messageText(last).includes(PROMPT) && q.tools.includes('monitor')) {
        return {
          blocks: [
            {
              type: 'tool_use',
              name: 'monitor',
              input: { command: 'sleep 4; echo "probe event one"; sleep 120', description: 'Watch the probe' },
            },
          ],
        };
      }
      return { blocks: [{ type: 'text', text: 'The probe reported.' }] };
    });
    try {
      const { fake, wire } = r;
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      expect(await runAllowing(fake, id, PROMPT)).toBe('completed');
      const started = xaiLines(fake, id, 'task_backgrounded');
      expect(started).toHaveLength(1);
      const taskId = started[0]!.task_id as string;
      expect(fake.host.liveActivity(id).tasks).toMatchObject([{ id: taskId, kind: 'monitor', events: 0 }]);

      await until(() => wire.out.some((m) => m.method === '_x.ai/monitor_event'), AGENT_WAIT_MS);
      const sent = wire.out.find((m) => m.method === '_x.ai/monitor_event')!;
      expect(sent.params).toEqual({
        sessionId: id,
        update: {
          sessionUpdate: 'monitor_event',
          task_id: taskId,
          description: 'Watch the probe',
          event_text: expect.stringContaining('probe event one'),
        },
      });
      // The agent drains the event, the session idle, as a turn of its own.
      await until(() => fake.events.ends.length === 2, AGENT_WAIT_MS);
      const [task] = fake.host.liveActivity(id).tasks;
      expect(task).toMatchObject({ id: taskId, events: 1 });
      expect(task!.recent[0]!.text).toContain('probe event one');
      const reports = fake.host.liveActivity(id).reports;
      expect(reports).toEqual([
        { wakeTs: expect.any(Number), monitors: [{ id: taskId, description: 'Watch the probe' }] },
      ]);
      const hostLine = fake.events.updates.find(
        (u) => u.sessionId === id && u.line.update.sessionUpdate === 'user_message_chunk',
      )!;
      expect(hostLine.line.update._meta).toEqual({ hideFromScrollback: true, mlxWakePromptId: 'notifications-' });
      expect(hostLine.line.meta).toEqual({ agentTimestampMs: reports[0]!.wakeTs });
      const wakeIds = xaiLines(fake, id, 'turn_completed').map((u) => String(u.prompt_id));
      expect(wakeIds).toHaveLength(2);
      expect(wakeIds[1]!.startsWith('notifications-')).toBe(true);
      expect(fake.events.updates.filter((u) => u.line.update.sessionUpdate === 'monitor_event')).toEqual([]);
      // The agent sent the monitor's output as a tick of its call too (no `_meta`, never stored); the host took it.
      const call = started[0]!.tool_call_id as string;
      const tick = wire.out.find(
        (m) =>
          m.method === 'session/update' &&
          ((m.params as Json).update as Json).toolCallId === call &&
          ((m.params as Json).update as Json).status === 'in_progress',
      );
      expect(tick).toBeDefined();
      expect(tick!.params).not.toHaveProperty('_meta');
      expect(
        fake.events.updates.filter((u) => u.line.update.toolCallId === call && u.line.update.status === 'in_progress'),
      ).toEqual([]);

      const reducer = new TranscriptReducer();
      reducer.userPrompt(PROMPT);
      for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
      const entries = reducer.snapshot().entries;
      const wake = entries.find((e) => e.kind === 'wake');
      if (wake?.kind !== 'wake') throw new Error('no wake row');
      expect(wake).toMatchObject({ cause: 'notifications', ts: reports[0]!.wakeTs });
      expect(wake.tasks).toBeUndefined();
      const monitorRow = (list: readonly TranscriptEntry[]) => {
        const row = list.find((e) => e.kind === 'tool' && e.task?.id === taskId);
        return row?.kind === 'tool' ? { status: row.status, task: row.task } : null;
      };
      expect(monitorRow(entries)).toEqual({
        status: 'completed',
        task: { id: taskId, kind: 'monitor', label: 'Watch the probe' },
      });
      const named = wakeReportOf(wake, wakeReportsOf(fake.host.liveActivity(id), entries));
      expect(wakeTitle(wake, named)).toBe(
        fill(WAKE_ENDED.named, { what: WAKE_REPORTED.one, labels: 'Watch the probe' }),
      );

      const files = readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
        .filter((f) => f.endsWith('updates.jsonl'))
        .map((f) => readFileSync(join(fake.root, f), 'utf8'))
        .filter((text) => text.includes(wakeIds[1]!));
      expect(files).toHaveLength(1);
      expect(files[0]).not.toContain('monitor_event');
      const disk = parseUpdates(files[0]!).transcript;
      const stored = disk.find((e) => e.kind === 'wake');
      if (stored?.kind !== 'wake') throw new Error('no stored wake row');
      expect([stored.cause, stored.tasks]).toEqual([wake.cause, wake.tasks]);
      expect(monitorRow(disk)).toEqual(monitorRow(entries));
      expect(await fake.host.stopTask(id, taskId, 'teardown')).toBe('stopped');
    } finally {
      await r.close();
    }
  }, 120_000);
});

describe.skipIf(BIN === undefined)('Delete a schedule on the real agent (T3)', () => {
  it('a paused schedule: the Delete loads the session (the agent announces it again), then deletes it, live and on disk', async () => {
    const PROMPT = 'SCHEDPROBE: check the deploy every hour.';
    const r = await rig((q) => {
      const last = lastMessage(q);
      if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'Scheduled.' }] };
      if (messageText(last).includes(PROMPT) && q.tools.includes('scheduler_create')) {
        return {
          blocks: [
            { type: 'tool_use', name: 'scheduler_create', input: { interval: '1h', prompt: 'Check the deploy' } },
          ],
        };
      }
      return { blocks: [{ type: 'text', text: 'ok' }] };
    });
    try {
      const { fake, wire } = r;
      const load = { modelId: MODEL, cwd: fake.project };
      const { sessionId: id } = await fake.host.createSession(load);
      expect(await runAllowing(fake, id, PROMPT)).toBe('completed');
      const created = xaiLines(fake, id, 'scheduled_task_created');
      expect(created).toHaveLength(1);
      const scheduleId = created[0]!.task_id as string;
      await until(() => fake.host.liveActivity(id).schedules.length === 1, AGENT_WAIT_MS);
      // The agent's own answer for an id it does not hold: `{result: {taskId, deleted: false}}`.
      expect(await fake.host.deleteSchedule(id, 'no-such-schedule', load)).toBe('gone');
      await until(() => scheduleOnDisk(fake.root, scheduleId), AGENT_WAIT_MS);

      await fake.host.restart();
      expect(fake.host.isHot(id)).toBe(false);
      expect(fake.host.liveActivity(id).schedules).toEqual([]);
      const end = fake.host.deleteSchedule(id, scheduleId, load);
      expect(fake.host.liveActivity(id).schedules).toEqual([{ id: scheduleId, deleting: 'opening', firing: null }]);
      expect(await end).toBe('deleted');
      expect(fake.host.isHot(id)).toBe(true);
      const calls = wire.in.filter((m) => m.method === 'session/load' || m.method === '_x.ai/scheduler/delete');
      expect(calls.map((m) => m.method)).toEqual(['_x.ai/scheduler/delete', 'session/load', '_x.ai/scheduler/delete']);
      expect(calls.at(-1)!.params).toEqual({ sessionId: id, taskId: scheduleId });
      // Request ids start again in each process: the replies are found by the schedule they name.
      const replyFor = (taskId: string): unknown =>
        wire.out.find(
          (m) =>
            m.method === undefined && ((m.result as Json | undefined)?.result as Json | undefined)?.taskId === taskId,
        )?.result;
      expect(['no-such-schedule', scheduleId].map(replyFor)).toEqual([
        { result: { taskId: 'no-such-schedule', deleted: false } },
        { result: { taskId: scheduleId, deleted: true } },
      ]);
      // On load the agent announced the stored schedule again, before the delete took it.
      expect(xaiLines(fake, id, 'scheduled_task_created').map((u) => u.task_id)).toEqual([scheduleId, scheduleId]);
      await until(
        () => xaiLines(fake, id, 'scheduled_task_deleted').some((u) => u.reason === 'deleted'),
        AGENT_WAIT_MS,
      );
      expect(fake.host.liveActivity(id).schedules).toEqual([]);
      const ended = (entries: readonly TranscriptEntry[]) =>
        entries.flatMap((e) => (e.kind === 'tool' && e.schedule !== undefined ? [e.schedule.endTs !== undefined] : []));
      const reducer = new TranscriptReducer();
      reducer.userPrompt(PROMPT);
      for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
      expect(ended(reducer.snapshot().entries)).toEqual([true]);
      await until(() => {
        const files = readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(fake.root, f), 'utf8'))
          .filter((text) => text.includes(scheduleId));
        return files.length === 1 && ended(parseUpdates(files[0]!).transcript).join() === 'true';
      }, AGENT_WAIT_MS);
    } finally {
      await r.close();
    }
  }, 120_000);
});

describe.skipIf(BIN === undefined)('A cold load past its soft deadline on the real agent (A4)', () => {
  // Mutation: load the Delete with the turn's hard path → it waits for the load and deletes, with no `load-slow`.
  it('a paused schedule’s Delete ends load-slow; another conversation’s command lives on; the late load lands, then it deletes', async () => {
    const SCHED = 'SCHEDPROBE: check the deploy every hour.';
    const BG = 'BGPROBE: start the waiter.';
    const r = await rig(
      (q) => {
        const last = lastMessage(q);
        if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'Done.' }] };
        const text = messageText(last);
        if (text.includes(SCHED) && q.tools.includes('scheduler_create')) {
          return {
            blocks: [
              { type: 'tool_use', name: 'scheduler_create', input: { interval: '1h', prompt: 'Check the deploy' } },
            ],
          };
        }
        if (text.includes(BG) && q.tools.includes('run_terminal_command')) {
          return {
            blocks: [
              {
                type: 'tool_use',
                name: 'run_terminal_command',
                input: { command: 'sleep 60', description: 'wait', block_until_ms: 0 },
              },
            ],
          };
        }
        return { blocks: [{ type: 'text', text: 'ok' }] };
      },
      { timings: { loadSoftMs: 1 } },
    );
    try {
      const { fake, wire } = r;
      const load = { modelId: MODEL, cwd: fake.project };
      const { sessionId: paused } = await fake.host.createSession(load);
      expect(await runAllowing(fake, paused, SCHED)).toBe('completed');
      const scheduleId = xaiLines(fake, paused, 'scheduled_task_created')[0]!.task_id as string;
      await until(() => scheduleOnDisk(fake.root, scheduleId), AGENT_WAIT_MS);
      await fake.host.restart();
      const { sessionId: live } = await fake.host.createSession(load);
      expect(await runAllowing(fake, live, BG)).toBe('completed');
      await until(() => fake.host.liveActivity(live).tasks.length === 1, AGENT_WAIT_MS);
      const starts = wire.in.filter((m) => m.method === 'initialize').length;

      await expect(fake.host.deleteSchedule(paused, scheduleId, load)).rejects.toMatchObject({ code: 'load-slow' });
      expect(fake.host.modeState().state).toBe('ready');
      expect(fake.host.liveActivity(live).tasks).toHaveLength(1);
      await until(() => fake.host.isHot(paused), AGENT_WAIT_MS);
      const sent = wire.in.filter((m) => m.method === 'session/load' || m.method === '_x.ai/scheduler/delete');
      expect(sent.map((m) => m.method)).toEqual(['session/load']);
      expect(wire.in.filter((m) => m.method === 'initialize')).toHaveLength(starts);
      expect(fake.host.liveActivity(live).tasks).toHaveLength(1);

      expect(await fake.host.deleteSchedule(paused, scheduleId, load)).toBe('deleted');
      expect(wire.in.filter((m) => m.method === 'session/load')).toHaveLength(1);
      expect(fake.host.liveActivity(live).tasks).toHaveLength(1);
      expect(fake.host.modeState().state).toBe('ready');
    } finally {
      await r.close();
    }
  }, 120_000);
});

describe.skipIf(BIN === undefined)('A schedule fires on the real agent (T3)', () => {
  it('scheduled_task_fired names its run by subagent_id: firing holds that run until its subagent_finished', async () => {
    const PROMPT = 'FIREPROBE: schedule the check and run it now.';
    const RUN = 'FIRERUN: check the deploy.';
    const r = await rig(
      (q) => {
        const all = JSON.stringify(q.messages);
        const last = lastMessage(q);
        if (!all.includes(PROMPT)) return { holdMs: 1_500, blocks: [{ type: 'text', text: 'Checked.' }] };
        if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'Scheduled.' }] };
        if (messageText(last).includes(PROMPT) && q.tools.includes('scheduler_create')) {
          return {
            blocks: [
              {
                type: 'tool_use',
                name: 'scheduler_create',
                input: { interval: '1h', prompt: RUN, fire_immediately: true },
              },
            ],
          };
        }
        return { blocks: [{ type: 'text', text: 'ok' }] };
      },
      { timings: { activityEveryMs: 50 } },
    );
    try {
      const { fake, wire } = r;
      const load = { modelId: MODEL, cwd: fake.project };
      const { sessionId: id } = await fake.host.createSession(load);
      expect(await runAllowing(fake, id, PROMPT)).toBe('completed');
      await until(() => fake.host.liveActivity(id).schedules[0]?.firing != null, AGENT_WAIT_MS);
      const run = fake.host.liveActivity(id).schedules[0]!.firing!;
      const fired = wire.out.find(
        (m) => m.method === '_x.ai/scheduled_task_fired' && ((m.params as Json).update as Json).subagent_id === run,
      );
      expect(fired).toBeDefined();
      // The fire's next time rides the live facts (the fire is never stored, so no entry takes it).
      const nextTs = rfc3339Ms(((fired!.params as Json).update as Json).next_fire_at);
      expect(nextTs).not.toBeNull();
      expect(fake.host.liveActivity(id).schedules[0]!.nextTs).toBe(nextTs);
      const spawned = xaiLines(fake, id, 'subagent_spawned');
      expect(spawned.map((u) => u.child_session_id)).toContain(run);
      await until(() => fake.host.liveActivity(id).schedules[0]?.firing === null, AGENT_WAIT_MS);
      const finished = xaiLines(fake, id, 'subagent_finished');
      expect(finished.map((u) => u.child_session_id)).toContain(run);
    } finally {
      await r.close();
    }
  }, 120_000);
});

describe.skipIf(BIN === undefined)('Stop or message a subagent on the real agent', () => {
  /** A background helper whose own model call holds `holdMs`; `childCalls` gets each of its calls' texts. */
  function helperRig(PROMPT: string, CHILD: string, holdMs: number, childCalls: string[]) {
    return rig((q) => {
      const all = JSON.stringify(q.messages);
      const last = lastMessage(q);
      if (!all.includes(PROMPT)) {
        childCalls.push(all);
        // The first call holds; the next one (after a message) ends the helper.
        return childCalls.length === 1
          ? { holdMs, blocks: [{ type: 'text', text: 'Waiting.' }] }
          : { blocks: [{ type: 'text', text: 'Done.' }] };
      }
      if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'Started.' }] };
      if (messageText(last).includes(PROMPT) && q.tools.includes('spawn_subagent')) {
        return {
          blocks: [
            {
              type: 'tool_use',
              name: 'spawn_subagent',
              input: { description: 'probe', prompt: CHILD, subagent_type: 'general-purpose', run_in_background: true },
            },
          ],
        };
      }
      return { blocks: [{ type: 'text', text: 'ok' }] };
    });
  }

  it('Stop: `_x.ai/subagent/cancel {subagentId}` answers cancelled in the result envelope; its end line says cancelled', async () => {
    const PROMPT = 'SUBSTOP: start the helper.';
    const childCalls: string[] = [];
    const r = await helperRig(PROMPT, 'CHILDPROBE: wait.', 25_000, childCalls);
    try {
      const { fake, wire } = r;
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      expect(await runAllowing(fake, id, PROMPT)).toBe('completed');
      const child = xaiLines(fake, id, 'subagent_spawned')[0]!.child_session_id as string;
      await until(() => childCalls.length === 1, AGENT_WAIT_MS);
      expect(fake.host.liveActivity(id).agents).toEqual([
        { childId: child, run: null, attempts: [child], stopping: false, messages: 'on' },
      ]);
      expect(await fake.host.stopSubagent(id, child)).toBe('stopped');
      const asked = wire.in.find((m) => m.method === '_x.ai/subagent/cancel')!;
      expect(asked.params).toEqual({ subagentId: child });
      const answer = wire.out.find((m) => m.id === asked.id && m.method === undefined)!;
      expect(answer.result).toEqual({
        result: { subagentId: child, cancelled: true, outcome: { kind: 'cancelled' } },
      });
      await until(() => xaiLines(fake, id, 'subagent_finished').length === 1, AGENT_WAIT_MS);
      expect(xaiLines(fake, id, 'subagent_finished')[0]).toMatchObject({
        child_session_id: child,
        status: 'cancelled',
      });
      await until(() => fake.host.liveActivity(id).agents.length === 0, AGENT_WAIT_MS);
      expect(await fake.host.stopSubagent(id, child)).toBe('not-live');
      expect(wire.in.filter((m) => m.method === '_x.ai/subagent/cancel')).toHaveLength(1);
    } finally {
      await r.close();
    }
  }, 90_000);

  it('Message: the live start names `agentAddress` (never stored); a steer by it is accepted and the helper reads it', async () => {
    const PROMPT = 'SUBMSG: start the helper.';
    const NOTE = 'MSGPROBE: skip the tests folder.';
    const childCalls: string[] = [];
    const r = await helperRig(PROMPT, 'CHILDPROBE: wait for a note.', 5_000, childCalls);
    try {
      const { fake, wire } = r;
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      expect(await runAllowing(fake, id, PROMPT)).toBe('completed');
      const spawned = xaiLines(fake, id, 'subagent_spawned')[0]!;
      const child = spawned.child_session_id as string;
      expect(typeof spawned.agentAddress).toBe('string');
      await until(() => childCalls.length === 1, AGENT_WAIT_MS);
      expect(await fake.host.messageSubagent(id, child, NOTE)).toBe('sent');
      const asked = wire.in.find((m) => m.method === '_x.ai/subagent/message')!;
      expect(asked.params).toEqual({
        sessionId: id,
        agentAddress: spawned.agentAddress,
        content: [{ type: 'text', text: NOTE }],
      });
      const answer = wire.out.find((m) => m.id === asked.id && m.method === undefined)!;
      expect(answer.result).toEqual({ result: { kind: 'accepted', messageId: expect.any(String) } });
      await until(() => childCalls.some((text) => text.includes(NOTE)), AGENT_WAIT_MS);
      await until(() => xaiLines(fake, id, 'subagent_finished').length === 1, AGENT_WAIT_MS);
      expect(fake.host.liveActivity(id).agents).toEqual([]);
      const stored = readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
        .filter((f) => f.endsWith('updates.jsonl'))
        .map((f) => readFileSync(join(fake.root, f), 'utf8'))
        .filter((text) => text.includes('"subagent_spawned"') && text.includes(child));
      expect(stored).toHaveLength(1);
      expect(stored[0]).not.toContain('agentAddress');
      expect(stored[0]).not.toContain(spawned.agentAddress as string);
    } finally {
      await r.close();
    }
  }, 90_000);
});
