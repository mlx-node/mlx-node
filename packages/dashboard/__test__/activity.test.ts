/**
 * Background commands, monitors and schedules of the activity rail, from `updates.jsonl` lines shaped like the agent's:
 * the command lines are copied from a real Work session (commands shortened), the monitor and schedule lines follow the
 * agent's wire types. Each scenario is read from disk and also sent live on the methods the agent uses, so both views
 * get the same rows.
 */

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer, type UpdateLine } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { readSessionUpdate } from '../src/chat/acp/messages.js';
import { ActivityTracker } from '../src/chat/agent/activity-tracker.js';
import { liveFilter } from '../src/chat/agent/live-filter.js';
import type { TranscriptOp } from '../src/chat/types.js';
import {
  activityOf,
  ASK_TITLE_MAX,
  askRun,
  asksOf,
  eventsWords,
  hasActivity,
  lastLine,
  schedulesOf,
  tasksOf,
  waitWords,
  wakeReportOf,
  wakeReportsOf,
} from '../ui/src/lib/activity.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type {
  HelperProgress,
  LiveActivity,
  PermissionInteraction,
  PlanInteraction,
  McpInteraction,
  QuestionInteraction,
} from '../ui/src/lib/chat-types.js';

const SID = '01a1d519-3110-7d00-8e40-968c8338ee5b';
const CWD = '/Users/dev/proj';
const T0 = 1_794_578_383_596;

function envelope(method: string, ms: number, update: object, meta: object = {}): string {
  return JSON.stringify({
    timestamp: Math.floor(ms / 1000),
    method,
    params: { sessionId: SID, update, _meta: { eventId: `${SID}-${ms}`, agentTimestampMs: ms, ...meta } },
  });
}
const acp = (ms: number, update: object): string => envelope('session/update', ms, update);
const xai = (ms: number, update: object, meta: object = {}): string =>
  envelope('_x.ai/session/update', ms, update, meta);

const iso = (ms: number): string => new Date(ms).toISOString().replace('Z', '+00:00');
const systemTime = (ms: number) => ({ secs_since_epoch: Math.floor(ms / 1000), nanos_since_epoch: (ms % 1000) * 1e6 });
const logOf = (task: string): string =>
  `/Users/dev/.mlx-agent/sessions/%2FUsers%2Fdev%2Fproj/${SID}/terminal/${task}.log`;

function toolMeta(name: string, kind: string, label: string) {
  return { 'x.ai/tool': { version: 1, name, kind, namespace: 'grok_build', label, read_only: false } };
}

interface Task {
  call: string;
  id: string;
  command: string;
  description: string;
  kind: 'bash' | 'monitor';
  startMs: number;
}

function listRow(task: Task, end?: { ms: number; exitCode: number | null; signal: string | null }) {
  return {
    task_id: task.id,
    command: task.command,
    ...(task.kind === 'monitor' ? { display_command: `[monitor] ${task.description}` } : {}),
    description: task.description,
    cwd: CWD,
    kind: task.kind,
    status: end === undefined ? 'running' : end.exitCode === 0 ? 'completed' : 'failed',
    started_at: iso(task.startMs),
    ...(end === undefined ? {} : { ended_at: iso(end.ms) }),
    output_file: logOf(task.id),
    ...(end?.exitCode !== undefined && end.exitCode !== null ? { exit_code: end.exitCode } : {}),
    ...(end?.signal !== undefined && end.signal !== null ? { signal: end.signal } : {}),
  };
}

/**
 * `run_terminal_command` with `block_until_ms`, moved to the background (real lines 392-397 of the session). `others`:
 * the tasks that still run beside it, which the agent's whole list names too.
 */
function backgroundCommand(
  at: number,
  task: Omit<Task, 'kind' | 'startMs'>,
  blockMs: number,
  others: readonly Task[] = [],
): string[] {
  const t: Task = { ...task, kind: 'bash', startMs: at + 39 + blockMs };
  const input = { command: t.command, description: t.description };
  const summary =
    blockMs === 0
      ? `Background task ${t.id} started`
      : `Command "${t.command}" is still running after ${blockMs / 1000}s (block_until_ms) and has been moved to the background as task ${t.id}.`;
  return [
    acp(at, {
      sessionUpdate: 'tool_call',
      toolCallId: t.call,
      title: 'run_terminal_command',
      rawInput: { ...input, block_until_ms: blockMs },
      _meta: toolMeta('run_terminal_command', 'execute', 'Run Command'),
    }),
    acp(at + 22, {
      sessionUpdate: 'tool_call_update',
      toolCallId: t.call,
      kind: 'execute',
      title: `Execute \`${t.command}\``,
      content: [{ type: 'content', content: { type: 'text', text: t.description } }],
      locations: [],
      rawInput: { variant: 'Bash', ...input, is_background: false, block_until_ms: blockMs },
    }),
    xai(t.startMs, {
      sessionUpdate: 'task_backgrounded',
      tool_call_id: t.call,
      task_id: t.id,
      command: t.command,
      cwd: CWD,
      output_file: logOf(t.id),
      description: t.description,
    }),
    xai(t.startMs + 1, { sessionUpdate: 'background_tasks', tasks: [listRow(t), ...others.map((o) => listRow(o))] }),
    acp(t.startMs + 61, {
      sessionUpdate: 'tool_call_update',
      toolCallId: t.call,
      status: 'completed',
      title: `[bg] ${t.command} (${t.id.slice(0, 8)})`,
      content: [{ type: 'content', content: { type: 'text', text: summary } }],
      rawOutput: {
        type: 'BackgroundTaskStarted',
        task_id: t.id,
        task_type: 'bash',
        output_file: logOf(t.id),
        status: 'running',
        command: t.command,
        summary,
        retrieval_hint: `Use get_command_or_subagent_output with task_ids=["${t.id}"] when you need the output.`,
        pid: 19677,
      },
    }),
  ];
}

/** The agent's end of a task: `task_completed`, then its whole list again (real lines 446 and 448). */
function taskEnd(
  task: Task,
  ms: number,
  exitCode: number | null,
  signal: string | null,
  others: readonly Task[] = [],
): string[] {
  const killed = signal === 'killed';
  return [
    xai(ms + 56, {
      sessionUpdate: 'task_completed',
      task_snapshot: {
        task_id: task.id,
        command: task.command,
        cwd: CWD,
        start_time: systemTime(task.startMs),
        end_time: systemTime(ms),
        output: 'RUN 1: exit=0 elapsed=39s\n',
        output_file: logOf(task.id),
        truncated: false,
        output_total_bytes: 26,
        exit_code: exitCode,
        signal,
        completed: true,
        kind: task.kind,
        block_waited: false,
        explicitly_killed: killed,
        kill_result_delivered: killed,
        owner_session_id: SID,
        description: task.description,
        is_backgrounded: true,
      },
      will_wake: !killed,
    }),
    xai(ms + 68, {
      sessionUpdate: 'background_tasks',
      tasks: [listRow(task, { ms, exitCode, signal }), ...others.map((o) => listRow(o))],
    }),
  ];
}

const CMD: Task = {
  call: 'toolu_f9d082df65b54a4b89665d9f5d0d96a2',
  id: '01a1d535-0313-7a03-a2cb-c89d310104c6',
  command: 'B=/tmp/repro2-tsf\nbash /tmp/repro2_loop.sh',
  description: 'Run heavy #2 race loop in background',
  kind: 'bash',
  startMs: T0 + 39,
};

const MONITOR: Task = {
  call: 'toolu_5c1e7a90d1f34f0e9b2a6c4d8e0f1a23',
  id: '01a24f10-7a2c-7c11-9d0e-3b1f5a6c7d80',
  command: 'gh run watch 123 --exit-status >/dev/null 2>&1 && echo DONE || echo FAILED',
  description: 'CI run 123',
  kind: 'monitor',
  startMs: T0 + 60_040,
};

/** The `monitor` tool: `task_backgrounded` with `monitor_description`, then its `Monitor` result (no content). */
function monitorStart(at: number, task: Task, others: readonly Task[] = []): string[] {
  return [
    acp(at, {
      sessionUpdate: 'tool_call',
      toolCallId: task.call,
      title: 'monitor',
      rawInput: { command: task.command, description: task.description },
      _meta: toolMeta('monitor', 'monitor', 'Monitor'),
    }),
    xai(task.startMs, {
      sessionUpdate: 'task_backgrounded',
      tool_call_id: task.call,
      task_id: task.id,
      command: task.command,
      cwd: CWD,
      output_file: `/Users/dev/.mlx-agent/sessions/%2FUsers%2Fdev%2Fproj/${SID}/terminal/monitor-${task.call}.log`,
      monitor_description: task.description,
      description: task.description,
    }),
    xai(task.startMs + 1, {
      sessionUpdate: 'background_tasks',
      tasks: [listRow(task), ...others.map((o) => listRow(o))],
    }),
    acp(task.startMs + 30, {
      sessionUpdate: 'tool_call_update',
      toolCallId: task.call,
      status: 'completed',
      rawOutput: { type: 'Monitor', taskId: task.id, timeoutMs: 36_000_000, persistent: false },
    }),
  ];
}

const SCHEDULE = '01a24f2b-0c3d-7e4f-8a5b-6c7d8e9f0a1b';
const SCHEDULE_PROMPT = 'Check the CI run and report failures';
const SCHEDULE_CALL = 'toolu_0a9b8c7d6e5f40312a3b4c5d6e7f8091';
const NEXT_FIRE = '2026-11-13T14:09:43.640123456+00:00';
const schedulerMeta = (revision: number) => ({
  'x.ai/schedulerGeneration': '01a24f2b-0000-7000-8000-000000000000',
  'x.ai/schedulerRevision': revision,
});

function scheduleCreated(ms: number, nextFireAt: string, revision: number): string {
  return xai(
    ms,
    {
      sessionUpdate: 'scheduled_task_created',
      task_id: SCHEDULE,
      prompt: SCHEDULE_PROMPT,
      human_schedule: 'every 10 minutes',
      next_fire_at: nextFireAt,
    },
    schedulerMeta(revision),
  );
}

/** `scheduler_create`: the agent writes `scheduled_task_created` before the tool result. */
function scheduleStart(at: number): string[] {
  return [
    acp(at, {
      sessionUpdate: 'tool_call',
      toolCallId: SCHEDULE_CALL,
      title: 'scheduler_create',
      rawInput: { interval: '10m', prompt: SCHEDULE_PROMPT },
      _meta: toolMeta('scheduler_create', 'other', 'Scheduler'),
    }),
    scheduleCreated(at + 4, NEXT_FIRE, 1),
    acp(at + 9, {
      sessionUpdate: 'tool_call_update',
      toolCallId: SCHEDULE_CALL,
      status: 'completed',
      rawOutput: { type: 'SchedulerCreate', id: SCHEDULE, humanSchedule: 'every 10 minutes', updated: false },
    }),
  ];
}

const scheduleRemoved = (ms: number, reason: string, revision: number): string =>
  xai(ms, { sessionUpdate: 'scheduled_task_deleted', task_id: SCHEDULE, reason }, schedulerMeta(revision));

/** The live method of each stored xAI kind the agent sends on a method of its own; the rest come as notifications. */
const LIVE_METHOD: Record<string, string> = {
  task_backgrounded: '_x.ai/task_backgrounded',
  task_completed: '_x.ai/task_completed',
  scheduled_task_created: '_x.ai/scheduled_task_created',
  scheduled_task_deleted: '_x.ai/scheduled_task_deleted',
};

function replay(ops: readonly TranscriptOp[], start: TranscriptEntry[] = []): TranscriptEntry[] {
  const entries = [...start];
  for (const op of ops) {
    if (op.op === 'push') entries.push(op.entry);
    else if (op.op === 'set') entries[op.index] = op.entry;
  }
  return entries;
}

function liveLine(text: string): UpdateLine {
  const disk = JSON.parse(text) as { method: string; params: { update: { sessionUpdate: string } } };
  const kind = disk.params.update.sessionUpdate;
  const method = disk.method === 'session/update' ? disk.method : (LIVE_METHOD[kind] ?? '_x.ai/session_notification');
  const message = readSessionUpdate(method, disk.params);
  if (message === null) throw new Error(`${method} is not read as an update`);
  const line: UpdateLine = { xai: message.xai, update: message.update, meta: message.meta };
  expect(liveFilter(line, { turnRunning: true, modelId: 'm' })).toBe('pass');
  return line;
}

/** The disk transcript of `lines`; the same lines sent live give the same entries. */
function read(lines: string[]): TranscriptEntry[] {
  const disk = parseUpdates(`${lines.join('\n')}\n`).transcript;
  const reducer = new TranscriptReducer({ now: () => 1 });
  const ops = lines.flatMap((text) => reducer.apply(liveLine(text)));
  expect(replay(ops)).toEqual(disk);
  return disk;
}

const toolAt = (entries: readonly TranscriptEntry[], call: string): number =>
  entries.findIndex((e) => e.kind === 'tool' && e.id === call);

describe('tasksOf: background commands', () => {
  it('a command moved to the background at once runs until the agent says it exited', () => {
    const start = backgroundCommand(T0, CMD, 0);
    const running = read(start);
    expect(tasksOf(running)).toEqual([
      {
        kind: 'command',
        id: CMD.id,
        title: CMD.description,
        startTs: T0,
        anchor: toolAt(running, CMD.call),
        stopping: false,
        live: false,
        events: 0,
        lastEvent: null,
      },
    ]);

    const endMs = T0 + 259_327;
    const ended = read([...start, ...taskEnd(CMD, endMs, 0, null)]);
    expect(tasksOf(ended)).toEqual([]);
    const entry = ended[toolAt(ended, CMD.call)];
    expect(entry?.kind === 'tool' ? entry.task : null).toEqual({
      id: CMD.id,
      kind: 'command',
      label: CMD.description,
      endTs: endMs,
    });
  });

  it('a kill ends it: the kill call, then the end the agent writes for it', () => {
    const killMs = T0 + 90_000;
    const kill = 'toolu_7d2e4c1b9a8f4e6d0c3b5a7f9e1d2c4b';
    const lines = [
      ...backgroundCommand(T0, CMD, 0),
      acp(killMs, {
        sessionUpdate: 'tool_call',
        toolCallId: kill,
        title: 'kill_command_or_subagent',
        rawInput: { task_id: CMD.id },
        _meta: toolMeta('kill_command_or_subagent', 'background_task_action', 'Background Task'),
      }),
      ...taskEnd(CMD, killMs + 20, null, 'killed'),
      acp(killMs + 120, {
        sessionUpdate: 'tool_call_update',
        toolCallId: kill,
        status: 'completed',
        title: `kill ${CMD.id.slice(0, 8)} (killed)`,
        rawOutput: { type: 'KillTask', Result: { task_id: CMD.id, outcome: 'killed', message: 'Task killed' } },
      }),
    ];
    const entries = read(lines);
    expect(tasksOf(entries)).toEqual([]);
    const entry = entries[toolAt(entries, CMD.call)];
    expect(entry?.kind === 'tool' ? entry.task?.endTs : null).toBe(killMs + 20);
  });

  it('a command still running at its block time is moved to the background and listed', () => {
    const entries = read(backgroundCommand(T0, CMD, 30_000));
    expect(tasksOf(entries).map((row) => [row.kind, row.id, row.anchor])).toEqual([
      ['command', CMD.id, toolAt(entries, CMD.call)],
    ]);
  });

  it('a foreground command that finished within its block time is not listed', () => {
    const call = 'toolu_61023e08b6f243cbb3ca166b5ba1c3db';
    const command = 'cd /tmp/scan-src && docker build -t oss-scan-lean .';
    const description = 'Build lean Docker image';
    const entries = read([
      acp(T0, {
        sessionUpdate: 'tool_call',
        toolCallId: call,
        title: 'run_terminal_command',
        rawInput: { command, description, block_until_ms: 600_000 },
        _meta: toolMeta('run_terminal_command', 'execute', 'Run Command'),
      }),
      acp(T0 + 42, {
        sessionUpdate: 'tool_call_update',
        toolCallId: call,
        kind: 'execute',
        title: `Execute \`${command}\``,
        content: [{ type: 'content', content: { type: 'text', text: description } }],
        locations: [],
        rawInput: { variant: 'Bash', command, description, is_background: false, block_until_ms: 600_000 },
      }),
      acp(T0 + 41_622, {
        sessionUpdate: 'tool_call_update',
        toolCallId: call,
        status: 'completed',
        content: [{ type: 'content', content: { type: 'text', text: 'exit: 0\n#13 DONE 10.3s\n' } }],
        rawOutput: {
          type: 'Bash',
          output: [101, 120, 105, 116],
          output_for_prompt: 'exit: 0\n#13 DONE 10.3s\n',
          exit_code: 0,
          command,
          truncated: false,
          signal: null,
          timed_out: false,
          description,
          current_dir: CWD,
          output_file: logOf('fg'),
          total_bytes: 24,
        },
      }),
    ]);
    expect(entries).toHaveLength(1);
    expect(entries[0]?.kind === 'tool' ? entries[0].task : 'not a tool').toBeUndefined();
    expect(tasksOf(entries)).toEqual([]);
  });

  it('a command that exits before its result arrives is ended once the result names it', () => {
    const [call, update, backgrounded, , result] = backgroundCommand(T0, CMD, 0);
    const endMs = T0 + 400;
    const entries = read([call!, update!, ...taskEnd(CMD, endMs, 0, null), backgrounded!, result!]);
    expect(tasksOf(entries)).toEqual([]);
    const entry = entries[toolAt(entries, CMD.call)];
    expect(entry?.kind === 'tool' ? entry.task?.endTs : null).toBe(endMs);
  });

  it("a new agent process lists no task: the old process's commands ended; a list cut to size proves nothing", () => {
    const start = backgroundCommand(T0, CMD, 0);
    const loadMs = T0 + 3_600_000;
    const cut = read([...start, xai(loadMs, { sessionUpdate: 'background_tasks', tasks: [], truncated: true })]);
    expect(tasksOf(cut)).toHaveLength(1);
    const reloaded = read([...start, xai(loadMs, { sessionUpdate: 'background_tasks', tasks: [] })]);
    expect(tasksOf(reloaded)).toEqual([]);
    const entry = reloaded[toolAt(reloaded, CMD.call)];
    expect(entry?.kind === 'tool' ? entry.task?.endTs : null).toBe(loadMs);
  });

  it('seeded from disk, the empty list of the next load ends it with one set op, and the disk entry stays as read', () => {
    const disk = parseUpdates(`${backgroundCommand(T0, CMD, 0).join('\n')}\n`).transcript;
    const reducer = new TranscriptReducer({ now: () => 1 });
    reducer.seed(disk, []);
    const loadMs = T0 + 3_600_000;
    const ops = reducer.apply(liveLine(xai(loadMs, { sessionUpdate: 'background_tasks', tasks: [] })));
    const at = toolAt(disk, CMD.call);
    expect(ops.map((op) => (op.op === 'set' ? [op.op, op.index] : [op.op]))).toEqual([['set', at]]);
    const after = replay(ops, disk);
    expect(tasksOf(after)).toEqual([]);
    expect(tasksOf(disk)).toHaveLength(1);
  });
});

describe('tasksOf: monitors', () => {
  it('a monitor runs until its script exits', () => {
    const start = [...backgroundCommand(T0, CMD, 0), ...monitorStart(T0 + 60_000, MONITOR, [CMD])];
    const running = read(start);
    expect(tasksOf(running)).toEqual([
      {
        kind: 'command',
        id: CMD.id,
        title: CMD.description,
        startTs: T0,
        anchor: toolAt(running, CMD.call),
        stopping: false,
        live: false,
        events: 0,
        lastEvent: null,
      },
      {
        kind: 'monitor',
        id: MONITOR.id,
        title: MONITOR.description,
        startTs: T0 + 60_000,
        anchor: toolAt(running, MONITOR.call),
        stopping: false,
        live: false,
        events: 0,
        lastEvent: null,
      },
    ]);
    const ended = read([...start, ...taskEnd(MONITOR, T0 + 600_000, 0, null, [CMD])]);
    expect(tasksOf(ended).map((row) => row.id)).toEqual([CMD.id]);
  });

  it('the Monitor result names a monitor even when task_backgrounded had no description to tell', () => {
    const blank: Task = { ...MONITOR, description: '' };
    const lines = monitorStart(T0, blank).map((text) => {
      const line = JSON.parse(text) as { params: { update: Record<string, unknown> } };
      delete line.params.update.monitor_description;
      delete line.params.update.description;
      return JSON.stringify(line);
    });
    const rows = tasksOf(read(lines));
    expect(rows.map((row) => [row.kind, row.title])).toEqual([
      ['monitor', 'gh run watch 123 --exit-status >/dev/null 2>&1 && echo DONE || echo FAILED'],
    ]);
  });
});

describe('schedulesOf', () => {
  const SET_AT = T0 + 120_000;

  it('a schedule is listed from scheduler_create until scheduler_delete removes it', () => {
    const start = scheduleStart(SET_AT);
    const made = read(start);
    expect(schedulesOf(made)).toEqual([
      {
        id: SCHEDULE,
        title: SCHEDULE_PROMPT,
        nextTs: Date.parse(NEXT_FIRE),
        repeat: 'every 10 minutes',
        anchor: toolAt(made, SCHEDULE_CALL),
        state: 'waiting',
      },
    ]);

    const del = 'toolu_9f8e7d6c5b4a43210f1e2d3c4b5a6978';
    const deleted = read([
      ...start,
      acp(SET_AT + 60_000, {
        sessionUpdate: 'tool_call',
        toolCallId: del,
        title: 'scheduler_delete',
        rawInput: { id: SCHEDULE },
        _meta: toolMeta('scheduler_delete', 'other', 'Scheduler'),
      }),
      scheduleRemoved(SET_AT + 60_003, 'deleted', 2),
      acp(SET_AT + 60_008, {
        sessionUpdate: 'tool_call_update',
        toolCallId: del,
        status: 'completed',
        rawOutput: { type: 'SchedulerDelete', success: true, message: `Task ${SCHEDULE} deleted` },
      }),
    ]);
    expect(schedulesOf(deleted)).toEqual([]);
    const entry = deleted[toolAt(deleted, SCHEDULE_CALL)];
    expect(entry?.kind === 'tool' ? entry.schedule?.endTs : null).toBe(SET_AT + 60_003);
  });

  it('the agent process stopping is no removal; the next load names the next run; expiry removes it', () => {
    const start = scheduleStart(SET_AT);
    const stopped = [...start, scheduleRemoved(SET_AT + 3_000_000, 'shutdown', 2)];
    expect(schedulesOf(read(stopped)).map((row) => row.nextTs)).toEqual([Date.parse(NEXT_FIRE)]);

    const later = '2026-11-13T16:00:00.000000001+00:00';
    const loaded = [...stopped, scheduleCreated(SET_AT + 7_200_000, later, 1)];
    expect(schedulesOf(read(loaded)).map((row) => [row.nextTs, row.anchor])).toEqual([
      [Date.parse(later), toolAt(read(loaded), SCHEDULE_CALL)],
    ]);

    expect(schedulesOf(read([...loaded, scheduleRemoved(SET_AT + 604_800_000, 'expired', 2)]))).toEqual([]);
  });

  // Mutation: let the reducer read `scheduled_task_fired` again → the live entry moves its time and disk does not.
  it('a live fire leaves the entry as disk has it; the next run it names rides the live facts', () => {
    const reducer = new TranscriptReducer({ now: () => 1 });
    const ops = scheduleStart(SET_AT).flatMap((text) => reducer.apply(liveLine(text)));
    const next = '2026-11-13T14:19:43.640000000+00:00';
    const run = '01a24f40-1111-7222-8333-444455556666';
    const fired = readSessionUpdate('_x.ai/scheduled_task_fired', {
      sessionId: SID,
      update: {
        sessionUpdate: 'scheduled_task_fired',
        task_id: SCHEDULE,
        prompt: SCHEDULE_PROMPT,
        human_schedule: 'every 10 minutes',
        next_fire_at: next,
        subagent_id: run,
      },
      _meta: { agentTimestampMs: SET_AT + 600_000, ...schedulerMeta(2) },
    })!;
    expect(reducer.apply({ xai: fired.xai, update: fired.update, meta: fired.meta })).toEqual([]);
    const live = replay(ops);
    expect(live).toEqual(read(scheduleStart(SET_AT)));
    expect(schedulesOf(live).map((row) => row.nextTs)).toEqual([Date.parse(NEXT_FIRE)]);

    const tracker = new ActivityTracker();
    tracker.apply({ xai: true, update: fired.update });
    const facts = tracker.live();
    expect(facts.schedules).toEqual([{ id: SCHEDULE, deleting: null, firing: run, nextTs: Date.parse(next) }]);
    expect(schedulesOf(live, facts).map((row) => [row.state, row.nextTs])).toEqual([['firing', Date.parse(next)]]);
  });

  const liveOf = (schedules: LiveActivity['schedules']): LiveActivity => ({
    helpers: [],
    agents: [],
    tasks: [],
    schedules,
    reports: [],
    stoppedByYou: [],
  });

  // Mutation: let `paused` win over a Delete's phase → a paused schedule being loaded for its Delete reads "Paused".
  it("a row reads its state: a Delete's phase first, then paused with no agent, then firing, else waiting", () => {
    const entries = read(scheduleStart(SET_AT));
    const stateOf = (live: LiveActivity | undefined, agentLive = true) =>
      schedulesOf(entries, live, agentLive)[0]!.state;
    expect(stateOf(undefined)).toBe('waiting');
    expect(stateOf(liveOf([{ id: SCHEDULE, deleting: null, firing: null }]))).toBe('waiting');
    expect(stateOf(liveOf([{ id: SCHEDULE, deleting: null, firing: 'run-1' }]))).toBe('firing');
    expect(stateOf(liveOf([{ id: 'other', deleting: 'removing', firing: 'run-1' }]))).toBe('waiting');
    expect(stateOf(liveOf([{ id: SCHEDULE, deleting: 'removing', firing: 'run-1' }]))).toBe('removing');
    expect(stateOf(undefined, false)).toBe('paused');
    expect(stateOf(liveOf([{ id: SCHEDULE, deleting: null, firing: 'run-1' }]), false)).toBe('paused');
    expect(stateOf(liveOf([{ id: SCHEDULE, deleting: 'opening', firing: null }]), false)).toBe('opening');
  });

  // Mutation: leave schedules out of the `agentLive` rule → a cold conversation's schedule reads "Next 14:05".
  it("a cold conversation's schedule is paused: the agent does not hold it", () => {
    const entries = read(scheduleStart(SET_AT));
    const live = liveOf([{ id: SCHEDULE, deleting: null, firing: null }]);
    expect(activityOf({ entries, runs: [], live, turnRunning: false }).scheduled.map((row) => row.state)).toEqual([
      'waiting',
    ]);
    expect(
      activityOf({ entries, runs: [], turnRunning: false, agentLive: false }).scheduled.map((row) => row.state),
    ).toEqual(['paused']);
  });
});

describe('activityOf with tasks and schedules', () => {
  it('lists the running command and monitor by start, and the schedule', () => {
    const entries = read([
      ...monitorStart(T0 + 60_000, MONITOR),
      ...backgroundCommand(T0 + 90_000, CMD, 0, [MONITOR]),
      ...scheduleStart(T0 + 120_000),
    ]);
    const activity = activityOf({ entries, runs: [], turnRunning: false });
    expect(activity.running.map((row) => [row.kind, row.id])).toEqual([
      ['monitor', MONITOR.id],
      ['command', CMD.id],
    ]);
    expect(activity.scheduled.map((row) => row.id)).toEqual([SCHEDULE]);
  });
});

describe('needs you: the open requests as rail rows', () => {
  const ask = (id: string, over: Partial<PermissionInteraction> = {}): PermissionInteraction => ({
    id,
    kind: 'permission',
    toolCallId: `call-${id}`,
    openedAt: T0,
    toolName: 'run_terminal_command',
    access: 'bash',
    title: 'Execute `rm -rf build/`',
    input: '{}',
    command: 'rm -rf build/',
    path: null,
    resolvedPath: null,
    options: [],
    flags: [],
    ...over,
  });
  const plan: PlanInteraction = { id: 'p', kind: 'plan', toolCallId: 'call-p', openedAt: T0 + 3, plan: '# Plan' };

  it('oldest first; a title per kind; who asks; the front card and Auto asks', () => {
    const rows = asksOf(
      [
        ask('a', { command: 'cargo test --workspace\nrm -rf target' }),
        ask('b', {
          openedAt: T0 + 1,
          access: 'edit',
          command: null,
          path: '/Users/dev/proj/src/app.ts',
          from: { label: 'fix tests', run: null, cwd: null, child: 'child-1' },
        }),
        ask('c', {
          openedAt: T0 + 2,
          access: 'mcp',
          command: null,
          title: 'github__create_issue',
          from: { label: 'packer', run: 'Release', cwd: null, child: 'child-2' },
          autoAsk: 'not-covered',
        }),
        plan,
      ],
      'c',
    );
    expect(
      rows.map((row) => [row.id, row.kind, row.access, row.title, row.path, row.since, row.front, row.autoAsk]),
    ).toEqual([
      ['a', 'permission', 'bash', 'cargo test --workspace', false, T0, false, false],
      ['b', 'permission', 'edit', '/Users/dev/proj/src/app.ts', true, T0 + 1, false, false],
      ['c', 'permission', 'mcp', 'github__create_issue', false, T0 + 2, true, true],
      ['p', 'plan', null, CHAT.needs.plan, false, T0 + 3, false, false],
    ]);
    expect(rows.map((row) => row.from)).toEqual([
      { who: 'agent' },
      { who: 'subagent', label: 'fix tests', childId: 'child-1' },
      { who: 'workflow', label: 'packer', run: 'Release', childId: 'child-2' },
      { who: 'agent' },
    ]);
  });

  it('with no front picked, or a front that is gone, the oldest is the front', () => {
    expect(asksOf([ask('a'), ask('b')], null).map((row) => row.front)).toEqual([true, false]);
    expect(asksOf([ask('a'), ask('b')], 'gone').map((row) => row.front)).toEqual([true, false]);
    expect(asksOf([], null)).toEqual([]);
  });

  it('a long command is cut at its first line and the cap; a request with no words of its own says what it is', () => {
    const long = 'x'.repeat(ASK_TITLE_MAX + 10);
    const [row] = asksOf([ask('a', { command: long })], null);
    expect(Array.from(row!.title)).toHaveLength(ASK_TITLE_MAX);
    expect(row!.title.endsWith('…')).toBe(true);
    const bare = asksOf(
      [ask('m', { access: 'mcp', command: null, title: '' }), ask('o', { access: 'other', command: null, title: ' ' })],
      null,
    );
    expect(bare.map((r) => r.title)).toEqual([CHAT.permTool, CHAT.permOther]);
  });

  it('an open request alone is activity: the docked rail opens for it; the rows come first', () => {
    const activity = activityOf({ entries: [], interactions: [ask('a')], runs: [], turnRunning: false });
    expect(activity.needs.map((row) => row.id)).toEqual(['a']);
    expect(hasActivity(activity)).toBe(true);
    expect(hasActivity(activityOf({ entries: [], runs: [], turnRunning: false }))).toBe(false);
  });

  it('a question: its first question is the title (one line, cut), the count rides along; no words says what it is', () => {
    const question = (id: string, texts: string[]): QuestionInteraction => ({
      id,
      kind: 'question',
      toolCallId: `call-${id}`,
      openedAt: T0,
      planMode: false,
      questions: texts.map((text) => ({ text, multi: false, options: [] })),
    });
    const rows = asksOf(
      [question('q', ['Which database\nshould it use?', 'Which checks?']), question('e', [' '])],
      null,
    );
    expect(rows.map((row) => [row.kind, row.access, row.title, row.questions, row.from])).toEqual([
      ['question', null, 'Which database should it use?', 2, { who: 'agent' }],
      ['question', null, CHAT.needs.question, 1, { who: 'agent' }],
    ]);
    const [long] = asksOf([question('l', ['q'.repeat(ASK_TITLE_MAX + 5)])], null);
    expect(Array.from(long!.title)).toHaveLength(ASK_TITLE_MAX);
    expect(asksOf([ask('a')], null)[0]).not.toHaveProperty('questions');
  });

  it('an MCP request: "{server} asks for input" on one line, cut; no path, no count', () => {
    const mcp = (id: string, server: string): McpInteraction => ({
      id,
      kind: 'mcp-input',
      toolCallId: `mcp-elicit-${id}`,
      openedAt: T0,
      mode: 'form',
      server,
      message: 'Which repository?',
      form: { fields: [], leftOut: 0 },
    });
    const rows = asksOf([mcp('m', 'git\nhub'), mcp('l', 's'.repeat(ASK_TITLE_MAX))], 'm');
    expect(rows.map((row) => [row.kind, row.access, row.title, row.path, row.front, row.from])).toEqual([
      ['mcp-input', null, fill(CHAT.mcp.formTitle, { server: 'git hub' }), false, true, { who: 'agent' }],
      ['mcp-input', null, rows[1]!.title, false, false, { who: 'agent' }],
    ]);
    expect(Array.from(rows[1]!.title)).toHaveLength(ASK_TITLE_MAX);
    expect(rows[0]).not.toHaveProperty('questions');
  });

  it('an MCP link request: "{server} asks you to open a page"', () => {
    const link: McpInteraction = {
      id: 'k',
      kind: 'mcp-input',
      toolCallId: 'mcp-elicit-k',
      openedAt: T0,
      mode: 'url',
      server: 'github',
      message: 'Sign in.',
      link: { url: 'https://github.com/login', host: 'github.com', secure: true },
    };
    expect(asksOf([link], 'k')[0]?.title).toBe(fill(CHAT.mcp.linkTitle, { server: 'github' }));
  });

  // Mutation: read only a permission's `from.run` in `askRun` → a run's MCP request takes over the composer.
  it("an MCP request while only a run works is the run's; a run helper's permission too; the rest are the answer's", () => {
    const mcp: McpInteraction = {
      id: 'r',
      kind: 'mcp-input',
      toolCallId: 'mcp-elicit-r',
      openedAt: T0,
      mode: 'form',
      server: 'github',
      message: 'Which repository?',
      form: { fields: [], leftOut: 0 },
      run: 'deep-research',
    };
    expect(asksOf([mcp], null)[0]?.from).toEqual({ who: 'run', run: 'deep-research' });
    expect(askRun(mcp)).toBe('deep-research');
    const { run: _run, ...own } = mcp;
    expect(askRun(own)).toBeNull();
    expect(asksOf([own], null)[0]?.from).toEqual({ who: 'agent' });
    const helper = ask('h', { from: { label: 'packer', run: 'Release', cwd: null, child: 'c-1' } });
    expect(askRun(helper)).toBe('Release');
    expect(askRun(ask('s', { from: { label: 'fix tests', run: null, cwd: null, child: 'c-2' } }))).toBeNull();
    expect(askRun(ask('a'))).toBeNull();
  });

  it('a wait in words', () => {
    expect(waitWords(1_000)).toBe(CHAT.needs.second);
    expect(waitWords(42_000)).toBe(fill(CHAT.needs.seconds, { n: 42 }));
    expect(waitWords(60_000)).toBe(CHAT.needs.minute);
    expect(waitWords(26 * 60_000 + 10_000)).toBe(fill(CHAT.needs.minutes, { n: 26 }));
    expect(waitWords(2 * 3_600_000)).toBe(fill(CHAT.needs.hours, { n: 2 }));
  });
});

describe('helper progress on the rail rows (#10)', () => {
  const spawn = (ms: number, child: string, run?: string): string =>
    xai(ms, {
      sessionUpdate: 'subagent_spawned',
      subagent_id: child,
      parent_session_id: SID,
      child_session_id: child,
      subagent_type: 'explore',
      description: `Look at ${child}`,
      ...(run === undefined ? {} : { workflow_run_id: run }),
    });
  const tick = (childId: string, toolCalls: number): HelperProgress => ({
    childId,
    contextWindow: null,
    now: null,
    turns: 2,
    toolCalls,
    tokens: 18_200,
    contextPct: 41,
    errors: 0,
    failedCalls: 0,
    durationMs: 9_000,
    at: T0 + 10_000,
  });

  // Mutation: look the tick up by row index → the second helper gets the first one's numbers.
  it("each helper row gets its own tick, a workflow helper's too; none before the first tick", () => {
    const entries = read([spawn(T0 + 1_000, 'c1'), spawn(T0 + 2_000, 'c2', 'wf_1'), spawn(T0 + 3_000, 'c3')]);
    const live = {
      helpers: [tick('c2', 7), tick('c1', 12)],
      agents: [],
      tasks: [],
      schedules: [],
      reports: [],
      stoppedByYou: [],
    };
    const rows = activityOf({ entries, runs: [], live, turnRunning: true }).running;
    expect(rows.map((row) => (row.kind === 'subagent' ? [row.id, row.progress?.toolCalls ?? null] : null))).toEqual([
      ['c1', 12],
      ['c2', 7],
      ['c3', null],
    ]);
    const none = activityOf({ entries, runs: [], turnRunning: true }).running;
    expect(none.map((row) => (row.kind === 'subagent' ? row.progress : 'other'))).toEqual([null, null, null]);
  });

  // Mutation: leave subagents out of the `agentLive` rule → a dead helper reads "Starting…" for ever.
  it('a conversation not loaded in a running agent lists no helper: it died with the process', () => {
    const entries = read([spawn(T0 + 1_000, 'c1')]);
    expect(activityOf({ entries, runs: [], turnRunning: false }).running).toHaveLength(1);
    expect(activityOf({ entries, runs: [], turnRunning: false, agentLive: false }).running).toEqual([]);
  });
});

describe('a Stop of a command or monitor (T1)', () => {
  // Mutation: read `stopping` off the row index of `live.tasks` → the wrong row reads Stopping….
  it('a row reads stopping from the live facts by its task id; with no live facts none does', () => {
    const entries = read([...backgroundCommand(T0, CMD, 0), ...monitorStart(T0 + 60_000, MONITOR, [CMD])]);
    const live = {
      helpers: [],
      agents: [],
      tasks: [
        { id: 'not-a-row', kind: 'command' as const, stopping: true, events: 0, recent: [] },
        { id: MONITOR.id, kind: 'monitor' as const, stopping: true, events: 0, recent: [] },
        { id: CMD.id, kind: 'command' as const, stopping: false, events: 0, recent: [] },
      ],
      schedules: [],
      reports: [],
      stoppedByYou: [],
    };
    const rows = activityOf({ entries, runs: [], live, turnRunning: false }).running;
    expect(rows.map((row) => [row.id, row.kind !== 'subagent' && row.kind !== 'workflow' && row.stopping])).toEqual([
      [CMD.id, false],
      [MONITOR.id, true],
    ]);
    expect(tasksOf(entries).map((row) => row.kind !== 'subagent' && row.kind !== 'workflow' && row.stopping)).toEqual([
      false,
      false,
    ]);
  });

  // Mutation: mark every transcript row live → a row the host no longer tracks offers a Stop that can only 404.
  it('a row is live only while the host tracks its task: a stored row the host dropped offers no Stop', () => {
    const entries = read([...backgroundCommand(T0, CMD, 0), ...monitorStart(T0 + 60_000, MONITOR, [CMD])]);
    const live = {
      helpers: [],
      agents: [],
      tasks: [{ id: MONITOR.id, kind: 'monitor' as const, stopping: false, events: 0, recent: [] }],
      schedules: [],
      reports: [],
      stoppedByYou: [],
    };
    const liveOf = (rows: ReturnType<typeof tasksOf>): Array<[string, boolean]> =>
      rows.map((row) => [row.id, row.kind !== 'subagent' && row.kind !== 'workflow' && row.live]);
    expect(liveOf(tasksOf(entries, live))).toEqual([
      [CMD.id, false],
      [MONITOR.id, true],
    ]);
    expect(liveOf(tasksOf(entries))).toEqual([
      [CMD.id, false],
      [MONITOR.id, false],
    ]);
  });
});

describe('monitor events (T2)', () => {
  const liveOf = (tasks: LiveActivity['tasks'], reports: LiveActivity['reports'] = []): LiveActivity => ({
    helpers: [],
    agents: [],
    tasks,
    schedules: [],
    reports,
    stoppedByYou: [],
  });
  const monitorLive = (events: number, texts: string[]): LiveActivity['tasks'][number] => ({
    id: MONITOR.id,
    kind: 'monitor',
    stopping: false,
    events,
    recent: texts.map((text, i) => ({ at: T0 + i, text })),
  });

  // Mutation: read the oldest kept event for the last line → the row shows a stale line.
  it('a monitor row carries its event count and the last line of its newest event; with none, 0 and null', () => {
    const entries = read([...backgroundCommand(T0, CMD, 0), ...monitorStart(T0 + 60_000, MONITOR, [CMD])]);
    const live = liveOf([
      { id: CMD.id, kind: 'command', stopping: false, events: 0, recent: [] },
      monitorLive(12, ['old line', 'src/app.ts(12,4): warning\nerror TS2345: Argument\n\n']),
    ]);
    const rows = tasksOf(entries, live).map((row) =>
      row.kind === 'command' || row.kind === 'monitor' ? [row.id, row.events, row.lastEvent] : null,
    );
    expect(rows).toEqual([
      [CMD.id, 0, null],
      [MONITOR.id, 12, 'error TS2345: Argument'],
    ]);
    expect(tasksOf(entries).map((row) => (row.kind === 'monitor' ? row.events : null))).toEqual([null, 0]);
    expect(lastLine('a\r\nb\r\n  \n')).toBe('b');
    expect(lastLine(' \n\n')).toBe('');
    expect([eventsWords(1), eventsWords(12)]).toEqual([CHAT.activity.eventsOne, fill(CHAT.activity.events, { n: 12 })]);
  });

  type Wake = Extract<TranscriptEntry, { kind: 'wake' }>;
  const wake = (ts: number, extra: Partial<Wake> = {}): Wake => ({
    kind: 'wake',
    ts,
    cause: 'notifications',
    text: '',
    ...extra,
  });

  // Mutation: match a report to any notifications row (not by `ts`) → an older drain reads as a report.
  it('a report names a notifications row with its ts and no ended task, each monitor as its tool row names it', () => {
    const entries = read([...backgroundCommand(T0, CMD, 0), ...monitorStart(T0 + 60_000, MONITOR, [CMD])]);
    const reports = wakeReportsOf(
      liveOf(
        [],
        [
          {
            wakeTs: T0 + 90_000,
            monitors: [
              { id: MONITOR.id, description: 'its event words' },
              { id: 'gone-monitor', description: 'Watch the queue' },
              { id: 'quiet-monitor', description: '' },
            ],
          },
        ],
      ),
      entries,
    );
    const named = [
      { id: MONITOR.id, kind: 'monitor', label: MONITOR.description },
      { id: 'gone-monitor', kind: 'monitor', label: 'Watch the queue' },
      { id: 'quiet-monitor', kind: 'monitor', label: '' },
    ];
    expect(wakeReportOf(wake(T0 + 90_000), reports)).toEqual(named);
    expect(wakeReportOf(wake(T0 + 80_000), reports)).toBeNull();
    expect(wakeReportOf(wake(T0 + 90_000, { cause: 'unknown' }), reports)).toBeNull();
    expect(
      wakeReportOf(wake(T0 + 90_000, { tasks: [{ id: CMD.id, kind: 'command', label: 'x' }] }), reports),
    ).toBeNull();
    expect(wakeReportsOf(undefined, entries).size).toBe(0);
    expect(wakeReportsOf(liveOf([]), entries).size).toBe(0);
  });
});

describe('the goal (#17)', () => {
  const goal = (over: Partial<Extract<TranscriptEntry, { kind: 'goal' }>> = {}): TranscriptEntry => ({
    kind: 'goal',
    ts: T0 + 1,
    goalId: 'g1',
    objective: 'Fix the flaky cache tests',
    status: 'active',
    phase: 'executing',
    tokensUsed: 31_000,
    elapsedMs: 60_000,
    asOf: T0 + 1,
    ...over,
  });
  const typed: TranscriptEntry = { kind: 'user', ts: T0, text: '/goal Fix the flaky cache tests', promptIndex: 0 };
  const of = (
    entries: TranscriptEntry[],
    extra: { agentLive?: boolean; interactions?: PermissionInteraction[] } = {},
  ) => activityOf({ entries, runs: [], turnRunning: false, ...extra });

  // Mutation: count the goal in `hasActivity` → a goal paused days ago opens the docked rail.
  it('is its own section, read from the last goal row; it never counts as something to show', () => {
    const running = of([typed, goal()]);
    expect(running.goal).toMatchObject({ state: 'working', anchor: 1, running: true });
    expect(hasActivity(running)).toBe(false);
    const paused = of([typed, goal({ status: 'user_paused' })]);
    expect(paused.goal?.state).toBe('paused');
    expect(hasActivity(paused)).toBe(false);
  });

  it('an open card makes it wait for you; a cold conversation’s active goal is not running', () => {
    const card: PermissionInteraction = {
      id: 'a',
      kind: 'permission',
      toolCallId: 'c',
      openedAt: T0 + 2,
      toolName: 'run_terminal_command',
      access: 'bash',
      title: 'Execute `ls`',
      input: '{}',
      command: 'ls',
      path: null,
      resolvedPath: null,
      options: [],
      flags: [],
    };
    expect(of([typed, goal()], { interactions: [card] }).goal?.state).toBe('waiting');
    expect(of([typed, goal()], { agentLive: false }).goal).toMatchObject({ state: 'cold', running: false });
  });

  it('none once cleared, or once done and a message was typed after it', () => {
    expect(of([typed, goal({ status: 'cleared' })]).goal).toBeNull();
    const done = goal({ status: 'complete', asOf: T0 + 5 });
    expect(of([typed, done]).goal?.state).toBe('done');
    expect(of([typed, done, { kind: 'user', ts: T0 + 9, text: 'next', promptIndex: 1 }]).goal).toBeNull();
    expect(of([]).goal ?? null).toBeNull();
  });
});
