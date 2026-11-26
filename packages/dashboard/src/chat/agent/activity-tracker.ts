/**
 * What one conversation's agent process runs in the background now, read from the agent's live lines: its helpers
 * (subagents, workflow agents, schedule runs), its commands and monitors, and its schedules. Host memory only, gone
 * with the process. The transcript stays the source of rows; this only says what runs, so the host holds the process
 * for it, and each helper's last progress tick, what it does now and each monitor's events, which the page adds to
 * its row.
 */

import { rfc3339Ms, stripAnsi, truncateField } from '../../agent-sessions/transcript.js';
import { isRecord, type Json, type SubagentProgress } from '../acp/messages.js';
import type {
  AgentDoing,
  AgentLive,
  HelperProgress,
  LiveActivity,
  ScheduleLive,
  TaskEvent,
  TaskLive,
  WakeReport,
} from '../types.js';

export type TaskKind = 'command' | 'monitor';

/** Where the user's Delete of a schedule is: loading the conversation first, or the agent deleting it. */
export type ScheduleDeleting = NonNullable<ScheduleLive['deleting']>;

/** Events a monitor keeps (the agent stores none; the rail needs only the latest). */
export const TASK_EVENTS_KEPT = 5;
/** An event's text is cut to this many characters, with `…`. */
export const TASK_EVENT_CHARS = 500;
/** Wake reports kept, the newest. */
export const WAKE_REPORTS_KEPT = 20;
/** A tool call's title in what a helper does now is cut to this many characters, with `…`. */
export const DOING_TITLE_CHARS = 120;
/** Ended workflow agents kept, the newest, so a contract retry that spawns after its first attempt ended finds it. */
export const ENDED_AGENTS_KEPT = 256;
/** Agents this host's Stop ended, kept the newest: their rows read "Stopped by you". */
export const STOPPED_KEPT = 256;

/** A wake the agent said it will start: a command or monitor ended, or a helper ended, with `will_wake`. */
export type WakeDue = { kind: 'task'; taskId: string } | { kind: 'subagent'; subagentId: string };

interface HelperLive {
  subagentId: string;
  /** The workflow run the helper works for, or null for the conversation's own subagent. */
  run: string | null;
  /** Its start line's `description`: a contract retry names the same. */
  description: string;
  /** Child sessions of its attempts, oldest first; the last is its own. */
  attempts: string[];
  /** Its last `subagent_progress` tick, with the host time; null before the first. */
  progress: (SubagentProgress & { at: number }) | null;
  /** What its own live lines said last. */
  doing: Doing;
  /** The {@link ActivityTracker.helperMark} it started at. */
  mark: number;
  /** The opaque address its live start line named (`agentAddress`): a message reaches it only by this. Host memory only. */
  address: string | null;
  /** Stop was sent and no end came yet. */
  stopping: boolean;
}

/** A helper's own live lines, as far as what it does now needs them. */
interface Doing {
  /** The kind of its newest model chunk; a tool call that ended reads as thinking (the model takes its result). */
  chunk: 'thinking' | 'writing' | null;
  /** Its open tool calls by call id, in start order, each with its newest title. */
  tools: Map<string, string>;
  compacting: boolean;
}

/** A workflow agent that ended, for the retry that may follow it. */
interface EndedAgent {
  run: string;
  description: string;
  attempts: string[];
  /** A retry took its place already: a later spawn that resumes it is an agent of its own. */
  taken: boolean;
}

function doingOf(d: Doing): AgentDoing | null {
  if (d.compacting) return { kind: 'compacting' };
  let title: string | null = null;
  for (const t of d.tools.values()) title = t;
  if (title !== null) return { kind: 'tool', title };
  return d.chunk === null ? null : { kind: d.chunk };
}

function sameDoing(a: AgentDoing | null, b: AgentDoing | null): boolean {
  if (a === null || b === null) return a === b;
  return a.kind === b.kind && (a.kind !== 'tool' || b.kind !== 'tool' || a.title === b.title);
}

/** A tool call's title as one line, cut at {@link DOING_TITLE_CHARS} with `…`; null when the line names none. */
function doingTitle(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  let line = '';
  for (const ch of value) {
    if (ch === '\n' || ch === '\r') break;
    line += ch;
  }
  const cut = truncateField(stripAnsi(line.trim()), DOING_TITLE_CHARS);
  return cut.truncated ? `${cut.text}…` : cut.text;
}

/** The lines the tracker reads: the same shape the host tees on to the reducer. */
export interface TrackedLine {
  xai: boolean;
  update: Json;
}

function id(value: unknown): string | null {
  return typeof value === 'string' && value !== '' ? value : null;
}

/** The task a tool result started: `BackgroundTaskStarted` is a command, `Monitor` a monitor; null for any other. */
function startedTask(raw: unknown): { id: string; kind: TaskKind } | null {
  if (!isRecord(raw)) return null;
  if (raw.type === 'BackgroundTaskStarted') {
    const task = id(raw.task_id);
    return task === null ? null : { id: task, kind: 'command' };
  }
  if (raw.type === 'Monitor') {
    const task = id(raw.taskId);
    return task === null ? null : { id: task, kind: 'monitor' };
  }
  return null;
}

/** A `will_wake: true` end line (xAI only): the wake it announces; null for any other line. */
export function wakeDueOf(line: TrackedLine): WakeDue | null {
  const u = line.update;
  if (!line.xai || u.will_wake !== true) return null;
  if (u.sessionUpdate === 'task_completed') {
    const task = isRecord(u.task_snapshot) ? id(u.task_snapshot.task_id) : null;
    return task === null ? null : { kind: 'task', taskId: task };
  }
  if (u.sessionUpdate === 'subagent_finished') {
    const subagent = id(u.subagent_id) ?? id(u.child_session_id);
    return subagent === null ? null : { kind: 'subagent', subagentId: subagent };
  }
  return null;
}

export class ActivityTracker {
  /** By child session id. */
  private readonly helpers = new Map<string, HelperLive>();
  /** An earlier attempt's child session id → the child session of the attempt that took its row and still runs. */
  private readonly aliases = new Map<string, string>();
  /** Workflow agents that ended in this process, by child session id, oldest first. */
  private readonly endedAgents = new Map<string, EndedAgent>();
  /** Child sessions whose Stop this host sent and the agent took, oldest first; kept past the process. */
  private readonly stoppedHere = new Set<string>();
  /** Raised each time a helper starts; never reset. */
  private spawns = 0;
  /** Raised each time the agents in {@link live} change: one starts, ends, turns stopping or is stopped here. */
  private agentChanges = 0;
  /** By task id, in start order. */
  private readonly tasks = new Map<
    string,
    { kind: TaskKind; stopping: boolean; events: number; recent: TaskEvent[] }
  >();
  /** Raised each time a task starts, ends or turns stopping: the page's task facts changed. */
  private taskChanges = 0;
  /**
   * By schedule id, in the order they came: the run its last fire started, until that run ends, and the next fire
   * time that fire named, until a created line names its own.
   */
  private readonly schedules = new Map<string, { firing: string | null; nextTs: number | null }>();
  /** Raised each time a schedule comes, goes, fires or its run ends. */
  private scheduleChanges = 0;
  /** Helpers that ended in this process, by child session id and subagent id: a fire read after its run ended. */
  private readonly helpersEnded = new Set<string>();
  /** Tasks that ended in this process: a start line read after the end does not bring one back. */
  private readonly ended = new Set<string>();
  /** The tool calls that started a background task in this process: their output ticks come at any time. */
  private readonly taskCalls = new Set<string>();
  /**
   * Monitors with an event since the last turn end, in first-event order, each with the description its last event
   * named: what the next wake reports.
   */
  private readonly reported = new Map<string, string>();
  /** Oldest first, at most {@link WAKE_REPORTS_KEPT}. */
  private readonly reports: WakeReport[] = [];

  /** Helpers of any kind: engine work. */
  helperCount(): number {
    return this.helpers.size;
  }

  /** The conversation's own subagents, not a workflow run's helpers. */
  ownHelperCount(): number {
    let n = 0;
    for (const helper of this.helpers.values()) if (helper.run === null) n++;
    return n;
  }

  /** A helper of the workflow run still runs. */
  runHelping(run: string): boolean {
    for (const helper of this.helpers.values()) if (helper.run === run) return true;
    return false;
  }

  /** Commands and monitors: work that needs only the process. */
  taskCount(): number {
    return this.tasks.size;
  }

  scheduleCount(): number {
    return this.schedules.size;
  }

  /** A counter that moves whenever the tasks in {@link live} change. */
  tasksVersion(): number {
    return this.taskChanges;
  }

  /** A counter that moves whenever the agents in {@link live} change. */
  agentsVersion(): number {
    return this.agentChanges;
  }

  /**
   * A helper by the child session id of any of its attempts, while it runs: a subagent of the conversation or an agent
   * of a workflow run. `childId` and `subagentId` are its newest attempt's, the one a Stop must reach.
   */
  agent(
    child: string,
  ): { childId: string; subagentId: string; run: string | null; address: string | null; stopping: boolean } | null {
    const at = this.resolve(child);
    const helper = at === null ? undefined : this.helpers.get(at);
    if (at === null || helper === undefined) return null;
    return {
      childId: at,
      subagentId: helper.subagentId,
      run: helper.run,
      address: helper.address,
      stopping: helper.stopping,
    };
  }

  /**
   * The child sessions of the attempts of the row that `child` (any attempt's id) is on, oldest first: of a running
   * helper, or of a workflow agent that ended in this process; `[child]` for any other.
   */
  attempts(child: string): string[] {
    const at = this.resolve(child);
    const helper = at === null ? undefined : this.helpers.get(at);
    if (helper !== undefined) return [...helper.attempts];
    const ended = [...this.endedAgents.values()].findLast((agent) => agent.attempts.includes(child));
    return ended === undefined ? [child] : [...ended.attempts];
  }

  /** Stop was sent (`true`) or failed (`false`) for a helper, by any attempt's id; true when that changed it. */
  setHelperStopping(child: string, stopping: boolean): boolean {
    const at = this.resolve(child);
    const helper = at === null ? undefined : this.helpers.get(at);
    if (helper === undefined || helper.stopping === stopping) return false;
    helper.stopping = stopping;
    this.agentChanges++;
    return true;
  }

  /** The agent took this host's Stop of the helper (any attempt's id): its row reads "Stopped by you". */
  stopTaken(child: string): boolean {
    const at = this.resolve(child) ?? child;
    if (this.stoppedHere.has(at)) return false;
    this.stoppedHere.add(at);
    for (const old of this.stoppedHere) {
      if (this.stoppedHere.size <= STOPPED_KEPT) break;
      this.stoppedHere.delete(old);
    }
    this.agentChanges++;
    return true;
  }

  /**
   * The agent says the helper (any attempt's id) already ended, or knows none by that id: no end line is coming. Only
   * while `subagentId` is still its newest attempt's: a retry that took the row since runs on. A workflow agent's
   * attempts stay for its retry, as at an end line: the reply can come before that line. True when it was live.
   */
  dropHelper(child: string, subagentId: string): boolean {
    const at = this.resolve(child);
    const helper = at === null ? undefined : this.helpers.get(at);
    if (at === null || helper === undefined || helper.subagentId !== subagentId) return false;
    this.removeHelper(at, helper);
    this.keepEnded(at, helper);
    return true;
  }

  /**
   * A line of the helper's own session (`child`, its own child session id, never an alias): what it does now. True
   * when that changed for a helper that ticked (the page shows it beside the tick).
   */
  childLine(child: string, update: Json): boolean {
    const helper = this.helpers.get(child);
    if (helper === undefined) return false;
    const d = helper.doing;
    const before = doingOf(d);
    switch (update.sessionUpdate) {
      case 'agent_thought_chunk':
        d.chunk = 'thinking';
        break;
      case 'agent_message_chunk':
        d.chunk = 'writing';
        break;
      case 'tool_call':
      case 'tool_call_update': {
        const call = id(update.toolCallId);
        if (call === null) return false;
        if (update.status === 'completed' || update.status === 'failed') {
          if (d.tools.delete(call) && d.tools.size === 0) d.chunk = 'thinking';
          break;
        }
        const title = doingTitle(update.title);
        if (title !== null && title !== '') d.tools.set(call, title);
        else if (!d.tools.has(call)) d.tools.set(call, '');
        break;
      }
      case 'auto_compact_started':
        d.compacting = true;
        break;
      case 'auto_compact_completed':
      case 'auto_compact_failed':
      case 'auto_compact_cancelled':
        d.compacting = false;
        d.chunk = 'thinking';
        break;
      case 'turn_completed':
        d.tools.clear();
        d.compacting = false;
        break;
      default:
        return false;
    }
    return helper.progress !== null && !sameDoing(before, doingOf(d));
  }

  /** A counter that moves whenever the schedules in {@link live} change. */
  schedulesVersion(): number {
    return this.scheduleChanges;
  }

  hasTask(task: string): boolean {
    return this.tasks.has(task);
  }

  hasSchedule(schedule: string): boolean {
    return this.schedules.has(schedule);
  }

  /** The agent deleted the schedule, or said it holds none by that id. True when it was live. */
  dropSchedule(schedule: string): boolean {
    if (!this.schedules.delete(schedule)) return false;
    this.scheduleChanges++;
    return true;
  }

  /** Stop was sent (`true`) or failed (`false`) for a live task; true when that changed it. */
  setStopping(task: string, stopping: boolean): boolean {
    const live = this.tasks.get(task);
    if (live === undefined || live.stopping === stopping) return false;
    live.stopping = stopping;
    this.taskChanges++;
    return true;
  }

  /** The agent no longer knows the task (its kill found none): it ended with no line. True when it was live. */
  dropTask(task: string): boolean {
    return this.endTask(task);
  }

  /**
   * A running command's or monitor's output tick: an `in_progress` `tool_call_update` of the call that started it, with
   * no `_meta` (`BashOutputChunk` in the agent, sent whenever the task prints and never stored). The stored lines of
   * that call carry `_meta`.
   */
  isTaskTick(update: Json, meta: Json | undefined): boolean {
    if (meta !== undefined || update.sessionUpdate !== 'tool_call_update' || update.status !== 'in_progress')
      return false;
    const call = id(update.toolCallId);
    return call !== null && this.taskCalls.has(call);
  }

  /** A live line that passed the live filter; true when what runs changed. */
  apply(line: TrackedLine): boolean {
    const u = line.update;
    if (!line.xai) {
      if (u.sessionUpdate !== 'tool_call' && u.sessionUpdate !== 'tool_call_update') return false;
      const task = startedTask(u.rawOutput);
      if (task === null) return false;
      const call = id(u.toolCallId);
      if (call !== null) this.taskCalls.add(call);
      return this.startTask(task.id, task.kind);
    }
    switch (u.sessionUpdate) {
      case 'subagent_spawned': {
        const child = id(u.child_session_id);
        if (child === null || this.helpers.has(child)) return false;
        const run = id(u.workflow_run_id);
        const description = typeof u.description === 'string' ? u.description : '';
        const prior = this.retried(id(u.resumed_from), run, description);
        const attempts = prior === null ? [child] : [...prior.attempts, child];
        for (const old of attempts) if (old !== child) this.aliases.set(old, child);
        this.helpers.set(child, {
          subagentId: id(u.subagent_id) ?? child,
          run,
          description,
          attempts,
          progress: null,
          doing: { chunk: null, tools: new Map(), compacting: false },
          mark: this.spawns++,
          address: id(u.agentAddress),
          stopping: false,
        });
        this.agentChanges++;
        return true;
      }
      case 'subagent_finished': {
        const child = id(u.child_session_id);
        const subagent = id(u.subagent_id);
        for (const ended of [child, subagent]) if (ended !== null) this.helperEnded(ended);
        const helper = child === null ? undefined : this.helpers.get(child);
        if (child === null || helper === undefined) return false;
        this.removeHelper(child, helper);
        if (helper.run !== null) this.keepEnded(child, helper);
        return true;
      }
      case 'task_backgrounded': {
        const call = id(u.tool_call_id);
        if (call !== null) this.taskCalls.add(call);
        const task = id(u.task_id);
        return task !== null && this.startTask(task, id(u.monitor_description) === null ? 'command' : 'monitor');
      }
      case 'task_completed': {
        const task = isRecord(u.task_snapshot) ? id(u.task_snapshot.task_id) : null;
        return task !== null && this.endTask(task);
      }
      case 'background_tasks':
        return this.taskList(u);
      case 'scheduled_task_created':
      case 'scheduled_task_fired': {
        const schedule = id(u.task_id);
        if (schedule === null) return false;
        const known = this.schedules.get(schedule);
        const fired = u.sessionUpdate === 'scheduled_task_fired';
        const run = fired ? id(u.subagent_id) : null;
        const firing = run !== null && !this.helpersEnded.has(run) ? run : null;
        const nextTs = fired ? rfc3339Ms(u.next_fire_at) : null;
        if (known === undefined) {
          this.schedules.set(schedule, { firing, nextTs });
          this.scheduleChanges++;
          return true;
        }
        // A created line (an update re-sends one) is stored with its own next time, newer than a fire's: the entry's wins.
        const nextMoved = fired ? nextTs !== null && known.nextTs !== nextTs : known.nextTs !== null;
        const moved = (run !== null && known.firing !== firing) || nextMoved;
        if (run !== null) known.firing = firing;
        if (nextMoved) known.nextTs = nextTs;
        if (moved) this.scheduleChanges++;
        return false;
      }
      case 'scheduled_task_deleted': {
        // `shutdown` is the process stopping: the schedule comes back with the next load.
        const schedule = id(u.task_id);
        return schedule !== null && u.reason !== 'shutdown' && this.dropSchedule(schedule);
      }
      case 'turn_completed':
        // The model read the events before it in that turn: as a reminder mid-turn, or as the turn that drained them.
        this.reported.clear();
        return false;
      default:
        return false;
    }
  }

  /**
   * A `monitor_event` (`at`: host clock): counted, kept without color codes and cut among the last
   * {@link TASK_EVENTS_KEPT}, and noted for the next wake with `description` (the host shows and cuts it), for a task
   * this process runs; dropped for any other. True when the task's live facts changed.
   */
  event(task: string, text: string, at: number, description = ''): boolean {
    const live = this.tasks.get(task);
    if (live === undefined) return false;
    const cut = truncateField(stripAnsi(text), TASK_EVENT_CHARS);
    live.events += 1;
    live.recent.push(cut.truncated ? { at, text: `${cut.text}…`, cut: true } : { at, text: cut.text });
    if (live.recent.length > TASK_EVENTS_KEPT) live.recent.shift();
    this.reported.set(task, description);
    this.taskChanges++;
    return true;
  }

  /**
   * A wake starts; `wakeTs` is its row's `ts`. The monitors that reported since the last turn end become its report,
   * and the next wake starts from none. True when a report was filed.
   */
  wakeStarted(wakeTs: number): boolean {
    if (this.reported.size === 0) return false;
    this.reports.push({
      wakeTs,
      monitors: [...this.reported].map(([id, description]) => ({ id, description })),
    });
    if (this.reports.length > WAKE_REPORTS_KEPT) this.reports.shift();
    this.reported.clear();
    return true;
  }

  /**
   * A tick (`at`: host clock): kept for a helper this process runs, by its own child session id (a late tick of an
   * earlier attempt is not the retry's); dropped for any other (a nested helper's).
   */
  progress(tick: SubagentProgress, at: number): boolean {
    const helper = this.helpers.get(tick.childId);
    if (helper === undefined) return false;
    helper.progress = { ...tick, at };
    return true;
  }

  /** Helpers with a tick: the ones the page's live facts hold. */
  tickedCount(): number {
    let n = 0;
    for (const helper of this.helpers.values()) if (helper.progress !== null) n++;
    return n;
  }

  /**
   * The live facts for the page: the helpers that ticked, in spawn order, with what each does now; every agent, in
   * spawn order (never its address; `messagesOff`: the process has no message method); the tasks, in start order; the
   * schedules with where the user's Delete of each is (`deleting`; one the process does not hold comes last); the wake
   * reports; and the agents this host's Stop ended.
   */
  live(deleting: ReadonlyMap<string, ScheduleDeleting> = new Map(), messagesOff = false): LiveActivity {
    const helpers: HelperProgress[] = [];
    const agents: AgentLive[] = [];
    for (const [child, helper] of this.helpers) {
      if (helper.progress !== null) helpers.push({ ...helper.progress, now: doingOf(helper.doing) });
      const messages = messagesOff ? 'off' : helper.address === null ? 'no-address' : 'on';
      agents.push({
        childId: child,
        run: helper.run,
        attempts: [...helper.attempts],
        stopping: helper.stopping,
        messages,
      });
    }
    const tasks: TaskLive[] = [];
    for (const [id, task] of this.tasks) {
      tasks.push({
        id,
        kind: task.kind,
        stopping: task.stopping,
        events: task.events,
        recent: task.recent.map((event) => ({ ...event })),
      });
    }
    const schedules: ScheduleLive[] = [];
    for (const [id, schedule] of this.schedules) {
      schedules.push({
        id,
        deleting: deleting.get(id) ?? null,
        firing: schedule.firing,
        ...(schedule.nextTs === null ? {} : { nextTs: schedule.nextTs }),
      });
    }
    for (const [id, phase] of deleting) {
      if (!this.schedules.has(id)) schedules.push({ id, deleting: phase, firing: null });
    }
    const reports = this.reports.map((report) => ({
      wakeTs: report.wakeTs,
      monitors: report.monitors.map((monitor) => ({ ...monitor })),
    }));
    return { helpers, agents, tasks, schedules, reports, stoppedByYou: [...this.stoppedHere] };
  }

  /** What {@link keepListed} takes: read when the request goes, it leaves out every helper that starts later. */
  helperMark(): number {
    return this.spawns;
  }

  /**
   * The idle guard: `listed` holds the child session ids and subagent ids `_x.ai/subagent/list_running` names, and
   * `asked` the {@link helperMark} read when that request went. A helper tracked then that it names under neither id
   * ended without its line, and goes; one that started since is not in the list the agent read. The list never names
   * a workflow run's agents: one of a run in `holding` (live, or draining a stop) stays. True when one went.
   */
  keepListed(listed: ReadonlySet<string>, asked: number, holding: ReadonlySet<string> = new Set()): boolean {
    let changed = false;
    for (const [child, helper] of this.helpers) {
      if (helper.run !== null && holding.has(helper.run)) continue;
      if (helper.mark >= asked || listed.has(child) || listed.has(helper.subagentId)) continue;
      this.removeHelper(child, helper);
      changed = true;
    }
    return changed;
  }

  clear(): void {
    if (this.helpers.size > 0) this.agentChanges++;
    this.helpers.clear();
    this.aliases.clear();
    this.endedAgents.clear();
    if (this.tasks.size > 0) this.taskChanges++;
    this.tasks.clear();
    if (this.schedules.size > 0) this.scheduleChanges++;
    this.schedules.clear();
    this.helpersEnded.clear();
    this.ended.clear();
    this.taskCalls.clear();
    this.reported.clear();
    this.reports.length = 0;
  }

  private removeHelper(child: string, helper: HelperLive): void {
    this.helpers.delete(child);
    for (const old of helper.attempts) if (old !== child) this.aliases.delete(old);
    this.helperEnded(child);
    this.helperEnded(helper.subagentId);
    this.agentChanges++;
  }

  /** The child session of the running attempt that `child` names: its own, or the retry that took its row. */
  private resolve(child: string): string | null {
    if (this.helpers.has(child)) return child;
    return this.aliases.get(child) ?? null;
  }

  /**
   * The ended agent a spawn retries: the same run (set) and description, resumed from it, and no retry took its place
   * yet. A script's `agent({resume_from})` under its own description, or a resumed Task subagent, is an agent of its
   * own.
   */
  private retried(from: string | null, run: string | null, description: string): EndedAgent | null {
    if (from === null || run === null) return null;
    const ended = this.endedAgents.get(from);
    if (ended === undefined || ended.taken || ended.run !== run || ended.description !== description) return null;
    ended.taken = true;
    return ended;
  }

  private keepEnded(child: string, helper: HelperLive): void {
    if (helper.run === null) return;
    this.endedAgents.set(child, {
      run: helper.run,
      description: helper.description,
      attempts: helper.attempts,
      taken: false,
    });
    for (const old of this.endedAgents.keys()) {
      if (this.endedAgents.size <= ENDED_AGENTS_KEPT) break;
      this.endedAgents.delete(old);
    }
  }

  /** A helper ended: a schedule whose fire started it runs nothing now. */
  private helperEnded(run: string): void {
    this.helpersEnded.add(run);
    for (const schedule of this.schedules.values()) {
      if (schedule.firing !== run) continue;
      schedule.firing = null;
      this.scheduleChanges++;
    }
  }

  private startTask(task: string, kind: TaskKind): boolean {
    if (this.ended.has(task)) return false;
    const known = this.tasks.get(task);
    if (known?.kind === kind) return false;
    // `task_backgrounded` reads a monitor with a blank description as a command; the tool result says which it is.
    if (known !== undefined && kind === 'command') return false;
    if (known === undefined) this.tasks.set(task, { kind, stopping: false, events: 0, recent: [] });
    else known.kind = kind;
    this.taskChanges++;
    return known === undefined;
  }

  private endTask(task: string): boolean {
    this.ended.add(task);
    if (!this.tasks.delete(task)) return false;
    this.taskChanges++;
    return true;
  }

  /**
   * `background_tasks`, the agent's whole list of this session's tasks (last wins): a running row is live, an ended
   * row ends its task, and a known task missing from a list that is not cut to size has ended. A running row brings
   * back a task an earlier line ended: the agent reads a list before it sends it, so a list read before a start can
   * come after that start's line and end a live task, and the list the start asked for puts it back.
   */
  private taskList(u: Json): boolean {
    if (!Array.isArray(u.tasks)) return false;
    let changed = false;
    const listed = new Set<string>();
    for (const row of u.tasks) {
      if (!isRecord(row)) continue;
      const task = id(row.task_id);
      if (task === null) continue;
      listed.add(task);
      if (row.status === 'running') {
        this.ended.delete(task);
        if (!this.tasks.has(task) && this.startTask(task, row.kind === 'monitor' ? 'monitor' : 'command'))
          changed = true;
      } else if (row.status === 'completed' || row.status === 'failed') {
        if (this.endTask(task)) changed = true;
      }
    }
    if (u.truncated === true) return changed;
    for (const task of this.tasks.keys()) {
      if (!listed.has(task) && this.endTask(task)) changed = true;
    }
    return changed;
  }
}
