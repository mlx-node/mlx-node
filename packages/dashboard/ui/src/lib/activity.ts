/**
 * What a conversation has going on now, for the activity rail: the requests that wait for the person, its plan, the
 * work that still runs (subagents, workflow runs, background commands, monitors) and its schedules. Built from the
 * transcript entries and the host's live state.
 */

import type { PlanItem } from '../components/arc-x/plan-list/plan-list';
import { agentLiveOf, agentsNow, runAgents } from './agents';
import { CHAT, fill } from './chat-strings';
import type { AgentLive, HelperProgress, Interaction, LiveActivity, RunView, TaskLive } from './chat-types';
import { formatCount } from './format';
import { goalOf, type GoalView } from './goal';
import { runCounts, stepOf } from './run-view';
import type { TranscriptEntry } from './types';
import { oneLine } from './web-tools';

/** Index of the entry in the conversation's `entries`: the thread draws it under `data-entry-at`. */
export type EntryAnchor = number;

/** One agent on the rail: a Task subagent's own row, or an agent under its run's row. */
export interface AgentRailRow {
  /** The child session of its newest attempt: the id its controls use. */
  id: string;
  /** Child sessions of its attempts, oldest first. */
  attempts: readonly string[];
  title: string;
  startTs: number;
  anchor: EntryAnchor;
  /** Its last live tick, with what it does now; null before the first. */
  progress: HelperProgress | null;
  /** The host's tracker of the running agent holds it: only then Stop (and Message) reach it. */
  live: boolean;
  /** Stop was sent and its end has not come: the row offers neither Stop nor Message. */
  stopping: boolean;
  /** Whether a message can reach it; null for a row the host does not hold. */
  messages: AgentLive['messages'] | null;
  /** One of its requests waits in "Needs you". */
  waits: boolean;
  /** Its transcript opens on the Sessions page: see {@link transcriptOpens}. */
  opens: boolean;
}

/** A workflow run's row: in "Running" while it runs or drains a stop, in "Needs you" while it waits on the person. */
export interface RunRailRow {
  kind: 'workflow';
  id: string;
  title: string;
  /** The run's status on the wire. */
  status: string;
  /** The phase that runs now, or null. */
  phase: string | null;
  /** `Step 2 of 3`: the phase's place among the declared ones; null with none, or one the script did not declare. */
  step: { i: number; n: number } | null;
  phasesDone: number;
  phasesTotal: number;
  /** Its agents that run (the rail lists them under this row, not as rows of their own). */
  agentsRunning: number;
  /** Its agents that ask or run, at most {@link RUN_AGENTS_SHOWN}: the asking ones first, then by start. */
  agents: readonly AgentRailRow[];
  /** All of them, in the same order: the palette lists each. */
  listed: readonly AgentRailRow[];
  /** Its other agents that ask or run: `+12 more`, which jumps to its block. */
  moreAgents: number;
  /** Its agents that failed, and the done and stopped ones: `✕ 1 failed · ✓ 4 done`. */
  failed: number;
  done: number;
  /** The host holds it as live. False for an ended run: it drains a stop, or waits on the person. */
  live: boolean;
  stopping: boolean;
  /** It waits on the person (paused, blocked, out of agents): only a resume goes on with it. */
  waiting: boolean;
  /** A Stop reaches it now: live and not stopping, or waiting and not out of agents (the agent refuses that one). */
  stoppable: boolean;
  /** Why it waits, in the agent's words; null with none. */
  pauseMessage: string | null;
  startTs: number;
  anchor: EntryAnchor;
}

export type ActivityRow =
  | ({ kind: 'subagent'; type: string } & AgentRailRow)
  | RunRailRow
  | {
      kind: 'command' | 'monitor';
      id: string;
      title: string;
      startTs: number;
      anchor: EntryAnchor;
      /** Stop was sent and its end has not come: the row offers no Stop. */
      stopping: boolean;
      /** The host's tracker of the running agent holds the task: only then can a Stop reach it. */
      live: boolean;
      /** A monitor's events in the running agent process. */
      events: number;
      /** The last line of its newest event (raw output, never Markdown); null with none. */
      lastEvent: string | null;
    };

/** A background command or monitor row: the rows a Stop acts on. */
export type TaskRow = Extract<ActivityRow, { kind: 'command' | 'monitor' }>;

/** A subagent row: the rows Stop and Message act on. */
export type HelperRow = Extract<ActivityRow, { kind: 'subagent' }>;

/** How many agents a run's row lists under it. */
export const RUN_AGENTS_SHOWN = 4;

/**
 * Where a schedule is: `opening` while a Delete loads the conversation first, `removing` while the agent deletes it,
 * `paused` while no running agent holds the conversation (it does not fire), `firing` while the run its fire started
 * runs, else `waiting` for its next time.
 */
export type ScheduleState = 'waiting' | 'firing' | 'opening' | 'removing' | 'paused';

export interface ScheduleRow {
  id: string;
  title: string;
  /** The next run time in epoch ms, or null when the agent did not say. */
  nextTs: number | null;
  /** How it repeats, in the agent's words (a cron line or an interval), or null for a one-time run. */
  repeat: string | null;
  anchor: EntryAnchor;
  state: ScheduleState;
}

export interface ActivityPlan {
  items: readonly PlanItem[];
  ts: number;
  anchor: EntryAnchor;
}

export type AskKind = Interaction['kind'];

/**
 * Who asks: the conversation's own agent, a helper it started, a helper of a workflow run, or a workflow run whose
 * helper the request does not name (an MCP server's request while only the run works).
 */
export type AskFrom =
  | { who: 'agent' }
  | { who: 'subagent'; label: string; childId: string }
  | { who: 'workflow'; label: string; run: string; childId: string }
  | { who: 'run'; run: string };

/** One open request in the rail's "Needs you" section. */
export interface AskRow {
  id: string;
  kind: AskKind;
  /** A permission's access; null for a plan, a question or an MCP server's request. */
  access: 'bash' | 'edit' | 'mcp' | 'other' | null;
  /**
   * One line: the command's first line, the file path, the tool call's title, the plan's words, the first question, or
   * "{server} asks for input".
   * Cut at {@link ASK_TITLE_MAX} characters, except a path, which the row cuts in its middle.
   */
  title: string;
  /** A question card: how many questions it asks. */
  questions?: number;
  /** `title` is the file path of an edit. */
  path: boolean;
  from: AskFrom;
  /** When its card opened (host clock, epoch ms). */
  since: number;
  /** Its card is the one the thread shows now. */
  front: boolean;
  /** The card opened in Auto and still asks. */
  autoAsk: boolean;
}

/** The rail's sections, in order. */
export interface Activity {
  needs: readonly AskRow[];
  /** The runs that wait on the person, by start: "Needs you" lists them after the requests. */
  needsRuns: readonly RunRailRow[];
  /** Work: the conversation's goal; absent or null with none. Never counts in {@link hasActivity}. */
  goal?: GoalView | null;
  plan: ActivityPlan | null;
  running: readonly ActivityRow[];
  scheduled: readonly ScheduleRow[];
}

export interface ActivityInput {
  entries: readonly TranscriptEntry[];
  /** Work: the open requests of the agent, oldest first. */
  interactions?: readonly Interaction[];
  /** The request whose card the thread shows; null or unknown: the oldest. */
  frontId?: string | null;
  /** The host's workflow runs: the live ones and the newest ended ones. */
  runs: readonly RunView[];
  /** Work: the host's live facts (helper progress, stopping tasks); none when absent. */
  live?: LiveActivity;
  /** A turn of this conversation runs now. */
  turnRunning: boolean;
  /** The clock a just-ended agent's state is read at (the end state rule of `agents.ts`). Default: now. */
  now?: number;
  /**
   * The conversation is loaded in a running agent process. False: its stored subagents, commands and monitors are not
   * live (they died with the process, which may have been killed before it wrote their end), and its schedules are
   * paused. Default true.
   */
  agentLive?: boolean;
}

export const ASK_TITLE_MAX = 80;

/** No tick for this long reads "Quiet for …": the agent sends one at least every 8 s, so two were missed. */
export const PROGRESS_QUIET_MS = 20_000;
/** A helper's context this full or more is drawn in warning ink. */
export const CONTEXT_HIGH_PCT = 80;

/** How long the helper has sent no tick at `now`, when that passed {@link PROGRESS_QUIET_MS}; null otherwise. */
export function quietFor(progress: HelperProgress, now: number): number | null {
  const quiet = now - progress.at;
  return quiet >= PROGRESS_QUIET_MS ? quiet : null;
}

/** The helper's last tick by its child session id; null with none. */
export function progressOf(live: LiveActivity | undefined, childId: string): HelperProgress | null {
  return live?.helpers.find((helper) => helper.childId === childId) ?? null;
}

/** The Sessions page of a subagent's own session. */
export function transcriptPath(childId: string): string {
  return `/sessions/${encodeURIComponent(childId)}`;
}

/**
 * A subagent's transcript can open: its folder was found when the stored transcript was read (`exists`), it runs and
 * ticked, or it ended after a turn or a tool call (its session was made at its start in both cases).
 */
export function transcriptOpens(
  exists: boolean | undefined,
  progress: HelperProgress | null,
  ended?: { turns?: number; toolCalls?: number },
): boolean {
  if (exists === true || progress !== null) return true;
  return ended !== undefined && ((ended.turns ?? 0) > 0 || (ended.toolCalls ?? 0) > 0);
}

/** "1 event", "12 events". */
export function eventsWords(n: number): string {
  return n === 1 ? CHAT.activity.eventsOne : fill(CHAT.activity.events, { n });
}

/** The live facts of a command or monitor the running agent holds; null for one it does not. */
export function taskLiveOf(live: LiveActivity | undefined, taskId: string): TaskLive | null {
  return live?.tasks.find((task) => task.id === taskId) ?? null;
}

/** The last line of `text` that holds more than blanks; '' with none. */
export function lastLine(text: string): string {
  let end = text.length;
  while (end > 0) {
    let start = end;
    while (start > 0 && text[start - 1] !== '\n' && text[start - 1] !== '\r') start -= 1;
    const line = text.slice(start, end);
    if (line.trim() !== '') return line;
    end = start - 1;
  }
  return '';
}

type WakeEntry = Extract<TranscriptEntry, { kind: 'wake' }>;
export type WakeTask = NonNullable<WakeEntry['task']>;

const NO_REPORTS: ReadonlyMap<number, readonly WakeTask[]> = new Map();

/**
 * The host's wake reports by wake `ts`: the monitors whose events came before that wake, each named as its tool row
 * names it, else (no such row in the thread, or a row with no words) by the description its event named.
 */
export function wakeReportsOf(
  live: LiveActivity | undefined,
  entries: readonly TranscriptEntry[],
): ReadonlyMap<number, readonly WakeTask[]> {
  const reports = live?.reports ?? [];
  if (reports.length === 0) return NO_REPORTS;
  const labels = new Map<string, string>();
  for (const report of reports) for (const monitor of report.monitors) labels.set(monitor.id, '');
  for (const entry of entries) {
    if (entry.kind === 'tool' && entry.task !== undefined && labels.has(entry.task.id))
      labels.set(entry.task.id, entry.task.label);
  }
  const byTs = new Map<number, readonly WakeTask[]>();
  for (const report of reports) {
    if (report.monitors.length === 0) continue;
    byTs.set(
      report.wakeTs,
      report.monitors.map(({ id, description }) => {
        const label = labels.get(id) ?? '';
        return { id, kind: 'monitor' as const, label: label.trim() === '' ? description : label };
      }),
    );
  }
  return byTs;
}

/**
 * The monitors whose events woke the agent, for a wake row the stored lines name no cause of (`notifications` with no
 * ended task): the report filed under the row's `ts`. Null for any other row.
 */
export function wakeReportOf(
  entry: WakeEntry,
  reports: ReadonlyMap<number, readonly WakeTask[]>,
): readonly WakeTask[] | null {
  if (entry.cause !== 'notifications' || (entry.tasks !== undefined && entry.tasks.length > 0)) return null;
  return reports.get(entry.ts) ?? null;
}

/** The first line of `text`, cut at {@link ASK_TITLE_MAX} characters. */
function firstLine(text: string): string {
  let line = '';
  for (const ch of text) {
    if (ch === '\n' || ch === '\r') break;
    line += ch;
  }
  return cutTitle(line);
}

function cutTitle(text: string): string {
  let count = 0;
  let kept = '';
  for (const ch of text) {
    count += 1;
    if (count < ASK_TITLE_MAX) kept += ch;
  }
  return count <= ASK_TITLE_MAX ? text : `${kept}…`;
}

function askTitle(item: Interaction): { title: string; path: boolean } {
  if (item.kind === 'plan') return { title: CHAT.needs.plan, path: false };
  if (item.kind === 'question') {
    const first = oneLine(item.questions[0]?.text ?? '').trim();
    return { title: first === '' ? CHAT.needs.question : cutTitle(first), path: false };
  }
  if (item.kind === 'mcp-input') {
    const words = item.mode === 'url' ? CHAT.mcp.linkTitle : CHAT.mcp.formTitle;
    return { title: cutTitle(fill(words, { server: oneLine(item.server) })), path: false };
  }
  if (item.access === 'bash' && item.command !== null && item.command.trim() !== '') {
    return { title: firstLine(item.command), path: false };
  }
  if (item.access === 'edit' && item.path !== null && item.path !== '') return { title: item.path, path: true };
  const title = oneLine(item.title).trim();
  if (title !== '') return { title: cutTitle(title), path: false };
  return { title: item.access === 'mcp' ? CHAT.permTool : CHAT.permOther, path: false };
}

/**
 * The workflow run an open request belongs to, by its name: a run helper's permission request, or an MCP server's
 * request while only the run works. Null for a request of the conversation's own answer or wake. The composer
 * follows only those.
 */
export function askRun(item: Interaction): string | null {
  if (item.kind === 'mcp-input') return item.run ?? null;
  return item.kind === 'permission' ? (item.from?.run ?? null) : null;
}

function askFrom(item: Interaction): AskFrom {
  if (item.kind === 'mcp-input' && item.run !== undefined) return { who: 'run', run: item.run };
  const from = item.from;
  if (from === undefined) return { who: 'agent' };
  return from.run === null
    ? { who: 'subagent', label: from.label, childId: from.child }
    : { who: 'workflow', label: from.label, run: from.run, childId: from.child };
}

/** The open requests as rail rows, oldest first; `frontId` names the card the thread shows, else the oldest is. */
export function asksOf(interactions: readonly Interaction[], frontId: string | null): AskRow[] {
  const front = interactions.some((item) => item.id === frontId) ? frontId : (interactions[0]?.id ?? null);
  return interactions.map((item) => ({
    id: item.id,
    kind: item.kind,
    access: item.kind === 'permission' ? item.access : null,
    ...askTitle(item),
    ...(item.kind === 'question' ? { questions: item.questions.length } : {}),
    from: askFrom(item),
    since: item.openedAt,
    front: item.id === front,
    autoAsk: item.kind === 'permission' && item.autoAsk !== undefined,
  }));
}

/** A wait in words for a row's name: seconds under a minute, minutes under an hour, else hours. */
export function waitWords(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 60) return seconds === 1 ? CHAT.needs.second : fill(CHAT.needs.seconds, { n: seconds });
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return minutes === 1 ? CHAT.needs.minute : fill(CHAT.needs.minutes, { n: minutes });
  const hours = Math.floor(minutes / 60);
  return hours === 1 ? CHAT.needs.hour : fill(CHAT.needs.hours, { n: hours });
}

const NONE: Activity = { needs: [], needsRuns: [], plan: null, running: [], scheduled: [] };

/** The item the agent works on: the one in progress, else the first not done, else the last. */
export function currentPlanItem(items: readonly PlanItem[]): PlanItem | null {
  return (
    items.find((item) => item.status === 'in_progress') ??
    items.find((item) => item.status !== 'completed') ??
    items.at(-1) ??
    null
  );
}

/** The latest plan while it matters: it has an item not done, or a turn runs. */
function planOf(entries: readonly TranscriptEntry[], turnRunning: boolean): ActivityPlan | null {
  for (let i = entries.length - 1; i >= 0; i--) {
    const entry = entries[i]!;
    if (entry.kind !== 'plan') continue;
    if (entry.entries.length === 0) return null;
    const open = entry.entries.some((item) => item.status !== 'completed');
    return open || turnRunning ? { items: entry.entries, ts: entry.ts, anchor: i } : null;
  }
  return null;
}

/** An agent's rail row: `entry` its newest attempt's entry at `anchor`. */
function agentRailRow(
  entry: Extract<TranscriptEntry, { kind: 'subagent' }>,
  anchor: EntryAnchor,
  attempts: readonly string[],
  title: string,
  live: LiveActivity | undefined,
  needs: readonly AskRow[],
): AgentRailRow {
  const held = agentLiveOf(live, { attempts });
  const progress = progressOf(live, entry.childId);
  return {
    id: entry.childId,
    attempts,
    title,
    startTs: entry.ts,
    anchor,
    progress,
    live: held !== null,
    stopping: held?.stopping === true,
    messages: held?.messages ?? null,
    waits: needs.some(
      (row) => row.from.who !== 'agent' && row.from.who !== 'run' && attempts.includes(row.from.childId),
    ),
    opens: transcriptOpens(entry.exists, progress),
  };
}

/** The Task subagents that run; an agent of a run the host holds (live or ended) is listed under its run's row. */
function subagentsOf(
  entries: readonly TranscriptEntry[],
  live: LiveActivity | undefined,
  needs: readonly AskRow[],
  runs: readonly RunView[],
): ActivityRow[] {
  const held = new Set(runs.map((run) => run.runId));
  const rows: ActivityRow[] = [];
  entries.forEach((entry, i) => {
    if (entry.kind !== 'subagent' || entry.endTs !== undefined || entry.noSpawn === true) return;
    if (entry.run !== undefined && held.has(entry.run)) return;
    const title = entry.description !== '' ? entry.description : entry.childId;
    rows.push({ kind: 'subagent', type: entry.type, ...agentRailRow(entry, i, [entry.childId], title, live, needs) });
  });
  return rows;
}

/**
 * A Stop reaches the run now: the host holds it live, or it waits on the person, with no stop sent. The agent refuses
 * to stop a run that is out of agents.
 */
export function runStoppable(run: Pick<RunView, 'live' | 'waiting' | 'stopping' | 'status'>): boolean {
  return !run.stopping && (run.live || (run.waiting && run.status !== 'budget_limited'));
}

/**
 * The runs the host holds that the rail shows: a live one, an ended one that still drains a stop (its agents stay
 * under it), and one that waits on the person ({@link RunRailRow.waiting}, for "Needs you"). A failed or stopped run
 * that drained leaves the rail.
 */
function workflowsOf(
  entries: readonly TranscriptEntry[],
  runs: readonly RunView[],
  live: LiveActivity | undefined,
  needs: readonly AskRow[],
  now: number,
): RunRailRow[] {
  const draining = new Set(live?.agents.flatMap((agent) => (agent.run === null ? [] : [agent.run])));
  const shown = new Map(
    runs
      .filter((run) => run.live || run.waiting || run.stopping || draining.has(run.runId))
      .map((run) => [run.runId, run]),
  );
  if (shown.size === 0) return [];
  const agents = runAgents(entries);
  const rows: RunRailRow[] = [];
  entries.forEach((entry, i) => {
    if (entry.kind !== 'workflow') return;
    const run = shown.get(entry.runId);
    if (run === undefined) return;
    const states = agentsNow(agents.get(entry.runId) ?? [], { live: true, now });
    const counts = runCounts(states);
    const asking: AgentRailRow[] = [];
    const running: AgentRailRow[] = [];
    for (const { row, state } of states) {
      const entryAt = row.at.at(-1);
      const last = entryAt === undefined ? undefined : entries[entryAt];
      if (state !== 'running' || last?.kind !== 'subagent') continue;
      if (!run.live && agentLiveOf(live, row) === null) continue;
      const rail = agentRailRow(last, entryAt!, row.attempts, row.label, live, needs);
      (rail.waits ? asking : running).push(rail);
    }
    const listed = [...asking, ...running];
    const step = stepOf(entry);
    rows.push({
      kind: 'workflow',
      id: entry.runId,
      title: entry.name,
      status: run.status,
      phase: entry.currentPhase,
      step: step?.kind === 'numbered' ? { i: step.i, n: step.n } : null,
      phasesDone: entry.phases.filter((phase) => phase.state === 'done').length,
      phasesTotal: entry.phases.length,
      agentsRunning: counts.running,
      agents: listed.slice(0, RUN_AGENTS_SHOWN),
      listed,
      moreAgents: Math.max(0, listed.length - RUN_AGENTS_SHOWN),
      failed: counts.failed,
      done: counts.done,
      live: run.live,
      stopping: run.stopping,
      waiting: run.waiting,
      stoppable: runStoppable(run),
      pauseMessage: entry.pauseMessage !== undefined && entry.pauseMessage !== '' ? entry.pauseMessage : null,
      startTs: entry.ts,
      anchor: i,
    });
  });
  return rows;
}

/**
 * Background commands and monitors with no end in the transcript; `live` says which the host holds and stops, and what
 * each monitor reported.
 */
export function tasksOf(entries: readonly TranscriptEntry[], live?: LiveActivity): ActivityRow[] {
  const held = new Map(live?.tasks.map((task) => [task.id, task]));
  const rows: ActivityRow[] = [];
  entries.forEach((entry, i) => {
    if (entry.kind !== 'tool' || entry.task === undefined || entry.task.endTs !== undefined) return;
    const { id, kind, label } = entry.task;
    const task = held.get(id);
    const newest = task?.recent.at(-1);
    const line = newest === undefined ? '' : lastLine(newest.text);
    rows.push({
      kind,
      id,
      title: label !== '' ? label : entry.title,
      startTs: entry.ts,
      anchor: i,
      stopping: task?.stopping === true,
      live: task !== undefined,
      events: task?.events ?? 0,
      lastEvent: line === '' ? null : line,
    });
  });
  return rows;
}

/**
 * Schedules the agent made and did not delete; `live` says which a Delete is on, which fire's run runs and the next
 * time a fire named (the stored entry keeps the first), `agentLive` whether a running agent holds the conversation.
 */
export function schedulesOf(entries: readonly TranscriptEntry[], live?: LiveActivity, agentLive = true): ScheduleRow[] {
  const held = new Map(live?.schedules.map((schedule) => [schedule.id, schedule]));
  const rows: ScheduleRow[] = [];
  entries.forEach((entry, i) => {
    if (entry.kind !== 'tool' || entry.schedule === undefined || entry.schedule.endTs !== undefined) return;
    const { id, prompt, repeat, nextTs } = entry.schedule;
    const facts = held.get(id);
    const state: ScheduleState =
      facts?.deleting ?? (!agentLive ? 'paused' : facts?.firing != null ? 'firing' : 'waiting');
    rows.push({
      id,
      title: prompt !== '' ? prompt : entry.title,
      nextTs: facts?.nextTs ?? nextTs,
      repeat,
      anchor: i,
      state,
    });
  });
  return rows;
}

export function activityOf(input: ActivityInput): Activity {
  const needs = asksOf(input.interactions ?? [], input.frontId ?? null);
  if (input.entries.length === 0) return needs.length === 0 ? NONE : { ...NONE, needs };
  const dead = input.agentLive === false;
  const subagents = dead ? [] : subagentsOf(input.entries, input.live, needs, input.runs);
  const tasks = dead ? [] : tasksOf(input.entries, input.live);
  const runs = workflowsOf(input.entries, input.runs, input.live, needs, input.now ?? Date.now());
  const running = [...subagents, ...runs.filter((run) => !run.waiting), ...tasks];
  running.sort((a, b) => a.startTs - b.startTs);
  return {
    needs,
    needsRuns: runs.filter((run) => run.waiting),
    goal: goalOf(input.entries, { hot: !dead, waiting: needs.length > 0 }),
    plan: planOf(input.entries, input.turnRunning),
    running,
    scheduled: schedulesOf(input.entries, input.live, !dead),
  };
}

/** The request an agent of the rail waits on (any of its attempts asks); null with none. */
export function askOfAgent(needs: readonly AskRow[], row: Pick<AgentRailRow, 'attempts'>): string | null {
  return (
    needs.find((ask) => ask.from.who !== 'agent' && ask.from.who !== 'run' && row.attempts.includes(ask.from.childId))
      ?.id ?? null
  );
}

/** `Step 2 of 3: Verify`, the step alone when the script did not declare it, `Starting`; null with no steps. */
export function runStepWords(row: Pick<RunRailRow, 'step' | 'phase' | 'phasesTotal'>): string | null {
  if (row.phase === null || row.phase === '') return row.phasesTotal > 0 ? CHAT.run.starting : null;
  return row.step === null ? row.phase : fill(CHAT.run.step, { i: row.step.i, n: row.step.n, phase: row.phase });
}

/**
 * A run row's second line: where it is and how many of its agents run (`Step 2 of 3: Verify · 1 running`), that it
 * stops, or why it waits on the person.
 */
export function runRailDetail(row: RunRailRow): string {
  if (row.stopping) return CHAT.run.status.stopping;
  if (row.waiting) {
    if (row.status === 'budget_limited') return CHAT.run.status.outOfAgents;
    return row.pauseMessage === null
      ? CHAT.run.status.paused
      : fill(CHAT.run.pausedBecause, { message: row.pauseMessage });
  }
  const parts: string[] = [];
  const step = runStepWords(row);
  if (step !== null) parts.push(step);
  if (row.agentsRunning > 0) parts.push(fill(CHAT.run.counts.running, { n: formatCount(row.agentsRunning) }));
  return parts.length === 0 ? CHAT.activity.kinds.workflow : parts.join(' · ');
}

/** One entry of the palette's Agents group: a running agent, or a run the host holds. */
export type AgentPaletteItem =
  | { kind: 'agent'; id: string; label: string; run: string | null; anchor: EntryAnchor }
  | {
      kind: 'run';
      id: string;
      name: string;
      detail: string;
      anchor: EntryAnchor;
      stoppable: boolean;
      /** Its agents that run: its Stop confirm counts them. */
      running: number;
    };

/**
 * The palette's Agents group, in the rail's order: each run (the running ones, then those that wait on the person)
 * followed by every agent of it that asks or runs, and each Task subagent. A run's agent is drawn in its block, so
 * it jumps there.
 */
export function agentPaletteOf(activity: Pick<Activity, 'running' | 'needsRuns'>): AgentPaletteItem[] {
  const out: AgentPaletteItem[] = [];
  const run = (row: RunRailRow): void => {
    out.push({
      kind: 'run',
      id: row.id,
      name: row.title,
      detail: runRailDetail(row),
      anchor: row.anchor,
      stoppable: row.stoppable,
      running: row.agentsRunning,
    });
    for (const agent of row.listed)
      out.push({ kind: 'agent', id: agent.id, label: agent.title, run: row.title, anchor: row.anchor });
  };
  for (const row of activity.running) {
    if (row.kind === 'workflow') run(row);
    else if (row.kind === 'subagent')
      out.push({ kind: 'agent', id: row.id, label: row.title, run: null, anchor: row.anchor });
  }
  for (const row of activity.needsRuns) run(row);
  return out;
}

/** Something to show: the rail opens by itself only then. A goal is not: the strip already shows it. */
export function hasActivity(activity: Activity): boolean {
  return (
    activity.needs.length > 0 ||
    activity.needsRuns.length > 0 ||
    activity.plan !== null ||
    activity.running.length > 0 ||
    activity.scheduled.length > 0
  );
}
