/**
 * A workflow run as its block, its end row and its rail row read it: the status words and lines, the step rule and
 * chips, the counts and the budget, where its report lives, and the resume and new limit rules. One function per
 * rule, so no surface decides alone.
 */

import type { AgentNow } from './agents';
import { CHAT, fill } from './chat-strings';
import type { RunBy, RunView } from './chat-types';
import { formatClock, formatCount, formatDuration } from './format';
import { type PdfRun, pdfRunOf } from './pdf-run';
import type { TranscriptEntry } from './types';
import { type RunPlace, runPlaces } from './work-tools';

export type RunEntry = Extract<TranscriptEntry, { kind: 'workflow' }>;
type WakeEntry = Extract<TranscriptEntry, { kind: 'wake' }>;
type ToolEntry = Extract<TranscriptEntry, { kind: 'tool' }>;
export type RunTone = 'success' | 'warning' | 'danger' | 'accent' | 'neutral';

/** Statuses a script pause leaves: only a resume goes on with the run. */
const PAUSED: ReadonlySet<string> = new Set([
  'user_paused',
  'back_off_paused',
  'no_progress_paused',
  'infra_paused',
  'blocked',
]);

/** The highest agent budget the agent takes. */
export const AGENT_BUDGET_CAP = 1024;
/** The agent cuts `result_summary` in bytes at 16 KiB and ends it with `…`; a report over this many bytes was cut. */
export const REPORT_CUT_BYTES = 16_380;
/** The budget meter shows from this share used. */
export const BUDGET_METER_FROM = 0.75;

export type Step =
  | { kind: 'numbered'; i: number; n: number; title: string }
  | { kind: 'bare'; title: string }
  | { kind: 'starting' };

/**
 * Where the run is: `current_phase` in `phases[]` gives its number; a title the script did not declare has none; no
 * `phase()` yet reads Starting. A run with no steps at all has no step (its glance line shows counts only).
 */
export function stepOf(entry: Pick<RunEntry, 'phases' | 'currentPhase'>): Step | null {
  if (entry.currentPhase === null) return entry.phases.length === 0 ? null : { kind: 'starting' };
  const at = entry.phases.findIndex((phase) => phase.title === entry.currentPhase);
  return at === -1
    ? { kind: 'bare', title: entry.currentPhase }
    : { kind: 'numbered', i: at + 1, n: entry.phases.length, title: entry.currentPhase };
}

export function stepWords(step: Step | null): string | null {
  if (step === null) return null;
  if (step.kind === 'starting') return CHAT.run.starting;
  if (step.kind === 'bare') return step.title;
  return fill(CHAT.run.step, { i: step.i, n: step.n, phase: step.title });
}

/** One of three status words, by where the run stopped: `…in step 2, Verify`, `…in Verify`, or the word alone. */
function where(step: Step | null, words: { bare: string; step: string; phase: string }): string {
  if (step === null || step.kind === 'starting') return words.bare;
  if (step.kind === 'bare') return fill(words.phase, { phase: step.title });
  return fill(words.step, { i: step.i, phase: step.title });
}

/**
 * Who stopped or paused the run: the host's own word while it holds the run (its Stop, a typed `/workflow` command,
 * the main agent's call, an engine change); else the newest control in the transcript since its last resume: the main
 * agent's `workflow` call, or a typed `/workflow pause|stop`. Null when no one is named.
 */
export function runBy(entries: readonly TranscriptEntry[], runId: string, view?: RunView): RunBy | null {
  if (view !== undefined) return view.by;
  for (let i = entries.length - 1; i >= 0; i -= 1) {
    const entry = entries[i]!;
    if (entry.kind === 'tool' && entry.run?.id === runId) {
      if (entry.run.kind === 'resume' || entry.run.kind === 'launch') return null;
      if ((entry.run.kind === 'stop' || entry.run.kind === 'pause') && entry.status === 'completed')
        return 'main-agent';
    }
    if (entry.kind === 'user' && entry.runControl?.runId === runId)
      return entry.runControl.verb === 'resume' ? null : 'you';
  }
  return null;
}

/** The run was stopped before it was interrupted: the drain took too long. */
function stoppedFirst(entry: RunEntry, by: RunBy | null): boolean {
  if (by !== null) return true;
  for (let i = (entry.history?.length ?? 0) - 1; i >= 0; i -= 1) {
    const event = entry.history![i]!.event;
    if (event === 'workflow_cancelled') return true;
    if (event === 'workflow_resumed') return false;
  }
  return false;
}

/** `Stopped by you in step 2, Verify`; an engine change says why; no name when no one is named. */
export function stoppedWords(by: RunBy | null, step: Step | null): string {
  const s = CHAT.run.status;
  if (by === 'you') return where(step, { bare: s.stoppedByYou, step: s.stoppedByYouStep, phase: s.stoppedByYouPhase });
  if (by === 'main-agent')
    return where(step, {
      bare: s.stoppedByMainAgent,
      step: s.stoppedByMainAgentStep,
      phase: s.stoppedByMainAgentPhase,
    });
  const words = where(step, { bare: s.stopped, step: s.stoppedStep, phase: s.stoppedPhase });
  return by === 'engine' ? `${words}. ${CHAT.run.stoppedEngine}` : words;
}

/** `Out of agents.` with the agents it had left: none, some but too few for its next group, or the cap reached. */
export function outOfAgentsWords(budget: number | null, used: number): string {
  const words = CHAT.run.outOfAgents;
  if (budget !== null && budget >= AGENT_BUDGET_CAP) return words.atCap;
  if (budget === null || used >= budget) return fill(words.all, { budget: formatCount(budget ?? used) });
  return fill(words.left, { left: formatCount(budget - used), budget: formatCount(budget) });
}

export interface BudgetWords {
  text: string;
  /** Draw the meter: from {@link BUDGET_METER_FROM} used. */
  meter: boolean;
}

/** `6 of 128 agents used`, `At least…` when the agent could not count them all, `6 agents used` with no budget. */
export function budgetWords(entry: Pick<RunEntry, 'agentBudget' | 'agentsUsed' | 'usageIncomplete'>): BudgetWords {
  const used = formatCount(entry.agentsUsed);
  if (entry.agentBudget === null) return { text: fill(CHAT.run.agentsUsedNoBudget, { used }), meter: false };
  const budget = formatCount(entry.agentBudget);
  const text = fill(entry.usageIncomplete === true ? CHAT.run.agentsUsedAtLeast : CHAT.run.agentsUsed, {
    used,
    budget,
  });
  return { text, meter: entry.agentBudget > 0 && entry.agentsUsed / entry.agentBudget >= BUDGET_METER_FROM };
}

export interface RunCounts {
  running: number;
  /** Done and stopped rows: the Done group. */
  done: number;
  failed: number;
  paused: number;
  total: number;
}

export function runCounts(agents: readonly AgentNow[]): RunCounts {
  const counts: RunCounts = { running: 0, done: 0, failed: 0, paused: 0, total: agents.length };
  for (const { state } of agents) {
    if (state === 'running') counts.running += 1;
    else if (state === 'done' || state === 'stopped') counts.done += 1;
    else if (state === 'failed') counts.failed += 1;
  }
  return counts;
}

/** `1 running · 5 done · 1 failed`: counts of zero are left out. */
export function countWords(counts: RunCounts): string[] {
  const words = CHAT.run.counts;
  const out: string[] = [];
  if (counts.running > 0) out.push(fill(words.running, { n: formatCount(counts.running) }));
  if (counts.paused > 0) out.push(fill(words.paused, { n: formatCount(counts.paused) }));
  if (counts.done > 0) out.push(fill(words.done, { n: formatCount(counts.done) }));
  if (counts.failed > 0) out.push(fill(words.failed, { n: formatCount(counts.failed) }));
  return out;
}

/** An ended run's glance line: `3 steps · 7 agents · 1 retry · 14.9M tokens used`. */
export function summaryWords(entry: RunEntry, agents: readonly AgentNow[]): string[] {
  const s = CHAT.run.summary;
  const out: string[] = [];
  const steps = entry.phases.length;
  if (steps > 0) out.push(steps === 1 ? s.stepsOne : fill(s.steps, { n: formatCount(steps) }));
  out.push(agents.length === 1 ? s.agentsOne : fill(s.agents, { n: formatCount(agents.length) }));
  let retries = 0;
  let tokens = 0;
  for (const { row } of agents) {
    retries += row.attempts.length - 1;
    if (row.tokens?.kind === 'used') tokens += row.tokens.n;
  }
  if (retries > 0) out.push(retries === 1 ? s.retriesOne : fill(s.retries, { n: formatCount(retries) }));
  if (tokens > 0) out.push(fill(s.tokens, { tokens: formatCount(tokens) }));
  return out;
}

export interface AgentGroups {
  /** Asking and failed rows: always open. */
  attention: AgentNow[];
  /** Spawn order, so rows never reorder; rows with no end that nothing runs stay here too. */
  running: AgentNow[];
  /** Done and stopped rows: closed, count only. */
  done: AgentNow[];
}

/** A live run's list by need; `asking` holds the child ids whose request waits. */
export function agentGroups(agents: readonly AgentNow[], asking: ReadonlySet<string>): AgentGroups {
  const groups: AgentGroups = { attention: [], running: [], done: [] };
  for (const agent of agents) {
    if (agent.state === 'failed' || agent.row.attempts.some((id) => asking.has(id))) groups.attention.push(agent);
    else if (agent.state === 'done' || agent.state === 'stopped') groups.done.push(agent);
    else groups.running.push(agent);
  }
  return groups;
}

export type RunStateKind =
  | 'running'
  | 'asking'
  | 'stopping'
  | 'paused'
  | 'out-of-agents'
  | 'done'
  | 'failed'
  | 'stopped'
  | 'ended-early'
  | 'not-finished';

export interface RunState {
  kind: RunStateKind;
  /** The short word: the head's dot, the rail row, what a screen reader hears. */
  word: string;
  /** The head line of the block: the word, or the sentence an ended state says in its place. */
  line: string;
  tone: RunTone;
  pulse: boolean;
  /** Its agents may still run: Stop shows and the list starts open. */
  working: boolean;
  /** The lines under the head, in order: the status line of a live run (the first that applies wins), an end's notes. */
  notes: string[];
  /** The status line's button. Resume comes with the agent control capability. */
  button: 'review' | null;
}

export interface RunContext {
  /** The open conversation's thread, where the host's runs are known. False on the Sessions page. */
  live: boolean;
  /** The host's view of the run (live or ended); none when it does not hold it. */
  view?: RunView;
  /** An agent of the run whose request waits: its label and the request's kind. */
  asking?: { label: string; access?: 'bash' | 'edit' | 'mcp' | 'other' } | null;
  /** Who stopped or paused it ({@link runBy}). */
  by: RunBy | null;
  /** Sessions page: the session is loaded in a running agent process. */
  sessionLive?: boolean;
}

/** A retried agent of the run whose newest attempt runs: the status line says it tries once more. */
function retrying(agents: readonly AgentNow[]): string | null {
  const agent = agents.find(({ row, state }) => state === 'running' && row.attempts.length > 1);
  return agent === undefined ? null : agent.row.label;
}

function pausedNotes(entry: RunEntry, by: RunBy | null): string[] {
  const r = CHAT.run;
  let line: string;
  if (by === 'main-agent') line = `${r.pausedByMainAgentLine} ${r.cancelPaused}`;
  else if (by === 'you') line = `${r.pausedByYouLine} ${r.cancelPaused}`;
  else if (entry.pauseMessage !== undefined && entry.pauseMessage !== '')
    line = fill(r.pausedBecause, { message: entry.pauseMessage });
  else return [r.askToResume];
  return [`${line} ${r.askToResume}`];
}

/**
 * The run's state from its status and what the host says of it: a live `active` run the host no longer holds ended
 * early; a stop reads Stopping until its agents drained; who stopped it is named only when known. The agent's own
 * `budget_limited` text has an em dash, so the app writes its own words.
 */
export function runState(entry: RunEntry, agents: readonly AgentNow[], ctx: RunContext): RunState {
  const r = CHAT.run;
  const s = r.status;
  const step = stepOf(entry);
  const state = (
    kind: RunStateKind,
    word: string,
    tone: RunTone,
    more: Partial<Omit<RunState, 'kind' | 'word' | 'tone'>> = {},
  ): RunState => ({ kind, word, line: word, tone, pulse: false, working: false, notes: [], button: null, ...more });
  const view = ctx.view;
  if (view?.stopping === true && (entry.status === 'active' || entry.status === 'cancelled'))
    return state('stopping', s.stopping, 'neutral', { pulse: true, working: true, notes: [r.stoppingNote] });
  switch (entry.status) {
    case 'active': {
      if (!ctx.live) {
        const time = formatClock(entry.updatedTs);
        return ctx.sessionLive === true
          ? state('running', fill(s.runningAsOf, { time }), 'success')
          : state('not-finished', fill(s.notFinished, { time }), 'neutral');
      }
      if (view === undefined) return state('ended-early', s.endedEarly, 'neutral', { line: r.endedEarly });
      const asking = ctx.asking ?? null;
      if (asking !== null)
        return state('asking', s.waitingYou, 'warning', {
          pulse: true,
          working: true,
          notes: [fill(r.asks[asking.access ?? 'other'], { label: asking.label })],
          button: 'review',
        });
      const again = retrying(agents);
      return state('running', s.running, 'success', {
        pulse: true,
        working: true,
        notes: again === null ? [] : [fill(r.retrying, { label: again })],
      });
    }
    case 'budget_limited': {
      const line = outOfAgentsWords(entry.agentBudget, entry.agentsUsed);
      const atCap = entry.agentBudget !== null && entry.agentBudget >= AGENT_BUDGET_CAP;
      return state('out-of-agents', s.outOfAgents, 'warning', { line, notes: atCap ? [] : [r.askToResume] });
    }
    case 'complete': {
      const counts = runCounts(agents);
      return counts.failed > 0
        ? state('done', fill(s.donePartly, { n: formatCount(counts.failed), m: formatCount(counts.total) }), 'warning')
        : state('done', s.done, 'success');
    }
    case 'failed': {
      const message = entry.pauseMessage ?? '';
      return state('failed', where(step, { bare: s.failed, step: s.failedStep, phase: s.failedPhase }), 'danger', {
        notes: message === '' ? [] : [message],
      });
    }
    case 'cancelled':
      return state('stopped', stoppedWords(ctx.by, step), 'neutral', { notes: ctx.by === 'you' ? [r.notTold] : [] });
    case 'interrupted':
      return stoppedFirst(entry, ctx.by)
        ? state('stopped', s.stopped, 'neutral', { line: r.stoppedSlow })
        : state('ended-early', s.endedEarly, 'neutral', { line: r.endedEarly });
    default: {
      if (!PAUSED.has(entry.status)) return state('stopped', entry.status, 'neutral');
      const word = ctx.by === 'you' ? s.pausedByYou : ctx.by === 'main-agent' ? s.pausedByMainAgent : s.paused;
      return state('paused', word, 'warning', { notes: pausedNotes(entry, ctx.by) });
    }
  }
}

export type ChipMark = 'done' | 'active' | 'pending' | 'failed' | 'stopped' | 'paused' | 'unfinished';

export interface StepChip {
  title: string;
  mark: ChipMark;
  /** Its agents that are done, of all its agents. */
  done: number;
  total: number;
  /** The script's words for the step (`state.json` only): the chip's tooltip. */
  detail: string | null;
  /** The chip of agents that name no step. */
  other?: true;
}

const END_MARKS: Readonly<Partial<Record<RunStateKind, ChipMark>>> = {
  done: 'done',
  failed: 'failed',
  stopped: 'stopped',
  stopping: 'stopped',
  paused: 'paused',
  'out-of-agents': 'paused',
  'ended-early': 'unfinished',
  'not-finished': 'unfinished',
};

/**
 * One chip per step: the declared steps, then titles the script entered without declaring them, in order of first
 * sight, then `Other` for agents with no step. A chip is active while one of its agents runs, whatever the wire says;
 * on an ended run the current step takes the run's end mark (the wire keeps it `active`), never "current".
 */
export function stepChips(entry: RunEntry, agents: readonly AgentNow[], state: RunState): StepChip[] {
  if (entry.phases.length === 0 && entry.currentPhase === null) return [];
  const chips: StepChip[] = entry.phases.map((phase) => ({
    title: phase.title,
    mark: phase.state === 'done' ? 'done' : phase.state === 'active' ? 'active' : 'pending',
    done: 0,
    total: 0,
    detail: phase.detail ?? null,
  }));
  const byTitle = new Map(chips.map((chip) => [chip.title, chip]));
  const see = (title: string | null): void => {
    if (title === null || byTitle.has(title)) return;
    const mark: ChipMark =
      title === entry.currentPhase
        ? 'active'
        : entry.phasesEntered.some((step) => step.title === title)
          ? 'done'
          : 'pending';
    const chip: StepChip = { title, mark, done: 0, total: 0, detail: null };
    chips.push(chip);
    byTitle.set(title, chip);
  };
  for (const step of entry.phasesEntered) see(step.title);
  see(entry.currentPhase);
  for (const { row } of agents) see(row.phase);
  const current = entry.currentPhase === null ? undefined : byTitle.get(entry.currentPhase);
  let other: StepChip | null = null;
  for (const agent of agents) {
    let chip = agent.row.phase === null ? null : (byTitle.get(agent.row.phase) ?? null);
    if (chip === null) {
      other ??= { title: CHAT.run.otherStep, mark: 'done', done: 0, total: 0, detail: null, other: true };
      chip = other;
    }
    chip.total += 1;
    if (agent.state === 'done') chip.done += 1;
  }
  if (other !== null) chips.push(other);
  const end = END_MARKS[state.kind];
  if (end !== undefined && current !== undefined) current.mark = end;
  for (const chip of chips) {
    const running = agents.some(
      ({ row, state: agentState }) =>
        agentState === 'running' && (chip.other === true ? row.phase === null : row.phase === chip.title),
    );
    if (running) chip.mark = 'active';
    else if (chip.other === true) chip.mark = chip.done === chip.total ? 'done' : 'pending';
  }
  return chips;
}

/** UTF-8 bytes of `text`, counted without encoding it. */
function utf8Bytes(text: string): number {
  let bytes = 0;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code < 0x80) bytes += 1;
    else if (code < 0x800) bytes += 2;
    else if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
      const next = text.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        bytes += 4;
        i += 1;
      } else bytes += 3;
    } else bytes += 3;
  }
  return bytes;
}

/** The agent cut the report: it cuts in bytes, so a report of fewer characters can still be over the line. */
export function reportCut(report: string): boolean {
  return report.endsWith('…') && utf8Bytes(report) > REPORT_CUT_BYTES;
}

export interface ReportPlace {
  /** Where the report is drawn; null with no report. It never moves and is never drawn twice. */
  home: 'end-row' | 'block' | null;
  /** The wake is due and its row is not here yet: the block says the main agent writes up the result. */
  writingUp: boolean;
}

/**
 * The report's one home: the end row when a wake is due for the run (`wakeDue`, from the host) or a wake row for this
 * revision is here with its stored update; else the block (stopped by a client, goal mode, wake off, a stored
 * session with no wake row).
 */
export function reportPlace(entry: RunEntry, entries: readonly TranscriptEntry[], wakeDue: boolean): ReportPlace {
  const woke = entries.some(
    (wake) =>
      wake.kind === 'wake' &&
      wake.workflow?.runId === entry.runId &&
      wake.workflow.revision === entry.revision &&
      wake.workflow.snapshot !== null,
  );
  const report = entry.report !== undefined && entry.report !== '';
  return { home: !report ? null : wakeDue || woke ? 'end-row' : 'block', writingUp: wakeDue && !woke };
}

/**
 * The newest wake row of the run, when its stored update has the run's status: the block's `The result ↓` link. Null
 * when none is in the loaded transcript, or when the run moved on since (a resume, then a Stop that wakes no one).
 */
export function wakeAt(entries: readonly TranscriptEntry[], run: Pick<RunEntry, 'runId' | 'status'>): number | null {
  for (let i = entries.length - 1; i >= 0; i -= 1) {
    const entry = entries[i]!;
    if (entry.kind === 'wake' && entry.workflow?.runId === run.runId)
      return entry.workflow.snapshot?.status === run.status ? i : null;
  }
  return null;
}

export interface EndRow {
  title: string;
  facts: string[];
  tone: RunTone;
  /** `result_summary` of the update the wake names; never the run's newest (a resume empties it). */
  report: string | null;
  /** The agent cut the report. */
  cut: boolean;
}

/**
 * The end row of a run's wake: everything from the stored update its prompt id names; `name` is the block's own name
 * when it has one (a PDF run's). Null for any other wake.
 */
export function endRowOf(wake: WakeEntry, shownName?: string): EndRow | null {
  const run = wake.workflow;
  if (wake.cause !== 'workflow' || run === undefined) return null;
  const f = CHAT.run.finished;
  const shot = run.snapshot;
  const name = shownName ?? run.name;
  if (shot === null) return { title: fill(f.ended, { name }), facts: [], tone: 'neutral', report: null, cut: false };
  const elapsed = formatDuration(shot.elapsedMs);
  const agents =
    shot.agentsUsed === 1
      ? CHAT.run.summary.agentsOne
      : fill(CHAT.run.summary.agents, { n: formatCount(shot.agentsUsed) });
  const failed = fill(CHAT.run.summary.failed, { n: formatCount(shot.agentsFailed), m: formatCount(shot.agentsUsed) });
  const report = shot.report !== undefined && shot.report !== '' ? shot.report : null;
  const cut = report !== null && reportCut(report);
  switch (shot.status) {
    case 'complete':
      return shot.agentsFailed > 0
        ? { title: fill(f.complete, { name }), facts: [failed, elapsed], tone: 'warning', report, cut }
        : { title: fill(f.complete, { name }), facts: [elapsed, agents], tone: 'success', report, cut };
    case 'failed':
      return { title: fill(f.failed, { name }), facts: [elapsed], tone: 'danger', report, cut };
    case 'budget_limited':
      return { title: fill(f.outOfAgents, { name }), facts: [elapsed, agents], tone: 'warning', report, cut };
    case 'cancelled':
    case 'interrupted':
      return { title: fill(f.stopped, { name }), facts: [elapsed], tone: 'neutral', report, cut };
    default:
      return { title: fill(f.ended, { name }), facts: [elapsed], tone: 'neutral', report, cut };
  }
}

export interface ResumeInput {
  status: string;
  /** `workflow_updated.failure` on `failed`: a script failure fails again at the same place. */
  failure?: 'script' | 'host';
  agentBudget: number | null;
  agentsUsed: number;
  /** The host's view; none when this agent process does not hold the run. */
  view?: RunView;
  /** `workflow_updated.hold`: a held run is released, not resumed. */
  hold?: 'holding' | 'held';
  /** `initialize._meta["x.ai/agentControl"] === true`. */
  agentControl: boolean;
}

export type ResumeOffer =
  | { offered: true; route: 'resume' | 'release' }
  /** `why`: the words shown in place of the button; null when nothing is shown. */
  | { offered: false; why: string | null };

const NOT_OFFERED: ResumeOffer = { offered: false, why: null };

/**
 * Resume shows for a script pause, `blocked`, `cancelled`, a host failure and an out-of-agents run that can take a new
 * limit, only while the host holds the run from this agent process and is not stopping it; a held run is released.
 * A script failure fails again at the same place and `interrupted` is terminal. Not offered without the capability.
 */
export function resumeOffer(input: ResumeInput): ResumeOffer {
  if (!input.agentControl || input.view?.stopping === true) return NOT_OFFERED;
  if (input.status === 'active') {
    return input.hold === 'held' && input.view !== undefined ? { offered: true, route: 'release' } : NOT_OFFERED;
  }
  const resumable =
    PAUSED.has(input.status) ||
    input.status === 'cancelled' ||
    (input.status === 'failed' && input.failure === 'host') ||
    (input.status === 'budget_limited' && newLimit(input.agentBudget, input.agentsUsed) !== null);
  if (!resumable) return NOT_OFFERED;
  if (input.view === undefined) return { offered: false, why: CHAT.run.cannotResume };
  return { offered: true, route: 'resume' };
}

export interface NewLimit {
  min: number;
  initial: number;
  max: number;
}

/**
 * The new limit field of an out-of-agents run: above the old budget and above what it used, at most 1,024; it starts
 * at twice the budget. None once the budget is 1,024.
 */
export function newLimit(budget: number | null, used: number): NewLimit | null {
  if (budget !== null && budget >= AGENT_BUDGET_CAP) return null;
  const min = Math.max(budget ?? 0, used) + 1;
  if (min > AGENT_BUDGET_CAP) return null;
  return { min, initial: Math.min(Math.max(2 * (budget ?? used), min), AGENT_BUDGET_CAP), max: AGENT_BUDGET_CAP };
}

/** The Stop confirm's text: how many agents stop and what a resume runs again. */
export function stopText(running: number): string {
  const words = CHAT.run.stopText;
  if (running === 0) return words.none;
  return running === 1 ? words.runningOne : fill(words.running, { n: formatCount(running) });
}

export interface LogStep {
  /** The step it entered (`phase_entered`); null for lines before the first step. */
  title: string | null;
  /** The script's words for the step. */
  detail: string | null;
  lines: { text: string; ts: number }[];
}

/**
 * The Log fold: each step the run entered, in order, its detail and the `log` lines that came while it was the step.
 * Logs and details are different texts, so both are kept. Empty without `state.json`'s history.
 */
export function logSteps(entry: Pick<RunEntry, 'history' | 'phases'>): LogStep[] {
  const details = new Map(entry.phases.map((phase) => [phase.title, phase.detail ?? null]));
  const steps: LogStep[] = [];
  let step: LogStep | null = null;
  for (const line of entry.history ?? []) {
    if (line.event === 'phase_entered' && line.detail !== null) {
      step = { title: line.detail, detail: details.get(line.detail) ?? null, lines: [] };
      steps.push(step);
    } else if (line.event === 'log' && line.detail !== null && line.detail !== '') {
      if (step === null) {
        step = { title: null, detail: null, lines: [] };
        steps.push(step);
      }
      step.lines.push({ text: line.detail, ts: line.ts });
    }
  }
  return steps;
}

export function logCount(steps: readonly LogStep[]): number {
  let n = 0;
  for (const step of steps) n += step.lines.length;
  return n;
}

/**
 * The run's one "now" line: the newest `log` line of its current step with its time, else that step's detail (no
 * time). Null before any step, or when the step has neither.
 */
export function nowLine(entry: Pick<RunEntry, 'history' | 'phases' | 'currentPhase'>): {
  text: string;
  ts: number | null;
} | null {
  if (entry.currentPhase === null) return null;
  const steps = logSteps(entry);
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    const step = steps[i]!;
    if (step.title !== entry.currentPhase) continue;
    const newest = step.lines.at(-1);
    if (newest !== undefined) return { text: newest.text, ts: newest.ts };
    break;
  }
  const detail = entry.phases.find((phase) => phase.title === entry.currentPhase)?.detail;
  return detail === undefined || detail === '' ? null : { text: detail, ts: null };
}

/** The agents a chip names: those of its step; `Other` holds those with no step. */
export function chipAgents(agents: readonly AgentNow[], chip: Pick<StepChip, 'title' | 'other'>): AgentNow[] {
  return agents.filter(({ row }) => (chip.other === true ? row.phase === null : row.phase === chip.title));
}

/** The run's clock: `elapsed_ms` as of its last update, plus the time since while it runs live. */
export function runElapsed(entry: Pick<RunEntry, 'elapsedMs' | 'updatedTs'>, running: boolean, now: number): number {
  return running ? entry.elapsedMs + Math.max(0, now - entry.updatedTs) : entry.elapsedMs;
}

export interface RunSnapshot {
  step: string | null;
  kind: RunStateKind;
  failed: ReadonlySet<string>;
}

export function runSnapshot(entry: RunEntry, agents: readonly AgentNow[], state: RunState): RunSnapshot {
  const failed = new Set<string>();
  for (const { row, state: end } of agents) if (end === 'failed') failed.add(row.childId);
  return { step: entry.currentPhase, kind: state.kind, failed };
}

const SAID_ENDS: Readonly<Partial<Record<RunStateKind, keyof typeof CHAT.run.announce>>> = {
  done: 'finished',
  failed: 'failed',
  stopped: 'stopped',
  'out-of-agents': 'outOfAgents',
};

/**
 * What the polite region says when a run moves: a new step, an agent that failed, the run's end. A done agent is not
 * said. `before` null: the block just mounted, so nothing is said.
 */
export function runAnnouncements(
  entry: RunEntry,
  agents: readonly AgentNow[],
  before: RunSnapshot | null,
  after: RunSnapshot,
): string[] {
  if (before === null) return [];
  const a = CHAT.run.announce;
  const out: string[] = [];
  if (after.step !== null && after.step !== before.step && after.kind === 'running') {
    const step = stepOf(entry);
    out.push(
      step?.kind === 'numbered'
        ? fill(a.step, { name: entry.name, i: step.i, n: step.n, phase: step.title })
        : fill(a.stepBare, { name: entry.name, phase: after.step }),
    );
  }
  for (const { row, state } of agents)
    if (state === 'failed' && !before.failed.has(row.childId)) out.push(fill(a.agentFailed, { label: row.label }));
  const end = SAID_ENDS[after.kind];
  if (end !== undefined && after.kind !== before.kind) out.push(fill(a[end], { name: entry.name }));
  return out;
}

export interface RunFacts {
  runId: string;
  /** Its newest update; null for a run only spawn lines name. */
  entry: RunEntry | null;
  /** Where its one block is drawn. */
  place: RunPlace;
  /** Its `workflow` entry's index: data only while the block stands at the launch call. */
  entryAt: number | null;
  /** The main agent's launch call. */
  launch: ToolEntry | null;
  /** Who stopped or paused it, read from the transcript alone ({@link runBy} with no host view). */
  by: RunBy | null;
  /** Its wake rows, oldest first. */
  wakes: WakeEntry[];
  /** The newest of them, by entry index ({@link wakeAt}). */
  resultAt: number | null;
  /** A run of the PDF tool ({@link pdfRunOf}); null for any other run. */
  pdf: PdfRun | null;
}

export interface RunsAt {
  runs: ReadonlyMap<string, RunFacts>;
  /** Entry index to the run whose block is drawn there. */
  blocks: ReadonlyMap<number, string>;
  /** Indexes of the `workflow` entries a launch call's block draws: they are data only. */
  dataOnly: ReadonlyMap<number, string>;
  /** Run id to its name. */
  names: ReadonlyMap<string, string>;
}

function samePdf(a: PdfRun | null, b: PdfRun | null): boolean {
  if (a === null || b === null) return a === b;
  return a.id === b.id && a.name === b.name && a.by === b.by && a.pages.join(',') === b.pages.join(',');
}

function sameFacts(a: RunFacts, b: RunFacts): boolean {
  return (
    samePdf(a.pdf, b.pdf) &&
    a.entry === b.entry &&
    a.place.at === b.place.at &&
    a.place.kind === b.place.kind &&
    a.entryAt === b.entryAt &&
    a.launch === b.launch &&
    a.by === b.by &&
    a.resultAt === b.resultAt &&
    a.wakes.length === b.wakes.length &&
    a.wakes.every((wake, n) => wake === b.wakes[n])
  );
}

function sameMap<K, V>(a: ReadonlyMap<K, V>, b: ReadonlyMap<K, V>): boolean {
  if (a.size !== b.size) return false;
  for (const [key, value] of a) if (b.get(key) !== value) return false;
  return true;
}

/**
 * Each run of a transcript drawn once, where {@link runPlaces} puts it, with what its block and end row read. A run
 * whose facts did not change keeps its object, and the whole answer is `previous` when nothing changed.
 */
export function runsAt(entries: readonly TranscriptEntry[], previous?: RunsAt): RunsAt {
  const places = runPlaces(entries);
  const runs = new Map<string, RunFacts>();
  const blocks = new Map<number, string>();
  const dataOnly = new Map<number, string>();
  const names = new Map<string, string>();
  const entryAt = new Map<string, number>();
  const wakes = new Map<string, WakeEntry[]>();
  entries.forEach((entry, at) => {
    if (entry.kind === 'workflow') {
      entryAt.set(entry.runId, at);
      names.set(entry.runId, entry.name);
    } else if (entry.kind === 'wake' && entry.workflow !== undefined) {
      const list = wakes.get(entry.workflow.runId);
      if (list === undefined) wakes.set(entry.workflow.runId, [entry]);
      else list.push(entry);
    }
  });
  for (const [runId, place] of places) {
    const at = entryAt.get(runId) ?? null;
    const entry = at === null ? null : (entries[at] as RunEntry);
    const launch = place.kind === 'launch' ? (entries[place.at] as ToolEntry) : null;
    const facts: RunFacts = {
      pdf: pdfRunOf(entries, { entry, launch, place }),
      runId,
      entry,
      place,
      entryAt: at,
      launch,
      by: runBy(entries, runId),
      wakes: wakes.get(runId) ?? [],
      resultAt: entry === null ? null : wakeAt(entries, entry),
    };
    const kept = previous?.runs.get(runId);
    runs.set(runId, kept !== undefined && sameFacts(kept, facts) ? kept : facts);
    blocks.set(place.at, runId);
    if (place.kind === 'launch' && at !== null) dataOnly.set(at, runId);
  }
  if (
    previous !== undefined &&
    sameMap(previous.runs, runs) &&
    sameMap(previous.blocks, blocks) &&
    sameMap(previous.dataOnly, dataOnly) &&
    sameMap(previous.names, names)
  )
    return previous;
  return { runs, blocks, dataOnly, names };
}

/** The PDF ids the conversation's runs of the PDF tool read. */
export function startedPdfs(runs: RunsAt): ReadonlySet<string> {
  const ids = new Set<string>();
  for (const facts of runs.runs.values()) if (facts.pdf !== null) ids.add(facts.pdf.id);
  return ids;
}
