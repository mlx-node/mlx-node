/**
 * One agent model for every surface: a Task subagent or an agent of a workflow run, one row per agent with all its
 * attempts. The rows come from the `subagent` entries (the agent stores every spawn and finish line); a run's roster
 * only adds to them, since a roster drops rows past 256 and a stored one is only as new as the last step change.
 */

import { CHAT, fill } from './chat-strings';
import type { AgentLive, LiveActivity } from './chat-types';
import { formatCount } from './format';
import type { TranscriptEntry } from './types';

type SubagentEntry = Extract<TranscriptEntry, { kind: 'subagent' }>;
type RunEntry = Extract<TranscriptEntry, { kind: 'workflow' }>;
export type RosterRow = RunEntry['agents'][number];

export interface AgentRow {
  /** The child session of its newest attempt: the id controls and `?agent=` use. */
  childId: string;
  label: string;
  /** The run that started it; null for a Task subagent. */
  run: string | null;
  /** The roster's `phase`, else the step the run entered last before its first spawn. */
  phase: string | null;
  /** The newest attempt's entry; none for a roster row with no spawn line here. */
  entry?: SubagentEntry;
  /** Child sessions of its attempts, oldest first. */
  attempts: string[];
  /** Indexes of its attempts' entries, oldest first. */
  at: number[];
  /** The roster row of its newest attempt, and the time of the update that carried it. */
  roster: RosterRow | null;
  rosterTs: number | null;
  /** The child its first spawn resumed: `continues {label}`. */
  continues: { childId: string; label: string } | null;
  /** The roster's total use once it ended (`used`), else the newest attempt's context at its end. */
  tokens: { n: number; kind: 'used' | 'context' } | null;
  /** The roster's time once it ended, else the sum of its attempts' times. */
  durationMs: number;
}

/** An agent's open permission request: its row waits for the person. */
export interface AgentAsk {
  /** The request's id: Review brings its card to the front. */
  id: string;
  /** The asking agent's child session. */
  child: string;
  /** The name of its run; null for a Task subagent. */
  run: string | null;
  label: string;
  /** The request's title. */
  title: string;
  access: 'bash' | 'edit' | 'mcp' | 'other';
  /** Host clock when the card opened. */
  openedAt: number;
}

/** A row's state: `unfinished` has no end and nothing runs it any more. */
export type AgentEnd = 'running' | 'done' | 'failed' | 'stopped' | 'unfinished';

export interface AgentClock {
  /** Its lines arrive live (the open thread while the host holds its run): a finish line waits for the roster. */
  live: boolean;
  now: number;
  /** A row with no end runs; else it reads `unfinished`. Default `live`. */
  open?: boolean;
}

export interface AgentNow {
  row: AgentRow;
  state: AgentEnd;
  /** When the state may change with no new line (the finish line's wait); null when it cannot. */
  until: number | null;
}

/** How long a live finish line of a workflow agent waits for its roster or its retry before it reads as ended. */
export const FINISH_WAIT_MS = 1000;

const ROSTER_ENDS: ReadonlyMap<string, AgentEnd> = new Map([
  ['done', 'done'],
  ['failed', 'failed'],
  ['cancelled', 'stopped'],
]);

interface Draft {
  attempts: string[];
  at: number[];
  /** The newest entry of each attempt. */
  entries: Map<string, SubagentEntry>;
  from: string | null;
}

/**
 * B is A's next attempt only when B's spawn names A's newest attempt as `retry_of` (the run's contract retry). Any
 * other spawn is its own row: a script's `agent({resume_from})`, also with A's label, and a resumed Task subagent.
 */
function fold(entries: readonly TranscriptEntry[]): { drafts: Draft[]; owner: Map<string, Draft> } {
  const drafts: Draft[] = [];
  const owner = new Map<string, Draft>();
  entries.forEach((entry, i) => {
    if (entry.kind !== 'subagent') return;
    const known = owner.get(entry.childId);
    if (known !== undefined) {
      known.at.push(i);
      known.entries.set(entry.childId, entry);
      return;
    }
    const retryOf = entry.retryOf;
    const prior = retryOf === undefined ? undefined : owner.get(retryOf);
    if (prior !== undefined && prior.attempts.at(-1) === retryOf) {
      prior.attempts.push(entry.childId);
      prior.at.push(i);
      prior.entries.set(entry.childId, entry);
      owner.set(entry.childId, prior);
      return;
    }
    const draft: Draft = {
      attempts: [entry.childId],
      at: [i],
      entries: new Map([[entry.childId, entry]]),
      from: entry.resumedFrom ?? null,
    };
    drafts.push(draft);
    owner.set(entry.childId, draft);
  });
  return { drafts, owner };
}

function enteredBefore(run: RunEntry | undefined, ts: number): string | null {
  let phase: string | null = null;
  for (const step of run?.phasesEntered ?? []) if (step.ts <= ts) phase = step.title;
  return phase;
}

function rowOf(draft: Draft | null, run: RunEntry | undefined, rostered: readonly RosterRow[]): AgentRow {
  const childId = draft?.attempts.at(-1) ?? rostered[0]!.agentId;
  const entry = draft?.entries.get(childId);
  const first = draft === null ? undefined : draft.entries.get(draft.attempts[0]!);
  const roster = rostered.find((row) => row.agentId === childId) ?? null;
  const named = roster ?? rostered[0] ?? null;
  const description = first?.description ?? '';
  const ended = roster !== null && ROSTER_ENDS.has(roster.state);
  let durationMs = 0;
  for (const id of draft?.attempts ?? []) durationMs += draft!.entries.get(id)!.durationMs ?? 0;
  const tokens: AgentRow['tokens'] = ended
    ? { n: roster.tokens, kind: 'used' }
    : entry?.endTs !== undefined && entry.tokens !== undefined
      ? { n: entry.tokens, kind: 'context' }
      : null;
  return {
    childId,
    label: named?.label ?? (description !== '' ? description : childId),
    run: first?.run ?? run?.runId ?? null,
    phase: named?.phase ?? (first === undefined ? null : enteredBefore(run, first.ts)),
    ...(entry === undefined ? {} : { entry }),
    attempts: draft?.attempts ?? [childId],
    at: draft?.at ?? [],
    roster,
    rosterTs: roster === null ? null : run!.updatedTs,
    continues: null,
    tokens,
    durationMs: ended ? roster.durationMs : durationMs,
  };
}

function sameRoster(a: RosterRow | null, b: RosterRow | null): boolean {
  if (a === b) return true;
  if (a === null || b === null) return false;
  const keys = Object.keys(a) as Array<keyof RosterRow>;
  return keys.length === Object.keys(b).length && keys.every((key) => a[key] === b[key]);
}

function sameRow(a: AgentRow, b: AgentRow): boolean {
  const list = (x: readonly (string | number)[], y: readonly (string | number)[]): boolean =>
    x.length === y.length && x.every((v, n) => v === y[n]);
  return (
    a.childId === b.childId &&
    a.label === b.label &&
    a.run === b.run &&
    a.phase === b.phase &&
    a.entry === b.entry &&
    list(a.attempts, b.attempts) &&
    list(a.at, b.at) &&
    a.rosterTs === b.rosterTs &&
    sameRoster(a.roster, b.roster) &&
    a.continues?.childId === b.continues?.childId &&
    a.continues?.label === b.continues?.label &&
    a.tokens?.n === b.tokens?.n &&
    a.tokens?.kind === b.tokens?.kind &&
    a.durationMs === b.durationMs
  );
}

/**
 * The agents of each run, by run id (every run a `workflow` entry or a spawn line names): one row per agent in spawn
 * order, so rows never reorder, then the roster's rows with no spawn line here, in its order. `previous`, the map this
 * gave before: a run whose rows did not change keeps its list, and the map is kept when no run changed.
 */
export function runAgents(
  entries: readonly TranscriptEntry[],
  previous?: ReadonlyMap<string, readonly AgentRow[]>,
): ReadonlyMap<string, readonly AgentRow[]> {
  const { drafts, owner } = fold(entries);
  const runs = new Map<string, RunEntry>();
  for (const entry of entries) if (entry.kind === 'workflow') runs.set(entry.runId, entry);
  const byRun = new Map<string, Draft[]>();
  for (const draft of drafts) {
    const run = draft.entries.get(draft.attempts[0]!)!.run;
    if (run === undefined) continue;
    const list = byRun.get(run);
    if (list === undefined) byRun.set(run, [draft]);
    else list.push(draft);
  }
  for (const id of runs.keys()) if (!byRun.has(id)) byRun.set(id, []);
  const labelOf = new Map<Draft, string>();
  const out = new Map<string, readonly AgentRow[]>();
  for (const [id, list] of byRun) {
    const run = runs.get(id);
    const rostered = new Map<Draft | string, RosterRow[]>();
    for (const agent of run?.agents ?? []) {
      const draft = owner.get(agent.agentId);
      const key = draft !== undefined && list.includes(draft) ? draft : agent.agentId;
      const rows = rostered.get(key);
      if (rows === undefined) rostered.set(key, [agent]);
      else rows.push(agent);
    }
    const rows: AgentRow[] = list.map((draft) => rowOf(draft, run, rostered.get(draft) ?? []));
    list.forEach((draft, n) => labelOf.set(draft, rows[n]!.label));
    for (const [key, agents] of rostered) if (typeof key === 'string') rows.push(rowOf(null, run, agents));
    out.set(id, rows);
  }
  for (const [id, rows] of out) {
    const list = byRun.get(id)!;
    const linked = rows.map((row, n) => {
      const from = n < list.length ? list[n]!.from : null;
      const source = from === null ? undefined : owner.get(from);
      if (from === null || source === undefined) return row;
      const sourceEntry = source.entries.get(from)!;
      const label = labelOf.get(source) ?? (sourceEntry.description !== '' ? sourceEntry.description : from);
      return { ...row, continues: { childId: from, label } };
    });
    const kept = previous?.get(id);
    out.set(
      id,
      kept !== undefined && kept.length === linked.length && linked.every((row, n) => sameRow(row, kept[n]!))
        ? kept
        : linked,
    );
  }
  if (previous !== undefined && previous.size === out.size && [...out].every(([id, rows]) => previous.get(id) === rows))
    return previous;
  return out;
}

/**
 * The Task subagents (no run), by child id: one row each, since only a workflow's contract retry is another attempt;
 * one the main agent resumed `continues` the one it resumed. `previous` as for {@link runAgents}.
 */
export function subagentRows(
  entries: readonly TranscriptEntry[],
  previous?: ReadonlyMap<string, AgentRow>,
): ReadonlyMap<string, AgentRow> {
  const { drafts, owner } = fold(entries);
  const out = new Map<string, AgentRow>();
  let same = previous !== undefined;
  for (const draft of drafts) {
    const first = draft.entries.get(draft.attempts[0]!)!;
    if (first.run !== undefined) continue;
    let row = rowOf(draft, undefined, []);
    const source = draft.from === null ? undefined : owner.get(draft.from)?.entries.get(draft.from);
    if (draft.from !== null && source !== undefined)
      row = {
        ...row,
        continues: { childId: draft.from, label: source.description !== '' ? source.description : draft.from },
      };
    const kept = previous?.get(row.childId);
    const next = kept !== undefined && sameRow(kept, row) ? kept : row;
    if (next !== kept) same = false;
    out.set(row.childId, next);
  }
  return same && previous !== undefined && previous.size === out.size ? previous : out;
}

/** The row an id names: any of its attempts resolves to it (a retry moves the row to the newest attempt's id). */
export function agentRowOf(rows: readonly AgentRow[], childId: string): AgentRow | null {
  return rows.find((row) => row.attempts.includes(childId)) ?? null;
}

/**
 * The id a control of the row that `childId` names goes to (Open, Message, Stop, `?agent=`): its newest attempt's
 * child session, never the first attempt's (it ended). Null when no row has that attempt.
 */
export function controlTarget(rows: readonly AgentRow[], childId: string): string | null {
  return agentRowOf(rows, childId)?.childId ?? null;
}

/** The host's live facts of a row: the agent it holds under any of the row's attempts; null while it holds none. */
export function agentLiveOf(
  live: LiveActivity | undefined,
  row: { readonly attempts: readonly string[] },
): AgentLive | null {
  return live?.agents.find((agent) => agent.attempts.some((id) => row.attempts.includes(id))) ?? null;
}

/** This window's host sent the Stop that ended the row: only then it reads "Stopped by you". */
export function stoppedByYou(live: LiveActivity | undefined, row: { readonly attempts: readonly string[] }): boolean {
  return live?.stoppedByYou.some((id) => row.attempts.includes(id)) === true;
}

/**
 * The end state rule, the only one any surface uses. A finish line that says cancelled is a stop. A roster as new as
 * the finish line gives its state (a contract failure ends `completed` on the line and `failed` on the roster). A
 * stored session takes the finish line. Live, a `completed` line waits for the roster, a retry (which takes the row)
 * or {@link FINISH_WAIT_MS}: attempt 1 ends 47 ms before its retry spawns, and the row must not read Done, Running,
 * Done.
 */
export function agentState(row: AgentRow, clock: AgentClock): AgentNow {
  const open = clock.open ?? clock.live;
  const entry = row.entry;
  const rosterEnd = row.roster === null ? undefined : ROSTER_ENDS.get(row.roster.state);
  const at = (state: AgentEnd, until: number | null = null): AgentNow => ({ row, state, until });
  if (entry?.endTs === undefined) {
    if (rosterEnd !== undefined) return at(rosterEnd);
    return at(open ? 'running' : 'unfinished');
  }
  if (entry.status === 'cancelled') return at('stopped');
  if (rosterEnd !== undefined && row.rosterTs !== null && row.rosterTs >= entry.endTs) return at(rosterEnd);
  if (entry.status === 'failed') return at('failed');
  if (clock.live && row.run !== null && clock.now < entry.endTs + FINISH_WAIT_MS)
    return at('running', entry.endTs + FINISH_WAIT_MS);
  return at('done');
}

export function agentsNow(rows: readonly AgentRow[], clock: AgentClock): AgentNow[] {
  return rows.map((row) => agentState(row, clock));
}

/** Why a row failed: the finish line says completed while the roster says failed (a contract failure), else its error. */
export function failedReason(row: AgentRow): string | null {
  if (row.entry?.status === 'completed' && row.roster?.state === 'failed') return CHAT.agent.wrongFormat;
  const error = row.entry?.error;
  return error === undefined || error === '' ? null : error;
}

/** The state word; `Stopped by you` only when this window's host sent the stop. */
export function agentStateWords(state: AgentEnd, stoppedByYou: boolean): string {
  const words = CHAT.agent.state;
  switch (state) {
    case 'running':
      return words.running;
    case 'done':
      return words.done;
    case 'failed':
      return words.failed;
    case 'stopped':
      return stoppedByYou ? words.stoppedByYou : words.stopped;
    case 'unfinished':
      return words.unfinished;
  }
}

/** `618K tokens used` from the roster, else `41K context at end`; null with neither. */
export function tokenWords(row: AgentRow): string | null {
  if (row.tokens === null) return null;
  const n = formatCount(row.tokens.n);
  return fill(row.tokens.kind === 'used' ? CHAT.agent.facts.tokensUsed : CHAT.agent.facts.contextAtEnd, { n });
}

/** The sub facts of its attempts: `attempt 2` for a retried row, `continues {label}` for a resumed one. */
export function attemptWords(row: AgentRow): string[] {
  const out: string[] = [];
  if (row.attempts.length > 1) out.push(fill(CHAT.agent.facts.attempt, { n: row.attempts.length }));
  if (row.continues !== null) out.push(fill(CHAT.agent.facts.continues, { label: row.continues.label }));
  return out;
}

/** A row's buttons now; the same on every surface. Pause and a workflow agent's Message wait for the agent build. */
export interface AgentButtons {
  /** Message, a Task subagent's only: null when not drawn; drawn off with `off` as its reason. */
  message: { off: string | null } | null;
  /** Review: the id of the request it waits on, brought to the front. */
  review: string | null;
  stop: boolean;
}

const NO_BUTTONS: AgentButtons = { message: null, review: null, stop: false };

export function anyButton(buttons: AgentButtons): boolean {
  return buttons.message !== null || buttons.review !== null || buttons.stop;
}

/**
 * The buttons of a running row the host holds (`held`) with no stop sent: Message (Task subagents) and Stop, or
 * Review and Stop while it waits on a request. A row that ended, stops, or that the host does not hold has none.
 */
export function agentButtons(input: {
  workflow: boolean;
  state: AgentEnd;
  held: Pick<AgentLive, 'stopping' | 'messages'> | null;
  askId: string | null;
}): AgentButtons {
  const { held } = input;
  if (input.state !== 'running' || held === null || held.stopping) return NO_BUTTONS;
  if (input.askId !== null) return { message: null, review: input.askId, stop: true };
  if (input.workflow) return { message: null, review: null, stop: true };
  const off =
    held.messages === 'on' ? null : held.messages === 'off' ? CHAT.activity.messageOff : CHAT.activity.messageNoAddress;
  return { message: { off }, review: null, stop: true };
}
