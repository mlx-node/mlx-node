/**
 * The conversation's goal as the page shows it: the strip above the composer, the thread row where it began and
 * the rail's Goal section. Read from the reducer's `goal` entries; between the agent's lines the clock runs on the page.
 */

import { CHAT, fill } from './chat-strings';
import { formatElapsed, isAside } from './chat-turns';
import type { GoalAction } from './chat-types';
import type { TranscriptEntry } from './types';
import { visibleText } from './work-tools';

export type GoalEntry = Extract<TranscriptEntry, { kind: 'goal' }>;

/** What the strip says the goal does now. */
export type GoalState =
  | 'planning'
  | 'working'
  | 'checking'
  | 'waiting'
  | 'paused'
  | 'blocked'
  | 'budget'
  | 'done'
  | 'cold';

export interface GoalView {
  entry: GoalEntry;
  /** Index of its row in the entries. */
  anchor: number;
  state: GoalState;
  /** The agent works on it now: the clock counts on, Pause is its only button and the queue waits. */
  running: boolean;
  /** The agent's own words on a pause or a block, whole; null with none. */
  reason: string | null;
  /** The strip's buttons, in order. */
  actions: readonly GoalAction[];
}

/** Paused by the agent itself, with the words `CHAT.goal.why` gives. */
const PAUSED_BY_AGENT = ['back_off_paused', 'no_progress_paused', 'infra_paused'] as const;
type AgentPause = (typeof PAUSED_BY_AGENT)[number];

function pausedByAgent(status: string): status is AgentPause {
  return (PAUSED_BY_AGENT as readonly string[]).includes(status);
}

/** Neither done nor cleared: the agent still holds it, so a rewind may not cut its row. */
export function goalOpen(entry: GoalEntry): boolean {
  return entry.status !== 'complete' && entry.status !== 'cleared';
}

/** The last goal row (the agent holds one goal at a time, so the last is the current one), or null. */
function lastGoal(entries: readonly TranscriptEntry[]): { entry: GoalEntry; at: number } | null {
  for (let at = entries.length - 1; at >= 0; at -= 1) {
    const entry = entries[at]!;
    if (entry.kind === 'goal') return { entry, at };
  }
  return null;
}

/** Index of the current goal's row while it is open, else null: a rewind to a message before it is off. */
export function openGoalAt(entries: readonly TranscriptEntry[]): number | null {
  const last = lastGoal(entries);
  return last !== null && goalOpen(last.entry) ? last.at : null;
}

/** The status of the current goal, or null with none; the page says each change once. */
export function goalStatusOf(entries: readonly TranscriptEntry[]): { goalId: string; status: string } | null {
  const last = lastGoal(entries);
  return last === null ? null : { goalId: last.entry.goalId, status: last.entry.status };
}

export interface GoalFacts {
  /** The conversation is loaded in a running agent process. */
  hot: boolean;
  /** A request of the agent waits for the person. */
  waiting: boolean;
}

/**
 * The goal the strip shows: the current one unless it was cleared; a done goal stays until a message is typed after it
 * ended. An active goal of a conversation whose agent is not running is not running.
 */
export function goalOf(entries: readonly TranscriptEntry[], facts: GoalFacts): GoalView | null {
  const last = lastGoal(entries);
  if (last === null) return null;
  const { entry, at } = last;
  if (entry.status === 'cleared') return null;
  if (entry.status === 'complete') {
    for (let i = at + 1; i < entries.length; i += 1) {
      const later = entries[i]!;
      if (later.kind === 'user' && !isAside(later) && later.ts > entry.asOf) return null;
    }
  }
  const base = { entry, anchor: at, reason: entry.pauseMessage ?? null };
  switch (entry.status) {
    case 'active':
      if (!facts.hot) return { ...base, state: 'cold', running: false, reason: null, actions: ['resume', 'clear'] };
      return { ...base, state: activeState(entry, facts.waiting), running: true, reason: null, actions: ['pause'] };
    case 'blocked':
      return { ...base, state: 'blocked', running: false, actions: ['resume', 'clear'] };
    case 'budget_limited':
      return { ...base, state: 'budget', running: false, actions: ['clear'] };
    case 'complete':
      return { ...base, state: 'done', running: false, actions: ['clear'] };
    default:
      return { ...base, state: 'paused', running: false, actions: ['resume', 'clear'] };
  }
}

function activeState(entry: GoalEntry, waiting: boolean): GoalState {
  if (waiting) return 'waiting';
  if (entry.planning === true || entry.phase === 'planning') return 'planning';
  return entry.verifying === true ? 'checking' : 'working';
}

/** A token count as the strip writes it: `950`, `31k`, `1.2M`. */
export function goalTokens(n: number): string {
  if (n < 1_000) return String(Math.max(0, Math.floor(n)));
  if (n < 1_000_000) return `${String(Math.floor(n / 1_000))}k`;
  return `${(Math.floor(n / 100_000) / 10).toFixed(1)}M`;
}

/** The first line of the agent's words, for one-line places, with its hidden characters shown. */
function firstLine(text: string): string {
  const end = text.indexOf('\n');
  return visibleText((end === -1 ? text : text.slice(0, end)).trim());
}

/** The state's words as the strip says them, with no clock or tokens. */
export function goalStateWords(view: GoalView): string {
  const { entry } = view;
  switch (view.state) {
    case 'planning':
    case 'working':
    case 'checking':
    case 'waiting':
      return CHAT.goal.state[view.state];
    case 'cold':
      return CHAT.goal.state.cold;
    case 'paused':
      return pausedByAgent(entry.status)
        ? fill(CHAT.goal.state.pausedWhy, { reason: CHAT.goal.why[entry.status] })
        : CHAT.goal.state.paused;
    case 'blocked':
      return view.reason === null || firstLine(view.reason) === ''
        ? CHAT.goal.state.blockedBare
        : fill(CHAT.goal.state.blocked, { reason: firstLine(view.reason) });
    case 'budget':
      return fill(CHAT.goal.state.budget, { tokens: goalTokens(entry.tokenBudget ?? entry.tokensUsed) });
    case 'done':
      return fill(CHAT.goal.state.done, {
        time: formatElapsed(entry.elapsedMs),
        tokens: goalTokens(entry.tokensUsed),
      });
  }
}

/** The time the goal has run: counted on from its last line while it runs. */
export function goalElapsed(view: GoalView, now: number): number {
  return view.running ? view.entry.elapsedMs + Math.max(0, now - view.entry.asOf) : view.entry.elapsedMs;
}

/** The strip's clock and tokens, shown only while it runs (planning and waiting have no tokens). */
export function goalMeta(view: GoalView, now: number): { clock: string | null; tokens: string | null } {
  if (!view.running) return { clock: null, tokens: null };
  const clock = formatElapsed(goalElapsed(view, now));
  if (view.state === 'planning' || view.state === 'waiting') return { clock, tokens: null };
  const { tokensUsed, tokenBudget } = view.entry;
  const tokens =
    tokenBudget === undefined
      ? fill(CHAT.goal.tokens, { n: goalTokens(tokensUsed) })
      : fill(CHAT.goal.tokensOf, { n: goalTokens(tokensUsed), max: goalTokens(tokenBudget) });
  return { clock, tokens };
}

/** The thread row's state: what the goal did last, as history (no clock, no "agent not running"). */
export function goalRowState(entry: GoalEntry): string {
  if (entry.status === 'complete') return fill(CHAT.goal.rowDone, { time: formatElapsed(entry.elapsedMs) });
  if (entry.status === 'cleared') return CHAT.goal.rowCleared;
  const view = goalOf([entry], { hot: true, waiting: false });
  return view === null ? CHAT.goal.rowCleared : goalStateWords(view);
}

/** What the page says once when the current goal's state changed, or null. */
export function goalStatusLine(
  before: { goalId: string; status: string } | null,
  entries: readonly TranscriptEntry[],
): string | null {
  const last = lastGoal(entries);
  if (before === null || last === null || last.entry.goalId !== before.goalId) return null;
  const { status, pauseMessage } = last.entry;
  if (status === before.status) return null;
  if (status === 'complete') return CHAT.goal.status.done;
  if (status === 'cleared') return CHAT.goal.status.cleared;
  if (status === 'blocked') {
    const reason = pauseMessage === undefined ? '' : firstLine(pauseMessage);
    return reason === '' ? CHAT.goal.status.blockedBare : fill(CHAT.goal.status.blocked, { reason });
  }
  if (status === 'user_paused' || pausedByAgent(status)) return CHAT.goal.status.paused;
  return null;
}
