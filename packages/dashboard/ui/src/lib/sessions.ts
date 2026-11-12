import { useEffect, useState } from 'react';

import { DashboardApiError } from './api';
import { fill } from './chat-strings';
import { prefillRate } from './prefill';
import { toast } from './toast';
import type { TranscriptEntry, TurnMetric } from './types';

/** The words for a session with no title: `id` is the first 8 characters of its id. */
export const UNTITLED_SESSION = 'Session {id}';

/** Display title: the stored title, or "Session 01a1cdb0" when the session has none. Never the whole id. */
export function sessionTitle(session: { id: string; title: string }): string {
  const title = session.title.trim();
  return title === '' ? fill(UNTITLED_SESSION, { id: session.id.slice(0, 8) }) : title;
}

/** Hover text of a printed session title: the whole id when the session has no title. */
export function sessionTitleTip(session: { id: string; title: string }): string {
  return session.title.trim() === '' ? session.id : sessionTitle(session);
}

/** The agent's own message count (summary `num_messages`): it differs from the transcript rows. */
export const AGENT_MESSAGES = 'Agent messages';
export const AGENT_MESSAGES_TIP =
  "The agent's own count. It includes tool results and attempts that have no row in the transcript.";
export const TRANSCRIPT_ROWS_TIP = 'Rows in the transcript';

/** ISO timestamp from `summary.json` → epoch ms; `NaN` for a blank or bad value. */
export function isoMs(iso: string): number {
  return iso === '' ? Number.NaN : Date.parse(iso);
}

/** A 409 from a write guard: the session is live, recently written, or changed underneath. */
export function isConflict(err: unknown): err is DashboardApiError {
  return err instanceof DashboardApiError && err.status === 409;
}

export { errMessage } from './errors';

export const LIVE_HINT = 'In use by a running agent. Rename and delete are off until it stops.';

export function copyResumeCommand(command: string): void {
  void navigator.clipboard.writeText(command).then(
    () => toast.success('Resume command copied', { description: command }),
    () => toast.error('Failed to copy to clipboard'),
  );
}

export interface TurnTokenBars {
  uncached: number;
  cached: number;
  output: number;
}

/**
 * Disjoint stack segments for one turn. grok's `inputTokens` already includes
 * the cache reads, so stacking `cachedTokens` on top would count them twice.
 */
export function turnTokenBars(t: Pick<TurnMetric, 'inputTokens' | 'cachedTokens' | 'outputTokens'>): TurnTokenBars {
  const cached = Math.min(t.cachedTokens, t.inputTokens);
  return { uncached: t.inputTokens - cached, cached, output: t.outputTokens };
}

export interface TurnPrefill {
  tokens: number;
  ms: number;
  calls: number;
  tps: number | null;
}

/** The turn's measured prefill; null when the session file holds none. `tps` is null under the size floor. */
export function turnPrefill(t: Pick<TurnMetric, 'prefillMs' | 'prefillTokens' | 'prefillCalls'>): TurnPrefill | null {
  if (t.prefillMs === null || t.prefillTokens === null) return null;
  const calls = t.prefillCalls ?? 1;
  return { tokens: t.prefillTokens, ms: t.prefillMs, calls, tps: prefillRate(t.prefillTokens, t.prefillMs, calls) };
}

function positive(v: number | null): number | null {
  return v !== null && Number.isFinite(v) && v > 0 ? v : null;
}

/** Token-weighted prefill rate over the rated turns: total tokens / total seconds. */
export function prefillAverage(prefills: Array<TurnPrefill | null>): number | null {
  let tokens = 0;
  let ms = 0;
  for (const p of prefills) {
    if (p === null || p.tps === null) continue;
    tokens += p.tokens;
    ms += p.ms;
  }
  return ms > 0 ? (tokens * 1000) / ms : null;
}

export interface TurnTimeParts {
  /** Time to first token of the turn's first model call. */
  firstTokenMs: number | null;
  decodeMs: number | null;
  otherMs: number | null;
  totalMs: number | null;
}

/**
 * Where a turn's wall time went: time to first token, then decode (output tokens ÷ decode speed), then the rest.
 * Each part is clamped to what is left, so the parts always sum to the total.
 */
export function turnTimeParts(
  t: Pick<TurnMetric, 'elapsedMs' | 'ttftMs' | 'decodeTps' | 'outputTokens'>,
): TurnTimeParts {
  const total = positive(t.elapsedMs);
  if (total === null) return { firstTokenMs: null, decodeMs: null, otherMs: null, totalMs: null };
  const rawFirst = positive(t.ttftMs);
  const first = rawFirst === null ? null : Math.min(rawFirst, total);
  const tps = t.decodeTps;
  const rawDecode =
    tps !== null && Number.isFinite(tps) && tps > 0 && t.outputTokens > 0 ? (t.outputTokens * 1000) / tps : null;
  const decode = rawDecode === null ? null : Math.min(rawDecode, total - (first ?? 0));
  const other = total - (first ?? 0) - (decode ?? 0);
  return { firstTokenMs: first, decodeMs: decode, otherMs: other, totalMs: total };
}

type TurnEnd = Extract<TranscriptEntry, { kind: 'turn-end' }>;

export interface TurnGroup {
  /** 1-based; 0 for entries that precede every turn and belong to none. */
  number: number;
  /** Without the closing turn-end. */
  entries: TranscriptEntry[];
  /** null = still open. */
  end: TurnEnd | null;
  metric: TurnMetric | null;
  startTs: number | null;
  endTs: number | null;
}

/** A turn's prompt: a typed message that is not an interjection, or a wake. The server's rule for a window. */
function isPrompt(entry: TranscriptEntry): boolean {
  return (entry.kind === 'user' && entry.interjection !== true) || entry.kind === 'wake';
}

function earliestTs(entries: readonly TranscriptEntry[], keep: (entry: TranscriptEntry) => boolean): number | null {
  let min: number | null = null;
  for (const entry of entries) if (keep(entry) && entry.ts > 0 && (min === null || entry.ts < min)) min = entry.ts;
  return min;
}

/**
 * Where a turn without a measured start began, in the server's order: its timed prompt, else its earliest
 * interjection or own work, else its earliest spawn line. Spawn lines come last because a woken turn holds the spawns
 * of the idle gap before its wake. A model switch never starts a turn.
 */
function unmeasuredStart(entries: readonly TranscriptEntry[]): number | null {
  const prompt = entries.find((entry) => isPrompt(entry) && entry.ts > 0);
  if (prompt !== undefined) return prompt.ts;
  return (
    earliestTs(
      entries,
      (entry) =>
        entry.kind === 'user' ||
        entry.kind === 'assistant' ||
        entry.kind === 'thinking' ||
        entry.kind === 'tool' ||
        entry.kind === 'plan',
    ) ?? earliestTs(entries, (entry) => entry.kind === 'subagent')
  );
}

function makeGroup(
  number: number,
  entries: TranscriptEntry[],
  end: TurnEnd | null,
  metric: TurnMetric | null,
): TurnGroup {
  const elapsed = metric?.elapsedMs ?? null;
  const startTs =
    end !== null && elapsed !== null && Number.isFinite(elapsed) && elapsed > 0 && end.ts > 0
      ? end.ts - elapsed
      : unmeasuredStart(entries);
  return { number, entries, end, metric, startTs, endTs: end === null ? null : end.ts };
}

/**
 * Turn groups in transcript order. The k-th turn-end closes a group and pairs with `turns[k]` (the server emits one
 * metric per turn-end, in order); extra metrics are ignored. Entries after the last turn-end form an open group.
 * Leading entries before the transcript's first prompt (typed or a wake) that hold more than model switches form
 * group 0.
 */
export function groupTurns(transcript: readonly TranscriptEntry[], turns: readonly TurnMetric[]): TurnGroup[] {
  const groups: TurnGroup[] = [];
  let start = 0;
  const firstPrompt = transcript.findIndex((entry) => entry.kind === 'user' || entry.kind === 'wake');
  const firstEnd = transcript.findIndex((entry) => entry.kind === 'turn-end');
  if (firstPrompt > 0 && (firstEnd === -1 || firstPrompt < firstEnd)) {
    const leading = transcript.slice(0, firstPrompt);
    if (leading.some((entry) => entry.kind !== 'model')) {
      groups.push(makeGroup(0, leading, null, null));
      start = firstPrompt;
    }
  }
  let current: TranscriptEntry[] = [];
  let closed = 0;
  for (let i = start; i < transcript.length; i += 1) {
    const entry = transcript[i]!;
    if (entry.kind !== 'turn-end') {
      current.push(entry);
      continue;
    }
    groups.push(makeGroup(closed + 1, current, entry, turns[closed] ?? null));
    closed += 1;
    current = [];
  }
  if (current.length > 0) groups.push(makeGroup(closed + 1, current, null, null));
  return groups;
}

/** A turn holds work when it has a prompt (typed or a wake), text or a tool call; the server's rule for the open turn. */
export function hasTurnWork(entries: readonly TranscriptEntry[]): boolean {
  return entries.some(
    (entry) =>
      entry.kind === 'user' ||
      entry.kind === 'wake' ||
      entry.kind === 'assistant' ||
      entry.kind === 'thinking' ||
      entry.kind === 'tool',
  );
}

const NO_PROMPT = '(no prompt)';

function firstLine(text: string): string {
  const trimmed = text.trim();
  const newline = trimmed.indexOf('\n');
  return newline === -1 ? trimmed : trimmed.slice(0, newline);
}

/** What woke the agent, in words: the title of a turn that a background event started. */
export function wakeTitle(entry: Extract<TranscriptEntry, { kind: 'wake' }>): string {
  switch (entry.cause) {
    case 'workflow': {
      const run = entry.workflow;
      if (run === undefined || run.name === '') return 'Woken by a background workflow';
      const status = run.status !== '' && run.status !== 'complete' ? ` · ${run.status}` : '';
      return `Woken by workflow ${run.name}${status}`;
    }
    case 'subagent': {
      const description = firstLine(entry.subagent?.description ?? '');
      return description === '' ? 'Woken by a background subagent' : `Woken by subagent: ${description}`;
    }
    case 'task':
      return 'Woken by a background task';
    case 'notifications':
      return 'Woken by background notifications';
    case 'goal':
      return 'Started by the goal loop';
    case 'unknown':
      return 'Woken by a background event';
  }
}

/**
 * A turn's title: the first line of its typed prompt, what woke the agent, or `(no prompt)`. An interjection is the
 * title only when the turn has no other prompt.
 */
export function turnTitle(entries: readonly TranscriptEntry[]): string {
  const prompt = entries.find(isPrompt) ?? entries.find((entry) => entry.kind === 'user');
  if (prompt === undefined) return NO_PROMPT;
  if (prompt.kind === 'wake') return wakeTitle(prompt);
  if (prompt.kind !== 'user') return NO_PROMPT;
  const text = firstLine(prompt.text);
  return text === '' ? NO_PROMPT : text;
}

/** The open turn: the last group when it has no turn end, a number and work. */
export function openTurnGroup(groups: readonly TurnGroup[]): TurnGroup | null {
  const last = groups[groups.length - 1];
  if (last === undefined || last.end !== null || last.number === 0 || !hasTurnWork(last.entries)) return null;
  return last;
}

/** How long a fetched session body counts as "now" for the live looks of the transcript and the agent timeline. */
export const LIVE_FRESH_MS = 10_000;

interface Arrival<T> {
  data: T | undefined;
  arrivedAt: number | null;
  generation: number;
}

/**
 * `arrivedAt` = client time when `data` last changed identity (null while undefined); `fresh` holds from then for
 * `ms`. One timer, re-armed by each new body.
 */
export function useFreshness<T>(data: T | undefined, ms: number): { fresh: boolean; arrivedAt: number | null } {
  const [arrival, setArrival] = useState<Arrival<T>>(() => ({
    data,
    arrivedAt: data === undefined ? null : Date.now(),
    generation: 0,
  }));
  let current = arrival;
  if (arrival.data !== data) {
    current = { data, arrivedAt: data === undefined ? null : Date.now(), generation: arrival.generation + 1 };
    setArrival(current);
  }
  const [expired, setExpired] = useState(-1);
  const { arrivedAt, generation } = current;

  useEffect(() => {
    if (arrivedAt === null) return;
    const handle = setTimeout(() => setExpired(generation), Math.max(0, arrivedAt + ms - Date.now()));
    return () => clearTimeout(handle);
  }, [arrivedAt, generation, ms]);

  return { fresh: arrivedAt !== null && expired !== generation, arrivedAt };
}

/** True until `deadlineMs` (epoch ms; false for null); re-renders once when it passes. One timer, re-armed per deadline. */
export function useFreshUntil(deadlineMs: number | null): boolean {
  const [passed, setPassed] = useState<number | null>(null);
  useEffect(() => {
    if (deadlineMs === null) return;
    const wait = deadlineMs - Date.now();
    if (wait <= 0) return;
    const handle = setTimeout(() => setPassed(deadlineMs), wait);
    return () => clearTimeout(handle);
  }, [deadlineMs]);
  return deadlineMs !== null && passed !== deadlineMs && Date.now() < deadlineMs;
}
