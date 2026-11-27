/**
 * The words of an agent's row on every surface (a Task subagent or a workflow agent): the head holds the state word
 * and three facts at most, the rest goes to a second muted line.
 */

import { quietFor } from './activity';
import { type AgentNow, agentStateWords, attemptWords, failedReason, tokenWords } from './agents';
import { CHAT, fill } from './chat-strings';
import { formatElapsed } from './chat-turns';
import type { AgentDoing, HelperProgress } from './chat-types';
import { formatCount, formatDuration } from './format';
import { openedWords, type ReaderLabel } from './pdf-run';

export function toolCallsWords(n: number): string {
  return n === 1 ? CHAT.subagent.toolCallsOne : fill(CHAT.subagent.toolCalls, { n: formatCount(n) });
}

export function turnsWords(n: number): string {
  return n === 1 ? CHAT.subagent.turnsOne : fill(CHAT.subagent.turns, { n: formatCount(n) });
}

/**
 * What a running agent does now, from the host's reading of its own live lines: `Thinking`, `Writing`, `Running:
 * {title}` for its newest open tool call (a call with no title reads `Running`), `Compacting`; `Starting` before any.
 */
export function nowWords(now: AgentDoing | null): string {
  const words = CHAT.agent.now;
  if (now === null) return words.starting;
  switch (now.kind) {
    case 'thinking':
      return words.thinking;
    case 'writing':
      return words.writing;
    case 'tool':
      return now.title === '' ? CHAT.agent.state.running : fill(words.running, { title: now.title });
    case 'compacting':
      return words.compacting;
  }
}

/** "Quiet for m:ss" once no tick came for 20 s; null before. */
export function quietWords(progress: HelperProgress, now: number): string | null {
  const quiet = quietFor(progress, now);
  return quiet === null ? null : fill(CHAT.activity.progressQuiet, { time: formatElapsed(quiet) });
}

/** The glyph of a row: the state, never colour alone. */
export type RowMark = 'running' | 'quiet' | 'asking' | 'stopping' | 'done' | 'failed' | 'stopped' | 'unfinished';

export interface RowLive {
  /** Its last tick while it runs in the agent process; null with none. */
  progress: HelperProgress | null;
  now: number;
  /** Its request that waits for the person: the request's title and when it opened. */
  ask?: { title: string; since: number } | null;
  /** The host sent its stop and it has not ended. */
  stopping?: boolean;
  /** This window's host sent its stop. */
  stoppedByYou?: boolean;
  /** An ended workflow agent leaves its step out of its facts (a PDF reader's step is the run's first). */
  noStep?: boolean;
}

export interface RowWords {
  mark: RowMark;
  /** The word beside the glyph. */
  word: string;
  tone: 'success' | 'warning' | 'danger' | 'neutral';
  /** At most three. */
  facts: string[];
  /** The second line: its counts and its attempts. */
  sub: string[];
  /** A failed row's reason, under the row in danger ink. */
  reason: string | null;
}

/** A context this full warns in place. */
export const CONTEXT_WARN_PCT = 90;

/** A running row's time: its earlier attempts' times plus the running attempt's. */
function elapsedOf(agent: AgentNow, live: RowLive): string | null {
  const { progress, now } = live;
  const earlier = agent.row.earlier.durationMs;
  if (progress !== null) return formatElapsed(earlier + progress.durationMs + Math.max(0, now - progress.at));
  const entry = agent.row.entry;
  return entry === undefined ? null : formatElapsed(earlier + now - entry.ts);
}

function plus(earlier: number, newest: number | undefined): number | undefined {
  return newest === undefined && earlier === 0 ? undefined : earlier + (newest ?? 0);
}

/** A tick's errors, then its failed tool calls (`1 error`, `2 failed calls`); none at 0. */
export function errorWords(errors: number, failedCalls: number): string[] {
  const a = CHAT.activity;
  const out: string[] = [];
  if (errors > 0) out.push(errors === 1 ? a.progressErrorsOne : fill(a.progressErrors, { n: formatCount(errors) }));
  if (failedCalls > 0)
    out.push(
      failedCalls === 1 ? a.progressFailedCallsOne : fill(a.progressFailedCalls, { n: formatCount(failedCalls) }),
    );
  return out;
}

/** Live, the turn it is on (`turn 2`); ended, how many it took (`3 turns`). */
function countsOf(
  live: boolean,
  turns: number | undefined,
  toolCalls: number | undefined,
  tick?: Pick<HelperProgress, 'errors' | 'failedCalls'>,
): string[] {
  const out: string[] = [];
  if (turns !== undefined && turns > 0)
    out.push(live ? fill(CHAT.activity.progressTurn, { n: turns }) : turnsWords(turns));
  if (toolCalls !== undefined) out.push(toolCallsWords(toolCalls));
  if (tick !== undefined) out.push(...errorWords(tick.errors, tick.failedCalls));
  return out;
}

/**
 * One row's words by its state: a running row says what it does now ({@link nowWords}), its time and how full its
 * context is; an ended workflow agent its step, time and total use; an ended subagent its time and its context at its
 * end; a failed row its reason on its own line. Counts and attempts go to the second line. A retried row's tool calls,
 * time and ended turns take in every attempt; a running row's turn is its running attempt's.
 */
export function rowWords(agent: AgentNow, live: RowLive): RowWords {
  const { row, state } = agent;
  const workflow = row.run !== null;
  const step = workflow && live.noStep !== true ? row.phase : null;
  const time = row.durationMs > 0 ? formatDuration(row.durationMs) : null;
  const entry = row.entry;
  const attempts = attemptWords(row);
  const ended = (mark: RowMark, word: string, tone: RowWords['tone'], facts: Array<string | null>): RowWords => ({
    mark,
    word,
    tone,
    facts: facts.filter((fact): fact is string => fact !== null && fact !== ''),
    sub: [
      ...countsOf(false, plus(row.earlier.turns, entry?.turns), plus(row.earlier.toolCalls, entry?.toolCalls)),
      ...attempts,
    ],
    reason: null,
  });
  switch (state) {
    case 'running': {
      const progress = live.progress;
      const sub = [
        ...(progress === null
          ? []
          : countsOf(true, progress.turns, row.earlier.toolCalls + progress.toolCalls, progress)),
        ...attempts,
      ];
      const elapsed = elapsedOf(agent, live);
      if (live.stopping === true)
        return {
          mark: 'stopping',
          word: CHAT.agent.state.stopping,
          tone: 'neutral',
          facts: elapsed === null ? [] : [elapsed],
          sub,
          reason: null,
        };
      const ask = live.ask ?? null;
      if (ask !== null)
        return {
          mark: 'asking',
          word: CHAT.agent.state.asking,
          tone: 'warning',
          facts: [ask.title, formatElapsed(live.now - ask.since)],
          sub,
          reason: null,
        };
      const context = progress === null ? null : fill(CHAT.agent.facts.contextPct, { pct: progress.contextPct });
      const facts = [elapsed, context].filter((fact): fact is string => fact !== null);
      const quiet = progress === null ? null : quietWords(progress, live.now);
      if (quiet !== null) return { mark: 'quiet', word: quiet, tone: 'warning', facts, sub, reason: null };
      const full = progress !== null && progress.contextPct >= CONTEXT_WARN_PCT;
      return {
        mark: 'running',
        word: nowWords(progress?.now ?? null),
        tone: full ? 'warning' : 'success',
        facts,
        sub,
        reason: null,
      };
    }
    case 'done':
      return ended('done', CHAT.agent.state.done, 'success', [step, time, tokenWords(row)]);
    case 'failed': {
      const words = ended('failed', CHAT.agent.state.failed, 'danger', [step, time]);
      return { ...words, reason: failedReason(row) };
    }
    case 'stopped':
      return ended('stopped', agentStateWords('stopped', live.stoppedByYou === true), 'neutral', [time]);
    case 'unfinished':
      return ended('unfinished', CHAT.agent.state.unfinished, 'neutral', [step]);
  }
}

/** What a reader row of the PDF tool adds to its words. */
export interface ReaderFacts {
  reader: ReaderLabel;
  /** What the engine does for the reader's session now (`readerFact`); null when no request names it. */
  engine: string | null;
  /** The newest page its `read_file` opened while it runs; null with none seen. */
  opened: number | null;
}

/**
 * A reader row's words: while it runs, the engine's words for its session in place of the agent's (then its time), and
 * the page it opened on the second line; a second try says so first; a failed reader says what becomes of its pages.
 */
export function readerWords(agent: AgentNow, live: RowLive, facts: ReaderFacts): RowWords {
  const words = rowWords(agent, { ...live, noStep: true });
  const p = CHAT.pdf;
  const again = facts.reader.again ? [p.readerFacts.again] : [];
  const runs = words.mark === 'running' || words.mark === 'quiet' || words.mark === 'asking';
  const sub = runs && facts.opened !== null ? [...words.sub, openedWords(facts.opened)] : words.sub;
  const capped = (list: string[]): string[] => list.slice(0, 3);
  if (words.mark === 'failed') {
    const reason = facts.reader.again ? p.readerFailedTwice : p.readerFailed;
    return { ...words, facts: capped([...again, ...words.facts]), sub, reason };
  }
  if ((words.mark === 'running' || words.mark === 'quiet') && facts.engine !== null) {
    const elapsed = elapsedOf(agent, live);
    return {
      ...words,
      mark: 'running',
      word: facts.engine,
      tone: 'success',
      facts: capped([...again, ...(elapsed === null ? [] : [elapsed])]),
      sub,
    };
  }
  return { ...words, facts: capped([...again, ...words.facts]), sub };
}
