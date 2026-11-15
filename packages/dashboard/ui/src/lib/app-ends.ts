/**
 * Which turn a Control Panel restart stopped. MAIN records every end of a Control Panel life; the next one sends the
 * ends in the host state, and a turn of conversation C gets the line of end E when:
 *
 * - its stored end is `cancelled`, C is in the list the dying panel cancelled, and the end lies between the decision
 *   and the exit (one second of slack each side: the agent stores whole seconds); or
 * - it has no stored end, C is in either list, it started by the exit, and the conversation's next turn (if any)
 *   started after it (same slack): the turn that ran when the panel died, last or not.
 *
 * With no cancelled list (the dying panel's note was lost) a stored `cancelled` end keeps its own words: a Stop the
 * user pressed is never blamed on the app.
 */

import { CHAT, fill } from './chat-strings';
import type { AppEnd } from './chat-types';

const SLACK_MS = 1_000;

export interface AppEndTurn {
  end: { ts: number; stopReason: string } | null;
  /** The time of the turn's prompt or wake; null when it has none. */
  startedAt: number | null;
}

export interface AppEndContext {
  ends: readonly AppEnd[];
  conversationId: string;
  /** The conversation's next turn; null for the last turn. */
  next: { startedAt: number | null } | null;
}

/** The end that stopped `turn`, the earliest one when several could, or null. */
export function appEndFor(turn: AppEndTurn, context: AppEndContext): AppEnd | null {
  const id = context.conversationId;
  let found: AppEnd | null = null;
  for (const end of context.ends) {
    if (turn.end !== null) {
      if (turn.end.stopReason !== 'cancelled' || end.cancelled === null || !end.cancelled.includes(id)) continue;
      if (turn.end.ts < end.decidedAt - SLACK_MS || turn.end.ts > end.exitAt + SLACK_MS) continue;
    } else {
      if (turn.startedAt === null) continue;
      if (!end.convs.includes(id) && !(end.cancelled ?? []).includes(id)) continue;
      if (turn.startedAt > end.exitAt + SLACK_MS) continue;
      const next = context.next;
      if (next !== null && (next.startedAt === null || next.startedAt < end.exitAt - SLACK_MS)) continue;
    }
    if (found === null || end.exitAt < found.exitAt) found = end;
  }
  return found;
}

/** The sentence of an end. */
export function appEndText(end: AppEnd): string {
  const o = CHAT.outcomes;
  switch (end.trigger) {
    case 'no-ping':
      return fill(o['app-no-ping'], { n: String(Math.round((end.silentMs ?? 0) / 1000)) });
    case 'crash':
      return end.code === null ? o['app-crash-no-code'] : fill(o['app-crash'], { code: String(end.code) });
    case 'signal':
      return o['app-signal'];
    case 'user':
      return o['app-quit'];
  }
}
