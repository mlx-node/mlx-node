/**
 * The Agent view beside the thread: where it stands at a width, the list it shows (one watch per window, its ops in
 * order), its message field, and the facts of its head.
 */

import { type AgentEnd, type AgentRow, agentRowOf } from './agents';
import { applyEntryOps } from './chat-store';
import { CHAT, fill } from './chat-strings';
import type { AgentLive, AgentWatchAnswer, ConversationEvent, HelperProgress, PermissionMark } from './chat-types';
import { formatCount } from './format';
import type { TranscriptEntry } from './types';

/** From this page width the view takes the docked rail's place: 1296 (the rail docks) + 440 (the view) − 288 (the rail). */
export const AGENT_VIEW_COLUMN_WIDTH = 1448;
/** From this page width the rail is docked, so the view is a drawer over it; under it, a full-height drawer. */
export const AGENT_VIEW_DRAWER_WIDTH = 1296;

/** `column`: in the rail's slot at 440 px. `drawer`: over the docked rail, min(560px, 100%). `full`: a full drawer. */
export type AgentViewPlace = 'column' | 'drawer' | 'full';

/** Where the view stands at a page width; a width of 0 (no layout yet) counts as wide. */
export function agentViewPlace(width: number): AgentViewPlace {
  if (width === 0 || width >= AGENT_VIEW_COLUMN_WIDTH) return 'column';
  return width >= AGENT_VIEW_DRAWER_WIDTH ? 'drawer' : 'full';
}

/** The list one attempt's view shows. */
export interface AgentFeed {
  /** The attempt it shows. */
  childId: string;
  /** The newest watch sent: the answer of an older one is dropped. */
  request: number;
  /** `loading` until the answer of `request`; `failed` when it did not come. */
  status: 'loading' | 'ready' | 'failed';
  entries: TranscriptEntry[];
  /** Its later lines come as `agent-entries` events; false after the stored list came, and while loading. */
  live: boolean;
  /** Child sessions of the attempts of its row, oldest first. */
  attempts: string[];
}

type AgentEntriesEvent = Extract<ConversationEvent, { type: 'agent-entries' }>;

/**
 * A watch went out. A watch of the same attempt again (after a `sync`) keeps its list on screen until the answer; no
 * event applies before it, as the ops that come before it belong to the watch it replaces.
 */
export function feedStart(previous: AgentFeed | null, childId: string, request: number): AgentFeed {
  const same = previous !== null && previous.childId === childId;
  return {
    childId,
    request,
    status: 'loading',
    entries: same ? previous.entries : [],
    live: false,
    attempts: same ? previous.attempts : [childId],
  };
}

/** The answer of watch `request`: its list, and whether its later lines come. */
export function feedAnswer(feed: AgentFeed, request: number, answer: AgentWatchAnswer): AgentFeed {
  if (request !== feed.request || answer.childId !== feed.childId) return feed;
  return { ...feed, status: 'ready', entries: answer.entries, live: answer.live, attempts: answer.attempts };
}

/** Watch `request` failed: the list stays as it was. */
export function feedFailed(feed: AgentFeed, request: number): AgentFeed {
  return request !== feed.request ? feed : { ...feed, status: 'failed', live: false };
}

/**
 * An `agent-entries` event: this window's, of the attempt shown, while its watch is live. The stored list at the
 * agent's end takes the list's place, and nothing more comes for it.
 */
export function feedEvent(feed: AgentFeed, event: AgentEntriesEvent, window: string): AgentFeed {
  if (event.window !== window || event.childId !== feed.childId || !feed.live) return feed;
  if (event.stored !== undefined) return { ...feed, entries: event.stored, live: false };
  if (event.ops.length === 0) return feed;
  return { ...feed, entries: applyEntryOps(feed.entries, event.ops) };
}

/** What stands where a Task subagent's message field goes. */
export type AgentField =
  | { kind: 'field'; notice: string | null }
  | {
      kind: 'note';
      text: string;
    };

/**
 * The field of a running Task subagent the host holds, with no stop sent, whose messages are on (with a notice while
 * it waits on a request). Else one line: a workflow agent takes no message from this agent build, an ended or stopping
 * one cannot get one, and an agent whose messages are off says why.
 */
export function agentField(input: {
  workflow: boolean;
  state: AgentEnd;
  held: Pick<AgentLive, 'stopping' | 'messages'> | null;
  waits: boolean;
}): AgentField {
  const { held } = input;
  if (input.state !== 'running') return { kind: 'note', text: CHAT.agent.ended };
  if (held?.stopping === true) return { kind: 'note', text: CHAT.agent.stoppingNote };
  if (input.workflow) return { kind: 'note', text: CHAT.agent.noMessages };
  if (held === null || held.messages === 'no-address') return { kind: 'note', text: CHAT.activity.messageNoAddress };
  if (held.messages === 'off') return { kind: 'note', text: CHAT.activity.messageOff };
  return { kind: 'field', notice: input.waits ? CHAT.activity.messageWaits : null };
}

/** The agent before (`-1`) or after (`1`) the one `childId` names in its list; null at an end or off the list. */
export function neighbourAgent(list: readonly AgentRow[], childId: string, step: -1 | 1): AgentRow | null {
  const row = agentRowOf(list, childId);
  if (row === null) return null;
  return list[list.indexOf(row) + step] ?? null;
}

/** The calls a mode allowed for the row's attempts and those the user denied with a note. */
export function agentMarkCounts(
  marks: readonly PermissionMark[] | undefined,
  attempts: readonly string[],
): { allowed: number; denied: number } {
  let allowed = 0;
  let denied = 0;
  for (const mark of marks ?? []) {
    if (mark.kind === 'allowed' && mark.child !== undefined && attempts.includes(mark.child)) allowed += 1;
    else if (mark.kind === 'denied' && mark.child !== undefined && attempts.includes(mark.child)) denied += 1;
  }
  return { allowed, denied };
}

/** `2 allowed, 1 denied`, `2 allowed`, `1 denied`; null with none. */
export function markCountWords(counts: { allowed: number; denied: number }): string | null {
  const v = CHAT.agent.view;
  if (counts.allowed > 0 && counts.denied > 0) return fill(v.marks, { a: counts.allowed, d: counts.denied });
  if (counts.allowed > 0) return fill(v.marksAllowed, { a: counts.allowed });
  if (counts.denied > 0) return fill(v.marksDenied, { d: counts.denied });
  return null;
}

/**
 * An attempt's list in its two parts: the lines it copied from the attempt it continues (they keep that attempt's
 * times), then its own. Its prompt is the first message of its own part.
 */
export function splitCopied(entries: readonly TranscriptEntry[]): {
  copied: TranscriptEntry[];
  own: TranscriptEntry[];
} {
  const copied: TranscriptEntry[] = [];
  const own: TranscriptEntry[] = [];
  for (const entry of entries) (entry.copied === true ? copied : own).push(entry);
  return { copied, own };
}

/** `41K of 128K context` from a tick that names the model's window, else `41% of context`. */
export function contextWords(progress: HelperProgress): string {
  if (progress.contextWindow === null || progress.contextWindow <= 0)
    return fill(CHAT.agent.facts.contextPct, { pct: progress.contextPct });
  return fill(CHAT.agent.view.context, {
    used: formatCount(progress.tokens),
    window: formatCount(progress.contextWindow),
  });
}
