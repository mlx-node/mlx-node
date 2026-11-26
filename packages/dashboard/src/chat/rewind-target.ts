/**
 * Which prompt of the agent a message of the thread is, and what a rewind to it removes. The agent names a rewind
 * target by prompt index (`_meta.promptIndex` of the stored user chunk); only the disk has it for sure. A message sent
 * in this app run has none in the view: it is paired, in order and by its text, with the stored prompts after the last
 * message that has one. A message whose turn ended before it went to the agent (the host knows it) takes no part. A
 * Send now that came when no answer ran is the agent's own turn (`interject-fallback-`, a `message` wake in the view)
 * and is stored as a typed prompt: that wake takes the stored prompt in its place. A message with no stored pair, or
 * one whose pair depends on the side the pairing starts from (the same text twice, one of them never stored), is no
 * rewind point. The host's own count of prompts and the agent's prompt texts are never used: a wrong index restores
 * the wrong files.
 */

import type { TranscriptEntry } from '../agent-sessions/types.js';
import { REWIND_FILES_MAX, type RewindFile, type RewindFileState } from './types.js';

type UserEntry = Extract<TranscriptEntry, { kind: 'user' }>;

/** A prompt the agent ran: a typed message or the automatic `continue`, never a note or a steer. */
function isPrompt(entry: TranscriptEntry): entry is UserEntry {
  return entry.kind === 'user' && entry.note !== true && entry.interjection !== true;
}

/** A message the user typed: a rewind point when the agent ran it. */
function isTyped(entry: TranscriptEntry): entry is UserEntry {
  return isPrompt(entry) && entry.continued !== true;
}

/** The live row of a Send now the agent ran as its own turn: its stored prompt is a typed one. */
function isSteerTurn(entry: TranscriptEntry): boolean {
  return entry.kind === 'wake' && entry.cause === 'message';
}

export type RewindTarget = { target: number } | { refused: 'not-a-message' | 'not-sent' };

/**
 * The prompt index of the view's entry `at`. `disk`: the conversation as read from disk now; `shown` turns a disk text
 * into the view's form (the view is redacted), so the two compare; `unsent`: the view's prompts that never went to the
 * agent.
 */
export function targetOf(
  view: readonly TranscriptEntry[],
  disk: readonly TranscriptEntry[],
  at: number,
  shown: (text: string) => string = (text) => text,
  unsent: ReadonlySet<number> = new Set(),
): RewindTarget {
  const entry = view[at];
  if (entry === undefined || !isTyped(entry)) return { refused: 'not-a-message' };
  if (unsent.has(at)) return { refused: 'not-sent' };
  const stored: Array<{ index: number; text: string }> = [];
  for (const e of disk) {
    if (isPrompt(e) && e.promptIndex !== null) stored.push({ index: e.promptIndex, text: shown(e.text) });
  }
  if (entry.promptIndex !== null) {
    const index = entry.promptIndex;
    return stored.some((s) => s.index === index && s.text === entry.text) ? { target: index } : { refused: 'not-sent' };
  }
  let from = 0;
  let first = 0;
  for (let i = at - 1; i >= 0; i--) {
    const e = view[i]!;
    if (!isPrompt(e) || e.promptIndex === null) continue;
    const index = e.promptIndex;
    const found = stored.findIndex((s) => s.index === index && s.text === e.text);
    if (found === -1) return { refused: 'not-sent' };
    from = i + 1;
    first = found + 1;
    break;
  }
  const rows: Array<{ at: number; text: string | null }> = [];
  for (let i = from; i < view.length; i++) {
    const e = view[i]!;
    if (isSteerTurn(e)) rows.push({ at: i, text: null });
    else if (isPrompt(e) && e.promptIndex === null && !unsent.has(i)) rows.push({ at: i, text: e.text });
  }
  // Each stored prompt pairs with the earliest row that can hold it, then with the latest: a row paired the same way
  // both times is that prompt in every pairing.
  const early = new Map<number, number>();
  let next = first;
  for (const row of rows) {
    if (next >= stored.length) break;
    if (row.text !== null && stored[next]!.text !== row.text) continue;
    early.set(row.at, next);
    next += 1;
  }
  const late = new Map<number, number>();
  let last = next - 1;
  for (let k = rows.length - 1; k >= 0 && last >= first; k--) {
    const row = rows[k]!;
    if (row.text !== null && stored[last]!.text !== row.text) continue;
    late.set(row.at, last);
    last -= 1;
  }
  const pair = early.get(at);
  return pair !== undefined && late.get(at) === pair ? { target: stored[pair]!.index } : { refused: 'not-sent' };
}

/**
 * The last prompt of the answer that prompt `index` starts, in `disk`: the automatic `continue` prompts right after it
 * are part of that answer. A fork names it as its last prompt kept (inclusive, unlike a rewind's target).
 */
export function answerEnd(disk: readonly TranscriptEntry[], index: number): number {
  let end = index;
  let inAnswer = false;
  for (const e of disk) {
    if (!isPrompt(e) || e.promptIndex === null) continue;
    if (e.promptIndex === index) {
      inAnswer = true;
      continue;
    }
    if (!inAnswer) continue;
    if (e.continued !== true) break;
    end = e.promptIndex;
  }
  return end;
}

/** What a rewind to the view's entry `at` removes, and the background work of that part it must end first. */
export interface RewindCounts {
  /** Typed messages removed, the one rewound to included. */
  messages: number;
  /** Entries after that message that go with it (a turn's end and a model line are not steps). */
  steps: number;
  /** The live commands and monitors the removed part started, by task id. */
  tasks: string[];
  /** The schedules the removed part made that the agent has not removed, by id. */
  schedules: string[];
  /** The workflow runs the removed part launched that the host can still stop, by run id. */
  runs: string[];
  /** Helper rows (subagents, workflow runs) go with it. */
  helpers: boolean;
  /** The message carried files or pictures: they are not attached again. */
  attachments: boolean;
}

export function rewindCounts(
  view: readonly TranscriptEntry[],
  at: number,
  liveTasks: ReadonlySet<string>,
  stoppableRuns: ReadonlySet<string> = new Set(),
): RewindCounts {
  const counts: RewindCounts = {
    messages: 0,
    steps: 0,
    tasks: [],
    schedules: [],
    runs: [],
    helpers: false,
    attachments: false,
  };
  for (const [runId, place] of launchPlaces(view)) {
    if (place >= at && stoppableRuns.has(runId)) counts.runs.push(runId);
  }
  const entry = view[at];
  if (entry?.kind === 'user') counts.attachments = (entry.attachments?.length ?? 0) > 0;
  for (let i = at; i < view.length; i++) {
    const e = view[i]!;
    if (isTyped(e)) counts.messages += 1;
    if (i > at && e.kind !== 'turn-end' && e.kind !== 'model') counts.steps += 1;
    if (e.kind === 'subagent' || e.kind === 'workflow') counts.helpers = true;
    if (e.kind !== 'tool') continue;
    if (
      e.task !== undefined &&
      e.task.endTs === undefined &&
      liveTasks.has(e.task.id) &&
      !counts.tasks.includes(e.task.id)
    )
      counts.tasks.push(e.task.id);
    if (e.schedule !== undefined && e.schedule.endTs === undefined && !counts.schedules.includes(e.schedule.id))
      counts.schedules.push(e.schedule.id);
  }
  return counts;
}

/**
 * Where each run was launched, by run id: its launch call (the main agent's `workflow` call that started it), else its
 * `workflow` entry when the view holds no launch of it (a cut, a fork, a client resume).
 */
function launchPlaces(view: readonly TranscriptEntry[]): Map<string, number> {
  const places = new Map<string, number>();
  view.forEach((e, i) => {
    if (e.kind !== 'tool' || e.run?.kind !== 'launch' || e.run.id === null || e.status !== 'completed') return;
    if (!places.has(e.run.id)) places.set(e.run.id, i);
  });
  view.forEach((e, i) => {
    if (e.kind === 'workflow' && !places.has(e.runId)) places.set(e.runId, i);
  });
  return places;
}

/**
 * The files of a rewind plan from the agent's dry run: each path through `shown` (redacted, hidden characters made
 * visible), the files changed outside the agent first, then by path; at most {@link REWIND_FILES_MAX}, and how many
 * more there are.
 */
export function planFiles(
  files: ReadonlyArray<{ path: string; state: RewindFileState }>,
  shown: (path: string) => string,
): { files: RewindFile[]; filesMore: number } {
  const sorted = files
    .map((file) => ({ path: shown(file.path), state: file.state }))
    .sort((a, b) => {
      const outside = Number(a.state === 'clean') - Number(b.state === 'clean');
      if (outside !== 0) return outside;
      return a.path < b.path ? -1 : a.path > b.path ? 1 : 0;
    });
  return { files: sorted.slice(0, REWIND_FILES_MAX), filesMore: Math.max(0, sorted.length - REWIND_FILES_MAX) };
}
