/**
 * The one way in for every history op: the menu rows, the palette and the slash commands call the same function, which
 * posts the conversation's op route. An op answers once it started; its end comes on the conversation stream.
 */

import { DashboardApiError, getJson, getMark, mutate } from '../../lib/api';
import { CHAT, fill } from '../../lib/chat-strings';
import type {
  ChatFailureCode,
  CompactOutcome,
  ConversationContext,
  RecapEnd,
  RewindEnd,
  RewindMode,
  RewindPlan,
} from '../../lib/chat-types';
import { formatCount } from '../../lib/format';
import { folderName } from '../../lib/permission-text';
import { toast } from '../../lib/toast';
import type { TranscriptEntry } from '../../lib/types';
import { BLOCKER_SENTENCES } from './blockers';
import { conversationPath } from './conversation-actions';

/** What the page knows that turns a history op off. */
export interface HistoryFacts {
  /** Why the agent cannot act now (the engine stopped, the agent missing…), or null. */
  blocked: string | null;
  /** An answer, a wake or a compaction runs. */
  busy: boolean;
  /** A workflow run or a subagent is live. */
  runLive: boolean;
  /** The live work is a subagent of the conversation's own, not a workflow run. */
  helperLive: boolean;
  /** The engine restarted under live commands: loading the conversation would restart the agent and stop them. */
  engineLost: boolean;
  /** Answers of the conversation that finished. */
  finished: number;
  /** A history op or a recap of the conversation runs now (this window's or another's). */
  op: 'rewind' | 'compact' | 'fork' | 'recap' | null;
}

/** Why the op that runs now turns another off: a fork and a recap say so; a rewind or a compaction is as an answer. */
function opRunning(op: HistoryFacts['op']): string | null {
  if (op === 'fork') return CHAT.history.forkRunning;
  if (op === 'recap') return CHAT.history.offRecap;
  return null;
}

/**
 * Why Compact now is off, or null when it may run. The host's own walls are the same (`engineBusy`, a fork or a recap,
 * `engine-lost`).
 */
export function compactOff(facts: HistoryFacts): string | null {
  if (facts.blocked !== null) return facts.blocked;
  const running = opRunning(facts.op);
  if (running !== null) return running;
  if (facts.busy) return CHAT.modelBusy;
  if (facts.engineLost) return CHAT.history.engineLost;
  if (facts.runLive) return facts.helperLive ? CHAT.history.offHelper : CHAT.history.offRun;
  if (facts.finished === 0) return CHAT.history.compactEmpty;
  return null;
}

/**
 * Why Rewind to here is off, or null when it may run. The host's own walls are the same (`engineBusy`, a fork or a
 * recap, `engine-lost`).
 */
export function rewindOff(facts: HistoryFacts): string | null {
  if (facts.blocked !== null) return facts.blocked;
  const running = opRunning(facts.op);
  if (running !== null) return running;
  if (facts.busy) return CHAT.modelBusy;
  if (facts.engineLost) return CHAT.history.engineLost;
  if (facts.runLive) return facts.helperLive ? CHAT.history.offHelper : CHAT.history.offRun;
  return null;
}

/** Why Rewind to the message at `at` is off for the goal: a rewind may not cut the row of the open goal (`goalAt`). */
export function rewindGoalOff(at: number, goalAt: number | null): string | null {
  return goalAt !== null && at < goalAt ? CHAT.history.offGoal : null;
}

/** Why Fork is off, or null when it may run: the walls of a rewind (`engineBusy`, `engine-lost`), on the host too. */
export function forkOff(facts: HistoryFacts): string | null {
  return rewindOff(facts);
}

/**
 * Why Recap is off, or null when it may run: a recap is being written, an answer, a wake or a history op runs, the
 * engine restarted under live commands, or no answer finished. A workflow run or a helper does not turn it off: it only
 * reads. The host's walls are the same (one recap at a time, `isBusy`, `engine-lost`).
 */
export function recapOff(facts: HistoryFacts): string | null {
  if (facts.blocked !== null) return facts.blocked;
  if (facts.op === 'recap') return CHAT.history.recapWriting;
  if (facts.busy) return CHAT.modelBusy;
  if (facts.engineLost) return CHAT.history.engineLost;
  if (facts.finished === 0) return CHAT.history.recapEmpty;
  return null;
}

/** What the page says when a recap ended with no row: a toast, or nothing (`shown`: the row; `cancelled`: a message went). */
export function recapEndToast(outcome: RecapEnd): { tone: 'info' | 'error'; text: string } | null {
  switch (outcome) {
    case 'none':
      return { tone: 'info', text: CHAT.history.recapNone };
    case 'off':
      return { tone: 'info', text: CHAT.history.recapOff };
    case 'failed':
      return { tone: 'error', text: CHAT.history.recapError };
    case 'load-slow':
      return { tone: 'error', text: CHAT.history.loadSlow };
    case 'shown':
    case 'cancelled':
      return null;
  }
}

/**
 * The toast of a refused fork: not a fork point, the message moved, busy, a named blocker, else "Could not fork the
 * conversation."
 */
export function forkRefusal(err: unknown): string {
  const reason = err instanceof DashboardApiError ? err.reason : undefined;
  if (reason === 'fork-not-point') return CHAT.history.forkUnsent;
  if (reason === 'fork-changed') return CHAT.history.forkChanged;
  return refusal(err, CHAT.history.forkError);
}

/** What the page does once the copy exists: open it; Back to the original goes to the conversation forked. */
export interface ForkOpener {
  /** Work: the folder both conversations work in, said in the toast. */
  folder: string | null;
  /** The page still shows the original: only then does the copy open by itself. */
  here: () => boolean;
  open: (copyId: string) => void;
  back: () => void;
}

/** The sentence of a refused rewind plan or commit, how loud it is, and whether asking again may help. */
export function rewindRefusal(err: unknown): { tone: 'info' | 'warning' | 'danger'; text: string; retry?: true } {
  const reason = err instanceof DashboardApiError ? err.reason : undefined;
  if (reason === 'agent-slow') return { tone: 'warning', text: CHAT.history.errSlow, retry: true };
  if (reason === 'load-slow') return { tone: 'warning', text: CHAT.history.loadSlow, retry: true };
  if (reason === 'rewind-not-point') return { tone: 'info', text: CHAT.history.errUnsent };
  if (reason === 'rewind-changed') return { tone: 'warning', text: CHAT.history.errChanged };
  if (reason === 'rewind-goal') return { tone: 'info', text: CHAT.history.offGoal };
  return { tone: 'danger', text: refusal(err, CHAT.history.errFailed) };
}

/** The sentence of a rewind that ended without one, by how it ended. */
export function rewindEndText(
  end: Extract<RewindEnd, { outcome: 'refused' | 'tasks' | 'stopped' | 'failed' | 'load-slow' }>,
): string {
  switch (end.outcome) {
    case 'refused':
      return CHAT.history.errRefused;
    case 'tasks':
      return CHAT.history.errTasks;
    case 'stopped':
      return CHAT.history.errStopped;
    case 'failed':
      return CHAT.history.errFailed;
    case 'load-slow':
      return CHAT.history.loadSlow;
  }
}

/** What the page says when a manual compaction ended: a toast for a stop or a failure, and the live region's words. */
export function compactEndWords(
  outcome: CompactOutcome,
  entries: readonly TranscriptEntry[],
): { toast: 'info' | 'error' | null; text: string } {
  if (outcome === 'stopped') return { toast: 'info', text: CHAT.compactStopped };
  if (outcome === 'failed') return { toast: 'error', text: CHAT.compactFailed };
  if (outcome === 'load-slow') return { toast: 'error', text: CHAT.history.loadSlow };
  for (let i = entries.length - 1; i >= 0; i -= 1) {
    const entry = entries[i]!;
    if (entry.kind !== 'compaction') continue;
    if (entry.outcome !== 'done' || entry.tokensBefore === undefined || entry.tokensAfter === undefined) break;
    return {
      toast: null,
      text: fill(CHAT.history.compactedSaid, {
        before: formatCount(entry.tokensBefore),
        after: formatCount(entry.tokensAfter),
      }),
    };
  }
  return { toast: null, text: CHAT.compacted };
}

/**
 * The toast of a refused op: a fork or a recap runs, busy, a named blocker (the engine stopped…), else the op's own
 * failure words.
 */
function refusal(err: unknown, failed: string): string {
  const reason = err instanceof DashboardApiError ? err.reason : undefined;
  if (reason === 'fork-running') return CHAT.history.forkRunning;
  if (reason === 'recap-running') return CHAT.history.offRecap;
  if (reason === 'conversation-busy') return CHAT.sendErrors['conversation-busy'][0];
  return (reason === undefined ? undefined : BLOCKER_SENTENCES[reason as ChatFailureCode]) ?? failed;
}

/** The conversation page's way to open Rewind on its last sent message. */
type RewindOpener = (conversationId: string) => void;
const rewindOpeners = new Set<RewindOpener>();

/** The page that shows a conversation opens Rewind for {@link historyActions.rewindLast}; call to stop. */
export function onRewindLast(open: RewindOpener): () => void {
  rewindOpeners.add(open);
  return () => {
    rewindOpeners.delete(open);
  };
}

/** The conversation page's way to open its Context panel. */
type ContextOpener = (conversationId: string) => void;
const contextOpeners = new Set<ContextOpener>();

/** The page that shows a conversation opens its Context panel for {@link historyActions.showContext}; call to stop. */
export function onShowContext(open: ContextOpener): () => void {
  contextOpeners.add(open);
  return () => {
    contextOpeners.delete(open);
  };
}

export const historyActions = {
  /** Compact now. A refusal is a toast; the run's end comes as the conversation's `compaction` end. */
  async compact(conversationId: string): Promise<void> {
    try {
      await mutate('POST', `${conversationPath(conversationId)}/compact`);
    } catch (err) {
      toast.error(refusal(err, CHAT.compactFailed));
    }
  },

  /**
   * What Rewind to the message at `entry` (its `ts`, as the page holds it) would remove. Rejects with the route's error
   * (see {@link rewindRefusal}).
   */
  rewindPlan(conversationId: string, entry: number, ts: number): Promise<RewindPlan> {
    return mutate<RewindPlan>('POST', `${conversationPath(conversationId)}/rewind/plan`, { entry, ts });
  },

  /** The plan's rewind of `mode`; it ends on the conversation's stream (`rewind`). Rejects with the route's error. */
  async rewind(conversationId: string, token: string, mode: RewindMode): Promise<void> {
    await mutate('POST', `${conversationPath(conversationId)}/rewind`, { token, mode });
  },

  /**
   * Fork: a copy of the whole conversation, or (`at`) up to and with the answer of the message at `at.entry` (`at.ts`:
   * its time, as the page holds it). A toast says "Forking…" and turns into "Forked. This is the copy." with Back to
   * the original once the copy opens; a page that left the original stays where it is, and the toast opens the copy.
   * A refusal turns it into an error and the page stays.
   */
  async fork(conversationId: string, at: { entry: number; ts: number } | null, then: ForkOpener): Promise<void> {
    const id = toast.loading(CHAT.history.forking);
    let copy: { id: string };
    try {
      copy = await mutate<{ id: string }>('POST', `${conversationPath(conversationId)}/fork`, at ?? {});
    } catch (err) {
      toast.update(id, { type: 'error', title: forkRefusal(err) });
      return;
    }
    const where =
      then.folder === null
        ? {}
        : { description: fill(CHAT.history.forkSameFolder, { folder: `${folderName(then.folder)}/` }) };
    if (!then.here()) {
      toast.update(id, {
        type: 'success',
        title: CHAT.history.forkedAway,
        ...where,
        action: { label: CHAT.history.forkOpen, onClick: () => then.open(copy.id) },
      });
      return;
    }
    then.open(copy.id);
    toast.update(id, {
      type: 'success',
      title: CHAT.history.forked,
      ...where,
      action: { label: CHAT.history.forkBack, onClick: () => then.back() },
    });
  },

  /** Recap. A refusal is a toast; the wait and its end come on the conversation's stream (`recap`). */
  async recap(conversationId: string): Promise<void> {
    try {
      await mutate('POST', `${conversationPath(conversationId)}/recap`);
    } catch (err) {
      toast.error(refusal(err, CHAT.history.recapError));
    }
  },

  /** `/rewind`, `/undo` and the palette: Rewind on the last message sent in the conversation the page shows. */
  rewindLast(conversationId: string): void {
    for (const open of rewindOpeners) open(conversationId);
  },

  /** Opens the Context panel of the conversation the page shows. */
  showContext(conversationId: string): void {
    for (const open of contextOpeners) open(conversationId);
  },

  /** The Context panel's facts, read fresh; null when the read failed (the panel then shows no parts and no error). */
  async context(conversationId: string): Promise<ConversationContext | null> {
    try {
      return await getJson<ConversationContext>(`${conversationPath(conversationId)}/context`, { after: getMark() });
    } catch {
      return null;
    }
  },
};
