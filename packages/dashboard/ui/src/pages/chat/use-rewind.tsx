/**
 * Rewind to here on the conversation page: the dialog's states from the plan to the end that rides the stream, Work's
 * choice of what to put back, the marks of a slow rewind (20 s, 120 s), and what the page does after: the message text
 * back in the box (a draft that is not empty stays, and a toast with no timeout holds the text for that conversation),
 * focus, and the live sentence. "Files only" keeps the conversation: nothing goes back in the box. The dialog is for the
 * message it was opened on: a sync that puts another entry at its place says the conversation changed. A file changed
 * since the check plans again; files the restore could not put back are named. A rewind that runs when the page opens
 * another conversation keeps that conversation's stream open: its end puts the text in that conversation's box and
 * says how it ended in a toast that opens it.
 */

import { useCallback, useEffect, useRef, useState, type ReactElement } from 'react';

import {
  defaultChoice,
  hasFileChoices,
  RewindDialog,
  type RewindDialogState,
} from '../../components/arc-x/rewind-dialog/rewind-dialog';
import { mutate } from '../../lib/api';
import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatMode, RewindEnd, RewindEndMark, RewindMode } from '../../lib/chat-types';
import { filesWord } from '../../lib/permission-text';
import { toast } from '../../lib/toast';
import type { TranscriptEntry } from '../../lib/types';
import { historyActions, onRewindLast, rewindEndText, rewindRefusal } from './history-actions';
import { watchConversation } from './use-conversation';

/** When the dialog says the rewind is slow, and when it offers to restart the agent. */
export const REWIND_MARKS = { slowMs: 20_000, longMs: 120_000 } as const;

export interface RewindOptions {
  conversationId: string | null;
  mode: ChatMode;
  /** The conversation is loaded in the running agent: Work's plan only checks the files. */
  hot: boolean;
  /** Work: the folder, named when the agent stopped while it put files back. */
  folder: string | null;
  entries: readonly TranscriptEntry[];
  /** The stream's last rewind end (`ConversationState.rewindEnd`), numbered; `sync` repeats it. */
  end: RewindEndMark | null;
  /** Why Rewind to here is off now; null when it may run. */
  off: string | null;
  /** The agent's live commands of every conversation: a restart stops them. */
  commands: number;
  /** The text in a conversation's message box. */
  readDraft: (conversationId: string) => string;
  /** Puts a rewound message in the message box of its conversation; the page opens it when it shows another. */
  putBack: (conversationId: string, text: string) => void;
  /** Puts a rewound message in the message box of a conversation the page does not show, and opens nothing. */
  keepText: (conversationId: string, text: string) => void;
  openConversation: (conversationId: string) => void;
  /** A conversation's name as the list shows it. */
  titleOf: (conversationId: string) => string;
  /** The message box: focus goes there after a rewind. */
  composer: () => HTMLElement | null;
  focusComposer: () => void;
  announce: (text: string) => void;
  marks?: { slowMs: number; longMs: number };
}

export interface RewindControls {
  /** Rewind to the message at entry `at`; `from` is the button focus goes back to when it is cancelled. */
  onRewind: (at: number, from: HTMLElement | null) => void;
  /** The thread was rewound to its start in this page: one muted line stands for it. */
  atStart: boolean;
  dialog: ReactElement | null;
}

interface Open {
  conversationId: string;
  at: number;
  /** The time of the entry at `at` when it was opened: the plan is asked for that message only. */
  ts: number;
  text: string;
  from: HTMLElement | null;
  state: RewindDialogState;
  /** What the rewind puts back: Work's choice once the plan has files, else the conversation. */
  choice: RewindMode;
  /** The rewind was sent and its end has not come. */
  sent: boolean;
  /** The number of the conversation's last rewind end when the rewind was sent: a higher one is its end. */
  endBefore: number;
}

/** What the page says of a done rewind: the toast and the live sentence. */
export function rewindDoneWords(end: Extract<RewindEnd, { outcome: 'done' }>): { toast: string; status: string } {
  const h = CHAT.history;
  const files = filesWord(end.restored);
  if (end.mode === 'files')
    return { toast: fill(h.doneFilesOnly, { files }), status: fill(h.statusFilesOnly, { files }) };
  if (end.restored > 0) return { toast: fill(h.doneFiles, { files }), status: fill(h.statusFiles, { files }) };
  return { toast: h.done, status: h.status };
}

/** A message the user typed and the agent may have run: notes, steers and the automatic `continue` are not. */
export function isRewindPoint(entry: TranscriptEntry): boolean {
  return entry.kind === 'user' && entry.note !== true && entry.interjection !== true && entry.continued !== true;
}

export function useRewind(options: RewindOptions): RewindControls {
  const marks = options.marks ?? REWIND_MARKS;
  const { conversationId, entries } = options;
  const [open, setOpen] = useState<Open | null>(null);
  const [atStart, setAtStart] = useState<string | null>(null);
  const opts = useRef(options);
  useEffect(() => {
    opts.current = options;
  });
  /** The number of the last rewind end this page had when it sent its rewind; a higher one is that rewind's end. */
  const endSeen = useRef(options.end?.n ?? 0);
  const timers = useRef<Array<ReturnType<typeof setTimeout>>>([]);

  const clearTimers = (): void => {
    for (const timer of timers.current.splice(0)) clearTimeout(timer);
  };
  useEffect(() => clearTimers, []);

  // The dialog stays drawn while it leaves, with the state it closed in.
  const kept = useRef<Open | null>(null);
  if (open !== null) kept.current = open;
  const finished = useRef(false);
  const openNow = useRef(open);
  openNow.current = open;

  /** Rewinds that run in a conversation the page no longer shows: the stop of each one's stream watch. */
  const away = useRef(new Map<string, () => void>());
  const stopAway = (id: string): boolean => {
    const stop = away.current.get(id);
    if (stop === undefined) return false;
    away.current.delete(id);
    stop();
    return true;
  };
  useEffect(() => {
    const watches = away.current;
    return () => {
      for (const stop of watches.values()) stop();
      watches.clear();
    };
  }, []);

  /** How a rewind ended that the dialog no longer shows: a toast that names the conversation and opens it. */
  const endAway = (was: Open, result: RewindEnd): void => {
    const now = opts.current;
    const id = was.conversationId;
    const here = now.conversationId === id;
    const h = CHAT.history;
    const where = {
      description: now.titleOf(id),
      ...(here
        ? {}
        : {
            action: {
              label: h.endOpen,
              onClick: (toastId: string) => {
                opts.current.openConversation(id);
                toast.dismiss(toastId);
              },
            },
          }),
    };
    if (result.outcome !== 'done' && result.outcome !== 'partial') {
      if (result.outcome === 'changed') toast.warning(h.planChanged, where);
      else toast.error(rewindEndText(result), where);
      return;
    }
    const partial = result.outcome === 'partial';
    const words = partial
      ? fill(result.mode === 'files' ? h.errPartialFiles : h.errPartial, {
          files: filesWord(result.failed.length + result.failedMore),
        })
      : rewindDoneWords(result).toast;
    const say = partial ? toast.error : toast.success;
    if (result.mode === 'files') {
      say(words, where);
      return;
    }
    if (here && result.entry === 0) setAtStart(id);
    if (now.readDraft(id).trim() === '') {
      if (here) now.putBack(id, was.text);
      else now.keepText(id, was.text);
      say(words, where);
      return;
    }
    say(words, {
      description: where.description,
      duration: Number.POSITIVE_INFINITY,
      action: {
        label: h.putBack,
        onClick: (toastId: string) => {
          opts.current.putBack(id, was.text);
          opts.current.focusComposer();
          toast.dismiss(toastId);
        },
      },
    });
  };

  // Another conversation opens: the dialog closes, and a rewind it sent keeps its conversation's stream until its end.
  useEffect(() => {
    const was = openNow.current;
    setOpen(null);
    clearTimers();
    if (was === null || !was.sent || was.conversationId === conversationId) return;
    const id = was.conversationId;
    stopAway(id);
    let ended = false;
    const stop = watchConversation(id, (snapshot) => {
      const mark = snapshot.state?.rewindEnd ?? null;
      if (ended || mark === null || mark.n <= was.endBefore) return;
      ended = true;
      stopAway(id);
      endAway(was, mark.end);
    });
    if (ended) stop();
    else away.current.set(id, stop);
  }, [conversationId]);

  const plan = useCallback((id: string, at: number, ts: number, changed = false): void => {
    const now = opts.current;
    const mode = now.mode;
    const same = (current: Open | null): current is Open =>
      current?.conversationId === id && current.at === at && current.ts === ts;
    historyActions.rewindPlan(id, at, ts).then(
      (found) =>
        setOpen((current) =>
          same(current) && current.state.phase === 'loading'
            ? {
                ...current,
                choice: defaultChoice(mode, found),
                state: { phase: 'ready', plan: found, ...(changed ? { changed: true } : {}) },
              }
            : current,
        ),
      (err: unknown) => {
        const refused = rewindRefusal(err);
        setOpen((current) =>
          same(current)
            ? {
                ...current,
                state: {
                  phase: 'error',
                  tone: refused.tone,
                  text: refused.text,
                  ...(refused.retry === true ? { retry: true } : {}),
                },
              }
            : current,
        );
      },
    );
  }, []);

  const loading = (): RewindDialogState => ({
    phase: 'loading',
    checking: opts.current.mode === 'work' && opts.current.hot,
  });

  const onRewind = useCallback(
    (at: number, from: HTMLElement | null): void => {
      const now = opts.current;
      const entry = now.entries[at];
      if (now.conversationId === null || now.off !== null || entry === undefined || entry.kind !== 'user') return;
      const id = now.conversationId;
      finished.current = false;
      setOpen({
        conversationId: id,
        at,
        ts: entry.ts,
        text: entry.text,
        from,
        state: { phase: 'loading', checking: now.mode === 'work' && now.hot },
        choice: 'conversation',
        sent: false,
        endBefore: 0,
      });
      plan(id, at, entry.ts);
    },
    [plan],
  );

  // A sync that puts another entry where the message was: the plan is not for what the dialog quotes.
  useEffect(() => {
    setOpen((current) => {
      if (current === null || current.sent || current.state.phase === 'error') return current;
      const entry = entries[current.at];
      if (entry?.kind === 'user' && entry.ts === current.ts && entry.text === current.text) return current;
      return { ...current, state: { phase: 'error', tone: 'warning', text: CHAT.history.errChanged } };
    });
  }, [entries]);

  useEffect(
    () =>
      onRewindLast((id) => {
        if (id !== opts.current.conversationId) return;
        const at = lastRewindPoint(opts.current.entries);
        if (at === null) toast.info(CHAT.history.errUnsent);
        else onRewind(at, null);
      }),
    [onRewind],
  );

  // The dialog cannot be closed while the rewind runs: Keep waiting arms this again, so Restart always comes back.
  const armLong = (): void => {
    timers.current.push(
      setTimeout(() => {
        setOpen((current) =>
          current?.state.phase === 'working'
            ? { ...current, state: { phase: 'long', plan: current.state.plan } }
            : current,
        );
      }, marks.longMs),
    );
  };

  const confirm = (): void => {
    if (open === null || open.state.phase !== 'ready') return;
    const shown = open.state.plan;
    const id = open.conversationId;
    const choice = hasFileChoices(opts.current.mode, shown) ? open.choice : 'conversation';
    const before = opts.current.end?.n ?? 0;
    endSeen.current = before;
    setOpen({
      ...open,
      choice,
      state: { phase: 'working', plan: shown, slow: false },
      sent: true,
      endBefore: before,
    });
    clearTimers();
    timers.current.push(
      setTimeout(() => {
        setOpen((current) =>
          current?.state.phase === 'working' ? { ...current, state: { ...current.state, slow: true } } : current,
        );
      }, marks.slowMs),
    );
    armLong();
    historyActions.rewind(id, shown.token, choice).catch((err: unknown) => {
      const refused = rewindRefusal(err);
      if (stopAway(id)) {
        toast.error(refused.text, {
          description: opts.current.titleOf(id),
          action: {
            label: CHAT.history.endOpen,
            onClick: (toastId: string) => {
              opts.current.openConversation(id);
              toast.dismiss(toastId);
            },
          },
        });
        return;
      }
      clearTimers();
      setOpen((current) =>
        current?.conversationId === id
          ? { ...current, sent: false, state: { phase: 'error', tone: refused.tone, text: refused.text } }
          : current,
      );
    });
  };

  /**
   * The removed message back in its conversation's box when the draft is empty, else a toast with no timeout holds it
   * (`said`: the toast's words; null says only that it holds the message).
   */
  const refill = (id: string, text: string, said: string | null): void => {
    const now = opts.current;
    if (now.readDraft(id).trim() === '') {
      now.putBack(id, text);
      if (said !== null) toast.success(said);
      return;
    }
    const input = {
      duration: Number.POSITIVE_INFINITY,
      action: {
        label: CHAT.history.putBack,
        onClick: (toastId: string) => {
          opts.current.putBack(id, text);
          opts.current.focusComposer();
          toast.dismiss(toastId);
        },
      },
    };
    if (said === null) toast.info(CHAT.history.keptMessage, input);
    else toast.success(said, input);
  };

  // The end rides the stream, after the cut's `sync`; a page that synced again in between finds it in the `sync`.
  const end = options.end;
  useEffect(() => {
    if (end === null || end.n <= endSeen.current) return;
    endSeen.current = end.n;
    // In the render that opens another conversation `end` is that one's while `open` is still the rewind left behind.
    if (open === null || !open.sent || open.conversationId !== conversationId) return;
    clearTimers();
    const result = end.end;
    const now = opts.current;
    if (result.outcome === 'changed') {
      setOpen({ ...open, sent: false, state: loading() });
      plan(open.conversationId, open.at, open.ts, true);
      return;
    }
    if (result.outcome === 'partial') {
      finished.current = true;
      const h = CHAT.history;
      const files = filesWord(result.failed.length + result.failedMore);
      const text = fill(result.mode === 'files' ? h.errPartialFiles : h.errPartial, { files });
      setOpen({
        ...open,
        sent: false,
        state: { phase: 'error', tone: 'danger', text, files: { paths: result.failed, more: result.failedMore } },
      });
      if (result.mode !== 'files') {
        if (result.entry === 0) setAtStart(open.conversationId);
        refill(open.conversationId, open.text, null);
      }
      now.announce(text);
      return;
    }
    if (result.outcome !== 'done') {
      const text =
        result.outcome === 'stopped' && open.choice !== 'conversation' && now.folder !== null
          ? fill(CHAT.history.errStoppedFiles, { folder: now.folder })
          : rewindEndText(result);
      setOpen({
        ...open,
        sent: false,
        state: { phase: 'error', tone: 'danger', text, ...(result.outcome === 'load-slow' ? { retry: true } : {}) },
      });
      return;
    }
    finished.current = true;
    setOpen(null);
    const words = rewindDoneWords(result);
    if (result.mode === 'files') {
      toast.success(words.toast);
      now.focusComposer();
      now.announce(words.status);
      return;
    }
    if (result.entry === 0) setAtStart(open.conversationId);
    refill(open.conversationId, open.text, words.toast);
    now.focusComposer();
    now.announce(words.status);
  }, [end, open, conversationId]);

  useEffect(() => {
    if (atStart !== null && (atStart !== conversationId || entries.length > 0)) setAtStart(null);
  }, [atStart, conversationId, entries.length]);

  const close = (): void => {
    clearTimers();
    setOpen(null);
  };

  const shown = open ?? kept.current;
  const dialog =
    shown === null ? null : (
      <RewindDialog
        open={open !== null}
        state={shown.state}
        mode={options.mode}
        message={shown.text}
        commands={options.commands}
        choice={shown.choice}
        onChoice={(choice) => {
          setOpen((current) => (current?.state.phase === 'ready' ? { ...current, choice } : current));
        }}
        onRetry={() => {
          if (open === null || open.state.phase !== 'error') return;
          setOpen({ ...open, state: loading() });
          plan(open.conversationId, open.at, open.ts);
        }}
        onClose={close}
        onConfirm={confirm}
        onKeepWaiting={() => {
          setOpen((current) =>
            current?.state.phase === 'long'
              ? { ...current, state: { phase: 'working', plan: current.state.plan, slow: true } }
              : current,
          );
          armLong();
        }}
        onRestart={() => {
          mutate('POST', '/chat/agent', { action: 'restart', mode: options.mode }).catch(() => {
            toast.error(CHAT.history.errFailed);
          });
        }}
        returnFocus={() => (finished.current ? opts.current.composer() : shown.from)}
      />
    );

  return { onRewind, atStart: atStart !== null && atStart === conversationId && entries.length === 0, dialog };
}

/** The index of the last message that can be a rewind point, or null when none. */
export function lastRewindPoint(entries: readonly TranscriptEntry[]): number | null {
  for (let i = entries.length - 1; i >= 0; i--) if (isRewindPoint(entries[i]!)) return i;
  return null;
}
