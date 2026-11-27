/**
 * The queue's routes and its words. The host owns the list: each call changes it there, and the new list comes back
 * on the conversation stream (`queue`). A row open for edit is told to the host, so it sends no row meanwhile.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';

import { DashboardApiError, mutate } from '../../lib/api';
import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatHostState, ChatMode, QueueHold, QueueState } from '../../lib/chat-types';
import { toast } from '../../lib/toast';
import { sendFailure } from './blockers';
import { conversationPath } from './conversation-actions';

/**
 * The header's words for why the list waits, or null while it sends by itself. A failed check says the page's own
 * sentence for its code, as a refused send would. The engine-lost count is every Work conversation's commands (the
 * agent process is shared), and names the ones that run in other conversations.
 */
export function heldSentence(
  held: QueueHold | null,
  host: ChatHostState | null,
  conversationId: string | null,
  mode: ChatMode,
): string | null {
  if (held === null) return null;
  switch (held.why) {
    case 'stopped':
      return CHAT.queue.held.stopped;
    case 'failed':
      return CHAT.queue.held.failed;
    case 'restart':
      return CHAT.queue.held.restart;
    case 'blocked': {
      const tasks = host?.modes.work.tasks ?? 0;
      if (host !== null && held.code === 'engine-lost' && tasks > 0) {
        const stops = tasks === 1 ? CHAT.queue.held.engineLostOne : fill(CHAT.queue.held.engineLost, { n: tasks });
        const here =
          host.active.find((entry) => entry.mode === 'work' && entry.conversationId === conversationId)?.tasks ?? 0;
        const others = tasks - here;
        return others > 0 ? `${stops} ${fill(CHAT.engineRestartOthers, { n: others })}` : stops;
      }
      const failure = sendFailure(held.code ?? undefined, undefined, mode, held.reason);
      return fill(CHAT.queue.held.blocked, { reason: 'toast' in failure ? failure.toast : failure.strip.text });
    }
  }
}

function rowPath(conversationId: string, rowId: string): string {
  return `${conversationPath(conversationId)}/queue/${encodeURIComponent(rowId)}`;
}

/** Tells why a Send now did not go: a warning for the host's refusals (the row stays queued), else a failure. */
export function toastSteerRefusal(reason: string | undefined): void {
  const words = steerRefusal(reason);
  if (words === null) return;
  if (words === CHAT.queue.failed) toast.error(words);
  else toast.warning(words);
}

/**
 * The words for a Send now the host did not take; the row stays queued. Null when the answer ended first: the list
 * sends the row by itself, or its held header says why it waits.
 */
export function steerRefusal(reason: string | undefined): string | null {
  switch (reason) {
    case 'steer-not-now':
      return null;
    case 'steer-busy':
      return CHAT.queue.notThisStep;
    case 'steer-late':
      return CHAT.queue.sendNowLate;
    case 'steer-files':
      return CHAT.queue.sendNowOffFiles;
    case 'steer-slash':
      return CHAT.queue.sendNowOffSlash;
    default:
      return CHAT.queue.failed;
  }
}

const NONE: ReadonlySet<string> = new Set();

/** A row that is gone already (sent or removed meanwhile) is no failure for the user. */
function gone(err: unknown): boolean {
  return err instanceof DashboardApiError && err.code === 'E_NOT_FOUND';
}

export interface QueueControls {
  /** The row open for edit in this window, or null. */
  editing: string | null;
  setEditing: (rowId: string | null) => void;
  save: (rowId: string, text: string) => Promise<boolean>;
  remove: (rowId: string) => void;
  clear: () => void;
  sendNext: () => void;
  /** Work: a row goes into the running answer. */
  sendNow: (rowId: string) => void;
  /** Rows whose Send now waits for its reply. */
  sending: ReadonlySet<string>;
}

/**
 * The page's side of one conversation's queue. Opening a row for edit tells the host before the box shows; leaving it
 * (Cancel, Esc, another conversation) tells it again. A later view where the host no longer holds the open row (it
 * ends every edit when no page watches the conversation) holds it again. `announce` says each change in the page's
 * live region; `keepEdit` gets the text of a save whose row had left the queue, true when it went to the message box.
 */
export function useQueueControls(
  conversationId: string | null,
  queue: QueueState,
  announce: (text: string) => void,
  keepEdit: (conversationId: string, text: string) => boolean,
): QueueControls {
  const [open, setOpen] = useState<{ conversation: string; row: string } | null>(null);
  const openRef = useRef(open);
  useLayoutEffect(() => {
    openRef.current = open;
  });
  /** The view when the open row was opened: the host's answer to that comes in a later one. */
  const openedOn = useRef<QueueState | null>(null);
  /** The row whose save went out: the host ends its edit itself, so it is never held again. */
  const saving = useRef<string | null>(null);

  const tell = useCallback((conversation: string, row: string, on: boolean): void => {
    mutate('PATCH', rowPath(conversation, row), { editing: on }).catch((err: unknown) => {
      if (!gone(err)) toast.error(CHAT.queue.failed);
    });
  }, []);

  // Another conversation: an edit left open there ends, so its queue can send again.
  useEffect(() => {
    const was = openRef.current;
    if (was === null || was.conversation === conversationId) return;
    setOpen(null);
    tell(was.conversation, was.row, false);
  }, [conversationId, tell]);

  const editing =
    open !== null && open.conversation === conversationId && queue.items.some((item) => item.id === open.row)
      ? open.row
      : null;

  useEffect(() => {
    if (open === null || open.conversation !== conversationId || queue === openedOn.current) return;
    if (saving.current === open.row) return;
    if (queue.items.some((item) => item.id === open.row && !item.editing)) tell(open.conversation, open.row, true);
  }, [open, conversationId, queue, tell]);

  const setEditing = useCallback(
    (rowId: string | null): void => {
      if (conversationId === null) return;
      const was = openRef.current;
      if (rowId === null) {
        setOpen(null);
        if (was !== null) tell(was.conversation, was.row, false);
        return;
      }
      if (was !== null && was.row !== rowId) tell(was.conversation, was.row, false);
      openedOn.current = queue;
      saving.current = null;
      setOpen({ conversation: conversationId, row: rowId });
      tell(conversationId, rowId, true);
    },
    [conversationId, queue, tell],
  );

  const save = useCallback(
    async (rowId: string, text: string): Promise<boolean> => {
      if (conversationId === null) return false;
      saving.current = rowId;
      try {
        await mutate('PATCH', rowPath(conversationId, rowId), { text });
      } catch (err) {
        if (gone(err)) {
          setOpen((current) => (current?.row === rowId ? null : current));
          if (keepEdit(conversationId, text)) toast.warning(CHAT.queue.editGoneBack);
          else toast.error(CHAT.queue.editGone);
          return true;
        }
        if (saving.current === rowId) saving.current = null;
        toast.error(CHAT.queue.failed);
        return false;
      }
      setOpen((current) => (current?.row === rowId ? null : current));
      return true;
    },
    [conversationId, keepEdit],
  );

  const remove = useCallback(
    (rowId: string): void => {
      if (conversationId === null) return;
      mutate('DELETE', rowPath(conversationId, rowId)).then(
        () => announce(CHAT.queue.status.removed),
        (err: unknown) => {
          if (!gone(err)) toast.error(CHAT.queue.failed);
        },
      );
    },
    [conversationId, announce],
  );

  const clear = useCallback((): void => {
    if (conversationId === null) return;
    mutate('DELETE', `${conversationPath(conversationId)}/queue`).then(
      () => announce(CHAT.queue.status.cleared),
      () => toast.error(CHAT.queue.failed),
    );
  }, [conversationId, announce]);

  const sendNext = useCallback((): void => {
    if (conversationId === null) return;
    mutate('POST', `${conversationPath(conversationId)}/queue/release`).catch(() => toast.error(CHAT.queue.failed));
  }, [conversationId]);

  const [sending, setSending] = useState<ReadonlySet<string>>(NONE);
  const sendNow = useCallback(
    (rowId: string): void => {
      if (conversationId === null) return;
      setSending((current) => new Set(current).add(rowId));
      mutate('POST', `${rowPath(conversationId, rowId)}/send`, { how: 'now' })
        .then(
          () => announce(CHAT.queue.status.sent),
          (err: unknown) => {
            if (!gone(err)) toastSteerRefusal(err instanceof DashboardApiError ? err.reason : undefined);
          },
        )
        .finally(() =>
          setSending((current) => {
            const next = new Set(current);
            next.delete(rowId);
            return next;
          }),
        );
    },
    [conversationId, announce],
  );

  return { editing, setEditing, save, remove, clear, sendNext, sendNow, sending };
}
