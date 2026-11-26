/**
 * The Context panel of the conversation the page shows: open or not, and the agent's parts, read when it opens and
 * again after each answer's end and each compaction's end while it stays open.
 */

import { useEffect, useRef, useState } from 'react';

import type { ContextParts } from '../../lib/chat-types';
import { historyActions, onShowContext } from './history-actions';

export interface ContextPanelState {
  open: boolean;
  setOpen: (open: boolean) => void;
  /** Undefined while the first read runs; null when the conversation is not loaded or the read failed. */
  parts: ContextParts | null | undefined;
  /** The use the last read gave, for a page that has none of its own. */
  used: number | null;
}

/**
 * `conversationId` null: no panel (a new conversation has no ring). `turns` and `compactionEnd` change at each answer's
 * end and each manual compaction's end; an open panel reads the parts again then.
 */
export function useContextPanel(
  conversationId: string | null,
  turns: number,
  compactionEnd: object | null,
): ContextPanelState {
  const [open, setOpen] = useState(false);
  const [read, setRead] = useState<{ id: string; parts: ContextParts | null; used: number | null } | null>(null);
  const asked = useRef(0);

  useEffect(() => {
    setOpen(false);
    setRead(null);
  }, [conversationId]);

  useEffect(
    () =>
      onShowContext((id) => {
        if (id === conversationId) setOpen(true);
      }),
    [conversationId],
  );

  useEffect(() => {
    if (!open || conversationId === null) return;
    const ask = ++asked.current;
    void historyActions.context(conversationId).then((answer) => {
      if (asked.current !== ask) return;
      setRead({ id: conversationId, parts: answer?.parts ?? null, used: answer?.used ?? null });
    });
  }, [open, conversationId, turns, compactionEnd]);

  useEffect(() => {
    if (!open) asked.current += 1;
  }, [open]);

  const current = read !== null && read.id === conversationId ? read : null;
  return { open, setOpen, parts: current === null ? undefined : current.parts, used: current?.used ?? null };
}
