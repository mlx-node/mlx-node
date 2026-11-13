/**
 * One conversation's stream for the page. The store (`createConversationStore`) opens the stream while it is read,
 * closes it while the window is hidden and re-opens it after a reconnect or a `seq` gap. This hook adds two things:
 * the page is told at most once per animation frame, and the arrival time of the last event (the thinking clock).
 */

import { useCallback, useRef, useSyncExternalStore } from 'react';

import { createConversationStore, type ChatStore, type ConversationSnapshot } from '../../lib/chat-store';

const stores = new Map<string, ChatStore<ConversationSnapshot>>();
const arrivals = new Map<string, number>();
const EMPTY: ConversationSnapshot = { state: null, gone: null };

function storeFor(id: string): ChatStore<ConversationSnapshot> {
  let store = stores.get(id);
  if (store === undefined) {
    store = createConversationStore(id);
    stores.set(id, store);
  }
  return store;
}

/** Forget a conversation's store (it was deleted): a later open starts from a fresh `sync`. */
export function forgetConversation(id: string): void {
  stores.delete(id);
  arrivals.delete(id);
}

function nextFrame(run: () => void): () => void {
  if (typeof requestAnimationFrame !== 'function') {
    const timer = setTimeout(run, 0);
    return () => clearTimeout(timer);
  }
  const frame = requestAnimationFrame(run);
  return () => cancelAnimationFrame(frame);
}

export interface ConversationView extends ConversationSnapshot {
  /** Arrival time of the last stream event, or undefined before the first. */
  asOfMs: number | undefined;
}

/** The open conversation (null on the New screen). */
export function useConversation(id: string | null): ConversationView {
  const pending = useRef<(() => void) | null>(null);
  const subscribe = useCallback(
    (listener: () => void): (() => void) => {
      if (id === null) return () => {};
      const store = storeFor(id);
      const stop = store.subscribe(() => {
        arrivals.set(id, Date.now());
        if (pending.current !== null) return;
        pending.current = nextFrame(() => {
          pending.current = null;
          listener();
        });
      });
      return () => {
        stop();
        pending.current?.();
        pending.current = null;
      };
    },
    [id],
  );
  const getSnapshot = useCallback(() => (id === null ? EMPTY : storeFor(id).getSnapshot()), [id]);
  const snapshot = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return { ...snapshot, asOfMs: id === null ? undefined : arrivals.get(id) };
}
