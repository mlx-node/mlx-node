/**
 * One conversation's stream for the page. The store (`createConversationStore`) opens the stream while it is read,
 * closes it while the window is hidden and re-opens it after a reconnect or a `seq` gap. This hook adds two things:
 * the page is told at most once per animation frame, and the arrival time of the last event (the thinking clock).
 */

import { useCallback, useRef, useSyncExternalStore } from 'react';

import { createConversationStore, type ConversationSnapshot, type ConversationStore } from '../../lib/chat-store';
import type { ConversationEvent } from '../../lib/chat-types';

const stores = new Map<string, ConversationStore>();
const arrivals = new Map<string, number>();
const EMPTY: ConversationSnapshot = { state: null, gone: null };

function storeFor(id: string): ConversationStore {
  let store = stores.get(id);
  if (store === undefined) {
    store = createConversationStore(id);
    stores.set(id, store);
  }
  return store;
}

/**
 * Keeps a conversation's stream open while the page shows another: `listener` gets its state at each change, and
 * once now. Call to stop.
 */
export function watchConversation(id: string, listener: (snapshot: ConversationSnapshot) => void): () => void {
  const store = storeFor(id);
  const stop = store.subscribe(() => listener(store.getSnapshot()));
  listener(store.getSnapshot());
  return stop;
}

/**
 * Each event of a conversation's stream as its store applies it (a `sync` too), while someone else keeps the stream
 * open. Call to stop.
 */
export function listenConversation(id: string, listener: (event: ConversationEvent) => void): () => void {
  return storeFor(id).listen(listener);
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
