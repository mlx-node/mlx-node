/**
 * The one Chat host stream of the window, shared by the Chat page, the sidebar badge and the palette. It is pushed
 * (`state`, `conversations-changed`) and re-attached after a reconnect by `createChatHostStore`; nothing polls.
 */

import { useSyncExternalStore } from 'react';

import { createChatHostStore, type ChatHostSnapshot } from '../../lib/chat-store';

const store = createChatHostStore();
const subscribe = (listener: () => void): (() => void) => store.subscribe(listener);
const getSnapshot = (): ChatHostSnapshot => store.getSnapshot();

export function useChatHost(): ChatHostSnapshot {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/** The latest snapshot, for code outside render (the palette's conversation rows). */
export function chatHostSnapshot(): ChatHostSnapshot {
  return store.getSnapshot();
}

/** Turns that run and conversations that wait for the user, over both modes. Null before the first `state`. */
export function chatActivity(snapshot: ChatHostSnapshot): { running: number; waiting: number } | null {
  const state = snapshot.state;
  if (state === null) return null;
  let running = 0;
  let waiting = 0;
  for (const item of state.active) {
    if (item.waiting) waiting += 1;
    else if (item.busy) running += 1;
  }
  return { running, waiting };
}
