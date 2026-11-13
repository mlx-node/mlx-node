/**
 * Chat streams for the page, on the pattern of `live.ts`: a stream is open only while someone reads it, a reconnect
 * (new connection generation) re-opens it on the new port, and a conversation stream closes while the window is
 * hidden (the host stream stays open: the nav badge needs it). `applyConversationEvent` is the pure reducer of one
 * conversation; a gap in `seq` asks for a fresh `sync`.
 */

import { subscribeChatHost, subscribeConversation, type Subscription } from './api';
import type {
  PermissionMark,
  ChatFailure,
  ChatHostEvent,
  ChatHostState,
  ConversationEvent,
  ConversationInfo,
  ConversationNotice,
  Interaction,
  TranscriptOp,
  TurnEnd,
  TurnOutcome,
  TurnState,
} from './chat-types';
import { getConnectionGeneration, getTransportUp, subscribeConnection } from './connection';
import type { TranscriptEntry, TurnMetric } from './types';

export interface ConversationState {
  seq: number;
  conversation: ConversationInfo;
  entries: TranscriptEntry[];
  turns: TurnMetric[];
  turn: TurnState | null;
  interactions: Interaction[];
  notices: ConversationNotice[];
  transcriptError: string | null;
  /** How the last turn ended, live or from the `sync`; null while none ended. */
  lastEnd: { turnId?: string; outcome?: TurnOutcome; failure?: ChatFailure; unsent?: true } | null;
  gone: 'deleted' | 'unknown' | null;
  /** Every permission mark of the conversation, oldest first: `sync` replaces them, `permission-mark` appends one. */
  permissionMarks: PermissionMark[];
}

function applyOp(entries: TranscriptEntry[], turns: TurnMetric[], op: TranscriptOp): void {
  switch (op.op) {
    case 'push':
      entries.push(op.entry);
      return;
    case 'text': {
      const entry = entries[op.index];
      if (entry !== undefined && (entry.kind === 'assistant' || entry.kind === 'thinking')) {
        entries[op.index] = {
          ...entry,
          text: entry.text + op.append,
          ...(op.endTs === undefined ? {} : { endTs: op.endTs }),
        };
      }
      return;
    }
    case 'set':
      if (op.index >= 0 && op.index < entries.length) entries[op.index] = op.entry;
      return;
    case 'metric':
      turns.push(op.metric);
      return;
  }
}

function endOf(end: Partial<TurnEnd>): NonNullable<ConversationState['lastEnd']> {
  const out: NonNullable<ConversationState['lastEnd']> = {};
  if (end.turnId !== undefined) out.turnId = end.turnId;
  if (end.outcome !== undefined) out.outcome = end.outcome;
  if (end.failure !== undefined) out.failure = end.failure;
  if (end.unsent === true) out.unsent = true;
  return out;
}

/** The next state, or `'resync'` when the event cannot be applied (no `sync` yet, or a `seq` gap). */
export function applyConversationEvent(
  state: ConversationState | null,
  event: ConversationEvent,
): ConversationState | 'resync' {
  if (event.type === 'sync') {
    return {
      seq: event.seq,
      conversation: event.conversation,
      entries: [...event.entries],
      turns: [...event.turns],
      turn: event.turn,
      interactions: [...event.interactions],
      notices: [...event.notices],
      transcriptError: event.transcriptError,
      // The same end again keeps its object: the page tells a new end from a repeated one by identity.
      lastEnd:
        event.lastEnd === undefined
          ? null
          : state !== null && state.lastEnd !== null && state.lastEnd.turnId === event.lastEnd.turnId
            ? state.lastEnd
            : endOf(event.lastEnd),
      gone: null,
      permissionMarks: [...event.permissionMarks],
    };
  }
  if (state === null || event.seq !== state.seq + 1) return 'resync';
  const next: ConversationState = { ...state, seq: event.seq };
  switch (event.type) {
    case 'ops': {
      const entries = [...state.entries];
      const turns = [...state.turns];
      for (const op of event.ops) applyOp(entries, turns, op);
      next.entries = entries;
      next.turns = turns;
      return next;
    }
    case 'notice':
      next.notices = [...state.notices, event.notice];
      return next;
    case 'turn':
      next.turn = event.turn;
      if (event.turn === null) next.lastEnd = endOf(event);
      return next;
    case 'interaction':
      next.interactions = [...state.interactions.filter((i) => i.id !== event.interaction.id), event.interaction];
      return next;
    case 'interaction-closed':
      next.interactions = state.interactions.filter((i) => i.id !== event.id);
      return next;
    case 'conversation':
      next.conversation = event.conversation;
      return next;
    case 'gone':
      next.gone = event.reason;
      return next;
    case 'permission-mark':
      next.permissionMarks = [...state.permissionMarks, event.mark];
      return next;
  }
}

function visible(): boolean {
  return typeof document === 'undefined' || document.visibilityState === 'visible';
}

/** What a conversation store reads: the state, or `gone` before any `sync` when the id names nothing. */
export interface ConversationSnapshot {
  state: ConversationState | null;
  gone: 'deleted' | 'unknown' | null;
}

export interface ChatStore<T> {
  subscribe(listener: () => void): () => void;
  getSnapshot(): T;
}

/**
 * One conversation's stream behind `useSyncExternalStore`. Open while it has readers, the window is visible and the
 * transport is up; re-opened after a reconnect and after a `seq` gap.
 */
export function createConversationStore(conversationId: string): ChatStore<ConversationSnapshot> {
  let snapshot: ConversationSnapshot = { state: null, gone: null };
  const listeners = new Set<() => void>();
  let subscription: Subscription | null = null;
  let boundGeneration = getConnectionGeneration();
  let detach: (() => void) | null = null;

  const notify = (): void => {
    for (const listener of listeners) listener();
  };

  const close = (): void => {
    subscription?.close();
    subscription = null;
  };

  const onEvent = (event: ConversationEvent): void => {
    if (event.type === 'gone' && snapshot.state === null) {
      snapshot = { state: null, gone: event.reason };
      close();
      notify();
      return;
    }
    const next = applyConversationEvent(snapshot.state, event);
    if (next === 'resync') {
      close();
      queueMicrotask(sync);
      return;
    }
    snapshot = { state: next, gone: next.gone };
    if (next.gone !== null) close();
    notify();
  };

  const sync = (): void => {
    const generation = getConnectionGeneration();
    const wanted = listeners.size > 0 && visible() && getTransportUp() && snapshot.gone === null;
    if (subscription !== null && (generation !== boundGeneration || !wanted)) close();
    boundGeneration = generation;
    if (subscription === null && wanted) {
      try {
        subscription = subscribeConversation(conversationId, onEvent);
      } catch {
        subscription = null;
      }
    }
  };

  return {
    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      if (detach === null) {
        const stopConnection = subscribeConnection(() => queueMicrotask(sync));
        const onVisibility = (): void => sync();
        if (typeof document !== 'undefined') document.addEventListener('visibilitychange', onVisibility);
        detach = () => {
          stopConnection();
          if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', onVisibility);
          detach = null;
        };
      }
      sync();
      return () => {
        listeners.delete(listener);
        sync();
        if (listeners.size === 0) detach?.();
      };
    },
    getSnapshot: () => snapshot,
  };
}

export interface ChatHostSnapshot {
  state: ChatHostState | null;
  /** Bumped by every `conversations-changed`: lists refetch on it. */
  conversationsVersion: number;
}

/** The Chat host state stream. It stays open while the window is hidden; a reconnect re-opens it. */
export function createChatHostStore(): ChatStore<ChatHostSnapshot> {
  let snapshot: ChatHostSnapshot = { state: null, conversationsVersion: 0 };
  const listeners = new Set<() => void>();
  let subscription: Subscription | null = null;
  let boundGeneration = getConnectionGeneration();
  let detach: (() => void) | null = null;

  const notify = (): void => {
    for (const listener of listeners) listener();
  };

  const onEvent = (event: ChatHostEvent): void => {
    snapshot =
      event.type === 'state'
        ? { ...snapshot, state: event.state }
        : { ...snapshot, conversationsVersion: snapshot.conversationsVersion + 1 };
    notify();
  };

  const sync = (): void => {
    const generation = getConnectionGeneration();
    const wanted = listeners.size > 0 && getTransportUp();
    if (subscription !== null && (generation !== boundGeneration || !wanted)) {
      subscription.close();
      subscription = null;
    }
    if (generation !== boundGeneration && snapshot.state !== null) {
      snapshot = { ...snapshot, state: null };
      notify();
    }
    boundGeneration = generation;
    if (subscription === null && wanted) {
      try {
        subscription = subscribeChatHost(onEvent);
      } catch {
        subscription = null;
      }
    }
  };

  return {
    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      if (detach === null) {
        const stopConnection = subscribeConnection(() => queueMicrotask(sync));
        detach = () => {
          stopConnection();
          detach = null;
        };
      }
      sync();
      return () => {
        listeners.delete(listener);
        sync();
        if (listeners.size === 0) detach?.();
      };
    },
    getSnapshot: () => snapshot,
  };
}
