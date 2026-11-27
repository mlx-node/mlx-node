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
  CommandView,
  CompactionState,
  CompactOutcome,
  ConversationEvent,
  ConversationInfo,
  ConversationNotice,
  HookRunState,
  Interaction,
  LiveActivity,
  ModelSwitch,
  QueueState,
  RecapEnd,
  RecapState,
  RetryLive,
  RewindEndMark,
  RunView,
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
  /**
   * Requests the agent stopped waiting on (`interaction-closed` as `expired`), questions and MCP cards aside, counted while the page
   * holds the stream.
   */
  withdrawn: number;
  /** Question cards that closed with no answer (Stop, the agent's time limit, no window), counted the same way. */
  unanswered: number;
  /** MCP cards that closed with no answer (Stop, the server stopped waiting, no window), counted the same way. */
  mcpUnanswered: number;
  notices: ConversationNotice[];
  transcriptError: string | null;
  /** How the last turn ended, live or from the `sync`; null while none ended. */
  lastEnd: {
    turnId?: string;
    outcome?: TurnOutcome;
    failure?: ChatFailure;
    unsent?: true;
    model?: ModelSwitch;
    blocked?: string;
  } | null;
  gone: 'deleted' | 'unknown' | null;
  /** Every permission mark of the conversation, oldest first: `sync` replaces them, `permission-mark` appends one. */
  permissionMarks: PermissionMark[];
  /** The conversation's current context in tokens; null while unknown. */
  contextUsed: number | null;
  /** The agent compacts the conversation now; null when it does not. */
  compaction: CompactionState | null;
  /** How the last manual compaction this stream saw ended; a new object per end. Null after a `sync`. */
  compactionEnd: { outcome: CompactOutcome } | null;
  /** The agent waits on a batch of hooks now; null when it does not. */
  hookRun: HookRunState | null;
  /** A recap the user asked for is being written; null while none is. */
  recap: RecapState | null;
  /** How the last recap this stream saw ended; a new object per end. Null after a `sync`. */
  recapEnd: { outcome: RecapEnd } | null;
  /**
   * How the last rewind of this app run ended, numbered: the event and `sync` carry it, and the same end again keeps
   * its object. Null before the first.
   */
  rewindEnd: RewindEndMark | null;
  /** The running answer's last retry episode (its row may not be placed yet); null while it has none. */
  retry: RetryLive | null;
  /** Work: the host's workflow runs, live and newest ended; `sync` and `runs` replace the list. */
  runs: RunView[];
  /** Work: the live facts of the background work (helper progress); `sync` and `activity` replace them. */
  live: LiveActivity;
  /** The queued messages the host holds; `sync` and `queue` replace them. */
  queue: QueueState;
  /** Work: the agent's commands the loaded session may send; null while it holds no list (`sync`, `commands`). */
  commands: CommandView[] | null;
}

/** A conversation with nothing queued and nothing held. */
export const NO_QUEUE: QueueState = { items: [], held: null, count: 0 };

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
    case 'context':
      return;
  }
}

/** `entries` after `ops`, a new list; turn metrics and context ops are left out. */
export function applyEntryOps(entries: readonly TranscriptEntry[], ops: readonly TranscriptOp[]): TranscriptEntry[] {
  const out = [...entries];
  for (const op of ops) applyOp(out, [], op);
  return out;
}

/** Applies a batch of ops to `next`, whose entry and turn lists are still the previous state's. */
function applyOps(next: ConversationState, ops: readonly TranscriptOp[]): void {
  const entries = [...next.entries];
  const turns = [...next.turns];
  for (const op of ops) {
    if (op.op === 'context') next.contextUsed = op.used;
    else applyOp(entries, turns, op);
  }
  next.entries = entries;
  next.turns = turns;
}

function endOf(end: Partial<TurnEnd>): NonNullable<ConversationState['lastEnd']> {
  const out: NonNullable<ConversationState['lastEnd']> = {};
  if (end.turnId !== undefined) out.turnId = end.turnId;
  if (end.outcome !== undefined) out.outcome = end.outcome;
  if (end.failure !== undefined) out.failure = end.failure;
  if (end.unsent === true) out.unsent = true;
  if (end.model !== undefined) out.model = end.model;
  if (end.blocked !== undefined) out.blocked = end.blocked;
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
      withdrawn: state?.withdrawn ?? 0,
      unanswered: state?.unanswered ?? 0,
      mcpUnanswered: state?.mcpUnanswered ?? 0,
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
      contextUsed: event.contextUsed ?? null,
      compaction: event.compaction ?? null,
      compactionEnd: null,
      hookRun: event.hookRun ?? null,
      recap: event.recap ?? null,
      recapEnd: null,
      rewindEnd:
        event.rewindEnd === undefined
          ? null
          : state?.rewindEnd?.n === event.rewindEnd.n
            ? state.rewindEnd
            : { n: event.rewindEnd.n, end: event.rewindEnd.end },
      retry: event.retry ?? null,
      runs: [...event.runs],
      live: event.live,
      queue: event.queue ?? NO_QUEUE,
      commands: event.commands ?? null,
    };
  }
  if (state === null || event.seq !== state.seq + 1) return 'resync';
  const next: ConversationState = { ...state, seq: event.seq };
  switch (event.type) {
    case 'ops':
      applyOps(next, event.ops);
      return next;
    case 'notice':
      next.notices = [...state.notices, event.notice];
      return next;
    case 'notice-cleared':
      next.notices = state.notices.filter((notice) => notice.code !== event.code);
      return next;
    case 'turn':
      next.turn = event.turn;
      if (event.turn === null) next.lastEnd = endOf(event);
      return next;
    case 'interaction':
      next.interactions = [...state.interactions.filter((i) => i.id !== event.interaction.id), event.interaction];
      return next;
    case 'interaction-closed': {
      next.interactions = state.interactions.filter((i) => i.id !== event.id);
      const closed = state.interactions.find((i) => i.id === event.id);
      if (closed?.kind === 'question') {
        if (event.reason !== 'answered') next.unanswered = state.unanswered + 1;
      } else if (closed?.kind === 'mcp-input') {
        if (event.reason !== 'answered') next.mcpUnanswered = state.mcpUnanswered + 1;
      } else if (event.reason === 'expired' && closed !== undefined) {
        next.withdrawn = state.withdrawn + 1;
      }
      return next;
    }
    case 'conversation':
      next.conversation = event.conversation;
      return next;
    case 'gone':
      next.gone = event.reason;
      return next;
    case 'permission-mark':
      next.permissionMarks = [...state.permissionMarks, event.mark];
      return next;
    case 'rule-gone':
      next.permissionMarks = state.permissionMarks.map((mark) =>
        mark.kind === 'denied' && mark.ruleId === event.ruleId ? { ...mark, words: null } : mark,
      );
      return next;
    case 'compaction':
      next.compaction = event.compaction;
      if (event.end !== undefined) next.compactionEnd = { outcome: event.end };
      return next;
    case 'hook-run':
      next.hookRun = event.hookRun;
      return next;
    case 'recap':
      next.recap = event.recap;
      if (event.end !== undefined) next.recapEnd = { outcome: event.end };
      return next;
    case 'rewind':
      next.rewindEnd = { n: event.n, end: event.end };
      return next;
    case 'retry':
      next.retry = event.retry;
      return next;
    case 'runs':
      if (event.ops !== undefined) applyOps(next, event.ops);
      next.runs = [...event.runs];
      return next;
    case 'activity':
      next.live = event.live;
      return next;
    case 'agent-entries':
      return next;
    case 'queue':
      next.queue = event.queue;
      return next;
    case 'commands':
      next.commands = event.commands === null ? null : [...event.commands];
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

/** A conversation's store; `listen` hears each event it applied, in order, and keeps no stream open by itself. */
export interface ConversationStore extends ChatStore<ConversationSnapshot> {
  listen(listener: (event: ConversationEvent) => void): () => void;
}

/**
 * One conversation's stream behind `useSyncExternalStore`. Open while it has readers, the window is visible and the
 * transport is up; re-opened after a reconnect and after a `seq` gap.
 */
export function createConversationStore(conversationId: string): ConversationStore {
  let snapshot: ConversationSnapshot = { state: null, gone: null };
  const listeners = new Set<() => void>();
  const hearers = new Set<(event: ConversationEvent) => void>();
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
    for (const hear of Array.from(hearers)) hear(event);
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
    listen(listener: (event: ConversationEvent) => void): () => void {
      hearers.add(listener);
      return () => {
        hearers.delete(listener);
      };
    },
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
