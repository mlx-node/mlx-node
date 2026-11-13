/**
 * One conversation's event stream: the transcript reducer (seeded from disk, then fed live lines), the redacted view
 * every attached page has been sent, the per-conversation `seq`, and the 33 ms op batching. Every string bound for
 * the page passes the redactor; an open text entry holds back its last characters so a secret split over two chunks
 * is still caught.
 */

import { TranscriptReducer, type UpdateLine } from '../agent-sessions/transcript.js';
import type { TranscriptEntry, TurnMetric } from '../agent-sessions/types.js';
import { needleList, redactDeep, redactEntry, StreamRedactor } from './redact.js';
import type {
  PermissionMark,
  ConversationEvent,
  ConversationInfo,
  ConversationNotice,
  Interaction,
  TranscriptOp,
  TurnEnd,
  TurnState,
} from './types.js';

export const CHAT_FLUSH_MS = 33;
/** Permission marks that keep their card text; older ones keep only the metadata. */
export const MARK_CARDS_KEPT = 200;

export type ConversationSink = (event: ConversationEvent) => void;

export interface ConversationDeps {
  /** Redaction needles: provider keys (current and retired) and the inference token. */
  needles: () => string[];
  flushMs?: number;
}

/** The live state the host owns, read when a page attaches. */
export interface HostView {
  turn: TurnState | null;
  interactions: Interaction[];
}

interface Base {
  reducer: TranscriptReducer;
  /** The disk entries the reducer was seeded with (the Chat address ledger is built from them on load). */
  disk: TranscriptEntry[];
  view: TranscriptEntry[];
  turns: TurnMetric[];
  transcriptError: string | null;
}

function isText(entry: TranscriptEntry): entry is Extract<TranscriptEntry, { kind: 'assistant' | 'thinking' }> {
  return entry.kind === 'assistant' || entry.kind === 'thinking';
}

export class Conversation {
  info: ConversationInfo;
  private readonly deps: ConversationDeps;
  private readonly flushMs: number;
  private seq = 0;
  private readonly sinks = new Set<ConversationSink>();
  private base: Base | null = null;
  private version = 0;
  private readonly open = new Map<number, StreamRedactor>();
  /** The needles the open redactors were built with. */
  private openNeedles: readonly string[] = [];
  private queue: TranscriptOp[] = [];
  /** A page that attaches after a turn ended still gets how it ended. */
  private lastEnd: TurnEnd | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  readonly notices: ConversationNotice[] = [];
  private closed = false;
  private marks: PermissionMark[] = [];
  /** Marks that hold card text, and the index from which the oldest of them is looked for. */
  private cards = 0;
  private cardFrom = 0;
  /** The permission history file was read (or the conversation is new in this app run). */
  marksLoaded = false;

  constructor(info: ConversationInfo, deps: ConversationDeps) {
    this.info = { ...info };
    this.deps = deps;
    this.flushMs = deps.flushMs ?? CHAT_FLUSH_MS;
  }

  get listeners(): number {
    return this.sinks.size;
  }

  /** Changes when the base is replaced and when the user adds to it (a prompt, a model switch). */
  get baseVersion(): number {
    return this.version;
  }

  hasBase(): boolean {
    return this.base !== null;
  }

  /** The disk entries of the current base, or undefined when there is none. */
  diskEntries(): TranscriptEntry[] | undefined {
    return this.base?.disk;
  }

  /** Starts the transcript from what is on disk (empty for a new conversation). Live ops index after it. */
  seed(entries: readonly TranscriptEntry[], turns: readonly TurnMetric[], transcriptError: string | null): void {
    this.version += 1;
    const reducer = new TranscriptReducer();
    reducer.seed(entries, turns);
    const needles = this.needles();
    this.open.clear();
    this.queue = [];
    this.clearTimer();
    this.base = {
      reducer,
      disk: entries.map((e) => ({ ...e })),
      view: entries.map((e) => redactEntry(e, needles)),
      turns: redactDeep(
        turns.map((t) => ({ ...t })),
        needles,
      ),
      transcriptError,
    };
  }

  dropBase(): void {
    this.version += 1;
    this.base = null;
    this.open.clear();
    this.queue = [];
    this.clearTimer();
  }

  /** Adds a page. It gets a complete `sync` first, carrying the current `seq`. */
  attach(sink: ConversationSink, host: HostView): () => void {
    if (this.base !== null) {
      const fixes: TranscriptOp[] = [];
      this.rekeyOpen(this.needles(), fixes);
      this.commit(fixes);
    }
    this.flush();
    this.sinks.add(sink);
    this.safeSend(sink, this.syncEvent(host, this.seq));
    return () => {
      this.sinks.delete(sink);
      if (this.sinks.size === 0) {
        this.queue = [];
        this.clearTimer();
      }
    };
  }

  /** A fresh `sync` to every attached page (after a hot → cold switch re-read the disk). */
  resync(host: HostView): void {
    this.flush();
    if (this.base !== null) {
      const fixes: TranscriptOp[] = [];
      this.rekeyOpen(this.needles(), fixes);
      for (const op of fixes) applyOp(this.base.view, this.base.turns, op);
    }
    this.broadcast(this.syncEvent(host, this.seq + 1), true);
  }

  /** A live update line from the agent. */
  applyLine(line: UpdateLine): void {
    if (this.base === null) return;
    this.emitOps(this.base.reducer.apply(line));
  }

  /** The user's prompt, pushed when it is accepted (the agent does not echo it). */
  userPrompt(text: string): void {
    this.lastEnd = null;
    if (this.base === null) return;
    this.version += 1;
    this.emitOps(this.base.reducer.userPrompt(text));
  }

  /** A model switch the user made between turns. */
  modelSwitch(modelId: string): void {
    if (this.base === null) return;
    this.version += 1;
    this.emitOps(this.base.reducer.modelSwitch(modelId));
  }

  setInfo(patch: Partial<ConversationInfo>): void {
    this.info = { ...this.info, ...patch };
    this.event({ type: 'conversation', seq: 0, conversation: this.publicInfo() });
  }

  turnChanged(turn: TurnState): void {
    this.event({ type: 'turn', seq: 0, turn });
  }

  turnEnded(end: TurnEnd): void {
    this.closeOpenText();
    const lastEnd: TurnEnd = { turnId: end.turnId };
    if (end.outcome !== undefined) lastEnd.outcome = end.outcome;
    if (end.failure !== undefined) lastEnd.failure = end.failure;
    if (end.unsent === true) lastEnd.unsent = true;
    this.lastEnd = lastEnd;
    this.event({ type: 'turn', seq: 0, turn: null, ...lastEnd });
  }

  interaction(interaction: Interaction): void {
    this.event({ type: 'interaction', seq: 0, interaction: redactDeep(interaction, this.needles()) });
  }

  interactionClosed(id: string, reason: 'answered' | 'cancelled' | 'expired'): void {
    this.event({ type: 'interaction-closed', seq: 0, id, reason });
  }

  /** Every permission mark, oldest first. */
  permissionMarks(): readonly PermissionMark[] {
    return this.marks;
  }

  /** A permission mark: kept (card text for the newest {@link MARK_CARDS_KEPT}) and sent, redacted. */
  permissionMark(mark: PermissionMark): void {
    const kept = redactDeep(mark, this.needles());
    this.marks.push(kept);
    if (kept.kind === 'allowed' && kept.card !== null) this.cards += 1;
    while (this.cards > MARK_CARDS_KEPT && this.cardFrom < this.marks.length) {
      const old = this.marks[this.cardFrom]!;
      if (old.kind === 'allowed' && old.card !== null) {
        this.marks[this.cardFrom] = { ...old, card: null };
        this.cards -= 1;
      }
      this.cardFrom += 1;
    }
    this.event({ type: 'permission-mark', seq: 0, mark: kept });
  }

  /** Each notice once: the same code for the same provider is not repeated. */
  notice(notice: ConversationNotice): void {
    if (this.notices.some((n) => n.code === notice.code && n.provider === notice.provider)) return;
    this.notices.push(notice);
    this.event({ type: 'notice', seq: 0, notice });
  }

  /** The conversation is deleted (or never existed): every page is told, and the stream ends. */
  gone(reason: 'deleted' | 'unknown'): void {
    this.event({ type: 'gone', seq: 0, reason });
    this.closed = true;
    this.sinks.clear();
    this.clearTimer();
  }

  publicInfo(): ConversationInfo {
    return redactDeep({ ...this.info }, this.needles());
  }

  /** Sends what is queued now. */
  flush(): void {
    this.clearTimer();
    if (this.queue.length === 0) return;
    const ops = mergeTextOps(this.queue);
    this.queue = [];
    this.broadcast({ type: 'ops', seq: 0, ops }, true);
  }

  // ── internals ───────────────────────────────────────────────────────────────────────────────────────────────────

  private needles(): string[] {
    return needleList(this.deps.needles());
  }

  private syncEvent(host: HostView, seq: number): ConversationEvent {
    const needles = this.needles();
    return {
      type: 'sync',
      seq,
      conversation: this.publicInfo(),
      entries: this.base === null ? [] : this.base.view.map((e) => redactEntry(e, needles)),
      turns:
        this.base === null
          ? []
          : redactDeep(
              this.base.turns.map((t) => ({ ...t })),
              needles,
            ),
      turn: host.turn,
      interactions: redactDeep(host.interactions, needles),
      notices: [...this.notices],
      transcriptError: this.base?.transcriptError ?? null,
      permissionMarks: redactDeep(
        this.marks.map((m) => ({ ...m })),
        needles,
      ),
      ...(this.lastEnd === null ? {} : { lastEnd: { ...this.lastEnd } }),
    };
  }

  /** Redacts and holds back text, applies the result to the view, and queues it for the next flush. */
  private emitOps(ops: TranscriptOp[]): void {
    if (this.base === null || ops.length === 0) return;
    const needles = this.needles();
    const out: TranscriptOp[] = [];
    this.rekeyOpen(needles, out);
    for (const op of ops) {
      if (op.op === 'push') {
        this.closeOpenTextInto(out);
        const entry = redactEntry(op.entry, needles);
        const index = this.base.view.length + countPushes(out);
        if (isText(entry)) {
          const redactor = new StreamRedactor(needles);
          entry.text = redactor.push((op.entry as typeof entry).text);
          this.open.set(index, redactor);
        }
        out.push({ op: 'push', entry });
      } else if (op.op === 'text') {
        let redactor = this.open.get(op.index);
        if (redactor === undefined) {
          const shown = this.base.view[op.index];
          redactor = new StreamRedactor(needles, shown !== undefined && isText(shown) ? shown.text : '');
          this.open.set(op.index, redactor);
        }
        const safe = redactor.push(op.append);
        if (safe !== '') out.push({ op: 'text', index: op.index, append: safe });
      } else if (op.op === 'set') {
        this.open.delete(op.index);
        out.push({ op: 'set', index: op.index, entry: redactEntry(op.entry, needles) });
      } else {
        this.closeOpenTextInto(out);
        out.push({ op: 'metric', metric: redactDeep({ ...op.metric }, needles) });
      }
    }
    this.commit(out);
  }

  private closeOpenText(): void {
    const out: TranscriptOp[] = [];
    this.rekeyOpen(this.needles(), out);
    this.closeOpenTextInto(out);
    this.commit(out);
  }

  /**
   * A key that became known while a text entry streams: the whole entry is rescanned, and a `set` op replaces the
   * text the page already has (part of the key may have been sent before it was a needle).
   */
  private rekeyOpen(needles: readonly string[], out: TranscriptOp[]): void {
    if (needles.length === this.openNeedles.length && needles.every((n, i) => n === this.openNeedles[i])) return;
    this.openNeedles = needles;
    if (this.base === null) return;
    for (const [index, redactor] of this.open) {
      const next = redactor.rekey(needles);
      this.open.set(index, next.redactor);
      const entry = this.base.view[index];
      if (entry !== undefined && isText(entry) && entry.text !== next.text) {
        out.push({ op: 'set', index, entry: { ...entry, text: next.text } });
      }
    }
  }

  private closeOpenTextInto(out: TranscriptOp[]): void {
    for (const [index, redactor] of this.open) {
      const rest = redactor.flush();
      if (rest !== '') out.push({ op: 'text', index, append: rest });
    }
    this.open.clear();
  }

  private commit(ops: TranscriptOp[]): void {
    if (this.base === null || ops.length === 0) return;
    for (const op of ops) applyOp(this.base.view, this.base.turns, op);
    if (this.sinks.size === 0 || this.closed) return;
    this.queue.push(...ops);
    if (this.timer === null) {
      this.timer = setTimeout(() => {
        this.timer = null;
        this.flush();
      }, this.flushMs);
    }
  }

  /** Every non-op event: the queue goes first, then the event at once. */
  private event(event: ConversationEvent): void {
    if (this.closed) return;
    this.flush();
    this.broadcast(event, true);
  }

  private broadcast(event: ConversationEvent, numbered: boolean): void {
    if (this.sinks.size === 0) return;
    if (numbered) this.seq += 1;
    const out = { ...event, seq: this.seq } as ConversationEvent;
    for (const sink of Array.from(this.sinks)) this.safeSend(sink, out);
  }

  private safeSend(sink: ConversationSink, event: ConversationEvent): void {
    try {
      sink(event);
    } catch {
      this.sinks.delete(sink);
    }
  }

  private clearTimer(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }
}

function countPushes(ops: readonly TranscriptOp[]): number {
  let n = 0;
  for (const op of ops) if (op.op === 'push') n++;
  return n;
}

/** Applies one op to an entry list, as the page does. */
export function applyOp(entries: TranscriptEntry[], turns: TurnMetric[], op: TranscriptOp): void {
  switch (op.op) {
    case 'push':
      entries.push({ ...op.entry });
      return;
    case 'text': {
      const entry = entries[op.index];
      if (entry !== undefined && isText(entry)) entries[op.index] = { ...entry, text: entry.text + op.append };
      return;
    }
    case 'set':
      if (op.index >= 0 && op.index < entries.length) entries[op.index] = { ...op.entry };
      return;
    case 'metric':
      turns.push({ ...op.metric });
      return;
  }
}

/** Consecutive `text` ops on one index become one. */
export function mergeTextOps(ops: readonly TranscriptOp[]): TranscriptOp[] {
  const out: TranscriptOp[] = [];
  for (const op of ops) {
    const last = out[out.length - 1];
    if (op.op === 'text' && last !== undefined && last.op === 'text' && last.index === op.index) {
      out[out.length - 1] = { op: 'text', index: op.index, append: last.append + op.append };
    } else {
      out.push(op);
    }
  }
  return out;
}
