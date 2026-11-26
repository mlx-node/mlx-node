/**
 * Work: the Agent view's live data. The host keeps each running agent's own live lines (the newest
 * {@link AGENT_LINES_BYTES}, until {@link AGENT_LINES_KEEP_MS} past its end). A watch of a window reads the agent's
 * `updates.jsonl` once, applies the kept lines past the seam, and runs each later line through the same reducer; at
 * the agent's end it sends the stored list once the file holds the agent's turn end. Every entry and op is redacted,
 * and an open text holds back an end that could start a key.
 */

import { lstat, readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { MAX_UPDATES_BYTES } from '../../agent-sessions/reader.js';
import { parseUpdates, storedEventIds, TranscriptReducer, type UpdateLine } from '../../agent-sessions/transcript.js';
import type { TranscriptEntry } from '../../agent-sessions/types.js';
import { applyOp, CHAT_FLUSH_MS, mergeTextOps } from '../conversation.js';
import { needleList, redactEntry, StreamRedactor } from '../redact.js';
import type { ConversationEvent, TranscriptOp } from '../types.js';

/** A running agent's live lines the host keeps for a watch that starts later: the newest, at most this many bytes. */
export const AGENT_LINES_BYTES = 512 * 1024;
/** Kept lines stay this long past the agent's end. */
export const AGENT_LINES_KEEP_MS = 30_000;
/** At the agent's end its file is read again this often, until it holds the agent's turn end. */
export const AGENT_SWAP_EVERY_MS = 250;
/** The file is read again for at most this long; then the list is the file and the kept lines past the seam. */
export const AGENT_SWAP_MAX_MS = 5_000;
/** Live watches one window may hold. */
export const AGENT_WATCHES_PER_WINDOW = 2;

export type AgentEntriesEvent = Extract<ConversationEvent, { type: 'agent-entries' }>;

export interface AgentWatchTimings {
  keepMs: number;
  swapEveryMs: number;
  swapMaxMs: number;
  flushMs: number;
}

export interface AgentWatchDeps {
  /** The agent's `updates.jsonl` text; '' when it has none yet or it cannot be read. */
  read(childId: string): Promise<string>;
  /** Redaction needles: provider keys (current and retired) and the inference token. */
  needles(): string[];
  /** An event for the pages of the session's conversation. */
  send(sessionId: string, event: AgentEntriesEvent): void;
  now?: () => number;
  timings?: Partial<AgentWatchTimings>;
}

/** What a watch answers: the list, and whether the window gets the agent's later lines. */
export interface AgentWatchStart {
  entries: TranscriptEntry[];
  live: boolean;
}

/** The agent's `updates.jsonl` in its session folder; '' when there is none, it is too large, or it cannot be read. */
export async function readAgentUpdates(dir: string | null): Promise<string> {
  if (dir === null) return '';
  const path = join(dir, 'updates.jsonl');
  try {
    const st = await lstat(path);
    if (!st.isFile() || st.size > MAX_UPDATES_BYTES) return '';
    return await readFile(path, 'utf8');
  } catch {
    return '';
  }
}

const TEXT_CHUNKS: ReadonlySet<string> = new Set(['agent_message_chunk', 'agent_thought_chunk', 'user_message_chunk']);

function eventIdOf(line: UpdateLine): string | null {
  const id = line.meta?.eventId;
  return typeof id === 'string' && id !== '' ? id : null;
}

/**
 * The kept lines a watch applies over the file. A line whose `eventId` the file holds is in it, and so is a text chunk
 * before the last ACP line the file holds: the agent stores a run of text chunks as one line that takes the newest
 * chunk's `_meta`, and holds each ACP line until the next one comes, while an xAI line is written at once.
 */
export function pastSeam(lines: readonly UpdateLine[], stored: ReadonlySet<string>): UpdateLine[] {
  let seam = -1;
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]!;
    const id = eventIdOf(line);
    if (!line.xai && id !== null && stored.has(id)) {
      seam = i;
      break;
    }
  }
  const out: UpdateLine[] = [];
  for (const [i, line] of lines.entries()) {
    const id = eventIdOf(line);
    if (id !== null && stored.has(id)) continue;
    if (i < seam && TEXT_CHUNKS.has(String(line.update.sessionUpdate))) continue;
    out.push(line);
  }
  return out;
}

function isText(entry: TranscriptEntry): entry is Extract<TranscriptEntry, { kind: 'assistant' | 'thinking' }> {
  return entry.kind === 'assistant' || entry.kind === 'thinking';
}

function sameNeedles(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((n, i) => n === b[i]);
}

/** A list as the page holds it: redacted, each open text holding back an end that could start a key. */
class ShownList {
  readonly entries: TranscriptEntry[];
  private readonly open = new Map<number, StreamRedactor>();
  private needles: readonly string[];

  constructor(entries: readonly TranscriptEntry[], needles: readonly string[]) {
    this.needles = needles;
    this.entries = entries.map((entry) => redactEntry(entry, needles));
  }

  /** The reducer's ops as the page may get them, applied here too. The view shows no turn metrics or context. */
  take(ops: readonly TranscriptOp[], needles: readonly string[]): TranscriptOp[] {
    const out: TranscriptOp[] = [];
    this.rekey(needles, out);
    let pushed = 0;
    for (const op of ops) {
      if (op.op === 'push') {
        this.closeInto(out);
        const entry = redactEntry(op.entry, this.needles);
        if (isText(entry) && isText(op.entry)) {
          const redactor = new StreamRedactor(this.needles);
          entry.text = redactor.push(op.entry.text);
          this.open.set(this.entries.length + pushed, redactor);
        }
        pushed += 1;
        out.push({ op: 'push', entry });
      } else if (op.op === 'text') {
        let redactor = this.open.get(op.index);
        if (redactor === undefined) {
          const shown = this.entries[op.index];
          redactor = new StreamRedactor(this.needles, shown !== undefined && isText(shown) ? shown.text : '');
          this.open.set(op.index, redactor);
        }
        const safe = redactor.push(op.append);
        if (safe !== '' || op.endTs !== undefined) {
          out.push({
            op: 'text',
            index: op.index,
            append: safe,
            ...(op.endTs === undefined ? {} : { endTs: op.endTs }),
          });
        }
      } else if (op.op === 'set') {
        this.open.delete(op.index);
        out.push({ op: 'set', index: op.index, entry: redactEntry(op.entry, this.needles) });
      }
    }
    for (const op of out) applyOp(this.entries, [], op);
    return out;
  }

  /** Every held-back end, redacted: each text is whole. */
  close(): TranscriptOp[] {
    const out: TranscriptOp[] = [];
    this.closeInto(out);
    for (const op of out) applyOp(this.entries, [], op);
    return out;
  }

  /** A key that became known while a text streams: the whole text is scanned again and replaces what was sent. */
  private rekey(needles: readonly string[], out: TranscriptOp[]): void {
    if (sameNeedles(needles, this.needles)) return;
    this.needles = needles;
    for (const [index, redactor] of this.open) {
      const next = redactor.rekey(needles);
      this.open.set(index, next.redactor);
      const entry = this.entries[index];
      if (entry !== undefined && isText(entry) && entry.text !== next.text) {
        out.push({ op: 'set', index, entry: { ...entry, text: next.text } });
      }
    }
  }

  private closeInto(out: TranscriptOp[]): void {
    for (const [index, redactor] of this.open) {
      const rest = redactor.flush();
      if (rest !== '') out.push({ op: 'text', index, append: rest });
    }
    this.open.clear();
  }
}

interface Kept {
  sessionId: string;
  lines: UpdateLine[];
  sizes: number[];
  bytes: number;
  /** The `eventId` of the newest `turn_completed` it sent. */
  turnEnd: string | null;
  /** Set at its end: the kept lines go when it fires. */
  drop: ReturnType<typeof setTimeout> | null;
}

interface Watch {
  sessionId: string;
  childId: string;
  window: string;
  reducer: TranscriptReducer;
  list: ShownList;
  queue: TranscriptOp[];
  flush: ReturnType<typeof setTimeout> | null;
  /** When the agent's end came; null while it runs. */
  endedAt: number | null;
  retry: ReturnType<typeof setTimeout> | null;
}

function keyOf(window: string, childId: string): string {
  return `${window}\u0000${childId}`;
}

export class AgentWatches {
  private readonly deps: AgentWatchDeps;
  private readonly timings: AgentWatchTimings;
  private readonly now: () => number;
  /** By child session id. */
  private readonly kept = new Map<string, Kept>();
  /** By window and child session id. */
  private readonly watches = new Map<string, Watch>();

  constructor(deps: AgentWatchDeps) {
    this.deps = deps;
    this.now = deps.now ?? Date.now;
    this.timings = {
      keepMs: AGENT_LINES_KEEP_MS,
      swapEveryMs: AGENT_SWAP_EVERY_MS,
      swapMaxMs: AGENT_SWAP_MAX_MS,
      flushMs: CHAT_FLUSH_MS,
      ...deps.timings,
    };
  }

  /**
   * A live line of one of the session's agents (its own session's line). Kept from the first line it sends while it
   * runs, and sent on to its watches. A line with no `eventId` is never stored, so no view holds it.
   */
  line(sessionId: string, childId: string, line: UpdateLine, running: boolean): void {
    const id = eventIdOf(line);
    if (id === null) return;
    let kept = this.kept.get(childId);
    if (kept === undefined && running) {
      kept = { sessionId, lines: [], sizes: [], bytes: 0, turnEnd: null, drop: null };
      this.kept.set(childId, kept);
    }
    if (kept !== undefined) {
      const size = Buffer.byteLength(JSON.stringify(line.update), 'utf8');
      kept.lines.push(line);
      kept.sizes.push(size);
      kept.bytes += size;
      while (kept.bytes > AGENT_LINES_BYTES && kept.lines.length > 1) {
        kept.lines.shift();
        kept.bytes -= kept.sizes.shift()!;
      }
      if (line.xai && line.update.sessionUpdate === 'turn_completed') kept.turnEnd = id;
    }
    for (const w of this.watchesOf(childId)) this.queue(w, w.list.take(w.reducer.apply(line), this.needles()));
  }

  /** Every agent of the session that is kept or watched and that `running` says runs no more has ended. */
  settle(sessionId: string, running: (childId: string) => boolean): void {
    const ended = new Set<string>();
    for (const [child, kept] of this.kept) {
      if (kept.sessionId === sessionId && kept.drop === null && !running(child)) ended.add(child);
    }
    for (const w of this.watches.values()) {
      if (w.sessionId === sessionId && w.endedAt === null && !running(w.childId)) ended.add(w.childId);
    }
    for (const child of ended) this.ended(child);
  }

  /** The agent ended: its kept lines stay {@link AGENT_LINES_KEEP_MS}, and each of its watches swaps to the stored list. */
  ended(childId: string): void {
    const kept = this.kept.get(childId);
    if (kept !== undefined && kept.drop === null) {
      kept.drop = setTimeout(() => {
        if (this.kept.get(childId) === kept) this.kept.delete(childId);
      }, this.timings.keepMs);
      kept.drop.unref?.();
    }
    for (const w of this.watchesOf(childId)) {
      if (w.endedAt !== null) continue;
      w.endedAt = this.now();
      void this.swap(w);
    }
  }

  /**
   * One agent's list for `window`: its file, then its kept lines past the seam. While `running()` holds once the file
   * is read, the window gets its later lines as `agent-entries` ops (in place of a watch of the same window and agent);
   * else the list is all there is. `too-many` when the window holds {@link AGENT_WATCHES_PER_WINDOW} other watches.
   */
  async watch(
    sessionId: string,
    childId: string,
    window: string,
    running: () => boolean,
  ): Promise<AgentWatchStart | 'too-many'> {
    const key = keyOf(window, childId);
    if (running() && this.full(window, key)) return 'too-many';
    const text = await this.deps.read(childId);
    const { reducer, list } = this.listOf(childId, text);
    if (!running()) {
      list.close();
      return { entries: [...list.entries], live: false };
    }
    if (this.full(window, key)) return 'too-many';
    const old = this.watches.get(key);
    if (old !== undefined) this.remove(old);
    this.watches.set(key, {
      sessionId,
      childId,
      window,
      reducer,
      list,
      queue: [],
      flush: null,
      endedAt: null,
      retry: null,
    });
    return { entries: [...list.entries], live: true };
  }

  /** The window's view of the agent closed; true when it was watched. */
  unwatch(childId: string, window: string): boolean {
    const w = this.watches.get(keyOf(window, childId));
    if (w === undefined) return false;
    this.remove(w);
    return true;
  }

  /** No page shows the session's conversation now: its watches go. */
  unwatchSession(sessionId: string): void {
    for (const w of this.watches.values()) if (w.sessionId === sessionId) this.remove(w);
  }

  /** The host no longer holds the session: its watches and kept lines go. */
  dropSession(sessionId: string): void {
    this.unwatchSession(sessionId);
    for (const [child, kept] of this.kept) {
      if (kept.sessionId !== sessionId) continue;
      if (kept.drop !== null) clearTimeout(kept.drop);
      this.kept.delete(child);
    }
  }

  clear(): void {
    for (const w of this.watches.values()) this.remove(w);
    for (const kept of this.kept.values()) if (kept.drop !== null) clearTimeout(kept.drop);
    this.kept.clear();
  }

  private needles(): string[] {
    return needleList(this.deps.needles());
  }

  private watchesOf(childId: string): Watch[] {
    const out: Watch[] = [];
    for (const w of this.watches.values()) if (w.childId === childId) out.push(w);
    return out;
  }

  private full(window: string, key: string): boolean {
    let n = 0;
    for (const [k, w] of this.watches) if (w.window === window && k !== key) n++;
    return n >= AGENT_WATCHES_PER_WINDOW;
  }

  /** The file's list, then the kept lines past the seam, through one reducer that later lines go on with. */
  private listOf(childId: string, text: string): { reducer: TranscriptReducer; list: ShownList } {
    const needles = this.needles();
    const reducer = new TranscriptReducer({ helper: true, sessionId: childId });
    reducer.applyText(text);
    const list = new ShownList(reducer.snapshot().entries, needles);
    const kept = this.kept.get(childId);
    if (kept !== undefined) {
      for (const line of pastSeam(kept.lines, storedEventIds(text).ids)) list.take(reducer.apply(line), needles);
    }
    return { reducer, list };
  }

  /**
   * The end swap: the stored list once the file holds the agent's turn end (the one it sent live, else one of its own
   * lines); past {@link AGENT_SWAP_MAX_MS}, the file and the kept lines past the seam.
   */
  private async swap(w: Watch): Promise<void> {
    const text = await this.deps.read(w.childId);
    if (this.watches.get(keyOf(w.window, w.childId)) !== w) return;
    const stored = storedEventIds(text);
    const turnEnd = this.kept.get(w.childId)?.turnEnd ?? null;
    const own = `${w.childId}-`;
    const done = turnEnd !== null ? stored.ids.has(turnEnd) : stored.turnEnds.some((id) => id.startsWith(own));
    if (!done && this.now() - (w.endedAt ?? 0) < this.timings.swapMaxMs) {
      w.retry = setTimeout(() => {
        w.retry = null;
        void this.swap(w);
      }, this.timings.swapEveryMs);
      w.retry.unref?.();
      return;
    }
    this.remove(w);
    let entries: TranscriptEntry[];
    if (done) {
      const needles = this.needles();
      entries = parseUpdates(text, { helper: true, sessionId: w.childId }).transcript.map((e) =>
        redactEntry(e, needles),
      );
    } else {
      const { list } = this.listOf(w.childId, text);
      list.close();
      entries = list.entries;
    }
    this.deps.send(w.sessionId, {
      type: 'agent-entries',
      seq: 0,
      window: w.window,
      childId: w.childId,
      ops: [],
      stored: entries,
    });
  }

  private queue(w: Watch, ops: TranscriptOp[]): void {
    if (ops.length === 0) return;
    w.queue.push(...ops);
    if (w.flush !== null) return;
    w.flush = setTimeout(() => {
      w.flush = null;
      if (this.watches.get(keyOf(w.window, w.childId)) !== w || w.queue.length === 0) return;
      const merged = mergeTextOps(w.queue);
      w.queue = [];
      this.deps.send(w.sessionId, { type: 'agent-entries', seq: 0, window: w.window, childId: w.childId, ops: merged });
    }, this.timings.flushMs);
    w.flush.unref?.();
  }

  private remove(w: Watch): void {
    if (w.flush !== null) clearTimeout(w.flush);
    if (w.retry !== null) clearTimeout(w.retry);
    w.flush = null;
    w.retry = null;
    w.queue = [];
    const key = keyOf(w.window, w.childId);
    if (this.watches.get(key) === w) this.watches.delete(key);
  }
}
