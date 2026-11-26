/**
 * One conversation's event stream: the transcript reducer (seeded from disk, then fed live lines), the redacted view
 * every attached page has been sent, the per-conversation `seq`, and the 33 ms op batching. Every string bound for
 * the page passes the redactor; an open text entry holds back its last characters so a secret split over two chunks
 * is still caught.
 */

import { mergeRunState, type RunState } from '../agent-sessions/run-state.js';
import { TranscriptReducer, type UpdateLine } from '../agent-sessions/transcript.js';
import type { TranscriptEntry, TurnMetric } from '../agent-sessions/types.js';
import { shownCut } from './agent/permissions.js';
import { heldImages, type UserAttachment } from './attachment-rule.js';
import {
  needleList,
  redactCutText,
  redactDeep,
  redactEntry,
  redactText,
  redactTitle,
  StreamRedactor,
} from './redact.js';
import type {
  AgentDoing,
  CommandView,
  CompactionState,
  CompactOutcome,
  HookRunState,
  PermissionMark,
  ConversationEvent,
  ConversationInfo,
  ConversationNotice,
  Interaction,
  LiveActivity,
  QueueState,
  RecapEnd,
  RecapState,
  RetryLive,
  RewindEnd,
  RewindEndMark,
  RunView,
  TaskEvent,
  TranscriptOp,
  TurnEnd,
  TurnState,
} from './types.js';

export const CHAT_FLUSH_MS = 33;
/** Permission marks that keep their card text; older ones keep only the metadata. */
export const MARK_CARDS_KEPT = 200;
/** A command's words and hint as the list shows them: one line, cut at this many characters. */
export const COMMAND_TEXT_MAX = 80;

export type ConversationSink = (event: ConversationEvent) => void;

export interface ConversationDeps {
  /** Redaction needles: provider keys (current and retired) and the inference token. */
  needles: () => string[];
  flushMs?: number;
  /** Work: the folder is one the app made for a conversation started without a folder. */
  noFolder?: (cwd: string) => boolean;
}

/** The live state the host owns, read when a page attaches. */
export interface HostView {
  turn: TurnState | null;
  interactions: Interaction[];
  compaction: CompactionState | null;
  /** The agent waits on a batch of hooks; null while it does not. */
  hookRun: HookRunState | null;
  /** A recap the user asked for is being written; null while none is. */
  recap: RecapState | null;
  runs: RunView[];
  live: LiveActivity;
  /** The loaded session's commands; null while it holds no list. */
  commands: CommandView[] | null;
}

interface Base {
  reducer: TranscriptReducer;
  /** The disk entries the reducer was seeded with (the Chat address ledger is built from them on load). */
  disk: TranscriptEntry[];
  view: TranscriptEntry[];
  turns: TurnMetric[];
  transcriptError: string | null;
  /** The conversation's current context in tokens, as the page was sent it; null while unknown. */
  contextUsed: number | null;
}

/** A monitor's output: a key cut in two by the host's cut no longer matches, so a cut end that could start one goes. */
function redactEvent(event: TaskEvent, needles: readonly string[]): TaskEvent {
  if (event.cut !== true) return { at: event.at, text: redactText(event.text, needles) };
  return { at: event.at, text: `${redactCutText(event.text.slice(0, -1), needles)}…`, cut: true };
}

/** What a helper does now: a tool call's title is agent text cut to length, so it is redacted as a cut title. */
function redactDoing(now: AgentDoing | null, needles: readonly string[]): AgentDoing | null {
  if (now === null) return null;
  return now.kind === 'tool' ? { kind: 'tool', title: redactTitle(now.title, needles) } : { ...now };
}

function liveCopy(live: LiveActivity, needles: readonly string[]): LiveActivity {
  return {
    helpers: live.helpers.map((h) => ({ ...h, now: redactDoing(h.now, needles) })),
    agents: live.agents.map((a) => ({ ...a, attempts: [...a.attempts] })),
    tasks: live.tasks.map((t) => ({
      ...t,
      recent: t.recent.map((event) => redactEvent(event, needles)),
    })),
    schedules: live.schedules.map((schedule) => ({ ...schedule })),
    reports: live.reports.map((report) => ({
      wakeTs: report.wakeTs,
      monitors: report.monitors.map((m) => ({ id: m.id, description: redactText(m.description, needles) })),
    })),
    stoppedByYou: [...live.stoppedByYou],
  };
}

/** The text up to its first line break. */
function firstLine(text: string): string {
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code === 0x0a || code === 0x0d) return text.slice(0, i);
  }
  return text;
}

/** Agent text of the command list as the page may draw it: a name with a key in it is dropped. */
function shownCommands(commands: readonly CommandView[], needles: readonly string[]): CommandView[] {
  const out: CommandView[] = [];
  for (const row of commands) {
    if (redactText(row.name, needles) !== row.name) continue;
    out.push({
      name: row.name,
      kind: row.kind,
      description: shownCut(firstLine(row.description), COMMAND_TEXT_MAX, needles),
      hint: row.hint === null ? null : shownCut(firstLine(row.hint), COMMAND_TEXT_MAX, needles),
    });
  }
  return out;
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
  /** The queued messages as the host holds them, redacted only on the way out (a key can become known later). */
  private queued: QueueState = { items: [], held: null, count: 0 };
  /** The last rewind end of this app run, numbered from 1. */
  private lastRewind: RewindEndMark | null = null;
  /** The running answer's last retry episode the pages were last sent (redacted), or null. */
  private retryShown: RetryLive | null = null;
  /** The view index and text of the prompt the user typed last in this base; null after a reseed. */
  private typed: { index: number; text: string } | null = null;
  /** The view indices of prompts whose turn ended before they went to the agent: no rewind or fork point. */
  private unsent = new Set<number>();

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

  /** What the user messages of the current base carry, live ones included. */
  userAttachments(): UserAttachment[] {
    const out: UserAttachment[] = [];
    for (const entry of this.base?.view ?? []) {
      if (entry.kind === 'user' && entry.attachments !== undefined) out.push(...entry.attachments);
    }
    return out;
  }

  /** The images the current base holds ({@link heldImages}), live ones included. */
  heldImages(): number {
    return heldImages(this.base?.view ?? []);
  }

  /** The disk entries of the current base, or undefined when there is none. */
  diskEntries(): TranscriptEntry[] | undefined {
    return this.base?.disk;
  }

  /** Starts the transcript from what is on disk (empty for a new conversation). Live ops index after it. */
  seed(
    entries: readonly TranscriptEntry[],
    turns: readonly TurnMetric[],
    transcriptError: string | null,
    contextUsed: number | null = null,
  ): void {
    this.version += 1;
    const reducer = new TranscriptReducer();
    reducer.seed(entries, turns, contextUsed);
    const needles = this.needles();
    this.open.clear();
    this.queue = [];
    this.clearTimer();
    this.typed = null;
    this.unsent = new Set();
    this.retryShown = null;
    this.base = {
      reducer,
      disk: entries.map((e) => ({ ...e })),
      view: entries.map((e) => redactEntry(e, needles)),
      turns: redactDeep(
        turns.map((t) => ({ ...t })),
        needles,
      ),
      transcriptError,
      contextUsed,
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

  /**
   * A rewind removed the entries from `index` on: the transcript starts again from the ones before it, with the metrics
   * of the turns whose end is kept, and every page gets a fresh `sync`. The context is unknown until it is read again.
   * Returns the kept entries as the reducer holds them (not redacted); none with no base.
   */
  cut(index: number, host: HostView): TranscriptEntry[] {
    if (this.base === null) return [];
    this.flush();
    const { entries, turns } = this.base.reducer.snapshot();
    const kept = entries.slice(0, Math.max(0, index)).map((e) => ({ ...e }));
    let ends = 0;
    for (const entry of kept) if (entry.kind === 'turn-end') ends += 1;
    this.lastEnd = null;
    const unsent = [...this.unsent].filter((at) => at < index);
    this.seed(kept, turns.slice(0, ends), this.base.transcriptError, null);
    this.unsent = new Set(unsent);
    this.resync(host);
    return kept;
  }

  /** The conversation's context in tokens, read with no line of the agent (after a rewind). */
  contextRead(used: number): void {
    this.emitOps([{ op: 'context', used }]);
  }

  /** The entries as the page was sent them (redacted); none with no base. */
  shownEntries(): readonly TranscriptEntry[] {
    return this.base?.view ?? [];
  }

  /** A rewind of the conversation ended: numbered, and kept for the `sync` of a page that missed the event. */
  rewound(end: RewindEnd): void {
    this.lastRewind = { n: (this.lastRewind?.n ?? 0) + 1, end };
    this.event({ type: 'rewind', seq: 0, n: this.lastRewind.n, end });
  }

  /** A live update line from the agent. */
  applyLine(line: UpdateLine): void {
    if (this.base === null) return;
    this.emitOps(this.base.reducer.apply(line));
    this.retryChanged();
  }

  /** The user's prompt, pushed when it is accepted (the agent does not echo it). */
  userPrompt(text: string, attachments: UserAttachment[] = []): void {
    this.lastEnd = null;
    if (this.base === null) return;
    this.version += 1;
    const ops = this.base.reducer.userPrompt(text, attachments);
    let index = this.base.view.length;
    let at: number | null = null;
    for (const op of ops) {
      if (op.op !== 'push') continue;
      if (op.entry.kind === 'user') {
        at = index;
        break;
      }
      index += 1;
    }
    this.emitOps(ops);
    this.retryChanged();
    const shown = at === null ? undefined : this.base.view[at];
    this.typed = shown?.kind === 'user' ? { index: at!, text: shown.text } : null;
  }

  /**
   * The prompt pushed last (typed, or the automatic `continue`), as the page was sent it, while it is still in the
   * view; null when this base holds none.
   */
  typedPrompt(): Extract<TranscriptEntry, { kind: 'user' }> | null {
    const typed = this.typed;
    if (this.base === null || typed === null) return null;
    const entry = this.base.view[typed.index];
    return entry?.kind === 'user' && entry.text === typed.text ? entry : null;
  }

  /**
   * A hook blocked the prompt pushed last: the agent stores no chunk of it, so it leaves the transcript (the rest stays
   * as it is) and every page gets a fresh `sync`.
   */
  dropTypedPrompt(host: HostView): void {
    if (this.base === null || this.typedPrompt() === null) return;
    const at = this.typed!.index;
    this.flush();
    const { entries, turns } = this.base.reducer.snapshot();
    const kept = entries.filter((_, i) => i !== at).map((e) => ({ ...e }));
    const unsent = [...this.unsent].filter((i) => i !== at).map((i) => (i > at ? i - 1 : i));
    this.seed(kept, turns, this.base.transcriptError, this.base.contextUsed);
    this.unsent = new Set(unsent);
    this.resync(host);
  }

  /** The turn of the prompt pushed last ended before the prompt went to the agent. */
  promptUnsent(): void {
    if (this.typedPrompt() !== null) this.unsent.add(this.typed!.index);
  }

  /** The view indices of prompts that never went to the agent, in this base. */
  unsentPrompts(): ReadonlySet<number> {
    return this.unsent;
  }

  /** The user's note on a call denied with one: the agent stores it but sends no live copy. Never starts a turn. */
  noteSent(text: string): void {
    if (this.base === null) return;
    this.version += 1;
    this.emitOps(this.base.reducer.note(text));
  }

  /** Work: a Send now the agent took: the agent stores it only once the model read it and sends no live copy. */
  steerSent(text: string): void {
    if (this.base === null) return;
    this.version += 1;
    this.emitOps(this.base.reducer.steer(text));
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

  compactionChanged(compaction: CompactionState | null, end?: CompactOutcome): void {
    // A compaction adds no entry and runs for minutes: the held-back tail of the open text must go out now.
    if (compaction !== null) this.closeOpenText();
    this.event({
      type: 'compaction',
      seq: 0,
      compaction: compaction === null ? null : { ...compaction },
      ...(end === undefined ? {} : { end }),
    });
  }

  hookRunChanged(hookRun: HookRunState | null): void {
    // A hook batch adds no entry and can hold the turn for minutes: the held-back tail of the open text must go out now.
    if (hookRun !== null) this.closeOpenText();
    this.event({
      type: 'hook-run',
      seq: 0,
      hookRun: hookRun === null ? null : redactDeep({ ...hookRun }, this.needles()),
    });
  }

  recapChanged(recap: RecapState | null, end?: RecapEnd): void {
    this.event({
      type: 'recap',
      seq: 0,
      recap: recap === null ? null : { ...recap },
      ...(end === undefined ? {} : { end }),
    });
  }

  /** The agent's session mode as its stored and live mode lines say; null with no base. */
  modeState(): { mode: 'plan' | 'default' | 'ask'; moves: number } | null {
    return this.base?.reducer.modeState() ?? null;
  }

  /** The conversation's context in tokens as the page was sent it; null while unknown or with no base. */
  contextUsed(): number | null {
    return this.base?.contextUsed ?? null;
  }

  /** Work: the host's workflow runs, the whole list, with the ops queued before it in the same event. */
  runsChanged(runs: readonly RunView[]): void {
    if (this.closed) return;
    this.clearTimer();
    const ops = this.queue.length === 0 ? null : mergeTextOps(this.queue);
    this.queue = [];
    this.broadcast({ type: 'runs', seq: 0, runs: runs.map((r) => ({ ...r })), ...(ops === null ? {} : { ops }) }, true);
  }

  /** Work: a run's `state.json`, read live: merged into its entry by the reader's rule, so the page gets its history. */
  runState(state: RunState, fileTs: number): void {
    if (this.base === null) return;
    this.emitOps(this.base.reducer.mergeRun(state.runId, (entry) => mergeRunState(entry, state, fileTs)));
  }

  /** Work: the live facts of the background work, the whole state; monitor output redacted. */
  activityChanged(live: LiveActivity): void {
    this.event({ type: 'activity', seq: 0, live: liveCopy(live, this.needles()) });
  }

  /** Work: an Agent view's ops, or the stored list at its agent's end; its watch redacted them. */
  agentEntries(event: Extract<ConversationEvent, { type: 'agent-entries' }>): void {
    this.event(event);
  }

  /**
   * The loaded session's command list changed (the whole list, as the page may draw it), or the agent exited (null).
   */
  commandsChanged(commands: readonly CommandView[] | null): void {
    this.event({
      type: 'commands',
      seq: 0,
      commands: commands === null ? null : shownCommands(commands, this.needles()),
    });
  }

  /** The queued messages changed: the whole state, redacted. */
  queueChanged(state: QueueState): void {
    this.queued = state;
    this.event({ type: 'queue', seq: 0, queue: redactDeep(state, this.needles()) });
  }

  turnEnded(end: TurnEnd): void {
    this.closeOpenText();
    const lastEnd: TurnEnd = { turnId: end.turnId };
    if (end.outcome !== undefined) lastEnd.outcome = end.outcome;
    if (end.failure !== undefined) lastEnd.failure = end.failure;
    if (end.unsent === true) lastEnd.unsent = true;
    if (end.blocked !== undefined) lastEnd.blocked = end.blocked;
    if (end.model !== undefined) {
      const needles = this.needles();
      const id = (value: string | null): string | null => (value === null ? null : redactText(value, needles));
      lastEnd.model = {
        previous: id(end.model.previous),
        next: id(end.model.next),
        reason: redactTitle(end.model.reason, needles),
      };
    }
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
    if ((kept.kind === 'allowed' || kept.kind === 'denied') && kept.card !== null) this.cards += 1;
    while (this.cards > MARK_CARDS_KEPT && this.cardFrom < this.marks.length) {
      const old = this.marks[this.cardFrom]!;
      if ((old.kind === 'allowed' || old.kind === 'denied') && old.card !== null) {
        this.marks[this.cardFrom] = { ...old, card: null };
        this.cards -= 1;
      }
      this.cardFrom += 1;
    }
    this.event({ type: 'permission-mark', seq: 0, mark: kept });
  }

  /** A never-allow rule was removed: the marks of its calls read "rule removed", on every page and in later syncs. */
  ruleGone(ruleId: string): void {
    let changed = false;
    for (let i = 0; i < this.marks.length; i++) {
      const mark = this.marks[i]!;
      if (mark.kind !== 'denied' || mark.ruleId !== ruleId || mark.words === null) continue;
      this.marks[i] = { ...mark, words: null };
      changed = true;
    }
    if (changed) this.event({ type: 'rule-gone', seq: 0, ruleId });
  }

  /** Each notice once: the same code for the same provider is not repeated. */
  notice(notice: ConversationNotice): void {
    if (this.notices.some((n) => n.code === notice.code && n.provider === notice.provider)) return;
    this.notices.push(notice);
    this.event({ type: 'notice', seq: 0, notice });
  }

  /** Every notice with this code goes away, on every page. */
  clearNotice(code: ConversationNotice['code']): void {
    const kept = this.notices.filter((n) => n.code !== code);
    if (kept.length === this.notices.length) return;
    this.notices.splice(0, this.notices.length, ...kept);
    this.event({ type: 'notice-cleared', seq: 0, code });
  }

  /** The conversation is deleted (or never existed): every page is told, and the stream ends. */
  gone(reason: 'deleted' | 'unknown'): void {
    this.event({ type: 'gone', seq: 0, reason });
    this.closed = true;
    this.sinks.clear();
    this.clearTimer();
  }

  publicInfo(): ConversationInfo {
    const info = { ...this.info };
    const cwd = info.mode === 'work' ? info.cwd : null;
    if (cwd !== null && cwd !== '' && this.deps.noFolder?.(cwd) === true) info.noFolder = true;
    return redactDeep(info, this.needles());
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
      ...(this.base === null || this.base.contextUsed === null ? {} : { contextUsed: this.base.contextUsed }),
      ...(host.compaction === null ? {} : { compaction: { ...host.compaction } }),
      ...(host.hookRun === null ? {} : { hookRun: redactDeep({ ...host.hookRun }, needles) }),
      ...(host.recap === null ? {} : { recap: { ...host.recap } }),
      ...(this.lastRewind === null ? {} : { rewindEnd: { n: this.lastRewind.n, end: { ...this.lastRewind.end } } }),
      ...(this.retryShown === null ? {} : { retry: { ...this.retryShown } }),
      runs: host.runs.map((r) => ({ ...r })),
      live: liveCopy(host.live, needles),
      ...(this.queued.count === 0 && this.queued.held === null ? {} : { queue: redactDeep(this.queued, needles) }),
      ...(host.commands === null ? {} : { commands: shownCommands(host.commands, needles) }),
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
        out.push({ op: 'set', index: op.index, entry: redactEntry(op.entry, needles) });
      } else if (op.op === 'context') {
        out.push(op);
      } else {
        this.closeOpenTextInto(out);
        out.push({ op: 'metric', metric: redactDeep({ ...op.metric }, needles) });
      }
    }
    this.commit(out);
  }

  /** The running answer's last retry episode changed: every page gets it (redacted), or null once the answer ended. */
  private retryChanged(): void {
    if (this.base === null) return;
    const open = this.base.reducer.retryLive();
    const shown = open === null ? null : redactEntry(open, this.needles());
    if (JSON.stringify(shown) === JSON.stringify(this.retryShown)) return;
    this.retryShown = shown?.kind === 'notice' && shown.code === 'retry' ? shown : null;
    this.event({ type: 'retry', seq: 0, retry: this.retryShown === null ? null : { ...this.retryShown } });
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
    for (const op of ops) {
      if (op.op === 'context') this.base.contextUsed = op.used;
      else applyOp(this.base.view, this.base.turns, op);
    }
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

/** Applies one op to an entry list, as the page does. A `context` op changes no entry. */
export function applyOp(entries: TranscriptEntry[], turns: TurnMetric[], op: TranscriptOp): void {
  switch (op.op) {
    case 'push':
      entries.push({ ...op.entry });
      return;
    case 'text': {
      const entry = entries[op.index];
      if (entry !== undefined && isText(entry)) {
        entries[op.index] = {
          ...entry,
          text: entry.text + op.append,
          ...(op.endTs === undefined ? {} : { endTs: op.endTs }),
        };
      }
      return;
    }
    case 'set':
      if (op.index >= 0 && op.index < entries.length) entries[op.index] = { ...op.entry };
      return;
    case 'metric':
      turns.push({ ...op.metric });
      return;
    case 'context':
      return;
  }
}

/** Consecutive `text` ops on one index become one; of the `context` ops only the last is kept, at the end. */
export function mergeTextOps(ops: readonly TranscriptOp[]): TranscriptOp[] {
  const out: TranscriptOp[] = [];
  let context: TranscriptOp | null = null;
  for (const op of ops) {
    if (op.op === 'context') {
      context = op;
      continue;
    }
    const last = out[out.length - 1];
    if (op.op === 'text' && last !== undefined && last.op === 'text' && last.index === op.index) {
      const endTs = op.endTs ?? last.endTs;
      out[out.length - 1] = {
        op: 'text',
        index: op.index,
        append: last.append + op.append,
        ...(endTs === undefined ? {} : { endTs }),
      };
    } else {
      out.push(op);
    }
  }
  if (context !== null) out.push(context);
  return out;
}
