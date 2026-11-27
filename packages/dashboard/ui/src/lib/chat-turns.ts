/**
 * The thread as turns: where each turn starts and ends, the words of how it ended (live from the `turn` event, or from
 * the stored `turn-end`), its numbers, and the Markdown a copy puts on the clipboard.
 */

import { appEndFor, appEndText, type AppEndContext } from './app-ends';
import { CHAT, fill } from './chat-strings';
import type {
  ChatFailure,
  ChatFailureCode,
  ChatHostState,
  ChatMode,
  HookRunState,
  ModelSwitch,
  TurnOutcome,
} from './chat-types';
import {
  formatBytes,
  formatClock,
  formatCount,
  formatCountNoun,
  formatDuration,
  formatNumber,
  formatTps,
} from './format';
import type { PictureSize, TranscriptEntry, TurnMetric } from './types';
import { oneLine, parseWebCall } from './web-tools';
import { visibleText } from './work-tools';

type TurnEnd = Extract<TranscriptEntry, { kind: 'turn-end' }>;

export interface ThreadTurn {
  /** Index of the turn's first entry: stable while entries are appended. */
  start: number;
  entries: TranscriptEntry[];
  /** The stored end; null while the turn is open or when it never got one. */
  end: TurnEnd | null;
  metric: TurnMetric | null;
}

/** A user entry inside a running answer: a steer or the note on a denied call. It never starts a turn. */
export function isAside(entry: TranscriptEntry): boolean {
  return entry.kind === 'user' && (entry.interjection === true || entry.note === true);
}

/**
 * No step of the answer: a notice, a mode line, a goal, a recap, or a steer or note inside it. It never starts a turn
 * and never counts as work.
 */
export function isSideEntry(entry: TranscriptEntry): boolean {
  return (
    entry.kind === 'notice' ||
    entry.kind === 'plan-mode' ||
    entry.kind === 'goal' ||
    entry.kind === 'recap' ||
    isAside(entry)
  );
}

function startsTurn(entry: TranscriptEntry): boolean {
  return !isSideEntry(entry) && (entry.kind === 'user' || entry.kind === 'wake');
}

/** The turn that holds entry `at` (its index in the conversation's entries); -1 before the first turn. */
export function turnIndexOf(list: readonly ThreadTurn[], at: number): number {
  let lo = 0;
  let hi = list.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid]!.start <= at) {
      found = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return found;
}

/**
 * How many of the newest turns the thread mounts so a jump to entry `at` lands: the older turns wait behind a button,
 * and a jump to one of them mounts it and every turn after it first. `mounted` when it already shows.
 */
export function mountedFor(list: readonly ThreadTurn[], at: number, mounted: number): number {
  const turn = turnIndexOf(list, at);
  if (turn === -1) return mounted;
  return Math.max(mounted, list.length - turn);
}

/**
 * A turn starts at each typed prompt or wake and closes at its `turn-end`. The k-th `turn-end` pairs with `turns[k]`.
 * A turn that ended with no `turn-end` (an error reply) still ends where the next prompt starts.
 */
export function splitTurns(entries: readonly TranscriptEntry[], turns: readonly TurnMetric[]): ThreadTurn[] {
  const out: ThreadTurn[] = [];
  let current: ThreadTurn | null = null;
  let closed = 0;
  entries.forEach((entry, index) => {
    if (current !== null && startsTurn(entry) && current.entries.some((e) => e.kind !== 'model')) {
      out.push(current);
      current = null;
    }
    current ??= { start: index, entries: [], end: null, metric: null };
    if (entry.kind === 'turn-end') {
      current.end = entry;
      current.metric = turns[closed] ?? null;
      closed += 1;
      out.push(current);
      current = null;
      return;
    }
    current.entries.push(entry);
  });
  if (current !== null) out.push(current);
  return out;
}

export type OutcomeTone = 'neutral' | 'warning' | 'danger';
export type TurnActionKind = 'send-again' | 'continue' | 'update-agent' | 'pick-model';

export interface OutcomeLine {
  tone: OutcomeTone;
  /** `stop` (Square), `ban`, or the tone's warning glyph. */
  glyph: 'stop' | 'ban' | 'alert';
  text: string;
  detail?: string;
  action: TurnActionKind | null;
  /** Line 2: what the engine does now (`engineNote`). Live only. */
  note?: EngineNote;
}

export interface EngineNote {
  text: string;
  /** `open-engine`: a link to the Engine page. `unload`: the Unload button. */
  action: 'open-engine' | 'unload' | null;
}

/**
 * The live outcome of a turn, or null for `completed`. `unsent`: the turn ended before its prompt went to the agent.
 * `failure`: why the guard refused it (a model it cannot use asks for another model, not an agent update).
 */
export function liveOutcome(
  outcome: TurnOutcome,
  mode: ChatMode,
  unsent = false,
  failure?: ChatFailureCode,
): OutcomeLine | null {
  const o = CHAT.outcomes;
  switch (outcome) {
    case 'completed':
      return null;
    case 'cancelled':
      return {
        tone: 'neutral',
        glyph: 'stop',
        text: unsent ? o['cancelled-unsent'] : o.cancelled,
        action: 'send-again',
      };
    case 'output-limit':
      return { tone: 'warning', glyph: 'alert', text: o['output-limit'], action: 'continue' };
    case 'tool-loop':
      return { tone: 'warning', glyph: 'alert', text: o['tool-loop'], action: 'send-again' };
    case 'tool-refused':
      return { tone: 'warning', glyph: 'alert', text: o['tool-refused'], action: 'send-again' };
    case 'arguments-unreadable':
      return { tone: 'warning', glyph: 'alert', text: o['arguments-unreadable'], action: 'send-again' };
    case 'denied':
      return { tone: 'neutral', glyph: 'ban', text: o.denied, action: null };
    case 'turn-timeout':
      return { tone: 'warning', glyph: 'alert', text: o['turn-timeout'], action: 'send-again' };
    case 'inference-lost':
      return { tone: 'danger', glyph: 'alert', text: o['inference-lost'], action: 'send-again' };
    case 'interrupted':
      return { tone: 'warning', glyph: 'alert', text: o.interrupted, action: 'send-again' };
    case 'agent-crashed':
      return { tone: 'danger', glyph: 'alert', text: o['agent-crashed'], action: 'send-again' };
    case 'model-error':
      return { tone: 'danger', glyph: 'alert', text: o['model-error'], action: 'send-again' };
    case 'no-model':
      return { tone: 'warning', glyph: 'alert', text: o['no-model'], action: 'send-again' };
    case 'hook-stopped':
      return HOOK_STOPPED;
    case 'never-loop':
      return { tone: 'neutral', glyph: 'ban', text: o['never-loop'], action: null };
    case 'refused':
      if (failure === 'model-unavailable') {
        return { tone: 'danger', glyph: 'alert', text: o['model-unavailable'], action: 'pick-model' };
      }
      return {
        tone: 'danger',
        glyph: 'alert',
        text: mode === 'work' ? o['refused-work'] : o['refused-chat'],
        action: 'update-agent',
      };
  }
}

/** Our sentence on a model the agent could not use, without the agent's words. */
export function modelHeadline(model: ModelSwitch): string {
  const m = CHAT.modelAuto;
  if (model.previous === null) return m.none;
  const previous = visibleText(model.previous);
  return model.next === null
    ? fill(m.refused, { previous })
    : fill(m.wanted, { previous, next: visibleText(model.next) });
}

/** The muted line under an answer's end: our sentence, then the agent's words as it wrote them. */
export function modelDetail(model: ModelSwitch): string {
  const head = modelHeadline(model);
  const reason = visibleText(model.reason);
  if (reason === '') return head;
  return head.endsWith('.') ? `${head} ${reason}` : `${head}: ${reason}`;
}

/** A stored end on the model's length limit: `max_tokens`, or the agent's length error. */
function isLengthEnd(end: TurnEnd): boolean {
  return end.stopReason === 'max_tokens' || end.lengthLimit === true;
}

/** The turn is a `continue` after a length end, and the model called no tool in it. */
function continuedWithoutTool(turn: ThreadTurn): boolean {
  const prompt = turn.entries.find(startsTurn);
  return prompt?.kind === 'user' && prompt.continued === true && !turn.entries.some((entry) => entry.kind === 'tool');
}

function twice(line: OutcomeLine | null): OutcomeLine | null {
  return line === null ? null : { ...line, text: CHAT.outcomes['output-limit-twice'] };
}

/** The outcome of a stored turn, from its `turn-end`. */
export function storedOutcome(end: TurnEnd): OutcomeLine | null {
  const o = CHAT.outcomes;
  if (end.lengthLimit === true) return { tone: 'warning', glyph: 'alert', text: o['output-limit'], action: 'continue' };
  const detail = end.error !== undefined && end.error !== '' ? end.error : undefined;
  let line: OutcomeLine | null;
  if (end.stopReason === 'end_turn') line = null;
  else if (end.stopReason === 'cancelled' && end.hookStopped === true) line = HOOK_STOPPED;
  else if (end.stopReason === 'cancelled')
    line = { tone: 'neutral', glyph: 'stop', text: o['stopped-stored'], action: 'send-again' };
  else if (end.stopReason === 'max_tokens')
    line = { tone: 'warning', glyph: 'alert', text: o['output-limit'], action: 'continue' };
  else line = { tone: 'neutral', glyph: 'alert', text: fill(o.ended, { stopReason: end.stopReason }), action: null };
  if (detail === undefined) return line;
  return line === null
    ? { tone: 'danger', glyph: 'alert', text: fill(o.ended, { stopReason: end.stopReason }), detail, action: null }
    : { ...line, tone: line.tone === 'neutral' ? 'danger' : line.tone, detail };
}

export interface TurnEndState {
  /** The turn this outcome belongs to is the last one and no turn runs. */
  last: boolean;
  running: boolean;
  live: { outcome?: TurnOutcome; failure?: ChatFailure; unsent?: true; model?: ModelSwitch } | null;
  mode: ChatMode;
  /** How earlier Control Panel lives ended, for this conversation: see `app-ends.ts`. */
  app?: AppEndContext;
}

/** The time of a turn's prompt or wake, or null. */
export function turnStartedAt(turn: ThreadTurn): number | null {
  return turn.entries.find(startsTurn)?.ts ?? null;
}

/**
 * The outcome line of a turn: live for the last turn when the stream saw it end, else the Control Panel restart that
 * stopped it, else stored, else "did not finish". A finished turn always has a stored end, so an earlier turn without
 * one did not finish either.
 */
export function turnOutcome(turn: ThreadTurn, state: TurnEndState): OutcomeLine | null {
  if (state.last && state.running) return null;
  if (state.last && state.live !== null && state.live.outcome !== undefined) {
    const live = liveOutcome(state.live.outcome, state.mode, state.live.unsent === true, state.live.failure?.code);
    const line = state.live.outcome === 'output-limit' && continuedWithoutTool(turn) ? twice(live) : live;
    const model = state.live.model;
    return line === null || model === undefined ? line : { ...line, detail: modelDetail(model) };
  }
  const appEnd =
    state.app === undefined ? null : appEndFor({ end: turn.end, startedAt: turnStartedAt(turn) }, state.app);
  if (appEnd !== null) return { tone: 'warning', glyph: 'alert', text: appEndText(appEnd), action: 'continue' };
  if (turn.end !== null) {
    const line = storedOutcome(turn.end);
    return isLengthEnd(turn.end) && continuedWithoutTool(turn) ? twice(line) : line;
  }
  if (turn.entries.some(startsTurn))
    return {
      tone: 'neutral',
      glyph: 'alert',
      text: CHAT.outcomes.unfinished,
      action: state.last ? 'send-again' : null,
    };
  return null;
}

/** `12.4 s`, `512 tokens`, `41.3 tok/s`; a null or zero part is left out. */
export function turnStats(metric: TurnMetric | null, end: TurnEnd | null): string[] {
  const parts: string[] = [];
  const elapsed = metric?.elapsedMs ?? end?.elapsedMs ?? null;
  if (elapsed !== null && elapsed !== undefined && elapsed > 0) parts.push(formatDuration(elapsed));
  if (metric !== null && metric.outputTokens > 0) parts.push(formatCountNoun(metric.outputTokens, 'token', 'tokens'));
  if (metric?.decodeTps !== null && metric?.decodeTps !== undefined && metric.decodeTps > 0)
    parts.push(`${formatTps(metric.decodeTps)} tok/s`);
  return parts;
}

/**
 * The stats tooltip; parts with no value are left out. The decode speed is the agent's rate over all model calls of
 * the turn; the first token is the first call's.
 */
export function turnStatsTitle(metric: TurnMetric | null, model: string | null): string | undefined {
  if (metric === null) return undefined;
  const words = CHAT.statsTitle;
  const parts: string[] = [];
  if (metric.decodeTps !== null && metric.decodeTps > 0)
    parts.push(fill(words.decode, { tps: formatTps(metric.decodeTps) }));
  if (metric.ttftMs !== null && metric.ttftMs > 0)
    parts.push(fill(words.firstToken, { ttft: formatDuration(metric.ttftMs) }));
  if (metric.inputTokens > 0) {
    const input = formatNumber(metric.inputTokens);
    parts.push(
      metric.cachedTokens > 0
        ? fill(words.inputCached, { input, cached: formatNumber(metric.cachedTokens) })
        : fill(words.input, { input }),
    );
  }
  if (metric.outputTokens > 0) parts.push(fill(words.output, { output: formatNumber(metric.outputTokens) }));
  const name = metric.model ?? model;
  if (name !== null && name !== '') parts.push(name);
  return parts.length === 0 ? undefined : parts.join(' · ');
}

/** The Markdown of a turn's answer: its assistant entries, joined by a blank line. */
export function answerMarkdown(entries: readonly TranscriptEntry[]): string {
  return entries
    .filter((entry): entry is Extract<TranscriptEntry, { kind: 'assistant' }> => entry.kind === 'assistant')
    .map((entry) => entry.text.trim())
    .filter((text) => text !== '')
    .join('\n\n');
}

/** The typed prompt of a turn (what "Send again" sends), or null. */
export function turnPrompt(turn: ThreadTurn): string | null {
  const user = turn.entries.find(
    (entry): entry is Extract<TranscriptEntry, { kind: 'user' }> => entry.kind === 'user' && !isAside(entry),
  );
  return user === undefined || user.text.trim() === '' ? null : user.text;
}

/** An ATX heading line whose words are those of a section of the export, at any level and spacing. */
function isSectionHeading(line: string): boolean {
  const t = line.trimStart();
  let hashes = 0;
  while (t[hashes] === '#') hashes += 1;
  if (hashes === 0 || hashes > 6 || (t[hashes] !== ' ' && t[hashes] !== '\t')) return false;
  const rest = t.slice(hashes).trim();
  let end = rest.length;
  while (end > 0 && rest[end - 1] === '#') end -= 1;
  const words = rest.slice(0, end).trim();
  return words === 'You' || words === 'Assistant';
}

/** A line of message text that would read as a section heading of the export gets a backslash. */
function escapeSections(text: string): string {
  return text
    .split('\n')
    .map((line) => (isSectionHeading(line) ? `\\${line.trimStart()}` : line))
    .join('\n');
}

/** The whole conversation as Markdown: `## You` / `## Assistant` sections, one quoted line per tool call. */
export function conversationMarkdown(entries: readonly TranscriptEntry[]): string {
  const parts: string[] = [];
  let section: 'you' | 'assistant' | null = null;
  const open = (next: 'you' | 'assistant'): void => {
    if (section === next) return;
    section = next;
    parts.push(next === 'you' ? '## You' : '## Assistant');
  };
  for (const entry of entries) {
    if (entry.kind === 'user') {
      open('you');
      parts.push(escapeSections(entry.text.trim()));
    } else if (entry.kind === 'assistant') {
      if (entry.text.trim() === '') continue;
      open('assistant');
      parts.push(escapeSections(entry.text.trim()));
    } else if (entry.kind === 'tool') {
      if (entry.name === 'search_tool') continue;
      open('assistant');
      const web = parseWebCall(entry.name, entry.input);
      if (web?.kind === 'search') parts.push(`> ${CHAT.searched} "${oneLine(web.query)}"`);
      else if (web?.kind === 'fetch') parts.push(`> ${CHAT.read} ${web.host}`);
      else parts.push(`> ${oneLine(entry.title !== '' ? entry.title : entry.name)}`);
    }
  }
  return parts.join('\n\n');
}

/** One load of `ChatHostState.loading`. */
export type EngineLoad = ChatHostState['loading'][number];

/** The load a conversation on `model` shows: that model's own load, else the oldest one. */
export function loadFor(loading: readonly EngineLoad[], model: string | null): EngineLoad | null {
  return loading.find((load) => load.modelId === model) ?? loading[0] ?? null;
}

/** One request the engine holds now: `ChatHostState.requests`. */
export type EngineRequest = ChatHostState['requests'][number];

export interface PendingInput {
  creating: boolean;
  phase: 'starting' | 'running' | 'cancelling' | null;
  wait: 'model' | null;
  /** The conversation's model. */
  model: string | null;
  /** The load this conversation shows ({@link loadFor}). */
  loading: { modelId: string; startedAt: number } | null;
  inference: 'ready' | 'starting' | 'loading' | 'stopped' | null;
  /** No stream event for 3 s while running. */
  quiet: boolean;
  /** The conversation's model is full: see {@link fullLimit}. */
  full: number | null;
  /** The app sent the automatic `continue` and nothing of its answer came yet. */
  continuing?: boolean;
  /** The turn has shown work (a thought, a tool row, text): with nothing known the line says "Working…". */
  work?: boolean;
  /** What the engine does about this conversation now ({@link engineWait}); null when nothing is known. */
  engine?: EngineWait | null;
  /** The PDF pages the turn's message carries ({@link messagePages}): the prompt read names them. */
  pages?: number;
}

/** The PDF pages a turn's message carries in itself: the pages of each PDF that went in the message. */
export function messagePages(turn: ThreadTurn | undefined): number {
  const first = turn?.entries[0];
  if (first?.kind !== 'user' || first.attachments === undefined) return 0;
  let pages = 0;
  for (const item of first.attachments) if (item.kind === 'pdf' && item.way === 'message') pages += item.pages;
  return pages;
}

/**
 * A prompt read: `done` of `total` tokens. No `done` while the engine has counted none: a one-shot prefill reports its
 * count only at its end, so its meter is busy with no fill and its words never say "0 of".
 */
export interface ReadCounts {
  done?: number;
  total: number;
}

export type PendingRow =
  | { kind: 'label'; text: string; orb: 'hold' | 'think'; progress?: ReadCounts }
  | { kind: 'load'; modelId: string };

/**
 * `ours`: a request that named this conversation's agent session. `model`: no request named it, but the model holds a
 * request that named no session and started in this wait, or requests wait in a queue the engine does not list while
 * every listed one is known to be another's; the words are then the model's, never this conversation's. `reading` is a
 * prefill on the model with its counts.
 */
export type EngineWait =
  | { kind: 'ours'; phase: EngineRequest['phase']; done: number | null; total: number | null }
  | { kind: 'model'; reading: ReadCounts | null };

/** The counts of a prefill that reports them, without `done` while it is 0; null for any other phase. */
function readCounts(phase: EngineRequest['phase'], done: number | null, total: number | null): ReadCounts | null {
  if (phase !== 'prefill' || done === null || total === null || total <= 0) return null;
  const counted = Math.min(Math.max(done, 0), total);
  return counted === 0 ? { total } : { done: counted, total };
}

/** `of` with both counts, or `all` with the total alone while nothing is counted. */
function readText(counts: ReadCounts, of: string, all: string): string {
  return counts.done === undefined
    ? fill(all, { total: formatCount(counts.total) })
    : fill(of, { done: formatCount(counts.done), total: formatCount(counts.total) });
}

/**
 * What the engine does for a conversation: its newest request that named `session`, else the model's state when one of
 * its requests may be this conversation's (it named no session and started at or after `since`, or `hidden` requests
 * wait unlisted behind listed ones that are all another's), else null.
 */
export function engineWait(input: {
  requests: readonly EngineRequest[];
  session: string | null;
  model: string | null;
  /** Start of this wait (epoch ms). */
  since: number | null;
  /** Requests on the model that wait where the engine lists none ({@link fullLimit}'s `queued`). */
  hidden: number;
}): EngineWait | null {
  if (input.model === null) return null;
  const onModel = input.requests.filter((request) => request.modelId === input.model);
  let ours: EngineRequest | null = null;
  if (input.session !== null)
    for (const request of onModel)
      if (request.session === input.session && (ours === null || request.startedAt >= ours.startedAt)) ours = request;
  if (ours !== null) {
    return { kind: 'ours', phase: ours.phase, done: ours.prefillDone, total: ours.prefillTotal };
  }
  const since = input.since;
  const mayBeOurs = (request: EngineRequest): boolean =>
    request.session === null && (since === null || request.startedAt >= since);
  const unknown = since !== null && onModel.some(mayBeOurs);
  const queuedBehind = input.hidden > 0 && onModel.length > 0 && !onModel.some(mayBeOurs);
  if (!unknown && !queuedBehind) return null;
  let reading: ReadCounts | null = null;
  for (const request of onModel) reading ??= readCounts(request.phase, request.prefillDone, request.prefillTotal);
  return { kind: 'model', reading };
}

/** What the pending row shows, first match wins. */
export function pendingRow(input: PendingInput): PendingRow {
  const label = (text: string, orb: 'hold' | 'think' = 'hold'): PendingRow => ({ kind: 'label', text, orb });
  const queued = (): PendingRow | null =>
    input.full !== null && input.model !== null
      ? label(
          input.full === 1
            ? fill(CHAT.pendingQueuedOne, { model: input.model })
            : fill(CHAT.pendingQueuedMany, { model: input.model, n: String(input.full) }),
        )
      : null;
  const reading = (text: string, counts: ReadCounts, orb: 'hold' | 'think'): PendingRow => ({
    kind: 'label',
    text,
    orb,
    progress: counts,
  });
  const engine = input.phase === 'running' ? (input.engine ?? null) : null;
  if (input.creating) return label(CHAT.pendingStarting);
  if (input.phase === 'cancelling') return label(CHAT.pendingStopping);
  if (engine?.kind === 'ours') {
    if (engine.phase === 'queued' || engine.phase === 'slot') return queued() ?? label(CHAT.pendingQueued);
    if (engine.phase === 'prepare' || engine.phase === 'prefill') {
      const counts = readCounts(engine.phase, engine.done, engine.total);
      // The pages are named only for the read of their own message: once the turn has worked, they are history.
      const pages = input.work === true ? 0 : (input.pages ?? 0);
      if (pages > 0) {
        const one = pages === 1;
        const n = formatCount(pages);
        if (counts === null)
          return label(one ? CHAT.pendingReadingPage : fill(CHAT.pendingReadingPages, { n }), 'think');
        // `fill` keeps the `{done}` and `{total}` it is not given for `readText`.
        const of = fill(one ? CHAT.pendingReadingPageOf : CHAT.pendingReadingPagesOf, { n });
        const all = fill(one ? CHAT.pendingReadingPageTotal : CHAT.pendingReadingPagesTotal, { n });
        return reading(readText(counts, of, all), counts, 'think');
      }
      return counts === null
        ? label(CHAT.pendingReading, 'think')
        : reading(readText(counts, CHAT.pendingReadingOf, CHAT.pendingReadingTotal), counts, 'think');
    }
    return label(CHAT.pendingThinking, 'think');
  }
  if (input.wait === 'model' && input.model !== null) return { kind: 'load', modelId: input.model };
  if (input.phase === 'running' && input.loading !== null) return { kind: 'load', modelId: input.loading.modelId };
  if (input.phase === 'starting') return label(CHAT.pendingStarting);
  if (input.inference === 'starting') return label(CHAT.pendingEngine);
  if (input.phase === 'running') {
    const full = queued();
    if (full !== null) return full;
  }
  if (engine?.kind === 'model')
    return engine.reading === null
      ? label(CHAT.pendingBusy)
      : reading(
          readText(engine.reading, CHAT.pendingBusyReading, CHAT.pendingBusyReadingTotal),
          engine.reading,
          'hold',
        );
  if (input.phase === 'running' && input.continuing === true) return label(CHAT.pendingContinuing, 'think');
  if (input.quiet) return label(CHAT.pendingWaiting);
  return input.work === true ? label(CHAT.pendingWorking, 'think') : label(CHAT.pendingThinking, 'think');
}

/** The quieter words of the compaction line: the summary request reads the conversation, then writes. */
export interface CompactionDetail {
  text: string;
  progress?: ReadCounts;
}

/**
 * What the conversation's newest request since the compaction started (`since`) does: its prompt read with counts, or
 * its writing with the tokens so far; null when no such request is listed or its phase says nothing.
 */
export function compactionDetail(input: {
  requests: readonly EngineRequest[];
  session: string | null;
  since: number;
}): CompactionDetail | null {
  let ours: EngineRequest | null = null;
  for (const request of input.requests)
    if (
      input.session !== null &&
      request.session === input.session &&
      request.startedAt >= input.since &&
      (ours === null || request.startedAt >= ours.startedAt)
    )
      ours = request;
  if (ours === null) return null;
  const counts = readCounts(ours.phase, ours.prefillDone, ours.prefillTotal);
  if (counts !== null)
    return { text: readText(counts, CHAT.compactingReading, CHAT.compactingReadingTotal), progress: counts };
  if (ours.phase !== 'decode') return null;
  return ours.generated === null || ours.generated <= 0
    ? { text: CHAT.compactingWriting }
    : { text: fill(CHAT.compactingWritingOf, { n: formatCount(ours.generated) }) };
}

/**
 * The words of a compaction marker; the counts only when the agent sent both and `after` is at least 1% below
 * `before` (an agent that reports the count from before the summary sends about the same number twice).
 */
export function compactionText(entry: Extract<TranscriptEntry, { kind: 'compaction' }>): string {
  if (entry.outcome === 'stopped') return CHAT.compactStopped;
  if (entry.outcome === 'failed') return entry.retry === true ? CHAT.compactFailedRetry : CHAT.compactFailed;
  const { tokensBefore: before, tokensAfter: after } = entry;
  if (before === undefined || after === undefined || after * 100 > before * 99) return CHAT.compacted;
  return fill(CHAT.compactedTokens, { before: formatCount(before), after: formatCount(after) });
}

export type RetryNotice = Extract<TranscriptEntry, { kind: 'notice'; code: 'retry' }>;

/**
 * The running answer's retry episode while it has no end (the agent still tries), from the stream's live copy: its row
 * is placed later (before the next call or at the answer's end), so the thread's entries cannot say it.
 */
export function openRetry(live: RetryNotice | null | undefined): RetryNotice | null {
  return live == null || live.endTs !== undefined ? null : live;
}

function retryTry(entry: RetryNotice): string {
  return entry.max >= 1
    ? fill(CHAT.retry.try, { n: entry.attempts, max: entry.max })
    : fill(CHAT.retry.tryNoMax, { n: entry.attempts });
}

/**
 * The prompt read of this conversation's newest engine request since `since` (the request the agent sent again), with
 * its counts; null when no such request is listed or it does not read.
 */
export function retryReading(input: {
  requests: readonly EngineRequest[];
  session: string | null;
  since: number;
}): ReadCounts | null {
  let ours: EngineRequest | null = null;
  for (const request of input.requests)
    if (
      input.session !== null &&
      request.session === input.session &&
      request.startedAt >= input.since &&
      (ours === null || request.startedAt >= ours.startedAt)
    )
      ours = request;
  return ours === null ? null : readCounts(ours.phase, ours.prefillDone, ours.prefillTotal);
}

/** The wait line of an open retry episode: what happened, then the try. */
export function retryLine(entry: RetryNotice): string {
  return `${CHAT.retry.live[entry.cause]} · ${retryTry(entry)}`;
}

/** One event row's parts; `detail` is the agent's own words. */
export interface NoticeRowView {
  label: string;
  text: string;
  facts: string[];
  tone: 'neutral' | 'warning' | 'danger';
  detail?: string;
}

/**
 * A retry episode as a row. `running`: its turn runs now, so an episode with no end still tries; with no turn running
 * it never finished (the agent stopped mid-retry).
 */
export function retryRow(entry: RetryNotice, running: boolean): NoticeRowView {
  const text = CHAT.retry.cause[entry.cause];
  const tries = entry.attempts === 1 ? CHAT.retry.triesOne : fill(CHAT.retry.tries, { n: entry.attempts });
  const detail = entry.detail === undefined ? {} : { detail: entry.detail };
  if (entry.endTs === undefined) {
    return running
      ? { label: CHAT.retry.rowLive, text, facts: [retryTry(entry)], tone: 'neutral', ...detail }
      : { label: CHAT.retry.rowOpen, text, facts: [tries], tone: 'warning', ...detail };
  }
  if (entry.outcome === 'gave-up')
    return { label: CHAT.retry.rowGaveUp, text, facts: [tries], tone: 'danger', ...detail };
  const facts = [tries];
  if (entry.outcome === 'recovered' && entry.ts > 0 && entry.endTs > entry.ts)
    facts.push(fill(CHAT.retry.waited, { time: formatDuration(entry.endTs - entry.ts) }));
  return { label: CHAT.retry.row, text, facts, tone: 'neutral', ...detail };
}

export type PictureNotice = Extract<TranscriptEntry, { kind: 'notice'; code: 'picture' }>;

/** Lines of the agent's notes or of sizes a picture row shows. */
const PICTURE_LINES_MAX = 6;

function pictureSize(size: PictureSize): string {
  return fill(CHAT.pictureNote.size, { from: formatBytes(size.fromBytes), to: formatBytes(size.toBytes) });
}

function picturePixels(size: PictureSize): string {
  return fill(CHAT.pictureNote.pixels, { w: size.from[0], h: size.from[1], w2: size.to[0], h2: size.to[1] });
}

/** The row's headline: pictures left out (no count: the agent's note under it says how many), or how many shrunk. */
export function pictureHeadline(entry: PictureNotice): string {
  const p = CHAT.pictureNote;
  if (entry.what === 'dropped') return p.dropped;
  return entry.count === 1 ? p.shrunk : fill(p.shrunkMany, { n: entry.count });
}

/**
 * Pictures left out (warning, the agent's notes under it) or made smaller (neutral; one picture's sizes as facts,
 * several as lines, else the agent's words).
 */
export function pictureRow(entry: PictureNotice): NoticeRowView {
  const text = pictureHeadline(entry);
  const notes = entry.notes.slice(0, PICTURE_LINES_MAX).join('\n');
  const detail = notes === '' ? {} : { detail: notes };
  if (entry.what === 'dropped') return { label: CHAT.pictureNote.label, text, facts: [], tone: 'warning', ...detail };
  const sizes = entry.sizes ?? [];
  if (sizes.length === 1)
    return {
      label: CHAT.pictureNote.label,
      text,
      facts: [pictureSize(sizes[0]!), picturePixels(sizes[0]!)],
      tone: 'neutral',
    };
  if (sizes.length > 1) {
    const lines = sizes.slice(0, PICTURE_LINES_MAX).map((size) => `${pictureSize(size)} · ${picturePixels(size)}`);
    return { label: CHAT.pictureNote.label, text, facts: [], tone: 'neutral', detail: lines.join('\n') };
  }
  return { label: CHAT.pictureNote.label, text, facts: [], tone: 'neutral', ...detail };
}

/** A turn whose prompt a hook blocked: sending it again meets the same hook. */
const HOOK_STOPPED: OutcomeLine = { tone: 'neutral', glyph: 'stop', text: CHAT.outcomes['hook-stopped'], action: null };

export type HookNotice = Extract<TranscriptEntry, { kind: 'notice'; code: 'hook' }>;
export type HookNoteNotice = Extract<TranscriptEntry, { kind: 'notice'; code: 'hook-note' }>;

/** The app's own Work guard (`profiles.ts` `workGuardHookName`) as the reducer keeps its name: no spec path tail. */
const GUARD_HEAD = 'plugin/mlx-work-guard-';
const GUARD_TAIL = '/plugin';
const GUARD_SUFFIX_CHARS = 16;

/** The hook's name as the row shows it: the app's own guard in words, any other as sent. */
export function hookName(hook: string): string {
  if (hook.startsWith(GUARD_HEAD) && hook.endsWith(GUARD_TAIL)) {
    const suffix = hook.slice(GUARD_HEAD.length, hook.length - GUARD_TAIL.length);
    let hex = suffix.length === GUARD_SUFFIX_CHARS;
    for (let i = 0; hex && i < suffix.length; i += 1) hex = '0123456789abcdef'.includes(suffix.charAt(i));
    if (hex) return CHAT.hook.ownGuard;
  }
  return visibleText(hook);
}

/** A tool as the hook words name it: the command, the edit, the read, else its name. */
function hookTool(tool: string | undefined): string {
  const t = CHAT.hook.tools;
  if (tool === undefined || tool === '') return t.any;
  if (tool === 'run_terminal_command') return t.command;
  if (tool === 'search_replace' || tool === 'write') return t.edit;
  if (tool === 'read_file') return t.read;
  return visibleText(tool);
}

/** When the hook ran: "before the command", "on your message", "before the answer ends", else the agent's event name. */
export function hookWhen(event: string, tool: string | undefined): string {
  const w = CHAT.hook.when;
  switch (event) {
    case 'pre_tool_use':
      return fill(w.pre_tool_use, { tool: hookTool(tool) });
    case 'post_tool_use':
      return fill(w.post_tool_use, { tool: hookTool(tool) });
    case 'user_prompt_submit':
    case 'stop':
    case 'session_start':
    case 'session_end':
      return w[event];
    default:
      return fill(w.other, { event: visibleText(event) });
  }
}

/** A failed hook: its name and when it ran, the error under it; after the first failure the count and the newest. */
export function hookRow(entry: HookNotice): NoticeRowView {
  const text = `${hookName(entry.hook)} · ${hookWhen(entry.event, entry.tool)}`;
  const facts =
    entry.times > 1
      ? [
          fill(CHAT.hook.times, { n: formatCount(entry.times) }),
          fill(CHAT.hook.last, { time: formatClock(entry.lastTs) }),
        ]
      : [];
  const detail = entry.detail === '' ? {} : { detail: entry.detail };
  return { label: CHAT.hook.failed, text, facts, tone: 'warning', ...detail };
}

/** A hook's note: the agent's words as the row's text. */
export function hookNoteRow(entry: HookNoteNotice): NoticeRowView {
  return { label: CHAT.hook.note, text: visibleText(entry.text), facts: [], tone: 'neutral' };
}

/** The first line of a hook row, said once when it arrives. */
export function hookHeadline(entry: HookNotice | HookNoteNotice): string {
  const view = entry.code === 'hook' ? hookRow(entry) : hookNoteRow(entry);
  return `${view.label}: ${view.text}`;
}

/** The wait line of a slow hook batch: how many run, and when. */
export function hookWaitLine(run: HookRunState): string {
  const head = run.count > 1 ? fill(CHAT.hook.waitMany, { n: run.count }) : CHAT.hook.wait;
  return `${head} · ${hookWhen(run.event, run.tool)}`;
}

/** The time of a thought, from its first to its last live chunk; null for a thought read from disk. */
export function thoughtDuration(entry: { ts: number; endTs?: number; timed?: true }): number | null {
  return entry.timed === true && entry.ts > 0 && entry.endTs !== undefined && entry.endTs > entry.ts
    ? entry.endTs - entry.ts
    : null;
}

/** The last turn holds only the automatic `continue` so far. */
export function continuingTurn(turn: ThreadTurn | undefined, continues: number | undefined): boolean {
  return (
    (continues ?? 0) > 0 &&
    turn !== undefined &&
    turn.end === null &&
    turn.entries.some((entry) => entry.kind === 'user' && entry.continued === true) &&
    turn.entries.every((entry) => entry.kind === 'model' || (entry.kind === 'user' && entry.continued === true))
  );
}

/**
 * `loading`: the engine loads the card's model. `other-first`: it loads another one first. With no load reported:
 * `busy` when the model is not in memory, `in-memory` when it is.
 */
export type LoadVariant = 'loading' | 'other-first' | 'busy' | 'in-memory';

export function loadVariant(
  modelId: string,
  loading: { modelId: string } | null,
  resident: ReadonlyArray<{ modelId: string }>,
): LoadVariant {
  if (loading === null) return resident.some((entry) => entry.modelId === modelId) ? 'in-memory' : 'busy';
  return loading.modelId === modelId ? 'loading' : 'other-first';
}

/**
 * The concurrent limit of `model` when its running answers fill that limit, a request waits in its queue, and every
 * running answer started before this turn (`turnStartedAt`), so none of them is this turn's own request; else null.
 */
export function fullLimit(
  answers: ReadonlyArray<{
    modelId: string;
    inFlight: number;
    maxConcurrent: number;
    queued: number;
    lastStart: number | null;
  }>,
  model: string | null,
  turnStartedAt: number | null,
): number | null {
  const entry = answers.find((a) => a.modelId === model);
  if (entry === undefined || entry.maxConcurrent < 1 || turnStartedAt === null) return null;
  if (entry.lastStart === null || entry.lastStart >= turnStartedAt) return null;
  return entry.inFlight >= entry.maxConcurrent && entry.queued > 0 ? entry.maxConcurrent : null;
}

/** `m:ss`, from one hour `h:mm:ss`; never below zero. */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const seconds = total % 60;
  const minutes = Math.floor(total / 60) % 60;
  const hours = Math.floor(total / 3600);
  const ss = String(seconds).padStart(2, '0');
  if (hours === 0) return `${String(minutes)}:${ss}`;
  return `${String(hours)}:${String(minutes).padStart(2, '0')}:${ss}`;
}

export interface EngineNoteInput {
  loading: { modelId: string } | null;
  /** The last end was before the prompt went to the agent. */
  unsent: boolean;
  /** The conversation's model. */
  model: string | null;
  resident: ReadonlyArray<{ modelId: string; sizeBytes: number | null }>;
  formatSize: (bytes: number) => string;
}

/** Line 2 of the last end line: the engine still loads, or the conversation's model is in memory. First match wins. */
export function engineNote(input: EngineNoteInput): EngineNote | null {
  if (input.loading !== null)
    return { text: fill(CHAT.engine.stillLoading, { model: input.loading.modelId }), action: 'open-engine' };
  if (!input.unsent || input.model === null) return null;
  const resident = input.resident.find((entry) => entry.modelId === input.model);
  if (resident === undefined) return null;
  return {
    text:
      resident.sizeBytes === null
        ? fill(CHAT.engine.residentNoSize, { model: input.model })
        : fill(CHAT.engine.resident, { model: input.model, size: input.formatSize(resident.sizeBytes) }),
    action: 'unload',
  };
}

/** A conversation's title, or the page's own words for one with no title yet. */
export function chatTitle(title: string, mode: ChatMode): string {
  const trimmed = title.trim();
  if (trimmed !== '') return trimmed;
  return mode === 'work' ? CHAT.newWork : CHAT.newChat;
}
