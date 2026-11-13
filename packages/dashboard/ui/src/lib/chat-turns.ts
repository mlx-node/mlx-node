/**
 * The thread as turns: where each turn starts and ends, the words of how it ended (live from the `turn` event, or from
 * the stored `turn-end`), its numbers, and the Markdown a copy puts on the clipboard.
 */

import { CHAT, fill } from './chat-strings';
import type { ChatFailure, ChatMode, TurnOutcome } from './chat-types';
import { formatCountNoun, formatDuration, formatNumber, formatTps } from './format';
import type { TranscriptEntry, TurnMetric } from './types';
import { oneLine, parseWebCall } from './web-tools';

type TurnEnd = Extract<TranscriptEntry, { kind: 'turn-end' }>;

export interface ThreadTurn {
  /** Index of the turn's first entry: stable while entries are appended. */
  start: number;
  entries: TranscriptEntry[];
  /** The stored end; null while the turn is open or when it never got one. */
  end: TurnEnd | null;
  metric: TurnMetric | null;
}

function startsTurn(entry: TranscriptEntry): boolean {
  return (entry.kind === 'user' && entry.interjection !== true) || entry.kind === 'wake';
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
export type TurnActionKind = 'send-again' | 'continue' | 'update-agent';

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

/** The live outcome of a turn, or null for `completed`. `unsent`: the turn ended before its prompt went to the agent. */
export function liveOutcome(outcome: TurnOutcome, mode: ChatMode, unsent = false): OutcomeLine | null {
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
    case 'address-refused':
      return { tone: 'warning', glyph: 'alert', text: o['address-refused'], action: 'send-again' };
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
    case 'refused':
      return {
        tone: 'danger',
        glyph: 'alert',
        text: mode === 'work' ? o['refused-work'] : o['refused-chat'],
        action: 'update-agent',
      };
  }
}

/** The outcome of a stored turn, from its `turn-end`. */
export function storedOutcome(end: TurnEnd): OutcomeLine | null {
  const o = CHAT.outcomes;
  const detail = end.error !== undefined && end.error !== '' ? end.error : undefined;
  let line: OutcomeLine | null;
  if (end.stopReason === 'end_turn') line = null;
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
  live: { outcome?: TurnOutcome; failure?: ChatFailure; unsent?: true } | null;
  mode: ChatMode;
}

/**
 * The outcome line of a turn: live for the last turn when the stream saw it end, else stored, else "did not finish".
 * A finished turn always has a stored end, so an earlier turn without one did not finish either.
 */
export function turnOutcome(turn: ThreadTurn, state: TurnEndState): OutcomeLine | null {
  if (state.last && state.running) return null;
  if (state.last && state.live !== null && state.live.outcome !== undefined)
    return liveOutcome(state.live.outcome, state.mode, state.live.unsent === true);
  if (turn.end !== null) return storedOutcome(turn.end);
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

/** The stats tooltip; parts with no value are left out. */
export function turnStatsTitle(metric: TurnMetric | null, model: string | null): string | undefined {
  if (metric === null) return undefined;
  const parts: string[] = [];
  if (metric.ttftMs !== null && metric.ttftMs > 0) parts.push(`First token ${formatDuration(metric.ttftMs)}`);
  if (metric.inputTokens > 0) {
    const cached = metric.cachedTokens > 0 ? `, ${formatNumber(metric.cachedTokens)} from cache` : '';
    parts.push(`${formatNumber(metric.inputTokens)} tokens in${cached}`);
  }
  if (metric.outputTokens > 0) parts.push(`${formatNumber(metric.outputTokens)} tokens out`);
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
  const user = turn.entries.find((entry): entry is Extract<TranscriptEntry, { kind: 'user' }> => entry.kind === 'user');
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

export interface PendingInput {
  creating: boolean;
  phase: 'starting' | 'running' | 'cancelling' | null;
  wait: 'model' | null;
  /** The conversation's model. */
  model: string | null;
  /** The model the engine loads now (`ChatHostState.loading`). */
  loading: { modelId: string; startedAt: number } | null;
  inference: 'ready' | 'starting' | 'loading' | 'stopped' | null;
  /** No stream event for 3 s while running. */
  quiet: boolean;
  /** The conversation's model is full: see {@link fullLimit}. */
  full: number | null;
}

export type PendingRow = { kind: 'label'; text: string; orb: 'hold' | 'think' } | { kind: 'load'; modelId: string };

/** What the pending row shows, first match wins. */
export function pendingRow(input: PendingInput): PendingRow {
  const label = (text: string, orb: 'hold' | 'think' = 'hold'): PendingRow => ({ kind: 'label', text, orb });
  if (input.creating) return label(CHAT.pendingStarting);
  if (input.phase === 'cancelling') return label(CHAT.pendingStopping);
  if (input.wait === 'model' && input.model !== null) return { kind: 'load', modelId: input.model };
  if (input.phase === 'running' && input.loading !== null) return { kind: 'load', modelId: input.loading.modelId };
  if (input.phase === 'starting') return label(CHAT.pendingStarting);
  if (input.inference === 'starting') return label(CHAT.pendingEngine);
  if (input.phase === 'running' && input.full !== null && input.model !== null)
    return label(
      input.full === 1
        ? fill(CHAT.pendingQueuedOne, { model: input.model })
        : fill(CHAT.pendingQueuedMany, { model: input.model, n: String(input.full) }),
    );
  if (input.quiet) return label(CHAT.pendingWaiting);
  return label(CHAT.pendingThinking, 'think');
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
