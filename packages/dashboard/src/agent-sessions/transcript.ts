/**
 * grok `updates.jsonl` → display transcript + per-turn metrics. Pure data → data.
 *
 * Line handling follows grok-build `session/storage`: a line with `params` is an
 * envelope (`method` `_x.ai/session/update` = xAI, anything else = ACP); a line
 * without `params` is a legacy bare ACP notification. Rewinds are applied to the
 * raw lines first, exactly like grok's `filter_rewind_lines`.
 */

import type { TranscriptEntry, TurnMetric } from './types.js';

const XAI_METHOD = '_x.ai/session/update';
const REWIND_MARKER = 'rewind_marker';

/** Per-field cap for tool input/output sent to the UI (a single line can exceed 300 KB). */
export const MAX_FIELD_CHARS = 16 * 1024;

export interface ParsedUpdates {
  transcript: TranscriptEntry[];
  turns: TurnMetric[];
  /** Number of `turn_completed` lines that carried `usage`. */
  turnsWithUsage: number;
  /** A user prompt or tool call is not yet followed by `turn_completed`. */
  openTurn: boolean;
}

type Json = Record<string, unknown>;

function isRecord(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isUsize(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function str(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

/**
 * Complete, non-blank lines. Text after the last `\n` is a torn append still in
 * flight and is dropped; surrounding ASCII whitespace is trimmed like grok does.
 */
export function splitCompleteLines(text: string): string[] {
  const end = text.lastIndexOf('\n');
  if (end < 0) return [];
  const out: string[] = [];
  for (const raw of text.slice(0, end).split('\n')) {
    let a = 0;
    let b = raw.length;
    while (a < b && isAsciiSpace(raw.charCodeAt(a))) a++;
    while (b > a && isAsciiSpace(raw.charCodeAt(b - 1))) b--;
    if (b > a) out.push(a === 0 && b === raw.length ? raw : raw.slice(a, b));
  }
  return out;
}

/** Rust `u8::is_ascii_whitespace` (no vertical tab). */
function isAsciiSpace(code: number): boolean {
  return code === 32 || code === 9 || code === 10 || code === 12 || code === 13;
}

type RewindStep = { kind: 'rewind'; target: number } | { kind: 'user'; promptIndex: number | null } | { kind: 'other' };

const OTHER: RewindStep = { kind: 'other' };

/** grok `rewind_step_for_line`, including where its serde peek would fail (→ other). */
function rewindStep(line: string): RewindStep {
  let peek: unknown;
  try {
    peek = JSON.parse(line);
  } catch {
    return OTHER;
  }
  let raw: unknown = peek;
  let isXai = false;
  if (isRecord(peek) && (peek.method === undefined || peek.method === null || typeof peek.method === 'string')) {
    if (peek.params !== undefined && peek.params !== null) raw = peek.params;
    isXai = peek.method === XAI_METHOD;
  }
  if (!isRecord(raw)) return OTHER;
  const u = raw.update;
  if (u === undefined || u === null) return OTHER;
  if (!isRecord(u) || typeof u.sessionUpdate !== 'string') return OTHER;
  if (u.status !== undefined && u.status !== null && typeof u.status !== 'string') return OTHER;
  const target = u.target_prompt_index;
  if (target !== undefined && target !== null && !isUsize(target)) return OTHER;
  const meta = u._meta;
  let promptIndex: number | null = null;
  let hostTurn = false;
  if (meta !== undefined && meta !== null) {
    if (!isRecord(meta)) return OTHER;
    const pi = meta.promptIndex;
    if (pi !== undefined && pi !== null) {
      if (!isUsize(pi)) return OTHER;
      promptIndex = pi;
    }
    const ht = meta.hostTurn;
    if (ht !== undefined && ht !== null) {
      if (typeof ht !== 'boolean') return OTHER;
      hostTurn = ht;
    }
  }
  if (isXai && u.sessionUpdate === REWIND_MARKER && isUsize(target)) return { kind: 'rewind', target };
  if (!isXai && !hostTurn && u.sessionUpdate === 'user_message_chunk') return { kind: 'user', promptIndex };
  return OTHER;
}

/** grok `UserRunTurnTracker`: which user chunks open a counted prompt. */
class UserRunTracker {
  private seenMarker = false;
  private inUser = false;
  private currentPi: number | null = null;

  onUser(pi: number | null): boolean {
    if (pi !== null) this.seenMarker = true;
    const counts = this.seenMarker ? pi !== null : true;
    const newRun = !this.inUser ? true : this.seenMarker || pi !== null ? pi !== this.currentPi : false;
    this.inUser = true;
    if (!newRun) return false;
    this.currentPi = pi;
    return counts;
  }

  onNonUser(): void {
    this.inUser = false;
    this.currentPi = null;
  }
}

/**
 * grok `filter_rewind_lines`: a marker's `target_prompt_index` is an ordinal into
 * the COUNTED user runs seen so far; everything from that run's first line up to
 * the marker is dropped, as is the marker. Out of range keeps everything.
 */
export function filterRewindLines(lines: string[]): string[] {
  if (!lines.some((l) => l.includes(REWIND_MARKER))) return lines;
  const result: string[] = [];
  const starts: number[] = [];
  const tracker = new UserRunTracker();
  for (const line of lines) {
    const step = rewindStep(line);
    if (step.kind === 'rewind') {
      result.length = step.target < starts.length ? starts[step.target]! : result.length;
      starts.length = Math.min(starts.length, step.target);
      tracker.onNonUser();
      continue;
    }
    if (step.kind === 'user') {
      if (tracker.onUser(step.promptIndex)) starts.push(result.length);
    } else {
      tracker.onNonUser();
    }
    result.push(line);
  }
  return result;
}

/**
 * Whether a string is raw binary read verbatim as "text" (a `read` of a PDF or
 * image). Control bytes other than tab/newline/CR never occur in real prose.
 */
export function isBinaryText(text: string): boolean {
  const limit = Math.min(text.length, 4096);
  for (let i = 0; i < limit; i++) {
    const code = text.charCodeAt(i);
    if (code <= 8 || code === 11 || code === 12 || (code >= 14 && code <= 31)) return true;
  }
  return false;
}

function sizeLabel(bytes: number): string {
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

/** A short `PDF · 32 KB`-style label for a binary blob, sniffing common magics. */
export function describeBinary(text: string): string {
  let format = 'binary';
  if (text.startsWith('%PDF')) format = 'PDF';
  else if (/^.{0,16}ftyp(heic|heix|hevc|mif1|msf1)/.test(text)) format = 'HEIC';
  else if (text.startsWith('\u0089PNG')) format = 'PNG';
  else if (text.startsWith('ÿØÿ')) format = 'JPEG';
  else if (text.startsWith('GIF8')) format = 'GIF';
  else if (text.startsWith('PK')) format = 'ZIP';
  return `[${format} · ${sizeLabel(text.length)}]`;
}

/** Cap a field at {@link MAX_FIELD_CHARS} without splitting a surrogate pair. */
export function truncateField(text: string, max = MAX_FIELD_CHARS): { text: string; truncated: boolean } {
  if (text.length <= max) return { text, truncated: false };
  let cut = max;
  const code = text.charCodeAt(cut - 1);
  if (code >= 0xd800 && code <= 0xdbff) cut -= 1;
  return { text: text.slice(0, cut), truncated: true };
}

function fieldText(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  return isBinaryText(text) ? describeBinary(text) : text;
}

function contentBlockText(block: unknown): string {
  if (!isRecord(block)) return '';
  switch (block.type) {
    case 'text':
      return str(block.text) ?? '';
    case 'image':
      return '[image]';
    case 'audio':
      return '[audio]';
    case 'resource_link':
      return `[${str(block.name) ?? str(block.uri) ?? 'resource'}]`;
    case 'resource':
      return '[resource]';
    default:
      return '';
  }
}

/** ACP `ToolCallContent[]` → display text. */
function toolContentText(content: unknown[]): string {
  const parts: string[] = [];
  for (const item of content) {
    if (!isRecord(item)) continue;
    if (item.type === 'content') {
      const text = contentBlockText(item.content);
      if (text !== '') parts.push(text);
    } else if (item.type === 'diff') {
      const path = str(item.path) ?? '';
      parts.push(`[diff] ${path}\n${str(item.newText) ?? ''}`);
    } else if (item.type === 'terminal') {
      parts.push('[terminal]');
    }
  }
  return parts.join('\n');
}

interface ToolAcc {
  entry: Extract<TranscriptEntry, { kind: 'tool' }>;
  rawInput: unknown;
  content: string | null;
  rawOutput: unknown;
}

function toolName(update: Json): string | null {
  const meta = update._meta;
  if (!isRecord(meta)) return null;
  const tool = meta['x.ai/tool'];
  return isRecord(tool) ? str(tool.name) : null;
}

interface Notification {
  xai: boolean;
  update: Json;
  ts: number;
}

/** One line → its notification, or `null` for a line grok's reader would skip. */
function parseLine(line: string): Notification | null {
  let obj: unknown;
  try {
    obj = JSON.parse(line);
  } catch {
    return null;
  }
  if (!isRecord(obj)) return null;
  let notification: unknown = obj;
  let xai = false;
  if ('params' in obj) {
    notification = obj.params;
    xai = obj.method === XAI_METHOD;
  }
  if (!isRecord(notification) || !isRecord(notification.update)) return null;
  if (typeof notification.update.sessionUpdate !== 'string') return null;
  const meta = notification._meta;
  const agentMs = isRecord(meta) ? num(meta.agentTimestampMs) : null;
  const seconds = num(obj.timestamp);
  return { xai, update: notification.update, ts: agentMs ?? (seconds !== null ? seconds * 1000 : 0) };
}

function turnModel(usage: Json | null, fallback: string | null): string | null {
  const modelUsage = usage !== null && isRecord(usage.modelUsage) ? usage.modelUsage : null;
  if (modelUsage === null) return fallback;
  let best: string | null = null;
  let bestOut = -1;
  for (const [model, row] of Object.entries(modelUsage)) {
    const out = isRecord(row) ? (num(row.outputTokens) ?? 0) : 0;
    if (out > bestOut) {
      best = model;
      bestOut = out;
    }
  }
  return best ?? fallback;
}

export function parseUpdates(text: string): ParsedUpdates {
  const lines = filterRewindLines(splitCompleteLines(text));
  const transcript: TranscriptEntry[] = [];
  const turns: TurnMetric[] = [];
  const tools = new Map<string, ToolAcc>();
  const subagents = new Map<string, Extract<TranscriptEntry, { kind: 'subagent' }>>();
  let model: string | null = null;
  let openTurn = false;
  let turnsWithUsage = 0;

  const last = (): TranscriptEntry | undefined => transcript[transcript.length - 1];

  const tool = (update: Json, ts: number): ToolAcc | null => {
    const id = str(update.toolCallId);
    if (id === null || id === '') return null;
    let acc = tools.get(id);
    if (acc === undefined) {
      const title = str(update.title) ?? '';
      acc = {
        entry: { kind: 'tool', ts, id, name: toolName(update) ?? (title || 'tool'), title, status: 'pending' },
        rawInput: undefined,
        content: null,
        rawOutput: undefined,
      };
      tools.set(id, acc);
      transcript.push(acc.entry);
    }
    const name = toolName(update);
    if (name !== null) acc.entry.name = name;
    const title = str(update.title);
    if (title !== null) acc.entry.title = title;
    const status = str(update.status);
    if (status !== null) acc.entry.status = status;
    if (update.rawInput !== undefined) acc.rawInput = update.rawInput;
    if (Array.isArray(update.content)) acc.content = toolContentText(update.content);
    if (update.rawOutput !== undefined) acc.rawOutput = update.rawOutput;
    return acc;
  };

  for (const line of lines) {
    const n = parseLine(line);
    if (n === null) continue;
    const { update, ts } = n;
    const tag = update.sessionUpdate;
    if (!n.xai) {
      switch (tag) {
        case 'user_message_chunk': {
          const meta = isRecord(update._meta) ? update._meta : {};
          if (str(meta.modelId) !== null) model = meta.modelId as string;
          openTurn = true;
          // grok's pager shows hostTurn echoes (slash commands); hostTurn only exempts them from rewind counting.
          if (meta.hideFromScrollback === true) break;
          const content = update.content;
          const displayText =
            isRecord(content) && content.type === 'text' && isRecord(content._meta)
              ? str(content._meta.displayText)
              : null;
          const body = displayText ?? contentBlockText(content);
          if (body === '') break;
          const promptIndex = isUsize(meta.promptIndex) ? meta.promptIndex : null;
          const interjection = meta.interjection === true;
          const prev = last();
          if (
            prev?.kind === 'user' &&
            !interjection &&
            prev.interjection !== true &&
            promptIndex !== null &&
            prev.promptIndex === promptIndex
          ) {
            prev.text += `\n${body}`;
          } else {
            transcript.push({
              kind: 'user',
              ts,
              text: body,
              promptIndex,
              ...(interjection ? { interjection: true } : {}),
            });
          }
          break;
        }
        case 'agent_message_chunk':
        case 'agent_thought_chunk': {
          const content = update.content;
          if (!isRecord(content) || content.type !== 'text') break;
          const body = str(content.text) ?? '';
          if (body === '') break;
          const kind = tag === 'agent_message_chunk' ? 'assistant' : 'thinking';
          const prev = last();
          if (prev?.kind === kind) prev.text += body;
          else transcript.push({ kind, ts, text: body });
          break;
        }
        case 'tool_call':
          if (tool(update, ts) !== null) openTurn = true;
          break;
        case 'tool_call_update':
          tool(update, ts);
          break;
        case 'plan': {
          if (!Array.isArray(update.entries)) break;
          transcript.push({
            kind: 'plan',
            ts,
            entries: update.entries.filter(isRecord).map((e) => ({
              content: str(e.content) ?? '',
              status: str(e.status) ?? 'pending',
              priority: str(e.priority) ?? 'medium',
            })),
          });
          break;
        }
      }
      continue;
    }
    switch (tag) {
      case 'turn_completed': {
        openTurn = false;
        const usage = isRecord(update.usage) ? update.usage : null;
        if (usage !== null) turnsWithUsage += 1;
        const perf = isRecord(update.perf) ? update.perf : null;
        const elapsedMs = num(update.elapsed_ms);
        const stopReason = str(update.stop_reason) ?? 'unknown';
        const errorText = str(update.agent_result);
        const isError = stopReason === 'error' || str(update.error_kind) !== null;
        transcript.push({
          kind: 'turn-end',
          ts,
          stopReason,
          ...(elapsedMs !== null ? { elapsedMs } : {}),
          ...(isError && errorText !== null && errorText !== '' ? { error: errorText } : {}),
        });
        turns.push({
          promptId: str(update.prompt_id) ?? '',
          ts,
          inputTokens: num(usage?.inputTokens) ?? 0,
          outputTokens: num(usage?.outputTokens) ?? 0,
          cachedTokens: num(usage?.cachedReadTokens) ?? 0,
          reasoningTokens: num(usage?.reasoningTokens) ?? 0,
          elapsedMs,
          ttftMs: num(perf?.ttft_ms),
          prefillTps: num(perf?.prefill_tokens_per_second),
          decodeTps: num(perf?.decode_tokens_per_second),
          model: turnModel(usage, model),
        });
        break;
      }
      case 'subagent_spawned': {
        const childId = str(update.child_session_id);
        if (childId === null || childId === '') break;
        const entry: Extract<TranscriptEntry, { kind: 'subagent' }> = {
          kind: 'subagent',
          ts,
          childId,
          type: str(update.subagent_type) ?? '',
          description: str(update.description) ?? '',
        };
        subagents.set(childId, entry);
        transcript.push(entry);
        break;
      }
      case 'subagent_finished': {
        const childId = str(update.child_session_id);
        if (childId === null || childId === '') break;
        let entry = subagents.get(childId);
        if (entry === undefined) {
          entry = { kind: 'subagent', ts, childId, type: '', description: '' };
          subagents.set(childId, entry);
          transcript.push(entry);
        }
        const status = str(update.status);
        if (status !== null) entry.status = status;
        const durationMs = num(update.duration_ms);
        if (durationMs !== null) entry.durationMs = durationMs;
        const tokens = num(update.tokens_used);
        if (tokens !== null) entry.tokens = tokens;
        const error = str(update.error);
        if (error !== null && error !== '') entry.error = error;
        break;
      }
      case 'model_changed': {
        const modelId = str(update.model_id);
        if (modelId === null || modelId === '') break;
        model = modelId;
        transcript.push({ kind: 'model', ts, modelId });
        break;
      }
    }
  }

  for (const acc of tools.values()) {
    const input = fieldText(acc.rawInput);
    const output = acc.content !== null && acc.content !== '' ? fieldText(acc.content) : fieldText(acc.rawOutput);
    let truncated = false;
    if (input !== undefined) {
      const t = truncateField(input);
      acc.entry.input = t.text;
      truncated ||= t.truncated;
    }
    if (output !== undefined) {
      const t = truncateField(output);
      acc.entry.output = t.text;
      truncated ||= t.truncated;
    }
    if (truncated) acc.entry.truncated = true;
  }

  return { transcript, turns, turnsWithUsage, openTurn };
}
