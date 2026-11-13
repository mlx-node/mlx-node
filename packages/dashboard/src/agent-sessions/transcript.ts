/**
 * grok `updates.jsonl` → display transcript + per-turn metrics. Pure data → data.
 *
 * Line handling follows grok-build `session/storage`: a line with `params` is an
 * envelope (`method` `_x.ai/session/update` = xAI, anything else = ACP); a line
 * without `params` is a legacy bare ACP notification. Rewinds are applied to the
 * raw lines first, exactly like grok's `filter_rewind_lines`.
 */

import type { TranscriptOp } from '../chat/types.js';
import { mcpOutputText } from './mcp-output.js';
import type { TranscriptEntry, TurnMetric, WakeCause } from './types.js';

const XAI_METHOD = '_x.ai/session/update';
const REWIND_MARKER = 'rewind_marker';
const MAX_DATE_MS = 8_640_000_000_000_000;

/** Per-field cap for tool input/output sent to the UI (a single line can exceed 300 KB). */
export const MAX_FIELD_CHARS = 16 * 1024;

export interface ParsedUpdates {
  transcript: TranscriptEntry[];
  turns: TurnMetric[];
  /** Number of `turn_completed` lines that carried `usage`. */
  turnsWithUsage: number;
  /** A user prompt or tool call is not yet followed by `turn_completed`. */
  openTurn: boolean;
  /** Model request retries (`retry_state` lines of type `retrying`), in file order. */
  retries: Array<{ ts: number; attempt: number; max: number }>;
  /** Largest line time of the file, also of lines that make no entry; 0 when no line has a time. */
  lastTs: number;
  /**
   * `_meta['x.ai/tool'].name` by tool call id. A tool entry's `name` falls back to its title, which can hold the call's
   * arguments; metadata-only readers take the name from here.
   */
  toolNames: ReadonlyMap<string, string>;
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

/** An integer ≥ 1, else null. */
function countOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 1 ? value : null;
}

/** `prefill_ms` and `prefill_tokens` of a `perf` block as a pair, or null. */
function measuredPrefill(perf: Json | null): { ms: number; tokens: number } | null {
  const ms = num(perf?.prefill_ms);
  const tokens = countOrNull(perf?.prefill_tokens);
  if (ms === null || ms <= 0 || tokens === null) return null;
  return { ms, tokens };
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

/**
 * `endTs` is emitted on tool and text entries but typed only here: the exported
 * `TranscriptEntry` must stay equal to the SPA's copy (`sessions-pages.test.ts`).
 */
type Timed<K extends TranscriptEntry['kind']> = Extract<TranscriptEntry, { kind: K }> & { endTs?: number };

interface ToolAcc {
  entry: Timed<'tool'>;
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
  return { xai, update: notification.update, ts: epochMs(agentMs ?? (seconds !== null ? seconds * 1000 : 0)) };
}

/** A time outside `Date`'s range (or one that overflowed to Infinity) is 0: untimed, like a missing one. */
function epochMs(ms: number): number {
  return Number.isFinite(ms) && ms > 0 && ms <= MAX_DATE_MS ? ms : 0;
}

const WAKE_PREFIXES: ReadonlyArray<readonly [string, WakeCause]> = [
  ['workflow-completed-', 'workflow'],
  ['subagent-completed-', 'subagent'],
  ['task-completed-', 'task'],
  ['notifications-', 'notifications'],
  ['goal-summary-', 'goal'],
  ['goal-classifier-nudge-', 'goal'],
];

/** grok `PromptOrigin::from_prompt_id`: the prompt id prefix names what woke the agent. */
export function wakeCause(promptId: string | null): WakeCause {
  if (promptId === null) return 'unknown';
  for (const [prefix, cause] of WAKE_PREFIXES) if (promptId.startsWith(prefix)) return cause;
  return 'unknown';
}

type WakeEntry = Extract<TranscriptEntry, { kind: 'wake' }>;
type SubagentEntry = Extract<TranscriptEntry, { kind: 'subagent' }>;
type WorkflowRun = { name: string; status: string };
type WorkflowState = Pick<ReadonlyMap<string, WorkflowRun>, 'get'>;
/** Each run's state from before its first update since a wake; undefined = the run was unknown then. */
type StatesAtWake = Map<string, WorkflowRun | undefined>;

/** Sets `cause` and the matching `workflow` / `subagent` field of a wake entry from a prompt id. */
function applyWake(
  entry: WakeEntry,
  promptId: string | null,
  workflows: WorkflowState,
  spawnedBy: ReadonlyMap<string, SubagentEntry>,
): void {
  const cause = wakeCause(promptId);
  entry.cause = cause;
  delete entry.workflow;
  delete entry.subagent;
  if (promptId === null) return;
  if (cause === 'workflow') {
    const rest = promptId.slice('workflow-completed-'.length);
    const dash = rest.lastIndexOf('-');
    const run = workflows.get(dash === -1 ? rest : rest.slice(0, dash));
    if (run !== undefined && run.name !== '') entry.workflow = { name: run.name, status: run.status };
  } else if (cause === 'subagent') {
    const child = spawnedBy.get(promptId.slice('subagent-completed-'.length));
    if (child !== undefined) entry.subagent = { childId: child.childId, description: child.description };
  }
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

/** One session update: from disk (`xai` = `_x.ai/session/update`) or live (`xai` = `_x.ai/session_notification`). */
export interface UpdateLine {
  xai: boolean;
  update: Record<string, unknown>;
  meta: Record<string, unknown> | undefined;
}

interface ReducerOptions {
  /** Time of a live line that carries no `agentTimestampMs`. */
  now?: () => number;
}

/**
 * The transcript reducer shared by the disk reader and the live stream. `apply` returns the ops that turn the previous
 * entry list into the new one; `snapshot` is the finished list. Ops copy entries, so a later change never reaches an
 * op already handed out.
 */
export class TranscriptReducer {
  private readonly transcript: TranscriptEntry[] = [];
  private readonly turns: TurnMetric[] = [];
  private readonly indexes = new WeakMap<TranscriptEntry, number>();
  private readonly tools = new Map<string, ToolAcc>();
  private readonly subagents = new Map<string, SubagentEntry>();
  private readonly workflows = new Map<string, WorkflowRun>();
  /** By `subagent_id`, else `child_session_id`. */
  private readonly spawnedBy = new Map<string, SubagentEntry>();
  /** `prompt_id` of the last `user_prompt_submit` hook line; null after a prompt chunk or a turn end. */
  private submitId: string | null = null;
  /** The first prompt entry since the last turn end; for a wake also the runs updated since it woke the agent. */
  private turnPrompt: { entry: WakeEntry; atWake: StatesAtWake } | { entry: TranscriptEntry; atWake: null } | null =
    null;
  private readonly wakeTexts = new Map<WakeEntry, string>();
  private wakeRun: { entry: WakeEntry; promptIndex: number } | null = null;
  private model: string | null = null;
  private openTurn = false;
  private turnsWithUsage = 0;
  private lastTs = 0;
  private readonly retries: ParsedUpdates['retries'] = [];
  private readonly toolNames = new Map<string, string>();
  private readonly now: () => number;
  private ops: TranscriptOp[] | null = null;

  constructor(options: ReducerOptions = {}) {
    this.now = options.now ?? Date.now;
  }

  /** Start from entries read earlier (a cold conversation's disk transcript); live ops index after them. */
  seed(entries: readonly TranscriptEntry[], turns: readonly TurnMetric[]): void {
    for (const entry of entries) this.add({ ...entry });
    for (const turn of turns) this.turns.push({ ...turn });
    for (let i = entries.length - 1; i >= 0; i--) {
      const entry = entries[i]!;
      if (entry.kind === 'model') {
        this.model = entry.modelId;
        break;
      }
    }
  }

  /** A whole `updates.jsonl` text, rewinds applied first. Makes no ops. */
  applyText(text: string): void {
    for (const line of filterRewindLines(splitCompleteLines(text))) {
      const n = parseLine(line);
      if (n !== null) this.step(n);
    }
  }

  /** One live line → its ops. */
  apply(line: UpdateLine): TranscriptOp[] {
    const agentMs = line.meta !== undefined ? num(line.meta.agentTimestampMs) : null;
    return this.collect(() => this.step({ xai: line.xai, update: line.update, ts: epochMs(agentMs ?? this.now()) }));
  }

  /** The user's prompt: the agent does not echo it live, so the host pushes it when the prompt is accepted. */
  userPrompt(text: string, ts: number = this.now()): TranscriptOp[] {
    return this.collect(() =>
      this.step({ xai: false, update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text } }, ts }),
    );
  }

  /** A model switch the user made between turns. */
  modelSwitch(modelId: string, ts: number = this.now()): TranscriptOp[] {
    return this.collect(() =>
      this.step({ xai: true, update: { sessionUpdate: 'model_changed', model_id: modelId }, ts }),
    );
  }

  snapshot(): { entries: TranscriptEntry[]; turns: TurnMetric[] } {
    for (const acc of this.tools.values()) finishTool(acc);
    return { entries: this.transcript, turns: this.turns };
  }

  parsed(): ParsedUpdates {
    const { entries, turns } = this.snapshot();
    return {
      transcript: entries,
      turns,
      turnsWithUsage: this.turnsWithUsage,
      openTurn: this.openTurn,
      retries: this.retries,
      lastTs: this.lastTs,
      toolNames: this.toolNames,
    };
  }

  private collect(run: () => void): TranscriptOp[] {
    const ops: TranscriptOp[] = [];
    this.ops = ops;
    try {
      run();
    } finally {
      this.ops = null;
    }
    return ops;
  }

  private add(entry: TranscriptEntry): void {
    this.indexes.set(entry, this.transcript.length);
    this.transcript.push(entry);
    this.ops?.push({ op: 'push', entry: { ...entry } });
  }

  private changed(entry: TranscriptEntry): void {
    if (this.ops === null) return;
    const index = this.indexes.get(entry);
    if (index !== undefined) this.ops.push({ op: 'set', index, entry: { ...entry } });
  }

  private last(): TranscriptEntry | undefined {
    return this.transcript[this.transcript.length - 1];
  }

  private setWakeText(entry: WakeEntry, text: string): void {
    this.wakeTexts.set(entry, text);
    const cut = truncateField(text);
    entry.text = cut.text;
    if (cut.truncated) entry.truncated = true;
    else delete entry.truncated;
  }

  private tool(update: Json, ts: number): ToolAcc | null {
    const id = str(update.toolCallId);
    if (id === null || id === '') return null;
    let acc = this.tools.get(id);
    const isNew = acc === undefined;
    if (acc === undefined) {
      const title = str(update.title) ?? '';
      acc = {
        entry: { kind: 'tool', ts, id, name: toolName(update) ?? (title || 'tool'), title, status: 'pending' },
        rawInput: undefined,
        content: null,
        rawOutput: undefined,
      };
      this.tools.set(id, acc);
    }
    const name = toolName(update);
    if (name !== null) {
      acc.entry.name = name;
      this.toolNames.set(id, name);
    }
    const title = str(update.title);
    if (title !== null) acc.entry.title = title;
    const status = str(update.status);
    if (status !== null) acc.entry.status = status;
    if (status === 'completed' || status === 'failed') acc.entry.endTs = ts;
    if (update.rawInput !== undefined) acc.rawInput = update.rawInput;
    if (Array.isArray(update.content)) acc.content = toolContentText(update.content);
    if (update.rawOutput !== undefined) acc.rawOutput = update.rawOutput;
    if (this.ops !== null) finishTool(acc);
    if (isNew) this.add(acc.entry);
    else this.changed(acc.entry);
    return acc;
  }

  private step(n: Notification): void {
    const { update, ts } = n;
    if (ts > this.lastTs) this.lastTs = ts;
    const tag = update.sessionUpdate;
    if (!n.xai) {
      switch (tag) {
        case 'user_message_chunk': {
          const meta = isRecord(update._meta) ? update._meta : {};
          if (str(meta.modelId) !== null) this.model = meta.modelId as string;
          this.openTurn = true;
          const hookId = this.submitId;
          this.submitId = null;
          const content = update.content;
          const displayText =
            isRecord(content) && content.type === 'text' && isRecord(content._meta)
              ? str(content._meta.displayText)
              : null;
          const body = displayText ?? contentBlockText(content);
          const promptIndex = isUsize(meta.promptIndex) ? meta.promptIndex : null;
          // A hidden chunk is a prompt the agent wrote itself, never a message of the user, even with hostTurn.
          if (meta.hideFromScrollback === true) {
            const prev = this.last();
            const run = this.wakeRun;
            if (run !== null && prev === run.entry && promptIndex === run.promptIndex) {
              this.setWakeText(run.entry, `${this.wakeTexts.get(run.entry) ?? ''}\n${body}`);
              this.changed(run.entry);
              break;
            }
            const entry: WakeEntry = { kind: 'wake', ts, cause: 'unknown', text: '' };
            applyWake(entry, hookId, this.workflows, this.spawnedBy);
            this.setWakeText(entry, body);
            this.add(entry);
            this.wakeRun = promptIndex === null ? null : { entry, promptIndex };
            if (this.turnPrompt === null) this.turnPrompt = { entry, atWake: new Map() };
            break;
          }
          if (body === '') break;
          const interjection = meta.interjection === true;
          const prev = this.last();
          if (
            prev?.kind === 'user' &&
            !interjection &&
            prev.interjection !== true &&
            promptIndex !== null &&
            prev.promptIndex === promptIndex
          ) {
            prev.text += `\n${body}`;
            this.changed(prev);
          } else {
            const entry: TranscriptEntry = {
              kind: 'user',
              ts,
              text: body,
              promptIndex,
              ...(interjection ? { interjection: true } : {}),
            };
            this.add(entry);
            if (this.turnPrompt === null && !interjection) this.turnPrompt = { entry, atWake: null };
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
          const prev = this.last();
          if (prev?.kind === kind) {
            const text = prev as Timed<'assistant' | 'thinking'>;
            text.text += body;
            text.endTs = ts;
            if (this.ops !== null) this.ops.push({ op: 'text', index: this.transcript.length - 1, append: body });
          } else {
            const entry: Timed<'assistant' | 'thinking'> = { kind, ts, text: body, endTs: ts };
            this.add(entry);
          }
          break;
        }
        case 'tool_call':
          if (this.tool(update, ts) !== null) this.openTurn = true;
          break;
        case 'tool_call_update':
          this.tool(update, ts);
          break;
        case 'plan': {
          if (!Array.isArray(update.entries)) break;
          this.add({
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
      return;
    }
    switch (tag) {
      case 'turn_completed': {
        this.openTurn = false;
        const promptId = str(update.prompt_id);
        const prompt = this.turnPrompt;
        if (prompt !== null && prompt.atWake !== null && wakeCause(promptId) !== 'unknown') {
          const atWake = prompt.atWake;
          const workflows = this.workflows;
          const then: WorkflowState = { get: (id) => (atWake.has(id) ? atWake.get(id) : workflows.get(id)) };
          applyWake(prompt.entry, promptId, then, this.spawnedBy);
          this.changed(prompt.entry);
        }
        this.turnPrompt = null;
        this.submitId = null;
        const usage = isRecord(update.usage) ? update.usage : null;
        if (usage !== null) this.turnsWithUsage += 1;
        const perf = isRecord(update.perf) ? update.perf : null;
        const prefill = measuredPrefill(perf);
        const elapsedMs = num(update.elapsed_ms);
        const stopReason = str(update.stop_reason) ?? 'unknown';
        const errorText = str(update.agent_result);
        const isError = stopReason === 'error' || str(update.error_kind) !== null;
        this.add({
          kind: 'turn-end',
          ts,
          stopReason,
          ...(elapsedMs !== null ? { elapsedMs } : {}),
          ...(isError && errorText !== null && errorText !== '' ? { error: errorText } : {}),
        });
        const metric: TurnMetric = {
          promptId: promptId ?? '',
          ts,
          inputTokens: num(usage?.inputTokens) ?? 0,
          outputTokens: num(usage?.outputTokens) ?? 0,
          cachedTokens: num(usage?.cachedReadTokens) ?? 0,
          reasoningTokens: num(usage?.reasoningTokens) ?? 0,
          elapsedMs,
          ttftMs: num(perf?.ttft_ms),
          prefillMs: prefill?.ms ?? null,
          prefillTokens: prefill?.tokens ?? null,
          prefillCalls: countOrNull(perf?.prefill_calls),
          modelCalls: countOrNull(usage?.modelCalls),
          decodeTps: num(perf?.decode_tokens_per_second),
          model: turnModel(usage, this.model),
        };
        this.turns.push(metric);
        this.ops?.push({ op: 'metric', metric: { ...metric } });
        break;
      }
      case 'subagent_spawned': {
        const childId = str(update.child_session_id);
        if (childId === null || childId === '') break;
        const entry: SubagentEntry = {
          kind: 'subagent',
          ts,
          childId,
          type: str(update.subagent_type) ?? '',
          description: str(update.description) ?? '',
        };
        this.subagents.set(childId, entry);
        const subagentId = str(update.subagent_id);
        this.spawnedBy.set(subagentId !== null && subagentId !== '' ? subagentId : childId, entry);
        this.add(entry);
        break;
      }
      case 'workflow_updated': {
        const runId = str(update.run_id);
        if (runId === null || runId === '') break;
        const atWake = this.turnPrompt?.atWake ?? null;
        if (atWake !== null && !atWake.has(runId)) atWake.set(runId, this.workflows.get(runId));
        this.workflows.set(runId, { name: str(update.name) ?? '', status: str(update.status) ?? '' });
        break;
      }
      case 'hook_execution':
        if (update.event_name === 'user_prompt_submit') this.submitId = str(update.prompt_id);
        break;
      case 'subagent_finished': {
        const childId = str(update.child_session_id);
        if (childId === null || childId === '') break;
        let entry = this.subagents.get(childId);
        const isNew = entry === undefined;
        if (entry === undefined) {
          entry = { kind: 'subagent', ts, childId, type: '', description: '', noSpawn: true };
          this.subagents.set(childId, entry);
        }
        entry.endTs = ts;
        const status = str(update.status);
        if (status !== null) entry.status = status;
        const durationMs = num(update.duration_ms);
        if (durationMs !== null) entry.durationMs = durationMs;
        const tokens = num(update.tokens_used);
        if (tokens !== null) entry.tokens = tokens;
        const error = str(update.error);
        if (error !== null && error !== '') entry.error = error;
        if (isNew) this.add(entry);
        else this.changed(entry);
        break;
      }
      case 'retry_state': {
        const attempt = num(update.attempt);
        if (update.type !== 'retrying' || attempt === null) break;
        this.retries.push({ ts, attempt, max: num(update.max_retries) ?? 0 });
        break;
      }
      case 'model_changed': {
        const modelId = str(update.model_id);
        if (modelId === null || modelId === '') break;
        this.model = modelId;
        this.add({ kind: 'model', ts, modelId });
        break;
      }
    }
  }
}

/** How a shell command ended, from the agent's `rawOutput` of `type: "Bash"`; null for any other shape. */
function bashEnd(raw: unknown): { exitCode: number | null; signal: string | null; timedOut: boolean } | null {
  if (!isRecord(raw) || raw.type !== 'Bash') return null;
  const code = raw.exit_code;
  return {
    exitCode: typeof code === 'number' && Number.isInteger(code) ? code : null,
    signal: str(raw.signal),
    timedOut: raw.timed_out === true,
  };
}

/**
 * A tool entry's `input` and `output`. `output` is the content text when there is one, else the text of a successful
 * MCP result, else the JSON text of `rawOutput`: an MCP result is kept as plain text, never as JSON cut mid-string.
 * A shell command's output is its content text only (none when it printed nothing); how it ended is kept apart.
 */
function finishTool(acc: ToolAcc): void {
  const input = fieldText(acc.rawInput);
  const mcp = mcpOutputText(acc.rawOutput);
  const bash = bashEnd(acc.rawOutput);
  const output =
    acc.content !== null && acc.content !== ''
      ? fieldText(acc.content)
      : bash !== null
        ? undefined
        : mcp !== null
          ? fieldText(mcp.text)
          : fieldText(acc.rawOutput);
  if (bash !== null && bash.exitCode !== null && bash.exitCode !== 0) acc.entry.exitCode = bash.exitCode;
  else delete acc.entry.exitCode;
  if (bash !== null && bash.signal !== null && bash.signal !== '') acc.entry.signal = bash.signal;
  else delete acc.entry.signal;
  if (bash?.timedOut === true) acc.entry.timedOut = true;
  else delete acc.entry.timedOut;
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
  } else if (bash !== null) delete acc.entry.output;
  if (truncated) acc.entry.truncated = true;
  else delete acc.entry.truncated;
}

export function parseUpdates(text: string): ParsedUpdates {
  const reducer = new TranscriptReducer();
  reducer.applyText(text);
  return reducer.parsed();
}
