/**
 * grok `updates.jsonl` → display transcript + per-turn metrics. Pure data → data.
 *
 * Line handling follows grok-build `session/storage`: a line with `params` is an
 * envelope (`method` `_x.ai/session/update` = xAI, anything else = ACP); a line
 * without `params` is a legacy bare ACP notification. Rewinds are applied to the
 * raw lines first, exactly like grok's `filter_rewind_lines`.
 */

import { readCompaction } from '../chat/acp/messages.js';
import { base64Bytes, nameOfUri, readPdfNote, utf8Bytes, type UserAttachment } from '../chat/attachment-rule.js';
import { CONTINUE_PROMPT, type TranscriptOp } from '../chat/types.js';
import { askOf } from './ask-result.js';
import { diffEdits, type RawDetail, type RawEdit } from './line-diff.js';
import { mcpOutputText } from './mcp-output.js';
import type { RetryCause, TranscriptEntry, TurnMetric, WakeCause } from './types.js';

const XAI_METHOD = '_x.ai/session/update';
const REWIND_MARKER = 'rewind_marker';
/** The agent's question tool: its entry carries `ask`. */
const ASK_TOOL = 'ask_user_question';
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
  /** The last `_meta.totalTokens` of the file: the conversation's current context in tokens; null when none. */
  contextUsed: number | null;
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

const ESC = 0x1b;

/** The agent's `FRONT_BACK_TRUNCATION_MARKER`: it joins the head and the tail of a command's output it cut. */
const CUT_MARKER = '\n\n... (output truncated) ...\n\n';

/**
 * Removes the terminal escape sequences a colored command prints, so its output does not read as binary: CSI
 * (`ESC [`, parameter bytes, one final byte) and OSC (`ESC ]` up to BEL or `ESC \`). A CSI cut before its final byte
 * (the agent's head-and-tail cut, a live tick's last read) loses only its `ESC [` and parameter bytes, and an ESC the
 * cut left alone (the last character, or right before the cut marker) goes too. Every other ESC stays, and so does an
 * OSC with no end.
 */
export function stripAnsi(text: string): string {
  if (!text.includes('\u001b')) return text;
  let out = '';
  let from = 0;
  let oscEnds = true;
  let i = 0;
  while (i < text.length) {
    if (text.charCodeAt(i) !== ESC) {
      i += 1;
      continue;
    }
    const kind = text.charCodeAt(i + 1);
    let end = -1;
    if (kind === 0x5b) {
      end = i + 2;
      while (end < text.length && isCsiParameter(text.charCodeAt(end))) end += 1;
      if (end < text.length && isCsiFinal(text.charCodeAt(end))) end += 1;
    } else if (kind === 0x5d && oscEnds) {
      end = oscEnd(text, i + 2);
      if (end === -1) oscEnds = false;
    } else if (i + 1 === text.length || text.startsWith(CUT_MARKER, i + 1)) {
      end = i + 1;
    }
    if (end === -1) {
      i += 1;
      continue;
    }
    out += text.slice(from, i);
    from = end;
    i = end;
  }
  return out + text.slice(from);
}

function isCsiParameter(code: number): boolean {
  return code >= 0x30 && code <= 0x3f;
}

function isCsiFinal(code: number): boolean {
  return code >= 0x40 && code <= 0x7e;
}

function oscEnd(text: string, from: number): number {
  for (let i = from; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code === 0x07) return i + 1;
    if (code === ESC && text.charCodeAt(i + 1) === 0x5c) return i + 2;
  }
  return -1;
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

/**
 * A prompt block of the user that is a file, a PDF note or a picture, as its view: the name from the resource uri (a
 * note's own name), the sizes of the stored text or base64. A message note's pages are image blocks `from` to `to`
 * (from 1) of its message, which the reader adds to the note's view. Null for any other block, and for a resource
 * with neither text nor blob.
 */
function userAttachment(block: unknown): { view: UserAttachment; pages?: { from: number; to: number } } | null {
  if (!isRecord(block)) return null;
  if (block.type === 'image') {
    return {
      view: { kind: 'image', mediaType: str(block.mimeType) ?? 'image', bytes: base64Bytes(str(block.data) ?? '') },
    };
  }
  if (block.type !== 'resource' || !isRecord(block.resource)) return null;
  const name = nameOfUri(str(block.resource.uri) ?? '');
  const text = str(block.resource.text);
  if (text !== null) {
    const note = readPdfNote(text);
    if (note === null) return { view: { kind: 'file', name, bytes: utf8Bytes(text) } };
    const pdf = { kind: 'pdf', name: note.name, pages: note.pages, way: note.way, bytes: 0, chars: 0 } as const;
    if (note.way === 'tool') return { view: { ...pdf, id: note.id } };
    return { view: pdf, pages: { from: note.from, to: note.from + note.pages - 1 } };
  }
  const blob = str(block.resource.blob);
  return blob === null ? null : { view: { kind: 'file', name, bytes: base64Bytes(blob) } };
}

/**
 * ACP `ToolCallContent[]` → display text. A `diff` block is no text: {@link readEdits} reads it; nor is a picture:
 * {@link readPictures} reads it.
 */
function toolContentText(content: unknown[]): string {
  const parts: string[] = [];
  for (const item of content) {
    if (!isRecord(item)) continue;
    if (item.type === 'content') {
      if (isRecord(item.content) && item.content.type === 'image') continue;
      const text = stripAnsi(contentBlockText(item.content));
      if (text !== '') parts.push(text);
    } else if (item.type === 'terminal') {
      parts.push('[terminal]');
    }
  }
  return parts.join('\n');
}

/** A 1-based line number, else null. */
function lineNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 1 ? value : null;
}

/** `_meta.details[]` of a result's diff block: the replaced matches that fit, or null for none. */
function readDetails(meta: Json | null): RawDetail[] | null {
  if (meta === null || !Array.isArray(meta.details)) return null;
  const details: RawDetail[] = [];
  for (const item of meta.details) {
    if (!isRecord(item)) continue;
    const oldString = str(item.old_string);
    const newString = str(item.new_string);
    const line = lineNumber(item.new_line);
    if (oldString === null || newString === null || line === null) continue;
    details.push({
      oldString,
      newString,
      line,
      before: str(item.context_before) ?? '',
      after: str(item.context_after) ?? '',
      prefix: str(item.line_prefix) ?? '',
    });
  }
  return details.length === 0 ? null : details;
}

/**
 * The `diff` blocks of a tool update, or null when it has none. `completed`: the update is the call's result, so an
 * empty old text means a new file.
 */
function readEdits(content: unknown[], completed: boolean): RawEdit[] | null {
  const edits: RawEdit[] = [];
  for (const item of content) {
    if (!isRecord(item) || item.type !== 'diff') continue;
    const meta = isRecord(item._meta) ? item._meta : null;
    const oldText = str(item.oldText) ?? '';
    const oldLine = lineNumber(meta?.old_line);
    const newLine = lineNumber(meta?.new_line);
    edits.push({
      path: str(item.path) ?? '',
      oldText,
      newText: str(item.newText) ?? '',
      details: readDetails(meta),
      line: oldLine === null || newLine === null ? null : [oldLine, newLine],
      create: completed && oldText === '',
    });
  }
  return edits.length === 0 ? null : edits;
}

/** A picture's media type as the reader keeps it: the agent's word, cut. */
const MEDIA_TYPE_CHARS = 100;

type ToolPicture = NonNullable<Extract<TranscriptEntry, { kind: 'tool' }>['pictures']>[number];

/** A PDF read's result (`ReadFile` `PdfPageImages`): each page's number by place, and the file's page count. */
function pdfPagesOf(raw: unknown): { numbers: Array<number | null>; total: number | null } | null {
  if (!isRecord(raw) || raw.type !== 'ReadFile' || !isRecord(raw.PdfPageImages)) return null;
  const pdf = raw.PdfPageImages;
  const pages = Array.isArray(pdf.pages) ? pdf.pages : [];
  return {
    numbers: pages.map((page) => (isRecord(page) ? lineNumber(page.page_number) : null)),
    total: lineNumber(pdf.total_pages),
  };
}

/**
 * The image blocks of a tool update as pictures (sizes, never bytes), with a PDF read's page numbers from `raw`; null
 * when the update has none.
 */
function readPictures(content: unknown[], raw: unknown): { pictures: ToolPicture[]; pdfPages: number | null } | null {
  const pdf = pdfPagesOf(raw);
  const pictures: ToolPicture[] = [];
  for (const item of content) {
    if (!isRecord(item) || item.type !== 'content' || !isRecord(item.content)) continue;
    const block = item.content;
    if (block.type !== 'image') continue;
    const n = pictures.length;
    const page = pdf?.numbers[n] ?? null;
    pictures.push({
      n,
      mediaType: truncateField(str(block.mimeType) ?? '', MEDIA_TYPE_CHARS).text,
      bytes: base64Bytes(str(block.data) ?? ''),
      ...(page === null ? {} : { page }),
    });
  }
  return pictures.length === 0 ? null : { pictures, pdfPages: pdf?.total ?? null };
}

/** A copy of `record` without the field `name`. */
function without(record: Json, name: string): Json {
  const out: Json = {};
  for (const [key, value] of Object.entries(record)) if (key !== name) out[key] = value;
  return out;
}

/** A `ReadFile` picture result without its bytes: `ImageContent` keeps its type, `PdfPageImages` its page facts. */
function slimReadFile(raw: Json): Json | null {
  if (isRecord(raw.ImageContent)) return { ...raw, ImageContent: without(raw.ImageContent, 'data') };
  if (!isRecord(raw.PdfPageImages)) return null;
  const pdf = raw.PdfPageImages;
  const pages = Array.isArray(pdf.pages)
    ? pdf.pages.map((page: unknown) => (isRecord(page) ? without(page, 'data') : page))
    : pdf.pages;
  return { ...raw, PdfPageImages: { ...pdf, pages } };
}

/**
 * A tool result without what holds whole files or pictures: an edit's `EditsApplied` (its diff blocks carry the
 * change), and every `data` of a picture read (its image blocks become `pictures`; the bytes stay on disk).
 */
export function slimRawOutput(raw: unknown): unknown {
  if (!isRecord(raw)) return raw;
  if (raw.type === 'ReadFile') return slimReadFile(raw) ?? raw;
  if (raw.type !== 'SearchReplace' || !('EditsApplied' in raw)) return raw;
  return without(raw, 'EditsApplied');
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
  /** The diff blocks of the newest update that had any, until `finishTool` makes `entry.edits` of them. */
  edits: RawEdit[] | null;
}

type ToolEntry = Timed<'tool'>;
type TaskKind = NonNullable<ToolEntry['task']>['kind'];
type ScheduleFacts = Omit<NonNullable<ToolEntry['schedule']>, 'id'>;

/** The task a tool result started: `BackgroundTaskStarted` is a command, `Monitor` a monitor; null for any other. */
function startedTask(raw: unknown): { id: string; kind: TaskKind } | null {
  if (!isRecord(raw)) return null;
  const id = raw.type === 'BackgroundTaskStarted' ? str(raw.task_id) : raw.type === 'Monitor' ? str(raw.taskId) : null;
  if (id === null || id === '') return null;
  return { id, kind: raw.type === 'Monitor' ? 'monitor' : 'command' };
}

/** The description, else the first line of the command that is not blank, else ''. */
function taskLabel(description: unknown, command: unknown): string {
  const said = str(description)?.trim() ?? '';
  if (said !== '') return said;
  for (const line of (str(command) ?? '').split('\n')) {
    const text = line.trim();
    if (text !== '') return text;
  }
  return '';
}

/** An RFC 3339 time of the agent in epoch ms, or null. */
export function rfc3339Ms(value: unknown): number | null {
  const text = str(value);
  const ms = text === null ? 0 : epochMs(Date.parse(text));
  return ms === 0 ? null : ms;
}

/** A Rust `SystemTime` as serde writes it (`secs_since_epoch`, `nanos_since_epoch`) in epoch ms, or null. */
function systemTimeMs(value: unknown): number | null {
  if (!isRecord(value) || !isUsize(value.secs_since_epoch)) return null;
  const nanos = isUsize(value.nanos_since_epoch) ? value.nanos_since_epoch : 0;
  const ms = epochMs(value.secs_since_epoch * 1000 + Math.floor(nanos / 1_000_000));
  return ms === 0 ? null : ms;
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
  /** `_meta.totalTokens` of the line: the agent's count of the conversation's current context. */
  contextUsed: number | null;
  /** `_meta.cancellationCategory` of the line (a turn end's): why the agent cancelled the turn. */
  cancelCategory?: string | null;
  /** `_meta.eventId`: `<session id>-<n>` of the session that wrote the line. */
  eventId?: string | null;
}

function contextOf(meta: unknown): number | null {
  return isRecord(meta) && isUsize(meta.totalTokens) ? meta.totalTokens : null;
}

function cancelCategoryOf(meta: unknown): string | null {
  return isRecord(meta) ? str(meta.cancellationCategory) : null;
}

/** A stored end on the model's length limit. */
function isLengthEnd(entry: TranscriptEntry | undefined): boolean {
  return entry?.kind === 'turn-end' && (entry.stopReason === 'max_tokens' || entry.lengthLimit === true);
}

type NoticeEntry = Extract<TranscriptEntry, { kind: 'notice' }>;
type RetryNotice = Extract<NoticeEntry, { code: 'retry' }>;
type PlanModeEntry = Extract<TranscriptEntry, { kind: 'plan-mode' }>;
type GoalEntry = Extract<TranscriptEntry, { kind: 'goal' }>;
/** An entry that is no step of the answer and is placed by the hold rules: a notice, a mode line or a goal. */
type SideEntry = NoticeEntry | PlanModeEntry | GoalEntry;

function isHeldKind(entry: TranscriptEntry): entry is SideEntry {
  return entry.kind === 'notice' || entry.kind === 'plan-mode' || entry.kind === 'goal';
}

/** A goal's tokens as the page shows them (in thousands): a change below that sends the page nothing. */
const GOAL_TOKENS_STEP = 1_000;

/** What the page shows of a goal: a line that leaves it equal changes the entry but sends no op. */
function goalShown(goal: GoalEntry): string {
  return JSON.stringify([
    goal.status,
    goal.phase,
    goal.objective,
    goal.tokenBudget ?? null,
    Math.floor(goal.tokensUsed / GOAL_TOKENS_STEP),
    goal.pauseMessage ?? null,
    goal.verifying === true,
    goal.planning === true,
  ]);
}

/**
 * The agent stores the prompt of `/goal <objective>`, and of a `/goal resume` that runs, as its goal rules (`turn.rs`,
 * `goal.rs` `setup_goal` / `resume_goal`, template `goal_rules.md`), not as typed.
 */
const GOAL_RULES_HEAD = '<system-reminder>\nA goal has been set: ';
const GOAL_RESUME_TAIL = '\nContinue working now.\n</system-reminder>';

/**
 * The typed text of a stored goal-rules prompt of the active goal `goal`: `/goal resume`, else `/goal <objective>` with
 * its budget as `parse_goal_budget` reads it; null for any other text.
 */
function goalPrompt(text: string, goal: GoalEntry | null): string | null {
  if (goal === null || goal.status !== 'active' || !text.startsWith(`${GOAL_RULES_HEAD}${goal.objective}\n`))
    return null;
  if (text.endsWith(GOAL_RESUME_TAIL)) return '/goal resume';
  const set = `/goal ${goal.objective}`;
  return goal.tokenBudget === undefined ? set : `${set} --budget ${goal.tokenBudget}`;
}

/** grok `SessionMode::from_id`: an id it does not know is its default mode. */
function sessionMode(id: string): PlanModeEntry['mode'] {
  return id === 'plan' || id === 'ask' ? id : 'default';
}

/** Characters of the agent's own words a notice keeps. */
export const NOTICE_TEXT_CHARS = 300;

/** The agent's words for a notice, cut at {@link NOTICE_TEXT_CHARS} with `…`; undefined for none. */
function noticeText(value: unknown): string | undefined {
  const text = str(value)?.trim() ?? '';
  if (text === '') return undefined;
  const cut = truncateField(text, NOTICE_TEXT_CHARS);
  return cut.truncated ? `${cut.text}…` : cut.text;
}

/** Characters of a recap the entry keeps. */
export const RECAP_TEXT_CHARS = 2_000;

/** Notes and sizes a picture notice keeps. */
const PICTURE_ITEMS_MAX = 6;

type PictureNotice = Extract<NoticeEntry, { code: 'picture' }>;

/** The agent's notes of an `image_dropped` line, each cut; blank and non-text notes are left out. */
function pictureNotes(value: unknown): string[] {
  const notes: string[] = [];
  if (!Array.isArray(value)) return notes;
  for (const item of value) {
    const note = noticeText(item);
    if (note !== undefined) notes.push(note);
  }
  return notes;
}

/** The sizes of an `image_compressed` line's `images`; an item with a count that is not a whole number is left out. */
function pictureSizes(images: readonly unknown[]): NonNullable<PictureNotice['sizes']> {
  const sizes: NonNullable<PictureNotice['sizes']> = [];
  for (const image of images) {
    if (!isRecord(image)) continue;
    const { original_bytes: fromBytes, compressed_bytes: toBytes } = image;
    const from = [image.original_width, image.original_height];
    const to = [image.compressed_width, image.compressed_height];
    if (!isUsize(fromBytes) || !isUsize(toBytes) || ![...from, ...to].every(isUsize)) continue;
    sizes.push({
      fromBytes,
      toBytes,
      from: [from[0] as number, from[1] as number],
      to: [to[0] as number, to[1] as number],
    });
  }
  return sizes;
}

/** What made the agent send a model request again: its `error_type`, else its reason's words. */
function retryCause(errorType: string | null, reason: string): RetryCause {
  switch (errorType) {
    case 'idle_timeout':
      return 'stalled';
    case 'http':
      return 'connection';
    case 'api':
    case 'serialization':
    case 'empty_response':
      return 'server';
    case 'rate_limited':
      return 'busy';
    case 'doom_loop_detected':
      return 'loop';
  }
  return reason.startsWith('Too many requests') ? 'busy' : 'other';
}

/** Characters of a hook's error, a hook's message or the agent's words on a hook's verdict that an entry keeps. */
export const HOOK_TEXT_CHARS = 600;

/** The start of the failure text the agent gives a call a hook denied (`hooks.rs` `block_tool_call`). */
export const HOOK_DENIED_TEXT = 'Hook denied: ';

/** The cancel category of a turn whose prompt a hook blocked (`_meta.cancellationCategory`). */
const HOOK_DENIED_CATEGORY = 'HookDenied';

type HookNotice = Extract<NoticeEntry, { code: 'hook' }>;

/** A hook text cut at {@link HOOK_TEXT_CHARS} with `…`, without the agent's leading ⚠ or ↩; undefined for none. */
function hookText(value: unknown): string | undefined {
  let text = str(value)?.trim() ?? '';
  const first = text.codePointAt(0);
  if (first === 0x26a0 || first === 0x21a9) {
    text = text.slice(1);
    if (text.codePointAt(0) === 0xfe0f) text = text.slice(1);
    text = text.trim();
  }
  if (text === '') return undefined;
  const cut = truncateField(text, HOOK_TEXT_CHARS);
  return cut.truncated ? `${cut.text}…` : cut.text;
}

function isDigits(text: string): boolean {
  if (text === '') return false;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code < 0x30 || code > 0x39) return false;
  }
  return true;
}

/** `{event}[{i}].hooks[{j}]`, the spec path the agent stamps on a hook's name (`config.rs` `is_spec_path`). */
function isSpecPath(tail: string): boolean {
  const open = tail.indexOf('[');
  if (open <= 0) return false;
  for (let i = 0; i < open; i++) {
    const code = tail.charCodeAt(i);
    if (!(code === 0x5f || (code >= 0x61 && code <= 0x7a))) return false;
  }
  const indices = tail.slice(open + 1);
  const mid = indices.indexOf('].hooks[');
  if (mid === -1) return false;
  const last = indices.slice(mid + '].hooks['.length);
  return isDigits(indices.slice(0, mid)) && last.endsWith(']') && isDigits(last.slice(0, -1));
}

/** A hook's name without its `:{event}[i].hooks[j]` tail (`config.rs` `strip_spec_path`); any other name as sent. */
export function hookSource(name: string): string {
  const colon = name.lastIndexOf(':');
  if (colon <= 0) return name;
  return isSpecPath(name.slice(colon + 1)) ? name.slice(0, colon) : name;
}

/** One failure's key: a later failure with the same key counts on the first one's row. */
function hookKey(entry: Pick<HookNotice, 'hook' | 'event' | 'tool' | 'detail'>): string {
  return `${entry.hook}\u0000${entry.event}\u0000${entry.tool ?? ''}\u0000${entry.detail}`;
}

/** A typed prompt or a wake: the entry a turn starts with. */
function isPromptEntry(entry: TranscriptEntry | undefined): boolean {
  return (entry?.kind === 'user' && entry.note !== true && entry.interjection !== true) || entry?.kind === 'wake';
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
  return {
    xai,
    update: notification.update,
    ts: epochMs(agentMs ?? (seconds !== null ? seconds * 1000 : 0)),
    contextUsed: contextOf(meta),
    cancelCategory: cancelCategoryOf(meta),
    eventId: isRecord(meta) ? str(meta.eventId) : null,
  };
}

/** A time outside `Date`'s range (or one that overflowed to Infinity) is 0: untimed, like a missing one. */
function epochMs(ms: number): number {
  return Number.isFinite(ms) && ms > 0 && ms <= MAX_DATE_MS ? ms : 0;
}

/** First match wins: `monitor-completed-` must stay before `monitor-`. */
const WAKE_PREFIXES: ReadonlyArray<readonly [string, WakeCause]> = [
  ['workflow-completed-', 'workflow'],
  ['subagent-completed-', 'subagent'],
  ['task-completed-', 'task'],
  ['bash-completed-', 'task'],
  ['monitor-completed-', 'task'],
  ['monitor-', 'notifications'],
  ['notifications-', 'notifications'],
  ['goal-summary-', 'goal'],
  ['goal-classifier-nudge-', 'goal'],
  ['interject-fallback-', 'message'],
];

/**
 * The `_meta` key of the app host's own wake line (the agent sends a wake's prompt to disk only): the prompt id the
 * host expects for that wake, so the row has its cause before the wake's `turn_completed`.
 */
export const WAKE_PROMPT_ID = 'mlxWakePromptId';

/** The cause a prompt id names, and what follows its prefix (a run, helper or task id). */
function wakeSource(promptId: string | null): { cause: WakeCause; rest: string } {
  if (promptId !== null) {
    for (const [prefix, cause] of WAKE_PREFIXES) {
      if (promptId.startsWith(prefix)) return { cause, rest: promptId.slice(prefix.length) };
    }
  }
  return { cause: 'unknown', rest: '' };
}

/** grok `PromptOrigin::from_prompt_id`: the prompt id prefix names what woke the agent. */
export function wakeCause(promptId: string | null): WakeCause {
  return wakeSource(promptId).cause;
}

type WakeEntry = Extract<TranscriptEntry, { kind: 'wake' }>;
type WakeTask = NonNullable<WakeEntry['task']>;
/** The wake's view of the session's commands and monitors. */
interface WakeTasks {
  /** A task as the row that started it names it, else as its end line does; null when neither was seen. */
  ref(id: string): WakeTask | null;
  /** Tasks that ended with no wake of their own since the last turn end, oldest first, at most 8. */
  quiet(): readonly string[];
}
/** The most ended tasks a `notifications` wake names. */
const WAKE_TASKS_MAX = 8;
type SubagentEntry = Extract<TranscriptEntry, { kind: 'subagent' }>;
type WorkflowEntry = Extract<TranscriptEntry, { kind: 'workflow' }>;
type RunSnapshot = NonNullable<NonNullable<WakeEntry['workflow']>['snapshot']>;
/** What a wake reads of the runs: a run's name, and the update of a revision that could wake the agent. */
interface WakeRuns {
  name(runId: string): string | undefined;
  snapshot(runId: string, revision: number): RunSnapshot | undefined;
}

/** `result_summary` cap: above the agent's own 16 KiB cut, so the agent's cut is the only one. */
export const MAX_REPORT_CHARS = 20 * 1024;
/** grok `is_completion_reportable`: the statuses whose update the agent names in a wake's prompt id. */
const WAKE_STATUSES: ReadonlySet<string> = new Set([
  'interrupted',
  'complete',
  'failed',
  'cancelled',
  'budget_limited',
]);
/** Kept updates of one run that could wake the agent; a stop's drain sends one per agent. */
const RUN_SNAPSHOTS_MAX = 32;

/** `workflow-completed-<run>-<revision>` without its prefix → the run and the revision; null without a revision. */
function wakeRun(rest: string): { runId: string; revision: number } | null {
  const dash = rest.lastIndexOf('-');
  if (dash <= 0) return null;
  const digits = rest.slice(dash + 1);
  if (!isDigits(digits)) return null;
  const revision = Number(digits);
  return Number.isSafeInteger(revision) ? { runId: rest.slice(0, dash), revision } : null;
}

/** The facts of one `workflow_updated` a wake row shows. */
function runSnapshot(update: Json): RunSnapshot {
  const snapshot: RunSnapshot = {
    status: str(update.status) ?? '',
    agentsUsed: isUsize(update.agents_used) ? update.agents_used : 0,
    agentBudget: isUsize(update.agent_budget) ? update.agent_budget : null,
    agentsFailed: Array.isArray(update.agents)
      ? update.agents.filter((a) => isRecord(a) && a.state === 'failed').length
      : 0,
    elapsedMs: isUsize(update.elapsed_ms) ? update.elapsed_ms : 0,
  };
  const summary = str(update.result_summary);
  if (summary !== null && summary !== '') {
    const cut = truncateField(summary, MAX_REPORT_CHARS);
    snapshot.report = cut.text;
    if (cut.truncated) snapshot.truncated = true;
  }
  const pause = str(update.pause_message);
  if (pause !== null && pause !== '') snapshot.pauseMessage = pause;
  return snapshot;
}

/**
 * Sets a run's entry from one `workflow_updated` (or `state.json` read as one) stamped `ts`. Steps keep the `detail`
 * of the same title, and a `phase_entered` it names joins `phasesEntered`; arrays are replaced, never changed in place.
 */
export function applyRunUpdate(entry: WorkflowEntry, update: Json, ts: number): void {
  entry.updatedTs = ts;
  if (isUsize(update.revision)) entry.revision = update.revision;
  entry.name = str(update.name) ?? '';
  entry.status = str(update.status) ?? '';
  const objective = str(update.objective);
  if (objective !== null && objective !== '') entry.objective = objective;
  else delete entry.objective;
  entry.elapsedMs = isUsize(update.elapsed_ms) ? update.elapsed_ms : 0;
  if (update.agent_usage_incomplete === true) entry.usageIncomplete = true;
  else delete entry.usageIncomplete;
  const details = new Map<string, string>();
  for (const phase of entry.phases) if (phase.detail !== undefined) details.set(phase.title, phase.detail);
  entry.phases = Array.isArray(update.phases)
    ? update.phases.filter(isRecord).map((p) => {
        const title = str(p.title) ?? '';
        const detail = details.get(title);
        return { title, state: str(p.state) ?? '', ...(detail === undefined ? {} : { detail }) };
      })
    : [];
  entry.currentPhase = str(update.current_phase);
  entry.agentBudget = isUsize(update.agent_budget) ? update.agent_budget : null;
  entry.agentsUsed = isUsize(update.agents_used) ? update.agents_used : 0;
  entry.agents = Array.isArray(update.agents)
    ? update.agents.filter(isRecord).map((a) => {
        const model = str(a.model);
        return {
          agentId: str(a.agent_id) ?? '',
          label: str(a.label) ?? '',
          phase: str(a.phase),
          ...(model === null || model === '' ? {} : { model }),
          state: str(a.state) ?? '',
          tokens: isUsize(a.tokens_used) ? a.tokens_used : 0,
          durationMs: isUsize(a.duration_ms) ? a.duration_ms : 0,
        };
      })
    : [];
  const entered = update.last_event === 'phase_entered' ? str(update.last_event_detail) : null;
  if (entered !== null && entered !== '') {
    const at = rfc3339Ms(update.last_event_timestamp) ?? ts;
    if (!entry.phasesEntered.some((p) => p.title === entered && p.ts === at))
      entry.phasesEntered = [...entry.phasesEntered, { title: entered, ts: at }];
  }
  const pause = str(update.pause_message);
  if (pause !== null && pause !== '') entry.pauseMessage = pause;
  else delete entry.pauseMessage;
  const summary = str(update.result_summary);
  delete entry.report;
  delete entry.truncated;
  if (summary !== null && summary !== '') {
    const cut = truncateField(summary, MAX_REPORT_CHARS);
    entry.report = cut.text;
    if (cut.truncated) entry.truncated = true;
  }
}

/** Sets `cause` and the matching `workflow` / `subagent` / `task` / `tasks` field of a wake entry from a prompt id. */
function applyWake(
  entry: WakeEntry,
  promptId: string | null,
  runs: WakeRuns,
  spawnedBy: ReadonlyMap<string, SubagentEntry>,
  tasks: WakeTasks,
): void {
  const { cause, rest } = wakeSource(promptId);
  entry.cause = cause;
  delete entry.workflow;
  delete entry.subagent;
  delete entry.task;
  delete entry.tasks;
  if (cause === 'workflow') {
    const run = wakeRun(rest);
    const name = run === null ? undefined : runs.name(run.runId);
    if (run !== null && name !== undefined && name !== '') {
      entry.workflow = { ...run, name, snapshot: runs.snapshot(run.runId, run.revision) ?? null };
    }
  } else if (cause === 'subagent') {
    const child = spawnedBy.get(rest);
    if (child !== undefined) entry.subagent = { childId: child.childId, description: child.description };
  } else if (cause === 'task') {
    const task = tasks.ref(rest);
    if (task !== null) entry.task = task;
  } else if (cause === 'notifications') {
    const ended = tasks.quiet().flatMap((id) => tasks.ref(id) ?? []);
    if (ended.length > 0) entry.tasks = ended;
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

/** The agent's tool that launches, checks, pauses, stops and resumes workflow runs. */
const WORKFLOW_TOOL = 'workflow';
type RunCall = NonNullable<ToolEntry['run']>;
type RunControl = NonNullable<Extract<TranscriptEntry, { kind: 'user' }>['runControl']>;

/** A `workflow` call's kind from its `rawInput`: the `source.type` the agent parsed; null for any other source. */
function runCallKind(input: Json): RunCall['kind'] | null {
  const source = isRecord(input.source) ? input.source : null;
  switch (source?.type) {
    case 'name':
    case 'script':
    case 'script_path':
      return input.validate_only === true ? 'check' : 'launch';
    case 'pause':
    case 'stop':
    case 'resume':
      return source.type;
    default:
      return null;
  }
}

/** The ops of grok's `/workflow` command that take a run (`save` takes one too, but changes nothing of it). */
const RUN_OPS: ReadonlySet<string> = new Set(['pause', 'resume', 'stop', 'save']);

function trimAscii(text: string): string {
  let a = 0;
  let b = text.length;
  while (a < b && isAsciiSpace(text.charCodeAt(a))) a++;
  while (b > a && isAsciiSpace(text.charCodeAt(b - 1))) b--;
  return text.slice(a, b);
}

function asciiWords(text: string): string[] {
  const out: string[] = [];
  let start = -1;
  for (let i = 0; i <= text.length; i++) {
    const space = i === text.length || isAsciiSpace(text.charCodeAt(i));
    if (!space && start === -1) start = i;
    else if (space && start !== -1) {
      out.push(text.slice(start, i));
      start = -1;
    }
  }
  return out;
}

/**
 * grok's `/workflow` resolver (`slash_commands.rs`): `/workflow <op> <name>`, or `/workflow <name> <op>` with nothing
 * after the op. The verb and the name of a pause, resume or stop; null for anything else.
 */
export function workflowCommand(text: string): { verb: RunControl['verb']; name: string } | null {
  const typed = trimAscii(text);
  if (!typed.startsWith('/workflow')) return null;
  const tail = typed.slice('/workflow'.length);
  if (tail !== '' && !isAsciiSpace(tail.charCodeAt(0))) return null;
  const args = trimAscii(tail);
  const parts = asciiWords(args);
  const first = parts[0] ?? '';
  const second = parts[1] ?? '';
  let op: string;
  let name: string;
  if (RUN_OPS.has(first.toLowerCase())) {
    op = first.toLowerCase();
    name = trimAscii(args.slice(first.length));
  } else if (parts.length === 2 && RUN_OPS.has(second.toLowerCase())) {
    op = second.toLowerCase();
    name = first;
  } else {
    return null;
  }
  if ((op !== 'pause' && op !== 'resume' && op !== 'stop') || name === '') return null;
  return { verb: op, name };
}

/** grok `manage_workflow_run`: the runs whose id or name starts with `key`, the exact ones when any is. */
export function runsNamed(key: string, runs: ReadonlyMap<string, string>): string[] {
  const prefix: string[] = [];
  const exact: string[] = [];
  for (const [id, name] of runs) {
    if (id === key || name === key) exact.push(id);
    else if (id.startsWith(key) || name.startsWith(key)) prefix.push(id);
  }
  return exact.length > 0 ? exact : prefix;
}

/**
 * The one run of {@link runsNamed}, or null for none or several. The agent then narrows several by what the verb
 * applies to, by statuses it holds before its own update; this transcript holds them only after it, so several stay
 * none.
 */
function pickRun(key: string, runs: ReadonlyMap<string, string>): string | null {
  const pool = runsNamed(key, runs);
  return pool.length === 1 ? pool[0]! : null;
}

/** grok `find_run_id`, which the `workflow` tool's pause, stop and resume use: the first run of that id or name. */
export function findRun(key: string, runs: ReadonlyMap<string, string>): string | null {
  for (const [id, name] of runs) if (id === key || name === key) return id;
  return null;
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
  /** A helper's own session (it has a parent link): its parent's messages are bare user chunks, never notes. */
  helper?: boolean;
  /** The session's own id: a line whose `_meta.eventId` does not start with it and `-` makes `copied` entries. */
  sessionId?: string;
}

/** The host's own line for a note: the agent stores a note but sends no live copy. */
const NOTE_META = 'mlxNote';

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
  /** Every run's name by run id, in the order first seen. */
  private readonly runNames = new Map<string, string>();
  /** Per run, the updates of each revision a wake can name. */
  private readonly runSnapshots = new Map<string, Map<number, RunSnapshot>>();
  private readonly wakeRuns: WakeRuns = {
    name: (runId) => this.runNames.get(runId),
    snapshot: (runId, revision) => this.runSnapshots.get(runId)?.get(revision),
  };
  private readonly runEntries = new Map<string, WorkflowEntry>();
  /** Each goal's one entry by goal id, and the goal made last (a `cleared` line names none). */
  private readonly goalEntries = new Map<string, GoalEntry>();
  private lastGoal: GoalEntry | null = null;
  /** By `subagent_id`, else `child_session_id`. */
  private readonly spawnedBy = new Map<string, SubagentEntry>();
  /**
   * The tool entry that started each background task or made each schedule, by its id. Their `task` and `schedule`
   * objects are replaced, never changed in place: ops and seeded entries share them.
   */
  private readonly taskEntries = new Map<string, ToolEntry>();
  private readonly scheduleEntries = new Map<string, ToolEntry>();
  /** What the agent said of a task or schedule before the tool result that names it (a short command can end first). */
  private readonly earlyTaskEnds = new Map<string, { ts: number; killed: boolean }>();
  private readonly earlySchedules = new Map<string, ScheduleFacts>();
  /** Each task as its `task_completed` line names it, by task id. */
  private readonly taskEnds = new Map<string, WakeTask>();
  /** Tasks that ended with no wake of their own since the last turn end: what a `notifications` wake reports. */
  private quietEnds: string[] = [];
  private readonly wakeTasks: WakeTasks = {
    ref: (id) => {
      const task = this.taskEntries.get(id)?.task;
      if (task !== undefined) return { id, kind: task.kind, label: task.label };
      const end = this.taskEnds.get(id);
      return end === undefined ? null : { ...end };
    },
    quiet: () => this.quietEnds,
  };
  /** `prompt_id` of the last `user_prompt_submit` hook line; null after a prompt chunk or a turn end. */
  private submitId: string | null = null;
  /** The first prompt entry since the last turn end. */
  private turnPrompt: TranscriptEntry | null = null;
  private readonly wakeTexts = new Map<WakeEntry, string>();
  private wakeRun: { entry: WakeEntry; promptIndex: number } | null = null;
  private model: string | null = null;
  private contextUsed: number | null = null;
  private openTurn = false;
  private readonly helper: boolean;
  /** A user chunk with a `promptIndex` was seen: this agent build marks typed prompts, so a bare one can be a note. */
  private promptSeen = false;
  /** The step is the host's own typed prompt: never a note. */
  private typedStep = false;
  /** Disk notes of the open turn by text: the agent stores each note twice, so every second equal one is dropped. */
  private readonly noteCounts = new Map<string, number>();
  /**
   * Per user entry: the image blocks it has read, and where each PDF note's pages are among them, with the note's
   * place in the entry's attachments.
   */
  private readonly pdfPages = new WeakMap<
    TranscriptEntry,
    { images: number; notes: Array<{ from: number; to: number; at: number }> }
  >();
  /**
   * Notices and mode lines made while no turn is open, and on disk while the last entry is the turn's prompt (its
   * chunks still merge): added right after that prompt, so neither starts a turn. The agent sends notices past lines
   * made before them, and stores a prompt's mode line before the prompt.
   */
  private readonly held: SideEntry[] = [];
  /** The agent's session mode as the last mode line said; a session starts in `default`. */
  private planMode: PlanModeEntry['mode'] = 'default';
  /** Mode lines that moved the mode since this reducer started. */
  private modeMoves = 0;
  /** The retry episode of the open turn that has no end yet. */
  private retryOpen: RetryNotice | null = null;
  /** The open turn's last retry episode, whether a tool call came since it ended, and the order of its last try. */
  private retryLast: {
    entry: RetryNotice;
    toolSince: boolean;
    tryOrder: EventOrder | null;
    tryTs: number;
  } | null = null;
  /** The row of each hook failure of the session, by {@link hookKey}: a repeat counts on it. */
  private readonly hookFailures = new Map<string, HookNotice>();
  /**
   * The open turn's calls that failed as a hook denied them, and the agent's words on such a verdict
   * (`tool_outcome`), each still unpaired. The agent sends its words past the call's lines, so they pair in order.
   */
  private hookDenied: ToolEntry[] = [];
  private hookVerdicts: Array<{ text: string; ts: number }> = [];
  private readonly hookDeniedSeen = new WeakSet<ToolEntry>();
  /**
   * One-shot rows not placed yet: failed hooks, hook notes, retry episodes, pictures. The agent's store writes an xAI
   * line before the last ACP line it still holds (and merges later text into that line), so a row goes before the
   * first call stamped after it, else at the turn's end: the same place live and from disk. Mode lines are ACP lines,
   * which the store keeps in order, so they are placed when they come.
   */
  private oneShots: NoticeEntry[] = [];
  private turnsWithUsage = 0;
  private lastTs = 0;
  private readonly retries: ParsedUpdates['retries'] = [];
  private readonly toolNames = new Map<string, string>();
  private readonly now: () => number;
  private ops: TranscriptOp[] | null = null;
  /** `<own session id>-`; null when the caller named no session. */
  private readonly ownEvents: string | null;
  /** The line in hand was written by another session. */
  private lineCopied = false;

  constructor(options: ReducerOptions = {}) {
    this.now = options.now ?? Date.now;
    this.helper = options.helper === true;
    this.ownEvents = options.sessionId === undefined ? null : `${options.sessionId}-`;
  }

  /** Start from entries read earlier (a cold conversation's disk transcript); live ops index after them. */
  seed(entries: readonly TranscriptEntry[], turns: readonly TurnMetric[], contextUsed: number | null = null): void {
    for (const entry of entries) {
      const copy = { ...entry };
      this.add(copy);
      if (copy.kind === 'notice' && copy.code === 'hook' && !this.hookFailures.has(hookKey(copy)))
        this.hookFailures.set(hookKey(copy), copy);
      if (copy.kind === 'tool') {
        if (copy.task !== undefined && !this.taskEntries.has(copy.task.id)) this.taskEntries.set(copy.task.id, copy);
        const schedule = copy.schedule;
        if (schedule !== undefined && !this.scheduleEntries.has(schedule.id))
          this.scheduleEntries.set(schedule.id, copy);
      }
      if (copy.kind === 'goal') {
        this.goalEntries.set(copy.goalId, copy);
        this.lastGoal = copy;
      }
      if (copy.kind !== 'workflow') continue;
      const known = this.runEntries.get(copy.runId);
      if (known === undefined || copy.revision > known.revision) this.runEntries.set(copy.runId, copy);
      this.runNames.set(copy.runId, copy.name);
    }
    for (const turn of turns) this.turns.push({ ...turn });
    this.contextUsed = contextUsed;
    for (let i = entries.length - 1; i >= 0; i--) {
      const entry = entries[i]!;
      if (entry.kind === 'model') {
        this.model = entry.modelId;
        break;
      }
    }
    const mode = entries.findLast((entry) => entry.kind === 'plan-mode');
    if (mode?.kind === 'plan-mode') this.planMode = mode.mode;
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
    return this.collect(() =>
      this.step({
        xai: line.xai,
        update: line.update,
        ts: epochMs(agentMs ?? this.now()),
        contextUsed: contextOf(line.meta),
        cancelCategory: cancelCategoryOf(line.meta),
        eventId: line.meta !== undefined ? str(line.meta.eventId) : null,
      }),
    );
  }

  /**
   * The user's prompt: the agent does not echo it live, so the host pushes it when the prompt is accepted, with the
   * views of its attachments as the reader makes them from the stored blocks.
   */
  userPrompt(text: string, attachments: UserAttachment[] = [], ts: number = this.now()): TranscriptOp[] {
    return this.collect(() => {
      const at = this.transcript.length;
      this.typedStep = true;
      try {
        this.step({
          xai: false,
          update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text } },
          ts,
          contextUsed: null,
        });
      } finally {
        this.typedStep = false;
      }
      const entry = this.transcript[at];
      if (attachments.length === 0 || entry?.kind !== 'user') return;
      entry.attachments = [...attachments];
      this.changed(entry);
    });
  }

  /**
   * Adds a block's view to a user entry: an image block inside a PDF note's pages adds its size to that note's view
   * instead of being a picture of its own.
   */
  private attach(
    entry: Extract<TranscriptEntry, { kind: 'user' }>,
    read: NonNullable<ReturnType<typeof userAttachment>>,
  ): void {
    let slots = this.pdfPages.get(entry);
    if (slots === undefined) {
      slots = { images: 0, notes: [] };
      this.pdfPages.set(entry, slots);
    }
    const list = [...(entry.attachments ?? [])];
    if (read.view.kind === 'image') {
      slots.images += 1;
      const seen = slots.images;
      const note = slots.notes.find((n) => seen >= n.from && seen <= n.to);
      const pdf = note === undefined ? undefined : list[note.at];
      if (note !== undefined && pdf?.kind === 'pdf') {
        const bytes = read.view.bytes;
        list[note.at] = { ...pdf, bytes: pdf.bytes + bytes, chars: pdf.chars + Math.ceil(bytes / 3) * 4 };
        entry.attachments = list;
        return;
      }
    }
    if (read.pages !== undefined) slots.notes.push({ ...read.pages, at: list.length });
    list.push(read.view);
    entry.attachments = list;
  }

  /** The user's note on a call denied with one, pushed by the host when it sends the deny: a `user` entry with `note`. */
  note(text: string, ts: number = this.now()): TranscriptOp[] {
    return this.collect(() =>
      this.step({
        xai: false,
        update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text }, _meta: { [NOTE_META]: true } },
        ts,
        contextUsed: null,
      }),
    );
  }

  /**
   * The user's Send now, pushed by the host when the agent took it: the agent stores it only once the model read it
   * (`_meta.interjection`, the typed text in `displayText`) and sends no live copy. A `user` entry with `interjection`.
   */
  steer(text: string, ts: number = this.now()): TranscriptOp[] {
    return this.collect(() =>
      this.step({
        xai: false,
        update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text }, _meta: { interjection: true } },
        ts,
        contextUsed: null,
      }),
    );
  }

  /** A model switch the user made between turns. */
  modelSwitch(modelId: string, ts: number = this.now()): TranscriptOp[] {
    return this.collect(() =>
      this.step({ xai: true, update: { sessionUpdate: 'model_changed', model_id: modelId }, ts, contextUsed: null }),
    );
  }

  /**
   * The run's entry replaced in place by `merge` of it (its `state.json`, read live): a `set` op when anything changed,
   * none for a run with no entry.
   */
  mergeRun(runId: string, merge: (entry: WorkflowEntry) => WorkflowEntry): TranscriptOp[] {
    return this.collect(() => {
      const entry = this.runEntries.get(runId);
      if (entry === undefined) return;
      const next = merge(entry);
      if (JSON.stringify(next) === JSON.stringify(entry)) return;
      const fields = entry as Record<string, unknown>;
      for (const key of Object.keys(fields)) if (!(key in next)) delete fields[key];
      Object.assign(entry, next);
      this.runNames.set(runId, entry.name);
      this.changed(entry);
    });
  }

  /** The last `_meta.totalTokens` seen; null while no line carried one. */
  context(): number | null {
    return this.contextUsed;
  }

  /** The agent's session mode as the last mode line said, and how many mode lines moved it since the start. */
  modeState(): { mode: PlanModeEntry['mode']; moves: number } {
    return { mode: this.planMode, moves: this.modeMoves };
  }

  /** A copy of the open turn's last retry episode (open or ended), placed as a row or not; null when it has none. */
  retryLive(): RetryNotice | null {
    return this.retryLast === null ? null : { ...this.retryLast.entry };
  }

  snapshot(): { entries: TranscriptEntry[]; turns: TurnMetric[] } {
    for (const acc of this.tools.values()) finishTool(acc);
    if (this.openTurn) {
      this.placeOneShots(Number.POSITIVE_INFINITY);
      this.flushHeld();
    }
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
      contextUsed: this.contextUsed,
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

  /** `prompt`: the entry starts a turn, so held notices go after it. */
  private add(entry: TranscriptEntry, prompt = false): void {
    if (!prompt && !isHeldKind(entry) && this.openTurn) this.flushHeld();
    if (this.lineCopied) entry.copied = true;
    this.indexes.set(entry, this.transcript.length);
    this.transcript.push(entry);
    this.ops?.push({ op: 'push', entry: { ...entry } });
  }

  private flushHeld(): void {
    for (const entry of this.held.splice(0)) this.add(entry);
  }

  /** A notice or a mode line: held by the rules of {@link held}, else added. */
  private side(entry: SideEntry): void {
    if (!this.openTurn || this.held.length > 0 || (this.ops === null && isPromptEntry(this.last()))) {
      this.held.push(entry);
      return;
    }
    this.add(entry);
  }

  /** A new prompt entry while a turn is open: what was held belongs to the turn before it, and so do its retries. */
  private promptStarts(wasOpen: boolean): void {
    if (!wasOpen) return;
    this.hookTurnEnds();
    this.flushHeld();
    this.retryOpen = null;
    this.retryLast = null;
  }

  /** One failed run of a hook: the first of its key is a row, a repeat counts on that row. */
  private hookFailed(run: Omit<HookNotice, 'kind' | 'ts' | 'code' | 'times' | 'lastTs'>, ts: number): void {
    const key = hookKey(run);
    const known = this.hookFailures.get(key);
    if (known !== undefined) {
      known.times += 1;
      if (ts > known.lastTs) known.lastTs = ts;
      this.changed(known);
      return;
    }
    const entry: HookNotice = { kind: 'notice', ts, code: 'hook', ...run, times: 1, lastTs: ts };
    this.hookFailures.set(key, entry);
    this.oneShots.push(entry);
  }

  /** Places the held one-shot rows stamped before `before`, in order. */
  private placeOneShots(before: number): void {
    let n = 0;
    while (n < this.oneShots.length && this.oneShots[n]!.ts < before) n += 1;
    for (const entry of this.oneShots.splice(0, n)) this.side(entry);
  }

  /**
   * Pairs the oldest denied call with the oldest verdict while both wait: the call gets the verdict's words. Returns
   * the calls that changed.
   */
  private hookPairs(): ToolEntry[] {
    const paired: ToolEntry[] = [];
    while (this.hookDenied.length > 0 && this.hookVerdicts.length > 0) {
      const entry = this.hookDenied.shift()!;
      entry.blocked = this.hookVerdicts.shift()!.text;
      paired.push(entry);
    }
    return paired;
  }

  /**
   * The turn ended: a verdict no denied call took is a note; a denied call no verdict came for stays a failed call.
   * Every held hook row is placed.
   */
  private hookTurnEnds(): void {
    for (const verdict of this.hookVerdicts)
      this.oneShots.push({ kind: 'notice', ts: verdict.ts, code: 'hook-note', text: verdict.text });
    this.hookVerdicts = [];
    this.hookDenied = [];
    this.placeOneShots(Number.POSITIVE_INFINITY);
  }

  /** A `retrying` try (or an auto-recovery start): the open episode takes it, else a recovered one reopens, else a new one. */
  private retrying(
    attempt: number,
    max: number,
    cause: RetryCause,
    detail: string | undefined,
    ts: number,
    eventId: string | null | undefined,
  ): void {
    const last = this.retryLast;
    let entry = this.retryOpen;
    if (
      entry === null &&
      last !== null &&
      !last.toolSince &&
      last.entry.outcome === 'recovered' &&
      last.entry.attempts === attempt - 1
    ) {
      entry = last.entry;
      delete entry.endTs;
      delete entry.outcome;
    }
    if (entry === null) {
      entry = { kind: 'notice', ts, code: 'retry', cause, attempts: attempt, max };
      if (detail !== undefined) entry.detail = detail;
      this.retryOpen = entry;
      this.retryLast = { entry, toolSince: false, tryOrder: eventOrder(eventId), tryTs: ts };
      this.oneShots.push(entry);
      return;
    }
    if (last !== null) {
      last.tryOrder = eventOrder(eventId);
      last.tryTs = ts;
    }
    entry.cause = cause;
    entry.attempts = attempt;
    entry.max = max;
    if (detail !== undefined) entry.detail = detail;
    else delete entry.detail;
    this.retryOpen = entry;
    this.changed(entry);
  }

  /**
   * Whether a line was made after the last try. The agent writes a try at once, ahead of output it made before the try
   * and still holds, so the event counters order the two; their times do when either has none.
   */
  private afterTry(eventId: string | null | undefined, ts: number): boolean {
    const last = this.retryLast;
    if (last === null) return true;
    const order = eventOrder(eventId);
    if (order !== null && last.tryOrder !== null && order.session === last.tryOrder.session)
      return order.n > last.tryOrder.n;
    return ts > last.tryTs;
  }

  /** Ends the open retry episode at `ts`: output came back, the agent gave up, or (no outcome) the turn ended. */
  private retryEnds(ts: number, outcome: RetryNotice['outcome'], detail?: string): void {
    const entry = this.retryOpen;
    if (entry === null) return;
    this.retryOpen = null;
    entry.endTs = ts;
    if (outcome !== undefined) entry.outcome = outcome;
    if (detail !== undefined) entry.detail = detail;
    this.changed(entry);
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

  /**
   * A goal's one entry: made at the first line of its id and placed like a notice (the agent stores the line before
   * the prompt that set the goal), then set again by each line of that id; an op only when a shown field moved. A
   * `cleared` line (no id) ends the goal made last, unless it was done.
   */
  private goalLine(update: Json, ts: number): void {
    const id = str(update.goal_id) ?? '';
    const status = str(update.status) ?? '';
    if (id === '') {
      const goal = this.lastGoal;
      if (status !== 'cleared' || goal === null || goal.status === 'complete' || goal.status === 'cleared') return;
      goal.status = 'cleared';
      goal.phase = str(update.phase) ?? 'idle';
      goal.asOf = ts;
      delete goal.pauseMessage;
      delete goal.verifying;
      delete goal.planning;
      this.changed(goal);
      return;
    }
    if (status === '') return;
    const known = this.goalEntries.get(id);
    const before = known === undefined ? null : goalShown(known);
    const goal: GoalEntry = known ?? {
      kind: 'goal',
      ts,
      goalId: id,
      objective: '',
      status,
      phase: '',
      tokensUsed: 0,
      elapsedMs: 0,
      asOf: ts,
    };
    goal.objective = str(update.objective) ?? goal.objective;
    goal.status = status;
    goal.phase = str(update.phase) ?? '';
    goal.tokensUsed = isUsize(update.tokens_used) ? update.tokens_used : 0;
    goal.elapsedMs = isUsize(update.elapsed_ms) ? update.elapsed_ms : 0;
    goal.asOf = ts;
    const budget = update.token_budget;
    if (isUsize(budget) && budget > 0) goal.tokenBudget = budget;
    else delete goal.tokenBudget;
    const pause = noticeText(update.pause_message);
    if (pause !== undefined) goal.pauseMessage = pause;
    else delete goal.pauseMessage;
    if (update.verifying_completion === true) goal.verifying = true;
    else delete goal.verifying;
    if (update.planning === true) goal.planning = true;
    else delete goal.planning;
    if (known === undefined) {
      this.goalEntries.set(id, goal);
      this.lastGoal = goal;
      this.side(goal);
    } else if (before !== goalShown(goal)) {
      this.changed(goal);
    }
  }

  /** Keeps the update of a revision that could wake the agent, the newest {@link RUN_SNAPSHOTS_MAX} of a run. */
  private keepSnapshot(runId: string, update: Json): void {
    if (!isUsize(update.revision) || !WAKE_STATUSES.has(str(update.status) ?? '')) return;
    let kept = this.runSnapshots.get(runId);
    if (kept === undefined) {
      kept = new Map();
      this.runSnapshots.set(runId, kept);
    }
    kept.set(update.revision, runSnapshot(update));
    if (kept.size > RUN_SNAPSHOTS_MAX) kept.delete(kept.keys().next().value!);
  }

  /** The run's one entry: made at its first update with a revision, replaced in place by each higher revision. */
  private workflowEntry(runId: string, update: Json, ts: number): void {
    const revision = isUsize(update.revision) ? update.revision : null;
    if (revision === null) return;
    const existing = this.runEntries.get(runId);
    if (existing !== undefined && revision <= existing.revision) return;
    const entry: WorkflowEntry = existing ?? {
      kind: 'workflow',
      ts,
      updatedTs: ts,
      runId,
      revision,
      name: '',
      status: '',
      elapsedMs: 0,
      phases: [],
      currentPhase: null,
      agentBudget: null,
      agentsUsed: 0,
      agents: [],
      phasesEntered: [],
    };
    applyRunUpdate(entry, update, ts);
    if (existing === undefined) {
      this.runEntries.set(runId, entry);
      this.add(entry);
    } else {
      this.changed(entry);
    }
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
        edits: null,
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
    const content = Array.isArray(update.content) ? update.content : null;
    if (content !== null) acc.edits = readEdits(content, acc.entry.status === 'completed') ?? acc.edits;
    const read = content === null ? null : readPictures(content, update.rawOutput);
    if (read !== null) {
      acc.entry.pictures = read.pictures;
      if (read.pdfPages === null) delete acc.entry.pdfPages;
      else acc.entry.pdfPages = read.pdfPages;
    }
    if (update.rawOutput !== undefined) acc.rawOutput = slimRawOutput(update.rawOutput);
    if (content !== null) acc.content = toolContentText(content);
    if (
      acc.entry.status === 'failed' &&
      acc.content?.startsWith(HOOK_DENIED_TEXT) === true &&
      !this.hookDeniedSeen.has(acc.entry)
    ) {
      this.hookDeniedSeen.add(acc.entry);
      this.hookDenied.push(acc.entry);
      this.hookPairs();
    }
    this.linkResult(acc);
    this.linkRun(acc);
    if (this.ops !== null) finishTool(acc);
    if (isNew) this.add(acc.entry);
    else this.changed(acc.entry);
    return acc;
  }

  /** A `workflow` call names its kind and its run; the object is replaced when either moves. */
  private linkRun(acc: ToolAcc): void {
    const input = acc.entry.name === WORKFLOW_TOOL && isRecord(acc.rawInput) ? acc.rawInput : null;
    const kind = input === null ? null : runCallKind(input);
    if (input === null || kind === null) {
      delete acc.entry.run;
      return;
    }
    const source = input.source as Json;
    const started = isRecord(acc.rawOutput) ? str(acc.rawOutput.run_id) : null;
    const named = kind === 'resume' ? str(source.resume_from_run_id) : str(source.run_id);
    const id =
      started !== null && started !== ''
        ? started
        : kind !== 'launch' && kind !== 'check' && named !== null && named !== ''
          ? findRun(named, this.runNames)
          : null;
    const known = acc.entry.run;
    if (known?.kind !== kind || known.id !== id) acc.entry.run = { id, kind };
  }

  /**
   * The agent's answer to a host turn (`_meta.hostTurn`, live and stored; the typed text is stored only): a turn typed
   * as `/workflow pause|resume|stop <name>` names the run its name picks.
   */
  private hostTurnSaid(): void {
    const prompt = this.turnPrompt;
    if (prompt?.kind !== 'user' || prompt.runControl !== undefined) return;
    const command = workflowCommand(prompt.text);
    const runId = command === null ? null : pickRun(command.name, this.runNames);
    if (command === null || runId === null) return;
    prompt.runControl = { verb: command.verb, runId };
    this.changed(prompt);
  }

  /** A tool result that started a background task or made a schedule names it on its entry. */
  private linkResult(acc: ToolAcc): void {
    const raw = acc.rawOutput;
    if (!isRecord(raw)) return;
    const input = isRecord(acc.rawInput) ? acc.rawInput : {};
    const task = startedTask(raw);
    if (task !== null) {
      this.linkTask(acc.entry, task.id, task.kind, taskLabel(input.description, input.command), true);
      return;
    }
    const id = raw.type === 'SchedulerCreate' ? str(raw.id) : null;
    if (id === null || id === '' || this.scheduleEntries.has(id) || acc.entry.schedule !== undefined) return;
    const facts = this.earlySchedules.get(id) ?? {
      prompt: str(input.prompt) ?? '',
      repeat: str(raw.humanSchedule) ?? str(input.interval),
      nextTs: null,
    };
    this.earlySchedules.delete(id);
    acc.entry.schedule = { id, ...facts };
    this.scheduleEntries.set(id, acc.entry);
  }

  /**
   * Names task `id` on the entry that started it; another entry never takes it over. Only the tool result may change
   * the kind afterwards: `task_backgrounded` reads a monitor with a blank description as a command. True when the
   * entry changed.
   */
  private linkTask(entry: ToolEntry, id: string, kind: TaskKind, label: string, fromResult: boolean): boolean {
    const known = this.taskEntries.get(id);
    if (known !== undefined) {
      if (known !== entry || !fromResult || known.task === undefined || known.task.kind === kind) return false;
      known.task = { ...known.task, kind };
      return true;
    }
    if (entry.task !== undefined) return false;
    const end = this.earlyTaskEnds.get(id);
    this.earlyTaskEnds.delete(id);
    entry.task = {
      id,
      kind,
      label,
      ...(end === undefined ? {} : { endTs: end.ts }),
      ...(end?.killed === true ? { killed: true as const } : {}),
    };
    this.taskEntries.set(id, entry);
    return true;
  }

  /**
   * A task ended; `killed`: its `task_completed` says it was stopped (a Stop, the model's kill, a rewind). The list
   * can end a task before that line comes, so `killed` still lands on an ended task.
   */
  private endTask(id: string, ts: number, killed = false): void {
    const entry = this.taskEntries.get(id);
    if (entry === undefined) {
      const early = this.earlyTaskEnds.get(id);
      this.earlyTaskEnds.set(id, { ts: early?.ts ?? ts, killed: killed || early?.killed === true });
      return;
    }
    const task = entry.task;
    if (task === undefined) return;
    if (task.endTs !== undefined) {
      if (!killed || task.killed === true) return;
      entry.task = { ...task, killed: true };
    } else {
      entry.task = { ...task, endTs: ts, ...(killed ? { killed: true as const } : {}) };
    }
    this.changed(entry);
  }

  /**
   * `scheduled_task_created` (also after an update or a reload): the schedule exists, as it says. A fire is never
   * stored, so it never reaches here: its run and next time ride the live activity channel.
   */
  private scheduleSaid(update: Json): void {
    const id = str(update.task_id);
    if (id === null || id === '') return;
    const facts: ScheduleFacts = {
      prompt: str(update.prompt) ?? '',
      repeat: str(update.human_schedule),
      nextTs: rfc3339Ms(update.next_fire_at),
    };
    const entry = this.scheduleEntries.get(id);
    if (entry === undefined) {
      this.earlySchedules.set(id, facts);
      return;
    }
    entry.schedule = { id, ...facts };
    this.changed(entry);
  }

  /** `scheduled_task_deleted`; reason `shutdown` is the agent process stopping, and the schedule comes back with it. */
  private scheduleRemoved(update: Json, ts: number): void {
    const id = str(update.task_id);
    if (id === null || id === '' || update.reason === 'shutdown') return;
    const entry = this.scheduleEntries.get(id);
    const schedule = entry?.schedule;
    if (entry === undefined || schedule === undefined) {
      const early = this.earlySchedules.get(id);
      if (early !== undefined) this.earlySchedules.set(id, { ...early, endTs: ts });
      return;
    }
    if (schedule.endTs !== undefined) return;
    entry.schedule = { ...schedule, endTs: ts };
    this.changed(entry);
  }

  /**
   * `background_tasks`, the agent's whole list (last wins): a row that is no longer running ends its task, and so does
   * a known task missing from a list that is not cut to size (a new agent process lists none).
   */
  private taskList(update: Json, ts: number): void {
    if (!Array.isArray(update.tasks)) return;
    const listed = new Set<string>();
    for (const row of update.tasks) {
      if (!isRecord(row)) continue;
      const id = str(row.task_id);
      if (id === null || id === '') continue;
      listed.add(id);
      if (row.status === 'completed' || row.status === 'failed') this.endTask(id, rfc3339Ms(row.ended_at) ?? ts);
    }
    if (update.truncated === true) return;
    for (const [id, entry] of this.taskEntries) {
      if (!listed.has(id) && entry.task?.endTs === undefined) this.endTask(id, ts);
    }
  }

  private step(n: Notification): void {
    this.lineCopied = this.ownEvents !== null && typeof n.eventId === 'string' && !n.eventId.startsWith(this.ownEvents);
    this.stepLine(n);
    // Live, the line that opened the turn was its prompt: what was held shows now, right after it.
    if (this.ops !== null && this.openTurn) this.flushHeld();
  }

  private stepLine(n: Notification): void {
    const { update, ts } = n;
    if (ts > this.lastTs) this.lastTs = ts;
    if (n.contextUsed !== null && n.contextUsed !== this.contextUsed) {
      this.contextUsed = n.contextUsed;
      this.ops?.push({ op: 'context', used: n.contextUsed });
    }
    const tag = update.sessionUpdate;
    if (!n.xai) {
      switch (tag) {
        case 'user_message_chunk': {
          const meta = isRecord(update._meta) ? update._meta : {};
          if (str(meta.modelId) !== null) this.model = meta.modelId as string;
          const wasOpen = this.openTurn;
          this.openTurn = true;
          const hookId = this.submitId;
          this.submitId = null;
          const content = update.content;
          const displayText =
            isRecord(content) && content.type === 'text' && isRecord(content._meta)
              ? str(content._meta.displayText)
              : null;
          const said = displayText ?? contentBlockText(content);
          const promptIndex = isUsize(meta.promptIndex) ? meta.promptIndex : null;
          const body =
            promptIndex !== null && meta.interjection !== true ? (goalPrompt(said, this.lastGoal) ?? said) : said;
          if (promptIndex !== null) this.promptSeen = true;
          // A hidden chunk is a prompt the agent wrote itself, never a message of the user, even with hostTurn.
          if (meta.hideFromScrollback === true) {
            const prev = this.last();
            const run = this.wakeRun;
            if (run !== null && prev === run.entry && promptIndex === run.promptIndex) {
              this.setWakeText(run.entry, `${this.wakeTexts.get(run.entry) ?? ''}\n${body}`);
              this.changed(run.entry);
              break;
            }
            // The host's empty wake line, then the agent's own prompt of the same wake: one row.
            const open = this.turnPrompt;
            if (open?.kind === 'wake' && prev === open && this.wakeTexts.get(open) === '') {
              if (hookId !== null) applyWake(open, hookId, this.wakeRuns, this.spawnedBy, this.wakeTasks);
              this.setWakeText(open, body);
              this.changed(open);
              this.wakeRun = promptIndex === null ? null : { entry: open, promptIndex };
              break;
            }
            const entry: WakeEntry = { kind: 'wake', ts, cause: 'unknown', text: '' };
            applyWake(entry, hookId ?? str(meta[WAKE_PROMPT_ID]), this.wakeRuns, this.spawnedBy, this.wakeTasks);
            this.setWakeText(entry, body);
            this.promptStarts(wasOpen);
            this.add(entry, true);
            this.wakeRun = promptIndex === null ? null : { entry, promptIndex };
            if (this.turnPrompt === null) this.turnPrompt = entry;
            break;
          }
          const read = userAttachment(content);
          const attachment = read?.view ?? null;
          if (attachment === null && body === '') break;
          const interjection = meta.interjection === true;
          const hostNote = meta[NOTE_META] === true;
          // On disk a note is a bare chunk inside an open turn of a build that marks typed prompts.
          const diskNote =
            !hostNote &&
            !this.typedStep &&
            !this.helper &&
            wasOpen &&
            this.promptSeen &&
            attachment === null &&
            !interjection &&
            meta.hostTurn !== true &&
            promptIndex === null;
          if (hostNote || diskNote) {
            if (diskNote) {
              const count = (this.noteCounts.get(body) ?? 0) + 1;
              this.noteCounts.set(body, count);
              if (count % 2 === 0) break;
            }
            this.add({ kind: 'user', ts, text: body, promptIndex: null, note: true });
            break;
          }
          const prev = this.last();
          if (
            prev?.kind === 'user' &&
            !interjection &&
            prev.interjection !== true &&
            promptIndex !== null &&
            prev.promptIndex === promptIndex
          ) {
            if (read !== null) this.attach(prev, read);
            else prev.text = prev.text === '' ? body : `${prev.text}\n${body}`;
            this.changed(prev);
          } else {
            const entry: TranscriptEntry = {
              kind: 'user',
              ts,
              text: attachment === null ? body : '',
              promptIndex,
              ...(interjection ? { interjection: true } : {}),
              ...(!interjection && body === CONTINUE_PROMPT && isLengthEnd(prev) ? { continued: true as const } : {}),
            };
            if (read !== null) this.attach(entry, read);
            if (!interjection) this.promptStarts(wasOpen);
            this.add(entry, !interjection);
            if (!interjection) this.noteCounts.clear();
            if (this.turnPrompt === null && !interjection) this.turnPrompt = entry;
          }
          break;
        }
        case 'agent_message_chunk':
        case 'agent_thought_chunk': {
          const content = update.content;
          if (!isRecord(content) || content.type !== 'text') break;
          const body = str(content.text) ?? '';
          if (body === '') break;
          if (this.afterTry(n.eventId, ts)) this.retryEnds(ts, 'recovered');
          if (tag === 'agent_message_chunk' && isRecord(update._meta) && update._meta.hostTurn === true)
            this.hostTurnSaid();
          const kind = tag === 'agent_message_chunk' ? 'assistant' : 'thinking';
          const prev = this.last();
          if (prev?.kind === kind) {
            const text = prev as Timed<'assistant' | 'thinking'>;
            text.text += body;
            text.endTs = ts;
            if (this.ops !== null) {
              this.ops.push({ op: 'text', index: this.transcript.length - 1, append: body, endTs: ts });
            }
          } else {
            const entry: Timed<'assistant' | 'thinking'> = { kind, ts, text: body, endTs: ts };
            // Only a thought made from live lines has a real time: the agent stores one merged chunk with its last stamp.
            if (kind === 'thinking' && this.ops !== null)
              (entry as Extract<TranscriptEntry, { kind: 'thinking' }>).timed = true;
            this.add(entry);
          }
          break;
        }
        case 'tool_call':
          this.placeOneShots(ts);
          if (this.tool(update, ts) === null) break;
          this.openTurn = true;
          if (!this.afterTry(n.eventId, ts)) break;
          this.retryEnds(ts, 'recovered');
          if (this.retryLast !== null) this.retryLast.toolSince = true;
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
        case 'current_mode_update': {
          const id = str(update.currentModeId);
          if (id === null) break;
          const mode = sessionMode(id);
          if (mode === this.planMode) break;
          this.planMode = mode;
          this.modeMoves += 1;
          this.side({ kind: 'plan-mode', ts, mode });
          break;
        }
      }
      return;
    }
    switch (tag) {
      case 'turn_completed': {
        this.hookTurnEnds();
        // A prompt a hook blocked stores no chunk, so its turn never opened: its notes still belong to it.
        this.flushHeld();
        this.retryEnds(ts, undefined);
        this.retryLast = null;
        this.openTurn = false;
        this.noteCounts.clear();
        const promptId = str(update.prompt_id);
        const prompt = this.turnPrompt;
        // The wake's own id wins over the host's guess, so a live row ends as the stored one.
        if (prompt?.kind === 'wake' && promptId !== null && promptId !== '') {
          applyWake(prompt, promptId, this.wakeRuns, this.spawnedBy, this.wakeTasks);
          this.changed(prompt);
        }
        this.quietEnds = [];
        this.turnPrompt = null;
        this.submitId = null;
        const usage = isRecord(update.usage) ? update.usage : null;
        if (usage !== null) this.turnsWithUsage += 1;
        const perf = isRecord(update.perf) ? update.perf : null;
        const prefill = measuredPrefill(perf);
        const elapsedMs = num(update.elapsed_ms);
        const stopReason = str(update.stop_reason) ?? 'unknown';
        const errorText = str(update.agent_result);
        const errorKind = str(update.error_kind);
        const isError = stopReason === 'error' || errorKind !== null;
        this.add({
          kind: 'turn-end',
          ts,
          stopReason,
          ...(elapsedMs !== null ? { elapsedMs } : {}),
          ...(isError && errorText !== null && errorText !== '' ? { error: errorText } : {}),
          ...(errorKind === 'max_tokens_truncation' ? { lengthLimit: true as const } : {}),
          ...(stopReason === 'cancelled' && n.cancelCategory === HOOK_DENIED_CATEGORY
            ? { hookStopped: true as const }
            : {}),
        });
        const metric: TurnMetric = {
          promptId: promptId ?? '',
          ts,
          ...(this.lineCopied ? { copied: true as const } : {}),
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
        const run = str(update.workflow_run_id);
        if (run !== null && run !== '') entry.run = run;
        const resumedFrom = str(update.resumed_from);
        if (resumedFrom !== null && resumedFrom !== '') entry.resumedFrom = resumedFrom;
        const model = str(update.model);
        if (model !== null && model !== '') entry.model = model;
        this.subagents.set(childId, entry);
        const subagentId = str(update.subagent_id);
        this.spawnedBy.set(subagentId !== null && subagentId !== '' ? subagentId : childId, entry);
        this.add(entry);
        break;
      }
      case 'workflow_updated': {
        const runId = str(update.run_id);
        if (runId === null || runId === '') break;
        this.runNames.set(runId, str(update.name) ?? '');
        this.keepSnapshot(runId, update);
        this.workflowEntry(runId, update, ts);
        break;
      }
      case 'hook_execution': {
        if (update.event_name === 'user_prompt_submit') this.submitId = str(update.prompt_id);
        if (!Array.isArray(update.runs)) break;
        const event = str(update.event_name) ?? '';
        const tool = str(update.tool_name);
        for (const run of update.runs) {
          // A `blocked` run is the hook's decision, not a failure: its note or the call's row tells it.
          if (!isRecord(run) || !isRecord(run.status) || run.status.status !== 'failed' || run.status.blocked === true)
            continue;
          this.hookFailed(
            {
              event,
              ...(tool === null || tool === '' ? {} : { tool }),
              hook: hookSource(str(run.name) ?? ''),
              detail: hookText(run.status.error) ?? '',
            },
            ts,
          );
        }
        break;
      }
      case 'hook_annotation': {
        const text = hookText(update.message);
        if (text === undefined) break;
        if (update.kind === 'tool_outcome') {
          this.hookVerdicts.push({ text, ts });
          for (const entry of this.hookPairs()) this.changed(entry);
          break;
        }
        this.oneShots.push({ kind: 'notice', ts, code: 'hook-note', text });
        break;
      }
      case 'task_backgrounded': {
        const id = str(update.task_id);
        const callId = str(update.tool_call_id);
        const acc = callId === null ? undefined : this.tools.get(callId);
        if (acc === undefined || id === null || id === '') break;
        const monitor = str(update.monitor_description);
        const label = taskLabel(monitor ?? update.description, update.command);
        if (this.linkTask(acc.entry, id, monitor !== null ? 'monitor' : 'command', label, false)) {
          this.changed(acc.entry);
        }
        break;
      }
      case 'task_completed': {
        const snapshot = isRecord(update.task_snapshot) ? update.task_snapshot : null;
        const id = str(snapshot?.task_id);
        if (snapshot === null || id === null || id === '') break;
        this.endTask(id, systemTimeMs(snapshot.end_time) ?? ts, snapshot.explicitly_killed === true);
        const kind = snapshot.kind === 'monitor' ? 'monitor' : 'command';
        this.taskEnds.set(id, {
          id,
          kind,
          label: taskLabel(snapshot.description, snapshot.display_command ?? snapshot.command),
        });
        // grok `is_auto_wake_suppressed`: the agent never reports such an end in a later turn.
        const unreported =
          snapshot.block_waited === true ||
          (snapshot.explicitly_killed === true && snapshot.kill_result_delivered === true);
        if (
          update.will_wake !== true &&
          !unreported &&
          !this.quietEnds.includes(id) &&
          this.quietEnds.length < WAKE_TASKS_MAX
        ) {
          this.quietEnds = [...this.quietEnds, id];
        }
        break;
      }
      case 'background_tasks':
        this.taskList(update, ts);
        break;
      case 'scheduled_task_created':
        this.scheduleSaid(update);
        break;
      case 'scheduled_task_deleted':
        this.scheduleRemoved(update, ts);
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
        const toolCalls = num(update.tool_calls);
        if (toolCalls !== null) entry.toolCalls = toolCalls;
        const turns = num(update.turns);
        if (turns !== null) entry.turns = turns;
        const error = str(update.error);
        if (error !== null && error !== '') entry.error = error;
        if (isNew) this.add(entry);
        else this.changed(entry);
        break;
      }
      case 'retry_state': {
        if (update.type === 'failed' || update.type === 'exhausted') {
          this.retryEnds(ts, 'gave-up', noticeText(update.type === 'failed' ? update.message : update.reason));
          break;
        }
        const attempt = num(update.attempt);
        if (update.type !== 'retrying' || attempt === null) break;
        this.retries.push({ ts, attempt, max: num(update.max_retries) ?? 0 });
        const tries = countOrNull(attempt);
        if (tries === null) break;
        const reason = str(update.reason) ?? '';
        const cause = retryCause(str(update.error_type), reason);
        this.retrying(
          tries,
          countOrNull(update.max_retries) ?? 0,
          cause,
          cause === 'other' ? noticeText(reason) : undefined,
          ts,
          n.eventId,
        );
        break;
      }
      case 'auto_recovery_started': {
        const tries = countOrNull(update.attempt);
        if (tries !== null)
          this.retrying(tries, countOrNull(update.max_retries) ?? 0, 'recovery', undefined, ts, n.eventId);
        break;
      }
      case 'auto_recovery_exhausted':
        this.retryEnds(ts, 'gave-up', noticeText(update.error));
        break;
      case 'image_dropped': {
        const notes = pictureNotes(update.notes);
        if (notes.length === 0) break;
        this.oneShots.push({
          kind: 'notice',
          ts,
          code: 'picture',
          what: 'dropped',
          count: notes.length,
          notes: notes.slice(0, PICTURE_ITEMS_MAX),
        });
        break;
      }
      case 'image_compressed': {
        // `images: []` is a re-encode that failed with the original kept: nothing changed for the user.
        if (!Array.isArray(update.images) || update.images.length === 0) break;
        const note = noticeText(update.message);
        const entry: PictureNotice = {
          kind: 'notice',
          ts,
          code: 'picture',
          what: 'shrunk',
          count: update.images.length,
          notes: note === undefined ? [] : [note],
        };
        const sizes = pictureSizes(update.images);
        if (sizes.length > 0) entry.sizes = sizes.slice(0, PICTURE_ITEMS_MAX);
        this.oneShots.push(entry);
        break;
      }
      case 'session_recap': {
        const text = str(update.summary)?.trim() ?? '';
        if (text === '') break;
        const cut = truncateField(text, RECAP_TEXT_CHARS);
        this.add({ kind: 'recap', ts, text: cut.truncated ? `${cut.text}…` : cut.text });
        break;
      }
      case 'session_recap_unavailable':
        // Stored, but no fact for the thread: the host reads it to end the wait for a recap.
        break;
      case 'goal_updated':
        this.goalLine(update, ts);
        break;
      case 'model_changed': {
        const modelId = str(update.model_id);
        if (modelId === null || modelId === '') break;
        this.model = modelId;
        this.add({ kind: 'model', ts, modelId });
        break;
      }
      case 'auto_compact_completed':
      case 'auto_compact_failed':
      case 'auto_compact_cancelled': {
        const end = readCompaction(update);
        if (end === null || end.kind === 'started') break;
        if (end.kind === 'completed') {
          // The lines after a compaction can keep the old count until the next answer: a lower count here wins now.
          if (end.tokensAfter !== null && this.contextUsed !== null && end.tokensAfter < this.contextUsed) {
            this.contextUsed = end.tokensAfter;
            this.ops?.push({ op: 'context', used: end.tokensAfter });
          }
          this.add({
            kind: 'compaction',
            ts,
            outcome: 'done',
            ...(end.tokensBefore === null ? {} : { tokensBefore: end.tokensBefore }),
            ...(end.tokensAfter === null ? {} : { tokensAfter: end.tokensAfter }),
          });
        } else if (end.kind === 'failed') {
          this.add({ kind: 'compaction', ts, outcome: 'failed', ...(end.retry ? { retry: true as const } : {}) });
        } else {
          this.add({ kind: 'compaction', ts, outcome: 'stopped' });
        }
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
 * A tool entry's `input`, `output` and `edits`. `output` is the content text when there is one, else the text of a
 * successful MCP result, else the JSON text of `rawOutput`: an MCP result is kept as plain text, never as JSON cut
 * mid-string. A shell command's output is its content text only (none when it printed nothing); how it ended is kept
 * apart. An edit's output is its content text only (the agent's error); its change is `edits`. A picture read's output
 * is its content text only (none); its pictures are `pictures`.
 */
function finishTool(acc: ToolAcc): void {
  if (acc.edits !== null) {
    acc.entry.edits = diffEdits(acc.edits);
    acc.edits = null;
  }
  const input = fieldText(acc.rawInput);
  const mcp = mcpOutputText(acc.rawOutput);
  const bash = bashEnd(acc.rawOutput);
  const shown = acc.entry.edits !== undefined || acc.entry.pictures !== undefined;
  const output =
    acc.content !== null && acc.content !== ''
      ? fieldText(acc.content)
      : bash !== null || shown
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
  } else if (bash !== null || shown) delete acc.entry.output;
  if (truncated) acc.entry.truncated = true;
  else delete acc.entry.truncated;
  if (acc.entry.name === ASK_TOOL) {
    const ask = askOf(acc.rawInput, acc.entry.status, acc.content);
    if (ask === null) delete acc.entry.ask;
    else acc.entry.ask = ask;
  }
}

/**
 * The image blocks of one tool call in a whole `updates.jsonl` text, rewinds applied first, as {@link readPictures}
 * numbers them: those of the call's newest line that has any. Null when no such line is left. `callId` is compared
 * for equality only.
 */
export function toolPictureBlocks(text: string, callId: string): Array<{ mediaType: string; data: string }> | null {
  let found: Array<{ mediaType: string; data: string }> | null = null;
  for (const line of filterRewindLines(splitCompleteLines(text))) {
    if (!line.includes('"image"')) continue;
    const n = parseLine(line);
    if (n === null || n.xai) continue;
    const update = n.update;
    if (update.sessionUpdate !== 'tool_call' && update.sessionUpdate !== 'tool_call_update') continue;
    if (update.toolCallId !== callId || !Array.isArray(update.content)) continue;
    const blocks: Array<{ mediaType: string; data: string }> = [];
    for (const item of update.content) {
      if (!isRecord(item) || item.type !== 'content' || !isRecord(item.content)) continue;
      const block = item.content;
      if (block.type === 'image') blocks.push({ mediaType: str(block.mimeType) ?? '', data: str(block.data) ?? '' });
    }
    if (blocks.length > 0) found = blocks;
  }
  return found;
}

export function parseUpdates(text: string, options: { helper?: boolean; sessionId?: string } = {}): ParsedUpdates {
  const reducer = new TranscriptReducer(options);
  reducer.applyText(text);
  return reducer.parsed();
}

/** The counter that ends an `eventId` (`<session id>-<n>`); null when it ends in none. */
export function counterOf(id: string): number | null {
  const tail = id.slice(id.lastIndexOf('-') + 1);
  if (tail === '') return null;
  for (const c of tail) if (c < '0' || c > '9') return null;
  return Number(tail);
}

interface EventOrder {
  session: string;
  n: number;
}

function eventOrder(id: string | null | undefined): EventOrder | null {
  if (typeof id !== 'string') return null;
  const n = counterOf(id);
  return n === null ? null : { session: id.slice(0, id.lastIndexOf('-')), n };
}

/**
 * The `_meta.eventId` of each line of a whole `updates.jsonl` text, rewinds applied first, those of its
 * `turn_completed` lines in file order, and that of its last ACP line.
 */
export function storedEventIds(text: string): { ids: Set<string>; turnEnds: string[]; lastAcp: string | null } {
  const ids = new Set<string>();
  const turnEnds: string[] = [];
  let lastAcp: string | null = null;
  for (const line of filterRewindLines(splitCompleteLines(text))) {
    const n = parseLine(line);
    if (n === null || typeof n.eventId !== 'string') continue;
    ids.add(n.eventId);
    if (!n.xai) lastAcp = n.eventId;
    else if (n.update.sessionUpdate === 'turn_completed') turnEnds.push(n.eventId);
  }
  return { ids, turnEnds, lastAcp };
}
