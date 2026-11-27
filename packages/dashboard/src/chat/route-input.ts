/**
 * The input law of the Chat routes: every body is checked against its exact field list, and nothing from the page
 * is passed through. Only the message text, its checked attachments (and, in Work, the folder) reach the agent. Text
 * that starts with `/` is a slash command the agent runs (`/yolo` turns every permission prompt off): one whose name
 * the agent would not read, an app command, a blocked or a hidden built-in is refused here, in both modes; the
 * service checks the rest against the conversation's mode and the agent's list (`commands.ts`).
 */

import { ApiError } from '../api/errors.js';
import {
  FIELDS_MAX,
  OPTIONS_MAX as MCP_OPTIONS_MAX,
  VALUE_MAX_CHARS,
  type McpAnswerInput,
  type McpValue,
} from './agent/elicit.js';
import { OPTIONS_MAX, QUESTIONS_MAX, type QuestionAnswerInput } from './agent/questions.js';
import {
  ATTACH_MAX,
  attachmentView,
  base64Bytes,
  FILE_NUMBERED_MAX_BYTES,
  IMAGE_MAX_BASE64_CHARS,
  IMAGE_MAX_BYTES,
  isBase64,
  isPdfNoteText,
  magicMatches,
  MESSAGE_FILES_MAX_BYTES,
  MESSAGE_IMAGES_MAX_BASE64,
  NAME_IN_MAX_CHARS,
  numberedBytes,
  PAGE_MAX_BASE64_CHARS,
  PAGE_MAX_BYTES,
  PDF_MESSAGE_PAGES,
  pdfNotes,
  safeName,
  storedPictureChars,
  wellFormed,
  utf8Bytes,
  type AttachmentInput,
} from './attachment-rule.js';
import { parseSlash, refusedName, startsWithSlash } from './commands.js';
import { isProviderId, isSearchProviderId } from './providers/settings.js';
import type { ProviderId, ProviderSelectionInput } from './providers/types.js';
import { chatRouteError } from './route-errors.js';
import { isThinkingLevel } from './thinking-levels.js';
import {
  THINKING_LEVELS,
  type ChatMode,
  type GoalAction,
  type PermissionMode,
  type RewindMode,
  type ThinkingLevel,
} from './types.js';
import type { WorkSettings } from './work-settings.js';

export const MAX_TEXT_CHARS = 100_000;
const MAX_ID_CHARS = 128;
const MAX_MODEL_ID_CHARS = 512;
export const MAX_FEEDBACK_CHARS = 4_000;
const MAX_CWD_CHARS = 4_096;

/** Whether the first character after leading white space is `/`. */
export const isSlashText = startsWithSlash;

function bad(message: string): ApiError {
  return ApiError.badRequest(message);
}

type FieldKind = 'string' | 'boolean' | 'array' | 'number';

/**
 * The body as a plain object whose keys are all in `allowed` with the listed types, and every `required` key set.
 * An unknown key, a wrong type or a missing required key is `E_BAD_REQUEST`.
 */
function exactBody(
  body: unknown,
  allowed: Readonly<Record<string, FieldKind>>,
  required: readonly string[],
): Record<string, unknown> {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) throw bad('The body must be an object.');
  const record = body as Record<string, unknown>;
  for (const key of Object.keys(record)) {
    const kind = Object.hasOwn(allowed, key) ? allowed[key] : undefined;
    if (kind === undefined) throw bad(`Unknown field "${fieldName(key)}".`);
    const fits = kind === 'array' ? Array.isArray(record[key]) : typeof record[key] === kind;
    if (!fits) throw bad(`Field "${fieldName(key)}" has the wrong type.`);
  }
  for (const key of required) if (!Object.hasOwn(record, key)) throw bad(`Field "${key}" is required.`);
  return record;
}

/** A field name as it may appear in an error: short, printable ASCII only. */
function fieldName(key: string): string {
  let out = '';
  for (const ch of key.slice(0, 40)) {
    const code = ch.charCodeAt(0);
    out += code >= 0x20 && code < 0x7f && ch !== '"' ? ch : '?';
  }
  return out;
}

/** Message text: 1 to 100,000 characters, not blank; slash text names a command no mode refuses. */
export function checkText(text: string): string {
  if (text.trim() === '') throw bad('The message is empty.');
  if (text.length > MAX_TEXT_CHARS) throw new ApiError('E_PAYLOAD_TOO_LARGE', 'Too long. 100,000 characters at most.');
  if (isSlashText(text)) {
    const slash = parseSlash(text);
    if (slash === null || refusedName(slash.name)) throw chatRouteError('slash-text');
  }
  return text;
}

function checkMode(value: unknown): ChatMode {
  if (value !== 'chat' && value !== 'work') throw bad('Field "mode" must be "chat" or "work".');
  return value;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasExactKeys(record: Record<string, unknown>, keys: readonly string[]): boolean {
  const own = Object.keys(record);
  return own.length === keys.length && keys.every((key) => Object.hasOwn(record, key));
}

function checkName(name: unknown): name is string {
  return typeof name === 'string' && name.length > 0 && name.length <= NAME_IN_MAX_CHARS;
}

function checkAttachment(item: unknown): { attachment: AttachmentInput; textBytes: number } {
  const refused = chatRouteError('attachment-refused');
  if (!isPlainObject(item)) throw refused;
  if (item.kind === 'file') {
    if (!hasExactKeys(item, ['kind', 'name', 'text'])) throw refused;
    const { name, text } = item;
    if (!checkName(name)) throw refused;
    if (typeof text !== 'string' || !wellFormed(text) || text.includes('\0') || text.trim() === '') throw refused;
    // The transcript reader takes a resource that starts with the note's line as a PDF.
    if (isPdfNoteText(text)) throw refused;
    const textBytes = numberedBytes(text);
    if (textBytes > FILE_NUMBERED_MAX_BYTES) throw refused;
    return { attachment: { kind: 'file', name: safeName(name), text }, textBytes };
  }
  if (item.kind === 'image') {
    if (!hasExactKeys(item, ['kind', 'mediaType', 'data'])) throw refused;
    const { mediaType, data } = item;
    if (mediaType !== 'image/png' && mediaType !== 'image/jpeg') throw refused;
    if (typeof data !== 'string' || data.length < 4 || data.length > IMAGE_MAX_BASE64_CHARS) throw refused;
    if (!isBase64(data) || base64Bytes(data) > IMAGE_MAX_BYTES || !magicMatches(mediaType, data)) throw refused;
    return { attachment: { kind: 'image', mediaType, data }, textBytes: 0 };
  }
  if (item.kind === 'pdf') {
    if (!hasExactKeys(item, ['kind', 'name', 'images'])) throw refused;
    const { name, images } = item;
    if (!checkName(name)) throw refused;
    if (!Array.isArray(images) || images.length === 0 || images.length > PDF_MESSAGE_PAGES) throw refused;
    const pages: string[] = [];
    for (const page of images as unknown[]) {
      if (typeof page !== 'string' || page.length < 4 || page.length > PAGE_MAX_BASE64_CHARS) throw refused;
      if (!isBase64(page) || base64Bytes(page) > PAGE_MAX_BYTES || !magicMatches('image/jpeg', page)) throw refused;
      // A page is stored as sent and counted as padded base64, so it must be exactly that or it outgrows its count.
      if (page.length !== Math.ceil(base64Bytes(page) / 3) * 4) throw refused;
      pages.push(page);
    }
    return { attachment: { kind: 'pdf', name: safeName(name), images: pages }, textBytes: 0 };
  }
  throw refused;
}

/**
 * The attachments of a message, every limit checked: at most {@link ATTACH_MAX}; a file is well-formed text with no
 * NUL and a cleaned name, and does not start like a PDF note; a picture is PNG or JPEG base64 whose first bytes match
 * its type; a PDF is 1 to 20 JPEG pages of at most {@link PAGE_MAX_BYTES} each, as padded base64 with no whitespace.
 * In all, at most {@link PDF_MESSAGE_PAGES} pages, {@link MESSAGE_IMAGES_MAX_BASE64} of picture and page base64, and
 * the message text with the numbered file and note text under the agent's prompt cut. Empty is no attachments. Any
 * failure is `attachment-refused`.
 */
export function checkAttachments(value: unknown[], text: string): AttachmentInput[] | undefined {
  if (value.length === 0) return undefined;
  const refused = chatRouteError('attachment-refused');
  if (value.length > ATTACH_MAX) throw refused;
  let bytes = utf8Bytes(text);
  let pages = 0;
  const out: AttachmentInput[] = [];
  for (const item of value) {
    const { attachment, textBytes } = checkAttachment(item);
    bytes += textBytes;
    if (attachment.kind === 'pdf') pages += attachment.images.length;
    out.push(attachment);
  }
  for (const note of pdfNotes(out)) bytes += numberedBytes(note);
  if (bytes > MESSAGE_FILES_MAX_BYTES || pages > PDF_MESSAGE_PAGES) throw refused;
  if (storedPictureChars(out.map(attachmentView)) > MESSAGE_IMAGES_MAX_BASE64) throw refused;
  return out;
}

function checkModelId(value: string): string {
  if (value === '' || value.length > MAX_MODEL_ID_CHARS) throw bad('Field "modelId" is not a model id.');
  return value;
}

export interface CreateConversationInput {
  mode: ChatMode;
  text: string;
  modelId?: string;
  cwd?: string;
  /** Work: no folder named; the app makes the conversation's own folder under the scratch root. */
  noFolder?: true;
  /** Work: the New screen's draft id. The mode picked for it is the Control Panel's record, never the page's word. */
  draft?: string;
  attachments?: AttachmentInput[];
  /** Work: the first message asks for plan mode. */
  plan?: boolean;
  /** The thinking level the conversation starts with; absent: the model's default. */
  thinking?: ThinkingLevel;
}

export const PLAN_WORK_ONLY = 'Field "plan" is only accepted in Work mode.';
const PLACE_TWICE = 'Name a folder or no folder, not both.';

export function createConversationInput(body: unknown): CreateConversationInput {
  const record = exactBody(
    body,
    {
      mode: 'string',
      text: 'string',
      modelId: 'string',
      cwd: 'string',
      noFolder: 'boolean',
      draft: 'string',
      attachments: 'array',
      plan: 'boolean',
      thinking: 'string',
    },
    ['mode', 'text'],
  );
  const mode = checkMode(record.mode);
  if (record.cwd !== undefined && mode !== 'work') throw bad('Field "cwd" is only accepted in Work mode.');
  if (record.noFolder !== undefined) {
    if (mode !== 'work') throw bad('Field "noFolder" is only accepted in Work mode.');
    if (record.noFolder !== true) throw bad('Field "noFolder" must be true.');
    if (record.cwd !== undefined) throw bad(PLACE_TWICE);
  }
  if (record.draft !== undefined && mode !== 'work') throw bad('Field "draft" is only accepted in Work mode.');
  if (record.plan !== undefined && mode !== 'work') throw bad(PLAN_WORK_ONLY);
  const input: CreateConversationInput = { mode, text: checkText(record.text as string) };
  if (record.plan !== undefined) input.plan = record.plan as boolean;
  if (record.thinking !== undefined) input.thinking = checkThinking(record.thinking);
  if (record.modelId !== undefined) input.modelId = checkModelId(record.modelId as string);
  if (record.cwd !== undefined) {
    const cwd = record.cwd as string;
    if (cwd === '' || cwd.length > MAX_CWD_CHARS) throw chatRouteError('cwd-invalid');
    input.cwd = cwd;
  }
  if (record.noFolder === true) input.noFolder = true;
  if (record.draft !== undefined) {
    const draft = record.draft as string;
    if (!isIdShape(draft)) throw bad('Field "draft" is not a draft id.');
    input.draft = draft;
  }
  const attachments = attachmentsOf(record.attachments, input.text);
  if (attachments !== undefined) input.attachments = attachments;
  return input;
}

function attachmentsOf(value: unknown, text: string): AttachmentInput[] | undefined {
  return value === undefined ? undefined : checkAttachments(value as unknown[], text);
}

export interface PromptInput {
  text: string;
  attachments?: AttachmentInput[];
  plan?: boolean;
  /** While the conversation is busy the message joins its queue (`queue`), or goes to the running answer (`now`). */
  whenBusy?: 'queue' | 'now';
}

/** `plan` is Work only: the service refuses it for a Chat conversation. */
export function promptInput(body: unknown): PromptInput {
  const record = exactBody(body, { text: 'string', attachments: 'array', plan: 'boolean', whenBusy: 'string' }, [
    'text',
  ]);
  const whenBusy = record.whenBusy;
  if (whenBusy !== undefined && whenBusy !== 'queue' && whenBusy !== 'now') {
    throw bad('Field "whenBusy" must be "queue" or "now".');
  }
  const text = checkText(record.text as string);
  const attachments = attachmentsOf(record.attachments, text);
  return {
    text,
    ...(attachments === undefined ? {} : { attachments }),
    ...(record.plan === undefined ? {} : { plan: record.plan as boolean }),
    ...(whenBusy === undefined ? {} : { whenBusy }),
  };
}

/** `POST …/queue/:qid/send`: exactly `{how: "now"}`. */
export function queueSendInput(body: unknown): { how: 'now' } {
  const record = exactBody(body, { how: 'string' }, ['how']);
  if (record.how !== 'now') throw bad('Field "how" must be "now".');
  return { how: 'now' };
}

/**
 * `{text}`: a message to a subagent, 1 to 100,000 characters, not blank. A slash is plain text there: it never runs
 * a command. The agent's own byte limit is the host's to answer.
 */
export function subagentMessageInput(body: unknown): { text: string } {
  const record = exactBody(body, { text: 'string' }, ['text']);
  const text = record.text as string;
  if (text.trim() === '') throw bad('The message is empty.');
  if (text.length > MAX_TEXT_CHARS) throw new ApiError('E_PAYLOAD_TOO_LARGE', 'Too long. 100,000 characters at most.');
  return { text };
}

/** `{window}`: the id the page made for its window, which names the window's Agent views. */
export function agentWatchInput(body: unknown): { window: string } {
  const record = exactBody(body, { window: 'string' }, ['window']);
  const window = record.window as string;
  if (!isIdShape(window)) throw bad('Field "window" is not a window id.');
  return { window };
}

/** `?window=<id>`: the same id, when the view closes. */
export function agentWatchQuery(query: URLSearchParams): { window: string } {
  const window = query.get('window');
  if (window === null || !isIdShape(window)) throw bad('Query "window" must be a window id.');
  return { window };
}

/** A rewind plan's token: what the plan route sent, at most this long. */
const MAX_REWIND_TOKEN_CHARS = 64;

/** `{entry, ts}`: the index of a message among the conversation's entries as the page holds them, and its time. */
export function rewindPlanInput(body: unknown): { entry: number; ts: number } {
  const record = exactBody(body, { entry: 'number', ts: 'number' }, ['entry', 'ts']);
  const entry = record.entry as number;
  if (!Number.isSafeInteger(entry) || entry < 0) throw bad('Field "entry" must be an entry index.');
  const ts = record.ts as number;
  if (!Number.isFinite(ts)) throw bad('Field "ts" must be the time of the entry.');
  return { entry, ts };
}

/**
 * `{}` or `{entry, ts}`: Fork; with `entry`, the message whose answer the copy ends with (its index among the page's
 * entries) and `ts`, its time.
 */
export function forkInput(body: unknown): { at?: { entry: number; ts: number } } {
  const record = exactBody(body, { entry: 'number', ts: 'number' }, []);
  if (record.entry === undefined && record.ts === undefined) return {};
  return { at: rewindPlanInput(record) };
}

/** `{action}`: a press on the goal strip. */
export function goalInput(body: unknown): { action: GoalAction } {
  const record = exactBody(body, { action: 'string' }, ['action']);
  const action = record.action;
  if (action !== 'pause' && action !== 'resume' && action !== 'clear')
    throw bad('Field "action" must be "pause", "resume" or "clear".');
  return { action };
}

export interface RewindInput {
  token: string;
  mode: RewindMode;
}

/** `{token, mode}`: the plan's token and what to rewind: the conversation and the files, the conversation, the files. */
export function rewindInput(body: unknown): RewindInput {
  const record = exactBody(body, { token: 'string', mode: 'string' }, ['token', 'mode']);
  const token = record.token as string;
  if (token === '' || token.length > MAX_REWIND_TOKEN_CHARS) throw bad('Field "token" is not a rewind token.');
  const mode = record.mode;
  if (mode !== 'all' && mode !== 'conversation' && mode !== 'files')
    throw bad('Field "mode" must be "all", "conversation" or "files".');
  return { token, mode };
}

export type QueuePatchInput = { text: string } | { editing: boolean };

/** Exactly one of `{text}` (the row's new text, under the message text law) or `{editing}`. */
export function queuePatchInput(body: unknown): QueuePatchInput {
  if (typeof body === 'object' && body !== null && Object.hasOwn(body, 'editing')) {
    const record = exactBody(body, { editing: 'boolean' }, ['editing']);
    return { editing: record.editing as boolean };
  }
  const record = exactBody(body, { text: 'string' }, ['text']);
  return { text: checkText(record.text as string) };
}

export type InteractionAnswerInput =
  | { optionId: string; note?: string }
  /** "Never allow" on a command card: the scope of words the card offered. */
  | { never: 'short' | 'exact' }
  | { decision: 'approved' | 'rejected'; feedback?: string }
  | QuestionAnswerInput
  | McpAnswerInput;

/** The option a note rides on: the agent reads a note only with a deny-once answer. */
const NOTE_OPTION_ID = 'reject-once';

/**
 * A question card's answer: per question, null or `{picks, other?}` with option indexes (each once) and the user's own
 * text. The host checks the indexes against the card.
 */
function questionAnswers(value: unknown[]): Array<{ picks: number[]; other?: string } | null> {
  if (value.length === 0 || value.length > QUESTIONS_MAX) throw bad('Field "answers" must list each question.');
  return value.map((item) => {
    if (item === null) return null;
    const record = exactBody(item, { picks: 'array', other: 'string' }, ['picks']);
    const picks = record.picks as unknown[];
    if (picks.length > OPTIONS_MAX) throw bad('Too many picks.');
    const seen = new Set<number>();
    for (const pick of picks) {
      if (typeof pick !== 'number' || !Number.isInteger(pick) || pick < 0 || pick >= OPTIONS_MAX || seen.has(pick)) {
        throw bad('A pick is not an option.');
      }
      seen.add(pick);
    }
    if (record.other === undefined) return { picks: picks as number[] };
    const other = record.other as string;
    if (other.length > MAX_FEEDBACK_CHARS) throw bad('Field "other" is too long.');
    if (!wellFormed(other)) throw bad('Field "other" is not text.');
    return { picks: picks as number[], other };
  });
}

function isIndex(value: unknown, max: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 && value < max;
}

/**
 * An MCP form's values, one per shown field: text, a switch, an option index, option indexes (each once), or null.
 * The host checks them against the form. No error names a value.
 */
function mcpValues(value: unknown[]): McpValue[] {
  if (value.length > FIELDS_MAX) throw bad('Too many values.');
  return value.map((item) => {
    if (item === null || typeof item === 'boolean') return item;
    if (typeof item === 'string') {
      if (item.length > VALUE_MAX_CHARS) throw bad('A value is too long.');
      if (!wellFormed(item)) throw bad('A value is not text.');
      return item;
    }
    if (isIndex(item, MCP_OPTIONS_MAX)) return item;
    if (!Array.isArray(item) || item.length > MCP_OPTIONS_MAX) throw bad('A value is not a value of the form.');
    const seen = new Set<number>();
    for (const pick of item) {
      if (!isIndex(pick, MCP_OPTIONS_MAX) || seen.has(pick)) throw bad('A pick is not an option.');
      seen.add(pick);
    }
    return item as number[];
  });
}

export function interactionInput(body: unknown): InteractionAnswerInput {
  if (typeof body === 'object' && body !== null && Object.hasOwn(body, 'never')) {
    const record = exactBody(body, { never: 'string' }, ['never']);
    if (record.never !== 'short' && record.never !== 'exact') throw bad('Field "never" must be "short" or "exact".');
    return { never: record.never };
  }
  if (typeof body === 'object' && body !== null && Object.hasOwn(body, 'mcp')) {
    const record = exactBody(body, { mcp: 'string', values: 'array' }, ['mcp']);
    if (record.mcp === 'decline') {
      if (record.values !== undefined) throw bad('A decline sends no values.');
      return { mcp: 'decline' };
    }
    if (record.mcp !== 'accept') throw bad('Field "mcp" must be "accept" or "decline".');
    return { mcp: 'accept', values: mcpValues((record.values as unknown[] | undefined) ?? []) };
  }
  if (typeof body === 'object' && body !== null && Object.hasOwn(body, 'reply')) {
    const record = exactBody(body, { reply: 'string', answers: 'array' }, ['reply']);
    const reply = record.reply;
    if (reply === 'skip') {
      if (record.answers !== undefined) throw bad('A skip sends no answers.');
      return { reply };
    }
    if (reply !== 'discuss' && reply !== 'plan-now') {
      throw bad('Field "reply" must be "skip", "discuss" or "plan-now".');
    }
    if (record.answers === undefined) return { reply };
    return { reply, answers: questionAnswers(record.answers as unknown[]) };
  }
  if (typeof body === 'object' && body !== null && Object.hasOwn(body, 'answers')) {
    const record = exactBody(body, { answers: 'array' }, ['answers']);
    return { answers: questionAnswers(record.answers as unknown[]) };
  }
  if (typeof body === 'object' && body !== null && Object.hasOwn(body, 'optionId')) {
    const record = exactBody(body, { optionId: 'string', note: 'string' }, ['optionId']);
    const optionId = record.optionId as string;
    if (optionId === '' || optionId.length > MAX_ID_CHARS) throw bad('Field "optionId" is not an option id.');
    if (record.note === undefined) return { optionId };
    const note = record.note as string;
    if (optionId !== NOTE_OPTION_ID) throw bad('A note goes only with "reject-once".');
    if (note.length > MAX_FEEDBACK_CHARS) throw bad('Field "note" is too long.');
    if (!wellFormed(note)) throw bad('Field "note" is not text.');
    return { optionId, note };
  }
  const record = exactBody(body, { decision: 'string', feedback: 'string' }, ['decision']);
  if (record.decision !== 'approved' && record.decision !== 'rejected') {
    throw bad('Field "decision" must be "approved" or "rejected".');
  }
  const answer: InteractionAnswerInput = { decision: record.decision };
  if (record.feedback !== undefined) {
    const feedback = record.feedback as string;
    if (feedback.length > MAX_FEEDBACK_CHARS) throw bad('Field "feedback" is too long.');
    answer.feedback = feedback;
  }
  return answer;
}

export type PatchConversationInput = { title: unknown } | { modelId: string } | { thinking: ThinkingLevel };

function checkThinking(value: unknown): ThinkingLevel {
  if (!isThinkingLevel(value)) throw bad(`Field "thinking" must be one of ${THINKING_LEVELS.join(', ')}.`);
  return value;
}

/** Exactly one of `{title}`, `{modelId}` or `{thinking}`. The title is checked by the rename rule of the session writer. */
export function patchConversationInput(body: unknown): PatchConversationInput {
  if (typeof body === 'object' && body !== null && Object.hasOwn(body, 'modelId')) {
    const record = exactBody(body, { modelId: 'string' }, ['modelId']);
    return { modelId: checkModelId(record.modelId as string) };
  }
  if (typeof body === 'object' && body !== null && Object.hasOwn(body, 'thinking')) {
    const record = exactBody(body, { thinking: 'string' }, ['thinking']);
    return { thinking: checkThinking(record.thinking) };
  }
  const record = exactBody(body, { title: 'string' }, ['title']);
  return { title: record.title };
}

/**
 * `{mode, up}` exactly: the permission mode of a Work conversation, one of `ask`, `auto`, `bypass`, and whether the
 * page sends it as a move up. Only the page's confirm sends `up: true`; without it no request moves the mode up.
 */
/** `temp`: the temp folder the confirm named, at most 4096 characters. */
export function permissionModeInput(body: unknown): { mode: PermissionMode; up: boolean; temp: string | null } {
  const record = exactBody(body, { mode: 'string', up: 'boolean', temp: 'string' }, ['mode', 'up']);
  const mode = record.mode;
  if (mode !== 'ask' && mode !== 'auto' && mode !== 'bypass') {
    throw bad('Field "mode" must be "ask", "auto" or "bypass".');
  }
  const temp = record.temp === undefined ? null : (record.temp as string);
  if (temp !== null && (temp === '' || temp.length > 4096)) throw bad('Field "temp" must be a path.');
  return { mode, up: record.up as boolean, temp };
}

export interface DraftPermissionModeInput {
  mode: PermissionMode;
  up: boolean;
  /** The place the confirm named, required with `up: true`: a folder, or null for no folder. Null with `up: false`. */
  cwd: string | null;
  temp: string | null;
}

/**
 * `{mode, up, cwd?, noFolder?, temp?}` exactly: the mode picked on the New Work screen. A move up names its folder or
 * `noFolder: true`, never both; a move down names neither.
 */
export function draftPermissionModeInput(body: unknown): DraftPermissionModeInput {
  const record = exactBody(
    body,
    { mode: 'string', up: 'boolean', cwd: 'string', noFolder: 'boolean', temp: 'string' },
    ['mode', 'up'],
  );
  const { mode, up, temp } = permissionModeInput({
    mode: record.mode,
    up: record.up,
    ...(record.temp === undefined ? {} : { temp: record.temp }),
  });
  const cwd = record.cwd === undefined ? null : (record.cwd as string);
  const none = record.noFolder !== undefined;
  if (none && record.noFolder !== true) throw bad('Field "noFolder" must be true.');
  if (!up && (cwd !== null || none || temp !== null)) throw bad('A move down names no folder.');
  if (up && cwd !== null && none) throw bad(PLACE_TWICE);
  if (up && !none && (cwd === null || cwd === '' || cwd.length > MAX_CWD_CHARS)) {
    throw bad('Field "cwd" must be a path.');
  }
  return { mode, up, cwd, temp };
}

function folderQuery(query: URLSearchParams): string {
  const cwd = query.get('cwd');
  if (cwd === null || cwd === '' || cwd.length > MAX_CWD_CHARS) throw chatRouteError('cwd-invalid');
  return cwd;
}

/** `?cwd=<path>` or `?noFolder=1` (null): the place whose hooks the draft confirm names. */
export function workHooksQuery(query: URLSearchParams): string | null {
  const none = query.get('noFolder');
  if (none === null) return folderQuery(query);
  if (none !== '1') throw bad('Query "noFolder" must be 1.');
  if (query.has('cwd')) throw bad(PLACE_TWICE);
  return null;
}

/** `?cwd=<path>`: the folder whose branch the New screen's tray shows. */
export function workGitQuery(query: URLSearchParams): string {
  return folderQuery(query);
}

const MAX_BRANCH_CHARS = 1_024;

/** `POST /api/chat/work/git/switch`: exactly `{cwd, branch}`. */
export function workSwitchInput(body: unknown): { cwd: string; branch: string } {
  const record = exactBody(body, { cwd: 'string', branch: 'string' }, ['cwd', 'branch']);
  const cwd = record.cwd as string;
  const branch = record.branch as string;
  if (cwd === '' || cwd.length > MAX_CWD_CHARS) throw chatRouteError('cwd-invalid');
  if (branch === '' || branch.length > MAX_BRANCH_CHARS) throw bad('Field "branch" must be 1 to 1,024 characters.');
  return { cwd, branch };
}

export interface AgentActionInput {
  action: 'restart' | 'install';
  mode?: ChatMode;
}

export function agentActionInput(body: unknown): AgentActionInput {
  const record = exactBody(body, { action: 'string', mode: 'string' }, ['action']);
  if (record.action !== 'restart' && record.action !== 'install') {
    throw bad('Field "action" must be "restart" or "install".');
  }
  const input: AgentActionInput = { action: record.action };
  if (record.mode !== undefined) input.mode = checkMode(record.mode);
  return input;
}

/** `?mode=chat|work`, required. */
export function modeQuery(query: URLSearchParams): ChatMode {
  return checkMode(query.get('mode'));
}

/** A route `:id` / `:iid`: a session or interaction id shape, else it names nothing this service knows. */
export function idParam(value: string | undefined): string {
  if (value === undefined || !isIdShape(value)) throw ApiError.notFound('No such conversation.');
  return value;
}

/** A queued row id from the path; another shape is 404 like a row that is gone. */
export function queueRowParam(value: string | undefined): string {
  if (value === undefined || !isIdShape(value)) throw ApiError.notFound('No such queued message.');
  return value;
}

/** A background task id from the path; another shape is 404 like a task that is not live. */
export function taskParam(value: string | undefined): string {
  if (value === undefined || !isIdShape(value)) throw ApiError.notFound('No such task.');
  return value;
}

/** A subagent's child session id from the path; another shape is 404 like a subagent that ended. */
export function subagentParam(value: string | undefined): string {
  if (value === undefined || !isIdShape(value)) throw ApiError.notFound('No such subagent.');
  return value;
}

/** A schedule id from the path; another shape is 404 like a schedule that is gone. */
export function scheduleParam(value: string | undefined): string {
  if (value === undefined || !isIdShape(value)) throw ApiError.notFound('No such schedule.');
  return value;
}

/** A never-allow rule id from the path; another shape is 404 like a rule that is gone. */
export function ruleParam(value: string | undefined): string {
  if (value === undefined || !isIdShape(value)) throw ApiError.notFound('No such rule.');
  return value;
}

/** A workflow run id from the path; another shape is 404 like an unknown run. */
export function runParam(value: string | undefined): string {
  if (value === undefined || !isIdShape(value)) throw ApiError.notFound('No such run.');
  return value;
}

function isIdShape(value: string): boolean {
  if (value === '' || value.length > MAX_ID_CHARS) return false;
  for (let i = 0; i < value.length; i++) {
    const c = value.charCodeAt(i);
    const ok =
      (c >= 0x30 && c <= 0x39) || (c >= 0x41 && c <= 0x5a) || (c >= 0x61 && c <= 0x7a) || c === 0x2d || c === 0x5f;
    if (!ok) return false;
  }
  return true;
}

/** A route with no body: anything sent is refused. */
export function noBody(body: unknown): void {
  if (body === null || body === undefined) return;
  if (typeof body === 'object' && !Array.isArray(body) && Object.keys(body).length === 0) return;
  throw bad('This request takes no body.');
}

/** A provider route `:id`: one of the four providers, else it names nothing. */
export function providerIdParam(value: string | undefined): ProviderId {
  if (!isProviderId(value)) throw ApiError.notFound('No such provider.');
  return value;
}

/** `{key}`: the format is checked by the providers service, and no error text ever quotes it. */
export function providerKeyInput(body: unknown): { key: string } {
  const record = exactBody(body, { key: 'string' }, ['key']);
  return { key: record.key as string };
}

/** `{search?, firecrawl?}`, at least one; `search` is a search provider id or null. */
/** `PUT /api/chat/work/settings`: exactly `{claudeCode: boolean}`. */
export function workSettingsInput(body: unknown): WorkSettings {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) throw bad('The body must be an object.');
  const record = body as Record<string, unknown>;
  for (const key of Object.keys(record)) {
    if (key !== 'claudeCode') throw bad(`Unknown field "${fieldName(key)}".`);
  }
  if (typeof record.claudeCode !== 'boolean') throw bad('Field "claudeCode" must be true or false.');
  return { claudeCode: record.claudeCode };
}

export function providerSelectionInput(body: unknown): ProviderSelectionInput {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) throw bad('The body must be an object.');
  const record = body as Record<string, unknown>;
  for (const key of Object.keys(record)) {
    if (key !== 'search' && key !== 'firecrawl') throw bad(`Unknown field "${fieldName(key)}".`);
  }
  const input: ProviderSelectionInput = {};
  if (Object.hasOwn(record, 'search')) {
    if (record.search !== null && !isSearchProviderId(record.search)) {
      throw bad('Field "search" must be "exa", "deepseek", "brave" or null.');
    }
    input.search = record.search;
  }
  if (Object.hasOwn(record, 'firecrawl')) {
    if (typeof record.firecrawl !== 'boolean') throw bad('Field "firecrawl" has the wrong type.');
    input.firecrawl = record.firecrawl;
  }
  if (input.search === undefined && input.firecrawl === undefined)
    throw bad('Field "search" or "firecrawl" is required.');
  return input;
}
