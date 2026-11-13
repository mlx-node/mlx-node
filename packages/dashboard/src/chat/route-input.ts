/**
 * The input law of the Chat routes: every body is checked against its exact field list, and nothing from the page
 * is passed through. Only the message text (and, in Work, the folder) reach the agent, and the text never starts
 * with `/`: the agent runs such text as a slash command (`/yolo` turns every permission prompt off).
 */

import { ApiError } from '../api/errors.js';
import { isProviderId, isSearchProviderId } from './providers/settings.js';
import type { ProviderId, ProviderSelectionInput } from './providers/types.js';
import { chatRouteError } from './route-errors.js';
import type { ChatMode, PermissionMode } from './types.js';

export const MAX_TEXT_CHARS = 100_000;
const MAX_ID_CHARS = 128;
const MAX_MODEL_ID_CHARS = 512;
export const MAX_FEEDBACK_CHARS = 4_000;
const MAX_CWD_CHARS = 4_096;

/** Unicode `White_Space` (what Rust's `trim` skips) plus U+FEFF (what JavaScript's `trim` also skips). */
function isTrimmed(code: number): boolean {
  return (
    (code >= 0x09 && code <= 0x0d) ||
    code === 0x20 ||
    code === 0x85 ||
    code === 0xa0 ||
    code === 0x1680 ||
    (code >= 0x2000 && code <= 0x200a) ||
    code === 0x2028 ||
    code === 0x2029 ||
    code === 0x202f ||
    code === 0x205f ||
    code === 0x3000 ||
    code === 0xfeff
  );
}

/** Whether the first character after leading white space is `/`. */
export function isSlashText(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (isTrimmed(code)) continue;
    return code === 0x2f;
  }
  return false;
}

function bad(message: string): ApiError {
  return ApiError.badRequest(message);
}

type FieldKind = 'string' | 'boolean';

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
    if (typeof record[key] !== kind) throw bad(`Field "${fieldName(key)}" has the wrong type.`);
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

/** Message text: 1 to 100,000 characters, not blank, not slash text. */
export function checkText(text: string): string {
  if (text.trim() === '') throw bad('The message is empty.');
  if (text.length > MAX_TEXT_CHARS) throw new ApiError('E_PAYLOAD_TOO_LARGE', 'Too long. 100,000 characters at most.');
  if (isSlashText(text)) throw chatRouteError('slash-text');
  return text;
}

function checkMode(value: unknown): ChatMode {
  if (value !== 'chat' && value !== 'work') throw bad('Field "mode" must be "chat" or "work".');
  return value;
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
  /** Work: the New screen's draft id. The mode picked for it is the Control Panel's record, never the page's word. */
  draft?: string;
}

export function createConversationInput(body: unknown): CreateConversationInput {
  const record = exactBody(
    body,
    { mode: 'string', text: 'string', modelId: 'string', cwd: 'string', draft: 'string' },
    ['mode', 'text'],
  );
  const mode = checkMode(record.mode);
  if (record.cwd !== undefined && mode !== 'work') throw bad('Field "cwd" is only accepted in Work mode.');
  if (record.draft !== undefined && mode !== 'work') throw bad('Field "draft" is only accepted in Work mode.');
  const input: CreateConversationInput = { mode, text: checkText(record.text as string) };
  if (record.modelId !== undefined) input.modelId = checkModelId(record.modelId as string);
  if (record.cwd !== undefined) {
    const cwd = record.cwd as string;
    if (cwd === '' || cwd.length > MAX_CWD_CHARS) throw chatRouteError('cwd-invalid');
    input.cwd = cwd;
  }
  if (record.draft !== undefined) {
    const draft = record.draft as string;
    if (!isIdShape(draft)) throw bad('Field "draft" is not a draft id.');
    input.draft = draft;
  }
  return input;
}

export function promptInput(body: unknown): { text: string } {
  const record = exactBody(body, { text: 'string' }, ['text']);
  return { text: checkText(record.text as string) };
}

export type InteractionAnswerInput = { optionId: string } | { decision: 'approved' | 'rejected'; feedback?: string };

export function interactionInput(body: unknown): InteractionAnswerInput {
  if (typeof body === 'object' && body !== null && Object.hasOwn(body, 'optionId')) {
    const record = exactBody(body, { optionId: 'string' }, ['optionId']);
    const optionId = record.optionId as string;
    if (optionId === '' || optionId.length > MAX_ID_CHARS) throw bad('Field "optionId" is not an option id.');
    return { optionId };
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

export type PatchConversationInput = { title: unknown } | { modelId: string };

/** Exactly one of `{title}` or `{modelId}`. The title is checked by the rename rule of the session writer. */
export function patchConversationInput(body: unknown): PatchConversationInput {
  if (typeof body === 'object' && body !== null && Object.hasOwn(body, 'modelId')) {
    const record = exactBody(body, { modelId: 'string' }, ['modelId']);
    return { modelId: checkModelId(record.modelId as string) };
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
  /** The folder the confirm named: required with `up: true`, refused with `up: false`. */
  cwd: string | null;
  temp: string | null;
}

/** `{mode, up, cwd?, temp?}` exactly: the mode picked on the New Work screen. A move up names its folder. */
export function draftPermissionModeInput(body: unknown): DraftPermissionModeInput {
  const record = exactBody(body, { mode: 'string', up: 'boolean', cwd: 'string', temp: 'string' }, ['mode', 'up']);
  const { mode, up, temp } = permissionModeInput({
    mode: record.mode,
    up: record.up,
    ...(record.temp === undefined ? {} : { temp: record.temp }),
  });
  const cwd = record.cwd === undefined ? null : (record.cwd as string);
  if (!up && (cwd !== null || temp !== null)) throw bad('A move down names no folder.');
  if (up && (cwd === null || cwd === '' || cwd.length > MAX_CWD_CHARS)) throw bad('Field "cwd" must be a path.');
  return { mode, up, cwd, temp };
}

/** `?cwd=<path>`: the folder whose hooks the draft confirm names. */
export function workHooksQuery(query: URLSearchParams): string {
  const cwd = query.get('cwd');
  if (cwd === null || cwd === '' || cwd.length > MAX_CWD_CHARS) throw chatRouteError('cwd-invalid');
  return cwd;
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
