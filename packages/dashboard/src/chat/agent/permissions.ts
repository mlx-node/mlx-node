/**
 * Permission answers. A "reject" answer ends the agent's whole turn, so Chat approves every call that can only reach
 * the `mlx_web` server and rejects only what must stop the answer. Options are always chosen by exact `optionId`,
 * never by kind: the agent offers `enable-always-approve` with kind `allow_once`.
 */

import { posix, win32 } from 'node:path';

import type { PermissionRequest, PromptResultFacts } from '../acp/messages.js';
import { redactText } from '../redact.js';
import type { ChatMode, CommandFlag, PermissionInteraction, PlanInteraction, TurnOutcome } from '../types.js';
import { agentPath, commandFlags, landsOutside, pathFlags, type WorkPlace } from './command-flags.js';
import {
  ALLOW_EDITS_SESSION_OPTION_ID,
  ALLOW_ONCE_OPTION_ID,
  ALWAYS_ALLOW_OPTION_ID,
  ENABLE_ALWAYS_APPROVE_OPTION_ID,
  REJECT_ONCE_OPTION_ID,
  isWorkGuardHook,
  WEB_TOOL_KEYS,
} from './profiles.js';
import { isWinPath } from './roots.js';
import { readToolInputText, webInputKeys } from './tool-input.js';
import { visibleText } from './visible-text.js';

/** Approvals per Chat turn (web calls and wrong guesses together); the next request ends the turn as `tool-loop`. */
export const CHAT_PERMISSIONS_PER_TURN_MAX = 8;

export type PermissionAnswer = { outcome: 'selected'; optionId: string } | { outcome: 'cancelled' };

/** Why the host ended or will end a turn; mapped to the outcome when the prompt reply arrives. */
export type TurnEndCause =
  | 'cancelled'
  | 'tool-refused'
  | 'arguments-unreadable'
  | 'tool-loop'
  | 'turn-timeout'
  | 'inference-lost'
  | 'denied'
  /** The Control Panel closed during the turn. */
  | 'app-closed'
  /** The agent said it has no model it can use (`model_auto_switched` with no new id). */
  | 'no-model'
  /** Work: a call matched a never-allow rule once more than an answer may (`NEVER_PER_TURN_MAX`); it got a plain deny. */
  | 'never-loop';

/** The outcome a recorded cause ends the turn with. */
export function causeOutcome(cause: TurnEndCause): TurnOutcome {
  return cause === 'app-closed' ? 'interrupted' : cause;
}

type RefusalCause = 'tool-refused' | 'arguments-unreadable';

export type ChatPermissionDecision =
  | { kind: 'approve'; answer: PermissionAnswer; row: 6 | 7 }
  /** Answered `reject-once` (or `cancelled` when not offered); the agent ends the turn itself. */
  | {
      kind: 'reject';
      answer: PermissionAnswer;
      cause: RefusalCause;
      cancelTurn: boolean;
      row: 2 | 3 | 8;
    }
  /** Answered `cancelled`, then `session/cancel`. */
  | { kind: 'cancel'; answer: PermissionAnswer; cause: 'tool-refused' | 'tool-loop'; cancelTurn: true; row: 1 | 2 | 4 };

export interface ChatPermissionContext {
  /** The request names a Chat session of this host whose guard passed. */
  sessionGuarded: boolean;
  /** Approvals already given in this turn. */
  approvals: number;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const own = Object.keys(value);
  return own.length === keys.length && keys.every((k) => Object.hasOwn(value, k));
}

const CALL_KEYS = ['variant', 'tool_name', 'tool_input'] as const;

/**
 * What the tool receives after an allow: `object` is that object; `none` is no arguments (or `{}`), which the tool
 * answers with `bad-request`; `unreadable` is a text that could still become an object the app cannot read the same way.
 * Only `web_fetch` has a rule on its input (the `mlx_web` server checks its address against the ledger), so only its
 * `unreadable` text is refused; any other tool gets the same answer as for an object.
 */
export type WebToolInput =
  | { kind: 'object'; value: Record<string, unknown> }
  | { kind: 'none' }
  | { kind: 'unreadable' };

export interface WebCallShape {
  shape: 'use_tool' | 'direct';
  toolName: string;
  input: WebToolInput;
}

/** The agent sends a direct call's non-object input as no arguments, and `use_tool` reads a text again: see tool-input.ts. */
function toolInputOf(shape: WebCallShape['shape'], toolName: string, raw: unknown): WebToolInput {
  if (isPlainObject(raw)) return { kind: 'object', value: raw };
  if (shape === 'direct' || typeof raw !== 'string') return { kind: 'none' };
  const read = readToolInputText(raw, webInputKeys(toolName));
  if (read.kind === 'object') return read;
  return read.kind === 'not-object' ? { kind: 'none' } : { kind: 'unreadable' };
}

/**
 * Call shape A (through `use_tool`) or B (a direct `mlx_web__…` call) with what its tool receives, or null for anything
 * else: a shell or edit request, any non-MCP tool, the `tool_input_file` / `file` forms, any extra key.
 */
export function webCallShape(req: PermissionRequest): WebCallShape | null {
  const raw = req.rawInput;
  if (req.tool === null || !isPlainObject(raw) || !hasExactKeys(raw, CALL_KEYS)) return null;
  if (typeof raw.tool_name !== 'string') return null;
  if (req.tool.name === 'use_tool' && raw.variant === 'UseTool') {
    return {
      shape: 'use_tool',
      toolName: raw.tool_name,
      input: toolInputOf('use_tool', raw.tool_name, raw.tool_input),
    };
  }
  const direct = req.tool.name;
  if (
    direct !== null &&
    (WEB_TOOL_KEYS as readonly string[]).includes(direct) &&
    req.tool.namespace === 'mcp' &&
    raw.variant === 'MCPTool' &&
    raw.tool_name === direct
  ) {
    return { shape: 'direct', toolName: raw.tool_name, input: toolInputOf('direct', raw.tool_name, raw.tool_input) };
  }
  return null;
}

const selected = (optionId: string): PermissionAnswer => ({ outcome: 'selected', optionId });
const CANCELLED: PermissionAnswer = { outcome: 'cancelled' };

/** The Chat decision ladder: first matching row wins. */
export function decideChatPermission(req: PermissionRequest, ctx: ChatPermissionContext): ChatPermissionDecision {
  if (!ctx.sessionGuarded)
    return { kind: 'cancel', answer: CANCELLED, cause: 'tool-refused', cancelTurn: true, row: 1 };
  const ids = req.options.map((o) => o.optionId);
  const canReject = ids.includes(REJECT_ONCE_OPTION_ID);
  const refuse = (cause: RefusalCause, row: 2 | 3 | 8): ChatPermissionDecision =>
    canReject
      ? { kind: 'reject', answer: selected(REJECT_ONCE_OPTION_ID), cause, cancelTurn: false, row }
      : { kind: 'reject', answer: CANCELLED, cause, cancelTurn: true, row };
  if (ids.includes(ENABLE_ALWAYS_APPROVE_OPTION_ID) || !ids.includes(ALLOW_ONCE_OPTION_ID)) {
    return canReject
      ? { kind: 'reject', answer: selected(REJECT_ONCE_OPTION_ID), cause: 'tool-refused', cancelTurn: false, row: 2 }
      : { kind: 'cancel', answer: CANCELLED, cause: 'tool-refused', cancelTurn: true, row: 2 };
  }
  const call = webCallShape(req);
  if (call === null) return refuse('tool-refused', 3);
  if (ctx.approvals >= CHAT_PERMISSIONS_PER_TURN_MAX) {
    return { kind: 'cancel', answer: CANCELLED, cause: 'tool-loop', cancelTurn: true, row: 4 };
  }
  if (call.toolName === 'mlx_web__web_fetch' && call.input.kind === 'unreadable') {
    return refuse('arguments-unreadable', 8);
  }
  if ((WEB_TOOL_KEYS as readonly string[]).includes(call.toolName)) {
    return { kind: 'approve', answer: selected(ALLOW_ONCE_OPTION_ID), row: 6 };
  }
  return { kind: 'approve', answer: selected(ALLOW_ONCE_OPTION_ID), row: 7 };
}

const TITLE_MAX = 200;
const OPTION_NAME_MAX = 120;
/**
 * A command, path or tool input longer than this is not shown: the request is denied, since a card could not show it
 * whole.
 */
export const SHOWN_COMMAND_MAX = 100_000;
const PLAN_MAX = 50_000;

/** Work options the page may offer, by id, with the kind each must have. Every other option is dropped. */
const WORK_OPTIONS: ReadonlyArray<{ optionId: string; kind: PermissionInteraction['options'][number]['kind'] }> = [
  { optionId: REJECT_ONCE_OPTION_ID, kind: 'reject_once' },
  { optionId: ALLOW_ONCE_OPTION_ID, kind: 'allow_once' },
  { optionId: ALWAYS_ALLOW_OPTION_ID, kind: 'allow_always' },
  { optionId: ALLOW_EDITS_SESSION_OPTION_ID, kind: 'allow_always' },
];

function cut(text: string, max: number): string {
  if (text.length <= max) return text;
  let end = max;
  const c = text.charCodeAt(end - 1);
  if (c >= 0xd800 && c <= 0xdbff) end -= 1;
  return text.slice(0, end);
}

export function accessOf(req: PermissionRequest): PermissionInteraction['access'] {
  const name = req.tool?.name ?? null;
  if (name === 'run_terminal_command') return 'bash';
  if (name === 'search_replace' || name === 'write') return 'edit';
  if (name === 'use_tool' || req.tool?.namespace === 'mcp') return 'mcp';
  return 'other';
}

/** Whether an offered option may be shown for this access kind: "always" only as each kind's own grant. */
function offered(optionId: string, access: PermissionInteraction['access']): boolean {
  if (optionId === ALWAYS_ALLOW_OPTION_ID) return access !== 'edit';
  if (optionId === ALLOW_EDITS_SESSION_OPTION_ID) return access === 'edit';
  return true;
}

function stringField(raw: unknown, key: string): string | null {
  if (!isPlainObject(raw)) return null;
  const value = raw[key];
  return typeof value === 'string' ? value : null;
}

/** The full command of a `bash` request, or the full path of an `edit` request, as it will reach the shell or disk. */
export function requestTarget(req: PermissionRequest): { command: string | null; path: string | null } {
  const access = accessOf(req);
  return {
    command: access === 'bash' ? stringField(req.rawInput, 'command') : null,
    path: access === 'edit' ? (stringField(req.rawInput, 'file_path') ?? stringField(req.rawInput, 'path')) : null,
  };
}

/** The tools that run their input's `command` in a shell: the ones the Work guard asks for (`WORK_GUARD_MATCHER`). */
const SHELL_TOOLS: readonly string[] = ['run_terminal_command', 'monitor'];

/**
 * The shell command a request runs, as it will reach the shell: `text` is null when its `command` is not text. Null
 * for a tool that runs no shell command.
 */
export function shellCommand(req: PermissionRequest): { text: string | null } | null {
  const name = req.tool?.name ?? null;
  if (name === null || !SHELL_TOOLS.includes(name)) return null;
  return { text: stringField(req.rawInput, 'command') };
}

function inputText(req: PermissionRequest): string {
  try {
    return req.rawInput === undefined ? '' : (JSON.stringify(req.rawInput, null, 2) ?? '');
  } catch {
    return '';
  }
}

/**
 * False when the command, path or shown input (every kind but a command, whose card shows the command) is too long to
 * show whole: such a request is denied, never shown in part.
 */
export function showableWhole(req: PermissionRequest): boolean {
  const target = requestTarget(req);
  return (
    (target.command?.length ?? 0) <= SHOWN_COMMAND_MAX &&
    (target.path?.length ?? 0) <= SHOWN_COMMAND_MAX &&
    (accessOf(req) === 'bash' || inputText(req).length <= SHOWN_COMMAND_MAX)
  );
}

/** The request title without the line the agent appends for the app's own guard hook; any other hook's line stays. */
function cardTitle(req: PermissionRequest): string {
  if (req.hookAsk === undefined || !isWorkGuardHook(req.hookAsk.hookName) || req.hookAsk.reason !== null) {
    return req.title;
  }
  const line = ` — hook '${req.hookAsk.hookName}' asks for confirmation`;
  return req.title.endsWith(line) ? req.title.slice(0, req.title.length - line.length) : req.title;
}

/** Keys are removed before the text is cut, so a key that crosses the cut can never show in part. */
export function shownCut(text: string, max: number, needles: readonly string[]): string {
  return visibleText(cut(redactText(text, needles), max));
}

/**
 * A Work permission request as the page sees it. Only the options of `WORK_OPTIONS` pass, so `enable-always-approve`, every
 * `reject_always` and any other "always" grant are never offered. `place` adds the "outside the work folder" check;
 * `landing` is where an edit really lands once the disk follows its links, which decides that check. A landing that
 * cannot be known (null) is warned about as outside the folder. `from`: the helper that asks, the name of its run, its
 * folder when that is not the Work folder, and its session. The host stamps `openedAt` when it opens the card.
 */
export function workInteraction(
  id: string,
  req: PermissionRequest,
  needles: readonly string[],
  place?: WorkPlace,
  landing?: string | null,
  from?: NonNullable<PermissionInteraction['from']>,
): Omit<PermissionInteraction, 'openedAt'> {
  const access = accessOf(req);
  const options: PermissionInteraction['options'] = [];
  for (const o of req.options) {
    const allowed = WORK_OPTIONS.find((w) => w.optionId === o.optionId);
    if (allowed === undefined || allowed.kind !== o.kind || !offered(o.optionId, access)) continue;
    if (options.some((x) => x.optionId === o.optionId)) continue;
    options.push({
      optionId: o.optionId,
      kind: allowed.kind,
      name: shownCut(o.name, OPTION_NAME_MAX, needles),
    });
  }
  const target = requestTarget(req);
  let flags: CommandFlag[] = [];
  let resolvedPath: string | null = null;
  if (target.command !== null) flags = commandFlags(target.command, place);
  else if (target.path !== null && place !== undefined) {
    if (landing === undefined) flags = pathFlags(target.path, place);
    else if (landing === null) flags = ['outside-folder'];
    else {
      flags = landsOutside(landing, place) ? ['outside-folder'] : [];
      const normalize = isWinPath(place.cwd) ? win32.normalize : posix.normalize;
      if (landing !== normalize(agentPath(target.path, place))) resolvedPath = landing;
    }
  }
  const shown = (text: string | null): string | null => (text === null ? null : visibleText(redactText(text, needles)));
  const unknown = target.command === null && target.path !== null && place !== undefined && landing === null;
  const title = cardTitle(req);
  return {
    ...(unknown ? { landingUnknown: true as const } : {}),
    ...(redactText(title, needles).length > TITLE_MAX ? { titleCut: true as const } : {}),
    id,
    kind: 'permission',
    toolCallId: req.toolCallId,
    toolName: shownCut(req.tool?.name ?? req.kind ?? 'tool', TITLE_MAX, needles),
    access,
    title: shownCut(title, TITLE_MAX, needles),
    input: shown(inputText(req)) ?? '',
    command: shown(target.command),
    path: shown(target.path),
    resolvedPath: shown(resolvedPath),
    options,
    flags,
    ...(from === undefined
      ? {}
      : {
          from: {
            label: shownCut(from.label, TITLE_MAX, needles),
            run: from.run === null ? null : shownCut(from.run, TITLE_MAX, needles),
            cwd: shown(from.cwd),
            child: from.child,
          },
        }),
  };
}

/** An `_x.ai/exit_plan_mode` request: the session, the tool call and the plan text. */
export interface PlanRequest {
  sessionId: string;
  toolCallId: string;
  plan: string;
}

export function readPlanRequest(params: unknown): PlanRequest | null {
  if (!isPlainObject(params)) return null;
  const { sessionId, toolCallId, planContent } = params;
  if (typeof sessionId !== 'string' || sessionId === '' || typeof toolCallId !== 'string') return null;
  if (planContent !== undefined && planContent !== null && typeof planContent !== 'string') return null;
  return { sessionId, toolCallId, plan: typeof planContent === 'string' ? planContent : '' };
}

export function planInteraction(
  id: string,
  req: PlanRequest,
  needles: readonly string[],
): Omit<PlanInteraction, 'openedAt'> {
  return {
    id,
    kind: 'plan',
    toolCallId: shownCut(req.toolCallId, TITLE_MAX, needles),
    plan: shownCut(req.plan, PLAN_MAX, needles),
  };
}

/** The agent's reply to a plan: approved, or cancelled with the user's feedback (the plan stays readable). */
export function planAnswer(decision: 'approved' | 'rejected', feedback?: string): Record<string, string> {
  if (decision === 'approved') return { outcome: 'approved' };
  return feedback === undefined || feedback.trim() === ''
    ? { outcome: 'cancelled' }
    : { outcome: 'cancelled', feedback };
}

/** Unicode `White_Space` (what Rust's `trim` skips) plus U+FEFF: every text the agent reads as blank is blank here. */
function isSpace(code: number): boolean {
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

/**
 * The reply to a permission request denied with the user's note: deny once, the note as `_meta.followup_message`.
 * The agent then skips the call, adds the note as a user message and goes on. Null for a blank note: the agent
 * would read that as a plain deny and end the turn.
 */
export function noteAnswer(optionId: string, note: string): Record<string, unknown> | null {
  let blank = true;
  for (let i = 0; i < note.length && blank; i++) blank = isSpace(note.charCodeAt(i));
  if (blank) return null;
  return { outcome: { outcome: 'selected', optionId }, _meta: { followup_message: note } };
}

export interface TurnEnd {
  mode: ChatMode;
  /** The `session/prompt` result, when it answered. */
  result?: PromptResultFacts;
  /** The `session/prompt` error: its code and fixed `error_kind`. */
  error?: { code: number; errorKind: string | null };
  /** The process exited before the reply. */
  exited?: boolean;
  /** What the host recorded when it rejected a permission or sent `session/cancel`. */
  recorded: TurnEndCause | null;
}

/** The turn outcome from the prompt reply and what the host recorded. A rejected permission is never "cancelled". */
export function turnOutcome(end: TurnEnd): TurnOutcome {
  if (end.exited === true) return 'agent-crashed';
  if (end.error !== undefined) {
    return end.error.code === -32603 && end.error.errorKind === 'max_tokens_truncation'
      ? 'output-limit'
      : 'model-error';
  }
  // The agent answers a blocked prompt `end_turn` at once.
  if (end.recorded === 'no-model') return 'no-model';
  const stop = end.result?.stopReason ?? null;
  if (stop === 'cancelled') {
    if (end.result?.cancellationCategory === 'HookDenied' && end.recorded === null) return 'hook-stopped';
    if (end.result?.cancellationCategory === 'PermissionRejected') {
      if (end.recorded !== null && end.recorded !== 'cancelled' && end.recorded !== 'app-closed') return end.recorded;
      return end.mode === 'chat' ? 'tool-refused' : 'denied';
    }
    return end.recorded === null ? 'interrupted' : causeOutcome(end.recorded);
  }
  if (stop === 'max_tokens') return 'output-limit';
  if (stop === 'refusal') return 'refused';
  if (stop === 'max_turn_requests') return 'tool-loop';
  return 'completed';
}
