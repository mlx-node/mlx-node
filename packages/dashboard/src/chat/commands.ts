/**
 * Slash commands: the agent's parse of a typed `/name args`, and the app's static allow rule over every name the
 * agent runs. Two classes are decided by name before anything is sent. App commands run in the app (the page calls
 * the history op) and are never prompt text. Agent commands are sent as typed and expanded by the agent: in Work the
 * allowed built-ins the session's list offers and the user's listed skills and workflows; in Chat none. Every other name is refused, so a typed
 * `/yolo` (every permission request off) never reaches the agent. Pure: read by the host and by the page.
 */

import type { ChatMode, CommandView } from './types.js';

/** What Rust's `char::is_whitespace` (Unicode `White_Space`) holds: the agent trims and splits the name on these. */
function isAgentSpace(code: number): boolean {
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
    code === 0x3000
  );
}

/**
 * The agent's slash parse (`slash_authority.rs` `parse_slash_prefix`): the text trimmed, a leading `/`, the name up to
 * the first white space with no `/` or `\` in it, the rest trimmed as the words. The name lower-cased, as the agent
 * matches it. Null when the agent would send the text to the model as it is.
 */
export function parseSlash(text: string): { name: string; args: string } | null {
  let start = 0;
  let end = text.length;
  while (start < end && isAgentSpace(text.charCodeAt(start))) start += 1;
  while (end > start && isAgentSpace(text.charCodeAt(end - 1))) end -= 1;
  if (start >= end || text.charCodeAt(start) !== 0x2f) return null;
  const from = start + 1;
  let stop = from;
  while (stop < end && !isAgentSpace(text.charCodeAt(stop))) stop += 1;
  const name = text.slice(from, stop);
  if (name === '' || name.includes('/') || name.includes('\\')) return null;
  let args = stop;
  while (args < end && isAgentSpace(text.charCodeAt(args))) args += 1;
  return { name: name.toLowerCase(), args: text.slice(args, end) };
}

/**
 * Every built-in of the agent (`BUILTIN_COMMANDS` and `PROMPT_COMMANDS`, `session/slash_commands.rs`, agent 0.3.4)
 * with its aliases. Each name and alias is in exactly one class below; a built-in the agent adds must be classed here
 * before it can run.
 */
export const AGENT_BUILTINS: readonly { name: string; aliases: readonly string[] }[] = [
  { name: 'compact', aliases: [] },
  { name: 'always-approve', aliases: ['yolo'] },
  { name: 'flush', aliases: [] },
  { name: 'dream', aliases: [] },
  { name: 'memory', aliases: ['mem'] },
  { name: 'context', aliases: [] },
  { name: 'hooks-trust', aliases: [] },
  { name: 'hooks-list', aliases: [] },
  { name: 'hooks-add', aliases: [] },
  { name: 'hooks-remove', aliases: [] },
  { name: 'hooks-untrust', aliases: [] },
  { name: 'plugins', aliases: ['plugin'] },
  { name: 'reload-plugins', aliases: [] },
  { name: 'session-info', aliases: ['status', 'info'] },
  { name: 'feedback', aliases: [] },
  { name: 'deep-research', aliases: [] },
  { name: 'workflow', aliases: [] },
  { name: 'goal', aliases: [] },
  { name: 'loop', aliases: [] },
];

/** The history ops a typed command runs in the app. */
export type AppCommand = 'compact' | 'context' | 'fork' | 'plan' | 'recap' | 'rewind';

/** App commands and the other names they are typed as: run by the page, refused as prompt text in both modes. */
export const APP_COMMANDS: Readonly<Record<AppCommand, readonly string[]>> = {
  compact: [],
  context: ['usage', 'cost'],
  fork: [],
  plan: [],
  recap: [],
  rewind: ['undo'],
};

/**
 * Refused in both modes: they change what runs outside the permission cards (`always-approve` turns every request off;
 * hooks run shell; a plugin can bring hooks), or have no screen here (memory, feedback).
 */
export const BLOCKED: ReadonlySet<string> = new Set([
  'always-approve',
  'yolo',
  'hooks-trust',
  'hooks-list',
  'hooks-add',
  'hooks-remove',
  'hooks-untrust',
  'plugins',
  'plugin',
  'reload-plugins',
  'memory',
  'mem',
  'flush',
  'dream',
  'feedback',
]);

/** Refused and never listed: Inspect shows the ids and paths it prints. */
export const HIDDEN: ReadonlySet<string> = new Set(['session-info', 'status', 'info']);

/** The agent's built-ins each mode sends as typed. */
export const ALLOWED_BUILTINS: Readonly<Record<ChatMode, readonly string[]>> = {
  chat: [],
  work: ['goal', 'loop', 'deep-research', 'workflow'],
};

/** The app command a typed name runs, or null. */
export function appCommandOf(name: string): AppCommand | null {
  for (const [command, aliases] of Object.entries(APP_COMMANDS) as Array<[AppCommand, readonly string[]]>) {
    if (name === command || aliases.includes(name)) return command;
  }
  return null;
}

/** Refused whatever the mode or the agent's list says: an app command, a blocked or a hidden built-in. */
export function refusedName(name: string): boolean {
  return appCommandOf(name) !== null || BLOCKED.has(name) || HIDDEN.has(name);
}

/** A row of the agent's command list as the wire has it (`AvailableCommand`), before it is classed. */
export interface CommandRow {
  name: string;
  description: string;
  /** `input.hint`, or null. */
  hint: string | null;
  /** `_meta.workflowSource` is set: a workflow row (also on a built-in that a workflow of its name projects). */
  workflow: boolean;
  /** `_meta.workflowPath`: the file a workflow row runs, or null. */
  workflowPath: string | null;
  /** `_meta.bareName` or `_meta.scope` is set: a skill row. */
  skill: boolean;
  /** A skill's `_meta.qualifiedName`: the agent runs the skill typed this way too. */
  qualifiedName: string | null;
}

/** A row of the agent's list, classed. Names lower-cased, as the agent matches them. */
export interface AgentCommand {
  name: string;
  kind: 'builtin' | 'skill' | 'workflow' | 'other';
  description: string;
  hint: string | null;
  qualifiedName: string | null;
}

const BUILTIN_NAMES: ReadonlySet<string> = new Set(AGENT_BUILTINS.flatMap((b) => [b.name, ...b.aliases]));

/** The agent's rows classed: a built-in by name first, then a workflow, then a skill by its `_meta`. */
export function readCommands(rows: readonly CommandRow[]): AgentCommand[] {
  return rows.map((row) => {
    const name = row.name.toLowerCase();
    const kind = BUILTIN_NAMES.has(name) ? 'builtin' : row.workflow ? 'workflow' : row.skill ? 'skill' : 'other';
    return {
      name,
      kind,
      description: row.description,
      hint: row.hint,
      qualifiedName: row.qualifiedName === null ? null : row.qualifiedName.toLowerCase(),
    };
  });
}

/** A workflow file the app wrote into the agent home: the name its script holds, and the paths the agent may list. */
export interface AppWorkflow {
  name: string;
  paths: readonly string[];
}

/**
 * The agent's rows less the app's own workflow: typed as `/name` it would start with no permission card and none of
 * the app's checks, so it is neither listed nor sent. A row of its name from another file stays; one with no path goes.
 */
export function withoutAppWorkflow(rows: readonly CommandRow[], app: AppWorkflow | null): CommandRow[] {
  if (app === null) return [...rows];
  return rows.filter(
    (row) =>
      !row.workflow ||
      row.name.toLowerCase() !== app.name ||
      (row.workflowPath !== null && !app.paths.includes(row.workflowPath)),
  );
}

/** A name the page can fill back as typed: printable, no white space, no slash, at most 64 characters. */
function plainName(name: string): boolean {
  if (name === '' || name.length > 64) return false;
  for (let i = 0; i < name.length; i++) {
    const code = name.charCodeAt(i);
    if (code <= 0x20 || (code >= 0x7f && code <= 0xa0) || code === 0x2f || code === 0x5c) return false;
    if (isAgentSpace(code) || (code >= 0x200b && code <= 0x200f) || (code >= 0x2060 && code <= 0x206f)) return false;
    if (code === 0xfeff || (code >= 0xd800 && code <= 0xdfff)) return false;
  }
  return true;
}

/**
 * The rows a mode may list and send: none in Chat; in Work the allowed built-ins the agent offers, then the user's
 * skills and workflows. A row named like a refused name never shows.
 */
export function allowedCommands(mode: ChatMode, rows: readonly AgentCommand[]): CommandView[] {
  if (mode === 'chat') return [];
  const out: CommandView[] = [];
  const seen = new Set<string>();
  for (const row of rows) {
    if (seen.has(row.name) || !plainName(row.name) || refusedName(row.name)) continue;
    const shown =
      row.kind === 'builtin'
        ? ALLOWED_BUILTINS.work.includes(row.name)
        : row.kind === 'skill' || row.kind === 'workflow';
    if (!shown) continue;
    seen.add(row.name);
    out.push({
      name: row.name,
      kind: row.kind === 'builtin' ? 'builtin' : row.kind === 'skill' ? 'skill' : 'workflow',
      description: row.description,
      hint: row.hint,
    });
  }
  return out;
}

/**
 * The allow rule for a message's text. `plain`: no slash command. `allowed`: send as typed. `refused`: `slash-text`.
 * `needs-list`: a Work name that only the session's list could allow (an allowed built-in the agent offers, a listed
 * skill or workflow), and that list is not known yet (a cold conversation; the host checks again once it is loaded).
 */
export function checkCommand(
  mode: ChatMode,
  text: string,
  rows: readonly AgentCommand[] | null,
): 'plain' | 'allowed' | 'refused' | 'needs-list' {
  if (!startsWithSlash(text)) return 'plain';
  const slash = parseSlash(text);
  if (slash === null || refusedName(slash.name)) return 'refused';
  const builtin = ALLOWED_BUILTINS[mode].includes(slash.name);
  if (!builtin && (mode === 'chat' || BUILTIN_NAMES.has(slash.name))) return 'refused';
  if (rows === null) return 'needs-list';
  const listed = rows.some((row) =>
    builtin
      ? row.kind === 'builtin' && row.name === slash.name
      : (row.kind === 'skill' && (row.name === slash.name || row.qualifiedName === slash.name)) ||
        (row.kind === 'workflow' && row.name === slash.name),
  );
  return listed ? 'allowed' : 'refused';
}

/**
 * Whether the first character after leading white space is `/`: Unicode `White_Space` plus U+FEFF, wider than the
 * agent's trim, so text that reads as a command is checked even where the agent would not run it.
 */
export function startsWithSlash(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (isAgentSpace(code) || code === 0xfeff) continue;
    return code === 0x2f;
  }
  return false;
}
