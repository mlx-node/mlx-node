/**
 * Work tool rows: the agent's input of a shell command, an edit, a read, a listing or a file search, read with a
 * shape check. An input that does not fit gives null, and the row falls back to the agent's title and the raw input.
 * The `workflow` calls have forms of their own ({@link workflowFormOf}).
 */

import { CHAT, fill } from './chat-strings';
import { truncateMiddle } from './labels';
import type { TranscriptEntry } from './types';
import { oneLine } from './web-tools';

type ToolEntry = Extract<TranscriptEntry, { kind: 'tool' }>;

export type WorkKind = 'run' | 'edit' | 'read' | 'list' | 'grep';

export interface WorkCall {
  kind: WorkKind;
  /** Running, done, failed. */
  words: readonly [string, string, string];
  /** The row title: a command's first line, a path, a quoted pattern. */
  title: string;
  /** `run`: the whole command. */
  command?: string;
  /** `edit`, `read`, `list`: the path as the agent wrote it. */
  path?: string;
  /** `edit`: the change as text lines (`-` removed, `+` added). */
  change?: string;
}

/** Characters of a command's first line in a row title. */
export const COMMAND_TITLE_CHARS = 80;

function record(input: string | undefined): Record<string, unknown> | null {
  if (input === undefined) return null;
  try {
    const value: unknown = JSON.parse(input);
    return typeof value === 'object' && value !== null && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function text(value: unknown): string | null {
  return typeof value === 'string' && value !== '' ? value : null;
}

function firstLine(value: string): string {
  const end = value.indexOf('\n');
  return oneLine(end === -1 ? value : value.slice(0, end));
}

function hidden(code: number): boolean {
  if (code === 0x09 || code === 0x0a) return false;
  if (code < 0x20 || (code >= 0x7f && code <= 0xa0)) return true;
  if (code === 0x00ad || code === 0x034f || code === 0x061c) return true;
  if (code === 0x115f || code === 0x1160 || code === 0x3164 || code === 0xffa0 || code === 0x2800) return true;
  if (code >= 0x180b && code <= 0x180f) return true;
  if (code >= 0x2000 && code <= 0x200f) return true;
  if (code >= 0x2028 && code <= 0x202f) return true;
  if (code >= 0x205f && code <= 0x206f) return true;
  if (code === 0x3000 || code === 0xfeff) return true;
  if (code >= 0xfe00 && code <= 0xfe0f) return true;
  if (code >= 0xfff9 && code <= 0xfffb) return true;
  if (code >= 0xe0000 && code <= 0xe007f) return true;
  return code >= 0xe0100 && code <= 0xe01ef;
}

/**
 * Model text in a code well: every character that can hide or reorder what is shown becomes its visible code
 * (`\u{202E}`); tab and new line stay. The same rule as the Control Panel's permission cards.
 */
export function visibleText(value: string): string {
  let out = '';
  let from = 0;
  for (let i = 0; i < value.length;) {
    const code = value.codePointAt(i)!;
    const size = code > 0xffff ? 2 : 1;
    if (hidden(code)) {
      out += `${value.slice(from, i)}\\u{${code.toString(16).toUpperCase().padStart(4, '0')}}`;
      from = i + size;
    }
    i += size;
  }
  return from === 0 ? value : out + value.slice(from);
}

/** `path` relative to `folder` when it lies inside it; else as written. */
export function relativeTo(path: string, folder: string | null): string {
  if (folder === null || folder === '' || !path.startsWith('/')) return path;
  let end = folder.length;
  while (end > 1 && folder[end - 1] === '/') end -= 1;
  const root = folder.slice(0, end);
  if (path === root) return '.';
  return path.startsWith(`${root}/`) ? path.slice(root.length + 1) : path;
}

function prefixed(value: string, mark: string): string {
  const lines = value.endsWith('\n') ? value.slice(0, -1).split('\n') : value.split('\n');
  return lines.map((line) => `${mark} ${line}`).join('\n');
}

/** An edit whose input was cut too long to read (a big write): its diff blocks still name the file. */
function cutEdit(entry: ToolEntry, folder: string | null): WorkCall | null {
  const path = entry.name === 'search_replace' || entry.name === 'write' ? text(entry.edits?.[0]?.path) : null;
  return path === null ? null : { kind: 'edit', words: CHAT.workEdit, title: oneLine(relativeTo(path, folder)), path };
}

/** The Work form of a tool row, or null for a tool this table does not know or an input that does not fit. */
export function workCallOf(entry: ToolEntry, folder: string | null): WorkCall | null {
  const input = record(entry.input);
  if (input === null) return cutEdit(entry, folder);
  switch (entry.name) {
    case 'run_terminal_command': {
      const command = text(input.command);
      if (command === null) return null;
      return {
        kind: 'run',
        words: CHAT.workRun,
        title: truncateMiddle(firstLine(command), COMMAND_TITLE_CHARS),
        command: visibleText(command),
      };
    }
    case 'search_replace':
    case 'write': {
      const path = text(input.file_path) ?? text(input.path);
      if (path === null) return null;
      const parts: string[] = [];
      if (entry.name === 'write') {
        if (typeof input.content === 'string') parts.push(prefixed(input.content, '+'));
      } else {
        if (typeof input.old_string === 'string') parts.push(prefixed(input.old_string, '-'));
        if (typeof input.new_string === 'string') parts.push(prefixed(input.new_string, '+'));
      }
      const call: WorkCall = { kind: 'edit', words: CHAT.workEdit, title: oneLine(relativeTo(path, folder)), path };
      if (parts.length > 0) call.change = visibleText(parts.join('\n'));
      return call;
    }
    case 'read_file': {
      const path = text(input.path) ?? text(input.file_path) ?? text(input.target_file);
      return path === null
        ? null
        : { kind: 'read', words: CHAT.workRead, title: oneLine(relativeTo(path, folder)), path };
    }
    case 'list_dir': {
      const path = text(input.target_directory) ?? text(input.path);
      return path === null
        ? null
        : { kind: 'list', words: CHAT.workList, title: oneLine(relativeTo(path, folder)), path };
    }
    case 'grep': {
      const pattern = text(input.pattern);
      return pattern === null ? null : { kind: 'grep', words: CHAT.workGrep, title: `"${firstLine(pattern)}"` };
    }
    default:
      return null;
  }
}

/**
 * A `workflow` call as the thread draws it, by its `rawInput.source.type` (the reducer's `run`): a launch that came
 * back with a run id is its run's block; a refused or failed launch is a row, never a block; a check passed or failed;
 * the main agent's pause, stop or resume names the run and links to it. Null while the call runs, for a source the
 * agent could not read, and for a control call that failed: the plain tool row shows those.
 */
export type WorkflowForm =
  | { form: 'block'; runId: string }
  | { form: 'row'; kind: 'not-started' | 'check-failed'; title: string; error: string | null }
  | { form: 'row'; kind: 'checked'; title: string }
  | { form: 'row'; kind: 'control'; verb: 'pause' | 'stop' | 'resume'; title: string; runId: string | null };

const WORKFLOW_TOOL = 'workflow';

function errorLine(entry: ToolEntry): string | null {
  const output = entry.output ?? '';
  let start = 0;
  while (start < output.length) {
    const end = output.indexOf('\n', start);
    const line = oneLine(output.slice(start, end === -1 ? output.length : end)).trim();
    if (line !== '') return line;
    if (end === -1) break;
    start = end + 1;
  }
  return null;
}

/** The run a control call names, as the agent's input gave it: a run id or a name. */
function namedRun(input: Record<string, unknown> | null, verb: 'pause' | 'stop' | 'resume'): string | null {
  const source = input !== null && typeof input.source === 'object' && input.source !== null ? input.source : null;
  if (source === null) return null;
  return text((source as Record<string, unknown>)[verb === 'resume' ? 'resume_from_run_id' : 'run_id']);
}

/** `names`: run id to its name, from the `workflow` entries. */
export function workflowFormOf(entry: ToolEntry, names: ReadonlyMap<string, string>): WorkflowForm | null {
  const run = entry.run;
  if (entry.name !== WORKFLOW_TOOL || run === undefined) return null;
  const done = entry.status === 'completed';
  if (!done && entry.status !== 'failed') return null;
  const words = CHAT.run.toolRows;
  switch (run.kind) {
    case 'launch':
      return done && run.id !== null
        ? { form: 'block', runId: run.id }
        : { form: 'row', kind: 'not-started', title: words.notStarted, error: errorLine(entry) };
    case 'check': {
      if (!done) return { form: 'row', kind: 'check-failed', title: words.checkFailed, error: errorLine(entry) };
      const output = record(entry.output);
      const source = record(entry.input)?.source;
      const name =
        (output === null ? null : text(output.name)) ??
        (typeof source === 'object' && source !== null ? text((source as Record<string, unknown>).name) : null);
      return name === null
        ? null
        : { form: 'row', kind: 'checked', title: fill(words.checked, { name: oneLine(name) }) };
    }
    default: {
      if (!done) return null;
      const name = (run.id === null ? null : names.get(run.id)) ?? namedRun(record(entry.input), run.kind);
      if (name === null || name === '') return null;
      const title = run.kind === 'pause' ? words.paused : run.kind === 'stop' ? words.stopped : words.resumed;
      return {
        form: 'row',
        kind: 'control',
        verb: run.kind,
        title: fill(title, { name: oneLine(name) }),
        runId: run.id,
      };
    }
  }
}

export interface LaunchDetails {
  source: 'name' | 'script' | 'script_path';
  /** `name`: the saved workflow's name; `script_path`: the path as the main agent wrote it. */
  value: string | null;
  /** The script's args as JSON text; null with none. */
  args: string | null;
  /** The agent limit the call asked for; null for the default. */
  budget: number | null;
}

const LAUNCH_SOURCES: ReadonlySet<string> = new Set(['name', 'script', 'script_path']);

/** What a launch call asked for, from its input: the source, its args and limit. Null for an input that does not fit. */
export function launchDetails(entry: ToolEntry): LaunchDetails | null {
  const input = record(entry.input);
  const source = input === null ? null : input.source;
  if (typeof source !== 'object' || source === null || Array.isArray(source)) return null;
  const fields = source as Record<string, unknown>;
  const type = fields.type;
  if (typeof type !== 'string' || !LAUNCH_SOURCES.has(type)) return null;
  const kind = type as LaunchDetails['source'];
  const value = kind === 'script' ? null : text(fields[kind]);
  const args = input!.args;
  const budget = input!.agent_budget;
  return {
    source: kind,
    value,
    args: args === undefined || args === null ? null : JSON.stringify(args, null, 2),
    budget: typeof budget === 'number' && Number.isInteger(budget) ? budget : null,
  };
}

/** The entry a run's block is drawn at, and what that entry is. */
export interface RunPlace {
  at: number;
  kind: 'launch' | 'entry' | 'bare';
}

const NO_NAMES: ReadonlyMap<string, string> = new Map();

/**
 * Where each run's block is drawn, by run id: its launch call when the transcript holds it; else its `workflow` entry
 * (a cut, a fork, a client resume); else the first spawn line that names it (`bare`: no run entry at all).
 */
export function runPlaces(entries: readonly TranscriptEntry[]): ReadonlyMap<string, RunPlace> {
  const places = new Map<string, RunPlace>();
  entries.forEach((entry, at) => {
    if (entry.kind !== 'tool') return;
    const form = workflowFormOf(entry, NO_NAMES);
    if (form?.form === 'block' && !places.has(form.runId)) places.set(form.runId, { at, kind: 'launch' });
  });
  entries.forEach((entry, at) => {
    if (entry.kind === 'workflow' && !places.has(entry.runId)) places.set(entry.runId, { at, kind: 'entry' });
  });
  entries.forEach((entry, at) => {
    if (entry.kind === 'subagent' && entry.run !== undefined && !places.has(entry.run))
      places.set(entry.run, { at, kind: 'bare' });
  });
  return places;
}
