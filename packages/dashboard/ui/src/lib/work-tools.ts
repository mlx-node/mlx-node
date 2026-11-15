/**
 * Work tool rows: the agent's input of a shell command, an edit, a read, a listing or a file search, read with a
 * shape check. An input that does not fit gives null, and the row falls back to the agent's title and the raw input.
 */

import { CHAT } from './chat-strings';
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

/** The Work form of a tool row, or null for a tool this table does not know or an input that does not fit. */
export function workCallOf(entry: ToolEntry, folder: string | null): WorkCall | null {
  const input = record(entry.input);
  if (input === null) return null;
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
