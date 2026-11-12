/**
 * `ROOT/permission/<id>.jsonl`: the permission record of one Work conversation. Metadata only: times, tool call ids,
 * access kinds, mode lines and a hash of the agent settings files; never a command, a path or a title. It is history,
 * never state: the mode lives in memory and is never read back from here.
 */

import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { chmod, lstat, mkdir, open, readFile, stat, unlink } from 'node:fs/promises';
import { join, sep } from 'node:path';

import type { PermissionMark, PermissionMode } from './types.js';

type Access = Extract<PermissionMark, { kind: 'allowed' }>['access'];

/** One line of the file. `print` is {@link settingsPrint} at the time of the line. */
export type PermissionLine =
  | {
      kind: 'mode';
      at: number;
      mode: PermissionMode;
      from: PermissionMode;
      by: 'user' | 'restart' | 'record-failed';
      print?: string;
      editsUnasked?: true;
      userHooks?: true;
    }
  | { kind: 'allowed'; at: number; mode: 'auto' | 'bypass'; toolCallId: string; access: Access }
  /** The allowed line with this `toolCallId` and `at === of` got no allow: the reader drops both. */
  | { kind: 'withdrawn'; at: number; of: number; toolCallId: string }
  | { kind: 'settings'; at: number; print: string };

/** A line the thread shows: every kind but `withdrawn`. */
export type ShownLine = Exclude<PermissionLine, { kind: 'withdrawn' }>;

const TOOL_CALL_ID_MAX = 128;
const LOG_READ_MAX = 16 * 1024 * 1024;
const SETTINGS_FILE_MAX = 16 * 1024 * 1024;

/** A conversation id that can name a file: letters, digits, `-` and `_`, at most 128 characters. */
export function safeId(id: string): boolean {
  if (id === '' || id.length > TOOL_CALL_ID_MAX) return false;
  for (let i = 0; i < id.length; i++) {
    const c = id.charCodeAt(i);
    const ok =
      (c >= 0x30 && c <= 0x39) || (c >= 0x41 && c <= 0x5a) || (c >= 0x61 && c <= 0x7a) || c === 0x2d || c === 0x5f;
    if (!ok) return false;
  }
  return true;
}

/** A tool call id as the file keeps it: printable ASCII, at most 128 characters, else empty. */
function storedCallId(id: string): string {
  if (id.length > TOOL_CALL_ID_MAX) return '';
  for (let i = 0; i < id.length; i++) {
    const c = id.charCodeAt(i);
    if (c < 0x21 || c > 0x7e) return '';
  }
  return id;
}

function isPrint(value: unknown): value is string {
  if (typeof value !== 'string' || value.length !== 64) return false;
  for (const c of value) if (!'0123456789abcdef'.includes(c)) return false;
  return true;
}

function isTime(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0;
}

function isMode(value: unknown): value is PermissionMode {
  return value === 'ask' || value === 'auto' || value === 'bypass';
}

function isAccess(value: unknown): value is Access {
  return value === 'bash' || value === 'edit' || value === 'mcp' || value === 'other';
}

/** Every key of `r` is one of `keys`. */
function onlyKeys(r: Record<string, unknown>, keys: readonly string[]): boolean {
  for (const key of Object.keys(r)) if (!keys.includes(key)) return false;
  return true;
}

/** One line as a {@link PermissionLine}, field by field; anything else (an unknown key or value too) is null. */
export function readPermissionLine(text: string): PermissionLine | null {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return null;
  const r = raw as Record<string, unknown>;
  if (!isTime(r.at)) return null;
  if (r.kind === 'mode') {
    if (!onlyKeys(r, ['kind', 'at', 'mode', 'from', 'by', 'print', 'editsUnasked', 'userHooks'])) return null;
    if (!isMode(r.mode) || !isMode(r.from)) return null;
    if (r.by !== 'user' && r.by !== 'restart' && r.by !== 'record-failed') return null;
    if (r.print !== undefined && !isPrint(r.print)) return null;
    if (r.editsUnasked !== undefined && r.editsUnasked !== true) return null;
    if (r.userHooks !== undefined && r.userHooks !== true) return null;
    const line: PermissionLine = { kind: 'mode', at: r.at, mode: r.mode, from: r.from, by: r.by };
    if (r.print !== undefined) line.print = r.print;
    if (r.editsUnasked === true) line.editsUnasked = true;
    if (r.userHooks === true) line.userHooks = true;
    return line;
  }
  if (r.kind === 'allowed') {
    if (!onlyKeys(r, ['kind', 'at', 'mode', 'toolCallId', 'access'])) return null;
    if (r.mode !== 'auto' && r.mode !== 'bypass') return null;
    if (typeof r.toolCallId !== 'string' || !isAccess(r.access)) return null;
    if (r.mode === 'auto' && r.access !== 'bash' && r.access !== 'edit') return null;
    return { kind: 'allowed', at: r.at, mode: r.mode, toolCallId: storedCallId(r.toolCallId), access: r.access };
  }
  if (r.kind === 'withdrawn') {
    if (!onlyKeys(r, ['kind', 'at', 'of', 'toolCallId'])) return null;
    if (!isTime(r.of) || typeof r.toolCallId !== 'string') return null;
    return { kind: 'withdrawn', at: r.at, of: r.of, toolCallId: storedCallId(r.toolCallId) };
  }
  if (r.kind === 'settings') {
    if (!onlyKeys(r, ['kind', 'at', 'print']) || !isPrint(r.print)) return null;
    return { kind: 'settings', at: r.at, print: r.print };
  }
  return null;
}

/**
 * The lines of a file; a bad line is skipped. A `withdrawn` line drops itself and the last earlier `allowed` line
 * with its `toolCallId` and `at === of`; with no such line it is skipped alone.
 */
export function readPermissionLines(text: string): ShownLine[] {
  const out: ShownLine[] = [];
  let start = 0;
  while (start < text.length) {
    let end = text.indexOf('\n', start);
    if (end === -1) end = text.length;
    const line = readPermissionLine(text.slice(start, end));
    start = end + 1;
    if (line === null) continue;
    if (line.kind !== 'withdrawn') {
      out.push(line);
      continue;
    }
    for (let i = out.length - 1; i >= 0; i--) {
      const prior = out[i]!;
      if (prior.kind === 'allowed' && prior.toolCallId === line.toolCallId && prior.at === line.of) {
        out.splice(i, 1);
        break;
      }
    }
  }
  return out;
}

/** A line as the thread shows it. Lines read from disk carry no card text. */
export function markOf(line: ShownLine): PermissionMark {
  if (line.kind === 'allowed') {
    return {
      kind: 'allowed',
      at: line.at,
      mode: line.mode,
      toolCallId: line.toolCallId,
      access: line.access,
      card: null,
    };
  }
  if (line.kind === 'settings') return { kind: 'settings', at: line.at };
  const mark: PermissionMark = { kind: 'mode', at: line.at, mode: line.mode, from: line.from, by: line.by };
  if (line.editsUnasked === true) mark.editsUnasked = true;
  if (line.userHooks === true) mark.userHooks = true;
  return mark;
}

/** The text of one line: the fields of the line only, in a fixed order. */
function lineText(line: PermissionLine): string {
  if (line.kind === 'allowed') {
    return JSON.stringify({
      kind: 'allowed',
      at: line.at,
      mode: line.mode,
      toolCallId: storedCallId(line.toolCallId),
      access: line.access,
    });
  }
  if (line.kind === 'withdrawn') {
    return JSON.stringify({ kind: 'withdrawn', at: line.at, of: line.of, toolCallId: storedCallId(line.toolCallId) });
  }
  if (line.kind === 'settings') return JSON.stringify({ kind: 'settings', at: line.at, print: line.print });
  const out: Record<string, unknown> = { kind: 'mode', at: line.at, mode: line.mode, from: line.from, by: line.by };
  if (line.print !== undefined) out.print = line.print;
  if (line.editsUnasked === true) out.editsUnasked = true;
  if (line.userHooks === true) out.userHooks = true;
  return JSON.stringify(out);
}

/** The history files of every Work conversation. Every write runs after the one before it. */
export class PermissionLog {
  readonly dir: string;
  private chain: Promise<void> = Promise.resolve();
  /** Conversations whose file was removed: a line queued after the delete must not create it again. */
  private readonly removed = new Set<string>();

  constructor(dir: string) {
    this.dir = dir;
  }

  private path(id: string): string {
    if (!safeId(id)) throw new Error('bad conversation id');
    return join(this.dir, `${id}.jsonl`);
  }

  private serial<T>(run: () => Promise<T>): Promise<T> {
    const next = this.chain.then(run, run);
    this.chain = next.then(
      () => {},
      () => {},
    );
    return next;
  }

  /** Appends one line; the promise rejects when it was not written. A link at the file's place is refused. */
  append(id: string, line: PermissionLine): Promise<void> {
    return this.serial(async () => {
      const path = this.path(id);
      if (this.removed.has(id)) throw new Error('conversation deleted');
      await mkdir(this.dir, { recursive: true, mode: 0o700 });
      await chmod(this.dir, 0o700);
      const flags = constants.O_WRONLY | constants.O_APPEND | constants.O_CREAT | constants.O_NOFOLLOW;
      const handle = await open(path, flags, 0o600);
      try {
        await handle.writeFile(`${lineText(line)}\n`, 'utf8');
      } finally {
        await handle.close();
      }
    });
  }

  /** The lines of a conversation; none when the file is missing, too large or cannot be read. */
  read(id: string): Promise<ShownLine[]> {
    return this.serial(async () => {
      try {
        const path = this.path(id);
        const info = await lstat(path);
        if (!info.isFile() || info.size > LOG_READ_MAX) return [];
        return readPermissionLines(await readFile(path, 'utf8'));
      } catch {
        return [];
      }
    });
  }

  /** Deletes the file; every later append of this conversation is refused for the rest of the run. */
  remove(id: string): Promise<void> {
    return this.serial(async () => {
      this.removed.add(id);
      try {
        await unlink(this.path(id));
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code !== 'ENOENT') throw error;
      }
    });
  }
}

export interface SettingsPlace {
  home: string;
  /** The agent home of Work (`GROK_HOME`). */
  agentHome: string;
  cwd: string;
}

/** The agent settings files that can make the agent stop asking: allow rules, hooks, MCP servers. */
export function settingsFiles(place: SettingsPlace): string[] {
  return [
    join(place.agentHome, 'config.toml'),
    join(place.agentHome, 'managed_config.toml'),
    join(place.agentHome, 'hooks-paths'),
    `${sep}etc${sep}grok${sep}managed_config.toml`,
    join(place.home, '.claude', 'settings.json'),
    join(place.home, '.claude', 'settings.local.json'),
    join(place.home, '.claude.json'),
    join(place.home, '.cursor', 'hooks.json'),
    join(place.cwd, '.grok', 'config.toml'),
    join(place.cwd, '.claude', 'settings.json'),
    join(place.cwd, '.claude', 'settings.local.json'),
    join(place.cwd, '.mcp.json'),
    join(place.cwd, '.cursor', 'hooks.json'),
  ];
}

/** One SHA-256 over the settings files: each path with its content, or a fixed word when it is missing or unreadable. */
export async function settingsPrint(place: SettingsPlace): Promise<string> {
  const hash = createHash('sha256');
  for (const path of settingsFiles(place)) {
    hash.update(path);
    hash.update('\0');
    try {
      const info = await stat(path);
      if (info.size > SETTINGS_FILE_MAX) hash.update(`large:${info.size}:${info.mtimeMs}`);
      else hash.update(await readFile(path));
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      hash.update(code === 'ENOENT' || code === 'ENOTDIR' ? 'missing' : 'unreadable');
    }
    hash.update('\0');
  }
  return hash.digest('hex');
}
