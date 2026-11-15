/**
 * `ROOT/permission/<id>.jsonl`: the permission record of one Work conversation. Metadata only: times, tool call ids,
 * access kinds, mode lines and a hash per agent settings file, keyed by the fixed names of {@link SETTINGS_KEYS};
 * never a command, a title or any other path. It is history, never state: the mode lives in memory and is never read
 * back from here.
 */

import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { chmod, lstat, mkdir, open, readFile, stat, unlink } from 'node:fs/promises';
import { join, sep } from 'node:path';

import { gitRoot } from './agent/auto.js';
import type { PermissionMark, PermissionMode } from './types.js';

type Access = Extract<PermissionMark, { kind: 'allowed' }>['access'];

/**
 * The agent settings files the app watches: `<root>:<path under the root>`, where the root is the agent home of Work
 * (`agent`), `/etc` (`etc`), the user's home (`home`) or the Work folder (`folder`).
 */
export const SETTINGS_KEYS = [
  'agent:config.toml',
  'agent:managed_config.toml',
  'agent:hooks-paths',
  'etc:grok/managed_config.toml',
  'home:.claude/settings.json',
  'home:.claude/settings.local.json',
  'home:.claude.json',
  'home:.cursor/hooks.json',
  'folder:.grok/config.toml',
  'folder:.claude/settings.json',
  'folder:.claude/settings.local.json',
  'folder:.mcp.json',
  'folder:.cursor/hooks.json',
] as const;

export type SettingsKey = (typeof SETTINGS_KEYS)[number];

/** Per watched file: the SHA-256 of what the agent reads from it (hex), `missing` or `unreadable`. */
export type SettingsPrints = Record<SettingsKey, string>;

/**
 * One line of the file. `prints` is {@link settingsPrints} at the time of the line; a `settings` line also names the
 * files whose print differs from the base before it, in the order of {@link SETTINGS_KEYS}.
 */
export type PermissionLine =
  | {
      kind: 'mode';
      at: number;
      mode: PermissionMode;
      from: PermissionMode;
      by: 'user' | 'restart' | 'record-failed';
      prints?: SettingsPrints;
      editsUnasked?: true;
      userHooks?: true;
      /** To Auto: the confirm named the temp folder. */
      temp?: true;
    }
  /** `temp` (Auto only): a real place of the call is in the temp root. */
  | { kind: 'allowed'; at: number; mode: 'auto' | 'bypass'; toolCallId: string; access: Access; temp?: true }
  /** The allowed line with this `toolCallId` and `at === of` got no allow: the reader drops both. */
  | { kind: 'withdrawn'; at: number; of: number; toolCallId: string }
  | { kind: 'settings'; at: number; prints: SettingsPrints; changed: SettingsKey[] };

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
  if (value === 'missing' || value === 'unreadable') return true;
  if (typeof value !== 'string' || value.length !== 64) return false;
  for (const c of value) if (!'0123456789abcdef'.includes(c)) return false;
  return true;
}

function isSettingsKey(value: unknown): value is SettingsKey {
  return typeof value === 'string' && (SETTINGS_KEYS as readonly string[]).includes(value);
}

/** An object with every key of {@link SETTINGS_KEYS} and no other, each a print; else null. */
function readPrints(value: unknown): SettingsPrints | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const r = value as Record<string, unknown>;
  if (Object.keys(r).length !== SETTINGS_KEYS.length) return null;
  const out = {} as SettingsPrints;
  for (const key of SETTINGS_KEYS) {
    if (!Object.hasOwn(r, key) || !isPrint(r[key])) return null;
    out[key] = r[key];
  }
  return out;
}

/** At least one settings key, each once, in the order of {@link SETTINGS_KEYS}; else null. */
function readChanged(value: unknown): SettingsKey[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  let last = -1;
  const out: SettingsKey[] = [];
  for (const item of value) {
    if (!isSettingsKey(item)) return null;
    const at = SETTINGS_KEYS.indexOf(item);
    if (at <= last) return null;
    last = at;
    out.push(item);
  }
  return out;
}

function printsText(prints: SettingsPrints): Record<string, string> {
  const out: Record<string, string> = {};
  for (const key of SETTINGS_KEYS) out[key] = prints[key];
  return out;
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
    if (!onlyKeys(r, ['kind', 'at', 'mode', 'from', 'by', 'prints', 'editsUnasked', 'userHooks', 'temp'])) return null;
    if (!isMode(r.mode) || !isMode(r.from)) return null;
    if (r.by !== 'user' && r.by !== 'restart' && r.by !== 'record-failed') return null;
    const prints = r.prints === undefined ? undefined : readPrints(r.prints);
    if (prints === null) return null;
    if (r.editsUnasked !== undefined && r.editsUnasked !== true) return null;
    if (r.userHooks !== undefined && r.userHooks !== true) return null;
    if (r.temp !== undefined && (r.temp !== true || r.mode !== 'auto')) return null;
    const line: PermissionLine = { kind: 'mode', at: r.at, mode: r.mode, from: r.from, by: r.by };
    if (prints !== undefined) line.prints = prints;
    if (r.editsUnasked === true) line.editsUnasked = true;
    if (r.userHooks === true) line.userHooks = true;
    if (r.temp === true) line.temp = true;
    return line;
  }
  if (r.kind === 'allowed') {
    if (!onlyKeys(r, ['kind', 'at', 'mode', 'toolCallId', 'access', 'temp'])) return null;
    if (r.mode !== 'auto' && r.mode !== 'bypass') return null;
    if (typeof r.toolCallId !== 'string' || !isAccess(r.access)) return null;
    if (r.mode === 'auto' && r.access !== 'bash' && r.access !== 'edit') return null;
    if (r.temp !== undefined && (r.temp !== true || r.mode !== 'auto')) return null;
    const line: PermissionLine = {
      kind: 'allowed',
      at: r.at,
      mode: r.mode,
      toolCallId: storedCallId(r.toolCallId),
      access: r.access,
    };
    if (r.temp === true) line.temp = true;
    return line;
  }
  if (r.kind === 'withdrawn') {
    if (!onlyKeys(r, ['kind', 'at', 'of', 'toolCallId'])) return null;
    if (!isTime(r.of) || typeof r.toolCallId !== 'string') return null;
    return { kind: 'withdrawn', at: r.at, of: r.of, toolCallId: storedCallId(r.toolCallId) };
  }
  if (r.kind === 'settings') {
    if (!onlyKeys(r, ['kind', 'at', 'prints', 'changed'])) return null;
    const prints = readPrints(r.prints);
    const changed = readChanged(r.changed);
    if (prints === null || changed === null) return null;
    return { kind: 'settings', at: r.at, prints, changed };
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
export function markOf(line: ShownLine, place: SettingsPlace): PermissionMark {
  if (line.kind === 'allowed') {
    return {
      kind: 'allowed',
      at: line.at,
      mode: line.mode,
      toolCallId: line.toolCallId,
      access: line.access,
      card: null,
      ...(line.temp === true ? { temp: true as const } : {}),
    };
  }
  if (line.kind === 'settings') return { kind: 'settings', at: line.at, changed: shownPaths(place, line.changed) };
  const mark: PermissionMark = { kind: 'mode', at: line.at, mode: line.mode, from: line.from, by: line.by };
  if (line.editsUnasked === true) mark.editsUnasked = true;
  if (line.userHooks === true) mark.userHooks = true;
  if (line.temp === true) mark.temp = true;
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
      ...(line.temp === true ? { temp: true } : {}),
    });
  }
  if (line.kind === 'withdrawn') {
    return JSON.stringify({ kind: 'withdrawn', at: line.at, of: line.of, toolCallId: storedCallId(line.toolCallId) });
  }
  if (line.kind === 'settings') {
    return JSON.stringify({ kind: 'settings', at: line.at, prints: printsText(line.prints), changed: line.changed });
  }
  const out: Record<string, unknown> = { kind: 'mode', at: line.at, mode: line.mode, from: line.from, by: line.by };
  if (line.prints !== undefined) out.prints = printsText(line.prints);
  if (line.editsUnasked === true) out.editsUnasked = true;
  if (line.userHooks === true) out.userHooks = true;
  if (line.temp === true) out.temp = true;
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

/** The path of a watched file. */
export function settingsPath(place: SettingsPlace, key: SettingsKey): string {
  const colon = key.indexOf(':');
  const root = key.slice(0, colon);
  const parts = key.slice(colon + 1).split('/');
  if (root === 'agent') return join(place.agentHome, ...parts);
  if (root === 'etc') return join(`${sep}etc`, ...parts);
  if (root === 'home') return join(place.home, ...parts);
  return join(place.cwd, ...parts);
}

/** A path as the thread shows it: the user's home as `~`. */
export function shownPath(place: SettingsPlace, path: string): string {
  if (path === place.home) return '~';
  const prefix = place.home.endsWith(sep) ? place.home : `${place.home}${sep}`;
  return path.startsWith(prefix) ? `~${sep}${path.slice(prefix.length)}` : path;
}

/** The changed files as the thread shows them, in key order, each path once. */
export function shownPaths(place: SettingsPlace, keys: readonly SettingsKey[]): string[] {
  const out: string[] = [];
  for (const key of keys) {
    const shown = shownPath(place, settingsPath(place, key));
    if (!out.includes(shown)) out.push(shown);
  }
  return out;
}

/** The files whose print differs, in the order of {@link SETTINGS_KEYS}. */
export function changedSettings(base: SettingsPrints, next: SettingsPrints): SettingsKey[] {
  return SETTINGS_KEYS.filter((key) => base[key] !== next[key]);
}

function sha256(data: string | Buffer): string {
  return createHash('sha256').update(data).digest('hex');
}

/** JSON text with the keys of every object sorted, so the same values give the same text. */
function sortedJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(sortedJson).join(',')}]`;
  if (typeof value === 'object' && value !== null) {
    const r = value as Record<string, unknown>;
    const keys = Object.keys(r).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${sortedJson(r[k])}`).join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

function own(value: unknown, key: string): unknown {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined;
  return Object.hasOwn(value, key) ? (value as Record<string, unknown>)[key] : undefined;
}

const PROJECT_KEYS = ['mcpServers', 'enabledMcpjsonServers', 'disabledMcpjsonServers'] as const;

/**
 * What the agent reads from `~/.claude.json`: the top-level `mcpServers`, and per folder (the Work folder and its git
 * root) the project entry's MCP keys. The rest of the file (usage counters, rewritten every few seconds) is not
 * part of it. Text that is not JSON is null.
 */
export function claudeJsonPart(text: string, folders: readonly string[]): string | null {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  const projects = own(raw, 'projects');
  const part: Record<string, unknown> = { mcpServers: own(raw, 'mcpServers') ?? null };
  const perFolder: Record<string, unknown> = {};
  for (const folder of folders) {
    const entry = own(projects, folder);
    const keys: Record<string, unknown> = {};
    for (const key of PROJECT_KEYS) keys[key] = own(entry, key) ?? null;
    perFolder[folder] = keys;
  }
  part.projects = perFolder;
  try {
    return sortedJson(part);
  } catch {
    return null;
  }
}

/**
 * The print of each watched file: the SHA-256 of its bytes (of {@link claudeJsonPart} for `~/.claude.json`), or
 * `missing` / `unreadable`. A file over 16 MB is not read: its size and modification time are hashed instead.
 */
export async function settingsPrints(place: SettingsPlace): Promise<SettingsPrints> {
  const folders = [place.cwd];
  const root = await gitRoot(place.cwd);
  if (root !== place.cwd) folders.push(root);
  const out = {} as SettingsPrints;
  for (const key of SETTINGS_KEYS) {
    const path = settingsPath(place, key);
    try {
      const info = await stat(path);
      if (info.size > SETTINGS_FILE_MAX) {
        out[key] = sha256(`large:${info.size}:${info.mtimeMs}`);
      } else if (key === 'home:.claude.json') {
        const part = claudeJsonPart(await readFile(path, 'utf8'), folders);
        out[key] = part === null ? 'unreadable' : sha256(part);
      } else {
        out[key] = sha256(await readFile(path));
      }
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      out[key] = code === 'ENOENT' || code === 'ENOTDIR' ? 'missing' : 'unreadable';
    }
  }
  return out;
}
