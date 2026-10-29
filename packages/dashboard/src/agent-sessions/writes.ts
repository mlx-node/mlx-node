/**
 * Rename and delete for grok-build sessions.
 *
 * grok has summary writers that take no lock or write a stale snapshot
 * (`restore_wake_summary_locked`, subagent `new_with_explicit_dir`,
 * `stamp_live_fork_session_metadata`), so an atomic write alone is not safe: the
 * only safety is that no grok process holds the session or any ancestor. Every
 * write therefore re-checks liveness immediately before touching disk. grok's
 * `*.lock` files are never opened (grok blocks on `LOCK_EX`).
 */

import { randomUUID } from 'node:crypto';
import {
  closeSync,
  fsyncSync,
  lstatSync,
  openSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  type Stats,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { basename, dirname, join } from 'node:path';

import { type LiveVerdict, OPEN_TURN_MS, RECENT_WRITE_MS } from './liveness.js';
import type { AgentSessionsReader, SessionIndex, SessionRef } from './reader.js';
import type { SessionDeleteResponse, SessionRenameResponse } from './types.js';

/** grok `MAX_TITLE_SCALARS`, counted after the control strip and trim. */
export const MAX_TITLE_SCALARS = 100;
/** grok `MAX_TITLE_BYTES`: UTF-8 length of the raw title, checked first. */
export const MAX_TITLE_BYTES = MAX_TITLE_SCALARS * 4 + 64;
/** grok `TITLE_REFRESH_TURNS.len()`: a watermark at the end freezes auto titles. */
const TITLE_REFRESH_FROZEN = '2';

export type SessionWriteErrorKind = 'bad-request' | 'not-found' | 'conflict';

export class SessionWriteError extends Error {
  readonly kind: SessionWriteErrorKind;

  constructor(kind: SessionWriteErrorKind, message: string) {
    super(message);
    this.name = 'SessionWriteError';
    this.kind = kind;
  }
}

/** grok `is_forbidden_title_char`: `char::is_control` (Cc) plus bidi marks/overrides. */
function isForbiddenTitleChar(cp: number): boolean {
  return (
    cp <= 0x1f ||
    (cp >= 0x7f && cp <= 0x9f) ||
    cp === 0x200e ||
    cp === 0x200f ||
    (cp >= 0x202a && cp <= 0x202e) ||
    (cp >= 0x2066 && cp <= 0x2069)
  );
}

/** Rust `char::is_whitespace` (White_Space). Unlike JS `trim`, U+FEFF is not whitespace. */
function isRustWhitespace(cp: number): boolean {
  return (
    (cp >= 0x09 && cp <= 0x0d) ||
    cp === 0x20 ||
    cp === 0x85 ||
    cp === 0xa0 ||
    cp === 0x1680 ||
    (cp >= 0x2000 && cp <= 0x200a) ||
    cp === 0x2028 ||
    cp === 0x2029 ||
    cp === 0x202f ||
    cp === 0x205f ||
    cp === 0x3000
  );
}

/** grok `validate_rename_title` for a manual title; throws `bad-request`. */
export function validateRenameTitle(raw: unknown): string {
  if (typeof raw !== 'string') throw new SessionWriteError('bad-request', 'Body must be {"title": string}');
  if (/[\uD800-\uDFFF]/u.test(raw)) throw new SessionWriteError('bad-request', 'Title is not valid Unicode');
  if (Buffer.byteLength(raw, 'utf8') > MAX_TITLE_BYTES) throw new SessionWriteError('bad-request', 'Title too large');
  // Unicode scalars, as grok counts them (not grapheme clusters).
  const scalars: string[] = [];
  for (const c of raw) if (!isForbiddenTitleChar(c.codePointAt(0)!)) scalars.push(c);
  let start = 0;
  let end = scalars.length;
  while (start < end && isRustWhitespace(scalars[start]!.codePointAt(0)!)) start++;
  while (end > start && isRustWhitespace(scalars[end - 1]!.codePointAt(0)!)) end--;
  if (start === end) throw new SessionWriteError('bad-request', 'Title must not be blank');
  if (end - start > MAX_TITLE_SCALARS) {
    throw new SessionWriteError(
      'bad-request',
      `Title too long (max ${MAX_TITLE_SCALARS} characters after removing control characters)`,
    );
  }
  return scalars.slice(start, end).join('');
}

/**
 * A session id is a single folder name. Rejects `.`/`..`, separators (a `%2F`
 * in the URL arrives decoded), NUL and dot-names, which the scan never indexes.
 */
export function assertSessionIdShape(id: string): void {
  if (id === '' || id.startsWith('.') || /[/\\\0]/.test(id)) {
    throw new SessionWriteError('bad-request', `Invalid session id "${id}"`);
  }
}

function lstatOrNull(path: string): Stats | null {
  try {
    return lstatSync(path);
  } catch {
    return null;
  }
}

/**
 * The session's folder, proven to be a real `<sessionsRoot>/<encodedCwd>/<id>`
 * directory: neither level is a symlink and its realpath sits exactly two
 * levels under the sessions root's realpath.
 */
export function containedSessionDir(sessionsRoot: string, ref: Pick<SessionRef, 'id' | 'dir'>): string {
  assertSessionIdShape(ref.id);
  const refuse = (why: string): never => {
    throw new SessionWriteError('conflict', `Refusing to modify session ${ref.id}: ${why}`);
  };
  const cwdDir = dirname(ref.dir);
  const cwdName = basename(cwdDir);
  if (basename(ref.dir) !== ref.id || dirname(cwdDir) !== sessionsRoot || cwdName === '' || cwdName.startsWith('.')) {
    refuse('not a direct <cwd>/<id> folder of the sessions root');
  }
  for (const path of [cwdDir, ref.dir]) {
    const st = lstatOrNull(path);
    if (st === null) refuse(`${path} is gone`);
    if (st!.isSymbolicLink() || !st!.isDirectory()) refuse(`${path} is not a real directory`);
  }
  let realRoot: string;
  let realDir: string;
  try {
    realRoot = realpathSync(sessionsRoot);
    realDir = realpathSync(ref.dir);
  } catch (err) {
    return refuse(`cannot resolve its path (${(err as Error).message})`);
  }
  if (realDir !== join(realRoot, cwdName, ref.id)) refuse('it resolves outside the sessions folder');
  return ref.dir;
}

function liveMessage(id: string, verdict: LiveVerdict): string {
  const who = verdict.sessionId === null || verdict.sessionId === id ? '' : ` (via parent ${verdict.sessionId})`;
  switch (verdict.reason) {
    case 'registered':
      return `Session ${id} is open in a running mlx agent${who}; close it first`;
    case 'open-turn':
      return `Session ${id} has a turn in progress or interrupted${who} (written in the last ${OPEN_TURN_MS / 60_000} min); retry later`;
    case 'registry-unreadable':
      return `Cannot read active_sessions.json, so session ${id} may be in use; refusing to modify it`;
    default:
      return `Session ${id} was written in the last ${RECENT_WRITE_MS / 1000} s${who} and may be in use; retry shortly`;
  }
}

function assertNotLive(reader: AgentSessionsReader, index: SessionIndex, ids: string[]): void {
  for (const id of ids) {
    const verdict = reader.liveness(id, true, index);
    if (verdict.live) throw new SessionWriteError('conflict', liveMessage(id, verdict));
  }
}

/** grok `write_bytes_atomic`: unique sibling temp, fsync, rename over, fsync the dir. */
export function writeFileAtomic(path: string, data: string): void {
  const tmp = `${path}.${randomUUID()}.tmp`;
  const fd = openSync(tmp, 'wx');
  try {
    writeFileSync(fd, data);
    fsyncSync(fd);
  } catch (err) {
    closeSync(fd);
    unlinkSync(tmp);
    throw err;
  }
  closeSync(fd);
  try {
    renameSync(tmp, path);
  } catch (err) {
    try {
      unlinkSync(tmp);
    } catch {
      // Already gone.
    }
    throw err;
  }
  try {
    const dirFd = openSync(dirname(path), 'r');
    try {
      fsyncSync(dirFd);
    } finally {
      closeSync(dirFd);
    }
  } catch {
    // Best effort, as for grok on filesystems that refuse a directory fsync.
  }
}

function sameFile(a: Stats, b: Stats): boolean {
  return a.ino === b.ino && a.size === b.size && a.mtimeMs === b.mtimeMs;
}

/**
 * grok's manual `/rename` on a dormant session (`Summary::apply_patch`):
 * `generated_title = title`, `session_summary = title` only while it is empty,
 * `title_is_manual = true`, `updated_at = now`, plus the frozen title-refresh
 * watermark. Every other field is kept as parsed.
 */
export function renameSession(
  reader: AgentSessionsReader,
  id: string,
  title: string,
  now: () => Date = () => new Date(),
): SessionRenameResponse {
  assertSessionIdShape(id);
  const index = reader.scan();
  const ref = index.sessions.get(id);
  if (ref === undefined) throw new SessionWriteError('not-found', `Session ${id} not found`);
  assertNotLive(reader, index, [id]);
  const dir = containedSessionDir(reader.sessionsRoot, ref);
  const summaryPath = join(dir, 'summary.json');

  const before = lstatOrNull(summaryPath);
  if (before === null || !before.isFile()) {
    throw new SessionWriteError('conflict', `Session ${id} has no regular summary.json`);
  }
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(summaryPath, 'utf8'), (_key, value: unknown) => {
      if (typeof value === 'number' && Number.isInteger(value) && !Number.isSafeInteger(value)) {
        throw new Error('it holds an integer JavaScript cannot round-trip');
      }
      return value;
    });
  } catch (err) {
    throw new SessionWriteError('conflict', `Cannot rewrite summary.json of ${id}: ${(err as Error).message}`);
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new SessionWriteError('conflict', `summary.json of ${id} is not an object`);
  }
  const summary = raw as Record<string, unknown>;
  const info = summary.info as Record<string, unknown> | undefined;
  if (typeof info !== 'object' || info === null || info.id !== id) {
    throw new SessionWriteError('conflict', `summary.json in ${id} names a different session`);
  }
  summary.generated_title = title;
  if (typeof summary.session_summary !== 'string' || summary.session_summary === '') {
    summary.session_summary = title;
  }
  summary.title_is_manual = true;
  summary.updated_at = now().toISOString();
  const text = JSON.stringify(summary, null, 2);

  assertNotLive(reader, index, [id]);
  const again = lstatOrNull(summaryPath);
  if (again === null || !sameFile(before, again)) {
    throw new SessionWriteError('conflict', `summary.json of ${id} changed during the rename; retry`);
  }
  writeFileAtomic(summaryPath, text);
  reader.invalidate(dir);
  reader.noteOwnWrite(summaryPath);
  const watermark = join(dir, 'title_refresh_idx');
  try {
    writeFileAtomic(watermark, TITLE_REFRESH_FROZEN);
    reader.noteOwnWrite(watermark);
  } catch {
    // Defense in depth only: `title_is_manual` already stops auto titles.
  }
  return { id, title };
}

/**
 * grok `delete_session_history` removes the one session folder. We also remove
 * its subagent children, found only through `<session>/subagents/<child>/meta.json`
 * whose `parent_session_id` names the parent (never through `subagent_spawned`
 * lines: forks copy `updates.jsonl`), each child looked up by id across every
 * cwd folder. Refused as a whole when any of them is live.
 */
export function deleteSession(reader: AgentSessionsReader, id: string): SessionDeleteResponse {
  assertSessionIdShape(id);
  const index = reader.scan();
  const root = index.sessions.get(id);
  if (root === undefined) throw new SessionWriteError('not-found', `Session ${id} not found`);

  const targets: SessionRef[] = [root];
  const seen = new Set([id]);
  for (let i = 0; i < targets.length; i++) {
    for (const link of index.childrenOf.get(targets[i]!.id) ?? []) {
      if (seen.has(link.childId)) continue;
      seen.add(link.childId);
      const child = index.sessions.get(link.childId);
      if (child !== undefined) targets.push(child);
    }
  }
  const ids = targets.map((t) => t.id);
  assertNotLive(reader, index, ids);
  const dirs = targets.map((t) => containedSessionDir(reader.sessionsRoot, t));
  assertNotLive(reader, index, ids);

  // Children first, so a failure part way leaves the parent to retry from.
  for (let i = dirs.length - 1; i >= 0; i--) {
    try {
      rmSync(dirs[i]!, { recursive: true, force: true });
    } finally {
      reader.invalidate(dirs[i]!);
    }
  }
  return { deleted: ids };
}
