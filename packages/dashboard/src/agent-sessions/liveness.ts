/**
 * Whether a grok process may be writing a session right now. Lock files are never
 * touched: grok blocks on `LOCK_EX`, and neither their existence nor their mtime
 * says anything (they are opened per write and never deleted).
 */

import { lstatSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

/** A write this recent may be from a process that never registered (ACP, headless `-p`). */
export const RECENT_WRITE_MS = 30_000;
/** An unterminated turn this recent may be a headless/ACP run inside a long tool call. */
export const OPEN_TURN_MS = 10 * 60_000;

export type ActiveRegistry = { ok: true; ids: Set<string> } | { ok: false; error: string };

/** grok `is_pid_alive`: `kill(pid, 0)` succeeds or fails with EPERM. */
export function isPidAlive(pid: number): boolean {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === 'EPERM';
  }
}

/**
 * Session ids registered in `<agentHome>/active_sessions.json` by a live pid. Only
 * the TUI registers. Read without grok's lock; grok writes it tmp+rename. A
 * missing or empty file is an empty registry; any other failure is reported so a
 * write guard can fail closed.
 */
export function readActiveRegistry(agentHome: string, alive: (pid: number) => boolean = isPidAlive): ActiveRegistry {
  let text: string;
  try {
    text = readFileSync(join(agentHome, 'active_sessions.json'), 'utf8');
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return { ok: true, ids: new Set() };
    return { ok: false, error: String(err) };
  }
  if (text.trim() === '') return { ok: true, ids: new Set() };
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (err) {
    return { ok: false, error: `active_sessions.json: ${String(err)}` };
  }
  if (!Array.isArray(raw)) return { ok: false, error: 'active_sessions.json is not an array' };
  const ids = new Set<string>();
  for (const item of raw) {
    if (typeof item !== 'object' || item === null) continue;
    const { session_id: id, pid } = item as { session_id?: unknown; pid?: unknown };
    if (typeof id === 'string' && typeof pid === 'number' && alive(pid)) ids.add(id);
  }
  return { ok: true, ids };
}

/**
 * grok `session_last_activity`: newest mtime of the session dir's own regular
 * files (not recursive, symlinks ignored), else the dir's mtime. `null` when the
 * dir is unreadable.
 */
export function newestFileMtimeMs(dir: string): number | null {
  let names: string[];
  try {
    names = readdirSync(dir);
  } catch {
    return null;
  }
  let newest: number | null = null;
  for (const name of names) {
    try {
      const st = lstatSync(join(dir, name));
      if (st.isFile() && (newest === null || st.mtimeMs > newest)) newest = st.mtimeMs;
    } catch {
      // Vanished between readdir and lstat.
    }
  }
  if (newest !== null) return newest;
  try {
    return lstatSync(dir).mtimeMs;
  } catch {
    return null;
  }
}

export type LiveReason = 'registered' | 'recent-write' | 'open-turn' | 'registry-unreadable';

export interface LiveVerdict {
  live: boolean;
  reason: LiveReason | null;
  /** The session the verdict is about: this one or the ancestor that is live. */
  sessionId: string | null;
}

export const NOT_LIVE: LiveVerdict = { live: false, reason: null, sessionId: null };

export interface OwnLivenessInput {
  id: string;
  dir: string;
  openTurn: boolean;
}

/** Rules 1, 3 and 4 for one session (rule 2, ancestors, is the caller's walk). */
export function ownLiveness(
  s: OwnLivenessInput,
  registry: ActiveRegistry,
  now: number,
  failClosed: boolean,
): LiveVerdict {
  if (!registry.ok) {
    if (failClosed) return { live: true, reason: 'registry-unreadable', sessionId: s.id };
  } else if (registry.ids.has(s.id)) {
    return { live: true, reason: 'registered', sessionId: s.id };
  }
  const newest = newestFileMtimeMs(s.dir);
  if (newest === null) return failClosed ? { live: true, reason: 'recent-write', sessionId: s.id } : NOT_LIVE;
  const age = now - newest;
  if (age < RECENT_WRITE_MS) return { live: true, reason: 'recent-write', sessionId: s.id };
  if (s.openTurn && age < OPEN_TURN_MS) return { live: true, reason: 'open-turn', sessionId: s.id };
  return NOT_LIVE;
}
