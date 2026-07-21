/**
 * Advisory per-model interprocess lock for the dashboard downloader.
 *
 * Two dashboard processes downloading the SAME model must not run concurrently:
 * their staging/backup swaps race and one can reclaim the other's active backup.
 * This is an O_EXCL lockfile on the same filesystem as `.staging`; liveness comes
 * from the holder's pid (a crashed holder is detected immediately by a dead pid),
 * with a wall-clock staleness backstop for the rare case a dead holder's pid is
 * later reused by an unrelated live process.
 *
 * Pure disk — never the native addon.
 */

import { open, readFile, rm, stat } from 'node:fs/promises';

/**
 * Backstop for pid-reuse ONLY: a holder older than this is stealable even when its
 * pid currently maps to a live, unrelated process. A dead holder is reaped at once
 * via {@link pidAlive}, so this only has to be longer than any realistic download.
 */
const DEFAULT_STALE_MS = 24 * 60 * 60 * 1000;
/** Total time to wait behind a LIVE same-model peer before giving up. */
const DEFAULT_WAIT_MS = 30_000;
/** Backoff between contention retries. */
const DEFAULT_POLL_MS = 100;
/** Bound on abandoned-lock steals so two peers can never ping-pong forever. */
const MAX_STEAL_ATTEMPTS = 100;

export interface LockOptions {
  staleMs?: number;
  waitMs?: number;
  pollMs?: number;
}

export interface LockHandle {
  /** Release the lock, but only if THIS process still owns it (never a stealer's). */
  release(): Promise<void>;
}

interface Holder {
  pid: number;
  at: number;
}

/**
 * Liveness of a pid via signal 0. An unparseable/invalid pid is treated as ALIVE
 * (never reaped) so a corrupt value can't drive a deletion; a delivered signal or
 * EPERM means alive; only ESRCH proves the process is gone.
 */
export function pidAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return (e as NodeJS.ErrnoException).code !== 'ESRCH';
  }
}

async function readHolder(lockPath: string): Promise<Holder | undefined> {
  try {
    const parsed = JSON.parse(await readFile(lockPath, 'utf8')) as { pid?: unknown; at?: unknown };
    if (typeof parsed.pid === 'number' && typeof parsed.at === 'number') {
      return { pid: parsed.pid, at: parsed.at };
    }
  } catch {
    // Absent / empty / corrupt: no readable holder.
  }
  return undefined;
}

async function lockMtimeMs(lockPath: string): Promise<number | undefined> {
  try {
    return (await stat(lockPath)).mtimeMs;
  } catch {
    return undefined;
  }
}

function makeHandle(lockPath: string): LockHandle {
  return {
    async release() {
      const holder = await readHolder(lockPath);
      // Only unlink our own lock: if a stealer took over (its pid, not ours), the
      // file now belongs to them and must be left in place.
      if (holder !== undefined && holder.pid === process.pid) {
        await rm(lockPath, { force: true }).catch(() => {});
      }
    },
  };
}

/** Create the lockfile exclusively; undefined if a holder already exists. Non-EEXIST errors propagate. */
async function tryCreate(lockPath: string): Promise<LockHandle | undefined> {
  let handle: Awaited<ReturnType<typeof open>>;
  try {
    handle = await open(lockPath, 'wx');
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === 'EEXIST') return undefined;
    throw e;
  }
  try {
    await handle.writeFile(JSON.stringify({ pid: process.pid, at: Date.now() }));
  } finally {
    await handle.close();
  }
  return makeHandle(lockPath);
}

const delay = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Acquire `lockPath`. Fresh acquisition wins immediately. Against an existing lock:
 * a dead-pid or stale holder is STOLEN (removed then re-created); a live holder is
 * waited out up to `waitMs` and then rejected — never silently bypassed. The
 * caller's `.staging` dir must already exist.
 */
export async function acquireLock(lockPath: string, opts: LockOptions = {}): Promise<LockHandle> {
  const staleMs = opts.staleMs ?? DEFAULT_STALE_MS;
  const waitMs = opts.waitMs ?? DEFAULT_WAIT_MS;
  const pollMs = opts.pollMs ?? DEFAULT_POLL_MS;
  const deadline = Date.now() + waitMs;
  let steals = 0;

  for (;;) {
    const handle = await tryCreate(lockPath);
    if (handle !== undefined) return handle;

    const holder = await readHolder(lockPath);
    let abandoned: boolean;
    if (holder !== undefined) {
      abandoned = !pidAlive(holder.pid) || Date.now() - holder.at > staleMs;
    } else {
      // Unparseable/empty holder (incl. a peer's brief open→write gap): use file
      // age so a fresh lock is waited out but a corrupt leftover eventually steals.
      const mtime = await lockMtimeMs(lockPath);
      abandoned = mtime === undefined || Date.now() - mtime > staleMs;
    }

    if (abandoned) {
      if (steals++ >= MAX_STEAL_ATTEMPTS) {
        throw new Error(`Could not acquire model lock "${lockPath}": an abandoned lock could not be cleared`);
      }
      await rm(lockPath, { force: true }).catch(() => {});
      continue;
    }

    if (Date.now() >= deadline) {
      const who = holder !== undefined ? `pid ${holder.pid}` : 'another process';
      throw new Error(
        `Another process (${who}) is already downloading this model; refusing to run concurrently`,
      );
    }
    await delay(pollMs);
  }
}
