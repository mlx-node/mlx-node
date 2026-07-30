/**
 * Pid-scoped temp roots for the host's paged-config overrides, plus the
 * startup sweep that reclaims the ones a killed host left behind.
 *
 * `PagedConfigOverrideManager` clones a checkpoint directory (config.json
 * rewritten, everything else symlinked) into a temp root and removes that root
 * in `cleanup()`. `cleanup()` runs on a normal shutdown — but the host's
 * headline deployment is an Electron `utilityProcess`, and a `utilityProcess`
 * can be SIGKILLed (app force-quit, OOM killer, `kill -9`). SIGKILL runs no
 * handler, so without a sweep EVERY hard kill leaks a root. The clones are
 * symlink farms rather than copies, so the leak is inodes and directory
 * entries rather than model-sized bytes — but the roots accumulate forever and
 * a partially-written clone can hold a real config.json.
 *
 * The reclaim strategy is "name the owner in the directory name": the manager
 * gets a `tempDirPrefix` of `mlx-inference-host-<pid>-`, `mkdtemp` appends its
 * own random suffix, and {@link sweepOrphanHostTempRoots} parses the pid back
 * out and removes any root whose owner is gone.
 *
 * Known limitation: pid reuse. A long-dead host's root whose pid has since
 * been recycled by an unrelated process is SPARED (never wrongly deleted), so
 * the failure mode is a leaked directory, not data loss.
 */

import { readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

/** Shared stem. A directory is host-owned iff its name starts with this. */
export const HOST_TEMP_DIR_STEM = 'mlx-inference-host-';

/** Matches `mlx-inference-host-<pid>-<mkdtemp suffix>`. */
const HOST_TEMP_DIR_RE = /^mlx-inference-host-(\d+)-/;

/**
 * `tempDirPrefix` to hand `PagedConfigOverrideManager` so the root it creates
 * carries its owner's pid.
 */
export function hostTempDirPrefix(pid: number = process.pid): string {
  return `${HOST_TEMP_DIR_STEM}${pid}-`;
}

/**
 * Does a process with this pid exist?
 *
 * `process.kill(pid, 0)` sends no signal; it only performs the permission +
 * existence check. `ESRCH` is the sole "gone" answer — `EPERM` means the
 * process exists but belongs to another user, which must count as ALIVE so a
 * multi-user box never has one user's sweep delete another user's live root.
 */
export function isProcessAlive(pid: number): boolean {
  // pid 0 addresses the caller's whole process group on POSIX and pid < 0 a
  // group by id; neither is ever a real owner, and signalling them would be
  // actively dangerous. Treat as alive so they are never swept.
  if (!Number.isInteger(pid) || pid <= 0) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code !== 'ESRCH';
  }
}

export interface SweepOrphanTempRootsOptions {
  /** Directory to scan. Defaults to the OS temp dir. */
  root?: string;
  /** Our own pid — never swept, however the liveness probe answers. */
  selfPid?: number;
  /** Injectable liveness probe. Defaults to {@link isProcessAlive}. */
  isAlive?: (pid: number) => boolean;
}

/**
 * Remove every host temp root whose owning pid is no longer alive.
 *
 * Best effort by contract: a scan or unlink failure (permissions, a root a
 * concurrently-exiting host is removing under us) is swallowed, because a
 * housekeeping step must never be the reason a host refuses to start.
 * Returns the absolute paths actually removed.
 */
export async function sweepOrphanHostTempRoots(opts: SweepOrphanTempRootsOptions = {}): Promise<string[]> {
  const root = opts.root ?? tmpdir();
  const selfPid = opts.selfPid ?? process.pid;
  const isAlive = opts.isAlive ?? isProcessAlive;

  let entries: string[];
  try {
    entries = await readdir(root);
  } catch {
    return [];
  }

  const removed: string[] = [];
  for (const name of entries) {
    const match = HOST_TEMP_DIR_RE.exec(name);
    if (match === null) continue;
    const pid = Number.parseInt(match[1], 10);
    if (pid === selfPid) continue;
    if (isAlive(pid)) continue;

    const full = join(root, name);
    try {
      await rm(full, { recursive: true, force: true });
      removed.push(full);
    } catch {
      /* another sweeper won the race, or we lack permission; leave it */
    }
  }
  return removed;
}
