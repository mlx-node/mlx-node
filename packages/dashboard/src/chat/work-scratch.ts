/**
 * Work without a folder: each such conversation gets a new empty folder of its own under the scratch root
 * (`~/Portato`), named by the local minute of its first send. The app makes these folders and removes one only when
 * it is empty (`rmdir`, never recursive): it never deletes a file in them.
 */

import { realpathSync } from 'node:fs';
import { mkdir, realpath, rmdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';

/** The scratch root's name in the home folder: the app's name, so the user sees who made it. */
export const SCRATCH_FOLDER = 'Portato';

/** Names tried in one minute: the stamp, then `-2` to `-9`. */
const NAMES_PER_MINUTE = 9;

/** The longest file name the agent can make: it keeps a folder's conversations under its encoded path. */
const AGENT_NAME_BYTES = 255;

export function scratchRoot(home: string): string {
  return join(home, SCRATCH_FOLDER);
}

function two(n: number): string {
  return String(n).padStart(2, '0');
}

/** `YYYY-MM-DD-HHmm` in local time. */
export function stampName(ms: number): string {
  const at = new Date(ms);
  const day = `${at.getFullYear()}-${two(at.getMonth() + 1)}-${two(at.getDate())}`;
  return `${day}-${two(at.getHours())}${two(at.getMinutes())}`;
}

/**
 * Makes a new folder under `root` (the root itself when missing, mode 0700; an existing root is left as it is): the
 * stamp of `ms`, else `-2` to `-9`. The real path of the new folder, or null when none could be made.
 */
export async function makeScratch(root: string, ms: number): Promise<string | null> {
  let top: string;
  try {
    await mkdir(root, { recursive: true, mode: 0o700 });
    top = await realpath(root);
  } catch {
    return null;
  }
  const stamp = stampName(ms);
  for (let n = 1; n <= NAMES_PER_MINUTE; n += 1) {
    const path = join(top, n === 1 ? stamp : `${stamp}-${n}`);
    if (encodeURIComponent(path).length > AGENT_NAME_BYTES) return null;
    try {
      await mkdir(path, { mode: 0o700 });
      return path;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') return null;
    }
  }
  return null;
}

/** The scratch root's real path, or null when it does not exist. */
export function realScratchRoot(root: string): string | null {
  try {
    return realpathSync(root);
  } catch {
    return null;
  }
}

/** A Work folder started without a folder: its parent is the real scratch root. `cwd` is a real path. */
export function isScratch(cwd: string, realRoot: string | null): boolean {
  return realRoot !== null && cwd !== realRoot && dirname(cwd) === realRoot;
}

/** Removes the folder only when it is empty. True when it was removed. */
export async function removeIfEmpty(path: string): Promise<boolean> {
  try {
    await rmdir(path);
    return true;
  } catch {
    return false;
  }
}
