/**
 * Whether a command is on a `PATH`: each `<dir>/<name>` is tested for a regular file the user may execute. Node only:
 * no shell, nothing is run.
 */

import { constants } from 'node:fs';
import { access, stat } from 'node:fs/promises';
import { isAbsolute, join } from 'node:path';

/** The first `<dir>/<name>` of `path` that is an executable regular file, or null. Relative entries are skipped. */
export async function findOnPath(name: string, path: string | undefined): Promise<string | null> {
  if (name === '' || name.includes('/') || path === undefined) return null;
  for (const dir of path.split(':')) {
    if (dir === '' || !isAbsolute(dir)) continue;
    const candidate = join(dir, name);
    try {
      if (!(await stat(candidate)).isFile()) continue;
      await access(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Not there, or not executable.
    }
  }
  return null;
}
