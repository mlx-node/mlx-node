import { execFile } from 'node:child_process';

/** How long one move to the Trash may take before the call gives up. */
const TRASH_TIMEOUT_MS = 60_000;

/**
 * Move `path` to the macOS Trash of the disk that holds it, as one item, with `/usr/bin/trash`. Rejects when the
 * tool cannot start or exits with an error. The exit code alone is not proof: a caller checks the path afterwards.
 */
export function moveToSystemTrash(path: string): Promise<void> {
  return new Promise((resolve, reject) => {
    execFile('/usr/bin/trash', ['-s', path], { timeout: TRASH_TIMEOUT_MS }, (error) => {
      if (error === null) resolve();
      else reject(error);
    });
  });
}
