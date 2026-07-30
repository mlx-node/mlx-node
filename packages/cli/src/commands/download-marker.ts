import { randomUUID } from 'node:crypto';
import { readFile, rename, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

/**
 * Filename of the completion marker written after a successful download.
 * MUST stay byte-identical to the dashboard's marker constant
 * (`packages/dashboard/src/models.ts` `DOWNLOAD_COMPLETE_MARKER`): both tools
 * read and write the same file so each recognizes the other's installs. The
 * CLI must not import the dashboard package, so the value is duplicated and a
 * contract test (`__test__/cli/download-marker.test.ts`) pins the two together.
 */
export const DOWNLOAD_COMPLETE_MARKER = '.mlx-download-complete.json';

export type DownloadScope = 'full' | 'partial';

/** Contents of {@link DOWNLOAD_COMPLETE_MARKER}: the pinned snapshot a dir holds. */
export interface DownloadCompletion {
  /** HuggingFace repo the checkpoint came from. */
  repo: string;
  /** The exact commit sha the whole download was pinned to (one snapshot). */
  revision: string;
  /** Repo-relative paths of every file this tool downloaded or verified. */
  files: string[];
  /**
   * Whether the marker covers a full-model run. Missing means `full` for
   * compatibility with dashboard and CLI markers written before this field.
   */
  scope?: DownloadScope;
  /** ISO timestamp of the last successful download/sync. */
  completedAt: string;
}

/**
 * Parse the completion marker in `dir`, or `null` when absent or invalid in
 * any way. Never throws: a corrupt marker must degrade to "no marker" (a sync
 * pass), not crash the download command.
 */
export async function readCompletion(dir: string): Promise<DownloadCompletion | null> {
  let raw: string;
  try {
    raw = await readFile(join(dir, DOWNLOAD_COMPLETE_MARKER), 'utf8');
  } catch {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== 'object') return null;
  const marker = parsed as Record<string, unknown>;
  if (
    typeof marker.repo !== 'string' ||
    typeof marker.revision !== 'string' ||
    typeof marker.completedAt !== 'string' ||
    !Array.isArray(marker.files) ||
    !marker.files.every((file) => typeof file === 'string') ||
    (marker.scope !== undefined && marker.scope !== 'full' && marker.scope !== 'partial')
  ) {
    return null;
  }
  return {
    repo: marker.repo,
    revision: marker.revision,
    files: marker.files as string[],
    scope: marker.scope as DownloadScope | undefined,
    completedAt: marker.completedAt,
  };
}

/**
 * Preserve a valid marker's ownership while a sync is in progress, but make
 * it ineligible for any "already complete/current" gate until finalization
 * atomically publishes a new full or partial result.
 */
export function markCompletionPartial(completion: DownloadCompletion): DownloadCompletion {
  return { ...completion, scope: 'partial' };
}

/**
 * Write the marker atomically (temp file + rename in the same directory), so
 * a crash mid-write can never leave a truncated marker that would then parse
 * as garbage on the next run.
 *
 * The temp name is unique per write (pid + random) and opened with `wx`: a
 * fixed `.tmp` name could be pre-planted as a symlink (making the write land
 * elsewhere) and races two concurrent writers onto the same temp file. On any
 * failure the temp file is removed; after a successful rename it no longer
 * exists and the cleanup is a no-op.
 */
export async function writeCompletion(dir: string, completion: DownloadCompletion): Promise<void> {
  const finalPath = join(dir, DOWNLOAD_COMPLETE_MARKER);
  const tmpPath = `${finalPath}.${process.pid}.${randomUUID().slice(0, 8)}.tmp`;
  try {
    await writeFile(tmpPath, `${JSON.stringify(completion, null, 2)}\n`, { flag: 'wx' });
    await rename(tmpPath, finalPath);
  } finally {
    await rm(tmpPath, { force: true });
  }
}
