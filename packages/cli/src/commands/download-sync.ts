import { createHash } from 'node:crypto';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { isAbsolute, join, resolve, sep } from 'node:path';

import type { ListFileEntry } from '@huggingface/hub';

import type { DownloadCompletion } from './download-marker.js';

/** Streaming hex digest of a file's raw bytes (no git header). */
async function hashFile(path: string, algo: 'sha1' | 'sha256'): Promise<string> {
  const hash = createHash(algo);
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest('hex');
}

/** Git blob sha1 (`sha1("blob <size>\0" + bytes)`) — the top-level `oid` of a non-LFS file. */
async function gitBlobSha1(path: string): Promise<string> {
  const size = statSync(path).size;
  const hash = createHash('sha1');
  hash.update(`blob ${size}\0`);
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest('hex');
}

/**
 * Is the local copy at `destPath` the SAME CONTENT as the remote `file`?
 * Port of the dashboard's `isStagedCopyComplete`
 * (`packages/dashboard/src/download.ts:162-178`):
 *
 *   - size must match the manifest (fast gate, catches truncation);
 *   - if the manifest advertises `lfs.oid` (sha256 of the content), the local
 *     sha256 must match. Every weight takes this branch, Xet included — a
 *     Xet-backed entry carries `lfs.oid` too and it is a plain sha256;
 *   - else, for a plain file (no LFS, no Xet), the top-level git `oid` is the
 *     git-blob sha1 of the content, so that must match;
 *   - otherwise (no usable hash) fall back to size alone.
 *
 * The hash branches are what catch the case the size gate cannot: a repo
 * re-upload whose files kept identical sizes (re-trained weights).
 */
export async function fileUpToDate(destPath: string, file: ListFileEntry): Promise<boolean> {
  if (!existsSync(destPath)) return false;
  if (file.size > 0) {
    try {
      if (statSync(destPath).size !== file.size) return false;
    } catch {
      return false;
    }
  }
  if (file.lfs?.oid !== undefined) {
    return (await hashFile(destPath, 'sha256')) === file.lfs.oid;
  }
  if (file.lfs === undefined && file.xetHash === undefined && file.oid !== undefined) {
    return (await gitBlobSha1(destPath)) === file.oid;
  }
  return true;
}

/** Does the marker prove `repo` is already synced to `remoteSha`? */
export function isCompletionCurrent(completion: DownloadCompletion | null, repo: string, remoteSha: string): boolean {
  return completion !== null && completion.repo === repo && completion.revision === remoteSha;
}

/**
 * Old-marker files to delete because the remote repo no longer has them.
 *
 * `remotePaths` must be EVERY path in the remote tree, not the selected
 * subset — pruning against the selection would delete weights whenever a
 * later run uses a narrower `--glob`. Only files the old marker listed are
 * ever eligible (no marker ⇒ nothing is deleted), and entries that are
 * absolute, empty, or resolve outside `outputDir` are dropped, never deleted.
 *
 * Nested entries (any path containing '/') are NEVER eligible either: the CLI
 * lists the repo non-recursively, while the marker is shared with the
 * dashboard, whose listing IS recursive — so a dashboard-written marker can
 * record `sub/dir/file.json` that the CLI's `remotePaths` can never contain.
 * Absence from a listing that cannot see the file proves nothing; pruning on
 * it would delete a file that is still upstream and that the CLI's own
 * non-recursive selection could never re-download.
 */
export function computePruneList(previousFiles: string[], remotePaths: string[], outputDir: string): string[] {
  const remote = new Set(remotePaths);
  const root = resolve(outputDir);
  const out: string[] = [];
  for (const rel of previousFiles) {
    if (remote.has(rel)) continue;
    if (rel.includes('/')) continue;
    if (rel.length === 0 || isAbsolute(rel)) continue;
    const abs = resolve(root, rel);
    if (abs === root || !abs.startsWith(root + sep)) continue;
    out.push(rel);
  }
  return out;
}

/**
 * File list for the next marker: the current selection plus every previous
 * marker file that is still on the remote AND still on disk. A `--glob` run
 * must not shrink the marker and forget files a previous full run downloaded.
 *
 * Previous NESTED entries (path containing '/') skip the remote check: they
 * can come from a dashboard-written marker (recursive listing), so the CLI's
 * non-recursive `remotePaths` can never contain them and the check could never
 * pass — presence on disk alone decides, so a CLI sync does not silently
 * forget the dashboard's nested files.
 */
export function buildMarkerFiles(
  previous: DownloadCompletion | null,
  remotePaths: string[],
  selectedPaths: string[],
  outputDir: string,
): string[] {
  const files = new Set(selectedPaths);
  if (previous !== null) {
    const remote = new Set(remotePaths);
    for (const file of previous.files) {
      const provenOnRemote = file.includes('/') || remote.has(file);
      if (provenOnRemote && existsSync(join(outputDir, file))) files.add(file);
    }
  }
  return [...files].sort();
}
