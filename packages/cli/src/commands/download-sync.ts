import { createHash } from 'node:crypto';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { isAbsolute, join, resolve, sep } from 'node:path';

import type { ListFileEntry } from '@huggingface/hub';

import type { DownloadCompletion } from './download-marker.js';

const STANDARD_SHARD_RE = /^(?:model-|model\.safetensors-).+-of-.+\.safetensors$/;

function isStandardTopLevelWeightArtifact(path: string): boolean {
  if (path.includes('/') || path.includes('\\')) return false;
  return (
    path === 'model.safetensors' ||
    path === 'weights.safetensors' ||
    path === 'model.safetensors.index.json' ||
    STANDARD_SHARD_RE.test(path)
  );
}

function isSafeOutputRelativePath(outputDir: string, rel: string): boolean {
  if (rel.length === 0 || isAbsolute(rel)) return false;
  const root = resolve(outputDir);
  const abs = resolve(root, rel);
  return abs !== root && abs.startsWith(root + sep);
}

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
  return (
    completion !== null &&
    completion.scope !== 'partial' &&
    completion.repo === repo &&
    completion.revision === remoteSha
  );
}

/**
 * May a marker-current no-glob run skip the sync entirely?
 *
 * The marker files all existing is NOT enough on its own: a `--glob "*.json"`
 * run writes a selection-only marker, and a later full run must still download
 * the weights that selection never covered. The caller passes the local SHAPE
 * predicate (`isModelAlreadyDownloaded`: config + weights present) so a
 * selection-only marker can never satisfy the full-run short-circuit. GGUF
 * dirs fail the shape predicate and fall through to the manifest-checked
 * loop — the same cost the legacy gate paid for them.
 */
export function canShortCircuitFullRun(
  completion: DownloadCompletion,
  outputDir: string,
  localShapeComplete: boolean,
): boolean {
  if (completion.scope === 'partial' || !localShapeComplete) return false;
  return completion.files.every((f) => existsSync(join(outputDir, f)));
}

/**
 * The previous completion, but ONLY when it belongs to `repo` — else `null`.
 *
 * The run boundary rejects a foreign-repo marker. This remains defensive for
 * marker-less/invalid legacy directories and any future caller.
 */
export function sameRepoCompletion(completion: DownloadCompletion | null, repo: string): DownloadCompletion | null {
  return completion !== null && completion.repo === repo ? completion : null;
}

/**
 * A completion marker is ownership metadata, not a hint. Reusing a
 * same-basename output directory for another repo would mix both repos' files
 * and then publish a false identity for the resulting checkpoint.
 */
export function assertCompletionRepoCompatible(
  completion: DownloadCompletion | null,
  repo: string,
  outputDir: string,
): void {
  if (completion !== null && completion.repo !== repo) {
    throw new Error(
      `Refusing to download "${repo}" into "${outputDir}": the directory is owned by "${completion.repo}". ` +
        'Choose a distinct directory with --output.',
    );
  }
}

/**
 * Revision the next marker may CLAIM.
 *
 * A `--glob` sync at a CHANGED revision hash-verifies only its selection, yet
 * the marker union also carries previous-marker files — stamping the new sha
 * would launder those unverified files into a marker that lets the next full
 * run short-circuit forever. In that one case keep the OLD revision
 * (conservative under-claim: the next full run sees a stale marker and still
 * syncs + hash-verifies everything). A no-glob run verified everything it
 * records, and a glob run with NO previous same-repo marker records only its
 * own verified selection — both claim `remoteSha`; the latter corner degrades
 * to legacy behavior because the full-run short-circuit also demands the
 * local shape predicate ({@link canShortCircuitFullRun}).
 */
export function markerRevisionToClaim(
  previous: DownloadCompletion | null,
  remoteSha: string,
  isGlobRun: boolean,
): string {
  return isGlobRun && previous !== null && previous.revision !== remoteSha ? previous.revision : remoteSha;
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
 * Full CLI listings are recursive, matching the dashboard, so nested entries
 * are eligible once their absence is proven against the complete remote tree.
 */
export function computePruneList(
  previousFiles: string[],
  remotePaths: string[],
  outputDir: string,
  isGlobRun: boolean,
): string[] {
  // A glob run verifies only its selection. Even when a disappeared old file
  // is proven absent remotely, deleting it here can break the old checkpoint
  // without downloading the replacement files outside the narrow selection.
  if (isGlobRun) return [];
  const remote = new Set(remotePaths);
  const out: string[] = [];
  for (const rel of previousFiles) {
    if (remote.has(rel)) continue;
    if (!isSafeOutputRelativePath(outputDir, rel)) continue;
    out.push(rel);
  }
  return out;
}

/**
 * Marker-less directories from older CLIs have no manifest to prune against.
 * On a full sync, remove only the standard top-level safetensors artifacts
 * that influence loader selection and are proven absent from a remote tree
 * that itself contains a standard safetensors layout. User/adapter files and
 * GGUF/Paddle-only repos are left untouched.
 */
export function computeLegacyWeightPruneList(
  localPaths: string[],
  remotePaths: string[],
  outputDir: string,
  isGlobRun: boolean,
): string[] {
  if (isGlobRun) return [];
  const remote = new Set(remotePaths);
  if (!remotePaths.some(isStandardTopLevelWeightArtifact)) return [];
  return localPaths.filter(
    (rel) => isStandardTopLevelWeightArtifact(rel) && !remote.has(rel) && isSafeOutputRelativePath(outputDir, rel),
  );
}

/**
 * File list for the next marker: the current selection plus every previous
 * marker file that is still on disk and either remains remote or cannot be
 * judged by this run. A `--glob` run preserves all previous on-disk entries:
 * it intentionally does not prune, and the next full sync still needs those
 * entries in the marker so it can remove files proven stale after replacements
 * have been synchronized.
 *
 * Full listings are recursive, so top-level and nested previous entries use
 * the same remote-presence proof.
 */
export function buildMarkerFiles(
  previous: DownloadCompletion | null,
  remotePaths: string[],
  selectedPaths: string[],
  outputDir: string,
  isGlobRun: boolean,
): string[] {
  const files = new Set(selectedPaths);
  if (previous !== null) {
    const remote = new Set(remotePaths);
    for (const file of previous.files) {
      const provenOnRemote = isGlobRun || remote.has(file);
      if (provenOnRemote && existsSync(join(outputDir, file))) files.add(file);
    }
  }
  return [...files].sort();
}
