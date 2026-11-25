/**
 * Helpers of Locate: make a catalog model's folder from model files that are already on this Mac, without writing
 * into the folder the user picked. Everything here only reads: `realpath`, `stat`, `readdir` and reads of small files.
 * The plan and the job that use these helpers live in `download.ts` (`DownloadManager.planLocate`).
 *
 *   user's folder (any disk)            <modelsDir>/<slug>/   a real folder the app makes
 *   X.gguf      --second name | link--> X.gguf                weight-like files: never copied
 *   config.json --copy----------------> config.json           small files: always the app's own files
 *   (missing)   --download------------> tokenizer.json, ...
 */

import { createHash } from 'node:crypto';
import { type BigIntStats, constants } from 'node:fs';
import { open, readdir, realpath, stat } from 'node:fs/promises';
import { basename, dirname, join, sep } from 'node:path';

import type { ListFileEntry } from '@huggingface/hub';

import {
  DOWNLOAD_COMPLETE_MARKER,
  type DownloadCompletion,
  isLinkableFile,
  isWeightFile,
  parseCompletion,
} from './models.js';

/** Child folders of the picked folder that Locate searches; with more, it searches the picked folder only. */
export const LOCATE_CHILD_MAX = 200;
/** A small file larger than this is not checked in the folder; it is downloaded. */
export const LOCATE_SMALL_MAX_BYTES = 64 * 2 ** 20;
/** How long the picked folder may take to answer (a disk that waits on an access prompt). */
export const LOCATE_DISK_DEADLINE_MS = 15_000;

/** The `stat` fields that change when a file is rewritten in place or replaced. */
export interface FileFingerprint {
  dev: bigint;
  ino: bigint;
  size: bigint;
  mtimeNs: bigint;
  ctimeNs: bigint;
}

export function fingerprintOf(stat: BigIntStats): FileFingerprint {
  return { dev: stat.dev, ino: stat.ino, size: stat.size, mtimeNs: stat.mtimeNs, ctimeNs: stat.ctimeNs };
}

export function sameFingerprint(a: FileFingerprint, b: FileFingerprint): boolean {
  return a.dev === b.dev && a.ino === b.ino && a.size === b.size && a.mtimeNs === b.mtimeNs && a.ctimeNs === b.ctimeNs;
}

/**
 * The digest that proves a manifest entry's bytes: the LFS sha256, or the git-blob sha1 of a plain file. `null` when
 * the entry has neither, and then nothing can prove it.
 */
export function checksumOf(file: ListFileEntry): { algo: 'sha256' | 'git-blob-sha1'; oid: string } | null {
  if (file.lfs?.oid !== undefined) return { algo: 'sha256', oid: file.lfs.oid };
  if (file.lfs === undefined && file.xetHash === undefined && file.oid !== undefined) {
    return { algo: 'git-blob-sha1', oid: file.oid };
  }
  return null;
}

/** The identity of a file Locate uses, as `stat` (links followed) saw it. `ctimeNs` is left out: `link()` moves it. */
export interface FoundFingerprint {
  dev: bigint;
  ino: bigint;
  size: bigint;
  mtimeNs: bigint;
}

/** A wanted file found in a folder: where its bytes really are, and their identity. */
export interface FoundFile {
  realPath: string;
  fingerprint: FoundFingerprint;
}

/** How a used file enters the app's folder: a second name, a link, or the app's own copy. */
export type LocateHow = 'link' | 'symlink' | 'copy';

/** A file of the picked folder that the app's folder will use. */
export interface LocateUse {
  /** The path in the app's folder (the catalog path). */
  path: string;
  realPath: string;
  how: LocateHow;
  /** The manifest entry, or `null` on the record path, where the app trusts its own record. */
  file: ListFileEntry | null;
  fingerprint: FoundFingerprint;
}

/** A wanted file the job downloads, with the repo and pinned revision it comes from. */
export interface LocateDownload {
  file: ListFileEntry;
  repo: string;
  revision: string;
}

/** A plan of Locate for one catalog folder: what the job uses, checks and downloads, and the token that names it. */
export interface LocatePlan {
  token: string;
  slug: string;
  repo: string;
  /** The real path of the folder the user picked; the job searches it again. */
  picked: string;
  /** The real path of the folder that holds the model: the picked folder or one of its child folders. */
  folder: string;
  mode: 'shared' | 'linked';
  /** The folder has this app's install record: its files are trusted, not read. */
  byRecord: boolean;
  revision: string;
  assetsRepo: string | null;
  assetsRevision: string | null;
  /** Every path the app's folder will list in its record. */
  wanted: string[];
  use: LocateUse[];
  download: LocateDownload[];
}

/** `promise`, or `onTimeout()` thrown when it has not settled in `ms`. */
export async function withDeadline<T>(promise: Promise<T>, ms: number, onTimeout: () => Error): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(onTimeout()), ms);
  });
  try {
    return await Promise.race([promise, deadline]);
  } finally {
    clearTimeout(timer);
  }
}

/** Whether `path` is `root` or inside it; both are real paths. */
export function isInside(path: string, root: string): boolean {
  return path === root || path.startsWith(root.endsWith(sep) ? root : root + sep);
}

/**
 * The folders Locate searches: the picked folder, then its direct child folders (links to folders followed, by their
 * real path), without names that start with `.` and without the models folder. More than {@link LOCATE_CHILD_MAX}
 * children leaves only the picked folder, with `tooMany`. Throws when the picked folder cannot be listed.
 */
export async function candidateFolders(
  picked: string,
  modelsDirReal: string,
): Promise<{ folders: string[]; tooMany: boolean }> {
  const entries = await readdir(picked, { withFileTypes: true });
  const children: string[] = [];
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    let real: string;
    if (entry.isDirectory()) real = join(picked, entry.name);
    else if (entry.isSymbolicLink()) {
      try {
        real = await realpath(join(picked, entry.name));
        if (!(await stat(real)).isDirectory()) continue;
      } catch {
        continue;
      }
    } else continue;
    if (isInside(real, modelsDirReal) || real === picked) continue;
    children.push(real);
  }
  if (children.length > LOCATE_CHILD_MAX) return { folders: [picked], tooMany: true };
  children.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  const folders = [picked];
  for (const child of children) if (!folders.includes(child)) folders.push(child);
  return { folders, tooMany: false };
}

/** Whether `path` is a folder, links followed. */
export async function isFolder(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory();
  } catch {
    return false;
  }
}

/** The file at `path` when it is a regular file (links followed), with its real path, or `null`. */
export async function foundAt(path: string): Promise<FoundFile | null> {
  try {
    const seen = await stat(path, { bigint: true });
    if (!seen.isFile()) return null;
    return {
      realPath: await realpath(path),
      fingerprint: { dev: seen.dev, ino: seen.ino, size: seen.size, mtimeNs: seen.mtimeNs },
    };
  } catch {
    return null;
  }
}

/** The wanted file `repoPath` in `root`: at its catalog path, else by its base name at the root of `root`. */
export async function findWanted(root: string, repoPath: string): Promise<FoundFile | null> {
  const atPath = await foundAt(join(root, ...repoPath.split('/')));
  if (atPath !== null) return atPath;
  const name = basename(repoPath);
  return name === repoPath ? null : foundAt(join(root, name));
}

/** Whether the bytes at `path` hash to `sum`, read once through a fd that `O_NONBLOCK` keeps off a FIFO. */
export async function digestMatches(
  path: string,
  sum: { algo: 'sha256' | 'git-blob-sha1'; oid: string },
): Promise<boolean> {
  const handle = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    const opened = await handle.stat({ bigint: true });
    if (!opened.isFile()) return false;
    const hash = createHash(sum.algo === 'sha256' ? 'sha256' : 'sha1');
    if (sum.algo === 'git-blob-sha1') hash.update(`blob ${opened.size}\0`);
    for await (const chunk of handle.createReadStream({ autoClose: false })) hash.update(chunk as Buffer);
    return hash.digest('hex') === sum.oid;
  } finally {
    await handle.close();
  }
}

/** This app's install record in `dir`, read without following a link at its name, or `undefined`. */
export async function readRecord(dir: string): Promise<DownloadCompletion | undefined> {
  let handle;
  try {
    handle = await open(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
    );
  } catch {
    return undefined;
  }
  try {
    if (!(await handle.stat()).isFile()) return undefined;
    return parseCompletion(JSON.parse(await handle.readFile('utf-8')) as unknown);
  } catch {
    return undefined;
  } finally {
    await handle.close();
  }
}

/** Whether `text` is a commit sha: 40 hex characters. */
export function isCommitSha(text: string): boolean {
  if (text.length !== 40) return false;
  for (const ch of text.toLowerCase()) {
    if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
  }
  return true;
}

/**
 * The record describes a complete install: it lists a weight and a `config.json` (no `config.json` when `configless`:
 * the build installs with none), and every file it lists is in `dir`.
 */
export async function recordInstalled(dir: string, record: DownloadCompletion, configless: boolean): Promise<boolean> {
  if (record.scope === 'partial') return false;
  if (!record.files.some(isWeightFile)) return false;
  if (configless ? record.files.includes('config.json') : !record.files.includes('config.json')) return false;
  for (const file of record.files) {
    if ((await foundAt(join(dir, ...file.split('/')))) === null) return false;
  }
  return true;
}

/** The folder part of a catalog path: `''` at the root. */
function catalogFolder(path: string): string {
  const at = path.lastIndexOf('/');
  return at < 0 ? '' : path.slice(0, at);
}

/**
 * The mode of a Locate. `shared` when every used weight-like file is on the disk of the models folder: the app gives
 * each a second name, and the engine reads beside the app's names. Otherwise `linked`, which the engine can use only
 * when it finds everything beside the real weights: each weight-like file keeps its catalog name (`name` otherwise,
 * the Hugging Face cache case), the weight-like files of one catalog folder are in one real folder, and every wanted
 * file of the primary weights' catalog folder is used from that real folder (`sibling` otherwise, naming the file).
 */
export function pickMode(
  used: readonly Pick<LocateUse, 'path' | 'realPath' | 'fingerprint'>[],
  wanted: readonly string[],
  primary: string,
  modelsDev: bigint,
  sameVolume: (a: bigint, b: bigint) => boolean,
): { mode: 'shared' | 'linked' } | { refuse: 'name' | 'sibling'; file: string } {
  const linkable = used.filter((item) => isLinkableFile(item.path));
  if (linkable.every((item) => sameVolume(item.fingerprint.dev, modelsDev))) return { mode: 'shared' };
  for (const item of linkable) {
    if (basename(item.realPath) !== basename(item.path)) return { refuse: 'name', file: basename(item.path) };
  }
  const realFolders = new Map<string, string>();
  for (const item of linkable) {
    const folder = catalogFolder(item.path);
    const real = dirname(item.realPath);
    const first = realFolders.get(folder);
    if (first === undefined) realFolders.set(folder, real);
    else if (first !== real) return { refuse: 'sibling', file: basename(item.path) };
  }
  const primaryFolder = catalogFolder(primary);
  const beside = realFolders.get(primaryFolder);
  for (const path of wanted) {
    if (catalogFolder(path) !== primaryFolder) continue;
    const item = used.find((candidate) => candidate.path === path);
    if (item === undefined || dirname(item.realPath) !== beside || basename(item.realPath) !== basename(path)) {
      return { refuse: 'sibling', file: basename(path) };
    }
  }
  return { mode: 'linked' };
}

/** The token of a Locate plan: a sha256 over everything the job must find again before it may build the folder. */
export function locateToken(plan: Omit<LocatePlan, 'token' | 'picked'>): string {
  const byFirst = (a: unknown[], b: unknown[]): number => {
    const [x, y] = [String(a[0]), String(b[0])];
    return x < y ? -1 : x > y ? 1 : 0;
  };
  const body = {
    slug: plan.slug,
    repo: plan.repo,
    revision: plan.revision,
    assetsRepo: plan.assetsRepo,
    assetsRevision: plan.assetsRevision,
    mode: plan.mode,
    byRecord: plan.byRecord,
    folder: plan.folder,
    wanted: [...plan.wanted].sort(),
    use: plan.use
      .map((item) => [
        item.path,
        item.realPath,
        item.fingerprint.dev.toString(),
        item.fingerprint.ino.toString(),
        item.fingerprint.size.toString(),
        item.fingerprint.mtimeNs.toString(),
        item.how,
        item.file === null ? null : [item.file.size, checksumOf(item.file)?.oid ?? null],
      ])
      .sort(byFirst),
    download: plan.download.map((item) => [item.file.path, item.file.size, item.repo, item.revision]).sort(byFirst),
  };
  return createHash('sha256').update(JSON.stringify(body)).digest('hex');
}

/**
 * Hash the file at `absPath` where it is, and say whether its bytes are the catalog file. The fd is opened without
 * following a link, and its `fstat` must equal `expect` before the read and be unchanged after it, so bytes that were
 * rewritten during the read are never taken as proven. Throws `changed()` otherwise, and when `hooks.cancelled()`
 * turns true between chunks.
 */
export async function proveAt(
  absPath: string,
  file: ListFileEntry,
  expect: { size: number; mtimeNs: bigint; ino: bigint },
  hooks: { cancelled(): boolean; onBytes(n: number): void },
  changed: () => Error,
  noChecksum: () => Error,
): Promise<{ match: boolean; fingerprint: FileFingerprint }> {
  const sum = checksumOf(file);
  if (sum === null) throw noChecksum();
  // O_NONBLOCK so that a FIFO swapped in after the plan cannot block the open.
  const handle = await open(absPath, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const opened = await handle.stat({ bigint: true });
    const before = fingerprintOf(opened);
    if (
      !opened.isFile() ||
      before.size !== BigInt(expect.size) ||
      before.mtimeNs !== expect.mtimeNs ||
      before.ino !== expect.ino
    ) {
      throw changed();
    }
    const hash = createHash(sum.algo === 'sha256' ? 'sha256' : 'sha1');
    if (sum.algo === 'git-blob-sha1') hash.update(`blob ${before.size}\0`);
    for await (const chunk of handle.createReadStream({ autoClose: false })) {
      if (hooks.cancelled()) throw new Error('Download cancelled');
      const bytes = chunk as Buffer;
      hash.update(bytes);
      hooks.onBytes(bytes.length);
    }
    const after = fingerprintOf(await handle.stat({ bigint: true }));
    if (!sameFingerprint(before, after)) throw changed();
    return { match: hash.digest('hex') === sum.oid, fingerprint: before };
  } finally {
    await handle.close();
  }
}
