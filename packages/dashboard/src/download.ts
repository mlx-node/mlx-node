/**
 * Resumable catalog download runner for the dashboard.
 *
 * Ports the manifest/resume behavior of `packages/cli/src/commands/download-model.ts`
 * (list files → per-file `downloadFileToCacheDir` into the HF cache → `copyFile`)
 * and adds structured, per-file byte progress by wrapping the injected `fetch` in
 * a counting `TransformStream`. Jobs run one at a time; consumers observe progress
 * through `subscribe`, which replays the last event on attach so a late attacher
 * (e.g. an SSE reconnect) is never left blank.
 *
 * Downloads are made atomic to close the partial/mixed-revision hole: the whole
 * job is pinned to ONE resolved IMMUTABLE commit sha (a mutable ref like a
 * branch name is refused — never pinned); files stage into a JOB-PRIVATE
 * `<modelsDir>/.staging/<slug>@<revision>.<pid>.<uuid>` dir, unique per
 * invocation so no two processes ever write into the same tree, and keyed by the
 * immutable sha so a different revision never reuses another's staged bytes.
 * Resume comes from the HF cache (`downloadFileToCacheDir`), not the staging dir,
 * so the job-private dir is removed on both success and failure. Each staged file
 * is content-verified right after it lands AND the whole staged set is re-verified
 * and pruned to exactly the manifest before publishing; only then is a completion
 * marker written and the staging dir swapped onto the final `<slug>`. An aborted
 * job never leaves a half-populated final dir that catalog state would call
 * "installed".
 *
 * Publishing never destroys a checkpoint the downloader does not own: a final
 * dir WITHOUT our marker (a manual / `mlx download` copy, possibly fine-tuned) is
 * refused — up front (a fast-fail ownership preflight before any download, so no
 * bytes move for a doomed job) and again adjacent to the destructive swap so a dir
 * that races in after the first check is still never overwritten. The one exception
 * is a cleanup (`replaceBlocker`): it replaces the unowned entry the user confirmed, and only while that entry is
 * still the same one (`dev:ino`, kind, link text), checked at the request, at job start, right before the swap and on
 * the moved entry. The old entry is never removed with `rm`: a link is unlinked, anything else goes to the Trash, and
 * when that fails it stays in `.staging/<slug>.kept-<uuid>`.
 * Both swaps go through a temp backup so a crash mid-rename rolls back to the original; an orphaned backup a
 * crash leaves behind is recovered (or reaped) by a best-effort sweep at the
 * start of the next download, and a cleanup's holder also when the manager starts.
 * The old folder of an install that Update adopted is not removed with `rm -rf`: `removeOwnedBackup` deletes only
 * the files its marker's `adopted` record proves and keeps the rest in `.staging/<slug>.kept-<uuid>`.
 *
 * Pure disk + network — never the native addon.
 */

import { createHash, randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import {
  accessSync,
  type BigIntStats,
  constants,
  createReadStream,
  type Dir,
  type Dirent,
  existsSync,
  lstatSync,
  realpathSync,
  statSync,
} from 'node:fs';
import {
  copyFile,
  link,
  lstat,
  mkdir,
  open,
  opendir,
  readdir,
  readlink,
  realpath,
  rename,
  rm,
  rmdir,
  symlink,
  unlink,
  writeFile,
} from 'node:fs/promises';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve, sep } from 'node:path';

import { downloadFileToCacheDir, getRepoFolderName, listFiles, type ListFileEntry, modelInfo } from '@huggingface/hub';
import { type CatalogTarget, catalogTargetForSlug, catalogTargets, catalogUpdateRepos } from '@mlx-node/agent/catalog';
import { isDFlash2Companion, QWEN38_DFLASH2 } from '@mlx-node/lm/draft-companion';

/** How long a resolved set of catalog shas is reused before re-dialling HF. */
const CATALOG_SHA_TTL_MS = 6 * 60 * 60 * 1000;
/** Retry window after a sweep where any repo failed. */
const CATALOG_SHA_NEGATIVE_TTL_MS = 60 * 1000;
/**
 * Deadline for one update probe.
 *
 * Load-bearing, not defensive. Hugging Face connections DO stall rather than
 * reject — one was observed here holding a socket in CLOSE_WAIT for hours at
 * zero CPU. Without a deadline the `Promise.all` below never settles and the
 * route never answers: every such call fails with `E_TIMEOUT` and the update
 * check never completes. A probe must degrade to `null`, never hang the route.
 */
const CATALOG_SHA_PROBE_TIMEOUT_MS = 15 * 1000;

import { catalogBlocker, heldByOtherBuild } from './catalog.js';
import {
  candidateFolders,
  checksumOf,
  digestMatches,
  type FileFingerprint,
  findWanted,
  type FoundFile,
  fingerprintOf,
  foundAt,
  isCommitSha,
  isFolder,
  isInside,
  LOCATE_DISK_DEADLINE_MS,
  LOCATE_SMALL_MAX_BYTES,
  type LocateDownload,
  type LocatePlan,
  type LocateUse,
  locateToken,
  pickMode,
  proveAt,
  readRecord,
  recordInstalled,
  sameFingerprint,
  withDeadline,
} from './locate.js';
import {
  type DownloadCompletion,
  DOWNLOAD_COMPLETE_MARKER,
  type ForeignEntrySummary,
  isDownloaderOwned,
  isLinkableFile,
  isModelFileName,
  isModelInstalled,
  isModelPresent,
  isPathOccupied,
  isSafeRelPath,
  isWeightFile,
  type LocatedRecord,
  modelsChildPath,
  modelsDirUnavailable,
  readCompletion,
  readLinkText,
  sameForeignEntry,
} from './models.js';
import { moveToSystemTrash } from './trash.js';

/** Bounded re-fetch attempts for a staged file that fails post-copy verification. */
const MAX_VERIFY_ATTEMPTS = 3;

/** Job-private staging dir name `<slug>@<40hex-sha>.<pid>.<uuid>` — captures the owner pid. */
const STAGING_DIR_RE = /@[0-9a-f]{40}\.(\d+)\.[0-9a-f-]+$/i;

/** Publish backup dir name `<slug>.backup-<pid>.<uuid>` — captures the owner pid (disjoint from STAGING_DIR_RE). */
const BACKUP_DIR_RE = /\.backup-(\d+)\.[0-9a-f-]+$/i;

/**
 * The folder a cleanup moves the entry the app did not create into, as `<holder>/<slug>`:
 * `<slug>.cleanup-<pid>.<ino>.<uuid>`, where `<ino>` is the inode of the new folder the cleanup renames in. Recovery
 * puts the entry back without the ownership test the owned `.backup-` rule needs, so the two names must stay apart.
 */
const CLEANUP_MARK = '.cleanup-';
/** An old entry the app could not remove or put back: `<slug>.kept-<uuid>`. Nothing ever removes it. */
const KEPT_MARK = '.kept-';

/** The leading decimal digits of `text` up to its first `.`, and the rest after that dot, or `null`. */
function digitsThenDot(text: string): { digits: string; rest: string } | null {
  const dot = text.indexOf('.');
  if (dot <= 0 || dot === text.length - 1) return null;
  const digits = text.slice(0, dot);
  for (const ch of digits) {
    if (ch < '0' || ch > '9') return null;
  }
  return { digits, rest: text.slice(dot + 1) };
}

/** The slug, owner pid and new-folder inode of a `<slug>.cleanup-<pid>.<ino>.<uuid>` name, or `null`. */
function parseCleanupHolder(name: string): { slug: string; pid: number; ino: bigint } | null {
  const at = name.lastIndexOf(CLEANUP_MARK);
  if (at <= 0) return null;
  const pid = digitsThenDot(name.slice(at + CLEANUP_MARK.length));
  if (pid === null) return null;
  const ino = digitsThenDot(pid.rest);
  if (ino === null) return null;
  return { slug: name.slice(0, at), pid: Number(pid.digits), ino: BigInt(ino.digits) };
}

/** Only an `lstat` that answers ENOENT says nothing is at `path`; any other failure says nothing. */
function isGone(path: string): boolean {
  try {
    lstatSync(path);
    return false;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'ENOENT';
  }
}

/**
 * Put an entry a cleanup moved aside back at `finalDir`. A file is linked back and a link is made again from its text,
 * so neither replaces what another program put at `finalDir` (EEXIST); a folder is renamed back, which can replace only
 * an empty folder.
 */
async function putBack(moved: string, finalDir: string): Promise<void> {
  const stat = await lstat(moved);
  if (stat.isDirectory()) {
    await rename(moved, finalDir);
    return;
  }
  if (stat.isSymbolicLink()) await symlink(await readlink(moved), finalDir);
  else await link(moved, finalDir);
  // The entry is back; a second name the unlink leaves stays in `.staging`.
  await unlink(moved).catch(() => {});
}

/**
 * Liveness of a pid via signal 0. An unparseable/invalid pid is treated as ALIVE
 * (never reaped) so a corrupt value can't drive a deletion; a delivered signal or
 * EPERM means alive; only ESRCH proves the process is gone. This is the sole
 * cross-process safety mechanism the sweep relies on to leave a live peer's
 * private staging tree and active publish backup untouched.
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

const DEFAULT_CACHE_DIR = join(homedir(), '.cache', 'huggingface');

const CORE_FILES = new Set([
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.json',
  'merges.txt',
]);

/** Mirrors the no-glob default filter of the CLI download command. */
function isWantedFile(path: string): boolean {
  return (
    CORE_FILES.has(path) ||
    path.endsWith('.safetensors') ||
    path.endsWith('.json') ||
    path.endsWith('.pdiparams') ||
    path.endsWith('.yml') ||
    path.endsWith('.gguf') ||
    path.endsWith('.jinja')
  );
}

// Glob matching below is byte-identical to the CLI's (`globToRegex` /
// `matchesAnyGlob` in `packages/cli/src/commands/download-model.ts`). The CLI
// does not export them and the dashboard must not depend on the CLI package,
// so the mirror lives here — keep the two in sync: a divergence would make
// the wizard's `mlx download model -g …` select a different file set than a
// dashboard install of the same catalog entry.
/** Convert a simple glob pattern (with * wildcards) to a RegExp. */
function globToRegex(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
  return new RegExp(`^${escaped}$`, 'i');
}

/** Check if a filename matches any of the glob patterns. */
function matchesAnyGlob(filename: string, patterns: RegExp[]): boolean {
  return patterns.some((re) => re.test(filename));
}

/**
 * The manifest filter for one job: the catalog entry's `globs` when it carries
 * them (multi-variant GGUF repos — match on basename OR full repo path, core
 * metadata always included, exactly the CLI's `--glob` semantics), else the
 * no-glob default.
 */
function isWantedFileFor(path: string, globs: RegExp[] | undefined): boolean {
  if (globs === undefined) return isWantedFile(path);
  const basename = path.split('/').pop() ?? path;
  return matchesAnyGlob(basename, globs) || matchesAnyGlob(path, globs) || CORE_FILES.has(path);
}

/**
 * Sidecar files a GGUF quantization repo typically lacks, fetched from the
 * catalog entry's `assetsRepo` (the base model's repo). Names absent from the
 * assetsRepo are skipped, so one fixed list serves every model family. See
 * {@link CatalogBuild.assetsRepo} for why these are mandatory for tool
 * calling rather than optional metadata.
 */
const ASSET_SIDECAR_CANDIDATES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'chat_template.jinja',
  'generation_config.json',
  'preprocessor_config.json',
  'video_preprocessor_config.json',
  'processor_config.json',
] as const;

/** Membership test for the candidate list above (insertion order is the fetch order). */
const ASSET_SIDECAR_CANDIDATE_SET: ReadonlySet<string> = new Set(ASSET_SIDECAR_CANDIDATES);

/**
 * True when the manifest actually describes a loadable model: it must carry a
 * `config.json` AND at least one weight file. A repo that resolves to a
 * one-sided set (config-only or weights-only — e.g. a catalog repo caught
 * mid-re-upload with the config committed before the weights) must error rather
 * than publish a hollow, one-sided "installed" directory that `loadModel` would
 * then fail to open while the catalog reports it Installed with retry disabled.
 *
 * When the job's selection names an `assetsRepo` FOR THIS REPO, the config
 * half is satisfied by the sidecar fetch that runs before publish (GGUF
 * quantization repos ship weights only), so the manifest itself only needs to
 * prove a weight payload. A platform override repo carries no such selection
 * even when the entry's GGUF build has one.
 */
function hasModelPayload(files: ListFileEntry[], waivesConfigRequirement: boolean): boolean {
  const hasConfig = waivesConfigRequirement || files.some((file) => file.path === 'config.json');
  return hasConfig && files.some((file) => isWeightFile(file.path));
}

function assertModelPayload(repoName: string, files: ListFileEntry[], assetsRepo: string | undefined): void {
  if (hasModelPayload(files, assetsRepo !== undefined)) return;
  const need =
    assetsRepo !== undefined
      ? 'at least one weight file (config.json comes from the assetsRepo sidecar fetch)'
      : 'both a config.json and a weight file';
  throw new Error(`Repo "${repoName}" is not a complete model (needs ${need})`);
}

/** The tokenizer/config sidecars one job fetches from the entry's `assetsRepo`, pinned to one commit. */
type SidecarPlan = { repo: { type: 'model'; name: string }; revision: string; files: ListFileEntry[] };

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
 * Resume check for a file already staged on disk. Verifies CONTENT identity, not
 * merely size, so a stale/truncated-but-right-size file is never trusted:
 *
 *   - size must match the manifest (fast gate, catches truncation);
 *   - if the manifest advertises `lfs.oid` (the LFS sha256 of the content), the
 *     file's sha256 must match it. This is the branch every weight takes, Xet
 *     included: a Xet-backed entry carries `oid`, `lfs.oid` AND `xetHash` at once,
 *     and its `lfs.oid` is a plain sha256 of the bytes — so Xet files are content-
 *     verified here, not size-checked;
 *   - else, for a plain file (no LFS, no Xet), the top-level git `oid` is the
 *     git-blob sha1 of the actual content, so the file's git-blob sha1 must match
 *     it. The `xetHash` clause excludes only the hypothetical Xet-without-LFS
 *     entry, whose `oid` would hash the POINTER rather than the content — it is
 *     not a statement that Xet files go unverified;
 *   - otherwise (no usable hash) fall back to size alone. No real catalog entry
 *     reaches this: across the API responses for the catalog repos every file
 *     lands on one of the two branches above. The runner also cross-checks the
 *     full manifest before writing the completion marker, so a partial set can
 *     never be published as complete.
 *
 * `xetHash` is deliberately never recomputed: it is a Merkle/chunk hash over the
 * Xet chunking, not a digest of the file, so verifying against it would mean
 * reimplementing that chunker for no gain over the sha256 already checked.
 */
async function isStagedCopyComplete(destPath: string, file: ListFileEntry): Promise<boolean> {
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

/** Recursive relative (posix) paths of every regular file under `dir`; empty if absent. */
async function listStagedFiles(dir: string): Promise<string[]> {
  let raw: string[];
  try {
    raw = await readdir(dir, { recursive: true });
  } catch {
    return [];
  }
  const out: string[] = [];
  for (const rel of raw) {
    try {
      if (statSync(join(dir, rel)).isFile()) out.push(rel.split(sep).join('/'));
    } catch {
      // Vanished/unreadable entry: skip.
    }
  }
  return out;
}

/**
 * Remove the entry at `full`, which a check just found to be one the caller may delete, without unlinking by name:
 * a writer could rename another file onto the name between that check and the unlink. The entry is renamed to `aside`
 * (a private path) and `owned` checks it there, where no path of the folder reaches it. An entry it rejects goes
 * back with `link`, which never overwrites (a folder with `rename`). Returns `gone` when the name no longer holds the
 * owned entry, `stays` when nothing was moved, `restored` when a rejected entry is back at its name, and `aside` when
 * something is left at `aside`.
 */
async function unlinkIfStillOwned(
  full: string,
  aside: string,
  owned: (stat: BigIntStats) => boolean,
): Promise<'gone' | 'stays' | 'restored' | 'aside'> {
  try {
    await mkdir(dirname(aside), { recursive: true });
  } catch {
    return 'stays';
  }
  try {
    await rename(full, aside);
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'ENOENT' ? 'gone' : 'stays';
  }
  let stat: BigIntStats;
  try {
    stat = await lstat(aside, { bigint: true });
  } catch {
    return 'aside';
  }
  if (owned(stat)) {
    try {
      await unlink(aside);
      return 'gone';
    } catch {
      return 'aside';
    }
  }
  try {
    if (stat.isDirectory()) {
      await rename(aside, full);
      return 'restored';
    }
    if (!stat.isFile()) return 'aside';
    await link(aside, full);
  } catch {
    return 'aside';
  }
  try {
    await unlink(aside);
    return 'restored';
  } catch {
    return 'aside';
  }
}

/** Remove `dir` and every folder under it that holds no file, bottom up. Returns whether `dir` is still there. */
async function removeEmptyDirs(dir: string): Promise<boolean> {
  let entries: Dirent[];
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch (error) {
    return (error as NodeJS.ErrnoException).code !== 'ENOENT';
  }
  for (const entry of entries) if (entry.isDirectory()) await removeEmptyDirs(join(dir, entry.name));
  try {
    await rmdir(dir);
    return false;
  } catch {
    return true;
  }
}

/**
 * Unlink each regular file under `dir` that the marker lists and whose `lstat` size and `mtimeNs` still equal its
 * `adopted` record, then remove the sub-folders left empty. Each file is moved to `<aside>/<path>` and checked there
 * before the unlink ({@link unlinkIfStillOwned}); a file that cannot go back to its name stays under `aside`. Anything
 * else stays: other files, links, special files, and the root marker, which must stay while anything else does so
 * that a second pass reads the same record. A file that cannot be read or unlinked stays too.
 */
async function removeProvenFiles(
  dir: string,
  prefix: string,
  listed: ReadonlySet<string>,
  adopted: Readonly<Record<string, { size: number; mtimeNs: string }>>,
  aside: string,
): Promise<void> {
  const names: string[] = [];
  try {
    for await (const entry of await opendir(dir)) names.push(entry.name);
  } catch {
    return;
  }
  for (const name of names) {
    const rel = prefix === '' ? name : `${prefix}/${name}`;
    if (rel === DOWNLOAD_COMPLETE_MARKER) continue;
    const full = join(dir, name);
    let stat;
    try {
      stat = await lstat(full, { bigint: true });
    } catch {
      continue;
    }
    if (stat.isDirectory()) {
      await removeProvenFiles(full, rel, listed, adopted, aside);
      await rmdir(full).catch(() => {});
      continue;
    }
    if (!stat.isFile() || !listed.has(rel) || !Object.hasOwn(adopted, rel)) continue;
    const record = adopted[rel];
    const proven = (s: BigIntStats): boolean =>
      s.isFile() && s.size === BigInt(record.size) && s.mtimeNs.toString() === record.mtimeNs;
    if (!proven(stat)) continue;
    const checked = stat;
    await unlinkIfStillOwned(
      full,
      join(aside, rel),
      (moved) => moved.dev === checked.dev && moved.ino === checked.ino && proven(moved),
    );
  }
}

/**
 * Best-effort `fsync` of a directory fd so a rename into it is durable before the
 * backup is removed. Filesystems that reject a directory fsync are tolerated —
 * durability is a nicety here, not a correctness gate.
 */
async function fsyncDir(dir: string): Promise<void> {
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    handle = await open(dir, 'r');
    await handle.sync();
  } catch {
    // Directory fsync unsupported on this platform/filesystem: ignore.
  } finally {
    await handle?.close();
  }
}

/**
 * Refuse a path that already exists but is a symlink or a non-directory; a
 * missing path is allowed (the caller then creates a real directory there).
 * `mkdir(..., { recursive })` FOLLOWS a pre-existing symlink, so without this a
 * planted `.staging` → external dir would silently redirect every staged write
 * and every recursive cleanup onto that foreign target. Same no-follow `lstat`
 * guard as `models.ts` `deleteLocalModel` and `cache.ts`.
 */
function assertRealDirOrAbsent(path: string): void {
  let stat;
  try {
    stat = lstatSync(path);
  } catch {
    return; // absent — safe to mkdir a real directory here
  }
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(
      `Refusing to use staging root "${path}": it must be a real directory (found a symlink or non-directory)`,
    );
  }
}

export type DownloadEvent =
  | { type: 'start'; id: string; repo: string; totalBytes: number; fileCount: number }
  | {
      type: 'progress';
      id: string;
      file: string;
      /** Bytes of THIS file so far. */
      receivedBytes: number;
      /**
       * Bytes of the WHOLE job so far — every already-finished file plus this
       * one. Absolute, not a delta, because the frame has to stand alone: only
       * the latest `progress` is replayed on attach (and the SSE sender
       * coalesces a backlog down to that same one frame), so a subscriber that
       * joins mid-download never witnesses the finished files' frames and
       * cannot recover the job total by summing what it received.
       */
      jobReceivedBytes: number;
      /** Size of THIS file; the job total travels on the `start` frame. */
      totalBytes: number;
      fileIndex: number;
      fileCount: number;
      /** Set while Update hashes a file that is already in the folder; absent while a file downloads. */
      phase?: 'check';
    }
  | {
      type: 'done';
      id: string;
      outputDir: string;
      /**
       * Where the old entry, or the files of the old folder that the app may not delete, were moved; set only when
       * there were some. A cleanup keeps its old entry there as `<keptDir>/<slug>` when it could not remove it.
       */
      keptDir?: string;
      /** How a cleanup removed the old entry: `link` unlinked it, `trash` moved it to the Trash. */
      removed?: 'link' | 'trash';
    }
  | { type: 'error'; id: string; message: string }
  | { type: 'cancelled'; id: string };

/** Cap on the failure text an `error` event carries. */
const MAX_ERROR_MESSAGE_CHARS = 2048;

/**
 * Bound the failure text of an `error` event. `@huggingface/hub` copies a remote
 * JSON error body straight into `Error.message`, so this string is unbounded and
 * shaped by the far end — and the event is retained in `lastEvent` and replayed
 * to every SSE subscriber that attaches afterwards.
 */
function truncateMessage(message: string): string {
  return message.length > MAX_ERROR_MESSAGE_CHARS ? `${message.slice(0, MAX_ERROR_MESSAGE_CHARS)}…` : message;
}

/**
 * The user-facing failure text for a job error. undici reports EVERY transport
 * failure as `TypeError: fetch failed` and hides the reason in `error.cause`
 * (ENETUNREACH, ETIMEDOUT, ECONNRESET, SELF_SIGNED_CERT_IN_CHAIN, EPERM …), so
 * without unwrapping, a blocked network, a TLS-inspecting proxy, and a dead DNS
 * all render as the same useless "fetch failed". Walk the cause chain and
 * append each link's code, so the toast names the actual reason.
 */
function describeFailure(error: unknown): string {
  if (!(error instanceof Error)) return String(error);
  const parts: string[] = [error.message];
  const seen = new Set<unknown>([error]);
  let current: unknown = error.cause;
  while (current instanceof Error && !seen.has(current)) {
    seen.add(current);
    const code = (current as NodeJS.ErrnoException).code;
    parts.push(code !== undefined ? `${current.message} [${code}]` : current.message);
    current = current.cause;
  }
  return parts.join(' → ');
}

/**
 * Thrown by {@link DownloadManager.start} once {@link DownloadManager.shutdown}
 * has begun. Distinct from the catalog rejection (a bad request) so the API can
 * answer 503 — the repo is fine, the server is just going away.
 */
export class DownloadsClosedError extends Error {
  constructor() {
    super('The dashboard is shutting down; not accepting new downloads');
    this.name = 'DownloadsClosedError';
  }
}

/** The models directory is a link whose target is missing ({@link modelsDirUnavailable}). The API answers 503. */
export class ModelsDirUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelsDirUnavailableError';
  }
}

/** Throw {@link ModelsDirUnavailableError} when the models directory is a link to a missing target. */
function assertModelsDirAvailable(modelsDir: string): void {
  const unavailable = modelsDirUnavailable(modelsDir);
  if (unavailable !== null) throw new ModelsDirUnavailableError(unavailable);
}

/** Locate could not read the catalog's file list from Hugging Face. The API answers 503. */
export class HubUnreachableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'HubUnreachableError';
  }
}

/**
 * A request that cannot be accepted as asked: another job holds the repo, the folder is not what the user confirmed,
 * or Update cannot use it. The API answers 409, with {@link reason} beside the message: `needs-cleanup` says that
 * Clean up & update can install the model where Update cannot.
 */
export class DownloadConflictError extends Error {
  readonly reason: 'needs-cleanup' | undefined;

  constructor(message: string, reason?: 'needs-cleanup') {
    super(message);
    this.name = 'DownloadConflictError';
    this.reason = reason;
  }
}

/** The catalog build or draft that installs to the folder `slug`; any other folder is refused. */
function catalogTarget(slug: string): CatalogTarget {
  const target = catalogTargetForSlug(slug);
  if (target === undefined) throw new Error(`"${slug}" is not a folder of the model catalog`);
  return target;
}

/** The globs and sidecar source a job of `target` plans; a draft has neither. */
function selectionOf(target: CatalogTarget): { globs?: readonly string[]; assetsRepo?: string } {
  if (target.kind === 'draft') return {};
  return {
    ...(target.globs !== undefined ? { globs: target.globs } : {}),
    ...(target.assetsRepo !== undefined ? { assetsRepo: target.assetsRepo } : {}),
  };
}

/** What the user confirmed about the entry in the way, echoed by `POST /api/downloads`. */
export type ReplaceBlocker = Pick<ForeignEntrySummary, 'kind' | 'fileCount' | 'sizeBytes' | 'truncated' | 'linkTarget'>;

/** The entry a cleanup replaces: the server's summary at the request, and the `lstat` identity it must keep. */
interface CleanupTarget {
  summary: ForeignEntrySummary;
  identity: EntryIdentity;
}

interface EntryIdentity {
  dev: bigint;
  ino: bigint;
}

/** What a job is, for the page: a plain install, an Update in place, a cleanup, or a Locate. */
export type DownloadJobKind = 'install' | 'update' | 'cleanup' | 'locate';

const NOTHING_REMOVED = 'Nothing was removed.';

/** `lstat` identity of the entry itself, never followed, or `null` when nothing is there. */
function entryIdentity(path: string): EntryIdentity | null {
  try {
    const stat = lstatSync(path, { bigint: true });
    return { dev: stat.dev, ino: stat.ino };
  } catch {
    return null;
  }
}

function sameIdentity(a: EntryIdentity | null, b: EntryIdentity): boolean {
  return a !== null && a.dev === b.dev && a.ino === b.ino;
}

/** The failure text of a cleanup that stopped before the swap: it says the entry in the way is unchanged. */
function withNothingRemoved(message: string): string {
  return message.endsWith('.') ? `${message} ${NOTHING_REMOVED}` : `${message}. ${NOTHING_REMOVED}`;
}

function isRealDirectory(path: string): boolean {
  try {
    return lstatSync(path).isDirectory();
  } catch {
    return false;
  }
}

function canWrite(path: string): boolean {
  try {
    accessSync(path, constants.W_OK);
    return true;
  } catch {
    return false;
  }
}

/**
 * The entry at `finalDir` is still the one this cleanup was accepted for: the same `dev:ino` and kind, and a link
 * still has the same text. Nothing in it is listed or opened.
 */
function assertStillReplaceable(kept: CleanupTarget, finalDir: string, slug: string): void {
  let stat: BigIntStats | undefined;
  try {
    stat = lstatSync(finalDir, { bigint: true });
  } catch {
    stat = undefined;
  }
  const kind =
    stat === undefined ? null : stat.isSymbolicLink() ? 'symlink' : stat.isDirectory() ? 'directory' : 'file';
  const same =
    stat !== undefined &&
    sameIdentity({ dev: stat.dev, ino: stat.ino }, kept.identity) &&
    kind === kept.summary.kind &&
    (kind !== 'symlink' || readLinkText(finalDir) === kept.summary.linkTarget);
  if (!same) throw new Error(`"${slug}" changed on disk during the download.`);
}

const NOTHING_CHANGED = 'Nothing was changed.';

/** The failure text of an Update that stopped before the folder gained a file: it says the folder is as it was. */
function withNothingChanged(message: string): string {
  if (message.includes(NOTHING_CHANGED)) return message;
  return message.endsWith('.') ? `${message} ${NOTHING_CHANGED}` : `${message}. ${NOTHING_CHANGED}`;
}

/** A catalog target that is a build, the only kind Locate takes. */
type BuildTarget = Extract<CatalogTarget, { kind: 'build' }>;

const LOCATE_QUEUED = 'A download of this model is already in the queue. Cancel it, then use Locate.';
const LOCATE_CHANGED = 'The folder changed after you confirmed. Use Locate again.';
const LOCATE_UNREADABLE = 'The app cannot read this folder. Pick the folder that holds the model files.';
const LOCATE_SLOW =
  `The app could not read this folder in ${LOCATE_DISK_DEADLINE_MS / 1000} seconds. ` +
  'If macOS asks for access to the disk, allow it, then use Locate again.';
const LOCATE_INSIDE_MODELS = 'This folder is inside the models folder. The app already lists the models there.';
const LOCATE_OTHER_NAMES =
  'These files are on another disk than the models folder, and they are stored there under other names ' +
  '(a Hugging Face cache does this). The app can use them only from the disk of the models folder. ' +
  'Use Install, or copy the folder to that disk and use Locate again.';

function locateNotBesideMessage(name: string): string {
  return (
    `These files are on another disk than the models folder, and "${name}" is not beside them. ` +
    'The app does not add files to your folder. ' +
    'Use Install, or copy the folder to the disk of the models folder and use Locate again.'
  );
}

function locateInCacheMessage(name: string): string {
  return `This folder is part of the app's download cache, and "${name}" is missing or different there. Use Install.`;
}

function locateHoldsCacheMessage(name: string): string {
  return `The app's download cache is inside this folder, and "${name}" must be downloaded. Use Install.`;
}

/** Where a write to `path` lands: the real path of its deepest existing part, then the rest of `path`. */
async function realWritePath(path: string): Promise<string> {
  const rest: string[] = [];
  for (let head = path; ; head = dirname(head)) {
    try {
      return join(await realpath(head), ...rest);
    } catch {
      if (dirname(head) === head) return path;
      rest.unshift(basename(head));
    }
  }
}

function locateOtherBuildMessage(build: string): string {
  return `This folder holds ${build}. Choose ${build} on the card, then use Locate again.`;
}

function locateTwoFoldersMessage(label: string, a: string, b: string): string {
  return `Two folders here hold ${label}: "${basename(a)}" and "${basename(b)}". Pick one of them.`;
}

/** The body of `POST /api/downloads/locate-plan`: what the confirmation of a Locate says. */
export interface LocatePlanSummary {
  slug: string;
  repo: string;
  token: string;
  /** The real path of the folder whose files the app uses. */
  folder: string;
  /** `shared`: the models folder shares these files on disk. `linked`: it links to them on another disk. */
  mode: 'shared' | 'linked';
  /** The folder has this app's install record, so the job trusts it and reads no file. */
  byRecord: boolean;
  revision: string;
  /** Every file of the folder that the app uses. */
  use: { count: number; bytes: number };
  /** The weight files the job reads once to check their content; zeros when {@link byRecord}. */
  read: { count: number; bytes: number };
  /** The files the job downloads into the models folder. */
  download: { count: number; bytes: number };
}

/** The `dev` of `path`, or of its nearest folder that exists: the disk a new folder there lands on. */
function deviceOf(path: string): bigint {
  let at = resolve(path);
  for (;;) {
    try {
      return statSync(at, { bigint: true }).dev;
    } catch (error) {
      const up = dirname(at);
      if (up === at) throw error;
      at = up;
    }
  }
}

/** A record Locate may trust for the catalog repo `repo`: a full install with a pinned revision and safe paths. */
function isLocatableRecord(record: DownloadCompletion, repo: string): boolean {
  return (
    record.repo.toLowerCase() === repo.toLowerCase() &&
    record.scope !== 'partial' &&
    isCommitSha(record.revision) &&
    record.files.length <= 200 &&
    record.files.every((path) => isSafeRelPath(path) && path !== DOWNLOAD_COMPLETE_MARKER)
  );
}

/** Whether `target`'s file filter takes `path`. */
function buildTakes(target: BuildTarget, path: string): boolean {
  return isWantedFileFor(path, target.globs?.map(globToRegex));
}

/** Entries {@link scanFolder} reads before it refuses a folder for Update. */
const ADOPT_SCAN_MAX_ENTRIES = 2_000;

/** Shortest gap between two `progress` frames of one file that Update hashes in the folder. */
const CHECK_PROGRESS_INTERVAL_MS = 100;

/** One entry of a folder Update looks at, by its posix path inside the folder. */
interface FolderEntry {
  path: string;
  isDirectory: boolean;
  stat: FileFingerprint;
}

/** A file in the folder that Update proves by hash, with the `lstat` the plan saw. */
interface AdoptCheck {
  file: ListFileEntry;
  size: number;
  mtimeNs: bigint;
  ino: bigint;
}

/** A wanted file, with the repo and pinned revision it comes from. */
interface AdoptWanted {
  file: ListFileEntry;
  repo: string;
  revision: string;
}

/** A plan of Update for one folder: what the job checks, downloads and leaves, and the token that names it. */
interface AdoptPlan {
  token: string;
  revision: string;
  assetsRepo: string | null;
  assetsRevision: string | null;
  folder: { dev: bigint; ino: bigint };
  /** Every file the catalog entry installs, primary files first, in manifest order. */
  wanted: AdoptWanted[];
  check: AdoptCheck[];
  download: AdoptWanted[];
  other: { count: number; bytes: number; names: string[] };
}

/** The body of `POST /api/downloads/adopt-plan`. */
export interface AdoptPlanSummary {
  repo: string;
  slug: string;
  token: string;
  revision: string;
  assetsRevision: string | null;
  check: { count: number; bytes: number };
  download: { count: number; bytes: number };
  other: { count: number; bytes: number; names: string[] };
}

/** A file whose hash did not match the catalog, as it was when it was hashed. */
interface MismatchRecord {
  size: number;
  mtimeNs: bigint;
  ino: bigint;
  oid: string;
}

function changedOnDiskMessage(slug: string): string {
  return `"${slug}" changed on disk. Check it, then try again.`;
}

/**
 * Every entry under `dir` by `lstat`, without opening a file or following a link. A link or a special file, or more
 * than {@link ADOPT_SCAN_MAX_ENTRIES} entries, refuses the folder.
 */
async function scanFolder(
  dir: string,
  slug: string,
): Promise<{ folder: { dev: bigint; ino: bigint }; entries: FolderEntry[] }> {
  const isGone = (error: unknown): boolean => (error as NodeJS.ErrnoException).code === 'ENOENT';
  let root: BigIntStats;
  try {
    root = await lstat(dir, { bigint: true });
  } catch (error) {
    if (isGone(error)) throw new DownloadConflictError(changedOnDiskMessage(slug));
    throw error;
  }
  if (!root.isDirectory()) throw new DownloadConflictError(changedOnDiskMessage(slug));
  const entries: FolderEntry[] = [];
  const pending: string[] = [''];
  // An entry that goes away during the scan is left out, not refused: a wanted path that is gone then reads as a
  // download, which classifyFolder and the commit barrier compare against the plan.
  while (pending.length > 0) {
    const prefix = pending.pop()!;
    let handle: Dir;
    try {
      handle = await opendir(prefix === '' ? dir : join(dir, prefix));
    } catch (error) {
      if (prefix !== '' && isGone(error)) continue;
      throw error;
    }
    for await (const dirent of handle) {
      if (entries.length >= ADOPT_SCAN_MAX_ENTRIES) {
        throw new DownloadConflictError(
          `"${slug}" holds more than ${ADOPT_SCAN_MAX_ENTRIES.toLocaleString('en-US')} files. Update cannot check it.`,
          'needs-cleanup',
        );
      }
      const path = prefix === '' ? dirent.name : `${prefix}/${dirent.name}`;
      let stat: BigIntStats;
      try {
        stat = await lstat(join(dir, path), { bigint: true });
      } catch (error) {
        if (isGone(error)) continue;
        throw error;
      }
      if (stat.isDirectory()) pending.push(path);
      else if (!stat.isFile()) {
        throw new DownloadConflictError(
          `"${slug}" holds links or special files. Update cannot check it.`,
          'needs-cleanup',
        );
      }
      entries.push({ path, isDirectory: stat.isDirectory(), stat: fingerprintOf(stat) });
    }
  }
  entries.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  return { folder: { dev: root.dev, ino: root.ino }, entries };
}

function notCatalogFileMessage(path: string, slug: string): string {
  return `"${path}" in "${slug}" is not the catalog file. Move it out of the folder, then use Update again.`;
}

/**
 * Sort the folder's entries into the files Update checks by hash, the wanted files it downloads, and the other files
 * it leaves where they are, or refuse the folder with a {@link DownloadConflictError} that names the file. The plan and
 * the job both run this, so they cannot disagree.
 */
function classifyFolder(
  entries: readonly FolderEntry[],
  wanted: readonly AdoptWanted[],
  memo: ReadonlyMap<string, MismatchRecord>,
  slug: string,
): { check: AdoptCheck[]; download: AdoptWanted[]; other: { count: number; bytes: number; names: string[] } } {
  const wantedPaths = new Set(wanted.map((item) => item.file.path));
  const wantedByLower = new Map(wanted.map((item) => [item.file.path.toLowerCase(), item.file.path]));
  const byPath = new Map(entries.map((entry) => [entry.path, entry]));
  const otherNames: string[] = [];
  let otherBytes = 0;
  for (const entry of entries) {
    if (wantedPaths.has(entry.path)) {
      if (entry.isDirectory) throw new DownloadConflictError(notCatalogFileMessage(entry.path, slug), 'needs-cleanup');
      continue;
    }
    if (entry.isDirectory) continue;
    if (entry.path === DOWNLOAD_COMPLETE_MARKER) {
      throw new DownloadConflictError(
        `"${slug}" holds a download record that the app cannot read (${DOWNLOAD_COMPLETE_MARKER}). ` +
          `Move it out of the folder, then use Update again.`,
        'needs-cleanup',
      );
    }
    // On a case-insensitive volume the link of the catalog name would land on this file.
    const catalogName = wantedByLower.get(entry.path.toLowerCase());
    if (catalogName !== undefined) {
      throw new DownloadConflictError(
        `"${entry.path}" in "${slug}" differs from the catalog name "${catalogName}" only in letter case. ` +
          `Rename it, then use Update again.`,
        'needs-cleanup',
      );
    }
    if (isModelFileName(basename(entry.path))) {
      throw new DownloadConflictError(
        `"${entry.path}" in "${slug}" is a model file that is not part of the catalog model. ` +
          `Move it out of the folder, then use Update again.`,
        'needs-cleanup',
      );
    }
    otherNames.push(entry.path);
    otherBytes += Number(entry.stat.size);
  }
  const check: AdoptCheck[] = [];
  const download: AdoptWanted[] = [];
  for (const item of wanted) {
    const entry = byPath.get(item.file.path);
    if (entry === undefined) {
      // The staged copy of a file with no digest would be checked by its size alone.
      if (checksumOf(item.file) === null) {
        throw new DownloadConflictError(
          `The catalog gives no checksum for "${item.file.path}", so Update cannot check what it downloads.`,
          'needs-cleanup',
        );
      }
      download.push(item);
      continue;
    }
    const size = Number(entry.stat.size);
    if (size !== item.file.size) {
      throw new DownloadConflictError(
        `"${item.file.path}" in "${slug}" is not the catalog file. Its size is different. ` +
          `Move it out of the folder, then use Update again. ` +
          `Update then downloads the catalog file (${item.file.size.toLocaleString('en-US')} bytes).`,
        'needs-cleanup',
      );
    }
    // A same-size file is never taken on its size: only its hash proves it.
    const sum = checksumOf(item.file);
    if (sum === null) {
      throw new DownloadConflictError(
        `The catalog gives no checksum for "${item.file.path}", so Update cannot check it. ` +
          `Move it out of the folder, then use Update again.`,
        'needs-cleanup',
      );
    }
    const failed = memo.get(`${slug}\0${item.file.path}`);
    if (
      failed !== undefined &&
      failed.size === size &&
      failed.mtimeNs === entry.stat.mtimeNs &&
      failed.ino === entry.stat.ino &&
      failed.oid === sum.oid
    ) {
      throw new DownloadConflictError(notCatalogFileMessage(item.file.path, slug), 'needs-cleanup');
    }
    check.push({ file: item.file, size, mtimeNs: entry.stat.mtimeNs, ino: entry.stat.ino });
  }
  return { check, download, other: { count: otherNames.length, bytes: otherBytes, names: otherNames.slice(0, 3) } };
}

/**
 * The token of an Update plan: a sha256 over what the job must find again before it may touch the folder. The other
 * files are left out on purpose, so a `.DS_Store` that Finder writes does not break the confirm.
 */
function adoptToken(plan: {
  repo: string;
  revision: string;
  assetsRepo: string | null;
  assetsRevision: string | null;
  wanted: readonly AdoptWanted[];
  check: readonly AdoptCheck[];
  download: readonly AdoptWanted[];
  folder: { dev: bigint; ino: bigint };
}): string {
  const byFirst = (a: unknown[], b: unknown[]): number => {
    const [x, y] = [String(a[0]), String(b[0])];
    return x < y ? -1 : x > y ? 1 : 0;
  };
  const body = {
    repo: plan.repo,
    revision: plan.revision,
    assetsRepo: plan.assetsRepo,
    assetsRevision: plan.assetsRevision,
    wanted: plan.wanted
      .map((item) => [item.file.path, item.file.size, checksumOf(item.file)?.oid ?? null])
      .sort(byFirst),
    check: plan.check
      .map((item) => [item.file.path, item.size, item.mtimeNs.toString(), item.ino.toString()])
      .sort(byFirst),
    download: plan.download.map((item) => item.file.path).sort(),
    folder: `${plan.folder.dev}:${plan.folder.ino}`,
  };
  return createHash('sha256').update(JSON.stringify(body)).digest('hex');
}

/**
 * Hash a file in the folder where it is, and say whether its bytes are the catalog file. The fd is opened without
 * following a link, and its `fstat` must equal `expect` before the read and be unchanged after it, so bytes that were
 * rewritten during the read are never taken as proven. Throws when `hooks.cancelled()` turns true between chunks.
 */
export async function proveInPlace(
  dir: string,
  file: ListFileEntry,
  expect: { size: number; mtimeNs: bigint; ino: bigint },
  hooks: { cancelled(): boolean; onBytes(n: number): void },
): Promise<{ match: boolean; fingerprint: FileFingerprint }> {
  return proveAt(
    join(dir, file.path),
    file,
    expect,
    hooks,
    () =>
      new Error(`"${file.path}" in "${basename(dir)}" changed during the check. ${NOTHING_CHANGED} Use Update again.`),
    () => new Error(`The catalog gives no checksum for "${file.path}", so Update cannot check it.`),
  );
}

/** A name that Update added to the folder: a second name of the staged file with this `dev:ino`. */
interface AddedName {
  path: string;
  dev: bigint;
  ino: bigint;
}

/**
 * Add the staged file `from` to the folder `dir` at `path`. Only `link` may do this: it fails with EEXIST when
 * anything is at the path, so a file that the user put there is never overwritten. A parent folder that is missing is
 * made with a plain `mkdir` and recorded in `madeDirs`; the name is recorded in `added` as soon as it exists.
 */
async function linkInto(
  dir: string,
  slug: string,
  path: string,
  from: string,
  staged: FileFingerprint,
  added: AddedName[],
  madeDirs: string[],
): Promise<void> {
  const changed = (): Error => new Error(`"${slug}" changed on disk during the update.`);
  const segments = path.split('/');
  let parent = dir;
  for (const segment of segments.slice(0, -1)) {
    parent = join(parent, segment);
    try {
      await mkdir(parent);
      madeDirs.push(parent);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
      if (!(await lstat(parent)).isDirectory()) throw changed();
    }
  }
  const to = join(dir, path);
  try {
    await link(from, to);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === 'EEXIST') throw new Error(`"${path}" appeared in "${slug}" during the update.`);
    throw new Error(
      `The app could not add "${path}" to "${slug}" (${code ?? String(error)}). Update needs a volume that can link files.`,
    );
  }
  added.push({ path, dev: staged.dev, ino: staged.ino });
  const now = await lstat(to, { bigint: true });
  if (!now.isFile() || now.dev !== staged.dev || now.ino !== staged.ino) throw changed();
}

/**
 * Undo the adds of an Update that failed, newest first. A name is removed only while `lstat` shows the job's own
 * staged file there, and through {@link unlinkIfStillOwned} with `<keptDir>/<path>` as its private path, so a file
 * renamed onto the name meanwhile is put back, or left in `keptDir` when its name was taken again. Whatever else took
 * the path is not the job's and stays. When `config.json` or the marker cannot be removed, every name not yet reached
 * stays as well: a config.json without the files added beside it reads as installed and missing its tokenizer.
 * Folders the job made are removed with `rmdir`, so one that something came into stays too. Returns the names of the
 * job's files that are still there, and `keptDir` when anything is left in it.
 */
async function removeAddedNames(
  dir: string,
  added: readonly AddedName[],
  madeDirs: readonly string[],
  keptDir: string,
): Promise<{ stays: string[]; kept?: string }> {
  const stays: string[] = [];
  const newestFirst = [...added].reverse();
  for (let index = 0; index < newestFirst.length; index++) {
    const name = newestFirst[index];
    const full = join(dir, name.path);
    let stat: BigIntStats | null = null;
    let removed = true;
    try {
      stat = await lstat(full, { bigint: true });
    } catch (error) {
      removed = (error as NodeJS.ErrnoException).code === 'ENOENT';
    }
    if (stat !== null && stat.isFile() && stat.dev === name.dev && stat.ino === name.ino) {
      const result = await unlinkIfStillOwned(
        full,
        join(keptDir, name.path),
        (moved) => moved.isFile() && moved.dev === name.dev && moved.ino === name.ino,
      );
      removed = result !== 'stays';
    }
    if (removed) continue;
    if (name.path === 'config.json' || name.path === DOWNLOAD_COMPLETE_MARKER) {
      stays.push(...newestFirst.slice(index).map((rest) => rest.path));
      return { stays, ...((await removeEmptyDirs(keptDir)) ? { kept: keptDir } : {}) };
    }
    stays.push(name.path);
  }
  for (const made of [...madeDirs].reverse()) await rmdir(made).catch(() => {});
  return { stays, ...((await removeEmptyDirs(keptDir)) ? { kept: keptDir } : {}) };
}

/** `fsync` of a regular file, so its bytes are on disk before a name of it is added to the folder. */
async function fsyncFile(path: string): Promise<void> {
  const handle = await open(path, 'r');
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
}

/**
 * `committing` is the brief, non-cancellable window from just before `publish`
 * until the job settles: `cancel()`'s `state !== 'running'` gate rejects it, so a
 * cancel that arrives once the atomic swap has begun can no longer report success
 * while the model still installs.
 */
type JobStatus = 'running' | 'committing' | 'done' | 'error' | 'cancelled';

/** A settled job: it holds no in-flight work and no longer owns its folder. */
function isTerminal(state: JobStatus): boolean {
  return state === 'done' || state === 'error' || state === 'cancelled';
}

function jobKind(job: Pick<JobState, 'adopt' | 'replace' | 'locate'>): DownloadJobKind {
  if (job.adopt !== null) return 'update';
  if (job.replace !== null) return 'cleanup';
  if (job.locate !== null) return 'locate';
  return 'install';
}

interface JobState {
  id: string;
  /** The folder the job installs to under the models directory: the key of a job. */
  slug: string;
  /** The catalog build or draft that installs to {@link slug}, looked up when the job was accepted. */
  target: CatalogTarget;
  repo: string;
  state: JobStatus;
  /** Job-level aggregate bytes across all files. */
  receivedBytes: number;
  /** Sum of manifest file sizes; 0 until the manifest is fetched. */
  totalBytes: number;
  /**
   * A cleanup: the server's own summary of the unowned entry it may replace and that entry's identity, taken when the
   * request was accepted. `null` for a plain install, which never replaces an unowned entry.
   */
  replace: CleanupTarget | null;
  /**
   * An Update: the plan the user confirmed, for the folder at the slug that holds a model the app did not create.
   * `null` for every other job.
   */
  adopt: AdoptPlan | null;
  /** A Locate: the plan the user confirmed, for a folder of files the user already has. `null` for every other job. */
  locate: LocatePlan | null;
  /**
   * From just before the final entry is moved aside until a rollback put it back: the old entry is not in place. For
   * an Update, from the first add until a rollback removed every added name: the folder holds files the job added.
   */
  swapStarted: boolean;
  /**
   * Set by {@link DownloadManager.cancel}. An in-flight `processJob` unwinds at
   * its next boundary check (and its in-flight fetch is aborted via `controller`),
   * so its `finally` removes the job-private staging dir. NEVER used to delete
   * shared HF-cache blobs — the `mlx download` CLI resumes from that cache.
   */
  cancelled: boolean;
  /** Aborts the job's in-flight `@huggingface/hub` fetch when it is cancelled. */
  controller: AbortController;
}

/**
 * Per-file byte accounting for the active download. Xet-backed files fan out to
 * several requests (distinct URLs) for one logical file, so bytes are keyed by
 * request URL and summed into the file total — the reported `receivedBytes`
 * grows monotonically across the whole file rather than resetting per request.
 */
interface FileContext {
  filePath: string;
  fileSize: number;
  fileIndex: number;
  fileCount: number;
  /** Job-level bytes accumulated by already-finished files in this job. */
  jobBaseBytes: number;
  perUrl: Map<string, number>;
  received: number;
}

export class DownloadManager {
  private readonly modelsDir: string;
  private readonly cacheDir: string;
  /** Memoized catalog shas; see {@link DownloadManager.checkCatalogUpdates}. */
  private catalogShaCache: { at: number; shas: Map<string, string | null>; ttlMs: number } | undefined;
  /**
   * The sweep in flight, shared by every caller that arrives while it runs.
   * Two concurrent sweeps each commit unconditionally, so a slow one landing
   * after a fast one pins the older sha it captured for the whole success TTL —
   * and with no job in the interleaving, `jobResolvedShas` holds nothing that
   * could repair it. Cleared once the sweep settles, so a failed sweep's short
   * TTL still buys a genuinely new attempt instead of replaying the failure.
   */
  private catalogSweep: Promise<ReadonlyMap<string, string | null>> | undefined;
  /** Deadline for one update probe; overridable so a test need not wait it out. */
  private readonly probeTimeoutMs: number;
  /** Sweep counter, so a job can tell whether its write-through predates the sweep now committing. */
  private catalogSweepSeq = 0;
  /**
   * Order in which sha reads were ISSUED — never completed. Upstream is observed
   * as of the request, so a probe that stalls still answers with the sha it held
   * when asked; ranking by completion would call that stale answer the newest read.
   */
  private resolveSeq = 0;
  /**
   * Revisions a JOB resolved, tagged with the sweep in flight when it did and
   * with its place in {@link DownloadManager.resolveSeq}.
   */
  private jobResolvedShas = new Map<string, { revision: string; seq: number; at: number }>();
  /**
   * Issue stamp behind each repo's sha as currently cached. A job that asked
   * BEFORE the sweep whose value is committed holds the older observation
   * however late its answer lands, so its write-through must not overwrite it.
   */
  private catalogReadAt = new Map<string, number>();
  private readonly fetchImpl: typeof fetch;
  private readonly wrappedFetch: typeof fetch;

  private readonly emitter = new EventEmitter();
  private readonly jobsById = new Map<string, JobState>();
  private readonly order: string[] = [];
  private readonly queue: string[] = [];
  private readonly lastEvent = new Map<string, DownloadEvent>();
  // The `start` frame (job `totalBytes` + `fileCount`) is emitted once, so a
  // subscriber that attaches mid-download would otherwise miss it and be stuck on
  // coarse file-index progress. Retain it separately from `lastEvent` (which
  // `progress` overwrites) to replay on attach.
  private readonly startEvent = new Map<string, DownloadEvent>();
  // The in-flight drain loop, retained (not fire-and-forget) so `shutdown` can
  // await every queued/running job's `finally` cleanup before the process exits;
  // `null` whenever no loop is running.
  private drainPromise: Promise<void> | null = null;
  // Set by `shutdown` before it snapshots the queue: a closing manager accepts no
  // new work. Without it a job enqueued during the drain lands outside that
  // snapshot yet is still picked up by the retained loop `shutdown` awaits.
  private closed = false;

  private currentJob: JobState | null = null;
  private currentFile: FileContext | null = null;
  /** The sweep of crashed cleanup holders started with the manager; every job waits for it. */
  private readonly cleanupRecovery: Promise<void>;
  /** The last Update plan per folder; a request must carry its token. */
  private readonly adoptPlans = new Map<string, AdoptPlan>();
  /**
   * Files whose hash did not match, keyed `<slug>\0<path>`, so the next plan refuses an unchanged file without hashing
   * it again. In memory only.
   */
  private readonly mismatchMemo = new Map<string, MismatchRecord>();
  /** The last Locate plan per folder; a request must carry its token. */
  private readonly locatePlans = new Map<string, LocatePlan>();
  /** Whether two `stat` devices are one disk: the choice between a second name and a link. */
  private readonly sameVolume: (a: bigint, b: bigint) => boolean;
  /** How long the folder a Locate reads may take to answer. */
  private readonly diskDeadlineMs: number;
  /** Moves an old entry a cleanup replaced to the Trash; only the `lstat` afterwards says whether it went. */
  private readonly moveToTrash: (path: string) => Promise<void>;

  constructor(opts: {
    modelsDir: string;
    cacheDir?: string;
    fetchImpl?: typeof fetch;
    probeTimeoutMs?: number;
    sameVolume?: (a: bigint, b: bigint) => boolean;
    diskDeadlineMs?: number;
    moveToTrash?: (path: string) => Promise<void>;
  }) {
    this.modelsDir = opts.modelsDir;
    this.cacheDir = opts.cacheDir ?? DEFAULT_CACHE_DIR;
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch;
    this.probeTimeoutMs = opts.probeTimeoutMs ?? CATALOG_SHA_PROBE_TIMEOUT_MS;
    this.sameVolume = opts.sameVolume ?? ((a, b) => a === b);
    this.diskDeadlineMs = opts.diskDeadlineMs ?? LOCATE_DISK_DEADLINE_MS;
    this.moveToTrash = opts.moveToTrash ?? moveToSystemTrash;
    this.wrappedFetch = this.makeCountingFetch();
    // Each job id can have many subscribers (SSE clients); lift the default cap.
    this.emitter.setMaxListeners(0);
    this.cleanupRecovery = this.recoverCleanupHolders(null, true);
  }

  /**
   * Queue the catalog build or draft that installs to the folder `slug`. Rejects a folder no catalog target installs
   * to, and refuses outright once {@link shutdown} has begun — the HTTP listener stays
   * attached across the drain, so a late POST would otherwise be adopted by the
   * loop `shutdown` is awaiting and could keep `close()` from ever completing.
   * A folder that holds another build of the same repository is refused with {@link DownloadConflictError}: the
   * owned swap would delete that build.
   * `replaceBlocker` makes the job a cleanup: the unowned entry at the slug, which must still match what the user
   * confirmed, is replaced once the new model is downloaded and verified, and then unlinked (a link) or moved to the
   * Trash. Refused with {@link DownloadConflictError} when it does not match or cannot be moved, and nothing is queued.
   *
   * At most ONE nonterminal job exists per folder: a request that arrives while a
   * job for that folder is queued or in flight hands back THAT job's id instead of
   * allocating another. Two builds of one repository install to two folders, so
   * each gets its own job. The Models page tracks a single job id per folder, so a
   * duplicate request (a double-click on Install, or a second dashboard tab whose
   * card still reads Install) would otherwise leave an untracked twin that Cancel
   * can no longer reach yet that still downloads and installs. Reusing keeps the
   * request idempotent — no 409 for the SPA to render as a red failure — and
   * matches the stance {@link processJob} already takes below, where re-asking for
   * an already-published repo+revision short-circuits to `done` rather than
   * erroring. A cleanup request while a plain job holds the folder is refused instead: that job would not replace
   * the entry the user asked to replace.
   *
   * `adopt` makes the job an Update: it must be the token of the last {@link planAdoption} for the folder, which the
   * request uses up. A repeat Update request while the Update job is active hands back its id; one while another job
   * holds the folder is refused. `locate` makes the job a Locate, by the same rules with {@link planLocate}, except
   * that only the active Locate's own token hands back its id.
   *
   * Answers the job's id, the repository it downloads, and its kind.
   */
  start(
    slug: string,
    opts?: { replaceBlocker?: ReplaceBlocker; adopt?: string; locate?: string },
  ): { id: string; repo: string; kind: DownloadJobKind } {
    if (this.closed) throw new DownloadsClosedError();
    // `hidden` entries are unpublished repos. They stay in MODEL_CATALOG so a
    // locally converted checkpoint at the canonical slug is still recognized as
    // Installed, but they are NOT downloadable: the remote does not exist yet,
    // so admitting one here allocates a job that fails mid-download with a 401
    // instead of being refused up front. No UI reaches this for a hidden entry
    // (the Models page filters `!item.hidden`); a direct API POST does.
    const target = catalogTarget(slug);
    // Refused before a job exists: nothing is fetched into the HF cache and nothing is made under the link.
    assertModelsDirAvailable(this.modelsDir);
    const asked = opts?.replaceBlocker;
    const token = opts?.adopt;
    const locating = opts?.locate;
    const active = this.activeJobFor(slug);
    if (active !== undefined) {
      if (token !== undefined && active.adopt === null) {
        throw new DownloadConflictError(
          'A download of this model is already in the queue. Cancel it, then use Update.',
        );
      }
      if (locating !== undefined && active.locate?.token !== locating) {
        throw new DownloadConflictError(LOCATE_QUEUED);
      }
      if (asked !== undefined && active.replace === null) {
        throw new DownloadConflictError(
          'A download of this model is already in the queue. Cancel it, then use Clean up & update.',
        );
      }
      return { id: active.id, repo: active.repo, kind: jobKind(active) };
    }
    this.assertNotHeld(target);
    const replace = asked === undefined ? null : this.acceptReplace(slug, asked);
    const adopt = token === undefined ? null : this.acceptAdopt(slug, token);
    const locate = locating === undefined ? null : this.acceptLocate(slug, locating);
    const id = randomUUID();
    const job: JobState = {
      id,
      slug,
      target,
      repo: target.repo,
      state: 'running',
      receivedBytes: 0,
      totalBytes: 0,
      replace,
      adopt,
      locate,
      swapStarted: false,
      cancelled: false,
      controller: new AbortController(),
    };
    this.jobsById.set(id, job);
    this.order.push(id);
    this.queue.push(id);
    void this.drain();
    return { id, repo: target.repo, kind: jobKind(job) };
  }

  /**
   * Refuses a build whose folder holds another build of the same repository: the owned swap of an install, and the
   * folder check of an Update, would act on that other build.
   */
  private assertNotHeld(target: CatalogTarget): void {
    if (target.kind !== 'build') return;
    const held = heldByOtherBuild(this.modelsDir, target.slug);
    if (held !== null) {
      throw new DownloadConflictError(
        `"${target.slug}" holds ${held}. Delete it in Local models, then install ${target.build.id}.`,
      );
    }
  }

  /**
   * The entry a cleanup of `slug` would replace, when it is still what the user confirmed: the server's summary and
   * the `lstat` identity the swap must find again. The target is the catalog folder under the models directory, never
   * a path from the caller.
   */
  private acceptReplace(slug: string, asked: ReplaceBlocker): CleanupTarget {
    const target = modelsChildPath(this.modelsDir, slug, 'replace');
    // Moving a directory to a new parent (the swap's rename into `.staging`) needs write access on the directory.
    if (isRealDirectory(target) && !canWrite(target)) {
      throw new DownloadConflictError(
        `The app cannot move "${slug}": it has no write permission on it. Change its permissions, then try again.`,
      );
    }
    const identity = entryIdentity(target);
    const now = catalogBlocker(this.modelsDir, slug);
    if (
      identity === null ||
      now === null ||
      !sameForeignEntry(now, asked) ||
      !sameIdentity(entryIdentity(target), identity)
    ) {
      throw new DownloadConflictError(`"${slug}" changed on disk. Check it, then try again.`);
    }
    return { summary: now, identity };
  }

  /** The plan an Update request confirmed, taken out of `adoptPlans`: it must be the server's last plan for the folder. */
  private acceptAdopt(slug: string, token: string): AdoptPlan {
    const plan = this.adoptPlans.get(slug);
    if (plan === undefined || plan.token !== token) {
      throw new DownloadConflictError(`"${slug}" changed on disk. Close this, then use Update again.`);
    }
    this.adoptPlans.delete(slug);
    return plan;
  }

  /**
   * Plan an Update of the catalog folder `slug` that holds a model the app did not create: which catalog files are
   * there to be checked by hash, which are missing and will be downloaded, and which other files stay. Reads only
   * `lstat` data and the remote manifests, through the raw fetch with a deadline, so a running job neither counts its
   * bytes nor cancels it. The plan is kept per folder, and a request to start the Update must carry its token. A
   * folder Update cannot take is refused with a {@link DownloadConflictError} that names the file.
   */
  async planAdoption(slug: string): Promise<AdoptPlanSummary> {
    if (this.closed) throw new DownloadsClosedError();
    assertModelsDirAvailable(this.modelsDir);
    const target = catalogTarget(slug);
    if (this.activeJobFor(slug) !== undefined) {
      throw new DownloadConflictError('A download of this model is already in the queue. Cancel it, then use Update.');
    }
    this.assertNotHeld(target);
    await this.cleanupRecovery;
    let plan: AdoptPlan;
    try {
      plan = await this.computeAdoptPlan(target, this.deadlineFetch());
    } catch (error) {
      if (error instanceof DownloadConflictError) throw error;
      throw new Error(`Could not check the folder: ${describeFailure(error)}`);
    }
    this.adoptPlans.set(slug, plan);
    const bytesOf = (items: ReadonlyArray<{ file: ListFileEntry }>): number =>
      items.reduce((sum, item) => sum + (item.file.size > 0 ? item.file.size : 0), 0);
    return {
      repo: target.repo,
      slug,
      token: plan.token,
      revision: plan.revision,
      assetsRevision: plan.assetsRevision,
      check: { count: plan.check.length, bytes: bytesOf(plan.check) },
      download: { count: plan.download.length, bytes: bytesOf(plan.download) },
      other: plan.other,
    };
  }

  /** The folder gates and the whole plan of an Update, read through `fetchImpl`. */
  private async computeAdoptPlan(catalog: CatalogTarget, fetchImpl: typeof fetch): Promise<AdoptPlan> {
    const { slug, repo } = catalog;
    const target = modelsChildPath(this.modelsDir, slug, 'update');
    if (!isRealDirectory(target) || isDownloaderOwned(target)) {
      throw new DownloadConflictError(changedOnDiskMessage(slug));
    }
    if (!canWrite(target)) {
      throw new DownloadConflictError(
        `The app cannot add files to "${slug}": it has no write permission on it. Change its permissions, then try again.`,
      );
    }
    await this.assertCanLink(slug);
    const { revision, selection, files, sidecarPlan } = await this.resolveWanted(catalog, fetchImpl);
    const wanted: AdoptWanted[] = files.map((file) => ({ file, repo, revision }));
    if (sidecarPlan !== null) {
      for (const file of sidecarPlan.files) {
        wanted.push({ file, repo: sidecarPlan.repo.name, revision: sidecarPlan.revision });
      }
    }
    if (!wanted.some((item) => item.file.path === 'config.json')) {
      throw new Error(
        `Repo "${repo}" is not a complete model (neither it nor its assets repo "${selection.assetsRepo}" provides a config.json)`,
      );
    }
    const { folder, entries } = await scanFolder(target, slug);
    const { check, download, other } = classifyFolder(entries, wanted, this.mismatchMemo, slug);
    const assetsRepo = sidecarPlan?.repo.name ?? null;
    const assetsRevision = sidecarPlan?.revision ?? null;
    const token = adoptToken({ repo, revision, assetsRepo, assetsRevision, wanted, check, download, folder });
    return { token, revision, assetsRepo, assetsRevision, folder, wanted, check, download, other };
  }

  /**
   * Update adds files to the folder as second names (`link`), so a disk that cannot make one refuses the plan. Tried
   * on a zero-byte file in `.staging`, beside the folder; both names are removed again.
   */
  private async assertCanLink(slug: string): Promise<void> {
    const stagingRoot = join(this.modelsDir, '.staging');
    assertRealDirOrAbsent(stagingRoot);
    const first = join(stagingRoot, `.link-test-${randomUUID()}`);
    const second = `${first}.2`;
    try {
      try {
        await mkdir(stagingRoot, { recursive: true });
        await writeFile(first, '', { flag: 'wx' });
      } catch {
        throw new DownloadConflictError(
          `The app cannot write in the models folder, so Update cannot add files to "${slug}".`,
        );
      }
      try {
        await link(first, second);
      } catch {
        throw new DownloadConflictError(
          `This disk cannot link files, so Update cannot add files to "${slug}".`,
          'needs-cleanup',
        );
      }
    } finally {
      await unlink(second).catch(() => {});
      await unlink(first).catch(() => {});
    }
  }

  /** The plan a Locate request confirmed, taken out of `locatePlans`: it must be the server's last plan for the folder. */
  private acceptLocate(slug: string, token: string): LocatePlan {
    const plan = this.locatePlans.get(slug);
    if (plan === undefined || plan.token !== token) {
      throw new DownloadConflictError(`The folder or "${slug}" changed. Close this, then use Locate again.`);
    }
    this.locatePlans.delete(slug);
    return plan;
  }

  /**
   * Plan a Locate of the catalog build that installs to `slug` from the folder `picked`: which files of the folder (or
   * of one of its child folders) the app's folder will use, which the job reads to check their content, which it
   * downloads, and whether the models folder shares the files on disk or links to them. Writes nothing and opens no
   * weight file. The plan is kept per folder, and a request to start the Locate must carry its token. A folder Locate
   * cannot take is refused with a {@link DownloadConflictError} whose text says why.
   */
  async planLocate(slug: string, picked: string): Promise<LocatePlanSummary> {
    if (this.closed) throw new DownloadsClosedError();
    const target = catalogTarget(slug);
    if (target.kind !== 'build') throw new DownloadConflictError('Locate is not offered for companion weights.');
    assertModelsDirAvailable(this.modelsDir);
    if (this.activeJobFor(slug) !== undefined) throw new DownloadConflictError(LOCATE_QUEUED);
    this.assertNotHeld(target);
    await this.cleanupRecovery;
    this.assertLocateSlugFree(slug);
    const pickedReal = await this.pickedFolder(picked);
    let plan: LocatePlan;
    try {
      plan = await this.computeLocatePlan(target, pickedReal, this.deadlineFetch());
    } catch (error) {
      if (error instanceof DownloadConflictError || error instanceof HubUnreachableError) throw error;
      throw new Error(`Could not check the folder: ${describeFailure(error)}`);
    }
    this.locatePlans.set(slug, plan);
    const count = (sizes: readonly number[]): { count: number; bytes: number } => ({
      count: sizes.length,
      bytes: sizes.reduce((sum, size) => sum + size, 0),
    });
    return {
      slug,
      repo: target.repo,
      token: plan.token,
      folder: plan.folder,
      mode: plan.mode,
      byRecord: plan.byRecord,
      revision: plan.revision,
      use: count(plan.use.map((item) => Number(item.fingerprint.size))),
      read: count(
        plan.byRecord
          ? []
          : plan.use.filter((item) => isLinkableFile(item.path)).map((item) => Number(item.fingerprint.size)),
      ),
      download: count(plan.download.map((item) => (item.file.size > 0 ? item.file.size : 0))),
    };
  }

  /**
   * The folder at `slug` that a Locate may build: nothing there, or the app's own folder that holds no loadable model
   * (a Locate whose disk went away). Returns its path.
   */
  private assertLocateSlugFree(slug: string): string {
    const finalDir = modelsChildPath(this.modelsDir, slug, 'locate');
    if (!isPathOccupied(finalDir)) return finalDir;
    if (isRealDirectory(finalDir) && isDownloaderOwned(finalDir) && !isModelPresent(finalDir)) return finalDir;
    throw new DownloadConflictError(
      `"${slug}" already exists in the models folder. Use the action on its card, or delete it in Local models.`,
    );
  }

  /** The real path of the picked folder, read within {@link diskDeadlineMs}; never one inside the models folder. */
  private async pickedFolder(picked: string): Promise<string> {
    const read = async (): Promise<string | null> => {
      const real = await realpath(picked);
      return (await isFolder(real)) ? real : null;
    };
    let real: string | null;
    try {
      real = await withDeadline(read(), this.diskDeadlineMs, () => new DownloadConflictError(LOCATE_SLOW));
    } catch (error) {
      if (error instanceof DownloadConflictError) throw error;
      real = null;
    }
    if (real === null) throw new DownloadConflictError(LOCATE_UNREADABLE);
    const modelsReal = await realpath(this.modelsDir).catch(() => resolve(this.modelsDir));
    if (isInside(real, modelsReal)) throw new DownloadConflictError(LOCATE_INSIDE_MODELS);
    return real;
  }

  /** The whole plan of a Locate from the real folder `picked`, the hub read through `fetchImpl`. */
  private async computeLocatePlan(target: BuildTarget, picked: string, fetchImpl: typeof fetch): Promise<LocatePlan> {
    const modelsReal = await realpath(this.modelsDir).catch(() => resolve(this.modelsDir));
    const modelsDev = deviceOf(this.modelsDir);
    let listing: { folders: string[]; tooMany: boolean };
    try {
      listing = await candidateFolders(picked, modelsReal);
    } catch {
      throw new DownloadConflictError(LOCATE_UNREADABLE);
    }
    const byRecord = await this.recordLocatePlan(target, picked, listing.folders, modelsDev);
    const plan = byRecord ?? (await this.hubLocatePlan(target, picked, listing, modelsDev, fetchImpl));
    // A link in the folder, or a catalog path into a child folder, can lead into the models folder.
    const inModels = plan.use.find((item) => isInside(item.realPath, modelsReal));
    if (inModels !== undefined) {
      throw new DownloadConflictError(
        `"${basename(inModels.path)}" here is a file in the models folder. ` +
          'Locate uses only files outside the models folder.',
      );
    }
    return plan;
  }

  /**
   * The plan of a Locate from a folder that has this app's install record for the build: its files are trusted, so
   * nothing is read and nothing is fetched. `null` when no candidate folder has one.
   */
  private async recordLocatePlan(
    target: BuildTarget,
    picked: string,
    folders: readonly string[],
    modelsDev: bigint,
  ): Promise<LocatePlan | null> {
    const found: { dir: string; record: DownloadCompletion }[] = [];
    let otherBuild: string | null = null;
    for (const dir of folders) {
      const record = await readRecord(dir);
      if (record === undefined || !isLocatableRecord(record, target.repo)) continue;
      if (!(await recordInstalled(dir, record))) continue;
      const weights = record.files.filter(isWeightFile);
      if (!weights.every((path) => buildTakes(target, path))) {
        otherBuild ??= this.otherBuildTaking(target, weights);
        continue;
      }
      found.push({ dir, record });
    }
    if (found.length > 1) {
      throw new DownloadConflictError(locateTwoFoldersMessage(target.entry.label, found[0]!.dir, found[1]!.dir));
    }
    if (found.length === 0) {
      if (otherBuild !== null) throw new DownloadConflictError(locateOtherBuildMessage(otherBuild));
      return null;
    }
    const { dir, record } = found[0]!;
    const use: LocateUse[] = [];
    for (const path of record.files) {
      const at = await foundAt(join(dir, path));
      if (at === null) throw new DownloadConflictError(LOCATE_UNREADABLE);
      use.push({ path, realPath: at.realPath, how: 'copy', file: null, fingerprint: at.fingerprint });
    }
    const mode = this.locateMode(use, record.files, record.files.find(isWeightFile)!, modelsDev);
    const hasAssets = record.assetsRepo !== undefined && record.assetsRevision !== undefined;
    const plan = {
      slug: target.slug,
      repo: target.repo,
      picked,
      folder: dir,
      mode,
      byRecord: true,
      revision: record.revision,
      assetsRepo: hasAssets ? record.assetsRepo! : null,
      assetsRevision: hasAssets ? record.assetsRevision! : null,
      wanted: [...record.files],
      use,
      download: [],
    };
    return { ...plan, token: locateToken(plan) };
  }

  /** The build of `target`'s model whose file filter takes every one of `weights`, other than `target`, or `null`. */
  private otherBuildTaking(target: BuildTarget, weights: readonly string[]): string | null {
    for (const other of catalogTargets()) {
      if (other.kind !== 'build' || other.entry !== target.entry || other.slug === target.slug) continue;
      if (other.repo.toLowerCase() !== target.repo.toLowerCase()) continue;
      if (weights.length > 0 && weights.every((path) => buildTakes(other, path))) return other.build.id;
    }
    return null;
  }

  /**
   * The mode of a Locate by {@link pickMode}, or its refusal; sets how each used file enters the app's folder: a
   * weight-like file as a second name (`shared`) or a link (`linked`), every other file as the app's own copy.
   */
  private locateMode(
    use: LocateUse[],
    wanted: readonly string[],
    primary: string,
    modelsDev: bigint,
  ): 'shared' | 'linked' {
    const picked = pickMode(use, wanted, primary, modelsDev, this.sameVolume);
    if ('refuse' in picked) {
      throw new DownloadConflictError(
        picked.refuse === 'name' ? LOCATE_OTHER_NAMES : locateNotBesideMessage(picked.file),
      );
    }
    for (const item of use) {
      item.how = !isLinkableFile(item.path) ? 'copy' : picked.mode === 'shared' ? 'link' : 'symlink';
    }
    return picked.mode;
  }

  /** A file that failed the content check of an earlier Locate and is unchanged since. */
  private failedBefore(
    at: { realPath: string; fingerprint: { size: bigint; mtimeNs: bigint; ino: bigint } },
    oid: string,
  ): boolean {
    const failed = this.mismatchMemo.get(`locate\0${at.realPath}`);
    return (
      failed !== undefined &&
      BigInt(failed.size) === at.fingerprint.size &&
      failed.mtimeNs === at.fingerprint.mtimeNs &&
      failed.ino === at.fingerprint.ino &&
      failed.oid === oid
    );
  }

  /**
   * The plan of a Locate checked against the catalog's file list: the candidate folder that holds every primary weight
   * of the build by name and size, the other wanted files found there (a small file only when its hash matches now),
   * and the rest to download.
   */
  private async hubLocatePlan(
    target: BuildTarget,
    picked: string,
    listing: { folders: string[]; tooMany: boolean },
    modelsDev: bigint,
    fetchImpl: typeof fetch,
  ): Promise<LocatePlan> {
    const label = target.entry.label;
    let resolved: Awaited<ReturnType<DownloadManager['resolveWanted']>>;
    try {
      resolved = await this.resolveWanted(target, fetchImpl);
    } catch (error) {
      throw new HubUnreachableError(
        `Could not reach Hugging Face: ${describeFailure(error)}. ` +
          `Locate needs the catalog's file list to check this folder. ${NOTHING_CHANGED}`,
      );
    }
    const { revision, selection, files, sidecarPlan } = resolved;
    const wanted: LocateDownload[] = files.map((file) => ({ file, repo: target.repo, revision }));
    for (const file of sidecarPlan?.files ?? []) {
      wanted.push({ file, repo: sidecarPlan!.repo.name, revision: sidecarPlan!.revision });
    }
    if (!wanted.some((item) => item.file.path === 'config.json')) {
      throw new Error(
        `Repo "${target.repo}" is not a complete model (neither it nor its assets repo "${selection.assetsRepo}" provides a config.json)`,
      );
    }
    const primaries = wanted.filter((item) => isWeightFile(item.file.path));

    const holders: { dir: string; found: Map<string, FoundFile>; key: string }[] = [];
    let best: { dir: string; missing: string[]; found: number } | null = null;
    let otherSize: string | null = null;
    for (const dir of listing.folders) {
      const found = new Map<string, FoundFile>();
      const missing: string[] = [];
      for (const item of primaries) {
        const at = await findWanted(dir, item.file.path);
        if (at !== null && Number(at.fingerprint.size) === item.file.size) found.set(item.file.path, at);
        else {
          if (at !== null) otherSize ??= basename(item.file.path);
          missing.push(basename(item.file.path));
        }
      }
      if (missing.length === 0) {
        // The picked folder and its child can reach the same files (`<repo>/UD-Q4_K_XL/x.gguf`): one holder.
        const key = [...found.values()]
          .map((at) => at.realPath)
          .sort()
          .join('\0');
        if (!holders.some((holder) => holder.key === key)) holders.push({ dir, found, key });
      } else if (found.size > 0 && (best === null || found.size > best.found)) {
        // Named by the folder the found parts are in, which is a variant folder for a repo-shaped folder.
        best = { dir: dirname([...found.values()][0]!.realPath), missing, found: found.size };
      }
    }
    if (holders.length > 1) {
      throw new DownloadConflictError(locateTwoFoldersMessage(label, holders[0]!.dir, holders[1]!.dir));
    }
    const first = basename(primaries[0]!.file.path);
    if (holders.length === 0) {
      if (otherSize !== null) {
        throw new DownloadConflictError(
          `"${otherSize}" here is not the file the catalog lists: its size is different. ` +
            'It can be an older version. Locate takes only the current file. Use Install.',
        );
      }
      if (best !== null) {
        throw new DownloadConflictError(
          `${best.missing.length} of ${primaries.length} weight files are missing in ${best.dir}: ` +
            `${best.missing.slice(0, 3).join(', ')}. Locate needs all of them in one folder.`,
        );
      }
      const other = await this.otherBuildIn(target, listing.folders, fetchImpl);
      if (other !== null) throw new DownloadConflictError(locateOtherBuildMessage(other));
      if (listing.tooMany) {
        throw new DownloadConflictError(
          `This folder has too many folders to search. Pick the folder that holds "${first}".`,
        );
      }
      throw new DownloadConflictError(
        `No ${label} files here. Pick the folder that holds "${first}". ` +
          'In a Hugging Face cache that is the folder inside "snapshots".',
      );
    }

    const holder = holders[0]!;
    const use: LocateUse[] = [];
    for (const item of primaries) {
      const at = holder.found.get(item.file.path)!;
      const name = basename(item.file.path);
      const sum = checksumOf(item.file);
      if (sum === null) {
        throw new DownloadConflictError(`The catalog gives no checksum for "${name}", so Locate cannot check it.`);
      }
      if (this.failedBefore(at, sum.oid)) {
        throw new DownloadConflictError(
          `"${name}" in ${holder.dir} is not the catalog file: its content is different. Use Install.`,
        );
      }
      use.push({
        path: item.file.path,
        realPath: at.realPath,
        how: 'copy',
        file: item.file,
        fingerprint: at.fingerprint,
      });
    }
    const download: LocateDownload[] = [];
    for (const item of wanted) {
      if (isWeightFile(item.file.path)) continue;
      const at = await findWanted(holder.dir, item.file.path);
      const sum = checksumOf(item.file);
      let usable = at !== null && sum !== null && Number(at.fingerprint.size) === item.file.size;
      if (usable && isLinkableFile(item.file.path)) usable = !this.failedBefore(at!, sum!.oid);
      else if (usable) {
        usable =
          item.file.size <= LOCATE_SMALL_MAX_BYTES && (await digestMatches(at!.realPath, sum!).catch(() => false));
      }
      if (usable) {
        use.push({
          path: item.file.path,
          realPath: at!.realPath,
          how: 'copy',
          file: item.file,
          fingerprint: at!.fingerprint,
        });
      } else download.push(item);
    }
    // A download first writes the cache's blob and its name of the file: in the folder that holds the user's files
    // when a link in the cache leads there, or when that folder is part of the cache.
    let throughFolder: LocateDownload | undefined;
    for (const item of download) {
      const repoDir = join(this.cacheDir, getRepoFolderName({ type: 'model', name: item.repo }));
      const writes = [
        await realWritePath(join(repoDir, 'snapshots', item.revision, item.file.path)),
        await realWritePath(join(repoDir, 'blobs')),
      ];
      if (writes.some((path) => isInside(path, holder.dir))) {
        throughFolder = item;
        break;
      }
    }
    if (throughFolder !== undefined) {
      const name = basename(throughFolder.file.path);
      const holdsCache = isInside(await realWritePath(this.cacheDir), holder.dir);
      throw new DownloadConflictError(holdsCache ? locateHoldsCacheMessage(name) : locateInCacheMessage(name));
    }
    const paths = wanted.map((item) => item.file.path);
    const mode = this.locateMode(use, paths, primaries[0]!.file.path, modelsDev);
    const plan = {
      slug: target.slug,
      repo: target.repo,
      picked,
      folder: holder.dir,
      mode,
      byRecord: false,
      revision,
      assetsRepo: sidecarPlan?.repo.name ?? null,
      assetsRevision: sidecarPlan?.revision ?? null,
      wanted: paths,
      use,
      download,
    };
    return { ...plan, token: locateToken(plan) };
  }

  /**
   * Another build of `target`'s model whose weight files are all in one of `folders` by name, or `null`. Reads that
   * build's file list; a build whose list cannot be read is passed over.
   */
  private async otherBuildIn(
    target: BuildTarget,
    folders: readonly string[],
    fetchImpl: typeof fetch,
  ): Promise<string | null> {
    for (const other of catalogTargets()) {
      if (other.kind !== 'build' || other.entry !== target.entry || other.slug === target.slug) continue;
      let weights: ListFileEntry[];
      try {
        const revision = await this.resolveRevision(other.repo, fetchImpl);
        const globs = selectionOf(other).globs?.map(globToRegex);
        weights = (await this.listWantedFiles(other.repo, revision, globs, fetchImpl)).filter((file) =>
          isWeightFile(file.path),
        );
      } catch {
        continue;
      }
      if (weights.length === 0) continue;
      for (const dir of folders) {
        let all = true;
        for (const file of weights) {
          if ((await findWanted(dir, file.path)) === null) {
            all = false;
            break;
          }
        }
        if (all) return other.build.id;
      }
    }
    return null;
  }

  /**
   * The one nonterminal (queued / in-flight / committing) job for the folder `slug`, if any.
   * Derived from `jobsById` on demand rather than kept in a slug→id index, so it
   * can never go stale: every terminal transition and every dismissal already
   * updates `jobsById`, and a second structure would have to be cleared at each of
   * those sites. The registry holds at most a handful of catalog folders.
   */
  private activeJobFor(slug: string): JobState | undefined {
    for (const job of this.jobsById.values()) {
      // `!job.cancelled` as well as the state: a cancel of the IN-FLIGHT job
      // returns while the job is still `running`, because `cancel()` leaves the
      // terminal transition to `processJob` as it unwinds (that is what keeps the
      // `cancelled` event from being emitted twice). Handing that job back would
      // answer a fresh Install with an id that is already stopping — the caller
      // subscribes to a stream whose next frame is `cancelled`, and no download
      // happens. A job on its way out is not an active job.
      if (job.slug === slug && !isTerminal(job.state) && !job.cancelled) return job;
    }
    return undefined;
  }

  /**
   * Cancel a running job, or DISMISS a terminal one. Behavior by state:
   *
   *   - unknown id → `false` (nothing to do);
   *   - `running` (in-flight or still-queued) → signal the in-flight `processJob`
   *     to unwind (abort its `@huggingface/hub` fetch, trip its next boundary
   *     check) so its `finally` removes the job-private staging dir; a still-queued
   *     job is dropped from the queue and settled here, marked `cancelled` with a
   *     terminal `cancelled` event. Returns `true`;
   *   - `committing` → `false`: a cancel racing an in-progress publish is refused
   *     rather than reporting success while the model still installs;
   *   - terminal (`error`/`cancelled`/`done`) → DISMISS: evict the job entirely
   *     (`jobsById` + `order` + `lastEvent`) and return `true`, so
   *     `DELETE /api/downloads/:id` clears a failed/finished card server-side
   *     instead of 404ing and retaining the row forever. A terminal id is never in
   *     `queue` and is never the in-flight `currentJob` (which still holds
   *     `running`/`committing`), so this only ever evicts a truly-finished job.
   *
   * Deliberately never touches the shared HF blob cache (`~/.cache/huggingface`):
   * the `mlx download` CLI relies on it to resume, so cancel only aborts the job
   * and lets its disposable staging dir clean up.
   */
  cancel(id: string): boolean {
    const job = this.jobsById.get(id);
    if (job === undefined) return false;

    // Terminal-dismiss: a settled job carries no in-flight work and is not the
    // `currentJob`, so evicting it from every tracking structure is safe and lets
    // the DELETE route succeed instead of 404ing on a failed card.
    if (isTerminal(job.state)) {
      this.jobsById.delete(id);
      const at = this.order.indexOf(id);
      if (at !== -1) this.order.splice(at, 1);
      this.lastEvent.delete(id);
      this.startEvent.delete(id);
      return true;
    }

    // Only `running` is cancellable; `committing` (the non-cancellable publish
    // window) is refused so a late cancel never reports success mid-install.
    if (job.state !== 'running') return false;
    job.cancelled = true;
    // Interrupt any in-flight hub fetch so a multi-GB download stops promptly
    // rather than only at the next between-files boundary check.
    job.controller.abort();
    if (this.currentJob?.id === id) {
      // In flight: let `processJob` emit the terminal `cancelled` event and run
      // its `finally` (staging cleanup) as it unwinds — avoid a double emit here.
      return true;
    }
    // Still queued (never started): drop it so it never runs, and settle it now.
    const queued = this.queue.indexOf(id);
    if (queued !== -1) this.queue.splice(queued, 1);
    job.state = 'cancelled';
    this.emit({ type: 'cancelled', id });
    return true;
  }

  /**
   * Abort every in-flight and queued job, then wait for their staging dirs to be
   * removed before returning. The server's `close()` awaits this on
   * SIGINT/SIGTERM so the process can never `process.exit` while a `processJob`
   * is mid-write — which would orphan a partial, potentially multi-GB `.staging`
   * tree (its `finally` never runs) — and so a background job can never publish a
   * model AFTER the server is considered closed. Reuses the per-job {@link cancel}
   * (its AbortController + `cancelled` boundary) rather than duplicating abort
   * logic: a queued job is dropped and settled `cancelled`, the in-flight job is
   * aborted, and a job already in its non-cancellable `committing` window is left
   * to finish its atomic swap and simply awaited to completion. Never throws.
   */
  async shutdown(): Promise<void> {
    // Refuse new work before anything else: the snapshot below is one-shot, so a
    // `start` racing the drain would slip past it and be run anyway.
    this.closed = true;
    // Snapshot the queue first: `cancel` splices each settled id out of it.
    for (const id of this.queue.slice()) this.cancel(id);
    // Abort the in-flight job (if any); its `processJob` unwinds via the aborted
    // fetch + boundary check and its `finally` removes the job-private staging dir.
    if (this.currentJob !== null) this.cancel(this.currentJob.id);
    // Await the retained drain loop so every unwinding `finally` (staging `rm`)
    // has completed before we resolve; `null` when nothing is in flight.
    await (this.drainPromise ?? Promise.resolve()).catch(() => {});
  }

  jobs(): Array<{
    id: string;
    slug: string;
    repo: string;
    kind: DownloadJobKind;
    state: JobStatus;
    receivedBytes: number;
    totalBytes: number;
    keptDir?: string;
    removed?: 'link' | 'trash';
    entry?: Pick<ForeignEntrySummary, 'kind' | 'linkToFile'>;
  }> {
    return this.order.map((id) => {
      const job = this.jobsById.get(id)!;
      // The page reads what a job did from here when it was not open as the job ended.
      const last = this.lastEvent.get(id);
      const done = last?.type === 'done' ? last : undefined;
      return {
        id: job.id,
        slug: job.slug,
        repo: job.repo,
        kind: jobKind(job),
        state: job.state,
        receivedBytes: job.receivedBytes,
        totalBytes: job.totalBytes,
        ...(done?.keptDir !== undefined ? { keptDir: done.keptDir } : {}),
        ...(done?.removed !== undefined ? { removed: done.removed } : {}),
        ...(job.replace !== null
          ? { entry: { kind: job.replace.summary.kind, linkToFile: job.replace.summary.linkToFile } }
          : {}),
      };
    });
  }

  /**
   * Subscribe to a job's events. On attach, replay the `start` frame first (so a
   * mid-download subscriber learns the job `totalBytes`/`fileCount` and renders an
   * aggregate byte bar instead of coarse file-index progress), then the most recent
   * event — skipping the duplicate when `start` IS the most recent.
   *
   * Those two frames are the WHOLE picture, which is why a `progress` frame
   * states `jobReceivedBytes` outright: the finished files' frames are gone by
   * then, so a subscriber that tried to add up what it received would report
   * only the file in flight.
   */
  subscribe(id: string, fn: (event: DownloadEvent) => void): () => void {
    const start = this.startEvent.get(id);
    const last = this.lastEvent.get(id);
    if (start !== undefined) fn(start);
    if (last !== undefined && last !== start) fn(last);
    this.emitter.on(id, fn);
    return () => {
      this.emitter.off(id, fn);
    };
  }

  private emit(event: DownloadEvent): void {
    if (event.type === 'start') this.startEvent.set(event.id, event);
    this.lastEvent.set(event.id, event);
    this.emitter.emit(event.id, event);
  }

  private makeCountingFetch(): typeof fetch {
    return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      // Thread the active job's abort signal into every hub fetch so a cancel
      // interrupts an in-flight download mid-stream (combined with any signal the
      // caller already supplied). No active job → pass the caller's init through.
      const jobSignal = this.currentJob?.controller.signal;
      let requestInit = init;
      if (jobSignal !== undefined) {
        const existing = init?.signal ?? undefined;
        const signal = existing !== undefined ? AbortSignal.any([existing, jobSignal]) : jobSignal;
        requestInit = { ...init, signal };
      }
      const response = await this.fetchImpl(input, requestInit);
      const file = this.currentFile;
      const job = this.currentJob;
      // No active file (manifest listing / metadata) or bodyless response:
      // pass straight through without counting.
      if (file === null || job === null || response.body === null) return response;

      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      const counter = new TransformStream<Uint8Array, Uint8Array>({
        transform: (chunk, controller) => {
          file.perUrl.set(url, (file.perUrl.get(url) ?? 0) + chunk.byteLength);
          let sum = 0;
          for (const value of file.perUrl.values()) sum += value;
          file.received = sum;
          job.receivedBytes = file.jobBaseBytes + sum;
          this.emit({
            type: 'progress',
            id: job.id,
            file: file.filePath,
            receivedBytes: file.received,
            jobReceivedBytes: job.receivedBytes,
            totalBytes: file.fileSize,
            fileIndex: file.fileIndex,
            fileCount: file.fileCount,
          });
          controller.enqueue(chunk);
        },
      });
      return new Response(response.body.pipeThrough(counter), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    };
  }

  /**
   * Process queued jobs one at a time. The running loop is retained on
   * `drainPromise` (not fire-and-forget) so {@link shutdown} can await every
   * queued/running job's `finally` cleanup before the process exits; a `start`
   * that arrives while a loop is already running joins that same promise instead
   * of spawning a second, concurrent drain.
   */
  private drain(): Promise<void> {
    if (this.drainPromise !== null) return this.drainPromise;
    this.drainPromise = (async () => {
      try {
        while (this.queue.length > 0) {
          const id = this.queue.shift()!;
          const job = this.jobsById.get(id);
          if (job !== undefined) await this.processJob(job);
        }
      } finally {
        this.drainPromise = null;
      }
    })();
    return this.drainPromise;
  }

  /**
   * Remote commit sha for every VISIBLE catalog repo, for the "update
   * available" check on the Models page.
   *
   * A re-upload lands new bytes at the SAME repo id, so local presence alone
   * can never tell a user their checkpoint is stale — only the sha can. This
   * is the read half; the write half already works: `processJob` re-downloads
   * whenever the installed marker's revision differs from the resolved one, so
   * the UI's "Update" is literally the existing install.
   *
   * Never throws and never rejects. A repo that cannot be resolved — offline,
   * 429, a deleted repo — maps to `null`, which the caller reads as "no badge",
   * never as "up to date". `hidden` entries are skipped: they are unpublished
   * repos that answer 401.
   *
   * Cached, because `useJson` refetches on every mount and every reconnect and
   * there is no polling to smooth the rate. An all-`null` result (the offline
   * case) is cached far more briefly so a machine that regains its network is
   * not stuck reporting nothing for hours.
   */
  async checkCatalogUpdates(now: number = Date.now()): Promise<ReadonlyMap<string, string | null>> {
    const cached = this.catalogShaCache;
    if (cached !== undefined && now - cached.at < cached.ttlMs) return cached.shas;
    if (this.catalogSweep === undefined) {
      this.catalogSweep = this.sweepCatalogShas(now).finally(() => {
        this.catalogSweep = undefined;
      });
    }
    return this.catalogSweep;
  }

  /**
   * The raw fetch with a deadline per request. `AbortSignal.any` keeps whatever signal the hub client supplies rather
   * than replacing it.
   */
  private deadlineFetch(): typeof fetch {
    return (input, init) => {
      const deadline = AbortSignal.timeout(this.probeTimeoutMs);
      const existing = init?.signal ?? undefined;
      const signal = existing !== undefined ? AbortSignal.any([existing, deadline]) : deadline;
      return this.fetchImpl(input, { ...init, signal });
    };
  }

  private async sweepCatalogShas(now: number): Promise<ReadonlyMap<string, string | null>> {
    const seq = ++this.catalogSweepSeq;
    const probeFetch = this.deadlineFetch();
    const shas = new Map<string, string | null>();
    const reads = new Map<string, number>();
    await Promise.all(
      catalogUpdateRepos().map(async (repo) => {
        const at = ++this.resolveSeq;
        try {
          // RAW fetch, never `wrappedFetch`. That wrapper attributes every
          // response body to `currentFile`, so a probe running while a model
          // downloads would inflate that file's progress past its own total;
          // it also threads the active job's abort signal, which would make a
          // user's cancel kill an unrelated update check.
          shas.set(repo, await this.resolveRevision(repo, probeFetch));
          // Only a SUCCESSFUL read claims the slot, so a job's real revision
          // still beats this sweep's `null`.
          reads.set(repo, at);
        } catch {
          shas.set(repo, null);
        }
      }),
    );
    // ANY unresolved repo takes the short TTL, not only an all-failed sweep. The
    // cache is one entry for the whole sweep, so a single transient failure
    // alongside successes would otherwise pin that repo's `null` for the full
    // six hours — no mount or reconnect in that window could surface its update.
    // A job that installed a revision WHILE this sweep ran has the newer read
    // only if it asked LAST: the sweep may have captured its repo early, then
    // waited on another, and since the whole map is replaced below that job's
    // write-through would be lost. The opposite interleaving is just as real —
    // the job asks, this sweep starts and asks the same repo, and the job's
    // slower answer still lands first — and there the sweep holds the newer sha.
    // Overlaying it would bury a genuine update for the whole success TTL, since
    // the post-download refresh reads this very map back.
    for (const [repo, resolved] of this.jobResolvedShas) {
      if (resolved.seq >= seq && shas.has(repo) && resolved.at > (reads.get(repo) ?? 0)) {
        shas.set(repo, resolved.revision);
        // Keeps `catalogReadAt` describing what was actually committed. No
        // reachable write depends on it — one repo holds one nonterminal job,
        // so the next stamp is always higher either way — but a reader who
        // trusts the field's meaning should not have to derive that.
        reads.set(repo, resolved.at);
      }
    }
    this.catalogReadAt = reads;
    const anyFailed = [...shas.values()].some((sha) => sha === null);
    this.catalogShaCache = { at: now, shas, ttlMs: anyFailed ? CATALOG_SHA_NEGATIVE_TTL_MS : CATALOG_SHA_TTL_MS };
    return shas;
  }

  /**
   * Resolve the repo's current commit sha so the WHOLE job reads one immutable
   * snapshot. Threading this same sha to `listFiles` and every
   * `downloadFileToCacheDir` means a repo update mid-download can never assemble
   * a mixed-revision checkpoint. The sha must be an immutable 40-hex commit: a
   * missing sha or a mutable ref (a branch like `main`) is refused, never pinned —
   * the completion marker's `revision` is trusted downstream as a stable identity,
   * so a mutable value there would let a later same-shaped fine-tune restore stale
   * state and would break this job's own mid-download atomicity.
   */
  private async resolveRevision(repoName: string, fetchImpl: typeof fetch = this.wrappedFetch): Promise<string> {
    const info = await modelInfo({ name: repoName, additionalFields: ['sha'], fetch: fetchImpl });
    const sha: unknown = info.sha;
    if (!(typeof sha === 'string' && /^[0-9a-f]{40}$/i.test(sha))) {
      throw new Error(`Cannot resolve an immutable commit for "${repoName}"; refusing to pin to a mutable ref`);
    }
    return sha;
  }

  /**
   * Fold a revision a job read upstream into the update-check cache. Without it a sweep cached before upstream
   * advanced keeps serving the older sha for up to its full TTL, so the card re-offers an update for a revision
   * already installed and every click completes as a no-op. `at` is the job's place in {@link resolveSeq}, taken
   * before the read was issued.
   */
  private noteJobRevision(repoName: string, revision: string, at: number): void {
    this.jobResolvedShas.set(repoName, { revision, seq: this.catalogSweepSeq, at });
    // The overlay guards a sweep that commits AFTER this. A sweep that already
    // committed is past it, so the same ordering test has to run here too.
    if (at > (this.catalogReadAt.get(repoName) ?? 0)) {
      this.catalogShaCache?.shas.set(repoName, revision);
      this.catalogReadAt.set(repoName, at);
    }
  }

  /** The manifest entries of `repoName` at `revision` that this catalog entry installs, in listing order. */
  private async listWantedFiles(
    repoName: string,
    revision: string,
    globs: RegExp[] | undefined,
    fetchImpl: typeof fetch,
  ): Promise<ListFileEntry[]> {
    const repo = { type: 'model' as const, name: repoName };
    const files: ListFileEntry[] = [];
    for await (const file of listFiles({ repo, recursive: true, revision, fetch: fetchImpl })) {
      if (file.type !== 'directory' && isWantedFileFor(file.path, globs)) {
        // Defense-in-depth: `file.path` is a third-party string about to become
        // a `join(stagingDir, file.path)` write/read/verify/publish target (every
        // downstream site derives from this `files[]` array). Refuse — fail
        // closed — any entry that could escape the staging dir before it is
        // admitted. Unreachable for the curated 4-repo catalog (git/HF tree
        // paths cannot express `..`/absolute), but never trust the boundary.
        if (!isSafeRelPath(file.path)) {
          throw new Error(`Refusing to download "${file.path}" from "${repoName}": path is not a safe relative path`);
        }
        files.push(file);
      }
    }
    return files;
  }

  /**
   * Everything a job installs for `target`, read through `fetchImpl`: one pinned revision, the glob-filtered
   * manifest (which must carry a weight payload), and the sidecar plan. `processJob` runs the same steps one by one,
   * because it records the revision in the update cache before it lists anything.
   */
  private async resolveWanted(
    target: CatalogTarget,
    fetchImpl: typeof fetch,
  ): Promise<{
    revision: string;
    selection: { globs?: readonly string[]; assetsRepo?: string };
    files: ListFileEntry[];
    sidecarPlan: SidecarPlan | null;
  }> {
    const repoName = target.repo;
    const revision = await this.resolveRevision(repoName, fetchImpl);
    const selection = selectionOf(target);
    const files = await this.listWantedFiles(repoName, revision, selection.globs?.map(globToRegex), fetchImpl);
    assertModelPayload(repoName, files, selection.assetsRepo);
    const sidecarPlan =
      selection.assetsRepo === undefined ? null : await this.planAssetSidecars(selection.assetsRepo, files, fetchImpl);
    return { revision, selection, files, sidecarPlan };
  }

  private async processJob(job: JobState): Promise<void> {
    const slug = job.slug;
    const finalDir = join(this.modelsDir, slug);
    const repo = { type: 'model' as const, name: job.repo };
    this.currentJob = job;

    const stagingRoot = join(this.modelsDir, '.staging');

    let stagingDir: string | undefined;
    try {
      // The occupancy preflight, factored so it reads identically at both
      // its call sites: a final dir the downloader does not own (a manual copy or an
      // `mlx download` install with no completion marker, a symlink into such an
      // UNMARKED dir, or a dangling link — `isDownloaderOwned` follows the link and
      // reads the TARGET's marker, so only a live link into an unmarked/absent
      // target stays unowned) is refused — so an unowned local
      // checkpoint never triggers a full multi-GB download+hash only to be refused at
      // publish. It reads ONLY `finalDir`, so it is order-independent of the staging
      // root and the resolved revision. Run it (a) before backup recovery — the
      // recovery reap classifies finalDir no-follow (`lstat`), so a refused link
      // never reaches it — and (b) again after
      // recovery, so a `finalDir` that recovery (or a race) replaced is refused
      // BEFORE skip-detection follows it. The publish-time checks
      // below still guard the narrow check-then-swap race for a dir that races in
      // after this point.
      // A cleanup checks the confirmed identity of whatever is at the path, owned or not, and never reads through it.
      const refuseIfUnownedFinal = (): void => {
        if (!isPathOccupied(finalDir)) return;
        if (job.replace !== null) {
          assertStillReplaceable(job.replace, finalDir, slug);
          return;
        }
        if (isDownloaderOwned(finalDir)) return;
        throw new Error(
          `Refusing to install over "${finalDir}": it was not created by the dashboard downloader. ` +
            `Remove it manually to reinstall.`,
        );
      };

      // Refuse a symlinked (or non-directory) `.staging` root BEFORE creating it:
      // otherwise `mkdir(..., { recursive })` follows the link and every staged
      // write, the sweep, the publish backup, and the `finally` cleanup all land
      // on the external target. Rejecting a symlinked root closes the escape at the
      // source — every downstream path is built under this root. Done first (it only
      // touches the staging root, not finalDir) so the dead-tree reap below can run.
      assertRealDirOrAbsent(stagingRoot);
      await mkdir(stagingRoot, { recursive: true });
      await this.cleanupRecovery;

      // Reap a crashed job's private staging tree BEFORE the ownership preflight, so
      // even a job the preflight REFUSES (an unowned/foreign finalDir, not a cleanup)
      // still frees a crashed multi-GB staging tree instead of leaking it until some
      // other eligible download for the slug runs. This touches only `.staging`
      // trees (never finalDir), so it is safe to run before the finalDir preflight.
      await this.reapDeadStagingTrees();

      if (job.adopt !== null) {
        await this.adoptInPlace(job, job.adopt, slug, (dir) => {
          stagingDir = dir;
        });
        return;
      }
      if (job.locate !== null) {
        await this.locateFromPlan(job, job.locate, slug, (dir) => {
          stagingDir = dir;
        });
        return;
      }

      // Ownership preflight: refuse an unowned/foreign finalDir up front (no bytes
      // move for a doomed job).
      refuseIfUnownedFinal();

      // Recover/reap an orphaned publish backup for THIS model — AFTER the preflight,
      // so a foreign finalDir link was already refused and can't be followed here.
      // Pid-gated: a live peer's active publish backup is never reclaimed.
      await this.recoverBackup(slug, finalDir);

      // Re-run the preflight after recovery: recovery puts back an owned backup or an
      // entry a crashed cleanup moved aside, and this refuses any unowned occupant at finalDir before
      // skip-detection (`readCompletion`/`isModelInstalled`, which FOLLOW symlinks)
      // could read an unowned dir's marker-free state and report a false `done`.
      refuseIfUnownedFinal();

      const at = ++this.resolveSeq;
      const revision = await this.resolveRevision(job.repo);
      this.noteJobRevision(job.repo, revision, at);

      // Job-private staging: keyed by the IMMUTABLE revision (a different
      // revision never reuses another's staged bytes) and unique per invocation
      // (`.<pid>.<uuid>`) so no two processes ever write into the same tree.
      // Resume is HF-cache-backed, not staging-backed, so this dir is disposable
      // and removed on both success and failure.
      stagingDir = join(this.modelsDir, '.staging', `${slug}@${revision}.${process.pid}.${randomUUID()}`);

      // The file-selection fields of the target `start` looked up: a draft and
      // a platform-specific override repo (the pre-converted CUDA build) carry
      // no globs and no sidecar source.
      const selection = selectionOf(job.target);
      const globs = selection.globs?.map(globToRegex);

      const files = await this.listWantedFiles(job.repo, revision, globs, this.wrappedFetch);
      let totalBytes = 0;
      for (const file of files) if (file.size > 0) totalBytes += file.size;

      assertModelPayload(job.repo, files, selection.assetsRepo);

      // Resolve the sidecar manifest BEFORE `start` so the emitted total and
      // file count cover everything this job will fetch. The UI keeps the
      // start event's total, so bytes discovered later would let the bar
      // reach 100% early and report received > total; a listing failure also
      // fails here, before any bytes move.
      const sidecarPlan =
        selection.assetsRepo === undefined ? null : await this.planAssetSidecars(selection.assetsRepo, files);
      if (sidecarPlan !== null) {
        for (const file of sidecarPlan.files) if (file.size > 0) totalBytes += file.size;
      }

      // One aggregate for every frame of this job: the UI reducer replaces
      // `fileCount` with each progress event's value, so a primary-loop frame
      // reporting the primary-only count would shrink the displayed total and
      // then jump when the sidecars start.
      const aggregateFileCount = files.length + (sidecarPlan?.files.length ?? 0);

      job.totalBytes = totalBytes;
      this.emit({
        type: 'start',
        id: job.id,
        repo: job.repo,
        totalBytes,
        fileCount: aggregateFileCount,
      });

      // Already published, complete, AND pinned to THIS repo+revision with every
      // planned primary file listed (marker present): the resume/skip case —
      // re-downloading nothing. Gating on the marker's identity is load-bearing:
      // a complete install of a DIFFERENT revision (or repo) of the same slug must
      // NOT read as done at this revision's byte total, and neither must an
      // install of another build of the same repo and revision, whose files are
      // not this build's. On a mismatch, fall through to download the resolved revision;
      // publish's `isDownloaderOwned`-guarded owned-swap then replaces the stale one
      // (an UNOWNED dir without our marker is still refused there, never destroyed).
      //
      // The marker pins only the PRIMARY repo, so the planned sidecars are
      // verified against their freshly resolved assets manifest before reading
      // this as done: a tokenizer/template fix upstream (the assets repo moves
      // on its own revision) still lands the next time a job runs, instead of
      // being skipped forever. A missing or stale sidecar falls through to the
      // full path, which re-fetches it through the plan.
      // A cleanup always replaces the entry it was confirmed for, so it never reads that entry as done.
      const installed = job.replace === null ? readCompletion(finalDir) : undefined;
      const installedIsCurrent =
        installed !== undefined &&
        installed.repo === job.repo &&
        installed.revision === revision &&
        files.every((file) => installed.files.includes(file.path)) &&
        isModelInstalled(finalDir) &&
        (job.repo !== QWEN38_DFLASH2.hfRepo || isDFlash2Companion(finalDir)) &&
        (sidecarPlan === null || (await this.refreshInstalledAssets(sidecarPlan, finalDir, files, installed)));
      // `refreshInstalledAssets` is the one pre-publish path that mutates the
      // live install (stale-sidecar deletes, marker rewrite) — and only on a
      // real dir: it refuses a symlinked install, so a needed refresh on a
      // linked-in model re-downloads through the link-safe publish swap. A
      // cancel accepted
      // during that await must still win over the done branch — and over the
      // fall-through below, which this single check also covers.
      if (job.cancelled) throw new Error('Download cancelled');
      if (installedIsCurrent) {
        job.receivedBytes = totalBytes;
        job.state = 'done';
        this.emit({ type: 'done', id: job.id, outputDir: finalDir });
        return;
      }

      await mkdir(stagingDir, { recursive: true });
      // Canonicalize the staging dir ONCE (it now exists as a real dir under the
      // symlink-checked root). Per-file write targets are validated against this
      // REAL path rather than a lexical `resolve` (which never resolves symlinks),
      // so a staged write can never land outside the canonical staging tree.
      const stagingReal = realpathSync(stagingDir);

      for (let index = 0; index < files.length; index++) {
        // Stop promptly between files when cancelled (the in-flight fetch of the
        // current file is aborted via the job's controller); the `finally` then
        // removes this staging dir.
        if (job.cancelled) throw new Error('Download cancelled');
        const file = files[index];
        const destPath = join(stagingDir, file.path);
        // Belt-and-braces at the write boundary: even with `isSafeRelPath` gating
        // ingestion, re-assert the target stays inside the CANONICAL stagingDir
        // before any mkdir/copy/read derived from it.
        const destResolved = resolve(stagingReal, file.path);
        if (destResolved !== stagingReal && !destResolved.startsWith(stagingReal + sep)) {
          throw new Error(`Refusing to write "${file.path}" outside the staging directory`);
        }
        const jobBaseBytes = job.receivedBytes;

        if (await isStagedCopyComplete(destPath, file)) {
          const fileBytes = file.size > 0 ? file.size : 0;
          job.receivedBytes = jobBaseBytes + fileBytes;
          this.emitFileProgress(job, file.path, fileBytes, file.size, index, aggregateFileCount);
          continue;
        }

        const context = await this.downloadVerifiedFile(repo, revision, file, destPath, {
          jobBaseBytes,
          fileIndex: index,
          fileCount: aggregateFileCount,
        });

        // Settle the file at its manifest size (or the counted bytes when the
        // manifest omits a size), so aggregate progress is exact even if a
        // cache hit or a metadata-only response streamed nothing.
        const fileBytes = file.size > 0 ? file.size : context.received;
        job.receivedBytes = jobBaseBytes + fileBytes;
        this.emitFileProgress(job, file.path, fileBytes, file.size, index, aggregateFileCount);
      }

      // A cancel issued after the fetch loop must still unwind here — before the
      // prune and the multi-GB whole-set hash — rather than silently install. The
      // `catch` reclassifies this throw as a clean `cancelled` terminal and the
      // `finally` removes the private staging dir.
      if (job.cancelled) throw new Error('Download cancelled');

      // Tokenizer/config sidecars from the entry's assetsRepo (GGUF repos ship
      // no tokenizer). Staged BEFORE the prune/verify below so the sidecars are
      // covered by both AND land in the completion marker's file list — resume
      // and update verification then cover them like any primary file.
      if (sidecarPlan !== null) {
        const sidecars = await this.downloadAssetSidecars(sidecarPlan, stagingDir, stagingReal, files, job);
        files.push(...sidecars);
        // `hasModelPayload` waived the config.json requirement on the promise
        // that the sidecar fetch supplies it. Verify that promise here: a GGUF
        // repo without a config.json whose base repo turns out not to carry
        // one either must fail the job, not publish a weights-only install
        // that renders as not-installed and only a re-run recovers.
        if (!files.some((file) => file.path === 'config.json')) {
          throw new Error(
            `Repo "${job.repo}" is not a complete model (neither it nor its assets repo "${selection.assetsRepo}" provides a config.json)`,
          );
        }
      }

      // Quarantine any staged entry not in the current manifest (a stale file
      // from an earlier interrupted run), so the published set is exactly the
      // manifest — no orphan (e.g. an old single-file weight) can win over it.
      await this.pruneToManifest(stagingDir, files);

      // Re-validate the WHOLE staged set against the manifest immediately before
      // publishing; any mismatch fails the job rather than shipping bad bytes.
      for (const file of files) {
        // Re-check each iteration so a cancel during the (minutes-long, multi-GB)
        // hash unwinds promptly instead of hashing every remaining file first.
        if (job.cancelled) throw new Error('Download cancelled');
        if (!(await isStagedCopyComplete(join(stagingDir, file.path), file))) {
          throw new Error(`Staged file "${file.path}" failed content verification before publish`);
        }
      }

      // Commit barrier: this is the last cancellable instant. Check, then flip to
      // the non-cancellable `committing` state with NO intervening await (atomic in
      // JS) so a cancel can never slip between the guard and publish. Once
      // `committing`, `cancel()`'s `state !== 'running'` gate rejects it (the route
      // 404s) and publish runs to completion.
      if (job.cancelled) throw new Error('Download cancelled');
      if (job.repo === QWEN38_DFLASH2.hfRepo && !isDFlash2Companion(stagingDir)) {
        throw new Error('DFlash2 download requires a DFlash2DraftModel config and model.safetensors');
      }
      job.state = 'committing';

      const { keptDir, removed } = await this.publish(stagingDir, finalDir, job, revision, files, sidecarPlan);

      job.state = 'done';
      this.emit({
        type: 'done',
        id: job.id,
        outputDir: finalDir,
        ...(keptDir !== undefined ? { keptDir } : {}),
        ...(removed !== undefined ? { removed } : {}),
      });
    } catch (error) {
      // A cancelled job unwinds via a boundary check or an aborted in-flight
      // fetch: classify EITHER as a clean `cancelled` terminal (not a red error),
      // so the abort's raw message never surfaces as a failure.
      if (job.cancelled) {
        job.state = 'cancelled';
        this.emit({ type: 'cancelled', id: job.id });
      } else {
        job.state = 'error';
        // The stack goes to the child's stderr (MAIN forwards it as
        // `[mlx] control panel: …`) — bounded like the event text, since a hub
        // error can embed an unbounded remote body inside `message` and stderr
        // is forwarded verbatim. The event carries the cause chain, since the
        // raw message alone is undici's content-free "fetch failed".
        console.error(
          `[downloads] job ${job.id} (${job.repo}) failed:`,
          truncateMessage(error instanceof Error ? (error.stack ?? error.message) : String(error)),
        );
        // A disk that went away under a queued or running job fails it with the plain sentence, not mkdir's ENOTDIR.
        let message = truncateMessage(modelsDirUnavailable(this.modelsDir) ?? describeFailure(error));
        if (job.replace !== null && !job.swapStarted) message = withNothingRemoved(message);
        if (job.adopt !== null && !job.swapStarted) message = withNothingChanged(message);
        if (job.locate !== null && !job.swapStarted) message = withNothingChanged(message);
        this.emit({ type: 'error', id: job.id, message });
      }
    } finally {
      // Job-private staging is scratch: a successful publish already renamed it
      // away (this is a no-op), and on failure it must not leak — resume never
      // reuses it (the HF cache does). Never touches the final dir. This recursive
      // rm cannot escape the store: the root symlink guard proved `.staging` is a
      // real directory, and `stagingDir` is a fresh-uuid child that cannot be a
      // pre-existing symlink — so no per-target canonical re-check is needed here.
      if (stagingDir !== undefined) {
        await rm(stagingDir, { recursive: true, force: true }).catch(() => {});
      }
      this.currentJob = null;
      this.currentFile = null;
    }
  }

  /**
   * Run an Update of the folder at `slug`. It never moves, renames, deletes or overwrites a file that was in the
   * folder: it reads the catalog files there, downloads the missing ones into its own staging tree, and adds them,
   * then the marker, with `link`. In order:
   *
   *   1. compute the plan again through the job's fetch; its token must be the one the user confirmed;
   *   2. hash every file to check where it is ({@link proveInPlace}); a file that is not the catalog file is
   *      remembered, so the next plan refuses it without a new hash, and fails the job once all are checked;
   *   3. download the missing files into staging and verify them;
   *   4. commit barrier: the folder is scanned again and must hold the checked files with the fingerprints the hash
   *      saw, nothing at a path the job adds, and nothing the plan would refuse; the marker, with its `adopted`
   *      record, is written and synced in staging;
   *   5. add each staged file, `config.json` last, then the marker; any failure removes the added names that are still
   *      the job's own files.
   *
   * A crash leaves complete added files and no marker, so the folder stays unowned and the next Update checks them.
   * `setStaging` hands the staging tree to `processJob`, whose `finally` removes it; that unlinks only second names.
   */
  private async adoptInPlace(
    job: JobState,
    confirmed: AdoptPlan,
    slug: string,
    setStaging: (dir: string) => void,
  ): Promise<void> {
    const target = modelsChildPath(this.modelsDir, slug, 'update');
    const at = ++this.resolveSeq;
    const plan = await this.computeAdoptPlan(job.target, this.wrappedFetch);
    this.noteJobRevision(job.repo, plan.revision, at);
    if (plan.token !== confirmed.token) {
      throw new Error(`"${slug}" or the catalog changed after you confirmed. ${NOTHING_CHANGED} Use Update again.`);
    }

    const sizeOf = (file: ListFileEntry): number => (file.size > 0 ? file.size : 0);
    const fileCount = plan.check.length + plan.download.length;
    job.totalBytes =
      plan.check.reduce((sum, item) => sum + sizeOf(item.file), 0) +
      plan.download.reduce((sum, item) => sum + sizeOf(item.file), 0);
    this.emit({ type: 'start', id: job.id, repo: job.repo, totalBytes: job.totalBytes, fileCount });

    const proven = new Map<string, FileFingerprint>();
    const mismatched: string[] = [];
    for (let index = 0; index < plan.check.length; index++) {
      if (job.cancelled) throw new Error('Download cancelled');
      const item = plan.check[index];
      const size = sizeOf(item.file);
      const base = job.receivedBytes;
      let read = 0;
      let shownAt = 0;
      const proof = await proveInPlace(target, item.file, item, {
        cancelled: () => job.cancelled,
        onBytes: (n) => {
          read = Math.min(size, read + n);
          job.receivedBytes = base + read;
          const now = Date.now();
          if (now - shownAt < CHECK_PROGRESS_INTERVAL_MS) return;
          shownAt = now;
          this.emitFileProgress(job, item.file.path, read, item.file.size, index, fileCount, 'check');
        },
      });
      if (proof.match) {
        proven.set(item.file.path, proof.fingerprint);
      } else {
        mismatched.push(item.file.path);
        this.mismatchMemo.set(`${slug}\0${item.file.path}`, {
          size: item.size,
          mtimeNs: item.mtimeNs,
          ino: item.ino,
          oid: checksumOf(item.file)!.oid,
        });
      }
      job.receivedBytes = base + size;
      this.emitFileProgress(job, item.file.path, size, item.file.size, index, fileCount, 'check');
    }
    if (mismatched.length > 0) {
      throw new Error(
        `"${mismatched.join('", "')}" in "${slug}" is not the catalog file. ${NOTHING_CHANGED} ` +
          `Move it out of the folder, then use Update again.`,
      );
    }

    const stagingDir = join(this.modelsDir, '.staging', `${slug}@${plan.revision}.${process.pid}.${randomUUID()}`);
    setStaging(stagingDir);
    await mkdir(stagingDir, { recursive: true });
    const stagingReal = realpathSync(stagingDir);
    for (let n = 0; n < plan.download.length; n++) {
      if (job.cancelled) throw new Error('Download cancelled');
      const item = plan.download[n];
      const destResolved = resolve(stagingReal, item.file.path);
      if (destResolved !== stagingReal && !destResolved.startsWith(stagingReal + sep)) {
        throw new Error(`Refusing to write "${item.file.path}" outside the staging directory`);
      }
      const jobBaseBytes = job.receivedBytes;
      const fileIndex = plan.check.length + n;
      const context = await this.downloadVerifiedFile(
        { type: 'model', name: item.repo },
        item.revision,
        item.file,
        join(stagingDir, item.file.path),
        { jobBaseBytes, fileIndex, fileCount },
      );
      const fileBytes = item.file.size > 0 ? item.file.size : context.received;
      job.receivedBytes = jobBaseBytes + fileBytes;
      this.emitFileProgress(job, item.file.path, fileBytes, item.file.size, fileIndex, fileCount);
    }
    for (const item of plan.download) {
      if (job.cancelled) throw new Error('Download cancelled');
      if (!(await isStagedCopyComplete(join(stagingDir, item.file.path), item.file))) {
        throw new Error(`Staged file "${item.file.path}" failed content verification before publish`);
      }
    }

    // Same commit barrier as an install: checked, then `committing` with no await between, so a cancel cannot
    // report success while files are being added.
    if (job.cancelled) throw new Error('Download cancelled');
    job.state = 'committing';
    await this.assertFolderAsChecked(target, slug, plan, proven);

    const staged = new Map<string, FileFingerprint>();
    const adopted: Record<string, { size: number; mtimeNs: string }> = {};
    for (const [path, fingerprint] of proven) {
      adopted[path] = { size: Number(fingerprint.size), mtimeNs: fingerprint.mtimeNs.toString() };
    }
    for (const item of plan.download) {
      const fingerprint = fingerprintOf(await lstat(join(stagingDir, item.file.path), { bigint: true }));
      staged.set(item.file.path, fingerprint);
      adopted[item.file.path] = { size: Number(fingerprint.size), mtimeNs: fingerprint.mtimeNs.toString() };
    }
    const marker: DownloadCompletion = {
      repo: job.repo,
      revision: plan.revision,
      files: plan.wanted.map((item) => item.file.path),
      scope: 'full',
      ...(plan.assetsRepo !== null && plan.assetsRevision !== null
        ? { assetsRepo: plan.assetsRepo, assetsRevision: plan.assetsRevision }
        : {}),
      completedAt: new Date().toISOString(),
      adopted,
    };
    const stagedMarker = join(stagingDir, DOWNLOAD_COMPLETE_MARKER);
    await writeFile(stagedMarker, `${JSON.stringify(marker, null, 2)}\n`, { flag: 'wx' });
    await fsyncFile(stagedMarker);
    const markerFingerprint = fingerprintOf(await lstat(stagedMarker, { bigint: true }));

    // The marker must go last: until it is there the folder is not owned, so a crash or a failed add leaves it to
    // the next Update and never to an owned swap that deletes from it. config.json goes last of the files, so a
    // config.json the job adds always has the other added files beside it.
    const order = [
      ...plan.download.filter((item) => item.file.path !== 'config.json'),
      ...plan.download.filter((item) => item.file.path === 'config.json'),
    ];
    const added: AddedName[] = [];
    const madeDirs: string[] = [];
    job.swapStarted = true;
    try {
      for (const item of order) {
        const path = item.file.path;
        await linkInto(target, slug, path, join(stagingDir, path), staged.get(path)!, added, madeDirs);
      }
      if (job.repo === QWEN38_DFLASH2.hfRepo && !isDFlash2Companion(target)) {
        throw new Error('DFlash2 download requires a DFlash2DraftModel config and model.safetensors');
      }
      // The adds and the fsync above take milliseconds; a checked file changed in that time must not get a marker.
      const [root, ...checked] = await Promise.all(
        [target, ...plan.check.map((item) => join(target, item.file.path))].map((path) =>
          lstat(path, { bigint: true }).catch(() => null),
        ),
      );
      if (
        root === null ||
        root.dev !== plan.folder.dev ||
        root.ino !== plan.folder.ino ||
        checked.some(
          (stat, i) =>
            stat === null ||
            !stat.isFile() ||
            !sameFingerprint(fingerprintOf(stat), proven.get(plan.check[i].file.path)!),
        )
      ) {
        throw new Error(`"${slug}" changed on disk during the update.`);
      }
      await linkInto(target, slug, DOWNLOAD_COMPLETE_MARKER, stagedMarker, markerFingerprint, added, madeDirs);
      await fsyncDir(target);
      if (!isModelInstalled(target)) throw new Error(`"${slug}" changed on disk during the update.`);
    } catch (error) {
      const keptDir = join(this.modelsDir, '.staging', `${slug}${KEPT_MARK}${randomUUID()}`);
      const { stays, kept } = await removeAddedNames(target, added, madeDirs, keptDir);
      if (stays.length > 0 || kept !== undefined) {
        // With config.json or the marker in it the folder reads as installed, and its card offers no Update.
        const blocking = stays.filter((path) => path === 'config.json' || path === DOWNLOAD_COMPLETE_MARKER);
        const next =
          blocking.length > 0
            ? `Remove ${blocking.join(' and ')} from the folder, then use Update again.`
            : 'Use Update again.';
        const left = stays.length > 0 ? `These files that it added stay in "${slug}": ${stays.join(', ')}. ` : '';
        const yours =
          kept === undefined
            ? 'No file of yours was changed. '
            : `A file that something put at one of the names it added while it removed them is now in "${kept}". ` +
              `Move it back if it is yours. `;
        throw new Error(`Update stopped. ${left}${yours}${next}`);
      }
      job.swapStarted = false;
      throw error;
    }

    job.receivedBytes = job.totalBytes;
    job.state = 'done';
    this.emit({ type: 'done', id: job.id, outputDir: target });
  }

  /**
   * Run a Locate: build the app's folder for `slug` from the files of the folder the user located, without writing
   * into that folder. In order:
   *
   *   1. compute the plan again through the job's fetch; its token must be the one the user confirmed;
   *   2. the folder at the slug must still be free (or the app's own folder with no loadable model);
   *   3. unless the folder has this app's record, hash every used weight-like file where it is ({@link proveAt});
   *      a file that is not the catalog file is remembered, so the next plan refuses it without a new hash;
   *   4. in a staging tree, give each used weight-like file a second name (`link`) or a link (`symlink`), and make the
   *      app's own copy of each used small file; every create fails when the name exists;
   *   5. download the missing files into staging;
   *   6. barrier: each staged weight-like entry still is the file that was hashed, and the tree holds a model;
   *   7. publish: the record, with `located`, then the rename onto the slug.
   *
   * `setStaging` hands the staging tree to `processJob`, whose `finally` removes it; that unlinks names only.
   */
  private async locateFromPlan(
    job: JobState,
    confirmed: LocatePlan,
    slug: string,
    setStaging: (dir: string) => void,
  ): Promise<void> {
    const target = job.target;
    if (target.kind !== 'build') throw new Error('Locate is not offered for companion weights.');
    const at = ++this.resolveSeq;
    const plan = await this.computeLocatePlan(target, confirmed.picked, this.wrappedFetch);
    if (!plan.byRecord) this.noteJobRevision(job.repo, plan.revision, at);
    if (plan.token !== confirmed.token) throw new Error(LOCATE_CHANGED);
    const finalDir = this.assertLocateSlugFree(slug);
    await this.recoverBackup(slug, finalDir);
    this.assertLocateSlugFree(slug);

    const sizeOf = (file: ListFileEntry): number => (file.size > 0 ? file.size : 0);
    const checks = plan.byRecord ? [] : plan.use.filter((item) => isLinkableFile(item.path));
    const fileCount = checks.length + plan.download.length;
    job.totalBytes =
      checks.reduce((sum, item) => sum + Number(item.fingerprint.size), 0) +
      plan.download.reduce((sum, item) => sum + sizeOf(item.file), 0);
    this.emit({ type: 'start', id: job.id, repo: job.repo, totalBytes: job.totalBytes, fileCount });

    const stagingDir = join(this.modelsDir, '.staging', `${slug}@${plan.revision}.${process.pid}.${randomUUID()}`);
    setStaging(stagingDir);
    await mkdir(stagingDir, { recursive: true });
    const stagingReal = realpathSync(stagingDir);
    const stagedPath = (path: string): string => {
      const resolved = resolve(stagingReal, path);
      if (resolved === stagingReal || !resolved.startsWith(stagingReal + sep)) {
        throw new Error(`Refusing to write "${path}" outside the staging directory`);
      }
      return join(stagingDir, path);
    };

    const mismatched: string[] = [];
    for (let index = 0; index < checks.length; index++) {
      if (job.cancelled) throw new Error('Download cancelled');
      const item = checks[index]!;
      const file = item.file!;
      const name = basename(item.path);
      const size = Number(item.fingerprint.size);
      const base = job.receivedBytes;
      let read = 0;
      let shownAt = 0;
      const proof = await proveAt(
        item.realPath,
        file,
        { size, mtimeNs: item.fingerprint.mtimeNs, ino: item.fingerprint.ino },
        {
          cancelled: () => job.cancelled,
          onBytes: (n) => {
            read = Math.min(size, read + n);
            job.receivedBytes = base + read;
            const now = Date.now();
            if (now - shownAt < CHECK_PROGRESS_INTERVAL_MS) return;
            shownAt = now;
            this.emitFileProgress(job, item.path, read, file.size, index, fileCount, 'check');
          },
        },
        () => new Error(`"${name}" changed during the check. Use Locate again.`),
        () => new Error(`The catalog gives no checksum for "${name}", so Locate cannot check it.`),
      );
      if (!proof.match) {
        mismatched.push(name);
        this.mismatchMemo.set(`locate\0${item.realPath}`, {
          size,
          mtimeNs: item.fingerprint.mtimeNs,
          ino: item.fingerprint.ino,
          oid: checksumOf(file)!.oid,
        });
      }
      job.receivedBytes = base + size;
      this.emitFileProgress(job, item.path, size, file.size, index, fileCount, 'check');
    }
    if (mismatched.length > 0) {
      throw new Error(
        `"${mismatched.join('", "')}" in ${plan.folder} is not the catalog file: its content is different.`,
      );
    }

    for (const item of plan.use) {
      if (job.cancelled) throw new Error('Download cancelled');
      const staged = stagedPath(item.path);
      await mkdir(dirname(staged), { recursive: true });
      if (item.how === 'copy') {
        await copyFile(item.realPath, staged, constants.COPYFILE_EXCL);
        if (item.file !== null && !(await isStagedCopyComplete(staged, item.file))) {
          throw new Error(`"${basename(item.path)}" changed during the check. Use Locate again.`);
        }
        continue;
      }
      try {
        if (item.how === 'link') await link(item.realPath, staged);
        else await symlink(item.realPath, staged);
      } catch (error) {
        if (item.how !== 'link') throw error;
        const code = (error as NodeJS.ErrnoException).code ?? String(error);
        throw new Error(`This disk cannot share a file between two folders (${code}).`);
      }
    }

    for (let n = 0; n < plan.download.length; n++) {
      if (job.cancelled) throw new Error('Download cancelled');
      const item = plan.download[n]!;
      const destPath = stagedPath(item.file.path);
      const jobBaseBytes = job.receivedBytes;
      const fileIndex = checks.length + n;
      const context = await this.downloadVerifiedFile(
        { type: 'model', name: item.repo },
        item.revision,
        item.file,
        destPath,
        { jobBaseBytes, fileIndex, fileCount },
      );
      const fileBytes = item.file.size > 0 ? item.file.size : context.received;
      job.receivedBytes = jobBaseBytes + fileBytes;
      this.emitFileProgress(job, item.file.path, fileBytes, item.file.size, fileIndex, fileCount);
    }

    // The barrier: a file replaced after its hash, or a staged name that is not what the job made, is not published.
    if (job.cancelled) throw new Error('Download cancelled');
    for (const item of plan.use) {
      if (item.how === 'copy') continue;
      const staged = join(stagingDir, item.path);
      const entry = await lstat(staged).catch(() => null);
      const now = await foundAt(staged);
      const kindHolds = entry !== null && (item.how === 'link' ? entry.isFile() : entry.isSymbolicLink());
      if (
        !kindHolds ||
        now === null ||
        now.fingerprint.dev !== item.fingerprint.dev ||
        now.fingerprint.ino !== item.fingerprint.ino ||
        now.fingerprint.size !== item.fingerprint.size ||
        now.fingerprint.mtimeNs !== item.fingerprint.mtimeNs
      ) {
        throw new Error(`"${basename(item.path)}" changed during the check. Use Locate again.`);
      }
    }
    if (!isModelPresent(stagingDir)) throw new Error(LOCATE_CHANGED);

    if (job.cancelled) throw new Error('Download cancelled');
    job.state = 'committing';
    const sidecar =
      plan.assetsRepo !== null && plan.assetsRevision !== null
        ? { repo: { name: plan.assetsRepo }, revision: plan.assetsRevision }
        : null;
    await this.publish(
      stagingDir,
      finalDir,
      job,
      plan.revision,
      plan.wanted.map((path) => ({ path })),
      sidecar,
      { from: plan.folder, mode: plan.mode },
    );
    job.receivedBytes = job.totalBytes;
    job.state = 'done';
    this.emit({ type: 'done', id: job.id, outputDir: finalDir });
  }

  /**
   * The commit barrier of an Update: the folder is the one the plan saw, still not owned, holds every checked file
   * with the fingerprint its hash saw, holds nothing at a path the job adds, and holds nothing the plan refuses.
   * Other files may have come or gone.
   */
  private async assertFolderAsChecked(
    target: string,
    slug: string,
    plan: AdoptPlan,
    proven: ReadonlyMap<string, FileFingerprint>,
  ): Promise<void> {
    const changed = (): Error => new Error(`"${slug}" changed on disk during the update.`);
    if (!isRealDirectory(target) || isDownloaderOwned(target)) throw changed();
    let scan: Awaited<ReturnType<typeof scanFolder>>;
    try {
      scan = await scanFolder(target, slug);
    } catch (error) {
      throw error instanceof DownloadConflictError ? changed() : error;
    }
    const { folder, entries } = scan;
    if (folder.dev !== plan.folder.dev || folder.ino !== plan.folder.ino) throw changed();
    const byPath = new Map(entries.map((entry) => [entry.path, entry]));
    for (const item of plan.download) {
      if (byPath.has(item.file.path)) throw new Error(`"${item.file.path}" appeared in "${slug}" during the update.`);
    }
    for (const item of plan.check) {
      const entry = byPath.get(item.file.path);
      const seen = proven.get(item.file.path);
      if (entry === undefined || entry.isDirectory || seen === undefined || !sameFingerprint(entry.stat, seen)) {
        throw changed();
      }
    }
    try {
      classifyFolder(entries, plan.wanted, this.mismatchMemo, slug);
    } catch (error) {
      throw error instanceof DownloadConflictError ? changed() : error;
    }
  }

  /**
   * PROCESS-WIDE reap of crashed jobs' private staging trees
   * (`<slug>@<sha>.<pid>.<uuid>`, every slug), gated entirely by pid-liveness — no
   * interprocess lock. Such a tree is cleaned only in its live owner's `finally`, so
   * a crash/SIGKILL leaks it forever otherwise; it is reaped when its owner pid is
   * gone. A live — or reused — pid is kept; pid-reuse only preserves a leak, it can
   * never reap a live job. This touches ONLY `.staging` trees (never a final dir),
   * so it carries no finalDir ownership dependency and runs BEFORE the ownership
   * preflight, ensuring a preflight-refused job still frees a crashed tree. Never
   * throws.
   */
  private async reapDeadStagingTrees(): Promise<void> {
    const stagingRoot = join(this.modelsDir, '.staging');
    let entries: string[];
    try {
      entries = await readdir(stagingRoot);
    } catch {
      return;
    }
    for (const name of entries) {
      const staleMatch = STAGING_DIR_RE.exec(name);
      if (staleMatch !== null && !pidAlive(Number(staleMatch[1]))) {
        await rm(join(stagingRoot, name), { recursive: true, force: true }).catch(() => {});
      }
    }
  }

  /**
   * SLUG-SCOPED recover/reap of an orphaned publish backup for THIS model
   * (`<slug>.backup-<pid>.<uuid>`), gated by pid-liveness. Runs AFTER the ownership
   * preflight (which already refused an unowned `finalDir` occupant), so nothing
   * here follows a foreign link into an unmarked dir. A backup whose owner pid is
   * still ALIVE belongs to a peer mid-publish and is never touched. For a
   * dead-owner backup, classify finalDir NO-FOLLOW first — `lstat` reports any
   * `finalDir` symlink as not-a-directory, so a link into an external marked model
   * can never be read as a complete install and REAP a recoverable rollback
   * backup:
   *
   *   - finalDir is a proven owned-complete REAL dir → the backup is redundant → rm;
   *   - finalDir is ABSENT (a crashed swap) AND the backup proves OWNED → restore
   *     it by renaming it back into place. Ownership here is `isDownloaderOwned`,
   *     which stat-FOLLOWS a symlink: an owned REAL dir and an owned LINK into a
   *     marked install both roll back. Restoring the link is safe because the
   *     publish swap can legitimately rename an admitted symlinked finalDir to its
   *     backup — the rollback reinstates exactly what the ownership preflight
   *     admitted — and post-restore skip-detection still re-reads the target's
   *     marker against THIS job's repo+revision, so a link into a marked install
   *     of a different model is just re-swapped aside, never reported `done`. A
   *     FOREIGN-symlink backup (unmarked or dangling target) fails ownership and
   *     is never promoted;
   *   - otherwise (foreign symlink / unowned dir / incomplete-owned dir, OR — with
   *     finalDir absent — a non-owned backup) → RETAIN the backup untouched
   *     and never clobber whatever occupies finalDir. Never throws.
   */
  private async recoverBackup(slug: string, finalDir: string): Promise<void> {
    const stagingRoot = join(this.modelsDir, '.staging');
    let entries: string[];
    try {
      entries = await readdir(stagingRoot);
    } catch {
      return;
    }
    const backupPrefix = `${slug}.backup-`;
    for (const name of entries) {
      if (!name.startsWith(backupPrefix)) continue;
      const backupMatch = BACKUP_DIR_RE.exec(name);
      // Fail closed on an unparseable owner: a name that carries the backup prefix
      // but no readable pid is left untouched (never promoted onto finalDir, never
      // reaped), mirroring `pidAlive`'s unknown-owner→alive stance. Without this,
      // the null match would short-circuit the live-PID guard below and treat the
      // dir as owner-dead.
      if (backupMatch === null) continue;
      // A backup whose owner pid is still alive is a peer mid-publish: never touch it.
      if (pidAlive(Number(backupMatch[1]))) continue;
      const backupPath = join(stagingRoot, name);
      let finalOwnedComplete = false;
      let finalAbsent = false;
      try {
        const st = lstatSync(finalDir);
        finalOwnedComplete = st.isDirectory() && isDownloaderOwned(finalDir) && isModelInstalled(finalDir);
      } catch {
        finalAbsent = true;
      }
      try {
        if (finalOwnedComplete) {
          const kept = await this.removeOwnedBackup(backupPath, slug); // redundant backup
          if (kept !== undefined) console.error(`[downloads] files of an old "${slug}" were kept in "${kept}"`);
        } else if (finalAbsent && isDownloaderOwned(backupPath)) {
          // Restore a crashed swap. `isDownloaderOwned` stat-follows a symlink
          // backup into the marked install it points at — the same proof that
          // admitted that link as finalDir — so owned real dirs and owned links
          // both roll back, while foreign/dangling links fail ownership.
          await rename(backupPath, finalDir);
        }
        // else foreign symlink / unowned dir / incomplete-owned → retain untouched
      } catch {
        // A single un-recoverable backup must not block the job.
      }
    }
    await this.recoverCleanupHolders(slug);
  }

  /**
   * Settle the holders a cleanup left when its process died (`<slug>.cleanup-<pid>.<ino>.<uuid>`, owner pid dead), for
   * one slug or for every slug when `onlySlug` is null. At manager start no job of this process exists yet, so a holder
   * with this process's pid is dead too (`ownPidDead`). Looks at the final path without following a link:
   *
   *   - an empty holder: the cleanup had finished, so the holder is removed;
   *   - the old entry in the holder and nothing at the path: the crash came before the new model was in place, so the
   *     old entry is put back ({@link putBack}; when something took the path meanwhile it is kept as `.kept-`);
   *   - the old entry in the holder and, at the path, the very folder the cleanup renamed in (the holder's device and
   *     the inode in its name): the cleanup is finished as the job would have (a link unlinked, anything else to the
   *     Trash, else kept);
   *   - the old entry in the holder and another complete model this app installed at the path: the old entry is kept
   *     as `.kept-`, never removed;
   *   - anything else: everything is left as it is.
   *
   * Never removes anything with `rm`. Never throws.
   */
  private async recoverCleanupHolders(onlySlug: string | null, ownPidDead = false): Promise<void> {
    const stagingRoot = join(this.modelsDir, '.staging');
    let entries: string[];
    try {
      if (!lstatSync(stagingRoot).isDirectory()) return;
      entries = await readdir(stagingRoot);
    } catch {
      return;
    }
    for (const name of entries) {
      const parsed = parseCleanupHolder(name);
      if (parsed === null || (onlySlug !== null && parsed.slug !== onlySlug)) continue;
      if (!(ownPidDead && parsed.pid === process.pid) && pidAlive(parsed.pid)) continue;
      const holder = join(stagingRoot, name);
      try {
        const holderStat = lstatSync(holder, { bigint: true });
        if (!holderStat.isDirectory()) continue;
        const inside = await readdir(holder);
        if (inside.length === 0) {
          await rmdir(holder);
          continue;
        }
        if (inside.length !== 1 || inside[0] !== parsed.slug) continue;
        const finalDir = modelsChildPath(this.modelsDir, parsed.slug, 'restore');
        const old = join(holder, parsed.slug);
        let finalStat;
        try {
          finalStat = lstatSync(finalDir, { bigint: true });
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== 'ENOENT') continue;
          try {
            await putBack(old, finalDir);
          } catch (putBackError) {
            if ((putBackError as NodeJS.ErrnoException).code !== 'EEXIST') throw putBackError;
            const keptDir = await this.keepHolder(holder, parsed.slug);
            console.error(`[downloads] something took the path of "${parsed.slug}"; the old one is in "${keptDir}"`);
            continue;
          }
          await rmdir(holder);
          continue;
        }
        if (!finalStat.isDirectory()) continue;
        if (finalStat.dev === holderStat.dev && finalStat.ino === parsed.ino) {
          await this.removeReplaced(holder, parsed.slug, lstatSync(old).isSymbolicLink());
          continue;
        }
        if (!isDownloaderOwned(finalDir) || !isModelInstalled(finalDir)) continue;
        const keptDir = await this.keepHolder(holder, parsed.slug);
        console.error(
          `[downloads] the model at "${finalDir}" is not the one a cleanup installed; the old one is in "${keptDir}"`,
        );
      } catch {
        // One holder that cannot be settled must not block the rest.
      }
    }
  }

  /** Rename a cleanup holder to `.staging/<slug>.kept-<uuid>`, which nothing removes; answers where it is now. */
  private async keepHolder(holder: string, slug: string): Promise<string> {
    const keptDir = join(this.modelsDir, '.staging', `${slug}${KEPT_MARK}${randomUUID()}`);
    try {
      await rename(holder, keptDir);
      return keptDir;
    } catch {
      return holder;
    }
  }

  /**
   * Remove the old entry a cleanup moved to `<holder>/<slug>` once the new model is in place: a link is unlinked after
   * an `lstat` that says link; anything else goes to the Trash, and it went only when `lstat` of it answers ENOENT
   * while the holder is still the same folder (an unreachable disk also answers ENOENT for the entry). On any other
   * outcome the holder is kept ({@link keepHolder}) and the core log names where. Never calls `rm`, never throws.
   */
  private async removeReplaced(
    holder: string,
    slug: string,
    isLink: boolean,
  ): Promise<{ removed?: 'link' | 'trash'; keptDir?: string }> {
    const old = join(holder, slug);
    let removed: 'link' | 'trash' | undefined;
    let holderReplaced = false;
    try {
      if (isLink) {
        if ((await lstat(old)).isSymbolicLink()) {
          await unlink(old);
          removed = 'link';
        }
      } else {
        const holderBefore = entryIdentity(holder);
        try {
          await this.moveToTrash(old);
        } catch (error) {
          console.error(`[downloads] the Trash refused "${old}":`, error instanceof Error ? error.message : error);
        }
        const holderAfter = entryIdentity(holder);
        holderReplaced = holderBefore !== null && holderAfter !== null && !sameIdentity(holderAfter, holderBefore);
        if (holderBefore !== null && isGone(old) && sameIdentity(holderAfter, holderBefore)) {
          removed = 'trash';
        }
      }
    } catch {
      removed = undefined;
    }
    if (removed !== undefined) {
      await rmdir(holder).catch(() => {});
      return { removed };
    }
    // Another folder at the holder path is not the holder: renaming it would name a folder without the old entry.
    const keptDir = holderReplaced ? holder : await this.keepHolder(holder, slug);
    console.error(`[downloads] the old "${slug}" was kept in "${keptDir}"`);
    return { keptDir };
  }

  /**
   * Download one manifest file into staging and verify the STAGED bytes against
   * the manifest (size + strongest advertised identity — the same check resume
   * uses). A mismatch drops the bad copy and re-fetches, bounded by
   * {@link MAX_VERIFY_ATTEMPTS}; exhaustion throws so the job errors rather than
   * publishing unverified content. A hard download error propagates immediately
   * (not a corruption to retry). Returns the final attempt's byte context.
   */
  private async downloadVerifiedFile(
    repo: { type: 'model'; name: string },
    revision: string,
    file: ListFileEntry,
    destPath: string,
    meta: { jobBaseBytes: number; fileIndex: number; fileCount: number },
  ): Promise<FileContext> {
    let context: FileContext | null = null;
    for (let attempt = 1; attempt <= MAX_VERIFY_ATTEMPTS; attempt++) {
      context = {
        filePath: file.path,
        fileSize: file.size,
        fileIndex: meta.fileIndex,
        fileCount: meta.fileCount,
        jobBaseBytes: meta.jobBaseBytes,
        perUrl: new Map(),
        received: 0,
      };
      this.currentFile = context;
      let snapshotPath = '';
      try {
        snapshotPath = await downloadFileToCacheDir({
          repo,
          path: file.path,
          revision,
          cacheDir: this.cacheDir,
          fetch: this.wrappedFetch,
        });
        await mkdir(dirname(destPath), { recursive: true });
        // A new file, never a write through a name that is there: that name can be a second name of a user file.
        await rm(destPath, { force: true });
        await copyFile(snapshotPath, destPath, constants.COPYFILE_EXCL);
      } finally {
        this.currentFile = null;
      }

      if (await isStagedCopyComplete(destPath, file)) return context;
      // A cancel that lands mid-verify must never mutate the shared HF cache (the
      // `mlx download` CLI resumes from it): unwind here — before the staged drop
      // and the pointer/blob invalidation below — so cancel never purges a blob.
      if (this.currentJob?.cancelled) throw new Error('Download cancelled');
      // Content did not match the manifest: drop the staged copy AND invalidate
      // the HF cache entry (the snapshot pointer + the blob it resolves to) so the
      // next attempt actually re-fetches. For a pinned commit `downloadFileToCacheDir`
      // returns the cached pointer without revalidating, so without this the retry
      // would recopy the same corrupt bytes.
      await rm(destPath, { force: true });
      // Invalidate the HF cache entry so the next attempt re-fetches, but never
      // delete anything outside the managed cache. Contain BOTH ends:
      //   - the POINTER by canonicalizing its PARENT (never follow the final
      //     symlink component) — a pointer whose parent escapes the cache is not
      //     unlinked at all;
      //   - the BLOB by canonicalizing the pointer's target — `etag`/`oid` is
      //     server-controlled and can contain `../`, so the blob may resolve
      //     OUTSIDE the cache (a poisoned repo, or a foreign symlink already in
      //     the shared HF cache); delete it ONLY when contained under the cache.
      const cacheRoot = await realpath(this.cacheDir).catch(() => undefined);
      const pointerParent = await realpath(dirname(snapshotPath)).catch(() => undefined);
      const inCache = (p: string | undefined): p is string =>
        cacheRoot !== undefined && p !== undefined && (p === cacheRoot || p.startsWith(cacheRoot + sep));
      // Out-of-cache pointer is impossible via the hub, but never delete a foreign
      // file. Skip invalidation; the bounded retry then surfaces a clean verify error.
      if (inCache(pointerParent)) {
        const blob = await realpath(snapshotPath).catch(() => undefined);
        await rm(snapshotPath, { force: true });
        if (inCache(blob)) await rm(blob, { force: true });
      }
    }
    throw new Error(`Downloaded file "${file.path}" failed content verification after ${MAX_VERIFY_ATTEMPTS} attempts`);
  }

  /**
   * Resolve which tokenizer/config sidecar files a GGUF repo is missing,
   * from the catalog entry's `assetsRepo`, pinned to ONE resolved commit (the
   * same immutable-snapshot discipline as the primary repo).
   *
   * Selection per {@link ASSET_SIDECAR_CANDIDATES} name: skipped when the name
   * is absent from the assetsRepo; skipped when the primary repo already
   * ships the same path — identical content is a pure duplicate, and on a
   * content conflict the PRIMARY repo's own file wins (it is the repo the
   * user asked for; the assetsRepo only fills gaps).
   *
   * Split from {@link downloadAssetSidecars} so `processJob` can resolve the
   * plan BEFORE emitting `start`: the UI keeps the start event's byte total,
   * so sidecar bytes discovered later would let the bar reach 100% early and
   * report received > total, and a late subscriber would replay the stale
   * total. A listing failure throws here — before any bytes move — rather
   * than publishing a GGUF without the official tokenizer.
   */
  private async planAssetSidecars(
    assetsRepo: string,
    primaryFiles: readonly ListFileEntry[],
    fetchImpl: typeof fetch = this.wrappedFetch,
  ): Promise<SidecarPlan> {
    const repo = { type: 'model' as const, name: assetsRepo };
    const revision = await this.resolveRevision(assetsRepo, fetchImpl);
    const byName = new Map<string, ListFileEntry>();
    for await (const file of listFiles({ repo, recursive: true, revision, fetch: fetchImpl })) {
      if (file.type === 'directory') continue;
      // Candidates are all root-level names; a nested same-named file is not
      // the base model's tokenizer/config.
      if (!file.path.includes('/')) byName.set(file.path, file);
    }

    const files: ListFileEntry[] = [];
    for (const name of ASSET_SIDECAR_CANDIDATES) {
      const file = byName.get(name);
      if (file === undefined) continue; // this family does not ship that file
      if (primaryFiles.some((staged) => staged.path === name)) continue; // primary repo ships it — primary wins
      files.push(file);
    }
    return { repo, revision, files };
  }

  /**
   * Decide whether an installed model's sidecars are already up to date, and
   * bring its bookkeeping up to date when they are. Returns false when a
   * planned sidecar is missing or stale — the caller falls through to the full
   * re-fetch path. When every planned sidecar verifies:
   *
   *   - sidecars the marker records that the assets repo no longer supplies
   *     (its candidate name vanished from the listing) are dropped from the
   *     marker and deleted: a repo that deleted a file must not leave that
   *     file installed forever;
   *   - the marker's `assetsRevision` is rewritten to the revision just
   *     verified against. Update discovery compares revisions, so an
   *     assets-repo advance that changed nothing installable (README, model
   *     weights) would otherwise raise a badge that can never clear: the job
   *     verifies the bytes, finds them current, and used to return without
   *     recording that verification — the card offered the same update again.
   *
   *   - a SYMLINKED `dir` is refused the refresh outright (returns false):
   *     the deletions and the marker write follow the link and would mutate
   *     the foreign target — the exact boundary `deleteLocalModel` draws by
   *     unlinking only the link. Returning false sends the job down the full
   *     download path, whose publish swap replaces the LINK with an owned real
   *     directory (rename moves the link aside, `rm` unlinks it), so a needed
   *     refresh still lands without ever writing into the external tree.
   *
   * The deletions run BEFORE the marker is published, so an interrupted
   * refresh fails with the marker untouched and retries (see the ordering note
   * in the body).
   */
  private async refreshInstalledAssets(
    plan: { repo: { type: 'model'; name: string }; revision: string; files: ListFileEntry[] },
    dir: string,
    primaryFiles: readonly ListFileEntry[],
    installed: DownloadCompletion,
  ): Promise<boolean> {
    for (const file of plan.files) {
      if (!(await isStagedCopyComplete(join(dir, file.path), file))) return false;
    }
    const plannedNames = new Set(plan.files.map((file) => file.path));
    const primaryNames = new Set(primaryFiles.map((file) => file.path));
    // Only the fixed sidecar-candidate names, only files the primary repo does
    // not supply, and only ones the assets repo no longer offers: a user's own
    // file, or a file another install path contributed, is never touched.
    const stale = installed.files.filter(
      (file) => ASSET_SIDECAR_CANDIDATE_SET.has(file) && !primaryNames.has(file) && !plannedNames.has(file),
    );
    // Repo identity participates in "verified clean" too: an upstream repo
    // TRANSFER keeps history, so the new source's HEAD is the sha the marker
    // already records. Returning here would leave `assetsRepo` naming the old
    // repo forever — a badge update discovery can never clear.
    if (stale.length === 0 && installed.assetsRepo === plan.repo.name && installed.assetsRevision === plan.revision)
      return true;
    // Everything past this line MUTATES `dir`. `lstat` (no-follow, same guard
    // as `assertRealDirOrAbsent` and `recoverBackup`) refuses it when `dir` is
    // a symlink into a marked external install: a live link is admitted as
    // downloader-owned and a same-revision marker reads as current, so it
    // reaches here — but nothing the downloader manages may write through it.
    // Returning false re-downloads and publishes an owned real dir in the
    // link's place; the external tree is never touched. A `dir` that vanished
    // between the marker read and now takes the same path (a refresh cannot
    // run on nothing, and publish handles the absent case).
    let finalIsLink = false;
    try {
      finalIsLink = lstatSync(dir).isSymbolicLink();
    } catch {
      finalIsLink = true;
    }
    if (finalIsLink) return false;
    // A folder Update adopted may hold the user's own file at a stale sidecar's name. The full path re-downloads, and
    // its swap deletes from the old folder only what the `adopted` record proves.
    if (stale.length > 0 && installed.adopted !== undefined) return false;
    if (stale.length > 0) {
      // Pruning must never destroy the install. The resulting manifest has to
      // stay loadable — a config and a weight — and the case that matters is
      // upstream dropping its config.json while the PRIMARY repo ships none
      // (the GGUF repos often don't): deleting it would leave a directory the
      // loader cannot open, and the marker would certify exactly that. Fail
      // without mutating anything; the next fix has to come from upstream.
      const remaining = installed.files.filter((file) => !stale.includes(file));
      const lostConfig = !remaining.includes('config.json');
      const lostWeight = !remaining.some((file) => isWeightFile(file));
      if (lostConfig || lostWeight) {
        throw new Error(
          `Refusing to apply "${plan.repo.name}" sidecar removals: they would leave no loadable checkpoint ` +
            `(missing ${lostConfig ? 'config.json' : 'model weights'}). The installed directory is unchanged.`,
        );
      }
    }
    // Delete BEFORE publishing the updated marker. The `stale` set is derived
    // from the marker, so an interrupted refresh must leave the marker still
    // listing what is still on disk: dropping the entry first and dying before
    // the delete would leave the file in place with nothing left to derive it
    // from — reported current forever, and still consumed. This way a failure
    // (a throwing `rm` included) happens with the marker untouched, so the
    // next run derives the same set and retries; and if the marker write
    // itself fails, the marker lists a file that is gone, which reads as
    // not-installed and sends the next job down the full re-stage path.
    for (const file of stale) await rm(join(dir, file), { force: true });
    const next: DownloadCompletion = {
      ...installed,
      ...(stale.length === 0 ? {} : { files: installed.files.filter((file) => !stale.includes(file)) }),
      assetsRepo: plan.repo.name,
      assetsRevision: plan.revision,
    };
    await this.writeMarkerAtomically(dir, next);
    return true;
  }

  /**
   * Publish `marker` over the live completion marker ATOMICALLY (unique temp
   * file + rename in the same directory), mirroring the CLI's writer. A plain
   * in-place write truncates the only ownership record first: a crash, ENOSPC,
   * or short I/O failure leaves invalid JSON, and `isDownloaderOwned()` then
   * classifies an otherwise intact model directory as FOREIGN — every later
   * repair or update job refuses to replace it.
   */
  private async writeMarkerAtomically(dir: string, marker: DownloadCompletion): Promise<void> {
    const path = join(dir, DOWNLOAD_COMPLETE_MARKER);
    const temp = `${path}.${process.pid}.${randomUUID().slice(0, 12)}.tmp`;
    await writeFile(temp, `${JSON.stringify(marker, null, 2)}\n`, { flag: 'wx' });
    try {
      await rename(temp, path);
    } catch (error) {
      // Never leave the temp behind: an orphan in the model dir would be
      // pruned by a later sync and is noise in every listing.
      await rm(temp, { force: true });
      throw error;
    }
  }

  /**
   * Fetch a {@link planAssetSidecars} plan into staging. Returns the manifest
   * entries actually downloaded, which the caller appends to the job's file
   * list so the prune, the whole-set re-verify, and the completion marker all
   * cover them. Downloads go through {@link downloadVerifiedFile}, so path
   * safety, bounded retries, and content verification against the assetsRepo
   * manifest are identical to a primary file, and a cancel unwinds the same
   * way.
   *
   * Provenance: the completion marker stays uniform — it pins only the
   * PRIMARY repo+revision, with sidecar paths folded into its file list.
   * Sidecar bytes are content-verified here against the assetsRepo manifest,
   * but their upstream identity is not recorded, so a sidecar-only upstream
   * change (a tokenizer tweak in the base repo) does not raise an update
   * badge for an installed model.
   *
   * A fetch failure fails the job rather than publishing without the
   * sidecars: a GGUF without the official tokenizer files is exactly the
   * broken-tool-calling state the fetch exists to prevent (see
   * {@link CatalogBuild.assetsRepo}).
   */
  private async downloadAssetSidecars(
    plan: { repo: { type: 'model'; name: string }; revision: string; files: ListFileEntry[] },
    stagingDir: string,
    stagingReal: string,
    files: ListFileEntry[],
    job: JobState,
  ): Promise<ListFileEntry[]> {
    const { repo, revision, files: planned } = plan;
    const downloaded: ListFileEntry[] = [];

    for (let index = 0; index < planned.length; index++) {
      // Stop promptly between files when cancelled; the `finally` removes the
      // staging dir. Same boundary discipline as the primary loop.
      if (job.cancelled) throw new Error('Download cancelled');
      const file = planned[index];
      // The candidate names are fixed constants, but the staged path is still
      // derived from third-party manifest data — re-assert containment against
      // the canonical staging dir, identical to the primary write boundary.
      if (!isSafeRelPath(file.path)) {
        throw new Error(`Refusing to download "${file.path}" from "${repo.name}": path is not a safe relative path`);
      }
      const destResolved = resolve(stagingReal, file.path);
      if (destResolved !== stagingReal && !destResolved.startsWith(stagingReal + sep)) {
        throw new Error(`Refusing to write "${file.path}" outside the staging directory`);
      }
      const destPath = join(stagingDir, file.path);
      const jobBaseBytes = job.receivedBytes;
      const fileIndex = files.length + index;
      const fileCount = files.length + planned.length;

      if (await isStagedCopyComplete(destPath, file)) {
        const fileBytes = file.size > 0 ? file.size : 0;
        job.receivedBytes = jobBaseBytes + fileBytes;
        this.emitFileProgress(job, file.path, fileBytes, file.size, fileIndex, fileCount);
        downloaded.push(file);
        continue;
      }

      const context = await this.downloadVerifiedFile(repo, revision, file, destPath, {
        jobBaseBytes,
        fileIndex,
        fileCount,
      });
      const fileBytes = file.size > 0 ? file.size : context.received;
      job.receivedBytes = jobBaseBytes + fileBytes;
      this.emitFileProgress(job, file.path, fileBytes, file.size, fileIndex, fileCount);
      downloaded.push(file);
    }
    return downloaded;
  }

  /** Remove staged files absent from the manifest (our marker is exempt). */
  private async pruneToManifest(stagingDir: string, files: ListFileEntry[]): Promise<void> {
    const manifestPaths = new Set(files.map((file) => file.path));
    for (const rel of await listStagedFiles(stagingDir)) {
      if (rel === DOWNLOAD_COMPLETE_MARKER || manifestPaths.has(rel)) continue;
      await rm(join(stagingDir, rel), { force: true });
    }
  }

  /**
   * Remove the old entry that an owned swap or a crashed one left at `backupDir`. A link, or a folder whose marker
   * has no `adopted` record, is removed whole. A folder that Update adopted loses only the files its record proves
   * unchanged ({@link removeProvenFiles}); when anything else is left, the rest, marker included, is moved to
   * `.staging/<slug>.kept-<uuid>`, which nothing removes. A folder whose marker cannot be read is moved there whole,
   * since nothing proves which of its files the app may delete. Returns where the kept files are (`backupDir` itself
   * when that move fails), or `undefined` when nothing was kept.
   */
  private async removeOwnedBackup(backupDir: string, slug: string): Promise<string | undefined> {
    let isLink: boolean;
    try {
      isLink = (await lstat(backupDir)).isSymbolicLink();
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
      throw error;
    }
    const marker = isLink ? undefined : readCompletion(backupDir);
    if (isLink || (marker !== undefined && marker.adopted === undefined)) {
      await rm(backupDir, { recursive: true, force: true });
      return undefined;
    }
    if (marker?.adopted !== undefined) {
      const aside = join(backupDir, `.${randomUUID()}`);
      await removeProvenFiles(backupDir, '', new Set(marker.files), marker.adopted, aside);
      await removeEmptyDirs(aside);
      const left = await readdir(backupDir).catch(() => null);
      if (left !== null && left.every((name) => name === DOWNLOAD_COMPLETE_MARKER)) {
        try {
          await unlink(join(backupDir, DOWNLOAD_COMPLETE_MARKER));
          await rmdir(backupDir);
          return undefined;
        } catch {
          // Something came in, or the marker could not go: keep what is there.
        }
      }
    }
    const keptDir = join(this.modelsDir, '.staging', `${slug}${KEPT_MARK}${randomUUID()}`);
    try {
      await rename(backupDir, keptDir);
      return keptDir;
    } catch {
      return backupDir;
    }
  }

  /**
   * Publish staging onto the final path without ever destroying a checkpoint the
   * downloader does not own. The completion marker is written as the LAST staged
   * step (so it travels with the rename and only coexists with a full, verified
   * set). Then:
   *   - final dir absent → rename staging → final (+ fsync parent);
   *   - a cleanup → the entry must still be the one the user confirmed, owned or not ({@link swapCleanup});
   *   - final dir present but UNOWNED (no marker) → refuse, leaving that dir and
   *     its files untouched;
   *   - otherwise → rollback-safe swap: move the existing entry to a temp backup,
   *     rename staging → final, fsync the parent, and only then remove the backup
   *     with {@link removeOwnedBackup}, which keeps the files of an adopted folder
   *     that its record does not prove and returns where it moved them;
   *     if the swap rename fails, restore the backup so a crash never leaves the
   *     model missing.
   * Answers where old files were kept, and how a cleanup removed the old entry.
   */
  private async publish(
    stagingDir: string,
    finalDir: string,
    job: JobState,
    revision: string,
    files: readonly { path: string }[],
    sidecarPlan: { repo: { name: string }; revision: string } | null,
    located?: LocatedRecord,
  ): Promise<{ keptDir?: string; removed?: 'link' | 'trash' }> {
    const repo = job.repo;
    const slug = basename(finalDir);
    await mkdir(dirname(finalDir), { recursive: true });

    if (job.replace !== null) {
      if (isPathOccupied(finalDir)) assertStillReplaceable(job.replace, finalDir, slug);
    } else if (isPathOccupied(finalDir) && !isDownloaderOwned(finalDir)) {
      throw new Error(
        `Refusing to overwrite "${finalDir}": it was not created by the dashboard downloader. ` +
          `Remove it manually to reinstall.`,
      );
    }

    const marker: DownloadCompletion = {
      repo,
      revision,
      files: files.map((file) => file.path),
      scope: 'full',
      // Record the sidecar source when this install had one: update discovery
      // compares BOTH revisions, so a base-repo tokenizer fix raises the
      // badge that lets the repair job run at all.
      ...(sidecarPlan !== null ? { assetsRepo: sidecarPlan.repo.name, assetsRevision: sidecarPlan.revision } : {}),
      completedAt: new Date().toISOString(),
      ...(located !== undefined ? { located } : {}),
    };
    await writeFile(join(stagingDir, DOWNLOAD_COMPLETE_MARKER), `${JSON.stringify(marker, null, 2)}\n`, { flag: 'wx' });

    if (!isPathOccupied(finalDir)) {
      await rename(stagingDir, finalDir);
      await fsyncDir(dirname(finalDir));
      return {};
    }

    // Checked again next to the swap: the first guard is a no-op when finalDir was absent then. A cleanup never takes
    // the owned swap, whether or not the entry now carries the app's record.
    if (job.replace !== null) {
      assertStillReplaceable(job.replace, finalDir, slug);
      return await this.swapCleanup(stagingDir, finalDir, job, job.replace);
    }
    if (!isDownloaderOwned(finalDir)) {
      throw new Error(`Refusing to overwrite "${finalDir}": it was not created by the dashboard downloader`);
    }

    // Rollback-safe swap of an owned final dir. The backup lives under `.staging` (a dotdir skipped by model
    // discovery) so a crash mid-swap never surfaces it as a phantom model.
    const backupDir = join(this.modelsDir, '.staging', `${slug}.backup-${process.pid}.${randomUUID()}`);
    job.swapStarted = true;
    try {
      await rename(finalDir, backupDir);
    } catch (error) {
      job.swapStarted = false;
      throw error;
    }
    try {
      await rename(stagingDir, finalDir);
    } catch (error) {
      try {
        await rename(backupDir, finalDir);
      } catch {
        throw new Error(`The install failed and "${slug}" could not be put back. It is now at "${backupDir}".`);
      }
      job.swapStarted = false;
      throw error;
    }
    await fsyncDir(dirname(finalDir));
    try {
      const keptDir = await this.removeOwnedBackup(backupDir, slug);
      return keptDir === undefined ? {} : { keptDir };
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      throw new Error(
        `${reason}. The model is installed. A part of the old "${slug}" can remain in "${join(this.modelsDir, '.staging')}".`,
      );
    }
  }

  /**
   * The swap of a cleanup. Nothing outside `.staging` changes until the entry at `finalDir` is renamed, whole, into a
   * new holder folder as `<holder>/<slug>`; a writer outside the app cannot reach it there by its old path, so the
   * identity check of the moved entry holds from then on. A failed check or a failed rename of staging puts it back.
   * Once the new folder is at `finalDir` the model is installed, and {@link removeReplaced} only reports what became
   * of the old entry.
   */
  private async swapCleanup(
    stagingDir: string,
    finalDir: string,
    job: JobState,
    kept: CleanupTarget,
  ): Promise<{ keptDir?: string; removed?: 'link' | 'trash' }> {
    const slug = basename(finalDir);
    const stagingRoot = join(this.modelsDir, '.staging');
    // Recovery finishes the cleanup only when the folder at the path is this new folder: a rename keeps its inode.
    const { ino } = await lstat(stagingDir, { bigint: true });
    const holder = join(stagingRoot, `${slug}${CLEANUP_MARK}${process.pid}.${ino}.${randomUUID()}`);
    const moved = join(holder, slug);
    await mkdir(holder);
    job.swapStarted = true;
    try {
      await rename(finalDir, moved);
    } catch (error) {
      job.swapStarted = false;
      await rmdir(holder).catch(() => {});
      throw error;
    }
    // A way back must not replace whatever another program put at `finalDir` since, so the old entry then stays aside.
    const keepAside = async (): Promise<never> => {
      const keptDir = join(stagingRoot, `${slug}${KEPT_MARK}${randomUUID()}`);
      let at = moved;
      try {
        await rename(holder, keptDir);
        at = join(keptDir, slug);
      } catch {}
      throw new Error(
        `"${slug}" changed on disk during the download, and something else is now at its path. ` +
          `The old entry is at "${at}". ${NOTHING_REMOVED}`,
      );
    };
    if (!sameIdentity(entryIdentity(moved), kept.identity)) {
      if (entryIdentity(finalDir) !== null) await keepAside();
      try {
        await putBack(moved, finalDir);
      } catch {
        await keepAside();
      }
      await rmdir(holder).catch(() => {});
      job.swapStarted = false;
      throw new Error(`"${slug}" changed on disk during the download.`);
    }
    try {
      await rename(stagingDir, finalDir);
    } catch (error) {
      if (entryIdentity(finalDir) !== null) await keepAside();
      try {
        await putBack(moved, finalDir);
      } catch (putBackError) {
        if ((putBackError as NodeJS.ErrnoException).code === 'EEXIST') await keepAside();
        throw new Error(`The install failed and "${slug}" could not be put back. It is now at "${moved}".`);
      }
      await rmdir(holder).catch(() => {});
      job.swapStarted = false;
      throw error;
    }
    await fsyncDir(dirname(finalDir));
    return await this.removeReplaced(holder, slug, kept.summary.kind === 'symlink');
  }

  /**
   * Settle one file at its final size. Takes the JOB (not just its id) so the
   * frame carries the job aggregate the caller just advanced — the running
   * total is the only part a mid-download subscriber cannot reconstruct.
   */
  private emitFileProgress(
    job: JobState,
    file: string,
    receivedBytes: number,
    totalBytes: number,
    fileIndex: number,
    fileCount: number,
    phase?: 'check',
  ): void {
    this.emit({
      type: 'progress',
      id: job.id,
      file,
      receivedBytes,
      jobReceivedBytes: job.receivedBytes,
      totalBytes,
      fileIndex,
      fileCount,
      ...(phase !== undefined ? { phase } : {}),
    });
  }
}
