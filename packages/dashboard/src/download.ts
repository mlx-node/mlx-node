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
 * refused unless an explicit `overwrite` is requested — the ownership guard is
 * re-checked adjacent to the destructive swap so a dir that races in after the
 * first check is still never overwritten. An OWNED upgrade swaps via a temp
 * backup so a crash mid-rename rolls back to the original; an orphaned backup a
 * crash leaves behind is recovered (or reaped) by a best-effort sweep at the
 * start of the next download.
 *
 * Pure disk + network — never the native addon.
 */

import { createHash, randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { copyFile, mkdir, open, readdir, realpath, rename, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { basename, dirname, join, sep } from 'node:path';

import { downloadFileToCacheDir, listFiles, type ListFileEntry, modelInfo } from '@huggingface/hub';
import { MODEL_CATALOG } from '@mlx-node/agent/catalog';

import { acquireLock, type LockHandle, pidAlive } from './lock.js';
import { type DownloadCompletion, DOWNLOAD_COMPLETE_MARKER, isDownloaderOwned, isModelInstalled } from './models.js';

/** Bounded re-fetch attempts for a staged file that fails post-copy verification. */
const MAX_VERIFY_ATTEMPTS = 3;

/** Job-private staging dir name `<slug>@<40hex-sha>.<pid>.<uuid>` — captures the owner pid. */
const STAGING_DIR_RE = /@[0-9a-f]{40}\.(\d+)\.[0-9a-f-]+$/i;

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

/** A file that makes a directory a model: a config, or any weight payload. */
function isWeightFile(path: string): boolean {
  return path.endsWith('.safetensors') || path.endsWith('.gguf') || path.endsWith('.pdiparams');
}

/**
 * True when the manifest actually describes a model: at least one file, and
 * among them a `config.json` or a weight file. A repo that resolves to zero
 * model files (404-shaped listing, auth-stripped repo, non-model repo) must
 * error rather than publish a hollow "installed" directory.
 */
function hasModelPayload(files: ListFileEntry[]): boolean {
  return files.some((file) => file.path === 'config.json' || isWeightFile(file.path));
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
 * Resume check for a file already staged on disk. Verifies CONTENT identity, not
 * merely size, so a stale/truncated-but-right-size file is never trusted:
 *
 *   - size must match the manifest (fast gate, catches truncation);
 *   - if the manifest advertises `lfs.oid` (the LFS sha256 of the content), the
 *     file's sha256 must match it;
 *   - else, for a plain file (no LFS, no Xet), the top-level git `oid` is the
 *     git-blob sha1 of the actual content, so the file's git-blob sha1 must match
 *     it. For Xet-backed files that `oid` hashes the pointer, not the content, so
 *     it is deliberately NOT used (a mismatch would force a needless re-fetch);
 *   - otherwise (no usable hash) fall back to size alone — the runner still
 *     cross-checks the full manifest before writing the completion marker, so a
 *     partial set can never be published as complete.
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

export type DownloadEvent =
  | { type: 'start'; id: string; repo: string; totalBytes: number; fileCount: number }
  | {
      type: 'progress';
      id: string;
      file: string;
      receivedBytes: number;
      totalBytes: number;
      fileIndex: number;
      fileCount: number;
    }
  | { type: 'done'; id: string; outputDir: string }
  | { type: 'error'; id: string; message: string };

interface JobState {
  id: string;
  repo: string;
  state: 'running' | 'done' | 'error';
  /** Job-level aggregate bytes across all files. */
  receivedBytes: number;
  /** Sum of manifest file sizes; 0 until the manifest is fetched. */
  totalBytes: number;
  /** Permit replacing a final dir the downloader does not own (default false). */
  overwrite: boolean;
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
  private readonly fetchImpl: typeof fetch;
  private readonly wrappedFetch: typeof fetch;

  private readonly emitter = new EventEmitter();
  private readonly jobsById = new Map<string, JobState>();
  private readonly order: string[] = [];
  private readonly queue: string[] = [];
  private readonly lastEvent = new Map<string, DownloadEvent>();
  private draining = false;

  private currentJob: JobState | null = null;
  private currentFile: FileContext | null = null;

  constructor(opts: { modelsDir: string; cacheDir?: string; fetchImpl?: typeof fetch }) {
    this.modelsDir = opts.modelsDir;
    this.cacheDir = opts.cacheDir ?? DEFAULT_CACHE_DIR;
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch;
    this.wrappedFetch = this.makeCountingFetch();
    // Each job id can have many subscribers (SSE clients); lift the default cap.
    this.emitter.setMaxListeners(0);
  }

  /**
   * Queue a catalog repo for download. Rejects any repo not in the catalog.
   * `overwrite` (default false) permits replacing a final dir the downloader does
   * not own; callers (the API/UI) leave it unset so an unowned local checkpoint
   * is never silently destroyed.
   */
  start(repo: string, opts?: { overwrite?: boolean }): string {
    if (!MODEL_CATALOG.some((entry) => entry.hfRepo === repo)) {
      throw new Error(`Repo "${repo}" is not in the model catalog`);
    }
    const id = randomUUID();
    this.jobsById.set(id, {
      id,
      repo,
      state: 'running',
      receivedBytes: 0,
      totalBytes: 0,
      overwrite: opts?.overwrite ?? false,
    });
    this.order.push(id);
    this.queue.push(id);
    void this.drain();
    return id;
  }

  jobs(): Array<{
    id: string;
    repo: string;
    state: 'running' | 'done' | 'error';
    receivedBytes: number;
    totalBytes: number;
  }> {
    return this.order.map((id) => {
      const job = this.jobsById.get(id)!;
      return {
        id: job.id,
        repo: job.repo,
        state: job.state,
        receivedBytes: job.receivedBytes,
        totalBytes: job.totalBytes,
      };
    });
  }

  /** Subscribe to a job's events; the most recent event is replayed on attach. */
  subscribe(id: string, fn: (event: DownloadEvent) => void): () => void {
    const last = this.lastEvent.get(id);
    if (last !== undefined) fn(last);
    this.emitter.on(id, fn);
    return () => {
      this.emitter.off(id, fn);
    };
  }

  private emit(event: DownloadEvent): void {
    this.lastEvent.set(event.id, event);
    this.emitter.emit(event.id, event);
  }

  private makeCountingFetch(): typeof fetch {
    return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const response = await this.fetchImpl(input, init);
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

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.queue.length > 0) {
        const id = this.queue.shift()!;
        const job = this.jobsById.get(id);
        if (job !== undefined) await this.processJob(job);
      }
    } finally {
      this.draining = false;
    }
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
  private async resolveRevision(repoName: string): Promise<string> {
    const info = await modelInfo({ name: repoName, additionalFields: ['sha'], fetch: this.wrappedFetch });
    const sha: unknown = info.sha;
    if (!(typeof sha === 'string' && /^[0-9a-f]{40}$/i.test(sha))) {
      throw new Error(
        `Cannot resolve an immutable commit for "${repoName}"; refusing to pin to a mutable ref`,
      );
    }
    return sha;
  }

  private async processJob(job: JobState): Promise<void> {
    const entry = MODEL_CATALOG.find((candidate) => candidate.hfRepo === job.repo)!;
    const slug = entry.hfRepo.split('/').pop()!.toLowerCase();
    const finalDir = join(this.modelsDir, slug);
    const repo = { type: 'model' as const, name: job.repo };
    this.currentJob = job;

    const stagingRoot = join(this.modelsDir, '.staging');
    const lockPath = join(stagingRoot, `${slug}.lock`);

    let stagingDir: string | undefined;
    let lock: LockHandle | undefined;
    try {
      // The lockfile lives on the same fs as staging; ensure the dir exists first.
      await mkdir(stagingRoot, { recursive: true });
      // Serialize same-model downloads across processes BEFORE any recovery sweep,
      // so a live peer's active backup/staging is never reclaimed out from under it.
      lock = await acquireLock(lockPath);

      // Reap a crashed job's private staging tree (pid-proven, process-wide) and
      // recover/reap an orphaned publish backup for THIS model (safe under our lock).
      await this.sweepStaging(slug);

      const revision = await this.resolveRevision(job.repo);
      // Job-private staging: keyed by the IMMUTABLE revision (a different
      // revision never reuses another's staged bytes) and unique per invocation
      // (`.<pid>.<uuid>`) so no two processes ever write into the same tree.
      // Resume is HF-cache-backed, not staging-backed, so this dir is disposable
      // and removed on both success and failure.
      stagingDir = join(this.modelsDir, '.staging', `${slug}@${revision}.${process.pid}.${randomUUID()}`);

      const files: ListFileEntry[] = [];
      let totalBytes = 0;
      for await (const file of listFiles({ repo, recursive: true, revision, fetch: this.wrappedFetch })) {
        if (file.type !== 'directory' && isWantedFile(file.path)) {
          files.push(file);
          if (file.size > 0) totalBytes += file.size;
        }
      }

      if (!hasModelPayload(files)) {
        throw new Error(`Repo "${job.repo}" has no downloadable model files (no config.json or weights)`);
      }

      job.totalBytes = totalBytes;
      this.emit({ type: 'start', id: job.id, repo: job.repo, totalBytes, fileCount: files.length });

      // Already published and complete (marker present): the resume/skip case —
      // re-downloading nothing.
      if (isModelInstalled(finalDir)) {
        job.receivedBytes = totalBytes;
        job.state = 'done';
        this.emit({ type: 'done', id: job.id, outputDir: finalDir });
        return;
      }

      await mkdir(stagingDir, { recursive: true });

      for (let index = 0; index < files.length; index++) {
        const file = files[index];
        const destPath = join(stagingDir, file.path);
        const jobBaseBytes = job.receivedBytes;

        if (await isStagedCopyComplete(destPath, file)) {
          const fileBytes = file.size > 0 ? file.size : 0;
          job.receivedBytes = jobBaseBytes + fileBytes;
          this.emitFileProgress(job.id, file.path, fileBytes, file.size, index, files.length);
          continue;
        }

        const context = await this.downloadVerifiedFile(repo, revision, file, destPath, {
          jobBaseBytes,
          fileIndex: index,
          fileCount: files.length,
        });

        // Settle the file at its manifest size (or the counted bytes when the
        // manifest omits a size), so aggregate progress is exact even if a
        // cache hit or a metadata-only response streamed nothing.
        const fileBytes = file.size > 0 ? file.size : context.received;
        job.receivedBytes = jobBaseBytes + fileBytes;
        this.emitFileProgress(job.id, file.path, fileBytes, file.size, index, files.length);
      }

      // Quarantine any staged entry not in the current manifest (a stale file
      // from an earlier interrupted run), so the published set is exactly the
      // manifest — no orphan (e.g. an old single-file weight) can win over it.
      await this.pruneToManifest(stagingDir, files);

      // Re-validate the WHOLE staged set against the manifest immediately before
      // publishing; any mismatch fails the job rather than shipping bad bytes.
      for (const file of files) {
        if (!(await isStagedCopyComplete(join(stagingDir, file.path), file))) {
          throw new Error(`Staged file "${file.path}" failed content verification before publish`);
        }
      }

      await this.publish(stagingDir, finalDir, job.repo, revision, files, job.overwrite);

      job.state = 'done';
      this.emit({ type: 'done', id: job.id, outputDir: finalDir });
    } catch (error) {
      job.state = 'error';
      this.emit({ type: 'error', id: job.id, message: (error as Error).message });
    } finally {
      // Job-private staging is scratch: a successful publish already renamed it
      // away (this is a no-op), and on failure it must not leak — resume never
      // reuses it (the HF cache does). Never touches the final dir.
      if (stagingDir !== undefined) {
        await rm(stagingDir, { recursive: true, force: true }).catch(() => {});
      }
      if (lock !== undefined) {
        await lock.release().catch(() => {});
      }
      this.currentJob = null;
      this.currentFile = null;
    }
  }

  /**
   * Startup recovery sweep of `.staging`, run under this job's per-model lock:
   *
   *   - Process-wide, pid-proven: a crashed/SIGKILLed job's private staging tree
   *     (`<slug>@<sha>.<pid>.<uuid>`) is reaped when its owner pid is gone (it is
   *     cleaned only in the live process's `finally`, so a kill leaks it forever
   *     otherwise). A live — or reused — pid is kept; pid-reuse only preserves a
   *     leak, it can never reap a live job.
   *
   *   - Slug-scoped (this model only): a publish backup this job's lock proves is
   *     orphaned (`<slug>.backup-<uuid>`) is renamed back into place when the swap
   *     died mid-rename (final dir absent), else the leaked copy is removed. Other
   *     models' backups are never touched — only their own lock-holder may, so a
   *     live peer's active backup is never reclaimed. Never throws.
   */
  private async sweepStaging(slug: string): Promise<void> {
    const stagingRoot = join(this.modelsDir, '.staging');
    let entries: string[];
    try {
      entries = await readdir(stagingRoot);
    } catch {
      return;
    }
    const backupPrefix = `${slug}.backup-`;
    const finalDir = join(this.modelsDir, slug);
    for (const name of entries) {
      const staleMatch = STAGING_DIR_RE.exec(name);
      if (staleMatch !== null && !pidAlive(Number(staleMatch[1]))) {
        await rm(join(stagingRoot, name), { recursive: true, force: true }).catch(() => {});
        continue;
      }
      if (name.startsWith(backupPrefix)) {
        const backupPath = join(stagingRoot, name);
        try {
          if (existsSync(finalDir)) {
            await rm(backupPath, { recursive: true, force: true });
          } else {
            await rename(backupPath, finalDir);
          }
        } catch {
          // A single un-recoverable backup must not block the job.
        }
      }
    }
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
        await copyFile(snapshotPath, destPath);
      } finally {
        this.currentFile = null;
      }

      if (await isStagedCopyComplete(destPath, file)) return context;
      // Content did not match the manifest: drop the staged copy AND invalidate
      // the HF cache entry (the snapshot pointer + the blob it resolves to) so the
      // next attempt actually re-fetches. For a pinned commit `downloadFileToCacheDir`
      // returns the cached pointer without revalidating, so without this the retry
      // would recopy the same corrupt bytes.
      await rm(destPath, { force: true });
      // Resolve the pointer's target BEFORE unlinking it (rm on a symlink never
      // follows it, so the pointer removal is always safe). `etag`/`oid` is
      // server-controlled and can contain `../`, so the blob may resolve OUTSIDE
      // the managed cache (a poisoned repo, or a foreign symlink already in the
      // shared HF cache); delete it ONLY when it is contained under the realpath'd
      // cache root, never an unrelated same-user file.
      const blob = await realpath(snapshotPath).catch(() => undefined);
      await rm(snapshotPath, { force: true });
      if (blob !== undefined) {
        const cacheRoot = await realpath(this.cacheDir).catch(() => undefined);
        if (cacheRoot !== undefined && (blob === cacheRoot || blob.startsWith(cacheRoot + sep))) {
          await rm(blob, { force: true });
        }
      }
    }
    throw new Error(`Downloaded file "${file.path}" failed content verification after ${MAX_VERIFY_ATTEMPTS} attempts`);
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
   * Publish staging onto the final path without ever destroying a checkpoint the
   * downloader does not own. The completion marker is written as the LAST staged
   * step (so it travels with the rename and only coexists with a full, verified
   * set). Then:
   *   - final dir absent → rename staging → final (+ fsync parent);
   *   - final dir present but UNOWNED (no marker) and no `overwrite` → refuse,
   *     leaving that dir and its files untouched;
   *   - otherwise (OWNED, or explicit `overwrite`) → rollback-safe swap: move the
   *     existing dir to a temp backup, rename staging → final, fsync the parent,
   *     and only then remove the backup; if the swap rename fails, restore the
   *     backup so a crash never leaves the model missing.
   */
  private async publish(
    stagingDir: string,
    finalDir: string,
    repo: string,
    revision: string,
    files: ListFileEntry[],
    overwrite: boolean,
  ): Promise<void> {
    await mkdir(dirname(finalDir), { recursive: true });

    if (existsSync(finalDir) && !isDownloaderOwned(finalDir) && !overwrite) {
      throw new Error(
        `Refusing to overwrite "${finalDir}": it was not created by the dashboard downloader. ` +
          `Remove it manually (or re-run with overwrite) to reinstall.`,
      );
    }

    const marker: DownloadCompletion = {
      repo,
      revision,
      files: files.map((file) => file.path),
      completedAt: new Date().toISOString(),
    };
    await writeFile(join(stagingDir, DOWNLOAD_COMPLETE_MARKER), `${JSON.stringify(marker, null, 2)}\n`);

    if (!existsSync(finalDir)) {
      await rename(stagingDir, finalDir);
      await fsyncDir(dirname(finalDir));
      return;
    }

    // Re-check ownership adjacent to the destructive swap: the first guard is a
    // no-op when finalDir was absent then, so an UNOWNED dir that raced in
    // between must still never be renamed away and deleted. This collapses the
    // window to the inherent FS-level race an interprocess lock would close.
    if (!isDownloaderOwned(finalDir) && !overwrite) {
      throw new Error(
        `Refusing to overwrite "${finalDir}": it was not created by the dashboard downloader`,
      );
    }

    // Rollback-safe swap of an owned (or explicitly overwritten) final dir. The
    // backup lives under `.staging` (a dotdir skipped by model discovery) so a
    // crash mid-swap never surfaces it as a phantom model.
    const backupDir = join(this.modelsDir, '.staging', `${basename(finalDir)}.backup-${randomUUID()}`);
    await rename(finalDir, backupDir);
    try {
      await rename(stagingDir, finalDir);
    } catch (error) {
      await rename(backupDir, finalDir).catch(() => {});
      throw error;
    }
    await fsyncDir(dirname(finalDir));
    await rm(backupDir, { recursive: true, force: true });
  }

  private emitFileProgress(
    id: string,
    file: string,
    receivedBytes: number,
    totalBytes: number,
    fileIndex: number,
    fileCount: number,
  ): void {
    this.emit({ type: 'progress', id, file, receivedBytes, totalBytes, fileIndex, fileCount });
  }
}
