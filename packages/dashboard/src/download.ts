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
 * refused unless an explicit `overwrite` is requested — up front (a fast-fail
 * ownership preflight before any download, so no bytes move for a doomed job) and
 * again adjacent to the destructive swap so a dir that races in after the first
 * check is still never overwritten. An OWNED upgrade swaps via a temp
 * backup so a crash mid-rename rolls back to the original; an orphaned backup a
 * crash leaves behind is recovered (or reaped) by a best-effort sweep at the
 * start of the next download.
 *
 * Pure disk + network — never the native addon.
 */

import { createHash, randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { createReadStream, existsSync, lstatSync, realpathSync, statSync } from 'node:fs';
import { copyFile, mkdir, open, readdir, realpath, rename, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve, sep } from 'node:path';

import { downloadFileToCacheDir, listFiles, type ListFileEntry, modelInfo } from '@huggingface/hub';
import { MODEL_CATALOG } from '@mlx-node/agent/catalog';

import {
  type DownloadCompletion,
  DOWNLOAD_COMPLETE_MARKER,
  isDownloaderOwned,
  isModelInstalled,
  isPathOccupied,
  isSafeRelPath,
  isWeightFile,
  readCompletion,
} from './models.js';

/** Bounded re-fetch attempts for a staged file that fails post-copy verification. */
const MAX_VERIFY_ATTEMPTS = 3;

/** Job-private staging dir name `<slug>@<40hex-sha>.<pid>.<uuid>` — captures the owner pid. */
const STAGING_DIR_RE = /@[0-9a-f]{40}\.(\d+)\.[0-9a-f-]+$/i;

/** Publish backup dir name `<slug>.backup-<pid>.<uuid>` — captures the owner pid (disjoint from STAGING_DIR_RE). */
const BACKUP_DIR_RE = /\.backup-(\d+)\.[0-9a-f-]+$/i;

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

/**
 * True when the manifest actually describes a loadable model: it must carry a
 * `config.json` AND at least one weight file. A repo that resolves to a
 * one-sided set (config-only or weights-only — e.g. a catalog repo caught
 * mid-re-upload with the config committed before the weights) must error rather
 * than publish a hollow, one-sided "installed" directory that `loadModel` would
 * then fail to open while the catalog reports it Installed with retry disabled.
 */
function hasModelPayload(files: ListFileEntry[]): boolean {
  return files.some((file) => file.path === 'config.json') && files.some((file) => isWeightFile(file.path));
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
    }
  | { type: 'done'; id: string; outputDir: string }
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

/**
 * `committing` is the brief, non-cancellable window from just before `publish`
 * until the job settles: `cancel()`'s `state !== 'running'` gate rejects it, so a
 * cancel that arrives once the atomic swap has begun can no longer report success
 * while the model still installs.
 */
type JobStatus = 'running' | 'committing' | 'done' | 'error' | 'cancelled';

/** A settled job: it holds no in-flight work and no longer owns its repo. */
function isTerminal(state: JobStatus): boolean {
  return state === 'done' || state === 'error' || state === 'cancelled';
}

interface JobState {
  id: string;
  repo: string;
  state: JobStatus;
  /** Job-level aggregate bytes across all files. */
  receivedBytes: number;
  /** Sum of manifest file sizes; 0 until the manifest is fetched. */
  totalBytes: number;
  /** Permit replacing a final dir the downloader does not own (default false). */
  overwrite: boolean;
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

  constructor(opts: { modelsDir: string; cacheDir?: string; fetchImpl?: typeof fetch }) {
    this.modelsDir = opts.modelsDir;
    this.cacheDir = opts.cacheDir ?? DEFAULT_CACHE_DIR;
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch;
    this.wrappedFetch = this.makeCountingFetch();
    // Each job id can have many subscribers (SSE clients); lift the default cap.
    this.emitter.setMaxListeners(0);
  }

  /**
   * Queue a catalog repo for download. Rejects any repo not in the catalog, and
   * refuses outright once {@link shutdown} has begun — the HTTP listener stays
   * attached across the drain, so a late POST would otherwise be adopted by the
   * loop `shutdown` is awaiting and could keep `close()` from ever completing.
   * `overwrite` (default false) permits replacing a final dir the downloader does
   * not own; callers (the API/UI) leave it unset so an unowned local checkpoint
   * is never silently destroyed.
   *
   * At most ONE nonterminal job exists per repo: a request that arrives while a
   * job for that repo is queued or in flight hands back THAT job's id instead of
   * allocating another. The Models page tracks a single job id per repo, so a
   * duplicate request (a double-click on Install, or a second dashboard tab whose
   * card still reads Install) would otherwise leave an untracked twin that Cancel
   * can no longer reach yet that still downloads and installs. Reusing keeps the
   * request idempotent — no 409 for the SPA to render as a red failure — and
   * matches the stance {@link processJob} already takes below, where re-asking for
   * an already-published repo+revision short-circuits to `done` rather than
   * erroring. The existing job's `overwrite` stands; it is not re-negotiated.
   */
  start(repo: string, opts?: { overwrite?: boolean }): string {
    if (this.closed) throw new DownloadsClosedError();
    if (!MODEL_CATALOG.some((entry) => entry.hfRepo === repo)) {
      throw new Error(`Repo "${repo}" is not in the model catalog`);
    }
    const active = this.activeJobFor(repo);
    if (active !== undefined) return active.id;
    const id = randomUUID();
    this.jobsById.set(id, {
      id,
      repo,
      state: 'running',
      receivedBytes: 0,
      totalBytes: 0,
      overwrite: opts?.overwrite ?? false,
      cancelled: false,
      controller: new AbortController(),
    });
    this.order.push(id);
    this.queue.push(id);
    void this.drain();
    return id;
  }

  /**
   * The one nonterminal (queued / in-flight / committing) job for `repo`, if any.
   * Derived from `jobsById` on demand rather than kept in a repo→id index, so it
   * can never go stale: every terminal transition and every dismissal already
   * updates `jobsById`, and a second structure would have to be cleared at each of
   * those sites. The registry holds at most a handful of catalog repos.
   */
  private activeJobFor(repo: string): JobState | undefined {
    for (const job of this.jobsById.values()) {
      // `!job.cancelled` as well as the state: a cancel of the IN-FLIGHT job
      // returns while the job is still `running`, because `cancel()` leaves the
      // terminal transition to `processJob` as it unwinds (that is what keeps the
      // `cancelled` event from being emitted twice). Handing that job back would
      // answer a fresh Install with an id that is already stopping — the caller
      // subscribes to a stream whose next frame is `cancelled`, and no download
      // happens. A job on its way out is not an active job.
      if (job.repo === repo && !isTerminal(job.state) && !job.cancelled) return job;
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
    repo: string;
    state: JobStatus;
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
      throw new Error(`Cannot resolve an immutable commit for "${repoName}"; refusing to pin to a mutable ref`);
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

    let stagingDir: string | undefined;
    try {
      // The no-follow ownership preflight, factored so it reads identically at both
      // its call sites: a final dir the downloader does not own (a manual copy or an
      // `mlx download` install with no completion marker, OR any symlink, which
      // `isDownloaderOwned` rejects no-follow) is refused — so an unowned local
      // checkpoint never triggers a full multi-GB download+hash only to be refused at
      // publish. It reads ONLY `finalDir`, so it is order-independent of the staging
      // root and the resolved revision. Run it (a) before backup recovery — the
      // recovery reap classifies finalDir with `isModelInstalled`, which FOLLOWS
      // symlinks, so a foreign `finalDir` link must be refused BEFORE recovery could
      // follow it and reap a recoverable rollback backup — and (b) again after
      // recovery, so a `finalDir` that recovery (or a race) restored to a foreign
      // symlink is refused BEFORE skip-detection follows it. The publish-time checks
      // below still guard the narrow check-then-swap race for a dir that races in
      // after this point.
      const refuseIfUnownedFinal = (): void => {
        if (isPathOccupied(finalDir) && !isDownloaderOwned(finalDir) && !job.overwrite) {
          throw new Error(
            `Refusing to install over "${finalDir}": it was not created by the dashboard downloader. ` +
              `Remove it manually to reinstall.`,
          );
        }
      };

      // Refuse a symlinked (or non-directory) `.staging` root BEFORE creating it:
      // otherwise `mkdir(..., { recursive })` follows the link and every staged
      // write, the sweep, the publish backup, and the `finally` cleanup all land
      // on the external target. Rejecting a symlinked root closes the escape at the
      // source — every downstream path is built under this root. Done first (it only
      // touches the staging root, not finalDir) so the dead-tree reap below can run.
      assertRealDirOrAbsent(stagingRoot);
      await mkdir(stagingRoot, { recursive: true });

      // Reap a crashed job's private staging tree BEFORE the ownership preflight, so
      // even a job the preflight REFUSES (an unowned/foreign finalDir, no overwrite)
      // still frees a crashed multi-GB staging tree instead of leaking it until some
      // other eligible download for the slug runs. This touches only `.staging`
      // trees (never finalDir), so it is safe to run before the finalDir preflight.
      await this.reapDeadStagingTrees();

      // Ownership preflight: refuse an unowned/foreign finalDir up front (no bytes
      // move for a doomed job).
      refuseIfUnownedFinal();

      // Recover/reap an orphaned publish backup for THIS model — AFTER the preflight,
      // so a foreign finalDir link was already refused and can't be followed here.
      // Pid-gated: a live peer's active publish backup is never reclaimed.
      await this.recoverBackup(slug, finalDir);

      // Re-run the preflight after recovery: recovery restores only an owned real-dir
      // backup, but this refuses any foreign symlink a race put at finalDir before
      // skip-detection (`readCompletion`/`isModelInstalled`, which FOLLOW symlinks)
      // could read the foreign marker and report a false `done` through the link.
      refuseIfUnownedFinal();

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
          // Defense-in-depth: `file.path` is a third-party string about to become
          // a `join(stagingDir, file.path)` write/read/verify/publish target (every
          // downstream site derives from this `files[]` array). Refuse — fail
          // closed — any entry that could escape the staging dir before it is
          // admitted. Unreachable for the curated 4-repo catalog (git/HF tree
          // paths cannot express `..`/absolute), but never trust the boundary.
          if (!isSafeRelPath(file.path)) {
            throw new Error(`Refusing to download "${file.path}" from "${job.repo}": path is not a safe relative path`);
          }
          files.push(file);
          if (file.size > 0) totalBytes += file.size;
        }
      }

      if (!hasModelPayload(files)) {
        throw new Error(`Repo "${job.repo}" is not a complete model (needs both a config.json and a weight file)`);
      }

      job.totalBytes = totalBytes;
      this.emit({ type: 'start', id: job.id, repo: job.repo, totalBytes, fileCount: files.length });

      // Already published, complete, AND pinned to THIS repo+revision (marker
      // present): the resume/skip case — re-downloading nothing. Gating on the
      // marker's identity is load-bearing: a complete install of a DIFFERENT
      // revision (or repo) of the same slug must NOT read as done at this revision's
      // byte total. On a mismatch, fall through to download the resolved revision;
      // publish's `isDownloaderOwned`-guarded owned-swap then replaces the stale one
      // (an UNOWNED dir without our marker is still refused there, never destroyed).
      const installed = readCompletion(finalDir);
      if (
        installed !== undefined &&
        installed.repo === job.repo &&
        installed.revision === revision &&
        isModelInstalled(finalDir)
      ) {
        job.receivedBytes = totalBytes;
        job.state = 'done';
        this.emit({ type: 'done', id: job.id, outputDir: finalDir });
        return;
      }

      if (job.cancelled) throw new Error('Download cancelled');

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
          this.emitFileProgress(job, file.path, fileBytes, file.size, index, files.length);
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
        this.emitFileProgress(job, file.path, fileBytes, file.size, index, files.length);
      }

      // A cancel issued after the fetch loop must still unwind here — before the
      // prune and the multi-GB whole-set hash — rather than silently install. The
      // `catch` reclassifies this throw as a clean `cancelled` terminal and the
      // `finally` removes the private staging dir.
      if (job.cancelled) throw new Error('Download cancelled');

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
      job.state = 'committing';

      await this.publish(stagingDir, finalDir, job.repo, revision, files, job.overwrite);

      job.state = 'done';
      this.emit({ type: 'done', id: job.id, outputDir: finalDir });
    } catch (error) {
      // A cancelled job unwinds via a boundary check or an aborted in-flight
      // fetch: classify EITHER as a clean `cancelled` terminal (not a red error),
      // so the abort's raw message never surfaces as a failure.
      if (job.cancelled) {
        job.state = 'cancelled';
        this.emit({ type: 'cancelled', id: job.id });
      } else {
        job.state = 'error';
        this.emit({ type: 'error', id: job.id, message: truncateMessage((error as Error).message) });
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
   * preflight (which already refused a foreign `finalDir` symlink), so nothing here
   * follows such a link. A backup whose owner pid is still ALIVE belongs to a peer
   * mid-publish and is never touched. For a dead-owner backup, classify finalDir
   * NO-FOLLOW first — `isModelInstalled` alone FOLLOWS symlinks, so a live `finalDir`
   * link into an external marked model would read as a complete install and REAP a
   * recoverable rollback backup:
   *
   *   - finalDir is a proven owned-complete REAL dir → the backup is redundant → rm;
   *   - finalDir is ABSENT (a crashed swap) AND the backup is itself an OWNED REAL
   *     directory (no-follow: `lstat().isDirectory()` + our marker) → restore it by
   *     renaming it back into place. A FOREIGN-SYMLINK backup is NEVER restored: a
   *     prior `overwrite` publish could have renamed a symlinked finalDir to its
   *     backup, and restoring that link at finalDir would let skip-detection follow
   *     it and report a false `done` through the foreign target;
   *   - otherwise (foreign symlink / unowned dir / incomplete-owned dir, OR — with
   *     finalDir absent — a non-owned/symlink backup) → RETAIN the backup untouched
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
          await rm(backupPath, { recursive: true, force: true }); // redundant backup
        } else if (finalAbsent && lstatSync(backupPath).isDirectory() && isDownloaderOwned(backupPath)) {
          // Restore a crashed swap — but ONLY an owned REAL-directory backup, so a
          // foreign-symlink backup is never promoted onto finalDir.
          await rename(backupPath, finalDir);
        }
        // else foreign symlink / unowned dir / incomplete-owned → retain untouched
      } catch {
        // A single un-recoverable backup must not block the job.
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

    if (isPathOccupied(finalDir) && !isDownloaderOwned(finalDir) && !overwrite) {
      throw new Error(
        `Refusing to overwrite "${finalDir}": it was not created by the dashboard downloader. ` +
          `Remove it manually to reinstall.`,
      );
    }

    const marker: DownloadCompletion = {
      repo,
      revision,
      files: files.map((file) => file.path),
      completedAt: new Date().toISOString(),
    };
    await writeFile(join(stagingDir, DOWNLOAD_COMPLETE_MARKER), `${JSON.stringify(marker, null, 2)}\n`);

    if (!isPathOccupied(finalDir)) {
      await rename(stagingDir, finalDir);
      await fsyncDir(dirname(finalDir));
      return;
    }

    // Re-check ownership adjacent to the destructive swap: the first guard is a
    // no-op when finalDir was absent then, so an UNOWNED dir that raced in
    // between must still never be renamed away and deleted. This narrows the
    // window to the irreducible FS-level check-then-swap race.
    if (!isDownloaderOwned(finalDir) && !overwrite) {
      throw new Error(`Refusing to overwrite "${finalDir}": it was not created by the dashboard downloader`);
    }

    // Rollback-safe swap of an owned (or explicitly overwritten) final dir. The
    // backup lives under `.staging` (a dotdir skipped by model discovery) so a
    // crash mid-swap never surfaces it as a phantom model.
    const backupDir = join(this.modelsDir, '.staging', `${basename(finalDir)}.backup-${process.pid}.${randomUUID()}`);
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
    });
  }
}
