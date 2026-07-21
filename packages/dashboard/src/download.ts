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
 * Downloads are made atomic to close finding-6's partial/mixed-revision hole:
 * the whole job is pinned to ONE resolved commit sha; files stage into a private
 * `<modelsDir>/.staging/<slug>` dir; and only after every file is present and
 * content-verified is a completion marker written and the staging dir renamed
 * onto the final `<slug>`. An aborted job leaves only the staging dir (resumable)
 * — never a half-populated final dir that catalog state would call "installed".
 *
 * Pure disk + network — never the native addon.
 */

import { createHash, randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { copyFile, mkdir, rename, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

import { downloadFileToCacheDir, listFiles, type ListFileEntry, modelInfo } from '@huggingface/hub';
import { MODEL_CATALOG } from '@mlx-node/agent/catalog';

import { type DownloadCompletion, DOWNLOAD_COMPLETE_MARKER, isModelInstalled } from './models.js';

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

  /** Queue a catalog repo for download. Rejects any repo not in the catalog. */
  start(repo: string): string {
    if (!MODEL_CATALOG.some((entry) => entry.hfRepo === repo)) {
      throw new Error(`Repo "${repo}" is not in the model catalog`);
    }
    const id = randomUUID();
    this.jobsById.set(id, { id, repo, state: 'running', receivedBytes: 0, totalBytes: 0 });
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
   * a mixed-revision checkpoint. Falls back to `main` only if the hub omits a sha.
   */
  private async resolveRevision(repoName: string): Promise<string> {
    const info = await modelInfo({ name: repoName, additionalFields: ['sha'], fetch: this.wrappedFetch });
    const sha: unknown = info.sha;
    return typeof sha === 'string' && sha.length > 0 ? sha : 'main';
  }

  private async processJob(job: JobState): Promise<void> {
    const entry = MODEL_CATALOG.find((candidate) => candidate.hfRepo === job.repo)!;
    const slug = entry.hfRepo.split('/').pop()!.toLowerCase();
    const finalDir = join(this.modelsDir, slug);
    const stagingDir = join(this.modelsDir, '.staging', slug);
    const repo = { type: 'model' as const, name: job.repo };
    this.currentJob = job;

    try {
      const revision = await this.resolveRevision(job.repo);

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

        const context: FileContext = {
          filePath: file.path,
          fileSize: file.size,
          fileIndex: index,
          fileCount: files.length,
          jobBaseBytes,
          perUrl: new Map(),
          received: 0,
        };
        this.currentFile = context;
        const snapshotPath = await downloadFileToCacheDir({
          repo,
          path: file.path,
          revision,
          cacheDir: this.cacheDir,
          fetch: this.wrappedFetch,
        });
        await mkdir(dirname(destPath), { recursive: true });
        await copyFile(snapshotPath, destPath);
        this.currentFile = null;

        // Settle the file at its manifest size (or the counted bytes when the
        // manifest omits a size), so aggregate progress is exact even if a
        // cache hit or a metadata-only response streamed nothing.
        const fileBytes = file.size > 0 ? file.size : context.received;
        job.receivedBytes = jobBaseBytes + fileBytes;
        this.emitFileProgress(job.id, file.path, fileBytes, file.size, index, files.length);
      }

      await this.publish(stagingDir, finalDir, job.repo, revision, files);

      job.state = 'done';
      this.emit({ type: 'done', id: job.id, outputDir: finalDir });
    } catch (error) {
      job.state = 'error';
      this.emit({ type: 'error', id: job.id, message: (error as Error).message });
    } finally {
      this.currentJob = null;
      this.currentFile = null;
    }
  }

  /**
   * Atomic publish: write the completion marker as the LAST staged step (so it
   * travels with the rename and only ever coexists with a full file set), then
   * rename the staging dir onto the final path. A pre-existing final dir here is
   * an incomplete/legacy copy — a complete one already short-circuited — so it is
   * removed first, otherwise the rename would fail with ENOTEMPTY.
   */
  private async publish(
    stagingDir: string,
    finalDir: string,
    repo: string,
    revision: string,
    files: ListFileEntry[],
  ): Promise<void> {
    const marker: DownloadCompletion = {
      repo,
      revision,
      files: files.map((file) => file.path),
      completedAt: new Date().toISOString(),
    };
    await writeFile(join(stagingDir, DOWNLOAD_COMPLETE_MARKER), `${JSON.stringify(marker, null, 2)}\n`);
    await mkdir(dirname(finalDir), { recursive: true });
    await rm(finalDir, { recursive: true, force: true });
    await rename(stagingDir, finalDir);
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
