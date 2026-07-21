/**
 * Resumable catalog download runner for the dashboard.
 *
 * Ports the manifest/resume behavior of `packages/cli/src/commands/download-model.ts`
 * (list files → per-file `downloadFileToCacheDir` into the HF cache → `copyFile`
 * into the model dir, skipping files already present at the right size) and adds
 * structured, per-file byte progress by wrapping the injected `fetch` in a
 * counting `TransformStream`. Jobs run one at a time; consumers observe progress
 * through `subscribe`, which replays the last event on attach so a late attacher
 * (e.g. an SSE reconnect) is never left blank.
 *
 * Pure disk + network — never the native addon.
 */

import { randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { existsSync, statSync } from 'node:fs';
import { copyFile, mkdir } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

import { downloadFileToCacheDir, listFiles, type ListFileEntry } from '@huggingface/hub';
import { MODEL_CATALOG } from '@mlx-node/agent/catalog';

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

/** Per-file resume check: present on disk AND byte size matches the manifest. */
function isLocalCopyComplete(destPath: string, expectedSize: number): boolean {
  if (!existsSync(destPath)) return false;
  if (expectedSize <= 0) return true;
  try {
    return statSync(destPath).size === expectedSize;
  } catch {
    return false;
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

  private async processJob(job: JobState): Promise<void> {
    const entry = MODEL_CATALOG.find((candidate) => candidate.hfRepo === job.repo)!;
    const slug = entry.hfRepo.split('/').pop()!.toLowerCase();
    const outputDir = join(this.modelsDir, slug);
    const repo = { type: 'model' as const, name: job.repo };
    this.currentJob = job;

    try {
      const files: ListFileEntry[] = [];
      let totalBytes = 0;
      for await (const file of listFiles({ repo, recursive: true, fetch: this.wrappedFetch })) {
        if (file.type !== 'directory' && isWantedFile(file.path)) {
          files.push(file);
          if (file.size > 0) totalBytes += file.size;
        }
      }
      job.totalBytes = totalBytes;
      this.emit({ type: 'start', id: job.id, repo: job.repo, totalBytes, fileCount: files.length });

      await mkdir(outputDir, { recursive: true });

      for (let index = 0; index < files.length; index++) {
        const file = files[index];
        const destPath = join(outputDir, file.path);
        const jobBaseBytes = job.receivedBytes;

        if (isLocalCopyComplete(destPath, file.size)) {
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

      job.state = 'done';
      this.emit({ type: 'done', id: job.id, outputDir });
    } catch (error) {
      job.state = 'error';
      this.emit({ type: 'error', id: job.id, message: (error as Error).message });
    } finally {
      this.currentJob = null;
      this.currentFile = null;
    }
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
