import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { MODEL_CATALOG } from '@mlx-node/agent/catalog';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { catalogWithState } from '../src/catalog.js';
import { type DownloadEvent, DownloadManager } from '../src/download.js';
import { DOWNLOAD_COMPLETE_MARKER } from '../src/models.js';

/** sha256 of `n` zero bytes — the content the stub download writes for a file of size `n`. */
function zerosSha256(n: number): string {
  return createHash('sha256').update(Buffer.alloc(n)).digest('hex');
}

/** git-blob sha1 of `n` zero bytes — the top-level `oid` of a plain (non-LFS) file. */
function zerosGitOid(n: number): string {
  return createHash('sha1').update(`blob ${n}\0`).update(Buffer.alloc(n)).digest('hex');
}

interface ManifestEntry {
  type: 'file' | 'directory';
  path: string;
  size: number;
  oid?: string;
  lfs?: { oid: string; size: number; pointerSize: number };
}

// Shared, hoisted state the mocked `@huggingface/hub` reads/writes. Reset per test.
const hub = vi.hoisted(() => ({
  manifest: [] as ManifestEntry[],
  blobDir: '',
  downloaded: [] as string[],
  /** The commit sha `modelInfo` resolves — the snapshot the whole job should pin. */
  sha: 'commit-deadbeef',
  /** Every `revision` the runner threaded into a list/download call. */
  revisions: [] as string[],
  /** Paths whose `downloadFileToCacheDir` should throw, to simulate a mid-job failure. */
  failOn: [] as string[],
}));

vi.mock('@huggingface/hub', () => ({
  modelInfo: async (params: { revision?: string }) => {
    if (params.revision !== undefined) hub.revisions.push(params.revision);
    return { sha: hub.sha };
  },
  listFiles: async function* (params: { revision?: string }) {
    if (params.revision !== undefined) hub.revisions.push(params.revision);
    for (const entry of hub.manifest) yield entry;
  },
  downloadFileToCacheDir: async (params: { path: string; revision?: string; fetch: typeof fetch }) => {
    if (params.revision !== undefined) hub.revisions.push(params.revision);
    if (hub.failOn.includes(params.path)) throw new Error(`simulated failure for ${params.path}`);
    hub.downloaded.push(params.path);
    // Drive the injected (counting) fetch so byte progress fires.
    const response = await params.fetch(`https://hf.example/${params.path}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    const dest = join(hub.blobDir, params.path);
    mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, bytes);
    return dest;
  },
}));

/** A stub fetch that streams `sizes[path]` zero-bytes in three chunks. */
function makeFetchImpl(sizes: Record<string, number>): typeof fetch {
  return async (input: RequestInfo | URL): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const path = url.split('/').slice(3).join('/');
    const size = sizes[path] ?? 0;
    const chunkSize = Math.max(1, Math.ceil(size / 3));
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        let sent = 0;
        while (sent < size) {
          const n = Math.min(chunkSize, size - sent);
          controller.enqueue(new Uint8Array(n));
          sent += n;
        }
        controller.close();
      },
    });
    return new Response(stream, { status: 200, headers: { 'content-length': String(size) } });
  };
}

async function waitFor(cond: () => boolean, timeoutMs = 5000): Promise<void> {
  const t0 = Date.now();
  while (!cond()) {
    if (Date.now() - t0 > timeoutMs) throw new Error('timed out waiting for condition');
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

const REPO = MODEL_CATALOG[0]!.hfRepo;
const SLUG = REPO.split('/').pop()!.toLowerCase();

let modelsDir: string;
let cacheDir: string;

function finalDir(): string {
  return join(modelsDir, SLUG);
}
function stagingDir(): string {
  return join(modelsDir, '.staging', SLUG);
}

beforeEach(() => {
  hub.manifest = [
    { type: 'file', path: 'config.json', size: 12 },
    { type: 'file', path: 'model.safetensors', size: 300 },
  ];
  hub.blobDir = mkdtempSync(join(tmpdir(), 'dash-hub-'));
  hub.downloaded = [];
  hub.sha = 'commit-deadbeef';
  hub.revisions = [];
  hub.failOn = [];
  modelsDir = mkdtempSync(join(tmpdir(), 'dash-dl-models-'));
  cacheDir = mkdtempSync(join(tmpdir(), 'dash-dl-cache-'));
});

afterEach(() => {
  for (const dir of [hub.blobDir, modelsDir, cacheDir]) rmSync(dir, { recursive: true, force: true });
});

describe('DownloadManager', () => {
  it('emits start → per-file progress → done and atomically publishes into modelsDir/<slug>', async () => {
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));

    await waitFor(() => events.some((event) => event.type === 'done'));

    const start = events.find((event) => event.type === 'start');
    expect(start).toMatchObject({ type: 'start', repo: REPO, totalBytes: 312, fileCount: 2 });

    const progress = events.filter((event) => event.type === 'progress');
    expect(progress.length).toBeGreaterThan(0);

    // Per-file byte counts grow monotonically and settle at the file size.
    const modelProgress = progress
      .filter((event) => event.type === 'progress' && event.file === 'model.safetensors')
      .map((event) => (event.type === 'progress' ? event.receivedBytes : 0));
    expect(modelProgress.length).toBeGreaterThan(1);
    for (let i = 1; i < modelProgress.length; i++) {
      expect(modelProgress[i]).toBeGreaterThanOrEqual(modelProgress[i - 1]);
    }
    expect(modelProgress[modelProgress.length - 1]).toBe(300);

    const done = events.find((event) => event.type === 'done');
    expect(done).toMatchObject({ type: 'done', outputDir: finalDir() });

    expect(existsSync(join(finalDir(), 'config.json'))).toBe(true);
    expect(readFileSync(join(finalDir(), 'model.safetensors')).length).toBe(300);
    // Completion marker is present in the published dir and the staging dir is gone.
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(existsSync(stagingDir())).toBe(false);

    const job = manager.jobs().find((j) => j.id === id)!;
    expect(job.state).toBe('done');
    expect(job.totalBytes).toBe(312);
    expect(job.receivedBytes).toBe(312);
  });

  it('pins one resolved commit sha and threads it into every list/download call', async () => {
    hub.sha = 'commit-cafef00d';
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // listFiles + both downloadFileToCacheDir calls all saw exactly one revision.
    expect(hub.revisions.length).toBeGreaterThan(0);
    expect(new Set(hub.revisions)).toEqual(new Set(['commit-cafef00d']));

    // The published marker records the pinned revision.
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      revision: string;
      repo: string;
      files: string[];
    };
    expect(marker.revision).toBe('commit-cafef00d');
    expect(marker.repo).toBe(REPO);
    expect(marker.files).toEqual(expect.arrayContaining(['config.json', 'model.safetensors']));
  });

  it('leaves NO final dir when a job errors mid-way; catalog shows not-installed', async () => {
    hub.failOn = ['model.safetensors'];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));

    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    // config.json got fetched before the failure — the job was genuinely mid-way.
    expect(hub.downloaded).toContain('config.json');
    // No half-populated FINAL dir; only the resumable staging dir remains.
    expect(existsSync(finalDir())).toBe(false);
    expect(existsSync(stagingDir())).toBe(true);
    // Catalog must NOT report the aborted download as installed.
    const item = catalogWithState(modelsDir).find((entry) => entry.slug === SLUG)!;
    expect(item.installed).toBe(false);
  });

  it('errors on an empty manifest instead of publishing a hollow dir', async () => {
    hub.manifest = [];
    const manager = new DownloadManager({ modelsDir, cacheDir, fetchImpl: makeFetchImpl({}) });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));

    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(events.find((event) => event.type === 'error')).toMatchObject({ type: 'error' });
    expect(existsSync(finalDir())).toBe(false);
    const item = catalogWithState(modelsDir).find((entry) => entry.slug === SLUG)!;
    expect(item.installed).toBe(false);
  });

  it('marks installed only with the completion marker, never bare directory existence', async () => {
    // A bare dir with config.json but no marker (a legacy/partial download).
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12));
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(false);

    // Add a marker whose listed files all exist → now installed.
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json'], completedAt: '2026-07-21T00:00:00Z' }),
    );
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);

    // A marker referencing a missing file must NOT count as installed.
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', 'model.safetensors'], completedAt: 'x' }),
    );
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(false);
  });

  it('re-verifies staged content on resume rather than trusting a right-size file', async () => {
    // Manifest carries an LFS sha256 for the weight file; the runner must match it.
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      {
        type: 'file',
        path: 'model.safetensors',
        size: 300,
        lfs: { oid: zerosSha256(300), size: 300, pointerSize: 100 },
      },
    ];
    // Pre-stage a same-SIZE but WRONG-content weight (0xFF instead of the 0x00 the
    // stub download writes) — size matches, sha256 does not.
    mkdirSync(stagingDir(), { recursive: true });
    writeFileSync(join(stagingDir(), 'model.safetensors'), Buffer.alloc(300, 0xff));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The stale-but-right-size file failed content verification and was re-fetched.
    expect(hub.downloaded).toContain('model.safetensors');
  });

  it('skips a staged file whose content hash matches the manifest (resume, no re-fetch)', async () => {
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      {
        type: 'file',
        path: 'model.safetensors',
        size: 300,
        lfs: { oid: zerosSha256(300), size: 300, pointerSize: 100 },
      },
    ];
    // Pre-stage BOTH files with the exact content the download would produce.
    mkdirSync(stagingDir(), { recursive: true });
    writeFileSync(join(stagingDir(), 'config.json'), Buffer.alloc(12));
    writeFileSync(join(stagingDir(), 'model.safetensors'), Buffer.alloc(300));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // Content matched → nothing re-fetched; the job still published atomically.
    expect(hub.downloaded).toEqual([]);
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('re-verifies a plain file against its git-blob oid on resume', async () => {
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12, oid: zerosGitOid(12) },
      { type: 'file', path: 'model.safetensors', size: 300 },
    ];
    // Pre-stage a same-size but wrong-content config.json (git-blob sha1 differs).
    mkdirSync(stagingDir(), { recursive: true });
    writeFileSync(join(stagingDir(), 'config.json'), Buffer.alloc(12, 0xff));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // Wrong git-blob oid → re-fetched despite the matching size.
    expect(hub.downloaded).toContain('config.json');
  });

  it('skips a size-matched metadata file with no manifest hash (resume)', async () => {
    // config.json has no oid/lfs in the manifest → size-only skip is retained.
    mkdirSync(stagingDir(), { recursive: true });
    writeFileSync(join(stagingDir(), 'config.json'), Buffer.alloc(12));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    expect(hub.downloaded).toContain('model.safetensors');
    expect(hub.downloaded).not.toContain('config.json');
  });

  it('replays the last event to a late subscriber', async () => {
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    const replayed: DownloadEvent[] = [];
    manager.subscribe(id, (event) => replayed.push(event));
    expect(replayed).toHaveLength(1);
    expect(replayed[0]).toMatchObject({ type: 'done' });
  });

  it('rejects a repo that is not in the catalog', () => {
    const manager = new DownloadManager({ modelsDir, cacheDir, fetchImpl: makeFetchImpl({}) });
    expect(() => manager.start('someone/not-in-catalog')).toThrow();
  });
});
