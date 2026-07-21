import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { MODEL_CATALOG } from '@mlx-node/agent/catalog';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { type DownloadEvent, DownloadManager } from '../src/download.js';

// Shared, hoisted state the mocked `@huggingface/hub` reads/writes. Reset per test.
const hub = vi.hoisted(() => ({
  manifest: [] as Array<{ type: 'file' | 'directory'; path: string; size: number }>,
  blobDir: '',
  downloaded: [] as string[],
}));

vi.mock('@huggingface/hub', () => ({
  listFiles: async function* (_params: unknown) {
    for (const entry of hub.manifest) yield entry;
  },
  downloadFileToCacheDir: async (params: { path: string; fetch: typeof fetch }) => {
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

beforeEach(() => {
  hub.manifest = [
    { type: 'file', path: 'config.json', size: 12 },
    { type: 'file', path: 'model.safetensors', size: 300 },
  ];
  hub.blobDir = mkdtempSync(join(tmpdir(), 'dash-hub-'));
  hub.downloaded = [];
  modelsDir = mkdtempSync(join(tmpdir(), 'dash-dl-models-'));
  cacheDir = mkdtempSync(join(tmpdir(), 'dash-dl-cache-'));
});

afterEach(() => {
  for (const dir of [hub.blobDir, modelsDir, cacheDir]) rmSync(dir, { recursive: true, force: true });
});

describe('DownloadManager', () => {
  it('emits start → per-file progress → done and writes files into modelsDir/<slug>', async () => {
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
    expect(done).toMatchObject({ type: 'done', outputDir: join(modelsDir, SLUG) });

    expect(existsSync(join(modelsDir, SLUG, 'config.json'))).toBe(true);
    expect(readFileSync(join(modelsDir, SLUG, 'model.safetensors')).length).toBe(300);

    const job = manager.jobs().find((j) => j.id === id)!;
    expect(job.state).toBe('done');
    expect(job.totalBytes).toBe(312);
    expect(job.receivedBytes).toBe(312);
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

  it('skips files already present at the right size (resume)', async () => {
    // Pre-place a correct-size config.json so the runner should not re-fetch it.
    mkdirSync(join(modelsDir, SLUG), { recursive: true });
    writeFileSync(join(modelsDir, SLUG, 'config.json'), Buffer.alloc(12));

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

  it('rejects a repo that is not in the catalog', () => {
    const manager = new DownloadManager({ modelsDir, cacheDir, fetchImpl: makeFetchImpl({}) });
    expect(() => manager.start('someone/not-in-catalog')).toThrow();
  });
});
