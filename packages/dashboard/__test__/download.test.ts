import { createHash } from 'node:crypto';
import {
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { MODEL_CATALOG } from '@mlx-node/agent/catalog';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { catalogWithState } from '../src/catalog.js';
import { type DownloadEvent, DownloadManager, pidAlive } from '../src/download.js';
import { DOWNLOAD_COMPLETE_MARKER } from '../src/models.js';

/** Filename of the atomic-publish completion marker (kept in sync with models.ts). */
const MARKER_FILE = '.mlx-download-complete.json';

// The runner now refuses to pin to anything but an immutable 40-hex commit, so
// every stubbed sha is a valid 40-hex string.
const SHA_DEFAULT = 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef';
const SHA_A = 'cafef00dcafef00dcafef00dcafef00dcafef00d';
const SHA_OLD = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const SHA_NEW = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';

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
  /** Present on every file in a Xet-backed repo, ALONGSIDE `oid` and `lfs.oid`. */
  xetHash?: string;
}

// Shared, hoisted state the mocked `@huggingface/hub` reads/writes. Reset per test.
// The mock models a real HF cache: a `snapshots/<rev>/<path>` symlink pointing at a
// `blobs/<...>` file. A pinned revision with an existing pointer is a CACHE HIT and
// is returned WITHOUT re-fetching — so resume (and cache invalidation) are testable.
const hub = vi.hoisted(() => ({
  manifest: [] as ManifestEntry[],
  downloaded: [] as string[],
  /** The commit sha `modelInfo` resolves — the snapshot the whole job should pin. */
  sha: 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef',
  /** Every `revision` the runner threaded into a list/download call. */
  revisions: [] as string[],
  /** Paths whose `downloadFileToCacheDir` should throw, to simulate a mid-job failure. */
  failOn: [] as string[],
  /** Overrides the thrown message, so a remote-sized error body can be simulated. */
  failMessage: null as string | null,
  /** The blob a snapshot pointer resolves to (content-addressed by rev + path). */
  cacheBlob: (cacheDir: string, revision: string, p: string): string =>
    `${cacheDir}/blobs/${revision}__${p.split('/').join('__')}`,
  /** The snapshot pointer (a symlink into blobs) for a file at a revision. */
  cachePointer: (cacheDir: string, revision: string, p: string): string => `${cacheDir}/snapshots/${revision}/${p}`,
}));

// Injectable rename fault used to exercise the publish swap's rollback. A source
// equal to `failFromPath`, or matching `failFromPrefix`, throws. The prefix form
// is needed because staging is now job-private (`<slug>@<sha>.<pid>.<uuid>`) — the
// test can only match its unpredictable name by the stable `<slug>@` prefix, which
// deliberately excludes the `<slug>.backup-` rollback rename.
const renameFault = vi.hoisted(() => ({
  failFromPath: null as string | null,
  failFromPrefix: null as string | null,
}));

// Fires once when the completion marker is written, to inject a directory that
// "races in" between publish's ownership check and its swap (Finding 1). Also
// reused (Finding E) to fire a cancel WHILE the job is committing (publishing).
const raceHook = vi.hoisted(() => ({ onMarkerWrite: null as (() => void) | null }));

// Fires once when the runner enters the post-fetch prune/verify window — the
// recursive `readdir` of the job-private staging dir inside `listStagedFiles`.
// Lets a test cancel AFTER the fetch loop but BEFORE the commit barrier
// (Finding E), the window where cancellation used to be ignored.
const stagingHook = vi.hoisted(() => ({ onVerifyWindow: null as (() => void) | null }));

vi.mock('node:fs/promises', async (importActual) => {
  const actual = await importActual<typeof import('node:fs/promises')>();
  return {
    ...actual,
    rename: async (from: string, to: string) => {
      const f = String(from);
      if (
        (renameFault.failFromPath !== null && f === renameFault.failFromPath) ||
        (renameFault.failFromPrefix !== null && f.startsWith(renameFault.failFromPrefix))
      ) {
        throw new Error(`simulated rename failure moving ${f}`);
      }
      return actual.rename(from, to);
    },
    writeFile: async (path: string, data: string | Uint8Array) => {
      if (raceHook.onMarkerWrite !== null && String(path).endsWith(MARKER_FILE)) {
        const hook = raceHook.onMarkerWrite;
        raceHook.onMarkerWrite = null;
        hook();
      }
      return actual.writeFile(path, data);
    },
    readdir: async (path: string, options?: { recursive?: boolean }) => {
      if (stagingHook.onVerifyWindow !== null && options?.recursive === true && path.includes('.staging')) {
        const hook = stagingHook.onVerifyWindow;
        stagingHook.onVerifyWindow = null;
        hook();
      }
      return actual.readdir(path, options);
    },
  };
});

vi.mock('@huggingface/hub', () => ({
  modelInfo: async (params: { revision?: string }) => {
    if (params.revision !== undefined) hub.revisions.push(params.revision);
    return { sha: hub.sha };
  },
  listFiles: async function* (params: { revision?: string }) {
    if (params.revision !== undefined) hub.revisions.push(params.revision);
    for (const entry of hub.manifest) yield entry;
  },
  downloadFileToCacheDir: async (params: {
    path: string;
    revision?: string;
    cacheDir: string;
    fetch: typeof fetch;
  }) => {
    if (params.revision !== undefined) hub.revisions.push(params.revision);
    if (hub.failOn.includes(params.path)) {
      throw new Error(hub.failMessage ?? `simulated failure for ${params.path}`);
    }
    const revision = params.revision ?? 'main';
    const pointer = hub.cachePointer(params.cacheDir, revision, params.path);
    // Cache hit: a pinned revision returns the existing pointer without re-fetching.
    if (existsSync(pointer)) return pointer;
    // Cache miss: drive the injected (counting) fetch so byte progress fires, then
    // write the blob and link the snapshot pointer at it.
    hub.downloaded.push(params.path);
    const response = await params.fetch(`https://hf.example/${params.path}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    const blob = hub.cacheBlob(params.cacheDir, revision, params.path);
    mkdirSync(dirname(blob), { recursive: true });
    writeFileSync(blob, bytes);
    mkdirSync(dirname(pointer), { recursive: true });
    rmSync(pointer, { force: true });
    symlinkSync(blob, pointer);
    return pointer;
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

/**
 * A fetch that streams normally EXCEPT for `blockPath`, whose response never
 * settles until the job's abort signal fires — simulating a long in-flight
 * download that only stops when the job is cancelled. `onBlock` fires once the
 * blocked fetch is entered so a test can cancel at a deterministic point.
 */
function makeCancelFetch(sizes: Record<string, number>, blockPath: string, onBlock: () => void): typeof fetch {
  const normal = makeFetchImpl(sizes);
  return (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const path = url.split('/').slice(3).join('/');
    if (path !== blockPath) return normal(input, init);
    return new Promise<Response>((_resolve, reject) => {
      onBlock();
      const signal = init?.signal;
      const abort = (): void => reject(new DOMException('The operation was aborted', 'AbortError'));
      if (signal?.aborted) {
        abort();
        return;
      }
      signal?.addEventListener('abort', abort);
    });
  };
}

/**
 * A fetch that streams normally EXCEPT for `stallPath`, which delivers exactly
 * `stallBytes` and then hangs until the job aborts — parking the job PART-WAY
 * through one file while the files before it are already settled. That is the
 * state a page reload lands in, and the only state where the replayed frames
 * have to carry more than the current file's own bytes.
 */
function makeStallingFetch(sizes: Record<string, number>, stallPath: string, stallBytes: number): typeof fetch {
  const normal = makeFetchImpl(sizes);
  return (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const path = url.split('/').slice(3).join('/');
    if (path !== stallPath) return normal(input, init);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(stallBytes));
        const signal = init?.signal;
        const abort = (): void => controller.error(new DOMException('The operation was aborted', 'AbortError'));
        if (signal?.aborted) {
          abort();
          return;
        }
        signal?.addEventListener('abort', abort);
      },
    });
    return Promise.resolve(
      new Response(stream, { status: 200, headers: { 'content-length': String(sizes[path] ?? 0) } }),
    );
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
/** A SECOND catalog repo, for the cases that need two genuinely distinct jobs. */
const REPO_OTHER = MODEL_CATALOG[1]!.hfRepo;

let modelsDir: string;
let cacheDir: string;

function finalDir(): string {
  return join(modelsDir, SLUG);
}
/** The `.staging` root (a dotdir model discovery skips). */
function stagingRoot(): string {
  return join(modelsDir, '.staging');
}
/** Job-private staging dirs left under `.staging` (`<slug>@...`); empty when clean. */
function jobStagingDirs(): string[] {
  return existsSync(stagingRoot()) ? readdirSync(stagingRoot()).filter((n) => n.startsWith(`${SLUG}@`)) : [];
}
/** Backup dirs left under `.staging` (`<slug>.backup-...`); empty when clean. */
function backupDirs(): string[] {
  return existsSync(stagingRoot()) ? readdirSync(stagingRoot()).filter((n) => n.includes('.backup-')) : [];
}
/** The legacy SHARED staging path (`<slug>@<rev>`), used only to plant pre-fix leftovers. */
function legacyStagingDir(revision: string = hub.sha): string {
  return join(stagingRoot(), `${SLUG}@${revision}`);
}
/** Plant a corrupt (wrong-content) HF cache entry — symlink pointer + blob. */
function seedCorruptCache(path: string, revision: string, size: number): { pointer: string; blob: string } {
  const blob = hub.cacheBlob(cacheDir, revision, path);
  const pointer = hub.cachePointer(cacheDir, revision, path);
  mkdirSync(dirname(blob), { recursive: true });
  writeFileSync(blob, Buffer.alloc(size, 0xff));
  mkdirSync(dirname(pointer), { recursive: true });
  rmSync(pointer, { force: true });
  symlinkSync(blob, pointer);
  return { pointer, blob };
}

beforeEach(() => {
  hub.manifest = [
    { type: 'file', path: 'config.json', size: 12 },
    { type: 'file', path: 'model.safetensors', size: 300 },
  ];
  hub.downloaded = [];
  hub.sha = SHA_DEFAULT;
  hub.revisions = [];
  hub.failOn = [];
  hub.failMessage = null;
  renameFault.failFromPath = null;
  renameFault.failFromPrefix = null;
  raceHook.onMarkerWrite = null;
  stagingHook.onVerifyWindow = null;
  modelsDir = mkdtempSync(join(tmpdir(), 'dash-dl-models-'));
  cacheDir = mkdtempSync(join(tmpdir(), 'dash-dl-cache-'));
});

afterEach(() => {
  for (const dir of [modelsDir, cacheDir]) rmSync(dir, { recursive: true, force: true });
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
    // Completion marker is present in the published dir and no job staging remains.
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(jobStagingDirs()).toEqual([]);

    const job = manager.jobs().find((j) => j.id === id)!;
    expect(job.state).toBe('done');
    expect(job.totalBytes).toBe(312);
    expect(job.receivedBytes).toBe(312);
  });

  it('pins one resolved commit sha and threads it into every list/download call', async () => {
    hub.sha = SHA_A;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // listFiles + both downloadFileToCacheDir calls all saw exactly one revision.
    expect(hub.revisions.length).toBeGreaterThan(0);
    expect(new Set(hub.revisions)).toEqual(new Set([SHA_A]));

    // The published marker records the pinned revision.
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      revision: string;
      repo: string;
      files: string[];
    };
    expect(marker.revision).toBe(SHA_A);
    expect(marker.repo).toBe(REPO);
    expect(marker.files).toEqual(expect.arrayContaining(['config.json', 'model.safetensors']));
  });

  it('refuses to pin a missing or mutable sha and pins the 40-hex commit on the normal path', async () => {
    // Missing sha → fail closed (never silently pin a mutable "main").
    hub.sha = '';
    {
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
      expect(existsSync(finalDir())).toBe(false);
    }

    // A branch name (mutable ref) is also refused.
    hub.sha = 'main';
    {
      const manager = new DownloadManager({
        modelsDir,
        cacheDir,
        fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
      });
      const events: DownloadEvent[] = [];
      const id = manager.start(REPO);
      manager.subscribe(id, (event) => events.push(event));
      await waitFor(() => events.some((event) => event.type === 'error'));
      expect(existsSync(finalDir())).toBe(false);
    }

    // Normal path: a 40-hex commit pins and publishes.
    hub.sha = SHA_A;
    {
      const manager = new DownloadManager({
        modelsDir,
        cacheDir,
        fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
      });
      const id = manager.start(REPO);
      await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));
      const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
        revision: string;
      };
      expect(marker.revision).toBe(SHA_A);
    }
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
    // No half-populated FINAL dir; job-private staging is cleaned on failure (resume
    // is HF-cache-backed, not staging-backed, so nothing is left to reuse). The
    // cleanup runs in the job's `finally`, which completes AFTER the `error` event
    // that `waitFor` above unblocked on, so wait for the private staging dir to be
    // reclaimed (a genuine leak times out here) rather than racing that teardown.
    expect(existsSync(finalDir())).toBe(false);
    await waitFor(() => jobStagingDirs().length === 0);
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

  // Finding G2: a one-sided manifest — config-only OR weights-only — is not a
  // loadable model. The job must error (no marker, nothing published, catalog
  // reports not-installed) rather than publish a hollow "installed" dir.
  it('errors on a one-sided manifest (config-only or weights-only) instead of publishing', async () => {
    const cases: ManifestEntry[][] = [
      [{ type: 'file', path: 'config.json', size: 12 }],
      [{ type: 'file', path: 'model.safetensors', size: 300 }],
    ];
    for (const manifest of cases) {
      hub.manifest = manifest;
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
      expect(existsSync(finalDir())).toBe(false);
      expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(false);
      expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(false);
      expect(jobStagingDirs()).toEqual([]);

      rmSync(finalDir(), { recursive: true, force: true });
    }
  });

  it('marks installed only with the completion marker, never bare directory existence', async () => {
    // A bare dir with config.json + a weight but no marker (a legacy/partial download).
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12));
    writeFileSync(join(finalDir(), 'model.safetensors'), Buffer.alloc(300));
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(false);

    // A marker listing both a config and a weight, all present → installed.
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: REPO,
        revision: hub.sha,
        files: ['config.json', 'model.safetensors'],
        completedAt: '2026-07-21T00:00:00Z',
      }),
    );
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);

    // A marker referencing a missing file must NOT count as installed.
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: REPO,
        revision: hub.sha,
        files: ['config.json', 'missing.safetensors'],
        completedAt: 'x',
      }),
    );
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(false);

    // A one-sided marker (config-only, no weight) is never installed either — a
    // hollow publish `loadModel` would reject (Finding G2).
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json'], completedAt: 'x' }),
    );
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(false);
  });

  it('resumes from the HF cache without re-fetching already-cached files', async () => {
    // First job caches config.json, then fails on the weight.
    hub.failOn = ['model.safetensors'];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events1: DownloadEvent[] = [];
    const id1 = manager.start(REPO);
    manager.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'error'));
    expect(hub.downloaded).toContain('config.json');

    // Second job: the weight succeeds; config.json is served from the HF cache.
    hub.failOn = [];
    hub.downloaded = [];
    const id2 = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id2 && j.state === 'done'));

    // config.json was a cache hit (no re-fetch); only the weight was fetched.
    expect(hub.downloaded).not.toContain('config.json');
    expect(hub.downloaded).toContain('model.safetensors');
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  // Finding #4: the install-skip gate must match the marker's repo AND revision,
  // not merely "a complete same-slug install exists". A same-slug install pinned to
  // the SAME resolved revision short-circuits (idempotent, no re-fetch).
  it('short-circuits to done when the installed marker matches the resolved repo+revision', async () => {
    // A complete owned install pinned to the exact revision resolveRevision returns.
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(finalDir(), 'model.safetensors'), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', 'model.safetensors'], completedAt: 'x' }),
    );

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // Idempotent: nothing re-fetched, the existing install untouched.
    expect(hub.downloaded).toEqual([]);
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    const job = manager.jobs().find((j) => j.id === id)!;
    expect(job.receivedBytes).toBe(job.totalBytes);
  });

  it('does not short-circuit a current partial CLI marker; downloads and publishes the full model', async () => {
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(finalDir(), 'model.safetensors'), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: REPO,
        revision: hub.sha,
        files: ['config.json', 'model.safetensors'],
        scope: 'partial',
        completedAt: 'x',
      }),
    );

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    expect(hub.downloaded).toEqual(expect.arrayContaining(['config.json', 'model.safetensors']));
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      scope?: string;
    };
    expect(marker.scope).toBe('full');
  });

  // Finding #4: a complete install of a DIFFERENT revision (same slug) must NOT
  // short-circuit to `done` at the new revision's byte total — the job must download
  // the resolved revision and let publish's owned-swap replace the stale one.
  it('does NOT short-circuit when the installed marker is a different revision; downloads and swaps', async () => {
    // A complete OWNED install pinned to an OLD revision (marker present + every file
    // there → isModelInstalled true), but resolveRevision returns hub.sha (newer).
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(finalDir(), 'model.safetensors'), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: SHA_OLD, files: ['config.json', 'model.safetensors'], completedAt: 'x' }),
    );
    hub.sha = SHA_NEW; // the resolved revision differs from the installed SHA_OLD

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // It genuinely downloaded the resolved revision (never falsely reported done)…
    expect(hub.downloaded).toEqual(expect.arrayContaining(['config.json', 'model.safetensors']));
    // …and the owned-swap replaced the old 0xAB content with the fresh zero-bytes.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    expect(readFileSync(join(finalDir(), 'model.safetensors'))).toEqual(Buffer.alloc(300));
    // The published marker now records the newly resolved revision; no backup leaked.
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      revision: string;
    };
    expect(marker.revision).toBe(SHA_NEW);
    expect(backupDirs()).toEqual([]);
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('re-fetches and recovers when a corrupt cached blob fails the sha256 check', async () => {
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      {
        type: 'file',
        path: 'model.safetensors',
        size: 300,
        lfs: { oid: zerosSha256(300), size: 300, pointerSize: 100 },
      },
    ];
    // A pre-existing corrupt cache blob (0xFF instead of the 0x00 the download writes).
    seedCorruptCache('model.safetensors', hub.sha, 300);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The corrupt cache entry failed sha256 → was invalidated → really re-fetched →
    // matched → published (without invalidation the retry would recopy the same blob).
    expect(hub.downloaded).toContain('model.safetensors');
    expect(readFileSync(join(finalDir(), 'model.safetensors'))).toEqual(Buffer.alloc(300));
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('content-verifies a XET-backed weight, which carries all three digest fields at once', async () => {
    // The shape a real Xet repo actually returns — checked against the live API for
    // `Brooooooklyn/Qwen3.6-27B-NVFP4-mlx`, where every weight has `oid` AND
    // `lfs.oid` AND `xetHash` together:
    //
    //   "oid": "a90b8dec…"                      git-blob sha1 of the POINTER
    //   "lfs": { "oid": "4f44f844…" }           sha256 of the CONTENT
    //   "xetHash": "8b5d1bdf…"                  Merkle/chunk hash, not recomputable here
    //
    // So a Xet weight is not a digest-less file: `lfs.oid` is a plain sha256 of the
    // bytes (verified by fetching one and hashing it), and the resume check must
    // take that branch. The `xetHash === undefined` clause on the git-blob branch
    // exists only to keep the POINTER `oid` from being hashed against content — it
    // must never read as "Xet files are unverifiable, accept on size".
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12, oid: zerosGitOid(12) },
      {
        type: 'file',
        path: 'model.safetensors',
        size: 300,
        oid: 'a'.repeat(40),
        lfs: { oid: zerosSha256(300), size: 300, pointerSize: 135 },
        xetHash: 'b'.repeat(64),
      },
    ];
    // Corrupt at EXACTLY the manifest size — the case size alone cannot catch.
    seedCorruptCache('model.safetensors', hub.sha, 300);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    expect(hub.downloaded, 'a same-length corrupt Xet blob was accepted without a re-fetch').toContain(
      'model.safetensors',
    );
    expect(readFileSync(join(finalDir(), 'model.safetensors'))).toEqual(Buffer.alloc(300));
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('re-fetches when a corrupt cached blob fails the git-blob oid check', async () => {
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12, oid: zerosGitOid(12) },
      { type: 'file', path: 'model.safetensors', size: 300 },
    ];
    seedCorruptCache('config.json', hub.sha, 12);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // Wrong git-blob oid → invalidated → re-fetched → matched.
    expect(hub.downloaded).toContain('config.json');
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('invalidates the cached pointer AND blob when staged bytes never match the manifest', async () => {
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      // lfs.oid deliberately WRONG (sha256 of different content); 300 zero-bytes can
      // never match it → post-copy verification always fails.
      {
        type: 'file',
        path: 'model.safetensors',
        size: 300,
        lfs: { oid: zerosSha256(299), size: 300, pointerSize: 100 },
      },
    ];
    const { pointer, blob } = seedCorruptCache('model.safetensors', hub.sha, 300);

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
    // Each failed verify removed the cache pointer + blob; the final attempt's
    // removal is not followed by a re-fetch, so both are gone.
    expect(existsSync(pointer)).toBe(false);
    expect(existsSync(blob)).toBe(false);
    // It genuinely re-fetched (bounded) rather than recopying the seeded blob.
    expect(hub.downloaded.filter((p) => p === 'model.safetensors').length).toBeGreaterThan(0);
    expect(existsSync(finalDir())).toBe(false);
  });

  it('errors without publishing when downloaded bytes never match the manifest hash', async () => {
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      {
        type: 'file',
        path: 'model.safetensors',
        size: 300,
        lfs: { oid: zerosSha256(299), size: 300, pointerSize: 100 },
      },
    ];
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
    // The mismatching file was re-fetched (bounded) before the job gave up.
    expect(hub.downloaded.filter((p) => p === 'model.safetensors').length).toBeGreaterThan(1);
    // No publish, no marker, catalog reports not-installed.
    expect(existsSync(finalDir())).toBe(false);
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(false);
  });

  it('fails fast on an unowned existing dir — before listing or downloading any bytes', async () => {
    // A valid checkpoint placed manually / by `mlx download` — no completion
    // marker, so the downloader does not own it. The refusal must land UP FRONT,
    // never after a multi-GB download+hash.
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(finalDir(), 'model.safetensors'), Buffer.alloc(300, 0xab));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    // Refused BEFORE the manifest was listed (no `start`) and BEFORE any file was
    // fetched (no bytes copied, no staging tree created).
    expect(events.some((event) => event.type === 'start')).toBe(false);
    expect(hub.downloaded).toEqual([]);
    expect(jobStagingDirs()).toEqual([]);

    const err = events.find((event) => event.type === 'error');
    expect(err?.type).toBe('error');
    if (err?.type === 'error') {
      // The message must not advertise an `overwrite` mode the API/UI never expose.
      expect(err.message).not.toMatch(/overwrite/i);
      expect(err.message).toMatch(/not created by the dashboard/i);
    }

    // The manual files are byte-for-byte intact; no marker written.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(false);
  });

  it('flags the state it refuses: an interrupted CLI download is blocked, not installable', async () => {
    // An interrupted `mlx download` leaves a config.json and no weights, so the dir
    // is NOT present — which used to render an enabled Install button whose job the
    // preflight below refuses every time. Catalog state and the runner must agree on
    // that dir, which is why both read the same no-follow occupancy/ownership pair.
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));

    const item = catalogWithState(modelsDir).find((entry) => entry.slug === SLUG)!;
    expect(item.present).toBe(false);
    expect(item.blockedByForeignDir).toBe(true);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    const err = events.find((event) => event.type === 'error');
    expect(err?.type).toBe('error');
    if (err?.type === 'error') expect(err.message).toMatch(/not created by the dashboard/i);
    // Zero bytes moved and the partial dir is untouched — a wart, not data loss.
    expect(hub.downloaded).toEqual([]);
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
  });

  it('fails fast on a DANGLING final symlink — before listing or downloading any bytes', async () => {
    // `<modelsDir>/<slug>` is a DANGLING symlink (its target does not exist — e.g. it
    // points at an unmounted volume). `existsSync` FOLLOWS the link and reports the
    // path ABSENT, which used to skip the ownership preflight and trigger a full
    // wasted download that only failed at publish. A no-follow occupancy check
    // (`lstatSync`) sees the link itself and refuses UP FRONT — the link is never
    // downloader-owned, so it is treated exactly like a foreign dir.
    symlinkSync(join(modelsDir, 'nonexistent-target'), finalDir());

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));

    // Refused BEFORE the manifest was listed (no `start`) and BEFORE any file was
    // fetched (no bytes copied, no staging tree created).
    expect(events.some((event) => event.type === 'start')).toBe(false);
    expect(hub.downloaded).toEqual([]);
    expect(jobStagingDirs()).toEqual([]);
    expect(events.some((event) => event.type === 'done')).toBe(false);

    const err = events.find((event) => event.type === 'error');
    expect(err?.type).toBe('error');
    if (err?.type === 'error') {
      expect(err.message).toMatch(/not created by the dashboard/i);
    }
  });

  it('fails fast on a LIVE final symlink → external marked dir — before any bytes (no-follow ownership)', async () => {
    // `<modelsDir>/<slug>` is a LIVE symlink whose target is an EXTERNAL directory
    // that itself carries a valid completion marker. `existsSync`/`readFileSync`
    // FOLLOW the link, so a laxer ownership check reads the foreign marker and thinks
    // the dir is downloader-owned — proceeding to overwrite/report-done through a path
    // the runner never wrote. The occupancy check is no-follow (`lstatSync` sees the
    // link) and `isDownloaderOwned` is no-follow (it `lstat`-gates on a real directory
    // before reading the marker), so the preflight refuses UP FRONT.
    const external = mkdtempSync(join(tmpdir(), 'dash-dl-ext-'));
    writeFileSync(join(external, 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(external, 'model.safetensors'), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(external, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: SHA_OLD, files: ['config.json', 'model.safetensors'], completedAt: 'x' }),
    );
    symlinkSync(external, finalDir());

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));

    // Refused BEFORE the manifest was listed (no `start`) and BEFORE any file was
    // fetched (no bytes copied, no staging tree created); never reported done.
    expect(events.some((event) => event.type === 'start')).toBe(false);
    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(hub.downloaded).toEqual([]);
    expect(jobStagingDirs()).toEqual([]);

    const err = events.find((event) => event.type === 'error');
    expect(err?.type).toBe('error');
    if (err?.type === 'error') {
      expect(err.message).toMatch(/not created by the dashboard/i);
    }

    // The external target is byte-for-byte intact — nothing was written through the link.
    expect(readFileSync(join(external, 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(external, 'model.safetensors'))).toEqual(Buffer.alloc(300, 0xab));

    rmSync(external, { recursive: true, force: true });
  });

  it('does NOT reap a dead-owner rollback backup when finalDir is a LIVE foreign symlink (preflight before sweep)', async () => {
    // finalDir is a LIVE symlink into an EXTERNAL complete marked model AND a
    // dead-owner rollback backup (`<slug>.backup-<deadpid>.<uuid>`) sits under
    // `.staging`. The recovery sweep's reap branch uses `isModelInstalled(finalDir)`,
    // which FOLLOWS symlinks: if the sweep runs before the no-follow ownership
    // preflight refuses the link, it follows the link → reads the foreign marker as
    // "installed" → permanently deletes the recoverable backup — data loss for a
    // doomed job. The preflight must run BEFORE the sweep (and the reap site must
    // classify finalDir no-follow), so a foreign symlink leaves the backup untouched.
    const external = mkdtempSync(join(tmpdir(), 'dash-dl-ext-'));
    writeFileSync(join(external, 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(external, 'model.safetensors'), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(external, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: SHA_OLD, files: ['config.json', 'model.safetensors'], completedAt: 'x' }),
    );
    // A recoverable dead-owner rollback backup (pid 999999999 is DEAD).
    const backup = join(stagingRoot(), `${SLUG}.backup-999999999.44444444-4444-4444-4444-444444444444`);
    mkdirSync(backup, { recursive: true });
    writeFileSync(join(backup, 'config.json'), Buffer.alloc(12, 0x5a));
    writeFileSync(join(backup, 'model.safetensors'), Buffer.alloc(300, 0x5a));
    writeFileSync(
      join(backup, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: SHA_OLD, files: ['config.json', 'model.safetensors'], completedAt: 'x' }),
    );
    symlinkSync(external, finalDir());

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));

    // Fast-fail: refused up front — no manifest listing, no bytes, never done.
    expect(events.some((event) => event.type === 'start')).toBe(false);
    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(hub.downloaded).toEqual([]);
    const err = events.find((event) => event.type === 'error');
    expect(err?.type).toBe('error');
    if (err?.type === 'error') {
      expect(err.message).toMatch(/not created by the dashboard/i);
    }

    // The recoverable dead-owner rollback backup was NOT reaped — byte-for-byte intact.
    expect(existsSync(backup)).toBe(true);
    expect(readFileSync(join(backup, 'config.json'))).toEqual(Buffer.alloc(12, 0x5a));
    expect(readFileSync(join(backup, 'model.safetensors'))).toEqual(Buffer.alloc(300, 0x5a));
    expect(backupDirs()).toEqual([`${SLUG}.backup-999999999.44444444-4444-4444-4444-444444444444`]);

    // The external symlink target is byte-for-byte intact — nothing written through the link.
    expect(readFileSync(join(external, 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(external, 'model.safetensors'))).toEqual(Buffer.alloc(300, 0xab));

    rmSync(external, { recursive: true, force: true });
  });

  it('refuses to overwrite an unowned existing dir and leaves it intact', async () => {
    // A valid checkpoint placed manually / by `mlx download` — no completion
    // marker, so the downloader does not own it.
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(finalDir(), 'model.safetensors'), Buffer.alloc(300, 0xab));

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
    // The manual files are byte-for-byte intact and no marker was written into them.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(finalDir(), 'model.safetensors'))).toEqual(Buffer.alloc(300, 0xab));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(false);
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(false);
  });

  it('refuses a dir whose marker is not the full shape we write, and leaves it intact', async () => {
    // Ownership is the ONLY gate between `POST /api/downloads` and
    // `rename(finalDir → backup)` + `rm(backup, { recursive: true })`: the route
    // never parses `overwrite`, so all three guards on that path reduce to this
    // one predicate. It used to accept anything carrying a `files` array, and the
    // marker we write has been the same four fields since the first commit that
    // emitted one — so a bare `{"files":[]}` authorized destroying a directory
    // that was never ours.
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'PRECIOUS.safetensors'), Buffer.alloc(4096, 0x7f));
    writeFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), '{"files":[]}');

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(
      readFileSync(join(finalDir(), 'PRECIOUS.safetensors')),
      'a marker carrying only `files` authorized the destructive publish swap',
    ).toEqual(Buffer.alloc(4096, 0x7f));
    expect(events.some((event) => event.type === 'done')).toBe(false);
    // And the catalog must agree, or the UI would keep offering the Install that
    // the runner now refuses every time.
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.blockedByForeignDir).toBe(true);
  });

  it('does not hand back an in-flight job that is already cancelling', async () => {
    // `cancel()` of the IN-FLIGHT job returns while the state is still `running`:
    // the terminal transition is left to `processJob` as it unwinds, which is what
    // stops the `cancelled` event being emitted twice. A repo lookup that reads
    // only the state would hand that job back to the next Install, so the caller
    // would subscribe to a stream whose next frame is `cancelled` — and nothing
    // would download. The window is a whole non-abortable file copy + hash wide.
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events1: DownloadEvent[] = [];
    const id1 = manager.start(REPO);
    manager.subscribe(id1, (event) => events1.push(event));
    let idDuringCancel: string | undefined;
    stagingHook.onVerifyWindow = () => {
      manager.cancel(id1);
      // SYNCHRONOUSLY inside the window — no await, so `processJob` has not yet
      // reached its catch and the job still reads `running`.
      idDuringCancel = manager.start(REPO);
    };
    await waitFor(() => events1.some((event) => event.type === 'cancelled'));
    expect(idDuringCancel).not.toBe(id1);
  });

  it('refuses to overwrite an unowned dir that races in between the ownership check and the swap', async () => {
    // finalDir is ABSENT at the first ownership check; an external process installs an
    // UNOWNED dir just before the swap (modeled by creating it when the marker is
    // written into staging). The swap must re-check ownership and refuse — never
    // rename+delete the raced-in dir.
    raceHook.onMarkerWrite = () => {
      mkdirSync(finalDir(), { recursive: true });
      writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xcd));
      writeFileSync(join(finalDir(), 'model.safetensors'), Buffer.alloc(300, 0xcd));
    };

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
    // The raced-in unowned dir is byte-for-byte intact; no marker written into it.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xcd));
    expect(readFileSync(join(finalDir(), 'model.safetensors'))).toEqual(Buffer.alloc(300, 0xcd));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(false);
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(false);
    // No leaked backup was left behind by the refused swap.
    expect(backupDirs()).toEqual([]);
  });

  it('replaces an unowned dir only when overwrite is explicitly requested', async () => {
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO, { overwrite: true });
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The freshly downloaded (zero-byte) content replaced the manual 0xAB copy.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('restores the original owned dir when the publish swap rename fails', async () => {
    // An OWNED but incomplete dir (marker lists a file that is missing) → not
    // "installed", so the job proceeds to re-download and reach the publish swap.
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: REPO,
        revision: SHA_OLD,
        files: ['config.json', 'model.safetensors'],
        completedAt: 'x',
      }),
    );

    // Fault the staging→final rename so the swap fails AFTER the backup move. Staging
    // is job-private, so match its unpredictable name by the stable `<slug>@` prefix
    // (which excludes the `<slug>.backup-` rollback rename).
    renameFault.failFromPrefix = join(stagingRoot(), `${SLUG}@`);

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
    // Original owned dir restored intact: its 0xAB config and old marker survive.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      revision: string;
    };
    expect(marker.revision).toBe(SHA_OLD);
    // No orphaned backup dir left under modelsDir/.staging.
    expect(backupDirs()).toEqual([]);
  });

  it('recovers an orphaned dead-owner publish backup whose final dir is missing on the next download', async () => {
    // A crash left the model missing under an orphaned backup (swap died after the
    // finalDir→backup move). The backup is a complete owned install; its owner pid
    // (999999999) is DEAD, so the sweep is free to reclaim it.
    const backup = join(stagingRoot(), `${SLUG}.backup-999999999.11111111-1111-1111-1111-111111111111`);
    mkdirSync(backup, { recursive: true });
    writeFileSync(join(backup, 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(backup, 'model.safetensors'), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(backup, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', 'model.safetensors'], completedAt: 'x' }),
    );
    expect(existsSync(finalDir())).toBe(false);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The sweep renamed the backup back into place; the restored (0xAB) content is
    // the completed install, so the job short-circuits without re-downloading.
    expect(existsSync(finalDir())).toBe(true);
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(backupDirs()).toEqual([]);
    expect(hub.downloaded).toEqual([]);
  });

  it('reaps a leaked dead-owner publish backup when its final dir already exists', async () => {
    // A complete owned install already present.
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12));
    writeFileSync(join(finalDir(), 'model.safetensors'), Buffer.alloc(300));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', 'model.safetensors'], completedAt: 'x' }),
    );
    // A leaked backup from a prior successful swap that never got cleaned; its owner
    // pid (999999999) is DEAD, so the sweep reaps it.
    const backup = join(stagingRoot(), `${SLUG}.backup-999999999.22222222-2222-2222-2222-222222222222`);
    mkdirSync(backup, { recursive: true });
    writeFileSync(join(backup, 'config.json'), Buffer.alloc(12, 0x5));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The leaked backup was removed; the existing install is untouched.
    expect(existsSync(backup)).toBe(false);
    expect(backupDirs()).toEqual([]);
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
  });

  it('never touches a live-owner publish backup (a peer mid-publish), and does not restore it over a fresh download', async () => {
    // finalDir is ABSENT; a backup tagged with a LIVE owner pid (this process) is a
    // peer mid-swap — the sweep must leave it alone rather than reclaim it. The job
    // then downloads fresh into finalDir; the live backup survives byte-for-byte.
    const liveBackup = join(stagingRoot(), `${SLUG}.backup-${process.pid}.33333333-3333-3333-3333-333333333333`);
    mkdirSync(liveBackup, { recursive: true });
    writeFileSync(join(liveBackup, 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(liveBackup, 'model.safetensors'), Buffer.alloc(300, 0xab));
    expect(existsSync(finalDir())).toBe(false);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The live-owner backup was NEVER restored/reaped: it survives intact…
    expect(existsSync(liveBackup)).toBe(true);
    expect(readFileSync(join(liveBackup, 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    // …and finalDir holds the freshly downloaded (zero-byte) content, not the backup's 0xAB.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
  });

  // F4: a MALFORMED backup dir name (`<slug>.backup-<garbage>` / `<slug>.backup-`)
  // whose owner pid can't be parsed by BACKUP_DIR_RE must fail CLOSED — left
  // untouched, never promoted into the model path and never reaped — mirroring
  // `pidAlive`'s unknown-owner→alive stance. Here finalDir is ABSENT: pre-fix, the
  // null match short-circuited the live-PID guard and the unparseable backup was
  // renamed onto finalDir; the fix leaves it in `.staging` and downloads fresh.
  it('leaves an unparseable backup dir untouched when finalDir is absent (fails closed)', async () => {
    const garbage = join(stagingRoot(), `${SLUG}.backup-garbage`);
    const emptyPid = join(stagingRoot(), `${SLUG}.backup-`);
    mkdirSync(garbage, { recursive: true });
    writeFileSync(join(garbage, 'config.json'), Buffer.alloc(12, 0xab));
    mkdirSync(emptyPid, { recursive: true });
    writeFileSync(join(emptyPid, 'config.json'), Buffer.alloc(12, 0xcd));
    expect(existsSync(finalDir())).toBe(false);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && (j.state === 'done' || j.state === 'error')));

    // Both unparseable backups survive byte-for-byte in `.staging` (not renamed away).
    expect(existsSync(garbage)).toBe(true);
    expect(readFileSync(join(garbage, 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(existsSync(emptyPid)).toBe(true);
    expect(readFileSync(join(emptyPid, 'config.json'))).toEqual(Buffer.alloc(12, 0xcd));
    // finalDir holds the FRESHLY downloaded (zero-byte) content + marker, not the garbage.
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
  });

  // F4: the same fail-closed guard when a COMPLETE owned install already occupies
  // finalDir. Pre-fix, the null match let the unparseable backup be `rm`'d (owner
  // treated as dead); the fix leaves it untouched while the job short-circuits on
  // the existing install.
  it('leaves an unparseable backup dir untouched when finalDir is already installed (fails closed)', async () => {
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12));
    writeFileSync(join(finalDir(), 'model.safetensors'), Buffer.alloc(300));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', 'model.safetensors'], completedAt: 'x' }),
    );
    const garbage = join(stagingRoot(), `${SLUG}.backup-garbage`);
    const emptyPid = join(stagingRoot(), `${SLUG}.backup-`);
    mkdirSync(garbage, { recursive: true });
    writeFileSync(join(garbage, 'config.json'), Buffer.alloc(12, 0xab));
    mkdirSync(emptyPid, { recursive: true });
    writeFileSync(join(emptyPid, 'config.json'), Buffer.alloc(12, 0xcd));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The unparseable backups were neither reaped nor promoted: both survive intact.
    expect(existsSync(garbage)).toBe(true);
    expect(readFileSync(join(garbage, 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(existsSync(emptyPid)).toBe(true);
    expect(readFileSync(join(emptyPid, 'config.json'))).toEqual(Buffer.alloc(12, 0xcd));
    // The existing install is untouched and the job resumed (no re-download).
    expect(hub.downloaded).toEqual([]);
  });

  it('does NOT invalidate or unlink a cache pointer whose PARENT resolves outside the cache dir', async () => {
    // The pointer's PARENT dir is a symlink escaping the managed cache (only possible
    // via a foreign/poisoned cache layout, never via the hub). On verify failure the
    // containment guard must skip invalidation entirely: neither the pointer symlink
    // nor its out-of-cache target may be removed.
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      {
        type: 'file',
        path: 'model.safetensors',
        // lfs.oid deliberately WRONG so post-copy verification always fails and the
        // invalidation path runs on every attempt.
        size: 300,
        lfs: { oid: zerosSha256(299), size: 300, pointerSize: 100 },
      },
    ];
    // Redirect the revision dir under snapshots/ to an EXTERNAL location so the
    // pointer's parent realpaths OUTSIDE cacheDir.
    const extRevDir = join(modelsDir, 'ext-rev');
    mkdirSync(extRevDir, { recursive: true });
    mkdirSync(join(cacheDir, 'snapshots'), { recursive: true });
    symlinkSync(extRevDir, join(cacheDir, 'snapshots', hub.sha));
    // A foreign blob (outside the cache) with wrong content, reached via a cache-HIT
    // pointer symlink that lives inside the escaped revision dir.
    const foreignBlob = join(modelsDir, 'ext-victim.bin');
    writeFileSync(foreignBlob, Buffer.alloc(300, 0xff));
    const pointerInExt = join(extRevDir, 'model.safetensors');
    symlinkSync(foreignBlob, pointerInExt);

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
    // The foreign blob SURVIVES (containment refused to delete it)…
    expect(existsSync(foreignBlob)).toBe(true);
    expect(readFileSync(foreignBlob)).toEqual(Buffer.alloc(300, 0xff));
    // …and the pointer symlink itself was NOT unlinked (its parent escaped the cache).
    expect(readdirSync(extRevDir)).toContain('model.safetensors');
    expect(existsSync(finalDir())).toBe(false);
  });

  it('publishes exactly the manifest, ignoring a stale file in a legacy shared staging path', async () => {
    // A stale weight left by a pre-fix shared-staging layout; job-private staging
    // never adopts it, so it can never enter the published set.
    mkdirSync(legacyStagingDir(), { recursive: true });
    writeFileSync(join(legacyStagingDir(), 'stale.safetensors'), Buffer.alloc(50, 0x7));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The published set is exactly the manifest — no stale orphan.
    expect(existsSync(join(finalDir(), 'stale.safetensors'))).toBe(false);
    expect(existsSync(join(finalDir(), 'config.json'))).toBe(true);
    expect(existsSync(join(finalDir(), 'model.safetensors'))).toBe(true);
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('does not reuse another revision staged files across a revision change', async () => {
    // An interrupted OLDER revision left a single-file weight in its own (legacy
    // shared) staging scope.
    mkdirSync(legacyStagingDir(SHA_OLD), { recursive: true });
    writeFileSync(join(legacyStagingDir(SHA_OLD), 'model.safetensors'), Buffer.alloc(999, 0x9));

    // The current revision is SHARDED — no single-file `model.safetensors`.
    hub.sha = SHA_NEW;
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'model-00001-of-00002.safetensors', size: 100 },
      { type: 'file', path: 'model-00002-of-00002.safetensors', size: 100 },
    ];

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({
        'config.json': 12,
        'model-00001-of-00002.safetensors': 100,
        'model-00002-of-00002.safetensors': 100,
      }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The stale single-file weight never enters the new sharded install.
    expect(existsSync(join(finalDir(), 'model.safetensors'))).toBe(false);
    expect(existsSync(join(finalDir(), 'model-00001-of-00002.safetensors'))).toBe(true);
    expect(existsSync(join(finalDir(), 'model-00002-of-00002.safetensors'))).toBe(true);
    // The old revision staging dir is a separate scope, left untouched.
    expect(existsSync(join(legacyStagingDir(SHA_OLD), 'model.safetensors'))).toBe(true);
  });

  it('replays the start frame then the last event to a late subscriber', async () => {
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    const replayed: DownloadEvent[] = [];
    manager.subscribe(id, (event) => replayed.push(event));
    // A late subscriber first gets the one-shot `start` frame (job totalBytes /
    // fileCount) so the UI renders an aggregate bar rather than coarse file-index
    // progress, then the terminal event — no duplicate when `start` IS the latest.
    expect(replayed).toHaveLength(2);
    expect(replayed[0]).toMatchObject({ type: 'start', totalBytes: 312, fileCount: 2 });
    expect(replayed[1]).toMatchObject({ type: 'done' });
  });

  it('replays a mid-job progress frame carrying the WHOLE job aggregate, not just the current file', async () => {
    // Three files of distinct sizes so a dropped one is a distinct number: the
    // replay is one `start` plus ONE `progress` frame, and that frame's own
    // `receivedBytes` is per-FILE. A subscriber that attaches here (a page
    // reload mid-download) has never seen the settled frames of files 1-2, so
    // unless the frame states the job aggregate it can only render the current
    // file's bytes — 1 MiB of a 4 MiB job, under-reporting by 3 MiB.
    const MIB = 1024 * 1024;
    const SHARD = 'model-00002-of-00002.safetensors';
    hub.manifest = [
      { type: 'file', path: 'config.json', size: MIB },
      { type: 'file', path: 'model-00001-of-00002.safetensors', size: 2 * MIB },
      { type: 'file', path: SHARD, size: 4 * MIB },
    ];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeStallingFetch(
        { 'config.json': MIB, 'model-00001-of-00002.safetensors': 2 * MIB, [SHARD]: 4 * MIB },
        SHARD,
        MIB,
      ),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    // Park on the third file, 1 MiB in: files 1-2 settled (1 + 2 MiB), so the
    // job has 4 MiB of its 7 MiB.
    await waitFor(() => events.some((event) => event.type === 'progress' && event.file === SHARD));

    // The aggregate advances by whole files and never rewinds: each finished
    // file's LAST frame states the running total, not the file's own bytes.
    const progress = events.filter((event) => event.type === 'progress');
    const settleOf = (path: string): DownloadEvent => progress.filter((event) => event.file === path).at(-1)!;
    expect(settleOf('config.json')).toMatchObject({ receivedBytes: MIB, jobReceivedBytes: MIB });
    expect(settleOf('model-00001-of-00002.safetensors')).toMatchObject({
      receivedBytes: 2 * MIB,
      jobReceivedBytes: 3 * MIB,
    });
    const aggregates = progress.map((event) => (event.type === 'progress' ? event.jobReceivedBytes : 0));
    for (let i = 1; i < aggregates.length; i++) expect(aggregates[i]).toBeGreaterThanOrEqual(aggregates[i - 1]);

    const replayed: DownloadEvent[] = [];
    manager.subscribe(id, (event) => replayed.push(event));
    expect(replayed).toHaveLength(2);
    expect(replayed[0]).toMatchObject({ type: 'start', totalBytes: 7 * MIB, fileCount: 3 });
    expect(replayed[1]).toMatchObject({
      type: 'progress',
      file: SHARD,
      fileIndex: 2,
      // Per-file bytes: what this one file has received so far…
      receivedBytes: MIB,
      totalBytes: 4 * MIB,
      // …and the job-level aggregate the bar is actually drawn from.
      jobReceivedBytes: 4 * MIB,
    });
    // The same number `GET /api/downloads` already reports for the job.
    expect(manager.jobs().find((j) => j.id === id)!.receivedBytes).toBe(4 * MIB);

    manager.cancel(id);
    await waitFor(() => events.some((event) => event.type === 'cancelled'));
  });

  it('unlinks the snapshot pointer but never deletes a blob resolving OUTSIDE the cache dir', async () => {
    // A poisoned/foreign snapshot pointer whose realpath escapes the managed cache
    // (a server-controlled `oid`/`etag` with `../`, or a pre-existing foreign symlink
    // in the shared HF cache). On verify failure the pointer must be unlinked, but the
    // out-of-cache target it resolves to must NOT be deleted.
    const victim = join(modelsDir, 'external', 'victim.bin');
    mkdirSync(dirname(victim), { recursive: true });
    writeFileSync(victim, Buffer.alloc(300, 0xff));

    // lfs.oid deliberately WRONG so post-copy verification always fails and the retry
    // path (the code that removes the pointer + blob) actually runs.
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      {
        type: 'file',
        path: 'model.safetensors',
        size: 300,
        lfs: { oid: zerosSha256(299), size: 300, pointerSize: 100 },
      },
    ];
    // Seed a cache-HIT pointer as a symlink pointing at the out-of-cache victim.
    const pointer = hub.cachePointer(cacheDir, hub.sha, 'model.safetensors');
    mkdirSync(dirname(pointer), { recursive: true });
    symlinkSync(victim, pointer);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    // The out-of-cache blob SURVIVES (the containment gate refused to delete it)…
    expect(existsSync(victim)).toBe(true);
    expect(readFileSync(victim)).toEqual(Buffer.alloc(300, 0xff));
    // …while the snapshot pointer itself was unlinked (safe — a symlink rm is never followed).
    expect(existsSync(pointer)).toBe(false);
    expect(existsSync(finalDir())).toBe(false);
  });

  it('reaps a dead-pid staging tree on the next download but keeps a live-pid one', async () => {
    mkdirSync(stagingRoot(), { recursive: true });
    // A crashed/SIGKILLed job's private staging tree: pid 999999999 cannot be live.
    const deadDir = join(stagingRoot(), `${SLUG}@${hub.sha}.999999999.11111111-1111-1111-1111-111111111111`);
    mkdirSync(deadDir, { recursive: true });
    writeFileSync(join(deadDir, 'model.safetensors'), Buffer.alloc(10, 0x1));
    // A concurrent live job's private staging tree (this process' own, alive pid).
    const liveDir = join(stagingRoot(), `${SLUG}@${hub.sha}.${process.pid}.22222222-2222-2222-2222-222222222222`);
    mkdirSync(liveDir, { recursive: true });
    writeFileSync(join(liveDir, 'model.safetensors'), Buffer.alloc(10, 0x2));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The dead-pid tree was reaped; the live-pid tree was left intact.
    expect(existsSync(deadDir)).toBe(false);
    expect(existsSync(liveDir)).toBe(true);
    expect(readFileSync(join(liveDir, 'model.safetensors'))).toEqual(Buffer.alloc(10, 0x2));
    // The job still published normally.
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
  });

  // Finding 3: the dead-owner private-staging-tree reap must run BEFORE the
  // ownership preflight, so a job the preflight REFUSES still frees a crashed
  // multi-GB staging tree instead of leaking it until some other eligible download
  // for the slug happens to run. Pre-fix (preflight above the sweep) the refused job
  // threw before the reap ever ran → the dead tree remained.
  it('reaps a dead-pid staging tree even when the ownership preflight refuses the job', async () => {
    mkdirSync(stagingRoot(), { recursive: true });
    // A crashed/SIGKILLed job's private staging tree (pid 999999999 cannot be live).
    const deadDir = join(stagingRoot(), `${SLUG}@${hub.sha}.999999999.11111111-1111-1111-1111-111111111111`);
    mkdirSync(deadDir, { recursive: true });
    writeFileSync(join(deadDir, 'model.safetensors'), Buffer.alloc(10, 0x1));

    // finalDir is an UNOWNED real dir (a manual / `mlx download` copy, no marker), so
    // the ownership preflight refuses this job (overwrite:false).
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(finalDir(), 'model.safetensors'), Buffer.alloc(300, 0xab));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    // The job was refused up front (unowned finalDir) — never listed, never done.
    expect(events.some((event) => event.type === 'start')).toBe(false);
    expect(events.some((event) => event.type === 'done')).toBe(false);
    const err = events.find((event) => event.type === 'error');
    expect(err?.type).toBe('error');
    if (err?.type === 'error') {
      expect(err.message).toMatch(/not created by the dashboard/i);
    }

    // …yet the crashed dead-owner staging tree WAS reaped despite the refusal.
    expect(existsSync(deadDir)).toBe(false);

    // The unowned finalDir is byte-for-byte intact — never overwritten, no marker.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(finalDir(), 'model.safetensors'))).toEqual(Buffer.alloc(300, 0xab));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(false);
  });

  // Finding 1: a rollback backup can be a FOREIGN SYMLINK (a prior `overwrite`
  // publish renamed a symlinked finalDir to its backup, then crashed before
  // installing staging). With finalDir ABSENT, pre-fix recovery renamed that symlink
  // onto finalDir AFTER the only ownership check, then skip-detection FOLLOWED the
  // link and emitted `done` through the foreign target. Recovery must restore only an
  // OWNED REAL-DIR backup, and a re-run preflight must refuse any foreign link at
  // finalDir before skip-detection can follow it.
  it('does NOT restore a foreign-symlink backup onto an absent finalDir, and never reports done through it', async () => {
    const external = mkdtempSync(join(tmpdir(), 'dash-dl-ext-'));
    writeFileSync(join(external, 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(external, 'model.safetensors'), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(external, DOWNLOAD_COMPLETE_MARKER),
      // repo + revision MATCH the resolved snapshot so pre-fix skip-detection would
      // read the foreign marker as already-installed and emit `done` through the link.
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', 'model.safetensors'], completedAt: 'x' }),
    );
    // A dead-owner (pid 999999999) rollback backup that is a SYMLINK into `external`.
    mkdirSync(stagingRoot(), { recursive: true });
    const backup = join(stagingRoot(), `${SLUG}.backup-999999999.44444444-4444-4444-4444-444444444444`);
    symlinkSync(external, backup);
    expect(existsSync(finalDir())).toBe(false);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));

    // The foreign-symlink backup was NEVER promoted onto finalDir as a live install:
    // finalDir is either absent (refused) or a REAL dir with our freshly downloaded
    // (zero-byte) content — never the foreign symlink, never the foreign 0xAB bytes.
    if (existsSync(finalDir())) {
      expect(lstatSync(finalDir()).isSymbolicLink()).toBe(false);
    }
    const done = events.find((event) => event.type === 'done');
    if (done?.type === 'done') {
      expect(lstatSync(finalDir()).isSymbolicLink()).toBe(false);
      expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    }

    // The external symlink target is byte-for-byte intact — nothing was written or
    // published through the foreign link.
    expect(readFileSync(join(external, 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(external, 'model.safetensors'))).toEqual(Buffer.alloc(300, 0xab));

    rmSync(external, { recursive: true, force: true });
  });

  it('rejects a repo that is not in the catalog', () => {
    const manager = new DownloadManager({ modelsDir, cacheDir, fetchImpl: makeFetchImpl({}) });
    expect(() => manager.start('someone/not-in-catalog')).toThrow();
  });

  // Two POSTs for one repo before the first response lands (a double-click on
  // Install, or a second dashboard tab whose card still reads Install) used to
  // allocate two jobs. The Models page keeps ONE job id per repo, so it tracked
  // only the last: cancelling the visible card left the other job downloading —
  // and publishing — with no card to stop it. A repo that already has a
  // nonterminal job hands back THAT job's id, so the second request is idempotent.
  it('reuses the in-flight job for a repo instead of allocating a second one', async () => {
    let blocked = false;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeCancelFetch({ 'config.json': 12, 'model.safetensors': 300 }, 'model.safetensors', () => {
        blocked = true;
      }),
    });
    const id1 = manager.start(REPO);
    await waitFor(() => blocked);

    expect(manager.start(REPO)).toBe(id1);
    expect(manager.jobs().filter((job) => job.repo === REPO)).toHaveLength(1);

    // The id the UI tracks is the only running job, so Cancel really stops it.
    expect(manager.cancel(id1)).toBe(true);
    await waitFor(() => manager.jobs().find((job) => job.id === id1)!.state === 'cancelled');
    await waitFor(() => jobStagingDirs().length === 0);
    expect(existsSync(finalDir())).toBe(false);
  });

  // The reuse covers a job that is still QUEUED (behind another repo's download)
  // as well as the in-flight one, and never collapses two DIFFERENT repos.
  it('reuses a queued job for the same repo but keeps a different repo on its own job', async () => {
    let blocked = false;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeCancelFetch({ 'config.json': 12, 'model.safetensors': 300 }, 'model.safetensors', () => {
        blocked = true;
      }),
    });
    // Occupy the single-threaded drain with a different repo's job.
    const idOther = manager.start(REPO_OTHER);
    await waitFor(() => blocked);

    const queued = manager.start(REPO);
    expect(queued).not.toBe(idOther);
    expect(manager.start(REPO)).toBe(queued);
    expect(manager.jobs()).toHaveLength(2);

    expect(manager.cancel(queued)).toBe(true);
    expect(manager.cancel(idOther)).toBe(true);
    await waitFor(() => manager.jobs().every((job) => job.state === 'cancelled'));
  });

  // Over-correction guard: reuse must apply ONLY while a job is nonterminal. Once
  // the first attempt has failed, Install must start a genuinely new job and
  // install — never hand back the settled failure forever.
  it('allocates a FRESH job once the previous one for that repo failed, so a retry installs', async () => {
    hub.failOn = ['model.safetensors'];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events1: DownloadEvent[] = [];
    const id1 = manager.start(REPO);
    manager.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'error'));

    hub.failOn = [];
    const id2 = manager.start(REPO);
    expect(id2).not.toBe(id1);
    await waitFor(() => manager.jobs().some((job) => job.id === id2 && job.state === 'done'));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
  });

  // Over-correction guard: the same for a CANCELLED job — the card resets to
  // Install, and that Install must run rather than resolve to the cancelled job.
  it('allocates a FRESH job once the previous one for that repo was cancelled', async () => {
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events1: DownloadEvent[] = [];
    const id1 = manager.start(REPO);
    manager.subscribe(id1, (event) => events1.push(event));
    stagingHook.onVerifyWindow = () => {
      manager.cancel(id1);
    };
    await waitFor(() => events1.some((event) => event.type === 'cancelled'));
    await waitFor(() => jobStagingDirs().length === 0);

    const id2 = manager.start(REPO);
    expect(id2).not.toBe(id1);
    await waitFor(() => manager.jobs().some((job) => job.id === id2 && job.state === 'done'));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
  });

  // Finding C: an untrusted `listFiles` path is about to become a
  // `join(stagingDir, path)` write target. A traversal or absolute path must be
  // refused (fail closed) at ingestion — nothing is written outside stagingDir.
  it('fails closed on an unsafe manifest path (traversal or absolute) and writes nothing outside staging', async () => {
    for (const badPath of ['../escape.json', '/etc/evil.json']) {
      hub.manifest = [
        { type: 'file', path: 'config.json', size: 12 },
        { type: 'file', path: badPath, size: 5 },
        { type: 'file', path: 'model.safetensors', size: 300 },
      ];
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
      const err = events.find((event) => event.type === 'error');
      expect(err !== undefined && err.type === 'error' ? err.message : '').toContain('safe relative path');
      // Nothing published, and no traversal/absolute target materialized anywhere.
      expect(existsSync(finalDir())).toBe(false);
      expect(existsSync(join(modelsDir, 'escape.json'))).toBe(false);
      expect(existsSync(join(stagingRoot(), 'escape.json'))).toBe(false);
      expect(existsSync(join(modelsDir, 'evil.json'))).toBe(false);
      expect(jobStagingDirs()).toEqual([]);

      rmSync(finalDir(), { recursive: true, force: true });
    }
  });

  // Finding E: cancelling an in-flight job aborts its download, unwinds so its
  // job-private staging dir is removed, and NEVER purges the shared HF blob cache
  // (the `mlx download` CLI resumes from it).
  it('cancels an in-flight job: aborts the download, cleans staging, leaves the HF cache intact', async () => {
    let blocked = false;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeCancelFetch({ 'config.json': 12, 'model.safetensors': 300 }, 'model.safetensors', () => {
        blocked = true;
      }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));

    // config.json completes and caches; the weight fetch blocks (in flight).
    await waitFor(() => blocked);
    expect(jobStagingDirs().length).toBe(1);
    const cachedPointer = hub.cachePointer(cacheDir, hub.sha, 'config.json');
    const cachedBlob = hub.cacheBlob(cacheDir, hub.sha, 'config.json');
    expect(existsSync(cachedPointer)).toBe(true);
    expect(existsSync(cachedBlob)).toBe(true);

    // Cancel the in-flight job → the aborted fetch unwinds processJob.
    expect(manager.cancel(id)).toBe(true);

    await waitFor(() => events.some((event) => event.type === 'cancelled'));
    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(events.some((event) => event.type === 'error')).toBe(false);
    expect(manager.jobs().find((j) => j.id === id)!.state).toBe('cancelled');

    // Staging cleaned; nothing published.
    expect(jobStagingDirs()).toEqual([]);
    expect(existsSync(finalDir())).toBe(false);
    // Shared HF cache UNTOUCHED — config.json's cached pointer + blob survive.
    expect(existsSync(cachedPointer)).toBe(true);
    expect(existsSync(cachedBlob)).toBe(true);

    // Dismissing an already-terminal id now evicts it (Finding G6) and returns
    // true; a second dismiss of the now-unknown id is a no-op.
    expect(manager.cancel(id)).toBe(true);
    expect(manager.jobs().some((j) => j.id === id)).toBe(false);
    expect(manager.cancel(id)).toBe(false);
    // An unknown id is a no-op too.
    expect(manager.cancel('no-such-job')).toBe(false);
  });

  // Finding E: a still-queued job (behind a busy one) is dropped and settled
  // terminally by cancel without ever starting a download.
  it('cancels a queued job before it starts, without fetching anything', async () => {
    let blocked = false;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeCancelFetch({ 'config.json': 12, 'model.safetensors': 300 }, 'model.safetensors', () => {
        blocked = true;
      }),
    });
    // Job 1 occupies the single-threaded drain (its weight fetch blocks).
    const id1 = manager.start(REPO);
    await waitFor(() => blocked);

    // Job 2 (a DIFFERENT repo — a same-repo request would reuse job 1) queues
    // behind it and is cancelled before it can run.
    const id2 = manager.start(REPO_OTHER);
    const events2: DownloadEvent[] = [];
    manager.subscribe(id2, (event) => events2.push(event));
    expect(manager.cancel(id2)).toBe(true);

    expect(manager.jobs().find((j) => j.id === id2)!.state).toBe('cancelled');
    expect(events2.some((event) => event.type === 'cancelled')).toBe(true);
    // The queued job never emitted a `start` (it never ran).
    expect(events2.some((event) => event.type === 'start')).toBe(false);

    // Clean up job 1 so no blocked fetch lingers.
    expect(manager.cancel(id1)).toBe(true);
    await waitFor(() => manager.jobs().find((j) => j.id === id1)!.state === 'cancelled');
  });

  // Finding E (regression): a cancel that lands AFTER the fetch loop — during the
  // prune/verify window, while the job is still `running` — must unwind to a
  // `cancelled` terminal and NEVER publish. Before the commit barrier this cancel
  // returned 200 yet the model still installed and the job marked `done`.
  it('cancels during the post-fetch verify window: ends cancelled, never publishes', async () => {
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    // Fire the cancel the instant the runner enters the prune/verify window (the
    // recursive readdir of staging) — after every file has been fetched, before
    // the commit barrier. Cancel is still accepted (job is `running`).
    stagingHook.onVerifyWindow = () => {
      expect(manager.cancel(id)).toBe(true);
    };

    await waitFor(() => events.some((event) => event.type === 'cancelled'));
    // No `done`, no `error` — a clean cancelled terminal.
    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(events.some((event) => event.type === 'error')).toBe(false);
    expect(manager.jobs().find((j) => j.id === id)!.state).toBe('cancelled');
    // Nothing published: no final dir, no install marker.
    expect(existsSync(finalDir())).toBe(false);
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(false);
    // The job-private staging dir is removed by the job's `finally` (async, runs
    // just after the terminal event) — wait for it rather than racing the rm.
    await waitFor(() => jobStagingDirs().length === 0);
  });

  // Finding E: once the job enters the non-cancellable `committing` state (the
  // atomic swap has begun), `cancel()` is REFUSED (route would 404) and the model
  // installs to completion — a late cancel can never report success mid-publish.
  it('refuses to cancel once committing (publish in progress) and still installs', async () => {
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    // The marker is written inside `publish`, AFTER the barrier set state to
    // `committing`; a cancel here must be refused (returns false).
    let cancelWhileCommitting: boolean | null = null;
    raceHook.onMarkerWrite = () => {
      cancelWhileCommitting = manager.cancel(id);
    };

    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));
    // The commit-time cancel was refused (the DELETE route would return 404).
    expect(cancelWhileCommitting).toBe(false);
    // The model installed to completion despite the racing cancel.
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(events.some((event) => event.type === 'cancelled')).toBe(false);
    expect(events.some((event) => event.type === 'done')).toBe(true);
    expect(manager.jobs().find((j) => j.id === id)!.state).toBe('done');
  });

  // Finding G6: a FAILED (terminal) job can be DISMISSED via cancel — it is
  // evicted from jobsById/order/lastEvent and no longer listed, so
  // DELETE /api/downloads/:id clears a failed card server-side instead of 404ing
  // and retaining the row forever.
  it('dismisses a failed job: cancel returns true and the job is fully evicted', async () => {
    hub.failOn = ['model.safetensors'];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'error'));

    // Present (and listed) before dismissal…
    expect(manager.jobs().some((j) => j.id === id)).toBe(true);
    // …dismiss evicts it entirely: gone from jobsById/order (so `jobs()` drops it)…
    expect(manager.cancel(id)).toBe(true);
    expect(manager.jobs().some((j) => j.id === id)).toBe(false);
    // …and lastEvent is cleared, so a late subscriber gets NO replayed frame.
    const replayed: DownloadEvent[] = [];
    manager.subscribe(id, (event) => replayed.push(event));
    expect(replayed).toEqual([]);
    // A second dismiss of the now-unknown id is a no-op.
    expect(manager.cancel(id)).toBe(false);
  });

  // Finding G6: dismiss is refused for a non-terminal `committing` job (the
  // publish window) — only settled jobs are evictable.
  it('refuses to dismiss a committing job (still non-terminal)', async () => {
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    let dismissWhileCommitting: boolean | null = null;
    const id = manager.start(REPO);
    // The marker is written inside `publish`, AFTER the barrier set state to
    // `committing`; a dismiss here must be refused (returns false) and the job
    // must remain listed and install to completion.
    raceHook.onMarkerWrite = () => {
      dismissWhileCommitting = manager.cancel(id);
    };

    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));
    expect(dismissWhileCommitting).toBe(false);
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(manager.jobs().find((j) => j.id === id)!.state).toBe('done');
  });

  // Finding #5: a pre-existing `.staging` SYMLINK pointing at an external dir must
  // be refused before any write — `mkdir(recursive)` would otherwise follow it and
  // redirect every staged write and recursive cleanup onto the foreign target.
  it('refuses a symlinked .staging root: writes nothing to and deletes nothing from the external target', async () => {
    // An external dir with a pre-existing file the runner must never touch.
    const external = mkdtempSync(join(tmpdir(), 'dash-dl-ext-'));
    const externalFile = join(external, 'do-not-touch.bin');
    writeFileSync(externalFile, Buffer.alloc(16, 0xee));
    // Plant `.staging` as a symlink → external (a local, pre-existing symlink).
    symlinkSync(external, stagingRoot());

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    // The job errored (refused the symlinked root) and never published.
    expect(events.some((event) => event.type === 'done')).toBe(false);
    const err = events.find((event) => event.type === 'error');
    expect(err !== undefined && err.type === 'error' ? err.message : '').toContain('real directory');
    expect(existsSync(finalDir())).toBe(false);
    // The external target is untouched: its file survives byte-for-byte and NO
    // staged files (config.json / model.safetensors) were written into it.
    expect(existsSync(externalFile)).toBe(true);
    expect(readFileSync(externalFile)).toEqual(Buffer.alloc(16, 0xee));
    expect(readdirSync(external)).toEqual(['do-not-touch.bin']);

    rmSync(external, { recursive: true, force: true });
  });

  // Regression: the server's `close()` must abort in-flight downloads and AWAIT
  // their staging cleanup. Without it, a SIGINT during a multi-GB download lets
  // the CLI `process.exit(0)` before `processJob`'s `finally` removes the
  // job-private `.staging` tree — orphaning a partial, potentially multi-GB
  // directory — and a background job could publish AFTER the server is "closed".
  // `shutdown()` aborts the in-flight job and RESOLVES only once every staging dir
  // is reclaimed (it must not hang).
  it('shutdown aborts an in-flight job, cleans its staging dir, and resolves', async () => {
    let blocked = false;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeCancelFetch({ 'config.json': 12, 'model.safetensors': 300 }, 'model.safetensors', () => {
        blocked = true;
      }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));

    // config.json caches; the weight fetch blocks (in flight) with a staging dir on disk.
    await waitFor(() => blocked);
    expect(jobStagingDirs().length).toBe(1);

    // Resolves (does not hang) only once the aborted job's `finally` has run.
    await manager.shutdown();

    // Job settled cancelled, nothing published, and the staging tree is reclaimed
    // synchronously by the time shutdown resolves (no post-await waitFor needed).
    expect(manager.jobs().find((j) => j.id === id)!.state).toBe('cancelled');
    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(events.some((event) => event.type === 'error')).toBe(false);
    expect(jobStagingDirs()).toEqual([]);
    expect(existsSync(finalDir())).toBe(false);
  });

  // Regression: `shutdown` must also DROP a still-queued job so it never starts,
  // then resolve — the queued job is settled `cancelled` without ever fetching.
  it('shutdown drops a still-queued job without ever fetching it', async () => {
    let blocked = false;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeCancelFetch({ 'config.json': 12, 'model.safetensors': 300 }, 'model.safetensors', () => {
        blocked = true;
      }),
    });
    // Job 1 occupies the single-threaded drain (its weight fetch blocks in flight).
    const id1 = manager.start(REPO);
    await waitFor(() => blocked);

    // Job 2 (a DIFFERENT repo — a same-repo request would reuse job 1) queues
    // behind it (drain is busy) and has not started.
    const id2 = manager.start(REPO_OTHER);
    const events2: DownloadEvent[] = [];
    manager.subscribe(id2, (event) => events2.push(event));

    await manager.shutdown();

    // The queued job was dropped + settled without ever running (no `start`); the
    // in-flight job was aborted; both are terminal and shutdown resolved cleanly.
    expect(manager.jobs().find((j) => j.id === id2)!.state).toBe('cancelled');
    expect(events2.some((event) => event.type === 'start')).toBe(false);
    expect(manager.jobs().find((j) => j.id === id1)!.state).toBe('cancelled');
    expect(jobStagingDirs()).toEqual([]);
    expect(existsSync(finalDir())).toBe(false);
  });

  // Regression: the server's `close()` leaves the HTTP listener attached across
  // `await downloads.shutdown()`, so a `POST /api/downloads` can land mid-drain.
  // `shutdown` snapshots the queue ONCE, but the retained drain loop re-reads the
  // live queue — so a job enqueued after that snapshot is adopted by the very
  // promise `shutdown` awaits, and is never cancelled: one such job hangs close()
  // forever. A closing manager must refuse new work instead.
  it('refuses a job enqueued after shutdown began, so the drain can never be extended', async () => {
    let blocked = false;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeCancelFetch({ 'config.json': 12, 'model.safetensors': 300 }, 'model.safetensors', () => {
        blocked = true;
      }),
    });
    const id1 = manager.start(REPO);
    await waitFor(() => blocked);

    // `shutdown` flips the manager closed before its first await, so the refusal
    // does not depend on how wide the in-flight job's unwind window happens to be.
    const shutdownP = manager.shutdown();
    expect(() => manager.start(REPO)).toThrow(/shutting down/i);
    await shutdownP;

    // The refused job was never registered, so nothing extended the drain and the
    // in-flight job settled cancelled with its staging tree reclaimed.
    expect(manager.jobs()).toHaveLength(1);
    expect(manager.jobs().find((j) => j.id === id1)!.state).toBe('cancelled');
    expect(jobStagingDirs()).toEqual([]);
    expect(existsSync(finalDir())).toBe(false);
  });

  // `error.message` is the only unbounded DownloadEvent field: `@huggingface/hub`
  // copies the REMOTE JSON error body into it verbatim. The frame is retained in
  // `lastEvent` and replayed to every SSE subscriber that attaches later, so a
  // remote-sized string must be truncated before it ever becomes an event.
  it('truncates a remote-sized failure message before broadcasting it', async () => {
    hub.failOn = ['model.safetensors'];
    hub.failMessage = `boom ${'x'.repeat(100_000)}`;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));

    await waitFor(() => events.some((event) => event.type === 'error'));
    const failure = events.find((event) => event.type === 'error')!;
    const message = failure.type === 'error' ? failure.message : '';

    expect(message.startsWith('boom ')).toBe(true);
    expect(message.length).toBeLessThan(4096);
    expect(message.endsWith('…')).toBe(true);
  });
});

describe('pidAlive (download.ts)', () => {
  it('reports the current process alive and a nonexistent pid dead', () => {
    expect(pidAlive(process.pid)).toBe(true);
    expect(pidAlive(999999999)).toBe(false);
  });

  it('treats an invalid pid as ALIVE so a corrupt value can never drive a deletion', () => {
    expect(pidAlive(0)).toBe(true);
    expect(pidAlive(-1)).toBe(true);
    expect(pidAlive(Number.NaN)).toBe(true);
  });
});
