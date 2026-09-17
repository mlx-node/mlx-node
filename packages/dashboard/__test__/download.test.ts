import { execFileSync } from 'node:child_process';
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

import { catalogUpdateRepos, catalogRepo, MODEL_CATALOG } from '@mlx-node/agent/catalog';
import { findDFlash2Draft, QWEN38_DFLASH2 } from '@mlx-node/lm/draft-companion';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { catalogWithState } from '../src/catalog.js';
import { type DownloadEvent, DownloadManager, pidAlive } from '../src/download.js';
import { DOWNLOAD_COMPLETE_MARKER, isModelInstalled } from '../src/models.js';

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
  /**
   * Per-repo `listFiles` override, keyed by repo name. A job dials TWO repos — the
   * catalog entry's repo and its `assetsRepo` — and the real ones differ, so a test
   * that needs a GGUF repo without tokenizer files gives the assets repo its own
   * listing. Absent key → {@link manifest}, the pre-assetsRepo behaviour.
   */
  manifests: {} as Record<string, ManifestEntry[]>,
  downloaded: [] as string[],
  /** Every cache-MISS download as `<repo>/<path>`, so attribution is assertable. */
  downloadedFrom: [] as string[],
  /** The commit sha `modelInfo` resolves — the snapshot the whole job should pin. */
  sha: 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef',
  /**
   * Per-repo `modelInfo` override, keyed by repo name; falls back to {@link sha}.
   * A job dials the entry's repo AND its `assetsRepo`, and those are two different
   * repos with two different heads, so "pinned to one snapshot" is only assertable
   * per repo when the mock can answer them differently.
   */
  shaByRepo: {} as Record<string, string>,
  /** Every `revision` the runner threaded into a list/download call. */
  revisions: [] as string[],
  /** The same revisions, keyed by the repo the call was made against. */
  revisionsByRepo: {} as Record<string, string[]>,
  /** Repos `modelInfo` was asked to resolve, for the catalog sha sweep. */
  modelInfoRepos: [] as string[],
  /** When set, every `modelInfo` call throws it — the offline case. */
  modelInfoError: null as string | null,
  /** Repos whose `modelInfo` throws, for a PARTIALLY failed sweep. */
  modelInfoFailRepos: [] as string[],
  /**
   * Make `modelInfo` actually CALL the fetch it is handed.
   *
   * Off by default so existing tests keep their cheap stub. The probe's
   * deadline and its abort signal live in that fetch, so they are unreachable
   * — and untestable — unless the mock exercises it.
   */
  modelInfoUsesFetch: false,
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

/**
 * Record a revision the runner threaded into a hub call — on the global list AND
 * keyed by the repo it was made against. A job spans TWO repos (the entry's and
 * its `assetsRepo`), so a global set can no longer say which repo was pinned to
 * what: each has its own head and must carry its own sha.
 */
function recordRevision(repo: string | undefined, revision: string | undefined): void {
  if (revision === undefined) return;
  hub.revisions.push(revision);
  if (repo !== undefined) (hub.revisionsByRepo[repo] ??= []).push(revision);
}

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
      // `includes` catches the atomic writer too: `writeMarkerAtomically` puts
      // the marker's bytes in `<marker>.<pid>.<uuid>.tmp` before the rename, so
      // a refresh-time hook fires on the temp write, not the final path.
      if (raceHook.onMarkerWrite !== null && String(path).includes(MARKER_FILE)) {
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
  modelInfo: async (params: { name?: string; revision?: string; fetch?: typeof fetch }) => {
    if (params.name !== undefined) hub.modelInfoRepos.push(params.name);
    if (hub.modelInfoError !== null) throw new Error(hub.modelInfoError);
    if (params.name !== undefined && hub.modelInfoFailRepos.includes(params.name)) {
      throw new Error(`simulated modelInfo failure for ${params.name}`);
    }
    // Snapshot BEFORE any await, the way a real server answers with the sha as
    // of the request. Reading it afterwards would let a parked call return a
    // value written while it waited — which silently made the sweep-race test
    // tautological, passing with the fix removed.
    const sha = hub.shaByRepo[params.name ?? ''] ?? hub.sha;
    if (hub.modelInfoUsesFetch && params.fetch !== undefined) {
      await params.fetch(`https://huggingface.co/api/models/${params.name ?? 'x'}`);
    }
    recordRevision(params.name, params.revision);
    return { sha };
  },
  listFiles: async function* (params: { repo?: { name?: string }; revision?: string }) {
    recordRevision(params.repo?.name, params.revision);
    const name = params.repo?.name;
    for (const entry of (name !== undefined ? hub.manifests[name] : undefined) ?? hub.manifest) yield entry;
  },
  downloadFileToCacheDir: async (params: {
    repo?: { name?: string };
    path: string;
    revision?: string;
    cacheDir: string;
    fetch: typeof fetch;
  }) => {
    recordRevision(params.repo?.name, params.revision);
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
    hub.downloadedFrom.push(`${params.repo?.name ?? ''}/${params.path}`);
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

// Resolved through `catalogRepo`, matching the allowlist gate in `start` — the
// catalog carries a different build per platform, so the raw `hfRepo` would be
// refused on a CUDA host.
const REPO = catalogRepo(MODEL_CATALOG[0]!);
const SLUG = REPO.split('/').pop()!.toLowerCase();
/** A SECOND catalog repo, for the cases that need two genuinely distinct jobs. */
const REPO_OTHER = catalogRepo(MODEL_CATALOG[1]!);
/**
 * The `Qwen/Qwen3.8-27B` base-model repo `MODEL_CATALOG[0].assetsRepo` names for
 * the tokenizer/config sidecars a GGUF quantization repo does not ship.
 */
const ASSETS_REPO = MODEL_CATALOG[0]!.assetsRepo!;
/**
 * The single UD-Q4_K_XL weight variant `MODEL_CATALOG[0].globs` selects out of the
 * multi-variant repo. Entry-based jobs stage THIS name: the entry ships its
 * weights as a `.gguf` and its globs (`*UD-Q4_K_XL*`, `config.json`)
 * match no `.safetensors` path at all, so a safetensors fixture would be filtered
 * out of the manifest and the job would fail the weight-payload gate.
 */
const WEIGHT = 'Qwen3.8-27B-UD-Q4_K_XL.gguf';

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
  // The default manifest is the CATALOG ENTRY's repo shape: the glob-matched
  // UD-Q4_K_XL weight plus the core `config.json` (a GGUF quantization repo ships
  // no tokenizer, so the assets repo supplies the rest — see `hub.manifests`).
  hub.manifest = [
    { type: 'file', path: 'config.json', size: 12 },
    { type: 'file', path: WEIGHT, size: 300 },
  ];
  hub.manifests = {};
  hub.downloaded = [];
  hub.downloadedFrom = [];
  hub.sha = SHA_DEFAULT;
  hub.shaByRepo = {};
  hub.revisions = [];
  hub.revisionsByRepo = {};
  hub.failOn = [];
  hub.failMessage = null;
  renameFault.failFromPath = null;
  renameFault.failFromPrefix = null;
  raceHook.onMarkerWrite = null;
  stagingHook.onVerifyWindow = null;
  hub.modelInfoRepos = [];
  hub.modelInfoError = null;
  hub.modelInfoFailRepos = [];
  hub.modelInfoUsesFetch = false;
  modelsDir = mkdtempSync(join(tmpdir(), 'dash-dl-models-'));
  cacheDir = mkdtempSync(join(tmpdir(), 'dash-dl-cache-'));
});

afterEach(() => {
  for (const dir of [modelsDir, cacheDir]) rmSync(dir, { recursive: true, force: true });
});

describe('DownloadManager.checkCatalogUpdates — the read half of the staleness check', () => {
  function manager(): DownloadManager {
    return new DownloadManager({ modelsDir, cacheDir });
  }

  it('resolves a sha for every VISIBLE catalog repo and its assetsRepo, and skips hidden ones', async () => {
    // `hidden` entries are unpublished repos: Hugging Face answers 401 for them,
    // so dialling would spend a request to learn nothing. The assetsRepos ARE
    // probed: a sidecar-only upstream change moves nothing in the primary repo,
    // and without a probed sha the badge — and the repair job behind it — would
    // be unreachable.
    const shas = await manager().checkCatalogUpdates();
    const expected = catalogUpdateRepos();
    expect([...shas.keys()].sort()).toEqual([...expected].sort());
    expect(shas.get(REPO)).toBe(hub.sha);
    for (const entry of MODEL_CATALOG.filter((e) => !e.hidden && e.assetsRepo !== undefined)) {
      expect(shas.has(entry.assetsRepo!)).toBe(true);
      expect(hub.modelInfoRepos).toContain(entry.assetsRepo);
    }
    for (const entry of MODEL_CATALOG.filter((e) => e.hidden)) {
      expect(hub.modelInfoRepos).not.toContain(catalogRepo(entry));
      if (entry.assetsRepo !== undefined) expect(hub.modelInfoRepos).not.toContain(entry.assetsRepo);
    }
  });

  it('maps an unreachable repo to null rather than rejecting', async () => {
    // Offline must never break the Models page. `null` reads as "no badge",
    // never as "up to date".
    hub.modelInfoError = 'getaddrinfo ENOTFOUND huggingface.co';
    const shas = await manager().checkCatalogUpdates();
    expect(shas.size).toBeGreaterThan(0);
    expect([...shas.values()].every((sha) => sha === null)).toBe(true);
  });

  it('reuses a resolved sweep instead of re-dialling on every mount', async () => {
    // `useJson` refetches on every mount and every reconnect, and nothing polls
    // to smooth the rate, so the cache is what keeps this off the network.
    const m = manager();
    await m.checkCatalogUpdates(1000);
    const afterFirst = hub.modelInfoRepos.length;
    expect(afterFirst).toBeGreaterThan(0);
    await m.checkCatalogUpdates(1000 + 60_000);
    expect(hub.modelInfoRepos).toHaveLength(afterFirst);
  });

  it('retries an all-failed sweep far sooner than a good one', async () => {
    // A machine that regains its network must not stay stuck reporting nothing
    // for the full success TTL.
    hub.modelInfoError = 'offline';
    const m = manager();
    await m.checkCatalogUpdates(1000);
    const afterFirst = hub.modelInfoRepos.length;
    hub.modelInfoError = null;
    await m.checkCatalogUpdates(1000 + 61_000);
    expect(hub.modelInfoRepos.length).toBeGreaterThan(afterFirst);
    expect((await m.checkCatalogUpdates(1000 + 61_000)).get(REPO)).toBe(hub.sha);
  });

  it('degrades a STALLED probe to null instead of hanging the route', async () => {
    // The failure mode this guards is not hypothetical: an HF socket was seen
    // here sitting in CLOSE_WAIT for hours at zero CPU. Unbounded, the
    // `Promise.all` never settles, the route never answers, and the RPC client
    // declares the whole runtime unresponsive and restarts it — killing any
    // download in flight.
    hub.modelInfoUsesFetch = true;
    const hang: typeof fetch = (_input, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new Error('aborted')));
      });
    const m = new DownloadManager({ modelsDir, cacheDir, fetchImpl: hang, probeTimeoutMs: 50 });
    const shas = await m.checkCatalogUpdates();
    expect(shas.size).toBeGreaterThan(0);
    expect([...shas.values()].every((sha) => sha === null)).toBe(true);
  });

  it('keeps a job-resolved revision that landed while a sweep was in flight', async () => {
    // The sweep replaces the WHOLE cache map when it commits. A job that
    // resolved and installed a newer sha mid-sweep would have its write-through
    // clobbered by the older value the sweep captured, so the card would
    // re-offer an update for the revision just installed.
    let releaseSweep: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseSweep = resolve;
    });
    hub.modelInfoUsesFetch = true;
    const inner = makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 });
    let parked = false;
    const slow: typeof fetch = async (input, init) => {
      // Park the sweep's FIRST request only; everything the job does afterwards
      // proceeds normally, so this is the real write-through path, not a stub.
      if (!parked) {
        parked = true;
        await gate;
      }
      return inner(input, init);
    };
    const m = new DownloadManager({ modelsDir, cacheDir, fetchImpl: slow });
    hub.sha = SHA_OLD;
    const sweep = m.checkCatalogUpdates(1000);
    await waitFor(() => parked);

    // With that sweep parked mid-flight, a real job resolves and installs NEW.
    hub.sha = SHA_NEW;
    const events: DownloadEvent[] = [];
    const id = m.start(REPO);
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done'));

    releaseSweep!();
    const shas = await sweep;
    // The sweep captured SHA_OLD for this repo; the job's newer read must win.
    expect(shas.get(REPO)).toBe(SHA_NEW);
  });

  it('keeps a sweep read that was ISSUED after the job asked for the same repo', async () => {
    // The overlay above assumes the job asked LAST. Reversed — the job's read
    // goes out, this sweep starts and asks the same repo, and the job's slower
    // answer still lands first — the sweep holds the NEWER sha, and letting the
    // job's older one overwrite it would bury a genuine update for the whole
    // success TTL, which the post-download refresh reads straight back.
    let releaseJob: (() => void) | undefined;
    const jobGate = new Promise<void>((resolve) => {
      releaseJob = resolve;
    });
    let releaseSweep: (() => void) | undefined;
    const sweepGate = new Promise<void>((resolve) => {
      releaseSweep = resolve;
    });
    hub.modelInfoUsesFetch = true;
    const inner = makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 });
    let jobAsked = false;
    let sweepAsked = 0;
    const gated: typeof fetch = async (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      // Only the sha reads for THIS repo are parked — the job's file fetches and
      // its assetsRepo sha read run unimpeded, so this is the real write-through
      // path, not a stub. (The sweep probes the whole visible catalog; parking
      // every probe would also park the assets repo's and deadlock the job.)
      if (url.endsWith(`/api/models/${REPO}`)) {
        if (!jobAsked) {
          jobAsked = true;
          await jobGate;
        } else {
          sweepAsked += 1;
          await sweepGate;
        }
      }
      return inner(input, init);
    };
    const m = new DownloadManager({ modelsDir, cacheDir, fetchImpl: gated });

    // The job asks first, so its answer is the sha upstream held BEFORE it moved.
    hub.sha = SHA_OLD;
    const events: DownloadEvent[] = [];
    const id = m.start(REPO);
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => jobAsked);

    // Upstream advances, and only THEN does the sweep ask.
    hub.sha = SHA_NEW;
    const sweep = m.checkCatalogUpdates(1000);
    await waitFor(() => sweepAsked === 1);

    // The job's older answer lands first and writes itself through.
    releaseJob!();
    await waitFor(() => events.some((event) => event.type === 'done'));

    releaseSweep!();
    const shas = await sweep;
    // The job pinned the older revision, so the sweep's read is a real update.
    expect(hub.revisions).toContain(SHA_OLD);
    expect(shas.get(REPO)).toBe(SHA_NEW);
  });

  it('refuses a job write-through into a cache a LATER read already filled', async () => {
    // The other half of the same inversion. Here the sweep commits FIRST, so the
    // job's write-through lands on an already-published map rather than being
    // folded in by the overlay — a path the overlay guard never sees. The job
    // still asked first, so its answer is the older observation, and letting it
    // overwrite the committed newer sha buries the update for the success TTL.
    let releaseJob: (() => void) | undefined;
    const jobGate = new Promise<void>((resolve) => {
      releaseJob = resolve;
    });
    hub.modelInfoUsesFetch = true;
    const inner = makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 });
    let jobAsked = false;
    const gated: typeof fetch = async (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      if (url.includes('/api/models/') && !jobAsked) {
        jobAsked = true;
        await jobGate;
      }
      return inner(input, init);
    };
    const m = new DownloadManager({ modelsDir, cacheDir, fetchImpl: gated });

    hub.sha = SHA_OLD;
    const events: DownloadEvent[] = [];
    const id = m.start(REPO);
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => jobAsked);

    // Upstream advances, and a whole sweep asks, answers and COMMITS while the
    // job's own read is still out.
    hub.sha = SHA_NEW;
    expect((await m.checkCatalogUpdates(1000)).get(REPO)).toBe(SHA_NEW);

    releaseJob!();
    await waitFor(() => events.some((event) => event.type === 'done'));

    // Inside the success TTL, so this reads the committed map back — the same
    // map the post-download refresh gets, and the one the card compares against.
    expect((await m.checkCatalogUpdates(1000 + 60_000)).get(REPO)).toBe(SHA_NEW);
  });

  it('serves one sweep to every check that overlaps it', async () => {
    // Nothing serialises these calls: the RPC host fires each frame off with
    // `void Promise.resolve().then(...)`, and `useJson`'s `reload()` starts a
    // second request without cancelling the first — which the Models page does
    // on mount, on reconcile, and at every job settle. Two independent sweeps
    // both commit unconditionally, so the slower one lands last and pins the
    // OLDER sha it captured for the full six-hour TTL, with no job in the
    // interleaving for `jobResolvedShas` to repair it from. Each overlap also
    // pays a second full fan-out, the very cost the cache exists to avoid.
    const visible = catalogUpdateRepos().length;
    let releaseSweep: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseSweep = resolve;
    });
    hub.modelInfoUsesFetch = true;
    const inner = makeFetchImpl({});
    let parked = 0;
    const slow: typeof fetch = async (input, init) => {
      // Park exactly the first sweep's probes. A second sweep, if one is ever
      // started, runs unimpeded and commits FIRST — the losing interleaving.
      if (parked < visible) {
        parked += 1;
        await gate;
      }
      return inner(input, init);
    };
    const m = new DownloadManager({ modelsDir, cacheDir, fetchImpl: slow });
    hub.sha = SHA_OLD;
    const first = m.checkCatalogUpdates(1000);
    await waitFor(() => parked === visible);

    // Upstream advances while that sweep is parked, then a second check arrives.
    hub.sha = SHA_NEW;
    const second = m.checkCatalogUpdates(1000);
    releaseSweep!();
    const [firstShas, secondShas] = await Promise.all([first, second]);

    // ONE fan-out for both callers, not two.
    expect(hub.modelInfoRepos).toHaveLength(visible);
    expect(secondShas).toBe(firstShas);
    expect(firstShas.get(REPO)).toBe(SHA_OLD);
    // Well inside the 6h TTL: the cache still holds that same map, so no late
    // commit rolled it back.
    expect(await m.checkCatalogUpdates(1000 + 60_000)).toBe(firstShas);
  });

  it('takes the short TTL when ANY repo failed, not only when all did', async () => {
    // One cache entry covers the whole sweep, so a single transient failure
    // beside successes would otherwise pin that repo's `null` for six hours —
    // no mount or reconnect in that window could surface its update.
    hub.modelInfoFailRepos = [REPO_OTHER];
    const m = manager();
    const first = await m.checkCatalogUpdates(1000);
    expect(first.get(REPO)).toBe(hub.sha);
    expect(first.get(REPO_OTHER)).toBeNull();
    const afterFirst = hub.modelInfoRepos.length;

    // Inside the SUCCESS TTL but past the negative one: must re-dial.
    hub.modelInfoFailRepos = [];
    const second = await m.checkCatalogUpdates(1000 + 61_000);
    expect(hub.modelInfoRepos.length).toBeGreaterThan(afterFirst);
    expect(second.get(REPO_OTHER)).toBe(hub.sha);
  });

  it("folds a job's freshly resolved revision into the cache", async () => {
    // Upstream can advance between a sweep and the user's click. `processJob`
    // then installs the NEW sha while the cache still holds the old one, so the
    // card re-offers an update for the revision just installed and every click
    // completes as a no-op. The job's own resolve IS a fresh upstream read, so
    // it is authoritative for that repo.
    hub.sha = SHA_OLD;
    const m = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    expect((await m.checkCatalogUpdates(1000)).get(REPO)).toBe(SHA_OLD);

    hub.sha = SHA_NEW;
    const events: DownloadEvent[] = [];
    const id = m.start(REPO);
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done'));

    // Well inside the 6h TTL, so this is the CACHE answering — and it must have
    // been corrected by the job rather than still serving the pre-install sha.
    const shas = await m.checkCatalogUpdates(1000 + 60_000);
    expect(shas.get(REPO)).toBe(SHA_NEW);
    // Untouched repos keep the cached sweep; the job speaks only for its own.
    expect(shas.get(REPO_OTHER)).toBe(SHA_OLD);
  });
});

describe('DownloadManager', () => {
  it('emits start → per-file progress → done and atomically publishes into modelsDir/<slug>', async () => {
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      .filter((event) => event.type === 'progress' && event.file === WEIGHT)
      .map((event) => (event.type === 'progress' ? event.receivedBytes : 0));
    expect(modelProgress.length).toBeGreaterThan(1);
    for (let i = 1; i < modelProgress.length; i++) {
      expect(modelProgress[i]).toBeGreaterThanOrEqual(modelProgress[i - 1]);
    }
    expect(modelProgress[modelProgress.length - 1]).toBe(300);

    const done = events.find((event) => event.type === 'done');
    expect(done).toMatchObject({ type: 'done', outputDir: finalDir() });

    expect(existsSync(join(finalDir(), 'config.json'))).toBe(true);
    expect(readFileSync(join(finalDir(), WEIGHT)).length).toBe(300);
    // Completion marker is present in the published dir and no job staging remains.
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(jobStagingDirs()).toEqual([]);

    const job = manager.jobs().find((j) => j.id === id)!;
    expect(job.state).toBe('done');
    expect(job.totalBytes).toBe(312);
    expect(job.receivedBytes).toBe(312);
  });

  it.each([true, false])('publishes only a complete DFlash2 companion (valid architecture: %s)', async (valid) => {
    const config = JSON.stringify({
      model_type: 'qwen3',
      architectures: [valid ? 'DFlash2DraftModel' : 'Qwen3ForCausalLM'],
    });
    // The draft repo is a catalog DOWNLOAD repo but not a catalog ENTRY, so it
    // carries no globs and no assetsRepo: the no-glob default filter applies and
    // the companion stays a safetensors repo (there is nothing to match a GGUF glob).
    hub.manifest = [
      { type: 'file', path: 'config.json', size: Buffer.byteLength(config) },
      { type: 'file', path: 'model.safetensors', size: 16 },
    ];
    const weightsFetch = makeFetchImpl({ 'model.safetensors': 16 });
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: (input, init) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
        return url.endsWith('/config.json') ? Promise.resolve(new Response(config)) : weightsFetch(input, init);
      },
    });
    const id = manager.start(QWEN38_DFLASH2.hfRepo);
    await waitFor(() => manager.jobs().some((job) => job.id === id && ['done', 'error'].includes(job.state)));
    expect(manager.jobs().find((job) => job.id === id)?.state).toBe(valid ? 'done' : 'error');
    const draftPath = join(modelsDir, 'qwen3.8-27b-dflash2');
    expect(existsSync(draftPath)).toBe(valid);
    expect(findDFlash2Draft(join(modelsDir, 'qwen3.8-27b-mxfp4-mlx'), 'qwen3_5')).toBe(valid ? draftPath : undefined);
    const target = catalogWithState(modelsDir).find((item) => item.label === 'Qwen3.8-27B')!;
    expect(target.present).toBe(false);
    expect(target.draft?.present).toBe(valid);
    expect(target.draft?.installed).toBe(valid);
    await manager.shutdown();
  });

  it('refuses a hidden catalog entry up front instead of failing mid-download', async () => {
    // A `hidden` entry is an UNPUBLISHED repo: it stays in MODEL_CATALOG so a
    // locally converted checkpoint at the canonical slug is still recognized as
    // Installed, but Hugging Face answers 401 for it. Membership alone is
    // therefore not a sufficient allowlist. No UI offers one (the Models page
    // filters `!item.hidden`), so the reachable route is a direct API POST.
    const hidden = MODEL_CATALOG.find((entry) => entry.hidden);
    expect(hidden, 'catalog must still carry a hidden entry for this gate to mean anything').toBeDefined();

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });

    // Through the resolver, mirroring the allowlist. Hidden entries carry no
    // CUDA build today, so this is the same string — but a raw `hfRepo` here
    // would silently start testing the wrong thing the day one does.
    expect(() => manager.start(catalogRepo(hidden!))).toThrow(/not in the model catalog/);
    // Rejected BEFORE any job is allocated — otherwise the SPA renders a job
    // that marches to a 401 instead of an immediate, actionable error.
    expect(manager.jobs()).toEqual([]);

    // Non-hidden entries are unaffected.
    expect(() => manager.start(REPO)).not.toThrow();
    // That last `start` allocated a REAL job whose `drain` runs detached. Without
    // this the test returns while `processJob` is still writing under `modelsDir`,
    // and the `afterEach` `rmSync` races it to an intermittent ENOTEMPTY.
    await manager.shutdown();
    // Deterministic proof the wait happened: an unawaited job reads `running`.
    expect(manager.jobs().map((job) => job.state)).toEqual(['cancelled']);
  });

  it('pins one resolved commit sha per repo and threads it into that repo list/download calls', async () => {
    hub.sha = SHA_A;
    // The base-model repo the sidecars come from has its OWN head. Pinning the
    // sidecars to the primary repo's sha would mix two snapshots into one install.
    hub.shaByRepo[ASSETS_REPO] = SHA_OLD;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The entry's list + both downloadFileToCacheDir calls all saw exactly one
    // revision — its own resolved sha, never the assets repo's.
    expect(hub.revisionsByRepo[REPO]!.length).toBeGreaterThan(1);
    expect(new Set(hub.revisionsByRepo[REPO])).toEqual(new Set([SHA_A]));
    // …and the sidecar listing was pinned to the ASSETS repo's resolved sha.
    expect(hub.revisionsByRepo[ASSETS_REPO]).toEqual([SHA_OLD]);

    // The published marker records the PRIMARY repo's pinned revision.
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      revision: string;
      repo: string;
      files: string[];
    };
    expect(marker.revision).toBe(SHA_A);
    expect(marker.repo).toBe(REPO);
    expect(marker.files).toEqual(expect.arrayContaining(['config.json', WEIGHT]));
  });

  it('refuses to pin a missing or mutable sha and pins the 40-hex commit on the normal path', async () => {
    // Missing sha → fail closed (never silently pin a mutable "main").
    hub.sha = '';
    {
      const manager = new DownloadManager({
        modelsDir,
        cacheDir,
        fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
        fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
        fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    hub.failOn = [WEIGHT];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
  // reports not-installed) rather than publish a hollow "installed" dir. Which
  // half is one-sided depends on the entry: a `assetsRepo` can only ever supply
  // the CONFIG side (see the payload-gate test below), so a config-only manifest
  // still errors on the GGUF entry.
  it('errors on a config-only manifest instead of publishing a hollow dir', async () => {
    hub.manifest = [{ type: 'file', path: 'config.json', size: 12 }];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
  });

  it('errors on a weights-only manifest when the repo has no assetsRepo to supply the config', async () => {
    // The DFlash2 companion is a catalog DOWNLOAD repo (so `start` admits it) with
    // neither globs nor an `assetsRepo` — nothing can supply the missing
    // `config.json`, so the weights-only manifest stays one-sided.
    hub.manifest = [{ type: 'file', path: 'model.safetensors', size: 300 }];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'model.safetensors': 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(QWEN38_DFLASH2.hfRepo);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    const draftDir = join(modelsDir, 'qwen3.8-27b-dflash2');
    expect(existsSync(draftDir)).toBe(false);
    expect(existsSync(join(draftDir, DOWNLOAD_COMPLETE_MARKER))).toBe(false);
    expect(jobStagingDirs()).toEqual([]);
  });

  it('marks installed only with the completion marker, never bare directory existence', async () => {
    // A bare dir with config.json + a weight but no marker (a legacy/partial download).
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12));
    writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300));
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(false);

    // A marker listing both a config and a weight, all present → installed.
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: REPO,
        revision: hub.sha,
        files: ['config.json', WEIGHT],
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
    hub.failOn = [WEIGHT];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    expect(hub.downloaded).toContain(WEIGHT);
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('downloads only the glob-matched variants of a multi-variant GGUF repo', async () => {
    // A real Unsloth repo ships dozens of quantization variants side by side. The
    // entry's globs (`*UD-Q4_K_XL*`, `config.json`) must select exactly the
    // one build the wizard installs — never the whole multi-hundred-GB repo.
    const OTHER_QUANT = 'Qwen3.8-27B-UD-Q8_K_XL.gguf';
    const NON_UD = 'Qwen3.8-27B-Q4_K_M.gguf';
    const MTP = 'MTP/mtp-Qwen3.8-27B-Q4_0.gguf';
    const MMPROJ = 'mmproj-BF16.gguf';
    const README = 'README.md';
    hub.manifest = [
      { type: 'file', path: WEIGHT, size: 300 },
      { type: 'file', path: OTHER_QUANT, size: 222 },
      { type: 'file', path: NON_UD, size: 111 },
      { type: 'file', path: MTP, size: 33 },
      { type: 'file', path: MMPROJ, size: 44 },
      { type: 'file', path: README, size: 55 },
    ];
    // The assets repo is a genuinely different repo, so it gets its own listing.
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'config.json', size: 12 }];

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, [MTP]: 33, 'config.json': 12 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done'));

    // Exactly the glob-matched primary file moved (the MTP artifact is NOT
    // globbed: the UD file carries its MTP layer inline, nothing pairs a GGUF
    // MTP sidecar). The `start` frame carries the FULL total — the primary
    // weight plus the assets repo's config.json sidecar (300 + 12) — because
    // the sidecar plan is resolved before `start` so the byte bar never
    // overshoots.
    expect(events.find((event) => event.type === 'start')).toMatchObject({
      type: 'start',
      repo: REPO,
      totalBytes: 312,
      fileCount: 2,
    });
    // EVERY frame must report the aggregate (1 primary + 1 sidecar): the UI
    // reducer replaces its count with each progress event's, so a primary-only
    // count here would shrink the displayed total and then jump when the
    // sidecar starts.
    const progressCounts = events.filter((event) => event.type === 'progress').map((event) => event.fileCount);
    expect(progressCounts.length).toBeGreaterThan(0);
    expect(new Set(progressCounts)).toEqual(new Set([2]));
    expect([...hub.downloaded].sort()).toEqual([WEIGHT, 'config.json'].sort());

    // The unmatched variants were never fetched and never published…
    for (const path of [OTHER_QUANT, NON_UD, MTP, MMPROJ, README]) {
      expect(hub.downloaded, `${path} must not be fetched`).not.toContain(path);
      expect(existsSync(join(finalDir(), path)), `${path} must not be published`).toBe(false);
    }
    // …while the selected build and the core metadata landed.
    expect(existsSync(join(finalDir(), WEIGHT))).toBe(true);
    expect(existsSync(join(finalDir(), 'config.json'))).toBe(true);
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('fetches the tokenizer sidecars from the entry assetsRepo and lists them in the completion marker', async () => {
    // A GGUF quantization repo ships weights only — no tokenizer, no config.
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
      { type: 'file', path: 'tokenizer_config.json', size: 8 },
      { type: 'file', path: 'chat_template.jinja', size: 6 },
      // In the assets repo but NOT a sidecar candidate: must be left alone.
      { type: 'file', path: 'README.md', size: 55 },
      // A nested same-named file is not the base model's root-level tokenizer.
      { type: 'file', path: 'onnx/tokenizer.json', size: 40 },
    ];
    const sidecars = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'];

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({
        [WEIGHT]: 300,
        'config.json': 12,
        'tokenizer.json': 20,
        'tokenizer_config.json': 8,
        'chat_template.jinja': 6,
      }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done'));

    // Every sidecar is published AND came off the assets repo, not the primary one.
    for (const name of sidecars) {
      expect(existsSync(join(finalDir(), name)), `${name} missing from the published dir`).toBe(true);
      expect(hub.downloadedFrom, `${name} was not fetched from the assets repo`).toContain(`${ASSETS_REPO}/${name}`);
      expect(hub.downloadedFrom, `${name} was fetched from the primary repo`).not.toContain(`${REPO}/${name}`);
    }
    expect(hub.downloaded, 'a non-candidate assets-repo file was fetched').not.toContain('README.md');
    expect(hub.downloaded, 'a nested same-named file was fetched').not.toContain('onnx/tokenizer.json');

    // The marker stays uniform: the PRIMARY repo+revision, with the sidecar paths
    // folded into the file list so resume/update verification covers them too.
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      repo: string;
      revision: string;
      files: string[];
    };
    expect(marker.repo).toBe(REPO);
    expect(marker.revision).toBe(hub.sha);
    expect(marker.files).toEqual(expect.arrayContaining([WEIGHT, ...sidecars]));
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('accepts a weights-only manifest when the entry names an assetsRepo for the config half', async () => {
    // The payload gate tolerates a manifest without `config.json` ONLY because the
    // sidecar fetch supplies it; the weight requirement is not relaxed.
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'config.json', size: 12 }];

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done'));

    expect(events.some((event) => event.type === 'error')).toBe(false);
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('errors without publishing when neither the repo nor its assetsRepo provides a config.json', async () => {
    // The payload gate waives the config requirement for an assetsRepo entry,
    // but the waiver is a promise the sidecar fetch has to keep: with no
    // config.json in EITHER listing the job must fail, not publish a
    // weights-only install that renders as not-installed.
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'tokenizer.json', size: 20 }];

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'tokenizer.json': 20 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error' || event.type === 'done'));

    expect(events.some((event) => event.type === 'error')).toBe(true);
    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(existsSync(finalDir())).toBe(false);
    // `processJob` emits `error` BEFORE its `finally` awaits the staging rm, so
    // a bare assertion here races the cleanup; wait for the directory to go.
    await waitFor(() => jobStagingDirs().length === 0);
  });

  it('errors when the only matched GGUF is a companion artifact, not the target', async () => {
    // The GEMMA entry is the one whose globs select a companion by name
    // (`mmproj-BF16.gguf`, for the vision tower). A partial upstream upload (or
    // a renamed target) can leave the manifest with the projector and nothing
    // else: publishing that would certify a directory model-discovery never
    // lists — "Installed" with no loadable model. (The Qwen3.8 entry cannot
    // express this: its globs filter a companion out before the gate runs.)
    const gemma = MODEL_CATALOG.find((entry) => entry.label === 'Gemma-4-26B-A4B')!;
    const gemmaRepo = catalogRepo(gemma);
    const gemmaSlug = gemmaRepo.split('/').pop()!.toLowerCase();
    expect(gemma.globs).toContain('mmproj-BF16.gguf');

    hub.manifest = [{ type: 'file', path: 'mmproj-BF16.gguf', size: 44 }];
    hub.manifests[gemma.assetsRepo!] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'mmproj-BF16.gguf': 44, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(gemmaRepo);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error' || event.type === 'done'));

    expect(events.some((event) => event.type === 'error')).toBe(true);
    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(existsSync(join(modelsDir, gemmaSlug))).toBe(false);
    await waitFor(() => jobStagingDirs().length === 0);
  });

  it('records the verified assets revision so a vacuous assets advance cannot loop the badge', async () => {
    // An assets-repo advance that changed nothing installable (README, model
    // weights) raises the badge — discovery compares revisions. The job then
    // verifies the sidecar BYTES, finds them current, and used to return done
    // without recording that verification: the card offered the same update
    // forever. Verifying must also record what was verified.
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];
    hub.shaByRepo[ASSETS_REPO] = SHA_OLD;
    const first = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events1: DownloadEvent[] = [];
    const id1 = first.start(REPO);
    first.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'done'));

    // The base repo moves; the sidecar bytes do not.
    hub.shaByRepo[ASSETS_REPO] = SHA_NEW;
    hub.downloaded = [];
    const second = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events2: DownloadEvent[] = [];
    const id2 = second.start(REPO);
    second.subscribe(id2, (event) => events2.push(event));
    await waitFor(() => events2.some((event) => event.type === 'done'));

    // Nothing re-downloaded, and the marker now names the revision it verified.
    expect(hub.downloaded).toEqual([]);
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      assetsRepo?: string;
      assetsRevision?: string;
    };
    expect(marker.assetsRepo).toBe(ASSETS_REPO);
    expect(marker.assetsRevision).toBe(SHA_NEW);
  });

  it('honours a cancel accepted while installed sidecars are being refreshed', async () => {
    // `refreshInstalledAssets` mutates the live install (deletes, marker
    // rewrite) while the job still reads `running`, so `cancel()` accepts.
    // Without a recheck between that await and the success branch the job
    // emitted `done` after the UI reported the cancel — cancelling during the
    // marker rewrite (the hook below) lands exactly in that window.
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];
    hub.shaByRepo[ASSETS_REPO] = SHA_OLD;
    const first = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events1: DownloadEvent[] = [];
    const id1 = first.start(REPO);
    first.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'done'));

    // Same verify-clean refresh shape as the test above: the assets revision
    // moved, the bytes did not, so the refresh rewrites the marker — the hook
    // cancels inside that write, before the success branch runs.
    hub.shaByRepo[ASSETS_REPO] = SHA_NEW;
    const second = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events2: DownloadEvent[] = [];
    const id2 = second.start(REPO);
    second.subscribe(id2, (event) => events2.push(event));
    raceHook.onMarkerWrite = () => second.cancel(id2);
    await waitFor(() => events2.some((event) => event.type === 'cancelled' || event.type === 'done'));

    expect(events2.some((event) => event.type === 'done')).toBe(false);
    expect(events2.some((event) => event.type === 'cancelled')).toBe(true);
    // The refresh itself completed before the cancel won: the marker records
    // the revision it verified, so the install stays consistent rather than
    // half-rewritten.
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      assetsRevision?: string;
    };
    expect(marker.assetsRevision).toBe(SHA_NEW);
  });

  it('records the new sidecar source when the assets repo moved at the same sha', async () => {
    // An HF repo TRANSFER keeps history: the new repo's HEAD is the sha the
    // marker already records. The verified-clean path must still rewrite the
    // marker's assetsRepo — the page compares THAT name against the entry's,
    // so a stale name means an update badge no job can ever clear.
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];
    hub.shaByRepo[ASSETS_REPO] = SHA_OLD;
    const first = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events1: DownloadEvent[] = [];
    const id1 = first.start(REPO);
    first.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'done'));

    // Same bytes, same pinned sha — but the marker names the repo the entry
    // used to point at (the transfer's old name).
    const markerPath = join(finalDir(), DOWNLOAD_COMPLETE_MARKER);
    const marker = JSON.parse(readFileSync(markerPath, 'utf-8')) as { assetsRepo?: string };
    marker.assetsRepo = 'base/old-model';
    writeFileSync(markerPath, JSON.stringify(marker));
    hub.downloaded = [];

    const second = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events2: DownloadEvent[] = [];
    const id2 = second.start(REPO);
    second.subscribe(id2, (event) => events2.push(event));
    await waitFor(() => events2.some((event) => event.type === 'done'));

    // Everything verified clean — nothing re-downloaded — but the marker now
    // names the CURRENT source, so the badge clears.
    expect(hub.downloaded).toEqual([]);
    const next = JSON.parse(readFileSync(markerPath, 'utf-8')) as { assetsRepo?: string };
    expect(next.assetsRepo).toBe(ASSETS_REPO);
  });

  it('drops and deletes a sidecar the assets repo no longer supplies', async () => {
    // A repo that deleted a tokenizer file must not leave it installed (and
    // listed) forever: the verified-done path prunes what the assets listing
    // no longer carries, marker first, then the file.
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];
    hub.shaByRepo[ASSETS_REPO] = SHA_OLD;
    const first = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events1: DownloadEvent[] = [];
    const id1 = first.start(REPO);
    first.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'done'));
    expect(existsSync(join(finalDir(), 'tokenizer.json'))).toBe(true);

    // Upstream drops the tokenizer from its listing entirely.
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'config.json', size: 12 }];
    hub.shaByRepo[ASSETS_REPO] = SHA_NEW;
    hub.downloaded = [];
    const second = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12 }),
    });
    const events2: DownloadEvent[] = [];
    const id2 = second.start(REPO);
    second.subscribe(id2, (event) => events2.push(event));
    await waitFor(() => events2.some((event) => event.type === 'done'));

    expect(existsSync(join(finalDir(), 'tokenizer.json'))).toBe(false);
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      files: string[];
      assetsRevision?: string;
    };
    expect(marker.files).not.toContain('tokenizer.json');
    expect(marker.files).toContain('config.json');
    expect(marker.assetsRevision).toBe(SHA_NEW);
  });

  // macOS-only: the immutable flag is the one way to keep a file regular (so
  // the install still reads as installed and the refresh path really runs)
  // while making `rm` refuse it.
  it.skipIf(process.platform !== 'darwin')(
    'leaves the marker untouched when a stale sidecar cannot be deleted',
    async () => {
      // The stale set is derived FROM the marker, so dropping an entry before its
      // file is actually gone would make an interrupted refresh unrecoverable:
      // the file stays, nothing derives it again, and it is reported current
      // forever. Deletion must fail with the marker still listing it, so the
      // next run retries.
      hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
      hub.manifests[ASSETS_REPO] = [
        { type: 'file', path: 'config.json', size: 12 },
        { type: 'file', path: 'tokenizer.json', size: 20 },
      ];
      hub.shaByRepo[ASSETS_REPO] = SHA_OLD;
      const first = new DownloadManager({
        modelsDir,
        cacheDir,
        fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
      });
      const idOne = first.start(REPO);
      const eventsOne: DownloadEvent[] = [];
      first.subscribe(idOne, (event) => eventsOne.push(event));
      await waitFor(() => eventsOne.some((event) => event.type === 'done'));

      // Upstream drops tokenizer.json; locally the file becomes immutable.
      hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'config.json', size: 12 }];
      hub.shaByRepo[ASSETS_REPO] = SHA_NEW;
      const tokenizerPath = join(finalDir(), 'tokenizer.json');
      execFileSync('/usr/bin/chflags', ['uchg', tokenizerPath]);
      try {
        const second = new DownloadManager({
          modelsDir,
          cacheDir,
          fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12 }),
        });
        const idTwo = second.start(REPO);
        const eventsTwo: DownloadEvent[] = [];
        second.subscribe(idTwo, (event) => eventsTwo.push(event));
        await waitFor(() => eventsTwo.some((event) => event.type === 'error' || event.type === 'done'));

        // The install still reads as installed (the file IS a regular file), so
        // this really exercised the refresh path — not the full re-stage one.
        expect(isModelInstalled(finalDir())).toBe(true);
        expect(eventsTwo.some((event) => event.type === 'error')).toBe(true);
        const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
          files: string[];
          assetsRevision?: string;
        };
        // Untouched: still listed, still pinned to the old revision — the next run
        // derives the same stale set and tries again.
        expect(marker.files).toContain('tokenizer.json');
        expect(marker.assetsRevision).toBe(SHA_OLD);
      } finally {
        execFileSync('/usr/bin/chflags', ['nouchg', tokenizerPath]);
      }
    },
  );

  it('refuses to prune away the last config, leaving the install intact', async () => {
    // The primary GGUF repo ships no config.json (AgentWorld's doesn't): the
    // sidecar is the ONLY one. Upstream dropping it must not delete the live
    // install into an unloadable shape — the refresh fails instead, marker and
    // files untouched.
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];
    hub.shaByRepo[ASSETS_REPO] = SHA_OLD;
    const first = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events1: DownloadEvent[] = [];
    const id1 = first.start(REPO);
    first.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'done'));
    expect(existsSync(join(finalDir(), 'config.json'))).toBe(true);

    // Upstream renames its config away; only tokenizer.json remains supplied.
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'tokenizer.json', size: 20 }];
    hub.shaByRepo[ASSETS_REPO] = SHA_NEW;
    hub.downloaded = [];
    const second = new DownloadManager({ modelsDir, cacheDir, fetchImpl: makeFetchImpl({ [WEIGHT]: 300 }) });
    const events2: DownloadEvent[] = [];
    const id2 = second.start(REPO);
    second.subscribe(id2, (event) => events2.push(event));
    await waitFor(() => events2.some((event) => event.type === 'error' || event.type === 'done'));

    expect(events2.some((event) => event.type === 'error')).toBe(true);
    expect(events2.some((event) => event.type === 'done')).toBe(false);
    // Nothing was mutated: the installation still loads.
    expect(existsSync(join(finalDir(), 'config.json'))).toBe(true);
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      files: string[];
      assetsRevision?: string;
    };
    expect(marker.files).toContain('config.json');
    expect(marker.assetsRevision).toBe(SHA_OLD);
    // And no temp marker residue from the atomic writer.
    expect(
      readdirSync(finalDir()).filter((name) => name.includes(DOWNLOAD_COMPLETE_MARKER) && name.endsWith('.tmp')),
    ).toEqual([]);
  });

  it('does not re-download verified sidecars on a second job over the same revision', async () => {
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const first: DownloadEvent[] = [];
    const id1 = manager.start(REPO);
    manager.subscribe(id1, (event) => first.push(event));
    await waitFor(() => first.some((event) => event.type === 'done'));
    expect([...hub.downloaded].sort()).toEqual([WEIGHT, 'config.json', 'tokenizer.json'].sort());

    // Downgrade the marker to a `partial` (CLI-style) one, so the install-skip gate
    // does NOT short-circuit and the job really re-walks the manifest and re-publishes.
    const markerPath = join(finalDir(), DOWNLOAD_COMPLETE_MARKER);
    const marker = JSON.parse(readFileSync(markerPath, 'utf-8')) as Record<string, unknown>;
    writeFileSync(markerPath, JSON.stringify({ ...marker, scope: 'partial' }));
    hub.downloaded = [];
    hub.downloadedFrom = [];

    const second: DownloadEvent[] = [];
    const id2 = manager.start(REPO);
    manager.subscribe(id2, (event) => second.push(event));
    await waitFor(() => second.some((event) => event.type === 'done'));

    // Same revision, so every file — sidecars included — is a shared-HF-cache hit.
    expect(hub.downloaded).toEqual([]);
    expect(existsSync(join(finalDir(), 'tokenizer.json'))).toBe(true);
    const republished = JSON.parse(readFileSync(markerPath, 'utf-8')) as { scope?: string; files: string[] };
    expect(republished.scope).toBe('full');
    expect(republished.files).toEqual(expect.arrayContaining([WEIGHT, 'config.json', 'tokenizer.json']));
  });

  it('re-fetches a sidecar that changed upstream while the primary revision stayed put', async () => {
    // The completion marker pins only the PRIMARY repo and the update sweep
    // probes primary repos only, so a tokenizer fix in the base model (the
    // assets repo moves on its own revision) raises no badge — but a job that
    // runs must still repair it instead of short-circuiting on the marker.
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];
    hub.shaByRepo[ASSETS_REPO] = SHA_OLD;
    const first = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events1: DownloadEvent[] = [];
    const id1 = first.start(REPO);
    first.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'done'));
    expect(readFileSync(join(finalDir(), 'tokenizer.json')).length).toBe(20);

    // The base model re-uploads the tokenizer at a NEW revision; the primary
    // repo (and therefore the marker) is untouched.
    hub.manifests[ASSETS_REPO] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 30 },
    ];
    hub.shaByRepo[ASSETS_REPO] = SHA_NEW;
    hub.downloaded = [];
    const second = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12, 'tokenizer.json': 30 }),
    });
    const events2: DownloadEvent[] = [];
    const id2 = second.start(REPO);
    second.subscribe(id2, (event) => events2.push(event));
    await waitFor(() => events2.some((event) => event.type === 'done'));

    expect(hub.downloaded).toContain('tokenizer.json');
    expect(readFileSync(join(finalDir(), 'tokenizer.json')).length).toBe(30);
  });

  // Finding #4: the install-skip gate must match the marker's repo AND revision,
  // not merely "a complete same-slug install exists". A same-slug install pinned to
  // the SAME resolved revision short-circuits (idempotent, no re-fetch).
  it('short-circuits to done when the installed marker matches the resolved repo+revision', async () => {
    // A complete owned install pinned to the exact revision resolveRevision returns.
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', WEIGHT], completedAt: 'x' }),
    );

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: REPO,
        revision: hub.sha,
        files: ['config.json', WEIGHT],
        scope: 'partial',
        completedAt: 'x',
      }),
    );

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    expect(hub.downloaded).toEqual(expect.arrayContaining(['config.json', WEIGHT]));
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
    writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: SHA_OLD, files: ['config.json', WEIGHT], completedAt: 'x' }),
    );
    hub.sha = SHA_NEW; // the resolved revision differs from the installed SHA_OLD

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // It genuinely downloaded the resolved revision (never falsely reported done)…
    expect(hub.downloaded).toEqual(expect.arrayContaining(['config.json', WEIGHT]));
    // …and the owned-swap replaced the old 0xAB content with the fresh zero-bytes.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300));
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
        path: WEIGHT,
        size: 300,
        lfs: { oid: zerosSha256(300), size: 300, pointerSize: 100 },
      },
    ];
    // A pre-existing corrupt cache blob (0xFF instead of the 0x00 the download writes).
    seedCorruptCache(WEIGHT, hub.sha, 300);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The corrupt cache entry failed sha256 → was invalidated → really re-fetched →
    // matched → published (without invalidation the retry would recopy the same blob).
    expect(hub.downloaded).toContain(WEIGHT);
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300));
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('content-verifies a XET-backed weight, which carries all three digest fields at once', async () => {
    // The shape a real Xet repo actually returns — checked against the live API for
    // `Brooooooklyn/Qwen3.6-27B-NVFP4-mlx`, where every weight (GGUF weights on
    // Xet-backed repos included) has `oid` AND `lfs.oid` AND `xetHash` together:
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
        path: WEIGHT,
        size: 300,
        oid: 'a'.repeat(40),
        lfs: { oid: zerosSha256(300), size: 300, pointerSize: 135 },
        xetHash: 'b'.repeat(64),
      },
    ];
    // Corrupt at EXACTLY the manifest size — the case size alone cannot catch.
    seedCorruptCache(WEIGHT, hub.sha, 300);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    expect(hub.downloaded, 'a same-length corrupt Xet blob was accepted without a re-fetch').toContain(WEIGHT);
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300));
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('re-fetches when a corrupt cached blob fails the git-blob oid check', async () => {
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12, oid: zerosGitOid(12) },
      { type: 'file', path: WEIGHT, size: 300 },
    ];
    seedCorruptCache('config.json', hub.sha, 12);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
        path: WEIGHT,
        size: 300,
        lfs: { oid: zerosSha256(299), size: 300, pointerSize: 100 },
      },
    ];
    const { pointer, blob } = seedCorruptCache(WEIGHT, hub.sha, 300);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    expect(hub.downloaded.filter((p) => p === WEIGHT).length).toBeGreaterThan(0);
    expect(existsSync(finalDir())).toBe(false);
  });

  it('errors without publishing when downloaded bytes never match the manifest hash', async () => {
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      {
        type: 'file',
        path: WEIGHT,
        size: 300,
        lfs: { oid: zerosSha256(299), size: 300, pointerSize: 100 },
      },
    ];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    // The mismatching file was re-fetched (bounded) before the job gave up.
    expect(hub.downloaded.filter((p) => p === WEIGHT).length).toBeGreaterThan(1);
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
    writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300, 0xab));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    writeFileSync(join(external, WEIGHT), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(external, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: SHA_OLD, files: ['config.json', WEIGHT], completedAt: 'x' }),
    );
    symlinkSync(external, finalDir());

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    expect(readFileSync(join(external, WEIGHT))).toEqual(Buffer.alloc(300, 0xab));

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
    writeFileSync(join(external, WEIGHT), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(external, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: SHA_OLD, files: ['config.json', WEIGHT], completedAt: 'x' }),
    );
    // A recoverable dead-owner rollback backup (pid 999999999 is DEAD).
    const backup = join(stagingRoot(), `${SLUG}.backup-999999999.44444444-4444-4444-4444-444444444444`);
    mkdirSync(backup, { recursive: true });
    writeFileSync(join(backup, 'config.json'), Buffer.alloc(12, 0x5a));
    writeFileSync(join(backup, WEIGHT), Buffer.alloc(300, 0x5a));
    writeFileSync(
      join(backup, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: SHA_OLD, files: ['config.json', WEIGHT], completedAt: 'x' }),
    );
    symlinkSync(external, finalDir());

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    expect(readFileSync(join(backup, WEIGHT))).toEqual(Buffer.alloc(300, 0x5a));
    expect(backupDirs()).toEqual([`${SLUG}.backup-999999999.44444444-4444-4444-4444-444444444444`]);

    // The external symlink target is byte-for-byte intact — nothing written through the link.
    expect(readFileSync(join(external, 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(external, WEIGHT))).toEqual(Buffer.alloc(300, 0xab));

    rmSync(external, { recursive: true, force: true });
  });

  it('refuses to overwrite an unowned existing dir and leaves it intact', async () => {
    // A valid checkpoint placed manually / by `mlx download` — no completion
    // marker, so the downloader does not own it.
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300, 0xab));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    // The manual files are byte-for-byte intact and no marker was written into them.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300, 0xab));
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300, 0xcd));
    };

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(REPO);
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    // The raced-in unowned dir is byte-for-byte intact; no marker written into it.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xcd));
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300, 0xcd));
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
        files: ['config.json', WEIGHT],
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    writeFileSync(join(backup, WEIGHT), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(backup, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', WEIGHT], completedAt: 'x' }),
    );
    expect(existsSync(finalDir())).toBe(false);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', WEIGHT], completedAt: 'x' }),
    );
    // A leaked backup from a prior successful swap that never got cleaned; its owner
    // pid (999999999) is DEAD, so the sweep reaps it.
    const backup = join(stagingRoot(), `${SLUG}.backup-999999999.22222222-2222-2222-2222-222222222222`);
    mkdirSync(backup, { recursive: true });
    writeFileSync(join(backup, 'config.json'), Buffer.alloc(12, 0x5));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    writeFileSync(join(liveBackup, WEIGHT), Buffer.alloc(300, 0xab));
    expect(existsSync(finalDir())).toBe(false);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', WEIGHT], completedAt: 'x' }),
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
        path: WEIGHT,
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
    const pointerInExt = join(extRevDir, WEIGHT);
    symlinkSync(foreignBlob, pointerInExt);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    expect(readdirSync(extRevDir)).toContain(WEIGHT);
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The published set is exactly the manifest — no stale orphan.
    expect(existsSync(join(finalDir(), 'stale.safetensors'))).toBe(false);
    expect(existsSync(join(finalDir(), 'config.json'))).toBe(true);
    expect(existsSync(join(finalDir(), WEIGHT))).toBe(true);
    expect(catalogWithState(modelsDir).find((e) => e.slug === SLUG)!.installed).toBe(true);
  });

  it('does not reuse another revision staged files across a revision change', async () => {
    // An interrupted OLDER revision left a single-file weight in its own (legacy
    // shared) staging scope.
    mkdirSync(legacyStagingDir(SHA_OLD), { recursive: true });
    writeFileSync(join(legacyStagingDir(SHA_OLD), WEIGHT), Buffer.alloc(999, 0x9));

    // The current revision is SHARDED — two shard files, not the single-file weight.
    hub.sha = SHA_NEW;
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf', size: 100 },
      { type: 'file', path: 'Qwen3.8-27B-UD-Q4_K_XL-00002-of-00002.gguf', size: 100 },
    ];

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({
        'config.json': 12,
        'Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf': 100,
        'Qwen3.8-27B-UD-Q4_K_XL-00002-of-00002.gguf': 100,
      }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The stale single-file weight never enters the new sharded install.
    expect(existsSync(join(finalDir(), WEIGHT))).toBe(false);
    expect(existsSync(join(finalDir(), 'Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf'))).toBe(true);
    expect(existsSync(join(finalDir(), 'Qwen3.8-27B-UD-Q4_K_XL-00002-of-00002.gguf'))).toBe(true);
    // The old revision staging dir is a separate scope, left untouched.
    expect(existsSync(join(legacyStagingDir(SHA_OLD), WEIGHT))).toBe(true);
  });

  it('replays the start frame then the last event to a late subscriber', async () => {
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    const SHARD = 'Qwen3.8-27B-UD-Q4_K_XL-00002-of-00002.gguf';
    hub.manifest = [
      { type: 'file', path: 'config.json', size: MIB },
      { type: 'file', path: 'Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf', size: 2 * MIB },
      { type: 'file', path: SHARD, size: 4 * MIB },
    ];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeStallingFetch(
        { 'config.json': MIB, 'Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf': 2 * MIB, [SHARD]: 4 * MIB },
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
    expect(settleOf('Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf')).toMatchObject({
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
        path: WEIGHT,
        size: 300,
        lfs: { oid: zerosSha256(299), size: 300, pointerSize: 100 },
      },
    ];
    // Seed a cache-HIT pointer as a symlink pointing at the out-of-cache victim.
    const pointer = hub.cachePointer(cacheDir, hub.sha, WEIGHT);
    mkdirSync(dirname(pointer), { recursive: true });
    symlinkSync(victim, pointer);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    writeFileSync(join(deadDir, WEIGHT), Buffer.alloc(10, 0x1));
    // A concurrent live job's private staging tree (this process' own, alive pid).
    const liveDir = join(stagingRoot(), `${SLUG}@${hub.sha}.${process.pid}.22222222-2222-2222-2222-222222222222`);
    mkdirSync(liveDir, { recursive: true });
    writeFileSync(join(liveDir, WEIGHT), Buffer.alloc(10, 0x2));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const id = manager.start(REPO);
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The dead-pid tree was reaped; the live-pid tree was left intact.
    expect(existsSync(deadDir)).toBe(false);
    expect(existsSync(liveDir)).toBe(true);
    expect(readFileSync(join(liveDir, WEIGHT))).toEqual(Buffer.alloc(10, 0x2));
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
    writeFileSync(join(deadDir, WEIGHT), Buffer.alloc(10, 0x1));

    // finalDir is an UNOWNED real dir (a manual / `mlx download` copy, no marker), so
    // the ownership preflight refuses this job (overwrite:false).
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300, 0xab));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300, 0xab));
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
    writeFileSync(join(external, WEIGHT), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(external, DOWNLOAD_COMPLETE_MARKER),
      // repo + revision MATCH the resolved snapshot so pre-fix skip-detection would
      // read the foreign marker as already-installed and emit `done` through the link.
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', WEIGHT], completedAt: 'x' }),
    );
    // A dead-owner (pid 999999999) rollback backup that is a SYMLINK into `external`.
    mkdirSync(stagingRoot(), { recursive: true });
    const backup = join(stagingRoot(), `${SLUG}.backup-999999999.44444444-4444-4444-4444-444444444444`);
    symlinkSync(external, backup);
    expect(existsSync(finalDir())).toBe(false);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    expect(readFileSync(join(external, WEIGHT))).toEqual(Buffer.alloc(300, 0xab));

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
      fetchImpl: makeCancelFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, () => {
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
      fetchImpl: makeCancelFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, () => {
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
    hub.failOn = [WEIGHT];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    // Both paths carry the UD-Q4_K_XL token so they clear the entry's glob filter
    // and actually reach the ingestion safety gate — a path the filter drops never
    // exercises `isSafeRelPath` at all.
    for (const badPath of ['../Qwen3.8-27B-UD-Q4_K_XL-escape.json', '/etc/Qwen3.8-27B-UD-Q4_K_XL-evil.json']) {
      hub.manifest = [
        { type: 'file', path: 'config.json', size: 12 },
        { type: 'file', path: badPath, size: 5 },
        { type: 'file', path: WEIGHT, size: 300 },
      ];
      const manager = new DownloadManager({
        modelsDir,
        cacheDir,
        fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      expect(existsSync(join(modelsDir, 'Qwen3.8-27B-UD-Q4_K_XL-escape.json'))).toBe(false);
      expect(existsSync(join(stagingRoot(), 'Qwen3.8-27B-UD-Q4_K_XL-escape.json'))).toBe(false);
      expect(existsSync(join(modelsDir, 'Qwen3.8-27B-UD-Q4_K_XL-evil.json'))).toBe(false);
      expect(existsSync('/etc/Qwen3.8-27B-UD-Q4_K_XL-evil.json')).toBe(false);
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
      fetchImpl: makeCancelFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, () => {
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

    // Staging cleaned; nothing published. The `cancelled` event is emitted in
    // `catch` before `finally` awaits the staging `rm`, so wait for the
    // directory to disappear rather than asserting on the event (same race as
    // the error-path tests above).
    await waitFor(() => jobStagingDirs().length === 0);
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
      fetchImpl: makeCancelFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, () => {
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    hub.failOn = [WEIGHT];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
    // staged files (config.json / the weight) were written into it.
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
      fetchImpl: makeCancelFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, () => {
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
      fetchImpl: makeCancelFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, () => {
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
      fetchImpl: makeCancelFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, () => {
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
    hub.failOn = [WEIGHT];
    hub.failMessage = `boom ${'x'.repeat(100_000)}`;
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
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
