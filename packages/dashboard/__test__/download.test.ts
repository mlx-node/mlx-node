import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  chmodSync,
  cpSync,
  existsSync,
  linkSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  renameSync,
  rmSync,
  symlinkSync,
  truncateSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';

import type { ListFileEntry } from '@huggingface/hub';
import {
  type CatalogBuild,
  catalogBuildRepo,
  catalogBuildSlug,
  catalogDraftSlug,
  catalogUpdateRepos,
  MODEL_CATALOG,
} from '@mlx-node/agent/catalog';
import { findDFlash2Draft, INKLING2_DFLASH2, QWEN38_DFLASH2 } from '@mlx-node/lm/draft-companion';
import { discoverLocalChatModels } from '@mlx-node/lm/model-discovery';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vite-plus/test';

import type { ApiContext } from '../src/api/context.js';
import { dispatch } from '../src/api/dispatch.js';
import { type CatalogBuildItem, catalogWithState, configlessBuildOf } from '../src/catalog.js';
import {
  DownloadConflictError,
  type DownloadEvent,
  DownloadManager,
  pidAlive,
  proveInPlace,
  type ReplaceBlocker,
  restoreWindowsSymlink,
} from '../src/download.js';
import {
  DOWNLOAD_COMPLETE_MARKER,
  deleteLocalModel,
  isModelInstalled,
  readCompletion,
  summarizeForeignEntry,
} from '../src/models.js';
import { hasUpdate, modelPresent } from '../ui/src/lib/catalog.js';
import { catalogActionState } from '../ui/src/pages/models/catalog.js';
import { ggufHeader } from './helpers/configless-gguf-fixtures.js';

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
  /** A second prefix, so a test can fail the swap AND its rollback. */
  alsoFailFromPrefix: null as string | null,
  /** A target prefix, so a test can fail one move by where it goes. */
  failToPrefix: null as string | null,
}));

// Fires once around the first rename whose source is `from`: `before` runs just before the real rename, `after`
// just after it, to act like a writer outside the app at the publish swap.
const renameHook = vi.hoisted(() => ({
  from: null as string | null,
  before: null as (() => void) | null,
  after: null as (() => void) | null,
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

// Runs before every `link` with its target, so a test can act like a writer in the folder during the adds, or make
// the link fail by throwing. `probe` runs instead for the plan's link test in `.staging` (a `.link-test-` name).
const linkHook = vi.hoisted(() => ({
  before: null as ((to: string) => void) | null,
  probe: null as ((to: string) => void) | null,
}));

// Runs before every async `unlink` with its path, so a test can make the unlink fail by throwing.
const unlinkHook = vi.hoisted(() => ({ before: null as ((path: string) => void) | null }));

// Every async `rm` the runner makes, by path, so a test can prove it never reaches an entry the app did not create.
const rmCalls = vi.hoisted(() => ({ paths: [] as string[] }));

// The Trash every manager in this file gets: the default `moveToSystemTrash` renames the path into `dir` (a fixture
// folder per test), so no test reaches the real Trash. `refuse` makes it reject and move nothing; `ignore` makes it
// answer success and move nothing. `before` runs first with the path, so a test can change the disk around the call.
const trash = vi.hoisted(() => ({
  dir: '',
  calls: [] as string[],
  refuse: false,
  ignore: false,
  before: null as ((path: string) => void) | null,
}));

vi.mock('../src/trash.js', async () => {
  const { existsSync: exists, renameSync } = await import('node:fs');
  const { basename, join } = await import('node:path');
  return {
    moveToSystemTrash: async (path: string) => {
      trash.calls.push(path);
      trash.before?.(path);
      if (trash.refuse) throw new Error('simulated Trash refusal');
      if (trash.ignore) return;
      // A second item of the same name gets a suffix, as in the real Trash.
      let target = join(trash.dir, basename(path));
      for (let n = 2; exists(target); n++) target = join(trash.dir, `${basename(path)} ${n}`);
      renameSync(path, target);
    },
  };
});

// Runs before every async `lstat` with its path, so a test can remove an entry between a folder listing and its stat.
// `after` runs once the real `lstat` has returned, so a test can act like a writer between a check and the next call.
const lstatHook = vi.hoisted(() => ({
  before: null as ((path: string) => void) | null,
  after: null as ((path: string) => void) | null,
}));

vi.mock('node:fs/promises', async (importActual) => {
  const actual = await importActual<typeof import('node:fs/promises')>();
  return {
    ...actual,
    rename: async (from: string, to: string) => {
      const f = String(from);
      if (
        (renameFault.failFromPath !== null && f === renameFault.failFromPath) ||
        (renameFault.failFromPrefix !== null && f.startsWith(renameFault.failFromPrefix)) ||
        (renameFault.alsoFailFromPrefix !== null && f.startsWith(renameFault.alsoFailFromPrefix)) ||
        (renameFault.failToPrefix !== null && String(to).startsWith(renameFault.failToPrefix))
      ) {
        throw new Error(`simulated rename failure moving ${f}`);
      }
      if (renameHook.from !== null && f === renameHook.from) {
        const { before, after } = renameHook;
        renameHook.from = null;
        before?.();
        await actual.rename(from, to);
        after?.();
        return;
      }
      return actual.rename(from, to);
    },
    writeFile: async (path: string, data: string | Uint8Array, options?: { flag?: string }) => {
      // `includes` catches the atomic writer too: `writeMarkerAtomically` puts
      // the marker's bytes in `<marker>.<pid>.<uuid>.tmp` before the rename, so
      // a refresh-time hook fires on the temp write, not the final path.
      if (raceHook.onMarkerWrite !== null && String(path).includes(MARKER_FILE)) {
        const hook = raceHook.onMarkerWrite;
        raceHook.onMarkerWrite = null;
        hook();
      }
      return actual.writeFile(path, data, options);
    },
    link: async (from: string, to: string) => {
      if (String(to).includes('/.link-test-')) linkHook.probe?.(String(to));
      else linkHook.before?.(String(to));
      return actual.link(from, to);
    },
    unlink: async (path: string) => {
      unlinkHook.before?.(String(path));
      return actual.unlink(path);
    },
    lstat: async (path: string, options?: { bigint?: boolean }) => {
      lstatHook.before?.(String(path));
      const stat = await actual.lstat(path, options as never);
      lstatHook.after?.(String(path));
      return stat;
    },
    rm: async (path: string, options?: { recursive?: boolean; force?: boolean }) => {
      rmCalls.paths.push(String(path));
      return actual.rm(path, options);
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

/** The build `id` of the catalog entry `label`. */
function catalogBuild(label: string, id: string): CatalogBuild {
  const build = MODEL_CATALOG.find((entry) => entry.label === label)?.builds.find((candidate) => candidate.id === id);
  if (build === undefined) throw new Error(`no build ${label} ${id}`);
  return build;
}

/** The first build of the default model (Qwen3.8-27B UD-Q4_K_XL): the build most cases install. */
const BUILD = catalogBuild('Qwen3.8-27B', 'UD-Q4_K_XL');
// Resolved through `catalogBuildRepo`, matching the target `start` looks up — the
// catalog can carry a different build per platform, so the raw `hfRepo` would be
// wrong on a CUDA host.
const REPO = catalogBuildRepo(BUILD);
/** The folder {@link BUILD} installs to: the key `start` takes. */
const SLUG = catalogBuildSlug(BUILD);
/** A build of ANOTHER catalog repo, for the cases that need two genuinely distinct jobs. Its `*UD-Q4_K_XL.gguf` glob takes {@link WEIGHT}. */
const BUILD_OTHER = catalogBuild('Inkling-2-Mini', 'UD-Q4_K_XL');
const REPO_OTHER = catalogBuildRepo(BUILD_OTHER);
const SLUG_OTHER = catalogBuildSlug(BUILD_OTHER);
/** The folder of the auto-installed Qwen3.8-27B DFlash2 draft. */
const DRAFT_SLUG = catalogDraftSlug(QWEN38_DFLASH2);
/** The folder of the auto-installed Inkling-2-Mini DFlash2 draft. */
const INKLING_DRAFT_SLUG = catalogDraftSlug(INKLING2_DFLASH2);
/**
 * The `Qwen/Qwen3.8-27B` base-model repo {@link BUILD}'s `assetsRepo` names for
 * the tokenizer/config sidecars a GGUF quantization repo does not ship.
 */
const ASSETS_REPO = BUILD.assetsRepo!;
/**
 * The single UD-Q4_K_XL weight variant {@link BUILD}'s globs select out of the
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
/** The catalog state of the build that installs to `slug`, from a fresh scan. */
function buildState(slug: string, platform: NodeJS.Platform = process.platform): CatalogBuildItem | undefined {
  return catalogWithState(modelsDir, platform)
    .flatMap((item) => item.builds)
    .find((build) => build.slug === slug);
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
/** Cleanup holders left under `.staging` (`<slug>.cleanup-...`); empty when clean. */
function holderDirs(): string[] {
  return existsSync(stagingRoot()) ? readdirSync(stagingRoot()).filter((n) => n.includes('.cleanup-')) : [];
}
/** Old entries a cleanup kept under `.staging` (`<slug>.kept-...`). */
function keptDirs(): string[] {
  return existsSync(stagingRoot()) ? readdirSync(stagingRoot()).filter((n) => n.startsWith(`${SLUG}.kept-`)) : [];
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
  renameFault.alsoFailFromPrefix = null;
  renameFault.failToPrefix = null;
  renameHook.from = null;
  renameHook.before = null;
  renameHook.after = null;
  raceHook.onMarkerWrite = null;
  stagingHook.onVerifyWindow = null;
  linkHook.before = null;
  linkHook.probe = null;
  lstatHook.before = null;
  lstatHook.after = null;
  unlinkHook.before = null;
  hub.modelInfoRepos = [];
  hub.modelInfoError = null;
  hub.modelInfoFailRepos = [];
  hub.modelInfoUsesFetch = false;
  modelsDir = mkdtempSync(join(tmpdir(), 'dash-dl-models-'));
  cacheDir = mkdtempSync(join(tmpdir(), 'dash-dl-cache-'));
  rmCalls.paths = [];
  trash.dir = mkdtempSync(join(tmpdir(), 'dash-dl-trash-'));
  trash.calls = [];
  trash.refuse = false;
  trash.ignore = false;
  trash.before = null;
});

afterEach(() => {
  for (const dir of [modelsDir, cacheDir, trash.dir]) rmSync(dir, { recursive: true, force: true });
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
    for (const build of MODEL_CATALOG.filter((e) => !e.hidden).flatMap((e) => e.builds)) {
      if (build.assetsRepo === undefined) continue;
      expect(shas.has(build.assetsRepo)).toBe(true);
      expect(hub.modelInfoRepos).toContain(build.assetsRepo);
    }
    for (const build of MODEL_CATALOG.filter((e) => e.hidden).flatMap((e) => e.builds)) {
      expect(hub.modelInfoRepos).not.toContain(catalogBuildRepo(build));
      if (build.assetsRepo !== undefined) expect(hub.modelInfoRepos).not.toContain(build.assetsRepo);
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
    const id = m.start(SLUG).id;
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
    const id = m.start(SLUG).id;
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
    const id = m.start(SLUG).id;
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
    const id = m.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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

  it.each([
    {
      label: 'Qwen3.8-27B',
      slug: DRAFT_SLUG,
      family: 'qwen3_5',
      probePath: 'qwen3.8-27b-mxfp4-mlx',
      badArch: 'Qwen3ForCausalLM',
      valid: true,
    },
    {
      label: 'Qwen3.8-27B',
      slug: DRAFT_SLUG,
      family: 'qwen3_5',
      probePath: 'qwen3.8-27b-mxfp4-mlx',
      badArch: 'Qwen3ForCausalLM',
      valid: false,
    },
    {
      label: 'Inkling-2-Mini',
      slug: INKLING_DRAFT_SLUG,
      family: 'inkling2',
      probePath: 'inkling-2-mini',
      badArch: 'Inkling2LoopForConditionalGeneration',
      valid: true,
    },
    {
      label: 'Inkling-2-Mini',
      slug: INKLING_DRAFT_SLUG,
      family: 'inkling2',
      probePath: 'inkling-2-mini',
      badArch: 'Inkling2LoopForConditionalGeneration',
      valid: false,
    },
  ])(
    'publishes only a complete DFlash2 companion ($label, valid architecture: $valid)',
    async ({ label, slug, family, probePath, badArch, valid }) => {
      const config = JSON.stringify({
        model_type: family === 'inkling2' ? 'inkling2_loop' : 'qwen3',
        architectures: [valid ? 'DFlash2DraftModel' : badArch],
        ...(family === 'inkling2' ? { hidden_size: 2304, num_target_layers: 48 } : {}),
      });
      // The draft repo is a catalog DOWNLOAD target with no globs and no assetsRepo:
      // the no-glob default filter applies and the companion stays a safetensors repo.
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
      const id = manager.start(slug).id;
      await waitFor(() => manager.jobs().some((job) => job.id === id && ['done', 'error'].includes(job.state)));
      expect(manager.jobs().find((job) => job.id === id)?.state).toBe(valid ? 'done' : 'error');
      const draftPath = join(modelsDir, slug);
      expect(existsSync(draftPath)).toBe(valid);
      expect(findDFlash2Draft(join(modelsDir, probePath), family, modelsDir)).toBe(valid ? draftPath : undefined);
      const target = catalogWithState(modelsDir).find((item) => item.label === label)!;
      expect(target.builds.some((build) => build.present)).toBe(false);
      expect(target.draft?.present).toBe(valid);
      expect(target.draft?.installed).toBe(valid);
      if (valid) {
        // Re-running a completed DFlash2 draft short-circuits without re-downloading,
        // and mutating config.json out of DFlash2DraftModel forces a re-download instead of skipping.
        writeFileSync(join(draftPath, 'config.json'), JSON.stringify({ architectures: [badArch] }));
        const repairId = manager.start(slug).id;
        await waitFor(() => manager.jobs().some((job) => job.id === repairId && job.state === 'done'));
        expect(JSON.parse(readFileSync(join(draftPath, 'config.json'), 'utf8')).architectures).toEqual([
          'DFlash2DraftModel',
        ]);
      }
      await manager.shutdown();
    },
  );

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

    // Its folder, through the resolver the catalog installs it with.
    expect(() => manager.start(catalogBuildSlug(hidden!.builds[0]))).toThrow('is not a folder of the model catalog');
    expect(() => manager.start('someone-not-in-catalog')).toThrow('is not a folder of the model catalog');
    // Rejected BEFORE any job is allocated — otherwise the SPA renders a job
    // that marches to a 401 instead of an immediate, actionable error.
    expect(manager.jobs()).toEqual([]);

    // Non-hidden entries are unaffected.
    expect(() => manager.start(SLUG)).not.toThrow();
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
    const id = manager.start(SLUG).id;
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
      const id = manager.start(SLUG).id;
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
      const id = manager.start(SLUG).id;
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
      const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const item = buildState(SLUG)!;
    expect(item.installed).toBe(false);
  });

  it('installs a repo that keeps its weights in a variant folder, and the catalog reads it installed and present', async () => {
    const build = catalogBuild('Qwen3.8-Flash-Next', 'UD-Q4_K_XL');
    const repo = catalogBuildRepo(build);
    const assetsRepo = build.assetsRepo!;
    const shards = [1, 2, 3, 4].map((part) => `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000${part}-of-00004.gguf`);
    hub.manifests[repo] = [
      { type: 'directory', path: 'UD-Q4_K_XL', size: 0 },
      ...shards.map((path) => ({ type: 'file' as const, path, size: 100 })),
      { type: 'file', path: 'UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf', size: 100 },
      { type: 'file', path: 'MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf', size: 100 },
      { type: 'file', path: 'mmproj-BF16.gguf', size: 100 },
      { type: 'file', path: 'README.md', size: 10 },
    ];
    hub.manifests[assetsRepo] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({
        ...Object.fromEntries(shards.map((path) => [path, 100])),
        'config.json': 12,
        'tokenizer.json': 20,
      }),
    });
    const id = manager.start(catalogBuildSlug(build)).id;
    await waitFor(() => manager.jobs().some((job) => job.id === id && ['done', 'error'].includes(job.state)));
    expect(manager.jobs().find((job) => job.id === id)?.state).toBe('done');
    expect([...hub.downloadedFrom].sort()).toEqual(
      [...shards.map((path) => `${repo}/${path}`), `${assetsRepo}/config.json`, `${assetsRepo}/tokenizer.json`].sort(),
    );
    const item = catalogWithState(modelsDir).find((candidate) => candidate.label === 'Qwen3.8-Flash-Next')!.builds[0];
    for (const shard of shards) expect(existsSync(join(modelsDir, item.slug, shard)), shard).toBe(true);
    expect(existsSync(join(modelsDir, item.slug, 'config.json'))).toBe(true);
    expect(item).toMatchObject({ installed: true, present: true, blockedByForeignDir: false, blocker: null });
    await manager.shutdown();
  });

  it('errors on an empty manifest instead of publishing a hollow dir', async () => {
    hub.manifest = [];
    const manager = new DownloadManager({ modelsDir, cacheDir, fetchImpl: makeFetchImpl({}) });
    const events: DownloadEvent[] = [];
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));

    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(events.find((event) => event.type === 'error')).toMatchObject({ type: 'error' });
    expect(existsSync(finalDir())).toBe(false);
    const item = buildState(SLUG)!;
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
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(existsSync(finalDir())).toBe(false);
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(false);
    expect(buildState(SLUG)!.installed).toBe(false);
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
    const id = manager.start(DRAFT_SLUG).id;
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
    expect(buildState(SLUG)!.installed).toBe(false);

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
    expect(buildState(SLUG)!.installed).toBe(true);

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
    expect(buildState(SLUG)!.installed).toBe(false);

    // A one-sided marker (config-only, no weight) is never installed either — a
    // hollow publish `loadModel` would reject (Finding G2).
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json'], completedAt: 'x' }),
    );
    expect(buildState(SLUG)!.installed).toBe(false);
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
    const id1 = manager.start(SLUG).id;
    manager.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'error'));
    expect(hub.downloaded).toContain('config.json');

    // Second job: the weight succeeds; config.json is served from the HF cache.
    hub.failOn = [];
    hub.downloaded = [];
    const id2 = manager.start(SLUG).id;
    await waitFor(() => manager.jobs().some((j) => j.id === id2 && j.state === 'done'));

    // config.json was a cache hit (no re-fetch); only the weight was fetched.
    expect(hub.downloaded).not.toContain('config.json');
    expect(hub.downloaded).toContain(WEIGHT);
    expect(buildState(SLUG)!.installed).toBe(true);
  });

  it('downloads only the glob-matched variants of a multi-variant GGUF repo', async () => {
    // A real Unsloth repo ships dozens of quantization variants side by side. The
    // entry's globs (`*UD-Q4_K_XL*`, `config.json`, `mmproj-BF16.gguf`) must select
    // exactly the one build the wizard installs and its vision projector — never the
    // whole multi-hundred-GB repo.
    const OTHER_QUANT = 'Qwen3.8-27B-UD-Q8_K_XL.gguf';
    const NON_UD = 'Qwen3.8-27B-Q4_K_M.gguf';
    const MTP = 'MTP/mtp-Qwen3.8-27B-Q4_0.gguf';
    const MMPROJ = 'mmproj-BF16.gguf';
    const OTHER_MMPROJ = 'mmproj-F16.gguf';
    const README = 'README.md';
    hub.manifest = [
      { type: 'file', path: WEIGHT, size: 300 },
      { type: 'file', path: OTHER_QUANT, size: 222 },
      { type: 'file', path: NON_UD, size: 111 },
      { type: 'file', path: MTP, size: 33 },
      { type: 'file', path: MMPROJ, size: 44 },
      { type: 'file', path: OTHER_MMPROJ, size: 43 },
      { type: 'file', path: README, size: 55 },
    ];
    // The assets repo is a genuinely different repo, so it gets its own listing.
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'config.json', size: 12 }];

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, [MTP]: 33, [MMPROJ]: 44, 'config.json': 12 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done'));

    // Exactly the glob-matched primary files moved (the MTP artifact is NOT
    // globbed: the UD file carries its MTP layer inline, nothing pairs a GGUF
    // MTP sidecar). The `start` frame carries the FULL total — the weight, the
    // projector and the assets repo's config.json sidecar (300 + 44 + 12) —
    // because the sidecar plan is resolved before `start` so the byte bar never
    // overshoots.
    expect(events.find((event) => event.type === 'start')).toMatchObject({
      type: 'start',
      repo: REPO,
      totalBytes: 356,
      fileCount: 3,
    });
    // EVERY frame must report the aggregate (2 primary + 1 sidecar): the UI
    // reducer replaces its count with each progress event's, so a primary-only
    // count here would shrink the displayed total and then jump when the
    // sidecar starts.
    const progressCounts = events.filter((event) => event.type === 'progress').map((event) => event.fileCount);
    expect(progressCounts.length).toBeGreaterThan(0);
    expect(new Set(progressCounts)).toEqual(new Set([3]));
    expect([...hub.downloaded].sort()).toEqual([WEIGHT, MMPROJ, 'config.json'].sort());

    // The unmatched variants were never fetched and never published…
    for (const path of [OTHER_QUANT, NON_UD, MTP, OTHER_MMPROJ, README]) {
      expect(hub.downloaded, `${path} must not be fetched`).not.toContain(path);
      expect(existsSync(join(finalDir(), path)), `${path} must not be published`).toBe(false);
    }
    // …while the selected build, its projector and the core metadata landed.
    for (const path of [WEIGHT, MMPROJ, 'config.json']) expect(existsSync(join(finalDir(), path)), path).toBe(true);
    expect(readCompletion(finalDir())!.files).toContain(MMPROJ);
    expect(buildState(SLUG)).toMatchObject({ installed: true });
    expect(buildState(SLUG)!.lacksNamedFile).toBeUndefined();
  });

  it('updates an install whose marker lacks the projector the catalog names, and publishes it', async () => {
    const MMPROJ = 'mmproj-BF16.gguf';
    hub.manifest = [
      { type: 'file', path: WEIGHT, size: 300 },
      { type: 'file', path: MMPROJ, size: 44 },
    ];
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'config.json', size: 12 }];
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, [MMPROJ]: 44, 'config.json': 12 }),
    });
    const runJob = async (): Promise<DownloadEvent[]> => {
      const events: DownloadEvent[] = [];
      const id = manager.start(SLUG).id;
      manager.subscribe(id, (event) => events.push(event));
      await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));
      return events;
    };
    expect((await runJob()).at(-1)?.type).toBe('done');

    // The install as a catalog without the projector made it: same revision, the projector neither on disk nor listed.
    const marker = readCompletion(finalDir())!;
    rmSync(join(finalDir(), MMPROJ));
    writeFileSync(
      join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ ...marker, files: marker.files.filter((file) => file !== MMPROJ) }),
    );
    expect(buildState(SLUG)).toMatchObject({ installed: true, lacksNamedFile: true });

    expect((await runJob()).at(-1)?.type).toBe('done');
    expect(existsSync(join(finalDir(), MMPROJ))).toBe(true);
    expect(readCompletion(finalDir())!.files).toContain(MMPROJ);
    expect(buildState(SLUG)!.lacksNamedFile).toBeUndefined();
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
    const id = manager.start(SLUG).id;
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
    expect(buildState(SLUG)!.installed).toBe(true);
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
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done'));

    expect(events.some((event) => event.type === 'error')).toBe(false);
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    expect(buildState(SLUG)!.installed).toBe(true);
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
    const id = manager.start(SLUG).id;
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
    // The Inkling GGUF builds' globs select a companion by name
    // (`mmproj-BF16.gguf`, for the vision tower). A partial upstream upload (or
    // a renamed target) can leave the manifest with the projector and nothing
    // else: publishing that would certify a directory model-discovery never
    // lists — "Installed" with no loadable model. (The Qwen3.8 builds cannot
    // express this: their globs filter a companion out before the gate runs.)
    expect(BUILD_OTHER.globs).toContain('mmproj-BF16.gguf');

    hub.manifest = [{ type: 'file', path: 'mmproj-BF16.gguf', size: 44 }];
    hub.manifests[BUILD_OTHER.assetsRepo!] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'mmproj-BF16.gguf': 44, 'config.json': 12, 'tokenizer.json': 20 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(SLUG_OTHER).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error' || event.type === 'done'));

    expect(events.some((event) => event.type === 'error')).toBe(true);
    expect(events.some((event) => event.type === 'done')).toBe(false);
    expect(existsSync(join(modelsDir, SLUG_OTHER))).toBe(false);
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
    const id1 = first.start(SLUG).id;
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
    const id2 = second.start(SLUG).id;
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
    const id1 = first.start(SLUG).id;
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
    const id2 = second.start(SLUG).id;
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
    const id1 = first.start(SLUG).id;
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
    const id2 = second.start(SLUG).id;
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
    const id1 = first.start(SLUG).id;
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
    const id2 = second.start(SLUG).id;
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

  it('never writes through a LIVE final symlink during the sidecar refresh — the refresh re-downloads instead', async () => {
    // Same-revision install whose marker records a sidecar the assets repo has
    // since dropped (tokenizer.json) AND a stale assetsRevision: on a real dir
    // the verified-done refresh would DELETE the stale sidecar and REWRITE the
    // marker in place. With `<modelsDir>/<slug>` a live symlink into that
    // external marked dir, those writes would land inside the foreign target —
    // the exact boundary `deleteLocalModel` draws by unlinking only the link.
    // The refresh must refuse the link; the job then takes the full download
    // path whose publish swap replaces the LINK with an owned real directory.
    const external = mkdtempSync(join(tmpdir(), 'dash-dl-ext-'));
    writeFileSync(join(external, 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(external, 'tokenizer.json'), Buffer.alloc(20, 0xab));
    writeFileSync(join(external, WEIGHT), Buffer.alloc(300, 0xab));
    const externalMarker = join(external, DOWNLOAD_COMPLETE_MARKER);
    writeFileSync(
      externalMarker,
      JSON.stringify({
        repo: REPO,
        revision: SHA_DEFAULT,
        files: ['config.json', 'tokenizer.json', WEIGHT],
        assetsRepo: ASSETS_REPO,
        assetsRevision: SHA_OLD,
        completedAt: 'x',
      }),
    );
    symlinkSync(external, finalDir());

    // Same primary revision (SHA_DEFAULT), so skip-detection reaches the
    // refresh; the assets repo advanced and no longer supplies tokenizer.json,
    // so the refresh wants to mutate — and must not, through the link.
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'config.json', size: 12 }];
    hub.shaByRepo[ASSETS_REPO] = SHA_NEW;

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'config.json': 12 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));

    expect(events.some((event) => event.type === 'error')).toBe(false);
    expect(events.some((event) => event.type === 'done')).toBe(true);

    // The refresh fell through to a real download+publish: the link was
    // replaced by an owned REAL directory carrying the current marker.
    expect(lstatSync(finalDir()).isSymbolicLink()).toBe(false);
    expect(isModelInstalled(finalDir())).toBe(true);
    const marker = JSON.parse(readFileSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as {
      assetsRepo?: string;
      assetsRevision?: string;
    };
    expect(marker.assetsRepo).toBe(ASSETS_REPO);
    expect(marker.assetsRevision).toBe(SHA_NEW);

    // The external target is byte-for-byte intact: no sidecar was deleted and
    // the marker was never rewritten through the link.
    expect(readFileSync(join(external, 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(external, 'tokenizer.json'))).toEqual(Buffer.alloc(20, 0xab));
    expect(readFileSync(join(external, WEIGHT))).toEqual(Buffer.alloc(300, 0xab));
    expect((JSON.parse(readFileSync(externalMarker, 'utf-8')) as { assetsRevision?: string }).assetsRevision).toBe(
      SHA_OLD,
    );

    rmSync(external, { recursive: true, force: true });
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
      const idOne = first.start(SLUG).id;
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
        const idTwo = second.start(SLUG).id;
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
    // The primary GGUF repo ships no config.json here: the
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
    const id1 = first.start(SLUG).id;
    first.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'done'));
    expect(existsSync(join(finalDir(), 'config.json'))).toBe(true);

    // Upstream renames its config away; only tokenizer.json remains supplied.
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'tokenizer.json', size: 20 }];
    hub.shaByRepo[ASSETS_REPO] = SHA_NEW;
    hub.downloaded = [];
    const second = new DownloadManager({ modelsDir, cacheDir, fetchImpl: makeFetchImpl({ [WEIGHT]: 300 }) });
    const events2: DownloadEvent[] = [];
    const id2 = second.start(SLUG).id;
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
    const id1 = manager.start(SLUG).id;
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
    const id2 = manager.start(SLUG).id;
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
    const id1 = first.start(SLUG).id;
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
    const id2 = second.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    expect(buildState(SLUG)!.installed).toBe(true);
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
    const id = manager.start(SLUG).id;
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The corrupt cache entry failed sha256 → was invalidated → really re-fetched →
    // matched → published (without invalidation the retry would recopy the same blob).
    expect(hub.downloaded).toContain(WEIGHT);
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300));
    expect(buildState(SLUG)!.installed).toBe(true);
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
    const id = manager.start(SLUG).id;
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    expect(hub.downloaded, 'a same-length corrupt Xet blob was accepted without a re-fetch').toContain(WEIGHT);
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300));
    expect(buildState(SLUG)!.installed).toBe(true);
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
    const id = manager.start(SLUG).id;
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // Wrong git-blob oid → invalidated → re-fetched → matched.
    expect(hub.downloaded).toContain('config.json');
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    expect(buildState(SLUG)!.installed).toBe(true);
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    // The mismatching file was re-fetched (bounded) before the job gave up.
    expect(hub.downloaded.filter((p) => p === WEIGHT).length).toBeGreaterThan(1);
    // No publish, no marker, catalog reports not-installed.
    expect(existsSync(finalDir())).toBe(false);
    expect(buildState(SLUG)!.installed).toBe(false);
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
    const id = manager.start(SLUG).id;
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
      // A plain install names the refusal; only a cleanup may replace the entry.
      expect(err.message.toLowerCase()).not.toContain('overwrite');
      expect(err.message).toContain('not created by the dashboard');
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

    const item = buildState(SLUG)!;
    expect(item.present).toBe(false);
    expect(item.blockedByForeignDir).toBe(true);

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    const err = events.find((event) => event.type === 'error');
    expect(err?.type).toBe('error');
    if (err?.type === 'error') expect(err.message).toContain('not created by the dashboard');
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
    const id = manager.start(SLUG).id;
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
      expect(err.message).toContain('not created by the dashboard');
    }
  });

  it('installs over a LIVE final symlink → external marked dir, replacing only the LINK', async () => {
    // `<modelsDir>/<slug>` is a LIVE symlink whose target is an EXTERNAL directory
    // carrying a valid completion marker. Ownership now FOLLOWS the link — the
    // foreign marker is a truthful record of the install the link names — so the
    // job is admitted, and the owned swap is safe by construction: `rename` moves
    // the LINK aside and `rm` on a dir-symlink unlinks it; the external target is
    // never written through.
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
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));

    expect(events.some((event) => event.type === 'error')).toBe(false);
    expect(events.some((event) => event.type === 'done')).toBe(true);

    // The link was replaced by a REAL staged dir — not a symlink, our marker inside.
    expect(lstatSync(finalDir()).isSymbolicLink()).toBe(false);
    expect(lstatSync(finalDir()).isDirectory()).toBe(true);
    expect(isModelInstalled(finalDir())).toBe(true);

    // The external target is byte-for-byte intact — nothing was written through the link.
    expect(readFileSync(join(external, 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(external, WEIGHT))).toEqual(Buffer.alloc(300, 0xab));

    rmSync(external, { recursive: true, force: true });
  });

  it('keeps a dead-owner rollback backup when finalDir is a LIVE foreign symlink, then installs over the link', async () => {
    // finalDir is a LIVE symlink into an EXTERNAL complete marked model AND a
    // dead-owner rollback backup (`<slug>.backup-<deadpid>.<uuid>`) sits under
    // `.staging`. The link admits the job now (its target is marked), so the
    // recovery sweep DOES run — but its final classification is `lstat`, which
    // reports any symlink as not-a-directory, so the link is classified
    // "other occupant" and the recoverable backup is RETAINED. The job then
    // installs: the swap moves the LINK aside and `rm` unlinks it.
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
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));

    expect(events.some((event) => event.type === 'error')).toBe(false);
    expect(events.some((event) => event.type === 'done')).toBe(true);

    // The recoverable dead-owner rollback backup was NOT reaped or restored —
    // byte-for-byte intact under `.staging`.
    expect(existsSync(backup)).toBe(true);
    expect(readFileSync(join(backup, 'config.json'))).toEqual(Buffer.alloc(12, 0x5a));
    expect(readFileSync(join(backup, WEIGHT))).toEqual(Buffer.alloc(300, 0x5a));
    expect(backupDirs()).toEqual([`${SLUG}.backup-999999999.44444444-4444-4444-4444-444444444444`]);

    // The link was replaced by a real install; the external target is untouched.
    expect(lstatSync(finalDir()).isSymbolicLink()).toBe(false);
    expect(isModelInstalled(finalDir())).toBe(true);
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
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    // The manual files are byte-for-byte intact and no marker was written into them.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300, 0xab));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(false);
    expect(buildState(SLUG)!.installed).toBe(false);
  });

  it('refuses a dir whose marker is not the full shape we write, and leaves it intact', async () => {
    // For a plain install, ownership is the ONLY gate between `POST /api/downloads`
    // and `rename(finalDir → backup)` + `rm(backup, { recursive: true })`: all
    // three guards on that path reduce to this one predicate. It used to accept anything carrying a `files` array, and the
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
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(
      readFileSync(join(finalDir(), 'PRECIOUS.safetensors')),
      'a marker carrying only `files` authorized the destructive publish swap',
    ).toEqual(Buffer.alloc(4096, 0x7f));
    expect(events.some((event) => event.type === 'done')).toBe(false);
    // And the catalog must agree, or the UI would keep offering the Install that
    // the runner now refuses every time.
    expect(buildState(SLUG)!.blockedByForeignDir).toBe(true);
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
    const id1 = manager.start(SLUG).id;
    manager.subscribe(id1, (event) => events1.push(event));
    let idDuringCancel: string | undefined;
    stagingHook.onVerifyWindow = () => {
      manager.cancel(id1);
      // SYNCHRONOUSLY inside the window — no await, so `processJob` has not yet
      // reached its catch and the job still reads `running`.
      idDuringCancel = manager.start(SLUG).id;
    };
    await waitFor(() => events1.some((event) => event.type === 'cancelled'));
    expect(idDuringCancel).not.toBe(id1);
    // The second `start` runs a real job detached: without this the `afterEach` `rmSync` races its writes under
    // `modelsDir` to an intermittent ENOTEMPTY.
    await manager.shutdown();
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
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    expect(events.some((event) => event.type === 'done')).toBe(false);
    // The raced-in unowned dir is byte-for-byte intact; no marker written into it.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xcd));
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300, 0xcd));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(false);
    expect(buildState(SLUG)!.installed).toBe(false);
    // No leaked backup was left behind by the refused swap.
    expect(backupDirs()).toEqual([]);
  });

  it('replaces an unowned dir only through a cleanup that echoes its numbers', async () => {
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'notes.txt'), Buffer.alloc(12, 0xab));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const id = manager.start(SLUG, {
      replaceBlocker: { kind: 'directory', fileCount: 1, sizeBytes: 12, truncated: false, linkTarget: null },
    }).id;
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The freshly downloaded (zero-byte) content replaced the manual directory.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12));
    expect(existsSync(join(finalDir(), 'notes.txt'))).toBe(false);
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(buildState(SLUG)!.installed).toBe(true);
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The sweep renamed the backup back into place; the restored (0xAB) content is
    // the completed install, so the job short-circuits without re-downloading.
    expect(existsSync(finalDir())).toBe(true);
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(backupDirs()).toEqual([]);
    expect(hub.downloaded).toEqual([]);
  });

  it('restores an owned dead-owner SYMLINK backup whose final dir is missing, keeping the model present when the follow-up fails', async () => {
    // An owned symlinked finalDir can legitimately end up as a rollback backup:
    // publish admits the link (isDownloaderOwned follows it), then renames the
    // LINK aside before installing staged content. A crash between those renames
    // leaves the backup a symlink into a marked external install — ownership
    // still proves through the link, so the sweep restores it. Every catalog
    // entry carries an assetsRepo, so the restored link's sidecar refresh is
    // refused (a refresh never writes through a live link) and the job takes the
    // full path anyway; the rollback's payoff is that an ERROR mid-download now
    // leaves the canonical path pointing at the complete install instead of
    // reporting the model missing.
    const external = mkdtempSync(join(tmpdir(), 'dash-dl-ext-'));
    writeFileSync(join(external, 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(external, WEIGHT), Buffer.alloc(300, 0xab));
    writeFileSync(
      join(external, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision: hub.sha, files: ['config.json', WEIGHT], completedAt: 'x' }),
    );
    mkdirSync(stagingRoot(), { recursive: true });
    const backup = join(stagingRoot(), `${SLUG}.backup-999999999.55555555-5555-5555-5555-555555555555`);
    symlinkSync(external, backup);
    expect(existsSync(finalDir())).toBe(false);
    hub.failOn = [WEIGHT];

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    // The backup LINK was renamed back onto finalDir before the download failed:
    // the canonical path resolves to the complete external install, and the link
    // itself is intact in .staging only as long as it took to be promoted.
    expect(lstatSync(finalDir()).isSymbolicLink()).toBe(true);
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(backupDirs()).toEqual([]);
    // The error came from the weight fetch — the job genuinely attempted the
    // follow-up download (the refresh-refusal path), not a preflight refuse.
    expect(events.some((event) => event.type === 'done')).toBe(false);

    rmSync(external, { recursive: true, force: true });
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
    await waitFor(() => manager.jobs().some((j) => j.id === id && j.state === 'done'));

    // The published set is exactly the manifest — no stale orphan.
    expect(existsSync(join(finalDir(), 'stale.safetensors'))).toBe(false);
    expect(existsSync(join(finalDir(), 'config.json'))).toBe(true);
    expect(existsSync(join(finalDir(), WEIGHT))).toBe(true);
    expect(buildState(SLUG)!.installed).toBe(true);
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    // the ownership preflight refuses this job (a plain install).
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(finalDir(), WEIGHT), Buffer.alloc(300, 0xab));

    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));

    // The job was refused up front (unowned finalDir) — never listed, never done.
    expect(events.some((event) => event.type === 'start')).toBe(false);
    expect(events.some((event) => event.type === 'done')).toBe(false);
    const err = events.find((event) => event.type === 'error');
    expect(err?.type).toBe('error');
    if (err?.type === 'error') {
      expect(err.message).toContain('not created by the dashboard');
    }

    // …yet the crashed dead-owner staging tree WAS reaped despite the refusal.
    expect(existsSync(deadDir)).toBe(false);

    // The unowned finalDir is byte-for-byte intact — never overwritten, no marker.
    expect(readFileSync(join(finalDir(), 'config.json'))).toEqual(Buffer.alloc(12, 0xab));
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300, 0xab));
    expect(existsSync(join(finalDir(), DOWNLOAD_COMPLETE_MARKER))).toBe(false);
  });

  // An owned-upgrade backup name (`.backup-`) holding a FOREIGN SYMLINK is never
  // promoted: the owned rule restores only what proves owned, and the second
  // preflight refuses a foreign link at finalDir before skip-detection can follow
  // it. A cleanup names its holder `.cleanup-`, which has its own rule.
  it('does NOT restore a foreign-symlink backup onto an absent finalDir, and never reports done through it', async () => {
    const external = mkdtempSync(join(tmpdir(), 'dash-dl-ext-'));
    writeFileSync(join(external, 'config.json'), Buffer.alloc(12, 0xab));
    writeFileSync(join(external, WEIGHT), Buffer.alloc(300, 0xab));
    // NO marker: the target is a FOREIGN install — `isDownloaderOwned` follows the
    // backup link, finds nothing it ever wrote, and the dead-owner backup fails
    // the restore gate even though it parses and its owner is dead.
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
    const id = manager.start(SLUG).id;
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

  it('rejects a folder no catalog build installs to', () => {
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
    const id1 = manager.start(SLUG).id;
    await waitFor(() => blocked);

    expect(manager.start(SLUG).id).toBe(id1);
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
    const idOther = manager.start(SLUG_OTHER).id;
    await waitFor(() => blocked);

    const queued = manager.start(SLUG).id;
    expect(queued).not.toBe(idOther);
    expect(manager.start(SLUG).id).toBe(queued);
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
    const id1 = manager.start(SLUG).id;
    manager.subscribe(id1, (event) => events1.push(event));
    await waitFor(() => events1.some((event) => event.type === 'error'));

    hub.failOn = [];
    const id2 = manager.start(SLUG).id;
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
    const id1 = manager.start(SLUG).id;
    manager.subscribe(id1, (event) => events1.push(event));
    stagingHook.onVerifyWindow = () => {
      manager.cancel(id1);
    };
    await waitFor(() => events1.some((event) => event.type === 'cancelled'));
    await waitFor(() => jobStagingDirs().length === 0);

    const id2 = manager.start(SLUG).id;
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
      const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id1 = manager.start(SLUG).id;
    await waitFor(() => blocked);

    // Job 2 (a DIFFERENT repo — a same-repo request would reuse job 1) queues
    // behind it and is cancelled before it can run.
    const id2 = manager.start(SLUG_OTHER).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id = manager.start(SLUG).id;
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
    const id1 = manager.start(SLUG).id;
    await waitFor(() => blocked);

    // Job 2 (a DIFFERENT repo — a same-repo request would reuse job 1) queues
    // behind it (drain is busy) and has not started.
    const id2 = manager.start(SLUG_OTHER).id;
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
    const id1 = manager.start(SLUG).id;
    await waitFor(() => blocked);

    // `shutdown` flips the manager closed before its first await, so the refusal
    // does not depend on how wide the in-flight job's unwind window happens to be.
    const shutdownP = manager.shutdown();
    expect(() => manager.start(SLUG)).toThrow('shutting down');
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
    const id = manager.start(SLUG).id;
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

describe('Clean up & update: replace the entry in the way after a verified download', () => {
  type DoneEvent = Extract<DownloadEvent, { type: 'done' }>;
  const LEFTOVER = 'imatrix_unsloth.gguf_file';
  const LEFTOVER_BYTES = 183;
  const ONE_FILE: ReplaceBlocker = {
    kind: 'directory',
    fileCount: 1,
    sizeBytes: LEFTOVER_BYTES,
    truncated: false,
    linkTarget: null,
  };
  const DEAD_PID = 999999999;

  function plantLeftover(): void {
    mkdirSync(finalDir(), { recursive: true });
    writeFileSync(join(finalDir(), LEFTOVER), Buffer.alloc(LEFTOVER_BYTES, 0x5c));
  }

  function leftoverIntact(): boolean {
    try {
      return readFileSync(join(finalDir(), LEFTOVER)).equals(Buffer.alloc(LEFTOVER_BYTES, 0x5c));
    } catch {
      return false;
    }
  }

  /**
   * Every entry under `root` by `lstat`, never followed: path, kind, size, mtime and inode. A root folder is named by
   * kind and inode only, since a rename to a new parent may touch its own times.
   */
  function treeOf(root: string): string[] {
    const out: string[] = [];
    const walk = (rel: string): void => {
      const path = rel === '' ? root : join(root, rel);
      const stat = lstatSync(path, { bigint: true });
      const kind = stat.isSymbolicLink() ? 'link' : stat.isDirectory() ? 'dir' : 'file';
      const facts = rel === '' && kind === 'dir' ? `${stat.ino}` : `${stat.size}|${stat.mtimeNs}|${stat.ino}`;
      out.push(`${rel === '' ? '.' : rel}|${kind}|${facts}`);
      if (kind === 'dir') for (const name of readdirSync(path).sort()) walk(rel === '' ? name : `${rel}/${name}`);
    };
    walk('');
    return out;
  }

  function identityOf(path: string): string {
    const stat = lstatSync(path, { bigint: true });
    return `${stat.dev}:${stat.ino}`;
  }

  function trashed(): string[] {
    return readdirSync(trash.dir);
  }

  /** `rm` calls on anything but a job's own staging tree or a file in it: there must never be one. */
  function foreignRm(): string[] {
    const prefix = join(stagingRoot(), `${SLUG}@`);
    return rmCalls.paths.filter((path) => !path.startsWith(prefix));
  }

  function manager(fetchImpl: typeof fetch = makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 })): DownloadManager {
    return new DownloadManager({ modelsDir, cacheDir, fetchImpl });
  }

  async function settled(m: DownloadManager, id: string): Promise<DownloadEvent[]> {
    const events: DownloadEvent[] = [];
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((e) => e.type === 'done' || e.type === 'error' || e.type === 'cancelled'));
    return events;
  }

  function errorOf(events: DownloadEvent[]): string {
    const error = events.find((e) => e.type === 'error');
    return error?.type === 'error' ? error.message : '';
  }

  function doneOf(events: DownloadEvent[]): DoneEvent | undefined {
    return events.find((e): e is DoneEvent => e.type === 'done');
  }

  function slugItem() {
    return buildState(SLUG)!;
  }

  /** A folder outside the models folder for a link to point at, removed when the test ends. */
  function externalFolder(): string {
    const external = mkdtempSync(join(tmpdir(), 'dash-dl-ext-'));
    onTestFinished(() => rmSync(external, { recursive: true, force: true }));
    return external;
  }

  /** A link at the slug to a folder of GGUF shards the app did not make (the owner's most likely card). */
  function plantLinkToShards(): string {
    const external = externalFolder();
    writeFileSync(join(external, 'Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf'), Buffer.alloc(90, 0x21));
    writeFileSync(join(external, 'Qwen3.8-27B-UD-Q4_K_XL-00002-of-00002.gguf'), Buffer.alloc(70, 0x22));
    symlinkSync(external, finalDir());
    return external;
  }

  it('replaces a one-file folder and moves it to the Trash whole, once the download is verified', async () => {
    plantLeftover();
    const tree = treeOf(finalDir());
    expect(slugItem().blocker).toMatchObject(ONE_FILE);
    const m = manager();
    const started = m.start(SLUG, { replaceBlocker: ONE_FILE });
    expect(started.kind).toBe('cleanup');
    expect(m.jobs().find((job) => job.id === started.id)?.kind).toBe('cleanup');
    const events = await settled(m, started.id);
    expect(errorOf(events)).toBe('');
    expect(doneOf(events)).toEqual({ type: 'done', id: started.id, outputDir: finalDir(), removed: 'trash' });
    expect(isModelInstalled(finalDir())).toBe(true);
    expect(trashed()).toEqual([SLUG]);
    expect(treeOf(join(trash.dir, SLUG))).toEqual(tree);
    expect(holderDirs()).toEqual([]);
    expect(backupDirs()).toEqual([]);
    expect(foreignRm()).toEqual([]);
    expect(slugItem()).toMatchObject({ installed: true, blocker: null, blockedByForeignDir: false });
  });

  it.each<[string, () => ReplaceBlocker]>([
    [
      'a folder with model files that Local models lists',
      () => {
        mkdirSync(finalDir());
        writeFileSync(join(finalDir(), 'config.json'), JSON.stringify({ model_type: 'qwen3' }));
        writeFileSync(join(finalDir(), 'model-00001-of-00002.safetensors'), 'valuable-1');
        writeFileSync(join(finalDir(), 'model-00002-of-00002.safetensors'), 'valuable-2');
        return slugItem().blocker!;
      },
    ],
    [
      'a folder of companion GGUFs',
      () => {
        mkdirSync(finalDir());
        writeFileSync(join(finalDir(), 'imatrix_unsloth.gguf'), Buffer.alloc(10));
        writeFileSync(join(finalDir(), 'mmproj-F16.gguf'), Buffer.alloc(20));
        return slugItem().blocker!;
      },
    ],
    [
      'a folder over the size walk budget',
      () => {
        const deep = join(finalDir(), 'a', 'b');
        mkdirSync(deep, { recursive: true });
        for (let i = 0; i < 2_001; i++) writeFileSync(join(deep, `f${i}`), '');
        const blocker = slugItem().blocker!;
        expect(blocker.truncated).toBe(true);
        return blocker;
      },
    ],
    [
      'a junk folder',
      () => {
        mkdirSync(join(finalDir(), 'logs'), { recursive: true });
        writeFileSync(join(finalDir(), 'logs', 'run.txt'), 'log');
        writeFileSync(join(finalDir(), '.DS_Store'), Buffer.alloc(16));
        return slugItem().blocker!;
      },
    ],
    [
      'an empty folder',
      () => {
        mkdirSync(finalDir());
        return slugItem().blocker!;
      },
    ],
    [
      'a file',
      () => {
        writeFileSync(finalDir(), Buffer.alloc(40, 0x1));
        return slugItem().blocker!;
      },
    ],
  ])('installs over %s and moves it to the Trash as one item', async (_name, plant) => {
    const blocker = plant();
    const tree = treeOf(finalDir());
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: blocker }).id);
    expect(errorOf(events)).toBe('');
    expect(doneOf(events)).toMatchObject({ removed: 'trash' });
    expect(doneOf(events)?.keptDir).toBeUndefined();
    expect(isModelInstalled(finalDir())).toBe(true);
    expect(trashed()).toEqual([SLUG]);
    expect(treeOf(join(trash.dir, SLUG))).toEqual(tree);
    expect(trash.calls).toHaveLength(1);
    expect(holderDirs()).toEqual([]);
    expect(foreignRm()).toEqual([]);
  });

  it('removes only a link to a folder of model files, and never writes or moves anything in that folder', async () => {
    const external = plantLinkToShards();
    const tree = treeOf(external);
    const blocker = slugItem().blocker!;
    expect(blocker).toMatchObject({ kind: 'symlink', dangling: false, linkTarget: external, holdsModel: true });
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: blocker }).id);
    expect(errorOf(events)).toBe('');
    expect(doneOf(events)).toMatchObject({ removed: 'link' });
    expect(lstatSync(finalDir()).isDirectory()).toBe(true);
    expect(isModelInstalled(finalDir())).toBe(true);
    expect(treeOf(external)).toEqual(tree);
    expect(existsSync(join(external, MARKER_FILE))).toBe(false);
    expect(trashed()).toEqual([]);
    expect(trash.calls).toEqual([]);
    expect(holderDirs()).toEqual([]);
    expect(foreignRm()).toEqual([]);
  });

  it('removes a dead link as a link', async () => {
    symlinkSync(join(modelsDir, 'nowhere'), finalDir());
    const blocker = slugItem().blocker!;
    expect(blocker).toMatchObject({ kind: 'symlink', dangling: true, fileCount: 0, sizeBytes: 0 });
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: blocker }).id);
    expect(doneOf(events)).toMatchObject({ removed: 'link' });
    expect(lstatSync(finalDir()).isSymbolicLink()).toBe(false);
    expect(isModelInstalled(finalDir())).toBe(true);
    expect(trashed()).toEqual([]);
  });

  it.each<[string, (external: string) => void]>([
    [
      'retargeted',
      (external) => {
        const other = join(dirname(external), `${basename(external)}-other`);
        mkdirSync(other);
        onTestFinished(() => rmSync(other, { recursive: true, force: true }));
        rmSync(finalDir());
        symlinkSync(other, finalDir());
      },
    ],
    [
      'replaced by a folder',
      () => {
        rmSync(finalDir());
        mkdirSync(finalDir());
      },
    ],
  ])('refuses a link that was %s during the download, and leaves it as the user left it', async (_name, change) => {
    const external = plantLinkToShards();
    const blocker = slugItem().blocker!;
    let after = '';
    stagingHook.onVerifyWindow = () => {
      change(external);
      after = identityOf(finalDir());
    };
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: blocker }).id);
    expect(errorOf(events)).toBe(`"${SLUG}" changed on disk during the download. Nothing was removed.`);
    expect(identityOf(finalDir())).toBe(after);
    expect(trashed()).toEqual([]);
    expect(holderDirs()).toEqual([]);
    expect(existsSync(join(finalDir(), MARKER_FILE))).toBe(false);
  });

  it('keeps the entry while the download runs, and after a cancel, for a folder and for a link', async () => {
    for (const plant of [plantLeftover, plantLinkToShards]) {
      rmSync(finalDir(), { recursive: true, force: true });
      plant();
      const identity = identityOf(finalDir());
      const tree = treeOf(finalDir());
      let duringDownload: string | undefined;
      let cancel = (): void => {};
      const m = manager(
        makeCancelFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, () => {
          duringDownload = identityOf(finalDir());
          queueMicrotask(cancel);
        }),
      );
      const id = m.start(SLUG, { replaceBlocker: slugItem().blocker! }).id;
      cancel = () => void m.cancel(id);
      const events = await settled(m, id);
      expect(duringDownload).toBe(identity);
      expect(events.some((e) => e.type === 'cancelled')).toBe(true);
      expect(identityOf(finalDir())).toBe(identity);
      expect(treeOf(finalDir())).toEqual(tree);
      await waitFor(() => jobStagingDirs().length === 0);
      expect(holderDirs()).toEqual([]);
      expect(trashed()).toEqual([]);
    }
  });

  it('says nothing was removed when the download fails, and removes nothing', async () => {
    plantLeftover();
    hub.failOn = [WEIGHT];
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id);
    expect(errorOf(events).endsWith('. Nothing was removed.')).toBe(true);
    expect(leftoverIntact()).toBe(true);
    await waitFor(() => jobStagingDirs().length === 0);
    expect(trashed()).toEqual([]);
  });

  it.each<[string, () => void, ReplaceBlocker]>([
    ['a wrong file count', plantLeftover, { ...ONE_FILE, fileCount: 2 }],
    ['a wrong size', plantLeftover, { ...ONE_FILE, sizeBytes: LEFTOVER_BYTES + 1 }],
    ['a wrong kind', plantLeftover, { ...ONE_FILE, kind: 'file' }],
    ['a path that is gone', () => {}, ONE_FILE],
    [
      'another link text',
      () => symlinkSync('nowhere', finalDir()),
      { kind: 'symlink', fileCount: 0, sizeBytes: 0, truncated: false, linkTarget: 'elsewhere' },
    ],
    [
      'a path the downloader owns',
      () => {
        plantLeftover();
        writeFileSync(
          join(finalDir(), DOWNLOAD_COMPLETE_MARKER),
          JSON.stringify({ repo: REPO, revision: SHA_OLD, files: ['config.json', WEIGHT], completedAt: 'x' }),
        );
      },
      ONE_FILE,
    ],
    [
      'a model Local models lists as present',
      () => {
        mkdirSync(join(finalDir(), 'UD-Q4_K_XL'), { recursive: true });
        writeFileSync(join(finalDir(), 'UD-Q4_K_XL', WEIGHT), qwen35Gguf());
        writeFileSync(join(finalDir(), 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
      },
      { kind: 'directory', fileCount: 2, sizeBytes: 0, truncated: false, linkTarget: null },
    ],
  ])('refuses at the request when the echo does not match: %s', (_name, plant, echo) => {
    plant();
    const before = existsSync(finalDir()) ? treeOf(finalDir()) : null;
    const m = manager();
    expect(() => m.start(SLUG, { replaceBlocker: echo })).toThrow(DownloadConflictError);
    expect(() => m.start(SLUG, { replaceBlocker: echo })).toThrow(
      `"${SLUG}" changed on disk. Check it, then try again.`,
    );
    expect(m.jobs()).toEqual([]);
    expect(existsSync(finalDir()) ? treeOf(finalDir()) : null).toEqual(before);
  });

  function qwen35Gguf(): Buffer {
    const string = (text: string): Buffer => {
      const bytes = Buffer.from(text);
      const length = Buffer.alloc(8);
      length.writeBigUInt64LE(BigInt(bytes.length));
      return Buffer.concat([length, bytes]);
    };
    const header = Buffer.alloc(24);
    header.write('GGUF');
    header.writeUInt32LE(3, 4);
    header.writeBigUInt64LE(1n, 16);
    const stringType = Buffer.alloc(4);
    stringType.writeUInt32LE(8);
    return Buffer.concat([header, string('general.architecture'), stringType, string('qwen35')]);
  }

  it.each<[string, () => void]>([
    [
      'a folder with a GGUF at its root',
      () => {
        mkdirSync(finalDir());
        writeFileSync(join(finalDir(), WEIGHT), qwen35Gguf());
      },
    ],
    [
      'a link to a folder with a GGUF in a variant folder',
      () => {
        const external = externalFolder();
        mkdirSync(join(external, 'UD-Q4_K_XL'));
        writeFileSync(join(external, 'UD-Q4_K_XL', WEIGHT), qwen35Gguf());
        symlinkSync(external, finalDir());
      },
    ],
  ])('refuses Clean up at the request and a plain install at the job for %s that Chat loads', async (_name, plant) => {
    plant();
    expect(await discoverLocalChatModels(modelsDir)).toHaveLength(1);
    // Before this folder counted as present, the card offered Clean up with this summary.
    const asked = summarizeForeignEntry(finalDir())!;
    expect(slugItem()).toMatchObject({ present: true, installed: false, blocker: null });
    const tree = treeOf(finalDir());
    const m = manager();
    expect(() => m.start(SLUG, { replaceBlocker: asked })).toThrow(
      `"${SLUG}" changed on disk. Check it, then try again.`,
    );
    const events = await settled(m, m.start(SLUG).id);
    expect(errorOf(events)).toContain('Refusing to install over');
    expect(treeOf(finalDir())).toEqual(tree);
    expect(trashed()).toEqual([]);
  });

  it.each<[string, number]>([
    ['cannot be read or written', 0o000],
    ['can be read but not written', 0o500],
  ])('refuses at the request a folder the swap cannot move: one that %s', (_name, mode) => {
    plantLeftover();
    chmodSync(finalDir(), mode);
    try {
      const echo = { ...slugItem().blocker! };
      expect(echo.locked).toBe(true);
      const m = manager();
      expect(() => m.start(SLUG, { replaceBlocker: echo })).toThrow(DownloadConflictError);
      expect(() => m.start(SLUG, { replaceBlocker: echo })).toThrow(
        `The app cannot move "${SLUG}": it has no write permission on it. Change its permissions, then try again.`,
      );
      expect(m.jobs()).toEqual([]);
      expect(hub.downloaded).toEqual([]);
    } finally {
      chmodSync(finalDir(), 0o700);
    }
    expect(leftoverIntact()).toBe(true);
  });

  it.each<[string, (m: DownloadManager) => void]>([
    ['after the request', () => writeFileSync(join(finalDir(), 'added.txt'), 'x')],
    [
      'during the download',
      () => (stagingHook.onVerifyWindow = () => writeFileSync(join(finalDir(), 'added.txt'), 'x')),
    ],
    ['right before the swap', () => (raceHook.onMarkerWrite = () => writeFileSync(join(finalDir(), 'added.txt'), 'x'))],
  ])('moves a file added %s to the Trash with the folder: the swap checks identity only', async (_name, add) => {
    plantLeftover();
    const m = manager();
    const id = m.start(SLUG, { replaceBlocker: ONE_FILE }).id;
    add(m);
    const events = await settled(m, id);
    expect(errorOf(events)).toBe('');
    expect(isModelInstalled(finalDir())).toBe(true);
    expect(readdirSync(join(trash.dir, SLUG)).sort()).toEqual(['added.txt', LEFTOVER]);
  });

  it('installs when the entry was removed by hand during the download', async () => {
    plantLeftover();
    stagingHook.onVerifyWindow = () => rmSync(finalDir(), { recursive: true, force: true });
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id);
    expect(errorOf(events)).toBe('');
    expect(isModelInstalled(finalDir())).toBe(true);
    expect(trashed()).toEqual([]);
  });

  it('refuses a folder that another folder replaced during the download, and leaves both', async () => {
    plantLeftover();
    const original = join(modelsDir, 'original');
    stagingHook.onVerifyWindow = () => {
      renameSync(finalDir(), original);
      plantLeftover();
    };
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id);
    expect(errorOf(events)).toBe(`"${SLUG}" changed on disk during the download. Nothing was removed.`);
    expect(leftoverIntact()).toBe(true);
    expect(readdirSync(original)).toEqual([LEFTOVER]);
    expect(trashed()).toEqual([]);
  });

  it('refuses at the publish a folder that replaced the confirmed one, even when it goes before the swap', async () => {
    plantLeftover();
    const original = join(modelsDir, 'original');
    stagingHook.onVerifyWindow = () => {
      renameSync(finalDir(), original);
      plantLeftover();
    };
    raceHook.onMarkerWrite = () => rmSync(finalDir(), { recursive: true, force: true });
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id);
    expect(errorOf(events)).toBe(`"${SLUG}" changed on disk during the download. Nothing was removed.`);
    expect(raceHook.onMarkerWrite).not.toBeNull();
    expect(leftoverIntact()).toBe(true);
    expect(readdirSync(original)).toEqual([LEFTOVER]);
    expect(trashed()).toEqual([]);
  });

  it('refuses before any download a folder that another folder of the same content replaced after the request', async () => {
    plantLeftover();
    const original = join(modelsDir, 'original');
    const m = manager();
    const id = m.start(SLUG, { replaceBlocker: ONE_FILE }).id;
    renameSync(finalDir(), original);
    plantLeftover();
    const events = await settled(m, id);
    expect(errorOf(events)).toBe(`"${SLUG}" changed on disk during the download. Nothing was removed.`);
    expect(hub.downloaded).toEqual([]);
    expect(leftoverIntact()).toBe(true);
    expect(readdirSync(original)).toEqual([LEFTOVER]);
    expect(holderDirs()).toEqual([]);
    expect(trashed()).toEqual([]);
  });

  it('puts the old entry back at its path when the rename of staging fails', async () => {
    plantLeftover();
    const identity = identityOf(finalDir());
    renameFault.failFromPrefix = join(stagingRoot(), `${SLUG}@`);
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id);
    expect(errorOf(events).endsWith('. Nothing was removed.')).toBe(true);
    expect(identityOf(finalDir())).toBe(identity);
    expect(readdirSync(finalDir())).toEqual([LEFTOVER]);
    expect(leftoverIntact()).toBe(true);
    expect(holderDirs()).toEqual([]);
    expect(backupDirs()).toEqual([]);
    expect(trashed()).toEqual([]);
    expect(trash.calls).toEqual([]);
  });

  it('names where the old entry is, without "Nothing was removed", when the way back fails too', async () => {
    plantLeftover();
    renameFault.failFromPrefix = join(stagingRoot(), `${SLUG}@`);
    renameFault.alsoFailFromPrefix = join(stagingRoot(), `${SLUG}.cleanup-`);
    const m = manager();
    const message = errorOf(await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id));
    const [holder] = holderDirs();
    const moved = join(stagingRoot(), holder!, SLUG);
    expect(message).toBe(`The install failed and "${SLUG}" could not be put back. It is now at "${moved}".`);
    expect(readFileSync(join(moved, LEFTOVER))).toEqual(Buffer.alloc(LEFTOVER_BYTES, 0x5c));
    expect(trashed()).toEqual([]);
  });

  it('never puts the old entry back over a file another program wrote at its path after the swap began', async () => {
    writeFileSync(finalDir(), Buffer.alloc(41, 0x1));
    const blocker = slugItem().blocker!;
    renameFault.failFromPrefix = join(stagingRoot(), `${SLUG}@`);
    renameHook.from = finalDir();
    renameHook.after = () => writeFileSync(finalDir(), 'newcomer');
    const m = manager();
    const message = errorOf(await settled(m, m.start(SLUG, { replaceBlocker: blocker }).id));
    const kept = keptDirs();
    expect(kept).toHaveLength(1);
    const keptDir = join(stagingRoot(), kept[0]!);
    expect(message).toBe(
      `"${SLUG}" changed on disk during the download, and something else is now at its path. ` +
        `The old entry is at "${join(keptDir, SLUG)}". Nothing was removed.`,
    );
    expect(readFileSync(finalDir(), 'utf8')).toBe('newcomer');
    expect(readFileSync(join(keptDir, SLUG))).toEqual(Buffer.alloc(41, 0x1));
    expect(trashed()).toEqual([]);
  });

  it('puts the entry back when a folder with another identity took its path before the rename', async () => {
    plantLeftover();
    const original = join(modelsDir, 'original');
    renameHook.from = finalDir();
    renameHook.before = () => {
      renameSync(finalDir(), original);
      plantLeftover();
    };
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id);
    expect(errorOf(events)).toBe(`"${SLUG}" changed on disk during the download. Nothing was removed.`);
    expect(leftoverIntact()).toBe(true);
    expect(readdirSync(original)).toEqual([LEFTOVER]);
    expect(holderDirs()).toEqual([]);
    expect(trashed()).toEqual([]);
  });

  it.each<[string, string[]]>([
    ['a folder with a file', ['other.txt']],
    ['an empty folder', []],
  ])('keeps a changed entry at a named path when %s takes its path after the rename', async (_name, names) => {
    plantLeftover();
    const original = join(modelsDir, 'original');
    renameHook.from = finalDir();
    renameHook.before = () => {
      renameSync(finalDir(), original);
      plantLeftover();
    };
    renameHook.after = () => {
      mkdirSync(finalDir());
      for (const name of names) writeFileSync(join(finalDir(), name), 'other');
    };
    const m = manager();
    const message = errorOf(await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id));
    const kept = keptDirs();
    expect(kept).toHaveLength(1);
    const keptDir = join(stagingRoot(), kept[0]!);
    expect(message).toBe(
      `"${SLUG}" changed on disk during the download, and something else is now at its path. ` +
        `The old entry is at "${join(keptDir, SLUG)}". Nothing was removed.`,
    );
    expect(readFileSync(join(keptDir, SLUG, LEFTOVER))).toEqual(Buffer.alloc(LEFTOVER_BYTES, 0x5c));
    expect(readdirSync(finalDir())).toEqual(names);
    expect(trashed()).toEqual([]);
  });

  it.each<[string, () => void]>([
    ['the Trash refuses', () => (trash.refuse = true)],
    ['the Trash answers success and the path is still there', () => (trash.ignore = true)],
  ])('keeps the old folder in .staging when %s; the model is installed', async (_name, set) => {
    plantLeftover();
    const tree = treeOf(finalDir());
    set();
    const m = manager();
    const events = await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id);
    expect(errorOf(events)).toBe('');
    expect(isModelInstalled(finalDir())).toBe(true);
    const kept = keptDirs();
    expect(kept).toHaveLength(1);
    const keptDir = join(stagingRoot(), kept[0]!);
    const done = doneOf(events)!;
    expect(done.keptDir).toBe(keptDir);
    expect(done.removed).toBeUndefined();
    expect(readdirSync(keptDir)).toEqual([SLUG]);
    expect(treeOf(join(keptDir, SLUG))).toEqual(tree);
    expect(holderDirs()).toEqual([]);
    expect(foreignRm()).toEqual([]);
    // No sweep removes a kept entry.
    const next = manager();
    await settled(next, next.start(SLUG).id);
    expect(treeOf(join(keptDir, SLUG))).toEqual(tree);
  });

  it('refuses a cleanup while a plain job holds the repo, and hands back a running cleanup', async () => {
    const block = makeCancelFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, () => {});
    const plain = manager(block);
    const plainId = plain.start(SLUG).id;
    expect(plain.jobs()[0]?.kind).toBe('install');
    plantLeftover();
    expect(() => plain.start(SLUG, { replaceBlocker: ONE_FILE })).toThrow(
      'A download of this model is already in the queue. Cancel it, then use Clean up & update.',
    );
    plain.cancel(plainId);
    await settled(plain, plainId);

    const m = manager(block);
    const id = m.start(SLUG, { replaceBlocker: ONE_FILE }).id;
    expect(m.start(SLUG, { replaceBlocker: ONE_FILE })).toMatchObject({ id, kind: 'cleanup' });
    m.cancel(id);
    await settled(m, id);
    expect(leftoverIntact()).toBe(true);
  });

  it('compares a truncated walk by kind only', async () => {
    const deep = join(finalDir(), 'a', 'b');
    mkdirSync(deep, { recursive: true });
    for (let i = 0; i < 2_001; i++) writeFileSync(join(deep, `f${i}`), '');
    expect(slugItem().blocker!.truncated).toBe(true);
    const m = manager();
    expect(() =>
      m.start(SLUG, { replaceBlocker: { ...ONE_FILE, fileCount: 7, sizeBytes: 0, truncated: false } }),
    ).toThrow(DownloadConflictError);
    const events = await settled(
      m,
      m.start(SLUG, { replaceBlocker: { ...ONE_FILE, fileCount: 7, sizeBytes: 99, truncated: true } }).id,
    );
    expect(errorOf(events)).toBe('');
    expect(isModelInstalled(finalDir())).toBe(true);
  });

  describe('a crash between moving the old entry into its holder and removing it', () => {
    /** A holder name as a cleanup makes it: owner pid, then the inode of the folder it renames in. */
    function holderPath(pid: number = DEAD_PID, ino: bigint = 1n): string {
      return join(stagingRoot(), `${SLUG}.cleanup-${pid}.${ino}.55555555-5555-5555-5555-555555555555`);
    }
    /** The inode of the folder at the slug: what the holder of the cleanup that put it there names. */
    function finalIno(): bigint {
      return lstatSync(finalDir(), { bigint: true }).ino;
    }
    /** The disk as a cleanup left it when it died right after `rename(finalDir, <holder>/<slug>)`. */
    function crashedAfterFirstRename(pid: number = DEAD_PID): string {
      plantLeftover();
      const holder = holderPath(pid);
      mkdirSync(holder, { recursive: true });
      renameSync(finalDir(), join(holder, SLUG));
      return holder;
    }
    async function installed(): Promise<void> {
      const m = manager();
      expect(errorOf(await settled(m, m.start(SLUG).id))).toBe('');
      expect(isModelInstalled(finalDir())).toBe(true);
    }
    /** A manager whose start-up recovery has run: its job waits for it. */
    async function recovered(): Promise<void> {
      const m = manager();
      await settled(m, m.start(SLUG_OTHER).id);
    }

    it('removes an empty holder', async () => {
      mkdirSync(holderPath(), { recursive: true });
      await recovered();
      expect(existsSync(holderPath())).toBe(false);
    });

    it('puts the old entry back when the path is free, and the card offers the cleanup again', async () => {
      const holder = crashedAfterFirstRename();
      expect(slugItem().blocker).toBeNull();
      await recovered();
      expect(existsSync(holder)).toBe(false);
      expect(leftoverIntact()).toBe(true);
      expect(slugItem().blocker).toMatchObject({ ...ONE_FILE, holdsModel: false });
    });

    it('puts the old entry back before a job for the slug runs, and refuses the plain install', async () => {
      const m = manager();
      await settled(m, m.start(SLUG_OTHER).id);
      const holder = crashedAfterFirstRename();
      const events = await settled(m, m.start(SLUG).id);
      expect(errorOf(events)).toContain('Refusing to install over');
      expect(existsSync(holder)).toBe(false);
      expect(leftoverIntact()).toBe(true);
      expect(jobStagingDirs()).toEqual([]);
    });

    it.each<[string, (holder: string) => void, (holder: string) => void]>([
      [
        'a file',
        (holder) => writeFileSync(join(holder, SLUG), 'old file'),
        (at) => expect(readFileSync(join(at, SLUG), 'utf-8')).toBe('old file'),
      ],
      [
        'a link',
        (holder) => symlinkSync('old-target', join(holder, SLUG)),
        (at) => expect(readlinkSync(join(at, SLUG))).toBe('old-target'),
      ],
    ])('never puts back %s over a file another program wrote at the path meanwhile', async (_name, plant, intact) => {
      const holder = holderPath();
      mkdirSync(holder, { recursive: true });
      plant(holder);
      lstatHook.after = (path) => {
        if (path !== join(holder, SLUG)) return;
        lstatHook.after = null;
        writeFileSync(finalDir(), 'newcomer');
      };
      const error = vi.spyOn(console, 'error').mockImplementation(() => {});
      onTestFinished(() => error.mockRestore());
      await recovered();
      expect(readFileSync(finalDir(), 'utf-8')).toBe('newcomer');
      const kept = keptDirs();
      expect(kept).toHaveLength(1);
      intact(join(stagingRoot(), kept[0]!));
      expect(existsSync(holder)).toBe(false);
      expect(error.mock.calls.some((call) => String(call[0]).includes(join(stagingRoot(), kept[0]!)))).toBe(true);
    });

    it('finishes the cleanup when the folder at the path is the one it renamed in: a folder goes to the Trash', async () => {
      await installed();
      const holder = holderPath(DEAD_PID, finalIno());
      mkdirSync(join(holder, SLUG, 'sub'), { recursive: true });
      writeFileSync(join(holder, SLUG, 'sub', LEFTOVER), 'old');
      const tree = treeOf(join(holder, SLUG));
      await recovered();
      expect(existsSync(holder)).toBe(false);
      expect(treeOf(join(trash.dir, SLUG))).toEqual(tree);
      expect(isModelInstalled(finalDir())).toBe(true);
    });

    it('finishes the cleanup when the folder at the path is the one it renamed in: a link is unlinked, its folder untouched', async () => {
      await installed();
      const external = externalFolder();
      writeFileSync(join(external, 'keep.gguf'), Buffer.alloc(64, 0x33));
      const tree = treeOf(external);
      const holder = holderPath(DEAD_PID, finalIno());
      mkdirSync(holder, { recursive: true });
      symlinkSync(external, join(holder, SLUG));
      await recovered();
      expect(existsSync(holder)).toBe(false);
      expect(treeOf(external)).toEqual(tree);
      expect(trashed()).toEqual([]);
    });

    it.each<[string, (holder: string) => void]>([
      [
        'a folder',
        (holder) => {
          mkdirSync(join(holder, SLUG));
          writeFileSync(join(holder, SLUG, LEFTOVER), 'old');
        },
      ],
      ['a link', (holder) => symlinkSync('old-target', join(holder, SLUG))],
    ])(
      'keeps %s, never removed, when the model at the path is not the one this cleanup installed',
      async (_name, plant) => {
        await installed();
        const holder = holderPath(DEAD_PID, finalIno() + 1n);
        mkdirSync(holder, { recursive: true });
        plant(holder);
        const tree = treeOf(join(holder, SLUG));
        const error = vi.spyOn(console, 'error').mockImplementation(() => {});
        onTestFinished(() => error.mockRestore());
        await recovered();
        const kept = keptDirs();
        expect(kept).toHaveLength(1);
        expect(treeOf(join(stagingRoot(), kept[0]!, SLUG))).toEqual(tree);
        expect(trashed()).toEqual([]);
        expect(trash.calls).toEqual([]);
        expect(error.mock.calls.some((call) => String(call[0]).includes(join(stagingRoot(), kept[0]!)))).toBe(true);
        expect(isModelInstalled(finalDir())).toBe(true);
      },
    );

    it('keeps the old entry when the Trash refuses during recovery', async () => {
      await installed();
      const holder = holderPath(DEAD_PID, finalIno());
      mkdirSync(join(holder, SLUG), { recursive: true });
      writeFileSync(join(holder, SLUG, LEFTOVER), 'old');
      trash.refuse = true;
      await recovered();
      expect(existsSync(holder)).toBe(false);
      const kept = keptDirs();
      expect(kept).toHaveLength(1);
      expect(readFileSync(join(stagingRoot(), kept[0]!, SLUG, LEFTOVER), 'utf-8')).toBe('old');
    });

    it('leaves everything when something else is at the path', async () => {
      mkdirSync(finalDir());
      writeFileSync(join(finalDir(), 'someone-else.txt'), 'x');
      const holder = holderPath();
      mkdirSync(join(holder, SLUG), { recursive: true });
      writeFileSync(join(holder, SLUG, LEFTOVER), 'old');
      await recovered();
      expect(readFileSync(join(holder, SLUG, LEFTOVER), 'utf-8')).toBe('old');
      expect(readdirSync(finalDir())).toEqual(['someone-else.txt']);
      expect(trashed()).toEqual([]);
    });

    it('leaves the holder of a live owner alone', async () => {
      expect(pidAlive(process.ppid)).toBe(true);
      const holder = crashedAfterFirstRename(process.ppid);
      await recovered();
      expect(readFileSync(join(holder, SLUG, LEFTOVER))).toEqual(Buffer.alloc(LEFTOVER_BYTES, 0x5c));
      expect(existsSync(finalDir())).toBe(false);
    });

    it('settles a holder with its own pid at manager start: no job of this process exists yet', async () => {
      const holder = crashedAfterFirstRename(process.pid);
      await recovered();
      expect(existsSync(holder)).toBe(false);
      expect(leftoverIntact()).toBe(true);
    });
  });

  it('never calls rm on an entry the app did not create, in the job or in recovery', async () => {
    const layouts: Array<() => ReplaceBlocker> = [
      () => {
        plantLeftover();
        return ONE_FILE;
      },
      () => {
        writeFileSync(finalDir(), Buffer.alloc(40, 0x1));
        return slugItem().blocker!;
      },
      () => {
        plantLinkToShards();
        return slugItem().blocker!;
      },
      () => {
        symlinkSync(join(modelsDir, 'nowhere'), finalDir());
        return slugItem().blocker!;
      },
      // The folder gains the app's record during the download: it still goes to the Trash whole.
      () => {
        plantLeftover();
        stagingHook.onVerifyWindow = () => writeMarker(finalDir(), SHA_OLD);
        return ONE_FILE;
      },
    ];
    for (const [index, plant] of layouts.entries()) {
      rmSync(finalDir(), { recursive: true, force: true });
      const blocker = plant();
      trash.refuse = index === 0;
      const m = manager();
      const events = await settled(m, m.start(SLUG, { replaceBlocker: blocker }).id);
      expect(errorOf(events)).toBe('');
      await waitFor(() => jobStagingDirs().length === 0);
    }
    trash.refuse = false;
    for (const pid of [DEAD_PID]) {
      const ino = lstatSync(finalDir(), { bigint: true }).ino;
      const holder = join(stagingRoot(), `${SLUG}.cleanup-${pid}.${ino}.88888888-8888-8888-8888-888888888888`);
      mkdirSync(join(holder, SLUG), { recursive: true });
      writeFileSync(join(holder, SLUG, LEFTOVER), 'old');
      const m = manager();
      await settled(m, m.start(SLUG).id);
      expect(existsSync(holder)).toBe(false);
    }
    expect(rmCalls.paths.length).toBeGreaterThan(0);
    expect(foreignRm()).toEqual([]);
  });

  /** The app's record in `dir`, of this repo at `revision`, listing the files of a plain install. */
  function writeMarker(dir: string, revision: string): void {
    writeFileSync(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: REPO, revision, files: ['config.json', WEIGHT], completedAt: 'x' }),
    );
  }

  describe('an entry that gains the app record during the job', () => {
    it('moves a folder that gained the record during the download to the Trash whole', async () => {
      plantLeftover();
      writeFileSync(join(finalDir(), 'notes.txt'), 'my notes');
      const m = manager();
      const id = m.start(SLUG, { replaceBlocker: slugItem().blocker! }).id;
      stagingHook.onVerifyWindow = () => writeMarker(finalDir(), SHA_OLD);
      const events = await settled(m, id);
      expect(errorOf(events)).toBe('');
      expect(doneOf(events)).toMatchObject({ removed: 'trash' });
      expect(trash.calls).toHaveLength(1);
      expect(trash.calls[0]!.startsWith(join(stagingRoot(), `${SLUG}.cleanup-`))).toBe(true);
      expect(readFileSync(join(trash.dir, SLUG, 'notes.txt'), 'utf-8')).toBe('my notes');
      expect(isModelInstalled(finalDir())).toBe(true);
      expect(foreignRm()).toEqual([]);
    });

    it('refuses a folder with the record that replaced the confirmed one, and removes nothing', async () => {
      plantLeftover();
      const original = join(modelsDir, 'original');
      const m = manager();
      const id = m.start(SLUG, { replaceBlocker: ONE_FILE }).id;
      stagingHook.onVerifyWindow = () => {
        renameSync(finalDir(), original);
        plantLeftover();
        writeMarker(finalDir(), SHA_OLD);
      };
      const events = await settled(m, id);
      expect(errorOf(events)).toBe(`"${SLUG}" changed on disk during the download. Nothing was removed.`);
      expect(readdirSync(finalDir()).sort()).toEqual([DOWNLOAD_COMPLETE_MARKER, LEFTOVER].sort());
      expect(leftoverIntact()).toBe(true);
      expect(readdirSync(original)).toEqual([LEFTOVER]);
      expect(trashed()).toEqual([]);
      expect(foreignRm()).toEqual([]);
    });

    it('refuses before any download a folder with the record that replaced the confirmed one after the request', async () => {
      plantLeftover();
      const original = join(modelsDir, 'original');
      const m = manager();
      const id = m.start(SLUG, { replaceBlocker: ONE_FILE }).id;
      renameSync(finalDir(), original);
      plantLeftover();
      writeMarker(finalDir(), SHA_OLD);
      const events = await settled(m, id);
      expect(errorOf(events)).toBe(`"${SLUG}" changed on disk during the download. Nothing was removed.`);
      expect(hub.downloaded).toEqual([]);
      expect(leftoverIntact()).toBe(true);
      expect(trashed()).toEqual([]);
    });

    it('replaces a folder that became a complete install of this revision after the request', async () => {
      const plain = manager();
      expect(errorOf(await settled(plain, plain.start(SLUG).id))).toBe('');
      // Outside the models folder: a marked folder there would read as this build installed elsewhere.
      const install = join(externalFolder(), 'install');
      renameSync(finalDir(), install);
      plantLeftover();
      const m = manager();
      const id = m.start(SLUG, { replaceBlocker: ONE_FILE }).id;
      cpSync(install, finalDir(), { recursive: true });
      expect(isModelInstalled(finalDir())).toBe(true);
      const events = await settled(m, id);
      expect(errorOf(events)).toBe('');
      expect(doneOf(events)).toMatchObject({ removed: 'trash' });
      expect(readdirSync(join(trash.dir, SLUG))).toContain(LEFTOVER);
      expect(isModelInstalled(finalDir())).toBe(true);
    });

    it('removes only the link when its folder gains the record during the download', async () => {
      const external = plantLinkToShards();
      const blocker = slugItem().blocker!;
      const m = manager();
      const id = m.start(SLUG, { replaceBlocker: blocker }).id;
      let tree: string[] = [];
      stagingHook.onVerifyWindow = () => {
        writeMarker(external, SHA_OLD);
        tree = treeOf(external);
      };
      const events = await settled(m, id);
      expect(errorOf(events)).toBe('');
      expect(doneOf(events)).toMatchObject({ removed: 'link' });
      expect(lstatSync(finalDir()).isDirectory()).toBe(true);
      expect(treeOf(external)).toEqual(tree);
      expect(existsSync(join(external, DOWNLOAD_COMPLETE_MARKER))).toBe(true);
      expect(trash.calls).toEqual([]);
      expect(foreignRm()).toEqual([]);
    });
  });

  describe('what the Trash step reports', () => {
    it('keeps the old folder when the entry cannot be read after a refused Trash', async () => {
      plantLeftover();
      const tree = treeOf(finalDir());
      onTestFinished(() => {
        if (!existsSync(stagingRoot())) return;
        for (const name of readdirSync(stagingRoot())) chmodSync(join(stagingRoot(), name), 0o700);
      });
      trash.before = (path) => chmodSync(dirname(path), 0o000);
      trash.refuse = true;
      const m = manager();
      const events = await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id);
      for (const name of readdirSync(stagingRoot())) chmodSync(join(stagingRoot(), name), 0o700);
      const done = doneOf(events)!;
      expect(done.removed).toBeUndefined();
      expect(done.keptDir).toBeDefined();
      expect(treeOf(join(done.keptDir!, SLUG))).toEqual(tree);
      expect(isModelInstalled(finalDir())).toBe(true);
    });

    it('names the holder when the holder is gone after a refused Trash', async () => {
      plantLeftover();
      const away = join(modelsDir, 'away');
      let holder = '';
      trash.before = (path) => {
        holder = dirname(path);
        renameSync(holder, away);
      };
      trash.refuse = true;
      const m = manager();
      const events = await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id);
      const done = doneOf(events)!;
      expect(done.removed).toBeUndefined();
      expect(done.keptDir).toBe(holder);
      expect(readFileSync(join(away, SLUG, LEFTOVER))).toEqual(Buffer.alloc(LEFTOVER_BYTES, 0x5c));
    });

    it('never renames a folder that took the holder path during a refused Trash', async () => {
      plantLeftover();
      const away = join(modelsDir, 'away');
      let holder = '';
      trash.before = (path) => {
        holder = dirname(path);
        renameSync(holder, away);
        mkdirSync(holder);
      };
      trash.refuse = true;
      const m = manager();
      const events = await settled(m, m.start(SLUG, { replaceBlocker: ONE_FILE }).id);
      const done = doneOf(events)!;
      expect(done.removed).toBeUndefined();
      expect(done.keptDir).toBe(holder);
      expect(keptDirs()).toEqual([]);
      expect(readdirSync(holder)).toEqual([]);
      expect(readFileSync(join(away, SLUG, LEFTOVER))).toEqual(Buffer.alloc(LEFTOVER_BYTES, 0x5c));
    });
  });

  it('keeps what a done job did in its row of the job list, with the kind of the cleaned entry', async () => {
    plantLeftover();
    trash.refuse = true;
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    onTestFinished(() => error.mockRestore());
    const m = manager();
    const id = m.start(SLUG, { replaceBlocker: ONE_FILE }).id;
    const done = doneOf(await settled(m, id))!;
    expect(done.keptDir).toBeDefined();
    // The core log names the kept folder too.
    expect(error.mock.calls.some((call) => String(call[0]).includes(done.keptDir!))).toBe(true);
    expect(m.jobs()).toEqual([
      expect.objectContaining({
        id,
        state: 'done',
        keptDir: done.keptDir,
        entry: { kind: 'directory', linkToFile: false },
      }),
    ]);
    expect(m.jobs()[0]!.removed).toBeUndefined();

    rmSync(finalDir(), { recursive: true, force: true });
    m.cancel(id);
    trash.refuse = false;
    plantLeftover();
    const next = m.start(SLUG, { replaceBlocker: ONE_FILE }).id;
    await settled(m, next);
    expect(m.jobs().find((job) => job.id === next)).toMatchObject({ removed: 'trash' });
    expect(m.jobs().find((job) => job.id === next)?.keptDir).toBeUndefined();
  });

  describe('the way back of a file or a link', () => {
    it.each<[string, () => void, (at: string) => void]>([
      [
        'a file',
        () => writeFileSync(finalDir(), Buffer.alloc(41, 0x1)),
        (at) => expect(readFileSync(at)).toEqual(Buffer.alloc(41, 0x1)),
      ],
      [
        'a link',
        () => symlinkSync(join(modelsDir, 'nowhere'), finalDir()),
        (at) => expect(readlinkSync(at)).toBe(join(modelsDir, 'nowhere')),
      ],
    ])('never replaces a file that lands at the path while %s is put back', async (_name, plant, intact) => {
      plant();
      const blocker = slugItem().blocker!;
      renameFault.failFromPrefix = join(stagingRoot(), `${SLUG}@`);
      lstatHook.after = (path) => {
        if (!path.includes(`${SLUG}.cleanup-`) || basename(path) !== SLUG) return;
        lstatHook.after = null;
        writeFileSync(finalDir(), 'newcomer');
      };
      const m = manager();
      const message = errorOf(await settled(m, m.start(SLUG, { replaceBlocker: blocker }).id));
      expect(readFileSync(finalDir(), 'utf-8')).toBe('newcomer');
      const kept = keptDirs();
      expect(kept).toHaveLength(1);
      const keptDir = join(stagingRoot(), kept[0]!);
      intact(join(keptDir, SLUG));
      expect(message).toBe(
        `"${SLUG}" changed on disk during the download, and something else is now at its path. ` +
          `The old entry is at "${join(keptDir, SLUG)}". Nothing was removed.`,
      );
      expect(trashed()).toEqual([]);
    });

    it.each<[string, () => void, (at: string) => void]>([
      [
        'a file',
        () => writeFileSync(finalDir(), Buffer.alloc(41, 0x1)),
        (at) => expect(readFileSync(at)).toEqual(Buffer.alloc(41, 0x1)),
      ],
      [
        'a link',
        () => symlinkSync(join(modelsDir, 'nowhere'), finalDir()),
        (at) => expect(readlinkSync(at)).toBe(join(modelsDir, 'nowhere')),
      ],
    ])(
      'says nothing was removed when %s is back but its name in the holder cannot be unlinked',
      async (_name, plant, intact) => {
        plant();
        const blocker = slugItem().blocker!;
        renameFault.failFromPrefix = join(stagingRoot(), `${SLUG}@`);
        unlinkHook.before = (path) => {
          if (path.includes(`${SLUG}.cleanup-`) && basename(path) === SLUG) {
            throw Object.assign(new Error('simulated unlink failure'), { code: 'EPERM' });
          }
        };
        const m = manager();
        const message = errorOf(await settled(m, m.start(SLUG, { replaceBlocker: blocker }).id));
        intact(finalDir());
        expect(message).not.toContain('could not be put back');
        expect(message.endsWith('. Nothing was removed.')).toBe(true);
        expect(keptDirs()).toEqual([]);
        expect(trashed()).toEqual([]);
      },
    );
  });

  describe('through the route', () => {
    function post(m: DownloadManager, body: unknown): ReturnType<typeof dispatch> {
      const ctx = { downloads: m, modelsDir } as unknown as ApiContext;
      return dispatch(ctx, { method: 'POST', pathname: '/api/downloads', query: new URLSearchParams(), body });
    }

    it.each<[string, unknown]>([
      ['a string', 'directory'],
      ['an unknown kind', { ...ONE_FILE, kind: 'folder' }],
      ['a negative count', { ...ONE_FILE, fileCount: -1 }],
      ['a fractional size', { ...ONE_FILE, sizeBytes: 1.5 }],
      ['a non-boolean truncated', { ...ONE_FILE, truncated: 'no' }],
      ['no link text', { kind: 'directory', fileCount: 1, sizeBytes: LEFTOVER_BYTES, truncated: false }],
      ['a link text of 202 characters', { ...ONE_FILE, kind: 'symlink', linkTarget: 'x'.repeat(202) }],
    ])('answers 400 for %s', async (_name, replaceBlocker) => {
      plantLeftover();
      const m = manager();
      const res = await post(m, { slug: SLUG, replaceBlocker });
      expect(res.status).toBe(400);
      expect(res.ok ? '' : res.message).toBe(
        '"replaceBlocker" must be { kind: "directory" | "symlink" | "file", fileCount, sizeBytes, truncated, linkTarget }',
      );
      expect(m.jobs()).toEqual([]);
    });

    it('ignores a path and a repo in the body: the target is the catalog folder', async () => {
      plantLeftover();
      const elsewhere = join(modelsDir, 'elsewhere');
      mkdirSync(elsewhere);
      writeFileSync(join(elsewhere, 'keep'), 'k');
      const m = manager();
      const res = await post(m, { slug: SLUG, replaceBlocker: ONE_FILE, path: elsewhere, repo: 'someone/elsewhere' });
      expect(res.status).toBe(202);
      expect(res.ok ? res.body : null).toMatchObject({ slug: SLUG, repo: REPO, kind: 'cleanup' });
      await settled(m, (res.ok ? (res.body as { id: string }) : { id: '' }).id);
      expect(readFileSync(join(elsewhere, 'keep'), 'utf-8')).toBe('k');
      expect(isModelInstalled(finalDir())).toBe(true);
    });

    it('maps a conflict to 409 with the gate text and no reason', async () => {
      plantLeftover();
      const res = await post(manager(), { slug: SLUG, replaceBlocker: { ...ONE_FILE, fileCount: 3 } });
      expect(res.status).toBe(409);
      expect(res.ok ? '' : res.message).toBe(`"${SLUG}" changed on disk. Check it, then try again.`);
      expect('reason' in res).toBe(false);
    });

    it('runs the whole flow: catalog blocker, POST with the five fields, done, installed', async () => {
      const external = plantLinkToShards();
      const ctx = { downloads: manager(), modelsDir } as unknown as ApiContext;
      const get = (pathname: string) => dispatch(ctx, { method: 'GET', pathname, query: new URLSearchParams() });
      type Build = { slug: string; listed: boolean; blocker: ReplaceBlocker & { locked: boolean } };
      const buildOf = async (): Promise<Build> => {
        const res = await get('/api/catalog');
        return (res.ok ? (res.body as { items: Array<{ builds: Build[] }> }) : { items: [] }).items
          .flatMap((entry) => entry.builds)
          .find((build) => build.slug === SLUG)!;
      };
      const before = await buildOf();
      expect(before).toMatchObject({
        listed: false,
        blocker: { kind: 'symlink', linkTarget: external, locked: false },
      });
      const { kind, fileCount, sizeBytes, truncated, linkTarget } = before.blocker;
      const res = await post(ctx.downloads as DownloadManager, {
        slug: SLUG,
        replaceBlocker: { kind, fileCount, sizeBytes, truncated, linkTarget },
      });
      expect(res.status).toBe(202);
      const events = await settled(
        ctx.downloads as DownloadManager,
        (res.ok ? (res.body as { id: string }) : { id: '' }).id,
      );
      expect(doneOf(events)).toMatchObject({ removed: 'link' });
      expect(await buildOf()).toMatchObject({ installed: true, blocker: null });
    });
  });
});

describe('owned update of a folder that Update adopted', () => {
  type DoneEvent = Extract<DownloadEvent, { type: 'done' }>;

  /** The `adopted` record entry of a file as it is on disk now. */
  function recordOf(path: string): { size: number; mtimeNs: string } {
    const stat = lstatSync(path, { bigint: true });
    return { size: Number(stat.size), mtimeNs: stat.mtimeNs.toString() };
  }

  /**
   * A folder Update adopted: `listed` files the marker names with a record taken now (or `adopted` when given), plus
   * `extra` files that only the user put there.
   */
  function writeAdoptedFolder(
    dir: string,
    listed: Record<string, Buffer>,
    opts: { extra?: Record<string, string>; adopted?: unknown; revision?: string; marker?: object } = {},
  ): void {
    mkdirSync(dir, { recursive: true });
    const record: Record<string, { size: number; mtimeNs: string }> = {};
    for (const [path, bytes] of Object.entries(listed)) {
      mkdirSync(dirname(join(dir, path)), { recursive: true });
      writeFileSync(join(dir, path), bytes);
      record[path] = recordOf(join(dir, path));
    }
    for (const [path, text] of Object.entries(opts.extra ?? {})) writeFileSync(join(dir, path), text);
    writeFileSync(
      join(dir, MARKER_FILE),
      JSON.stringify({
        repo: REPO,
        revision: opts.revision ?? SHA_OLD,
        files: Object.keys(listed),
        scope: 'full',
        completedAt: 'x',
        adopted: opts.adopted ?? record,
        ...opts.marker,
      }),
    );
  }

  const OLD_FILES = (): Record<string, Buffer> => ({
    'config.json': Buffer.alloc(12, 0xab),
    [WEIGHT]: Buffer.alloc(300, 0xab),
  });

  /** `.staging/<slug>.kept-<uuid>` folders, by name. */
  function keptDirs(): string[] {
    return existsSync(stagingRoot()) ? readdirSync(stagingRoot()).filter((n) => n.startsWith(`${SLUG}.kept-`)) : [];
  }

  async function runJob(): Promise<DoneEvent> {
    const manager = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300, 'tokenizer.json': 20 }),
    });
    const events: DownloadEvent[] = [];
    const id = manager.start(SLUG).id;
    manager.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));
    await manager.shutdown();
    expect(events.find((event) => event.type === 'error')).toBeUndefined();
    return events.find((event): event is DoneEvent => event.type === 'done')!;
  }

  it('deletes the files that still match the record and leaves no kept folder', async () => {
    writeAdoptedFolder(finalDir(), { ...OLD_FILES(), 'sub/extra.json': Buffer.from('{}') });

    const done = await runJob();

    expect(done.keptDir).toBeUndefined();
    expect(keptDirs()).toEqual([]);
    expect(backupDirs()).toEqual([]);
    expect(hub.downloaded).toContain(WEIGHT);
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300));
    expect(existsSync(join(finalDir(), 'sub'))).toBe(false);
  });

  it('keeps a file the user added and names the kept folder on done', async () => {
    writeAdoptedFolder(finalDir(), OLD_FILES(), { extra: { 'README.md': '# my notes' } });

    const done = await runJob();

    expect(keptDirs()).toHaveLength(1);
    const kept = join(stagingRoot(), keptDirs()[0]!);
    expect(done.keptDir).toBe(kept);
    expect(readdirSync(kept).sort()).toEqual([MARKER_FILE, 'README.md'].sort());
    expect(readFileSync(join(kept, 'README.md'), 'utf-8')).toBe('# my notes');
    expect(existsSync(join(finalDir(), 'README.md'))).toBe(false);
    expect(backupDirs()).toEqual([]);
  });

  it('keeps a listed file that was rewritten in place with the same size', async () => {
    writeAdoptedFolder(finalDir(), OLD_FILES());
    const rewritten = Buffer.alloc(300, 0xcd);
    writeFileSync(join(finalDir(), WEIGHT), rewritten);
    utimesSync(join(finalDir(), WEIGHT), 1_000_000, 1_000_000);

    const done = await runJob();

    const kept = join(stagingRoot(), keptDirs()[0] ?? 'none');
    expect(done.keptDir).toBe(kept);
    expect(readdirSync(kept).sort()).toEqual([MARKER_FILE, WEIGHT].sort());
    expect(readFileSync(join(kept, WEIGHT))).toEqual(rewritten);
  });

  it('keeps everything when the record is malformed', async () => {
    writeAdoptedFolder(finalDir(), OLD_FILES(), { adopted: { 'config.json': { size: '12', mtimeNs: '1' } } });

    const done = await runJob();

    const kept = join(stagingRoot(), keptDirs()[0] ?? 'none');
    expect(done.keptDir).toBe(kept);
    expect(readdirSync(kept).sort()).toEqual([MARKER_FILE, 'config.json', WEIGHT].sort());
    expect(readFileSync(join(kept, WEIGHT))).toEqual(Buffer.alloc(300, 0xab));
  });

  it('a second pass over a backup that could not be renamed keeps the same files', async () => {
    writeAdoptedFolder(finalDir(), OLD_FILES(), { extra: { 'README.md': '# my notes' } });
    renameFault.failToPrefix = join(stagingRoot(), `${SLUG}.kept-`);

    const done = await runJob();

    expect(keptDirs()).toEqual([]);
    expect(backupDirs()).toHaveLength(1);
    const backup = join(stagingRoot(), backupDirs()[0]!);
    expect(done.keptDir).toBe(backup);
    expect(readdirSync(backup).sort()).toEqual([MARKER_FILE, 'README.md'].sort());

    // As if the app had died: the backup's owner pid is gone, so the next job's recovery takes it.
    renameFault.failToPrefix = null;
    renameSync(backup, join(stagingRoot(), `${SLUG}.backup-999999999.33333333-3333-3333-3333-333333333333`));
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    onTestFinished(() => errors.mockRestore());

    await runJob();

    expect(backupDirs()).toEqual([]);
    const kept = join(stagingRoot(), keptDirs()[0] ?? 'none');
    expect(readdirSync(kept).sort()).toEqual([MARKER_FILE, 'README.md'].sort());
    expect(readFileSync(join(kept, 'README.md'), 'utf-8')).toBe('# my notes');
  });

  it('recovery of a dead-pid backup uses the same rule', async () => {
    await runJob();
    const backup = join(stagingRoot(), `${SLUG}.backup-999999999.44444444-4444-4444-4444-444444444444`);
    writeAdoptedFolder(backup, OLD_FILES(), { extra: { 'README.md': '# my notes' } });
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    onTestFinished(() => errors.mockRestore());

    await runJob();

    expect(existsSync(backup)).toBe(false);
    const kept = join(stagingRoot(), keptDirs()[0] ?? 'none');
    expect(readdirSync(kept).sort()).toEqual([MARKER_FILE, 'README.md'].sort());
    expect(errors.mock.calls.some((call) => String(call[0]).includes(kept))).toBe(true);
  });

  it('a stale sidecar in an adopted folder is not deleted in place', async () => {
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'config.json', size: 12 }];
    hub.shaByRepo[ASSETS_REPO] = SHA_NEW;
    const users = Buffer.alloc(20, 0x75);
    writeAdoptedFolder(
      finalDir(),
      { [WEIGHT]: Buffer.alloc(300), 'config.json': Buffer.alloc(12), 'tokenizer.json': users },
      { revision: hub.sha, marker: { assetsRepo: ASSETS_REPO, assetsRevision: SHA_NEW } },
    );
    // The user rewrote the file after Update, so its record no longer proves it.
    utimesSync(join(finalDir(), 'tokenizer.json'), 1_000_000, 1_000_000);

    const done = await runJob();

    expect(hub.downloaded).toContain(WEIGHT);
    expect(existsSync(join(finalDir(), 'tokenizer.json'))).toBe(false);
    const kept = join(stagingRoot(), keptDirs()[0] ?? 'none');
    expect(done.keptDir).toBe(kept);
    expect(readFileSync(join(kept, 'tokenizer.json'))).toEqual(users);
  });

  it('a file renamed onto a proven name between the check and the removal is kept and named', async () => {
    writeAdoptedFolder(finalDir(), OLD_FILES(), { extra: { 'README.md': '# my notes' } });
    lstatHook.after = (path) => {
      if (!path.includes(`${SLUG}.backup-`) || !path.endsWith(`/${WEIGHT}`)) return;
      lstatHook.after = null;
      renameSync(join(dirname(path), 'README.md'), path);
    };

    const done = await runJob();

    const kept = join(stagingRoot(), keptDirs()[0] ?? 'none');
    expect(done.keptDir).toBe(kept);
    expect(readdirSync(kept).sort()).toEqual([MARKER_FILE, WEIGHT].sort());
    expect(readFileSync(join(kept, WEIGHT), 'utf-8')).toBe('# my notes');
  });

  it('a file that cannot be put back at its name stays in the kept folder', async () => {
    writeAdoptedFolder(finalDir(), OLD_FILES(), { extra: { 'README.md': '# my notes' } });
    let backupWeight = '';
    lstatHook.after = (path) => {
      if (!path.includes(`${SLUG}.backup-`) || !path.endsWith(`/${WEIGHT}`)) return;
      lstatHook.after = null;
      backupWeight = path;
      renameSync(join(dirname(path), 'README.md'), path);
    };
    linkHook.before = (to) => {
      if (backupWeight !== '' && to === backupWeight) writeFileSync(to, 'later');
    };

    const done = await runJob();

    const kept = join(stagingRoot(), keptDirs()[0] ?? 'none');
    expect(done.keptDir).toBe(kept);
    const files = (readdirSync(kept, { recursive: true }) as string[]).filter((rel) =>
      lstatSync(join(kept, rel)).isFile(),
    );
    const texts = files.map((rel) => readFileSync(join(kept, rel), 'utf-8'));
    expect(texts).toContain('# my notes');
    expect(texts).toContain('later');
  });

  it('a backup whose marker cannot be read is kept whole', async () => {
    await runJob();
    const backup = join(stagingRoot(), `${SLUG}.backup-999999999.55555555-5555-5555-5555-555555555555`);
    writeAdoptedFolder(backup, OLD_FILES(), { extra: { 'README.md': '# my notes' } });
    writeFileSync(join(backup, MARKER_FILE), '{"truncated');
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    onTestFinished(() => errors.mockRestore());

    await runJob();

    expect(existsSync(backup)).toBe(false);
    const kept = join(stagingRoot(), keptDirs()[0] ?? 'none');
    expect(readdirSync(kept).sort()).toEqual([MARKER_FILE, 'README.md', 'config.json', WEIGHT].sort());
    expect(readFileSync(join(kept, 'README.md'), 'utf-8')).toBe('# my notes');
  });

  it('an install without the record is removed as before', async () => {
    writeAdoptedFolder(finalDir(), OLD_FILES(), {
      extra: { 'README.md': '# my notes' },
      marker: { adopted: undefined },
    });

    const done = await runJob();

    expect(done.keptDir).toBeUndefined();
    expect(keptDirs()).toEqual([]);
    expect(backupDirs()).toEqual([]);
    expect(existsSync(join(finalDir(), 'README.md'))).toBe(false);
  });
});

const FLASH_BUILD = catalogBuild('Clairic Flash (9B)', 'UD-Q4_K_XL');
const FLASH = catalogBuildRepo(FLASH_BUILD);
const FLASH_ASSETS = FLASH_BUILD.assetsRepo!;
const FLASH_SLUG = catalogBuildSlug(FLASH_BUILD);
const FLASH_WEIGHT = 'Clairic-Flash-UD-Q4_K_XL.gguf';
const ASSETS_SHA = SHA_NEW;

/** The files of the user's folder, the five the catalog wants from the GGUF repo. */
const USER_FILES: Record<string, Buffer> = {
  [FLASH_WEIGHT]: Buffer.alloc(64, 0x11),
  'mmproj-BF16.gguf': Buffer.alloc(32, 0x22),
  'clair_head.safetensors': Buffer.alloc(48, 0x33),
  'clair_head_config.json': Buffer.from('{"head":1}'),
  'preprocessor_config.json': Buffer.from('{"size":448}'),
};
/** The files of the assets repo, in its listing order. */
const ASSET_FILES: Record<string, Buffer> = {
  'clairic_runtime/api.py': Buffer.from('pass'),
  LICENSE: Buffer.from('license'),
  'chat_template.jinja': Buffer.from('{{ messages }}'),
  'config.json': Buffer.from('{"model_type":"clairic"}'),
  'generation_config.json': Buffer.from('{}'),
  'merges.txt': Buffer.from('a b'),
  'preprocessor_config.json': Buffer.from('{"size":448}'),
  'tokenizer.json': Buffer.alloc(40, 0x44),
  'tokenizer_config.json': Buffer.from('{"t":1}'),
  'video_preprocessor_config.json': Buffer.from('{"v":1}'),
};
const LFS = new Set([FLASH_WEIGHT, 'mmproj-BF16.gguf', 'clair_head.safetensors', 'tokenizer.json']);

/** A manifest entry for `bytes` as Hugging Face lists it: LFS (with Xet) for weights, a git blob for small files. */
function entryOf(path: string, bytes: Buffer): ManifestEntry {
  const blob = createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
  if (!LFS.has(path)) return { type: 'file', path, size: bytes.length, oid: blob };
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  return {
    type: 'file',
    path,
    size: bytes.length,
    oid: blob,
    lfs: { oid: sha256, size: bytes.length, pointerSize: 134 },
    xetHash: sha256.slice(0, 32),
  };
}

/** The two repos the catalog entry reads, shaped like their real listings, and the user's folder. */
function flashFolder(): string {
  hub.manifests[FLASH] = [
    entryOf('.gitattributes', Buffer.from('*.gguf filter=lfs')),
    ...Object.entries(USER_FILES).map(([path, bytes]) => entryOf(path, bytes)),
    entryOf('README.md', Buffer.from('# Clairic Flash')),
  ];
  hub.manifests[FLASH_ASSETS] = [
    { type: 'directory', path: 'clairic_runtime', size: 0 },
    ...Object.entries(ASSET_FILES).map(([path, bytes]) => entryOf(path, bytes)),
  ];
  hub.shaByRepo[FLASH_ASSETS] = ASSETS_SHA;
  const dir = join(modelsDir, FLASH_SLUG);
  mkdirSync(dir, { recursive: true });
  for (const [path, bytes] of Object.entries(USER_FILES)) writeFileSync(join(dir, path), bytes);
  writeFileSync(join(dir, 'README.md'), 'notes');
  return dir;
}

describe('Update: plan for a folder the app did not create', () => {
  function manager(opts: { fetchImpl?: typeof fetch; probeTimeoutMs?: number } = {}): DownloadManager {
    return new DownloadManager({ modelsDir, cacheDir, ...opts });
  }

  /**
   * The text of the plan's refusal, which must be a conflict with the reason `reason`: `needs-cleanup` for a folder
   * that a cleanup can replace, none for a refusal that passes or that a cleanup cannot solve.
   */
  async function refusal(m: DownloadManager, reason: 'needs-cleanup' | 'none' = 'needs-cleanup'): Promise<string> {
    const error = await m.planAdoption(FLASH_SLUG).then(
      () => null,
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(DownloadConflictError);
    expect((error as DownloadConflictError).reason).toBe(reason === 'none' ? undefined : reason);
    return (error as Error).message;
  }

  it("plans a folder like the user's: 5 to check, 6 to download, README.md stays", async () => {
    flashFolder();
    const plan = await manager().planAdoption(FLASH_SLUG);
    const sizes = (paths: string[]): number =>
      paths.reduce(
        (sum, path) =>
          sum + (hub.manifests[FLASH]!.concat(hub.manifests[FLASH_ASSETS]!).find((e) => e.path === path)?.size ?? 0),
        0,
      );
    expect(plan).toEqual({
      repo: FLASH,
      slug: FLASH_SLUG,
      token: plan.token,
      revision: hub.sha,
      assetsRevision: ASSETS_SHA,
      check: { count: 5, bytes: sizes(Object.keys(USER_FILES)) },
      download: {
        count: 6,
        bytes: sizes([
          'config.json',
          'tokenizer.json',
          'tokenizer_config.json',
          'chat_template.jinja',
          'generation_config.json',
          'video_preprocessor_config.json',
        ]),
      },
      other: { count: 1, bytes: 5, names: ['README.md'] },
    });
    expect(plan.token.length).toBe(64);
    expect(plan.token.split('').every((ch) => (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))).toBe(true);
    expect(hub.revisionsByRepo[FLASH]).toEqual([hub.sha]);
    expect(hub.revisionsByRepo[FLASH_ASSETS]).toEqual([ASSETS_SHA]);
    expect(hub.downloaded).toEqual([]);
  });

  it('reads no file content', async () => {
    const dir = flashFolder();
    const paths = [...Object.keys(USER_FILES), 'README.md'].map((path) => join(dir, path));
    for (const path of paths) chmodSync(path, 0o000);
    const plan = await manager()
      .planAdoption(FLASH_SLUG)
      .finally(() => {
        for (const path of paths) chmodSync(path, 0o644);
      });
    expect(plan.check.count).toBe(5);
    expect(plan.other.names).toEqual(['README.md']);
  });

  it('refuses a file with the catalog name and another size, and names it', async () => {
    const dir = flashFolder();
    writeFileSync(join(dir, FLASH_WEIGHT), Buffer.alloc(65, 0x11));
    expect(await refusal(manager())).toBe(
      `"${FLASH_WEIGHT}" in "${FLASH_SLUG}" is not the catalog file. Its size is different. ` +
        `Move it out of the folder, then use Update again. Update then downloads the catalog file (64 bytes).`,
    );
  });

  it('refuses a catalog file that has no checksum in the manifest', async () => {
    flashFolder();
    hub.manifests[FLASH] = hub.manifests[FLASH]!.map((entry) =>
      entry.path === 'clair_head_config.json' ? { type: 'file', path: entry.path, size: entry.size } : entry,
    );
    const m = manager();
    expect(await refusal(m)).toBe(
      `The catalog gives no checksum for "clair_head_config.json", so Update cannot check it. ` +
        `Move it out of the folder, then use Update again.`,
    );
  });

  it('refuses a missing catalog file that has no checksum in the manifest, and names it', async () => {
    flashFolder();
    hub.manifests[FLASH_ASSETS] = hub.manifests[FLASH_ASSETS]!.map((entry) =>
      entry.path === 'tokenizer.json' ? { type: 'file', path: entry.path, size: entry.size } : entry,
    );
    expect(await refusal(manager())).toBe(
      `The catalog gives no checksum for "tokenizer.json", so Update cannot check what it downloads.`,
    );
  });

  it('refuses an extra model file', async () => {
    const dir = flashFolder();
    writeFileSync(join(dir, 'Clairic-Flash-Q8_0.gguf'), 'q8');
    expect(await refusal(manager())).toBe(
      `"Clairic-Flash-Q8_0.gguf" in "${FLASH_SLUG}" is a model file that is not part of the catalog model. ` +
        `Move it out of the folder, then use Update again.`,
    );
  });

  it('refuses a name that differs only in letter case', async () => {
    const dir = flashFolder();
    rmSync(join(dir, 'clair_head_config.json'));
    writeFileSync(join(dir, 'Clair_Head_Config.json'), USER_FILES['clair_head_config.json']!);
    expect(await refusal(manager())).toBe(
      `"Clair_Head_Config.json" in "${FLASH_SLUG}" differs from the catalog name "clair_head_config.json" only in ` +
        `letter case. Rename it, then use Update again.`,
    );
  });

  it.each<[string, (dir: string) => void, string]>([
    ['a link', (dir) => symlinkSync(join(dir, 'README.md'), join(dir, 'notes.md')), 'holds links or special files'],
    ['a special file', (dir) => execFileSync('/usr/bin/mkfifo', [join(dir, 'pipe')]), 'holds links or special files'],
    [
      'more than 2,000 entries',
      (dir) => {
        mkdirSync(join(dir, 'pages'));
        for (let n = 0; n < 1_995; n++) writeFileSync(join(dir, 'pages', `${n}.txt`), '');
      },
      'holds more than 2,000 files',
    ],
    ['a folder with no write permission', (dir) => chmodSync(dir, 0o555), 'no write permission'],
    [
      'an owned folder',
      (dir) =>
        writeFileSync(
          join(dir, MARKER_FILE),
          JSON.stringify({ repo: FLASH, revision: SHA_OLD, files: Object.keys(USER_FILES), completedAt: 'x' }),
        ),
      'changed on disk',
    ],
    [
      'a linked slug',
      (dir) => {
        const elsewhere = join(modelsDir, 'elsewhere');
        renameSync(dir, elsewhere);
        symlinkSync(elsewhere, dir);
      },
      'changed on disk',
    ],
    ['an unreadable marker file', (dir) => writeFileSync(join(dir, MARKER_FILE), '{'), 'cannot read'],
  ])('refuses %s', async (_name, plant, words) => {
    const dir = flashFolder();
    plant(dir);
    const texts: Record<string, string> = {
      'holds links or special files': `"${FLASH_SLUG}" holds links or special files. Update cannot check it.`,
      'holds more than 2,000 files': `"${FLASH_SLUG}" holds more than 2,000 files. Update cannot check it.`,
      'no write permission':
        `The app cannot add files to "${FLASH_SLUG}": it has no write permission on it. ` +
        `Change its permissions, then try again.`,
      'changed on disk': `"${FLASH_SLUG}" changed on disk. Check it, then try again.`,
      'cannot read':
        `"${FLASH_SLUG}" holds a download record that the app cannot read (${MARKER_FILE}). ` +
        `Move it out of the folder, then use Update again.`,
    };
    const reason = words === 'no write permission' || words === 'changed on disk' ? 'none' : 'needs-cleanup';
    try {
      expect(await refusal(manager(), reason)).toBe(texts[words]);
    } finally {
      if (lstatSync(dir).isDirectory()) chmodSync(dir, 0o755);
    }
  });

  it('plans a folder of exactly 2,000 entries', async () => {
    const dir = flashFolder();
    mkdirSync(join(dir, 'pages'));
    // 6 files and the folder itself are already 7 entries.
    for (let n = 0; n < 1_993; n++) writeFileSync(join(dir, 'pages', `${n}.txt`), '');
    expect((await manager().planAdoption(FLASH_SLUG)).other.count).toBe(1_994);
  });

  it('a new .DS_Store does not change the token; a touched check file does', async () => {
    const dir = flashFolder();
    const m = manager();
    const first = await m.planAdoption(FLASH_SLUG);
    writeFileSync(join(dir, '.DS_Store'), Buffer.alloc(6148));
    const second = await m.planAdoption(FLASH_SLUG);
    expect(second.other.names).toEqual(['.DS_Store', 'README.md']);
    expect(second.token).toBe(first.token);
    utimesSync(join(dir, FLASH_WEIGHT), 1_000_000, 1_000_000);
    expect((await m.planAdoption(FLASH_SLUG)).token).not.toBe(first.token);
  });

  it('uses the raw fetch with a deadline while another job downloads', async () => {
    flashFolder();
    hub.modelInfoUsesFetch = true;
    const stalling = makeStallingFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, 100);
    let cancelJob = (): void => {};
    const fetchImpl: typeof fetch = (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      if (!url.endsWith(`/api/models/${FLASH}`)) return stalling(input, init);
      // The plan's metadata read hangs; the running job is cancelled while it waits.
      return new Promise<Response>((_resolve, reject) => {
        const signal = init!.signal!;
        signal.addEventListener('abort', () => reject(signal.reason as Error));
        cancelJob();
      });
    };
    const m = manager({ fetchImpl, probeTimeoutMs: 200 });
    const events: DownloadEvent[] = [];
    const id = m.start(SLUG).id;
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'progress' && event.file === WEIGHT));
    const received = m.jobs().find((job) => job.id === id)!.receivedBytes;
    cancelJob = () => {
      expect(m.jobs().find((job) => job.id === id)!.receivedBytes).toBe(received);
      m.cancel(id);
    };

    const error = await m.planAdoption(FLASH_SLUG).then(
      () => null,
      (e: unknown) => e as Error,
    );

    expect(error?.message).toBe('Could not check the folder: The operation was aborted due to timeout');
    await waitFor(() => events.some((event) => event.type === 'cancelled'));
  });

  it('says it could not check the folder when Hugging Face does not answer', async () => {
    flashFolder();
    hub.modelInfoError = 'getaddrinfo ENOTFOUND huggingface.co';
    await expect(manager().planAdoption(FLASH_SLUG)).rejects.toThrow(
      'Could not check the folder: getaddrinfo ENOTFOUND huggingface.co',
    );
  });

  it('answers 409 while a job holds the repo', async () => {
    flashFolder();
    const m = manager({ fetchImpl: makeStallingFetch({ 'config.json': 12, [WEIGHT]: 300 }, WEIGHT, 100) });
    const running = m.start(SLUG).id;
    const events: DownloadEvent[] = [];
    m.subscribe(running, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'progress'));
    m.start(FLASH_SLUG);
    expect(await refusal(m, 'none')).toBe(
      'A download of this model is already in the queue. Cancel it, then use Update.',
    );
    await m.shutdown();
  });

  it('refuses a disk that cannot link files before it reads the network, and leaves no test file', async () => {
    const dir = flashFolder();
    const before = readdirSync(dir).sort();
    linkHook.probe = () => {
      throw Object.assign(new Error('operation not supported'), { code: 'ENOTSUP' });
    };
    expect(await refusal(manager())).toBe(
      `This disk cannot link files, so Update cannot add files to "${FLASH_SLUG}".`,
    );
    expect(hub.modelInfoRepos).toEqual([]);
    expect(readdirSync(stagingRoot())).toEqual([]);
    expect(readdirSync(dir).sort()).toEqual(before);
  });

  it('refuses in a plain sentence when the app cannot write its link test file', async () => {
    flashFolder();
    mkdirSync(stagingRoot(), { recursive: true });
    chmodSync(stagingRoot(), 0o500);
    try {
      expect(await refusal(manager(), 'none')).toBe(
        `The app cannot write in the models folder, so Update cannot add files to "${FLASH_SLUG}".`,
      );
    } finally {
      chmodSync(stagingRoot(), 0o700);
    }
    expect(hub.modelInfoRepos).toEqual([]);
    expect(readdirSync(stagingRoot())).toEqual([]);
  });

  it('sends the reason beside the 409 text through the route, and none for a refusal that passes', async () => {
    const dir = flashFolder();
    writeFileSync(join(dir, 'Clairic-Flash-Q8_0.gguf'), 'q8');
    const ctx = { downloads: manager(), modelsDir } as unknown as ApiContext;
    const plan = () =>
      dispatch(ctx, {
        method: 'POST',
        pathname: '/api/downloads/adopt-plan',
        query: new URLSearchParams(),
        body: { slug: FLASH_SLUG },
      });
    const refused = await plan();
    expect(refused).toMatchObject({ ok: false, status: 409, code: 'E_CONFLICT', reason: 'needs-cleanup' });
    rmSync(join(dir, 'Clairic-Flash-Q8_0.gguf'));
    chmodSync(dir, 0o555);
    try {
      const locked = await plan();
      expect(locked).toMatchObject({ ok: false, status: 409 });
      expect('reason' in locked).toBe(false);
    } finally {
      chmodSync(dir, 0o755);
    }
  });
});

describe('Update: proof in place', () => {
  const NAME = 'weights.gguf';
  const BYTES = Buffer.alloc(1024 * 1024, 0x5a);

  function lfs(bytes: Buffer, path = NAME): ManifestEntry {
    const sha256 = createHash('sha256').update(bytes).digest('hex');
    return {
      type: 'file',
      path,
      size: bytes.length,
      oid: 'f'.repeat(40),
      lfs: { oid: sha256, size: bytes.length, pointerSize: 134 },
    };
  }

  function expected(path: string): { size: number; mtimeNs: bigint; ino: bigint } {
    const stat = lstatSync(path, { bigint: true });
    return { size: Number(stat.size), mtimeNs: stat.mtimeNs, ino: stat.ino };
  }

  function folder(): string {
    const dir = join(modelsDir, 'user-folder');
    mkdirSync(dir);
    writeFileSync(join(dir, NAME), BYTES);
    return dir;
  }

  const quiet = { cancelled: () => false, onBytes: () => {} };

  it('proves a file by sha256 and by git-blob sha1', async () => {
    const dir = folder();
    const config = Buffer.from('{"model_type":"clairic"}');
    writeFileSync(join(dir, 'config.json'), config);
    const blob: ManifestEntry = {
      type: 'file',
      path: 'config.json',
      size: config.length,
      oid: createHash('sha1').update(`blob ${config.length}\0`).update(config).digest('hex'),
    };

    const weights = await proveInPlace(dir, lfs(BYTES) as ListFileEntry, expected(join(dir, NAME)), quiet);
    const small = await proveInPlace(dir, blob as ListFileEntry, expected(join(dir, 'config.json')), quiet);
    const wrong = await proveInPlace(
      dir,
      lfs(Buffer.alloc(BYTES.length, 0x5b)) as ListFileEntry,
      expected(join(dir, NAME)),
      quiet,
    );

    expect(weights.match).toBe(true);
    expect(weights.fingerprint.ino).toBe(lstatSync(join(dir, NAME), { bigint: true }).ino);
    expect(small.match).toBe(true);
    expect(wrong.match).toBe(false);
  });

  it('does not follow a link', async () => {
    const dir = folder();
    symlinkSync(join(dir, NAME), join(dir, 'linked.gguf'));
    const error = await proveInPlace(
      dir,
      lfs(BYTES, 'linked.gguf') as ListFileEntry,
      expected(join(dir, NAME)),
      quiet,
    ).then(
      () => null,
      (e: unknown) => e as NodeJS.ErrnoException,
    );
    expect(error?.code).toBe('ELOOP');
  });

  it('refuses a file that is not the one the plan saw', async () => {
    const dir = folder();
    const plan = expected(join(dir, NAME));
    utimesSync(join(dir, NAME), 1_000_000, 1_000_000);
    await expect(proveInPlace(dir, lfs(BYTES) as ListFileEntry, plan, quiet)).rejects.toThrow(
      `"${NAME}" in "user-folder" changed during the check. Nothing was changed. Use Update again.`,
    );
  });

  it('fails when the file is rewritten with the same size during the read', async () => {
    const dir = folder();
    // Only the first chunk differs, and it is read before the rewrite, so the bytes read hash to the catalog file.
    const rewritten = Buffer.from(BYTES);
    rewritten[0] = 0x00;
    let first = true;
    const hooks = {
      cancelled: () => false,
      onBytes: () => {
        if (!first) return;
        first = false;
        writeFileSync(join(dir, NAME), rewritten);
      },
    };
    await expect(proveInPlace(dir, lfs(BYTES) as ListFileEntry, expected(join(dir, NAME)), hooks)).rejects.toThrow(
      `"${NAME}" in "user-folder" changed during the check. Nothing was changed. Use Update again.`,
    );
  });

  it('stops inside a file when cancelled', async () => {
    const dir = folder();
    let read = 0;
    const hooks = { cancelled: () => read > 0, onBytes: (n: number) => void (read += n) };
    await expect(proveInPlace(dir, lfs(BYTES) as ListFileEntry, expected(join(dir, NAME)), hooks)).rejects.toThrow(
      'Download cancelled',
    );
    expect(read).toBeGreaterThan(0);
    expect(read).toBeLessThan(BYTES.length);
  });

  it('reports every byte once', async () => {
    const dir = folder();
    const chunks: number[] = [];
    const hooks = { cancelled: () => false, onBytes: (n: number) => void chunks.push(n) };
    const proof = await proveInPlace(dir, lfs(BYTES) as ListFileEntry, expected(join(dir, NAME)), hooks);
    expect(proof.match).toBe(true);
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.reduce((sum, n) => sum + n, 0)).toBe(BYTES.length);
  });
});

describe('Update: adopt a folder the app did not create, in place', () => {
  /** What the job downloads for the user's folder, in the order it adds them: config.json last. */
  const DOWNLOADED = [
    'tokenizer.json',
    'tokenizer_config.json',
    'chat_template.jinja',
    'generation_config.json',
    'video_preprocessor_config.json',
    'config.json',
  ];
  const BYTES: Record<string, Buffer> = { ...ASSET_FILES, ...USER_FILES };
  let fetched: string[] = [];

  beforeEach(() => {
    fetched = [];
  });

  /**
   * A fetch that serves each file's own bytes, so its download verifies, and logs the path. `hold` may answer a path
   * itself, for a download that waits.
   */
  function manager(hold?: (path: string, init?: RequestInit) => Promise<Response> | undefined): DownloadManager {
    const fetchImpl: typeof fetch = async (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      const path = url.split('/').slice(3).join('/');
      fetched.push(path);
      const held = hold?.(path, init);
      if (held !== undefined) return held;
      return new Response(new Uint8Array(BYTES[path] ?? Buffer.alloc(0)), { status: 200 });
    };
    return new DownloadManager({ modelsDir, cacheDir, fetchImpl });
  }

  /** A download that only ends when the job's fetch is aborted. */
  function waitForAbort(init: RequestInit | undefined): Promise<Response> {
    return new Promise<Response>((_resolve, reject) => {
      const abort = (): void => reject(new DOMException('The operation was aborted', 'AbortError'));
      if (init?.signal?.aborted) abort();
      else init?.signal?.addEventListener('abort', abort);
    });
  }

  interface FileState {
    ino: bigint;
    mtimeNs: bigint;
    bytes: Buffer;
  }

  /** Every file under `dir` by path, with its inode, mtime and bytes. */
  function filesOf(dir: string): Map<string, FileState> {
    const out = new Map<string, FileState>();
    for (const rel of readdirSync(dir, { recursive: true }) as string[]) {
      const stat = lstatSync(join(dir, rel), { bigint: true });
      if (stat.isFile()) out.set(rel, { ino: stat.ino, mtimeNs: stat.mtimeNs, bytes: readFileSync(join(dir, rel)) });
    }
    return out;
  }

  /** Every file that was in the folder is still there at the same inode, with the same mtime and bytes. */
  function expectKept(before: Map<string, FileState>, dir: string): void {
    const now = filesOf(dir);
    for (const [path, state] of before) expect(now.get(path), path).toEqual(state);
  }

  /** The names the folder has now that it did not have in `before`. */
  function addedNames(before: Map<string, FileState>, dir: string): string[] {
    return [...filesOf(dir).keys()].filter((path) => !before.has(path)).sort();
  }

  function flashStagingDirs(): string[] {
    return existsSync(stagingRoot()) ? readdirSync(stagingRoot()).filter((n) => n.startsWith(`${FLASH_SLUG}@`)) : [];
  }

  function errorOf(events: DownloadEvent[]): string | undefined {
    const error = events.find((event) => event.type === 'error');
    return error?.type === 'error' ? error.message : undefined;
  }

  /**
   * Plan an Update, run `between` (a change after the confirm), start the job with the plan's token, and wait for its
   * last event. `onEvent` sees every event as it comes.
   */
  async function runUpdate(
    m: DownloadManager,
    opts: { between?: () => void; onEvent?: (event: DownloadEvent, id: string) => void } = {},
  ): Promise<{ id: string; events: DownloadEvent[]; last: DownloadEvent }> {
    const plan = await m.planAdoption(FLASH_SLUG);
    opts.between?.();
    const id = m.start(FLASH_SLUG, { adopt: plan.token }).id;
    const events: DownloadEvent[] = [];
    m.subscribe(id, (event) => {
      events.push(event);
      opts.onEvent?.(event, id);
    });
    await waitFor(() =>
      events.some((event) => event.type === 'done' || event.type === 'error' || event.type === 'cancelled'),
    );
    return { id, events, last: events.at(-1)! };
  }

  it('keeps every checked file at its inode, adds only the missing files, writes the marker last, and the catalog says installed', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    const order: string[] = [];
    linkHook.before = (to) => void order.push(to.slice(dir.length + 1));
    const m = manager();

    const { last } = await runUpdate(m);
    await m.shutdown();

    expect(last).toEqual({ type: 'done', id: last.id, outputDir: dir });
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([...DOWNLOADED, MARKER_FILE].sort());
    expect(order).toEqual([...DOWNLOADED, MARKER_FILE]);
    expect(fetched.sort()).toEqual([...DOWNLOADED].sort());
    for (const path of DOWNLOADED) {
      expect(readFileSync(join(dir, path))).toEqual(ASSET_FILES[path]);
      // The staging tree is gone, so the folder holds the only name of each added file.
      expect(lstatSync(join(dir, path), { bigint: true }).nlink).toBe(1n);
    }
    expect(flashStagingDirs()).toEqual([]);
    expect(isModelInstalled(dir)).toBe(true);
    expect(buildState(FLASH_SLUG)).toMatchObject({
      installed: true,
      blocker: null,
    });
  });

  it('a crash at any add leaves a folder the app does not own', async () => {
    const dir = flashFolder();
    const ownedAtAdd: Array<[string, boolean]> = [];
    linkHook.before = (to) => {
      const path = to.slice(dir.length + 1);
      if (path !== MARKER_FILE) ownedAtAdd.push([path, existsSync(join(dir, MARKER_FILE))]);
    };
    const m = manager();

    const { last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('done');
    expect(ownedAtAdd).toEqual(DOWNLOADED.map((path) => [path, false]));
  });

  it('leaves README.md in place and records every listed file in adopted', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    const m = manager();

    expect((await runUpdate(m)).last.type).toBe('done');
    await m.shutdown();

    expectKept(before, dir);
    expect(readFileSync(join(dir, 'README.md'), 'utf-8')).toBe('notes');
    const marker = JSON.parse(readFileSync(join(dir, MARKER_FILE), 'utf-8')) as {
      repo: string;
      revision: string;
      files: string[];
      scope: string;
      assetsRepo: string;
      assetsRevision: string;
      adopted: Record<string, { size: number; mtimeNs: string }>;
    };
    const wanted = [...Object.keys(USER_FILES), ...DOWNLOADED].sort();
    expect(marker).toMatchObject({
      repo: FLASH,
      revision: hub.sha,
      scope: 'full',
      assetsRepo: FLASH_ASSETS,
      assetsRevision: ASSETS_SHA,
    });
    expect([...marker.files].sort()).toEqual(wanted);
    expect(Object.keys(marker.adopted).sort()).toEqual(wanted);
    for (const path of wanted) {
      const stat = lstatSync(join(dir, path), { bigint: true });
      expect(marker.adopted[path], path).toEqual({ size: Number(stat.size), mtimeNs: stat.mtimeNs.toString() });
    }
  });

  it('stops and changes nothing when a same-size file is not the catalog file; the next plan refuses it without a new hash', async () => {
    const dir = flashFolder();
    writeFileSync(join(dir, 'clair_head.safetensors'), Buffer.alloc(48, 0x34));
    const before = filesOf(dir);
    const m = manager();

    const { events, last } = await runUpdate(m);

    expect(last.type).toBe('error');
    expect(errorOf(events)).toBe(
      `"clair_head.safetensors" in "${FLASH_SLUG}" is not the catalog file. Nothing was changed. ` +
        `Move it out of the folder, then use Update again.`,
    );
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([]);
    expect(fetched).toEqual([]);
    expect(flashStagingDirs()).toEqual([]);
    // No file is read while planning, so the refusal comes from what the job's hash found.
    chmodSync(join(dir, 'clair_head.safetensors'), 0o000);
    const refused = await m.planAdoption(FLASH_SLUG).then(
      () => null,
      (e: unknown) => e as Error,
    );
    chmodSync(join(dir, 'clair_head.safetensors'), 0o644);
    expect(refused).toBeInstanceOf(DownloadConflictError);
    expect((refused as DownloadConflictError).reason).toBe('needs-cleanup');
    expect(refused?.message).toBe(
      `"clair_head.safetensors" in "${FLASH_SLUG}" is not the catalog file. Move it out of the folder, then use Update again.`,
    );
    await m.shutdown();
  });

  it('downloads a catalog file that is missing from the folder', async () => {
    const dir = flashFolder();
    rmSync(join(dir, FLASH_WEIGHT));
    const before = filesOf(dir);
    const m = manager();
    expect((await m.planAdoption(FLASH_SLUG)).download.count).toBe(7);

    const { last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('done');
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([...DOWNLOADED, FLASH_WEIGHT, MARKER_FILE].sort());
    expect(hub.downloadedFrom).toContain(`${FLASH}/${FLASH_WEIGHT}`);
    expect(readFileSync(join(dir, FLASH_WEIGHT))).toEqual(USER_FILES[FLASH_WEIGHT]);
    expect(isModelInstalled(dir)).toBe(true);
  });

  it('refuses a token that is not the last plan (409) and queues nothing', async () => {
    const dir = flashFolder();
    const m = manager();
    const first = await m.planAdoption(FLASH_SLUG);
    const refusedText = `"${FLASH_SLUG}" changed on disk. Close this, then use Update again.`;

    expect(() => m.start(FLASH_SLUG, { adopt: 'f'.repeat(64) })).toThrow(refusedText);
    utimesSync(join(dir, FLASH_WEIGHT), 1_000_000, 1_000_000);
    const second = await m.planAdoption(FLASH_SLUG);
    expect(second.token).not.toBe(first.token);
    expect(() => m.start(FLASH_SLUG, { adopt: first.token })).toThrow(DownloadConflictError);
    expect(() => m.start(FLASH_SLUG, { adopt: first.token })).toThrow(refusedText);
    expect(m.jobs()).toEqual([]);
    expect(existsSync(join(dir, MARKER_FILE))).toBe(false);

    const id = m.start(FLASH_SLUG, { adopt: second.token }).id;
    const events: DownloadEvent[] = [];
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done'));
    // A token starts one job only.
    expect(() => m.start(FLASH_SLUG, { adopt: second.token })).toThrow(refusedText);
    await m.shutdown();
  });

  it.each<[string, (dir: string) => void]>([
    ['upstream moved', () => void (hub.sha = SHA_NEW)],
    ['the assets repo moved', () => void (hub.shaByRepo[FLASH_ASSETS] = SHA_A)],
    ['a check file was touched', (dir) => utimesSync(join(dir, 'clair_head.safetensors'), 1_000_000, 1_000_000)],
  ])('fails at job start when %s after the confirm', async (_name, change) => {
    const dir = flashFolder();
    let before = new Map<string, FileState>();
    const m = manager();

    const { events, last } = await runUpdate(m, {
      between: () => {
        change(dir);
        before = filesOf(dir);
      },
    });
    await m.shutdown();

    expect(last.type).toBe('error');
    expect(errorOf(events)).toBe(
      `"${FLASH_SLUG}" or the catalog changed after you confirmed. Nothing was changed. Use Update again.`,
    );
    expect(events.some((event) => event.type === 'start')).toBe(false);
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([]);
    expect(fetched).toEqual([]);
  });

  it('fails at the commit when a check file was touched after its hash', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    const m = manager((path) => {
      if (path === 'tokenizer.json') utimesSync(join(dir, 'clair_head_config.json'), 1_000_000, 1_000_000);
      return undefined;
    });

    const { events, last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('error');
    expect(errorOf(events)).toBe(`"${FLASH_SLUG}" changed on disk during the update. Nothing was changed.`);
    expect(addedNames(before, dir)).toEqual([]);
    expect(readFileSync(join(dir, 'clair_head_config.json'))).toEqual(USER_FILES['clair_head_config.json']);
    expect(lstatSync(join(dir, 'clair_head_config.json'), { bigint: true }).ino).toBe(
      before.get('clair_head_config.json')!.ino,
    );
    expect(flashStagingDirs()).toEqual([]);
  });

  it('an other file that goes away during the commit scan does not stop the Update', async () => {
    const dir = flashFolder();
    const notes = join(dir, 'notes.txt');
    writeFileSync(notes, 'temp');
    let scans = 0;
    // The 1st lstat is the plan's scan, the 2nd the job's own plan, the 3rd the commit scan: the file is gone after
    // the scan listed it and before its lstat.
    lstatHook.before = (path) => {
      if (path === notes && ++scans === 3) rmSync(notes);
    };
    const m = manager();

    const { last } = await runUpdate(m);
    await m.shutdown();

    expect(scans).toBe(3);
    expect(last).toEqual({ type: 'done', id: last.id, outputDir: dir });
    expect(isModelInstalled(dir)).toBe(true);
  });

  it('a check file that goes away during the commit scan still fails the Update', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    const head = join(dir, 'clair_head_config.json');
    const kept = join(modelsDir, 'clair_head_config.json');
    let scans = 0;
    lstatHook.before = (path) => {
      if (path === head && ++scans === 3) renameSync(head, kept);
    };
    const m = manager();

    const { events, last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('error');
    expect(errorOf(events)).toBe(`"${FLASH_SLUG}" changed on disk during the update. Nothing was changed.`);
    renameSync(kept, head);
    expect(addedNames(before, dir)).toEqual([]);
    expect(existsSync(join(dir, MARKER_FILE))).toBe(false);
  });

  it('a file that appears at a path the job adds makes link fail, and the job removes what it added', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    const mine = Buffer.from('{"model_type":"mine"}');
    // Lands after the commit scan, while the marker is written into staging.
    raceHook.onMarkerWrite = () => writeFileSync(join(dir, 'config.json'), mine);
    const m = manager();

    const { events, last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('error');
    expect(errorOf(events)).toBe(`"config.json" appeared in "${FLASH_SLUG}" during the update. Nothing was changed.`);
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual(['config.json']);
    expect(readFileSync(join(dir, 'config.json'))).toEqual(mine);
    expect(existsSync(join(dir, MARKER_FILE))).toBe(false);
    expect(flashStagingDirs()).toEqual([]);
  });

  it("never removes an added name that is no longer the job's inode", async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    const mine = Buffer.from('my tokenizer');
    linkHook.before = (to) => {
      if (to !== join(dir, 'config.json')) return;
      // The user replaces a file the job added, and puts a config.json where the job adds one.
      rmSync(join(dir, 'tokenizer.json'));
      writeFileSync(join(dir, 'tokenizer.json'), mine);
      writeFileSync(to, '{}');
    };
    const m = manager();

    const { events, last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('error');
    expect(errorOf(events)).toBe(`"config.json" appeared in "${FLASH_SLUG}" during the update. Nothing was changed.`);
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual(['config.json', 'tokenizer.json']);
    expect(readFileSync(join(dir, 'tokenizer.json'))).toEqual(mine);
    expect(existsSync(join(dir, MARKER_FILE))).toBe(false);
  });

  it('keeps every added file when it cannot remove the added config.json, and says to remove it', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    linkHook.before = (to) => {
      if (to === join(dir, MARKER_FILE)) throw Object.assign(new Error('simulated'), { code: 'EIO' });
    };
    renameFault.failFromPath = join(dir, 'config.json');
    const m = manager();

    const { events, last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('error');
    const stays = [...DOWNLOADED].reverse();
    expect(errorOf(events)).toBe(
      `Update stopped. These files that it added stay in "${FLASH_SLUG}": ${stays.join(', ')}. ` +
        `No file of yours was changed. Remove config.json from the folder, then use Update again.`,
    );
    expectKept(before, dir);
    // A config.json without its tokenizer would read as installed; with every added file beside it the folder is whole.
    expect(addedNames(before, dir)).toEqual([...DOWNLOADED].sort());
    for (const path of DOWNLOADED) expect(readFileSync(join(dir, path))).toEqual(ASSET_FILES[path]);
    expect(buildState(FLASH_SLUG)).toMatchObject({
      present: true,
      installed: false,
      blocker: null,
    });
    rmSync(join(dir, 'config.json'));
    expect(buildState(FLASH_SLUG)?.blocker).toMatchObject({
      holdsModel: true,
      kind: 'directory',
    });
  });

  it('a file renamed onto an added name between the check and the removal is not deleted', async () => {
    const dir = flashFolder();
    const notes = Buffer.from('my notes');
    writeFileSync(join(dir, 'my-notes.txt'), notes);
    let rollingBack = false;
    linkHook.before = (to) => {
      if (to !== join(dir, MARKER_FILE)) return;
      rollingBack = true;
      throw Object.assign(new Error('simulated'), { code: 'EIO' });
    };
    lstatHook.after = (path) => {
      if (!rollingBack || path !== join(dir, 'config.json')) return;
      lstatHook.after = null;
      renameSync(join(dir, 'my-notes.txt'), path);
    };
    const m = manager();

    const { events, last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('error');
    expect(errorOf(events)).toContain(`The app could not add "${MARKER_FILE}" to "${FLASH_SLUG}" (EIO).`);
    expect(readFileSync(join(dir, 'config.json'))).toEqual(notes);
    expect(existsSync(join(dir, MARKER_FILE))).toBe(false);
    for (const path of DOWNLOADED.filter((path) => path !== 'config.json')) {
      expect(existsSync(join(dir, path)), path).toBe(false);
    }
    expect(readdirSync(stagingRoot()).filter((name) => name.startsWith(`${FLASH_SLUG}.kept-`))).toEqual([]);
  });

  it('a file that cannot be put back at its name is kept, and the error names where', async () => {
    const dir = flashFolder();
    const notes = Buffer.from('my notes');
    const later = Buffer.from('{"later":1}');
    writeFileSync(join(dir, 'my-notes.txt'), notes);
    let rollingBack = false;
    linkHook.before = (to) => {
      if (to === join(dir, MARKER_FILE)) {
        rollingBack = true;
        throw Object.assign(new Error('simulated'), { code: 'EIO' });
      }
      // Something takes the name while the moved file is checked, so it cannot go back.
      if (rollingBack && to === join(dir, 'config.json')) writeFileSync(to, later);
    };
    lstatHook.after = (path) => {
      if (!rollingBack || path !== join(dir, 'config.json')) return;
      lstatHook.after = null;
      renameSync(join(dir, 'my-notes.txt'), path);
    };
    const m = manager();

    const { events, last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('error');
    const kept = readdirSync(stagingRoot()).filter((name) => name.startsWith(`${FLASH_SLUG}.kept-`));
    expect(kept).toHaveLength(1);
    const keptDir = join(stagingRoot(), kept[0]!);
    expect(readFileSync(join(keptDir, 'config.json'))).toEqual(notes);
    expect(readFileSync(join(dir, 'config.json'))).toEqual(later);
    const message = errorOf(events)!;
    expect(message).toContain(`"${keptDir}"`);
    expect(message).not.toContain('No file of yours was changed');
    expect(message).not.toContain('Nothing was changed');
  });

  it('a checked file that changes while the files are added stops the Update before the marker', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    linkHook.before = (to) => {
      if (to === join(dir, 'config.json')) truncateSync(join(dir, FLASH_WEIGHT), 3);
    };
    const m = manager();

    const { events, last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('error');
    expect(errorOf(events)).toBe(`"${FLASH_SLUG}" changed on disk during the update. Nothing was changed.`);
    expect(existsSync(join(dir, MARKER_FILE))).toBe(false);
    expect(addedNames(before, dir)).toEqual([]);
    before.delete(FLASH_WEIGHT);
    expectKept(before, dir);
  });

  it('a change to a checked file at the marker link itself is not seen (accepted)', async () => {
    const dir = flashFolder();
    linkHook.before = (to) => {
      if (to === join(dir, MARKER_FILE)) truncateSync(join(dir, FLASH_WEIGHT), 3);
    };
    const m = manager();

    const { last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('done');
    expect(lstatSync(join(dir, FLASH_WEIGHT)).size).toBe(3);
  });

  it('a missing file whose checksum leaves the manifest after the confirm stops the job before any download', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    const m = manager();

    const { events, last } = await runUpdate(m, {
      between: () => {
        hub.manifests[FLASH_ASSETS] = hub.manifests[FLASH_ASSETS]!.map((entry) =>
          entry.path === 'tokenizer.json' ? { type: 'file', path: entry.path, size: entry.size } : entry,
        );
      },
    });
    await m.shutdown();

    expect(last.type).toBe('error');
    expect(errorOf(events)).toContain(`The catalog gives no checksum for "tokenizer.json"`);
    expect(fetched).toEqual([]);
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([]);
  });

  it('stops with a clear text when the volume cannot link', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    linkHook.before = () => {
      throw Object.assign(new Error('operation not supported'), { code: 'ENOTSUP' });
    };
    const m = manager();

    const { events, last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('error');
    expect(errorOf(events)).toBe(
      `The app could not add "tokenizer.json" to "${FLASH_SLUG}" (ENOTSUP). Update needs a volume that can link ` +
        `files. Nothing was changed.`,
    );
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([]);
  });

  it('cancel during the check stops inside the file and leaves the folder as it was', async () => {
    const dir = flashFolder();
    const big = Buffer.alloc(1024 * 1024, 0x11);
    writeFileSync(join(dir, FLASH_WEIGHT), big);
    hub.manifests[FLASH] = hub.manifests[FLASH]!.map((entry) =>
      entry.path === FLASH_WEIGHT ? entryOf(FLASH_WEIGHT, big) : entry,
    );
    const before = filesOf(dir);
    const m = manager();

    const { events, last } = await runUpdate(m, {
      onEvent: (event, id) => {
        if (event.type === 'progress' && event.phase === 'check') m.cancel(id);
      },
    });
    await m.shutdown();

    expect(last.type).toBe('cancelled');
    const read = events.flatMap((event) => (event.type === 'progress' ? [event.receivedBytes] : []));
    expect(read.length).toBeGreaterThan(0);
    expect(Math.max(...read)).toBeLessThan(big.length);
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([]);
    expect(fetched).toEqual([]);
    expect(flashStagingDirs()).toEqual([]);
  });

  it('cancel during the sidecar download leaves the folder as it was', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    let cancel = (): void => {};
    const m = manager((path, init) => {
      if (path !== 'tokenizer_config.json') return undefined;
      cancel();
      return waitForAbort(init);
    });

    const { last } = await runUpdate(m, { onEvent: (_event, id) => void (cancel = () => void m.cancel(id)) });
    await m.shutdown();

    expect(last.type).toBe('cancelled');
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([]);
    expect(flashStagingDirs()).toEqual([]);
  });

  it('refuses to cancel once committing and still finishes', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    let cancelled: boolean | null = null;
    let jobId = '';
    raceHook.onMarkerWrite = () => {
      cancelled = m.cancel(jobId);
    };
    const m = manager();

    const { last } = await runUpdate(m, { onEvent: (_event, id) => void (jobId = id) });
    await m.shutdown();

    expect(cancelled).toBe(false);
    expect(last.type).toBe('done');
    expectKept(before, dir);
    expect(isModelInstalled(dir)).toBe(true);
  });

  it('after a crash in the middle of the adds, the next Update finishes', async () => {
    const dir = flashFolder();
    // What an Update that died after two adds leaves: two complete catalog files and no marker.
    writeFileSync(join(dir, 'tokenizer.json'), ASSET_FILES['tokenizer.json']!);
    writeFileSync(join(dir, 'tokenizer_config.json'), ASSET_FILES['tokenizer_config.json']!);
    const before = filesOf(dir);
    const m = manager();
    const plan = await m.planAdoption(FLASH_SLUG);
    expect([plan.check.count, plan.download.count]).toEqual([7, 4]);

    const { last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('done');
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([...DOWNLOADED.slice(2), MARKER_FILE].sort());
    expect(fetched.sort()).toEqual(DOWNLOADED.slice(2).sort());
    expect(isModelInstalled(dir)).toBe(true);
  });

  it("a dead job's staging tree with second names is reaped and the folder keeps its files", async () => {
    const dir = flashFolder();
    const dead = join(stagingRoot(), `${FLASH_SLUG}@${hub.sha}.999999999.55555555-5555-5555-5555-555555555555`);
    mkdirSync(dead, { recursive: true });
    writeFileSync(join(dead, 'tokenizer.json'), ASSET_FILES['tokenizer.json']!);
    linkSync(join(dead, 'tokenizer.json'), join(dir, 'tokenizer.json'));
    const ino = lstatSync(join(dir, 'tokenizer.json'), { bigint: true }).ino;
    const m = manager();

    const { last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('done');
    expect(existsSync(dead)).toBe(false);
    const stat = lstatSync(join(dir, 'tokenizer.json'), { bigint: true });
    expect([stat.ino, stat.nlink]).toEqual([ino, 1n]);
    expect(readFileSync(join(dir, 'tokenizer.json'))).toEqual(ASSET_FILES['tokenizer.json']);
    expect(fetched).not.toContain('tokenizer.json');
    expect(isModelInstalled(dir)).toBe(true);
  });

  it('the start total covers checked and downloaded bytes and no frame passes it; check frames carry phase check', async () => {
    flashFolder();
    const m = manager();

    const { events, last } = await runUpdate(m);
    await m.shutdown();

    expect(last.type).toBe('done');
    const sum = (paths: string[]): number => paths.reduce((total, path) => total + BYTES[path]!.length, 0);
    const total = sum([...Object.keys(USER_FILES), ...DOWNLOADED]);
    expect(events[0]).toEqual({ type: 'start', id: last.id, repo: FLASH, totalBytes: total, fileCount: 11 });
    const frames = events.filter((event) => event.type === 'progress');
    for (const frame of frames) {
      expect(frame.jobReceivedBytes).toBeLessThanOrEqual(total);
      expect(frame.receivedBytes).toBeLessThanOrEqual(frame.totalBytes);
      expect(frame.fileCount).toBe(11);
      expect(frame.phase, frame.file).toBe(Object.hasOwn(USER_FILES, frame.file) ? 'check' : undefined);
    }
    expect(new Set(frames.map((frame) => frame.file))).toEqual(new Set([...Object.keys(USER_FILES), ...DOWNLOADED]));
    expect(frames.at(-1)!.jobReceivedBytes).toBe(total);
  });

  it('hands back a running Update and refuses one while a plain job holds the repo', async () => {
    const dir = flashFolder();
    const before = filesOf(dir);
    const m = manager((path, init) => (path === 'tokenizer.json' ? waitForAbort(init) : undefined));
    const plan = await m.planAdoption(FLASH_SLUG);
    const id = m.start(FLASH_SLUG, { adopt: plan.token }).id;
    expect(m.start(FLASH_SLUG, { adopt: plan.token }).id).toBe(id);
    expect(() =>
      m.start(FLASH_SLUG, {
        replaceBlocker: { kind: 'directory', fileCount: 6, sizeBytes: 1, truncated: false, linkTarget: null },
      }),
    ).toThrow('A download of this model is already in the queue. Cancel it, then use Clean up & update.');
    await m.shutdown();
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([]);

    const other = manager();
    const second = await other.planAdoption(FLASH_SLUG);
    const plain = other.start(FLASH_SLUG).id;
    expect(() => other.start(FLASH_SLUG, { adopt: second.token })).toThrow(
      'A download of this model is already in the queue. Cancel it, then use Update.',
    );
    const events: DownloadEvent[] = [];
    other.subscribe(plain, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error'));
    await other.shutdown();
    expectKept(before, dir);
    expect(addedNames(before, dir)).toEqual([]);
  });

  it('a later Update available job on the adopted folder keeps what was added after Update', async () => {
    const dir = flashFolder();
    const m = manager();
    expect((await runUpdate(m)).last.type).toBe('done');
    writeFileSync(join(dir, 'notes.txt'), 'mine');
    hub.sha = SHA_NEW;

    const id = m.start(FLASH_SLUG).id;
    const events: DownloadEvent[] = [];
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));
    await m.shutdown();

    expect(errorOf(events)).toBeUndefined();
    const done = events.find((event) => event.type === 'done');
    const keptDir = done?.type === 'done' ? done.keptDir : undefined;
    expect(keptDir).toBeDefined();
    expect(readdirSync(keptDir!).sort()).toEqual([MARKER_FILE, 'README.md', 'notes.txt'].sort());
    expect(readFileSync(join(keptDir!, 'README.md'), 'utf-8')).toBe('notes');
    expect(readFileSync(join(keptDir!, 'notes.txt'), 'utf-8')).toBe('mine');
    expect(readdirSync(dir).sort()).toEqual([...Object.keys(USER_FILES), ...DOWNLOADED, MARKER_FILE].sort());
    expect(JSON.parse(readFileSync(join(dir, MARKER_FILE), 'utf-8'))).toMatchObject({ revision: SHA_NEW });
    expect(isModelInstalled(dir)).toBe(true);
  });
});

describe('a models directory that is a link to a missing disk', () => {
  let root: string;
  let link: string;
  let target: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'dash-dl-link-'));
    link = join(root, 'models');
    target = join(root, 'Volumes', 'Disk', 'models');
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  const sentence = (): string =>
    `The models folder ${link} is a link to ${target}, which is not available. Connect that disk, or change the link, then try again.`;

  it('refuses Install with the plain sentence and writes nothing anywhere', async () => {
    symlinkSync(target, link);
    hub.modelInfoUsesFetch = true;
    const fetched: string[] = [];
    const inner = makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 });
    const m = new DownloadManager({
      modelsDir: link,
      cacheDir,
      fetchImpl: (input, init) => {
        fetched.push(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url);
        return inner(input, init);
      },
    });
    expect(() => m.start(SLUG)).toThrow(sentence());
    const ctx = { downloads: m, modelsDir: link } as unknown as ApiContext;
    const res = await dispatch(ctx, {
      method: 'POST',
      pathname: '/api/downloads',
      query: new URLSearchParams(),
      body: { slug: SLUG },
    });
    expect(res.status).toBe(503);
    expect(res.ok ? '' : res.message).toBe(sentence());
    expect(m.jobs()).toEqual([]);
    // No folder in place of the link, none at its target, nothing in the HF cache, no request.
    expect(lstatSync(link).isSymbolicLink()).toBe(true);
    expect(existsSync(join(root, 'Volumes'))).toBe(false);
    expect(readdirSync(cacheDir)).toEqual([]);
    expect(fetched).toEqual([]);
    expect(hub.downloaded).toEqual([]);

    await expect(m.planAdoption(SLUG)).rejects.toThrow(sentence());
    const plan = await dispatch(ctx, {
      method: 'POST',
      pathname: '/api/downloads/adopt-plan',
      query: new URLSearchParams(),
      body: { slug: SLUG },
    });
    expect(plan.status).toBe(503);
    expect(plan.ok ? '' : plan.message).toBe(sentence());
    expect(existsSync(join(root, 'Volumes'))).toBe(false);
    expect(fetched).toEqual([]);

    // The disk is back: the same Install runs to the end into the link target.
    mkdirSync(target, { recursive: true });
    const id = m.start(SLUG).id;
    const events: DownloadEvent[] = [];
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'error' || event.type === 'done'));
    expect(events.map((event) => event.type)).toContain('done');
    expect(lstatSync(link).isSymbolicLink()).toBe(true);
    expect(readdirSync(target)).toContain(REPO.split('/').pop()!.toLowerCase());
  });

  it('fails a running and a queued job whose disk went away with the same sentence', async () => {
    mkdirSync(target, { recursive: true });
    symlinkSync(target, link);
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    hub.modelInfoUsesFetch = true;
    const inner = makeFetchImpl({ 'config.json': 12, [WEIGHT]: 300 });
    let parked = false;
    const gated: typeof fetch = async (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      if (!parked && url.includes('/api/models/')) {
        parked = true;
        await gate;
      }
      return inner(input, init);
    };
    const m = new DownloadManager({ modelsDir: link, cacheDir, fetchImpl: gated });
    const first = m.start(SLUG).id;
    const second = m.start(SLUG_OTHER).id;
    const events: DownloadEvent[] = [];
    m.subscribe(first, (event) => events.push(event));
    m.subscribe(second, (event) => events.push(event));
    await waitFor(() => parked);

    // The disk goes away while the first job waits on the network.
    rmSync(join(root, 'Volumes'), { recursive: true, force: true });
    release!();
    await waitFor(() => events.filter((event) => event.type === 'error' || event.type === 'done').length === 2);

    for (const id of [first, second]) {
      const error = events.find((event) => event.type === 'error' && event.id === id);
      expect(error?.type === 'error' ? error.message : '', id === first ? 'running' : 'queued').toBe(sentence());
    }
    expect(lstatSync(link).isSymbolicLink()).toBe(true);
    expect(existsSync(join(root, 'Volumes'))).toBe(false);
  });
});

describe('two builds of one repository: one folder, one job, one marker each', () => {
  const BUILD_M = catalogBuild('Qwen3.8-27B', 'UD-Q4_K_M');
  const SLUG_M = catalogBuildSlug(BUILD_M);
  const WEIGHT_M = 'Qwen3.8-27B-UD-Q4_K_M.gguf';
  const SIZES = { [WEIGHT]: 300, [WEIGHT_M]: 280, 'config.json': 12, 'tokenizer.json': 20 };

  beforeEach(() => {
    // The repository holds both weights; each build's glob takes one. The base model supplies the sidecars.
    hub.manifests[REPO] = [
      { type: 'file', path: WEIGHT, size: 300 },
      { type: 'file', path: WEIGHT_M, size: 280 },
    ];
    hub.manifests[ASSETS_REPO] = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'tokenizer.json', size: 20 },
    ];
  });

  function manager(fetchImpl: typeof fetch = makeFetchImpl(SIZES)): DownloadManager {
    return new DownloadManager({ modelsDir, cacheDir, fetchImpl });
  }

  /** An install of `files` in the folder `slug` with the shared marker, pinned to `revision`. */
  function plantInstall(slug: string, files: Record<string, Buffer>, revision: string = hub.sha): void {
    const dir = join(modelsDir, slug);
    mkdirSync(dir, { recursive: true });
    for (const [path, bytes] of Object.entries(files)) writeFileSync(join(dir, path), bytes);
    writeFileSync(
      join(dir, MARKER_FILE),
      JSON.stringify({
        repo: REPO,
        revision,
        files: Object.keys(files),
        scope: 'full',
        assetsRepo: ASSETS_REPO,
        assetsRevision: revision,
        completedAt: 'x',
      }),
    );
  }

  function post(m: DownloadManager, pathname: string, body: unknown): ReturnType<typeof dispatch> {
    const ctx = { downloads: m, modelsDir } as unknown as ApiContext;
    return dispatch(ctx, { method: 'POST', pathname, query: new URLSearchParams(), body });
  }

  it('downloads both builds at the same time into two folders, as two jobs', async () => {
    expect(SLUG_M).toBe('qwen3.8-27b-gguf-ud-q4_k_m');
    const m = manager();
    const first = m.start(SLUG).id;
    // Asked while the first job is in the queue: its own job, not the first one's id.
    const second = m.start(SLUG_M).id;
    expect(second).not.toBe(first);
    expect(m.start(SLUG_M).id).toBe(second);
    await waitFor(() => m.jobs().every((job) => job.state === 'done' || job.state === 'error'));
    expect(m.jobs().map((job) => [job.id, job.slug, job.repo, job.state])).toEqual([
      [first, SLUG, REPO, 'done'],
      [second, SLUG_M, REPO, 'done'],
    ]);
    expect(readdirSync(join(modelsDir, SLUG)).sort()).toEqual([MARKER_FILE, WEIGHT, 'config.json', 'tokenizer.json']);
    expect(readdirSync(join(modelsDir, SLUG_M)).sort()).toEqual([
      MARKER_FILE,
      WEIGHT_M,
      'config.json',
      'tokenizer.json',
    ]);
    expect(buildState(SLUG)).toMatchObject({ installed: true, heldBy: null, localRevision: hub.sha });
    expect(buildState(SLUG_M)).toMatchObject({ installed: true, heldBy: null, localRevision: hub.sha });

    // Delete of one folder leaves the other build installed.
    deleteLocalModel(modelsDir, SLUG_M);
    expect(buildState(SLUG)).toMatchObject({ installed: true, present: true });
    expect(buildState(SLUG_M)).toMatchObject({ installed: false, present: false });
  });

  it('gives build B its own job while build A downloads', async () => {
    let blocked = false;
    const m = manager(
      makeCancelFetch(SIZES, WEIGHT, () => {
        blocked = true;
      }),
    );
    const first = m.start(SLUG).id;
    await waitFor(() => blocked);
    const res = await post(m, '/api/downloads', { slug: SLUG_M });
    expect(res.status).toBe(202);
    const body = (res.ok ? res.body : {}) as { id: string; slug: string; repo: string };
    expect(body).toEqual({ id: body.id, slug: SLUG_M, repo: REPO, kind: 'install' });
    expect(body.id).not.toBe(first);
    expect(m.cancel(body.id)).toBe(true);
    expect(m.cancel(first)).toBe(true);
    await waitFor(() => m.jobs().every((job) => job.state === 'cancelled'));
  });

  it('does not read build B as done from a marker of the same repository and revision that lacks its weight', async () => {
    // The folder of UD-Q4_K_XL holds both weights (two hand runs), with the
    // marker of a UD-Q4_K_M run at the current revision: complete, current and
    // owned, but its files are not this build's.
    plantInstall(SLUG, {
      'config.json': Buffer.alloc(12),
      'tokenizer.json': Buffer.alloc(20),
      [WEIGHT_M]: Buffer.alloc(280),
      [WEIGHT]: Buffer.alloc(300, 0xab),
    });
    const marker = JSON.parse(readFileSync(join(finalDir(), MARKER_FILE), 'utf-8')) as { files: string[] };
    writeFileSync(
      join(finalDir(), MARKER_FILE),
      JSON.stringify({ ...marker, files: marker.files.filter((path) => path !== WEIGHT) }),
    );
    expect(isModelInstalled(finalDir())).toBe(true);
    const m = manager();
    const id = m.start(SLUG).id;
    await waitFor(() => m.jobs().some((job) => job.id === id && (job.state === 'done' || job.state === 'error')));
    expect(m.jobs().find((job) => job.id === id)?.state).toBe('done');
    expect(hub.downloaded).toContain(WEIGHT);
    const published = JSON.parse(readFileSync(join(finalDir(), MARKER_FILE), 'utf-8')) as { files: string[] };
    expect(published.files).toContain(WEIGHT);
    expect(readFileSync(join(finalDir(), WEIGHT))).toEqual(Buffer.alloc(300));
  });

  it('refuses an install into a folder that holds the other build, and changes nothing', async () => {
    // A hand run of the CLI put UD-Q4_K_M in the first build's folder; the
    // owned swap of a UD-Q4_K_XL install would delete it.
    plantInstall(SLUG, { 'config.json': Buffer.alloc(12), [WEIGHT_M]: Buffer.alloc(280) });
    const sentence = `"${SLUG}" holds UD-Q4_K_M. Delete it in Local models, then install UD-Q4_K_XL.`;
    const m = manager();
    expect(() => m.start(SLUG)).toThrow(DownloadConflictError);
    expect(() => m.start(SLUG)).toThrow(sentence);
    await expect(m.planAdoption(SLUG)).rejects.toThrow(sentence);
    await expect(m.planAdoption(SLUG)).rejects.toMatchObject({ reason: undefined });
    const res = await post(m, '/api/downloads', { slug: SLUG });
    expect(res.status).toBe(409);
    expect(res.ok ? '' : res.message).toBe(sentence);
    expect('reason' in res).toBe(false);
    expect(m.jobs()).toEqual([]);
    expect(readdirSync(finalDir()).sort()).toEqual([MARKER_FILE, WEIGHT_M, 'config.json']);
    expect(buildState(SLUG)).toMatchObject({ heldBy: 'UD-Q4_K_M', present: false, blocker: null });
    expect(buildState(SLUG_M)).toMatchObject({ present: true, installed: false });
    expect(hub.modelInfoRepos).toEqual([]);
  });

  it('lets the update sweep mark both installed builds of one repository', async () => {
    plantInstall(SLUG, { 'config.json': Buffer.alloc(12), [WEIGHT]: Buffer.alloc(300) }, SHA_OLD);
    plantInstall(SLUG_M, { 'config.json': Buffer.alloc(12), [WEIGHT_M]: Buffer.alloc(280) }, SHA_OLD);
    hub.sha = SHA_NEW;
    const shas = await manager().checkCatalogUpdates();
    expect(catalogUpdateRepos().filter((repo) => repo === REPO)).toEqual([REPO]);
    expect(shas.get(REPO)).toBe(SHA_NEW);
    // The page compares each build's repo and local revision with the sweep: both builds read as stale.
    for (const slug of [SLUG, SLUG_M]) {
      expect(buildState(slug)).toMatchObject({ installed: true, hfRepo: REPO, localRevision: SHA_OLD });
    }
  });

  it('keys the adopt plan by folder', async () => {
    const res = await post(manager(), '/api/downloads/adopt-plan', { repo: REPO });
    expect(res.status).toBe(400);
    expect(res.ok ? '' : res.message).toBe('Body must include a "slug" string');
    const unknown = await post(manager(), '/api/downloads/adopt-plan', { slug: 'someone-not-in-catalog' });
    expect(unknown.status).toBe(400);
    expect(unknown.ok ? '' : unknown.message).toBe('"someone-not-in-catalog" is not a folder of the model catalog');
  });
});

describe('a CLEF build: the GGUF alone, with no config.json', () => {
  const CLEF_FLASH = MODEL_CATALOG.find((entry) => entry.label === 'CLEF Flash (9B)')!;
  const Q8 = catalogBuild('CLEF Flash (9B)', 'Q8_0');
  const Q4 = catalogBuild('CLEF Flash (9B)', 'Q4_K_M');
  const CLEF_REPO = catalogBuildRepo(Q8);
  const SLUG_Q8 = catalogBuildSlug(Q8);
  const SLUG_Q4 = catalogBuildSlug(Q4);
  const FILE_Q8 = 'Clef-Flash-Q8_0.gguf';
  const FILE_Q4 = 'Clef-Flash-Q4_K_M.gguf';
  /** The bytes the hub serves, by repo path: each weight a real `clef` header with its own tail. */
  let served: Record<string, Buffer>;

  function gguf(architecture: string, tail: number): Buffer {
    return Buffer.concat([ggufHeader(architecture), Buffer.alloc(16, tail)]);
  }

  function entry(path: string, bytes: Buffer): ManifestEntry {
    const blob = createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
    if (!path.endsWith('.gguf')) return { type: 'file', path, size: bytes.length, oid: blob };
    const sha256 = createHash('sha256').update(bytes).digest('hex');
    return {
      type: 'file',
      path,
      size: bytes.length,
      oid: blob,
      lfs: { oid: sha256, size: bytes.length, pointerSize: 134 },
    };
  }

  /** The repository as Hugging Face lists it: three weights, two projectors and the notes; no config.json. */
  function listRepo(): void {
    hub.manifests[CLEF_REPO] = Object.entries(served).map(([path, bytes]) => entry(path, bytes));
  }

  beforeEach(() => {
    served = {
      '.gitattributes': Buffer.from('*.gguf filter=lfs'),
      '.src_sha': Buffer.from('src'),
      'Clef-Flash-BF16.gguf': gguf('clef', 0x01),
      [FILE_Q4]: gguf('clef', 0x02),
      [FILE_Q8]: gguf('clef', 0x03),
      'README.md': Buffer.from('# CLEF Flash'),
      'convert.log': Buffer.from('log'),
      'mmproj-Clef-Flash-BF16.gguf': gguf('clip', 0x04),
      'mmproj-Clef-Flash-Q8_0.gguf': gguf('clip', 0x05),
    };
    listRepo();
  });

  const serveBytes: typeof fetch = async (input) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    return new Response(new Uint8Array(served[url.split('/').slice(3).join('/')] ?? []), { status: 200 });
  };

  function manager(): DownloadManager {
    return new DownloadManager({ modelsDir, cacheDir, fetchImpl: serveBytes });
  }

  async function run(m: DownloadManager, slug: string, opts?: Parameters<DownloadManager['start']>[1]) {
    const id = m.start(slug, opts).id;
    const events: DownloadEvent[] = [];
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some((event) => event.type === 'done' || event.type === 'error'));
    return events;
  }

  function errorOf(events: DownloadEvent[]): string | undefined {
    const error = events.find((event) => event.type === 'error');
    return error?.type === 'error' ? error.message : undefined;
  }

  function marker(slug: string): { repo: string; revision: string; files: string[]; assetsRepo?: string } {
    return JSON.parse(readFileSync(join(modelsDir, slug, MARKER_FILE), 'utf-8')) as {
      repo: string;
      revision: string;
      files: string[];
    };
  }

  it('installs the GGUF alone; the card is installed at its revision and shows an update only for a new one', async () => {
    expect([SLUG_Q8, SLUG_Q4]).toEqual(['clef-flash-gguf', 'clef-flash-gguf-q4_k_m']);
    const events = await run(manager(), SLUG_Q8);
    expect(errorOf(events)).toBeUndefined();
    expect(events.find((event) => event.type === 'start')).toMatchObject({
      fileCount: 1,
      totalBytes: served[FILE_Q8]!.length,
    });
    expect(hub.downloaded).toEqual([FILE_Q8]);
    expect(readdirSync(join(modelsDir, SLUG_Q8)).sort()).toEqual([MARKER_FILE, FILE_Q8]);
    expect(marker(SLUG_Q8)).toMatchObject({ repo: CLEF_REPO, revision: hub.sha, files: [FILE_Q8] });
    expect(marker(SLUG_Q8).assetsRepo).toBeUndefined();
    expect(isModelInstalled(join(modelsDir, SLUG_Q8), configlessBuildOf(CLEF_FLASH, Q8))).toBe(true);
    const item = buildState(SLUG_Q8)!;
    expect(item).toMatchObject({ installed: true, present: true, heldBy: null, localRevision: hub.sha, listed: true });
    expect(buildState(SLUG_Q4)).toMatchObject({ installed: false, present: false, heldBy: null, blocker: null });
    // System One lists it by its file name, as a CLEF model.
    expect(
      (await discoverLocalChatModels(modelsDir, { includeDecisions: true })).map((model) => [
        model.name,
        model.modelType,
      ]),
    ).toEqual([['Clef-Flash-Q8_0', 'clef']]);

    expect(hasUpdate(item, await manager().checkCatalogUpdates())).toBe(false);
    hub.sha = SHA_NEW;
    expect(hasUpdate(buildState(SLUG_Q8)!, await manager().checkCatalogUpdates())).toBe(true);
  });

  it('reads an install at the same revision as done and downloads nothing', async () => {
    await run(manager(), SLUG_Q8);
    hub.downloaded = [];
    const before = lstatSync(join(modelsDir, SLUG_Q8, FILE_Q8)).ino;
    const events = await run(manager(), SLUG_Q8);
    expect(events.map((event) => event.type)).toContain('done');
    expect(hub.downloaded).toEqual([]);
    expect(lstatSync(join(modelsDir, SLUG_Q8, FILE_Q8)).ino).toBe(before);
  });

  it('fails and publishes nothing when the GGUF it downloaded states another family', async () => {
    served[FILE_Q4] = gguf('qwen35', 0x02);
    listRepo();
    const events = await run(manager(), SLUG_Q4);
    expect(errorOf(events)).toBe(
      'The downloaded file is not CLEF Flash (9B) Q4_K_M, so the app cannot load it. Nothing was installed.',
    );
    expect(existsSync(join(modelsDir, SLUG_Q4))).toBe(false);
    await waitFor(() => readdirSync(stagingRoot()).every((name) => !name.startsWith(`${SLUG_Q4}@`)));
  });

  it('refuses a repository with a config.json before it downloads anything', async () => {
    served['config.json'] = Buffer.from('{"model_type":"qwen3_5"}');
    listRepo();
    const events = await run(manager(), SLUG_Q8);
    expect(errorOf(events)).toBe(
      `Repo "${CLEF_REPO}" is not a complete model (needs a GGUF weight file and no config.json)`,
    );
    expect(hub.downloaded).toEqual([]);
    expect(existsSync(join(modelsDir, SLUG_Q8))).toBe(false);
  });

  it('a build with an assetsRepo still needs a config.json', async () => {
    hub.manifest = [{ type: 'file', path: WEIGHT, size: 300 }];
    hub.manifests[ASSETS_REPO] = [{ type: 'file', path: 'tokenizer.json', size: 20 }];
    const m = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: makeFetchImpl({ [WEIGHT]: 300, 'tokenizer.json': 20 }),
    });
    expect(errorOf(await run(m, SLUG))).toBe(
      `Repo "${REPO}" is not a complete model (neither it nor its assets repo "${ASSETS_REPO}" provides a config.json)`,
    );
    expect(existsSync(finalDir())).toBe(false);
  });

  it('installs both builds of one repository in their two folders, and a folder with the other build is held', async () => {
    const m = manager();
    expect(errorOf(await run(m, SLUG_Q8))).toBeUndefined();
    expect(errorOf(await run(m, SLUG_Q4))).toBeUndefined();
    expect(readdirSync(join(modelsDir, SLUG_Q4)).sort()).toEqual([MARKER_FILE, FILE_Q4]);
    expect(buildState(SLUG_Q8)).toMatchObject({ installed: true, heldBy: null });
    expect(buildState(SLUG_Q4)).toMatchObject({ installed: true, heldBy: null });
    deleteLocalModel(modelsDir, SLUG_Q4);
    expect(buildState(SLUG_Q8)).toMatchObject({ installed: true, present: true });
    expect(buildState(SLUG_Q4)).toMatchObject({ installed: false, present: false });

    // The Q8_0 folder now holds the Q4_K_M file: its card is held, and the other card is present by it.
    rmSync(join(modelsDir, SLUG_Q8), { recursive: true });
    mkdirSync(join(modelsDir, SLUG_Q8));
    writeFileSync(join(modelsDir, SLUG_Q8, FILE_Q4), served[FILE_Q4]!);
    expect(buildState(SLUG_Q8)).toMatchObject({ present: false, installed: false, heldBy: 'Q4_K_M', blocker: null });
    expect(buildState(SLUG_Q4)).toMatchObject({ present: true, installed: false });
    expect(() => m.start(SLUG_Q8)).toThrow(`"${SLUG_Q8}" holds Q4_K_M. Delete it in Local models, then install Q8_0.`);
  });

  it('Update in place hashes the GGUF of a folder the app did not create and writes the record, downloading nothing', async () => {
    const dir = join(modelsDir, SLUG_Q8);
    mkdirSync(dir);
    writeFileSync(join(dir, FILE_Q8), served[FILE_Q8]!);
    writeFileSync(join(dir, 'notes.txt'), 'mine');
    expect(buildState(SLUG_Q8)).toMatchObject({ present: true, installed: false, localRevision: null });
    const m = manager();
    const plan = await m.planAdoption(SLUG_Q8);
    expect(plan).toMatchObject({
      repo: CLEF_REPO,
      assetsRevision: null,
      check: { count: 1, bytes: served[FILE_Q8]!.length },
      download: { count: 0, bytes: 0 },
      other: { count: 1, names: ['notes.txt'] },
    });
    expect(errorOf(await run(m, SLUG_Q8, { adopt: plan.token }))).toBeUndefined();
    expect(hub.downloaded).toEqual([]);
    expect(readdirSync(dir).sort()).toEqual([MARKER_FILE, FILE_Q8, 'notes.txt']);
    expect(marker(SLUG_Q8)).toMatchObject({ revision: hub.sha, files: [FILE_Q8] });
    expect(buildState(SLUG_Q8)).toMatchObject({ present: true, installed: true, localRevision: hub.sha });
  });

  it('Update in place takes a folder whose GGUF is of another family as not present, and refuses it by its hash', async () => {
    const dir = join(modelsDir, SLUG_Q8);
    mkdirSync(dir);
    writeFileSync(join(dir, FILE_Q4), gguf('qwen35', 0x02));
    const item = buildState(SLUG_Q8)!;
    expect(item).toMatchObject({ present: false, heldBy: null, blockedByForeignDir: true });
    expect(catalogActionState(item, { job: undefined, updateAvailable: false })).toBe('adopt');
    await expect(manager().planAdoption(SLUG_Q8)).rejects.toThrow(
      `"${FILE_Q4}" in "${SLUG_Q8}" is a model file that is not part of the catalog model.`,
    );
  });

  it('names the file of the build when the Hub list does not have it', async () => {
    delete served[FILE_Q8];
    listRepo();
    expect(errorOf(await run(manager(), SLUG_Q8))).toBe(`The Hub has no file "${FILE_Q8}" for this build.`);
    expect(hub.downloaded).toEqual([]);
    expect(existsSync(join(modelsDir, SLUG_Q8))).toBe(false);
  });

  it.each<[string]>([['hy_v4_flash'], ['qwen35']])(
    'downloads again when the GGUF of an install was replaced by one of %s under the same name',
    async (architecture) => {
      await run(manager(), SLUG_Q8);
      writeFileSync(join(modelsDir, SLUG_Q8, FILE_Q8), gguf(architecture, 0x03));
      expect(isModelInstalled(join(modelsDir, SLUG_Q8), configlessBuildOf(CLEF_FLASH, Q8))).toBe(false);
      expect(buildState(SLUG_Q8)).toMatchObject({ present: false, installed: false, localRevision: null });
      const events = await run(manager(), SLUG_Q8);
      expect(errorOf(events)).toBeUndefined();
      expect(readFileSync(join(modelsDir, SLUG_Q8, FILE_Q8)).equals(served[FILE_Q8]!)).toBe(true);
      expect(buildState(SLUG_Q8)).toMatchObject({ present: true, installed: true, localRevision: hub.sha });
    },
  );

  it('keeps a backup of a good install when the folder now holds a GGUF of another family', async () => {
    await run(manager(), SLUG_Q8);
    mkdirSync(stagingRoot(), { recursive: true });
    const backup = join(stagingRoot(), `${SLUG_Q8}.backup-999999999.11111111-1111-1111-1111-111111111111`);
    cpSync(join(modelsDir, SLUG_Q8), backup, { recursive: true });
    writeFileSync(join(modelsDir, SLUG_Q8, FILE_Q8), gguf('hy_v4_flash', 0x03));
    rmCalls.paths = [];
    expect(errorOf(await run(manager(), SLUG_Q8))).toBeUndefined();
    expect(rmCalls.paths).not.toContain(backup);
    expect(existsSync(join(backup, FILE_Q8))).toBe(true);
  });

  it('a folder with a config.json beside the GGUF is not present: Install with the record, Update in place without', async () => {
    await run(manager(), SLUG_Q8);
    writeFileSync(join(modelsDir, SLUG_Q8, 'config.json'), '{"model_type":"qwen3_5"}');
    hub.sha = SHA_NEW;
    const updates = await manager().checkCatalogUpdates();
    const owned = buildState(SLUG_Q8)!;
    expect(owned).toMatchObject({ present: false, installed: false, localRevision: null, blocker: null });
    expect(hasUpdate(owned, updates)).toBe(false);
    expect(catalogActionState(owned, { job: undefined, updateAvailable: false })).toBe('install');

    const dir = join(modelsDir, SLUG_Q4);
    mkdirSync(dir);
    writeFileSync(join(dir, FILE_Q4), served[FILE_Q4]!);
    writeFileSync(join(dir, 'config.json'), '{}');
    const foreign = buildState(SLUG_Q4)!;
    expect(foreign).toMatchObject({ present: false, installed: false, heldBy: null, blockedByForeignDir: true });
    expect(catalogActionState(foreign, { job: undefined, updateAvailable: false })).toBe('adopt');
    const entry = catalogWithState(modelsDir).find((item) => item.label === 'CLEF Flash (9B)')!;
    expect(modelPresent(entry)).toBe(false);
  });

  it('a CLI folder with every file of the repository is not this build installed, and offers no Update', async () => {
    const dir = join(modelsDir, SLUG_Q8);
    mkdirSync(dir);
    const files = [
      'Clef-Flash-BF16.gguf',
      FILE_Q4,
      FILE_Q8,
      'mmproj-Clef-Flash-BF16.gguf',
      'mmproj-Clef-Flash-Q8_0.gguf',
    ];
    for (const file of files) writeFileSync(join(dir, file), served[file]!);
    writeFileSync(
      join(dir, MARKER_FILE),
      JSON.stringify({ repo: CLEF_REPO, revision: SHA_OLD, files, scope: 'full', completedAt: 'x' }),
    );
    expect(isModelInstalled(dir, configlessBuildOf(CLEF_FLASH, Q8))).toBe(false);
    const updates = await manager().checkCatalogUpdates();
    const item = buildState(SLUG_Q8)!;
    expect(item).toMatchObject({ present: true, installed: false, localRevision: null });
    expect(hasUpdate(item, updates)).toBe(false);
    expect(catalogActionState(item, { job: undefined, updateAvailable: hasUpdate(item, updates) })).toBe('installed');
  });
});

describe('a full record with no config.json of a build that needs one, as before CLEF', () => {
  /** Plants `files` (each a GGUF of `architecture`) in the folder of `build` with a full record of its repository. */
  function plantCliFolder(build: CatalogBuild, files: readonly string[], architecture: string): string {
    const dir = join(modelsDir, catalogBuildSlug(build));
    mkdirSync(dir);
    for (const file of files)
      writeFileSync(join(dir, file), Buffer.concat([ggufHeader(architecture), Buffer.alloc(8)]));
    writeFileSync(
      join(dir, MARKER_FILE),
      JSON.stringify({ repo: catalogBuildRepo(build), revision: hub.sha, files, scope: 'full', completedAt: 'x' }),
    );
    return dir;
  }

  it.each<[string, CatalogBuild, string[], string, NodeJS.Platform]>([
    [
      'the Qwen3.8-27B XL folder with two quants',
      BUILD,
      ['Qwen3.8-27B-UD-Q4_K_XL.gguf', 'Qwen3.8-27B-UD-Q6_K_XL.gguf'],
      'qwen35',
      process.platform,
    ],
    [
      'the Hy4-flash folder with its split set',
      catalogBuild('Hy4-flash', 'Q4_K_M'),
      ['Hy4-flash-Q4_K_M-00001-of-00002.gguf', 'Hy4-flash-Q4_K_M-00002-of-00002.gguf'],
      'hy_v4_flash',
      'darwin',
    ],
    [
      'the Hy4-flash SPARK-4Bit folder',
      catalogBuild('Hy4-flash', 'SPARK-4Bit'),
      ['Hy4-flash-SPARK-4Bit.gguf'],
      'hy_v4_flash',
      'win32',
    ],
  ])('%s: present, not installed, no Update', (_name, build, files, architecture, platform) => {
    const dir = plantCliFolder(build, files, architecture);
    expect(isModelInstalled(dir)).toBe(false);
    const item = buildState(catalogBuildSlug(build, platform), platform)!;
    expect(item).toMatchObject({ present: true, installed: false, localRevision: null });
    const updates = new Map<string, string | null>([[catalogBuildRepo(build, platform), SHA_NEW]]);
    expect(hasUpdate(item, updates)).toBe(false);
    expect(catalogActionState(item, { job: undefined, updateAvailable: hasUpdate(item, updates) })).toBe('installed');
  });

  it('propagates entry platforms in catalogWithState while retaining platform-restricted entries', () => {
    const items = catalogWithState(modelsDir);
    const hy4 = items.find((entry) => entry.label === 'Hy4-flash');
    const qwen = items.find((entry) => entry.label === 'Qwen3.8-27B');
    expect(hy4?.platforms).toEqual(['darwin', 'win32', 'linux']);
    expect(qwen?.platforms).toBeUndefined();
  });
});

describe('restoreWindowsSymlink', () => {
  it('restores a file symlink with type "file"', async () => {
    const calls: Array<{ target: string; path: string; type: 'file' | 'dir' | 'junction' }> = [];
    const stated: string[] = [];
    await restoreWindowsSymlink('C:\\models\\weights.gguf', 'C:\\models\\slug', {
      stat: async (p) => {
        stated.push(p);
        return { isDirectory: () => false };
      },
      symlink: async (target, path, type) => {
        calls.push({ target, path, type });
      },
    });
    expect(stated).toEqual(['C:\\models\\weights.gguf']);
    expect(calls).toEqual([{ target: 'C:\\models\\weights.gguf', path: 'C:\\models\\slug', type: 'file' }]);
  });

  it.each(['\\\\server\\share\\model', '\\\\?\\UNC\\server\\share\\model', '//server/share/model'])(
    'restores UNC directory symlink %s with type "dir" only and never attempts "junction"',
    async (uncTarget) => {
      const okCalls: Array<'file' | 'dir' | 'junction'> = [];
      await restoreWindowsSymlink(uncTarget, 'C:\\models\\slug', {
        stat: async () => ({ isDirectory: () => true }),
        symlink: async (_target, _path, type) => {
          okCalls.push(type);
        },
      });
      expect(okCalls).toEqual(['dir']);

      const permCalls: Array<'file' | 'dir' | 'junction'> = [];
      await expect(
        restoreWindowsSymlink(uncTarget, 'C:\\models\\slug', {
          stat: async () => ({ isDirectory: () => true }),
          symlink: async (_target, _path, type) => {
            permCalls.push(type);
            throw Object.assign(new Error('operation not permitted'), { code: 'EPERM' });
          },
        }),
      ).rejects.toThrow('operation not permitted');
      expect(permCalls).toEqual(['dir']);
    },
  );

  it('restores a local directory symlink with type "dir" when permitted', async () => {
    const calls: Array<{ target: string; path: string; type: 'file' | 'dir' | 'junction' }> = [];
    await restoreWindowsSymlink('..\\shared\\model', 'C:\\models\\slug', {
      stat: async () => ({ isDirectory: () => true }),
      symlink: async (target, path, type) => {
        calls.push({ target, path, type });
      },
    });
    expect(calls).toEqual([{ target: '..\\shared\\model', path: 'C:\\models\\slug', type: 'dir' }]);
  });

  it('falls back to "junction" for local directories only when "dir" throws EPERM or EACCES', async () => {
    for (const code of ['EPERM', 'EACCES'] as const) {
      const calls: Array<'file' | 'dir' | 'junction'> = [];
      await restoreWindowsSymlink('C:\\local\\model', 'C:\\models\\slug', {
        stat: async () => ({ isDirectory: () => true }),
        symlink: async (_target, _path, type) => {
          calls.push(type);
          if (type === 'dir') throw Object.assign(new Error(`symlink ${code}`), { code });
        },
      });
      expect(calls).toEqual(['dir', 'junction']);
    }
  });

  it('rethrows EEXIST immediately without falling back to "junction"', async () => {
    const calls: Array<'file' | 'dir' | 'junction'> = [];
    await expect(
      restoreWindowsSymlink('C:\\local\\model', 'C:\\models\\slug', {
        stat: async () => ({ isDirectory: () => true }),
        symlink: async (_target, _path, type) => {
          calls.push(type);
          throw Object.assign(new Error('file already exists'), { code: 'EEXIST' });
        },
      }),
    ).rejects.toThrow('file already exists');
    expect(calls).toEqual(['dir']);
  });

  it('rejects without guessing "file" when the link target is offline or unreachable', async () => {
    const calls: Array<'file' | 'dir' | 'junction'> = [];
    await expect(
      restoreWindowsSymlink('\\\\nas\\models\\qwen', 'C:\\models\\slug', {
        stat: async () => {
          throw Object.assign(new Error('network path not found'), { code: 'ENOENT' });
        },
        symlink: async (_target, _path, type) => {
          calls.push(type);
        },
      }),
    ).rejects.toThrow('network path not found');
    expect(calls).toEqual([]);
  });
});
