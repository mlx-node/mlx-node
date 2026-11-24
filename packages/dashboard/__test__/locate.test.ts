import { createHash } from 'node:crypto';
import {
  existsSync,
  linkSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import {
  type CatalogBuild,
  catalogBuildRepo,
  catalogBuildSlug,
  catalogDraftSlug,
  MODEL_CATALOG,
} from '@mlx-node/agent/catalog';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import type { ApiContext } from '../src/api/context.js';
import { dispatch } from '../src/api/dispatch.js';
import { type CatalogBuildItem, catalogWithState, configlessBuildOf } from '../src/catalog.js';
import { DownloadConflictError, type DownloadEvent, DownloadManager, HubUnreachableError } from '../src/download.js';
import {
  DOWNLOAD_COMPLETE_MARKER,
  deleteLocalModel,
  discoverLocalModels,
  isModelInstalled,
  locatedState,
  readCompletion,
} from '../src/models.js';
import { ggufHeader } from './helpers/configless-gguf-fixtures.js';

interface ManifestEntry {
  type: 'file' | 'directory';
  path: string;
  size: number;
  oid?: string;
  lfs?: { oid: string; size: number; pointerSize: number };
  xetHash?: string;
}

// The hub mock: `listFiles` per repo, `modelInfo` per repo, and a cache of snapshot links into blobs, as in
// download.test.ts. Downloads write what the job's fetch streams.
const hub = vi.hoisted(() => ({
  manifests: {} as Record<string, ManifestEntry[]>,
  shaByRepo: {} as Record<string, string>,
  sha: 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef',
  modelInfoRepos: [] as string[],
  modelInfoError: null as string | null,
  downloaded: [] as string[],
  /** Runs with the file's path before a download reads the cache. */
  beforeFetch: null as ((path: string) => void) | null,
}));

// `link` runs `linkBefore` with its target first; `realpath` of `hangOn` never settles; `open` runs `onOpen`;
// `copyFile` and `writeFile` run `createBefore` with the path they create.
const fsHooks = vi.hoisted(() => ({
  linkBefore: null as ((to: string) => void) | null,
  createBefore: null as ((path: string) => void) | null,
  hangOn: null as string | null,
  onOpen: null as ((path: string) => void) | null,
}));

vi.mock('node:fs/promises', async (importActual) => {
  const actual = await importActual<typeof import('node:fs/promises')>();
  return {
    ...actual,
    link: async (from: string, to: string) => {
      fsHooks.linkBefore?.(String(to));
      return actual.link(from, to);
    },
    realpath: async (path: string, options?: unknown) => {
      if (fsHooks.hangOn !== null && String(path) === fsHooks.hangOn) return new Promise<string>(() => {});
      return actual.realpath(path, options as never);
    },
    open: async (path: string, flags?: unknown, mode?: unknown) => {
      fsHooks.onOpen?.(String(path));
      return actual.open(path, flags as never, mode as never);
    },
    copyFile: async (from: string, to: string, mode?: number) => {
      fsHooks.createBefore?.(String(to));
      return actual.copyFile(from, to, mode);
    },
    writeFile: async (path: string, data: string, options?: unknown) => {
      fsHooks.createBefore?.(String(path));
      return actual.writeFile(path, data, options as never);
    },
  };
});

// A Locate never moves anything to the Trash, and no test reaches the real one.
vi.mock('../src/trash.js', () => ({
  moveToSystemTrash: async () => {
    throw new Error('a Locate test reached the Trash');
  },
}));

vi.mock('@huggingface/hub', () => ({
  getRepoFolderName: ({ name, type }: { name: string; type: string }) => [`${type}s`, ...name.split('/')].join('--'),
  modelInfo: async (params: { name?: string }) => {
    if (params.name !== undefined) hub.modelInfoRepos.push(params.name);
    if (hub.modelInfoError !== null) throw new Error(hub.modelInfoError);
    return { sha: hub.shaByRepo[params.name ?? ''] ?? hub.sha };
  },
  listFiles: async function* (params: { repo?: { name?: string } }) {
    for (const entry of hub.manifests[params.repo?.name ?? ''] ?? []) yield entry;
  },
  downloadFileToCacheDir: async (params: {
    repo?: { name?: string };
    path: string;
    revision?: string;
    cacheDir: string;
    fetch: typeof fetch;
  }) => {
    hub.beforeFetch?.(params.path);
    const revision = params.revision ?? 'main';
    const storage = `${params.cacheDir}/models--${(params.repo?.name ?? '').split('/').join('--')}`;
    const pointer = `${storage}/snapshots/${revision}/${params.path}`;
    if (existsSync(pointer)) return pointer;
    hub.downloaded.push(params.path);
    const response = await params.fetch(`https://hf.example/${params.repo?.name ?? ''}/${params.path}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    const blob = `${storage}/blobs/${revision}__${params.path.split('/').join('__')}`;
    mkdirSync(dirname(blob), { recursive: true });
    writeFileSync(blob, bytes);
    mkdirSync(dirname(pointer), { recursive: true });
    symlinkSync(blob, pointer);
    return pointer;
  },
}));

function catalogBuild(label: string, id: string): CatalogBuild {
  const build = MODEL_CATALOG.find((entry) => entry.label === label)?.builds.find((candidate) => candidate.id === id);
  if (build === undefined) throw new Error(`no build ${label} ${id}`);
  return build;
}

const BUILD = catalogBuild('Qwen3.8-27B', 'UD-Q4_K_XL');
const REPO = catalogBuildRepo(BUILD);
const SLUG = catalogBuildSlug(BUILD);
const ASSETS = BUILD.assetsRepo!;
const SLUG_M = catalogBuildSlug(catalogBuild('Qwen3.8-27B', 'UD-Q4_K_M'));
const WEIGHT = 'Qwen3.8-27B-UD-Q4_K_XL.gguf';
const WEIGHT_M = 'Qwen3.8-27B-UD-Q4_K_M.gguf';
const ASSETS_SHA = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';

/** The catalog's bytes of every file, by path; a download serves these. */
const CANON: Record<string, Buffer> = {
  'config.json': Buffer.from('{"model_type":"qwen3_5","text_config":{"max_position_embeddings":262144}}'),
  [WEIGHT]: Buffer.alloc(300, 0x11),
  [WEIGHT_M]: Buffer.alloc(280, 0x12),
  'tokenizer.json': Buffer.alloc(40, 0x44),
  'tokenizer_config.json': Buffer.from('{"t":1}'),
  'chat_template.jinja': Buffer.from('{{ messages }}'),
};
const LFS = new Set([WEIGHT, WEIGHT_M, 'tokenizer.json']);
/** The five files a Locate of {@link BUILD} installs. */
const WANTED = ['config.json', WEIGHT, 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'];
const SMALL = WANTED.filter((path) => path !== WEIGHT);

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

/** A fetch that streams the catalog bytes of the file a URL names. */
const serveCanon: typeof fetch = async (input) => {
  const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
  const path = url.split('/').slice(5).join('/');
  return new Response(new Uint8Array(CANON[path] ?? []), { status: 200 });
};

let root: string;
let modelsDir: string;
let cacheDir: string;

beforeEach(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), 'dash-locate-')));
  modelsDir = join(root, 'models');
  cacheDir = join(root, 'cache');
  mkdirSync(modelsDir);
  mkdirSync(cacheDir);
  hub.manifests = {
    [REPO]: [
      entryOf('.gitattributes', Buffer.from('*.gguf filter=lfs')),
      entryOf('config.json', CANON['config.json']!),
      entryOf(WEIGHT, CANON[WEIGHT]!),
      entryOf(WEIGHT_M, CANON[WEIGHT_M]!),
      entryOf('README.md', Buffer.from('# Qwen')),
    ],
    [ASSETS]: [
      entryOf('config.json', CANON['config.json']!),
      entryOf('tokenizer.json', CANON['tokenizer.json']!),
      entryOf('tokenizer_config.json', CANON['tokenizer_config.json']!),
      entryOf('chat_template.jinja', CANON['chat_template.jinja']!),
    ],
  };
  hub.shaByRepo = { [ASSETS]: ASSETS_SHA };
  hub.sha = 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef';
  hub.modelInfoRepos = [];
  hub.modelInfoError = null;
  hub.downloaded = [];
  hub.beforeFetch = null;
  fsHooks.linkBefore = null;
  fsHooks.createBefore = null;
  fsHooks.hangOn = null;
  fsHooks.onOpen = null;
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

function manager(opts: { sameVolume?: (a: bigint, b: bigint) => boolean; diskDeadlineMs?: number } = {}) {
  return new DownloadManager({ modelsDir, cacheDir, fetchImpl: serveCanon, ...opts });
}

const otherDisk = (): boolean => false;

/** A user's folder at `<root>/<name>` holding `files` (catalog paths → bytes; the catalog's bytes by default). */
function userFolder(name: string, files: Record<string, Buffer | undefined>): string {
  const dir = join(root, name);
  mkdirSync(dir, { recursive: true });
  for (const [path, bytes] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), bytes ?? CANON[path]!);
  }
  return dir;
}

/** Every entry under `dir` by `lstat`: kind, size, mtime, inode and content hash, or a link's text. */
function tree(dir: string): Record<string, string> {
  const out: Record<string, string> = {};
  const walk = (rel: string): void => {
    for (const name of readdirSync(join(dir, rel)).sort()) {
      const path = rel === '' ? name : `${rel}/${name}`;
      const full = join(dir, path);
      const stat = lstatSync(full, { bigint: true });
      if (stat.isSymbolicLink()) out[path] = `link ${readlinkSync(full)}`;
      else if (stat.isDirectory()) {
        out[path] = 'dir';
        walk(path);
      } else {
        const sum = createHash('sha256').update(readFileSync(full)).digest('hex');
        out[path] = `file ${stat.size} ${stat.mtimeNs} ${stat.ino} ${sum}`;
      }
    }
  };
  walk('');
  return out;
}

function buildState(slug: string = SLUG): CatalogBuildItem | undefined {
  return catalogWithState(modelsDir)
    .flatMap((item) => item.builds)
    .find((build) => build.slug === slug);
}

async function waitFor(cond: () => boolean, timeoutMs = 5000): Promise<void> {
  const t0 = Date.now();
  while (!cond()) {
    if (Date.now() - t0 > timeoutMs) throw new Error('timed out waiting for condition');
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

function isTerminal(event: DownloadEvent): boolean {
  return event.type === 'done' || event.type === 'error' || event.type === 'cancelled';
}

/** Start a job and wait for its last event. */
async function run(m: DownloadManager, slug: string, opts?: Parameters<DownloadManager['start']>[1]) {
  const id = m.start(slug, opts).id;
  const events: DownloadEvent[] = [];
  m.subscribe(id, (event) => events.push(event));
  await waitFor(() => events.some(isTerminal));
  return events;
}

/** Plan and run a Locate of `slug` from `folder`, and its events. */
async function locate(m: DownloadManager, folder: string, slug: string = SLUG) {
  const plan = await m.planLocate(slug, folder);
  return { plan, events: await run(m, slug, { locate: plan.token }) };
}

function errorOf(events: DownloadEvent[]): string {
  const error = events.find((event) => event.type === 'error');
  return error?.type === 'error' ? error.message : `no error: ${events.map((event) => event.type).join(',')}`;
}

/** The refusal text of a plan, which must be a conflict. */
async function refusal(m: DownloadManager, folder: string, slug: string = SLUG): Promise<string> {
  const error = await m.planLocate(slug, folder).then(
    () => null,
    (e: unknown) => e,
  );
  expect(error).toBeInstanceOf(DownloadConflictError);
  return (error as Error).message;
}

/** The path `path` would have in the staging folder of the running Locate of {@link SLUG}. */
function stagedNow(path: string): string {
  const staging = join(modelsDir, '.staging');
  const job = readdirSync(staging).find((name) => name.startsWith(`${SLUG}@`));
  if (job === undefined) throw new Error('no staging folder');
  return join(staging, job, path);
}

/** A file outside every folder of the test, and a check that its bytes and inode are as they were. */
function outsideFile(): { path: string; unchanged: () => void } {
  const path = join(userFolder('outside', { 'keep.bin': Buffer.alloc(32, 0x5a) }), 'keep.bin');
  const before = tree(dirname(path));
  return { path, unchanged: () => expect(tree(dirname(path))).toEqual(before) };
}

function stagingLeft(): string[] {
  const staging = join(modelsDir, '.staging');
  return existsSync(staging) ? readdirSync(staging) : [];
}

/** The job's `finally` removes its staging folder just after the last event; a leak times out here. */
async function expectNoStaging(): Promise<void> {
  await waitFor(() => stagingLeft().length === 0);
}

const NOT_BESIDE = (name: string): string =>
  `These files are on another disk than the models folder, and "${name}" is not beside them. ` +
  'The app does not add files to your folder. ' +
  'Use Install, or copy the folder to the disk of the models folder and use Locate again.';

describe('Locate: plan', () => {
  it('plans a flat folder on the same disk: the weight is shared, the small files are downloaded', async () => {
    const folder = userFolder('lmstudio/unsloth/Qwen3.8-27B-GGUF', { [WEIGHT]: undefined });
    const before = tree(root);
    const plan = await manager().planLocate(SLUG, folder);
    expect(plan).toEqual({
      slug: SLUG,
      repo: REPO,
      token: plan.token,
      folder,
      mode: 'shared',
      byRecord: false,
      revision: hub.sha,
      use: { count: 1, bytes: 300 },
      read: { count: 1, bytes: 300 },
      download: { count: 4, bytes: SMALL.reduce((sum, path) => sum + CANON[path]!.length, 0) },
    });
    expect(plan.token.length).toBe(64);
    // Nothing anywhere changed: not the user's folder, not the models folder, not the cache.
    expect(tree(root)).toEqual(before);
    expect(hub.downloaded).toEqual([]);
  });

  it('uses a small file of the folder only when its bytes are the catalog file', async () => {
    const folder = userFolder('flat', {
      [WEIGHT]: undefined,
      'tokenizer.json': undefined,
      'config.json': Buffer.from('{"model_type":"other"}'),
    });
    const plan = await manager().planLocate(SLUG, folder);
    expect(plan.use).toEqual({ count: 2, bytes: 300 + 40 });
    expect(plan.download.count).toBe(3);
  });

  it('finds the weights in a child folder when the folder above is picked', async () => {
    const parent = join(root, 'models-of-another-tool');
    userFolder('models-of-another-tool/qwen', { [WEIGHT]: undefined });
    userFolder('models-of-another-tool/notes', { 'readme.txt': Buffer.from('x') });
    const plan = await manager().planLocate(SLUG, parent);
    expect(plan.folder).toBe(join(parent, 'qwen'));
  });

  it('takes a Hugging Face snapshot on the same disk: the links into blobs/ are followed', async () => {
    const cache = join(root, 'hf', 'models--unsloth--Qwen3.8-27B-GGUF');
    const blob = join(cache, 'blobs', createHash('sha256').update(CANON[WEIGHT]!).digest('hex'));
    mkdirSync(dirname(blob), { recursive: true });
    writeFileSync(blob, CANON[WEIGHT]!);
    const snapshot = join(cache, 'snapshots', hub.sha);
    mkdirSync(snapshot, { recursive: true });
    symlinkSync(join('..', '..', 'blobs', blob.split('/').pop()!), join(snapshot, WEIGHT));
    const before = tree(root);
    const m = manager();
    const { plan, events } = await locate(m, snapshot);
    expect(plan.mode).toBe('shared');
    expect(events.at(-1)?.type).toBe('done');
    const staged = lstatSync(join(modelsDir, SLUG, WEIGHT));
    expect(staged.isFile()).toBe(true);
    expect(staged.ino).toBe(lstatSync(blob).ino);
    const after = tree(root);
    for (const [path, entry] of Object.entries(before)) expect(after[path]).toBe(entry);
  });

  it('refuses a folder with no file of the build, and names the file to look for', async () => {
    const folder = userFolder('empty', { 'readme.txt': Buffer.from('x') });
    expect(await refusal(manager(), folder)).toBe(
      `No Qwen3.8-27B files here. Pick the folder that holds "${WEIGHT}". ` +
        'In a Hugging Face cache that is the folder inside "snapshots".',
    );
  });

  it('refuses a folder that holds another build of the model and names it', async () => {
    const folder = userFolder('m-build', { [WEIGHT_M]: undefined });
    expect(await refusal(manager(), folder)).toBe(
      'This folder holds UD-Q4_K_M. Choose UD-Q4_K_M on the card, then use Locate again.',
    );
  });

  it('refuses a weight with the catalog name and another size', async () => {
    const folder = userFolder('old', { [WEIGHT]: Buffer.alloc(299, 0x11) });
    expect(await refusal(manager(), folder)).toBe(
      `"${WEIGHT}" here is not the file the catalog lists: its size is different. ` +
        'It can be an older version. Locate takes only the current file. Use Install.',
    );
  });

  it('refuses two folders that each hold the model, and names both', async () => {
    const parent = join(root, 'two');
    userFolder('two/a', { [WEIGHT]: undefined });
    userFolder('two/b', { [WEIGHT]: undefined });
    expect(await refusal(manager(), parent)).toBe('Two folders here hold Qwen3.8-27B: "a" and "b". Pick one of them.');
  });

  it('searches only the picked folder when it has more than 200 child folders', async () => {
    const parent = join(root, 'many');
    for (let i = 0; i < 201; i++) mkdirSync(join(parent, `d${i}`), { recursive: true });
    userFolder('many/d7', { [WEIGHT]: undefined });
    expect(await refusal(manager(), parent)).toBe(
      `This folder has too many folders to search. Pick the folder that holds "${WEIGHT}".`,
    );
  });

  it('refuses the models folder and a folder inside it', async () => {
    mkdirSync(join(modelsDir, 'other'));
    for (const folder of [modelsDir, join(modelsDir, 'other')]) {
      expect(await refusal(manager(), folder)).toBe(
        'This folder is inside the models folder. The app already lists the models there.',
      );
    }
  });

  it('takes a folder that contains the models folder, and writes only in the models folder', async () => {
    const disk = userFolder('disk', { [WEIGHT]: undefined, 'notes.txt': Buffer.from('mine') });
    modelsDir = join(disk, 'models');
    mkdirSync(modelsDir);
    const outsideModels = (): Record<string, string> =>
      Object.fromEntries(
        Object.entries(tree(disk)).filter(([path]) => path !== 'models' && !path.startsWith('models/')),
      );
    const before = outsideModels();
    const { plan, events } = await locate(manager(), disk);
    expect(plan.mode).toBe('shared');
    expect(plan.folder).toBe(disk);
    expect(events.at(-1)).toEqual({ type: 'done', id: expect.any(String), outputDir: join(modelsDir, SLUG) });
    expect(lstatSync(join(modelsDir, SLUG, WEIGHT)).ino).toBe(lstatSync(join(disk, WEIGHT)).ino);
    expect(outsideModels()).toEqual(before);
  });

  it('never takes a file of the models folder through a link in the picked folder', async () => {
    const owned = userFolder(`models/${SLUG_M}`, { [WEIGHT]: undefined, 'config.json': undefined });
    const folder = userFolder('links', {});
    symlinkSync(join(owned, WEIGHT), join(folder, WEIGHT));
    const before = tree(root);
    expect(await refusal(manager(), folder)).toBe(
      `"${WEIGHT}" here is a file in the models folder. Locate uses only files outside the models folder.`,
    );
    expect(tree(root)).toEqual(before);
  });

  it("refuses a snapshot of the app's download cache when a file must be downloaded into it", async () => {
    const snapshot = join(cacheDir, `models--${REPO.split('/').join('--')}`, 'snapshots', hub.sha);
    userFolder(snapshot.slice(root.length + 1), {
      [WEIGHT]: undefined,
      'config.json': Buffer.alloc(CANON['config.json']!.length, 0x20),
    });
    const before = tree(root);
    expect(await refusal(manager(), snapshot)).toBe(
      'This folder is part of the app\'s download cache, and "config.json" is missing or different there. Use Install.',
    );
    expect(tree(root)).toEqual(before);
  });

  it("refuses when the app cache's repo folder is a link to the folder that holds the model", async () => {
    const repoFolder = `models--${REPO.split('/').join('--')}`;
    const snapshot = join(root, 'hf', repoFolder, 'snapshots', hub.sha);
    userFolder(snapshot.slice(root.length + 1), {
      [WEIGHT]: undefined,
      'config.json': Buffer.alloc(CANON['config.json']!.length, 0x20),
    });
    symlinkSync(join(root, 'hf', repoFolder), join(cacheDir, repoFolder));
    const before = tree(root);
    expect(await refusal(manager(), snapshot)).toBe(
      'This folder is part of the app\'s download cache, and "config.json" is missing or different there. Use Install.',
    );
    expect(tree(root)).toEqual(before);
    expect(hub.downloaded).toEqual([]);
  });

  it("refuses when the app cache's blobs folder is a link into the folder that holds the model", async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    mkdirSync(join(folder, 'blobstore'));
    const repoDir = join(cacheDir, `models--${REPO.split('/').join('--')}`);
    mkdirSync(repoDir, { recursive: true });
    symlinkSync(join(folder, 'blobstore'), join(repoDir, 'blobs'));
    const before = tree(folder);
    expect(await refusal(manager(), folder)).toBe(
      'This folder is part of the app\'s download cache, and "config.json" is missing or different there. Use Install.',
    );
    expect(tree(folder)).toEqual(before);
    expect(hub.downloaded).toEqual([]);
  });

  it('takes a picked folder that holds the download cache beside the folder of the model files', async () => {
    const home = join(root, 'home');
    const folder = userFolder('home/qwen', { [WEIGHT]: undefined });
    const before = tree(folder);
    const m = new DownloadManager({ modelsDir, cacheDir: join(home, '.cache', 'hf'), fetchImpl: serveCanon });
    const { events } = await locate(m, home);
    expect(events.at(-1)?.type).toBe('done');
    expect(tree(folder)).toEqual(before);
  });

  it('names the cache when the folder of the model files holds the download cache', async () => {
    const home = userFolder('home', { [WEIGHT]: undefined });
    const before = tree(home);
    const m = new DownloadManager({ modelsDir, cacheDir: join(home, '.cache', 'hf'), fetchImpl: serveCanon });
    expect(await refusal(m, home)).toBe(
      'The app\'s download cache is inside this folder, and "config.json" must be downloaded. Use Install.',
    );
    expect(tree(home)).toEqual(before);
    expect(hub.downloaded).toEqual([]);
  });

  it("takes a snapshot of the app's download cache that needs nothing downloaded into it", async () => {
    const snapshot = join(cacheDir, `models--${REPO.split('/').join('--')}`, 'snapshots', hub.sha);
    userFolder(snapshot.slice(root.length + 1), { [WEIGHT]: undefined, 'config.json': undefined });
    const before = tree(snapshot);
    const { plan, events } = await locate(manager(), snapshot);
    expect(plan.mode).toBe('shared');
    expect(events.at(-1)?.type).toBe('done');
    expect(tree(snapshot)).toEqual(before);
  });

  it('refuses a path that is a file or is gone', async () => {
    const file = join(userFolder('f', { [WEIGHT]: undefined }), WEIGHT);
    for (const path of [file, join(root, 'gone')]) {
      expect(await refusal(manager(), path)).toBe(
        'The app cannot read this folder. Pick the folder that holds the model files.',
      );
    }
  });

  it('refuses when the folder does not answer in time', async () => {
    const folder = userFolder('slow', { [WEIGHT]: undefined });
    fsHooks.hangOn = folder;
    expect(await refusal(manager({ diskDeadlineMs: 30 }), folder)).toBe(
      'The app could not read this folder in 15 seconds. If macOS asks for access to the disk, allow it, then use Locate again.',
    );
  });

  it('refuses a slug that holds something the app may not replace', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    mkdirSync(join(modelsDir, SLUG));
    writeFileSync(join(modelsDir, SLUG, 'notes.txt'), 'mine');
    expect(await refusal(manager(), folder)).toBe(
      `"${SLUG}" already exists in the models folder. Use the action on its card, or delete it in Local models.`,
    );
  });

  it('refuses while a job for the model is in the queue', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const parked = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: async (input, init) => {
        await gate;
        return serveCanon(input, init);
      },
    });
    const id = parked.start(SLUG).id;
    expect(await refusal(parked, folder)).toBe(
      'A download of this model is already in the queue. Cancel it, then use Locate.',
    );
    parked.cancel(id);
    release!();
    await parked.shutdown();
    expect((await manager().planLocate(SLUG, folder)).mode).toBe('shared');
  });

  it('says Hugging Face could not be reached when the folder has no install record', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    hub.modelInfoError = 'getaddrinfo ENOTFOUND huggingface.co';
    const error = await manager()
      .planLocate(SLUG, folder)
      .catch((e: unknown) => e);
    expect(error).toBeInstanceOf(HubUnreachableError);
    expect((error as Error).message).toBe(
      'Could not reach Hugging Face: getaddrinfo ENOTFOUND huggingface.co. ' +
        "Locate needs the catalog's file list to check this folder. Nothing was changed.",
    );
  });

  it('refuses the companion draft', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const draft = MODEL_CATALOG.find((entry) => entry.draft !== undefined)!.draft!;
    expect(await refusal(manager(), folder, catalogDraftSlug(draft))).toBe(
      'Locate is not offered for companion weights.',
    );
  });

  it('refuses a catalog weight with no checksum', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    hub.manifests[REPO] = hub.manifests[REPO]!.map((entry) =>
      entry.path === WEIGHT ? { type: 'file', path: WEIGHT, size: entry.size } : entry,
    );
    expect(await refusal(manager(), folder)).toBe(
      `The catalog gives no checksum for "${WEIGHT}", so Locate cannot check it.`,
    );
  });
});

describe('Locate: a split set', () => {
  const FLASH = catalogBuild('Qwen3.8-Flash-Next', 'UD-Q4_K_XL');
  const FLASH_REPO = catalogBuildRepo(FLASH);
  const FLASH_SLUG = catalogBuildSlug(FLASH);
  const SHARDS = [1, 2, 3, 4].map((i) => `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000${i}-of-00004.gguf`);

  beforeEach(() => {
    hub.manifests[FLASH_REPO] = SHARDS.map((path, i) => entryOf(path, Buffer.alloc(64 + i, 0x21)));
    hub.manifests[FLASH.assetsRepo!] = [entryOf('config.json', CANON['config.json']!)];
  });

  it('names the parts that are missing', async () => {
    const folder = userFolder('flash', {
      [SHARDS[0]!]: Buffer.alloc(64, 0x21),
      [SHARDS[1]!]: Buffer.alloc(65, 0x21),
    });
    expect(await refusal(manager(), folder, FLASH_SLUG)).toBe(
      `2 of 4 weight files are missing in ${folder}/UD-Q4_K_XL: ` +
        'Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf, Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf. ' +
        'Locate needs all of them in one folder.',
    );
  });

  it('reads the repo folder and its variant folder as one holder, and keeps the catalog paths', async () => {
    const folder = userFolder('flash', Object.fromEntries(SHARDS.map((path, i) => [path, Buffer.alloc(64 + i, 0x21)])));
    const m = manager({ sameVolume: otherDisk });
    const { plan, events } = await locate(m, folder, FLASH_SLUG);
    // Linked: the shards keep their names and sit together; the root config.json is the app's own.
    expect(plan.mode).toBe('linked');
    expect(events.at(-1)?.type).toBe('done');
    for (const path of SHARDS) {
      expect(readlinkSync(join(modelsDir, FLASH_SLUG, path))).toBe(join(folder, path));
    }
    expect(lstatSync(join(modelsDir, FLASH_SLUG, 'config.json')).isFile()).toBe(true);
  });
});

describe('Locate: job, shared', () => {
  it('gives each weight a second name, copies the small files, and writes the record', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined, 'tokenizer.json': undefined });
    const before = tree(folder);
    const m = manager();
    const { plan, events } = await locate(m, folder);
    expect(events.at(-1)).toEqual({ type: 'done', id: expect.any(String), outputDir: join(modelsDir, SLUG) });
    expect(events.filter((event) => event.type === 'progress' && event.phase === 'check').length).toBeGreaterThan(0);
    const dir = join(modelsDir, SLUG);
    const weight = lstatSync(join(dir, WEIGHT));
    expect(weight.isFile()).toBe(true);
    expect(weight.nlink).toBe(2);
    expect(weight.ino).toBe(lstatSync(join(folder, WEIGHT)).ino);
    for (const path of SMALL) {
      const small = lstatSync(join(dir, path));
      expect(small.isFile(), path).toBe(true);
      expect(small.nlink, path).toBe(1);
    }
    const record = readCompletion(dir)!;
    expect(record.located).toEqual({ from: folder, mode: 'shared' });
    expect(record.adopted).toBeUndefined();
    expect(record.revision).toBe(plan.revision);
    expect(record.assetsRevision).toBe(ASSETS_SHA);
    expect([...record.files].sort()).toEqual([...WANTED].sort());
    expect(isModelInstalled(dir)).toBe(true);
    expect(buildState()).toMatchObject({
      present: true,
      installed: true,
      blocker: null,
      located: { from: folder, mode: 'shared', filesAvailable: true },
    });
    // The weight was never fetched; the user's folder is as it was (only its link count moved).
    expect(hub.downloaded).not.toContain(WEIGHT);
    expect(tree(folder)).toEqual(before);
    expect(stagingLeft().filter((name) => name.includes('@'))).toEqual([]);
  });

  it('says the files stay on Delete while the user still has them, and removes only the app folder', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const m = manager();
    await locate(m, folder);
    const dir = join(modelsDir, SLUG);
    expect(locatedState(dir)).toMatchObject({ mode: 'shared', filesStay: true });
    expect(discoverLocalModels(modelsDir).models.find((model) => model.name === SLUG)?.located).toEqual({
      from: folder,
      mode: 'shared',
      filesAvailable: true,
      filesStay: true,
    });
    const before = tree(folder);
    deleteLocalModel(modelsDir, SLUG);
    expect(existsSync(dir)).toBe(false);
    expect(tree(folder)).toEqual(before);
  });

  it('Remove deletes only while the files still stay, checked at the delete', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    await locate(manager(), folder);
    const dir = join(modelsDir, SLUG);
    const ctx = { modelsDir } as unknown as ApiContext;
    const remove = (body?: unknown) =>
      dispatch(ctx, { method: 'DELETE', pathname: `/api/models/${SLUG}`, query: new URLSearchParams(), body });
    // The page read the list while the user still had the weight; then the user deleted their copy.
    expect(discoverLocalModels(modelsDir).models.find((model) => model.name === SLUG)?.located?.filesStay).toBe(true);
    rmSync(join(folder, WEIGHT));
    const refused = await remove({ keepsFiles: true });
    expect(refused).toEqual({
      ok: false,
      code: 'E_CONFLICT',
      status: 409,
      message: `The model files of "${SLUG}" are no longer anywhere else on disk. Delete now removes them.`,
    });
    expect(lstatSync(join(dir, WEIGHT)).nlink).toBe(1);
    for (const body of [{ keepsFiles: 'yes' }, { keepsFiles: true, more: 1 }, [true]]) {
      expect((await remove(body)).status, JSON.stringify(body)).toBe(400);
    }
    expect(isModelInstalled(dir)).toBe(true);
    expect((await remove()).status).toBe(200);
    expect(existsSync(dir)).toBe(false);
  });

  it('Remove deletes the app folder while the files stay, and the user files are unchanged', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    await locate(manager(), folder);
    const before = tree(folder);
    const ctx = { modelsDir } as unknown as ApiContext;
    const removed = await dispatch(ctx, {
      method: 'DELETE',
      pathname: `/api/models/${SLUG}`,
      query: new URLSearchParams(),
      body: { keepsFiles: true },
    });
    expect(removed).toMatchObject({ ok: true, status: 200 });
    expect(existsSync(join(modelsDir, SLUG))).toBe(false);
    expect(tree(folder)).toEqual(before);
  });

  it('reads as a plain install when the app holds the last name of the weight', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    await locate(manager(), folder);
    expect(locatedState(join(modelsDir, SLUG))).toMatchObject({ filesStay: true, filesAvailable: true });
    rmSync(join(folder, WEIGHT));
    expect(locatedState(join(modelsDir, SLUG))).toBeNull();
    expect(isModelInstalled(join(modelsDir, SLUG))).toBe(true);
  });
});

describe('Locate: job, linked', () => {
  /** A folder with every file the engine reads beside the weight. */
  function completeFolder(name = 'disk/qwen'): string {
    return userFolder(name, Object.fromEntries(WANTED.map((path) => [path, undefined])));
  }

  it('links the weights, copies the small files, and shows Files not found while the disk is away', async () => {
    const folder = completeFolder();
    const before = tree(folder);
    const m = manager({ sameVolume: otherDisk });
    const { plan, events } = await locate(m, folder);
    expect(plan).toMatchObject({ mode: 'linked', use: { count: 5 }, read: { count: 1 }, download: { count: 0 } });
    expect(events.at(-1)?.type).toBe('done');
    const dir = join(modelsDir, SLUG);
    expect(readlinkSync(join(dir, WEIGHT))).toBe(join(folder, WEIGHT));
    for (const path of SMALL) expect(lstatSync(join(dir, path)).isFile(), path).toBe(true);
    expect(readCompletion(dir)?.located).toEqual({ from: folder, mode: 'linked' });
    expect(isModelInstalled(dir)).toBe(true);
    expect(locatedState(dir)?.filesStay).toBe(true);
    expect(hub.downloaded).toEqual([]);

    // The disk goes away: the folder stays the app's own, with no blocker and nothing removed.
    renameSync(join(root, 'disk'), join(root, 'away'));
    expect(buildState()).toMatchObject({
      present: false,
      installed: false,
      blocker: null,
      blockedByForeignDir: false,
      located: { from: folder, mode: 'linked', filesAvailable: false },
    });
    expect(discoverLocalModels(modelsDir).models.find((model) => model.name === SLUG)?.located).toMatchObject({
      filesAvailable: false,
    });
    expect(existsSync(join(dir, 'config.json'))).toBe(true);

    // Back again: installed, with no job.
    renameSync(join(root, 'away'), join(root, 'disk'));
    expect(buildState()).toMatchObject({ present: true, installed: true, located: { filesAvailable: true } });
    expect(tree(folder)).toEqual(before);
  });

  it('refuses a Hugging Face snapshot on another disk: its files have other names', async () => {
    const blobs = join(root, 'hf', 'blobs');
    const snapshot = join(root, 'hf', 'snapshots', hub.sha);
    mkdirSync(blobs, { recursive: true });
    mkdirSync(snapshot, { recursive: true });
    for (const path of WANTED) {
      const blob = join(blobs, createHash('sha256').update(CANON[path]!).digest('hex'));
      writeFileSync(blob, CANON[path]!);
      symlinkSync(blob, join(snapshot, path));
    }
    expect(await refusal(manager({ sameVolume: otherDisk }), snapshot)).toBe(
      'These files are on another disk than the models folder, and they are stored there under other names ' +
        '(a Hugging Face cache does this). The app can use them only from the disk of the models folder. ' +
        'Use Install, or copy the folder to that disk and use Locate again.',
    );
  });

  it('refuses a folder whose weights have no config.json beside them', async () => {
    const folder = userFolder('disk/lmstudio', { [WEIGHT]: undefined });
    expect(await refusal(manager({ sameVolume: otherDisk }), folder)).toBe(NOT_BESIDE('config.json'));
  });

  it('refuses a folder whose tokenizer beside the weights has other bytes', async () => {
    const folder = completeFolder();
    writeFileSync(join(folder, 'tokenizer.json'), Buffer.alloc(40, 0x55));
    expect(await refusal(manager({ sameVolume: otherDisk }), folder)).toBe(NOT_BESIDE('tokenizer.json'));
  });

  it('builds again over its own folder whose links are dead, by the owned swap', async () => {
    const folder = completeFolder();
    const m = manager({ sameVolume: otherDisk });
    await locate(m, folder);
    renameSync(join(root, 'disk'), join(root, 'away'));
    const moved = join(root, 'away', 'qwen');
    const { plan, events } = await locate(m, moved);
    expect(events.at(-1)?.type).toBe('done');
    expect(plan.folder).toBe(moved);
    expect(readlinkSync(join(modelsDir, SLUG, WEIGHT))).toBe(join(moved, WEIGHT));
    expect(readCompletion(join(modelsDir, SLUG))?.located?.from).toBe(moved);
    await expectNoStaging();
  });
});

describe('Locate: the install record', () => {
  /** A folder this app installed in an older models folder: the files and the app's record. */
  function recordFolder(name: string, record: Record<string, unknown> = {}): string {
    const folder = userFolder(name, Object.fromEntries(WANTED.map((path) => [path, undefined])));
    writeFileSync(
      join(folder, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: REPO,
        revision: 'a'.repeat(40),
        files: WANTED,
        scope: 'full',
        assetsRepo: ASSETS,
        assetsRevision: 'c'.repeat(40),
        completedAt: '2026-11-01T00:00:00.000Z',
        adopted: { [WEIGHT]: { size: 300, mtimeNs: '1' } },
        ...record,
      }),
    );
    return folder;
  }

  it('trusts the record: no network, no read of a weight, and the record revision is kept', async () => {
    const folder = recordFolder('old-models/qwen');
    hub.modelInfoError = 'offline';
    const m = manager();
    const opened: string[] = [];
    fsHooks.onOpen = (path) => opened.push(path);
    const { plan, events } = await locate(m, folder);
    expect(plan).toMatchObject({ byRecord: true, mode: 'shared', read: { count: 0, bytes: 0 }, use: { count: 5 } });
    expect(events.at(-1)?.type).toBe('done');
    expect(events.some((event) => event.type === 'progress' && event.phase === 'check')).toBe(false);
    expect(hub.modelInfoRepos).toEqual([]);
    expect(opened.filter((path) => path.endsWith(WEIGHT))).toEqual([]);
    const record = readCompletion(join(modelsDir, SLUG))!;
    expect(record).toMatchObject({
      revision: 'a'.repeat(40),
      assetsRepo: ASSETS,
      assetsRevision: 'c'.repeat(40),
      located: { from: folder, mode: 'shared' },
    });
    expect(record.adopted).toBeUndefined();
    // An older revision than upstream: the card can offer Update.
    expect(buildState()?.localRevision).toBe('a'.repeat(40));
  });

  it.each([
    ['another repo', { repo: 'someone/else' }],
    ['a partial scope', { scope: 'partial' }],
    ['an unsafe path', { files: [...WANTED, '../x'] }],
    ['a revision that is not a commit', { revision: 'main' }],
  ])('falls to the catalog check for a record of %s', async (_name, record) => {
    const folder = recordFolder('old', record);
    const plan = await manager().planLocate(SLUG, folder);
    expect(plan.byRecord).toBe(false);
    expect(hub.modelInfoRepos).toContain(REPO);
  });

  it('refuses two child folders that both have the record', async () => {
    recordFolder('old/a');
    recordFolder('old/b');
    hub.modelInfoError = 'offline';
    expect(await refusal(manager(), join(root, 'old'))).toBe(
      'Two folders here hold Qwen3.8-27B: "a" and "b". Pick one of them.',
    );
  });

  it('refuses a record of another build of the repository, with no network', async () => {
    const folder = userFolder('old-m', { [WEIGHT_M]: undefined, 'config.json': undefined });
    writeFileSync(
      join(folder, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: REPO,
        revision: 'a'.repeat(40),
        files: ['config.json', WEIGHT_M],
        completedAt: '2026-11-01T00:00:00.000Z',
      }),
    );
    hub.modelInfoError = 'offline';
    expect(await refusal(manager(), folder)).toBe(
      'This folder holds UD-Q4_K_M. Choose UD-Q4_K_M on the card, then use Locate again.',
    );
    expect(buildState(SLUG_M)?.present).toBe(false);
  });
});

describe('Locate: the job fails, and nothing is changed', () => {
  it('refuses a same-size weight with other bytes, and the next plan refuses it without a read', async () => {
    const folder = userFolder('flat', { [WEIGHT]: Buffer.alloc(300, 0x77) });
    const before = tree(folder);
    const m = manager();
    const { events } = await locate(m, folder);
    expect(errorOf(events)).toBe(
      `"${WEIGHT}" in ${folder} is not the catalog file: its content is different. Nothing was changed.`,
    );
    expect(existsSync(join(modelsDir, SLUG))).toBe(false);
    await expectNoStaging();
    const opened: string[] = [];
    fsHooks.onOpen = (path) => opened.push(path);
    expect(await refusal(m, folder)).toBe(
      `"${WEIGHT}" in ${folder} is not the catalog file: its content is different. Use Install.`,
    );
    expect(opened.filter((path) => path.endsWith(WEIGHT))).toEqual([]);
    expect(tree(folder)).toEqual(before);
  });

  it('stops when a used file changed after the confirmation', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const m = manager();
    const plan = await m.planLocate(SLUG, folder);
    utimesSync(join(folder, WEIGHT), new Date(), new Date(Date.now() - 60_000));
    const events = await run(m, SLUG, { locate: plan.token });
    expect(errorOf(events)).toBe('The folder changed after you confirmed. Use Locate again. Nothing was changed.');
    expect(existsSync(join(modelsDir, SLUG))).toBe(false);
  });

  it('takes only the last token, once, and refuses it with adopt', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const m = manager();
    const plan = await m.planLocate(SLUG, folder);
    const ctx = { downloads: m, modelsDir } as unknown as ApiContext;
    const both = await dispatch(ctx, {
      method: 'POST',
      pathname: '/api/downloads',
      query: new URLSearchParams(),
      body: { slug: SLUG, locate: plan.token, adopt: plan.token },
    });
    expect(both.status).toBe(400);
    expect(() => m.start(SLUG, { locate: 'f'.repeat(64) })).toThrow(
      `The folder or "${SLUG}" changed. Close this, then use Locate again.`,
    );
    expect(m.jobs()).toEqual([]);
    await run(m, SLUG, { locate: plan.token });
    const again = await dispatch(ctx, {
      method: 'POST',
      pathname: '/api/downloads',
      query: new URLSearchParams(),
      body: { slug: SLUG, locate: plan.token },
    });
    expect(again.status).toBe(409);
    expect(again.ok ? '' : again.message).toBe(`The folder or "${SLUG}" changed. Close this, then use Locate again.`);
  });

  it("hands back the active Locate's id only for its own token", async () => {
    const folderA = userFolder('a', { [WEIGHT]: undefined });
    const folderB = userFolder('b', { [WEIGHT]: undefined });
    const m = manager();
    const a = await m.planLocate(SLUG, folderA);
    const pb = m.planLocate(SLUG, folderB);
    const id = m.start(SLUG, { locate: a.token }).id;
    const events: DownloadEvent[] = [];
    m.subscribe(id, (event) => events.push(event));
    const b = await pb;
    expect(() => m.start(SLUG, { locate: b.token })).toThrow(
      'A download of this model is already in the queue. Cancel it, then use Locate.',
    );
    expect(m.start(SLUG, { locate: a.token }).id).toBe(id);
    await waitFor(() => events.some(isTerminal));
    expect(events.at(-1)?.type).toBe('done');
    expect(readCompletion(join(modelsDir, SLUG))?.located?.from).toBe(folderA);
  });

  it('is cancelled during the check, and only its staging folder is removed', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const before = tree(folder);
    const m = manager();
    const plan = await m.planLocate(SLUG, folder);
    let id = '';
    fsHooks.onOpen = (path) => {
      if (path === join(folder, WEIGHT) && id !== '') m.cancel(id);
    };
    id = m.start(SLUG, { locate: plan.token }).id;
    const events: DownloadEvent[] = [];
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => events.some(isTerminal));
    expect(events.at(-1)?.type).toBe('cancelled');
    expect(existsSync(join(modelsDir, SLUG))).toBe(false);
    await expectNoStaging();
    expect(tree(folder)).toEqual(before);
  });

  it('is cancelled during a download, after the weight got its second name, and only staging goes', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const before = tree(folder);
    let blocked = false;
    const m = new DownloadManager({
      modelsDir,
      cacheDir,
      fetchImpl: (input, init) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
        if (!url.endsWith('/tokenizer.json')) return serveCanon(input, init);
        blocked = true;
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
        });
      },
    });
    const plan = await m.planLocate(SLUG, folder);
    const id = m.start(SLUG, { locate: plan.token }).id;
    const events: DownloadEvent[] = [];
    m.subscribe(id, (event) => events.push(event));
    await waitFor(() => blocked);
    expect(lstatSync(join(folder, WEIGHT)).nlink).toBe(2);
    m.cancel(id);
    await waitFor(() => events.some(isTerminal));
    expect(events.at(-1)?.type).toBe('cancelled');
    expect(existsSync(join(modelsDir, SLUG))).toBe(false);
    await expectNoStaging();
    expect(lstatSync(join(folder, WEIGHT)).nlink).toBe(1);
    expect(tree(folder)).toEqual(before);
  });

  it('says so when the disk cannot give a file a second name', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    fsHooks.linkBefore = () => {
      throw Object.assign(new Error('not supported'), { code: 'ENOTSUP' });
    };
    const { events } = await locate(manager(), folder);
    expect(errorOf(events)).toBe('This disk cannot share a file between two folders (ENOTSUP). Nothing was changed.');
    expect(existsSync(join(modelsDir, SLUG))).toBe(false);
    await expectNoStaging();
  });

  // Mutation: the small-file copy without COPYFILE_EXCL → red (the copy writes through the name into the outside file).
  it('fails when a name is at the staged path of a small file it copies, and writes nothing through it', async () => {
    const folder = userFolder('flat', Object.fromEntries(WANTED.map((path) => [path, undefined])));
    const before = tree(folder);
    const outside = outsideFile();
    fsHooks.createBefore = (path) => {
      if (!path.includes('/.staging/') || !path.endsWith('/config.json')) return;
      fsHooks.createBefore = null;
      linkSync(outside.path, path);
    };
    const { events } = await locate(manager(), folder);
    expect(errorOf(events)).toContain('EEXIST');
    expect(existsSync(join(modelsDir, SLUG))).toBe(false);
    outside.unchanged();
    expect(tree(folder)).toEqual(before);
  });

  // Mutation: the download without the rm and without COPYFILE_EXCL → red (it writes through the name).
  it('a download removes a name that is at its staged path and makes a new file', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const outside = outsideFile();
    hub.beforeFetch = (path) => {
      if (path !== 'tokenizer.json') return;
      hub.beforeFetch = null;
      linkSync(outside.path, stagedNow(path));
    };
    const { events } = await locate(manager(), folder);
    expect(events.at(-1)?.type).toBe('done');
    outside.unchanged();
    const staged = lstatSync(join(modelsDir, SLUG, 'tokenizer.json'));
    expect(staged.nlink).toBe(1);
    expect(readFileSync(join(modelsDir, SLUG, 'tokenizer.json'))).toEqual(CANON['tokenizer.json']);
  });

  // Mutation: the staged record written without flag 'wx' → red (the record goes into the outside file).
  it('fails when a name is at the staged path of the record, and writes nothing through it', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const outside = outsideFile();
    fsHooks.createBefore = (path) => {
      if (!path.includes('/.staging/') || !path.endsWith(`/${DOWNLOAD_COMPLETE_MARKER}`)) return;
      fsHooks.createBefore = null;
      linkSync(outside.path, path);
    };
    const { events } = await locate(manager(), folder);
    expect(errorOf(events)).toContain('EEXIST');
    expect(existsSync(join(modelsDir, SLUG))).toBe(false);
    outside.unchanged();
  });

  it('does not publish a weight that was replaced after its check', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    fsHooks.linkBefore = (to) => {
      if (!to.endsWith(WEIGHT)) return;
      fsHooks.linkBefore = null;
      writeFileSync(join(folder, 'swap.tmp'), Buffer.alloc(300, 0x66));
      renameSync(join(folder, 'swap.tmp'), join(folder, WEIGHT));
    };
    const { events } = await locate(manager(), folder);
    expect(errorOf(events)).toBe(`"${WEIGHT}" changed during the check. Use Locate again. Nothing was changed.`);
    expect(existsSync(join(modelsDir, SLUG))).toBe(false);
    expect(readFileSync(join(folder, WEIGHT))).toEqual(Buffer.alloc(300, 0x66));
  });
});

describe('Locate: after Locate', () => {
  it('Update over a located model downloads a full new folder and leaves the user files as they are', async () => {
    const folder = userFolder('disk/qwen', Object.fromEntries(WANTED.map((path) => [path, undefined])));
    const m = manager();
    await locate(m, folder);
    const before = tree(folder);
    hub.sha = 'ffffffffffffffffffffffffffffffffffffffff';
    const events = await run(m, SLUG);
    expect(events.at(-1)?.type).toBe('done');
    const dir = join(modelsDir, SLUG);
    const weight = lstatSync(join(dir, WEIGHT));
    expect(weight.isFile()).toBe(true);
    expect(weight.ino).not.toBe(lstatSync(join(folder, WEIGHT)).ino);
    expect(readCompletion(dir)?.located).toBeUndefined();
    expect(buildState()?.located).toBeNull();
    // The user's file has one name again: the app's old name went with the old folder.
    expect(lstatSync(join(folder, WEIGHT)).nlink).toBe(1);
    expect(tree(folder)).toEqual(before);
  });

  it('Download instead over a folder whose links are dead replaces it and touches nothing outside', async () => {
    const folder = userFolder('disk/qwen', Object.fromEntries(WANTED.map((path) => [path, undefined])));
    const m = manager({ sameVolume: otherDisk });
    await locate(m, folder);
    renameSync(join(root, 'disk'), join(root, 'away'));
    const before = tree(join(root, 'away'));
    const events = await run(m, SLUG);
    expect(events.at(-1)?.type).toBe('done');
    expect(lstatSync(join(modelsDir, SLUG, WEIGHT)).isFile()).toBe(true);
    expect(tree(join(root, 'away'))).toEqual(before);
  });

  it('the sidecar refresh and the record rewrite never write through a second name of a user file', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const m = manager();
    await locate(m, folder);
    const dir = join(modelsDir, SLUG);
    // The app's names of the stale sidecar and of the record are second names of files outside the models folder.
    const outside = userFolder('outside', {
      'chat_template.jinja': CANON['chat_template.jinja'],
      'record.json': readFileSync(join(dir, DOWNLOAD_COMPLETE_MARKER)),
    });
    for (const [name, from] of [
      ['chat_template.jinja', 'chat_template.jinja'],
      [DOWNLOAD_COMPLETE_MARKER, 'record.json'],
    ] as const) {
      rmSync(join(dir, name));
      linkSync(join(outside, from), join(dir, name));
    }
    const before = tree(outside);
    hub.manifests[ASSETS] = hub.manifests[ASSETS]!.filter((entry) => entry.path !== 'chat_template.jinja');
    hub.downloaded = [];
    const events = await run(m, SLUG);
    expect(events.at(-1)?.type).toBe('done');
    // The refresh of the installed folder ran, not a new download.
    expect(hub.downloaded).toEqual([]);
    expect(lstatSync(join(dir, WEIGHT)).ino).toBe(lstatSync(join(folder, WEIGHT)).ino);
    expect(existsSync(join(dir, 'chat_template.jinja'))).toBe(false);
    expect(lstatSync(join(dir, DOWNLOAD_COMPLETE_MARKER)).ino).not.toBe(lstatSync(join(outside, 'record.json')).ino);
    expect(readCompletion(dir)?.located).toEqual({ from: folder, mode: 'shared' });
    expect(readCompletion(dir)?.files).not.toContain('chat_template.jinja');
    expect(tree(outside)).toEqual(before);
  });

  it('a crashed job: the reap of its staging tree leaves the targets of its names', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined, 'tokenizer.json': undefined });
    const before = tree(folder);
    const dead = join(modelsDir, '.staging', `${SLUG}@${hub.sha}.999999999.0f0f0f0f-0f0f-0f0f-0f0f-0f0f0f0f0f0f`);
    mkdirSync(dead, { recursive: true });
    linkSync(join(folder, WEIGHT), join(dead, WEIGHT));
    symlinkSync(join(folder, 'tokenizer.json'), join(dead, 'tokenizer.json'));
    const m = manager();
    await run(m, SLUG);
    expect(existsSync(dead)).toBe(false);
    expect(tree(folder)).toEqual(before);
  });

  it('Delete of a linked model removes only the app folder', async () => {
    const folder = userFolder('disk/qwen', Object.fromEntries(WANTED.map((path) => [path, undefined])));
    await locate(manager({ sameVolume: otherDisk }), folder);
    const before = tree(join(root, 'disk'));
    deleteLocalModel(modelsDir, SLUG);
    expect(existsSync(join(modelsDir, SLUG))).toBe(false);
    expect(tree(join(root, 'disk'))).toEqual(before);
  });
});

describe('Locate: the route', () => {
  it('plans through POST /api/downloads/locate-plan and starts with the token', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const m = manager();
    const ctx = { downloads: m, modelsDir } as unknown as ApiContext;
    const plan = await dispatch(ctx, {
      method: 'POST',
      pathname: '/api/downloads/locate-plan',
      query: new URLSearchParams(),
      body: { slug: SLUG, path: folder },
    });
    expect(plan.status).toBe(200);
    const token = (plan.ok ? (plan.body as { token: string }) : { token: '' }).token;
    const started = await dispatch(ctx, {
      method: 'POST',
      pathname: '/api/downloads',
      query: new URLSearchParams(),
      body: { slug: SLUG, locate: token },
    });
    expect(started.status).toBe(202);
    await waitFor(() => m.jobs().every((job) => job.state !== 'running' && job.state !== 'committing'));
    expect(readCompletion(join(modelsDir, SLUG))?.located?.from).toBe(folder);
  });

  it.each([
    ['a relative path', { slug: SLUG, path: 'models/qwen' }],
    ['a NUL', { slug: SLUG, path: '/a\0b' }],
    ['a path that is too long', { slug: SLUG, path: `/${'a'.repeat(4096)}` }],
    ['no slug', { path: '/a' }],
  ])('answers 400 for %s', async (_name, body) => {
    const ctx = { downloads: manager(), modelsDir } as unknown as ApiContext;
    const res = await dispatch(ctx, {
      method: 'POST',
      pathname: '/api/downloads/locate-plan',
      query: new URLSearchParams(),
      body,
    });
    expect(res.status).toBe(400);
    expect(res.ok ? '' : res.message).toBe('Body must include a "slug" and an absolute "path".');
  });

  it('answers 503 with the plain sentence when the models folder is a link to a missing disk', async () => {
    const folder = userFolder('flat', { [WEIGHT]: undefined });
    const link = join(root, 'linked-models');
    const target = join(root, 'Volumes', 'Disk', 'models');
    symlinkSync(target, link);
    const m = new DownloadManager({ modelsDir: link, cacheDir, fetchImpl: serveCanon });
    const ctx = { downloads: m, modelsDir: link } as unknown as ApiContext;
    const res = await dispatch(ctx, {
      method: 'POST',
      pathname: '/api/downloads/locate-plan',
      query: new URLSearchParams(),
      body: { slug: SLUG, path: folder },
    });
    expect(res.status).toBe(503);
    expect(res.ok ? '' : res.message).toBe(
      `The models folder ${link} is a link to ${target}, which is not available. Connect that disk, or change the link, then try again.`,
    );
    expect(existsSync(join(root, 'Volumes'))).toBe(false);
  });

  it('answers 409 with the refusal text', async () => {
    const folder = userFolder('empty', { 'readme.txt': Buffer.from('x') });
    const ctx = { downloads: manager(), modelsDir } as unknown as ApiContext;
    const res = await dispatch(ctx, {
      method: 'POST',
      pathname: '/api/downloads/locate-plan',
      query: new URLSearchParams(),
      body: { slug: SLUG, path: folder },
    });
    expect(res.status).toBe(409);
  });
});

describe('Locate of a CLEF build: a folder that holds only the GGUF', () => {
  const Q8 = catalogBuild('CLEF Flash (9B)', 'Q8_0');
  const CLEF_FLASH = MODEL_CATALOG.find((entry) => entry.label === 'CLEF Flash (9B)')!;
  const CLEF_REPO = catalogBuildRepo(Q8);
  const CLEF_SLUG = catalogBuildSlug(Q8);
  const FILE = 'Clef-Flash-Q8_0.gguf';
  const BYTES = Buffer.concat([ggufHeader('clef'), Buffer.alloc(16, 0x03)]);

  beforeEach(() => {
    const sha256 = createHash('sha256').update(BYTES).digest('hex');
    const blob = createHash('sha1').update(`blob ${BYTES.length}\0`).update(BYTES).digest('hex');
    hub.manifests[CLEF_REPO] = [
      entryOf('.gitattributes', Buffer.from('*.gguf filter=lfs')),
      {
        type: 'file',
        path: FILE,
        size: BYTES.length,
        oid: blob,
        lfs: { oid: sha256, size: BYTES.length, pointerSize: 134 },
      },
      entryOf('README.md', Buffer.from('# CLEF Flash')),
    ];
  });

  function clefManager(opts: { sameVolume?: (a: bigint, b: bigint) => boolean } = {}): DownloadManager {
    return new DownloadManager({ modelsDir, cacheDir, fetchImpl: serveCanon, ...opts });
  }

  it.each<['shared' | 'linked']>([['shared'], ['linked']])(
    'uses the GGUF %s and needs no config.json',
    async (mode) => {
      const folder = userFolder('clef', { [FILE]: BYTES, 'README.md': Buffer.from('notes') });
      const m = clefManager(mode === 'linked' ? { sameVolume: otherDisk } : {});
      const { plan, events } = await locate(m, folder, CLEF_SLUG);
      expect(plan).toMatchObject({
        repo: CLEF_REPO,
        folder,
        mode,
        byRecord: false,
        use: { count: 1, bytes: BYTES.length },
        download: { count: 0, bytes: 0 },
      });
      expect(events.map((event) => event.type)).toContain('done');
      expect(hub.downloaded).toEqual([]);
      const dir = join(modelsDir, CLEF_SLUG);
      expect(readdirSync(dir).sort()).toEqual([DOWNLOAD_COMPLETE_MARKER, FILE]);
      expect(lstatSync(join(dir, FILE)).isSymbolicLink()).toBe(mode === 'linked');
      expect(readCompletion(dir)).toMatchObject({ files: [FILE], located: { from: folder, mode } });
      expect(isModelInstalled(dir, configlessBuildOf(CLEF_FLASH, Q8))).toBe(true);
      expect(buildState(CLEF_SLUG)).toMatchObject({
        installed: true,
        present: true,
        localRevision: hub.sha,
        located: { from: folder, mode, filesAvailable: true },
      });
      // Local models has no build for the row, and reads the files as there too.
      expect(discoverLocalModels(modelsDir).models).toEqual([
        expect.objectContaining({ name: CLEF_SLUG, located: expect.objectContaining({ filesAvailable: true }) }),
      ]);
    },
  );

  it('refuses a link to a GGUF with a config.json beside it, which the engine would read as another model', async () => {
    const folder = userFolder('clef', { [FILE]: BYTES, 'config.json': Buffer.from('{"model_type":"qwen3_5"}') });
    expect(await refusal(clefManager({ sameVolume: otherDisk }), folder, CLEF_SLUG)).toBe(
      'These files are on another disk than the models folder, and a config.json is beside them, ' +
        'so the app would not load them as CLEF Flash (9B). ' +
        'Move the config.json out of the folder, or copy the GGUF to the disk of the models folder, then use Locate again.',
    );
    // On the disk of the models folder the app's folder has no config.json, so the same folder is fine.
    expect((await clefManager().planLocate(CLEF_SLUG, folder)).mode).toBe('shared');
  });

  it('trusts the app record of a CLEF install in the picked folder', async () => {
    const folder = userFolder('copy', { [FILE]: BYTES });
    writeFileSync(
      join(folder, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: CLEF_REPO, revision: hub.sha, files: [FILE], scope: 'full', completedAt: 'x' }),
    );
    const { plan, events } = await locate(clefManager(), folder, CLEF_SLUG);
    expect(plan).toMatchObject({ byRecord: true, read: { count: 0, bytes: 0 }, download: { count: 0, bytes: 0 } });
    expect(events.map((event) => event.type)).toContain('done');
    expect(buildState(CLEF_SLUG)).toMatchObject({ installed: true, localRevision: hub.sha });
  });

  it('does not trust an app record whose GGUF now states another family', async () => {
    const folder = userFolder('copy', { [FILE]: Buffer.concat([ggufHeader('hy_v4_flash'), Buffer.alloc(16, 0x03)]) });
    writeFileSync(
      join(folder, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: CLEF_REPO, revision: hub.sha, files: [FILE], scope: 'full', completedAt: 'x' }),
    );
    expect(await refusal(clefManager(), folder, CLEF_SLUG)).toContain(
      `"${FILE}" here is not the file the catalog lists`,
    );
  });
});
