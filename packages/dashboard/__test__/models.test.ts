import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { MODEL_CATALOG } from '@mlx-node/agent/catalog';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { type CatalogItem, catalogSlug, catalogWithState } from '../src/catalog.js';
import {
  DOWNLOAD_COMPLETE_MARKER,
  defaultModelsDir,
  deleteLocalModel,
  discoverLocalModels,
  isDownloaderOwned,
  isModelInstalled,
  isModelPresent,
  readCompletion,
} from '../src/models.js';

let modelsDir: string;

const CONFIG_A = JSON.stringify({ model_type: 'qwen3', max_position_embeddings: 40960 });
const CONFIG_B = JSON.stringify({
  model_type: 'qwen3_5',
  quantization_config: { mode: 'affine', bits: 4, group_size: 64 },
  text_config: { max_position_embeddings: 262144 },
});
const WEIGHT_A_BYTES = 2048;
const WEIGHT_B_BYTES = 4096;

function writeModel(dir: string, name: string, config: string, weightBytes: number): void {
  const full = join(dir, name);
  mkdirSync(full, { recursive: true });
  writeFileSync(join(full, 'config.json'), config);
  writeFileSync(join(full, 'model.safetensors'), Buffer.alloc(weightBytes));
}

beforeEach(() => {
  modelsDir = mkdtempSync(join(tmpdir(), 'dash-models-'));
  writeModel(modelsDir, 'model-a', CONFIG_A, WEIGHT_A_BYTES);
  writeModel(modelsDir, 'model-b', CONFIG_B, WEIGHT_B_BYTES);
  // A junk subdirectory with no config.json → warned + skipped.
  const junk = join(modelsDir, 'junk');
  mkdirSync(junk, { recursive: true });
  writeFileSync(join(junk, 'notes.txt'), 'not a model');
  // A stray file at the root → ignored (not a directory).
  writeFileSync(join(modelsDir, 'README.md'), '# models');
});

afterEach(() => {
  rmSync(modelsDir, { recursive: true, force: true });
});

describe('discoverLocalModels', () => {
  it('discovers models with correct type/quant/ctx/size and warns on junk', () => {
    const { models, warnings } = discoverLocalModels(modelsDir);

    expect(models.map((m) => m.name)).toEqual(['model-a', 'model-b']);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain('junk');

    const a = models.find((m) => m.name === 'model-a')!;
    expect(a.modelType).toBe('qwen3');
    expect(a.quant).toBeNull();
    expect(a.contextWindow).toBe(40960);
    expect(a.fileCount).toBe(2);
    expect(a.sizeBytes).toBe(Buffer.byteLength(CONFIG_A) + WEIGHT_A_BYTES);

    const b = models.find((m) => m.name === 'model-b')!;
    expect(b.modelType).toBe('qwen3_5');
    expect(b.quant).toBe('affine-4bit');
    expect(b.contextWindow).toBe(262144);
    expect(b.fileCount).toBe(2);
    expect(b.sizeBytes).toBe(Buffer.byteLength(CONFIG_B) + WEIGHT_B_BYTES);
  });

  it('returns empty (no warning) for a missing directory', () => {
    const { models, warnings } = discoverLocalModels(join(modelsDir, 'does-not-exist'));
    expect(models).toHaveLength(0);
    expect(warnings).toHaveLength(0);
  });

  // A checkpoint dir can contain symlinks (HF-cache blob links, or hostile links a
  // caller planted). Sizing must NEVER descend into a symlinked directory: a
  // self-referential link is a cycle that used to recurse until the OS symlink-loop
  // limit, inflating the size/count (a directory with two `-> .` links is an
  // exponential ~2^31-stat freeze of the synchronous /api/models handler), and a
  // link to an external tree would wrongly fold that tree's bytes into the model.
  it('does not recurse into symlinked directories (breaks cycles, excludes external trees)', () => {
    const modelDir = join(modelsDir, 'linky');
    mkdirSync(modelDir, { recursive: true });
    const CONFIG = JSON.stringify({ model_type: 'qwen3' });
    const WEIGHT_BYTES = 512;
    writeFileSync(join(modelDir, 'config.json'), CONFIG);
    writeFileSync(join(modelDir, 'model.safetensors'), Buffer.alloc(WEIGHT_BYTES));

    // A self-referential symlink → a 1-cycle. Following it recurses without bound.
    symlinkSync('.', join(modelDir, 'self'));

    // A symlink to an external directory holding a large file — must not be counted.
    const externalDir = mkdtempSync(join(tmpdir(), 'dash-ext-'));
    writeFileSync(join(externalDir, 'huge.bin'), Buffer.alloc(1_000_000));
    symlinkSync(externalDir, join(modelDir, 'ext'));

    const t0 = Date.now();
    const { models } = discoverLocalModels(modelsDir);
    const elapsedMs = Date.now() - t0;

    const linky = models.find((m) => m.name === 'linky')!;
    expect(linky).toBeDefined();
    // Exactly the two real files — the symlinked dirs contribute neither bytes nor
    // recursion, and the external tree's 1 MB file is excluded.
    expect(linky.fileCount).toBe(2);
    expect(linky.sizeBytes).toBe(Buffer.byteLength(CONFIG) + WEIGHT_BYTES);
    // And it returned promptly rather than spinning on the cycle.
    expect(elapsedMs).toBeLessThan(5000);

    rmSync(externalDir, { recursive: true, force: true });
  });
});

describe('deleteLocalModel', () => {
  it('removes an existing model directory', () => {
    expect(existsSync(join(modelsDir, 'model-a'))).toBe(true);
    deleteLocalModel(modelsDir, 'model-a');
    expect(existsSync(join(modelsDir, 'model-a'))).toBe(false);
    // Siblings untouched.
    expect(existsSync(join(modelsDir, 'model-b'))).toBe(true);
  });

  it('throws on a path that escapes the models directory', () => {
    expect(() => deleteLocalModel(modelsDir, '../../etc')).toThrow();
    // The escape target is never touched (still present).
    expect(existsSync(join(modelsDir, 'model-b'))).toBe(true);
  });

  it('throws when targeting the models directory root itself', () => {
    expect(() => deleteLocalModel(modelsDir, '.')).toThrow();
    expect(existsSync(modelsDir)).toBe(true);
  });

  it('throws when the model does not exist', () => {
    expect(() => deleteLocalModel(modelsDir, 'ghost')).toThrow();
  });

  it('throws on a name containing a path separator', () => {
    expect(() => deleteLocalModel(modelsDir, 'a/b')).toThrow();
    expect(() => deleteLocalModel(modelsDir, 'a\\b')).toThrow();
    // Siblings untouched.
    expect(existsSync(join(modelsDir, 'model-a'))).toBe(true);
  });

  it("throws on a '..' traversal name", () => {
    expect(() => deleteLocalModel(modelsDir, '..')).toThrow();
    expect(existsSync(modelsDir)).toBe(true);
  });

  it('refuses to delete a dot-prefixed reserved dir (e.g. .staging) and leaves it intact', () => {
    const staging = join(modelsDir, '.staging');
    mkdirSync(staging, { recursive: true });
    writeFileSync(join(staging, 'in-flight.txt'), 'active publish');

    expect(() => deleteLocalModel(modelsDir, '.staging')).toThrow(/reserved/i);
    expect(existsSync(staging)).toBe(true);
    expect(existsSync(join(staging, 'in-flight.txt'))).toBe(true);
  });

  it('refuses to delete a symlinked child and leaves its target untouched', () => {
    const outside = mkdtempSync(join(tmpdir(), 'dash-outside-'));
    const keep = join(outside, 'victim.txt');
    writeFileSync(keep, 'important');
    // <modelsDir>/link -> outside (an intermediate symlink rmSync must not follow).
    symlinkSync(outside, join(modelsDir, 'link'));

    expect(() => deleteLocalModel(modelsDir, 'link')).toThrow(/symlink/i);
    // The symlink target's contents survive.
    expect(existsSync(keep)).toBe(true);
    // And the encoded child-of-a-symlink attack is rejected by the separator guard.
    expect(() => deleteLocalModel(modelsDir, 'link/victim.txt')).toThrow();
    expect(existsSync(keep)).toBe(true);

    rmSync(outside, { recursive: true, force: true });
  });

  it('refuses to delete a non-checkpoint directory (no config.json) and a stray file', () => {
    // `junk/` (a dir without config.json) and `README.md` (a regular file) both
    // sit directly under modelsDir with safe names, but neither is a checkpoint
    // discovery would list — a typo/hand-crafted DELETE must not rmSync them.
    expect(() => deleteLocalModel(modelsDir, 'junk')).toThrow(/not a model checkpoint/i);
    expect(existsSync(join(modelsDir, 'junk'))).toBe(true);
    expect(() => deleteLocalModel(modelsDir, 'README.md')).toThrow(/not a model checkpoint/i);
    expect(existsSync(join(modelsDir, 'README.md'))).toBe(true);
  });
});

describe('isModelInstalled', () => {
  /** Write a completion marker listing `files` into `<modelsDir>/<name>`. */
  function writeMarker(name: string, files: string[], scope?: 'full' | 'partial'): string {
    const dir = join(modelsDir, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: 'owner/repo', revision: 'a'.repeat(40), files, scope, completedAt: 'x' }),
    );
    return dir;
  }

  it('is installed only when the marker lists a config AND a weight, all present', () => {
    const dir = writeMarker('good', ['config.json', 'model.safetensors']);
    writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
    writeFileSync(join(dir, 'model.safetensors'), Buffer.alloc(8));
    expect(isModelInstalled(dir)).toBe(true);
  });

  it('does not install a partial glob marker even when config and one weight are present', () => {
    const dir = writeMarker('partial-glob', ['config.json', 'model-00001-of-00002.safetensors'], 'partial');
    writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
    writeFileSync(join(dir, 'model-00001-of-00002.safetensors'), Buffer.alloc(8));
    expect(isModelInstalled(dir)).toBe(false);
    expect(isDownloaderOwned(dir)).toBe(true);
  });

  it('is NOT installed for a one-sided marker listing only config.json (Finding G2)', () => {
    const dir = writeMarker('config-only', ['config.json']);
    writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
    expect(isModelInstalled(dir)).toBe(false);
  });

  it('is NOT installed for a weights-only marker (no config.json)', () => {
    const dir = writeMarker('weights-only', ['model.safetensors']);
    writeFileSync(join(dir, 'model.safetensors'), Buffer.alloc(8));
    expect(isModelInstalled(dir)).toBe(false);
  });

  it('is NOT installed when a listed file is missing on disk, nor without a marker', () => {
    const dir = writeMarker('missing-weight', ['config.json', 'model.safetensors']);
    writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
    // model.safetensors is listed but never written.
    expect(isModelInstalled(dir)).toBe(false);

    // A bare dir with no marker at all is never installed.
    const bare = join(modelsDir, 'bare');
    mkdirSync(bare, { recursive: true });
    writeFileSync(join(bare, 'config.json'), Buffer.alloc(4));
    writeFileSync(join(bare, 'model.safetensors'), Buffer.alloc(8));
    expect(isModelInstalled(bare)).toBe(false);
  });
});

describe('isModelPresent', () => {
  it('is present for a markerless config + weight dir (CLI/wizard install)', () => {
    // Exactly what `mlx download model` leaves on disk: a loadable checkpoint with
    // NO dashboard completion marker. `isModelInstalled` rejects it (unowned), but
    // it must read as PRESENT so the UI does not offer an Install that would then
    // refuse to overwrite the unowned directory.
    const dir = join(modelsDir, 'cli-installed');
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
    writeFileSync(join(dir, 'model.safetensors'), Buffer.alloc(8));
    expect(isModelInstalled(dir)).toBe(false);
    expect(isModelPresent(dir)).toBe(true);
  });

  it('is NOT present without a config.json, without a weight, or for a missing dir', () => {
    const weightOnly = join(modelsDir, 'weight-only');
    mkdirSync(weightOnly, { recursive: true });
    writeFileSync(join(weightOnly, 'model.safetensors'), Buffer.alloc(8));
    expect(isModelPresent(weightOnly)).toBe(false);

    const configOnly = join(modelsDir, 'config-only-present');
    mkdirSync(configOnly, { recursive: true });
    writeFileSync(join(configOnly, 'config.json'), Buffer.alloc(4));
    expect(isModelPresent(configOnly)).toBe(false);

    expect(isModelPresent(join(modelsDir, 'does-not-exist'))).toBe(false);
  });

  it('requires every shard of a sharded checkpoint, not just the first', () => {
    // config.json + index.json + ALL shards → present.
    const full = join(modelsDir, 'sharded-full');
    mkdirSync(full, { recursive: true });
    writeFileSync(join(full, 'config.json'), Buffer.alloc(4));
    writeFileSync(
      join(full, 'model.safetensors.index.json'),
      JSON.stringify({
        weight_map: { 'a.weight': 'model-00001-of-00002.safetensors', 'b.weight': 'model-00002-of-00002.safetensors' },
      }),
    );
    writeFileSync(join(full, 'model-00001-of-00002.safetensors'), Buffer.alloc(8));
    writeFileSync(join(full, 'model-00002-of-00002.safetensors'), Buffer.alloc(8));
    expect(isModelPresent(full)).toBe(true);

    // Interrupted: index + only the first shard → NOT present (a partial download
    // must not read as installed).
    const partial = join(modelsDir, 'sharded-partial');
    mkdirSync(partial, { recursive: true });
    writeFileSync(join(partial, 'config.json'), Buffer.alloc(4));
    writeFileSync(
      join(partial, 'model.safetensors.index.json'),
      JSON.stringify({
        weight_map: { 'a.weight': 'model-00001-of-00002.safetensors', 'b.weight': 'model-00002-of-00002.safetensors' },
      }),
    );
    writeFileSync(join(partial, 'model-00001-of-00002.safetensors'), Buffer.alloc(8));
    expect(isModelPresent(partial)).toBe(false);
  });
});

describe('a payload must be a REGULAR FILE, not merely a name on disk', () => {
  /**
   * A named pipe. Node has no `mkfifo`, and it is the one shape that turns a
   * wrong answer into a hang: `readFileSync` on a FIFO blocks until a writer
   * appears, and every reader here is on the synchronous request path.
   *
   * Note for whoever breaks this later: a regression does NOT fail these tests,
   * it HANGS them. Blocking happens inside a sync syscall, so vitest's timeout
   * never gets a turn — the run wedges instead of going red. That is the same
   * property the server has, which is the whole reason for the gate.
   */
  function mkfifo(path: string): void {
    execFileSync('mkfifo', [path]);
  }

  function writeMarker(dir: string, files: string[]): void {
    writeFileSync(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: 'org/repo', revision: 'main', completedAt: new Date().toISOString(), files }),
    );
  }

  it('is NOT present when the weight is a directory — the shape a download can really leave', () => {
    // Reachable without touching the disk by hand. `POST /api/downloads` takes an
    // arbitrary repo, a repo file at `model.safetensors/x.safetensors` passes the
    // runner's path filter, and publishing it runs `mkdir(dirname(dest))` — which
    // creates a DIRECTORY named `model.safetensors`. A name-only check called that
    // present, so the card showed a disabled "Installed" and a retry short-circuited
    // to `done`: nothing the user could do from the dashboard, forever.
    const dir = join(modelsDir, 'weight-is-a-dir');
    mkdirSync(join(dir, 'model.safetensors'), { recursive: true });
    writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
    writeFileSync(join(dir, 'model.safetensors', 'x.safetensors'), Buffer.alloc(8));
    expect(isModelPresent(dir)).toBe(false);
  });

  it('is NOT present when config.json is a directory', () => {
    const dir = join(modelsDir, 'config-is-a-dir');
    mkdirSync(join(dir, 'config.json'), { recursive: true });
    writeFileSync(join(dir, 'model.safetensors'), Buffer.alloc(8));
    expect(isModelPresent(dir)).toBe(false);
  });

  it('STAYS present when the weight is a symlink to a real file', () => {
    // The guard on the fix itself. The obvious implementation of "must be a regular
    // file" is `lstat`/`Dirent.isFile()`, and both answer FALSE for a symlink — which
    // is how a HuggingFace snapshot stores every payload, as a link into `blobs/`.
    // That version passes every other test in this block and silently unlists a
    // checkpoint that loads perfectly. Only a FOLLOWING `statSync` gets both right.
    const dir = join(modelsDir, 'weight-is-a-symlink');
    const blobs = join(modelsDir, 'blobs-store');
    mkdirSync(dir, { recursive: true });
    mkdirSync(blobs, { recursive: true });
    writeFileSync(join(blobs, 'sha256-abc'), Buffer.alloc(8));
    writeFileSync(join(blobs, 'sha256-cfg'), Buffer.alloc(4));
    symlinkSync(join(blobs, 'sha256-abc'), join(dir, 'model.safetensors'));
    symlinkSync(join(blobs, 'sha256-cfg'), join(dir, 'config.json'));
    expect(isModelPresent(dir)).toBe(true);
  });

  it('is NOT installed when a file the marker lists is a directory', () => {
    const dir = join(modelsDir, 'marker-lists-a-dir');
    mkdirSync(join(dir, 'model.safetensors'), { recursive: true });
    writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
    writeMarker(dir, ['config.json', 'model.safetensors']);
    expect(isModelInstalled(dir)).toBe(false);
  });

  it('is NOT present when the shard index points at a directory', () => {
    const dir = join(modelsDir, 'shard-is-a-dir');
    mkdirSync(join(dir, 'model-00001-of-00001.safetensors'), { recursive: true });
    writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
    writeFileSync(
      join(dir, 'model.safetensors.index.json'),
      JSON.stringify({ weight_map: { 'a.weight': 'model-00001-of-00001.safetensors' } }),
    );
    expect(isModelPresent(dir)).toBe(false);
  });

  it('RETURNS instead of blocking when the weight or the shard index is a FIFO', () => {
    const weight = join(modelsDir, 'weight-is-a-fifo');
    mkdirSync(weight, { recursive: true });
    writeFileSync(join(weight, 'config.json'), Buffer.alloc(4));
    mkfifo(join(weight, 'model.safetensors'));
    expect(isModelPresent(weight)).toBe(false);

    // The index is the one that used to block: `shardsComplete` read it with a
    // bare `readFileSync(path)`, so a single FIFO under the models dir hung
    // `/api/catalog`, the event loop, and the process's own signal handling —
    // it took SIGKILL to reap. The marker reader already had this gate; these
    // are the rest of it.
    const index = join(modelsDir, 'index-is-a-fifo');
    mkdirSync(index, { recursive: true });
    writeFileSync(join(index, 'config.json'), Buffer.alloc(4));
    mkfifo(join(index, 'model.safetensors.index.json'));
    expect(isModelPresent(index)).toBe(false);
  });

  it('RETURNS instead of blocking when a discovered checkpoint has a FIFO config.json', () => {
    // `discoverLocalModels` walks every child of the models dir and read each
    // `config.json` with a bare `readFileSync`, so this wedged `/api/models` the
    // same way — and needs no catalog entry to reach.
    const dir = join(modelsDir, 'fifo-config');
    mkdirSync(dir, { recursive: true });
    mkfifo(join(dir, 'config.json'));
    const { models, warnings } = discoverLocalModels(modelsDir);
    expect(models.map((m) => m.name)).not.toContain('fifo-config');
    expect(warnings.some((w) => w.startsWith('fifo-config:'))).toBe(true);
  });
});

describe('isDownloaderOwned', () => {
  /** Write a valid completion marker into `dir` (an owned-looking install). */
  function writeOwnedMarker(dir: string): void {
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: 'owner/repo', revision: 'a'.repeat(40), files: ['config.json'], completedAt: 'x' }),
    );
  }

  it('is true for a real directory carrying our marker', () => {
    const owned = join(modelsDir, 'owned');
    writeOwnedMarker(owned);
    expect(isDownloaderOwned(owned)).toBe(true);
  });

  it('keeps a partial or interrupted CLI marker downloader-owned', () => {
    const owned = join(modelsDir, 'partial-owned');
    mkdirSync(owned, { recursive: true });
    writeFileSync(
      join(owned, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: 'owner/repo',
        revision: 'a'.repeat(40),
        files: ['config.json'],
        scope: 'partial',
        completedAt: 'x',
      }),
    );
    expect(isDownloaderOwned(owned)).toBe(true);
    expect(isModelInstalled(owned)).toBe(false);
  });

  it('is false for a regular marker that is not the full shape we write', () => {
    // Ownership authorizes `rename(dir → backup)` + `rm(backup, {recursive:true})`,
    // and it is the ONLY gate on that path — the download route never parses
    // `overwrite`. The marker we write has carried the same four fields since the
    // first commit that emitted one, so none of these is a directory of ours, and
    // treating one as ours means deleting somebody else's checkpoint.
    for (const [name, body] of [
      ['files-only', '{"files":[]}'],
      ['files-not-strings', '{"files":[1,2]}'],
      ['partial-shape', '{"repo":"owner/repo","files":["config.json"]}'],
      ['wrong-types', '{"repo":1,"revision":2,"completedAt":3,"files":[]}'],
    ]) {
      const dir = join(modelsDir, name);
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, DOWNLOAD_COMPLETE_MARKER), body);
      expect([name, isDownloaderOwned(dir)]).toEqual([name, false]);
    }
  });

  // A LIVE symlink whose target is an EXTERNAL directory carrying a valid marker must
  // NOT read as owned: `readFileSync` follows the link and would otherwise report the
  // foreign install as ours, letting the download runner overwrite/report-done through
  // a path we never wrote. The no-follow `lstat` gate refuses any non-directory.
  it('is false for a live symlink to an external marked dir (no-follow ownership)', () => {
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-owned-'));
    writeFileSync(
      join(external, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: 'owner/repo', revision: 'b'.repeat(40), files: ['config.json'], completedAt: 'x' }),
    );
    const link = join(modelsDir, 'linked-final');
    symlinkSync(external, link);

    // The link resolves to a valid foreign marker, yet ownership is refused.
    expect(existsSync(join(link, DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(isDownloaderOwned(link)).toBe(false);

    rmSync(external, { recursive: true, force: true });
  });

  it('is false for a dangling symlink and for an absent path', () => {
    const dangling = join(modelsDir, 'dangling');
    symlinkSync(join(modelsDir, 'no-such-target'), dangling);
    expect(isDownloaderOwned(dangling)).toBe(false);
    expect(isDownloaderOwned(join(modelsDir, 'absent'))).toBe(false);
  });

  // The directory gate is not enough on its own: a REAL unowned directory whose
  // MARKER is a symlink to any parseable marker used to read as owned, and
  // ownership is what lets the publish swap move that directory aside and delete
  // the backup. A link's target is proof about the target, never about the
  // directory holding the link.
  it('is false when the marker itself is a symlink to a foreign marker', () => {
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-marker-'));
    const foreign = join(external, DOWNLOAD_COMPLETE_MARKER);
    writeFileSync(
      foreign,
      JSON.stringify({ repo: 'owner/repo', revision: 'c'.repeat(40), files: ['config.json'], completedAt: 'x' }),
    );
    const victim = join(modelsDir, 'real-unowned-dir');
    mkdirSync(victim, { recursive: true });
    writeFileSync(join(victim, 'PRECIOUS.txt'), 'hand-made, not ours');
    symlinkSync(foreign, join(victim, DOWNLOAD_COMPLETE_MARKER));

    // The link resolves and parses — the only thing refusing it is the no-follow open.
    expect(existsSync(join(victim, DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(isDownloaderOwned(victim)).toBe(false);
    // The identity reader shares the same gate, so the catalog cannot claim the dir either.
    expect(readCompletion(victim)).toBeUndefined();
    expect(isModelInstalled(victim)).toBe(false);

    rmSync(external, { recursive: true, force: true });
  });
});

describe('defaultModelsDir', () => {
  const savedEnv = process.env.MLX_MODELS_DIR;

  afterEach(() => {
    if (savedEnv === undefined) delete process.env.MLX_MODELS_DIR;
    else process.env.MLX_MODELS_DIR = savedEnv;
  });

  it('honors MLX_MODELS_DIR when set', () => {
    process.env.MLX_MODELS_DIR = modelsDir;
    expect(defaultModelsDir()).toBe(modelsDir);
  });
});

/** The default recommendation (Qwen3.6-27B) and the folder a download lands in. */
const RECOMMENDED = MODEL_CATALOG[0]!;
const RECOMMENDED_SLUG = catalogSlug(RECOMMENDED);

/** The catalog row for `label`, from a fresh scan of the temp models dir. */
function catalogItem(label: string): CatalogItem {
  return catalogWithState(modelsDir).find((entry) => entry.label === label)!;
}

/** Write a download completion marker naming `repo` into `<modelsDir>/<name>`. */
function writeCompletion(name: string, repo: string, files = ['config.json', 'model.safetensors']): void {
  const dir = join(modelsDir, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, DOWNLOAD_COMPLETE_MARKER),
    JSON.stringify({ repo, revision: 'a'.repeat(40), files, completedAt: new Date().toISOString() }),
  );
}

describe('catalogWithState — a recommended model is identified by download provenance', () => {
  // The Qwen3.6-27B recommendation is an nvfp4 qwen3_5 checkpoint. `mlx convert`
  // mandates bits=4 / group_size=16 for nvfp4, so that triple is a FORMAT CONSTANT
  // shared by every nvfp4 checkpoint of the family — config shape cannot tell two
  // different weight sets apart, only a recorded repo can.
  const QWEN_27B_NVFP4 = JSON.stringify({
    model_type: 'qwen3_5',
    quantization: { bits: 4, mode: 'nvfp4', group_size: 16 },
  });

  it('does NOT mark present for a look-alike fine-tune with no download provenance', () => {
    // A local fine-tune of the same base, same quant, under a name that starts with
    // the catalog label. Marking it present disables the ONLY Install affordance on
    // the Models page, hard-blocking the user from installing the real checkpoint.
    writeModel(modelsDir, 'qwen3.6-27b-custom', QWEN_27B_NVFP4, 2048);
    expect(catalogItem('Qwen3.6-27B').present).toBe(false);
    expect(catalogItem('Qwen3.6-27B').installed).toBe(false);
  });

  it('marks present for a renamed folder carrying our completion marker for this repo', () => {
    // The capability the folder-prefix heuristic was reaching for, done exactly: a
    // dashboard install the user renamed is still recognized, because the marker
    // pins WHICH repo those bytes came from.
    writeModel(modelsDir, 'renamed-by-hand', QWEN_27B_NVFP4, 2048);
    writeCompletion('renamed-by-hand', RECOMMENDED.hfRepo);
    expect(catalogItem('Qwen3.6-27B').present).toBe(true);
  });

  it('marks present for a complete CLI install under the canonical slug', () => {
    // `mlx download` writes no marker, so provenance comes from the slug itself.
    writeModel(modelsDir, RECOMMENDED_SLUG, QWEN_27B_NVFP4, 2048);
    expect(catalogItem('Qwen3.6-27B').present).toBe(true);
    expect(catalogItem('Qwen3.6-27B').installed).toBe(false);
  });

  it('does NOT cross-match a marker that names a different repo', () => {
    // Same architecture, same quant, marker present — but from another checkpoint.
    const MOE_NVFP4 = JSON.stringify({
      model_type: 'qwen3_5_moe',
      quantization: { bits: 4, mode: 'nvfp4', group_size: 16 },
    });
    writeModel(modelsDir, 'qwen3.6-35b-a3b-nvfp4-mlx', MOE_NVFP4, 2048);
    writeCompletion('qwen3.6-35b-a3b-nvfp4-mlx', 'Brooooooklyn/Qwen3.6-35B-A3B-nvfp4-mlx');
    expect(catalogItem('Qwen-AgentWorld-35B').present).toBe(false);
  });

  it('does NOT mark present when the marker survived but the checkpoint did not', () => {
    // The marker records what was published, not what is still there: a dir gutted
    // down to its marker is not a loadable checkpoint.
    writeCompletion('gutted', RECOMMENDED.hfRepo);
    expect(catalogItem('Qwen3.6-27B').present).toBe(false);
  });
});

describe('catalogWithState — an occupied, unowned slug dir blocks Install', () => {
  /** An interrupted `mlx download`: the index landed, one of two shards did not. */
  function writePartialShardedInstall(name: string): void {
    const dir = join(modelsDir, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
    writeFileSync(
      join(dir, 'model.safetensors.index.json'),
      JSON.stringify({
        weight_map: { 'a.weight': 'model-00001-of-00002.safetensors', 'b.weight': 'model-00002-of-00002.safetensors' },
      }),
    );
    writeFileSync(join(dir, 'model-00001-of-00002.safetensors'), Buffer.alloc(8));
  }

  it('flags an interrupted CLI download as blocked rather than installable', () => {
    // `present` is correctly false (the checkpoint cannot load), but Install would
    // hit the runner's ownership preflight and error every single time, so the card
    // must state the blockage instead of offering the button.
    writePartialShardedInstall(RECOMMENDED_SLUG);
    expect(catalogItem('Qwen3.6-27B').present).toBe(false);
    expect(catalogItem('Qwen3.6-27B').blockedByForeignDir).toBe(true);
  });

  it('flags an occupant with no config.json at all as blocked', () => {
    // Ctrl-C during the CLI's manifest fetch leaves a bare directory. Model
    // discovery folds it into the skipped-directories warning, so this state has no
    // Delete row to recover through — all the more reason not to offer Install.
    mkdirSync(join(modelsDir, RECOMMENDED_SLUG), { recursive: true });
    expect(catalogItem('Qwen3.6-27B').blockedByForeignDir).toBe(true);
  });

  it('flags a dangling symlink at the slug as blocked (no-follow occupancy)', () => {
    // `existsSync` FOLLOWS the link and reads it as absent; the runner's preflight
    // is `lstat`-based and refuses it, so catalog state must be `lstat`-based too.
    symlinkSync(join(modelsDir, 'no-such-target'), join(modelsDir, RECOMMENDED_SLUG));
    expect(catalogItem('Qwen3.6-27B').blockedByForeignDir).toBe(true);
  });

  it('flags a LIVE symlink to an external checkpoint as blocked, not installed', () => {
    // Both discovery walks and the CLI's filter on `Dirent.isDirectory()`, false for a
    // symlink, and delete refuses one outright — so a followed presence check would
    // label "Installed" a model nothing can list, pick or remove, and would suppress
    // the cleanup notice by short-circuiting the occupancy test.
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-ckpt-'));
    writeFileSync(join(external, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
    writeFileSync(join(external, 'model.safetensors'), Buffer.alloc(2048));
    const link = join(modelsDir, RECOMMENDED_SLUG);
    symlinkSync(external, link);
    try {
      // The link is LIVE — without this the case would hold for the wrong reason.
      expect(existsSync(join(link, 'config.json'))).toBe(true);
      expect(isModelPresent(link)).toBe(false);
      expect(catalogItem('Qwen3.6-27B').present).toBe(false);
      expect(catalogItem('Qwen3.6-27B').blockedByForeignDir).toBe(true);
      // The same directory the catalog now calls blocked is absent from discovery.
      expect(discoverLocalModels(modelsDir).models.map((model) => model.name)).not.toContain(RECOMMENDED_SLUG);
    } finally {
      rmSync(external, { recursive: true, force: true });
    }
  });

  it('stays blocked when the occupant fakes our marker with a symlink', () => {
    // This is the state that decides whether the runner may delete the directory:
    // an enabled Install here would move the user's folder to the backup and
    // `rm -rf` it after publishing. A symlinked marker must not buy that.
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-marker2-'));
    const foreign = join(external, DOWNLOAD_COMPLETE_MARKER);
    writeFileSync(
      foreign,
      JSON.stringify({ repo: RECOMMENDED.hfRepo, revision: 'd'.repeat(40), files: ['config.json'], completedAt: 'x' }),
    );
    const victim = join(modelsDir, RECOMMENDED_SLUG);
    mkdirSync(victim, { recursive: true });
    writeFileSync(join(victim, 'PRECIOUS.txt'), 'hand-made, not ours');
    symlinkSync(foreign, join(victim, DOWNLOAD_COMPLETE_MARKER));

    expect(catalogItem('Qwen3.6-27B').present).toBe(false);
    expect(catalogItem('Qwen3.6-27B').blockedByForeignDir).toBe(true);

    rmSync(external, { recursive: true, force: true });
  });

  it('leaves an OWNED but incomplete dir installable', () => {
    // Our marker is there, so the preflight permits the owned swap and a reinstall
    // genuinely works — blocking it would remove the one recovery that functions.
    writeCompletion(RECOMMENDED_SLUG, RECOMMENDED.hfRepo);
    writeFileSync(join(modelsDir, RECOMMENDED_SLUG, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
    expect(catalogItem('Qwen3.6-27B').present).toBe(false);
    expect(catalogItem('Qwen3.6-27B').blockedByForeignDir).toBe(false);
  });

  it('is not blocked when the slug is free, nor when a complete install occupies it', () => {
    expect(catalogItem('Qwen3.6-27B').blockedByForeignDir).toBe(false);
    writeModel(modelsDir, RECOMMENDED_SLUG, JSON.stringify({ model_type: 'qwen3_5' }), 2048);
    expect(catalogItem('Qwen3.6-27B').present).toBe(true);
    expect(catalogItem('Qwen3.6-27B').blockedByForeignDir).toBe(false);
  });
});
