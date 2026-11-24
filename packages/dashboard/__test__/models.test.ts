import { execFileSync } from 'node:child_process';
import {
  chmodSync,
  existsSync,
  linkSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  truncateSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';

import { type CatalogBuild, catalogBuildRepo, catalogBuildSlug, MODEL_CATALOG } from '@mlx-node/agent/catalog';
import { discoverLocalChatModels } from '@mlx-node/lm/model-discovery';
import { afterEach, beforeEach, describe, expect, it, onTestFinished } from 'vite-plus/test';

import { modelsDirUnavailableMessage } from '../../server/src/host/paths.js';
import { ApiError } from '../src/api/errors.js';
import { handleModels } from '../src/api/handlers/models.js';
import { type CatalogBuildItem, catalogWithState, heldByOtherBuild } from '../src/catalog.js';
import {
  DOWNLOAD_COMPLETE_MARKER,
  defaultModelsDir,
  deleteLocalModel,
  discoverLocalModels,
  holdsModelFiles,
  isDownloaderOwned,
  isModelInstalled,
  isModelPresent,
  locatedState,
  modelsDirUnavailable,
  readCompletion,
  readLinkText,
  sameForeignEntry,
  summarizeForeignEntry,
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

function ggufHeader(architecture: string): Buffer {
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
  return Buffer.concat([header, string('general.architecture'), stringType, string(architecture)]);
}

describe('GET /api/models', () => {
  const paths = (dir: string) => ({
    modelsDir: dir,
    tracesDir: '',
    runtimeDir: '',
    cacheRoot: undefined,
    prefixCacheRoot: undefined,
  });

  it('servedIds lists the served names of a GGUF directory', async () => {
    const repo = join(modelsDir, 'gguf-repo');
    mkdirSync(repo, { recursive: true });
    writeFileSync(join(repo, 'config.json'), JSON.stringify({ model_type: 'gemma4' }));
    writeFileSync(join(repo, 'tokenizer.json'), '{}');
    writeModel(modelsDir, 'unserved', JSON.stringify({ model_type: 'not_a_chat_family' }), 16);
    for (const name of ['gemma4-Q6_K_XL', 'gemma4-Q4_K_XL', 'gemma4-mmproj-Q4_K_XL', 'gemma4-draft-Q4_K_XL']) {
      writeFileSync(join(repo, `${name}.gguf`), ggufHeader('gemma4'));
    }

    const body = await handleModels(paths(modelsDir));
    const served = Object.fromEntries(body.models.map((m) => [m.name, m.servedIds]));
    expect(served['gguf-repo']).toEqual(['gemma4-Q4_K_XL', 'gemma4-Q6_K_XL']);
    expect(served['model-a']).toEqual(['model-a']);
    // Not a chat family: listed as a checkpoint, served by nothing.
    expect(served.unserved).toEqual([]);
  });

  it('detects Clairic and CLEF sidecars, lists GGUF Clairic without config.json, and populates servedIds', async () => {
    const clairicDir = join(modelsDir, 'clairic-flash-4bit');
    mkdirSync(clairicDir, { recursive: true });
    writeFileSync(join(clairicDir, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
    writeFileSync(join(clairicDir, 'model.safetensors'), Buffer.alloc(64));
    writeFileSync(join(clairicDir, 'clair_head_config.json'), '{}');
    writeFileSync(join(clairicDir, 'clair_head.safetensors'), Buffer.alloc(32));

    const clefDir = join(modelsDir, 'clef-local');
    mkdirSync(clefDir, { recursive: true });
    writeFileSync(join(clefDir, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
    writeFileSync(join(clefDir, 'model.safetensors'), Buffer.alloc(64));
    writeFileSync(join(clefDir, 'joint_head_config.json'), '{}');
    writeFileSync(join(clefDir, 'joint_head.safetensors'), Buffer.alloc(32));

    const ggufClairic = join(modelsDir, 'clairic-gguf');
    mkdirSync(ggufClairic, { recursive: true });
    writeFileSync(join(ggufClairic, 'clairic-UD-Q4_K_XL.gguf'), ggufHeader('qwen35'));
    writeFileSync(join(ggufClairic, 'clair_head_config.json'), '{}');
    writeFileSync(join(ggufClairic, 'clair_head.safetensors'), Buffer.alloc(32));

    const body = await handleModels(paths(modelsDir));
    const byName = Object.fromEntries(body.models.map((m) => [m.name, m]));
    expect(byName['clairic-flash-4bit']?.modelType).toBe('clairic');
    expect(byName['clairic-flash-4bit']?.servedIds).toEqual(['clairic-flash-4bit']);
    expect(byName['clef-local']?.modelType).toBe('clef');
    expect(byName['clef-local']?.servedIds).toEqual(['clef-local']);
    expect(byName['clairic-gguf']?.modelType).toBe('clairic');
    expect(byName['clairic-gguf']?.servedIds).toEqual(['clairic-UD-Q4_K_XL']);
    expect(body.warnings.some((w) => w.includes('clairic-gguf'))).toBe(false);
  });

  it('answers the inventory only: memory and free disk are GET /api/host', async () => {
    const body = await handleModels(paths(modelsDir));
    expect(Object.keys(body).sort()).toEqual(['companions', 'dir', 'models', 'warnings']);
    const missing = await handleModels(paths(join(modelsDir, 'does-not-exist')));
    expect(missing.models).toEqual([]);
  });
});

describe('a models directory that is a link to a missing disk', () => {
  it('is an error with the plain sentence, not an empty library', async () => {
    const link = join(modelsDir, 'models-link');
    const target = join(modelsDir, 'Volumes', 'Disk', 'models');
    symlinkSync(target, link);
    // The engine and the CLI say the same words (`ModelsDirUnavailableError`).
    const sentence = modelsDirUnavailableMessage(link, target);
    expect(sentence).toBe(
      `The models folder ${link} is a link to ${target}, which is not available. Connect that disk, or change the link, then try again.`,
    );
    expect(modelsDirUnavailable(link)).toBe(sentence);
    let caught: unknown;
    try {
      await handleModels({
        modelsDir: link,
        tracesDir: '',
        runtimeDir: '',
        cacheRoot: undefined,
        prefixCacheRoot: undefined,
      });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(ApiError);
    expect(caught).toMatchObject({ code: 'E_UNAVAILABLE', message: sentence, reason: 'models-dir-unavailable' });
    expect(lstatSync(link).isSymbolicLink()).toBe(true);
    expect(existsSync(join(modelsDir, 'Volumes'))).toBe(false);
  });

  it('is not raised for a missing folder, a live link or a real folder', () => {
    expect(modelsDirUnavailable(join(modelsDir, 'does-not-exist'))).toBeNull();
    expect(modelsDirUnavailable(modelsDir)).toBeNull();
    const live = join(modelsDir, 'live-link');
    symlinkSync(modelsDir, live);
    expect(modelsDirUnavailable(live)).toBeNull();
  });
});

describe('discoverLocalModels', () => {
  it('keeps draft-only checkpoints out of models while retaining their storage inventory', () => {
    const config = JSON.stringify({ model_type: 'qwen3', architectures: ['DFlash2DraftModel'] });
    writeModel(modelsDir, 'renamed-companion', config, 128);
    writeModel(modelsDir, 'qwen3.8-27b-dflash2', config, 256);
    // An incomplete draft still needs to be accounted for and removable.
    rmSync(join(modelsDir, 'renamed-companion', 'model.safetensors'));
    const inventory = discoverLocalModels(modelsDir);
    expect(inventory.models.map((model) => model.name)).toEqual(['model-a', 'model-b']);
    expect(inventory.companions).toEqual([
      {
        name: 'qwen3.8-27b-dflash2',
        path: join(modelsDir, 'qwen3.8-27b-dflash2'),
        sizeBytes: Buffer.byteLength(config) + 256,
        fileCount: 2,
      },
      {
        name: 'renamed-companion',
        path: join(modelsDir, 'renamed-companion'),
        sizeBytes: Buffer.byteLength(config),
        fileCount: 1,
      },
    ]);
    expect(inventory.warnings).toHaveLength(1);
    expect(inventory.warnings[0]).toContain('junk');
  });

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

  it('admits a symlinked model directory and sizes its real tree', () => {
    // A Dirent reports the link itself — `isDirectory()` is false for a symlink —
    // so a checkpoint linked into the store (a shared HF checkout, a mount) used
    // to be invisible to listing. It must list like a copied dir, and
    // `walkDirStats` must follow the ROOT link to report the real tree's size.
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-model-'));
    try {
      writeFileSync(join(external, 'config.json'), CONFIG_A);
      writeFileSync(join(external, 'model.safetensors'), Buffer.alloc(WEIGHT_A_BYTES));
      symlinkSync(external, join(modelsDir, 'linked-model'));

      const { models } = discoverLocalModels(modelsDir);
      const linked = models.find((m) => m.name === 'linked-model')!;
      expect(linked).toBeDefined();
      expect(linked.modelType).toBe('qwen3');
      expect(linked.fileCount).toBe(2);
      expect(linked.sizeBytes).toBe(Buffer.byteLength(CONFIG_A) + WEIGHT_A_BYTES);
      // The row says it is a link, so its Delete says that only the link goes.
      expect(linked.link).toBe(true);
      expect(models.filter((m) => m.name !== 'linked-model').map((m) => m.link)).toEqual([false, false]);
      // And the internal no-follow walk still holds: a link inside a model dir is
      // never descended into (covered by the cycle test above).
    } finally {
      rmSync(external, { recursive: true, force: true });
    }
  });

  it('skips a dangling symlink and a symlink to a file without warning', () => {
    // Neither resolves to a directory — they stay invisible, never warnings.
    symlinkSync(join(modelsDir, 'gone'), join(modelsDir, 'dangling'));
    writeFileSync(join(modelsDir, 'plain-file'), 'x');
    symlinkSync(join(modelsDir, 'plain-file'), join(modelsDir, 'file-link'));

    const { models, warnings } = discoverLocalModels(modelsDir);
    expect(models.map((m) => m.name)).toEqual(['model-a', 'model-b']);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain('junk');
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

    expect(() => deleteLocalModel(modelsDir, '.staging')).toThrow('reserved');
    expect(existsSync(staging)).toBe(true);
    expect(existsSync(join(staging, 'in-flight.txt'))).toBe(true);
  });

  it('deletes ONLY the link for a symlinked checkpoint and leaves its target untouched', () => {
    const outside = mkdtempSync(join(tmpdir(), 'dash-outside-'));
    writeFileSync(join(outside, 'config.json'), CONFIG_A);
    writeFileSync(join(outside, 'model.safetensors'), Buffer.alloc(64));
    const keep = join(outside, 'victim.txt');
    writeFileSync(keep, 'important');
    // <modelsDir>/link -> outside: deleting must remove the link, never follow it.
    symlinkSync(outside, join(modelsDir, 'link'));

    deleteLocalModel(modelsDir, 'link');
    // The link itself is gone (lstat throws — not merely a dangling link).
    expect(() => lstatSync(join(modelsDir, 'link'))).toThrow();
    // The symlink target — including a file NOT in the checkpoint set — survives.
    expect(existsSync(keep)).toBe(true);
    expect(existsSync(join(outside, 'config.json'))).toBe(true);
    // And the encoded child-of-a-symlink attack is rejected by the separator guard.
    expect(() => deleteLocalModel(modelsDir, 'link/victim.txt')).toThrow();
    expect(existsSync(keep)).toBe(true);

    rmSync(outside, { recursive: true, force: true });
  });

  it('refuses to delete a symlink whose target is not a checkpoint', () => {
    const outside = mkdtempSync(join(tmpdir(), 'dash-outside-'));
    writeFileSync(join(outside, 'victim.txt'), 'important');
    symlinkSync(outside, join(modelsDir, 'link'));

    expect(() => deleteLocalModel(modelsDir, 'link')).toThrow('not a model checkpoint');
    // The link itself survives — it was refused, not unlinked.
    expect(lstatSync(join(modelsDir, 'link')).isSymbolicLink()).toBe(true);
    expect(existsSync(join(outside, 'victim.txt'))).toBe(true);

    rmSync(outside, { recursive: true, force: true });
  });

  it('refuses to delete a non-checkpoint directory (no config.json) and a stray file', () => {
    // `junk/` (a dir without config.json) and `README.md` (a regular file) both
    // sit directly under modelsDir with safe names, but neither is a checkpoint
    // discovery would list — a typo/hand-crafted DELETE must not rmSync them.
    expect(() => deleteLocalModel(modelsDir, 'junk')).toThrow('not a model checkpoint');
    expect(existsSync(join(modelsDir, 'junk'))).toBe(true);
    expect(() => deleteLocalModel(modelsDir, 'README.md')).toThrow('not a model checkpoint');
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
  it.each(['model.safetensors', 'weights.safetensors'])(
    'is present for a markerless config + %s (CLI/wizard install)',
    (weight) => {
      // Exactly what `mlx download model` leaves on disk: a loadable checkpoint with
      // NO dashboard completion marker. `isModelInstalled` rejects it (unowned), but
      // it must read as PRESENT so the UI does not offer an Install that would then
      // refuse to overwrite the unowned directory.
      const dir = join(modelsDir, 'cli-installed');
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
      writeFileSync(join(dir, weight), Buffer.alloc(8));
      expect(isModelInstalled(dir)).toBe(false);
      expect(isModelPresent(dir)).toBe(true);
    },
  );

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

  it.each(['model.safetensors', 'weights.safetensors'])('is NOT present when %s is a directory', (weight) => {
    // Reachable without touching the disk by hand. `POST /api/downloads` takes an
    // arbitrary repo, a repo file at `model.safetensors/x.safetensors` passes the
    // runner's path filter, and publishing it runs `mkdir(dirname(dest))` — which
    // creates a DIRECTORY named `model.safetensors`. A name-only check called that
    // present, so the card showed a disabled "Installed" and a retry short-circuited
    // to `done`: nothing the user could do from the dashboard, forever.
    const dir = join(modelsDir, 'weight-is-a-dir');
    mkdirSync(join(dir, weight), { recursive: true });
    writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
    writeFileSync(join(dir, weight, 'x.safetensors'), Buffer.alloc(8));
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

  // A LIVE symlink whose target is an EXTERNAL directory carrying a valid marker
  // reads as OWNED: the gate follows the link (a symlinked model dir is admitted
  // everywhere else — listing, presence, loading), so the foreign install's
  // marker is a truthful ownership record. This stays safe because every
  // mutation of the final path acts on the LINK — rename moves it, rm unlinks
  // it — never writes through to the target.
  it('is true for a live symlink to an external marked dir (the link resolves to our marker)', () => {
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-owned-'));
    writeFileSync(
      join(external, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: 'owner/repo', revision: 'b'.repeat(40), files: ['config.json'], completedAt: 'x' }),
    );
    const link = join(modelsDir, 'linked-final');
    symlinkSync(external, link);

    // The link resolves to a valid foreign marker, which now proves ownership.
    expect(existsSync(join(link, DOWNLOAD_COMPLETE_MARKER))).toBe(true);
    expect(isDownloaderOwned(link)).toBe(true);

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

describe('readCompletion — the adopted record', () => {
  const BASE = {
    repo: 'owner/repo',
    revision: 'a'.repeat(40),
    files: ['config.json', 'model.safetensors'],
    scope: 'full',
    completedAt: 'x',
  };

  function writeMarker(dir: string, body: unknown): void {
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, DOWNLOAD_COMPLETE_MARKER), JSON.stringify(body));
  }

  // A record lost in the reader would make a later owned update delete the whole old folder.
  it('reads the adopted record of a marker', () => {
    const dir = join(modelsDir, 'adopted');
    const adopted = {
      'config.json': { size: 12, mtimeNs: '1763700000123456789' },
      'model.safetensors': { size: 0, mtimeNs: '0' },
    };
    writeMarker(dir, { ...BASE, adopted });
    expect(readCompletion(dir)).toEqual({ ...BASE, adopted });
  });

  // A bad record must never unlock a delete, and must never cost the folder its owner.
  it('reads a malformed adopted record as empty, and the marker still counts as owned', () => {
    const good = { size: 12, mtimeNs: '1' };
    const cases: [string, unknown][] = [
      ['string', 'yes'],
      ['array', [good]],
      ['null', null],
      ['entry-not-object', { 'config.json': 12 }],
      ['size-string', { 'config.json': { size: '12', mtimeNs: '1' } }],
      ['size-negative', { 'config.json': { size: -1, mtimeNs: '1' } }],
      ['size-fraction', { 'config.json': { size: 1.5, mtimeNs: '1' } }],
      ['size-unsafe', { 'config.json': { size: 2 ** 53, mtimeNs: '1' } }],
      ['mtime-number', { 'config.json': { size: 12, mtimeNs: 1 } }],
      ['mtime-empty', { 'config.json': { size: 12, mtimeNs: '' } }],
      ['mtime-not-digits', { 'config.json': { size: 12, mtimeNs: '1e9' } }],
      ['one-bad-entry', { 'config.json': good, 'model.safetensors': { size: 4 } }],
    ];
    for (const [name, adopted] of cases) {
      const dir = join(modelsDir, `malformed-${name}`);
      writeMarker(dir, { ...BASE, adopted });
      expect([name, readCompletion(dir)?.adopted]).toEqual([name, {}]);
      expect([name, isDownloaderOwned(dir)]).toEqual([name, true]);
    }
  });

  it('a marker without the record reads as before', () => {
    const dir = join(modelsDir, 'plain');
    writeMarker(dir, BASE);
    const read = readCompletion(dir);
    expect(read).toEqual(BASE);
    expect(read !== undefined && 'adopted' in read).toBe(false);
  });
});

describe('readCompletion — the located record', () => {
  const BASE = {
    repo: 'owner/repo',
    revision: 'a'.repeat(40),
    files: ['config.json', 'model.gguf'],
    scope: 'full',
    completedAt: 'x',
  };

  function writeMarker(dir: string, body: unknown): void {
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, DOWNLOAD_COMPLETE_MARKER), JSON.stringify(body));
  }

  it('reads the located record of a marker', () => {
    const dir = join(modelsDir, 'located');
    for (const mode of ['shared', 'linked']) {
      writeMarker(dir, { ...BASE, located: { from: '/Users/me/models/qwen', mode } });
      expect(readCompletion(dir)).toEqual({ ...BASE, located: { from: '/Users/me/models/qwen', mode } });
    }
  });

  it('reads any other shape as absent, and the marker still counts as owned', () => {
    const cases: [string, unknown][] = [
      ['string', '/Users/me'],
      ['null', null],
      ['array', ['/Users/me', 'shared']],
      ['relative', { from: 'me/models', mode: 'shared' }],
      ['nul', { from: '/a\0b', mode: 'shared' }],
      ['from-number', { from: 1, mode: 'shared' }],
      ['mode-copy', { from: '/a', mode: 'copy' }],
      ['mode-missing', { from: '/a' }],
    ];
    for (const [name, located] of cases) {
      const dir = join(modelsDir, `located-${name}`);
      writeMarker(dir, { ...BASE, located });
      const read = readCompletion(dir);
      expect([name, read !== undefined && 'located' in read]).toEqual([name, false]);
      expect([name, isDownloaderOwned(dir)]).toEqual([name, true]);
      expect([name, locatedState(dir)]).toEqual([name, null]);
    }
  });

  it('reads the state of a located folder: files there, and whether a delete leaves them', () => {
    const user = join(modelsDir, '..', `${basename(modelsDir)}-user`);
    mkdirSync(user, { recursive: true });
    onTestFinished(() => rmSync(user, { recursive: true, force: true }));
    writeFileSync(join(user, 'model.gguf'), 'w');
    const dir = join(modelsDir, 'located-state');
    writeMarker(dir, { ...BASE, located: { from: user, mode: 'shared' } });
    writeFileSync(join(dir, 'config.json'), '{}');
    linkSync(join(user, 'model.gguf'), join(dir, 'model.gguf'));
    expect(locatedState(dir)).toEqual({ from: user, mode: 'shared', filesAvailable: true, filesStay: true });
    // The user removed their name: the app's is the last one, so the folder is a plain install.
    rmSync(join(user, 'model.gguf'));
    expect(locatedState(dir)).toBeNull();
    // A link to a file that is gone: nothing is there to free, and the files are not available.
    rmSync(join(dir, 'model.gguf'));
    symlinkSync(join(user, 'model.gguf'), join(dir, 'model.gguf'));
    writeMarker(dir, { ...BASE, located: { from: user, mode: 'linked' } });
    expect(locatedState(dir)).toEqual({ from: user, mode: 'linked', filesAvailable: false, filesStay: true });
    // `mlx download` put its own new copy in place of the link and kept the record: the folder is a plain install.
    rmSync(join(dir, 'model.gguf'));
    writeFileSync(join(dir, 'model.gguf'), 'w');
    expect(locatedState(dir)).toBeNull();
    // One weight file is the folder's own and one is still a link: located, and a delete frees the own one.
    writeMarker(dir, { ...BASE, files: [...BASE.files, 'part.gguf'], located: { from: user, mode: 'linked' } });
    writeFileSync(join(user, 'part.gguf'), 'p');
    symlinkSync(join(user, 'part.gguf'), join(dir, 'part.gguf'));
    expect(locatedState(dir)).toEqual({ from: user, mode: 'linked', filesAvailable: true, filesStay: false });
    // A link at the slug is never read as a located folder.
    symlinkSync(dir, join(modelsDir, 'located-link'));
    expect(locatedState(join(modelsDir, 'located-link'))).toBeNull();
  });
});

describe('catalog companion state', () => {
  it('recognizes a manual shared companion without claiming ownership or installing the target', () => {
    writeModel(modelsDir, 'qwen3.8-27b-dflash2', JSON.stringify({ architectures: ['DFlash2DraftModel'] }), 32);
    const target = catalogWithState(modelsDir).find((item) => item.label === 'Qwen3.8-27B')!;
    expect(target.builds.some((build) => build.present)).toBe(false);
    expect(target.draft).toMatchObject({
      present: true,
      installed: false,
      blockedByForeignDir: false,
      localRevision: null,
    });
  });

  it('blocks an incomplete manual companion directory and recognizes its completed weights', () => {
    const draft = join(modelsDir, 'qwen3.8-27b-dflash2');
    mkdirSync(draft);
    writeFileSync(join(draft, 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'] }));
    const state = () => catalogWithState(modelsDir).find((item) => item.label === 'Qwen3.8-27B')!.draft;
    expect(state()).toMatchObject({ present: false, blockedByForeignDir: true });
    writeFileSync(join(draft, 'model.safetensors'), 'weights');
    expect(state()).toMatchObject({ present: true, installed: false, blockedByForeignDir: false });
  });

  it('tracks Inkling-2-Mini DFlash2 independently of bundled DSpark fallbacks', () => {
    // Plant Inkling-2-Mini Original with bundled DSpark (`mtp.safetensors`) AND GGUF folder with a DSpark `.gguf`.
    const origDir = join(modelsDir, 'inkling-2-mini');
    mkdirSync(origDir);
    writeFileSync(
      join(origDir, 'config.json'),
      JSON.stringify({
        model_type: 'inkling2_loop',
        mtp_config: { markov_rank: 256, target_layer_ids: [5, 23, 41, 48] },
      }),
    );
    writeFileSync(join(origDir, 'model.safetensors'), 'target');
    writeFileSync(join(origDir, 'mtp.safetensors'), 'dspark');

    const state = () => catalogWithState(modelsDir).find((item) => item.label === 'Inkling-2-Mini')!.draft;
    // Bundled DSpark fallback must NEVER make DFlash2 appear present.
    expect(state()).toMatchObject({
      slug: 'inkling-2-mini-dflash2',
      hfRepo: 'incoai/Inkling-2-Mini-DFlash2',
      autoInstall: true,
      present: false,
      installed: false,
      blockedByForeignDir: false,
    });

    // An incomplete foreign `inkling-2-mini-dflash2` dir still reports as blocked even with bundled DSpark.
    const draft = join(modelsDir, 'inkling-2-mini-dflash2');
    mkdirSync(draft);
    writeFileSync(
      join(draft, 'config.json'),
      JSON.stringify({ architectures: ['DFlash2DraftModel'], hidden_size: 2304, num_target_layers: 48 }),
    );
    expect(state()).toMatchObject({ present: false, installed: false, blockedByForeignDir: true });

    // Completing the DFlash2 companion marks it present (still unowned without completion marker).
    writeFileSync(join(draft, 'model.safetensors'), 'dflash2');
    expect(state()).toMatchObject({
      present: true,
      installed: false,
      blockedByForeignDir: false,
      localRevision: null,
    });
  });

  it('recognizes an embedded DFlash2 draft under a non-first Inkling-2-Mini build', () => {
    const embedded = join(modelsDir, 'inkling-2-mini', 'draft');
    mkdirSync(embedded, { recursive: true });
    writeFileSync(
      join(embedded, 'config.json'),
      JSON.stringify({ architectures: ['DFlash2DraftModel'], hidden_size: 2304, num_target_layers: 48 }),
    );
    writeFileSync(join(embedded, 'model.safetensors'), 'dflash2');
    const draft = catalogWithState(modelsDir).find((item) => item.label === 'Inkling-2-Mini')!.draft;
    expect(draft).toMatchObject({ present: true, installed: false, blockedByForeignDir: false });
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

/** The build `id` of the catalog entry `label`. */
function catalogBuild(label: string, id: string): CatalogBuild {
  const build = MODEL_CATALOG.find((entry) => entry.label === label)?.builds.find((candidate) => candidate.id === id);
  if (build === undefined) throw new Error(`no build ${label} ${id}`);
  return build;
}

/**
 * A build whose repository no other build installs (Inkling-2-Mini Original, a safetensors checkpoint), and the
 * folder a download lands in.
 */
const SOLE = catalogBuild('Inkling-2-Mini', 'Original');
const SOLE_SLUG = catalogBuildSlug(SOLE);
/**
 * The repo THIS platform installs. `catalogWithState` matches provenance
 * against `catalogBuildRepo(build)`, so a marker written with the raw `hfRepo`
 * names the macOS build and silently fails every one of these tests on Linux.
 */
const SOLE_REPO = catalogBuildRepo(SOLE);
/**
 * A repo that holds the SAME model but is not the one this platform installs:
 * the pre-UD-Q4_K_XL CUDA build the catalog used to recommend for this entry.
 * Every current entry resolves to one GGUF repo on every platform, so provenance
 * equivalence between two builds of one model can only be asserted against a
 * literal here.
 */
const RETIRED_PLATFORM_ALIAS = 'Brooooooklyn/Qwen3.8-27B-NVFP4-mlx';

/** Every catalog build with its state, from a fresh scan of the temp models dir. */
function allBuilds(): CatalogBuildItem[] {
  return catalogWithState(modelsDir).flatMap((item) => item.builds);
}

/** The catalog build that installs to the folder `slug`, from a fresh scan of the temp models dir. */
function buildItem(slug: string): CatalogBuildItem {
  return allBuilds().find((build) => build.slug === slug)!;
}

/** Write a download completion marker naming `repo` into `<modelsDir>/<name>`. */
function writeCompletion(
  name: string,
  repo: string,
  files = ['config.json', 'model.safetensors'],
  assets?: { repo: string; revision: string },
): void {
  const dir = join(modelsDir, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, DOWNLOAD_COMPLETE_MARKER),
    JSON.stringify({
      repo,
      revision: 'a'.repeat(40),
      files,
      ...(assets === undefined ? {} : { assetsRepo: assets.repo, assetsRevision: assets.revision }),
      completedAt: new Date().toISOString(),
    }),
  );
}

describe('catalogWithState — a catalog build is identified by download provenance', () => {
  // A checkpoint of mxfp4 qwen3_5 shape. MX block scaling fixes the block at 32,
  // so bits=4 / group_size=32 is a FORMAT CONSTANT shared by every mxfp4
  // checkpoint of the family — config shape cannot tell two different weight
  // sets apart, only a recorded repo can.
  const QWEN_27B_MXFP4 = JSON.stringify({
    model_type: 'qwen3_5',
    quantization: { bits: 4, mode: 'mxfp4', group_size: 32 },
  });

  it('does NOT mark present for a look-alike fine-tune with no download provenance', () => {
    // A local fine-tune of the same base, same quant, under a name that starts with
    // the catalog label. Marking it present disables the ONLY Install affordance on
    // the Models page, hard-blocking the user from installing the real checkpoint.
    writeModel(modelsDir, 'qwen3.8-27b-custom', QWEN_27B_MXFP4, 2048);
    expect(buildItem(SOLE_SLUG).present).toBe(false);
    expect(buildItem(SOLE_SLUG).installed).toBe(false);
  });

  it('marks present for a renamed folder carrying our completion marker for this repo', () => {
    // The capability the folder-prefix heuristic was reaching for, done exactly: a
    // dashboard install the user renamed is still recognized, because the marker
    // pins WHICH repo those bytes came from.
    writeModel(modelsDir, 'renamed-by-hand', QWEN_27B_MXFP4, 2048);
    writeCompletion('renamed-by-hand', SOLE_REPO);
    expect(buildItem(SOLE_SLUG).present).toBe(true);
  });

  it('marks present for a complete CLI install under the canonical slug', () => {
    // `mlx download` writes no marker, so provenance comes from the slug itself.
    writeModel(modelsDir, SOLE_SLUG, QWEN_27B_MXFP4, 2048);
    expect(buildItem(SOLE_SLUG).present).toBe(true);
    expect(buildItem(SOLE_SLUG).installed).toBe(false);
  });

  it('does NOT cross-match a marker that names a different repo', () => {
    // Same architecture, same quant, marker present — but from another checkpoint.
    const MOE_NVFP4 = JSON.stringify({
      model_type: 'qwen3_5_moe',
      quantization: { bits: 4, mode: 'nvfp4', group_size: 16 },
    });
    writeModel(modelsDir, 'qwen3.6-35b-a3b-nvfp4-mlx', MOE_NVFP4, 2048);
    writeCompletion('qwen3.6-35b-a3b-nvfp4-mlx', 'Brooooooklyn/Qwen3.6-35B-A3B-nvfp4-mlx');
    expect(allBuilds().filter((item) => item.present)).toEqual([]);
  });

  it('reports the marker revision as localRevision for an owned canonical install', () => {
    // The read half of the update check: a re-upload lands new bytes at the SAME
    // repo id, so only this recorded sha can tell the user their copy is stale.
    writeModel(modelsDir, SOLE_SLUG, QWEN_27B_MXFP4, 2048);
    writeCompletion(SOLE_SLUG, SOLE_REPO);
    const item = buildItem(SOLE_SLUG);
    expect(item.installed).toBe(true);
    expect(item.localRevision).toBe('a'.repeat(40));
  });

  it('reports localRevision null for a markerless CLI install', () => {
    // No marker means the sha is genuinely unknowable. Null must read as "no
    // update badge", never as "up to date".
    writeModel(modelsDir, SOLE_SLUG, QWEN_27B_MXFP4, 2048);
    const item = buildItem(SOLE_SLUG);
    expect(item.present).toBe(true);
    expect(item.localRevision).toBeNull();
  });

  it('reports localRevision null for a renamed install matched only by provenance', () => {
    // `present` comes from the marker under another folder name, but `installed`
    // is false, and the runner would refuse to re-install over the unowned
    // canonical slug — so an update affordance here could only ever fail.
    writeModel(modelsDir, 'renamed-by-hand', QWEN_27B_MXFP4, 2048);
    writeCompletion('renamed-by-hand', SOLE_REPO);
    const item = buildItem(SOLE_SLUG);
    expect(item.present).toBe(true);
    expect(item.installed).toBe(false);
    expect(item.localRevision).toBeNull();
  });

  it('does NOT mark present for a build of the same model that came from a DIFFERENT repo', () => {
    // Regression: the catalog recommended the nvfp4 builds to EVERY platform
    // before it became platform-conditional, so an existing macOS install is
    // exactly the CUDA alias. Counting it renders the card "Installed" with
    // `installed` false — no Install button and no update affordance — which
    // permanently strands the users this change exists to move onto the
    // Metal-native build.
    //
    // The current catalog serves ONE UD-Q4_K_XL GGUF per model on every platform
    // (no entry sets `hfRepoCuda`), so the "other platform's build" is now any
    // OTHER repo holding the same model — the retired platform alias below. The
    // assertion is unchanged: provenance matching is per RESOLVED repo, so a
    // checkpoint from a repo this platform does not install is never "present".
    const other = RETIRED_PLATFORM_ALIAS;
    expect(other).not.toBe(SOLE_REPO);
    writeModel(modelsDir, 'other-platform-build', QWEN_27B_MXFP4, 2048);
    writeCompletion('other-platform-build', other);
    const item = buildItem(SOLE_SLUG);
    expect(item.present).toBe(false);
    expect(item.installed).toBe(false);
    // And the card therefore still offers the install rather than a dead button.
    expect(item.blockedByForeignDir).toBe(false);
  });

  it('does NOT mark present when the marker survived but the checkpoint did not', () => {
    // The marker records what was published, not what is still there: a dir gutted
    // down to its marker is not a loadable checkpoint.
    writeCompletion('gutted', SOLE_REPO);
    expect(buildItem(SOLE_SLUG).present).toBe(false);
  });
});

describe('catalogWithState — sidecar provenance reaches update discovery', () => {
  it('surfaces the marker assetsRepo/assetsRevision, or null when absent', () => {
    // Update discovery compares BOTH repo/revision pairs: a tokenizer fix in
    // the base model moves nothing in the primary repo, so without this
    // provenance the badge never appears and the repair job cannot run.
    const build = catalogBuild('Hy4-flash', 'Q4_K_M');
    const slug = catalogBuildSlug(build);
    writeModel(modelsDir, slug, CONFIG_A, 2048);
    writeCompletion(slug, catalogBuildRepo(build), ['config.json', 'model.safetensors'], {
      repo: build.assetsRepo!,
      revision: 'b'.repeat(40),
    });

    const item = buildItem(slug);
    expect(item.localAssetsRepo).toBe(build.assetsRepo);
    expect(item.localAssetsRevision).toBe('b'.repeat(40));

    // A marker without the fields (pre-provenance install) reports null, never
    // a value that could compare as "up to date".
    writeModel(modelsDir, 'plain-install', CONFIG_A, 2048);
    writeCompletion('plain-install', 'owner/plain');
    const plain = allBuilds().find((candidate) => candidate.slug === 'plain-install');
    expect(plain).toBeUndefined();
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
    writePartialShardedInstall(SOLE_SLUG);
    expect(buildItem(SOLE_SLUG).present).toBe(false);
    expect(buildItem(SOLE_SLUG).blockedByForeignDir).toBe(true);
  });

  it('flags an occupant with no config.json at all as blocked', () => {
    // Ctrl-C during the CLI's manifest fetch leaves a bare directory. Model
    // discovery folds it into the skipped-directories warning, so this state has no
    // Delete row to recover through — all the more reason not to offer Install.
    mkdirSync(join(modelsDir, SOLE_SLUG), { recursive: true });
    expect(buildItem(SOLE_SLUG).blockedByForeignDir).toBe(true);
  });

  it('flags a dangling symlink at the slug as blocked (no-follow occupancy)', () => {
    // `existsSync` FOLLOWS the link and reads it as absent; the runner's preflight
    // is `lstat`-based and refuses it, so catalog state must be `lstat`-based too.
    symlinkSync(join(modelsDir, 'no-such-target'), join(modelsDir, SOLE_SLUG));
    expect(buildItem(SOLE_SLUG).blockedByForeignDir).toBe(true);
  });

  it('admits a LIVE symlink to an external checkpoint as present, not blocked', () => {
    // Discovery, the loader and delete all resolve a symlinked checkpoint dir —
    // a followed presence check is now the only consistent answer: the card
    // reads Present like a markerless CLI install, and the occupancy fallback
    // never fires (a symlink is not a "foreign dir" to clean up).
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-ckpt-'));
    writeFileSync(join(external, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
    writeFileSync(join(external, 'model.safetensors'), Buffer.alloc(2048));
    const link = join(modelsDir, SOLE_SLUG);
    symlinkSync(external, link);
    try {
      // The link is LIVE — without this the case would hold for the wrong reason.
      expect(existsSync(join(link, 'config.json'))).toBe(true);
      expect(isModelPresent(link)).toBe(true);
      const item = buildItem(SOLE_SLUG);
      expect(item.present).toBe(true);
      expect(item.installed).toBe(false);
      expect(item.blockedByForeignDir).toBe(false);
      // Discovery admits the same linked dir the catalog now calls present.
      expect(discoverLocalModels(modelsDir).models.map((model) => model.name)).toContain(SOLE_SLUG);
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
      JSON.stringify({ repo: SOLE_REPO, revision: 'd'.repeat(40), files: ['config.json'], completedAt: 'x' }),
    );
    const victim = join(modelsDir, SOLE_SLUG);
    mkdirSync(victim, { recursive: true });
    writeFileSync(join(victim, 'PRECIOUS.txt'), 'hand-made, not ours');
    symlinkSync(foreign, join(victim, DOWNLOAD_COMPLETE_MARKER));

    expect(buildItem(SOLE_SLUG).present).toBe(false);
    expect(buildItem(SOLE_SLUG).blockedByForeignDir).toBe(true);

    rmSync(external, { recursive: true, force: true });
  });

  it('leaves an OWNED but incomplete dir installable', () => {
    // Our marker is there, so the preflight permits the owned swap and a reinstall
    // genuinely works — blocking it would remove the one recovery that functions.
    writeCompletion(SOLE_SLUG, SOLE_REPO);
    writeFileSync(join(modelsDir, SOLE_SLUG, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
    expect(buildItem(SOLE_SLUG).present).toBe(false);
    expect(buildItem(SOLE_SLUG).blockedByForeignDir).toBe(false);
  });

  it('is not blocked when the slug is free, nor when a complete install occupies it', () => {
    expect(buildItem(SOLE_SLUG).blockedByForeignDir).toBe(false);
    writeModel(modelsDir, SOLE_SLUG, JSON.stringify({ model_type: 'qwen3_5' }), 2048);
    expect(buildItem(SOLE_SLUG).present).toBe(true);
    expect(buildItem(SOLE_SLUG).blockedByForeignDir).toBe(false);
  });
});

describe('catalogWithState — two builds of one repository', () => {
  const XL = catalogBuild('Qwen3.8-27B', 'UD-Q4_K_XL');
  const M = catalogBuild('Qwen3.8-27B', 'UD-Q4_K_M');
  const REPO = catalogBuildRepo(XL);
  const XL_FILE = 'Qwen3.8-27B-UD-Q4_K_XL.gguf';
  const M_FILE = 'Qwen3.8-27B-UD-Q4_K_M.gguf';
  const QWEN38 = JSON.stringify({ model_type: 'qwen3_5' });

  /** A GGUF install of `file` in the folder `name`, with the shared marker the app and the CLI write. */
  function writeGgufInstall(name: string, file: string): void {
    const dir = join(modelsDir, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'config.json'), QWEN38);
    writeFileSync(join(dir, file), ggufHeader('qwen35'));
    writeCompletion(name, REPO, ['config.json', file]);
  }

  it('an install made before a second build existed stays the first build', () => {
    // Today's folder, file and marker: the first build of a repository keeps its folder.
    expect(catalogBuildSlug(XL)).toBe('qwen3.8-27b-gguf');
    writeGgufInstall('qwen3.8-27b-gguf', XL_FILE);
    expect(buildItem('qwen3.8-27b-gguf')).toMatchObject({
      id: 'UD-Q4_K_XL',
      present: true,
      installed: true,
      heldBy: null,
      blocker: null,
      localRevision: 'a'.repeat(40),
    });
    // A UD-Q4_K_XL marker names the repository, not the build: UD-Q4_K_M is not on disk.
    expect(buildItem('qwen3.8-27b-gguf-ud-q4_k_m')).toMatchObject({
      id: 'UD-Q4_K_M',
      present: false,
      installed: false,
      heldBy: null,
      blocker: null,
      localRevision: null,
    });
  });

  it('a renamed install of one build makes only that build present', () => {
    writeGgufInstall('renamed-by-hand', M_FILE);
    expect(buildItem(catalogBuildSlug(M))).toMatchObject({ present: true, installed: false });
    expect(buildItem(catalogBuildSlug(XL))).toMatchObject({ present: false, installed: false });
  });

  it('a folder is the build it holds', () => {
    // A hand run of `mlx download model -m unsloth/Qwen3.8-27B-GGUF -g '*UD-Q4_K_M*'`
    // lands in the first build's folder and writes the shared marker.
    writeGgufInstall('qwen3.8-27b-gguf', M_FILE);
    expect(buildItem(catalogBuildSlug(M))).toMatchObject({ present: true, installed: false, heldBy: null });
    expect(buildItem(catalogBuildSlug(XL))).toMatchObject({
      present: false,
      installed: false,
      heldBy: 'UD-Q4_K_M',
      blockedByForeignDir: false,
      blocker: null,
      localRevision: null,
    });
    expect(heldByOtherBuild(modelsDir, 'qwen3.8-27b-gguf')).toBe('UD-Q4_K_M');
    expect(heldByOtherBuild(modelsDir, 'qwen3.8-27b-gguf-ud-q4_k_m')).toBeNull();
  });

  it('a build whose repository is its own is never held', () => {
    writeModel(modelsDir, SOLE_SLUG, CONFIG_A, 8);
    expect(buildItem(SOLE_SLUG)).toMatchObject({ present: true, heldBy: null });
    expect(heldByOtherBuild(modelsDir, SOLE_SLUG)).toBeNull();
  });
  it('two installed builds are two folders, and Delete of one leaves the other', () => {
    writeGgufInstall('qwen3.8-27b-gguf', XL_FILE);
    writeGgufInstall('qwen3.8-27b-gguf-ud-q4_k_m', M_FILE);
    expect(buildItem('qwen3.8-27b-gguf')).toMatchObject({ installed: true, heldBy: null });
    expect(buildItem('qwen3.8-27b-gguf-ud-q4_k_m')).toMatchObject({ installed: true, heldBy: null });
    // Each folder is one Local models row, labelled by its build.
    const rows = discoverLocalModels(modelsDir, { includeStats: false }).models.filter((row) =>
      row.name.startsWith('qwen3.8-27b'),
    );
    expect(rows.map((row) => [row.name, row.quant])).toEqual([
      ['qwen3.8-27b-gguf', 'UD-Q4_K_XL'],
      ['qwen3.8-27b-gguf-ud-q4_k_m', 'UD-Q4_K_M'],
    ]);
    deleteLocalModel(modelsDir, 'qwen3.8-27b-gguf-ud-q4_k_m');
    expect(buildItem('qwen3.8-27b-gguf')).toMatchObject({ present: true, installed: true });
    expect(buildItem('qwen3.8-27b-gguf-ud-q4_k_m')).toMatchObject({ present: false, installed: false });
  });

  it('a folder of a model that left the catalog still lists in Local models', () => {
    const dir = join(modelsDir, 'gemma-4-26b-a4b-it-gguf');
    mkdirSync(dir);
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'gemma4' }));
    writeFileSync(join(dir, 'gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf'), ggufHeader('gemma4'));
    writeCompletion('gemma-4-26b-a4b-it-gguf', 'unsloth/gemma-4-26B-A4B-it-GGUF', [
      'config.json',
      'gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf',
    ]);
    expect(allBuilds().map((build) => build.slug)).not.toContain('gemma-4-26b-a4b-it-gguf');
    const row = discoverLocalModels(modelsDir, { includeStats: false }).models.find(
      (model) => model.name === 'gemma-4-26b-a4b-it-gguf',
    );
    expect(row?.quant).toBe('UD-Q4_K_XL');
  });
});

describe('isModelPresent — a companion GGUF is not a loadable checkpoint', () => {
  it('does not count a projector-only directory as present', () => {
    // After the CLI refuses to certify a companion-only selection, the files it
    // downloaded still sit in the canonical slug dir. Counting the projector as
    // present rendered the card Installed-and-disabled for a directory the
    // loader never lists; the directory should read as an occupied unowned one
    // (actionable) instead.
    const dir = join(modelsDir, 'projector-only');
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'config.json'), CONFIG_A);
    writeFileSync(join(dir, 'mmproj-BF16.gguf'), Buffer.alloc(64));
    expect(isModelPresent(dir)).toBe(false);

    // The real target beside it still counts.
    writeFileSync(join(dir, 'gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf'), Buffer.alloc(64));
    expect(isModelPresent(dir)).toBe(true);
  });
});

describe('isModelPresent — a GGUF in a variant folder one level down', () => {
  const QWEN35 = JSON.stringify({ model_type: 'qwen3_5' });
  const TARGET = 'Qwen3.8-27B-UD-Q4_K_XL.gguf';
  const shard = (part: number): string => `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000${part}-of-00004.gguf`;

  function plant(name: string, files: Record<string, string | Buffer>): string {
    const dir = join(modelsDir, name);
    for (const [file, body] of Object.entries(files)) {
      mkdirSync(dirname(join(dir, file)), { recursive: true });
      writeFileSync(join(dir, file), body);
    }
    return dir;
  }

  it('is present with config.json at the root and a target GGUF in the variant folder', () => {
    expect(isModelPresent(plant('single', { 'config.json': QWEN35, [`UD-Q4_K_XL/${TARGET}`]: 'w' }))).toBe(true);
    const split = plant('split', {
      'config.json': QWEN35,
      ...Object.fromEntries([1, 2, 3, 4].map((part) => [`UD-Q4_K_XL/${shard(part)}`, 'w'])),
    });
    expect(isModelPresent(split)).toBe(true);
  });

  it('is not present without the root config.json', () => {
    expect(isModelPresent(plant('no-config', { [`UD-Q4_K_XL/${TARGET}`]: 'w' }))).toBe(false);
  });

  it('is not present when the variant folder holds only companion files', () => {
    const dir = plant('companions', {
      'config.json': QWEN35,
      'UD-Q4_K_XL/mmproj-BF16.gguf': 'x',
      'MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf': 'x',
      'UD-Q4_K_XL/imatrix_unsloth.gguf_file': 'x',
      'dspark/dspark-Inkling-2-Mini-BF16.gguf': 'x',
    });
    expect(isModelPresent(dir)).toBe(false);
  });

  it('counts a split set only from its first part and only with every part beside it', () => {
    const missing = plant('missing-part', {
      'config.json': QWEN35,
      ...Object.fromEntries([1, 2, 4].map((part) => [`UD-Q4_K_XL/${shard(part)}`, 'w'])),
    });
    expect(isModelPresent(missing)).toBe(false);
    const noFirst = plant('no-first-part', {
      'config.json': QWEN35,
      ...Object.fromEntries([2, 3, 4].map((part) => [`UD-Q4_K_XL/${shard(part)}`, 'w'])),
    });
    expect(isModelPresent(noFirst)).toBe(false);
    mkdirSync(join(missing, 'UD-Q4_K_XL', shard(3)));
    expect(isModelPresent(missing)).toBe(false);
  });

  it('follows a link to a variant folder, like the top level, and never a dangling one', () => {
    const external = mkdtempSync(join(tmpdir(), 'dash-variant-'));
    onTestFinished(() => rmSync(external, { recursive: true, force: true }));
    const dir = plant('linked', { 'config.json': QWEN35 });
    symlinkSync(join(external, 'gone'), join(dir, 'UD-Q4_K_XL'));
    expect(isModelPresent(dir)).toBe(false);
    rmSync(join(dir, 'UD-Q4_K_XL'));
    symlinkSync(external, join(dir, 'UD-Q4_K_XL'));
    expect(isModelPresent(dir)).toBe(false);
    writeFileSync(join(external, TARGET), 'w');
    expect(isModelPresent(dir)).toBe(true);
  });

  it('follows a link to a GGUF file in the variant folder', () => {
    const external = mkdtempSync(join(tmpdir(), 'dash-variant-'));
    onTestFinished(() => rmSync(external, { recursive: true, force: true }));
    writeFileSync(join(external, 'blob'), 'w');
    const dir = plant('linked-file', { 'config.json': QWEN35 });
    mkdirSync(join(dir, 'UD-Q4_K_XL'));
    symlinkSync(join(external, 'blob'), join(dir, 'UD-Q4_K_XL', TARGET));
    expect(isModelPresent(dir)).toBe(true);
  });

  it('reads types, never names: a file named like a folder, a folder named like a GGUF', () => {
    expect(isModelPresent(plant('file-folder', { 'config.json': QWEN35, 'UD-Q4_K_XL': 'w' }))).toBe(false);
    const dir = plant('folder-gguf', { 'config.json': QWEN35 });
    mkdirSync(join(dir, 'UD-Q4_K_XL', TARGET), { recursive: true });
    expect(isModelPresent(dir)).toBe(false);
  });

  it('never reads two folders down or a dot-folder', () => {
    expect(isModelPresent(plant('deep', { 'config.json': QWEN35, [`a/UD-Q4_K_XL/${TARGET}`]: 'w' }))).toBe(false);
    expect(isModelPresent(plant('dot', { 'config.json': QWEN35, [`.cache/${TARGET}`]: 'w' }))).toBe(false);
  });

  it('never reads variant folders beside primary safetensors, nor a variant folder that holds them', () => {
    const root = plant('root-shards', {
      'config.json': QWEN35,
      'model-00001-of-00002.safetensors': 'w',
      [`UD-Q4_K_XL/${TARGET}`]: 'w',
    });
    expect(isModelPresent(root)).toBe(false);
    const converted = plant('converted-variant', {
      'config.json': QWEN35,
      'mlx/model.safetensors': 'w',
      [`mlx/${TARGET}`]: 'w',
    });
    expect(isModelPresent(converted)).toBe(false);
  });

  it.each<[string, Record<string, string | Buffer>]>([
    ['a target GGUF in a variant folder', { [`UD-Q4_K_XL/${TARGET}`]: ggufHeader('qwen35') }],
    ['a GGUF two folders down', { [`a/UD-Q4_K_XL/${TARGET}`]: ggufHeader('qwen35') }],
    ['a GGUF in a dot-folder', { [`.cache/${TARGET}`]: ggufHeader('qwen35') }],
    ['a projector in a variant folder', { 'UD-Q4_K_XL/mmproj-BF16.gguf': ggufHeader('clip') }],
    [
      'a GGUF in a variant folder beside root primary safetensors',
      { 'model-00001-of-00002.safetensors': 'w', [`UD-Q4_K_XL/${TARGET}`]: ggufHeader('qwen35') },
    ],
  ])('agrees with discovery on %s', async (_name, files) => {
    const home = mkdtempSync(join(tmpdir(), 'dash-variant-home-'));
    onTestFinished(() => rmSync(home, { recursive: true, force: true }));
    const entry = join(home, 'entry');
    for (const [file, body] of Object.entries({ 'config.json': QWEN35, 'tokenizer.json': '{}', ...files })) {
      mkdirSync(dirname(join(entry, file)), { recursive: true });
      writeFileSync(join(entry, file), body);
    }
    const listsGguf = (await discoverLocalChatModels(home)).some((model) => model.format === 'gguf');
    expect(isModelPresent(entry)).toBe(listsGguf);
  });
});

describe('discoverLocalModels — GGUF weights take their quant label from the GGUF file name', () => {
  /** The fields of `thinkingmachines/Inkling-2-Mini`'s config.json this list reads: a quant block, no position limit. */
  const INKLING_BASE = JSON.stringify({
    model_type: 'inkling2_loop',
    quantization_config: { quant_method: 'modelopt', quant_algo: 'NVFP4' },
    mtp_config: { max_position_embeddings: 1048576 },
    text_config: { model_type: 'inkling2_loop_text', model_max_length: 1048576 },
  });
  const QWEN38 = JSON.stringify({ model_type: 'qwen3_5', text_config: { max_position_embeddings: 262144 } });

  function plant(name: string, files: Record<string, string | Buffer>): void {
    const dir = join(modelsDir, name);
    for (const [file, body] of Object.entries(files)) {
      mkdirSync(dirname(join(dir, file)), { recursive: true });
      writeFileSync(join(dir, file), body);
    }
  }

  function listed(name: string): { quant: string | null; contextWindow: number | null } {
    const model = discoverLocalModels(modelsDir, { includeStats: false }).models.find((m) => m.name === name);
    if (model === undefined) throw new Error(`${name} is not listed`);
    return { quant: model.quant, contextWindow: model.contextWindow };
  }

  it('a root GGUF beside a base config with a quant block takes the build of the file, never the block', () => {
    plant('inkling-2-mini-gguf', {
      'config.json': INKLING_BASE,
      'Inkling-2-Mini-UD-Q4_K_XL.gguf': ggufHeader('inkling2'),
      'mmproj-BF16.gguf': '',
      'dspark-Inkling-2-Mini-Q8_0.gguf': '',
    });
    expect(listed('inkling-2-mini-gguf')).toEqual({ quant: 'UD-Q4_K_XL', contextWindow: 1048576 });
  });

  it('a modelopt safetensors folder takes its algorithm in lower case', () => {
    plant('inkling-2-mini', { 'config.json': INKLING_BASE, 'model-00001-of-00001.safetensors': Buffer.alloc(8) });
    writeFileSync(
      join(modelsDir, 'inkling-2-mini', 'model.safetensors.index.json'),
      JSON.stringify({ weight_map: { 'a.weight': 'model-00001-of-00001.safetensors' } }),
    );
    expect(listed('inkling-2-mini')).toEqual({ quant: 'nvfp4', contextWindow: 1048576 });
  });

  it('a root GGUF beside a base config with no quant block is not full precision', () => {
    plant('qwen3.8-27b-gguf', { 'config.json': QWEN38, 'Qwen3.8-27B-UD-Q4_K_XL.gguf': '' });
    expect(listed('qwen3.8-27b-gguf')).toEqual({ quant: 'UD-Q4_K_XL', contextWindow: 262144 });
  });

  it('a root GGUF whose extension is upper case counts like one in a variant folder', () => {
    plant('upper-root', { 'config.json': QWEN38, 'Qwen3.8-27B-UD-Q4_K_XL.GGUF': '' });
    expect(listed('upper-root').quant).toBe('UD-Q4_K_XL');
    expect(isModelPresent(join(modelsDir, 'upper-root'))).toBe(true);
  });

  it('a root GGUF that is a later split part is no build', () => {
    plant('stray-part', {
      'config.json': QWEN38,
      'Qwen3.8-27B-UD-Q4_K_XL.gguf': '',
      'Qwen3.8-27B-Q8_0-00002-of-00002.gguf': '',
    });
    expect(listed('stray-part').quant).toBe('UD-Q4_K_XL');
  });

  it('a split set in a variant folder takes its build, from the file name or else the folder name', () => {
    plant('qwen3.8-flash-next-gguf', {
      'config.json': QWEN38,
      ...Object.fromEntries(
        [1, 2, 3, 4].map((part) => [`UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000${part}-of-00004.gguf`, '']),
      ),
    });
    expect(listed('qwen3.8-flash-next-gguf').quant).toBe('UD-Q4_K_XL');
    plant('tag-only-file', { 'config.json': QWEN38, 'BF16/model-00001-of-00001.gguf': '' });
    expect(listed('tag-only-file').quant).toBe('BF16');
  });

  it('companion GGUFs never give the label', () => {
    plant('hy4-flash-gguf', {
      'config.json': JSON.stringify({ model_type: 'hy_v4_flash', quantization_config: { quant_method: 'fp8' } }),
      'Hy4-flash-Q4_K_M-00001-of-00002.gguf': '',
      'Hy4-flash-Q4_K_M-00002-of-00002.gguf': '',
      'mmproj-Hy4-flash-bf16.gguf': '',
      'mtp-Hy4-flash-Q8_0.gguf': '',
    });
    expect(listed('hy4-flash-gguf').quant).toBe('Q4_K_M');
    // Companions alone are no GGUF weights: the config's label stays.
    plant('projector-only', {
      'config.json': JSON.stringify({ model_type: 'gemma4', quantization: { mode: 'mxfp4', bits: 4 } }),
      'mmproj-BF16.gguf': '',
      'UD-Q4_K_XL/mmproj-F16.gguf': '',
    });
    expect(listed('projector-only').quant).toBe('mxfp4');
  });

  it('a folder whose weights are safetensors keeps the config label, a GGUF beside them included', () => {
    expect(listed('model-a').quant).toBeNull();
    expect(listed('model-b').quant).toBe('affine-4bit');
    plant('converted', {
      'config.json': JSON.stringify({ model_type: 'qwen3_5', quantization: { mode: 'mxfp4', bits: 4 } }),
      'model-00001-of-00002.safetensors': '',
      'Qwen3.8-27B-UD-Q4_K_XL.gguf': '',
      'UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf': '',
    });
    expect(listed('converted').quant).toBe('mxfp4');
  });

  it('a folder with two builds names none of them, at the root or in variant folders', () => {
    plant('two-root', { 'config.json': QWEN38, 'Qwen3.8-27B-Q4_K_M.gguf': '', 'Qwen3.8-27B-Q8_0.gguf': '' });
    expect(listed('two-root').quant).toBe('GGUF');
    plant('two-variants', {
      'config.json': QWEN38,
      'Q4_K_M/Qwen3.8-27B-Q4_K_M.gguf': '',
      'Q8_0/Qwen3.8-27B-Q8_0.gguf': '',
    });
    expect(listed('two-variants').quant).toBe('GGUF');
    // A variant folder whose split set misses a part is not a build.
    plant('one-whole-variant', {
      'config.json': QWEN38,
      'Q4_K_M/Qwen3.8-27B-Q4_K_M.gguf': '',
      'Q8_0/Qwen3.8-27B-Q8_0-00001-of-00002.gguf': '',
    });
    expect(listed('one-whole-variant').quant).toBe('Q4_K_M');
    // A name that states no build.
    plant('unnamed', { 'config.json': QWEN38, 'model.gguf': '' });
    expect(listed('unnamed').quant).toBe('GGUF');
  });

  it.each<[string, Record<string, unknown>, number | null]>([
    [
      'root max_position_embeddings over everything',
      {
        max_position_embeddings: 40960,
        model_max_length: 8192,
        text_config: { max_position_embeddings: 1, model_max_length: 2 },
      },
      40960,
    ],
    [
      'text_config max_position_embeddings over both model_max_length',
      { model_max_length: 8192, text_config: { max_position_embeddings: 262144, model_max_length: 2 } },
      262144,
    ],
    [
      'text_config model_max_length over the root one',
      { model_max_length: 8192, text_config: { model_max_length: 1048576 } },
      1048576,
    ],
    ['root model_max_length last', { model_max_length: 8192 }, 8192],
    [
      'a model_max_length that is not a positive safe integer',
      { model_max_length: 1e30, text_config: { model_max_length: 4096.5 } },
      null,
    ],
    ['a zero or negative model_max_length', { model_max_length: 0, text_config: { model_max_length: -1 } }, null],
    ['nothing stated', {}, null],
  ])('context window: %s', (_name, fields, window) => {
    writeModel(modelsDir, 'ctx', JSON.stringify({ model_type: 'qwen3', ...fields }), 8);
    expect(listed('ctx').contextWindow).toBe(window);
  });
});

describe('isWeightFile — companion GGUFs (projector, calibration, draft, MTP) are not a model payload', () => {
  it('rejects companion names that discovery also excludes', async () => {
    const { isWeightFile } = await import('../src/models.js');
    // A Gemma-style manifest can carry mmproj without the UD target (partial
    // upstream upload, renamed file). Publishing on the projector alone would
    // certify a directory model-discovery never lists as a model.
    // Exactly the names discovery excludes (its GGUF_COMPANION_TOKENS list) —
    // kept in sync deliberately: a name discovery treats as a candidate is a
    // legitimate payload here too, or the two would disagree.
    expect(isWeightFile('mmproj-BF16.gguf')).toBe(false);
    expect(isWeightFile('imatrix_unsloth.gguf')).toBe(false);
    expect(isWeightFile('draft-qwen3.8-27b.gguf')).toBe(false);
    expect(isWeightFile('something.dflash.gguf')).toBe(false);
    // The catalog ships MTP weights BESIDE a target and nothing pairs a
    // standalone GGUF MTP file: counting one as weights certifies a directory
    // the loader cannot open.
    expect(isWeightFile('mtp-gemma-4-26B-A4B-it.gguf')).toBe(false);
    expect(isWeightFile('MTP/mtp-Qwen3.8-27B-Q4_0.gguf')).toBe(false);

    // The real targets and non-GGUF weights stay payloads.
    expect(isWeightFile('gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf')).toBe(true);
    expect(isWeightFile('Qwen3.8-27B-UD-Q4_K_XL.gguf')).toBe(true);
    expect(isWeightFile('model.safetensors')).toBe(true);
    expect(isWeightFile('model.pdiparams')).toBe(true);
  });
});

describe('catalog blocker: what is in the way, measured without following or opening', () => {
  const QWEN = MODEL_CATALOG.find((entry) => entry.label === 'Qwen3.8-27B')!;
  const QWEN_SLUG = catalogBuildSlug(QWEN.builds[0]);
  const slugDir = (): string => join(modelsDir, QWEN_SLUG);
  const draftDir = (): string => join(modelsDir, QWEN.draft!.hfRepo.split('/')[1]!.toLowerCase());

  it.each<[string, () => void, boolean]>([
    ['absent', () => {}, false],
    ['a foreign directory', () => mkdirSync(slugDir()), true],
    [
      'an owned partial directory',
      () => writeCompletion(QWEN_SLUG, catalogBuildRepo(QWEN.builds[0]), ['config.json', 'model.safetensors']),
      false,
    ],
    ['a present CLI install', () => writeModel(modelsDir, QWEN_SLUG, CONFIG_A, 8), false],
    ['a dangling link', () => symlinkSync(join(modelsDir, 'nowhere'), slugDir()), true],
    ['a file', () => writeFileSync(slugDir(), 'x'), true],
    [
      'a foreign draft directory',
      () => {
        mkdirSync(draftDir());
        writeFileSync(join(draftDir(), 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'] }));
      },
      false,
    ],
  ])('blocker is set exactly when blockedByForeignDir is (%s)', (name, plant, blocked) => {
    plant();
    const items = catalogWithState(modelsDir);
    expect(items.find((item) => item.label === QWEN.label)!.builds[0].blockedByForeignDir).toBe(blocked);
    for (const item of items) {
      for (const build of item.builds) expect(build.blocker !== null, build.slug).toBe(build.blockedByForeignDir);
      if (item.draft !== undefined)
        expect(item.draft.blocker !== null, item.draft.slug).toBe(item.draft.blockedByForeignDir);
    }
    expect(items.some((item) => item.draft?.blockedByForeignDir === true)).toBe(name === 'a foreign draft directory');
  });

  it('describes a one-file leftover directory', () => {
    mkdirSync(slugDir());
    writeFileSync(join(slugDir(), 'imatrix_unsloth.gguf_file'), Buffer.alloc(183));
    expect(buildItem(QWEN_SLUG).blocker).toEqual({
      kind: 'directory',
      fileCount: 1,
      sizeBytes: 183,
      truncated: false,
      names: ['imatrix_unsloth.gguf_file'],
      listed: false,
      modelLike: false,
      holdsModel: false,
      linkTarget: null,
      dangling: false,
      linkToFile: false,
      locked: false,
    });
  });

  it.each<[string, number]>([
    ['cannot be read or written', 0o000],
    ['can be read but not written', 0o500],
  ])('marks a folder the app has no write permission on as locked: one that %s', (_name, mode) => {
    mkdirSync(slugDir());
    writeFileSync(join(slugDir(), 'notes.txt'), 'x');
    chmodSync(slugDir(), mode);
    try {
      expect(buildItem(QWEN_SLUG).blocker).toMatchObject({ kind: 'directory', locked: true });
    } finally {
      chmodSync(slugDir(), 0o755);
    }
    expect(buildItem(QWEN_SLUG).blocker).toMatchObject({ kind: 'directory', locked: false });
  });

  it('never marks a link or a file as locked, whatever its target or mode', () => {
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-'));
    onTestFinished(() => {
      chmodSync(external, 0o755);
      rmSync(external, { recursive: true, force: true });
    });
    chmodSync(external, 0o500);
    symlinkSync(external, slugDir());
    expect(buildItem(QWEN_SLUG).blocker).toMatchObject({ kind: 'symlink', locked: false, linkTarget: external });
    rmSync(slugDir());
    writeFileSync(slugDir(), 'x');
    chmodSync(slugDir(), 0o400);
    expect(buildItem(QWEN_SLUG).blocker).toMatchObject({ kind: 'file', locked: false });
  });

  it('cuts the link text at 200 characters, and compares it when it matches an echo', () => {
    const long = `${'a'.repeat(250)}`;
    symlinkSync(long, slugDir());
    expect(readLinkText(slugDir())).toBe(`${'a'.repeat(200)}…`);
    expect(readLinkText(modelsDir)).toBeNull();
    const now = buildItem(QWEN_SLUG).blocker!;
    expect(now.linkTarget).toBe(`${'a'.repeat(200)}…`);
    expect(sameForeignEntry(now, { ...now })).toBe(true);
    expect(sameForeignEntry(now, { ...now, linkTarget: 'b' })).toBe(false);
    expect(sameForeignEntry(now, { ...now, linkTarget: null })).toBe(false);
  });

  it('says whether Local models lists the folder of a build', () => {
    expect(buildItem(QWEN_SLUG).listed).toBe(false);
    writeModel(modelsDir, QWEN_SLUG, CONFIG_A, 8);
    expect(buildItem(QWEN_SLUG).listed).toBe(true);
    rmSync(slugDir(), { recursive: true });
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-'));
    onTestFinished(() => rmSync(external, { recursive: true, force: true }));
    writeFileSync(join(external, GGUF), ggufHeader('gemma4'));
    symlinkSync(external, slugDir());
    expect(buildItem(QWEN_SLUG)).toMatchObject({ listed: false, blocker: { kind: 'symlink', listed: false } });
    writeFileSync(join(external, GGUF), ggufHeader('qwen35'));
    expect(buildItem(QWEN_SLUG)).toMatchObject({ listed: true, present: true, blocker: null });
  });

  it('names up to three files, sorted, with their relative paths; none for four', () => {
    const dir = join(modelsDir, 'three');
    mkdirSync(join(dir, 'sub'), { recursive: true });
    writeFileSync(join(dir, 'b.bin'), 'bb');
    writeFileSync(join(dir, 'a.bin'), 'a');
    writeFileSync(join(dir, 'sub', 'c.bin'), 'ccc');
    expect(summarizeForeignEntry(dir)).toMatchObject({
      fileCount: 3,
      sizeBytes: 6,
      names: ['a.bin', 'b.bin', 'sub/c.bin'],
    });
    writeFileSync(join(dir, 'd.bin'), 'd');
    expect(summarizeForeignEntry(dir)).toMatchObject({ fileCount: 4, sizeBytes: 7, names: [] });
  });

  it('counts no files in an empty directory or one with only empty subdirectories', () => {
    const dir = join(modelsDir, 'empty');
    mkdirSync(dir);
    expect(summarizeForeignEntry(dir)).toMatchObject({
      kind: 'directory',
      fileCount: 0,
      sizeBytes: 0,
      truncated: false,
    });
    mkdirSync(join(dir, 'a', 'b'), { recursive: true });
    expect(summarizeForeignEntry(dir)).toMatchObject({ fileCount: 0, sizeBytes: 0, names: [] });
  });

  it('marks a config.json directory listed only when the config is a JSON object', () => {
    const dir = join(modelsDir, 'half');
    mkdirSync(dir);
    writeFileSync(join(dir, 'config.json'), CONFIG_A);
    expect(summarizeForeignEntry(dir)).toMatchObject({ listed: true, modelLike: true });
    writeFileSync(join(dir, 'config.json'), '[1, 2]');
    expect(summarizeForeignEntry(dir)).toMatchObject({ listed: false, modelLike: true });
  });

  const SHARD_1 = 'model-00001-of-00002.safetensors';
  const SHARD_2 = 'model-00002-of-00002.safetensors';
  const INDEX = 'model.safetensors.index.json';
  const GGUF = 'Qwen3.8-27B-UD-Q4_K_XL.gguf';
  const indexOf = (shards: string[]): string =>
    JSON.stringify({ weight_map: Object.fromEntries(shards.map((shard, i) => [`w${i}`, shard])) });
  type Layout = Record<string, string | Buffer>;

  function plantLayout(dir: string, layout: Layout): void {
    mkdirSync(dir, { recursive: true });
    for (const [file, body] of Object.entries(layout)) {
      mkdirSync(dirname(join(dir, file)), { recursive: true });
      writeFileSync(join(dir, file), body);
    }
  }

  it.each<[string, Layout]>([
    ['shards with no index', { 'config.json': CONFIG_A, [SHARD_1]: 'w', [SHARD_2]: 'w' }],
    [
      'shards named model.safetensors-N-of-M with no index',
      {
        'config.json': CONFIG_A,
        'model.safetensors-00001-of-00002.safetensors': 'w',
        'model.safetensors-00002-of-00002.safetensors': 'w',
      },
    ],
    ['complete shards with an index that does not parse', { 'config.json': CONFIG_A, [INDEX]: '{', [SHARD_1]: 'w' }],
    ['complete shards with an empty weight_map', { 'config.json': CONFIG_A, [INDEX]: indexOf([]), [SHARD_1]: 'w' }],
    [
      'an index naming a missing shard',
      { 'config.json': CONFIG_A, [INDEX]: indexOf([SHARD_1, SHARD_2]), [SHARD_1]: 'w' },
    ],
    ['model.safetensors', { 'config.json': CONFIG_A, 'model.safetensors': 'w' }],
    ['weights.safetensors', { 'config.json': CONFIG_A, 'weights.safetensors': 'w' }],
    ['consolidated.safetensors', { 'config.json': CONFIG_A, 'consolidated.safetensors': 'w' }],
    ['weights-N-of-M shards', { 'config.json': CONFIG_A, 'weights-00001-of-00002.safetensors': 'w' }],
    ['only a config.json', { 'config.json': CONFIG_A }],
    ['weights in a folder under a root config', { 'config.json': CONFIG_A, 'sub/model.safetensors': 'w' }],
    ['a GGUF at the root and no config.json', { [GGUF]: ggufHeader('qwen35') }],
    [
      'a config.json and a GGUF in a variant folder',
      { 'config.json': JSON.stringify({ model_type: 'qwen3_5' }), [`UD-Q4_K_XL/${GGUF}`]: ggufHeader('qwen35') },
    ],
  ])('holds model files for a layout mlx serve lists: %s', async (_name, layout) => {
    const home = mkdtempSync(join(tmpdir(), 'dash-serve-'));
    onTestFinished(() => rmSync(home, { recursive: true, force: true }));
    const entry = join(home, 'entry');
    plantLayout(entry, layout);
    const served = (await discoverLocalChatModels(home)).map((model) => model.path);
    expect(served.length).toBeGreaterThan(0);
    expect(served.every((path) => path.startsWith(entry))).toBe(true);
    expect(holdsModelFiles(entry)).toBe(true);
    expect(summarizeForeignEntry(entry)).toMatchObject({ kind: 'directory', holdsModel: true });
  });

  it.each<[string, Layout]>([
    [
      'a config.json and model.safetensors only in a variant folder',
      { 'v/config.json': CONFIG_A, 'v/model.safetensors': 'w' },
    ],
    ['shards with no index only in a variant folder', { [`v/${SHARD_1}`]: 'w', [`v/${SHARD_2}`]: 'w' }],
    ['only companion GGUFs', { 'imatrix_unsloth.gguf': 'x', 'mmproj-F16.gguf': 'x', 'UD-Q4_K_XL/m-mtp.gguf': 'x' }],
    ['a GGUF with an upper-case extension', { 'm-Q8_0.GGUF': 'x' }],
    ['a GGUF in a dot-folder', { '.cache/m-UD-Q4_K_XL.gguf': 'x' }],
    ['a GGUF two folders down', { 'a/b/m-UD-Q4_K_XL.gguf': 'x' }],
    ['PyTorch weights', { 'pytorch_model.bin': 'x' }],
    ['Paddle weights', { 'inference.pdiparams': 'x' }],
    ['MLX arrays', { 'weights.npz': 'x' }],
  ])('holds model files for weights mlx serve does not list: %s', async (_name, layout) => {
    const home = mkdtempSync(join(tmpdir(), 'dash-serve-'));
    onTestFinished(() => rmSync(home, { recursive: true, force: true }));
    const entry = join(home, 'entry');
    plantLayout(entry, layout);
    expect(await discoverLocalChatModels(home)).toEqual([]);
    expect(holdsModelFiles(entry)).toBe(true);
  });

  it.each<[string, Layout]>([
    ['a calibration file that is not a GGUF', { 'imatrix_unsloth.gguf_file': 'x' }],
    ['a README', { 'README.md': '# x' }],
    ['an empty directory', {}],
    ['a marker the app did not write', { [DOWNLOAD_COMPLETE_MARKER]: '{"files":[]}' }],
    [
      'Hugging Face download metadata and a tokenizer',
      { '.cache/huggingface/download/model.safetensors.metadata': 'x', 'tokenizer.json': '{}' },
    ],
  ])('holds no model files for a leftover: %s', (_name, layout) => {
    const entry = join(modelsDir, 'leftover');
    plantLayout(entry, layout);
    expect(holdsModelFiles(entry)).toBe(false);
    expect(summarizeForeignEntry(entry)).toMatchObject({ kind: 'directory', holdsModel: false });
  });

  it('follows a link to a folder inside the entry, and stops at a link cycle', () => {
    const external = mkdtempSync(join(tmpdir(), 'dash-holds-'));
    onTestFinished(() => rmSync(external, { recursive: true, force: true }));
    const entry = join(modelsDir, 'linked');
    mkdirSync(entry);
    symlinkSync(entry, join(entry, 'self'));
    expect(holdsModelFiles(entry)).toBe(false);
    symlinkSync(external, join(entry, 'UD-Q4_K_XL'));
    writeFileSync(join(external, GGUF), 'x');
    expect(holdsModelFiles(entry)).toBe(true);
  });

  it('marks a model through a link at the root, and never for a file', () => {
    const external = mkdtempSync(join(tmpdir(), 'dash-holds-'));
    onTestFinished(() => rmSync(external, { recursive: true, force: true }));
    mkdirSync(join(external, 'UD-Q4_K_XL'));
    writeFileSync(join(external, 'UD-Q4_K_XL', 'm-UD-Q4_K_XL.gguf'), 'x');
    symlinkSync(external, slugDir());
    expect(summarizeForeignEntry(slugDir())).toMatchObject({ kind: 'symlink', fileCount: 0, holdsModel: true });
    rmSync(slugDir());
    writeFileSync(slugDir(), 'x');
    expect(summarizeForeignEntry(slugDir())).toMatchObject({ kind: 'file', holdsModel: false });
  });

  it('holds model files for a config whose name differs only in case, which a case-insensitive volume opens', () => {
    const entry = join(modelsDir, 'cased');
    plantLayout(entry, { 'Config.json': CONFIG_A });
    expect(holdsModelFiles(entry)).toBe(true);
  });

  it('counts a complete checkpoint as a model too', () => {
    writeModel(modelsDir, 'checkpoint', CONFIG_A, 8);
    expect(holdsModelFiles(join(modelsDir, 'checkpoint'))).toBe(true);
  });

  it('describes a file at the slug path by its own size', () => {
    writeFileSync(slugDir(), Buffer.alloc(40));
    expect(buildItem(QWEN_SLUG).blocker).toMatchObject({ kind: 'file', fileCount: 1, sizeBytes: 40, names: [] });
  });

  it('never follows a link: dangling or live, its counts are 0', () => {
    const target = join(modelsDir, 'nowhere');
    symlinkSync(target, slugDir());
    expect(buildItem(QWEN_SLUG).blocker).toMatchObject({
      kind: 'symlink',
      dangling: true,
      linkToFile: false,
      linkTarget: target,
      fileCount: 0,
      sizeBytes: 0,
    });
    rmSync(slugDir());
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-'));
    onTestFinished(() => rmSync(external, { recursive: true, force: true }));
    writeFileSync(join(external, 'big.bin'), Buffer.alloc(1024 * 1024));
    symlinkSync(external, slugDir());
    expect(buildItem(QWEN_SLUG).blocker).toMatchObject({
      kind: 'symlink',
      dangling: false,
      linkToFile: false,
      fileCount: 0,
      sizeBytes: 0,
    });
    rmSync(slugDir());
    symlinkSync(join(external, 'big.bin'), slugDir());
    expect(buildItem(QWEN_SLUG).blocker).toMatchObject({
      kind: 'symlink',
      dangling: false,
      linkToFile: true,
      fileCount: 0,
      sizeBytes: 0,
    });
  });

  it('counts a link and a FIFO inside as files of no size, without opening them', () => {
    const dir = join(modelsDir, 'odd');
    mkdirSync(dir);
    const external = mkdtempSync(join(tmpdir(), 'dash-ext-'));
    onTestFinished(() => rmSync(external, { recursive: true, force: true }));
    const big = join(external, 'big.bin');
    writeFileSync(big, '');
    truncateSync(big, 64 * 1024 * 1024);
    symlinkSync(big, join(dir, 'link.bin'));
    execFileSync('mkfifo', [join(dir, 'pipe')]);
    writeFileSync(join(dir, 'real.bin'), 'abc');
    expect(summarizeForeignEntry(dir)).toMatchObject({ fileCount: 3, sizeBytes: 3, truncated: false });
  });
});
