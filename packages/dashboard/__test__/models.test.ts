import { existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  DOWNLOAD_COMPLETE_MARKER,
  defaultModelsDir,
  deleteLocalModel,
  discoverLocalModels,
  isModelInstalled,
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
});

describe('isModelInstalled', () => {
  /** Write a completion marker listing `files` into `<modelsDir>/<name>`. */
  function writeMarker(name: string, files: string[]): string {
    const dir = join(modelsDir, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({ repo: 'owner/repo', revision: 'a'.repeat(40), files, completedAt: 'x' }),
    );
    return dir;
  }

  it('is installed only when the marker lists a config AND a weight, all present', () => {
    const dir = writeMarker('good', ['config.json', 'model.safetensors']);
    writeFileSync(join(dir, 'config.json'), Buffer.alloc(4));
    writeFileSync(join(dir, 'model.safetensors'), Buffer.alloc(8));
    expect(isModelInstalled(dir)).toBe(true);
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
