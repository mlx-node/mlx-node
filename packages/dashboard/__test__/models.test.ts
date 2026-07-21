import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { defaultModelsDir, deleteLocalModel, discoverLocalModels } from '../src/models.js';

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
