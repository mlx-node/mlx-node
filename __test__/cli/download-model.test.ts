import { mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { isGlobVariantPresent, isModelAlreadyDownloaded } from '../../packages/cli/src/commands/download-model.js';

describe('isModelAlreadyDownloaded', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-download-test-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  function write(name: string, contents: string): void {
    writeFileSync(join(dir, name), contents);
  }

  it('returns false when config.json is missing', () => {
    write('model.safetensors', 'x');
    expect(isModelAlreadyDownloaded(dir, readdirSync(dir))).toBe(false);
  });

  it('returns true for a single-file safetensors model with config', () => {
    write('config.json', '{}');
    write('model.safetensors', 'x');
    expect(isModelAlreadyDownloaded(dir, readdirSync(dir))).toBe(true);
  });

  it('returns true for a Paddle model (inference.pdiparams) with config', () => {
    write('config.json', '{}');
    write('inference.pdiparams', 'x');
    expect(isModelAlreadyDownloaded(dir, readdirSync(dir))).toBe(true);
  });

  it('returns false when a sharded index references shards that are missing on disk', () => {
    // Regression: previously the early-return only checked that
    // model.safetensors.index.json was present. An interrupted prior
    // download that landed the index but not all shards would silently
    // be declared "already downloaded".
    write('config.json', '{}');
    write(
      'model.safetensors.index.json',
      JSON.stringify({
        metadata: { total_size: 12345 },
        weight_map: {
          'layer.0.weight': 'model-00001-of-00002.safetensors',
          'layer.1.weight': 'model-00002-of-00002.safetensors',
        },
      }),
    );
    // Only the first shard exists; the second is missing.
    write('model-00001-of-00002.safetensors', 'shard-1');

    expect(isModelAlreadyDownloaded(dir, readdirSync(dir))).toBe(false);
  });

  it('returns true for a sharded model when ALL referenced shards exist', () => {
    write('config.json', '{}');
    write(
      'model.safetensors.index.json',
      JSON.stringify({
        metadata: { total_size: 12345 },
        weight_map: {
          'layer.0.weight': 'model-00001-of-00002.safetensors',
          'layer.1.weight': 'model-00002-of-00002.safetensors',
          'layer.2.weight': 'model-00002-of-00002.safetensors', // duplicate target dedups
        },
      }),
    );
    write('model-00001-of-00002.safetensors', 'shard-1');
    write('model-00002-of-00002.safetensors', 'shard-2');

    expect(isModelAlreadyDownloaded(dir, readdirSync(dir))).toBe(true);
  });

  it('returns false when the index file is malformed JSON', () => {
    write('config.json', '{}');
    write('model.safetensors.index.json', '{not json');
    expect(isModelAlreadyDownloaded(dir, readdirSync(dir))).toBe(false);
  });

  it('returns false when the index file lacks weight_map', () => {
    write('config.json', '{}');
    write('model.safetensors.index.json', JSON.stringify({ metadata: { total_size: 0 } }));
    expect(isModelAlreadyDownloaded(dir, readdirSync(dir))).toBe(false);
  });

  it('returns false when weight_map is empty', () => {
    write('config.json', '{}');
    write('model.safetensors.index.json', JSON.stringify({ weight_map: {} }));
    expect(isModelAlreadyDownloaded(dir, readdirSync(dir))).toBe(false);
  });

  it('still considers single-file safetensors complete even alongside an unverified index', () => {
    // If both `model.safetensors` and `model.safetensors.index.json` are
    // present, the single file wins — no need to parse the index.
    write('config.json', '{}');
    write('model.safetensors', 'x');
    write('model.safetensors.index.json', JSON.stringify({ weight_map: { x: 'never-existed.safetensors' } }));
    expect(isModelAlreadyDownloaded(dir, readdirSync(dir))).toBe(true);
  });
});

describe('isGlobVariantPresent', () => {
  it('returns false when no patterns are provided', () => {
    expect(isGlobVariantPresent(['config.json', 'tokenizer.json', 'model.Q8_0.gguf'], [])).toBe(false);
  });

  it('returns false when a prior Q8 download leaves only CORE_FILES + a non-matching gguf', () => {
    // Regression: previously the early-return counted CORE_FILES toward
    // the "matched" set, so any prior gguf download (which lays down
    // config.json + tokenizer.json) auto-satisfied the >1 threshold and
    // a fresh `--glob "*Q4*"` exited as "already downloaded" without
    // ever fetching the Q4 weights. The helper must look ONLY at user-
    // glob matches.
    const files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.Q8_0.gguf'];
    expect(isGlobVariantPresent(files, ['*Q4*'])).toBe(false);
  });

  it('returns true when an existing file matches one of the glob patterns', () => {
    const files = ['config.json', 'tokenizer.json', 'model.Q4_K_M.gguf'];
    expect(isGlobVariantPresent(files, ['*Q4*'])).toBe(true);
  });

  it('returns true when ANY pattern matches (multi-glob OR semantics)', () => {
    const files = ['config.json', 'model.Q8_0.gguf'];
    expect(isGlobVariantPresent(files, ['*Q4*', '*Q8*'])).toBe(true);
  });

  it('returns false when no file matches any pattern (CORE_FILES alone do not count)', () => {
    const files = ['config.json', 'tokenizer.json', 'tokenizer_config.json'];
    expect(isGlobVariantPresent(files, ['*BF16*'])).toBe(false);
  });

  it('matches case-insensitively (gguf repos vary in capitalization)', () => {
    expect(isGlobVariantPresent(['model.q4_k_m.gguf'], ['*Q4_K_M*'])).toBe(true);
    expect(isGlobVariantPresent(['model.Q4_K_M.gguf'], ['*q4_k_m*'])).toBe(true);
  });
});
