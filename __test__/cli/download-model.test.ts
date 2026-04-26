import { mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  isGgufRepoComplete,
  isGlobMatchedSetComplete,
  isGlobVariantPresent,
  isLocalCopyComplete,
  isModelAlreadyDownloaded,
} from '../../packages/cli/src/commands/download-model.js';

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

describe('isGgufRepoComplete', () => {
  it('returns false when only some of the remote GGUF variants are present locally', () => {
    // Regression: previously a no-glob re-run after an interrupted
    // download (e.g. only Q2_K landed) silently exited as "already
    // downloaded" because the early-return only checked
    // `files.some(.gguf)`. The fix compares against the remote
    // manifest and refuses to short-circuit until every advertised
    // GGUF variant is on disk.
    const local = ['model.Q2_K.gguf', 'config.json'];
    const remote = ['model.Q2_K.gguf', 'model.Q4_K_M.gguf', 'model.Q8_0.gguf'];
    expect(isGgufRepoComplete(local, remote)).toBe(false);
  });

  it('returns true when every remote GGUF variant is present locally', () => {
    const local = ['model.Q4_K_M.gguf', 'config.json'];
    const remote = ['model.Q4_K_M.gguf'];
    expect(isGgufRepoComplete(local, remote)).toBe(true);
  });

  it('returns false when the remote repo is not a GGUF repo (no .gguf files in manifest)', () => {
    // Caller should route through `isModelAlreadyDownloaded` for
    // safetensors / Paddle repos. A `false` return here tells the
    // caller "do not take the GGUF early-return branch".
    const local = ['model.safetensors', 'config.json'];
    const remote = ['model.safetensors', 'config.json', 'tokenizer.json'];
    expect(isGgufRepoComplete(local, remote)).toBe(false);
  });

  it('returns false on an empty remote manifest (likely upstream error)', () => {
    // An empty manifest is almost certainly a network / auth failure
    // rather than a legitimate empty repo. Returning false routes the
    // caller through the download loop where the real error will
    // surface (404 / auth) instead of being masked as "already
    // downloaded".
    expect(isGgufRepoComplete(['model.Q4_K_M.gguf'], [])).toBe(false);
    expect(isGgufRepoComplete([], [])).toBe(false);
  });

  it('compares basenames so a sub-directory remote layout still resolves cleanly', () => {
    // Some repos publish under a prefix (e.g. `models/foo.gguf`); the
    // local `readdir(outputDir)` is always flat, so the helper compares
    // basenames on both sides.
    const local = ['model.Q4_K_M.gguf'];
    const remote = ['models/model.Q4_K_M.gguf'];
    expect(isGgufRepoComplete(local, remote)).toBe(true);
  });

  it('returns false when the local file list is empty', () => {
    // Defensive: a fresh outputDir against a non-empty manifest is
    // never complete.
    expect(isGgufRepoComplete([], ['model.Q4_K_M.gguf'])).toBe(false);
  });
});

describe('isGlobMatchedSetComplete', () => {
  it('returns false when only some of the remote glob-matched files are present locally', () => {
    // Regression: previously the early-return used `isGlobVariantPresent`,
    // which only required AT LEAST ONE local hit. An interrupted prior
    // `--glob "*Q4*"` run that fetched one Q4 shard but not the others
    // would silently exit as "Matched files already downloaded" while
    // leaving the local copy incomplete.
    const remote = ['model.Q4_0.gguf', 'model.Q4_K_M.gguf', 'model.Q8_0.gguf', 'config.json'];
    const local = ['model.Q4_0.gguf', 'config.json'];
    expect(isGlobMatchedSetComplete(local, remote, ['*Q4*'])).toBe(false);
  });

  it('returns true when every remote glob-matched file is present locally', () => {
    const remote = ['model.Q4_0.gguf', 'model.Q4_K_M.gguf', 'model.Q8_0.gguf', 'config.json'];
    const local = ['model.Q4_0.gguf', 'model.Q4_K_M.gguf', 'config.json'];
    expect(isGlobMatchedSetComplete(local, remote, ['*Q4*'])).toBe(true);
  });

  it('returns false when the remote manifest has no files matching the glob', () => {
    // Empty intersection: nothing was supposed to be downloaded.
    // Declaring "complete" here would be wrong — the downstream
    // "no files matched the given criteria" path handles this case
    // after listing available variants.
    const remote = ['model.safetensors', 'config.json'];
    const local: string[] = [];
    expect(isGlobMatchedSetComplete(local, remote, ['*Q4*'])).toBe(false);
  });

  it('returns false on an empty remote manifest (likely upstream error)', () => {
    expect(isGlobMatchedSetComplete(['model.Q4_K_M.gguf'], [], ['*Q4*'])).toBe(false);
  });

  it('compares basenames so a sub-directory remote layout still resolves cleanly', () => {
    // Some repos publish under a prefix (e.g. `models/Q4_K_M.gguf`); the
    // local `readdir(outputDir)` is always flat. Mirrors `isGgufRepoComplete`.
    const remote = ['models/model.Q4_K_M.gguf'];
    const local = ['model.Q4_K_M.gguf'];
    expect(isGlobMatchedSetComplete(local, remote, ['*Q4*'])).toBe(true);
  });

  it('returns false when no glob patterns are provided', () => {
    // Defensive: the helper requires at least one pattern to compare against.
    expect(isGlobMatchedSetComplete(['model.Q4_K_M.gguf'], ['model.Q4_K_M.gguf'], [])).toBe(false);
  });
});

describe('isLocalCopyComplete', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-download-test-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('returns false when the destination file does not exist', () => {
    expect(isLocalCopyComplete(join(dir, 'missing.bin'), 100)).toBe(false);
  });

  it('returns true when the destination exists and size matches', () => {
    // Regression: previously the download loop unconditionally called
    // copyFile for every file in `filesToDownload`, re-copying gigabytes
    // of already-complete shards from the HF cache to outputDir on every
    // resume. The skip is gated on size-equality so a single Edit catches
    // truncated/interrupted prior copies.
    const path = join(dir, 'shard.bin');
    writeFileSync(path, 'x'.repeat(100));
    expect(isLocalCopyComplete(path, 100)).toBe(true);
  });

  it('returns false when the destination is truncated (interrupted prior copy)', () => {
    // A previous `copyFile` killed mid-write would leave a smaller-than-
    // expected file. The size mismatch must trigger a re-copy so the resume
    // doesn't ship a corrupt shard to disk.
    const path = join(dir, 'shard.bin');
    writeFileSync(path, 'x'.repeat(50));
    expect(isLocalCopyComplete(path, 100)).toBe(false);
  });

  it('returns false when the destination is larger than expected (corrupt write)', () => {
    const path = join(dir, 'shard.bin');
    writeFileSync(path, 'x'.repeat(150));
    expect(isLocalCopyComplete(path, 100)).toBe(false);
  });

  it('falls back to existence-only when expectedSize is non-positive', () => {
    // The HF manifest occasionally returns size=0 for tiny metadata files
    // or when the expand=true field isn't populated. Existence is the
    // best signal we can use without re-fetching the LFS pointer.
    const path = join(dir, 'meta.json');
    writeFileSync(path, '{}');
    expect(isLocalCopyComplete(path, 0)).toBe(true);
    expect(isLocalCopyComplete(path, -1)).toBe(true);
  });
});
