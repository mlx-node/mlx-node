import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { discoverModels } from '../../../packages/server/src/host/discover.js';

describe('discoverModels', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-discover-test-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function makeModel(name: string, config: Record<string, unknown>): void {
    const dir = join(root, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'config.json'), JSON.stringify(config));
  }

  it('returns [] when the directory does not exist', async () => {
    const got = await discoverModels(join(root, 'missing'));
    expect(got).toEqual([]);
  });

  it('returns [] when the directory is empty', async () => {
    const got = await discoverModels(root);
    expect(got).toEqual([]);
  });

  it('discovers a Qwen3.5 directory with modelType and preset', async () => {
    makeModel('qwen3.5-demo', { model_type: 'qwen3_5' });
    const got = await discoverModels(root);
    expect(got).toHaveLength(1);
    expect(got[0].name).toBe('qwen3.5-demo');
    expect(got[0].modelType).toBe('qwen3_5');
    expect(got[0].preset.maxOutputTokens).toBeGreaterThan(0);
    expect(got[0].path.endsWith('qwen3.5-demo')).toBe(true);
  });

  it('filters out Harrier (non-generative) model entries', async () => {
    // NOTE: two guards can produce this result — the explicit NON_GENERATIVE
    // set and the "no LAUNCH_PRESETS entry" skip. Today `harrier` has no
    // preset either, so this assertion cannot tell them apart; deleting the
    // NON_GENERATIVE check leaves it green. Treat it as coverage of the
    // OUTCOME, not of which guard fired.
    makeModel('harrier-emb', { model_type: 'qwen3', architectures: ['Qwen3Model'] });
    const got = await discoverModels(root);
    expect(got).toEqual([]);
  });

  it('skips regular files in the directory', async () => {
    writeFileSync(join(root, 'stray.txt'), 'not a model');
    makeModel('qwen3-demo', { model_type: 'qwen3' });
    const got = await discoverModels(root);
    expect(got).toHaveLength(1);
    expect(got[0].name).toBe('qwen3-demo');
  });

  it('sorts discovered entries by name ascending', async () => {
    // CAVEAT: on macOS/APFS `readdir` already hands back byte-sorted names, so
    // this assertion cannot distinguish "we sorted" from "the filesystem did".
    // It is a real gate on hash-ordered filesystems (ext4 in CI). The
    // comparator's semantics are pinned platform-independently below.
    makeModel('zzz-model', { model_type: 'qwen3_5' });
    makeModel('aaa-model', { model_type: 'qwen3_5' });
    makeModel('mmm-model', { model_type: 'qwen3_5' });
    const got = await discoverModels(root);
    expect(got.map((e) => e.name)).toEqual(['aaa-model', 'mmm-model', 'zzz-model']);
  });

  it('orders by byte value, not case-insensitively', async () => {
    // `boundModel` defaults to `discovered[0]`, so the tie-break rule decides
    // which model a bare `mlx serve` binds. Byte order puts every capitalised
    // name ahead of every lowercase one; a case-folding comparator would flip
    // this pair and silently change the default on a user's disk.
    makeModel('alpha-model', { model_type: 'qwen3_5' });
    makeModel('Zeta-model', { model_type: 'qwen3_5' });
    const got = await discoverModels(root);
    expect(got.map((e) => e.name)).toEqual(['Zeta-model', 'alpha-model']);
  });

  it('skips subdirectories without config.json', async () => {
    mkdirSync(join(root, 'empty-dir'), { recursive: true });
    makeModel('qwen3-demo', { model_type: 'qwen3' });
    const got = await discoverModels(root);
    expect(got).toHaveLength(1);
  });
});
