import { mkdir, mkdtemp, readFile, readlink, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import {
  cleanupPagedOverrides,
  resolvePagedAwareModelPath,
} from '../../../packages/cli/src/commands/launch-claude/paged-config-override.js';

async function makeFixture(modelType: string, extraConfig: Record<string, unknown> = {}): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-paged-override-fixture-'));
  await writeFile(
    join(dir, 'config.json'),
    JSON.stringify({ model_type: modelType, ...extraConfig }, null, 2),
    'utf-8',
  );
  await writeFile(join(dir, 'tokenizer.json'), '{"vocab":[]}', 'utf-8');
  await writeFile(join(dir, 'model-00001-of-00001.safetensors'), 'fake', 'utf-8');
  return dir;
}

describe('resolvePagedAwareModelPath', () => {
  it('returns the original path for non-Qwen3.5 model types', async () => {
    const src = await makeFixture('qwen3');
    try {
      const result = await resolvePagedAwareModelPath(src);
      expect(result).toBe(src);
    } finally {
      await rm(src, { recursive: true, force: true });
    }
  });

  it('returns the original path when both the flag and the mem floor are already satisfied', async () => {
    const src = await makeFixture('qwen3_5', {
      use_block_paged_cache: true,
      paged_cache_memory_mb: 32768,
    });
    try {
      const result = await resolvePagedAwareModelPath(src);
      expect(result).toBe(src);
    } finally {
      await rm(src, { recursive: true, force: true });
    }
  });

  it('clones and bumps paged_cache_memory_mb when flag is set but mem is unset', async () => {
    const src = await makeFixture('qwen3_5', { use_block_paged_cache: true });
    try {
      const result = await resolvePagedAwareModelPath(src);
      expect(result).not.toBe(src);

      const cfg = JSON.parse(await readFile(join(result, 'config.json'), 'utf-8')) as Record<string, unknown>;
      expect(cfg.use_block_paged_cache).toBe(true);
      expect(typeof cfg.paged_cache_memory_mb).toBe('number');
      expect(cfg.paged_cache_memory_mb as number).toBeGreaterThanOrEqual(16384);
    } finally {
      await cleanupPagedOverrides();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('preserves a source paged_cache_memory_mb that already exceeds the floor', async () => {
    const src = await makeFixture('qwen3_5_moe', {
      use_block_paged_cache: true,
      paged_cache_memory_mb: 65536,
    });
    try {
      const result = await resolvePagedAwareModelPath(src);
      // Both knobs already satisfied → no clone, source path returned as-is.
      expect(result).toBe(src);
    } finally {
      await rm(src, { recursive: true, force: true });
    }
  });

  it('clones to an override dir for qwen3_5 with paged flag injected', async () => {
    const src = await makeFixture('qwen3_5');
    try {
      const result = await resolvePagedAwareModelPath(src);
      expect(result).not.toBe(src);

      const cfg = JSON.parse(await readFile(join(result, 'config.json'), 'utf-8')) as Record<string, unknown>;
      expect(cfg.use_block_paged_cache).toBe(true);
      expect(cfg.model_type).toBe('qwen3_5');
      expect(cfg.paged_cache_memory_mb as number).toBeGreaterThanOrEqual(16384);

      // Weights should be symlinks to the source.
      const weightLink = await readlink(join(result, 'model-00001-of-00001.safetensors'));
      expect(weightLink).toBe(join(src, 'model-00001-of-00001.safetensors'));

      // tokenizer.json too — anything that isn't config.json gets symlinked.
      const tokLink = await readlink(join(result, 'tokenizer.json'));
      expect(tokLink).toBe(join(src, 'tokenizer.json'));
    } finally {
      await cleanupPagedOverrides();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('clones to an override dir for qwen3_5_moe', async () => {
    const src = await makeFixture('qwen3_5_moe');
    try {
      const result = await resolvePagedAwareModelPath(src);
      expect(result).not.toBe(src);

      const cfg = JSON.parse(await readFile(join(result, 'config.json'), 'utf-8')) as Record<string, unknown>;
      expect(cfg.use_block_paged_cache).toBe(true);
      expect(cfg.model_type).toBe('qwen3_5_moe');
    } finally {
      await cleanupPagedOverrides();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('returns the same override dir on repeat calls (stable hash)', async () => {
    const src = await makeFixture('qwen3_5');
    try {
      const r1 = await resolvePagedAwareModelPath(src);
      const r2 = await resolvePagedAwareModelPath(src);
      expect(r1).toBe(r2);
    } finally {
      await cleanupPagedOverrides();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('returns the original path when config.json is missing', async () => {
    const src = await mkdtemp(join(tmpdir(), 'mlx-paged-override-no-config-'));
    try {
      const result = await resolvePagedAwareModelPath(src);
      expect(result).toBe(src);
    } finally {
      await rm(src, { recursive: true, force: true });
    }
  });

  it('returns the original path when config.json is malformed', async () => {
    const src = await mkdtemp(join(tmpdir(), 'mlx-paged-override-bad-config-'));
    try {
      await writeFile(join(src, 'config.json'), 'not json {', 'utf-8');
      const result = await resolvePagedAwareModelPath(src);
      expect(result).toBe(src);
    } finally {
      await rm(src, { recursive: true, force: true });
    }
  });

  it('skips subdirectories (only symlinks regular files)', async () => {
    const src = await makeFixture('qwen3_5');
    try {
      await mkdir(join(src, 'subdir'));
      await writeFile(join(src, 'subdir', 'inner.txt'), 'inner', 'utf-8');
      const result = await resolvePagedAwareModelPath(src);
      // No subdir in the override — only files at the top level.
      let sawSubdir = false;
      try {
        await readlink(join(result, 'subdir'));
        sawSubdir = true;
      } catch {
        /* expected — no symlink */
      }
      expect(sawSubdir).toBe(false);
    } finally {
      await cleanupPagedOverrides();
      await rm(src, { recursive: true, force: true });
    }
  });
});
