import { mkdir, mkdtemp, readFile, readlink, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { AGENT_PAGED_MODEL_TYPES, PagedConfigOverrideManager, QWEN35_PAGED_MODEL_TYPES } from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

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

function launchClaudeManager(): PagedConfigOverrideManager {
  return new PagedConfigOverrideManager({
    modelTypes: QWEN35_PAGED_MODEL_TYPES,
  });
}

async function readConfig(path: string): Promise<Record<string, unknown>> {
  return JSON.parse(await readFile(join(path, 'config.json'), 'utf-8')) as Record<string, unknown>;
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

describe('PagedConfigOverrideManager launch-claude policy', () => {
  it('returns the original path for model types outside the configured policy', async () => {
    const src = await makeFixture('qwen3');
    const manager = launchClaudeManager();
    try {
      expect(await manager.resolve(src)).toBe(src);
    } finally {
      await manager.cleanup();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('returns the original path when the paged flag and Qwen memory floor are satisfied', async () => {
    const src = await makeFixture('qwen3_5', {
      use_block_paged_cache: true,
      paged_cache_memory_mb: 32_768,
    });
    const manager = launchClaudeManager();
    try {
      expect(await manager.resolve(src)).toBe(src);
    } finally {
      await manager.cleanup();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('clones and applies the default Qwen memory floor when memory is unset', async () => {
    const src = await makeFixture('qwen3_5', { use_block_paged_cache: true });
    const manager = launchClaudeManager();
    try {
      const result = await manager.resolve(src);
      expect(result).not.toBe(src);
      const config = await readConfig(result);
      expect(config.use_block_paged_cache).toBe(true);
      expect(config.paged_cache_memory_mb).toBeGreaterThanOrEqual(16_384);
    } finally {
      await manager.cleanup();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('honors MLX_PAGED_CACHE_MEMORY_MB for Qwen3.5 overrides', async () => {
    const src = await makeFixture('qwen3_5');
    const manager = launchClaudeManager();
    const previous = process.env.MLX_PAGED_CACHE_MEMORY_MB;
    process.env.MLX_PAGED_CACHE_MEMORY_MB = '24576';
    try {
      const config = await readConfig(await manager.resolve(src));
      expect(config.paged_cache_memory_mb).toBe(24_576);
    } finally {
      if (previous === undefined) delete process.env.MLX_PAGED_CACHE_MEMORY_MB;
      else process.env.MLX_PAGED_CACHE_MEMORY_MB = previous;
      await manager.cleanup();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('preserves source files while patching config and symlinking model files', async () => {
    const src = await makeFixture('qwen3_5', { custom_marker: 'source' });
    const originalConfig = await readFile(join(src, 'config.json'), 'utf-8');
    const manager = launchClaudeManager();
    try {
      const result = await manager.resolve(src);
      expect(result).not.toBe(src);
      expect((await readConfig(result)).custom_marker).toBe('source');
      expect(await readFile(join(src, 'config.json'), 'utf-8')).toBe(originalConfig);
      expect(await readlink(join(result, 'model-00001-of-00001.safetensors'))).toBe(
        join(src, 'model-00001-of-00001.safetensors'),
      );
      expect(await readlink(join(result, 'tokenizer.json'))).toBe(join(src, 'tokenizer.json'));
    } finally {
      await manager.cleanup();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('returns one stable override for repeated and concurrent resolution', async () => {
    const src = await makeFixture('qwen3_5_moe');
    const manager = launchClaudeManager();
    try {
      const [first, second, third] = await Promise.all([
        manager.resolve(src),
        manager.resolve(src),
        manager.resolve(src),
      ]);
      expect(second).toBe(first);
      expect(third).toBe(first);
    } finally {
      await manager.cleanup();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('passes through missing or malformed config files', async () => {
    const missing = await mkdtemp(join(tmpdir(), 'mlx-paged-override-no-config-'));
    const malformed = await mkdtemp(join(tmpdir(), 'mlx-paged-override-bad-config-'));
    await writeFile(join(malformed, 'config.json'), 'not json {', 'utf-8');
    const manager = launchClaudeManager();
    try {
      expect(await manager.resolve(missing)).toBe(missing);
      expect(await manager.resolve(malformed)).toBe(malformed);
    } finally {
      await manager.cleanup();
      await rm(missing, { recursive: true, force: true });
      await rm(malformed, { recursive: true, force: true });
    }
  });

  it('skips source subdirectories', async () => {
    const src = await makeFixture('qwen3_5');
    await mkdir(join(src, 'subdir'));
    await writeFile(join(src, 'subdir', 'inner.txt'), 'inner', 'utf-8');
    const manager = launchClaudeManager();
    try {
      const result = await manager.resolve(src);
      expect(await pathExists(join(result, 'subdir'))).toBe(false);
    } finally {
      await manager.cleanup();
      await rm(src, { recursive: true, force: true });
    }
  });
});

describe('PagedConfigOverrideManager agent policy and lifecycle', () => {
  it('forces paged config for every agent chat family', async () => {
    for (const modelType of AGENT_PAGED_MODEL_TYPES) {
      const src = await makeFixture(modelType, {
        use_block_paged_cache: false,
      });
      const manager = new PagedConfigOverrideManager();
      try {
        const result = await manager.resolve(src);
        expect(result).not.toBe(src);
        expect((await readConfig(result)).use_block_paged_cache).toBe(true);
      } finally {
        await manager.cleanup();
        await rm(src, { recursive: true, force: true });
      }
    }
  });

  it('prefers the canonical discovered family over a nested raw model type', async () => {
    const src = await makeFixture('gemma4_unified', {
      use_block_paged_cache: false,
    });
    const manager = new PagedConfigOverrideManager();
    try {
      const result = await manager.resolve(src, 'gemma4');
      expect(result).not.toBe(src);
      const config = await readConfig(result);
      expect(config.model_type).toBe('gemma4_unified');
      expect(config.use_block_paged_cache).toBe(true);
    } finally {
      await manager.cleanup();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('forces quantized LFM2 paged without applying the Qwen memory floor', async () => {
    const src = await makeFixture('lfm2', {
      use_block_paged_cache: false,
      paged_cache_memory_mb: 512,
      quantization: { mode: 'mxfp8' },
    });
    const manager = new PagedConfigOverrideManager();
    try {
      const config = await readConfig(await manager.resolve(src));
      expect(config.use_block_paged_cache).toBe(true);
      expect(config.paged_cache_memory_mb).toBe(512);
      expect(config.quantization).toEqual({ mode: 'mxfp8' });
    } finally {
      await manager.cleanup();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('isolates manager roots and cleanup', async () => {
    const src = await makeFixture('qwen3_5');
    const firstManager = new PagedConfigOverrideManager();
    const secondManager = new PagedConfigOverrideManager();
    try {
      const first = await firstManager.resolve(src);
      const second = await secondManager.resolve(src);
      expect(dirname(first)).not.toBe(dirname(second));

      await firstManager.cleanup();
      expect(await pathExists(first)).toBe(false);
      expect(await pathExists(second)).toBe(true);

      await secondManager.cleanup();
      expect(await pathExists(second)).toBe(false);
    } finally {
      await firstManager.cleanup();
      await secondManager.cleanup();
      await rm(src, { recursive: true, force: true });
    }
  });

  it('rejects reuse after cleanup', async () => {
    const src = await makeFixture('qwen3_5');
    const manager = new PagedConfigOverrideManager();
    try {
      await manager.cleanup();
      await expect(manager.resolve(src)).rejects.toThrow(/after cleanup/);
    } finally {
      await rm(src, { recursive: true, force: true });
    }
  });
});
