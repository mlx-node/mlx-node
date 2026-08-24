/**
 * `PagedConfigOverrideManager` persist-paged-cache injection.
 *
 * The agent enables each allowlisted family's cold tier by asking the config overlay to
 * write `persist_paged_cache: true` into the cloned `config.json` the native
 * loader reads — never mutating the downloaded checkpoint. These tests pin that
 * injection (and its absence) without loading any weights.
 */

import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { PagedConfigOverrideManager } from '@mlx-node/lm';
import { afterEach, describe, expect, it } from 'vite-plus/test';

const cleanups: Array<() => Promise<void>> = [];

afterEach(async () => {
  while (cleanups.length > 0) await cleanups.pop()!();
});

async function makeModelDir(config: Record<string, unknown>): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'mlx-persist-src-'));
  cleanups.push(() => rm(root, { recursive: true, force: true }));
  const dir = join(root, 'model');
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, 'config.json'), JSON.stringify(config), 'utf-8');
  await writeFile(join(dir, 'weights.safetensors'), 'stub', 'utf-8');
  return dir;
}

async function readOverrideConfig(overridePath: string): Promise<Record<string, unknown>> {
  return JSON.parse(await readFile(join(overridePath, 'config.json'), 'utf-8')) as Record<string, unknown>;
}

describe('PagedConfigOverrideManager persist-paged-cache', () => {
  it('injects persist_paged_cache into a qwen3 overlay when requested', async () => {
    const source = await makeModelDir({ model_type: 'qwen3' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(source, 'qwen3', true);
    expect(resolved).not.toBe(source);

    const config = await readOverrideConfig(resolved);
    expect(config.persist_paged_cache).toBe(true);
    expect(config.use_block_paged_cache).toBe(true);
  });

  it('does not write persist_paged_cache when persistence is not requested', async () => {
    const source = await makeModelDir({ model_type: 'qwen3' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(source, 'qwen3');
    const config = await readOverrideConfig(resolved);
    expect(config.use_block_paged_cache).toBe(true);
    expect('persist_paged_cache' in config).toBe(false);
  });

  it('still injects persistence for an already-paged qwen3 checkpoint (fast-path override)', async () => {
    const source = await makeModelDir({ model_type: 'qwen3', use_block_paged_cache: true });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(source, 'qwen3', true);
    // Without the persist request this config would pass through unchanged; the
    // request must force a clone so the flag actually reaches the loader.
    expect(resolved).not.toBe(source);
    const config = await readOverrideConfig(resolved);
    expect(config.persist_paged_cache).toBe(true);
  });

  // Finding 9: an explicit `false` directive is AUTHORITATIVE — it must override
  // a checkpoint whose config.json hard-codes persistence on, so
  // `mlx agent --no-persist-cache` truly wins.
  it('writes an authoritative persist_paged_cache: false over a persist-enabled checkpoint', async () => {
    const source = await makeModelDir({ model_type: 'qwen3', persist_paged_cache: true });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(source, 'qwen3', false);
    // A clone is forced so the false actually reaches the loader.
    expect(resolved).not.toBe(source);
    const config = await readOverrideConfig(resolved);
    expect(config.persist_paged_cache).toBe(false);
  });

  it('reconciles a stray camelCase persistPagedCache alias to the authoritative value', async () => {
    const source = await makeModelDir({ model_type: 'qwen3', persistPagedCache: true });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(source, 'qwen3', false);
    expect(resolved).not.toBe(source);
    const config = await readOverrideConfig(resolved);
    // Both spellings agree on the authoritative false the loader will read.
    expect(config.persist_paged_cache).toBe(false);
    expect(config.persistPagedCache).toBe(false);
  });

  // Finding 12: snake-first precedence. A config with persist_paged_cache:false
  // AND persistPagedCache:true reads as the authoritative snake=false (native
  // reads snake first), so a requested `true` DISAGREES and must force a clone
  // that writes snake=true. The old OR misread this as already-true and passed
  // an already-paged checkpoint straight through, silently dropping persistence.
  it('forces a clone writing persist_paged_cache:true when snake=false conflicts with camel=true', async () => {
    const source = await makeModelDir({
      model_type: 'qwen3',
      use_block_paged_cache: true, // already paged: only the persist override can force a clone
      persist_paged_cache: false,
      persistPagedCache: true,
    });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(source, 'qwen3', true);
    expect(resolved).not.toBe(source);
    const config = await readOverrideConfig(resolved);
    expect(config.persist_paged_cache).toBe(true);
    expect(config.persistPagedCache).toBe(true);
  });

  // Finding D: the memo key folds in the persist tri-state, so resolving the SAME
  // source with persist=true then persist=false yields two DISTINCT overrides —
  // the second must not return the first (persist=true) clone cached under the
  // bare path. Same directive still memoizes.
  it('resolves distinct overrides for the same source under different persist directives', async () => {
    const source = await makeModelDir({ model_type: 'qwen3' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const enabled = await manager.resolve(source, 'qwen3', true);
    const disabled = await manager.resolve(source, 'qwen3', false);
    expect(disabled).not.toBe(enabled);

    const enabledConfig = await readOverrideConfig(enabled);
    const disabledConfig = await readOverrideConfig(disabled);
    expect(enabledConfig.persist_paged_cache).toBe(true);
    expect(disabledConfig.persist_paged_cache).toBe(false);

    // Re-resolving with the same directive returns the memoized override.
    expect(await manager.resolve(source, 'qwen3', true)).toBe(enabled);
  });

  // A family missing from AGENT_PAGED_MODEL_TYPES is not "unpaged by policy" —
  // `resolveInternal` returns the SOURCE path untouched, so `mlx agent` /
  // `mlx launch claude` load it with whatever `use_block_paged_cache` the
  // downloaded checkpoint carries (absent on the shipped nemotron_h build =
  // flat), silently dropping continuous batching, prefix reuse, and the cold
  // tier for that family. Mutation caught: deleting `'nemotron_h'` from
  // AGENT_PAGED_MODEL_TYPES makes both assertions below fail.
  it('forces the paged overlay for a nemotron_h checkpoint', async () => {
    const source = await makeModelDir({ model_type: 'nemotron_h' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(source, 'nemotron_h');
    expect(resolved).not.toBe(source);
    expect((await readOverrideConfig(resolved)).use_block_paged_cache).toBe(true);
  });

  it('leaves persist untouched for an undirected (undefined) resolve', async () => {
    const source = await makeModelDir({ model_type: 'qwen3' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(source, 'qwen3');
    const config = await readOverrideConfig(resolved);
    expect('persist_paged_cache' in config).toBe(false);
    expect('persistPagedCache' in config).toBe(false);
  });
});

describe('PagedConfigOverrideManager initial pool sizing', () => {
  // The qwen3_5/qwen3_5_moe families get an authoritative start-small initial
  // budget: 2048 MiB by default, `MLX_PAGED_CACHE_INITIAL_MB` env wins, and
  // the value is clamped to the resolved max (`paged_cache_memory_mb` floor).
  // Other families are untouched — the native loader has no initial knob for
  // them and the clone must not invent a field it would ignore or reject.
  it('writes the 2048 default initial next to the unchanged max floor for qwen3_5', async () => {
    const source = await makeModelDir({ model_type: 'qwen3_5' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const config = await readOverrideConfig(await manager.resolve(source, 'qwen3_5'));
    expect(config.paged_cache_initial_memory_mb).toBe(2_048);
    // The max floor logic is unchanged: an unset max still becomes 16 GiB.
    expect(config.paged_cache_memory_mb).toBe(16_384);
  });

  it('does not touch a source initial already at the resolved default', async () => {
    const source = await makeModelDir({
      model_type: 'qwen3_5',
      use_block_paged_cache: true,
      paged_cache_memory_mb: 32_768,
      paged_cache_initial_memory_mb: 2_048,
    });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    expect(await manager.resolve(source, 'qwen3_5')).toBe(source);
  });

  it('blocks the pass-through when the initial knob is unsatisfied', async () => {
    // Already paged, max above the floor, but NO initial field: the source
    // must clone so the new default actually reaches the loader instead of
    // silently loading as a static full-size pool.
    const source = await makeModelDir({
      model_type: 'qwen3_5',
      use_block_paged_cache: true,
      paged_cache_memory_mb: 32_768,
    });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(source, 'qwen3_5');
    expect(resolved).not.toBe(source);
    expect((await readOverrideConfig(resolved)).paged_cache_initial_memory_mb).toBe(2_048);
  });

  it('honors MLX_PAGED_CACHE_INITIAL_MB over the 2048 default', async () => {
    const source = await makeModelDir({ model_type: 'qwen3_5_moe' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const previous = process.env.MLX_PAGED_CACHE_INITIAL_MB;
    process.env.MLX_PAGED_CACHE_INITIAL_MB = '8192';
    try {
      const config = await readOverrideConfig(await manager.resolve(source, 'qwen3_5_moe'));
      expect(config.paged_cache_initial_memory_mb).toBe(8_192);
    } finally {
      if (previous === undefined) delete process.env.MLX_PAGED_CACHE_INITIAL_MB;
      else process.env.MLX_PAGED_CACHE_INITIAL_MB = previous;
    }
  });

  it('falls back to the 2048 default for an unparseable MLX_PAGED_CACHE_INITIAL_MB', async () => {
    const source = await makeModelDir({ model_type: 'qwen3_5' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const previous = process.env.MLX_PAGED_CACHE_INITIAL_MB;
    process.env.MLX_PAGED_CACHE_INITIAL_MB = 'not-a-number';
    try {
      const config = await readOverrideConfig(await manager.resolve(source, 'qwen3_5'));
      expect(config.paged_cache_initial_memory_mb).toBe(2_048);
    } finally {
      if (previous === undefined) delete process.env.MLX_PAGED_CACHE_INITIAL_MB;
      else process.env.MLX_PAGED_CACHE_INITIAL_MB = previous;
    }
  });

  it('clamps the initial budget to the resolved max', async () => {
    const source = await makeModelDir({ model_type: 'qwen3_5' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const previousInitial = process.env.MLX_PAGED_CACHE_INITIAL_MB;
    const previousMemory = process.env.MLX_PAGED_CACHE_MEMORY_MB;
    process.env.MLX_PAGED_CACHE_INITIAL_MB = '8192';
    process.env.MLX_PAGED_CACHE_MEMORY_MB = '4096';
    try {
      const config = await readOverrideConfig(await manager.resolve(source, 'qwen3_5'));
      expect(config.paged_cache_memory_mb).toBe(4_096);
      expect(config.paged_cache_initial_memory_mb).toBe(4_096);
    } finally {
      if (previousInitial === undefined) delete process.env.MLX_PAGED_CACHE_INITIAL_MB;
      else process.env.MLX_PAGED_CACHE_INITIAL_MB = previousInitial;
      if (previousMemory === undefined) delete process.env.MLX_PAGED_CACHE_MEMORY_MB;
      else process.env.MLX_PAGED_CACHE_MEMORY_MB = previousMemory;
    }
  });

  it('leaves non-qwen3_5 families without an initial field', async () => {
    for (const modelType of ['qwen3', 'lfm2', 'gemma4_unified', 'nemotron_h'] as const) {
      const source = await makeModelDir({ model_type: modelType });
      const manager = new PagedConfigOverrideManager();
      cleanups.push(() => manager.cleanup());

      const canonical = modelType === 'gemma4_unified' ? 'gemma4' : modelType;
      const config = await readOverrideConfig(await manager.resolve(source, canonical));
      expect('paged_cache_initial_memory_mb' in config).toBe(false);
    }
  });
});
