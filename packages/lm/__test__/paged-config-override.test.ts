/// <reference types="node" />

/**
 * `PagedConfigOverrideManager`: the paged config overlay the host loads from,
 * pinned without loading any weights.
 */

import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import * as core from '@mlx-node/core';
import { CHAT_FAMILY_IDS, PagedConfigOverrideManager } from '@mlx-node/lm';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

const cleanups: Array<() => Promise<void>> = [];

afterEach(async () => {
  vi.restoreAllMocks();
  while (cleanups.length > 0) await cleanups.pop()!();
});

async function makeModelDir(config: Record<string, unknown>): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'mlx-override-src-'));
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

describe('PagedConfigOverrideManager overlay', () => {
  it('muse overlay keeps gguf-source.json and has_dflash', async () => {
    const sourceConfig = { model_type: 'muse_glimmer', use_block_paged_cache: false };
    const prepared = await makeModelDir(sourceConfig);
    const input = join(prepared, 'Muse-Glimmer-Q4_K_M.gguf');
    const ggufSource = {
      main: input,
      companion: join(prepared, 'dflash-kquant.gguf'),
      main_identity: { source_identity: '1'.repeat(64), header_digest: 'a'.repeat(64) },
      companion_identity: { source_identity: '2'.repeat(64), header_digest: 'b'.repeat(64) },
    };
    await writeFile(join(prepared, 'gguf-source.json'), JSON.stringify(ggufSource), 'utf-8');
    await writeFile(join(prepared, 'draft.safetensors'), 'draft weights');
    const prepare = vi.spyOn(core, 'prepareMuseGlimmerGguf').mockResolvedValue(prepared);
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(input, 'muse_glimmer');
    expect(prepare).toHaveBeenCalledWith(input);
    expect(resolved).not.toBe(prepared);
    expect(await readOverrideConfig(resolved)).toMatchObject({ use_block_paged_cache: true });
    expect(JSON.parse(await readFile(join(resolved, 'gguf-source.json'), 'utf8'))).toEqual(ggufSource);
    expect(await readFile(join(resolved, 'draft.safetensors'), 'utf8')).toBe('draft weights');
    expect(await readOverrideConfig(prepared)).toEqual(sourceConfig);
    expect(await manager.resolve(input, 'muse_glimmer')).toBe(resolved);
  });

  // Each resolve of a muse_glimmer .gguf acquires one prepared-asset lease
  // reference. `claimPreparedLease` hands references out one per load attempt,
  // and `cleanup()` releases whatever was never claimed — otherwise a resolve
  // whose load never ran would keep the lazy assets protected from stale/debris
  // cleanup for the process lifetime.
  it('hands out one prepared lease per resolve attempt and releases unclaimed ones on cleanup', async () => {
    const prepared = await makeModelDir({ model_type: 'muse_glimmer', use_block_paged_cache: false });
    const input = join(prepared, 'Muse-Glimmer-Q4_K_M.gguf');
    vi.spyOn(core, 'prepareMuseGlimmerGguf').mockResolvedValue(prepared);
    const release = vi.spyOn(core, 'releaseNativeGgufLease').mockImplementation(() => {});
    const manager = new PagedConfigOverrideManager();

    const first = await manager.resolve(input, 'muse_glimmer');
    const second = await manager.resolve(input, 'muse_glimmer');
    expect(first).toBe(second); // one memoized overlay, two lease references

    // Repeated failed retries: each claim yields the prepared dir so the host
    // can release exactly that attempt's reference.
    expect(manager.claimPreparedLease(first)).toBe(prepared);
    expect(manager.claimPreparedLease(first)).toBe(prepared);
    expect(manager.claimPreparedLease(first)).toBeUndefined();

    // cleanup() releases any resolve-acquired reference no load ever claimed.
    const manager2 = new PagedConfigOverrideManager();
    const third = await manager2.resolve(input, 'muse_glimmer');
    expect(manager2.claimPreparedLease(third)).toBe(prepared);
    await manager2.resolve(input, 'muse_glimmer'); // claim left pending
    await manager2.cleanup();
    expect(release).toHaveBeenCalledWith(prepared);
  });

  it('releases the prepared lease when overlay creation fails after prepare', async () => {
    const prepared = await makeModelDir({ model_type: 'muse_glimmer', use_block_paged_cache: false });
    const input = join(prepared, 'Muse-Glimmer-Q4_K_M.gguf');
    vi.spyOn(core, 'prepareMuseGlimmerGguf').mockResolvedValue(prepared);
    const release = vi.spyOn(core, 'releaseNativeGgufLease').mockImplementation(() => {});
    // A separator inside the prefix makes the mkdtemp parent path nonexistent,
    // so createOverride deterministically throws AFTER prepare leased the dir.
    const manager = new PagedConfigOverrideManager({ tempDirPrefix: join('nonexistent-parent-dir', 'overrides-') });
    cleanups.push(() => manager.cleanup());

    await expect(manager.resolve(input, 'muse_glimmer')).rejects.toThrow();
    expect(release).toHaveBeenCalledWith(prepared);
  });

  it('does not prepare Muse GGUF assets for an unmanaged family', async () => {
    const prepare = vi.spyOn(core, 'prepareMuseGlimmerGguf');
    const manager = new PagedConfigOverrideManager({ modelTypes: ['qwen3_5'] });
    cleanups.push(() => manager.cleanup());
    expect(await manager.resolve('/models/muse.gguf', 'muse_glimmer')).toBe('/models/muse.gguf');
    expect(prepare).not.toHaveBeenCalled();
  });

  // A family missing from CHAT_FAMILY_IDS is not "unpaged by policy" —
  // `resolveInternal` returns the SOURCE path untouched, so `mlx agent` /
  // `mlx launch claude` load it with whatever `use_block_paged_cache` the
  // downloaded checkpoint carries (absent on the shipped nemotron_h build =
  // flat), silently dropping continuous batching, prefix reuse, and the cold
  // tier for that family. Mutation caught: flipping the nemotron_h row's kind
  // out of trainable|loadable makes both assertions below fail.
  it('forces the paged overlay for a nemotron_h checkpoint', async () => {
    const source = await makeModelDir({ model_type: 'nemotron_h' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const resolved = await manager.resolve(source, 'nemotron_h');
    expect(resolved).not.toBe(source);
    expect((await readOverrideConfig(resolved)).use_block_paged_cache).toBe(true);
  });

  // Pins the derived default paged policy set: every chat family, and nothing
  // else.
  it('derives the default paged policy set from the chat families', () => {
    expect([...CHAT_FAMILY_IDS].sort()).toEqual(
      [
        'qwen3',
        'qwen3_5',
        'qwen3_5_moe',
        'qwen4_exp',
        'deepseek_v41',
        'gemma4',
        'muse_glimmer',
        'lfm2',
        'lfm2_moe',
        'nemotron_h',
        'k2_horizon',
      ].sort(),
    );
  });
});

describe('PagedConfigOverrideManager pool sizing', () => {
  // A Qwen3.5 load sizes its pool from its memory plan, so the clone writes
  // no pool size: it only forces `use_block_paged_cache`. A source
  // `paged_cache_memory_mb` passes through to the plan's check.
  // Mutation: restore the 16,384 MiB floor (or the 2,048 MiB initial) in
  // `createOverride` → red.
  it.each(['qwen3_5', 'qwen3_5_moe'])('writes no pool size for %s', async (modelType) => {
    const source = await makeModelDir({ model_type: modelType });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const config = await readOverrideConfig(await manager.resolve(source, modelType));
    expect(config.use_block_paged_cache).toBe(true);
    expect('paged_cache_memory_mb' in config).toBe(false);
    expect('paged_cache_initial_memory_mb' in config).toBe(false);
  });

  // Mutation: read MLX_PAGED_CACHE_MEMORY_MB / MLX_PAGED_CACHE_INITIAL_MB
  // again → red.
  it('ignores the deleted pool-size env vars', async () => {
    const source = await makeModelDir({ model_type: 'qwen3_5' });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const previousInitial = process.env.MLX_PAGED_CACHE_INITIAL_MB;
    const previousMemory = process.env.MLX_PAGED_CACHE_MEMORY_MB;
    process.env.MLX_PAGED_CACHE_INITIAL_MB = '8192';
    process.env.MLX_PAGED_CACHE_MEMORY_MB = '4096';
    try {
      const config = await readOverrideConfig(await manager.resolve(source, 'qwen3_5'));
      expect('paged_cache_memory_mb' in config).toBe(false);
      expect('paged_cache_initial_memory_mb' in config).toBe(false);
    } finally {
      if (previousInitial === undefined) delete process.env.MLX_PAGED_CACHE_INITIAL_MB;
      else process.env.MLX_PAGED_CACHE_INITIAL_MB = previousInitial;
      if (previousMemory === undefined) delete process.env.MLX_PAGED_CACHE_MEMORY_MB;
      else process.env.MLX_PAGED_CACHE_MEMORY_MB = previousMemory;
    }
  });

  it('passes a source paged_cache_memory_mb through to the native plan check', async () => {
    const source = await makeModelDir({ model_type: 'qwen3_5_moe', paged_cache_memory_mb: 2_048 });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    const config = await readOverrideConfig(await manager.resolve(source, 'qwen3_5_moe'));
    expect(config.paged_cache_memory_mb).toBe(2_048);
  });

  // Mutation: keep a size condition in the pass-through test → red (an
  // already-paged source without a size clones).
  it('passes an already-paged qwen3_5 source through unchanged', async () => {
    const source = await makeModelDir({ model_type: 'qwen3_5', use_block_paged_cache: true });
    const manager = new PagedConfigOverrideManager();
    cleanups.push(() => manager.cleanup());

    expect(await manager.resolve(source, 'qwen3_5')).toBe(source);
  });
});
