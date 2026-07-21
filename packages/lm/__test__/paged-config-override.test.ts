/**
 * `PagedConfigOverrideManager` persist-paged-cache injection.
 *
 * The agent enables the qwen3 dense cold tier by asking the config overlay to
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
