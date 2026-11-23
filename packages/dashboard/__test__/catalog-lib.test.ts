import { describe, expect, it } from 'vite-plus/test';

import { MODEL_CATALOG } from '../../agent/src/catalog.js';
import { catalogBuildLabel, hasUpdate, modelBlurb, offeredOnHost } from '../ui/src/lib/catalog.js';

describe('catalogBuildLabel', () => {
  it('takes the build from the first starred glob', () => {
    expect(catalogBuildLabel({ hfRepo: 'unsloth/Qwen3.8-27B-GGUF', globs: ['*.json', '*UD-Q4_K_XL*', '*Q8_0*'] })).toBe(
      'UD-Q4_K_XL',
    );
  });

  it('takes the build from a glob that ends in .gguf', () => {
    expect(
      catalogBuildLabel({
        hfRepo: 'unsloth/Inkling-2-Mini-GGUF',
        globs: ['*UD-Q4_K_XL.gguf', 'mmproj-BF16.gguf', 'dspark-Inkling-2-Mini-Q8_0.gguf'],
      }),
    ).toBe('UD-Q4_K_XL');
  });

  it('skips globs that name a file type or are not starred on both ends', () => {
    expect(
      catalogBuildLabel({
        hfRepo: 'org/Model-GGUF',
        globs: ['*Q4.gguf*', 'Q4*', '**', '*', '*.gguf', 'Q4.gguf', '*Q4.Q8.gguf', '*Q4*.gguf', '*Q4.safetensors'],
      }),
    ).toBeNull();
  });

  it('gives every visible catalog entry a build', () => {
    for (const entry of MODEL_CATALOG.filter((item) => !item.hidden)) {
      expect(catalogBuildLabel(entry), entry.label).not.toBeNull();
    }
  });

  it('reads an mxfp / nvfp part of the repo name, upper-cased', () => {
    expect(catalogBuildLabel({ hfRepo: 'mlx-node/Qwen3.6-35B-A3B-mxfp4' })).toBe('MXFP4');
    expect(catalogBuildLabel({ hfRepo: 'mlx-node/gemma-4-26b-NVFP4-text' })).toBe('NVFP4');
    expect(catalogBuildLabel({ hfRepo: 'mlx-node/Model-mxfp8', globs: [] })).toBe('MXFP8');
  });

  it('is null when nothing names a build', () => {
    expect(catalogBuildLabel({ hfRepo: 'mlx-node/Qwen3.6-35B-A3B-4bit' })).toBeNull();
    expect(catalogBuildLabel({ hfRepo: 'mlx-node/notmxfp4' })).toBeNull();
  });
});

describe('modelBlurb', () => {
  it('keeps the two special cases, then the catalog description', () => {
    expect(modelBlurb({ label: 'Gemma-4-26B-A4B', description: 'x', isDefault: true })).toBe(
      'Fast responses, with a smaller download to get started.',
    );
    expect(modelBlurb({ label: 'Qwen3.6', description: 'x', isDefault: true })).toBe(
      'Our recommended starting point for working with tools.',
    );
    expect(modelBlurb({ label: 'Qwen3.6', description: 'Plain description.' })).toBe('Plain description.');
  });

  it('gives the catalog entries without a special case their own description', () => {
    for (const label of ['Inkling-2-Mini', 'Qwen3.8-Flash-Next', 'Hy4-flash']) {
      const entry = MODEL_CATALOG.find((item) => item.label === label)!;
      expect(modelBlurb(entry), label).toBe(entry.description);
    }
  });
});

describe('hasUpdate', () => {
  const base = {
    hfRepo: 'org/model',
    installed: true,
    localRevision: 'aaa',
    localAssetsRepo: null,
    localAssetsRevision: null,
  };

  it('compares the primary revision only for a dashboard-owned install', () => {
    expect(hasUpdate(base, new Map([['org/model', 'bbb']]))).toBe(true);
    expect(hasUpdate(base, new Map([['org/model', 'aaa']]))).toBe(false);
    expect(hasUpdate({ ...base, installed: false }, new Map([['org/model', 'bbb']]))).toBe(false);
    expect(hasUpdate({ ...base, localRevision: null }, new Map([['org/model', 'bbb']]))).toBe(false);
    expect(hasUpdate(base, new Map([['org/model', null]]))).toBe(false);
  });

  it('treats a switched or missing assets source as an update, and a moved sidecar repo too', () => {
    expect(hasUpdate({ ...base, assetsRepo: 'org/base' }, new Map())).toBe(true);
    const sidecar = { ...base, assetsRepo: 'org/base', localAssetsRepo: 'org/base', localAssetsRevision: 's1' };
    expect(hasUpdate(sidecar, new Map([['org/base', 's1']]))).toBe(false);
    expect(hasUpdate(sidecar, new Map([['org/base', 's2']]))).toBe(true);
  });
});

describe('offeredOnHost', () => {
  const GIB = 2 ** 30;
  const host = (gib: number) => ({ totalMemBytes: gib * GIB, diskFreeBytes: null });

  it('offers an entry without a floor on any host, and before the host is known', () => {
    expect(offeredOnHost({}, host(8))).toBe(true);
    expect(offeredOnHost({}, undefined)).toBe(true);
  });

  it('never offers a hidden entry', () => {
    expect(offeredOnHost({ hidden: true }, host(512))).toBe(false);
  });

  it('offers an entry with a floor only once the host meets it', () => {
    expect(offeredOnHost({ minMemoryGb: 39 }, undefined)).toBe(false);
    expect(offeredOnHost({ minMemoryGb: 39 }, { totalMemBytes: 39 * GIB - 1, diskFreeBytes: null })).toBe(false);
    expect(offeredOnHost({ minMemoryGb: 39 }, host(39))).toBe(true);
    expect(offeredOnHost({ minMemoryGb: 39 }, host(128))).toBe(true);
  });

  it('offers Qwen3.8-Flash-Next and Hy4-flash from the 96 GiB a 96 GB Mac reports', () => {
    for (const label of ['Qwen3.8-Flash-Next', 'Hy4-flash']) {
      const entry = MODEL_CATALOG.find((item) => item.label === label)!;
      expect(offeredOnHost(entry, host(64)), label).toBe(false);
      expect(offeredOnHost(entry, { totalMemBytes: 96 * GIB - 1, diskFreeBytes: null }), label).toBe(false);
      expect(offeredOnHost(entry, host(96)), label).toBe(true);
      expect(offeredOnHost(entry, host(128)), label).toBe(true);
    }
  });
});
