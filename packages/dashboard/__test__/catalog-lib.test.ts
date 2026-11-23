import { describe, expect, it } from 'vite-plus/test';

import { MODEL_CATALOG } from '../../agent/src/catalog.js';
import {
  catalogBuildAt,
  hasUpdate,
  modelBlurb,
  modelPresent,
  offeredOnHost,
  onboardingBuild,
  shownBuild,
} from '../ui/src/lib/catalog.js';
import type { CatalogBuildItem, CatalogItem } from '../ui/src/lib/types.js';

describe('modelBlurb', () => {
  it('keeps the default case, then the catalog description', () => {
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

describe('catalogBuildAt', () => {
  function build(id: string, slug: string): CatalogBuildItem {
    return {
      id,
      hfRepo: 'org/model-GGUF',
      sizeGb: 1,
      source: 'GGUF',
      slug,
      installed: false,
      present: false,
      heldBy: null,
      blockedByForeignDir: false,
      blocker: null,
      localRevision: null,
      localAssetsRepo: null,
      localAssetsRevision: null,
    };
  }
  const items: CatalogItem[] = [
    { label: 'A', description: '', builds: [build('UD-Q4_K_XL', 'a-gguf')] },
    { label: 'B', description: '', builds: [build('UD-Q4_K_XL', 'b-gguf'), build('UD-Q4_K_M', 'b-gguf-ud-q4_k_m')] },
  ];

  it('finds a later build of an entry by its folder, and nothing for a folder no build installs to', () => {
    expect(catalogBuildAt(items, 'b-gguf-ud-q4_k_m')?.id).toBe('UD-Q4_K_M');
    expect(catalogBuildAt(items, 'a-gguf')?.id).toBe('UD-Q4_K_XL');
    expect(catalogBuildAt(items, 'b')).toBeUndefined();
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

describe('the build a card or a choice stands for', () => {
  function build(id: string, slug: string, present = false): CatalogBuildItem {
    return {
      id,
      hfRepo: 'org/model-GGUF',
      sizeGb: 1,
      source: 'GGUF',
      slug,
      installed: present,
      present,
      heldBy: null,
      blockedByForeignDir: false,
      blocker: null,
      localRevision: null,
      localAssetsRepo: null,
      localAssetsRevision: null,
    };
  }
  const entry = (builds: CatalogBuildItem[]): CatalogItem => ({ label: 'M', description: '', builds });
  const plain = entry([build('XL', 'm'), build('K_M', 'm-k_m'), build('Original', 'orig')]);
  const secondOnDisk = entry([build('XL', 'm'), build('K_M', 'm-k_m', true), build('Original', 'orig', true)]);

  it('counts a model when any build is on disk', () => {
    expect(modelPresent(plain)).toBe(false);
    expect(modelPresent(secondOnDisk)).toBe(true);
  });

  it('Onboarding: the build with a job, else the first on disk, else the first', () => {
    expect(onboardingBuild(secondOnDisk, ['orig']).slug).toBe('orig');
    // Mutation: skip the on-disk step → `m` → red.
    expect(onboardingBuild(secondOnDisk, []).slug).toBe('m-k_m');
    expect(onboardingBuild(plain, ['elsewhere']).slug).toBe('m');
  });

  it('a card: the chosen build first, then the Onboarding rule', () => {
    expect(shownBuild(secondOnDisk, { M: 'm' }, { orig: {} }).slug).toBe('m');
    expect(shownBuild(secondOnDisk, {}, { orig: {} }).slug).toBe('orig');
    expect(shownBuild(secondOnDisk, {}, {}).slug).toBe('m-k_m');
    expect(shownBuild(plain, { Other: 'orig' }, {}).slug).toBe('m');
    // A chosen folder the entry no longer has falls back to the rule.
    expect(shownBuild(plain, { M: 'gone' }, {}).slug).toBe('m');
  });
});
