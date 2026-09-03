import { describe, expect, it } from 'vite-plus/test';

import { catalogRepo, catalogRepoFor, MODEL_CATALOG, visibleCatalog } from '../src/catalog.js';

describe('MODEL_CATALOG', () => {
  it('is non-empty', () => {
    expect(MODEL_CATALOG.length).toBeGreaterThan(0);
  });

  it('has exactly one default entry', () => {
    const defaults = MODEL_CATALOG.filter((entry) => entry.isDefault);
    expect(defaults).toHaveLength(1);
    expect(defaults[0]!.label).toBe('Qwen3.8-27B');
  });

  it('resolves the CUDA build on linux and the Metal build everywhere else', () => {
    // The whole platform split rides on this. Getting it wrong is silent: the
    // wrong-platform repo is a real, downloadable checkpoint, so nothing errors
    // — the user simply installs the build this catalog exists to steer them
    // away from. It also silently breaks every provenance test that writes a
    // marker naming `hfRepo` instead of the resolved repo.
    const withCuda = MODEL_CATALOG.find((entry) => entry.hfRepoCuda !== undefined);
    expect(withCuda, 'at least one entry must carry a CUDA build').toBeDefined();
    // Via the pure helper, never by mutating `process.platform`: that global is
    // shared with every test file on this worker, and stubbing it here made a
    // sibling suite's download allowlist reject its own module-level repo.
    expect(catalogRepoFor(withCuda!, 'linux')).toBe(withCuda!.hfRepoCuda);
    expect(catalogRepoFor(withCuda!, 'darwin')).toBe(withCuda!.hfRepo);
    // An entry with no CUDA build serves both platforms from `hfRepo`.
    const noCuda = MODEL_CATALOG.find((entry) => entry.hfRepoCuda === undefined);
    if (noCuda !== undefined) {
      expect(catalogRepoFor(noCuda, 'linux')).toBe(noCuda.hfRepo);
    }
    // And the live resolver agrees with the helper on THIS platform.
    expect(catalogRepo(withCuda!)).toBe(catalogRepoFor(withCuda!, process.platform));
  });

  it('every hfRepo is a Brooooooklyn HF slug', () => {
    for (const entry of MODEL_CATALOG) {
      expect(entry.hfRepo, entry.label).toMatch(/^Brooooooklyn\/[A-Za-z0-9._-]+$/);
    }
  });

  it('labels are unique', () => {
    const labels = MODEL_CATALOG.map((entry) => entry.label);
    expect(new Set(labels).size).toBe(labels.length);
  });

  it('every size is positive and every description is non-empty', () => {
    for (const entry of MODEL_CATALOG) {
      expect(entry.sizeGb, entry.label).toBeGreaterThan(0);
      expect(entry.description.trim().length, entry.label).toBeGreaterThan(0);
    }
  });
});

describe('visibleCatalog', () => {
  it('excludes hidden entries', () => {
    const visible = visibleCatalog();
    expect(visible.length).toBeGreaterThan(0);
    expect(visible.length).toBeLessThan(MODEL_CATALOG.length);
    for (const entry of visible) {
      expect(entry.hidden, entry.label).not.toBe(true);
    }
  });

  it('still contains the default entry', () => {
    const visible = visibleCatalog();
    expect(visible.some((entry) => entry.isDefault)).toBe(true);
  });

  it('is a subset of MODEL_CATALOG', () => {
    const all = new Set(MODEL_CATALOG);
    for (const entry of visibleCatalog()) {
      expect(all.has(entry), entry.label).toBe(true);
    }
  });
});
