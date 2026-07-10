import { describe, expect, it } from 'vite-plus/test';

import { MODEL_CATALOG, visibleCatalog } from '../src/catalog.js';

describe('MODEL_CATALOG', () => {
  it('is non-empty', () => {
    expect(MODEL_CATALOG.length).toBeGreaterThan(0);
  });

  it('has exactly one default entry', () => {
    const defaults = MODEL_CATALOG.filter((entry) => entry.isDefault);
    expect(defaults).toHaveLength(1);
    expect(defaults[0]!.label).toBe('Qwen3.6-27B');
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
