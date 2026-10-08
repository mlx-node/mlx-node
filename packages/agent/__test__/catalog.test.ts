import { describe, expect, it } from 'vite-plus/test';

import {
  type CatalogEntry,
  catalogEntryForRepo,
  catalogRepo,
  catalogRepoFor,
  MODEL_CATALOG,
  visibleCatalog,
} from '../src/catalog.js';

describe('catalogUpdateRepos — update discovery covers the sidecar sources', () => {
  it('adds every visible entry assetsRepo to the download repo set', async () => {
    const { catalogDownloadRepos, catalogUpdateRepos, MODEL_CATALOG } = await import('../src/catalog.js');
    const downloads = catalogDownloadRepos();
    const updates = catalogUpdateRepos();
    // Superset, deduped.
    for (const repo of downloads) expect(updates).toContain(repo);
    expect(new Set(updates).size).toBe(updates.length);
    // The assets sources are the point: a sidecar-only upstream change moves
    // nothing in the primary repo, so without these it raises no badge and the
    // repair job is unreachable.
    const assets = MODEL_CATALOG.filter((entry) => !entry.hidden && entry.assetsRepo !== undefined).map(
      (entry) => entry.assetsRepo!,
    );
    expect(assets.length).toBeGreaterThan(0);
    for (const repo of assets) expect(updates).toContain(repo);
    // Hidden entries stay out of both (their repos 401).
    for (const entry of MODEL_CATALOG.filter((item) => item.hidden)) {
      expect(updates).not.toContain(entry.hfRepo);
      if (entry.assetsRepo !== undefined) expect(updates).not.toContain(entry.assetsRepo);
    }
  });
});

describe('MODEL_CATALOG', () => {
  it('is non-empty', () => {
    expect(MODEL_CATALOG.length).toBeGreaterThan(0);
  });

  it('has exactly one default entry', () => {
    const defaults = MODEL_CATALOG.filter((entry) => entry.isDefault);
    expect(defaults).toHaveLength(1);
    expect(defaults[0]!.label).toBe('Qwen3.8-27B');
  });

  it('resolves one repo per entry, with the CUDA build only when an entry carries one', () => {
    // The whole platform split rides on this. Getting it wrong is silent: the
    // wrong-platform repo is a real, downloadable checkpoint, so nothing errors
    // — the user simply installs the build this catalog exists to steer them
    // away from. It also silently breaks every provenance test that writes a
    // marker naming `hfRepo` instead of the resolved repo.
    //
    // Via the pure helper, never by mutating `process.platform`: that global is
    // shared with every test file on this worker, and stubbing it here made a
    // sibling suite's download allowlist reject its own module-level repo.
    const withCuda: CatalogEntry = { label: 't', hfRepo: 'a/metal', hfRepoCuda: 'a/cuda', sizeGb: 1, description: 't' };
    expect(catalogRepoFor(withCuda, 'linux')).toBe('a/cuda');
    expect(catalogRepoFor(withCuda, 'darwin')).toBe('a/metal');
    // Every CURRENT entry is single-repo (the UD-Q4_K_XL GGUF build runs
    // identically on Metal and CUDA), so both platforms install `hfRepo`.
    for (const entry of MODEL_CATALOG) {
      expect(entry.hfRepoCuda, entry.label).toBeUndefined();
      expect(catalogRepoFor(entry, 'linux'), entry.label).toBe(entry.hfRepo);
      expect(catalogRepoFor(entry, 'darwin'), entry.label).toBe(entry.hfRepo);
      // And the live resolver agrees with the helper on THIS platform.
      expect(catalogRepo(entry), entry.label).toBe(catalogRepoFor(entry, process.platform));
    }
  });

  it('pins the three visible entries to their verified unsloth GGUF repos + sidecar sources', () => {
    // Load-bearing strings: a typo here is a 404 at best and the wrong weights
    // at worst. Verified on Hugging Face — the GGUF repos ship the UD-Q4_K_XL
    // variant (plus MTP/mmproj where globbed in), the assetsRepos ship the
    // tokenizer/config files the GGUF repos lack (ungated, unlike google/*).
    expect(visibleCatalog().map((entry) => [entry.hfRepo, entry.assetsRepo])).toEqual([
      ['unsloth/Qwen3.8-27B-GGUF', 'Qwen/Qwen3.8-27B'],
      ['unsloth/Qwen-AgentWorld-35B-A3B-GGUF', 'Qwen/Qwen-AgentWorld-35B-A3B'],
      ['unsloth/gemma-4-26B-A4B-it-GGUF', 'unsloth/gemma-4-26B-A4B-it'],
    ]);
  });

  it('every hfRepo is a plausible HF slug', () => {
    for (const entry of MODEL_CATALOG) {
      expect(entry.hfRepo, entry.label).toMatch(/^[A-Za-z0-9_-]+\/[A-Za-z0-9._-]+$/);
    }
  });

  it('every visible entry carries a glob filter that selects exactly one weight variant', () => {
    // Without the filter the downloader takes every .gguf in a multi-variant
    // unsloth repo — 100+ GB of quants the user never asked for.
    for (const entry of visibleCatalog()) {
      expect(entry.globs, entry.label).toBeDefined();
      expect(entry.globs!.length, entry.label).toBeGreaterThan(0);
      expect(
        entry.globs!.some((glob) => glob.includes('UD-Q4_K_XL')),
        entry.label,
      ).toBe(true);
    }
  });

  it('every visible entry names an assetsRepo for tokenizer sidecars', () => {
    // The GGUF repos ship no tokenizer files; without the official sidecars
    // the native runtime extracts the embedded tokenizer and tool calling
    // silently breaks (see the assetsRepo field comment).
    for (const entry of visibleCatalog()) {
      expect(entry.assetsRepo, entry.label).toMatch(/^[A-Za-z0-9_-]+\/[A-Za-z0-9._-]+$/);
    }
  });

  it('Qwen3.8-27B keeps its optional DFlash2 companion beside the MTP weights', () => {
    // MTP is inline in the UD-Q4_K_XL file itself (auto-detected at load);
    // the repo's separate MTP/*.gguf is deliberately not downloaded
    // and covers the default path; the DFlash2 draft stays offered as an
    // optional, never-auto-installed companion for checkpoints that pair
    // with it, so the dashboard companion card and its download allowlist
    // entry survive.
    const qwen = MODEL_CATALOG.find((entry) => entry.label === 'Qwen3.8-27B');
    expect(qwen?.draft?.hfRepo).toBe('z-lab/Qwen3.8-27B-DFlash2');
    for (const entry of MODEL_CATALOG) {
      if (entry.label === 'Qwen3.8-27B') continue;
      expect(entry.draft, entry.label).toBeUndefined();
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

describe('catalogEntryForRepo', () => {
  it('finds an entry by its resolved repo on any platform', () => {
    // Current entries are single-repo, so the platform argument cannot change
    // the answer; it exists for the day a platform-split entry returns.
    const entry = MODEL_CATALOG.find((candidate) => candidate.isDefault)!;
    expect(catalogEntryForRepo(entry.hfRepo, 'darwin')).toBe(entry);
    expect(catalogEntryForRepo(entry.hfRepo, 'linux')).toBe(entry);
  });

  it('returns undefined for a repo the catalog does not carry', () => {
    expect(catalogEntryForRepo('someone/not-in-catalog', process.platform)).toBeUndefined();
  });

  it('matches through catalogRepoFor, never a raw hfRepo comparison', () => {
    // The matching rule matters the day a platform-split entry returns: a
    // lookup must resolve against the repo THIS platform installs. Pinned
    // with the pure helper since no current entry carries the split.
    const split: CatalogEntry = { label: 't', hfRepo: 'a/metal', hfRepoCuda: 'a/cuda', sizeGb: 1, description: 't' };
    expect(catalogRepoFor(split, 'linux')).toBe('a/cuda');
    expect(catalogRepoFor(split, 'darwin')).toBe('a/metal');
    for (const entry of MODEL_CATALOG) {
      expect(catalogEntryForRepo(catalogRepoFor(entry, 'linux'), 'linux'), entry.label).toBe(entry);
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
