import { totalmem } from 'node:os';

import { describe, expect, it } from 'vite-plus/test';

import {
  type CatalogEntry,
  catalogDownloadRepos,
  catalogEntryForRepo,
  catalogFitsMemory,
  catalogRepo,
  catalogRepoFor,
  catalogUpdateRepos,
  MODEL_CATALOG,
  visibleCatalog,
  visibleSystemOneCatalog,
} from '../src/catalog.js';

const GIB = 2 ** 30;
/** A 128 GB Mac: every entry's memory floor is met. */
const BIG_MAC = 128 * GIB;
/** A 96 GB Mac: macOS reports exactly 96 GiB, the floor of the SSD-streaming entries. */
const MAC_96 = 96 * GIB;
/** A 36 GB Mac: below every memory floor. */
const SMALL_MAC = 36 * GIB;
/** The entries the catalog offers only from 96 GiB. */
const FLOORED = ['Qwen3.8-Flash-Next', 'Hy4-flash'];

/** `mlx download model --glob` matching for `*`-only globs: anchored, case-insensitive. */
function globMatches(pattern: string, text: string): boolean {
  const glob = pattern.toLowerCase();
  const name = text.toLowerCase();
  const parts = glob.split('*');
  if (parts.length === 1) return name === glob;
  const head = parts[0]!;
  const tail = parts[parts.length - 1]!;
  const end = name.length - tail.length;
  if (end < head.length || !name.startsWith(head) || !name.endsWith(tail)) return false;
  let at = head.length;
  for (const part of parts.slice(1, -1)) {
    const found = name.indexOf(part, at);
    if (found === -1 || found + part.length > end) return false;
    at = found + part.length;
  }
  return true;
}

/** Repo files the downloader always takes beside a glob selection. */
const CORE_FILES = new Set([
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.json',
  'merges.txt',
]);

/** The files a glob-filtered install takes: a glob matches the basename or the full repo path, plus the core files. */
function selectedFiles(entry: CatalogEntry, files: readonly string[]): string[] {
  return files.filter(
    (file) =>
      CORE_FILES.has(file) ||
      entry.globs!.some((glob) => globMatches(glob, file) || globMatches(glob, file.split('/').pop()!)),
  );
}

/** Expected weight variant per visible entry; every unsloth repo ships UD-Q4_K_XL, AngelSlim ships no UD quants. */
const VARIANT: Record<string, string> = {
  'Qwen3.8-27B': 'UD-Q4_K_XL',
  'Inkling-2-Mini': 'UD-Q4_K_XL',
  'Gemma-4-26B-A4B': 'UD-Q4_K_XL',
  'Qwen3.8-Flash-Next': 'UD-Q4_K_XL',
  'Hy4-flash': 'Q4_K_M',
};

/** `AngelSlim/Hy4-flash-GGUF` files on Hugging Face (2026-11-19, main ecd803c859a3). */
const HY4_REPO_FILES = [
  '.gitattributes',
  'Hy4-flash-IQ2_AH-00001-of-00002.gguf',
  'Hy4-flash-IQ2_AH-00002-of-00002.gguf',
  'Hy4-flash-Q3_K-00001-of-00002.gguf',
  'Hy4-flash-Q3_K-00002-of-00002.gguf',
  'Hy4-flash-Q4_K_M-00001-of-00002.gguf',
  'Hy4-flash-Q4_K_M-00002-of-00002.gguf',
  'Hy4-flash-SPARK-4Bit-Fast.gguf',
  'Hy4-flash-SPARK-4Bit.gguf',
  'LICENSE',
  'README.md',
  'README_CN.md',
  'README_SPARK-4Bit.md',
  'hy4-flash-gguf-benchmark.png',
  'mmproj-Hy4-flash-bf16.gguf',
  'patch/hy4-flash-mac-minimal.patch',
  'patch/hy4-flash-spark.patch',
];

/** `unsloth/Inkling-2-Mini-GGUF` files on Hugging Face (2026-11-23, main 800c946d7b9c). */
const INKLING_REPO_FILES = [
  '.gitattributes',
  'Inkling-2-Mini-UD-IQ1_M.gguf',
  'Inkling-2-Mini-UD-IQ1_S.gguf',
  'Inkling-2-Mini-UD-IQ2_XXS.gguf',
  'Inkling-2-Mini-UD-IQ3_XXS.gguf',
  'Inkling-2-Mini-UD-IQ4_XS.gguf',
  'Inkling-2-Mini-UD-Q2_K_XL.gguf',
  'Inkling-2-Mini-UD-Q3_K_XL.gguf',
  'Inkling-2-Mini-UD-Q4_K_M.gguf',
  'Inkling-2-Mini-UD-Q4_K_XL.gguf',
  'Inkling-2-Mini-UD-Q8_K_XL.gguf',
  'Mainline_Support/Inkling-2-Mini-UD-IQ1_M.gguf_file',
  'Mainline_Support/Inkling-2-Mini-UD-IQ1_S.gguf_file',
  'Mainline_Support/Inkling-2-Mini-UD-IQ2_XXS.gguf_file',
  'Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file',
  'Mainline_Support/Inkling-2-Mini-UD-IQ4_XS.gguf_file',
  'Mainline_Support/Inkling-2-Mini-UD-Q2_K_XL.gguf_file',
  'Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file',
  'Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file',
  'Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file',
  'Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file',
  'Mainline_Support/README.md',
  'Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file',
  'README.md',
  'assets/size_vs_kld.png',
  'assets/size_vs_top1.png',
  'dspark-Inkling-2-Mini-Q8_0.gguf',
  'dspark/README.md',
  'dspark/dspark-Inkling-2-Mini-BF16-mainline.gguf_file',
  'dspark/dspark-Inkling-2-Mini-BF16.gguf',
  'mmproj-BF16.gguf',
  'mmproj-F16.gguf',
  'mmproj-F32.gguf',
];

/** `unsloth/Qwen3.8-Flash-Next-GGUF` files on Hugging Face (2026-11-23, main f5d175de2cf7). */
const FLASH_NEXT_REPO_FILES = [
  '.gitattributes',
  'BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf',
  'BF16/Qwen3.8-Flash-Next-BF16-00002-of-00008.gguf',
  'BF16/Qwen3.8-Flash-Next-BF16-00003-of-00008.gguf',
  'BF16/Qwen3.8-Flash-Next-BF16-00004-of-00008.gguf',
  'BF16/Qwen3.8-Flash-Next-BF16-00005-of-00008.gguf',
  'BF16/Qwen3.8-Flash-Next-BF16-00006-of-00008.gguf',
  'BF16/Qwen3.8-Flash-Next-BF16-00007-of-00008.gguf',
  'BF16/Qwen3.8-Flash-Next-BF16-00008-of-00008.gguf',
  'MTP/README.md',
  'MTP/mtp-Qwen3.8-Flash-Next-BF16.gguf',
  'MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf',
  'MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf',
  'MTP/mtp-Qwen3.8-Flash-Next-shared-BF16.gguf',
  'MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf',
  'MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf',
  'Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf',
  'Q8_0/Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf',
  'Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf',
  'Q8_0/Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf',
  'Q8_0/Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf',
  'Q8_0/Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf',
  'README.md',
  'UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf',
  'UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf',
  'UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf',
  'UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf',
  'UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf',
  'UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf',
  'UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf',
  'UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-00002-of-00003.gguf',
  'UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-00003-of-00003.gguf',
  'UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf',
  'UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf',
  'UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00004.gguf',
  'UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf',
  'UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf',
  'UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf',
  'UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf',
  'UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf',
  'UD-Q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf',
  'UD-Q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00002-of-00003.gguf',
  'UD-Q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00003-of-00003.gguf',
  'UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00004.gguf',
  'UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00004.gguf',
  'UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00004.gguf',
  'UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00004-of-00004.gguf',
  'UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf',
  'UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00004.gguf',
  'UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00004.gguf',
  'UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00004-of-00004.gguf',
  'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
  'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf',
  'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf',
  'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf',
  'UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf',
  'UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf',
  'UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf',
  'UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf',
  'UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf',
  'UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf',
  'UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf',
  'UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf',
  'UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf',
  'UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf',
  'UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf',
  'UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf',
  'engram/Qwen3.8-Flash-Next-engram-BF16.gguf_file',
  'engram/Qwen3.8-Flash-Next-engram-IQ4_NL.gguf_file',
  'engram/Qwen3.8-Flash-Next-engram-Q5_1.gguf_file',
  'engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file',
  'engram/README.md',
  'engram/set_split.py',
  'imatrix_unsloth.gguf_file',
  'mmproj-BF16.gguf',
  'mmproj-F16.gguf',
  'mtp-Qwen3.8-Flash-Next-Q8_0.gguf',
];

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

  it('pins the five visible entries to their verified GGUF repos + sidecar sources', () => {
    // Load-bearing strings: a typo here is a 404 at best and the wrong weights
    // at worst. Verified on Hugging Face — the GGUF repos ship the globbed
    // variant (plus MTP/mmproj where globbed in), the assetsRepos ship the
    // tokenizer/config files the GGUF repos lack (ungated, unlike google/*).
    expect(visibleCatalog(BIG_MAC).map((entry) => [entry.hfRepo, entry.assetsRepo])).toEqual([
      ['unsloth/Qwen3.8-27B-GGUF', 'Qwen/Qwen3.8-27B'],
      ['unsloth/Inkling-2-Mini-GGUF', 'thinkingmachines/Inkling-2-Mini'],
      ['unsloth/gemma-4-26B-A4B-it-GGUF', 'unsloth/gemma-4-26B-A4B-it'],
      ['unsloth/Qwen3.8-Flash-Next-GGUF', 'Qwen/Qwen3.8-Flash-Next'],
      ['AngelSlim/Hy4-flash-GGUF', 'tencent/Hy4-flash'],
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
    const visible = visibleCatalog(BIG_MAC);
    expect(visible.map((entry) => entry.label).sort()).toEqual(Object.keys(VARIANT).sort());
    for (const entry of visible) {
      expect(entry.globs, entry.label).toBeDefined();
      expect(entry.globs!.length, entry.label).toBeGreaterThan(0);
      expect(
        entry.globs!.some((glob) => glob.includes(VARIANT[entry.label]!)),
        entry.label,
      ).toBe(true);
    }
  });

  it('Hy4-flash globs select exactly the two Q4_K_M shards out of the AngelSlim repo', () => {
    // The repo also ships Q3_K, IQ2_AH, two SPARK builds and a vision
    // projector: one of them in the selection is 63-162 GB of extra download.
    const hy4 = MODEL_CATALOG.find((entry) => entry.label === 'Hy4-flash')!;
    expect(selectedFiles(hy4, HY4_REPO_FILES)).toEqual([
      'Hy4-flash-Q4_K_M-00001-of-00002.gguf',
      'Hy4-flash-Q4_K_M-00002-of-00002.gguf',
    ]);
  });

  it('Inkling-2-Mini globs select the root weight, vision tower and DSpark draft, never the Mainline_Support copy', () => {
    // `*UD-Q4_K_XL*` would also take `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file`,
    // 17.6 GB more of the same tensors.
    const inkling = MODEL_CATALOG.find((entry) => entry.label === 'Inkling-2-Mini')!;
    const selected = selectedFiles(inkling, INKLING_REPO_FILES);
    expect(selected).toEqual(['Inkling-2-Mini-UD-Q4_K_XL.gguf', 'dspark-Inkling-2-Mini-Q8_0.gguf', 'mmproj-BF16.gguf']);
    expect(selected).not.toContain('Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file');
    // 17,629,759,456 + 402,278,560 + 641,805,760 B.
    expect(inkling.sizeGb).toBe(18.7);
  });

  it('Qwen3.8-Flash-Next globs select exactly the four UD-Q4_K_XL shards in their repo folder', () => {
    const flashNext = MODEL_CATALOG.find((entry) => entry.label === 'Qwen3.8-Flash-Next')!;
    expect(selectedFiles(flashNext, FLASH_NEXT_REPO_FILES)).toEqual([
      'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
      'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf',
      'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf',
      'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf',
    ]);
    // 10,946,618 + 49,720,539,552 + 32,803,030,176 + 28,800,138,432 B.
    expect(flashNext.sizeGb).toBe(111.3);
  });

  it('Hy4-flash is sized to its two shards', () => {
    const hy4 = MODEL_CATALOG.find((entry) => entry.label === 'Hy4-flash')!;
    // 104,986,517,504 + 63,360,616,128 B.
    expect(hy4.sizeGb).toBe(168.3);
    expect(hy4.hidden).toBeUndefined();
    expect(hy4.draft).toBeUndefined();
  });

  it('only Qwen3.8-Flash-Next and Hy4-flash carry a memory floor, 96 GiB each', () => {
    for (const entry of MODEL_CATALOG) {
      expect(entry.minMemoryGb, entry.label).toBe(FLOORED.includes(entry.label) ? 96 : undefined);
    }
    // Hy4-flash's load admission accepts 39 GiB; the catalog floor must never sit below it.
    expect(MODEL_CATALOG.find((entry) => entry.label === 'Hy4-flash')!.minMemoryGb).toBeGreaterThanOrEqual(39);
  });

  it('every visible entry names an assetsRepo for tokenizer sidecars', () => {
    // The GGUF repos ship no tokenizer files; without the official sidecars
    // the native runtime extracts the embedded tokenizer and tool calling
    // silently breaks (see the assetsRepo field comment).
    for (const entry of visibleCatalog(BIG_MAC)) {
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
  it('offers Qwen3.8-Flash-Next and Hy4-flash from 96 GiB, at exactly 96 GiB too', () => {
    for (const label of FLOORED) {
      const entry = MODEL_CATALOG.find((candidate) => candidate.label === label)!;
      expect(visibleCatalog(SMALL_MAC), label).not.toContain(entry);
      expect(visibleCatalog(64 * GIB), label).not.toContain(entry);
      expect(visibleCatalog(MAC_96 - 1), label).not.toContain(entry);
      expect(visibleCatalog(MAC_96), label).toContain(entry);
      expect(visibleCatalog(BIG_MAC), label).toContain(entry);
    }
    // Entries without a floor stay on the smallest machine.
    const small = visibleCatalog(SMALL_MAC);
    for (const entry of MODEL_CATALOG.filter(
      (item) => !item.hidden && (item.category ?? 'chat') === 'chat' && item.minMemoryGb === undefined,
    )) {
      expect(small, entry.label).toContain(entry);
    }
  });

  it('catalogFitsMemory passes an entry without a floor on any machine', () => {
    expect(catalogFitsMemory({}, 0)).toBe(true);
    expect(catalogFitsMemory({ minMemoryGb: 64 }, 64 * GIB)).toBe(true);
    expect(catalogFitsMemory({ minMemoryGb: 64 }, 64 * GIB - 1)).toBe(false);
  });

  it("defaults to this machine's RAM", () => {
    expect(visibleCatalog()).toEqual(visibleCatalog(totalmem()));
  });

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
      expect(entry.category ?? 'chat', entry.label).toBe('chat');
    }
  });
});

describe('visibleSystemOneCatalog', () => {
  it('returns the verified Clairic Hugging Face entries and keeps them out of visibleCatalog', () => {
    const sysOne = visibleSystemOneCatalog();
    expect(sysOne.map((entry) => [entry.label, entry.hfRepo, entry.assetsRepo, entry.sizeGb, entry.category])).toEqual([
      ['Clairic Flash (9B)', 'Brooooooklyn/Clairic-Flash-GGUF', 'Cloudflare/clairic-flash', 7.6, 'systemone'],
      ['Clairic (27B)', 'Brooooooklyn/Clairic-GGUF', 'Cloudflare/clairic', 18.9, 'systemone'],
    ]);
    const chatRepos = new Set(visibleCatalog().map((entry) => entry.hfRepo));
    for (const entry of sysOne) {
      expect(chatRepos.has(entry.hfRepo)).toBe(false);
    }
  });

  it('includes System One repos in both catalogDownloadRepos and catalogUpdateRepos', () => {
    const downloads = catalogDownloadRepos();
    const updates = catalogUpdateRepos();
    for (const entry of visibleSystemOneCatalog()) {
      expect(downloads).toContain(entry.hfRepo);
      expect(updates).toContain(entry.hfRepo);
    }
  });
});
