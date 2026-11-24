import { totalmem } from 'node:os';

import { familyDataFor } from '@mlx-node/lm/family-data';
import { ggufNames, ggufVariant } from '@mlx-node/lm/model-variant';
import { describe, expect, it } from 'vite-plus/test';

import {
  type CatalogBuild,
  catalogBuildRepo,
  catalogBuildSlug,
  catalogDraftSlug,
  catalogFitsMemory,
  catalogSharesRepo,
  catalogSupportedOnPlatform,
  catalogTargetForSlug,
  catalogTargets,
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

/** `[path, bytes]` of every file of a repository listing. */
type RepoFiles = ReadonlyArray<readonly [string, number]>;

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

/** Repo files the downloader always takes beside a glob selection (`CORE_FILES` in `packages/dashboard/src/download.ts`). */
const CORE_FILES = new Set([
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.json',
  'merges.txt',
]);

/** The downloader's filter with no globs (`isWantedFile` in `packages/dashboard/src/download.ts`). */
function defaultWanted(path: string): boolean {
  return (
    CORE_FILES.has(path) ||
    ['.safetensors', '.json', '.pdiparams', '.yml', '.gguf', '.jinja'].some((suffix) => path.endsWith(suffix))
  );
}

/**
 * The files an install of `build` takes, in listing order: with globs, a glob matches the basename or the full repo
 * path, plus the core files; without, the default filter (`isWantedFileFor` in `packages/dashboard/src/download.ts`).
 */
function selectedFiles(build: CatalogBuild, files: RepoFiles): string[] {
  const globs = build.globs;
  return files
    .map(([path]) => path)
    .filter((path) =>
      globs === undefined
        ? defaultWanted(path)
        : CORE_FILES.has(path) ||
          globs.some((glob) => globMatches(glob, path) || globMatches(glob, path.split('/').pop()!)),
    );
}

/** The byte sum of `paths` in `files`. */
function bytesOf(files: RepoFiles, paths: readonly string[]): number {
  const sizes = new Map(files);
  return paths.reduce((sum, path) => sum + sizes.get(path)!, 0);
}

/** The build `id` of the entry `label`. */
function buildOf(label: string, id: string): CatalogBuild {
  const build = MODEL_CATALOG.find((entry) => entry.label === label)?.builds.find((candidate) => candidate.id === id);
  if (build === undefined) throw new Error(`no build ${label} ${id}`);
  return build;
}

/** Every build of every entry, hidden ones included. */
const ALL_BUILDS = MODEL_CATALOG.flatMap((entry) => entry.builds.map((build) => ({ entry, build })));

/** `unsloth/Qwen3.8-27B-GGUF` files on Hugging Face (2026-11-23, main 4af51b35). */
const QWEN_REPO_FILES: RepoFiles = [
  ['.gitattributes', 4238],
  ['BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf', 49986159616],
  ['BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf', 4671576000],
  ['MTP/mtp-Qwen3.8-27B-Q4_0.gguf', 1369590656],
  ['Qwen3.8-27B-Q4_0.gguf', 16056478688],
  ['Qwen3.8-27B-Q4_1.gguf', 17540705248],
  ['Qwen3.8-27B-Q8_0.gguf', 29047086048],
  ['Qwen3.8-27B-UD-IQ1_M.gguf', 6729166848],
  ['Qwen3.8-27B-UD-IQ1_S.gguf', 6192222208],
  ['Qwen3.8-27B-UD-IQ2_S.gguf', 8422719488],
  ['Qwen3.8-27B-UD-IQ2_XS.gguf', 7625883648],
  ['Qwen3.8-27B-UD-IQ2_XXS.gguf', 7206043648],
  ['Qwen3.8-27B-UD-IQ3_S.gguf', 12040883104],
  ['Qwen3.8-27B-UD-IQ3_XXS.gguf', 10934860704],
  ['Qwen3.8-27B-UD-IQ4_XS.gguf', 14352911264],
  ['Qwen3.8-27B-UD-Q2_K_XL.gguf', 9948666784],
  ['Qwen3.8-27B-UD-Q3_K_XL.gguf', 13146393504],
  ['Qwen3.8-27B-UD-Q4_K_M.gguf', 16564259744],
  ['Qwen3.8-27B-UD-Q4_K_S.gguf', 15358213024],
  ['Qwen3.8-27B-UD-Q4_K_XL.gguf', 17659407264],
  ['Qwen3.8-27B-UD-Q5_K_M.gguf', 19870878624],
  ['Qwen3.8-27B-UD-Q5_K_S.gguf', 18764794784],
  ['Qwen3.8-27B-UD-Q5_K_XL.gguf', 20976962464],
  ['Qwen3.8-27B-UD-Q6_K.gguf', 21980441504],
  ['Qwen3.8-27B-UD-Q6_K_L.gguf', 24181549984],
  ['Qwen3.8-27B-UD-Q6_K_M.gguf', 23087180704],
  ['Qwen3.8-27B-UD-Q6_K_XL.gguf', 25299061664],
  ['Qwen3.8-27B-UD-Q8_K_L.gguf', 28045695904],
  ['Qwen3.8-27B-UD-Q8_K_XL.gguf', 31457991680],
  ['README.md', 6945],
  ['config.json', 3760],
  ['imatrix_unsloth.gguf', 13642656],
  ['mmproj-BF16.gguf', 931146432],
  ['mmproj-F16.gguf', 927607488],
];

/** `unsloth/Inkling-2-Mini-GGUF` files on Hugging Face (2026-11-23, main 800c946d7b9c). */
const INKLING_REPO_FILES: RepoFiles = [
  ['.gitattributes', 5674],
  ['Inkling-2-Mini-UD-IQ1_M.gguf', 8024441024],
  ['Inkling-2-Mini-UD-IQ1_S.gguf', 7764291776],
  ['Inkling-2-Mini-UD-IQ2_XXS.gguf', 8486236352],
  ['Inkling-2-Mini-UD-IQ3_XXS.gguf', 11663309504],
  ['Inkling-2-Mini-UD-IQ4_XS.gguf', 15733755232],
  ['Inkling-2-Mini-UD-Q2_K_XL.gguf', 9138715328],
  ['Inkling-2-Mini-UD-Q3_K_XL.gguf', 13631778016],
  ['Inkling-2-Mini-UD-Q4_K_M.gguf', 16610543328],
  ['Inkling-2-Mini-UD-Q4_K_XL.gguf', 17629759456],
  ['Inkling-2-Mini-UD-Q8_K_XL.gguf', 22867451616],
  ['Mainline_Support/Inkling-2-Mini-UD-IQ1_M.gguf_file', 8024442304],
  ['Mainline_Support/Inkling-2-Mini-UD-IQ1_S.gguf_file', 7764293056],
  ['Mainline_Support/Inkling-2-Mini-UD-IQ2_XXS.gguf_file', 8486237632],
  ['Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file', 11663310784],
  ['Mainline_Support/Inkling-2-Mini-UD-IQ4_XS.gguf_file', 15733756512],
  ['Mainline_Support/Inkling-2-Mini-UD-Q2_K_XL.gguf_file', 9138716608],
  ['Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file', 13631779296],
  ['Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file', 16610544608],
  ['Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file', 17629760736],
  ['Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file', 22867452896],
  ['Mainline_Support/README.md', 5733],
  ['Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file', 503889952],
  ['README.md', 56464],
  ['assets/size_vs_kld.png', 273844],
  ['assets/size_vs_top1.png', 275713],
  ['dspark-Inkling-2-Mini-Q8_0.gguf', 402278560],
  ['dspark/README.md', 4144],
  ['dspark/dspark-Inkling-2-Mini-BF16-mainline.gguf_file', 749371936],
  ['dspark/dspark-Inkling-2-Mini-BF16.gguf', 647792800],
  ['mmproj-BF16.gguf', 641805760],
  ['mmproj-F16.gguf', 639348160],
  ['mmproj-F32.gguf', 1273048512],
];

/** `thinkingmachines/Inkling-2-Mini` files on Hugging Face (2026-11-23, main 7a67ae23). */
const INKLING_ORIGINAL_REPO_FILES: RepoFiles = [
  ['.gitattributes', 1570],
  ['README.md', 54898],
  ['chat_template.jinja', 6868],
  ['config.json', 2722],
  ['generation_config.json', 98],
  ['hf_quant_config.json', 67924],
  ['model-00001-of-00001.safetensors', 19526019168],
  ['model.safetensors.index.json', 109300],
  ['mtp.safetensors', 635801416],
  ['processor_config.json', 461],
  ['special_tokens_map.json', 529],
  ['tiktoken/tokenizer.model', 3386494],
  ['tokenizer.json', 26178678],
  ['tokenizer_config.json', 13235],
];

/** `unsloth/Qwen3.8-Flash-Next-GGUF` files on Hugging Face (2026-11-23, main f5d175de2cf7). */
const FLASH_NEXT_REPO_FILES: RepoFiles = [
  ['.gitattributes', 9116],
  ['BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf', 10946368],
  ['BF16/Qwen3.8-Flash-Next-BF16-00002-of-00008.gguf', 10447201664],
  ['BF16/Qwen3.8-Flash-Next-BF16-00003-of-00008.gguf', 102400491712],
  ['BF16/Qwen3.8-Flash-Next-BF16-00004-of-00008.gguf', 48394236672],
  ['BF16/Qwen3.8-Flash-Next-BF16-00005-of-00008.gguf', 48367899040],
  ['BF16/Qwen3.8-Flash-Next-BF16-00006-of-00008.gguf', 48615233984],
  ['BF16/Qwen3.8-Flash-Next-BF16-00007-of-00008.gguf', 48367899040],
  ['BF16/Qwen3.8-Flash-Next-BF16-00008-of-00008.gguf', 47426022016],
  ['MTP/README.md', 5906],
  ['MTP/mtp-Qwen3.8-Flash-Next-BF16.gguf', 7770760320],
  ['MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf', 2786204800],
  ['MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf', 4137429120],
  ['MTP/mtp-Qwen3.8-Flash-Next-shared-BF16.gguf', 5227963456],
  ['MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf', 1907151936],
  ['MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf', 2786568256],
  ['Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf', 10946624],
  ['Q8_0/Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf', 682434912],
  ['Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf', 54400261312],
  ['Q8_0/Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf', 49446841216],
  ['Q8_0/Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf', 49668930400],
  ['Q8_0/Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf', 34015618784],
  ['README.md', 60460],
  ['UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf', 10946618],
  ['UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf', 39196009888],
  ['UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf', 28800138432],
  ['UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf', 10946618],
  ['UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf', 37568095648],
  ['UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf', 28800138432],
  ['UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf', 10946618],
  ['UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-00002-of-00003.gguf', 47021008288],
  ['UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-00003-of-00003.gguf', 28800138432],
  ['UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf', 10946618],
  ['UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf', 49946027488],
  ['UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00004.gguf', 4869325888],
  ['UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf', 28800138432],
  ['UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf', 10946618],
  ['UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf', 49604491008],
  ['UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf', 15256783680],
  ['UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf', 28800138432],
  ['UD-Q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf', 10946618],
  ['UD-Q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00002-of-00003.gguf', 41738806688],
  ['UD-Q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00003-of-00003.gguf', 28800138432],
  ['UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00004.gguf', 10946618],
  ['UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00004.gguf', 49732486272],
  ['UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00004.gguf', 1037181344],
  ['UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00004-of-00004.gguf', 28800138432],
  ['UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf', 10946618],
  ['UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00004.gguf', 49742117248],
  ['UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00004.gguf', 10911746208],
  ['UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00004-of-00004.gguf', 28800138432],
  ['UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf', 10946618],
  ['UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf', 49720539552],
  ['UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf', 32803030176],
  ['UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf', 28800138432],
  ['UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf', 10946618],
  ['UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf', 682434912],
  ['UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf', 54400261312],
  ['UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf', 49990245824],
  ['UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf', 49882416192],
  ['UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf', 3320101792],
  ['UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf', 10946624],
  ['UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf', 682434912],
  ['UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf', 54400261312],
  ['UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf', 49814108512],
  ['UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf', 49419317344],
  ['UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf', 14838313984],
  ['engram/Qwen3.8-Flash-Next-engram-BF16.gguf_file', 102400491712],
  ['engram/Qwen3.8-Flash-Next-engram-IQ4_NL.gguf_file', 28800138432],
  ['engram/Qwen3.8-Flash-Next-engram-Q5_1.gguf_file', 38400184512],
  ['engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file', 54400261312],
  ['engram/README.md', 5097],
  ['engram/set_split.py', 350],
  ['imatrix_unsloth.gguf_file', 580038720],
  ['mmproj-BF16.gguf', 907542944],
  ['mmproj-F16.gguf', 904004000],
  ['mtp-Qwen3.8-Flash-Next-Q8_0.gguf', 4137429120],
];

/** `AngelSlim/Hy4-flash-GGUF` files on Hugging Face (2026-11-19, main ecd803c859a3). */
const HY4_REPO_FILES: RepoFiles = [
  ['.gitattributes', 2753],
  ['Hy4-flash-IQ2_AH-00001-of-00002.gguf', 71635656768],
  ['Hy4-flash-IQ2_AH-00002-of-00002.gguf', 63360616128],
  ['Hy4-flash-Q3_K-00001-of-00002.gguf', 92010389504],
  ['Hy4-flash-Q3_K-00002-of-00002.gguf', 63360616128],
  ['Hy4-flash-Q4_K_M-00001-of-00002.gguf', 104986517504],
  ['Hy4-flash-Q4_K_M-00002-of-00002.gguf', 63360616128],
  ['Hy4-flash-SPARK-4Bit-Fast.gguf', 159611554144],
  ['Hy4-flash-SPARK-4Bit.gguf', 162509777248],
  ['LICENSE', 13942],
  ['README.md', 4868],
  ['README_CN.md', 4612],
  ['README_MAC-Metal.md', 10447],
  ['README_SPARK-4Bit.md', 13234],
  ['mmproj-Hy4-flash-bf16.gguf', 1044397440],
  ['patch/hy4-flash-mac-minimal.patch', 303694],
  ['patch/hy4-flash-spark.patch', 315266],
];

/** `ggml-org/Clef-Flash-GGUF` files on Hugging Face (2026-11-12, main c3d92545). */
const CLEF_FLASH_REPO_FILES: RepoFiles = [
  ['.gitattributes', 1820],
  ['.src_sha', 49],
  ['Clef-Flash-BF16.gguf', 18164488448],
  ['Clef-Flash-Q4_K_M.gguf', 6732781824],
  ['Clef-Flash-Q8_0.gguf', 9657260288],
  ['README.md', 559],
  ['convert.log', 337134],
  ['mmproj-Clef-Flash-BF16.gguf', 921704928],
  ['mmproj-Clef-Flash-Q8_0.gguf', 624229728],
];

/** `ggml-org/Clef-GGUF` files on Hugging Face (2026-11-12, main 415c1d23). */
const CLEF_REPO_FILES: RepoFiles = [
  ['.gitattributes', 1790],
  ['.src_sha', 49],
  ['Clef-BF16.gguf', 54064664736],
  ['Clef-Q4_K_M.gguf', 19540136096],
  ['Clef-Q8_0.gguf', 28732215456],
  ['README.md', 535],
  ['convert.log', 523228],
  ['mmproj-Clef-BF16.gguf', 931146336],
  ['mmproj-Clef-Q8_0.gguf', 629247456],
];

/** The real file list of each published GGUF build's repository. */
const REPO_FILES: ReadonlyMap<string, RepoFiles> = new Map([
  ['unsloth/Qwen3.8-27B-GGUF', QWEN_REPO_FILES],
  ['unsloth/Inkling-2-Mini-GGUF', INKLING_REPO_FILES],
  ['thinkingmachines/Inkling-2-Mini', INKLING_ORIGINAL_REPO_FILES],
  ['unsloth/Qwen3.8-Flash-Next-GGUF', FLASH_NEXT_REPO_FILES],
  ['AngelSlim/Hy4-flash-GGUF', HY4_REPO_FILES],
  ['ggml-org/Clef-Flash-GGUF', CLEF_FLASH_REPO_FILES],
  ['ggml-org/Clef-GGUF', CLEF_REPO_FILES],
]);

describe('install folders', () => {
  it('pins every install folder', () => {
    const folders = [
      ...ALL_BUILDS.map(({ entry, build }) => [entry.label, build.id, catalogBuildSlug(build, 'darwin')]),
      ...MODEL_CATALOG.flatMap((entry) =>
        entry.draft === undefined ? [] : [[entry.label, entry.draft.label, catalogDraftSlug(entry.draft)]],
      ),
    ];
    expect(folders).toEqual([
      ['Qwen3.8-27B', 'UD-Q4_K_XL', 'qwen3.8-27b-gguf'],
      ['Qwen3.8-27B', 'UD-Q4_K_M', 'qwen3.8-27b-gguf-ud-q4_k_m'],
      ['Inkling-2-Mini', 'UD-Q4_K_XL', 'inkling-2-mini-gguf'],
      ['Inkling-2-Mini', 'UD-Q4_K_M', 'inkling-2-mini-gguf-ud-q4_k_m'],
      ['Inkling-2-Mini', 'Original', 'inkling-2-mini'],
      ['Qwen3.8-Flash-Next', 'UD-Q4_K_XL', 'qwen3.8-flash-next-gguf'],
      ['Hy4-flash', 'Q4_K_M', 'hy4-flash-gguf'],
      ['Gemma-4-12B', 'MXFP', 'gemma-4-12b-it-mxfp-mlx'],
      ['Nemotron-3.5-Lightning-30B-A3B', 'NVFP4', 'nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4-mlx'],
      ['Clairic Flash (9B)', 'UD-Q4_K_XL', 'clairic-flash-gguf'],
      ['Clairic (27B)', 'UD-Q4_K_XL', 'clairic-gguf'],
      ['CLEF Flash (9B)', 'Q8_0', 'clef-flash-gguf'],
      ['CLEF Flash (9B)', 'Q4_K_M', 'clef-flash-gguf-q4_k_m'],
      ['CLEF (27B)', 'Q8_0', 'clef-gguf'],
      ['CLEF (27B)', 'Q4_K_M', 'clef-gguf-q4_k_m'],
      ['Qwen3.8-27B', 'DFlash2', 'qwen3.8-27b-dflash2'],
      ['Inkling-2-Mini', 'Inkling-2-Mini DFlash2', 'inkling-2-mini-dflash2'],
    ]);
    const names = folders.map((row) => row[2]);
    expect(new Set(names).size).toBe(names.length);
  });

  it('gives at most one build of a repository the folder with no suffix', () => {
    const plain = new Map<string, number>();
    for (const { build } of ALL_BUILDS) {
      if (build.folderSuffix !== undefined) continue;
      plain.set(build.hfRepo, (plain.get(build.hfRepo) ?? 0) + 1);
    }
    for (const [repo, count] of plain) expect(count, repo).toBe(1);
    // A suffix is only for a build whose repository another build already installs.
    for (const { build } of ALL_BUILDS) {
      if (build.folderSuffix !== undefined) expect(catalogSharesRepo(build, 'darwin'), build.id).toBe(true);
    }
  });

  it('names the repositories two builds share, and only those', () => {
    const shared = ALL_BUILDS.filter(({ build }) => catalogSharesRepo(build, 'darwin')).map(({ entry, build }) => [
      entry.label,
      build.id,
    ]);
    expect(shared).toEqual([
      ['Qwen3.8-27B', 'UD-Q4_K_XL'],
      ['Qwen3.8-27B', 'UD-Q4_K_M'],
      ['Inkling-2-Mini', 'UD-Q4_K_XL'],
      ['Inkling-2-Mini', 'UD-Q4_K_M'],
      ['CLEF Flash (9B)', 'Q8_0'],
      ['CLEF Flash (9B)', 'Q4_K_M'],
      ['CLEF (27B)', 'Q8_0'],
      ['CLEF (27B)', 'Q4_K_M'],
    ]);
  });

  it('resolves the CUDA build only when a build carries one, and its folder follows the repo', () => {
    // Via the pure parameter, never by mutating `process.platform`: that global is
    // shared with every test file on this worker.
    const withCuda: CatalogBuild = { id: 't', hfRepo: 'a/Metal', hfRepoCuda: 'a/Cuda-X', sizeGb: 1, source: 'MLX' };
    expect(catalogBuildRepo(withCuda, 'linux')).toBe('a/Cuda-X');
    expect(catalogBuildRepo(withCuda, 'darwin')).toBe('a/Metal');
    expect(catalogBuildSlug(withCuda, 'linux')).toBe('cuda-x');
    expect(catalogBuildSlug({ ...withCuda, folderSuffix: 'q4' }, 'darwin')).toBe('metal-q4');
    for (const { entry, build } of ALL_BUILDS) {
      expect(build.hfRepoCuda, entry.label).toBeUndefined();
      expect(catalogBuildRepo(build, 'linux'), entry.label).toBe(build.hfRepo);
      expect(catalogBuildRepo(build), entry.label).toBe(catalogBuildRepo(build, process.platform));
    }
  });
});

describe('builds', () => {
  it('pins the four chat models and their builds to their verified repos + sidecar sources', () => {
    // Load-bearing strings: a typo here is a 404 at best and the wrong weights
    // at worst. Verified on Hugging Face.
    expect(
      visibleCatalog(BIG_MAC, 'darwin').map((entry) => [
        entry.label,
        entry.builds.map((build) => [build.id, build.hfRepo, build.assetsRepo, build.sizeGb, build.source]),
      ]),
    ).toEqual([
      [
        'Qwen3.8-27B',
        [
          ['UD-Q4_K_XL', 'unsloth/Qwen3.8-27B-GGUF', 'Qwen/Qwen3.8-27B', 17.7, 'Unsloth GGUF'],
          ['UD-Q4_K_M', 'unsloth/Qwen3.8-27B-GGUF', 'Qwen/Qwen3.8-27B', 16.6, 'Unsloth GGUF'],
        ],
      ],
      [
        'Inkling-2-Mini',
        [
          ['UD-Q4_K_XL', 'unsloth/Inkling-2-Mini-GGUF', 'thinkingmachines/Inkling-2-Mini', 18.7, 'Unsloth GGUF'],
          ['UD-Q4_K_M', 'unsloth/Inkling-2-Mini-GGUF', 'thinkingmachines/Inkling-2-Mini', 17.7, 'Unsloth GGUF'],
          ['Original', 'thinkingmachines/Inkling-2-Mini', undefined, 20.2, 'Thinking Machines'],
        ],
      ],
      [
        'Qwen3.8-Flash-Next',
        [['UD-Q4_K_XL', 'unsloth/Qwen3.8-Flash-Next-GGUF', 'Qwen/Qwen3.8-Flash-Next', 111.3, 'Unsloth GGUF']],
      ],
      ['Hy4-flash', [['Q4_K_M', 'AngelSlim/Hy4-flash-GGUF', 'tencent/Hy4-flash', 168.3, 'AngelSlim GGUF']]],
    ]);
    expect(MODEL_CATALOG.map((entry) => entry.label)).not.toContain('Gemma-4-26B-A4B');
  });

  it('every repo is a plausible HF slug, and every build id is unique in its entry', () => {
    for (const entry of MODEL_CATALOG) {
      const ids = entry.builds.map((build) => build.id);
      expect(new Set(ids).size, entry.label).toBe(ids.length);
      for (const build of entry.builds) {
        expect(build.hfRepo, entry.label).toMatch(/^[A-Za-z0-9_-]+\/[A-Za-z0-9._-]+$/);
        if (build.assetsRepo !== undefined) {
          expect(build.assetsRepo, entry.label).toMatch(/^[A-Za-z0-9_-]+\/[A-Za-z0-9._-]+$/);
        }
      }
    }
  });

  it('every GGUF build names an assetsRepo for tokenizer sidecars', () => {
    // The GGUF repos ship no tokenizer files; without the official sidecars
    // the native runtime extracts the embedded tokenizer and tool calling
    // silently breaks (see the assetsRepo field comment).
    for (const entry of visibleCatalog(BIG_MAC, 'darwin')) {
      for (const build of entry.builds) {
        if (build.hfRepo.endsWith('-GGUF')) expect(build.assetsRepo, `${entry.label} ${build.id}`).toBeDefined();
      }
    }
  });

  it('a build id is the tag discovery reads from its weight file', () => {
    // The weight file of each build, in the folder it installs to, through the
    // engine's own name parser.
    const weights: ReadonlyArray<readonly [CatalogBuild, string, string | undefined]> = [
      [buildOf('Qwen3.8-27B', 'UD-Q4_K_XL'), 'Qwen3.8-27B-UD-Q4_K_XL.gguf', undefined],
      [buildOf('Qwen3.8-27B', 'UD-Q4_K_M'), 'Qwen3.8-27B-UD-Q4_K_M.gguf', undefined],
      [buildOf('Inkling-2-Mini', 'UD-Q4_K_XL'), 'Inkling-2-Mini-UD-Q4_K_XL.gguf', undefined],
      [buildOf('Inkling-2-Mini', 'UD-Q4_K_M'), 'Inkling-2-Mini-UD-Q4_K_M.gguf', undefined],
      [buildOf('Qwen3.8-Flash-Next', 'UD-Q4_K_XL'), 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf', 'UD-Q4_K_XL'],
      [buildOf('Hy4-flash', 'Q4_K_M'), 'Hy4-flash-Q4_K_M-00001-of-00002.gguf', undefined],
      [buildOf('CLEF Flash (9B)', 'Q8_0'), 'Clef-Flash-Q8_0.gguf', undefined],
      [buildOf('CLEF Flash (9B)', 'Q4_K_M'), 'Clef-Flash-Q4_K_M.gguf', undefined],
      [buildOf('CLEF (27B)', 'Q8_0'), 'Clef-Q8_0.gguf', undefined],
      [buildOf('CLEF (27B)', 'Q4_K_M'), 'Clef-Q4_K_M.gguf', undefined],
    ];
    for (const [build, file, variantFolder] of weights) {
      const folder = catalogBuildSlug(build, 'darwin');
      const names = ggufNames(file, folder, variantFolder);
      expect(ggufVariant(names.stem, folder, variantFolder, names.id).variant, file).toBe(build.id);
    }
  });
});

describe('the files each build installs', () => {
  it('selects the real files of every published build, and its size is their byte sum', () => {
    const expected: ReadonlyArray<readonly [string, string, readonly string[], number]> = [
      ['Qwen3.8-27B', 'UD-Q4_K_XL', ['Qwen3.8-27B-UD-Q4_K_XL.gguf', 'config.json'], 17_659_411_024],
      ['Qwen3.8-27B', 'UD-Q4_K_M', ['Qwen3.8-27B-UD-Q4_K_M.gguf', 'config.json'], 16_564_263_504],
      [
        'Inkling-2-Mini',
        'UD-Q4_K_XL',
        ['Inkling-2-Mini-UD-Q4_K_XL.gguf', 'dspark-Inkling-2-Mini-Q8_0.gguf', 'mmproj-BF16.gguf'],
        18_673_843_776,
      ],
      [
        'Inkling-2-Mini',
        'UD-Q4_K_M',
        ['Inkling-2-Mini-UD-Q4_K_M.gguf', 'dspark-Inkling-2-Mini-Q8_0.gguf', 'mmproj-BF16.gguf'],
        17_654_627_648,
      ],
      [
        'Inkling-2-Mini',
        'Original',
        [
          'chat_template.jinja',
          'config.json',
          'generation_config.json',
          'hf_quant_config.json',
          'model-00001-of-00001.safetensors',
          'model.safetensors.index.json',
          'mtp.safetensors',
          'processor_config.json',
          'special_tokens_map.json',
          'tokenizer.json',
          'tokenizer_config.json',
        ],
        20_188_200_399,
      ],
      [
        'Qwen3.8-Flash-Next',
        'UD-Q4_K_XL',
        [
          'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
          'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf',
          'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf',
          'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf',
        ],
        111_334_654_778,
      ],
      [
        'Hy4-flash',
        'Q4_K_M',
        ['Hy4-flash-Q4_K_M-00001-of-00002.gguf', 'Hy4-flash-Q4_K_M-00002-of-00002.gguf'],
        168_347_133_632,
      ],
      ['CLEF Flash (9B)', 'Q8_0', ['Clef-Flash-Q8_0.gguf'], 9_657_260_288],
      ['CLEF Flash (9B)', 'Q4_K_M', ['Clef-Flash-Q4_K_M.gguf'], 6_732_781_824],
      ['CLEF (27B)', 'Q8_0', ['Clef-Q8_0.gguf'], 28_732_215_456],
      ['CLEF (27B)', 'Q4_K_M', ['Clef-Q4_K_M.gguf'], 19_540_136_096],
    ];
    const covered = new Set<CatalogBuild>();
    for (const [label, id, files, bytes] of expected) {
      const build = buildOf(label, id);
      covered.add(build);
      const listing = REPO_FILES.get(build.hfRepo)!;
      const selected = selectedFiles(build, listing);
      expect(selected, `${label} ${id}`).toEqual(files);
      expect(bytesOf(listing, selected), `${label} ${id}`).toBe(bytes);
      expect(Math.round(bytes / 1e8) / 10, `${label} ${id}`).toBe(build.sizeGb);
    }
    // Every chat build and every CLEF build of the catalog is checked here.
    for (const entry of [...visibleCatalog(BIG_MAC, 'darwin'), ...visibleSystemOneCatalog(BIG_MAC, 'darwin')]) {
      if (entry.label.startsWith('Clairic')) continue;
      for (const build of entry.builds) expect(covered.has(build), `${entry.label} ${build.id}`).toBe(true);
    }
  });

  it('a CLEF glob is the exact file name: a wildcard on the quant also takes the mmproj', () => {
    const wildcard: CatalogBuild = { ...buildOf('CLEF Flash (9B)', 'Q8_0'), globs: ['*Q8_0*'] };
    expect(selectedFiles(wildcard, CLEF_FLASH_REPO_FILES)).toEqual([
      'Clef-Flash-Q8_0.gguf',
      'mmproj-Clef-Flash-Q8_0.gguf',
    ]);
    // A base config.json beside the GGUF makes discovery drop it: no CLEF build takes a sidecar repo.
    const clef = MODEL_CATALOG.filter((item) => item.ggufFamily === 'clef');
    expect(clef.map((entry) => entry.label)).toEqual(['CLEF Flash (9B)', 'CLEF (27B)']);
    for (const entry of clef) {
      expect(entry.minMemoryGb, entry.label).toBeUndefined();
      for (const build of entry.builds) expect(build.assetsRepo, `${entry.label} ${build.id}`).toBeUndefined();
    }
  });

  it('the Qwen UD-Q4_K_M glob takes one weight and not the XL file', () => {
    const selected = selectedFiles(buildOf('Qwen3.8-27B', 'UD-Q4_K_M'), QWEN_REPO_FILES);
    expect(selected.filter((path) => path.endsWith('.gguf'))).toEqual(['Qwen3.8-27B-UD-Q4_K_M.gguf']);
    expect(selected).not.toContain('Qwen3.8-27B-UD-Q4_K_XL.gguf');
  });

  it('the Inkling UD-Q4_K_M glob leaves Mainline_Support alone', () => {
    // `*UD-Q4_K_M*` would also take `Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file`,
    // 16.6 GB more of the same tensors.
    const selected = selectedFiles(buildOf('Inkling-2-Mini', 'UD-Q4_K_M'), INKLING_REPO_FILES);
    expect(selected).toHaveLength(3);
    expect(selected).not.toContain('Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file');
    expect(selected).not.toContain('Inkling-2-Mini-UD-Q4_K_XL.gguf');
  });

  it('Inkling Original takes the default filter: no globs, no sidecar repo, and not the tiktoken file', () => {
    const original = buildOf('Inkling-2-Mini', 'Original');
    expect(original.globs).toBeUndefined();
    expect(original.assetsRepo).toBeUndefined();
    expect(selectedFiles(original, INKLING_ORIGINAL_REPO_FILES)).not.toContain('tiktoken/tokenizer.model');
  });
});

describe('download targets', () => {
  it('lists every build and draft of the visible entries, hidden entries left out', () => {
    expect(catalogTargets('darwin').map((target) => [target.kind, target.slug, target.repo])).toEqual([
      ['build', 'qwen3.8-27b-gguf', 'unsloth/Qwen3.8-27B-GGUF'],
      ['build', 'qwen3.8-27b-gguf-ud-q4_k_m', 'unsloth/Qwen3.8-27B-GGUF'],
      ['draft', 'qwen3.8-27b-dflash2', 'z-lab/Qwen3.8-27B-DFlash2'],
      ['build', 'inkling-2-mini-gguf', 'unsloth/Inkling-2-Mini-GGUF'],
      ['build', 'inkling-2-mini-gguf-ud-q4_k_m', 'unsloth/Inkling-2-Mini-GGUF'],
      ['build', 'inkling-2-mini', 'thinkingmachines/Inkling-2-Mini'],
      ['draft', 'inkling-2-mini-dflash2', 'incoai/Inkling-2-Mini-DFlash2'],
      ['build', 'qwen3.8-flash-next-gguf', 'unsloth/Qwen3.8-Flash-Next-GGUF'],
      ['build', 'hy4-flash-gguf', 'AngelSlim/Hy4-flash-GGUF'],
      ['build', 'clairic-flash-gguf', 'Brooooooklyn/Clairic-Flash-GGUF'],
      ['build', 'clairic-gguf', 'Brooooooklyn/Clairic-GGUF'],
      ['build', 'clef-flash-gguf', 'ggml-org/Clef-Flash-GGUF'],
      ['build', 'clef-flash-gguf-q4_k_m', 'ggml-org/Clef-Flash-GGUF'],
      ['build', 'clef-gguf', 'ggml-org/Clef-GGUF'],
      ['build', 'clef-gguf-q4_k_m', 'ggml-org/Clef-GGUF'],
    ]);
    for (const entry of MODEL_CATALOG.filter((item) => item.hidden)) {
      for (const build of entry.builds)
        expect(catalogTargetForSlug(catalogBuildSlug(build)), entry.label).toBeUndefined();
    }
  });

  it("carries a build's globs and sidecar source", () => {
    const target = catalogTargetForSlug('qwen3.8-27b-gguf-ud-q4_k_m', 'darwin');
    expect(target?.kind).toBe('build');
    if (target?.kind !== 'build') return;
    expect(target.build).toBe(buildOf('Qwen3.8-27B', 'UD-Q4_K_M'));
    expect(target.globs).toEqual(['*UD-Q4_K_M*', 'config.json']);
    expect(target.assetsRepo).toBe('Qwen/Qwen3.8-27B');
    const original = catalogTargetForSlug('inkling-2-mini', 'darwin');
    expect(original?.kind === 'build' && original.globs === undefined && original.assetsRepo === undefined).toBe(true);
    expect(catalogTargetForSlug('someone-not-in-catalog')).toBeUndefined();
  });
});

describe('catalogUpdateRepos — update discovery covers the sidecar sources', () => {
  it('lists each target repo and each build assetsRepo once, and no hidden repo', () => {
    const updates = catalogUpdateRepos('darwin');
    expect(new Set(updates).size).toBe(updates.length);
    for (const target of catalogTargets('darwin')) expect(updates).toContain(target.repo);
    // The assets sources are the point: a sidecar-only upstream change moves
    // nothing in the primary repo, so without these it raises no badge and the
    // repair job is unreachable.
    for (const entry of visibleCatalog(BIG_MAC, 'darwin')) {
      for (const build of entry.builds) {
        if (build.assetsRepo !== undefined) expect(updates).toContain(build.assetsRepo);
      }
    }
    // Hidden entries stay out (their repos 401).
    for (const entry of MODEL_CATALOG.filter((item) => item.hidden)) {
      for (const build of entry.builds) expect(updates).not.toContain(build.hfRepo);
    }
  });

  it('omits darwin-only Hy4-flash from targets, slug allowlist, and update repos on non-darwin platforms', () => {
    expect(catalogTargetForSlug('hy4-flash-gguf', 'darwin')?.repo).toBe('AngelSlim/Hy4-flash-GGUF');
    for (const platform of ['win32', 'linux'] as const) {
      expect(catalogTargets(platform).map((t) => t.slug)).not.toContain('hy4-flash-gguf');
      expect(catalogTargetForSlug('hy4-flash-gguf', platform)).toBeUndefined();
      const updates = catalogUpdateRepos(platform);
      expect(updates).not.toContain('AngelSlim/Hy4-flash-GGUF');
      expect(updates).not.toContain('tencent/Hy4-flash');
      expect(updates).toContain('unsloth/Qwen3.8-27B-GGUF');
      expect(updates).toContain('unsloth/Qwen3.8-Flash-Next-GGUF');
    }
  });
});

describe('MODEL_CATALOG', () => {
  it('has exactly one default entry', () => {
    const defaults = MODEL_CATALOG.filter((entry) => entry.isDefault);
    expect(defaults).toHaveLength(1);
    expect(defaults[0]!.label).toBe('Qwen3.8-27B');
  });

  it('only Qwen3.8-Flash-Next and Hy4-flash carry a memory floor, 96 GiB each', () => {
    for (const entry of MODEL_CATALOG) {
      expect(entry.minMemoryGb, entry.label).toBe(FLOORED.includes(entry.label) ? 96 : undefined);
    }
    // Hy4-flash's load admission accepts 39 GiB; the catalog floor must never sit below it.
    expect(MODEL_CATALOG.find((entry) => entry.label === 'Hy4-flash')!.minMemoryGb).toBeGreaterThanOrEqual(39);
  });

  it('Qwen3.8-27B and Inkling-2-Mini carry auto-installed DFlash2 companions', () => {
    const qwen = MODEL_CATALOG.find((entry) => entry.label === 'Qwen3.8-27B');
    expect(qwen?.draft).toEqual({
      label: 'DFlash2',
      hfRepo: 'z-lab/Qwen3.8-27B-DFlash2',
      sizeGb: 3.85,
      autoInstall: true,
    });
    const inkling = MODEL_CATALOG.find((entry) => entry.label === 'Inkling-2-Mini');
    expect(inkling?.draft).toEqual({
      label: 'Inkling-2-Mini DFlash2',
      hfRepo: 'incoai/Inkling-2-Mini-DFlash2',
      sizeGb: 1.0,
      autoInstall: true,
    });
    for (const entry of MODEL_CATALOG) {
      if (entry.label === 'Qwen3.8-27B' || entry.label === 'Inkling-2-Mini') continue;
      expect(entry.draft, entry.label).toBeUndefined();
    }
  });

  it('names the GGUF family of an entry only for a family whose GGUF discovery needs no asset files', () => {
    expect(
      MODEL_CATALOG.flatMap((entry) => (entry.ggufFamily === undefined ? [] : [[entry.label, entry.ggufFamily]])),
    ).toEqual([
      ['Qwen3.8-27B', 'qwen3_5'],
      ['Inkling-2-Mini', 'inkling2'],
      ['Qwen3.8-Flash-Next', 'qwen4_exp'],
      ['Hy4-flash', 'hy_v4_flash'],
      ['CLEF Flash (9B)', 'clef'],
      ['CLEF (27B)', 'clef'],
    ]);
    for (const entry of MODEL_CATALOG) {
      if (entry.ggufFamily === undefined) continue;
      const discovery = familyDataFor(entry.ggufFamily)?.ggufDiscovery;
      expect(discovery, entry.label).toBeDefined();
      expect(discovery?.requiresAssets, entry.label).toBeUndefined();
      expect(
        entry.builds.some((build) => build.hfRepo.endsWith('-GGUF')),
        entry.label,
      ).toBe(true);
    }
  });

  it('labels are unique', () => {
    const labels = MODEL_CATALOG.map((entry) => entry.label);
    expect(new Set(labels).size).toBe(labels.length);
  });

  it('every size is positive and every description and source is non-empty', () => {
    for (const { entry, build } of ALL_BUILDS) {
      expect(build.sizeGb, entry.label).toBeGreaterThan(0);
      expect(build.source.trim().length, entry.label).toBeGreaterThan(0);
      expect(entry.description.trim().length, entry.label).toBeGreaterThan(0);
    }
  });
});

describe('visibleCatalog', () => {
  it('offers Qwen3.8-Flash-Next and Hy4-flash from 96 GiB on darwin, at exactly 96 GiB too', () => {
    for (const label of FLOORED) {
      const entry = MODEL_CATALOG.find((candidate) => candidate.label === label)!;
      expect(visibleCatalog(SMALL_MAC, 'darwin'), label).not.toContain(entry);
      expect(visibleCatalog(64 * GIB, 'darwin'), label).not.toContain(entry);
      expect(visibleCatalog(MAC_96 - 1, 'darwin'), label).not.toContain(entry);
      expect(visibleCatalog(MAC_96, 'darwin'), label).toContain(entry);
      expect(visibleCatalog(BIG_MAC, 'darwin'), label).toContain(entry);
    }
    // Entries without a floor stay on the smallest machine.
    const small = visibleCatalog(SMALL_MAC, 'darwin');
    for (const entry of MODEL_CATALOG.filter(
      (item) => !item.hidden && (item.category ?? 'chat') === 'chat' && item.minMemoryGb === undefined,
    )) {
      expect(small, entry.label).toContain(entry);
    }
  });

  it('gates Hy4-flash to darwin and leaves cross-platform entries visible on win32 and linux', () => {
    const hy4 = MODEL_CATALOG.find((entry) => entry.label === 'Hy4-flash')!;
    expect(hy4.platforms).toEqual(['darwin']);
    expect(catalogSupportedOnPlatform(hy4, 'darwin')).toBe(true);
    expect(catalogSupportedOnPlatform(hy4, 'win32')).toBe(false);
    expect(catalogSupportedOnPlatform(hy4, 'linux')).toBe(false);
    expect(catalogSupportedOnPlatform({}, 'win32')).toBe(true);

    for (const platform of ['win32', 'linux'] as const) {
      const labels = visibleCatalog(BIG_MAC, platform).map((entry) => entry.label);
      expect(labels, platform).toEqual(['Qwen3.8-27B', 'Inkling-2-Mini', 'Qwen3.8-Flash-Next']);
      expect(visibleSystemOneCatalog(BIG_MAC, platform).length).toBeGreaterThan(0);
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
    expect(
      sysOne.map((entry) => [
        entry.label,
        entry.builds.map((build) => [build.id, build.hfRepo, build.assetsRepo, build.sizeGb]),
        entry.category,
      ]),
    ).toEqual([
      [
        'Clairic Flash (9B)',
        [['UD-Q4_K_XL', 'Brooooooklyn/Clairic-Flash-GGUF', 'Cloudflare/clairic-flash', 7.6]],
        'systemone',
      ],
      ['Clairic (27B)', [['UD-Q4_K_XL', 'Brooooooklyn/Clairic-GGUF', 'Cloudflare/clairic', 18.9]], 'systemone'],
      [
        'CLEF Flash (9B)',
        [
          ['Q8_0', 'ggml-org/Clef-Flash-GGUF', undefined, 9.7],
          ['Q4_K_M', 'ggml-org/Clef-Flash-GGUF', undefined, 6.7],
        ],
        'systemone',
      ],
      [
        'CLEF (27B)',
        [
          ['Q8_0', 'ggml-org/Clef-GGUF', undefined, 28.7],
          ['Q4_K_M', 'ggml-org/Clef-GGUF', undefined, 19.5],
        ],
        'systemone',
      ],
    ]);
    const chatRepos = new Set(visibleCatalog().flatMap((entry) => entry.builds.map((build) => build.hfRepo)));
    for (const entry of sysOne) expect(chatRepos.has(entry.builds[0].hfRepo)).toBe(false);
    // The wizard offers no System One entry on any machine.
    for (const memory of [SMALL_MAC, BIG_MAC]) {
      expect(visibleCatalog(memory).filter((entry) => sysOne.includes(entry))).toEqual([]);
    }
  });

  it('includes System One repos in the download targets and catalogUpdateRepos', () => {
    const updates = catalogUpdateRepos();
    for (const entry of visibleSystemOneCatalog()) {
      const build = entry.builds[0];
      expect(catalogTargetForSlug(catalogBuildSlug(build))?.repo).toBe(build.hfRepo);
      expect(updates).toContain(build.hfRepo);
    }
  });
});
