/**
 * Curated model catalog for `mlx agent`.
 *
 * The first-run download wizard offers `visibleCatalog()` and feeds the chosen
 * build's `catalogBuildRepo()` to `mlx download model`. Slugs are verified
 * against Hugging Face — use them verbatim.
 */

import { totalmem } from 'node:os';

import { QWEN38_DFLASH2 } from '@mlx-node/lm/draft-companion';

const GIB = 2 ** 30;

/** One downloadable build of a catalog model: a repository, the files it takes, and the folder it installs to. */
export interface CatalogBuild {
  /**
   * The build's name on every surface (`UD-Q4_K_XL`, `Original`), and the quant tag discovery reads from its weight
   * file name. Unique in its entry.
   */
  id: string;
  /** HF slug for `mlx download model`, every platform. */
  hfRepo: string;
  /**
   * HF slug on Linux + NVIDIA CUDA, when a platform-specific build exists.
   *
   * Absent means {@link hfRepo} serves both — the case for every current
   * build. {@link globs} and {@link assetsRepo} describe {@link hfRepo} only
   * and never apply to an override repo. Resolve with
   * {@link catalogBuildRepo}, never by reading the field.
   */
  hfRepoCuda?: string;
  /**
   * Download file filter for multi-variant repos (GGUF), as simple `*`-wildcard
   * globs. A file is downloaded when its basename OR its full repo path matches
   * any glob; core metadata files (config/tokenizer) present in the repo are
   * always included. Mirrors the `mlx download model --glob` semantics so the
   * wizard, the dashboard downloader, and a manual CLI run select the same
   * file set out of a repo that ships dozens of quantization variants. Absent
   * means the downloader's default filter.
   *
   * Applies to {@link hfRepo} only — the pre-converted CUDA repo has no
   * quantization variants to filter.
   */
  globs?: string[];
  /**
   * Base-model HF repo supplying the tokenizer/config/processor sidecar files
   * a GGUF repo lacks (GGUF quantization repos ship weights, no tokenizer).
   * Applies to {@link hfRepo} only.
   *
   * Mandatory for correct tool calling, not a nicety: when no sidecar
   * `tokenizer.json` sits next to the `.gguf`, the native GGUF runtime
   * extracts the embedded tokenizer, and that extraction marks
   * `<tool_call>`/`</tool_call>` `special: true` (the HF files mark them
   * false) — every decode path then skips special tokens, the tool-call
   * wrapper is stripped, and tool calling silently breaks. With the official
   * sidecar files beside the `.gguf`, the runtime copies them into its native
   * cache (the `GGUF_RUNTIME_ASSET_FILES` copy in
   * `crates/mlx-core/src/utils/gguf.rs`) and skips the embedded extraction.
   */
  assetsRepo?: string;
  /** Approximate download size in GB, for display. */
  sizeGb: number;
  /** Who made the files and in which format, for the build line: `Unsloth GGUF`, `Thinking Machines`, `MLX`. */
  source: string;
  /** One short fact after the source: `1.1 GB smaller`, `NVFP4 + MXFP8`. */
  note?: string;
  /** What comes with the weights, for the extras line. */
  included?: string;
  /**
   * Only on a build whose repository an earlier build of the catalog already installs: the tail of its folder name
   * ({@link catalogBuildSlug}). Written here, never derived from the order of builds, so moving a build inside its
   * entry never moves a folder.
   */
  folderSuffix?: string;
}

export interface CatalogEntry {
  /** Display name; unique, and the key of an entry in the dashboard and the wizard. */
  label: string;
  /** One line for the wizard. */
  description: string;
  /** Which product surface this catalog entry belongs to (default `'chat'`). */
  category?: 'chat' | 'systemone';
  /**
   * Physical RAM in GiB below which no UI offers the entry: not the wizard
   * ({@link visibleCatalog}), and not the dashboard Models page or Onboarding
   * (`offeredOnHost` in `packages/dashboard/ui/src/lib/catalog.ts`). Never
   * lower than the RAM at which the model's load admission always refuses.
   */
  minMemoryGb?: number;
  /** Exactly one entry carries this. */
  isDefault?: boolean;
  /**
   * The repo is not published yet, so no UI may offer it as a download.
   *
   * Two consumers honour this: the agent wizard via {@link visibleCatalog},
   * and the dashboard Models page, which filters `!item.hidden` before
   * rendering cards (`packages/dashboard/ui/src/pages/models.tsx`).
   * `catalogWithState()` deliberately keeps hidden entries so the UI, not the
   * dashboard core, decides.
   *
   * Also honoured by the download allowlist: {@link catalogTargets} leaves a
   * hidden entry out, so `DownloadManager.start`
   * (`packages/dashboard/src/download.ts`) refuses its folder up front rather
   * than allocating a job that fails mid-download with a 401 from Hugging
   * Face. No UI reaches that path for a hidden entry, but a direct API call
   * does.
   */
  hidden?: boolean;
  /** `builds[0]` is the build every surface chooses first. */
  builds: readonly [CatalogBuild, ...CatalogBuild[]];
  /** Optional companion, downloaded separately and never offered as a chat model. */
  draft?: { label: string; hfRepo: string; sizeGb: number };
}

/** The files Inkling-2-Mini's GGUF builds take beside the weight: the vision tower and the root DSpark draft. */
const INKLING_COMPANIONS = ['mmproj-BF16.gguf', 'dspark-Inkling-2-Mini-Q8_0.gguf'];

/** The files a Clairic build takes beside its weight: vision tower, Clair head and its image processor. */
const CLAIRIC_SIDECARS = [
  'mmproj-BF16.gguf',
  'clair_head.safetensors',
  'clair_head_config.json',
  'preprocessor_config.json',
];

export const MODEL_CATALOG: readonly CatalogEntry[] = [
  {
    label: 'Qwen3.8-27B',
    description: 'Best tool use — recommended default',
    isDefault: true,
    builds: [
      {
        // The UD-Q4_K_XL file carries its MTP layer INLINE (block 64 of 65,
        // `blk.64.nextn.*`), which the native prepare converts to `mtp.*` in the
        // cache — hasMtpWeights() is true from the primary file alone (validated:
        // ~4.8 tokens accepted per cycle). The repo's separate
        // `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` is deliberately NOT globbed: nothing
        // in the runtime pairs a GGUF MTP sidecar (`mtp_sidecar_candidates` is
        // safetensors-only), so it would be 1.28 GB of dead weight per install.
        id: 'UD-Q4_K_XL',
        hfRepo: 'unsloth/Qwen3.8-27B-GGUF',
        globs: ['*UD-Q4_K_XL*', 'config.json'],
        assetsRepo: 'Qwen/Qwen3.8-27B',
        sizeGb: 17.7,
        source: 'Unsloth GGUF',
      },
      {
        // The repo has no plain Q4_K_M file; this glob takes only
        // `Qwen3.8-27B-UD-Q4_K_M.gguf`.
        id: 'UD-Q4_K_M',
        hfRepo: 'unsloth/Qwen3.8-27B-GGUF',
        globs: ['*UD-Q4_K_M*', 'config.json'],
        assetsRepo: 'Qwen/Qwen3.8-27B',
        sizeGb: 16.6,
        source: 'Unsloth GGUF',
        note: '1.1 GB smaller',
        folderSuffix: 'ud-q4_k_m',
      },
    ],
    // The optional z-lab DFlash2 draft pairs with either build by name; it is
    // never auto-installed.
    draft: QWEN38_DFLASH2,
  },
  {
    label: 'Inkling-2-Mini',
    description: 'Looped MoE with vision and DSpark drafts, 1M context',
    builds: [
      {
        // The weight glob ends in `.gguf` because `*UD-Q4_K_XL*` also matches
        // `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file`, a second 17.6 GB
        // copy. The native loader pairs the vision tower (`mmproj-BF16.gguf`) and
        // the root DSpark draft from the model's folder; the drafter under
        // `dspark/` is never scanned, so it is not globbed.
        id: 'UD-Q4_K_XL',
        hfRepo: 'unsloth/Inkling-2-Mini-GGUF',
        globs: ['*UD-Q4_K_XL.gguf', ...INKLING_COMPANIONS],
        assetsRepo: 'thinkingmachines/Inkling-2-Mini',
        sizeGb: 18.7,
        source: 'Unsloth GGUF',
        included: 'Vision and DSpark draft included',
      },
      {
        // `.gguf` ending for the same reason: `*UD-Q4_K_M*` doubles the download
        // with the Mainline_Support copy.
        id: 'UD-Q4_K_M',
        hfRepo: 'unsloth/Inkling-2-Mini-GGUF',
        globs: ['*UD-Q4_K_M.gguf', ...INKLING_COMPANIONS],
        assetsRepo: 'thinkingmachines/Inkling-2-Mini',
        sizeGb: 17.7,
        source: 'Unsloth GGUF',
        note: '1 GB smaller',
        included: 'Vision and DSpark draft included',
        folderSuffix: 'ud-q4_k_m',
      },
      {
        // The original safetensors release: experts NVFP4 (layers 0-19) and
        // MXFP8 (layers 20-23), the rest BF16, vision in the same file and its
        // DSpark draft in `mtp.safetensors`. The default filter takes every
        // file the loader reads; it skips `tiktoken/tokenizer.model`, which the
        // loader does not read (it reads `tokenizer.json`).
        id: 'Original',
        hfRepo: 'thinkingmachines/Inkling-2-Mini',
        sizeGb: 20.2,
        source: 'Thinking Machines',
        note: 'NVFP4 + MXFP8',
        included: 'Vision and DSpark draft included',
      },
    ],
  },
  {
    label: 'Qwen3.8-Flash-Next',
    description: '125B MoE, 512 experts streamed from SSD',
    // The load has no fixed admission floor; the catalog offers it from 96 GiB.
    minMemoryGb: 96,
    builds: [
      {
        // The four shards sit in the repo folder `UD-Q4_K_XL/`, and downloads keep
        // repo paths, so they install to `<slug>/UD-Q4_K_XL/`. The routed experts
        // and the shard-4 n-gram table stream from SSD, so the weights may exceed
        // RAM. A GGUF load is text-only: the repo's `mmproj-*.gguf` and MTP GGUF
        // files are never paired, so they are not globbed.
        id: 'UD-Q4_K_XL',
        hfRepo: 'unsloth/Qwen3.8-Flash-Next-GGUF',
        globs: ['*UD-Q4_K_XL*'],
        assetsRepo: 'Qwen/Qwen3.8-Flash-Next',
        sizeGb: 111.3,
        source: 'Unsloth GGUF',
        included: 'Text only · streams from SSD',
      },
    ],
  },
  {
    label: 'Hy4-flash',
    description: '512-expert MoE streamed from SSD, 256K context',
    // The load admission accepts 39 GiB: its fixed bytes at its smallest
    // settings (16K context, 2048-row chunk, MTP on, one staging layer) are
    // 26.52 GiB, plus the 12 GiB reserve. The catalog offers it from 96 GiB.
    minMemoryGb: 96,
    builds: [
      {
        // Shard 2 is the Engram table, read by row from SSD; the routed experts
        // stream from SSD too, so the weights may exceed RAM. The loader reads the
        // embedded GGUF tokenizer, never the sidecar files.
        id: 'Q4_K_M',
        hfRepo: 'AngelSlim/Hy4-flash-GGUF',
        globs: ['*Q4_K_M*'],
        assetsRepo: 'tencent/Hy4-flash',
        sizeGb: 168.3,
        source: 'AngelSlim GGUF',
        included: 'Streams from SSD',
      },
    ],
  },
  {
    label: 'Gemma-4-12B',
    description: 'Compact (mxfp4 MLP + mxfp8 attention), fits smaller machines',
    hidden: true,
    builds: [
      {
        // Produced + validated locally as mxfp4 (MLP) + mxfp8 (attention) via
        // `mlx convert --q-recipe nvidia` on gemma-4-12b-it (coherent + tool
        // calling through `mlx agent`). Provisional slug — the user finalizes it
        // on HF upload; entry stays hidden until the repo exists.
        id: 'MXFP',
        hfRepo: 'Brooooooklyn/Gemma-4-12B-IT-mxfp-mlx',
        sizeGb: 8.6,
        source: 'MLX',
      },
    ],
  },
  {
    label: 'Nemotron-3.5-Lightning-30B-A3B',
    description: 'Hybrid Mamba-2 + MoE, native MTP, 1M context',
    hidden: true,
    builds: [
      {
        // NOT DOWNLOADABLE. This repo does not exist: the slug below is a
        // placeholder the user has not uploaded to, and Hugging Face answers 401
        // for it. Nothing may offer it as an install until the upload happens and
        // `hidden` is dropped — and do not guess a substitute slug, because a
        // wrong-but-live repo would install the wrong weights silently.
        //
        // The only route to this model today is local conversion from NVIDIA's
        // modelopt checkpoint:
        //   mlx convert -m nemotron_h \
        //     -i <nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4> \
        //     -o <models>/nemotron-3.5-lightning-30b-a3b-nvfp4-mlx
        // (NVFP4 preserved byte-for-byte; the FP8 Mamba-2 projections are
        // re-quantized. See docs/cli.md "modelopt NVFP4 ingest".)
        //
        // The entry is kept — rather than deleted — so the wizard, the dashboard
        // catalog state, and `catalogBuildSlug()` recognize a locally converted
        // checkpoint sitting at the canonical slug.
        id: 'NVFP4',
        hfRepo: 'Brooooooklyn/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-nvfp4-mlx',
        sizeGb: 23,
        source: 'MLX',
      },
    ],
  },
  {
    label: 'Clairic Flash (9B)',
    description: 'Fast 9B extraction, vision, PDF & forecasting',
    category: 'systemone',
    builds: [
      {
        id: 'UD-Q4_K_XL',
        hfRepo: 'Brooooooklyn/Clairic-Flash-GGUF',
        globs: ['*UD-Q4_K_XL*', ...CLAIRIC_SIDECARS],
        assetsRepo: 'Cloudflare/clairic-flash',
        sizeGb: 7.6,
        source: 'GGUF',
        included: 'Vision and Clair head included',
      },
    ],
  },
  {
    label: 'Clairic (27B)',
    description: 'High-accuracy 27B extraction, tables & PDFs',
    category: 'systemone',
    builds: [
      {
        id: 'UD-Q4_K_XL',
        hfRepo: 'Brooooooklyn/Clairic-GGUF',
        globs: ['*UD-Q4_K_XL*', ...CLAIRIC_SIDECARS],
        assetsRepo: 'Cloudflare/clairic',
        sizeGb: 18.9,
        source: 'GGUF',
        included: 'Vision and Clair head included',
      },
    ],
  },
];

/**
 * The repo THIS platform installs for `build`.
 *
 * The chat GGUF builds run on every platform: Unsloth Dynamic quants keep
 * embeddings and `lm_head` at high precision, which measured as the best
 * output quality AND 1.44× decode over MXFP4 on Qwen3.8-27B with MTP — with
 * smaller downloads on top — and ggml K-quants are losslessly repacked into
 * MLX layout at first load, so the GGUF is as native as a pre-converted
 * safetensors repo once cached. GGUF quantization repos ship no tokenizer
 * files, so each such build pairs its repo with {@link CatalogBuild.assetsRepo}
 * sidecars — required for correct tool calling (see that field's comment).
 *
 * CUDA caveat, measured not assumed: the native import repacks K-quants into
 * the `q4k`/`q5k`/`q6k` modes, and the CUDA backend's `QuantizedMatmul`,
 * `GatherQMM`, and dequantize currently throw
 * `"Quantization mode … is not implemented on the CUDA backend"`
 * (`reject_kquant`, `crates/mlx-sys/mlx/mlx/backend/cuda/quantized/
 * quantized.cpp`). Linux inference on these checkpoints therefore waits on
 * CUDA K-quant kernels; the DGX Spark PoC instead converted its GGUF with the
 * NVIDIA recipes (see README "CUDA preview"), which produce CUDA-executable
 * affine/NVFP4 layouts.
 *
 * Every consumer that turns a build into a download, a folder, or an
 * allowlist check must go through here. Reading `build.hfRepo` directly
 * installs the macOS build on a CUDA box the day a CUDA build returns. The
 * platform is a parameter so both branches can be asserted without touching
 * `process.platform`, a global shared with every test file on the worker.
 */
export function catalogBuildRepo(build: CatalogBuild, platform: NodeJS.Platform = process.platform): string {
  return platform === 'linux' && build.hfRepoCuda !== undefined ? build.hfRepoCuda : build.hfRepo;
}

/** The last part of `repo` (`owner/name`), in lower case. */
function repoFolderName(repo: string): string {
  const at = repo.lastIndexOf('/');
  return repo.slice(at + 1).toLowerCase();
}

/**
 * The folder `build` installs to under the models directory: the last part of THIS platform's repo in lower case,
 * then `-` and {@link CatalogBuild.folderSuffix} when the build has one. The first build of a repository keeps the
 * repository's folder, so an install made before a second build existed stays where it is.
 */
export function catalogBuildSlug(build: CatalogBuild, platform: NodeJS.Platform = process.platform): string {
  const folder = repoFolderName(catalogBuildRepo(build, platform));
  return build.folderSuffix === undefined ? folder : `${folder}-${build.folderSuffix}`;
}

/** The folder a companion draft installs to: its repository's name, in lower case. */
export function catalogDraftSlug(draft: NonNullable<CatalogEntry['draft']>): string {
  return repoFolderName(draft.hfRepo);
}

/** One thing the dashboard downloader may start a job for, keyed by the folder it installs to. */
export type CatalogTarget =
  | {
      kind: 'build';
      slug: string;
      entry: CatalogEntry;
      build: CatalogBuild;
      /** {@link catalogBuildRepo} of the build. */
      repo: string;
      /** The build's globs, only when {@link repo} is the build's GGUF repo. */
      globs?: readonly string[];
      /** The build's sidecar source, only when {@link repo} is the build's GGUF repo. */
      assetsRepo?: string;
    }
  | { kind: 'draft'; slug: string; entry: CatalogEntry; repo: string };

/**
 * Every build and every draft of the entries that are not hidden, in catalog order. `globs` and `assetsRepo`
 * describe the GGUF repo only, so a platform override repo (a plain safetensors layout) gets neither: globbing it
 * would filter out every weight, and its config/tokenizer files are its own.
 */
export function catalogTargets(platform: NodeJS.Platform = process.platform): CatalogTarget[] {
  const targets: CatalogTarget[] = [];
  for (const entry of MODEL_CATALOG) {
    if (entry.hidden) continue;
    for (const build of entry.builds) {
      const repo = catalogBuildRepo(build, platform);
      const own = repo === build.hfRepo;
      targets.push({
        kind: 'build',
        slug: catalogBuildSlug(build, platform),
        entry,
        build,
        repo,
        ...(own && build.globs !== undefined ? { globs: build.globs } : {}),
        ...(own && build.assetsRepo !== undefined ? { assetsRepo: build.assetsRepo } : {}),
      });
    }
    if (entry.draft !== undefined) {
      targets.push({ kind: 'draft', slug: catalogDraftSlug(entry.draft), entry, repo: entry.draft.hfRepo });
    }
  }
  return targets;
}

/** The target that installs to the folder `slug`: the download allowlist and the lookup of a job. */
export function catalogTargetForSlug(
  slug: string,
  platform: NodeJS.Platform = process.platform,
): CatalogTarget | undefined {
  return catalogTargets(platform).find((target) => target.slug === slug);
}

/**
 * Whether another build of the catalog, hidden entries included, installs the same repository as `build` on this
 * platform. Such a build's folder must name the build in its weight file to count as that build.
 */
export function catalogSharesRepo(build: CatalogBuild, platform: NodeJS.Platform = process.platform): boolean {
  const repo = catalogBuildRepo(build, platform);
  return MODEL_CATALOG.some((entry) =>
    entry.builds.some((other) => other !== build && catalogBuildRepo(other, platform) === repo),
  );
}

/**
 * Every repo whose upstream revision participates in update discovery: the
 * repos of {@link catalogTargets}, plus the `assetsRepo` each build sources
 * its tokenizer sidecars from.
 *
 * The download allowlist is deliberately narrower — sidecars are fetched
 * INSIDE a job, never started as one — but a tokenizer or chat-template fix
 * in the base model is an update a user must be able to see. Without these, a
 * sidecar-only upstream change raises no badge, the update endpoint never
 * reports it, and the UI's Installed button stays disabled: the repair path is
 * unreachable by design.
 */
export function catalogUpdateRepos(platform: NodeJS.Platform = process.platform): string[] {
  const repos = new Set<string>();
  for (const target of catalogTargets(platform)) {
    repos.add(target.repo);
    if (target.kind === 'build' && target.assetsRepo !== undefined) repos.add(target.assetsRepo);
  }
  return [...repos];
}

/** Whether the catalog offers `entry` on a machine with `totalMemBytes` of physical RAM ({@link CatalogEntry.minMemoryGb}). */
export function catalogFitsMemory(entry: Pick<CatalogEntry, 'minMemoryGb'>, totalMemBytes: number): boolean {
  return entry.minMemoryGb === undefined || totalMemBytes >= entry.minMemoryGb * GIB;
}

/** Catalog entries the wizard offers: visible chat entries whose memory floor `totalMemBytes` of RAM meets. */
export function visibleCatalog(totalMemBytes: number = totalmem()): CatalogEntry[] {
  return MODEL_CATALOG.filter(
    (entry) => !entry.hidden && (entry.category ?? 'chat') === 'chat' && catalogFitsMemory(entry, totalMemBytes),
  );
}

/** Visible System One (Clairic / CLEF) catalog entries whose memory floor `totalMemBytes` of RAM meets. */
export function visibleSystemOneCatalog(totalMemBytes: number = totalmem()): CatalogEntry[] {
  return MODEL_CATALOG.filter(
    (entry) => !entry.hidden && entry.category === 'systemone' && catalogFitsMemory(entry, totalMemBytes),
  );
}

/**
 * Cold-tier facts and family registration data, re-exported through this
 * subpath.
 *
 * `@mlx-node/agent/catalog` is a NATIVE-FREE entry point, alongside the
 * `delegate` client and `models` discovery subpaths:
 * the package root re-exports `provider/index.ts`, which value-imports
 * `@mlx-node/core`. The dashboard is a separate viewer process that must never
 * link the addon (docs/dashboard.md: "no Metal init, instant start"), and
 * `mlx agent --help` must print without loading weights — so both reach the
 * cold-tier allowlist, the cache-root canonicalizer, and the family detection
 * data through here.
 *
 * Every module reachable from here must therefore stay free of runtime addon
 * imports. `packages/agent/__test__/catalog-native-free.test.ts` gates that in a
 * real subprocess.
 */
export { COLD_TIER_RESTORE_FAMILIES, canonicalCacheRoot, coldTierRestoreFamilyList } from './cold-tier.js';
export {
  CHAT_FAMILY_IDS,
  matchFamily,
  NON_GENERATIVE_FAMILY_IDS,
  rawModelTypeToCanonical,
} from '@mlx-node/lm/family-data';
