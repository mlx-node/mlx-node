/**
 * Curated model catalog for `mlx agent`.
 *
 * The first-run download wizard offers `visibleCatalog()` and feeds the chosen
 * entry's `catalogRepo()` to `mlx download model`. Slugs are verified against
 * Hugging Face — use them verbatim.
 */

import { QWEN38_DFLASH2 } from '@mlx-node/lm/draft-companion';

export interface CatalogEntry {
  /** Wizard display name. */
  label: string;
  /** HF slug for `mlx download model` — the UD-Q4_K_XL GGUF, every platform. */
  hfRepo: string;
  /**
   * HF slug on Linux + NVIDIA CUDA, when a platform-specific build exists.
   *
   * Absent means {@link hfRepo} serves both — the case for every current
   * entry: all platforms install the Unsloth UD quant from one GGUF repo.
   * Retained for future platform-specific builds; {@link globs} and
   * {@link assetsRepo} describe {@link hfRepo} only and never apply to an
   * override repo. Resolve with {@link catalogRepo}, never by reading the
   * field.
   */
  hfRepoCuda?: string;
  /**
   * Download file filter for multi-variant repos (GGUF), as simple `*`-wildcard
   * globs. A file is downloaded when its basename OR its full repo path matches
   * any glob; core metadata files (config/tokenizer) present in the repo are
   * always included. Mirrors the `mlx download model --glob` semantics so the
   * wizard, the dashboard downloader, and a manual CLI run select the same
   * file set out of a repo that ships dozens of quantization variants.
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
  /** Optional companion, downloaded separately and never offered as a chat model. */
  draft?: { label: string; hfRepo: string; sizeGb: number };
  /** One line for the wizard. */
  description: string;
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
   * Also honoured by the download allowlist: `DownloadManager.start`
   * (`packages/dashboard/src/download.ts`) refuses a hidden repo up front
   * rather than allocating a job that fails mid-download with a 401 from
   * Hugging Face. No UI reaches that path for a hidden entry, but a direct
   * API call does.
   */
  hidden?: boolean;
}

export const MODEL_CATALOG: readonly CatalogEntry[] = [
  {
    // The UD-Q4_K_XL file carries its MTP layer INLINE (block 64 of 65,
    // `blk.64.nextn.*`), which the native prepare converts to `mtp.*` in the
    // cache — hasMtpWeights() is true from the primary file alone (validated:
    // ~4.8 tokens accepted per cycle). The repo's separate
    // `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` is deliberately NOT globbed: nothing
    // in the runtime pairs a GGUF MTP sidecar (`mtp_sidecar_candidates` is
    // safetensors-only), so it would be 1.28 GB of dead weight per install.
    // The optional z-lab DFlash2 draft stays offered beside the target for
    // checkpoints that pair with it; it is never auto-installed.
    label: 'Qwen3.8-27B',
    hfRepo: 'unsloth/Qwen3.8-27B-GGUF',
    globs: ['*UD-Q4_K_XL*', 'config.json'],
    assetsRepo: 'Qwen/Qwen3.8-27B',
    sizeGb: 17.6,
    description: 'Best tool use — recommended default',
    isDefault: true,
    draft: QWEN38_DFLASH2,
  },
  {
    label: 'Qwen-AgentWorld-35B-A3B',
    hfRepo: 'unsloth/Qwen-AgentWorld-35B-A3B-GGUF',
    globs: ['*UD-Q4_K_XL*'],
    assetsRepo: 'Qwen/Qwen-AgentWorld-35B-A3B',
    sizeGb: 22.3,
    description: 'Agent-tuned MoE, fast decode',
  },
  {
    // `mmproj-BF16.gguf` is the SigLIP vision tower — the gemma4v projector
    // importer converts it to `vision.safetensors` during native prepare, so
    // this entry supports images. The repo's `mtp-*.gguf` files are
    // deliberately NOT globbed — mlx-node's Gemma speculative decode is the
    // DSpark draft path, nothing pairs those llama.cpp MTP artifacts, and
    // they would add ~2.5 GB of dead weight per install.
    label: 'Gemma-4-26B-A4B',
    hfRepo: 'unsloth/gemma-4-26B-A4B-it-GGUF',
    globs: ['*UD-Q4_K_XL*', 'mmproj-BF16.gguf', 'config.json'],
    // The license-gated google/gemma-4-26B-A4B-it would 401 for most users;
    // unsloth's mirror ships the same tokenizer/config files ungated.
    assetsRepo: 'unsloth/gemma-4-26B-A4B-it',
    sizeGb: 18.2,
    description: 'MoE, fast decode',
  },
  {
    // Produced + validated locally as mxfp4 (MLP) + mxfp8 (attention) via
    // `mlx convert --q-recipe nvidia` on gemma-4-12b-it (coherent + tool
    // calling through `mlx agent`). Provisional slug — the user finalizes it
    // on HF upload; entry stays hidden until the repo exists.
    label: 'Gemma-4-12B',
    hfRepo: 'Brooooooklyn/Gemma-4-12B-IT-mxfp-mlx',
    sizeGb: 8.6,
    description: 'Compact (mxfp4 MLP + mxfp8 attention), fits smaller machines',
    hidden: true,
  },
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
    // catalog state, and `catalogSlug()` recognize a locally converted
    // checkpoint sitting at the canonical slug.
    label: 'Nemotron-3.5-Lightning-30B-A3B',
    hfRepo: 'Brooooooklyn/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-nvfp4-mlx',
    sizeGb: 23,
    description: 'Hybrid Mamba-2 + MoE, native MTP, 1M context',
    hidden: true,
  },
];

/**
 * Every repo whose upstream revision participates in update discovery: the
 * visible target repos and their companions, plus the `assetsRepo` each entry
 * sources its tokenizer sidecars from.
 *
 * The download allowlist ({@link catalogDownloadRepos}) is deliberately
 * narrower — sidecars are fetched INSIDE a job, never started as one — but a
 * tokenizer or chat-template fix in the base model is an update a user must be
 * able to see. Without these, a sidecar-only upstream change raises no badge,
 * the update endpoint never reports it, and the UI's Installed button stays
 * disabled: the repair path is unreachable by design.
 */
export function catalogUpdateRepos(): string[] {
  const repos = new Set(catalogDownloadRepos());
  for (const entry of MODEL_CATALOG) {
    if (!entry.hidden && entry.assetsRepo !== undefined) repos.add(entry.assetsRepo);
  }
  return [...repos];
}

/** Visible target repos plus their optional companion downloads. */
export function catalogDownloadRepos(): string[] {
  return [
    ...new Set(
      MODEL_CATALOG.filter((entry) => !entry.hidden).flatMap((entry) =>
        entry.draft === undefined ? [catalogRepo(entry)] : [catalogRepo(entry), entry.draft.hfRepo],
      ),
    ),
  ];
}

/**
 * The repo THIS platform installs for `entry`.
 *
 * All platforms take the Unsloth UD-Q4_K_XL GGUF: Unsloth Dynamic quants
 * keep embeddings and `lm_head` at high precision, which measured as the best
 * output quality AND 1.44× decode over MXFP4 on Qwen3.8-27B with MTP — with
 * smaller downloads on top — and ggml K-quants are losslessly repacked into
 * MLX layout at first load, so the GGUF is as native as a pre-converted
 * safetensors repo once cached. GGUF quantization repos ship no tokenizer
 * files, so each entry pairs its repo with {@link CatalogEntry.assetsRepo}
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
 * Every consumer that turns a catalog entry into a download, a slug, or an
 * allowlist check must go through here. Reading `entry.hfRepo` directly
 * installs the macOS build on a CUDA box the day a CUDA build returns.
 */
export function catalogRepo(entry: CatalogEntry): string {
  return catalogRepoFor(entry, process.platform);
}

/**
 * The file-selection fields that apply to `repo` when it is `entry`'s GGUF
 * repo: {@link CatalogEntry.globs} and {@link CatalogEntry.assetsRepo}.
 *
 * Both describe the multi-variant GGUF artifact only. A platform-specific
 * override repo (the pre-converted CUDA build) is a plain safetensors layout:
 * globbing it would filter out every weight, and its config/tokenizer files
 * are its own. Consumers must resolve selection through here rather than
 * reading the entry fields directly.
 */
export function catalogSelectionForRepo(
  entry: CatalogEntry,
  repo: string,
): { globs?: readonly string[]; assetsRepo?: string } {
  if (repo !== entry.hfRepo) return {};
  return { globs: entry.globs, assetsRepo: entry.assetsRepo };
}

/**
 * {@link catalogRepo} with the platform passed in — the pure half.
 *
 * Exists so both branches can be asserted without touching `process.platform`.
 * Mutating that global leaks across test files sharing a worker: it made
 * `catalogRepo` disagree with a sibling suite's module-level constant and fail
 * a download allowlist check that has nothing to do with the catalog.
 */
export function catalogRepoFor(entry: CatalogEntry, platform: NodeJS.Platform): string {
  return platform === 'linux' && entry.hfRepoCuda !== undefined ? entry.hfRepoCuda : entry.hfRepo;
}

/**
 * The catalog entry whose THIS-platform download repo is `repo`, if any.
 *
 * The dashboard downloader looks entries up by the repo a job was started
 * with (already allowlisted through `catalogDownloadRepos`, so a match is
 * guaranteed in practice) to find the entry's `globs` file filter and
 * `assetsRepo` sidecar source. Matching goes through
 * {@link catalogRepoFor} — never a raw `hfRepo` comparison — so a future
 * platform-split entry resolves against the repo this platform actually
 * installs.
 */
export function catalogEntryForRepo(repo: string, platform: NodeJS.Platform): CatalogEntry | undefined {
  return MODEL_CATALOG.find((entry) => catalogRepoFor(entry, platform) === repo);
}

/** Catalog entries the wizard offers (hidden entries filtered out). */
export function visibleCatalog(): CatalogEntry[] {
  return MODEL_CATALOG.filter((entry) => !entry.hidden);
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
