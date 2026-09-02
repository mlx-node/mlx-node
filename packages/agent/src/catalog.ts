/**
 * Curated model catalog for `mlx agent`.
 *
 * The first-run download wizard offers `visibleCatalog()` and feeds the chosen
 * entry's `catalogRepo()` to `mlx download model`. Slugs are verified against the Brooooooklyn
 * HF account — use them verbatim.
 */

export interface CatalogEntry {
  /** Wizard display name. */
  label: string;
  /**
   * HF slug for `mlx download model` on Apple Silicon — the MXFP4 build.
   *
   * Metal's block scaling is MXFP4-shaped: the scale plane must be
   * `metal_fp8_ue8m0_format` at block 32 (MSL Spec 4.1 Table 2.28). NVFP4's
   * E4M3 scales at block 16 cannot be declared there, so it decodes through a
   * dequant path instead of the native one.
   */
  hfRepo: string;
  /**
   * HF slug on Linux + NVIDIA CUDA — the NVFP4 build, which is what
   * `CublasQQMM` consumes natively (`nvfp4` -> `CUDA_R_4F_E2M1`).
   *
   * Absent means the entry has no CUDA-specific build and {@link hfRepo}
   * serves both. Resolve with {@link catalogRepo}, never by reading the field.
   */
  hfRepoCuda?: string;
  /** Approximate download size in GB, for display. */
  sizeGb: number;
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
    label: 'Qwen3.8-27B',
    hfRepo: 'Brooooooklyn/Qwen3.8-27B-MXFP4-mlx',
    hfRepoCuda: 'Brooooooklyn/Qwen3.8-27B-NVFP4-mlx',
    sizeGb: 23.3,
    description: 'Best tool use — recommended default',
    isDefault: true,
  },
  {
    label: 'Qwen-AgentWorld-35B',
    hfRepo: 'Brooooooklyn/Qwen-AgentWorld-35B-A3B-mxfp4-mlx',
    hfRepoCuda: 'Brooooooklyn/Qwen-AgentWorld-35B-A3B-nvfp4-mlx',
    sizeGb: 23.3,
    description: 'Agent-tuned MoE, fast decode',
  },
  {
    label: 'Gemma-4-26B-A4B',
    hfRepo: 'Brooooooklyn/Gemma-4-26B-A4B-Unsloth-MXFP4-mlx',
    hfRepoCuda: 'Brooooooklyn/Gemma-4-26B-A4B-Unsloth-NVFP4-mlx',
    sizeGb: 16.2,
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
 * The repo THIS platform installs for `entry`.
 *
 * The quantization format is not a preference, it is a backend fact: Metal
 * expresses MXFP4 natively and cannot express NVFP4 at all, while CUDA's
 * `CublasQQMM` takes NVFP4 directly. Linux is the CUDA preview target
 * (README "Platform Support"); everything else is Apple Silicon.
 *
 * Every consumer that turns a catalog entry into a download, a slug, or an
 * allowlist check must go through here. Reading `entry.hfRepo` directly
 * installs the macOS build on a CUDA box.
 */
export function catalogRepo(entry: CatalogEntry): string {
  return process.platform === 'linux' && entry.hfRepoCuda !== undefined ? entry.hfRepoCuda : entry.hfRepo;
}

/** Catalog entries the wizard offers (hidden entries filtered out). */
export function visibleCatalog(): CatalogEntry[] {
  return MODEL_CATALOG.filter((entry) => !entry.hidden);
}

/**
 * Cold-tier facts and family registration data, re-exported through this
 * subpath.
 *
 * `@mlx-node/agent/catalog` is the agent package's one NATIVE-FREE entry point:
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
