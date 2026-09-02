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
  /** HF slug for `mlx download model` on Apple Silicon — the MXFP4 build. */
  hfRepo: string;
  /**
   * HF slug on Linux + NVIDIA CUDA — the NVFP4 build.
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
 * The repo THIS platform installs for `entry`. Linux is the CUDA preview
 * target (README "Platform Support"); everything else is Apple Silicon.
 *
 * NOT because Metal cannot run NVFP4 — it can. MLX ships the same 234
 * quantized Metal kernels for `nvfp4` as for `mxfp4`, NAX variants included,
 * and only `fp8_e4m3` reconstructs BF16 at load. The MSL 4.1 hardware
 * block-scale format (`metal_fp8_ue8m0_format`) appears nowhere in MLX's Metal
 * backend, so it constrains neither format here.
 *
 * The split is:
 * - CUDA takes NVFP4 through `CublasQQMM` (`nvfp4` -> `CUDA_R_4F_E2M1`), a
 *   native path with no MXFP4 equivalent.
 * - Metal has no such native-path advantage either way, so prefer the format
 *   that survives quantization better. NVFP4 stores a block scale as `amax/6`
 *   in E4M3, and real FFN blocks land in its subnormal band;
 *   `apply_nvfp4_pow2_lift` repairs that for dense SwiGLU FFNs but SKIPS MoE
 *   experts by design (`NVFP4_LIFT_MOE_MARKERS`, `crates/mlx-core/src/
 *   convert.rs`), because the norm there also drives the router and the
 *   shared-expert gate, neither scale-invariant. Two of the three visible
 *   entries are MoE. MXFP4's E8M0 block scales have no such failure.
 *
 * Unmeasured: whether nvfp4 or mxfp4 decodes faster on Metal. The choice above
 * is made on quantization quality, not throughput.
 *
 * Every consumer that turns a catalog entry into a download, a slug, or an
 * allowlist check must go through here. Reading `entry.hfRepo` directly
 * installs the macOS build on a CUDA box.
 */
export function catalogRepo(entry: CatalogEntry): string {
  return catalogRepoFor(entry, process.platform);
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
