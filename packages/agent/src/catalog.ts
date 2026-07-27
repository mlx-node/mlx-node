/**
 * Curated model catalog for `mlx agent`.
 *
 * The first-run download wizard (Task 8) offers `visibleCatalog()` and
 * feeds the chosen `hfRepo` to `mlx download model`. Slugs are settled
 * with the user and verified against the Brooooooklyn HF account —
 * use them verbatim.
 */

/**
 * Content identity of a published checkpoint — its `config.json` `model_type`
 * plus quant block (`bits`/`mode`/`group_size`). The dashboard matches this
 * against local checkpoints so a recommended model is recognized as installed
 * even under a DIFFERENT folder name (e.g. an upload whose HF repo was renamed,
 * or a re-quantized copy). Matching also requires the local folder name to start
 * with the entry {@link CatalogEntry.label}, so two same-architecture, same-quant
 * models (AgentWorld-35B-A3B vs Qwen3.6-35B-A3B — both `qwen3_5_moe` nvfp4) never
 * cross-match on fingerprint alone.
 */
export interface ModelFingerprint {
  /** Canonical `model_type` family label (`qwen3_5`, `qwen3_5_moe`, `gemma4`). */
  modelType: string;
  /** Quantization block: bit width, mode (`nvfp4`, `mxfp4`, `affine`), group size. */
  quant: { bits: number; mode: string; group: number };
}

export interface CatalogEntry {
  /** Wizard display name. Also the base-model folder-name prefix used for matching. */
  label: string;
  /** HF slug for `mlx download model`. */
  hfRepo: string;
  /** Approximate download size in GB, for display. */
  sizeGb: number;
  /** One line for the wizard. */
  description: string;
  /** Exactly one entry carries this. */
  isDefault?: boolean;
  /** Not offered by the wizard (repo not yet published). */
  hidden?: boolean;
  /**
   * Published checkpoint identity, for recognizing this model on disk under a
   * non-canonical folder name. Absent for a hidden/unpublished entry whose
   * `config.json` is unknown offline (such an entry matches by exact slug only).
   */
  fingerprint?: ModelFingerprint;
}

export const MODEL_CATALOG: readonly CatalogEntry[] = [
  {
    label: 'Qwen3.6-27B',
    hfRepo: 'Brooooooklyn/Qwen3.6-27B-NVFP4-mlx',
    sizeGb: 22.2,
    description: 'Best tool use — recommended default',
    isDefault: true,
    fingerprint: { modelType: 'qwen3_5', quant: { bits: 4, mode: 'nvfp4', group: 16 } },
  },
  {
    label: 'Qwen-AgentWorld-35B',
    hfRepo: 'Brooooooklyn/Qwen-AgentWorld-35B-A3B-nvfp4-mlx',
    sizeGb: 22.7,
    description: 'Agent-tuned MoE, fast decode',
    fingerprint: { modelType: 'qwen3_5_moe', quant: { bits: 4, mode: 'nvfp4', group: 16 } },
  },
  {
    label: 'Gemma-4-26B-A4B',
    hfRepo: 'Brooooooklyn/Gemma-4-26B-A4B-NVFP4-mlx',
    sizeGb: 18.8,
    description: 'MoE, fast decode',
    fingerprint: { modelType: 'gemma4', quant: { bits: 4, mode: 'nvfp4', group: 16 } },
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
];

/** Catalog entries the wizard offers (hidden entries filtered out). */
export function visibleCatalog(): CatalogEntry[] {
  return MODEL_CATALOG.filter((entry) => !entry.hidden);
}

/**
 * Cold-tier facts, re-exported through this subpath.
 *
 * `@mlx-node/agent/catalog` is the agent package's one NATIVE-FREE entry point:
 * the package root re-exports `provider/index.ts`, which value-imports
 * `@mlx-node/core`. The dashboard is a separate viewer process that must never
 * link the addon (docs/dashboard.md: "no Metal init, instant start"), and
 * `mlx agent --help` must print without loading weights — so both reach the
 * cold-tier allowlist and the cache-root canonicalizer through here.
 *
 * These are RE-EXPORTS. The definitions live in `./cold-tier.ts` and there is
 * exactly one of each; `packages/agent/__test__/cold-tier-families.test.ts`
 * guards the allowlist against the native side.
 */
export { COLD_TIER_RESTORE_FAMILIES, canonicalCacheRoot, coldTierRestoreFamilyList } from './cold-tier.js';
