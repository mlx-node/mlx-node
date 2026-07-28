/**
 * Curated model catalog for `mlx agent`.
 *
 * The first-run download wizard (Task 8) offers `visibleCatalog()` and
 * feeds the chosen `hfRepo` to `mlx download model`. Slugs are settled
 * with the user and verified against the Brooooooklyn HF account —
 * use them verbatim.
 */

export interface CatalogEntry {
  /** Wizard display name. */
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
}

export const MODEL_CATALOG: readonly CatalogEntry[] = [
  {
    label: 'Qwen3.6-27B',
    hfRepo: 'Brooooooklyn/Qwen3.6-27B-NVFP4-mlx',
    sizeGb: 22.2,
    description: 'Best tool use — recommended default',
    isDefault: true,
  },
  {
    label: 'Qwen-AgentWorld-35B',
    hfRepo: 'Brooooooklyn/Qwen-AgentWorld-35B-A3B-nvfp4-mlx',
    sizeGb: 22.7,
    description: 'Agent-tuned MoE, fast decode',
  },
  {
    label: 'Gemma-4-26B-A4B',
    hfRepo: 'Brooooooklyn/Gemma-4-26B-A4B-NVFP4-mlx',
    sizeGb: 18.8,
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
