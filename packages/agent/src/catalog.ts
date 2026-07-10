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
    // Repo publishes after Task 9's local convert + validation; the slug
    // is the intended name and may be corrected in Task 9.
    label: 'Gemma-4-12B',
    hfRepo: 'Brooooooklyn/Gemma-4-12B-IT-q4-mlx',
    sizeGb: 8.4,
    description: 'Compact, fits smaller machines',
    hidden: true,
  },
];

/** Catalog entries the wizard offers (hidden entries filtered out). */
export function visibleCatalog(): CatalogEntry[] {
  return MODEL_CATALOG.filter((entry) => !entry.hidden);
}
