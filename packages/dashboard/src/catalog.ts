/**
 * Dashboard-side view of the curated model catalog. The list itself lives in
 * `packages/agent/src/catalog.ts` (single source of truth), imported through the
 * agent's `./catalog` subpath export so the dashboard never touches the agent's
 * index (which transitively loads the native addon). This module only overlays
 * installed/slug state derived from the local models directory.
 */

import { readdirSync } from 'node:fs';
import { join } from 'node:path';

import { type CatalogEntry, MODEL_CATALOG } from '@mlx-node/agent/catalog';

import { isModelInstalled, isModelPresent, readCheckpointFingerprint } from './models.js';

export interface CatalogItem extends CatalogEntry {
  /** Local directory name a download lands in (`hfRepo` basename, lowercased). */
  slug: string;
  /**
   * Whether a dashboard-OWNED completed download of `<slug>` is present under
   * `modelsDir` — keyed on the atomic-publish completion marker, not bare directory
   * existence, so a partial/aborted download never masquerades as installed.
   */
  installed: boolean;
  /**
   * Whether a loadable checkpoint for `<slug>` is present on disk regardless of the
   * dashboard marker — true for {@link installed}, and additionally for a model the
   * user installed via the `mlx download` CLI / agent wizard. The UI uses this to
   * show the model as present instead of offering an Install that would refuse to
   * overwrite the unowned directory and fail.
   */
  present: boolean;
}

/** The slug a catalog entry installs to: the `hfRepo` basename, lowercased. */
export function catalogSlug(entry: CatalogEntry): string {
  return entry.hfRepo.split('/').pop()!.toLowerCase();
}

/**
 * The full catalog with each entry tagged by its install slug and whether that
 * slug's directory is present under `modelsDir`. Hidden entries are retained
 * here (the UI decides what to show); the agent wizard's `visibleCatalog()`
 * filter is a separate concern.
 */
/**
 * Whether any local checkpoint IS this recommended model under a different folder
 * name — a renamed or re-quantized upload the canonical-slug check misses. Guards
 * against a false match two ways: the folder name must start with the entry
 * `label` (the base model name — so AgentWorld-35B-A3B and Qwen3.6-35B-A3B, both
 * `qwen3_5_moe` nvfp4, never cross-match), AND the checkpoint's `config.json`
 * fingerprint (family + quant bits/mode/group) must equal the entry's. Only reads
 * `config.json` for the few prefix-matching, loadable dirs, not the whole store.
 */
function matchesLocalVariant(entry: CatalogEntry, modelsDir: string): boolean {
  const fp = entry.fingerprint;
  if (fp === undefined) return false;
  const baseToken = entry.label.toLowerCase();
  let names: string[];
  try {
    names = readdirSync(modelsDir);
  } catch {
    return false;
  }
  for (const name of names) {
    if (!name.toLowerCase().startsWith(baseToken)) continue;
    const dir = join(modelsDir, name);
    if (!isModelPresent(dir)) continue;
    const local = readCheckpointFingerprint(dir);
    if (
      local !== undefined &&
      local.modelType === fp.modelType &&
      local.quant !== null &&
      local.quant.bits === fp.quant.bits &&
      local.quant.mode === fp.quant.mode &&
      local.quant.group === fp.quant.group
    ) {
      return true;
    }
  }
  return false;
}

export function catalogWithState(modelsDir: string): CatalogItem[] {
  return MODEL_CATALOG.map((entry) => {
    const slug = catalogSlug(entry);
    const dir = join(modelsDir, slug);
    // `present` is true for the canonical-slug dir OR any local checkpoint whose
    // fingerprint matches — so a recommended model installed under a different
    // folder name (a renamed / re-quantized upload) is still recognized.
    const present = isModelPresent(dir) || matchesLocalVariant(entry, modelsDir);
    return { ...entry, slug, installed: isModelInstalled(dir), present };
  });
}
