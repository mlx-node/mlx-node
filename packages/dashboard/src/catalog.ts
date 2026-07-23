/**
 * Dashboard-side view of the curated model catalog. The list itself lives in
 * `packages/agent/src/catalog.ts` (single source of truth), imported through the
 * agent's `./catalog` subpath export so the dashboard never touches the agent's
 * index (which transitively loads the native addon). This module only overlays
 * installed/slug state derived from the local models directory.
 */

import { join } from 'node:path';

import { type CatalogEntry, MODEL_CATALOG } from '@mlx-node/agent/catalog';

import { isModelInstalled, isModelPresent } from './models.js';

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
export function catalogWithState(modelsDir: string): CatalogItem[] {
  return MODEL_CATALOG.map((entry) => {
    const slug = catalogSlug(entry);
    const dir = join(modelsDir, slug);
    return { ...entry, slug, installed: isModelInstalled(dir), present: isModelPresent(dir) };
  });
}
