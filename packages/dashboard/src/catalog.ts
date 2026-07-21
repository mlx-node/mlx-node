/**
 * Dashboard-side view of the curated model catalog. The list itself lives in
 * `packages/agent/src/catalog.ts` (single source of truth), imported through the
 * agent's `./catalog` subpath export so the dashboard never touches the agent's
 * index (which transitively loads the native addon). This module only overlays
 * installed/slug state derived from the local models directory.
 */

import { existsSync } from 'node:fs';
import { join } from 'node:path';

import { type CatalogEntry, MODEL_CATALOG } from '@mlx-node/agent/catalog';

export interface CatalogItem extends CatalogEntry {
  /** Local directory name a download lands in (`hfRepo` basename, lowercased). */
  slug: string;
  /** Whether `<modelsDir>/<slug>` already exists on disk. */
  installed: boolean;
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
    return { ...entry, slug, installed: existsSync(join(modelsDir, slug)) };
  });
}
