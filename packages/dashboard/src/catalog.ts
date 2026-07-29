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

import { isDownloaderOwned, isModelInstalled, isModelPresent, isPathOccupied, readCompletion } from './models.js';

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
  /**
   * Whether `<slug>` is occupied by something the downloader may not touch — an
   * interrupted `mlx download`, a hand-made directory, a stray file, a symlink —
   * while holding no loadable checkpoint. An Install in this state cannot succeed:
   * the runner's ownership preflight refuses every occupied, unowned final dir, so
   * the UI states the blockage instead of offering a button that always errors.
   * Never true for a dir carrying OUR marker: an owned dir, however incomplete, is
   * legitimately re-installable through the owned swap.
   */
  blockedByForeignDir: boolean;
}

/** The slug a catalog entry installs to: the `hfRepo` basename, lowercased. */
export function catalogSlug(entry: CatalogEntry): string {
  return entry.hfRepo.split('/').pop()!.toLowerCase();
}

/**
 * HF repos (lowercased) a local checkpoint under ANY folder name provably came
 * from — read off the download completion marker, which pins the repo and the
 * revision the bytes were fetched at. This is how a dashboard install the user
 * RENAMED is still recognized as its catalog entry.
 *
 * Provenance, never config shape: a checkpoint's `model_type` + quant triple says
 * what FORMAT it is, not which weights it holds. `mlx convert` mandates bits=4 /
 * group_size=16 for nvfp4, so that triple is a constant across the family, and a
 * local fine-tune of the same base is indistinguishable from the recommendation
 * by config alone — matching on it marked such a fine-tune "Installed" and
 * disabled the page's only Install button, hard-blocking the real download.
 *
 * A directory with no marker (hand-copied, or installed by an older
 * `mlx download` CLI — the current CLI writes the shared marker) is
 * deliberately NOT matched here: the canonical-slug check below
 * still covers it, and the worst case of missing a renamed copy is ONE redundant
 * download, against a hard block for the false match.
 *
 * The marker records what was published, not what survives, so it is paired with
 * an on-disk check ({@link isModelPresent}) — a dir gutted down to its marker is
 * not a present checkpoint. Scanned once for the whole catalog.
 */
function downloadedRepos(modelsDir: string): Set<string> {
  const repos = new Set<string>();
  let names: string[];
  try {
    names = readdirSync(modelsDir);
  } catch {
    return repos;
  }
  for (const name of names) {
    const dir = join(modelsDir, name);
    const completion = readCompletion(dir);
    if (completion === undefined || !isModelPresent(dir)) continue;
    repos.add(completion.repo.toLowerCase());
  }
  return repos;
}

/**
 * The full catalog with each entry tagged by its install slug and whether that
 * slug's directory is present under `modelsDir`. Hidden entries are retained
 * here (the UI decides what to show); the agent wizard's `visibleCatalog()`
 * filter is a separate concern.
 */
export function catalogWithState(modelsDir: string): CatalogItem[] {
  const downloaded = downloadedRepos(modelsDir);
  return MODEL_CATALOG.map((entry) => {
    const slug = catalogSlug(entry);
    const dir = join(modelsDir, slug);
    // `present` is true for a loadable checkpoint at the canonical slug OR under any
    // folder name whose completion marker names this entry's repo.
    const present = isModelPresent(dir) || downloaded.has(entry.hfRepo.toLowerCase());
    // Exactly the state the download runner's ownership preflight refuses, computed
    // with the SAME no-follow predicates it uses so the two cannot disagree.
    const blockedByForeignDir = !present && isPathOccupied(dir) && !isDownloaderOwned(dir);
    return { ...entry, slug, installed: isModelInstalled(dir), present, blockedByForeignDir };
  });
}
