/**
 * Dashboard-side view of the curated model catalog. The list itself lives in
 * `packages/agent/src/catalog.ts` (single source of truth), imported through the
 * agent's `./catalog` subpath export so the dashboard never touches the agent's
 * index (which transitively loads the native addon). This module only overlays
 * installed/slug state derived from the local models directory.
 */

import { readdirSync } from 'node:fs';
import { join } from 'node:path';

import { type CatalogEntry, catalogRepo, MODEL_CATALOG } from '@mlx-node/agent/catalog';

import { isDownloaderOwned, isModelInstalled, isModelPresent, isPathOccupied, readCompletion } from './models.js';

export interface CatalogItem extends CatalogEntry {
  /**
   * The repo THIS platform installs, already resolved by
   * {@link catalogRepo} — MXFP4 on Apple Silicon, NVFP4 on CUDA. It shadows
   * `CatalogEntry.hfRepo` so every downstream consumer (the Models page, the
   * download POST, the runner's allowlist) sees ONE repo and cannot
   * accidentally install the other platform's build.
   */
  hfRepo: string;
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
  /**
   * The commit sha the local bytes were pinned to, read off the completion
   * marker at the canonical slug — or `null` when there is no marker to read.
   *
   * Only ever set when {@link installed} is true, i.e. for a dashboard-owned
   * install at the canonical slug. Three cases deliberately report `null`:
   * a hand-copied dir and a pre-marker CLI install (no marker exists, so
   * staleness is genuinely unknowable), and a dashboard install the user
   * RENAMED (matched into {@link present} by provenance, but the runner would
   * refuse to re-install over the unowned canonical slug, so an update
   * affordance there could only ever fail).
   *
   * A `null` here means "no update badge", never "up to date".
   */
  localRevision: string | null;
}

/**
 * The slug a catalog entry installs to: the basename of THIS platform's repo,
 * lowercased. Resolved through {@link catalogRepo}, so the macOS and CUDA
 * builds of one model occupy different directories and never overwrite each
 * other on a shared models dir.
 */
export function catalogSlug(entry: CatalogEntry): string {
  return catalogRepo(entry).split('/').pop()!.toLowerCase();
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
function downloadedRepos(modelsDir: string): Map<string, string> {
  const repos = new Map<string, string>();
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
    repos.set(completion.repo.toLowerCase(), completion.revision);
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
    const hfRepo = catalogRepo(entry);
    const slug = catalogSlug(entry);
    const dir = join(modelsDir, slug);
    // `present` is true for a loadable checkpoint at the canonical slug OR under any
    // folder name whose completion marker names THIS PLATFORM's repo.
    //
    // Matching the other platform's build too would be actively harmful, and not
    // hypothetically: the catalog recommended the nvfp4 builds to every platform
    // before this became platform-conditional, so an existing macOS install of
    // e.g. `Qwen-AgentWorld-35B-A3B-nvfp4-mlx` is exactly the CUDA alias. Counting
    // it would render the card "Installed" with `installed` false — no Install
    // button, and no update affordance either — permanently stranding the very
    // users this change exists to move onto the mxfp4 build.
    const present = isModelPresent(dir) || downloaded.has(catalogRepo(entry).toLowerCase());
    // Exactly the state the download runner's ownership preflight refuses, computed
    // with the SAME no-follow predicates it uses so the two cannot disagree.
    const blockedByForeignDir = !present && isPathOccupied(dir) && !isDownloaderOwned(dir);
    const installed = isModelInstalled(dir);
    // Read from the canonical slug ONLY — see `localRevision` on CatalogItem for why
    // a provenance-matched renamed dir deliberately reports null.
    const completion = installed ? readCompletion(dir) : undefined;
    return {
      ...entry,
      hfRepo,
      slug,
      installed,
      present,
      blockedByForeignDir,
      localRevision: completion?.revision ?? null,
    };
  });
}
