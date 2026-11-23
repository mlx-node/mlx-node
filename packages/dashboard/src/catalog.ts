/**
 * Dashboard-side view of the curated model catalog. The list itself lives in
 * `packages/agent/src/catalog.ts` (single source of truth), imported through the
 * agent's `./catalog` subpath export so the dashboard never touches the agent's
 * index (which transitively loads the native addon). This module only overlays
 * installed/slug state derived from the local models directory.
 */

import { readdirSync } from 'node:fs';
import { join } from 'node:path';

import {
  type CatalogBuild,
  type CatalogEntry,
  catalogBuildRepo,
  catalogBuildSlug,
  catalogDraftSlug,
  catalogSharesRepo,
  MODEL_CATALOG,
} from '@mlx-node/agent/catalog';
import { findDFlash2Draft, isDFlash2Companion } from '@mlx-node/lm/draft-companion';

import {
  type ForeignEntrySummary,
  ggufBuilds,
  isDownloaderOwned,
  isModelInstalled,
  isModelPresent,
  isPathOccupied,
  readCompletion,
  summarizeForeignEntry,
} from './models.js';

/** What is in the way of a blocked install: the numbers Reinstall shows and echoes back. */
export type CatalogBlocker = ForeignEntrySummary;

/** One build of a catalog model with the state of its install folder. */
export interface CatalogBuildItem {
  /** The build's name on every surface: `UD-Q4_K_XL`, `Original`. */
  id: string;
  /**
   * The repo THIS platform installs, already resolved by `catalogBuildRepo`, so every downstream consumer (the
   * Models page, update discovery) sees ONE repo and cannot install the other platform's build.
   */
  hfRepo: string;
  sizeGb: number;
  /** Who made the files and in which format: `Unsloth GGUF`, `Thinking Machines`. */
  source: string;
  /** One short fact after the source: `1.1 GB smaller`. */
  note?: string;
  /** What comes with the weights. */
  included?: string;
  /**
   * The sidecar source a job plans for this repo: the build's `assetsRepo` only when {@link hfRepo} is the build's
   * GGUF repo. A platform-override build plans none, and serving the raw field would pin an update badge its marker
   * could never record.
   */
  assetsRepo?: string;
  /** The folder under `modelsDir` this build installs to; the key of its download. */
  slug: string;
  /**
   * Whether a dashboard-OWNED completed download of this build is at `<slug>` — keyed on the atomic-publish
   * completion marker, not bare directory existence, so a partial/aborted download never masquerades as installed.
   * For a build whose repository another build shares, the folder's weight file must also name this build.
   */
  installed: boolean;
  /**
   * Whether a loadable checkpoint of this build is on disk regardless of the dashboard marker — true for
   * {@link installed}, and also for a model the user installed with the `mlx download` CLI / agent wizard, or a
   * dashboard install the user renamed (found by its marker's repo). For a build whose repository another build
   * shares, only a folder whose weight file names this build counts.
   */
  present: boolean;
  /**
   * The build(s) the loadable model at `<slug>` is, when it is another build of the same repository: `UD-Q4_K_M`.
   * The server refuses an install into that folder, because the owned swap would delete the other build. `null` for
   * every other state, and always for a build whose repository no other build shares.
   */
  heldBy: string | null;
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
  /** What occupies `<slug>`; non-null exactly when {@link blockedByForeignDir} is true. */
  blocker: CatalogBlocker | null;
  /**
   * The commit sha the local bytes were pinned to, read off the completion
   * marker at `<slug>` — or `null` when there is no marker to read.
   *
   * Only ever set when {@link installed} is true. Three cases deliberately report `null`:
   * a hand-copied dir and a pre-marker CLI install (no marker exists, so
   * staleness is genuinely unknowable), and a dashboard install the user
   * RENAMED (matched into {@link present} by provenance, but the runner would
   * refuse to re-install over the unowned canonical slug, so an update
   * affordance there could only ever fail).
   *
   * A `null` here means "no update badge", never "up to date".
   */
  localRevision: string | null;
  /**
   * The base-model repo the install's tokenizer sidecars came from, and the
   * commit they were pinned to (marker `assetsRepo`/`assetsRevision`) — or
   * `null` when the install used none, or predates the field.
   *
   * Update discovery compares this pair in addition to `repo`/`revision`: a
   * sidecar-only upstream change moves nothing in the primary repo, so
   * without it the badge never appears and the repair job — which verifies
   * planned sidecars before reading an install as done — can never run.
   * `null` means "no assets-side update known", never "up to date".
   */
  localAssetsRepo: string | null;
  localAssetsRevision: string | null;
}

export interface CatalogItem {
  label: string;
  description: string;
  /** Which product surface this catalog entry belongs to. */
  category: 'chat' | 'systemone';
  minMemoryGb?: number;
  isDefault?: boolean;
  hidden?: boolean;
  /** In catalog order; `builds[0]` is the build every surface chooses first. */
  builds: CatalogBuildItem[];
  draft?: CatalogDraftItem;
}

export interface CatalogDraftItem {
  label: string;
  hfRepo: string;
  sizeGb: number;
  slug: string;
  installed: boolean;
  present: boolean;
  blockedByForeignDir: boolean;
  blocker: CatalogBlocker | null;
  localRevision: string | null;
}

/** The blocker at `dir` when the catalog calls it blocked; walked only then. */
function blockerAt(dir: string, blocked: boolean): CatalogBlocker | null {
  return blocked ? summarizeForeignEntry(dir) : null;
}

/**
 * HF repos (lowercased) a local checkpoint under ANY folder name provably came
 * from, each with the names of the folders that hold one — read off the download completion
 * marker, which pins the repo and the revision the bytes were fetched at. This is
 * how a dashboard install the user RENAMED is still recognized as its catalog build.
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
 * deliberately NOT matched here: the folder check below
 * still covers it, and the worst case of missing a renamed copy is ONE redundant
 * download, against a hard block for the false match.
 *
 * The marker records what was published, not what survives, so it is paired with
 * an on-disk check ({@link isModelPresent}) — a dir gutted down to its marker is
 * not a present checkpoint. Scanned once for the whole catalog.
 */
function downloadedRepos(modelsDir: string): Map<string, string[]> {
  const repos = new Map<string, string[]>();
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
    const repo = completion.repo.toLowerCase();
    const folders = repos.get(repo);
    if (folders === undefined) repos.set(repo, [name]);
    else folders.push(name);
  }
  return repos;
}

/** Whether the folder `name` under `modelsDir` holds a loadable model whose GGUF weight file names `buildId`. */
function holdsBuild(modelsDir: string, name: string, buildId: string): boolean {
  const dir = join(modelsDir, name);
  return isModelPresent(dir) && ggufBuilds(dir, name).includes(buildId);
}

/**
 * What the loadable model at the folder of `build` is when it is not `build`, for a build whose repository another
 * build shares: its build names, or `another model` when its weights state none. `null` when the folder holds no
 * loadable model, holds `build`, or the repository is the build's alone.
 */
function heldByOther(modelsDir: string, build: CatalogBuild): string | null {
  if (!catalogSharesRepo(build)) return null;
  const slug = catalogBuildSlug(build);
  const dir = join(modelsDir, slug);
  if (!isModelPresent(dir)) return null;
  const builds = ggufBuilds(dir, slug);
  if (builds.includes(build.id)) return null;
  return builds.length > 0 ? builds.join(', ') : 'another model';
}

/** The state of one build's install folder, by the rules on {@link CatalogBuildItem}. */
function buildWithState(
  modelsDir: string,
  build: CatalogBuild,
  downloaded: ReadonlyMap<string, readonly string[]>,
): CatalogBuildItem {
  const hfRepo = catalogBuildRepo(build);
  const slug = catalogBuildSlug(build);
  const dir = join(modelsDir, slug);
  const markedFolders = downloaded.get(hfRepo.toLowerCase()) ?? [];
  let present: boolean;
  let installed: boolean;
  let heldBy: string | null = null;
  if (catalogSharesRepo(build)) {
    // The folder is the build it HOLDS, never the build its name suggests: a
    // marker names only the repository, and two builds share it.
    const holds = (name: string): boolean => holdsBuild(modelsDir, name, build.id);
    const siblings = MODEL_CATALOG.flatMap((entry) => entry.builds).filter(
      (other) => other !== build && catalogBuildRepo(other) === hfRepo,
    );
    present = holds(slug) || siblings.some((other) => holds(catalogBuildSlug(other))) || markedFolders.some(holds);
    installed = isModelInstalled(dir) && holds(slug);
    heldBy = heldByOther(modelsDir, build);
  } else {
    // A loadable checkpoint at the folder OR under any folder name whose
    // completion marker names THIS PLATFORM's repo. Matching the other
    // platform's build too would render the card "Installed" with `installed`
    // false — no Install button and no update affordance.
    present = isModelPresent(dir) || markedFolders.length > 0;
    installed = isModelInstalled(dir);
  }
  // Exactly the state the download runner's ownership preflight refuses, computed
  // with the SAME predicates it uses (isPathOccupied stays no-follow; a live
  // symlink into a marked dir counts as owned) so the two cannot disagree.
  const blocker = blockerAt(dir, !present && heldBy === null && isPathOccupied(dir) && !isDownloaderOwned(dir));
  // Read from the build's own folder ONLY — see `localRevision` for why a
  // provenance-matched renamed dir deliberately reports null.
  const completion = installed ? readCompletion(dir) : undefined;
  return {
    id: build.id,
    hfRepo,
    sizeGb: build.sizeGb,
    source: build.source,
    ...(build.note !== undefined ? { note: build.note } : {}),
    ...(build.included !== undefined ? { included: build.included } : {}),
    ...(hfRepo === build.hfRepo && build.assetsRepo !== undefined ? { assetsRepo: build.assetsRepo } : {}),
    slug,
    installed,
    present,
    heldBy,
    blockedByForeignDir: blocker !== null,
    blocker,
    localRevision: completion?.revision ?? null,
    localAssetsRepo: completion?.assetsRepo ?? null,
    localAssetsRevision: completion?.assetsRevision ?? null,
  };
}

/** The state of `entry`'s companion draft; its lookup starts from the folder of the entry's first build. */
function draftWithState(modelsDir: string, entry: CatalogEntry): CatalogDraftItem | undefined {
  if (entry.draft === undefined) return undefined;
  const draftSlug = catalogDraftSlug(entry.draft);
  const canonicalDraftDir = join(modelsDir, draftSlug);
  const targetDir = join(modelsDir, catalogBuildSlug(entry.builds[0]));
  const draftDir = findDFlash2Draft(targetDir, 'qwen3_5', modelsDir) ?? canonicalDraftDir;
  const draftPresent = isModelPresent(draftDir) && isDFlash2Companion(draftDir);
  const draftInstalled = draftPresent && draftDir === canonicalDraftDir && isModelInstalled(draftDir);
  const draftBlocker = blockerAt(
    canonicalDraftDir,
    !draftPresent && isPathOccupied(canonicalDraftDir) && !isDownloaderOwned(canonicalDraftDir),
  );
  return {
    ...entry.draft,
    slug: draftSlug,
    present: draftPresent,
    installed: draftInstalled,
    blockedByForeignDir: draftBlocker !== null,
    blocker: draftBlocker,
    localRevision: draftInstalled ? (readCompletion(draftDir)?.revision ?? null) : null,
  };
}

/**
 * The full catalog with each build tagged by its install folder and that folder's state under `modelsDir`. Hidden
 * entries are retained here (the UI decides what to show); the agent wizard's `visibleCatalog()` filter is a separate
 * concern.
 */
export function catalogWithState(modelsDir: string): CatalogItem[] {
  const downloaded = downloadedRepos(modelsDir);
  return MODEL_CATALOG.map((entry) => ({
    label: entry.label,
    description: entry.description,
    category: entry.category ?? 'chat',
    ...(entry.minMemoryGb !== undefined ? { minMemoryGb: entry.minMemoryGb } : {}),
    ...(entry.isDefault !== undefined ? { isDefault: entry.isDefault } : {}),
    ...(entry.hidden !== undefined ? { hidden: entry.hidden } : {}),
    builds: entry.builds.map((build) => buildWithState(modelsDir, build, downloaded)),
    draft: draftWithState(modelsDir, entry),
  }));
}

/** The blocker of the catalog build or draft that installs to `slug`, read the same way the card reads it. */
export function catalogBlocker(modelsDir: string, slug: string): CatalogBlocker | null {
  for (const item of catalogWithState(modelsDir)) {
    for (const build of item.builds) if (build.slug === slug) return build.blocker;
    if (item.draft?.slug === slug) return item.draft.blocker;
  }
  return null;
}

/**
 * What another build of the same repository the folder `slug` holds, the {@link CatalogBuildItem.heldBy} of the
 * build that installs there; `null` when none does or `slug` is no build's folder.
 */
export function heldByOtherBuild(modelsDir: string, slug: string): string | null {
  for (const entry of MODEL_CATALOG) {
    for (const build of entry.builds) {
      if (catalogBuildSlug(build) === slug) return heldByOther(modelsDir, build);
    }
  }
  return null;
}
