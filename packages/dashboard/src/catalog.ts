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
  catalogSupportedOnPlatform,
  MODEL_CATALOG,
} from '@mlx-node/agent/catalog';
import { findDFlash2Draft, isDFlash2Companion } from '@mlx-node/lm/draft-companion';

import {
  catalogFolderModel,
  type ConfiglessBuild,
  type ForeignEntrySummary,
  isDownloaderOwned,
  isListedFolder,
  isModelInstalled,
  isModelPresent,
  isPathOccupied,
  type LocatedState,
  locatedState,
  missingNamedFile,
  readCompletion,
  summarizeForeignEntry,
  updateDropsWeight,
} from './models.js';

/** What is in the way of a blocked install: what the cleanup confirmation shows and its request echoes back. */
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
  /** Minimum CUDA device memory in GiB required on non-macOS hosts. */
  minGpuMemoryGb?: number;
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
  /**
   * Set when {@link installed} and neither the marker nor the folder holds a file named by a glob of the build with no
   * `*`: an install made before the catalog named that file. Only for a build of its own repository, and never when
   * the marker or the folder holds a weight of another build, which an Update would delete.
   */
  lacksNamedFile?: true;
  /**
   * Set when Locate made `<slug>` from files the user already had: the folder they are in, whether the models folder
   * shares them on disk or links to them, and whether they can be reached now. A folder whose files cannot be reached
   * stays the app's own: not present, no blocker.
   */
  located: Pick<LocatedState, 'from' | 'mode' | 'filesAvailable'> | null;
  /** Local models lists the folder at `<slug>`, by discovery's own rule. */
  listed: boolean;
}

export interface CatalogItem {
  label: string;
  description: string;
  /** Which product surface this catalog entry belongs to. */
  category: 'chat' | 'systemone';
  minMemoryGb?: number;
  isDefault?: boolean;
  hidden?: boolean;
  platforms?: readonly NodeJS.Platform[];
  /** In catalog order; `builds[0]` is the build every surface chooses first. */
  builds: CatalogBuildItem[];
  draft?: CatalogDraftItem;
}

export interface CatalogDraftItem {
  label: string;
  hfRepo: string;
  sizeGb: number;
  /** When true, installing a build of this entry downloads this draft alongside it. */
  autoInstall?: boolean;
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

/**
 * The family a folder with no `config.json` must state in a GGUF header to be `build` of `entry`
 * ({@link CatalogEntry.ggufFamily}): only for a build of a GGUF repository on this platform.
 */
function ggufFamilyOf(
  entry: CatalogEntry,
  build: CatalogBuild,
  platform: NodeJS.Platform = process.platform,
): string | undefined {
  return catalogBuildRepo(build, platform).endsWith('-GGUF') ? entry.ggufFamily : undefined;
}

/**
 * The {@link ConfiglessBuild} of `build` when it installs with no `config.json`: a GGUF build with no
 * {@link CatalogBuild.assetsRepo} on this platform whose entry names a {@link CatalogEntry.ggufFamily}. The globs and
 * the sidecar source are the build's only for its own repository, as a job plans them (`catalogTargets`).
 */
export function configlessBuildOf(
  entry: CatalogEntry,
  build: CatalogBuild,
  platform: NodeJS.Platform = process.platform,
): ConfiglessBuild | undefined {
  const own = catalogBuildRepo(build, platform) === build.hfRepo;
  if (own && build.assetsRepo !== undefined) return undefined;
  const family = ggufFamilyOf(entry, build, platform);
  return family === undefined ? undefined : { family, globs: own ? build.globs : undefined };
}

/**
 * Whether the folder `name` under `modelsDir` holds `build` ({@link catalogFolderModel} with the build's
 * {@link ggufFamilyOf}, and `configless` for a build that installs with no `config.json`): a GGUF weight file names the
 * build, or, with `byName` false, the folder has a `config.json`.
 */
function holdsBuild(
  modelsDir: string,
  name: string,
  build: CatalogBuild,
  family: string | undefined,
  configless: boolean,
  byName: boolean,
): boolean {
  const model = catalogFolderModel(join(modelsDir, name), name, family, configless);
  return model !== undefined && ((!byName && !model.configless) || model.builds.includes(build.id));
}

/**
 * Whether any other catalog build (on this platform or another platform) shares
 * `build`'s repository, requiring variant-filename identity checks so one
 * platform's build can never spoof or overwrite another build's checkpoint.
 */
function repoHasOtherBuilds(build: CatalogBuild, platform: NodeJS.Platform): boolean {
  if (catalogSharesRepo(build, platform)) return true;
  const hfRepo = catalogBuildRepo(build, platform);
  return MODEL_CATALOG.some((candidate) =>
    candidate.builds.some((other) => other !== build && other.hfRepo === hfRepo),
  );
}

/**
 * What the loadable model at the folder of `build` is when it is not `build`, for a build whose repository another
 * build shares: its build names, or `another model` when its weights state none. `null` when the folder holds no
 * loadable model, holds `build`, or the repository is the build's alone.
 */
function heldByOther(
  modelsDir: string,
  build: CatalogBuild,
  family: string | undefined,
  configless: boolean,
  platform: NodeJS.Platform = process.platform,
): string | null {
  if (!repoHasOtherBuilds(build, platform)) return null;
  const slug = catalogBuildSlug(build, platform);
  const model = catalogFolderModel(join(modelsDir, slug), slug, family, configless);
  if (model === undefined || model.builds.includes(build.id)) return null;
  return model.builds.length > 0 ? model.builds.join(', ') : 'another model';
}

/** The entry names of `dir`; none when it cannot be read. */
function namesIn(dir: string): string[] {
  try {
    return readdirSync(dir);
  } catch {
    return [];
  }
}

/** The state of one build's install folder, by the rules on {@link CatalogBuildItem}. */
function buildWithState(
  modelsDir: string,
  entry: CatalogEntry,
  build: CatalogBuild,
  downloaded: ReadonlyMap<string, readonly string[]>,
  platform: NodeJS.Platform = process.platform,
): CatalogBuildItem {
  const hfRepo = catalogBuildRepo(build, platform);
  const slug = catalogBuildSlug(build, platform);
  const dir = join(modelsDir, slug);
  const markedFolders = downloaded.get(hfRepo.toLowerCase()) ?? [];
  const family = ggufFamilyOf(entry, build, platform);
  const configless = configlessBuildOf(entry, build, platform);
  let present: boolean;
  let installed: boolean;
  let heldBy: string | null = null;
  if (repoHasOtherBuilds(build, platform)) {
    // The folder is the build it HOLDS, never the build its name suggests: a
    // marker names only the repository, and multiple builds share it (on this
    // platform or across platforms, e.g. Hy4-flash Q4_K_M vs SPARK-4Bit*).
    const holds = (name: string): boolean => holdsBuild(modelsDir, name, build, family, configless !== undefined, true);
    const siblings = MODEL_CATALOG.flatMap((candidate) => candidate.builds).filter(
      (other) =>
        other !== build &&
        ((catalogSupportedOnPlatform(other, platform) && catalogBuildRepo(other, platform) === hfRepo) ||
          other.hfRepo === hfRepo),
    );
    present =
      holds(slug) || siblings.some((other) => holds(catalogBuildSlug(other, platform))) || markedFolders.some(holds);
    installed = isModelInstalled(dir, configless) && holds(slug);
    heldBy = heldByOther(modelsDir, build, family, configless !== undefined, platform);
  } else {
    // A loadable checkpoint at the folder OR under any folder name whose
    // completion marker names THIS PLATFORM's repo. Matching the other
    // platform's build too would render the card "Installed" with `installed`
    // false — no Install button and no update affordance.
    const holds = (name: string): boolean =>
      holdsBuild(modelsDir, name, build, family, configless !== undefined, false);
    present = holds(slug) || markedFolders.some(holds);
    installed = isModelInstalled(dir, configless);
  }
  // Exactly the state the download runner's ownership preflight refuses, computed
  // with the SAME predicates it uses (isPathOccupied stays no-follow; a live
  // symlink into a marked dir counts as owned) so the two cannot disagree.
  const blocker = blockerAt(dir, !present && heldBy === null && isPathOccupied(dir) && !isDownloaderOwned(dir));
  // Read from the build's own folder ONLY — see `localRevision` for why a
  // provenance-matched renamed dir deliberately reports null.
  const completion = installed ? readCompletion(dir) : undefined;
  const held = completion === undefined ? [] : [...completion.files, ...namesIn(dir)];
  const lacksNamedFile =
    completion !== undefined &&
    hfRepo === build.hfRepo &&
    missingNamedFile(build.globs, held) !== undefined &&
    !updateDropsWeight(build.globs, held);
  const located = locatedState(dir, configless);
  return {
    id: build.id,
    hfRepo,
    sizeGb: build.sizeGb,
    source: build.source,
    ...(build.note !== undefined ? { note: build.note } : {}),
    ...(build.included !== undefined ? { included: build.included } : {}),
    ...(build.minGpuMemoryGb !== undefined ? { minGpuMemoryGb: build.minGpuMemoryGb } : {}),
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
    ...(lacksNamedFile ? { lacksNamedFile: true } : {}),
    located:
      located === null ? null : { from: located.from, mode: located.mode, filesAvailable: located.filesAvailable },
    listed: isListedFolder(dir),
  };
}

/** The state of `entry`'s companion draft; its lookup checks the entry's builds and canonical shared folder. */
function draftWithState(
  modelsDir: string,
  entry: CatalogEntry,
  platform: NodeJS.Platform = process.platform,
): CatalogDraftItem | undefined {
  if (entry.draft === undefined) return undefined;
  const draftSlug = catalogDraftSlug(entry.draft);
  const canonicalDraftDir = join(modelsDir, draftSlug);
  let found: string | undefined;
  if (entry.ggufFamily !== undefined) {
    for (const build of entry.builds) {
      if (!catalogSupportedOnPlatform(build, platform)) continue;
      const candidate = findDFlash2Draft(
        join(modelsDir, catalogBuildSlug(build, platform)),
        entry.ggufFamily,
        modelsDir,
      );
      // `findDFlash2Draft` falls back to bundled DSpark files for `inkling2`; only a real DFlash2 companion counts here.
      if (candidate !== undefined && isDFlash2Companion(candidate)) {
        found = candidate;
        break;
      }
    }
  }
  const draftDir = found ?? canonicalDraftDir;
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
 * The full catalog with each platform-supported build tagged by its install folder and that folder's state under
 * `modelsDir`. Hidden and entry-level platform-restricted entries are retained here (the UI decides what to show via
 * `offeredOnHost`), while build-level `platforms` are filtered for `platform` so `builds[0]` and every selectable
 * build match `catalogTargets(platform)`.
 */
export function catalogWithState(modelsDir: string, platform: NodeJS.Platform = process.platform): CatalogItem[] {
  const downloaded = downloadedRepos(modelsDir);
  return MODEL_CATALOG.map((entry) => ({
    label: entry.label,
    description: entry.description,
    category: entry.category ?? 'chat',
    ...(entry.minMemoryGb !== undefined ? { minMemoryGb: entry.minMemoryGb } : {}),
    ...(entry.isDefault !== undefined ? { isDefault: entry.isDefault } : {}),
    ...(entry.hidden !== undefined ? { hidden: entry.hidden } : {}),
    ...(entry.platforms !== undefined ? { platforms: entry.platforms } : {}),
    builds: entry.builds
      .filter((build) => catalogSupportedOnPlatform(build, platform))
      .map((build) => buildWithState(modelsDir, entry, build, downloaded, platform)),
    draft: draftWithState(modelsDir, entry, platform),
  }));
}

/** The blocker of the catalog build or draft that installs to `slug`, read the same way the card reads it. */
export function catalogBlocker(
  modelsDir: string,
  slug: string,
  platform: NodeJS.Platform = process.platform,
): CatalogBlocker | null {
  for (const item of catalogWithState(modelsDir, platform)) {
    for (const build of item.builds) if (build.slug === slug) return build.blocker;
    if (item.draft?.slug === slug) return item.draft.blocker;
  }
  return null;
}

/**
 * What another build of the same repository the folder `slug` holds, the {@link CatalogBuildItem.heldBy} of the
 * build that installs there; `null` when none does or `slug` is no build's folder.
 */
export function heldByOtherBuild(
  modelsDir: string,
  slug: string,
  platform: NodeJS.Platform = process.platform,
): string | null {
  for (const entry of MODEL_CATALOG) {
    if (!catalogSupportedOnPlatform(entry, platform)) continue;
    for (const build of entry.builds) {
      if (!catalogSupportedOnPlatform(build, platform)) continue;
      if (catalogBuildSlug(build, platform) === slug) {
        return heldByOther(
          modelsDir,
          build,
          ggufFamilyOf(entry, build, platform),
          configlessBuildOf(entry, build, platform) !== undefined,
          platform,
        );
      }
    }
  }
  return null;
}
