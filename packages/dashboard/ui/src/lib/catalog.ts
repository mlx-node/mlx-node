/** Pure helpers for catalog cards (Models) and the first-model choice (Onboarding). */

import type { CatalogBuildItem, CatalogItem, ModelsResponse } from './types';

const GIB = 2 ** 30;

/**
 * Whether the page offers `item` on this Mac: not hidden, and its memory floor met. An entry with a floor waits
 * for the host reading.
 */
export function offeredOnHost(
  item: Pick<CatalogItem, 'hidden' | 'minMemoryGb'>,
  host: ModelsResponse['host'] | undefined,
): boolean {
  if (item.hidden === true) return false;
  if (item.minMemoryGb === undefined) return true;
  return host !== undefined && host.totalMemBytes >= item.minMemoryGb * GIB;
}

/** The build that installs to the folder `slug`, from any entry. */
export function catalogBuildAt(items: readonly CatalogItem[], slug: string): CatalogBuildItem | undefined {
  for (const item of items) {
    const build = item.builds.find((candidate) => candidate.slug === slug);
    if (build !== undefined) return build;
  }
  return undefined;
}

/** A model counts as on this Mac when at least one of its builds is on disk. */
export function modelPresent(item: Pick<CatalogItem, 'builds'>): boolean {
  return item.builds.some((build) => build.present);
}

/** The build Onboarding stands a model for: the one with a job, else the first on disk, else the first. */
export function onboardingBuild(item: Pick<CatalogItem, 'builds'>, activeSlugs: readonly string[]): CatalogBuildItem {
  return (
    item.builds.find((build) => activeSlugs.includes(build.slug)) ??
    item.builds.find((build) => build.present) ??
    item.builds[0]
  );
}

/**
 * The build a catalog card shows: the one the user chose on this page visit, else the one with a job, else the first
 * on disk, else the first. `chosen` maps an entry label to a build folder.
 */
export function shownBuild(
  item: Pick<CatalogItem, 'label' | 'builds'>,
  chosen: Readonly<Record<string, string>>,
  active: Readonly<Record<string, unknown>>,
): CatalogBuildItem {
  const picked = Object.hasOwn(chosen, item.label)
    ? item.builds.find((build) => build.slug === chosen[item.label])
    : undefined;
  return picked ?? onboardingBuild(item, Object.keys(active));
}

/** One plain sentence about a catalog model. */
export function modelBlurb(item: Pick<CatalogItem, 'label' | 'description' | 'isDefault'>): string {
  if (item.isDefault === true) return 'Our recommended starting point for working with tools.';
  return item.description;
}

/**
 * Whether an installed catalog build has newer bytes upstream.
 *
 * Gated on `installed`, never `present`: only a dashboard-owned install at the
 * canonical slug carries a revision to compare AND can actually be re-installed.
 * A `null` on either side means staleness is unknowable, never "up to date".
 */
export function hasUpdate(
  item: Pick<
    CatalogBuildItem,
    'hfRepo' | 'installed' | 'localRevision' | 'assetsRepo' | 'localAssetsRepo' | 'localAssetsRevision'
  >,
  remoteRevisions: ReadonlyMap<string, string | null>,
): boolean {
  if (!item.installed || item.localRevision === null) return false;
  const remoteRevision = remoteRevisions.get(item.hfRepo) ?? null;
  if (remoteRevision !== null && remoteRevision !== item.localRevision) return true;
  // The catalog's CURRENT assets repo participates too: when the entry
  // switches sources the marker still names the old one — which the sweep
  // never probes — so the lookup below returns null and the badge could
  // never appear; a changed source is an update on its own (the job re-plans
  // sidecars from it). A marker with no recorded source gets the same
  // verdict: the entry prescribes sidecars the install lacks. The reverse —
  // a catalog that dropped its assetsRepo — adds no verdict of its own: the
  // entry prescribes no sidecars for a job to apply.
  if (item.assetsRepo !== undefined && item.assetsRepo !== item.localAssetsRepo) return true;
  // The tokenizer sidecars come from a second repo that moves on its own
  // revision; a fix there repairs nothing unless the badge appears, because
  // the Installed button is otherwise disabled and no job ever runs.
  if (item.localAssetsRepo === null || item.localAssetsRevision === null) return false;
  const remoteAssets = remoteRevisions.get(item.localAssetsRepo) ?? null;
  return remoteAssets !== null && remoteAssets !== item.localAssetsRevision;
}
