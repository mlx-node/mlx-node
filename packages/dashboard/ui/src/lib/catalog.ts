/** Pure helpers for catalog cards (Models) and the first-model choice (Onboarding). */

import type { CatalogItem } from './types';

const BUILD_NAMES = new Set(['mxfp', 'mxfp4', 'mxfp8', 'nvfp', 'nvfp4', 'nvfp8']);

/** The text between the stars of a glob like `*UD-Q4_K_XL*`, or null for any other shape. */
function starredBuild(glob: string): string | null {
  if (glob.length < 3 || !glob.startsWith('*') || !glob.endsWith('*')) return null;
  const inner = glob.slice(1, -1);
  if (inner.includes('.') || inner.includes('*')) return null;
  return inner;
}

/** "UD-Q4_K_XL" from globs ['*UD-Q4_K_XL*', …]; else "MXFP4" / "NVFP4" / "MXFP8" from the repo name; else null. */
export function catalogBuildLabel(item: Pick<CatalogItem, 'hfRepo' | 'globs'>): string | null {
  for (const glob of item.globs ?? []) {
    const build = starredBuild(glob);
    if (build !== null) return build;
  }
  const slash = item.hfRepo.lastIndexOf('/');
  const base = slash === -1 ? item.hfRepo : item.hfRepo.slice(slash + 1);
  for (const part of base.split('-')) {
    const lower = part.toLowerCase();
    if (BUILD_NAMES.has(lower)) return part.toUpperCase();
  }
  return null;
}

/** One plain sentence about a catalog model. */
export function modelBlurb(item: Pick<CatalogItem, 'label' | 'description' | 'isDefault'>): string {
  const label = item.label.toLowerCase();
  if (label.includes('agentworld')) return 'Tuned for agent workflows, with fast responses.';
  if (label.includes('gemma-4-26b')) return 'Fast responses, with a smaller download to get started.';
  if (item.isDefault === true) return 'Our recommended starting point for working with tools.';
  return item.description;
}

/**
 * Whether an installed catalog model has newer bytes upstream.
 *
 * Gated on `installed`, never `present`: only a dashboard-owned install at the
 * canonical slug carries a revision to compare AND can actually be re-installed.
 * A `null` on either side means staleness is unknowable, never "up to date".
 */
export function hasUpdate(
  item: Pick<
    CatalogItem,
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
