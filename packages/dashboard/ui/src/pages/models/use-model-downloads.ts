import { useEffect, useRef, useState, useSyncExternalStore } from 'react';

import { mutate } from '../../lib/api';
import { hasUpdate } from '../../lib/catalog';
import { getConnectionGeneration, subscribeConnection } from '../../lib/connection';
import { errMessage } from '../../lib/errors';
import { toast } from '../../lib/toast';
import type {
  AdoptPlanResponse,
  CancelDownloadResponse,
  CatalogBuildItem,
  CatalogItem,
  CatalogResponse,
  CatalogUpdatesResponse,
  DownloadJob,
  DownloadStartResponse,
  DownloadsResponse,
  ModelsResponse,
} from '../../lib/types';
import { type AsyncState, useJson } from '../../lib/use-api';

/** The card of `item` offers Update in place, so its job is an Update: Reinstall refuses a folder that holds a model. */
function isUpdateInPlace(item: Pick<DownloadItem, 'blocker'>): boolean {
  return item.blocker?.holdsModel === true;
}

/** The build or draft that installs to the folder `slug` in a catalog body. */
function catalogItemOf(body: CatalogResponse | undefined, slug: string): Pick<DownloadItem, 'blocker'> | undefined {
  for (const item of body?.items ?? []) {
    const build = item.builds.find((candidate) => candidate.slug === slug);
    if (build !== undefined) return build;
    if (item.draft?.slug === slug) return item.draft;
  }
  return undefined;
}

export interface ModelDownloadError {
  slug: string;
  message: string;
}

/**
 * A settled job: it holds no in-flight work and the server never moves it out of this state again. Mirrors
 * `isTerminal` in `src/download.ts`; NOT `state !== 'running'`, because `committing` is the non-cancellable publish
 * window of a job that is still installing a model.
 */
function isTerminalJob(state: DownloadJob['state']): boolean {
  return state === 'done' || state === 'error' || state === 'cancelled';
}

/** A download this page follows: the one job the server holds for a folder while it is neither settled nor absent. */
export interface ActiveJob {
  id: string;
  /** `cancel()` refuses a publishing job, so the card drops its Cancel here. */
  committing: boolean;
  /**
   * This page's own cancel was accepted but the terminal frame has not arrived. The card drops its Cancel (a second one
   * answers 404) while staying subscribed, so the frame still lands and `onCancelled` refreshes the snapshots.
   */
  cancelling: boolean;
  /** Runtime generation that produced this id; ids do not survive a reconnect. */
  connection: number;
  /** A POST result is newer than a GET already in flight; server entries are reconciled against the next snapshot. */
  source: 'local' | 'server';
}

/**
 * The two bodies a card's verdict is joined from, captured when a job settled. Identity is the point: a reload replaces
 * the object, so `!==` means "the refreshed body arrived".
 */
interface SettledBodies {
  catalog: unknown;
  updates: unknown;
}

export type DownloadItem = Pick<
  CatalogBuildItem,
  | 'hfRepo'
  | 'installed'
  | 'localRevision'
  | 'assetsRepo'
  | 'localAssetsRepo'
  | 'localAssetsRevision'
  | 'present'
  | 'blockedByForeignDir'
  | 'blocker'
  | 'heldBy'
  | 'slug'
>;

export interface CatalogDownloadProps {
  installLabel?: string;
  settlingLabel?: string;
  /** Onboarding: the one loud button of its surface. */
  prominent?: boolean;
  item: DownloadItem;
  /** Upstream has different bytes at this build's repo than the local marker records. */
  updateAvailable: boolean;
  /** This folder's job settled but the catalog body it invalidated has not arrived: every field is pre-download. */
  settling: boolean;
  /** The same, for the separately fetched remote shas `updateAvailable` is computed against. */
  updateSettling: boolean;
  job: ActiveJob | undefined;
  onInstall: () => Promise<void>;
  /** Reinstall over `item.blocker`: posts the numbers the user saw and keeps the job id. Rejects with the server text. */
  onReinstall: () => Promise<void>;
  /** After the confirmation's success pill: the card follows the kept job (Reinstall or Update). */
  onReinstallStarted: () => void;
  /** Update of a folder that holds a model the app did not create: loads what it would do. Rejects with the server text. */
  onPlanUpdate: () => Promise<AdoptPlanResponse>;
  /** Starts the Update the plan with `token` describes and keeps the job id. Rejects with the server text. */
  onUpdateInPlace: (token: string) => Promise<void>;
  onDone: (keptDir: string | undefined) => void;
  onError: (message: string) => void;
  onCancelled: () => void;
  onCancel: (id: string) => void;
}

export interface ModelDownloads {
  /** folder → the job this page follows. */
  active: Readonly<Record<string, ActiveJob>>;
  downloadError: ModelDownloadError | null;
  downloadProps: (item: DownloadItem) => CatalogDownloadProps;
  downloadsLoading: boolean;
  downloadsError: Error | undefined;
  reloadDownloads: () => void;
  /** Empty while the update check failed or has not answered: no badge, never "up to date". */
  remoteRevisions: ReadonlyMap<string, string | null>;
}

/**
 * The download state machine of the Models page and Onboarding: hydration from `/downloads`, terminal dismiss, the
 * settle window after a job ends, and cancel marking. `models`, `catalog` and `updates` are the page's own queries.
 */
export function useModelDownloads(
  models: AsyncState<ModelsResponse>,
  catalog: AsyncState<CatalogResponse>,
  updates: AsyncState<CatalogUpdatesResponse>,
): ModelDownloads {
  const downloads = useJson<DownloadsResponse>('/downloads');
  const connection = useSyncExternalStore(subscribeConnection, getConnectionGeneration, getConnectionGeneration);

  /**
   * For a folder whose job just settled, the bodies that were current when it did. `setActive` clears synchronously while
   * `reload()` only marks the old body refreshing, so one committed render still carries the pre-download state: a live
   * Install (or Update available) that would start a second job. Holding the bodies makes the window self-closing:
   * a new identity IS the refreshed body. Both, because the verdict joins both and they arrive independently.
   */
  const [settledOn, setSettledOn] = useState<ReadonlyMap<string, SettledBodies>>(() => new Map());
  /** The same bodies, read by the reconcile effect; it must not depend on them (it reloads them). */
  const catalogBody = useRef(catalog.data);
  catalogBody.current = catalog.data;
  const updatesBody = useRef(updates.data);
  updatesBody.current = updates.data;
  const settledBodies = (): SettledBodies => ({ catalog: catalog.data, updates: updates.data });

  const [active, setActive] = useState<Record<string, ActiveJob>>({});
  const [downloadError, setDownloadError] = useState<ModelDownloadError | null>(null);
  /** Folders whose job failed while this page was unmounted, announced once the catalog body is here. */
  const [failedAway, setFailedAway] = useState<readonly string[]>([]);
  // `useJson` keeps the old body on screen across a reconnect until the new runtime answers. Remember that exact
  // object so the reconcile effect does not read it as the new runtime's registry.
  const seenConnection = useRef(connection);
  const staleDownloads = useRef<DownloadsResponse | undefined>(undefined);
  if (seenConnection.current !== connection) {
    seenConnection.current = connection;
    staleDownloads.current = downloads.data;
  }
  // Destructured on purpose: the state objects are new every render, so naming them in the effect deps would loop.
  const { reload: reloadModels } = models;
  const { reload: reloadCatalog } = catalog;
  const { reload: reloadUpdates } = updates;

  useEffect(() => {
    const snapshot = downloads.data;
    const jobs = snapshot?.jobs;
    if (jobs === undefined) return;
    if (snapshot === staleDownloads.current) return;

    // Every NONTERMINAL job, from the authoritative snapshot (job ids belong to one runtime), plus jobs this page
    // started on this connection whose POST beat the older GET.
    setActive((prev) => {
      const next: Record<string, ActiveJob> = {};
      for (const job of jobs) {
        if (isTerminalJob(job.state)) continue;
        next[job.slug] = {
          id: job.id,
          committing: job.state === 'committing',
          cancelling: false,
          connection,
          source: 'server',
        };
      }
      for (const [slug, job] of Object.entries(prev)) {
        if (job.source === 'local' && job.connection === connection && next[slug]?.id !== job.id) {
          next[slug] = job;
        }
      }
      return next;
    });
    // A job that settled while this page was unmounted got no terminal-dismiss DELETE. Dismiss it here; announce a
    // failure first (its card reverts to Install, so the toast is its only trace). A duplicate dismiss answers 404.
    let settled = false;
    const failed: string[] = [];
    for (const job of jobs) {
      if (!isTerminalJob(job.state)) continue;
      if (job.state === 'error') failed.push(job.slug);
      settled = true;
      void mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(job.id)}`).catch(() => {});
    }
    // Any terminal state can have installed (publish renames before its last rm; recoverBackup runs before the first
    // cancel boundary), and the snapshots may predate the publish: reload all three, inside a settle window.
    if (failed.length > 0) setFailedAway((prev) => [...prev, ...failed]);
    if (settled) {
      setSettledOn((prev) => {
        const next = new Map(prev);
        for (const job of jobs) {
          if (isTerminalJob(job.state))
            next.set(job.slug, { catalog: catalogBody.current, updates: updatesBody.current });
        }
        return next;
      });
      reloadModels();
      reloadCatalog();
      reloadUpdates();
    }
  }, [downloads.data, connection, reloadModels, reloadCatalog, reloadUpdates]);

  // The toast title needs the card's blocker, and `/catalog` (a worker route) can answer after `/downloads`.
  useEffect(() => {
    if (failedAway.length === 0) return;
    if (catalog.data === undefined && catalog.error === undefined) return;
    for (const slug of failedAway) {
      const item = catalogItemOf(catalog.data, slug);
      toast.error(item !== undefined && isUpdateInPlace(item) ? 'Update failed' : 'Download failed', {
        description: slug,
      });
    }
    setFailedAway([]);
  }, [failedAway, catalog.data, catalog.error]);

  const starting = useRef(new Set<string>());
  const install = async (slug: string): Promise<void> => {
    if (starting.current.has(slug)) return;
    starting.current.add(slug);
    setDownloadError(null);
    const startedConnection = connection;
    try {
      const res = await mutate<DownloadStartResponse>('POST', '/downloads', { slug });
      if (getConnectionGeneration() !== startedConnection) return;
      setActive((prev) => ({
        ...prev,
        [slug]: { id: res.id, committing: false, cancelling: false, connection: startedConnection, source: 'local' },
      }));
    } catch (err) {
      setDownloadError({ slug, message: errMessage(err) });
      toast.error('Failed to start download', { description: errMessage(err) });
    } finally {
      starting.current.delete(slug);
    }
  };

  /** Reinstall and Update jobs the server accepted, held until the confirmation hands over (setting `active` would unmount it). */
  const reinstalls = useRef(new Map<string, { id: string; connection: number }>());
  const reinstall = async (item: DownloadItem): Promise<void> => {
    const blocker = item.blocker;
    if (blocker === null) return;
    const startedConnection = connection;
    try {
      const res = await mutate<DownloadStartResponse>('POST', '/downloads', {
        slug: item.slug,
        replaceBlocker: {
          kind: blocker.kind,
          fileCount: blocker.fileCount,
          sizeBytes: blocker.sizeBytes,
          truncated: blocker.truncated,
        },
      });
      reinstalls.current.set(item.slug, { id: res.id, connection: startedConnection });
    } catch (err) {
      reloadCatalog();
      throw err;
    }
  };
  const planUpdate = (item: DownloadItem): Promise<AdoptPlanResponse> =>
    mutate<AdoptPlanResponse>('POST', '/downloads/adopt-plan', { slug: item.slug });
  const updateInPlace = async (item: DownloadItem, token: string): Promise<void> => {
    const startedConnection = connection;
    try {
      const res = await mutate<DownloadStartResponse>('POST', '/downloads', { slug: item.slug, adopt: token });
      reinstalls.current.set(item.slug, { id: res.id, connection: startedConnection });
    } catch (err) {
      reloadCatalog();
      throw err;
    }
  };
  const reinstallStarted = (slug: string): void => {
    const kept = reinstalls.current.get(slug);
    reinstalls.current.delete(slug);
    if (kept === undefined || kept.connection !== getConnectionGeneration()) return;
    setDownloadError(null);
    setActive((prev) =>
      prev[slug]?.id === kept.id
        ? prev
        : {
            ...prev,
            [slug]: { id: kept.id, committing: false, cancelling: false, connection: kept.connection, source: 'local' },
          },
    );
  };

  const onDownloadDone = (item: DownloadItem, keptDir: string | undefined): void => {
    const slug = item.slug;
    toast.success(isUpdateInPlace(item) ? 'Update complete' : 'Download complete', {
      description:
        keptDir === undefined ? slug : `Files that were added or changed in the old folder are in ${keptDir}.`,
    });
    const id = active[slug]?.id;
    setActive((prev) => {
      const next = { ...prev };
      delete next[slug];
      return next;
    });
    // Dismiss the settled job, or a later delete of this model would remount progress on the stale id and replay done.
    if (id !== undefined) {
      void mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(id)}`).catch(() => {});
    }
    setSettledOn((prev) => new Map(prev).set(slug, settledBodies()));
    models.reload();
    catalog.reload();
    updates.reload();
  };

  const onDownloadError = (item: DownloadItem, message: string): void => {
    const slug = item.slug;
    setDownloadError({ slug, message });
    toast.error(isUpdateInPlace(item) ? 'Update failed' : 'Download failed', { description: message });
    // `error` can still have installed: refresh both halves of the update verdict.
    setSettledOn((prev) => new Map(prev).set(slug, settledBodies()));
    models.reload();
    catalog.reload();
    updates.reload();
    const id = active[slug]?.id;
    setActive((prev) => {
      const next = { ...prev };
      delete next[slug];
      return next;
    });
    if (id !== undefined) {
      void mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(id)}`).catch(() => {});
    }
  };

  /** The job stopped without finishing, learned from the STREAM (a cancel from elsewhere, or shutdown). Silent. */
  const onDownloadCancelled = (slug: string): void => {
    setActive((prev) => {
      if (prev[slug] === undefined) return prev;
      const next = { ...prev };
      delete next[slug];
      return next;
    });
    setSettledOn((prev) => new Map(prev).set(slug, settledBodies()));
    models.reload();
    catalog.reload();
    updates.reload();
  };

  const cancel = async (slug: string, id: string): Promise<void> => {
    try {
      await mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(id)}`);
      toast.success('Download cancelled', { description: slug });
      // Mark, never drop: the job emits its terminal frame as it unwinds, and only that frame proves it is done
      // touching the disk. Dropping here would close the stream before it lands.
      setActive((prev) => {
        const job = prev[slug];
        // A queued job settles inside cancel(), so its frame can beat this reply: never resurrect or mark another id.
        if (job === undefined || job.id !== id) return prev;
        return { ...prev, [slug]: { ...job, cancelling: true } };
      });
    } catch (err) {
      toast.error('Failed to cancel download', { description: errMessage(err) });
    }
  };

  // `updates.error` gates it: a failed reload keeps the old data, which would re-offer the pre-install verdict.
  const remoteRevisions = new Map<string, string | null>(
    updates.error !== undefined ? [] : (updates.data?.items ?? []).map((item) => [item.hfRepo, item.remoteRevision]),
  );

  const downloadProps = (item: DownloadItem): CatalogDownloadProps => ({
    item,
    updateAvailable: hasUpdate(item, remoteRevisions),
    settling: settledOn.has(item.slug) && settledOn.get(item.slug)?.catalog === catalog.data,
    updateSettling: settledOn.has(item.slug) && settledOn.get(item.slug)?.updates === updates.data,
    job: active[item.slug],
    onInstall: () => install(item.slug),
    onReinstall: () => reinstall(item),
    onReinstallStarted: () => reinstallStarted(item.slug),
    onPlanUpdate: () => planUpdate(item),
    onUpdateInPlace: (token) => updateInPlace(item, token),
    onDone: (keptDir) => onDownloadDone(item, keptDir),
    onError: (message) => onDownloadError(item, message),
    onCancelled: () => onDownloadCancelled(item.slug),
    onCancel: (id) => void cancel(item.slug, id),
  });

  return {
    active,
    downloadError,
    downloadProps,
    downloadsLoading: downloads.loading,
    downloadsError: downloads.error,
    reloadDownloads: downloads.reload,
    remoteRevisions,
  };
}

/** The download props of every build of `item`, in catalog order, and of its draft. */
export function entryDownloads(
  downloads: Pick<ModelDownloads, 'downloadProps'>,
  item: Pick<CatalogItem, 'builds' | 'draft'>,
): { builds: CatalogDownloadProps[]; draft: CatalogDownloadProps | undefined } {
  return {
    builds: item.builds.map((build) => downloads.downloadProps(build)),
    draft:
      item.draft === undefined
        ? undefined
        : // Drafts carry no tokenizer sidecars of their own, and no other build shares their folder.
          downloads.downloadProps({ ...item.draft, heldBy: null, localAssetsRepo: null, localAssetsRevision: null }),
  };
}
