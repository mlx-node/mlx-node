import { useEffect, useRef, useState, useSyncExternalStore } from 'react';

import type { DownloadDoneResult } from '../../components/download-progress';
import { mutate } from '../../lib/api';
import { hasUpdate } from '../../lib/catalog';
import { getConnectionGeneration, subscribeConnection } from '../../lib/connection';
import { errMessage } from '../../lib/errors';
import { toast } from '../../lib/toast';
import type {
  AdoptPlanResponse,
  CancelDownloadResponse,
  CatalogBlocker,
  CatalogBuildItem,
  CatalogItem,
  CatalogResponse,
  CatalogUpdatesResponse,
  DownloadJob,
  DownloadJobKind,
  DownloadStartResponse,
  DownloadsResponse,
  HostResponse,
  LocatePlanResponse,
  ModelsResponse,
} from '../../lib/types';
import { type AsyncState, useJson } from '../../lib/use-api';

const FAILED_TITLE: Record<DownloadJobKind, string> = {
  install: 'Download failed',
  cleanup: 'Download failed',
  update: 'Update failed',
  locate: 'Locate failed',
};

/**
 * The toast of a job that ended well. `blocker` is what was in the way when the job started; `folder` is the folder a
 * Locate this page started uses.
 */
export function jobDoneToast(
  kind: DownloadJobKind,
  slug: string,
  blocker: Pick<CatalogBlocker, 'kind' | 'linkToFile'> | null,
  result: DownloadDoneResult,
  folder: string | undefined,
): { title: string; description: string } {
  const { keptDir, removed } = result;
  if (kind === 'locate')
    return { title: 'Model located', description: folder === undefined ? slug : `Uses the files in ${folder}.` };
  const title = kind === 'update' ? 'Update complete' : 'Download complete';
  if (kind === 'cleanup') {
    const link = blocker?.kind === 'symlink';
    const unchanged = blocker?.linkToFile === true ? '' : ' Its folder was not changed.';
    if (keptDir !== undefined) {
      return {
        title,
        description: link
          ? `The link ${slug} could not be removed. It is in ${keptDir}.${unchanged}`
          : `The old ${slug} could not go to the Trash. It is in ${keptDir}. Nothing was deleted.`,
      };
    }
    if (removed === 'link') return { title, description: `The link ${slug} was removed.${unchanged}` };
    if (removed === 'trash') {
      return { title, description: `The old ${blocker?.kind === 'file' ? 'file' : 'folder'} ${slug} is in the Trash.` };
    }
    return { title, description: slug };
  }
  return {
    title,
    description: keptDir === undefined ? slug : `Files that were added or changed in the old folder are in ${keptDir}.`,
  };
}

/** What a refused Update is kept by: the numbers of the blocker it was refused for. */
function blockerKey(blocker: Pick<CatalogBlocker, 'kind' | 'fileCount' | 'sizeBytes'>): string {
  return `${blocker.kind}:${String(blocker.fileCount)}:${String(blocker.sizeBytes)}`;
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
  kind: DownloadJobKind;
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
  /** The entry in the way that a cleanup this page started was confirmed for; the catalog drops it once it reloads. */
  blocker: CatalogBlocker | null;
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
  | 'sizeGb'
  | 'located'
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
  /** Update refused the folder in `item.blocker` for a reason a cleanup solves, and the folder is as it was then. */
  refused: boolean;
  /** The server's sentence of that refusal, while {@link refused}. */
  refusedSentence: string | undefined;
  onInstall: () => Promise<void>;
  /** Clean up & update over `item.blocker`: posts the numbers the user saw and keeps the job id. Rejects with the server text. */
  onCleanup: () => Promise<void>;
  /** After a confirmation's success pill: the card follows the kept job. */
  onConfirmStarted: () => void;
  /** Update of a folder that holds a model the app did not create: loads what it would do. Rejects with the server text. */
  onPlanUpdate: () => Promise<AdoptPlanResponse>;
  /** The plan was refused for a reason a cleanup solves: the card offers Clean up & update with `message`. */
  onUpdateRefused: (message: string) => void;
  /** Starts the Update the plan with `token` describes and keeps the job id. Rejects with the server text. */
  onUpdateInPlace: (token: string) => Promise<void>;
  /** What Locate would use from the folder at `path`; writes nothing. Rejects with the server text. */
  onPlanLocate: (path: string) => Promise<LocatePlanResponse>;
  /** Starts the Locate that `plan` describes and keeps the job id. Rejects with the server text. */
  onLocate: (plan: LocatePlanResponse) => Promise<void>;
  /** Reads the folders again: a disk that was away may be back. */
  onRecheck: () => void;
  /**
   * Update of a located build: downloads the whole build into the models folder and keeps the job id until the
   * confirmation hands over. Rejects with the server text.
   */
  onReplaceLocated: () => Promise<void>;
  onDone: (result: DownloadDoneResult) => void;
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
 * settle window after a job ends, and cancel marking. `models`, `catalog`, `updates` and `host` are the page's own
 * queries; a job that ends reloads them all.
 */
export function useModelDownloads(
  models: AsyncState<ModelsResponse>,
  catalog: AsyncState<CatalogResponse>,
  updates: AsyncState<CatalogUpdatesResponse>,
  host: AsyncState<HostResponse>,
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
  /** Folders whose Update plan was refused for a reason a cleanup solves, for this visit: the sentence and the blocker. */
  const [refusals, setRefusals] = useState<ReadonlyMap<string, { message: string; key: string }>>(() => new Map());
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
  const { reload: reloadHost } = host;

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
        const before = prev[job.slug];
        next[job.slug] = {
          id: job.id,
          kind: job.kind,
          committing: job.state === 'committing',
          cancelling: false,
          connection,
          source: 'server',
          blocker: before?.id === job.id ? before.blocker : null,
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
    for (const job of jobs) {
      if (!isTerminalJob(job.state)) continue;
      if (job.state === 'error') toast.error(FAILED_TITLE[job.kind], { description: job.slug });
      settled = true;
      void mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(job.id)}`).catch(() => {});
    }
    // Any terminal state can have installed (publish renames before its last step; recovery runs before the first
    // cancel boundary), and the snapshots may predate the publish: reload them all, inside a settle window.
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
      reloadHost();
    }
  }, [downloads.data, connection, reloadModels, reloadCatalog, reloadUpdates, reloadHost]);

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
        [slug]: {
          id: res.id,
          kind: res.kind,
          committing: false,
          cancelling: false,
          connection: startedConnection,
          source: 'local',
          blocker: null,
        },
      }));
    } catch (err) {
      setDownloadError({ slug, message: errMessage(err) });
      toast.error('Failed to start download', { description: errMessage(err) });
    } finally {
      starting.current.delete(slug);
    }
  };

  /** Jobs a confirmation started, held until the confirmation hands over (setting `active` would unmount it). */
  const confirmed = useRef(
    new Map<string, { id: string; kind: DownloadJobKind; connection: number; blocker: CatalogBlocker | null }>(),
  );
  /** Starts the job `body` asks for and keeps it for {@link confirmStarted}; a refusal reloads the catalog. */
  const startConfirmed = async (
    slug: string,
    body: Record<string, unknown>,
    blocker: CatalogBlocker | null = null,
  ): Promise<void> => {
    const startedConnection = connection;
    try {
      const res = await mutate<DownloadStartResponse>('POST', '/downloads', { slug, ...body });
      confirmed.current.set(slug, { id: res.id, kind: res.kind, connection: startedConnection, blocker });
    } catch (err) {
      reloadCatalog();
      throw err;
    }
  };
  const cleanup = (item: DownloadItem): Promise<void> => {
    const blocker = item.blocker;
    if (blocker === null) return Promise.resolve();
    const { kind, fileCount, sizeBytes, truncated, linkTarget } = blocker;
    return startConfirmed(
      item.slug,
      { replaceBlocker: { kind, fileCount, sizeBytes, truncated, linkTarget } },
      blocker,
    );
  };
  const planUpdate = (item: DownloadItem): Promise<AdoptPlanResponse> =>
    mutate<AdoptPlanResponse>('POST', '/downloads/adopt-plan', { slug: item.slug });
  const updateRefused = (item: DownloadItem, message: string): void => {
    const blocker = item.blocker;
    if (blocker === null) return;
    setRefusals((prev) => new Map(prev).set(item.slug, { message, key: blockerKey(blocker) }));
  };
  const updateInPlace = (item: DownloadItem, token: string): Promise<void> =>
    startConfirmed(item.slug, { adopt: token });
  /** The folder each Locate job this page started uses, by slug, for its toast. */
  const locates = useRef(new Map<string, string>());
  const planLocate = (item: DownloadItem, path: string): Promise<LocatePlanResponse> =>
    mutate<LocatePlanResponse>('POST', '/downloads/locate-plan', { slug: item.slug, path });
  const locate = async (item: DownloadItem, plan: LocatePlanResponse): Promise<void> => {
    await startConfirmed(item.slug, { locate: plan.token });
    locates.current.set(item.slug, plan.folder);
  };
  const replaceLocated = (item: DownloadItem): Promise<void> => startConfirmed(item.slug, {});
  const recheck = (): void => {
    reloadModels();
    reloadCatalog();
    reloadHost();
  };
  const confirmStarted = (slug: string): void => {
    const kept = confirmed.current.get(slug);
    confirmed.current.delete(slug);
    if (kept === undefined || kept.connection !== getConnectionGeneration()) return;
    setDownloadError(null);
    setActive((prev) =>
      prev[slug]?.id === kept.id
        ? prev
        : {
            ...prev,
            [slug]: {
              id: kept.id,
              kind: kept.kind,
              committing: false,
              cancelling: false,
              connection: kept.connection,
              source: 'local',
              blocker: kept.blocker,
            },
          },
    );
  };

  const reloadAll = (): void => {
    models.reload();
    catalog.reload();
    updates.reload();
    host.reload();
  };

  const onDownloadDone = (item: DownloadItem, result: DownloadDoneResult): void => {
    const slug = item.slug;
    const folder = locates.current.get(slug);
    locates.current.delete(slug);
    const job = active[slug];
    const done = jobDoneToast(job?.kind ?? 'install', slug, job?.blocker ?? item.blocker, result, folder);
    toast.success(done.title, { description: done.description });
    const id = job?.id;
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
    reloadAll();
  };

  const onDownloadError = (item: DownloadItem, message: string): void => {
    const slug = item.slug;
    locates.current.delete(slug);
    const kind = active[slug]?.kind ?? 'install';
    if (kind !== 'locate') setDownloadError({ slug, message });
    toast.error(FAILED_TITLE[kind], { description: message });
    // `error` can still have installed: refresh both halves of the update verdict.
    setSettledOn((prev) => new Map(prev).set(slug, settledBodies()));
    reloadAll();
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
    locates.current.delete(slug);
    setActive((prev) => {
      if (prev[slug] === undefined) return prev;
      const next = { ...prev };
      delete next[slug];
      return next;
    });
    setSettledOn((prev) => new Map(prev).set(slug, settledBodies()));
    reloadAll();
  };

  const cancel = async (slug: string, id: string): Promise<void> => {
    const kind = active[slug]?.kind;
    try {
      await mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(id)}`);
      toast.success(kind === 'locate' ? 'Locate cancelled' : 'Download cancelled', { description: slug });
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

  const downloadProps = (item: DownloadItem): CatalogDownloadProps => {
    const refusal = refusals.get(item.slug);
    const refused = refusal !== undefined && item.blocker !== null && refusal.key === blockerKey(item.blocker);
    return {
      item,
      updateAvailable: hasUpdate(item, remoteRevisions),
      settling: settledOn.has(item.slug) && settledOn.get(item.slug)?.catalog === catalog.data,
      updateSettling: settledOn.has(item.slug) && settledOn.get(item.slug)?.updates === updates.data,
      job: active[item.slug],
      refused,
      refusedSentence: refused ? refusal.message : undefined,
      onInstall: () => install(item.slug),
      onCleanup: () => cleanup(item),
      onConfirmStarted: () => confirmStarted(item.slug),
      onPlanUpdate: () => planUpdate(item),
      onUpdateRefused: (message) => updateRefused(item, message),
      onUpdateInPlace: (token) => updateInPlace(item, token),
      onPlanLocate: (path) => planLocate(item, path),
      onLocate: (plan) => locate(item, plan),
      onRecheck: recheck,
      onReplaceLocated: () => replaceLocated(item),
      onDone: (result) => onDownloadDone(item, result),
      onError: (message) => onDownloadError(item, message),
      onCancelled: () => onDownloadCancelled(item.slug),
      onCancel: (id) => void cancel(item.slug, id),
    };
  };

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
          downloads.downloadProps({
            ...item.draft,
            heldBy: null,
            localAssetsRepo: null,
            localAssetsRevision: null,
            located: null,
          }),
  };
}
