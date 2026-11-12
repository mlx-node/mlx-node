import { useEffect, useRef, useState, useSyncExternalStore } from 'react';

import { mutate } from '../../lib/api';
import { hasUpdate } from '../../lib/catalog';
import { getConnectionGeneration, subscribeConnection } from '../../lib/connection';
import { errMessage } from '../../lib/errors';
import { toast } from '../../lib/toast';
import type {
  CancelDownloadResponse,
  CatalogItem,
  CatalogResponse,
  CatalogUpdatesResponse,
  DownloadJob,
  DownloadStartResponse,
  DownloadsResponse,
  ModelsResponse,
} from '../../lib/types';
import { type AsyncState, useJson } from '../../lib/use-api';

export interface ModelDownloadError {
  repo: string;
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

/** A download this page follows: the one job the server holds for a repo while it is neither settled nor absent. */
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
  CatalogItem,
  | 'hfRepo'
  | 'installed'
  | 'localRevision'
  | 'assetsRepo'
  | 'localAssetsRepo'
  | 'localAssetsRevision'
  | 'present'
  | 'blockedByForeignDir'
  | 'blocker'
  | 'slug'
>;

export interface CatalogDownloadProps {
  installLabel?: string;
  settlingLabel?: string;
  /** Onboarding: the one loud button of its surface. */
  prominent?: boolean;
  item: DownloadItem;
  /** Upstream has different bytes at this repo than the local marker records. */
  updateAvailable: boolean;
  /** This repo's job settled but the catalog body it invalidated has not arrived: every field is pre-download. */
  settling: boolean;
  /** The same, for the separately fetched remote shas `updateAvailable` is computed against. */
  updateSettling: boolean;
  job: ActiveJob | undefined;
  onInstall: () => Promise<void>;
  /** Reinstall over `item.blocker`: posts the numbers the user saw and keeps the job id. Rejects with the server text. */
  onReinstall: () => Promise<void>;
  /** After the confirmation's success pill: the card follows the kept job. */
  onReinstallStarted: () => void;
  onDone: () => void;
  onError: (message: string) => void;
  onCancelled: () => void;
  onCancel: (id: string) => void;
}

export interface ModelDownloads {
  /** repo → the job this page follows. */
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
   * For a repo whose job just settled, the bodies that were current when it did. `setActive` clears synchronously while
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
        next[job.repo] = {
          id: job.id,
          committing: job.state === 'committing',
          cancelling: false,
          connection,
          source: 'server',
        };
      }
      for (const [repo, job] of Object.entries(prev)) {
        if (job.source === 'local' && job.connection === connection && next[repo]?.id !== job.id) {
          next[repo] = job;
        }
      }
      return next;
    });
    // A job that settled while this page was unmounted got no terminal-dismiss DELETE. Dismiss it here; announce a
    // failure first (its card reverts to Install, so the toast is its only trace). A duplicate dismiss answers 404.
    let settled = false;
    for (const job of jobs) {
      if (!isTerminalJob(job.state)) continue;
      if (job.state === 'error') toast.error('Download failed', { description: job.repo });
      settled = true;
      void mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(job.id)}`).catch(() => {});
    }
    // Any terminal state can have installed (publish renames before its last rm; recoverBackup runs before the first
    // cancel boundary), and the snapshots may predate the publish: reload all three, inside a settle window.
    if (settled) {
      setSettledOn((prev) => {
        const next = new Map(prev);
        for (const job of jobs) {
          if (isTerminalJob(job.state))
            next.set(job.repo, { catalog: catalogBody.current, updates: updatesBody.current });
        }
        return next;
      });
      reloadModels();
      reloadCatalog();
      reloadUpdates();
    }
  }, [downloads.data, connection, reloadModels, reloadCatalog, reloadUpdates]);

  const starting = useRef(new Set<string>());
  const install = async (repo: string): Promise<void> => {
    if (starting.current.has(repo)) return;
    starting.current.add(repo);
    setDownloadError(null);
    const startedConnection = connection;
    try {
      const res = await mutate<DownloadStartResponse>('POST', '/downloads', { repo });
      if (getConnectionGeneration() !== startedConnection) return;
      setActive((prev) => ({
        ...prev,
        [repo]: { id: res.id, committing: false, cancelling: false, connection: startedConnection, source: 'local' },
      }));
    } catch (err) {
      setDownloadError({ repo, message: errMessage(err) });
      toast.error('Failed to start download', { description: errMessage(err) });
    } finally {
      starting.current.delete(repo);
    }
  };

  /** Reinstall jobs the server accepted, held until the confirmation hands over (setting `active` would unmount it). */
  const reinstalls = useRef(new Map<string, { id: string; connection: number }>());
  const reinstall = async (item: DownloadItem): Promise<void> => {
    const blocker = item.blocker;
    if (blocker === null) return;
    const startedConnection = connection;
    try {
      const res = await mutate<DownloadStartResponse>('POST', '/downloads', {
        repo: item.hfRepo,
        replaceBlocker: {
          kind: blocker.kind,
          fileCount: blocker.fileCount,
          sizeBytes: blocker.sizeBytes,
          truncated: blocker.truncated,
        },
      });
      reinstalls.current.set(item.hfRepo, { id: res.id, connection: startedConnection });
    } catch (err) {
      reloadCatalog();
      throw err;
    }
  };
  const reinstallStarted = (repo: string): void => {
    const kept = reinstalls.current.get(repo);
    reinstalls.current.delete(repo);
    if (kept === undefined || kept.connection !== getConnectionGeneration()) return;
    setDownloadError(null);
    setActive((prev) =>
      prev[repo]?.id === kept.id
        ? prev
        : {
            ...prev,
            [repo]: { id: kept.id, committing: false, cancelling: false, connection: kept.connection, source: 'local' },
          },
    );
  };

  const onDownloadDone = (repo: string): void => {
    toast.success('Download complete', { description: repo });
    const id = active[repo]?.id;
    setActive((prev) => {
      const next = { ...prev };
      delete next[repo];
      return next;
    });
    // Dismiss the settled job, or a later delete of this model would remount progress on the stale id and replay done.
    if (id !== undefined) {
      void mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(id)}`).catch(() => {});
    }
    setSettledOn((prev) => new Map(prev).set(repo, settledBodies()));
    models.reload();
    catalog.reload();
    updates.reload();
  };

  const onDownloadError = (repo: string, message: string): void => {
    setDownloadError({ repo, message });
    toast.error('Download failed', { description: message });
    // `error` can still have installed: refresh both halves of the update verdict.
    setSettledOn((prev) => new Map(prev).set(repo, settledBodies()));
    models.reload();
    catalog.reload();
    updates.reload();
    const id = active[repo]?.id;
    setActive((prev) => {
      const next = { ...prev };
      delete next[repo];
      return next;
    });
    if (id !== undefined) {
      void mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(id)}`).catch(() => {});
    }
  };

  /** The job stopped without finishing, learned from the STREAM (a cancel from elsewhere, or shutdown). Silent. */
  const onDownloadCancelled = (repo: string): void => {
    setActive((prev) => {
      if (prev[repo] === undefined) return prev;
      const next = { ...prev };
      delete next[repo];
      return next;
    });
    setSettledOn((prev) => new Map(prev).set(repo, settledBodies()));
    models.reload();
    catalog.reload();
    updates.reload();
  };

  const cancel = async (repo: string, id: string): Promise<void> => {
    try {
      await mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(id)}`);
      toast.success('Download cancelled', { description: repo });
      // Mark, never drop: the job emits its terminal frame as it unwinds, and only that frame proves it is done
      // touching the disk. Dropping here would close the stream before it lands.
      setActive((prev) => {
        const job = prev[repo];
        // A queued job settles inside cancel(), so its frame can beat this reply: never resurrect or mark another id.
        if (job === undefined || job.id !== id) return prev;
        return { ...prev, [repo]: { ...job, cancelling: true } };
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
    settling: settledOn.has(item.hfRepo) && settledOn.get(item.hfRepo)?.catalog === catalog.data,
    updateSettling: settledOn.has(item.hfRepo) && settledOn.get(item.hfRepo)?.updates === updates.data,
    job: active[item.hfRepo],
    onInstall: () => install(item.hfRepo),
    onReinstall: () => reinstall(item),
    onReinstallStarted: () => reinstallStarted(item.hfRepo),
    onDone: () => onDownloadDone(item.hfRepo),
    onError: (message) => onDownloadError(item.hfRepo, message),
    onCancelled: () => onDownloadCancelled(item.hfRepo),
    onCancel: (id) => void cancel(item.hfRepo, id),
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
