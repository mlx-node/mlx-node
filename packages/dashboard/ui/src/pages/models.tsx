import { DownloadProgress } from '@/components/download-progress';
import { StatTile } from '@/components/stat-tile';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/skeleton';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { mutate } from '@/lib/api';
import { getConnectionGeneration, subscribeConnection } from '@/lib/connection';
import { formatBytes, formatCount, formatNumber } from '@/lib/format';
import type {
  CancelDownloadResponse,
  CatalogItem,
  CatalogResponse,
  DeleteModelResponse,
  DownloadJob,
  DownloadsResponse,
  DownloadStartResponse,
  LocalModel,
  ModelsResponse,
} from '@/lib/types';
import { useJson } from '@/lib/use-api';
import {
  AlertCircle,
  Check,
  ChevronRight,
  Download,
  FileWarning,
  Inbox,
  Loader2,
  Package,
  Trash2,
  X,
} from 'lucide-react';
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { toast } from 'sonner';

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * A settled job — it holds no in-flight work and the server will never move it
 * out of this state again. Mirrors `isTerminal` in `src/download.ts`, and is
 * deliberately not spelled `state !== 'running'`: `committing` is the brief
 * non-cancellable publish window of a job that is still installing a model.
 */
function isTerminalJob(state: DownloadJob['state']): boolean {
  return state === 'done' || state === 'error' || state === 'cancelled';
}

/**
 * A download this page is following: the one job the server holds for a repo
 * while it is neither settled nor absent. `committing` is carried because the
 * card has to drop its Cancel there — `cancel()` refuses a publishing job, so the
 * button could only ever raise a red toast.
 */
interface ActiveJob {
  id: string;
  committing: boolean;
  /** Runtime generation that produced this id; ids do not survive a reconnect. */
  connection: number;
  /**
   * A POST result is newer than the GET that was already in flight when the
   * user clicked Install. Server entries, by contrast, are reconciled exactly
   * against the next authoritative snapshot.
   */
  source: 'local' | 'server';
}

function QuantBadge({ quant }: { quant: string | null }) {
  if (quant === null) {
    return (
      <Badge variant="outline" className="text-muted-foreground font-normal">
        full precision
      </Badge>
    );
  }
  return (
    <Badge variant="secondary" className="font-mono font-normal">
      {quant}
    </Badge>
  );
}

/**
 * The column set of the local-models table, rendered by the loading branch and
 * the loaded branch from this one definition.
 *
 * Both branches need it, and they must never disagree. A header that only the
 * loaded branch renders is not a cosmetic difference: it appears out of nowhere
 * when the request lands and pushes every row down by its own 40px, which is the
 * jump this page was reported for. Sharing the markup is what makes the two
 * branches structurally incapable of drifting apart again.
 */
function LocalModelsHead() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead>Name</TableHead>
        <TableHead>Family</TableHead>
        <TableHead>Quantization</TableHead>
        <TableHead className="text-right">Size</TableHead>
        <TableHead className="text-right">Context</TableHead>
        <TableHead className="w-0" />
      </TableRow>
    </TableHeader>
  );
}

/**
 * Three placeholder rows in the real table, under the real header.
 *
 * Three because a working install carries a handful of checkpoints, and because
 * a fresh one resolves to the empty state below rather than to rows — three rows
 * sit within half a row of that empty state's own height, so both outcomes land
 * near the space held for them. The exact length cannot be known before the
 * response arrives and that residual is unavoidable; what is fixed here is the
 * header and the per-row box.
 *
 * Every bar is `h-[1lh]` — one line box of the text of the very cell it sits in
 * — and the last cell reserves the delete button's `size-9`, which is what
 * actually sets the row height (36px of button beats 20px of text inside the
 * cell's `p-2`). Sized bars rather than full-width ones because this table uses
 * the browser's automatic column algorithm, so the columns are solved from their
 * content: a row of full-width bars would widen every column and then snap back.
 */
function LocalModelsSkeletonRows() {
  return (
    <TableBody>
      {Array.from({ length: 3 }).map((_, i) => (
        <TableRow key={i}>
          <TableCell className="max-w-[22rem] truncate font-medium">
            <Skeleton className="h-[1lh] w-52" />
          </TableCell>
          <TableCell className="text-muted-foreground">
            <Skeleton className="h-[1lh] w-20" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-[1lh] w-16" />
          </TableCell>
          <TableCell className="text-right tabular-nums">
            <Skeleton className="ml-auto h-[1lh] w-14" />
          </TableCell>
          <TableCell className="text-muted-foreground text-right tabular-nums">
            <Skeleton className="ml-auto h-[1lh] w-14" />
          </TableCell>
          <TableCell className="text-right">
            <Skeleton className="ml-auto size-9" />
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  );
}

export default function Models() {
  const models = useJson<ModelsResponse>('/models');
  const catalog = useJson<CatalogResponse>('/catalog');
  const downloads = useJson<DownloadsResponse>('/downloads');
  const connection = useSyncExternalStore(subscribeConnection, getConnectionGeneration, getConnectionGeneration);

  const [pendingDelete, setPendingDelete] = useState<LocalModel | null>(null);
  const [deleting, setDeleting] = useState(false);
  /** repo → active download job (seeded from the server, added on install). */
  const [active, setActive] = useState<Record<string, ActiveJob>>({});
  // `useJson` is stale-while-revalidate: on a connection bump it intentionally
  // keeps the old body on screen until the replacement runtime answers. Remember
  // that exact object so the reconciliation effect below does not mistake it for
  // the new runtime's authoritative download registry.
  const seenConnection = useRef(connection);
  const staleDownloads = useRef<DownloadsResponse | undefined>(undefined);
  if (seenConnection.current !== connection) {
    seenConnection.current = connection;
    staleDownloads.current = downloads.data;
  }
  // Destructured, and that is load-bearing: `models`/`catalog` are new objects on
  // every render, so naming THEM in the effect's deps below would re-run it every
  // render — a dismiss/reload loop. `reload` is a `useCallback(…, [])`, so it is
  // stable for the life of the hook and the deps array stays honest.
  const { reload: reloadModels } = models;
  const { reload: reloadCatalog } = catalog;

  useEffect(() => {
    const snapshot = downloads.data;
    const jobs = snapshot?.jobs;
    if (jobs === undefined) return;
    // On a reconnect, the first render still carries the previous runtime's
    // stale-while-revalidate body. It is neither authoritative presence nor
    // authoritative absence for this connection.
    if (snapshot === staleDownloads.current) return;

    // Every NONTERMINAL job — the exact complement of the dismiss loop below.
    // Start from the authoritative snapshot rather than merging it into `prev`:
    // job ids belong to one runtime, so omission removes a dead runtime's card
    // and the same repo under a new id must replace it.
    setActive((prev) => {
      const next: Record<string, ActiveJob> = {};
      for (const job of jobs) {
        if (isTerminalJob(job.state)) continue;
        next[job.repo] = {
          id: job.id,
          committing: job.state === 'committing',
          connection,
          source: 'server',
        };
      }
      // The GET above begins in the hook's mount/reconnect effect. A user can
      // click Install while it is in flight and receive a POST result before
      // that older snapshot arrives. Preserve only those locally-started jobs
      // from THIS connection; all server-hydrated and prior-runtime entries are
      // governed by the snapshot.
      for (const [repo, job] of Object.entries(prev)) {
        if (job.source === 'local' && job.connection === connection && next[repo]?.id !== job.id) {
          next[repo] = job;
        }
      }
      return next;
    });
    // A job that settled while this page was unmounted got no DownloadProgress
    // callback, so nothing sent its terminal-dismiss DELETE: it sits in the
    // server registry for the life of the process and Overview keeps counting a
    // "completed" download beside a model that is simply installed. Dismiss it
    // here, exactly as onDownloadDone/onDownloadError would have.
    //
    // A FAILED job is announced first. It is the one terminal state whose
    // outcome is invisible on this page — its card reverts to a plain Install,
    // as if the download had never been started — so a silent dismiss would be
    // the only trace of the failure. `done` and `cancelled` stay quiet: the
    // models table, the Installed card and the user's own Cancel already say it.
    // The failure TEXT lives only in the SSE stream (`/api/downloads` carries no
    // message), so this names the repo the way the toasts below do.
    //
    // Racing is safe: terminal is final on the server, so a job that reads
    // terminal here cannot be running by the time the DELETE lands, and a
    // duplicate dismiss (a second dashboard tab) answers 404 and is swallowed.
    let settled = false;
    for (const job of jobs) {
      if (!isTerminalJob(job.state)) continue;
      if (job.state === 'error') toast.error('Download failed', { description: job.repo });
      settled = true;
      void mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(job.id)}`).catch(() => {});
    }
    // The catalog and models snapshots may predate the publish this job just
    // finished: the runner sets `done` strictly AFTER `publish()` renames the
    // staging dir into place, so a mount that spans that instant reads
    // `present: false` from one request and `done` from the other. Nothing else
    // repairs it — neither query polls, and a terminal job gets no
    // `DownloadProgress`, so `onDownloadDone`'s reload never runs.
    //
    // Any terminal state, not just `done`. A model can be installed under an
    // `error` too: `publish()` renames staging into final and only THEN removes
    // the backup, and that `rm` is inside the job's try — so an EPERM there
    // reports a failed download for a model that is on disk. `recoverBackup`
    // likewise restores a crashed publish before the job can be cancelled.
    if (settled) {
      reloadModels();
      reloadCatalog();
    }
  }, [downloads.data, connection, reloadModels, reloadCatalog]);

  const install = async (repo: string): Promise<void> => {
    const startedConnection = connection;
    try {
      const res = await mutate<DownloadStartResponse>('POST', '/downloads', { repo });
      // A late response from the client that was just replaced cannot name a
      // job in the new runtime. `connectDashboardApi` normally rejects it while
      // closing the old client; this is the final generation guard.
      if (getConnectionGeneration() !== startedConnection) return;
      setActive((prev) => ({
        ...prev,
        [repo]: { id: res.id, committing: false, connection: startedConnection, source: 'local' },
      }));
    } catch (err) {
      toast.error('Failed to start download', { description: errMessage(err) });
    }
  };

  const onDownloadDone = (repo: string): void => {
    toast.success('Download complete', { description: repo });
    const id = active[repo]?.id;
    setActive((prev) => {
      const next = { ...prev };
      delete next[repo];
      return next;
    });
    // The job has settled terminally on the server; drop it from `active` and
    // dismiss it via DELETE (mirroring onDownloadError/cancel). Otherwise the
    // completed job id lingers, and a later delete of this model would remount
    // DownloadProgress on that stale job — replaying `done` → a spurious "complete"
    // toast + a Complete/Cancel card instead of an Install button.
    if (id !== undefined) {
      void mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(id)}`).catch(() => {});
    }
    models.reload();
    catalog.reload();
  };

  const onDownloadError = (repo: string, message: string): void => {
    toast.error('Download failed', { description: message });
    const id = active[repo]?.id;
    setActive((prev) => {
      const next = { ...prev };
      delete next[repo];
      return next;
    });
    // The job has settled terminally on the server; dismiss it via DELETE so
    // GET /api/downloads stops returning the failed row and a retry (Install)
    // starts clean. Best-effort — a failed eviction just leaves an inert row the
    // next dismiss/restart can still clear. Mirrors `cancel`'s DELETE (no abort).
    if (id !== undefined) {
      void mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(id)}`).catch(() => {});
    }
  };

  /**
   * The job stopped without finishing, learned from the STREAM rather than from a
   * DELETE this page issued. `cancel()` below already clears `active` when the
   * user's own cancel resolves, so this is the case it cannot see: the server
   * answers a cancel while the job is still `running` and lets `processJob` emit
   * the terminal frame as it unwinds, so a cancel from a second tab — or from
   * `shutdown()` aborting the in-flight job — arrives here having clicked nothing.
   * Without this the card renders "Cancelled" beside a live Cancel button and
   * never recovers: no query polls, and `downloads` is never reloaded.
   *
   * Silent, unlike `onDownloadError`. A cancel is not a failure, and whoever
   * issued it already got their own confirmation.
   */
  const onDownloadCancelled = (repo: string): void => {
    setActive((prev) => {
      if (prev[repo] === undefined) return prev;
      const next = { ...prev };
      delete next[repo];
      return next;
    });
  };

  const cancel = async (repo: string, id: string): Promise<void> => {
    try {
      await mutate<CancelDownloadResponse>('DELETE', `/downloads/${encodeURIComponent(id)}`);
      toast.success('Download cancelled', { description: repo });
      // Mirror onDownloadError's reset: drop the active job so the card reverts to
      // the Install state. Cancel only aborts the job (its staging cleans up); the
      // shared HF cache is deliberately left intact for `mlx download` to resume.
      setActive((prev) => {
        const next = { ...prev };
        delete next[repo];
        return next;
      });
    } catch (err) {
      toast.error('Failed to cancel download', { description: errMessage(err) });
    }
  };

  const confirmDelete = async (): Promise<void> => {
    if (pendingDelete === null) return;
    const { name } = pendingDelete;
    setDeleting(true);
    try {
      await mutate<DeleteModelResponse>('DELETE', `/models/${encodeURIComponent(name)}`);
      toast.success('Model deleted', { description: name });
      setPendingDelete(null);
      models.reload();
      catalog.reload();
    } catch (err) {
      toast.error('Failed to delete model', { description: errMessage(err) });
    } finally {
      setDeleting(false);
    }
  };

  const localModels = models.data?.models ?? [];
  const warnings = models.data?.warnings ?? [];
  const modelsDir = models.data?.dir ?? '';
  const totalBytes = localModels.reduce((sum, m) => sum + m.sizeBytes, 0);
  const catalogItems = (catalog.data?.items ?? []).filter((item) => !item.hidden);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Models</h1>
        <p className="text-muted-foreground text-sm">Local checkpoints and downloads from the recommended list.</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <StatTile
          label="Installed"
          icon={Package}
          value={models.loading ? <Skeleton className="h-[1lh] w-12" /> : formatCount(localModels.length)}
          sub={
            models.loading ? (
              // Two lines, because the loaded sub is two lines: the directory
              // below is drawn whenever the server names one, which is every
              // case except an unconfigured install. Reserving one line here
              // would leave the tile 22px short and shove the table below it
              // down the moment the response lands — and the response is what
              // says whether the second line exists at all, so the shape has to
              // be guessed. Guess the common one.
              <>
                <Skeleton className="block h-[1lh] w-24" />
                <span className="mt-0.5 block font-mono text-[11px]">
                  <Skeleton className="h-[1lh] w-40 max-w-full" />
                </span>
              </>
            ) : (
              <>
                <span className="block">{formatBytes(totalBytes)} on disk</span>
                {modelsDir !== '' && (
                  // The models directory is configurable (`--models-dir`), so name
                  // it — a bare count doesn't say where these checkpoints live.
                  <span className="mt-0.5 block truncate font-mono text-[11px]" title={modelsDir}>
                    {modelsDir}
                  </span>
                )}
              </>
            )
          }
        />
        <StatTile
          label="Recommended"
          icon={Download}
          value={catalog.loading ? <Skeleton className="h-[1lh] w-12" /> : formatCount(catalogItems.length)}
          sub={
            catalog.loading ? (
              <Skeleton className="h-[1lh] w-28" />
            ) : (
              // Count `present` (loadable on disk, incl. a renamed dashboard
              // install / a CLI install), matching what each card labels
              // "Installed" — not the dashboard-owned `installed` marker.
              `${formatCount(catalogItems.filter((i) => i.present).length)} installed`
            )
          }
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Local models</CardTitle>
          <CardDescription>Checkpoints discovered in the models directory.</CardDescription>
        </CardHeader>
        <CardContent>
          {warnings.length > 0 && (
            // Skipped directories are routine noise (GGUF folders, source trees)
            // and can run to dozens of lines, so keep the list folded behind a
            // count and let anyone who cares expand it.
            <Collapsible className="text-muted-foreground bg-muted/50 mb-4 rounded-md border">
              <CollapsibleTrigger className="hover:bg-muted/70 group flex w-full items-center gap-2 rounded-md p-3 text-xs transition-colors data-[state=open]:rounded-b-none">
                <FileWarning className="size-4 shrink-0" aria-hidden />
                <span className="font-medium">
                  {formatCount(warnings.length)} {warnings.length === 1 ? 'directory' : 'directories'} skipped
                </span>
                <ChevronRight
                  className="ml-auto size-3.5 shrink-0 transition-transform group-data-[state=open]:rotate-90"
                  aria-hidden
                />
              </CollapsibleTrigger>
              <CollapsibleContent className="px-3 pb-3 text-xs">
                <ul className="space-y-0.5 pl-6">
                  {warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </CollapsibleContent>
            </Collapsible>
          )}

          {models.error ? (
            <div className="text-destructive flex items-center gap-2 text-sm">
              <AlertCircle className="size-4 shrink-0" aria-hidden />
              {models.error.message}
            </div>
          ) : models.loading ? (
            <Table>
              <LocalModelsHead />
              <LocalModelsSkeletonRows />
            </Table>
          ) : localModels.length === 0 ? (
            <div className="text-muted-foreground flex flex-col items-center gap-2 py-10 text-sm">
              <Inbox className="size-6" aria-hidden />
              No local models yet — install one from the recommended list below.
            </div>
          ) : (
            <Table>
              <LocalModelsHead />
              <TableBody>
                {localModels.map((model) => (
                  <TableRow key={model.name}>
                    <TableCell className="max-w-[22rem] truncate font-medium" title={model.path}>
                      {model.name}
                    </TableCell>
                    <TableCell className="text-muted-foreground">{model.modelType}</TableCell>
                    <TableCell>
                      <QuantBadge quant={model.quant} />
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{formatBytes(model.sizeBytes)}</TableCell>
                    <TableCell className="text-muted-foreground text-right tabular-nums">
                      {model.contextWindow === null ? '—' : formatNumber(model.contextWindow)}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-muted-foreground hover:text-destructive"
                        aria-label={`Delete ${model.name}`}
                        onClick={() => setPendingDelete(model)}
                      >
                        <Trash2 className="size-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <div>
        <h2 className="text-lg font-semibold tracking-tight">Recommended models</h2>
        <p className="text-muted-foreground text-sm">
          Curated checkpoints — install downloads them into the models directory.
        </p>
      </div>

      {catalog.error ? (
        <Card>
          <CardContent className="text-destructive flex items-center gap-2 py-6 text-sm">
            <AlertCircle className="size-4 shrink-0" aria-hidden />
            {catalog.error.message}
          </CardContent>
        </Card>
      ) : catalog.loading ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <CatalogCardSkeleton key={i} />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {catalogItems.map((item) => (
            <CatalogCard
              key={item.hfRepo}
              item={item}
              job={active[item.hfRepo]}
              onInstall={() => install(item.hfRepo)}
              onDone={() => onDownloadDone(item.hfRepo)}
              onError={(message) => onDownloadError(item.hfRepo, message)}
              onCancelled={() => onDownloadCancelled(item.hfRepo)}
              onCancel={(id) => void cancel(item.hfRepo, id)}
            />
          ))}
        </div>
      )}

      <Dialog open={pendingDelete !== null} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete model?</DialogTitle>
            <DialogDescription>
              This permanently removes <span className="text-foreground font-medium">{pendingDelete?.name}</span> (
              {pendingDelete !== null ? formatBytes(pendingDelete.sizeBytes) : ''}) from disk. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" disabled={deleting}>
                Cancel
              </Button>
            </DialogClose>
            <Button variant="destructive" onClick={() => void confirmDelete()} disabled={deleting}>
              {deleting && <Loader2 className="size-4 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

/**
 * A placeholder that is a real {@link CatalogCard} with its text taken out:
 * same `Card`/`CardHeader`/`CardContent` wrappers, same `gap-4`, same padding,
 * same install button box — only the leaves are grey.
 *
 * It replaces one fixed 176px block, a height that stood for nothing in
 * particular. The card these stand in for measures 170px (24px of `py-6`, a 16px
 * `leading-none` title, `gap-1.5`, a 20px description line, `gap-4`, a 16px
 * `text-xs` meta row, `space-y-3`, the 36px button, 24px of `py-6`) and grows
 * another 20px the moment a description wraps to a second line, so no constant
 * was ever going to be right. Built from the same wrappers there is no constant
 * to get wrong: the height is whatever the loaded card computes to.
 *
 * Three of them because the served catalog holds exactly three visible entries.
 * One residual is left standing: the single `isDefault` entry carries a badge
 * that makes its title row 6px taller, and grid items stretch, so the row it
 * lands in grows by that much. Painting a badge on all three to absorb it would
 * claim a default on cards that have none.
 */
function CatalogCardSkeleton() {
  return (
    <Card className="gap-4">
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base">
            <Skeleton className="h-[1lh] w-40" />
          </CardTitle>
        </div>
        <CardDescription>
          <Skeleton className="h-[1lh] w-full" />
        </CardDescription>
      </CardHeader>
      <CardContent className="mt-auto space-y-3">
        <div className="flex items-center justify-between gap-2 text-xs">
          <Skeleton className="h-[1lh] w-48" />
          <Skeleton className="h-[1lh] w-12" />
        </div>
        {/* The button box, not a button: `h-9` is what `Button`'s default size
            resolves to, and it is the tallest thing in the card's content. */}
        <Skeleton className="h-9 w-full" />
      </CardContent>
    </Card>
  );
}

interface CatalogCardProps {
  item: CatalogItem;
  job: ActiveJob | undefined;
  onInstall: () => void;
  onDone: () => void;
  onError: (message: string) => void;
  onCancelled: () => void;
  onCancel: (id: string) => void;
}

function CatalogCard({ item, job, onInstall, onDone, onError, onCancelled, onCancel }: CatalogCardProps) {
  // Gate on `present` (loadable checkpoint on disk), not `installed` (dashboard
  // marker): a model installed via the `mlx download` CLI / wizard is present but
  // unowned, so offering Install would refuse to overwrite it and fail.
  const downloading = job !== undefined && !item.present;

  return (
    <Card className="gap-4">
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base">{item.label}</CardTitle>
          {item.isDefault === true && (
            <Badge variant="secondary" className="font-normal">
              default
            </Badge>
          )}
        </div>
        <CardDescription>{item.description}</CardDescription>
      </CardHeader>
      <CardContent className="mt-auto space-y-3">
        <div className="text-muted-foreground flex items-center justify-between text-xs">
          <span className="truncate font-mono" title={item.hfRepo}>
            {item.hfRepo}
          </span>
          <span className="tabular-nums">~{item.sizeGb} GB</span>
        </div>
        {downloading && job !== undefined ? (
          <div className="space-y-2">
            <DownloadProgress id={job.id} onDone={onDone} onError={onError} onCancelled={onCancelled} />
            {/* No Cancel while publishing: `cancel()` refuses a `committing` job
                and the route 404s, so the button could only report a failure for
                an install that then succeeds anyway. The progress bar stays. */}
            {!job.committing && (
              <Button
                variant="ghost"
                size="sm"
                className="text-muted-foreground hover:text-destructive w-full"
                onClick={() => onCancel(job.id)}
              >
                <X className="size-4" />
                Cancel
              </Button>
            )}
          </div>
        ) : item.present ? (
          <Button variant="outline" className="w-full" disabled>
            <Check className="size-4" />
            Installed
          </Button>
        ) : item.blockedByForeignDir ? (
          // The download's ownership preflight refuses an occupied directory it
          // did not write, so an Install here can only ever raise a red toast.
          // Name the directory in the way and the one step that clears it.
          <div className="space-y-2">
            <Button variant="outline" className="w-full" disabled>
              <AlertCircle className="size-4" />
              Needs cleanup
            </Button>
            <p className="text-muted-foreground text-xs">
              <span className="font-mono">{item.slug}</span> already exists and was not created by the dashboard. Remove
              it (or delete it below, if it is listed) to install.
            </p>
          </div>
        ) : (
          <Button className="w-full" onClick={onInstall}>
            <Download className="size-4" />
            Install
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
