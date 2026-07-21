import { DownloadProgress } from '@/components/download-progress';
import { StatTile } from '@/components/stat-tile';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
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
import { formatBytes, formatCount, formatNumber } from '@/lib/format';
import type {
  CatalogItem,
  CatalogResponse,
  DeleteModelResponse,
  DownloadsResponse,
  DownloadStartResponse,
  LocalModel,
  ModelsResponse,
} from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { AlertCircle, Check, Download, FileWarning, Inbox, Loader2, Package, Trash2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { toast } from 'sonner';

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
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

export default function Models() {
  const models = useJson<ModelsResponse>('/models');
  const catalog = useJson<CatalogResponse>('/catalog');
  const downloads = useJson<DownloadsResponse>('/downloads');

  const [pendingDelete, setPendingDelete] = useState<LocalModel | null>(null);
  const [deleting, setDeleting] = useState(false);
  /** repo → active download job id (seeded from the server, added on install). */
  const [active, setActive] = useState<Record<string, string>>({});

  useEffect(() => {
    const jobs = downloads.data?.jobs;
    if (jobs === undefined) return;
    setActive((prev) => {
      const next = { ...prev };
      for (const job of jobs) {
        if (job.state === 'running' && next[job.repo] === undefined) next[job.repo] = job.id;
      }
      return next;
    });
  }, [downloads.data]);

  const install = async (repo: string): Promise<void> => {
    try {
      const res = await mutate<DownloadStartResponse>('POST', '/downloads', { repo });
      setActive((prev) => ({ ...prev, [repo]: res.id }));
    } catch (err) {
      toast.error('Failed to start download', { description: errMessage(err) });
    }
  };

  const onDownloadDone = (repo: string): void => {
    toast.success('Download complete', { description: repo });
    models.reload();
    catalog.reload();
  };

  const onDownloadError = (repo: string, message: string): void => {
    toast.error('Download failed', { description: message });
    setActive((prev) => {
      const next = { ...prev };
      delete next[repo];
      return next;
    });
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
          value={models.loading ? <Skeleton className="h-8 w-12" /> : formatCount(localModels.length)}
          sub={models.loading ? <Skeleton className="h-4 w-24" /> : `${formatBytes(totalBytes)} on disk`}
        />
        <StatTile
          label="Recommended"
          icon={Download}
          value={catalog.loading ? <Skeleton className="h-8 w-12" /> : formatCount(catalogItems.length)}
          sub={
            catalog.loading ? (
              <Skeleton className="h-4 w-28" />
            ) : (
              `${formatCount(catalogItems.filter((i) => i.installed).length)} installed`
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
            <div className="text-muted-foreground bg-muted/50 mb-4 flex items-start gap-2 rounded-md border p-3 text-xs">
              <FileWarning className="mt-0.5 size-4 shrink-0" aria-hidden />
              <ul className="space-y-0.5">
                {warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}

          {models.error ? (
            <div className="text-destructive flex items-center gap-2 text-sm">
              <AlertCircle className="size-4 shrink-0" aria-hidden />
              {models.error.message}
            </div>
          ) : models.loading ? (
            <div className="space-y-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : localModels.length === 0 ? (
            <div className="text-muted-foreground flex flex-col items-center gap-2 py-10 text-sm">
              <Inbox className="size-6" aria-hidden />
              No local models yet — install one from the recommended list below.
            </div>
          ) : (
            <Table>
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
            <Skeleton key={i} className="h-44 w-full rounded-xl" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {catalogItems.map((item) => (
            <CatalogCard
              key={item.hfRepo}
              item={item}
              jobId={active[item.hfRepo]}
              onInstall={() => install(item.hfRepo)}
              onDone={() => onDownloadDone(item.hfRepo)}
              onError={(message) => onDownloadError(item.hfRepo, message)}
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

interface CatalogCardProps {
  item: CatalogItem;
  jobId: string | undefined;
  onInstall: () => void;
  onDone: () => void;
  onError: (message: string) => void;
}

function CatalogCard({ item, jobId, onInstall, onDone, onError }: CatalogCardProps) {
  const downloading = jobId !== undefined && !item.installed;

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
        {downloading && jobId !== undefined ? (
          <DownloadProgress id={jobId} onDone={onDone} onError={onError} />
        ) : item.installed ? (
          <Button variant="outline" className="w-full" disabled>
            <Check className="size-4" />
            Installed
          </Button>
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
