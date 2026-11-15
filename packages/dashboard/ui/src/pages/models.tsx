import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react';
import { useSearchParams } from 'react-router-dom';

import { StatGroup, StatTile } from '../components/arc-x/stat-tile/stat-tile';
import { SwapText } from '../components/arc-x/swap-text/swap-text';
import { Alert } from '../components/arc/alert/alert';
import { Badge } from '../components/arc/badge/badge';
import { Button } from '../components/arc/button/button';
import { EmptyState } from '../components/arc/empty-state/empty-state';
import { formatBytes, formatBytesParts, formatCount } from '../lib/format';
import { useHomeDir } from '../lib/home-dir';
import { useDataVersion, useRuntimeLive } from '../lib/live';
import { joinModelUsage } from '../lib/model-usage';
import { tildePath } from '../lib/paths';
import { alignedRange } from '../lib/range';
import { toast } from '../lib/toast';
import type {
  CatalogItem,
  CatalogResponse,
  CatalogUpdatesResponse,
  MetricsSummary,
  MetricsTimeseries,
  ModelsResponse,
  RuntimeLive,
} from '../lib/types';
import { useJson } from '../lib/use-api';
import { useCommands, type AppCommand } from '../shell/commands';
import { Page, PageHeader, PageSection } from '../shell/page';
import { CatalogCard, CatalogCardPlaceholder, CatalogDownloadAction, CatalogGrid } from './models/catalog';
import { LocalModels, type UsageView } from './models/local-models';
import { ModelDrawer } from './models/model-drawer';
import { useModelDownloads } from './models/use-model-downloads';
import Onboarding from './onboarding';

export type { ModelDownloadError } from './models/use-model-downloads';

const CHECK_NOTE_MS = 6_000;

function loadedNow(live: RuntimeLive | undefined): { value: string | null; hint: string } {
  if (live === undefined) return { value: null, hint: 'no live status yet' };
  if (live.status === 'stopped') return { value: '0', hint: 'inference stopped' };
  const names = new Set<string>();
  let inFlight = 0;
  for (const process of live.processes) {
    if (process.status !== 'running') continue;
    for (const model of process.models) {
      names.add(model.name);
      inFlight += model.inFlight;
    }
  }
  return {
    value: formatCount(names.size),
    hint: inFlight > 0 ? `${formatCount(inFlight)} turn${inFlight === 1 ? '' : 's'} in flight` : 'idle',
  };
}

/** The body without the checkpoint the runtime just deleted, so its row and its bytes leave at once. */
export function withoutEntry(body: ModelsResponse, name: string): ModelsResponse {
  return {
    ...body,
    models: body.models.filter((model) => model.name !== name),
    ...(body.companions === undefined
      ? {}
      : { companions: body.companions.filter((companion) => companion.name !== name) }),
  };
}

/** After a check the user asked for: one short line, gone after a few seconds. */
function useCheckNote(updates: { refreshing: boolean; loading: boolean; error: Error | undefined }) {
  const [asked, setAsked] = useState(false);
  const [note, setNote] = useState<'ok' | 'error' | null>(null);
  const busy = updates.refreshing || updates.loading;
  if (asked && !busy) {
    setAsked(false);
    setNote(updates.error === undefined ? 'ok' : 'error');
  }
  useEffect(() => {
    if (note === null) return;
    const handle = setTimeout(() => setNote(null), CHECK_NOTE_MS);
    return () => clearTimeout(handle);
  }, [note]);
  return { note, ask: () => setAsked(true) };
}

export default function Models({ onboarding = false }: { onboarding?: boolean }): ReactElement {
  const models = useJson<ModelsResponse>('/models');
  const catalog = useJson<CatalogResponse>('/catalog');
  // A separate request on purpose: it dials Hugging Face and may fail; a failure only means no update badge.
  const updates = useJson<CatalogUpdatesResponse>('/catalog/updates');
  const downloads = useModelDownloads(models, catalog, updates);

  if (onboarding) {
    return (
      <Onboarding
        catalog={catalog}
        models={models}
        activeRepos={Object.keys(downloads.active)}
        downloadError={downloads.downloadError}
        downloadsLoading={downloads.downloadsLoading}
        downloadsError={downloads.downloadsError}
        reloadDownloads={downloads.reloadDownloads}
        renderDownload={(item) => (
          <CatalogDownloadAction
            {...downloads.downloadProps(item)}
            prominent
            installLabel="Download model"
            settlingLabel="Checking download…"
          />
        )}
      />
    );
  }
  return <ModelsPage models={models} catalog={catalog} updates={updates} downloads={downloads} />;
}

interface ModelsPageProps {
  models: ReturnType<typeof useJson<ModelsResponse>>;
  catalog: ReturnType<typeof useJson<CatalogResponse>>;
  updates: ReturnType<typeof useJson<CatalogUpdatesResponse>>;
  downloads: ReturnType<typeof useModelDownloads>;
}

function ModelsPage({ models, catalog, updates, downloads }: ModelsPageProps): ReactElement {
  const dataVersion = useDataVersion();
  const range = useMemo(() => alignedRange('30d'), [dataVersion]);
  const scope = `from=${String(range.from)}&to=${String(range.to)}`;
  const summary = useJson<MetricsSummary>(`/metrics/summary?${scope}`, { live: true });
  const series = useJson<MetricsTimeseries>(`/metrics/timeseries?${scope}&bucket=day&groupBy=model`, {
    live: true,
  });
  const live = useRuntimeLive();
  const [params, setParams] = useSearchParams();
  const focus = params.get('focus');
  const deletedRef = useRef(false);
  const check = useCheckNote(updates);

  const localModels = models.data?.models ?? [];
  const companions = models.data?.companions ?? [];
  const usage = useMemo(() => joinModelUsage(models.data?.models ?? [], summary.data), [models.data, summary.data]);
  const usageFailed = summary.error !== undefined && summary.data === undefined;
  const view: UsageView = {
    usage,
    points: series.data?.points,
    starts: range.starts,
    loading: (summary.data === undefined && !usageFailed) || (series.data === undefined && series.error === undefined),
    failed: usageFailed || (series.error !== undefined && series.data === undefined),
    usageFailed,
    retry: () => {
      summary.reload();
      series.reload();
    },
  };
  const focused = focus === null ? undefined : localModels.find((model) => model.name === focus);
  const catalogItems = (catalog.data?.items ?? []).filter((item) => !item.hidden);
  const present = catalogItems.filter((item) => item.present).length;
  const activeCount = Object.keys(downloads.active).length;
  const host = models.data?.host;

  const setFocus = (name: string | null): void => {
    setParams(
      (current) => {
        const next = new URLSearchParams(current);
        if (name === null) next.delete('focus');
        else next.set('focus', name);
        return next;
      },
      { replace: true },
    );
  };
  const unknownFocus = focus !== null && models.data !== undefined && focused === undefined;
  useEffect(() => {
    if (unknownFocus) setFocus(null);
  });

  const { reload: reloadCatalog } = catalog;
  const { reload: reloadUpdates } = updates;
  const onDeleted = (name: string): void => {
    if (focus === name) {
      deletedRef.current = true;
      setFocus(null);
    }
    models.patch((body) => withoutEntry(body, name));
    reloadCatalog();
    reloadUpdates();
  };

  const checkUpdates = (): void => {
    check.ask();
    reloadUpdates();
  };
  const dir = models.data?.dir ?? '';
  const home = useHomeDir();
  const settlingRepos = catalogItems
    .filter((item) => {
      const props = downloads.downloadProps(item);
      return props.settling || props.updateSettling;
    })
    .map((item) => item.hfRepo)
    .join('\n');
  const commands = useMemo((): AppCommand[] => {
    const list: AppCommand[] = [];
    for (const item of catalogItems) {
      const props = downloads.downloadProps(item);
      if (props.job !== undefined || props.settling) continue;
      if (!item.present && !item.blockedByForeignDir) {
        list.push({
          id: `models:install:${item.hfRepo}`,
          label: `Install ${item.label}`,
          group: 'Models',
          run: props.onInstall,
        });
      } else if (item.present && props.updateAvailable && !props.updateSettling) {
        list.push({
          id: `models:update:${item.hfRepo}`,
          label: `Update ${item.label}`,
          group: 'Models',
          run: props.onInstall,
        });
      }
    }
    list.push({ id: 'models:check-updates', label: 'Check for updates', group: 'Models', run: checkUpdates });
    if (dir !== '') {
      list.push({
        id: 'models:copy-dir',
        label: 'Copy models directory path',
        group: 'Models',
        run: () =>
          void navigator.clipboard.writeText(dir).then(
            () => toast.success('Path copied', { description: tildePath(dir, home) }),
            () => toast.error('Failed to copy to clipboard'),
          ),
      });
    }
    return list;
  }, [catalog.data, updates.data, updates.error, downloads.active, settlingRepos, dir, home]);
  useCommands(commands);

  const totalBytes = [...localModels, ...companions].reduce((sum, model) => sum + model.sizeBytes, 0);
  const disk = formatBytesParts(totalBytes);
  const diskHint =
    host !== undefined && host.diskFreeBytes !== null
      ? `${formatBytes(host.diskFreeBytes)} free`
      : companions.length > 0
        ? 'including companions'
        : undefined;
  let used = 0;
  for (const model of localModels) if ((usage.get(model.name)?.turns ?? 0) > 0) used += 1;
  const loaded = loadedNow(live);
  const modelsLoading = models.loading;
  const modelsError = models.error !== undefined && models.data === undefined ? 'Failed to load models' : undefined;
  const usageTile =
    usageFailed || summary.error !== undefined
      ? { value: null, hint: 'usage unavailable' }
      : {
          value: `${formatCount(used)} of ${formatCount(localModels.length)}`,
          hint: used < localModels.length ? `${formatCount(localModels.length - used)} not used` : 'all in use',
        };

  const jumpToDownload = (): void => {
    const section = document.getElementById('recommended-models');
    section?.scrollIntoView({ block: 'start' });
    const card = section?.querySelector<HTMLElement>('[data-downloading] a');
    card?.focus();
  };

  return (
    <Page busy={models.refreshing || catalog.refreshing}>
      <PageHeader
        title="Models"
        description="Checkpoints on this Mac and the recommended list."
        meta={
          activeCount > 0 ? (
            <a
              href="#recommended-models"
              onClick={(event) => {
                event.preventDefault();
                jumpToDownload();
              }}
            >
              <Badge tone="info" size="sm" dot>
                {`${formatCount(activeCount)} downloading`}
              </Badge>
            </a>
          ) : undefined
        }
      />

      <StatGroup label="Models summary">
        <StatTile
          label="Installed"
          value={models.data === undefined ? null : formatCount(localModels.length)}
          hint={companions.length > 0 ? `${formatCount(companions.length)} companion weights` : undefined}
          loading={modelsLoading}
          error={modelsError}
        />
        <StatTile
          label="On disk"
          value={models.data === undefined ? null : disk.value}
          unit={disk.unit}
          hint={diskHint}
          loading={modelsLoading}
          error={modelsError}
        />
        <StatTile
          label="Used in 30 days"
          value={usageTile.value}
          hint={usageTile.hint}
          loading={modelsLoading || (summary.data === undefined && summary.error === undefined)}
        />
        <StatTile label="Loaded now" value={loaded.value} hint={loaded.hint} to="/engine" />
      </StatGroup>

      {models.error !== undefined && models.data !== undefined ? (
        <Alert tone="warning" title="Could not refresh the model list">
          <span className="grid justify-items-start gap-2">
            <span>{models.error.message}</span>
            <Button variant="secondary" size="sm" onClick={models.reload}>
              Try again
            </Button>
          </span>
        </Alert>
      ) : null}

      <LocalModels
        data={models.data}
        loading={modelsLoading}
        error={models.data === undefined ? models.error : undefined}
        retry={models.reload}
        usage={view}
        live={live}
        focus={focus}
        onDeleted={onDeleted}
      />

      <div id="recommended-models" className="min-w-0">
        <PageSection
          title="Recommended models"
          description="Curated checkpoints. Install downloads into the models directory."
          actions={
            <span className="flex flex-wrap items-center justify-end gap-2">
              {catalog.data !== undefined ? (
                <span className="text-body text-secondary tabular-nums">
                  {`${formatCount(present)} of ${formatCount(catalogItems.length)} installed`}
                </span>
              ) : null}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                loading={updates.refreshing || (updates.loading && check.note === null)}
                onClick={checkUpdates}
              >
                Check for updates
              </Button>
              {check.note !== null ? (
                <span className={check.note === 'error' ? 'text-body text-danger' : 'text-body text-secondary'}>
                  <SwapText text={check.note === 'ok' ? 'Checked just now' : 'Could not reach Hugging Face'} />
                </span>
              ) : null}
            </span>
          }
        >
          {catalog.error !== undefined ? (
            <Alert tone="danger" title="Could not load the recommended list">
              <span className="grid justify-items-start gap-2">
                <span>{catalog.error.message}</span>
                <Button variant="secondary" size="sm" onClick={catalog.reload}>
                  Try again
                </Button>
              </span>
            </Alert>
          ) : null}
          {catalog.data === undefined ? (
            catalog.error !== undefined ? null : (
              <CatalogGrid>
                {[0, 1, 2].map((index) => (
                  <CatalogCardPlaceholder key={index} withDraft={index === 0} />
                ))}
              </CatalogGrid>
            )
          ) : catalogItems.length === 0 ? (
            <EmptyState
              title="Nothing to recommend yet"
              description="New models appear here when they are published."
            />
          ) : (
            <CatalogGrid>
              {catalogItems.map((item) => (
                <CatalogCardEntry
                  key={item.hfRepo}
                  item={item}
                  downloads={downloads}
                  host={host}
                  detailsTo={
                    localModels.some((model) => model.name === item.slug)
                      ? `/models?focus=${encodeURIComponent(item.slug)}`
                      : undefined
                  }
                />
              ))}
            </CatalogGrid>
          )}
        </PageSection>
      </div>

      <ModelDrawer
        model={focused ?? null}
        usage={usage}
        catalog={catalog.data?.items ?? []}
        remoteRevisions={downloads.remoteRevisions}
        live={live}
        range={range}
        onClose={() => setFocus(null)}
        onDeleted={onDeleted}
        deletedRef={deletedRef}
      />
    </Page>
  );
}

function CatalogCardEntry(props: {
  item: CatalogItem;
  downloads: ReturnType<typeof useModelDownloads>;
  host: ModelsResponse['host'] | undefined;
  detailsTo: string | undefined;
}): ReactElement {
  const { item, downloads } = props;
  const download = downloads.downloadProps(item);
  const draft =
    item.draft === undefined
      ? undefined
      : // Drafts carry no tokenizer sidecars of their own.
        downloads.downloadProps({ ...item.draft, localAssetsRepo: null, localAssetsRevision: null });
  const busy = download.job !== undefined || draft?.job !== undefined;
  return (
    <div className="min-w-0" data-downloading={busy ? '' : undefined}>
      <CatalogCard {...download} item={item} draftDownload={draft} host={props.host} detailsTo={props.detailsTo} />
    </div>
  );
}
