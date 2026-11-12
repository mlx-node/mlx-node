import { Trash2 } from 'lucide-react';
import type { ReactElement, ReactNode } from 'react';

import { ConfirmPopover, type ConfirmContent } from '../../components/arc-x/confirm-popover/confirm-popover';
import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { Disclosure } from '../../components/arc-x/disclosure/disclosure';
import { MiniSpark } from '../../components/arc-x/mini-spark/mini-spark';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Panel } from '../../components/arc-x/panel/panel';
import { PathText } from '../../components/arc-x/path-text/path-text';
import { Placeholder } from '../../components/arc-x/placeholder/placeholder';
import { SplitBar, type BarSegment } from '../../components/arc-x/split-bar/split-bar';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { CopyButton } from '../../components/arc/copy-button/copy-button';
import { DashboardApiError, mutate } from '../../lib/api';
import { formatBytes, formatCount, formatNumber, formatRelativeTime, formatTps } from '../../lib/format';
import { modelColor } from '../../lib/model-colors';
import { dailyTurns, matchesModel, usageKeys, type ModelUsage } from '../../lib/model-usage';
import { otherLabel } from '../../lib/timeseries';
import { toast } from '../../lib/toast';
import {
  LOW_SAMPLE_N,
  type DeleteModelResponse,
  type LocalModel,
  type MetricsTimeseries,
  type ModelsResponse,
  type RuntimeLive,
} from '../../lib/types';

const NO_USAGE = 'No recorded turns for this model in the last 30 days';
const USAGE_FAILED = 'Usage could not be loaded';
const DAILY_FAILED = 'Daily turns could not be loaded';
const SEGMENT_CAP = 8;

/** The key a folder's usage and color go by: the busiest served id, else its first usage key. */
export function usageKeyOf(model: Pick<LocalModel, 'name' | 'servedIds'>, usage: ModelUsage | undefined): string {
  return usage?.key ?? usageKeys(model)[0]!;
}

export interface LiveState {
  /** The running process's entry for this folder, or undefined. */
  resident: RuntimeLive['processes'][number]['models'][number] | undefined;
  loading: boolean;
}

/** What the live snapshot says about one folder: resident in a running process, or loading into one. */
export function liveStateOf(model: Pick<LocalModel, 'name' | 'servedIds'>, live: RuntimeLive | undefined): LiveState {
  let resident: LiveState['resident'];
  let loading = false;
  for (const process of live?.processes ?? []) {
    if (process.status !== 'running') continue;
    for (const entry of process.models) {
      if (resident === undefined && matchesModel(model, entry.name)) resident = entry;
    }
    if (process.loading !== null && matchesModel(model, process.loading.model)) loading = true;
  }
  return { resident, loading };
}

export function LiveBadge({ state }: { state: LiveState }): ReactElement | null {
  if (state.resident !== undefined) {
    const inFlight = state.resident.inFlight;
    return (
      <Badge
        tone="success"
        size="sm"
        dot
        title={inFlight > 0 ? `${inFlight} turn${inFlight === 1 ? '' : 's'} in flight` : undefined}
      >
        Loaded
      </Badge>
    );
  }
  if (state.loading) {
    return (
      <Badge tone="info" size="sm" dot>
        Loading
      </Badge>
    );
  }
  return null;
}

export interface DeleteTarget {
  name: string;
  sizeBytes: number;
  companion?: boolean;
}

export interface DeleteHandlers {
  /** The model is resident in a running process. */
  loaded?: boolean;
  /** Runs after the success pill: reload and, for the drawer's model, close it. */
  onDeleted: (name: string) => void;
}

/**
 * Names whose DELETE answered not-found, kept until that delete's `onDone`. A module set and not a closure flag,
 * because the page rebuilds this content on every render and the popover runs the newest `onDone`.
 */
const goneBeforeDelete = new Set<string>();

/**
 * The confirmation of a model or companion delete, shown at the row, the companion row and the drawer footer. A
 * not-found answer means the entry left another way (Finder, the CLI, a lost answer), so it runs the success path and
 * the row leaves.
 */
export function deleteContent(target: DeleteTarget, handlers: DeleteHandlers): ConfirmContent {
  const companion = target.companion === true;
  return {
    title: companion ? 'Delete companion weights?' : 'Delete model?',
    description: (
      <>
        This permanently removes <span className="font-medium text-foreground">{target.name}</span> (
        {formatBytes(target.sizeBytes)}) from disk. This cannot be undone.
      </>
    ),
    notice:
      !companion && handlers.loaded === true
        ? 'This model is loaded. It keeps working until inference restarts. After that it is gone.'
        : undefined,
    confirmLabel: 'Delete',
    workingLabel: 'Deleting…',
    successLabel: 'Deleted',
    tone: 'danger',
    errorTitle: companion ? 'Failed to delete companion' : 'Failed to delete model',
    onConfirm: async () => {
      goneBeforeDelete.delete(target.name);
      try {
        await mutate<DeleteModelResponse>('DELETE', `/models/${encodeURIComponent(target.name)}`);
      } catch (err) {
        if (!(err instanceof DashboardApiError) || err.code !== 'E_NOT_FOUND') throw err;
        goneBeforeDelete.add(target.name);
      }
    },
    onDone: () => {
      const gone = goneBeforeDelete.delete(target.name);
      const what = companion ? 'Companion' : 'Model';
      toast.success(gone ? `${what} was already gone` : `${what} deleted`, { description: target.name });
      handlers.onDeleted(target.name);
    },
  };
}

/** Usage of the 30-day window as the table reads it. */
export interface UsageView {
  usage: ReadonlyMap<string, ModelUsage>;
  points: MetricsTimeseries['points'] | undefined;
  starts: readonly number[];
  loading: boolean;
  failed: boolean;
  /** The per-model summary failed: every usage cell is unknown, not zero. */
  usageFailed: boolean;
  retry: () => void;
}

export interface LocalModelsProps {
  data: ModelsResponse | undefined;
  loading: boolean;
  error: Error | undefined;
  retry: () => void;
  usage: UsageView;
  live: RuntimeLive | undefined;
  focus: string | null;
  onDeleted: (name: string) => void;
}

function UsageCell({ view, children }: { view: UsageView; children: ReactNode }): ReactElement {
  if (view.loading) return <Placeholder width="70%" />;
  return <>{children}</>;
}

function storageSegments(models: readonly LocalModel[], view: UsageView, companionBytes: number): BarSegment[] {
  const ordered = [...models].sort((a, b) => b.sizeBytes - a.sizeBytes);
  const shown = ordered.length > SEGMENT_CAP ? ordered.slice(0, SEGMENT_CAP - 1) : ordered;
  const segments: BarSegment[] = shown.map((model) => {
    const color = modelColor(usageKeyOf(model, view.usage.get(model.name)));
    return {
      key: model.name,
      label: model.name,
      value: model.sizeBytes,
      color: color.color,
      pattern: color.pattern === 'alt' ? 'hatch' : 'solid',
    };
  });
  const rest = ordered.slice(shown.length);
  if (rest.length > 0) {
    segments.push({
      key: 'other',
      label: otherLabel(rest.length),
      value: rest.reduce((sum, model) => sum + model.sizeBytes, 0),
      color: 'var(--model-other)',
    });
  }
  if (companionBytes > 0) {
    segments.push({
      key: 'companions',
      label: 'Companion weights',
      value: companionBytes,
      color: 'var(--model-other)',
      pattern: 'hatch',
    });
  }
  return segments;
}

/** Storage bar, the table of checkpoint folders, companion weights and skipped directories. */
export function LocalModels(props: LocalModelsProps): ReactElement {
  const { data, usage: view, live } = props;
  const models = data?.models ?? [];
  const companions = data?.companions ?? [];
  const warnings = data?.warnings ?? [];
  const dir = data?.dir ?? '';
  const companionBytes = companions.reduce((sum, companion) => sum + companion.sizeBytes, 0);
  const settled = !props.loading && props.error === undefined;
  const noUsage = <span title={view.usageFailed ? USAGE_FAILED : NO_USAGE}>—</span>;

  const columns: DataColumn<LocalModel>[] = [
    {
      id: 'model',
      header: 'Model',
      sortValue: (model) => model.name,
      cell: (model) => {
        const state = liveStateOf(model, live);
        return (
          <span className="flex min-w-0 items-center gap-2" title={model.path}>
            <ModelBadge
              name={model.name}
              colorKey={usageKeyOf(model, view.usage.get(model.name))}
              showLogo
              maxChars={30}
            />
            <LiveBadge state={state} />
          </span>
        );
      },
    },
    {
      id: 'family',
      header: 'Family',
      width: 96,
      minTableWidth: 860,
      sortValue: (model) => model.modelType,
      cell: (model) => <span className="text-secondary">{model.modelType}</span>,
    },
    {
      id: 'quant',
      header: 'Quant',
      width: 104,
      sortValue: (model) => model.quant ?? '',
      cell: (model) =>
        model.quant === null ? (
          <span className="text-secondary">full precision</span>
        ) : (
          <Badge size="sm" outline>
            {model.quant}
          </Badge>
        ),
    },
    {
      id: 'context',
      header: 'Context',
      align: 'end',
      width: 84,
      minTableWidth: 800,
      sortValue: (model) => model.contextWindow,
      cell: (model) => (
        <span title="Trained context window, tokens">
          {model.contextWindow === null ? '—' : formatNumber(model.contextWindow)}
        </span>
      ),
    },
    {
      id: 'size',
      header: 'Size',
      align: 'end',
      width: 80,
      sortValue: (model) => model.sizeBytes,
      cell: (model) => formatBytes(model.sizeBytes),
    },
    {
      id: 'activity',
      header: 'Activity (30 d)',
      align: 'end',
      width: 132,
      minTableWidth: 600,
      sortValue: (model) => view.usage.get(model.name)?.turns ?? null,
      cell: (model) => {
        const usage = view.usage.get(model.name);
        return (
          <UsageCell view={view}>
            {usage === undefined ? (
              noUsage
            ) : view.points === undefined ? (
              <span className="inline-block min-w-[5ch] text-end" title={DAILY_FAILED}>
                {formatCount(usage.turns)}
              </span>
            ) : (
              <span className="inline-flex items-center justify-end gap-2">
                <MiniSpark
                  data={dailyTurns(view.points, usageKeys(model), view.starts)}
                  label={`${model.name}, turns per day, last 30 days`}
                  color={modelColor(usage.key).color}
                  width={56}
                  height={20}
                />
                <span className="inline-block min-w-[5ch] text-end">{formatCount(usage.turns)}</span>
              </span>
            )}
          </UsageCell>
        );
      },
    },
    {
      id: 'decode',
      header: 'Decode',
      align: 'end',
      width: 96,
      minTableWidth: 700,
      sortValue: (model) => view.usage.get(model.name)?.decodeTps.p50 ?? null,
      cell: (model) => {
        const usage = view.usage.get(model.name);
        const tps = usage?.decodeTps;
        return (
          <UsageCell view={view}>
            {tps === undefined || tps.p50 === null ? (
              noUsage
            ) : (
              <span className="whitespace-nowrap">
                {formatTps(tps.p50)} <span className="text-secondary">tok/s</span>
                {tps.n < LOW_SAMPLE_N ? <span className="text-muted"> n={tps.n}</span> : null}
              </span>
            )}
          </UsageCell>
        );
      },
    },
    {
      id: 'last',
      header: 'Last used',
      align: 'end',
      width: 92,
      minTableWidth: 940,
      sortValue: (model) => view.usage.get(model.name)?.lastTs ?? null,
      cell: (model) => {
        const usage = view.usage.get(model.name);
        return (
          <UsageCell view={view}>
            {usage === undefined ? (
              noUsage
            ) : (
              <span title={new Date(usage.lastTs).toISOString()}>{formatRelativeTime(usage.lastTs)}</span>
            )}
          </UsageCell>
        );
      },
    },
    {
      id: 'actions',
      header: '',
      align: 'end',
      width: 68,
      cell: (model) => (
        <ConfirmPopover
          trigger={{
            kind: 'icon',
            label: `Delete ${model.name}`,
            icon: <Trash2 size={16} strokeWidth={1.75} aria-hidden="true" />,
            size: 'sm',
          }}
          align="end"
          {...deleteContent(model, {
            loaded: liveStateOf(model, live).resident !== undefined,
            onDeleted: props.onDeleted,
          })}
        />
      ),
    },
  ];

  return (
    <Panel
      title="Local models"
      padding="none"
      actions={
        dir === '' ? undefined : (
          <span className="flex min-w-0 items-center gap-1">
            <PathText path={dir} maxChars={44} className="truncate text-caption text-secondary" />
            <CopyButton value={dir} iconOnly variant="plain" label="Copy models directory" />
          </span>
        )
      }
    >
      <div className="grid min-w-0 gap-3">
        {props.loading ? (
          <div className="px-4">
            <Placeholder height={8} radius="pill" />
          </div>
        ) : settled && models.length > 0 ? (
          <div className="px-4">
            <SplitBar
              label="Disk use by model"
              size="thin"
              legend={false}
              segments={storageSegments(models, view, companionBytes)}
              formatValue={formatBytes}
            />
          </div>
        ) : null}
        <DataTable
          caption="Local models"
          frame={false}
          columns={columns}
          rows={models}
          rowKey={(model) => model.name}
          rowAttrs={(model) => ({ 'data-model': model.name, 'data-selected': String(model.name === props.focus) })}
          rowTo={(model) => `/models?focus=${encodeURIComponent(model.name)}`}
          defaultSort={{ id: 'size', direction: 'desc' }}
          loading={props.loading}
          loadingRows={3}
          error={props.error === undefined ? undefined : { message: props.error.message, retry: props.retry }}
          empty={{ title: 'No local models yet', description: 'Install one from the recommended list below.' }}
          footer={
            view.failed && settled && models.length > 0 ? (
              <span className="flex flex-wrap items-center gap-2">
                Usage could not be loaded.
                <Button type="button" variant="ghost" size="sm" onClick={view.retry}>
                  Try again
                </Button>
              </span>
            ) : undefined
          }
        />
        {settled && companions.length > 0 ? (
          <Disclosure title={`Companion weights (${formatCount(companions.length)})`}>
            <div className="grid gap-2 px-4 pb-3">
              <p className="m-0 text-caption text-secondary">Optional weights used with a model.</p>
              <ul className="m-0 grid list-none gap-1 p-0">
                {companions.map((companion) => (
                  <li key={companion.name} className="flex min-w-0 items-center gap-3 text-body">
                    <span className="min-w-0 flex-1 truncate" title={companion.path}>
                      {companion.name}
                    </span>
                    <span className="shrink-0 text-caption text-secondary tabular-nums">
                      {formatBytes(companion.sizeBytes)}
                    </span>
                    <ConfirmPopover
                      trigger={{
                        kind: 'icon',
                        label: `Delete companion ${companion.name}`,
                        icon: <Trash2 size={16} strokeWidth={1.75} aria-hidden="true" />,
                        size: 'sm',
                      }}
                      align="end"
                      {...deleteContent({ ...companion, companion: true }, { onDeleted: props.onDeleted })}
                    />
                  </li>
                ))}
              </ul>
            </div>
          </Disclosure>
        ) : null}
        {settled && warnings.length > 0 ? (
          <Disclosure
            title={`${formatCount(warnings.length)} ${warnings.length === 1 ? 'directory' : 'directories'} skipped`}
          >
            <ul className="m-0 grid gap-0.5 px-4 pb-3 pl-8 text-caption text-secondary">
              {warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </Disclosure>
        ) : null}
      </div>
    </Panel>
  );
}
