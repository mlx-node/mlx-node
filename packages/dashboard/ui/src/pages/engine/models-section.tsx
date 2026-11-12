import type { ReactElement } from 'react';

import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { LiveLabel } from '../../components/arc-x/live-label/live-label';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Panel } from '../../components/arc-x/panel/panel';
import { Meter } from '../../components/arc-x/split-bar/split-bar';
import { SwapText } from '../../components/arc-x/swap-text/swap-text';
import { Badge } from '../../components/arc/badge/badge';
import {
  formatBytes,
  formatClock,
  formatCount,
  formatDuration,
  formatNumber,
  formatPercent,
  formatRelativeTime,
} from '../../lib/format';
import { findLocalModel } from '../../lib/model-usage';
import type { LocalModel, RuntimeProcessLive } from '../../lib/types';

type LoadedModel = RuntimeProcessLive['models'][number];

type Row = { kind: 'model'; model: LoadedModel } | { kind: 'loading'; name: string; startedAt: number };

function rowName(row: Row): string {
  return row.kind === 'model' ? row.model.name : row.name;
}

export function ModelsSection({
  process,
  now,
  locals,
  modelUrl,
}: {
  process: RuntimeProcessLive;
  now: number;
  locals: readonly LocalModel[] | undefined;
  /** The current URL with `model` set (selects the model in the pool section). */
  modelUrl: (name: string) => string;
}): ReactElement {
  const rows: Row[] = process.models.map((model) => ({ kind: 'model', model }));
  if (process.loading !== null)
    rows.unshift({ kind: 'loading', name: process.loading.model, startedAt: process.loading.startedAt });
  const live = process.status === 'running';

  const columns: DataColumn<Row>[] = [
    {
      id: 'model',
      header: 'Model',
      width: '26%',
      truncate: true,
      cell: (row) => {
        const name = rowName(row);
        const local = findLocalModel(locals, name);
        if (row.kind === 'loading') {
          return (
            <span className="inline-flex min-w-0 flex-wrap items-center gap-2">
              <ModelBadge name={name} size="sm" fit="width" />
              <Badge tone="info" dot size="sm">
                <LiveLabel active={live}>Loading</LiveLabel>
              </Badge>
              <span className="text-caption text-secondary tabular-nums">
                <SwapText text={formatDuration(Math.max(0, now - row.startedAt))} granularity="char" />
              </span>
            </span>
          );
        }
        return (
          <span className="grid min-w-0 justify-items-start">
            <ModelBadge name={name} size="sm" fit="width" />
            {local === undefined ? null : (
              <span className="max-w-full truncate text-caption text-muted">
                {[local.quant, `${formatBytes(local.sizeBytes)} on disk`]
                  .filter((part) => part !== null && part !== undefined && part !== '')
                  .join(' · ')}
              </span>
            )}
          </span>
        );
      },
    },
    {
      id: 'family',
      header: 'Family',
      minTableWidth: 700,
      cell: (row) => (row.kind === 'model' ? <span className="font-mono">{row.model.modelType}</span> : '—'),
    },
    {
      id: 'window',
      header: 'Window',
      align: 'end',
      minTableWidth: 640,
      cell: (row) => (row.kind === 'model' ? formatCount(row.model.ctxWindow) : '—'),
    },
    {
      id: 'cache',
      header: 'Cache',
      minTableWidth: 560,
      cell: (row) =>
        row.kind === 'model' ? (
          <Badge size="sm" outline>
            {row.model.paged ? 'Paged' : 'Flat'}
          </Badge>
        ) : (
          '—'
        ),
    },
    {
      id: 'running',
      header: 'Running',
      align: 'end',
      cell: (row) =>
        row.kind === 'model' ? `${formatNumber(row.model.inFlight)} of ${formatNumber(row.model.maxConcurrent)}` : '—',
    },
    {
      id: 'queue',
      header: 'Queue',
      align: 'end',
      cell: (row) =>
        row.kind === 'model'
          ? row.model.queueLimit === null
            ? formatNumber(row.model.queued)
            : `${formatNumber(row.model.queued)} / ${formatNumber(row.model.queueLimit)}`
          : '—',
    },
    {
      id: 'loaded',
      header: 'Loaded',
      align: 'end',
      width: 116,
      truncate: true,
      minTableWidth: 760,
      cell: (row) => {
        if (row.kind !== 'model' || row.model.loadedAt === null) return '—';
        const when = formatRelativeTime(row.model.loadedAt);
        const took = row.model.loadMs === null ? null : ` · ${formatDuration(row.model.loadMs)}`;
        return (
          <span className="whitespace-nowrap tabular-nums" title={took === null ? when : `${when}${took}`}>
            {when}
            {took === null ? null : <span className="text-muted">{took}</span>}
          </span>
        );
      },
    },
    {
      id: 'pool',
      header: 'Pool at rest',
      cell: (row) => {
        if (row.kind !== 'model') return '—';
        const pool = row.model.pool;
        if (pool === null) {
          return (
            <span className="text-muted" title={row.model.paged ? 'No snapshot yet' : 'This model has no block pool'}>
              —
            </span>
          );
        }
        const share = pool.blockCapacity > 0 ? pool.allocatedBlocks / pool.blockCapacity : null;
        return (
          <span className="grid gap-1">
            <Meter
              size="dense"
              label={`Block pool of ${row.model.name}`}
              value={pool.allocatedBlocks}
              max={pool.blockCapacity > 0 ? pool.blockCapacity : null}
              unit="blocks"
              formatValue={formatNumber}
            />
            <span className="text-caption text-secondary tabular-nums">{`${formatPercent(share)} · ${formatClock(pool.at)}`}</span>
          </span>
        );
      },
    },
  ];

  return (
    <Panel title="Loaded models" padding="none">
      <DataTable
        caption="Loaded models"
        columns={columns}
        rows={rows}
        rowKey={(row) => (row.kind === 'loading' ? `loading-${row.name}` : row.model.name)}
        rowAttrs={(row) => ({ 'data-model': rowName(row) })}
        rowTo={(row) => (row.kind === 'model' && row.model.paged ? modelUrl(row.model.name) : undefined)}
        empty={{ title: 'No model loaded', description: 'A model loads on its first request.' }}
      />
    </Panel>
  );
}
