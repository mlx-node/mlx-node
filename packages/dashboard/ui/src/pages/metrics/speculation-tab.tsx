import { useMemo, type ReactElement } from 'react';

import { ChartCard } from '../../components/arc-x/chart-card/chart-card';
import type { ChartPoint } from '../../components/arc-x/chart/types';
import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { DraftSweep } from '../../components/arc-x/draft-sweep/draft-sweep';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { QuantileBar } from '../../components/arc-x/quantile-bar/quantile-bar';
import { StackedBarChart } from '../../components/arc-x/stacked-bar-chart/stacked-bar-chart';
import { StatGroup, StatTile } from '../../components/arc-x/stat-tile/stat-tile';
import { TrendChart } from '../../components/arc-x/trend-chart/trend-chart';
import { EmptyState } from '../../components/arc/empty-state/empty-state';
import { formatCountNoun, formatNumber, formatPercentValue, formatRate, formatTps } from '../../lib/format';
import { bucketWord, speculationView } from '../../lib/metrics-view';
import { modelColor, modelSeries } from '../../lib/model-colors';
import { bucketPoint, moreInTableLabel } from '../../lib/timeseries';
import { LOW_SAMPLE_N, type MetricsSpeculative } from '../../lib/types';
import { PageGrid } from '../../shell/page';
import {
  MedianCell,
  quantileMax,
  RefreshAlert,
  regionError,
  tileState,
  useHeldQuery,
  useMetricsScope,
  withModel,
} from './shared';

type SpecModel = MetricsSpeculative['models'][number];

const TREND_MODELS = 8;

function byCycles(rows: readonly SpecModel[]): SpecModel[] {
  return [...rows].sort((a, b) => b.cycles - a.cycles || (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
}

function fixed(value: number | null, digits: number): string | null {
  return value === null || !Number.isFinite(value) ? null : value.toFixed(digits);
}

function gainValue(row: SpecModel): number | null {
  const withP = row.decodeTpsWithMtp;
  const without = row.decodeTpsWithoutMtp;
  if (
    withP.n < LOW_SAMPLE_N ||
    without.n < LOW_SAMPLE_N ||
    withP.p50 === null ||
    without.p50 === null ||
    without.p50 <= 0
  ) {
    return null;
  }
  return withP.p50 / without.p50;
}

function gain(row: SpecModel): string {
  const value = gainValue(row);
  return value === null ? '—' : `${value.toFixed(2)}×`;
}

export function SpeculationTab(): ReactElement {
  const scope = useMetricsScope();
  const { q, range, model } = scope;
  const spec = useHeldQuery<MetricsSpeculative>(`/metrics/speculative?${q}&bucket=${range.bucket}`, { range });
  const specRange = spec.range ?? range;
  const view = useMemo(() => speculationView(spec.data?.models ?? []), [spec.data]);
  const tile = tileState(spec);

  const positions = useMemo(() => {
    const top = byCycles(view.models).slice(0, 4);
    const count = view.rates.length;
    const data: ChartPoint[] = [];
    for (let pos = 1; pos <= count; pos += 1) {
      const values: Record<string, number | null> = {};
      const samples: Record<string, number> = {};
      for (const row of top) {
        const entry = row.acceptByPos.find((item) => item.pos === pos - 1);
        values[row.model] = entry === undefined ? null : entry.rate * 100;
        samples[row.model] = entry?.cycles ?? 0;
      }
      data.push({ key: String(pos), label: `Position ${String(pos)}`, axisLabel: String(pos), values, samples });
    }
    return { data, series: modelSeries(top.map((row) => row.model)), extra: view.models.length - top.length };
  }, [view]);

  const trend = useMemo(() => {
    const names = byCycles(view.models)
      .slice(0, TREND_MODELS)
      .map((row) => row.model);
    const byStart = new Map<number, Map<string, { value: number | null; turns: number }>>();
    for (const row of spec.data?.trend ?? []) {
      const bucket = byStart.get(row.t) ?? new Map<string, { value: number | null; turns: number }>();
      bucket.set(row.model, { value: row.meanAccepted, turns: row.mtpTurns });
      byStart.set(row.t, bucket);
    }
    const now = specRange.to - 1;
    const data: ChartPoint[] = specRange.starts.map((t) => {
      const bucket = byStart.get(t);
      const values: Record<string, number | null> = {};
      const samples: Record<string, number> = {};
      for (const name of names) {
        const entry = bucket?.get(name);
        values[name] = entry?.value ?? null;
        samples[name] = entry?.turns ?? 0;
      }
      return { ...bucketPoint(t, specRange.bucket, now), values, samples };
    });
    return { data, series: modelSeries(names), extra: view.models.length - names.length };
  }, [spec.data, view, specRange]);

  const single =
    view.models.length === 1
      ? view.models[0]
      : model !== null
        ? view.models.find((row) => row.model === model)
        : undefined;
  const withQ = single?.decodeTpsWithMtp;
  const withoutQ = single?.decodeTpsWithoutMtp;
  const speedTile =
    withQ !== undefined &&
    withoutQ !== undefined &&
    withQ.n > 0 &&
    withoutQ.n > 0 &&
    withQ.p50 !== null &&
    withoutQ.p50 !== null &&
    withoutQ.p50 > 0
      ? { with: withQ, without: withoutQ, ratio: withQ.p50 / withoutQ.p50 }
      : null;

  const barMax = quantileMax([
    ...view.models.map((row) => row.decodeTpsWithMtp),
    ...view.models.map((row) => row.decodeTpsWithoutMtp),
  ]);
  const columns: DataColumn<SpecModel>[] = [
    {
      id: 'model',
      header: 'Model',
      flex: true,
      minWidth: 140,
      cell: (row) => <ModelBadge name={row.model} size="sm" fit="width" />,
      sortValue: (row) => row.model,
      truncate: true,
    },
    {
      id: 'mtpTurns',
      header: 'MTP turns',
      align: 'end',
      width: 96,
      cell: (row) => (
        <span className="whitespace-nowrap tabular-nums">
          {formatNumber(row.mtpTurns)} <span className="text-muted">{`of ${formatNumber(row.turns)}`}</span>
        </span>
      ),
      sortValue: (row) => row.mtpTurns,
    },
    {
      id: 'cycles',
      header: 'Cycles',
      align: 'end',
      width: 84,
      cell: (row) => formatNumber(row.cycles),
      sortValue: (row) => row.cycles,
    },
    {
      id: 'tokens',
      header: 'Tokens/cycle',
      align: 'end',
      width: 96,
      minTableWidth: 660,
      cell: (row) => fixed(row.meanAccepted, 2) ?? '—',
      sortValue: (row) => row.meanAccepted,
    },
    {
      id: 'drafts',
      header: 'Drafts/cycle',
      align: 'end',
      width: 96,
      minTableWidth: 960,
      cell: (row) => fixed(row.meanDrafts, 2) ?? '—',
      sortValue: (row) => row.meanDrafts,
    },
    {
      id: 'depth',
      header: 'Depth',
      align: 'end',
      width: 72,
      minTableWidth: 700,
      cell: (row) => fixed(row.meanDepth, 1) ?? '—',
      sortValue: (row) => row.meanDepth,
    },
    {
      id: 'with',
      header: 'With MTP tok/s',
      width: 148,
      cell: (row) => (
        <span className="inline-flex items-center gap-2">
          <MedianCell text={formatTps(row.decodeTpsWithMtp.p50)} n={row.decodeTpsWithMtp.n} />
          <QuantileBar
            label={`Decode speed of ${row.model} with MTP`}
            quantiles={row.decodeTpsWithMtp}
            min={0}
            max={barMax}
            unit="tok/s"
            formatValue={formatTps}
            color={modelColor(row.model).color}
            width={72}
          />
        </span>
      ),
      sortValue: (row) => row.decodeTpsWithMtp.p50,
    },
    {
      id: 'without',
      header: 'Without tok/s',
      width: 148,
      minTableWidth: 640,
      cell: (row) => (
        <span className="inline-flex items-center gap-2">
          <MedianCell text={formatTps(row.decodeTpsWithoutMtp.p50)} n={row.decodeTpsWithoutMtp.n} />
          <QuantileBar
            label={`Decode speed of ${row.model} without MTP`}
            quantiles={row.decodeTpsWithoutMtp}
            min={0}
            max={barMax}
            unit="tok/s"
            formatValue={formatTps}
            color="var(--model-other)"
            width={72}
          />
        </span>
      ),
      sortValue: (row) => row.decodeTpsWithoutMtp.p50,
    },
    {
      id: 'gain',
      header: 'Gain',
      align: 'end',
      width: 72,
      cell: (row) => gain(row),
      sortValue: (row) => gainValue(row),
    },
  ];

  if (spec.data !== undefined && view.models.length === 0) {
    return (
      <div className="grid gap-3 pt-3">
        <RefreshAlert queries={[spec]} />
        <EmptyState
          title="No speculative decoding recorded"
          description="Turn on MTP (enableMtp) for a model that supports it to record acceptance."
        />
      </div>
    );
  }

  const cardState =
    spec.data === undefined ? (spec.state.error !== undefined ? 'error' : 'loading') : spec.held ? 'held' : 'ready';

  return (
    <div className="grid gap-3 pt-3">
      <RefreshAlert queries={[spec]} />
      <StatGroup label="Speculation">
        <StatTile
          label="Tokens per cycle"
          value={fixed(view.tokensPerCycle, 2)}
          unit="tokens"
          hint="1.0 = no draft kept"
          {...tile}
        />
        <StatTile
          label="Drafts accepted per cycle"
          value={fixed(view.draftsPerCycle, 2)}
          hint={`mean draft depth ${fixed(view.meanDepth, 1) ?? '—'}`}
          footer={
            spec.data === undefined ? undefined : (
              <DraftSweep label="Share of drafts kept at each position" rates={view.rates} samples={view.samples} />
            )
          }
          {...tile}
        />
        <StatTile
          label="Turns that used it"
          value={spec.data === undefined ? null : formatRate(view.mtpTurns, view.turns)}
          hint={`${formatNumber(view.mtpTurns)} of ${formatCountNoun(view.turns, 'turn', 'turns')}`}
          {...tile}
        />
        {speedTile === null ? null : (
          <StatTile
            label="Speed with it"
            value={`${speedTile.ratio.toFixed(2)}×`}
            hint={`${formatTps(speedTile.with.p50)} vs ${formatTps(speedTile.without.p50)} tok/s`}
            sampleCount={Math.min(speedTile.with.n, speedTile.without.n)}
            {...tile}
          />
        )}
      </StatGroup>
      <PageGrid>
        <ChartCard
          className="col-span-12 @md:col-span-7"
          title="Acceptance by draft position"
          subtitle={withModel('Share of drafts kept at each position · %', model)}
          height={200}
          state={cardState}
          error={regionError(spec)}
          footnote={`${formatCountNoun(view.cycles, 'cycle', 'cycles')}${positions.extra > 0 ? ` · ${moreInTableLabel(positions.extra)}` : ''}`}
        >
          <StackedBarChart
            label="Acceptance by draft position"
            unit="%"
            mode="grouped"
            height={200}
            data={positions.data}
            series={positions.series}
            lowSampleBelow={5}
            formatValue={(value) => formatPercentValue(value / 100)}
          />
        </ChartCard>
        <ChartCard
          className="col-span-12 @md:col-span-5"
          title="Tokens per cycle over time"
          subtitle={withModel(`Cycle-weighted mean per ${bucketWord(specRange.bucket)} · tokens`, model)}
          height={200}
          state={cardState}
          error={regionError(spec)}
          footnote={`1.0 = no draft kept${trend.extra > 0 ? ` · ${moreInTableLabel(trend.extra)}` : ''}`}
        >
          <TrendChart
            label="Tokens per cycle over time"
            unit="tokens"
            height={200}
            curve="linear"
            data={trend.data}
            series={trend.series}
            lowSampleBelow={5}
            formatValue={(value) => value.toFixed(2)}
          />
        </ChartCard>
      </PageGrid>
      <div className="grid gap-2">
        <DataTable
          caption="Speculation by model"
          columns={columns}
          rows={view.models}
          rowKey={(row) => row.model}
          rowAttrs={(row) => ({ 'data-model': row.model })}
          defaultSort={{ id: 'cycles', direction: 'desc' }}
          loading={spec.data === undefined && spec.state.error === undefined}
          error={spec.data === undefined ? regionError(spec) : undefined}
          empty={{
            title: 'No speculative decoding recorded',
            description: 'Turn on MTP (enableMtp) for a model that supports it to record acceptance.',
          }}
        />
        <p className="text-caption text-secondary">
          With and without are different turns, not a controlled test. Prompt and answer length differ between them.
        </p>
      </div>
    </div>
  );
}
