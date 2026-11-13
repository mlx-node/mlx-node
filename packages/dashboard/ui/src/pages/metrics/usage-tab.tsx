import { useState, type ReactElement } from 'react';
import { Link } from 'react-router-dom';

import { BarList, type BarListRow } from '../../components/arc-x/bar-list/bar-list';
import { ChartCard } from '../../components/arc-x/chart-card/chart-card';
import { Histogram } from '../../components/arc-x/histogram/histogram';
import { KeyValue } from '../../components/arc-x/key-value/key-value';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Panel } from '../../components/arc-x/panel/panel';
import { Placeholder } from '../../components/arc-x/placeholder/placeholder';
import { StackedBarChart } from '../../components/arc-x/stacked-bar-chart/stacked-bar-chart';
import SegmentedControl, { type Segment } from '../../components/arc/segmented-control/segmented-control';
import { formatCount, formatCountNoun, formatNumber, formatRate } from '../../lib/format';
import { bucketWord, cacheReuse, dispatchText, RECORD_HINT, toBins } from '../../lib/metrics-view';
import { modelColor } from '../../lib/model-colors';
import { TOKEN_SERIES } from '../../lib/series';
import { pivotMeasures } from '../../lib/timeseries';
import {
  LOW_SAMPLE_TAIL_N,
  type MetricsDistributions,
  type MetricsSummary,
  type MetricsTimeseries,
} from '../../lib/types';
import { PageGrid } from '../../shell/page';
import { modelPaint, RefreshAlert, regionError, regionState, useHeldQuery, useMetricsScope, withModel } from './shared';

const MEASURE_OPTIONS: Segment[] = [
  { value: 'turns', label: 'Turns' },
  { value: 'tokens', label: 'Output tokens' },
];

/** Reasoning tokens are a part of the output count, so the stack shows the rest of the output beside them. */
function answerTokens(totals: { outputTokens: number; reasoningTokens: number }): number {
  return Math.max(0, totals.outputTokens - totals.reasoningTokens);
}

function percent(value: number): string {
  return `${String(Math.round(value))}%`;
}

function percentNumber(value: number): string {
  return String(Math.round(value));
}

export function UsageTab(): ReactElement {
  const scope = useMetricsScope();
  const { q, range, model } = scope;
  const summary = useHeldQuery<MetricsSummary>(`/metrics/summary?${q}`);
  const noneTs = useHeldQuery<MetricsTimeseries>(
    `/metrics/timeseries?${q}&bucket=${range.bucket}&groupBy=${model === null ? 'none' : 'model'}`,
    { range },
  );
  const tsRange = noneTs.range ?? range;
  const dist = useHeldQuery<MetricsDistributions>(`/metrics/distributions?${q}&groupBy=none`);
  const [measure, setMeasure] = useState<'turns' | 'tokens'>('turns');
  const totals = summary.data?.totals;
  const series = dist.data?.series[0];
  const edges = dist.data?.edges;
  const histColor = model === null ? undefined : modelColor(model).color;

  const tokenData = pivotMeasures(
    noneTs.data?.points ?? [],
    tsRange,
    {
      cached: (p) => p.totals.cachedTokens,
      input: (p) => p.totals.inputTokens,
      answer: (p) => answerTokens(p.totals),
      reasoning: (p) => p.totals.reasoningTokens,
    },
    0,
  );
  const tokenSum =
    totals === undefined ? 0 : totals.cachedTokens + totals.inputTokens + answerTokens(totals) + totals.reasoningTokens;

  const reuse = totals === undefined ? null : cacheReuse(totals.cachedTokens, totals.inputTokens);
  const dispatches = summary.data?.dispatches ?? [];

  const byModel = summary.data?.byModel.filter((row) => row.totals.turns > 0) ?? [];
  const shareTotal = byModel.reduce(
    (sum, row) => sum + (measure === 'turns' ? row.totals.turns : row.totals.outputTokens),
    0,
  );
  const shareRows: BarListRow[] = byModel.map((row) => {
    const value = measure === 'turns' ? row.totals.turns : row.totals.outputTokens;
    return {
      key: row.model,
      label: <ModelBadge name={row.model} size="sm" />,
      text: row.model,
      value,
      hint: formatRate(value, shareTotal),
      to: scope.urlWith({ model: row.model }),
      ...modelPaint(row.model),
    };
  });
  const fillRows: BarListRow[] = byModel.map((row) => {
    const fill = row.latency.ctxFillPct;
    const tail = fill.n >= LOW_SAMPLE_TAIL_N;
    return {
      key: row.model,
      label: <ModelBadge name={row.model} size="sm" />,
      text: row.model,
      value: tail ? fill.p95 : fill.p50,
      samples: fill.n,
      hint: tail ? undefined : `median, n=${String(fill.n)}`,
      ...modelPaint(row.model),
    };
  });

  const histogram = (
    title: string,
    subtitle: string,
    key: 'promptTokens' | 'outputTokens' | 'ctxFillPct',
    unit: string,
    format: (value: number) => string,
    extra: { markers?: Array<{ value: number; label: string }>; empty?: string; footnote?: string } = {},
  ): ReactElement => (
    <ChartCard
      className="col-span-12 @md:col-span-4"
      title={title}
      subtitle={withModel(subtitle, model)}
      height={160}
      state={regionState(dist, (series?.[key].n ?? 0) === 0)}
      empty={{ title: extra.empty ?? 'No turns in this range', hint: RECORD_HINT }}
      error={regionError(dist)}
      footnote={extra.footnote}
    >
      <Histogram
        label={title}
        unit={unit}
        height={160}
        countLabel={{ one: 'turn', other: 'turns' }}
        bins={toBins(edges?.[key] ?? [], series?.[key])}
        markers={extra.markers}
        formatBound={format}
        color={histColor}
      />
    </ChartCard>
  );

  const fillQ = summary.data?.latency.ctxFillPct;
  const fillMarkers: Array<{ value: number; label: string }> = [];
  if (fillQ?.p50 != null) fillMarkers.push({ value: fillQ.p50, label: 'p50' });
  if (fillQ?.p95 != null && fillQ.n >= LOW_SAMPLE_TAIL_N) fillMarkers.push({ value: fillQ.p95, label: 'p95' });

  return (
    <div className="grid gap-3 pt-3">
      <RefreshAlert queries={[summary, noneTs, dist]} />
      <PageGrid>
        <ChartCard
          className="col-span-12 @md:col-span-8"
          title="Tokens"
          subtitle={withModel(`Per ${bucketWord(tsRange.bucket)}, by kind · tokens`, model)}
          height={200}
          state={summary.data === undefined ? regionState(summary, false) : regionState(noneTs, tokenSum === 0)}
          empty={{ title: 'No token use in this range', hint: RECORD_HINT }}
          error={regionError(summary.data === undefined ? summary : noneTs)}
          footnote={
            totals === undefined
              ? undefined
              : `${formatCount(totals.outputTokens)} output · ${formatCount(totals.inputTokens)} new input · ${formatCount(totals.cachedTokens)} cached`
          }
        >
          <StackedBarChart
            label="Tokens"
            unit="tokens"
            height={200}
            data={tokenData}
            series={TOKEN_SERIES}
            formatValue={formatNumber}
            formatTick={formatCount}
            onSelect={() => scope.setParams({ tab: 'turns' })}
          />
        </ChartCard>
        <Panel className="col-span-12 @md:col-span-4" title="Prompt reuse" busy={summary.data === undefined}>
          <div className="grid gap-3">
            {reuse === null ? (
              <Placeholder width="40%" height={32} />
            ) : (
              <p className="flex items-baseline gap-2">
                <span className="font-heading text-figure tabular-nums">{reuse.text}</span>
                <span className="text-body text-secondary">
                  {reuse.fraction === null ? 'No prompts in this range' : 'of prompt tokens came from cache'}
                </span>
              </p>
            )}
            <KeyValue
              layout="rows"
              ariaLabel="Prompt reuse by request kind"
              loading={summary.data === undefined}
              items={dispatches.map((row) => ({
                key: row.dispatch,
                label: dispatchText(row.dispatch),
                value: formatRate(row.cachedTokens, row.promptTokens),
                hint: formatCountNoun(row.turns, 'turn', 'turns'),
                numeric: true,
              }))}
            />
            <p className="text-caption text-secondary">
              Cached tokens skip prefill. This is the in-memory cache; the disk tier is on the{' '}
              <Link to="/cache" className="link">
                Cache page
              </Link>
              .
            </p>
          </div>
        </Panel>
        {histogram('Prompt size', 'Turns per size range · tokens', 'promptTokens', 'tokens', formatCount)}
        {histogram('Output size', 'Turns per size range · tokens', 'outputTokens', 'tokens', formatCount)}
        {histogram('Context fill', 'Turns per share of the window at turn end · %', 'ctxFillPct', '%', percentNumber, {
          markers: fillMarkers,
          empty: 'No turn recorded its context window',
          footnote:
            series === undefined
              ? undefined
              : `${formatCountNoun(series.ctxFillPct.n, 'turn', 'turns')} with a known window`,
        })}
        <Panel
          className="col-span-12 @md:col-span-6"
          title="Turns by model"
          busy={summary.data === undefined}
          actions={
            <SegmentedControl
              label="Measure"
              options={MEASURE_OPTIONS}
              value={measure}
              onValueChange={(value) => {
                if (value === 'turns' || value === 'tokens') setMeasure(value);
              }}
            />
          }
        >
          <BarList
            label="Turns by model"
            unit={measure === 'turns' ? 'turns' : 'tokens'}
            rows={shareRows}
            limit={8}
            formatValue={measure === 'turns' ? formatNumber : formatCount}
            emptyLabel="No turns in this range"
          />
        </Panel>
        <Panel
          className="col-span-12 @md:col-span-6"
          title="Context fill by model"
          description="p95 of the window in use · %"
          busy={summary.data === undefined}
        >
          <div className="grid gap-2">
            <BarList
              label="Context fill by model"
              unit=""
              rows={fillRows}
              max={100}
              sort="desc"
              formatValue={percent}
              lowSampleBelow={5}
              emptyLabel="No turns in this range"
            />
            <p className="text-caption text-secondary">Above 80% a model is close to its limit.</p>
          </div>
        </Panel>
      </PageGrid>
    </div>
  );
}
