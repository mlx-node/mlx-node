import { useEffect, useMemo, useState, type ReactElement } from 'react';
import { useSearchParams } from 'react-router-dom';

import { CompactField } from '../components/arc-x/compact-field/compact-field';
import { EmptyState } from '../components/arc/empty-state/empty-state';
import SegmentedControl from '../components/arc/segmented-control/segmented-control';
import { Select } from '../components/arc/select/select';
import { engineSummary, pickProcess, procKeyOf, seriesStarts } from '../lib/engine-view';
import { useHeld } from '../lib/held';
import { useDataVersion, useRuntimeLive } from '../lib/live';
import { useLiveHistory } from '../lib/live-history';
import { ENGINE_RANGE_OPTIONS, engineRange, isEngineRangeKey, type EngineRangeKey } from '../lib/range';
import type { ModelsResponse, RuntimeSeries } from '../lib/types';
import { LIVE_REFETCH_FAST_MS, useJson } from '../lib/use-api';
import { Page, PageHeader } from '../shell/page';
import { FlightSection } from './engine/flight-section';
import { HistorySection } from './engine/history-section';
import { ModelsSection } from './engine/models-section';
import { NowTicker, RateAndMemory } from './engine/now-section';
import { PoolSection } from './engine/pool-section';
import { StatusAlert, StatusBadge, SupervisorPanel } from './engine/process-section';

const SUMMARY_EVERY_MS = 10_000;

export default function Engine(): ReactElement {
  const [params, setSearch] = useSearchParams();
  const live = useRuntimeLive();
  const history = useLiveHistory();
  const dataVersion = useDataVersion();
  const rawPid = params.get('pid');
  const pidParam = rawPid === null ? null : Number.parseInt(rawPid, 10);
  const pid = pidParam !== null && Number.isFinite(pidParam) ? pidParam : null;
  const rawHistory = params.get('history');
  const historyKey: EngineRangeKey = rawHistory !== null && isEngineRangeKey(rawHistory) ? rawHistory : '1h';
  const poolModel = params.get('model');

  const setParams = (updates: Readonly<Record<string, string | null>>): void => {
    setSearch(
      (current) => {
        const next = new URLSearchParams(current);
        for (const [key, value] of Object.entries(updates)) {
          if (value === null) next.delete(key);
          else next.set(key, value);
        }
        return next;
      },
      { replace: true },
    );
  };
  const modelUrl = (name: string): string => {
    const next = new URLSearchParams(params);
    next.set('model', name);
    return `/engine?${next.toString()}`;
  };

  // A data event re-aligns the window; inside one bucket the request path stays equal.
  const range = useMemo(() => engineRange(historyKey), [historyKey, dataVersion]);
  const seriesState = useJson<RuntimeSeries>(
    `/runtime/series?from=${String(range.from)}&to=${String(range.to)}&bucketMs=${String(range.bucketMs)}`,
    { live: { minIntervalMs: LIVE_REFETCH_FAST_MS } },
  );
  const series = useHeld(seriesState);
  const locals = useJson<ModelsResponse>('/models');

  const process = pickProcess(live, pid);
  const choices = (live?.processes ?? []).filter((item) => item.status !== 'dead');
  const procKey = process === null ? null : procKeyOf(process);

  const [summary, setSummary] = useState<{ text: string; at: number; status: string | null }>({
    text: '',
    at: 0,
    status: null,
  });
  useEffect(() => {
    if (live === undefined) {
      if (summary.status !== null) setSummary({ text: '', at: 0, status: null });
      return;
    }
    const statusChanged = summary.status !== live.status;
    if (!statusChanged && live.now - summary.at < SUMMARY_EVERY_MS) return;
    setSummary({ text: engineSummary(live, process), at: live.now, status: live.status });
  }, [live, process, summary]);

  const seriesChart =
    series.data === undefined
      ? seriesState.error !== undefined
        ? ('error' as const)
        : ('loading' as const)
      : series.held
        ? ('held' as const)
        : ('ready' as const);
  const seriesError =
    seriesState.error === undefined ? undefined : { message: seriesState.error.message, retry: seriesState.reload };

  const rangeControl = (
    <SegmentedControl
      label="History range"
      options={ENGINE_RANGE_OPTIONS}
      value={historyKey}
      onValueChange={(value) => {
        if (isEngineRangeKey(value)) setParams({ history: value === '1h' ? null : value });
      }}
    />
  );

  return (
    <Page busy={live === undefined || seriesState.loading}>
      <PageHeader
        title="Engine"
        description="The inference process, live."
        meta={<StatusBadge status={live?.status} />}
        controls={
          choices.length >= 2 ? (
            <CompactField>
              <Select
                label="Process"
                options={choices.map((item) => ({
                  value: String(item.pid),
                  label: `${item.host} · :${String(item.port)} · pid ${String(item.pid)}`,
                }))}
                value={process === null ? undefined : String(process.pid)}
                onValueChange={(value) => setParams({ pid: value })}
              />
            </CompactField>
          ) : undefined
        }
      />
      <p className="sr-only" aria-live="polite">
        {summary.text}
      </p>
      <StatusAlert live={live} process={process} />
      {live !== undefined && process === null ? (
        <EmptyState
          title="Inference is not running"
          description="Start it from the menu bar, or run mlx serve in a terminal. Live status appears here within a second."
        />
      ) : (
        <>
          <NowTicker live={live} process={process} history={history} />
          {live === undefined || process === null ? null : (
            <>
              <RateAndMemory live={live} process={process} history={history} series={series.data} />
              <FlightSection process={process} now={live.now} />
              <ModelsSection process={process} now={live.now} locals={locals.data?.models} modelUrl={modelUrl} />
            </>
          )}
        </>
      )}
      <PoolSection
        series={series.data}
        state={seriesChart}
        error={series.data === undefined ? seriesError : undefined}
        procKey={procKey}
        process={process}
        model={poolModel}
        onModelChange={(name) => setParams({ model: name })}
        rangeControl={rangeControl}
      />
      <HistorySection
        series={series.data}
        state={seriesChart}
        error={series.data === undefined ? seriesError : undefined}
        live={live}
        starts={series.data === undefined ? range.starts : seriesStarts(series.data)}
        rangeControl={rangeControl}
      />
      {live?.supervisor == null ? null : <SupervisorPanel supervisor={live.supervisor} />}
    </Page>
  );
}
