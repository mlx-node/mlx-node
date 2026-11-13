import { RefreshCw } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, type ReactElement } from 'react';
import { useSearchParams } from 'react-router-dom';

import { useChartSize } from '../components/arc-x/chart/use-chart-size';
import { CompactField } from '../components/arc-x/compact-field/compact-field';
import { IconButton } from '../components/arc-x/icon-button/icon-button';
import SegmentedControl from '../components/arc/segmented-control/segmented-control';
import { Select } from '../components/arc/select/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/arc/tabs/tabs';
import { useTurnDrawer } from '../components/turn-drawer';
import { mutate } from '../lib/api';
import { useDataVersion } from '../lib/live';
import { isMetricsTab, METRICS_TABS, type MetricsTab } from '../lib/metrics-view';
import { alignedRange, isRangeKey, RANGE_OPTIONS, RANGE_OPTIONS_SHORT, useRange } from '../lib/range';
import type { MetricsModels, TurnRow } from '../lib/types';
import { reloadMountedQueries, useJson } from '../lib/use-api';
import { useCommands, type AppCommand } from '../shell/commands';
import { Page, PageHeader } from '../shell/page';
import { FailuresTab } from './metrics/failures-tab';
import { MetricsScopeContext, scopeQuery, type MetricsScope } from './metrics/shared';
import { SpeculationTab } from './metrics/speculation-tab';
import { SpeedTab } from './metrics/speed-tab';
import { SummaryTab } from './metrics/summary-tab';
import { TurnsTab } from './metrics/turns-tab';
import { UsageTab } from './metrics/usage-tab';

const ALL_MODELS = '__all__';

const RANGE_COMMANDS = [
  { key: '1h', label: 'last hour' },
  { key: '24h', label: '24 hours' },
  { key: '7d', label: '7 days' },
  { key: '30d', label: '30 days' },
] as const;

function TabBody({ tab }: { tab: MetricsTab }): ReactElement {
  switch (tab) {
    case 'summary':
      return <SummaryTab />;
    case 'speed':
      return <SpeedTab />;
    case 'usage':
      return <UsageTab />;
    case 'speculation':
      return <SpeculationTab />;
    case 'failures':
      return <FailuresTab />;
    case 'turns':
      return <TurnsTab />;
  }
}

export default function Metrics(): ReactElement {
  const [params, setSearch] = useSearchParams();
  const [rangeKey, setRange] = useRange('7d');
  const modelParam = params.get('model');
  const model = modelParam === null || modelParam === '' ? null : modelParam;
  const rawTab = params.get('tab');
  const tab: MetricsTab = rawTab !== null && isMetricsTab(rawTab) ? rawTab : 'summary';
  const dataVersion = useDataVersion();
  // A data event re-aligns the window; inside one bucket the result (and so every request path) is equal.
  const range = useMemo(() => alignedRange(rangeKey), [rangeKey, dataVersion]);
  const q = scopeQuery(range, model);

  const sizer = useRef<HTMLDivElement>(null);
  const width = useChartSize(sizer, 1180);
  const models = useJson<MetricsModels>('/metrics/models', { live: true });
  const [refreshing, setRefreshing] = useState(false);
  const [known, setKnown] = useState<readonly TurnRow[]>([]);
  const drawer = useTurnDrawer(known);

  const setParams = useCallback(
    (updates: Readonly<Record<string, string | null>>) => {
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
    },
    [setSearch],
  );

  const urlWith = useCallback(
    (updates: Readonly<Record<string, string | null>>) => {
      const next = new URLSearchParams(params);
      for (const [key, value] of Object.entries(updates)) {
        if (value === null) next.delete(key);
        else next.set(key, value);
      }
      const text = next.toString();
      return text === '' ? '/metrics' : `/metrics?${text}`;
    },
    [params],
  );

  const reportTurns = useCallback((rows: readonly TurnRow[]) => {
    setKnown((current) => (current === rows ? current : rows));
  }, []);

  const [busyQueries, setBusyQueries] = useState<ReadonlySet<string>>(() => new Set());
  const reportBusy = useCallback((id: string, busy: boolean) => {
    setBusyQueries((current) => {
      if (current.has(id) === busy) return current;
      const next = new Set(current);
      if (busy) next.add(id);
      else next.delete(id);
      return next;
    });
  }, []);

  const scope = useMemo(
    (): MetricsScope => ({ rangeKey, range, model, q, setParams, urlWith, reportTurns, reportBusy }),
    [rangeKey, range, model, q, setParams, urlWith, reportTurns, reportBusy],
  );

  const modelOptions = [{ value: ALL_MODELS, label: 'All models' }];
  for (const row of models.data?.models ?? []) modelOptions.push({ value: row.model, label: row.model });
  if (model !== null && !modelOptions.some((option) => option.value === model)) {
    modelOptions.push({ value: model, label: model });
  }

  const refresh = async (): Promise<void> => {
    setRefreshing(true);
    try {
      await mutate('POST', '/ingest');
    } catch {
      // A failed ingest still leaves the last index to read.
    } finally {
      reloadMountedQueries();
      setRefreshing(false);
    }
  };

  const commands = useMemo((): AppCommand[] => {
    const list: AppCommand[] = RANGE_COMMANDS.map((item) => ({
      id: `metrics-range-${item.key}`,
      label: `Metrics: ${item.label}`,
      group: 'Metrics',
      run: () => setRange(item.key),
    }));
    for (const item of METRICS_TABS) {
      list.push({
        id: `metrics-tab-${item.value}`,
        label: `Metrics: show ${item.label}`,
        group: 'Metrics',
        run: () => setParams({ tab: item.value === 'summary' ? null : item.value, detail: null }),
      });
    }
    list.push({
      id: 'metrics-all-models',
      label: 'Metrics: all models',
      group: 'Metrics',
      run: () => setParams({ model: null }),
    });
    return list;
  }, [setRange, setParams]);
  useCommands(commands);

  return (
    <Page busy={refreshing || models.loading || busyQueries.size > 0}>
      <PageHeader
        title="Metrics"
        description="Throughput, latency and token use from recorded turns."
        sticky
        controls={
          <>
            <CompactField>
              <Select
                label="Model"
                options={modelOptions}
                value={model ?? ALL_MODELS}
                onValueChange={(value) => setParams({ model: value === ALL_MODELS ? null : value })}
              />
            </CompactField>
            <SegmentedControl
              label="Range"
              options={width < 640 ? RANGE_OPTIONS_SHORT : RANGE_OPTIONS}
              value={rangeKey}
              onValueChange={(value) => {
                if (isRangeKey(value)) setRange(value);
              }}
            />
          </>
        }
        actions={
          <IconButton
            label="Refresh data"
            variant="secondary"
            icon={<RefreshCw size={16} aria-hidden="true" />}
            loading={refreshing}
            onClick={() => void refresh()}
          />
        }
      />
      <MetricsScopeContext.Provider value={scope}>
        <div ref={sizer} className="min-w-0">
          <Tabs
            value={tab}
            onValueChange={(value) => {
              if (isMetricsTab(value)) setParams({ tab: value === 'summary' ? null : value, detail: null });
            }}
          >
            <TabsList aria-label="Metrics views">
              {METRICS_TABS.map((item) => (
                <TabsTrigger key={item.value} value={item.value}>
                  {item.label}
                </TabsTrigger>
              ))}
            </TabsList>
            {METRICS_TABS.map((item) => (
              <TabsContent key={item.value} value={item.value}>
                {item.value === tab ? <TabBody tab={tab} /> : null}
              </TabsContent>
            ))}
          </Tabs>
        </div>
      </MetricsScopeContext.Provider>
      {drawer.element}
    </Page>
  );
}
