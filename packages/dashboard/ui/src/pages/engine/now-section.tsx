import { useMemo, type ReactElement } from 'react';

import { PHASE_BANDS } from '../../components/arc-x/activity-strip/phase-stack';
import { KeyValue } from '../../components/arc-x/key-value/key-value';
import { LiveStream, type LiveStreamReadout } from '../../components/arc-x/live-stream/live-stream';
import { EngineStateBadge } from '../../components/arc-x/live-turn-row/live-turn-row';
import { MetricTicker, type TickerItem } from '../../components/arc-x/metric-ticker/metric-ticker';
import { Panel } from '../../components/arc-x/panel/panel';
import { Meter, SplitBar } from '../../components/arc-x/split-bar/split-bar';
import { fill } from '../../lib/chat-strings';
import { ENGINE } from '../../lib/engine-strings';
import { engineState, nowItems, procKeyOf, rateText, stackPoints } from '../../lib/engine-view';
import { formatBytes, formatDuration, formatNumber, formatTps } from '../../lib/format';
import type { LiveSample } from '../../lib/live-history';
import { LANES, METRIC } from '../../lib/metric-strings';
import { axisNumber } from '../../lib/metrics-view';
import { SERIES_A, SERIES_B } from '../../lib/series';
import type { RuntimeLive, RuntimeProcessLive, RuntimeSeries } from '../../lib/types';
import { PageGrid } from '../../shell/page';

export function NowTicker({
  live,
  process,
  history,
}: {
  live: RuntimeLive | undefined;
  process: RuntimeProcessLive | null;
  history: readonly LiveSample[];
}): ReactElement {
  const items = useMemo((): TickerItem[] => {
    if (live === undefined || process === null) return [];
    return nowItems(process, history, live.now).map((item) =>
      item.key === 'mem'
        ? {
            ...item,
            footer: (
              <Meter
                size="thin"
                label="MLX memory against the wired limit"
                value={process.mem.active}
                max={process.mem.wiredLimit > 0 ? process.mem.wiredLimit : null}
                formatValue={formatBytes}
              />
            ),
          }
        : item,
    );
  }, [live, process, history]);
  const stale = process?.status === 'unresponsive' ? `${formatDuration(process.ageMs)} ago` : undefined;
  return <MetricTicker label="Now" items={items} loading={live === undefined} stale={stale} />;
}

export function RateAndMemory({
  live,
  process,
  history,
  series,
}: {
  live: RuntimeLive;
  process: RuntimeProcessLive;
  history: readonly LiveSample[];
  series: RuntimeSeries | undefined;
}): ReactElement {
  const points = useMemo(
    () =>
      history
        .filter((sample) => sample.pid === process.pid)
        .map((sample) => ({ t: sample.t, value: sample.outputTps })),
    [history, process.pid],
  );
  const stack = useMemo(() => stackPoints(history, process.pid), [history, process.pid]);
  const running = process.status === 'running';
  const state = engineState(process);
  // A readout whose phase no lane is in now shows the last window, not this instant: drawn quiet.
  const readouts: LiveStreamReadout[] = [
    {
      key: 'out',
      label: METRIC.outputRate.label,
      value: rateText(process.rates.outputTps),
      unit: METRIC.outputRate.unit,
      tip: METRIC.outputRate.tip,
      quiet: state.lanes.decode === 0,
    },
    {
      key: 'prefill',
      label: METRIC.prefillSpeed.label,
      value: rateText(process.rates.prefillTps),
      unit: METRIC.prefillSpeed.unit,
      tip: METRIC.prefillSpeed.tip,
      quiet: state.lanes.prefill === 0,
    },
  ];
  const mem = process.mem;
  const key = procKeyOf(process);
  const proc = series?.procs.find((row) => row.procKey === key);
  const pools = series?.sched.filter((row) => row.procKey === key) ?? [];
  const poolBytes = pools.reduce((sum, row) => sum + row.latest.memoryCapacityBytes, 0);
  return (
    <PageGrid>
      <Panel className="col-span-12 @md:col-span-7" ariaLabel={METRIC.outputRate.label}>
        <LiveStream
          key={key}
          label={METRIC.outputRate.label}
          unit={METRIC.outputRate.unit}
          points={points}
          nowMs={live.now}
          live={running}
          state={<EngineStateBadge state={state.key} lanes={state.lanes} live={running} />}
          readouts={readouts}
          stack={{ label: LANES.byPhase, unit: LANES.unit, bands: PHASE_BANDS, points: stack }}
          formatValue={formatTps}
          formatTick={axisNumber}
          height={160}
        />
      </Panel>
      <Panel className="col-span-12 @md:col-span-5" title="Memory">
        <div className="grid gap-3">
          <SplitBar
            size="tall"
            label="MLX memory"
            total={mem.wiredLimit > 0 ? mem.wiredLimit : null}
            unknownLabel="limit unknown"
            formatValue={formatBytes}
            segments={[
              { key: 'active', label: 'MLX active', value: mem.active, color: SERIES_A },
              { key: 'cache', label: 'MLX cache (reusable)', value: mem.cache, color: SERIES_B },
            ]}
            threshold={{ value: mem.peak, label: 'Peak' }}
          />
          <KeyValue
            layout="rows"
            ariaLabel="Memory"
            items={[
              { key: 'peak', label: 'Peak', value: formatBytes(mem.peak), hint: 'highest since start', numeric: true },
              {
                key: 'limit',
                label: 'Wired limit',
                value: mem.wiredLimit > 0 ? formatBytes(mem.wiredLimit) : null,
                numeric: true,
              },
              {
                key: 'rss',
                label: 'Process RSS',
                labelTip:
                  'Memory the engine process holds outside Metal: its own heap and mapped files. Most MLX memory is in Metal and is not counted here.',
                value: formatBytes(mem.rss),
                numeric: true,
              },
              {
                key: 'ram',
                label: 'Physical RAM',
                value: proc?.totalMemBytes == null ? null : formatBytes(proc.totalMemBytes),
                numeric: true,
              },
              { key: 'chip', label: 'Chip', value: proc?.chip ?? null },
              {
                key: 'pools',
                label: ENGINE.memory.pools,
                value: pools.length === 0 ? null : formatBytes(poolBytes),
                hint:
                  pools.length === 0
                    ? undefined
                    : pools.length === 1
                      ? ENGINE.memory.poolsHintOne
                      : fill(ENGINE.memory.poolsHint, { n: formatNumber(pools.length) }),
                numeric: true,
              },
            ]}
          />
        </div>
      </Panel>
    </PageGrid>
  );
}
