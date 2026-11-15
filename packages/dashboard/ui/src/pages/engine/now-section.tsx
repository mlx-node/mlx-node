import { useMemo, type ReactElement } from 'react';

import { KeyValue } from '../../components/arc-x/key-value/key-value';
import { LiveStream } from '../../components/arc-x/live-stream/live-stream';
import { MetricTicker, type TickerItem } from '../../components/arc-x/metric-ticker/metric-ticker';
import { Panel } from '../../components/arc-x/panel/panel';
import { Meter, SplitBar } from '../../components/arc-x/split-bar/split-bar';
import { fill } from '../../lib/chat-strings';
import { ENGINE } from '../../lib/engine-strings';
import { nowItems, procKeyOf } from '../../lib/engine-view';
import { formatBytes, formatDuration, formatNumber, formatTps } from '../../lib/format';
import type { LiveSample } from '../../lib/live-history';
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
      history.filter((sample) => sample.pid === process.pid).map((sample) => ({ t: sample.t, value: sample.outTps })),
    [history, process.pid],
  );
  const running = process.status === 'running';
  const tps = process.decodeTps ?? 0;
  const status = !running ? 'Not responding' : tps > 0 || process.turns.length > 0 ? 'All turns in flight' : 'Idle';
  const mem = process.mem;
  const key = procKeyOf(process);
  const proc = series?.procs.find((row) => row.procKey === key);
  const pools = series?.sched.filter((row) => row.procKey === key) ?? [];
  const poolBytes = pools.reduce((sum, row) => sum + row.latest.memoryCapacityBytes, 0);
  return (
    <PageGrid>
      <Panel className="col-span-12 @md:col-span-7" ariaLabel="Output rate">
        <LiveStream
          key={key}
          label="Output rate"
          unit="tok/s"
          points={points}
          nowMs={live.now}
          live={running}
          status={status}
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
                label: 'Node RSS',
                value: formatBytes(mem.rss),
                hint: 'counted apart from MLX',
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
