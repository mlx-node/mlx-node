import type { ReactElement, ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { CellGrid } from '../../components/arc-x/cell-grid/cell-grid';
import { ChartCard, type ChartState } from '../../components/arc-x/chart-card/chart-card';
import { KeyValue, type KeyValueItem } from '../../components/arc-x/key-value/key-value';
import { Panel } from '../../components/arc-x/panel/panel';
import { Placeholder } from '../../components/arc-x/placeholder/placeholder';
import { StackedBarChart } from '../../components/arc-x/stacked-bar-chart/stacked-bar-chart';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import SegmentedControl, { type Segment } from '../../components/arc/segmented-control/segmented-control';
import { fill } from '../../lib/chat-strings';
import { ENGINE } from '../../lib/engine-strings';
import { batchUse, occupancyPoints, poolSegments, type BatchUse } from '../../lib/engine-view';
import {
  formatBytes,
  formatClock,
  formatCount,
  formatCountNoun,
  formatDuration,
  formatNumber,
  formatRelativeTime,
} from '../../lib/format';
import { modelColor } from '../../lib/model-colors';
import type { RuntimeProcessLive, RuntimeSeries } from '../../lib/types';
import { PageGrid, PageSection } from '../../shell/page';

const S = ENGINE.pool;

type SchedEntry = RuntimeSeries['sched'][number];
type LiveModel = RuntimeProcessLive['models'][number];

function Pressure({ on, children }: { on: boolean; children: string }): ReactElement {
  if (!on) return <>{children}</>;
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <Badge size="sm" tone="warning">
        {S.pressure}
      </Badge>
      {children}
    </span>
  );
}

function MetricsLine({ before, after }: { before: string; after: string }): ReactElement {
  return (
    <p className="text-caption text-secondary">
      {before}
      <Link to="/metrics" className="link">
        {S.metrics}
      </Link>
      {after}
    </p>
  );
}

/** ` · after 9 turns`, or '' when the engine did not write the count. */
function afterTurns(turns: number | null): string {
  if (turns === null) return '';
  return turns === 1 ? S.afterTurn : fill(S.afterTurns, { n: formatNumber(turns) });
}

/** `14:37:33`, plus `, 4m ago` while a turn runs, plus the turn count. */
function readAt(latest: SchedEntry['latest'], busy: boolean): string {
  const ago = busy ? fill(S.ago, { ago: formatRelativeTime(latest.ts) }) : '';
  return `${formatClock(latest.ts, true)}${ago}${afterTurns(latest.turns)}`;
}

function PoolCard({
  entry,
  use,
  busy,
  chosen,
  className,
}: {
  entry: SchedEntry;
  use: BatchUse;
  busy: boolean;
  chosen: string;
  className: string;
}): ReactElement {
  const latest = entry.latest;
  const footer = `${S.readIdle}${busy ? S.readBusy : ''}`;
  if (latest.allocatedBlocks > 0 || use === 'batched') {
    const pool = poolSegments(latest);
    return (
      <Panel className={className} title={S.poolTitle}>
        <div className="grid gap-3">
          <p className="text-caption text-secondary tabular-nums">
            {fill(S.caption, {
              held: formatNumber(latest.allocatedBlocks),
              capacity: formatNumber(latest.blockCapacity),
              read: readAt(latest, busy),
            })}
          </p>
          <CellGrid
            label={fill(S.gridLabel, { model: chosen })}
            capacity={latest.blockCapacity}
            segments={pool.segments}
            reserved={pool.reserved > 0 ? { value: pool.reserved, label: S.reserved } : undefined}
            threshold={pool.threshold > 0 ? { value: pool.threshold, label: S.watermark } : undefined}
            unit={{ one: 'block', other: 'blocks' }}
            formatValue={formatNumber}
          />
          <p className="text-caption text-secondary">{footer}</p>
        </div>
      </Panel>
    );
  }
  const rows: KeyValueItem[] = [
    {
      key: 'capacity',
      label: S.capacity,
      value: fill(S.capacityValue, { n: formatNumber(latest.blockCapacity) }),
      hint: formatBytes(latest.memoryCapacityBytes),
      numeric: true,
    },
    { key: 'held', label: S.heldAtRest, value: S.heldNone, hint: readAt(latest, busy), numeric: true },
  ];
  return (
    <Panel className={className} title={S.poolTitle}>
      <div className="grid gap-3">
        <KeyValue layout="rows" ariaLabel={S.poolTitle} items={rows} />
        {use === 'alone' ? <p className="text-caption text-secondary">{S.noneHeld}</p> : null}
        <p className="text-caption text-secondary">{footer}</p>
      </div>
    </Panel>
  );
}

function BatchingNote({
  entry,
  use,
  live,
  className,
}: {
  entry: SchedEntry;
  use: 'alone' | 'quiet';
  live: LiveModel | undefined;
  className: string;
}): ReactElement {
  const busy = (live?.inFlight ?? 0) > 0;
  let lines: ReactNode;
  if (use === 'alone') {
    const n = entry.delta.turns;
    lines = (
      <>
        <p className="text-body">{live?.maxConcurrent === 1 ? S.oneAtATime : S.eachAlone}</p>
        <p className="text-caption text-secondary tabular-nums">
          {n === 1 ? S.turnRan : fill(S.turnsRan, { n: formatNumber(n) })}
        </p>
        <p className="text-caption text-secondary">{S.noScheduler}</p>
      </>
    );
  } else {
    let words: string = S.quietRange;
    if (busy) words = S.quietBusy;
    // No reading in the range: turns can run there without an idle moment to read them.
    else if (entry.snapshots.length === 0) words = S.quietUnread;
    else if (entry.latest.turns === 0) words = S.quietNever;
    lines = <p className="text-body">{words}</p>;
  }
  return (
    <Panel className={className} title={S.batchingTitle}>
      <div className="grid gap-2">
        {lines}
        <MetricsLine before={S.seeBefore} after={S.seeAfter} />
      </div>
    </Panel>
  );
}

export function PoolSection({
  series,
  state,
  error,
  procKey,
  process,
  model,
  onModelChange,
  rangeControl,
}: {
  series: RuntimeSeries | undefined;
  state: ChartState;
  error?: { message: string; retry: () => void };
  procKey: string | null;
  process: RuntimeProcessLive | null;
  model: string | null;
  onModelChange: (model: string) => void;
  rangeControl: ReactNode;
}): ReactElement {
  const entries = series?.sched.filter((entry) => entry.procKey === procKey) ?? [];
  const loaded = process?.models ?? [];
  const names = loaded.map((item) => item.name);
  for (const entry of entries) if (!names.includes(entry.model)) names.push(entry.model);
  const chosen = model !== null && names.includes(model) ? model : (names[0] ?? model);
  const entry = entries.find((item) => item.model === chosen);
  const live = loaded.find((item) => item.name === chosen);
  const options: Segment[] = names.map((name) => ({ value: name, label: name }));

  const actions = rangeControl;
  // In the grid, not in the section head: a long list of names then scrolls inside the switch, not past the page.
  const switcher =
    options.length > 1 && chosen !== null ? (
      <div className="col-span-12 min-w-0">
        <SegmentedControl
          label={S.modelSwitch}
          options={options}
          value={chosen}
          onValueChange={(value) => {
            if (names.includes(value)) onModelChange(value);
          }}
        />
      </div>
    ) : null;

  if (series === undefined) {
    const failed = state === 'error';
    return (
      <PageSection title={S.section} actions={actions}>
        <PageGrid>
          {switcher}
          <Panel className="col-span-12 @md:col-span-6" title={S.poolTitle} busy={!failed}>
            {failed ? (
              <div className="grid min-h-40 place-content-center justify-items-center gap-2" role="alert">
                <span className="text-body text-secondary">{error?.message ?? S.loadFailed}</span>
                {error === undefined ? null : (
                  <Button variant="secondary" size="sm" onClick={error.retry}>
                    {S.retry}
                  </Button>
                )}
              </div>
            ) : (
              <div className="grid gap-3">
                <Placeholder width="50%" />
                <Placeholder height={120} radius="control" />
              </div>
            )}
          </Panel>
          <ChartCard
            className="col-span-12 @md:col-span-6"
            title={S.batchTitle}
            subtitle={S.batchSubtitle}
            height={200}
            state={failed ? 'error' : 'loading'}
            error={error}
          >
            {null}
          </ChartCard>
        </PageGrid>
      </PageSection>
    );
  }

  if (entry === undefined) {
    const flat = live !== undefined && !live.paged;
    return (
      <PageSection title={S.section} actions={actions}>
        <PageGrid>
          {switcher}
          <Panel className="col-span-12">
            {flat ? (
              <MetricsLine before={fill(S.flat, { model: chosen ?? S.thisModel })} after={S.flatEnd} />
            ) : (
              <p className="text-caption text-secondary">{S.unread}</p>
            )}
          </Panel>
        </PageGrid>
      </PageSection>
    );
  }

  const name = chosen ?? entry.model;
  const use = batchUse(entry);
  const busy = (live?.inFlight ?? 0) > 0;

  if (use !== 'batched') {
    const half = 'col-span-12 @md:col-span-6 self-start';
    return (
      <PageSection title={S.section} actions={actions}>
        <PageGrid>
          {switcher}
          <PoolCard entry={entry} use={use} busy={busy} chosen={name} className={half} />
          <BatchingNote entry={entry} use={use} live={live} className={half} />
        </PageGrid>
      </PageSection>
    );
  }

  const latest = entry.latest;
  const occupancy = occupancyPoints(entry.delta);
  const delta = entry.delta;
  const color = modelColor(name).color;
  const counters: KeyValueItem[] = [
    ...(latest.turns === null
      ? []
      : [{ key: 'turns', label: S.turns, value: formatNumber(delta.turns), hint: S.turnsHint, numeric: true }]),
    { key: 'steps', label: S.steps, value: formatNumber(delta.globalSteps), numeric: true },
    {
      key: 'admitted',
      label: S.admitted,
      value: formatNumber(delta.admitted),
      hint: fill(S.completed, { n: formatNumber(delta.completed) }),
      numeric: true,
    },
    {
      key: 'preemptions',
      label: S.preemptions,
      value: <Pressure on={delta.preemptions > 0}>{formatNumber(delta.preemptions)}</Pressure>,
      hint: fill(S.preemptionsHint, {
        recomputed: formatNumber(delta.preemptionsRecompute),
        parked: formatNumber(delta.preemptionsSsd),
      }),
      wrapHint: true,
      numeric: true,
    },
    {
      key: 'deferred',
      label: S.deferred,
      value: <Pressure on={delta.admissionDeferredBlocks > 0}>{formatNumber(delta.admissionDeferredBlocks)}</Pressure>,
      hint: fill(S.deferredHint, { n: formatNumber(delta.admissionDeferredState) }),
      wrapHint: true,
      numeric: true,
    },
    { key: 'evicted', label: S.evicted, value: formatNumber(delta.rowsAllocEvicted), numeric: true },
    {
      key: 'restored',
      label: S.restored,
      value: formatBytes(delta.ssdRestoreBytes),
      hint: fill(S.restoredHint, { time: formatDuration(delta.ssdRestoreWaitMs) }),
      numeric: true,
    },
  ];

  return (
    <PageSection title={S.section} actions={actions}>
      <PageGrid>
        {switcher}
        <PoolCard entry={entry} use={use} busy={busy} chosen={name} className="col-span-12 @md:col-span-6" />
        <ChartCard
          className="col-span-12 @md:col-span-6"
          title={S.batchTitle}
          subtitle={S.batchSubtitle}
          height={200}
          state={state === 'ready' || state === 'held' ? (occupancy.steps === 0 ? 'empty' : state) : state}
          empty={{ title: S.batchEmptyTitle, hint: S.batchEmptyHint }}
          footnote={fill(S.batchFootnote, {
            mean: occupancy.mean === null ? '—' : occupancy.mean.toFixed(1),
            largest: formatNumber(latest.maxBatchOccupancy),
            steps: formatCountNoun(occupancy.steps, 'step', 'steps', formatCount),
          })}
        >
          <StackedBarChart
            label={S.batchTitle}
            unit="steps"
            height={200}
            data={occupancy.data}
            series={[{ key: 'steps', label: S.batchSeries, color }]}
            formatValue={formatNumber}
            formatTick={formatCount}
          />
        </ChartCard>
        <Panel className="col-span-12" title={S.countersTitle} description={S.countersDescription}>
          <KeyValue layout="grid" ariaLabel={S.countersTitle} items={counters} />
        </Panel>
      </PageGrid>
    </PageSection>
  );
}
