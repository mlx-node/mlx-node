import { ChevronLeft, ChevronRight } from 'lucide-react';
import { type ReactElement, useMemo, useState } from 'react';

import { ChartCard } from '../../components/arc-x/chart-card/chart-card';
import type { ChartPoint, ChartSeries } from '../../components/arc-x/chart/types';
import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { IconButton } from '../../components/arc-x/icon-button/icon-button';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Panel } from '../../components/arc-x/panel/panel';
import { SplitBar } from '../../components/arc-x/split-bar/split-bar';
import { StackedBarChart } from '../../components/arc-x/stacked-bar-chart/stacked-bar-chart';
import { StatGroup, StatTile } from '../../components/arc-x/stat-tile/stat-tile';
import { TrendChart } from '../../components/arc-x/trend-chart/trend-chart';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { EmptyState } from '../../components/arc/empty-state/empty-state';
import { sessionUrl } from '../../components/session-dialogs';
import {
  formatClock,
  formatCount,
  formatCountNoun,
  formatDuration,
  formatDurationParts,
  formatNumber,
  formatRate,
  formatTps,
} from '../../lib/format';
import { axisNumber } from '../../lib/metrics-view';
import { modelSeries } from '../../lib/model-colors';
import { PREFILL_MIN_TOKENS, PREFILL_SMALL_HINT } from '../../lib/prefill';
import {
  openTurnGroup,
  prefillAverage,
  sessionTitle,
  sessionTitleTip,
  turnPrefill,
  turnTimeParts,
  turnTokenBars,
  type TurnGroup,
} from '../../lib/sessions';
import type { SessionDetailResponse, SubagentRef, TranscriptEntry, TurnMetric } from '../../lib/types';
import { PageGrid, PageSection } from '../../shell/page';
import { TurnInspector } from './turn-inspector';

const WINDOW_TURNS = 60;
const CHART_HEIGHT = 180;
const UNKNOWN_MODEL = 'unknown';

type SubagentEntry = Extract<TranscriptEntry, { kind: 'subagent' }>;

function finite(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function mean(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function modelKeyOf(turn: TurnMetric): string {
  return turn.model !== null && turn.model !== '' ? turn.model : UNKNOWN_MODEL;
}

function speedSeries(turns: readonly TurnMetric[]): ChartSeries[] {
  const names: string[] = [];
  let unknown = false;
  for (const turn of turns) {
    const key = modelKeyOf(turn);
    if (key === UNKNOWN_MODEL) unknown = true;
    else if (!names.includes(key)) names.push(key);
  }
  const series = modelSeries(names);
  if (unknown) series.push({ key: UNKNOWN_MODEL, label: 'Unknown model', color: 'var(--model-other)' });
  return series;
}

function speedPoint(turn: TurnMetric, number: number, value: number | null, label?: string): ChartPoint {
  return {
    key: String(number),
    label: label ?? `Turn ${String(number)} · ${formatClock(turn.ts)}`,
    axisLabel: String(number),
    values: { [modelKeyOf(turn)]: value },
  };
}

interface SubagentRow extends SubagentRef {
  entry: SubagentEntry | undefined;
  turn: number | null;
}

/** Totals over every turn of the session; the token sums leave out a turn with no prompt tokens (it has no usage). */
function SessionTotals({ turns }: { turns: readonly TurnMetric[] }): ReactElement {
  let input = 0;
  let output = 0;
  let cached = 0;
  let reasoning = 0;
  let elapsed = 0;
  let timed = 0;
  let used = 0;
  for (const turn of turns) {
    if (turn.inputTokens > 0) {
      used += 1;
      input += turn.inputTokens;
      output += turn.outputTokens;
      cached += Math.min(turn.cachedTokens, turn.inputTokens);
      reasoning += turn.reasoningTokens;
    }
    if (finite(turn.elapsedMs)) {
      elapsed += turn.elapsedMs;
      timed += 1;
    }
  }
  const ttfts = turns.map((turn) => turn.ttftMs).filter(finite);
  const decodes = turns.map((turn) => turn.decodeTps).filter(finite);
  const prefills = turns.map(turnPrefill);
  const measuredCount = prefills.filter((prefill) => prefill !== null).length;
  const ratedCount = prefills.filter((prefill) => prefill !== null && prefill.tps !== null).length;
  const prefill = ratedCount > 0 ? prefillAverage(prefills) : null;
  let prefillHint = `rated ${formatNumber(ratedCount)} of ${formatCountNoun(measuredCount, 'measured turn', 'measured turns')}`;
  if (ratedCount === 0) {
    prefillHint =
      measuredCount === 0 ? 'not measured in this session' : `no prefill of ${String(PREFILL_MIN_TOKENS)}+ tokens`;
  }
  const ttft = mean(ttfts);
  const ttftParts = formatDurationParts(ttft);
  return (
    <StatGroup label="Session totals" maxColumns={3}>
      <StatTile
        label="Turns"
        value={formatNumber(turns.length)}
        hint={timed > 0 ? `${formatDuration(elapsed)} active` : undefined}
      />
      <StatTile
        label="Prompt"
        value={used === 0 ? '—' : formatCount(input)}
        unit={used === 0 ? undefined : 'tokens'}
        hint={input > 0 ? `${formatRate(cached, input)} cached` : undefined}
        footer={
          input > 0 ? (
            <SplitBar
              label="Prompt tokens"
              size="dense"
              legend={false}
              formatValue={formatCount}
              segments={[
                { key: 'cached', label: 'Cached', value: cached, color: 'var(--data-cached)' },
                { key: 'input', label: 'New', value: input - cached, color: 'var(--data-input)' },
              ]}
            />
          ) : undefined
        }
      />
      <StatTile
        label="Output"
        value={used === 0 ? '—' : formatCount(output)}
        unit={used === 0 ? undefined : 'tokens'}
        hint={reasoning > 0 ? `${formatCount(reasoning)} reasoning` : undefined}
      />
      <StatTile
        label="First token"
        value={ttft === null ? '—' : ttftParts.value}
        unit={ttft === null ? undefined : ttftParts.unit}
        hint={ttfts.length > 0 ? `mean of ${formatCountNoun(ttfts.length, 'turn', 'turns')}` : undefined}
        sampleCount={ttfts.length}
      />
      <StatTile
        label="Prefill speed"
        value={prefill === null ? '—' : formatTps(prefill)}
        unit={prefill === null ? undefined : 'tok/s'}
        hint={prefillHint}
        sampleCount={prefill === null ? undefined : ratedCount}
      />
      <StatTile
        label="Decode speed"
        value={decodes.length === 0 ? '—' : formatTps(mean(decodes))}
        unit={decodes.length === 0 ? undefined : 'tok/s'}
        hint={decodes.length > 0 ? `mean of ${formatCountNoun(decodes.length, 'turn', 'turns')}` : undefined}
        sampleCount={decodes.length}
      />
    </StatGroup>
  );
}

export interface ActivityProps {
  data: SessionDetailResponse;
  groups: readonly TurnGroup[];
  /** 1-based, within 1..turns.length, or turns.length + 1 for the open turn. */
  selected: number;
  onSelect: (turn: number) => void;
  transcriptLink: (turn: number) => string;
  /** The detail query's live flag. */
  liveQuery: boolean;
  /** The session runs and its last body is fresh. */
  live: boolean;
}

/** What each turn cost, the selected turn up close, and the subagents the session spawned. */
export function Activity(props: ActivityProps): ReactElement {
  const { data, selected } = props;
  const turns = data.turns;
  const session = data.session;
  const openGroup = openTurnGroup(props.groups);
  const count = turns.length + (openGroup !== null ? 1 : 0);

  const [windowStart, setWindowStart] = useState<number | null>(null);
  const [seenSelected, setSeenSelected] = useState(selected);
  const lastStart = Math.max(0, turns.length - WINDOW_TURNS);
  let start = windowStart ?? lastStart;
  if (seenSelected !== selected) {
    setSeenSelected(selected);
    const index = selected - 1;
    if (index < start || index >= start + WINDOW_TURNS) {
      start = Math.min(lastStart, Math.max(0, index - WINDOW_TURNS + 1));
      setWindowStart(start);
    }
  }
  const end = Math.min(turns.length, start + WINDOW_TURNS);
  const shown = useMemo(() => turns.slice(start, end), [turns, start, end]);

  const subagentRows = useMemo((): SubagentRow[] => {
    const entries = new Map<string, { entry: SubagentEntry; turn: number | null }>();
    for (const group of props.groups) {
      for (const entry of group.entries) {
        if (entry.kind === 'subagent')
          entries.set(entry.childId, { entry, turn: group.number > 0 ? group.number : null });
      }
    }
    return session.subagents.map((child) => {
      const found = entries.get(child.id);
      const turn = found?.turn ?? null;
      return { ...child, entry: found?.entry, turn: turn !== null && turn <= turns.length ? turn : null };
    });
  }, [props.groups, session.subagents, turns.length]);

  const subagents = session.subagents.length > 0 ? <Subagents rows={subagentRows} onSelect={props.onSelect} /> : null;
  const metric = turns[selected - 1] ?? null;
  const group =
    metric === null
      ? (openGroup ?? undefined)
      : props.groups.find((candidate) => candidate.number === selected && candidate.metric !== null);
  const inspector =
    selected >= 1 && selected <= count ? (
      <TurnInspector
        sessionId={session.id}
        number={selected}
        count={count}
        metric={metric}
        group={group}
        transcriptFailed={data.transcriptError !== undefined}
        liveQuery={props.liveQuery}
        live={props.live}
        sessionLive={session.live}
        onSelect={props.onSelect}
        transcriptLink={props.transcriptLink(selected)}
      />
    ) : null;

  if (turns.length === 0) {
    return (
      <div className="grid min-w-0 gap-4">
        <EmptyState title="No completed turns yet" description="Turn metrics appear when a turn finishes." />
        {inspector}
        {subagents}
      </div>
    );
  }

  const tokenData: ChartPoint[] = shown.map((turn, offset) => {
    const number = start + offset + 1;
    const bars = turnTokenBars(turn);
    return {
      key: String(number),
      label: `Turn ${String(number)} · ${formatClock(turn.ts)}`,
      axisLabel: String(number),
      values: { cached: bars.cached, uncached: bars.uncached, output: bars.output },
    };
  });
  const tokenEmpty = tokenData.every(
    (point) => (point.values.cached ?? 0) + (point.values.uncached ?? 0) + (point.values.output ?? 0) === 0,
  );
  const timeData: ChartPoint[] = shown.map((turn, offset) => {
    const number = start + offset + 1;
    const parts = turnTimeParts(turn);
    const seconds = (ms: number | null): number | null => (ms === null ? null : ms / 1000);
    return {
      key: String(number),
      label: `Turn ${String(number)} · ${formatClock(turn.ts)}`,
      axisLabel: String(number),
      values: { first: seconds(parts.firstTokenMs), decode: seconds(parts.decodeMs), other: seconds(parts.otherMs) },
    };
  });
  const timeEmpty = timeData.every((point) => point.values.first === null);
  const series = speedSeries(turns);
  const decodeData = shown.map((turn, offset) =>
    speedPoint(turn, start + offset + 1, finite(turn.decodeTps) ? turn.decodeTps : null),
  );
  const decodeEmpty = decodeData.every((point) => Object.values(point.values).every((value) => value === null));
  const prefillData = shown.map((turn, offset) => {
    const number = start + offset + 1;
    const prefill = turnPrefill(turn);
    if (prefill === null || prefill.tps === null) return speedPoint(turn, number, null);
    const calls =
      turn.modelCalls !== null && prefill.calls < turn.modelCalls
        ? ` · ${String(prefill.calls)} of ${String(turn.modelCalls)} model calls`
        : '';
    return speedPoint(
      turn,
      number,
      prefill.tps,
      `Turn ${String(number)} · ${formatCount(prefill.tokens)} tokens in ${formatDuration(prefill.ms)}${calls}`,
    );
  });
  const prefillEmpty = prefillData.every((point) => Object.values(point.values).every((value) => value === null));
  const prefillMeasuredHere = shown.some((turn) => turnPrefill(turn) !== null);
  const prefillMeasuredAnywhere = prefillMeasuredHere || turns.some((turn) => turnPrefill(turn) !== null);
  const selectedKey = String(selected);
  const selectPoint = (point: ChartPoint): void => props.onSelect(Number(point.key));

  return (
    <div className="grid min-w-0 gap-4">
      <SessionTotals turns={turns} />

      <PageSection
        title="Per turn"
        actions={
          turns.length > WINDOW_TURNS ? (
            <span className="flex items-center gap-1">
              <span className="text-body text-secondary tabular-nums">
                {`Turns ${String(start + 1)}–${String(end)} of ${String(turns.length)}`}
              </span>
              <IconButton
                label="Earlier turns"
                size="sm"
                icon={<ChevronLeft size={16} aria-hidden="true" />}
                disabled={start === 0}
                onClick={() => setWindowStart(Math.max(0, start - WINDOW_TURNS))}
              />
              <IconButton
                label="Later turns"
                size="sm"
                icon={<ChevronRight size={16} aria-hidden="true" />}
                disabled={end >= turns.length}
                onClick={() => setWindowStart(Math.min(lastStart, start + WINDOW_TURNS))}
              />
            </span>
          ) : undefined
        }
      >
        <PageGrid>
          <ChartCard
            className="col-span-12 @md:col-span-6"
            title="Tokens per turn"
            subtitle="Prompt (cached and new) and output · tokens"
            height={CHART_HEIGHT}
            state={tokenEmpty ? 'empty' : 'ready'}
            empty={{ title: 'No token usage recorded' }}
          >
            <StackedBarChart
              label="Tokens per turn"
              unit="tokens"
              height={CHART_HEIGHT}
              data={tokenData}
              series={[
                { key: 'cached', label: 'Prompt, cached', color: 'var(--data-cached)' },
                { key: 'uncached', label: 'Prompt, new', color: 'var(--data-input)' },
                { key: 'output', label: 'Output', color: 'var(--data-output)' },
              ]}
              formatValue={formatNumber}
              formatTick={formatCount}
              onSelect={selectPoint}
              selectedKey={selectedKey}
            />
          </ChartCard>
          <ChartCard
            className="col-span-12 @md:col-span-6"
            title="Time per turn"
            subtitle="To first token, decode, tools and other · seconds"
            height={CHART_HEIGHT}
            state={timeEmpty ? 'empty' : 'ready'}
            empty={{ title: 'No turn durations recorded' }}
            footnote="Decode time is output tokens ÷ decode speed. Prompt processing of later model calls is in Tools and other."
          >
            <StackedBarChart
              label="Time per turn"
              unit="s"
              height={CHART_HEIGHT}
              data={timeData}
              series={[
                { key: 'first', label: 'To first token', color: 'var(--phase-prefill)', pattern: 'alt' },
                { key: 'decode', label: 'Decode', color: 'var(--phase-decode)' },
                { key: 'other', label: 'Tools and other', color: 'var(--phase-tool)' },
              ]}
              formatTick={axisNumber}
              onSelect={selectPoint}
              selectedKey={selectedKey}
            />
          </ChartCard>
          <ChartCard
            className="col-span-12 @md:col-span-6"
            title="Decode speed per turn"
            subtitle="Tokens per second"
            height={CHART_HEIGHT}
            state={decodeEmpty ? 'empty' : 'ready'}
            empty={{ title: 'No decode speed reported' }}
          >
            <TrendChart
              label="Decode speed per turn"
              unit="tok/s"
              height={CHART_HEIGHT}
              data={decodeData}
              series={series}
              formatValue={formatTps}
              formatTick={axisNumber}
              selectedKey={selectedKey}
            />
          </ChartCard>
          <ChartCard
            className="col-span-12 @md:col-span-6"
            title="Prefill speed per turn"
            subtitle={`Tokens per second · measured prefill of ${String(PREFILL_MIN_TOKENS)}+ tokens`}
            height={CHART_HEIGHT}
            state={prefillEmpty ? 'empty' : 'ready'}
            empty={
              prefillMeasuredHere
                ? { title: `No prefill of ${String(PREFILL_MIN_TOKENS)}+ tokens`, hint: PREFILL_SMALL_HINT }
                : {
                    title: prefillMeasuredAnywhere
                      ? 'Prefill was not measured for these turns'
                      : 'Prefill was not measured in this session',
                    hint: "The session file holds the first model call's time to first token only. Prefill speed needs the prefill time of each call.",
                  }
            }
          >
            <TrendChart
              label="Prefill speed per turn"
              unit="tok/s"
              height={CHART_HEIGHT}
              data={prefillData}
              series={series}
              formatValue={formatTps}
              formatTick={axisNumber}
              selectedKey={selectedKey}
            />
          </ChartCard>
        </PageGrid>
      </PageSection>

      {inspector}

      {subagents}
    </div>
  );
}

function Subagents({
  rows,
  onSelect,
}: {
  rows: readonly SubagentRow[];
  onSelect: (turn: number) => void;
}): ReactElement {
  const columns: DataColumn<SubagentRow>[] = [
    {
      id: 'subagent',
      header: 'Subagent',
      cell: (row) =>
        row.exists ? (
          <span className="block truncate font-medium text-foreground" title={sessionTitleTip(row)}>
            {sessionTitle(row)}
          </span>
        ) : (
          <span className="block truncate text-muted" title={`${sessionTitleTip(row)} (folder missing)`}>
            {`${sessionTitle(row)} (folder missing)`}
          </span>
        ),
      sortValue: (row) => sessionTitle(row).toLowerCase(),
    },
    {
      id: 'type',
      header: 'Type',
      cell: (row) =>
        row.type !== '' ? (
          <Badge size="sm" outline>
            {row.type}
          </Badge>
        ) : null,
      sortValue: (row) => row.type,
      width: 168,
      minTableWidth: 620,
    },
    {
      id: 'model',
      header: 'Model',
      cell: (row) => (row.model !== null ? <ModelBadge name={row.model} size="sm" maxChars={20} /> : <span>—</span>),
      sortValue: (row) => row.model,
      width: 160,
      minTableWidth: 700,
    },
    {
      id: 'status',
      header: 'Status',
      cell: (row) => (
        <span
          className={row.status === 'failed' ? 'text-danger' : 'text-secondary'}
          title={row.entry?.error ?? undefined}
        >
          {row.status}
        </span>
      ),
      sortValue: (row) => row.status,
      width: 96,
    },
    {
      id: 'duration',
      header: 'Duration',
      cell: (row) => <span className="tabular-nums">{formatDuration(row.entry?.durationMs ?? null)}</span>,
      sortValue: (row) => row.entry?.durationMs ?? null,
      align: 'end',
      width: 88,
    },
    {
      id: 'tokens',
      header: 'Tokens',
      cell: (row) => <span className="tabular-nums">{formatNumber(row.entry?.tokens ?? null)}</span>,
      sortValue: (row) => row.entry?.tokens ?? null,
      align: 'end',
      width: 88,
      minTableWidth: 640,
    },
    {
      id: 'turn',
      header: 'Turn',
      cell: (row) =>
        row.turn !== null ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            aria-label={`Select turn ${String(row.turn)}`}
            onClick={(event) => {
              event.stopPropagation();
              if (row.turn !== null) onSelect(row.turn);
            }}
          >
            {String(row.turn)}
          </Button>
        ) : (
          <span>—</span>
        ),
      sortValue: (row) => row.turn,
      align: 'end',
      width: 72,
    },
  ];
  return (
    <Panel title="Subagents" padding="none">
      <DataTable
        caption="Subagents"
        density="compact"
        columns={columns}
        rows={rows}
        rowKey={(row) => row.id}
        rowTo={(row) => (row.exists ? sessionUrl(row.id) : undefined)}
        empty={{ title: 'No subagents', description: 'This session spawned none.' }}
      />
    </Panel>
  );
}
