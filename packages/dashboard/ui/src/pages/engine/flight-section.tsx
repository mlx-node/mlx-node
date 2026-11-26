import type { ReactElement } from 'react';

import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import {
  ContextFill,
  LaneSpeed,
  LiveTurnRow,
  PhaseBadge,
  TurnProgress,
  type LaneSpeedFields,
  type LiveTurnRowProps,
} from '../../components/arc-x/live-turn-row/live-turn-row';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Panel } from '../../components/arc-x/panel/panel';
import { SwapText } from '../../components/arc-x/swap-text/swap-text';
import { dispatchWord } from '../../lib/engine-view';
import { formatClock, formatCountNoun } from '../../lib/format';
import { useLiveLanes, visibleLanes, type LiveLane, type LiveTurn } from '../../lib/live-lanes';
import { METRIC } from '../../lib/metric-strings';
import type { RuntimeProcessLive } from '../../lib/types';

const TURN_CAP = 32;

/** The model that serves the turn: the checkpoint, else the name the client used when the engine did not resolve it. */
export function servedModel(turn: Pick<LiveTurn, 'model' | 'modelId'>): string {
  return turn.modelId ?? turn.model;
}

/** The name the client asked for, when it is not the served model (an alias such as `mlx-local`). */
export function askedAlias(turn: Pick<LiveTurn, 'model' | 'modelId'>): string | undefined {
  return turn.modelId !== null && turn.modelId !== turn.model ? turn.model : undefined;
}

/** "2 turns on 1 model": the alias a client asked through is not a second model. */
export function flightSummary(turns: ReadonlyArray<Pick<LiveTurn, 'model' | 'modelId'>>): string {
  const models = new Set(turns.map(servedModel)).size;
  return `${formatCountNoun(turns.length, 'turn', 'turns')} on ${formatCountNoun(models, 'model', 'models')}`;
}

/** "8s ago", "4m 05s ago", "1h 02m ago" from the snapshot clock. */
export function startedAgo(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  if (total < 60) return `${String(total)}s ago`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${String(minutes)}m ${String(total % 60).padStart(2, '0')}s ago`;
  return `${String(Math.floor(minutes / 60))}h ${String(minutes % 60).padStart(2, '0')}m ago`;
}

/** The lane's own clocks and counts: its wait reason, one-shot prefill time and prefill or decode speed. */
function laneFields(lane: LiveLane): LaneSpeedFields & Pick<LiveTurnRowProps, 'waitMs' | 'slotMs'> {
  const turn = lane.turn;
  return {
    phase: lane.endedAt === null ? turn.phase : 'idle',
    prefillDone: turn.prefillDone,
    prefillTotal: turn.prefillTotal,
    generatedTokens: turn.generatedTokens,
    waitMs: turn.waitMs,
    slotMs: turn.slotMs,
    prefillElapsedMs: turn.prefillElapsedMs,
    decodeElapsedMs: turn.decodeElapsedMs,
  };
}

/** One live turn row of the Overview list; display and color use `modelId ?? model`. */
export function TurnRowFor({
  lane,
  now,
  live,
  density,
}: {
  lane: LiveLane;
  now: number;
  live: boolean;
  density?: 'default' | 'compact';
}): ReactElement {
  const turn = lane.turn;
  return (
    <LiveTurnRow
      {...laneFields(lane)}
      model={servedModel(turn)}
      alias={askedAlias(turn)}
      elapsedMs={Math.max(0, now - (lane.endedAt ?? turn.startedAt))}
      promptTokens={turn.promptTokens}
      cachedTokens={turn.cachedTokens}
      ctxWindow={turn.ctxWindow}
      dispatchLabel={dispatchWord(turn.dispatch)}
      density={density}
      live={live}
    />
  );
}

export function FlightSection({ process, now }: { process: RuntimeProcessLive; now: number }): ReactElement {
  const lanes = visibleLanes(useLiveLanes(process.turns, now, process.pid), TURN_CAP);
  const live = process.status === 'running';
  const progressOf = (lane: LiveLane) => ({
    ...laneFields(lane),
    model: servedModel(lane.turn),
    promptTokens: lane.turn.promptTokens,
    cachedTokens: lane.turn.cachedTokens,
    ctxWindow: lane.turn.ctxWindow,
    live,
  });

  const columns: DataColumn<LiveLane>[] = [
    {
      id: 'model',
      header: 'Model',
      width: '32%',
      truncate: true,
      cell: ({ turn }) => {
        const alias = askedAlias(turn);
        return (
          <ModelBadge
            name={servedModel(turn)}
            size="sm"
            fit="width"
            detail={
              alias === undefined ? undefined : (
                <span title={`The client asked for ${alias}; this model served it.`}>
                  asked for <span className="font-mono">{alias}</span>
                </span>
              )
            }
          />
        );
      },
    },
    {
      id: 'phase',
      header: 'Phase',
      width: 116,
      cell: (lane) => <PhaseBadge phase={laneFields(lane).phase} live={live} />,
    },
    {
      id: 'progress',
      header: 'Progress',
      flex: true,
      minWidth: 200,
      truncate: true,
      cell: (lane) => <TurnProgress {...progressOf(lane)} />,
    },
    {
      id: 'speed',
      header: METRIC.laneSpeed.label,
      headerTitle: METRIC.laneSpeed.tip,
      align: 'end',
      width: 96,
      minTableWidth: 700,
      cell: (lane) => <LaneSpeed {...laneFields(lane)} />,
    },
    {
      id: 'started',
      header: 'Started',
      headerTitle: 'How long ago the turn started; for an idle session, how long ago its last call ended',
      align: 'end',
      width: 104,
      cell: ({ turn, endedAt }) => (
        <span
          className="tabular-nums"
          title={endedAt === null ? `Started at ${formatClock(turn.startedAt)}` : `Ended at ${formatClock(endedAt)}`}
        >
          <SwapText text={startedAgo(now - (endedAt ?? turn.startedAt))} granularity="char" />
        </span>
      ),
    },
    {
      id: 'context',
      header: 'Context',
      headerTitle: 'Prompt and output against the model window',
      width: 128,
      minTableWidth: 700,
      cell: (lane) => <ContextFill {...progressOf(lane)} />,
    },
  ];

  if (lanes.length === 0) {
    return (
      <Panel title="In flight" description="Turns running in this process.">
        <p className="text-body text-secondary">No turns in flight.</p>
      </Panel>
    );
  }
  return (
    <Panel title="In flight" description={flightSummary(process.turns)} padding="none">
      <DataTable
        caption="Turns in flight"
        columns={columns}
        rows={lanes}
        rowKey={(lane) => lane.key}
        rowAttrs={(lane) => ({ 'data-trace-id': lane.turn.traceId })}
        footer={process.turns.length >= TURN_CAP ? `Showing the first ${String(TURN_CAP)}.` : undefined}
      />
    </Panel>
  );
}
