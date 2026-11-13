import type { ReactElement } from 'react';

import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { ContextFill, LiveTurnRow, PhaseBadge, TurnProgress } from '../../components/arc-x/live-turn-row/live-turn-row';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { Panel } from '../../components/arc-x/panel/panel';
import { SwapText } from '../../components/arc-x/swap-text/swap-text';
import { dispatchWord } from '../../lib/engine-view';
import { formatClock, formatCountNoun } from '../../lib/format';
import type { RuntimeProcessLive } from '../../lib/types';

const TURN_CAP = 32;

type LiveTurn = RuntimeProcessLive['turns'][number];

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

/** One live turn row of the Overview list; display and color use `modelId ?? model`. */
export function TurnRowFor({
  turn,
  now,
  live,
  density,
}: {
  turn: LiveTurn;
  now: number;
  live: boolean;
  density?: 'default' | 'compact';
}): ReactElement {
  return (
    <LiveTurnRow
      model={servedModel(turn)}
      alias={askedAlias(turn)}
      phase={turn.phase}
      elapsedMs={Math.max(0, now - turn.startedAt)}
      prefillDone={turn.prefillDone}
      prefillTotal={turn.prefillTotal}
      generatedTokens={turn.generatedTokens}
      promptTokens={turn.promptTokens}
      ctxWindow={turn.ctxWindow}
      dispatchLabel={dispatchWord(turn.dispatch)}
      density={density}
      live={live}
    />
  );
}

export function FlightSection({ process, now }: { process: RuntimeProcessLive; now: number }): ReactElement {
  const turns = [...process.turns].sort((a, b) => a.startedAt - b.startedAt).slice(0, TURN_CAP);
  const live = process.status === 'running';
  const progressOf = (turn: LiveTurn) => ({
    model: servedModel(turn),
    phase: turn.phase,
    prefillDone: turn.prefillDone,
    prefillTotal: turn.prefillTotal,
    generatedTokens: turn.generatedTokens,
    promptTokens: turn.promptTokens,
    ctxWindow: turn.ctxWindow,
    live,
  });

  const columns: DataColumn<LiveTurn>[] = [
    {
      id: 'model',
      header: 'Model',
      width: '32%',
      truncate: true,
      cell: (turn) => {
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
    { id: 'phase', header: 'Phase', width: 100, cell: (turn) => <PhaseBadge phase={turn.phase} live={live} /> },
    {
      id: 'progress',
      header: 'Progress',
      flex: true,
      minWidth: 200,
      truncate: true,
      cell: (turn) => <TurnProgress {...progressOf(turn)} />,
    },
    {
      id: 'started',
      header: 'Started',
      headerTitle: 'How long ago the turn started',
      align: 'end',
      width: 104,
      cell: (turn) => (
        <span className="tabular-nums" title={`Started at ${formatClock(turn.startedAt)}`}>
          <SwapText text={startedAgo(now - turn.startedAt)} granularity="char" />
        </span>
      ),
    },
    {
      id: 'context',
      header: 'Context',
      headerTitle: 'Prompt and output against the model window',
      width: 128,
      minTableWidth: 700,
      cell: (turn) => <ContextFill {...progressOf(turn)} />,
    },
  ];

  if (turns.length === 0) {
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
        rows={turns}
        rowKey={(turn) => turn.traceId}
        rowAttrs={(turn) => ({ 'data-trace-id': turn.traceId })}
        footer={process.turns.length >= TURN_CAP ? `Showing the first ${String(TURN_CAP)}.` : undefined}
      />
    </Panel>
  );
}
