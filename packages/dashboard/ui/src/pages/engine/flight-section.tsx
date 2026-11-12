import type { ReactElement } from 'react';

import { LiveTurnList, LiveTurnRow } from '../../components/arc-x/live-turn-row/live-turn-row';
import { Panel } from '../../components/arc-x/panel/panel';
import { dispatchWord } from '../../lib/engine-view';
import type { RuntimeProcessLive } from '../../lib/types';

const TURN_CAP = 32;

type LiveTurn = RuntimeProcessLive['turns'][number];

/** One live turn row; display and color use `modelId ?? model` (the alias the client used matches no model). */
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
      model={turn.modelId ?? turn.model}
      alias={turn.modelId !== null && turn.modelId !== turn.model ? turn.model : undefined}
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
  return (
    <Panel title="In flight" description="Turns running in this process.">
      {turns.length === 0 ? (
        <p className="text-body text-secondary">No turns in flight.</p>
      ) : (
        <div className="grid gap-2">
          <LiveTurnList label="Turns in flight">
            {turns.map((turn) => (
              <TurnRowFor key={turn.traceId} turn={turn} now={now} live={live} />
            ))}
          </LiveTurnList>
          {process.turns.length >= TURN_CAP ? (
            <p className="text-caption text-secondary">Showing the first 32.</p>
          ) : null}
        </div>
      )}
    </Panel>
  );
}
