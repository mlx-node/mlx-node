import type { ReactElement } from 'react';

import { AgentTimeline, AgentTimelineLoading } from '../../components/arc-x/agent-timeline/agent-timeline';
import { Alert } from '../../components/arc/alert/alert';
import { Button } from '../../components/arc/button/button';
import { expectedLanes, timelineAxis, timelineModel } from '../../lib/agent-timeline';
import { formatDuration } from '../../lib/format';
import { LIVE_FRESH_MS, turnTitle, useFreshUntil, type TurnGroup } from '../../lib/sessions';
import type { SessionTimelineResponse } from '../../lib/types';
import { useJson } from '../../lib/use-api';

export const NO_TRANSCRIPT_TEXT = 'The transcript could not be read, so there is no timeline for this turn.';
const NO_TIMES_TEXT = 'No event times were recorded for this turn.';

export interface TurnTimelineProps {
  sessionId: string;
  /** 1-based turn number; the query key. */
  turn: number;
  /** The transcript group of this turn: gives the title and the loading shape. */
  group: TurnGroup;
  /** Same flag as the detail query: refetch on data events while the session runs. */
  liveQuery: boolean;
}

export function TimelineLoadError(props: { message: string; onRetry: () => void }): ReactElement {
  return (
    <Alert tone="danger" title="Could not load this turn's timeline">
      <span className="grid justify-items-start gap-2">
        <span>{props.message}</span>
        <Button variant="secondary" size="sm" onClick={props.onRetry}>
          Retry
        </Button>
      </span>
    </Alert>
  );
}

export function TimelineNoTimes(props: { label: string; title: string }): ReactElement {
  return (
    <section className="grid min-w-0 gap-1" aria-label={props.label}>
      <h3 className="m-0 truncate text-strong font-medium">{props.title}</h3>
      <p className="m-0 text-caption text-secondary">{NO_TIMES_TEXT}</p>
    </section>
  );
}

/** The selected turn as an agent timeline, from the timeline route. */
export function TurnTimeline(props: TurnTimelineProps): ReactElement {
  const path = `/sessions/${encodeURIComponent(props.sessionId)}/timeline?turn=${String(props.turn)}`;
  const timeline = useJson<SessionTimelineResponse>(path, { live: props.liveQuery });
  const body = timeline.data;
  const fresh = useFreshUntil(body === undefined ? null : body.asOfMs + LIVE_FRESH_MS);
  const title = turnTitle(props.group.entries);
  const label = `Turn ${String(props.turn)} timeline`;

  if (body === undefined) {
    if (timeline.error !== undefined)
      return <TimelineLoadError message={timeline.error.message} onRetry={timeline.reload} />;
    return (
      <AgentTimelineLoading label={label} title={title} lanes={expectedLanes(props.group.entries)} surface="flush" />
    );
  }
  if (body.transcriptError !== undefined)
    return <p className="m-0 text-caption text-secondary">{NO_TRANSCRIPT_TEXT}</p>;
  if (body.range === null) return <TimelineNoTimes label={label} title={title} />;
  const model = timelineModel(body, { fresh, formatDuration });
  return (
    <div className="grid min-w-0 gap-3">
      <AgentTimeline
        key={props.turn}
        surface="flush"
        label={label}
        title={title}
        lanes={model.lanes}
        spans={model.spans}
        handoffs={model.handoffs}
        reports={model.reports}
        durationMs={model.durationMs}
        turnStartMs={model.turnStartMs}
        mode={model.mode}
        outcome={model.outcome}
        formatDuration={formatDuration}
        axisUnit={timelineAxis}
      />
      <p className="m-0 text-caption text-secondary">{model.footnote}</p>
    </div>
  );
}
