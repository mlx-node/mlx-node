import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { ReactElement } from 'react';
import { Link } from 'react-router-dom';

import { IconButton } from '../../components/arc-x/icon-button/icon-button';
import { KeyValue, type KeyValueItem } from '../../components/arc-x/key-value/key-value';
import { ModelBadge } from '../../components/arc-x/model-badge/model-badge';
import { LiveLabel } from '../../components/arc-x/live-label/live-label';
import { Panel } from '../../components/arc-x/panel/panel';
import { Badge, type BadgeTone } from '../../components/arc/badge/badge';
import { formatClock, formatCount, formatDuration, formatNumber, formatRate, formatTps } from '../../lib/format';
import { PREFILL_UNDER_TEXT } from '../../lib/prefill';
import { turnPrefill, type TurnGroup } from '../../lib/sessions';
import type { TurnMetric } from '../../lib/types';
import { NO_TRANSCRIPT_TEXT, TurnTimeline } from './turn-timeline';

export const INSPECTOR_ID = 'turn-inspector';
const NO_GROUP_TEXT = 'The transcript holds no entries of this turn, so there is no timeline for it.';

function stopTone(reason: string): BadgeTone {
  if (reason === 'error') return 'danger';
  if (reason === 'cancelled') return 'warning';
  return 'neutral';
}

function prefillHint(metric: TurnMetric): string {
  const prefill = turnPrefill(metric);
  if (prefill === null) return 'not measured';
  const size = `${formatCount(prefill.tokens)} tokens in ${formatDuration(prefill.ms)}`;
  if (prefill.tps === null) return `${size} · ${PREFILL_UNDER_TEXT}`;
  return metric.modelCalls !== null && prefill.calls < metric.modelCalls
    ? `${size} · ${String(prefill.calls)} of ${String(metric.modelCalls)} model calls`
    : size;
}

function figures(metric: TurnMetric): KeyValueItem[] {
  const prefill = turnPrefill(metric);
  const cached = Math.min(metric.cachedTokens, metric.inputTokens);
  return [
    { key: 'duration', label: 'Duration', value: formatDuration(metric.elapsedMs), numeric: true },
    {
      key: 'ttft',
      label: 'First token',
      value: formatDuration(metric.ttftMs),
      hint:
        metric.modelCalls !== null && metric.modelCalls > 1
          ? `first of ${String(metric.modelCalls)} model calls`
          : undefined,
      numeric: true,
    },
    {
      key: 'prefill',
      label: 'Prefill',
      value: prefill === null || prefill.tps === null ? null : `${formatTps(prefill.tps)} tok/s`,
      hint: prefillHint(metric),
      numeric: true,
    },
    {
      key: 'decode',
      label: 'Decode',
      value: metric.decodeTps === null ? null : `${formatTps(metric.decodeTps)} tok/s`,
      numeric: true,
    },
    {
      key: 'prompt',
      label: 'Prompt',
      value: formatNumber(metric.inputTokens),
      hint: `${formatRate(cached, metric.inputTokens)} cached`,
      numeric: true,
    },
    {
      key: 'output',
      label: 'Output',
      value: formatNumber(metric.outputTokens),
      hint: metric.reasoningTokens > 0 ? `${formatNumber(metric.reasoningTokens)} reasoning` : undefined,
      numeric: true,
    },
  ];
}

export interface TurnInspectorProps {
  sessionId: string;
  number: number;
  /** Selectable turns: the finished ones and the open turn when there is one. */
  count: number;
  /** null for the open turn. */
  metric: TurnMetric | null;
  /** The transcript group of this turn; undefined when the transcript could not be read or holds no such turn. */
  group: TurnGroup | undefined;
  /** The detail body carries a transcript error. */
  transcriptFailed: boolean;
  /** The detail query's live flag: the timeline refetches under the same rule. */
  liveQuery: boolean;
  /** The session runs and its last body is fresh: the open turn's word shimmers. */
  live: boolean;
  /** The session runs (the last body says so). */
  sessionLive: boolean;
  onSelect: (turn: number) => void;
  transcriptLink: string;
}

/** One turn: its figures and its agent timeline. */
export function TurnInspector(props: TurnInspectorProps): ReactElement {
  const { metric, number, group } = props;
  const stop = group?.end?.stopReason ?? null;
  const label = `Turn ${String(number)}`;
  const open = metric === null;
  const suffix = open ? (
    <>
      {' · '}
      {props.sessionLive ? <LiveLabel active={props.live}>in progress</LiveLabel> : 'not finished'}
    </>
  ) : undefined;
  return (
    <div id={INSPECTOR_ID} className="min-w-0 scroll-mt-4">
      <Panel
        title={label}
        titleSuffix={suffix}
        actions={
          <span className="flex flex-wrap items-center justify-end gap-2">
            {metric !== null && metric.model !== null ? <ModelBadge name={metric.model} size="sm" /> : null}
            {stop !== null ? (
              <Badge size="sm" tone={stopTone(stop)}>
                {stop}
              </Badge>
            ) : null}
            {metric !== null ? (
              <span className="text-caption text-secondary tabular-nums" title={new Date(metric.ts).toISOString()}>
                {formatClock(metric.ts)}
              </span>
            ) : null}
            <IconButton
              label="Previous turn"
              size="sm"
              icon={<ChevronLeft size={16} aria-hidden="true" />}
              disabled={number <= 1}
              onClick={() => props.onSelect(number - 1)}
            />
            <IconButton
              label="Next turn"
              size="sm"
              icon={<ChevronRight size={16} aria-hidden="true" />}
              disabled={number >= props.count}
              onClick={() => props.onSelect(number + 1)}
            />
            <Link className="link text-body" to={props.transcriptLink}>
              Show in transcript
            </Link>
          </span>
        }
      >
        <div className="grid min-w-0 gap-4">
          <p className="sr-only" aria-live="polite">
            {label}
          </p>
          {metric !== null ? <KeyValue ariaLabel={`${label} figures`} items={figures(metric)} /> : null}
          {group !== undefined ? (
            <TurnTimeline sessionId={props.sessionId} turn={number} group={group} liveQuery={props.liveQuery} />
          ) : (
            <p className="m-0 text-caption text-secondary">
              {props.transcriptFailed ? NO_TRANSCRIPT_TEXT : NO_GROUP_TEXT}
            </p>
          )}
        </div>
      </Panel>
    </div>
  );
}
