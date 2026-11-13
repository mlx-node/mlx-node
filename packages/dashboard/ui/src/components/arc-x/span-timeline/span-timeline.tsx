import { useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactElement } from 'react';

import { clipBoxes, fitTop } from '../../../lib/float-box';
import { formatDuration as defaultDuration } from '../../../lib/format';
import { ChartTable } from '../chart/chart-table';
import { ChartTooltip } from '../chart/chart-tooltip';
import { niceTicks } from '../chart/scale';
import { useArrivals } from '../chart/use-arrivals';
import { useEntrance, useRichMotion } from '../motion/use-entrance';
import styles from './span-timeline.module.css';

export type SpanKind = 'queue' | 'first' | 'setup' | 'prefill' | 'decode' | 'wait' | 'load' | 'other';

export interface TimelineRow {
  id: string;
  label: string;
  /** Model, token count. */
  sublabel?: string;
  /** Dot color, e.g. modelColor(name).color. */
  color?: string;
}

export interface TimelineSpan {
  id: string;
  rowId: string;
  kind: SpanKind;
  /** Relative to the timeline origin. */
  startMs: number;
  endMs: number;
  /** "Prefill" */
  label: string;
  status?: 'ok' | 'error' | 'cancelled';
  /** Derived, not measured (e.g. prefill time from tokens / tps). Drawn with a dashed outline and said so in the tooltip. */
  estimated?: boolean;
  /** Tooltip rows: tokens, tok/s. */
  detail?: ReadonlyArray<{ label: string; value: string }>;
}

/** e.g. "First token". */
export interface TimelineMarker {
  id: string;
  rowId?: string;
  atMs: number;
  label: string;
}

export interface SpanTimelineProps {
  label: string;
  rows: readonly TimelineRow[];
  /** ≤ 400 drawn; more -> the last row reads "+n more spans" and the table lists all. */
  spans: readonly TimelineSpan[];
  markers?: readonly TimelineMarker[];
  /** Default [0, max end]. */
  domainMs?: readonly [number, number];
  /** Hides the label column: the single-turn strip. */
  compact?: boolean;
  /** Default 28. */
  rowHeight?: number;
  /** Default 148. */
  labelWidth?: number;
  selectedId?: string | null;
  onSelect?: (span: TimelineSpan | null) => void;
  formatDuration?: (ms: number) => string;
}

const MAX_SPANS = 400;
const MAX_ROW_STEP = 4;

const KIND_NAME: Record<SpanKind, string> = {
  queue: 'Queue',
  load: 'Loading the model',
  wait: 'Waiting for a model load',
  first: 'To first token',
  setup: 'Setup',
  prefill: 'Prefill',
  decode: 'Decode',
  other: 'Other',
};
const KIND_ORDER: readonly SpanKind[] = ['queue', 'load', 'wait', 'first', 'setup', 'prefill', 'decode', 'other'];

interface Placed {
  span: TimelineSpan;
  rowIndex: number;
  start: number;
  end: number;
}

function spanName(span: TimelineSpan, durationText: string, startText: string, endText: string): string {
  const parts = [span.label, durationText, `from ${startText} to ${endText}`];
  if (span.status === 'error') parts.push('error');
  if (span.status === 'cancelled') parts.push('cancelled');
  if (span.estimated === true) parts.push('estimated');
  return parts.join(', ');
}

/** A waterfall of spans per row (one turn, or a session's agents). One tab stop; arrows move between spans. */
export function SpanTimeline(props: SpanTimelineProps): ReactElement {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const entrance = useEntrance(rootRef);
  const rich = useRichMotion();
  const format = props.formatDuration ?? ((ms: number) => defaultDuration(ms));
  const rowHeight = props.rowHeight ?? 28;
  const labelWidth = props.compact === true ? 0 : (props.labelWidth ?? 148);
  const rowIndex = new Map(props.rows.map((row, index) => [row.id, index]));

  const all: Placed[] = [];
  for (const span of props.spans) {
    const index = rowIndex.get(span.rowId);
    if (index === undefined) continue;
    all.push({ span, rowIndex: index, start: span.startMs, end: span.endMs });
  }
  const drawable = all.filter((placed) => placed.end > placed.start);
  const drawn = drawable.slice(0, MAX_SPANS);
  const hiddenCount = drawable.length - drawn.length;

  let d0 = props.domainMs?.[0] ?? 0;
  let d1 = props.domainMs?.[1] ?? 0;
  if (props.domainMs === undefined) {
    for (const placed of all) d1 = Math.max(d1, placed.end);
    for (const marker of props.markers ?? []) d1 = Math.max(d1, marker.atMs);
    if (d1 <= d0) d1 = d0 + 1;
  }
  if (d1 <= d0) d1 = d0 + 1;
  const span = d1 - d0;
  const pct = (ms: number): number => ((Math.min(d1, Math.max(d0, ms)) - d0) / span) * 100;
  const axis = niceTicks(d0, d1, 5).ticks.filter((tick) => tick >= d0 && tick <= d1);

  // Spans in reading order: by row, then by start.
  const ordered = [...drawn].sort((a, b) => a.rowIndex - b.rowIndex || a.start - b.start);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [focused, setFocused] = useState(false);
  const rovingId =
    focusId !== null && ordered.some((placed) => placed.span.id === focusId)
      ? focusId
      : (ordered.find((placed) => placed.span.id === props.selectedId)?.span.id ?? ordered[0]?.span.id ?? null);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const arrivals = useArrivals(
    drawn.map((placed) => placed.span.id),
    rich && entrance === 'done',
  );

  const moveTo = (id: string | undefined): void => {
    if (id === undefined) return;
    setFocusId(id);
    buttons.current.get(id)?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const current = ordered.findIndex((placed) => placed.span.id === rovingId);
    if (current === -1) return;
    const here = ordered[current];
    const inRow = ordered.filter((placed) => placed.rowIndex === here.rowIndex);
    const at = inRow.indexOf(here);
    const nearestIn = (row: number): string | undefined => {
      const candidates = ordered.filter((placed) => placed.rowIndex === row);
      let best: Placed | undefined;
      for (const candidate of candidates) {
        if (best === undefined || Math.abs(candidate.start - here.start) < Math.abs(best.start - here.start)) {
          best = candidate;
        }
      }
      return best?.span.id;
    };
    const rowsWithSpans = [...new Set(ordered.map((placed) => placed.rowIndex))];
    const rowAt = rowsWithSpans.indexOf(here.rowIndex);
    switch (event.key) {
      case 'ArrowRight':
        moveTo(inRow[Math.min(inRow.length - 1, at + 1)]?.span.id);
        break;
      case 'ArrowLeft':
        moveTo(inRow[Math.max(0, at - 1)]?.span.id);
        break;
      case 'ArrowDown':
        if (rowAt < rowsWithSpans.length - 1) moveTo(nearestIn(rowsWithSpans[rowAt + 1]));
        break;
      case 'ArrowUp':
        if (rowAt > 0) moveTo(nearestIn(rowsWithSpans[rowAt - 1]));
        break;
      case 'Home':
        moveTo(ordered[0]?.span.id);
        break;
      case 'End':
        moveTo(ordered[ordered.length - 1]?.span.id);
        break;
      case 'Escape':
        props.onSelect?.(null);
        break;
      default:
        return;
    }
    event.preventDefault();
  };

  const markerRow = (marker: TimelineMarker): number | null =>
    marker.rowId === undefined ? null : (rowIndex.get(marker.rowId) ?? null);
  const markers = props.markers ?? [];
  const lastRowStep = Math.min(Math.max(0, props.rows.length - 1), MAX_ROW_STEP);
  let lastMarker = -1;
  let lastMarkerStep = -1;
  markers.forEach((marker, index) => {
    const row = markerRow(marker);
    const step = row === null ? lastRowStep : Math.min(row, MAX_ROW_STEP);
    if (step >= lastMarkerStep) {
      lastMarkerStep = step;
      lastMarker = index;
    }
  });
  const lastRowWithSpans = drawn.reduce((max, placed) => Math.max(max, placed.rowIndex), -1);

  const kinds = KIND_ORDER.filter((kind) => drawn.some((placed) => placed.span.kind === kind));
  const tooltipId = hoverId ?? (focused ? rovingId : null);
  const tipRef = useRef<HTMLDivElement | null>(null);
  const [tipShift, setTipShift] = useState(0);

  useLayoutEffect(() => {
    const tip = tipRef.current;
    const plot = tip?.parentElement;
    let next = 0;
    if (tip !== null && plot !== null && plot !== undefined) {
      const box = plot.getBoundingClientRect();
      const above = box.top + 4 - tip.offsetHeight;
      next = Math.round(fitTop(above, box.bottom - 4, tip.offsetHeight, clipBoxes(tip).view).y - above);
    }
    if (next !== tipShift) setTipShift(next);
  });

  const durationText = (placed: Placed): string => format(placed.end - placed.start);
  const tableRows = all.map((placed) => {
    const rowLabel = props.rows[placed.rowIndex]?.label ?? '';
    return {
      key: placed.span.id,
      current: placed.span.id === props.selectedId,
      cells: [
        rowLabel,
        placed.span.label,
        KIND_NAME[placed.span.kind],
        format(placed.start),
        format(placed.span.endMs),
        format(placed.end - placed.start),
        placed.span.status ?? 'ok',
      ],
    };
  });

  return (
    <div
      ref={rootRef}
      className={styles.root}
      role="group"
      aria-label={props.label}
      data-entrance={entrance}
      data-compact={props.compact === true ? '' : undefined}
      data-selecting={props.selectedId !== undefined && props.selectedId !== null ? '' : undefined}
      style={{ '--row-h': `${String(rowHeight)}px`, '--label-w': `${String(labelWidth)}px` } as CSSProperties}
    >
      <div className={styles.grid}>
        <div className={styles.axisLabel} aria-hidden="true" />
        <div className={styles.axis} aria-hidden="true">
          {axis.map((tick) => (
            <span key={tick} className={styles.tick} style={{ left: `${String(pct(tick))}%` }}>
              {tick === d0 ? '0' : format(tick - d0)}
            </span>
          ))}
        </div>
        {props.rows.map((row, index) => {
          const rowSpans = drawn.filter((placed) => placed.rowIndex === index);
          const rowMarkers = markers.filter((marker) => markerRow(marker) === index || marker.rowId === undefined);
          const step = Math.min(index, MAX_ROW_STEP);
          return (
            <div key={row.id} className={styles.row} style={{ '--step': step } as CSSProperties}>
              {props.compact === true ? null : (
                <div className={styles.label}>
                  <span
                    className={styles.dot}
                    style={{ '--c': row.color ?? 'var(--text-muted)' } as CSSProperties}
                    aria-hidden="true"
                  />
                  <span className={styles.names}>
                    <span className={styles.name}>{row.label}</span>
                    {row.sublabel !== undefined ? <span className={styles.sublabel}>{row.sublabel}</span> : null}
                  </span>
                </div>
              )}
              <div className={styles.plot} onKeyDown={onKeyDown}>
                {axis.map((tick) => (
                  <span
                    key={tick}
                    className={styles.rule}
                    style={{ left: `${String(pct(tick))}%` }}
                    aria-hidden="true"
                  />
                ))}
                <div
                  className={styles.spans}
                  data-entrance-last={index === lastRowWithSpans && lastMarker === -1 ? '' : undefined}
                >
                  {rowSpans.map((placed) => {
                    const id = placed.span.id;
                    const left = pct(placed.start);
                    const right = pct(placed.end);
                    const style = {
                      left: `calc(${String(left)}% + 0.75px)`,
                      width: `max(2px, calc(${String(right - left)}% - 1.5px))`,
                    } as CSSProperties;
                    const name = spanName(placed.span, durationText(placed), format(placed.start), format(placed.end));
                    return (
                      <button
                        key={id}
                        ref={(node) => {
                          if (node === null) buttons.current.delete(id);
                          else buttons.current.set(id, node);
                        }}
                        type="button"
                        className={styles.span}
                        style={style}
                        tabIndex={id === rovingId ? 0 : -1}
                        aria-label={name}
                        aria-pressed={props.selectedId === id}
                        data-kind={placed.span.kind}
                        data-status={placed.span.status ?? 'ok'}
                        data-estimated={placed.span.estimated === true ? '' : undefined}
                        data-arrive={arrivals.arriving.has(id) ? '' : undefined}
                        onAnimationEnd={(event) => {
                          if (event.target === event.currentTarget) arrivals.settle(id);
                        }}
                        onPointerEnter={() => setHoverId(id)}
                        onPointerLeave={() => setHoverId((current) => (current === id ? null : current))}
                        onFocus={() => {
                          setFocusId(id);
                          setFocused(true);
                        }}
                        onBlur={() => setFocused(false)}
                        onClick={() => {
                          setFocusId(id);
                          props.onSelect?.(props.selectedId === id ? null : placed.span);
                        }}
                      >
                        <span className={styles.fill} />
                      </button>
                    );
                  })}
                </div>
                {rowSpans
                  .filter((placed) => placed.span.status === 'error')
                  .map((placed) => (
                    <span
                      key={`${placed.span.id}-x`}
                      className={styles.errorGlyph}
                      style={{ left: `calc(${String(pct(placed.end))}% + 3px)` }}
                      aria-hidden="true"
                    >
                      ⊗
                    </span>
                  ))}
                {rowMarkers.map((marker) => {
                  const index2 = markers.indexOf(marker);
                  const own = markerRow(marker);
                  return (
                    <span
                      key={marker.id}
                      className={styles.marker}
                      data-knob={own === index || (own === null && index === 0) ? '' : undefined}
                      style={{ left: `${String(pct(marker.atMs))}%` }}
                      title={marker.label}
                      data-entrance-last={
                        index2 === lastMarker && (own === index || (own === null && index === props.rows.length - 1))
                          ? ''
                          : undefined
                      }
                      aria-hidden="true"
                    />
                  );
                })}
                {rowSpans.map((placed) => {
                  if (placed.span.id !== tooltipId) return null;
                  // Shifting by the same share of the tip's own width keeps it inside the plot at either edge.
                  const at = `${String((pct(placed.start) + pct(placed.end)) / 2)}%`;
                  return (
                    <div
                      key={`${placed.span.id}-tip`}
                      ref={tipRef}
                      className={styles.tip}
                      style={{
                        left: at,
                        transform: `translateX(-${at})`,
                        translate: tipShift !== 0 ? `0 ${String(tipShift)}px` : undefined,
                      }}
                    >
                      <ChartTooltip
                        title={placed.span.label}
                        rows={[
                          { key: 'duration', label: 'Duration', value: durationText(placed) },
                          {
                            key: 'range',
                            label: 'From',
                            value: `${format(placed.start)} → ${format(placed.end)}`,
                          },
                          ...(placed.span.detail ?? []).map((detail) => ({
                            key: detail.label,
                            label: detail.label,
                            value: detail.value,
                          })),
                        ]}
                        note={
                          placed.span.estimated === true
                            ? 'Estimated, not measured'
                            : placed.span.status === 'cancelled'
                              ? 'Cancelled'
                              : placed.span.status === 'error'
                                ? 'Ended with an error'
                                : undefined
                        }
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
      {hiddenCount > 0 ? <p className={styles.more}>+{hiddenCount} more spans</p> : null}
      {kinds.length > 0 ? (
        <div className={styles.legend} aria-hidden="true">
          {kinds.map((kind) => (
            <span key={kind} className={styles.legendItem}>
              <span className={styles.legendKey} data-kind={kind} />
              {KIND_NAME[kind]}
            </span>
          ))}
        </div>
      ) : null}
      <ChartTable
        caption={props.label}
        table={{ head: ['Row', 'Span', 'Kind', 'Start', 'End', 'Duration', 'Status'], rows: tableRows }}
      />
    </div>
  );
}
