import {
  createContext,
  useContext,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type ReactElement,
  type ReactNode,
  type RefObject,
} from 'react';

import { cn } from '../../../lib/cn';
import { clipBoxes, fitLeft, fitTop } from '../../../lib/float-box';
import { useEntrance, type EntranceState } from '../motion/use-entrance';
import { SwapText } from '../swap-text/swap-text';
import { ChartTable, type ChartTableSpec } from './chart-table';
import styles from './chart.module.css';

/** The y gutter: 44 px of tick labels, then 10 px before the plot. */
export const GUTTER_PX = 44;
export const GUTTER_GAP_PX = 10;

/** Plot width for a figure of `width` px. */
export function plotWidthOf(width: number, rightPad = 0, gutter = true): number {
  return Math.max(40, width - (gutter ? GUTTER_PX + GUTTER_GAP_PX : 0) - rightPad);
}

/** True while the card shows the previous answer (ChartCard state="held"): the scrub is off. */
export const ChartHeldContext = createContext(false);

export interface YAxisSpec {
  ticks: readonly number[];
  /** Value -> plot y in px. */
  y: (value: number) => number;
  format: (value: number) => string;
}

export interface XLabel {
  key: string;
  x: number;
  text: string;
}

export interface ScrubSpec {
  /** Positions the slider steps through. */
  count: number;
  /** Plot coordinates of a position; y null = the tooltip floats at the top. */
  anchor: (index: number) => { x: number; y: number | null };
  /** The position under the pointer, or null. */
  nearest: (x: number, y: number) => number | null;
  /** aria-valuetext of a position. */
  reading: (index: number) => string;
  tooltip: (index: number) => ReactNode;
  onActivate?: (index: number) => void;
  /** Draw a vertical crosshair at the active position. */
  crosshair?: boolean;
  /** The position read when none is active (usually the newest). */
  rest: number | null;
  /** aria-valuetext when there is nothing to read. */
  emptyReading?: string;
  /** Extra keys (LiveStream: Space pauses). Return true when handled. */
  onKey?: (event: KeyboardEvent<HTMLDivElement>) => boolean;
  /** Fires whenever the active position changes (null = released). */
  onActiveChange?: (index: number | null) => void;
}

export interface MarksState {
  active: number | null;
  entrance: EntranceState;
}

export interface ChartFrameProps {
  figureRef: RefObject<HTMLElement | null>;
  label: string;
  unit: string;
  /** Plot height in px. */
  height: number;
  plotWidth: number;
  /** Extra room right of the plot (direct labels). */
  rightPad?: number;
  yAxis: YAxisSpec | null;
  xLabels?: readonly XLabel[];
  /** Centered under the x labels: "New prompt tokens · tokens". */
  xTitle?: string;
  legend?: ReactNode;
  table: ChartTableSpec;
  scrub: ScrubSpec | null;
  /** Centered text in the plot box instead of marks. */
  empty?: string | null;
  marks: (state: MarksState) => ReactNode;
  /** A div with no figure role (the caller's root is the figure). */
  bare?: boolean;
  /** The entrance state comes from the caller (it observes its own root). */
  entrance?: EntranceState;
  className?: string;
  style?: CSSProperties;
}

const PAGE_FRACTION = 0.1;

function clampIndex(index: number, count: number): number {
  return Math.min(count - 1, Math.max(0, index));
}

/**
 * The figure every own chart draws into: legend, y gutter (unit on top), grid, x labels, the one-tab-stop scrub
 * (role="slider"), the tooltip, the hidden data table and the entrance state. The chart supplies only the marks.
 */
export function ChartFrame(props: ChartFrameProps): ReactElement {
  const held = useContext(ChartHeldContext);
  const ownEntrance = useEntrance(props.figureRef);
  const entrance = props.entrance ?? ownEntrance;
  const [activeRaw, setActiveRaw] = useState<number | null>(null);
  const scrub = props.scrub;
  const count = scrub?.count ?? 0;
  const active = activeRaw !== null && activeRaw < count && !held ? activeRaw : null;

  const setActive = (next: number | null): void => {
    setActiveRaw(next);
    scrub?.onActiveChange?.(next);
  };

  const gutter = props.yAxis !== null;
  const rightPad = props.rightPad ?? 0;
  const y = props.yAxis;
  const bottom = props.height;

  const onPointerMove = (event: PointerEvent<HTMLDivElement>): void => {
    if (scrub === null || held || count === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const next = scrub.nearest(event.clientX - rect.left, event.clientY - rect.top);
    if (next !== active) setActive(next);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (scrub === null || held) return;
    if (scrub.onKey?.(event) === true) {
      event.preventDefault();
      return;
    }
    if (count === 0) return;
    const start = active ?? scrub.rest ?? count - 1;
    const page = Math.max(1, Math.round(count * PAGE_FRACTION));
    let next: number | null | undefined;
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        next = active === null ? start : clampIndex(start + 1, count);
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
        next = active === null ? start : clampIndex(start - 1, count);
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = count - 1;
        break;
      case 'PageUp':
        next = clampIndex(start + page, count);
        break;
      case 'PageDown':
        next = clampIndex(start - page, count);
        break;
      case 'Escape':
        next = null;
        break;
      case 'Enter':
        if (scrub.onActivate !== undefined && (active ?? scrub.rest) !== null) {
          event.preventDefault();
          scrub.onActivate(active ?? scrub.rest ?? 0);
        }
        return;
      default:
        return;
    }
    event.preventDefault();
    setActive(next);
  };

  const readIndex = active ?? scrub?.rest ?? null;
  const reading =
    scrub === null || readIndex === null || readIndex >= count
      ? (scrub?.emptyReading ?? 'No data')
      : scrub.reading(readIndex);

  const anchor = scrub !== null && active !== null ? scrub.anchor(active) : null;
  const edge =
    anchor === null ? 'middle' : anchor.x < 72 ? 'start' : anchor.x > props.plotWidth - 72 ? 'end' : 'middle';
  const tooltipOffset = edge === 'start' ? '8px' : edge === 'end' ? 'calc(-100% - 8px)' : '-50%';
  const tipRef = useRef<HTMLDivElement | null>(null);
  const [tipShift, setTipShift] = useState({ x: 0, y: 0 });
  const tipX = anchor?.x ?? null;
  const tipY = anchor?.y ?? null;

  useLayoutEffect(() => {
    const tip = tipRef.current;
    const plot = tip?.parentElement;
    let next = { x: 0, y: 0 };
    if (tip !== null && plot !== null && plot !== undefined && tipX !== null) {
      const box = plot.getBoundingClientRect();
      const view = clipBoxes(tip).view;
      const width = tip.offsetWidth;
      const height = tip.offsetHeight;
      const point = { x: box.left + tipX, y: box.top + Math.max(0, tipY ?? 0) };
      const left = edge === 'start' ? point.x + 8 : edge === 'end' ? point.x - width - 8 : point.x - width / 2;
      const top = tipY === null ? point.y + 4 : point.y - 10 - height;
      const below = tipY === null ? top : point.y + 10;
      next = {
        x: Math.round(fitLeft(left, width, view) - left),
        y: Math.round(fitTop(top, below, height, view).y - top),
      };
    }
    if (next.x !== tipShift.x || next.y !== tipShift.y) setTipShift(next);
  });

  const Root = props.bare === true ? 'div' : 'figure';
  return (
    <Root
      ref={props.figureRef as RefObject<HTMLDivElement>}
      className={cn(styles.root, props.className)}
      style={props.style}
      aria-label={props.bare === true ? undefined : props.label}
      data-entrance={entrance}
      data-held={held ? '' : undefined}
    >
      {props.legend}
      <div className={styles.frame} data-gutter={gutter ? undefined : 'none'}>
        <span className={styles.unit} aria-hidden="true">
          {gutter ? props.unit : ''}
        </span>
        {gutter ? (
          <div className={styles.gutter} style={{ height: props.height }} aria-hidden="true">
            {y?.ticks.map((tick, index) => (
              <span
                key={index}
                className={styles.tick}
                style={{ transform: `translateY(calc(${String(y.y(tick))}px - 50%))` }}
              >
                <SwapText text={y.format(tick)} />
              </span>
            ))}
          </div>
        ) : null}
        <div className={styles.plot} style={{ height: props.height }}>
          <svg
            className={styles.layer}
            width={props.plotWidth + rightPad}
            height={props.height}
            aria-hidden="true"
            focusable="false"
          >
            {y?.ticks.map((tick, index) => (
              <line
                key={index}
                className={styles.gridLine}
                x1={0}
                x2={props.plotWidth}
                y1={0}
                y2={0}
                style={{ transform: `translateY(${String(Math.round(y.y(tick)) + 0.5)}px)` }}
                data-baseline={Math.abs(y.y(tick) - bottom) < 0.5 ? '' : undefined}
              />
            ))}
          </svg>
          {props.empty !== undefined && props.empty !== null ? (
            <div className={styles.empty}>{props.empty}</div>
          ) : (
            props.marks({ active, entrance })
          )}
          {anchor !== null && scrub?.crosshair === true ? (
            <svg className={styles.layer} width={props.plotWidth} height={props.height} aria-hidden="true">
              <line
                className={styles.crosshair}
                x1={Math.round(anchor.x) + 0.5}
                x2={Math.round(anchor.x) + 0.5}
                y1={0}
                y2={props.height}
              />
            </svg>
          ) : null}
          {scrub !== null && (props.empty === undefined || props.empty === null) ? (
            <div
              className={styles.scrub}
              role="slider"
              tabIndex={0}
              aria-label={props.label}
              aria-orientation="horizontal"
              aria-valuemin={0}
              aria-valuemax={Math.max(0, count - 1)}
              aria-valuenow={readIndex ?? 0}
              aria-valuetext={reading}
              onPointerMove={onPointerMove}
              onPointerLeave={() => {
                if (active !== null) setActive(null);
              }}
              onClick={() => {
                if (active !== null) scrub.onActivate?.(active);
              }}
              onKeyDown={onKeyDown}
              onBlur={() => {
                if (active !== null) setActive(null);
              }}
            />
          ) : null}
          {anchor !== null && scrub !== null && active !== null ? (
            <div
              ref={tipRef}
              className={styles.tooltipAnchor}
              style={{
                transform: `translate(${String(anchor.x)}px, ${String(Math.max(0, anchor.y ?? 0))}px) translate(${tooltipOffset}, ${anchor.y === null ? '4px' : 'calc(-100% - 10px)'})`,
                translate:
                  tipShift.x !== 0 || tipShift.y !== 0 ? `${String(tipShift.x)}px ${String(tipShift.y)}px` : undefined,
              }}
            >
              {scrub.tooltip(active)}
            </div>
          ) : null}
        </div>
        {props.xLabels !== undefined ? (
          <div className={styles.xaxis} aria-hidden="true">
            {props.xLabels.map((label) => (
              <span
                key={label.key}
                className={styles.xlabel}
                style={{ left: label.x }}
                data-edge={label.x < 16 ? 'start' : label.x > props.plotWidth - 16 ? 'end' : undefined}
              >
                {label.text}
              </span>
            ))}
          </div>
        ) : null}
        {props.xTitle !== undefined ? (
          <span className={styles.xtitle} aria-hidden="true">
            {props.xTitle}
          </span>
        ) : null}
      </div>
      <ChartTable caption={props.label} table={props.table} />
    </Root>
  );
}
