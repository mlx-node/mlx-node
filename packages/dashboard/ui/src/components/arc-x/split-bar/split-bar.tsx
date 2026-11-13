import { useRef, useState, type CSSProperties, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { formatPercent, percentInt } from '../../../lib/format';
import { defaultChartFormat, withUnit } from '../chart/value-format';
import { useEntrance } from '../motion/use-entrance';
import { NumberValue } from '../number-value/number-value';
import { SwapText } from '../swap-text/swap-text';
import styles from './split-bar.module.css';

export interface BarSegment {
  key: string;
  label: string;
  value: number;
  color: string;
  pattern?: 'solid' | 'hatch';
}

export interface SplitBarProps {
  label: string;
  segments: readonly BarSegment[];
  /** Capacity. When set, the rest is an empty track. null / 0 = unknown capacity: no track is drawn and the caption says so. */
  total?: number | null;
  unit?: string;
  formatValue?: (value: number) => string;
  /** 4 px | 8 px | 28 px. Default 'thin'. */
  size?: 'dense' | 'thin' | 'tall';
  /** Default true for ≥ 2 segments. */
  legend?: boolean;
  /** Dashed marker on the bar, e.g. an eviction watermark. */
  threshold?: { value: number; label: string };
  /** "no quota" */
  unknownLabel?: string;
  /** `tall` only, default true: a one-line readout above the bar (total, or the hovered / focused segment). */
  readout?: boolean;
}

export interface MeterProps {
  label: string;
  value: number | null;
  max: number | null;
  unit?: string;
  formatValue?: (value: number) => string;
  /** Share of max at which the fill turns warning / danger. Defaults 0.8 / 0.95. */
  warnAt?: number;
  dangerAt?: number;
  /** Default 'thin'. */
  size?: 'dense' | 'thin';
  unknownLabel?: string;
  /** The measured quantity is still moving (a prefill in progress): the fill carries the `shine` loop. */
  busy?: boolean;
  className?: string;
}

/** Parts of a whole on one bar: widths by value, 2 px gaps, a legend that dims the others; boundaries move with `fill`. */
export function SplitBar(props: SplitBarProps): ReactElement {
  const ref = useRef<HTMLDivElement | null>(null);
  const entrance = useEntrance(ref);
  const [active, setActive] = useState<string | null>(null);
  const size = props.size ?? 'thin';
  const format = (value: number): string => {
    return withUnit((props.formatValue ?? defaultChartFormat)(value), props.unit);
  };
  const parts = props.segments.filter((segment) => Number.isFinite(segment.value) && segment.value > 0);
  const sum = parts.reduce((total, segment) => total + segment.value, 0);
  const capacity = props.total === null || props.total === undefined || props.total <= 0 ? null : props.total;
  const unknown = props.total === null || props.total === 0;
  const denominator = capacity ?? sum;
  const rest = capacity === null ? 0 : Math.max(0, capacity - sum);
  const shareOf = (value: number): number | null => (denominator > 0 ? value / denominator : null);
  const showLegend = props.legend ?? props.segments.length >= 2;
  const readout = size === 'tall' && props.readout !== false;
  const activeSegment = parts.find((segment) => segment.key === active) ?? null;

  const description = props.segments
    .map((segment) => `${segment.label} ${format(segment.value)}, ${formatPercent(shareOf(segment.value))}`)
    .concat(capacity !== null && rest > 0 ? [`free ${format(rest)}`] : [])
    .concat(unknown ? [props.unknownLabel ?? 'capacity unknown'] : [])
    .join('; ');

  const threshold =
    props.threshold !== undefined && denominator > 0 && props.threshold.value > 0 && props.threshold.value < denominator
      ? { left: (props.threshold.value / denominator) * 100, label: props.threshold.label }
      : null;
  const widest = props.segments.reduce((max, segment) => Math.max(max, segment.label.length), 'Total'.length);

  return (
    <div
      ref={ref}
      className={styles.root}
      data-size={size}
      data-entrance={entrance}
      data-active={active !== null ? '' : undefined}
    >
      {readout ? (
        <div className={styles.readout} aria-hidden="true">
          {activeSegment !== null ? (
            <span
              className={styles.readoutKey}
              style={{ '--c': activeSegment.color } as CSSProperties}
              data-pattern={activeSegment.pattern}
            />
          ) : null}
          <span className={styles.readoutName} style={{ minWidth: `${String(widest)}ch` }}>
            <SwapText text={activeSegment?.label ?? 'Total'} />
          </span>
          {activeSegment !== null ? (
            <span className={styles.readoutShare}>
              <NumberValue text={formatPercent(shareOf(activeSegment.value))} countUp={false} />
            </span>
          ) : null}
          <span className={styles.readoutValue}>
            <SwapText text={format(activeSegment?.value ?? sum)} />
          </span>
        </div>
      ) : null}
      <div className={styles.barWrap}>
        <div className={styles.bar} role="img" aria-label={`${props.label}: ${description}`}>
          {parts.map((segment, index) => (
            <span
              key={segment.key}
              className={styles.segment}
              style={{ flexGrow: segment.value, '--c': segment.color } as CSSProperties}
              data-pattern={segment.pattern}
              data-on={active === segment.key ? '' : undefined}
              data-entrance-last={index === parts.length - 1 && !showLegend ? '' : undefined}
              onPointerEnter={() => setActive(segment.key)}
              onPointerLeave={() => setActive(null)}
            />
          ))}
          {capacity !== null && rest > 0 ? <span className={styles.rest} style={{ flexGrow: rest }} /> : null}
          {parts.length === 0 && (capacity !== null || unknown) ? (
            <span className={styles.rest} style={{ flexGrow: 1 }} />
          ) : null}
        </div>
        {threshold !== null ? (
          <span className={styles.threshold} style={{ left: `${String(threshold.left)}%` }} title={threshold.label} />
        ) : null}
      </div>
      {unknown ? <span className={styles.caption}>{props.unknownLabel ?? 'Capacity unknown'}</span> : null}
      {threshold !== null ? (
        <span className={styles.caption}>
          <span className={styles.thresholdKey} aria-hidden="true" /> {threshold.label}
        </span>
      ) : null}
      {showLegend ? (
        <ul className={styles.legend} aria-label={`${props.label} legend`}>
          {props.segments.map((segment, index) => (
            <li
              key={segment.key}
              className={styles.legendRow}
              tabIndex={0}
              style={{ '--c': segment.color, '--step': Math.min(index, 7) } as CSSProperties}
              data-on={active === segment.key ? '' : undefined}
              data-entrance-last={index === props.segments.length - 1 ? '' : undefined}
              onPointerEnter={() => setActive(segment.key)}
              onPointerLeave={() => setActive(null)}
              onFocus={() => setActive(segment.key)}
              onBlur={() => setActive(null)}
            >
              <span className={styles.key} data-pattern={segment.pattern} aria-hidden="true" />
              <span className={styles.legendName}>{segment.label}</span>
              <span className={styles.legendValue}>
                <SwapText text={format(segment.value)} />
              </span>
              <span className={styles.legendShare}>
                <SwapText text={formatPercent(shareOf(segment.value))} />
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function toneOf(ratio: number, warnAt: number, dangerAt: number): 'accent' | 'warning' | 'danger' {
  if (ratio >= dangerAt) return 'danger';
  if (ratio >= warnAt) return 'warning';
  return 'accent';
}

/** One ratio of a capacity: the bar, aria-valuenow and the text agree (a 99.9 % fill never reads 100). */
export function Meter(props: MeterProps): ReactElement {
  const ref = useRef<HTMLDivElement | null>(null);
  const entrance = useEntrance(ref);
  const known =
    props.value !== null &&
    Number.isFinite(props.value) &&
    props.max !== null &&
    Number.isFinite(props.max) &&
    props.max > 0;
  const ratio = known ? (props.value ?? 0) / (props.max ?? 1) : null;
  const pct = ratio === null ? 0 : percentInt(ratio);
  const tone = ratio === null ? 'accent' : toneOf(ratio, props.warnAt ?? 0.8, props.dangerAt ?? 0.95);
  const valueText =
    ratio === null
      ? (props.unknownLabel ?? 'unknown')
      : props.formatValue !== undefined && props.value !== null
        ? `${formatPercent(ratio)}, ${withUnit(props.formatValue(props.value), props.unit)}`
        : formatPercent(ratio);
  return (
    <div ref={ref} className={cn(styles.meterRoot, props.className)} data-entrance={entrance}>
      <div
        className={styles.meter}
        role="meter"
        aria-label={props.label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        aria-valuetext={valueText}
        data-size={props.size ?? 'thin'}
        data-tone={tone}
        data-unknown={ratio === null ? '' : undefined}
        style={{ '--rest': `${String(100 - pct)}%` } as CSSProperties}
      >
        {ratio !== null ? (
          <span className={styles.fill} data-busy={props.busy === true ? '' : undefined} data-entrance-last="" />
        ) : null}
      </div>
      {ratio === null ? <span className={styles.caption}>{props.unknownLabel ?? 'Unknown'}</span> : null}
    </div>
  );
}
