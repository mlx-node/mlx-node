import { Maximize2 } from 'lucide-react';
import { LayoutGroup, motion } from 'motion/react';
import { type CSSProperties, useCallback, useContext, useId, useRef, type ReactElement, type ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { cn } from '../../../lib/cn';
import { formatDelta } from '../../../lib/format';
import { LOW_SAMPLE_N } from '../../../lib/types';
import { IconButton } from '../icon-button/icon-button';
import { DrilldownContext } from '../kpi-drilldown/drilldown-context';
import { MiniSpark } from '../mini-spark/mini-spark';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import { NumberValue } from '../number-value/number-value';
import { Placeholder } from '../placeholder/placeholder';
import { SwapText } from '../swap-text/swap-text';
import styles from './stat-tile.module.css';

export interface StatDelta {
  current: number | null;
  previous: number | null;
  /** Which direction is good. Decides the tone; 'neutral' = grey. */
  goodWhen: 'up' | 'down' | 'neutral';
  /** Names the comparison: "vs previous 7 days". */
  versus: string;
}

export interface StatTileProps {
  /** Sentence case, no colon. */
  label: string;
  /** Already formatted; null -> "—". */
  value: string | null;
  /** "tok/s", "ms", "GB" */
  unit?: string;
  /** Hidden when formatDelta() is null. */
  delta?: StatDelta;
  spark?: { data: readonly (number | null)[]; label: string; color?: string };
  /** One line under the value. */
  hint?: ReactNode;
  /** e.g. <Meter/> */
  footer?: ReactNode;
  /** Shows an "n=3" chip when < LOW_SAMPLE_N. */
  sampleCount?: number;
  /** The whole tile is a link. */
  to?: string;
  /** The whole tile is a button (KPI tab). */
  onSelect?: () => void;
  /** 2 px accent underline + aria-pressed. */
  selected?: boolean;
  /** Inside a KpiDrilldown: the key this tile opens. */
  detailKey?: string;
  /** false = the value appears at once, without the count-up from zero. Default true. */
  countUp?: boolean;
  /** The value answers the previous question while a new one loads: data-held, dimmed after the hold. */
  held?: boolean;
  loading?: boolean;
  /** Short text instead of the value, in the danger tone. */
  error?: string;
  className?: string;
}

const GLYPH: Record<'up' | 'down' | 'flat', string> = { up: '↗', down: '↘', flat: '→' };

/** The delta line of a tile: glyph + signed text in the tone of `goodWhen`, then the comparison. */
export function StatDeltaLine({ delta }: { delta: StatDelta }): ReactElement | null {
  const formatted = formatDelta(delta.current, delta.previous);
  if (formatted === null) return null;
  const tone =
    formatted.direction === 'flat' || delta.goodWhen === 'neutral'
      ? 'neutral'
      : formatted.direction === delta.goodWhen
        ? 'good'
        : 'bad';
  return (
    <span className={styles.deltaLine}>
      <span className={styles.delta} data-tone={tone}>
        <span className={styles.glyph} aria-hidden="true">
          {GLYPH[formatted.direction]}
        </span>
        <SwapText text={formatted.text} />
      </span>
      <span className={styles.versus}>{delta.versus}</span>
    </span>
  );
}

/** The "n=3" chip of a low-sample value. */
export function SampleChip({ sampleCount }: { sampleCount: number | undefined }): ReactElement | null {
  if (sampleCount === undefined || sampleCount >= LOW_SAMPLE_N) return null;
  return <span className={styles.chip}>n={sampleCount}</span>;
}

/** The value and its unit in the node `statValue` reads. */
export function StatReading(props: {
  value: string | null;
  unit?: string;
  countUp?: boolean;
  error?: string;
}): ReactElement {
  if (props.error !== undefined) {
    return (
      <span className={styles.reading} data-stat-value="">
        <span className={styles.error}>{props.error}</span>
      </span>
    );
  }
  const measured = props.value !== null && props.value !== '—';
  return (
    <span className={styles.reading} data-stat-value="">
      <NumberValue text={props.value} unit={props.unit} countUp={props.countUp} className={styles.value} />
      {measured && props.unit !== undefined && props.unit.length > 0 ? (
        <>
          {' '}
          <span className={styles.unit}>{props.unit}</span>
        </>
      ) : null}
    </span>
  );
}

/** One measure: label, value (counts up once, rolls on change), delta, hint. A link, a KPI tab or a drill-down trigger. */
export function StatTile(props: StatTileProps): ReactElement {
  const drill = useContext(DrilldownContext);
  const staticMotion = useStaticMotion();
  const rootRef = useRef<HTMLElement | null>(null);
  const loading = props.loading === true;
  const detailKey = drill !== null ? props.detailKey : undefined;
  const opens = detailKey !== undefined;
  const isOpen = opens && drill?.open === detailKey;
  const selectable = props.onSelect !== undefined;
  const wholeTrigger = opens && !selectable && props.to === undefined;
  const detailLabel = `Open ${props.label} details`;

  const registerTrigger = useCallback(
    (node: HTMLElement | null) => {
      if (detailKey !== undefined) drill?.registerTrigger(detailKey, node);
    },
    [detailKey, drill],
  );
  const openDetail = (): void => {
    if (detailKey !== undefined && rootRef.current !== null) drill?.toggle(detailKey, rootRef.current);
  };

  return (
    <section
      ref={rootRef}
      role="group"
      aria-label={props.label}
      className={cn(styles.root, props.className)}
      data-selectable={selectable || wholeTrigger ? '' : undefined}
      data-selected={props.selected === true ? '' : undefined}
      data-held={props.held === true ? '' : undefined}
      data-detail-open={isOpen ? '' : undefined}
      data-link={props.to !== undefined ? '' : undefined}
    >
      {selectable ? (
        <button
          type="button"
          className={styles.stretch}
          aria-pressed={props.selected === true}
          aria-label={props.label}
          onClick={props.onSelect}
        />
      ) : null}
      {wholeTrigger ? (
        <button
          ref={registerTrigger}
          type="button"
          className={styles.stretch}
          aria-expanded={isOpen}
          aria-controls={drill?.panelId}
          aria-label={detailLabel}
          onClick={openDetail}
        />
      ) : null}
      <div className={styles.content}>
        <div className={styles.head}>
          <span className={styles.label}>
            {props.to !== undefined ? (
              <Link className={styles.link} to={props.to}>
                {props.label}
              </Link>
            ) : (
              props.label
            )}
          </span>
          <SampleChip sampleCount={props.sampleCount} />
          {wholeTrigger ? <Maximize2 className={styles.mark} size={14} aria-hidden="true" /> : null}
        </div>
        <div className={styles.valueRow}>
          <Placeholder loading={loading} width="7ch" step={0} className={styles.valueSlot}>
            <StatReading value={props.value} unit={props.unit} countUp={props.countUp} error={props.error} />
          </Placeholder>
          {props.spark !== undefined && !loading ? (
            <MiniSpark data={props.spark.data} label={props.spark.label} color={props.spark.color} />
          ) : null}
        </div>
        {props.delta !== undefined ? (
          <Placeholder loading={loading} width="12ch" step={1}>
            <StatDeltaLine delta={props.delta} />
          </Placeholder>
        ) : null}
        {props.hint !== undefined ? (
          <Placeholder loading={loading} width="60%" step={2}>
            <span className={styles.hint} title={typeof props.hint === 'string' ? props.hint : undefined}>
              {props.hint}
            </span>
          </Placeholder>
        ) : null}
        {props.footer !== undefined ? <div className={styles.footer}>{props.footer}</div> : null}
      </div>
      {opens && selectable ? (
        <IconButton
          ref={registerTrigger}
          className={styles.expand}
          size="sm"
          variant="ghost"
          label={detailLabel}
          icon={<Maximize2 size={14} aria-hidden="true" />}
          aria-expanded={isOpen}
          aria-controls={drill?.panelId}
          onClick={openDetail}
        />
      ) : null}
      {props.selected === true ? (
        staticMotion ? (
          <span className={styles.underline} aria-hidden="true" />
        ) : (
          <motion.span
            layoutId="stat-underline"
            transition={motionX.spring.glide}
            className={styles.underline}
            aria-hidden="true"
          />
        )
      ) : null}
    </section>
  );
}

/**
 * One bordered group of tiles with hairline dividers; wraps. A selectable group shares one underline.
 * `maxColumns` keeps rows even: all tiles in one row while they fit, else two per row, else one.
 */
export function StatGroup(props: { label: string; maxColumns?: number; children: ReactNode }): ReactElement {
  const id = useId();
  const style =
    props.maxColumns !== undefined ? ({ '--stat-max-columns': props.maxColumns } as CSSProperties) : undefined;
  return (
    <div
      role="group"
      aria-label={props.label}
      className={styles.group}
      style={style}
      data-max-columns={props.maxColumns !== undefined ? '' : undefined}
    >
      <LayoutGroup id={id}>{props.children}</LayoutGroup>
    </div>
  );
}
