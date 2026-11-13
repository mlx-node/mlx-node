import { Fragment, useRef, type CSSProperties, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { formatCount, formatPercent } from '../../../lib/format';
import { LOW_SAMPLE_N } from '../../../lib/types';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { useEntrance } from '../motion/use-entrance';
import { SwapText } from '../swap-text/swap-text';
import styles from './draft-sweep.module.css';

export interface DraftSweepProps {
  /** "Share of drafts kept at each position" */
  label: string;
  /** Acceptance rate 0–1 per draft position 1…n (cycle-weighted over the models shown). null = no cycles at that position. */
  rates: readonly (number | null)[];
  /** Cycles behind each rate, for the low-sample mark. */
  samples?: readonly number[];
  /** Default var(--series-1). */
  color?: string;
  className?: string;
}

const MAX_CELLS = 8;

function clampRate(rate: number): number {
  return Math.min(1, Math.max(0, rate));
}

/** "position 1 92%, position 2 71%, position 3 no data" */
export function draftSweepText(rates: readonly (number | null)[], samples?: readonly number[]): string {
  return rates
    .map((rate, index) => {
      const low = rate !== null && samples?.[index] !== undefined && samples[index] < LOW_SAMPLE_N;
      return `position ${String(index + 1)} ${rate === null ? 'no data' : formatPercent(rate)}${low ? ' (low sample)' : ''}`;
    })
    .join(', ');
}

/** The verify sweep of speculative decoding: one cell per draft position, filled to its acceptance rate. */
export function DraftSweep(props: DraftSweepProps): ReactElement {
  const ref = useRef<HTMLDivElement | null>(null);
  const entrance = useEntrance(ref);
  const shown = props.rates.slice(0, MAX_CELLS);
  const more = props.rates.length - shown.length;
  const lastIndex = shown.length - 1;
  const description = `${props.label}: ${props.rates.length === 0 ? 'no draft positions' : draftSweepText(props.rates, props.samples)}`;

  return (
    <div
      ref={ref}
      role="img"
      aria-label={description}
      className={cn(styles.root, props.className)}
      data-entrance={entrance}
      style={{ '--c': props.color ?? 'var(--series-1)', '--n': Math.max(1, shown.length) } as CSSProperties}
    >
      <div className={styles.cells}>
        {shown.map((rate, index) => {
          const cycles = props.samples?.[index];
          const low = rate !== null && cycles !== undefined && cycles < LOW_SAMPLE_N;
          const last = index === lastIndex;
          const tip = [
            `Position ${String(index + 1)}`,
            rate === null ? 'no cycles' : `${formatPercent(rate)} kept`,
            ...(cycles !== undefined ? [`${formatCount(cycles)} ${cycles === 1 ? 'cycle' : 'cycles'}`] : []),
          ].join(' · ');
          const style = {
            '--i': index,
            '--rest': rate === null ? '100%' : `${String((1 - clampRate(rate)) * 100)}%`,
          } as CSSProperties;
          return (
            <Fragment key={index}>
              <Tooltip content={tip}>
                <span
                  className={styles.cell}
                  style={style}
                  data-empty={rate === null ? '' : undefined}
                  data-low={low ? '' : undefined}
                >
                  {rate !== null ? (
                    <>
                      <span className={styles.fill} data-pattern={low ? 'hatch' : undefined} />
                      <span className={styles.cap} data-entrance-last={last ? '' : undefined} />
                    </>
                  ) : null}
                </span>
              </Tooltip>
              <span
                className={styles.caption}
                style={{ '--i': index } as CSSProperties}
                data-entrance-last={last && rate === null ? '' : undefined}
              >
                <SwapText text={rate === null ? '—' : formatPercent(rate)} />
              </span>
            </Fragment>
          );
        })}
        {more > 0 ? <span className={styles.more}>+{more}</span> : null}
      </div>
    </div>
  );
}
