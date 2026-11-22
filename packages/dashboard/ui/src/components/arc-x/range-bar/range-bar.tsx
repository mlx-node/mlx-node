import { motion } from 'motion/react';
import type { CSSProperties, ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { formatPercent } from '../../../lib/format';
import { Badge } from '../../arc/badge/badge';
import { Disclosure } from '../disclosure/disclosure';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './range-bar.module.css';

export interface RangeBarProps {
  label: string;
  min?: number | string;
  max?: number | string;
  p10: number | string;
  median: number | string;
  p90: number | string;
  mean?: number | string;
  unit?: string;
  pBelow?: number;
  pAbove?: number;
  quantiles?: readonly (number | string)[];
  color?: string;
  className?: string;
}

function toNumeric(value: number | string | undefined): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string') {
    const num = Number(value);
    if (Number.isFinite(num)) return num;
    const parsedDate = Date.parse(value);
    if (Number.isFinite(parsedDate)) return parsedDate;
  }
  return null;
}

function formatVal(value: number | string): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  return value;
}

function quantileLabel(index: number, total: number): string {
  if (total <= 1) return 'p50';
  const pct = Math.round(((index + 1) / (total + 1)) * 100);
  return `p${String(pct)}`;
}

/** Bounded distributional range bar (`min -> p10 -> median -> p90 -> max`) with optional expandable quantiles table. */
export function RangeBar({
  label,
  min,
  max,
  p10,
  median,
  p90,
  mean,
  unit,
  pBelow,
  pAbove,
  quantiles,
  color = 'var(--model-1)',
  className,
}: RangeBarProps): ReactElement {
  const staticMotion = useStaticMotion();
  const nP10 = toNumeric(p10);
  const nMed = toNumeric(median);
  const nP90 = toNumeric(p90);
  const nMean = toNumeric(mean);
  const nMin = toNumeric(min) ?? nP10;
  const nMax = toNumeric(max) ?? nP90;

  const hasScale = nMin !== null && nMax !== null && nMax > nMin && nP10 !== null && nMed !== null && nP90 !== null;
  const clampPct = (v: number): number =>
    Math.min(100, Math.max(0, ((v - (nMin ?? 0)) / ((nMax ?? 1) - (nMin ?? 0))) * 100));

  const leftPct = hasScale ? clampPct(nP10) : 20;
  const rightPct = hasScale ? clampPct(nP90) : 80;
  const widthPct = Math.max(2, rightPct - leftPct);
  const medPct = hasScale ? clampPct(nMed) : 50;
  const meanPct = hasScale && nMean !== null ? clampPct(nMean) : null;

  const transition = staticMotion ? { duration: 0 } : motionX.spring.reshape;
  const barStyle = { '--c': color } as CSSProperties;

  return (
    <div className={cn(styles.root, className)} style={barStyle} aria-label={`${label} estimate range`}>
      <div className={styles.summary}>
        <span className={styles.medianText}>
          {formatVal(median)}
          {unit !== undefined && unit !== '' ? <span className={styles.unit}>{unit}</span> : null}
        </span>
        <span className={styles.intervalText}>
          {`p10 ${formatVal(p10)} · p90 ${formatVal(p90)}`}
          {mean !== undefined ? ` · mean ${formatVal(mean)}` : ''}
        </span>
      </div>

      <div
        className={styles.trackWrap}
        role="img"
        aria-label={`Median ${formatVal(median)}, p10 to p90 from ${formatVal(p10)} to ${formatVal(p90)}`}
      >
        <div className={styles.track}>
          <motion.div
            className={styles.band}
            initial={staticMotion ? false : { left: `${String(medPct)}%`, width: '0%' }}
            animate={{ left: `${String(leftPct)}%`, width: `${String(widthPct)}%` }}
            transition={transition}
          />
          {meanPct !== null ? (
            <motion.span
              className={styles.meanDot}
              title={`Mean: ${formatVal(mean!)}`}
              initial={false}
              animate={{ left: `${String(meanPct)}%` }}
              transition={transition}
            />
          ) : null}
          <motion.span
            className={styles.medianPin}
            title={`Median: ${formatVal(median)}`}
            initial={false}
            animate={{ left: `${String(medPct)}%` }}
            transition={transition}
          />
        </div>
      </div>

      <div className={styles.bounds}>
        <span>{min !== undefined ? `min ${formatVal(min)}` : `p10 ${formatVal(p10)}`}</span>
        <span>{max !== undefined ? `max ${formatVal(max)}` : `p90 ${formatVal(p90)}`}</span>
      </div>

      {(pBelow !== undefined && pBelow > 0.005) || (pAbove !== undefined && pAbove > 0.005) ? (
        <div className={styles.tails}>
          {pBelow !== undefined && pBelow > 0.005 ? (
            <Badge size="sm" tone="warning" outline>
              {`p_below ${formatPercent(pBelow)}`}
            </Badge>
          ) : null}
          {pAbove !== undefined && pAbove > 0.005 ? (
            <Badge size="sm" tone="warning" outline>
              {`p_above ${formatPercent(pAbove)}`}
            </Badge>
          ) : null}
        </div>
      ) : null}

      {quantiles !== undefined && quantiles.length > 0 ? (
        <Disclosure title={`Quantiles (${String(quantiles.length)})`}>
          <table className={styles.quantTable}>
            <thead>
              <tr>
                <th>Quantile</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {quantiles.map((q, index) => (
                <tr key={index}>
                  <td>{quantileLabel(index, quantiles.length)}</td>
                  <td>
                    {formatVal(q)}
                    {unit !== undefined && unit !== '' ? ` ${unit}` : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Disclosure>
      ) : null}
    </div>
  );
}
