/**
 * Presentation-only formatting helpers for dashboard UI. No units are inferred
 * from the API — byte fields are bytes, token/count fields are integers.
 *
 * A value that was not measured (`null`, `undefined`, NaN, ±Infinity) prints as
 * an em dash, never as 0.
 */

import type { MetricsBucket } from './types';

/** What every formatter prints for a value that was not measured. */
export const NO_VALUE = '—';

type Maybe = number | null | undefined;

function measured(value: Maybe): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'] as const;

/** Human-readable base-1024 byte size (`8.6 GB`, `512 KB`, `0 B`). */
export function formatBytes(bytes: Maybe): string {
  if (!measured(bytes)) return NO_VALUE;
  if (bytes <= 0) return '0 B';
  const exp = Math.max(0, Math.min(Math.floor(Math.log2(bytes) / 10), BYTE_UNITS.length - 1));
  const value = bytes / 1024 ** exp;
  const digits = exp === 0 ? 0 : value >= 100 ? 0 : 1;
  return `${value.toFixed(digits)} ${BYTE_UNITS[exp]}`;
}

/**
 * Stat-tile value formatting: grouped up to 10K (`1,284`), compact above
 * (`12.9K`, `4.2M`). Proportional figures — reserve tabular-nums for columns.
 */
export function formatCount(n: Maybe): string {
  if (!measured(n)) return NO_VALUE;
  if (Math.abs(n) < 10_000) return new Intl.NumberFormat('en-US').format(Math.round(n));
  return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(n);
}

/** Grouped integer with thousands separators, for aligned table columns. */
export function formatNumber(n: Maybe): string {
  if (!measured(n)) return NO_VALUE;
  return new Intl.NumberFormat('en-US').format(Math.round(n));
}

/** A count and its noun, the noun agreeing with the count: `1 model`, `3 models`. */
export function formatCountNoun(
  n: Maybe,
  one: string,
  other: string,
  format: (n: Maybe) => string = formatNumber,
): string {
  return `${format(n)} ${n === 1 ? one : other}`;
}

/**
 * Whole-percent string (`73%`); `fraction` is 0..1.
 *
 * SATURATING at both ends: `100%` and `0%` are reachable ONLY from an exact 1
 * and an exact 0. A plain `Math.round(fraction * 100)` renders 0.996982 — a
 * real cache with 16189 hits and 49 misses — as a flat "100%", i.e. a page
 * claiming perfection directly above the line "49 misses". Anything short of
 * the endpoint now reads `>99%` / `<1%` instead of rounding into a lie.
 */
export function formatPercent(fraction: Maybe): string {
  const text = formatPercentValue(fraction);
  return text === NO_VALUE ? text : `${text}%`;
}

/** `formatPercent` without the sign, for a chart whose unit is '%': '91', '>99', '<1'. */
export function formatPercentValue(fraction: Maybe): string {
  if (!measured(fraction)) return NO_VALUE;
  const pct = fraction * 100;
  if (pct >= 100) return '100';
  if (pct <= 0) return '0';
  if (pct >= 99.5) return '>99';
  if (pct < 0.5) return '<1';
  return String(Math.round(pct));
}

/**
 * Percent as an integer for `aria-valuenow` and CSS widths, saturated the same
 * way {@link formatPercent} is so the accessible value can never announce 100
 * while the visible label says `>99%`.
 */
export function percentInt(fraction: number): number {
  if (!Number.isFinite(fraction)) return 0;
  const pct = fraction * 100;
  if (pct >= 100) return 100;
  if (pct <= 0) return 0;
  return Math.min(99, Math.max(1, Math.round(pct)));
}

/**
 * Success rate from the RAW COUNTS rather than a pre-divided fraction, so the
 * endpoint cases are decided by integers instead of by rounding.
 *
 * A fraction alone cannot tell 0.9996 from 1.0 once rounded, and a cache page
 * that prints "100%" while a non-zero miss count sits beside it is lying. Here
 * `100%` requires `total - hits === 0` and `0%` requires `hits === 0`; every
 * other value saturates to `>99%` / `<1%`. `total <= 0` is "no lookups", not
 * zero percent, so it renders as an em dash.
 */
export function formatRate(hits: Maybe, total: Maybe): string {
  if (!measured(hits) || !measured(total) || total <= 0) return NO_VALUE;
  if (hits <= 0) return '0%';
  const misses = total - hits;
  if (misses <= 0) return '100%';
  const pct = (hits / total) * 100;
  if (pct >= 99.5) return '>99%';
  if (pct < 0.5) return '<1%';
  return `${Math.round(pct)}%`;
}

/** Coarse relative time from an ms-epoch timestamp (`3m ago`, `2d ago`). */
export function formatRelativeTime(ms: Maybe): string {
  if (!measured(ms)) return NO_VALUE;
  if (ms <= 0) return 'unknown';
  const diffSec = Math.round((Date.now() - ms) / 1000);
  if (diffSec < 45) return 'just now';
  const min = Math.round(diffSec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.round(hr / 24);
  if (day < 30) return `${day}d ago`;
  return new Date(ms).toLocaleDateString();
}

/**
 * `<1 ms`; under 1 s an integer `412 ms`; under a minute 3 significant digits
 * `1.24 s` / `12.4 s`; under an hour `2m 05s`; else `1h 12m`.
 */
export function formatDuration(ms: Maybe): string {
  const parts = formatDurationParts(ms);
  return parts.unit === '' ? parts.value : `${parts.value} ${parts.unit}`;
}

export function formatDurationParts(ms: Maybe): { value: string; unit: string } {
  if (!measured(ms)) return { value: NO_VALUE, unit: '' };
  if (ms < 1) return { value: '<1', unit: 'ms' };
  if (Math.round(ms) < 1000) return { value: String(Math.round(ms)), unit: 'ms' };
  const seconds = ms / 1000;
  const precise = Number(seconds.toPrecision(3));
  if (precise < 60) return { value: seconds.toPrecision(3), unit: 's' };
  const totalSeconds = Math.round(seconds);
  if (totalSeconds < 3600) {
    const minutes = Math.floor(totalSeconds / 60);
    const rest = totalSeconds % 60;
    return { value: `${minutes}m ${rest < 10 ? '0' : ''}${rest}s`, unit: '' };
  }
  const totalMinutes = Math.round(seconds / 60);
  return { value: `${Math.floor(totalMinutes / 60)}h ${totalMinutes % 60}m`, unit: '' };
}

/** Tokens per second without its unit: one decimal below 100, an integer from 100. */
export function formatTps(value: Maybe): string {
  if (!measured(value)) return NO_VALUE;
  const fixed = value.toFixed(1);
  return Math.abs(Number(fixed)) < 100 ? fixed : formatNumber(value);
}

export interface FormattedDelta {
  text: string;
  direction: 'up' | 'down' | 'flat';
}

/** Change from `prev` to `cur` in whole percent: `+12%`, `−3%` (U+2212); below 0.5 % it is `flat`. Null when `prev` is missing or 0. */
export function formatDelta(cur: Maybe, prev: Maybe): FormattedDelta | null {
  if (!measured(cur) || !measured(prev) || prev === 0) return null;
  const pct = ((cur - prev) / Math.abs(prev)) * 100;
  if (Math.abs(pct) < 0.5) return { text: '0%', direction: 'flat' };
  const whole = Math.round(Math.abs(pct));
  return pct > 0 ? { text: `+${whole}%`, direction: 'up' } : { text: `\u2212${whole}%`, direction: 'down' };
}

const DAY = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });
const DAY_LONG = new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
const CLOCK = new Intl.DateTimeFormat('en-US', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
const CLOCK_SECONDS = new Intl.DateTimeFormat('en-US', {
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hourCycle: 'h23',
});

/** `Nov 3`, local time. */
export function formatDay(ms: Maybe): string {
  return measured(ms) ? DAY.format(ms) : NO_VALUE;
}

/** `Mon, Nov 3`, local time. */
export function formatDayLong(ms: Maybe): string {
  return measured(ms) ? DAY_LONG.format(ms) : NO_VALUE;
}

/** `14:05` (24 h), or `14:05:12` with seconds; local time. */
export function formatClock(ms: Maybe, withSeconds = false): string {
  if (!measured(ms)) return NO_VALUE;
  return (withSeconds ? CLOCK_SECONDS : CLOCK).format(ms);
}

/** `5m` and `hour` buckets read as a clock, `day` buckets as a day. */
export function formatBucketLabel(ms: Maybe, bucket: MetricsBucket): string {
  return bucket === 'day' ? formatDay(ms) : formatClock(ms);
}

/** {@link formatBytes} split into value and unit: `{ '8.6', 'GB' }`, `{ '—', '' }`. */
export function formatBytesParts(bytes: Maybe): { value: string; unit: string } {
  const text = formatBytes(bytes);
  const space = text.lastIndexOf(' ');
  return space === -1 ? { value: text, unit: '' } : { value: text.slice(0, space), unit: text.slice(space + 1) };
}
