/**
 * Presentation-only formatting helpers for dashboard UI. No units are inferred
 * from the API — byte fields are bytes, token/count fields are integers.
 */

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'] as const;

/** Human-readable base-1024 byte size (`8.6 GB`, `512 KB`, `0 B`). */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const exp = Math.min(Math.floor(Math.log2(bytes) / 10), BYTE_UNITS.length - 1);
  const value = bytes / 1024 ** exp;
  const digits = exp === 0 ? 0 : value >= 100 ? 0 : 1;
  return `${value.toFixed(digits)} ${BYTE_UNITS[exp]}`;
}

/**
 * Stat-tile value formatting: grouped up to 10K (`1,284`), compact above
 * (`12.9K`, `4.2M`). Proportional figures — reserve tabular-nums for columns.
 */
export function formatCount(n: number): string {
  if (!Number.isFinite(n)) return '0';
  if (Math.abs(n) < 10_000) return new Intl.NumberFormat('en-US').format(Math.round(n));
  return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(n);
}

/** Grouped integer with thousands separators, for aligned table columns. */
export function formatNumber(n: number): string {
  if (!Number.isFinite(n)) return '0';
  return new Intl.NumberFormat('en-US').format(Math.round(n));
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
export function formatPercent(fraction: number): string {
  if (!Number.isFinite(fraction)) return '0%';
  const pct = fraction * 100;
  if (pct >= 100) return '100%';
  if (pct <= 0) return '0%';
  if (pct >= 99.5) return '>99%';
  if (pct < 0.5) return '<1%';
  return `${Math.round(pct)}%`;
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
export function formatRate(hits: number, total: number): string {
  if (!Number.isFinite(hits) || !Number.isFinite(total) || total <= 0) return '—';
  if (hits <= 0) return '0%';
  const misses = total - hits;
  if (misses <= 0) return '100%';
  const pct = (hits / total) * 100;
  if (pct >= 99.5) return '>99%';
  if (pct < 0.5) return '<1%';
  return `${Math.round(pct)}%`;
}

/** Absolute local date + time from an ms-epoch timestamp (`Jul 20, 2026, 3:41 PM`). */
export function formatDateTime(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return 'unknown';
  return new Date(ms).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

/** Coarse relative time from an ms-epoch timestamp (`3m ago`, `2d ago`). */
export function formatRelativeTime(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return 'unknown';
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
