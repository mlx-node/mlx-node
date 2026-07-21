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

/** Whole-percent string (`73%`); `fraction` is 0..1. */
export function formatPercent(fraction: number): string {
  if (!Number.isFinite(fraction)) return '0%';
  return `${Math.round(fraction * 100)}%`;
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
