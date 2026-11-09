/** `--max-memory SIZE`: Splash `parse_max_memory`, character by character. */

export const MAX_MEMORY_ERROR = "--max-memory must be 'auto' or a positive byte count such as 32G or 2.5G";

/** Longest first, as Splash `_SIZE_UNITS` is searched. */
const SIZE_UNITS: readonly (readonly [string, bigint])[] = [
  ['KIB', 1024n],
  ['MIB', 1024n ** 2n],
  ['GIB', 1024n ** 3n],
  ['KB', 1024n],
  ['MB', 1024n ** 2n],
  ['GB', 1024n ** 3n],
  ['K', 1024n],
  ['M', 1024n ** 2n],
  ['G', 1024n ** 3n],
];

function isDigits(text: string): boolean {
  if (text.length === 0) return false;
  for (let index = 0; index < text.length; index++) {
    const code = text.charCodeAt(index);
    if (code < 48 || code > 57) return false;
  }
  return true;
}

/**
 * `'auto'` → `undefined`; else a byte count with an optional K, M or G
 * suffix (also KB/KiB, …; powers of 1024). A count with a suffix may carry a
 * fraction (`2.5G`), rounded down to whole bytes. The value crosses napi as
 * a JS number, so it must be a safe integer. Throws on anything else.
 */
export function parseMaxMemory(value: string): number | undefined {
  let normalized = value.trim().toUpperCase();
  if (normalized === 'AUTO') return undefined;
  let multiplier = 1n;
  for (const [suffix, unit] of SIZE_UNITS) {
    if (normalized.endsWith(suffix)) {
      normalized = normalized.slice(0, normalized.length - suffix.length);
      multiplier = unit;
      break;
    }
  }
  const dot = normalized.indexOf('.');
  const whole = dot < 0 ? normalized : normalized.slice(0, dot);
  const fraction = dot < 0 ? '' : normalized.slice(dot + 1);
  let size = 0n;
  if (isDigits(whole) && (dot < 0 || (multiplier > 1n && isDigits(fraction)))) {
    size = BigInt(whole) * multiplier;
    if (fraction.length > 0) size += (BigInt(fraction) * multiplier) / 10n ** BigInt(fraction.length);
  }
  if (size < 1n || size > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(MAX_MEMORY_ERROR);
  return Number(size);
}
