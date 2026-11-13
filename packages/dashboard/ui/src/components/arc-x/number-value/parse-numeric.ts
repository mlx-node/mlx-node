import { isAsciiDigit, stripCommas } from '../../../lib/text-scan';

export interface ParsedNumber {
  /** '', '−', '+', '$', '~' */
  prefix: string;
  /** 1284, 42.5 (never negative: a sign is part of the prefix). */
  value: number;
  /** Fraction digits written in the text. */
  decimals: number;
  /** '', '%', 'k', 'M', ' ms' */
  suffix: string;
}

/**
 * Splits a formatted number into prefix, one number and suffix; null when the text is not exactly one number that
 * en-US formatting writes back character for character (no digit, a second number in the suffix, odd grouping).
 */
export function parseNumeric(text: string): ParsedNumber | null {
  let start = 0;
  while (start < text.length && !isAsciiDigit(text[start])) start += 1;
  if (start === text.length) return null;

  let end = start;
  while (end < text.length && (isAsciiDigit(text[end]) || (text[end] === ',' && isAsciiDigit(text[end + 1] ?? '')))) {
    end += 1;
  }
  let decimals = 0;
  if (text[end] === '.' && isAsciiDigit(text[end + 1] ?? '')) {
    end += 1;
    while (end < text.length && isAsciiDigit(text[end])) {
      end += 1;
      decimals += 1;
    }
  }

  const prefix = text.slice(0, start);
  const suffix = text.slice(end);
  for (let i = 0; i < suffix.length; i += 1) if (isAsciiDigit(suffix[i])) return null;

  const value = Number(stripCommas(text.slice(start, end)));
  if (!Number.isFinite(value)) return null;
  const written = new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
  if (prefix + written + suffix !== text) return null;
  return { prefix, value, decimals, suffix };
}
