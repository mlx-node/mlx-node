/** `--max-context TOKENS`: Splash `parse_max_context`, character by character. */

/** Splash `MAX_YARN_CONTEXT_TOKENS`: the YaRN window of the Qwen targets. */
export const MAX_YARN_CONTEXT_TOKENS = 1_010_000;

export const MAX_CONTEXT_ERROR =
  "--max-context must be 'auto' or a token count up to 1010000, such as 100K or 1M (K = 1024 tokens, M = 1000000 tokens)";

/** Splash `_CONTEXT_UNITS`. */
const CONTEXT_UNITS: ReadonlyMap<string, number> = new Map([
  ['K', 1024],
  ['M', 1_000_000],
]);

function isDigits(text: string): boolean {
  if (text.length === 0) return false;
  for (let index = 0; index < text.length; index++) {
    const code = text.charCodeAt(index);
    if (code < 48 || code > 57) return false;
  }
  return true;
}

/**
 * `'auto'` → `undefined`; else a token count with an optional K (1024) or
 * M (1,000,000) suffix, from 1 to {@link MAX_YARN_CONTEXT_TOKENS}. Throws
 * {@link MAX_CONTEXT_ERROR} on anything else.
 */
export function parseMaxContext(value: string): number | undefined {
  const normalized = value.trim().toUpperCase();
  if (normalized === 'AUTO') return undefined;
  let number = normalized;
  let multiplier = 1;
  const unit = CONTEXT_UNITS.get(normalized.slice(-1));
  if (unit !== undefined) {
    number = normalized.slice(0, -1);
    multiplier = unit;
  }
  const tokens = isDigits(number) ? Number(number) * multiplier : 0;
  if (!(tokens >= 1 && tokens <= MAX_YARN_CONTEXT_TOKENS)) throw new Error(MAX_CONTEXT_ERROR);
  return tokens;
}
