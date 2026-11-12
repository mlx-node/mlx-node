/** `--idle-release DURATION`: Splash `parse_idle_release`, character by character. */

/** Splash `metal::kResidencyKeepAliveSeconds`, the engine's default idle release. */
export const DEFAULT_IDLE_RELEASE_SECONDS = 600;

export const IDLE_RELEASE_ERROR = '--idle-release must be off or a positive duration such as 30m, 2h or 600';

/** Splash `_DURATION_UNITS`. */
const DURATION_UNITS: ReadonlyMap<string, number> = new Map([
  ['s', 1],
  ['m', 60],
  ['h', 3600],
]);

function isDigit(char: string): boolean {
  return char >= '0' && char <= '9';
}

/**
 * Digits with `_` only between two digits (Python `float()`); the digits
 * without the separators, or `undefined`.
 */
function digitRun(text: string, start: number): { digits: string; end: number } | undefined {
  let digits = '';
  let index = start;
  while (index < text.length) {
    const char = text[index]!;
    if (isDigit(char)) {
      digits += char;
      index += 1;
      continue;
    }
    if (char === '_' && digits.length > 0 && index + 1 < text.length && isDigit(text[index + 1]!)) {
      index += 1;
      continue;
    }
    break;
  }
  return digits.length > 0 ? { digits, end: index } : undefined;
}

/**
 * Python `float()` of a decimal literal, surrounding whitespace allowed:
 * `[+-] (digits ['.' [digits]] | '.' digits) [(e|E) [+-] digits]`. Python's
 * `inf` / `nan` spellings parse there too but are never finite, so Splash
 * refuses them; they are refused here before any number is made.
 */
export function parsePythonFloat(raw: string): number | undefined {
  const text = raw.trim();
  let index = 0;
  let sign = '';
  if (text[index] === '+' || text[index] === '-') {
    sign = text[index]!;
    index += 1;
  }
  let whole = '';
  const wholeRun = digitRun(text, index);
  if (wholeRun !== undefined) {
    whole = wholeRun.digits;
    index = wholeRun.end;
  }
  let fraction = '';
  if (text[index] === '.') {
    index += 1;
    const fractionRun = digitRun(text, index);
    if (fractionRun !== undefined) {
      fraction = fractionRun.digits;
      index = fractionRun.end;
    }
  }
  if (whole.length === 0 && fraction.length === 0) return undefined;
  let exponent = '';
  if (text[index] === 'e' || text[index] === 'E') {
    index += 1;
    let exponentSign = '';
    if (text[index] === '+' || text[index] === '-') {
      exponentSign = text[index]!;
      index += 1;
    }
    const exponentRun = digitRun(text, index);
    if (exponentRun === undefined) return undefined;
    exponent = `e${exponentSign}${exponentRun.digits}`;
    index = exponentRun.end;
  }
  if (index !== text.length) return undefined;
  return Number(`${sign}${whole.length > 0 ? whole : '0'}.${fraction.length > 0 ? fraction : '0'}${exponent}`);
}

/**
 * `off` → `Infinity`; else seconds, written bare or with an s, m or h suffix
 * (Splash `_duration_seconds`), finite and positive. Throws
 * {@link IDLE_RELEASE_ERROR} on anything else.
 */
export function parseIdleRelease(value: string): number {
  const text = value.trim().toLowerCase();
  if (text === 'off') return Infinity;
  const unit = DURATION_UNITS.get(text.slice(-1));
  const number = parsePythonFloat(unit === undefined ? text : text.slice(0, -1));
  const seconds = number === undefined ? Number.NaN : number * (unit ?? 1);
  if (!(Number.isFinite(seconds) && seconds > 0)) throw new Error(IDLE_RELEASE_ERROR);
  return seconds;
}

/** Splash `idle_release_text`: the value as a command line spells it. */
export function idleReleaseText(seconds: number): string {
  return seconds === Infinity ? 'off' : String(seconds);
}
