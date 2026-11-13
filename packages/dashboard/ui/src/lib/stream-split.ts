/**
 * Where a streaming answer may be cut into a settled head and an open tail, so the head is parsed once and only the
 * tail is parsed on every paint. A cut sits at a blank line outside a code fence, after a line that does not belong
 * to a list, quote, table or indented block, and before a line that starts one of none of those at column 0. Cutting
 * there never changes what either side parses to.
 */

function leading(text: string): number {
  let i = 0;
  while (i < text.length && (text[i] === ' ' || text[i] === '\t')) i += 1;
  return i;
}

function isBlankLine(text: string): boolean {
  return leading(text) === text.length;
}

/** The fence a line opens or closes: its character and run length, else null. */
function fenceRun(text: string): { ch: string; len: number } | null {
  const lead = leading(text);
  if (lead > 3) return null;
  const ch = text[lead];
  if (ch !== '`' && ch !== '~') return null;
  let len = 0;
  while (text[lead + len] === ch) len += 1;
  return len >= 3 ? { ch, len } : null;
}

const CONTAINER_START = '-*+>|';

function isDigit(ch: string | undefined): boolean {
  return ch !== undefined && ch >= '0' && ch <= '9';
}

/** The line belongs to a list, quote, table or indented block. */
function opensContainer(text: string): boolean {
  if (text.startsWith('    ') || text.startsWith('\t')) return true;
  const t = text.slice(leading(text));
  const first = t[0];
  if (first === undefined) return false;
  if (first === '>' || first === '|') return true;
  if ((first === '-' || first === '*' || first === '+') && (t.length === 1 || t[1] === ' ' || t[1] === '\t'))
    return true;
  let i = 0;
  while (isDigit(t[i])) i += 1;
  return i > 0 && (t[i] === '.' || t[i] === ')');
}

/** May a block that starts with this line follow a cut? */
function startsClean(text: string): boolean {
  const first = text[0];
  if (first === undefined || first === ' ' || first === '\t') return false;
  return !CONTAINER_START.includes(first) && !isDigit(first);
}

/**
 * The offset of the last allowed cut (the start of the line after it), or 0. `from` is an earlier cut of the same
 * text: the scan starts there, as a scan from 0 would find the same state at a cut.
 */
export function settledSplit(text: string, from = 0): number {
  let best = from;
  let fence: { ch: string; len: number } | null = null;
  let lastContent: string | null = null;
  let sawBlank = false;
  let start = from;
  while (start <= text.length) {
    let end = text.indexOf('\n', start);
    if (end === -1) end = text.length;
    const line = text.slice(start, end);
    if (fence !== null) {
      const run = fenceRun(line);
      if (run !== null && run.ch === fence.ch && run.len >= fence.len && line.trim().length === run.len) fence = null;
      lastContent = '`';
      sawBlank = false;
    } else if (isBlankLine(line)) {
      sawBlank = lastContent !== null;
    } else {
      if (sawBlank && lastContent !== null && !opensContainer(lastContent) && startsClean(line)) best = start;
      const run = fenceRun(line);
      if (run !== null) fence = run;
      lastContent = run !== null ? '`' : line;
      sawBlank = false;
    }
    if (end === text.length) break;
    start = end + 1;
  }
  return best;
}
