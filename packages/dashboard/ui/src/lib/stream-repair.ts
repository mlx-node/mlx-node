/**
 * The open tail of a streaming answer made stable for one paint: a line that is only a marker waits for its first
 * word, an unclosed emphasis or code mark is closed, a link whose address is still arriving shows its words only, and
 * a table row waits for its line end. The result is never stored; the finished answer is rendered as written.
 */

function leading(text: string): number {
  let i = 0;
  while (i < text.length && (text[i] === ' ' || text[i] === '\t')) i += 1;
  return i;
}

function isDigit(ch: string | undefined): boolean {
  return ch !== undefined && ch >= '0' && ch <= '9';
}

function isWordChar(ch: string | undefined): boolean {
  if (ch === undefined) return false;
  return ch.toLowerCase() !== ch.toUpperCase() || isDigit(ch);
}

function isSpace(ch: string | undefined): boolean {
  return ch === undefined || ch === ' ' || ch === '\t';
}

function fenceRun(text: string): { ch: string; len: number } | null {
  const lead = leading(text);
  if (lead > 3) return null;
  const ch = text[lead];
  if (ch !== '`' && ch !== '~') return null;
  let len = 0;
  while (text[lead + len] === ch) len += 1;
  return len >= 3 ? { ch, len } : null;
}

/** The fence still open at the end of `text`, if any. */
function openFence(text: string): boolean {
  let fence: { ch: string; len: number } | null = null;
  let start = 0;
  while (start <= text.length) {
    let end = text.indexOf('\n', start);
    if (end === -1) end = text.length;
    const line = text.slice(start, end);
    const run = fenceRun(line);
    if (fence === null) {
      if (run !== null) fence = run;
    } else if (run !== null && run.ch === fence.ch && run.len >= fence.len && line.trim().length === run.len) {
      fence = null;
    }
    if (end === text.length) break;
    start = end + 1;
  }
  return fence !== null;
}

/** A line that is only a list, heading or fence marker (with an optional fence language), or a table row start. */
function heldBack(line: string): boolean {
  const t = line.slice(leading(line));
  if (t === '') return false;
  if (t[0] === '|') return true;
  const word = t.trimEnd();
  if (word === '-' || word === '*' || word === '+') return true;
  let i = 0;
  while (isDigit(word[i])) i += 1;
  if (i > 0 && i === word.length - 1 && (word[i] === '.' || word[i] === ')')) return true;
  let hashes = 0;
  while (word[hashes] === '#') hashes += 1;
  if (hashes > 0 && hashes <= 6 && hashes === word.length) return true;
  return fenceRun(line) !== null;
}

/** `[words](partial` at the end of the line → `words`. */
function openLink(line: string): string {
  const mark = line.lastIndexOf('](');
  if (mark === -1 || line.indexOf(')', mark) !== -1) return line;
  const open = line.lastIndexOf('[', mark);
  if (open === -1) return line;
  const words = line.slice(open + 1, mark);
  const start = open > 0 && line[open - 1] === '!' ? open - 1 : open;
  return line.slice(0, start) + words;
}

interface Mark {
  text: string;
  at: number;
}

/** Closes the emphasis, strike and code marks the last line leaves open. */
function closeMarks(line: string): string {
  const stack: Mark[] = [];
  let i = 0;
  const lead = leading(line);
  const t = line.slice(lead);
  if ((t[0] === '-' || t[0] === '*' || t[0] === '+') && isSpace(t[1])) i = lead + 2;
  let code = -1;
  let closedAt = -1;
  while (i < line.length) {
    const c = line[i]!;
    if (code !== -1) {
      if (c === '`') code = -1;
      i += 1;
      continue;
    }
    if (c === '\\') {
      i += 2;
      continue;
    }
    if (c === '`') {
      let run = 0;
      while (line[i + run] === '`') run += 1;
      if (run === 1) code = i;
      i += run;
      continue;
    }
    let mark: string | null = null;
    if (c === '*' || c === '~' || c === '_') {
      if (line[i + 1] === c && c !== '_') mark = c + c;
      else if (c !== '~') mark = c;
    }
    if (mark === null) {
      i += 1;
      continue;
    }
    const before = line[i - 1];
    const after = line[i + mark.length];
    const top = stack[stack.length - 1];
    if (top !== undefined && top.text === mark && !isSpace(before) && !(c === '_' && isWordChar(after))) {
      stack.pop();
      closedAt = i + mark.length;
    } else if (!isSpace(after) && !(c === '_' && isWordChar(before))) {
      stack.push({ text: mark, at: i });
    }
    i += mark.length;
  }
  let out = line;
  if (code !== -1) {
    if (out.slice(code + 1).trim() === '') return out.slice(0, code).trimEnd();
    out += '`';
  }
  // The first half of a closing `**` or `~~` is arriving: it is not drawn as a stray mark before the closer.
  const top = stack[stack.length - 1];
  if (code === -1 && top !== undefined && top.text.length === 2 && closedAt !== line.length) {
    const ch = top.text[0]!;
    let run = 0;
    while (run < out.length && out[out.length - 1 - run] === ch) run += 1;
    if (run > 0 && run < 2 && out.length - run > top.at + 2) out = out.slice(0, out.length - run);
  }
  // One pass, innermost mark first: a mark with only space after it is dropped, else every mark left is closed.
  let baseLen = out.length;
  let last = lastNonSpace(out, baseLen);
  const closers: string[] = [];
  while (stack.length > 0) {
    const mark = stack.pop()!;
    if (closers.length > 0) {
      closers.push(mark.text);
      continue;
    }
    if (mark.at >= baseLen) continue;
    if (last < mark.at + mark.text.length) {
      baseLen = mark.at;
      last = lastNonSpace(out, baseLen);
    } else {
      baseLen = last + 1;
      closers.push(mark.text);
    }
  }
  return out.slice(0, baseLen) + closers.join('');
}

/** The index of the last character before `end` that `trim` keeps, or -1. */
function lastNonSpace(text: string, end: number): number {
  let i = end - 1;
  while (i >= 0 && text[i]!.trim() === '') i -= 1;
  return i;
}

/** The tail as it is painted while it still arrives. */
export function repairTail(tail: string): string {
  if (tail === '') return tail;
  const cut = tail.lastIndexOf('\n') + 1;
  const head = tail.slice(0, cut);
  const line = tail.slice(cut);
  if (openFence(head)) return tail;
  if (line === '') return tail;
  if (heldBack(line)) return head;
  return head + dropBareMarks(closeMarks(openLink(line)));
}

/** A run of marks at the end of the line with nothing after it yet is not drawn. */
function dropBareMarks(line: string): string {
  const text = line.trimEnd();
  let start = text.length;
  while (start > 0 && '*_~`'.includes(text[start - 1]!)) start -= 1;
  if (start === text.length || !isSpace(text[start - 1])) return line;
  return text.slice(0, start).trimEnd();
}
