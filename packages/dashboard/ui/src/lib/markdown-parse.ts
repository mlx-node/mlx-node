/**
 * A hand-written Markdown reader for model answers: blocks (paragraphs, headings, fenced and indented code, quotes,
 * lists, rules, tables) and inlines (emphasis, strike, code, links, images, `[n]` marks, breaks). Raw HTML is text.
 * Every text node keeps its source offset so a streaming answer can fade in only the part that just arrived. Nothing
 * here is rendered or followed: the renderer decides what may become a link.
 */

export type MdInline =
  /** `len` is the source length; it differs from `value.length` only for an escape. */
  | { type: 'text'; value: string; at: number; len: number }
  | { type: 'code'; value: string; at: number }
  | { type: 'em' | 'strong' | 'del'; children: MdInline[] }
  | { type: 'link'; href: string; children: MdInline[] }
  | { type: 'image'; href: string; alt: string; at: number }
  /** `[n]` with 1 ≤ n ≤ 99, outside code and links. */
  | { type: 'cite'; n: number; at: number }
  | { type: 'break' }
  | { type: 'soft' };

export type MdAlign = 'left' | 'center' | 'right' | null;

export type MdBlock =
  | { type: 'paragraph'; at: number; children: MdInline[] }
  | { type: 'heading'; at: number; level: number; children: MdInline[] }
  | { type: 'code'; at: number; lang: string; value: string; closed: boolean }
  | { type: 'quote'; at: number; children: MdBlock[] }
  | { type: 'list'; at: number; ordered: boolean; start: number; tight: boolean; items: MdBlock[][] }
  | { type: 'rule'; at: number }
  | { type: 'table'; at: number; align: MdAlign[]; head: MdInline[][]; rows: MdInline[][][] };

interface Line {
  text: string;
  /** Source offset of `text[0]`. */
  at: number;
}

/** Containers nest at most this deep; deeper markers are read as text. */
const MAX_DEPTH = 24;
const MAX_LINK_DEST = 4096;
/** Emphasis, strike and links nest at most this deep; deeper delimiters stay text (rendering recurses per level). */
const MAX_INLINE_DEPTH = 32;
/** A header wider than this is not a table; a table stops before the row that would pass the cell cap. */
const MAX_TABLE_COLUMNS = 64;
const MAX_TABLE_CELLS = 20_000;

function isBlank(text: string): boolean {
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (c !== ' ' && c !== '\t') return false;
  }
  return true;
}

/** Columns of leading white space (a tab advances to the next multiple of 4). */
function indentOf(text: string): number {
  let cols = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (text[i] === ' ') cols += 1;
    else if (text[i] === '\t') cols += 4 - (cols % 4);
    else break;
  }
  return cols;
}

/** The line with `cols` columns of leading white space removed (fewer when the line has fewer). */
function dropCols(line: Line, cols: number): Line {
  let used = 0;
  let i = 0;
  while (i < line.text.length && used < cols) {
    const c = line.text[i];
    if (c === ' ') used += 1;
    else if (c === '\t') used += 4 - (used % 4);
    else break;
    i += 1;
  }
  return { text: line.text.slice(i), at: line.at + i };
}

function dropSpaces(line: Line): Line {
  let i = 0;
  while (i < line.text.length && (line.text[i] === ' ' || line.text[i] === '\t')) i += 1;
  return { text: line.text.slice(i), at: line.at + i };
}

function trimEndSpaces(text: string): string {
  let end = text.length;
  while (end > 0 && (text[end - 1] === ' ' || text[end - 1] === '\t')) end -= 1;
  return text.slice(0, end);
}

function isDigit(ch: string | undefined): boolean {
  return ch !== undefined && ch >= '0' && ch <= '9';
}

function splitLines(text: string, base: number): Line[] {
  const lines: Line[] = [];
  let start = 0;
  for (let i = 0; i <= text.length; i += 1) {
    if (i === text.length || text[i] === '\n') {
      let end = i;
      if (end > start && text[end - 1] === '\r') end -= 1;
      lines.push({ text: text.slice(start, end), at: base + start });
      start = i + 1;
    }
  }
  return lines;
}

// ---- block starts -------------------------------------------------------------------------------------------------

interface Fence {
  ch: string;
  len: number;
  indent: number;
  lang: string;
}

function fenceOpen(text: string): Fence | null {
  const indent = indentOf(text);
  if (indent > 3) return null;
  const t = text.trimStart();
  const ch = t[0];
  if (ch !== '`' && ch !== '~') return null;
  let len = 0;
  while (t[len] === ch) len += 1;
  if (len < 3) return null;
  const info = t.slice(len).trim();
  if (ch === '`' && info.includes('`')) return null;
  let lang = '';
  for (const c of info) {
    if (c === ' ' || c === '\t') break;
    lang += c;
  }
  return { ch, len, indent, lang };
}

function fenceCloses(text: string, fence: Fence): boolean {
  if (indentOf(text) > 3) return false;
  const t = text.trim();
  if (t.length < fence.len) return false;
  for (const c of t) if (c !== fence.ch) return false;
  return true;
}

function atxHeading(text: string): { level: number; content: number } | null {
  if (indentOf(text) > 3) return null;
  let i = 0;
  while (text[i] === ' ') i += 1;
  let level = 0;
  while (text[i + level] === '#') level += 1;
  if (level === 0 || level > 6) return null;
  const after = text[i + level];
  if (after !== undefined && after !== ' ' && after !== '\t') return null;
  return { level, content: i + level };
}

function isRule(text: string): boolean {
  if (indentOf(text) > 3) return false;
  let ch = '';
  let count = 0;
  for (const c of text) {
    if (c === ' ' || c === '\t') continue;
    if (c !== '-' && c !== '*' && c !== '_') return false;
    if (ch === '') ch = c;
    else if (c !== ch) return false;
    count += 1;
  }
  return count >= 3;
}

function setextLevel(text: string): number {
  if (indentOf(text) > 3) return 0;
  const t = text.trim();
  if (t === '') return 0;
  const ch = t[0];
  if (ch !== '=' && ch !== '-') return 0;
  for (const c of t) if (c !== ch) return 0;
  return ch === '=' ? 1 : 2;
}

function quoteStart(text: string): boolean {
  return indentOf(text) <= 3 && text.trimStart().startsWith('>');
}

interface ListMarker {
  ordered: boolean;
  /** `-`, `+`, `*`, `.` or `)`. */
  kind: string;
  start: number;
  /** Columns from the line start to the item content. */
  content: number;
  /** The marker line has no content. */
  empty: boolean;
  /** Index in the line where the content starts. */
  offset: number;
}

function listMarker(text: string): ListMarker | null {
  const indent = indentOf(text);
  if (indent > 3) return null;
  let i = 0;
  while (text[i] === ' ' || text[i] === '\t') i += 1;
  let ordered = false;
  let kind: string;
  let start = 1;
  let width: number;
  const c = text[i];
  if (c === '-' || c === '+' || c === '*') {
    kind = c;
    width = 1;
  } else if (isDigit(c)) {
    let j = i;
    while (isDigit(text[j]) && j - i < 9) j += 1;
    if (text[j] !== '.' && text[j] !== ')') return null;
    ordered = true;
    kind = text[j]!;
    start = Number(text.slice(i, j));
    width = j - i + 1;
  } else {
    return null;
  }
  const after = i + width;
  if (after >= text.length || isBlank(text.slice(after))) {
    return { ordered, kind, start, content: indent + width + 1, empty: true, offset: text.length };
  }
  if (text[after] !== ' ' && text[after] !== '\t') return null;
  let spaces = 0;
  let k = after;
  while (k < text.length && text[k] === ' ' && spaces < 5) {
    spaces += 1;
    k += 1;
  }
  if (text[k] === '\t') spaces += 1;
  let offset = k;
  if (spaces > 4) {
    spaces = 1;
    offset = after + 1;
  } else if (text[k] === '\t') {
    offset = k + 1;
  }
  return { ordered, kind, start, content: indent + width + Math.max(1, spaces), empty: false, offset };
}

/** Starts a block that may cut a paragraph short. */
function interruptsParagraph(text: string, depth: number): boolean {
  if (fenceOpen(text) !== null || atxHeading(text) !== null || isRule(text)) return true;
  if (depth < MAX_DEPTH && quoteStart(text)) return true;
  const marker = depth < MAX_DEPTH ? listMarker(text) : null;
  return marker !== null && !marker.empty && (!marker.ordered || marker.start === 1);
}

// ---- tables -------------------------------------------------------------------------------------------------------

interface Cell {
  text: string;
  at: number;
}

/** Cells of a table row split at `|` that is not escaped; one leading and one trailing pipe are dropped. */
function tableCells(line: Line): Cell[] {
  const text = line.text;
  let start = 0;
  let end = text.length;
  while (start < end && (text[start] === ' ' || text[start] === '\t')) start += 1;
  while (end > start && (text[end - 1] === ' ' || text[end - 1] === '\t')) end -= 1;
  if (text[start] === '|') start += 1;
  if (end > start && text[end - 1] === '|' && text[end - 2] !== '\\') end -= 1;
  const cells: Cell[] = [];
  let cellStart = start;
  for (let i = start; i <= end; i += 1) {
    if (i === end || (text[i] === '|' && text[i - 1] !== '\\')) {
      const raw = text.slice(cellStart, i);
      let lead = 0;
      while (lead < raw.length && (raw[lead] === ' ' || raw[lead] === '\t')) lead += 1;
      cells.push({ text: raw.trim(), at: line.at + cellStart + lead });
      cellStart = i + 1;
    }
  }
  return cells;
}

function delimiterAlign(cell: string): MdAlign | undefined {
  const t = cell.trim();
  if (t === '') return undefined;
  const left = t.startsWith(':');
  const right = t.endsWith(':') && t.length > 1;
  const body = t.slice(left ? 1 : 0, right ? -1 : undefined);
  if (body === '') return undefined;
  for (const c of body) if (c !== '-') return undefined;
  if (left && right) return 'center';
  if (left) return 'left';
  if (right) return 'right';
  return null;
}

function tableAlign(line: Line, columns: number): MdAlign[] | null {
  if (!line.text.includes('-') || indentOf(line.text) > 3) return null;
  const cells = tableCells(line);
  if (cells.length !== columns) return null;
  const align: MdAlign[] = [];
  for (const cell of cells) {
    const a = delimiterAlign(cell.text);
    if (a === undefined) return null;
    align.push(a);
  }
  return align;
}

// ---- blocks -------------------------------------------------------------------------------------------------------

function paragraphInlines(lines: readonly Line[]): MdInline[] {
  const parts = lines.map((line, index) => (index === 0 ? line : dropSpaces(line)));
  if (parts.length > 0) {
    const last = parts[parts.length - 1]!;
    parts[parts.length - 1] = { text: trimEndSpaces(last.text), at: last.at };
  }
  return parseInlines(parts);
}

function parseBlocks(lines: readonly Line[], depth: number): MdBlock[] {
  const blocks: MdBlock[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    const text = line.text;
    if (isBlank(text)) {
      i += 1;
      continue;
    }
    const indent = indentOf(text);

    if (indent >= 4) {
      const body: Line[] = [];
      let end = i;
      while (end < lines.length && (isBlank(lines[end]!.text) || indentOf(lines[end]!.text) >= 4)) end += 1;
      while (end > i && isBlank(lines[end - 1]!.text)) end -= 1;
      for (let k = i; k < end; k += 1) body.push(dropCols(lines[k]!, 4));
      blocks.push({ type: 'code', at: body[0]!.at, lang: '', value: body.map((l) => l.text).join('\n'), closed: true });
      i = end;
      continue;
    }

    const fence = fenceOpen(text);
    if (fence !== null) {
      const body: string[] = [];
      let k = i + 1;
      let closed = false;
      while (k < lines.length) {
        if (fenceCloses(lines[k]!.text, fence)) {
          closed = true;
          break;
        }
        body.push(dropCols(lines[k]!, fence.indent).text);
        k += 1;
      }
      blocks.push({ type: 'code', at: line.at, lang: fence.lang, value: body.join('\n'), closed });
      i = closed ? k + 1 : k;
      continue;
    }

    const heading = atxHeading(text);
    if (heading !== null) {
      let content = text.slice(heading.content);
      content = trimEndSpaces(content);
      let end = content.length;
      while (end > 0 && content[end - 1] === '#') end -= 1;
      if (end < content.length && (end === 0 || content[end - 1] === ' ' || content[end - 1] === '\t')) {
        content = trimEndSpaces(content.slice(0, end));
      }
      const inner = dropSpaces({ text: content, at: line.at + heading.content });
      blocks.push({ type: 'heading', at: line.at, level: heading.level, children: parseInlines([inner]) });
      i += 1;
      continue;
    }

    if (isRule(text)) {
      blocks.push({ type: 'rule', at: line.at });
      i += 1;
      continue;
    }

    if (depth < MAX_DEPTH && quoteStart(text)) {
      const inner: Line[] = [];
      let k = i;
      let lastBlank = false;
      while (k < lines.length) {
        const candidate = lines[k]!;
        if (quoteStart(candidate.text)) {
          const stripped = dropSpaces(candidate);
          let rest: Line = { text: stripped.text.slice(1), at: stripped.at + 1 };
          if (rest.text.startsWith(' ')) rest = { text: rest.text.slice(1), at: rest.at + 1 };
          inner.push(rest);
          lastBlank = isBlank(rest.text);
          k += 1;
          continue;
        }
        if (!isBlank(candidate.text) && !lastBlank && !interruptsParagraph(candidate.text, depth)) {
          inner.push(candidate);
          k += 1;
          continue;
        }
        break;
      }
      blocks.push({ type: 'quote', at: line.at, children: parseBlocks(inner, depth + 1) });
      i = k;
      continue;
    }

    const marker = depth < MAX_DEPTH ? listMarker(text) : null;
    if (marker !== null) {
      const list = parseList(lines, i, marker, depth);
      blocks.push(list.block);
      i = list.next;
      continue;
    }

    if (text.includes('|') && i + 1 < lines.length) {
      const head = tableCells(line);
      const align = head.length > 0 && head.length <= MAX_TABLE_COLUMNS ? tableAlign(lines[i + 1]!, head.length) : null;
      if (align !== null) {
        const rows: MdInline[][][] = [];
        let k = i + 2;
        let total = head.length;
        while (k < lines.length) {
          const row = lines[k]!;
          if (isBlank(row.text) || interruptsParagraph(row.text, depth)) break;
          const cells = tableCells(row);
          // A short row is padded with at most as many empty cells as it has: a few bytes never make a huge table.
          const width = Math.min(head.length, cells.length * 2);
          total += width;
          if (total > MAX_TABLE_CELLS) break;
          const shaped: MdInline[][] = [];
          for (let c = 0; c < width; c += 1) {
            const cell = cells[c];
            shaped.push(cell === undefined ? [] : parseInlines([{ text: cell.text, at: cell.at }]));
          }
          rows.push(shaped);
          k += 1;
        }
        blocks.push({
          type: 'table',
          at: line.at,
          align,
          head: head.map((cell) => parseInlines([{ text: cell.text, at: cell.at }])),
          rows,
        });
        i = k;
        continue;
      }
    }

    const para: Line[] = [line];
    let k = i + 1;
    let level = 0;
    while (k < lines.length) {
      const next = lines[k]!;
      if (isBlank(next.text)) break;
      const setext = setextLevel(next.text);
      if (setext !== 0) {
        level = setext;
        k += 1;
        break;
      }
      if (interruptsParagraph(next.text, depth)) break;
      para.push(next);
      k += 1;
    }
    if (level !== 0) blocks.push({ type: 'heading', at: line.at, level, children: paragraphInlines(para) });
    else blocks.push({ type: 'paragraph', at: line.at, children: paragraphInlines(para) });
    i = k;
  }
  return blocks;
}

function sameList(a: ListMarker, b: ListMarker): boolean {
  return a.ordered === b.ordered && a.kind === b.kind;
}

function parseList(
  lines: readonly Line[],
  from: number,
  first: ListMarker,
  depth: number,
): { block: MdBlock; next: number } {
  const items: Line[][] = [];
  let tight = true;
  let i = from;
  let marker: ListMarker | null = first;
  while (marker !== null && i < lines.length) {
    const line = lines[i]!;
    const content = marker.content;
    const item: Line[] = [];
    item.push({ text: line.text.slice(marker.offset), at: line.at + marker.offset });
    i += 1;
    let blankRun = 0;
    let lastContent = !marker.empty;
    while (i < lines.length) {
      const next = lines[i]!;
      if (isBlank(next.text)) {
        blankRun += 1;
        item.push({ text: '', at: next.at });
        i += 1;
        continue;
      }
      if (indentOf(next.text) >= content) {
        if (blankRun > 0 && item.some((l) => !isBlank(l.text))) tight = false;
        blankRun = 0;
        item.push(dropCols(next, content));
        lastContent = true;
        i += 1;
        continue;
      }
      if (blankRun === 0 && lastContent && !interruptsParagraph(next.text, depth) && listMarker(next.text) === null) {
        item.push(next);
        i += 1;
        continue;
      }
      break;
    }
    while (item.length > 1 && isBlank(item[item.length - 1]!.text)) item.pop();
    items.push(item);
    if (i >= lines.length) break;
    const following = listMarker(lines[i]!.text);
    if (following !== null && sameList(following, first) && !isRule(lines[i]!.text)) {
      if (blankRun > 0) tight = false;
      marker = following;
    } else {
      marker = null;
      i -= blankRun;
    }
  }
  return {
    block: {
      type: 'list',
      at: lines[from]!.at,
      ordered: first.ordered,
      start: first.start,
      tight,
      items: items.map((item) => parseBlocks(item, depth + 1)),
    },
    next: Math.max(i, from + 1),
  };
}

// ---- inlines ------------------------------------------------------------------------------------------------------

const ASCII_PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~';

function isPunctuation(ch: string | undefined): boolean {
  if (ch === undefined) return false;
  if (ASCII_PUNCTUATION.includes(ch)) return true;
  const c = ch.charCodeAt(0);
  return (c >= 0x2010 && c <= 0x2027) || (c >= 0x3001 && c <= 0x3003) || (c >= 0xff01 && c <= 0xff0f);
}

function isSpace(ch: string | undefined): boolean {
  if (ch === undefined) return true;
  const c = ch.charCodeAt(0);
  return c <= 32 || c === 0xa0 || c === 0x3000 || (c >= 0x2000 && c <= 0x200a) || c === 0x2028 || c === 0x2029;
}

interface ListNode {
  node: MdInline | null;
  delim: Delim | null;
  prev: ListNode | null;
  next: ListNode | null;
}

interface Delim {
  ch: string;
  count: number;
  origCount: number;
  canOpen: boolean;
  canClose: boolean;
  /** Source offset of the first character still in the run. */
  at: number;
  holder: ListNode;
  prev: Delim | null;
  next: Delim | null;
}

interface Bracket {
  holder: ListNode;
  image: boolean;
  /** Order of opening: a link bracket at or below `InlineParser.inactiveUpTo` may no longer make a link. */
  id: number;
  /** The top of the delimiter stack when the bracket opened. */
  bottom: Delim | null;
  at: number;
  prev: Bracket | null;
}

/** Parse the inline content of these lines (joined by soft breaks). */
export function parseInlines(lines: readonly Line[]): MdInline[] {
  return new InlineParser(lines).run();
}

class InlineParser {
  private readonly s: string;
  private readonly starts: number[] = [];
  private readonly ats: number[] = [];
  private head: ListNode | null = null;
  private tail: ListNode | null = null;
  private delims: Delim | null = null;
  private brackets: Bracket | null = null;
  private bracketCount = 0;
  /** The next `>` at or after the last angle destination looked at: each search starts where the last one ended. */
  private nextGt = 0;
  private inactiveUpTo = 0;
  /** Nesting depth of each emphasis, strike and link node made so far. */
  private readonly depths = new Map<MdInline, number>();
  private pos = 0;
  private textStart = -1;
  /** Backtick run starts by run length; `next` is the first one not yet used or passed. */
  private codeCloses: Map<number, { list: number[]; next: number }> | null = null;

  constructor(lines: readonly Line[]) {
    let joined = '';
    lines.forEach((line, index) => {
      if (index > 0) joined += '\n';
      this.starts.push(joined.length);
      this.ats.push(line.at);
      joined += line.text;
    });
    this.s = joined;
  }

  /** Source offset of index `j` of the joined text. */
  private src(j: number): number {
    let lo = 0;
    let hi = this.starts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.starts[mid]! <= j) lo = mid;
      else hi = mid - 1;
    }
    return (this.ats[lo] ?? 0) + (j - (this.starts[lo] ?? 0));
  }

  private append(node: MdInline | null, delim: Delim | null = null): ListNode {
    const holder: ListNode = { node, delim, prev: this.tail, next: null };
    if (this.tail === null) this.head = holder;
    else this.tail.next = holder;
    this.tail = holder;
    return holder;
  }

  private text(value: string, at: number, len: number): void {
    if (value === '') return;
    this.append({ type: 'text', value, at, len });
  }

  private flushText(end: number): void {
    if (this.textStart === -1) return;
    if (end > this.textStart)
      this.text(this.s.slice(this.textStart, end), this.src(this.textStart), end - this.textStart);
    this.textStart = -1;
  }

  private remove(holder: ListNode): void {
    if (holder.prev === null) this.head = holder.next;
    else holder.prev.next = holder.next;
    if (holder.next === null) this.tail = holder.prev;
    else holder.next.prev = holder.prev;
  }

  private removeDelim(d: Delim): void {
    if (d.prev !== null) d.prev.next = d.next;
    if (d.next !== null) d.next.prev = d.prev;
    else this.delims = d.prev;
  }

  run(): MdInline[] {
    const s = this.s;
    while (this.pos < s.length) {
      const j = this.pos;
      const c = s[j]!;
      if (c === '\\') {
        const next = s[j + 1];
        if (next === '\n') {
          this.flushText(j);
          this.append({ type: 'break' });
          this.pos = j + 2;
          continue;
        }
        if (next !== undefined && ASCII_PUNCTUATION.includes(next)) {
          this.flushText(j);
          this.text(next, this.src(j), 2);
          this.pos = j + 2;
          continue;
        }
      } else if (c === '\n') {
        let cut = j;
        while (cut > 0 && s[cut - 1] === ' ' && this.textStart !== -1 && cut > this.textStart) cut -= 1;
        const hard = j - cut >= 2;
        this.flushText(cut);
        this.append({ type: hard ? 'break' : 'soft' });
        this.pos = j + 1;
        continue;
      } else if (c === '`') {
        if (this.codeSpan(j)) continue;
      } else if (c === '*' || c === '_' || c === '~') {
        this.delimRun(j);
        continue;
      } else if (c === '[' || (c === '!' && s[j + 1] === '[')) {
        this.flushText(j);
        const image = c === '!';
        const holder = this.append({ type: 'text', value: image ? '![' : '[', at: this.src(j), len: image ? 2 : 1 });
        this.bracketCount += 1;
        this.brackets = { holder, image, id: this.bracketCount, bottom: this.delims, at: j, prev: this.brackets };
        this.pos = j + (image ? 2 : 1);
        continue;
      } else if (c === ']') {
        this.flushText(j);
        this.closeBracket(j);
        continue;
      } else if (c === '<') {
        if (this.angleLink(j)) continue;
      } else if (c === 'h' && (s.startsWith('http://', j) || s.startsWith('https://', j))) {
        if (this.bareLink(j)) continue;
      }
      if (this.textStart === -1) this.textStart = j;
      this.pos = j + 1;
    }
    this.flushText(s.length);
    this.processEmphasis(null);
    return this.collect(this.head, null);
  }

  /** The deepest nesting among these nodes (0 when none nests). */
  private deepest(nodes: readonly MdInline[]): number {
    let max = 0;
    for (const node of nodes) max = Math.max(max, this.depths.get(node) ?? 0);
    return max;
  }

  /** Nodes from `from` up to (not including) `to`, with every leftover delimiter as text. */
  private collect(from: ListNode | null, to: ListNode | null): MdInline[] {
    const out: MdInline[] = [];
    for (let n = from; n !== null && n !== to; n = n.next) {
      if (n.delim !== null) {
        if (n.delim.count > 0)
          out.push({ type: 'text', value: n.delim.ch.repeat(n.delim.count), at: n.delim.at, len: n.delim.count });
      } else if (n.node !== null) {
        out.push(n.node);
      }
    }
    return out;
  }

  private codeSpan(j: number): boolean {
    const s = this.s;
    let len = 0;
    while (s[j + len] === '`') len += 1;
    if (this.codeCloses === null) {
      this.codeCloses = new Map();
      let k = 0;
      while (k < s.length) {
        if (s[k] !== '`') {
          k += 1;
          continue;
        }
        let run = 0;
        while (s[k + run] === '`') run += 1;
        const runs = this.codeCloses.get(run);
        if (runs === undefined) this.codeCloses.set(run, { list: [k], next: 0 });
        else runs.list.push(k);
        k += run;
      }
    }
    const runs = this.codeCloses.get(len);
    let close = -1;
    if (runs !== undefined) {
      while (runs.next < runs.list.length && runs.list[runs.next]! <= j) runs.next += 1;
      if (runs.next < runs.list.length) {
        close = runs.list[runs.next]!;
        runs.next += 1;
      }
    }
    if (close === -1) {
      if (this.textStart === -1) this.textStart = j;
      this.pos = j + len;
      return true;
    }
    this.flushText(j);
    let body = s.slice(j + len, close);
    let value = '';
    for (const ch of body) value += ch === '\n' ? ' ' : ch;
    body = value;
    if (body.length > 2 && body.startsWith(' ') && body.endsWith(' ') && body.trim() !== '') body = body.slice(1, -1);
    this.append({ type: 'code', value: body, at: this.src(j + len) });
    this.pos = close + len;
    return true;
  }

  private delimRun(j: number): void {
    const s = this.s;
    const ch = s[j]!;
    let count = 0;
    while (s[j + count] === ch) count += 1;
    this.flushText(j);
    this.pos = j + count;
    if (ch === '~' && count !== 2) {
      this.text(ch.repeat(count), this.src(j), count);
      return;
    }
    const before = j === 0 ? undefined : s[j - 1];
    const after = s[j + count];
    const left = !isSpace(after) && (!isPunctuation(after) || isSpace(before) || isPunctuation(before));
    const right = !isSpace(before) && (!isPunctuation(before) || isSpace(after) || isPunctuation(after));
    let canOpen = left;
    let canClose = right;
    if (ch === '_') {
      canOpen = left && (!right || isPunctuation(before));
      canClose = right && (!left || isPunctuation(after));
    }
    const delim: Delim = {
      ch,
      count,
      origCount: count,
      canOpen,
      canClose,
      at: this.src(j),
      holder: null as unknown as ListNode,
      prev: this.delims,
      next: null,
    };
    delim.holder = this.append(null, delim);
    if (this.delims !== null) this.delims.next = delim;
    this.delims = delim;
  }

  /** CommonMark "process emphasis" over the delimiters above `bottom`. */
  private processEmphasis(bottom: Delim | null): void {
    let first: Delim | null = this.delims;
    while (first !== null && first.prev !== bottom) first = first.prev;
    if (first === null) return;
    const openersBottom = new Map<string, Delim | null>();
    let closer: Delim | null = first;
    while (closer !== null) {
      if (!closer.canClose) {
        closer = closer.next;
        continue;
      }
      const key = `${closer.ch}${closer.canOpen ? 1 : 0}${closer.origCount % 3}`;
      const floor = openersBottom.has(key) ? openersBottom.get(key)! : bottom;
      let opener: Delim | null = closer.prev;
      let found = false;
      while (opener !== null && opener !== bottom && opener !== floor) {
        if (opener.ch === closer.ch && opener.canOpen) {
          const odd =
            (opener.canClose || closer.canOpen) &&
            (opener.origCount + closer.origCount) % 3 === 0 &&
            !(opener.origCount % 3 === 0 && closer.origCount % 3 === 0);
          if (!odd || closer.ch === '~') {
            found = true;
            break;
          }
        }
        opener = opener.prev;
      }
      const tooDeep =
        found && opener !== null && this.deepest(this.collect(opener.holder.next, closer.holder)) >= MAX_INLINE_DEPTH;
      if (!found || opener === null || tooDeep) {
        openersBottom.set(key, closer.prev);
        const next: Delim | null = closer.next;
        if (!closer.canOpen) this.removeDelim(closer);
        closer = next;
        continue;
      }
      const use = closer.ch === '~' ? 2 : opener.count >= 2 && closer.count >= 2 ? 2 : 1;
      const type = closer.ch === '~' ? 'del' : use === 2 ? 'strong' : 'em';
      const children = this.collect(opener.holder.next, closer.holder);
      for (let d = closer.prev; d !== null && d !== opener; d = d.prev) this.removeDelim(d);
      const made: MdInline = { type, children };
      this.depths.set(made, this.deepest(children) + 1);
      const wrapper: ListNode = { node: made, delim: null, prev: opener.holder, next: closer.holder };
      opener.holder.next = wrapper;
      closer.holder.prev = wrapper;
      opener.count -= use;
      closer.count -= use;
      closer.at += use;
      if (opener.count === 0) {
        this.removeDelim(opener);
        this.remove(opener.holder);
      }
      if (closer.count === 0) {
        const next: Delim | null = closer.next;
        this.removeDelim(closer);
        this.remove(closer.holder);
        closer = next;
      }
    }
    while (this.delims !== null && this.delims !== bottom) this.removeDelim(this.delims);
  }

  private closeBracket(j: number): void {
    const opener = this.brackets;
    if (opener === null) {
      this.text(']', this.src(j), 1);
      this.pos = j + 1;
      return;
    }
    this.brackets = opener.prev;
    if (!opener.image && opener.id <= this.inactiveUpTo) {
      this.text(']', this.src(j), 1);
      this.pos = j + 1;
      return;
    }
    const dest = this.s[j + 1] === '(' ? this.linkTail(j + 2) : null;
    if (dest !== null) {
      this.processEmphasis(opener.bottom);
      const children = this.collect(opener.holder.next, null);
      const node: MdInline = opener.image
        ? { type: 'image', href: dest.href, alt: plainText(children), at: this.src(opener.at) }
        : { type: 'link', href: dest.href, children: unlink(children) };
      if (!opener.image) this.depths.set(node, this.deepest(children) + 1);
      this.tail = opener.holder.prev;
      if (this.tail === null) this.head = null;
      else this.tail.next = null;
      this.append(node);
      if (!opener.image) this.inactiveUpTo = this.bracketCount;
      this.pos = dest.end;
      return;
    }
    const only = opener.holder.next;
    if (!opener.image && only !== null && only.next === null && only.node?.type === 'text') {
      const n = citeNumber(only.node.value);
      if (n !== null) {
        this.tail = opener.holder.prev;
        if (this.tail === null) this.head = null;
        else this.tail.next = null;
        this.append({ type: 'cite', n, at: this.src(opener.at) });
        this.pos = j + 1;
        return;
      }
    }
    this.text(']', this.src(j), 1);
    this.pos = j + 1;
  }

  /** `dest [title])` from `k` (just after the opening parenthesis), or null. */
  private linkTail(k: number): { href: string; end: number } | null {
    const s = this.s;
    let i = k;
    const skip = (): void => {
      let newlines = 0;
      while (i < s.length && (s[i] === ' ' || s[i] === '\t' || (s[i] === '\n' && newlines++ === 0))) i += 1;
    };
    skip();
    let href = '';
    if (s[i] === '<') {
      if (this.nextGt !== -1 && this.nextGt <= i) this.nextGt = s.indexOf('>', i + 1);
      const close = this.nextGt;
      if (close === -1 || close - i > MAX_LINK_DEST) return null;
      href = s.slice(i + 1, close);
      if (href.includes('\n') || href.includes('<')) return null;
      i = close + 1;
    } else {
      let depth = 0;
      const start = i;
      while (i < s.length && i - start <= MAX_LINK_DEST) {
        const c = s[i]!;
        if (c === '\\' && i + 1 < s.length && ASCII_PUNCTUATION.includes(s[i + 1]!)) {
          href += s[i + 1];
          i += 2;
          continue;
        }
        if (c.charCodeAt(0) <= 32) break;
        if (c === '(') {
          depth += 1;
          if (depth > 32) return null;
        } else if (c === ')') {
          if (depth === 0) break;
          depth -= 1;
        }
        href += c;
        i += 1;
      }
      if (i - start > MAX_LINK_DEST || depth !== 0) return null;
    }
    const beforeTitle = i;
    skip();
    const q = s[i];
    if (i > beforeTitle && (q === '"' || q === "'" || q === '(')) {
      const end = q === '(' ? ')' : q;
      let t = i + 1;
      while (t < s.length && s[t] !== end && t - i <= MAX_LINK_DEST) t += s[t] === '\\' ? 2 : 1;
      if (t >= s.length || s[t] !== end) return null;
      i = t + 1;
      skip();
    }
    if (s[i] !== ')') return null;
    return { href, end: i + 1 };
  }

  private angleLink(j: number): boolean {
    const s = this.s;
    if (!(s.startsWith('<http://', j) || s.startsWith('<https://', j) || s.startsWith('<mailto:', j))) return false;
    let k = j + 1;
    while (k < s.length && k - j <= MAX_LINK_DEST) {
      const c = s[k]!;
      if (c === '>') break;
      if (c === '<' || c.charCodeAt(0) <= 32) return false;
      k += 1;
    }
    if (s[k] !== '>') return false;
    this.flushText(j);
    const href = s.slice(j + 1, k);
    this.append({
      type: 'link',
      href,
      children: [{ type: 'text', value: href, at: this.src(j + 1), len: href.length }],
    });
    this.pos = k + 1;
    return true;
  }

  private bareLink(j: number): boolean {
    const s = this.s;
    const before = j === 0 ? undefined : s[j - 1];
    if (!(isSpace(before) || before === '(' || before === '*' || before === '_' || before === '~')) return false;
    let k = j;
    while (k < s.length && !isSpace(s[k]) && s[k] !== '<' && s[k] !== '>' && s[k] !== '"' && s[k] !== '`') k += 1;
    let end = k;
    const balance = { ')': 0, ']': 0 };
    for (let m = j; m < end; m += 1) {
      if (s[m] === '(') balance[')'] -= 1;
      else if (s[m] === ')') balance[')'] += 1;
      else if (s[m] === '[') balance[']'] -= 1;
      else if (s[m] === ']') balance[']'] += 1;
    }
    for (;;) {
      const last = s[end - 1];
      if (last !== undefined && '?!.,:;*_~\'"'.includes(last)) {
        end -= 1;
        continue;
      }
      if ((last === ')' || last === ']') && balance[last] > 0) {
        balance[last] -= 1;
        end -= 1;
        continue;
      }
      break;
    }
    const href = s.slice(j, end);
    if (href === 'http://' || href === 'https://' || href.length <= 'https://'.length - 1) return false;
    this.flushText(j);
    this.append({ type: 'link', href, children: [{ type: 'text', value: href, at: this.src(j), len: href.length }] });
    this.pos = end;
    return true;
  }
}

function citeNumber(text: string): number | null {
  if (text.length === 0 || text.length > 2) return null;
  for (const ch of text) if (ch < '0' || ch > '9') return null;
  if (text[0] === '0') return null;
  return Number(text);
}

/** Links inside link text become their text: a link never holds a link. */
function unlink(nodes: MdInline[]): MdInline[] {
  const out: MdInline[] = [];
  for (const node of nodes) {
    if (node.type === 'link') out.push(...unlink(node.children));
    else if (node.type === 'em' || node.type === 'strong' || node.type === 'del')
      out.push({ type: node.type, children: unlink(node.children) });
    else out.push(node);
  }
  return out;
}

/** The text of inline nodes, as an image's alt text or a link's visible words. */
export function plainText(nodes: readonly MdInline[]): string {
  let out = '';
  for (const node of nodes) {
    switch (node.type) {
      case 'text':
      case 'code':
        out += node.value;
        break;
      case 'em':
      case 'strong':
      case 'del':
      case 'link':
        out += plainText(node.children);
        break;
      case 'image':
        out += node.alt;
        break;
      case 'cite':
        out += `[${node.n}]`;
        break;
      case 'break':
      case 'soft':
        out += ' ';
        break;
    }
  }
  return out;
}

/** The blocks of `text`; offsets are `base` + the index in `text`. */
export function parseMarkdown(text: string, base = 0): MdBlock[] {
  return parseBlocks(splitLines(text, base), 0);
}
