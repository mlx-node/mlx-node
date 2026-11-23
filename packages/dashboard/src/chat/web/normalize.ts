/**
 * Caps and cleaning of provider text before the model reads it, all by hand-written scanners. Every search field is
 * one line (line breaks become spaces before the caps), so a snippet can never add a result block; the whole search
 * text keeps exactly the `results=N` blocks it announces. A provider may echo a key: every key is redacted.
 */

import {
  formatErrorText,
  formatPageText,
  formatSearchText,
  type WebFetchProvider,
  type WebSearchItem,
  type WebSearchProvider,
} from './result-text.js';

export const MAX_RESULTS = 8;
export const TITLE_CHARS = 160;
export const URL_CHARS = 500;
export const SNIPPET_CHARS = 500;
export const SEARCH_TEXT_CHARS = 6_000;
export const PAGE_CHARS = 12_000;
export const PAGE_BYTES = 16_000;
export const REDACTED = '[redacted]';

/** C0 controls except `\n` and `\t`, DEL, C1 controls, zero-width and bidi marks, the byte order mark. */
function isRemoved(code: number): boolean {
  if (code < 0x20) return code !== 0x0a && code !== 0x09;
  if (code >= 0x7f && code <= 0x9f) return true;
  return (
    (code >= 0x200b && code <= 0x200f) ||
    (code >= 0x202a && code <= 0x202e) ||
    (code >= 0x2066 && code <= 0x2069) ||
    code === 0xfeff
  );
}

function isLineBreak(code: number): boolean {
  return code === 0x0a || code === 0x0d || code === 0x09 || code === 0x85 || code === 0x2028 || code === 0x2029;
}

export function cleanText(text: string): string {
  let out = '';
  for (let i = 0; i < text.length; i++) if (!isRemoved(text.charCodeAt(i))) out += text[i];
  return out;
}

/** One line: each line break or tab becomes a space first, then the removed characters go, then space runs collapse. */
export function oneLineField(text: string): string {
  let out = '';
  let space = false;
  for (let i = 0; i < text.length; i++) {
    let code = text.charCodeAt(i);
    if (isLineBreak(code)) code = 0x20;
    if (isRemoved(code)) continue;
    if (code === 0x20) {
      space = true;
      continue;
    }
    if (space && out !== '') out += ' ';
    space = false;
    out += text[i];
  }
  return out;
}

/** At most `max` UTF-16 units, never splitting a surrogate pair. */
export function capChars(text: string, max: number): string {
  if (text.length <= max) return text;
  let end = max;
  const last = text.charCodeAt(end - 1);
  if (last >= 0xd800 && last <= 0xdbff) end -= 1;
  return text.slice(0, end);
}

/** At most `max` UTF-8 bytes, cut on a code point boundary. */
export function capBytes(text: string, max: number): string {
  if (Buffer.byteLength(text, 'utf8') <= max) return text;
  let bytes = 0;
  let end = 0;
  for (const ch of text) {
    const size = Buffer.byteLength(ch, 'utf8');
    if (bytes + size > max) break;
    bytes += size;
    end += ch.length;
  }
  return text.slice(0, end);
}

/** No more than two blank lines in a row. */
export function collapseBlankLines(text: string): string {
  const out: string[] = [];
  let blank = 0;
  for (const line of text.split('\n')) {
    if (line.trim() === '') {
      blank += 1;
      if (blank > 2) continue;
    } else {
      blank = 0;
    }
    out.push(line);
  }
  return out.join('\n');
}

export function redactKeys(text: string, keys: readonly string[]): string {
  let out = text;
  for (const key of keys) if (key !== '') out = out.split(key).join(REDACTED);
  return out;
}

function digits(text: string, from: number, to: number): boolean {
  for (let i = from; i < to; i++) {
    const c = text.charCodeAt(i);
    if (c < 0x30 || c > 0x39) return false;
  }
  return true;
}

/** `YYYY-MM-DD` from a text that starts with an ISO date, else null. */
export function isoDate(value: unknown): string | null {
  if (typeof value !== 'string' || value.length < 10) return null;
  if (!digits(value, 0, 4) || value[4] !== '-' || !digits(value, 5, 7) || value[7] !== '-' || !digits(value, 8, 10)) {
    return null;
  }
  if (value.length > 10 && value[10] !== 'T' && value[10] !== ' ') return null;
  const month = Number(value.slice(5, 7));
  const day = Number(value.slice(8, 10));
  if (month < 1 || month > 12 || day < 1 || day > 31) return null;
  return value.slice(0, 10);
}

/** The date `days` before `now`, as `YYYY-MM-DD` (UTC). */
export function dateBefore(now: number, days: number): string {
  return new Date(now - days * 86_400_000).toISOString().slice(0, 10);
}

/** The same rule as the result-text reader: an http(s) address with no white space at either end. */
function isHttpUrl(text: string): boolean {
  if (text.trim() !== text) return false;
  try {
    const url = new URL(text);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
}

/** A provider's result before cleaning: any of its fields may be missing or of the wrong type. */
export interface RawResult {
  title: unknown;
  url: unknown;
  date: unknown;
  snippet: unknown;
}

function field(value: unknown, max: number, keys: readonly string[]): string {
  return capChars(redactKeys(oneLineField(typeof value === 'string' ? value : ''), keys), max);
}

/** The cleaned results: at most `count` (≤ 8); a result without a usable http(s) address is dropped. */
export function cleanResults(
  raw: readonly RawResult[],
  count: number,
  keys: readonly string[],
): Array<Omit<WebSearchItem, 'n'>> {
  const max = Math.min(Math.max(1, Math.trunc(count)), MAX_RESULTS);
  const items: Array<Omit<WebSearchItem, 'n'>> = [];
  for (const r of raw) {
    if (items.length >= max) break;
    const url = redactKeys(oneLineField(typeof r.url === 'string' ? r.url : ''), keys);
    if (url === '' || url.length > URL_CHARS || !isHttpUrl(url)) continue;
    items.push({
      title: field(r.title, TITLE_CHARS, keys),
      url,
      date: isoDate(r.date),
      snippet: field(r.snippet, SNIPPET_CHARS, keys),
    });
  }
  return items;
}

/** The search result text, at most 6,000 characters: whole results are dropped from the end and `results=N` follows. */
export function searchText(
  provider: WebSearchProvider,
  raw: readonly RawResult[],
  count: number,
  keys: readonly string[],
): string {
  const items = cleanResults(raw, count, keys);
  let text = formatSearchText(provider, items);
  while (text.length > SEARCH_TEXT_CHARS && items.length > 0) {
    items.pop();
    text = formatSearchText(provider, items);
  }
  return text;
}

/** The page result text: cleaned Markdown cut at 12,000 characters and 16,000 bytes, or `no-content` when empty. */
export function pageText(
  provider: WebFetchProvider,
  url: string,
  title: unknown,
  markdown: unknown,
  keys: readonly string[],
): string {
  const body = redactKeys(collapseBlankLines(cleanText(typeof markdown === 'string' ? markdown : '')).trim(), keys);
  if (body === '') return formatErrorText('no-content', provider);
  const capped = capBytes(capChars(body, PAGE_CHARS), PAGE_BYTES);
  return formatPageText(provider, {
    url,
    title: field(title, TITLE_CHARS, keys),
    cut: capped.length < body.length,
    text: capped,
  });
}
