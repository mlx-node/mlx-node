/**
 * The app's web calls as the thread reads them. A call reaches `mlx_web` in two shapes (through `use_tool`, or by
 * its own name); both are read here. Result text is trusted only through `parseWebResultText`, and only on an entry
 * that is a web call: sources, citations and the hosts a Chat link may open come from nothing else.
 */

import { CHAT } from './chat-strings';
import type { TurnOutcome } from './chat-types';
import type { TranscriptEntry } from './types';
import { parseWebResultText, type WebErrorCode, type WebResult } from './web-result-text';

export const WEB_SEARCH_TOOL = 'mlx_web__web_search';
export const WEB_FETCH_TOOL = 'mlx_web__web_fetch';

export type WebCall = { kind: 'search'; query: string } | { kind: 'fetch'; url: string; host: string };

export interface WebSource {
  url: string;
  /** Display host: lower case, no leading `www.`. */
  host: string;
  title: string;
  date: string | null;
  via: 'search' | 'page';
}

type ToolEntry = Extract<TranscriptEntry, { kind: 'tool' }>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseJson(text: string | undefined): unknown {
  if (text === undefined) return undefined;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  }
}

/** The URL when it parses and is http(s), else null. */
export function httpUrl(text: string): URL | null {
  let url: URL;
  try {
    url = new URL(text);
  } catch {
    return null;
  }
  return url.protocol === 'http:' || url.protocol === 'https:' ? url : null;
}

/** Host for comparing: lower case, one trailing dot dropped. */
export function hostKey(url: URL): string {
  const host = url.hostname.toLowerCase();
  return host.endsWith('.') ? host.slice(0, -1) : host;
}

/** Host for showing: `hostKey` without a leading `www.`. */
export function displayHost(url: URL): string {
  const host = hostKey(url);
  return host.startsWith('www.') ? host.slice(4) : host;
}

function hexDigit(ch: string | undefined): boolean {
  return ch !== undefined && ((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F'));
}

/** A JSON string from its opening quote at `at`, to its closing quote or the end of a cut text. */
function cutString(text: string, at: number): { value: string; end: number } {
  let value = '';
  let i = at + 1;
  while (i < text.length && text[i] !== '"') {
    if (text[i] !== '\\') {
      value += text[i];
      i += 1;
      continue;
    }
    const e = text[i + 1];
    if (e === 'u' && hexDigit(text[i + 2]) && hexDigit(text[i + 3]) && hexDigit(text[i + 4]) && hexDigit(text[i + 5])) {
      value += String.fromCharCode(Number.parseInt(text.slice(i + 2, i + 6), 16));
      i += 6;
      continue;
    }
    if (e === 'n' || e === 'r' || e === 't') value += ' ';
    else if (e !== undefined) value += e;
    i += 2;
  }
  return { value, end: i + 1 };
}

/**
 * The leading string fields of an object text that may be cut short, for showing only: `{"query": "words` gives
 * `{query: "words"}`. Reading stops at the first value that is not a string.
 */
function cutObjectFields(text: string): Record<string, string> | null {
  const fields = Object.create(null) as Record<string, string>;
  let found = false;
  let i = 0;
  const space = (): void => {
    while (i < text.length && (text[i] === ' ' || text[i] === '\n' || text[i] === '\r' || text[i] === '\t')) i += 1;
  };
  space();
  if (text[i] !== '{') return null;
  i += 1;
  for (;;) {
    space();
    if (text[i] !== '"') break;
    const key = cutString(text, i);
    i = key.end;
    space();
    if (text[i] !== ':') break;
    i += 1;
    space();
    if (text[i] !== '"') break;
    const value = cutString(text, i);
    i = value.end;
    if (!Object.hasOwn(fields, key.value)) fields[key.value] = value.value;
    found = true;
    space();
    if (text[i] !== ',') break;
    i += 1;
  }
  return found ? fields : null;
}

/** The fields of a tool input: an object, or a text the model wrote in place of one. */
function inputFields(input: unknown): Record<string, unknown> | null {
  if (isRecord(input)) return input;
  if (typeof input !== 'string') return null;
  const parsed = parseJson(input);
  return isRecord(parsed) ? parsed : cutObjectFields(input);
}

function callOf(tool: string, raw: unknown): WebCall | null {
  const input = inputFields(raw);
  if (input === null) return null;
  if (tool === WEB_SEARCH_TOOL) {
    const query = input.query;
    return typeof query === 'string' && query.trim() !== '' ? { kind: 'search', query } : null;
  }
  if (tool === WEB_FETCH_TOOL) {
    const url = input.url;
    if (typeof url !== 'string' || url.trim() === '') return null;
    const parsed = httpUrl(url.trim());
    return { kind: 'fetch', url, host: parsed === null ? oneLine(url) : displayHost(parsed) };
  }
  return null;
}

/** A web call of either shape, or null (another tool, or an input that does not fit). */
export function parseWebCall(name: string, input: string | undefined): WebCall | null {
  const raw = parseJson(input);
  if (typeof raw === 'string' && (name === WEB_SEARCH_TOOL || name === WEB_FETCH_TOOL)) return callOf(name, raw);
  if (!isRecord(raw)) return null;
  if (name === 'use_tool') {
    if (raw.variant !== undefined && raw.variant !== 'UseTool') return null;
    const tool = raw.tool_name;
    return typeof tool === 'string' ? callOf(tool, raw.tool_input) : null;
  }
  if (name !== WEB_SEARCH_TOOL && name !== WEB_FETCH_TOOL) return null;
  if (raw.variant !== undefined || raw.tool_name !== undefined || raw.tool_input !== undefined) {
    if (raw.variant !== undefined && raw.variant !== 'MCPTool') return null;
    if (raw.tool_name !== name) return null;
    return callOf(name, raw.tool_input);
  }
  return callOf(name, raw);
}

/** The tool's text read by the strict reader; undefined or refused text is null. */
export function parseWebResult(output: string | undefined): WebResult | null {
  return output === undefined ? null : parseWebResultText(output);
}

/** The call and its result for a tool entry, or null when it is not a web call. */
export function webCallOf(entry: ToolEntry): { call: WebCall; result: WebResult | null } | null {
  const call = parseWebCall(entry.name, entry.input);
  if (call === null) return null;
  const result = parseWebResult(entry.output);
  if (result === null) return { call, result: null };
  const fits = call.kind === 'search' ? result.kind !== 'page' : result.kind !== 'search';
  return { call, result: fits ? plainTitles(result) : null };
}

/** Why the app refused a call: the tool or its form, the address of a read, or an input text it could not read. */
export type Refusal = 'tool' | 'address' | 'arguments';

const REFUSALS: Partial<Record<TurnOutcome, Refusal>> = {
  'tool-refused': 'tool',
  'address-refused': 'address',
  'arguments-unreadable': 'arguments',
};

/**
 * In a turn that ended on a refusal, the call the app refused: the agent ends the turn on that rejection, so it is the
 * turn's last tool call (a read for `address-refused`; a `use_tool` read whose input is a text for
 * `arguments-unreadable`, whatever that text holds), ended without a result. Null when that call has a result or is not
 * over.
 */
function fetchText(entry: ToolEntry): boolean {
  const raw = parseJson(entry.input);
  return (
    entry.name === 'use_tool' && isRecord(raw) && raw.tool_name === WEB_FETCH_TOOL && typeof raw.tool_input === 'string'
  );
}

export function refusedCall(
  entries: readonly TranscriptEntry[],
  outcome: TurnOutcome | undefined,
): { entry: TranscriptEntry; refusal: Refusal } | null {
  const refusal = outcome === undefined ? undefined : REFUSALS[outcome];
  if (refusal === undefined) return null;
  for (let i = entries.length - 1; i >= 0; i -= 1) {
    const entry = entries[i]!;
    if (entry.kind !== 'tool') continue;
    const web = webCallOf(entry);
    if (refusal === 'address' && web?.call.kind !== 'fetch') continue;
    if (refusal === 'arguments' && !fetchText(entry)) continue;
    const over = entry.status !== 'completed' && entry.status !== 'in_progress' && entry.status !== 'pending';
    return over && (web?.result ?? null) === null ? { entry, refusal } : null;
  }
  return null;
}

/** Page titles as one plain line: a bidi override in a title could make a host or a name read in another order. */
function plainTitles(result: WebResult): WebResult {
  if (result.kind === 'search')
    return { ...result, results: result.results.map((item) => ({ ...item, title: oneLine(item.title) })) };
  if (result.kind === 'page') return { ...result, title: oneLine(result.title) };
  return result;
}

/** Scheme and host lower-cased, fragment and one trailing `/` of the path dropped. */
export function sourceKey(url: URL): string {
  let path = url.pathname;
  if (path.endsWith('/')) path = path.slice(0, -1);
  const port = url.port === '' ? '' : `:${url.port}`;
  return `${url.protocol}//${hostKey(url)}${port}${path}${url.search}`;
}

/** Every source the web calls among `entries` returned, de-duplicated by address, in arrival order. */
export function turnSources(entries: readonly TranscriptEntry[]): WebSource[] {
  const seen = new Set<string>();
  const out: WebSource[] = [];
  const add = (address: string, title: string, date: string | null, via: WebSource['via']): void => {
    const url = httpUrl(address);
    if (url === null) return;
    const key = sourceKey(url);
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ url: address, host: displayHost(url), title, date, via });
  };
  for (const entry of entries) {
    if (entry.kind !== 'tool') continue;
    const web = webCallOf(entry);
    if (web === null || web.result === null) continue;
    if (web.result.kind === 'search') {
      for (const item of web.result.results) add(item.url, item.title, item.date, 'search');
    } else if (web.result.kind === 'page') {
      add(web.result.url, web.result.title, null, 'page');
    }
  }
  return out;
}

function isAddressEnd(ch: string): boolean {
  const c = ch.charCodeAt(0);
  if (ch === '<' || ch === '>' || ch === '"' || ch === '`') return true;
  return (
    c <= 32 || c === 0x85 || c === 0xa0 || c === 0x2028 || c === 0x2029 || c === 0x3000 || (c >= 0x2000 && c <= 0x200a)
  );
}

const TRAILING_PUNCTUATION = '.,;:!?)]';

/** Every `http://` / `https://` address in a text, read to white space, `<`, `>`, `"` or a back-tick. */
export function scanAddresses(text: string): string[] {
  const found: string[] = [];
  let i = 0;
  while (i < text.length) {
    const http = text.indexOf('http://', i);
    const https = text.indexOf('https://', i);
    const start = http === -1 ? https : https === -1 ? http : Math.min(http, https);
    if (start === -1) break;
    let end = start;
    while (end < text.length && !isAddressEnd(text[end]!)) end += 1;
    let address = text.slice(start, end);
    while (address.length > 0 && TRAILING_PUNCTUATION.includes(address[address.length - 1]!))
      address = address.slice(0, -1);
    if (address !== '') found.push(address);
    i = end > start ? end : start + 1;
  }
  return found;
}

/**
 * The hosts a link in a Chat answer may open: every address the user wrote and every search result or read page of
 * the conversation. Hosts as `hostKey` gives them.
 */
export function conversationHosts(entries: readonly TranscriptEntry[]): ReadonlySet<string> {
  const hosts = new Set<string>();
  const add = (address: string): void => {
    const url = httpUrl(address);
    if (url !== null) hosts.add(hostKey(url));
  };
  for (const entry of entries) {
    if (entry.kind === 'user') {
      for (const address of scanAddresses(entry.text)) add(address);
    } else if (entry.kind === 'tool') {
      const web = webCallOf(entry);
      if (web === null || web.result === null) continue;
      if (web.result.kind === 'search') for (const item of web.result.results) add(item.url);
      else if (web.result.kind === 'page') add(web.result.url);
    }
  }
  return hosts;
}

const SOURCE_HEADINGS = new Set(['sources', 'sources:', '**sources**', '**sources:**', '**sources**:']);

function isSourcesHeading(line: string): boolean {
  let text = line.trim();
  let hashes = 0;
  while (hashes < text.length && text[hashes] === '#') hashes += 1;
  if (hashes > 0 && hashes <= 6 && text[hashes] === ' ') text = text.slice(hashes + 1).trim();
  return SOURCE_HEADINGS.has(text.toLowerCase());
}

function digitsValue(text: string): number | null {
  if (text === '' || text.length > 2) return null;
  for (const ch of text) if (ch < '0' || ch > '9') return null;
  const n = Number(text);
  return n >= 1 ? n : null;
}

/** `[n] <title> - <address>` (also with a list marker, or as `[n] [title](address)`), else null. */
function listedLine(line: string): { n: number; address: string } | null {
  let text = line.trim();
  if (text.startsWith('- ') || text.startsWith('* ')) text = text.slice(2).trimStart();
  if (!text.startsWith('[')) return null;
  const close = text.indexOf(']');
  const n = close === -1 ? null : digitsValue(text.slice(1, close));
  if (n === null) return null;
  const rest = text.slice(close + 1).trim();
  let address: string;
  const link = rest.lastIndexOf('](');
  if (rest.startsWith('[') && link !== -1 && rest.endsWith(')')) {
    address = rest.slice(link + 2, -1).trim();
  } else {
    let cut = -1;
    for (const sep of [' - ', ' — ', ' – ']) cut = Math.max(cut, rest.lastIndexOf(sep));
    address = (cut === -1 ? rest : rest.slice(cut + 3)).trim();
  }
  if (address.startsWith('<') && address.endsWith('>')) address = address.slice(1, -1);
  return address === '' ? null : { n, address };
}

/**
 * A trailing "Sources" block of an answer: a heading line, then only `[n] … - <address>` lines. `body` is the text
 * before it; with no such block `body` is the whole text and `listed` is empty.
 */
export function splitAnswerSources(text: string): { body: string; listed: Map<number, string> } {
  const lines = text.split('\n');
  let end = lines.length;
  while (end > 0 && lines[end - 1]!.trim() === '') end -= 1;
  let at = end - 1;
  const items: Array<{ n: number; address: string }> = [];
  while (at >= 0) {
    const line = lines[at]!;
    if (line.trim() === '') {
      at -= 1;
      continue;
    }
    const item = listedLine(line);
    if (item === null) break;
    items.unshift(item);
    at -= 1;
  }
  if (items.length === 0 || at < 0 || !isSourcesHeading(lines[at]!)) return { body: text, listed: new Map() };
  const listed = new Map<number, string>();
  for (const item of items) if (!listed.has(item.n)) listed.set(item.n, item.address);
  return { body: lines.slice(0, at).join('\n').trimEnd(), listed };
}

/** The listed addresses that are real sources of the turn, as citation number → source. */
export function matchCitations(
  listed: ReadonlyMap<number, string>,
  sources: readonly WebSource[],
): Map<number, WebSource> {
  const byKey = new Map<string, WebSource>();
  for (const source of sources) {
    const url = httpUrl(source.url);
    if (url !== null) byKey.set(sourceKey(url), source);
  }
  const cited = new Map<number, WebSource>();
  for (const [n, address] of listed) {
    const url = httpUrl(address);
    const source = url === null ? undefined : byKey.get(sourceKey(url));
    if (source !== undefined) cited.set(n, source);
  }
  return cited;
}

const PROVIDER_NAMES: Readonly<Record<string, string>> = {
  exa: 'Exa',
  deepseek: 'DeepSeek',
  brave: 'Brave',
  firecrawl: 'Firecrawl',
};

export function providerName(id: string): string {
  return PROVIDER_NAMES[id] ?? id;
}

export function webErrorText(code: WebErrorCode): string {
  return CHAT.webErrors[code];
}

/** Codes whose row offers the Web Search settings link. */
export function errorNeedsSettings(code: WebErrorCode): boolean {
  return code === 'not-configured' || code === 'key-rejected' || code === 'no-credit';
}

/** "5 results", "1 result", "No results". */
export function resultCountText(n: number): string {
  if (n === 0) return CHAT.noResults;
  if (n === 1) return CHAT.oneResult;
  return CHAT.results.split('{n}').join(String(n));
}

/** Control characters, bidi embeddings, overrides and isolates, and direction marks: they could reorder a title. */
function isHiddenChar(code: number): boolean {
  return (
    code <= 0x1f ||
    (code >= 0x7f && code <= 0x9f) ||
    code === 0x061c ||
    code === 0x200e ||
    code === 0x200f ||
    (code >= 0x202a && code <= 0x202e) ||
    (code >= 0x2066 && code <= 0x2069)
  );
}

/** One line of text: line breaks and tabs become spaces, the other hidden characters are dropped. */
export function oneLine(text: string): string {
  let out = '';
  for (const ch of text) {
    if (ch === '\n' || ch === '\r' || ch === '\t') out += ' ';
    else if (!isHiddenChar(ch.charCodeAt(0))) out += ch;
  }
  return out;
}
