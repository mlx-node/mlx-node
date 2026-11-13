/**
 * The text of a web tool result: the server writes it only through the `format…` functions and the host and the page
 * read it only through `parseWebResultText`, so writer and reader cannot drift. This module has no imports: the SPA
 * keeps a byte-equal copy.
 */

export type WebSearchProvider = 'exa' | 'deepseek' | 'brave';
export type WebFetchProvider = WebSearchProvider | 'firecrawl';

export const WEB_ERROR_CODES = [
  'not-configured',
  'key-rejected',
  'no-credit',
  'rate-limited',
  'timeout',
  'offline',
  'blocked',
  'no-content',
  'bad-request',
  'provider-error',
] as const;
export type WebErrorCode = (typeof WEB_ERROR_CODES)[number];

/** The fixed sentence the model reads after each error code. Provider bodies are never passed through. */
export const WEB_ERROR_SENTENCES: Readonly<Record<WebErrorCode, string>> = {
  'not-configured':
    'Web search is off in this app. Answer from what you know and tell the user to add a search provider in Settings.',
  'key-rejected':
    'The search provider rejected the API key. Tell the user to check the key in Settings. Do not call the tool again.',
  'no-credit': 'The search provider account has no credit left. Tell the user. Do not call the tool again.',
  'rate-limited': 'The search provider is busy. Do not call the tool again in this answer.',
  timeout: 'The request took too long. You may try once more with a shorter query or another address.',
  offline: 'The Mac seems to be offline. Tell the user. Do not call the tool again.',
  blocked: 'That address cannot be read. Use another address from the search results.',
  'no-content': 'That page has no readable text. Use another address or answer from the search results.',
  'bad-request': 'The tool input was wrong. Call it again with {"query": "<words>"} or {"url": "<https address>"}.',
  'provider-error': 'The search provider failed. Answer from what you know and say that the search failed.',
};

const SEARCH_PROVIDERS: readonly string[] = ['exa', 'deepseek', 'brave'];
const FETCH_PROVIDERS: readonly string[] = ['exa', 'deepseek', 'brave', 'firecrawl'];

export interface WebSearchItem {
  /** 1-based, no gap. */
  n: number;
  title: string;
  url: string;
  /** `YYYY-MM-DD`, or null when the result had no date line. */
  date: string | null;
  snippet: string;
}

export type WebResult =
  | { kind: 'search'; provider: WebSearchProvider; results: WebSearchItem[] }
  | { kind: 'page'; provider: WebFetchProvider; url: string; title: string; cut: boolean; text: string }
  | { kind: 'error'; code: WebErrorCode; provider: WebFetchProvider | 'none'; sentence: string };

export const PAGE_CUT_NOTE = 'note: cut at 12000 characters';

/** Line breaks and tabs become one space each, so a field can never add a line. */
function oneLine(text: string): string {
  let out = '';
  for (const ch of text) {
    const c = ch.charCodeAt(0);
    out += c === 10 || c === 13 || c === 9 || c === 0x85 || c === 0x2028 || c === 0x2029 ? ' ' : ch;
  }
  return out;
}

function nonEmpty(text: string, fallback: string): string {
  return text.trim() === '' ? fallback : text;
}

export function formatSearchText(
  provider: WebSearchProvider,
  results: ReadonlyArray<Omit<WebSearchItem, 'n'>>,
): string {
  const blocks = results.map((r, i) => {
    const lines = [`[${i + 1}] ${nonEmpty(oneLine(r.title), '(no title)')}`, oneLine(r.url)];
    if (r.date !== null && isDate(r.date)) lines.push(r.date);
    lines.push(nonEmpty(oneLine(r.snippet), '(no text)'));
    return lines.join('\n');
  });
  const head = `WEB_SEARCH provider=${provider} results=${results.length}`;
  return blocks.length === 0 ? head : `${head}\n${blocks.join('\n\n')}`;
}

export function formatPageText(
  provider: WebFetchProvider,
  page: { url: string; title: string; cut: boolean; text: string },
): string {
  const head = [`WEB_PAGE provider=${provider} url=${oneLine(page.url)}`, `title: ${oneLine(page.title)}`];
  if (page.cut) head.push(PAGE_CUT_NOTE);
  return `${head.join('\n')}\n\n${page.text}`;
}

export function formatErrorText(code: WebErrorCode, provider: WebFetchProvider | 'none'): string {
  return `WEB_ERROR ${code} ${provider}\n${WEB_ERROR_SENTENCES[code]}`;
}

function isDate(text: string): boolean {
  if (text.length !== 10 || text[4] !== '-' || text[7] !== '-') return false;
  for (const i of [0, 1, 2, 3, 5, 6, 8, 9]) {
    const c = text.charCodeAt(i);
    if (c < 48 || c > 57) return false;
  }
  return true;
}

function httpUrl(text: string): boolean {
  if (text === '' || text.trim() !== text) return false;
  try {
    const url = new URL(text);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
}

/** A positive decimal integer without a leading zero, else null. */
function count(text: string): number | null {
  if (text === '' || text.length > 4 || (text.length > 1 && text[0] === '0')) return null;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c < 48 || c > 57) return null;
  }
  return Number(text);
}

function after(line: string, prefix: string): string | null {
  return line.startsWith(prefix) ? line.slice(prefix.length) : null;
}

function parseSearch(provider: string, rest: string, body: string[]): WebResult | null {
  const n = count(rest);
  if (n === null || !SEARCH_PROVIDERS.includes(provider)) return null;
  if (n === 0)
    return body.length === 0 ? { kind: 'search', provider: provider as WebSearchProvider, results: [] } : null;
  const blocks: string[][] = [[]];
  for (const line of body) {
    if (line === '') blocks.push([]);
    else blocks[blocks.length - 1]!.push(line);
  }
  if (blocks.length !== n) return null;
  const results: WebSearchItem[] = [];
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i]!;
    if (b.length !== 3 && b.length !== 4) return null;
    const title = after(b[0]!, `[${i + 1}] `);
    if (title === null || !httpUrl(b[1]!)) return null;
    const date = b.length === 4 ? b[2]! : null;
    if (date !== null && !isDate(date)) return null;
    results.push({ n: i + 1, title, url: b[1]!, date, snippet: b[b.length - 1]! });
  }
  return { kind: 'search', provider: provider as WebSearchProvider, results };
}

/**
 * The whole text, or null. A search must hold exactly `results=N` blocks `[1]…[N]` of 3 or 4 lines each, separated
 * by exactly one blank line, each with an http(s) address; any mismatch refuses the whole text.
 */
export function parseWebResultText(text: string): WebResult | null {
  const nl = text.indexOf('\n');
  const first = nl === -1 ? text : text.slice(0, nl);
  const rest = nl === -1 ? '' : text.slice(nl + 1);
  const words = first.split(' ');
  if (words[0] === 'WEB_SEARCH' && words.length === 3) {
    const provider = after(words[1]!, 'provider=');
    const results = after(words[2]!, 'results=');
    if (provider === null || results === null) return null;
    return parseSearch(provider, results, nl === -1 ? [] : rest.split('\n'));
  }
  if (words[0] === 'WEB_PAGE' && words.length === 3) {
    const provider = after(words[1]!, 'provider=');
    const url = after(words[2]!, 'url=');
    if (provider === null || url === null || !FETCH_PROVIDERS.includes(provider) || !httpUrl(url)) return null;
    const lines = rest.split('\n');
    const title = after(lines[0] ?? '', 'title: ');
    if (title === null) return null;
    let at = 1;
    const cut = lines[at] === PAGE_CUT_NOTE;
    if (cut) at += 1;
    if (lines[at] !== '') return null;
    return {
      kind: 'page',
      provider: provider as WebFetchProvider,
      url,
      title,
      cut,
      text: lines.slice(at + 1).join('\n'),
    };
  }
  if (words[0] === 'WEB_ERROR' && words.length === 3) {
    const code = words[1]!;
    const provider = words[2]!;
    if (!(WEB_ERROR_CODES as readonly string[]).includes(code)) return null;
    if (provider !== 'none' && !FETCH_PROVIDERS.includes(provider)) return null;
    return {
      kind: 'error',
      code: code as WebErrorCode,
      provider: provider as WebFetchProvider | 'none',
      sentence: rest,
    };
  }
  return null;
}
