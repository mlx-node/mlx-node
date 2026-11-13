/**
 * Brave: search (`GET /res/v1/web/search`) and page excerpts (`GET /res/v1/llm/context` with the address as the
 * query), key in `X-Subscription-Token`. Brave has no page reader: an address it returns no excerpt for has no text.
 */

import {
  type AdapterContext,
  FETCH_TIMEOUT_MS,
  httpJson,
  list,
  providerJson,
  record,
  type Recency,
  SEARCH_TIMEOUT_MS,
  type SearchArgs,
} from '../http.js';
import { pageText, searchText } from '../normalize.js';
import { formatErrorText } from '../result-text.js';

const FRESHNESS: Readonly<Record<Recency, string>> = { day: 'pd', week: 'pw', month: 'pm', year: 'py' };

const ENTITIES: ReadonlyArray<readonly [string, string]> = [
  ['&amp;', '&'],
  ['&lt;', '<'],
  ['&gt;', '>'],
  ['&quot;', '"'],
  ['&#39;', "'"],
];

/** Tags removed (`<` up to the next `>`), then the five basic entities decoded. */
export function stripTags(text: string): string {
  let out = '';
  let i = 0;
  let closed = true;
  while (i < text.length) {
    if (text[i] === '<' && closed) {
      const end = text.indexOf('>', i + 1);
      // No `>` after this one means none after any later `<` either: searching again would be quadratic.
      closed = end !== -1;
      if (end !== -1) {
        i = end + 1;
        continue;
      }
    }
    if (text[i] === '&') {
      const entity = ENTITIES.find(([name]) => text.startsWith(name, i));
      if (entity !== undefined) {
        out += entity[1];
        i += entity[0].length;
        continue;
      }
    }
    out += text[i];
    i += 1;
  }
  return out;
}

function text(value: unknown): string {
  return typeof value === 'string' ? stripTags(value) : '';
}

export async function braveSearch(args: SearchArgs, ctx: AdapterContext): Promise<string> {
  const query: Array<[string, string]> = [
    ['q', args.query],
    ['count', String(args.count)],
  ];
  if (args.recency !== null) query.push(['freshness', FRESHNESS[args.recency]]);
  const reply = providerJson(
    await httpJson({
      fetch: ctx.fetch,
      baseUrl: ctx.baseUrl,
      method: 'GET',
      path: '/res/v1/web/search',
      query,
      auth: { 'X-Subscription-Token': ctx.key },
      timeoutMs: SEARCH_TIMEOUT_MS,
    }),
  );
  if (!reply.ok) return formatErrorText(reply.code, 'brave');
  const web = reply.json.web;
  if (web !== undefined && record(web) === null) return formatErrorText('provider-error', 'brave');
  const raw = list(record(web)?.results).map((value) => {
    const r = record(value) ?? {};
    return {
      title: text(r.title),
      url: r.url,
      date: typeof r.page_age === 'string' ? r.page_age : r.age,
      snippet: text(r.description),
    };
  });
  return searchText('brave', raw, args.count, ctx.keys);
}

/** Scheme and host lower-cased, fragment and one trailing `/` dropped. */
export function sameAddress(a: string, b: string): boolean {
  const normal = (value: string): string | null => {
    try {
      const url = new URL(value);
      let path = url.pathname;
      if (path.endsWith('/')) path = path.slice(0, -1);
      return `${url.protocol}//${url.host.toLowerCase()}${path}${url.search}`;
    } catch {
      return null;
    }
  };
  const left = normal(a);
  return left !== null && left === normal(b);
}

export async function braveFetch(url: string, ctx: AdapterContext): Promise<string> {
  const reply = providerJson(
    await httpJson({
      fetch: ctx.fetch,
      baseUrl: ctx.baseUrl,
      method: 'GET',
      path: '/res/v1/llm/context',
      query: [
        ['q', url],
        ['count', '5'],
        ['maximum_number_of_tokens', '4096'],
      ],
      auth: { 'X-Subscription-Token': ctx.key },
      timeoutMs: FETCH_TIMEOUT_MS,
    }),
  );
  if (!reply.ok) return formatErrorText(reply.code, 'brave');
  const entries = list(record(reply.json.grounding)?.generic).map((value) => record(value) ?? {});
  const entry = entries.find((e) => typeof e.url === 'string' && sameAddress(e.url, url));
  if (entry === undefined) return formatErrorText('no-content', 'brave');
  const snippets = list(entry.snippets).filter((s): s is string => typeof s === 'string');
  return pageText('brave', url, text(entry.title), snippets.join('\n\n'), ctx.keys);
}
