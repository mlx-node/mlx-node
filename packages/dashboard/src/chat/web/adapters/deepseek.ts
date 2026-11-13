/** DeepSeek: search (`POST /v1/tools/search`) and page text (`POST /v1/tools/fetch`), key as a Bearer token. */

import {
  type AdapterContext,
  FETCH_TIMEOUT_MS,
  firstText,
  httpJson,
  list,
  providerJson,
  record,
  SEARCH_TIMEOUT_MS,
  type SearchArgs,
} from '../http.js';
import { capChars, dateBefore, pageText, searchText } from '../normalize.js';
import { formatErrorText } from '../result-text.js';
import { RECENCY_DAYS } from './exa.js';

/** The API takes at most 128 characters of query, `after:` included. */
export const DEEPSEEK_QUERY_CHARS = 128;

export function deepseekQuery(args: SearchArgs, now: number): string {
  const suffix = args.recency === null ? '' : ` after:${dateBefore(now, RECENCY_DAYS[args.recency])}`;
  return `${capChars(args.query, DEEPSEEK_QUERY_CHARS - suffix.length).trimEnd()}${suffix}`;
}

export async function deepseekSearch(args: SearchArgs, ctx: AdapterContext): Promise<string> {
  const reply = providerJson(
    await httpJson({
      fetch: ctx.fetch,
      baseUrl: ctx.baseUrl,
      method: 'POST',
      path: '/v1/tools/search',
      auth: { Authorization: `Bearer ${ctx.key}` },
      body: { query: deepseekQuery(args, ctx.now()), count: args.count },
      timeoutMs: SEARCH_TIMEOUT_MS,
    }),
  );
  if (!reply.ok) return formatErrorText(reply.code, 'deepseek');
  if (!Array.isArray(reply.json.results)) return formatErrorText('provider-error', 'deepseek');
  const raw = list(reply.json.results).map((value) => {
    const r = record(value) ?? {};
    return { title: r.title, url: r.url, date: r.date, snippet: firstText(r.summary, r.snippet) };
  });
  return searchText('deepseek', raw, args.count, ctx.keys);
}

export async function deepseekFetch(url: string, ctx: AdapterContext): Promise<string> {
  const reply = providerJson(
    await httpJson({
      fetch: ctx.fetch,
      baseUrl: ctx.baseUrl,
      method: 'POST',
      path: '/v1/tools/fetch',
      auth: { Authorization: `Bearer ${ctx.key}` },
      body: { url },
      timeoutMs: FETCH_TIMEOUT_MS,
    }),
    (status) => (status === 403 ? 'blocked' : status === 404 ? 'no-content' : null),
  );
  if (!reply.ok) return formatErrorText(reply.code, 'deepseek');
  return pageText('deepseek', url, reply.json.title, reply.json.content, ctx.keys);
}
