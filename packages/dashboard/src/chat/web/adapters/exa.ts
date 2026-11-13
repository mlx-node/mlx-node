/** Exa: search (`POST /search`) and page text (`POST /contents`), key in `x-api-key`. */

import {
  type AdapterContext,
  FETCH_TIMEOUT_MS,
  firstText,
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

export const RECENCY_DAYS: Readonly<Record<Recency, number>> = { day: 1, week: 7, month: 30, year: 365 };

export async function exaSearch(args: SearchArgs, ctx: AdapterContext): Promise<string> {
  const body: Record<string, unknown> = {
    query: args.query,
    type: 'auto',
    numResults: args.count,
    contents: { highlights: { numSentences: 3, highlightsPerUrl: 1 } },
  };
  if (args.recency !== null) {
    body.startPublishedDate = new Date(ctx.now() - RECENCY_DAYS[args.recency] * 86_400_000).toISOString();
  }
  const reply = providerJson(
    await httpJson({
      fetch: ctx.fetch,
      baseUrl: ctx.baseUrl,
      method: 'POST',
      path: '/search',
      auth: { 'x-api-key': ctx.key },
      body,
      timeoutMs: SEARCH_TIMEOUT_MS,
    }),
  );
  if (!reply.ok) return formatErrorText(reply.code, 'exa');
  if (!Array.isArray(reply.json.results)) return formatErrorText('provider-error', 'exa');
  const raw = list(reply.json.results).map((value) => {
    const r = record(value) ?? {};
    return {
      title: r.title,
      url: r.url,
      date: r.publishedDate,
      snippet: firstText(list(r.highlights)[0], r.summary, r.text),
    };
  });
  return searchText('exa', raw, args.count, ctx.keys);
}

export async function exaFetch(url: string, ctx: AdapterContext): Promise<string> {
  const reply = providerJson(
    await httpJson({
      fetch: ctx.fetch,
      baseUrl: ctx.baseUrl,
      method: 'POST',
      path: '/contents',
      auth: { 'x-api-key': ctx.key },
      body: { urls: [url], text: { maxCharacters: 12_000 } },
      timeoutMs: FETCH_TIMEOUT_MS,
    }),
  );
  if (!reply.ok) return formatErrorText(reply.code, 'exa');
  const first = record(list(reply.json.results)[0]);
  if (first === null) return formatErrorText('no-content', 'exa');
  return pageText('exa', url, first.title, first.text, ctx.keys);
}
