/**
 * Firecrawl: page text (`POST /v2/scrape`, Markdown, main content only). Firecrawl Cloud gets the key as a Bearer
 * token. Firecrawl on this Mac gets no Authorization header at all: it would pass a caller's key on to Firecrawl Cloud.
 */

import {
  type AdapterContext,
  FETCH_TIMEOUT_MS,
  type HttpReply,
  httpJson,
  type ProviderJson,
  providerJson,
  record,
} from '../http.js';
import { pageText } from '../normalize.js';
import { formatErrorText, type WebErrorCode } from '../result-text.js';

function scrape(url: string, ctx: AdapterContext, auth: Readonly<Record<string, string>>): Promise<HttpReply> {
  return httpJson({
    fetch: ctx.fetch,
    baseUrl: ctx.baseUrl,
    method: 'POST',
    path: '/v2/scrape',
    auth,
    body: { url, formats: ['markdown'], onlyMainContent: true, timeout: 25_000 },
    timeoutMs: FETCH_TIMEOUT_MS,
  });
}

function readPage(reply: ProviderJson, url: string, ctx: AdapterContext, local: boolean): string {
  if (!reply.ok) return formatErrorText(reply.code, 'firecrawl');
  const data = record(reply.json.data);
  if (data === null) return formatErrorText('provider-error', 'firecrawl');
  if (local && data.blocked === true) return formatErrorText('no-content', 'firecrawl');
  const metadata = record(data.metadata) ?? {};
  if (typeof metadata.statusCode === 'number' && metadata.statusCode >= 400) {
    return formatErrorText('no-content', 'firecrawl');
  }
  const cloudRead = local && data.source === 'firecrawl-cloud';
  return pageText('firecrawl', url, metadata.title, data.markdown, ctx.keys, cloudRead);
}

export async function firecrawlFetch(url: string, ctx: AdapterContext): Promise<string> {
  const reply = providerJson(await scrape(url, ctx, { Authorization: `Bearer ${ctx.key}` }));
  return readPage(reply, url, ctx, false);
}

/** 401: the server asks for a token, which the app never has. 403: the server refused the target (`BLOCKED_TARGET`). */
function localStatus(status: number): WebErrorCode | null {
  if (status === 401) return 'local-down';
  if (status === 403) return 'blocked';
  return null;
}

/** Firecrawl on this Mac at `ctx.baseUrl`; `ctx.key` is never read. */
export async function firecrawlLocalFetch(url: string, ctx: AdapterContext): Promise<string> {
  const http = await scrape(url, ctx, {});
  const reply: ProviderJson =
    http.kind === 'failed' && http.code === 'offline'
      ? { ok: false, code: 'local-down' }
      : providerJson(http, localStatus);
  return readPage(reply, url, ctx, true);
}
