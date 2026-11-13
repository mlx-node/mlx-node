/** Firecrawl: page text (`POST /v2/scrape`, Markdown, main content only), key as a Bearer token. */

import { type AdapterContext, FETCH_TIMEOUT_MS, httpJson, providerJson, record } from '../http.js';
import { pageText } from '../normalize.js';
import { formatErrorText } from '../result-text.js';

export async function firecrawlFetch(url: string, ctx: AdapterContext): Promise<string> {
  const reply = providerJson(
    await httpJson({
      fetch: ctx.fetch,
      baseUrl: ctx.baseUrl,
      method: 'POST',
      path: '/v2/scrape',
      auth: { Authorization: `Bearer ${ctx.key}` },
      body: { url, formats: ['markdown'], onlyMainContent: true, timeout: 25_000 },
      timeoutMs: FETCH_TIMEOUT_MS,
    }),
  );
  if (!reply.ok) return formatErrorText(reply.code, 'firecrawl');
  const data = record(reply.json.data);
  if (data === null) return formatErrorText('provider-error', 'firecrawl');
  const metadata = record(data.metadata) ?? {};
  if (typeof metadata.statusCode === 'number' && metadata.statusCode >= 400) {
    return formatErrorText('no-content', 'firecrawl');
  }
  return pageText('firecrawl', url, metadata.title, data.markdown, ctx.keys);
}
