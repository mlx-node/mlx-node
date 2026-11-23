/**
 * Outbound HTTPS to the four provider APIs, for the web tools and for key validation. A request goes only to the
 * origin it was built on, and that origin is one of the four provider hosts over `https` on the default port (or a
 * loopback `http` origin a test or the dev harness passed in). No redirect is followed, the body read stops at 2 MiB,
 * and every failure is a code: no error text, header or body of a provider is ever passed on.
 */

import type { WebErrorCode, WebFetchProvider } from './result-text.js';

export const PROVIDER_ORIGINS: Readonly<Record<WebFetchProvider, string>> = {
  exa: 'https://api.exa.ai',
  deepseek: 'https://api.deepseek.com',
  brave: 'https://api.search.brave.com',
  firecrawl: 'https://api.firecrawl.dev',
};

const PROVIDER_HOSTS: readonly string[] = [
  'api.exa.ai',
  'api.deepseek.com',
  'api.search.brave.com',
  'api.firecrawl.dev',
];

export const SEARCH_TIMEOUT_MS = 15_000;
export const FETCH_TIMEOUT_MS = 30_000;
export const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;
export const USER_AGENT = 'mlx-node';

export type Recency = 'day' | 'week' | 'month' | 'year';

export interface SearchArgs {
  query: string;
  count: number;
  recency: Recency | null;
}

/** What one provider call needs. `keys` are redacted from every text the call returns. */
export interface AdapterContext {
  baseUrl: string;
  key: string;
  fetch: typeof fetch;
  now: () => number;
  keys: readonly string[];
}

/** `http://127.0.0.1:<port>`, `http://localhost:<port>` or `http://[::1]:<port>`, with nothing after the origin. */
export function isLoopbackHttpOrigin(text: string): boolean {
  let url: URL;
  try {
    url = new URL(text);
  } catch {
    return false;
  }
  if (url.protocol !== 'http:' || url.username !== '' || url.password !== '') return false;
  if (url.hostname !== '127.0.0.1' && url.hostname !== 'localhost' && url.hostname !== '[::1]') return false;
  if (url.search !== '' || url.hash !== '' || url.pathname !== '/') return false;
  return text === url.origin || text === `${url.origin}/`;
}

/** The final URL is on `base`'s origin, and that origin is a provider host over https or a loopback test origin. */
export function targetAllowed(url: URL, base: string): boolean {
  let baseUrl: URL;
  try {
    baseUrl = new URL(base);
  } catch {
    return false;
  }
  if (url.origin !== baseUrl.origin || url.username !== '' || url.password !== '') return false;
  if (url.protocol === 'https:') return url.port === '' && PROVIDER_HOSTS.includes(url.hostname);
  return isLoopbackHttpOrigin(base);
}

export interface HttpRequest {
  fetch: typeof fetch;
  baseUrl: string;
  method: 'GET' | 'POST';
  path: string;
  query?: ReadonlyArray<readonly [string, string]>;
  /** The provider's auth header, and nothing else. */
  auth: Readonly<Record<string, string>>;
  body?: unknown;
  timeoutMs: number;
}

export type HttpReply =
  | { kind: 'response'; status: number; json: unknown; parsed: boolean }
  | { kind: 'failed'; code: 'timeout' | 'offline' | 'provider-error' };

async function readCapped(response: Response, limit: number): Promise<Buffer | null> {
  const body = response.body;
  if (body === null) return Buffer.alloc(0);
  const reader = body.getReader();
  const parts: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > limit) {
      await reader.cancel().catch(() => {});
      return null;
    }
    parts.push(value);
  }
  return Buffer.concat(parts, size);
}

/** One JSON request to a provider. Never throws. */
export async function httpJson(request: HttpRequest): Promise<HttpReply> {
  let url: URL;
  try {
    url = new URL(request.path, request.baseUrl);
    for (const [name, value] of request.query ?? []) url.searchParams.append(name, value);
  } catch {
    return { kind: 'failed', code: 'provider-error' };
  }
  if (!targetAllowed(url, request.baseUrl)) return { kind: 'failed', code: 'provider-error' };
  const headers: Record<string, string> = { ...request.auth, Accept: 'application/json', 'User-Agent': USER_AGENT };
  if (request.body !== undefined) headers['Content-Type'] = 'application/json';
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, request.timeoutMs);
  try {
    let response: Response;
    try {
      response = await request.fetch(url, {
        method: request.method,
        headers,
        body: request.body === undefined ? undefined : JSON.stringify(request.body),
        redirect: 'manual',
        signal: controller.signal,
      });
    } catch {
      return { kind: 'failed', code: timedOut ? 'timeout' : 'offline' };
    }
    if (response.type === 'opaqueredirect' || (response.status >= 300 && response.status < 400)) {
      controller.abort();
      return { kind: 'failed', code: 'provider-error' };
    }
    let data: Buffer | null;
    try {
      data = await readCapped(response, MAX_RESPONSE_BYTES);
    } catch {
      return { kind: 'failed', code: timedOut ? 'timeout' : 'offline' };
    }
    if (data === null) {
      controller.abort();
      return { kind: 'failed', code: 'provider-error' };
    }
    try {
      return { kind: 'response', status: response.status, json: JSON.parse(data.toString('utf8')), parsed: true };
    } catch {
      return { kind: 'response', status: response.status, json: undefined, parsed: false };
    }
  } finally {
    clearTimeout(timer);
  }
}

/** The web tool code of a status that is not 2xx. */
export function statusCode(status: number): WebErrorCode {
  if (status === 401 || status === 403) return 'key-rejected';
  if (status === 402) return 'no-credit';
  if (status === 429) return 'rate-limited';
  if (status === 408 || status === 504) return 'timeout';
  if (status === 400 || status === 422) return 'bad-request';
  return 'provider-error';
}

export type ProviderJson = { ok: true; json: Record<string, unknown> } | { ok: false; code: WebErrorCode };

/**
 * A reply as the JSON object of a 2xx answer, or the code of the failure. `override` maps a provider's own status
 * meanings (DeepSeek fetch: 403 blocked, 404 no content) before the common table.
 */
export function providerJson(reply: HttpReply, override?: (status: number) => WebErrorCode | null): ProviderJson {
  if (reply.kind === 'failed') return { ok: false, code: reply.code };
  if (reply.status < 200 || reply.status > 299) {
    return { ok: false, code: override?.(reply.status) ?? statusCode(reply.status) };
  }
  const json = record(reply.json);
  if (!reply.parsed || json === null) return { ok: false, code: 'provider-error' };
  return { ok: true, json };
}

export function record(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

/** The first argument that is a non-blank string, else ''. */
export function firstText(...values: unknown[]): string {
  for (const value of values) if (typeof value === 'string' && value.trim() !== '') return value;
  return '';
}
