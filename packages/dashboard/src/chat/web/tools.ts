/**
 * The two tools of the `mlx_web` server. Arguments are read leniently where that is safe (small models send sloppy
 * input); every result is one text written by the `result-text` writers, and every failure is a `WEB_ERROR` code:
 * nothing is rethrown and nothing reaches stderr. A key reaches this process only through its environment and leaves
 * it only in the provider's auth header.
 */

import { braveFetch, braveSearch } from './adapters/brave.js';
import { deepseekFetch, deepseekSearch } from './adapters/deepseek.js';
import { exaFetch, exaSearch } from './adapters/exa.js';
import { firecrawlFetch } from './adapters/firecrawl.js';
import { type AdapterContext, PROVIDER_ORIGINS, type SearchArgs } from './http.js';
import { redactKeys } from './normalize.js';
import { formatErrorText, type WebFetchProvider, type WebSearchProvider } from './result-text.js';
import { fetchAllowed, forwardedAddress } from './url-policy.js';

export type { Recency, SearchArgs } from './http.js';

export const TOOL_DEFINITIONS = [
  {
    name: 'web_search',
    description:
      'Search the web. Returns numbered results: title, address, date, short text. Use it for recent or uncertain facts.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      required: ['query'],
      properties: {
        query: { type: 'string', minLength: 1, maxLength: 200, description: '2 to 6 search words' },
        count: { type: 'integer', minimum: 1, maximum: 8, description: 'How many results. Default 5' },
        recency: {
          type: 'string',
          enum: ['day', 'week', 'month', 'year'],
          description: 'Only results newer than this',
        },
      },
    },
  },
  {
    name: 'web_fetch',
    description: 'Read one web page. Returns its main text as Markdown, cut to a fixed length.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      required: ['url'],
      properties: { url: { type: 'string', maxLength: 2048, description: 'Full https address' } },
    },
  },
] as const;

export const MAX_CALLS_IN_FLIGHT = 4;
export const MAX_QUERY_CHARS = 200;
export const MAX_URL_CHARS = 2048;
export const DEFAULT_COUNT = 5;
export const MAX_COUNT = 8;

export interface WebConfig {
  searchProvider: WebSearchProvider | 'none';
  searchKey: string;
  fetchProvider: WebFetchProvider | 'none';
  fetchKey: string;
}

const SEARCH_IDS: readonly string[] = ['exa', 'deepseek', 'brave'];
const FETCH_IDS: readonly string[] = ['exa', 'deepseek', 'brave', 'firecrawl'];

/** The four `MLX_WEB_*` variables; an unknown provider reads as `none`. */
export function readWebConfig(env: NodeJS.ProcessEnv): WebConfig {
  const search = env.MLX_WEB_SEARCH_PROVIDER ?? '';
  const fetch = env.MLX_WEB_FETCH_PROVIDER ?? '';
  return {
    searchProvider: SEARCH_IDS.includes(search) ? (search as WebSearchProvider) : 'none',
    searchKey: env.MLX_WEB_SEARCH_KEY ?? '',
    fetchProvider: FETCH_IDS.includes(fetch) ? (fetch as WebFetchProvider) : 'none',
    fetchKey: env.MLX_WEB_FETCH_KEY ?? '',
  };
}

function endsWith(text: string, suffix: string): boolean {
  return text.length >= suffix.length && text.slice(text.length - suffix.length) === suffix;
}

/** Deletes the `MLX_WEB_*` variables, `XAI_API_KEY`, and every name that ends in `_API_KEY` or `_TOKEN`. */
export function scrubEnv(env: NodeJS.ProcessEnv): void {
  for (const name of Object.keys(env)) {
    if (
      name.startsWith('MLX_WEB_') ||
      name === 'XAI_API_KEY' ||
      endsWith(name, '_API_KEY') ||
      endsWith(name, '_TOKEN')
    ) {
      delete env[name];
    }
  }
}

function record(args: unknown): Record<string, unknown> | null {
  return typeof args === 'object' && args !== null && !Array.isArray(args) ? (args as Record<string, unknown>) : null;
}

function digitsOnly(text: string): boolean {
  if (text === '' || text.length > 6) return false;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c < 48 || c > 57) return false;
  }
  return true;
}

function clampCount(value: unknown): number {
  let n: number | null = null;
  if (typeof value === 'number' && Number.isFinite(value)) n = Math.trunc(value);
  else if (typeof value === 'string' && digitsOnly(value.trim())) n = Number(value.trim());
  if (n === null) return DEFAULT_COUNT;
  return Math.min(MAX_COUNT, Math.max(1, n));
}

/** Search arguments, or null for a missing or empty query. Unknown properties are ignored. */
export function searchArgs(args: unknown): SearchArgs | null {
  const r = record(args);
  if (r === null || typeof r.query !== 'string') return null;
  const query = Array.from(r.query.trim()).slice(0, MAX_QUERY_CHARS).join('');
  if (query === '') return null;
  const recency =
    r.recency === 'day' || r.recency === 'week' || r.recency === 'month' || r.recency === 'year' ? r.recency : null;
  return { query, count: clampCount(r.count), recency };
}

/** The address to read, without its fragment, or null for a missing, empty or over-long one. */
export function fetchArgs(args: unknown): { url: string } | null {
  const r = record(args);
  if (r === null || typeof r.url !== 'string') return null;
  const url = forwardedAddress(r.url);
  if (url === '' || url.length > MAX_URL_CHARS) return null;
  return { url };
}

/** Every occurrence of a key is replaced: a provider may echo it. */
export function scrubKeys(text: string, keys: readonly string[]): string {
  return redactKeys(text, keys);
}

/** Where the provider calls go. Production uses the fixed provider origins; tests pass loopback fakes. */
export interface WebDeps {
  fetch: typeof fetch;
  baseUrls: Readonly<Record<WebFetchProvider, string>>;
  now: () => number;
}

export const PRODUCTION_DEPS: WebDeps = {
  fetch: (input, init) => fetch(input, init),
  baseUrls: PROVIDER_ORIGINS,
  now: () => Date.now(),
};

const SEARCH: Readonly<Record<WebSearchProvider, (args: SearchArgs, ctx: AdapterContext) => Promise<string>>> = {
  exa: exaSearch,
  deepseek: deepseekSearch,
  brave: braveSearch,
};

const FETCH: Readonly<Record<WebFetchProvider, (url: string, ctx: AdapterContext) => Promise<string>>> = {
  exa: exaFetch,
  deepseek: deepseekFetch,
  brave: braveFetch,
  firecrawl: firecrawlFetch,
};

export class WebTools {
  private inFlight = 0;
  private readonly config: WebConfig;
  private readonly deps: WebDeps;

  constructor(config: WebConfig, deps: WebDeps = PRODUCTION_DEPS) {
    this.config = config;
    this.deps = deps;
  }

  definitions(): readonly unknown[] {
    return TOOL_DEFINITIONS;
  }

  async call(name: string, args: unknown): Promise<string> {
    if (name !== 'web_search' && name !== 'web_fetch') return formatErrorText('bad-request', 'none');
    if (name === 'web_search') {
      const provider = this.config.searchProvider;
      const parsed = searchArgs(args);
      if (parsed === null) return formatErrorText('bad-request', provider);
      if (provider === 'none' || this.config.searchKey === '') return notConfigured();
      const ctx = this.context(provider, this.config.searchKey);
      return this.run(provider, () => SEARCH[provider](parsed, ctx));
    }
    const provider = this.config.fetchProvider;
    const parsed = fetchArgs(args);
    if (parsed === null) return formatErrorText('bad-request', provider);
    if (provider === 'none' || this.config.fetchKey === '') return notConfigured();
    if (!fetchAllowed(parsed.url)) return formatErrorText('blocked', provider);
    const ctx = this.context(provider, this.config.fetchKey);
    return this.run(provider, () => FETCH[provider](parsed.url, ctx));
  }

  private context(provider: WebFetchProvider, key: string): AdapterContext {
    return {
      baseUrl: this.deps.baseUrls[provider],
      key,
      fetch: this.deps.fetch,
      now: this.deps.now,
      keys: [this.config.searchKey, this.config.fetchKey],
    };
  }

  /** One provider call: at most {@link MAX_CALLS_IN_FLIGHT} at a time, any throw a `provider-error`. */
  private async run(provider: WebFetchProvider, request: () => Promise<string>): Promise<string> {
    if (this.inFlight >= MAX_CALLS_IN_FLIGHT) return formatErrorText('rate-limited', provider);
    this.inFlight += 1;
    try {
      return scrubKeys(await request(), [this.config.searchKey, this.config.fetchKey]);
    } catch {
      return formatErrorText('provider-error', provider);
    } finally {
      this.inFlight -= 1;
    }
  }
}

function notConfigured(): string {
  return formatErrorText('not-configured', 'none');
}
