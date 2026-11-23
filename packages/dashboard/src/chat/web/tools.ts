/**
 * The two tools of the `mlx_web` server. Arguments are read leniently where that is safe (small models send sloppy
 * input); every result is one text written by the `result-text` writers, and every failure is a `WEB_ERROR` code:
 * nothing is rethrown and nothing reaches stderr. A key reaches this process only through its environment and leaves
 * it only in the provider's auth header.
 */

import { braveFetch, braveSearch } from './adapters/brave.js';
import { deepseekFetch, deepseekSearch } from './adapters/deepseek.js';
import { exaFetch, exaSearch } from './adapters/exa.js';
import { firecrawlFetch, firecrawlLocalFetch } from './adapters/firecrawl.js';
import { type AdapterContext, PROVIDER_ORIGINS, type SearchArgs } from './http.js';
import { checkLocal, type LocalCheck, localOrigin } from './local.js';
import { redactKeys } from './normalize.js';
import { formatErrorText, type WebErrorCode, type WebFetchProvider, type WebSearchProvider } from './result-text.js';
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
  /** The saved address of Firecrawl on this Mac, or '' for the provider's own servers. */
  fetchUrl: string;
}

const SEARCH_IDS: readonly string[] = ['exa', 'deepseek', 'brave'];
const FETCH_IDS: readonly string[] = ['exa', 'deepseek', 'brave', 'firecrawl'];

/**
 * The five `MLX_WEB_*` variables; an unknown provider reads as `none`. `MLX_WEB_FETCH_URL` is kept only for Firecrawl
 * and only in the exact saved form of a loopback address; whenever it is set the fetch key reads '', so a page read
 * meant for this Mac never falls back to a provider with a key.
 */
export function readWebConfig(env: NodeJS.ProcessEnv): WebConfig {
  const search = env.MLX_WEB_SEARCH_PROVIDER ?? '';
  const fetch = env.MLX_WEB_FETCH_PROVIDER ?? '';
  const fetchProvider = FETCH_IDS.includes(fetch) ? (fetch as WebFetchProvider) : 'none';
  const rawUrl = env.MLX_WEB_FETCH_URL ?? '';
  return {
    searchProvider: SEARCH_IDS.includes(search) ? (search as WebSearchProvider) : 'none',
    searchKey: env.MLX_WEB_SEARCH_KEY ?? '',
    fetchProvider,
    fetchKey: rawUrl === '' ? (env.MLX_WEB_FETCH_KEY ?? '') : '',
    fetchUrl: fetchProvider === 'firecrawl' ? (localOrigin(rawUrl) ?? '') : '',
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

/**
 * Where the provider calls go. Production uses the fixed provider origins; tests pass loopback fakes. Firecrawl on
 * this Mac is the one exception: its base is `WebConfig.fetchUrl`, never this table.
 */
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
  /** Firecrawl on this Mac, or null. */
  private readonly localBase: string | null;

  constructor(config: WebConfig, deps: WebDeps = PRODUCTION_DEPS) {
    this.config = config;
    this.deps = deps;
    this.localBase = config.fetchProvider === 'firecrawl' ? localOrigin(config.fetchUrl) : null;
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
    const local = this.localBase;
    if (provider === 'none') return notConfigured();
    // A fetch URL that is not Firecrawl on this Mac never falls back to a provider's own servers.
    if (local === null && (this.config.fetchUrl !== '' || this.config.fetchKey === '')) return notConfigured();
    if (!fetchAllowed(parsed.url)) return formatErrorText('blocked', provider);
    if (local !== null) {
      const ctx: AdapterContext = { ...this.context(provider, ''), baseUrl: local };
      return this.run(provider, async () => {
        // Before every read: the server could have been restarted with --allow-private or its Cloud on since the last.
        const { check } = await checkLocal(local, { fetch: this.deps.fetch });
        const refused = localRefusal(check);
        if (refused !== null) return formatErrorText(refused, 'firecrawl');
        return firecrawlLocalFetch(parsed.url, ctx);
      });
    }
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

/** The error a page read on this Mac answers after the check, or null when the page address may be sent. */
function localRefusal(check: LocalCheck): WebErrorCode | null {
  switch (check) {
    case 'ok':
      return null;
    case 'cloud-on':
      return 'local-cloud';
    case 'not-running':
    case 'not-firecrawl':
    case 'open':
    case 'token':
      return 'local-down';
  }
}

function notConfigured(): string {
  return formatErrorText('not-configured', 'none');
}
