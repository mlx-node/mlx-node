/** Wire shapes of the web search provider settings: the SPA's copy of `src/chat/providers/types.ts`, pinned equal by type. */

export type SearchProviderId = 'exa' | 'deepseek' | 'brave';
export type ProviderId = SearchProviderId | 'firecrawl';
export type KeyCheck = 'ok' | 'unverified' | 'rejected' | 'no-credit';

/** What a key check reported, in memory only: numbers and 3-letter currency codes, never provider text. */
export interface ProviderFacts {
  remainingCredits?: number;
  planCredits?: number;
  /** Epoch ms. */
  periodEnd?: number;
  balances?: Array<{ currency: string; amount: number }>;
}

export interface ProviderState {
  id: ProviderId;
  role: 'search' | 'crawl';
  /** A key is in the Keychain (from the key cache). */
  configured: boolean;
  check: KeyCheck | null;
  /** Epoch ms. */
  checkedAt: number | null;
  selected: boolean;
  /** Variable names used in Work. */
  keyEnv: readonly string[];
  skill: {
    installed: boolean;
    names: readonly string[];
    upstream: { repo: string; commit: string; license: string } | null;
  };
  /** Whether this provider's skills are usable in Work. */
  work: { ready: boolean; missing: 'firecrawl-cli' | null };
  facts: ProviderFacts | null;
}

/**
 * The last check of Firecrawl on this Mac. `ok`: Firecrawl Local, refusing private targets, its Cloud setting off.
 * `not-running`: no answer. `not-firecrawl`: something else answers. `open`: it reads private addresses. `token`: it
 * asks for a token. `cloud-on`: it may send a page to Firecrawl Cloud, or its Cloud setting could not be read.
 */
export type LocalCheck = 'ok' | 'not-running' | 'not-firecrawl' | 'open' | 'token' | 'cloud-on';

/** `off`: no page leaves this Mac. `fallback`: a page the server cannot load is sent to Firecrawl Cloud. */
export type LocalCloud = 'off' | 'fallback';

/** Firecrawl on this Mac: the saved address and the last check of it, in memory only. */
export interface LocalFirecrawl {
  /** The saved address, `http://127.0.0.1:<port>` or `http://[::1]:<port>`. */
  url: string | null;
  /** Null before the first check of this address ends. */
  check: LocalCheck | null;
  /** Epoch ms. */
  checkedAt: number | null;
  /** The server's version, digits and dots only. */
  version: string | null;
  /** The server's Cloud setting: `off` with `ok`; `fallback` or null (not readable) with `cloud-on`; else null. */
  cloud: LocalCloud | null;
}

export interface ProvidersResponse {
  /** Exactly one or none. */
  search: SearchProviderId | null;
  /** Independent of `search`. */
  firecrawl: boolean;
  /** Firecrawl is on and reads pages on this Mac, with no key. */
  firecrawlLocal: boolean;
  /** What `web_fetch` uses now. */
  fetch: ProviderId | null;
  /** Always four, fixed order exa, deepseek, brave, firecrawl. */
  providers: ProviderState[];
  local: LocalFirecrawl;
  chatSkill: { installed: boolean };
  keychain: 'ok' | 'unavailable';
  /** A host still runs on the old selection. */
  applyPending: boolean;
  /** The provider in use whose key was not in the Keychain at app start: in replies until the first list. */
  vanished: ProviderId | null;
}

/** `PUT /api/providers/:id/key`. A `rejected` key is not saved and any earlier key stays. */
export interface ProviderKeyReply {
  saved: boolean;
  check: KeyCheck;
  providers: ProvidersResponse;
}

/** `POST /api/providers/:id/validate`. `check` is null when the Keychain no longer holds a key for the provider. */
export interface ProviderCheckReply {
  check: KeyCheck | null;
  providers: ProvidersResponse;
}

/** `PUT /api/providers/selection`: each field optional, at least one. */
export interface ProviderSelectionInput {
  search?: SearchProviderId | null;
  firecrawl?: boolean;
  /** Where Firecrawl reads pages: on this Mac (true) or in Firecrawl Cloud (false). */
  firecrawlLocal?: boolean;
}

/**
 * `PUT /api/providers/local`. An address whose check is not `ok` is not saved, and any earlier address stays. Nor is
 * one whose check passed after an Off or a Remove of the crawler landed during it (`saved: false`, `check: 'ok'`).
 */
export interface ProviderLocalReply {
  saved: boolean;
  check: LocalCheck;
  providers: ProvidersResponse;
}

/** `POST /api/providers/local/check`. `check` is null when no address is saved. */
export interface ProviderLocalCheckReply {
  check: LocalCheck | null;
  providers: ProvidersResponse;
}
