/** Wire shapes of the web search provider settings. The SPA keeps a copy (`ui/src/lib/providers-types.ts`), pinned equal by type. */

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

export interface ProvidersResponse {
  /** Exactly one or none. */
  search: SearchProviderId | null;
  /** Independent of `search`. */
  firecrawl: boolean;
  /** What `web_fetch` uses now. */
  fetch: ProviderId | null;
  /** Always four, fixed order exa, deepseek, brave, firecrawl. */
  providers: ProviderState[];
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
}
