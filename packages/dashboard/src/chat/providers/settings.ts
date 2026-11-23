/**
 * `ROOT/providers.json` (0600): which search provider is selected, whether Firecrawl reads pages and where (Firecrawl
 * Cloud, or Firecrawl on this Mac at a saved loopback address), and the last check of each key. Enums, times and that
 * address only: no key and no part of one. A missing or unreadable file reads as the defaults.
 */

import { chmod, mkdir, readFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { writeFileAtomic } from '../paths.js';
import { localOrigin } from '../web/local.js';
import type { KeyCheck, ProviderId, SearchProviderId } from './types.js';

export const PROVIDER_IDS: readonly ProviderId[] = ['exa', 'deepseek', 'brave', 'firecrawl'];
export const SEARCH_PROVIDER_IDS: readonly SearchProviderId[] = ['exa', 'deepseek', 'brave'];
const KEY_CHECKS: readonly string[] = ['ok', 'unverified', 'rejected', 'no-credit'];

export interface ProviderSettings {
  version: 1;
  search: SearchProviderId | null;
  firecrawl: boolean;
  /** Firecrawl reads pages on this Mac at `localUrl`, not in Firecrawl Cloud. Only with a `localUrl`. */
  firecrawlLocal: boolean;
  /** The address of Firecrawl on this Mac, only in its saved form (see `parseLocalAddress`). */
  localUrl: string | null;
  checks: Partial<Record<ProviderId, { status: KeyCheck; at: number }>>;
}

export function defaultSettings(): ProviderSettings {
  return { version: 1, search: null, firecrawl: false, firecrawlLocal: false, localUrl: null, checks: {} };
}

export function isProviderId(value: unknown): value is ProviderId {
  return typeof value === 'string' && (PROVIDER_IDS as readonly string[]).includes(value);
}

export function isSearchProviderId(value: unknown): value is SearchProviderId {
  return typeof value === 'string' && (SEARCH_PROVIDER_IDS as readonly string[]).includes(value);
}

/** Only the known fields with known values survive; anything else is dropped. */
export function parseSettings(text: string): ProviderSettings {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return defaultSettings();
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return defaultSettings();
  const r = raw as Record<string, unknown>;
  if (r.version !== 1) return defaultSettings();
  const settings = defaultSettings();
  if (isSearchProviderId(r.search)) settings.search = r.search;
  settings.firecrawl = r.firecrawl === true;
  settings.localUrl = typeof r.localUrl === 'string' ? localOrigin(r.localUrl) : null;
  settings.firecrawlLocal = settings.localUrl !== null && r.firecrawlLocal === true;
  // A pick of this Mac whose address is dropped turns the crawler off; it never becomes a pick of the cloud.
  if (r.firecrawlLocal === true && settings.localUrl === null) settings.firecrawl = false;
  const checks = r.checks;
  if (typeof checks === 'object' && checks !== null && !Array.isArray(checks)) {
    for (const id of PROVIDER_IDS) {
      const check = (checks as Record<string, unknown>)[id];
      if (typeof check !== 'object' || check === null) continue;
      const { status, at } = check as { status?: unknown; at?: unknown };
      if (typeof status === 'string' && KEY_CHECKS.includes(status) && typeof at === 'number' && Number.isFinite(at)) {
        settings.checks[id] = { status: status as KeyCheck, at };
      }
    }
  }
  return settings;
}

export async function readSettings(path: string): Promise<ProviderSettings> {
  try {
    return parseSettings(await readFile(path, 'utf8'));
  } catch {
    return defaultSettings();
  }
}

export async function writeSettings(path: string, settings: ProviderSettings): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  await chmod(dirname(path), 0o700);
  const checks: ProviderSettings['checks'] = {};
  for (const id of PROVIDER_IDS) {
    const check = settings.checks[id];
    if (check !== undefined) checks[id] = { status: check.status, at: check.at };
  }
  const clean: ProviderSettings = {
    version: 1,
    search: settings.search,
    firecrawl: settings.firecrawl,
    firecrawlLocal: settings.firecrawlLocal,
    localUrl: settings.localUrl,
    checks,
  };
  await writeFileAtomic(path, `${JSON.stringify(clean)}\n`);
}

/**
 * The invariants: a selected search provider has a key; Firecrawl reads pages in the cloud only with a key, and on this
 * Mac only with a saved address (it never moves to the other place). `configured` answers null for a provider whose
 * key is not known (the Keychain did not answer): such a choice is kept.
 */
export function repairSettings(settings: ProviderSettings, configured: (id: ProviderId) => boolean | null): boolean {
  let changed = false;
  if (settings.search !== null && configured(settings.search) === false) {
    settings.search = null;
    changed = true;
  }
  if (settings.firecrawlLocal && settings.localUrl === null) {
    settings.firecrawlLocal = false;
    settings.firecrawl = false;
    changed = true;
  }
  if (settings.firecrawl && !settings.firecrawlLocal && configured('firecrawl') === false) {
    settings.firecrawl = false;
    changed = true;
  }
  return changed;
}
