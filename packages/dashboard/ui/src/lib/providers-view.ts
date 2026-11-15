/** Pure view rules of the Web Search page: sentences, card states, the key format check, progress lines. */

import { formatCount, formatDay, formatRelativeTime } from './format';
import { PROVIDERS } from './providers-strings';
import type { ProviderId, ProviderState, ProvidersResponse, SearchProviderId } from './providers-types';

export const SEARCH_IDS: readonly SearchProviderId[] = ['exa', 'deepseek', 'brave'];
export const PROVIDER_ORDER: readonly ProviderId[] = ['exa', 'deepseek', 'brave', 'firecrawl'];

/** Where each provider hands out keys. */
export const KEY_PAGES: Readonly<Record<ProviderId, string>> = {
  exa: 'https://dashboard.exa.ai/api-keys',
  deepseek: 'https://platform.deepseek.com/api_keys',
  brave: 'https://api-dashboard.search.brave.com/app/keys',
  firecrawl: 'https://www.firecrawl.dev/app/api-keys',
};

/** Where credit is added (the "no credit" line). */
export const BILLING_PAGES: Readonly<Record<ProviderId, string>> = {
  exa: 'https://dashboard.exa.ai',
  deepseek: 'https://platform.deepseek.com',
  brave: 'https://api-dashboard.search.brave.com',
  firecrawl: 'https://www.firecrawl.dev/app',
};

/** `{name}` placeholders filled from `values`; an unknown name stays as written. */
export function fill(template: string, values: Readonly<Record<string, string | number>>): string {
  let out = '';
  let i = 0;
  while (i < template.length) {
    const open = template.indexOf('{', i);
    if (open === -1) break;
    const close = template.indexOf('}', open + 1);
    if (close === -1) break;
    const name = template.slice(open + 1, close);
    out += template.slice(i, open);
    out += Object.hasOwn(values, name) ? String(values[name]) : template.slice(open, close + 1);
    i = close + 1;
  }
  return out + template.slice(i);
}

export function providerName(id: ProviderId): string {
  return PROVIDERS.names[id];
}

export function roleWords(id: ProviderId): string {
  return PROVIDERS.roles[id].join(' · ');
}

export function providerById(data: ProvidersResponse, id: ProviderId): ProviderState | undefined {
  return data.providers.find((p) => p.id === id);
}

/** The key as the server will store it, or null: trimmed, 8 to 512 printable ASCII characters, no spaces. */
export function checkKeyFormat(raw: string): string | null {
  const key = raw.trim();
  if (key.length < 8 || key.length > 512) return null;
  for (let i = 0; i < key.length; i += 1) {
    const c = key.charCodeAt(i);
    if (c <= 0x20 || c >= 0x7f) return null;
  }
  return key;
}

export interface Line {
  text: string;
  tone?: 'danger' | 'warning';
}

/** The "Chat" sentence of the Now panel, plus a line per selected provider whose key failed its check. */
export function chatLines(data: ProvidersResponse): Line[] {
  const search = data.search;
  const fetch = data.fetch;
  let text: string = PROVIDERS.chatOff;
  if (search !== null && fetch === 'firecrawl') {
    text = fill(PROVIDERS.chatBoth, { search: providerName(search), fetch: providerName('firecrawl') });
  } else if (search !== null) {
    text = search === 'brave' ? PROVIDERS.chatBrave : fill(PROVIDERS.chatSame, { search: providerName(search) });
  } else if (fetch === 'firecrawl') {
    text = PROVIDERS.chatPagesOnly;
  }
  const lines: Line[] = [{ text }];
  for (const p of data.providers) {
    if (!p.selected) continue;
    if (p.check === 'rejected')
      lines.push({ text: fill(PROVIDERS.nowRejected, { provider: providerName(p.id) }), tone: 'danger' });
    if (p.check === 'no-credit')
      lines.push({ text: fill(PROVIDERS.nowNoCredit, { provider: providerName(p.id) }), tone: 'warning' });
  }
  return lines;
}

export interface WorkPart {
  text: string;
  /** The selected bundle is not installed. */
  missing: boolean;
}

/** The parts of the "Work" sentence: one per selected bundle, or the one "No web skills." part. */
export function workParts(data: ProvidersResponse): WorkPart[] {
  const parts: WorkPart[] = [];
  for (const p of data.providers) {
    if (!p.selected) continue;
    const provider = providerName(p.id);
    if (p.id === 'firecrawl' && p.work.missing === 'firecrawl-cli') {
      parts.push({ text: PROVIDERS.workNoCli, missing: true });
      continue;
    }
    if (!p.skill.installed) {
      parts.push({ text: fill(PROVIDERS.workMissing, { provider }), missing: true });
      continue;
    }
    const n = p.skill.names.length;
    parts.push({
      text: n === 1 ? fill(PROVIDERS.workOneSkill, { provider }) : fill(PROVIDERS.workSkills, { n, provider }),
      missing: false,
    });
  }
  return parts.length === 0 ? [{ text: PROVIDERS.workNone, missing: false }] : parts;
}

/** The Work sentence as one string: parts joined with commas, one full stop. */
export function workSentence(parts: readonly WorkPart[]): string {
  const text = parts.map((part) => part.text).join(', ');
  return text.endsWith('.') ? text : `${text}.`;
}

export function skillsForWork(p: ProviderState): string {
  const n = p.skill.names.length;
  return n === 1 ? PROVIDERS.oneSkillForWork : fill(PROVIDERS.skillsForWork, { n });
}

export type BadgeState = 'no-key' | 'checking' | 'valid' | 'unverified' | 'rejected' | 'no-credit';

/** The status badge of a provider card. `checking` while a save or a check of this provider is in flight. */
export function badgeState(p: ProviderState, checking: boolean): BadgeState {
  if (checking) return 'checking';
  if (!p.configured) return 'no-key';
  if (p.check === 'ok') return 'valid';
  if (p.check === 'rejected') return 'rejected';
  if (p.check === 'no-credit') return 'no-credit';
  return 'unverified';
}

/** The facts line of a valid key: numbers only, each part only when present. */
export function factsLine(p: ProviderState): string | null {
  if (p.check !== 'ok') return null;
  if (p.id === 'brave') return PROVIDERS.factsBrave;
  const facts = p.facts;
  if (facts === null) return null;
  if (p.id === 'firecrawl' && facts.remainingCredits !== undefined) {
    const left = formatCount(facts.remainingCredits);
    if (facts.planCredits !== undefined && facts.periodEnd !== undefined) {
      return fill(PROVIDERS.factsFirecrawl, {
        left,
        plan: formatCount(facts.planCredits),
        date: formatDay(facts.periodEnd),
      });
    }
    return fill(PROVIDERS.factsFirecrawlLeft, { left });
  }
  if (p.id === 'deepseek' && facts.balances !== undefined && facts.balances.length > 0) {
    return facts.balances
      .map((b) => fill(PROVIDERS.factsBalance, { amount: b.amount.toFixed(2), currency: b.currency }))
      .join(' · ');
  }
  return null;
}

export function checkedLine(p: ProviderState): string | null {
  return p.checkedAt === null ? null : fill(PROVIDERS.checked, { when: formatRelativeTime(p.checkedAt) });
}

/** `{repo} @ {7 chars} · {license}`, or "written by mlx-node" for an own bundle. */
export function skillSource(p: ProviderState): string {
  const up = p.skill.upstream;
  if (up === null) return PROVIDERS.skillsOwn;
  return fill(PROVIDERS.skillsSource, { repo: up.repo, commit: up.commit.slice(0, 7), license: up.license });
}

/** Whether this provider's key reaches the Work agent now: the selected search provider; Firecrawl when its skills do. */
export function keyReachesWork(p: ProviderState): boolean {
  if (p.id === 'firecrawl') return p.configured && p.selected && p.work.ready;
  return p.configured && p.selected;
}

export interface Progress {
  /** Which part of the Use panel changed. */
  area: 'search' | 'firecrawl';
  text: string;
  /** Skill names that were installed. */
  names: readonly string[];
  /** One sentence for the status region. */
  status: string;
}

function installedText(p: ProviderState, named: boolean): string {
  const n = p.skill.names.length;
  if (named) {
    return fill(n === 1 ? PROVIDERS.installedNamedOne : PROVIDERS.installedNamed, { n, provider: providerName(p.id) });
  }
  return n === 1 ? PROVIDERS.installedOne : fill(PROVIDERS.installed, { n });
}

function removedText(p: ProviderState): string {
  const n = p.skill.names.length;
  return fill(n === 1 ? PROVIDERS.removedOne : PROVIDERS.removed, { n, provider: providerName(p.id) });
}

/**
 * The progress line of a change, read from the replies before and after it. "Installed" is said only when the reply
 * after the change reports the bundle installed.
 */
export function progressOf(before: ProvidersResponse, after: ProvidersResponse): Progress | null {
  if (before.search !== after.search) {
    const old = before.search === null ? undefined : providerById(before, before.search);
    const next = after.search === null ? undefined : providerById(after, after.search);
    if (next !== undefined && next.skill.installed) {
      const names = next.skill.names;
      const text = old === undefined ? installedText(next, false) : `${installedText(next, true)}. ${removedText(old)}`;
      const status = fill(names.length === 1 ? PROVIDERS.selectedStatusOne : PROVIDERS.selectedStatus, {
        provider: providerName(next.id),
        n: names.length,
      });
      return { area: 'search', text, names, status };
    }
    if (next === undefined && old !== undefined) {
      return { area: 'search', text: removedText(old), names: [], status: PROVIDERS.offStatus };
    }
    return null;
  }
  if (before.firecrawl !== after.firecrawl) {
    const fire = providerById(after, 'firecrawl');
    const was = providerById(before, 'firecrawl');
    if (after.firecrawl && fire !== undefined && fire.skill.installed) {
      const text = installedText(fire, true);
      return { area: 'firecrawl', text, names: fire.skill.names, status: `${text}.` };
    }
    if (!after.firecrawl && was !== undefined) {
      const text = removedText(was);
      return { area: 'firecrawl', text, names: [], status: text };
    }
  }
  return null;
}

/**
 * A provider that was in use and is no longer configured although the page did not remove it: its key left the
 * Keychain outside the app.
 */
export function vanished(
  before: ProvidersResponse,
  after: ProvidersResponse,
  removed: ProviderId | null,
): ProviderId | null {
  for (const p of before.providers) {
    if (!p.configured || !p.selected || p.id === removed) continue;
    const now = providerById(after, p.id);
    if (now !== undefined && !now.configured) return p.id;
  }
  return null;
}
