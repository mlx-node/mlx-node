/**
 * One key check per provider, from the Control Panel process: the same host allowlist, no redirect, 10 s. Only the
 * status and a few numbers are read; anything that is not a clear yes, no or "no credit" is `unverified` and the key
 * is still saved.
 */

import { httpJson, list, record, type HttpRequest } from '../web/http.js';
import type { KeyCheck, ProviderFacts, ProviderId } from './types.js';

export const VALIDATE_TIMEOUT_MS = 10_000;

export interface KeyValidation {
  check: KeyCheck;
  facts: ProviderFacts | null;
}

export interface ValidateDeps {
  baseUrl: string;
  fetch: typeof fetch;
  timeoutMs?: number;
}

function request(id: ProviderId, key: string, deps: ValidateDeps): HttpRequest {
  const common = {
    fetch: deps.fetch,
    baseUrl: deps.baseUrl,
    method: 'GET' as const,
    timeoutMs: deps.timeoutMs ?? VALIDATE_TIMEOUT_MS,
  };
  switch (id) {
    case 'exa':
      return { ...common, path: '/v0/teams/me', auth: { 'x-api-key': key } };
    case 'deepseek':
      return { ...common, path: '/user/balance', auth: { Authorization: `Bearer ${key}` } };
    case 'brave':
      return {
        ...common,
        path: '/res/v1/web/search',
        query: [
          ['q', 'test'],
          ['count', '1'],
        ],
        auth: { 'X-Subscription-Token': key },
      };
    case 'firecrawl':
      return { ...common, path: '/v2/team/credit-usage', auth: { Authorization: `Bearer ${key}` } };
  }
}

function finite(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function isCurrency(value: unknown): value is string {
  if (typeof value !== 'string' || value.length !== 3) return false;
  for (let i = 0; i < 3; i++) {
    const c = value.charCodeAt(i);
    if (c < 0x41 || c > 0x5a) return false;
  }
  return true;
}

function decimal(value: unknown): number | undefined {
  if (typeof value === 'number') return finite(value);
  if (typeof value !== 'string' || value === '' || value.length > 24) return undefined;
  let dots = 0;
  for (let i = 0; i < value.length; i++) {
    const c = value.charCodeAt(i);
    if (c === 0x2e) dots += 1;
    else if (c < 0x30 || c > 0x39) return undefined;
  }
  return dots > 1 ? undefined : finite(Number(value));
}

/** DeepSeek answers one balance per currency; more than this is not a real answer and is cut. */
const MAX_BALANCES = 4;

function deepseekFacts(json: Record<string, unknown>): ProviderFacts | null {
  const balances: Array<{ currency: string; amount: number }> = [];
  for (const value of list(json.balance_infos)) {
    if (balances.length >= MAX_BALANCES) break;
    const b = record(value);
    const amount = decimal(b?.total_balance);
    if (b === null || !isCurrency(b.currency) || amount === undefined) continue;
    const currency = b.currency;
    if (!balances.some((x) => x.currency === currency)) balances.push({ currency, amount });
  }
  return balances.length === 0 ? null : { balances };
}

function firecrawlFacts(data: Record<string, unknown>): ProviderFacts | null {
  const facts: ProviderFacts = {};
  const remaining = finite(data.remainingCredits);
  const plan = finite(data.planCredits);
  if (remaining !== undefined) facts.remainingCredits = remaining;
  if (plan !== undefined) facts.planCredits = plan;
  if (typeof data.billingPeriodEnd === 'string') {
    const end = Date.parse(data.billingPeriodEnd);
    if (Number.isFinite(end)) facts.periodEnd = end;
  }
  return Object.keys(facts).length === 0 ? null : facts;
}

/** Checks one key. Never throws. */
export async function validateKey(id: ProviderId, key: string, deps: ValidateDeps): Promise<KeyValidation> {
  const reply = await httpJson(request(id, key, deps));
  if (reply.kind === 'failed') return { check: 'unverified', facts: null };
  if (reply.status === 401 || reply.status === 403) return { check: 'rejected', facts: null };
  if (reply.status === 402) return { check: 'no-credit', facts: null };
  if (reply.status !== 200) return { check: 'unverified', facts: null };
  const json = record(reply.json);
  if (id === 'exa' || id === 'brave') return { check: 'ok', facts: null };
  if (json === null) return { check: 'unverified', facts: null };
  if (id === 'deepseek') {
    if (json.is_available === true) return { check: 'ok', facts: deepseekFacts(json) };
    if (json.is_available === false) return { check: 'no-credit', facts: deepseekFacts(json) };
    return { check: 'unverified', facts: null };
  }
  const data = record(json.data);
  const remaining = finite(data?.remainingCredits);
  if (data === null || remaining === undefined) return { check: 'unverified', facts: null };
  return { check: remaining > 0 ? 'ok' : 'no-credit', facts: firecrawlFacts(data) };
}
