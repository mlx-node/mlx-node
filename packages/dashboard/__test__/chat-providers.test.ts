/**
 * The providers service and its routes, on a memory key store and fake provider servers on 127.0.0.1: the exclusive
 * search choice (exactly one or none, Firecrawl independent), the key check table, the key cache (no Keychain call on
 * a create or load path, the 8 s timeout, retired keys kept for redaction), skills installed on select and removed on
 * deselect, a damaged bundle that changes nothing, stale hosts after a change, and a marker key that never shows up in
 * a reply, a log, a thrown error or a file.
 */

import {
  chmodSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { ApiError, type ApiResponse } from '../src/api/errors.js';
import { chatPaths, type ChatPaths } from '../src/chat/paths.js';
import { checkKeyFormat, MemorySecretStore, type SecretStore } from '../src/chat/providers/secrets.js';
import { ProvidersService, type ProvidersServiceOptions } from '../src/chat/providers/service.js';
import { parseSettings, repairSettings, defaultSettings } from '../src/chat/providers/settings.js';
import type { ProviderId, ProvidersResponse } from '../src/chat/providers/types.js';
import { validateKey } from '../src/chat/providers/validate.js';
import { routeError } from '../src/chat/route-errors.js';
import { providerIdParam, providerKeyInput, providerSelectionInput } from '../src/chat/route-input.js';
import type { ChatService } from '../src/chat/service.js';
import { defaultSkillsDir } from '../src/chat/skills/manifest.js';
import { formatErrorText } from '../src/chat/web/result-text.js';
import { createDashboardRuntime, type DashboardRuntime } from '../src/runtime.js';
import {
  fakeService,
  FAKE_AGENT,
  INFERENCE_TOKEN,
  NO_NETWORK,
  readScenario,
  until,
  type Json,
} from './helpers/chat-fake.js';
import {
  CHECK_OK,
  type FakeAnswer,
  type FakeProvider,
  type FakeRequest,
  keyOf,
  startFakeProvider,
  startHarnessProviders,
} from './helpers/fake-providers.js';

const MARKER = 'mk-MARKERKEY-7f3a9c2e5b1d';
const IDS: readonly ProviderId[] = ['exa', 'deepseek', 'brave', 'firecrawl'];

let root: string;
let paths: ChatPaths;
const servers: FakeProvider[] = [];

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'mlx-providers-'));
  paths = chatPaths(join(root, 'chat'));
});

afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(servers.splice(0).map((s) => s.close()));
  rmSync(root, { recursive: true, force: true });
});

/** One fake per provider; each answers by `answer(id, request)`, by default a valid key check. */
async function providerServers(
  answer: (id: ProviderId, request: FakeRequest) => FakeAnswer = (id) => ({ json: CHECK_OK[id] }),
): Promise<Record<ProviderId, string>> {
  const started = await Promise.all(IDS.map((id) => startFakeProvider((r) => answer(id, r))));
  servers.push(...started);
  return {
    exa: started[0]!.baseUrl,
    deepseek: started[1]!.baseUrl,
    brave: started[2]!.baseUrl,
    firecrawl: started[3]!.baseUrl,
  };
}

/** A key store that counts calls and can be made to hang or throw. */
class CountingStore implements SecretStore {
  readonly inner = new MemorySecretStore();
  calls: string[] = [];
  hang = false;
  fail: string | null = null;
  readonly waiting: Array<() => void> = [];

  private async gate(kind: string): Promise<void> {
    this.calls.push(kind);
    if (this.fail !== null) throw new Error(this.fail);
    if (this.hang) await new Promise<void>((resolve) => this.waiting.push(resolve));
  }

  async get(id: ProviderId): Promise<string | null> {
    await this.gate(`get ${id}`);
    return this.inner.get(id);
  }

  async set(id: ProviderId, key: string): Promise<void> {
    await this.gate(`set ${id}`);
    return this.inner.set(id, key);
  }

  async delete(id: ProviderId): Promise<void> {
    await this.gate(`delete ${id}`);
    return this.inner.delete(id);
  }
}

function service(over: Partial<ProvidersServiceOptions> = {}): ProvidersService {
  return new ProvidersService({
    paths,
    secrets: new MemorySecretStore(),
    fetch: () => Promise.reject(new Error('no network')),
    workPath: () => '',
    ...over,
  });
}

function selected(response: ProvidersResponse): ProviderId[] {
  return response.providers.filter((p) => p.selected).map((p) => p.id);
}

function pluginFolders(): string[] {
  return existsSync(paths.plugins) ? readdirSync(paths.plugins).sort() : [];
}

async function failure(
  promise: Promise<unknown>,
  route: Parameters<typeof routeError>[0] = 'provider-key',
): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    return routeError(route, error);
  }
  throw new Error('expected a failure');
}

/** Every file under `dir`, as text. */
function filesUnder(dir: string): Array<[string, string]> {
  const out: Array<[string, string]> = [];
  if (!existsSync(dir)) return out;
  for (const name of readdirSync(dir, { recursive: true }).map(String)) {
    const path = join(dir, name);
    try {
      out.push([path, readFileSync(path, 'latin1')]);
    } catch {
      // A folder.
    }
  }
  return out;
}

/** Grows by one entry each time a turn of the conversation ends. */
function turnEnds(svc: ChatService, id: string): number[] {
  const ends: number[] = [];
  svc.watchConversation(id, (event) => {
    if (event.type === 'turn' && event.turn === null) ends.push(ends.length);
  });
  return ends;
}

/** Every 8-character window of the marker. */
function windows(text: string): string[] {
  return Array.from({ length: text.length - 7 }, (_, i) => text.slice(i, i + 8));
}

function containsMarker(text: string): boolean {
  return windows(MARKER).some((w) => text.includes(w));
}

describe('settings file', () => {
  it('reads only known fields and values; a bad file reads as the defaults', () => {
    expect(parseSettings('not json')).toEqual(defaultSettings());
    expect(parseSettings('{"version":2,"search":"exa"}')).toEqual(defaultSettings());
    expect(
      parseSettings(
        JSON.stringify({
          version: 1,
          search: 'google',
          firecrawl: 'yes',
          checks: { exa: { status: 'ok', at: 5 }, brave: { status: 'great', at: 1 }, x: { status: 'ok', at: 1 } },
          key: MARKER,
        }),
      ),
    ).toEqual({ version: 1, search: null, firecrawl: false, checks: { exa: { status: 'ok', at: 5 } } });
  });

  it('repairs a selection without a key, and keeps one whose key is not known yet', () => {
    const settings = { ...defaultSettings(), search: 'brave' as const, firecrawl: true };
    expect(repairSettings(settings, () => null)).toBe(false);
    expect(settings).toMatchObject({ search: 'brave', firecrawl: true });
    expect(repairSettings(settings, () => false)).toBe(true);
    expect(settings).toMatchObject({ search: null, firecrawl: false });
  });
});

describe('route inputs', () => {
  it('takes exactly the listed fields; a bad key format never quotes the key', async () => {
    expect(providerKeyInput({ key: 'abc' })).toEqual({ key: 'abc' });
    expect(() => providerKeyInput({ key: 'abc', search: 'exa' })).toThrow('Unknown field');
    expect(() => providerKeyInput({ key: 5 })).toThrow('wrong type');
    expect(() => providerKeyInput({})).toThrow('required');
    expect(providerSelectionInput({ search: null })).toEqual({ search: null });
    expect(providerSelectionInput({ search: 'brave', firecrawl: false })).toEqual({
      search: 'brave',
      firecrawl: false,
    });
    expect(() => providerSelectionInput({ search: 'firecrawl' })).toThrow('must be');
    expect(() => providerSelectionInput({ firecrawl: 'yes' })).toThrow('wrong type');
    expect(() => providerSelectionInput({})).toThrow('required');
    expect(() => providerSelectionInput({ search: 'exa', pluginDirs: ['/tmp'] })).toThrow('Unknown field');
    expect(() => providerIdParam('google')).toThrow('No such provider');
    expect(providerIdParam('firecrawl')).toBe('firecrawl');
    expect(checkKeyFormat('  abcdefgh  ')).toBe('abcdefgh');
    for (const bad of ['short', 'has space inside', `${'x'.repeat(513)}`, 'tab\there12', 'ünicode-key']) {
      expect(checkKeyFormat(bad)).toBeNull();
    }
    const error = await failure(service().putKey('exa', `${MARKER} spaced`));
    expect([error.code, error.message]).toEqual(['E_BAD_REQUEST', 'That does not look like an API key.']);
  });
});

describe('key checks', () => {
  it('maps each provider answer to ok, rejected, no-credit or unverified, with facts as numbers only', async () => {
    const answers: Record<string, FakeAnswer> = {};
    const urls = await providerServers((id, r) => answers[`${id}:${keyOf(r)}`] ?? { json: CHECK_OK[id] });
    const check = (id: ProviderId, key: string) =>
      validateKey(id, key, { baseUrl: urls[id], fetch: (i, n) => fetch(i, n) });
    for (const id of IDS) {
      expect([id, await check(id, 'good-key-1')]).toMatchObject([id, { check: 'ok' }]);
      answers[`${id}:k401-0000`] = { status: 401, json: { error: MARKER } };
      answers[`${id}:k403-0000`] = { status: 403, json: {} };
      answers[`${id}:k402-0000`] = { status: 402, json: {} };
      answers[`${id}:k500-0000`] = { status: 500, json: {} };
      answers[`${id}:k429-0000`] = { status: 429, json: {} };
      answers[`${id}:k404-0000`] = { status: 404, json: {} };
      answers[`${id}:kdrop-000`] = { drop: true };
      answers[`${id}:khtml-000`] = { body: '<html>' };
      expect([id, (await check(id, 'k401-0000')).check]).toEqual([id, 'rejected']);
      expect([id, (await check(id, 'k403-0000')).check]).toEqual([id, 'rejected']);
      expect([id, (await check(id, 'k402-0000')).check]).toEqual([id, 'no-credit']);
      for (const key of ['k500-0000', 'k429-0000', 'k404-0000', 'kdrop-000']) {
        expect([id, key, (await check(id, key)).check]).toEqual([id, key, 'unverified']);
      }
    }
    expect(await check('deepseek', 'good-key-1')).toEqual({
      check: 'ok',
      facts: { balances: [{ currency: 'USD', amount: 12.5 }] },
    });
    answers['deepseek:empty-000'] = {
      json: { is_available: false, balance_infos: [{ currency: 'usd!', total_balance: MARKER }] },
    };
    expect(await check('deepseek', 'empty-000')).toEqual({ check: 'no-credit', facts: null });
    expect(await check('firecrawl', 'good-key-1')).toEqual({
      check: 'ok',
      facts: { remainingCredits: 742, planCredits: 1000, periodEnd: Date.parse('2026-12-01T00:00:00Z') },
    });
    answers['firecrawl:zero-0000'] = { json: { success: true, data: { remainingCredits: 0 } } };
    expect((await check('firecrawl', 'zero-0000')).check).toBe('no-credit');
    const requests = servers.flatMap((s) => s.requests);
    const exa = requests.find((r) => r.path === '/v0/teams/me')!;
    expect([exa.method, exa.headers['x-api-key']]).toEqual(['GET', 'good-key-1']);
    expect(requests.find((r) => r.path === '/user/balance')!.headers.authorization).toBe('Bearer good-key-1');
    const brave = requests.find((r) => r.path === '/res/v1/web/search')!;
    expect(brave.query).toEqual([
      ['q', 'test'],
      ['count', '1'],
    ]);
    expect(brave.headers['x-subscription-token']).toBe('good-key-1');
    expect(requests.find((r) => r.path === '/v2/team/credit-usage')!.headers.authorization).toBe('Bearer good-key-1');
  });

  it('keeps at most four DeepSeek balances, one per currency', async () => {
    const balances = Array.from({ length: 5_000 }, (_, i) => ({
      currency: ['USD', 'CNY', 'USD', 'EUR', 'JPY', 'GBP'][i % 6],
      total_balance: String(i),
    }));
    const urls = await providerServers(() => ({ json: { is_available: true, balance_infos: balances } }));
    const result = await validateKey('deepseek', 'good-key-1', {
      baseUrl: urls.deepseek,
      fetch: (i, n) => fetch(i, n),
    });
    expect(result.facts?.balances?.map((b) => b.currency)).toEqual(['USD', 'CNY', 'EUR', 'JPY']);
  });

  it('the harness providers: valid, rejected, no credit, and the first keys become the selection', async () => {
    const harness = await startHarnessProviders();
    try {
      const svc = service({ baseUrls: harness.baseUrls, fetch: (i, n) => fetch(i, n) });
      expect(await svc.putKey('exa', 'good-key-1234')).toMatchObject({ saved: true, check: 'ok' });
      expect(await svc.putKey('brave', 'bad-key-1234')).toMatchObject({ saved: false, check: 'rejected' });
      const credit = await svc.putKey('firecrawl', 'nocredit-1234');
      expect(credit).toMatchObject({ saved: true, check: 'no-credit' });
      expect(credit.providers).toMatchObject({ search: 'exa', firecrawl: true, fetch: 'firecrawl' });
      expect(JSON.stringify(credit)).not.toContain('good-key-1234');
    } finally {
      await harness.close();
    }
  });
});

describe('the exclusive choice', () => {
  it('exactly one search provider or none, even with all three keys; Firecrawl is independent', async () => {
    const urls = await providerServers();
    const svc = service({ baseUrls: urls, fetch: (i, n) => fetch(i, n) });
    const first = await svc.putKey('exa', 'exa-key-0001');
    expect(first).toMatchObject({ saved: true, check: 'ok' });
    expect(first.providers.search).toBe('exa');
    await svc.putKey('deepseek', 'ds-key-00001');
    const three = await svc.putKey('brave', 'brave-key-001');
    expect(three.providers.search).toBe('exa');
    expect(selected(three.providers)).toEqual(['exa']);
    expect(three.providers.providers.map((p) => p.configured)).toEqual([true, true, true, false]);

    const ds = await svc.select({ search: 'deepseek' });
    expect([ds.search, selected(ds)]).toEqual(['deepseek', ['deepseek']]);
    expect(pluginFolders()).toEqual(['chat-web', 'deepseek']);

    await svc.putKey('firecrawl', 'fc-key-000001');
    const both = await svc.list();
    expect([both.search, both.firecrawl, both.fetch, selected(both)]).toEqual([
      'deepseek',
      true,
      'firecrawl',
      ['deepseek', 'firecrawl'],
    ]);

    const brave = await svc.select({ search: 'brave' });
    expect([brave.search, brave.firecrawl, selected(brave)]).toEqual(['brave', true, ['brave', 'firecrawl']]);
    expect(pluginFolders()).toEqual(['brave', 'chat-web', 'firecrawl']);

    const off = await svc.select({ search: null, firecrawl: false });
    expect([off.search, off.firecrawl, off.fetch, selected(off)]).toEqual([null, false, null, []]);
    expect(pluginFolders()).toEqual(['chat-web']);
    expect(JSON.parse(readFileSync(paths.providers, 'utf8'))).toMatchObject({
      version: 1,
      search: null,
      firecrawl: false,
    });
  });

  it('refuses to select a provider without a key', async () => {
    const svc = service();
    const error = await failure(svc.select({ search: 'exa' }), 'provider-selection');
    expect([error.code, error.message]).toEqual(['E_CONFLICT', 'Add a key for Exa first.']);
    expect((await failure(svc.select({ firecrawl: true }), 'provider-selection')).code).toBe('E_CONFLICT');
  });

  it('a damaged bundle changes nothing: the selection and the folders stay as they were', async () => {
    const skills = join(root, 'skills');
    cpSync(defaultSkillsDir(), skills, { recursive: true });
    writeFileSync(join(skills, 'vendor/brave/skills/web-search/SKILL.md'), 'tampered');
    const store = new MemorySecretStore();
    await store.set('exa', 'exa-key-0001');
    await store.set('brave', 'brave-key-001');
    const svc = service({ secrets: store, skillsDir: skills });
    expect((await svc.select({ search: 'exa' })).search).toBe('exa');
    const error = await failure(svc.select({ search: 'brave' }), 'provider-selection');
    expect([error.code, error.reason]).toEqual(['E_UNAVAILABLE', 'skill-bundle-corrupt']);
    const after = await svc.list();
    expect([after.search, after.providers[0]!.skill.installed, after.providers[2]!.skill.installed]).toEqual([
      'exa',
      true,
      false,
    ]);
    expect(pluginFolders()).toEqual(['chat-web', 'exa']);
    expect(JSON.parse(readFileSync(paths.providers, 'utf8')).search).toBe('exa');
  });

  it('a failed switch keeps the old provider: its skill folder stays and Work still gets it', async () => {
    const skills = join(root, 'skills');
    cpSync(defaultSkillsDir(), skills, { recursive: true });
    const store = new MemorySecretStore();
    await store.set('exa', 'exa-key-0001');
    await store.set('brave', 'brave-key-001');
    const svc = service({ secrets: store, skillsDir: skills });
    expect((await svc.select({ search: 'exa' })).search).toBe('exa');
    writeFileSync(join(skills, 'vendor/exa/skills/exa-search/SKILL.md'), 'tampered');
    writeFileSync(join(skills, 'vendor/brave/skills/web-search/SKILL.md'), 'tampered');
    const error = await failure(svc.select({ search: 'brave' }), 'provider-selection');
    expect([error.code, error.reason]).toEqual(['E_UNAVAILABLE', 'skill-bundle-corrupt']);
    const after = await svc.list();
    expect([after.search, after.providers[0]!.skill.installed]).toEqual(['exa', true]);
    expect(pluginFolders()).toEqual(['chat-web', 'exa']);
    expect(svc.workPluginDirs()).toEqual([join(paths.plugins, 'exa')]);
  });

  it('a folder a failed sync leaves behind is not reported installed and does not reach Work', async () => {
    const skills = join(root, 'skills');
    cpSync(defaultSkillsDir(), skills, { recursive: true });
    const store = new MemorySecretStore();
    await store.set('exa', 'exa-key-0001');
    await store.set('firecrawl', 'fc-key-000001');
    const svc = service({ secrets: store, skillsDir: skills });
    await svc.select({ search: 'exa', firecrawl: true });
    writeFileSync(join(skills, 'vendor/firecrawl/skills/firecrawl/SKILL.md'), 'tampered');
    rmSync(join(paths.plugins, 'firecrawl'), { recursive: true, force: true });
    await svc.deleteKey('exa');
    expect(pluginFolders()).toEqual(['chat-web', 'exa']);
    const exa = (await svc.list()).providers[0]!;
    expect([exa.configured, exa.selected, exa.skill.installed]).toEqual([false, false, false]);
    expect(svc.workPluginDirs()).toEqual([]);
  });
});

describe('keys', () => {
  it('a rejected key is not saved and the earlier key stays; a key that cannot be checked is saved', async () => {
    const urls = await providerServers((id, r) =>
      keyOf(r).startsWith('bad')
        ? { status: 401, json: {} }
        : keyOf(r).startsWith('down')
          ? { status: 503, json: {} }
          : { json: CHECK_OK[id] },
    );
    const store = new MemorySecretStore();
    const svc = service({ secrets: store, baseUrls: urls, fetch: (i, n) => fetch(i, n) });
    await svc.putKey('exa', 'exa-key-0001');
    const rejected = await svc.putKey('exa', 'bad-key-0001');
    expect([rejected.saved, rejected.check, rejected.providers.providers[0]!.check]).toEqual([false, 'rejected', 'ok']);
    expect(await store.get('exa')).toBe('exa-key-0001');
    const unverified = await svc.putKey('exa', 'down-key-001');
    expect([unverified.saved, unverified.check]).toEqual([true, 'unverified']);
    expect(await store.get('exa')).toBe('down-key-001');
  });

  it('a removed key clears the selection and its skills; replaced and removed keys stay redaction needles', async () => {
    const urls = await providerServers();
    const svc = service({ baseUrls: urls, fetch: (i, n) => fetch(i, n) });
    await svc.putKey('brave', 'brave-key-old');
    await svc.putKey('brave', 'brave-key-new');
    expect(svc.secretNeedles().sort()).toEqual(['brave-key-new', 'brave-key-old']);
    expect(svc.workKeys()).toEqual({ BRAVE_SEARCH_API_KEY: 'brave-key-new', BRAVE_API_KEY: 'brave-key-new' });
    expect(svc.chatWeb()).toEqual({
      searchProvider: 'brave',
      searchKey: 'brave-key-new',
      fetchProvider: 'brave',
      fetchKey: 'brave-key-new',
    });
    const removed = await svc.deleteKey('brave');
    expect([removed.search, removed.providers[2]!.configured, removed.providers[2]!.check]).toEqual([
      null,
      false,
      null,
    ]);
    expect(pluginFolders()).toEqual(['chat-web']);
    expect(svc.secretNeedles().sort()).toEqual(['brave-key-new', 'brave-key-old']);
    expect(svc.workKeys()).toEqual({});
    expect(svc.chatWeb()).toEqual({ searchProvider: 'none', searchKey: '', fetchProvider: 'none', fetchKey: '' });
  });

  it('a Check re-reads the Keychain: a key removed outside the app turns the provider off', async () => {
    const store = new MemorySecretStore();
    await store.set('exa', 'exa-key-0001');
    const svc = service({ secrets: store });
    await svc.select({ search: 'exa' });
    await store.delete('exa');
    expect((await svc.list()).search).toBe('exa');
    const checked = await svc.validate('exa');
    expect([checked.check, checked.providers.search, checked.providers.providers[0]!.configured]).toEqual([
      null,
      null,
      false,
    ]);
    expect(pluginFolders()).toEqual(['chat-web']);
  });

  it('a web result that says the key was rejected marks the check, in the file too', async () => {
    const store = new MemorySecretStore();
    await store.set('exa', 'exa-key-0001');
    const svc = service({ secrets: store });
    await svc.select({ search: 'exa' });
    svc.markKeyCheck('exa', 'rejected', 'exa-key-0001');
    await svc.settled();
    expect((await svc.list()).providers[0]!.check).toBe('rejected');
    expect(svc.summary()).toMatchObject({ search: 'exa', searchCheck: 'rejected' });
    expect(JSON.parse(readFileSync(paths.providers, 'utf8')).checks.exa.status).toBe('rejected');
  });

  it('a web result about a replaced key leaves the new key alone', async () => {
    const urls = await providerServers();
    const svc = service({ baseUrls: urls, fetch: (i, n) => fetch(i, n) });
    await svc.putKey('exa', 'exa-key-0001');
    expect((await svc.putKey('exa', 'exa-key-0002')).check).toBe('ok');
    svc.markKeyCheck('exa', 'rejected', 'exa-key-0001');
    svc.markKeyCheck('exa', 'no-credit', '');
    await svc.settled();
    expect((await svc.list()).providers[0]!.check).toBe('ok');
    expect(JSON.parse(readFileSync(paths.providers, 'utf8')).checks.exa.status).toBe('ok');
  });

  it('a first key whose skills do not install is saved, but its provider is not turned on', async () => {
    const skills = join(root, 'skills');
    cpSync(defaultSkillsDir(), skills, { recursive: true });
    writeFileSync(join(skills, 'vendor/exa/skills/exa-search/SKILL.md'), 'tampered');
    const svc = service({ skillsDir: skills });
    const reply = await svc.putKey('exa', 'exa-key-0001');
    const exa = reply.providers.providers[0]!;
    expect([reply.saved, reply.providers.search, exa.configured, exa.selected]).toEqual([true, null, true, false]);
    expect(JSON.parse(readFileSync(paths.providers, 'utf8')).search).toBeNull();
    expect(svc.workKeys()).toEqual({});
    expect(svc.chatWeb().searchProvider).toBe('none');
    const select = await failure(svc.select({ search: 'exa' }), 'provider-selection');
    expect([select.code, select.reason]).toEqual(['E_UNAVAILABLE', 'skill-bundle-corrupt']);
  });

  it('a key saved while the skill folders cannot be written is saved, off, and not a settings failure', async () => {
    const svc = service();
    await svc.list();
    rmSync(paths.plugins, { recursive: true, force: true });
    writeFileSync(paths.plugins, 'not a folder');
    const reply = await svc.putKey('exa', 'exa-key-0001');
    expect([reply.saved, reply.providers.search, reply.providers.providers[0]!.configured]).toEqual([true, null, true]);
    expect(JSON.parse(readFileSync(paths.providers, 'utf8')).search).toBeNull();
  });

  it('a Remove that timed out holds the provider until it lands, then completes the change', async () => {
    const inner = new MemorySecretStore();
    let gate: Promise<void> | null = null;
    let open = (): void => {};
    // The native call blocks before it changes the Keychain, as a Keychain prompt does.
    const store: SecretStore = {
      get: (id) => inner.get(id),
      set: (id, key) => inner.set(id, key),
      delete: async (id) => {
        if (gate !== null) await gate;
        await inner.delete(id);
      },
    };
    const svc = service({ secrets: store, keychainCallMs: 20 });
    expect((await svc.putKey('exa', 'exa-key-OLD1')).providers.search).toBe('exa');
    gate = new Promise<void>((resolve) => {
      open = resolve;
    });
    expect((await failure(svc.deleteKey('exa'))).reason).toBe('keychain-unavailable');
    const blocked = await failure(svc.putKey('exa', 'exa-key-NEW2'));
    expect([blocked.code, blocked.reason]).toEqual(['E_UNAVAILABLE', 'keychain-unavailable']);
    expect((await failure(svc.deleteKey('exa'))).reason).toBe('keychain-unavailable');
    expect(await inner.get('exa')).toBe('exa-key-OLD1');

    gate = null;
    open();
    await until(() => svc.chatWeb().searchProvider === 'none');
    await svc.settled();
    expect(await inner.get('exa')).toBeNull();
    expect(JSON.parse(readFileSync(paths.providers, 'utf8')).search).toBeNull();
    const listed = await svc.list();
    expect([listed.search, listed.providers[0]!.configured, listed.providers[0]!.skill.installed]).toEqual([
      null,
      false,
      false,
    ]);
    expect(pluginFolders()).toEqual(['chat-web']);

    const again = await svc.putKey('exa', 'exa-key-NEW2');
    expect([again.saved, again.providers.search]).toEqual([true, 'exa']);
    expect(await inner.get('exa')).toBe('exa-key-NEW2');
  });

  it('a Check that ends after a Replace or a Remove leaves the new state alone', async () => {
    for (const change of ['replace', 'remove'] as const) {
      const store = new MemorySecretStore();
      await store.set('exa', 'OLD-KEY-1111');
      let release = (): void => {};
      const held = new Promise<void>((resolve) => {
        release = resolve;
      });
      let oldChecks = 0;
      const slowForOld: typeof fetch = async (_input, init) => {
        if (new Headers(init?.headers).get('x-api-key') === 'OLD-KEY-1111') {
          oldChecks += 1;
          await held;
        }
        return new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } });
      };
      const svc = service({ secrets: store, fetch: slowForOld });
      await svc.select({ search: 'exa' });
      const checking = svc.validate('exa');
      await until(() => oldChecks === 1);
      if (change === 'replace') expect((await svc.putKey('exa', 'NEW-KEY-2222')).saved).toBe(true);
      else await svc.deleteKey('exa');
      release();
      const reply = await checking;
      if (change === 'replace') {
        expect(await store.get('exa')).toBe('NEW-KEY-2222');
        expect(svc.chatWeb()).toMatchObject({ searchKey: 'NEW-KEY-2222', fetchKey: 'NEW-KEY-2222' });
        expect(svc.workKeys()).toEqual({ EXA_API_KEY: 'NEW-KEY-2222' });
        expect([reply.check, reply.providers.search]).toEqual(['ok', 'exa']);
      } else {
        expect(await store.get('exa')).toBeNull();
        expect([reply.check, reply.providers.search, reply.providers.providers[0]!.configured]).toEqual([
          null,
          null,
          false,
        ]);
        expect(svc.chatWeb()).toEqual({ searchProvider: 'none', searchKey: '', fetchProvider: 'none', fetchKey: '' });
        expect((await failure(svc.select({ search: 'exa' }), 'provider-selection')).code).toBe('E_CONFLICT');
      }
      rmSync(paths.root, { recursive: true, force: true });
    }
  });
});

describe('key cache', () => {
  it('reads each key once; hosts read only memory afterwards', async () => {
    const store = new CountingStore();
    await store.inner.set('exa', 'exa-key-0001');
    const svc = service({ secrets: store });
    await svc.settled();
    expect(store.calls.sort()).toEqual(['get brave', 'get deepseek', 'get exa', 'get firecrawl']);
    await svc.select({ search: 'exa' });
    for (let i = 0; i < 3; i++) {
      svc.chatWeb();
      svc.workKeys();
      svc.workPluginDirs();
      svc.secretNeedles();
      svc.summary();
      await svc.settled();
      await svc.chatSkillReady();
      await svc.list();
    }
    expect(store.calls).toHaveLength(4);
  });

  it('no Chat create or load path calls the Keychain past the first fill', async () => {
    const store = new CountingStore();
    await store.inner.set('exa', 'exa-key-0001');
    const fake = fakeService('chat-basic', {
      providers: (p) => new ProvidersService({ paths: p, secrets: store, fetch: NO_NETWORK, workPath: () => '' }),
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello' });
      const ends = turnEnds(fake.service, conversation.id);
      await until(() => ends.length === 1, 5_000);
      await fake.service.prompt(conversation.id, 'second');
      await until(() => ends.length === 2, 5_000);
      expect(store.calls).toHaveLength(4);
      await fake.service.close();
      const svc = fake.restart();
      const cold = turnEnds(svc, conversation.id);
      await svc.prompt(conversation.id, 'after a restart');
      await until(() => cold.length === 1, 5_000);
      expect(fake.sent().filter((m) => m.method === 'session/load')).toHaveLength(1);
      expect(store.calls).toHaveLength(8);
      expect(store.calls.filter((c) => !c.startsWith('get '))).toEqual([]);
    } finally {
      await fake.close();
    }
  }, 20_000);

  it('a Keychain that does not answer in time: the page sees unavailable, hosts start without a key, a write fails typed', async () => {
    const store = new CountingStore();
    await store.inner.set('exa', 'exa-key-0001');
    await store.inner.set('firecrawl', 'fc-key-000001');
    mkdirSync(paths.root, { recursive: true });
    writeFileSync(paths.providers, JSON.stringify({ version: 1, search: 'exa', firecrawl: true, checks: {} }));
    store.hang = true;
    const svc = service({ secrets: store, keychainCallMs: 50 });
    const started = Date.now();
    const listed = await svc.list();
    expect(Date.now() - started).toBeLessThan(2_000);
    expect(listed.keychain).toBe('unavailable');
    expect(svc.chatWeb()).toEqual({ searchProvider: 'none', searchKey: '', fetchProvider: 'none', fetchKey: '' });
    expect(svc.workKeys()).toEqual({});
    expect(JSON.parse(readFileSync(paths.providers, 'utf8'))).toMatchObject({ search: 'exa', firecrawl: true });
    const write = await failure(svc.putKey('brave', 'brave-key-001'));
    expect([write.code, write.reason, write.message]).toEqual([
      'E_UNAVAILABLE',
      'keychain-unavailable',
      'The Keychain cannot be reached.',
    ]);
    const changed = vi.fn();
    svc.attach({ changed, keysChanged: () => {}, applyPending: () => false });
    store.hang = false;
    for (const resolve of store.waiting.splice(0)) resolve();
    await until(() => svc.chatWeb().searchProvider === 'exa');
    expect(svc.chatWeb()).toMatchObject({
      searchKey: 'exa-key-0001',
      fetchProvider: 'firecrawl',
      fetchKey: 'fc-key-000001',
    });
    expect(changed).toHaveBeenCalled();
  });
});

describe('the Keychain mark', () => {
  it('clears once the Keychain answers again: a denied first read, a write and a Check that land late', async () => {
    const store = new CountingStore();
    await store.inner.set('exa', 'exa-key-0001');
    mkdirSync(paths.root, { recursive: true });
    writeFileSync(paths.providers, JSON.stringify({ version: 1, search: 'exa', firecrawl: false, checks: {} }));
    store.fail = 'denied';
    const svc = service({ secrets: store, keychainCallMs: 50 });
    expect((await svc.list()).keychain).toBe('unavailable');
    expect((await svc.list()).keychain).toBe('unavailable');
    const changed = vi.fn();
    svc.attach({ changed, keysChanged: () => {}, applyPending: () => false });
    store.fail = null;
    const back = await svc.list();
    expect([back.keychain, back.search, back.providers[0]!.configured]).toEqual(['ok', 'exa', true]);
    expect(svc.chatWeb().searchKey).toBe('exa-key-0001');
    expect(changed).toHaveBeenCalledTimes(1);

    store.hang = true;
    const write = await failure(svc.putKey('brave', 'brave-key-001'));
    expect(write.reason).toBe('keychain-unavailable');
    store.hang = false;
    for (const resolve of store.waiting.splice(0)) resolve();
    await until(() => svc.secretNeedles().includes('brave-key-001'));
    const afterWrite = await svc.select({ firecrawl: false });
    expect([afterWrite.keychain, afterWrite.providers[2]!.configured]).toEqual(['ok', true]);

    store.hang = true;
    const check = await failure(svc.validate('exa'), 'provider-validate');
    expect(check.reason).toBe('keychain-unavailable');
    expect((await svc.select({ firecrawl: false })).keychain).toBe('unavailable');
    await store.inner.delete('exa');
    store.hang = false;
    for (const resolve of store.waiting.splice(0)) resolve();
    await until(() => svc.chatWeb().searchProvider === 'none');
    const afterCheck = await svc.select({ firecrawl: false });
    expect([afterCheck.keychain, afterCheck.search, afterCheck.providers[0]!.configured]).toEqual(['ok', null, false]);
  });

  it('a key change whose settings cannot be written still reaches the hosts, with its own sentence', async () => {
    const urls = await providerServers();
    const store = new MemorySecretStore();
    const svc = service({ secrets: store, baseUrls: urls, fetch: (i, n) => fetch(i, n) });
    await svc.putKey('exa', 'exa-key-0001');
    const changed = vi.fn();
    svc.attach({ changed, keysChanged: () => {}, applyPending: () => false });
    rmSync(paths.providers);
    mkdirSync(join(paths.providers, 'blocked'), { recursive: true });
    for (const run of [() => svc.putKey('exa', 'exa-key-0002'), () => svc.deleteKey('exa')]) {
      changed.mockClear();
      const error = await failure(run());
      expect([error.code, error.message]).toEqual([
        'E_INTERNAL',
        'The key changed, but the settings could not be saved.',
      ]);
      expect(changed).toHaveBeenCalledTimes(1);
    }
    expect(await store.get('exa')).toBeNull();
    expect(svc.chatWeb().searchKey).toBe('');
    await store.set('exa', 'exa-key-0003');
    changed.mockClear();
    await failure(svc.validate('exa'), 'provider-validate');
    expect(changed).toHaveBeenCalledTimes(1);
    expect(svc.secretNeedles()).toContain('exa-key-0003');
  });

  it('tells the page once about a provider in use whose key was gone at app start', async () => {
    const store = new MemorySecretStore();
    await store.set('firecrawl', 'fc-key-000001');
    mkdirSync(paths.root, { recursive: true });
    writeFileSync(paths.providers, JSON.stringify({ version: 1, search: 'exa', firecrawl: true, checks: {} }));
    const svc = service({ secrets: store });
    const first = await svc.list();
    expect([first.vanished, first.search, first.firecrawl]).toEqual(['exa', null, true]);
    expect((await svc.list()).vanished).toBeNull();
  });

  it('a skill copy that fails on disk is not reported as a damaged bundle', async () => {
    const store = new MemorySecretStore();
    await store.set('exa', 'exa-key-0001');
    const svc = service({ secrets: store });
    await svc.list();
    rmSync(paths.plugins, { recursive: true, force: true });
    writeFileSync(paths.plugins, 'not a folder');
    const error = await failure(svc.select({ search: 'exa' }), 'provider-selection');
    expect([error.code, error.reason, error.message]).toEqual([
      'E_UNAVAILABLE',
      undefined,
      'Skills could not be installed.',
    ]);
    expect((await svc.list()).search).toBeNull();
    rmSync(paths.plugins);
    expect(await svc.chatSkillReady()).toBe(true);

    const skills = join(root, 'skills');
    cpSync(defaultSkillsDir(), skills, { recursive: true });
    const unreadable = join(skills, 'vendor/exa/skills/exa-search/SKILL.md');
    chmodSync(unreadable, 0o000);
    try {
      const copy = service({ secrets: store, skillsDir: skills });
      const failed = await failure(copy.select({ search: 'exa' }), 'provider-selection');
      expect([failed.reason, failed.message]).toEqual([undefined, 'Skills could not be installed.']);
    } finally {
      chmodSync(unreadable, 0o644);
    }
  });
});

describe('Work view of the providers', () => {
  it('lists the Firecrawl folder only when the firecrawl CLI is on the Work PATH', async () => {
    const store = new MemorySecretStore();
    await store.set('deepseek', 'ds-key-00001');
    await store.set('firecrawl', 'fc-key-000001');
    let path = '';
    const svc = service({ secrets: store, workPath: () => path });
    await svc.select({ search: 'deepseek', firecrawl: true });
    let listed = await svc.list();
    expect(listed.providers[3]!.work).toEqual({ ready: false, missing: 'firecrawl-cli' });
    expect(listed.providers[1]!.work).toEqual({ ready: true, missing: null });
    expect(svc.workPluginDirs()).toEqual([join(paths.plugins, 'deepseek')]);
    expect(svc.workKeys()).toEqual({ DEEPSEEK_API_KEY: 'ds-key-00001' });
    const bin = join(root, 'bin');
    mkdirSync(bin);
    writeFileSync(join(bin, 'firecrawl'), '#!/bin/sh\n');
    chmodSync(join(bin, 'firecrawl'), 0o755);
    path = `/nonexistent:${bin}`;
    listed = await svc.list();
    expect(listed.providers[3]!.work).toEqual({ ready: true, missing: null });
    expect(svc.workPluginDirs()).toEqual([join(paths.plugins, 'deepseek'), join(paths.plugins, 'firecrawl')]);
    expect(svc.workKeys()).toEqual({ DEEPSEEK_API_KEY: 'ds-key-00001', FIRECRAWL_API_KEY: 'fc-key-000001' });
    expect(listed.providers[3]!.skill).toMatchObject({
      installed: true,
      names: ['firecrawl', 'firecrawl-search', 'firecrawl-scrape', 'firecrawl-crawl', 'firecrawl-map'],
      upstream: { repo: 'firecrawl/skills', license: 'ISC' },
    });
    expect(listed.providers[1]!.skill).toEqual({ installed: true, names: ['deepseek-search'], upstream: null });
    expect(listed.providers[1]!.keyEnv).toEqual(['DEEPSEEK_API_KEY']);
    expect(listed.chatSkill).toEqual({ installed: true });
  });

  it('re-verifies installed folders on every list: a changed file reads as not installed', async () => {
    const store = new MemorySecretStore();
    await store.set('exa', 'exa-key-0001');
    const svc = service({ secrets: store });
    await svc.select({ search: 'exa' });
    expect((await svc.list()).providers[0]!.skill.installed).toBe(true);
    writeFileSync(join(paths.plugins, 'exa', 'skills', 'exa-search', 'SKILL.md'), 'changed');
    expect((await svc.list()).providers[0]!.skill.installed).toBe(false);
    expect(svc.workPluginDirs()).toEqual([]);
  });
});

describe('a change and the agent hosts', () => {
  it('marks both hosts stale: an idle Chat host restarts and its conversation goes cold; a running turn keeps the old key', async () => {
    const holder: { svc: ProvidersService | null } = { svc: null };
    const urls = await providerServers();
    const fake = fakeService('chat-basic', {
      providers: (p) => {
        holder.svc = new ProvidersService({
          paths: p,
          secrets: new MemorySecretStore(),
          baseUrls: urls,
          fetch: (i, n) => fetch(i, n),
          workPath: () => '',
        });
        return holder.svc;
      },
      mutate: (s) => {
        s.prompts = [
          {
            match: 'slow',
            steps: [{ text: 'Thinking', chunk: 3 }, { sleepMs: 1500 }, { end: { stopReason: 'end_turn' } }],
          },
          { match: '', steps: [{ text: 'Hi', chunk: 2 }, { end: { stopReason: 'end_turn' } }] },
        ];
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'slow please' });
      const ends = turnEnds(fake.service, conversation.id);
      await until(() => fake.sent().some((m) => m.method === 'session/prompt'), 5_000);
      const svc = holder.svc!;
      await svc.putKey('exa', 'exa-key-0001');
      expect((await svc.list()).applyPending).toBe(true);
      expect((await fake.service.state()).providers).toMatchObject({ search: 'exa', applyPending: true });
      const starts = (): number => fake.records().filter((r) => r.kind === 'start').length;
      await until(() => ends.length === 1, 5_000);
      await until(() => !svc.summary().applyPending, 5_000);
      expect(starts()).toBe(1);
      await fake.service.prompt(conversation.id, 'hello again');
      await until(() => ends.length === 2, 5_000);
      expect(starts()).toBe(2);
      const load = fake.sent().filter((m) => m.method === 'session/load');
      expect(load).toHaveLength(1);
      const env = ((load[0]!.params as Json).mcpServers as Json[])[0]!.env as Array<{ name: string; value: string }>;
      expect(env.find((e) => e.name === 'MLX_WEB_SEARCH_PROVIDER')!.value).toBe('exa');
      expect(env.find((e) => e.name === 'MLX_WEB_SEARCH_KEY')!.value).toBe('exa-key-0001');
      const first = (fake.sent().find((m) => m.method === 'session/new')!.params as Json).mcpServers as Json[];
      expect(
        (first[0]!.env as Array<{ name: string; value: string }>).find((e) => e.name === 'MLX_WEB_SEARCH_PROVIDER')!
          .value,
      ).toBe('none');
      const argv = JSON.stringify(
        fake
          .records()
          .filter((r) => r.kind === 'start')
          .map((r) => r.argv),
      );
      expect(argv).not.toContain('exa-key-0001');
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('a key failure from a turn still on the replaced key does not mark the new key', async () => {
    const holder: { svc: ProvidersService | null } = { svc: null };
    const urls = await providerServers();
    const rejected = formatErrorText('key-rejected', 'exa');
    const webCall = {
      webCall: { tool_name: 'mlx_web__web_search', tool_input: { query: 'mlx' } },
      onAllow: { okayOutput: rejected },
    };
    const fake = fakeService('chat-basic', {
      providers: (p) => {
        holder.svc = new ProvidersService({
          paths: p,
          secrets: new MemorySecretStore(),
          baseUrls: urls,
          fetch: (i, n) => fetch(i, n),
          workPath: () => '',
        });
        return holder.svc;
      },
      mutate: (s) => {
        s.prompts = [
          { match: 'slow', steps: [{ sleepMs: 1500 }, webCall, { end: { stopReason: 'end_turn' } }] },
          { match: 'search', steps: [webCall, { end: { stopReason: 'end_turn' } }] },
          { match: '', steps: [{ text: 'Hi', chunk: 2 }, { end: { stopReason: 'end_turn' } }] },
        ];
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello' });
      const ends = turnEnds(fake.service, conversation.id);
      await until(() => ends.length === 1, 5_000);
      const svc = holder.svc!;
      await svc.putKey('exa', 'exa-key-0001');
      await until(() => !svc.summary().applyPending, 5_000);
      await fake.service.prompt(conversation.id, 'slow please');
      await until(() => fake.sent().filter((m) => m.method === 'session/prompt').length === 2, 5_000);
      expect((await svc.putKey('exa', 'exa-key-0002')).check).toBe('ok');
      await until(() => ends.length === 2, 5_000);
      await svc.settled();
      expect((await svc.list()).providers[0]!.check).toBe('ok');
      await until(() => !svc.summary().applyPending, 5_000);
      await fake.service.prompt(conversation.id, 'search now');
      await until(() => ends.length === 3, 5_000);
      await until(() => svc.summary().searchCheck === 'rejected', 5_000);
    } finally {
      await fake.close();
    }
  }, 30_000);
});

describe('a marker key never leaves the Control Panel', () => {
  let runtime: DashboardRuntime | null = null;

  afterEach(async () => {
    await runtime?.close();
    runtime = null;
  });

  function startRuntime(secrets: SecretStore, urls: Record<ProviderId, string>): DashboardRuntime {
    const base = join(root, 'rt');
    const scenarioPath = join(root, 'scenario.json');
    writeFileSync(
      scenarioPath,
      JSON.stringify({ ...readScenario('chat-basic'), recordTo: join(root, 'record.jsonl') }),
    );
    runtime = createDashboardRuntime({
      dbPath: ':memory:',
      tracesDir: join(base, 'traces'),
      modelsDir: join(base, 'models'),
      runtimeDir: join(base, 'runtime'),
      supervisorPath: join(base, 'supervisor.json'),
      agentHome: join(base, 'agent-home'),
      chat: {
        root: paths.root,
        homeDir: join(base, 'home'),
        agentCommand: { command: process.execPath, args: [FAKE_AGENT, scenarioPath] },
        secrets,
        providerBaseUrls: urls,
        readEndpoint: async () => ({
          version: 1,
          pid: process.pid,
          url: 'http://127.0.0.1:9',
          token: INFERENCE_TOKEN,
          models: [{ name: 'qwen3.5-4b', path: '/models/qwen3.5-4b' }],
        }),
        timings: { stopEofMs: 300, stopTermMs: 300 },
        flushMs: 5,
        stateCoalesceMs: 5,
      },
    });
    return runtime;
  }

  it('through every route, a provider that echoes it, a store and a fetch that throw it', async () => {
    const logged: string[] = [];
    for (const method of ['log', 'info', 'warn', 'error', 'debug'] as const) {
      vi.spyOn(console, method).mockImplementation((...args: unknown[]) => {
        logged.push(args.map(String).join(' '));
      });
    }
    const stderr = vi.spyOn(process.stderr, 'write');
    const stdout = vi.spyOn(process.stdout, 'write');
    const urls = await providerServers((id, r) =>
      keyOf(r) === `${MARKER}-bad`
        ? { status: 401, json: { error: `invalid key ${keyOf(r)}` } }
        : { json: { ...(CHECK_OK[id] as object), echo: keyOf(r) } },
    );
    const store = new CountingStore();
    const rt = startRuntime(store, urls);
    const replies: ApiResponse[] = [];
    const call = async (method: string, path: string, body?: unknown): Promise<ApiResponse> => {
      const reply = await rt.call({ method, path, ...(body === undefined ? {} : { body }) });
      replies.push(reply);
      return reply;
    };
    expect((await call('GET', '/api/providers')).ok).toBe(true);
    expect(await call('PUT', '/api/providers/exa/key', { key: MARKER })).toMatchObject({
      ok: true,
      body: { saved: true, check: 'ok' },
    });
    expect(await call('PUT', '/api/providers/firecrawl/key', { key: `${MARKER}-bad` })).toMatchObject({
      ok: true,
      body: { saved: false, check: 'rejected' },
    });
    expect(await call('PUT', '/api/providers/brave/key', { key: `${MARKER} x` })).toMatchObject({
      ok: false,
      code: 'E_BAD_REQUEST',
    });
    expect(await call('PUT', '/api/providers/brave/key', { key: MARKER, extra: MARKER })).toMatchObject({
      ok: false,
      code: 'E_BAD_REQUEST',
    });
    expect((await call('POST', '/api/providers/exa/validate')).ok).toBe(true);
    expect((await call('PUT', '/api/providers/selection', { search: 'exa', firecrawl: false })).ok).toBe(true);
    expect((await call('GET', '/api/chat/state')).ok).toBe(true);
    expect(await call('PUT', '/api/providers/google/key', { key: MARKER })).toMatchObject({
      ok: false,
      code: 'E_NOT_FOUND',
    });

    store.fail = `keychain refused ${MARKER} at /Users/someone/Library/Keychains`;
    expect(await call('PUT', '/api/providers/deepseek/key', { key: `${MARKER}-2` })).toMatchObject({
      ok: false,
      code: 'E_UNAVAILABLE',
      reason: 'keychain-unavailable',
      message: 'The Keychain cannot be reached.',
    });
    expect(await call('POST', '/api/providers/exa/validate')).toMatchObject({ ok: false, code: 'E_UNAVAILABLE' });
    expect(await call('DELETE', '/api/providers/exa/key')).toMatchObject({ ok: false, code: 'E_UNAVAILABLE' });
    store.fail = null;
    expect((await call('DELETE', '/api/providers/exa/key')).ok).toBe(true);

    const otherRoot = join(root, 'other');
    const throwing = service({
      paths: chatPaths(otherRoot),
      fetch: () => Promise.reject(new TypeError(`Invalid header value ${MARKER}`)),
      secrets: new MemorySecretStore(),
    });
    const offline = await throwing.putKey('exa', MARKER);
    expect([offline.saved, offline.check]).toEqual([true, 'unverified']);
    expect(JSON.stringify(offline)).not.toContain(MARKER);

    for (const reply of replies)
      expect([JSON.stringify(reply), containsMarker(JSON.stringify(reply))]).toEqual([JSON.stringify(reply), false]);
    expect(logged.filter(containsMarker)).toEqual([]);
    for (const spy of [stderr, stdout]) {
      expect(spy.mock.calls.map((c) => String(c[0])).filter(containsMarker)).toEqual([]);
    }
    const files = [...filesUnder(paths.root), ...filesUnder(otherRoot)];
    expect(files.length).toBeGreaterThan(0);
    expect(files.filter(([, text]) => containsMarker(text)).map(([path]) => path)).toEqual([]);
    expect(JSON.parse(readFileSync(paths.providers, 'utf8'))).toMatchObject({ version: 1 });
  }, 30_000);
});

describe('tests never reach the real Keychain', () => {
  it('every test that builds a runtime and calls a Chat or provider route pins the memory store', () => {
    const dir = join(import.meta.dirname);
    for (const name of readdirSync(dir).filter((n) => n.endsWith('.test.ts'))) {
      const text = readFileSync(join(dir, name), 'utf8');
      if (!text.includes('createDashboardRuntime(')) continue;
      const calls = ["', '/api/chat", "', '/api/providers", "path: '/api/chat", "path: '/api/providers", '`/api/chat'];
      if (!calls.some((c) => text.includes(c))) continue;
      expect([name, text.includes('MemorySecretStore') || text.includes('CountingStore')]).toEqual([name, true]);
    }
  });
});
