/** @vitest-environment happy-dom */

import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import { PROVIDERS } from '../ui/src/lib/providers-strings.js';
import type { ProviderId, ProviderState, ProvidersResponse } from '../ui/src/lib/providers-types.js';
import {
  BILLING_PAGES,
  chatLines,
  checkKeyFormat,
  fill,
  KEY_PAGES,
  progressOf,
  workParts,
  workSentence,
} from '../ui/src/lib/providers-view.js';
import { toast } from '../ui/src/lib/toast.js';
import Providers from '../ui/src/pages/providers.js';
import {
  accessibleName,
  allByRole,
  byRole,
  deferred,
  renderPage,
  type RenderedPage,
  sequence,
  STUB_FAILURE,
  stubApi,
  stubError,
  waitFor,
} from './render.js';

/** A key no test may find anywhere after the one PUT that carries it. */
const MARKER = 'mk-live-9F3cA7e21Bd4X0qZ';

const NAMES: Record<ProviderId, string[]> = {
  exa: ['exa-search', 'exa-contents'],
  deepseek: ['deepseek-search'],
  brave: ['web-search', 'llm-context', 'news-search'],
  firecrawl: ['firecrawl', 'firecrawl-search', 'firecrawl-scrape', 'firecrawl-crawl', 'firecrawl-map'],
};
const UPSTREAM: Record<ProviderId, ProviderState['skill']['upstream']> = {
  exa: { repo: 'exa-labs/agent-skills', commit: 'a0c2df86b1e2c3d4e5f6a7b8c9d0e1f2a3b4c5d6', license: 'MIT' },
  deepseek: null,
  brave: { repo: 'brave/brave-search-skills', commit: '62793e0c1e2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e', license: 'MIT' },
  firecrawl: { repo: 'firecrawl/skills', commit: '667d3f7f1e2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e', license: 'ISC' },
};

function provider(id: ProviderId, over: Partial<ProviderState> = {}): ProviderState {
  const configured = over.configured ?? false;
  const selected = over.selected ?? false;
  return {
    id,
    role: id === 'firecrawl' ? 'crawl' : 'search',
    configured,
    check: configured ? 'ok' : null,
    checkedAt: configured ? Date.now() - 120_000 : null,
    selected,
    keyEnv: [],
    skill: { installed: selected, names: NAMES[id], upstream: UPSTREAM[id] },
    work: { ready: selected, missing: null },
    facts: null,
    ...over,
  };
}

interface Setup {
  search?: ProvidersResponse['search'];
  firecrawl?: boolean;
  configured?: ProviderId[];
  over?: Partial<Record<ProviderId, Partial<ProviderState>>>;
  keychain?: ProvidersResponse['keychain'];
  applyPending?: boolean;
}

function response(setup: Setup = {}): ProvidersResponse {
  const search = setup.search ?? null;
  const firecrawl = setup.firecrawl ?? false;
  const configured = new Set(setup.configured ?? []);
  if (search !== null) configured.add(search);
  if (firecrawl) configured.add('firecrawl');
  const ids: ProviderId[] = ['exa', 'deepseek', 'brave', 'firecrawl'];
  return {
    search,
    firecrawl,
    fetch: firecrawl ? 'firecrawl' : search,
    providers: ids.map((id) =>
      provider(id, {
        configured: configured.has(id),
        selected: id === 'firecrawl' ? firecrawl : search === id,
        ...setup.over?.[id],
      }),
    ),
    chatSkill: { installed: true },
    keychain: setup.keychain ?? 'ok',
    applyPending: setup.applyPending ?? false,
    vanished: null,
  };
}

let page: RenderedPage | undefined;
let dispose: (() => void) | undefined;
afterEach(() => {
  page?.unmount();
  dispose?.();
  page = undefined;
  dispose = undefined;
  vi.restoreAllMocks();
});

async function open(routes: Record<string, unknown>, until: string, calls: ApiCall[] = []): Promise<RenderedPage> {
  dispose = stubApi(routes, { onCall: (call) => calls.push(call) });
  page = await renderPage(createElement(MemoryRouter, null, createElement(Providers)), (text) => text.includes(until));
  return page;
}

function card(name: string): HTMLElement {
  const found = page!.container.querySelector<HTMLElement>(`section[aria-label="${name}"]`);
  if (found === null) throw new Error(`no card ${name}`);
  return found;
}

/** The text a user sees: a swapped-out form or key row stays laid out, but hidden and inert. */
function shownText(scope: Element): string {
  const copy = scope.cloneNode(true) as Element;
  for (const hidden of copy.querySelectorAll('[inert]')) hidden.remove();
  return copy.textContent ?? '';
}

/** Buttons a user can reach: a swapped-out form or key row stays laid out, but hidden and inert. */
function reachable(scope: ParentNode): HTMLButtonElement[] {
  return [...scope.querySelectorAll('button')].filter((b) => b.closest('[inert]') === null);
}

function button(scope: ParentNode, label: string): HTMLButtonElement {
  const found = reachable(scope).find((b) => accessibleName(b) === label);
  if (found === undefined) throw new Error(`no button ${label}`);
  return found;
}

function maybeButton(scope: ParentNode, label: string): HTMLButtonElement | undefined {
  return reachable(scope).find((b) => accessibleName(b) === label);
}

async function typeInto(input: HTMLInputElement, value: string): Promise<void> {
  await act(async () => {
    Reflect.set(HTMLInputElement.prototype, 'value', value, input);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

async function click(element: HTMLElement): Promise<void> {
  await act(async () => {
    element.click();
  });
}

async function key(element: HTMLElement, name: string): Promise<void> {
  await act(async () => {
    element.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true }));
  });
}

function keyField(scope: HTMLElement): HTMLInputElement {
  const input = scope.querySelector<HTMLInputElement>('input[autocomplete="off"]');
  if (input === null) throw new Error('no key field');
  return input;
}

/** Every 8-character window of the marker key. */
function windows(): string[] {
  const out: string[] = [];
  for (let i = 0; i + 8 <= MARKER.length; i += 1) out.push(MARKER.slice(i, i + 8));
  return out;
}

function leaks(text: string): string[] {
  return windows().filter((w) => text.includes(w));
}

const radios = (): HTMLElement[] => allByRole(page!.container, 'radio');
const radioTitle = (radio: Element): string => radio.children[1]?.children[0]?.textContent ?? '';
const radioNames = (): string[] => radios().map(radioTitle);
const checkedRadio = (): string | undefined => {
  const checked = radios().find((r) => r.getAttribute('aria-checked') === 'true');
  return checked === undefined ? undefined : radioTitle(checked);
};

describe('Now sentences', () => {
  it('reads every row of the Chat table', () => {
    const chat = (setup: Setup): string => chatLines(response(setup))[0]!.text;
    expect(chat({ search: 'brave', firecrawl: true })).toBe('Web search via Brave, pages via Firecrawl.');
    expect(chat({ search: 'exa' })).toBe('Web search and pages via Exa.');
    expect(chat({ search: 'deepseek' })).toBe('Web search and pages via DeepSeek.');
    expect(chat({ search: 'brave' })).toBe('Web search and page excerpts via Brave.');
    expect(chat({ firecrawl: true })).toBe('No web search. Pages via Firecrawl when you give an address.');
    expect(chat({})).toBe('Web search is off.');
  });

  it('adds a line for a selected provider whose key failed its check', () => {
    expect(chatLines(response({ search: 'brave', over: { brave: { check: 'rejected' } } }))[1]).toEqual({
      text: 'Brave rejected the key. Searches will fail.',
      tone: 'danger',
    });
    expect(chatLines(response({ search: 'brave', over: { brave: { check: 'no-credit' } } }))[1]).toEqual({
      text: 'The Brave account has no credit.',
      tone: 'warning',
    });
  });

  it('reads every row of the Work table', () => {
    const work = (setup: Setup): string => workSentence(workParts(response(setup)));
    expect(work({ search: 'brave', firecrawl: true })).toBe('3 Brave skills, 5 Firecrawl skills.');
    expect(work({ search: 'brave' })).toBe('3 Brave skills.');
    expect(work({ search: 'deepseek' })).toBe('1 DeepSeek skill.');
    expect(work({ firecrawl: true })).toBe('5 Firecrawl skills.');
    expect(work({})).toBe('No web skills.');
    const missing = workParts(
      response({
        search: 'brave',
        over: { brave: { skill: { installed: false, names: NAMES.brave, upstream: null } } },
      }),
    );
    expect(missing).toEqual([{ text: 'Brave skills not installed', missing: true }]);
  });

  it('checks a key format with a character loop', () => {
    expect(checkKeyFormat('  abcdefgh  ')).toBe('abcdefgh');
    expect(checkKeyFormat('short')).toBeNull();
    expect(checkKeyFormat('has space inside')).toBeNull();
    expect(checkKeyFormat('naïve-key-1234')).toBeNull();
    expect(checkKeyFormat('x'.repeat(513))).toBeNull();
  });

  it('links every key page over https on the provider’s own domain', () => {
    const domains: Record<ProviderId, string> = {
      exa: 'exa.ai',
      deepseek: 'deepseek.com',
      brave: 'brave.com',
      firecrawl: 'firecrawl.dev',
    };
    for (const id of ['exa', 'deepseek', 'brave', 'firecrawl'] as const) {
      for (const page of [KEY_PAGES[id], BILLING_PAGES[id]]) {
        const url = new URL(page);
        expect(url.protocol).toBe('https:');
        expect(url.hostname === domains[id] || url.hostname.endsWith(`.${domains[id]}`)).toBe(true);
      }
    }
  });
});

describe('Web Search page', () => {
  it('with no keys: four "No key" cards, no radios, the Firecrawl switch off, the always-on notice', async () => {
    await open({ '/providers': response() }, PROVIDERS.noneConfigured);
    const text = page!.text();
    expect(page!.container.querySelector('h1')?.textContent).toBe('Web Search');
    expect(text).toContain(PROVIDERS.chatOff);
    expect(text).toContain(PROVIDERS.workNone);
    expect(radios()).toHaveLength(0);
    for (const name of ['Exa', 'DeepSeek', 'Brave', 'Firecrawl']) {
      expect(card(name).textContent).toContain(PROVIDERS.badgeNoKey);
      expect(keyField(card(name)).type).toBe('password');
    }
    const sw = byRole(page!.container, 'switch');
    expect(sw?.getAttribute('aria-checked')).toBe('false');
    expect((sw as HTMLButtonElement).disabled).toBe(true);
    expect(text).toContain(PROVIDERS.firecrawlNoKey);
    expect(text).toContain(PROVIDERS.leavesTitle);
    expect(text).not.toContain(PROVIDERS.historyTitle);
    expect(shownText(card('Brave'))).toContain(PROVIDERS.saveHelpBrave);
    expect(shownText(card('Exa'))).toContain('Saving checks the key with Exa.');
  });

  it('shows the loading panels while the first body is out, and a retry when it fails', async () => {
    const held = deferred(response());
    await open({ '/providers': held.body }, 'Loading Web Search settings');
    expect(page!.container.querySelector('[aria-busy="true"]')).not.toBeNull();
    held.release();
    await waitFor(() => page!.text().includes(PROVIDERS.noneConfigured));
    page!.unmount();
    dispose!();
    await open({ '/providers': STUB_FAILURE }, PROVIDERS.loadError);
    expect(maybeButton(page!.container, PROVIDERS.tryAgain)).toBeDefined();
  });

  it('saves a key: one PUT carries it, then "Key saved", the proof row, and no part of the key anywhere', async () => {
    const calls: ApiCall[] = [];
    const saved = response({ search: 'exa' });
    const success = vi.spyOn(toast, 'success');
    const error = vi.spyOn(toast, 'error');
    await open(
      {
        '/providers': sequence(response(), saved),
        '/providers/exa/key': { saved: true, check: 'ok', providers: saved },
      },
      PROVIDERS.noneConfigured,
      calls,
    );
    const exa = card('Exa');
    const field = keyField(exa);
    const reveal = button(exa, `${PROVIDERS.showKey} for Exa`);
    await click(reveal);
    expect(field.type).toBe('text');
    expect(button(exa, `${PROVIDERS.hideKey} for Exa`).getAttribute('aria-pressed')).toBe('true');
    await typeInto(field, MARKER);
    await key(field, 'Enter');
    await waitFor(() => shownText(card('Exa')).includes(PROVIDERS.keySaved) === true);

    const puts = calls.filter((call) => call.method === 'PUT');
    expect(puts).toEqual([{ method: 'PUT', path: '/api/providers/exa/key', body: { key: MARKER } }]);
    const others = calls.filter((call) => call !== puts[0]);
    expect(leaks(JSON.stringify(others))).toEqual([]);

    const after = card('Exa');
    expect(after.querySelector('[data-key-form]')?.closest('[inert]')).not.toBeNull();
    expect(keyField(after).value).toBe('');
    expect(shownText(after)).toContain(PROVIDERS.badgeValid);
    expect(shownText(after)).toContain('exa-search · exa-contents');
    expect(shownText(after)).toContain('exa-labs/agent-skills @ a0c2df8 · MIT');
    expect(shownText(after)).toContain('shell commands, MCP servers and hooks can read this key');
    expect(page!.text()).toContain('Installed: 2 skills');
    expect(page!.container.querySelector('[role="status"][aria-live="polite"]')?.textContent).toBe(
      'Exa selected. 2 skills installed.',
    );
    expect(checkedRadio()).toBe('Exa');
    expect(page!.text()).toContain('Web search and pages via Exa.');
    expect(page!.text()).toContain(PROVIDERS.historyTitle);
    const cacheLink = [...page!.container.querySelectorAll('a')].find((a) => a.textContent === PROVIDERS.cacheLink);
    expect(cacheLink?.getAttribute('href')).toBe('/cache');

    expect(leaks(document.body.innerHTML)).toEqual([]);
    expect(leaks(JSON.stringify([...success.mock.calls, ...error.mock.calls]))).toEqual([]);
  });

  it('a first key whose skills did not install is saved but not chosen; choosing it says reinstall', async () => {
    const notInstalled = response({
      configured: ['exa'],
      over: { exa: { check: 'unverified', skill: { installed: false, names: NAMES.exa, upstream: UPSTREAM.exa } } },
    });
    await open(
      {
        '/providers': sequence(response(), notInstalled),
        '/providers/exa/key': { saved: true, check: 'unverified', providers: notInstalled },
        '/providers/selection': stubError(
          'E_UNAVAILABLE',
          "The app's skill files are damaged. Reinstall the app.",
          'skill-bundle-corrupt',
        ),
      },
      PROVIDERS.noneConfigured,
    );
    const field = keyField(card('Exa'));
    await typeInto(field, MARKER);
    await key(field, 'Enter');
    await waitFor(() => shownText(card('Exa')).includes(PROVIDERS.keySaved));
    expect(shownText(card('Exa'))).toContain(PROVIDERS.badgeUnverified);
    expect(shownText(card('Exa'))).not.toContain(PROVIDERS.skills);
    expect(checkedRadio()).toBe('Off');
    expect(page!.text()).toContain(PROVIDERS.chatOff);
    expect(page!.text()).not.toContain('Installed:');
    await click(radios().find((r) => radioTitle(r) === 'Exa')!);
    await waitFor(() => page!.text().includes(PROVIDERS.installFailedTitle));
    expect(page!.text()).toContain(PROVIDERS.installCorrupt);
    expect(checkedRadio()).toBe('Off');
  });

  it('refuses a key that does not look like one, in the page, with no call', async () => {
    const calls: ApiCall[] = [];
    await open({ '/providers': response() }, PROVIDERS.noneConfigured, calls);
    const exa = card('Exa');
    await typeInto(keyField(exa), 'not a key');
    await click(button(exa, `${PROVIDERS.save} for Exa`));
    expect(exa.querySelector('[role="alert"]')?.textContent).toBe(PROVIDERS.badFormat);
    expect(calls.filter((call) => call.method !== 'GET')).toEqual([]);
  });

  it('a key rejected on save is not saved; the field keeps its text and says why', async () => {
    await open(
      { '/providers': response(), '/providers/brave/key': { saved: false, check: 'rejected', providers: response() } },
      PROVIDERS.noneConfigured,
    );
    const brave = card('Brave');
    await typeInto(keyField(brave), MARKER);
    await click(button(brave, `${PROVIDERS.save} for Brave`));
    await waitFor(() => brave.querySelector('[role="alert"]') !== null);
    expect(brave.querySelector('[role="alert"]')?.textContent).toBe('Brave rejected this key. It was not saved.');
    expect(keyField(brave).value).toBe(MARKER);
    expect(shownText(brave)).toContain(PROVIDERS.badgeNoKey);
  });

  it('a rejected replacement says the earlier key is still in use', async () => {
    const before = response({ search: 'exa' });
    await open(
      { '/providers': before, '/providers/exa/key': { saved: false, check: 'rejected', providers: before } },
      PROVIDERS.keySaved,
    );
    const exa = card('Exa');
    await click(button(exa, PROVIDERS.replace));
    expect(shownText(exa)).toContain(PROVIDERS.replaceHelp);
    expect(maybeButton(exa, PROVIDERS.cancel)).toBeDefined();
    await typeInto(keyField(exa), MARKER);
    await click(button(exa, `${PROVIDERS.save} for Exa`));
    await waitFor(() => exa.querySelector('[role="alert"]') !== null);
    expect(exa.querySelector('[role="alert"]')?.textContent).toBe(
      'Exa rejected this key. It was not saved. The earlier key is still in use.',
    );
    await click(button(exa, PROVIDERS.cancel));
    expect(exa.querySelector('[data-key-form]')?.closest('[inert]')).not.toBeNull();
    expect(keyField(exa).value).toBe('');
    expect(shownText(exa)).toContain(PROVIDERS.keySaved);
  });

  it('a Keychain that refuses the write keeps the text and says the key was not saved', async () => {
    await open(
      {
        '/providers': response(),
        '/providers/exa/key': stubError('E_UNAVAILABLE', 'The Keychain cannot be reached.', 'keychain-unavailable'),
      },
      PROVIDERS.noneConfigured,
    );
    const exa = card('Exa');
    await typeInto(keyField(exa), MARKER);
    await click(button(exa, `${PROVIDERS.save} for Exa`));
    await waitFor(() => exa.querySelector('[role="alert"]') !== null);
    expect(exa.querySelector('[role="alert"]')?.textContent).toBe(PROVIDERS.keychainRefused);
    expect(keyField(exa).value).toBe(MARKER);
  });

  it('shows "Checking…" while a save is out', async () => {
    const held = deferred({ saved: true, check: 'ok', providers: response({ search: 'exa' }) });
    await open(
      { '/providers': sequence(response(), response({ search: 'exa' })), '/providers/exa/key': held.body },
      PROVIDERS.noneConfigured,
    );
    const exa = card('Exa');
    await typeInto(keyField(exa), MARKER);
    await click(button(exa, `${PROVIDERS.save} for Exa`));
    await waitFor(() => shownText(exa).includes(PROVIDERS.badgeChecking) === true);
    expect(keyField(exa).readOnly).toBe(true);
    held.release();
    await waitFor(() => shownText(exa).includes(PROVIDERS.keySaved) === true);
  });

  it('draws every saved-key state with its line and its buttons', async () => {
    const data = response({
      search: 'deepseek',
      firecrawl: true,
      configured: ['exa', 'brave'],
      over: {
        exa: { check: 'unverified' },
        deepseek: { check: 'ok', facts: { balances: [{ currency: 'USD', amount: 12.5 }] } },
        brave: { check: 'rejected' },
        firecrawl: {
          check: 'ok',
          facts: { remainingCredits: 742, planCredits: 1000, periodEnd: new Date(2026, 11, 1).getTime() },
        },
      },
    });
    await open({ '/providers': data }, PROVIDERS.keySaved);

    const exa = card('Exa');
    expect(shownText(exa)).toContain(PROVIDERS.badgeUnverified);
    expect(shownText(exa)).toContain('Saved. No connection to Exa.');
    expect(maybeButton(exa, PROVIDERS.checkAgain)).toBeDefined();

    const deepseek = card('DeepSeek');
    expect(shownText(deepseek)).toContain(PROVIDERS.badgeValid);
    expect(shownText(deepseek)).toContain('Checked 2m ago');
    expect(shownText(deepseek)).toContain('Balance 12.50 USD');
    expect(shownText(deepseek)).toContain('deepseek-search');
    expect(shownText(deepseek)).toContain(PROVIDERS.skillsOwn);
    expect(reachable(deepseek).map((b) => b.textContent)).toEqual(
      expect.arrayContaining([PROVIDERS.check, PROVIDERS.replace, PROVIDERS.remove]),
    );

    const brave = card('Brave');
    expect(shownText(brave)).toContain(PROVIDERS.badgeRejected);
    expect(shownText(brave)).toContain('Brave rejected the key.');
    const order = reachable(brave).map((b) => b.textContent);
    expect(order.indexOf(PROVIDERS.replace)).toBeLessThan(order.indexOf(PROVIDERS.checkAgain));

    const fire = card('Firecrawl');
    expect(shownText(fire)).toContain('742 of 1,000 credits left · renews Dec 1');
    expect(shownText(fire)).toContain(PROVIDERS.workKeyNotice);
    expect(shownText(card('Brave'))).not.toContain(PROVIDERS.workKeyNotice);

    expect(radioNames()).toEqual(['Exa', 'DeepSeek', 'Brave', 'Off']);
    expect(checkedRadio()).toBe('DeepSeek');
    expect(radios()[2]?.textContent).toContain(PROVIDERS.badgeRejected);
  });

  it('a key with no credit links the provider dashboard by its address; Brave says its check is billed', async () => {
    await open(
      { '/providers': response({ search: 'brave', over: { brave: { check: 'no-credit' } } }) },
      PROVIDERS.keySaved,
    );
    const brave = card('Brave');
    expect(shownText(brave)).toContain(PROVIDERS.badgeNoCredit);
    expect(shownText(brave)).toContain('The Brave account has no credit. Add credit at ');
    const link = [...brave.querySelectorAll('a')].find((a) => a.textContent === BILLING_PAGES.brave);
    expect(link?.getAttribute('target')).toBe('_blank');
    expect(page!.text()).toContain('The Brave account has no credit.');
    expect(page!.text()).toContain(PROVIDERS.historyBrave.slice(0, 40));
  });

  it('Brave valid shows the billed-request fact', async () => {
    await open({ '/providers': response({ search: 'brave' }) }, PROVIDERS.keySaved);
    expect(shownText(card('Brave'))).toContain(PROVIDERS.factsBrave);
  });

  it('shows "Checking…" during a check and the new state after it', async () => {
    const held = deferred({
      check: 'unverified',
      providers: response({ search: 'exa', over: { exa: { check: 'unverified' } } }),
    });
    const calls: ApiCall[] = [];
    const unchecked = response({ search: 'exa', over: { exa: { check: 'unverified' } } });
    await open(
      { '/providers': sequence(response({ search: 'exa' }), unchecked), '/providers/exa/validate': held.body },
      PROVIDERS.keySaved,
      calls,
    );
    await click(button(card('Exa'), PROVIDERS.check));
    await waitFor(() => shownText(card('Exa')).includes(PROVIDERS.badgeChecking) === true);
    held.release();
    await waitFor(() => shownText(card('Exa')).includes('Saved. No connection to Exa.') === true);
    expect(calls.some((call) => call.method === 'POST' && call.path === '/api/providers/exa/validate')).toBe(true);
  });

  it('the radio group: arrow keys move focus and never select; Space selects and installs', async () => {
    const calls: ApiCall[] = [];
    const before = response({ search: 'exa', configured: ['brave'] });
    const after = response({ search: 'brave', configured: ['exa'] });
    const held = deferred(after);
    await open({ '/providers': before, '/providers/selection': held.body }, PROVIDERS.keySaved, calls);
    expect(radioNames()).toEqual(['Exa', 'Brave', 'Off']);
    const group = byRole(page!.container, 'radiogroup', PROVIDERS.searchProvider)!;
    expect(group).not.toBeNull();
    const [exa, brave] = radios();
    expect(exa!.tabIndex).toBe(0);
    expect(brave!.tabIndex).toBe(-1);
    act(() => exa!.focus());
    await key(exa!, 'ArrowDown');
    expect(document.activeElement).toBe(brave);
    await key(brave!, 'End');
    expect(document.activeElement?.textContent).toContain('Off');
    await key(document.activeElement as HTMLElement, 'Home');
    expect(document.activeElement).toBe(exa);
    expect(calls.filter((call) => call.method !== 'GET')).toEqual([]);

    await key(brave!, ' ');
    await waitFor(() => page!.text().includes(PROVIDERS.installing));
    expect(byRole(page!.container, 'switch')?.hasAttribute('disabled')).toBe(true);
    expect(calls.filter((call) => call.method === 'PUT')).toEqual([
      { method: 'PUT', path: '/api/providers/selection', body: { search: 'brave' } },
    ]);
    held.release();
    await waitFor(() => checkedRadio() === 'Brave');
    expect(page!.text()).toContain('Installed: 3 Brave skills. Removed: 2 Exa skills.');
    for (const name of NAMES.brave) expect(page!.text()).toContain(name);
    expect(page!.container.querySelector('[role="status"][aria-live="polite"]')?.textContent).toBe(
      'Brave selected. 3 skills installed.',
    );
  });

  it('choosing Off removes the skills and says web search is off', async () => {
    await open(
      { '/providers': response({ search: 'exa' }), '/providers/selection': response({ configured: ['exa'] }) },
      PROVIDERS.keySaved,
    );
    const off = radios().find((r) => r.textContent?.includes(PROVIDERS.off))!;
    await click(off);
    await waitFor(() => page!.text().includes('Removed: 2 Exa skills.'));
    expect(checkedRadio()).toBe('Off');
    expect(page!.container.querySelector('[role="status"][aria-live="polite"]')?.textContent).toBe(PROVIDERS.offStatus);
  });

  it('the Firecrawl switch installs its skills; a missing CLI changes its words and shows the install line', async () => {
    const calls: ApiCall[] = [];
    const missing = { work: { ready: false, missing: 'firecrawl-cli' as const } };
    const before = response({ configured: ['firecrawl'], over: { firecrawl: missing } });
    const after = response({ firecrawl: true, over: { firecrawl: missing } });
    await open({ '/providers': before, '/providers/selection': after }, PROVIDERS.keySaved, calls);
    expect(page!.text()).toContain(PROVIDERS.firecrawlOff);
    const fire = card('Firecrawl');
    expect(shownText(fire)).not.toContain(PROVIDERS.firecrawlCli);
    await click(byRole(page!.container, 'switch')!);
    await waitFor(() => page!.text().includes('Installed: 5 Firecrawl skills'));
    expect(calls.filter((call) => call.method === 'PUT')[0]?.body).toEqual({ firecrawl: true });
    expect(page!.text()).toContain(PROVIDERS.firecrawlOnNoCli);
    expect(shownText(card('Firecrawl'))).toContain(PROVIDERS.firecrawlCli);
    expect(shownText(card('Firecrawl'))).toContain(PROVIDERS.firecrawlCliCommand);
    expect(shownText(card('Firecrawl'))).toContain(PROVIDERS.firecrawlCliChat);
    expect(shownText(card('Firecrawl'))).not.toContain(PROVIDERS.workKeyNotice);
    expect(page!.text()).toContain(`Work${PROVIDERS.workNoCli}.`);
    expect(byRole(page!.container, 'switch')?.getAttribute('aria-checked')).toBe('true');
  });

  it('removing the selected key: the confirm says so, then Off, the removed line and a toast', async () => {
    const success = vi.spyOn(toast, 'success');
    await open(
      { '/providers': sequence(response({ search: 'exa' }), response()), '/providers/exa/key': response() },
      PROVIDERS.keySaved,
    );
    await click(button(card('Exa'), PROVIDERS.remove));
    const dialog = await (async () => {
      await waitFor(() => document.querySelector('[role="dialog"]') !== null);
      return document.querySelector<HTMLElement>('[role="dialog"]')!;
    })();
    expect(dialog.textContent).toContain('Remove the Exa key?');
    expect(dialog.textContent).toContain(
      'The key is deleted from the Keychain. Exa is turned off for Chat and Work, and its skills are removed.',
    );
    await click(button(dialog, PROVIDERS.removeConfirm));
    await waitFor(() => page!.text().includes('Removed: 2 Exa skills.'));
    await waitFor(() => success.mock.calls.length > 0);
    expect(success.mock.calls[0]?.[0]).toBe('Exa removed. Web search is off.');
    expect(page!.text()).toContain(PROVIDERS.noneConfigured);
    expect(page!.text()).not.toContain(fill(PROVIDERS.vanishedTitle, { provider: 'Exa' }));
    await waitFor(() => document.activeElement === keyField(card('Exa')));
  });

  it('Cancel in the Replace form gives focus back to Replace; the form and the key row share one cell', async () => {
    await open({ '/providers': response({ search: 'exa' }) }, PROVIDERS.keySaved);
    const exa = card('Exa');
    const cells = exa.querySelectorAll('[data-key-form], [data-key-row]');
    expect(cells).toHaveLength(2);
    expect(cells[0]!.parentElement!.parentElement).toBe(cells[1]!.parentElement!.parentElement!.parentElement);
    await click(button(exa, PROVIDERS.replace));
    expect(document.activeElement).toBe(keyField(exa));
    await click(button(exa, PROVIDERS.cancel));
    await waitFor(() => document.activeElement === button(exa, PROVIDERS.replace));
  });

  it('one DeepSeek skill reads in the singular', async () => {
    const after = response({ search: 'deepseek', configured: ['exa'] });
    await open(
      { '/providers': response({ search: 'exa', configured: ['deepseek'] }), '/providers/selection': after },
      PROVIDERS.keySaved,
    );
    await click(radios()[1]!);
    await waitFor(() => checkedRadio() === 'DeepSeek');
    expect(page!.text()).toContain('Installed: 1 DeepSeek skill. Removed: 2 Exa skills.');
    expect(page!.container.querySelector('[role="status"][aria-live="polite"]')?.textContent).toBe(
      'DeepSeek selected. 1 skill installed.',
    );
    expect(progressOf(after, response({ search: 'exa', configured: ['deepseek'] }))?.text).toBe(
      'Installed: 2 Exa skills. Removed: 1 DeepSeek skill.',
    );
    expect(progressOf(after, response({ configured: ['exa', 'deepseek'] }))?.text).toBe('Removed: 1 DeepSeek skill.');
  });

  it('a key that was gone at app start shows the "key not found" alert from the first body', async () => {
    await open({ '/providers': { ...response(), vanished: 'exa' } }, PROVIDERS.noneConfigured);
    expect(page!.text()).toContain('Exa key not found');
  });

  it('the Keychain cannot be reached: one alert, and every control is off', async () => {
    await open({ '/providers': response({ search: 'exa', keychain: 'unavailable' }) }, PROVIDERS.keychainTitle);
    expect(page!.text()).toContain(PROVIDERS.keychainText);
    expect(maybeButton(page!.container, PROVIDERS.tryAgain)).toBeDefined();
    for (const label of [PROVIDERS.check, PROVIDERS.replace, PROVIDERS.remove, PROVIDERS.save]) {
      for (const b of page!.container.querySelectorAll('section[aria-label] button')) {
        if (b.textContent?.startsWith(label) !== true) continue;
        expect((b as HTMLButtonElement).disabled || b.getAttribute('aria-disabled') === 'true').toBe(true);
      }
    }
    expect(byRole(page!.container, 'radiogroup')?.getAttribute('aria-disabled')).toBe('true');
    expect(byRole(page!.container, 'switch')?.hasAttribute('disabled')).toBe(true);
    expect(button(page!.container, PROVIDERS.checkAll).disabled).toBe(true);
  });

  it('a damaged skill bundle: the alert says reinstall, nothing changed, no retry', async () => {
    await open(
      {
        '/providers': response({ search: 'exa', configured: ['brave'] }),
        '/providers/selection': stubError(
          'E_UNAVAILABLE',
          "The app's skill files are damaged. Reinstall the app.",
          'skill-bundle-corrupt',
        ),
      },
      PROVIDERS.keySaved,
    );
    await click(radios()[1]!);
    await waitFor(() => page!.text().includes(PROVIDERS.installFailedTitle));
    expect(page!.text()).toContain(PROVIDERS.installCorrupt);
    expect(checkedRadio()).toBe('Exa');
  });

  it('any other install failure offers Try again, which sends the same selection', async () => {
    const calls: ApiCall[] = [];
    await open(
      {
        '/providers': response({ search: 'exa', configured: ['brave'] }),
        '/providers/selection': sequence(
          stubError('E_UNAVAILABLE', 'Skills could not be installed.'),
          response({ search: 'brave', configured: ['exa'] }),
        ),
      },
      PROVIDERS.keySaved,
      calls,
    );
    await click(radios()[1]!);
    await waitFor(() => page!.text().includes(PROVIDERS.installOther));
    await click(button(page!.container, PROVIDERS.tryAgain));
    await waitFor(() => checkedRadio() === 'Brave');
    expect(calls.filter((call) => call.method === 'PUT').map((call) => call.body)).toEqual([
      { search: 'brave' },
      { search: 'brave' },
    ]);
    expect(page!.text()).not.toContain(PROVIDERS.installFailedTitle);
  });

  it('selecting a provider whose key is gone (a race) toasts "Add a key … first."', async () => {
    const error = vi.spyOn(toast, 'error');
    await open(
      {
        '/providers': response({ search: 'exa', configured: ['brave'] }),
        '/providers/selection': stubError('E_CONFLICT', 'Add a key for Brave first.'),
      },
      PROVIDERS.keySaved,
    );
    await click(radios()[1]!);
    await waitFor(() => error.mock.calls.length > 0);
    expect(error.mock.calls[0]?.[0]).toBe('Add a key for Brave first.');
  });

  it('a check that finds the selected key gone shows the "key not found" alert', async () => {
    await open(
      {
        '/providers': sequence(response({ search: 'exa' }), response()),
        '/providers/exa/validate': { check: null, providers: response() },
      },
      PROVIDERS.keySaved,
    );
    await click(button(card('Exa'), PROVIDERS.check));
    await waitFor(() => page!.text().includes('Exa key not found'));
    expect(page!.text()).toContain(PROVIDERS.vanishedText);
    expect(shownText(card('Exa'))).toContain(PROVIDERS.badgeNoKey);
  });

  it('a change waiting for running answers adds the third Now row', async () => {
    await open({ '/providers': response({ search: 'exa', applyPending: true }) }, PROVIDERS.pending);
  });

  it('a selected bundle that is not installed reads as missing in the Work sentence', async () => {
    await open(
      {
        '/providers': response({
          search: 'brave',
          over: { brave: { skill: { installed: false, names: NAMES.brave, upstream: UPSTREAM.brave } } },
        }),
      },
      'Brave skills not installed',
    );
    expect(shownText(card('Brave'))).not.toContain(PROVIDERS.skills);
  });

  it('"Check all keys" checks each configured provider, one after another', async () => {
    const calls: ApiCall[] = [];
    const data = response({ search: 'exa', firecrawl: true });
    await open(
      {
        '/providers': data,
        '/providers/exa/validate': { check: 'ok', providers: data },
        '/providers/firecrawl/validate': { check: 'ok', providers: data },
      },
      PROVIDERS.keySaved,
      calls,
    );
    const checkAll = button(page!.container, PROVIDERS.checkAll);
    await click(checkAll);
    expect([checkAll.disabled, checkAll.getAttribute('aria-disabled')]).toEqual([false, 'true']);
    await waitFor(() => calls.filter((call) => call.method === 'POST').length === 2);
    expect(calls.filter((call) => call.method === 'POST').map((call) => call.path)).toEqual([
      '/api/providers/exa/validate',
      '/api/providers/firecrawl/validate',
    ]);
  });
});
