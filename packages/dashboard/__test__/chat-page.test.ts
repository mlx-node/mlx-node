/** @vitest-environment happy-dom */

import { act, createElement, type ReactElement } from 'react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { CHAT_PICTURES_MAX_BASE64, numberedBytes, type UserAttachment } from '../src/chat/attachment-rule.js';
import type { ApiCall } from '../src/runtime.js';
import type { DraftAttachment } from '../ui/src/lib/chat-attachments.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { ChatHostState, ChatMode, ConversationInfo, ConversationRow } from '../ui/src/lib/chat-types.js';
import { formatBytes } from '../ui/src/lib/format.js';
import { PROVIDERS } from '../ui/src/lib/providers-strings.js';
import type {
  KeyCheck,
  LocalCheck,
  ProviderId,
  ProvidersResponse,
  SearchProviderId,
} from '../ui/src/lib/providers-types.js';
import type { CrawlerValue } from '../ui/src/lib/providers-view.js';
import { toast } from '../ui/src/lib/toast.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import Chat from '../ui/src/pages/chat.js';
import { modeBlocker, sendFailure } from '../ui/src/pages/chat/blockers.js';
import {
  type CrawlerChoiceRow,
  crawlerChoices,
  type CrawlerPanelState,
  crawlerRow,
  type WebChoiceRow,
  webChoices,
  type WebPanelState,
  webRow,
} from '../ui/src/pages/chat/composer-tools.js';
import {
  forgetPlace,
  placeEpoch,
  NEW_PLACE,
  readAttachments,
  readDraft,
  readFailed,
  writeAttachments,
  writeDraft,
  writeFailed,
} from '../ui/src/pages/chat/drafts.js';
import { chatHostSnapshot } from '../ui/src/pages/chat/use-chat-host.js';
import {
  answer,
  end,
  FORGED_CITATION,
  RECORDED_DIRECT,
  RECORDED_USE_TOOL,
  user,
} from './helpers/chat-thread-fixtures.js';
import {
  accessibleName,
  allByRole,
  byRole,
  type ChatSync,
  deferred,
  hostModel,
  READY_CHAT_HOST,
  renderPage,
  type RenderedPage,
  sequence,
  STUB_FAILURE,
  stubApi,
  stubError,
  type StubbedApi,
  waitFor,
} from './render.js';

const NOW = Date.now();

let where = { pathname: '', search: '' };
let page: RenderedPage | undefined;
let api: StubbedApi | undefined;
let calls: ApiCall[] = [];
const used: string[] = [];

function LocationSpy(): null {
  const at = useLocation();
  where = { pathname: at.pathname, search: at.search };
  return null;
}

function app(initial: string): ReactElement {
  return createElement(
    MemoryRouter,
    { initialEntries: [initial] },
    createElement(LocationSpy),
    createElement(
      Routes,
      null,
      createElement(Route, { path: '/chat', element: createElement(Chat) }),
      createElement(Route, { path: '*', element: createElement('p', null, 'Elsewhere') }),
    ),
  );
}

function info(id: string, over: Partial<ConversationInfo> = {}): ConversationInfo {
  used.push(id);
  return {
    id,
    mode: 'chat',
    title: `Title ${id}`,
    cwd: null,
    modelId: 'qwen3.5-4b',
    createdAt: NOW - 60_000,
    updatedAt: NOW - 60_000,
    hot: false,
    permissionMode: 'ask',
    ...over,
  };
}

function row(id: string, over: Partial<ConversationInfo> = {}): ConversationRow {
  return { ...info(id, over), turns: 1 };
}

function sync(id: string, over: Partial<ChatSync> = {}, conversation: Partial<ConversationInfo> = {}): ChatSync {
  return {
    type: 'sync',
    seq: 0,
    conversation: info(id, conversation),
    entries: [user('Hello'), answer('Hi there'), end()],
    turns: [],
    turn: null,
    interactions: [],
    notices: [],
    transcriptError: null,
    permissionMarks: [],
    runs: [],
    ...over,
  };
}

interface Open {
  routes?: Record<string, unknown>;
  host?: ChatHostState;
  conversations?: Record<string, ChatSync>;
  rows?: ConversationRow[];
}

async function open(url: string, until: (text: string) => boolean, setup: Open = {}): Promise<RenderedPage> {
  calls = [];
  api = stubApi(
    { '/chat/conversations': { conversations: setup.rows ?? [] }, ...setup.routes },
    {
      onCall: (call) => calls.push(call),
      chat: { host: setup.host ?? READY_CHAT_HOST, conversations: setup.conversations ?? {} },
    },
  );
  page = await renderPage(app(url), until);
  const want = JSON.stringify(setup.host ?? READY_CHAT_HOST);
  await waitFor(() => JSON.stringify(chatHostSnapshot().state) === want, 2_000);
  return page;
}

function pathOf(call: ApiCall): string {
  const bare = call.path.split('?')[0]!;
  return bare.startsWith('/api') ? bare.slice('/api'.length) : bare;
}

function writes(): Array<{ method: string; path: string; body: unknown }> {
  return calls
    .filter((call) => call.method !== 'GET')
    .map((call) => ({ method: call.method, path: pathOf(call), body: call.body }));
}

function composer(): HTMLTextAreaElement {
  const area = byRole(page!.container, 'textbox', CHAT.composerLabel);
  if (area === null) throw new Error('no composer');
  return area as HTMLTextAreaElement;
}

function type(text: string): void {
  const area = composer();
  act(() => {
    Reflect.set(HTMLTextAreaElement.prototype, 'value', text, area);
    area.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function key(target: Element, init: KeyboardEventInit): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }));
  });
}

/** Buttons a user can reach: a swapped-out cell stays laid out, but inert. */
function reachable(scope: ParentNode = document.body): HTMLButtonElement[] {
  return [...scope.querySelectorAll('button')].filter((b) => b.closest('[inert]') === null);
}

function button(label: string, scope: ParentNode = document.body): HTMLButtonElement {
  const found = reachable(scope).find((b) => accessibleName(b) === label);
  if (found === undefined) throw new Error(`no button ${label}`);
  return found;
}

function hasButton(label: string, scope: ParentNode = document.body): boolean {
  return reachable(scope).some((b) => accessibleName(b) === label);
}

function click(element: HTMLElement): void {
  act(() => element.click());
}

/** The text a user sees: what a swap left behind is inert. */
function shown(scope: Element = page!.container): string {
  const copy = scope.cloneNode(true) as Element;
  for (const hidden of copy.querySelectorAll('[inert]')) hidden.remove();
  return copy.textContent ?? '';
}

function heading(): string {
  return byRole(page!.container, 'heading')?.textContent ?? '';
}

function sendFace(): HTMLButtonElement {
  const form = page!.container.querySelector<HTMLFormElement>(`form[aria-label="${CHAT.composerLabel}"]`)!;
  const found = [...form.querySelectorAll('button')].at(-1);
  if (found === undefined) throw new Error('no send button');
  return found;
}

function setVisibility(state: 'visible' | 'hidden'): void {
  act(() => {
    Object.defineProperty(document, 'visibilityState', { value: state, configurable: true });
    document.dispatchEvent(new Event('visibilitychange'));
  });
}

/** The list row of a conversation, as a user clicks it. */
function rowLink(title: string): HTMLAnchorElement {
  const nav = page!.container.querySelector(`nav[aria-label="${CHAT.conversations}"]`);
  const found = [...(nav?.querySelectorAll('a') ?? [])].find((a) => a.textContent?.includes(title) === true);
  if (found === undefined) throw new Error(`no row ${title}`);
  return found;
}

/** The message column of the thread on screen (a swapped-out cell is inert). */
function threadColumn(): Element | null {
  const turn = [...page!.container.querySelectorAll('[data-turn]')].find((node) => node.closest('[inert]') === null);
  return turn?.parentElement ?? null;
}

/** Let the page's zero-delay timers run. */
async function flushTimers(): Promise<void> {
  await act(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));
  await act(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));
}

function host(over: Partial<ChatHostState>): ChatHostState {
  return { ...READY_CHAT_HOST, ...over };
}

/** A `GET /providers` body: `keys` names the search providers with a key and their check. */
function providersBody(
  search: SearchProviderId | null,
  keys: Partial<Record<SearchProviderId, KeyCheck>>,
  over: Partial<ProvidersResponse> = {},
): ProvidersResponse {
  const ids: ProviderId[] = ['exa', 'deepseek', 'brave', 'firecrawl'];
  return {
    search,
    firecrawl: false,
    firecrawlLocal: false,
    fetch: search,
    local: { url: null, check: null, checkedAt: null, version: null, cloud: null },
    providers: ids.map((id) => ({
      id,
      role: id === 'firecrawl' ? 'crawl' : 'search',
      configured: id !== 'firecrawl' && keys[id] !== undefined,
      check: id === 'firecrawl' ? null : (keys[id] ?? null),
      checkedAt: null,
      selected: id === search,
      keyEnv: [],
      skill: { installed: id === search, names: [], upstream: null },
      work: { ready: id === search, missing: null },
      facts: null,
    })),
    chatSkill: { installed: true },
    keychain: 'ok',
    applyPending: false,
    vanished: null,
    ...over,
  };
}

function summary(search: SearchProviderId | null, searchCheck: KeyCheck | null = 'ok'): ChatHostState['providers'] {
  return { ...READY_CHAT_HOST.providers, search, searchCheck: search === null ? null : searchCheck, fetch: search };
}

function reads(): number {
  return calls.filter((call) => call.method === 'GET' && pathOf(call) === '/providers').length;
}

function selections(): unknown[] {
  return writes()
    .filter((call) => call.path === '/providers/selection')
    .map((call) => call.body);
}

/** The "Web search" row of the `+` menu. */
function webMenuRow(): HTMLElement {
  return allByRole(byRole(document.body, 'menu', CHAT.tools)!, 'menuitem')[1]!;
}

/** A row of the web search panel by its label. */
function choice(label: string): HTMLElement {
  const found = allByRole(document.body, 'menuitemradio').find(
    (row) => row.getAttribute('data-label') === label.toLowerCase(),
  );
  if (found === undefined) throw new Error(`no choice ${label}`);
  return found;
}

/** Opens Chat with `search` selected, then the `+` menu and the web search panel. */
async function openWeb(
  search: SearchProviderId | null,
  routes: Record<string, unknown>,
  searchCheck: KeyCheck = 'ok',
  over: Partial<ChatHostState['providers']> = {},
): Promise<HTMLElement> {
  page?.unmount();
  api?.();
  await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
    host: host({ providers: { ...summary(search, searchCheck), ...over } }),
    routes,
  });
  const plus = reachable().find((b) => accessibleName(b).startsWith(CHAT.tools))!;
  click(plus);
  await waitFor(() => byRole(document.body, 'menu', CHAT.tools) !== null);
  click(webMenuRow());
  await waitFor(() => byRole(document.body, 'dialog', CHAT.webSearch) !== null);
  return byRole(document.body, 'dialog', CHAT.webSearch)!;
}

/** Escape until the `+` menu is gone: one level per press. */
function closeMenu(): void {
  for (let level = 0; level < 2; level += 1) {
    const target = document.activeElement ?? document.body;
    act(() => {
      target.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    });
  }
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  page?.unmount();
  api?.();
  page = undefined;
  api = undefined;
  writeDraft(NEW_PLACE, '');
  writeAttachments(NEW_PLACE, []);
  writeFailed(NEW_PLACE, null);
  for (const id of used.splice(0)) forgetPlace(id);
  Object.defineProperty(document, 'visibilityState', { value: 'visible', configurable: true });
  vi.restoreAllMocks();
});

describe('Chat page: places and modes', () => {
  it('opens the New screen of the stored mode; ?mode= wins and is stored', async () => {
    localStorage.setItem('mlx-node:chat-mode:v1', 'work');
    await open('/chat', (text) => text.includes(CHAT.heroWorkTitle));
    expect(heading()).toBe(CHAT.newWork);
    type('Summarize');
    expect(shown()).toContain(CHAT.folderNeeded);
    expect(sendFace().disabled).toBe(true);
    page!.unmount();
    api!();
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle));
    expect(heading()).toBe(CHAT.newChat);
    await waitFor(() => localStorage.getItem('mlx-node:chat-mode:v1') === 'chat');
  });

  it('?c= shows the thread, names it, and puts focus in the composer', async () => {
    await open('/chat?c=p-open', (text) => text.includes('Hi there'), { conversations: { 'p-open': sync('p-open') } });
    expect(heading()).toBe('Title p-open');
    await waitFor(() => document.activeElement === composer());
    expect(document.title).toBe('Title p-open — mlx-node');
    const said = allByRole(page!.container, 'status').map((node) => node.textContent ?? '');
    expect(said).toContain(fill(CHAT.status.opened, { title: 'Title p-open' }));
  });

  it('switching on the New screen keeps the draft and replaces the address', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle));
    type('Half a thought');
    click(button(CHAT.modeWork));
    await waitFor(() => where.search === '?mode=work');
    await waitFor(() => shown().includes(CHAT.heroWorkTitle));
    expect(composer().value).toBe('Half a thought');
    expect(writes()).toEqual([]);
  });

  it('switching inside a conversation goes to the other New screen and cancels nothing', async () => {
    await open('/chat?c=p-switch', (text) => text.includes('Hi there'), {
      conversations: {
        'p-switch': sync('p-switch', { turn: { turnId: 't', phase: 'running', startedAt: NOW, wait: null } }),
      },
    });
    await waitFor(() => hasButton(CHAT.stop));
    key(document.body, { key: 'm', metaKey: true, shiftKey: true });
    await waitFor(() => where.search === '?mode=work');
    expect(writes()).toEqual([]);
  });

  it('a conversation that does not exist says so and offers a New chat', async () => {
    await open('/chat?c=p-missing', (text) => text.includes(CHAT.goneUnknown));
    used.push('p-missing');
    expect(heading()).toBe(CHAT.goneUnknown);
    click(button(CHAT.newChat, page!.container.querySelector('main')!));
    await waitFor(() => where.search === '?mode=chat');
  });
});

describe('Chat page: sending', () => {
  it('New screen: the bubble shows at once, Starting while the create runs, then the conversation opens', async () => {
    const create = deferred({ conversation: info('p-new'), turnId: 't1' });
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': create.body },
      conversations: {
        'p-new': sync('p-new', { entries: [], turn: { turnId: 't1', phase: 'starting', startedAt: NOW, wait: null } }),
      },
    });
    type('Hello there');
    key(composer(), { key: 'Enter' });
    await waitFor(() => shown().includes('Hello there'));
    expect(accessibleName(sendFace())).toBe(CHAT.starting);
    expect(composer().value).toBe('');
    await waitFor(() => writes().length === 1);
    expect(writes()).toEqual([
      {
        method: 'POST',
        path: '/chat/conversations',
        body: { mode: 'chat', text: 'Hello there', modelId: 'qwen3.5-4b' },
      },
    ]);
    await act(async () => create.release());
    await waitFor(() => where.search === '?c=p-new');
    await waitFor(() => api!.chatWatchers('p-new') === 1);
    act(() =>
      api!.emitChat('p-new', {
        type: 'ops',
        seq: 1,
        ops: [{ op: 'push', entry: { kind: 'user', ts: NOW, text: 'Hello there', promptIndex: null } }],
      }),
    );
    await waitFor(() => shown().split('Hello there').length === 2, 2_000);
  });

  it('a first turn that ended before the stream opened shows how it ended', async () => {
    const create = deferred({ conversation: info('p-fast'), turnId: 't1' });
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': create.body },
      conversations: {
        'p-fast': sync('p-fast', {
          entries: [user('Read that page'), end('cancelled')],
          lastEnd: { turnId: 't1', outcome: 'tool-refused' },
        }),
      },
    });
    type('Read that page');
    key(composer(), { key: 'Enter' });
    await act(async () => create.release());
    await waitFor(() => where.search === '?c=p-fast');
    await waitFor(() => shown().includes(CHAT.outcomes['tool-refused']), 2_000);
    expect(shown()).not.toContain(CHAT.outcomes.cancelled);
    expect(accessibleName(sendFace())).toBe(CHAT.send);
  });

  it('a first turn that failed before it ran and before the stream opened gives its text back', async () => {
    const create = deferred({ conversation: info('p-early'), turnId: 't1' });
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': create.body },
      conversations: {
        'p-early': sync('p-early', {
          entries: [],
          lastEnd: { turnId: 't1', failure: { code: 'agent-protocol-error', message: 'garbage' } },
        }),
      },
    });
    type('Early text');
    key(composer(), { key: 'Enter' });
    await act(async () => create.release());
    await waitFor(() => where.search === '?c=p-early');
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]), 2_000);
    expect(accessibleName(sendFace())).toBe(CHAT.send);
    expect(composer().value).toBe('Early text');
  });

  it('a create that fails after the user moved on leaves the other conversation alone', async () => {
    const create = deferred(STUB_FAILURE);
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': create.body },
      rows: [row('p-b')],
      conversations: { 'p-b': sync('p-b') },
    });
    await waitFor(() => shown().includes('Title p-b'));
    type('Text for a new chat');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    click(rowLink('Title p-b'));
    await waitFor(() => where.search === '?c=p-b' && shown().includes('Hi there'));
    await act(async () => create.release());
    await waitFor(() => readDraft(NEW_PLACE) === 'Text for a new chat');
    expect(composer().value).toBe('');
    expect(shown()).not.toContain(CHAT.sendErrors.other[0]);
    expect(writes()).toHaveLength(1);
    click(button(CHAT.newChat));
    await waitFor(() => where.search === '?mode=chat');
    await waitFor(() => composer().value === 'Text for a new chat');
  });

  it('a send that fails late keeps the text typed while it was pending, and Retry sends the failed text', async () => {
    const create = deferred(STUB_FAILURE);
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': create.body },
    });
    type('first prompt');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    type('newer text');
    await act(async () => create.release());
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    expect(composer().value).toBe('newer text');
    expect(readDraft(NEW_PLACE)).toBe('newer text');
    click(button(CHAT.sendErrors.other[1]));
    await waitFor(() => writes().length === 2);
    expect((writes()[1]!.body as { text: string }).text).toBe('first prompt');
    expect(composer().value).toBe('newer text');
  });

  it('a send that fails after the user moved on keeps the newer draft of its place', async () => {
    const error = vi.spyOn(toast, 'error');
    const create = deferred(STUB_FAILURE);
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': create.body },
      rows: [row('p-b3')],
      conversations: { 'p-b3': sync('p-b3') },
    });
    await waitFor(() => shown().includes('Title p-b3'));
    type('first prompt');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    type('newer text');
    click(rowLink('Title p-b3'));
    await waitFor(() => where.search === '?c=p-b3' && shown().includes('Hi there'));
    await act(async () => create.release());
    await waitFor(() => error.mock.calls.length > 0);
    expect(readDraft(NEW_PLACE)).toBe('newer text');
    click(button(CHAT.newChat));
    await waitFor(() => where.search === '?mode=chat');
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    expect(composer().value).toBe('newer text');
    click(button(CHAT.sendErrors.other[1]));
    await waitFor(() => writes().length === 2);
    expect((writes()[1]!.body as { text: string }).text).toBe('first prompt');
  });

  it('a prompt that fails late in an open conversation keeps the text typed while it was pending', async () => {
    const prompt = deferred(STUB_FAILURE);
    await open('/chat?c=p-o', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-o/prompt': prompt.body },
      conversations: { 'p-o': sync('p-o') },
    });
    type('first prompt');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    type('newer text');
    await act(async () => prompt.release());
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    expect(composer().value).toBe('newer text');
    expect(readDraft('p-o')).toBe('newer text');
    click(button(CHAT.sendErrors.other[1]));
    await waitFor(() => writes().length === 2);
    expect((writes()[1]!.body as { text: string }).text).toBe('first prompt');
    expect(composer().value).toBe('newer text');
  });

  it('a create that succeeds after the user moved on does not pull them back', async () => {
    const create = deferred({ conversation: info('p-made'), turnId: 't1' });
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': create.body },
      rows: [row('p-b2')],
      conversations: { 'p-b2': sync('p-b2'), 'p-made': sync('p-made') },
    });
    await waitFor(() => shown().includes('Title p-b2'));
    type('Another new chat');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    click(rowLink('Title p-b2'));
    await waitFor(() => where.search === '?c=p-b2' && shown().includes('Hi there'));
    const lists = (): number =>
      calls.filter((call) => call.method === 'GET' && pathOf(call) === '/chat/conversations').length;
    const before = lists();
    await act(async () => create.release());
    await waitFor(() => lists() > before);
    await flushTimers();
    expect(where.search).toBe('?c=p-b2');
    expect(accessibleName(sendFace())).toBe(CHAT.send);
  });

  it('a refused send strip stays with its conversation', async () => {
    await open('/chat?c=p-strip', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-strip/prompt': stubError('E_UNAVAILABLE', 'slow', 'agent-slow') },
      conversations: { 'p-strip': sync('p-strip') },
    });
    type('Message for A');
    click(sendFace());
    await waitFor(() => shown().includes(CHAT.sendErrors['agent-slow'][0]));
    click(button(CHAT.newChat));
    await waitFor(() => where.search === '?mode=chat');
    await waitFor(() => shown().includes(CHAT.heroChatTitle));
    expect(shown()).not.toContain(CHAT.sendErrors['agent-slow'][0]);
    expect(hasButton(CHAT.sendErrors['agent-slow'][1])).toBe(false);
  });

  it('coming back to a conversation made on the New screen shows it in its own cell', async () => {
    const create = deferred({ conversation: info('p-cell'), turnId: 't1' });
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': create.body },
      rows: [row('p-elsewhere'), row('p-cell')],
      conversations: {
        'p-cell': sync('p-cell', { entries: [user('Cell question'), answer('Cell answer'), end()] }),
        'p-elsewhere': sync('p-elsewhere'),
      },
    });
    await waitFor(() => shown().includes('Title p-elsewhere'));
    type('Cell question');
    key(composer(), { key: 'Enter' });
    await act(async () => create.release());
    await waitFor(() => where.search === '?c=p-cell' && shown().includes('Cell answer'));
    click(rowLink('Title p-elsewhere'));
    await waitFor(() => where.search === '?c=p-elsewhere' && shown().includes('Hi there'));
    const other = threadColumn();
    click(rowLink('Title p-cell'));
    await waitFor(() => where.search === '?c=p-cell' && shown().includes('Cell answer'));
    expect(threadColumn()).not.toBe(other);
  });

  it('a turn that failed before it ran brings its text back, and Try again sends it', async () => {
    await open('/chat?c=p-lost', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-lost/prompt': {} },
      conversations: { 'p-lost': sync('p-lost') },
    });
    type('Lost text');
    click(sendFace());
    await waitFor(() => writes().length === 1);
    act(() =>
      api!.emitChat('p-lost', {
        type: 'turn',
        seq: 1,
        turn: null,
        failure: { code: 'agent-protocol-error', message: 'garbage' },
      }),
    );
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    expect(composer().value).toBe('Lost text');
    click(button(CHAT.sendErrors.other[1]));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-lost/prompt',
      body: { text: 'Lost text' },
    });
  });

  it('⌘⇧C copies the last finished answer, not the one still streaming', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    const original = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    try {
      await open('/chat?c=p-copy', (text) => text.includes('Half an answer'), {
        conversations: {
          'p-copy': sync('p-copy', {
            entries: [user('Hello'), answer('Hi there'), end(), user('More'), answer('Half an answer')],
            turn: { turnId: 't2', phase: 'running', startedAt: NOW, wait: null },
          }),
        },
      });
      key(document.body, { key: 'c', metaKey: true, shiftKey: true });
      await waitFor(() => writeText.mock.calls.length === 1);
      expect(writeText).toHaveBeenCalledWith('Hi there');
    } finally {
      if (original === undefined) Reflect.deleteProperty(navigator, 'clipboard');
      else Object.defineProperty(navigator, 'clipboard', original);
    }
  });

  it('a refused send brings the text back with the reason and its action', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': stubError('E_UNAVAILABLE', 'slow', 'agent-slow') },
    });
    type('Are you there?');
    click(sendFace());
    await waitFor(() => shown().includes(CHAT.sendErrors['agent-slow'][0]));
    expect(composer().value).toBe('Are you there?');
    expect(byRole(page!.container, 'alert')?.textContent).toContain(CHAT.sendErrors['agent-slow'][0]);
    click(button(CHAT.sendErrors['agent-slow'][1]));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]!.body).toMatchObject({ text: 'Are you there?' });
  });

  // Mutation: leave a failed send's strip up after its own action -> the refusal stays on screen.
  it("a refused send's own action takes its strip; a message the composer does not hold keeps Retry", async () => {
    const prompt = deferred(stubError('E_CONFLICT', 'gone', 'model-unavailable'));
    await open('/chat?c=p-act', (text) => text.includes('Hi there'), {
      routes: {
        '/chat/conversations/p-act/prompt': sequence(prompt.body, stubError('E_CONFLICT', 'gone', 'model-unavailable')),
      },
      conversations: { 'p-act': sync('p-act') },
    });
    type('first prompt');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    type('newer text');
    await act(async () => prompt.release());
    await waitFor(() => shown().includes(CHAT.sendErrors['model-unavailable'][0]));
    click(button(CHAT.sendErrors['model-unavailable'][1]));
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    expect(shown()).not.toContain(CHAT.sendErrors['model-unavailable'][0]);
    await flushTimers();
    key(document.activeElement ?? document.body, { key: 'Escape' });
    await waitFor(() => hasButton(CHAT.sendErrors.other[1]));
    click(button(CHAT.sendErrors.other[1]));
    await waitFor(() => writes().length === 2);
    expect((writes()[1]!.body as { text: string }).text).toBe('first prompt');
    await waitFor(() => shown().includes(CHAT.sendErrors['model-unavailable'][0]));
    expect(composer().value).toBe('newer text');
    type('');
    type('second');
    click(sendFace());
    await waitFor(() => writes().length === 3);
    await waitFor(() => composer().value === 'second');
    click(button(CHAT.sendErrors['model-unavailable'][1]));
    await waitFor(() => !shown().includes(CHAT.sendErrors['model-unavailable'][0]));
    expect(shown()).not.toContain(CHAT.sendErrors.other[0]);
    expect(composer().value).toBe('second');
  });

  it('slash text keeps Send off, also after spaces or a new line, and Enter sends nothing', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle));
    for (const text of ['/help', '   /help', '\n/help']) {
      type(text);
      expect(sendFace().disabled).toBe(true);
      expect(shown()).toContain(CHAT.slashText);
      key(composer(), { key: 'Enter' });
    }
    type('Hello');
    key(composer(), { key: 'Enter', isComposing: true });
    expect(writes()).toEqual([]);
    expect(sendFace().disabled).toBe(false);
  });

  it('streams through the store, Stop and Esc in the composer cancel, the stop shows its outcome', async () => {
    const prompt = deferred({});
    await open('/chat?c=p-stream', (text) => text.includes('Hi there'), {
      routes: {
        '/chat/conversations/p-stream/prompt': prompt.body,
        '/chat/conversations/p-stream/cancel': {},
      },
      conversations: { 'p-stream': sync('p-stream') },
    });
    type('Next question');
    click(sendFace());
    await waitFor(() => shown().includes('Next question'));
    expect(accessibleName(sendFace())).toBe(CHAT.starting);
    await act(async () => prompt.release());
    act(() => {
      api!.emitChat('p-stream', {
        type: 'turn',
        seq: 1,
        turn: { turnId: 't2', phase: 'running', startedAt: NOW, wait: null },
      });
      api!.emitChat('p-stream', {
        type: 'ops',
        seq: 2,
        ops: [
          { op: 'push', entry: { kind: 'user', ts: NOW, text: 'Next question', promptIndex: null } },
          { op: 'push', entry: { kind: 'assistant', ts: NOW + 1, text: 'Streaming answer' } },
        ],
      });
      api!.emitChat('p-stream', { type: 'ops', seq: 3, ops: [{ op: 'text', index: 4, append: ' grows' }] });
    });
    await waitFor(() => shown().includes('Streaming answer grows'));
    await waitFor(() => accessibleName(sendFace()) === CHAT.stop);
    expect(shown().split('Next question').length).toBe(2);
    click(sendFace());
    key(composer(), { key: 'Escape' });
    await waitFor(() => writes().filter((w) => w.path.endsWith('/cancel')).length === 2);
    act(() => api!.emitChat('p-stream', { type: 'turn', seq: 4, turn: null, outcome: 'cancelled' }));
    await waitFor(() => shown().includes(CHAT.outcomes.cancelled));
    expect(accessibleName(sendFace())).toBe(CHAT.send);
  });

  it('a turn that hit the length limit offers Continue, which sends "continue"', async () => {
    await open('/chat?c=p-limit', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-limit/prompt': {} },
      conversations: { 'p-limit': sync('p-limit') },
    });
    act(() => api!.emitChat('p-limit', { type: 'turn', seq: 1, turn: null, outcome: 'output-limit' }));
    await waitFor(() => shown().includes(CHAT.outcomes['output-limit']));
    click(button(CHAT.continueAnswer));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-limit/prompt',
      body: { text: CHAT.continuePrompt },
    });
  });

  it.each([
    ['tool-refused', CHAT.outcomes['tool-refused']],
    ['address-refused', CHAT.outcomes['address-refused']],
    ['arguments-unreadable', CHAT.outcomes['arguments-unreadable']],
    ['tool-loop', CHAT.outcomes['tool-loop']],
  ] as const)('a %s end says why and offers Send again', async (outcome, words) => {
    const id = `p-${outcome}`;
    await open(`/chat?c=${id}`, (text) => text.includes('Hi there'), {
      routes: { [`/chat/conversations/${id}/prompt`]: {} },
      conversations: { [id]: sync(id) },
    });
    act(() => api!.emitChat(id, { type: 'turn', seq: 1, turn: null, outcome }));
    await waitFor(() => shown().includes(words));
    click(button(CHAT.sendAgain));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]!.body).toEqual({ text: 'Hello' });
  });

  it('a turn that failed before it ran shows its strip until the user types', async () => {
    await open('/chat?c=p-failed', (text) => text.includes('Hi there'), {
      conversations: { 'p-failed': sync('p-failed') },
    });
    act(() =>
      api!.emitChat('p-failed', {
        type: 'turn',
        seq: 1,
        turn: null,
        failure: { code: 'model-unavailable', message: 'gone' },
      }),
    );
    await waitFor(() => shown().includes(CHAT.sendErrors['model-unavailable'][0]));
    type('x');
    await waitFor(() => !shown().includes(CHAT.sendErrors['model-unavailable'][0]));
  });
});

describe('Chat page: Escape', () => {
  const RUNNING = { turnId: 't', phase: 'running' as const, startedAt: NOW, wait: null };
  const cancels = (): number => writes().filter((w) => w.path.endsWith('/cancel')).length;
  const openRunning = async (id: string): Promise<void> => {
    await open(`/chat?c=${id}`, (text) => text.includes('Hi there'), {
      routes: { [`/chat/conversations/${id}/cancel`]: {} },
      conversations: { [id]: sync(id, { turn: RUNNING }) },
      rows: [row(id), row(`${id}-other`)],
    });
    await waitFor(() => hasButton(CHAT.stop));
  };

  it('Escape outside the composer never stops the answer: thread, page, picker, list row', async () => {
    await openRunning('p-esc-out');
    const thread = page!.container.querySelector('[data-turn]')!;
    key(thread, { key: 'Escape' });
    key(document.body, { key: 'Escape' });
    key(rowLink('Title p-esc-out-other'), { key: 'Escape' });
    const trigger = byRole(
      page!.container,
      'button',
      fill(CHAT.pillName, { model: 'qwen3.5-4b MLX', state: CHAT.pillStates.idle }),
    );
    expect(trigger).not.toBeNull();
    key(trigger!, { key: 'Escape' });
    await flushTimers();
    expect(cancels()).toBe(0);
    expect(hasButton(CHAT.stop)).toBe(true);
  });

  it('Escape closes the web search panel, then the + menu, and a third Escape on the + stops nothing', async () => {
    await openRunning('p-esc-pop');
    const plus = button(CHAT.tools);
    click(plus);
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) !== null);
    click(byRole(document.body, 'menuitem', CHAT.webNameOff)!);
    await waitFor(() => byRole(document.body, 'dialog', CHAT.webSearch) !== null);
    key(byRole(document.body, 'dialog', CHAT.webSearch)!, { key: 'Escape' });
    await waitFor(() => byRole(document.body, 'dialog', CHAT.webSearch) === null);
    expect(byRole(document.body, 'menu', CHAT.tools)).not.toBeNull();
    key(byRole(document.body, 'menu', CHAT.tools)!, { key: 'Escape' });
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) === null);
    key(plus, { key: 'Escape' });
    await flushTimers();
    expect(cancels()).toBe(0);
  });

  it('Escape in the composer stops once: not on a key repeat, not while composing', async () => {
    await openRunning('p-esc-in');
    key(composer(), { key: 'Escape', repeat: true });
    key(composer(), { key: 'Escape', isComposing: true });
    await flushTimers();
    expect(cancels()).toBe(0);
    key(composer(), { key: 'Escape' });
    await waitFor(() => cancels() === 1);
  });

  it('the tools row draws no key legend while an answer runs', async () => {
    await openRunning('p-esc-hint');
    act(() => composer().focus());
    await flushTimers();
    expect(composer().closest('form')!.querySelector('[data-side] kbd')).toBeNull();
    expect(accessibleName(sendFace())).toBe(CHAT.stop);
  });
});

describe('Chat page: the context hint', () => {
  const hint = (): HTMLElement | null =>
    page!.container.querySelector<HTMLElement>('form [role="img"][aria-label^="Context:"]');

  it('the composer shows the conversation’s use of its model’s window, and it moves with the stream', async () => {
    const host: ChatHostState = { ...READY_CHAT_HOST, models: [hostModel('qwen3.5-4b', 262_144)] };
    await open('/chat?c=p-context', (text) => text.includes('Hi there'), {
      host,
      conversations: { 'p-context': sync('p-context', { contextUsed: 11_971 }) },
    });
    await waitFor(() => hint() !== null);
    expect(hint()!.getAttribute('aria-label')).toBe('Context: 11,971 of 262,144 tokens (5%)');
    act(() => api!.emitChat('p-context', { type: 'ops', seq: 1, ops: [{ op: 'context', used: 220_000 }] }));
    await waitFor(() => hint()?.getAttribute('aria-label') === 'Context: 220,000 of 262,144 tokens (84%)');
    expect(hint()!.getAttribute('data-tone')).toBe('warning');
  });

  it('no hint while the used count or the window is unknown', async () => {
    await open('/chat?c=p-no-count', (text) => text.includes('Hi there'), {
      host: { ...READY_CHAT_HOST, models: [hostModel('qwen3.5-4b', 262_144)] },
      conversations: { 'p-no-count': sync('p-no-count') },
    });
    expect(hint()).toBeNull();
    page!.unmount();
    api!();
    await open('/chat?c=p-no-window', (text) => text.includes('Hi there'), {
      host: { ...READY_CHAT_HOST, models: [hostModel('qwen3.5-4b', null)] },
      conversations: { 'p-no-window': sync('p-no-window', { contextUsed: 11_971 }) },
    });
    expect(hint()).toBeNull();
  });
});

describe('Chat page: a message that waits for its model', () => {
  it('Stop during the load keeps the bubble, says it was not sent, and Send again posts the same text', async () => {
    await open('/chat?c=p-unsent', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-unsent/prompt': {}, '/chat/conversations/p-unsent/cancel': {} },
      conversations: { 'p-unsent': sync('p-unsent') },
      host: host({
        inference: 'loading',
        loading: [{ modelId: 'qwen3.5-4b', startedAt: NOW - 3_000, phase: 'weights', stepsDone: 0, steps: 0 }],
      }),
    });
    type('Wait for the model');
    click(sendFace());
    await waitFor(() => writes().length === 1);
    act(() => {
      api!.emitChat('p-unsent', {
        type: 'turn',
        seq: 1,
        turn: { turnId: 't2', phase: 'starting', startedAt: NOW, wait: 'model' },
      });
      api!.emitChat('p-unsent', {
        type: 'ops',
        seq: 2,
        ops: [{ op: 'push', entry: { kind: 'user', ts: NOW, text: 'Wait for the model', promptIndex: null } }],
      });
    });
    await waitFor(() => page!.container.querySelector('[data-load-card]') !== null);
    expect(shown()).toContain(CHAT.load.title);
    expect(shown()).not.toContain(CHAT.pendingStarting);
    click(sendFace());
    await waitFor(() => writes().filter((w) => w.path.endsWith('/cancel')).length === 1);
    act(() =>
      api!.emitChat('p-unsent', {
        type: 'turn',
        seq: 3,
        turn: null,
        turnId: 't2',
        outcome: 'cancelled',
        unsent: true,
      }),
    );
    await waitFor(() => shown().includes(CHAT.outcomes['cancelled-unsent']));
    expect(shown().split('Wait for the model').length).toBe(2);
    expect(shown()).toContain(fill(CHAT.engine.stillLoading, { model: 'qwen3.5-4b' }));
    expect(page!.container.querySelector('[data-load-card]')).toBeNull();
    click(button(CHAT.sendAgain));
    await waitFor(() => writes().filter((w) => w.path.endsWith('/prompt')).length === 2);
    expect(writes().at(-1)).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-unsent/prompt',
      body: { text: 'Wait for the model' },
    });
  });

  it('a turn on a full model says it waits for a running answer until its first output', async () => {
    const full = (inFlight: number, maxConcurrent: number, queued: number, lastStart = NOW - 1): ChatHostState =>
      host({
        resident: [{ modelId: 'qwen3.5-4b', sizeBytes: null }],
        answers: [{ modelId: 'qwen3.5-4b', inFlight, maxConcurrent, queued, lastStart }],
      });
    const queuedLine = fill(CHAT.pendingQueuedOne, { model: 'qwen3.5-4b' });
    await open('/chat?c=p-queued', (text) => text.includes('Second'), {
      conversations: {
        'p-queued': sync('p-queued', {
          entries: [user('Hello'), answer('Hi there'), end(), user('Second')],
          turn: { turnId: 't2', phase: 'running', startedAt: NOW, wait: null },
        }),
      },
      host: full(1, 1, 1),
    });
    await waitFor(() => shown().includes(queuedLine));
    expect(page!.container.querySelector('[data-load-card]')).toBeNull();
    act(() => api!.emitChatHost({ type: 'state', state: full(1, 4, 0) }));
    await waitFor(() => !shown().includes(queuedLine));
    expect(shown()).toContain(CHAT.pendingThinking);
    act(() => api!.emitChatHost({ type: 'state', state: full(1, 1, 1) }));
    await waitFor(() => shown().includes(queuedLine));
    act(() => api!.emitChatHost({ type: 'state', state: full(1, 1, 1, NOW) }));
    await waitFor(() => !shown().includes(queuedLine));
    expect(shown()).toContain(CHAT.pendingThinking);
    act(() => api!.emitChatHost({ type: 'state', state: full(1, 1, 1) }));
    await waitFor(() => shown().includes(queuedLine));
    act(() =>
      api!.emitChat('p-queued', {
        type: 'ops',
        seq: 1,
        ops: [{ op: 'push', entry: { kind: 'assistant', ts: NOW + 1, text: 'First words' } }],
      }),
    );
    await waitFor(() => shown().includes('First words'));
    expect(shown()).not.toContain(queuedLine);
  });

  it("between two steps the thread says what the engine does with this conversation's request, from the host stream", async () => {
    const request = (
      phase: 'queued' | 'prefill',
      prefill: [number, number] | null = null,
    ): ChatHostState['requests'][number] => ({
      session: 'p-steps',
      modelId: 'qwen3.5-4b',
      phase,
      startedAt: NOW,
      prefillDone: prefill?.[0] ?? null,
      prefillTotal: prefill?.[1] ?? null,
      generated: null,
    });
    const engine = (requests: ChatHostState['requests']): ChatHostState =>
      host({
        resident: [{ modelId: 'qwen3.5-4b', sizeBytes: null }],
        answers: [{ modelId: 'qwen3.5-4b', inFlight: 2, maxConcurrent: 4, queued: 0, lastStart: NOW }],
        requests,
      });
    const read: TranscriptEntry = {
      kind: 'tool',
      ts: NOW - 5_000,
      id: 'r1',
      name: 'probe',
      title: 'Probe the file',
      status: 'completed',
      endTs: NOW - 4_000,
    };
    await open('/chat?c=p-steps', (text) => text.includes('Probe the file'), {
      conversations: {
        'p-steps': sync('p-steps', {
          entries: [user('Hello'), read],
          turn: { turnId: 't', phase: 'running', startedAt: NOW - 9_000, wait: null },
        }),
      },
      host: engine([request('queued')]),
    });
    await waitFor(() => shown().includes(CHAT.pendingQueued));
    act(() => api!.emitChatHost({ type: 'state', state: engine([request('prefill', [69_300, 121_300])]) }));
    await waitFor(() => shown().includes(fill(CHAT.pendingReadingOf, { done: '69.3K', total: '121.3K' })));
    expect(byRole(page!.container, 'meter', CHAT.pendingReadMeter)).not.toBeNull();
    act(() =>
      api!.emitChat('p-steps', {
        type: 'ops',
        seq: 1,
        ops: [
          {
            op: 'push',
            entry: { kind: 'tool', ts: NOW + 1, id: 'r2', name: 'probe', title: 'Probe again', status: 'in_progress' },
          },
        ],
      }),
    );
    await waitFor(() => shown().includes('Probe again'));
    expect(page!.container.querySelector('[data-wait-line]')).toBeNull();
  });

  it('a compaction on the conversation stream takes the wait line, Stop stays, and its end leaves the marker', async () => {
    await open('/chat?c=p-compact', (text) => text.includes('Probe the file'), {
      routes: { '/chat/conversations/p-compact/cancel': { cancelled: true } },
      conversations: {
        'p-compact': sync('p-compact', {
          entries: [
            user('Hello'),
            {
              kind: 'tool',
              ts: NOW - 5_000,
              id: 'r1',
              name: 'probe',
              title: 'Probe the file',
              status: 'completed',
              endTs: NOW - 4_000,
            },
          ],
          turn: { turnId: 't', phase: 'running', startedAt: NOW - 9_000, wait: null },
          compaction: { phase: 'running', startedAt: NOW - 1_000, tokensBefore: 222_822 },
        }),
      },
    });
    await waitFor(() => page!.container.querySelector('[data-wait-line="fold"]') !== null);
    expect(page!.container.querySelectorAll('[data-wait-line]')).toHaveLength(1);
    expect(shown()).toContain(CHAT.compacting);
    expect(accessibleName(sendFace())).toBe(CHAT.stop);
    click(sendFace());
    await waitFor(() => writes().filter((w) => w.path.endsWith('/p-compact/cancel')).length === 1);
    act(() =>
      api!.emitChat('p-compact', {
        type: 'ops',
        seq: 1,
        ops: [
          {
            op: 'push',
            entry: { kind: 'compaction', ts: NOW, outcome: 'done', tokensBefore: 222_822, tokensAfter: 31_204 },
          },
        ],
      }),
    );
    act(() => api!.emitChat('p-compact', { type: 'compaction', seq: 2, compaction: null }));
    await waitFor(() => page!.container.querySelector('[data-wait-line="fold"]') === null);
    expect(page!.container.querySelector('[data-compaction-mark="done"]')?.textContent).toContain(
      fill(CHAT.compactedTokens, { before: '222.8K', after: '31.2K' }),
    );
    expect(page!.container.querySelectorAll('[data-wait-line]')).toHaveLength(1);
  });

  it('a select that waits on a model in memory shows the in-memory card, not the load words', async () => {
    await open('/chat?c=p-in-memory', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-in-memory/prompt': {} },
      conversations: { 'p-in-memory': sync('p-in-memory') },
      host: host({ resident: [{ modelId: 'qwen3.5-4b', sizeBytes: null }] }),
    });
    type('Are you there');
    click(sendFace());
    await waitFor(() => writes().length === 1);
    act(() =>
      api!.emitChat('p-in-memory', {
        type: 'turn',
        seq: 1,
        turn: { turnId: 't2', phase: 'starting', startedAt: NOW, wait: 'model' },
      }),
    );
    await waitFor(() => page!.container.querySelector('[data-load-card]') !== null);
    const card = page!.container.querySelector<HTMLElement>('[data-load-card]')!;
    expect(card.getAttribute('data-variant')).toBe('in-memory');
    expect(card.textContent).toContain(CHAT.load.inMemory);
    expect(card.textContent).not.toContain(CHAT.load.busyFirst);
    expect(card.textContent?.toLowerCase()).not.toContain('load');
  });

  it('a load that failed says which model, and Try again sends the text again', async () => {
    await open('/chat?c=p-load-failed', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-load-failed/prompt': {} },
      conversations: { 'p-load-failed': sync('p-load-failed') },
    });
    type('Load it');
    click(sendFace());
    await waitFor(() => writes().length === 1);
    act(() =>
      api!.emitChat('p-load-failed', {
        type: 'turn',
        seq: 1,
        turn: null,
        failure: { code: 'model-load-failed', message: 'The model did not load.' },
        unsent: true,
      }),
    );
    const words = fill(CHAT.sendErrors['model-load-failed'][0], { model: 'qwen3.5-4b' });
    await waitFor(() => shown().includes(words));
    click(button(CHAT.sendErrors['model-load-failed'][1]));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]!.body).toEqual({ text: 'Load it' });
  });
});

describe('Chat page: Unload', () => {
  const RESIDENT = host({ resident: [{ modelId: 'qwen3.5-4b', sizeBytes: 2 * 1024 ** 3 }] });
  const inMemory = fill(CHAT.engine.resident, { model: 'qwen3.5-4b', size: '2.0 GB' });

  /** The text a user sees in a button: a label that animates out is aria-hidden. */
  function face(b: HTMLButtonElement): string {
    const copy = b.cloneNode(true) as HTMLElement;
    for (const hidden of copy.querySelectorAll('[aria-hidden="true"]')) hidden.remove();
    return copy.textContent ?? '';
  }

  function unloadButtons(): HTMLButtonElement[] {
    return reachable().filter((b) => face(b) === CHAT.engine.unload || face(b) === CHAT.engine.unloading);
  }

  async function openPicker(state: keyof typeof CHAT.pillStates): Promise<HTMLElement> {
    key(button(fill(CHAT.pillName, { model: 'qwen3.5-4b MLX', state: CHAT.pillStates[state] })), { key: 'ArrowDown' });
    await waitFor(() => byRole(document.body, 'listbox') !== null);
    return byRole(document.body, 'listbox')!;
  }

  async function stopUnsent(id: string): Promise<void> {
    act(() =>
      api!.emitChat(id, { type: 'turn', seq: 1, turn: null, turnId: 't2', outcome: 'cancelled', unsent: true }),
    );
    await waitFor(() => shown().includes(CHAT.outcomes['cancelled-unsent']));
  }

  it('the picker and the note offer Unload; one press makes one call; after it nothing says the model is in memory', async () => {
    const unload = deferred({ unloaded: ['qwen3.5-4b'] });
    const success = vi.spyOn(toast, 'success');
    await open('/chat?c=p-unload', (text) => text.includes('Hi there'), {
      routes: { 'POST /chat/engine/unload': unload.body },
      conversations: { 'p-unload': sync('p-unload') },
      host: RESIDENT,
    });
    await stopUnsent('p-unload');
    expect(shown()).toContain(inMemory);
    const list = await openPicker('loaded');
    const memoryRow = list.parentElement!.parentElement!.parentElement!.lastElementChild!;
    expect(memoryRow.textContent).toContain(inMemory);
    expect(unloadButtons()).toHaveLength(2);
    expect(list.contains(unloadButtons()[0]!)).toBe(false);
    click(unloadButtons().find((b) => memoryRow.contains(b))!);
    await waitFor(() => unloadButtons().every((b) => face(b) === CHAT.engine.unloading));
    await waitFor(() => writes().some((w) => w.path === '/chat/engine/unload'));
    for (const b of unloadButtons()) click(b);
    unload.release();
    await waitFor(() => !document.body.textContent!.includes(inMemory));
    // A second call sent by those presses went out before the reply to the first one came back.
    expect(writes().filter((w) => w.path === '/chat/engine/unload')).toEqual([
      { method: 'POST', path: '/chat/engine/unload', body: undefined },
    ]);
    expect(success).toHaveBeenCalledWith(fill(CHAT.engine.unloaded, { model: 'qwen3.5-4b' }));
    expect(unloadButtons()).toHaveLength(0);
    expect(shown()).toContain(CHAT.outcomes['cancelled-unsent']);
    expect(hasButton(fill(CHAT.pillName, { model: 'qwen3.5-4b MLX', state: CHAT.pillStates.idle }))).toBe(true);
    act(() => api!.emitChatHost({ type: 'state', state: host({}) }));
    expect(document.body.textContent).not.toContain(inMemory);
    act(() => api!.emitChatHost({ type: 'state', state: RESIDENT }));
    await waitFor(() => shown().includes(inMemory));
    success.mockRestore();
  });

  it('Unload is off while an answer runs; an engine that is busy or fails says so', async () => {
    const error = vi.spyOn(toast, 'error');
    const busy = host({
      ...RESIDENT,
      active: [{ conversationId: 'p-other', mode: 'chat', busy: true, waiting: false, background: false }],
    });
    await open('/chat?c=p-unload-busy', (text) => text.includes('Hi there'), {
      routes: {
        'POST /chat/engine/unload': sequence(
          stubError('E_CONFLICT', 'The engine is busy: an answer or a model load runs.', 'engine-busy'),
          stubError('E_UNAVAILABLE', 'The inference engine did not answer.', 'inference-failed'),
        ),
      },
      conversations: { 'p-unload-busy': sync('p-unload-busy') },
      host: busy,
    });
    await stopUnsent('p-unload-busy');
    expect(unloadButtons().map((b) => b.disabled)).toEqual([true]);
    act(() => api!.emitChatHost({ type: 'state', state: RESIDENT }));
    await waitFor(() => unloadButtons().length === 1 && !unloadButtons()[0]!.disabled);
    click(unloadButtons()[0]!);
    await waitFor(() => error.mock.calls.length === 1);
    expect(error).toHaveBeenLastCalledWith(CHAT.engine.unloadBusy);
    expect(shown()).toContain(inMemory);
    await waitFor(() => face(unloadButtons()[0]!) === CHAT.engine.unload);
    click(unloadButtons()[0]!);
    await waitFor(() => error.mock.calls.length === 2);
    expect(error).toHaveBeenLastCalledWith(CHAT.engine.unloadFailed);
    expect(shown()).toContain(inMemory);
    error.mockRestore();
  });
});

describe('Chat page: titles', () => {
  it('a conversation with no title yet reads as New chat in the heading, the window, the list and the palette', async () => {
    const id = '01a1cdb0-5c3e-4d6f-9a1b-2c3d4e5f6a7b';
    await open(`/chat?c=${id}`, (text) => text.includes('Hi there'), {
      conversations: { [id]: sync(id, {}, { title: '' }) },
      rows: [row(id, { title: '' })],
    });
    expect(heading()).toBe(CHAT.newChat);
    expect(document.title).toBe(`${CHAT.newChat} — mlx-node`);
    expect(rowLink(CHAT.newChat).textContent).toContain(CHAT.newChat);
    expect(shown()).not.toContain(id);
  });
});

describe('Chat page: the open conversation in the list', () => {
  it('lists the open conversation while the stored list does not have it yet', async () => {
    const id = '01a1cdb0-5c3e-4d6f-9a1b-2c3d4e5f6a7c';
    await open(`/chat?c=${id}`, (text) => text.includes('Hi there'), {
      conversations: { [id]: sync(id, {}, { title: 'Tell me about tides.' }) },
      rows: [],
    });
    expect(rowLink('Tell me about tides.').textContent).toContain('Tell me about tides.');
    expect(shown()).not.toContain(CHAT.listEmptyChat[0]);
  });
});

describe('Chat page: states that stop every message', () => {
  const agent = READY_CHAT_HOST.agent;
  const modes = READY_CHAT_HOST.modes;
  const failure = (code: 'inference-failed' | 'agent-incompatible' | 'skill-bundle-corrupt') => ({
    ...modes,
    chat: { ...modes.chat, state: 'blocked' as const, failure: { code, message: code } },
  });

  it.each([
    [
      'agent missing',
      host({ agent: { ...agent, binary: 'missing' } }),
      'Agent not installed',
      'Install',
      { action: 'install' },
    ],
    [
      'install failed',
      host({ agent: { ...agent, binary: 'missing', install: 'failed' } }),
      CHAT.blockers['agent-install-failed'][1],
      'Try again',
      { action: 'install' },
    ],
    [
      'agent too old',
      host({ agent: { ...agent, binary: 'too-old' } }),
      'Agent out of date',
      'Update',
      { action: 'install' },
    ],
    [
      'incompatible',
      host({ modes: failure('agent-incompatible') }),
      'Agent does not match',
      'Update',
      { action: 'install' },
    ],
    [
      'keeps stopping',
      host({ modes: { ...modes, chat: { ...modes.chat, state: 'failed' } } }),
      'Agent keeps stopping',
      'Restart',
      { action: 'restart', mode: 'chat' },
    ],
  ] as const)('%s: the New screen panel and its action', async (_name, state, title, label, body) => {
    await open('/chat?mode=chat', (text) => text.includes(title), { host: state });
    click(button(label));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({ method: 'POST', path: '/chat/agent', body });
    type('Hello');
    expect(sendFace().disabled).toBe(true);
  });

  it.each([
    ['engine stopped', host({ inference: 'stopped' }), 'Engine stopped', 'Open Engine', '/engine'],
    [
      'engine unreachable',
      host({ modes: failure('inference-failed') }),
      'Engine not reachable',
      'Open Engine',
      '/engine',
    ],
    ['no models', host({ models: [], defaultModelId: null }), 'No model yet', 'Open Models', '/models'],
  ] as const)('%s: the panel opens the page that fixes it', async (_name, state, title, label, to) => {
    await open('/chat?mode=chat', (text) => text.includes(title), { host: state });
    click(button(label));
    await waitFor(() => where.pathname === to);
  });

  it('installing and damaged app files have no button; Copy details copies versions only', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.blockers['agent-installing']), {
      host: host({ agent: { ...agent, binary: 'missing', install: 'running' } }),
    });
    expect(page!.container.querySelector('[data-hero-action]')).toBeNull();
    page!.unmount();
    api!();
    await open('/chat?mode=chat', (text) => text.includes('App files damaged'), {
      host: host({ modes: failure('skill-bundle-corrupt') }),
    });
    expect(page!.container.querySelector('[data-hero-action]')).toBeNull();
    page!.unmount();
    api!();
    const writeText = vi.fn(() => Promise.resolve());
    const original = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    try {
      await open('/chat?mode=chat', (text) => text.includes('Agent does not match'), {
        host: host({ modes: failure('agent-incompatible') }),
      });
      click(button(CHAT.blockers.copyDetails));
      await waitFor(() => writeText.mock.calls.length === 1);
      expect(writeText).toHaveBeenCalledWith('agent mlx-agent-v0.3.3 (1.0.45) · needs mlx-agent-v0.3.3');
    } finally {
      if (original === undefined) Reflect.deleteProperty(navigator, 'clipboard');
      else Object.defineProperty(navigator, 'clipboard', original);
    }
  });

  it('inside a conversation the state is a strip over the composer', async () => {
    await open('/chat?c=p-stopped', (text) => text.includes('Hi there'), {
      host: host({ inference: 'stopped' }),
      conversations: { 'p-stopped': sync('p-stopped') },
    });
    await waitFor(() => shown().includes(CHAT.blockers['inference-stopped'][1]));
    type('Hello');
    expect(sendFace().disabled).toBe(true);
    click(button('Open Engine'));
    await waitFor(() => where.pathname === '/engine');
  });

  it('a starting engine is a note that blocks nothing', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.blockers['inference-starting']), {
      host: host({ inference: 'starting' }),
    });
    type('Hello');
    expect(sendFace().disabled).toBe(false);
  });

  it('the order of the rows: transport, agent, mode, engine, models', () => {
    const stopped = host({ inference: 'stopped', models: [] });
    expect(modeBlocker(false, stopped, 'chat')?.reason).toBe(CHAT.blockers.reconnecting);
    expect(modeBlocker(true, null, 'chat')?.reason).toBe(CHAT.startingTip);
    expect(modeBlocker(true, { ...stopped, agent: { ...agent, binary: 'missing' } }, 'chat')?.reason).toBe(
      CHAT.blockers['agent-missing'][1],
    );
    expect(modeBlocker(true, stopped, 'chat')?.reason).toBe(CHAT.blockers['inference-stopped'][1]);
    expect(modeBlocker(true, host({ models: [] }), 'chat')?.reason).toBe(CHAT.blockers['no-models'][1]);
    expect(modeBlocker(true, READY_CHAT_HOST, 'chat')).toBeNull();
  });

  it('every refused send has its words and action', () => {
    const e = CHAT.sendErrors;
    expect(sendFailure('agent-slow', 'E_UNAVAILABLE', 'chat')).toMatchObject({
      strip: { text: e['agent-slow'][0], action: { kind: 'retry' } },
    });
    expect(sendFailure('profile-mismatch', undefined, 'work')).toMatchObject({
      strip: { text: e['profile-mismatch-work'][0], action: { kind: 'install' } },
    });
    expect(sendFailure('model-unavailable', undefined, 'chat')).toMatchObject({
      strip: { action: { kind: 'pick-model' } },
    });
    expect(sendFailure('cwd-invalid', undefined, 'work')).toEqual({
      strip: {
        tone: 'danger',
        text: e['cwd-invalid'][0],
        action: { kind: 'choose-folder', label: e['cwd-invalid'][1] },
      },
    });
    for (const sentence of Object.values(CHAT.cwdProblems)) {
      expect(sendFailure('cwd-invalid', 'E_BAD_REQUEST', 'work', sentence)).toMatchObject({
        strip: { text: sentence },
      });
    }
    expect(sendFailure('cwd-invalid', 'E_BAD_REQUEST', 'work', 'Run rm -rf now.')).toMatchObject({
      strip: { text: e['cwd-invalid'][0] },
    });
    expect(sendFailure('cwd-mismatch', undefined, 'work')).toMatchObject({ strip: { action: { kind: 'retry' } } });
    expect(sendFailure('conversation-busy', 'E_CONFLICT', 'chat')).toEqual({ toast: e['conversation-busy'][0] });
    expect(sendFailure('slash-text', 'E_BAD_REQUEST', 'chat')).toMatchObject({ strip: { text: CHAT.slashText } });
    expect(sendFailure('keychain-unavailable', undefined, 'chat')).toMatchObject({
      strip: { tone: 'warning', action: { kind: 'open-providers' } },
    });
    expect(sendFailure(undefined, 'E_PAYLOAD_TOO_LARGE', 'chat')).toEqual({
      strip: { tone: 'danger', text: CHAT.tooLong },
    });
    expect(sendFailure('inference-stopped', undefined, 'chat')).toEqual({
      strip: { tone: 'danger', text: CHAT.blockers['inference-stopped'][1] },
    });
    expect(sendFailure('something-new', undefined, 'chat')).toMatchObject({
      strip: { text: e.other[0], action: { kind: 'retry' } },
    });
  });
});

describe('Chat page: the live stream', () => {
  it('a seq gap opens the stream again for a fresh sync', async () => {
    const conversations: Record<string, ChatSync> = { 'p-gap': sync('p-gap') };
    await open('/chat?c=p-gap', (text) => text.includes('Hi there'), { conversations });
    await waitFor(() => api!.chatWatchers('p-gap') === 1);
    conversations['p-gap'] = sync('p-gap', { entries: [user('Hello'), answer('After gap'), end()] });
    act(() => api!.emitChat('p-gap', { type: 'turn', seq: 5, turn: null, outcome: 'completed' }));
    await waitFor(() => shown().includes('After gap'));
    expect(api!.chatWatchers('p-gap')).toBe(1);
    expect(calls.filter((call) => call.method !== 'GET')).toEqual([]);
  });

  it('a hidden window lets go of the conversation stream and takes it back when it shows; the host stream stays', async () => {
    await open('/chat?c=p-hide', (text) => text.includes('Hi there'), { conversations: { 'p-hide': sync('p-hide') } });
    await waitFor(() => api!.chatWatchers('p-hide') === 1 && api!.chatWatchers() >= 1);
    setVisibility('hidden');
    await waitFor(() => api!.chatWatchers('p-hide') === 0);
    expect(api!.chatWatchers()).toBeGreaterThanOrEqual(1);
    setVisibility('visible');
    await waitFor(() => api!.chatWatchers('p-hide') === 1 && api!.chatWatchers() >= 1);
  });

  it('after the port drops the composer says Reconnecting, and a new port re-attaches', async () => {
    await open('/chat?c=p-drop', (text) => text.includes('Hi there'), { conversations: { 'p-drop': sync('p-drop') } });
    await waitFor(() => api!.chatWatchers('p-drop') === 1);
    type('Hello');
    api!.dropPort();
    api!();
    await waitFor(() => shown().includes(CHAT.blockers.reconnecting), 3_000);
    expect(sendFace().disabled).toBe(true);
    const second = stubApi(
      { '/chat/conversations': { conversations: [] } },
      {
        chat: {
          conversations: { 'p-drop': sync('p-drop', { entries: [user('Hello'), answer('Back again'), end()] }) },
        },
      },
    );
    api = second;
    await waitFor(() => second.chatWatchers('p-drop') === 1, 3_000);
    await waitFor(() => shown().includes('Back again'), 3_000);
    expect(sendFace().disabled).toBe(false);
  });

  it("polls nothing: its one interval is the wait line's 1 s clock, and a tick sends no request", async () => {
    const spy = vi.spyOn(globalThis, 'setInterval');
    await open('/chat?c=p-timer', (text) => text.includes('Hi there'), {
      conversations: { 'p-timer': sync('p-timer') },
    });
    expect(spy).not.toHaveBeenCalled();
    act(() =>
      api!.emitChat('p-timer', {
        type: 'turn',
        seq: 1,
        turn: { turnId: 't', phase: 'running', startedAt: NOW, wait: null },
      }),
    );
    await waitFor(() => accessibleName(sendFace()) === CHAT.stop);
    expect(page!.container.querySelector('[data-wait-clock]')).not.toBeNull();
    expect(spy.mock.calls.map(([, ms]) => ms)).toEqual([1000]);
    const sent = calls.length;
    await new Promise((resolve) => setTimeout(resolve, 1_100));
    expect(calls.length).toBe(sent);
  });
});

describe('Chat page: web calls', () => {
  it.each([
    ['use_tool', RECORDED_USE_TOOL],
    ['direct', RECORDED_DIRECT],
  ] as const)('the %s shape reads as a search with its sources', async (name, sample) => {
    const id = `p-web-${name}`;
    await open(`/chat?c=${id}`, (text) => text.includes('Rust 1.99.0'), {
      conversations: { [id]: sync(id, { entries: sample.entries, turns: sample.turns }) },
    });
    expect(shown()).toContain(CHAT.searched);
    expect(shown()).not.toContain('mlx_web__web_search');
  });

  it('a link to a host that is not in the conversation is plain text', async () => {
    await open('/chat?c=p-forged', (text) => text.includes('Rust 1.99.0 is out'), {
      conversations: {
        'p-forged': sync('p-forged', { entries: FORGED_CITATION.entries, turns: FORGED_CITATION.turns }),
      },
    });
    const links = [...page!.container.querySelectorAll('a')].map((a) => a.getAttribute('href') ?? '');
    expect(links.some((href) => href.includes('evil.example'))).toBe(false);
    expect(shown()).toContain('here');
  });

  it('the web search row follows the provider selection', () => {
    const base = READY_CHAT_HOST.providers;
    expect(webRow(base)).toEqual({ end: CHAT.webEndOff, name: CHAT.webNameOff, problem: null });
    expect(webRow({ ...base, firecrawl: true, fetch: 'firecrawl' })).toEqual({
      end: CHAT.webEndOff,
      name: CHAT.webNameOff,
      problem: null,
    });
    expect(webRow({ ...base, firecrawl: true, firecrawlLocal: true, fetch: 'firecrawl' }).end).toBe(CHAT.webEndOff);
    expect(webRow({ ...base, search: 'exa', searchCheck: 'ok', fetch: 'exa' })).toEqual({
      end: 'Exa',
      name: 'Web search, Exa',
      problem: null,
    });
    expect(webRow({ ...base, search: 'exa', searchCheck: 'rejected', fetch: 'exa' })).toEqual({
      end: CHAT.webEndRejected,
      name: 'Web search, Exa, key rejected',
      dot: 'danger',
      problem: CHAT.webProblemRejected,
    });
    expect(webRow({ ...base, search: 'exa', searchCheck: 'no-credit', fetch: 'exa' })).toEqual({
      end: CHAT.webEndNoCredit,
      name: 'Web search, Exa, no credit',
      dot: 'warning',
      problem: CHAT.webProblemNoCredit,
    });
  });

  it('webChoices: the rows, the settings note and the caption for each key state', () => {
    const base = READY_CHAT_HOST.providers;
    const exa = { ...base, search: 'exa' as const, searchCheck: 'ok' as const, fetch: 'exa' as const };
    const idle: WebPanelState = { body: null, read: 'done', busy: null, error: null, vanished: null };
    const view = (row: WebChoiceRow) => [row.label, row.checked, row.disabled, row.end ?? '', row.endTone ?? ''];

    const unread = webChoices(exa, { ...idle, read: 'reading' });
    expect(unread.rows.map(view)).toEqual([
      ['Exa', true, false, '', ''],
      ['DeepSeek', false, true, '', ''],
      ['Brave', false, true, '', ''],
      [CHAT.webEndOff, false, false, '', ''],
    ]);
    expect(unread.rows[1]!.name).toBe('DeepSeek, key not read yet');
    expect(unread.note).toBe(CHAT.webSettingsNote);
    expect(unread.caption).toEqual({ text: CHAT.webReading, tone: 'secondary' });
    expect(webChoices(exa, { ...idle, read: 'failed' }).caption).toEqual({
      text: CHAT.webReadFailed,
      tone: 'secondary',
    });
    expect(webChoices({ ...exa, searchCheck: 'rejected' }, idle).rows[0]).toMatchObject({
      end: CHAT.webEndRejected,
      endTone: 'danger',
      name: 'Exa, key rejected',
    });
    expect(webChoices({ ...exa, searchCheck: 'no-credit' }, idle).rows[0]).toMatchObject({
      checked: true,
      end: CHAT.webEndNoCredit,
      endTone: 'warning',
      name: 'Exa, no credit',
    });
    const pagesOnly = webChoices({ ...base, firecrawl: true, fetch: 'firecrawl' }, idle).rows;
    expect(pagesOnly.map((row) => row.checked)).toEqual([false, false, false, true]);

    const none = webChoices(base, { ...idle, body: providersBody(null, {}) });
    expect(none.rows.map(view)).toEqual([
      ['Exa', false, true, CHAT.webNoKey, ''],
      ['DeepSeek', false, true, CHAT.webNoKey, ''],
      ['Brave', false, true, CHAT.webNoKey, ''],
      [CHAT.webEndOff, true, false, '', ''],
    ]);
    expect(none.rows[0]!.name).toBe('Exa, no key. Add it in Web Search settings.');
    expect(none.note).toBe('Add a key for Exa, DeepSeek or Brave.');
    expect(none.caption).toBeNull();
    expect(webChoices(exa, { ...idle, body: providersBody('exa', { exa: 'ok' }) }).note).toBe(
      'Add a key for DeepSeek or Brave.',
    );
    expect(webChoices(exa, { ...idle, body: providersBody('exa', { exa: 'ok', brave: 'ok' }) }).note).toBe(
      'Add a key for DeepSeek.',
    );
    const all = webChoices(exa, { ...idle, body: providersBody('exa', { exa: 'ok', deepseek: 'ok', brave: 'ok' }) });
    expect(all.note).toBe(CHAT.webSettingsNote);
    expect(all.rows.map(view)).toEqual([
      ['Exa', true, false, '', ''],
      ['DeepSeek', false, false, '', ''],
      ['Brave', false, false, '', ''],
      [CHAT.webEndOff, false, false, '', ''],
    ]);

    const checks = webChoices(exa, {
      ...idle,
      body: providersBody('exa', { exa: 'unverified', deepseek: 'no-credit', brave: 'rejected' }),
    });
    expect(checks.rows.map(view)).toEqual([
      ['Exa', true, false, '', ''],
      ['DeepSeek', false, false, CHAT.webEndNoCredit, 'warning'],
      ['Brave', false, false, CHAT.webEndRejected, 'danger'],
      [CHAT.webEndOff, false, false, '', ''],
    ]);
    expect(checks.rows.map((row) => row.name)).toEqual([
      'Exa',
      'DeepSeek, no credit',
      'Brave, key rejected',
      CHAT.webEndOff,
    ]);

    const down = webChoices(exa, {
      ...idle,
      body: providersBody('exa', { exa: 'ok', brave: 'rejected' }, { keychain: 'unavailable' }),
    });
    expect(down.rows.map(view)).toEqual([
      ['Exa', true, false, '', ''],
      ['DeepSeek', false, true, '', ''],
      ['Brave', false, false, CHAT.webEndRejected, 'danger'],
      [CHAT.webEndOff, false, false, '', ''],
    ]);
    expect(down.rows[1]!.name).toBe('DeepSeek, key not read yet');
    expect(down.note).toBe(CHAT.webSettingsNote);
    expect(down.caption).toEqual({ text: CHAT.webKeychain, tone: 'warning' });

    const body = providersBody('exa', { exa: 'ok', brave: 'ok' });
    const busy = webChoices(exa, { ...idle, body, busy: { search: 'brave' } });
    expect(busy.rows.map(view)).toEqual([
      ['Exa', true, true, '', ''],
      ['DeepSeek', false, true, CHAT.webNoKey, ''],
      ['Brave', false, false, CHAT.webSwitching, ''],
      [CHAT.webEndOff, false, true, '', ''],
    ]);
    expect(webChoices(exa, { ...idle, body, busy: { search: null } }).rows[3]).toMatchObject({
      end: CHAT.webSwitching,
      disabled: false,
    });

    const pending = { ...exa, applyPending: true };
    const downBody = providersBody('exa', { exa: 'ok' }, { keychain: 'unavailable' });
    expect(webChoices(pending, { ...idle, body }).caption).toEqual({ text: CHAT.webPending, tone: 'secondary' });
    expect(webChoices(pending, { ...idle, body: downBody }).caption?.text).toBe(CHAT.webKeychain);
    expect(webChoices(pending, { ...idle, body: downBody, vanished: 'brave' }).caption).toEqual({
      text: 'The Brave key is no longer in the Keychain. Web search was turned off.',
      tone: 'warning',
    });
    expect(webChoices(pending, { ...idle, body: downBody, vanished: 'brave', error: 'Nope.' }).caption).toEqual({
      text: 'Nope.',
      tone: 'danger',
    });
  });

  it('no web chip in the tools row: the first button of the row is the +', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      host: host({ providers: { ...READY_CHAT_HOST.providers, search: 'deepseek', searchCheck: 'ok' } }),
    });
    const form = composer().closest('form')!;
    const names = reachable(form).map(accessibleName);
    expect(names.filter((name) => name.startsWith('Web'))).toEqual([]);
    expect(names[0]).toBe(CHAT.tools);
  });

  it('a rejected key or no credit puts a dot on the +, and its name says it; ok and off do not', async () => {
    const plus = async (searchCheck: ChatHostState['providers']['searchCheck'], name: string) => {
      page?.unmount();
      api?.();
      await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
        host: host({ providers: { ...READY_CHAT_HOST.providers, search: 'exa', searchCheck, fetch: 'exa' } }),
      });
      return button(name);
    };
    const rejected = await plus('rejected', 'Tools, web search key rejected');
    expect(rejected.querySelector('[data-tone="danger"]')).not.toBeNull();
    const broke = await plus('no-credit', 'Tools, web search has no credit');
    expect(broke.querySelector('[data-tone="warning"]')).not.toBeNull();
    const ok = await plus('ok', CHAT.tools);
    expect(ok.querySelector('[data-tone]')).toBeNull();
    page!.unmount();
    api!();
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle));
    expect(button(CHAT.tools).querySelector('[data-tone]')).toBeNull();
  });

  it('the panel lists the providers with a check mark on the current one; a row with no key is disabled and says where to add it', async () => {
    const dialog = await openWeb('exa', { '/providers': providersBody('exa', { exa: 'ok' }) });
    await waitFor(() => choice('DeepSeek').textContent === `DeepSeek${CHAT.webNoKey}`);
    expect(webMenuRow().textContent).toBe(`${CHAT.webSearch}Exa`);
    const list = byRole(dialog, 'menu', CHAT.webSearch)!;
    const group = byRole(list, 'group', CHAT.webProvider)!;
    expect(allByRole(group, 'menuitemradio').map((row) => row.getAttribute('data-label'))).toEqual([
      'exa',
      'deepseek',
      'brave',
      'off',
    ]);
    expect(choice('Exa').getAttribute('aria-checked')).toBe('true');
    expect(choice('Exa').getAttribute('aria-disabled')).toBeNull();
    for (const name of ['DeepSeek', 'Brave']) {
      expect(choice(name).getAttribute('aria-checked')).toBe('false');
      expect(choice(name).getAttribute('aria-disabled')).toBe('true');
      expect(choice(name).textContent).toBe(`${name}${CHAT.webNoKey}`);
      expect(accessibleName(choice(name))).toBe(`${name}, no key. Add it in Web Search settings.`);
    }
    expect(choice(CHAT.webEndOff).getAttribute('aria-disabled')).toBeNull();
    const settings = byRole(list, 'menuitem', CHAT.webSettings)!;
    expect(document.getElementById(settings.getAttribute('aria-describedby')!)?.textContent).toBe(
      'Add a key for DeepSeek or Brave.',
    );
    expect(shown(dialog)).not.toContain(CHAT.webPending);
    click(settings);
    await waitFor(() => where.pathname === '/providers');
    await waitFor(
      () =>
        byRole(document.body, 'menu', CHAT.tools) === null && byRole(document.body, 'dialog', CHAT.webSearch) === null,
    );
  });

  it('with no key at all Off is checked and the note names all three; a rejected or no-credit main row keeps its dot', async () => {
    await openWeb(null, { '/providers': providersBody(null, {}) });
    await waitFor(() => choice('Exa').textContent === `Exa${CHAT.webNoKey}`);
    expect(choice(CHAT.webEndOff).getAttribute('aria-checked')).toBe('true');
    const settings = byRole(document.body, 'menuitem', CHAT.webSettings)!;
    expect(document.getElementById(settings.getAttribute('aria-describedby')!)?.textContent).toBe(
      'Add a key for Exa, DeepSeek or Brave.',
    );
    await openWeb('exa', { '/providers': providersBody('exa', { exa: 'rejected' }) }, 'rejected');
    expect(webMenuRow().textContent).toBe(`${CHAT.webSearch}${CHAT.webEndRejected}`);
    expect(webMenuRow().querySelector('[data-tone="danger"]')).not.toBeNull();
    await waitFor(() => choice('Exa').textContent === `Exa${CHAT.webEndRejected}`);
    expect(choice('Exa').querySelector('[data-tone="danger"]')).not.toBeNull();
    await openWeb('exa', { '/providers': providersBody('exa', { exa: 'no-credit' }) }, 'no-credit');
    expect(webMenuRow().textContent).toBe(`${CHAT.webSearch}${CHAT.webEndNoCredit}`);
    expect(webMenuRow().querySelector('[data-tone="warning"]')).not.toBeNull();
  });

  it('a pick sends only the search; the row says Switching… until the host state follows, and a second click sends nothing', async () => {
    const reply = deferred(providersBody('brave', { exa: 'ok', deepseek: 'ok', brave: 'ok' }));
    await openWeb('exa', {
      '/providers': providersBody('exa', { exa: 'ok', brave: 'ok' }),
      'PUT /providers/selection': reply.body,
    });
    await waitFor(() => choice('DeepSeek').textContent === `DeepSeek${CHAT.webNoKey}`);
    click(choice('Brave'));
    await waitFor(() => choice('Brave').textContent === `Brave${CHAT.webSwitching}`);
    expect(choice('Exa').getAttribute('aria-disabled')).toBe('true');
    expect(choice(CHAT.webEndOff).getAttribute('aria-disabled')).toBe('true');
    expect(choice('Brave').getAttribute('aria-disabled')).toBeNull();
    click(choice('Brave'));
    click(choice('Brave'));
    await waitFor(() => selections().length > 0);
    reply.release();
    // The reply's body is in: DeepSeek has a key now, and waits for the pick like every other row.
    await waitFor(() => choice('DeepSeek').textContent === 'DeepSeek');
    expect(choice('DeepSeek').getAttribute('aria-disabled')).toBe('true');
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(choice('Brave').textContent).toBe(`Brave${CHAT.webSwitching}`);
    expect(choice('Exa').getAttribute('aria-checked')).toBe('true');
    act(() => api!.emitChatHost({ type: 'state', state: host({ providers: summary('brave') }) }));
    await waitFor(() => choice('Brave').getAttribute('aria-checked') === 'true');
    expect(choice('Brave').textContent).toBe('Brave');
    expect(choice('Exa').getAttribute('aria-checked')).toBe('false');
    expect(byRole(document.body, 'menuitem', 'Web search, Brave')).not.toBeNull();
    expect(byRole(document.body, 'menu', CHAT.tools)).not.toBeNull();
    expect(byRole(document.body, 'dialog', CHAT.webSearch)).not.toBeNull();
    expect(selections()).toEqual([{ search: 'brave' }]);
  });

  it('Off sends a null search; the checked row and a row with no key send nothing', async () => {
    await openWeb('exa', {
      '/providers': providersBody('exa', { exa: 'ok' }),
      'PUT /providers/selection': providersBody(null, { exa: 'ok' }),
    });
    await waitFor(() => choice('Brave').textContent === `Brave${CHAT.webNoKey}`);
    click(choice('Exa'));
    click(choice('Brave'));
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(selections()).toEqual([]);
    click(choice(CHAT.webEndOff));
    await waitFor(() => selections().length === 1);
    expect(selections()).toEqual([{ search: null }]);
  });

  it('a provider whose key was rejected or has no credit can still be picked, with its dot and words', async () => {
    await openWeb('exa', {
      '/providers': providersBody('exa', { exa: 'ok', deepseek: 'no-credit', brave: 'rejected' }),
      'PUT /providers/selection': providersBody('brave', { exa: 'ok', deepseek: 'no-credit', brave: 'rejected' }),
    });
    await waitFor(() => choice('Brave').textContent === `Brave${CHAT.webEndRejected}`);
    expect(choice('Brave').querySelector('[data-tone="danger"]')).not.toBeNull();
    expect(choice('Brave').getAttribute('aria-disabled')).toBeNull();
    expect(accessibleName(choice('Brave'))).toBe('Brave, key rejected');
    expect(choice('DeepSeek').textContent).toBe(`DeepSeek${CHAT.webEndNoCredit}`);
    expect(choice('DeepSeek').querySelector('[data-tone="warning"]')).not.toBeNull();
    expect(choice('DeepSeek').getAttribute('aria-disabled')).toBeNull();
    click(choice('Brave'));
    await waitFor(() => selections().length === 1);
    expect(selections()).toEqual([{ search: 'brave' }]);
  });

  it('before the keys are read only the current row and Off can be picked, and the caption says so', async () => {
    const read = deferred(providersBody('exa', { exa: 'ok', brave: 'ok' }));
    const dialog = await openWeb('exa', { '/providers': read.body });
    expect(choice('Exa').getAttribute('aria-disabled')).toBeNull();
    expect(choice(CHAT.webEndOff).getAttribute('aria-disabled')).toBeNull();
    expect(choice('DeepSeek').getAttribute('aria-disabled')).toBe('true');
    expect(choice('Brave').getAttribute('aria-disabled')).toBe('true');
    expect(accessibleName(choice('Brave'))).toBe('Brave, key not read yet');
    expect(shown(dialog)).not.toContain(CHAT.webNoKey);
    expect(byRole(dialog, 'status')!.textContent).toBe(CHAT.webReading);
    read.release();
    await waitFor(() => choice('Brave').getAttribute('aria-disabled') === null);
    expect(byRole(dialog, 'status')!.textContent).toBe('');
    expect(choice('DeepSeek').textContent).toBe(`DeepSeek${CHAT.webNoKey}`);
  });

  it('a read that fails says so, keeps the current row and Off, and the next open reads again', async () => {
    const dialog = await openWeb('exa', { '/providers': STUB_FAILURE });
    await waitFor(() => byRole(dialog, 'status')!.textContent === CHAT.webReadFailed);
    expect(choice('Exa').getAttribute('aria-disabled')).toBeNull();
    expect(choice(CHAT.webEndOff).getAttribute('aria-disabled')).toBeNull();
    expect(choice('Brave').getAttribute('aria-disabled')).toBe('true');
    expect(reads()).toBe(1);
    closeMenu();
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) === null);
    click(button(CHAT.tools));
    await waitFor(() => reads() === 2);
  });

  it('a Keychain that does not answer: a known key can still be picked, an unknown one cannot, and the caption says why', async () => {
    const dialog = await openWeb('exa', {
      '/providers': providersBody('exa', { exa: 'ok', brave: 'ok' }, { keychain: 'unavailable' }),
    });
    await waitFor(() => byRole(dialog, 'status')!.textContent === CHAT.webKeychain);
    expect(byRole(dialog, 'status')!.querySelector('.text-warning')).not.toBeNull();
    expect(choice('Brave').getAttribute('aria-disabled')).toBeNull();
    expect(choice('DeepSeek').getAttribute('aria-disabled')).toBe('true');
    expect(choice('DeepSeek').textContent).toBe('DeepSeek');
    expect(shown(dialog)).not.toContain(CHAT.webNoKey);
  });

  it('a failed pick: the caption says why, the check stays, the keys are read again; the next pick clears it', async () => {
    const last = deferred(providersBody('brave', { exa: 'ok', brave: 'ok' }));
    const dialog = await openWeb('exa', {
      '/providers': providersBody('exa', { exa: 'ok', brave: 'ok' }),
      'PUT /providers/selection': sequence(
        stubError('E_CONFLICT', 'Add a key for Brave first.'),
        stubError('E_UNAVAILABLE', "The app's skill files are damaged. Reinstall the app.", 'skill-bundle-corrupt'),
        stubError('E_UNAVAILABLE', 'Skills could not be installed.'),
        last.body,
      ),
    });
    const status = (): string => byRole(dialog, 'status')!.textContent ?? '';
    await waitFor(() => choice('Brave').getAttribute('aria-disabled') === null);
    const before = reads();
    click(choice('Brave'));
    await waitFor(() => status() === 'Add a key for Brave in Web Search settings.');
    expect(byRole(dialog, 'status')!.querySelector('.text-danger')).not.toBeNull();
    expect(choice('Exa').getAttribute('aria-checked')).toBe('true');
    expect(choice('Brave').textContent).toBe('Brave');
    await waitFor(() => reads() === before + 1);
    click(choice('Brave'));
    await waitFor(() => status() === CHAT.webCorrupt);
    click(choice('Brave'));
    await waitFor(() => status() === CHAT.webSwitchFailed);
    click(choice('Brave'));
    await waitFor(() => status() === '');
    last.release();
  });

  it('a pick that fails after the menu closed shows the sentence as a toast', async () => {
    const errors = vi.spyOn(toast, 'error');
    const reply = deferred(stubError('E_UNAVAILABLE', 'Skills could not be installed.'));
    await openWeb('exa', {
      '/providers': providersBody('exa', { exa: 'ok', brave: 'ok' }),
      'PUT /providers/selection': reply.body,
    });
    await waitFor(() => choice('Brave').getAttribute('aria-disabled') === null);
    click(choice('Brave'));
    closeMenu();
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) === null);
    reply.release();
    await waitFor(() => errors.mock.calls.length === 1);
    expect(errors.mock.calls[0]![0]).toBe(CHAT.webSwitchFailed);
  });

  it('a change that waits for running answers puts the pending caption above the rows', async () => {
    const dialog = await openWeb('exa', { '/providers': providersBody('exa', { exa: 'ok' }) }, 'ok', {
      applyPending: true,
    });
    await waitFor(() => byRole(dialog, 'status')!.textContent === CHAT.webPending);
    const text = shown(dialog);
    expect(text.indexOf(CHAT.webPending)).toBeLessThan(text.indexOf('Exa'));
  });

  it('opening the + reads the keys once in Chat; a new host summary reads again; a read from before a write is dropped', async () => {
    const first = deferred(providersBody('exa', { exa: 'ok' }));
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      host: host({ providers: summary('exa') }),
      routes: {
        '/providers': sequence(first.body, providersBody(null, { exa: 'ok', brave: 'ok' })),
        'PUT /providers/selection': providersBody(null, { exa: 'ok', brave: 'ok' }),
      },
    });
    expect(reads()).toBe(0);
    click(button(CHAT.tools));
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) !== null);
    await waitFor(() => reads() === 1);
    click(webMenuRow());
    await waitFor(() => byRole(document.body, 'dialog', CHAT.webSearch) !== null);
    expect(accessibleName(choice('Brave'))).toBe('Brave, key not read yet');
    click(choice(CHAT.webEndOff));
    await waitFor(() => accessibleName(choice('Brave')) === 'Brave');
    first.release();
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(accessibleName(choice('Brave'))).toBe('Brave');
    expect(reads()).toBe(1);
    act(() => api!.emitChatHost({ type: 'state', state: host({ providers: summary(null) }) }));
    await waitFor(() => reads() === 2);
    await waitFor(() => choice(CHAT.webEndOff).getAttribute('aria-checked') === 'true');
  });

  it('a key gone at app start shows once in the panel; with the panel never opened it is a toast', async () => {
    const warnings = vi.spyOn(toast, 'warning');
    const gone = providersBody(null, {}, { vanished: 'exa' });
    const dialog = await openWeb(null, { '/providers': sequence(gone, providersBody(null, {})) });
    const text = 'The Exa key is no longer in the Keychain. Web search was turned off.';
    await waitFor(() => byRole(dialog, 'status')!.textContent === text);
    closeMenu();
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) === null);
    expect(warnings).not.toHaveBeenCalled();
    page!.unmount();
    api!();

    await open('/chat?mode=chat', (shownText) => shownText.includes(CHAT.heroChatTitle), {
      routes: { '/providers': sequence(gone, providersBody(null, {})) },
    });
    click(button(CHAT.tools));
    await waitFor(() => reads() === 1);
    await new Promise((resolve) => setTimeout(resolve, 30));
    closeMenu();
    await waitFor(() => warnings.mock.calls.length === 1);
    expect(warnings.mock.calls[0]![0]).toBe('Exa key not found');
  });

  it('the hero names the search provider, or links to set one up', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle));
    expect(shown()).toContain(CHAT.heroChatWebOff);
    expect(byRole(page!.container, 'link', CHAT.heroChatSetUp)?.getAttribute('href')).toBe('/providers');
    page!.unmount();
    api!();
    await open('/chat?mode=chat', (text) => text.includes(fill(CHAT.heroChatWeb, { provider: 'Brave' })), {
      host: host({ providers: { ...READY_CHAT_HOST.providers, search: 'brave', searchCheck: 'ok', fetch: 'brave' } }),
    });
    click(button(CHAT.tools));
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) !== null);
    expect(byRole(document.body, 'menuitem', 'Web search, Brave')).not.toBeNull();
  });
});

describe('Chat page: the crawler', () => {
  const LOCAL = 'http://127.0.0.1:3002';

  interface CrawlerSetup {
    /** The Firecrawl key's check; absent: no key. */
    key?: KeyCheck;
    /** The saved address's last check; absent: no address; null: not checked yet. */
    local?: LocalCheck | null;
    search?: SearchProviderId | null;
  }

  /** A `GET /providers` body with the crawler on `place`. */
  function crawlerBody(
    place: CrawlerValue,
    setup: CrawlerSetup = {},
    over: Partial<ProvidersResponse> = {},
  ): ProvidersResponse {
    const search = setup.search ?? null;
    const base = providersBody(search, search === null ? {} : { [search]: 'ok' });
    const on = place !== 'off';
    const local = setup.local;
    return {
      ...base,
      firecrawl: on,
      firecrawlLocal: place === 'local',
      fetch: on ? 'firecrawl' : search,
      local:
        local === undefined
          ? base.local
          : {
              url: LOCAL,
              check: local,
              checkedAt: local === null ? null : 1_000,
              version: local === 'ok' ? '0.1.1' : null,
              cloud: null,
            },
      providers: base.providers.map((p) =>
        p.id !== 'firecrawl'
          ? p
          : { ...p, configured: setup.key !== undefined, check: setup.key ?? null, selected: on },
      ),
      ...over,
    };
  }

  function crawlSummary(place: CrawlerValue, search: SearchProviderId | null = null): ChatHostState['providers'] {
    return {
      ...summary(search),
      firecrawl: place !== 'off',
      firecrawlLocal: place === 'local',
      fetch: place !== 'off' ? 'firecrawl' : search,
    };
  }

  function crawlerMenuRow(): HTMLElement {
    return allByRole(byRole(document.body, 'menu', CHAT.tools)!, 'menuitem')[2]!;
  }

  function checks(): number {
    return calls.filter((call) => call.method === 'POST' && pathOf(call) === '/providers/local/check').length;
  }

  function checkedRow(): string | null {
    return allByRole(byRole(document.body, 'menu', CHAT.crawler)!, 'menuitemradio')
      .filter((row) => row.getAttribute('aria-checked') === 'true')
      .map((row) => row.getAttribute('data-label'))[0]!;
  }

  /** Opens Chat with the crawler on `place`, then the `+` menu and the crawler panel. */
  async function openCrawler(
    place: CrawlerValue,
    routes: Record<string, unknown>,
    over: Partial<ChatHostState['providers']> = {},
  ): Promise<HTMLElement> {
    page?.unmount();
    api?.();
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      host: host({ providers: { ...crawlSummary(place), ...over } }),
      routes,
    });
    click(reachable().find((b) => accessibleName(b).startsWith(CHAT.tools))!);
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) !== null);
    click(crawlerMenuRow());
    await waitFor(() => byRole(document.body, 'dialog', CHAT.crawler) !== null);
    return byRole(document.body, 'dialog', CHAT.crawler)!;
  }

  it('crawlerRow: on or off and the place from the host summary, what is wrong from the body', () => {
    const off = crawlSummary('off');
    const cloud = crawlSummary('cloud');
    const local = crawlSummary('local');
    expect(crawlerRow(off, null)).toEqual({ end: CHAT.webEndOff, name: CHAT.crawlerNameOff, problem: null });
    expect(crawlerRow(off, crawlerBody('local', { local: 'not-running' }))).toEqual({
      end: CHAT.webEndOff,
      name: CHAT.crawlerNameOff,
      problem: null,
    });
    expect(crawlerRow(local, crawlerBody('local', { local: 'not-running' }))).toEqual({
      end: CHAT.crawlerNotRunning,
      dot: 'warning',
      name: CHAT.crawlerNameNotRunning,
      problem: null,
    });
    for (const check of ['not-firecrawl', 'open', 'token'] as const) {
      expect(crawlerRow(local, crawlerBody('local', { local: check }))).toEqual({
        end: CHAT.crawlerCheckFailed,
        dot: 'danger',
        name: CHAT.crawlerNameCheckFailed,
        problem: null,
      });
    }
    const atMac = { end: CHAT.crawlerEndLocal, name: CHAT.crawlerNameLocal, problem: null };
    expect(crawlerRow(local, crawlerBody('local', { local: 'ok' }))).toEqual(atMac);
    expect(crawlerRow(local, crawlerBody('local', { local: null }))).toEqual(atMac);
    expect(crawlerRow(local, null)).toEqual(atMac);
    expect(crawlerRow(cloud, crawlerBody('cloud', { key: 'rejected' }))).toEqual({
      end: CHAT.webEndRejected,
      dot: 'danger',
      name: CHAT.crawlerNameRejected,
      problem: null,
    });
    expect(crawlerRow(cloud, crawlerBody('cloud', { key: 'no-credit' }))).toEqual({
      end: CHAT.webEndNoCredit,
      dot: 'warning',
      name: CHAT.crawlerNameNoCredit,
      problem: null,
    });
    const atCloud = { end: CHAT.crawlerEndCloud, name: CHAT.crawlerNameCloud, problem: null };
    expect(crawlerRow(cloud, crawlerBody('cloud', { key: 'ok' }))).toEqual(atCloud);
    expect(crawlerRow(cloud, null)).toEqual(atCloud);
    // The body is one read old: the place still comes from the summary.
    expect(crawlerRow(cloud, crawlerBody('local', { key: 'ok', local: 'not-running' }))).toEqual(atCloud);
  });

  it('crawlerChoices: the rows, the Off note, the settings note and the caption for each state', () => {
    const idle: CrawlerPanelState = { body: null, read: 'done', busy: null, error: null };
    const view = (row: CrawlerChoiceRow) => [row.value, row.checked, row.disabled, row.end ?? '', row.endTone ?? ''];

    const none = crawlerChoices(crawlSummary('off'), { ...idle, body: crawlerBody('off') });
    expect(none.rows.map(view)).toEqual([
      ['cloud', false, true, CHAT.webNoKey, ''],
      ['local', false, true, CHAT.crawlerNotSetUp, ''],
      ['off', true, false, '', ''],
    ]);
    expect(none.rows.map((row) => row.name)).toEqual([CHAT.crawlerRowNoKey, CHAT.crawlerRowNotSetUp, CHAT.webEndOff]);
    expect(none.rows.map((row) => row.label)).toEqual([CHAT.crawlerCloud, CHAT.crawlerLocal, CHAT.webEndOff]);
    expect(none.note).toBe(CHAT.crawlerAddBoth);
    expect(none.caption).toBeNull();
    expect(crawlerChoices(crawlSummary('off'), { ...idle, body: crawlerBody('off', { key: 'ok' }) }).note).toBe(
      CHAT.crawlerAddLocal,
    );
    expect(crawlerChoices(crawlSummary('off'), { ...idle, body: crawlerBody('off', { local: 'ok' }) }).note).toBe(
      CHAT.crawlerAddKey,
    );
    const both = crawlerChoices(crawlSummary('off'), { ...idle, body: crawlerBody('off', { key: 'ok', local: 'ok' }) });
    expect(both.note).toBe(CHAT.webSettingsNote);
    expect(both.rows.map(view)).toEqual([
      ['cloud', false, false, '', ''],
      ['local', false, false, '', ''],
      ['off', true, false, '', ''],
    ]);
    expect(crawlerChoices(crawlSummary('off'), idle).note).toBe(CHAT.webSettingsNote);

    // The Off note follows the search provider of the host summary, with the Web Search page's words.
    const offNote = (search: SearchProviderId | null): string | undefined =>
      crawlerChoices(crawlSummary('off', search), idle).rows[2]!.note;
    expect(offNote('exa')).toBe('Exa reads pages for Chat.');
    expect(offNote('deepseek')).toBe('DeepSeek reads pages for Chat.');
    expect(offNote('brave')).toBe(CHAT.crawlerOffExcerpts);
    expect(offNote(null)).toBe(CHAT.crawlerOffNone);
    expect([CHAT.crawlerOffReads, CHAT.crawlerOffExcerpts, CHAT.crawlerOffNone]).toEqual([
      PROVIDERS.crawlerOffReads,
      PROVIDERS.crawlerOffExcerpts,
      PROVIDERS.crawlerOffNone,
    ]);

    // A rejected key, no credit, a server that does not answer and a failed check can all still be picked.
    const problems = crawlerChoices(crawlSummary('cloud'), {
      ...idle,
      body: crawlerBody('cloud', { key: 'rejected', local: 'not-running' }),
    });
    expect(problems.rows.map(view)).toEqual([
      ['cloud', true, false, CHAT.webEndRejected, 'danger'],
      ['local', false, false, CHAT.crawlerNotRunning, 'warning'],
      ['off', false, false, '', ''],
    ]);
    expect(problems.rows.map((row) => row.name)).toEqual([
      CHAT.crawlerRowRejected,
      CHAT.crawlerRowNotRunning,
      CHAT.webEndOff,
    ]);
    const failed = crawlerChoices(crawlSummary('local'), {
      ...idle,
      body: crawlerBody('local', { key: 'no-credit', local: 'open' }),
    });
    expect(failed.rows.map(view)).toEqual([
      ['cloud', false, false, CHAT.webEndNoCredit, 'warning'],
      ['local', true, false, CHAT.crawlerCheckFailed, 'danger'],
      ['off', false, false, '', ''],
    ]);
    expect(failed.rows.map((row) => row.name)).toEqual([
      CHAT.crawlerRowNoCredit,
      CHAT.crawlerRowCheckFailed,
      CHAT.webEndOff,
    ]);
    const unchecked = crawlerChoices(crawlSummary('local'), { ...idle, body: crawlerBody('local', { local: null }) });
    expect(unchecked.rows[1]).toMatchObject({ checked: true, disabled: false, name: CHAT.crawlerLocal });
    expect(unchecked.rows[1]!.end).toBeUndefined();

    // No body yet: only the current row and Off.
    const reading = crawlerChoices(crawlSummary('cloud'), { ...idle, read: 'reading' });
    expect(reading.rows.map(view)).toEqual([
      ['cloud', true, false, '', ''],
      ['local', false, true, '', ''],
      ['off', false, false, '', ''],
    ]);
    expect(reading.rows[1]!.name).toBe(CHAT.crawlerRowLocalUnread);
    expect(reading.caption).toEqual({ text: CHAT.webReading, tone: 'secondary' });
    const unreadLocal = crawlerChoices(crawlSummary('local'), { ...idle, read: 'failed' });
    expect(unreadLocal.rows.map(view)).toEqual([
      ['cloud', false, true, '', ''],
      ['local', true, false, '', ''],
      ['off', false, false, '', ''],
    ]);
    expect(unreadLocal.rows[0]!.name).toBe(CHAT.crawlerRowUnread);
    expect(unreadLocal.caption).toEqual({ text: CHAT.webReadFailed, tone: 'secondary' });

    // A Keychain that does not answer: a key in the cache can be picked, no key reads as not known; This Mac needs none.
    const down = { keychain: 'unavailable' as const };
    const cached = crawlerChoices(crawlSummary('off'), { ...idle, body: crawlerBody('off', { key: 'ok' }, down) });
    expect(cached.rows[0]).toMatchObject({ disabled: false, name: CHAT.crawlerCloud });
    expect(cached.caption).toEqual({ text: CHAT.webKeychain, tone: 'warning' });
    const unknown = crawlerChoices(crawlSummary('off'), { ...idle, body: crawlerBody('off', { local: 'ok' }, down) });
    expect(unknown.rows.map(view)).toEqual([
      ['cloud', false, true, '', ''],
      ['local', false, false, '', ''],
      ['off', true, false, '', ''],
    ]);
    expect(unknown.rows[0]!.name).toBe(CHAT.crawlerRowUnread);
    expect(unknown.note).toBe(CHAT.webSettingsNote);

    // A write in flight: its row says Switching…, the others wait, the check mark stays.
    const body = crawlerBody('cloud', { key: 'ok', local: 'ok' });
    const busy = crawlerChoices(crawlSummary('cloud'), { ...idle, body, busy: 'local' });
    expect(busy.rows.map(view)).toEqual([
      ['cloud', true, true, '', ''],
      ['local', false, false, CHAT.webSwitching, ''],
      ['off', false, true, '', ''],
    ]);
    expect(crawlerChoices(crawlSummary('cloud'), { ...idle, body, busy: 'off' }).rows[2]).toMatchObject({
      end: CHAT.webSwitching,
      disabled: false,
      checked: false,
    });

    // Captions: the write error, then the Keychain, then the read, then a change that waits.
    const pending = { ...crawlSummary('cloud'), applyPending: true };
    const downBody = crawlerBody('cloud', { key: 'ok' }, down);
    expect(crawlerChoices(pending, { ...idle, body }).caption).toEqual({ text: CHAT.webPending, tone: 'secondary' });
    expect(crawlerChoices(pending, { ...idle, read: 'reading' }).caption?.text).toBe(CHAT.webReading);
    expect(crawlerChoices(pending, { ...idle, body: downBody }).caption?.text).toBe(CHAT.webKeychain);
    expect(crawlerChoices(pending, { ...idle, body: downBody, error: 'Nope.' }).caption).toEqual({
      text: 'Nope.',
      tone: 'danger',
    });
  });

  it('Crawler is the third row of the +; its panel has Cloud, This Mac and Off, the Off note and the settings row', async () => {
    const dialog = await openCrawler('off', { '/providers': crawlerBody('off', { search: 'exa' }) }, { search: 'exa' });
    await waitFor(() => choice(CHAT.crawlerCloud).textContent === `${CHAT.crawlerCloud}${CHAT.webNoKey}`);
    expect(crawlerMenuRow().textContent).toBe(`${CHAT.crawler}${CHAT.webEndOff}`);
    expect(accessibleName(crawlerMenuRow())).toBe(CHAT.crawlerNameOff);
    const list = byRole(dialog, 'menu', CHAT.crawler)!;
    const group = byRole(list, 'group', CHAT.crawlerGroup)!;
    expect(allByRole(group, 'menuitemradio').map((row) => row.getAttribute('data-label'))).toEqual([
      'firecrawl cloud',
      'firecrawl on this mac',
      'off',
    ]);
    expect(checkedRow()).toBe('off');
    expect(choice(CHAT.crawlerLocal).textContent).toBe(`${CHAT.crawlerLocal}${CHAT.crawlerNotSetUp}`);
    expect(choice(CHAT.crawlerLocal).getAttribute('aria-disabled')).toBe('true');
    expect(accessibleName(choice(CHAT.crawlerLocal))).toBe(CHAT.crawlerRowNotSetUp);
    const off = choice(CHAT.webEndOff);
    expect(document.getElementById(off.getAttribute('aria-describedby')!)?.textContent).toBe(
      'Exa reads pages for Chat.',
    );
    const settings = byRole(list, 'menuitem', CHAT.webSettings)!;
    expect(document.getElementById(settings.getAttribute('aria-describedby')!)?.textContent).toBe(CHAT.crawlerAddBoth);
    click(settings);
    await waitFor(() => where.pathname === '/providers');
    await waitFor(
      () =>
        byRole(document.body, 'menu', CHAT.tools) === null && byRole(document.body, 'dialog', CHAT.crawler) === null,
    );
  });

  it('the main row says what is wrong once the body is read; the dot on the + stays search only', async () => {
    await openCrawler('local', { '/providers': crawlerBody('local', { local: 'not-running' }) });
    await waitFor(() => crawlerMenuRow().textContent === `${CHAT.crawler}${CHAT.crawlerNotRunning}`);
    expect(crawlerMenuRow().querySelector('[data-tone="warning"]')).not.toBeNull();
    expect(accessibleName(crawlerMenuRow())).toBe(CHAT.crawlerNameNotRunning);
    expect(choice(CHAT.crawlerLocal).getAttribute('aria-checked')).toBe('true');
    expect(choice(CHAT.crawlerLocal).querySelector('[data-tone="warning"]')).not.toBeNull();
    expect(button(CHAT.tools).querySelector('[data-tone]')).toBeNull();
    await openCrawler('cloud', { '/providers': crawlerBody('cloud', { key: 'rejected' }) });
    await waitFor(() => crawlerMenuRow().textContent === `${CHAT.crawler}${CHAT.webEndRejected}`);
    expect(crawlerMenuRow().querySelector('[data-tone="danger"]')).not.toBeNull();
    expect(accessibleName(crawlerMenuRow())).toBe(CHAT.crawlerNameRejected);
    expect(button(CHAT.tools).querySelector('[data-tone]')).toBeNull();
  });

  it('each pick sends its own body and never `search`; the check mark follows the host summary', async () => {
    const both = { key: 'ok' as const, local: 'ok' as const };
    await openCrawler('cloud', {
      '/providers': crawlerBody('cloud', both),
      'PUT /providers/selection': crawlerBody('cloud', both),
    });
    await waitFor(() => choice(CHAT.crawlerLocal).getAttribute('aria-disabled') === null);
    const to = async (label: string, place: CrawlerValue): Promise<void> => {
      const before = selections().length;
      click(choice(label));
      await waitFor(() => selections().length === before + 1);
      act(() => api!.emitChatHost({ type: 'state', state: host({ providers: crawlSummary(place) }) }));
      await waitFor(() => choice(label).getAttribute('aria-checked') === 'true');
      expect(choice(label).textContent).not.toContain(CHAT.webSwitching);
    };
    await to(CHAT.crawlerLocal, 'local');
    await to(CHAT.webEndOff, 'off');
    await to(CHAT.crawlerCloud, 'cloud');
    expect(selections()).toEqual([
      { firecrawl: true, firecrawlLocal: true },
      { firecrawl: false },
      { firecrawl: true, firecrawlLocal: false },
    ]);
    expect(byRole(document.body, 'dialog', CHAT.crawler)).not.toBeNull();
  });

  it('a pick is busy until the host summary shows it, even while the reply waits; a second click sends nothing', async () => {
    const both = { key: 'ok' as const, local: 'ok' as const };
    const reply = deferred(crawlerBody('local', both));
    await openCrawler('cloud', {
      '/providers': crawlerBody('cloud', both),
      'PUT /providers/selection': reply.body,
    });
    await waitFor(() => choice(CHAT.crawlerLocal).getAttribute('aria-disabled') === null);
    click(choice(CHAT.crawlerLocal));
    await waitFor(() => choice(CHAT.crawlerLocal).textContent === `${CHAT.crawlerLocal}${CHAT.webSwitching}`);
    expect(choice(CHAT.crawlerCloud).getAttribute('aria-disabled')).toBe('true');
    expect(choice(CHAT.webEndOff).getAttribute('aria-disabled')).toBe('true');
    click(choice(CHAT.crawlerLocal));
    click(choice(CHAT.crawlerLocal));
    await waitFor(() => selections().length > 0);
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(checkedRow()).toBe('firecrawl cloud');
    expect(crawlerMenuRow().textContent).toBe(`${CHAT.crawler}${CHAT.crawlerEndCloud}`);
    act(() => api!.emitChatHost({ type: 'state', state: host({ providers: crawlSummary('local') }) }));
    await waitFor(() => checkedRow() === 'firecrawl on this mac');
    expect(choice(CHAT.crawlerLocal).textContent).toBe(CHAT.crawlerLocal);
    expect(choice(CHAT.crawlerCloud).getAttribute('aria-disabled')).toBeNull();
    expect(crawlerMenuRow().textContent).toBe(`${CHAT.crawler}${CHAT.crawlerEndLocal}`);
    expect(selections()).toEqual([{ firecrawl: true, firecrawlLocal: true }]);
    reply.release();
  });

  it('the checked row, a Cloud row with no key and a This Mac row with no address send nothing', async () => {
    await openCrawler('off', {
      '/providers': crawlerBody('off'),
      'PUT /providers/selection': crawlerBody('off'),
    });
    await waitFor(() => choice(CHAT.crawlerCloud).textContent === `${CHAT.crawlerCloud}${CHAT.webNoKey}`);
    expect(choice(CHAT.crawlerCloud).getAttribute('aria-disabled')).toBe('true');
    expect(accessibleName(choice(CHAT.crawlerCloud))).toBe(CHAT.crawlerRowNoKey);
    click(choice(CHAT.crawlerCloud));
    click(choice(CHAT.crawlerLocal));
    click(choice(CHAT.webEndOff));
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(selections()).toEqual([]);
    expect(checkedRow()).toBe('off');
  });

  it('a failed pick: the caption says why by the pick, the check stays, the keys are read again; closed, a toast', async () => {
    const both = { key: 'ok' as const, local: 'ok' as const };
    const last = deferred(stubError('E_UNAVAILABLE', 'Skills could not be installed.'));
    const dialog = await openCrawler('off', {
      '/providers': crawlerBody('off', both),
      'PUT /providers/selection': sequence(
        stubError('E_CONFLICT', 'Add a key for Firecrawl first.'),
        stubError('E_CONFLICT', 'Set up Firecrawl on this Mac first.', 'local-not-set-up'),
        stubError('E_UNAVAILABLE', 'Skills could not be installed.'),
        stubError('E_UNAVAILABLE', "The app's skill files are damaged. Reinstall the app.", 'skill-bundle-corrupt'),
        last.body,
      ),
    });
    const status = (): string => byRole(dialog, 'status')!.textContent ?? '';
    await waitFor(() => choice(CHAT.crawlerLocal).getAttribute('aria-disabled') === null);
    const before = reads();
    click(choice(CHAT.crawlerCloud));
    await waitFor(() => status() === CHAT.crawlerNeedKey);
    expect(byRole(dialog, 'status')!.querySelector('.text-danger')).not.toBeNull();
    expect(checkedRow()).toBe('off');
    expect(choice(CHAT.crawlerCloud).textContent).toBe(CHAT.crawlerCloud);
    await waitFor(() => reads() === before + 1);
    click(choice(CHAT.crawlerLocal));
    await waitFor(() => status() === CHAT.crawlerNeedLocal);
    click(choice(CHAT.crawlerLocal));
    await waitFor(() => status() === CHAT.crawlerSwitchFailed);
    click(choice(CHAT.crawlerCloud));
    await waitFor(() => status() === CHAT.webCorrupt);
    expect(checkedRow()).toBe('off');

    const errors = vi.spyOn(toast, 'error');
    click(choice(CHAT.crawlerLocal));
    closeMenu();
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) === null);
    last.release();
    await waitFor(() => errors.mock.calls.length === 1);
    expect(errors.mock.calls[0]![0]).toBe(CHAT.crawlerSwitchFailed);
  });

  it('one read for both panels; a new firecrawlLocal in the host summary reads again', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      host: host({ providers: crawlSummary('cloud', 'exa') }),
      routes: { '/providers': crawlerBody('cloud', { key: 'ok', local: 'ok', search: 'exa' }) },
    });
    click(button(CHAT.tools));
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) !== null);
    await waitFor(() => reads() === 1);
    click(webMenuRow());
    await waitFor(() => byRole(document.body, 'dialog', CHAT.webSearch) !== null);
    await waitFor(() => choice('Exa').getAttribute('aria-checked') === 'true');
    key(byRole(document.body, 'dialog', CHAT.webSearch)!, { key: 'ArrowLeft' });
    await waitFor(() => byRole(document.body, 'dialog', CHAT.webSearch) === null);
    click(crawlerMenuRow());
    await waitFor(() => byRole(document.body, 'dialog', CHAT.crawler) !== null);
    await waitFor(() => choice(CHAT.crawlerLocal).getAttribute('aria-disabled') === null);
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(reads()).toBe(1);
    act(() => api!.emitChatHost({ type: 'state', state: host({ providers: crawlSummary('local', 'exa') }) }));
    await waitFor(() => reads() === 2);
  });

  it('opening the Crawler panel checks this Mac now and uses the answer for the crawler rows only', async () => {
    const read = crawlerBody('local', { local: 'ok', search: 'exa' });
    // The check's body says Brave has a key: the web search rows must not take it.
    const fresh = providersBody('exa', { exa: 'ok', brave: 'ok' });
    const reply = {
      check: 'not-running',
      providers: {
        ...fresh,
        firecrawl: true,
        firecrawlLocal: true,
        local: { url: LOCAL, check: 'not-running', checkedAt: 2_000, version: null, cloud: null },
      },
    };
    await openCrawler(
      'local',
      { '/providers': read, 'POST /providers/local/check': reply },
      { search: 'exa', searchCheck: 'ok' },
    );
    await waitFor(() => checks() === 1);
    await waitFor(() => choice(CHAT.crawlerLocal).textContent === `${CHAT.crawlerLocal}${CHAT.crawlerNotRunning}`);
    expect(crawlerMenuRow().textContent).toBe(`${CHAT.crawler}${CHAT.crawlerNotRunning}`);
    key(byRole(document.body, 'dialog', CHAT.crawler)!, { key: 'ArrowLeft' });
    await waitFor(() => byRole(document.body, 'dialog', CHAT.crawler) === null);
    click(webMenuRow());
    await waitFor(() => byRole(document.body, 'dialog', CHAT.webSearch) !== null);
    await waitFor(() => choice('Brave').textContent === `Brave${CHAT.webNoKey}`);
    expect(checks()).toBe(1);
  });

  /** Opens Chat with the crawler on `place`, then the `+` menu only. */
  async function openPlus(place: CrawlerValue, routes: Record<string, unknown>): Promise<void> {
    page?.unmount();
    api?.();
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      host: host({ providers: crawlSummary(place) }),
      routes,
    });
    click(reachable().find((b) => accessibleName(b).startsWith(CHAT.tools))!);
    await waitFor(() => byRole(document.body, 'menu', CHAT.tools) !== null);
  }

  const checkReply = (url: string | null, check: LocalCheck, checkedAt: number) => ({
    check,
    providers: crawlerBody('local', {}, { local: { url, check, checkedAt, version: null, cloud: null } }),
  });

  it('opening the + while this Mac is in use checks it now, and the Crawler row shows the answer', async () => {
    await openPlus('local', {
      '/providers': crawlerBody('local', { local: 'ok' }),
      'POST /providers/local/check': checkReply(LOCAL, 'not-running', 2_000),
    });
    await waitFor(() => checks() === 1);
    await waitFor(() => accessibleName(crawlerMenuRow()) === CHAT.crawlerNameNotRunning);
    expect(crawlerMenuRow().textContent).toBe(`${CHAT.crawler}${CHAT.crawlerNotRunning}`);
    click(crawlerMenuRow());
    await waitFor(() => byRole(document.body, 'dialog', CHAT.crawler) !== null);
    await waitFor(() => choice(CHAT.crawlerLocal).textContent === `${CHAT.crawlerLocal}${CHAT.crawlerNotRunning}`);
    expect(checks()).toBe(1);

    await openPlus('local', {
      '/providers': crawlerBody('local', { local: 'not-running' }),
      'POST /providers/local/check': checkReply(LOCAL, 'ok', 2_000),
    });
    await waitFor(() => checks() === 1);
    await waitFor(() => accessibleName(crawlerMenuRow()) === CHAT.crawlerNameLocal);
  });

  it('a check answer older than the read, or of another address, never replaces the read', async () => {
    await openCrawler('local', {
      '/providers': crawlerBody(
        'local',
        {},
        { local: { url: LOCAL, check: 'not-running', checkedAt: 2_000, version: null, cloud: null } },
      ),
      'POST /providers/local/check': checkReply(LOCAL, 'ok', 1_000),
    });
    await waitFor(() => checks() === 1);
    await waitFor(() => choice(CHAT.crawlerLocal).textContent === `${CHAT.crawlerLocal}${CHAT.crawlerNotRunning}`);
    await flushTimers();
    expect(choice(CHAT.crawlerLocal).textContent).toBe(`${CHAT.crawlerLocal}${CHAT.crawlerNotRunning}`);
    expect(accessibleName(crawlerMenuRow())).toBe(CHAT.crawlerNameNotRunning);

    await openCrawler('local', {
      '/providers': crawlerBody('local', { local: 'not-running' }),
      'POST /providers/local/check': checkReply('http://127.0.0.1:4000', 'ok', 2_000),
    });
    await waitFor(() => checks() === 1);
    await waitFor(() => choice(CHAT.crawlerLocal).textContent === `${CHAT.crawlerLocal}${CHAT.crawlerNotRunning}`);
    await flushTimers();
    expect(choice(CHAT.crawlerLocal).textContent).toBe(`${CHAT.crawlerLocal}${CHAT.crawlerNotRunning}`);

    await openCrawler('off', {
      '/providers': crawlerBody('off', { key: 'ok' }),
      'POST /providers/local/check': checkReply(LOCAL, 'ok', 2_000),
    });
    await waitFor(() => checks() === 1);
    await waitFor(() => choice(CHAT.crawlerCloud).getAttribute('aria-disabled') === null);
    await flushTimers();
    expect(choice(CHAT.crawlerLocal).getAttribute('aria-disabled')).toBe('true');
    expect(choice(CHAT.crawlerLocal).textContent).toBe(`${CHAT.crawlerLocal}${CHAT.crawlerNotSetUp}`);
  });

  it('the keyboard: right arrow enters on the checked row, down moves, left and Escape close only the panel', async () => {
    await openPlus('local', {
      '/providers': crawlerBody('local', { local: 'ok', key: 'ok' }),
      'POST /providers/local/check': checkReply(LOCAL, 'ok', 2_000),
    });
    const row = crawlerMenuRow();
    act(() => row.focus());
    key(row, { key: 'ArrowRight' });
    await waitFor(() => byRole(document.body, 'dialog', CHAT.crawler) !== null);
    await waitFor(() => document.activeElement?.getAttribute('data-label') === CHAT.crawlerLocal.toLowerCase());
    expect(document.activeElement?.getAttribute('aria-checked')).toBe('true');
    key(document.activeElement!, { key: 'ArrowDown' });
    await waitFor(() => document.activeElement?.getAttribute('data-label') === CHAT.webEndOff.toLowerCase());
    key(document.activeElement!, { key: 'ArrowLeft' });
    await waitFor(() => byRole(document.body, 'dialog', CHAT.crawler) === null);
    await waitFor(() => document.activeElement === crawlerMenuRow());
    key(crawlerMenuRow(), { key: 'ArrowRight' });
    await waitFor(() => byRole(document.body, 'dialog', CHAT.crawler) !== null);
    key(byRole(document.body, 'dialog', CHAT.crawler)!, { key: 'Escape' });
    await waitFor(() => byRole(document.body, 'dialog', CHAT.crawler) === null);
    expect(byRole(document.body, 'menu', CHAT.tools)).not.toBeNull();
  });

  it('opening the Web search panel never waits for the check of this Mac', async () => {
    const held = deferred({ check: 'ok', providers: crawlerBody('local', { local: 'ok' }) });
    await openWeb(
      'exa',
      {
        '/providers': crawlerBody(
          'local',
          { local: 'ok', search: 'exa' },
          {
            providers: providersBody('exa', { exa: 'ok', brave: 'ok' }).providers,
          },
        ),
        'POST /providers/local/check': held.body,
      },
      'ok',
      { firecrawl: true, firecrawlLocal: true, fetch: 'firecrawl' },
    );
    await waitFor(() => choice('Brave').getAttribute('aria-disabled') === null);
    expect(choice('DeepSeek').textContent).toBe(`DeepSeek${CHAT.webNoKey}`);
    // The `+` opening started the check (this Mac is in use); its answer is still held.
    expect(checks()).toBe(1);
    held.release();
  });
});

describe('Chat page: conversation actions', () => {
  it('rename from the bar PATCHes the title; a busy conversation says to wait', async () => {
    await open('/chat?c=p-rename', (text) => text.includes('Hi there'), {
      routes: { 'PATCH /chat/conversations/p-rename': { ...info('p-rename'), title: 'Better name' } },
      conversations: { 'p-rename': sync('p-rename') },
    });
    click(button(CHAT.rename));
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    const input = byRole(document.body, 'textbox', CHAT.renameLabel) as HTMLInputElement;
    act(() => {
      Reflect.set(HTMLInputElement.prototype, 'value', 'Better name', input);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    click(button(CHAT.renameSave));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'PATCH',
      path: '/chat/conversations/p-rename',
      body: { title: 'Better name' },
    });
    await waitFor(() => byRole(document.body, 'dialog') === null);
  });

  it('rename that meets a running answer shows the wait note and keeps Save off', async () => {
    await open('/chat?c=p-conflict', (text) => text.includes('Hi there'), {
      routes: { 'PATCH /chat/conversations/p-conflict': stubError('E_CONFLICT', 'busy', 'conversation-busy') },
      conversations: { 'p-conflict': sync('p-conflict') },
    });
    key(composer(), { key: 'F2' });
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    const input = byRole(document.body, 'textbox', CHAT.renameLabel) as HTMLInputElement;
    act(() => {
      Reflect.set(HTMLInputElement.prototype, 'value', 'Other', input);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    click(button(CHAT.renameSave));
    await waitFor(() => (document.body.textContent ?? '').includes(CHAT.renameBusy));
    expect(button(CHAT.renameSave).disabled).toBe(true);
    click(button(CHAT.cancel));
    await waitFor(() => byRole(document.body, 'dialog') === null && document.activeElement === composer());
  });

  it('delete from the bar DELETEs and lands on the New screen', async () => {
    await open('/chat?c=p-delete', (text) => text.includes('Hi there'), {
      routes: { 'DELETE /chat/conversations/p-delete': { ok: true } },
      conversations: { 'p-delete': sync('p-delete') },
    });
    click(button(CHAT.remove, page!.container));
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    expect(document.body.textContent).toContain(fill(CHAT.deleteText, { title: 'Title p-delete' }));
    const confirm = reachable(byRole(document.body, 'dialog')!).find((b) => accessibleName(b) === CHAT.deleteConfirm)!;
    click(confirm);
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({ method: 'DELETE', path: '/chat/conversations/p-delete', body: undefined });
    await waitFor(() => where.search === '?mode=chat', 3_000);
  });

  it('a running conversation keeps rename, delete and the model picker off', async () => {
    await open('/chat?c=p-busy', (text) => text.includes('Hi there'), {
      conversations: {
        'p-busy': sync('p-busy', { turn: { turnId: 't', phase: 'running', startedAt: NOW, wait: null } }),
      },
    });
    await waitFor(() => accessibleName(sendFace()) === CHAT.stop);
    expect(button(CHAT.rename).disabled).toBe(true);
    expect(button(CHAT.remove, page!.container).getAttribute('aria-disabled')).toBe('true');
    expect(button(fill(CHAT.pillName, { model: 'qwen3.5-4b MLX', state: CHAT.pillStates.idle })).disabled).toBe(true);
  });

  it('picking a model in a conversation PATCHes it; on the New screen it is stored', async () => {
    const two = host({
      models: [hostModel('qwen3.5-4b', 32768), hostModel('qwen3.5-9b', 32768)],
    });
    await open('/chat?c=p-model', (text) => text.includes('Hi there'), {
      host: two,
      routes: { 'PATCH /chat/conversations/p-model': { ...info('p-model'), modelId: 'qwen3.5-9b' } },
      conversations: { 'p-model': sync('p-model') },
    });
    const trigger = button(fill(CHAT.pillName, { model: 'qwen3.5-4b MLX', state: CHAT.pillStates.idle }));
    key(trigger, { key: 'ArrowDown' });
    await waitFor(() => byRole(document.body, 'listbox') !== null);
    const list = byRole(document.body, 'listbox')!;
    key(list, { key: 'End' });
    key(list, { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'PATCH',
      path: '/chat/conversations/p-model',
      body: { modelId: 'qwen3.5-9b' },
    });
    expect(JSON.parse(localStorage.getItem('mlx-node:chat-model:v1') ?? '{}')).toEqual({ chat: 'qwen3.5-9b' });
    page!.unmount();
    api!();
    await open('/chat?mode=chat', (text) => text.includes(fill(CHAT.heroChatModel, { model: 'qwen3.5-9b' })), {
      host: two,
    });
  });

  it('a quantization is picked in the right list: one PATCH, stored for its base; a new chat starts on the one in memory', async () => {
    const XL = 'Qwen3.8-27B-UD-Q4_K_XL';
    const M = 'Qwen3.8-27B-UD-Q4_K_M';
    const build = (id: string, variant: string, gb: number): ChatHostState['models'][number] => ({
      id,
      base: 'Qwen3.8-27B',
      variant,
      format: 'gguf',
      contextTokens: 262144,
      sizeBytes: gb * 1024 ** 3,
      images: false,
    });
    const models = [hostModel('qwen3.5-4b', 32768), build(XL, 'UD-Q4_K_XL', 17.6), build(M, 'UD-Q4_K_M', 16.5)];
    await open('/chat?c=p-quant', (text) => text.includes('Hi there'), {
      host: host({ models }),
      routes: { 'PATCH /chat/conversations/p-quant': { ...info('p-quant'), modelId: M } },
      conversations: { 'p-quant': sync('p-quant', {}, { modelId: XL }) },
    });
    const trigger = button(fill(CHAT.pillName, { model: 'Qwen3.8-27B UD-Q4_K_XL', state: CHAT.pillStates.idle }));
    key(trigger, { key: 'ArrowDown' });
    await waitFor(() => allByRole(document.body, 'listbox').length === 2);
    const left = allByRole(document.body, 'listbox')[0]!;
    key(left, { key: 'ArrowRight' });
    const right = (): HTMLElement => allByRole(document.body, 'listbox')[1]!;
    await waitFor(() => document.activeElement === right());
    key(right(), { key: 'ArrowUp' });
    key(right(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    await flushTimers();
    expect(writes()).toEqual([{ method: 'PATCH', path: '/chat/conversations/p-quant', body: { modelId: M } }]);
    expect(JSON.parse(localStorage.getItem('mlx-node:chat-variant:v1') ?? '{}')).toEqual({ 'qwen3.8-27b': M });
    page!.unmount();
    api!();
    await open('/chat?mode=chat', (text) => text.includes(fill(CHAT.heroChatModel, { model: XL })), {
      host: host({ models, resident: [{ modelId: XL, sizeBytes: null }] }),
    });
    expect(hasButton(fill(CHAT.pillName, { model: 'Qwen3.8-27B UD-Q4_K_XL', state: CHAT.pillStates.loaded }))).toBe(
      true,
    );
  });

  it('the list filters by mode; search covers both modes with their tags', async () => {
    const rows = [
      row('p-l1', { title: 'Rust question' }),
      row('p-l2', { title: 'Refactor', mode: 'work', cwd: '/tmp/p' }),
    ];
    await open('/chat?c=p-l1', (text) => text.includes('Hi there'), {
      rows,
      conversations: { 'p-l1': sync('p-l1', {}, { title: 'Rust question' }) },
    });
    const nav = (): HTMLElement => page!.container.querySelector(`nav[aria-label="${CHAT.conversations}"]`)!;
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '/', bubbles: true }));
    });
    await waitFor(() => nav() !== null && shown(nav()).includes('Rust question'));
    expect(shown(nav())).not.toContain('Refactor');
    const search = byRole(document.body, 'searchbox', CHAT.listSearch) as HTMLInputElement;
    act(() => {
      Reflect.set(HTMLInputElement.prototype, 'value', 'r', search);
      search.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitFor(() => shown(nav()).includes('Refactor'));
    expect(shown()).toContain(fill(CHAT.listCount, { n: 2 }));
    expect(allByRole(nav(), 'link').length).toBeGreaterThanOrEqual(2);
  });

  it('the mode switch keeps focus when it moves to the other workspace', async () => {
    await open('/chat?c=p-switch-focus', (text) => text.includes('Hi there'), {
      conversations: { 'p-switch-focus': sync('p-switch-focus') },
    });
    await waitFor(() => document.activeElement === composer());
    const work = page!.container.querySelector<HTMLButtonElement>('button[data-value="work"]')!;
    act(() => work.focus());
    click(work);
    await waitFor(() => where.search === '?mode=work');
    await waitFor(() => shown().includes(CHAT.heroWorkTitle));
    await flushTimers();
    expect(document.activeElement?.closest(`[aria-label="${CHAT.mode}"]`)).not.toBeNull();
  });

  it('rename puts a title with line breaks on one line', async () => {
    await open('/chat?c=p-lines', (text) => text.includes('Hi there'), {
      conversations: { 'p-lines': sync('p-lines', {}, { title: 'Tell me\nLine 1' }) },
    });
    click(button(CHAT.rename));
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    const input = byRole(document.body, 'textbox', CHAT.renameLabel) as HTMLInputElement;
    expect(input.value).toBe('Tell me Line 1');
    expect(button(CHAT.renameSave).disabled).toBe(true);
  });

  it('rename refuses a title longer than the agent keeps, before it is sent', async () => {
    await open('/chat?c=p-long-name', (text) => text.includes('Hi there'), {
      conversations: { 'p-long-name': sync('p-long-name') },
    });
    click(button(CHAT.rename));
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    const input = byRole(document.body, 'textbox', CHAT.renameLabel) as HTMLInputElement;
    act(() => {
      Reflect.set(HTMLInputElement.prototype, 'value', 'x'.repeat(101), input);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitFor(() => (document.body.textContent ?? '').includes(CHAT.renameTooLong));
    expect(button(CHAT.renameSave).disabled).toBe(true);
    act(() => {
      Reflect.set(HTMLInputElement.prototype, 'value', 'x'.repeat(100), input);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitFor(() => !button(CHAT.renameSave).disabled);
  });

  it('one search hit reads as one conversation', async () => {
    await open('/chat?c=p-one', (text) => text.includes('Hi there'), {
      rows: [row('p-one', { title: 'Only match' })],
      conversations: { 'p-one': sync('p-one', {}, { title: 'Only match' }) },
    });
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '/', bubbles: true }));
    });
    const search = (): HTMLInputElement | null =>
      byRole(document.body, 'searchbox', CHAT.listSearch) as HTMLInputElement;
    await waitFor(() => search() !== null);
    act(() => {
      Reflect.set(HTMLInputElement.prototype, 'value', 'only', search()!);
      search()!.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitFor(() => shown().includes(CHAT.listCountOne));
    expect(shown()).not.toContain('1 conversations');
  });

  it('New from a conversation lands focus on the panel action when a state blocks the mode', async () => {
    await open('/chat?c=p-focus', (text) => text.includes('Hi there'), {
      host: host({ inference: 'stopped' }),
      conversations: { 'p-focus': sync('p-focus') },
    });
    click(button(CHAT.newChat));
    await waitFor(() => where.search === '?mode=chat');
    await waitFor(() => document.activeElement?.hasAttribute('data-hero-action') === true);
    expect(accessibleName(document.activeElement!)).toBe('Open Engine');
  });
});

describe('Chat page: Work tab', () => {
  it('shows the folder step: Choose folder is on, every other Work action is off', async () => {
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle));
    expect(button(CHAT.folderChoose, page!.container.querySelector('main')!).disabled).toBe(false);
    for (const text of CHAT.startersWork) expect(button(text).disabled).toBe(true);
    expect(composer().placeholder).toBe(CHAT.placeholderWorkNoFolder);
    type('Run the tests');
    expect(sendFace().disabled).toBe(true);
  });

  it('a Work row the page does not know yet still reads as the conversation', async () => {
    const mode: ChatMode = 'work';
    await open('/chat?c=p-work', (text) => text.includes('Hi there'), {
      conversations: { 'p-work': sync('p-work', {}, { mode, cwd: '/tmp/project' }) },
    });
    expect(heading()).toBe('Title p-work');
    expect(page!.container.querySelector('button[data-value="work"]')?.getAttribute('aria-pressed')).toBe('true');
    expect(composer().placeholder).toBe(CHAT.placeholderReply);
  });
});

describe('Chat page: files and pictures', () => {
  const PNG = 'iVBORw0KGgoAAAANSUhEUg==';
  const picture = (id: string): DraftAttachment => ({
    id,
    kind: 'image',
    name: 'cat.png',
    mediaType: 'image/png',
    bytes: 16,
    dataUrl: `data:image/png;base64,${PNG}`,
  });
  /** A text file of exactly `numbered` bytes with its line number. */
  const textFile = (id: string, numbered: number, name = `${id}.txt`): DraftAttachment => {
    const text = 'a'.repeat(numbered - 4);
    return { id, kind: 'file', name, bytes: text.length, numbered: numberedBytes(text), text };
  };
  const seeing = (id: string, images: boolean, contextTokens: number | null = 32768) => ({
    ...hostModel(id, contextTokens),
    images,
  });
  const sentWith = (text: string, attachments: UserAttachment[]): TranscriptEntry => ({
    ...(user(text) as Extract<TranscriptEntry, { kind: 'user' }>),
    attachments,
  });
  const visionHost = (images: boolean): ChatHostState => host({ models: [seeing('qwen3.5-4b', images)] });

  function fileInput(): HTMLInputElement {
    const input = page!.container.querySelector<HTMLInputElement>(`input[type="file"]`);
    if (input === null || accessibleName(input) !== CHAT.attach.row) throw new Error('no file input');
    return input;
  }

  function pick(...files: File[]): void {
    const input = fileInput();
    act(() => {
      Object.defineProperty(input, 'files', { value: files, configurable: true });
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
  }

  /** A text file whose read waits for `release`. */
  function slowFile(name: string, text: string): { file: File; release: () => void } {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const file = new File([text], name, { type: 'text/plain' });
    const slice = file.slice.bind(file);
    file.slice = (...args: Parameters<File['slice']>) => {
      const part = slice(...args);
      const read = part.arrayBuffer.bind(part);
      part.arrayBuffer = () => gate.then(read);
      return part;
    };
    return { file, release };
  }

  const form = (): HTMLFormElement => composer().closest('form')!;
  /** The chips of the composer, as read. */
  const chipText = (): string =>
    allByRole(form(), 'listitem')
      .map((chip) => chip.textContent ?? '')
      .join('|');

  it('a picked text file shows a chip, goes with the message, and leaves the composer with it', async () => {
    const create = deferred({ conversation: info('p-att-send'), turnId: 't1' });
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': create.body },
      conversations: { 'p-att-send': sync('p-att-send', { entries: [] }) },
    });
    pick(new File(['hello\nworld\n'], 'notes.ts', { type: 'video/mp2t' }));
    await waitFor(() => chipText().includes('notes.ts'));
    expect(composer().placeholder).toBe(CHAT.attach.placeholder);
    expect(shown(form())).toContain(fill(CHAT.attach.tokens, { n: 7 }));
    type('Read this');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    expect(writes()[0]!.body).toEqual({
      mode: 'chat',
      text: 'Read this',
      modelId: 'qwen3.5-4b',
      attachments: [{ kind: 'file', name: 'notes.ts', text: 'hello\nworld\n' }],
    });
    expect(composer().value).toBe('');
    expect(allByRole(form(), 'listitem')).toEqual([]);
    expect(readAttachments(NEW_PLACE)).toEqual([]);
    const bubble = [...page!.container.querySelectorAll('[role="list"]')].find(
      (list) => list.closest('form') === null && accessibleName(list) === CHAT.attach.list,
    );
    expect(bubble?.textContent).toContain('notes.ts');
    await act(async () => create.release());
    await waitFor(() => where.search === '?c=p-att-send');
  });

  it('a read that ends after the user moved on lands in the place it started in', async () => {
    await open('/chat?c=p-att-a', (text) => text.includes('Hi there'), {
      rows: [row('p-att-a'), row('p-att-b')],
      conversations: { 'p-att-a': sync('p-att-a'), 'p-att-b': sync('p-att-b') },
    });
    const slow = slowFile('late.txt', 'late text');
    pick(slow.file);
    expect(shown(form())).not.toContain('late.txt');
    click(rowLink('Title p-att-b'));
    await waitFor(() => where.search === '?c=p-att-b' && shown().includes('Hi there'));
    await act(async () => slow.release());
    await waitFor(() => readAttachments('p-att-a').length === 1);
    await flushTimers();
    expect(readAttachments('p-att-b')).toEqual([]);
    expect(allByRole(form(), 'listitem')).toEqual([]);
    click(rowLink('Title p-att-a'));
    await waitFor(() => where.search === '?c=p-att-a');
    await waitFor(() => chipText().includes('late.txt'));
  });

  it('two picks at once add at most 4 files; the rest get one sentence', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle));
    const three = (tag: string): File[] =>
      [1, 2, 3].map((n) => new File([`text ${tag}${n}`], `${tag}${n}.txt`, { type: 'text/plain' }));
    pick(...three('a'));
    pick(...three('b'));
    await waitFor(() => shown(form()).includes(fill(CHAT.attach.errors.tooMany, { n: 2 })));
    expect(readAttachments(NEW_PLACE).map((item) => item.name)).toEqual(['a1.txt', 'a2.txt', 'a3.txt', 'b1.txt']);
    expect(allByRole(form(), 'listitem')).toHaveLength(4);
  });

  it('a refused file says why in the strip while a notice of the mode shows above it', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      host: host({ inference: 'starting' }),
    });
    pick(new File(['%PDF-1.7'], 'report.pdf', { type: 'application/pdf' }));
    const sentence = fill(CHAT.attach.errors.pdf, { name: 'report.pdf' });
    await waitFor(() => allByRole(form(), 'status').some((node) => node.textContent === sentence));
    expect(shown(form())).toContain(CHAT.blockers['inference-starting']);
    type('Now text');
    expect(shown(form())).not.toContain(sentence);
  });

  it('a failed send puts the text and the files back when both are empty', async () => {
    await open('/chat?c=p-att-back', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-att-back/prompt': stubError('E_UNAVAILABLE', 'slow', 'agent-slow') },
      conversations: { 'p-att-back': sync('p-att-back') },
    });
    pick(new File(['some notes'], 'back.txt', { type: 'text/plain' }));
    await waitFor(() => chipText().includes('back.txt'));
    type('With a file');
    click(sendFace());
    await waitFor(() => shown().includes(CHAT.sendErrors['agent-slow'][0]));
    expect(composer().value).toBe('With a file');
    expect(chipText()).toContain('back.txt');
    expect(readAttachments('p-att-back').map((item) => item.name)).toEqual(['back.txt']);
    click(button(CHAT.sendErrors['agent-slow'][1]));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]!.body).toEqual({
      text: 'With a file',
      attachments: [{ kind: 'file', name: 'back.txt', text: 'some notes' }],
    });
  });

  it('text typed while the send failed keeps the composer; Retry sends the files once', async () => {
    const prompt = deferred(STUB_FAILURE);
    await open('/chat?c=p-att-keep', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-att-keep/prompt': sequence(prompt.body, {}) },
      conversations: { 'p-att-keep': sync('p-att-keep') },
    });
    pick(new File(['kept'], 'keep.txt', { type: 'text/plain' }));
    await waitFor(() => chipText().includes('keep.txt'));
    type('first prompt');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    type('newer text');
    await act(async () => prompt.release());
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    expect(composer().value).toBe('newer text');
    expect(allByRole(form(), 'listitem')).toEqual([]);
    click(button(CHAT.sendErrors.other[1]));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]!.body).toEqual({
      text: 'first prompt',
      attachments: [{ kind: 'file', name: 'keep.txt', text: 'kept' }],
    });
    expect(composer().value).toBe('newer text');
    expect(readAttachments('p-att-keep')).toEqual([]);
    expect(allByRole(form(), 'listitem')).toEqual([]);
  });

  const KEEP_BODY = { text: 'first prompt', attachments: [{ kind: 'file', name: 'keep.txt', text: 'kept' }] };

  /** Sends "first prompt" with keep.txt from the composer on screen, then types "newer text" while it is pending. */
  async function sendKeepThenType(): Promise<void> {
    pick(new File(['kept'], 'keep.txt', { type: 'text/plain' }));
    await waitFor(() => chipText().includes('keep.txt'));
    type('first prompt');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    type('newer text');
  }

  /** Back on a place whose send failed: the newer draft stays, and Retry sends the failed text with its files. */
  async function retryOnReturn(): Promise<void> {
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    expect(composer().value).toBe('newer text');
    expect(allByRole(form(), 'listitem')).toEqual([]);
    click(button(CHAT.sendErrors.other[1]));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]!.body).toMatchObject(KEEP_BODY);
    expect(composer().value).toBe('newer text');
  }

  // Mutation: keep one failed record per window, cleared on a place switch -> no Retry on return.
  it('a prompt that fails after the user moved on keeps its files for Retry on return', async () => {
    const error = vi.spyOn(toast, 'error');
    const prompt = deferred(STUB_FAILURE);
    await open('/chat?c=p-fail-a', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-fail-a/prompt': sequence(prompt.body, {}) },
      rows: [row('p-fail-a'), row('p-fail-b')],
      conversations: { 'p-fail-a': sync('p-fail-a'), 'p-fail-b': sync('p-fail-b') },
    });
    await sendKeepThenType();
    click(rowLink('Title p-fail-b'));
    await waitFor(() => where.search === '?c=p-fail-b' && shown().includes('Hi there'));
    await act(async () => prompt.release());
    await waitFor(() => error.mock.calls.length > 0);
    expect(shown()).not.toContain(CHAT.sendErrors.other[0]);
    click(rowLink('Title p-fail-a'));
    await waitFor(() => where.search === '?c=p-fail-a');
    await retryOnReturn();
    await flushTimers();
    expect(shown()).not.toContain(CHAT.sendErrors.other[0]);
  });

  it('a failure seen on screen keeps its Retry after the user leaves and comes back', async () => {
    const prompt = deferred(STUB_FAILURE);
    await open('/chat?c=p-fail-c', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-fail-c/prompt': sequence(prompt.body, {}) },
      rows: [row('p-fail-c'), row('p-fail-d')],
      conversations: { 'p-fail-c': sync('p-fail-c'), 'p-fail-d': sync('p-fail-d') },
    });
    await sendKeepThenType();
    await act(async () => prompt.release());
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    click(rowLink('Title p-fail-d'));
    await waitFor(() => where.search === '?c=p-fail-d' && shown().includes('Hi there'));
    expect(shown()).not.toContain(CHAT.sendErrors.other[0]);
    click(rowLink('Title p-fail-c'));
    await waitFor(() => where.search === '?c=p-fail-c');
    await retryOnReturn();
  });

  it('a later send that goes through takes the Retry strip of a failed message the composer did not hold', async () => {
    const prompt = deferred(STUB_FAILURE);
    await open('/chat?c=p-fail-ok', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-fail-ok/prompt': sequence(prompt.body, {}) },
      conversations: { 'p-fail-ok': sync('p-fail-ok') },
    });
    await sendKeepThenType();
    await act(async () => prompt.release());
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    expect(readFailed('p-fail-ok')?.restored).toBe(false);
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 2);
    expect(writes()[1]!.body).toEqual({ text: 'newer text' });
    await waitFor(() => readFailed('p-fail-ok') === null);
    await flushTimers();
    expect(shown()).not.toContain(CHAT.sendErrors.other[0]);
    expect(writes()).toHaveLength(2);
  });

  it('a create that fails after the user moved on keeps its files for Retry on the New screen', async () => {
    const create = deferred(STUB_FAILURE);
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      routes: { 'POST /chat/conversations': create.body },
      rows: [row('p-fail-e')],
      conversations: { 'p-fail-e': sync('p-fail-e') },
    });
    await waitFor(() => shown().includes('Title p-fail-e'));
    await sendKeepThenType();
    click(rowLink('Title p-fail-e'));
    await waitFor(() => where.search === '?c=p-fail-e' && shown().includes('Hi there'));
    await act(async () => create.release());
    await waitFor(() => readDraft(NEW_PLACE) === 'newer text');
    click(button(CHAT.newChat));
    await waitFor(() => where.search === '?mode=chat');
    await retryOnReturn();
    expect(readAttachments(NEW_PLACE)).toEqual([]);
  });

  it('a send refused as busy beside a newer draft keeps its files for Retry', async () => {
    const prompt = deferred(stubError('E_CONFLICT', 'busy', 'conversation-busy'));
    await open('/chat?c=p-fail-busy', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-fail-busy/prompt': sequence(prompt.body, {}) },
      conversations: { 'p-fail-busy': sync('p-fail-busy') },
    });
    await sendKeepThenType();
    await act(async () => prompt.release());
    await waitFor(() => shown().includes(CHAT.sendErrors['conversation-busy'][0]));
    expect(composer().value).toBe('newer text');
    click(button(CHAT.tryAgain));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]!.body).toMatchObject(KEEP_BODY);
  });

  // Mutation: give each Chat mount its own queue -> the store holds 5 and the screen shows 4 stale chips.
  it('a read started before Chat was left and opened again lands in the chips of the new screen', async () => {
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle));
    const slow = slowFile('late.txt', 'late text');
    pick(slow.file);
    page!.unmount();
    api!();
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle));
    pick(...[0, 1, 2, 3].map((n) => new File([`text ${n}`], `t${n}.txt`, { type: 'text/plain' })));
    await flushTimers();
    await act(async () => slow.release());
    await waitFor(() => allByRole(form(), 'listitem').length === 4 && chipText().includes('late.txt'));
    await flushTimers();
    expect(readAttachments(NEW_PLACE).map((item) => item.name)).toEqual(['late.txt', 't0.txt', 't1.txt', 't2.txt']);
    expect(allByRole(form(), 'listitem')).toHaveLength(4);
    expect(shown(form())).toContain(fill(CHAT.attach.errors.tooMany, { n: 1 }));
  });

  it('a send refused as busy puts the text and the files back on screen, not only in the store', async () => {
    await open('/chat?c=p-att-conflict', (text) => text.includes('Hi there'), {
      routes: {
        '/chat/conversations/p-att-conflict/prompt': stubError('E_CONFLICT', 'busy', 'conversation-busy'),
      },
      conversations: { 'p-att-conflict': sync('p-att-conflict') },
    });
    pick(new File(['busy'], 'busy.txt', { type: 'text/plain' }));
    await waitFor(() => chipText().includes('busy.txt'));
    type('With a file');
    click(sendFace());
    await waitFor(() => writes().length === 1);
    await waitFor(() => composer().value === 'With a file');
    expect(chipText()).toContain('busy.txt');
    expect(readAttachments('p-att-conflict').map((item) => item.name)).toEqual(['busy.txt']);
  });

  it('a failed send does not put its files back beside a file that is still being read', async () => {
    const prompt = deferred(STUB_FAILURE);
    await open('/chat?c=p-att-race', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-att-race/prompt': prompt.body },
      conversations: { 'p-att-race': sync('p-att-race') },
    });
    pick(...[1, 2, 3, 4].map((n) => new File([`text ${n}`], `f${n}.txt`, { type: 'text/plain' })));
    await waitFor(() => allByRole(form(), 'listitem').length === 4);
    type('Four files');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 1);
    const slow = slowFile('late.txt', 'late text');
    pick(slow.file);
    await act(async () => prompt.release());
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    await act(async () => slow.release());
    await waitFor(() => chipText().includes('late.txt'));
    expect(readAttachments('p-att-race').map((item) => item.name)).toEqual(['late.txt']);
    expect(composer().value).toBe('');
  });

  it('a read or a send that ends after its conversation was deleted brings no files and no failed record back', async () => {
    const late = deferred(STUB_FAILURE);
    await open('/chat?c=p-att-del', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-att-del/prompt': sequence(STUB_FAILURE, late.body) },
      conversations: { 'p-att-del': sync('p-att-del') },
    });
    used.push('p-att-del');
    type('first prompt');
    key(composer(), { key: 'Enter' });
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    expect(readFailed('p-att-del')).not.toBeNull();
    const slow = slowFile('late.txt', 'late text');
    pick(slow.file);
    forgetPlace('p-att-del');
    expect(readFailed('p-att-del')).toBeNull();
    await act(async () => slow.release());
    await waitFor(() => !shown(form()).includes(CHAT.attach.reading));
    expect(readAttachments('p-att-del')).toEqual([]);
    type('second prompt');
    key(composer(), { key: 'Enter' });
    await waitFor(() => writes().length === 2);
    forgetPlace('p-att-del');
    await act(async () => late.release());
    await waitFor(() => composer().value === 'second prompt');
    await flushTimers();
    expect(readFailed('p-att-del')).toBeNull();
  });

  it('a file picked on the screen of a gone conversation stays in its composer', async () => {
    await open('/chat?c=p-att-gone', (text) => text.includes(CHAT.goneUnknown));
    used.push('p-att-gone');
    pick(new File(['some notes'], 'kept.txt', { type: 'text/plain' }));
    await waitFor(() => chipText().includes('kept.txt'));
    expect(readAttachments('p-att-gone').map((item) => item.name)).toEqual(['kept.txt']);
  });

  it('Retry with a picture after the model turned out not to see images posts nothing and says why', async () => {
    writeAttachments('p-att-pic', [picture('pic-1')]);
    await open('/chat?c=p-att-pic', (text) => text.includes('Hi there'), {
      host: visionHost(true),
      routes: { '/chat/conversations/p-att-pic/prompt': stubError('E_UNAVAILABLE', 'slow', 'agent-slow') },
      conversations: { 'p-att-pic': sync('p-att-pic') },
    });
    await waitFor(() => chipText().includes('cat.png'));
    type('What is this?');
    click(sendFace());
    await waitFor(() => shown().includes(CHAT.sendErrors['agent-slow'][0]));
    expect(writes()[0]!.body).toEqual({
      text: 'What is this?',
      attachments: [{ kind: 'image', mediaType: 'image/png', data: PNG }],
    });
    act(() => api!.emitChatHost({ type: 'state', state: visionHost(false) }));
    await waitFor(() => chipText().includes(CHAT.attach.chipModel));
    click(button(CHAT.sendErrors['agent-slow'][1]));
    await waitFor(() => shown().includes(CHAT.attach.blockedModel));
    await flushTimers();
    expect(writes()).toHaveLength(1);
  });

  it('Retry is not held back by a file still being read into the draft', async () => {
    await open('/chat?c=p-att-busy', (text) => text.includes('Hi there'), {
      routes: {
        '/chat/conversations/p-att-busy/prompt': sequence(stubError('E_UNAVAILABLE', 'slow', 'agent-slow'), {}),
      },
      conversations: { 'p-att-busy': sync('p-att-busy') },
    });
    type('Plain text');
    click(sendFace());
    await waitFor(() => shown().includes(CHAT.sendErrors['agent-slow'][0]));
    const slow = slowFile('later.txt', 'later');
    pick(slow.file);
    expect(shown(form())).toContain(CHAT.attach.reading);
    click(button(CHAT.sendErrors['agent-slow'][1]));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]!.body).toEqual({ text: 'Plain text' });
    await act(async () => slow.release());
  });

  it('Send again sends the files while this window holds them; otherwise it says it sends none', async () => {
    await open('/chat?c=p-att-again', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-att-again/prompt': {} },
      conversations: { 'p-att-again': sync('p-att-again') },
    });
    pick(new File(['again'], 'again.txt', { type: 'text/plain' }));
    await waitFor(() => chipText().includes('again.txt'));
    type('Look at it');
    click(sendFace());
    await waitFor(() => writes().length === 1);
    act(() => {
      api!.emitChat('p-att-again', {
        type: 'ops',
        seq: 1,
        ops: [
          {
            op: 'push',
            entry: {
              kind: 'user',
              ts: NOW,
              text: 'Look at it',
              promptIndex: null,
              attachments: [{ kind: 'file', name: 'again.txt', bytes: 5 }],
            },
          },
        ],
      });
      api!.emitChat('p-att-again', { type: 'turn', seq: 2, turn: null, turnId: 't2', outcome: 'cancelled' });
    });
    await waitFor(() => hasButton(CHAT.sendAgain));
    expect(hasButton(CHAT.sendAgainNoFiles)).toBe(false);
    click(button(CHAT.sendAgain));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]!.body).toEqual({
      text: 'Look at it',
      attachments: [{ kind: 'file', name: 'again.txt', text: 'again' }],
    });
    page!.unmount();
    api!();
    await open('/chat?c=p-att-again2', (text) => text.includes('Look at it'), {
      routes: { '/chat/conversations/p-att-again2/prompt': {} },
      conversations: {
        'p-att-again2': sync('p-att-again2', {
          entries: [sentWith('Look at it', [{ kind: 'file', name: 'again.txt', bytes: 5 }]), end('cancelled')],
        }),
      },
    });
    await waitFor(() => hasButton(CHAT.sendAgainNoFiles));
    click(button(CHAT.sendAgainNoFiles));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]!.body).toEqual({ text: 'Look at it' });
  });

  it('a picture on a model that does not see images: the chip says so and Send waits until the model can', async () => {
    writeAttachments(NEW_PLACE, [picture('pic-2')]);
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), { host: visionHost(true) });
    type('Describe it');
    expect(sendFace().disabled).toBe(false);
    expect(chipText()).not.toContain(CHAT.attach.chipModel);
    act(() => api!.emitChatHost({ type: 'state', state: visionHost(false) }));
    await waitFor(() => chipText().includes(CHAT.attach.chipModel));
    expect(sendFace().disabled).toBe(true);
    expect(shown(form())).toContain(CHAT.attach.blockedModel);
    key(composer(), { key: 'Enter' });
    act(() => api!.emitChatHost({ type: 'state', state: visionHost(true) }));
    await waitFor(() => !chipText().includes(CHAT.attach.chipModel));
    expect(sendFace().disabled).toBe(false);
    expect(writes()).toEqual([]);
  });

  it('a conversation with pictures on a model that cannot see them says the pictures are left out', async () => {
    const entries = [
      sentWith('Look', [{ kind: 'image', mediaType: 'image/png', bytes: 2048 }]),
      answer('A cat'),
      end(),
    ];
    await open('/chat?c=p-att-unseen', (text) => text.includes('A cat'), {
      host: visionHost(false),
      conversations: { 'p-att-unseen': sync('p-att-unseen', { entries }) },
    });
    expect(shown(form())).toContain(CHAT.attach.imagesUnseen);
    act(() => api!.emitChatHost({ type: 'state', state: visionHost(true) }));
    await waitFor(() => !shown(form()).includes(CHAT.attach.imagesUnseen));
  });

  it('the thread shows the chips of a stored message: file name and size, picture type and size', async () => {
    const entries = [
      sentWith('Both', [
        { kind: 'file', name: 'stored.md', bytes: 2048 },
        { kind: 'image', mediaType: 'image/jpeg', bytes: 4096 },
      ]),
      answer('Seen'),
      end(),
    ];
    await open('/chat?c=p-att-stored', (text) => text.includes('Seen'), {
      host: visionHost(true),
      conversations: { 'p-att-stored': sync('p-att-stored', { entries }) },
    });
    const list = byRole(threadColumn()!, 'list', CHAT.attach.list)!;
    const items = allByRole(list, 'listitem').map((item) => item.textContent);
    expect(items).toEqual([`stored.md${formatBytes(2048)}`, `${CHAT.attach.image}JPEG · ${formatBytes(4096)}`]);
    expect(list.querySelector('button, img')).toBeNull();
    expect(shown(form())).not.toContain(CHAT.attach.imagesUnseen);
  });

  it('files and text over 90,000 bytes together are too long; a window decides only when it is known', async () => {
    const four = (tag: string): DraftAttachment[] => [1, 2, 3, 4].map((n) => textFile(`${tag}${n}`, 20_000));
    writeAttachments(NEW_PLACE, four('long'));
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle), {
      host: host({ models: [hostModel('qwen3.5-4b', null)] }),
    });
    type('b'.repeat(10_001));
    expect(shown(form())).toContain(CHAT.attach.blockedLong);
    expect(sendFace().disabled).toBe(true);
    type('hi');
    expect(sendFace().disabled).toBe(false);
    expect(shown(form())).toContain(fill(CHAT.attach.tokens, { n: '26,668' }));
    page!.unmount();
    api!();
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle));
    type('hi');
    expect(shown(form())).toContain(CHAT.attach.blockedWindow);
    expect(shown(form())).toContain(fill(CHAT.attach.tokensNearFull, { n: '26,668' }));
    expect(sendFace().disabled).toBe(true);
  });

  it('a removed chip leaves the store of its place; a forget drops the writes that started before it', async () => {
    used.push('p-att-store');
    writeAttachments('p-att-store', [textFile('s1', 100)]);
    writeAttachments('p-att-other', [textFile('s2', 100)]);
    expect(readAttachments('p-att-store').map((item) => item.id)).toEqual(['s1']);
    const before = placeEpoch('p-att-store');
    forgetPlace('p-att-store');
    expect(readAttachments('p-att-store')).toEqual([]);
    writeAttachments('p-att-store', [textFile('s3', 100)], before);
    expect(readAttachments('p-att-store')).toEqual([]);
    writeAttachments('p-att-store', [textFile('s6', 100)]);
    expect(readAttachments('p-att-store').map((item) => item.id)).toEqual(['s6']);
    expect(readAttachments('p-att-other').map((item) => item.id)).toEqual(['s2']);
    forgetPlace('p-att-other');
    writeAttachments(NEW_PLACE, [textFile('s4', 100, 'gone.txt'), textFile('s5', 100, 'stays.txt')]);
    await open('/chat?mode=chat', (text) => text.includes(CHAT.heroChatTitle));
    click(button(fill(CHAT.attach.remove, { name: 'gone.txt' })));
    expect(readAttachments(NEW_PLACE).map((item) => item.id)).toEqual(['s5']);
    await waitFor(() => !chipText().includes('gone.txt'));
    expect(chipText()).toContain('stays.txt');
  });

  it('a refused attachment says so in the strip, with no action', () => {
    expect(sendFailure('attachment-refused', 'E_BAD_REQUEST', 'chat')).toEqual({
      strip: { tone: 'danger', text: CHAT.sendErrors['attachment-refused'][0], action: undefined },
    });
    expect(sendFailure('pictures-full', 'E_CONFLICT', 'chat')).toEqual({
      strip: { tone: 'danger', text: CHAT.sendErrors['pictures-full'][0], action: undefined },
    });
  });

  // Mutation: drop the page's count of the pictures the chat holds → Send posts the picture.
  it('a chat that holds as many pictures as it can keep says so before a picture is sent; a file still goes', async () => {
    const full = sentWith('many', [
      { kind: 'image', mediaType: 'image/png', bytes: (CHAT_PICTURES_MAX_BASE64 / 4) * 3 },
    ]);
    writeAttachments('p-att-full', [picture('pic-full')]);
    await open('/chat?c=p-att-full', (text) => text.includes('Hi there'), {
      host: visionHost(true),
      routes: { '/chat/conversations/p-att-full/prompt': {} },
      conversations: { 'p-att-full': sync('p-att-full', { entries: [full, answer('Hi there'), end()] }) },
    });
    await waitFor(() => chipText().includes('cat.png'));
    type('One more');
    expect(shown(form())).toContain(CHAT.sendErrors['pictures-full'][0]);
    click(sendFace());
    await flushTimers();
    expect(writes()).toEqual([]);
    click(button(fill(CHAT.attach.remove, { name: 'cat.png' })));
    pick(new File(['notes'], 'notes.txt', { type: 'text/plain' }));
    await waitFor(() => chipText().includes('notes.txt'));
    expect(shown(form())).not.toContain(CHAT.sendErrors['pictures-full'][0]);
    click(sendFace());
    await waitFor(() => writes().length === 1);
  });
});
