/** @vitest-environment happy-dom */

import { act, createElement, type ReactElement } from 'react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { ChatHostState, ChatMode, ConversationInfo, ConversationRow } from '../ui/src/lib/chat-types.js';
import { toast } from '../ui/src/lib/toast.js';
import Chat from '../ui/src/pages/chat.js';
import { modeBlocker, sendFailure } from '../ui/src/pages/chat/blockers.js';
import { webChip } from '../ui/src/pages/chat/composer-tools.js';
import { forgetPlace, NEW_PLACE, readDraft, writeDraft } from '../ui/src/pages/chat/drafts.js';
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

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  page?.unmount();
  api?.();
  page = undefined;
  api = undefined;
  writeDraft(NEW_PLACE, '');
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

describe('Chat page: Escape and the stop hint', () => {
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

  it('Escape closes an open popover, and a second Escape on its trigger stops nothing', async () => {
    await openRunning('p-esc-pop');
    const chip = button(CHAT.webOff);
    click(chip);
    await waitFor(() => byRole(document.body, 'dialog', CHAT.webSettings) !== null);
    key(byRole(document.body, 'dialog', CHAT.webSettings)!, { key: 'Escape' });
    await waitFor(() => byRole(document.body, 'dialog', CHAT.webSettings) === null);
    key(chip, { key: 'Escape' });
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

  it('the hint names Esc only while an answer runs and the composer has the focus', async () => {
    await openRunning('p-esc-hint');
    const hint = (): boolean => shown().includes(`${CHAT.keyNames.esc} ${CHAT.hintStop[0].label}`);
    act(() => composer().focus());
    await waitFor(hint);
    act(() => composer().blur());
    await waitFor(() => !hint());
    expect(shown()).not.toContain(CHAT.hintSend[1].label);
    expect(accessibleName(sendFace())).toBe(CHAT.stop);
  });
});

describe('Chat page: a message that waits for its model', () => {
  it('Stop during the load keeps the bubble, says it was not sent, and Send again posts the same text', async () => {
    await open('/chat?c=p-unsent', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-unsent/prompt': {}, '/chat/conversations/p-unsent/cancel': {} },
      conversations: { 'p-unsent': sync('p-unsent') },
      host: host({ inference: 'loading', loading: { modelId: 'qwen3.5-4b', startedAt: NOW - 3_000 } }),
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
      active: [{ conversationId: 'p-other', mode: 'chat', busy: true, waiting: false }],
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

  it('starts no interval timer', async () => {
    const spy = vi.spyOn(globalThis, 'setInterval');
    await open('/chat?c=p-timer', (text) => text.includes('Hi there'), {
      conversations: { 'p-timer': sync('p-timer') },
    });
    act(() =>
      api!.emitChat('p-timer', {
        type: 'turn',
        seq: 1,
        turn: { turnId: 't', phase: 'running', startedAt: NOW, wait: null },
      }),
    );
    await waitFor(() => accessibleName(sendFace()) === CHAT.stop);
    expect(spy).not.toHaveBeenCalled();
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

  it('the web chip follows the provider selection', () => {
    const base = READY_CHAT_HOST.providers;
    expect(webChip(base)).toMatchObject({
      label: CHAT.webOff,
      state: 'off',
      action: { label: CHAT.webSetUp, to: '/providers' },
    });
    expect(webChip({ ...base, firecrawl: true, fetch: 'firecrawl' })).toMatchObject({ label: CHAT.webPagesOnly });
    expect(webChip({ ...base, search: 'exa', searchCheck: 'ok', fetch: 'exa' })).toMatchObject({
      label: 'Web · Exa',
      lines: [fill(CHAT.webChatSame, { search: 'Exa' })],
    });
    expect(webChip({ ...base, search: 'brave', searchCheck: 'ok', fetch: 'brave' }).lines).toEqual([
      fill(CHAT.webChatExcerpts, { search: 'Brave' }),
    ]);
    expect(webChip({ ...base, search: 'brave', searchCheck: 'ok', firecrawl: true, fetch: 'firecrawl' }).lines).toEqual(
      [fill(CHAT.webChatBoth, { search: 'Brave', fetch: 'Firecrawl' })],
    );
    expect(webChip({ ...base, search: 'exa', searchCheck: 'rejected', fetch: 'exa' })).toMatchObject({
      name: 'Web, Exa, key rejected',
      dot: 'danger',
      action: { label: CHAT.webFix },
    });
    expect(webChip({ ...base, search: 'exa', searchCheck: 'no-credit', fetch: 'exa' })).toMatchObject({
      dot: 'warning',
    });
    expect(webChip({ ...base, applyPending: true }).pending).toBe(true);
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
    expect(byRole(page!.container, 'button', 'Web, Brave')).not.toBeNull();
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
