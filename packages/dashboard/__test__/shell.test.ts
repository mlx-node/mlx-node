/** @vitest-environment happy-dom */

import { Storage } from 'happy-dom';
import {
  act,
  createElement,
  Fragment,
  Suspense,
  useContext,
  useLayoutEffect,
  useState,
  type ReactElement,
} from 'react';
import { flushSync } from 'react-dom';
import {
  BrowserRouter,
  MemoryRouter,
  type Navigator,
  Route,
  Routes,
  UNSAFE_NavigationContext,
  useLocation,
  useNavigate,
  useSearchParams,
} from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import { ToastStack } from '../ui/src/components/arc/toast-stack/toast-stack.js';
import { CHAT } from '../ui/src/lib/chat-strings.js';
import type { ChatHostState, ConversationInfo, ConversationRow, RewindPlan } from '../ui/src/lib/chat-types.js';
import { modelColor, resetModelColors } from '../ui/src/lib/model-colors.js';
import { toast } from '../ui/src/lib/toast.js';
import type {
  EngineStatus,
  MetricsModels,
  ModelsResponse,
  RuntimeLive,
  RuntimeProcessLive,
  SessionDetailResponse,
} from '../ui/src/lib/types.js';
import Chat from '../ui/src/pages/chat.js';
import { forgetPlace, NEW_PLACE, readDraft } from '../ui/src/pages/chat/drafts.js';
import { historyActions } from '../ui/src/pages/chat/history-actions.js';
import { chatHostSnapshot } from '../ui/src/pages/chat/use-chat-host.js';
import { forgetConversation } from '../ui/src/pages/chat/use-conversation.js';
import Engine from '../ui/src/pages/engine.js';
import Models from '../ui/src/pages/models.js';
import SessionDetail from '../ui/src/pages/session-detail.js';
import { AppShell } from '../ui/src/shell/app-shell.js';
import { matchHotkey } from '../ui/src/shell/hotkeys.js';
import { Page, PageHeader } from '../ui/src/shell/page.js';
import { useScrollRoot } from '../ui/src/shell/scroll-root.js';
import { footerLine } from '../ui/src/shell/status-footer.js';
import type { ShellStatus } from '../ui/src/shell/use-shell-status.js';
import { answer, end, metric, user } from './helpers/chat-thread-fixtures.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import {
  accessibleName,
  allByRole,
  byRole,
  type ChatSync,
  renderPage,
  type RenderedPage,
  deferred,
  observeWidth,
  READY_CHAT_HOST,
  skipAnimations,
  STUB_FAILURE,
  stubApi,
  type StubbedApi,
  waitFor,
} from './render.js';

interface HappyWindow {
  happyDOM: { setViewport(viewport: { width: number; height: number }): void };
}

function setViewport(width: number, height: number): void {
  (window as unknown as HappyWindow).happyDOM.setViewport({ width, height });
  window.dispatchEvent(new Event('resize'));
}

let page: RenderedPage | undefined;
let api: StubbedApi | undefined;
let location = '';
let search = '';

beforeEach(() => {
  vi.stubGlobal('localStorage', new Storage());
  setViewport(1180, 800);
});

afterEach(async () => {
  page?.unmount();
  page = undefined;
  const closed = api?.closed;
  api?.();
  api = undefined;
  await closed;
  await new Promise((resolve) => setImmediate(resolve));
  vi.restoreAllMocks();
  toast.dismiss();
  vi.unstubAllGlobals();
  document.title = '';
});

function Probe({ title, crumbs }: { title: string; crumbs?: { label: string; to?: string }[] }): ReactElement {
  return createElement(Page, null, createElement(PageHeader, { title, crumbs, description: `${title} page` }));
}

function LocationSpy(): null {
  const where = useLocation();
  location = where.pathname;
  search = where.search;
  return null;
}

function shell(initial: string): ReactElement {
  return createElement(
    MemoryRouter,
    { initialEntries: [initial] },
    createElement(LocationSpy),
    createElement(
      Routes,
      null,
      createElement(
        Route,
        { element: createElement(AppShell) },
        createElement(Route, { index: true, element: createElement(Probe, { title: 'Overview probe' }) }),
        createElement(Route, { path: 'engine', element: createElement(Engine) }),
        createElement(Route, { path: 'metrics', element: createElement(Probe, { title: 'Metrics probe' }) }),
        createElement(Route, {
          path: 'sessions/:id',
          element: createElement(Probe, {
            title: 'Fix the parser',
            crumbs: [{ label: 'Sessions', to: '/sessions' }],
          }),
        }),
      ),
    ),
  );
}

async function mount(
  initial: string,
  until: string,
  routes: Record<string, unknown> = {},
  onCall?: (call: ApiCall) => void,
): Promise<RenderedPage> {
  api = stubApi({ '/downloads': { jobs: [] }, ...routes }, { onCall });
  page = await renderPage(shell(initial), (text) => text.includes(until), { scrollRoot: false });
  return page;
}

function sections(container: ParentNode): HTMLElement {
  const nav = container.querySelector<HTMLElement>('nav[aria-label="Sections"]');
  if (nav === null) throw new Error('no Sections nav');
  return nav;
}

function navLabels(container: ParentNode): string[] {
  return [...sections(container).querySelectorAll('a')].map((link) => (link.textContent ?? '').trim());
}

function current(container: ParentNode): string[] {
  return [...sections(container).querySelectorAll('[aria-current="page"]')].map((link) =>
    (link.textContent ?? '').trim(),
  );
}

function crumbs(container: ParentNode): string[] {
  const nav = container.querySelector('nav[aria-label="Location"]');
  return [...(nav?.querySelectorAll('li') ?? [])].map((item) => (item.textContent ?? '').trim());
}

async function press(key: string, meta = true): Promise<void> {
  await act(async () => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key, metaKey: meta, bubbles: true }));
  });
}

function process(overrides: Partial<RuntimeProcessLive> = {}): RuntimeProcessLive {
  const model = (name: string, inFlight: number): RuntimeProcessLive['models'][number] => ({
    name,
    modelType: 'qwen3_5',
    ctxWindow: 32768,
    paged: true,
    maxConcurrent: 4,
    loadedAt: 1,
    loadMs: 900,
    inFlight,
    queued: 0,
    queueLimit: null,
    pool: null,
    experts: null,
  });
  return {
    status: 'running',
    ageMs: 0,
    pid: 4242,
    startedAt: 1,
    ts: 2,
    host: 'serve',
    version: '0.0.0',
    port: 52114,
    persistPagedCache: true,
    coldCapture: { blocks: 63_125, ms: 10_000 },
    health: 'ok',
    mem: { active: 1, peak: 1, cache: 0, wiredLimit: 1, rss: 1 },
    work: { ms: { loading: 0, waiting: 0, prepare: 0, prefill: 0, decode: 0, running: 0 } },
    rates: { windowMs: 10_000, outputTps: null, prefillTps: null },
    loading: [],
    counters: { ok: 0, error: 0, cancelled: 0, outputTokens: 0, loads: 1, rejected: 0 },
    models: [model('qwen3.5-9b', 1), model('gemma-4-e4b', 0)],
    turns: [],
    ...overrides,
  };
}

function snapshot(status: EngineStatus, processes: RuntimeProcessLive[]): RuntimeLive {
  return { now: 3, status, supervisor: null, processes };
}

describe('shell navigation', () => {
  it('lists the ten nav items in table order and marks the current one', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    expect(navLabels(container)).toEqual([
      'Overview',
      'Chat',
      'System One',
      'Sessions',
      'Metrics',
      'Engine',
      'Cache',
      'Models',
      'Coding Agents',
      'Web Search',
    ]);
    expect(current(container)).toEqual(['Engine']);
  });

  it('has no tab stop in the sidebar before the first nav link', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    const aside = container.querySelector('aside[data-nav]');
    const stops = [...(aside?.querySelectorAll<HTMLElement>('a[href], button, input, [tabindex]') ?? [])].filter(
      (node) => node.tabIndex >= 0,
    );
    expect(stops[0]?.textContent?.trim()).toBe('Overview');
    expect(stops[0]?.tagName).toBe('A');
  });

  it('marks the parent item of a detail route and puts the parent in the breadcrumb', async () => {
    const { container } = await mount('/sessions/abc', 'Fix the parser page');
    expect(current(container)).toEqual(['Sessions']);
    await waitFor(() => crumbs(container).join(' / ') === 'Sessions / Fix the parser');
    const parent = container.querySelector('nav[aria-label="Location"] a');
    expect(parent?.getAttribute('href')).toBe('/sessions');
  });

  it('shows the page title as the one crumb of a top-level page', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    await waitFor(() => crumbs(container).join(' / ') === 'Engine');
  });

  it('is a rail below 880 px: no collapse toggle, labels kept for assistive tech', async () => {
    setViewport(720, 480);
    const { container } = await mount('/engine', 'The inference process, live.');
    expect(container.querySelector('aside[data-nav]')?.getAttribute('data-mode')).toBe('rail');
    expect(byRole(container, 'button', 'Collapse sidebar')).toBeNull();
    expect(byRole(container, 'button', 'Search')).not.toBeNull();
    expect(navLabels(container)).toHaveLength(10);
  });

  it('is expanded at 1180 px and the toggle turns it into a rail', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    const aside = container.querySelector('aside[data-nav]');
    expect(aside?.getAttribute('data-mode')).toBe('expanded');
    await act(async () => byRole(container, 'button', 'Collapse sidebar')?.click());
    expect(aside?.getAttribute('data-mode')).toBe('rail');
    expect(localStorage.getItem('mlx-node:sidebar:v1')).toBe('rail');
    expect(byRole(container, 'button', 'Expand sidebar')).not.toBeNull();
  });

  it('labels, badges and the footer text fade in after the width starts, on mount, only after a mode switch', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    const aside = container.querySelector('aside[data-nav]');
    expect(aside?.hasAttribute('data-switched')).toBe(false);
    await act(async () => byRole(container, 'button', 'Collapse sidebar')?.click());
    expect(aside?.hasAttribute('data-switched')).toBe(true);
    await act(async () => byRole(container, 'button', 'Expand sidebar')?.click());
    expect(aside?.getAttribute('data-mode')).toBe('expanded');
    expect(aside?.hasAttribute('data-switched')).toBe(true);

    const rules = cssRules('shell/sidebar.module.css');
    for (const part of ['.label', '.badge', '.statusText']) {
      expect(cssValue(rules, `.root[data-switched][data-mode='expanded'] ${part}`, 'animation')).toBe(
        'nav-label-in var(--duration-instant) var(--ease-enter) var(--duration-instant) both',
      );
    }
    expect(cssValue(rules, ".root[data-switched][data-mode='rail'] .label", 'animation')).toBe(
      'nav-label-out var(--duration-instant) var(--ease-standard) both',
    );
  });

  it('under reduced motion the labels switch: no fade, no slide', () => {
    const rules = cssRules('shell/sidebar.module.css');
    for (const selector of ['.sectionText', ".root[data-mode='rail'] .sectionText", ".root[data-mode='rail'] .label"]) {
      expect(cssValue(rules, selector, 'transition', 'reduce')).toBe('none');
      expect(cssValue(rules, selector, 'translate', 'reduce')).toBe('none');
    }
    for (const selector of [
      ".root[data-switched][data-mode='expanded'] .label",
      ".root[data-switched][data-mode='expanded'] .badge",
      ".root[data-switched][data-mode='expanded'] .statusText",
      ".root[data-switched][data-mode='rail'] .label",
    ]) {
      expect(cssValue(rules, selector, 'animation', 'reduce')).toBe('none');
    }
  });
});

describe('page swap', () => {
  it('replaces the page in one commit, moves focus to main and renames the window', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    expect(document.title).toBe('Engine — Portato');
    const metrics = sections(container).querySelector<HTMLAnchorElement>('a[href="/metrics"]');
    await act(async () => metrics?.click());
    const headings = allByRole(container, 'heading').map((heading) => heading.textContent);
    expect(headings).toContain('Metrics probe');
    expect(headings).not.toContain('Engine');
    expect(document.activeElement?.tagName).toBe('MAIN');
    expect(document.activeElement?.hasAttribute('data-scroll-root')).toBe(true);
    expect(document.title).toBe('Metrics probe — Portato');
    expect(current(container)).toEqual(['Metrics']);
    await waitFor(
      () => container.querySelector('[aria-live="polite"][role="status"]')?.textContent === 'Metrics probe',
    );
  });

  it('freezes an exiting page on its own location while fading out, and updates search params live on the same path', async () => {
    const previous = skipAnimations(false);
    // happy-dom's WAAPI stub rejects Animation.finished on cancel() and desyncs document.timeline from performance.now;
    // falling back to Motion's JS animation driver runs the real AnimatePresence exit lifecycle cleanly.
    const animateDescriptor = Object.getOwnPropertyDescriptor(Element.prototype, 'animate');
    Reflect.deleteProperty(Element.prototype, 'animate');
    onTestFinished(() => {
      if (animateDescriptor !== undefined) Object.defineProperty(Element.prototype, 'animate', animateDescriptor);
      skipAnimations(previous);
    });

    const seen: Array<{ pathname: string; search: string; c: string | null }> = [];
    let bump: (() => void) | undefined;
    let go: ((to: string) => void) | undefined;

    function ChatProbe(): ReactElement {
      const where = useLocation();
      const [params] = useSearchParams();
      const navigate = useNavigate();
      const [, setTick] = useState(0);
      bump = () => setTick((count) => count + 1);
      go = (to: string) => void navigate(to);
      seen.push({ pathname: where.pathname, search: where.search, c: params.get('c') });
      return createElement(
        Page,
        null,
        createElement(PageHeader, { title: 'Chat probe', description: `chat:${params.get('c') ?? 'none'}` }),
      );
    }

    api = stubApi({ '/downloads': { jobs: [] } });
    page = await renderPage(
      createElement(
        MemoryRouter,
        { initialEntries: ['/chat?c=conv-1'] },
        createElement(LocationSpy),
        createElement(
          Routes,
          null,
          createElement(
            Route,
            { element: createElement(AppShell) },
            createElement(Route, { path: 'chat', element: createElement(ChatProbe) }),
            createElement(Route, { path: 'engine', element: createElement(Probe, { title: 'Engine probe' }) }),
          ),
        ),
      ),
      (text) => text.includes('chat:conv-1'),
      { scrollRoot: false },
    );
    const { container } = page;
    expect(seen.at(-1)).toEqual({ pathname: '/chat', search: '?c=conv-1', c: 'conv-1' });

    await act(async () => go!('/chat?c=conv-2'));
    expect(location).toBe('/chat');
    expect(search).toBe('?c=conv-2');
    expect(container.textContent).toContain('chat:conv-2');
    expect(seen.at(-1)).toEqual({ pathname: '/chat', search: '?c=conv-2', c: 'conv-2' });

    seen.length = 0;
    await act(async () => go!('/engine'));
    expect(location).toBe('/engine');
    expect(search).toBe('');
    expect(container.textContent).toContain('chat:conv-2');
    expect(container.textContent).not.toContain('chat:none');
    expect(container.textContent).not.toContain('Engine probe');

    await act(async () => bump!());
    expect(seen.length).toBeGreaterThan(0);
    expect(
      seen.every((entry) => entry.pathname === '/chat' && entry.search === '?c=conv-2' && entry.c === 'conv-2'),
    ).toBe(true);
    expect(container.textContent).toContain('chat:conv-2');
    expect(container.textContent).not.toContain('chat:none');

    await waitFor(() => container.textContent?.includes('Engine probe') === true, 3_000);
    expect(
      seen.every((entry) => entry.pathname === '/chat' && entry.search === '?c=conv-2' && entry.c === 'conv-2'),
    ).toBe(true);
  });

  function enableAnimatedPageSwaps(): void {
    const previous = skipAnimations(false);
    // happy-dom's WAAPI stub rejects Animation.finished on cancel() and desyncs document.timeline from performance.now;
    // falling back to Motion's JS animation driver runs the real AnimatePresence exit lifecycle cleanly.
    const animateDescriptor = Object.getOwnPropertyDescriptor(Element.prototype, 'animate');
    Reflect.deleteProperty(Element.prototype, 'animate');
    onTestFinished(() => {
      if (animateDescriptor !== undefined) Object.defineProperty(Element.prototype, 'animate', animateDescriptor);
      skipAnimations(previous);
    });
  }

  function chatConvInfo(id: string, over: Partial<ConversationInfo> = {}): ConversationInfo {
    const now = Date.now();
    return {
      id,
      mode: 'chat',
      title: `Title ${id}`,
      cwd: null,
      modelId: 'qwen3.5-4b',
      createdAt: now - 60_000,
      updatedAt: now - 60_000,
      hot: false,
      permissionMode: 'ask',
      ...over,
    };
  }

  function chatConvRow(id: string, over: Partial<ConversationInfo> = {}): ConversationRow {
    return { ...chatConvInfo(id, over), turns: 1 };
  }

  function chatConvSync(id: string, reply: string, over: Partial<ConversationInfo> = {}): ChatSync {
    return {
      type: 'sync',
      seq: 0,
      conversation: chatConvInfo(id, over),
      entries: [user('Hello'), answer(reply), end()],
      turns: [metric()],
      turn: null,
      interactions: [],
      notices: [],
      transcriptError: null,
      permissionMarks: [],
      runs: [],
      live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
    };
  }

  async function startForkAndLeaveToEngine(
    fromId: string,
    copyId: string,
  ): Promise<{ container: HTMLElement; release: () => void; notifications: () => HTMLElement }> {
    enableAnimatedPageSwaps();
    onTestFinished(() => {
      forgetPlace(fromId);
      forgetPlace(copyId);
      forgetConversation(fromId);
      forgetConversation(copyId);
    });

    const held = deferred({ id: copyId });
    let forkPosted = false;
    api = stubApi(
      {
        '/downloads': { jobs: [] },
        '/chat/conversations': {
          conversations: [
            chatConvRow(copyId, { forkOf: { id: fromId, title: `Title ${fromId}` } }),
            chatConvRow(fromId),
          ],
        },
        [`POST /chat/conversations/${fromId}/fork`]: held.body,
      },
      {
        onCall: (call) => {
          if (call.method === 'POST' && call.path.endsWith(`/chat/conversations/${fromId}/fork`)) forkPosted = true;
        },
        chat: {
          host: READY_CHAT_HOST,
          conversations: {
            [fromId]: chatConvSync(fromId, `Answer from ${fromId}`),
            [copyId]: chatConvSync(copyId, `Answer from ${copyId}`, {
              forkOf: { id: fromId, title: `Title ${fromId}` },
            }),
          },
        },
      },
    );

    page = await renderPage(
      createElement(
        Fragment,
        null,
        createElement(
          MemoryRouter,
          { initialEntries: [`/chat?c=${fromId}`] },
          createElement(LocationSpy),
          createElement(
            Routes,
            null,
            createElement(
              Route,
              { element: createElement(AppShell) },
              createElement(Route, { path: 'chat', element: createElement(Chat) }),
              createElement(Route, { path: 'engine', element: createElement(Engine) }),
            ),
          ),
        ),
        createElement(ToastStack),
      ),
      (text) => text.includes(`Answer from ${fromId}`),
      { scrollRoot: false },
    );
    const { container } = page;
    const wantHost = JSON.stringify(READY_CHAT_HOST);
    await waitFor(() => JSON.stringify(chatHostSnapshot().state) === wantHost, 2_000);

    const composer = byRole(container, 'textbox', CHAT.composerLabel) as HTMLTextAreaElement;
    await act(async () => {
      Reflect.set(HTMLTextAreaElement.prototype, 'value', '/fork', composer);
      composer.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => {
      composer.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    await waitFor(() => forkPosted);

    const notifications = (): HTMLElement => {
      const region = container.querySelector<HTMLElement>('section[aria-label^="Notifications"]');
      if (region === null) throw new Error('no Notifications region');
      return region;
    };
    await waitFor(() => notifications().textContent?.includes(CHAT.history.forking) === true);

    const engineLink = sections(container).querySelector<HTMLAnchorElement>('a[href="/engine"]');
    await act(async () => engineLink?.click());
    expect(location).toBe('/engine');
    expect(search).toBe('');

    return { container, release: held.release, notifications };
  }

  it('stays on Engine and offers the toast Open action when an in-flight Chat fork finishes after leaving Chat', async () => {
    const fromId = 'shell-fork-unmounted';
    const copyId = 'shell-fork-unmounted-copy';
    const { container, release, notifications } = await startForkAndLeaveToEngine(fromId, copyId);

    // Wait for Chat to finish its exit animation and unmount under FrozenLocation.
    await waitFor(() => container.textContent?.includes('The inference process, live.') === true, 3_000);
    expect(location).toBe('/engine');
    expect(search).toBe('');

    release();
    await waitFor(
      () =>
        notifications().textContent?.includes(CHAT.history.forkedAway) === true &&
        byRole(notifications(), 'button', CHAT.history.forkOpen) !== null,
    );
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
    expect(location).toBe('/engine');
    expect(search).toBe('');
    expect(container.textContent).toContain('The inference process, live.');

    await act(async () => byRole(notifications(), 'button', CHAT.history.forkOpen)?.click());
    expect(location).toBe('/chat');
    expect(search).toBe(`?c=${copyId}`);
    await waitFor(() => container.textContent?.includes(`Answer from ${copyId}`) === true, 3_000);
  });

  it('stays on Engine and offers the toast Open action when an in-flight Chat fork finishes during Chat exit animation', async () => {
    const fromId = 'shell-fork-exiting';
    const copyId = 'shell-fork-exiting-copy';
    const { container, release, notifications } = await startForkAndLeaveToEngine(fromId, copyId);

    // Chat is still mounted and frozen while fading out; Engine has not mounted yet.
    expect(container.textContent).toContain(`Answer from ${fromId}`);
    expect(container.textContent).not.toContain('The inference process, live.');

    release();
    await waitFor(
      () =>
        notifications().textContent?.includes(CHAT.history.forkedAway) === true &&
        byRole(notifications(), 'button', CHAT.history.forkOpen) !== null,
    );
    await waitFor(() => container.textContent?.includes('The inference process, live.') === true, 3_000);
    expect(location).toBe('/engine');
    expect(search).toBe('');

    await act(async () => byRole(notifications(), 'button', CHAT.history.forkOpen)?.click());
    expect(location).toBe('/chat');
    expect(search).toBe(`?c=${copyId}`);
    await waitFor(() => container.textContent?.includes(`Answer from ${copyId}`) === true, 3_000);
  });

  it('restores a failed send prompt into the composer when navigating back during Chat exit', async () => {
    enableAnimatedPageSwaps();
    const convId = 'shell-send-fail-return';
    onTestFinished(() => {
      forgetPlace(convId);
      forgetConversation(convId);
    });

    let nav: ReturnType<typeof useNavigate> | undefined;
    function NavSpy(): null {
      nav = useNavigate();
      return null;
    }

    const failPrompt = deferred(STUB_FAILURE);
    let promptPosted = false;
    api = stubApi(
      {
        '/downloads': { jobs: [] },
        '/chat/conversations': { conversations: [chatConvRow(convId)] },
        [`POST /chat/conversations/${convId}/prompt`]: failPrompt.body,
      },
      {
        onCall: (call) => {
          if (call.method === 'POST' && call.path.endsWith(`/chat/conversations/${convId}/prompt`)) promptPosted = true;
        },
        chat: {
          host: READY_CHAT_HOST,
          conversations: { [convId]: chatConvSync(convId, 'Initial reply') },
        },
      },
    );

    page = await renderPage(
      createElement(
        Fragment,
        null,
        createElement(
          MemoryRouter,
          { initialEntries: [`/chat?c=${convId}`] },
          createElement(LocationSpy),
          createElement(NavSpy),
          createElement(
            Routes,
            null,
            createElement(
              Route,
              { element: createElement(AppShell) },
              createElement(Route, { path: 'chat', element: createElement(Chat) }),
              createElement(Route, { path: 'engine', element: createElement(Engine) }),
            ),
          ),
        ),
        createElement(ToastStack),
      ),
      (text) => text.includes('Initial reply'),
      { scrollRoot: false },
    );
    const { container } = page;
    const wantHost = JSON.stringify(READY_CHAT_HOST);
    await waitFor(() => JSON.stringify(chatHostSnapshot().state) === wantHost, 2_000);

    const composer = byRole(container, 'textbox', CHAT.composerLabel) as HTMLTextAreaElement;
    await act(async () => {
      Reflect.set(HTMLTextAreaElement.prototype, 'value', 'Keep this prompt', composer);
      composer.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => {
      composer.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    await waitFor(() => promptPosted && composer.value === '');

    const engineLink = sections(container).querySelector<HTMLAnchorElement>('a[href="/engine"]');
    await act(async () => engineLink?.click());
    expect(location).toBe('/engine');
    expect(search).toBe('');
    expect(container.textContent).toContain('Initial reply');
    expect(container.textContent).not.toContain('The inference process, live.');

    failPrompt.release();
    await waitFor(() => readDraft(convId) === 'Keep this prompt');
    expect(container.textContent).toContain('Initial reply');
    expect(container.textContent).not.toContain('The inference process, live.');

    await act(async () => void nav!(-1));
    expect(location).toBe('/chat');
    expect(search).toBe(`?c=${convId}`);
    const returnedBox = byRole(container, 'textbox', CHAT.composerLabel) as HTMLTextAreaElement;
    expect(returnedBox).toBe(composer);
    await waitFor(() => returnedBox.value === 'Keep this prompt');

    await act(async () => {
      Reflect.set(HTMLTextAreaElement.prototype, 'value', 'Keep this prompt!', returnedBox);
      returnedBox.dispatchEvent(new Event('input', { bubbles: true }));
    });
    expect(returnedBox.value).toBe('Keep this prompt!');
    expect(readDraft(convId)).toBe('Keep this prompt!');
  });

  it('keeps Engine active when a hook-blocked prompt arrives during Chat exit and saves the prompt to drafts', async () => {
    enableAnimatedPageSwaps();
    const convId = 'shell-hook-block-exit';
    onTestFinished(() => {
      forgetPlace(convId);
      forgetConversation(convId);
    });

    let nav: ReturnType<typeof useNavigate> | undefined;
    function NavSpy(): null {
      nav = useNavigate();
      return null;
    }

    let promptPosted = false;
    api = stubApi(
      {
        '/downloads': { jobs: [] },
        '/chat/conversations': { conversations: [chatConvRow(convId)] },
        [`POST /chat/conversations/${convId}/prompt`]: { turnId: 't1' },
      },
      {
        onCall: (call) => {
          if (call.method === 'POST' && call.path.endsWith(`/chat/conversations/${convId}/prompt`)) promptPosted = true;
        },
        chat: {
          host: READY_CHAT_HOST,
          conversations: { [convId]: chatConvSync(convId, 'Ready for work') },
        },
      },
    );

    page = await renderPage(
      createElement(
        Fragment,
        null,
        createElement(
          MemoryRouter,
          { initialEntries: [`/chat?c=${convId}`] },
          createElement(LocationSpy),
          createElement(NavSpy),
          createElement(
            Routes,
            null,
            createElement(
              Route,
              { element: createElement(AppShell) },
              createElement(Route, { path: 'chat', element: createElement(Chat) }),
              createElement(Route, { path: 'engine', element: createElement(Engine) }),
            ),
          ),
        ),
        createElement(ToastStack),
      ),
      (text) => text.includes('Ready for work'),
      { scrollRoot: false },
    );
    const { container } = page;
    const wantHost = JSON.stringify(READY_CHAT_HOST);
    await waitFor(() => JSON.stringify(chatHostSnapshot().state) === wantHost, 2_000);

    const composer = byRole(container, 'textbox', CHAT.composerLabel) as HTMLTextAreaElement;
    await act(async () => {
      Reflect.set(HTMLTextAreaElement.prototype, 'value', 'deploy to prod', composer);
      composer.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => {
      composer.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    await waitFor(() => promptPosted && composer.value === '');

    const engineLink = sections(container).querySelector<HTMLAnchorElement>('a[href="/engine"]');
    await act(async () => engineLink?.click());
    expect(location).toBe('/engine');
    expect(search).toBe('');
    expect(container.textContent).toContain('Ready for work');
    expect(container.textContent).not.toContain('The inference process, live.');

    act(() => {
      api!.emitChat(convId, {
        type: 'turn',
        seq: 1,
        turn: null,
        turnId: 't1',
        outcome: 'hook-stopped',
        blocked: 'deploy to prod',
      });
    });

    await waitFor(() => readDraft(convId) === 'deploy to prod');
    expect(location).toBe('/engine');
    expect(search).toBe('');
    expect(container.textContent).not.toContain('The inference process, live.');

    await waitFor(() => container.textContent?.includes('The inference process, live.') === true, 3_000);
    expect(location).toBe('/engine');
    expect(search).toBe('');
    expect(readDraft(convId)).toBe('deploy to prod');

    await act(async () => void nav!(`/chat?c=${convId}`));
    await waitFor(
      () =>
        (byRole(container, 'textbox', CHAT.composerLabel) as HTMLTextAreaElement | null)?.value === 'deploy to prod',
      3_000,
    );
  });

  it('drops navigator.replace from an exiting Models page so a late response does not overwrite Engine in history', async () => {
    enableAnimatedPageSwaps();
    const restoreWidth = observeWidth(1180);
    onTestFinished(restoreWidth);

    let nav: ReturnType<typeof useNavigate> | undefined;
    function NavSpy(): null {
      nav = useNavigate();
      return null;
    }

    const modelName = 'qwen3.8-27b-gguf';
    const modelsBody: ModelsResponse = {
      models: [
        {
          name: modelName,
          path: `/models/${modelName}`,
          modelType: 'qwen3_5',
          quant: 'Q4_K_XL',
          contextWindow: 262_144,
          sizeBytes: 17 * 1024 ** 3,
          fileCount: 2,
          servedIds: [modelName],
          located: null,
          link: false,
        },
      ],
      warnings: [],
      dir: '/models',
    };
    const heldModels = deferred(modelsBody);
    let modelsRequested = false;

    api = stubApi(
      {
        '/downloads': { jobs: [] },
        '/models': heldModels.body,
        '/catalog': { items: [] },
        '/catalog/updates': { items: [] },
        '/metrics/models': { models: [] },
        '/metrics/summary': {
          turns: 0,
          promptTokens: 0,
          completionTokens: 0,
          cachedTokens: 0,
          errors: 0,
          avgTtftMs: null,
          avgTps: null,
          byModel: [],
        },
        '/metrics/timeseries': { bucket: 'day', points: [] },
      },
      {
        onCall: (call) => {
          if (call.method === 'GET' && call.path.split('?')[0].endsWith('/models')) modelsRequested = true;
        },
      },
    );

    page = await renderPage(
      createElement(
        MemoryRouter,
        { initialEntries: ['/models?focus=missing'] },
        createElement(LocationSpy),
        createElement(NavSpy),
        createElement(
          Routes,
          null,
          createElement(
            Route,
            { element: createElement(AppShell) },
            createElement(Route, { path: 'models', element: createElement(Models) }),
            createElement(Route, { path: 'engine', element: createElement(Engine) }),
          ),
        ),
      ),
      (text) => text.includes('Checkpoints on this Mac and the recommended list.'),
      { scrollRoot: false },
    );
    const { container } = page;
    await waitFor(() => modelsRequested);
    expect(location).toBe('/models');
    expect(search).toBe('?focus=missing');
    expect(container.textContent).not.toContain(modelName);

    const engineLink = sections(container).querySelector<HTMLAnchorElement>('a[href="/engine"]');
    await act(async () => engineLink?.click());
    expect(location).toBe('/engine');
    expect(search).toBe('');
    expect(container.textContent).toContain('Checkpoints on this Mac');
    expect(container.textContent).not.toContain('The inference process, live.');

    heldModels.release();
    await waitFor(() => container.textContent?.includes(modelName) === true);
    expect(container.textContent).not.toContain('The inference process, live.');
    expect(location).toBe('/engine');
    expect(search).toBe('');

    await waitFor(() => container.textContent?.includes('The inference process, live.') === true, 3_000);
    expect(location).toBe('/engine');
    expect(search).toBe('');

    await act(async () => void nav!(-1));
    await waitFor(() => container.textContent?.includes('Checkpoints on this Mac') === true, 3_000);
    await waitFor(() => location === '/models' && search === '');

    await act(async () => void nav!(1));
    expect(location).toBe('/engine');
    expect(search).toBe('');
    await waitFor(() => container.textContent?.includes('The inference process, live.') === true, 3_000);
  });

  it('drops a guarded replace fired in the pre-commit window after history.push updates navigator.location', async () => {
    enableAnimatedPageSwaps();
    let outerNavigator: Navigator | undefined;
    let childNavigate: ReturnType<typeof useNavigate> | undefined;

    function OuterNavigatorSpy(): null {
      outerNavigator = useContext(UNSAFE_NavigationContext).navigator;
      return null;
    }

    function ReplaceProbe(): ReactElement {
      childNavigate = useNavigate();
      const [params] = useSearchParams();
      return createElement(
        Page,
        null,
        createElement(PageHeader, { title: 'Replace probe', description: `focus:${params.get('focus') ?? 'none'}` }),
      );
    }

    api = stubApi({ '/downloads': { jobs: [] } });
    page = await renderPage(
      createElement(
        MemoryRouter,
        { initialEntries: ['/models?focus=init'] },
        createElement(LocationSpy),
        createElement(OuterNavigatorSpy),
        createElement(
          Routes,
          null,
          createElement(
            Route,
            { element: createElement(AppShell) },
            createElement(Route, { path: 'models', element: createElement(ReplaceProbe) }),
            createElement(Route, { path: 'engine', element: createElement(Probe, { title: 'Engine probe' }) }),
          ),
        ),
      ),
      (text) => text.includes('focus:init'),
      { scrollRoot: false },
    );

    // While still active on /models, replace works normally.
    await act(async () => void childNavigate!('/models?focus=kept', { replace: true }));
    expect(location).toBe('/models');
    expect(search).toBe('?focus=kept');

    // Push /engine synchronously on history, then fire replace from the old route before React commits.
    let livePathBeforeCommit: string | undefined;
    await act(async () => {
      outerNavigator!.push('/engine');
      livePathBeforeCommit = (outerNavigator as { location?: { pathname?: string } }).location?.pathname;
      void childNavigate!('/models?focus=stale', { replace: true });
    });

    expect(livePathBeforeCommit).toBe('/engine');
    expect(location).toBe('/engine');
    expect(search).toBe('');
    await waitFor(() => page!.container.textContent?.includes('Engine probe') === true, 3_000);
    expect(location).toBe('/engine');
    expect(search).toBe('');
  });

  it('drops a same-path stale replace fired in the pre-commit window while allowing back-to-back active replaces', async () => {
    let outerNavigator: Navigator | undefined;
    let childNavigate: ReturnType<typeof useNavigate> | undefined;

    function OuterNavigatorSpy(): null {
      outerNavigator = useContext(UNSAFE_NavigationContext).navigator;
      return null;
    }

    function ReplaceProbe(): ReactElement {
      childNavigate = useNavigate();
      const [params] = useSearchParams();
      return createElement(
        Page,
        null,
        createElement(PageHeader, { title: 'Replace probe', description: `focus:${params.get('focus') ?? 'none'}` }),
      );
    }

    api = stubApi({ '/downloads': { jobs: [] } });
    page = await renderPage(
      createElement(
        MemoryRouter,
        { initialEntries: ['/models?focus=a'] },
        createElement(LocationSpy),
        createElement(OuterNavigatorSpy),
        createElement(
          Routes,
          null,
          createElement(
            Route,
            { element: createElement(AppShell) },
            createElement(Route, { path: 'models', element: createElement(ReplaceProbe) }),
          ),
        ),
      ),
      (text) => text.includes('focus:a'),
      { scrollRoot: false },
    );

    // Back-to-back legitimate replaces on the active route in the same tick both succeed.
    await act(async () => {
      void childNavigate!('/models?focus=mid', { replace: true });
      void childNavigate!('/models?focus=kept', { replace: true });
    });
    expect(location).toBe('/models');
    expect(search).toBe('?focus=kept');

    // Push /models?focus=b synchronously on history, then fire a stale replace before React commits.
    let liveSearchBeforeCommit: string | undefined;
    await act(async () => {
      outerNavigator!.push('/models?focus=b');
      liveSearchBeforeCommit = (outerNavigator as { location?: { search?: string } }).location?.search;
      void childNavigate!('/models?focus=stale', { replace: true });
    });

    expect(liveSearchBeforeCommit).toBe('?focus=b');
    expect(location).toBe('/models');
    expect(search).toBe('?focus=b');
    expect(page.container.textContent).toContain('focus:b');

    // After React commits /models?focus=b, a fresh legitimate replace works again.
    await act(async () => void childNavigate!('/models?focus=after', { replace: true }));
    expect(location).toBe('/models');
    expect(search).toBe('?focus=after');
  });

  it('drops a stale replace fired during an uncommitted concurrent render after history moves to ?focus=b', async () => {
    window.history.replaceState({ usr: null, key: 'init-concurrent', idx: 0 }, '', '/models?focus=a');
    let resolveGate: (() => void) | undefined;
    let gateResolved = false;
    const gate = new Promise<void>((resolve) => {
      resolveGate = () => {
        gateResolved = true;
        resolve();
      };
    });
    onTestFinished(() => {
      resolveGate?.();
      window.history.replaceState(null, '', '/');
    });

    let outerNavigator: Navigator | undefined;
    let childNavigate: ReturnType<typeof useNavigate> | undefined;
    let committedSearch = '';
    let suspendedBAttempts = 0;

    function OuterNavigatorSpy(): null {
      outerNavigator = useContext(UNSAFE_NavigationContext).navigator;
      return null;
    }

    function CommittedLocationSpy(): null {
      const where = useLocation();
      useLayoutEffect(() => {
        committedSearch = where.search;
      }, [where.search]);
      return null;
    }

    function SuspendingProbe(): ReactElement {
      childNavigate = useNavigate();
      const [params] = useSearchParams();
      const focus = params.get('focus') ?? 'none';
      if (focus === 'b' && !gateResolved) {
        suspendedBAttempts += 1;
        throw gate;
      }
      return createElement(
        Page,
        null,
        createElement(PageHeader, { title: 'Concurrent probe', description: `focus:${focus}` }),
      );
    }

    api = stubApi({ '/downloads': { jobs: [] } });
    page = await renderPage(
      createElement(
        BrowserRouter,
        null,
        createElement(LocationSpy),
        createElement(CommittedLocationSpy),
        createElement(OuterNavigatorSpy),
        createElement(
          Routes,
          null,
          createElement(
            Route,
            { element: createElement(AppShell) },
            createElement(Route, {
              path: 'models',
              element: createElement(
                Suspense,
                { fallback: createElement('p', null, 'loading') },
                createElement(SuspendingProbe),
              ),
            }),
          ),
        ),
      ),
      (text) => text.includes('focus:a'),
      { scrollRoot: false },
    );

    // Move BrowserRouter history to ?focus=b; FrozenLocation renders ?focus=b concurrently, then the child suspends before commit.
    await act(async () => {
      outerNavigator!.push('/models?focus=b');
    });
    expect(suspendedBAttempts).toBeGreaterThan(0);
    expect(window.location.pathname).toBe('/models');
    expect(window.location.search).toBe('?focus=b');
    expect(committedSearch).toBe('?focus=a');
    expect(page.container.textContent).toContain('focus:a');
    expect(page.container.textContent).not.toContain('focus:b');
    expect(page.container.textContent).not.toContain('loading');

    // A stale replace from committed ?focus=a must be dropped so history stays on ?focus=b.
    await act(async () => {
      void childNavigate!('/models?focus=stale', { replace: true });
    });
    expect(window.location.pathname).toBe('/models');
    expect(window.location.search).toBe('?focus=b');
    expect((outerNavigator as { location?: { search?: string } }).location?.search).toBe('?focus=b');

    // Once ?focus=b finishes committing, a fresh legitimate replace works normally.
    await act(async () => {
      resolveGate!();
      await gate;
    });
    await waitFor(() => page!.container.textContent?.includes('focus:b') === true);
    expect(location).toBe('/models');
    expect(search).toBe('?focus=b');
    expect(committedSearch).toBe('?focus=b');
    expect(window.location.search).toBe('?focus=b');

    await act(async () => void childNavigate!('/models?focus=after', { replace: true }));
    expect(location).toBe('/models');
    expect(search).toBe('?focus=after');
    expect(committedSearch).toBe('?focus=after');
    expect(window.location.search).toBe('?focus=after');
  });

  it('preserves an accepted replace across an intervening urgent re-render of the committed location so a second replace still succeeds', async () => {
    window.history.replaceState({ usr: null, key: 'init-urgent', idx: 0 }, '', '/models?focus=a');
    onTestFinished(() => {
      window.history.replaceState(null, '', '/');
    });

    let childNavigate: ReturnType<typeof useNavigate> | undefined;
    let bumpUrgent: (() => void) | undefined;
    let committedSearch = '';
    let committedARenders = 0;

    function CommittedLocationSpy(): null {
      const where = useLocation();
      useLayoutEffect(() => {
        committedSearch = where.search;
      }, [where.search]);
      return null;
    }

    function UrgentProbe({ tick }: { tick: number }): ReactElement {
      childNavigate = useNavigate();
      const [params] = useSearchParams();
      const focus = params.get('focus') ?? 'none';
      if (focus === 'a') committedARenders += 1;
      return createElement(
        Page,
        null,
        createElement(PageHeader, { title: 'Urgent probe', description: `focus:${focus}:tick:${tick}` }),
      );
    }

    function ShellHost(): ReactElement {
      const [tick, setTick] = useState(0);
      bumpUrgent = () => setTick((count) => count + 1);
      return createElement(
        Routes,
        null,
        createElement(
          Route,
          { element: createElement(AppShell) },
          createElement(Route, {
            path: 'models',
            element: createElement(UrgentProbe, { tick }),
          }),
        ),
      );
    }

    api = stubApi({ '/downloads': { jobs: [] } });
    page = await renderPage(
      createElement(
        BrowserRouter,
        null,
        createElement(LocationSpy),
        createElement(CommittedLocationSpy),
        createElement(ShellHost),
      ),
      (text) => text.includes('focus:a:tick:0'),
      { scrollRoot: false },
    );

    const rendersBefore = committedARenders;
    let urgentSearchBeforeSecondReplace = '';
    let urgentDomBeforeSecondReplace = '';
    let liveAfterFirstReplace = '';
    let liveAfterSecondReplace = '';

    // First replace moves history to ?focus=a1 (pending transition), an urgent flushSync re-renders committed ?focus=a,
    // and a second legitimate replace (a1 -> a2) runs before ?focus=a1 commits.
    await act(async () => {
      void childNavigate!('/models?focus=a1', { replace: true });
      liveAfterFirstReplace = window.location.search;
      flushSync(() => {
        bumpUrgent!();
      });
      urgentSearchBeforeSecondReplace = committedSearch;
      urgentDomBeforeSecondReplace = page!.container.textContent ?? '';
      void childNavigate!('/models?focus=a2', { replace: true });
      liveAfterSecondReplace = window.location.search;
    });

    expect(liveAfterFirstReplace).toBe('?focus=a1');
    expect(committedARenders).toBeGreaterThan(rendersBefore);
    expect(urgentSearchBeforeSecondReplace).toBe('?focus=a');
    expect(urgentDomBeforeSecondReplace).toContain('focus:a:tick:1');
    expect(liveAfterSecondReplace).toBe('?focus=a2');
    expect(location).toBe('/models');
    expect(search).toBe('?focus=a2');
    expect(committedSearch).toBe('?focus=a2');
    expect(window.location.search).toBe('?focus=a2');
    expect(page.container.textContent).toContain('focus:a2:tick:1');
  });

  it('stays on Engine and shows the Open toast when an in-flight BrowserRouter fork resolves in the pre-commit window', async () => {
    const fromId = 'shell-browser-fork-from';
    const copyId = 'shell-browser-fork-copy';
    window.history.replaceState({ usr: null, key: 'init-chat', idx: 0 }, '', `/chat?c=${fromId}`);
    onTestFinished(() => {
      window.history.replaceState(null, '', '/');
      forgetPlace(fromId);
      forgetPlace(copyId);
      forgetConversation(fromId);
      forgetConversation(copyId);
    });

    const held = deferred({ id: copyId });
    let forkPosted = false;
    api = stubApi(
      {
        '/downloads': { jobs: [] },
        '/chat/conversations': {
          conversations: [
            chatConvRow(copyId, { forkOf: { id: fromId, title: `Title ${fromId}` } }),
            chatConvRow(fromId),
          ],
        },
        [`POST /chat/conversations/${fromId}/fork`]: held.body,
      },
      {
        onCall: (call) => {
          if (call.method === 'POST' && call.path.endsWith(`/chat/conversations/${fromId}/fork`)) forkPosted = true;
        },
        chat: {
          host: READY_CHAT_HOST,
          conversations: {
            [fromId]: chatConvSync(fromId, `Answer from ${fromId}`),
            [copyId]: chatConvSync(copyId, `Answer from ${copyId}`, {
              forkOf: { id: fromId, title: `Title ${fromId}` },
            }),
          },
        },
      },
    );

    page = await renderPage(
      createElement(
        Fragment,
        null,
        createElement(
          BrowserRouter,
          null,
          createElement(LocationSpy),
          createElement(
            Routes,
            null,
            createElement(
              Route,
              { element: createElement(AppShell) },
              createElement(Route, { path: 'chat', element: createElement(Chat) }),
              createElement(Route, { path: 'engine', element: createElement(Engine) }),
            ),
          ),
        ),
        createElement(ToastStack),
      ),
      (text) => text.includes(`Answer from ${fromId}`),
      { scrollRoot: false },
    );
    const { container } = page;
    const wantHost = JSON.stringify(READY_CHAT_HOST);
    await waitFor(() => JSON.stringify(chatHostSnapshot().state) === wantHost, 2_000);

    const composer = byRole(container, 'textbox', CHAT.composerLabel) as HTMLTextAreaElement;
    await act(async () => {
      Reflect.set(HTMLTextAreaElement.prototype, 'value', '/fork', composer);
      composer.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => {
      composer.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    await waitFor(() => forkPosted);

    const notifications = (): HTMLElement => {
      const region = container.querySelector<HTMLElement>('section[aria-label^="Notifications"]');
      if (region === null) throw new Error('no Notifications region');
      return region;
    };
    await waitFor(() => notifications().textContent?.includes(CHAT.history.forking) === true);

    // Move browser history to /engine BEFORE React commits the route transition so Chat's React state is still on fromId.
    const nextIdx = ((window.history.state as { idx?: number } | null)?.idx ?? 0) + 1;
    window.history.pushState({ usr: null, key: 'precommit-engine', idx: nextIdx }, '', '/engine');
    expect(location).toBe('/chat');
    expect(search).toBe(`?c=${fromId}`);
    expect(window.location.pathname).toBe('/engine');

    held.release();
    await waitFor(
      () =>
        notifications().textContent?.includes(CHAT.history.forkedAway) === true &&
        byRole(notifications(), 'button', CHAT.history.forkOpen) !== null,
    );
    expect(window.location.pathname).toBe('/engine');
    expect(window.location.search).toBe('');

    // Let BrowserRouter commit /engine, then confirm Open from the toast opens the forked copy.
    await act(async () => {
      window.dispatchEvent(new PopStateEvent('popstate'));
    });
    await waitFor(() => container.textContent?.includes('The inference process, live.') === true, 3_000);
    expect(location).toBe('/engine');
    expect(window.location.pathname).toBe('/engine');

    await act(async () => byRole(notifications(), 'button', CHAT.history.forkOpen)?.click());
    await waitFor(() => location === '/chat' && search === `?c=${copyId}`);
    expect(window.location.pathname).toBe('/chat');
    expect(window.location.search).toBe(`?c=${copyId}`);
    await waitFor(() => container.textContent?.includes(`Answer from ${copyId}`) === true, 3_000);
  });

  it('stays on Engine when a first-send BrowserRouter conversation create resolves in the pre-commit window', async () => {
    const newId = 'shell-browser-create-new';
    window.history.replaceState({ usr: null, key: 'init-new-chat', idx: 0 }, '', '/chat?mode=chat');
    let resolveEngineGate: (() => void) | undefined;
    let engineGateResolved = false;
    const engineGate = new Promise<void>((resolve) => {
      resolveEngineGate = () => {
        engineGateResolved = true;
        resolve();
      };
    });
    onTestFinished(() => {
      resolveEngineGate?.();
      window.history.replaceState(null, '', '/');
      forgetPlace(newId);
      forgetPlace(NEW_PLACE);
      forgetConversation(newId);
    });

    const heldCreate = deferred({ conversation: chatConvInfo(newId), turnId: 't1' });
    let createPosted = false;
    let listCalls = 0;
    api = stubApi(
      {
        '/downloads': { jobs: [] },
        '/chat/conversations': { conversations: [] },
        'POST /chat/conversations': heldCreate.body,
      },
      {
        onCall: (call) => {
          if (call.method === 'POST' && call.path.split('?')[0].endsWith('/chat/conversations')) createPosted = true;
          if (call.method === 'GET' && call.path.split('?')[0].endsWith('/chat/conversations')) listCalls += 1;
        },
        chat: {
          host: READY_CHAT_HOST,
          conversations: {
            [newId]: chatConvSync(newId, `Answer from ${newId}`),
          },
        },
      },
    );

    let committedPath = '';
    let committedSearch = '';
    function CommittedLocationSpy(): null {
      const where = useLocation();
      useLayoutEffect(() => {
        committedPath = where.pathname;
        committedSearch = where.search;
      }, [where.pathname, where.search]);
      return null;
    }

    let engineSuspended = 0;
    function GatedEngine(): ReactElement {
      if (!engineGateResolved) {
        engineSuspended += 1;
        throw engineGate;
      }
      return createElement(Engine);
    }

    page = await renderPage(
      createElement(
        Fragment,
        null,
        createElement(
          BrowserRouter,
          null,
          createElement(LocationSpy),
          createElement(CommittedLocationSpy),
          createElement(
            Suspense,
            { fallback: createElement('p', null, 'loading') },
            createElement(
              Routes,
              null,
              createElement(
                Route,
                { element: createElement(AppShell) },
                createElement(Route, { path: 'chat', element: createElement(Chat) }),
                createElement(Route, { path: 'engine', element: createElement(GatedEngine) }),
              ),
            ),
          ),
        ),
        createElement(ToastStack),
      ),
      (text) => text.length > 0 && document.querySelector(`[aria-label="${CHAT.composerLabel}"]`) !== null,
      { scrollRoot: false },
    );
    const { container } = page;
    const wantHost = JSON.stringify(READY_CHAT_HOST);
    await waitFor(() => JSON.stringify(chatHostSnapshot().state) === wantHost, 2_000);

    const composer = byRole(container, 'textbox', CHAT.composerLabel) as HTMLTextAreaElement;
    await act(async () => {
      Reflect.set(HTMLTextAreaElement.prototype, 'value', 'Hello first message', composer);
      composer.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => {
      composer.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    await waitFor(() => createPosted);

    // Click Engine: BrowserRouter updates window.location to /engine immediately, while GatedEngine holds commit pending.
    const engineLink = sections(container).querySelector<HTMLAnchorElement>('a[href="/engine"]');
    await act(async () => engineLink?.click());
    expect(engineSuspended).toBeGreaterThan(0);
    expect(window.location.pathname).toBe('/engine');
    expect(window.location.search).toBe('');
    expect(committedPath).toBe('/chat');
    expect(committedSearch).toBe('?mode=chat');
    expect(byRole(container, 'textbox', CHAT.composerLabel)).not.toBeNull();
    expect(container.textContent).not.toContain('loading');
    expect(container.textContent).not.toContain('The inference process, live.');

    // Resolve POST /chat/conversations inside the pre-commit window; Chat must drop navigation and leave URL on /engine.
    const listCallsBefore = listCalls;
    heldCreate.release();
    await waitFor(() => listCalls > listCallsBefore);
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
    expect(window.location.pathname).toBe('/engine');
    expect(window.location.search).toBe('');
    expect(committedPath).toBe('/chat');
    expect(committedSearch).toBe('?mode=chat');

    // Release the pending /engine commit and verify the app settles on Engine, not /chat?c=<newId>.
    await act(async () => {
      resolveEngineGate!();
      await engineGate;
    });
    await waitFor(() => container.textContent?.includes('The inference process, live.') === true, 3_000);
    expect(committedPath).toBe('/engine');
    expect(committedSearch).toBe('');
    expect(location).toBe('/engine');
    expect(search).toBe('');
    expect(window.location.pathname).toBe('/engine');
    expect(window.location.search).toBe('');
  });

  it('opens the newly created conversation when starting a first send from /chat/?mode=chat on BrowserRouter', async () => {
    const newId = 'shell-browser-trailing-slash-new';
    window.history.replaceState({ usr: null, key: 'init-trailing-slash-chat', idx: 0 }, '', '/chat/?mode=chat');
    onTestFinished(() => {
      window.history.replaceState(null, '', '/');
      forgetPlace(newId);
      forgetPlace(NEW_PLACE);
      forgetConversation(newId);
    });

    let createPosted = false;
    api = stubApi(
      {
        '/downloads': { jobs: [] },
        '/chat/conversations': { conversations: [] },
        'POST /chat/conversations': { conversation: chatConvInfo(newId), turnId: 't1' },
      },
      {
        onCall: (call) => {
          if (call.method === 'POST' && call.path.split('?')[0].endsWith('/chat/conversations')) createPosted = true;
        },
        chat: {
          host: READY_CHAT_HOST,
          conversations: {
            [newId]: chatConvSync(newId, `Answer from ${newId}`),
          },
        },
      },
    );

    page = await renderPage(
      createElement(
        Fragment,
        null,
        createElement(
          BrowserRouter,
          null,
          createElement(LocationSpy),
          createElement(
            Routes,
            null,
            createElement(
              Route,
              { element: createElement(AppShell) },
              createElement(Route, { path: 'chat', element: createElement(Chat) }),
            ),
          ),
        ),
        createElement(ToastStack),
      ),
      (text) => text.length > 0 && document.querySelector(`[aria-label="${CHAT.composerLabel}"]`) !== null,
      { scrollRoot: false },
    );
    const { container } = page;
    const wantHost = JSON.stringify(READY_CHAT_HOST);
    await waitFor(() => JSON.stringify(chatHostSnapshot().state) === wantHost, 2_000);
    expect(window.location.pathname).toBe('/chat/');
    expect(window.location.search).toBe('?mode=chat');

    const composer = byRole(container, 'textbox', CHAT.composerLabel) as HTMLTextAreaElement;
    await act(async () => {
      Reflect.set(HTMLTextAreaElement.prototype, 'value', 'Hello from trailing slash', composer);
      composer.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => {
      composer.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    await waitFor(() => createPosted);
    await waitFor(() => location === '/chat' && search === `?c=${newId}`, 3_000);
    expect(window.location.pathname).toBe('/chat');
    expect(window.location.search).toBe(`?c=${newId}`);
    await waitFor(() => container.textContent?.includes(`Answer from ${newId}`) === true, 3_000);
  });

  it('under reduced motion drops SessionDetail delete replace after unmounting to Engine', async () => {
    const previous = skipAnimations(true);
    onTestFinished(() => {
      skipAnimations(previous);
    });

    const sessionId = '019a0000-0000-7000-8000-00000000dead';
    const detail: SessionDetailResponse = {
      session: {
        id: sessionId,
        origin: 'agent',
        title: 'Refactor the planner',
        cwd: '/work/mlx-node',
        model: 'qwen3.8-27b',
        branch: 'main',
        createdAt: '2026-10-20T09:30:00.000Z',
        lastActiveAt: '2026-10-20T11:00:00.000Z',
        messageCount: 2,
        live: false,
        kind: null,
        parentId: null,
        subagentCount: 0,
        tokens: { input: 100, output: 50, cached: 0 },
        gitRoot: '/work/mlx-node',
        headCommit: 'abc1234',
        remotes: [],
        agentName: null,
        reasoningEffort: null,
        lastTurnSummary: null,
        parent: null,
        subagents: [],
      },
      transcript: [{ kind: 'user', ts: Date.UTC(2026, 9, 20, 10), text: 'Hello', promptIndex: 0 }],
      turns: [],
      resumeCommand: null,
    };
    const heldDelete = deferred({ deleted: [sessionId] });
    let deleteCalled = false;

    api = stubApi(
      {
        '/downloads': { jobs: [] },
        [`/sessions/${sessionId}`]: detail,
        [`DELETE /sessions/${sessionId}`]: heldDelete.body,
      },
      {
        onCall: (call) => {
          if (call.method === 'DELETE' && call.path.endsWith(`/sessions/${sessionId}`)) deleteCalled = true;
        },
      },
    );

    page = await renderPage(
      createElement(
        Fragment,
        null,
        createElement(
          MemoryRouter,
          { initialEntries: [`/sessions/${sessionId}`] },
          createElement(LocationSpy),
          createElement(
            Routes,
            null,
            createElement(
              Route,
              { element: createElement(AppShell) },
              createElement(Route, { path: 'sessions/:id', element: createElement(SessionDetail) }),
              createElement(Route, { path: 'sessions', element: createElement(Probe, { title: 'Sessions probe' }) }),
              createElement(Route, { path: 'engine', element: createElement(Engine) }),
            ),
          ),
        ),
        createElement(ToastStack),
      ),
      (text) => text.includes('Refactor the planner'),
      { scrollRoot: false },
    );
    const { container } = page;

    const trigger = byRole(container, 'button', 'Delete');
    expect(trigger).not.toBeNull();
    await act(async () => trigger!.click());
    await waitFor(() => document.querySelector('[data-radix-popper-content-wrapper]') !== null);
    const popover = document.querySelector<HTMLElement>('[data-radix-popper-content-wrapper]')!;
    const confirmButton = byRole(popover, 'button', 'Delete');
    expect(confirmButton).not.toBeNull();
    await act(async () => confirmButton!.click());
    await waitFor(() => deleteCalled);

    const engineLink = sections(container).querySelector<HTMLAnchorElement>('a[href="/engine"]');
    await act(async () => engineLink?.click());
    expect(location).toBe('/engine');
    expect(container.textContent).toContain('The inference process, live.');
    expect(container.textContent).not.toContain('Refactor the planner');

    const notifications = (): HTMLElement => {
      const region = container.querySelector<HTMLElement>('section[aria-label^="Notifications"]');
      if (region === null) throw new Error('no Notifications region');
      return region;
    };

    heldDelete.release();
    await waitFor(() => notifications().textContent?.includes('Session deleted') === true);
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
    expect(location).toBe('/engine');
    expect(container.textContent).toContain('The inference process, live.');
    expect(container.textContent).not.toContain('Sessions probe');
  });

  it('restores and focuses the remounted Chat composer when clicking Put message back after leaving and returning', async () => {
    const previous = skipAnimations(true);
    const convId = 'shell-rewind-remount';
    onTestFinished(() => {
      skipAnimations(previous);
      forgetPlace(convId);
      forgetConversation(convId);
    });

    let nav: ReturnType<typeof useNavigate> | undefined;
    function NavSpy(): null {
      nav = useNavigate();
      return null;
    }

    const plan: RewindPlan = {
      token: 'v1:3',
      removes: { messages: 1, steps: 0 },
      background: { commands: 0, schedules: 0, runs: 0 },
      helpers: false,
      attachments: false,
      files: [],
      filesMore: 0,
    };
    let rewindPosted = false;
    api = stubApi(
      {
        '/downloads': { jobs: [] },
        '/chat/conversations': { conversations: [chatConvRow(convId)] },
        [`POST /chat/conversations/${convId}/rewind/plan`]: plan,
        [`POST /chat/conversations/${convId}/rewind`]: {},
      },
      {
        onCall: (call) => {
          if (call.method === 'POST' && call.path.endsWith(`/chat/conversations/${convId}/rewind`)) rewindPosted = true;
        },
        chat: {
          host: READY_CHAT_HOST,
          conversations: {
            [convId]: {
              ...chatConvSync(convId, 'First answer'),
              entries: [user('Original prompt'), answer('First answer'), end()],
            },
          },
        },
      },
    );

    page = await renderPage(
      createElement(
        Fragment,
        null,
        createElement(
          MemoryRouter,
          { initialEntries: [`/chat?c=${convId}`] },
          createElement(LocationSpy),
          createElement(NavSpy),
          createElement(
            Routes,
            null,
            createElement(
              Route,
              { element: createElement(AppShell) },
              createElement(Route, { path: 'chat', element: createElement(Chat) }),
              createElement(Route, { path: 'engine', element: createElement(Engine) }),
            ),
          ),
        ),
        createElement(ToastStack),
      ),
      (text) => text.includes('First answer'),
      { scrollRoot: false },
    );
    const { container } = page;
    const wantHost = JSON.stringify(READY_CHAT_HOST);
    await waitFor(() => JSON.stringify(chatHostSnapshot().state) === wantHost, 2_000);

    const firstComposer = byRole(container, 'textbox', CHAT.composerLabel) as HTMLTextAreaElement;
    await act(async () => {
      Reflect.set(HTMLTextAreaElement.prototype, 'value', 'unsent draft', firstComposer);
      firstComposer.dispatchEvent(new Event('input', { bubbles: true }));
    });
    expect(firstComposer.value).toBe('unsent draft');

    act(() => historyActions.rewindLast(convId));
    await waitFor(() => {
      const dlg = byRole(document.body, 'dialog', CHAT.history.rewindTitle);
      return dlg !== null && byRole(dlg, 'button', CHAT.history.confirm) !== null;
    });
    const dlg = byRole(document.body, 'dialog', CHAT.history.rewindTitle)!;
    await act(async () => byRole(dlg, 'button', CHAT.history.confirm)!.click());
    await waitFor(() => rewindPosted);

    act(() => {
      api!.emitChat(convId, {
        ...chatConvSync(convId, 'First answer'),
        seq: 1,
        entries: [],
        turns: [],
      });
      api!.emitChat(convId, {
        type: 'rewind',
        seq: 2,
        n: 1,
        end: { outcome: 'done', mode: 'conversation', restored: 0, entry: 0 },
      });
    });

    const notifications = (): HTMLElement => {
      const region = container.querySelector<HTMLElement>('section[aria-label^="Notifications"]');
      if (region === null) throw new Error('no Notifications region');
      return region;
    };
    await waitFor(
      () =>
        byRole(document.body, 'dialog', CHAT.history.rewindTitle) === null &&
        byRole(notifications(), 'button', CHAT.history.putBack) !== null,
    );
    expect(firstComposer.value).toBe('unsent draft');

    // Navigate away to Engine so Chat Mount #1 unmounts completely.
    const engineLink = sections(container).querySelector<HTMLAnchorElement>('a[href="/engine"]');
    await act(async () => engineLink?.click());
    expect(location).toBe('/engine');
    expect(container.textContent).toContain('The inference process, live.');
    expect(firstComposer.isConnected).toBe(false);

    // Navigate back to the same conversation so Chat Mount #2 mounts fresh.
    await act(async () => void nav!(`/chat?c=${convId}`));
    await waitFor(() => byRole(container, 'textbox', CHAT.composerLabel) !== null);
    const secondComposer = byRole(container, 'textbox', CHAT.composerLabel) as HTMLTextAreaElement;
    expect(secondComposer).not.toBe(firstComposer);
    expect(secondComposer.value).toBe('unsent draft');

    // Wait for Mount #2's landing focus timer, then move focus onto the toast action before clicking it.
    await waitFor(() => document.activeElement === secondComposer);
    const putBackButton = byRole(notifications(), 'button', CHAT.history.putBack)!;
    act(() => putBackButton.focus());
    expect(document.activeElement).toBe(putBackButton);

    await act(async () => putBackButton.click());
    await waitFor(() => secondComposer.value === 'Original prompt' && document.activeElement === secondComposer);
    expect(readDraft(convId)).toBe('Original prompt');
    expect(location).toBe('/chat');
    expect(search).toBe(`?c=${convId}`);
  });
});

describe('keyboard', () => {
  it('⌘K opens the command palette and toggles it closed', async () => {
    await mount('/engine', 'The inference process, live.');
    await press('k');
    const palette = document.querySelector('[role="dialog"]');
    expect(palette).not.toBeNull();
    expect(accessibleName(palette!)).toBe('Command palette');
    expect(allByRole(palette!, 'option').some((option) => accessibleName(option).startsWith('Engine'))).toBe(true);
    await press('k');
    await waitFor(() => document.querySelector('[role="dialog"]') === null);
  });

  it('the palette opens with focus in its search field and Escape gives focus back to the element focused before', async () => {
    await mount('/engine', 'The inference process, live.');
    const link = [...sections(document).querySelectorAll<HTMLAnchorElement>('a')].find(
      (anchor) => (anchor.textContent ?? '').trim() === 'Metrics',
    );
    expect(link).toBeDefined();
    act(() => link!.focus());
    expect(document.activeElement).toBe(link);
    await press('k');
    await waitFor(() => document.activeElement instanceof HTMLInputElement);
    expect(document.querySelector('[role="dialog"]')?.contains(document.activeElement)).toBe(true);
    await act(async () => {
      document.activeElement!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });
    await waitFor(() => document.querySelector('[role="dialog"]') === null);
    await waitFor(() => document.activeElement === link);
  });

  it('⌘2 goes to Chat', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    const link = sections(container).querySelector<HTMLAnchorElement>('a[href="/chat"]');
    expect((link?.textContent ?? '').trim()).toBe('Chat');
    await press('2');
    expect(location).toBe('/chat');
  });

  it('⌘3 goes to System One, the third nav item', async () => {
    await mount('/engine', 'The inference process, live.');
    await press('3');
    expect(location).toBe('/system-one');
  });

  it('⌘4 goes to Sessions', async () => {
    await mount('/engine', 'The inference process, live.');
    await press('4');
    expect(location).toBe('/sessions');
  });

  it('⌘5 goes to Metrics', async () => {
    await mount('/engine', 'The inference process, live.');
    await press('5');
    expect(location).toBe('/metrics');
    expect(page!.text()).toContain('Metrics probe page');
  });

  it('⌘9 goes to Coding Agents, the ninth nav item', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    const link = sections(container).querySelector<HTMLAnchorElement>('a[href="/coding-agents"]');
    expect((link?.textContent ?? '').trim()).toBe('Coding Agents');
    await press('9');
    expect(location).toBe('/coding-agents');
  });

  it('⌘⇧O opens a new conversation in the last used mode from any page', async () => {
    localStorage.setItem('mlx-node:chat-mode:v1', 'work');
    await mount('/engine', 'The inference process, live.');
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'O', metaKey: true, shiftKey: true, bubbles: true }));
    });
    expect(location).toBe('/chat');
    expect(search).toBe('?mode=work');
  });

  it('the Chat item carries a live dot: waiting for you before running', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    const chatItem = (): HTMLElement => sections(container).querySelector<HTMLElement>('a[href="/chat"]')!;
    const dot = (): string | null => chatItem().querySelector('[role="img"]')?.getAttribute('aria-label') ?? null;
    expect(dot()).toBeNull();
    const state = (active: ChatHostState['active']): ChatHostState => ({ ...READY_CHAT_HOST, active });
    api!.emitChatHost({
      type: 'state',
      state: state([
        {
          conversationId: 'a',
          mode: 'chat',
          busy: true,
          waiting: false,
          asks: 0,
          background: false,
          helpers: false,
          tasks: 0,
          op: null,
        },
        {
          conversationId: 'b',
          mode: 'chat',
          busy: true,
          waiting: false,
          asks: 0,
          background: false,
          helpers: false,
          tasks: 0,
          op: null,
        },
      ]),
    });
    await waitFor(() => dot() === '2 running');
    api!.emitChatHost({
      type: 'state',
      state: state([
        {
          conversationId: 'a',
          mode: 'chat',
          busy: true,
          waiting: false,
          asks: 0,
          background: false,
          helpers: false,
          tasks: 0,
          op: null,
        },
        {
          conversationId: 'w',
          mode: 'work',
          busy: true,
          waiting: true,
          asks: 1,
          background: false,
          helpers: false,
          tasks: 0,
          op: null,
        },
      ]),
    });
    await waitFor(() => dot() === '1 waiting for you');
    api!.emitChatHost({ type: 'state', state: state([]) });
    await waitFor(() => dot() === null);
  });

  it('a binding without meta never fires while typing; meta bindings that allow it do', () => {
    const run = (): void => {};
    const input = document.createElement('input');
    const plain = { key: '/', run };
    const meta = { key: 'k', meta: true, allowInInput: true, run };
    const typing = (key: string, metaKey: boolean): KeyboardEvent => {
      const event = new KeyboardEvent('keydown', { key, metaKey });
      Object.defineProperty(event, 'target', { value: input });
      return event;
    };
    expect(matchHotkey([plain], typing('/', false))).toBeUndefined();
    expect(matchHotkey([meta], typing('k', true))).toBe(meta);
    expect(matchHotkey([meta], new KeyboardEvent('keydown', { key: 'k' }))).toBeUndefined();
  });
});

describe('status footer', () => {
  const base: ShellStatus = { engine: undefined, primary: null, runningDownloads: null, connected: true, chat: null };

  it.each<[EngineStatus | undefined, string, string | null, boolean]>([
    ['loading', 'Loading model…', null, true],
    ['starting', 'Starting inference…', null, true],
    ['restarting', 'Restarting inference…', null, true],
    ['unresponsive', 'Inference not responding', null, false],
    ['failed', 'Inference failed', 'Open Engine for details', false],
    ['untrusted', 'Output not trustworthy', 'Open Engine for details', false],
    ['stopped', 'Inference stopped', 'Start it from the menu bar', false],
    [undefined, 'Control Panel ready', null, false],
  ])('%s reads "%s"', (engine, line1, line2, working) => {
    const line = footerLine({ ...base, engine });
    expect([line.line1, line.line2, line.working]).toEqual([line1, line2, working]);
  });

  it('running names the port, the models and the active turns; a dropped port says Reconnecting', () => {
    const running = footerLine({ ...base, engine: 'running', primary: { port: 52114, modelCount: 1, inFlight: 0 } });
    expect([running.line1, running.line2, running.working]).toEqual(['Inference running', ':52114 · 1 model', false]);
    const busy = footerLine({ ...base, engine: 'running', primary: { port: 52114, modelCount: 2, inFlight: 3 } });
    expect(busy.line2).toBe('2 models · 3 active');
    expect(footerLine({ ...base, engine: 'running', connected: false }).line1).toBe('Reconnecting…');
  });

  it('follows the pushed live channel: status line, Engine badge, and a solid label under the harness', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    const status = (): string => accessibleName(container.querySelector('aside footer a[href="/engine"]')!);
    expect(status()).toBe('Control Panel ready');
    await waitFor(() => {
      api!.emitLive({ type: 'runtime', snapshot: snapshot('running', [process()]) });
      return status() === 'Inference running, 2 models · 1 active';
    });
    expect(byRole(sections(container), 'img', '1 active')).not.toBeNull();
    api!.emitLive({ type: 'runtime', snapshot: snapshot('loading', [process({ health: 'loading' })]) });
    await waitFor(() => status() === 'Loading model…');
    expect(container.querySelector('aside footer [aria-busy]')).toBeNull();
  });

  it('a port that dies says Reconnecting at once, with no focus or navigation, until a new port arrives', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    const status = (): string => accessibleName(container.querySelector('aside footer a[href="/engine"]')!);
    const banner = (): boolean => (container.textContent ?? '').includes('Reconnecting to the control panel service');
    await waitFor(() => {
      api!.emitLive({ type: 'runtime', snapshot: snapshot('running', [process()]) });
      return status() === 'Inference running, 2 models · 1 active';
    });
    expect(banner()).toBe(false);

    api!.dropPort();
    await waitFor(() => status() === 'Reconnecting…' && banner());
    expect(container.textContent).not.toContain('Inference running');

    api!();
    api = stubApi({ '/downloads': { jobs: [] } });
    await waitFor(() => !banner() && status() === 'Control Panel ready');
  });
});

describe('settings popover', () => {
  it('links to the Web Search page and closes on the way', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    await act(async () => byRole(container, 'button', 'Settings')?.click());
    await waitFor(() => document.querySelector('[aria-label="Settings"] a[href="/providers"]') !== null);
    const link = document.querySelector<HTMLAnchorElement>('[aria-label="Settings"] a[href="/providers"]')!;
    expect(link.textContent).toBe('Open Web Search settings');
    expect(link.closest('[aria-label="Settings"]')?.textContent).toContain('Web search');
    await act(async () => link.click());
    expect(location).toBe('/providers');
    await waitFor(() => document.querySelector('[aria-label="Settings"] a[href="/providers"]') === null);
  });

  it('separates its blocks with a rule in --border, which shows on the raised surface in both themes', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    await act(async () => byRole(container, 'button', 'Settings')?.click());
    await waitFor(() => document.querySelector('[aria-label="Settings"] hr') !== null);
    expect(document.querySelector('[aria-label="Settings"] hr')?.getAttribute('data-tone')).toBe('strong');
    const rules = cssRules('components/arc-x/rule/rule.module.css');
    expect(cssValue(rules, '.root', 'background')).toBe('var(--border-subtle)');
    expect(cssValue(rules, ".root[data-tone='strong']", 'background')).toBe('var(--border)');
  });
});

describe('settings data paths', () => {
  const HEALTH = {
    status: 'ok',
    modelsDir: '/Users/someone/.mlx-node/models',
    tracesDir: '/Users/someone/.mlx-node/metrics/traces',
    runtimeDir: '/Users/someone/.mlx-node/runtime',
    homeDir: '/Users/someone',
  };

  it('shows where the data lives, each path in full on hover and copyable', async () => {
    const { container } = await mount('/engine', 'The inference process, live.', { '/health': HEALTH });
    await act(async () => byRole(container, 'button', 'Settings')?.click());
    await waitFor(() => document.querySelector('[aria-label="Settings"] [title="' + HEALTH.tracesDir + '"]') !== null);
    const settings = [...document.querySelectorAll('[aria-label="Settings"]')].find(
      (el) => el.querySelector('hr') !== null,
    )!;
    expect(settings.textContent).toContain('Data');
    for (const path of [HEALTH.modelsDir, HEALTH.tracesDir, HEALTH.runtimeDir]) {
      expect(settings.querySelector('[title="' + path + '"]')).not.toBeNull();
    }
    // Short enough for the value column, so only one cut shows and the folder name stays.
    const traces = settings.querySelector('[title="' + HEALTH.tracesDir + '"]')!.textContent!;
    expect(traces.length).toBeLessThanOrEqual(23);
    expect(traces.endsWith('traces')).toBe(true);
    expect(allByRole(settings, 'button').filter((button) => accessibleName(button).startsWith('Copy')).length).toBe(3);
  });

  it('shows each path under the home folder as ~/… and copies the full path', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    const original = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    onTestFinished(() => {
      if (original === undefined) Reflect.deleteProperty(navigator, 'clipboard');
      else Object.defineProperty(navigator, 'clipboard', original);
    });
    const { container } = await mount('/engine', 'The inference process, live.', { '/health': HEALTH });
    await act(async () => byRole(container, 'button', 'Settings')?.click());
    const shown = (path: string): string =>
      document.querySelector('[aria-label="Settings"] [title="' + path + '"]')?.textContent ?? '';
    await waitFor(() => shown(HEALTH.modelsDir) === '~/.mlx-node/models');
    expect(shown(HEALTH.runtimeDir)).toBe('~/.mlx-node/runtime');
    expect(shown(HEALTH.tracesDir).startsWith('~/')).toBe(true);
    const settings = [...document.querySelectorAll('[aria-label="Settings"]')].find(
      (el) => el.querySelector('hr') !== null,
    )!;
    expect(settings.textContent).not.toContain('/Users/someone');
    await act(async () => byRole(settings, 'button', 'Copy Models')?.click());
    expect(writeText).toHaveBeenCalledWith(HEALTH.modelsDir);
  });

  it('leaves the Data block out when the paths cannot be read', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    await act(async () => byRole(container, 'button', 'Settings')?.click());
    await waitFor(() => document.querySelector('[aria-label="Settings"] hr') !== null);
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
    const settings = [...document.querySelectorAll('[aria-label="Settings"]')].find(
      (el) => el.querySelector('hr') !== null,
    );
    expect(settings?.textContent).toContain('Model colors');
    expect(settings?.textContent).not.toContain('Traces');
  });
});

describe('palette contents', () => {
  const MODELS = {
    models: [
      {
        name: 'gemma-4-e4b-it-4bit',
        path: '/m/gemma-4-e4b-it-4bit',
        modelType: 'gemma4',
        quant: '4bit',
        contextWindow: 131_072,
        sizeBytes: 5 * 1024 ** 3,
        fileCount: 3,
        servedIds: [],
        located: null,
      },
    ],
    warnings: [],
    dir: '/m',
  };
  const SESSION_ID = '019a0000-0000-7000-8000-00000000abcd';
  const SESSIONS = {
    sessions: [
      {
        id: SESSION_ID,
        title: '',
        cwd: '/work/planner-app',
        model: 'gemma-4-e4b-it-4bit',
        branch: 'main',
        createdAt: '2026-11-10T09:00:00.000Z',
        lastActiveAt: '2026-11-10T10:00:00.000Z',
        messageCount: 3,
        live: true,
        kind: null,
        parentId: null,
        subagentCount: 0,
        tokens: null,
      },
    ],
    total: 1,
    cwds: ['/work/planner-app'],
  };

  async function search(text: string): Promise<void> {
    await waitFor(() => document.activeElement instanceof HTMLInputElement);
    const input = document.activeElement as HTMLInputElement;
    await act(async () => {
      Reflect.set(HTMLInputElement.prototype, 'value', text, input);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
  }

  function options(): string[] {
    return allByRole(document, 'option').map((option) => accessibleName(option));
  }

  it('finds a local model and a session by what the user types', async () => {
    await mount('/engine', 'The inference process, live.', { '/models': MODELS, '/sessions': SESSIONS });
    await press('k');
    await waitFor(() => options().some((name) => name.startsWith('gemma-4-e4b-it-4bit')));
    await search('planner');
    const untitled = `Session ${SESSION_ID.slice(0, 8)}`;
    await waitFor(() => options().some((name) => name.startsWith(untitled)));
    const session = options().find((name) => name.startsWith(untitled))!;
    expect(session).not.toContain(SESSION_ID);
    expect(session).toContain('planner-app');
    expect(session).toContain('live');
    await search('gemma4');
    await waitFor(() => options().some((name) => name.startsWith('gemma-4-e4b-it-4bit')));
    const model = allByRole(document, 'option').find((option) =>
      accessibleName(option).startsWith('gemma-4-e4b-it-4bit'),
    )!;
    await act(async () => model.click());
    await waitFor(() => location === '/models');
  });
});

describe('model colors', () => {
  const traced: MetricsModels = {
    models: [
      { model: 'zeta-7b', modelType: 'qwen3', firstTs: 1, lastTs: 2, turns: 3 },
      { model: 'alpha-2b', modelType: 'lfm2', firstTs: 2, lastTs: 3, turns: 1 },
    ],
  };
  const local: ModelsResponse = {
    models: [
      {
        name: 'gguf-dir',
        path: '/m/gguf-dir',
        modelType: 'qwen3',
        quant: null,
        contextWindow: null,
        sizeBytes: 1,
        fileCount: 1,
        servedIds: ['gguf-dir/q8', 'gguf-dir/q4'],
        located: null,
        link: false,
      },
      {
        name: 'b-plain',
        path: '/m/b-plain',
        modelType: 'qwen3',
        quant: null,
        contextWindow: null,
        sizeBytes: 1,
        fileCount: 1,
        servedIds: [],
        located: null,
        link: false,
      },
      {
        name: 'zeta-7b',
        path: '/m/zeta-7b',
        modelType: 'qwen3',
        quant: null,
        contextWindow: null,
        sizeBytes: 1,
        fileCount: 1,
        servedIds: ['zeta-7b'],
        located: null,
        link: false,
      },
    ],
    warnings: [],
    dir: '/m',
  };

  beforeEach(() => resetModelColors());
  afterEach(() => resetModelColors());

  it('seeds traced models in first-seen order, then served ids sorted, even when the models list answers first', async () => {
    const held = deferred(traced);
    const calls: string[] = [];
    await mount('/engine', 'The inference process, live.', { '/metrics/models': held.body, '/models': local }, (call) =>
      calls.push(call.path),
    );
    await waitFor(() => calls.some((path) => path.endsWith('/models') && !path.includes('metrics')));
    for (let tick = 0; tick < 5; tick += 1) {
      await act(async () => {
        await new Promise((resolve) => {
          setTimeout(resolve, 0);
        });
      });
    }
    expect(localStorage.getItem('mlx-node:model-colors:v1')).toBeNull();
    held.release();
    await waitFor(() => localStorage.getItem('mlx-node:model-colors:v1') !== null);
    expect(JSON.parse(localStorage.getItem('mlx-node:model-colors:v1')!)).toEqual([
      'zeta-7b',
      'alpha-2b',
      'b-plain',
      'gguf-dir/q4',
      'gguf-dir/q8',
    ]);
    expect(modelColor('zeta-7b').ordinal).toBe(0);
    expect(modelColor('gguf-dir/q4').ordinal).toBe(3);
  });

  it('seeds from the models list alone when traced models are not available', async () => {
    await mount('/engine', 'The inference process, live.', { '/models': local });
    await waitFor(() => localStorage.getItem('mlx-node:model-colors:v1') !== null);
    expect(JSON.parse(localStorage.getItem('mlx-node:model-colors:v1')!)).toEqual([
      'b-plain',
      'gguf-dir/q4',
      'gguf-dir/q8',
      'zeta-7b',
    ]);
  });
});

describe('palette refresh', () => {
  it('Refresh data ingests, then every mounted query fetches again', async () => {
    const calls: string[] = [];
    const isDownloads = (path: string): boolean => path.endsWith('/downloads');
    await mount('/engine', 'The inference process, live.', { '/ingest': { ok: true } }, (call) =>
      calls.push(call.path),
    );
    await waitFor(() => calls.some(isDownloads));
    const before = calls.filter(isDownloads).length;
    await press('k');
    const option = allByRole(document, 'option').find((node) => accessibleName(node).startsWith('Refresh data'));
    expect(option).toBeDefined();
    await act(async () => option!.click());
    await waitFor(() => calls.some((path) => path.endsWith('/ingest')));
    await waitFor(() => calls.filter(isDownloads).length > before);
    const ingestAt = calls.findIndex((path) => path.endsWith('/ingest'));
    expect(calls.slice(ingestAt).some(isDownloads)).toBe(true);
  });
});

describe('page test providers', () => {
  it('renderPage puts the page inside a focusable main[data-scroll-root] that ScrollRootContext names', async () => {
    let seen: HTMLElement | null = null;
    function ScrollProbe(): ReactElement {
      const root = useScrollRoot();
      seen = root;
      return createElement('p', null, root === null ? 'no root' : 'has root');
    }
    page = await renderPage(createElement(ScrollProbe), (text) => text.includes('has root'));
    const probe = page.container.querySelector('p');
    expect(page.container.querySelectorAll('[data-scroll-root]')).toHaveLength(1);
    expect(seen).not.toBeNull();
    expect(seen!.tagName).toBe('MAIN');
    expect(seen!.hasAttribute('data-scroll-root')).toBe(true);
    expect(seen!.tabIndex).toBe(-1);
    expect(probe?.closest('[data-scroll-root]')).toBe(seen);
  });
});
