/** @vitest-environment happy-dom */

import { Storage } from 'happy-dom';
import { act, createElement, type ReactElement } from 'react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import type { ChatHostState } from '../ui/src/lib/chat-types.js';
import { modelColor, resetModelColors } from '../ui/src/lib/model-colors.js';
import type {
  EngineStatus,
  MetricsModels,
  ModelsResponse,
  RuntimeLive,
  RuntimeProcessLive,
} from '../ui/src/lib/types.js';
import Engine from '../ui/src/pages/engine.js';
import { AppShell } from '../ui/src/shell/app-shell.js';
import { matchHotkey } from '../ui/src/shell/hotkeys.js';
import { Page, PageHeader } from '../ui/src/shell/page.js';
import { useScrollRoot } from '../ui/src/shell/scroll-root.js';
import { footerLine } from '../ui/src/shell/status-footer.js';
import type { ShellStatus } from '../ui/src/shell/use-shell-status.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import {
  accessibleName,
  allByRole,
  byRole,
  renderPage,
  type RenderedPage,
  deferred,
  READY_CHAT_HOST,
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

afterEach(() => {
  page?.unmount();
  page = undefined;
  api?.();
  api = undefined;
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
    decodeTps: null,
    loading: null,
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
  it('lists the nine nav items in table order and marks the current one', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    expect(navLabels(container)).toEqual([
      'Overview',
      'Chat',
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
    expect(navLabels(container)).toHaveLength(9);
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
    expect(document.title).toBe('Engine — mlx-node');
    const metrics = sections(container).querySelector<HTMLAnchorElement>('a[href="/metrics"]');
    await act(async () => metrics?.click());
    const headings = allByRole(container, 'heading').map((heading) => heading.textContent);
    expect(headings).toContain('Metrics probe');
    expect(headings).not.toContain('Engine');
    expect(document.activeElement?.tagName).toBe('MAIN');
    expect(document.activeElement?.hasAttribute('data-scroll-root')).toBe(true);
    expect(document.title).toBe('Metrics probe — mlx-node');
    expect(current(container)).toEqual(['Metrics']);
    await waitFor(
      () => container.querySelector('[aria-live="polite"][role="status"]')?.textContent === 'Metrics probe',
    );
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

  it('⌘2 goes to Chat and ⌘3 to Sessions', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    const link = sections(container).querySelector<HTMLAnchorElement>('a[href="/chat"]');
    expect((link?.textContent ?? '').trim()).toBe('Chat');
    await press('2');
    expect(location).toBe('/chat');
  });

  it('⌘3 goes to Sessions, the third nav item', async () => {
    await mount('/engine', 'The inference process, live.');
    await press('3');
    expect(location).toBe('/sessions');
  });

  it('⌘4 goes to Metrics', async () => {
    await mount('/engine', 'The inference process, live.');
    await press('4');
    expect(location).toBe('/metrics');
    expect(page!.text()).toContain('Metrics probe page');
  });

  it('⌘9 goes to Web Search, the last nav item', async () => {
    const { container } = await mount('/engine', 'The inference process, live.');
    const link = sections(container).querySelector<HTMLAnchorElement>('a[href="/providers"]');
    expect((link?.textContent ?? '').trim()).toBe('Web Search');
    await press('9');
    expect(location).toBe('/providers');
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
        { conversationId: 'a', mode: 'chat', busy: true, waiting: false },
        { conversationId: 'b', mode: 'chat', busy: true, waiting: false },
      ]),
    });
    await waitFor(() => dot() === '2 running');
    api!.emitChatHost({
      type: 'state',
      state: state([
        { conversationId: 'a', mode: 'chat', busy: true, waiting: false },
        { conversationId: 'w', mode: 'work', busy: true, waiting: true },
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
      },
    ],
    warnings: [],
    dir: '/m',
    host: { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
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
      },
    ],
    warnings: [],
    dir: '/m',
    host: { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
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
