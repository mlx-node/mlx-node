/**
 * Minimal React render harness for the dashboard SPA pages.
 *
 * The pages are the only place several of the Cache/Overview fixes actually
 * live — `coldObjectCounts`, `formatRate`, `percentInt` and the cold-tier
 * health tiles are wired up in JSX, not in a helper — so a suite that only
 * imports `ui/src/lib/*` cannot fail when a page is reverted. This module makes
 * a page reachable by an executing test.
 *
 * Deliberately dependency-free beyond React itself: no `@testing-library/*`.
 * `act()` ships in React 19, and `createRoot` + a container is all a page
 * needs. Files that use this MUST carry the `@vitest-environment happy-dom`
 * docblock — the repo-wide default environment is `node` and stays that way.
 */

import { MessageChannel } from 'node:worker_threads';

import { MotionGlobalConfig } from 'motion/react';
import { act, createElement, type ReactElement, type ReactNode, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';

import { type ApiErrorCode, failure } from '../src/api/errors.js';
import type { ChatHostEvent, ChatHostState, ConversationEvent } from '../src/chat/types.js';
import type { DownloadEvent } from '../src/download.js';
import type { LiveEvent as RuntimeLiveEvent } from '../src/live.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import type { ChatScope } from '../src/rpc/protocol.js';
import type { ApiCall } from '../src/runtime.js';
import { ToastStackProvider } from '../ui/src/components/arc/toast-stack/toast-stack.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import { splitWhitespace } from '../ui/src/lib/text-scan.js';
import { ToastBridge } from '../ui/src/lib/toast.js';
import type { LiveEvent } from '../ui/src/lib/types.js';
import { ScrollRootContext } from '../ui/src/shell/scroll-root.js';

declare global {
  /** React's own flag for "an `act()`-aware environment"; see react-dom docs. */
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}

/**
 * Set `MotionGlobalConfig.skipAnimations`, which `useStaticMotion()` reads: every own animation renders its end
 * state and Arc's motion finishes at once. Returns the previous value so a case that turns it off can restore it
 * in `finally`. On (true) from the moment this module loads.
 */
export function skipAnimations(on: boolean): boolean {
  const previous = MotionGlobalConfig.skipAnimations === true;
  MotionGlobalConfig.skipAnimations = on;
  return previous;
}

skipAnimations(true);

/** A mounted page plus the handles a test needs to read and unmount it. */
export interface RenderedPage {
  container: HTMLElement;
  root: Root;
  /** All rendered text with runs of whitespace collapsed, for substring asserts. */
  text(): string;
  unmount(): void;
}

/**
 * Route table for the stubbed API: path as the SPA spells it (e.g. `/cache`), or
 * `METHOD path` (e.g. `POST /chat/conversations`), which wins over the bare path →
 * JSON body. An unlisted path answers 404, which the pages render as their error
 * state rather than hanging forever.
 */
export type ApiStub = Record<string, unknown>;

/** Brand for a route whose body CHANGES between calls; see {@link sequence}. */
const SEQUENCE = Symbol('api-stub-sequence');
const DEFERRED = Symbol('api-stub-deferred');
/**
 * Sequence element that makes the stub answer a FAILURE for that call.
 *
 * `useJson` keeps its previous `data` when a reload fails and only sets
 * `error`, so "first call succeeds, reload fails" is the ONLY way to reach the
 * data-and-error state a page must branch on. Without this the stub could
 * model an always-failing route but never a route that goes bad.
 */
export const STUB_FAILURE = Symbol('api-stub-failure');
const STUB_ERROR = Symbol('api-stub-error');

/** A route body that answers this exact failure envelope (a Chat route's typed failure carries a `reason`). */
export function stubError(code: ApiErrorCode, message: string, reason?: string): unknown {
  return { [STUB_ERROR]: { code, message, reason } };
}

/** What a Chat host stream answers first when a test sets nothing: an engine and an agent that are ready. */
export const READY_CHAT_HOST: ChatHostState = {
  inference: 'ready',
  loading: null,
  resident: [],
  answers: [],
  models: [{ id: 'qwen3.5-4b', contextTokens: 32768 }],
  defaultModelId: 'qwen3.5-4b',
  agent: {
    binary: 'ok',
    source: 'bundled',
    tag: 'mlx-agent-v0.3.3',
    minTag: 'mlx-agent-v0.3.3',
    versionVerified: true,
    install: 'idle',
  },
  modes: {
    chat: { state: 'ready', stale: false, failure: null, agentVersion: '1.0.45', turnsInFlight: 0 },
    work: { state: 'stopped', stale: false, failure: null, agentVersion: null, turnsInFlight: 0 },
  },
  providers: { search: null, searchCheck: null, firecrawl: false, fetch: null, applyPending: false },
  active: [],
  permissionRaised: [],
};

export type ChatSync = Extract<ConversationEvent, { type: 'sync' }>;

/**
 * A route body that differs per request: `bodies[n]` answers the n-th call and
 * the last entry answers every call after it.
 *
 * This is how a test reaches a REFETCH. A single static body cannot distinguish
 * "the page reloaded and got fresh state" from "the page never reloaded", so any
 * assertion about repairing a stale snapshot is unfalsifiable without it.
 *
 * Branded rather than inferred from a bare array on purpose: a route body may
 * legitimately BE an array, and guessing would make one indistinguishable from a
 * two-call sequence.
 */
export function sequence(...bodies: unknown[]): unknown {
  return { [SEQUENCE]: bodies };
}

/**
 * A route body whose REPLY is withheld until the returned `release` is called.
 *
 * The only way to assert on a render that happens WHILE a reload is in flight.
 * `sequence` controls what a refetch answers but never when, so a page bug that
 * lives in the gap between "reload issued" and "body arrived" — where the hook
 * still serves the previous body — is invisible to any assertion taken after
 * the port has drained. Composes with `sequence`: hold only the second call.
 * Pass `STUB_FAILURE` to delay an API failure instead of a successful body.
 */
export function deferred(body: unknown): { body: unknown; release: () => void } {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { body: { [DEFERRED]: { body, gate } }, release };
}

/**
 * Serve `routes` to the SPA over a real MessagePort, and connect `ui/src/lib/api`
 * to it. Returns a disposer.
 *
 * There is no `fetch` left to stub: the SPA speaks the RPC protocol over an
 * injected port. Stubbing at the PORT rather than at the api module is what keeps
 * the real client, the real envelope and a real structured-clone hop inside the
 * test — a stub that replaced `getJson` would assert against nothing but itself.
 *
 * Query strings are ignored when matching, so `/metrics/summary?from=…` hits the
 * `/metrics/summary` entry.
 *
 * A real port also supplies for free what the old `fetch` stub had to fake: a
 * reply arrives on a MACROTASK, never the microtask queue. A harness that flushed
 * a fixed number of microtasks would appear to work right up until the machine is
 * loaded — {@link renderPage} waits on a caller-named condition instead, and the
 * port is what makes that wait test the real hop rather than an accident of
 * timing.
 */
export interface ApiStubOptions {
  /**
   * Every call the SPA made, in order. Some behaviour has no rendered
   * consequence at all — dismissing a settled job is a `DELETE` and nothing else
   * — so a test has to be able to read the calls, not only the DOM.
   */
  onCall?: (call: ApiCall) => void;
  /**
   * Stand in for the runtime's download subscribe, returning the unsubscribe
   * exactly as the real one does. Unlike the default no-op it can DELIVER
   * events, which is the only way to reach what a page does when a job settles
   * from its subscription rather than from a request the page made itself.
   */
  subscribe?: (jobId: string, listener: (event: DownloadEvent) => void) => () => void;
  /**
   * What a Chat stream answers first: the host state (default {@link READY_CHAT_HOST}) and the `sync` of each known
   * conversation. A conversation stream for any other id answers one `gone` event.
   */
  chat?: { host?: ChatHostState; conversations?: Record<string, ChatSync> };
}

/** The disposer {@link stubApi} returns, plus a way to push live-channel events to the SPA. */
export interface StubbedApi {
  (): void;
  /** Deliver one event to every open live watch (the footer, the Engine page, live queries). */
  emitLive(event: LiveEvent): void;
  /** Deliver one event to every open stream of that conversation. */
  emitChat(conversationId: string, event: ConversationEvent): void;
  /** Deliver one event to every open Chat host stream. */
  emitChatHost(event: ChatHostEvent): void;
  /** Open Chat streams: of one conversation, or (no id) of the host. */
  chatWatchers(conversationId?: string): number;
  /** Close the runtime's end of the port, as a crashed control panel service does. Every stream ends with it. */
  dropPort(): void;
}

export function stubApi(routes: ApiStub, options: ApiStubOptions = {}): StubbedApi {
  const { port1, port2 } = new MessageChannel();
  // Never hold the runner open on the channel's account.
  port1.unref();
  port2.unref();
  const served = new Map<string, number>();
  const liveListeners = new Set<(event: RuntimeLiveEvent) => void>();
  const hostWatchers = new Set<(event: ChatHostEvent) => void>();
  const conversationWatchers = new Map<string, Set<(event: ConversationEvent) => void>>();
  const watchChat = (scope: ChatScope, send: (event: ChatHostEvent | ConversationEvent) => void): (() => void) => {
    if (scope.scope === 'host') {
      hostWatchers.add(send);
      send({ type: 'state', state: options.chat?.host ?? READY_CHAT_HOST });
      return () => {
        hostWatchers.delete(send);
      };
    }
    const id = scope.conversationId;
    const sync = options.chat?.conversations?.[id];
    let set = conversationWatchers.get(id);
    if (set === undefined) {
      set = new Set();
      conversationWatchers.set(id, set);
    }
    const watchers = set;
    watchers.add(send);
    send(sync ?? { type: 'gone', seq: 1, reason: 'unknown' });
    return () => {
      watchers.delete(send);
    };
  };
  const bodyFor = (path: string): unknown => {
    const body = routes[path];
    if (typeof body !== 'object' || body === null || !(SEQUENCE in body)) return body;
    const bodies = (body as Record<symbol, unknown>)[SEQUENCE] as unknown[];
    const nth = served.get(path) ?? 0;
    served.set(path, nth + 1);
    return bodies[Math.min(nth, bodies.length - 1)];
  };
  const dispose = serveRuntimeOverPort(
    {
      call: (call: ApiCall) => {
        options.onCall?.(call);
        const bare = call.path.split('?')[0];
        const bare0 = bare.startsWith('/api') ? bare.slice('/api'.length) : bare;
        const keyed = `${call.method.toUpperCase()} ${bare0}`;
        const path = Object.hasOwn(routes, keyed) ? keyed : bare0;
        if (!Object.hasOwn(routes, path)) return Promise.resolve(failure('E_NOT_FOUND', `no stub for ${path}`));
        const body = bodyFor(path);
        if (body === STUB_FAILURE) return Promise.resolve(failure('E_UNAVAILABLE', `stubbed failure for ${path}`));
        if (typeof body === 'object' && body !== null && STUB_ERROR in body) {
          const e = (body as Record<symbol, { code: ApiErrorCode; message: string; reason?: string }>)[STUB_ERROR];
          return Promise.resolve(failure(e.code, e.message, e.reason));
        }
        if (typeof body === 'object' && body !== null && DEFERRED in body) {
          const held = (body as Record<symbol, unknown>)[DEFERRED] as { body: unknown; gate: Promise<void> };
          return held.gate.then(() =>
            held.body === STUB_FAILURE
              ? failure('E_UNAVAILABLE', `stubbed failure for ${path}`)
              : { ok: true as const, status: 200, body: held.body },
          );
        }
        return Promise.resolve({ ok: true as const, status: 200, body });
      },
      subscribe: options.subscribe ?? (() => () => {}),
      watchLive: (listener: (event: RuntimeLiveEvent) => void) => {
        liveListeners.add(listener);
        return () => {
          liveListeners.delete(listener);
        };
      },
      watchChat,
    },
    bindEventTargetPort(port2),
  );
  connectDashboardApi(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });
  const stop = (): void => {
    liveListeners.clear();
    hostWatchers.clear();
    conversationWatchers.clear();
    disconnectDashboardApi();
    dispose();
    port1.close();
    port2.close();
  };
  return Object.assign(stop, {
    emitLive(event: LiveEvent): void {
      // The SPA's copy of the type omits writer-only fields the SPA never reads.
      for (const listener of liveListeners) listener(event as RuntimeLiveEvent);
    },
    emitChat(conversationId: string, event: ConversationEvent): void {
      for (const send of Array.from(conversationWatchers.get(conversationId) ?? [])) send(event);
    },
    emitChatHost(event: ChatHostEvent): void {
      for (const send of Array.from(hostWatchers)) send(event);
    },
    chatWatchers(conversationId?: string): number {
      return conversationId === undefined ? hostWatchers.size : (conversationWatchers.get(conversationId)?.size ?? 0);
    },
    dropPort(): void {
      port2.close();
    },
  });
}

/** Every ResizeObserver reports a content box `width` px wide (happy-dom has no layout). Returns the restore. */
export function observeWidth(width: number): () => void {
  const original = globalThis.ResizeObserver;
  class Fixed {
    constructor(private readonly callback: ResizeObserverCallback) {}
    observe(target: Element): void {
      const entry = { target, contentRect: { width, height: 400 } } as unknown as ResizeObserverEntry;
      queueMicrotask(() => this.callback([entry], this as unknown as ResizeObserver));
    }
    unobserve(): void {}
    disconnect(): void {}
  }
  globalThis.ResizeObserver = Fixed as unknown as typeof ResizeObserver;
  return () => {
    globalThis.ResizeObserver = original;
  };
}

/** How long {@link renderPage} waits for `until` before failing the test. */
const SETTLE_TIMEOUT_MS = 2_000;

export interface ShellProviderOptions {
  /** Default true. False when the element renders the shell itself, which owns the one scroll root. */
  scrollRoot?: boolean;
}

/**
 * Mount `element` and flush until `until(text)` holds.
 *
 * `until` is REQUIRED, and that is the whole design. A fixed number of flush
 * turns is load-dependent: `useJson` chains fetch → `res.json()` → setState, and
 * `Response.json()` does not promise to resolve on a microtask, so under load
 * the page can still be in its loading state when the assertions run. That is
 * not a slow test, it is a test that silently asserts against a skeleton — and
 * a page whose assertions are all negative (`not.toContain(...)`) then passes
 * for the worst possible reason: nothing rendered at all.
 *
 * Waiting on a caller-named condition removes both failure modes. Every caller
 * must state a POSITIVE string that proves the page reached its loaded state,
 * and a page that never gets there throws with the text it did render, so the
 * failure is loud and diagnosable instead of a green run.
 */
export async function renderPage(
  element: ReactElement,
  until: (text: string) => boolean,
  options: ShellProviderOptions = {},
): Promise<RenderedPage> {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  const readText = (): string => collapse(container.textContent ?? '');
  await act(async () => {
    root.render(withShellProviders(element, options));
  });
  const deadline = Date.now() + SETTLE_TIMEOUT_MS;
  while (!until(readText())) {
    if (Date.now() > deadline) {
      // Read BEFORE unmounting — tearing the container down first makes every
      // timeout report "(empty)" and throws away the one diagnostic that says
      // whether the page rendered the wrong thing or never rendered at all.
      const rendered = readText() || '(empty)';
      root.unmount();
      container.remove();
      throw new Error(
        `renderPage: page did not settle within ${SETTLE_TIMEOUT_MS}ms.\nLast rendered text: ${rendered}`,
      );
    }
    // A macrotask turn, not `Promise.resolve()`: this drains the microtask
    // queue AND lets anything that landed on the task queue (a `json()` body
    // read) run, which is exactly the hop a fixed microtask count misses.
    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  }
  return {
    container,
    root,
    text: readText,
    unmount: () => {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
  };
}

/** Whitespace runs collapsed to one space, trimmed. */
function collapse(text: string): string {
  return splitWhitespace(text).join(' ');
}

function ScrollRootFrame({ children }: { children: ReactNode }): ReactElement {
  const [main, setMain] = useState<HTMLElement | null>(null);
  return createElement(
    ScrollRootContext.Provider,
    { value: main },
    createElement('main', { ref: setMain, 'data-scroll-root': '', tabIndex: -1 }, children),
  );
}

/**
 * The providers every page expects from the shell: the toast stack and its module bridge, and a
 * `<main data-scroll-root>` around the page given to it as ScrollRootContext.
 */
export function withShellProviders(element: ReactElement, options: ShellProviderOptions = {}): ReactElement {
  const page = options.scrollRoot === false ? element : createElement(ScrollRootFrame, null, element);
  return createElement(ToastStackProvider, null, createElement(ToastBridge), page);
}

/**
 * Wait until `predicate()` holds, flushing React between checks. Every "it is gone" assertion uses it: Arc
 * removes a leaving layer a tick after its exit, even with animations skipped.
 */
export async function waitFor(predicate: () => boolean, timeoutMs = 1000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error(`waitFor: condition not met within ${timeoutMs}ms`);
    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  }
}

type Scope = ParentNode;

const NAME_FROM_CONTENT = new Set([
  'button',
  'link',
  'heading',
  'tab',
  'option',
  'menuitem',
  'menuitemradio',
  'menuitemcheckbox',
  'cell',
  'columnheader',
  'rowheader',
  'row',
  'checkbox',
  'radio',
  'switch',
  'tooltip',
  'treeitem',
]);

const INPUT_ROLES: Record<string, string> = {
  checkbox: 'checkbox',
  radio: 'radio',
  range: 'slider',
  button: 'button',
  submit: 'button',
  reset: 'button',
  image: 'button',
  search: 'searchbox',
  number: 'spinbutton',
};

function implicitRole(element: Element): string | null {
  const tag = element.tagName.toLowerCase();
  switch (tag) {
    case 'button':
      return 'button';
    case 'a':
    case 'area':
      return element.hasAttribute('href') ? 'link' : null;
    case 'h1':
    case 'h2':
    case 'h3':
    case 'h4':
    case 'h5':
    case 'h6':
      return 'heading';
    case 'nav':
      return 'navigation';
    case 'main':
      return 'main';
    case 'aside':
      return 'complementary';
    case 'section':
      return element.hasAttribute('aria-label') || element.hasAttribute('aria-labelledby') ? 'region' : null;
    case 'form':
      return 'form';
    case 'table':
      return 'table';
    case 'thead':
    case 'tbody':
    case 'tfoot':
      return 'rowgroup';
    case 'tr':
      return 'row';
    case 'th':
      return element.getAttribute('scope') === 'row' ? 'rowheader' : 'columnheader';
    case 'td':
      return 'cell';
    case 'ul':
    case 'ol':
      return 'list';
    case 'li':
      return 'listitem';
    case 'figure':
      return 'figure';
    case 'fieldset':
      return 'group';
    case 'dialog':
      return 'dialog';
    case 'img':
      return element.getAttribute('alt') === '' ? 'presentation' : 'img';
    case 'textarea':
      return 'textbox';
    case 'select':
      return element.hasAttribute('multiple') ? 'listbox' : 'combobox';
    case 'progress':
      return 'progressbar';
    case 'hr':
      return 'separator';
    case 'output':
      return 'status';
    case 'input': {
      const type = (element.getAttribute('type') ?? 'text').toLowerCase();
      return INPUT_ROLES[type] ?? (type === 'hidden' ? null : 'textbox');
    }
    default:
      return null;
  }
}

function roleOf(element: Element): string | null {
  const explicit = element.getAttribute('role');
  if (explicit !== null) {
    const first = splitWhitespace(explicit)[0];
    if (first !== undefined) return first;
  }
  return implicitRole(element);
}

function isHidden(element: Element): boolean {
  for (let node: Element | null = element; node !== null; node = node.parentElement) {
    if (node.getAttribute('aria-hidden') === 'true' || node.hasAttribute('hidden')) return true;
  }
  return false;
}

function textOfIds(element: Element, ids: string): string {
  const doc = element.ownerDocument;
  return splitWhitespace(ids)
    .map((id) => collapse(doc.getElementById(id)?.textContent ?? ''))
    .filter((part) => part.length > 0)
    .join(' ');
}

function labelText(element: Element): string {
  const id = element.getAttribute('id');
  if (id !== null && id.length > 0) {
    for (const label of element.ownerDocument.querySelectorAll('label')) {
      if (label.htmlFor === id) return collapse(label.textContent ?? '');
    }
  }
  const wrapping = element.closest('label');
  return wrapping === null ? '' : collapse(wrapping.textContent ?? '');
}

/** The accessible name, by the rules the app's markup uses (aria-labelledby, aria-label, label, caption, content, title). */
export function accessibleName(element: Element): string {
  const labelledBy = element.getAttribute('aria-labelledby');
  if (labelledBy !== null) {
    const text = textOfIds(element, labelledBy);
    if (text.length > 0) return text;
  }
  const ariaLabel = element.getAttribute('aria-label');
  if (ariaLabel !== null && collapse(ariaLabel).length > 0) return collapse(ariaLabel);
  const tag = element.tagName.toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') {
    const text = labelText(element);
    if (text.length > 0) return text;
  }
  if (tag === 'img') return collapse(element.getAttribute('alt') ?? '');
  const captionTag =
    tag === 'fieldset' ? 'legend' : tag === 'figure' ? 'figcaption' : tag === 'table' ? 'caption' : null;
  if (captionTag !== null) {
    const caption = element.querySelector(captionTag);
    if (caption !== null) return collapse(caption.textContent ?? '');
  }
  const role = roleOf(element);
  if (role !== null && NAME_FROM_CONTENT.has(role)) {
    const text = collapse(element.textContent ?? '');
    if (text.length > 0) return text;
  }
  return collapse(element.getAttribute('title') ?? '');
}

/** Every element with `role` (explicit or implicit) and, when given, exactly that accessible name. Hidden subtrees are skipped. */
export function allByRole(container: Scope, role: string, name?: string): HTMLElement[] {
  const found: HTMLElement[] = [];
  for (const element of container.querySelectorAll<HTMLElement>('*')) {
    if (roleOf(element) !== role || isHidden(element)) continue;
    if (name !== undefined && accessibleName(element) !== name) continue;
    found.push(element);
  }
  return found;
}

/** The first element {@link allByRole} finds, or null. */
export function byRole(container: Scope, role: string, name?: string): HTMLElement | null {
  return allByRole(container, role, name)[0] ?? null;
}

/**
 * The control a `<label>` with exactly this text names (Arc `Select` and `Input` name their control with
 * `<label for>` only); else the element whose `aria-label` is `text`.
 */
export function byLabel(container: Scope, text: string): HTMLElement | null {
  for (const label of container.querySelectorAll('label')) {
    if (collapse(label.textContent ?? '') !== text) continue;
    if (label.htmlFor.length > 0) return label.ownerDocument.getElementById(label.htmlFor);
    const control = label.querySelector<HTMLElement>('input, select, textarea, button');
    if (control !== null) return control;
  }
  for (const element of container.querySelectorAll<HTMLElement>('[aria-label]')) {
    if (collapse(element.getAttribute('aria-label') ?? '') === text) return element;
  }
  return null;
}

/** The body rows of the hidden data table inside the figure named `label`, as cell texts. */
export function chartTable(container: Scope, label: string): string[][] {
  const figure = byRole(container, 'figure', label);
  if (figure === null) throw new Error(`chartTable: no figure named "${label}"`);
  const table = figure.querySelector('table');
  if (table === null) throw new Error(`chartTable: the figure "${label}" holds no table`);
  const body = table.querySelector('tbody');
  const rows = [...(body ?? table).querySelectorAll('tr')].filter((row) => body !== null || row.querySelector('td'));
  return rows.map((row) => [...row.querySelectorAll('th, td')].map((cell) => collapse(cell.textContent ?? '')));
}

/** The value and unit of the tile or ticker item named `label`: the text of its `[data-stat-value]` node. */
export function statValue(container: Scope, label: string): string {
  const group = byRole(container, 'group', label);
  if (group === null) throw new Error(`statValue: no group named "${label}"`);
  const value = group.querySelector('[data-stat-value]');
  if (value === null) throw new Error(`statValue: the group "${label}" holds no [data-stat-value]`);
  return collapse(value.textContent ?? '');
}
