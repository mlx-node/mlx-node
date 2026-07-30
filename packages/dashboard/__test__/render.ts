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

import { act, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

import { failure } from '../src/api/errors.js';
import type { DownloadEvent } from '../src/download.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import type { ApiCall } from '../src/runtime.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';

declare global {
  /** React's own flag for "an `act()`-aware environment"; see react-dom docs. */
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}

/** A mounted page plus the handles a test needs to read and unmount it. */
export interface RenderedPage {
  container: HTMLElement;
  root: Root;
  /** All rendered text with runs of whitespace collapsed, for substring asserts. */
  text(): string;
  unmount(): void;
}

/**
 * Route table for the stubbed API: path as the SPA spells it (e.g. `/cache`) →
 * JSON body. An unlisted path answers 404, which the pages render as their error
 * state rather than hanging forever.
 */
export type ApiStub = Record<string, unknown>;

/** Brand for a route whose body CHANGES between calls; see {@link sequence}. */
const SEQUENCE = Symbol('api-stub-sequence');

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
 * Serve `routes` to the SPA over a real MessagePort, and connect `ui/src/lib/api`
 * to it. Returns a disposer.
 *
 * There is no `fetch` left to stub: the SPA speaks the RPC protocol over an
 * injected port. Stubbing at the PORT rather than at the api module is what keeps
 * the real client, the real envelope and a real structured-clone hop inside the
 * test — a stub that replaced `getJson` would assert against nothing but itself.
 *
 * Query strings are ignored when matching, so `/metrics/overview?from=…` hits the
 * `/metrics/overview` entry.
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
}

export function stubApi(routes: ApiStub, options: ApiStubOptions = {}): () => void {
  const { port1, port2 } = new MessageChannel();
  // Never hold the runner open on the channel's account.
  port1.unref();
  port2.unref();
  const served = new Map<string, number>();
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
        const path = call.path.split('?')[0].replace(/^\/api/, '');
        if (!Object.hasOwn(routes, path)) return Promise.resolve(failure('E_NOT_FOUND', `no stub for ${path}`));
        return Promise.resolve({ ok: true as const, status: 200, body: bodyFor(path) });
      },
      subscribe: options.subscribe ?? (() => () => {}),
    },
    bindEventTargetPort(port2),
  );
  connectDashboardApi(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });
  return () => {
    disconnectDashboardApi();
    dispose();
    port1.close();
    port2.close();
  };
}

/**
 * Advance width of one character of a 12px axis tick, and the line height of
 * that same text. Both measured in Chrome against the SPA's font stack
 * (`ui-sans-serif, system-ui, sans-serif`): `Gemma-4-31B-IT-UD-Q4_…` renders
 * 161.5px wide and 14px tall, i.e. 7.34px per character. Rounded to 7.3.
 */
export const TICK_CHAR_PX = 7.3;
export const TICK_LINE_PX = 14;

/** Plot width a stubbed chart container reports, matching a half-width card. */
const PLOT_WIDTH_PX = 420;

/** Plot height for a chart whose card did not set an explicit pixel height. */
const DEFAULT_PLOT_HEIGHT_PX = 224;

function fakeRect(width: number, height: number): DOMRect {
  return { width, height, top: 0, left: 0, right: width, bottom: height, x: 0, y: 0, toJSON: () => ({}) } as DOMRect;
}

/**
 * Give recharts real text and container measurements under happy-dom.
 *
 * happy-dom performs no layout, so `getBoundingClientRect()` answers 0 for
 * everything. That is not a neutral default here — it silently disables the two
 * recharts behaviours a per-model bar chart depends on. recharts hides category
 * ticks it measures as overlapping (`getStringSize` → a hidden span → a rect),
 * and sizes a `width="auto"` axis from its rendered ticks (each tick's own
 * rect). At zero everything "fits" and the axis collapses, so a chart that drops
 * half its labels in a browser renders all of them in a test, and a clipped
 * label reports a left edge of 0. Both defects become invisible.
 *
 * The stub answers three questions and delegates the rest:
 *  - recharts' hidden measurement span → `chars × TICK_CHAR_PX`,
 *  - a rendered axis tick `<text>` → the same,
 *  - a `ResponsiveContainer` → {@link PLOT_WIDTH_PX} by the pixel height its
 *    parent carries inline (what `ChartCard`'s `heightPx` sets), else
 *    {@link DEFAULT_PLOT_HEIGHT_PX}. Reading the parent is deliberate: it is
 *    what makes a chart that outgrows its card observable from a test.
 */
export function stubChartMetrics(options: { tickLinePx?: number } = {}): () => void {
  // `tickLinePx` above the row height is how a test reaches the case a normal
  // render never shows: tick text taller than the space one row gives it, e.g.
  // a browser minimum-font-size that scales the label but not the plot. That is
  // where recharts' overlap rule starts hiding category names, so it is where a
  // chart that must name every row has to prove it still does.
  const lineHeight = options.tickLinePx ?? TICK_LINE_PX;
  const proto = Element.prototype as unknown as { getBoundingClientRect: () => DOMRect };
  const original = proto.getBoundingClientRect;
  proto.getBoundingClientRect = function (this: Element): DOMRect {
    const text = this.textContent ?? '';
    if (this.id === 'recharts_measurement_span') return fakeRect(text.length * TICK_CHAR_PX, lineHeight);
    if (this.classList?.contains('recharts-cartesian-axis-tick-value')) {
      return fakeRect(text.length * TICK_CHAR_PX, lineHeight);
    }
    if (this.classList?.contains('recharts-responsive-container')) {
      const declared = Number.parseFloat((this.parentElement as HTMLElement | null)?.style.height ?? '');
      return fakeRect(PLOT_WIDTH_PX, Number.isFinite(declared) && declared > 0 ? declared : DEFAULT_PLOT_HEIGHT_PX);
    }
    return original.call(this) as DOMRect;
  };
  return () => {
    proto.getBoundingClientRect = original;
  };
}

/** How long {@link renderPage} waits for `until` before failing the test. */
const SETTLE_TIMEOUT_MS = 2_000;

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
export async function renderPage(element: ReactElement, until: (text: string) => boolean): Promise<RenderedPage> {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  const readText = (): string => (container.textContent ?? '').replace(/\s+/g, ' ').trim();
  await act(async () => {
    root.render(element);
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
