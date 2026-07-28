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

import { act, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

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
 * Route table for the stubbed `fetch`: API path (as the SPA spells it, e.g.
 * `/cache`) → JSON body. An unlisted path resolves 404, which the pages render
 * as their error state rather than hanging forever.
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
 * Install a `globalThis.fetch` that answers from `routes`. Returns a disposer.
 * Query strings are ignored when matching, so `/metrics/overview?from=…` hits
 * the `/metrics/overview` entry.
 */
export function stubFetch(routes: ApiStub): () => void {
  const previous = globalThis.fetch;
  const requestPath = (input: RequestInfo | URL): string => {
    if (typeof input === 'string') return input;
    if (input instanceof URL) return input.pathname;
    return input.url;
  };
  // Deliberately resolved on a MACROTASK, not via Promise.resolve(). A real
  // fetch never settles on the microtask queue, and a stub that does lets a
  // harness which flushes a fixed number of microtasks appear to work — right
  // up until the machine is loaded and the same tests start failing. Forcing
  // the slower, truthful hop makes {@link renderPage}'s condition wait the
  // thing the tests actually depend on, rather than an accident of timing.
  const respond = (body: unknown, status: number): Promise<Response> =>
    new Promise((resolve) => {
      setTimeout(() => {
        resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }));
      }, 0);
    });
  const served = new Map<string, number>();
  const bodyFor = (path: string): unknown => {
    const body = routes[path];
    if (typeof body !== 'object' || body === null || !(SEQUENCE in body)) return body;
    const bodies = (body as Record<symbol, unknown>)[SEQUENCE] as unknown[];
    const nth = served.get(path) ?? 0;
    served.set(path, nth + 1);
    return bodies[Math.min(nth, bodies.length - 1)];
  };
  globalThis.fetch = ((input: RequestInfo | URL): Promise<Response> => {
    const path = requestPath(input)
      .split('?')[0]
      .replace(/^\/api/, '');
    if (!Object.hasOwn(routes, path)) return respond({ error: `no stub for ${path}` }, 404);
    return respond(bodyFor(path), 200);
  }) as typeof globalThis.fetch;
  return () => {
    globalThis.fetch = previous;
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
