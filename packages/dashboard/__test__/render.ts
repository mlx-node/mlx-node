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
  globalThis.fetch = ((input: RequestInfo | URL): Promise<Response> => {
    const path = requestPath(input)
      .split('?')[0]
      .replace(/^\/api/, '');
    if (!Object.hasOwn(routes, path)) return respond({ error: `no stub for ${path}` }, 404);
    return respond(routes[path], 200);
  }) as typeof globalThis.fetch;
  return () => {
    globalThis.fetch = previous;
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
