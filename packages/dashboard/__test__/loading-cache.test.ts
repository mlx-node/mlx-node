/**
 * @vitest-environment happy-dom
 */

/**
 * Guards for the stale-while-revalidate half of the loading-flicker fix.
 *
 * The user-visible defect was that every tab switch repainted a fully-loaded
 * page as skeletons: react-router unmounts the old route, so each visit started
 * from `data === undefined`. `useJson` now seeds from a module-level cache, and
 * the property that matters is a timing one — the data must be there on the
 * FIRST commit, before any response could possibly have arrived. A test that
 * flushes until the page settles cannot tell that apart from a normal fetch, so
 * these mount with a single synchronous `act()` and read the DOM immediately.
 *
 * `happy-dom` is scoped to this file by the docblock above; the repo default
 * environment stays `node`.
 */

import { mutate } from '@/lib/api';
import { useRuntimeLive } from '@/lib/live';
import { splitWhitespace } from '@/lib/text-scan';
import type { LiveEvent, RuntimeLive } from '@/lib/types';
import { useJson, type JsonOptions } from '@/lib/use-api';
import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeAll, describe, expect, it } from 'vite-plus/test';

import { renderPage, sequence, stubApi, waitFor, type RenderedPage, type StubbedApi } from './render.js';

interface Counter {
  n: number;
}

/** Renders the hook's three observable states as text, for substring asserts. */
function Probe({ path }: { path: string }): ReactElement {
  const state = useJson<Counter>(path);
  const phase = state.loading ? 'LOADING' : state.refreshing ? 'REFRESHING' : 'IDLE';
  return createElement('div', null, `${phase} n=${state.data?.n ?? '-'} ${state.error?.message ?? ''}`);
}

interface Mounted {
  text: () => string;
  /** Re-render on the SAME root, so React keeps the component instance and its state. */
  rerender: (next: ReactElement) => void;
  unmount: () => void;
}

/**
 * Mount and flush exactly one synchronous `act()` — render plus effects, and
 * nothing else.
 *
 * This is the whole point of the file. {@link renderPage} loops on macrotasks
 * until a condition holds, which is right for asserting on loaded content and
 * useless here: the RPC reply arrives on a macrotask, so anything this mount
 * shows was in hand BEFORE the runtime was asked. That is exactly the claim a
 * cache hit makes and a cache miss cannot.
 */
function mountOnce(element: ReactElement): Mounted {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(element);
  });
  return {
    text: () => splitWhitespace(container.textContent ?? '').join(' '),
    rerender: (next: ReactElement) => {
      act(() => {
        root.render(next);
      });
    },
    unmount: () => {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
  };
}

describe('useJson stale-while-revalidate', () => {
  let dispose: (() => void) | undefined;
  let mounted: Mounted | undefined;

  afterEach(() => {
    mounted?.unmount();
    mounted = undefined;
    // Disconnecting clears the cache, so every test starts cold whether or not
    // it remembered to.
    dispose?.();
    dispose = undefined;
  });

  it('shows a loading state on the first ever visit to a path', () => {
    dispose = stubApi({ '/counter': { n: 1 } });

    mounted = mountOnce(createElement(Probe, { path: '/counter' }));

    expect(mounted.text()).toContain('LOADING');
  });

  it('renders a previously fetched body on the first commit of a later visit', async () => {
    dispose = stubApi({ '/counter': { n: 7 } });

    const first = await renderPage(createElement(Probe, { path: '/counter' }), (text) => text.includes('n=7'));
    first.unmount();

    // Same connection, so the cache survives — this is a tab switch, not a
    // reconnect.
    mounted = mountOnce(createElement(Probe, { path: '/counter' }));

    expect(mounted.text()).toContain('n=7');
    expect(mounted.text()).not.toContain('LOADING');
  });

  it('reports a cache-seeded visit as refreshing, so the page keeps its layout', async () => {
    dispose = stubApi({ '/counter': { n: 7 } });

    const first = await renderPage(createElement(Probe, { path: '/counter' }), (text) => text.includes('n=7'));
    first.unmount();

    mounted = mountOnce(createElement(Probe, { path: '/counter' }));

    expect(mounted.text()).toContain('REFRESHING');
  });

  it('revalidates a seeded visit and replaces the stale body', async () => {
    dispose = stubApi({ '/counter': sequence({ n: 1 }, { n: 2 }) });

    const first = await renderPage(createElement(Probe, { path: '/counter' }), (text) => text.includes('n=1'));
    first.unmount();

    // Seeded with the stale 1, then corrected to 2 without ever showing
    // LOADING in between.
    const second = await renderPage(createElement(Probe, { path: '/counter' }), (text) => text.includes('n=2'));
    expect(second.text()).toContain('IDLE');
    second.unmount();
  });

  it('does not seed one path from another when the path changes', async () => {
    dispose = stubApi({ '/counter': { n: 1 }, '/other': { n: 9 } });

    const page = await renderPage(createElement(Probe, { path: '/counter' }), (text) => text.includes('n=1'));
    page.unmount();

    // `/other` has never been fetched, so the hook must not carry `/counter`'s
    // body across — a filter change that showed the previous query's rows would
    // be worse than a skeleton, not better.
    mounted = mountOnce(createElement(Probe, { path: '/other' }));

    expect(mounted.text()).toContain('LOADING');
    expect(mounted.text()).not.toContain('n=1');
  });

  it('never paints the previous path over a live component when the path changes', async () => {
    dispose = stubApi({ '/counter': { n: 1 }, '/other': { n: 9 } });

    const page = await renderPage(createElement(Probe, { path: '/counter' }), (text) => text.includes('n=1'));
    page.unmount();

    // The mounted-component case, which unmount-and-remount above cannot reach:
    // a live `useJson` handed a NEW path. This is what the metrics range
    // selector does on every change, and it is the one
    // route by which state belonging to the old path can survive into a render
    // of the new one. `useState`'s initializer does not re-run for an existing
    // instance, so only the render-phase reset stands between the user and a
    // frame of the previous query's results.
    mounted = mountOnce(createElement(Probe, { path: '/counter' }));
    mounted.rerender(createElement(Probe, { path: '/other' }));

    expect(mounted.text()).not.toContain('n=1');
    expect(mounted.text()).toContain('LOADING');
  });

  it('drops cached bodies after a mutation', async () => {
    dispose = stubApi({ '/counter': { n: 1 }, '/thing': { ok: true } });

    const first = await renderPage(createElement(Probe, { path: '/counter' }), (text) => text.includes('n=1'));
    first.unmount();

    await mutate('DELETE', '/thing');

    // A write can change a resource it did not address — deleting a model moves
    // the Overview counts — so the next visit must ask again rather than paint
    // what it remembered.
    mounted = mountOnce(createElement(Probe, { path: '/counter' }));

    expect(mounted.text()).toContain('LOADING');
  });
});

function LiveProbe(): ReactElement {
  const live = useRuntimeLive();
  return createElement('div', null, `status=${live?.status ?? '-'}`);
}

/** A counting query: the text is how many answers it has seen. */
function CountingProbe({ live }: { live: JsonOptions['live'] }): ReactElement {
  const state = useJson<Counter>('/counter', { live });
  return createElement('div', null, `n=${state.data?.n ?? '-'}`);
}

function runtimeEvent(status: RuntimeLive['status']): LiveEvent {
  return { type: 'runtime', snapshot: { now: 1, status, supervisor: null, processes: [] } };
}

const DATA_EVENT: LiveEvent = { type: 'data', version: 1, traces: true, runtime: false };

function sleep(ms: number): Promise<void> {
  return act(async () => {
    await new Promise((resolve) => setTimeout(resolve, ms));
  });
}

describe('live channel', () => {
  let api: StubbedApi | null = null;
  let page: RenderedPage | null = null;
  let visibility: DocumentVisibilityState = 'visible';

  beforeAll(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => visibility });
  });

  afterEach(() => {
    page?.unmount();
    page = null;
    api?.();
    api = null;
    visibility = 'visible';
  });

  it('shows the pushed engine snapshot, only while the window is visible, and drops it on a reconnect', async () => {
    api = stubApi({});
    page = await renderPage(createElement(LiveProbe), (text) => text === 'status=-');
    await waitFor(() => {
      api?.emitLive(runtimeEvent('running'));
      return page?.text() === 'status=running';
    });

    visibility = 'hidden';
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    api.emitLive(runtimeEvent('stopped'));
    await sleep(20);
    expect(page.text()).toBe('status=running');

    visibility = 'visible';
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await waitFor(() => {
      api?.emitLive(runtimeEvent('loading'));
      return page?.text() === 'status=loading';
    });

    act(() => {
      api?.();
      api = stubApi({});
    });
    await waitFor(() => page?.text() === 'status=-');
    await waitFor(() => {
      api?.emitLive(runtimeEvent('failed'));
      return page?.text() === 'status=failed';
    });
  });

  it('refetches a live query on a data event at most once per interval, with one trailing refetch', async () => {
    let served = 0;
    api = stubApi(
      { '/counter': { n: 0 } },
      {
        onCall: () => {
          served += 1;
        },
      },
    );
    page = await renderPage(createElement(CountingProbe, { live: { minIntervalMs: 150 } }), (text) => text === 'n=0');
    expect(served).toBe(1);

    await sleep(180);
    api.emitLive(DATA_EVENT);
    await waitFor(() => served === 2);

    api.emitLive(DATA_EVENT);
    api.emitLive(DATA_EVENT);
    await sleep(60);
    expect(served).toBe(2);
    await waitFor(() => served === 3, 600);
    await sleep(250);
    expect(served).toBe(3);
  });

  it('refetches a live query once when the window shows again after it was hidden', async () => {
    let served = 0;
    api = stubApi(
      { '/counter': { n: 0 } },
      {
        onCall: () => {
          served += 1;
        },
      },
    );
    page = await renderPage(createElement(CountingProbe, { live: { minIntervalMs: 50 } }), (text) => text === 'n=0');
    expect(served).toBe(1);

    visibility = 'hidden';
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    api.emitLive(DATA_EVENT);
    await sleep(80);
    expect(served).toBe(1);

    visibility = 'visible';
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await waitFor(() => served === 2, 600);
    await sleep(120);
    expect(served).toBe(2);
  });

  it('refetches a live query once on show when it first mounted while the window was hidden', async () => {
    let served = 0;
    visibility = 'hidden';
    api = stubApi(
      { '/counter': sequence({ n: 0 }, { n: 1 }) },
      {
        onCall: () => {
          served += 1;
        },
      },
    );
    page = await renderPage(createElement(CountingProbe, { live: { minIntervalMs: 50 } }), (text) => text === 'n=0');
    expect(served).toBe(1);
    await sleep(80);

    visibility = 'visible';
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await waitFor(() => page?.text() === 'n=1', 600);
    await sleep(120);
    expect(served).toBe(2);
  });

  it('ignores data events in a query that is not live', async () => {
    let served = 0;
    api = stubApi(
      { '/counter': { n: 0 } },
      {
        onCall: () => {
          served += 1;
        },
      },
    );
    page = await renderPage(
      createElement('div', null, createElement(CountingProbe, { live: false }), createElement(LiveProbe)),
      (text) => text.includes('n=0'),
    );
    await waitFor(() => {
      api?.emitLive(runtimeEvent('running'));
      return page?.text().includes('status=running') === true;
    });
    api.emitLive(DATA_EVENT);
    await sleep(60);
    expect(served).toBe(1);
  });
});
