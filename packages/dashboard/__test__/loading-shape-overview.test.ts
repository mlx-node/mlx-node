/**
 * @vitest-environment happy-dom
 */

/**
 * Structural guards for the Overview page's loading state.
 *
 * The page fires independent requests and swaps a skeleton for each answer as
 * it lands. Whenever a skeleton does not measure the same as the value that
 * replaces it, the swap RESIZES the page: the tiles grew 6px apiece when their
 * request answered (`h-8` against a 30px value, `h-4` against a 20px sub-line),
 * which moved the cards below them. Read as a user, that is not a page filling
 * in — it is a page flinching, in an order that changes on every load.
 *
 * happy-dom performs no layout, so `getBoundingClientRect()` answers 0 and the
 * heights themselves are unassertable here. What IS assertable is the property
 * the fix rests on: a skeleton bar is `h-[1lh]`, one line box of the font its own
 * wrapper carries, so it cannot disagree with that wrapper's text.
 *
 * `happy-dom` is scoped to this file by the docblock above — the repo default
 * environment stays `node` for every other suite.
 */

import { MessageChannel } from 'node:worker_threads';

import Overview from '@/pages/overview';
import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import { renderPage, type RenderedPage } from './render.js';

/**
 * Serve the SPA a runtime that ACCEPTS every call and answers none of them.
 *
 * Reaching the loading state by racing the harness — asserting before the port
 * hop lands — would make these tests a function of how loaded the machine is,
 * and a page that had already settled would satisfy every "is the skeleton
 * shaped like the row" assertion vacuously by rendering no skeleton at all. A
 * reply that never comes removes the race outright: `loading` is true for as
 * long as the test cares to look, and the assertions below flush macrotasks
 * first precisely so a settled page would be caught rather than assumed away.
 *
 * The client's own 30s deadline eventually settles each call as a failure; the
 * disposer closes it long before that, which also clears the pending timers so
 * nothing is left holding the runner open.
 */
function stubPendingApi(): () => void {
  const { port1, port2 } = new MessageChannel();
  port1.unref();
  port2.unref();
  const dispose = serveRuntimeOverPort(
    { call: () => new Promise<never>(() => {}), subscribe: () => () => {} },
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

/** Let any pending port hop, and the render it would cause, land. */
async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) {
    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  }
}

/** Skeleton bars whose immediate wrapper carries every one of `classes`. */
function barsUnder(root: Element, ...classes: string[]): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>('[data-slot="skeleton"]')].filter((bar) =>
    classes.every((cls) => bar.parentElement?.classList.contains(cls) ?? false),
  );
}

let mounted: RenderedPage | undefined;
let restoreApi: (() => void) | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
});

/**
 * Mount Overview against a runtime that never answers, then flush.
 *
 * The settle condition is a static heading, so it holds on the first check and
 * the page is read at its first paint; the flush afterwards is what proves the
 * state is genuinely pending rather than merely early.
 */
async function mountLoading(): Promise<RenderedPage> {
  restoreApi = stubPendingApi();
  mounted = await renderPage(createElement(MemoryRouter, null, createElement(Overview)), (text) =>
    text.includes('Overview'),
  );
  await settle();
  // Nothing answered, so nothing rendered a value — if this fails the page has
  // left the loading state and every assertion after it is vacuous.
  expect(mounted.container.querySelectorAll('[data-slot="skeleton"]').length).toBeGreaterThan(0);
  return mounted;
}

describe('Overview loading state — the skeleton stands in the same box', () => {
  it('sizes the stat-tile bars from the tile fonts rather than a guessed pixel height', async () => {
    const loading = await mountLoading();
    // The headline value: three tiles, each bar inside the `text-3xl leading-none`
    // wrapper StatTile puts `value` in. `h-8` was 32px against a 30px line.
    const values = barsUnder(loading.container, 'text-3xl');
    expect(values).toHaveLength(3);
    for (const bar of values) expect(bar.className).toContain('h-[1lh]');

    // The sub-line: the muted `text-sm` wrapper, and only that one — the page
    // heading is `text-sm` too but carries neither the muted colour nor a bar.
    // `h-4` was 16px against a 20px line.
    const subs = barsUnder(loading.container, 'text-sm', 'text-muted-foreground');
    expect(subs).toHaveLength(3);
    for (const bar of subs) expect(bar.className).toContain('h-[1lh]');
  });
});
