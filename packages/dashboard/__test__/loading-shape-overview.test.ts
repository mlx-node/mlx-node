/**
 * @vitest-environment happy-dom
 */

/**
 * Structural guards for the Overview page's loading state.
 *
 * The page fires four independent requests and swaps a skeleton for each answer
 * as it lands. Whenever a skeleton does not measure the same as the value that
 * replaces it, the swap RESIZES the page: the tiles grew 6px apiece when their
 * request answered (`h-8` against a 30px value, `h-4` against a 20px sub-line),
 * which moved the card below them, and that card then grew by roughly 170px of
 * its own when four `h-12` bars were replaced by six real rows. Read as a user,
 * that is not a page filling in — it is a page flinching, four times, in an
 * order that changes on every load.
 *
 * happy-dom performs no layout, so `getBoundingClientRect()` answers 0 and the
 * heights themselves are unassertable here. What IS assertable is the property
 * the fix rests on: a skeleton bar is `h-[1lh]`, one line box of the font its own
 * wrapper carries, so it cannot disagree with that wrapper's text; and the
 * skeleton list is the loaded list with its text blanked, same `<ul>`, same row
 * box, same row count. Both are structural, and both fail on the revert.
 *
 * `happy-dom` is scoped to this file by the docblock above — the repo default
 * environment stays `node` for every other suite.
 */

import { MessageChannel } from 'node:worker_threads';

import type { SessionRow, SessionsResponse } from '@/lib/types';
import Overview from '@/pages/overview';
import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import { renderPage, stubApi, type RenderedPage } from './render.js';

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
 * disposer closes it long before that, which also clears the four pending timers
 * so nothing is left holding the runner open.
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

/** More sessions than the page shows, so the loaded list is capped rather than short. */
function sessionList(count: number): SessionsResponse {
  const sessions: SessionRow[] = Array.from({ length: count }, (_, i) => ({
    id: `s-${i}`,
    path: `/tmp/sessions/s-${i}.jsonl`,
    cwd: '/tmp/work',
    name: `Session ${i}`,
    created: 1_700_000_000_000 + i,
    modified: 1_700_000_000_000 + i,
    messageCount: 4,
    firstMessage: null,
    // Rows normally carry model badges, and a badge is taller than the plain
    // text beside it — so the loaded row this skeleton stands in for has to
    // carry one too, or the comparison is against a row nobody sees.
    models: ['qwen3-8b'],
    inputTokens: 100,
    outputTokens: 200,
  }));
  return { sessions, total: count, tokens: 300 * count, cwds: ['/tmp/work'] };
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

/** The single recent-sessions list on the page, loading or loaded. */
function recentList(page: RenderedPage): HTMLElement {
  const lists = [...page.container.querySelectorAll<HTMLElement>('ul')];
  // One list, in both states. More than one means something other than the
  // recent-sessions card answered this query and every shape below is measuring
  // the wrong element.
  expect(lists).toHaveLength(1);
  return lists[0];
}

/**
 * The four class strings that decide a row's box, read off the DOM by position
 * rather than by name: `<li>` → row box → text column → title line, meta line.
 * A skeleton row that took a different shape cannot answer this walk at all.
 */
function rowShape(li: Element): { box: string[]; column: string; title: string; meta: string } {
  const box = li.firstElementChild;
  const column = box?.firstElementChild;
  const title = column?.children[0];
  const meta = column?.children[1];
  if (box == null || column == null || title == null || meta == null) {
    throw new Error(`recent-session row is not a box around a title and a meta line: ${li.outerHTML}`);
  }
  return {
    box: box.className.split(' ').filter(Boolean),
    column: column.className,
    title: title.className,
    meta: meta.className,
  };
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
    text.includes('Recent sessions'),
  );
  await settle();
  // Nothing answered, so nothing rendered a value — if this fails the page has
  // left the loading state and every assertion after it is vacuous.
  expect(mounted.container.querySelectorAll('[data-slot="skeleton"]').length).toBeGreaterThan(0);
  expect(mounted.text()).not.toContain('Session 0');
  return mounted;
}

/**
 * Mount Overview against a session list that answers. Only `/sessions` is
 * stubbed: the other three tiles are irrelevant to the row shape, and leaving
 * them to 404 keeps the fixture down to the response under comparison.
 */
async function mountLoaded(count: number): Promise<RenderedPage> {
  restoreApi = stubApi({ '/sessions': sessionList(count) });
  mounted = await renderPage(createElement(MemoryRouter, null, createElement(Overview)), (text) =>
    text.includes('Session 0'),
  );
  return mounted;
}

describe('Overview loading state — the skeleton stands in the same box', () => {
  it('fills the recent-sessions list with as many skeleton rows as it will show sessions', async () => {
    const loading = recentList(await mountLoading());
    const skeletonRows = [...loading.querySelectorAll(':scope > li')];
    mounted?.unmount();
    mounted = undefined;
    restoreApi?.();
    restoreApi = undefined;

    // Ten sessions offered, six shown: the loaded list is at the cap, which is
    // the count the skeleton has to match for a full list to swap in without
    // moving the page. Four bars against six rows is the ~170px jolt.
    const loadedRows = [...recentList(await mountLoaded(10)).querySelectorAll(':scope > li')];
    expect(loadedRows).toHaveLength(6);
    expect(skeletonRows).toHaveLength(loadedRows.length);
  });

  it('renders those rows inside the same list wrapper the loaded rows use', async () => {
    const loadingList = recentList(await mountLoading());
    const loadingClass = loadingList.className;
    const skeletonShapes = [...loadingList.querySelectorAll(':scope > li')].map(rowShape);
    mounted?.unmount();
    mounted = undefined;
    restoreApi?.();
    restoreApi = undefined;

    const loadedList = recentList(await mountLoaded(10));
    // Same divided list, not a free-standing stack of bars in a padded div.
    expect(loadingClass).toBe(loadedList.className);
    expect(loadingClass).toContain('divide-y');

    const loaded = rowShape(loadedList.querySelector(':scope > li')!);
    for (const shape of skeletonShapes) {
      // The row's own padding is what makes it as tall as it is, so it is named
      // outright rather than only compared — an empty class list would satisfy
      // the subset check below on its own.
      expect(shape.box).toContain('px-6');
      expect(shape.box).toContain('py-3');
      // The loaded row adds its hover affordances on top of the shared box; every
      // class that decides the box itself must appear on both.
      for (const cls of shape.box) expect(loaded.box).toContain(cls);
      expect(shape.column).toBe(loaded.column);
      expect(shape.title).toBe(loaded.title);
      expect(shape.meta).toBe(loaded.meta);
    }
  });

  it('sizes every bar in a row from the line box of the text it stands in for', async () => {
    const loading = recentList(await mountLoading());
    for (const li of loading.querySelectorAll(':scope > li')) {
      const bars = [...li.querySelectorAll<HTMLElement>('[data-slot="skeleton"]')];
      // A title bar plus the timestamp and the model badge under it.
      expect(bars).toHaveLength(3);
      // `1lh` resolves against the wrapper's own font, so a bar cannot drift from
      // the line it replaces. A fixed `h-12` could, and did.
      for (const bar of bars) expect(bar.className).toContain('1lh');
    }
  });

  it('sizes the stat-tile bars from the tile fonts rather than a guessed pixel height', async () => {
    const loading = await mountLoading();
    // The headline value: four tiles, each bar inside the `text-3xl leading-none`
    // wrapper StatTile puts `value` in. `h-8` was 32px against a 30px line.
    const values = barsUnder(loading.container, 'text-3xl');
    expect(values).toHaveLength(4);
    for (const bar of values) expect(bar.className).toContain('h-[1lh]');

    // The sub-line: the muted `text-sm` wrapper, and only that one — the page
    // heading and the row titles are `text-sm` too but carry neither the muted
    // colour nor a bar. `h-4` was 16px against a 20px line.
    const subs = barsUnder(loading.container, 'text-sm', 'text-muted-foreground');
    expect(subs).toHaveLength(4);
    for (const bar of subs) expect(bar.className).toContain('h-[1lh]');
  });
});
