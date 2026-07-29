/**
 * @vitest-environment happy-dom
 */

/**
 * What a mounted page does when ADMIN dies and comes back.
 *
 * `broker.ts` goes to real trouble here: when the ADMIN utilityProcess exits it
 * respawns it and hands the SAME live renderer a replacement port, precisely so
 * the user does not have to think to reload. The renderer then dropped it on the
 * floor. `root.render` on an existing root matches on element type and key, so
 * the tree UPDATES rather than remounts — fibers kept, effects not re-run — and
 * every mounted `useJson` stayed bound to the dead generation.
 *
 * `clearCache()` did not cover it. The cache is read only by `useJson`'s state
 * initializer, so clearing it changes what a FUTURE mount sees and is invisible
 * to anything already on screen. The visible result was a page that looked fine
 * and was frozen: cards stuck on `E_UNAVAILABLE` against a runtime that was
 * already healthy again.
 *
 * These tests drive the real thing — a real `MessagePort`, the real RPC client,
 * a real structured-clone hop — and never unmount between generations. A test
 * that remounted would pass against the bug, because a fresh mount always
 * refetches.
 */

import { createElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { disconnectDashboardApi } from '../ui/src/lib/api.js';
import { useJson } from '../ui/src/lib/use-api.js';
import { renderPage, sequence, stubApi, type RenderedPage } from './render.js';

/** Renders whatever `/probe` currently answers, or the error it failed with. */
function Probe(): ReturnType<typeof createElement> {
  const { data, error } = useJson<{ v: string }>('/probe');
  if (error !== undefined) return createElement('div', null, `ERR:${error.message}`);
  return createElement('div', null, data?.v ?? 'pending');
}

let page: RenderedPage | undefined;
let disposers: (() => void)[] = [];

afterEach(() => {
  page?.unmount();
  page = undefined;
  for (const dispose of disposers.reverse()) dispose();
  disposers = [];
  disconnectDashboardApi();
});

describe('a replacement port revives what is already on screen', () => {
  it('refetches a mounted hook when the runtime is replaced', async () => {
    // `sequence` matters: with one static body, "refetched and got the same
    // answer" and "never refetched" are indistinguishable, and the assertion
    // would hold against the bug.
    disposers.push(stubApi({ '/probe': sequence({ v: 'gen-1' }, { v: 'gen-1-refetch' }) }));

    page = await renderPage(createElement(Probe), (t) => t.includes('gen-1'));
    expect(page.text()).toBe('gen-1');

    // ADMIN crashed and the broker handed this same live page a new port. No
    // unmount, no navigation — exactly what the renderer receives in the app.
    disposers.push(stubApi({ '/probe': sequence({ v: 'gen-2' }) }));

    await waitForText(page, 'gen-2');
    expect(page.text()).toBe('gen-2');
  });

  it('clears a stuck E_UNAVAILABLE without navigating away', async () => {
    // Mount while the runtime is unreachable — the state a page lands in when
    // it was open at the moment ADMIN died.
    disconnectDashboardApi();
    page = await renderPage(createElement(Probe), (t) => t.startsWith('ERR:'));
    expect(page.text()).toContain('ERR:');

    disposers.push(stubApi({ '/probe': sequence({ v: 'recovered' }) }));

    await waitForText(page, 'recovered');
    expect(page.text()).toBe('recovered');
  });
});

/** Flush until `page` renders `needle`, or fail loudly with what it did render. */
async function waitForText(page: RenderedPage, needle: string): Promise<void> {
  const { act } = await import('react');
  const deadline = Date.now() + 2_000;
  while (!page.text().includes(needle)) {
    if (Date.now() > deadline) {
      throw new Error(`never rendered ${needle}; last text: ${page.text() || '(empty)'}`);
    }
    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  }
}
