/**
 * @vitest-environment happy-dom
 */

/**
 * The shell must contain exactly one scroll container.
 *
 * It briefly had two. The shell was `min-h-screen`, which lets the DOCUMENT
 * scroll, while `main` was independently declared `overflow-auto` — two nested
 * scrolling regions for one scrolling area. Most pages hid it, because `main`
 * stretches to its content and so never actually overflows. Metrics did not:
 * recharts positions a legend `absolute`, so an 18-model legend contributed
 * 117px to `main`'s scrollHeight without ever growing its box, and both drew a
 * bar. Measured over the Chrome DevTools Protocol against the real Control Panel window:
 *
 *   before   HTML  scrollH 2294 / clientH  800     <- bar 1
 *            MAIN  scrollH 2411 / clientH 2294     <- bar 2
 *   after    MAIN only
 *
 * happy-dom performs no layout, so a test cannot re-measure that; scrollHeight
 * is 0 for everything. What it CAN pin is the class contract those measurements
 * justify — the shell sized to the viewport and refusing to scroll, with the
 * scrolling delegated to exactly one child. Reintroducing `min-h-screen` here is
 * the specific regression that brought the second bar back, and it fails this.
 */

import App from '@/App';
import { createElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { renderPage, stubApi, type RenderedPage } from './render.js';

/** Overview is the index route, so `App` lands here with no navigation. */
const ROUTES = {
  '/models': { models: [], modelsDir: '/tmp/models', warnings: [] },
  '/sessions': { sessions: [], total: 0, tokens: 0, cwds: [] },
  '/downloads': { jobs: [] },
  '/cache': {
    disk: { totalBytes: 0, quotaBytes: 0, blocks: 0, objects: 0 },
    trend: [],
    scope: { root: '/tmp/cache' },
  },
};

describe('app shell scrolling', () => {
  let dispose: (() => void) | undefined;
  let page: RenderedPage | undefined;

  afterEach(() => {
    page?.unmount();
    page = undefined;
    dispose?.();
    dispose = undefined;
  });

  async function mount(): Promise<RenderedPage> {
    dispose = stubApi(ROUTES);
    page = await renderPage(createElement(App), (text) => text.includes('Overview'));
    return page;
  }

  it('sizes the shell to the viewport and forbids it scrolling', async () => {
    const { container } = await mount();

    const shell = container.querySelector('.app-shell');
    expect(shell).not.toBeNull();
    const cls = shell?.getAttribute('class') ?? '';

    expect(cls).toContain('h-screen');
    expect(cls).toContain('overflow-hidden');
    // `min-h-screen` is the exact regression: it makes the document the second
    // scroller. `h-screen` is a substring of it, so assert its absence outright
    // rather than relying on the check above.
    expect(cls).not.toContain('min-h-screen');
  });

  it('delegates scrolling to exactly one element', async () => {
    const { container } = await mount();

    const scrollers = [...container.querySelectorAll('*')].filter((el) => {
      const cls = el.getAttribute('class') ?? '';
      return /(^|\s)overflow-(y-)?(auto|scroll)(\s|$)/.test(cls);
    });

    expect(scrollers.map((el) => el.tagName)).toEqual(['MAIN']);
  });
});
