/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { useHeld } from '../ui/src/lib/held.js';
import { useJson } from '../ui/src/lib/use-api.js';
import {
  deferred,
  renderPage,
  STUB_FAILURE,
  stubApi,
  waitFor,
  withShellProviders,
  type RenderedPage,
} from './render.js';

interface Body {
  v: string;
}

function Probe({ path }: { path: string }): ReactElement {
  const { data, held } = useHeld(useJson<Body>(path));
  return createElement('div', null, `${held ? 'HELD' : 'FRESH'} v=${data?.v ?? '-'}`);
}

let restore: (() => void) | null = null;
let page: RenderedPage | null = null;

afterEach(() => {
  page?.unmount();
  page = null;
  restore?.();
  restore = null;
});

function show(path: string): void {
  act(() => {
    page?.root.render(withShellProviders(createElement(Probe, { path })));
  });
}

describe('useHeld', () => {
  it('passes data through, holds it while a new question loads, and lets the answer replace it', async () => {
    const slow = deferred({ v: 'week' });
    restore = stubApi({ '/day': { v: 'day' }, '/week': slow.body });
    page = await renderPage(createElement(Probe, { path: '/day' }), (text) => text.includes('v=day'));
    expect(page.text()).toBe('FRESH v=day');

    show('/week');
    expect(page.text()).toBe('HELD v=day');

    slow.release();
    await waitFor(() => page?.text() === 'FRESH v=week');
  });

  it('forgets on an error without data', async () => {
    restore = stubApi({ '/day': { v: 'day' }, '/broken': STUB_FAILURE });
    page = await renderPage(createElement(Probe, { path: '/day' }), (text) => text.includes('v=day'));
    show('/broken');
    expect(page.text()).toBe('HELD v=day');
    await waitFor(() => page?.text() === 'FRESH v=-');
  });

  it('forgets on a reconnect: the old runtime is not shown as current', async () => {
    restore = stubApi({ '/day': { v: 'old-runtime' } });
    page = await renderPage(createElement(Probe, { path: '/day' }), (text) => text.includes('v=old-runtime'));
    const next = deferred({ v: 'new-runtime' });
    act(() => {
      restore?.();
      restore = stubApi({ '/day': next.body });
    });
    expect(page.text()).toBe('FRESH v=-');
    next.release();
    await waitFor(() => page?.text() === 'FRESH v=new-runtime');
  });
});
