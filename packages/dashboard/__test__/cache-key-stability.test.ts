/**
 * @vitest-environment happy-dom
 */

/**
 * A page whose request path embeds the clock can never hit its own cache.
 *
 * The Metrics page once built its window with `to=Date.now()` inside a `useMemo`. React Router unmounts a route on
 * navigation, so that memo re-ran on every visit and minted a fresh key each time: Metrics missed its cache on every
 * visit and flashed a skeleton. The window now ends at the end of the current bucket (`alignedRange`), so two visits
 * inside one bucket ask for the same paths.
 *
 * Fake timers are the whole point of this file. Two mounts in the same millisecond produce identical paths WITH the
 * bug present; advancing the clock between mounts is what makes the assertion mean anything.
 */

import Metrics from '@/pages/metrics';
import Sessions, { sessionsPath } from '@/pages/sessions';
import { createElement, type ReactElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import { emptySummary, timeseries } from './helpers/observability-fixtures.js';
import { renderPage, stubApi, type RenderedPage } from './render.js';

const EMPTY_SESSIONS = { sessions: [], total: 0, cwds: [] };

const ROUTES = {
  '/sessions': EMPTY_SESSIONS,
  '/metrics/models': { models: [] },
  '/metrics/summary': emptySummary(),
  '/metrics/timeseries': timeseries([]),
  '/models': { models: [], warnings: [], dir: '/models' },
};

/** Local wall-clock start, five seconds into an hour, so the first mount sits well inside one bucket. */
const T0 = new Date(2026, 6, 29, 12, 0, 5).getTime();

describe('request paths survive a remount', () => {
  let dispose: (() => void) | undefined;
  let page: RenderedPage | undefined;

  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(T0);
  });

  afterEach(() => {
    page?.unmount();
    page = undefined;
    dispose?.();
    dispose = undefined;
    vi.useRealTimers();
  });

  async function pathsAcrossTwoMounts(element: ReactElement, settled: (t: string) => boolean, match: string) {
    const calls: string[] = [];
    dispose = stubApi(ROUTES, { onCall: (call: ApiCall) => calls.push(call.path) });

    const first = await renderPage(createElement(MemoryRouter, null, element), settled);
    first.unmount();

    // Far enough that a raw `Date.now()` would certainly differ, well inside one
    // bucket so a correct implementation cannot.
    vi.setSystemTime(T0 + 5_000);

    page = await renderPage(createElement(MemoryRouter, null, element), settled);
    return calls.filter((p) => p.includes(match));
  }

  for (const route of ['/metrics/summary?', '/metrics/timeseries?']) {
    it(`asks for the same ${route.slice(0, -1)} window on a second visit`, async () => {
      const paths = await pathsAcrossTwoMounts(
        createElement(Metrics),
        (text) => text.includes('Where the time goes') && text.includes('No turns in this range'),
        route,
      );

      expect(paths.length).toBeGreaterThanOrEqual(2);
      expect(new Set(paths).size).toBe(route === '/metrics/summary?' ? 1 : 2);
    });
  }
  it('asks for the same sessions list on a second visit', async () => {
    const paths = await pathsAcrossTwoMounts(
      createElement(Sessions),
      (text) => text.includes('No agent sessions yet'),
      '/api/sessions',
    );

    expect(paths.length).toBeGreaterThanOrEqual(2);
    expect(new Set(paths).size).toBe(1);
  });

  it('builds the sessions path from the filters alone', () => {
    vi.setSystemTime(T0);
    const before = sessionsPath({ query: 'cache', cwd: '/work' });
    vi.setSystemTime(T0 + 180_000);
    expect(sessionsPath({ query: 'cache', cwd: '/work' })).toBe(before);
    expect(before).toBe('/sessions?q=cache&cwd=%2Fwork&limit=500');
    expect(sessionsPath({ query: '', cwd: 'all' })).toBe('/sessions?limit=500');
  });
});
