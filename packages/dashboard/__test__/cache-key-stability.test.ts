/**
 * @vitest-environment happy-dom
 */

/**
 * A page whose request path embeds the clock can never hit its own cache.
 *
 * The Metrics page built `/metrics/overview?from=…&to=Date.now()` inside a
 * `useMemo`. React Router unmounts a route on navigation, so that memo re-ran on
 * every visit and minted a fresh key each time. Measured over CDP against the
 * real Control Panel window, every other tab was served from cache in 20-48ms while
 * Metrics missed on every single visit and took 154ms — long enough to sit blank
 * and then flash a skeleton for one frame.
 *
 * Fake timers are the whole point of this file. Two mounts in the same
 * millisecond produce identical paths WITH the bug present, so a test that just
 * mounted twice would pass against the broken code. Advancing the clock between
 * mounts is what makes the assertion mean anything.
 */

import { NOW_QUANTUM_MS, stableNow } from '@/lib/stable-now';
import Metrics from '@/pages/metrics';
import { createElement, type ReactElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import { renderPage, stubApi, type RenderedPage } from './render.js';

const EMPTY_METRICS = {
  tokensByDay: [],
  throughputByModel: [],
  throughputTrend: [],
  mtpByModel: [],
  modelShare: [],
};

/** Wall-clock start, chosen so the first mount sits well inside one quantum. */
const T0 = Date.UTC(2026, 6, 29, 12, 0, 5);

describe('stableNow', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('holds still within a quantum and moves across one', () => {
    vi.setSystemTime(T0);
    const first = stableNow();

    vi.setSystemTime(T0 + 5_000);
    expect(stableNow()).toBe(first);

    // Not merely constant — it must still advance, or the range would freeze at
    // whatever the app was launched at.
    vi.setSystemTime(T0 + NOW_QUANTUM_MS);
    expect(stableNow()).toBeGreaterThan(first);
  });
});

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

  async function pathsAcrossTwoMounts(element: ReactElement, settled: (t: string) => boolean, match: RegExp) {
    const calls: string[] = [];
    dispose = stubApi(
      { '/metrics/overview': EMPTY_METRICS },
      { onCall: (call: ApiCall) => calls.push(call.path) },
    );

    const first = await renderPage(createElement(MemoryRouter, null, element), settled);
    first.unmount();

    // Far enough that a raw `Date.now()` would certainly differ, well inside one
    // quantum so a correct implementation cannot.
    vi.setSystemTime(T0 + 5_000);

    page = await renderPage(createElement(MemoryRouter, null, element), settled);
    return calls.filter((p) => match.test(p));
  }

  it('asks for the same metrics window on a second visit', async () => {
    const paths = await pathsAcrossTwoMounts(
      createElement(Metrics),
      (text) => text.includes('Tokens per day'),
      /metrics\/overview/,
    );

    expect(paths.length).toBeGreaterThanOrEqual(2);
    expect(new Set(paths).size).toBe(1);
  });
});
