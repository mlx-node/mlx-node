/**
 * @vitest-environment happy-dom
 */

/**
 * Structural guards for the Overview page's loading state.
 *
 * The page fires independent requests and swaps a placeholder for each answer as it lands. Whenever a loading tile
 * holds other slots than the loaded one, the swap RESIZES the tile and moves the blocks below it — a page flinching,
 * in an order that changes on every load. happy-dom performs no layout, so heights are unassertable here; what IS
 * assertable is that a loading tile holds the same slots, in the same order, as the loaded tile, with a placeholder
 * in its value slot.
 */

import { MessageChannel } from 'node:worker_threads';

import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import Overview from '../ui/src/pages/overview.js';
import { cacheResponse, summary, timeseries, turnRow, turnsPage } from './helpers/observability-fixtures.js';
import { byRole, renderPage, stubApi, type RenderedPage } from './render.js';

/**
 * Serve the SPA a runtime that ACCEPTS every call and answers none of them, so `loading` holds for as long as the
 * test looks: a page that had already settled would otherwise satisfy every shape assertion vacuously.
 */
function stubPendingApi(): () => void {
  const { port1, port2 } = new MessageChannel();
  port1.unref();
  port2.unref();
  const dispose = serveRuntimeOverPort(
    { call: () => new Promise<never>(() => {}), subscribe: () => () => {}, watchLive: () => () => {} },
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

async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) {
    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  }
}

const TILES = [
  'Turns',
  'Output tokens',
  'Decode speed',
  'First token',
  'Failed turns',
  'Local models',
  'Cold cache',
  'Active downloads',
];

/** The tile's content slots: tag and class of each child of its content box, in order. */
function slots(page: RenderedPage, label: string): string[] {
  const tile = byRole(page.container, 'group', label);
  if (tile === null) throw new Error(`no tile ${label}`);
  const content = [...tile.children].find((child) => child.tagName === 'DIV');
  return [...(content?.children ?? [])].map((child) => `${child.tagName}.${child.className}`);
}

let mounted: RenderedPage | undefined;
let restoreApi: (() => void) | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
});

async function shapes(loading: boolean): Promise<Map<string, string[]>> {
  if (loading) {
    restoreApi = stubPendingApi();
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Overview)), (text) =>
      text.includes('Overview'),
    );
    await settle();
    expect(mounted.container.querySelectorAll('[data-placeholder]').length).toBeGreaterThan(0);
  } else {
    restoreApi = stubApi({
      '/metrics/summary': summary(),
      '/metrics/timeseries': timeseries([]),
      '/metrics/turns': turnsPage([turnRow('t1')]),
      '/models': { models: [], warnings: [], dir: '/models' },
      '/cache': cacheResponse(),
      '/downloads': { jobs: [] },
    });
    mounted = await renderPage(
      createElement(MemoryRouter, null, createElement(Overview)),
      (text) => text.includes('install a model') && text.includes('of turns') && text.includes('hit rate'),
    );
  }
  const result = new Map(TILES.map((label) => [label, slots(mounted as RenderedPage, label)]));
  if (loading) {
    for (const label of TILES) {
      const value = byRole(mounted.container, 'group', label)?.querySelector('[data-placeholder]');
      expect(value, label).not.toBeNull();
      expect(byRole(mounted.container, 'group', label)?.querySelector('[data-stat-value]'), label).toBeNull();
    }
  }
  mounted.unmount();
  mounted = undefined;
  restoreApi();
  restoreApi = undefined;
  return result;
}

describe('Overview loading state — the placeholder stands in the same slots', () => {
  it('gives every tile the same slots loading and loaded, with a placeholder in its value', async () => {
    const loading = await shapes(true);
    const loaded = await shapes(false);
    for (const label of TILES) expect(loading.get(label), label).toEqual(loaded.get(label));
  });
});
