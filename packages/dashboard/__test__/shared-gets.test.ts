/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import { mutate } from '../ui/src/lib/api.js';
import type { LiveEvent } from '../ui/src/lib/types.js';
import { useJson, type AsyncState } from '../ui/src/lib/use-api.js';
import {
  deferred,
  renderPage,
  sequence,
  STUB_FAILURE,
  stubApi,
  waitFor,
  withShellProviders,
  type RenderedPage,
  type StubbedApi,
} from './render.js';

interface Body {
  n: number;
}

const hooks = new Map<string, AsyncState<Body>>();

function Probe({ id, live }: { id: string; live?: boolean }): ReactElement {
  const state = useJson<Body>('/models', live === true ? { live: { minIntervalMs: 0 } } : {});
  hooks.set(id, state);
  const text = state.error !== undefined ? `ERR:${state.error.message}` : `n=${state.data?.n ?? '-'}`;
  return createElement('div', { 'data-id': id }, `${id}:${text}`);
}

function Probes({ ids, live }: { ids: readonly string[]; live?: boolean }): ReactElement {
  return createElement(
    'div',
    null,
    ids.map((id) => createElement(Probe, { key: id, id, live })),
  );
}

function hook(id: string): AsyncState<Body> {
  const found = hooks.get(id);
  if (found === undefined) throw new Error(`no hook ${id}`);
  return found;
}

let api: StubbedApi | undefined;
let page: RenderedPage | undefined;
let calls: ApiCall[] = [];

function modelGets(): number {
  return calls.filter((call) => call.method === 'GET' && call.path === '/api/models').length;
}

function connect(routes: Record<string, unknown>): void {
  api = stubApi(routes, { onCall: (call) => calls.push(call) });
}

async function mount(ids: readonly string[], until: (text: string) => boolean, live = false): Promise<void> {
  page = await renderPage(createElement(Probes, { ids, live }), until);
}

function rerender(ids: readonly string[], live = false): void {
  act(() => {
    page!.root.render(withShellProviders(createElement(Probes, { ids, live })));
  });
}

async function turns(count: number): Promise<void> {
  for (let i = 0; i < count; i++) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
}

afterEach(() => {
  page?.unmount();
  page = undefined;
  api?.();
  api = undefined;
  calls = [];
  hooks.clear();
});

describe('identical GETs in flight share one request', () => {
  it('three hooks mounted together on /models make one call, and a GET after a mutation makes its own', async () => {
    const held = deferred({ n: 1 });
    connect({ '/models': sequence(held.body, { n: 2 }), '/thing': { ok: true } });
    await mount(['a', 'b', 'c'], (text) => text.includes('a:n=-'));
    await turns(3);
    expect(modelGets()).toBe(1);

    await act(async () => {
      await mutate('DELETE', '/thing');
    });
    rerender(['a', 'b', 'c', 'd']);
    await waitFor(() => page!.text().includes('d:n=2'));
    expect(modelGets()).toBe(2);
    held.release();
    await waitFor(() => page!.text().includes('a:n=1'));
  });

  it('a patch refetches with its own request, never the one in flight before it', async () => {
    const held = deferred({ n: 1 });
    connect({ '/models': sequence({ n: 0 }, held.body, { n: 3 }) });
    await mount(['a', 'b'], (text) => text.includes('a:n=0') && text.includes('b:n=0'));
    act(() => hook('b').reload());
    await waitFor(() => modelGets() === 2);
    act(() => hook('a').patch((body) => ({ n: body.n + 10 })));
    await waitFor(() => page!.text().includes('a:n=3'));
    expect(modelGets()).toBe(3);
    held.release();
  });

  it('a reload never joins a request that started before it; reloads called together share one', async () => {
    const held = deferred({ n: 1 });
    connect({ '/models': sequence(held.body, { n: 2 }) });
    await mount(['a', 'b'], (text) => text.includes('a:n=-'));
    await turns(3);
    expect(modelGets()).toBe(1);
    act(() => {
      hook('a').reload();
      hook('b').reload();
    });
    await waitFor(() => page!.text().includes('a:n=2') && page!.text().includes('b:n=2'));
    expect(modelGets()).toBe(2);
    held.release();
  });

  it('a reconnect never joins a request of the old connection', async () => {
    const held = deferred({ n: 1 });
    connect({ '/models': held.body });
    await mount(['a', 'b'], (text) => text.includes('a:n=-'));
    await turns(3);
    expect(modelGets()).toBe(1);
    const old = api!;
    const oldCalls = calls;
    calls = [];
    api = stubApi({ '/models': { n: 5 } }, { onCall: (call) => calls.push(call) });
    await waitFor(() => page!.text().includes('a:n=5') && page!.text().includes('b:n=5'));
    expect(modelGets()).toBe(1);
    expect(oldCalls.length).toBe(1);
    held.release();
    old();
  });

  it('one sharer that unmounts leaves the request to the others, and it completes into the cache', async () => {
    const held = deferred({ n: 4 });
    connect({ '/models': sequence(held.body, deferred({ n: 9 }).body) });
    await mount(['a', 'b'], (text) => text.includes('a:n=-'));
    await turns(3);
    rerender(['b']);
    held.release();
    await waitFor(() => page!.text().includes('b:n=4'));
    expect(modelGets()).toBe(1);
    page!.unmount();
    page = await renderPage(createElement(Probes, { ids: ['c'] }), (text) => text.includes('c:n=4'));
  });

  it('an error reaches every sharer and a failed request is not reused', async () => {
    const held = deferred(STUB_FAILURE);
    connect({ '/models': sequence(held.body, { n: 6 }) });
    await mount(['a', 'b'], (text) => text.includes('a:n=-'));
    await turns(3);
    held.release();
    await waitFor(() => page!.text().includes('a:ERR:') && page!.text().includes('b:ERR:'));
    expect(modelGets()).toBe(1);
    rerender(['a', 'b', 'c']);
    await waitFor(() => page!.text().includes('c:n=6'));
    expect(modelGets()).toBe(2);
  });

  it('live queries on one path refetch once per data event', async () => {
    const data: LiveEvent = { type: 'data', version: 1, traces: true, runtime: false };
    connect({ '/models': sequence({ n: 0 }, { n: 1 }) });
    await mount(['a', 'b'], (text) => text.includes('a:n=0') && text.includes('b:n=0'), true);
    expect(modelGets()).toBe(1);
    await turns(2);
    api!.emitLive(data);
    await waitFor(() => page!.text().includes('a:n=1') && page!.text().includes('b:n=1'));
    await turns(3);
    expect(modelGets()).toBe(2);
  });
});
