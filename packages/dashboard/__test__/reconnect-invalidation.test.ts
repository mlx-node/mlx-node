/**
 * @vitest-environment happy-dom
 */

/**
 * What a mounted page does when CONTROL PANEL dies and comes back.
 *
 * `broker.ts` goes to real trouble here: when the CONTROL PANEL utilityProcess exits it
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

import { act, createElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import type { RpcPort, RpcPortEvents } from '../src/rpc/port.js';
import type { RpcRequest } from '../src/rpc/protocol.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import { readCache } from '../ui/src/lib/json-cache.js';
import { useJson } from '../ui/src/lib/use-api.js';
import { renderPage, sequence, stubApi, type RenderedPage } from './render.js';

/** Renders whatever `/probe` currently answers, or the error it failed with. */
function Probe(): ReturnType<typeof createElement> {
  const { data, error } = useJson<{ v: string }>('/probe');
  if (error !== undefined) return createElement('div', null, `ERR:${error.message}`);
  return createElement('div', null, data?.v ?? 'pending');
}

/** Makes the pre-response state observable instead of waiting until it settles. */
function StateProbe(): ReturnType<typeof createElement> {
  const { data, error, loading, refreshing } = useJson<{ v: string }>('/probe');
  if (error !== undefined) return createElement('div', null, `ERR:${error.message}`);
  const phase = loading ? 'LOADING' : refreshing ? 'REFRESHING' : 'IDLE';
  return createElement('div', null, `${phase}:${data?.v ?? 'pending'}`);
}

let page: RenderedPage | undefined;
let disposers: (() => void)[] = [];

interface ControlledApi {
  posted: RpcRequest[];
  respond(body: unknown): void;
}

/** A deterministic RpcPort whose authoritative replies are released by the test. */
function connectControlledApi(): ControlledApi {
  const posted: RpcRequest[] = [];
  let events: RpcPortEvents | undefined;
  connectDashboardApi(
    {
      postMessage(message: unknown): void {
        posted.push(message as RpcRequest);
      },
      listen(next): () => void {
        events = next;
        return () => {};
      },
      close: () => {},
    } satisfies RpcPort,
    { onUnresponsive: () => {} },
  );
  return {
    posted,
    respond(body: unknown): void {
      const call = posted.find((message) => message.kind === 'call');
      if (call === undefined || call.kind !== 'call') throw new Error('controlled API has no pending call');
      events?.onMessage({ kind: 'response', id: call.id, response: { ok: true, status: 200, body } });
    },
  };
}

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

    // CONTROL PANEL crashed and the broker handed this same live page a new port. No
    // unmount, no navigation — exactly what the renderer receives in the app.
    disposers.push(stubApi({ '/probe': sequence({ v: 'gen-2' }) }));

    await waitForText(page, 'gen-2');
    expect(page.text()).toBe('gen-2');
  });

  it('clears a stuck E_UNAVAILABLE without navigating away', async () => {
    // Mount while the runtime is unreachable — the state a page lands in when
    // it was open at the moment CONTROL PANEL died.
    disconnectDashboardApi();
    page = await renderPage(createElement(Probe), (t) => t.startsWith('ERR:'));
    expect(page.text()).toContain('ERR:');

    disposers.push(stubApi({ '/probe': sequence({ v: 'recovered' }) }));

    await waitForText(page, 'recovered');
    expect(page.text()).toBe('recovered');
  });

  it('removes retired data while the replacement runtime is still answering', async () => {
    const old = connectControlledApi();
    page = await renderPage(createElement(StateProbe), (text) => text === 'LOADING:pending');

    await act(async () => {
      old.respond({ v: 'gen-1' });
      await Promise.resolve();
    });
    expect(page.text()).toBe('IDLE:gen-1');

    let replacement: ControlledApi | undefined;
    await act(async () => {
      replacement = connectControlledApi();
      await Promise.resolve();
    });

    // The new call is deliberately withheld. Clearing only the shared cache
    // would leave the mounted request at IDLE:gen-1 until this call settled.
    expect(replacement?.posted.map((message) => message.kind)).toEqual(['call']);
    expect(page.text()).toBe('LOADING:pending');

    await act(async () => {
      replacement?.respond({ v: 'gen-2' });
      await Promise.resolve();
    });
    expect(page.text()).toBe('IDLE:gen-2');
  });

  it('does not let an authoritative result from the retired runtime repopulate the cache', async () => {
    const old = connectControlledApi();
    page = await renderPage(createElement(Probe), (text) => text === 'pending');
    expect(old.posted.map((message) => message.kind)).toEqual(['call']);

    let replacement: ControlledApi | undefined;
    await act(async () => {
      replacement = connectControlledApi();
      await Promise.resolve();
    });
    expect(old.posted.map((message) => message.kind)).toEqual(['call', 'cancel']);
    expect(replacement?.posted.map((message) => message.kind)).toEqual(['call']);

    // The replacement runtime wins first and owns both the visible state and
    // the cache entry future mounts will seed from.
    await act(async () => {
      replacement?.respond({ v: 'gen-2' });
      await Promise.resolve();
    });
    expect(page.text()).toBe('gen-2');
    expect(readCache<{ v: string }>('/probe')?.value).toEqual({ v: 'gen-2' });

    // Closing the old RpcClient does not discard authoritative work that had
    // already started. Its late result must remain deliverable without being
    // allowed to poison the replacement generation's cache.
    await act(async () => {
      old.respond({ v: 'gen-1-late' });
      await Promise.resolve();
    });
    expect(page.text()).toBe('gen-2');
    expect(readCache<{ v: string }>('/probe')?.value).toEqual({ v: 'gen-2' });
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
