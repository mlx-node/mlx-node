/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import { formatBytes, formatRelativeTime } from '../ui/src/lib/format.js';
import { deleteNamespace, prefixTotals, STEP_GAP_MS } from '../ui/src/lib/prefix-cache.js';
import { toast } from '../ui/src/lib/toast.js';
import Cache from '../ui/src/pages/cache.js';
import {
  cacheResponse,
  GIB,
  prefixCacheResponse,
  prefixNamespace,
  prefixRoot,
} from './helpers/observability-fixtures.js';
import {
  byRole,
  deferred,
  renderPage,
  sequence,
  STUB_FAILURE,
  stubApi,
  stubError,
  waitFor,
  type RenderedPage,
  type StubbedApi,
} from './render.js';

const HELD = 'c1c3c580f932a1b2c3d4e5f65b0bd7af';
const IDLE = '3e5088fc2280a1b2c3d4e5f677c10b0f';
const OTHER = '9a41e07c55d1a1b2c3d4e5f6e0a922c8';

let mounted: RenderedPage | undefined;
let restoreApi: StubbedApi | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

async function mountCache(prefix: unknown, extra: Record<string, unknown> = {}, onCall?: (call: ApiCall) => void) {
  restoreApi = stubApi({ '/cache': cacheResponse(), '/cache/prefix': prefix, ...extra }, { onCall });
  mounted = await renderPage(createElement(MemoryRouter, null, createElement(Cache)), (text) =>
    text.includes('mlx-paged-v1'),
  );
  return mounted;
}

function click(element: Element | null | undefined): void {
  if (element === null || element === undefined) throw new Error('click: no element');
  act(() => {
    (element as HTMLElement).click();
  });
}

function dialog(): HTMLElement | null {
  const all = document.querySelectorAll<HTMLElement>('[role="dialog"]');
  return all.length === 0 ? null : all[all.length - 1];
}

function buttonNamed(scope: ParentNode, name: string): HTMLButtonElement | null {
  return [...scope.querySelectorAll('button')].find((button) => (button.textContent ?? '').trim() === name) ?? null;
}

function panels(page: RenderedPage): HTMLElement[] {
  return [...page.container.querySelectorAll<HTMLElement>('section')].filter((section) =>
    (section.querySelector(':scope > header h2')?.textContent ?? '').startsWith('Model caches'),
  );
}

function row(scope: ParentNode, namespace: string): HTMLElement | null {
  return scope.querySelector<HTMLElement>(`tr[data-namespace="${namespace}"]`);
}

function prefixGets(calls: ApiCall[]): number {
  return calls.filter((call) => call.method === 'GET' && call.path.split('?')[0] === '/api/cache/prefix').length;
}

const lastUsed = Date.now() - 3 * 86_400_000;

function listing() {
  return prefixCacheResponse({
    roots: [
      prefixRoot({
        namespaces: [
          prefixNamespace({ namespace: HELD, lock: 'held', bytes: 12 * GIB, files: 27_004 }),
          prefixNamespace({ namespace: IDLE, lastUsedMs: lastUsed, openedMs: lastUsed }),
        ],
        engines: [{ pid: 4242, host: 'desktop', limitBytes: 32 * GIB }],
      }),
      prefixRoot({
        root: '/Volumes/fast/prefix-cache',
        isDefault: false,
        namespaces: [prefixNamespace({ namespace: OTHER, bytes: 112 * 1024 * 1024, files: 14 })],
        engines: [{ pid: 4411, host: 'serve', limitBytes: null }],
      }),
    ],
  });
}

describe('Cache page model caches', () => {
  it('lists the model caches', async () => {
    const page = await mountCache(listing());
    await waitFor(() => panels(page).length === 2 && row(panels(page)[1]!, OTHER) !== null);
    const [main, other] = panels(page);
    expect(main!.querySelector('h2')?.textContent).toBe('Model caches');
    expect(other!.querySelector('h2')?.textContent).toBe('Model caches · mlx serve engine, pid 4411');
    expect(row(main!, OTHER)).toBeNull();
    expect(row(other!, HELD)).toBeNull();
    const held = row(main!, HELD)?.textContent ?? '';
    expect(held).toContain('In use');
    expect(held).toContain('12.0 GB');
    expect(held).toContain('in use now');
    const idle = row(main!, IDLE)?.textContent ?? '';
    expect(idle).toContain('Idle');
    expect(idle).toContain(formatRelativeTime(lastUsed));
    expect(row(other!, OTHER)?.textContent).toContain('112 MB');
    expect(main!.textContent).toContain('56% of 32.0 GB limit');
    expect(main!.textContent).toContain('1 in use · 1 idle');
  });

  it('names the model a running engine reports, and only the hash of any other cache', async () => {
    const body = listing();
    body.roots[0]!.namespaces[0] = { ...body.roots[0]!.namespaces[0]!, model: 'Qwen3.6-27B-4bit' };
    const page = await mountCache(body);
    await waitFor(() => row(page.container, HELD) !== null);
    const named = row(page.container, HELD)!.querySelector('td');
    expect(named?.textContent).toContain('Qwen3.6-27B-4bit');
    expect(named?.textContent).toContain(HELD.slice(0, 8));
    const unnamed = row(page.container, IDLE)!.querySelector('td');
    expect(unnamed?.textContent).toContain(IDLE);
    expect(unnamed?.textContent).not.toContain('Qwen');
  });

  it('the delete button of a cache in use is off and says why', async () => {
    const page = await mountCache(listing());
    await waitFor(() => row(page.container, HELD) !== null);
    const trigger = byRole(row(page.container, HELD)!, 'button', `Delete cache ${HELD.slice(0, 8)}…`);
    expect(trigger?.getAttribute('aria-disabled')).toBe('true');
    click(trigger);
    expect(dialog()).toBeNull();
    const idle = byRole(row(page.container, IDLE)!, 'button', `Delete cache ${IDLE.slice(0, 8)}…`);
    expect(idle?.getAttribute('aria-disabled')).toBeNull();
  });

  it('deletes in steps after the confirm', async () => {
    const success = vi.spyOn(toast, 'success');
    const calls: ApiCall[] = [];
    const after = listing();
    after.roots[0]!.namespaces = after.roots[0]!.namespaces.filter((ns) => ns.namespace !== IDLE);
    const step = { removed: 4096, freedBytes: 2 * GIB, done: false, left: 0, directoryRemoved: false };
    const last = { removed: 10, freedBytes: GIB, done: true, left: 1, directoryRemoved: false };
    const page = await mountCache(
      sequence(listing(), after),
      { [`DELETE /cache/prefix/${IDLE}`]: sequence(step, last) },
      (call) => calls.push(call),
    );
    await waitFor(() => row(page.container, IDLE) !== null);
    click(byRole(row(page.container, IDLE)!, 'button', `Delete cache ${IDLE.slice(0, 8)}…`));
    await waitFor(() => dialog() !== null);
    expect(dialog()?.textContent).toContain('Delete this model cache?');
    expect(dialog()?.textContent).toContain(`Removes 14,190 files · 6.0 GB from ${IDLE.slice(0, 8)}…${IDLE.slice(-8)}`);
    click(buttonNamed(dialog()!, 'Delete'));
    await waitFor(() => success.mock.calls.length > 0, 3000);
    const deletes = calls.filter((call) => call.method === 'DELETE');
    expect(deletes.map((call) => [call.path, call.body])).toEqual([
      [`/api/cache/prefix/${IDLE}`, { root: '/tmp/prefix-cache', openedMs: lastUsed }],
      [`/api/cache/prefix/${IDLE}`, { root: '/tmp/prefix-cache', openedMs: lastUsed }],
    ]);
    expect(success.mock.calls[0]).toEqual([
      'Deleted model cache',
      { description: `4,106 files removed · ${formatBytes(3 * GIB)} freed · 1 folder inside was left` },
    ]);
    await waitFor(() => row(page.container, IDLE) === null);
  });

  it('shows lower bounds while counting and asks again', async () => {
    const calls: ApiCall[] = [];
    const counting = prefixCacheResponse({
      counting: true,
      roots: [
        prefixRoot({ namespaces: [prefixNamespace({ namespace: IDLE, complete: false, bytes: 112 * 1024 * 1024 })] }),
      ],
    });
    const done = deferred(
      prefixCacheResponse({ roots: [prefixRoot({ namespaces: [prefixNamespace({ namespace: IDLE })] })] }),
    );
    const page = await mountCache(sequence(counting, done.body), {}, (call) => calls.push(call));
    await waitFor(() => (row(page.container, IDLE)?.textContent ?? '').includes('≥'));
    expect(page.text()).toContain('counting files…');
    done.release();
    await waitFor(() => (row(page.container, IDLE)?.textContent ?? '').includes('6.0 GB'));
    expect(row(page.container, IDLE)?.textContent).not.toContain('≥');
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(prefixGets(calls)).toBe(2);
  });

  it('a failed listing leaves the paged cache on the page', async () => {
    const page = await mountCache(STUB_FAILURE);
    await waitFor(() => page.text().includes('stubbed failure for /cache/prefix'));
    expect(byRole(page.container, 'group', 'Usage')).not.toBeNull();
    expect(page.text()).toContain('Objects by age');
  });

  it('prefixTotals adds the rows and takes the smallest limit', () => {
    const totals = prefixTotals({
      namespaces: [
        prefixNamespace({ bytes: 10, files: 1, lock: 'held' }),
        prefixNamespace({ bytes: 20, files: 2, lock: 'free', complete: false }),
        prefixNamespace({ bytes: 30, files: 3, lock: 'unknown' }),
      ],
      engines: [
        { pid: 1, host: 'desktop', limitBytes: 500 },
        { pid: 2, host: 'serve', limitBytes: null },
        { pid: 3, host: 'serve', limitBytes: 300 },
      ],
    });
    expect(totals).toEqual({
      bytes: 60,
      files: 6,
      caches: 3,
      held: 1,
      idle: 1,
      unknown: 1,
      complete: false,
      limitBytes: 300,
      limitEngines: 2,
    });
  });

  it('says the folder is not created yet', async () => {
    const page = await mountCache(prefixCacheResponse({ roots: [prefixRoot({ exists: false, volume: null })] }));
    await waitFor(() => page.text().includes('No model cache yet'));
    expect(panels(page)[0]?.textContent).toContain('/tmp/prefix-cache · not created yet');
    expect(panels(page)[0]?.textContent).toContain('A Qwen3.5 or Flash Next model writes here');
    expect(panels(page)[0]?.textContent).not.toContain('could not be measured');
  });

  it('says a folder it could not read, not that it is empty', async () => {
    const page = await mountCache(
      prefixCacheResponse({ roots: [prefixRoot({ exists: true, readError: 'EACCES', namespaces: [] })] }),
    );
    await waitFor(() => page.text().includes('Could not read this folder'));
    const panel = panels(page)[0]?.textContent ?? '';
    expect(panel).toContain('/tmp/prefix-cache · could not be read');
    expect(panel).toContain('The system answered EACCES');
    expect(panel).not.toContain('No model cache yet');
  });

  it('leaves the lock free between two delete steps for longer than the engine waits between its tries', async () => {
    const events: string[] = [];
    const step = { removed: 4096, freedBytes: 1, done: false, left: 0, directoryRemoved: false };
    const last = { removed: 1, freedBytes: 1, done: true, left: 0, directoryRemoved: true };
    restoreApi = stubApi(
      { [`DELETE /cache/prefix/${IDLE}`]: sequence(step, last) },
      { onCall: (call) => events.push(call.method) },
    );
    const result = await deleteNamespace('/tmp/prefix-cache', IDLE, lastUsed, async (ms) => {
      events.push(`wait ${String(ms)}`);
    });
    expect(events).toEqual(['DELETE', `wait ${String(STEP_GAP_MS)}`, 'DELETE']);
    expect(STEP_GAP_MS).toBeGreaterThan(200);
    expect(result).toMatchObject({ removed: 4097, done: true, directoryRemoved: true });
  });

  it('keeps the rows when a refresh fails', async () => {
    const page = await mountCache(sequence(listing(), STUB_FAILURE));
    await waitFor(() => row(page.container, IDLE) !== null);
    click(byRole(page.container, 'button', 'Refresh data'));
    await waitFor(() => (panels(page)[0]?.textContent ?? '').includes('Could not refresh'));
    expect(row(page.container, IDLE)).not.toBeNull();
    expect(row(page.container, HELD)).not.toBeNull();
  });

  it('shows why a delete was refused and reloads the list', async () => {
    const calls: ApiCall[] = [];
    const message = 'An engine opened this cache after it was listed. Refresh and look again.';
    const page = await mountCache(
      listing(),
      { [`DELETE /cache/prefix/${IDLE}`]: stubError('E_CONFLICT', message, 'changed') },
      (call) => calls.push(call),
    );
    await waitFor(() => row(page.container, IDLE) !== null);
    expect(prefixGets(calls)).toBe(1);
    click(byRole(row(page.container, IDLE)!, 'button', `Delete cache ${IDLE.slice(0, 8)}…`));
    await waitFor(() => dialog() !== null);
    click(buttonNamed(dialog()!, 'Delete'));
    await waitFor(() => (dialog()?.querySelector('[role="alert"]')?.textContent ?? '').includes(message), 3000);
    expect(dialog()?.textContent).toContain('Could not delete the cache');
    await waitFor(() => prefixGets(calls) === 2);
  });
});
