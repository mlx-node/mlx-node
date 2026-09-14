/** @vitest-environment happy-dom */

import App from '@/App';
import type { CatalogItem } from '@/lib/types';
import Models from '@/pages/models';
import { Storage } from 'happy-dom';
import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { MODEL_CATALOG } from '../../agent/src/catalog.js';
import type { DownloadEvent } from '../src/download.js';
import type { ApiCall } from '../src/runtime.js';
import { deferred, renderPage, type RenderedPage, sequence, STUB_FAILURE, stubApi } from './render.js';

const items: CatalogItem[] = MODEL_CATALOG.map((item) => ({
  ...item,
  draft: undefined,
  slug: item.hfRepo.split('/').at(-1)!.toLowerCase(),
  installed: false,
  present: false,
  blockedByForeignDir: false,
  localRevision: null,
}));
const visible = items.filter((item) => !item.hidden);
let page: RenderedPage | undefined;
let dispose: (() => void) | undefined;

beforeEach(() => {
  // Node 25 also exposes localStorage; use browser storage for these UI tests.
  vi.stubGlobal('localStorage', new Storage());
});

afterEach(() => {
  page?.unmount();
  page = undefined;
  dispose?.();
  dispose = undefined;
  window.history.replaceState({}, '', '/');
  vi.unstubAllGlobals();
});

function routes(dir = '/test/onboarding') {
  return {
    '/models': { models: [], companions: [], warnings: [], dir },
    '/catalog': { items },
    '/catalog/updates': { items: [] },
    '/downloads': { jobs: [] },
  };
}

async function mount() {
  page = await renderPage(createElement(MemoryRouter, null, createElement(Models, { onboarding: true })), (text) =>
    text.includes('Download model'),
  );
}

function button(label: string): HTMLButtonElement {
  const found = [...page!.container.querySelectorAll('button')].find((el) => el.textContent?.trim() === label);
  if (!found) throw new Error(`Missing button ${label}: ${page!.text()}`);
  return found;
}

async function waitFor(check: () => void) {
  await vi.waitFor(async () => {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    check();
  });
}

describe('first model onboarding', () => {
  it('routes an empty first-run library into onboarding without downloading anything', async () => {
    const calls: ApiCall[] = [];
    dispose = stubApi(routes('/test/first-run'), { onCall: (call) => calls.push(call) });
    page = await renderPage(createElement(App), (text) => text.includes('Download model'));
    expect(window.location.pathname).toBe('/welcome');
    expect(page.container.querySelector('aside')).toBeNull();
    expect(calls.some((call) => call.method === 'POST')).toBe(false);
    expect(page.container.querySelectorAll('input[type=radio]')).toHaveLength(visible.length);
    const selected = page.container.querySelector<HTMLInputElement>('input:checked');
    expect(selected?.value).toBe(visible.find((item) => item.isDefault)!.hfRepo);
    for (const item of items.filter((item) => item.hidden)) expect(page.text()).not.toContain(item.label);
  });

  it('keeps returning users in their existing workspace', async () => {
    dispose = stubApi({
      ...routes(),
      '/models': { ...routes()['/models'], models: [{ name: 'existing', sizeBytes: 100 }] },
    });
    page = await renderPage(createElement(App), (text) => text.includes('Overview'));
    expect(window.location.pathname).toBe('/');
    expect(page.container.querySelector('aside')).not.toBeNull();
  });

  it('remembers Set up later for this model library', async () => {
    const dir = '/test/skipped-onboarding';
    dispose = stubApi(routes(dir));
    page = await renderPage(createElement(App), (text) => text.includes('Download model'));
    await act(async () => button('Set up later').click());
    await waitFor(() => expect(page!.text()).toContain('Overview'));
    expect(localStorage.getItem(`mlx-node:onboarding:v1:${dir}`)).toBe('done');
    expect(window.location.pathname).toBe('/');

    page.unmount();
    page = await renderPage(createElement(App), (text) => text.includes('Choose your first model'));
    expect(window.location.pathname).toBe('/');
    const resume = page.container.querySelector<HTMLAnchorElement>('a[href="/welcome"]');
    expect(resume).not.toBeNull();
    await act(async () => resume!.click());
    await waitFor(() => expect(page!.text()).toContain('Download model'));
    expect(window.location.pathname).toBe('/welcome');
  });

  it.each(['Set up later', 'Explore while downloading', 'mlx-node home'])(
    'waits for the library before allowing %s from a direct welcome visit',
    async (action) => {
      const dir = `/test/delayed-library/${action}`;
      const library = deferred(routes(dir)['/models']);
      window.history.replaceState({}, '', '/welcome');
      dispose = stubApi({
        ...routes(dir),
        '/models': library.body,
        ...(action === 'Explore while downloading'
          ? { '/downloads': { jobs: [{ id: 'ongoing', repo: visible[0].hfRepo, state: 'running' }] } }
          : {}),
      });
      page = await renderPage(createElement(App), (text) => text.includes('YOUR LOCAL AI STARTS HERE'));
      const control = () =>
        action === 'mlx-node home'
          ? page!.container.querySelector<HTMLButtonElement>('[aria-label="mlx-node home"]')!
          : button(action);
      await waitFor(() => expect(control().disabled).toBe(true));
      await act(async () => control().click());
      expect(window.location.pathname).toBe('/welcome');

      await act(async () => library.release());
      await waitFor(() => expect(control().disabled).toBe(false));
      await act(async () => control().click());
      await waitFor(() => expect(window.location.pathname).toBe('/'));
      expect(localStorage.getItem(`mlx-node:onboarding:v1:${dir}`)).toBe('done');

      page.unmount();
      page = await renderPage(createElement(App), (text) => text.includes('Overview'));
      expect(window.location.pathname).toBe('/');
    },
  );

  it('keeps exits disabled after a library failure until retry identifies the library', async () => {
    const dir = '/test/retry-library';
    const library = deferred(routes(dir)['/models']);
    window.history.replaceState({}, '', '/welcome');
    dispose = stubApi({ ...routes(dir), '/models': sequence(STUB_FAILURE, library.body) });
    page = await renderPage(createElement(App), (text) => text.includes('Try again'));
    const home = page.container.querySelector<HTMLButtonElement>('[aria-label="mlx-node home"]')!;
    expect(button('Set up later').disabled).toBe(true);
    expect(home.disabled).toBe(true);
    await act(async () => {
      button('Set up later').click();
      home.click();
    });
    expect(window.location.pathname).toBe('/welcome');

    await act(async () => button('Try again').click());
    expect(button('Set up later').disabled).toBe(true);
    await act(async () => library.release());
    await waitFor(() => expect(button('Set up later').disabled).toBe(false));
    await act(async () => button('Set up later').click());
    await waitFor(() => expect(window.location.pathname).toBe('/'));
    expect(localStorage.getItem(`mlx-node:onboarding:v1:${dir}`)).toBe('done');
  });

  it('keeps deliberate workspace navigation out of the first-launch redirect', async () => {
    window.history.replaceState({}, '', '/models');
    dispose = stubApi(routes('/test/direct-models'));
    page = await renderPage(createElement(App), (text) => text.includes('No local models yet'));
    await act(async () => page!.container.querySelector<HTMLAnchorElement>('nav a[href="/"]')!.click());
    await waitFor(() => expect(page!.text()).toContain('Choose your first model'));
    expect(window.location.pathname).toBe('/');
    expect(page.container.querySelector('aside')).not.toBeNull();
  });

  it('continues from a ready model to coding agent setup and remembers completion', async () => {
    const dir = '/test/completed-onboarding';
    window.history.replaceState({}, '', '/welcome');
    dispose = stubApi({
      ...routes(dir),
      '/catalog': { items: items.map((item) => ({ ...item, present: item.isDefault })) },
    });
    page = await renderPage(createElement(App), (text) => text.includes('Set up coding agents'));
    await act(async () => button('Set up coding agents').click());
    await waitFor(() => expect(window.location.pathname).toBe('/coding-agents'));
    expect(localStorage.getItem(`mlx-node:onboarding:v1:${dir}`)).toBe('done');
    expect(page.container.querySelector('aside')).not.toBeNull();
  });

  it('downloads the chosen model and waits for refreshed install state before continuing', async () => {
    const chosen = visible[1];
    const calls: ApiCall[] = [];
    let listener: ((event: DownloadEvent) => void) | undefined;
    const refreshed = deferred({ items: items.map((item) => ({ ...item, present: item.hfRepo === chosen.hfRepo })) });
    dispose = stubApi(
      {
        ...routes(),
        '/catalog': sequence({ items }, refreshed.body),
        '/downloads': sequence({ jobs: [] }, { id: 'chosen-job', repo: chosen.hfRepo }),
        '/downloads/chosen-job': { cancelled: true, id: 'chosen-job' },
      },
      {
        onCall: (call) => calls.push(call),
        subscribe: (_id, fn) => {
          listener = fn;
          return () => {
            listener = undefined;
          };
        },
      },
    );
    await mount();
    const radio = page!.container.querySelector<HTMLInputElement>(`input[value="${chosen.hfRepo}"]`)!;
    await act(async () => radio.click());
    expect(radio.checked).toBe(true);
    await act(async () => button('Download model').click());
    await waitFor(() => expect(listener).toBeDefined());
    expect(calls.find((call) => call.method === 'POST')?.body).toEqual({ repo: chosen.hfRepo });
    await act(async () => listener!({ type: 'done', id: 'chosen-job', outputDir: `/models/${chosen.slug}` }));
    await waitFor(() => expect(page!.text()).toContain('Checking download…'));
    expect(page!.text()).not.toContain('Set up coding agents');
    await act(async () => refreshed.release());
    await waitFor(() => expect(page!.text()).toContain('Set up coding agents'));
    expect(page!.text()).toContain('YOUR MODEL IS READY');
    expect(page!.container.querySelector<HTMLInputElement>('input:checked')?.value).toBe(chosen.hfRepo);
  });

  it('restores an existing non-default download and recovers after cancellation', async () => {
    const chosen = visible[1];
    let listener: ((event: DownloadEvent) => void) | undefined;
    dispose = stubApi(
      {
        ...routes(),
        '/downloads': {
          jobs: [{ id: 'existing-job', repo: chosen.hfRepo, state: 'running', receivedBytes: 10, totalBytes: 100 }],
        },
        '/downloads/existing-job': { cancelled: true, id: 'existing-job' },
      },
      {
        subscribe: (_id, fn) => {
          listener = fn;
          return () => {
            listener = undefined;
          };
        },
      },
    );
    page = await renderPage(createElement(MemoryRouter, null, createElement(Models, { onboarding: true })), (text) =>
      text.includes('MAKING IT LOCAL'),
    );
    await waitFor(() => expect(listener).toBeDefined());
    expect(page.container.querySelector<HTMLInputElement>('input:checked')?.value).toBe(chosen.hfRepo);
    expect(page.container.querySelector('fieldset')?.disabled).toBe(true);
    await act(async () => button('Cancel').click());
    await act(async () => listener!({ type: 'cancelled', id: 'existing-job' }));
    await waitFor(() => expect(page!.text()).toContain('Download model'));
    expect(page.container.querySelector('fieldset')?.disabled).toBe(false);
  });

  it('shows a failed download only for its model and makes retry available', async () => {
    let listener: ((event: DownloadEvent) => void) | undefined;
    dispose = stubApi(
      {
        ...routes(),
        '/downloads': { jobs: [{ id: 'failed-job', repo: visible[0].hfRepo, state: 'running' }] },
        '/downloads/failed-job': {},
      },
      {
        subscribe: (_id, fn) => {
          listener = fn;
          return () => {
            listener = undefined;
          };
        },
      },
    );
    page = await renderPage(createElement(MemoryRouter, null, createElement(Models, { onboarding: true })), (text) =>
      text.includes('MAKING IT LOCAL'),
    );
    await waitFor(() => expect(listener).toBeDefined());
    await act(async () => listener!({ type: 'error', id: 'failed-job', message: 'Connection lost' }));
    await waitFor(() => expect(page!.text()).toContain('Download model'));
    expect(page.container.querySelector('[role=alert]')?.textContent).toContain('Connection lost');
    expect(button('Download model').disabled).toBe(false);

    await act(async () =>
      page!.container.querySelector<HTMLInputElement>(`input[value="${visible[1].hfRepo}"]`)!.click(),
    );
    expect(page.container.querySelector('[role=alert]')).toBeNull();
    expect(button('Download model').disabled).toBe(false);
    await act(async () =>
      page!.container.querySelector<HTMLInputElement>(`input[value="${visible[0].hfRepo}"]`)!.click(),
    );
    expect(page.container.querySelector('[role=alert]')?.textContent).toContain('Connection lost');
  });

  it('keeps a late start failure with the requested model after selection changes', async () => {
    const failedStart = deferred(STUB_FAILURE);
    const calls: ApiCall[] = [];
    dispose = stubApi(
      { ...routes(), '/downloads': sequence({ jobs: [] }, failedStart.body) },
      { onCall: (call) => calls.push(call) },
    );
    await mount();
    const requested = page!.container.querySelector<HTMLInputElement>('input:checked')!.value;
    const other = visible.find((item) => item.hfRepo !== requested)!;
    await act(async () => button('Download model').click());
    await waitFor(() => expect(calls.some((call) => call.method === 'POST')).toBe(true));
    await act(async () => page!.container.querySelector<HTMLInputElement>(`input[value="${other.hfRepo}"]`)!.click());
    await act(async () => failedStart.release());
    await waitFor(() => expect(button('Download model').disabled).toBe(false));
    expect(page!.container.querySelector('[role=alert]')).toBeNull();

    await act(async () => page!.container.querySelector<HTMLInputElement>(`input[value="${requested}"]`)!.click());
    expect(page!.container.querySelector('[role=alert]')?.textContent).toContain('stubbed failure for /downloads');
    expect(calls.filter((call) => call.method === 'POST')).toHaveLength(1);
  });

  it('offers a retry when library loading fails instead of offering a download', async () => {
    dispose = stubApi({ ...routes(), '/catalog': sequence(STUB_FAILURE, { items }) });
    page = await renderPage(createElement(MemoryRouter, null, createElement(Models, { onboarding: true })), (text) =>
      text.includes('Try again'),
    );
    expect(page.text()).not.toContain('Download model');
    await act(async () => button('Try again').click());
    await waitFor(() => expect(page!.text()).toContain('Download model'));
  });
});
