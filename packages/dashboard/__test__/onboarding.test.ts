/** @vitest-environment happy-dom */

import App from '@/App';
import type { CatalogBuildItem, CatalogItem } from '@/lib/types';
import Models from '@/pages/models';
import { Storage } from 'happy-dom';
import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { catalogBuildRepo, catalogBuildSlug, MODEL_CATALOG } from '../../agent/src/catalog.js';
import type { DownloadEvent } from '../src/download.js';
import type { ApiCall } from '../src/runtime.js';
import { splitWhitespace } from '../ui/src/lib/text-scan.js';
import { deferred, renderPage, type RenderedPage, sequence, STUB_FAILURE, stubApi, stubError } from './render.js';

const items: CatalogItem[] = MODEL_CATALOG.map((item) => ({
  ...item,
  draft: undefined,
  builds: item.builds.map((build): CatalogBuildItem => ({
    id: build.id,
    hfRepo: catalogBuildRepo(build),
    sizeGb: build.sizeGb,
    source: build.source,
    slug: catalogBuildSlug(build),
    installed: false,
    present: false,
    heldBy: null,
    blockedByForeignDir: false,
    blocker: null,
    localRevision: null,
    localAssetsRepo: null,
    localAssetsRevision: null,
    located: null,
    listed: false,
  })),
}));

/** The folder of the build every surface chooses first. */
function slugOf(item: CatalogItem): string {
  return item.builds[0].slug;
}

/** `item` with its first build on disk or not. */
function withPresent(item: CatalogItem, present: boolean): CatalogItem {
  const [first, ...rest] = item.builds;
  return { ...item, builds: [{ ...first, present }, ...rest] };
}
const visible = items.filter((item) => !item.hidden && item.category !== 'systemone');
/** The choices on the 64 GiB Mac of {@link routes}: the entries with a 96 GiB floor are left out. */
const unfloored = visible.filter((item) => item.minMemoryGb === undefined);
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
    '/models': {
      models: [],
      companions: [],
      warnings: [],
      dir,
    },
    '/host': { totalMemBytes: 64 * 1024 ** 3, diskFreeBytes: null },
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

function hasButton(label: string): boolean {
  return [...page!.container.querySelectorAll('button')].some((el) => el.textContent?.trim() === label);
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
    expect(page.container.querySelectorAll('input[type=radio]')).toHaveLength(unfloored.length);
    const selected = page.container.querySelector<HTMLInputElement>('input:checked');
    expect(selected?.value).toBe(visible.find((item) => item.isDefault)!.label);
    for (const item of items.filter((item) => item.hidden)) expect(page.text()).not.toContain(item.label);
  });

  it.each([
    [36, false],
    [64, false],
    [96, true],
    [128, true],
  ])('on a %i GiB Mac offers Qwen3.8-Flash-Next and Hy4-flash: %s', async (gib, offeredFloored) => {
    const floored = visible.filter((item) => item.minMemoryGb !== undefined);
    expect(floored.map((item) => item.label).sort()).toEqual(['Hy4-flash', 'Qwen3.8-Flash-Next']);
    const mac = routes(`/test/mac-${gib}`);
    dispose = stubApi({
      ...mac,
      '/host': { totalMemBytes: gib * 1024 ** 3, diskFreeBytes: null },
    });
    page = await renderPage(createElement(App), (text) => text.includes('Download model'));
    const offered = [...page.container.querySelectorAll<HTMLInputElement>('input[type=radio]')].map((el) => el.value);
    expect(offered).toEqual((offeredFloored ? visible : unfloored).map((item) => item.label));
    if (offeredFloored) expect(page.text()).not.toContain("larger than this Mac's memory");
  });

  it('opens the Models page, not onboarding, when the models folder is a link to a missing disk', async () => {
    const sentence =
      'The models folder /Users/x/.mlx-node/models is a link to /Volumes/Disk/models, which is not available. Connect that disk, or change the link, then try again.';
    dispose = stubApi({
      ...routes(),
      '/models': stubError('E_UNAVAILABLE', sentence, 'models-dir-unavailable'),
    });
    page = await renderPage(createElement(App), (text) => text.includes(sentence));
    // Mutation: drop the reason branch from the gate → it stays on the Overview → red.
    expect(window.location.pathname).toBe('/models');
    expect(page.text()).not.toContain('No local models yet');
    expect(page.text()).not.toContain('Download model');
  });

  it('keeps any other model list failure on the Overview', async () => {
    dispose = stubApi({ ...routes(), '/models': stubError('E_UNAVAILABLE', 'Dashboard database is unavailable') });
    page = await renderPage(createElement(App), (text) => text.includes('Overview'));
    expect(window.location.pathname).toBe('/');
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
          ? {
              '/downloads': {
                jobs: [
                  { id: 'ongoing', slug: slugOf(visible[0]), repo: visible[0].builds[0].hfRepo, state: 'running' },
                ],
              },
            }
          : {}),
      });
      page = await renderPage(createElement(App), (text) => text.includes('Choose your first model'));
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
    expect(splitWhitespace(home.className)).toEqual(expect.arrayContaining(['font-heading', 'text-strong']));
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

  it('offers one choice per model and names the build of a model with more builds in the dock', async () => {
    dispose = stubApi(routes('/test/builds'));
    await mount();
    const radios = [...page!.container.querySelectorAll<HTMLInputElement>('input[type=radio]')];
    expect(radios.map((radio) => radio.value)).toEqual(unfloored.map((item) => item.label));
    const qwen = visible.find((item) => item.isDefault === true)!;
    expect(qwen.builds.length).toBeGreaterThan(1);
    expect(page!.text()).toContain(`Build ${qwen.builds[0].id}. More builds are on the Models page.`);
    const single = unfloored.find((item) => item.builds.length === 1);
    if (single !== undefined) {
      await act(async () => page!.container.querySelector<HTMLInputElement>(`input[value="${single.label}"]`)!.click());
      expect(page!.text()).not.toContain('More builds are on the Models page.');
    }
  });

  it('stands a model for its build on disk when that is not its first build', async () => {
    const inkling = visible.find((item) => item.label === 'Inkling-2-Mini')!;
    const [first, second, ...rest] = inkling.builds;
    const onDisk: CatalogItem = { ...inkling, builds: [first, { ...second, present: true }, ...rest] };
    dispose = stubApi({
      ...routes('/test/second-build'),
      '/catalog': { items: items.map((item) => (item.label === inkling.label ? onDisk : item)) },
    });
    await mount();
    const choice = page!.container.querySelector<HTMLInputElement>(`input[value="${inkling.label}"]`)!;
    const card = choice.closest('label')!;
    // Mutation: Onboarding reads builds[0] → the first build's size and "One-time download" → red.
    expect(card.textContent).toContain(`~${second.sizeGb} GB · Downloaded`);
    expect(card.textContent).toContain('On this Mac');
  });

  it.each<[string, Partial<CatalogBuildItem>, string]>([
    ['held', { heldBy: 'UD-Q4_K_M' }, 'Holds UD-Q4_K_M'],
    [
      'locked',
      {
        blockedByForeignDir: true,
        blocker: {
          kind: 'directory',
          fileCount: 1,
          sizeBytes: 10,
          truncated: false,
          names: ['a.txt'],
          listed: false,
          modelLike: false,
          holdsModel: false,
          linkTarget: null,
          dangling: false,
          linkToFile: false,
          locked: true,
        },
      },
      'No access',
    ],
    [
      'needs-update',
      {
        blockedByForeignDir: true,
        blocker: {
          kind: 'directory',
          fileCount: 2,
          sizeBytes: 10,
          truncated: false,
          names: ['a.gguf', 'config.json'],
          listed: false,
          modelLike: true,
          holdsModel: true,
          linkTarget: null,
          dangling: false,
          linkToFile: false,
          locked: false,
        },
      },
      'Folder needs update',
    ],
  ])('names a %s folder for what it is, not as one that needs cleanup', async (_name, change, word) => {
    const qwen = visible.find((item) => item.isDefault === true)!;
    const [first, ...rest] = qwen.builds;
    const blockedEntry: CatalogItem = { ...qwen, builds: [{ ...first, ...change }, ...rest] };
    dispose = stubApi({
      ...routes(`/test/blocked-${_name}`),
      '/catalog': { items: items.map((item) => (item.label === qwen.label ? blockedEntry : item)) },
    });
    await mount();
    const card = page!.container.querySelector<HTMLInputElement>(`input[value="${qwen.label}"]`)!.closest('label')!;
    expect(card.textContent).toContain(`~${first.sizeGb} GB · ${word}`);
    expect(card.textContent).not.toContain('needs cleanup');
  });

  it('continues from a ready model to coding agent setup and remembers completion', async () => {
    const dir = '/test/completed-onboarding';
    window.history.replaceState({}, '', '/welcome');
    dispose = stubApi({
      ...routes(dir),
      '/catalog': { items: items.map((item) => withPresent(item, item.isDefault === true)) },
    });
    page = await renderPage(createElement(App), (text) => text.includes('Model ready'));
    // True for a download and for Locate alike.
    expect(page.text()).toContain('The model is ready.');
    await act(async () => button('Set up coding agents').click());
    await waitFor(() => expect(window.location.pathname).toBe('/coding-agents'));
    expect(localStorage.getItem(`mlx-node:onboarding:v1:${dir}`)).toBe('done');
    expect(page.container.querySelector('aside')).not.toBeNull();
  });

  it('downloads the chosen model and waits for refreshed install state before continuing', async () => {
    const chosen = visible[1];
    const calls: ApiCall[] = [];
    let listener: ((event: DownloadEvent) => void) | undefined;
    const refreshed = deferred({ items: items.map((item) => withPresent(item, item.label === chosen.label)) });
    dispose = stubApi(
      {
        ...routes(),
        '/catalog': sequence({ items }, refreshed.body),
        '/downloads': sequence({ jobs: [] }, { id: 'chosen-job', slug: slugOf(chosen), repo: chosen.builds[0].hfRepo }),
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
    const radio = page!.container.querySelector<HTMLInputElement>(`input[value="${chosen.label}"]`)!;
    await act(async () => radio.click());
    expect(radio.checked).toBe(true);
    await act(async () => button('Download model').click());
    await waitFor(() => expect(listener).toBeDefined());
    expect(calls.find((call) => call.method === 'POST')?.body).toEqual({ slug: slugOf(chosen) });
    await act(async () => listener!({ type: 'done', id: 'chosen-job', outputDir: `/models/${slugOf(chosen)}` }));
    await waitFor(() => expect(page!.text()).toContain('Checking download…'));
    expect(page!.text()).not.toContain('Model ready');
    expect(hasButton('Set up coding agents')).toBe(false);
    await act(async () => refreshed.release());
    await waitFor(() => expect(hasButton('Set up coding agents')).toBe(true));
    expect(page!.text()).toContain('Model ready');
    expect(page!.container.querySelector<HTMLInputElement>('input:checked')?.value).toBe(chosen.label);
  });

  it('restores an existing non-default download and recovers after cancellation', async () => {
    const chosen = visible[1];
    let listener: ((event: DownloadEvent) => void) | undefined;
    dispose = stubApi(
      {
        ...routes(),
        '/downloads': {
          jobs: [
            {
              id: 'existing-job',
              slug: slugOf(chosen),
              repo: chosen.builds[0].hfRepo,
              state: 'running',
              receivedBytes: 10,
              totalBytes: 100,
            },
          ],
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
      text.includes('Getting the model'),
    );
    await waitFor(() => expect(listener).toBeDefined());
    expect(page.text()).toContain('Keep the app open. You can look around while it finishes.');
    expect(page.container.querySelector<HTMLInputElement>('input:checked')?.value).toBe(chosen.label);
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
        '/downloads': {
          jobs: [{ id: 'failed-job', slug: slugOf(visible[0]), repo: visible[0].builds[0].hfRepo, state: 'running' }],
        },
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
      text.includes('Getting the model'),
    );
    await waitFor(() => expect(listener).toBeDefined());
    await act(async () => listener!({ type: 'error', id: 'failed-job', message: 'Connection lost' }));
    await waitFor(() => expect(page!.text()).toContain('Download model'));
    expect(page.container.querySelector('[role=alert]')?.textContent).toContain('Connection lost');
    expect(button('Download model').disabled).toBe(false);

    await act(async () =>
      page!.container.querySelector<HTMLInputElement>(`input[value="${visible[1].label}"]`)!.click(),
    );
    expect(page.container.querySelector('[role=alert]')).toBeNull();
    expect(button('Download model').disabled).toBe(false);
    await act(async () =>
      page!.container.querySelector<HTMLInputElement>(`input[value="${visible[0].label}"]`)!.click(),
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
    const other = visible.find((item) => item.label !== requested)!;
    await act(async () => button('Download model').click());
    await waitFor(() => expect(calls.some((call) => call.method === 'POST')).toBe(true));
    await act(async () => page!.container.querySelector<HTMLInputElement>(`input[value="${other.label}"]`)!.click());
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
