/**
 * @vitest-environment happy-dom
 */

/**
 * "Open" on a System One card of the Models page: the link names the card's folder, and the System One page selects
 * the model in that folder and the mode of its kind.
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { useHost } from '@/lib/host';
import { requestedModel } from '@/lib/system-one';
import type {
  CatalogResponse,
  CatalogUpdatesResponse,
  ModelsResponse,
  SystemOneModelInfo,
  SystemOneState,
} from '@/lib/types';
import { useJson } from '@/lib/use-api';
import Models from '@/pages/models';
import { useModelDownloads } from '@/pages/models/use-model-downloads';
import SystemOne from '@/pages/system-one';
import { systemOneModelBases } from '@/pages/system-one/model-options';
import { NoSystemOneModelsCard, SystemOneModelsDrawer } from '@/pages/system-one/models-drawer';
import { createElement, type ReactNode } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { catalogWithState } from '../src/catalog.js';
import { discoverLocalModels } from '../src/models.js';
import { ggufHeader } from './helpers/configless-gguf-fixtures.js';
import { deferred, renderPage, type RenderedPage, stubApi, waitFor } from './render.js';

let root: string;
let modelsDir: string;
let mounted: RenderedPage | undefined;
let restoreApi: (() => void) | undefined;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'dash-system-one-open-'));
  modelsDir = join(root, 'models');
  mkdirSync(modelsDir);
});

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
  rmSync(root, { recursive: true, force: true });
});

/** The System One models the core lists for the four catalog folders: GGUF ids are file names. */
function systemOneModels(): SystemOneModelInfo[] {
  return [
    systemOneModel(join(modelsDir, 'clairic-flash-gguf'), 'Clairic-Flash-UD-Q4_K_XL.gguf', 'UD-Q4_K_XL', 'clairic'),
    systemOneModel(join(modelsDir, 'clairic-gguf'), 'Clairic-UD-Q4_K_XL.gguf', 'UD-Q4_K_XL', 'clairic'),
    // A CLI folder with two builds of CLEF Flash.
    systemOneModel(join(modelsDir, 'clef-flash-gguf'), 'Clef-Flash-Q4_K_M.gguf', 'Q4_K_M', 'clef'),
    systemOneModel(join(modelsDir, 'clef-flash-gguf'), 'Clef-Flash-Q8_0.gguf', 'Q8_0', 'clef'),
    systemOneModel(join(modelsDir, 'clef-gguf-q4_k_m'), 'Clef-Q4_K_M.gguf', 'Q4_K_M', 'clef'),
  ];
}

function systemOneModel(
  folder: string,
  file: string,
  variant: string,
  modelType: 'clairic' | 'clef' = 'clef',
): SystemOneModelInfo {
  return {
    id: file.slice(0, -'.gguf'.length),
    name: file.slice(0, -'.gguf'.length),
    modelType,
    format: 'gguf',
    variant,
    supportsImages: modelType === 'clairic',
    contextWindow: modelType === 'clairic' ? 65536 : 16384,
    resident: false,
    path: join(folder, file),
  };
}

function routes(): Record<string, unknown> {
  const state: SystemOneState = { sidecarRunning: false, models: systemOneModels(), activeJobId: null, history: [] };
  return {
    '/systemone/state': state,
    '/models': { ...discoverLocalModels(modelsDir), dir: modelsDir },
    '/host': { totalMemBytes: 128 * 1024 ** 3, diskFreeBytes: null },
    '/catalog': { items: catalogWithState(modelsDir) },
    '/catalog/updates': { items: [] },
    '/downloads': { jobs: [] },
  };
}

/** The `href` of the Open link on the System One card of `slug`'s entry, as the Models page renders it. */
async function openLink(slug: string, chosen: string): Promise<string> {
  restoreApi = stubApi(routes());
  mounted = await renderPage(createElement(MemoryRouter, null, createElement(Models)), (text) =>
    text.includes('System One models'),
  );
  await waitFor(() => mounted!.container.querySelector(`#system-one-models [data-model="${slug}"]`) !== null, 3000);
  const card = mounted.container.querySelector<HTMLElement>(`#system-one-models [data-model="${slug}"]`)!;
  if (chosen !== slug) {
    const option = card.querySelector<HTMLButtonElement>(`[data-value="${chosen}"]`)!;
    option.click();
    await waitFor(() => card.getAttribute('data-build') === chosen);
  }
  await waitFor(() => card.querySelector('a[href^="/system-one"]') !== null);
  const href = card.querySelector('a[href^="/system-one"]')!.getAttribute('href')!;
  mounted.unmount();
  mounted = undefined;
  restoreApi();
  restoreApi = undefined;
  return href;
}

/** The System One page opened at `href`: the pressed mode and the id of the model the picker shows. */
async function openAt(href: string): Promise<{ mode: string | null; model: string | null }> {
  restoreApi = stubApi(routes());
  mounted = await renderPage(
    createElement(MemoryRouter, { initialEntries: [href] }, createElement(SystemOne)),
    (text) => text.includes('Try an example'),
  );
  const picked = (): string | null =>
    mounted!.container.querySelector('button[aria-haspopup="menu"]')?.getAttribute('title') ?? null;
  await waitFor(() => picked() !== null);
  const mode = mounted.container
    .querySelector('[aria-label="System One mode"] [aria-pressed="true"]')
    ?.getAttribute('data-value');
  return { mode: mode ?? null, model: picked() };
}

function plant(folder: string, files: Record<string, Buffer | string>): void {
  mkdirSync(join(modelsDir, folder));
  for (const [name, bytes] of Object.entries(files)) writeFileSync(join(modelsDir, folder, name), bytes);
}

describe('Open on a System One card', () => {
  // With no match the page keeps the first Clairic model in Extract.
  it.each<[string, string, string, string, string, 'extract' | 'decide']>([
    ['Clairic (27B)', 'clairic-gguf', 'clairic-gguf', 'UD-Q4_K_XL', 'Clairic-UD-Q4_K_XL', 'extract'],
    // The folder holds two builds: the card's build picks its file.
    ['CLEF Flash (9B)', 'clef-flash-gguf', 'clef-flash-gguf', 'Q8_0', 'Clef-Flash-Q8_0', 'decide'],
    // The second build of a CLEF entry, in the folder with the suffix.
    ['CLEF (27B)', 'clef-gguf', 'clef-gguf-q4_k_m', 'Q4_K_M', 'Clef-Q4_K_M', 'decide'],
  ])('%s opens its model in its mode', async (_label, entrySlug, buildSlug, build, id, mode) => {
    plant('clairic-flash-gguf', { 'config.json': '{}', 'Clairic-Flash-UD-Q4_K_XL.gguf': 'w' });
    plant('clairic-gguf', { 'config.json': '{}', 'Clairic-UD-Q4_K_XL.gguf': 'w' });
    plant('clef-flash-gguf', {
      'Clef-Flash-Q4_K_M.gguf': ggufHeader('clef'),
      'Clef-Flash-Q8_0.gguf': ggufHeader('clef'),
    });
    plant('clef-gguf-q4_k_m', { 'Clef-Q4_K_M.gguf': ggufHeader('clef') });
    const href = await openLink(entrySlug, buildSlug);
    expect(href).toBe(`/system-one?model=${buildSlug}&build=${build}`);
    expect(await openAt(href)).toEqual({ mode, model: id });
  });

  it('matches a model id before a folder name, and a folder only with the build of its file', () => {
    const models = [
      systemOneModel('/m/Clef-Q4_K_M', 'A-model.gguf', 'Q4_K_M'),
      systemOneModel('/m/clef-gguf-q4_k_m', 'Clef-Q4_K_M.gguf', 'Q4_K_M'),
      systemOneModel('/m/clef-flash-gguf', 'Clef-Flash-BF16.gguf', 'BF16'),
      systemOneModel('/m/clef-flash-gguf', 'Clef-Flash-Q8_0.gguf', 'Q8_0'),
      { ...systemOneModel('/m', 'Clairic-Flash-4bit.gguf', '4bit', 'clairic'), path: '/m/clairic-flash' },
    ];
    const pick = (model: string, build: string | null): string | undefined => requestedModel(models, model, build)?.id;
    expect(pick('Clef-Q4_K_M', 'Q4_K_M')).toBe('Clef-Q4_K_M');
    expect(pick('Clef-Q4_K_M', null)).toBe('Clef-Q4_K_M');
    expect(pick('clef-flash-gguf', 'Q8_0')).toBe('Clef-Flash-Q8_0');
    expect(pick('clef-flash-gguf', 'BF16')).toBe('Clef-Flash-BF16');
    expect(pick('clef-flash-gguf', 'Q4_K_M')).toBeUndefined();
    expect(pick('clef-flash-gguf', null)).toBeUndefined();
    // A folder model's path is the folder.
    expect(pick('clairic-flash', '4bit')).toBe('Clairic-Flash-4bit');
  });
});

describe('the System One drawer and the empty card while they load', () => {
  function Both(): ReactNode {
    const models = useJson<ModelsResponse>('/models');
    const catalog = useJson<CatalogResponse>('/catalog');
    const host = useHost();
    const updates = useJson<CatalogUpdatesResponse>('/catalog/updates');
    const downloads = useModelDownloads(models, catalog, updates, host);
    return createElement(
      'div',
      null,
      createElement('div', { id: 'empty-card' }, createElement(NoSystemOneModelsCard, { catalog, host, downloads })),
      createElement(SystemOneModelsDrawer, { open: true, onOpenChange: () => {}, catalog, host, updates, downloads }),
    );
  }

  function placeholders(): { card: number; drawer: number } {
    const card = document.querySelectorAll('#empty-card [aria-label="Loading recommended model"]').length;
    return { card, drawer: document.querySelectorAll('[aria-label="Loading recommended model"]').length - card };
  }

  it('shows one placeholder per System One entry once the catalog is there, and names both kinds', async () => {
    const held = deferred({ totalMemBytes: 128 * 1024 ** 3, diskFreeBytes: null });
    restoreApi = stubApi({ ...routes(), '/host': held.body });
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Both)), (text) =>
      text.includes('Install a System One model to get started'),
    );
    await waitFor(() => placeholders().card === 4);
    expect(placeholders()).toEqual({ card: 4, drawer: 4 });
    const text = document.body.textContent ?? '';
    expect(text).toContain('Download Clairic models for extraction and CLEF models for decisions to this Mac.');
    expect(text).toContain('CLEF models answer yes/no, choice and score questions.');
    held.release();
    await waitFor(() => document.querySelectorAll('#empty-card [data-model]').length === 4);
    expect(
      [...document.querySelectorAll('#empty-card [data-model]')].map((card) => card.getAttribute('data-model')),
    ).toEqual(['clairic-flash-gguf', 'clairic-gguf', 'clef-flash-gguf', 'clef-gguf']);
  });

  it('shows two placeholders before the catalog answers', async () => {
    const held = deferred({ items: catalogWithState(modelsDir) });
    restoreApi = stubApi({ ...routes(), '/catalog': held.body });
    mounted = await renderPage(createElement(MemoryRouter, null, createElement(Both)), (text) =>
      text.includes('Install a System One model to get started'),
    );
    expect(placeholders()).toEqual({ card: 2, drawer: 2 });
    held.release();
    await waitFor(() => document.querySelectorAll('#empty-card [data-model]').length === 4);
  });
});

describe('systemOneModelBases catalog slug matching', () => {
  it('matches System One catalog metadata via parent and grandparent GGUF directory slugs from model.path', () => {
    const catalog = catalogWithState(modelsDir);
    const models: SystemOneModelInfo[] = [
      // 1-level GGUF folder slug (id/name do not match catalog label or slug).
      {
        id: 'alias-clef-q4',
        name: 'AliasOne',
        modelType: 'clef',
        format: 'gguf',
        variant: 'Q4_K_M',
        supportsImages: false,
        contextWindow: 16384,
        resident: false,
        path: '/models/clef-gguf-q4_k_m/alias-clef-q4.gguf',
      },
      // 2-level nested GGUF folder slug (Windows-style backslashes).
      {
        id: 'alias-clairic-split',
        name: 'AliasTwo',
        modelType: 'clairic',
        format: 'gguf',
        variant: 'UD-Q4_K_XL',
        supportsImages: true,
        contextWindow: 65536,
        resident: false,
        path: 'C:\\models\\clairic-gguf\\UD-Q4_K_XL\\alias-00001-of-00002.gguf',
      },
      // Non-catalog local model should remain non-recommended.
      {
        id: 'custom-local',
        name: 'CustomLocal',
        modelType: 'clef',
        format: 'gguf',
        variant: 'Q4_K_M',
        supportsImages: false,
        contextWindow: 16384,
        resident: false,
        path: '/models/custom-lab/custom-local.gguf',
      },
    ];

    const bases = systemOneModelBases(models, null, false, catalog);
    const byBase = new Map(bases.map((b) => [b.base, b]));

    expect(byBase.get('AliasOne')?.recommended).toBe(true);
    expect(byBase.get('AliasOne')?.note).toBeTruthy();
    expect(byBase.get('AliasTwo')?.recommended).toBe(true);
    expect(byBase.get('AliasTwo')?.note).toBeTruthy();
    expect(byBase.get('CustomLocal')?.recommended).toBeUndefined();
    expect(byBase.get('CustomLocal')?.note).toBeUndefined();
  });
});
