/**
 * @vitest-environment happy-dom
 */

/**
 * Shape guards for the LOADING tree of the two list pages.
 *
 * Both pages used to answer `loading` with a bare stack of grey bars and then
 * swap in a `<Table>` that carries a `<TableHeader>` the bars never reserved:
 * the header appeared out of nowhere, pushed every row down by its own height,
 * and the column grid re-solved underneath. That reads as a flash of movement on
 * every navigation, and no assertion in `pages.test.ts` could see it — every
 * test there waits for the loaded tree precisely so it never reads a skeleton.
 *
 * So these tests read the tree the others skip. `renderPage` commits the first
 * render before any reply can reach it, so the page it hands back for a `() =>
 * true` condition is the loading one; the page is then walked forward on the
 * same root and the two trees compared.
 *
 * happy-dom performs no layout — `getBoundingClientRect()` answers 0 for
 * everything — so a pixel assertion here would be worth nothing. Structure is
 * what these check: the same column headers, the same cells per row, the same
 * card wrappers. That is what a box you cannot measure is made of.
 *
 * `happy-dom` is scoped to this file by the docblock above; the repo default
 * environment stays `node`.
 */

import type {
  CatalogResponse,
  DownloadsResponse,
  LocalModel,
  ModelsResponse,
  SessionRow,
  SessionsResponse,
} from '@/lib/types';
import Models from '@/pages/models';
import Sessions from '@/pages/sessions';
import { act, createElement, type ReactElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { renderPage, type RenderedPage, stubApi } from './render.js';

let mounted: RenderedPage | undefined;
let restoreApi: (() => void) | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
});

/** How long {@link walkToLoaded} waits before failing with what it did render. */
const SETTLE_TIMEOUT_MS = 2_000;

/**
 * Drive an already-mounted page forward until `loadedWhen` appears in its text.
 *
 * `renderPage` hands back exactly one tree and these tests need two, so the
 * first comes from the mount itself and the walk to the second happens here, on
 * the same root — the container is mutated in place, which is why every
 * comparison below is taken as plain strings and counts rather than as node
 * references that would silently follow the DOM forward.
 */
async function walkToLoaded(page: RenderedPage, loadedWhen: (text: string) => boolean): Promise<void> {
  const deadline = Date.now() + SETTLE_TIMEOUT_MS;
  while (!loadedWhen(page.text())) {
    if (Date.now() > deadline) {
      throw new Error(`page did not reach its loaded state.\nLast rendered text: ${page.text() || '(empty)'}`);
    }
    // A macrotask turn: the stub answers over a real MessagePort, so the reply
    // lands on the task queue and a microtask drain alone would spin forever.
    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  }
}

/**
 * Mount `element` against `routes` and take a shape snapshot either side of the
 * response.
 *
 * The `() => true` settle condition is deliberate, and the one place in the
 * suite where `renderPage`'s "name a string only the loaded page renders" rule
 * is inverted on purpose: the tree wanted here is the one committed BEFORE the
 * reply, and no positive string can name it (a skeleton renders no text at all).
 * What replaces that safety is the caller's own check that the snapshot really
 * is a loading one — {@link ShapeSnapshot.skeletons} is asserted non-zero by
 * every test below, so a mount that had already settled fails instead of quietly
 * comparing the loaded tree against itself.
 */
async function shapesAcrossLoad(
  element: ReactElement,
  routes: Record<string, unknown>,
  loadedWhen: (text: string) => boolean,
): Promise<{ loading: ShapeSnapshot; loaded: ShapeSnapshot }> {
  restoreApi = stubApi(routes);
  mounted = await renderPage(element, () => true);
  const loading = shapeOf(mounted.container);
  await walkToLoaded(mounted, loadedWhen);
  return { loading, loaded: shapeOf(mounted.container) };
}

/** Everything about a rendered list that survives having no layout engine. */
interface ShapeSnapshot {
  /** Column header texts, in document order — `''` for an unlabelled column. */
  heads: string[];
  /** Cells per body row: one entry per row, so a short row is visible. */
  rowWidths: number[];
  /** Cards inside the recommended-models grid; `-1` when the grid is absent. */
  catalogCards: number;
  /** Placeholders on the page — zero means this is not a loading tree. */
  skeletons: number;
}

function shapeOf(container: HTMLElement): ShapeSnapshot {
  return {
    heads: [...container.querySelectorAll('thead th')].map((th) => (th.textContent ?? '').trim()),
    rowWidths: [...container.querySelectorAll('tbody tr')].map((tr) => tr.querySelectorAll('td').length),
    catalogCards: catalogCardCount(container),
    skeletons: container.querySelectorAll('[data-slot="skeleton"]').length,
  };
}

/**
 * Cards in the recommended-models grid, found by the grid's own column class.
 *
 * `-1` rather than `0` when the grid is missing: two zeroes compare equal, and a
 * grid that stopped rendering would then read as a match.
 */
function catalogCardCount(container: HTMLElement): number {
  const grid = [...container.querySelectorAll('div')].find((div) => div.className.includes('xl:grid-cols-3'));
  return grid === undefined ? -1 : grid.querySelectorAll('[data-slot="card"]').length;
}

function localModel(overrides: Partial<LocalModel>): LocalModel {
  return {
    name: 'qwen3-8b',
    path: '/models/qwen3-8b',
    modelType: 'qwen3',
    quant: '4bit',
    contextWindow: 262_144,
    sizeBytes: 4_800_000_000,
    fileCount: 7,
    ...overrides,
  };
}

/** The three routes the Models page reads, with a two-model directory. */
function modelsRoutes(): Record<string, unknown> {
  const models: ModelsResponse = {
    models: [localModel({}), localModel({ name: 'gemma-4-12b', modelType: 'gemma4', quant: null })],
    warnings: [],
    dir: '/models',
  };
  // Three visible entries, matching the served catalog — one of them the default.
  const catalog: CatalogResponse = {
    items: [
      {
        label: 'Qwen3.6-27B',
        hfRepo: 'Brooooooklyn/Qwen3.6-27B-NVFP4-mlx',
        sizeGb: 22.2,
        description: 'Best tool use — recommended default',
        slug: 'qwen3.6-27b-nvfp4-mlx',
        isDefault: true,
        installed: false,
        present: false,
        blockedByForeignDir: false,
      },
      {
        label: 'Qwen-AgentWorld-35B',
        hfRepo: 'Brooooooklyn/Qwen-AgentWorld-35B-A3B-nvfp4-mlx',
        sizeGb: 22.7,
        description: 'Agent-tuned MoE, fast decode',
        slug: 'qwen-agentworld-35b-a3b-nvfp4-mlx',
        installed: false,
        present: false,
        blockedByForeignDir: false,
      },
      {
        label: 'Gemma-4-26B-A4B',
        hfRepo: 'Brooooooklyn/Gemma-4-26B-A4B-NVFP4-mlx',
        sizeGb: 18.8,
        description: 'MoE, fast decode',
        slug: 'gemma-4-26b-a4b-nvfp4-mlx',
        installed: false,
        present: false,
        blockedByForeignDir: false,
      },
    ],
  };
  const downloads: DownloadsResponse = { jobs: [] };
  return { '/models': models, '/catalog': catalog, '/downloads': downloads };
}

function sessionRow(overrides: Partial<SessionRow>): SessionRow {
  return {
    id: 'sess-1',
    path: '/sessions/sess-1.jsonl',
    cwd: '/work/mlx-node',
    name: 'Refactor the planner',
    created: 1_700_000_000_000,
    modified: 1_700_000_100_000,
    messageCount: 42,
    firstMessage: 'hello',
    models: ['qwen3-8b'],
    inputTokens: 1_000,
    outputTokens: 2_000,
    ...overrides,
  };
}

function sessionsRoutes(): Record<string, unknown> {
  const sessions: SessionsResponse = {
    sessions: [sessionRow({}), sessionRow({ id: 'sess-2', name: 'Bump MLX' })],
    total: 2,
    tokens: 6_000,
    cwds: ['/work/mlx-node'],
  };
  return { '/sessions': sessions };
}

describe('Models page — the loading tree holds the loaded page’s box', () => {
  /**
   * The regression guard for the whole class. A header rendered by only one of
   * the two branches is 40px that appears the instant the request lands, and it
   * takes every row below it along.
   */
  it('renders the same column headers while loading as it does when loaded', async () => {
    const { loading, loaded } = await shapesAcrossLoad(createElement(Models), modelsRoutes(), (text) =>
      text.includes('qwen3-8b'),
    );
    expect(loading.skeletons).toBeGreaterThan(0);
    expect(loaded.heads).toEqual(['Name', 'Family', 'Quantization', 'Size', 'Context', '']);
    expect(loading.heads).toEqual(loaded.heads);
  });

  it('gives every placeholder row the cells a real row has', async () => {
    // A skeleton row one cell short would let the fixed-width action column
    // collapse and pull the whole grid sideways when the rows arrive.
    const { loading, loaded } = await shapesAcrossLoad(createElement(Models), modelsRoutes(), (text) =>
      text.includes('qwen3-8b'),
    );
    expect(loading.skeletons).toBeGreaterThan(0);
    expect(new Set(loaded.rowWidths)).toEqual(new Set([6]));
    expect(new Set(loading.rowWidths)).toEqual(new Set(loaded.rowWidths));
  });

  /**
   * The catalog grid's own version of the defect: three fixed-height blocks
   * stood in for cards whose height comes from their padding, their title line,
   * their description line and their button. Building the placeholder from the
   * same `Card` wrappers is what makes the two heights the same expression
   * rather than a number someone measured once.
   */
  it('builds the recommended-models placeholders out of real cards', async () => {
    const { loading, loaded } = await shapesAcrossLoad(createElement(Models), modelsRoutes(), (text) =>
      text.includes('Qwen-AgentWorld-35B'),
    );
    expect(loading.skeletons).toBeGreaterThan(0);
    expect(loaded.catalogCards).toBe(3);
    expect(loading.catalogCards).toBe(loaded.catalogCards);
  });
});

describe('Sessions page — the loading tree holds the loaded page’s box', () => {
  function page(): ReactElement {
    return createElement(MemoryRouter, null, createElement(Sessions));
  }

  it('renders the same column headers while loading as it does when loaded', async () => {
    const { loading, loaded } = await shapesAcrossLoad(page(), sessionsRoutes(), (text) =>
      text.includes('Refactor the planner'),
    );
    expect(loading.skeletons).toBeGreaterThan(0);
    expect(loaded.heads).toEqual(['Session', 'Directory', 'Models', 'Modified', 'Messages', 'Tokens', '']);
    expect(loading.heads).toEqual(loaded.heads);
  });

  it('gives every placeholder row the cells a real row has', async () => {
    // This table is `table-fixed` with percentage column widths, so the header
    // is not merely a strip of height — it is what solves the grid. A body of
    // full-width bars under no header is laid out by a different algorithm
    // entirely, and every column moves when the rows replace it.
    const { loading, loaded } = await shapesAcrossLoad(page(), sessionsRoutes(), (text) =>
      text.includes('Refactor the planner'),
    );
    expect(loading.skeletons).toBeGreaterThan(0);
    expect(new Set(loaded.rowWidths)).toEqual(new Set([7]));
    expect(new Set(loading.rowWidths)).toEqual(new Set(loaded.rowWidths));
  });
});
