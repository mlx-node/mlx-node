/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { KeyValue } from '../ui/src/components/arc-x/key-value/key-value.js';
import { PathText } from '../ui/src/components/arc-x/path-text/path-text.js';
import { tildePath } from '../ui/src/lib/paths.js';
import type { SessionRow } from '../ui/src/lib/types.js';
import { sessionCommands } from '../ui/src/shell/commands.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { stubApi, waitFor, type StubbedApi } from './render.js';

const HOME = '/Users/ann';

describe('tildePath', () => {
  it('writes the home folder itself as ~', () => {
    expect(tildePath('/Users/ann', HOME)).toBe('~');
  });

  it('writes a path under the home folder as ~/…', () => {
    expect(tildePath('/Users/ann/workspace/x', HOME)).toBe('~/workspace/x');
    expect(tildePath('/Users/ann/workspace/x/', HOME)).toBe('~/workspace/x/');
  });

  it('ignores a trailing slash on the home folder', () => {
    expect(tildePath('/Users/ann/workspace', '/Users/ann/')).toBe('~/workspace');
    expect(tildePath('/Users/ann', '/Users/ann//')).toBe('~');
  });

  it('leaves a sibling folder that only shares the prefix alone', () => {
    expect(tildePath('/Users/ann2/x', HOME)).toBe('/Users/ann2/x');
    expect(tildePath('/Users/an', HOME)).toBe('/Users/an');
  });

  it('leaves every path alone without a usable home folder', () => {
    expect(tildePath('/Users/ann/x', null)).toBe('/Users/ann/x');
    expect(tildePath('/Users/ann/x', '')).toBe('/Users/ann/x');
    expect(tildePath('/Users/ann/x', 'Users/ann')).toBe('/Users/ann/x');
    expect(tildePath('/Users/ann/x', '/')).toBe('/Users/ann/x');
    expect(tildePath('/tmp/x', HOME)).toBe('/tmp/x');
  });
});

let mounted: { root: Root; container: HTMLElement; api: StubbedApi } | undefined;

async function mount(element: ReactElement, routes: Record<string, unknown>): Promise<HTMLElement> {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const api = stubApi(routes);
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => root.render(element));
  mounted = { root, container, api };
  return container;
}

afterEach(() => {
  if (mounted === undefined) return;
  act(() => mounted!.root.unmount());
  mounted.container.remove();
  mounted.api();
  mounted = undefined;
});

const HEALTH = { status: 'ok', modelsDir: '', tracesDir: '', runtimeDir: '', homeDir: HOME };

describe('PathText', () => {
  it('shows the home folder as ~ and keeps the full path as the hover title', async () => {
    const container = await mount(createElement(PathText, { path: '/Users/ann/workspace/mlx-node' }), {
      '/health': HEALTH,
    });
    await waitFor(() => container.textContent === '~/workspace/mlx-node');
    expect(container.querySelector('span')?.getAttribute('title')).toBe('/Users/ann/workspace/mlx-node');
  });

  it('cuts in the middle after the home folder became ~', async () => {
    const path = '/Users/ann/workspace/github/mlx-node/packages';
    const container = await mount(createElement(PathText, { path, maxChars: 24 }), { '/health': HEALTH });
    await waitFor(() => (container.textContent ?? '').startsWith('~/'));
    expect(container.textContent).toBe('~/workspace/…de/packages');
    expect(container.textContent?.length).toBe(24);
    expect(container.querySelector('span')?.getAttribute('title')).toBe(path);
  });

  it('shows the full path while the home folder is unknown', async () => {
    const container = await mount(createElement(PathText, { path: '/Users/ann/workspace/mlx-node' }), {});
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
    expect(container.textContent).toBe('/Users/ann/workspace/mlx-node');
  });
});

describe('KeyValue copy button', () => {
  it('marks a copyable value so its button keeps the text line', async () => {
    const container = await mount(
      createElement(KeyValue, {
        items: [
          { key: 'cwd', label: 'Directory', value: 'x', mono: true, copy: '/Users/ann/x' },
          { key: 'branch', label: 'Branch', value: 'main' },
        ],
      }),
      {},
    );
    const values = [...container.querySelectorAll('dd')];
    expect(values[0]?.hasAttribute('data-copy')).toBe(true);
    expect(values[1]?.hasAttribute('data-copy')).toBe(false);
  });

  it('keeps the button beside the text in both layouts, the text giving way (the rules are read)', () => {
    const rules = cssRules('components/arc-x/key-value/key-value.module.css');
    expect(cssValue(rules, ".root[data-layout='grid'] .value[data-copy] .slot", 'flex-wrap')).toBe('nowrap');
    expect(cssValue(rules, '.copy', 'flex')).toBe('none');
    expect(cssValue(rules, '.text', 'min-width')).toBe('0');
    expect(cssValue(rules, '.text', 'text-overflow')).toBe('ellipsis');
    expect(cssValue(rules, ".root[data-layout='rows'] > .row", 'grid-template-columns')).toBe('subgrid');
  });

  it('never breaks a path over two lines: one line, cut at the end when the cell is too narrow (the rules are read)', () => {
    const rules = cssRules('components/arc-x/path-text/path-text.module.css');
    expect(cssValue(rules, '.root', 'white-space')).toBe('nowrap');
    expect(cssValue(rules, '.root', 'text-overflow')).toBe('ellipsis');
    expect(cssValue(rules, '.root', 'overflow')).toBe('hidden');
    expect(cssValue(rules, '.root', 'max-width')).toBe('100%');
  });
});

describe('palette session subtitle', () => {
  const session = (cwd: string): SessionRow =>
    ({
      id: 's1',
      title: 'A session',
      cwd,
      branch: null,
      model: null,
      live: false,
      lastActiveAt: new Date().toISOString(),
    }) as SessionRow;

  it('names a session run in the home folder itself ~, and keeps the full path searchable', () => {
    const [home] = sessionCommands([session(HOME)], () => {}, HOME);
    expect(home?.description?.startsWith('~ · ')).toBe(true);
    expect(home?.keywords).toContain(HOME);
    const [below] = sessionCommands([session('/Users/ann/workspace/x')], () => {}, HOME);
    expect(below?.description?.startsWith('x · ')).toBe(true);
    const [unknown] = sessionCommands([session(HOME)], () => {}, null);
    expect(unknown?.description?.startsWith('ann · ')).toBe(true);
  });
});
