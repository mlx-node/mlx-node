/**
 * @vitest-environment happy-dom
 */

/**
 * The shell has exactly one scroll container: `<main data-scroll-root>`. The frame is sized to the viewport and
 * refuses to scroll; a second scroller (a document that scrolls around a scrolling `main`) once drew two bars on
 * Metrics. happy-dom does no layout, so the DOM contract is checked here and the CSS rule text beside it.
 */

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import App from '@/App';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { Page, PageHeader } from '../ui/src/shell/page.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { renderPage, stubApi, type RenderedPage } from './render.js';

const ROUTES = {
  '/models': { models: [{ name: 'existing', sizeBytes: 100 }], warnings: [], dir: '/tmp/models' },
  '/downloads': { jobs: [] },
  '/cache': {
    disk: { totalBytes: 0, quotaBytes: 0, blocks: 0, objects: 0 },
    trend: [],
    scope: { root: '/tmp/cache' },
  },
};

const css = readFileSync(join(import.meta.dirname, '../ui/src/shell/app-shell.module.css'), 'utf8');

/** The body of the first rule whose selector is exactly `selector`. */
function rule(selector: string): string {
  for (const chunk of css.split('}')) {
    const open = chunk.indexOf('{');
    if (open === -1) continue;
    if (chunk.slice(0, open).trim() === selector) return chunk.slice(open + 1);
  }
  throw new Error(`no rule ${selector}`);
}

describe('app shell scrolling', () => {
  let dispose: (() => void) | undefined;
  let page: RenderedPage | undefined;

  afterEach(() => {
    page?.unmount();
    page = undefined;
    dispose?.();
    dispose = undefined;
    window.history.replaceState({}, '', '/');
  });

  it('renders one frame, one sidebar and exactly one scroll root, which is MAIN', async () => {
    dispose = stubApi(ROUTES);
    page = await renderPage(createElement(App), (text) => text.includes('Overview'), { scrollRoot: false });
    const { container } = page;
    expect(container.querySelectorAll('[data-shell]')).toHaveLength(1);
    expect(container.querySelector('aside[data-nav]')).not.toBeNull();
    const roots = [...container.querySelectorAll('[data-scroll-root]')];
    expect(roots.map((element) => element.tagName)).toEqual(['MAIN']);
  });

  it('sizes the frame to the viewport, hides its overflow, and scrolls only in main', () => {
    const shell = rule('.shell');
    expect(shell).toContain('height: 100vh');
    expect(shell).toContain('overflow: hidden');
    expect(shell).not.toContain('min-height');
    expect(rule('.main')).toContain('overflow-y: auto');
    const scrollers = css
      .split('}')
      .filter((chunk) => chunk.includes('overflow-y: auto') || chunk.includes('overflow: auto'))
      .map((chunk) => chunk.slice(0, chunk.indexOf('{')).trim());
    expect(scrollers).toEqual(['.main']);
  });
});

describe('sticky page header', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  function mountHeader(sticky: boolean): HTMLElement {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() =>
      root.render(
        createElement(
          MemoryRouter,
          null,
          createElement(
            Page,
            null,
            createElement(PageHeader, { title: 'Metrics', description: 'Throughput.', sticky }),
            createElement('section', null, 'body'),
          ),
        ),
      ),
    );
    const column = container.querySelector<HTMLElement>('[data-page]');
    if (column === null) throw new Error('no page column');
    return column;
  }

  it('the sticky row is a direct child of the page column, so the column (not a short header) bounds it', () => {
    const column = mountHeader(true);
    const row = column.querySelector('[data-sticky]');
    expect(row?.parentElement).toBe(column);
    expect(row?.querySelector('h1')?.textContent).toBe('Metrics');
    expect(row?.textContent).not.toContain('Throughput.');
    expect([...column.children].map((child) => child.tagName)).toEqual(['DIV', 'HEADER', 'P', 'SECTION']);
    const rules = cssRules('shell/page.module.css');
    expect(cssValue(rules, '.titleRow[data-sticky]', 'position')).toBe('sticky');
    expect(row?.previousElementSibling?.getAttribute('aria-hidden')).toBe('true');
    expect(cssValue(rules, '.sentinel', 'position')).toBeUndefined();
    expect(cssValue(rules, '.titleRow[data-sticky]', 'margin-top')).toBe('calc(-1 * var(--page-gap))');
  });

  it('measures the row by its border box, so the stuck padding and rule reach --page-stuck-height', async () => {
    const observed: { target: Element; box: string | undefined }[] = [];
    class RecordingObserver {
      observe(target: Element, options?: { box?: string }): void {
        observed.push({ target, box: options?.box });
      }
      unobserve(): void {}
      disconnect(): void {}
    }
    vi.stubGlobal('ResizeObserver', RecordingObserver);
    try {
      const page = await renderPage(
        createElement(
          MemoryRouter,
          null,
          createElement(Page, null, createElement(PageHeader, { title: 'Metrics', sticky: true })),
        ),
        (text) => text.includes('Metrics'),
      );
      const row = page.container.querySelector('[data-sticky]');
      expect(observed.filter((entry) => entry.target === row).map((entry) => entry.box)).toEqual(['border-box']);
      page.unmount();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('stacks the stuck page row over the stuck table head, and the head over the controls in a row', () => {
    const z = (sheet: string, selector: string): number =>
      Number(cssValue(cssRules(sheet), selector, 'z-index') ?? Number.NaN);
    const table = 'components/arc-x/data-table/data-table.module.css';
    const row = z('shell/page.module.css', '.titleRow[data-sticky]');
    const head = z(table, '.head');
    const control = z(table, ".cell :is(button, a:not(.rowLink), input, [role='button'])");
    expect(control).toBeGreaterThan(0);
    expect(head).toBeGreaterThan(control);
    expect(row).toBeGreaterThan(head);
  });

  it('keeps every z-index inside the page content under the stuck page row; overlays stay above it', () => {
    const row = Number(cssValue(cssRules('shell/page.module.css'), '.titleRow[data-sticky]', 'z-index'));
    const portaled = new Set([
      'components/arc/popover/popover.module.css .content',
      'components/arc/select/select.module.css .content',
      'components/arc/tooltip/tooltip.module.css .tooltip',
      'components/arc-x/confirm-popover/confirm-popover.module.css .content',
    ]);
    const isolatedBy = new Map([
      ['components/arc-x/kpi-drilldown/kpi-drilldown.module.css .veil', '.stage'],
      ['components/arc-x/kpi-drilldown/kpi-drilldown.module.css .panel', '.stage'],
    ]);
    const ui = join(import.meta.dirname, '../ui/src');
    const sheets = readdirSync(ui, { recursive: true, encoding: 'utf8' }).filter((path) => path.endsWith('.css'));
    const above: string[] = [];
    const unread: string[] = [];
    for (const sheet of sheets) {
      const rules = cssRules(sheet);
      let read = 0;
      for (const rule of rules) {
        const value = rule.decls.get('z-index');
        if (value === undefined) continue;
        read += 1;
        const z = Number(value);
        for (const selector of rule.selectors) {
          const key = `${sheet} ${selector}`;
          if (key === 'shell/page.module.css .titleRow[data-sticky]') continue;
          if (rule.decls.get('position') === 'fixed' || portaled.has(key)) {
            expect(z, key).toBeGreaterThan(row);
            continue;
          }
          const stage = isolatedBy.get(key);
          if (stage !== undefined) {
            expect(cssValue(rules, stage, 'isolation'), key).toBe('isolate');
            continue;
          }
          if (!(z < row)) above.push(`${key}: ${value}`);
        }
      }
      if (readFileSync(join(ui, sheet), 'utf8').split('z-index').length - 1 !== read) unread.push(sheet);
    }
    expect(above).toEqual([]);
    expect(unread).toEqual([]);
  });

  it('holds the title to one control height from the top, however tall the controls beside it are', () => {
    const rules = cssRules('shell/page.module.css');
    expect(cssValue(rules, '.titleGroup', 'min-height')).toBe('var(--control-height-md)');
    expect(cssValue(rules, '.titleGroup', 'align-self')).toBe('flex-start');
    expect(cssValue(rules, '.titleGroup', 'align-items')).toBe('center');
  });

  it('drops section actions under the title when both do not fit on one line, and only then', () => {
    const rules = cssRules('shell/page.module.css');
    expect(cssValue(rules, '.sectionHead', 'flex-wrap')).toBe('wrap');
    expect(cssValue(rules, '.sectionHeading', 'flex')).toBe('1 1 auto');
    expect(cssValue(rules, '.sectionDescription', 'contain')).toBe('inline-size');
    expect(cssValue(rules, '.sectionActions', 'margin-left')).toBe('auto');
    expect(cssValue(rules, '.sectionActions', 'justify-content')).toBe('flex-end');
  });

  it('a plain header keeps title and description together in one header', () => {
    const column = mountHeader(false);
    expect(column.querySelector('[data-sticky]')).toBeNull();
    const header = column.querySelector('header');
    expect(header?.parentElement).toBe(column);
    expect(header?.textContent).toContain('Throughput.');
  });
});
