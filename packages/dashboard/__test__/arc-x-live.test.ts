/**
 * @vitest-environment happy-dom
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { CellGrid } from '../ui/src/components/arc-x/cell-grid/cell-grid.js';
import { LiveStream, type LiveStreamPoint } from '../ui/src/components/arc-x/live-stream/live-stream.js';
import {
  LiveTurnList,
  LiveTurnRow,
  type LiveTurnRowProps,
} from '../ui/src/components/arc-x/live-turn-row/live-turn-row.js';
import { ModelBadge } from '../ui/src/components/arc-x/model-badge/model-badge.js';
import { QuantileBar } from '../ui/src/components/arc-x/quantile-bar/quantile-bar.js';
import { ScatterChart, type ScatterPoint } from '../ui/src/components/arc-x/scatter-chart/scatter-chart.js';
import { StatTile } from '../ui/src/components/arc-x/stat-tile/stat-tile.js';
import { formatCount } from '../ui/src/lib/format.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { allByRole, byRole, statValue } from './render.js';

interface Mounted {
  container: HTMLElement;
  rerender(next: ReactElement): void;
}

function mount(element: ReactElement): Mounted {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  return { container, rerender: (next) => act(() => root.render(next)) };
}

afterEach(() => {
  document.body.innerHTML = '';
});

function press(target: Element, key: string): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
  });
}

function tableRows(scope: Element): string[][] {
  const body = scope.querySelector('table tbody');
  return [...(body?.querySelectorAll('tr') ?? [])].map((row) =>
    [...row.querySelectorAll('th, td')].map((cell) => (cell.textContent ?? '').trim()),
  );
}

const NOW = 1_000_000;

function stream(seconds: number, hole?: [number, number]): LiveStreamPoint[] {
  const out: LiveStreamPoint[] = [];
  for (let s = seconds; s >= 0; s -= 1) {
    if (hole !== undefined && s <= hole[0] && s >= hole[1]) continue;
    out.push({ t: NOW - s * 1000, value: 40 + (s % 5) });
  }
  return out;
}

function liveStream(points: readonly LiveStreamPoint[], live = true, nowMs = NOW): ReactElement {
  return createElement(LiveStream, { label: 'Output rate', unit: 'tok/s', points, nowMs, live });
}

describe('LiveStream', () => {
  it('breaks the line where samples are more than 20 s apart', () => {
    const { container } = mount(liveStream(stream(90, [60, 31])));
    const group = byRole(container, 'group', 'Output rate') as HTMLElement;
    const line = group.querySelector('g > path:nth-of-type(2)');
    const moves = (line?.getAttribute('d') ?? '').split('M').length - 1;
    expect(moves).toBe(2);
  });

  it('a hole under 20 s stays one line', () => {
    const { container } = mount(liveStream(stream(90, [40, 31])));
    const group = byRole(container, 'group', 'Output rate') as HTMLElement;
    const line = group.querySelector('g > path:nth-of-type(2)');
    expect((line?.getAttribute('d') ?? '').split('M').length - 1).toBe(1);
  });

  it('reports live, paused and stopped in data-state', () => {
    const view = mount(liveStream(stream(30)));
    const group = (): HTMLElement => byRole(view.container, 'group', 'Output rate') as HTMLElement;
    expect(group().getAttribute('data-state')).toBe('live');
    const pill = byRole(group(), 'button', 'Live') as HTMLButtonElement;
    act(() => pill.click());
    expect(group().getAttribute('data-state')).toBe('paused');
    view.rerender(liveStream(stream(30), false));
    expect(group().getAttribute('data-state')).toBe('stopped');
  });

  it('a pause holds the points and counts the new ones', () => {
    const view = mount(liveStream(stream(30)));
    const group = (): HTMLElement => byRole(view.container, 'group', 'Output rate') as HTMLElement;
    const before = tableRows(group());
    act(() => (byRole(group(), 'button', 'Live') as HTMLButtonElement).click());
    const later = [...stream(30), { t: NOW + 1000, value: 99 }, { t: NOW + 2000, value: 98 }];
    view.rerender(liveStream(later, true, NOW + 2000));
    expect(tableRows(group())).toEqual(before);
    expect(byRole(group(), 'button', 'Paused, 2 new')).not.toBeNull();
  });

  it('the hidden table holds the last value per 10 s step, newest first', () => {
    const { container } = mount(liveStream(stream(35)));
    const rows = tableRows(byRole(container, 'group', 'Output rate') as HTMLElement);
    expect(rows).toHaveLength(4);
    expect(rows[0]).toEqual(['0 seconds ago', '40.0']);
    expect(rows[1][0]).toBe('10 seconds ago');
  });

  it('the value is one text node', () => {
    const { container } = mount(liveStream(stream(10)));
    const value = container.querySelector('[data-stat-value]')?.firstElementChild;
    expect(value?.childNodes).toHaveLength(1);
    expect(value?.firstChild?.nodeType).toBe(Node.TEXT_NODE);
    expect(statValue(container, 'Output rate')).toBe('40.0 tok/s');
  });
});

describe('CellGrid', () => {
  const unit = { one: 'block', other: 'blocks' };

  function cells(container: HTMLElement): Element[] {
    const grid = byRole(container, 'img');
    return [...(grid?.querySelectorAll('[data-kind]') ?? [])];
  }

  it('cell counts by kind sum to 200 and a non-zero segment gets a cell', () => {
    const { container } = mount(
      createElement(CellGrid, {
        label: 'KV block pool',
        capacity: 10_000,
        unit,
        segments: [
          { key: 'used', label: 'In use', value: 6_200, color: 'var(--model-1)' },
          { key: 'tiny', label: 'Pinned', value: 3, color: 'var(--model-2)' },
        ],
        reserved: { value: 500, label: 'Reserved' },
      }),
    );
    const all = cells(container);
    expect(all).toHaveLength(200);
    const count = (kind: string): number => all.filter((cell) => cell.getAttribute('data-kind') === kind).length;
    expect(count('tiny')).toBeGreaterThanOrEqual(1);
    expect(count('used') + count('tiny') + count('free') + count('reserved')).toBe(200);
    expect(count('reserved')).toBe(10);
  });

  it('skips a threshold at 0', () => {
    const { container } = mount(
      createElement(CellGrid, {
        label: 'Pool',
        capacity: 100,
        unit,
        segments: [{ key: 'used', label: 'In use', value: 40, color: 'var(--model-1)' }],
        threshold: { value: 0, label: 'Eviction starts' },
      }),
    );
    expect(container.textContent).not.toContain('Eviction starts');
    const shown = mount(
      createElement(CellGrid, {
        label: 'Pool 2',
        capacity: 100,
        unit,
        segments: [{ key: 'used', label: 'In use', value: 40, color: 'var(--model-1)' }],
        threshold: { value: 80, label: 'Eviction starts' },
      }),
    );
    expect(shown.container.textContent).toContain('Eviction starts');
  });

  it('a capacity of 0 is the empty state', () => {
    const { container } = mount(
      createElement(CellGrid, { label: 'Pool', capacity: 0, unit, segments: [], emptyLabel: 'Pool not read yet' }),
    );
    expect(cells(container)).toHaveLength(0);
    expect(byRole(container, 'img', 'Pool: Pool not read yet')).not.toBeNull();
  });
});

const BASE: LiveTurnRowProps = {
  model: 'qwen3.6-27b',
  phase: 'decode',
  elapsedMs: 12_000,
  prefillDone: null,
  prefillTotal: null,
  generatedTokens: 1204,
  promptTokens: 3000,
  ctxWindow: 32_000,
};

function rowOf(container: HTMLElement, index = 0): HTMLElement {
  const list = byRole(container, 'list', 'In flight') as HTMLElement;
  return list.querySelectorAll('li')[index];
}

describe('LiveTurnRow', () => {
  it('prints the progress of each phase', () => {
    const cases: Array<[Partial<LiveTurnRowProps>, string, string]> = [
      [
        { phase: 'prefill', prefillDone: 12_400, prefillTotal: 31_000 },
        'Prefill',
        `${formatCount(12_400)} of ${formatCount(31_000)} tokens`,
      ],
      [{ phase: 'decode' }, 'Decode', '1,204 tokens out'],
      [{ phase: 'decode', generatedTokens: 1 }, 'Decode', '1 token out'],
      [{ phase: 'queued' }, 'Queued', 'waiting for a slot'],
      [{ phase: null }, 'Running', 'no progress for non-streamed turns'],
    ];
    for (const [patch, word, progress] of cases) {
      const { container } = mount(
        createElement(LiveTurnList, {
          label: 'In flight',
          children: createElement(LiveTurnRow, { ...BASE, ...patch }),
        }),
      );
      const row = rowOf(container);
      expect(row.textContent).toContain(word);
      expect(row.textContent).toContain(progress);
      expect(row.getAttribute('aria-label')).toContain(`${word.toLowerCase()}, ${progress}`);
      document.body.innerHTML = '';
    }
  });

  it('shows — for the context when the window is unknown', () => {
    const { container } = mount(
      createElement(LiveTurnList, {
        label: 'In flight',
        children: createElement(LiveTurnRow, { ...BASE, ctxWindow: null }),
      }),
    );
    const row = rowOf(container);
    expect(row.textContent).toContain('ctx —');
    expect(row.getAttribute('aria-label')).toContain('context unknown');
    expect(allByRole(row, 'meter')).toHaveLength(0);
  });

  it('a narrow prefill cell keeps a readable meter and shortens the count instead', () => {
    const rules = cssRules('components/arc-x/live-turn-row/live-turn-row.module.css');
    expect(cssValue(rules, '.meter', 'min-width')).toBe('48px');
    expect(cssValue(rules, '.progress', 'container')).toBe('progress / inline-size');
    expect(cssValue(rules, '.progressText[data-short]', 'display')).toBe('none');
    const narrow = rules.filter((rule) => rule.conditional && rule.selectors.includes('.progressText[data-short]'));
    expect(narrow.map((rule) => rule.decls.get('display'))).toEqual(['inline']);
    const source = readFileSync(
      join(import.meta.dirname, '../ui/src/components/arc-x/live-turn-row/live-turn-row.module.css'),
      'utf8',
    );
    expect(source.includes('@container progress (max-width: 239px)')).toBe(true);

    const { container } = mount(
      createElement(LiveTurnList, {
        label: 'In flight',
        children: createElement(LiveTurnRow, { ...BASE, phase: 'prefill', prefillDone: 1792, prefillTotal: 7168 }),
      }),
    );
    const full = rowOf(container).querySelector('[data-full]');
    const short = rowOf(container).querySelector('[data-short]');
    expect(full?.textContent).toBe('1,792 of 7,168 tokens');
    expect(short?.textContent).toBe('1,792 / 7,168');
    expect(short?.getAttribute('title')).toBe('1,792 of 7,168 tokens');
  });

  it('rows 9 and later show the same words and numbers', () => {
    const rows = Array.from({ length: 10 }, (_, index) =>
      createElement(LiveTurnRow, { ...BASE, key: `t${String(index)}` }),
    );
    const { container } = mount(createElement(LiveTurnList, { label: 'In flight', children: rows }));
    expect(rowOf(container, 9).textContent).toBe(rowOf(container, 0).textContent);
    expect(rowOf(container, 9).getAttribute('aria-label')).toBe(rowOf(container, 0).getAttribute('aria-label'));
  });
});

describe('QuantileBar', () => {
  function bar(n: number): Element {
    const { container } = mount(
      createElement(QuantileBar, {
        label: 'Decode speed, qwen',
        quantiles: { n, p05: 20, p50: 40, p95: 60 },
        min: 0,
        max: 100,
        unit: 'tok/s',
      }),
    );
    return byRole(container, 'img') as Element;
  }

  it('draws the p05–p95 band only from 20 samples', () => {
    expect(bar(19).querySelector('[data-kind="band"]')).toBeNull();
    expect(bar(20).querySelector('[data-kind="band"]')).not.toBeNull();
  });

  it('draws a ring instead of a tick below 5 samples', () => {
    const low = bar(4);
    expect(low.querySelector('[data-kind="ring"]')).not.toBeNull();
    expect(low.querySelector('[data-kind="median"]')).toBeNull();
    expect(low.getAttribute('aria-label')).toContain('low sample');
    expect(bar(5).querySelector('[data-kind="median"]')).not.toBeNull();
  });
});

describe('ScatterChart', () => {
  const series = [{ key: 'a', label: 'qwen', color: 'var(--model-1)' }];
  const points: ScatterPoint[] = Array.from({ length: 250 }, (_, index) => ({
    key: `trace-${String(index)}`,
    x: index * 10,
    y: 100 + index,
    series: 'a',
  }));

  it('caps the hidden table at 200 rows plus a pointer to the full list', () => {
    const { container } = mount(
      createElement(ScatterChart, {
        label: 'Prefill time',
        xLabel: 'New prompt tokens',
        xUnit: 'tokens',
        yUnit: 'ms',
        points,
        series,
      }),
    );
    const rows = tableRows(byRole(container, 'figure', 'Prefill time') as Element);
    expect(rows).toHaveLength(201);
    expect(rows[200][0]).toBe('+50 more points. Open the Turns tab for the full list.');
  });

  it('Enter on the plot selects the active point', () => {
    const onSelect = vi.fn();
    const { container } = mount(
      createElement(ScatterChart, {
        label: 'Prefill time',
        xLabel: 'New prompt tokens',
        xUnit: 'tokens',
        yUnit: 'ms',
        points: points.slice(0, 5),
        series,
        onSelect,
      }),
    );
    const slider = byRole(container, 'slider') as HTMLElement;
    press(slider, 'Home');
    press(slider, 'Enter');
    expect(onSelect).toHaveBeenCalledWith(points[0]);
  });

  function pathNumbers(d: string): number[] {
    const out: number[] = [];
    let token = '';
    for (const char of d) {
      const numeric = (char >= '0' && char <= '9') || char === '.' || (char === '-' && token === '');
      if (numeric) token += char;
      else {
        if (token !== '') out.push(Number(token));
        token = char === '-' ? '-' : '';
      }
    }
    if (token !== '') out.push(Number(token));
    return out;
  }

  it('an alt marker carries no transform attribute and is centred on its point', () => {
    const marks = (pattern: 'solid' | 'alt'): Element[] => {
      const { container } = mount(
        createElement(ScatterChart, {
          label: `Prefill ${pattern}`,
          xLabel: 'New prompt tokens',
          xUnit: 'tokens',
          yUnit: 'ms',
          points: points.slice(0, 5),
          series: [{ ...series[0], pattern }],
        }),
      );
      const figure = byRole(container, 'figure', `Prefill ${pattern}`) as Element;
      return [...figure.querySelectorAll('svg[aria-hidden] circle, svg[aria-hidden] path, svg[aria-hidden] rect')];
    };
    const circles = marks('solid').filter((mark) => mark.tagName.toLowerCase() === 'circle');
    const diamonds = marks('alt');
    expect(diamonds).toHaveLength(5);
    expect(circles).toHaveLength(5);
    for (const [index, diamond] of diamonds.entries()) {
      expect(diamond.hasAttribute('transform')).toBe(false);
      const coords = pathNumbers(diamond.getAttribute('d') ?? '');
      expect(coords).toHaveLength(8);
      const xs = [coords[0], coords[2], coords[4], coords[6]];
      const ys = [coords[1], coords[3], coords[5], coords[7]];
      const centreX = (Math.min(...xs) + Math.max(...xs)) / 2;
      const centreY = (Math.min(...ys) + Math.max(...ys)) / 2;
      expect(centreX).toBeCloseTo(Number(circles[index].getAttribute('cx')), 6);
      expect(centreY).toBeCloseTo(Number(circles[index].getAttribute('cy')), 6);
    }
  });
});

describe('live value and unit', () => {
  function reservedInside(scope: Element | null | undefined): Element[] {
    return [...(scope?.querySelectorAll('[style]') ?? [])].filter((node) =>
      (node.getAttribute('style') ?? '').includes('min-width'),
    );
  }

  it('reserves the width on the value and its unit together, so no gap opens between them', () => {
    const { container } = mount(liveStream(stream(90)));
    const line = byRole(container, 'group', 'Output rate')?.querySelector('[data-stat-value]');
    expect(line?.textContent).toContain('tok/s');
    expect(reservedInside(line)).toHaveLength(0);
    const rules = cssRules('components/arc-x/live-stream/live-stream.module.css');
    expect(cssValue(rules, '.valueLine', 'min-width')).toBe('5ch');
    document.body.innerHTML = '';
    const row = mount(createElement(LiveTurnList, { label: 'In flight', children: createElement(LiveTurnRow, BASE) }));
    const count = [...rowOf(row.container).querySelectorAll('span')].find((node) =>
      (node.textContent ?? '').endsWith('tokens out'),
    );
    expect(count).not.toBeUndefined();
    expect(reservedInside(count)).toHaveLength(0);
  });
});

describe('ModelBadge', () => {
  function hover(node: Element): void {
    act(() => {
      node.dispatchEvent(new PointerEvent('pointerover', { bubbles: true }));
    });
  }

  function widths(node: HTMLElement, scroll: number, client: number): void {
    Object.defineProperty(node, 'scrollWidth', { configurable: true, value: scroll });
    Object.defineProperty(node, 'clientWidth', { configurable: true, value: client });
  }

  it('names the whole model on hover when its box cuts the name, and only then', () => {
    const { container } = mount(createElement(ModelBadge, { name: 'Llama-3.2-3B-Instruct-4bit' }));
    const name = container.querySelector('[data-model] > span:last-child') as HTMLElement;
    expect(name.textContent).toBe('Llama-3.2-3B-Instruct-4bit');
    expect(name.hasAttribute('title')).toBe(false);
    widths(name, 137, 98);
    hover(name);
    expect(name.getAttribute('title')).toBe('Llama-3.2-3B-Instruct-4bit');
    widths(name, 137, 160);
    hover(name);
    expect(name.hasAttribute('title')).toBe(false);
  });

  it('names the whole model when it cuts the name in the middle', () => {
    const long = 'Gemma-4-31B-IT-UD-Q4_K_XL-mlx-extra-long';
    const { container } = mount(createElement(ModelBadge, { name: long }));
    const name = container.querySelector('[data-model] > span:last-child') as HTMLElement;
    expect(name.textContent).not.toBe(long);
    expect(name.getAttribute('title')).toBe(long);
  });
});

describe('StatTile hint', () => {
  it('carries a cut plain-text hint in full in its title; an element hint keeps its own', () => {
    const text = '5% of 100 GB quota · 4.2 GB blocks + 545 MB sidecars';
    const { container } = mount(
      createElement(
        'div',
        null,
        createElement(StatTile, { label: 'Disk', value: '4.7', unit: 'GB', hint: text }),
        createElement(StatTile, {
          label: 'Errors',
          value: '0',
          hint: createElement('span', { title: 'own title' }, 'no write errors'),
        }),
      ),
    );
    const plain = [...container.querySelectorAll('span')].find((node) => node.textContent === text);
    expect(plain?.getAttribute('title')).toBe(text);
    const own = [...container.querySelectorAll('span[title="own title"]')];
    expect(own).toHaveLength(1);
    expect(own[0].parentElement?.hasAttribute('title')).toBe(false);
  });
});
