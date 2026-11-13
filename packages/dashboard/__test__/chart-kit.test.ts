/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { BarList } from '../ui/src/components/arc-x/bar-list/bar-list.js';
import { bandLayout, niceTicks } from '../ui/src/components/arc-x/chart/scale.js';
import type { ChartPoint, ChartSeries } from '../ui/src/components/arc-x/chart/types.js';
import { MiniSpark } from '../ui/src/components/arc-x/mini-spark/mini-spark.js';
import { groupedLayout, StackedBarChart } from '../ui/src/components/arc-x/stacked-bar-chart/stacked-bar-chart.js';
import { spreadLabels, TrendChart } from '../ui/src/components/arc-x/trend-chart/trend-chart.js';
import { modelSeries, resetModelColors } from '../ui/src/lib/model-colors.js';
import { alignedRange } from '../ui/src/lib/range.js';
import { OTHER_KEY, pivotByModel } from '../ui/src/lib/timeseries.js';
import type { MetricsTimeseries } from '../ui/src/lib/types.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { byRole, chartTable } from './render.js';

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  return container;
}

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

function points(values: readonly (number | null)[], samples?: readonly number[]): ChartPoint[] {
  return values.map((value, index) => ({
    key: String(index),
    label: `Hour ${String(index)}`,
    axisLabel: String(index),
    values: { a: value },
    ...(samples !== undefined ? { samples: { a: samples[index] } } : {}),
  }));
}

const ONE: ChartSeries[] = [{ key: 'a', label: 'Decode speed', color: 'var(--series-1)' }];

function tabStops(root: Element): Element[] {
  return [...root.querySelectorAll('[tabindex], button, a[href]')].filter(
    (element) => element.getAttribute('tabindex') !== '-1',
  );
}

describe('scale', () => {
  it('niceTicks gives 3 to 5 ticks on 1/2/5 steps that cover the domain', () => {
    for (const [min, max] of [
      [0, 1],
      [0, 7],
      [0, 42.7],
      [0, 999],
      [12, 13],
      [0, 0],
    ]) {
      const { ticks } = niceTicks(min, max);
      expect(ticks.length).toBeGreaterThanOrEqual(3);
      expect(ticks.length).toBeLessThanOrEqual(5);
      expect(ticks[0]).toBeLessThanOrEqual(min);
      expect(ticks[ticks.length - 1]).toBeGreaterThanOrEqual(max);
      const step = ticks[1] - ticks[0];
      const lead = step / 10 ** Math.floor(Math.log10(step));
      expect([1, 2, 5]).toContain(Math.round(lead * 1000) / 1000);
    }
  });

  it('bandLayout caps the band and centers it in its slot', () => {
    expect(bandLayout(10, 400)).toEqual({ step: 40, band: 24, offset: 8 });
    expect(bandLayout(100, 400)).toEqual({ step: 4, band: 2, offset: 1 });
    expect(bandLayout(0, 400)).toEqual({ step: 0, band: 0, offset: 0 });
  });
});

describe('chart tooltip', () => {
  it('moves a tooltip that would be cut above its point below it, inside what can show it', () => {
    const tooltip = (top: number, height: number): HTMLElement => {
      vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(() =>
        DOMRect.fromRect({ x: 0, y: top, width: 600, height: 200 }),
      );
      vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockImplementation(() => height);
      vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockImplementation(() => 100);
      const container = mount(
        createElement(StackedBarChart, { label: 'Turns', unit: 'turns', data: points([1, 2, 3]), series: ONE }),
      );
      const slider = byRole(container, 'slider') as HTMLElement;
      act(() => {
        slider.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true }));
      });
      const tip = container.querySelector<HTMLElement>('figure [style*="calc(-100% - 10px)"]');
      expect(tip).not.toBeNull();
      vi.restoreAllMocks();
      return tip!;
    };
    // Below the point is the tooltip's height plus both 10 px gaps further down.
    expect(tooltip(20, 300).getAttribute('style')).toContain('translate: 0px 320px');
    expect(tooltip(400, 40).getAttribute('style')).not.toContain('translate:');
  });
});

describe('StackedBarChart grouped', () => {
  it('keeps each bar at least 3 px wide for 30 columns of 2 series in 280 px', () => {
    const { layout, barWidth, innerGap } = groupedLayout(30, 280, 2);
    expect(barWidth).toBeGreaterThanOrEqual(3);
    expect(barWidth * 2 + innerGap).toBeLessThanOrEqual(layout.step);
    expect(groupedLayout(7, 280, 2).barWidth).toBe(15);
  });
});

describe('TrendChart', () => {
  it('a null bucket is "no data" in the table and the line does not cross it', () => {
    const container = mount(
      createElement(TrendChart, {
        label: 'Decode speed over time',
        unit: 'tok/s',
        data: points([40, 41, null, 44, 46]),
        series: ONE,
      }),
    );
    const rows = chartTable(container, 'Decode speed over time');
    expect(rows[2]).toEqual(['Hour 2', 'no data']);
    expect(rows[0][1]).not.toBe('no data');
    const figure = byRole(container, 'figure', 'Decode speed over time');
    const line = [...(figure?.querySelectorAll('g > path') ?? [])].find(
      (path) => (path.getAttribute('d') ?? '').length > 0,
    );
    const d = line?.getAttribute('d') ?? '';
    const moves = d.split('M').length - 1;
    expect(moves).toBe(2);
  });

  it('marks a low-sample point in the table', () => {
    const container = mount(
      createElement(TrendChart, {
        label: 'Speed',
        unit: 'tok/s',
        data: points([40, 41, 42], [20, 3, 20]),
        series: ONE,
      }),
    );
    const rows = chartTable(container, 'Speed');
    expect(rows[1][1].endsWith('(n=3)')).toBe(true);
    expect(rows[0][1].includes('n=')).toBe(false);
  });

  it('keeps the plot mounted when the column count changes, so the entrance does not play again', () => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    const chart = (count: number): ReactElement =>
      createElement(TrendChart, {
        label: 'Turns',
        unit: 'turns',
        data: points(Array.from({ length: count }, (_, index) => index + 1)),
        series: ONE,
      });
    act(() => root.render(chart(7)));
    const figure = container.querySelector('figure');
    const line = container.querySelector('path');
    act(() => root.render(chart(30)));
    expect(container.querySelector('figure')).toBe(figure);
    expect(container.querySelector('path')).toBe(line);
    expect(chartTable(container, 'Turns')).toHaveLength(30);
    act(() => root.unmount());
  });

  it('keeps two line-end labels apart when both lines end at the bottom', () => {
    const placed = spreadLabels(
      [
        { key: 'inFlight', y: 200 },
        { key: 'queued', y: 200 },
      ],
      200,
    );
    const a = placed.get('inFlight') ?? 0;
    const b = placed.get('queued') ?? 0;
    expect(Math.abs(a - b)).toBeGreaterThanOrEqual(13);
    expect(Math.max(a, b)).toBeLessThanOrEqual(200);
    expect(spreadLabels([{ key: 'x', y: 40 }], 200).get('x')).toBe(40);
  });

  it('a one-series chart has exactly one tab stop', () => {
    const container = mount(
      createElement(TrendChart, { label: 'Speed', unit: 'tok/s', data: points([1, 2, 3]), series: ONE }),
    );
    const figure = byRole(container, 'figure', 'Speed');
    expect(figure).not.toBeNull();
    expect(tabStops(figure as Element)).toHaveLength(1);
  });
});

describe('MiniSpark', () => {
  it('keeps one svg and path when the number of points changes, so the trace does not play again', () => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    const spark = (count: number): ReactElement =>
      createElement(MiniSpark, { label: 'Output rate', data: Array.from({ length: count }, (_, index) => index + 1) });
    act(() => root.render(spark(5)));
    const svg = container.querySelector('svg');
    const line = svg?.querySelector('path');
    act(() => root.render(spark(6)));
    expect(container.querySelector('svg')).toBe(svg);
    expect(container.querySelector('svg path')).toBe(line);
    expect(svg?.getAttribute('data-entrance')).not.toBe('pending');
    act(() => root.unmount());
  });
});

describe('model series past eight', () => {
  it('the ninth model is drawn with the alt pattern (hatched bars)', () => {
    resetModelColors();
    const names = Array.from({ length: 9 }, (_, index) => `model-${String(index + 1)}`);
    const series = modelSeries(names);
    expect(series[7].pattern).not.toBe('alt');
    expect(series[8].pattern).toBe('alt');
    const drawn = [...series.slice(0, 7), series[8]];
    const data: ChartPoint[] = [{ key: 'd', label: 'Day', values: Object.fromEntries(names.map((name) => [name, 1])) }];
    const container = mount(createElement(StackedBarChart, { label: 'Turns', unit: 'turns', data, series: drawn }));
    const figure = byRole(container, 'figure', 'Turns');
    expect(figure?.querySelectorAll('pattern').length).toBeGreaterThan(0);
  });

  it('a stack of the top eight models plus Other totals every model of the range', () => {
    resetModelColors();
    const range = alignedRange('24h', new Date(2026, 10, 11, 14, 37, 0).getTime());
    const names = Array.from({ length: 10 }, (_, index) => `model-${String(index + 1)}`);
    const rows = names.map(
      (model, index) =>
        ({ t: range.starts[2], model, totals: { turns: index + 1 } }) as MetricsTimeseries['points'][number],
    );
    const top = names.slice(0, 8);
    const pivot = pivotByModel(rows, range, (p) => p.totals.turns, { empty: 0, models: top, fold: 'sum' });
    const series = [...modelSeries(top), { key: OTHER_KEY, label: 'Other (2 models)', color: 'var(--model-other)' }];
    const container = mount(
      createElement(StackedBarChart, { label: 'Turns per hour', unit: 'turns', data: pivot.data, series }),
    );
    const table = chartTable(container, 'Turns per hour');
    expect(table[2][9]).toBe('19');
    expect(table[2][10]).toBe('55');
  });
});

describe('BarList', () => {
  it('cuts a long plain label with an ellipsis and keeps the whole label in its title', () => {
    const text = 'Failed · model reported an error · during prefill of a long prompt';
    const container = mount(
      createElement(BarList, {
        label: 'Where turns stop',
        unit: 'turns',
        rows: [{ key: 'a', label: text, text, value: 3 }],
      }),
    );
    const holder = container.querySelector(`li [title="${text}"]`);
    expect(holder).not.toBeNull();
    const inner = holder?.firstElementChild;
    expect(inner?.textContent).toBe(text);
    expect(inner?.className ?? '').toContain('labelText');
    const rules = cssRules('components/arc-x/bar-list/bar-list.module.css');
    expect(cssValue(rules, '.labelText', 'text-overflow', 'base')).toBe('ellipsis');
    expect(cssValue(rules, '.labelText', 'min-width', 'base')).toBe('0');
  });
});
