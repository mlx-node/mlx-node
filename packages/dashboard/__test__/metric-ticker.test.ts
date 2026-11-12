/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { MetricTicker, type TickerItem } from '../ui/src/components/arc-x/metric-ticker/metric-ticker.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { allByRole, byRole, statValue, waitFor } from './render.js';

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

const ITEMS: TickerItem[] = [
  {
    key: 'rate',
    label: 'Output rate',
    value: '42.7',
    unit: 'tok/s',
    delta: { text: '+4.2', direction: 'up', goodWhen: 'up' },
    spark: { data: [1, 2, 3, null, 5], label: 'Output rate, last 10 minutes' },
  },
  {
    key: 'queued',
    label: 'Queued',
    value: '0',
    delta: { text: '−1', direction: 'down', goodWhen: 'down' },
    facts: [
      { label: 'Queued', value: '0' },
      { label: 'Max at once', value: '4' },
    ],
  },
  {
    key: 'mem',
    label: 'MLX memory',
    value: null,
    unit: 'GB',
    delta: { text: '+0.3', direction: 'up', goodWhen: 'neutral' },
  },
  { key: 'uptime', label: 'Uptime', value: '3h 12m' },
];

function ticker(props: Partial<Parameters<typeof MetricTicker>[0]> = {}): ReactElement {
  return createElement(MetricTicker, { label: 'Now', items: ITEMS, ...props });
}

function strip(container: HTMLElement): HTMLElement {
  return byRole(container, 'group', 'Now') as HTMLElement;
}

describe('MetricTicker', () => {
  it('reserves the width on the value and its unit together, so no gap opens between them', () => {
    const { container } = mount(ticker());
    const node = byRole(strip(container), 'group', 'Output rate')?.querySelector('[data-stat-value]') as HTMLElement;
    const reserve = node.style.minWidth;
    expect(reserve.endsWith('ch')).toBe(true);
    expect(Number.parseFloat(reserve)).toBeGreaterThan('42.7'.length);
    const inner = [...node.querySelectorAll('[style]')].filter((el) =>
      (el.getAttribute('style') ?? '').includes('min-width'),
    );
    expect(inner).toHaveLength(0);
  });

  it('renders one group per item, named by its label, and statValue reads value + unit', () => {
    const { container } = mount(ticker());
    for (const item of ITEMS) expect(allByRole(strip(container), 'group', item.label)).toHaveLength(1);
    expect(statValue(container, 'Output rate')).toBe('42.7 tok/s');
    expect(statValue(container, 'Uptime')).toBe('3h 12m');
  });

  it('a null value prints — and no chip', () => {
    const { container } = mount(ticker());
    const item = byRole(container, 'group', 'MLX memory') as HTMLElement;
    expect(statValue(container, 'MLX memory')).toBe('—');
    expect(item.querySelector('[aria-label$="in the last minute"]')).toBeNull();
  });

  it('a delta renders its text and names its direction', () => {
    const { container } = mount(ticker());
    const rate = byRole(container, 'group', 'Output rate') as HTMLElement;
    const chip = byLabel(rate, 'up 4.2 in the last minute');
    expect(chip?.textContent).toContain('+4.2');
    expect(byLabel(byRole(container, 'group', 'Queued') as HTMLElement, 'down 1 in the last minute')).not.toBeNull();
  });

  it('an item with facts opens its panel on focus and Escape closes it', async () => {
    const { container } = mount(ticker());
    const button = byRole(byRole(container, 'group', 'Queued') as HTMLElement, 'button') as HTMLButtonElement;
    expect(button.getAttribute('aria-expanded')).toBe('false');
    act(() => button.focus());
    const panel = byRole(container, 'group', 'Queued details') as HTMLElement;
    expect(panel).not.toBeNull();
    expect(panel.textContent).toContain('Max at once');
    expect(button.getAttribute('aria-expanded')).toBe('true');
    expect(button.getAttribute('aria-controls')).toBe(panel.id);
    act(() => {
      button.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });
    await waitFor(() => byRole(container, 'group', 'Queued details') === null);
    expect(button.getAttribute('aria-expanded')).toBe('false');
  });

  it('loading renders placeholders and aria-busy', () => {
    const { container } = mount(ticker({ items: [], loading: true }));
    const root = strip(container);
    expect(root.getAttribute('aria-busy')).toBe('true');
    expect(root.querySelectorAll('[data-placeholder]').length).toBeGreaterThanOrEqual(6);
  });

  it('stale shows the heartbeat cell, hides chips, never moves, and keeps the values', () => {
    const view = mount(ticker());
    view.rerender(ticker({ stale: '12 s ago' }));
    const root = strip(view.container);
    expect(root.textContent).toContain('Last heartbeat');
    expect(root.textContent).toContain('12 s ago');
    expect(root.querySelectorAll('[aria-label$="in the last minute"]')).toHaveLength(0);
    expect(root.hasAttribute('data-moving')).toBe(false);
    expect(statValue(view.container, 'Output rate')).toBe('42.7 tok/s');
  });

  it('a tile takes the width of its content and never shrinks below it (happy-dom has no layout: the rules are read)', () => {
    const rules = cssRules('components/arc-x/metric-ticker/metric-ticker.module.css');
    for (const rule of rules) {
      if (!rule.selectors.some((selector) => selector.endsWith('.item'))) continue;
      const flex = rule.decls.get('flex');
      if (flex !== undefined) expect(flex.split(' ').slice(0, 3)).toEqual(['1', '0', 'auto']);
      expect(rule.decls.has('flex-basis')).toBe(false);
      expect(rule.decls.has('flex-shrink')).toBe(false);
    }
    expect(cssValue(rules, '.item', 'flex')).toBe('1 0 auto');
    expect(cssValue(rules, '.item', 'min-width')).toBe('132px');
  });

  it('a tile with a delta member holds a chip slot that only grows; a tile without one has none', () => {
    const slot = (container: HTMLElement, label: string): HTMLElement | null =>
      (byRole(strip(container), 'group', label) as HTMLElement).querySelector('[data-chip-slot]');
    const reserve = (node: HTMLElement | null): string => node?.style.minWidth ?? '';
    const items = (rate: TickerItem['delta']): TickerItem[] => [
      { key: 'rate', label: 'Output rate', value: '78.9', unit: 'tok/s', delta: rate },
      { key: 'uptime', label: 'Uptime', value: '3h 12m' },
    ];
    const view = mount(ticker({ items: items(null) }));
    expect(slot(view.container, 'Uptime')).toBeNull();
    const empty = slot(view.container, 'Output rate');
    expect(empty).not.toBeNull();
    expect(empty?.childElementCount).toBe(0);
    expect(reserve(empty).startsWith('calc(5ch')).toBe(true);
    view.rerender(ticker({ items: items({ text: '+1,234.5', direction: 'up', goodWhen: 'up' }) }));
    expect(reserve(slot(view.container, 'Output rate')).startsWith('calc(9ch')).toBe(true);
    view.rerender(ticker({ items: items({ text: '+2.0', direction: 'up', goodWhen: 'up' }) }));
    expect(reserve(slot(view.container, 'Output rate')).startsWith('calc(9ch')).toBe(true);
    view.rerender(ticker({ items: items({ text: '+2.0', direction: 'up', goodWhen: 'up' }), stale: '12 s ago' }));
    const stale = slot(view.container, 'Output rate');
    expect(stale).not.toBeNull();
    expect(stale?.childElementCount).toBe(0);
    expect(reserve(stale).startsWith('calc(9ch')).toBe(true);
  });

  it('holds exactly one copy of the items', () => {
    const { container } = mount(ticker());
    expect(container.querySelectorAll('[data-stat-value]')).toHaveLength(ITEMS.length);
    expect(container.querySelectorAll('button')).toHaveLength(ITEMS.length);
    expect((strip(container).textContent ?? '').split('Uptime')).toHaveLength(2);
    expect(strip(container).getAttribute('data-mode')).toBe('fit');
  });
});

function byLabel(scope: HTMLElement, label: string): HTMLElement | null {
  for (const element of scope.querySelectorAll<HTMLElement>('[aria-label]')) {
    if (element.getAttribute('aria-label') === label) return element;
  }
  return null;
}
