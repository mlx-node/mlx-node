/**
 * @vitest-environment happy-dom
 */

import { act, createElement, Fragment, useState, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { KpiDrilldown, type KpiDetail } from '../ui/src/components/arc-x/kpi-drilldown/kpi-drilldown.js';
import { StatGroup, StatTile } from '../ui/src/components/arc-x/stat-tile/stat-tile.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { allByRole, byRole, skipAnimations, statValue, waitFor } from './render.js';

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  return container;
}

afterEach(() => {
  document.body.innerHTML = '';
});

const DETAILS: Record<string, KpiDetail> = {
  decode: {
    label: 'Decode speed',
    value: '42.7',
    unit: 'tok/s',
    description: 'Median decode speed of completed turns in the last 24 hours.',
    children: createElement('p', null, 'Trend and breakdown'),
    footer: createElement('a', { href: '/metrics?kpi=decode' }, 'Open in Metrics'),
  },
  turns: {
    label: 'Turns',
    value: '2,904',
    description: 'Turns that finished in the last 24 hours.',
    children: createElement('p', null, 'Turn breakdown'),
  },
};

interface HarnessProps {
  initial?: string | null;
  spy: (key: string | null) => void;
  selectable?: boolean;
}

function Harness({ initial = null, spy, selectable = false }: HarnessProps): ReactElement {
  const [open, setOpen] = useState<string | null>(initial);
  const [kpi, setKpi] = useState('decode');
  const tiles = createElement(
    Fragment,
    null,
    createElement(StatTile, {
      label: 'Decode speed',
      value: '42.7',
      unit: 'tok/s',
      detailKey: 'decode',
      ...(selectable ? { onSelect: () => setKpi('decode'), selected: kpi === 'decode' } : {}),
    }),
    createElement(StatTile, { label: 'Turns', value: '2,904', detailKey: 'turns' }),
  );
  return createElement(KpiDrilldown, {
    label: 'Last 24 hours',
    open,
    onOpenChange: (key: string | null) => {
      spy(key);
      setOpen(key);
    },
    detail: (key: string) => DETAILS[key] ?? null,
    children: createElement(
      Fragment,
      null,
      createElement(StatGroup, { label: 'Last 24 hours', children: tiles }),
      createElement('div', null, createElement('button', { type: 'button' }, 'Chart')),
    ),
  });
}

function tile(container: HTMLElement, label: string): HTMLElement {
  return byRole(container, 'group', label) as HTMLElement;
}

describe('KpiDrilldown', () => {
  it('a tile with detailKey and no onSelect holds one collapsed button', () => {
    const container = mount(createElement(Harness, { spy: vi.fn() }));
    const buttons = allByRole(tile(container, 'Turns'), 'button');
    expect(buttons).toHaveLength(1);
    expect(buttons[0].getAttribute('aria-expanded')).toBe('false');
    expect(buttons[0].getAttribute('aria-label')).toBe('Open Turns details');
  });

  it('a selectable tile holds a select button and, beside it, the details button', () => {
    const container = mount(createElement(Harness, { spy: vi.fn(), selectable: true }));
    const group = tile(container, 'Decode speed');
    const select = byRole(group, 'button', 'Decode speed') as HTMLElement;
    expect(select.getAttribute('aria-pressed')).toBe('true');
    const details = byRole(group, 'button', 'Open Decode speed details') as HTMLElement;
    expect(details.getAttribute('aria-expanded')).toBe('false');
    expect(select.contains(details)).toBe(false);
    for (const button of container.querySelectorAll('button')) {
      expect(button.querySelector('button')).toBeNull();
    }
    expect(statValue(container, 'Decode speed')).toBe('42.7 tok/s');
  });

  it('opening mounts the region, focuses its heading and makes the stage inert', () => {
    const spy = vi.fn();
    const container = mount(createElement(Harness, { spy }));
    const trigger = byRole(tile(container, 'Decode speed'), 'button', 'Open Decode speed details') as HTMLElement;
    act(() => trigger.click());
    expect(spy).toHaveBeenLastCalledWith('decode');
    const region = byRole(container, 'region', 'Decode speed') as HTMLElement;
    expect(region).not.toBeNull();
    expect(document.activeElement).toBe(byRole(region, 'heading', 'Decode speed'));
    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    expect(trigger.getAttribute('aria-controls')).toBe(region.id);
    expect(trigger.closest('[inert]')).not.toBeNull();
    expect(byRole(container, 'button', 'Chart')?.closest('[inert]')).not.toBeNull();
    expect(region.closest('[inert]')).toBeNull();
    expect(region.textContent).toContain('Trend and breakdown');
  });

  it('the open tile and the panel head show the same value', () => {
    const container = mount(createElement(Harness, { spy: vi.fn(), initial: 'decode' }));
    const region = byRole(container, 'region', 'Decode speed') as HTMLElement;
    const head = region.querySelector('[data-stat-value]')?.textContent;
    expect(head).toBe(statValue(container, 'Decode speed'));
  });

  it('Escape closes and focus returns to the trigger', async () => {
    const spy = vi.fn();
    const container = mount(createElement(Harness, { spy }));
    const trigger = byRole(tile(container, 'Turns'), 'button', 'Open Turns details') as HTMLElement;
    act(() => trigger.click());
    const heading = byRole(byRole(container, 'region', 'Turns') as HTMLElement, 'heading', 'Turns') as HTMLElement;
    act(() => {
      heading.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });
    expect(spy).toHaveBeenLastCalledWith(null);
    expect(byRole(container, 'region', 'Turns')).toBeNull();
    await waitFor(() => document.activeElement === trigger);
    expect(trigger.closest('[inert]')).toBeNull();
  });

  it('the close button closes and focus returns to the trigger', async () => {
    const spy = vi.fn();
    const container = mount(createElement(Harness, { spy }));
    const trigger = byRole(tile(container, 'Decode speed'), 'button', 'Open Decode speed details') as HTMLElement;
    act(() => trigger.click());
    const close = byRole(byRole(container, 'region', 'Decode speed') as HTMLElement, 'button', 'Close details');
    act(() => {
      close?.focus();
      close?.click();
    });
    expect(spy).toHaveBeenLastCalledWith(null);
    await waitFor(() => document.activeElement === trigger);
  });

  it('under full motion, gives focus back to the trigger as the panel starts to shrink, not when it unmounts', async () => {
    const previous = skipAnimations(false);
    try {
      for (const how of ['Escape', 'Close details'] as const) {
        document.body.innerHTML = '';
        const container = mount(createElement(Harness, { spy: vi.fn() }));
        const trigger = byRole(tile(container, 'Turns'), 'button', 'Open Turns details') as HTMLElement;
        act(() => trigger.click());
        const region = byRole(container, 'region', 'Turns') as HTMLElement;
        const heading = byRole(region, 'heading', 'Turns') as HTMLElement;
        expect(document.activeElement).toBe(heading);
        if (how === 'Escape') {
          act(() => {
            heading.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
          });
        } else {
          const close = byRole(region, 'button', 'Close details') as HTMLElement;
          act(() => {
            close.focus();
            close.click();
          });
        }
        expect(region.isConnected).toBe(true);
        expect(region.hasAttribute('data-closing')).toBe(true);
        expect(document.activeElement).toBe(trigger);

        const chart = byRole(container, 'button', 'Chart') as HTMLElement;
        act(() => chart.focus());
        await act(async () => {
          await new Promise((resolve) => {
            setTimeout(resolve, 50);
          });
        });
        expect(document.activeElement).toBe(chart);
      }
    } finally {
      skipAnimations(previous);
    }
  });

  it('draws its veil and panel inside the stage, which keeps their z-index out of the page', () => {
    const container = mount(createElement(Harness, { spy: vi.fn(), initial: 'decode' }));
    const region = byRole(container, 'region', 'Decode speed') as HTMLElement;
    const stage = region.parentElement as HTMLElement;
    expect(stage.hasAttribute('data-open')).toBe(true);
    expect(stage.querySelector(':scope > [aria-hidden="true"]')).not.toBeNull();
    expect(cssValue(cssRules('components/arc-x/kpi-drilldown/kpi-drilldown.module.css'), '.stage', 'isolation')).toBe(
      'isolate',
    );
  });

  it('a deep link mounts the panel without moving focus', () => {
    const container = mount(createElement(Harness, { spy: vi.fn(), initial: 'turns' }));
    expect(byRole(container, 'region', 'Turns')).not.toBeNull();
    expect(document.activeElement).toBe(document.body);
  });

  it('an unknown key opens nothing and the param is dropped', () => {
    const spy = vi.fn();
    const container = mount(createElement(Harness, { spy, initial: 'nope' }));
    expect(allByRole(container, 'region')).toHaveLength(0);
    expect(spy).toHaveBeenCalledWith(null);
    expect(container.querySelector('[inert]')).toBeNull();
  });
});
