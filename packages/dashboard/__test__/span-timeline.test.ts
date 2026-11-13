/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  SpanTimeline,
  type TimelineRow,
  type TimelineSpan,
} from '../ui/src/components/arc-x/span-timeline/span-timeline.js';
import { allByRole, byRole } from './render.js';

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

/** happy-dom has no layout: every box is 600 × `height` at `top`, and every element is `own` px tall. */
function layout(top: number, height: number, own: number): void {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(() =>
    DOMRect.fromRect({ x: 0, y: top, width: 600, height }),
  );
  vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockImplementation(() => own);
}

const ROWS: TimelineRow[] = [
  { id: 'r1', label: 'Turn 1' },
  { id: 'r2', label: 'Turn 2' },
];

const SPANS: TimelineSpan[] = [
  { id: 'q', rowId: 'r1', kind: 'queue', startMs: 0, endMs: 38, label: 'Queue' },
  { id: 'p', rowId: 'r1', kind: 'prefill', startMs: 38, endMs: 450, label: 'Prefill', estimated: true },
  { id: 'd', rowId: 'r1', kind: 'decode', startMs: 450, endMs: 1200, label: 'Decode', status: 'error' },
  { id: 'x', rowId: 'r2', kind: 'decode', startMs: 100, endMs: 1800, label: 'Decode' },
];

function spanButtons(container: HTMLElement): HTMLButtonElement[] {
  const group = byRole(container, 'group', 'Turn timeline');
  return [...(group?.querySelectorAll<HTMLButtonElement>('button[aria-pressed]') ?? [])];
}

function ticks(container: HTMLElement, label: string): string[] {
  const group = byRole(container, 'group', label);
  return [...(group?.querySelectorAll('[aria-hidden="true"] > span[style*="left"]') ?? [])].map(
    (tick) => tick.textContent ?? '',
  );
}

describe('SpanTimeline', () => {
  it('prints its axis with formatDuration', () => {
    const spans: TimelineSpan[] = [
      { id: 'd', rowId: 'r1', kind: 'decode', startMs: 0, endMs: 12_000, label: 'Decode' },
    ];
    const plain = mount(createElement(SpanTimeline, { label: 'Plain timeline', rows: ROWS, spans }));
    const own = ticks(plain, 'Plain timeline');
    expect(own[0]).toBe('0');
    expect(own).toContain('10.0 s');
  });

  it('names each span with its duration, bounds and flags', () => {
    const container = mount(createElement(SpanTimeline, { label: 'Turn timeline', rows: ROWS, spans: SPANS }));
    const names = spanButtons(container).map((button) => button.getAttribute('aria-label') ?? '');
    expect(names).toHaveLength(4);
    const prefill = names.find((name) => name.startsWith('Prefill')) ?? '';
    expect(prefill.endsWith('estimated')).toBe(true);
    expect(names.some((name) => name.startsWith('Decode') && name.endsWith('error'))).toBe(true);
  });

  it('has one tab stop and the arrow keys move it', () => {
    const container = mount(createElement(SpanTimeline, { label: 'Turn timeline', rows: ROWS, spans: SPANS }));
    const buttons = spanButtons(container);
    expect(buttons.filter((button) => button.tabIndex === 0)).toHaveLength(1);
    const first = buttons.find((button) => button.tabIndex === 0) as HTMLButtonElement;
    act(() => first.focus());
    act(() => {
      first.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    });
    const after = spanButtons(container);
    expect(after.filter((button) => button.tabIndex === 0)).toHaveLength(1);
    expect(after.find((button) => button.tabIndex === 0)).not.toBe(first);
    expect(document.activeElement).toBe(after.find((button) => button.tabIndex === 0));
  });

  it('a click selects the span and Escape clears it', () => {
    const onSelect = vi.fn();
    const container = mount(
      createElement(SpanTimeline, { label: 'Turn timeline', rows: ROWS, spans: SPANS, onSelect }),
    );
    const prefill = spanButtons(container).find((button) =>
      (button.getAttribute('aria-label') ?? '').startsWith('Prefill'),
    );
    act(() => prefill?.click());
    expect(onSelect).toHaveBeenLastCalledWith(SPANS[1]);
    act(() => {
      prefill?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });
    expect(onSelect).toHaveBeenLastCalledWith(null);
  });

  it('shifts the tooltip by its midpoint share, so a span at either edge keeps it inside the plot', () => {
    const spans: TimelineSpan[] = [
      { id: 'a', rowId: 'r1', kind: 'setup', startMs: 0, endMs: 20, label: 'Setup' },
      { id: 'b', rowId: 'r1', kind: 'decode', startMs: 20, endMs: 980, label: 'Decode' },
      { id: 'c', rowId: 'r1', kind: 'decode', startMs: 980, endMs: 1000, label: 'Tail' },
    ];
    const container = mount(
      createElement(SpanTimeline, {
        label: 'Turn timeline',
        rows: [ROWS[0]!],
        spans,
        domainMs: [0, 1000],
      }),
    );
    const tipOf = (name: string): HTMLElement => {
      const button = spanButtons(container).find((candidate) =>
        (candidate.getAttribute('aria-label') ?? '').startsWith(name),
      );
      act(() => {
        button?.dispatchEvent(new PointerEvent('pointerover', { bubbles: true }));
      });
      const tip = Array.from(container.querySelectorAll<HTMLElement>('div')).find(
        (element) => element.style.left !== '' && element.textContent?.startsWith(name),
      );
      expect(tip).toBeTruthy();
      return tip!;
    };
    const setup = tipOf('Setup');
    expect(setup.style.left).toBe('1%');
    expect(setup.style.transform).toBe('translateX(-1%)');
    const tail = tipOf('Tail');
    expect(tail.style.left).toBe('99%');
    expect(tail.style.transform).toBe('translateX(-99%)');
  });

  it('moves the tooltip below its row when it would be cut above, and leaves it above when it fits', () => {
    const hover = (top: number): HTMLElement => {
      layout(top, 32, 120);
      const container = mount(createElement(SpanTimeline, { label: 'Turn timeline', rows: ROWS, spans: SPANS }));
      const button = spanButtons(container)[0];
      act(() => {
        button?.dispatchEvent(new PointerEvent('pointerover', { bubbles: true }));
      });
      const tip = Array.from(container.querySelectorAll<HTMLElement>('div')).find(
        (element) => element.style.left !== '' && element.textContent?.startsWith('Queue'),
      );
      expect(tip).toBeTruthy();
      vi.restoreAllMocks();
      return tip!;
    };
    // Row at 20 px: above would start at 20 + 4 − 120 = −96; below starts at 20 + 32 − 4 = 48.
    expect(hover(20).getAttribute('style')).toContain('translate: 0 144px');
    expect(hover(400).getAttribute('style')).not.toContain('translate:');
  });

  it('names the setup and to-first-token kinds in the legend and the hidden table', () => {
    const spans: TimelineSpan[] = [
      { id: 'f', rowId: 'r1', kind: 'first', startMs: 0, endMs: 300, label: 'To first token' },
      { id: 's', rowId: 'r2', kind: 'setup', startMs: 0, endMs: 50, label: 'Setup', estimated: true },
      { id: 'p', rowId: 'r2', kind: 'prefill', startMs: 50, endMs: 300, label: 'Prefill' },
    ];
    const container = mount(createElement(SpanTimeline, { label: 'Turn timeline', rows: ROWS, spans }));
    const legend = container.querySelector('[aria-hidden="true"][class*="legend"]');
    expect(legend?.textContent).toBe('To first tokenSetupPrefill');
    const kinds = [...(container.querySelectorAll('table tbody tr') ?? [])].map(
      (row) => row.querySelectorAll('td, th')[2]?.textContent,
    );
    expect(kinds).toEqual(['To first token', 'Setup', 'Prefill']);
  });

  it('lists every span in the hidden table, also past the drawing cap', () => {
    const many: TimelineSpan[] = Array.from({ length: 405 }, (_, index) => ({
      id: `s${String(index)}`,
      rowId: 'r1',
      kind: 'decode',
      startMs: index,
      endMs: index + 1,
      label: 'Decode',
    }));
    const container = mount(createElement(SpanTimeline, { label: 'Turn timeline', rows: ROWS, spans: many }));
    expect(container.textContent).toContain('+5 more spans');
    const table = container.querySelector('table');
    expect(table?.querySelectorAll('tbody tr')).toHaveLength(405);
    expect(allByRole(container, 'group', 'Turn timeline')).toHaveLength(1);
  });
});
