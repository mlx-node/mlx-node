/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { SwapText } from '../ui/src/components/arc-x/swap-text/swap-text.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import './render.js';

function mount(element: ReactElement): { container: HTMLElement; rerender(next: ReactElement): void } {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  return { container, rerender: (next) => act(() => root.render(next)) };
}

/** The cells of a SwapText: the elements that carry `data-key`. */
function cells(container: HTMLElement): HTMLElement[] {
  return [...container.querySelectorAll<HTMLElement>('[data-key]')];
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('SwapText', () => {
  it('whole text: textContent is exactly the new text and data-prev holds the old one', () => {
    const view = mount(createElement(SwapText, { text: 'Idle' }));
    expect(view.container.textContent).toBe('Idle');
    expect(view.container.querySelector('[data-prev]')).toBeNull();

    view.rerender(createElement(SwapText, { text: 'Running' }));
    expect(view.container.textContent).toBe('Running');
    const swapped = view.container.querySelectorAll('[data-prev]');
    expect(swapped).toHaveLength(1);
    expect(swapped[0].getAttribute('data-prev')).toBe('Idle');
  });

  it('char: only the cells whose character changed are remounted', () => {
    const view = mount(createElement(SwapText, { text: '0:09', granularity: 'char' }));
    const before = cells(view.container);
    expect(before).toHaveLength(4);

    view.rerender(createElement(SwapText, { text: '0:10', granularity: 'char' }));
    expect(view.container.textContent).toBe('0:10');
    const after = cells(view.container);
    expect(after).toHaveLength(4);
    expect(after[0]).toBe(before[0]);
    expect(after[1]).toBe(before[1]);
    expect(after[2]).not.toBe(before[2]);
    expect(after[3]).not.toBe(before[3]);
    expect(after.map((cell) => cell.getAttribute('data-prev'))).toEqual([null, null, '0', '9']);
  });

  it('char: a length change swaps the whole text', () => {
    const view = mount(createElement(SwapText, { text: '9:59', granularity: 'char' }));
    view.rerender(createElement(SwapText, { text: '10:00', granularity: 'char' }));
    expect(view.container.textContent).toBe('10:00');
    const swapped = view.container.querySelectorAll('[data-prev]');
    expect(swapped).toHaveLength(1);
    expect(swapped[0].getAttribute('data-prev')).toBe('9:59');

    view.rerender(createElement(SwapText, { text: '10:01', granularity: 'char' }));
    expect(view.container.textContent).toBe('10:01');
    expect([...view.container.querySelectorAll('[data-prev]')].map((cell) => cell.getAttribute('data-prev'))).toEqual([
      '0',
    ]);
  });

  it('reserves its width when asked', () => {
    const { container } = mount(createElement(SwapText, { text: '1', reserve: '6ch' }));
    expect((container.firstElementChild as HTMLElement).style.minWidth).toBe('6ch');
  });

  it('char: sits at the end of a reserved width, but text wider than its box clips at the end, not the start', () => {
    const rules = cssRules('components/arc-x/swap-text/swap-text.module.css');
    expect(cssValue(rules, ".root[data-granularity='char']", 'justify-content')).toBe('safe flex-end');
  });

  it('every box that clips the text has room for descenders and takes no more room in its line', () => {
    const rules = cssRules('components/arc-x/swap-text/swap-text.module.css');
    for (const clip of [".root[data-granularity='char']", ".root[data-granularity='text'] > .cell", '.now']) {
      expect(cssValue(rules, clip, 'padding-block')).toBe('0.25em');
      expect(cssValue(rules, clip, 'margin-block')).toBe('-0.25em');
    }
  });
});
