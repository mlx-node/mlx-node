/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { Placeholder } from '../ui/src/components/arc-x/placeholder/placeholder.js';
import './render.js';

function mount(element: ReactElement): { container: HTMLElement; rerender(next: ReactElement): void } {
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

describe('Placeholder', () => {
  it('without children is one hidden bone of the given size', () => {
    const { container } = mount(createElement(Placeholder, { width: '7ch', height: 20 }));
    const bones = container.querySelectorAll<HTMLElement>('[data-placeholder]');
    expect(bones).toHaveLength(1);
    expect(bones[0].getAttribute('aria-hidden')).toBe('true');
    expect(bones[0].style.width).toBe('7ch');
    expect(bones[0].style.height).toBe('20px');
  });

  it('with children shows the bone while loading and the children after, with no bone left', () => {
    const view = mount(createElement(Placeholder, { loading: true, width: '7ch' }, createElement('b', null, '1,284')));
    expect(view.container.querySelectorAll('[data-placeholder]')).toHaveLength(1);
    expect(view.container.textContent).toBe('');

    view.rerender(createElement(Placeholder, { loading: false, width: '7ch' }, createElement('b', null, '1,284')));
    expect(view.container.querySelector('[data-placeholder]')).toBeNull();
    expect(view.container.querySelector('[aria-hidden="true"]')).toBeNull();
    expect(view.container.textContent).toBe('1,284');
  });

  it('shows the bone again at once when loading turns back on', () => {
    const view = mount(createElement(Placeholder, { loading: false }, 'ready'));
    expect(view.container.querySelector('[data-placeholder]')).toBeNull();
    view.rerender(createElement(Placeholder, { loading: true }, 'ready'));
    expect(view.container.querySelectorAll('[data-placeholder]')).toHaveLength(1);
    expect(view.container.textContent).toBe('');
  });

  it('defaults to loading when children are given without a loading flag', () => {
    const { container } = mount(createElement(Placeholder, null, 'value'));
    expect(container.querySelectorAll('[data-placeholder]')).toHaveLength(1);
  });
});
