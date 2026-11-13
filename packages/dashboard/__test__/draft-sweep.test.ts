/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { DraftSweep } from '../ui/src/components/arc-x/draft-sweep/draft-sweep.js';
import { byRole } from './render.js';

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

const LABEL = 'Share of drafts kept at each position';

function sweep(container: HTMLElement): HTMLElement {
  const root = container.querySelector<HTMLElement>('[role="img"]');
  if (root === null) throw new Error('no sweep');
  return root;
}

describe('DraftSweep', () => {
  it('draws one cell per rate and folds positions past 8 into +n', () => {
    const rates = [0.92, 0.71, 0.48, 0.22, 0.1, 0.05, 0.04, 0.03, 0.77, 0.66];
    const container = mount(createElement(DraftSweep, { label: LABEL, rates }));
    const text = sweep(container).textContent ?? '';
    expect(text).toContain('92%');
    expect(text).toContain('3%');
    expect(text).not.toContain('77%');
    expect(text).not.toContain('66%');
    expect(text).toContain('+2');
  });

  it('a null rate prints —', () => {
    const container = mount(createElement(DraftSweep, { label: LABEL, rates: [0.9, null] }));
    expect(sweep(container).textContent).toContain('—');
  });

  it('the aria-label lists every position, also the folded ones', () => {
    const rates = [0.92, 0.71, null, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05];
    const container = mount(createElement(DraftSweep, { label: LABEL, rates }));
    const name = sweep(container).getAttribute('aria-label') ?? '';
    expect(name.startsWith(`${LABEL}: position 1 92%, position 2 71%, position 3 no data`)).toBe(true);
    expect(name.endsWith('position 9 5%')).toBe(true);
    expect(byRole(container, 'img', name)).not.toBeNull();
  });

  it('marks a low-sample cell', () => {
    const container = mount(createElement(DraftSweep, { label: LABEL, rates: [0.9, 0.5], samples: [400, 3] }));
    const name = sweep(container).getAttribute('aria-label') ?? '';
    expect(name).toContain('position 2 50% (low sample)');
    expect(name).not.toContain('position 1 90% (low sample)');
  });
});
