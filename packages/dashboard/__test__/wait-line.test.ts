/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it } from 'vite-plus/test';

import { WaitLine } from '../ui/src/components/arc-x/wait-line/wait-line.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import './render.js';

describe('WaitLine', () => {
  it('is a busy assistant article: one aria-hidden orb and the status as text', () => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => root.render(createElement(WaitLine, { orb: 'hold', text: 'Waiting for the model…' })));
    const line = container.querySelector('article')!;
    expect(line.getAttribute('aria-busy')).toBe('true');
    expect(line.getAttribute('aria-label')).toBe('Assistant');
    expect(line.getAttribute('data-wait-line')).toBe('hold');
    expect(line.hasAttribute('data-enter')).toBe(false);
    expect(line.textContent).toBe('Waiting for the model…');
    const orbs = line.querySelectorAll('[data-wait-orb]');
    expect(orbs).toHaveLength(1);
    expect(orbs[0]!.getAttribute('data-wait-orb')).toBe('hold');
    expect(orbs[0]!.getAttribute('aria-hidden')).toBe('true');
    const orb = orbs[0];

    act(() => root.render(createElement(WaitLine, { orb: 'think', text: 'Thinking…', enter: true })));
    expect(line.getAttribute('data-enter')).toBe('');
    expect(line.getAttribute('data-wait-line')).toBe('think');
    expect(line.textContent).toBe('Thinking…');
    expect(line.querySelector('[data-wait-orb]')).toBe(orb);
    act(() => root.unmount());
    container.remove();
  });

  it('keeps one box: at least 24 px high, 13 px secondary text, and no rise under reduced motion', () => {
    const rules = cssRules('components/arc-x/wait-line/wait-line.module.css');
    expect(cssValue(rules, '.root', 'min-height')).toBe('24px');
    expect(cssValue(rules, '.root', 'font-size')).toBe('var(--text-sm)');
    expect(cssValue(rules, '.root', 'color')).toBe('var(--text-secondary)');
    expect(cssValue(rules, '.root[data-enter]', 'animation', 'reduce')).toBe('none');
  });
});
