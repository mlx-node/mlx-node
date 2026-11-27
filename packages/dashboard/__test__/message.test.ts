/**
 * @vitest-environment happy-dom
 */

import { createElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { Message } from '../ui/src/components/arc-x/message/message.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, unmountAll } from './helpers/mount.js';

afterEach(unmountAll);

const CSS = 'components/arc-x/message/message.module.css';

describe('Message caption', () => {
  it('draws one line under the body, before the footer, outside it', () => {
    const container = mount(
      createElement(Message, {
        role: 'user',
        variant: 'chat',
        ts: 1,
        caption: 'Sent while working · 14:02',
        footer: createElement('button', { type: 'button' }, 'Copy'),
        children: 'use pnpm',
      }),
    );
    const article = container.querySelector('article')!;
    const parts = Array.from(article.children).map((child) => child.textContent);
    expect(parts).toEqual(['use pnpm', 'Sent while working · 14:02', 'Copy']);
    const footer = article.lastElementChild!;
    expect(footer.textContent).not.toContain('Sent while working');
  });

  it('no caption, no line', () => {
    const container = mount(createElement(Message, { role: 'user', variant: 'chat', children: 'hello' }));
    expect(container.querySelector('article')!.children).toHaveLength(1);
  });

  it('the caption is shown with no hover; the footer keeps its hover rule', () => {
    const rules = cssRules(CSS);
    expect(cssValue(rules, '.caption', 'opacity')).toBeUndefined();
    expect(rules.some((rule) => rule.selectors.some((s) => s.includes('.caption') && s.includes(':hover')))).toBe(
      false,
    );
    expect(cssValue(rules, '.caption', 'color')).toBe('var(--text-muted)');
    expect(cssValue(rules, '.caption', 'font-size')).toBe('var(--text-xs)');
    expect(cssValue(rules, '.caption', 'text-align')).toBe('right');
    expect(cssValue(rules, '.footer', 'opacity')).toBe('0');
    expect(cssValue(rules, '.root:hover > .footer', 'opacity')).toBe('1');
  });
});
