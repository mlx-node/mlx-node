/**
 * @vitest-environment happy-dom
 */

import { createElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { MiddleText } from '../ui/src/components/arc-x/middle-text/middle-text.js';
import { ModelBadge } from '../ui/src/components/arc-x/model-badge/model-badge.js';
import { ThreadBar } from '../ui/src/pages/chat/thread-bar.js';
import { TitleBar } from '../ui/src/shell/title-bar.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, unmountAll } from './helpers/mount.js';

afterEach(unmountAll);

const LONG = 'qwen3.6-35b-a3b-agentworld-instruct-ud-q4-k-xl-mlx-dwq-8bits';

describe('MiddleText', () => {
  it('head and tail are the whole name; the tail keeps its last characters; the title is the name', () => {
    const container = mount(createElement(MiddleText, { text: LONG }));
    const root = container.firstElementChild!;
    expect(root.getAttribute('title')).toBe(LONG);
    const [head, tail, spoken] = [...root.children];
    expect(`${head!.textContent}${tail!.textContent}`).toBe(LONG);
    expect(tail!.textContent).toBe('wq-8bits');
    expect(head!.getAttribute('aria-hidden')).toBe('true');
    expect(tail!.getAttribute('aria-hidden')).toBe('true');
    expect(spoken!.getAttribute('aria-hidden')).toBeNull();
    expect(spoken!.textContent).toBe(LONG);
  });

  it('a short name is one span', () => {
    const container = mount(createElement(MiddleText, { text: 'qwen3-8b' }));
    const root = container.firstElementChild!;
    expect(root.children).toHaveLength(0);
    expect(root.textContent).toBe('qwen3-8b');
    expect(root.getAttribute('title')).toBe('qwen3-8b');
  });

  it('only the head shrinks, with an ellipsis; the tail never shrinks', () => {
    const rules = cssRules('components/arc-x/middle-text/middle-text.module.css');
    expect(cssValue(rules, '.head', 'text-overflow')).toBe('ellipsis');
    expect(cssValue(rules, '.head', 'min-width')).toBe('0');
    expect(cssValue(rules, '.tail', 'flex')).toBe('none');
    expect(cssValue(rules, '.root', 'white-space')).toBe('nowrap');
  });

  it('ModelBadge fit="width" prints the whole name and never cuts it by a character count', () => {
    const fit = mount(createElement(ModelBadge, { name: LONG, fit: 'width', maxChars: 10 }));
    expect([...fit.querySelectorAll('[aria-hidden="true"]')].map((n) => n.textContent).join('')).toBe(LONG);
    const chars = mount(createElement(ModelBadge, { name: LONG, maxChars: 28 }));
    expect(chars.textContent).not.toBe(LONG);
  });
});

describe('a cut heading or crumb', () => {
  const cut = (node: Element, cutNow: boolean): void => {
    Object.defineProperty(node, 'scrollWidth', { configurable: true, value: cutNow ? 900 : 100 });
    Object.defineProperty(node, 'clientWidth', { configurable: true, value: 100 });
  };

  it('the Chat h1 and the last crumb show their whole text as the title only while cut', () => {
    const title = 'A'.repeat(100);
    const bar = mount(createElement(ThreadBar, { title, open: null, onRename: () => {} }));
    const h1 = bar.querySelector('h1')!;
    cut(h1, true);
    h1.dispatchEvent(new PointerEvent('pointerover', { bubbles: true }));
    expect(h1.getAttribute('title')).toBe(title);
    cut(h1, false);
    h1.dispatchEvent(new PointerEvent('pointerover', { bubbles: true }));
    expect(h1.getAttribute('title')).toBeNull();

    const shell = mount(
      createElement(TitleBar, {
        crumbs: [{ label: 'Chat' }, { label: title }],
        narrow: false,
        rail: false,
        onToggleSidebar: () => {},
        onSearch: () => {},
      }),
    );
    const last = shell.querySelector('li:last-child > span')!;
    cut(last, true);
    last.dispatchEvent(new PointerEvent('pointerover', { bubbles: true }));
    expect(last.getAttribute('title')).toBe(title);
  });
});
