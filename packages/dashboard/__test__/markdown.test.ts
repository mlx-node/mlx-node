/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { freshBlocks, type HastNodeLike } from '../ui/src/components/arc-x/markdown/fresh-blocks.js';
import { Markdown } from '../ui/src/components/arc-x/markdown/markdown.js';
import { skipAnimations } from './render.js';

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  return container;
}

async function withRichMotion(body: () => void | Promise<void>): Promise<void> {
  const previous = skipAnimations(false);
  try {
    await body();
  } finally {
    skipAnimations(previous);
  }
}

function element(tag: string, offset: number, children: HastNodeLike[] = []): HastNodeLike {
  return { type: 'element', position: { start: { offset } }, properties: { tag }, children };
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('freshBlocks', () => {
  it('marks the root children that start at or after the offset, and nothing nested', () => {
    const nestedLate = element('li', 30);
    const tree: HastNodeLike = {
      type: 'root',
      children: [
        element('p', 0),
        { type: 'text', position: { start: { offset: 9 } } },
        element('ul', 10, [element('li', 10), nestedLate]),
        element('p', 40),
        element('pre', 55),
      ],
    };
    freshBlocks(40)(tree);
    const marked = (tree.children ?? []).map((child) => child.properties?.dataFresh === '');
    expect(marked).toEqual([false, false, false, true, true]);
    expect(nestedLate.properties?.dataFresh).toBeUndefined();
    expect(tree.children?.[3].properties?.tag).toBe('p');
  });

  it('marks nothing when the offset is past the end', () => {
    const tree: HastNodeLike = { type: 'root', children: [element('p', 0), element('p', 12)] };
    freshBlocks(100)(tree);
    expect((tree.children ?? []).some((child) => child.properties?.dataFresh !== undefined)).toBe(false);
  });
});

describe('Markdown', () => {
  const text = 'First paragraph.\n\n- one\n- two\n\nSecond paragraph.';
  const secondAt = text.indexOf('Second');

  it('renders the blocks on its own elements, with no raw HTML', () => {
    const container = mount(createElement(Markdown, { text: 'Hello <b>bold</b> and `code`.\n\n```\nlet a = 1;\n```' }));
    expect(container.querySelector('b')).toBeNull();
    expect(container.querySelector('p code')?.textContent).toBe('code');
    expect(container.querySelector('pre code')?.textContent).toBe('let a = 1;');
  });

  it('under the harness no block is marked fresh', () => {
    const container = mount(createElement(Markdown, { text, freshFrom: secondAt }));
    expect(container.querySelector('[data-fresh]')).toBeNull();
    expect(container.textContent).toContain('Second paragraph.');
  });

  it('with motion on, marks exactly the top-level blocks at or after freshFrom', async () => {
    await withRichMotion(() => {
      const container = mount(createElement(Markdown, { text, freshFrom: secondAt }));
      const fresh = [...container.querySelectorAll('[data-fresh]')];
      expect(fresh.map((node) => node.textContent)).toEqual(['Second paragraph.']);
    });
  });

  it('with motion on, a new item of a list that started earlier is not marked', async () => {
    const list = '- one\n- two\n- three';
    await withRichMotion(() => {
      const container = mount(createElement(Markdown, { text: list, freshFrom: list.indexOf('- three') }));
      expect(container.querySelector('[data-fresh]')).toBeNull();
      expect(container.querySelectorAll('li')).toHaveLength(3);
    });
  });

  it('with motion on, a fresh code block carries the mark on its wrapper', async () => {
    const withCode = 'Intro.\n\n```\nnew code\n```';
    await withRichMotion(() => {
      const container = mount(createElement(Markdown, { text: withCode, freshFrom: withCode.indexOf('```') }));
      const fresh = [...container.querySelectorAll('[data-fresh]')];
      expect(fresh).toHaveLength(1);
      expect(fresh[0].querySelector('pre code')?.textContent).toBe('new code');
    });
  });

  it('without freshFrom nothing fades, with motion on', async () => {
    await withRichMotion(() => {
      const container = mount(createElement(Markdown, { text }));
      expect(container.querySelector('[data-fresh]')).toBeNull();
    });
  });
});
