/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { NumberValue } from '../ui/src/components/arc-x/number-value/number-value.js';
import { parseNumeric } from '../ui/src/components/arc-x/number-value/parse-numeric.js';
import { skipAnimations } from './render.js';

function mount(element: ReactElement): { container: HTMLElement; rerender(next: ReactElement): void } {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  return { container, rerender: (next) => act(() => root.render(next)) };
}

/** Elements whose whole text is exactly `text` and that hold no element children. */
function leavesWithText(container: HTMLElement, text: string): Element[] {
  return [...container.querySelectorAll('*')].filter((node) => node.children.length === 0 && node.textContent === text);
}

async function withRichMotion(body: () => void | Promise<void>): Promise<void> {
  const previous = skipAnimations(false);
  try {
    await body();
  } finally {
    skipAnimations(previous);
  }
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('parseNumeric', () => {
  it.each([
    ['1,284', { prefix: '', value: 1284, decimals: 0, suffix: '' }],
    ['42.5', { prefix: '', value: 42.5, decimals: 1, suffix: '' }],
    ['98.2%', { prefix: '', value: 98.2, decimals: 1, suffix: '%' }],
    ['1.2k', { prefix: '', value: 1.2, decimals: 1, suffix: 'k' }],
    ['−3.1%', { prefix: '−', value: 3.1, decimals: 1, suffix: '%' }],
    ['0', { prefix: '', value: 0, decimals: 0, suffix: '' }],
    ['512 ms', { prefix: '', value: 512, decimals: 0, suffix: ' ms' }],
    ['$1,000.50', { prefix: '$', value: 1000.5, decimals: 2, suffix: '' }],
  ])('splits %s', (text, expected) => {
    expect(parseNumeric(text)).toEqual(expected);
  });

  it.each(['—', '', 'n/a', '2m 05s', '14:05', '3 / 8', '1,28', '01', '1,2345'])('returns null for %s', (text) => {
    expect(parseNumeric(text)).toBeNull();
  });
});

describe('NumberValue under the harness', () => {
  it('renders — for null and for —', () => {
    const view = mount(createElement(NumberValue, { text: null }));
    expect(view.container.textContent).toBe('—');
    view.rerender(createElement(NumberValue, { text: '—' }));
    expect(view.container.textContent).toBe('—');
  });

  it('renders a number as its final text in one text node', () => {
    const view = mount(createElement(NumberValue, { text: '1,284', unit: 'ms' }));
    const root = view.container.firstElementChild;
    expect(root?.childNodes).toHaveLength(1);
    expect(root?.firstChild?.nodeType).toBe(Node.TEXT_NODE);
    expect(root?.textContent).toBe('1,284');

    view.rerender(createElement(NumberValue, { text: '1,300', unit: 'ms' }));
    expect(view.container.firstElementChild?.textContent).toBe('1,300');
  });

  it('reserves its width when asked', () => {
    const { container } = mount(createElement(NumberValue, { text: '42', reserve: '5ch' }));
    expect((container.firstElementChild as HTMLElement).style.minWidth).toBe('5ch');
  });
});

describe('NumberValue with motion on', () => {
  it('sends compound text to a character swap: no wheels, textContent is the text', async () => {
    await withRichMotion(() => {
      const view = mount(createElement(NumberValue, { text: '2m 05s' }));
      expect(view.container.textContent).toBe('2m 05s');
      expect(view.container.querySelector('[aria-hidden="true"]')).toBeNull();
      view.rerender(createElement(NumberValue, { text: '2m 06s' }));
      expect(view.container.textContent).toBe('2m 06s');
    });
  });

  it('a measured number gets wheels whose hidden text is the final value from the first frame', async () => {
    await withRichMotion(() => {
      const view = mount(createElement(NumberValue, { text: '1,284' }));
      expect(view.container.querySelector('[aria-hidden="true"]')).not.toBeNull();
      expect(leavesWithText(view.container, '1,284')).toHaveLength(1);
    });
  });

  it('never mounts wheels for — or null', async () => {
    await withRichMotion(() => {
      const view = mount(createElement(NumberValue, { text: null }));
      expect(view.container.textContent).toBe('—');
      expect(view.container.querySelector('[aria-hidden="true"]')).toBeNull();
      view.rerender(createElement(NumberValue, { text: '—' }));
      expect(view.container.querySelector('[aria-hidden="true"]')).toBeNull();
    });
  });

  it('compound text turning into a number swaps to the value: no wheel starts at 0', async () => {
    await withRichMotion(() => {
      const view = mount(createElement(NumberValue, { text: '1m 05s' }));
      view.rerender(createElement(NumberValue, { text: '59.8 s' }));
      expect(view.container.querySelector('[data-prev]')?.getAttribute('data-prev')).toBe('1m 05s');
      const shown = [...view.container.querySelectorAll<HTMLElement>('[aria-hidden="true"] span')]
        .filter((node) => node.children.length === 0 && node.style.visibility === 'visible')
        .map((node) => node.textContent)
        .join('');
      expect(shown).toBe('598');
    });
  });

  it('a new shape mounts one new counter: one hidden text with the new value, the old text only as data-prev', async () => {
    await withRichMotion(() => {
      const view = mount(createElement(NumberValue, { text: '950' }));
      view.rerender(createElement(NumberValue, { text: '1.2k' }));
      expect(leavesWithText(view.container, '1.2k')).toHaveLength(1);
      expect(leavesWithText(view.container, '950')).toHaveLength(0);
      expect(view.container.querySelector('[data-prev]')?.getAttribute('data-prev')).toBe('950');
    });
  });
});

describe('NumberValue count-up and the viewport', () => {
  /** Puts every element's box at `top` (40 px high, 100 px wide) for the body. */
  async function withBoxAt(top: number, body: () => void | Promise<void>): Promise<void> {
    const original = Object.getOwnPropertyDescriptor(Element.prototype, 'getBoundingClientRect')!;
    Object.defineProperty(Element.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: (): DOMRect => ({
        x: 0,
        y: top,
        top,
        left: 0,
        right: 100,
        bottom: top + 40,
        width: 100,
        height: 40,
        toJSON: () => ({}),
      }),
    });
    try {
      await withRichMotion(body);
    } finally {
      Object.defineProperty(Element.prototype, 'getBoundingClientRect', original);
    }
  }

  function wheelDigits(container: HTMLElement): string {
    return [...container.querySelectorAll<HTMLElement>('[aria-hidden="true"] span')]
      .filter((node) => node.children.length === 0 && node.style.visibility === 'visible')
      .map((node) => node.textContent)
      .join('');
  }

  it('a number below the fold when its value first shows appears at its value: no zeros wait for a scroll', async () => {
    await withBoxAt(window.innerHeight + 600, () => {
      const view = mount(createElement(NumberValue, { text: '637' }));
      expect(wheelDigits(view.container)).toBe('637');
    });
  });

  it('a value that arrives after mount below the fold appears at its value too', async () => {
    await withBoxAt(window.innerHeight + 600, () => {
      const view = mount(createElement(NumberValue, { text: null }));
      view.rerender(createElement(NumberValue, { text: '637' }));
      expect(wheelDigits(view.container)).toBe('637');
      view.rerender(createElement(NumberValue, { text: '640' }));
      expect(leavesWithText(view.container, '640')).toHaveLength(1);
    });
  });

  it('a number below the fold whose wheels first draw when motion turns on appears at its value', async () => {
    await withBoxAt(window.innerHeight + 600, () => {
      skipAnimations(true);
      const view = mount(createElement(NumberValue, { text: '637' }));
      expect(wheelDigits(view.container)).toBe('');
      expect(leavesWithText(view.container, '637')).toHaveLength(1);
      skipAnimations(false);
      view.rerender(createElement(NumberValue, { text: '637' }));
      expect(wheelDigits(view.container)).toBe('637');
    });
  });

  it('a number on screen when its value first shows still counts up from zero', async () => {
    await withBoxAt(100, () => {
      const view = mount(createElement(NumberValue, { text: '637' }));
      expect(wheelDigits(view.container)).toBe('000');
      expect(leavesWithText(view.container, '637')).toHaveLength(1);
    });
  });
});
