/**
 * @vitest-environment happy-dom
 */

import { act, createElement, useRef, type ReactElement, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { LiveDot } from '../ui/src/components/arc-x/live-dot/live-dot.js';
import { LiveLabel } from '../ui/src/components/arc-x/live-label/live-label.js';
import { motionX } from '../ui/src/components/arc-x/motion/motion.js';
import { MotionGate, useEntrance, useRichMotion } from '../ui/src/components/arc-x/motion/use-entrance.js';
import { useStaticMotion } from '../ui/src/components/arc-x/motion/use-static-motion.js';
import { motionTokens } from '../ui/src/components/arc/motion-tokens.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { skipAnimations } from './render.js';

interface Mounted {
  container: HTMLElement;
  rerender(element: ReactElement): void;
  unmount(): void;
}

function mount(element: ReactElement): Mounted {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  return {
    container,
    rerender: (next) => act(() => root.render(next)),
    unmount: () => {
      act(() => root.unmount());
      container.remove();
    },
  };
}

function StaticProbe(): ReactElement {
  return createElement('span', { 'data-probe': 'static' }, String(useStaticMotion()));
}

function RichProbe(): ReactElement {
  return createElement('span', { 'data-probe': 'rich' }, String(useRichMotion()));
}

/** A figure with one mark; prints its entrance state on the figure. */
function EntranceProbe({ marks = true }: { marks?: boolean }): ReactElement {
  const ref = useRef<HTMLElement | null>(null);
  const state = useEntrance(ref);
  return createElement(
    'figure',
    { ref, 'data-entrance': state },
    marks ? createElement('span', { 'data-entrance-last': '' }, 'mark') : null,
  );
}

function gate(open: boolean, children: ReactNode): ReactElement {
  return createElement(MotionGate, { open, children });
}

/** Runs `body` with animations NOT skipped, restoring the harness flag after. */
async function withRichMotion(body: () => void | Promise<void>): Promise<void> {
  const previous = skipAnimations(false);
  try {
    await body();
  } finally {
    skipAnimations(previous);
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('motionX', () => {
  it('reuses the Arc free springs where they exist', () => {
    expect(motionX.spring.expand).toBe(motionTokens.spring.snappy);
    expect(motionX.spring.morph).toBe(motionTokens.spring.morph);
    expect(motionX.spring.reshape).toBe(motionTokens.spring.smooth);
  });

  it('never overshoots', () => {
    for (const [name, spring] of Object.entries(motionX.spring)) expect(spring.bounce, name).toBe(0);
  });

  it('keeps the stagger budget at the considered duration', () => {
    expect(motionX.staggerBudget).toBe(motionTokens.duration.considered);
  });
});

describe('useStaticMotion and the harness', () => {
  it('is true once render.ts is imported', () => {
    const view = mount(createElement(StaticProbe));
    expect(view.container.textContent).toBe('true');
    view.unmount();
  });

  it('is false with animations on and no reduced-motion request', async () => {
    await withRichMotion(() => {
      const view = mount(createElement(StaticProbe));
      expect(view.container.textContent).toBe('false');
      view.unmount();
    });
  });
});

describe('useRichMotion and MotionGate', () => {
  it('is false under the harness, even inside an open gate', () => {
    const view = mount(gate(true, createElement(RichProbe)));
    expect(view.container.textContent).toBe('false');
    view.unmount();
  });

  it('with animations on: true in an open gate, false in a closed one and in an open gate inside it', async () => {
    await withRichMotion(() => {
      const view = mount(
        createElement(
          'div',
          null,
          gate(true, createElement(RichProbe)),
          gate(false, createElement(RichProbe)),
          gate(false, gate(true, createElement(RichProbe))),
          createElement(RichProbe),
        ),
      );
      const values = [...view.container.querySelectorAll('[data-probe="rich"]')].map((node) => node.textContent);
      expect(values).toEqual(['true', 'false', 'false', 'true']);
      view.unmount();
    });
  });
});

describe('useEntrance', () => {
  it('is done on the first render under the harness', () => {
    const view = mount(createElement(EntranceProbe));
    expect(view.container.querySelector('figure')?.getAttribute('data-entrance')).toBe('done');
    view.unmount();
  });

  it('is done at once inside a closed gate, with animations on', async () => {
    await withRichMotion(() => {
      const view = mount(gate(false, createElement(EntranceProbe)));
      expect(view.container.querySelector('figure')?.getAttribute('data-entrance')).toBe('done');
      view.unmount();
    });
  });

  it('observes the viewport, runs when on screen with a width, and ends on the last mark', async () => {
    const observers: Array<{ options: IntersectionObserverInit | undefined; fire(entry: object): void }> = [];
    class StubObserver {
      private readonly callback: IntersectionObserverCallback;
      constructor(callback: IntersectionObserverCallback, options?: IntersectionObserverInit) {
        this.callback = callback;
        observers.push({
          options,
          fire: (entry) => this.callback([entry as IntersectionObserverEntry], this as unknown as IntersectionObserver),
        });
      }
      observe(): void {}
      disconnect(): void {}
      unobserve(): void {}
      takeRecords(): IntersectionObserverEntry[] {
        return [];
      }
    }
    vi.stubGlobal('IntersectionObserver', StubObserver);

    await withRichMotion(() => {
      const view = mount(createElement(EntranceProbe));
      const figure = view.container.querySelector('figure');
      expect(figure?.getAttribute('data-entrance')).toBe('pending');
      expect(observers).toHaveLength(1);
      expect(observers[0].options).toBeDefined();
      expect(observers[0].options !== undefined && 'root' in observers[0].options).toBe(false);

      act(() => observers[0].fire({ isIntersecting: true, boundingClientRect: { width: 0 } }));
      expect(figure?.getAttribute('data-entrance')).toBe('pending');
      act(() => observers[0].fire({ isIntersecting: false, boundingClientRect: { width: 320 } }));
      expect(figure?.getAttribute('data-entrance')).toBe('pending');
      act(() => observers[0].fire({ isIntersecting: true, boundingClientRect: { width: 320 } }));
      expect(figure?.getAttribute('data-entrance')).toBe('run');

      act(() => {
        figure?.dispatchEvent(new Event('animationend', { bubbles: true }));
      });
      expect(figure?.getAttribute('data-entrance')).toBe('run');
      const mark = figure?.querySelector('[data-entrance-last]');
      act(() => {
        mark?.dispatchEvent(new Event('animationend', { bubbles: true }));
      });
      expect(figure?.getAttribute('data-entrance')).toBe('done');

      view.rerender(createElement(EntranceProbe));
      expect(figure?.getAttribute('data-entrance')).toBe('done');
      view.unmount();
    });
  });

  it('goes from run to done at once when there are no marks', async () => {
    const fire: Array<(entry: object) => void> = [];
    vi.stubGlobal(
      'IntersectionObserver',
      class {
        constructor(callback: IntersectionObserverCallback) {
          fire.push((entry) => callback([entry as IntersectionObserverEntry], this as unknown as IntersectionObserver));
        }
        observe(): void {}
        disconnect(): void {}
      },
    );
    await withRichMotion(() => {
      const view = mount(createElement(EntranceProbe, { marks: false }));
      act(() => fire[0]({ isIntersecting: true, boundingClientRect: { width: 100 } }));
      expect(view.container.querySelector('figure')?.getAttribute('data-entrance')).toBe('done');
      view.unmount();
    });
  });

  it('is done when IntersectionObserver does not exist', async () => {
    vi.stubGlobal('IntersectionObserver', undefined);
    await withRichMotion(() => {
      const view = mount(createElement(EntranceProbe));
      expect(view.container.querySelector('figure')?.getAttribute('data-entrance')).toBe('done');
      view.unmount();
    });
  });
});

describe('LiveLabel', () => {
  it('under the harness is solid text: the label, no busy mark', () => {
    const view = mount(createElement(LiveLabel, { active: true, as: 'p', children: 'Downloading' }));
    const root = view.container.firstElementChild;
    expect(root?.tagName).toBe('P');
    expect(root?.textContent).toBe('Downloading');
    expect(view.container.querySelector('[aria-busy]')).toBeNull();
    view.rerender(createElement(LiveLabel, { active: true, as: 'p', children: 'Verifying' }));
    expect(view.container.firstElementChild?.textContent).toBe('Verifying');
    view.unmount();
  });

  it('with animations on, shimmers while active and settles when it is not', async () => {
    await withRichMotion(() => {
      const view = mount(createElement(LiveLabel, { active: true, children: 'Loading' }));
      const root = view.container.firstElementChild;
      expect(root?.textContent).toBe('Loading');
      expect(root?.getAttribute('data-state')).toBe('active');
      view.rerender(createElement(LiveLabel, { active: false, children: 'Loading' }));
      expect(view.container.firstElementChild?.getAttribute('data-state')).toBe('idle');
      expect(view.container.firstElementChild?.textContent).toBe('Loading');
      view.unmount();
    });
  });

  it('gives the painted label room for descenders without a change of the line height', async () => {
    const rules = cssRules('components/arc-x/live-label/live-label.module.css');
    expect(cssValue(rules, '.shimmer > span > span', 'padding-block')).toBe('0.25em');
    expect(cssValue(rules, '.shimmer > span > span', 'margin-block')).toBe('-0.25em');
    await withRichMotion(() => {
      const view = mount(createElement(LiveLabel, { active: false, className: 'own', children: 'Running' }));
      const root = view.container.firstElementChild;
      expect(root?.classList.contains('own')).toBe(true);
      expect(root?.classList.length).toBe(3);
      const painted = root?.querySelector(':scope > span > span');
      expect(painted?.textContent).toBe('Running');
      expect(painted?.childElementCount).toBe(0);
      expect(root?.querySelectorAll('span').length).toBe(2);
      view.unmount();
    });
  });

  it('inside a closed gate, with animations on, is solid text', async () => {
    await withRichMotion(() => {
      const view = mount(gate(false, createElement(LiveLabel, { active: true, children: 'Prefill' })));
      expect(view.container.textContent).toBe('Prefill');
      expect(view.container.querySelector('[data-state]')).toBeNull();
      view.unmount();
    });
  });
});

describe('LiveDot shimmer', () => {
  it('reads the same label with and without shimmer, and nothing is busy under the harness', () => {
    const plain = mount(createElement(LiveDot, { tone: 'warning', label: 'Starting', showLabel: true }));
    const shimmering = mount(
      createElement(LiveDot, { tone: 'warning', label: 'Starting', showLabel: true, shimmer: true }),
    );
    expect(plain.container.textContent).toBe('Starting');
    expect(shimmering.container.textContent).toBe('Starting');
    expect(document.querySelector('[aria-busy]')).toBeNull();
    plain.unmount();
    shimmering.unmount();
  });

  it('without a visible label is an image named by the label', () => {
    const view = mount(createElement(LiveDot, { tone: 'success', label: 'Running', pulse: true }));
    const root = view.container.firstElementChild;
    expect(root?.getAttribute('role')).toBe('img');
    expect(root?.getAttribute('aria-label')).toBe('Running');
    expect(view.container.textContent).toBe('');
    view.unmount();
  });
});
