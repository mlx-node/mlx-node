/**
 * @vitest-environment happy-dom
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { WaitOrb, type WaitOrbState } from '../ui/src/components/arc-x/wait-orb/wait-orb.js';
import { skipAnimations } from './render.js';

const ARC_X = join(import.meta.dirname, '../ui/src/components/arc-x');
const STATES: readonly WaitOrbState[] = ['hold', 'load', 'think'];

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

/** One draw: how many dots it painted and the strongest dot alpha. */
interface Draw {
  dots: number;
  alpha: number;
}

/** A 2D context that records each draw (a clear starts one). */
function stubContext(): { draws: Draw[] } {
  const draws: Draw[] = [];
  let alpha = 1;
  const ctx = {
    setTransform: () => {},
    clearRect: () => draws.push({ dots: 0, alpha: 0 }),
    beginPath: () => {},
    arc: () => {
      const draw = draws[draws.length - 1]!;
      draw.dots += 1;
      draw.alpha = Math.max(draw.alpha, alpha);
    },
    fill: () => {},
    fillStyle: '',
    get globalAlpha(): number {
      return alpha;
    },
    set globalAlpha(value: number) {
      alpha = value;
    },
  };
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as RenderingContext);
  return { draws };
}

/** rAF under the test's hand: `tick(ms)` runs the waiting callbacks with that timestamp. */
function stubFrames(): { tick(ms: number): void; requested: () => number; cancelled: number[] } {
  let queue: Array<{ id: number; run: FrameRequestCallback }> = [];
  let next = 0;
  const cancelled: number[] = [];
  vi.stubGlobal('requestAnimationFrame', (run: FrameRequestCallback) => {
    next += 1;
    queue.push({ id: next, run });
    return next;
  });
  vi.stubGlobal('cancelAnimationFrame', (id: number) => {
    cancelled.push(id);
    queue = queue.filter((entry) => entry.id !== id);
  });
  return {
    tick: (ms) => {
      const due = queue;
      queue = [];
      for (const entry of due) entry.run(ms);
    },
    requested: () => next,
    cancelled,
  };
}

/**
 * Every observed element is in view at once. motion's `inView` takes an element only when it is an `EventTarget`, and
 * the test's global `EventTarget` is Node's, not the DOM's.
 */
function stubInView(): void {
  vi.stubGlobal('EventTarget', Object.getPrototypeOf(Node.prototype).constructor);
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      private readonly callback: IntersectionObserverCallback;
      constructor(callback: IntersectionObserverCallback) {
        this.callback = callback;
      }
      observe(target: Element): void {
        this.callback(
          [{ target, isIntersecting: true } as unknown as IntersectionObserverEntry],
          this as unknown as IntersectionObserver,
        );
      }
      unobserve(): void {}
      disconnect(): void {}
      takeRecords(): IntersectionObserverEntry[] {
        return [];
      }
    },
  );
}

async function withMotion(run: () => void): Promise<void> {
  const previous = skipAnimations(false);
  try {
    run();
  } finally {
    skipAnimations(previous);
  }
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('WaitOrb', () => {
  it('is one 20 px canvas per state, aria-hidden with no label, an image with one', () => {
    for (const state of STATES) {
      const view = mount(createElement(WaitOrb, { state }));
      const canvas = view.container.querySelector<HTMLCanvasElement>('canvas')!;
      expect(view.container.children).toHaveLength(1);
      expect(canvas.getAttribute('data-wait-orb')).toBe(state);
      expect(canvas.style.width).toBe('20px');
      expect(canvas.style.height).toBe('20px');
      expect(canvas.getAttribute('aria-hidden')).toBe('true');
      expect(canvas.hasAttribute('role')).toBe(false);
      view.rerender(createElement(WaitOrb, { state, label: 'Loading the model' }));
      expect(canvas.getAttribute('role')).toBe('img');
      expect(canvas.getAttribute('aria-label')).toBe('Loading the model');
      expect(canvas.hasAttribute('aria-hidden')).toBe(false);
      view.unmount();
    }
  });

  it('under static motion draws one still frame and asks for no animation frame', () => {
    const { draws } = stubContext();
    const frames = vi.spyOn(globalThis, 'requestAnimationFrame');
    stubInView();
    for (const state of STATES) {
      const view = mount(createElement(WaitOrb, { state }));
      view.rerender(createElement(WaitOrb, { state: state === 'think' ? 'hold' : 'think' }));
      view.unmount();
    }
    expect(frames).not.toHaveBeenCalled();
    expect(draws.length).toBe(STATES.length * 2);
    expect(draws.every((draw) => draw.dots > 0)).toBe(true);
  });

  it('with motion draws 30 times per second: ticks at 0, 10, 20, 34, 50, 67 ms are 3 draws', async () => {
    const { draws } = stubContext();
    const frames = stubFrames();
    stubInView();
    await withMotion(() => {
      const view = mount(createElement(WaitOrb, { state: 'think' }));
      draws.length = 0;
      for (const ms of [0, 10, 20, 34, 50, 67]) frames.tick(ms);
      expect(draws).toHaveLength(3);
      expect(draws.every((draw) => draw.dots === 208 && draw.alpha > 0.5)).toBe(true);
      view.unmount();
    });
  });

  it('a state change fades the old shape out, then the new one in', async () => {
    const { draws } = stubContext();
    const frames = stubFrames();
    stubInView();
    vi.spyOn(performance, 'now').mockReturnValue(1_000);
    await withMotion(() => {
      const view = mount(createElement(WaitOrb, { state: 'think' }));
      frames.tick(1_000);
      view.rerender(createElement(WaitOrb, { state: 'hold' }));
      draws.length = 0;
      for (let ms = 1_000; ms <= 1_400; ms += 1000 / 30) frames.tick(ms);
      const shapes = draws.map((draw) => draw.dots);
      const old = draws.filter((draw) => draw.dots === 208);
      const fresh = draws.filter((draw) => draw.dots === 120);
      expect(old.length).toBeGreaterThan(1);
      expect(fresh.length).toBeGreaterThan(1);
      expect(shapes.indexOf(120)).toBe(old.length);
      const fading = old.map((draw) => draw.alpha);
      expect(fading).toEqual(fading.toSorted((a, b) => b - a));
      expect(old[old.length - 1]!.alpha).toBeLessThan(old[0]!.alpha);
      expect(fresh[0]!.alpha).toBeLessThan(fresh[fresh.length - 1]!.alpha);
      view.unmount();
    });
  });

  it('cancels its frame on unmount, and starts no loop while the page is hidden', async () => {
    stubContext();
    const frames = stubFrames();
    stubInView();
    await withMotion(() => {
      const view = mount(createElement(WaitOrb, { state: 'load' }));
      expect(frames.requested()).toBe(1);
      frames.tick(0);
      const last = frames.requested();
      view.unmount();
      expect(frames.cancelled).toEqual([last]);

      const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
      const before = frames.requested();
      const quiet = mount(createElement(WaitOrb, { state: 'load' }));
      expect(frames.requested()).toBe(before);
      hidden.mockReturnValue(false);
      act(() => {
        document.dispatchEvent(new Event('visibilitychange'));
      });
      expect(frames.requested()).toBe(before + 1);
      quiet.unmount();
    });
  });

  it('takes every timing number from motionX.orb', () => {
    const source = readFileSync(join(ARC_X, 'wait-orb/wait-orb.tsx'), 'utf8');
    for (const name of ['motionX.orb.rate', 'motionX.orb.still', 'motionX.orb.holdPace']) {
      expect(source).toContain(name);
    }
    const motion = readFileSync(join(ARC_X, 'motion/motion.ts'), 'utf8');
    const at = motion.indexOf('  orb: {');
    expect(at).toBeGreaterThan(-1);
    const block = motion.slice(at, motion.indexOf('  },', at));
    for (const line of ['rate: 30,', 'still: 0.6,', 'holdPace: 0.45,']) expect(block).toContain(line);
  });
});
