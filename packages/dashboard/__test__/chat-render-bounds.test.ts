/**
 * @vitest-environment happy-dom
 */

import { act, createElement, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactElement } from 'react';
import { createPortal } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { ChatFrame } from '../ui/src/components/arc-x/chat-frame/chat-frame.js';
import { ReasoningBlock } from '../ui/src/components/arc-x/reasoning-block/reasoning-block.js';
import { ChatThread } from '../ui/src/components/chat-thread.js';
import { useStickToBottom, type StickToBottom } from '../ui/src/lib/stick-to-bottom.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import { formatSearchText } from '../ui/src/lib/web-result-text.js';
import { conversationHosts } from '../ui/src/lib/web-tools.js';
import { answer, directSearch, end, user } from './helpers/chat-thread-fixtures.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { observeWidth, skipAnimations } from './render.js';

let mounted: { root: Root; container: HTMLElement } | undefined;

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  mounted = { root, container };
  return container;
}

function rerender(element: ReactElement): void {
  act(() => mounted!.root.render(element));
}

afterEach(() => {
  skipAnimations(true);
  vi.restoreAllMocks();
  if (mounted === undefined) return;
  const { root, container } = mounted;
  act(() => root.unmount());
  container.remove();
  mounted = undefined;
});

const RLO = String.fromCharCode(0x202e);
const BIDI = [0x202a, 0x202b, 0x202c, 0x202d, 0x202e, 0x2066, 0x2067, 0x2068, 0x2069, 0x200e, 0x200f].map((code) =>
  String.fromCharCode(code),
);
const FORBIDDEN_TAGS = new Set([
  'script',
  'img',
  'iframe',
  'object',
  'embed',
  'style',
  'link',
  'meta',
  'base',
  'form',
  'frame',
  'video',
  'audio',
  'source',
  'foreignobject',
  'math',
]);

/** A line of about 1 MB with a hostile piece every few KB. */
function megabyteLine(): string {
  const filler = 'lorem ipsum dolor sit amet '.repeat(150);
  const parts: string[] = [];
  let size = 0;
  for (let i = 0; size < 1_000_000; i += 1) {
    const piece = `${filler}**[x${i}](https://evil.example/${i} "t") <b onclick=alert(${i})>b</b> [y](javascript:alert(${i})) *${RLO}`;
    parts.push(piece);
    size += piece.length;
  }
  return parts.join('');
}

function hostileTurn(): TranscriptEntry[] {
  const results = formatSearchText('exa', [
    {
      title: `Release notes ${RLO}txt.exe`,
      url: 'https://ok.example/notes',
      date: null,
      snippet: '<script>alert(1)</script>',
    },
  ]);
  const tool: TranscriptEntry = {
    kind: 'tool',
    ts: 1,
    id: 'g1',
    name: 'other_tool',
    title: `Run ${RLO}gpj.exe <img src=x onerror=alert(1)>`,
    status: 'completed',
    input: '<script>alert(1)</script>',
    output: '<iframe src="https://evil.example/frame"></iframe>',
    endTs: 2,
  };
  const text = [
    'Here <script>alert(1)</script> and <img src=x onerror=alert(1)> and <a href="https://evil.example/raw">raw</a>.',
    `[click](javascript:alert(1)) [here](https://evil.example/?q=secret) ![pix](https://evil.example/p.png) <https://evil.example/auto> https://evil.example/bare [notes](https://ok.example/notes) [1] [2] file ${RLO}gpj.exe`,
    '',
    megabyteLine(),
    '',
    'Sources:',
    '[1] Release notes - https://ok.example/notes',
    '[2] Official guide - https://evil.example/guide',
  ].join('\n');
  return [user('Find the release notes'), directSearch(`notes ${RLO}gpj`, results), tool, answer(text)];
}

/** Every element and attribute came from the components, not from the text; anchors go only to allowed hosts. */
function expectInert(container: HTMLElement, allowed: ReadonlySet<string>): void {
  const wrong: string[] = [];
  for (const element of container.querySelectorAll('*')) {
    const tag = element.tagName.toLowerCase();
    if (FORBIDDEN_TAGS.has(tag)) wrong.push(`<${tag}>`);
    for (const attribute of element.attributes) {
      const { name, value } = attribute;
      if (name.startsWith('on') || name === 'src' || name === 'srcdoc') wrong.push(`${tag}[${name}]`);
      // A name or tooltip may quote the text; any other attribute holding it came from it.
      const quotes = name === 'aria-label' || name === 'title';
      if (
        (!quotes && (value.includes('javascript:') || value.includes('alert('))) ||
        BIDI.some((m) => value.includes(m))
      )
        wrong.push(`${tag}[${name}=${value.slice(0, 60)}]`);
    }
  }
  for (const anchor of container.querySelectorAll('a')) {
    const href = anchor.getAttribute('href') ?? '';
    if (href.startsWith('/') && !href.startsWith('//')) continue;
    const url = URL.canParse(href) ? new URL(href) : null;
    if (url === null || (url.protocol !== 'https:' && url.protocol !== 'http:') || !allowed.has(url.hostname))
      wrong.push(`a[href=${href.slice(0, 60)}]`);
  }
  expect(wrong).toEqual([]);
}

describe('a hostile turn', () => {
  it('renders as text: no element, attribute or anchor comes from it, and it renders in time', () => {
    const entries = hostileTurn();
    const allowed = conversationHosts(entries);
    expect([...allowed]).toEqual(['ok.example']);
    const thread = (finished: boolean): ReactElement =>
      createElement(
        MemoryRouter,
        null,
        createElement(ChatThread, {
          mode: 'chat',
          entries: finished ? [...entries, end()] : entries,
          turns: [],
          turn: finished ? null : { turnId: 't1', phase: 'running', startedAt: 0, wait: null },
          lastEnd: null,
        }),
      );

    let started = performance.now();
    const container = mount(thread(false));
    expect(performance.now() - started).toBeLessThan(5_000);
    expectInert(container, allowed);

    started = performance.now();
    rerender(thread(true));
    expect(performance.now() - started).toBeLessThan(5_000);
    expectInert(container, allowed);

    const text = container.textContent ?? '';
    expect(text.length).toBeGreaterThan(900_000);
    expect(text).toContain('<script>alert(1)</script>');
    expect(text).toContain('here (evil.example)');
    for (const mark of BIDI) expect(text.includes(mark)).toBe(false);
    const sourceButtons = [...container.querySelectorAll('button')].filter((b) =>
      (b.getAttribute('aria-label') ?? '').startsWith('Source '),
    );
    expect(sourceButtons.map((b) => b.getAttribute('aria-label'))).toEqual([
      'Source 1: Release notes txt.exe, ok.example',
    ]);
  }, 60_000);
});

describe('streaming cost', () => {
  it('a live thinking tail draws only the end of the text', () => {
    skipAnimations(false);
    const text = 'thought '.repeat(125_000);
    const container = mount(
      createElement(ReasoningBlock, { text, label: 'Thinking', live: true, startTs: 0, asOfMs: 1_000 }),
    );
    const longest = Math.max(...[...container.querySelectorAll('div')].map((node) => node.textContent?.length ?? 0));
    expect(longest).toBeGreaterThan(1_000);
    expect(longest).toBeLessThan(10_000);
  });
});

describe('jump to latest', () => {
  interface ScrollRig {
    root: HTMLDivElement;
    attachRoot: (node: HTMLDivElement) => void;
    observers: Array<() => void>;
    notify: (target: Element) => void;
    setBox: (next: { width: number; height: number }) => void;
    setScrollHeight: (next: number) => void;
    setScrollTopRaw: (next: number) => void;
    rafCalls: () => number;
    pendingFrames: () => number;
    flushFrame: (now: number) => void;
    cleanup: () => void;
  }

  function createScrollRig(initial: { width: number; height: number; clientHeight?: number }): ScrollRig {
    const observers: Array<() => void> = [];
    const targetObservers = new Map<Element, Array<() => void>>();
    vi.stubGlobal(
      'ResizeObserver',
      class {
        private readonly callback: () => void;
        constructor(callback: () => void) {
          this.callback = callback;
          observers.push(callback);
        }
        observe(target: Element): void {
          const list = targetObservers.get(target);
          if (list === undefined) targetObservers.set(target, [this.callback]);
          else list.push(this.callback);
        }
        disconnect(): void {
          for (const [target, list] of targetObservers) {
            const next = list.filter((cb) => cb !== this.callback);
            if (next.length === 0) targetObservers.delete(target);
            else targetObservers.set(target, next);
          }
        }
      },
    );

    let nextFrameId = 1;
    let totalRafCalls = 0;
    const queued = new Map<number, FrameRequestCallback>();
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback): number => {
      totalRafCalls += 1;
      const id = nextFrameId;
      nextFrameId += 1;
      queued.set(id, cb);
      return id;
    });
    vi.stubGlobal('cancelAnimationFrame', (id: number): void => {
      queued.delete(id);
    });

    let box = { width: initial.width, height: initial.height };
    let scrollHeight = initial.height;
    const clientHeight = initial.clientHeight ?? 500;
    let scrollTop = 0;

    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(
      () => ({ ...box, top: 0, left: 0, right: box.width, bottom: box.height, x: 0, y: 0 }) as DOMRect,
    );

    const attachRoot = (node: HTMLDivElement): void => {
      Object.defineProperties(node, {
        clientHeight: { get: () => clientHeight, configurable: true },
        scrollHeight: { get: () => scrollHeight, configurable: true },
        scrollTop: {
          get: () => scrollTop,
          set: (value: number) => {
            const maxTop = Math.max(0, scrollHeight - clientHeight);
            scrollTop = Math.max(0, Math.min(maxTop, value));
          },
          configurable: true,
        },
      });
    };

    const root = document.createElement('div');
    attachRoot(root);
    document.body.appendChild(root);

    return {
      root,
      attachRoot,
      observers,
      notify: (target) => {
        for (const cb of targetObservers.get(target) ?? []) cb();
      },
      setBox: (next) => {
        box = next;
      },
      setScrollHeight: (next) => {
        scrollHeight = next;
        const maxTop = Math.max(0, scrollHeight - clientHeight);
        if (scrollTop > maxTop) scrollTop = maxTop;
      },
      setScrollTopRaw: (next) => {
        scrollTop = next;
      },
      rafCalls: () => totalRafCalls,
      pendingFrames: () => queued.size,
      flushFrame: (now: number) => {
        const batch = [...queued.entries()];
        queued.clear();
        for (const [, cb] of batch) cb(now);
      },
      cleanup: () => {
        vi.unstubAllGlobals();
        root.remove();
      },
    };
  }

  it('a wider or narrower column is not new content', () => {
    const rig = createScrollRig({ width: 600, height: 400 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'resize');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.observers).toHaveLength(1);
      act(() => {
        rig.root.dispatchEvent(new WheelEvent('wheel', { deltaY: -40 }));
      });
      expect(stick!.pinned).toBe(false);
      rig.setBox({ width: 400, height: 700 });
      act(() => rig.observers[0]!());
      expect(stick!.fresh).toBe(false);
      rig.setBox({ width: 400, height: 900 });
      act(() => rig.observers[0]!());
      expect(stick!.fresh).toBe(true);
    } finally {
      rig.cleanup();
    }
  });

  it('smoothly chases content growth over multiple rAF frames, tracks mid-chase growth without restarting, and stops at target', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'smooth-chase');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);
      expect(rig.rafCalls()).toBe(0);

      rig.setBox({ width: 600, height: 1_400 });
      rig.setScrollHeight(1_400);
      act(() => rig.observers[0]!());

      // Does not teleport in the same tick; schedules one chase frame.
      expect(rig.root.scrollTop).toBe(500);
      expect(rig.pendingFrames()).toBe(1);

      act(() => rig.flushFrame(16));
      const afterFirst = rig.root.scrollTop;
      expect(afterFirst).toBeGreaterThan(500);
      expect(afterFirst).toBeLessThan(900);
      expect(rig.pendingFrames()).toBe(1);

      // More content arrives mid-chase: active rAF continues without scheduling a duplicate loop.
      const callsBeforeSecondGrow = rig.rafCalls();
      rig.setBox({ width: 600, height: 1_700 });
      rig.setScrollHeight(1_700);
      act(() => rig.observers[0]!());
      expect(rig.rafCalls()).toBe(callsBeforeSecondGrow);
      expect(rig.pendingFrames()).toBe(1);

      let prev = afterFirst;
      let t = 16;
      let steps = 0;
      while (rig.pendingFrames() > 0 && steps < 100) {
        t += 16;
        steps += 1;
        act(() => rig.flushFrame(t));
        expect(rig.root.scrollTop).toBeGreaterThanOrEqual(prev);
        prev = rig.root.scrollTop;
      }

      expect(steps).toBeGreaterThan(3);
      expect(rig.pendingFrames()).toBe(0);
      expect(rig.root.scrollTop).toBe(1_200);
      expect(stick!.pinned).toBe(true);
    } finally {
      rig.cleanup();
    }
  });

  it('cancels an active chase and unpins on upward wheel or backward scrollbar scroll', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'chase-cancel');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      rig.setBox({ width: 600, height: 1_400 });
      rig.setScrollHeight(1_400);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const midWheel = rig.root.scrollTop;
      expect(midWheel).toBeGreaterThan(500);
      expect(rig.pendingFrames()).toBe(1);

      // Upward wheel immediately cancels chase and unpins.
      act(() => {
        rig.root.dispatchEvent(new WheelEvent('wheel', { deltaY: -40 }));
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);
      expect(rig.root.scrollTop).toBe(midWheel);

      // Re-pin and start another chase, then drag scrollbar backward while content has not shrunk.
      act(() => stick!.pin());
      expect(stick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);
      act(() => rig.flushFrame(32));
      const midScroll = rig.root.scrollTop;
      expect(midScroll).toBeGreaterThan(midWheel);
      expect(rig.pendingFrames()).toBe(1);

      rig.setScrollTopRaw(midScroll - 40);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);
      expect(rig.root.scrollTop).toBe(midScroll - 40);
    } finally {
      rig.cleanup();
    }
  });

  it('writes directly with zero rAF frames under reduced motion', () => {
    skipAnimations(true);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'reduced-motion');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);

      rig.setBox({ width: 600, height: 1_400 });
      rig.setScrollHeight(1_400);
      act(() => rig.observers[0]!());

      expect(rig.root.scrollTop).toBe(900);
      expect(rig.rafCalls()).toBe(0);
      expect(rig.pendingFrames()).toBe(0);
      expect(stick!.pinned).toBe(true);
    } finally {
      rig.cleanup();
    }
  });

  it('snaps immediately without rAF chase when the column width changes', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'width-snap');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);

      rig.setBox({ width: 420, height: 1_400 });
      rig.setScrollHeight(1_400);
      act(() => rig.observers[0]!());

      expect(rig.root.scrollTop).toBe(900);
      expect(rig.rafCalls()).toBe(0);
      expect(rig.pendingFrames()).toBe(0);
      expect(stick!.pinned).toBe(true);
    } finally {
      rig.cleanup();
    }
  });

  it('does not unpin when content shrinks and clamps scrollTop near the end during a chase', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'shrink-clamp');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      rig.setBox({ width: 600, height: 1_600 });
      rig.setScrollHeight(1_600);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      act(() => rig.flushFrame(32));
      const beforeShrink = rig.root.scrollTop;
      expect(beforeShrink).toBeGreaterThan(650);

      // Content shrinks so the browser clamps scrollTop down to the new end (below lastWritten - 2).
      rig.setBox({ width: 600, height: 1_120 });
      rig.setScrollHeight(1_120);
      expect(rig.root.scrollTop).toBe(620);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(true);

      act(() => rig.flushFrame(48));
      expect(stick!.pinned).toBe(true);
      expect(rig.root.scrollTop).toBe(620);
      expect(rig.pendingFrames()).toBe(0);
    } finally {
      rig.cleanup();
    }
  });

  it('keeps chasing when content shrinks above the viewport far from the end and restores overflowAnchor', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'shrink-anchor-far');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.style.overflowAnchor).toBe('');

      // Grow by 800 px so the chase starts far from the end (> PIN_SLACK).
      rig.setBox({ width: 600, height: 1_800 });
      rig.setScrollHeight(1_800);
      act(() => rig.observers[0]!());
      expect(rig.root.style.overflowAnchor).toBe('none');
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(1_300 - midTop).toBeGreaterThan(500);

      // Simulate a 30 px collapse above the viewport with matching 30 px scroll-anchoring adjustment while still > 500 px from end.
      rig.setBox({ width: 600, height: 1_770 });
      rig.setScrollHeight(1_770);
      rig.setScrollTopRaw(midTop - 30);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      let t = 16;
      let steps = 0;
      while (rig.pendingFrames() > 0 && steps < 100) {
        t += 16;
        steps += 1;
        act(() => rig.flushFrame(t));
      }
      expect(stick!.pinned).toBe(true);
      expect(rig.root.scrollTop).toBe(1_270);
      expect(rig.root.style.overflowAnchor).toBe('');
    } finally {
      rig.cleanup();
    }
  });

  function dispatchTouch(
    target: EventTarget,
    type: 'touchstart' | 'touchmove' | 'touchend' | 'touchcancel',
    touches: ReadonlyArray<{ identifier: number; clientX: number; clientY: number; target?: EventTarget }>,
  ): void {
    const event = new Event(type, { bubbles: true, cancelable: true });
    Object.defineProperty(event, 'touches', { value: touches });
    target.dispatchEvent(event);
  }

  it('cancels an active chase and unpins on upward touch swipe even with a simultaneous 30px shrink', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'touch-up-with-shrink');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      rig.setBox({ width: 600, height: 1_800 });
      rig.setScrollHeight(1_800);
      act(() => rig.observers[0]!());
      expect(rig.pendingFrames()).toBe(1);
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(midTop).toBeGreaterThan(500);
      expect(rig.pendingFrames()).toBe(1);

      // Drag finger downward (scrolling content upward) while content shrinks by the same 30px.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300 }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 330 }]);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);
      expect(rig.root.style.overflowAnchor).toBe('');

      rig.setBox({ width: 600, height: 1_770 });
      rig.setScrollHeight(1_770);
      rig.setScrollTopRaw(midTop - 30);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
        rig.flushFrame(32);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);
      expect(rig.root.scrollTop).toBe(midTop - 30);
    } finally {
      rig.cleanup();
    }
  });

  it('ignores horizontal touch swipes and downward touch scrolls during an active chase', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'touch-non-up');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      rig.setBox({ width: 600, height: 1_800 });
      rig.setScrollHeight(1_800);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(rig.pendingFrames()).toBe(1);

      // Horizontal swipe (dx = 60, dy = 4) does not unpin or cancel the chase.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300 }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 260, clientY: 304 }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      // Downward touch scroll (clientY decreasing, scrolling content toward bottom) does not unpin or cancel.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 2, clientX: 200, clientY: 300 }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 2, clientX: 200, clientY: 240 }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      act(() => rig.flushFrame(32));
      expect(rig.root.scrollTop).toBeGreaterThan(midTop);
      expect(stick!.pinned).toBe(true);
    } finally {
      rig.cleanup();
    }
  });

  it('does not unpin when touching and dragging inside an editable element', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    const textarea = document.createElement('textarea');
    rig.root.appendChild(textarea);
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'touch-editable');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      rig.setBox({ width: 600, height: 1_800 });
      rig.setScrollHeight(1_800);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(rig.pendingFrames()).toBe(1);

      act(() => {
        dispatchTouch(textarea, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: textarea }]);
        dispatchTouch(textarea, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 360, target: textarea }]);
        dispatchTouch(textarea, 'touchend', []);
      });
      expect(stick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      act(() => rig.flushFrame(32));
      expect(rig.root.scrollTop).toBeGreaterThan(midTop);
      expect(stick!.pinned).toBe(true);
    } finally {
      rig.cleanup();
    }
  });

  it('keeps touch release across upward and shrink-clamped scroll events within PIN_SLACK until scrolling back down', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'touch-short-up-within-slack');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);

      // Start an active chase within PIN_SLACK (48px) of the end.
      rig.setBox({ width: 600, height: 1_040 });
      rig.setScrollHeight(1_040);
      act(() => rig.observers[0]!());
      expect(rig.pendingFrames()).toBe(1);
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(midTop).toBeGreaterThan(500);
      expect(1_040 - 500 - midTop).toBeLessThanOrEqual(48);
      expect(rig.pendingFrames()).toBe(1);

      // Short 20px upward touch swipe + 30px content shrink + 20px scrollTop drop, staying within 48px of the new end.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 320, target: rig.root }]);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      const swipedTop = midTop - 20;
      rig.setBox({ width: 600, height: 1_010 });
      rig.setScrollHeight(1_010);
      rig.setScrollTopRaw(swipedTop);
      expect(1_010 - 500 - swipedTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.observers[0]!();
        rig.root.dispatchEvent(new Event('scroll'));
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      // Post-touchend momentum upward frame still within PIN_SLACK keeps the release held.
      const momentumTop = swipedTop - 10;
      rig.setScrollTopRaw(momentumTop);
      expect(1_010 - 500 - momentumTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Subsequent content growth marks fresh without restarting chase or moving scrollTop.
      rig.setBox({ width: 600, height: 1_024 });
      rig.setScrollHeight(1_024);
      expect(1_024 - 500 - momentumTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.observers[0]!();
        rig.flushFrame(32);
      });
      expect(stick!.pinned).toBe(false);
      expect(stick!.fresh).toBe(true);
      expect(rig.pendingFrames()).toBe(0);
      expect(rig.root.scrollTop).toBe(momentumTop);

      // Scrolling back downward via a user gesture within PIN_SLACK clears the hold and cleanly re-pins.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 2, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 2, clientX: 200, clientY: 280, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      rig.setScrollTopRaw(510);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(true);
      expect(stick!.fresh).toBe(false);
    } finally {
      rig.cleanup();
    }
  });

  it('ignores upward touch swipes inside a contained nested scroller while letting the nested scroller release', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    const inner = document.createElement('div');
    inner.style.overflowY = 'auto';
    inner.style.overscrollBehavior = 'contain';
    let innerScrollHeight = 800;
    const innerClientHeight = 300;
    let innerScrollTop = 0;
    Object.defineProperties(inner, {
      clientHeight: { get: () => innerClientHeight, configurable: true },
      scrollHeight: { get: () => innerScrollHeight, configurable: true },
      scrollTop: {
        get: () => innerScrollTop,
        set: (value: number) => {
          const maxTop = Math.max(0, innerScrollHeight - innerClientHeight);
          innerScrollTop = Math.max(0, Math.min(maxTop, value));
        },
        configurable: true,
      },
    });
    const innerChild = document.createElement('div');
    inner.appendChild(innerChild);
    rig.root.appendChild(inner);

    const horiz = document.createElement('div');
    horiz.style.overflowX = 'auto';
    horiz.style.overflowY = 'hidden';
    horiz.style.overscrollBehaviorX = 'contain';
    const horizChild = document.createElement('div');
    horiz.appendChild(horizChild);
    rig.root.appendChild(horiz);

    let outerStick: StickToBottom | undefined;
    let innerStick: StickToBottom | undefined;
    function Probe(): ReactElement {
      outerStick = useStickToBottom(rig.root, 'nested-outer');
      innerStick = useStickToBottom(inner, 'nested-inner');
      return createElement(
        'div',
        null,
        createElement('div', { ref: outerStick.contentRef }),
        createElement('div', { ref: innerStick.contentRef }),
      );
    }
    try {
      mount(createElement(Probe));
      expect(rig.observers).toHaveLength(2);
      expect(rig.root.scrollTop).toBe(500);
      expect(inner.scrollTop).toBe(500);

      // Grow both outer and inner threads so both are actively chasing.
      rig.setBox({ width: 600, height: 1_800 });
      rig.setScrollHeight(1_800);
      innerScrollHeight = 1_200;
      act(() => {
        rig.observers[0]!();
        rig.observers[1]!();
      });
      expect(rig.pendingFrames()).toBe(2);

      act(() => rig.flushFrame(16));
      const outerMid = rig.root.scrollTop;
      const innerMid = inner.scrollTop;
      expect(outerMid).toBeGreaterThan(500);
      expect(innerMid).toBeGreaterThan(500);
      expect(rig.pendingFrames()).toBe(2);

      // Upward touch swipe inside the docked inner scroller releases only the inner scroller.
      act(() => {
        dispatchTouch(innerChild, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: innerChild }]);
        dispatchTouch(innerChild, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 340, target: innerChild }]);
        dispatchTouch(innerChild, 'touchend', []);
      });
      expect(innerStick!.pinned).toBe(false);
      expect(outerStick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      act(() => rig.flushFrame(32));
      const outerAfter1 = rig.root.scrollTop;
      expect(outerAfter1).toBeGreaterThan(outerMid);
      expect(inner.scrollTop).toBe(innerMid);
      expect(outerStick!.pinned).toBe(true);
      expect(innerStick!.pinned).toBe(false);

      // Even at inner.scrollTop === 0, overscroll-behavior: contain still prevents releasing the outer scroller.
      innerScrollTop = 0;
      act(() => {
        dispatchTouch(innerChild, 'touchstart', [{ identifier: 2, clientX: 200, clientY: 300, target: innerChild }]);
        dispatchTouch(innerChild, 'touchmove', [{ identifier: 2, clientX: 200, clientY: 340, target: innerChild }]);
        dispatchTouch(innerChild, 'touchend', []);
      });
      expect(outerStick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);
      act(() => rig.flushFrame(48));
      expect(rig.root.scrollTop).toBeGreaterThan(outerAfter1);

      // Horizontal-only overscroll-behavior-x: contain does NOT block vertical touch release of the outer scroller.
      act(() => {
        dispatchTouch(horizChild, 'touchstart', [{ identifier: 3, clientX: 200, clientY: 300, target: horizChild }]);
        dispatchTouch(horizChild, 'touchmove', [{ identifier: 3, clientX: 200, clientY: 340, target: horizChild }]);
        dispatchTouch(horizChild, 'touchend', []);
      });
      expect(outerStick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);
    } finally {
      rig.cleanup();
    }
  });

  it('ignores upward touch swipes inside an uncontained nested scroller while it can still scroll up, and releases once at scrollTop 0', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    const inner = document.createElement('div');
    inner.style.overflowY = 'auto';
    let innerScrollTop = 100;
    Object.defineProperties(inner, {
      clientHeight: { get: () => 200, configurable: true },
      scrollHeight: { get: () => 800, configurable: true },
      scrollTop: {
        get: () => innerScrollTop,
        set: (value: number) => {
          innerScrollTop = Math.max(0, Math.min(600, value));
        },
        configurable: true,
      },
    });
    const innerChild = document.createElement('div');
    inner.appendChild(innerChild);
    rig.root.appendChild(inner);

    let outerStick: StickToBottom | undefined;
    function Probe(): ReactElement {
      outerStick = useStickToBottom(rig.root, 'nested-uncontained-outer');
      return createElement('div', { ref: outerStick.contentRef });
    }
    try {
      mount(createElement(Probe));
      rig.setBox({ width: 600, height: 1_800 });
      rig.setScrollHeight(1_800);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const mid1 = rig.root.scrollTop;
      expect(mid1).toBeGreaterThan(500);
      expect(rig.pendingFrames()).toBe(1);

      // Starting with inner.scrollTop > 0: upward swipe scrolls the inner container, outer keeps chasing.
      act(() => {
        dispatchTouch(innerChild, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: innerChild }]);
        dispatchTouch(innerChild, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 340, target: innerChild }]);
        dispatchTouch(innerChild, 'touchend', []);
      });
      expect(outerStick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);
      act(() => rig.flushFrame(32));
      const mid2 = rig.root.scrollTop;
      expect(mid2).toBeGreaterThan(mid1);

      // Starting at inner.scrollTop === 0, scrolling inner down mid-gesture, then reversing up: outer still keeps chasing.
      innerScrollTop = 0;
      act(() => {
        dispatchTouch(innerChild, 'touchstart', [{ identifier: 2, clientX: 200, clientY: 300, target: innerChild }]);
        dispatchTouch(innerChild, 'touchmove', [{ identifier: 2, clientX: 200, clientY: 240, target: innerChild }]);
        innerScrollTop = 80;
        dispatchTouch(innerChild, 'touchmove', [{ identifier: 2, clientX: 200, clientY: 280, target: innerChild }]);
        dispatchTouch(innerChild, 'touchend', []);
      });
      expect(outerStick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);
      act(() => rig.flushFrame(48));
      expect(rig.root.scrollTop).toBeGreaterThan(mid2);

      // Upward swipe while inner.scrollTop === 0 chains to the outer scroller and releases it.
      innerScrollTop = 0;
      act(() => {
        dispatchTouch(innerChild, 'touchstart', [{ identifier: 3, clientX: 200, clientY: 300, target: innerChild }]);
        dispatchTouch(innerChild, 'touchmove', [{ identifier: 3, clientX: 200, clientY: 340, target: innerChild }]);
        dispatchTouch(innerChild, 'touchend', []);
      });
      expect(outerStick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);
    } finally {
      rig.cleanup();
    }
  });

  it('preserves touchHold across above-viewport CSS scroll anchoring within PIN_SLACK until explicit downward return', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'touch-anchor-within-slack');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);

      // Start an active chase within PIN_SLACK (48px) of the end.
      rig.setBox({ width: 600, height: 1_020 });
      rig.setScrollHeight(1_020);
      act(() => rig.observers[0]!());
      expect(rig.pendingFrames()).toBe(1);
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(midTop).toBeGreaterThan(500);
      expect(1_020 - 500 - midTop).toBeLessThanOrEqual(48);

      // Swipe up 20px and stay within 48px of the end.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 320, target: rig.root }]);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      const swipedTop = midTop - 20;
      rig.setScrollTopRaw(swipedTop);
      expect(1_020 - 500 - swipedTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);

      // Above-viewport content growth (+10px) with matching +10px scroll-anchoring shift must keep pinned false.
      const anchoredTop = swipedTop + 10;
      rig.setBox({ width: 600, height: 1_030 });
      rig.setScrollHeight(1_030);
      rig.setScrollTopRaw(anchoredTop);
      expect(1_030 - 500 - anchoredTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Further content growth does not restart chase or move scrollTop.
      rig.setBox({ width: 600, height: 1_036 });
      rig.setScrollHeight(1_036);
      expect(1_036 - 500 - anchoredTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.observers[0]!();
        rig.flushFrame(32);
      });
      expect(stick!.pinned).toBe(false);
      expect(stick!.fresh).toBe(true);
      expect(rig.pendingFrames()).toBe(0);
      expect(rig.root.scrollTop).toBe(anchoredTop);

      // Horizontal touch swipe does NOT clear touchHold; another above-viewport anchor adjustment stays unpinned.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 2, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 2, clientX: 260, clientY: 295, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      const anchoredTop2 = anchoredTop + 4;
      rig.setBox({ width: 600, height: 1_040 });
      rig.setScrollHeight(1_040);
      rig.setScrollTopRaw(anchoredTop2);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Explicit downward touch swipe (finger moving upward) clears touchHold even when rise <= content growth.
      rig.setBox({ width: 600, height: 1_050 });
      rig.setScrollHeight(1_050);
      act(() => {
        rig.observers[0]!();
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 3, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 3, clientX: 200, clientY: 280, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      const touchReturnTop = anchoredTop2 + 8;
      rig.setScrollTopRaw(touchReturnTop);
      expect(1_050 - 500 - touchReturnTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(true);
      expect(stick!.fresh).toBe(false);

      // Explicit downward wheel (deltaY > 0) also clears touchHold when scrolling back within PIN_SLACK.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 4, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 4, clientX: 200, clientY: 320, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      rig.setBox({ width: 600, height: 1_060 });
      rig.setScrollHeight(1_060);
      act(() => {
        rig.observers[0]!();
        rig.root.dispatchEvent(new WheelEvent('wheel', { deltaY: 20 }));
      });
      const wheelReturnTop = touchReturnTop + 8;
      rig.setScrollTopRaw(wheelReturnTop);
      expect(1_060 - 500 - wheelReturnTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(true);
      expect(stick!.fresh).toBe(false);

      // Same-gesture downward reversal clears touchHold before lifting the finger.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 5, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 5, clientX: 200, clientY: 320, target: rig.root }]);
      });
      expect(stick!.pinned).toBe(false);
      rig.setBox({ width: 600, height: 1_070 });
      rig.setScrollHeight(1_070);
      act(() => {
        rig.observers[0]!();
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 5, clientX: 200, clientY: 310, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      const reverseReturnTop = wheelReturnTop + 9;
      rig.setScrollTopRaw(reverseReturnTop);
      expect(1_070 - 500 - reverseReturnTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(true);
      expect(stick!.fresh).toBe(false);
    } finally {
      rig.cleanup();
    }
  });

  it('ignores wheel and keyboard scrolling inside a docked contained pane during thread touchHold and active chase', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    const pane = document.createElement('div');
    pane.style.overflowY = 'auto';
    pane.style.overscrollBehavior = 'contain';
    Object.defineProperties(pane, {
      clientHeight: { get: () => 240, configurable: true },
      scrollHeight: { get: () => 800, configurable: true },
      scrollTop: { get: () => 120, configurable: true },
    });
    const paneChild = document.createElement('div');
    pane.appendChild(paneChild);
    rig.root.appendChild(pane);

    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'docked-pane-wheel-key');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);

      // Start an active chase within PIN_SLACK and confirm docked-pane upward wheel/key does not release the thread.
      rig.setBox({ width: 600, height: 1_020 });
      rig.setScrollHeight(1_020);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(rig.pendingFrames()).toBe(1);

      act(() => {
        paneChild.dispatchEvent(new WheelEvent('wheel', { deltaY: -40, bubbles: true }));
        paneChild.dispatchEvent(new KeyboardEvent('keydown', { key: 'PageUp', bubbles: true }));
      });
      expect(stick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      // Establish active touchHold on the thread within PIN_SLACK.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 320, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      const heldTop = midTop - 20;
      rig.setScrollTopRaw(heldTop);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Downward wheel and ArrowDown originating inside the docked pane must NOT clear the thread's touchHold.
      act(() => {
        paneChild.dispatchEvent(new WheelEvent('wheel', { deltaY: 40, bubbles: true }));
        paneChild.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
      });

      // Above-viewport growth (+15px) with matching +15px scroll-anchor shift must keep pinned false.
      const anchoredTop = heldTop + 15;
      rig.setBox({ width: 600, height: 1_035 });
      rig.setScrollHeight(1_035);
      rig.setScrollTopRaw(anchoredTop);
      expect(1_035 - 500 - anchoredTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      // Thread-owned downward key clears touchHold and allows scroll within PIN_SLACK to re-pin.
      act(() => {
        rig.root.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
      });
      rig.setScrollTopRaw(anchoredTop + 10);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(true);
    } finally {
      rig.cleanup();
    }
  });

  it('preserves touchHold across simultaneous growth above (+25px) and shrink below (-15px) within PIN_SLACK', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'touch-hold-mixed-growth-shrink');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);

      // Establish active touchHold on the thread within PIN_SLACK.
      rig.setBox({ width: 600, height: 1_020 });
      rig.setScrollHeight(1_020);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;

      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 320, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      const swipedTop = midTop - 20;
      rig.setScrollTopRaw(swipedTop);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Simultaneous +25px growth above viewport and -15px shrink below viewport: net scrollHeight +10px, anchor scrollTop +25px.
      const shiftedTop = swipedTop + 25;
      rig.setBox({ width: 600, height: 1_030 });
      rig.setScrollHeight(1_030);
      rig.setScrollTopRaw(shiftedTop);
      expect(1_030 - 500 - shiftedTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      // Subsequent growth marks fresh without restarting chase or moving scrollTop.
      rig.setBox({ width: 600, height: 1_045 });
      rig.setScrollHeight(1_045);
      expect(1_045 - 500 - shiftedTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.observers[0]!();
        rig.flushFrame(32);
      });
      expect(stick!.pinned).toBe(false);
      expect(stick!.fresh).toBe(true);
      expect(rig.pendingFrames()).toBe(0);
      expect(rig.root.scrollTop).toBe(shiftedTop);

      // Verified downward wheel on root + scroll within PIN_SLACK clears touchHold and re-pins.
      act(() => {
        rig.root.dispatchEvent(new WheelEvent('wheel', { deltaY: 20 }));
      });
      rig.setScrollTopRaw(shiftedTop + 10);
      expect(1_045 - 500 - (shiftedTop + 10)).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(true);
      expect(stick!.fresh).toBe(false);
    } finally {
      rig.cleanup();
    }
  });

  it('ignores overscroll-behavior: contain on a non-scroll-container wrapper (overflow: visible)', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    const wrapper = document.createElement('div');
    wrapper.style.overflow = 'visible';
    wrapper.style.overflowY = 'visible';
    wrapper.style.overscrollBehavior = 'contain';
    const inner = document.createElement('div');
    inner.style.overflowY = 'auto';
    Object.defineProperties(inner, {
      clientHeight: { get: () => 200, configurable: true },
      scrollHeight: { get: () => 800, configurable: true },
      scrollTop: { get: () => 0, configurable: true },
    });
    const innerChild = document.createElement('div');
    inner.appendChild(innerChild);
    wrapper.appendChild(inner);
    rig.root.appendChild(wrapper);
    expect(getComputedStyle(wrapper).overflowY).toBe('visible');
    expect(getComputedStyle(wrapper).overscrollBehavior).toBe('contain');

    const hiddenContainer = document.createElement('div');
    hiddenContainer.style.overflow = 'hidden';
    hiddenContainer.style.overscrollBehavior = 'contain';
    const hiddenChild = document.createElement('div');
    hiddenContainer.appendChild(hiddenChild);
    rig.root.appendChild(hiddenContainer);

    let outerStick: StickToBottom | undefined;
    function Probe(): ReactElement {
      outerStick = useStickToBottom(rig.root, 'non-scroll-wrapper-outer');
      return createElement('div', { ref: outerStick.contentRef });
    }
    try {
      mount(createElement(Probe));
      rig.setBox({ width: 600, height: 1_800 });
      rig.setScrollHeight(1_800);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      expect(outerStick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      // overflow: hidden + overscroll-behavior: contain IS a scroll container, so it blocks release.
      act(() => {
        dispatchTouch(hiddenChild, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: hiddenChild }]);
        dispatchTouch(hiddenChild, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 340, target: hiddenChild }]);
        dispatchTouch(hiddenChild, 'touchend', []);
      });
      expect(outerStick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      // overflow: visible + overscroll-behavior: contain is NOT a scroll container, so an uncontained inner at scrollTop 0 releases outer.
      act(() => {
        dispatchTouch(innerChild, 'touchstart', [{ identifier: 2, clientX: 200, clientY: 300, target: innerChild }]);
        dispatchTouch(innerChild, 'touchmove', [{ identifier: 2, clientX: 200, clientY: 340, target: innerChild }]);
        dispatchTouch(innerChild, 'touchend', []);
      });
      expect(outerStick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);
    } finally {
      rig.cleanup();
    }
  });

  it('contains vertical overscroll on docked ChatFrame ScrollArea viewports and overlay roots rather than the overflow-visible list wrapper', () => {
    const rules = cssRules('components/arc-x/chat-frame/chat-frame.module.css');
    expect(cssValue(rules, '.paneViewport', 'overscroll-behavior')).toBe('contain');
    expect(cssValue(rules, '.listScroll', 'overflow')).toBe('hidden');
    expect(cssValue(rules, '.listScroll', 'overscroll-behavior')).toBe('contain');
    expect(cssValue(rules, '.railScroll', 'overflow')).toBe('hidden');
    expect(cssValue(rules, '.railScroll', 'overscroll-behavior')).toBe('contain');
    expect(cssValue(rules, '.list', 'overscroll-behavior')).toBeUndefined();
    expect(cssValue(rules, '.list', 'overscroll-behavior-y')).toBeUndefined();
    expect(cssValue(rules, '.rail', 'overflow')).toBe('hidden');
    expect(cssValue(rules, '.rail', 'overscroll-behavior')).toBe('contain');

    const restore = observeWidth(1_400);
    try {
      const container = mount(
        createElement(ChatFrame, {
          barStart: createElement('span', null, 'MODE'),
          barEnd: createElement('h1', null, 'Title'),
          list: createElement('a', { href: '#c1' }, 'Conversation one'),
          listLabel: 'Chats',
          listOpen: false,
          onListOpenChange: () => undefined,
          dock: createElement('textarea', { 'aria-label': 'Message' }),
          rail: {
            content: createElement('div', { 'data-testid': 'rail-body' }, 'Activity item'),
            label: 'Activity',
            docked: true,
            open: true,
            onOpenChange: () => undefined,
          },
          children: createElement('p', null, 'MESSAGE BODY'),
        }),
      );

      const listViewport = container.querySelector('a[href="#c1"]')?.parentElement;
      const railViewport = container.querySelector('[data-testid="rail-body"]')?.parentElement;
      expect(listViewport?.className).toContain('viewport');
      expect(listViewport?.className).toContain('paneViewport');
      expect(listViewport?.parentElement?.className).toContain('listScroll');
      expect(railViewport?.className).toContain('viewport');
      expect(railViewport?.className).toContain('paneViewport');
      expect(railViewport?.parentElement?.className).toContain('railScroll');
    } finally {
      restore();
    }
  });

  it('re-pins immediately on End without an intervening scroll event when touchHold is clamped at the bottom', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'touch-hold-shrink-clamped-end');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);

      // Start an active chase.
      rig.setBox({ width: 600, height: 1_040 });
      rig.setScrollHeight(1_040);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(midTop).toBeGreaterThan(500);
      expect(rig.pendingFrames()).toBe(1);

      // Swipe up on touch to establish touchHold and unpin.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 320, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      rig.setScrollTopRaw(midTop - 20);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Shrink scrollHeight so the viewport clamps to the exact bottom (distanceToEnd === 0).
      rig.setBox({ width: 600, height: 960 });
      rig.setScrollHeight(960);
      expect(rig.root.scrollTop).toBe(460);
      expect(960 - 500 - rig.root.scrollTop).toBe(0);
      act(() => {
        rig.observers[0]!();
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Pressing End when already clamped at bottom produces no scroll event, yet immediately re-pins.
      act(() => {
        rig.root.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
      });
      expect(stick!.pinned).toBe(true);
      expect(stick!.fresh).toBe(false);

      // Subsequent content growth smoothly chases to the new end via rAF.
      rig.setBox({ width: 600, height: 1_260 });
      rig.setScrollHeight(1_260);
      act(() => rig.observers[0]!());
      expect(rig.root.scrollTop).toBe(460);
      expect(rig.pendingFrames()).toBe(1);

      act(() => rig.flushFrame(32));
      const firstChaseTop = rig.root.scrollTop;
      expect(firstChaseTop).toBeGreaterThan(460);
      expect(firstChaseTop).toBeLessThan(760);
      expect(rig.pendingFrames()).toBe(1);

      let t = 32;
      let steps = 0;
      while (rig.pendingFrames() > 0 && steps < 100) {
        t += 16;
        steps += 1;
        act(() => rig.flushFrame(t));
      }
      expect(rig.pendingFrames()).toBe(0);
      expect(rig.root.scrollTop).toBe(760);
      expect(stick!.pinned).toBe(true);
    } finally {
      rig.cleanup();
    }
  });

  it('ignores wheel over docked ChatFrame list ScrollArea overlay track and thumb during active chase and touchHold', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 1_200, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'docked-list-scrollarea-overlay');
      return createPortal(
        createElement(ChatFrame, {
          barStart: createElement('span', null, 'MODE'),
          barEnd: createElement('h1', null, 'Title'),
          list: createElement('a', { href: '#c1' }, 'Conversation one'),
          listLabel: 'Chats',
          listOpen: false,
          onListOpenChange: () => undefined,
          dock: createElement('textarea', { 'aria-label': 'Message' }),
          children: createElement('div', { ref: stick.contentRef, 'data-testid': 'thread-content' }),
        }),
        rig.root,
      );
    }
    try {
      mount(createElement(Probe));
      const contentEl = rig.root.querySelector('[data-testid="thread-content"]');
      expect(contentEl).not.toBeNull();
      const listViewport = rig.root.querySelector('a[href="#c1"]')?.parentElement;
      const listScrollRoot = listViewport?.parentElement;
      expect(listScrollRoot?.className).toContain('listScroll');
      const track = listScrollRoot?.querySelector<HTMLElement>('[data-axis="y"]');
      const thumb = track?.firstElementChild as HTMLElement | null;
      expect(track?.className).toContain('track');
      expect(thumb?.className).toContain('thumb');

      // Mirror the `.listScroll` CSS contract in happy-dom (verified against chat-frame.module.css above).
      listScrollRoot!.style.overflow = 'hidden';
      listScrollRoot!.style.overscrollBehavior = 'contain';
      expect(getComputedStyle(listScrollRoot!).overflow).toBe('hidden');
      expect(getComputedStyle(listScrollRoot!).overscrollBehavior).toBe('contain');

      // Start an active chase within PIN_SLACK.
      rig.setBox({ width: 1_200, height: 1_024 });
      rig.setScrollHeight(1_024);
      act(() => rig.notify(contentEl!));
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(midTop).toBeGreaterThan(500);
      expect(rig.pendingFrames()).toBe(1);

      // Upward wheel over the docked list ScrollArea overlay track and thumb must not cancel the outer chase.
      act(() => {
        track!.dispatchEvent(new WheelEvent('wheel', { deltaY: -40, bubbles: true }));
        thumb!.dispatchEvent(new WheelEvent('wheel', { deltaY: -40, bubbles: true }));
      });
      expect(stick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      act(() => rig.flushFrame(32));
      const nextTop = rig.root.scrollTop;
      expect(nextTop).toBeGreaterThan(midTop);

      // Establish active touchHold on the outer thread within PIN_SLACK.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 600, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 600, clientY: 320, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      const heldTop = nextTop - 16;
      rig.setScrollTopRaw(heldTop);
      expect(1_024 - 500 - heldTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Downward wheel over the overlay track and thumb must not clear the outer thread's touchHold.
      act(() => {
        track!.dispatchEvent(new WheelEvent('wheel', { deltaY: 40, bubbles: true }));
        thumb!.dispatchEvent(new WheelEvent('wheel', { deltaY: 40, bubbles: true }));
      });
      expect(stick!.pinned).toBe(false);

      // Subsequent layout-anchor scroll within PIN_SLACK confirms touchHold remained intact.
      const anchoredTop = heldTop + 8;
      rig.setBox({ width: 1_200, height: 1_032 });
      rig.setScrollHeight(1_032);
      rig.setScrollTopRaw(anchoredTop);
      expect(1_032 - 500 - anchoredTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);
    } finally {
      rig.cleanup();
    }
  });

  it('ignores Space and Shift+Space on a focused button inside the thread during active chase and touchHold', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = 'Thinking';
    rig.root.appendChild(button);

    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'space-on-button');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);

      // Start an active chase within PIN_SLACK.
      rig.setBox({ width: 600, height: 1_020 });
      rig.setScrollHeight(1_020);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(midTop).toBeGreaterThan(500);
      expect(rig.pendingFrames()).toBe(1);

      // Shift+Space on a focused button does not release a pinned chase.
      button.focus();
      act(() => {
        button.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', shiftKey: true, bubbles: true }));
      });
      expect(stick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      // Establish active touchHold within PIN_SLACK.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 320, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      const swipedTop = midTop - 10;
      rig.setScrollTopRaw(swipedTop);
      expect(1_020 - 500 - swipedTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Space on the focused disclosure button must not clear touchHold or re-pin.
      act(() => {
        button.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
      });
      expect(stick!.pinned).toBe(false);

      // Layout shift + scroll within PIN_SLACK keeps pinned false because touchHold survived.
      const shiftedTop = swipedTop + 10;
      rig.setBox({ width: 600, height: 1_030 });
      rig.setScrollHeight(1_030);
      rig.setScrollTopRaw(shiftedTop);
      expect(1_030 - 500 - shiftedTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Subsequent growth marks fresh without re-pinning or starting a chase.
      rig.setBox({ width: 600, height: 1_040 });
      rig.setScrollHeight(1_040);
      expect(1_040 - 500 - shiftedTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.observers[0]!();
        rig.flushFrame(32);
      });
      expect(stick!.pinned).toBe(false);
      expect(stick!.fresh).toBe(true);
      expect(rig.pendingFrames()).toBe(0);
      expect(rig.root.scrollTop).toBe(shiftedTop);

      // Default-prevented downward key also does not clear touchHold.
      const prevent = (event: KeyboardEvent): void => event.preventDefault();
      button.addEventListener('keydown', prevent);
      act(() => {
        button.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true, cancelable: true }));
      });
      button.removeEventListener('keydown', prevent);
      expect(stick!.pinned).toBe(false);

      // Non-interactive Space on the thread root still clears touchHold and re-pins within PIN_SLACK.
      act(() => {
        rig.root.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
      });
      expect(stick!.pinned).toBe(true);
      expect(stick!.fresh).toBe(false);
    } finally {
      rig.cleanup();
    }
  });

  it('respects React synthetic onKeyDown preventDefault on descendants for Home and End', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    let stick: StickToBottom | undefined;
    let scrollRoot: HTMLDivElement | null = null;
    function Probe(): ReactElement {
      const [rootEl, setRootEl] = useState<HTMLDivElement | null>(null);
      stick = useStickToBottom(rootEl, 'react-synthetic-keydown-cancel');
      return createElement(
        'div',
        {
          ref: (node: HTMLDivElement | null) => {
            if (node !== null && node !== scrollRoot) {
              rig.attachRoot(node);
              scrollRoot = node;
            }
            setRootEl(node);
          },
        },
        createElement(
          'div',
          { ref: stick.contentRef },
          createElement(
            'button',
            {
              type: 'button',
              onKeyDown: (event: ReactKeyboardEvent<HTMLButtonElement>) => {
                if (event.key === 'Home' || event.key === 'End') event.preventDefault();
              },
            },
            'Tool call',
          ),
        ),
      );
    }
    try {
      const container = mount(createElement(Probe));
      expect(scrollRoot).not.toBeNull();
      expect(container.contains(scrollRoot)).toBe(true);
      expect(scrollRoot).not.toBe(container);
      const button = scrollRoot!.querySelector('button')!;
      expect(button).not.toBeNull();
      expect(scrollRoot!.scrollTop).toBe(500);

      // Start an active chase within PIN_SLACK.
      rig.setBox({ width: 600, height: 1_020 });
      rig.setScrollHeight(1_020);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const midTop = scrollRoot!.scrollTop;
      expect(midTop).toBeGreaterThan(500);
      expect(rig.pendingFrames()).toBe(1);

      // Home canceled by a React descendant onKeyDown must not release an active chase.
      button.focus();
      act(() => {
        button.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true, cancelable: true }));
      });
      expect(stick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      act(() => rig.flushFrame(32));
      expect(scrollRoot!.scrollTop).toBeGreaterThan(midTop);

      // Establish active touchHold within PIN_SLACK.
      act(() => {
        dispatchTouch(scrollRoot!, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: scrollRoot! }]);
        dispatchTouch(scrollRoot!, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 320, target: scrollRoot! }]);
        dispatchTouch(scrollRoot!, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      const heldTop = scrollRoot!.scrollTop - 10;
      rig.setScrollTopRaw(heldTop);
      expect(1_020 - 500 - heldTop).toBeLessThanOrEqual(48);
      act(() => {
        scrollRoot!.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // End canceled by the React descendant onKeyDown must not clear touchHold or re-pin.
      act(() => {
        button.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true, cancelable: true }));
      });
      expect(stick!.pinned).toBe(false);

      // Subsequent above-viewport anchor shift within PIN_SLACK confirms touchHold remained intact.
      const anchoredTop = heldTop + 8;
      rig.setBox({ width: 600, height: 1_028 });
      rig.setScrollHeight(1_028);
      rig.setScrollTopRaw(anchoredTop);
      expect(1_028 - 500 - anchoredTop).toBeLessThanOrEqual(48);
      act(() => {
        scrollRoot!.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      // Uncanceled End on the scroll root still clears touchHold and re-pins immediately.
      act(() => {
        scrollRoot!.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true, cancelable: true }));
      });
      expect(stick!.pinned).toBe(true);
    } finally {
      rig.cleanup();
    }
  });

  it('ignores Ctrl+wheel zoom gestures during active chase and touchHold', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    const ctrlWheel = (deltaY: number): WheelEvent => {
      const event = new WheelEvent('wheel', { deltaY, ctrlKey: true, bubbles: true });
      // happy-dom's WheelEvent inherits UIEvent instead of MouseEvent and drops MouseEventInit.ctrlKey.
      if (!event.ctrlKey) Object.defineProperty(event, 'ctrlKey', { value: true });
      return event;
    };
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'ctrl-wheel-zoom');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);

      // Start an active chase within PIN_SLACK.
      rig.setBox({ width: 600, height: 1_020 });
      rig.setScrollHeight(1_020);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(midTop).toBeGreaterThan(500);
      expect(rig.pendingFrames()).toBe(1);

      // Ctrl+wheel up (zoom in) must neither release nor cancel an active chase.
      act(() => {
        rig.root.dispatchEvent(ctrlWheel(-40));
      });
      expect(stick!.pinned).toBe(true);
      expect(rig.pendingFrames()).toBe(1);

      act(() => rig.flushFrame(32));
      expect(rig.root.scrollTop).toBeGreaterThan(midTop);

      // Establish active touchHold within PIN_SLACK.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 320, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      const heldTop = rig.root.scrollTop - 10;
      rig.setScrollTopRaw(heldTop);
      expect(1_020 - 500 - heldTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Ctrl+wheel down (zoom out) must not clear touchHold or re-pin.
      act(() => {
        rig.root.dispatchEvent(ctrlWheel(40));
      });
      expect(stick!.pinned).toBe(false);

      // Subsequent above-viewport anchor shift within PIN_SLACK confirms touchHold stayed active.
      const anchoredTop = heldTop + 8;
      rig.setBox({ width: 600, height: 1_028 });
      rig.setScrollHeight(1_028);
      rig.setScrollTopRaw(anchoredTop);
      expect(1_028 - 500 - anchoredTop).toBeLessThanOrEqual(48);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);
    } finally {
      rig.cleanup();
    }
  });

  it('clears touchHold and re-pins immediately on Space from a focused link at a shrink-clamped bottom', () => {
    skipAnimations(false);
    const rig = createScrollRig({ width: 600, height: 1_000, clientHeight: 500 });
    const link = document.createElement('a');
    link.href = 'https://example.com';
    link.textContent = 'Transcript link';
    rig.root.appendChild(link);

    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(rig.root, 'space-on-link-clamped-bottom');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(rig.root.scrollTop).toBe(500);

      // Start an active chase.
      rig.setBox({ width: 600, height: 1_040 });
      rig.setScrollHeight(1_040);
      act(() => rig.observers[0]!());
      act(() => rig.flushFrame(16));
      const midTop = rig.root.scrollTop;
      expect(midTop).toBeGreaterThan(500);

      // Establish touchHold via upward swipe.
      act(() => {
        dispatchTouch(rig.root, 'touchstart', [{ identifier: 1, clientX: 200, clientY: 300, target: rig.root }]);
        dispatchTouch(rig.root, 'touchmove', [{ identifier: 1, clientX: 200, clientY: 320, target: rig.root }]);
        dispatchTouch(rig.root, 'touchend', []);
      });
      expect(stick!.pinned).toBe(false);
      expect(rig.pendingFrames()).toBe(0);

      rig.setScrollTopRaw(midTop - 20);
      act(() => {
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Shrink scrollHeight so the viewport clamps to the exact bottom (distanceToEnd === 0).
      rig.setBox({ width: 600, height: 960 });
      rig.setScrollHeight(960);
      expect(rig.root.scrollTop).toBe(460);
      expect(960 - 500 - rig.root.scrollTop).toBe(0);
      act(() => {
        rig.observers[0]!();
        rig.root.dispatchEvent(new Event('scroll'));
      });
      expect(stick!.pinned).toBe(false);

      // Pressing Space on a focused link scrolls rather than activating the link, so it clears touchHold and re-pins immediately.
      link.focus();
      act(() => {
        link.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
      });
      expect(stick!.pinned).toBe(true);
      expect(stick!.fresh).toBe(false);
    } finally {
      rig.cleanup();
    }
  });
});
