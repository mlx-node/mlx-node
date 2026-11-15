/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  AgentTimeline,
  type AgentClock,
  type AgentHandoff,
  type AgentLane,
  type AgentSpan,
  type AgentTimelineMode,
  type AgentTimelineProps,
} from '../ui/src/components/arc-x/agent-timeline/agent-timeline.js';
import { timelineAxis, timelineModel } from '../ui/src/lib/agent-timeline.js';
import { formatDuration } from '../ui/src/lib/format.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { wokenTimeline } from './helpers/woken-turn.js';
import { accessibleName, allByRole, byRole, skipAnimations } from './render.js';

const CSS = 'components/arc-x/agent-timeline/agent-timeline.module.css';

const LABEL = 'Turn 6 timeline';

const LANES: AgentLane[] = [
  { id: 'main', name: 'Main agent', model: 'Qwen3.5-27B-4bit', color: 'var(--lane-main)' },
  {
    id: 'kid',
    name: 'Map the scheduler queues',
    model: 'Qwen3-8B-4bit',
    color: 'var(--lane-1)',
    to: '/sessions/kid',
  },
  { id: 'bad', name: 'Bisect the cache', color: 'var(--lane-2)', outcome: { text: 'Failed', tone: 'danger' } },
];

const SPANS: AgentSpan[] = [
  {
    id: 'm0',
    laneId: 'main',
    kind: 'wait',
    startMs: 0,
    endMs: 800,
    status: 'ok',
    label: 'Waiting for the first token',
  },
  {
    id: 'm1',
    laneId: 'main',
    kind: 'generate',
    startMs: 800,
    endMs: 3_000,
    status: 'ok',
    label: 'Reasoning',
    retry: 'Retry 1 of 3',
  },
  {
    id: 'm2',
    laneId: 'main',
    kind: 'wait',
    startMs: 3_000,
    endMs: 60_000,
    status: 'ok',
    label: 'Waiting for subagents',
    ownsHandoffs: true,
  },
  {
    id: 'm3',
    laneId: 'main',
    kind: 'generate',
    startMs: 60_000,
    endMs: 72_000,
    status: 'ok',
    label: 'Writing the answer',
  },
  { id: 'k0', laneId: 'kid', kind: 'tool', startMs: 3_200, endMs: 3_612, status: 'failed', label: 'Tool call · grep' },
  {
    id: 'k1',
    laneId: 'kid',
    kind: 'generate',
    startMs: 4_000,
    endMs: 30_000,
    status: 'ok',
    label: 'Writing the answer',
  },
  {
    id: 'b0',
    laneId: 'bad',
    kind: 'run',
    startMs: 3_100,
    endMs: 50_000,
    status: 'unfinished',
    open: 'end',
    label: 'Ran, no step record',
    note: 'No step record: the subagent wrote no events.',
  },
];

const HANDOFFS: AgentHandoff[] = [
  { id: 'kid:spawn:0', laneId: 'kid', kind: 'spawn', atMs: 3_000 },
  { id: 'kid:return:0', laneId: 'kid', kind: 'return', atMs: 30_000 },
  { id: 'bad:spawn:0', laneId: 'bad', kind: 'spawn', atMs: 3_000 },
  { id: 'bad:return:0', laneId: 'bad', kind: 'return', atMs: 50_000, failed: true },
];

const REPORTS = [
  { laneId: 'kid', atMs: 30_000, tokens: 9_000 },
  { laneId: 'main', atMs: 72_000, tokens: 38_827 },
];

interface FakeClock {
  clock: AgentClock;
  /** Advances the clock by `ms` and runs the frames that were requested. */
  frame(ms: number): void;
  frames(count: number, ms: number): void;
  requests(): number;
  waiting(): number;
}

function fakeClock(start = 1_000_000): FakeClock {
  let now = start;
  let handle = 0;
  let requests = 0;
  const pending = new Map<number, (time: number) => void>();
  const frame = (ms: number): void => {
    now += ms;
    const callbacks = [...pending.values()];
    pending.clear();
    act(() => {
      for (const callback of callbacks) callback(now);
    });
  };
  return {
    clock: {
      now: () => now,
      requestFrame: (callback) => {
        requests += 1;
        handle += 1;
        pending.set(handle, callback);
        return handle;
      },
      cancelFrame: (id) => {
        pending.delete(id);
      },
    },
    frame,
    frames: (count, ms) => {
      for (let i = 0; i < count; i += 1) frame(ms);
    },
    requests: () => requests,
    waiting: () => pending.size,
  };
}

interface Mounted {
  container: HTMLElement;
  root: Root;
  rerender: (props: Partial<AgentTimelineProps>) => void;
}

let roots: Root[] = [];

function mount(props: Partial<AgentTimelineProps> = {}): Mounted {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  roots.push(root);
  let current: AgentTimelineProps = {
    label: LABEL,
    title: 'Why does the scheduler stall?',
    lanes: LANES,
    spans: SPANS,
    handoffs: HANDOFFS,
    reports: REPORTS,
    durationMs: 72_000,
    mode: { kind: 'recorded' },
    outcome: { word: 'Finished', tone: 'success' },
    formatDuration,
    axisUnit: timelineAxis,
    ...props,
  };
  const render = (): void => {
    const element: ReactElement = createElement(MemoryRouter, null, createElement(AgentTimeline, current));
    act(() => root.render(element));
  };
  render();
  return {
    container,
    root,
    rerender: (next) => {
      current = { ...current, ...next };
      render();
    },
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  for (const root of roots) act(() => root.unmount());
  roots = [];
  document.body.innerHTML = '';
});

function section(container: HTMLElement): HTMLElement {
  return byRole(container, 'region', LABEL) ?? (container.querySelector('section') as HTMLElement);
}

function spanButtons(container: HTMLElement): HTMLButtonElement[] {
  return [...container.querySelectorAll<HTMLButtonElement>('button[data-span]')];
}

function spanNamed(container: HTMLElement, start: string): HTMLButtonElement {
  const found = spanButtons(container).find((button) => (button.getAttribute('aria-label') ?? '').startsWith(start));
  if (found === undefined) throw new Error(`no span named ${start}`);
  return found;
}

function statusLine(container: HTMLElement): HTMLElement {
  return section(container).querySelector('header p') as HTMLElement;
}

function statusText(container: HTMLElement): string {
  return (statusLine(container).textContent ?? '').split(/* plain */ ' ').join(' ');
}

function summary(container: HTMLElement): string {
  const id = section(container).getAttribute('aria-describedby') ?? '';
  return document.getElementById(id)?.textContent ?? '';
}

function plot(container: HTMLElement): HTMLElement {
  return section(container).querySelector('[aria-label^="Steps"]') as HTMLElement;
}

function slider(container: HTMLElement): HTMLElement {
  return byRole(container, 'slider', 'Turn time') as HTMLElement;
}

function liveRegion(container: HTMLElement): string {
  return section(container).querySelector('[aria-live="polite"]')?.textContent ?? '';
}

function key(target: Element, name: string, init: KeyboardEventInit = {}): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, ...init }));
  });
}

function playButton(container: HTMLElement): HTMLButtonElement {
  return section(container).querySelector('footer button') as HTMLButtonElement;
}

function tableRows(container: HTMLElement, caption: string): string[][] {
  const table = [...section(container).querySelectorAll('table')].find(
    (candidate) => candidate.querySelector('caption')?.textContent === caption,
  );
  if (table === undefined) throw new Error(`no table ${caption}`);
  return [...table.querySelectorAll('tbody tr')].map((row) =>
    [...row.querySelectorAll('th, td')].map((cell) => cell.textContent ?? ''),
  );
}

function focusedName(): string {
  return document.activeElement?.getAttribute('aria-label') ?? '';
}

/**
 * happy-dom has no layout: span buttons get a 20 × 14 box at `at` (window px, null: no box), every other element a zero
 * box at the window origin.
 */
function layoutSpans(at: (button: HTMLElement) => { left: number; top: number } | null): void {
  const zero = (): DOMRect => DOMRect.fromRect({ x: 0, y: 0, width: 0, height: 0 });
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element): DOMRect {
    if (!(this instanceof HTMLElement) || this.dataset.span === undefined) return zero();
    const place = at(this);
    return place === null ? zero() : DOMRect.fromRect({ x: place.left, y: place.top, width: 20, height: 14 });
  });
}

/** The detail card on show (the layer's first child is the hidden measure copy). */
function shownCard(): HTMLElement | null {
  const layer = [...document.body.children].find(
    (node) => node.getAttribute('aria-hidden') === 'true' && node.children.length > 0,
  );
  return (layer?.children[1] as HTMLElement | undefined) ?? null;
}

/** The card's top in window px, from motion's transform. */
function cardY(): number | null {
  const transform = shownCard()?.style.transform ?? '';
  const start = transform.indexOf('translateY(');
  if (start === -1) return null;
  return Number.parseFloat(transform.slice(start + 'translateY('.length));
}

function tabStops(container: HTMLElement): HTMLElement[] {
  const inPlot = [...plot(container).querySelectorAll<HTMLElement>('[tabindex="0"]')];
  return plot(container).tabIndex === 0 ? [plot(container), ...inPlot] : inPlot;
}

describe('AgentTimeline settled', () => {
  it('names every span in full and states the finished turn', () => {
    const { container } = mount();
    const names = spanButtons(container).map((button) => button.getAttribute('aria-label'));
    expect(names).toHaveLength(SPANS.length);
    expect(names).toContain('Map the scheduler queues, Tool call · grep, 412 ms, from 3.2 s to 3.61 s, failed');
    expect(names).toContain('Main agent, Reasoning, 2.20 s, from 0.8 s to 3 s, Retry 1 of 3');
    expect(names).toContain(
      'Bisect the cache, Ran, no step record, 46.9 s, from 3.1 s to 50 s, not finished, end not recorded',
    );
    expect(statusText(container)).toBe('Finished · 1m 12s · 38,827 tokens · 9,000 in 1 subagent');
    expect(section(container).hasAttribute('data-settled')).toBe(true);
    expect(accessibleName(playButton(container))).toBe('Replay');
  });

  it('keeps the final sentence as the description, also while a replay runs', () => {
    const fake = fakeClock();
    const { container } = mount({ clock: fake.clock });
    const final = 'Finished. 1m 12s. 38,827 tokens, 9,000 in 1 subagent.';
    expect(summary(container)).toBe(final);
    expect(statusLine(container).getAttribute('aria-hidden')).toBe('true');
    act(() => playButton(container).click());
    expect(summary(container)).toBe(final);
    fake.frames(5, 100);
    expect(summary(container)).toBe(final);
    act(() => playButton(container).click());
    expect(summary(container)).toBe(final);
  });

  it('keeps the tick at the axis end inside the plot', () => {
    const { container } = mount({ durationMs: 60_000 });
    const ticks = [...section(container).querySelectorAll<HTMLElement>('[style*="--t"]')].filter(
      (node) => node.textContent !== '',
    );
    expect(ticks.filter((tick) => tick.hasAttribute('data-end')).map((tick) => tick.textContent)).toEqual(['60 s']);
    expect(
      ticks
        .filter((tick) => tick.style.getPropertyValue('--t') !== '60')
        .every((tick) => !tick.hasAttribute('data-end')),
    ).toBe(true);
    const end = cssRules(CSS).find((rule) => rule.selectors.includes('.tick[data-end]'));
    expect(end?.decls.get('transform')).toBe('translateX(-100%)');
  });

  it('places no tick past the axis and renders a zero-length record', () => {
    const { container } = mount();
    const ticks = [...section(container).querySelectorAll<HTMLElement>('[style*="--t"]')].map((tick) =>
      Number(tick.style.getPropertyValue('--t')),
    );
    expect(ticks.length).toBeGreaterThan(1);
    expect(Math.max(...ticks)).toBeLessThanOrEqual(72);
    const flat = mount({ durationMs: 0, spans: [{ ...SPANS[0]!, startMs: 0, endMs: 0 }], handoffs: [] });
    const track = plot(flat.container).parentElement as HTMLElement;
    expect(track.style.getPropertyValue('--axis')).toBe('0.001');
  });

  it('lists every span and handoff in the hidden tables, also at 0 of a replay', () => {
    const fake = fakeClock();
    const { container } = mount({ clock: fake.clock });
    act(() => playButton(container).click());
    act(() => playButton(container).click());
    expect(tableRows(container, LABEL)).toHaveLength(SPANS.length);
    expect(tableRows(container, LABEL)[4]).toEqual([
      'Map the scheduler queues · Qwen3-8B-4bit',
      'Tool call',
      'Tool call · grep',
      '3.2 s',
      '3.61 s',
      '412 ms',
      'failed',
    ]);
    expect(tableRows(container, `${LABEL}: handoffs`)).toEqual([
      ['Map the scheduler queues', 'spawned', '3 s', 'ok'],
      ['Map the scheduler queues', 'returned', '30 s', 'ok'],
      ['Bisect the cache', 'spawned', '3 s', 'ok'],
      ['Bisect the cache', 'returned', '50 s', 'failed'],
    ]);
  });

  it('draws a span cut by the turn square on that side, unfaded, and says it went on', () => {
    const cut: AgentSpan[] = [
      { ...SPANS[5]!, id: 'k1', cut: 'end', status: 'unfinished' },
      { ...SPANS[0]!, id: 'm0', cut: 'start' },
    ];
    const { container } = mount({ spans: cut, handoffs: [], reports: [] });
    const end = spanNamed(container, 'Map the scheduler queues, Writing the answer');
    expect(end.dataset.cut).toBe('end');
    expect(end.hasAttribute('data-tail')).toBe(false);
    expect(end.getAttribute('aria-label')).toBe(
      'Map the scheduler queues, Writing the answer, 26.0 s, from 4 s to 30 s, not finished, went on after this turn',
    );
    expect(spanNamed(container, 'Main agent, Waiting').getAttribute('aria-label')).toContain(
      ', began before this turn',
    );
    expect(tableRows(container, LABEL).map((row) => row[6])).toEqual([
      'ok, began before this turn',
      'not finished, went on after this turn',
    ]);
    const rules = cssRules(CSS);
    const corners = (value: string): Map<string, string> | undefined =>
      rules.find((rule) => rule.selectors.includes(`.span[data-cut='${value}']`))?.decls;
    expect(corners('end')?.get('border-top-right-radius')).toBe('0');
    expect(corners('end')?.get('border-bottom-right-radius')).toBe('0');
    expect(corners('start')?.get('border-top-left-radius')).toBe('0');
    expect(corners('both')?.get('border-radius')).toBe('0');
    const fades = rules.filter((rule) => rule.decls.has('mask-image'));
    for (const rule of fades) {
      for (const selector of rule.selectors) {
        if (selector.includes("data-status='unfinished'"))
          expect(selector).toContain(":not([data-cut='end'], [data-cut='both'])");
      }
    }
  });

  it('ends every status part but the last with its separator, so a wrapped line never starts with one', () => {
    const { container } = mount();
    const parts = [...statusLine(container).children].filter((node) => !node.textContent?.startsWith(','));
    expect(parts.length).toBe(4);
    parts.forEach((part, index) => {
      const last = part.lastElementChild;
      if (index === parts.length - 1) expect(last?.textContent?.includes('·')).toBe(false);
      else expect([last?.getAttribute('aria-hidden'), last?.textContent]).toEqual(['true', ' ·']);
      expect(part.textContent?.startsWith('·')).toBe(false);
    });
  });

  it('lists only the legend entries that occur', () => {
    const { container } = mount();
    const legend = byRole(container, 'list', 'Legend');
    expect([...(legend?.querySelectorAll('li') ?? [])].map((item) => item.textContent)).toEqual([
      'Generating',
      'Tool call',
      'Waiting',
      'Handoff',
      'Failed',
      'Retried',
      'No step record',
      'Time not recorded',
    ]);
    const plain = mount({ spans: [SPANS[1]!], handoffs: [] });
    const short = byRole(plain.container, 'list', 'Legend');
    expect([...(short?.querySelectorAll('li') ?? [])].map((item) => item.textContent)).toEqual([
      'Generating',
      'Retried',
    ]);
  });
});

describe('AgentTimeline keyboard', () => {
  it('has exactly one tab stop; at 0 of a replay it is the plot, named for the time', () => {
    const fake = fakeClock();
    const { container } = mount({ clock: fake.clock });
    expect(tabStops(container)).toHaveLength(1);
    act(() => playButton(container).click());
    act(() => playButton(container).click());
    expect(tabStops(container)).toEqual([plot(container)]);
    expect(plot(container).getAttribute('aria-label')).toBe('Steps, none reached at 0 s');
    expect(spanButtons(container).every((button) => button.hasAttribute('data-pending'))).toBe(true);
  });

  it('hands focus to the plot when the focused span becomes pending, never to the body', () => {
    const { container } = mount();
    const answer = spanNamed(container, 'Main agent, Writing the answer');
    act(() => answer.focus());
    expect(document.activeElement).toBe(answer);
    key(slider(container), 'Home');
    expect(answer.hasAttribute('data-pending')).toBe(true);
    expect(document.activeElement).toBe(plot(container));
  });

  it('arrows move real focus; Home and End stay in the lane', () => {
    const { container } = mount();
    const first = spanButtons(container).find((button) => button.tabIndex === 0)!;
    act(() => first.focus());
    key(first, 'ArrowRight');
    expect(focusedName().startsWith('Main agent, Reasoning')).toBe(true);
    expect((document.activeElement as HTMLElement).tabIndex).toBe(0);
    key(document.activeElement!, 'ArrowDown');
    expect(focusedName().startsWith('Map the scheduler queues, Tool call')).toBe(true);
    key(document.activeElement!, 'End');
    expect(focusedName().startsWith('Map the scheduler queues, Writing')).toBe(true);
    key(document.activeElement!, 'Home');
    expect(focusedName().startsWith('Map the scheduler queues, Tool call')).toBe(true);
    expect(tabStops(container)).toHaveLength(1);
  });

  it('Enter selects with its path, Escape clears', () => {
    const { container } = mount();
    const wait = spanNamed(container, 'Main agent, Waiting for subagents');
    act(() => wait.focus());
    act(() => wait.click());
    expect(wait.getAttribute('aria-pressed')).toBe('true');
    expect(spanNamed(container, 'Main agent, Reasoning').hasAttribute('data-dim')).toBe(true);
    expect(wait.hasAttribute('data-dim')).toBe(false);
    expect(spanNamed(container, 'Map the scheduler queues, Tool call').hasAttribute('data-dim')).toBe(false);
    const hot = [...section(container).querySelectorAll('[data-hot]')];
    expect(hot.length).toBe(8);
    key(wait, 'Escape');
    expect(wait.getAttribute('aria-pressed')).toBe('false');
    expect(section(container).querySelectorAll('[data-dim]')).toHaveLength(0);
  });
});

describe('AgentTimeline slider', () => {
  it('is a named slider with the recorded value', () => {
    const { container } = mount();
    const knob = slider(container);
    expect(knob.getAttribute('aria-valuemin')).toBe('0');
    expect(knob.getAttribute('aria-valuemax')).toBe('72');
    expect(knob.getAttribute('aria-valuenow')).toBe('72');
    expect(knob.getAttribute('aria-valuetext')).toBe('1m 12s of 1m 12s');
    expect(knob.tabIndex).toBe(0);
  });

  it('a press on the axis strip leaves focus on the knob', () => {
    const { container } = mount();
    const knob = slider(container);
    const strip = knob.parentElement as HTMLElement;
    let kept = true;
    act(() => {
      kept = strip.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true, pointerId: 1 }));
    });
    expect(kept).toBe(false);
    expect(document.activeElement).toBe(knob);
  });

  it('reserves the widest replay time, so the parts after it never move', () => {
    const { container } = mount();
    key(slider(container), 'Home');
    const reserved = [...statusLine(container).querySelectorAll('[data-reserve]')].map((node) =>
      node.getAttribute('data-reserve'),
    );
    expect(reserved).toEqual(['1m 12s of 1m 12s', '59.9 s of 1m 12s', '1:11 of 1m 12s']);
  });

  it('steps by scrubStep, ten steps with Shift or PageUp, and jumps with Home and End', () => {
    const { container } = mount();
    const knob = slider(container);
    key(knob, 'Home');
    expect(knob.getAttribute('aria-valuenow')).toBe('0');
    key(knob, 'ArrowRight');
    expect(knob.getAttribute('aria-valuenow')).toBe('2');
    key(knob, 'ArrowRight', { shiftKey: true });
    expect(knob.getAttribute('aria-valuenow')).toBe('22');
    key(knob, 'PageUp');
    expect(knob.getAttribute('aria-valuenow')).toBe('42');
    key(knob, 'ArrowLeft');
    expect(knob.getAttribute('aria-valuenow')).toBe('40');
    expect(knob.getAttribute('aria-valuetext')).toBe('40.0 s of 1m 12s');
    expect(statusText(container)).toBe('Paused · 40.0 s of 1m 12s · 9,000 in subagents');
    key(knob, 'End');
    expect(section(container).hasAttribute('data-settled')).toBe(true);
    key(knob, 'ArrowLeft');
    key(knob, 'ArrowRight');
    expect(section(container).hasAttribute('data-settled')).toBe(false);
    expect(statusText(container)).toBe('Paused · 1m 12s of 1m 12s · 38,827 tokens · 9,000 in subagents');
  });

  it('a key pauses a running replay', () => {
    const fake = fakeClock();
    const { container } = mount({ clock: fake.clock });
    act(() => playButton(container).click());
    fake.frames(3, 100);
    key(slider(container), 'ArrowRight');
    expect(section(container).hasAttribute('data-playing')).toBe(false);
    expect(accessibleName(playButton(container))).toBe('Play');
  });
});

describe('AgentTimeline replay clock', () => {
  it('replays recorded times at a true rate and settles at the end', () => {
    const fake = fakeClock();
    const { container } = mount({ clock: fake.clock });
    expect(fake.requests()).toBe(0);
    act(() => playButton(container).click());
    expect(slider(container).getAttribute('aria-valuenow')).toBe('0');
    expect(statusText(container)).toBe('Replaying · 0 s of 1m 12s');
    fake.frame(16);
    fake.frames(20, 100);
    expect(slider(container).getAttribute('aria-valuenow')).toBe('10');
    expect(statusText(container)).toBe('Replaying · 10.0 s of 1m 12s');
    const answer = spanNamed(container, 'Main agent, Writing the answer');
    expect(answer.hasAttribute('data-pending')).toBe(true);
    expect(answer.tabIndex).toBe(-1);
    expect(spanNamed(container, 'Main agent, Reasoning').hasAttribute('data-pending')).toBe(false);
    fake.frames(40, 100);
    expect(statusText(container)).toBe('Replaying · 30.0 s of 1m 12s · 9,000 in subagents');
    fake.frames(90, 100);
    expect(section(container).hasAttribute('data-settled')).toBe(true);
    expect(accessibleName(playButton(container))).toBe('Replay');
    expect(liveRegion(container)).toBe('Replay finished');
    expect(statusText(container)).toBe('Finished · 1m 12s · 38,827 tokens · 9,000 in 1 subagent');
    const after = fake.requests();
    fake.frames(3, 100);
    expect(fake.requests()).toBe(after);
    expect(fake.waiting()).toBe(0);
  });

  it('pauses with an announcement and requests no frame while paused', () => {
    const fake = fakeClock();
    const { container } = mount({ clock: fake.clock });
    act(() => playButton(container).click());
    fake.frame(16);
    fake.frames(10, 100);
    act(() => playButton(container).click());
    expect(liveRegion(container)).toBe('Paused at 5.0 s');
    expect(statusText(container)).toBe('Paused · 5.0 s of 1m 12s');
    expect(fake.waiting()).toBe(0);
  });

  it('requests no frame while the window is hidden and resumes without a jump', async () => {
    const fake = fakeClock();
    const { container } = mount({ clock: fake.clock });
    act(() => playButton(container).click());
    fake.frame(16);
    fake.frames(10, 100);
    expect(slider(container).getAttribute('aria-valuenow')).toBe('5');
    await act(async () => {
      document.documentElement.setAttribute('data-hidden', '');
      await Promise.resolve();
    });
    try {
      expect(fake.waiting()).toBe(0);
      const before = fake.requests();
      fake.frames(5, 100);
      expect(fake.requests()).toBe(before);
    } finally {
      await act(async () => {
        document.documentElement.removeAttribute('data-hidden');
        await Promise.resolve();
      });
    }
    expect(fake.waiting()).toBe(1);
    fake.frame(5_000);
    fake.frame(100);
    act(() => playButton(container).click());
    expect(slider(container).getAttribute('aria-valuenow')).toBe('5.5');
  });

  it('the second speed doubles the advance and a stalled frame advances by 100 ms × rate', () => {
    const fake = fakeClock();
    const { container } = mount({ clock: fake.clock });
    const fast = byRole(container, 'button', '10×') as HTMLButtonElement;
    act(() => fast.click());
    act(() => playButton(container).click());
    fake.frame(16);
    fake.frames(10, 100);
    expect(slider(container).getAttribute('aria-valuenow')).toBe('10');
    fake.frame(5_000);
    act(() => playButton(container).click());
    expect(slider(container).getAttribute('aria-valuenow')).toBe('11');
  });
});

describe('AgentTimeline motion', () => {
  it('pops a glyph only when a playing replay first reaches its span, not again on resume', () => {
    const previous = skipAnimations(false);
    try {
      const fake = fakeClock();
      const { container } = mount({ clock: fake.clock });
      const retried = (): HTMLButtonElement => spanNamed(container, 'Main agent, Reasoning');
      act(() => playButton(container).click());
      expect(retried().hasAttribute('data-arrive')).toBe(false);
      fake.frame(16);
      fake.frames(3, 100);
      expect(retried().hasAttribute('data-arrive')).toBe(true);
      expect(spanNamed(container, 'Map the scheduler queues, Tool call').hasAttribute('data-arrive')).toBe(false);
      act(() => playButton(container).click());
      act(() => playButton(container).click());
      expect(retried().hasAttribute('data-arrive')).toBe(true);
      key(slider(container), 'Home');
      expect(retried().hasAttribute('data-arrive')).toBe(false);
    } finally {
      skipAnimations(previous);
    }
    const glyphRules = cssRules(CSS).filter(
      (rule) => rule.decls.has('animation') && rule.selectors.some((selector) => selector.endsWith('.glyph')),
    );
    expect(glyphRules.length).toBeGreaterThan(0);
    for (const rule of glyphRules) {
      for (const selector of rule.selectors) expect(selector.includes('data-playing'), selector).toBe(false);
    }
  });

  it('swaps the replay time by character from a minute on, inside the reserved box', () => {
    const previous = skipAnimations(false);
    try {
      const { container } = mount();
      const timePart = (): HTMLElement => statusLine(container).children[1] as HTMLElement;
      key(slider(container), 'ArrowLeft');
      expect(statusText(container).startsWith('Paused · 1:10 of 1m 12s')).toBe(true);
      const clock = timePart().querySelector('[data-granularity="char"]');
      expect(clock?.textContent).toBe('1:10');
      expect(timePart().querySelector('[data-reserve]')).not.toBeNull();
      key(slider(container), 'ArrowLeft');
      expect(timePart().querySelector('[data-granularity="char"]')?.textContent).toBe('1:08');
      expect(timePart().querySelectorAll('[data-granularity="char"] [data-prev]').length).toBeGreaterThan(0);
      key(slider(container), 'Home');
      expect(timePart().querySelector('[data-granularity]')).toBeNull();
    } finally {
      skipAnimations(previous);
    }
  });

  it('wipes the lanes in at most four stagger steps', () => {
    const lanes: AgentLane[] = Array.from({ length: 6 }, (_, index) => ({
      id: index === 0 ? 'main' : `c${String(index)}`,
      name: `Lane ${String(index)}`,
      color: 'var(--lane-main)',
    }));
    const { container } = mount({ lanes, handoffs: [], reports: [] });
    const steps = [...plot(container).querySelectorAll<HTMLElement>('[role="group"][aria-labelledby]')].map((lane) =>
      Number(lane.style.getPropertyValue('--step')),
    );
    expect(steps).toEqual([0, 1, 2, 3, 3, 3]);
  });

  it('draws handoff links with a mask, not with borders that snap to whole pixels', () => {
    const links = cssRules(CSS).filter((rule) => rule.selectors.some((selector) => selector.startsWith('.link')));
    expect(links.some((rule) => rule.decls.has('mask'))).toBe(true);
    for (const rule of links) {
      for (const property of rule.decls.keys()) expect(property.startsWith('border'), property).toBe(false);
    }
  });
});

describe('AgentTimeline live and stale', () => {
  const running: AgentSpan[] = [
    {
      id: 'm0',
      laneId: 'main',
      kind: 'wait',
      startMs: 0,
      endMs: 500,
      status: 'ok',
      label: 'Waiting for the first token',
    },
    {
      id: 'm1',
      laneId: 'main',
      kind: 'wait',
      startMs: 600,
      endMs: null,
      status: 'running',
      label: 'Waiting for subagents',
    },
    { id: 'k0', laneId: 'kid', kind: 'tool', startMs: 700, endMs: null, status: 'running', label: 'Tool call · grep' },
  ];

  it('follows the clock, disables replay and steps once a second under static motion', () => {
    const fake = fakeClock(5_000_000);
    const mode: AgentTimelineMode = { kind: 'live', originMs: 5_000_000 - 1_000 };
    const { container, rerender } = mount({
      clock: fake.clock,
      mode,
      spans: running,
      handoffs: [],
      reports: [],
      durationMs: 800,
      outcome: { word: 'In progress', tone: 'neutral' },
    });
    const root = section(container);
    expect(root.getAttribute('data-mode')).toBe('live');
    expect(spanButtons(container).filter((button) => button.dataset.status === 'running')).toHaveLength(2);
    expect(playButton(container).disabled).toBe(true);
    expect(byRole(container, 'button', 'Restart')?.hasAttribute('disabled')).toBe(true);
    expect(byRole(container, 'group', 'Playback speed')).toBeNull();
    expect(slider(container).getAttribute('aria-disabled')).toBe('true');
    expect(slider(container).tabIndex).toBe(-1);
    expect(statusLine(container).hasAttribute('aria-hidden')).toBe(false);
    expect(root.hasAttribute('aria-describedby')).toBe(false);
    const track = plot(container).parentElement as HTMLElement;
    expect(track.style.getPropertyValue('--now')).toBe('1');
    fake.frame(16);
    fake.frames(24, 100);
    expect(track.style.getPropertyValue('--now')).toBe('3');
    expect(statusText(container)).toBe('Live · 3 s, 2 agents working');

    rerender({ mode: { kind: 'stale', atMs: 3_000, asOfText: '14:32:10' } });
    expect(liveRegion(container)).toBe('No update for 10 seconds');
    const tool = spanNamed(container, 'Map the scheduler queues, Tool call');
    expect(tool.dataset.status).toBe('unfinished');
    expect(tool.dataset.open).toBe('end');
    expect(tool.getAttribute('aria-label')).toContain('running as of 14:32:10');
    expect(statusText(container)).toBe('In progress · as of 14:32:10');
    expect(fake.waiting()).toBe(0);

    rerender({
      mode: { kind: 'recorded' },
      spans: running.map((span) => ({ ...span, endMs: 2_500, status: 'ok' as const })),
      durationMs: 2_500,
      outcome: { word: 'Finished', tone: 'success' },
    });
    expect(liveRegion(container)).toBe('Turn finished: Finished, 2.50 s');
    expect(byRole(container, 'group', 'Playback speed')).not.toBeNull();
    expect(root.hasAttribute('data-settled')).toBe(true);
  });

  it('announces a turn that stopped without an end as not finished, never as finished', () => {
    const fake = fakeClock(5_000_000);
    const { container, rerender } = mount({
      clock: fake.clock,
      mode: { kind: 'live', originMs: 5_000_000 - 1_000 },
      spans: running,
      handoffs: [],
      reports: [],
      durationMs: 800,
      outcome: { word: 'In progress', tone: 'neutral' },
    });
    rerender({ mode: { kind: 'stale', atMs: 1_000, asOfText: '14:32:10' } });
    rerender({
      mode: { kind: 'recorded' },
      spans: running.map((span) =>
        span.endMs === null ? { ...span, endMs: 800, status: 'unfinished' as const, open: 'end' as const } : span,
      ),
      outcome: { word: 'Not finished', tone: 'neutral', detail: 'last event 14:32:10' },
    });
    expect(liveRegion(container)).toBe('Not finished: last event 14:32:10');
  });

  it('swaps the tick texts in place when the live axis steps', () => {
    const previous = skipAnimations(false);
    try {
      const fake = fakeClock(5_000_000);
      const { container } = mount({
        clock: fake.clock,
        mode: { kind: 'live', originMs: 5_000_000 - 17_500 },
        spans: running,
        handoffs: [],
        reports: [],
        durationMs: 800,
        outcome: { word: 'In progress', tone: 'neutral' },
      });
      const track = plot(container).parentElement as HTMLElement;
      const ticks = (): HTMLElement[] => [
        ...(plot(container).nextElementSibling as HTMLElement).querySelectorAll<HTMLElement>(':scope > [style*="--t"]'),
      ];
      expect(track.style.getPropertyValue('--axis')).toBe('20');
      const before = ticks().map((tick) => tick.textContent);
      expect(section(container).querySelectorAll('[data-prev]')).toHaveLength(0);
      fake.frames(10, 100);
      expect(track.style.getPropertyValue('--axis')).toBe('50');
      const after = ticks();
      expect(after.map((tick) => tick.textContent)).not.toEqual(before);
      const swapped = after.filter((tick, index) => index < before.length && tick.textContent !== before[index]);
      expect(swapped.length).toBeGreaterThan(0);
      for (const tick of swapped) {
        expect(tick.querySelector('[data-granularity="text"] [data-prev]'), tick.textContent ?? '').not.toBeNull();
      }
    } finally {
      skipAnimations(previous);
    }
  });

  it('keeps the axis the live clock reached when the turn goes stale', () => {
    const fake = fakeClock(5_000_000);
    const { container, rerender } = mount({
      clock: fake.clock,
      mode: { kind: 'live', originMs: 5_000_000 - 95_000 },
      spans: running,
      handoffs: [],
      reports: [],
      durationMs: 800,
      outcome: { word: 'In progress', tone: 'neutral' },
    });
    const track = plot(container).parentElement as HTMLElement;
    expect(track.style.getPropertyValue('--axis')).toBe('120');
    rerender({ mode: { kind: 'stale', atMs: 88_000, asOfText: '14:32:10' } });
    expect(track.style.getPropertyValue('--axis')).toBe('120');
    const cached = mount({
      mode: { kind: 'stale', atMs: 88_000, asOfText: '14:32:10' },
      spans: running,
      handoffs: [],
      reports: [],
      durationMs: 800,
      outcome: { word: 'In progress', tone: 'neutral' },
    });
    expect((plot(cached.container).parentElement as HTMLElement).style.getPropertyValue('--axis')).toBe('100');
  });

  it('marks the last unfinished span of a lane as its tail', () => {
    const { container } = mount();
    expect(spanNamed(container, 'Bisect the cache').hasAttribute('data-tail')).toBe(true);
    expect(spanNamed(container, 'Main agent, Writing the answer').hasAttribute('data-tail')).toBe(false);
  });
});

describe('AgentTimeline end state under static motion', () => {
  it('mounts settled with no entrance, painted links and ports, and a card in place on focus', () => {
    const { container } = mount();
    const root = section(container);
    expect(root.getAttribute('data-entrance')).toBe('done');
    expect(root.querySelectorAll('[data-arrive]')).toHaveLength(0);
    expect(root.querySelectorAll('[data-kind="spawn"], [data-kind="return"]')).toHaveLength(4);
    layoutSpans(() => ({ left: 300, top: 300 }));
    const tool = spanNamed(container, 'Map the scheduler queues, Tool call');
    act(() => tool.focus());
    expect(shownCard()?.textContent).toContain('Tool call · grep');
    expect(shownCard()?.textContent).toContain('412 ms · 3.2 s → 3.61 s');
  });

  it('draws the card on the body, out of every box of the timeline, and keeps it out of the accessibility tree', () => {
    const { container } = mount();
    layoutSpans(() => ({ left: 300, top: 300 }));
    act(() => spanNamed(container, 'Map the scheduler queues, Tool call').focus());
    const card = shownCard();
    expect(card).not.toBeNull();
    expect(section(container).contains(card)).toBe(false);
    expect(card!.parentElement!.parentElement).toBe(document.body);
    expect(card!.parentElement!.getAttribute('aria-hidden')).toBe('true');
  });

  it('a selected card follows its span on a scroll and is removed while the span is out of view', async () => {
    const { container } = mount();
    let at: { left: number; top: number } = { left: 300, top: 300 };
    layoutSpans(() => at);
    act(() => spanNamed(container, 'Main agent, Writing the answer').click());
    expect(cardY()).toBe(300 - 6);
    at = { left: 300, top: 500 };
    await act(async () => {
      window.dispatchEvent(new Event('scroll'));
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    expect(cardY()).toBe(500 - 6);
    at = { left: -80, top: 500 };
    await act(async () => {
      plot(container).parentElement!.parentElement!.parentElement!.dispatchEvent(new Event('scroll'));
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    expect(shownCard()).toBeNull();
    at = { left: 300, top: 400 };
    await act(async () => {
      window.dispatchEvent(new Event('resize'));
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    expect(cardY()).toBe(400 - 6);
  });

  it('writes offsets on a minutes axis in seconds under a minute, then m:ss; ticks keep the axis unit', () => {
    const spans: AgentSpan[] = [
      {
        id: 'a',
        laneId: 'main',
        kind: 'tool',
        startMs: 12_400,
        endMs: 25_000,
        status: 'ok',
        label: 'Tool call · grep',
      },
      {
        id: 'b',
        laneId: 'main',
        kind: 'tool',
        startMs: 133_200,
        endMs: 146_400,
        status: 'ok',
        label: 'Tool call · ls',
      },
    ];
    const { container } = mount({
      spans,
      handoffs: [{ id: 'kid:spawn:0', laneId: 'kid', kind: 'spawn', atMs: 61_500 }],
      reports: [],
      durationMs: 150_000,
    });
    expect(spanNamed(container, 'Main agent, Tool call · grep').getAttribute('aria-label')).toBe(
      'Main agent, Tool call · grep, 12.6 s, from 12.4 s to 25 s',
    );
    expect(spanNamed(container, 'Main agent, Tool call · ls').getAttribute('aria-label')).toBe(
      'Main agent, Tool call · ls, 13.2 s, from 2:13 to 2:26',
    );
    expect(tableRows(container, LABEL).map((row) => row.slice(3, 5))).toEqual([
      ['12.4 s', '25 s'],
      ['2:13', '2:26'],
    ]);
    expect(tableRows(container, `${LABEL}: handoffs`)[0]![2]).toBe('1:01');
    layoutSpans(() => ({ left: 300, top: 300 }));
    act(() => spanNamed(container, 'Main agent, Tool call · ls').focus());
    expect(shownCard()?.textContent).toContain('13.2 s · 2:13 → 2:26');
    const ticks = [
      ...(plot(container).nextElementSibling as HTMLElement).querySelectorAll<HTMLElement>(':scope > [style*="--t"]'),
    ].map((tick) => tick.textContent ?? '');
    expect(ticks.slice(1).every((text) => text.endsWith(' min'))).toBe(true);
  });

  it('a narrow container hides nothing from the table and keeps the model in it', () => {
    const box = document.createElement('div');
    box.style.width = '360px';
    document.body.appendChild(box);
    const { container } = mount();
    box.appendChild(container);
    const rows = tableRows(container, LABEL);
    expect(rows).toHaveLength(SPANS.length);
    expect(rows.some((row) => row[0]!.includes('Qwen3-8B-4bit'))).toBe(true);
  });

  it('draws at most 400 bars and lists all 500 in the table', () => {
    const many: AgentSpan[] = Array.from({ length: 500 }, (_, index) => ({
      id: `s${String(index)}`,
      laneId: 'main',
      kind: 'tool',
      startMs: index * 10,
      endMs: index * 10 + 5 + (index % 2),
      status: 'ok',
      label: 'Tool call · ls',
    }));
    const { container } = mount({ spans: many, handoffs: [], durationMs: 5_000 });
    expect(spanButtons(container)).toHaveLength(400);
    expect(section(container).textContent).toContain('+100 short spans not drawn');
    expect(tableRows(container, LABEL)).toHaveLength(500);
    expect(allByRole(container, 'region', LABEL)).toHaveLength(1);
  });
});

describe('AgentTimeline woken turn', () => {
  const model = timelineModel(wokenTimeline(2), { fresh: false, formatDuration });
  const woken = (props: Partial<AgentTimelineProps> = {}): Mounted =>
    mount({
      title: 'Woken by workflow deep-research',
      lanes: model.lanes,
      spans: model.spans,
      handoffs: model.handoffs,
      reports: model.reports,
      durationMs: model.durationMs,
      turnStartMs: model.turnStartMs,
      outcome: model.outcome,
      ...props,
    });

  it('counts the turn alone in the status line and the final sentence', () => {
    const { container } = woken();
    expect(statusText(container)).toBe(
      'Finished · 4m 12s · after 33m 51s idle · 1,502,141 tokens · 1,697,987 in 7 subagents before the turn',
    );
    expect(summary(container)).toBe(
      'Finished. 4m 12s, after 33m 51s idle. 1,502,141 tokens, 1,697,987 in 7 subagents before the turn.',
    );
  });

  it('draws the idle stretch as a named span, in the legend and the table, with the turn start rule', () => {
    const { container } = woken();
    const idle = spanNamed(container, 'Main agent, Idle between turns');
    expect(idle.getAttribute('aria-label')).toBe('Main agent, Idle between turns, 33m 51s, from 0 s to 33:50');
    expect(idle.dataset.kind).toBe('idle');
    expect(idle.tagName).toBe('BUTTON');
    const legend = byRole(container, 'list', 'Legend')?.textContent ?? '';
    expect(legend).toContain('WaitingIdle');
    expect(tableRows(container, LABEL)[0]).toEqual([
      'Main agent · main-model',
      'Idle',
      'Idle between turns',
      '0 s',
      '33:50',
      '33m 51s',
      'ok',
    ]);
    const rule = plot(container).querySelector<HTMLElement>('[data-turn-start]');
    expect(rule?.getAttribute('aria-hidden')).toBe('true');
    expect(rule?.style.getPropertyValue('--t')).toBe('2030.588');
    expect(rule?.hasAttribute('data-pending')).toBe(false);
  });

  it('paints the Idle legend swatch with the line of the idle span', () => {
    const rules = cssRules(CSS);
    for (const property of ['--idle-line', 'background', 'border-radius']) {
      const line = cssValue(rules, ".span[data-kind='idle']", property);
      expect(line).toBeDefined();
      expect(cssValue(rules, ".swatch[data-swatch='idle']", property)).toBe(line);
    }
  });

  it('replays the whole axis from its start', () => {
    const fake = fakeClock();
    const { container } = woken({ clock: fake.clock });
    expect(slider(container).getAttribute('aria-valuemax')).toBe('2282.925');
    act(() => playButton(container).click());
    expect(slider(container).getAttribute('aria-valuenow')).toBe('0');
    expect(statusText(container).startsWith('Replaying · 0 s of 38m 03s')).toBe(true);
    fake.frames(30, 100);
    expect(statusText(container)).toContain('122,388 in subagents before the turn');
  });

  it('live: the time counts from the turn start', () => {
    const fake = fakeClock(5_000_000);
    // Under static motion the live clock moves in whole seconds of the axis: 2,034 s here, 3.4 s after the turn start.
    const origin = 5_000_000 - 2_034_000;
    const { container } = woken({
      clock: fake.clock,
      mode: { kind: 'live', originMs: origin },
      durationMs: model.turnStartMs + 2_000,
      outcome: { word: 'In progress', tone: 'neutral' },
    });
    expect(statusText(container)).toBe(
      'Live · 3 s · after 33m 51s idle · 1,697,987 in subagents before the turn, 1 agent working',
    );
  });

  it('a turn without a turn start draws no rule and says nothing of idle time', () => {
    const { container } = mount();
    expect(plot(container).querySelector('[data-turn-start]')).toBeNull();
    expect(statusText(container)).not.toContain('idle');
    expect(summary(container)).not.toContain('idle');
  });
});
