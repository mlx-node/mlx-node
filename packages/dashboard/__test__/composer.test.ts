/**
 * @vitest-environment happy-dom
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { act, createElement, useState, type ReactElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  Composer,
  composerCanSend,
  contextText,
  sendFace,
  type ComposerProps,
} from '../ui/src/components/arc-x/composer/composer.js';
import { CHAT } from '../ui/src/lib/chat-strings.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { byRole } from './render.js';

afterEach(unmountAll);

function props(extra: Partial<ComposerProps> = {}): ComposerProps {
  return {
    value: '',
    onValueChange: () => undefined,
    placeholder: 'Ask anything',
    phase: 'idle',
    onSend: () => undefined,
    onStop: () => undefined,
    tools: createElement('span', null, 'TOOLS'),
    ...extra,
  };
}

function key(target: Element, init: KeyboardEventInit): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }));
  });
}

describe('composerCanSend and sendFace', () => {
  it('sends only idle, unblocked, non-blank text within the limit', () => {
    expect(composerCanSend('hi', 'idle', undefined, 10)).toBe(true);
    expect(composerCanSend('  \n ', 'idle', undefined, 10)).toBe(false);
    expect(composerCanSend('hi', 'running', undefined, 10)).toBe(false);
    expect(composerCanSend('hi', 'idle', 'Engine stopped', 10)).toBe(false);
    expect(composerCanSend('x'.repeat(11), 'idle', undefined, 10)).toBe(false);
  });

  it('the button face per phase', () => {
    expect(sendFace('idle', false, 'Engine stopped')).toMatchObject({
      glyph: 'send',
      enabled: false,
      tip: 'Engine stopped',
    });
    expect(sendFace('idle', true).tip).toBe(CHAT.sendTip);
    expect(sendFace('idle', false).tip).toBe(CHAT.sendTip);
    expect(sendFace('running', false).tip).toBe(CHAT.stopTip);
    expect(CHAT.sendTip).toContain('↵');
    expect(CHAT.stopTip).toContain('Esc');
    expect(sendFace('creating', false)).toMatchObject({ enabled: false, spin: true });
    expect(sendFace('running', false)).toMatchObject({ glyph: 'stop', enabled: true, name: 'Stop' });
    expect(sendFace('cancelling', false)).toMatchObject({ enabled: false, name: 'Stopping' });
  });
});

describe('Composer', () => {
  function Controlled(extra: Partial<ComposerProps>): ReactElement {
    const [value, setValue] = useState(extra.value ?? '');
    return createElement(Composer, props({ ...extra, value, onValueChange: setValue }));
  }

  it('Enter sends, Shift+Enter does not, an IME composition does not', () => {
    const onSend = vi.fn();
    const container = mount(createElement(Composer, props({ value: 'hello', onSend })));
    const area = byRole(container, 'textbox', 'Message')!;
    key(area, { key: 'Enter', shiftKey: true });
    key(area, { key: 'Enter', isComposing: true });
    expect(onSend).not.toHaveBeenCalled();
    key(area, { key: 'Enter' });
    expect(onSend).toHaveBeenCalledTimes(1);
    act(() => byRole(container, 'button', 'Send')!.click());
    expect(onSend).toHaveBeenCalledTimes(2);
  });

  it('blank text does not send; Esc stops a running turn and the button becomes Stop', () => {
    const onSend = vi.fn();
    const onStop = vi.fn();
    const container = mount(createElement(Composer, props({ value: '   ', onSend, onStop })));
    key(byRole(container, 'textbox', 'Message')!, { key: 'Enter' });
    expect(onSend).not.toHaveBeenCalled();
    expect((byRole(container, 'button', 'Send') as HTMLButtonElement).disabled).toBe(true);
    rerender(createElement(Composer, props({ value: 'next', phase: 'running', onSend, onStop })));
    key(byRole(container, 'textbox', 'Message')!, { key: 'Escape' });
    expect(onStop).toHaveBeenCalledTimes(1);
    act(() => byRole(container, 'button', 'Stop')!.click());
    expect(onStop).toHaveBeenCalledTimes(2);
    expect(onSend).not.toHaveBeenCalled();
  });

  it('Esc stops once: a held key repeats nothing, an IME composition stops nothing', () => {
    const onStop = vi.fn();
    const container = mount(createElement(Composer, props({ value: 'x', phase: 'starting', onStop })));
    const area = byRole(container, 'textbox', 'Message')!;
    key(area, { key: 'Escape', repeat: true });
    key(area, { key: 'Escape', isComposing: true });
    expect(onStop).not.toHaveBeenCalled();
    key(area, { key: 'Escape' });
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it('the tools row has no key legend: the keys are in the button tooltip', () => {
    const container = mount(createElement(Composer, props({ value: 'x' })));
    const row = container.querySelector('[data-side]')!;
    expect(container.querySelector('kbd')).toBeNull();
    expect(row.textContent).toBe('TOOLS');
    const area = byRole(container, 'textbox', 'Message') as HTMLTextAreaElement;
    act(() => area.focus());
    expect(row.textContent).toBe('TOOLS');
    rerender(createElement(Composer, props({ value: '', phase: 'running' })));
    expect(row.textContent).toBe('TOOLS');
  });

  it('the empty text field shows the new line keys beside it; text in the field removes them', () => {
    const hint = `${CHAT.keyNames.shift} ${CHAT.keyNames.enter} ${CHAT.newLine}`;
    const container = mount(createElement(Composer, props()));
    const area = byRole(container, 'textbox', 'Message')!;
    const part = area.parentElement!.querySelector('span')!;
    expect(part.textContent).toBe(hint);
    const keys = part.querySelector('kbd')!;
    expect(keys.querySelectorAll('svg[aria-hidden="true"]').length).toBe(2);
    expect(container.textContent).not.toContain('↵');
    expect(container.textContent).not.toContain('⇧');
    rerender(createElement(Composer, props({ value: ' ' })));
    expect(container.textContent).not.toContain(CHAT.newLine);
    expect(container.querySelector('kbd')).toBeNull();
  });

  it('the side text: the reason when there is text, the counter near the limit, else nothing', () => {
    const container = mount(createElement(Composer, props({ value: '', blocked: 'Engine stopped' })));
    expect(container.querySelector('[data-side]')!.textContent).toBe('TOOLS');
    rerender(createElement(Composer, props({ value: 'x', blocked: 'Engine stopped' })));
    expect(container.textContent).toContain('Engine stopped');
    rerender(createElement(Composer, props({ value: 'x'.repeat(95), maxLength: 100 })));
    expect(container.textContent).toContain('95 / 100');
    rerender(createElement(Composer, props({ value: 'x'.repeat(101), maxLength: 100 })));
    expect(container.textContent).toContain('Too long');
  });

  it('the textarea is described by the reason that keeps Send off', () => {
    const container = mount(createElement(Composer, props({ value: '/help', blocked: 'Start with a word' })));
    const area = container.querySelector('textarea')!;
    const id = area.getAttribute('aria-describedby');
    expect(id).not.toBeNull();
    expect(container.querySelector(`[id="${id}"]`)?.textContent).toBe('Start with a word');
    rerender(createElement(Composer, props({ value: 'fine' })));
    expect(area.hasAttribute('aria-describedby')).toBe(false);
  });

  it('a notice is a status, a danger notice an alert, with its actions', () => {
    const onSelect = vi.fn();
    const container = mount(
      createElement(Controlled, {
        notice: { tone: 'warning', text: 'Engine starting', action: { label: 'Open Engine', onSelect } },
      }),
    );
    expect(byRole(container, 'status')?.textContent).toContain('Engine starting');
    act(() => byRole(container, 'button', 'Open Engine')!.click());
    expect(onSelect).toHaveBeenCalled();
    rerender(createElement(Composer, props({ notice: { tone: 'danger', text: 'Agent missing' } })));
    expect(byRole(container, 'alert')?.textContent).toContain('Agent missing');
  });

  it('typing updates the value; the tools row is shown', () => {
    const container = mount(createElement(Controlled, {}));
    const area = byRole(container, 'textbox', 'Message') as HTMLTextAreaElement;
    act(() => {
      Reflect.set(HTMLTextAreaElement.prototype, 'value', 'typed', area);
      area.dispatchEvent(new Event('input', { bubbles: true }));
    });
    expect(area.value).toBe('typed');
    expect((byRole(container, 'button', 'Send') as HTMLButtonElement).disabled).toBe(false);
    expect(container.textContent).toContain('TOOLS');
  });
});

describe('Composer tools row', () => {
  const rules = cssRules('components/arc-x/composer/composer.module.css');
  const value = (selector: string, property: string): string | undefined => cssValue(rules, selector, property);

  it('Send has its own column at the right edge, outside the box that shrinks', () => {
    expect(value('.tools', 'display')).toBe('grid');
    expect(value('.tools', 'grid-template-columns')).toBe('minmax(0, 1fr) auto');
    const container = mount(createElement(Composer, props()));
    const send = byRole(container, 'button', 'Send')!;
    const fit = container.querySelector('[data-side]')!;
    expect(fit.contains(send)).toBe(false);
    expect(fit.parentElement!.contains(send)).toBe(true);
  });

  it('the new line hint lets a click through to the text field and is one line', () => {
    expect([value('.newLine', 'pointer-events'), value('.newLine', 'white-space')]).toEqual(['none', 'nowrap']);
    expect(value('.field', 'position')).toBe('relative');
  });

  it('the tools row is one clipped line; a reason or a counter can grow it', () => {
    expect([value('.fit', 'flex-wrap'), value('.fit', 'height'), value('.fit', 'overflow')]).toEqual([
      'wrap',
      'var(--control-height-sm)',
      'clip',
    ]);
    expect(value(".fit[data-side='reason']", 'flex-wrap')).toBe('nowrap');
  });

  it('the chips give way in order: folder, then web, then the model, each down to its floor', () => {
    const shrink = (give: number): number =>
      Number(value(`.toolsStart > [data-give='${String(give)}']`, 'flex-shrink'));
    expect(shrink(1)).toBeGreaterThan(shrink(2) * 1000);
    expect(shrink(2)).toBeGreaterThan(shrink(3) * 1000);
    expect([1, 2, 3].map((give) => value(`.toolsStart > [data-give='${String(give)}']`, 'min-width'))).toEqual([
      '4.75rem',
      '4.75rem',
      '4.25rem',
    ]);
  });

  it('a part marked data-fit-whole is drawn on the row line or not at all, so its room goes back to the chips', () => {
    expect(value('.toolsStart [data-fit-whole][data-dropped]', 'display')).toBe('none');
    let partTop = 36;
    const box = (top: number, height: number): DOMRect =>
      ({ top, height, bottom: top + height, left: 0, right: 0, width: 0, x: 0, y: top }) as DOMRect;
    const spy = vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function rect(this: Element) {
      return this.hasAttribute('data-fit-whole') ? box(partTop, 20) : box(0, 28);
    });
    try {
      const tools = createElement('span', { 'data-fit-whole': '' }, 'Q4_K_M');
      const container = mount(createElement(Composer, props({ tools })));
      const part = container.querySelector('[data-fit-whole]')!;
      expect(part.hasAttribute('data-dropped')).toBe(true);
      partTop = 4;
      rerender(createElement(Composer, props({ tools })));
      expect(part.hasAttribute('data-dropped')).toBe(false);
    } finally {
      spy.mockRestore();
    }
  });
});

describe('Composer context ring', () => {
  const rules = cssRules('components/arc-x/composer/composer.module.css');
  const value = (selector: string, property: string): string | undefined => cssValue(rules, selector, property);
  const ring = (container: HTMLElement): HTMLElement | null =>
    container.querySelector<HTMLElement>('[role="img"][aria-label^="Context:"]');
  const arc = (container: HTMLElement): string | null =>
    ring(container)?.querySelectorAll('circle')[1]?.getAttribute('stroke-dasharray') ?? null;

  it('the sentence: whole numbers with separators and a whole percent, <1% for a tiny share, 100% past the window', () => {
    expect(contextText({ used: 11_971, window: 262_144 })).toBe('Context: 11,971 of 262,144 tokens (5%)');
    expect(contextText({ used: 1, window: 262_144 })).toBe('Context: 1 of 262,144 tokens (<1%)');
    expect(contextText({ used: 300_000, window: 262_144 })).toBe('Context: 300,000 of 262,144 tokens (100%)');
  });

  it('a ring with no text: the numbers are only in its name, and it takes keyboard focus for its tooltip', () => {
    const container = mount(createElement(Composer, props({ context: { used: 11_971, window: 262_144 } })));
    const hint = byRole(container, 'img', 'Context: 11,971 of 262,144 tokens (5%)')!;
    expect(hint).not.toBeNull();
    expect(hint.tabIndex).toBe(0);
    expect(hint.querySelector('svg')!.getAttribute('aria-hidden')).toBe('true');
    const row = container.querySelector('[data-side]')!.parentElement!;
    expect(row.contains(hint)).toBe(true);
    const text = row.textContent ?? '';
    for (const ch of text) expect(ch >= '0' && ch <= '9').toBe(false);
    expect(text).toBe('TOOLS');
    // At the end of the row beside Send, outside the clipped box the chips shrink in.
    expect(container.querySelector('[data-side]')!.contains(hint)).toBe(false);
    expect(hint.nextElementSibling?.contains(byRole(container, 'button', 'Send')!)).toBe(true);
    expect(document.body.textContent).not.toContain('11,971');
    act(() => hint.focus());
    expect(document.activeElement).toBe(hint);
    expect(document.body.textContent).toContain('Context: 11,971 of 262,144 tokens (5%)');
  });

  it('the arc covers the used share, never less than a small minimum above 0, and stops at full', () => {
    const container = mount(createElement(Composer, props({ context: { used: 25, window: 100 } })));
    expect(arc(container)).toBe('25 100');
    rerender(createElement(Composer, props({ context: { used: 100, window: 100 } })));
    expect(arc(container)).toBe('100 100');
    rerender(createElement(Composer, props({ context: { used: 500, window: 100 } })));
    expect(arc(container)).toBe('100 100');
    rerender(createElement(Composer, props({ context: { used: 1, window: 262_144 } })));
    expect(arc(container)).toBe('4 100');
    const circle = ring(container)!.querySelectorAll('circle')[1]!;
    expect([circle.getAttribute('pathLength'), circle.getAttribute('transform')]).toEqual(['100', 'rotate(-90 9 9)']);
    rerender(createElement(Composer, props({ context: { used: 0, window: 100 } })));
    expect(ring(container)!.querySelectorAll('circle')).toHaveLength(1);
  });

  it('the arc turns warning at 80 % and danger at 95 %', () => {
    const tone = (used: number): string | null => {
      const container = mount(createElement(Composer, props({ context: { used, window: 1000 } })));
      return ring(container)!.getAttribute('data-tone');
    };
    expect([tone(799), tone(800), tone(949), tone(950), tone(1200)]).toEqual([
      null,
      'warning',
      'warning',
      'danger',
      'danger',
    ]);
    expect(value('.contextArc', 'stroke')).toBe('currentColor');
    expect(value('.contextTrack', 'stroke')).toBe('var(--border-strong)');
    expect(value('.context', 'color')).toBe('var(--info)');
    expect(value(".context[data-tone='warning']", 'color')).toBe('var(--warning)');
    expect(value(".context[data-tone='danger']", 'color')).toBe('var(--danger)');
  });

  it('is not drawn without the prop, beside a reason, or beside the counter', () => {
    const context = { used: 11_971, window: 262_144 };
    const container = mount(createElement(Composer, props()));
    expect(ring(container)).toBeNull();
    rerender(createElement(Composer, props({ value: 'x', blocked: 'Engine stopped', context })));
    expect(container.textContent).toContain('Engine stopped');
    expect(ring(container)).toBeNull();
    rerender(createElement(Composer, props({ value: 'x'.repeat(95), maxLength: 100, context })));
    expect(container.textContent).toContain('95 / 100');
    expect(ring(container)).toBeNull();
    rerender(createElement(Composer, props({ value: 'x', context })));
    expect(ring(container)).not.toBeNull();
  });

  it('is not drawn under 320 px of composer width', () => {
    const source = readFileSync(
      join(import.meta.dirname, '../ui/src/components/arc-x/composer/composer.module.css'),
      'utf8',
    );
    const head = '@container composer (max-width: 319px)';
    const start = source.indexOf(head);
    expect(start).toBeGreaterThan(-1);
    const open = source.indexOf('{', start);
    let depth = 1;
    let end = open + 1;
    while (end < source.length && depth > 0) {
      if (source[end] === '{') depth += 1;
      else if (source[end] === '}') depth -= 1;
      end += 1;
    }
    const narrow = cssRules('components/arc-x/composer/composer.module.css').filter(
      (rule) => rule.conditional && rule.selectors.includes('.context'),
    );
    expect(narrow.map((rule) => rule.decls.get('display'))).toEqual(['none']);
    expect(source.slice(open, end).includes('.context {')).toBe(true);
    expect(value('.context', 'display')).toBe('inline-flex');
  });
});
