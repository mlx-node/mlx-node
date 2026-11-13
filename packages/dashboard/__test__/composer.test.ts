/**
 * @vitest-environment happy-dom
 */

import { act, createElement, useState, type ReactElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  Composer,
  composerCanSend,
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
    hint: CHAT.hintSend,
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

  it('a focus-only hint shows only while the text field has the focus', () => {
    const container = mount(
      createElement(Composer, props({ value: '', phase: 'running', hint: CHAT.hintStop, hintOnFocus: true })),
    );
    const stop = `${CHAT.keyNames.esc} ${CHAT.hintStop[0].label}`;
    const area = byRole(container, 'textbox', 'Message') as HTMLTextAreaElement;
    expect(container.textContent).not.toContain(stop);
    act(() => area.focus());
    expect(container.textContent).toContain(stop);
    act(() => area.blur());
    expect(container.textContent).not.toContain(stop);
  });

  it('the legend: one part per key hint, Send first; each key an icon with its hidden name; no key glyph as text', () => {
    const container = mount(createElement(Composer, props()));
    const parts = [...container.querySelectorAll('form > div:last-child > div > span')].filter(
      (part) => part.querySelector('kbd') !== null,
    );
    expect(parts.map((part) => part.textContent)).toEqual([
      `${CHAT.keyNames.enter} ${CHAT.hintSend[0].label}`,
      `${CHAT.keyNames.shift} ${CHAT.keyNames.enter} ${CHAT.hintSend[1].label}`,
    ]);
    for (const part of parts) {
      const keys = part.querySelector('kbd')!;
      expect(keys.querySelectorAll('svg[aria-hidden="true"]').length).toBe(keys.querySelectorAll('span').length);
    }
    expect(container.textContent).not.toContain('↵');
    expect(container.textContent).not.toContain('⇧');
  });

  it('the side text: the reason when there is text, the counter near the limit, else the hint', () => {
    const container = mount(createElement(Composer, props({ value: '', blocked: 'Engine stopped' })));
    expect(container.textContent).toContain(CHAT.hintSend[0].label);
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

  it('a legend part is never cut: no ellipsis, one line, whole or wrapped out of the clipped box', () => {
    expect([value('.hint', 'flex'), value('.hint', 'white-space')]).toEqual(['none', 'nowrap']);
    for (const property of ['overflow', 'text-overflow']) expect(value('.hint', property)).toBeUndefined();
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

  it('the key icon and its word share one centre line', () => {
    expect([value('.hint', 'display'), value('.hint', 'align-items'), value('.hint', 'line-height')]).toEqual([
      'inline-flex',
      'center',
      '1',
    ]);
    expect([value('.keys', 'align-items'), value('.keys > svg', 'display'), value('.keys > svg', 'width')]).toEqual([
      'center',
      'block',
      '12px',
    ]);
  });
});
