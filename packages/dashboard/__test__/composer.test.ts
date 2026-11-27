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

  it('an op (fork, rewind, recap): with text the button queues, with none it is a still ring; never Stop', () => {
    const queueing = { enabled: true, onQueue: () => undefined };
    expect(sendFace('op', false, undefined, queueing)).toMatchObject({
      glyph: 'send',
      action: 'queue',
      enabled: true,
      name: CHAT.queue.add,
      tip: CHAT.queue.addTipOp,
    });
    expect(sendFace('op', false)).toMatchObject({
      glyph: 'ring',
      action: 'none',
      enabled: false,
      name: CHAT.opBusy,
      tip: CHAT.opBusyTip,
    });
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

  // Mutation: leave `op` out of the queueing phases → Enter does nothing while a fork runs.
  it('while an op runs Enter queues, the round button is Queue, there is no Stop and Esc stops nothing', () => {
    const onQueue = vi.fn();
    const onStop = vi.fn();
    const onSend = vi.fn();
    const container = mount(createElement(Composer, props({ value: 'next', phase: 'op', onQueue, onStop, onSend })));
    const area = byRole(container, 'textbox', 'Message')!;
    key(area, { key: 'Enter' });
    expect(onQueue).toHaveBeenCalledTimes(1);
    expect(byRole(container, 'button', CHAT.queue.add)).not.toBeNull();
    expect(byRole(container, 'button', CHAT.stop)).toBeNull();
    key(area, { key: 'Escape' });
    expect(onStop).not.toHaveBeenCalled();
    expect(onSend).not.toHaveBeenCalled();
    rerender(createElement(Composer, props({ value: '', phase: 'op', onQueue, onStop, onSend })));
    const ring = byRole(container, 'button', CHAT.opBusy) as HTMLButtonElement;
    expect(ring.disabled).toBe(true);
    expect(byRole(container, 'button', CHAT.stop)).toBeNull();
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

/** A DataTransfer stand-in: happy-dom has none that holds files. */
function transfer(types: string[], files: File[]): DataTransfer {
  const items = files.map((file) => ({ kind: 'file', type: file.type, getAsFile: () => file }));
  return { types, files, items } as unknown as DataTransfer;
}

function dispatchWith(
  target: Element,
  name: string,
  field: 'clipboardData' | 'dataTransfer',
  data: DataTransfer,
): Event {
  const event = new Event(name, { bubbles: true, cancelable: true });
  Object.defineProperty(event, field, { value: data });
  act(() => {
    target.dispatchEvent(event);
  });
  return event;
}

describe('Composer files', () => {
  const png = new File(['png'], 'shot.png', { type: 'image/png' });
  const notes = new File(['notes'], 'notes.md', { type: 'text/markdown' });

  it('the attachments slot renders between the notice and the text field', () => {
    const container = mount(
      createElement(
        Composer,
        props({
          notice: { tone: 'warning', text: 'Engine starting' },
          attachments: createElement('div', { 'data-testid': 'strip' }, 'STRIP'),
        }),
      ),
    );
    const form = container.querySelector('form')!;
    const strip = form.querySelector('[data-testid="strip"]')!;
    const notice = byRole(container, 'status')!;
    const area = byRole(container, 'textbox', 'Message')!;
    const order = [...form.children];
    const at = (node: Element): number => order.findIndex((child) => child.contains(node));
    expect(at(notice)).toBeLessThan(at(strip));
    expect(at(strip)).toBeLessThan(at(area));
  });

  it('a paste with files and no plain text calls onFiles and stops the paste; with plain text it does not', () => {
    const onFiles = vi.fn();
    const container = mount(createElement(Composer, props({ onFiles })));
    const area = byRole(container, 'textbox', 'Message')!;
    const fileOnly = dispatchWith(area, 'paste', 'clipboardData', transfer(['Files'], [png]));
    expect(onFiles).toHaveBeenCalledWith([png]);
    expect(fileOnly.defaultPrevented).toBe(true);
    onFiles.mockClear();
    const withText = dispatchWith(area, 'paste', 'clipboardData', transfer(['text/plain', 'Files'], [png]));
    expect(onFiles).not.toHaveBeenCalled();
    expect(withText.defaultPrevented).toBe(false);
    const textOnly = dispatchWith(area, 'paste', 'clipboardData', transfer(['text/plain'], []));
    expect(textOnly.defaultPrevented).toBe(false);
    expect(onFiles).not.toHaveBeenCalled();
  });

  it('a file drop on the form calls onFiles; dragging marks the form until the pointer leaves it', () => {
    const onFiles = vi.fn();
    const container = mount(createElement(Composer, props({ onFiles })));
    const form = container.querySelector('form')!;
    const area = byRole(container, 'textbox', 'Message')!;
    const over = dispatchWith(area, 'dragover', 'dataTransfer', transfer(['Files'], [notes]));
    expect(over.defaultPrevented).toBe(true);
    expect(form.hasAttribute('data-dragging')).toBe(true);
    const inside = new Event('dragleave', { bubbles: true });
    Object.defineProperty(inside, 'relatedTarget', { value: area });
    act(() => {
      form.dispatchEvent(inside);
    });
    expect(form.hasAttribute('data-dragging')).toBe(true);
    act(() => {
      form.dispatchEvent(new Event('dragleave', { bubbles: true }));
    });
    expect(form.hasAttribute('data-dragging')).toBe(false);
    dispatchWith(form, 'dragover', 'dataTransfer', transfer(['Files'], [notes]));
    const drop = dispatchWith(form, 'drop', 'dataTransfer', transfer(['Files'], [notes, png]));
    expect(drop.defaultPrevented).toBe(true);
    expect(onFiles).toHaveBeenCalledWith([notes, png]);
    expect(form.hasAttribute('data-dragging')).toBe(false);
    expect(
      cssValue(cssRules('components/arc-x/composer/composer.module.css'), '.root[data-dragging]', 'border-color'),
    ).toBe('var(--foreground)');
  });

  it('without onFiles a file drop on the text field is still ignored and the form is not marked', () => {
    const container = mount(createElement(Composer, props()));
    const form = container.querySelector('form')!;
    const area = byRole(container, 'textbox', 'Message')!;
    const over = dispatchWith(area, 'dragover', 'dataTransfer', transfer(['Files'], [notes]));
    const drop = dispatchWith(area, 'drop', 'dataTransfer', transfer(['Files'], [notes]));
    expect([over.defaultPrevented, drop.defaultPrevented]).toEqual([true, true]);
    expect(form.hasAttribute('data-dragging')).toBe(false);
    const paste = dispatchWith(area, 'paste', 'clipboardData', transfer(['Files'], [png]));
    expect(paste.defaultPrevented).toBe(false);
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

  it('the chips give way in order: tier 1, then tier 2, then the model, each down to its floor', () => {
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

  it('the sentence: sizes in base-1024 K and M, a whole percent, <1% for a tiny share, 100% past the window', () => {
    expect(contextText({ used: 11_971, window: 262_144 })).toBe('Context: 11.7K of 256K tokens (5%)');
    expect(contextText({ used: 1, window: 262_144 })).toBe('Context: 1 of 256K tokens (<1%)');
    expect(contextText({ used: 300_000, window: 262_144 })).toBe('Context: 293K of 256K tokens (100%)');
    expect(contextText({ used: 34_958, window: 1_048_576 })).toBe('Context: 34.1K of 1M tokens (3%)');
    expect(contextText({ used: 1_600_000, window: 2_097_152 })).toBe('Context: 1.5M of 2M tokens (76%)');
  });

  it('a ring with no text: the numbers are only in its name, and it takes keyboard focus for its tooltip', () => {
    const container = mount(createElement(Composer, props({ context: { used: 11_971, window: 262_144 } })));
    const hint = byRole(container, 'img', 'Context: 11.7K of 256K tokens (5%)')!;
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
    expect(document.body.textContent).toContain('Context: 11.7K of 256K tokens (5%)');
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

  it('is the Context panel button only with the panel; without it, an image with no press', () => {
    const context = { used: 11_971, window: 262_144 };
    const name = 'Context: 11.7K of 256K tokens (5%)';
    const onOpenChange = vi.fn();
    const container = mount(createElement(Composer, props({ context })));
    expect(byRole(container, 'button', name)).toBeNull();
    expect(byRole(container, 'img', name)).not.toBeNull();
    rerender(
      createElement(Composer, props({ context, contextPanel: { open: false, onOpenChange, content: 'PANEL' } })),
    );
    expect(byRole(container, 'img', name)).toBeNull();
    const button = byRole(container, 'button', name)!;
    expect([button.getAttribute('type'), button.getAttribute('aria-haspopup')]).toEqual(['button', 'dialog']);
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(button.querySelector('svg')!.getAttribute('aria-hidden')).toBe('true');
    act(() => button.click());
    expect(onOpenChange).toHaveBeenCalledWith(true);
    rerender(createElement(Composer, props({ context, contextPanel: { open: true, onOpenChange, content: 'PANEL' } })));
    expect(byRole(container, 'button', name)!.getAttribute('aria-expanded')).toBe('true');
    expect(value('.contextButton', 'cursor')).toBe('pointer');
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

describe('Composer queue', () => {
  const buttons = (container: Element): string[] =>
    Array.from(container.querySelectorAll('form > div:last-child button')).map(
      (b) => b.getAttribute('aria-label') ?? '',
    );

  it('the button table: idle Send; an answer with text: small Stop, then Queue; with no text: Stop; stopping: Stop off', () => {
    const onQueue = vi.fn();
    const container = mount(createElement(Composer, props({ value: 'next', phase: 'idle', onQueue })));
    expect(buttons(container)).toEqual([CHAT.send]);
    for (const phase of ['starting', 'running'] as const) {
      rerender(createElement(Composer, props({ value: 'next', phase, onQueue })));
      expect(buttons(container)).toEqual([CHAT.stop, CHAT.queue.add]);
      const queue = byRole(container, 'button', CHAT.queue.add) as HTMLButtonElement;
      expect(queue.disabled).toBe(false);
      expect(queue.getAttribute('aria-keyshortcuts')).toBe('Enter');
    }
    rerender(createElement(Composer, props({ value: '  ', phase: 'running', onQueue })));
    expect(buttons(container)).toEqual([CHAT.stop]);
    rerender(createElement(Composer, props({ value: 'next', phase: 'cancelling', onQueue })));
    expect(buttons(container)).toEqual([CHAT.stopping]);
    expect((byRole(container, 'button', CHAT.stopping) as HTMLButtonElement).disabled).toBe(true);
    rerender(createElement(Composer, props({ value: 'next', phase: 'running' })));
    expect(buttons(container)).toEqual([CHAT.stop]);
  });

  it('while an answer runs Enter and ⌘Enter queue and never stop; the small Stop and Esc stop', () => {
    const onQueue = vi.fn();
    const onStop = vi.fn();
    const onSend = vi.fn();
    const container = mount(
      createElement(Composer, props({ value: 'later', phase: 'running', onQueue, onStop, onSend })),
    );
    const area = byRole(container, 'textbox', 'Message')!;
    key(area, { key: 'Enter' });
    key(area, { key: 'Enter', metaKey: true });
    key(area, { key: 'Enter', shiftKey: true });
    expect(onQueue).toHaveBeenCalledTimes(2);
    act(() => byRole(container, 'button', CHAT.queue.add)!.click());
    expect(onQueue).toHaveBeenCalledTimes(3);
    expect(onStop).not.toHaveBeenCalled();
    act(() => byRole(container, 'button', CHAT.stop)!.click());
    key(area, { key: 'Escape' });
    expect(onStop).toHaveBeenCalledTimes(2);
    expect(onSend).not.toHaveBeenCalled();
  });

  it('Work: ⌘Enter (or Ctrl+Enter) sends now while Queue is on; Enter still queues; idle ⌘Enter sends', () => {
    const onQueue = vi.fn();
    const onSendNow = vi.fn();
    const onSend = vi.fn();
    const container = mount(
      createElement(Composer, props({ value: 'use pnpm', phase: 'running', onQueue, onSendNow, onSend })),
    );
    const area = (): Element => byRole(container, 'textbox', 'Message')!;
    key(area(), { key: 'Enter', metaKey: true });
    key(area(), { key: 'Enter', ctrlKey: true });
    key(area(), { key: 'Enter' });
    expect(onSendNow).toHaveBeenCalledTimes(2);
    expect(onQueue).toHaveBeenCalledTimes(1);
    key(area(), { key: 'Enter', metaKey: true, shiftKey: true });
    expect(onSendNow).toHaveBeenCalledTimes(2);
    rerender(
      createElement(Composer, props({ value: 'use pnpm', phase: 'running', onQueue, onSendNow, blocked: 'no' })),
    );
    key(area(), { key: 'Enter', metaKey: true });
    expect(onSendNow).toHaveBeenCalledTimes(2);
    rerender(createElement(Composer, props({ value: 'use pnpm', phase: 'idle', onQueue, onSendNow, onSend })));
    key(area(), { key: 'Enter', metaKey: true });
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSendNow).toHaveBeenCalledTimes(2);
  });

  // Mutation: gate ⌘Enter on `canQueue` again → a full list turns Send now off too.
  it('Work, the list full (`queueOff`): Queue is off and says why; ⌘Enter still sends now', () => {
    const onQueue = vi.fn();
    const onSendNow = vi.fn();
    const full = CHAT.queue.full;
    const container = mount(
      createElement(Composer, props({ value: 'use pnpm', phase: 'running', onQueue, onSendNow, queueOff: full })),
    );
    const area = (): Element => byRole(container, 'textbox', 'Message')!;
    key(area(), { key: 'Enter' });
    expect(onQueue).not.toHaveBeenCalled();
    expect(container.textContent).toContain(full);
    expect((byRole(container, 'button', CHAT.queue.add) as HTMLButtonElement).disabled).toBe(true);
    key(area(), { key: 'Enter', metaKey: true });
    expect(onSendNow).toHaveBeenCalledTimes(1);
    expect(onQueue).not.toHaveBeenCalled();
    rerender(createElement(Composer, props({ value: 'use pnpm', phase: 'running', onQueue, queueOff: full })));
    key(area(), { key: 'Enter', metaKey: true });
    expect(onQueue).not.toHaveBeenCalled();
  });

  it('stopping keeps the text: Enter does nothing until the agent is idle, then it sends', () => {
    const onQueue = vi.fn();
    const onSend = vi.fn();
    const container = mount(createElement(Composer, props({ value: 'keep me', phase: 'cancelling', onQueue, onSend })));
    key(byRole(container, 'textbox', 'Message')!, { key: 'Enter' });
    expect(onQueue).not.toHaveBeenCalled();
    expect(onSend).not.toHaveBeenCalled();
    expect((byRole(container, 'textbox', 'Message') as HTMLTextAreaElement).value).toBe('keep me');
    rerender(createElement(Composer, props({ value: 'keep me', phase: 'idle', onQueue, onSend })));
    key(byRole(container, 'textbox', 'Message')!, { key: 'Enter' });
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onQueue).not.toHaveBeenCalled();
  });

  it('a full queue: the reason shows and Queue is off; the queue slot sits above the attachments', () => {
    const onQueue = vi.fn();
    const container = mount(
      createElement(
        Composer,
        props({
          value: 'sixth',
          phase: 'running',
          onQueue,
          blocked: CHAT.queue.full,
          queue: createElement('div', { 'data-testid': 'queue' }),
          attachments: createElement('div', { 'data-testid': 'files' }),
        }),
      ),
    );
    expect(container.textContent).toContain(CHAT.queue.full);
    const queue = byRole(container, 'button', CHAT.queue.add) as HTMLButtonElement;
    expect(queue.disabled).toBe(true);
    key(byRole(container, 'textbox', 'Message')!, { key: 'Enter' });
    expect(onQueue).not.toHaveBeenCalled();
    const form = container.querySelector('form')!;
    const order = Array.from(form.children).map((child) => child.getAttribute('data-testid'));
    expect(order.indexOf('queue')).toBeLessThan(order.indexOf('files'));
    expect(sendFace('running', false, CHAT.queue.full, { enabled: false, onQueue })).toMatchObject({
      name: CHAT.queue.add,
      enabled: false,
      tip: CHAT.queue.full,
    });
    expect(sendFace('running', false, undefined, { enabled: true, onQueue }).tip).toBe(CHAT.queue.addTip);
  });

  it('Esc closes an open row edit first; only the next Esc stops the answer', () => {
    const onCancelEdit = vi.fn();
    const onStop = vi.fn();
    const container = mount(createElement(Composer, props({ value: 'x', phase: 'running', onStop, onCancelEdit })));
    key(byRole(container, 'textbox', 'Message')!, { key: 'Escape' });
    expect(onCancelEdit).toHaveBeenCalledTimes(1);
    expect(onStop).not.toHaveBeenCalled();
    rerender(createElement(Composer, props({ value: 'x', phase: 'running', onStop })));
    key(byRole(container, 'textbox', 'Message')!, { key: 'Escape' });
    expect(onStop).toHaveBeenCalledTimes(1);
    expect(onCancelEdit).toHaveBeenCalledTimes(1);
  });

  it('Alt+↑ in an empty box opens the last queued row; with text it does nothing', () => {
    const onEditLast = vi.fn();
    const container = mount(createElement(Composer, props({ value: '', phase: 'running', onEditLast })));
    key(byRole(container, 'textbox', 'Message')!, { key: 'ArrowUp', altKey: true });
    expect(onEditLast).toHaveBeenCalledTimes(1);
    rerender(createElement(Composer, props({ value: 'x', phase: 'running', onEditLast })));
    key(byRole(container, 'textbox', 'Message')!, { key: 'ArrowUp', altKey: true });
    key(byRole(container, 'textbox', 'Message')!, { key: 'ArrowUp' });
    expect(onEditLast).toHaveBeenCalledTimes(1);
  });
});
