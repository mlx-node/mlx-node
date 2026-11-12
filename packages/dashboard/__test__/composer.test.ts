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
    hint: 'Enter to send',
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
      createElement(Composer, props({ value: '', phase: 'running', hint: 'Esc to stop', hintOnFocus: true })),
    );
    const area = byRole(container, 'textbox', 'Message') as HTMLTextAreaElement;
    expect(container.textContent).not.toContain('Esc to stop');
    act(() => area.focus());
    expect(container.textContent).toContain('Esc to stop');
    act(() => area.blur());
    expect(container.textContent).not.toContain('Esc to stop');
  });

  it('the side text: the reason when there is text, the counter near the limit, else the hint', () => {
    const container = mount(createElement(Composer, props({ value: '', blocked: 'Engine stopped' })));
    expect(container.textContent).toContain('Enter to send');
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
  it('a long reason wraps beside the chips and never hides one', () => {
    const rules = cssRules('components/arc-x/composer/composer.module.css');
    expect(cssValue(rules, '.toolsStart', 'flex')).toBe('0 1 auto');
    expect(cssValue(rules, '.toolsEnd', 'flex')).toBe('1 1 0');
    expect(cssValue(rules, ".side[data-kind='reason']", 'white-space')).toBe('normal');
  });
});
