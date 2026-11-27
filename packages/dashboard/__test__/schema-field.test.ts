/**
 * @vitest-environment happy-dom
 */

/**
 * One field of an MCP server's form (`ui/src/components/arc-x/schema-field`): its label and name, the description
 * and the host's error tied to the control, the control for each kind, and Enter in a one-line field.
 */

import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { VALUE_MAX_CHARS as HOST_VALUE_MAX_CHARS } from '../src/chat/agent/elicit.js';
import type { McpField } from '../src/chat/types.js';
import {
  initialValue,
  LONG_TEXT_CHARS,
  rfc3339,
  SchemaField,
  type SchemaValue,
  SELECT_MENU_OVER,
  VALUE_MAX_CHARS,
} from '../ui/src/components/arc-x/schema-field/schema-field.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { accessibleName } from './render.js';

let mounted: { root: Root; container: HTMLElement } | undefined;

afterEach(() => {
  vi.restoreAllMocks();
  if (mounted === undefined) return;
  act(() => mounted!.root.unmount());
  mounted.container.remove();
  mounted = undefined;
  document.body.innerHTML = '';
});

const BASE = { name: 'f', title: 'Title', description: null, required: false };

function text(over: Partial<Extract<McpField, { type: 'text' }>> = {}): McpField {
  return { ...BASE, type: 'text', format: null, minLength: null, maxLength: null, initial: '', ...over };
}

function options(n: number): Array<{ label: string }> {
  return Array.from({ length: n }, (_, i) => ({ label: `option ${i}` }));
}

interface Field {
  root: HTMLElement;
  changes: SchemaValue[];
  enters: number;
  rerender: (value: SchemaValue, error?: string | null) => void;
}

function mount(field: McpField, value: SchemaValue, error: string | null = null): Field {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  const out: Field = { root: container, changes: [], enters: 0, rerender: () => {} };
  const render = (v: SchemaValue, e: string | null): void =>
    act(() =>
      root.render(
        createElement(SchemaField, {
          field,
          index: 3,
          value: v,
          error: e,
          onChange: (next: SchemaValue) => out.changes.push(next),
          onEnter: () => {
            out.enters += 1;
          },
        }),
      ),
    );
  render(value, error);
  out.rerender = (v, e = null) => render(v, e);
  mounted = { root, container };
  return out;
}

function described(control: Element): string[] {
  return (control.getAttribute('aria-describedby') ?? '')
    .split(' ')
    .filter((id) => id !== '')
    .map((id) => document.getElementById(id)?.textContent ?? `missing ${id}`);
}

function type(control: HTMLInputElement | HTMLTextAreaElement, value: string): void {
  const proto = control instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  act(() => {
    Reflect.set(proto, 'value', value, control);
    control.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function key(target: Element, name: string, init: KeyboardEventInit = {}): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true, ...init }));
  });
}

describe('SchemaField: label, name, description, error', () => {
  it('a needed field: "*" shown, "needed" in its name; description and error described; aria-invalid', () => {
    const { root } = mount(
      text({ required: true, description: 'One line about the bug.' }),
      '',
      'At least 3 characters.',
    );
    const input = root.querySelector('input')!;
    expect(accessibleName(input)).toBe(fill(CHAT.mcp.fieldName, { title: 'Title', needed: CHAT.mcp.needed }));
    expect(input.getAttribute('aria-required')).toBe('true');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(described(input)).toEqual(['One line about the bug.', 'At least 3 characters.']);
    const label = root.querySelector('label')!;
    expect(label.getAttribute('for')).toBe(input.id);
    expect(label.textContent).toBe('Title *');
    expect(label.querySelector('[aria-hidden="true"]')?.textContent).toBe(' *');
    expect(root.querySelector('[data-field="3"]')?.hasAttribute('data-invalid')).toBe(true);
  });

  it('an optional field with no description and no error: its title, nothing described, not invalid', () => {
    const { root, rerender } = mount(text(), '');
    const input = root.querySelector('input')!;
    expect(accessibleName(input)).toBe('Title');
    expect(input.hasAttribute('aria-required')).toBe(false);
    expect(input.hasAttribute('aria-describedby')).toBe(false);
    expect(input.hasAttribute('aria-invalid')).toBe(false);
    expect(root.querySelector('label')?.textContent).toBe('Title');
    rerender('', 'Not an email address.');
    expect(described(input)).toEqual(['Not an email address.']);
    expect(input.getAttribute('aria-invalid')).toBe('true');
  });

  it('server texts are drawn as plain text', () => {
    const { root } = mount(text({ title: '<b>T</b>', description: '<i>d</i>' }), '', '<u>e</u>');
    expect(root.querySelector('b, i, u')).toBeNull();
    expect(root.textContent).toContain('<b>T</b>');
    expect(root.textContent).toContain('<i>d</i>');
  });
});

describe('SchemaField: the control for each kind', () => {
  it.each<[Partial<Extract<McpField, { type: 'text' }>>, string, string]>([
    [{ format: 'email' }, 'email', 'a@b.co'],
    [{ format: 'uri' }, 'url', 'https://example.com'],
    [{ format: 'date' }, 'date', '2026-08-19'],
    [{ format: 'date-time' }, 'datetime-local', '2026-08-19T10:00'],
    [{ format: null }, 'text', 'x'],
    [{ maxLength: LONG_TEXT_CHARS }, 'text', 'x'],
  ])('text %j is an input of type %s', (over, kind, value) => {
    const { root, changes } = mount(text(over), '');
    const input = root.querySelector('input')!;
    expect(input.type).toBe(kind);
    expect(root.querySelector('textarea')).toBeNull();
    type(input, value);
    expect(changes).toEqual([value]);
  });

  it('a text over the long-text size is a multi-line box; Enter there is a new line, not a send', () => {
    const field = mount(text({ maxLength: LONG_TEXT_CHARS + 1 }), 'a');
    const box = field.root.querySelector('textarea')!;
    expect(field.root.querySelector('input')).toBeNull();
    key(box, 'Enter');
    expect(field.enters).toBe(0);
    type(box, 'a\nb');
    expect(field.changes).toEqual(['a\nb']);
  });

  it('numbers are text boxes with the right keypad; the value stays text', () => {
    const integer = mount({ ...BASE, type: 'integer', min: 1, max: 5, initial: '' }, '');
    const input = integer.root.querySelector('input')!;
    expect(input.type).toBe('text');
    expect(input.inputMode).toBe('numeric');
    type(input, ' 3 ');
    expect(integer.changes).toEqual([' 3 ']);
    act(() => mounted!.root.unmount());
    mounted = undefined;
    const number = mount({ ...BASE, type: 'number', min: null, max: null, initial: '' }, '');
    expect(number.root.querySelector('input')!.inputMode).toBe('decimal');
  });

  it(`every box stops at ${VALUE_MAX_CHARS} characters, the most the host takes`, () => {
    expect(VALUE_MAX_CHARS).toBe(HOST_VALUE_MAX_CHARS);
    const boxes: McpField[] = [
      text(),
      text({ maxLength: LONG_TEXT_CHARS + 1 }),
      { ...BASE, type: 'integer', min: null, max: null, initial: '' },
      { ...BASE, type: 'number', min: null, max: null, initial: '' },
    ];
    for (const field of boxes) {
      const { root } = mount(field, '');
      expect(root.querySelector<HTMLInputElement | HTMLTextAreaElement>('input, textarea')!.maxLength).toBe(
        VALUE_MAX_CHARS,
      );
      act(() => mounted!.root.unmount());
      mounted = undefined;
    }
  });

  it('a yes/no field is a switch named by the field', () => {
    const { root, changes } = mount({ ...BASE, title: 'Assign to me', type: 'boolean', initial: true }, true);
    const toggle = root.querySelector<HTMLButtonElement>('button[role="switch"]')!;
    expect(accessibleName(toggle)).toBe('Assign to me');
    expect(toggle.getAttribute('aria-checked')).toBe('true');
    act(() => toggle.click());
    expect(changes).toEqual([false]);
  });

  it(`one pick of up to ${SELECT_MENU_OVER} options: radio rows in a named group; the value is the index`, () => {
    const { root, changes } = mount(
      {
        ...BASE,
        title: 'Repository',
        required: true,
        type: 'choice',
        options: options(SELECT_MENU_OVER),
        initial: null,
      },
      null,
    );
    const group = root.querySelector('[role="radiogroup"]')!;
    expect(accessibleName(group)).toBe(fill(CHAT.mcp.fieldName, { title: 'Repository', needed: CHAT.mcp.needed }));
    expect(group.getAttribute('aria-required')).toBe('true');
    const radios = [...root.querySelectorAll<HTMLInputElement>('input[type="radio"]')];
    expect(radios.map((r) => r.checked)).toEqual([false, false, false, false]);
    expect(new Set(radios.map((r) => r.name)).size).toBe(1);
    expect(root.querySelector('label[for]')).toBeNull();
    act(() => radios[2]!.click());
    expect(changes).toEqual([2]);
  });

  it(`one pick of more than ${SELECT_MENU_OVER} options: a menu that starts on "${CHAT.mcp.pick}"`, () => {
    const { root, changes, rerender } = mount(
      { ...BASE, type: 'choice', options: options(SELECT_MENU_OVER + 2), initial: null },
      null,
    );
    const select = root.querySelector('select')!;
    expect([...select.options].map((o) => o.textContent)).toEqual([CHAT.mcp.pick, ...options(6).map((o) => o.label)]);
    expect(select.value).toBe('');
    act(() => {
      Reflect.set(HTMLSelectElement.prototype, 'value', '5', select);
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(changes).toEqual([5]);
    rerender(5);
    act(() => {
      Reflect.set(HTMLSelectElement.prototype, 'value', '', select);
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(changes).toEqual([5, null]);
  });

  it('many picks: check boxes in a named group; the value is the sorted indexes', () => {
    const field: McpField = {
      ...BASE,
      title: 'Labels',
      type: 'choices',
      options: options(3),
      minItems: null,
      maxItems: null,
      initial: [],
    };
    const { root, changes, rerender } = mount(field, [2]);
    const group = root.querySelector('[role="group"]')!;
    expect(accessibleName(group)).toBe('Labels');
    const boxes = [...root.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')];
    expect(boxes.map((b) => b.checked)).toEqual([false, false, true]);
    act(() => boxes[0]!.click());
    expect(changes).toEqual([[0, 2]]);
    rerender([0, 2]);
    act(() => boxes[2]!.click());
    expect(changes).toEqual([[0, 2], [0]]);
  });

  it('Enter in a one-line field calls onEnter; Shift+Enter does not', () => {
    const field = mount(text(), '');
    const input = field.root.querySelector('input')!;
    key(input, 'Enter', { shiftKey: true });
    key(input, 'a');
    expect(field.enters).toBe(0);
    key(input, 'Enter');
    expect(field.enters).toBe(1);
  });
});

describe('initialValue and rfc3339', () => {
  it('a field starts on the server default; a date-time default the box cannot hold starts empty', () => {
    expect(initialValue(text({ initial: 'hi' }))).toBe('hi');
    expect(initialValue(text({ format: 'date-time', initial: '2026-08-19T10:00' }))).toBe('2026-08-19T10:00');
    expect(initialValue(text({ format: 'date-time', initial: '2026-08-19T10:00:30' }))).toBe('2026-08-19T10:00:30');
    expect(initialValue(text({ format: 'date-time', initial: '2026-08-19T10:00:00Z' }))).toBe('');
    const initial = [1];
    const picks = initialValue({
      ...BASE,
      type: 'choices',
      options: options(2),
      minItems: null,
      maxItems: null,
      initial,
    });
    expect(picks).toEqual([1]);
    expect(picks).not.toBe(initial);
    expect(initialValue({ ...BASE, type: 'choice', options: options(2), initial: null })).toBeNull();
    expect(initialValue({ ...BASE, type: 'boolean', initial: false })).toBe(false);
  });

  it('adds seconds and the local offset; other text goes as it is', () => {
    vi.spyOn(Date.prototype, 'getTimezoneOffset').mockReturnValue(-330);
    expect(rfc3339('2026-08-19T10:00')).toBe('2026-08-19T10:00:00+05:30');
    vi.spyOn(Date.prototype, 'getTimezoneOffset').mockReturnValue(150);
    expect(rfc3339('2026-08-19T10:00:30')).toBe('2026-08-19T10:00:30-02:30');
    vi.spyOn(Date.prototype, 'getTimezoneOffset').mockReturnValue(0);
    expect(rfc3339('2026-08-19T10:00')).toBe('2026-08-19T10:00:00+00:00');
    expect(rfc3339('')).toBe('');
    expect(rfc3339('tomorrow')).toBe('tomorrow');
  });

  it('the offset is the local zone of that day: the text reads back as the same instant', () => {
    expect(new Date(rfc3339('2026-01-19T10:00')).getTime()).toBe(new Date(2026, 0, 19, 10, 0, 0).getTime());
    expect(new Date(rfc3339('2026-08-19T23:30:15')).getTime()).toBe(new Date(2026, 7, 19, 23, 30, 15).getTime());
  });
});
