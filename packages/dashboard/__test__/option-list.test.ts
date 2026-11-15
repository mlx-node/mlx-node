/** @vitest-environment happy-dom */

import { act, createElement, type ReactElement, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vite-plus/test';

import { type OptionItem, OptionList } from '../ui/src/components/arc-x/option-list/option-list.js';
import { allByRole, byRole } from './render.js';

let root: Root | null = null;
let container: HTMLElement | null = null;

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

const OPTIONS: OptionItem[] = [
  { value: 'exa', title: 'Exa', note: 'Search · Page text' },
  { value: 'brave', title: 'Brave', note: 'Search · Page excerpts' },
  { value: 'off', title: 'Off' },
];

function Harness(props: {
  onChange?: (value: string) => void;
  options?: OptionItem[];
  disabled?: boolean;
}): ReactElement {
  const [value, setValue] = useState('brave');
  return createElement(OptionList, {
    label: 'Search provider',
    value,
    options: props.options ?? OPTIONS,
    disabled: props.disabled,
    onValueChange: (next: string) => {
      props.onChange?.(next);
      setValue(next);
    },
  });
}

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root?.render(element));
  return container;
}

function press(target: Element, key: string): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
  });
}

it('is a named radio group with one tab stop on the checked row', () => {
  const scope = mount(createElement(Harness));
  expect(byRole(scope, 'radiogroup', 'Search provider')).not.toBeNull();
  const rows = allByRole(scope, 'radio');
  expect(rows.map((row) => row.getAttribute('aria-checked'))).toEqual(['false', 'true', 'false']);
  expect(rows.map((row) => row.tabIndex)).toEqual([-1, 0, -1]);
});

it('arrow keys, Home and End move focus and never select; Space and Enter select', () => {
  const onChange = vi.fn();
  const scope = mount(createElement(Harness, { onChange }));
  const rows = allByRole(scope, 'radio');
  act(() => rows[1]!.focus());
  press(rows[1]!, 'ArrowDown');
  expect(document.activeElement).toBe(rows[2]);
  press(rows[2]!, 'ArrowDown');
  expect(document.activeElement).toBe(rows[0]);
  press(rows[0]!, 'End');
  expect(document.activeElement).toBe(rows[2]);
  press(rows[2]!, 'Home');
  expect(document.activeElement).toBe(rows[0]);
  press(rows[0]!, 'ArrowUp');
  expect(document.activeElement).toBe(rows[2]);
  expect(onChange).not.toHaveBeenCalled();
  press(rows[2]!, ' ');
  expect(onChange).toHaveBeenLastCalledWith('off');
  press(rows[0]!, 'Enter');
  expect(onChange).toHaveBeenLastCalledWith('exa');
  expect(allByRole(scope, 'radio')[0]!.getAttribute('aria-checked')).toBe('true');
});

it('a disabled group selects nothing and says so', () => {
  const onChange = vi.fn();
  const scope = mount(createElement(Harness, { onChange, disabled: true }));
  const rows = allByRole(scope, 'radio');
  expect(byRole(scope, 'radiogroup')?.getAttribute('aria-disabled')).toBe('true');
  act(() => rows[0]!.click());
  press(rows[0]!, ' ');
  expect(onChange).not.toHaveBeenCalled();
  expect(rows.every((row) => row.getAttribute('aria-disabled') === 'true')).toBe(true);
});

it('a working row shows its working label in place of the note and is busy', () => {
  const options = OPTIONS.map((o) =>
    o.value === 'exa' ? { ...o, working: true, workingLabel: 'Installing skills…' } : o,
  );
  const scope = mount(createElement(Harness, { options }));
  const row = allByRole(scope, 'radio')[0]!;
  expect(row.getAttribute('aria-busy')).toBe('true');
  expect(row.textContent).toContain('Installing skills…');
  expect(row.textContent).not.toContain('Search · Page text');
});
