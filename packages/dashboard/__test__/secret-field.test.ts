/** @vitest-environment happy-dom */

import { act, createElement, type ReactElement, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vite-plus/test';

import { SecretField, type SecretFieldProps } from '../ui/src/components/arc-x/secret-field/secret-field.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { accessibleName, byRole } from './render.js';

let root: Root | null = null;
let container: HTMLElement | null = null;

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

function Harness(props: Partial<SecretFieldProps>): ReactElement {
  const [value, setValue] = useState('');
  return createElement(SecretField, { label: 'API key', value, onValueChange: setValue, ...props });
}

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root?.render(element));
  return container;
}

function input(scope: HTMLElement): HTMLInputElement {
  return scope.querySelector('input')!;
}

it('is a labelled password field that never offers autofill, spelling or capitals', () => {
  const scope = mount(createElement(Harness, { placeholder: 'Paste your Exa API key' }));
  const field = input(scope);
  expect(field.type).toBe('password');
  expect(scope.querySelector(`label[for="${field.id}"]`)?.textContent).toBe('API key');
  expect(field.getAttribute('autocomplete')).toBe('off');
  expect(field.getAttribute('spellcheck')).toBe('false');
  expect(field.getAttribute('autocapitalize')).toBe('off');
  expect(field.getAttribute('autocorrect')).toBe('off');
  expect(field.placeholder).toBe('Paste your Exa API key');
});

it('the Show button reveals the text in the code font and is a pressed toggle', () => {
  const scope = mount(createElement(Harness));
  const show = byRole(scope, 'button', 'Show key')!;
  expect(show.getAttribute('aria-pressed')).toBe('false');
  act(() => show.click());
  expect(input(scope).type).toBe('text');
  expect(input(scope).hasAttribute('data-shown')).toBe(true);
  const hide = byRole(scope, 'button', 'Hide key')!;
  expect(hide.getAttribute('aria-pressed')).toBe('true');
  act(() => hide.click());
  expect(input(scope).type).toBe('password');
  expect(
    cssValue(cssRules('components/arc-x/secret-field/secret-field.module.css'), '.input[data-shown]', 'font-family'),
  ).toBe('var(--font-code)');
});

it('Enter submits, but not while an input method composes', () => {
  const onSubmit = vi.fn();
  const scope = mount(createElement(Harness, { onSubmit }));
  act(() => {
    input(scope).dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, isComposing: true }));
  });
  expect(onSubmit).not.toHaveBeenCalled();
  act(() => {
    input(scope).dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  });
  expect(onSubmit).toHaveBeenCalledTimes(1);
});

it('an error is an alert, marks the field invalid and describes it', () => {
  const scope = mount(
    createElement(Harness, { description: 'Saving checks the key.', error: 'That does not look like an API key.' }),
  );
  const alert = scope.querySelector('[role="alert"]')!;
  expect(alert.textContent).toBe('That does not look like an API key.');
  expect(input(scope).getAttribute('aria-invalid')).toBe('true');
  expect(input(scope).getAttribute('aria-describedby')).toContain(alert.id);
  expect(accessibleName(input(scope))).toBe('API key');
});

it('never reads the clipboard', () => {
  const read = vi.fn();
  Object.defineProperty(navigator, 'clipboard', { value: { readText: read, read }, configurable: true });
  const scope = mount(createElement(Harness));
  act(() => {
    input(scope).dispatchEvent(new Event('paste', { bubbles: true }));
  });
  expect(read).not.toHaveBeenCalled();
});
