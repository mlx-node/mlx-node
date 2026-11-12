/**
 * @vitest-environment happy-dom
 */

import { act, createElement, useState, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { ChoiceCard } from '../ui/src/components/arc-x/choice-card/choice-card.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import './render.js';

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

afterEach(() => {
  if (mounted === undefined) return;
  act(() => mounted!.root.unmount());
  mounted.container.remove();
  mounted = undefined;
});

const ITEMS = [
  { value: 'org/a', title: 'Model A', blocked: false },
  { value: 'org/b', title: 'Model B', blocked: false },
  { value: 'org/c', title: 'Model C', blocked: true },
];

function Picker({ onPick }: { onPick: (value: string) => void }): ReactElement {
  const [value, setValue] = useState('org/a');
  return createElement(
    'fieldset',
    null,
    ITEMS.map((item) =>
      createElement(ChoiceCard, {
        key: item.value,
        name: 'first-model',
        value: item.value,
        checked: value === item.value,
        disabled: item.blocked,
        onChange: (next: string) => {
          setValue(next);
          onPick(next);
        },
        ariaLabel: item.title,
        describedBy: `description-${item.value}`,
        descriptionId: `description-${item.value}`,
        title: item.title,
        description: `${item.title} description`,
        meta: '~5 GB',
      }),
    ),
  );
}

describe('ChoiceCard', () => {
  it('is a radio in a named group: one checked, the label wraps the card, the description describes it', () => {
    const picks: string[] = [];
    const container = mount(createElement(Picker, { onPick: (value) => picks.push(value) }));
    const radios = [...container.querySelectorAll<HTMLInputElement>('input[type="radio"]')];
    expect(radios.map((radio) => radio.getAttribute('aria-label'))).toEqual(['Model A', 'Model B', 'Model C']);
    expect(radios.map((radio) => radio.checked)).toEqual([true, false, false]);
    expect(new Set(radios.map((radio) => radio.name))).toEqual(new Set(['first-model']));
    const describedBy = radios[1]!.getAttribute('aria-describedby')!;
    expect(document.getElementById(describedBy)?.textContent).toBe('Model B description');
    expect(radios[0]!.closest('label')?.hasAttribute('data-selected')).toBe(true);

    act(() => radios[1]!.click());
    expect(picks).toEqual(['org/b']);
    expect(radios.map((radio) => radio.checked)).toEqual([false, true, false]);
    expect(radios[1]!.closest('label')?.hasAttribute('data-selected')).toBe(true);
    expect(radios[0]!.closest('label')?.hasAttribute('data-selected')).toBe(false);
  });

  it('a disabled card cannot be chosen and says it is blocked', () => {
    const picks: string[] = [];
    const container = mount(createElement(Picker, { onPick: (value) => picks.push(value) }));
    const blocked = container.querySelectorAll<HTMLInputElement>('input[type="radio"]')[2]!;
    expect(blocked.disabled).toBe(true);
    expect(blocked.closest('label')?.hasAttribute('data-blocked')).toBe(true);
    act(() => blocked.click());
    expect(picks).toEqual([]);
    expect(blocked.checked).toBe(false);
  });

  it('keeps the badge row when there is no badge, so titles in one row line up', () => {
    const container = mount(createElement(Picker, { onPick: () => undefined }));
    const cards = [...container.querySelectorAll('label')];
    // input, top row, badge row, title
    expect(cards.map((card) => card.children[3]?.textContent)).toEqual(['Model A', 'Model B', 'Model C']);
    expect(cards.every((card) => card.children[2]?.tagName === 'DIV' && card.children[2].textContent === '')).toBe(
      true,
    );
    expect(cssValue(cssRules('components/arc-x/choice-card/choice-card.module.css'), '.badge', 'min-height')).toBe(
      '20px',
    );
  });

  it('a loading card holds placeholders and no radio', () => {
    const container = mount(
      createElement(ChoiceCard, {
        name: 'first-model',
        value: '',
        checked: false,
        onChange: () => undefined,
        ariaLabel: '',
        title: '',
        description: '',
        loading: true,
      }),
    );
    expect(container.querySelector('input')).toBeNull();
    expect(container.querySelectorAll('[data-placeholder]').length).toBeGreaterThan(0);
    expect(container.firstElementChild?.getAttribute('aria-hidden')).toBe('true');
  });
});
