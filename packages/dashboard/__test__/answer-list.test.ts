/**
 * @vitest-environment happy-dom
 */

/**
 * `AnswerList`: a question's options as a radio group or a group of checkboxes, numbered, with an "Other" row whose
 * own field follows the group. Roles, the roving tab stop, keys, and the Other row.
 */

import { act, createElement, type ReactElement, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import {
  AnswerList,
  NO_ANSWER,
  pickRow,
  type AnswerState,
} from '../ui/src/components/arc-x/answer-list/answer-list.js';
import './render.js';

let mounted: { root: Root; container: HTMLElement } | undefined;

afterEach(() => {
  if (mounted === undefined) return;
  act(() => mounted!.root.unmount());
  mounted.container.remove();
  mounted = undefined;
});

const OPTIONS = [
  { label: 'Postgres', description: 'Relational.', tag: 'Recommended' },
  { label: 'Redis', description: 'In memory.' },
];

interface Seen {
  states: AnswerState[];
  enters: number;
  ready: boolean;
}

function List(props: { multi: boolean; seen: Seen; count?: number }): ReactElement {
  const [value, setValue] = useState<AnswerState>(NO_ANSWER);
  const options =
    props.count === undefined ? OPTIONS : Array.from({ length: props.count }, (_, i) => ({ label: `Option ${i + 1}` }));
  return createElement(AnswerList, {
    label: 'Which database?',
    multi: props.multi,
    options,
    value,
    onChange: (next: AnswerState) => {
      props.seen.states.push(next);
      setValue(next);
    },
    onEnter: () => {
      props.seen.enters += 1;
    },
    enterReady: () => props.seen.ready,
    other: { label: 'Other', fieldLabel: 'Your own answer to: Which database?', placeholder: 'Type your own answer…' },
  });
}

function mount(multi: boolean, count?: number): { root: HTMLElement; seen: Seen } {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  const seen: Seen = { states: [], enters: 0, ready: true };
  act(() => root.render(createElement(List, { multi, seen, count })));
  mounted = { root, container };
  return { root: container, seen };
}

function rows(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>('[data-row]')];
}

function field(root: HTMLElement): HTMLTextAreaElement {
  return root.querySelector<HTMLTextAreaElement>('textarea')!;
}

function key(target: Element, name: string, init: KeyboardEventInit = {}): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true, ...init }));
  });
}

function type(area: HTMLTextAreaElement, text: string): void {
  act(() => {
    Reflect.set(HTMLTextAreaElement.prototype, 'value', text, area);
    area.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function checked(root: HTMLElement): boolean[] {
  return rows(root).map((row) => row.getAttribute('aria-checked') === 'true');
}

function stops(root: HTMLElement): number[] {
  return rows(root)
    .map((row, index) => (row.tabIndex === 0 ? index : -1))
    .filter((index) => index >= 0);
}

describe('pickRow', () => {
  it('single: the row alone, Other clears the picks; multi: adds, or flips with flip', () => {
    expect(pickRow({ picks: [1], otherOn: false, otherText: 'x' }, 0, 2, false)).toEqual({
      picks: [0],
      otherOn: false,
      otherText: 'x',
    });
    expect(pickRow({ picks: [1], otherOn: false, otherText: 'x' }, 2, 2, false)).toEqual({
      picks: [],
      otherOn: true,
      otherText: 'x',
    });
    expect(pickRow({ picks: [2], otherOn: false, otherText: '' }, 0, 3, true).picks).toEqual([0, 2]);
    expect(pickRow({ picks: [0, 2], otherOn: false, otherText: '' }, 0, 3, true).picks).toEqual([0, 2]);
    expect(pickRow({ picks: [0, 2], otherOn: false, otherText: '' }, 0, 3, true, true).picks).toEqual([2]);
    expect(pickRow({ picks: [], otherOn: true, otherText: '' }, 3, 3, true, true).otherOn).toBe(false);
  });
});

describe('AnswerList', () => {
  it('single is a radio group named by the question; multi a group of checkboxes; rows are numbered', () => {
    const single = mount(false);
    const group = single.root.querySelector('[role="radiogroup"]')!;
    expect(group.getAttribute('aria-label')).toBe('Which database?');
    expect(rows(single.root).map((row) => row.getAttribute('role'))).toEqual(['radio', 'radio', 'radio']);
    expect(rows(single.root).map((row) => row.textContent)).toEqual([
      '1PostgresRecommendedRelational.',
      '2RedisIn memory.',
      '3Other',
    ]);
    expect(checked(single.root)).toEqual([false, false, false]);
    expect(field(single.root).getAttribute('aria-label')).toBe('Your own answer to: Which database?');
    act(() => mounted!.root.unmount());
    mounted!.container.remove();
    mounted = undefined;

    const multi = mount(true);
    expect(multi.root.querySelector('[role="group"]')?.getAttribute('aria-label')).toBe('Which database?');
    expect(rows(multi.root).map((row) => row.getAttribute('role'))).toEqual(['checkbox', 'checkbox', 'checkbox']);
  });

  it('only the first nine rows show a number key', () => {
    const { root } = mount(false, 10);
    expect(rows(root).map((row) => row.querySelector('[aria-hidden="true"]:nth-child(2)')?.textContent)).toEqual([
      '1',
      '2',
      '3',
      '4',
      '5',
      '6',
      '7',
      '8',
      '9',
      '',
      '',
    ]);
  });

  it('single: one tab stop, the first row when nothing is picked, the picked row after; multi: every row', () => {
    const { root } = mount(false);
    expect(stops(root)).toEqual([0]);
    act(() => rows(root)[1]!.click());
    expect(checked(root)).toEqual([false, true, false]);
    act(() => rows(root)[1]!.blur());
    expect(stops(root)).toEqual([1]);
    act(() => mounted!.root.unmount());
    mounted!.container.remove();
    mounted = undefined;

    const multi = mount(true);
    expect(stops(multi.root)).toEqual([0, 1, 2]);
    act(() => rows(multi.root)[1]!.click());
    expect(stops(multi.root)).toEqual([0, 1, 2]);
  });

  it('single: arrows move and pick, wrap, and onto Other move focus into its field', () => {
    const { root } = mount(false);
    act(() => rows(root)[0]!.focus());
    key(rows(root)[0]!, 'ArrowDown');
    expect(checked(root)).toEqual([false, true, false]);
    expect(document.activeElement).toBe(rows(root)[1]);
    key(rows(root)[1]!, 'ArrowDown');
    expect(checked(root)).toEqual([false, false, true]);
    expect(document.activeElement).toBe(field(root));
    key(rows(root)[2]!, 'ArrowDown');
    expect(checked(root)).toEqual([true, false, false]);
    key(rows(root)[0]!, 'ArrowUp');
    expect(checked(root)).toEqual([false, false, true]);
    key(rows(root)[2]!, 'Home');
    expect(checked(root)).toEqual([true, false, false]);
  });

  it('multi: arrows only move; Space toggles; picking Other moves focus into its field', () => {
    const { root, seen } = mount(true);
    act(() => rows(root)[0]!.focus());
    key(rows(root)[0]!, 'ArrowDown');
    expect(checked(root)).toEqual([false, false, false]);
    expect(document.activeElement).toBe(rows(root)[1]);
    key(rows(root)[1]!, ' ');
    key(rows(root)[0]!, ' ');
    expect(checked(root)).toEqual([true, true, false]);
    key(rows(root)[0]!, ' ');
    expect(checked(root)).toEqual([false, true, false]);
    key(rows(root)[0]!, ' ', { repeat: true });
    expect(checked(root)).toEqual([false, true, false]);
    key(rows(root)[2]!, ' ');
    expect(checked(root)).toEqual([false, true, true]);
    expect(document.activeElement).toBe(field(root));
    expect(seen.enters).toBe(0);
  });

  it('Enter on a row that is not picked only picks it; on a picked row it goes on; not before it is ready', () => {
    const { root, seen } = mount(false);
    seen.ready = false;
    key(rows(root)[0]!, 'Enter');
    expect(checked(root)).toEqual([false, false, false]);
    seen.ready = true;
    key(rows(root)[0]!, 'Enter');
    expect(checked(root)).toEqual([true, false, false]);
    expect(seen.enters).toBe(0);
    key(rows(root)[0]!, 'Enter', { repeat: true });
    expect(seen.enters).toBe(0);
    key(rows(root)[0]!, 'Enter');
    expect(seen.enters).toBe(1);
  });

  it('Other: typing picks it, a blank field does not; Escape goes back to the row; Enter goes on, Shift+Enter does not', () => {
    const { root, seen } = mount(false);
    act(() => rows(root)[1]!.click());
    type(field(root), '   ');
    expect(checked(root)).toEqual([false, true, false]);
    type(field(root), 'DynamoDB');
    expect(checked(root)).toEqual([false, false, true]);
    expect(seen.states.at(-1)).toEqual({ picks: [], otherOn: true, otherText: 'DynamoDB' });
    act(() => field(root).focus());
    key(field(root), 'Enter', { shiftKey: true });
    expect(seen.enters).toBe(0);
    key(field(root), 'Enter');
    expect(seen.enters).toBe(1);
    key(field(root), 'Escape');
    expect(document.activeElement).toBe(rows(root)[2]);
    // A pick after the field keeps the text but turns Other off.
    act(() => rows(root)[0]!.click());
    expect(seen.states.at(-1)).toEqual({ picks: [0], otherOn: false, otherText: 'DynamoDB' });
  });
});
