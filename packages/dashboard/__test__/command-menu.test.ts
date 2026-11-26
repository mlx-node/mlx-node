/**
 * @vitest-environment happy-dom
 */

/**
 * The `/` command list in the composer: it opens on a first word that starts with `/` while the text field has focus,
 * filters and groups the rows, and is driven from the text field (it never takes focus). Enter or Tab on a row fills
 * `/name ` and sends nothing; a second Enter sends; a name typed in full sends on the first Enter; Esc closes the list
 * only, also while an answer runs.
 */

import { act, createElement, useState, type ReactElement } from 'react';
import { afterEach, describe, expect, it, vi, type Mock } from 'vite-plus/test';

import { commandOrder, type CommandRow } from '../ui/src/components/arc-x/command-menu/command-menu.js';
import { Composer, type ComposerPhase } from '../ui/src/components/arc-x/composer/composer.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { mount, unmountAll } from './helpers/mount.js';

afterEach(unmountAll);

const ROWS: CommandRow[] = [
  { name: 'compact', group: 'builtin', about: CHAT.slash.about.compact, words: null },
  { name: 'goal', group: 'builtin', about: CHAT.slash.about.goal, words: CHAT.slash.hint.goal },
  { name: 'loop', group: 'builtin', about: CHAT.slash.about.loop, words: CHAT.slash.hint.loop },
  {
    name: 'deep-research',
    group: 'builtin',
    about: CHAT.slash.about.deepResearch,
    words: CHAT.slash.hint.deepResearch,
  },
  { name: 'commit', group: 'skill', about: 'Write a commit message', words: '[scope]' },
  { name: 'release', group: 'workflow', about: 'Workflow: cut a release', words: null },
];

interface Spies {
  onSend: Mock<() => void>;
  onStop: Mock<() => void>;
  onQueue: Mock<() => void>;
}

const spiesOf = (): Spies => ({ onSend: vi.fn(), onStop: vi.fn(), onQueue: vi.fn() });

function Harness(props: { spies: Spies; phase: ComposerPhase; later?: boolean; start?: string }): ReactElement {
  const [value, setValue] = useState(props.start ?? '');
  return createElement(Composer, {
    value,
    onValueChange: setValue,
    placeholder: 'Ask anything',
    phase: props.phase,
    onSend: props.spies.onSend,
    onStop: props.spies.onStop,
    onQueue: props.spies.onQueue,
    tools: createElement('span', null, 'TOOLS'),
    commands: { rows: ROWS, later: props.later ?? false },
  });
}

function setup(phase: ComposerPhase = 'idle', later = false): { root: HTMLElement; area: HTMLTextAreaElement } & Spies {
  const spies = spiesOf();
  const root = mount(createElement(Harness, { spies, phase, later }));
  const area = root.querySelector('textarea')!;
  act(() => area.focus());
  return { root, area, ...spies };
}

function type(area: HTMLTextAreaElement, text: string): void {
  act(() => {
    Reflect.set(HTMLTextAreaElement.prototype, 'value', text, area);
    area.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function key(area: HTMLTextAreaElement, init: KeyboardEventInit): KeyboardEvent {
  const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
  act(() => {
    area.dispatchEvent(event);
  });
  return event;
}

const listbox = (root: HTMLElement): HTMLElement | null => root.querySelector('[role="listbox"]');
const options = (root: HTMLElement): HTMLElement[] => [...root.querySelectorAll<HTMLElement>('[role="option"]')];
const names = (root: HTMLElement): string[] => options(root).map((o) => o.querySelector('span')!.textContent ?? '');
const selected = (root: HTMLElement): string =>
  options(root)
    .find((o) => o.getAttribute('aria-selected') === 'true')!
    .querySelector('span')!.textContent ?? '';

describe('commandOrder', () => {
  it('every row while nothing is typed; the matches by group, best first, as typed', () => {
    expect(commandOrder(ROWS, '').map((r) => r.name)).toEqual([
      'compact',
      'goal',
      'loop',
      'deep-research',
      'commit',
      'release',
    ]);
    expect(commandOrder(ROWS, 'co').map((r) => r.name)).toEqual(['compact', 'commit']);
    expect(commandOrder(ROWS, 'GO').map((r) => r.name)).toEqual(['goal']);
    expect(commandOrder(ROWS, 'research').map((r) => r.name)).toEqual(['deep-research']);
    expect(commandOrder(ROWS, 'xyz')).toEqual([]);
  });
});

describe('the / command list', () => {
  it('opens on a first word with /, groups the rows and wires the text field to the list', () => {
    const { root, area } = setup();
    expect(area.getAttribute('aria-expanded')).toBe('false');
    expect(area.getAttribute('aria-autocomplete')).toBe('list');
    type(area, '/');
    const list = listbox(root)!;
    expect(list.getAttribute('aria-label')).toBe(CHAT.slash.label);
    expect(names(root)).toEqual(['/compact', '/goal', '/loop', '/deep-research', '/commit', '/release']);
    const groups = [...list.querySelectorAll('[role="group"]')].map(
      (g) => document.getElementById(g.getAttribute('aria-labelledby')!)!.textContent,
    );
    expect(groups).toEqual([CHAT.slash.groups.builtin, CHAT.slash.groups.skill, CHAT.slash.groups.workflow]);
    expect(area.getAttribute('aria-expanded')).toBe('true');
    expect(area.getAttribute('aria-controls')).toBe(list.id);
    expect(area.getAttribute('aria-activedescendant')).toBe(options(root)[0]!.id);
    expect(root.querySelector('[role="status"]')!.textContent).toBe(fill(CHAT.slash.count, { n: 6 }));
    expect(document.activeElement).toBe(area);

    type(area, '/go');
    expect(names(root)).toEqual(['/goal']);
    expect(root.querySelector('[role="status"]')!.textContent).toBe(CHAT.slash.countOne);
    // No match, text that is not a command, a command past its first word: no list.
    for (const text of ['/xyz', 'hello /goal', '/goal ship it']) {
      type(area, text);
      expect([text, listbox(root)]).toEqual([text, null]);
      expect(area.hasAttribute('aria-activedescendant')).toBe(false);
    }
  });

  it('↑ and ↓ move the row and wrap; the text field keeps focus', () => {
    const { root, area } = setup();
    type(area, '/');
    expect(selected(root)).toBe('/compact');
    expect(key(area, { key: 'ArrowDown' }).defaultPrevented).toBe(true);
    expect(selected(root)).toBe('/goal');
    expect(area.getAttribute('aria-activedescendant')).toBe(options(root)[1]!.id);
    key(area, { key: 'ArrowUp' });
    key(area, { key: 'ArrowUp' });
    expect(selected(root)).toBe('/release');
    key(area, { key: 'ArrowDown' });
    expect(selected(root)).toBe('/compact');
    expect(document.activeElement).toBe(area);
  });

  it('Enter or Tab on a row fills /name and sends nothing; the second Enter sends', () => {
    const { root, area, onSend } = setup();
    type(area, '/go');
    key(area, { key: 'Enter' });
    expect(area.value).toBe('/goal ');
    expect(listbox(root)).toBeNull();
    expect(onSend).not.toHaveBeenCalled();
    key(area, { key: 'Enter' });
    expect(onSend).toHaveBeenCalledTimes(1);

    type(area, '/');
    key(area, { key: 'ArrowDown' });
    key(area, { key: 'ArrowDown' });
    expect(key(area, { key: 'Tab' }).defaultPrevented).toBe(true);
    expect(area.value).toBe('/loop ');
    expect(onSend).toHaveBeenCalledTimes(1);

    // A row picked with the words already typed keeps them.
    type(area, '/co the diff');
    act(() => {
      area.setSelectionRange(2, 2);
      area.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'ArrowLeft' }));
    });
    key(area, { key: 'Enter' });
    expect(area.value).toBe('/compact the diff');
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it('a name typed in full sends on the first Enter', () => {
    const { area, onSend } = setup();
    type(area, '/compact');
    key(area, { key: 'Enter' });
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(area.value).toBe('/compact');
  });

  it('a click fills the row and keeps focus in the text field', () => {
    const { root, area } = setup();
    type(area, '/');
    const row = options(root)[4]!;
    const down = new MouseEvent('mousedown', { bubbles: true, cancelable: true });
    act(() => {
      row.dispatchEvent(down);
    });
    expect(down.defaultPrevented).toBe(true);
    act(() => row.click());
    expect(area.value).toBe('/commit ');
  });

  it('Esc closes the list only, also while an answer runs; the next Esc stops', () => {
    const { root, area, onStop } = setup('running');
    type(area, '/go');
    expect(listbox(root)).not.toBeNull();
    key(area, { key: 'Escape' });
    expect(listbox(root)).toBeNull();
    expect(onStop).not.toHaveBeenCalled();
    expect(area.value).toBe('/go');
    // Typing on opens it again.
    type(area, '/goa');
    expect(listbox(root)).not.toBeNull();
    key(area, { key: 'Escape' });
    key(area, { key: 'Escape' });
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it('closes while the text field has no focus; the muted line says more come later', () => {
    const { root, area } = setup('idle', true);
    type(area, '/');
    expect(root.textContent).toContain(CHAT.slash.later);
    act(() => area.blur());
    expect(listbox(root)).toBeNull();
  });

  it('the hint shows in the reason slot and leaves Send on; a reason replaces it', () => {
    const spies = spiesOf();
    const root = mount(
      createElement(Composer, {
        value: '/goal ',
        onValueChange: () => undefined,
        placeholder: 'Ask anything',
        phase: 'idle',
        onSend: spies.onSend,
        onStop: spies.onStop,
        tools: createElement('span', null, 'TOOLS'),
        hint: CHAT.slash.hint.goal,
        context: { used: 10, window: 100 },
      }),
    );
    const area = root.querySelector('textarea')!;
    const hint = [...root.querySelectorAll('span')].find((s) => s.textContent === CHAT.slash.hint.goal)!;
    expect(area.getAttribute('aria-describedby')).toBe(hint.id);
    key(area, { key: 'Enter' });
    expect(spies.onSend).toHaveBeenCalledTimes(1);
  });
});
