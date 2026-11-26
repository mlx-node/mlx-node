/**
 * @vitest-environment happy-dom
 */

import { act, createElement, Fragment, type ReactElement, useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  Menu,
  MenuGroup,
  MenuItem,
  MenuList,
  MenuRadio,
  MenuSeparator,
  MenuSub,
  SUB_OPEN_MS,
} from '../ui/src/components/arc-x/menu/menu.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { accessibleName, allByRole, byRole, waitFor } from './render.js';

afterEach(unmountAll);

const onAlpha = vi.fn();
const onBeta = vi.fn();
const changes: boolean[] = [];

afterEach(() => {
  onAlpha.mockClear();
  onBeta.mockClear();
  changes.length = 0;
});

function Harness({ shut = false }: { shut?: boolean }): ReactElement {
  const [open, setOpen] = useState(false);
  return createElement(Menu, {
    label: 'Things',
    open: open && !shut,
    onOpenChange: (next: boolean) => {
      changes.push(next);
      setOpen(next);
    },
    trigger: createElement('button', { type: 'button' }, 'Open'),
    children: createElement(
      Fragment,
      null,
      createElement(MenuItem, { label: 'Alpha', end: '⌘U', keys: 'Meta+U', onSelect: onAlpha }),
      createElement(MenuItem, { label: 'Beta', note: 'Only in local sessions', disabled: true, onSelect: onBeta }),
      createElement(MenuSub, {
        label: 'Gamma',
        end: 'On',
        children: createElement(
          Fragment,
          null,
          createElement('p', null, 'Gamma is on.'),
          createElement('a', { href: '/settings' }, 'Gamma settings'),
        ),
      }),
    ),
  });
}

function trigger(): HTMLButtonElement {
  return [...document.querySelectorAll('button')].find((b) => b.textContent === 'Open')!;
}

function menu(): HTMLElement | null {
  return byRole(document.body, 'menu', 'Things');
}

function panel(): HTMLElement | null {
  return byRole(document.body, 'dialog', 'Gamma');
}

function row(name: string): HTMLElement {
  const found = allByRole(document.body, 'menuitem').find((item) => accessibleName(item) === name);
  if (found === undefined) throw new Error(`no row ${name}`);
  return found;
}

function key(target: Element, name: string): KeyboardEvent {
  const event = new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true });
  act(() => {
    target.dispatchEvent(event);
  });
  return event;
}

function move(target: Element, x: number, y: number): void {
  act(() => {
    target.dispatchEvent(
      new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse', clientX: x, clientY: y }),
    );
  });
}

async function rest(ms: number): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  });
}

async function openMenu(): Promise<void> {
  mount(createElement(Harness));
  act(() => trigger().click());
  await waitFor(() => menu() !== null);
}

describe('Menu', () => {
  it('the trigger says it opens a menu; open, the rows are menu items and the first has focus', async () => {
    await openMenu();
    expect(trigger().getAttribute('aria-haspopup')).toBe('menu');
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    expect(allByRole(menu()!, 'menuitem').map(accessibleName)).toEqual(['Alpha', 'Beta', 'Gamma']);
    expect(document.activeElement).toBe(row('Alpha'));
  });

  it('keys: arrows move and wrap, Home / End, a letter, Tab stays', async () => {
    await openMenu();
    key(document.activeElement!, 'ArrowUp');
    expect(document.activeElement).toBe(row('Gamma'));
    key(document.activeElement!, 'ArrowDown');
    expect(document.activeElement).toBe(row('Alpha'));
    key(document.activeElement!, 'End');
    expect(document.activeElement).toBe(row('Gamma'));
    key(document.activeElement!, 'Home');
    expect(document.activeElement).toBe(row('Alpha'));
    key(document.activeElement!, 'g');
    expect(document.activeElement).toBe(row('Gamma'));
    const tab = key(document.activeElement!, 'Tab');
    expect(tab.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(row('Gamma'));
  });

  it('MenuItem: a click selects once and closes; a disabled row takes focus, reads its note and does nothing', async () => {
    await openMenu();
    key(document.activeElement!, 'ArrowDown');
    const beta = row('Beta');
    expect(document.activeElement).toBe(beta);
    expect(beta.getAttribute('aria-disabled')).toBe('true');
    const note = document.getElementById(beta.getAttribute('aria-describedby')!);
    expect(note?.textContent).toBe('Only in local sessions');
    act(() => beta.click());
    expect(onBeta).not.toHaveBeenCalled();
    expect(menu()).not.toBeNull();
    act(() => row('Alpha').click());
    expect(onAlpha).toHaveBeenCalledTimes(1);
    expect(changes.at(-1)).toBe(false);
    await waitFor(() => menu() === null);
  });

  it('a row that moves focus elsewhere keeps it there after the close; a row that does not gives it back to the trigger', async () => {
    function Elsewhere(): ReactElement {
      const [open, setOpen] = useState(false);
      return createElement(
        Fragment,
        null,
        createElement('input', { 'aria-label': 'Message' }),
        createElement(Menu, {
          label: 'Things',
          open,
          onOpenChange: setOpen,
          trigger: createElement('button', { type: 'button' }, 'Open'),
          children: createElement(
            Fragment,
            null,
            createElement(MenuItem, {
              label: 'Alpha',
              // Moved after the panel is gone, as a page's frame-late focus is.
              onSelect: () =>
                queueMicrotask(() => document.querySelector<HTMLInputElement>('input[aria-label="Message"]')!.focus()),
            }),
            createElement(MenuItem, { label: 'Beta', onSelect: onBeta }),
          ),
        }),
      );
    }
    const input = (): HTMLInputElement => document.querySelector<HTMLInputElement>('input[aria-label="Message"]')!;
    mount(createElement(Elsewhere));
    act(() => trigger().click());
    await waitFor(() => menu() !== null);
    act(() => row('Alpha').click());
    await waitFor(() => menu() === null);
    await rest(50);
    expect(document.activeElement).toBe(input());

    act(() => trigger().click());
    await waitFor(() => menu() !== null);
    act(() => row('Beta').click());
    expect(onBeta).toHaveBeenCalledTimes(1);
    await waitFor(() => menu() === null);
    await waitFor(() => document.activeElement === trigger());
  });

  it('keys: the row names its shortcut as aria-keyshortcuts and draws `end`; a row without keys names none', async () => {
    await openMenu();
    const alpha = row('Alpha');
    expect(alpha.getAttribute('aria-keyshortcuts')).toBe('Meta+U');
    expect(alpha.textContent).toBe('Alpha⌘U');
    expect(accessibleName(alpha)).toBe('Alpha');
    expect(row('Beta').hasAttribute('aria-keyshortcuts')).toBe(false);
  });

  it('a MenuSub row keeps the row class beside the class its popover trigger adds', async () => {
    await openMenu();
    const plain = [...row('Alpha').classList];
    expect(plain.length).toBeGreaterThan(0);
    const sub = row('Gamma').classList;
    for (const name of plain) expect(sub.contains(name)).toBe(true);
  });

  it('MenuSub by keys: the right arrow opens the panel into its link, the left arrow and Escape close one level', async () => {
    await openMenu();
    key(document.activeElement!, 'End');
    key(row('Gamma'), 'ArrowRight');
    await waitFor(() => panel() !== null);
    expect(row('Gamma').getAttribute('aria-haspopup')).toBe('dialog');
    expect(row('Gamma').getAttribute('aria-expanded')).toBe('true');
    await waitFor(() => document.activeElement?.textContent === 'Gamma settings');
    key(document.activeElement!, 'ArrowLeft');
    await waitFor(() => panel() === null);
    expect(document.activeElement).toBe(row('Gamma'));

    key(row('Gamma'), 'ArrowRight');
    await waitFor(() => document.activeElement?.textContent === 'Gamma settings');
    key(document.activeElement!, 'Escape');
    await waitFor(() => panel() === null);
    expect(menu()).not.toBeNull();
    await waitFor(() => document.activeElement === row('Gamma'));
    key(document.activeElement!, 'Escape');
    await waitFor(() => menu() === null);
    await waitFor(() => document.activeElement === trigger());
  });

  it('MenuSub by pointer: a rest opens the panel with focus on the row; another row closes it; a still pointer does not count', async () => {
    await openMenu();
    move(row('Gamma'), 10, 70);
    expect(document.activeElement).toBe(row('Gamma'));
    await rest(SUB_OPEN_MS + 30);
    await waitFor(() => panel() !== null);
    expect(document.activeElement).toBe(row('Gamma'));
    move(row('Alpha'), 10, 10);
    await waitFor(() => panel() === null);
    expect(document.activeElement).toBe(row('Alpha'));

    move(row('Gamma'), 10, 70);
    await rest(SUB_OPEN_MS + 30);
    await waitFor(() => panel() !== null);
    // The same coordinates again (a layout change under a still pointer) is not a move.
    move(row('Alpha'), 10, 70);
    expect(panel()).not.toBeNull();
    expect(document.activeElement).toBe(row('Gamma'));
  });

  it('keys inside the panel stay there; closing from the parent removes the panel with the menu', async () => {
    await openMenu();
    key(document.activeElement!, 'End');
    key(row('Gamma'), 'ArrowRight');
    await waitFor(() => document.activeElement?.textContent === 'Gamma settings');
    const link = document.activeElement!;
    key(link, 'ArrowDown');
    expect(document.activeElement).toBe(link);
    rerender(createElement(Harness, { shut: true }));
    await waitFor(() => menu() === null && panel() === null);
  });

  it('CSS: the focused row is the highlight, the open row is pressed, the panel is 18rem', () => {
    const rules = cssRules('components/arc-x/menu/menu.module.css');
    expect(cssValue(rules, '.row:focus', 'background')).toBe('var(--row-hover)');
    expect(cssValue(rules, ".row[aria-expanded='true']", 'background')).toBe('var(--row-active)');
    expect(cssValue(rules, '.sub.sub', 'width')).toBe('18rem');
  });
});

const onPick = vi.fn();
const onSettings = vi.fn();

afterEach(() => {
  onPick.mockClear();
  onSettings.mockClear();
});

/** A menu whose submenu panel holds a list: a caption, a group of three choices, a separator and one item. */
function ListHarness({ checked }: { checked: string | null }): ReactElement {
  const [open, setOpen] = useState(false);
  const choice = (label: string, over: { disabled?: boolean; end?: string } = {}) =>
    createElement(MenuRadio, { label, checked: checked === label, onSelect: () => onPick(label), ...over });
  return createElement(Menu, {
    label: 'Choices',
    open,
    onOpenChange: setOpen,
    trigger: createElement('button', { type: 'button' }, 'Open'),
    children: createElement(
      Fragment,
      null,
      createElement(MenuItem, { label: 'Plain', icon: createElement('i'), onSelect: () => undefined }),
      createElement(MenuSub, {
        label: 'Pick',
        children: createElement(
          Fragment,
          null,
          createElement('p', null, 'A caption'),
          createElement(
            MenuList,
            null,
            createElement(MenuGroup, {
              label: 'Fruit',
              children: createElement(
                Fragment,
                null,
                choice('Apple'),
                choice('Banana', { disabled: true, end: 'No key' }),
                choice('Cherry'),
              ),
            }),
            createElement(MenuSeparator),
            createElement(MenuItem, { label: 'Settings', note: 'Keys', onSelect: onSettings }),
          ),
        ),
      }),
    ),
  });
}

function choices(): HTMLElement | null {
  return byRole(document.body, 'menu', 'Choices');
}

function list(): HTMLElement | null {
  return byRole(document.body, 'menu', 'Pick');
}

function radio(name: string): HTMLElement {
  const found = allByRole(document.body, 'menuitemradio', name)[0];
  if (found === undefined) throw new Error(`no radio ${name}`);
  return found;
}

async function openList(checked: string | null = 'Cherry'): Promise<void> {
  mount(createElement(ListHarness, { checked }));
  act(() => trigger().click());
  await waitFor(() => choices() !== null);
  key(document.activeElement!, 'End');
  key(row('Pick'), 'ArrowRight');
  await waitFor(() => list() !== null);
}

describe('Menu: a list in a submenu panel', () => {
  it('the list is a menu named by the panel, with a group of radio rows, a separator and an item; only the checked row draws the check', async () => {
    await openList();
    expect(allByRole(choices()!, 'menuitem').map(accessibleName)).toEqual(['Plain', 'Pick']);
    expect(allByRole(choices()!, 'menuitemradio')).toEqual([]);
    const shown = list()!;
    expect(shown.closest('[role="dialog"]')?.getAttribute('aria-label')).toBe('Pick');
    expect(shown.textContent).not.toContain('A caption');
    const group = byRole(shown, 'group', 'Fruit')!;
    expect(allByRole(group, 'menuitemradio').map(accessibleName)).toEqual(['Apple', 'Banana', 'Cherry']);
    expect(['Apple', 'Banana', 'Cherry'].map((name) => radio(name).getAttribute('aria-checked'))).toEqual([
      'false',
      'false',
      'true',
    ]);
    expect(radio('Cherry').querySelector('svg')).not.toBeNull();
    expect(radio('Apple').querySelector('svg')).toBeNull();
    expect(byRole(shown, 'separator')).not.toBeNull();
    expect(allByRole(shown, 'menuitem').map(accessibleName)).toEqual(['Settings']);
  });

  it('a row with no icon draws no icon span; a row with one does', async () => {
    await openList();
    expect(radio('Apple').querySelector('[class*="icon"]')).toBeNull();
    expect(row('Plain').querySelector('[class*="icon"]')).not.toBeNull();
  });

  it('the right arrow opens the list with focus on the checked row; with none checked, on the first row', async () => {
    await openList();
    await waitFor(() => document.activeElement === radio('Cherry'));
    unmountAll();
    await openList(null);
    await waitFor(() => document.activeElement === radio('Apple'));
  });

  it('keys in the list: arrows wrap over every row with the disabled one, Home / End, a letter, Tab stays; the left arrow closes', async () => {
    await openList('Apple');
    await waitFor(() => document.activeElement === radio('Apple'));
    key(document.activeElement!, 'ArrowDown');
    expect(document.activeElement).toBe(radio('Banana'));
    key(document.activeElement!, 'ArrowDown');
    key(document.activeElement!, 'ArrowDown');
    expect(document.activeElement).toBe(row('Settings'));
    key(document.activeElement!, 'ArrowDown');
    expect(document.activeElement).toBe(radio('Apple'));
    key(document.activeElement!, 'ArrowUp');
    expect(document.activeElement).toBe(row('Settings'));
    key(document.activeElement!, 'Home');
    expect(document.activeElement).toBe(radio('Apple'));
    key(document.activeElement!, 'End');
    expect(document.activeElement).toBe(row('Settings'));
    key(document.activeElement!, 'c');
    expect(document.activeElement).toBe(radio('Cherry'));
    const tab = key(document.activeElement!, 'Tab');
    expect(tab.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(radio('Cherry'));
    key(document.activeElement!, 'ArrowLeft');
    await waitFor(() => list() === null);
    expect(document.activeElement).toBe(row('Pick'));
  });

  it('a radio click picks once and the menu stays open; a disabled radio takes focus and does nothing; an item closes both', async () => {
    await openList();
    act(() => radio('Apple').click());
    expect(onPick.mock.calls).toEqual([['Apple']]);
    expect(choices()).not.toBeNull();
    expect(list()).not.toBeNull();
    const banana = radio('Banana');
    expect(banana.getAttribute('aria-disabled')).toBe('true');
    expect(banana.textContent).toBe('BananaNo key');
    act(() => banana.focus());
    expect(document.activeElement).toBe(banana);
    act(() => banana.click());
    expect(onPick).toHaveBeenCalledTimes(1);
    act(() => row('Settings').click());
    expect(onSettings).toHaveBeenCalledTimes(1);
    await waitFor(() => choices() === null && list() === null);
  });

  it('the pointer over a list row focuses it and the panel stays open', async () => {
    await openList();
    move(radio('Apple'), 300, 10);
    expect(document.activeElement).toBe(radio('Apple'));
    move(row('Settings'), 300, 90);
    expect(document.activeElement).toBe(row('Settings'));
    await rest(SUB_OPEN_MS + 30);
    expect(list()).not.toBeNull();
  });

  it('CSS: the panel brings its own padding, the check mark sits after the right slot', () => {
    const rules = cssRules('components/arc-x/menu/menu.module.css');
    expect(cssValue(rules, '.sub.sub', 'padding')).toBe('var(--space-1)');
    expect(cssValue(rules, '.check', 'margin-inline-start')).toBe('var(--space-2)');
  });
});
