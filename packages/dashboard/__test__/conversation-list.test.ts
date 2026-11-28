/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  ConversationList,
  relativeTime,
  type ConversationItem,
  type ConversationListProps,
} from '../ui/src/components/arc-x/conversation-list/conversation-list.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { groupByDay } from '../ui/src/lib/conversations.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { allByRole, byRole } from './render.js';

afterEach(unmountAll);

const NOW = new Date(2026, 10, 12, 15, 0, 0).getTime();

const ITEMS: ConversationItem[] = [
  { id: 'a', title: 'Rust versions', mode: 'chat', updatedAt: NOW - 60_000, state: 'running' },
  {
    id: 'b',
    title: 'Fix the build',
    mode: 'work',
    updatedAt: NOW - 3_600_000,
    folder: '/Users/me/src/app',
    state: 'waiting',
  },
  { id: 'c', title: 'Old chat', mode: 'chat', updatedAt: new Date(2026, 5, 3).getTime(), state: 'idle' },
];

function list(extra: Partial<ConversationListProps> = {}) {
  return createElement(ConversationList, {
    label: 'Chats',
    groups: groupByDay(ITEMS, NOW),
    currentId: 'b',
    hrefFor: (id: string) => `#/chat/${id}`,
    onOpen: () => undefined,
    onRename: () => undefined,
    renderDelete: (item: ConversationItem) =>
      createElement('button', { type: 'button', 'aria-label': `Delete ${item.title}` }),
    now: NOW,
    ...extra,
  });
}

function key(target: Element, value: string, init: KeyboardEventInit = {}): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true, ...init }));
  });
}

function rows(container: HTMLElement): HTMLAnchorElement[] {
  return [...container.querySelectorAll<HTMLAnchorElement>('a[data-conversation-row]')];
}

describe('relativeTime', () => {
  it('reads like a list row', () => {
    expect(relativeTime(NOW - 10_000, NOW)).toBe('just now');
    expect(relativeTime(NOW - 120_000, NOW)).toBe('2m ago');
    expect(relativeTime(NOW - 3 * 3_600_000, NOW)).toBe('3h ago');
    expect(relativeTime(NOW - 5 * 86_400_000, NOW)).toBe('5d ago');
  });
});

describe('ConversationList', () => {
  it('groups with labelled lists, live dots, the current row; a Work row names its folder only with showFolder', () => {
    const container = mount(list());
    expect(byRole(container, 'navigation', 'Chats')).not.toBeNull();
    expect(byRole(container, 'list', 'Today')).not.toBeNull();
    expect(byRole(container, 'list', 'June 2026')).not.toBeNull();
    expect(container.querySelector('[aria-label="Running"]')).not.toBeNull();
    expect(container.querySelector('[aria-label="Waiting for you"]')).not.toBeNull();
    expect(container.querySelector('[aria-current="page"]')?.textContent).toContain('Fix the build');
    expect(container.querySelector('[data-two-line]')).toBeNull();
    expect(container.textContent).not.toContain('Work');
    rerender(list({ showFolder: true }));
    expect(container.querySelector('[data-two-line]')?.textContent).toContain('app');
  });

  it('one tab stop on the current row; arrows, Home and End move it', () => {
    const container = mount(list());
    const links = rows(container);
    expect(links.map((link) => link.tabIndex)).toEqual([-1, 0, -1]);
    for (const button of container.querySelectorAll('[data-conversation-actions] button'))
      expect((button as HTMLElement).tabIndex).toBe(-1);
    act(() => links[1]!.focus());
    key(links[1]!, 'ArrowDown');
    expect(document.activeElement).toBe(links[2]);
    key(links[2]!, 'Home');
    expect(document.activeElement).toBe(links[0]);
    key(links[0]!, 'End');
    expect(document.activeElement).toBe(links[2]);
  });

  it('→ reaches the actions, ← returns; F2 renames; ⌘⌫ presses delete', () => {
    const onRename = vi.fn();
    const deleted = vi.fn();
    const container = mount(
      list({
        onRename,
        renderDelete: (item) =>
          createElement('button', {
            type: 'button',
            'aria-label': `Delete ${item.title}`,
            onClick: () => deleted(item.id),
          }),
      }),
    );
    const links = rows(container);
    act(() => links[0]!.focus());
    key(links[0]!, 'ArrowRight');
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Rename');
    key(document.activeElement!, 'ArrowLeft');
    expect(document.activeElement).toBe(links[0]);
    key(links[0]!, 'F2');
    expect(onRename).toHaveBeenCalledWith(ITEMS[0]);
    key(links[0]!, 'Backspace', { metaKey: true });
    expect(deleted).toHaveBeenCalledWith('a');
  });

  it('a plain click opens in place; a modified click is left to the browser', () => {
    const onOpen = vi.fn();
    const container = mount(list({ onOpen }));
    act(() => rows(container)[0]!.click());
    expect(onOpen).toHaveBeenCalledWith('a');
    act(() => {
      rows(container)[0]!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, metaKey: true }));
    });
    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  it('search results tag the mode; loading shows a status; empty shows its slot', () => {
    const container = mount(list({ showMode: true }));
    expect(container.textContent).toContain('Work');
    expect(container.textContent).toContain('Chat');
    rerender(list({ loading: true }));
    expect(byRole(container, 'status')?.textContent).toContain('Loading conversations');
    rerender(list({ groups: [], empty: createElement('p', null, 'No chats yet') }));
    expect(container.textContent).toContain('No chats yet');
    expect(allByRole(container, 'link')).toHaveLength(0);
  });

  it('leaves list swap motion to SwapCell with no unconditional mount animation on .root', () => {
    const rules = cssRules('components/arc-x/conversation-list/conversation-list.module.css');
    expect(cssValue(rules, '.root', 'animation')).toBeUndefined();
    expect(cssValue(rules, '.root', 'animation-name')).toBeUndefined();
    expect(cssValue(rules, '.root', 'animation', 'reduce')).toBeUndefined();
    expect(cssValue(rules, '.row[data-enter]', 'animation')).toBe(
      'row-enter var(--duration-standard) var(--ease-enter) both',
    );
    expect(cssValue(rules, '.row[data-enter]', 'animation', 'reduce')).toBe('none');
  });
});

describe('ConversationList: open requests', () => {
  const at = (container: HTMLElement, id: string): HTMLElement =>
    container.querySelector<HTMLElement>(`a[data-conversation-row="${id}"]`)!.closest('li')!;

  it('a count badge takes the time slot and its words are in the row link; none at 0', () => {
    const items: ConversationItem[] = [
      { ...ITEMS[1]!, asks: 2 },
      { ...ITEMS[0]!, id: 'one', state: 'waiting', asks: 1 },
      { ...ITEMS[2]!, asks: 0 },
    ];
    const container = mount(list({ groups: groupByDay(items, NOW) }));
    const two = at(container, 'b');
    const badge = two.querySelector('[data-asks]')!;
    expect(badge.textContent).toBe('2');
    expect(badge.getAttribute('aria-hidden')).toBe('true');
    expect(two.querySelector('[class*="time"]')?.textContent).toBe('2');
    expect(two.querySelector('a')!.textContent).toContain(fill(CHAT.needs.listBadge, { n: 2 }));
    expect(two.querySelector('[aria-label="Waiting for you"]')).not.toBeNull();
    expect(at(container, 'one').querySelector('a')!.textContent).toContain(CHAT.needs.listBadgeOne);
    const idle = at(container, 'c');
    expect(idle.querySelector('[data-asks]')).toBeNull();
    expect(idle.querySelector('[class*="time"]')?.textContent).toBe(new Date(2026, 5, 3).toLocaleDateString());
  });
});

describe('ConversationList: forks', () => {
  const at = (container: HTMLElement, id: string): HTMLElement =>
    container.querySelector<HTMLElement>(`a[data-conversation-row="${id}"]`)!.closest('li')!;

  // Mutation: draw the tag only when `showMode` is on → a fork row reads like its original.
  it('a fork has an outline Fork tag after its title, read in the row link; the dot column holds only the state dot', () => {
    const items: ConversationItem[] = [
      { ...ITEMS[1]!, fork: true },
      { ...ITEMS[1]!, id: 'orig' },
    ];
    const container = mount(list({ groups: groupByDay(items, NOW) }));
    const copy = at(container, 'b');
    const tag = copy.querySelector('[data-fork]')!;
    expect(tag.textContent).toBe(CHAT.history.forkTag);
    expect(copy.querySelector('a')!.textContent).toBe(`${ITEMS[1]!.title}${CHAT.history.forkTag}`);
    expect(copy.querySelector('[class*="dot"]')!.children).toHaveLength(1);
    expect(copy.querySelector('a svg')).toBeNull();
    const original = at(container, 'orig');
    expect(original.querySelector('[data-fork]')).toBeNull();
    expect(original.querySelector('[class*="dot"]')!.innerHTML).toBe(copy.querySelector('[class*="dot"]')!.innerHTML);
  });
});
