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
import { groupByDay } from '../ui/src/lib/conversations.js';
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
  it('groups with labelled lists, live dots, the current row and a two-line Work row', () => {
    const container = mount(list());
    expect(byRole(container, 'navigation', 'Chats')).not.toBeNull();
    expect(byRole(container, 'list', 'Today')).not.toBeNull();
    expect(byRole(container, 'list', 'June 2026')).not.toBeNull();
    expect(container.querySelector('[aria-label="Running"]')).not.toBeNull();
    expect(container.querySelector('[aria-label="Waiting for you"]')).not.toBeNull();
    expect(container.querySelector('[aria-current="page"]')?.textContent).toContain('Fix the build');
    expect(container.querySelector('[data-two-line]')?.textContent).toContain('app');
    expect(container.textContent).not.toContain('Work');
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
});
