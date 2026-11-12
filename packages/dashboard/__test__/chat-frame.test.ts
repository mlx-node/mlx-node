/**
 * @vitest-environment happy-dom
 */

import { act, createElement, useState, type ReactElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { ChatFrame, JumpButton } from '../ui/src/components/arc-x/chat-frame/chat-frame.js';
import { flush, mount, unmountAll } from './helpers/mount.js';
import { byRole, observeWidth, waitFor } from './render.js';

let restore: (() => void) | undefined;

afterEach(() => {
  unmountAll();
  restore?.();
  restore = undefined;
});

function Frame(): ReactElement {
  const [open, setOpen] = useState(false);
  return createElement(ChatFrame, {
    barStart: createElement('span', null, 'MODE'),
    barEnd: createElement('h1', null, 'Title'),
    list: createElement('a', { href: '#c1' }, 'Conversation one'),
    listLabel: 'Chats',
    listOpen: open,
    onListOpenChange: setOpen,
    dock: createElement('textarea', { 'aria-label': 'Message' }),
    jump: createElement(JumpButton, { onJump: () => undefined }),
    children: createElement('p', null, 'MESSAGE BODY'),
  });
}

describe('ChatFrame', () => {
  it('docked (wide): the list is a named column, no drawer toggle', async () => {
    restore = observeWidth(1200);
    const container = mount(createElement(Frame));
    await flush();
    expect(container.textContent).toContain('Conversation one');
    expect(byRole(container, 'button', 'Chats')).toBeNull();
    expect(byRole(container, 'region', 'Messages')).not.toBeNull();
    expect(container.textContent).toContain('MESSAGE BODY');
    expect(container.querySelector('textarea')).not.toBeNull();
    expect(byRole(container, 'region', 'Chats')).toBeNull();
    const scroller = container.querySelector('a[href="#c1"]')?.parentElement;
    expect(scroller?.getAttribute('tabindex')).toBe('-1');
  });

  it('narrow: the list moves into a drawer opened from the bar', async () => {
    restore = observeWidth(668);
    const container = mount(createElement(Frame));
    await flush();
    expect(container.textContent).not.toContain('Conversation one');
    const toggle = byRole(container, 'button', 'Chats');
    expect(toggle?.getAttribute('aria-expanded')).toBe('false');
    act(() => toggle!.click());
    await waitFor(() => document.body.textContent?.includes('Conversation one') === true);
    expect(byRole(document.body, 'dialog', 'Chats')).not.toBeNull();
  });

  it('narrow: Esc closes the drawer and focus goes back to the button that opened it', async () => {
    restore = observeWidth(668);
    const container = mount(createElement(Frame));
    await flush();
    const toggle = byRole(container, 'button', 'Chats')!;
    act(() => toggle.focus());
    act(() => toggle.click());
    await waitFor(() => byRole(document.body, 'dialog', 'Chats') !== null);
    const dialog = byRole(document.body, 'dialog', 'Chats')!;
    act(() => {
      dialog.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });
    await waitFor(() => byRole(document.body, 'dialog', 'Chats') === null, 3_000);
    await waitFor(() => document.activeElement === toggle);
  });
});

describe('JumpButton', () => {
  it('names new content and calls back', () => {
    const onJump = vi.fn();
    const container = mount(createElement(JumpButton, { onJump, fresh: true }));
    const button = byRole(container, 'button', 'Jump to latest, new messages');
    expect(button).not.toBeNull();
    act(() => button!.click());
    expect(onJump).toHaveBeenCalledTimes(1);
  });
});
