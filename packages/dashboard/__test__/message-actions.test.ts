/**
 * @vitest-environment happy-dom
 */

/**
 * Under a sent message: Copy and `⋯` with Rewind to here. The row reads its note, or why it is off (it stays focusable
 * and does nothing then); a choice hands over the `⋯` button. In the thread only a message the user sent has `⋯`: not a
 * steer, a note, the automatic continue or a wake.
 */

import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { MessageActions } from '../ui/src/components/arc-x/message-actions/message-actions.js';
import { ChatThread, type MessageRewind } from '../ui/src/components/chat-thread.js';
import { CHAT } from '../ui/src/lib/chat-strings.js';
import { CONTINUE_PROMPT } from '../ui/src/lib/chat-types.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import { answer, end, T0, user } from './helpers/chat-thread-fixtures.js';
import { mount, unmountAll } from './helpers/mount.js';
import { accessibleName, allByRole, byRole, waitFor } from './render.js';

afterEach(unmountAll);

const H = CHAT.history;

function menuButtons(scope: Element = document.body): HTMLElement[] {
  return [...scope.querySelectorAll<HTMLElement>('[data-message-menu]')];
}

async function openMenu(button: HTMLElement): Promise<HTMLElement> {
  act(() => button.click());
  await waitFor(() => byRole(document.body, 'menu', H.messageMenu) !== null);
  return byRole(document.body, 'menu', H.messageMenu)!;
}

function actions(off: string | null, onSelect: (from: HTMLElement | null) => void) {
  return createElement(MessageActions, {
    text: 'fix the build',
    copyLabel: CHAT.copyMessage,
    menuLabel: H.messageMenu,
    rewind: { label: H.rewind, note: H.rewindNote, off, onSelect },
  });
}

describe('MessageActions', () => {
  // Mutation: hand `null` to onSelect → the `from` check fails.
  it('Copy and ⋯; the row reads its note, and a choice hands over the ⋯ button', async () => {
    const onSelect = vi.fn();
    const container = mount(actions(null, onSelect));
    expect(allByRole(container, 'button').map(accessibleName)).toEqual([CHAT.copyMessage, H.messageMenu]);
    const trigger = menuButtons(container)[0]!;
    expect(trigger.getAttribute('aria-haspopup')).toBe('menu');
    const menu = await openMenu(trigger);
    const rows = allByRole(menu, 'menuitem');
    expect(rows.map(accessibleName)).toEqual([H.rewind]);
    expect(document.getElementById(rows[0]!.getAttribute('aria-describedby')!)?.textContent).toBe(H.rewindNote);
    act(() => rows[0]!.click());
    expect(onSelect).toHaveBeenCalledWith(trigger);
  });

  it('Fork with this answer is the second row when given, with its own note; off reads why', async () => {
    const onFork = vi.fn();
    const make = (off: string | null) =>
      createElement(MessageActions, {
        text: 'fix the build',
        copyLabel: CHAT.copyMessage,
        menuLabel: H.messageMenu,
        rewind: { label: H.rewind, note: H.rewindNote, off, onSelect: () => {} },
        fork: { label: H.forkHere, note: H.forkHereNote, off, onSelect: onFork },
      });
    const container = mount(make(null));
    const menu = await openMenu(menuButtons(container)[0]!);
    const rows = allByRole(menu, 'menuitem');
    expect(rows.map(accessibleName)).toEqual([H.rewind, H.forkHere]);
    expect(document.getElementById(rows[1]!.getAttribute('aria-describedby')!)?.textContent).toBe(H.forkHereNote);
    act(() => rows[1]!.click());
    expect(onFork).toHaveBeenCalledTimes(1);
    unmountAll();
    const off = mount(make(CHAT.modelBusy));
    const row = allByRole(await openMenu(menuButtons(off)[0]!), 'menuitem')[1]!;
    expect(row.getAttribute('aria-disabled')).toBe('true');
    expect(document.getElementById(row.getAttribute('aria-describedby')!)?.textContent).toBe(CHAT.modelBusy);
    act(() => row.click());
    expect(onFork).toHaveBeenCalledTimes(1);
  });

  it('off: the row reads why, stays focusable and does nothing', async () => {
    const onSelect = vi.fn();
    const container = mount(actions(CHAT.modelBusy, onSelect));
    const menu = await openMenu(menuButtons(container)[0]!);
    const row = allByRole(menu, 'menuitem')[0]!;
    expect(row.getAttribute('aria-disabled')).toBe('true');
    expect(document.getElementById(row.getAttribute('aria-describedby')!)?.textContent).toBe(CHAT.modelBusy);
    act(() => row.click());
    expect(onSelect).not.toHaveBeenCalled();
  });
});

describe('the thread: which rows have ⋯', () => {
  const entries: TranscriptEntry[] = [
    user('first'),
    answer('one'),
    end(),
    { kind: 'user', ts: T0 + 2_000, text: 'steer', promptIndex: null, interjection: true },
    { kind: 'user', ts: T0 + 2_100, text: 'a note', promptIndex: null, note: true },
    { kind: 'user', ts: T0 + 2_200, text: CONTINUE_PROMPT, promptIndex: null, continued: true },
    { kind: 'wake', ts: T0 + 2_300, cause: 'task', text: 'woken' },
    user('second', T0 + 3_000),
    answer('two', T0 + 3_100),
    end('end_turn', T0 + 3_200),
  ];

  function thread(rewind?: MessageRewind) {
    return createElement(
      MemoryRouter,
      null,
      createElement(ChatThread, {
        mode: 'work',
        entries,
        turns: [],
        turn: null,
        lastEnd: null,
        ...(rewind === undefined ? {} : { rewind }),
      }),
    );
  }

  // Mutation: draw MessageActions on a steer → three ⋯ buttons.
  it('only the two sent messages; their choice names their own entry', async () => {
    const onRewind = vi.fn();
    const container = mount(thread({ off: null, onRewind }));
    const buttons = menuButtons(container);
    expect(buttons).toHaveLength(2);
    const menu = await openMenu(buttons[1]!);
    act(() => allByRole(menu, 'menuitem')[0]!.click());
    expect(onRewind).toHaveBeenCalledWith(7, buttons[1]);
  });

  // Mutation: hand the fork the rewind's entry of the first message → the second message forks the wrong answer.
  it('Fork with this answer names the message’s own entry', async () => {
    const onFork = vi.fn();
    const container = mount(thread({ off: null, onRewind: () => {}, onFork }));
    const buttons = menuButtons(container);
    const menu = await openMenu(buttons[1]!);
    act(() => byRole(menu, 'menuitem', H.forkHere)!.click());
    expect(onFork).toHaveBeenCalledWith(7);
  });

  it('with no rewind the messages have Copy only', () => {
    const container = mount(thread());
    expect(menuButtons(container)).toEqual([]);
    expect(allByRole(container, 'button').filter((b) => accessibleName(b) === CHAT.copyMessage).length).toBeGreaterThan(
      1,
    );
  });
});
