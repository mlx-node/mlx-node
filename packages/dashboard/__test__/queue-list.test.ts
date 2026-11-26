/**
 * @vitest-environment happy-dom
 */

import { act, createElement, useState, type ReactElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { QueueList, type QueueListProps } from '../ui/src/components/arc-x/queue-list/queue-list.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { ChatHostState, QueuedMessage } from '../ui/src/lib/chat-types.js';
import { formatBytes } from '../ui/src/lib/format.js';
import { heldSentence } from '../ui/src/pages/chat/queue-actions.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { accessibleName, byRole, waitFor } from './render.js';

afterEach(unmountAll);

function item(id: string, text: string, extra: Partial<QueuedMessage> = {}): QueuedMessage {
  return { id, text, files: [], queuedAt: 1, editing: false, ...extra };
}

function props(extra: Partial<QueueListProps> = {}): QueueListProps {
  return {
    items: [item('a', 'also update the changelog'), item('b', 'and the README')],
    held: null,
    editing: null,
    onEditingChange: () => undefined,
    onSave: () => Promise.resolve(true),
    onRemove: () => undefined,
    onClear: () => undefined,
    onSendNext: () => undefined,
    onFocusOut: () => undefined,
    ...extra,
  };
}

function key(target: Element, init: KeyboardEventInit): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }));
  });
}

function typeInto(area: HTMLTextAreaElement, text: string): void {
  act(() => {
    Reflect.set(HTMLTextAreaElement.prototype, 'value', text, area);
    area.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function rows(container: Element): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>('[role="listitem"]'));
}

describe('QueueList', () => {
  it('a named list of rows: number, text, files by name and size, the plan tag, Edit and Remove; no row is no list', () => {
    const container = mount(
      createElement(
        QueueList,
        props({
          items: [
            item('a', 'see the log', { files: [{ kind: 'file', name: 'build.log', bytes: 12_288 }], plan: true }),
            item('b', 'then ship'),
          ],
        }),
      ),
    );
    const list = byRole(container, 'list', CHAT.queue.label)!;
    expect(list).not.toBeNull();
    expect(container.textContent).toContain(fill(CHAT.queue.title, { n: 2 }));
    const [first, second] = rows(container);
    expect(first!.textContent).toContain('1');
    expect(first!.textContent).toContain('see the log');
    expect(first!.textContent).toContain('build.log');
    expect(first!.textContent).toContain(formatBytes(12_288));
    expect(first!.textContent).toContain(CHAT.queue.plan);
    expect(second!.textContent).not.toContain(CHAT.queue.plan);
    expect(Array.from(first!.querySelectorAll('button')).map(accessibleName)).toEqual([
      `${CHAT.queue.edit}: see the log`,
      CHAT.queue.edit,
      CHAT.queue.remove,
    ]);
    expect(byRole(container, 'button', CHAT.queue.sendNext)).toBeNull();
    rerender(createElement(QueueList, props({ items: [] })));
    expect(container.textContent).toBe('');
  });

  it('Work while an answer runs: Send now first in each row (tab order Send now, Edit, Remove); a press sends that row', () => {
    const onSend = vi.fn();
    const container = mount(createElement(QueueList, props({ sendNow: { off: () => null, busy: new Set(), onSend } })));
    const [first, second] = rows(container);
    expect(Array.from(first!.querySelectorAll('button')).map(accessibleName)).toEqual([
      `${CHAT.queue.edit}: also update the changelog`,
      CHAT.queue.sendNow,
      CHAT.queue.edit,
      CHAT.queue.remove,
    ]);
    act(() => byRole(second!, 'button', CHAT.queue.sendNow)!.click());
    expect(onSend).toHaveBeenCalledWith('b');
    // Chat or no answer: no Send now.
    rerender(createElement(QueueList, props()));
    expect(byRole(container, 'button', CHAT.queue.sendNow)).toBeNull();
  });

  it('Send now off: it stays focusable, a press does nothing, and its tooltip says why; a press in flight is off too', async () => {
    const onSend = vi.fn();
    const off = (row: QueuedMessage): string | null =>
      row.files.length > 0 ? CHAT.queue.sendNowOffFiles : row.text.startsWith('/') ? CHAT.queue.sendNowOffSlash : null;
    const container = mount(
      createElement(
        QueueList,
        props({
          items: [
            item('a', 'see the log', { files: [{ kind: 'file', name: 'build.log', bytes: 12 }] }),
            item('b', 'then ship'),
          ],
          sendNow: { off, busy: new Set(['b']), onSend },
        }),
      ),
    );
    const [first, second] = rows(container);
    const files = byRole(first!, 'button', CHAT.queue.sendNow) as HTMLButtonElement;
    expect(files.getAttribute('aria-disabled')).toBe('true');
    expect(files.disabled).toBe(false);
    act(() => files.click());
    const busy = byRole(second!, 'button', CHAT.queue.sendNow) as HTMLButtonElement;
    expect(busy.getAttribute('aria-disabled')).toBe('true');
    act(() => busy.click());
    expect(onSend).not.toHaveBeenCalled();
    act(() => files.focus());
    await waitFor(
      () => document.querySelector('[role="tooltip"]')?.textContent?.includes(CHAT.queue.sendNowOffFiles) === true,
    );
  });

  it('a held list says why in its header and offers Send next; Clear is always there', () => {
    const onSendNext = vi.fn();
    const onClear = vi.fn();
    const container = mount(createElement(QueueList, props({ held: CHAT.queue.held.stopped, onSendNext, onClear })));
    expect(container.textContent).toContain(`${fill(CHAT.queue.title, { n: 2 })} · ${CHAT.queue.held.stopped}`);
    act(() => byRole(container, 'button', CHAT.queue.sendNext)!.click());
    act(() => byRole(container, 'button', CHAT.queue.clear)!.click());
    expect(onSendNext).toHaveBeenCalledTimes(1);
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  function Editable(extra: Partial<QueueListProps>): ReactElement {
    const [editing, setEditing] = useState<string | null>(null);
    return createElement(
      QueueList,
      props({
        ...extra,
        editing,
        onEditingChange: (id) => {
          extra.onEditingChange?.(id);
          setEditing(id);
        },
        onSave: async (id, text) => {
          const saved = (await extra.onSave?.(id, text)) ?? true;
          if (saved) setEditing(null);
          return saved;
        },
      }),
    );
  }

  it('the row text opens the edit; Enter saves, Shift+Enter is a new line, Esc leaves and gives focus to ✎', async () => {
    const onEditingChange = vi.fn();
    const onSave = vi.fn(() => Promise.resolve(true));
    const container = mount(createElement(Editable, { onEditingChange, onSave }));
    act(() => byRole(container, 'button', `${CHAT.queue.edit}: and the README`)!.click());
    expect(onEditingChange).toHaveBeenLastCalledWith('b');
    let area = byRole(container, 'textbox', CHAT.queue.editLabel) as HTMLTextAreaElement;
    expect(area.value).toBe('and the README');
    expect(document.activeElement).toBe(area);
    key(area, { key: 'Enter', shiftKey: true });
    expect(onSave).not.toHaveBeenCalled();
    key(area, { key: 'Escape' });
    expect(onEditingChange).toHaveBeenLastCalledWith(null);
    expect(byRole(container, 'textbox', CHAT.queue.editLabel)).toBeNull();
    const editB = rows(container)[1]!.querySelectorAll('button')[1]!;
    expect(document.activeElement).toBe(editB);

    act(() => (editB as HTMLButtonElement).click());
    area = byRole(container, 'textbox', CHAT.queue.editLabel) as HTMLTextAreaElement;
    typeInto(area, 'and the README too');
    await act(async () => {
      key(area, { key: 'Enter' });
      await Promise.resolve();
    });
    expect(onSave).toHaveBeenCalledWith('b', 'and the README too');
    expect(byRole(container, 'textbox', CHAT.queue.editLabel)).toBeNull();
  });

  it('a blank edit cannot be saved; a save that failed keeps the edit open', async () => {
    const onSave = vi.fn(() => Promise.resolve(false));
    const container = mount(createElement(Editable, { onSave }));
    act(() => byRole(container, 'button', `${CHAT.queue.edit}: also update the changelog`)!.click());
    const area = byRole(container, 'textbox', CHAT.queue.editLabel) as HTMLTextAreaElement;
    typeInto(area, '   ');
    key(area, { key: 'Enter' });
    expect(onSave).not.toHaveBeenCalled();
    expect((byRole(container, 'button', CHAT.queue.save) as HTMLButtonElement).disabled).toBe(true);
    typeInto(area, 'changed');
    await act(async () => {
      byRole(container, 'button', CHAT.queue.save)!.click();
      await Promise.resolve();
    });
    expect(onSave).toHaveBeenCalledWith('a', 'changed');
    expect(byRole(container, 'textbox', CHAT.queue.editLabel)).not.toBeNull();
  });

  it('after Remove focus goes to the next row’s Remove, else back to the message box', () => {
    const onRemove = vi.fn();
    const onFocusOut = vi.fn();
    const container = mount(createElement(QueueList, props({ onRemove, onFocusOut })));
    act(() => (rows(container)[0]!.querySelectorAll('button')[2] as HTMLButtonElement).click());
    expect(onRemove).toHaveBeenLastCalledWith('a');
    rerender(createElement(QueueList, props({ onRemove, onFocusOut, items: [item('b', 'and the README')] })));
    const removeB = rows(container)[0]!.querySelectorAll('button')[2];
    expect(document.activeElement).toBe(removeB);
    expect(onFocusOut).not.toHaveBeenCalled();
    act(() => (removeB as HTMLButtonElement).click());
    expect(onRemove).toHaveBeenLastCalledWith('b');
    expect(onFocusOut).toHaveBeenCalledTimes(1);
  });
});

describe('heldSentence', () => {
  /** `tasks` live commands in all; `here` of them in conversation `c1`. */
  const host = (tasks: number, here = tasks): ChatHostState =>
    ({
      modes: { work: { tasks } },
      active: [{ conversationId: 'c1', mode: 'work', tasks: here }],
    }) as unknown as ChatHostState;

  it('names why the list waits in the page words, the engine-lost hold with the command count', () => {
    expect(heldSentence(null, null, 'c1', 'work')).toBeNull();
    expect(heldSentence({ why: 'stopped' }, null, 'c1', 'work')).toBe(CHAT.queue.held.stopped);
    expect(heldSentence({ why: 'failed' }, null, 'c1', 'work')).toBe(CHAT.queue.held.failed);
    expect(heldSentence({ why: 'restart' }, null, 'c1', 'work')).toBe(CHAT.queue.held.restart);
    // A failed check says the page's sentence for its code, never the host's text as such.
    expect(
      heldSentence({ why: 'blocked', reason: CHAT.cwdProblems.missing, code: 'cwd-invalid' }, null, 'c1', 'work'),
    ).toBe(fill(CHAT.queue.held.blocked, { reason: CHAT.cwdProblems.missing }));
    expect(heldSentence({ why: 'blocked', reason: 'odd host words', code: 'cwd-invalid' }, null, 'c1', 'work')).toBe(
      fill(CHAT.queue.held.blocked, { reason: CHAT.sendErrors['cwd-invalid'][0] }),
    );
    expect(heldSentence({ why: 'blocked', reason: 'host words', code: 'pictures-full' }, null, 'c1', 'chat')).toBe(
      fill(CHAT.queue.held.blocked, { reason: CHAT.sendErrors['pictures-full'][0] }),
    );
    expect(heldSentence({ why: 'blocked', reason: 'host words', code: null }, null, 'c1', 'chat')).toBe(
      fill(CHAT.queue.held.blocked, { reason: CHAT.sendErrors.other[0] }),
    );
    const lost = { why: 'blocked', reason: 'host words', code: 'engine-lost' } as const;
    expect(heldSentence(lost, host(2), 'c1', 'work')).toBe(fill(CHAT.queue.held.engineLost, { n: 2 }));
    expect(heldSentence(lost, host(1), 'c1', 'work')).toBe(CHAT.queue.held.engineLostOne);
    expect(heldSentence(lost, host(0), 'c1', 'work')).toBe(
      fill(CHAT.queue.held.blocked, { reason: CHAT.history.engineLost }),
    );
    // The agent process is shared: the count is every conversation's, and names the others.
    expect(heldSentence(lost, host(3, 1), 'c1', 'work')).toBe(
      `${fill(CHAT.queue.held.engineLost, { n: 3 })} ${fill(CHAT.engineRestartOthers, { n: 2 })}`,
    );
    expect(heldSentence(lost, host(1, 1), 'c2', 'work')).toBe(
      `${CHAT.queue.held.engineLostOne} ${fill(CHAT.engineRestartOthers, { n: 1 })}`,
    );
  });
});
