/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { ModelPicker, type ModelPickerProps } from '../ui/src/components/arc-x/model-picker/model-picker.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, unmountAll } from './helpers/mount.js';
import { allByRole, byRole, waitFor } from './render.js';

afterEach(unmountAll);

const OPTIONS: ModelPickerProps['options'] = [
  { id: 'qwen3.5-4b', state: 'loaded', facts: ['32K context'] },
  { id: 'qwen3.5-9b', state: 'idle', facts: ['Q4'], warning: 'small for Work' },
  { id: 'gemma4-e4b', state: 'loading', facts: [] },
];

function picker(extra: Partial<ModelPickerProps> = {}) {
  return createElement(ModelPicker, {
    value: 'qwen3.5-4b',
    options: OPTIONS,
    onValueChange: () => undefined,
    footer: 'Models load on first use.',
    swapWarning: 'Switching loads this model.',
    ...extra,
  });
}

const MANY: ModelPickerProps['options'] = Array.from({ length: 40 }, (_, index) => ({
  id: `model-${String(index)}`,
  state: 'loaded' as const,
  facts: [],
}));

const ROW = 44;
const VIEW = 374;

/**
 * happy-dom does no layout: each option is ROW tall at its index, the listbox shows VIEW px, and scrollTop is stored.
 * Returns the restore function.
 */
function stubLayout(): () => void {
  const scroll = new WeakMap<Element, number>();
  const place = (element: HTMLElement): number =>
    element.getAttribute('role') === 'option' && element.parentElement !== null
      ? [...element.parentElement.children].indexOf(element) * ROW
      : 0;
  const stubs: Record<string, PropertyDescriptor> = {
    offsetTop: {
      configurable: true,
      get(this: HTMLElement) {
        return place(this);
      },
    },
    offsetHeight: {
      configurable: true,
      get(this: HTMLElement) {
        return this.getAttribute('role') === 'option' ? ROW : 0;
      },
    },
    clientHeight: {
      configurable: true,
      get(this: HTMLElement) {
        return this.getAttribute('role') === 'listbox' ? VIEW : 0;
      },
    },
    scrollTop: {
      configurable: true,
      get(this: HTMLElement) {
        return scroll.get(this) ?? 0;
      },
      set(this: HTMLElement, value: number) {
        scroll.set(this, value);
      },
    },
  };
  const saved = new Map<string, PropertyDescriptor | undefined>();
  for (const [name, descriptor] of Object.entries(stubs)) {
    saved.set(name, Object.getOwnPropertyDescriptor(HTMLElement.prototype, name));
    Object.defineProperty(HTMLElement.prototype, name, descriptor);
  }
  return () => {
    for (const [name, descriptor] of saved) {
      if (descriptor === undefined) Reflect.deleteProperty(HTMLElement.prototype, name);
      else Object.defineProperty(HTMLElement.prototype, name, descriptor);
    }
  };
}

function pointerMove(target: Element, x: number, y: number): void {
  act(() => {
    target.dispatchEvent(new PointerEvent('pointermove', { clientX: x, clientY: y, bubbles: true }));
  });
}

function key(target: Element, value: string): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true }));
  });
}

async function open(container: HTMLElement): Promise<HTMLElement> {
  const trigger = container.querySelector<HTMLElement>('button[aria-haspopup="listbox"]')!;
  key(trigger, 'ArrowDown');
  await waitFor(() => byRole(document.body, 'listbox') !== null);
  return byRole(document.body, 'listbox')!;
}

describe('ModelPicker', () => {
  it('names the model, opens on ArrowDown with the selected row active', async () => {
    const container = mount(picker());
    const trigger = byRole(
      container,
      'button',
      fill(CHAT.pillName, { model: 'qwen3.5-4b', state: CHAT.pillStates.loaded }),
    );
    expect(trigger?.getAttribute('aria-haspopup')).toBe('listbox');
    expect(trigger?.getAttribute('data-state')).toBe('loaded');
    const list = await open(container);
    const options = allByRole(list, 'option');
    expect(options).toHaveLength(3);
    expect(options[0]!.getAttribute('aria-selected')).toBe('true');
    expect(list.getAttribute('aria-activedescendant')).toBe(options[0]!.id);
    expect(list.textContent).toContain('small for Work');
    expect(document.body.textContent).toContain('Switching loads this model.');
  });

  it('the footer is the same for every active row: the warning while a row is not loaded, else the footer', async () => {
    const container = mount(picker());
    const list = await open(container);
    const options = allByRole(list, 'option');
    expect(list.getAttribute('aria-activedescendant')).toBe(options[0]!.id);
    expect(document.body.textContent).toContain('Switching loads this model.');
    expect(document.body.textContent).not.toContain('Models load on first use.');
    key(list, 'ArrowDown');
    expect(list.getAttribute('aria-activedescendant')).toBe(options[1]!.id);
    expect(document.body.textContent).toContain('Switching loads this model.');
    unmountAll();
    const loaded = mount(picker({ value: 'model-0', options: MANY }));
    const all = await open(loaded);
    expect(document.body.textContent).toContain('Models load on first use.');
    key(all, 'ArrowDown');
    expect(document.body.textContent).toContain('Models load on first use.');
    expect(document.body.textContent).not.toContain('Switching loads this model.');
  });

  it('the memory row comes last, outside the listbox, with its action; the shared footer sits over it', async () => {
    const onUnload = vi.fn();
    const action = createElement('button', { type: 'button', onClick: onUnload }, 'Unload');
    const container = mount(
      picker({ value: 'model-0', options: MANY, memory: { text: 'model-0 is in memory.', action } }),
    );
    const list = await open(container);
    const panel = list.parentElement!;
    const memory = panel.lastElementChild!;
    expect(memory.textContent).toBe('model-0 is in memory.Unload');
    expect(list.contains(memory)).toBe(false);
    expect(memory.previousElementSibling?.textContent).toBe('Models load on first use.');
    act(() => memory.querySelector('button')!.click());
    expect(onUnload).toHaveBeenCalledTimes(1);
    const rules = cssRules('components/arc-x/model-picker/model-picker.module.css');
    expect(cssValue(rules, '.memory', 'flex')).toBe('none');
    expect(cssValue(rules, '.row', 'min-height')).toBe('var(--picker-row)');
    unmountAll();
    const none = mount(picker());
    await open(none);
    expect(document.body.textContent).not.toContain('is in memory');
  });

  it('arrows move, Enter picks', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ onValueChange }));
    const list = await open(container);
    key(list, 'ArrowDown');
    const options = allByRole(list, 'option');
    expect(list.getAttribute('aria-activedescendant')).toBe(options[1]!.id);
    key(list, 'End');
    expect(list.getAttribute('aria-activedescendant')).toBe(options[2]!.id);
    key(list, 'Enter');
    expect(onValueChange).toHaveBeenCalledWith('gemma4-e4b');
  });

  it('the pill says whether its model is in memory, by state and by name', () => {
    for (const [value, state] of [
      ['qwen3.5-9b', 'idle'],
      ['gemma4-e4b', 'loading'],
    ] as const) {
      const container = mount(picker({ value }));
      const trigger = container.querySelector('button[aria-haspopup="listbox"]')!;
      expect(trigger.getAttribute('data-state')).toBe(state);
      expect(trigger.getAttribute('aria-label')).toBe(
        fill(CHAT.pillName, { model: value, state: CHAT.pillStates[state] }),
      );
      unmountAll();
    }
  });

  it('a long row name is whole in the row, cut in the middle only by the width', async () => {
    const LONG = 'qwen3.6-35b-a3b-agentworld-instruct-ud-q4-k-xl-mlx-dwq-8bits';
    const container = mount(picker({ options: [...OPTIONS, { id: LONG, state: 'idle', facts: [] }] }));
    const list = await open(container);
    const row = allByRole(list, 'option')[3]!;
    expect(row.textContent).toContain(LONG);
    expect(row.querySelector(`[title="${LONG}"]`)).not.toBeNull();
    expect(document.body.textContent).toContain(CHAT.modelIdle);
  });

  it('no model: a disabled trigger that says so', () => {
    const container = mount(picker({ value: null, options: [] }));
    const trigger = byRole(container, 'button', 'No model') as HTMLButtonElement;
    expect(trigger.disabled).toBe(true);
  });

  it('disabled with a reason: the trigger is off and the reason is its tooltip', () => {
    const container = mount(picker({ disabled: true, disabledReason: 'A turn is running.' }));
    const trigger = container.querySelector<HTMLButtonElement>('button[aria-haspopup="listbox"]')!;
    expect(trigger.disabled).toBe(true);
  });

  describe('a long list', () => {
    let restore: (() => void) | undefined;
    afterEach(() => {
      restore?.();
      restore = undefined;
    });

    it('scrolls inside the panel: the list is the bounded scroller, heading and footer stay outside it', async () => {
      const rules = cssRules('components/arc-x/model-picker/model-picker.module.css');
      expect(cssValue(rules, '.panel.panel', 'max-height')).toBe(
        'var(--radix-popover-content-available-height, 100vh)',
      );
      expect(cssValue(rules, '.panel.panel', 'flex-direction')).toBe('column');
      expect(cssValue(rules, '.list', 'max-height')).toBe('calc(var(--picker-row) * 8.5)');
      expect(cssValue(rules, '.list', 'min-height')).toBe('0');
      expect(cssValue(rules, '.list', 'overflow-y')).toBe('auto');
      expect(cssValue(rules, '.list', 'overscroll-behavior')).toBe('contain');
      expect(cssValue(rules, '.heading', 'flex')).toBe('none');
      expect(cssValue(rules, '.footer', 'flex')).toBe('none');

      const container = mount(picker({ options: MANY, value: 'model-0' }));
      const list = await open(container);
      expect(allByRole(list, 'option')).toHaveLength(40);
      expect(list.className.startsWith('_list_')).toBe(true);
      const panel = list.parentElement!;
      const heading = panel.querySelector('p')!;
      expect(heading.textContent).toBe('Model');
      expect(list.contains(heading)).toBe(false);
      const footer = [...panel.children].find((child) => child.textContent === 'Models load on first use.');
      expect(footer).toBeDefined();
      expect(list.contains(footer!)).toBe(false);
      expect(heading.compareDocumentPosition(list) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(list.compareDocumentPosition(footer!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });

    it('opens with the selected row in view', async () => {
      restore = stubLayout();
      const container = mount(picker({ options: MANY, value: 'model-30' }));
      const list = await open(container);
      expect(list.getAttribute('aria-activedescendant')).toBe(allByRole(list, 'option')[30]!.id);
      await waitFor(() => list.scrollTop === 31 * ROW - VIEW);
    });

    it('ArrowDown, ArrowUp, Home and End keep the active row in view', async () => {
      restore = stubLayout();
      const container = mount(picker({ options: MANY, value: 'model-0' }));
      const list = await open(container);
      expect(list.scrollTop).toBe(0);
      for (let step = 0; step < 9; step += 1) key(list, 'ArrowDown');
      expect(list.getAttribute('aria-activedescendant')).toBe(allByRole(list, 'option')[9]!.id);
      expect(list.scrollTop).toBe(10 * ROW - VIEW);
      key(list, 'End');
      expect(list.scrollTop).toBe(40 * ROW - VIEW);
      for (let step = 0; step < 10; step += 1) key(list, 'ArrowUp');
      expect(list.getAttribute('aria-activedescendant')).toBe(allByRole(list, 'option')[29]!.id);
      expect(list.scrollTop).toBe(29 * ROW);
      key(list, 'Home');
      expect(list.scrollTop).toBe(0);
    });

    it('a pointer that rests while the list scrolls by key keeps the key row; a real move takes it', async () => {
      const container = mount(picker({ options: MANY, value: 'model-0' }));
      const list = await open(container);
      const options = allByRole(list, 'option');
      pointerMove(options[5]!, 40, 200);
      expect(list.getAttribute('aria-activedescendant')).toBe(options[5]!.id);
      key(list, 'ArrowDown');
      expect(list.getAttribute('aria-activedescendant')).toBe(options[6]!.id);
      pointerMove(options[3]!, 40, 200);
      expect(list.getAttribute('aria-activedescendant')).toBe(options[6]!.id);
      pointerMove(options[3]!, 40, 204);
      expect(list.getAttribute('aria-activedescendant')).toBe(options[3]!.id);
    });
  });
});
