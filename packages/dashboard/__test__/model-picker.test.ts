/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  type ModelBaseOption,
  ModelPicker,
  type ModelPickerProps,
} from '../ui/src/components/arc-x/model-picker/model-picker.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { allByRole, byRole, waitFor } from './render.js';

afterEach(unmountAll);

const XL = 'Qwen3.8-27B-UD-Q4_K_XL';
const M = 'Qwen3.8-27B-UD-Q4_K_M';
const BF16 = 'qwen3.8-27b-bf16';

const QWEN27: ModelBaseOption = {
  base: 'Qwen3.8-27B',
  state: 'loaded',
  stateVariant: 'UD-Q4_K_XL',
  facts: ['3 quantizations', '256K context'],
  preselect: XL,
  variants: [
    { id: XL, label: 'UD-Q4_K_XL', state: 'loaded', facts: ['GGUF', '17.6 GB'] },
    { id: M, label: 'UD-Q4_K_M', state: 'idle', facts: ['GGUF', '16.5 GB'] },
    { id: BF16, label: 'BF16', state: 'idle', facts: ['MLX', '55.6 GB'] },
  ],
};
const AGENTS: ModelBaseOption = {
  base: 'agents-a1',
  state: 'idle',
  facts: ['256K context', 'MLX', '70.2 GB'],
  preselect: 'agents-a1',
  variants: [{ id: 'agents-a1', label: 'BF16', state: 'idle', facts: ['MLX', '70.2 GB'] }],
};
const QWEN36: ModelBaseOption = {
  base: 'Qwen3.6-35B-A3B',
  state: 'idle',
  facts: ['4 quantizations', '256K context'],
  preselect: 'Qwen3.6-35B-A3B-UD-Q4_K_XL',
  variants: [
    { id: 'Qwen3.6-35B-A3B-UD-IQ2_M', label: 'UD-IQ2_M', state: 'idle', facts: ['GGUF', '11.9 GB'] },
    { id: 'Qwen3.6-35B-A3B-UD-Q4_K_XL', label: 'UD-Q4_K_XL', state: 'idle', facts: ['GGUF', '22.9 GB'] },
    { id: 'qwen3.6-35b-a3b-nvfp4', label: 'NVFP4', state: 'idle', facts: ['MLX', '26.5 GB'] },
    { id: 'qwen3.6-35b-a3b-bf16', label: 'BF16', state: 'idle', facts: ['MLX', '71.9 GB'] },
  ],
};
const SMALL: ModelBaseOption = {
  base: 'qwen3-0.6b',
  state: 'loading',
  facts: ['16K context', 'MLX', '1.2 GB'],
  warning: 'small for Work',
  preselect: 'qwen3-0.6b',
  variants: [
    { id: 'qwen3-0.6b', label: 'BF16', state: 'loading', facts: ['MLX', '1.2 GB'], warning: 'small for Work' },
  ],
};
const BASES = [QWEN27, AGENTS, QWEN36, SMALL];

function picker(extra: Partial<ModelPickerProps> = {}) {
  return createElement(ModelPicker, {
    value: XL,
    bases: BASES,
    onValueChange: () => undefined,
    footer: 'Models load on first use.',
    ...extra,
  });
}

/** `n` bases with one variant each, all loaded. */
function many(n: number): ModelBaseOption[] {
  return Array.from({ length: n }, (_, index) => ({
    base: `model-${String(index)}`,
    state: 'loaded' as const,
    facts: [],
    preselect: `model-${String(index)}`,
    variants: [{ id: `model-${String(index)}`, label: 'BF16', state: 'loaded' as const, facts: [] }],
  }));
}

/** One base with `n` variants. */
function wide(n: number): ModelBaseOption {
  const variants = Array.from({ length: n }, (_, index) => ({
    id: `wide-q${String(index)}`,
    label: `Q${String(index)}`,
    state: 'loaded' as const,
    facts: [],
  }));
  return { base: 'wide', state: 'loaded', facts: [], preselect: variants[0]!.id, variants };
}

const ROW = 44;
const VIEW = 374;

/**
 * happy-dom does no layout: each option is ROW tall at its index, a listbox shows VIEW px, and scrollTop is stored.
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

function key(target: Element, value: string, extra: KeyboardEventInit = {}): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true, ...extra }));
  });
}

function click(target: HTMLElement): void {
  act(() => target.click());
}

function trigger(container: HTMLElement): HTMLButtonElement {
  return container.querySelector<HTMLButtonElement>('button[aria-haspopup="listbox"]')!;
}

/** Opens the popover; returns the base list and a reader of the quantization list (it is replaced per base). */
async function open(container: HTMLElement): Promise<{ left: HTMLElement; right: () => HTMLElement }> {
  key(trigger(container), 'ArrowDown');
  await waitFor(() => allByRole(document.body, 'listbox').length === 2);
  const [left] = allByRole(document.body, 'listbox');
  return { left: left!, right: () => allByRole(document.body, 'listbox')[1]! };
}

function active(list: HTMLElement): string {
  const id = list.getAttribute('aria-activedescendant');
  const row = id === null ? null : document.getElementById(id);
  return row?.querySelector('[title]')?.getAttribute('title') ?? '';
}

function labels(list: HTMLElement): string[] {
  return allByRole(list, 'option').map((row) => row.querySelector('[title]')?.getAttribute('title') ?? '');
}

describe('ModelPicker: the pill', () => {
  it('shows the base and the variant badge; its name says the state; the title is the full id', () => {
    const container = mount(picker());
    const button = trigger(container);
    expect(button.getAttribute('aria-label')).toBe(
      fill(CHAT.pillName, { model: 'Qwen3.8-27B UD-Q4_K_XL', state: CHAT.pillStates.loaded }),
    );
    expect(button.getAttribute('data-state')).toBe('loaded');
    expect(button.getAttribute('title')).toBe(XL);
    expect(button.querySelector('[data-model]')?.getAttribute('data-model')).toBe('Qwen3.8-27B');
    expect(button.querySelector('[data-chip-label]')?.textContent).toBe('UD-Q4_K_XL');
  });

  it('a one-variant base still carries its badge; a long variant is cut in the middle at 14 characters', () => {
    const long: ModelBaseOption = {
      ...AGENTS,
      variants: [{ ...AGENTS.variants[0]!, label: 'BF16 (lfm2.5-1.2b-thinking-mlx)' }],
    };
    const container = mount(picker({ value: 'agents-a1', bases: [long] }));
    const badge = trigger(container).querySelector('[data-chip-label]')!;
    expect(badge.textContent!.length).toBe(14);
    expect(badge.textContent!.startsWith('BF16')).toBe(true);
    expect(badge.textContent!.endsWith('-mlx)')).toBe(true);
  });

  it('crowded (Work): the badge is whole on the pill line or not drawn, never cut; elsewhere it hides only under 520 px', () => {
    const rules = cssRules('components/arc-x/model-picker/model-picker.module.css');
    expect(rules.some((rule) => rule.selectors.includes('.trigger[data-crowded] .variant'))).toBe(false);
    expect(
      ['flex-wrap', 'overflow', 'align-self'].map((property) =>
        cssValue(rules, '.trigger[data-crowded] .name', property),
      ),
    ).toEqual(['wrap', 'clip', 'stretch']);
    expect(cssValue(rules, '.trigger[data-crowded] .name::before', 'height')).toBe('var(--control-height-sm)');
    const crowded = trigger(mount(picker({ crowded: true })));
    expect(crowded.hasAttribute('data-crowded')).toBe(true);
    expect(crowded.querySelector('[data-chip-label]')?.hasAttribute('data-fit-whole')).toBe(true);
    const chat = trigger(mount(picker()));
    expect(chat.hasAttribute('data-crowded')).toBe(false);
    expect(chat.querySelector('[data-chip-label]')?.hasAttribute('data-fit-whole')).toBe(false);
  });

  it('a model that is not served: the stored id, no badge, no state', () => {
    const old = 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004';
    const container = mount(picker({ value: old }));
    const button = trigger(container);
    expect(button.getAttribute('aria-label')).toBe(`${CHAT.model}: ${old}`);
    expect(button.getAttribute('data-state')).toBeNull();
    expect(button.querySelector('[data-chip-label]')).toBeNull();
    expect(button.querySelector('[data-model]')?.getAttribute('data-model')).toBe(old);
  });

  it('no model: a disabled trigger that says so; disabled with a reason stays off', () => {
    const none = mount(picker({ value: null, bases: [] }));
    expect((byRole(none, 'button', CHAT.noModel) as HTMLButtonElement).disabled).toBe(true);
    unmountAll();
    const busy = mount(picker({ disabled: true, disabledReason: 'Wait for the answer to finish.' }));
    expect(trigger(busy).disabled).toBe(true);
  });
});

describe('ModelPicker: the popover', () => {
  it('opens on the base of the model, focus in the base list; both checks; the right list is that base', async () => {
    const container = mount(picker());
    const { left, right } = await open(container);
    expect(document.activeElement).toBe(left);
    expect(active(left)).toBe('Qwen3.8-27B');
    expect(labels(left)).toEqual(['Qwen3.8-27B', 'agents-a1', 'Qwen3.6-35B-A3B', 'qwen3-0.6b']);
    const selected = allByRole(left, 'option').filter((row) => row.getAttribute('aria-selected') === 'true');
    expect(selected.map((row) => row.querySelector('[title]')!.getAttribute('title'))).toEqual(['Qwen3.8-27B']);
    expect(left.getAttribute('aria-controls')).toBe(right().id);
    expect(right().getAttribute('aria-label')).toBe(fill(CHAT.modelQuantList, { base: 'Qwen3.8-27B' }));
    expect(labels(right())).toEqual(['UD-Q4_K_XL', 'UD-Q4_K_M', 'BF16']);
    const checked = allByRole(right(), 'option').filter((row) => row.getAttribute('aria-selected') === 'true');
    expect(checked).toHaveLength(1);
    expect(checked[0]!.textContent).toContain('UD-Q4_K_XL');
    expect(document.body.textContent).toContain(CHAT.modelQuant);
  });

  it('line 2: the state part, the count before the context; an idle base has no state word', async () => {
    const loading: ModelBaseOption = { ...QWEN27, state: 'loading', stateVariant: 'UD-Q4_K_M' };
    const container = mount(picker({ bases: [loading, AGENTS, SMALL] }));
    const { left, right } = await open(container);
    const rows = allByRole(left, 'option');
    expect(rows[0]!.textContent).toContain('Loading… · UD-Q4_K_M · 3 quantizations · 256K context');
    expect(rows[1]!.textContent).toBe('agents-a1256K context · MLX · 70.2 GBBF16');
    expect(rows[1]!.textContent).not.toContain(CHAT.modelLoaded);
    expect(rows[2]!.textContent).toContain('small for Work');
    expect(allByRole(right(), 'option')[0]!.textContent).toContain('Loaded · GGUF · 17.6 GB');
  });

  it('→ then Enter lands on the conversation model: nothing is sent and the popover closes', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ onValueChange }));
    const { left, right } = await open(container);
    key(left, 'ArrowRight');
    await waitFor(() => document.activeElement === right());
    expect(active(right())).toBe('UD-Q4_K_XL');
    key(right(), 'Enter');
    await waitFor(() => allByRole(document.body, 'listbox').length === 0);
    expect(onValueChange).not.toHaveBeenCalled();
  });

  it('a pointer over other bases moves the active row but never the right list', async () => {
    const container = mount(picker());
    const { left, right } = await open(container);
    const rows = allByRole(left, 'option');
    pointerMove(rows[2]!, 10, 100);
    pointerMove(rows[3]!, 10, 150);
    expect(active(left)).toBe('qwen3-0.6b');
    expect(labels(right())).toEqual(['UD-Q4_K_XL', 'UD-Q4_K_M', 'BF16']);
  });

  it('a click on a base with 2+ opens it and picks nothing; a click on a quantization picks once and closes', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ onValueChange }));
    const { left, right } = await open(container);
    click(allByRole(left, 'option')[2]!);
    await waitFor(() => labels(right())[0] === 'UD-IQ2_M');
    expect(onValueChange).not.toHaveBeenCalled();
    expect(allByRole(left, 'option')[2]!.hasAttribute('data-open')).toBe(true);
    expect(allByRole(left, 'option')[0]!.hasAttribute('data-open')).toBe(false);
    await waitFor(() => document.activeElement === right());
    expect(active(right())).toBe('UD-Q4_K_XL');
    click(allByRole(right(), 'option')[2]!);
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith('qwen3.6-35b-a3b-nvfp4');
    await waitFor(() => allByRole(document.body, 'listbox').length === 0);
  });

  it('a click on a base with one quantization picks it', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ onValueChange }));
    const { left } = await open(container);
    click(allByRole(left, 'option')[1]!);
    expect(onValueChange).toHaveBeenCalledWith('agents-a1');
    await waitFor(() => allByRole(document.body, 'listbox').length === 0);
  });

  it('keys: arrows and letters move the base row; Enter opens a base with 2+ and picks a base with 1', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ onValueChange }));
    const { left, right } = await open(container);
    key(left, 'ArrowDown');
    expect(active(left)).toBe('agents-a1');
    expect(labels(right())[0]).toBe('UD-Q4_K_XL');
    key(left, 'q');
    expect(active(left)).toBe('Qwen3.6-35B-A3B');
    key(left, 'Q');
    expect(active(left)).toBe('qwen3-0.6b');
    key(left, 'q');
    expect(active(left)).toBe('Qwen3.8-27B');
    key(left, 'End');
    expect(active(left)).toBe('qwen3-0.6b');
    key(left, 'Home');
    expect(active(left)).toBe('Qwen3.8-27B');
    key(left, 'ArrowDown');
    key(left, 'ArrowDown');
    key(left, 'Enter');
    await waitFor(() => document.activeElement === right());
    expect(labels(right())[0]).toBe('UD-IQ2_M');
    expect(active(right())).toBe('UD-Q4_K_XL');
    key(right(), 'ArrowDown');
    expect(active(right())).toBe('NVFP4');
    key(right(), 'Home');
    expect(active(right())).toBe('UD-IQ2_M');
    key(right(), 'End');
    expect(active(right())).toBe('BF16');
    key(right(), 'u');
    expect(active(right())).toBe('UD-IQ2_M');
    key(right(), 'ArrowLeft');
    expect(document.activeElement).toBe(left);
    expect(active(left)).toBe('Qwen3.6-35B-A3B');
    key(left, 'ArrowDown');
    key(left, ' ');
    expect(onValueChange).toHaveBeenCalledWith('qwen3-0.6b');
  });

  it('Shift+Tab from the quantizations makes the open base active in the base list', async () => {
    const container = mount(picker());
    const { left, right } = await open(container);
    key(left, 'ArrowDown');
    key(left, 'ArrowDown');
    key(left, 'ArrowRight');
    await waitFor(() => document.activeElement === right());
    key(left, 'ArrowUp');
    expect(active(left)).toBe('agents-a1');
    key(right(), 'Tab', { shiftKey: true });
    expect(active(left)).toBe('Qwen3.6-35B-A3B');
  });

  it('new bases in another order (a load ended) keep the open base and both active rows on the same keys', async () => {
    const container = mount(picker());
    const { left, right } = await open(container);
    click(allByRole(left, 'option')[2]!);
    await waitFor(() => document.activeElement === right());
    key(right(), 'ArrowDown');
    expect(active(right())).toBe('NVFP4');
    key(left, 'End');
    expect(active(left)).toBe('qwen3-0.6b');
    const list = right();
    const resorted: ModelBaseOption[] = [
      { ...QWEN36, variants: [QWEN36.variants[3]!, ...QWEN36.variants.slice(0, 3)] },
      SMALL,
      AGENTS,
      QWEN27,
    ];
    rerender(picker({ bases: resorted }));
    expect(labels(left)).toEqual(['Qwen3.6-35B-A3B', 'qwen3-0.6b', 'agents-a1', 'Qwen3.8-27B']);
    expect(active(left)).toBe('qwen3-0.6b');
    expect(allByRole(left, 'option')[0]!.hasAttribute('data-open')).toBe(true);
    expect(labels(right())[0]).toBe('BF16');
    expect(active(right())).toBe('NVFP4');
    expect(right()).toBe(list);
    expect(list.className).toContain('_swap_');
    expect(allByRole(list, 'option').filter((row) => row.className.includes('_swap_'))).toEqual([]);
  });

  it('a conversation model that is not served: the fallback base opens, no check in either list', async () => {
    const container = mount(picker({ value: 'gone-model', fallback: 'qwen3.6-35b-a3b-nvfp4' }));
    const { left, right } = await open(container);
    expect(active(left)).toBe('Qwen3.6-35B-A3B');
    expect(labels(right())[0]).toBe('UD-IQ2_M');
    expect(document.querySelectorAll('[role="option"][aria-selected="true"]')).toHaveLength(0);
  });

  it('the footer: the swap warning while a variant is not loaded, else the footer; the memory row comes last', async () => {
    const onUnload = vi.fn();
    const action = createElement('button', { type: 'button', onClick: onUnload }, 'Unload');
    const container = mount(
      picker({ swapWarning: 'A running answer waits.', memory: { text: `${XL} is in memory.`, action } }),
    );
    const { left } = await open(container);
    expect(document.body.textContent).toContain('A running answer waits.');
    expect(document.body.textContent).not.toContain('Models load on first use.');
    const panel = left.parentElement!.parentElement!.parentElement!;
    const memory = panel.lastElementChild!;
    expect(memory.textContent).toBe(`${XL} is in memory.Unload`);
    expect(left.contains(memory)).toBe(false);
    act(() => memory.querySelector('button')!.click());
    expect(onUnload).toHaveBeenCalledTimes(1);
    unmountAll();
    const all = mount(picker({ value: 'model-0', bases: many(3), swapWarning: 'A running answer waits.' }));
    await open(all);
    expect(document.body.textContent).toContain('Models load on first use.');
    expect(document.body.textContent).not.toContain('A running answer waits.');
  });

  it('structure: the left list sets the height, the right list is a scroller inside its column', async () => {
    const rules = cssRules('components/arc-x/model-picker/model-picker.module.css');
    expect(cssValue(rules, '.panel.panel', 'max-height')).toBe('var(--radix-popover-content-available-height, 100vh)');
    expect(cssValue(rules, '.panel.panel', 'flex-direction')).toBe('column');
    expect(cssValue(rules, '.columns', 'flex')).toBe(
      '0 1 calc(var(--control-height-sm) + var(--picker-row) * var(--picker-rows))',
    );
    expect(cssValue(rules, '.columns', 'min-height')).toBe('0');
    expect(cssValue(rules, '.column', 'min-height')).toBe('0');
    expect(cssValue(rules, '.list', 'max-height')).toBeUndefined();
    expect(cssValue(rules, '.list', 'overflow-y')).toBe('auto');
    expect(cssValue(rules, '.list', 'overscroll-behavior')).toBe('contain');
    expect(cssValue(rules, '.body', 'position')).toBe('relative');
    expect(cssValue(rules, '.list.inner', 'position')).toBe('absolute');
    expect(cssValue(rules, '.list.inner', 'inset')).toBe('0');
    expect(cssValue(rules, '.row', 'min-height')).toBe('var(--picker-row)');
    expect(cssValue(rules, '.heading', 'flex')).toBe('none');
    expect(cssValue(rules, '.footer', 'flex')).toBe('none');
    expect(cssValue(rules, '.memory', 'flex')).toBe('none');
    expect(rules.some((rule) => rule.reduce && rule.selectors.includes('.swap'))).toBe(true);
    expect(cssValue(rules, '.warning > svg', 'display')).toBe('inline-block');
    for (const [n, rows] of [
      [2, '4'],
      [6, '6'],
      [40, '8.5'],
    ] as const) {
      const container = mount(picker({ value: 'model-0', bases: many(n) }));
      const { left, right } = await open(container);
      const columns = left.parentElement!.parentElement!;
      expect(columns.style.getPropertyValue('--picker-rows')).toBe(rows);
      expect(right().className).toContain('_inner_');
      expect(right().parentElement!.className).toContain('_body_');
      unmountAll();
    }
  });

  describe('a long list', () => {
    let restore: (() => void) | undefined;
    afterEach(() => {
      restore?.();
      restore = undefined;
    });

    it('opens with the selected row in view, in both lists', async () => {
      restore = stubLayout();
      const bases = [...many(40), { ...wide(30), preselect: 'wide-q25' }];
      const container = mount(picker({ bases, value: 'wide-q25' }));
      const { left, right } = await open(container);
      expect(active(left)).toBe('wide');
      await waitFor(() => left.scrollTop === 41 * ROW - VIEW);
      expect(active(right())).toBe('Q25');
      await waitFor(() => right().scrollTop === 26 * ROW - VIEW);
    });

    it('ArrowDown, ArrowUp, Home and End keep the active row in view', async () => {
      restore = stubLayout();
      const container = mount(picker({ bases: [wide(40), ...many(40)], value: 'wide-q0' }));
      const { left, right } = await open(container);
      for (const list of [left, right()]) {
        expect(list.scrollTop).toBe(0);
        for (let step = 0; step < 9; step += 1) key(list, 'ArrowDown');
        expect(list.scrollTop).toBe(10 * ROW - VIEW);
        key(list, 'End');
        expect(list.scrollTop).toBe(41 * ROW - VIEW - (list === left ? 0 : ROW));
        for (let step = 0; step < 10; step += 1) key(list, 'ArrowUp');
        expect(list.scrollTop).toBe((list === left ? 30 : 29) * ROW);
        key(list, 'Home');
        expect(list.scrollTop).toBe(0);
      }
    });

    it('a pointer that rests while a list scrolls by key keeps the key row; a real move takes it', async () => {
      const container = mount(picker({ bases: [wide(40), ...many(40)], value: 'wide-q0' }));
      const { left, right } = await open(container);
      for (const list of [left, right()]) {
        const rows = allByRole(list, 'option');
        pointerMove(rows[5]!, 40, 200);
        expect(list.getAttribute('aria-activedescendant')).toBe(rows[5]!.id);
        key(list, 'ArrowDown');
        expect(list.getAttribute('aria-activedescendant')).toBe(rows[6]!.id);
        pointerMove(rows[3]!, 40, 200);
        expect(list.getAttribute('aria-activedescendant')).toBe(rows[6]!.id);
        pointerMove(rows[3]!, 40, 204);
        expect(list.getAttribute('aria-activedescendant')).toBe(rows[3]!.id);
      }
    });
  });
});
