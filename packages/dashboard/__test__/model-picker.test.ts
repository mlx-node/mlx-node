/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  type ModelBaseOption,
  type ModelEffort,
  ModelPicker,
  type ModelPickerProps,
} from '../ui/src/components/arc-x/model-picker/model-picker.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { flush, mount, rerender, unmountAll } from './helpers/mount.js';
import { byRole, waitFor } from './render.js';

afterEach(unmountAll);

const XL = 'Qwen3.8-27B-UD-Q4_K_XL';
const M = 'Qwen3.8-27B-UD-Q4_K_M';
const BF16 = 'qwen3.8-27b-bf16';

const QWEN27: ModelBaseOption = {
  base: 'Qwen3.8-27B',
  note: 'Strong coding and reasoning for everyday work',
  recommended: true,
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
  note: 'Long-horizon agent work and tool loops',
  recommended: true,
  state: 'idle',
  facts: ['256K context', 'MLX', '70.2 GB'],
  preselect: 'agents-a1',
  variants: [{ id: 'agents-a1', label: 'BF16', state: 'idle', facts: ['MLX', '70.2 GB'] }],
};

const GEMMA: ModelBaseOption = {
  base: 'gemma-4-26b',
  state: 'idle',
  facts: ['2 quantizations', '128K context'],
  preselect: 'gemma-4-26b-q4',
  variants: [
    { id: 'gemma-4-26b-q4', label: 'Q4_K_M', state: 'idle', facts: ['GGUF', '14.2 GB'] },
    { id: 'gemma-4-26b-bf16', label: 'BF16', state: 'idle', facts: ['MLX', '52.0 GB'] },
  ],
};

const LIQUID: ModelBaseOption = {
  base: 'lfm2.5-1.2b',
  state: 'loading',
  facts: ['16K context', 'MLX', '1.2 GB'],
  warning: 'small for Work',
  preselect: 'lfm2.5-1.2b',
  variants: [
    { id: 'lfm2.5-1.2b', label: 'BF16', state: 'loading', facts: ['MLX', '1.2 GB'], warning: 'small for Work' },
  ],
};

const CUSTOM: ModelBaseOption = {
  base: 'acme-7b',
  state: 'idle',
  facts: ['32K context', 'MLX', '4.1 GB'],
  preselect: 'acme-7b',
  variants: [{ id: 'acme-7b', label: 'MLX', state: 'idle', facts: ['MLX', '4.1 GB'] }],
};

const BASES = [QWEN27, AGENTS, GEMMA, LIQUID, CUSTOM];

const LEVELS = (['none', 'low', 'medium', 'high'] as const).map((value) => ({
  value,
  label: CHAT.thinkingLevel.levels[value],
  note: CHAT.thinkingLevel.levelNotes[value],
}));

/** BASES with levels on the conversation's model (XL). */
function levelled(extra: Partial<ModelEffort> = {}): ModelBaseOption[] {
  const effort: ModelEffort = {
    levels: LEVELS,
    value: 'high',
    defaultValue: 'medium',
    note: CHAT.thinkingLevel.about,
    ...extra,
  };
  return [
    { ...QWEN27, variants: [{ ...QWEN27.variants[0]!, effort }, ...QWEN27.variants.slice(1)] },
    AGENTS,
    GEMMA,
    LIQUID,
    CUSTOM,
  ];
}

function picker(extra: Partial<ModelPickerProps> = {}) {
  return createElement(ModelPicker, {
    value: XL,
    bases: BASES,
    onValueChange: () => undefined,
    footer: 'Models load on first use.',
    ...extra,
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

function pointerDown(target: Element, pointerType = 'mouse'): void {
  act(() => {
    target.dispatchEvent(new PointerEvent('pointerdown', { button: 0, pointerType, bubbles: true, cancelable: true }));
  });
}

function trigger(container: HTMLElement): HTMLButtonElement {
  return container.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!;
}

async function openMenu(container: HTMLElement): Promise<HTMLElement> {
  key(trigger(container), 'ArrowDown');
  await waitFor(() => byRole(document.body, 'menu', CHAT.model) !== null);
  return byRole(document.body, 'menu', CHAT.model)!;
}

function menuItems(menu: HTMLElement): HTMLElement[] {
  return [...menu.querySelectorAll<HTMLElement>('[role="menuitem"], [role="menuitemradio"]')];
}

function findItem(menu: HTMLElement, text: string): HTMLElement {
  const hit = menuItems(menu).find((item) => item.textContent?.includes(text));
  if (hit === undefined) throw new Error(`No menu item containing "${text}" in "${menu.textContent}"`);
  return hit;
}

/** Stub window.matchMedia to match (or not match) the narrow viewport query. Returns a restore function. */
function stubNarrow(matches: boolean): () => void {
  const original = window.matchMedia;
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    writable: true,
    value: (query: string) => ({
      matches,
      media: query,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
      onchange: null,
    }),
  });
  return () => {
    Object.defineProperty(window, 'matchMedia', { configurable: true, writable: true, value: original });
  };
}

describe('ModelPicker: the pill', () => {
  it('shows the base name, state, title, aria-haspopup="menu", and [data-model]', () => {
    const container = mount(picker());
    const button = trigger(container);
    expect(button.getAttribute('aria-haspopup')).toBe('menu');
    expect(button.getAttribute('aria-label')).toBe(
      fill(CHAT.pillName, { model: 'Qwen3.8-27B UD-Q4_K_XL', state: CHAT.pillStates.loaded }),
    );
    expect(button.getAttribute('data-state')).toBe('loaded');
    expect(button.getAttribute('title')).toBe(XL);
    expect(button.querySelector('[data-model]')?.getAttribute('data-model')).toBe(XL);
    expect(button.querySelector('[data-model]')?.textContent).toBe('Qwen3.8-27B');
    expect(button.querySelector('[data-chip-label]')).toBeNull();
  });

  it('shows effort label, warning dot, crowded attributes, and working spinner', () => {
    const crowded = trigger(
      mount(picker({ crowded: true, levelWarning: true, working: true, bases: levelled({ value: 'low' }) })),
    );
    expect(crowded.hasAttribute('data-crowded')).toBe(true);
    const chip = crowded.querySelector('[data-chip-label][data-chip-level]')!;
    expect(chip).not.toBeNull();
    expect(chip.textContent).toBe('Low');
    expect(chip.hasAttribute('data-fit-whole')).toBe(true);
    expect(chip.querySelector('span[aria-hidden="true"]')).not.toBeNull();
    expect(crowded.getAttribute('title')).toBe(fill(CHAT.thinkingLevel.chip, { model: XL, level: 'Low' }));
    expect(crowded.getAttribute('aria-label')).toBe(
      fill(CHAT.thinkingLevel.chipName, {
        model: fill(CHAT.pillName, { model: 'Qwen3.8-27B UD-Q4_K_XL', state: CHAT.pillStates.loaded }),
        level: 'Low',
      }),
    );
    // When working, MorphLoader replaces the ChevronDown SVG.
    expect(crowded.querySelector('[role="status"], svg')).not.toBeNull();

    unmountAll();
    const chat = trigger(mount(picker({ bases: levelled({ value: 'high' }) })));
    expect(chat.hasAttribute('data-crowded')).toBe(false);
    expect(chat.querySelector('[data-chip-label]')?.hasAttribute('data-fit-whole')).toBe(false);
    expect(chat.querySelector('[data-chip-level] span[aria-hidden="true"]')).toBeNull();
  });

  it('a model that is not served: shows the stored id, no level badge, no state', () => {
    const old = 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004';
    const container = mount(picker({ value: old }));
    const button = trigger(container);
    expect(button.getAttribute('aria-label')).toBe(`${CHAT.model}: ${old}`);
    expect(button.getAttribute('data-state')).toBeNull();
    expect(button.querySelector('[data-chip-label]')).toBeNull();
    expect(button.querySelector('[data-model]')?.getAttribute('data-model')).toBe(old);
    expect(button.querySelector('[data-model]')?.textContent).toBe(old);
  });

  it('no model: disabled trigger saying No model; disabled with a reason wraps in Tooltip', () => {
    const none = mount(picker({ value: null, bases: [] }));
    expect((byRole(none, 'button', CHAT.noModel) as HTMLButtonElement).disabled).toBe(true);

    unmountAll();
    const busy = mount(picker({ disabled: true, disabledReason: 'Wait for the answer to finish.' }));
    expect(trigger(busy).disabled).toBe(true);
    expect(trigger(busy).parentElement?.tagName.toLowerCase()).toBe('span');
  });

  it('programmatic trigger.click() toggles the menu open and closed', async () => {
    const container = mount(picker());
    const button = trigger(container);
    click(button);
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) !== null);
    expect(byRole(document.body, 'menu', CHAT.model)).not.toBeNull();

    click(button);
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) === null);
    expect(byRole(document.body, 'menu', CHAT.model)).toBeNull();
  });

  it('a real pointer click sequence (pointerdown + click) opens once without double-toggling closed', async () => {
    const container = mount(picker());
    const button = trigger(container);
    pointerDown(button, 'mouse');
    act(() => {
      window.dispatchEvent(new PointerEvent('pointerup', { button: 0, pointerType: 'mouse', bubbles: true }));
    });
    click(button);
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) !== null);
    expect(byRole(document.body, 'menu', CHAT.model)).not.toBeNull();

    // Subsequent programmatic click still closes the open menu.
    click(button);
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) === null);

    // Dragging away before releasing clears the flag on the next tick so a later programmatic click opens it.
    pointerDown(button, 'mouse');
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) !== null);
    act(() => {
      window.dispatchEvent(new PointerEvent('pointerup', { button: 0, pointerType: 'mouse', bubbles: true }));
    });
    await act(async () => new Promise<void>((resolve) => setTimeout(resolve, 0)));
    click(button);
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) === null);
  });

  it('forwards side and align placement to the menu content', async () => {
    const container = mount(picker({ side: 'bottom', align: 'end' }));
    const menu = await openMenu(container);
    expect(menu.getAttribute('data-side')).toBe('bottom');
    expect(menu.getAttribute('data-align')).toBe('end');
  });

  it('toggling disabled false -> true -> false while open closes the menu and does not spontaneously re-open it', async () => {
    const container = mount(picker({ disabled: false }));
    await openMenu(container);
    expect(byRole(document.body, 'menu', CHAT.model)).not.toBeNull();

    rerender(picker({ disabled: true }));
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) === null);
    expect(byRole(document.body, 'menu', CHAT.model)).toBeNull();

    rerender(picker({ disabled: false }));
    await flush();
    expect(byRole(document.body, 'menu', CHAT.model)).toBeNull();
  });

  it('delayed touch click (detail >= 1) after pointerToggled reset does not double-toggle the trigger', async () => {
    const container = mount(picker());
    const button = trigger(container);
    pointerDown(button, 'touch');
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) !== null);
    act(() => {
      window.dispatchEvent(new PointerEvent('pointerup', { button: 0, pointerType: 'touch', bubbles: true }));
    });
    await act(async () => new Promise<void>((resolve) => setTimeout(resolve, 0)));

    act(() => {
      button.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 1 }));
    });
    await flush();
    expect(byRole(document.body, 'menu', CHAT.model)).not.toBeNull();
  });

  it('crowded trigger keeps flexible width with no .measure lock, and non-crowded .measure shares [data-chip-label]', () => {
    const crowdedContainer = mount(picker({ crowded: true, bases: levelled({ value: 'high' }) }));
    const crowdedButton = trigger(crowdedContainer);
    const effortChip = crowdedButton.querySelector<HTMLElement>('[data-chip-label][data-fit-whole]')!;
    expect(effortChip).not.toBeNull();
    expect(effortChip.hasAttribute('data-chip-level')).toBe(true);
    expect(effortChip.textContent).toBe('High');
    // Work mode keeps .triggerText flexible (no .measure locking an inline pixel width over data-dropped).
    expect(crowdedButton.querySelector('[data-key]')).toBeNull();
    const triggerText = effortChip.parentElement?.parentElement as HTMLElement | null;
    expect(triggerText).not.toBeNull();
    expect(triggerText?.style.width === '' || triggerText?.style.width === 'auto').toBe(true);

    unmountAll();
    const chatContainer = mount(picker({ crowded: false, bases: levelled({ value: 'high' }) }));
    const chatButton = trigger(chatContainer);
    const firstChip = chatButton.querySelector<HTMLElement>('[data-chip-label]')!;
    expect(firstChip).not.toBeNull();
    expect(firstChip.hasAttribute('data-chip-level')).toBe(true);
    expect(firstChip.hasAttribute('aria-hidden')).toBe(false);
    const measure = chatButton.querySelector<HTMLElement>('[data-key]')!;
    expect(measure).not.toBeNull();
    const measureEffort = measure.querySelector<HTMLElement>('[data-chip-label]')!;
    expect(measureEffort).not.toBeNull();
    expect(measureEffort.hasAttribute('data-chip-level')).toBe(false);
    expect(measureEffort.hasAttribute('data-fit-whole')).toBe(false);
    expect(measureEffort.textContent).toBe('High');
    expect([...chatButton.querySelectorAll<HTMLElement>('[data-chip-label]')]).toEqual([firstChip, measureEffort]);
  });
});

describe('ModelPicker: recommended outside vs More models inside', () => {
  it('renders recommended models outside with blurbs and non-recommended under More models grouped by provider', async () => {
    const container = mount(picker());
    const menu = await openMenu(container);
    expect(menu.textContent).toContain('Qwen3.8-27B');
    expect(menu.textContent).toContain('Strong coding and reasoning for everyday work');
    expect(menu.textContent).toContain('agents-a1');
    expect(menu.textContent).toContain('Long-horizon agent work and tool loops');
    expect(menu.textContent).not.toContain('gemma-4-26b');
    expect(menu.textContent).not.toContain('lfm2.5-1.2b');
    expect(menu.textContent).toContain(CHAT.modelMore);

    // Open More models submenu via ArrowRight.
    const moreTrigger = findItem(menu, CHAT.modelMore);
    act(() => moreTrigger.focus());
    key(moreTrigger, 'ArrowRight');
    await waitFor(() => byRole(document.body, 'menu', CHAT.modelMoreList) !== null);
    const moreMenu = byRole(document.body, 'menu', CHAT.modelMoreList)!;
    expect(moreMenu.textContent).toContain('Gemma');
    expect(moreMenu.textContent).toContain('gemma-4-26b');
    expect(moreMenu.textContent).toContain('Liquid');
    expect(moreMenu.textContent).toContain('lfm2.5-1.2b');
    expect(moreMenu.textContent).toContain('small for Work');
    expect(moreMenu.textContent).toContain(CHAT.modelOtherProvider);
    expect(moreMenu.textContent).toContain('acme-7b');
  });

  it('falls back to showing all bases outside when none are marked recommended', async () => {
    const plain = BASES.map((base) => ({ ...base, recommended: false }));
    const container = mount(picker({ bases: plain }));
    const menu = await openMenu(container);
    expect(menu.textContent).toContain('Qwen3.8-27B');
    expect(menu.textContent).toContain('gemma-4-26b');
    expect(menu.textContent).toContain('lfm2.5-1.2b');
    expect(menu.textContent).not.toContain(CHAT.modelMore);
  });

  it('More models row shows the currently active non-recommended model name and respects explicit provider overrides', async () => {
    const overridden: ModelBaseOption = {
      ...CUSTOM,
      base: 'custom-reasoner',
      provider: 'Partner Lab',
      preselect: 'custom-reasoner',
      variants: [{ id: 'custom-reasoner', label: 'MLX', state: 'idle', facts: [] }],
    };
    const container = mount(picker({ value: 'lfm2.5-1.2b', bases: [QWEN27, LIQUID, overridden] }));
    const menu = await openMenu(container);
    const moreTrigger = findItem(menu, CHAT.modelMore);
    expect(moreTrigger.textContent).toContain('lfm2.5-1.2b');

    act(() => moreTrigger.focus());
    key(moreTrigger, 'ArrowRight');
    await waitFor(() => byRole(document.body, 'menu', CHAT.modelMoreList) !== null);
    const moreMenu = byRole(document.body, 'menu', CHAT.modelMoreList)!;
    expect(moreMenu.textContent).toContain('Partner Lab');
    expect(moreMenu.textContent).toContain('custom-reasoner');
  });
});

describe('ModelPicker: multi-variant quantization submenus', () => {
  it('ArrowRight on a multi-variant outside base opens its quantization submenu and picks a quantization', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ onValueChange }));
    const menu = await openMenu(container);
    const qwenRow = findItem(menu, 'Qwen3.8-27B');
    act(() => qwenRow.focus());
    key(qwenRow, 'ArrowRight');
    const quantLabel = fill(CHAT.modelQuantList, { base: 'Qwen3.8-27B' });
    await waitFor(() => byRole(document.body, 'menu', quantLabel) !== null);
    const quantMenu = byRole(document.body, 'menu', quantLabel)!;
    expect(quantMenu.textContent).toContain('UD-Q4_K_XL');
    expect(quantMenu.textContent).toContain('Loaded · GGUF · 17.6 GB');
    expect(quantMenu.textContent).toContain('UD-Q4_K_M');
    expect(quantMenu.textContent).toContain('BF16');

    click(findItem(quantMenu, 'UD-Q4_K_M'));
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith(M);
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) === null);
  });

  it('direct mouse click on a multi-variant base row commits its preselect immediately; touch tap opens the submenu instead', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ value: 'agents-a1', onValueChange }));
    const menu = await openMenu(container);
    const qwenRow = findItem(menu, 'Qwen3.8-27B');
    pointerDown(qwenRow, 'mouse');
    click(qwenRow);
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith(XL);
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) === null);

    unmountAll();
    const touchChange = vi.fn();
    const touchContainer = mount(picker({ value: 'agents-a1', onValueChange: touchChange }));
    const touchMenu = await openMenu(touchContainer);
    const touchRow = findItem(touchMenu, 'Qwen3.8-27B');
    pointerDown(touchRow, 'touch');
    click(touchRow);
    expect(touchChange).not.toHaveBeenCalled();
    const quantLabel = fill(CHAT.modelQuantList, { base: 'Qwen3.8-27B' });
    await waitFor(() => byRole(document.body, 'menu', quantLabel) !== null);
  });

  it('picking a model with effort levels sends both the model id and its effort value', async () => {
    const onValueChange = vi.fn();
    const bases = levelled({ value: 'medium' });
    const container = mount(picker({ value: 'agents-a1', bases, onValueChange }));
    const menu = await openMenu(container);
    const qwenRow = findItem(menu, 'Qwen3.8-27B');
    pointerDown(qwenRow, 'mouse');
    click(qwenRow);
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith(XL, 'medium');
  });

  it('picking a single-variant base commits it directly; picking the already-selected model does not fire onValueChange', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ onValueChange }));
    const menu = await openMenu(container);
    click(findItem(menu, 'agents-a1'));
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith('agents-a1');
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) === null);

    unmountAll();
    const sameChange = vi.fn();
    const sameContainer = mount(picker({ value: 'agents-a1', onValueChange: sameChange }));
    const sameMenu = await openMenu(sameContainer);
    click(findItem(sameMenu, 'agents-a1'));
    expect(sameChange).not.toHaveBeenCalled();
  });

  it('multi-variant base inside More models also opens its own quantization submenu', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ onValueChange }));
    const menu = await openMenu(container);
    const moreTrigger = findItem(menu, CHAT.modelMore);
    act(() => moreTrigger.focus());
    key(moreTrigger, 'ArrowRight');
    await waitFor(() => byRole(document.body, 'menu', CHAT.modelMoreList) !== null);
    const moreMenu = byRole(document.body, 'menu', CHAT.modelMoreList)!;

    const gemmaRow = findItem(moreMenu, 'gemma-4-26b');
    act(() => gemmaRow.focus());
    key(gemmaRow, 'ArrowRight');
    const quantLabel = fill(CHAT.modelQuantList, { base: 'gemma-4-26b' });
    await waitFor(() => byRole(document.body, 'menu', quantLabel) !== null);
    const quantMenu = byRole(document.body, 'menu', quantLabel)!;
    click(findItem(quantMenu, 'BF16'));
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith('gemma-4-26b-bf16');
  });

  it('VoiceOver/AT synthetic click without prior mouse pointerdown on a multi-variant SubTrigger does not commit preselect', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ value: 'agents-a1', onValueChange }));
    // Simulate prior mouse pointer activity elsewhere in the document.
    pointerDown(document.body, 'mouse');
    const menu = await openMenu(container);
    const qwenRow = findItem(menu, 'Qwen3.8-27B');

    // Synthetic AT click with no pointerdown on this row must not commit preselect; Radix opens Quantization.
    click(qwenRow);
    expect(onValueChange).not.toHaveBeenCalled();
    const quantLabel = fill(CHAT.modelQuantList, { base: 'Qwen3.8-27B' });
    await waitFor(() => byRole(document.body, 'menu', quantLabel) !== null);
  });

  it('fast mouse click after hover on a multi-variant SubTrigger closes cleanly without flashing the submenu open', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ value: 'agents-a1', onValueChange }));
    const menu = await openMenu(container);
    const qwenRow = findItem(menu, 'Qwen3.8-27B');
    act(() => {
      qwenRow.dispatchEvent(new PointerEvent('pointermove', { pointerType: 'mouse', bubbles: true }));
    });
    pointerDown(qwenRow, 'mouse');
    click(qwenRow);
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith(XL);
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) === null);
    await act(async () => new Promise<void>((resolve) => setTimeout(resolve, 140)));
    const quantLabel = fill(CHAT.modelQuantList, { base: 'Qwen3.8-27B' });
    expect(byRole(document.body, 'menu', quantLabel)).toBeNull();
  });
});

describe('ModelPicker: Effort submenu', () => {
  it('shows Effort row with current level, Default tag, level notes, and subNote; hidden when model has no levels', async () => {
    const onValueChange = vi.fn();
    const container = mount(picker({ bases: levelled(), onValueChange }));
    const menu = await openMenu(container);
    const effortTrigger = findItem(menu, CHAT.thinkingLevel.effort);
    expect(effortTrigger.textContent).toContain('High');

    act(() => effortTrigger.focus());
    key(effortTrigger, 'ArrowRight');
    const effortLabel = fill(CHAT.thinkingLevel.list, { model: 'Qwen3.8-27B' });
    await waitFor(() => byRole(document.body, 'menu', effortLabel) !== null);
    const effortMenu = byRole(document.body, 'menu', effortLabel)!;
    expect(effortMenu.textContent).toContain(CHAT.thinkingLevel.defaultTag);
    expect(effortMenu.textContent).toContain(CHAT.thinkingLevel.levelNotes.low);
    expect(effortMenu.textContent).toContain(CHAT.thinkingLevel.about);

    click(findItem(effortMenu, 'Low'));
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith(XL, 'low');

    unmountAll();
    const noLevels = mount(picker({ bases: BASES }));
    const plainMenu = await openMenu(noLevels);
    expect(plainMenu.textContent).not.toContain(CHAT.thinkingLevel.effort);
  });

  it('stays enabled when listDisabled is true while model rows and More models are disabled', async () => {
    const onValueChange = vi.fn();
    const container = mount(
      picker({
        onValueChange,
        listDisabled: true,
        listDisabledReason: CHAT.modelBusy,
        bases: levelled({ note: CHAT.thinkingLevel.next }),
      }),
    );
    expect(trigger(container).disabled).toBe(false);
    const menu = await openMenu(container);
    expect(menu.textContent).toContain(CHAT.modelBusy);
    expect(menu.textContent).not.toContain('Models load on first use.');

    // Model rows and More models are disabled.
    expect(findItem(menu, 'agents-a1').getAttribute('aria-disabled')).toBe('true');
    expect(findItem(menu, 'Qwen3.8-27B').getAttribute('aria-disabled')).toBe('true');
    expect(findItem(menu, CHAT.modelMore).getAttribute('aria-disabled')).toBe('true');
    click(findItem(menu, 'agents-a1'));
    expect(onValueChange).not.toHaveBeenCalled();

    // Effort stays enabled and commits the new level.
    const effortTrigger = findItem(menu, CHAT.thinkingLevel.effort);
    expect(effortTrigger.getAttribute('aria-disabled')).toBeNull();
    act(() => effortTrigger.focus());
    key(effortTrigger, 'ArrowRight');
    const effortLabel = fill(CHAT.thinkingLevel.list, { model: 'Qwen3.8-27B' });
    await waitFor(() => byRole(document.body, 'menu', effortLabel) !== null);
    const effortMenu = byRole(document.body, 'menu', effortLabel)!;
    expect(effortMenu.textContent).toContain(CHAT.thinkingLevel.next);
    click(findItem(effortMenu, 'Low'));
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith(XL, 'low');
  });
});

describe('ModelPicker: narrow viewport pages', () => {
  let restore: (() => void) | undefined;
  afterEach(() => {
    restore?.();
    restore = undefined;
  });

  it('opens Effort, More models, and Quantization as in-menu pages with Back and ArrowLeft/ArrowRight navigation', async () => {
    restore = stubNarrow(true);
    const onValueChange = vi.fn();
    const container = mount(picker({ bases: levelled(), onValueChange }));
    const menu = await openMenu(container);

    // Navigate to Effort page via ArrowRight, wait for focus to settle, then ArrowLeft back to main.
    const effortRow = findItem(menu, CHAT.thinkingLevel.effort);
    act(() => effortRow.focus());
    key(effortRow, 'ArrowRight');
    await waitFor(
      () =>
        menu.querySelector('[data-page="effort"]') !== null &&
        document.activeElement?.getAttribute('role') === 'menuitemradio',
    );
    expect(menu.textContent).toContain(CHAT.thinkingLevel.levelNotes.medium);
    key(document.activeElement!, 'ArrowLeft');
    await waitFor(
      () =>
        menu.querySelector('[data-page="main"]') !== null &&
        (document.activeElement as HTMLElement | null)?.dataset.row === 'effort',
    );

    // Navigate to More models -> gemma-4-26b quant page -> pick BF16.
    click(findItem(menu, CHAT.modelMore));
    await waitFor(
      () =>
        menu.querySelector('[data-page="more"]') !== null &&
        menu.contains(document.activeElement) &&
        document.activeElement !== menu,
    );
    click(findItem(menu, 'gemma-4-26b'));
    await waitFor(
      () =>
        menu.querySelector('[data-page="quant:gemma-4-26b"]') !== null &&
        document.activeElement?.getAttribute('role') === 'menuitemradio',
    );
    click(findItem(menu, 'BF16'));
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith('gemma-4-26b-bf16');
    await waitFor(() => byRole(document.body, 'menu', CHAT.model) === null);
  });

  it('focuses the selected multi-variant row inside narrow More models and falls back to main when listDisabled flips true', async () => {
    restore = stubNarrow(true);
    const props: Partial<ModelPickerProps> = {
      value: 'gemma-4-26b-q4',
      bases: levelled(),
      listDisabled: false,
      listDisabledReason: CHAT.modelBusy,
    };
    const container = mount(picker(props));
    const menu = await openMenu(container);

    click(findItem(menu, CHAT.modelMore));
    await waitFor(
      () =>
        menu.querySelector('[data-page="more"]') !== null &&
        (document.activeElement as HTMLElement | null)?.dataset.row === 'quant:gemma-4-26b',
    );
    const active = document.activeElement as HTMLElement;
    expect(active.hasAttribute('data-checked')).toBe(true);
    expect(active.hasAttribute('data-back')).toBe(false);

    rerender(picker({ ...props, listDisabled: true }));
    await waitFor(() => menu.querySelector('[data-page="main"]') !== null);
    expect(menu.textContent).toContain(CHAT.modelBusy);

    // ArrowRight on disabled More models row does nothing.
    const disabledMore = findItem(menu, CHAT.modelMore);
    key(disabledMore, 'ArrowRight');
    await flush();
    expect(menu.querySelector('[data-page="main"]')).not.toBeNull();
  });

  it('keeps the focus highlight aligned when a transforming .page wrapper becomes item.offsetParent', async () => {
    restore = stubNarrow(true);
    const container = mount(picker({ bases: levelled() }));
    const menu = await openMenu(container);
    const list = menu.querySelector<HTMLElement>('[data-page="main"]')!;
    const highlight = list.firstElementChild as HTMLElement;
    const pageEl = list.children[1]?.firstElementChild as HTMLElement;
    const effortRow = findItem(menu, CHAT.thinkingLevel.effort);

    Object.defineProperty(pageEl, 'offsetParent', { configurable: true, get: () => list });
    Object.defineProperty(pageEl, 'offsetTop', { configurable: true, get: () => 6 });
    Object.defineProperty(effortRow, 'offsetParent', { configurable: true, get: () => pageEl });
    Object.defineProperty(effortRow, 'offsetTop', { configurable: true, get: () => 120 });
    Object.defineProperty(effortRow, 'offsetHeight', { configurable: true, get: () => 40 });

    act(() => effortRow.focus());
    await waitFor(() => highlight.style.height === '40px');
    expect(highlight.style.height).toBe('40px');
    expect(highlight.style.transform).toContain('126px');
  });
});

describe('ModelPicker: footer, memory, and CSS structure', () => {
  it('applies footer precedence (listDisabledReason > swapWarning > footer) and keeps memory action clickable', async () => {
    const onUnload = vi.fn();
    const action = createElement('button', { type: 'button', onClick: onUnload }, 'Unload');
    const container = mount(
      picker({ swapWarning: 'A running answer waits.', memory: { text: `${XL} is in memory.`, action } }),
    );
    const menu = await openMenu(container);
    expect(menu.textContent).toContain('A running answer waits.');
    expect(menu.textContent).not.toContain('Models load on first use.');
    expect(menu.textContent).toContain(`${XL} is in memory.`);
    const unload = byRole(menu, 'button', 'Unload')!;
    expect(unload).not.toBeNull();
    click(unload);
    expect(onUnload).toHaveBeenCalledTimes(1);

    unmountAll();
    const allLoaded: ModelBaseOption[] = [
      {
        base: 'model-0',
        recommended: true,
        state: 'loaded',
        facts: [],
        preselect: 'model-0',
        variants: [{ id: 'model-0', label: 'BF16', state: 'loaded', facts: [] }],
      },
    ];
    const quiet = mount(picker({ value: 'model-0', bases: allLoaded, swapWarning: 'A running answer waits.' }));
    const quietMenu = await openMenu(quiet);
    expect(quietMenu.textContent).toContain('Models load on first use.');
    expect(quietMenu.textContent).not.toContain('A running answer waits.');
  });

  it('cycles Tab focus between menu items and Unload in .foot while shielding Space and letters from menu typeahead', async () => {
    const onUnload = vi.fn();
    const action = createElement('button', { type: 'button', onClick: onUnload }, 'Unload');
    const container = mount(picker({ memory: { text: `${XL} is in memory.`, action } }));
    const menu = await openMenu(container);
    const unload = byRole(menu, 'button', 'Unload')!;
    const qwenRow = findItem(menu, 'Qwen3.8-27B');

    act(() => qwenRow.focus());
    expect(document.activeElement).toBe(qwenRow);

    key(qwenRow, 'Tab');
    expect(document.activeElement).toBe(unload);

    // Letters and Space while focused on Unload must not trigger Radix menu typeahead and steal focus.
    key(unload, 'a');
    key(unload, ' ');
    expect(document.activeElement).toBe(unload);

    // Tab cycles back to the menu items.
    key(unload, 'Tab');
    expect(document.activeElement).not.toBe(unload);
    expect(document.activeElement?.getAttribute('role')?.startsWith('menuitem')).toBe(true);
  });

  it('skips disabled tabIndex={0} buttons in .foot on Tab while still reaching enabled controls', async () => {
    const busyUnload = createElement('button', { type: 'button', tabIndex: 0, disabled: true }, 'Unload');
    const container = mount(picker({ memory: { text: `${XL} is in memory.`, action: busyUnload } }));
    const menu = await openMenu(container);
    const unload = byRole(menu, 'button', 'Unload') as HTMLButtonElement;
    expect(unload.disabled).toBe(true);
    expect(unload.getAttribute('tabindex')).toBe('0');
    const unloadFocus = vi.spyOn(unload, 'focus');
    const qwenRow = findItem(menu, 'Qwen3.8-27B');

    act(() => qwenRow.focus());
    key(qwenRow, 'Tab');
    expect(unloadFocus).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(qwenRow);
    expect(document.activeElement).not.toBe(unload);

    unmountAll();
    const mixed = createElement(
      'span',
      null,
      createElement('button', { type: 'button', tabIndex: 0, disabled: true }, 'Unload'),
      createElement('button', { type: 'button', tabIndex: 0 }, 'Details'),
    );
    const mixedContainer = mount(picker({ memory: { text: `${XL} is in memory.`, action: mixed } }));
    const mixedMenu = await openMenu(mixedContainer);
    const disabledBtn = byRole(mixedMenu, 'button', 'Unload') as HTMLButtonElement;
    const enabledBtn = byRole(mixedMenu, 'button', 'Details') as HTMLButtonElement;
    const disabledFocus = vi.spyOn(disabledBtn, 'focus');
    const row = findItem(mixedMenu, 'Qwen3.8-27B');

    act(() => row.focus());
    key(row, 'Tab');
    expect(disabledFocus).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(enabledBtn);

    key(enabledBtn, 'Tab');
    expect(document.activeElement?.getAttribute('role')?.startsWith('menuitem')).toBe(true);
  });

  it('enforces overflow: clip layout-shift guards on trigger elements and Arc menu surface rules', () => {
    const rules = cssRules('components/arc-x/model-picker/model-picker.module.css');
    expect(cssValue(rules, '.trigger', 'overflow')).toBe('clip');
    expect(cssValue(rules, '.triggerMark', 'overflow')).toBe('clip');
    expect(cssValue(rules, '.triggerText', 'overflow')).toBe('clip');
    expect(cssValue(rules, '.menu', 'width')).toBe('304px');
    expect(cssValue(rules, '.effortMenu', 'width')).toBe('264px');
    expect(cssValue(rules, '.moreMenu', 'width')).toBe('272px');
    expect(cssValue(rules, '.quantMenu', 'width')).toBe('260px');
    expect(cssValue(rules, '.autoHeight', 'display')).toBe('flex');
    expect(cssValue(rules, '.autoHeight', 'flex-direction')).toBe('column');
    expect(cssValue(rules, '.list', 'min-height')).toBe('0');
    expect(cssValue(rules, '.list', 'overflow-x')).toBe('hidden');
    expect(cssValue(rules, '.list', 'overscroll-behavior')).toBe('contain');
    expect(cssValue(rules, '.item', 'min-height')).toBe('54px');
    expect(cssValue(rules, '.effortItem', 'min-height')).toBe('50px');
    expect(cssValue(rules, '.row', 'min-height')).toBe('40px');
    expect(cssValue(rules, '.compact', 'min-height')).toBe('38px');
    expect(rules.some((rule) => rule.reduce && rule.selectors.includes(".menu[data-state='closed']"))).toBe(true);
  });
});
