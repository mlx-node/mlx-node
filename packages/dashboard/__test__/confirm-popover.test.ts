/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { ConfirmDialog } from '../ui/src/components/arc-x/confirm-dialog/confirm-dialog.js';
import {
  ConfirmPopover,
  type ConfirmContent,
  type ConfirmPopoverProps,
  fitAlign,
} from '../ui/src/components/arc-x/confirm-popover/confirm-popover.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { byRole, skipAnimations, waitFor } from './render.js';

let root: Root | null = null;

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root?.render(element));
  return container;
}

/** The last dialog in the document: an older one may still be leaving. */
function dialog(): HTMLElement | null {
  const all = document.querySelectorAll<HTMLElement>('[role="dialog"]');
  return all.length === 0 ? null : all[all.length - 1];
}

function deferred(): { promise: Promise<void>; resolve(): void; reject(error: Error): void } {
  let resolve: () => void = () => {};
  let reject: (error: Error) => void = () => {};
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function flush(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  });
}

function pressEscape(): void {
  const target = document.activeElement ?? document.body;
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
  });
}

function click(element: Element | null): void {
  if (element === null) throw new Error('click: no element');
  act(() => {
    (element as HTMLElement).click();
  });
}

function content(overrides: Partial<ConfirmContent> = {}): ConfirmContent {
  return {
    title: 'Delete qwen3-8b?',
    description: 'Frees 5.2 GB. The files are removed from disk.',
    confirmLabel: 'Delete',
    workingLabel: 'Deleting…',
    successLabel: 'Deleted',
    onConfirm: () => Promise.resolve(),
    ...overrides,
  };
}

function popover(overrides: Partial<ConfirmPopoverProps> = {}): ReactElement {
  return createElement(ConfirmPopover, {
    ...content(),
    trigger: { kind: 'button', label: 'Delete model' },
    ...overrides,
  });
}

function trigger(container: HTMLElement): HTMLElement {
  const button = byRole(container, 'button', 'Delete model');
  if (button === null) throw new Error('no trigger');
  return button;
}

/** Every value `data-phase` takes on the anchor from now on, in order. */
function recordPhases(container: HTMLElement): { values: string[]; stop(): void } {
  const anchor = container.querySelector('[data-phase]');
  if (anchor === null) throw new Error('no anchor');
  const values: string[] = [];
  const observer = new MutationObserver(() => {
    const value = anchor.getAttribute('data-phase') ?? '';
    if (values[values.length - 1] !== value) values.push(value);
  });
  observer.observe(anchor, { attributes: true, attributeFilter: ['data-phase'] });
  return { values, stop: () => observer.disconnect() };
}

/** `data-phase` of the always-mounted anchor around the trigger. */
function phase(container: HTMLElement): string | null {
  return container.querySelector('[data-phase]')?.getAttribute('data-phase') ?? null;
}

afterEach(() => {
  act(() => root?.unmount());
  root = null;
  document.body.innerHTML = '';
});

describe('ConfirmPopover', () => {
  it('opens at the trigger with Cancel focused, and Escape closes it back to the trigger', async () => {
    const onOpenChange = vi.fn();
    const container = mount(popover({ onOpenChange }));
    expect(phase(container)).toBe('idle');

    click(trigger(container));
    await flush();
    const surface = dialog();
    expect(surface).not.toBeNull();
    expect(surface?.getAttribute('aria-labelledby')).toBe(byRole(document, 'heading', 'Delete qwen3-8b?')?.id);
    expect(document.activeElement?.textContent).toBe('Cancel');
    expect(phase(container)).toBe('confirming');
    expect(document.querySelector('[data-shape]')?.getAttribute('data-shape')).toBe('panel');
    expect(onOpenChange).toHaveBeenLastCalledWith(true);

    pressEscape();
    await waitFor(() => dialog() === null);
    expect(onOpenChange).toHaveBeenLastCalledWith(false);
    expect(phase(container)).toBe('idle');
    await waitFor(() => document.activeElement === trigger(container));
  });

  it('gives focus back to the trigger as the surface starts to shrink, not when the exit ends', async () => {
    const previous = skipAnimations(false);
    try {
      const container = mount(popover());
      const button = trigger(container);
      click(button);
      await flush();
      expect(document.activeElement?.textContent).toBe('Cancel');
      pressEscape();
      await flush();
      expect(dialog()).not.toBeNull();
      expect(document.activeElement).toBe(button);

      act(() => root?.unmount());
      document.body.innerHTML = '';
      const again = trigger(mount(popover()));
      click(again);
      await flush();
      expect(document.activeElement?.textContent).toBe('Cancel');
      click(document.activeElement);
      await flush();
      expect(dialog()).not.toBeNull();
      expect(document.activeElement).toBe(again);
    } finally {
      skipAnimations(previous);
    }
  });

  it('sizes the open layer to the measured panel, which Radix Portal mounts one commit late', async () => {
    const original = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'offsetHeight');
    Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
      configurable: true,
      get(this: HTMLElement) {
        return this.querySelector(':scope > h2') === null ? 0 : 136;
      },
    });
    try {
      const container = mount(popover());
      click(trigger(container));
      await flush();
      expect(dialog()?.style.height).toBe('136px');
    } finally {
      if (original !== undefined) Object.defineProperty(HTMLElement.prototype, 'offsetHeight', original);
    }
  });

  it('confirms once, ignores Escape while working, and shows a rejection in place', async () => {
    const pending = deferred();
    const onConfirm = vi.fn(() => pending.promise);
    const container = mount(popover({ onConfirm, errorTitle: 'Failed to delete model' }));
    click(trigger(container));
    await flush();

    click(byRole(dialog() as HTMLElement, 'button', 'Delete'));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(phase(container)).toBe('working');
    await waitFor(() => byRole(dialog() as HTMLElement, 'button', 'Deleting…') !== null);
    const working = byRole(dialog() as HTMLElement, 'button', 'Deleting…');
    expect(working?.getAttribute('aria-busy')).toBe('true');
    expect(byRole(dialog() as HTMLElement, 'button', 'Cancel')?.hasAttribute('disabled')).toBe(true);

    click(working);
    expect(onConfirm).toHaveBeenCalledTimes(1);
    pressEscape();
    await flush();
    expect(dialog()).not.toBeNull();
    expect(phase(container)).toBe('working');

    await act(async () => {
      pending.reject(new Error('The model is loaded (409)'));
      await pending.promise.catch(() => undefined);
    });
    const alert = (dialog() as HTMLElement).querySelector('[role="alert"]');
    expect(alert?.textContent).toContain('Failed to delete model');
    expect(alert?.textContent).toContain('The model is loaded (409)');
    expect(phase(container)).toBe('confirming');
    expect(dialog()).not.toBeNull();
  });

  it('passes through the four phases, then closes and calls onDone once', async () => {
    const pending = deferred();
    const onDone = vi.fn();
    const container = mount(popover({ onConfirm: () => pending.promise, onDone }));
    const phases = recordPhases(container);
    click(trigger(container));
    await flush();
    click(byRole(dialog() as HTMLElement, 'button', 'Delete'));

    await act(async () => {
      pending.resolve();
      await pending.promise;
    });
    await waitFor(() => dialog() === null);
    phases.stop();
    expect(phases.values).toEqual(['confirming', 'working', 'success', 'idle']);
    await waitFor(() => onDone.mock.calls.length > 0);
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(phase(container)).toBe('idle');
  });

  it('still closes and calls onDone once when it unmounts during the success hold', async () => {
    const pending = deferred();
    const onDone = vi.fn();
    const onOpenChange = vi.fn();
    const container = mount(popover({ onConfirm: () => pending.promise, onDone, onOpenChange }));
    click(trigger(container));
    await flush();
    click(byRole(dialog() as HTMLElement, 'button', 'Delete'));
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      await act(async () => {
        pending.resolve();
        await pending.promise;
      });
      expect(phase(container)).toBe('success');
      expect(onDone).not.toHaveBeenCalled();
      act(() => root?.unmount());
      root = null;
      expect(onOpenChange).toHaveBeenLastCalledWith(false);
      expect(onDone).toHaveBeenCalledTimes(1);
      vi.runAllTimers();
      expect(onDone).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('calls onDone once when it unmounts while onConfirm is pending and the promise then resolves', async () => {
    const pending = deferred();
    const onDone = vi.fn();
    const onOpenChange = vi.fn();
    const container = mount(popover({ onConfirm: () => pending.promise, onDone, onOpenChange }));
    click(trigger(container));
    await flush();
    click(byRole(dialog() as HTMLElement, 'button', 'Delete'));
    act(() => root?.unmount());
    root = null;
    expect(onOpenChange).toHaveBeenLastCalledWith(false);
    const closes = onOpenChange.mock.calls.length;
    pending.resolve();
    await pending.promise;
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(onOpenChange).toHaveBeenCalledTimes(closes);
  });

  it('gives focus to the scroll root when onDone removes the trigger later, after a fetch', async () => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const scroller = document.createElement('main');
    scroller.setAttribute('data-scroll-root', '');
    scroller.tabIndex = -1;
    document.body.appendChild(scroller);
    let reloaded: (() => void) | undefined;
    function Row(): ReactElement | null {
      const [shown, setShown] = useState(true);
      if (!shown) return null;
      return popover({ onDone: () => (reloaded = () => setShown(false)) });
    }
    root = createRoot(scroller);
    act(() => root?.render(createElement(Row)));
    click(trigger(scroller));
    await flush();
    click(byRole(dialog() as HTMLElement, 'button', 'Delete'));
    await waitFor(() => dialog() === null);
    const button = byRole(scroller, 'button', 'Delete model');
    expect(button).not.toBeNull();
    act(() => button!.focus());
    expect(document.activeElement).toBe(button);
    await waitFor(() => reloaded !== undefined);
    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
      reloaded!();
    });
    expect(byRole(scroller, 'button', 'Delete model')).toBeNull();
    await flush();
    expect(document.activeElement).toBe(scroller);
  });

  it('grows from the other corner when the asked-for side has no room for the 300 px panel', () => {
    expect(fitAlign('end', { left: 16, right: 110 }, 1180)).toBe('start');
    expect(fitAlign('start', { left: 1000, right: 1100 }, 1180)).toBe('end');
    expect(fitAlign('end', { left: 900, right: 1000 }, 1180)).toBe('end');
    expect(fitAlign('start', { left: 16, right: 110 }, 1180)).toBe('start');
    expect(fitAlign('end', { left: 16, right: 110 }, 200)).toBe('end');
  });

  it('a disabled trigger looks off: aria-disabled is styled like Arc :disabled, hover included', () => {
    const rules = cssRules('components/arc-x/confirm-popover/confirm-popover.module.css');
    expect(cssValue(rules, ".trigger[aria-disabled='true']", 'color')).toBe('var(--text-muted)');
    expect(cssValue(rules, ".trigger[aria-disabled='true']", 'cursor')).toBe('not-allowed');
    expect(cssValue(rules, ".trigger[aria-disabled='true'][data-trigger='ghost']:hover", 'background')).toBe(
      'transparent',
    );
    const container = mount(popover({ disabled: true, disabledReason: 'Unload the model first' }));
    expect(trigger(container).getAttribute('data-trigger')).toBe('solid');
  });

  it('a long unbroken title or path wraps inside the 300 px panel and the dialog, so the buttons stay in view', () => {
    for (const file of ['confirm-popover/confirm-popover', 'confirm-dialog/confirm-dialog']) {
      const rules = cssRules(`components/arc-x/${file}.module.css`);
      expect(cssValue(rules, '.description', 'overflow-wrap')).toBe('anywhere');
      expect(cssValue(rules, '.notice', 'overflow-wrap')).toBe('anywhere');
    }
  });

  it('a disabled trigger does not open', async () => {
    const container = mount(popover({ disabled: true, disabledReason: 'Unload the model first' }));
    expect(trigger(container).getAttribute('aria-disabled')).toBe('true');
    click(trigger(container));
    await flush();
    expect(dialog()).toBeNull();
    expect(phase(container)).toBe('idle');
  });

  it('an icon trigger is named by its label', async () => {
    const container = mount(
      popover({
        trigger: { kind: 'icon', label: 'Delete model', icon: createElement('svg', { 'aria-hidden': true }) },
      }),
    );
    click(trigger(container));
    await flush();
    expect(dialog()).not.toBeNull();
    expect(document.activeElement?.textContent).toBe('Cancel');
  });

  it('a controlled open opens and cancels', async () => {
    const container = mount(popover({ open: true }));
    await flush();
    expect(dialog()).not.toBeNull();
    act(() => root?.render(popover({ open: false })));
    await waitFor(() => dialog() === null);
    expect(phase(container)).toBe('idle');
  });
});

describe('ConfirmDialog', () => {
  function dialogElement(overrides: Partial<ConfirmContent> = {}, onOpenChange = vi.fn()): ReactElement {
    return createElement(ConfirmDialog, { ...content(overrides), open: true, onOpenChange });
  }

  it('is described by its description paragraph', async () => {
    mount(dialogElement());
    await flush();
    const element = dialog() as HTMLElement;
    const describedBy = element.getAttribute('aria-describedby');
    expect(describedBy).not.toBeNull();
    const description = document.getElementById(describedBy ?? '');
    expect(description?.tagName).toBe('P');
    expect(description?.textContent).toBe('Frees 5.2 GB. The files are removed from disk.');
  });

  it('keeps the dialog open on a rejection and shows the error title and message', async () => {
    const onOpenChange = vi.fn();
    mount(
      dialogElement(
        { onConfirm: () => Promise.reject(new Error('Session is live')), errorTitle: 'Could not delete' },
        onOpenChange,
      ),
    );
    await flush();
    click(byRole(dialog() as HTMLElement, 'button', 'Delete'));
    await flush();
    const alert = (dialog() as HTMLElement).querySelector('[role="alert"]');
    expect(alert?.textContent).toContain('Could not delete');
    expect(alert?.textContent).toContain('Session is live');
    expect(onOpenChange).not.toHaveBeenCalledWith(false);
  });

  it('on resolve closes and calls onDone once', async () => {
    const onDone = vi.fn();
    let open = true;
    const render = (): ReactElement =>
      createElement(ConfirmDialog, {
        ...content({ onDone }),
        open,
        onOpenChange: (next: boolean) => {
          open = next;
          act(() => root?.render(render()));
        },
      });
    mount(render());
    await flush();
    click(byRole(dialog() as HTMLElement, 'button', 'Delete'));
    await waitFor(() => dialog() === null);
    expect(onDone).toHaveBeenCalledTimes(1);
  });
});
