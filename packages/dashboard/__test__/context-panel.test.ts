/**
 * @vitest-environment happy-dom
 */

import { act, createElement, useState, type ReactElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { Composer, type ComposerProps } from '../ui/src/components/arc-x/composer/composer.js';
import {
  autoLine,
  ContextPanel,
  contextParts,
  type ContextPanelProps,
} from '../ui/src/components/arc-x/context-panel/context-panel.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { ContextParts } from '../ui/src/lib/chat-types.js';
import type { SessionUsage } from '../ui/src/lib/session-usage.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { accessibleName, byRole, waitFor } from './render.js';

afterEach(unmountAll);

const PARTS: ContextParts = {
  systemTokens: 4_000,
  toolCount: 42,
  toolTokens: 10_000,
  messageTokens: 146_000,
  autoCompactPercent: 85,
  compactions: 1,
};

const USAGE: SessionUsage = {
  turns: 14,
  modelCalls: 87,
  readTokens: 2_100_000,
  cachedTokens: 1_800_000,
  writtenTokens: 46_000,
  elapsedMs: 760_000,
};

function props(extra: Partial<ContextPanelProps> = {}): ContextPanelProps {
  return {
    used: 160_000,
    window: 262_144,
    parts: PARTS,
    usage: USAGE,
    compact: { off: null, onCompact: () => undefined },
    inspectTo: '/sessions/s1?tab=activity',
    ...extra,
  };
}

function panel(extra: Partial<ContextPanelProps> = {}): ReactElement {
  return createElement(MemoryRouter, null, createElement(ContextPanel, props(extra)));
}

function text(container: Element): string {
  return container.textContent ?? '';
}

describe('ContextPanel', () => {
  it('hot: the use of the window, the bar whose name holds every part, the parts as a list, the compaction point', () => {
    const container = mount(panel());
    expect(text(container)).toContain(fill(CHAT.history.contextHead, { used: '156K', window: '256K', percent: '61%' }));
    const bar = byRole(container, 'img')!;
    const name = accessibleName(bar);
    const list = contextParts(PARTS, 160_000, 262_144);
    expect(list.map((part) => part.key)).toEqual(['system', 'tools', 'messages', 'free']);
    for (const part of list) expect(name).toContain(part.label);
    expect(name).toContain(CHAT.history.partFree);
    const terms = [...container.querySelectorAll('dl dt')].map((dt) => dt.textContent);
    expect(terms.slice(0, 4)).toEqual([
      CHAT.history.partSystem,
      fill(CHAT.history.partTools, { n: 42 }),
      CHAT.history.partMessages,
      CHAT.history.partFree,
    ]);
    expect(text(container)).toContain(`${fill(CHAT.history.contextAuto, { n: 85 })} · ${CHAT.history.compactedOnce}`);
    expect(text(container)).not.toContain(CHAT.history.contextCold);
  });

  it('the parts are scaled to the use and the free rest fills the window; the agent’s count stands when no use is known', () => {
    const list = contextParts(PARTS, 80_000, 262_144);
    expect(list.slice(0, 3).reduce((sum, part) => sum + part.tokens, 0)).toBe(80_000);
    expect(list[3]).toMatchObject({ key: 'free', tokens: 262_144 - 80_000 });
    expect(contextParts(PARTS, null, null).map((part) => part.tokens)).toEqual([4_000, 10_000, 146_000]);
    const odd = contextParts({ ...PARTS, systemTokens: 1, toolTokens: 1, messageTokens: 1 }, 4, 100);
    expect(odd.map((part) => part.tokens)).toEqual([1, 1, 2, 96]);
    expect(autoLine({ ...PARTS, compactions: 0 })).toBe(fill(CHAT.history.contextAuto, { n: 85 }));
    expect(autoLine({ ...PARTS, compactions: 3 })).toBe(
      `${fill(CHAT.history.contextAuto, { n: 85 })} · ${fill(CHAT.history.compactedTimes, { n: 3 })}`,
    );
  });

  it('rounded parts leave no stray rest in the bar’s name, and a compaction point the bar cannot mark is a line', () => {
    const container = mount(
      panel({ used: 4, window: 100, parts: { ...PARTS, systemTokens: 1, toolTokens: 1, messageTokens: 1 } }),
    );
    const name = accessibleName(byRole(container, 'img')!);
    expect(name.split(`${CHAT.history.partFree} `)).toHaveLength(2);
    expect(name.toLowerCase().split('free')).toHaveLength(2);
    const auto = `${fill(CHAT.history.contextAuto, { n: 100 })} · ${CHAT.history.compactedOnce}`;
    rerender(panel({ parts: { ...PARTS, autoCompactPercent: 100 } }));
    expect(text(container)).toContain(auto);
    rerender(panel({ parts: { ...PARTS, autoCompactPercent: 85 } }));
    const line = `${fill(CHAT.history.contextAuto, { n: 85 })} · ${CHAT.history.compactedOnce}`;
    expect(text(container).split(line)).toHaveLength(2);
  });

  it('cold or a failed read: a meter of the use, the line that asks for a message, no parts and no error', () => {
    const container = mount(panel({ parts: null }));
    expect(text(container)).toContain(CHAT.history.contextCold);
    expect(byRole(container, 'meter', CHAT.history.context)).not.toBeNull();
    expect(byRole(container, 'img')).toBeNull();
    expect(container.querySelector('dl dt')?.textContent).toBe(CHAT.history.usageTurns);
    expect(byRole(container, 'button', CHAT.history.compact)).not.toBeNull();
  });

  it('loading: three placeholder rows where the parts go', () => {
    const container = mount(panel({ parts: undefined }));
    const loading = container.querySelector('[aria-busy="true"]')!;
    expect(loading.children).toHaveLength(3);
    expect(text(container)).not.toContain(CHAT.history.contextCold);
    expect(text(container)).not.toContain(CHAT.history.partSystem);
  });

  it('no window known: no bar and no free part, the use alone, the compaction point as a line', () => {
    const container = mount(panel({ window: null }));
    expect(byRole(container, 'img')).toBeNull();
    expect(byRole(container, 'meter')).toBeNull();
    expect(text(container)).toContain(fill(CHAT.history.usageTokens, { n: '156K' }));
    expect(text(container)).not.toContain(CHAT.history.partFree);
    expect(text(container)).toContain(fill(CHAT.history.contextAuto, { n: 85 }));
  });

  it('the totals of the conversation, local, with no cost row, and a way to Inspect', () => {
    const container = mount(panel());
    const totals = container.querySelector(`dl[aria-label="${CHAT.history.usage}"]`)!;
    const rows = [...totals.querySelectorAll(':scope > div')].map((row) => [
      row.querySelector('dt')!.textContent,
      row.querySelector('dd')!.textContent,
    ]);
    expect(rows).toEqual([
      [CHAT.history.usageTurns, '14'],
      [CHAT.history.usageCalls, '87'],
      [CHAT.history.usageRead, fill(CHAT.history.usageReadValue, { n: '2.1M', cached: '1.8M' })],
      [CHAT.history.usageWritten, fill(CHAT.history.usageTokens, { n: '46K' })],
      [CHAT.history.usageTime, '12m 40s'],
    ]);
    expect(text(container)).toContain(CHAT.history.usageLocal);
    expect(text(container).toLowerCase()).not.toContain('$');
    const inspect = byRole(container, 'link', CHAT.inspect)!;
    expect(inspect.getAttribute('href')).toBe('/sessions/s1?tab=activity');
    rerender(
      panel({
        usage: {
          turns: 2,
          modelCalls: null,
          readTokens: null,
          cachedTokens: null,
          writtenTokens: null,
          elapsedMs: null,
        },
      }),
    );
    const empty = [...container.querySelectorAll(`dl[aria-label="${CHAT.history.usage}"] dd`)].map(
      (dd) => dd.textContent,
    );
    expect(empty[0]).toBe('2');
    expect(empty.slice(1).every((value) => value === '—')).toBe(true);
  });

  it('Compact now runs when it may; off, it stays focusable, reads its reason and does nothing', () => {
    const onCompact = vi.fn();
    const container = mount(panel({ compact: { off: null, onCompact } }));
    const button = byRole(container, 'button', CHAT.history.compact)!;
    act(() => button.click());
    expect(onCompact).toHaveBeenCalledTimes(1);
    for (const off of [CHAT.modelBusy, CHAT.history.offRun, CHAT.history.compactEmpty, CHAT.history.engineLost]) {
      rerender(panel({ compact: { off, onCompact } }));
      const offButton = byRole(container, 'button', CHAT.history.compact)!;
      expect(offButton.getAttribute('aria-disabled')).toBe('true');
      expect(offButton.hasAttribute('disabled')).toBe(false);
      const note = document.getElementById(offButton.getAttribute('aria-describedby')!)!;
      expect(note.textContent).toBe(off);
      act(() => offButton.click());
    }
    expect(onCompact).toHaveBeenCalledTimes(1);
  });
});

describe('the Context panel from the composer', () => {
  function Host(extra: {
    context?: ComposerProps['context'];
    blocked?: string;
    value?: string;
    open?: boolean;
  }): ReactElement {
    const [open, setOpen] = useState(extra.open === true);
    return createElement(
      MemoryRouter,
      null,
      createElement(Composer, {
        value: extra.value ?? '',
        onValueChange: () => undefined,
        placeholder: 'Ask anything',
        phase: 'idle',
        onSend: () => undefined,
        onStop: () => undefined,
        tools: createElement('span', null, 'TOOLS'),
        blocked: extra.blocked,
        context: extra.context,
        contextPanel: {
          open,
          onOpenChange: setOpen,
          content: createElement(ContextPanel, props()),
        },
      }),
    );
  }
  const dialog = (): HTMLElement | null => byRole(document.body, 'dialog', CHAT.history.context);

  it('the ring is the button: it opens the panel upward, says so, and a second press closes it', async () => {
    const container = mount(createElement(Host, { context: { used: 160_000, window: 262_144 } }));
    const ring = byRole(container, 'button', 'Context: 156K of 256K tokens (61%)')!;
    expect(ring.getAttribute('aria-haspopup')).toBe('dialog');
    expect(ring.getAttribute('aria-expanded')).toBe('false');
    act(() => ring.click());
    await waitFor(() => dialog() !== null);
    expect(ring.getAttribute('aria-expanded')).toBe('true');
    expect(dialog()!.getAttribute('data-side')).toBe('top');
    expect(dialog()!.textContent).toContain(CHAT.history.usage);
    // A press is two events with a render between: the pointer down must not close it before the click toggles.
    act(() => {
      ring.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    act(() => ring.click());
    await waitFor(() => ring.getAttribute('aria-expanded') === 'false');
    await waitFor(() => dialog() === null);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(dialog()).toBeNull();
  });

  it('Esc closes it and focus goes back to the ring', async () => {
    const container = mount(createElement(Host, { context: { used: 160_000, window: 262_144 } }));
    const ring = byRole(container, 'button', 'Context: 156K of 256K tokens (61%)')!;
    act(() => ring.click());
    await waitFor(() => dialog() !== null);
    act(() => {
      dialog()!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    });
    await waitFor(() => dialog() === null);
    expect(ring.getAttribute('aria-expanded')).toBe('false');
    await waitFor(() => document.activeElement === ring);
  });

  it('opened another way while a reason hides the ring: it hangs from the tools row and Esc gives focus to the box', async () => {
    const container = mount(
      createElement(Host, {
        context: { used: 160_000, window: 262_144 },
        value: 'x',
        blocked: 'Engine stopped',
        open: true,
      }),
    );
    expect(byRole(container, 'button', 'Context: 156K of 256K tokens (61%)')).toBeNull();
    await waitFor(() => dialog() !== null);
    act(() => {
      dialog()!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    });
    await waitFor(() => dialog() === null);
    await waitFor(() => document.activeElement === container.querySelector('textarea'));
  });
});
