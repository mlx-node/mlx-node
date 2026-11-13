/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { TurnFooter } from '../ui/src/components/arc-x/turn-footer/turn-footer.js';
import { mount, unmountAll } from './helpers/mount.js';
import { byRole } from './render.js';

afterEach(unmountAll);

describe('TurnFooter', () => {
  it('renders nothing when it has nothing to say', () => {
    const container = mount(createElement(TurnFooter, { stats: [] }));
    expect(container.innerHTML).toBe('');
  });

  it('stats joined with a dot, their title, the outcome with its detail, copy and the action', () => {
    const onSelect = vi.fn();
    const container = mount(
      createElement(TurnFooter, {
        stats: ['12.4 s', '512 tokens'],
        statsTitle: 'First token 420 ms',
        outcome: { tone: 'danger', text: 'The model call failed.', detail: 'HTTP 500' },
        action: { label: 'Send again', onSelect },
        copy: { value: 'answer', label: 'Copy answer' },
      }),
    );
    expect(container.textContent).toContain('12.4 s · 512 tokens');
    expect(container.querySelector('[title="First token 420 ms"]')).not.toBeNull();
    expect(container.querySelector('[data-tone="danger"]')?.textContent).toContain('The model call failed.');
    expect(container.textContent).toContain('HTTP 500');
    expect(byRole(container, 'button', 'Copy answer')).not.toBeNull();
    act(() => byRole(container, 'button', 'Send again')!.click());
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it('a disabled action does not fire; quiet marks the footer', () => {
    const onSelect = vi.fn();
    const container = mount(
      createElement(TurnFooter, {
        stats: [],
        action: { label: 'Send again', onSelect, disabled: true, disabledReason: 'The engine is stopped.' },
        quiet: true,
      }),
    );
    const button = byRole(container, 'button', 'Send again') as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    act(() => button.click());
    expect(onSelect).not.toHaveBeenCalled();
    expect(container.querySelector('[data-quiet]')).not.toBeNull();
  });
});
