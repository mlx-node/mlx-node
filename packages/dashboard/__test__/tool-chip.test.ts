/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { ToolChip } from '../ui/src/components/arc-x/tool-chip/tool-chip.js';
import { mount, unmountAll } from './helpers/mount.js';
import { byRole } from './render.js';

afterEach(unmountAll);

const icon = createElement('svg', { 'data-icon': '' });

describe('ToolChip', () => {
  it('on by default with its label; off is marked; a click reaches the handler', () => {
    const onClick = vi.fn();
    const container = mount(createElement(ToolChip, { icon, label: 'Web Search', onClick }));
    const chip = byRole(container, 'button', 'Web Search')!;
    expect(chip.getAttribute('data-chip')).toBe('on');
    expect(chip.querySelector('[data-chip-label]')?.textContent).toBe('Web Search');
    act(() => chip.click());
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('off with a dot: the words for the dot are in the accessible name', () => {
    const container = mount(
      createElement(ToolChip, {
        icon,
        label: 'Web Search',
        state: 'off',
        dot: 'warning',
        'aria-label': 'Web Search, no key',
      }),
    );
    const chip = byRole(container, 'button', 'Web Search, no key')!;
    expect(chip.getAttribute('data-chip')).toBe('off');
    expect(chip.querySelector('[data-tone="warning"]')).not.toBeNull();
  });

  it('compact: icon only, the label stays the name', () => {
    const container = mount(createElement(ToolChip, { icon, label: 'Web Search', compact: true }));
    const chip = byRole(container, 'button', 'Web Search')!;
    expect(chip.querySelector('[data-chip-label]')).toBeNull();
    expect(chip.hasAttribute('data-compact')).toBe(true);
  });

  it('give: a tier only on a label longer than 8 characters, so a short chip never gets a floor wider than its text', () => {
    const tier = (label: string, give?: 1 | 2): string | null =>
      mount(createElement(ToolChip, { icon, label, give })).querySelector('button')!.getAttribute('data-give');
    expect([tier('napi-rsxx', 1), tier('napi-rsx', 1), tier('Web · DeepSeek', 2), tier('Web · DeepSeek')]).toEqual([
      '1',
      null,
      '2',
      null,
    ]);
  });
});
