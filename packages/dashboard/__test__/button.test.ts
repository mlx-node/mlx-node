/**
 * @vitest-environment happy-dom
 */

import { createElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { Button } from '../ui/src/components/arc/button/button.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { byRole, waitFor } from './render.js';

afterEach(unmountAll);

describe('Button', () => {
  it('declares overflow: clip on .button so popLayout exit phases cannot escape into scroll roots', () => {
    const rules = cssRules('components/arc/button/button.module.css');
    expect(cssValue(rules, '.button', 'position')).toBe('relative');
    expect(cssValue(rules, '.button', 'overflow')).toBe('clip');
    expect(cssValue(rules, '.labelSlot', 'position')).toBe('relative');
    expect(cssValue(rules, '.labelSlot[data-morphing]', 'clip-path')).toBe('inset(-50% calc(var(--space-3) * -1))');
  });

  it('renders labelSlot and labelPhase inside .button across label changes', async () => {
    const container = mount(
      createElement(
        Button,
        { type: 'button', variant: 'ghost', size: 'sm' },
        createElement('span', null, 'Running'),
        createElement('span', null, '2'),
      ),
    );

    const button = byRole(container, 'button', 'Running2');
    const labelSlot = button?.querySelector('[class*="labelSlot"]');
    const initialPhases = labelSlot ? [...labelSlot.querySelectorAll('[class*="labelPhase"]')] : [];
    expect(button).not.toBeNull();
    expect(button?.className).toContain('button');
    expect(labelSlot).not.toBeNull();
    expect(initialPhases).toHaveLength(1);
    expect(button?.contains(labelSlot!)).toBe(true);
    expect(labelSlot?.contains(initialPhases[0]!)).toBe(true);
    expect(initialPhases[0]?.textContent).toBe('Running2');

    rerender(createElement(Button, { type: 'button', variant: 'ghost', size: 'sm', 'aria-label': 'Activity' }, 'Done'));

    const swapped = byRole(container, 'button', 'Activity');
    const phasesDuringSwap = [...container.querySelectorAll('[class*="labelPhase"]')];
    expect(swapped).toBe(button);
    expect(phasesDuringSwap.length).toBeGreaterThanOrEqual(1);
    for (const phase of phasesDuringSwap) {
      expect(labelSlot?.contains(phase)).toBe(true);
      expect(button?.contains(phase)).toBe(true);
    }

    await waitFor(() => {
      const phases = [...container.querySelectorAll('[class*="labelPhase"]')];
      return phases.length === 1 && phases[0]?.textContent === 'Done';
    });
    const settledPhase = container.querySelector('[class*="labelPhase"]');
    expect(labelSlot?.contains(settledPhase!)).toBe(true);
    expect(button?.contains(settledPhase!)).toBe(true);
  });
});
