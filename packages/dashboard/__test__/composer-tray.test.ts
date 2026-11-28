/**
 * @vitest-environment happy-dom
 */

import { createElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { ComposerTray } from '../ui/src/components/arc-x/composer-tray/composer-tray.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { byRole, waitFor } from './render.js';

afterEach(unmountAll);

describe('ComposerTray', () => {
  it('clips both paint and scrollable overflow with .pocket instead of clip-path alone on .clip', () => {
    const rules = cssRules('components/arc-x/composer-tray/composer-tray.module.css');
    expect(cssValue(rules, '.pocket', 'overflow')).toBe('clip');
    expect(cssValue(rules, '.pocket', 'margin-top')).toBe('calc(-1 * var(--radius-surface))');
    expect(cssValue(rules, '.pocket', 'padding-top')).toBe('var(--radius-surface)');
    expect(cssValue(rules, '.pocket', 'pointer-events')).toBe('none');
    expect(cssValue(rules, '.clip', 'clip-path')).toBeUndefined();
    expect(cssValue(rules, '.clip', 'display')).toBe('flex');
    expect(cssValue(rules, '.clip', 'align-items')).toBe('flex-end');
    expect(cssValue(rules, '.tray', 'width')).toBe('100%');
    expect(cssValue(rules, '.tray', 'pointer-events')).toBe('auto');
  });

  it('wraps the animated tray inside .pocket and unmounts the group when tray is null', async () => {
    const card = createElement('textarea', { 'aria-label': 'Message' });
    const chip = createElement('button', { type: 'button', 'data-tray-part': 'folder' }, 'app');
    const container = mount(createElement(ComposerTray, { tray: chip, label: 'Conversation options', children: card }));

    const pocket = container.querySelector('[class*="pocket"]');
    const clip = pocket?.querySelector('[class*="clip"]');
    const group = byRole(container, 'group', 'Conversation options');
    expect(pocket).not.toBeNull();
    expect(clip).not.toBeNull();
    expect(group).not.toBeNull();
    expect(clip?.contains(group!)).toBe(true);
    expect(pocket?.contains(container.querySelector('textarea')!)).toBe(false);

    rerender(createElement(ComposerTray, { tray: null, label: 'Conversation options', children: card }));
    expect(container.querySelector('[class*="pocket"]')).not.toBeNull();
    await waitFor(() => byRole(container, 'group', 'Conversation options') === null);
  });
});
