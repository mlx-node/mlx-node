/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  initTheme,
  readThemePreference,
  resolveTheme,
  setThemePreference,
  THEME_STORAGE_KEY,
  useTheme,
} from '../ui/src/theme/theme.js';

/** A `matchMedia('(prefers-color-scheme: dark)')` whose answer the test flips. */
function fakeScheme(dark: boolean): { set(dark: boolean): void } {
  const listeners = new Set<(event: MediaQueryListEvent) => void>();
  const list = {
    matches: dark,
    media: '(prefers-color-scheme: dark)',
    addEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) => listeners.add(listener),
    removeEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) => listeners.delete(listener),
  };
  window.matchMedia = (() => list) as unknown as typeof window.matchMedia;
  return {
    set(next: boolean): void {
      list.matches = next;
      for (const listener of listeners) listener({ matches: next } as MediaQueryListEvent);
    },
  };
}

const root = document.documentElement;

beforeEach(() => {
  localStorage.clear();
  delete root.dataset.theme;
  delete root.dataset.themeSwitching;
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe('theme', () => {
  it('resolves a preference against the OS scheme', () => {
    expect(resolveTheme('system', true)).toBe('dark');
    expect(resolveTheme('system', false)).toBe('light');
    expect(resolveTheme('light', true)).toBe('light');
    expect(resolveTheme('dark', false)).toBe('dark');
  });

  it('reads a missing or bad preference as system, and survives storage that throws', () => {
    expect(readThemePreference()).toBe('system');
    localStorage.setItem(THEME_STORAGE_KEY, 'sepia');
    expect(readThemePreference()).toBe('system');
    localStorage.setItem(THEME_STORAGE_KEY, 'dark');
    expect(readThemePreference()).toBe('dark');
    vi.stubGlobal('localStorage', {
      getItem(): never {
        throw new Error('blocked');
      },
    });
    expect(readThemePreference()).toBe('system');
    vi.unstubAllGlobals();
  });

  it('always sets data-theme explicitly and follows the OS while on system', () => {
    const scheme = fakeScheme(false);
    initTheme();
    expect(root.dataset.theme).toBe('light');
    scheme.set(true);
    expect(root.dataset.theme).toBe('dark');
    scheme.set(false);
    expect(root.dataset.theme).toBe('light');
  });

  it('a stored preference wins over the OS', () => {
    const scheme = fakeScheme(true);
    localStorage.setItem(THEME_STORAGE_KEY, 'light');
    initTheme();
    expect(root.dataset.theme).toBe('light');
    scheme.set(false);
    expect(root.dataset.theme).toBe('light');
  });

  it('stores a chosen preference, applies it with transitions off for one frame, and tells the hook', async () => {
    fakeScheme(false);
    initTheme();
    const seen: string[] = [];
    function Probe(): ReactElement {
      const theme = useTheme();
      seen.push(`${theme.preference}/${theme.resolved}`);
      return createElement('span', null, theme.resolved);
    }
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const container = document.createElement('div');
    const reactRoot = createRoot(container);
    act(() => reactRoot.render(createElement(Probe)));

    act(() => setThemePreference('dark'));
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
    expect(root.dataset.theme).toBe('dark');
    expect(root.dataset.themeSwitching).toBe('');
    expect(container.textContent).toBe('dark');
    await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    expect(root.dataset.themeSwitching).toBeUndefined();
    expect(seen.at(-1)).toBe('dark/dark');

    act(() => setThemePreference('system'));
    expect(container.textContent).toBe('light');
    act(() => reactRoot.unmount());
  });

  it('picks up a preference changed in another window', () => {
    fakeScheme(false);
    initTheme();
    localStorage.setItem(THEME_STORAGE_KEY, 'dark');
    window.dispatchEvent(new StorageEvent('storage', { key: THEME_STORAGE_KEY }));
    expect(root.dataset.theme).toBe('dark');
  });

  it('mirrors document.hidden into data-hidden, which pauses CSS loops', () => {
    fakeScheme(false);
    const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
    try {
      initTheme();
      expect(root.dataset.hidden).toBe('');
      hidden.mockReturnValue(false);
      document.dispatchEvent(new Event('visibilitychange'));
      expect(root.dataset.hidden).toBeUndefined();
    } finally {
      hidden.mockRestore();
    }
  });
});
