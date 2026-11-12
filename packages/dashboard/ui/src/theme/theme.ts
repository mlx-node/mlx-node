import { useSyncExternalStore } from 'react';

export type ThemePreference = 'system' | 'light' | 'dark';
export type ResolvedTheme = 'light' | 'dark';

export const THEME_STORAGE_KEY = 'mlx-node:theme:v1';

const DARK_QUERY = '(prefers-color-scheme: dark)';

interface ThemeSnapshot {
  preference: ThemePreference;
  resolved: ResolvedTheme;
}

let snapshot: ThemeSnapshot = { preference: 'system', resolved: 'light' };
const listeners = new Set<() => void>();
let teardown: (() => void) | null = null;

function isThemePreference(value: unknown): value is ThemePreference {
  return value === 'system' || value === 'light' || value === 'dark';
}

export function readThemePreference(): ThemePreference {
  try {
    const stored = globalThis.localStorage?.getItem(THEME_STORAGE_KEY);
    return isThemePreference(stored) ? stored : 'system';
  } catch {
    return 'system';
  }
}

export function resolveTheme(preference: ThemePreference, systemDark: boolean): ResolvedTheme {
  if (preference === 'system') return systemDark ? 'dark' : 'light';
  return preference;
}

function systemPrefersDark(): boolean {
  try {
    return typeof window.matchMedia === 'function' && window.matchMedia(DARK_QUERY).matches;
  } catch {
    return false;
  }
}

function apply(preference: ThemePreference): void {
  const resolved = resolveTheme(preference, systemPrefersDark());
  document.documentElement.dataset.theme = resolved;
  if (snapshot.preference === preference && snapshot.resolved === resolved) return;
  snapshot = { preference, resolved };
  for (const listener of listeners) listener();
}

function syncHidden(): void {
  if (document.hidden) document.documentElement.dataset.hidden = '';
  else delete document.documentElement.dataset.hidden;
}

/** Call once, first statement of main.tsx, before createRoot. */
export function initTheme(): void {
  teardown?.();
  apply(readThemePreference());
  syncHidden();

  const onSystemChange = (): void => apply(snapshot.preference);
  const onStorage = (event: StorageEvent): void => {
    if (event.key === null || event.key === THEME_STORAGE_KEY) apply(readThemePreference());
  };
  let media: MediaQueryList | null = null;
  try {
    media = typeof window.matchMedia === 'function' ? window.matchMedia(DARK_QUERY) : null;
  } catch {
    media = null;
  }
  media?.addEventListener('change', onSystemChange);
  window.addEventListener('storage', onStorage);
  document.addEventListener('visibilitychange', syncHidden);
  teardown = () => {
    media?.removeEventListener('change', onSystemChange);
    window.removeEventListener('storage', onStorage);
    document.removeEventListener('visibilitychange', syncHidden);
    teardown = null;
  };
}

export function setThemePreference(preference: ThemePreference): void {
  try {
    globalThis.localStorage?.setItem(THEME_STORAGE_KEY, preference);
  } catch {
    // A preference that cannot be stored still applies to this window.
  }
  const root = document.documentElement;
  root.dataset.themeSwitching = '';
  apply(preference);
  const clear = (): void => {
    delete root.dataset.themeSwitching;
  };
  if (typeof requestAnimationFrame === 'function') requestAnimationFrame(clear);
  else setTimeout(clear, 0);
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): ThemeSnapshot {
  return snapshot;
}

export function useTheme(): {
  preference: ThemePreference;
  resolved: ResolvedTheme;
  setPreference: (p: ThemePreference) => void;
} {
  const current = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return { preference: current.preference, resolved: current.resolved, setPreference: setThemePreference };
}
