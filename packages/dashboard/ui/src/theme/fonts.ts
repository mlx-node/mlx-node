import { useSyncExternalStore } from 'react';

import { hasNonWhitespace } from '../lib/text-scan';

export type FontRole = 'reading' | 'code' | 'ui';

/** `default` is the role's built-in font, `system` is system-ui, `name:<family>` an installed or bundled family. */
export type FontFamily = 'default' | 'system' | `name:${string}`;

export interface FontSetting {
  family: FontFamily;
  size: number;
}

export type FontSettings = Readonly<Record<FontRole, Readonly<FontSetting>>>;

export const FONTS_STORAGE_KEY = 'mlx-node:fonts:v1';

export const FONT_ROLES: readonly FontRole[] = ['reading', 'code', 'ui'];

export const FONT_LIMITS: Readonly<Record<FontRole, { readonly min: number; readonly max: number }>> = {
  reading: { min: 11, max: 24 },
  code: { min: 10, max: 20 },
  ui: { min: 12, max: 15 },
};

export const FONT_DEFAULTS: FontSettings = {
  reading: { family: 'default', size: 13 },
  code: { family: 'default', size: 12 },
  ui: { family: 'default', size: 13 },
};

/** What follows a chosen family, so a missing font or glyph falls to the face the app ships. */
export const FONT_FALLBACK: Readonly<Record<FontRole, string>> = {
  reading: '"Inter Variable", sans-serif',
  code: '"Geist Mono Variable", Menlo, monospace',
  ui: '"Inter Variable", sans-serif',
};

const SYSTEM_STACK = 'system-ui, sans-serif';
const NAME_PREFIX = 'name:';
const NAME_MAX = 128;

const STOCK_SIZES: ReadonlyArray<readonly [string, number]> = [
  ['--text-xs', 11],
  ['--text-sm', 13],
  ['--text-base', 14],
  ['--text-lg', 16],
  ['--text-xl', 20],
  ['--text-2xl', 24],
  ['--text-3xl', 32],
  ['--text-mono', 12],
  ['--text-field', 13],
];

function rem(px: number): string {
  return `${Number((px / 16).toFixed(4))}rem`;
}

/** The 13 interface properties as the stylesheets set them: equal to `tokens.css` and Arc's `foundation.css`. */
export const STOCK_TYPE: Readonly<Record<string, string>> = {
  '--font-body': 'var(--font-inter, "Inter Variable"), sans-serif',
  '--font-display': 'var(--font-geist, "Geist Variable"), sans-serif',
  '--tracking-body': '-.01em',
  '--tracking-display': '-.03em',
  ...Object.fromEntries(STOCK_SIZES.map(([name, px]) => [name, rem(px)])),
};

const MANAGED: readonly string[] = [
  '--font-reading',
  '--tracking-reading',
  '--text-reading',
  '--font-code',
  '--text-code',
  ...Object.keys(STOCK_TYPE),
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasControlChar(text: string): boolean {
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code < 0x20 || (code >= 0x7f && code <= 0x9f)) return true;
  }
  return false;
}

function repairFamily(role: FontRole, value: unknown): FontFamily {
  if (value === 'default') return 'default';
  if (value === 'system') return role === 'code' ? 'default' : 'system';
  if (typeof value !== 'string' || !value.startsWith(NAME_PREFIX)) return 'default';
  const name = value.slice(NAME_PREFIX.length);
  if (name.length > NAME_MAX || !hasNonWhitespace(name) || hasControlChar(name)) return 'default';
  return `name:${name}`;
}

function repairSize(role: FontRole, value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return FONT_DEFAULTS[role].size;
  const { min, max } = FONT_LIMITS[role];
  return Math.min(max, Math.max(min, Math.round(value)));
}

export function repairFontSettings(value: unknown): FontSettings {
  const source = isRecord(value) ? value : {};
  const repaired = {} as Record<FontRole, FontSetting>;
  for (const role of FONT_ROLES) {
    const entry = isRecord(source[role]) ? source[role] : {};
    repaired[role] = { family: repairFamily(role, entry.family), size: repairSize(role, entry.size) };
  }
  return repaired;
}

export function parseStoredFonts(text: string | null | undefined): FontSettings {
  if (typeof text !== 'string') return FONT_DEFAULTS;
  try {
    return repairFontSettings(JSON.parse(text));
  } catch {
    return FONT_DEFAULTS;
  }
}

export function cssFamilyName(name: string): string {
  let out = '"';
  for (const ch of name) out += ch === '\\' || ch === '"' ? `\\${ch}` : ch;
  return `${out}"`;
}

function familyStack(role: FontRole, family: FontFamily): string | null {
  if (family === 'default') return null;
  if (family === 'system') return SYSTEM_STACK;
  return `${cssFamilyName(family.slice(NAME_PREFIX.length))}, ${FONT_FALLBACK[role]}`;
}

/** The custom properties a value sets on `<html>`. A setting at its default sets nothing, so the stylesheet rules. */
export function fontProperties(settings: FontSettings): Record<string, string> {
  const props: Record<string, string> = {};
  const reading = familyStack('reading', settings.reading.family);
  if (reading !== null) {
    props['--font-reading'] = reading;
    props['--tracking-reading'] = '0';
  }
  if (settings.reading.size !== FONT_DEFAULTS.reading.size) props['--text-reading'] = rem(settings.reading.size);
  const code = familyStack('code', settings.code.family);
  if (code !== null) props['--font-code'] = code;
  if (settings.code.size !== FONT_DEFAULTS.code.size) props['--text-code'] = rem(settings.code.size);
  const ui = familyStack('ui', settings.ui.family);
  if (ui !== null) {
    props['--font-body'] = ui;
    props['--font-display'] = ui;
    props['--tracking-body'] = '0';
    props['--tracking-display'] = '0';
  }
  if (settings.ui.size !== FONT_DEFAULTS.ui.size) {
    for (const [name, px] of STOCK_SIZES) props[name] = rem((px * settings.ui.size) / FONT_DEFAULTS.ui.size);
  }
  return props;
}

export function stockTypeProperties(): Record<string, string> {
  return { ...STOCK_TYPE };
}

export function isDefaultFonts(settings: FontSettings): boolean {
  return FONT_ROLES.every(
    (role) => settings[role].family === FONT_DEFAULTS[role].family && settings[role].size === FONT_DEFAULTS[role].size,
  );
}

function sameFonts(a: FontSettings, b: FontSettings): boolean {
  return FONT_ROLES.every((role) => a[role].family === b[role].family && a[role].size === b[role].size);
}

let snapshot: FontSettings = FONT_DEFAULTS;
const listeners = new Set<() => void>();
let teardown: (() => void) | null = null;

function readStoredFonts(): FontSettings {
  try {
    return parseStoredFonts(globalThis.localStorage?.getItem(FONTS_STORAGE_KEY));
  } catch {
    return FONT_DEFAULTS;
  }
}

function apply(settings: FontSettings): void {
  const props = fontProperties(settings);
  const style = document.documentElement.style;
  for (const name of MANAGED) {
    const value = props[name];
    if (value === undefined) style.removeProperty(name);
    else style.setProperty(name, value);
  }
  if (sameFonts(snapshot, settings)) return;
  snapshot = settings;
  for (const listener of listeners) listener();
}

/** Call once in main.tsx, right after initTheme() and before createRoot. */
export function initFonts(): void {
  teardown?.();
  apply(readStoredFonts());
  const onStorage = (event: StorageEvent): void => {
    if (event.key === null || event.key === FONTS_STORAGE_KEY) apply(readStoredFonts());
  };
  window.addEventListener('storage', onStorage);
  teardown = () => {
    window.removeEventListener('storage', onStorage);
    teardown = null;
  };
}

export function setFont(role: FontRole, change: { family: FontFamily } | { size: number }): void {
  const next = repairFontSettings({ ...snapshot, [role]: { ...snapshot[role], ...change } });
  try {
    globalThis.localStorage?.setItem(FONTS_STORAGE_KEY, JSON.stringify(next));
  } catch {
    // A value that cannot be stored still applies to this window.
  }
  apply(next);
}

export function resetFonts(): void {
  try {
    globalThis.localStorage?.removeItem(FONTS_STORAGE_KEY);
  } catch {
    // A reset that cannot be stored still applies to this window.
  }
  apply(FONT_DEFAULTS);
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): FontSettings {
  return snapshot;
}

export function useFonts(): {
  settings: FontSettings;
  setFont: typeof setFont;
  resetFonts: typeof resetFonts;
} {
  const settings = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return { settings, setFont, resetFonts };
}
