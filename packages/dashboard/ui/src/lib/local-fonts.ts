import { useEffect, useSyncExternalStore } from 'react';

import type { ComboboxOption } from '../components/arc/combobox/combobox';
import { cssFamilyName, type FontFamily, type FontRole } from '../theme/fonts';
import { fill } from './chat-strings';
import { FONTS } from './fonts-strings';
import { hasNonWhitespace } from './text-scan';

export interface LocalFontFace {
  readonly family: string;
}

export type MeasureText = (font: string, text: string) => number;

export interface FamilyClass {
  readonly latin: boolean;
  readonly fixed: boolean;
}

export type LocalFontList =
  | { readonly status: 'loading' }
  | { readonly status: 'unavailable' }
  | {
      readonly status: 'ready';
      readonly families: readonly string[];
      readonly latin: readonly string[];
      readonly fixed: readonly string[];
    };

const NAME_PREFIX = 'name:';
const BUNDLED = new Map<string, string>([
  ['Inter Variable', 'Inter'],
  ['Geist Variable', 'Geist'],
  ['Geist Mono Variable', 'Geist Mono'],
]);
const LATIN_SAMPLE = 'Hamburgefonstiv 0123456789';
const MEASURE_SIZE = '16px';
const FAMILIES_PER_FRAME = 32;
const LOADING_VALUE = 'loading';
const LOADING: LocalFontList = { status: 'loading' };
const UNAVAILABLE: LocalFontList = { status: 'unavailable' };
const collator = new Intl.Collator('en', { sensitivity: 'base' });

function compareNames(a: string, b: string): number {
  return collator.compare(a, b) || (a < b ? -1 : a > b ? 1 : 0);
}

export function familiesOf(faces: readonly LocalFontFace[]): string[] {
  const names = new Set<string>();
  for (const face of faces) if (hasNonWhitespace(face.family)) names.add(face.family);
  return [...names].sort(compareNames);
}

function sameWidth(a: number, b: number): boolean {
  return Math.abs(a - b) < 0.01;
}

export function classifyFamily(family: string, measure: MeasureText): FamilyClass {
  const name = cssFamilyName(family);
  const overSerif = `${MEASURE_SIZE} ${name}, serif`;
  const latin = sameWidth(
    measure(overSerif, LATIN_SAMPLE),
    measure(`${MEASURE_SIZE} ${name}, monospace`, LATIN_SAMPLE),
  );
  const fixed =
    latin &&
    sameWidth(measure(overSerif, 'iiiiiiiiii'), measure(overSerif, 'WWWWWWWWWW')) &&
    sameWidth(measure(overSerif, '..........'), measure(overSerif, 'mmmmmmmmmm'));
  return { latin, fixed };
}

export function localFontList(
  faces: readonly LocalFontFace[] | null,
  classes: ReadonlyMap<string, FamilyClass>,
): LocalFontList {
  if (faces === null) return UNAVAILABLE;
  const families = familiesOf(faces);
  if (families.length === 0) return UNAVAILABLE;
  return {
    status: 'ready',
    families,
    latin: families.filter((family) => classes.get(family)?.latin === true),
    fixed: families.filter((family) => classes.get(family)?.fixed === true),
  };
}

function storedName(stored: FontFamily): string | null {
  return stored.startsWith(NAME_PREFIX) ? stored.slice(NAME_PREFIX.length) : null;
}

export function missingFamily(stored: FontFamily, list: LocalFontList): string | null {
  const name = storedName(stored);
  if (name === null || list.status !== 'ready' || BUNDLED.has(name) || list.families.includes(name)) return null;
  return name;
}

function builtInOptions(role: FontRole): ComboboxOption[] {
  if (role === 'code') return [{ value: 'default', label: FONTS.options.geistMono, keywords: ['default', 'built-in'] }];
  return [
    { value: 'default', label: FONTS.options.inter, keywords: ['default', 'built-in'] },
    { value: `${NAME_PREFIX}Geist Variable`, label: FONTS.options.geist, keywords: ['built-in'] },
    { value: 'system', label: FONTS.options.system, keywords: ['sf', 'san francisco', 'segoe'] },
  ];
}

export function fontOptions(role: FontRole, list: LocalFontList, stored: FontFamily): ComboboxOption[] {
  const options = builtInOptions(role);
  const taken = new Set(options.map((option) => option.value));
  const installed = list.status === 'ready' ? (role === 'code' ? list.fixed : list.latin) : [];
  const name = storedName(stored);
  if (name !== null && !taken.has(stored) && !installed.includes(name)) {
    const label =
      missingFamily(stored, list) === null
        ? (BUNDLED.get(name) ?? name)
        : fill(FONTS.options.notInstalled, { family: name });
    options.push({ value: stored, label });
    taken.add(stored);
  }
  for (const family of installed) {
    const value = `${NAME_PREFIX}${family}`;
    if (taken.has(value)) continue;
    taken.add(value);
    options.push({ value, label: family });
  }
  if (list.status === 'loading') options.push({ value: LOADING_VALUE, label: FONTS.options.loading, disabled: true });
  return options;
}

const classes = new Map<string, FamilyClass>();
let current: LocalFontList = LOADING;
let running: Promise<LocalFontList> | null = null;
const listeners = new Set<() => void>();

function publish(next: LocalFontList): void {
  current = next;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getCurrent(): LocalFontList {
  return current;
}

function whenVisible(): Promise<void> {
  if (document.visibilityState === 'visible') return Promise.resolve();
  return new Promise((resolve) => {
    const onChange = (): void => {
      if (document.visibilityState !== 'visible') return;
      document.removeEventListener('visibilitychange', onChange);
      resolve();
    };
    document.addEventListener('visibilitychange', onChange);
  });
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}

type QueryLocalFonts = () => Promise<readonly LocalFontFace[]>;

async function queryFaces(): Promise<readonly LocalFontFace[] | null> {
  const query = (window as { queryLocalFonts?: QueryLocalFonts }).queryLocalFonts;
  if (typeof query !== 'function') return null;
  for (;;) {
    await whenVisible();
    try {
      return await query.call(window);
    } catch {
      // Chromium refuses the call while the page is hidden: wait until it shows again.
      if (document.visibilityState === 'visible') return null;
    }
  }
}

function canvasMeasure(): MeasureText | null {
  const context = document.createElement('canvas').getContext('2d');
  if (context === null) return null;
  return (font, text) => {
    context.font = font;
    return context.measureText(text).width;
  };
}

async function readLocalFonts(): Promise<LocalFontList> {
  const faces = await queryFaces();
  if (faces === null) return UNAVAILABLE;
  const fresh = familiesOf(faces).filter((family) => !classes.has(family));
  if (fresh.length > 0) {
    const measure = canvasMeasure();
    if (measure === null) return UNAVAILABLE;
    for (let start = 0; start < fresh.length; start += FAMILIES_PER_FRAME) {
      await nextFrame();
      for (const family of fresh.slice(start, start + FAMILIES_PER_FRAME)) {
        classes.set(family, classifyFamily(family, measure));
      }
    }
  }
  return localFontList(faces, classes);
}

export function loadLocalFonts(): Promise<LocalFontList> {
  running ??= readLocalFonts()
    .catch(() => UNAVAILABLE)
    .then((list) => {
      running = null;
      publish(list);
      return list;
    });
  return running;
}

export function useLocalFonts(): LocalFontList {
  const list = useSyncExternalStore(subscribe, getCurrent, getCurrent);
  useEffect(() => {
    void loadLocalFonts();
  }, []);
  return list;
}
