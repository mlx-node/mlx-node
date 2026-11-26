import { describe, expect, it } from 'vite-plus/test';

import {
  classifyFamily,
  type FamilyClass,
  familiesOf,
  fontOptions,
  type LocalFontList,
  localFontList,
  type MeasureText,
  missingFamily,
} from '../ui/src/lib/local-fonts.js';

type Widths = (ch: string) => number;

const proportional =
  (table: Readonly<Record<string, number>>, rest: number): Widths =>
  (ch) =>
    table[ch] ?? rest;

const DRAWN: Readonly<Record<string, Widths>> = {
  Menlo: () => 7.2,
  'Geist Variable': proportional({ i: 3, W: 11, '.': 3, m: 10 }, 6),
  'Helvetica Neue': proportional({ i: 3, W: 11, '.': 3, m: 10 }, 6),
  'Half Mono': proportional({ i: 6, W: 6, '.': 4, m: 6 }, 6),
};
const FALLBACK: Readonly<Record<string, Widths>> = {
  serif: proportional({ i: 3, W: 12, '.': 3, m: 11 }, 7),
  monospace: () => 8,
};

const measure: MeasureText = (font, text) => {
  const open = font.indexOf('"');
  const close = font.indexOf('"', open + 1);
  const family = font.slice(open + 1, close);
  const fallback = font.slice(font.lastIndexOf(', ') + 2);
  const widths = DRAWN[family] ?? FALLBACK[fallback];
  if (open === -1 || close === -1 || widths === undefined) throw new Error(`unexpected font ${font}`);
  let width = 0;
  for (const ch of text) width += widths(ch);
  return width;
};

const FAMILIES = ['Menlo', 'Helvetica Neue', 'Half Mono', 'Noto Sans Arabic'];

function readyList(families: readonly string[] = FAMILIES): LocalFontList {
  const classes = new Map<string, FamilyClass>(families.map((family) => [family, classifyFamily(family, measure)]));
  return localFontList(
    families.map((family) => ({ family })),
    classes,
  );
}

const LOADING: LocalFontList = { status: 'loading' };

function labels(list: LocalFontList, role: 'reading' | 'code' | 'ui', stored: Parameters<typeof fontOptions>[2]) {
  return fontOptions(role, list, stored).map((option) => option.label);
}

describe('local fonts', () => {
  it('L1: merges the faces of one family and sorts names A to Z without case', () => {
    expect(
      familiesOf([{ family: 'Helvetica Neue' }, { family: 'Helvetica Neue' }, { family: 'Helvetica Neue' }]),
    ).toEqual(['Helvetica Neue']);
    expect(
      familiesOf([{ family: 'Zapfino' }, { family: 'apple braille' }, { family: 'Menlo' }, { family: 'Avenir Next' }]),
    ).toEqual(['apple braille', 'Avenir Next', 'Menlo', 'Zapfino']);
  });

  it('L2: offers no family whose two fallback widths differ', () => {
    expect(classifyFamily('Noto Sans Arabic', measure)).toEqual({ latin: false, fixed: false });
    const list = readyList();
    for (const role of ['reading', 'code', 'ui'] as const) {
      expect(labels(list, role, 'default'), role).not.toContain('Noto Sans Arabic');
    }
  });

  it('L3: puts a proportional family in Reading and not in Code', () => {
    expect(classifyFamily('Helvetica Neue', measure)).toEqual({ latin: true, fixed: false });
    const list = readyList();
    expect(labels(list, 'reading', 'default')).toContain('Helvetica Neue');
    expect(labels(list, 'code', 'default')).not.toContain('Helvetica Neue');
    expect(labels(list, 'code', 'default')).toContain('Menlo');
  });

  it('L4: needs both pairs to match before a family is fixed width', () => {
    expect(classifyFamily('Half Mono', measure)).toEqual({ latin: true, fixed: false });
    expect(labels(readyList(), 'code', 'default')).not.toContain('Half Mono');
  });

  it('L5: offers neither System font nor Geist for code', () => {
    const code = fontOptions('code', readyList(), 'default');
    expect(code.map((option) => option.value)).not.toContain('system');
    expect(code.map((option) => option.value)).not.toContain('name:Geist Variable');
    expect(code.map((option) => option.label)).not.toContain('System font');
    expect(code.map((option) => option.label)).not.toContain('Geist');
    expect(code[0]).toMatchObject({ value: 'default', label: 'Geist Mono (default)' });
    expect(labels(readyList(), 'reading', 'default').slice(0, 3)).toEqual(['Inter (default)', 'Geist', 'System font']);
  });

  it('L6: keeps a missing stored font in the field, and calls nothing missing while the list loads', () => {
    const ready = readyList();
    expect(fontOptions('code', ready, 'name:Fira Code')).toContainEqual({
      value: 'name:Fira Code',
      label: 'Fira Code (not installed)',
    });
    expect(missingFamily('name:Fira Code', ready)).toBe('Fira Code');

    const loading = fontOptions('code', LOADING, 'name:Fira Code');
    expect(loading).toContainEqual({ value: 'name:Fira Code', label: 'Fira Code' });
    expect(loading.at(-1)).toMatchObject({ label: 'Loading installed fonts…', disabled: true });
    expect(missingFamily('name:Fira Code', LOADING)).toBeNull();
  });

  it('L7: never calls a built-in or an installed font missing', () => {
    const ready = readyList();
    expect(missingFamily('name:Geist Variable', ready)).toBeNull();
    for (const role of ['reading', 'code', 'ui'] as const) {
      const shown = labels(ready, role, 'name:Geist Variable');
      expect(
        shown.filter((label) => label.endsWith('(not installed)')),
        role,
      ).toEqual([]);
    }
    expect(missingFamily('name:Helvetica Neue', ready)).toBeNull();
    expect(fontOptions('code', ready, 'name:Helvetica Neue')).toContainEqual({
      value: 'name:Helvetica Neue',
      label: 'Helvetica Neue',
    });
  });

  it('L8: reads an empty or failed list as not available', () => {
    expect(localFontList([], new Map())).toEqual({ status: 'unavailable' });
    expect(localFontList(null, new Map())).toEqual({ status: 'unavailable' });
    expect(readyList().status).toBe('ready');
    expect(labels({ status: 'unavailable' }, 'reading', 'default')).toEqual([
      'Inter (default)',
      'Geist',
      'System font',
    ]);
  });

  it('lists an installed family that shares a built-in name once', () => {
    const values = fontOptions('reading', readyList(['Geist Variable', 'Menlo']), 'default').map(
      (option) => option.value,
    );
    expect(values).toEqual(['default', 'name:Geist Variable', 'system', 'name:Menlo']);
  });
});
