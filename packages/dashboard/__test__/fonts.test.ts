/**
 * @vitest-environment happy-dom
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  cssFamilyName,
  FONT_DEFAULTS,
  FONT_FALLBACK,
  FONTS_STORAGE_KEY,
  fontProperties,
  type FontSettings,
  initFonts,
  isDefaultFonts,
  parseStoredFonts,
  repairFontSettings,
  resetFonts,
  setFont,
  STOCK_TYPE,
  stockTypeProperties,
} from '../ui/src/theme/fonts.js';
import { cssRules, cssRulesOf, cssValue } from './helpers/css-rules.js';

const root = document.documentElement;

function settings(change: Partial<Record<keyof FontSettings, Partial<FontSettings['reading']>>>): FontSettings {
  return {
    reading: { ...FONT_DEFAULTS.reading, ...change.reading },
    code: { ...FONT_DEFAULTS.code, ...change.code },
    ui: { ...FONT_DEFAULTS.ui, ...change.ui },
  };
}

function inlineNames(): string[] {
  const names: string[] = [];
  for (let i = 0; i < root.style.length; i += 1) names.push(root.style.item(i));
  return names;
}

/** `0.6875rem` -> 11 (px at a 16 px root). */
function remPx(value: string | undefined): number {
  expect(value?.endsWith('rem'), value).toBe(true);
  return Number((value ?? '').slice(0, -'rem'.length)) * 16;
}

beforeEach(() => {
  localStorage.clear();
  root.removeAttribute('style');
  initFonts();
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
  resetFonts();
  root.removeAttribute('style');
});

describe('font settings, pure', () => {
  it('the defaults write no property', () => {
    expect(fontProperties(FONT_DEFAULTS)).toEqual({});
    expect(isDefaultFonts(FONT_DEFAULTS)).toBe(true);
    expect(isDefaultFonts(settings({ code: { size: 13 } }))).toBe(false);
  });

  it('a size is rounded, then held to its range; a size that is not a number is the default', () => {
    const reading = (size: unknown): number =>
      repairFontSettings({ reading: { family: 'default', size } }).reading.size;
    expect([99, 3, 13.6, Number.NaN, '13'].map(reading)).toEqual([24, 11, 14, 13, 13]);
    expect(repairFontSettings({ code: { size: 9 } }).code.size).toBe(10);
    expect(repairFontSettings({ ui: { size: 16 } }).ui.size).toBe(15);
  });

  it('a bad field never resets the others', () => {
    const repaired = repairFontSettings({
      reading: { family: 'name:Avenir Next', size: 18 },
      code: { family: 42, size: 14 },
      ui: { family: 'system', size: 15 },
    });
    expect(repaired).toEqual({
      reading: { family: 'name:Avenir Next', size: 18 },
      code: { family: 'default', size: 14 },
      ui: { family: 'system', size: 15 },
    });
    expect(repairFontSettings('sepia')).toEqual(FONT_DEFAULTS);
    expect(repairFontSettings([1, 2])).toEqual(FONT_DEFAULTS);
    expect(repairFontSettings({ reading: 'big' }).reading).toEqual(FONT_DEFAULTS.reading);
  });

  it('code has no System font', () => {
    expect(repairFontSettings({ code: { family: 'system', size: 12 } }).code.family).toBe('default');
    expect(repairFontSettings({ reading: { family: 'system', size: 13 } }).reading.family).toBe('system');
    expect(repairFontSettings({ ui: { family: 'system', size: 13 } }).ui.family).toBe('system');
  });

  it('a family name is escaped into a CSS string; an empty, long or control-character name is the default', () => {
    expect(cssFamilyName('A"B\\C')).toBe('"A\\"B\\\\C"');
    expect(cssFamilyName('Fira Code')).toBe('"Fira Code"');
    const family = (value: string): string => repairFontSettings({ reading: { family: value } }).reading.family;
    expect(family('name:Fira\nCode')).toBe('default');
    expect(family('name:Fira\u0007Code')).toBe('default');
    expect(family(`name:${'a'.repeat(129)}`)).toBe('default');
    expect(family(`name:${'a'.repeat(128)}`)).toBe(`name:${'a'.repeat(128)}`);
    expect(family('name:')).toBe('default');
    expect(family('name:   ')).toBe('default');
    expect(family('serif')).toBe('default');
    expect(family('name:A"B\\C')).toBe('name:A"B\\C');
  });

  it('an Interface font sets the body and display fonts and both trackings', () => {
    expect(fontProperties(settings({ ui: { family: 'name:Avenir Next' } }))).toEqual({
      '--font-body': '"Avenir Next", "Inter Variable", sans-serif',
      '--font-display': '"Avenir Next", "Inter Variable", sans-serif',
      '--tracking-body': '0',
      '--tracking-display': '0',
    });
    expect(fontProperties(settings({ ui: { family: 'system' } }))).toEqual({
      '--font-body': 'system-ui, sans-serif',
      '--font-display': 'system-ui, sans-serif',
      '--tracking-body': '0',
      '--tracking-display': '0',
    });
  });

  it('a Reading font other than the default drops the tracking', () => {
    expect(fontProperties(settings({ reading: { family: 'system' } }))).toEqual({
      '--font-reading': 'system-ui, sans-serif',
      '--tracking-reading': '0',
    });
    expect(fontProperties(settings({ reading: { family: 'name:Geist Variable' } }))).toEqual({
      '--font-reading': '"Geist Variable", "Inter Variable", sans-serif',
      '--tracking-reading': '0',
    });
  });

  it('an Interface size scales all nine text tokens by n/13', () => {
    const props = fontProperties(settings({ ui: { size: 15 } }));
    const names = Object.keys(STOCK_TYPE).filter((name) => name.startsWith('--text-'));
    expect(names).toEqual([
      '--text-xs',
      '--text-sm',
      '--text-base',
      '--text-lg',
      '--text-xl',
      '--text-2xl',
      '--text-3xl',
      '--text-mono',
      '--text-field',
    ]);
    expect(Object.keys(props).sort()).toEqual([...names].sort());
    for (const name of names) {
      expect(remPx(props[name]), name).toBeCloseTo((remPx(STOCK_TYPE[name]) * 15) / 13, 2);
    }
    expect(remPx(props['--text-xs'])).toBeCloseTo(12.69, 2);
  });

  it('a Reading or Code size moves its own token only', () => {
    expect(fontProperties(settings({ reading: { size: 20 } }))).toEqual({ '--text-reading': '1.25rem' });
    expect(fontProperties(settings({ code: { size: 14 } }))).toEqual({ '--text-code': '0.875rem' });
  });

  it('every chosen family ends in the bundled font of its role', () => {
    expect(fontProperties(settings({ reading: { family: 'name:Fira Sans' } }))['--font-reading']).toBe(
      '"Fira Sans", "Inter Variable", sans-serif',
    );
    expect(fontProperties(settings({ code: { family: 'name:Fira Code' } }))).toEqual({
      '--font-code': '"Fira Code", "Geist Mono Variable", Menlo, monospace',
    });
  });

  it('the stock copy equals the stylesheets, and the role defaults equal the tokens', () => {
    const tokens = cssRules('styles/tokens.css');
    const foundation = cssRules('components/arc/foundation.css');
    const stylesheet = (name: string): string | undefined =>
      cssValue(tokens, ':root', name) ?? cssValue(foundation, ':root', name);
    expect(Object.keys(STOCK_TYPE)).toHaveLength(13);
    for (const [name, value] of Object.entries(STOCK_TYPE)) expect(stylesheet(name), name).toBe(value);
    expect(stockTypeProperties()).toEqual(STOCK_TYPE);
    expect(remPx(cssValue(tokens, ':root', '--text-reading'))).toBe(FONT_DEFAULTS.reading.size);
    expect(remPx(cssValue(tokens, ':root', '--text-code'))).toBe(FONT_DEFAULTS.code.size);
    expect(remPx(cssValue(tokens, ':root', '--text-sm'))).toBe(FONT_DEFAULTS.ui.size);
    expect(cssValue(tokens, ':root', '--font-reading')?.replaceAll("'", '"')).toBe(FONT_FALLBACK.reading);
    expect(cssValue(tokens, ':root', '--tracking-reading')).toBe('-0.01em');
  });
});

describe('font settings, applied', () => {
  it('a reset removes every property it wrote and stores the defaults', () => {
    setFont('reading', { family: 'name:Avenir Next' });
    setFont('reading', { size: 18 });
    setFont('code', { family: 'name:Fira Code' });
    setFont('code', { size: 14 });
    setFont('ui', { family: 'system' });
    setFont('ui', { size: 15 });
    expect(root.style.getPropertyValue('--text-reading')).toBe('1.125rem');
    expect(root.style.getPropertyValue('--font-code')).toBe('"Fira Code", "Geist Mono Variable", Menlo, monospace');
    expect(inlineNames()).toHaveLength(18);
    expect(parseStoredFonts(localStorage.getItem(FONTS_STORAGE_KEY))).toEqual({
      reading: { family: 'name:Avenir Next', size: 18 },
      code: { family: 'name:Fira Code', size: 14 },
      ui: { family: 'system', size: 15 },
    });

    resetFonts();
    expect(inlineNames()).toEqual([]);
    expect(parseStoredFonts(localStorage.getItem(FONTS_STORAGE_KEY))).toEqual(FONT_DEFAULTS);
  });

  it('a size back at its default removes its property', () => {
    setFont('reading', { size: 16 });
    expect(inlineNames()).toEqual(['--text-reading']);
    setFont('reading', { size: 13 });
    expect(inlineNames()).toEqual([]);
  });

  it('a value written in another window applies here; a write under another key does not', () => {
    localStorage.setItem(FONTS_STORAGE_KEY, JSON.stringify(settings({ reading: { size: 20 } })));
    window.dispatchEvent(new StorageEvent('storage', { key: FONTS_STORAGE_KEY }));
    expect(root.style.getPropertyValue('--text-reading')).toBe('1.25rem');

    localStorage.setItem(FONTS_STORAGE_KEY, JSON.stringify(settings({ reading: { size: 22 } })));
    window.dispatchEvent(new StorageEvent('storage', { key: 'mlx-node:theme:v1' }));
    expect(root.style.getPropertyValue('--text-reading')).toBe('1.25rem');
  });

  it('boot survives a bad stored value and storage that throws; a change still applies', () => {
    expect(parseStoredFonts('{"reading":')).toEqual(FONT_DEFAULTS);
    expect(parseStoredFonts(null)).toEqual(FONT_DEFAULTS);
    localStorage.setItem(FONTS_STORAGE_KEY, '{"reading":');
    initFonts();
    expect(inlineNames()).toEqual([]);

    const blocked = (): never => {
      throw new Error('blocked');
    };
    vi.stubGlobal('localStorage', { getItem: blocked, setItem: blocked, removeItem: blocked, clear: blocked });
    expect(() => initFonts()).not.toThrow();
    expect(inlineNames()).toEqual([]);
    setFont('reading', { size: 18 });
    expect(root.style.getPropertyValue('--text-reading')).toBe('1.125rem');
    resetFonts();
    expect(inlineNames()).toEqual([]);
  });
});

describe('font settings, stylesheets', () => {
  it('the reading surfaces draw in the three reading tokens; headings and inline code follow their line', () => {
    const reading = (path: string, selector: string): void => {
      const rules = cssRules(path);
      expect(cssValue(rules, selector, 'font-family'), `${path} ${selector}`).toBe('var(--font-reading)');
      expect(cssValue(rules, selector, 'font-size'), `${path} ${selector}`).toBe('var(--text-reading)');
      expect(cssValue(rules, selector, 'letter-spacing'), `${path} ${selector}`).toBe('var(--tracking-reading)');
    };
    reading('components/arc-x/chat-markdown/chat-markdown.module.css', '.root');
    reading('components/arc-x/message/message.module.css', '.body');
    reading('components/arc-x/composer/composer.module.css', '.input');
    reading('components/arc-x/reasoning-block/reasoning-block.module.css', '.tail');
    reading('components/arc-x/reasoning-block/reasoning-block.module.css', '.body');

    const markdown = cssRules('components/arc-x/chat-markdown/chat-markdown.module.css');
    expect(cssValue(markdown, '.h1', 'font-size')).toBe('calc(1em * 16 / 13)');
    expect(cssValue(markdown, '.h2', 'font-size')).toBe('calc(1em * 14 / 13)');
    expect(cssValue(markdown, '.h3', 'font-size')).toBe('calc(1em * 14 / 13)');
    expect(cssValue(markdown, '.inlineCode', 'font-size')).toBe('calc(1em * 12 / 13)');
    expect(cssValue(markdown, '.inlineCode', 'font-family')).toBe('var(--font-code)');
  });

  it('both boot screens draw one fixed font that no setting reaches', () => {
    const html = readFileSync(join(import.meta.dirname, '..', 'ui', 'index.html'), 'utf8');
    const open = html.indexOf('<style>');
    const close = html.indexOf('</style>');
    expect(open).toBeGreaterThan(-1);
    expect(close).toBeGreaterThan(open);
    const boot = cssRulesOf(html.slice(open + '<style>'.length, close));
    const shell = cssRules('shell/app-shell.module.css');
    const fixed = '13px/1.4 system-ui, BlinkMacSystemFont, sans-serif';
    expect(cssValue(boot, '.mlx-boot', 'font')).toBe(fixed);
    expect(cssValue(boot, '.mlx-boot', 'letter-spacing')).toBe('normal');
    expect(cssValue(shell, '.boot', 'font')).toBe(fixed);
    expect(cssValue(shell, '.boot', 'letter-spacing')).toBe('normal');
    expect(html.includes('-apple-system')).toBe(false);
  });

  it('mono text in controls takes the interface mono size, not the code size', () => {
    expect(
      cssValue(cssRules('components/arc-x/key-value/key-value.module.css'), '.value[data-mono] .text', 'font-size'),
    ).toBe('var(--text-mono)');
    expect(
      cssValue(cssRules('components/arc-x/secret-field/secret-field.module.css'), '.input[data-shown]', 'font-size'),
    ).toBe('var(--text-mono)');
  });
});
