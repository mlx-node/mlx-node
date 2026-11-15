import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vite-plus/test';

import { motionX } from '../ui/src/components/arc-x/motion/motion.js';
import { motionTokens } from '../ui/src/components/arc/motion-tokens.js';

const foundation = readFileSync(new URL('../ui/src/components/arc/foundation.css', import.meta.url), 'utf8');
const tokens = readFileSync(new URL('../ui/src/styles/tokens.css', import.meta.url), 'utf8');
const indexHtml = readFileSync(new URL('../ui/index.html', import.meta.url), 'utf8');
const arcDialog = readFileSync(new URL('../ui/src/components/arc/dialog/dialog.module.css', import.meta.url), 'utf8');

/** Top-level rule blocks `selector { body }`, in source order; nested at-rule bodies are skipped. */
function topLevelRules(css: string): Array<{ selector: string; body: string }> {
  const rules: Array<{ selector: string; body: string }> = [];
  let i = 0;
  let selectorStart = 0;
  while (i < css.length) {
    if (css.startsWith('/*', i)) {
      const end = css.indexOf('*/', i + 2);
      i = end === -1 ? css.length : end + 2;
      selectorStart = i;
      continue;
    }
    if (css[i] === ';') selectorStart = i + 1;
    if (css[i] === '{') {
      let depth = 1;
      let j = i + 1;
      while (j < css.length && depth > 0) {
        if (css[j] === '{') depth += 1;
        else if (css[j] === '}') depth -= 1;
        j += 1;
      }
      rules.push({ selector: css.slice(selectorStart, i).trim(), body: css.slice(i + 1, j - 1) });
      i = j;
      selectorStart = j;
      continue;
    }
    i += 1;
  }
  return rules;
}

/** Custom properties declared in every top-level block with exactly this selector; later blocks win. */
function declarations(css: string, selector: string): Map<string, string> {
  const values = new Map<string, string>();
  for (const rule of topLevelRules(css)) {
    if (rule.selector !== selector) continue;
    for (const declaration of rule.body.split(';')) {
      const colon = declaration.indexOf(':');
      if (colon === -1) continue;
      const name = declaration.slice(0, colon).trim();
      if (name.startsWith('--')) values.set(name, declaration.slice(colon + 1).trim());
    }
  }
  return values;
}

/** Follows `var(--x)` through the same declaration maps (the theme block first, then :root). */
function resolve(value: string, ...scopes: Map<string, string>[]): string {
  let current = value;
  for (let guard = 0; guard < 8 && current.startsWith('var(--') && current.endsWith(')'); guard += 1) {
    const name = current.slice('var('.length, -1).trim();
    const next = scopes.map((scope) => scope.get(name)).find((found) => found !== undefined);
    if (next === undefined) break;
    current = next;
  }
  return current;
}

/** `oklch(L% 0 0)` (a gray) to `#rrggbb`: with no chroma, linear sRGB = L^3 on every channel. */
function grayOklchToHex(value: string): string {
  expect(value.startsWith('oklch(') && value.endsWith(')'), value).toBe(true);
  const parts = value.slice('oklch('.length, -1).trim().split(' ');
  expect(parts.slice(1)).toEqual(['0', '0']);
  const lightness = Number(parts[0].slice(0, -1)) / 100;
  const linear = lightness ** 3;
  const encoded = linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055;
  const byte = Math.round(Math.min(1, Math.max(0, encoded)) * 255);
  const hex = byte.toString(16).padStart(2, '0');
  return `#${hex}${hex}${hex}`;
}

/** `35ms` -> 0.035 (seconds), `2px` -> 2. */
function cssNumber(value: string | undefined): number {
  expect(value).toBeDefined();
  const text = value ?? '';
  if (text.endsWith('ms')) return Number(text.slice(0, -2)) / 1000;
  if (text.endsWith('px')) return Number(text.slice(0, -2));
  return Number(text);
}

const arcLight = declarations(foundation, ':root');
const arcDark = declarations(foundation, ':root[data-theme="dark"]');
const ownLight = declarations(tokens, ':root');
const ownDark = declarations(tokens, ":root[data-theme='dark']");

describe('window and boot background', () => {
  it('is Arc --background in both schemes, the same hex the desktop window and the boot screen paint', () => {
    expect(grayOklchToHex(resolve(arcLight.get('--background') ?? '', arcLight))).toBe('#ffffff');
    expect(grayOklchToHex(resolve(arcDark.get('--background') ?? '', arcDark, arcLight))).toBe('#141414');
    expect(indexHtml).toContain('light-dark(#ffffff, #141414)');
  });

  it('follows the OS scheme until a theme is set, so the loaded stylesheet never repaints the boot screen light', () => {
    const rules = topLevelRules(tokens);
    const media = rules.find((rule) => rule.selector === '@media (prefers-color-scheme: dark)');
    expect(media).toBeDefined();
    const inner = topLevelRules(media!.body);
    expect(inner.map((rule) => rule.selector)).toContain(':root:not([data-theme])');
    expect(inner.find((rule) => rule.selector === ':root:not([data-theme])')?.body).toContain('color-scheme: dark');
    const body = rules.find((rule) => rule.selector === ':root:not([data-theme]) body')?.body ?? '';
    expect(body).toContain('background: light-dark(#ffffff, #141414)');
    expect(body).toContain('color: light-dark(#585858, #b4b4b4)');
    expect(indexHtml).toContain('color: light-dark(#585858, #b4b4b4)');
  });
});

describe('density and type overrides', () => {
  it('sets the tighter scale on :root', () => {
    expect(Object.fromEntries([...ownLight].filter(([name]) => name.startsWith('--control-height-')))).toEqual({
      '--control-height-sm': '1.75rem',
      '--control-height-md': '2rem',
      '--control-height-lg': '2.5rem',
    });
    expect(ownLight.get('--radius-control')).toBe('0.625rem');
    expect(ownLight.get('--radius-panel')).toBe('0.875rem');
    expect(ownLight.get('--radius-surface')).toBe('1.25rem');
    expect(ownLight.get('--text-xs')).toBe('0.6875rem');
    expect(ownLight.get('--text-sm')).toBe('0.8125rem');
    expect(ownLight.get('--text-3xl')).toBe('2rem');
  });

  it('only overrides tokens Arc defines, plus its own layer', () => {
    for (const name of ['--control-height-sm', '--radius-control', '--text-sm', '--text-3xl']) {
      expect(arcLight.has(name), name).toBe(true);
    }
  });
});

describe('data colors', () => {
  it('keeps the eight validated slots in order, light and dark', () => {
    const light = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
    const dark = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
    light.forEach((hex, i) => expect(ownLight.get(`--model-${i + 1}`)).toBe(hex));
    dark.forEach((hex, i) => expect(ownDark.get(`--model-${i + 1}`)).toBe(hex));
  });

  it('gives the eight agent lanes the eight slots, each once, and a stronger span tint in dark', () => {
    const lanes = Array.from({ length: 8 }, (_, i) => ownLight.get(`--lane-${i + 1}`));
    expect(new Set(lanes).size).toBe(8);
    expect(lanes.every((value) => value !== undefined && value.startsWith('var(--model-'))).toBe(true);
    expect(ownLight.get('--lane-tint')).toBe('24%');
    expect(ownDark.get('--lane-tint')).toBe('34%');
  });

  it('draws an all-models series in the accent in both themes, so it never looks like the first model', () => {
    expect(ownLight.get('--series-1')).toBe('var(--accent)');
    expect(ownDark.get('--series-1')).toBe('var(--accent)');
  });
});

describe('scrim', () => {
  it('is the Arc dialog overlay: one color and one blur for both themes', () => {
    expect(arcDialog).toContain(`background: ${ownLight.get('--scrim')?.replace('0.46', '.46')};`);
    expect(arcDialog).toContain(`backdrop-filter: blur(${ownLight.get('--scrim-blur')});`);
    expect(ownDark.has('--scrim')).toBe(false);
  });
});

describe('CSS and JS motion values stay equal', () => {
  it('--stagger-item = motionTokens.stagger.item', () => {
    expect(cssNumber(ownLight.get('--stagger-item'))).toBeCloseTo(motionTokens.stagger.item, 6);
  });

  it('--stagger-char = motionTokens.stagger.char', () => {
    expect(cssNumber(ownLight.get('--stagger-char'))).toBeCloseTo(motionTokens.stagger.char, 6);
  });

  it('--duration-trace = motionTokens.duration.considered × 1.75', () => {
    expect(cssNumber(ownLight.get('--duration-trace'))).toBeCloseTo(motionTokens.duration.considered * 1.75, 6);
  });

  it('--duration-spin = motionX.spin', () => {
    expect(cssNumber(ownLight.get('--duration-spin'))).toBeCloseTo(motionX.spin, 6);
  });

  it('--stagger-budget = motionX.staggerBudget', () => {
    expect(cssNumber(ownLight.get('--stagger-budget'))).toBeCloseTo(motionX.staggerBudget, 6);
  });

  it('--blur-subtle = motionTokens.blur.subtle', () => {
    expect(cssNumber(ownLight.get('--blur-subtle'))).toBe(motionTokens.blur.subtle);
  });

  it('--blur-soft = motionTokens.blur.soft', () => {
    expect(cssNumber(ownLight.get('--blur-soft'))).toBe(motionTokens.blur.soft);
  });
});
