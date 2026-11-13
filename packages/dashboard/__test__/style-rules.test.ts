/**
 * The style contract of the redesign, as string scans (no regex):
 * - pages use Tailwind for layout and text only, with the locked theme's keys;
 * - own components (`components/arc-x`) and the shell use CSS modules only;
 * - motion values come from tokens, springs only from `arc-x/motion/motion.ts`, and every own animated
 *   stylesheet carries a reduced-motion block.
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import { splitWhitespace } from '../ui/src/lib/text-scan.js';
import { cssRules, cssValue } from './helpers/css-rules.js';

const PACKAGE = fileURLToPath(new URL('..', import.meta.url));
const UI = join(PACKAGE, 'ui/src');

/** Vendored Arc files: under Arc's own rules, not ours. */
const VENDORED = join(UI, 'components/arc');

/** CSS-modules-only folders. */
const MODULE_ONLY_ROOTS: readonly string[] = [join(UI, 'components/arc-x'), join(UI, 'shell')];

const within = (path: string, root: string): boolean => path === root || path.startsWith(`${root}/`);

/** Every own script: all of `ui/src` but the vendored Arc files. */
const ownScripts = (): string[] => walk(UI, isScript).filter((file) => !within(file, VENDORED));

/** Scripts under the pages rule: every own script outside the CSS-modules folders (pages, composites, lib, App). */
const pageRuleScripts = (): string[] =>
  ownScripts().filter((file) => !MODULE_ONLY_ROOTS.some((root) => within(file, root)));

/** Own stylesheets under the token, motion and paint scans: every CSS module in `ui/src` but the vendored ones. */
const ownModuleCss = (): string[] => walk(UI, isModuleCss).filter((file) => !within(file, VENDORED));

const MOTION_FILE = join(UI, 'components/arc-x/motion/motion.ts');

/** The one folder allowed to mount Arc TextShimmer, and the vendored item itself. */
const SHIMMER_HOME = join(UI, 'components/arc-x/live-label');
const SHIMMER_ITEM = join(UI, 'components/arc/text-shimmer');

function walk(root: string, keep: (path: string) => boolean): string[] {
  if (!existsSync(root)) return [];
  if (!statSync(root).isDirectory()) return keep(root) ? [root] : [];
  const files: string[] = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) files.push(...walk(path, keep));
    else if (keep(path)) files.push(path);
  }
  return files;
}

const isScript = (path: string): boolean => (path.endsWith('.ts') || path.endsWith('.tsx')) && !path.endsWith('.d.ts');
const isModuleCss = (path: string): boolean => path.endsWith('.module.css');

/** The `--color-*` and `--text-*` keys of the locked Tailwind theme, without the prefix. */
function themeKeys(): { colors: Set<string>; texts: Set<string> } {
  const css = readFileSync(join(UI, 'styles/tailwind-theme.css'), 'utf8');
  const colors = new Set<string>();
  const texts = new Set<string>();
  for (const raw of css.split('\n')) {
    const line = raw.trim();
    const colon = line.indexOf(':');
    if (colon === -1) continue;
    const name = line.slice(0, colon);
    if (name.startsWith('--color-')) colors.add(name.slice('--color-'.length));
    else if (name.startsWith('--text-')) texts.add(name.slice('--text-'.length));
  }
  return { colors, texts };
}

/** Contents of the plain string literals in `text` (quotes and template strings without `${`). */
function stringLiterals(text: string): string[] {
  const found: string[] = [];
  let i = 0;
  while (i < text.length) {
    const quote = text[i];
    if (quote === "'" || quote === '"' || quote === '`') {
      let j = i + 1;
      while (j < text.length && text[j] !== quote) j += text[j] === '\\' ? 2 : 1;
      const body = text.slice(i + 1, j);
      if (!(quote === '`' && body.includes('${'))) found.push(body);
      i = j + 1;
      continue;
    }
    i += 1;
  }
  return found;
}

/** Every class string handed to `className=`: a quoted value, or the string literals inside `{…}`. */
function classStrings(source: string): string[] {
  const strings: string[] = [];
  let at = source.indexOf('className=');
  while (at !== -1) {
    const start = at + 'className='.length;
    const open = source[start];
    if (open === '"' || open === "'") {
      const end = source.indexOf(open, start + 1);
      strings.push(source.slice(start + 1, end === -1 ? source.length : end));
    } else if (open === '{') {
      let depth = 1;
      let j = start + 1;
      while (j < source.length && depth > 0) {
        if (source[j] === '{') depth += 1;
        else if (source[j] === '}') depth -= 1;
        j += 1;
      }
      strings.push(...stringLiterals(source.slice(start + 1, j - 1)));
    }
    at = source.indexOf('className=', start);
  }
  return strings;
}

const TEXT_WORDS = new Set([
  'left',
  'center',
  'right',
  'start',
  'end',
  'justify',
  'wrap',
  'nowrap',
  'balance',
  'pretty',
  'ellipsis',
  'clip',
]);
const FORBIDDEN_PREFIXES = ['bg-', 'border-', 'rounded-', 'shadow-', 'ring-', 'outline-', 'animate-'];
const UNUSED_PREFIXES = [
  'from-',
  'via-',
  'to-',
  'fill-',
  'stroke-',
  'decoration-',
  'caret-',
  'divide-',
  'placeholder-',
];
const ARBITRARY_COLORS = ['[#', '[oklch', '[rgb'];

/** Why a page class token breaks the rule, or null. */
function pageTokenProblem(token: string, keys: { colors: Set<string>; texts: Set<string> }): string | null {
  const variants = token.split(':');
  if (variants.slice(0, -1).includes('dark')) return 'dark: does not exist';
  const base = variants[variants.length - 1];
  const utility = base.includes('/') ? base.slice(0, base.indexOf('/')) : base;
  if (ARBITRARY_COLORS.some((needle) => utility.includes(needle))) return 'arbitrary color';
  if (FORBIDDEN_PREFIXES.some((prefix) => utility.startsWith(prefix))) return 'pages do not paint surfaces';
  if (UNUSED_PREFIXES.some((prefix) => utility.startsWith(prefix))) return 'not in the locked theme';
  if (utility.startsWith('text-')) {
    const rest = utility.slice('text-'.length);
    if (rest.startsWith('[')) return 'arbitrary text value';
    if (!keys.colors.has(rest) && !keys.texts.has(rest) && !TEXT_WORDS.has(rest)) return 'text key not in the theme';
  }
  return null;
}

function pageProblems(path: string, source: string, keys: { colors: Set<string>; texts: Set<string> }): string[] {
  const problems: string[] = [];
  for (const value of classStrings(source)) {
    for (const token of splitWhitespace(value)) {
      const problem = pageTokenProblem(token, keys);
      if (problem !== null) problems.push(`${path}: ${token} (${problem})`);
    }
  }
  return problems;
}

/** `className="…"` / `className='…'` in a CSS-modules-only folder. */
function literalClassNames(path: string, source: string): string[] {
  const problems: string[] = [];
  source.split('\n').forEach((line, index) => {
    if (line.includes('className="') || line.includes("className='")) problems.push(`${path}:${index + 1}`);
  });
  return problems;
}

/** The line with every `var(…)` removed. */
function withoutVars(line: string): string {
  let out = '';
  let i = 0;
  while (i < line.length) {
    if (line.startsWith('var(', i)) {
      let depth = 1;
      let j = i + 'var('.length;
      while (j < line.length && depth > 0) {
        if (line[j] === '(') depth += 1;
        else if (line[j] === ')') depth -= 1;
        j += 1;
      }
      out += ' ';
      i = j;
      continue;
    }
    out += line[i];
    i += 1;
  }
  return out;
}

function isDigit(ch: string | undefined): boolean {
  return ch !== undefined && ch >= '0' && ch <= '9';
}

function isLetter(ch: string | undefined): boolean {
  return ch !== undefined && ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'));
}

/** A digit followed by `ms`, or by `s` that ends the word, outside `var(…)`. */
function hasDurationLiteral(line: string): boolean {
  const text = withoutVars(line);
  for (let i = 0; i < text.length; i += 1) {
    if (!isDigit(text[i])) continue;
    if (text.startsWith('ms', i + 1)) return true;
    if (text[i + 1] === 's' && !isLetter(text[i + 2]) && !isDigit(text[i + 2])) return true;
  }
  return false;
}

function durationProblems(path: string, css: string): string[] {
  const problems: string[] = [];
  css.split('\n').forEach((line, index) => {
    if (hasDurationLiteral(line)) problems.push(`${path}:${index + 1}: ${line.trim()}`);
  });
  return problems;
}

function springProblems(path: string, source: string): string[] {
  return ['stiffness', 'visualDuration', 'damping']
    .filter((word) => source.includes(word))
    .map((word) => `${path}: ${word}`);
}

interface CssRule {
  selectors: string[];
  decls: Map<string, string>;
  /** Inside `@media (prefers-reduced-motion: reduce)`. */
  reduce: boolean;
  /** Inside `@media (prefers-reduced-motion: no-preference)`. */
  motionOnly: boolean;
}

function stripComments(css: string): string {
  let out = '';
  let i = 0;
  while (i < css.length) {
    if (css[i] === '/' && css[i + 1] === '*') {
      const end = css.indexOf('*/', i + 2);
      i = end === -1 ? css.length : end + 2;
      continue;
    }
    out += css[i];
    i += 1;
  }
  return out;
}

/** Index of the `}` that closes the block opened just before `from`. */
function blockEnd(css: string, from: number): number {
  let depth = 1;
  let quote = '';
  for (let i = from; i < css.length; i += 1) {
    const ch = css[i];
    if (quote !== '') {
      if (ch === quote) quote = '';
    } else if (ch === '"' || ch === "'") quote = ch;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return css.length;
}

/** Split at `sep` outside brackets, parentheses and quotes. */
function splitTop(text: string, sep: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let quote = '';
  let start = 0;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quote !== '') {
      if (ch === quote) quote = '';
    } else if (ch === '"' || ch === "'") quote = ch;
    else if (ch === '(' || ch === '[') depth += 1;
    else if (ch === ')' || ch === ']') depth -= 1;
    else if (ch === sep && depth === 0) {
      parts.push(text.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(text.slice(start));
  return parts;
}

/** At-rules whose body holds style rules; every other at-rule block (`@keyframes`, `@font-face`) is skipped. */
const GROUPING_AT_RULES = new Set(['@media', '@container', '@supports', '@layer', '@scope', '@starting-style']);

/** `@media` for `@media (x)`, '' for a selector. */
function atRuleName(prelude: string): string {
  if (!prelude.startsWith('@')) return '';
  let end = 1;
  while (end < prelude.length && isNameChar(prelude[end])) end += 1;
  return prelude.slice(0, end);
}

function parseRules(source: string, reduce = false, motionOnly = false): CssRule[] {
  const css = stripComments(source);
  const rules: CssRule[] = [];
  let i = 0;
  while (i < css.length) {
    const open = css.indexOf('{', i);
    if (open === -1) break;
    const statements = splitTop(css.slice(i, open), ';');
    const prelude = statements[statements.length - 1].trim();
    const end = blockEnd(css, open + 1);
    const body = css.slice(open + 1, end);
    const atRule = atRuleName(prelude);
    if (atRule !== '' && atRule !== '@media' && GROUPING_AT_RULES.has(atRule)) {
      rules.push(...parseRules(body, reduce, motionOnly));
    } else if (atRule === '@media') {
      const media = splitWhitespace(prelude).join(' ');
      rules.push(
        ...parseRules(
          body,
          reduce || media.includes('prefers-reduced-motion: reduce'),
          motionOnly || media.includes('prefers-reduced-motion: no-preference'),
        ),
      );
    } else if (!prelude.startsWith('@')) {
      const decls = new Map<string, string>();
      for (const decl of splitTop(body, ';')) {
        const colon = decl.indexOf(':');
        if (colon === -1) continue;
        decls.set(decl.slice(0, colon).trim(), decl.slice(colon + 1).trim());
      }
      const selectors = splitTop(prelude, ',').map((selector) => splitWhitespace(selector).join(' '));
      rules.push({ selectors, decls, reduce, motionOnly });
    }
    i = end + 1;
  }
  return rules;
}

type Specificity = [number, number, number];

const isNameChar = (ch: string | undefined): boolean =>
  ch !== undefined && (isLetter(ch) || isDigit(ch) || ch === '-' || ch === '_');

function specificity(selector: string): Specificity {
  const total: Specificity = [0, 0, 0];
  let i = 0;
  const readName = (): string => {
    const start = i;
    while (isNameChar(selector[i])) i += 1;
    return selector.slice(start, i);
  };
  while (i < selector.length) {
    const ch = selector[i];
    if (ch === '#') {
      i += 1;
      readName();
      total[0] += 1;
    } else if (ch === '.') {
      i += 1;
      readName();
      total[1] += 1;
    } else if (ch === '[') {
      const close = selector.indexOf(']', i);
      i = close === -1 ? selector.length : close + 1;
      total[1] += 1;
    } else if (ch === ':' && selector[i + 1] === ':') {
      i += 2;
      readName();
      total[2] += 1;
    } else if (ch === ':') {
      i += 1;
      const name = readName();
      if (selector[i] === '(') {
        let depth = 1;
        const start = i + 1;
        i += 1;
        while (i < selector.length && depth > 0) {
          if (selector[i] === '(') depth += 1;
          else if (selector[i] === ')') depth -= 1;
          i += 1;
        }
        const inner = selector.slice(start, i - 1);
        if (name === 'where') continue;
        if (name === 'not' || name === 'is' || name === 'has') {
          const best = splitTop(inner, ',')
            .map(specificity)
            .reduce<Specificity>((a, b) => (compareSpecificity(a, b) >= 0 ? a : b), [0, 0, 0]);
          total[0] += best[0];
          total[1] += best[1];
          total[2] += best[2];
          continue;
        }
      }
      total[1] += 1;
    } else if (isLetter(ch)) {
      readName();
      total[2] += 1;
    } else {
      i += 1;
    }
  }
  return total;
}

function compareSpecificity(a: Specificity, b: Specificity): number {
  for (let k = 0; k < 3; k += 1) if (a[k] !== b[k]) return a[k] - b[k];
  return 0;
}

/** The element a selector paints: the first class of its last compound plus its pseudo-element. */
function subject(selector: string): string {
  const compounds = splitTop(selector.replaceAll('>', ' ').replaceAll('+', ' ').replaceAll('~', ' '), ' ').filter(
    (part) => part.length > 0,
  );
  const last = compounds.at(-1) ?? '';
  let className = '';
  let pseudo = '';
  let i = 0;
  while (i < last.length) {
    if (last[i] === '.' && className === '') {
      const start = i;
      i += 1;
      while (isNameChar(last[i])) i += 1;
      className = last.slice(start, i);
    } else if (last[i] === ':' && last[i + 1] === ':') {
      const start = i;
      i += 2;
      while (isNameChar(last[i])) i += 1;
      pseudo = last.slice(start, i);
    } else {
      i += 1;
    }
  }
  return className + pseudo;
}

function startsAnimation(decls: Map<string, string>): boolean {
  const shorthand = decls.get('animation');
  const name = decls.get('animation-name');
  return (shorthand !== undefined && shorthand !== 'none') || (name !== undefined && name !== 'none');
}

const REDUCE_KEYS = ['animation', 'animation-name', 'animation-duration'];

/**
 * Selectors that start a keyframe animation outside a no-preference block, with no reduced-motion rule for the same
 * element at equal or higher specificity: under reduce the animation would still run.
 */
function unreducedAnimations(file: string, css: string): string[] {
  const rules = parseRules(css);
  const reducers = rules.filter(
    (rule) => rule.reduce && (REDUCE_KEYS.some((key) => rule.decls.has(key)) || rule.decls.get('display') === 'none'),
  );
  const problems: string[] = [];
  rules.forEach((rule, index) => {
    if (rule.reduce || rule.motionOnly || !startsAnimation(rule.decls)) return;
    for (const selector of rule.selectors) {
      const target = subject(selector);
      const own = specificity(selector);
      const covered = reducers.some((reducer) => {
        if (rules.indexOf(reducer) < index) return false;
        return reducer.selectors.some(
          (candidate) => subject(candidate) === target && compareSpecificity(specificity(candidate), own) >= 0,
        );
      });
      if (!covered) problems.push(`${file}: ${selector}`);
    }
  });
  return problems;
}

/** The pseudo-element ('' for the element itself) of every selector that runs an infinite animation. */
function loopPseudos(css: string): string[] {
  const pseudos: string[] = [];
  for (const rule of parseRules(css)) {
    const shorthand = splitWhitespace((rule.decls.get('animation') ?? '').replaceAll(',', ' '));
    const count = splitWhitespace((rule.decls.get('animation-iteration-count') ?? '').replaceAll(',', ' '));
    if (!shorthand.includes('infinite') && !count.includes('infinite')) continue;
    for (const selector of rule.selectors) {
      const target = subject(selector);
      const at = target.indexOf('::');
      const pseudo = at === -1 ? '' : target.slice(at);
      if (!pseudos.includes(pseudo)) pseudos.push(pseudo);
    }
  }
  return pseudos;
}

const COLOR_FUNCTIONS = ['rgb(', 'rgba(', 'hsl(', 'hsla(', 'hwb(', 'lab(', 'lch(', 'oklab(', 'oklch(', 'color('];

function isHex(ch: string | undefined): boolean {
  return ch !== undefined && (isDigit(ch) || ('a' <= ch && ch <= 'f') || ('A' <= ch && ch <= 'F'));
}

/** Declarations that paint a literal color or a literal blur radius instead of a token. */
function literalPaint(file: string, css: string): string[] {
  const problems: string[] = [];
  for (const rule of parseRules(css)) {
    for (const [property, value] of rule.decls) {
      if (property.startsWith('--')) continue;
      const lower = value.toLowerCase();
      let bad = COLOR_FUNCTIONS.some((fn) => {
        const at = lower.indexOf(fn);
        return at !== -1 && (at === 0 || !isNameChar(lower[at - 1]));
      });
      const hash = value.indexOf('#');
      if (hash !== -1 && isHex(value[hash + 1])) bad = true;
      const blur = lower.indexOf('blur(');
      if (blur !== -1 && isDigit(lower[blur + 'blur('.length])) bad = true;
      if (bad) problems.push(`${file}: ${rule.selectors.join(', ')} { ${property}: ${value} }`);
    }
  }
  return problems;
}

describe('style rules', () => {
  const keys = themeKeys();

  it('reads the locked theme keys', () => {
    expect(keys.colors.has('foreground')).toBe(true);
    expect(keys.colors.has('danger')).toBe(true);
    expect(keys.texts.has('body')).toBe(true);
    expect(keys.texts.has('sm')).toBe(false);
  });

  it('the page rule catches paint, palettes and unknown text keys', () => {
    const sample = `
      <div className="flex gap-3 text-body text-secondary md:text-strong truncate text-center" />
      <div className={cn('grid', ok && 'text-danger/80')} />
      <div className="bg-surface rounded-lg text-emerald-600 text-white text-sm dark:text-foreground" />
      <div className={cn('from-violet-300', 'border-[#ff0000]', 'w-[220px]', 'text-[13px]')} />
    `;
    expect(pageProblems('sample', sample, keys).map((problem) => problem.slice('sample: '.length))).toEqual([
      'bg-surface (pages do not paint surfaces)',
      'rounded-lg (pages do not paint surfaces)',
      'text-emerald-600 (text key not in the theme)',
      'text-white (text key not in the theme)',
      'text-sm (text key not in the theme)',
      'dark:text-foreground (dark: does not exist)',
      'from-violet-300 (not in the locked theme)',
      'border-[#ff0000] (arbitrary color)',
      'text-[13px] (arbitrary text value)',
    ]);
  });

  it('pages under the rule use layout and text utilities only', () => {
    const files = pageRuleScripts();
    const problems = files.flatMap((file) => pageProblems(relative(PACKAGE, file), readFileSync(file, 'utf8'), keys));
    expect(problems).toEqual([]);
  });

  it('own components and the shell pass no string-literal class', () => {
    expect(literalClassNames('sample', `<div className="flex" />\n<div className={styles.root} />`)).toEqual([
      'sample:1',
    ]);
    const files = MODULE_ONLY_ROOTS.flatMap((root) => walk(root, isScript));
    const problems = files.flatMap((file) => literalClassNames(relative(PACKAGE, file), readFileSync(file, 'utf8')));
    expect(problems).toEqual([]);
  });

  it('own stylesheets write no duration literal outside var()', () => {
    expect(hasDurationLiteral('transition: opacity var(--duration-fast) var(--ease-standard);')).toBe(false);
    expect(hasDurationLiteral('animation-delay: calc(var(--stagger-item) * 3);')).toBe(false);
    expect(hasDurationLiteral('transition: opacity 160ms ease;')).toBe(true);
    expect(hasDurationLiteral('animation: spin 0.8s linear infinite;')).toBe(true);
    expect(hasDurationLiteral('transition-delay: 2s;')).toBe(true);
    expect(hasDurationLiteral('width: 12px; height: 4svh;')).toBe(false);
    const files = ownModuleCss();
    const problems = files.flatMap((file) => durationProblems(relative(PACKAGE, file), readFileSync(file, 'utf8')));
    expect(problems).toEqual([]);
  });

  it('only arc-x/motion/motion.ts names a spring', () => {
    expect(springProblems('sample', '{ type: "spring", visualDuration: 0.4 }')).toEqual(['sample: visualDuration']);
    const files = ownScripts().filter((file) => file !== MOTION_FILE);
    const problems = files.flatMap((file) => springProblems(relative(PACKAGE, file), readFileSync(file, 'utf8')));
    expect(problems).toEqual([]);
  });

  it('every own stylesheet that animates has a reduced-motion block', () => {
    const files = ownModuleCss();
    const problems = files
      .filter((file) => {
        const css = readFileSync(file, 'utf8');
        return (css.includes('animation') || css.includes('transition')) && !css.includes('prefers-reduced-motion');
      })
      .map((file) => relative(PACKAGE, file));
    expect(problems).toEqual([]);
  });

  it('a reduced-motion rule cancels every keyframe animation at equal or higher specificity', () => {
    expect(specificity(".root[data-granularity='text'] > .cell[data-prev] > .now")).toEqual([0, 5, 0]);
    expect(specificity(".surface:not([data-shape='panel']) .panel")).toEqual([0, 3, 0]);
    expect(specificity('.cell[data-prev]::before')).toEqual([0, 2, 1]);
    expect(subject('.root > .cell[data-prev]::before')).toBe('.cell::before');
    const weak =
      '.root[data-g="a"] > .now { animation: x 1s; }\n@media (prefers-reduced-motion: reduce) { .root .now { animation: none; } }';
    expect(unreducedAnimations('sample', weak)).toEqual(['sample: .root[data-g="a"] > .now']);
    const gated = '@media (prefers-reduced-motion: no-preference) { .root[data-g="a"] > .now { animation: x 1s; } }';
    expect(unreducedAnimations('sample', gated)).toEqual([]);
    const supported =
      '@supports (display: grid) { .a { animation: x 1s; } }\n@media (prefers-reduced-motion: reduce) { .b { animation: none; } }';
    expect(unreducedAnimations('sample', supported)).toEqual(['sample: .a']);
    const nestedReduce =
      '.a { animation: x 1s; }\n@container (max-width: 1px) { @media (prefers-reduced-motion: reduce) { .a { animation: none; } } }';
    expect(unreducedAnimations('sample', nestedReduce)).toEqual([]);
    const files = ownModuleCss();
    const problems = files.flatMap((file) => unreducedAnimations(relative(PACKAGE, file), readFileSync(file, 'utf8')));
    expect(problems).toEqual([]);
  });

  it('every infinite loop pauses while the window is hidden, pseudo-elements included', () => {
    expect(loopPseudos('.a::after { animation: x 1s infinite; } .b { animation: y 1s; }')).toEqual(['::after']);
    expect(loopPseudos('.a { animation-name: x; animation-iteration-count: infinite; }')).toEqual(['']);
    expect(loopPseudos('@container (max-width: 1px) { .a::marker { animation: x 1s infinite; } }')).toEqual([
      '::marker',
    ]);
    expect(loopPseudos('@layer l { @scope (.r) { .a::backdrop { animation: x 1s infinite; } } }')).toEqual([
      '::backdrop',
    ]);
    expect(loopPseudos('@layer a, b;\n.a::placeholder { animation: x 1s infinite; }')).toEqual(['::placeholder']);
    expect(loopPseudos('@keyframes k { from { animation: x 1s infinite; } }')).toEqual([]);
    const needed = new Set(
      walk(UI, (path) => path.endsWith('.css')).flatMap((file) => loopPseudos(readFileSync(file, 'utf8'))),
    );
    expect(needed.has('::after')).toBe(true);
    const paused = new Set<string>();
    for (const rule of parseRules(readFileSync(join(UI, 'styles/tokens.css'), 'utf8'))) {
      if (!(rule.decls.get('animation-play-state') ?? '').startsWith('paused')) continue;
      for (const selector of rule.selectors) {
        const prefix = ':root[data-hidden] ';
        if (selector.startsWith(prefix)) paused.add(selector.slice(prefix.length));
      }
    }
    const unpaused = [...needed].filter((pseudo) => !paused.has(`*${pseudo}`));
    expect(unpaused).toEqual([]);
  });

  it('own stylesheets paint colors and blur radii from tokens only', () => {
    expect(
      literalPaint('sample', '.a { background: oklch(10% 0 0 / 0.46); backdrop-filter: blur(7px); color: #fff; }'),
    ).toHaveLength(3);
    expect(literalPaint('sample', '@layer l { .a { color: #fff; } }')).toHaveLength(1);
    expect(literalPaint('sample', '@starting-style { .a { background: rgb(0 0 0); } }')).toHaveLength(1);
    expect(
      literalPaint('sample', '@media (min-width: 1px) { @supports (x: y) { .a { filter: blur(2px); } } }'),
    ).toHaveLength(1);
    expect(
      literalPaint(
        'sample',
        '.a { background: color-mix(in oklab, var(--x) 4%, transparent); filter: blur(var(--b)); }',
      ),
    ).toEqual([]);
    const files = ownModuleCss();
    const problems = files.flatMap((file) => literalPaint(relative(PACKAGE, file), readFileSync(file, 'utf8')));
    expect(problems).toEqual([]);
  });

  it('only arc-x/live-label imports arc/text-shimmer', () => {
    const importers = walk(UI, isScript)
      .filter((file) => !file.startsWith(SHIMMER_HOME) && !file.startsWith(SHIMMER_ITEM))
      .filter((file) => readFileSync(file, 'utf8').includes('arc/text-shimmer'))
      .map((file) => relative(PACKAGE, file));
    expect(importers).toEqual([]);
    const home = walk(SHIMMER_HOME, isScript).filter((file) => readFileSync(file, 'utf8').includes('arc/text-shimmer'));
    expect(home).toHaveLength(1);
  });
  it('every h1 is set in the display font', () => {
    const problems: string[] = [];
    let seen = 0;
    for (const file of ownScripts()) {
      const source = readFileSync(file, 'utf8');
      let at = source.indexOf('<h1');
      while (at !== -1) {
        seen += 1;
        const tag = source.slice(at, source.indexOf('>', at) + 1);
        const moduleRef = 'className={styles.';
        const ref = tag.indexOf(moduleRef);
        let display = tag.includes('font-heading');
        if (ref !== -1) {
          const name = tag.slice(ref + moduleRef.length, tag.indexOf('}', ref));
          const sheet = relative(UI, file).split('.tsx').join('.module.css');
          display = cssValue(cssRules(sheet), '.' + name, 'font-family') === 'var(--font-display)';
        }
        if (!display) problems.push(`${relative(PACKAGE, file)}: ${tag}`);
        at = source.indexOf('<h1', at + 3);
      }
    }
    expect(seen).toBeGreaterThan(1);
    expect(problems).toEqual([]);
  });
});
