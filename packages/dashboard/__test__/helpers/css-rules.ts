import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { isWordChar, splitWhitespace } from '../../ui/src/lib/text-scan.js';

const UI = join(import.meta.dirname, '..', '..', 'ui', 'src');

export interface CssRule {
  selectors: string[];
  decls: Map<string, string>;
  /** Inside `@media (prefers-reduced-motion: reduce)`. */
  reduce: boolean;
  /** Inside `@container`, `@supports`, `@scope` or `@starting-style`: never read by `cssValue`. */
  conditional: boolean;
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

/** Grouping at-rules whose body applies only under a condition the stylesheet alone cannot decide. */
const CONDITIONAL_AT_RULES = new Set(['@container', '@supports', '@scope', '@starting-style']);

/** `@media` for `@media (x)`, '' for a selector. */
function atRuleName(prelude: string): string {
  if (!prelude.startsWith('@')) return '';
  let end = 1;
  while (end < prelude.length && (isWordChar(prelude[end]) || prelude[end] === '-' || prelude[end] === '_')) end += 1;
  return prelude.slice(0, end);
}

function parse(source: string, reduce: boolean, conditional: boolean): CssRule[] {
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
    if (atRule === '@media') {
      const media = splitWhitespace(prelude).join(' ');
      rules.push(...parse(body, reduce || media.includes('prefers-reduced-motion: reduce'), conditional));
    } else if (atRule === '@layer') {
      rules.push(...parse(body, reduce, conditional));
    } else if (CONDITIONAL_AT_RULES.has(atRule)) {
      rules.push(...parse(body, reduce, true));
    } else if (atRule === '') {
      const decls = new Map<string, string>();
      for (const decl of splitTop(body, ';')) {
        const colon = decl.indexOf(':');
        if (colon === -1) continue;
        decls.set(decl.slice(0, colon).trim(), splitWhitespace(decl.slice(colon + 1)).join(' '));
      }
      const selectors = splitTop(prelude, ',').map((selector) => splitWhitespace(selector).join(' '));
      rules.push({ selectors, decls, reduce, conditional });
    }
    i = end + 1;
  }
  return rules;
}

/** Every rule of a stylesheet under `ui/src` (path relative to it), in source order. */
export function cssRules(path: string): CssRule[] {
  return parse(readFileSync(join(UI, path), 'utf8'), false, false);
}

/**
 * The value the last unconditional rule naming exactly `selector` gives `property`, read outside or inside the
 * reduced-motion block. undefined when no such rule sets it.
 */
export function cssValue(
  rules: readonly CssRule[],
  selector: string,
  property: string,
  where: 'base' | 'reduce' = 'base',
): string | undefined {
  let value: string | undefined;
  for (const rule of rules) {
    if (rule.conditional || rule.reduce !== (where === 'reduce')) continue;
    if (!rule.selectors.includes(selector)) continue;
    const found = rule.decls.get(property);
    if (found !== undefined) value = found;
  }
  return value;
}
