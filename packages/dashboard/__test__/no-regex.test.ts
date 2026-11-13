/**
 * No regular expressions in the SPA, the dev scripts or the dashboard tests: this app commits hand-written scanners
 * only (`ui/src/lib/text-scan.ts`). Every file is parsed to its syntax tree, so strings, comments, JSX text and
 * division never read as a pattern, and a file the parser rejects fails the guard instead of passing unread.
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseSync } from 'oxc-parser';
import { describe, expect, it } from 'vite-plus/test';

const PACKAGE = fileURLToPath(new URL('..', import.meta.url));
const UI = join(PACKAGE, 'ui/src');

/** What is scanned: the whole SPA, the dev scripts and the dashboard tests. */
const NO_REGEX_ROOTS: readonly string[] = [UI, join(PACKAGE, 'scripts'), join(PACKAGE, '__test__')];

/** String methods that compile a string argument into a pattern. */
const PATTERN_METHODS = new Set(['match', 'matchAll', 'search']);

type Node = { readonly type: string; readonly start: number; readonly [key: string]: unknown };

function isNode(value: unknown): value is Node {
  return typeof value === 'object' && value !== null && typeof (value as { type?: unknown }).type === 'string';
}

function propertyName(node: unknown): string | null {
  if (!isNode(node)) return null;
  if (node.type === 'Identifier' && typeof node.name === 'string') return node.name;
  if (node.type === 'Literal' && typeof node.value === 'string') return node.value;
  return null;
}

/** What makes this node a pattern, or null. */
function patternKind(node: Node): string | null {
  if (node.type === 'Literal' && node.regex != null) return 'regex literal';
  if (node.type === 'Identifier' && node.name === 'RegExp') return 'RegExp';
  if (node.type === 'MemberExpression' && node.computed === true && propertyName(node.property) === 'RegExp')
    return 'RegExp';
  if (node.type === 'CallExpression' && isNode(node.callee) && node.callee.type === 'MemberExpression') {
    const method = propertyName(node.callee.property);
    if (method !== null && PATTERN_METHODS.has(method)) return `.${method}()`;
  }
  return null;
}

function lineOf(text: string, offset: number): number {
  let line = 1;
  for (let i = 0; i < offset && i < text.length; i += 1) if (text[i] === '\n') line += 1;
  return line;
}

function isSource(path: string): boolean {
  return (path.endsWith('.ts') || path.endsWith('.tsx')) && !path.endsWith('.d.ts');
}

function sourceFiles(root: string): string[] {
  if (!existsSync(root)) return [];
  if (!statSync(root).isDirectory()) return isSource(root) ? [root] : [];
  const files: string[] = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    if (entry.name === 'node_modules') continue;
    const path = join(root, entry.name);
    if (entry.isDirectory()) files.push(...sourceFiles(path));
    else if (isSource(path)) files.push(path);
  }
  return files;
}

/** `file:line: kind` for every pattern in the source, and `file: parse error` when it does not parse. */
function regexHits(path: string, text: string): string[] {
  const result = parseSync(path, text);
  if (result.errors.length > 0) return result.errors.map((error) => `${path}: parse error: ${error.message}`);
  const hits: string[] = [];
  const visit = (value: unknown): void => {
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    if (!isNode(value)) return;
    const kind = patternKind(value);
    if (kind !== null) hits.push(`${path}:${lineOf(text, value.start)}: ${kind}`);
    for (const key of Object.keys(value)) if (key !== 'parent') visit(value[key]);
  };
  visit(result.program);
  return hits;
}

describe('no regex', () => {
  it.each([
    ['an arrow body', 'export const p = () => /a+/u;'],
    ['a return', 'function f(s: string) { return /a/.exec(s); }'],
    ['a line of its own', 'const xs = [\n  /a/,\n];'],
    ['a template interpolation', 'const t = `${/a/.source}`;'],
    ['a string with an apostrophe before it', 'const q = "don\'t" + String(/a/);'],
    ['typeof', 'const r = typeof /a/;'],
    ['a replace call', 's.replace(/a/g, "");'],
    ['a test call', '/a/u.test(s);'],
    ['a toThrow argument', 'expect(f).toThrow(/gone/);'],
    ['new RegExp', 'const r = new RegExp(x);'],
    ['a RegExp call', 'const r = RegExp("a");'],
    ['a RegExp type', 'let r: RegExp | null = null;'],
    ['a computed RegExp', 'const R = globalThis["RegExp"];'],
    ['a match call', 's.match("a+");'],
    ['a matchAll call', 's.matchAll(pattern);'],
    ['a search call', 's.search("a");'],
    ['a JSX expression', 'const e = <b title={String(/a/)}>x</b>;'],
  ])('flags a pattern in %s', (_where, source) => {
    expect(regexHits('sample.tsx', source)).toHaveLength(1);
  });

  it.each([
    ['division', 'const half = total / 2;'],
    ['compound division', 'width /= 2;'],
    ['a chained division', 'const r = (a / b) / c;'],
    ['a self-closing tag', 'const e = <br />;'],
    ['a path attribute', 'const e = <a href="/sessions">x</a>;'],
    ['JSX text shaped like a pattern', 'const e = <p>/a+/u.test(s) and new RegExp(x)</p>;'],
    ['strings', 'const s = "/a/" + \'new RegExp(x)\' + "s.match(/a/)";'],
    ['a template with a path', 'const url = `${base}/api/x`;'],
    ['a line comment', 'x = y; // see (/path) and /a+/u'],
    ['a block comment', '/* RegExp and /a/g */ const z = 1;'],
    ['split and exec', 'ok.split(","); raw.exec(sql);'],
  ])('leaves alone %s', (_what, source) => {
    expect(regexHits('sample.tsx', source)).toEqual([]);
  });

  it('fails a file it cannot parse', () => {
    expect(regexHits('sample.ts', 'const = ;')).not.toEqual([]);
  });

  it('finds none in the scanned roots', () => {
    const files = NO_REGEX_ROOTS.flatMap(sourceFiles);
    expect(files.length).toBeGreaterThan(0);
    const hits = files.flatMap((file) => regexHits(relative(PACKAGE, file), readFileSync(file, 'utf8')));
    expect(hits).toEqual([]);
  });
});
