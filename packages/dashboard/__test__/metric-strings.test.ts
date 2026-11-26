/**
 * The metric words live in `ui/src/lib/metric-strings.ts` only. A surface that types `'Output rate'` itself would
 * drift from its definition, so the source scan fails on any other file that holds one of the four speed words.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import { BANDS, ENGINE_STATE, METRIC } from '../ui/src/lib/metric-strings';

const PACKAGE = fileURLToPath(new URL('..', import.meta.url));
const UI = join(PACKAGE, 'ui/src');
const OWNER = 'lib/metric-strings.ts';

// Plain words, so JSX text, double quotes, template strings and comments all count.
const WORDS: readonly string[] = ['Output rate', 'Decode speed', 'Prefill speed', 'First token'];

/** Files that still type one of the words. A slice that converts a file removes it here. */
const STILL_TO_CONVERT: readonly string[] = [];

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) out.push(...sourceFiles(path));
    else if (name.endsWith('.ts') || name.endsWith('.tsx')) out.push(path);
  }
  return out;
}

describe('metric strings', () => {
  it('gives every label one meaning', () => {
    for (const group of [
      Object.values(METRIC).map((word) => word.label),
      Object.values(BANDS),
      Object.values(ENGINE_STATE),
    ]) {
      expect(new Set(group).size).toBe(group.length);
    }
  });

  it('gives every word a short tip', () => {
    for (const word of Object.values(METRIC)) {
      expect(word.tip.length).toBeGreaterThan(0);
      expect(word.tip.length).toBeLessThanOrEqual(110);
    }
  });

  it('keeps the four speed words out of every other source file', () => {
    const holders = sourceFiles(UI)
      .map((path) => relative(UI, path))
      .filter((path) => path !== OWNER)
      .filter((path) => {
        const text = readFileSync(join(UI, path), 'utf8');
        return WORDS.some((word) => text.includes(word));
      })
      .sort();
    expect(holders).toEqual([...STILL_TO_CONVERT].sort());
  });
});
