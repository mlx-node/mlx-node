/**
 * The command palette's scorer (`components/arc/command-palette/command-palette.tsx`, which stays as vendored) for one
 * typed word: a name that starts with it first, then a word of the name, then the letters inside a name, then the
 * words of the description, then the letters typed in order. Ties keep the order the rows came in. A hand-written
 * scan, no regex.
 */

import { isWordChar } from './text-scan';

/** The first index where `word` starts a word of `text`, or -1. */
function wordStart(text: string, word: string): number {
  for (let at = text.indexOf(word); at !== -1; at = text.indexOf(word, at + 1)) {
    if (at === 0 || !isWordChar(text.charAt(at - 1))) return at;
  }
  return -1;
}

/** Letters typed in order, starting at a word, within a short span: "dr" finds "deep-research". */
function subsequence(text: string, word: string): boolean {
  if (word.length < 2) return false;
  for (let start = text.indexOf(word.charAt(0)); start !== -1; start = text.indexOf(word.charAt(0), start + 1)) {
    if (start > 0 && isWordChar(text.charAt(start - 1))) continue;
    let at = start;
    for (let i = 1; i < word.length && at !== -1; i++) at = text.indexOf(word.charAt(i), at + 1);
    if (at !== -1 && at - start < word.length * 3) return true;
  }
  return false;
}

function score(name: string, description: string, word: string): number | null {
  const inName = wordStart(name, word);
  if (inName === 0) return 100;
  if (inName > 0) return 80;
  if (word.length >= 3 && name.includes(word)) return 50;
  if (wordStart(description, word) !== -1) return 25;
  return subsequence(name, word) ? 10 : null;
}

/** The rows that match `typed`, best first; every row, in its order, while nothing is typed. */
export function matchCommands<T extends { name: string; about: string }>(rows: readonly T[], typed: string): T[] {
  const word = typed.toLowerCase();
  if (word === '') return [...rows];
  const ranked: Array<{ row: T; score: number; order: number }> = [];
  rows.forEach((row, order) => {
    const value = score(row.name.toLowerCase(), row.about.toLowerCase(), word);
    if (value !== null) ranked.push({ row, score: value, order });
  });
  return ranked.sort((a, b) => b.score - a.score || a.order - b.order).map((r) => r.row);
}
