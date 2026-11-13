import { describe, expect, it } from 'vite-plus/test';

import {
  hasNonWhitespace,
  isAsciiDigit,
  isIdentifier,
  isNumericWord,
  isWhitespace,
  isWordChar,
  keepIdChars,
  splitShortcut,
  splitWhitespace,
  stripCommas,
  stripTags,
} from '../ui/src/lib/text-scan.js';

describe('text-scan', () => {
  it('knows the whitespace set', () => {
    for (const ch of [' ', '\t', '\n', '\v', '\f', '\r', ' ', ' ', ' ', ' ', ' ', ' ']) {
      expect(isWhitespace(ch), JSON.stringify(ch)).toBe(true);
    }
    for (const ch of [' ', ' ', '　', '﻿']) expect(isWhitespace(ch)).toBe(true);
    for (const ch of ['a', '0', '-', '​', '']) expect(isWhitespace(ch), JSON.stringify(ch)).toBe(false);
  });

  it('knows ASCII digits only', () => {
    expect(isAsciiDigit('0')).toBe(true);
    expect(isAsciiDigit('9')).toBe(true);
    expect(isAsciiDigit('a')).toBe(false);
    expect(isAsciiDigit('٣')).toBe(false);
  });

  it('treats letters and digits of any script as word characters, punctuation and spaces not', () => {
    for (const ch of ['a', 'Z', '7', 'é', 'ж', '模']) expect(isWordChar(ch), ch).toBe(true);
    for (const ch of [' ', '-', '_', '.', '—', '、', ' ', '']) expect(isWordChar(ch), ch).toBe(false);
  });

  it('finds non-whitespace', () => {
    expect(hasNonWhitespace(' \n\t')).toBe(false);
    expect(hasNonWhitespace('')).toBe(false);
    expect(hasNonWhitespace('  x ')).toBe(true);
  });

  it('splits on whitespace runs with no empty parts', () => {
    expect(splitWhitespace('  open   the\tcache \n')).toEqual(['open', 'the', 'cache']);
    expect(splitWhitespace('')).toEqual([]);
    expect(splitWhitespace('one')).toEqual(['one']);
  });

  it('drops tags and keeps an unclosed angle bracket', () => {
    expect(stripTags('<svg><path/></svg>')).toBe('');
    expect(stripTags('Save <b>now</b>')).toBe('Save now');
    expect(stripTags('a < b')).toBe('a < b');
  });

  it('strips commas', () => {
    expect(stripCommas('1,284,000')).toBe('1284000');
  });

  it('recognizes numeric words the way the input counter needs', () => {
    for (const word of ['12', '1,284', '3.5', '12/40', '14:05', '98%']) expect(isNumericWord(word), word).toBe(true);
    for (const word of ['', 'a1', '%', '12a', '1%%', '-3']) expect(isNumericWord(word), word).toBe(false);
  });

  it('recognizes ASCII identifiers', () => {
    for (const text of ['a', '_x', '$el', 'camelCase9']) expect(isIdentifier(text), text).toBe(true);
    for (const text of ['', '9a', 'a-b', 'é']) expect(isIdentifier(text), text).toBe(false);
  });

  it('keeps id characters', () => {
    expect(keepIdChars('a b_c-1:2', false)).toBe('abc-12');
    expect(keepIdChars('a b_c-1:2', true)).toBe('ab_c-12');
  });

  it('splits shortcuts on whitespace and on a plus that is not last', () => {
    expect(splitShortcut('⌘ K')).toEqual(['⌘', 'K']);
    expect(splitShortcut('Shift+D')).toEqual(['Shift', 'D']);
    expect(splitShortcut('Shift++')).toEqual(['Shift', '+']);
    expect(splitShortcut('+')).toEqual(['+']);
    expect(splitShortcut('  ⌘  1 ')).toEqual(['⌘', '1']);
  });
});
