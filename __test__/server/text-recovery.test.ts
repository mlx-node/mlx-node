import { describe, expect, it } from 'vite-plus/test';

import { longestSuffixPrefixOverlap } from '../../packages/server/src/text-recovery.js';

describe('longestSuffixPrefixOverlap', () => {
  it('returns 0 when there is no overlap', () => {
    expect(longestSuffixPrefixOverlap('abc', 'def')).toBe(0);
    expect(longestSuffixPrefixOverlap('', 'abc')).toBe(0);
    expect(longestSuffixPrefixOverlap('abc', '')).toBe(0);
    expect(longestSuffixPrefixOverlap('', '')).toBe(0);
  });

  it('returns full final length when streamed ends with full final', () => {
    expect(longestSuffixPrefixOverlap('hello world', 'world')).toBe(5);
    expect(longestSuffixPrefixOverlap('foo', 'foo')).toBe(3);
  });

  it('returns longest matching prefix-of-final-as-suffix-of-streamed', () => {
    // streamed="hello ", final="hello world" -> overlap "hello " (length 6)
    expect(longestSuffixPrefixOverlap('hello ', 'hello world')).toBe(6);
    // streamed="abc", final="bcdef" -> overlap "bc" (length 2)
    expect(longestSuffixPrefixOverlap('abc', 'bcdef')).toBe(2);
  });

  it('returns longest match when multiple prefixes work', () => {
    // streamed="aaa", final="aaab" -> "aaa" matches at length 3
    expect(longestSuffixPrefixOverlap('aaa', 'aaab')).toBe(3);
  });

  it('handles whitespace-divergence case (post-</think> trim)', () => {
    // The original bug: streamed="\n\n", final="<tool_call>..." -> 0
    expect(longestSuffixPrefixOverlap('\n\n', '<tool_call>x')).toBe(0);
    // The duplicate-trim case: streamed="Let me check. ", final="Let me check." -> 0
    // (the trimmed final is NOT a suffix of trailing-space-having streamed)
    expect(longestSuffixPrefixOverlap('Let me check. ', 'Let me check.')).toBe(0);
  });

  it('caps overlap at min(streamed.length, final.length)', () => {
    // streamed shorter than final
    expect(longestSuffixPrefixOverlap('xy', 'xyzabc')).toBe(2);
    // final shorter than streamed
    expect(longestSuffixPrefixOverlap('abcxy', 'xy')).toBe(2);
  });
});
