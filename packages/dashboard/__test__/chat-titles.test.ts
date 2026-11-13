/** The one title rule: the agent's title, else the first prompt's title, else ''. */

import { describe, expect, it } from 'vite-plus/test';

import { PROMPT_TITLE_MAX, promptTitle, resolveTitle } from '../src/chat/titles.js';

const length = (text: string): number => Array.from(text).length;

describe('promptTitle', () => {
  it('takes the first non-empty line, cleaned, on one line with single spaces', () => {
    const rlo = String.fromCharCode(0x202e);
    expect(promptTitle('\n  \n  Fix ' + rlo + 'the\t\tbuild   now  \nsecond line')).toBe('Fix the build now');
    expect(promptTitle('')).toBe('');
    const invisible = [0x200b, 0x2060, 0x2064, 0xfeff].map((cp) => String.fromCharCode(cp)).join('');
    expect(promptTitle(`tab${invisible}zw end`)).toBe('tabzw end');
    const family = String.fromCodePoint(0x1f468, 0x200d, 0x1f469);
    expect(promptTitle(`${family} trip`)).toBe(`${family} trip`);
    expect(promptTitle(' \n\t\n')).toBe('');
  });

  it('keeps a title of exactly the limit whole', () => {
    const text = 'a'.repeat(PROMPT_TITLE_MAX);
    expect(promptTitle(text)).toBe(text);
    expect(promptTitle(`${text} `)).toBe(text);
  });

  it('cuts a longer title at the last word that fits and adds an ellipsis', () => {
    const text = 'Explain how the paged cache picks a block size for each model family in detail';
    const title = promptTitle(text);
    expect(title).toBe('Explain how the paged cache picks a block size for each…');
    expect(length(title)).toBeLessThanOrEqual(PROMPT_TITLE_MAX);
  });

  it('cuts one long word hard at the limit', () => {
    const title = promptTitle('x'.repeat(80));
    expect(title).toBe(`${'x'.repeat(PROMPT_TITLE_MAX - 1)}…`);
    expect(length(title)).toBe(PROMPT_TITLE_MAX);
  });

  it('does not cut at a space that leaves too short a title', () => {
    const title = promptTitle(`short ${'y'.repeat(70)}`);
    expect(title).toBe(`short ${'y'.repeat(PROMPT_TITLE_MAX - 7)}…`);
  });
});

describe('resolveTitle', () => {
  it('prefers the agent title, then the prompt, then nothing', () => {
    expect(resolveTitle('  Named  ', 'the prompt')).toBe('Named');
    expect(resolveTitle(' \n', 'the prompt')).toBe('the prompt');
    expect(resolveTitle('', null)).toBe('');
  });
});
