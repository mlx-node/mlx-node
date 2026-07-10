import { describe, expect, it } from 'vite-plus/test';

import { isColdPrompt } from '../../demo/jspace/cold-prompt';

// The render view and the permalink write gate both consult isColdPrompt so they
// can never diverge: ONLY the exactly-empty prompt is "cold" (shows the model-free
// starter grid). Whitespace is content — a custom prompt not yet run shows the
// skeleton, never a starter grid under someone else's text.
describe('isColdPrompt', () => {
  it('is true only for the exactly-empty prompt', () => {
    expect(isColdPrompt('')).toBe(true);
  });

  it('is false for whitespace-only prompts (whitespace is content)', () => {
    expect(isColdPrompt(' ')).toBe(false);
    expect(isColdPrompt('\n')).toBe(false);
    expect(isColdPrompt('\t')).toBe(false);
    expect(isColdPrompt('  \n  ')).toBe(false);
  });

  it('is false for a non-empty prompt', () => {
    expect(isColdPrompt('a')).toBe(false);
  });
});
