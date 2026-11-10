import { MAX_CONTEXT_ERROR, MAX_YARN_CONTEXT_TOKENS, parseMaxContext } from '@mlx-node/server/host/max-context';
import { describe, expect, it } from 'vite-plus/test';

// Splash `parse_max_context` and its `test_serve_options.py` table: K is 1024
// tokens, M a million, and the count runs from 1 to 1,010,000.
describe('parseMaxContext', () => {
  // Mutation: cap at 1,000,000 instead of MAX_YARN_CONTEXT_TOKENS → red
  // ('1010000' is refused).
  it.each([
    ['100K', 102_400],
    ['256k', 262_144],
    ['262144', 262_144],
    ['1', 1],
    ['262145', 262_145],
    ['512K', 524_288],
    ['1m', 1_000_000],
    ['1000000', 1_000_000],
    ['1010000', 1_010_000],
    [' 400K ', 409_600],
  ])('reads %j as %d tokens', (value, tokens) => {
    expect(parseMaxContext(value)).toBe(tokens);
  });

  it("returns undefined for 'auto'", () => {
    expect(parseMaxContext('auto')).toBeUndefined();
    expect(parseMaxContext(' AUTO ')).toBeUndefined();
  });

  // Mutation: read the number with `Number()` instead of the digit check → red
  // ('0.5M' gives 500,000, '1e5' 100,000).
  it.each(['0', '-1', '1010001', '987K', '2M', '1.5M', '0.5M', 'M', 'K', 'bad', '', '1e5', '1G'])(
    'refuses %j with the Splash message',
    (value) => {
      expect(() => parseMaxContext(value)).toThrow(MAX_CONTEXT_ERROR);
    },
  );

  it('names the most any family serves', () => {
    expect(MAX_YARN_CONTEXT_TOKENS).toBe(1_010_000);
    expect(MAX_CONTEXT_ERROR).toBe(
      "--max-context must be 'auto' or a token count up to 1010000, such as 100K or 1M (K = 1024 tokens, M = 1000000 tokens)",
    );
  });
});
