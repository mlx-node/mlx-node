import { describe, expect, it } from 'vite-plus/test';

import { countDelegateTokens } from '../src/ingest/tokenizer.js';

// Reference counts from o200k_base before replacing the JavaScript encoder.
const cases: Array<[string, number]> = [
  ['', 0],
  ['OK', 1],
  ['alpha beta gamma', 3],
  ['Hello, world!', 4],
  ['<|endoftext|> <|im_start|> <|endofprompt|>', 20],
  ['你好，世界！こんにちは世界 🌍', 8],
  ['مرحبا بالعالم — café e\u0301', 8],
  ["We're testing I'M and 1234567890.\r\n\tend", 12],
  ['🙂👨‍👩‍👧‍👦 \u0000 \ud800 end', 16],
  ['a'.repeat(10000), 1250],
];

describe('native delegate tokenizer', () => {
  it('preserves ordinary o200k_base counts, including literal special-token text and untruncated long inputs', async () => {
    expect(await countDelegateTokens(cases.map(([text]) => text))).toEqual(cases.map(([, count]) => count));
  });

  it('keeps order across batches and supports concurrent callers without padding', async () => {
    const inputs = Array.from({ length: 40 }, (_, i) => cases[i % cases.length]!);
    const actual = await Promise.all([
      countDelegateTokens(inputs.map(([text]) => text)),
      countDelegateTokens(['OK', '']),
      countDelegateTokens([]),
    ]);
    expect(actual).toEqual([inputs.map(([, count]) => count), [1, 0], []]);
  });
});
