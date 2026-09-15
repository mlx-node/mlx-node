import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';

import type { Tokenizer } from 'tokenizers';

let tokenizer: Promise<Tokenizer> | undefined;

function loadTokenizer(): Promise<Tokenizer> {
  // Source and emitted modules have the same relative path to package assets.
  // Load lazily: ordinary sessions never need the native addon or vocabulary.
  tokenizer ??= Promise.all([
    import('tokenizers'),
    readFile(new URL('../../assets/o200k_base.json.gz', import.meta.url)),
  ]).then(([{ Tokenizer }, data]) => Tokenizer.fromString(gunzipSync(data).toString('utf8')));
  return tokenizer;
}

/** Fixed, offline o200k_base counts, without padding, truncation or special tokens. */
export async function countDelegateTokens(texts: readonly string[]): Promise<number[]> {
  if (texts.length === 0) return [];
  const encoder = await loadTokenizer();
  const counts: number[] = [];
  // Bound native encoding/alignment allocations for large session histories.
  for (let i = 0; i < texts.length; i += 16) {
    const encodings = await encoder.encodeBatch(texts.slice(i, i + 16), { addSpecialTokens: false });
    for (const encoding of encodings) counts.push(encoding.getLength());
  }
  return counts;
}
