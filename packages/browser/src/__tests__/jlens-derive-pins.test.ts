import { describe, expect, it } from 'vite-plus/test';

import { derivePins, type Encode } from '../../demo/jlens-core/derive-pins';

const ID_TO_TEXT: Record<number, string> = {
  3098: ' season', 7891: 'season', 220: ' ', 22: '7', 500: ' autom', 501: 'ne',
};

/** Worker-shaped: `tokenize` already returns {id,text}[]. */
const qwenish: Encode = async (text) => {
  // Qwen fuses a leading space into an alphabetic token, but splits it off a digit.
  if (text === ' season') return [{ id: 3098, text: ' season' }];
  if (text === 'season') return [{ id: 7891, text: 'season' }];
  if (text === ' 7') return [{ id: 220, text: ' ' }, { id: 22, text: '7' }];
  if (text === '7') return [{ id: 22, text: '7' }];
  if (text === ' automne') return [{ id: 500, text: ' autom' }, { id: 501, text: 'ne' }];
  if (text === ' ') return [{ id: 220, text: ' ' }];
  if (text === ' ∅' || text === '∅') return [];
  throw new Error(`unexpected encode(${JSON.stringify(text)})`);
};

// Node-shaped: `encodeTokens` yields ids only and `decodeTokens` yields texts, so
// bake.mts must ADAPT. This is that adapter — the thing the parity test exists to
// exercise. It must not delegate to `qwenish`, or the test asserts f(x) === f(x).
const encodeTokens = async (text: string): Promise<number[]> => (await qwenish(text)).map((t) => t.id);
const decodeTokens = async (ids: number[]): Promise<string[]> => ids.map((id) => ID_TO_TEXT[id] ?? '');
const nodeish: Encode = async (text) => {
  const ids = await encodeTokens(text);
  const texts = await decodeTokens(ids);
  return ids.map((id, i) => ({ id, text: texts[i] ?? '' }));
};

describe('derivePins', () => {
  it('pins the leading-space form when it is a real token', async () => {
    const r = await derivePins(['season'], qwenish);
    expect(r.pinnedIds).toEqual([3098]);
    expect(r.partialFlags).toEqual([false]);
  });

  it('falls back to the space-less form when the first token is whitespace', async () => {
    const r = await derivePins(['7'], qwenish);
    expect(r.pinnedIds).toEqual([22]); // NOT 220, the bare space
    expect(r.partialFlags).toEqual([false]);
  });

  it('flags a concept that needs more than one token', async () => {
    const r = await derivePins(['automne'], qwenish);
    expect(r.pinnedIds).toEqual([500]);
    expect(r.partialFlags).toEqual([true]);
  });

  it('skips a concept that encodes to nothing', async () => {
    const r = await derivePins(['∅', 'season'], qwenish);
    expect(r.pinnedIds).toEqual([3098]);
    expect(r.partialFlags).toEqual([false]);
  });

  it('caps at LENS_MAX_PINNED = 8', async () => {
    const nine = Array.from({ length: 9 }, () => 'season');
    const r = await derivePins(nine, qwenish);
    expect(r.pinnedIds).toHaveLength(8);
    expect(r.partialFlags).toHaveLength(8);
  });

  it('agrees across the worker tokenizer and the node ids+decode adapter', async () => {
    const a = await derivePins(['season', '7', 'automne'], qwenish);
    const b = await derivePins(['season', '7', 'automne'], nodeish);
    expect(a).toEqual(b);
    expect(a.pinnedIds).toEqual([3098, 22, 500]); // pin the fallback, not the space
  });
});
