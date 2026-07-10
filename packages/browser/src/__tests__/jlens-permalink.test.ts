import { describe, expect, it } from 'vite-plus/test';

import { decodePermalink, encodePermalink, type JSpaceState } from '../../demo/jlens-core/permalink';

const base: JSpaceState = {
  prompt: "La saison après l'été est l'",
  mode: 'jacobian',
  pins: [3098, 9871],
  sel: { layerIdx: 6, pos: 8 },
};

describe('permalink codec', () => {
  it('round-trips every field', () => {
    expect(decodePermalink(encodePermalink(base))).toEqual(base);
  });

  it('round-trips a 128-token-ish prompt with newlines and #', () => {
    const s = { ...base, prompt: 'a #b\nc&d=e '.repeat(50) };
    expect(decodePermalink(encodePermalink(s)).prompt).toBe(s.prompt);
  });

  it('tolerates a leading # on decode', () => {
    expect(decodePermalink('#' + encodePermalink(base))).toEqual(base);
  });

  it('clamps an over-long pin list to LENS_MAX_PINNED', () => {
    const hash = 'p=hi&mode=j&pins=' + Array.from({ length: 12 }, (_, i) => i + 1).join('-');
    expect(decodePermalink(hash).pins).toHaveLength(8);
  });

  it('drops junk rather than throwing', () => {
    const r = decodePermalink('p=hi&mode=nonsense&pins=a-b-c&sel=zz');
    expect(r.prompt).toBe('hi');
    expect(r.mode).toBeUndefined();
    expect(r.pins).toEqual([]);
    expect(r.sel).toBeNull();
  });

  it('returns an empty object for an empty hash', () => {
    expect(decodePermalink('')).toEqual({});
    expect(decodePermalink('#')).toEqual({});
  });
});
