import { describe, expect, it } from 'vite-plus/test';

import {
  applyPermalink,
  decodePermalink,
  encodePermalink,
  type JSpaceDefaults,
  type JSpaceState,
} from '../../demo/jlens-core/permalink';
import { LENS_MAX_PINNED } from '../inspector-types';

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

  it('clamps an over-long pin list to LENS_MAX_PINNED on decode', () => {
    const over = Array.from({ length: LENS_MAX_PINNED + 4 }, (_, i) => i + 1);
    const hash = 'p=hi&mode=j&pins=' + over.join('-');
    // Assert the exact RETAINED ids, not just the length: a length-only check stays
    // green even if the impl kept the LAST N instead of the first N.
    expect(decodePermalink(hash).pins).toEqual(over.slice(0, LENS_MAX_PINNED));
  });

  it('clamps an over-long pin list to LENS_MAX_PINNED on encode', () => {
    const over = Array.from({ length: LENS_MAX_PINNED + 4 }, (_, i) => i + 1);
    // Inspect the WIRE string directly: routing through decode would clamp too and
    // mask a missing encoder clamp.
    const pinsPart = new URLSearchParams(encodePermalink({ ...base, pins: over })).get('pins');
    expect(pinsPart).toBe(over.slice(0, LENS_MAX_PINNED).join('-'));
  });

  it('drops junk rather than throwing', () => {
    const r = decodePermalink('p=hi&mode=nonsense&pins=a-b-c&sel=zz');
    expect(r.prompt).toBe('hi');
    // A junk `mode` decodes to ABSENT (not a present `undefined`); `toBeUndefined`
    // alone can't tell the two apart, so assert absence explicitly.
    expect(Object.hasOwn(r, 'mode')).toBe(false);
    expect(r.mode).toBeUndefined();
    expect(r.pins).toEqual([]);
    expect(r.sel).toBeNull();
  });

  it('drops junk pins instead of coercing prefixes', () => {
    // `-1` and `7abc` must NOT survive as `1` / `7`: the field grammar rejects the
    // whole value, so `pins` decodes to `[]`.
    expect(decodePermalink('pins=-1').pins).toEqual([]);
    expect(decodePermalink('pins=7abc').pins).toEqual([]);
    // Digits-only but not a safe integer -> dropped.
    expect(decodePermalink('pins=99999999999999999999').pins).toEqual([]);
    // Empty value -> [] (grammar rejects the empty string).
    expect(decodePermalink('pins=').pins).toEqual([]);
  });

  it('drops a junk or wrong-arity sel to null', () => {
    expect(decodePermalink('sel=6abc,8xyz').sel).toBeNull(); // non-digit components
    expect(decodePermalink('sel=1,2,3').sel).toBeNull(); // three coordinates
    expect(decodePermalink('sel=5').sel).toBeNull(); // one coordinate
  });

  it('rejects a sel whose coord is a digit run beyond Number.MAX_SAFE_INTEGER', () => {
    // The regex passes (all digits, one comma) but parseInt rounds/overflows.
    // Encode never emits such a sel, so decode must not accept it.
    expect(decodePermalink('sel=9007199254740993,1').sel).toBeNull(); // rounds to 9007199254740992
    expect(decodePermalink('sel=' + '9'.repeat(400) + ',1').sel).toBeNull(); // overflows to Infinity
    // Guard the SECOND capture too: a fix that only checks layerIdx would pass the
    // first two but leak here.
    expect(decodePermalink('sel=1,9007199254740993').sel).toBeNull();
  });

  it('sanitizes a lone surrogate instead of throwing, at the cost of that one char', () => {
    const encoded = encodePermalink({ prompt: '\uD800', mode: 'logit', pins: [], sel: null });
    // The lone surrogate becomes U+FFFD (the replacement char), not a URIError.
    expect(decodePermalink(encoded).prompt).toBe('�');
  });

  it('round-trips a valid astral character byte-for-byte (toWellFormed left it intact)', () => {
    // A valid surrogate PAIR must survive untouched — proves the sanitizer only
    // rewrites LONE surrogates and does not damage well-formed input.
    const s: JSpaceState = { ...base, prompt: '😀' };
    expect(decodePermalink(encodePermalink(s)).prompt).toBe('😀');
  });

  it('decodes the two golden mode values', () => {
    expect(decodePermalink('mode=l').mode).toBe('logit');
    expect(decodePermalink('mode=j').mode).toBe('jacobian');
  });

  it('ENCODES mode=l for a logit state and mode=j for a jacobian state', () => {
    // Inspect the WIRE `mode` value directly: every existing round-trip test uses
    // `jacobian`, so an encoder that hard-coded `mode=j` would leave them all green
    // while silently restoring every `logit` link as `jacobian`.
    expect(new URLSearchParams(encodePermalink({ ...base, mode: 'logit' })).get('mode')).toBe('l');
    expect(new URLSearchParams(encodePermalink({ ...base, mode: 'jacobian' })).get('mode')).toBe('j');
  });

  it('round-trips every field of a LOGIT state (pins + sel present)', () => {
    // Mirror the jacobian `round-trips every field` test with the other mode so the
    // encoder's `logit` branch is proven end-to-end, not only through the decoder.
    const s: JSpaceState = { ...base, mode: 'logit' };
    expect(decodePermalink(encodePermalink(s))).toEqual(s);
  });

  it('encoder OMITS sel when a coordinate is negative or beyond MAX_SAFE_INTEGER', () => {
    // Inspect the WIRE string: routing through decode would also drop these and mask
    // a missing encoder guard.
    const negLayer = encodePermalink({ ...base, sel: { layerIdx: -1, pos: 8 } });
    expect(new URLSearchParams(negLayer).has('sel')).toBe(false);
    const unsafeCoord = encodePermalink({ ...base, sel: { layerIdx: Number.MAX_SAFE_INTEGER + 1, pos: 8 } });
    expect(new URLSearchParams(unsafeCoord).has('sel')).toBe(false);
  });

  it('emits no pins field when encoding a state whose pins contain a negative id', () => {
    const encoded = encodePermalink({ ...base, pins: [-1] });
    expect(new URLSearchParams(encoded).has('pins')).toBe(false);
  });

  it('returns an empty object for an empty hash', () => {
    expect(decodePermalink('')).toEqual({});
    expect(decodePermalink('#')).toEqual({});
  });
});

describe('applyPermalink overlay', () => {
  // Non-default `pins`/`sel` and a `mode` that DIFFERS from the golden `mode=j` so
  // the fallback cases actually prove the overlay rather than coincidentally
  // matching `[]`/`null`/`jacobian`.
  const defaults: JSpaceDefaults = {
    mode: 'logit',
    pins: [42],
    sel: { layerIdx: 1, pos: 2 },
  };

  it('yields exactly the defaults for an empty hash', () => {
    const s = applyPermalink(defaults, '');
    expect(s.prompt).toBe('');
    expect(s.mode).toBe(defaults.mode);
    expect(s.pins).toEqual(defaults.pins);
    expect(s.sel).toEqual(defaults.sel);
  });

  it('treats a bare "#" hash like an empty hash and yields the defaults', () => {
    // Proves the leading-`#` strip in applyPermalink feeds the empty-hash check:
    // '#' -> '' -> full defaults, not a non-empty snapshot that would zero pins/sel.
    const s = applyPermalink(defaults, '#');
    expect(s.prompt).toBe('');
    expect(s.mode).toBe(defaults.mode);
    expect(s.pins).toEqual(defaults.pins);
    expect(s.sel).toEqual(defaults.sel);
  });

  it('restores canonical empties (not defaults) for a non-empty mode-only hash', () => {
    // A non-empty hash IS a complete snapshot: `pins`/`sel` absent on the wire mean
    // the sender had none, so they must restore to []/null — NOT to the recipient's
    // nonempty defaults, which would silently graft foreign pins onto a shared link.
    const s = applyPermalink(defaults, 'mode=j');
    expect(s.mode).toBe('jacobian'); // overlaid, differs from defaults.mode ('logit')
    expect(s.pins).toEqual([]); // absent on the wire -> canonical empty, NOT defaults.pins
    expect(s.sel).toBeNull(); // absent on the wire -> canonical null, NOT defaults.sel
  });

  it('overrides pins and sel from a hash that carries them', () => {
    const s = applyPermalink(defaults, 'mode=j&pins=7-8&sel=3,4');
    expect(s.mode).toBe('jacobian');
    expect(s.pins).toEqual([7, 8]);
    expect(s.sel).toEqual({ layerIdx: 3, pos: 4 });
  });

  it('restores an empty-pins/null-sel snapshot without inheriting nonempty defaults', () => {
    const defaults = { mode: 'logit' as const, pins: [42], sel: { layerIdx: 1, pos: 2 } };
    const wire = encodePermalink({ prompt: 'x', mode: 'jacobian', pins: [], sel: null });
    const restored = applyPermalink(defaults, wire);
    expect(restored).toEqual({ prompt: 'x', mode: 'jacobian', pins: [], sel: null });
  });
});
