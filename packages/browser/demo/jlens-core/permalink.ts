/**
 * The /jspace permalink lives in the URL HASH, not the query string: the root
 * route runs `validateSearch: searchSchema.parse` over a plain z.object
 * (routes/__root.tsx:141), which strips unknown query keys on the first client
 * navigation. The hash also keeps user prompts out of server logs and referrers.
 *
 * Pure — no `window`, no `document`. The route component owns the hash I/O.
 *
 * A permalink restores STATE. It never runs: auto-running would auto-download
 * 1.6 GB of weights for a stranger who clicked a link.
 */
import { LENS_MAX_PINNED } from '../../src/inspector-types';

export type LensMode = 'logit' | 'jacobian';

export type JSpaceState = {
  prompt: string;
  mode: LensMode;
  pins: number[];
  sel: { layerIdx: number; pos: number } | null;
};

export function encodePermalink(s: JSpaceState): string {
  const parts = [`p=${encodeURIComponent(s.prompt)}`, `mode=${s.mode === 'jacobian' ? 'j' : 'l'}`];
  // Never emit a wire value that `decodePermalink` reads back as different, valid
  // state: filter pins to non-negative safe ints BEFORE the cap, and omit `sel`
  // unless both coordinates are non-negative safe ints.
  const pins = s.pins.filter((n) => Number.isSafeInteger(n) && n >= 0).slice(0, LENS_MAX_PINNED);
  if (pins.length > 0) parts.push(`pins=${pins.join('-')}`);
  if (
    s.sel &&
    Number.isSafeInteger(s.sel.layerIdx) &&
    s.sel.layerIdx >= 0 &&
    Number.isSafeInteger(s.sel.pos) &&
    s.sel.pos >= 0
  )
    parts.push(`sel=${s.sel.layerIdx},${s.sel.pos}`);
  return parts.join('&');
}

export function decodePermalink(hash: string): Partial<JSpaceState> {
  const raw = hash.startsWith('#') ? hash.slice(1) : hash;
  if (raw === '') return {};

  const out: Partial<JSpaceState> = {};
  const params = new URLSearchParams(raw);

  const prompt = params.get('p');
  if (prompt !== null) out.prompt = prompt;

  const mode = params.get('mode');
  if (mode === 'j') out.mode = 'jacobian';
  else if (mode === 'l') out.mode = 'logit';

  const pins = params.get('pins');
  if (pins !== null) {
    // Validate the WHOLE field against a grammar before converting: a per-element
    // predicate is useless here because `parseInt` is prefix-tolerant
    // (`parseInt('7abc') === 7`) and `'-1'.split('-')` is `['', '1']`. Only after
    // the grammar accepts do we map, drop unsafe ints (e.g. `99999999999999999999`),
    // and cap. The grammar rejects `''`, so an empty `pins=` yields `[]`.
    out.pins = /^\d+(-\d+)*$/.test(pins)
      ? pins
          .split('-')
          .map((x) => Number.parseInt(x, 10))
          .filter((n) => Number.isSafeInteger(n))
          .slice(0, LENS_MAX_PINNED)
      : [];
  }

  const sel = params.get('sel');
  if (sel !== null) {
    // Accept exactly two comma-separated non-negative integers, or reject to null.
    // Decode CANNOT range-check layerIdx/pos: it does not know the grid dimensions
    // (layers.length, promptLen). The consumer must clamp them against the actual
    // grid before use.
    const m = /^(\d+),(\d+)$/.exec(sel);
    out.sel = m ? { layerIdx: Number.parseInt(m[1]!, 10), pos: Number.parseInt(m[2]!, 10) } : null;
  }

  return out;
}
