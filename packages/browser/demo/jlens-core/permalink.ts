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
  /**
   * Which model-free gallery tile the COLD starter view is showing. Carried on the
   * wire ONLY for a cold prompt (a live/custom prompt ignores it), so a shared cold
   * URL restores the exact tile the sender was viewing — not just the default.
   *
   * OPTIONAL because it is conditionally meaningful: a live/custom prompt has no
   * relevant tile, and the field is genuinely absent from the wire there (mirroring
   * how `pins: []` / `sel: null` are omitted). Absent ⇒ the app's default tile
   * (`defaults.starterSlug` in {@link applyPermalink}); the app always supplies one.
   */
  starterSlug?: string;
};

export function encodePermalink(s: JSpaceState): string {
  // `encodeURIComponent` THROWS `URIError` on a lone surrogate (e.g. '\uD800'), so
  // a type-valid state would crash permalink creation. `toWellFormed()` replaces
  // any unpaired surrogate with U+FFFD first. This is the codec's ONLY lossy path:
  // for a malformed-UTF-16 prompt `decode(encode(s)).prompt !== s.prompt`. That is
  // the right call — such a prompt cannot be tokenized anyway, so a non-throwing
  // lossy permalink beats a crash. Well-formed input (incl. astral pairs) is
  // untouched and round-trips byte-for-byte.
  const parts = [`p=${encodeURIComponent(s.prompt.toWellFormed())}`, `mode=${s.mode === 'jacobian' ? 'j' : 'l'}`];
  // `s=` (the gallery tile) is meaningful ONLY for the cold starter view; a live
  // prompt renders its own read, so we omit it there to keep custom permalinks clean.
  // The write-effect's cold-default guard suppresses the whole hash when the tile AND
  // mode are the app defaults, so a fresh cold visit still gets an empty URL.
  if (s.prompt.trim() === '' && s.starterSlug) parts.push(`s=${encodeURIComponent(s.starterSlug)}`);
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

/**
 * Decode a /jspace permalink hash into a SPARSE state patch.
 *
 * The result is a `Partial<JSpaceState>`: a KEY THAT IS ABSENT means "use the
 * app default" — `pins` → `[]`, `sel` → `null`, `mode` → the app's default
 * mode. The encoder deliberately OMITS empty/default fields from the wire
 * (`{ pins: [], sel: null }` encodes to just `p=…&mode=…`), so the decoder
 * reports those fields as absent own-properties, never as a present `undefined`.
 *
 * Consuming this correctly requires seeding a FULL default state and overlaying
 * the decoded keys — e.g.
 *
 *     const state = { pins: [], sel: null, mode: DEFAULT_MODE, ...decodePermalink(hash) };
 *
 * or, equivalently and preferred, via {@link applyPermalink}. You MUST NOT patch
 * the sparse result over live state (`Object.assign(live, decodePermalink(hash))`
 * or a spread over the current session): an absent `pins` would leave the OLD
 * pins in place instead of clearing them, so a shared permalink would leak the
 * previous session's pins/selection — the opposite of full-state restoration.
 *
 * Never throws on malformed input: junk `mode`/`pins`/`sel` decode to
 * absent/`[]`/`null` respectively rather than raising.
 */
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
    // The regex constrains only the CHARACTER SET, not numeric representability:
    // `parseInt` silently rounds `9007199254740993` and overflows a 400-digit run
    // to `Infinity`, which `encodePermalink` would never emit (it omits `sel` for
    // an unsafe coord). Range-check BOTH captures so decode rejects exactly what
    // encode refuses to produce. The regex forbids `-`, so a >= 0 check is
    // redundant and omitted.
    const m = /^(\d+),(\d+)$/.exec(sel);
    if (m) {
      const layerIdx = Number.parseInt(m[1]!, 10);
      const pos = Number.parseInt(m[2]!, 10);
      out.sel = Number.isSafeInteger(layerIdx) && Number.isSafeInteger(pos) ? { layerIdx, pos } : null;
    } else {
      out.sel = null;
    }
  }

  // Gallery tile slug (cold view only). Any non-empty string is accepted; the app
  // clamps an unknown slug to its default tile (`STARTERS[slug] ?? default`), so the
  // codec stays gallery-agnostic. An empty `s=` decodes as absent (→ app default).
  const starterSlug = params.get('s');
  if (starterSlug !== null && starterSlug !== '') out.starterSlug = starterSlug;

  return out;
}

export type JSpaceDefaults = Pick<JSpaceState, 'mode' | 'pins' | 'sel' | 'starterSlug'>;

/**
 * Resolve a /jspace hash into a FULL state, distinguishing two cases the encoder
 * conflates on the wire:
 *
 *   - EMPTY hash (`''` or bare `'#'`, a cold page load with no permalink): the
 *     app's `defaults` verbatim, plus `prompt: ''`.
 *   - NON-EMPTY hash (a real permalink): a COMPLETE state snapshot. The encoder
 *     omits fields it left empty (`pins: []`) or null (`sel: null`), so those
 *     keys are absent on the wire — but they restore to the CANONICAL EMPTIES
 *     (`[]` / `null`), NEVER to `defaults`. Otherwise a shared "no pins / no
 *     selection" link would silently inherit the recipient's pins and selection,
 *     defeating the whole point of a full-state permalink. Only `mode` may fall
 *     back to `defaults` (a hand-edited hash can drop it; a generated permalink
 *     always carries it), and `prompt` falls back to `''`.
 *
 * This is the ONLY correct way to consume {@link decodePermalink} — never patch
 * the sparse result over live state, or a shared permalink would leak the
 * previous session's pins/selection.
 *
 * Pure: no globals, no I/O. The leading `#` is stripped here (as well as inside
 * `decodePermalink`) so a bare `'#'` is treated as an empty hash.
 */
export function applyPermalink(defaults: JSpaceDefaults, hash: string): JSpaceState {
  const raw = hash.startsWith('#') ? hash.slice(1) : hash;
  // No permalink at all (bare page load): the app's cold-start defaults, with a
  // blank prompt. Project the three fields EXPLICITLY, never `...defaults`:
  // `JSpaceDefaults` is a structural `Pick`, so a full `JSpaceState` is an
  // accepted argument, and a spread would copy its `prompt` over the `''` (and
  // could drop keys if `defaults` were getter-backed).
  if (raw === '') {
    return { prompt: '', mode: defaults.mode, pins: defaults.pins, sel: defaults.sel, starterSlug: defaults.starterSlug };
  }
  // A non-empty hash IS a complete state snapshot. Fields the encoder omits
  // because they were empty (`pins`) or null (`sel`) restore to those CANONICAL
  // EMPTIES — never to `defaults`. `starterSlug` is the exception: it has no
  // canonical empty (the cold view always shows SOME tile), so an absent `s=`
  // restores the app's default tile via `defaults.starterSlug`.
  const patch = decodePermalink(raw);
  return {
    prompt: patch.prompt ?? '',
    mode: patch.mode ?? defaults.mode,
    pins: patch.pins ?? [],
    sel: patch.sel ?? null,
    starterSlug: patch.starterSlug ?? defaults.starterSlug,
  };
}
