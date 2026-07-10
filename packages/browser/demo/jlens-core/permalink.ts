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
  if (s.pins.length > 0) parts.push(`pins=${s.pins.slice(0, LENS_MAX_PINNED).join('-')}`);
  if (s.sel) parts.push(`sel=${s.sel.layerIdx},${s.sel.pos}`);
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
    out.pins = pins
      .split('-')
      .map((x) => Number.parseInt(x, 10))
      .filter((n) => Number.isInteger(n) && n >= 0)
      .slice(0, LENS_MAX_PINNED);
  }

  const sel = params.get('sel');
  if (sel !== null) {
    const [a, b] = sel.split(',').map((x) => Number.parseInt(x, 10));
    out.sel =
      Number.isInteger(a) && Number.isInteger(b) && a! >= 0 && b! >= 0
        ? { layerIdx: a!, pos: b! }
        : null;
  }

  return out;
}
