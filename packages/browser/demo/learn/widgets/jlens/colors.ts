// colors.ts — the tiny colour kit shared by the jlens children.
//
// Two independent scales live here so RankHeatmap and PinChips agree pixel for
// pixel:
//   1. A stable per-pin accent PALETTE — one colour per pinned token, reused by
//      PinChips (chip border/text) and RankHeatmap (that column's header) so a
//      chip and its heatmap column read as the same thing.
//   2. A compact inline VIRIDIS ramp for the rank fills. No d3 / colormap
//      dependency — five interpolated anchor stops are plenty for a ≤16×8 grid,
//      and staying dependency-free keeps the widget SSR-clean.
//
// Rank → colour goes through `rankToScore`: rank 1 (the answer is the model's
// top guess) maps to 1.0 (brightest / yellow), a rank at or past the display
// cap maps to 0.0 (darkest / purple). The mapping is logarithmic because ranks
// span 1..vocab and the interesting motion all happens in the first decade.

/** Display cap for full-vocab ranks — the worker already caps `ranks` at 999. */
export const RANK_CAP = 999;

/** Rank at/below this is considered "surfaced" (broken into the top of the list). */
export const SURFACE_RANK = 10;

/** Stable per-pin accents (index-aligned with the pinned tokens). Chosen to
 *  stay legible in both light and dark themes; used as literal hex so an 8-digit
 *  alpha suffix (e.g. `${c}1a`) yields a valid tint. */
export const PIN_PALETTE = ['#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#ef4444', '#22c55e', '#eab308'];

export function pinColor(i: number): string {
  return PIN_PALETTE[i % PIN_PALETTE.length]!;
}

type RGB = [number, number, number];

// Standard viridis anchor stops (dark purple → teal → yellow).
const VIRIDIS_STOPS: Array<[number, RGB]> = [
  [0.0, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.5, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1.0, [253, 231, 37]],
];

/** Piecewise-linear viridis lookup. `t` is clamped to [0,1]. Returns `rgb(...)`. */
export function viridis(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < VIRIDIS_STOPS.length; i++) {
    const [t1, c1] = VIRIDIS_STOPS[i]!;
    if (x <= t1) {
      const [t0, c0] = VIRIDIS_STOPS[i - 1]!;
      const f = t1 === t0 ? 0 : (x - t0) / (t1 - t0);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  const last = VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1]![1];
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
}

/** The `stop-color` values for a left(dark)→right(bright) legend gradient. */
export const VIRIDIS_LEGEND_STOPS: Array<{ offset: string; color: string }> = VIRIDIS_STOPS.map(([t, [r, g, b]]) => ({
  offset: `${Math.round(t * 100)}%`,
  color: `rgb(${r}, ${g}, ${b})`,
}));

/** Map a 1-based rank (1 = best) to a viridis score in [0,1] (1 = brightest). */
export function rankToScore(rank: number): number {
  const r = Math.max(1, Math.min(RANK_CAP, rank));
  return 1 - Math.log(r) / Math.log(RANK_CAP);
}

/** Readable ink for text drawn on a viridis fill of the given score. */
export function readableInk(score: number): string {
  return score > 0.55 ? '#0b0b0f' : '#f5f5f5';
}
