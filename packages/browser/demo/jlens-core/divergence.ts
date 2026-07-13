// divergence.ts — pure per-cell logit↔Jacobian divergence metrics. Compares two
// slices of the SAME source (never baked-vs-live). All values are divergence-
// valued: higher = the two lenses disagree more.
import type { LensCell } from '../../src/inspector-types';
import type { LensSliceData } from './types';

/** Binary top-1 divergence: 0 if the two lenses share an argmax, 1 if not. */
export function argmaxDisagree(a: LensCell, b: LensCell): 0 | 1 {
  return a.argmaxId === b.argmaxId ? 0 : 1;
}

/** Jaccard DISTANCE in [0,1] over the two top-K id sets: 0 = identical, 1 =
 *  disjoint. The default cell fill. Truncated at the shipped top-K (10). */
export function jaccardTopK(a: LensCell, b: LensCell): number {
  const A = a.topKIds;
  const B = b.topKIds;
  if (A.length === 0 && B.length === 0) return 0;
  const setB = new Set(B);
  let inter = 0;
  for (const id of A) if (setB.has(id)) inter++;
  const union = A.length + B.length - inter;
  return union === 0 ? 0 : 1 - inter / union;
}

/** |rank_A − rank_B| for a pinned concept at one cell, from the EXACT full-vocab
 *  rank tracks (escapes the depth-10 truncation). Both slices must share pin order. */
export function pinnedRankDelta(
  a: LensSliceData,
  b: LensSliceData,
  pinIdx: number,
  layerIdx: number,
  pos: number,
): number {
  return Math.abs(a.rankAt(pinIdx, layerIdx, pos) - b.rankAt(pinIdx, layerIdx, pos));
}
