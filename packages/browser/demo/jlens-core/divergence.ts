// divergence.ts — pure per-cell logit↔Jacobian divergence metrics. Compares two
// slices of the SAME source (never baked-vs-live). All values are divergence-
// valued: higher = the two lenses disagree more.
//
// Both metrics read only `topKIds`/`argmaxId` — the actual shipped top-10, which
// is exact set membership (NOT rank-censored). A pinned-rank delta was
// deliberately NOT shipped: `rankAt` returns RANK_CAP (999) both for a genuine
// rank at/beyond the cap AND for a missing lookup, so |rankA − rankB| would
// report false agreement (both capped) or a fabricated exact gap (one capped).
// Top-K set overlap has no such censoring, so it is the honest divergence signal
// (its only limit is depth 10, a stated scope bound — not a hidden lie).
import type { LensCell } from '../../src/inspector-types';

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
