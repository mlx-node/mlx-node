// workspace-metrics.ts — pure across-layer readouts that locate the "workspace
// band" from what lensReadout actually ships. FAITHFUL metrics read the pinned
// full-vocab rank track (exact for ranks below the display cap); PROXY metrics
// read the visible top-10 and are labeled as readout-space proxies in the UI
// (they are NOT the paper's activation/residual quantities). The impossible
// full-distribution metrics (excess kurtosis, residual autocorrelation,
// activation participation ratio) are intentionally absent — they need data no
// cell ships.
//
// CENSORING CAVEAT: `rankAt` returns RANK_CAP (999) both for a genuine rank
// at/beyond the cap and for any out-of-range lookup. A trajectory value of 999
// therefore means "at or beyond the cap — off-scale / not surfaced", NOT an
// exact rank. The UI must render 999 as an off-scale floor (RankChart already
// does) and label the trajectory accordingly — never as an exact measurement.
// `conceptTopKAccuracy` sidesteps the cap AND cutoff ties entirely: it reads
// actual top-k SET membership from `topKIds`, not the rank track — see its doc.
import type { LensSliceData } from './types';
import { RANK_CAP, SURFACE_RANK } from './colors';

/** FAITHFUL (with the RANK_CAP caveat above) — the pinned concept's full-vocab
 *  rank per displayed layer. Exact where rank < {@link RANK_CAP}; a value of
 *  {@link RANK_CAP} (999) is a censored floor (at/beyond cap → off-scale), NOT an
 *  exact rank. Render 999 as off-scale, never as a precise value. */
export function conceptRankTrajectory(slice: LensSliceData, pinIdx: number, pos = slice.promptLen - 1): number[] {
  return slice.layers.map((_, layerIdx) => slice.rankAt(pinIdx, layerIdx, pos));
}

/** True iff `rank` is the censored floor ({@link RANK_CAP}) rather than an exact
 *  value — a helper for the UI to mark off-scale trajectory points. */
export function isCensoredRank(rank: number): boolean {
  return rank >= RANK_CAP;
}

/** FAITHFUL, exact — per layer, 1 iff the pinned concept is IN the shipped top-k
 *  SET at `pos` (actual `topKIds` membership, NOT rank<=k). This is deliberate:
 *  backend rank = 1 + count(logit > pinned), so tokens tied at the cutoff can
 *  share rank k while being excluded from the selected top-k — a `rank<=k` test
 *  then FALSE-POSITIVES (verified on the grammar-error starter at ℓ22, where the
 *  pinned token has rank 10 yet is absent from the ten topKIds). Set membership is
 *  unambiguous, so this pip is genuinely exact. `k` is clamped to the shipped
 *  top-K width. */
export function conceptTopKAccuracy(
  slice: LensSliceData,
  pinIdx: number,
  k = SURFACE_RANK,
  pos = slice.promptLen - 1,
): Array<0 | 1> {
  const tokenId = slice.pinned[pinIdx]?.tokenId;
  if (tokenId == null) return slice.layers.map(() => 0);
  return slice.layers.map((_, layerIdx) => {
    const ids = slice.cellAt(layerIdx, pos).topKIds;
    const topk = k >= ids.length ? ids : ids.slice(0, k);
    return topk.includes(tokenId) ? 1 : 0;
  });
}

/** PROXY (readout-space, top-10) — Shannon entropy (nats) of the visible probs.
 *  Honest LOWER BOUND on the true entropy (omitted tail can only add). */
export function readoutEntropy(slice: LensSliceData, layerIdx: number, pos: number): number {
  const p = slice.cellAt(layerIdx, pos).topKProbs;
  let h = 0;
  for (let i = 0; i < p.length; i++) {
    const pi = p[i]!;
    if (pi > 0) h -= pi * Math.log(pi);
  }
  return h;
}

/** PROXY (readout-space, top-10) — inverse participation ratio 1/Σp² over the
 *  visible probs. OVERestimates the true PR (omitted tail can only lower it). */
export function readoutEffectiveDim(slice: LensSliceData, layerIdx: number, pos: number): number {
  const p = slice.cellAt(layerIdx, pos).topKProbs;
  let s = 0;
  for (let i = 0; i < p.length; i++) s += p[i]! * p[i]!;
  return s > 0 ? 1 / s : 0;
}

/** PROXY (readout-space) — adjacent-layer top-K set stability at `pos`: Jaccard
 *  SIMILARITY of topKIds between ℓ and ℓ+1. Stand-in for autocorrelation, NOT the
 *  residual metric. Length = layers − 1. */
export function topKSetStability(slice: LensSliceData, pos: number): number[] {
  const out: number[] = [];
  for (let l = 0; l < slice.layers.length - 1; l++) {
    const A = slice.cellAt(l, pos).topKIds;
    const B = slice.cellAt(l + 1, pos).topKIds;
    const setB = new Set(B);
    let inter = 0;
    for (const id of A) if (setB.has(id)) inter++;
    const union = A.length + B.length - inter;
    out.push(union === 0 ? 1 : inter / union);
  }
  return out;
}

/** The commit ("motor-flip") layer at `pos`: the LOWEST displayed layer index
 *  whose argmax equals the top layer's argmax AND stays equal for every layer
 *  above it. `layers−1` = the top guess only locks in at the final layer;
 *  `−1` for an empty slice. */
export function motorFlipLayer(slice: LensSliceData, pos: number): number {
  const top = slice.layers.length - 1;
  if (top < 0) return -1;
  const target = slice.cellAt(top, pos).argmaxId;
  let flip = top;
  for (let l = top - 1; l >= 0; l--) {
    if (slice.cellAt(l, pos).argmaxId === target) flip = l;
    else break;
  }
  return flip;
}
