// self-test.ts — the /jspace Jacobian self-test ORACLE.
//
// When the reader first switches to the fitted Jacobian lens, /jspace recomputes
// the committed `french-season` frame LIVE at its own settings (layers =
// JACOBIAN_LAYERS, topK = 10, the four french-season concept pins, promptLen = 9)
// and compares it against the offline bake. If they diverge, the fitted-Jacobian
// badge is refused and a warning surfaces — the reader is never shown a fitted
// lens the browser cannot actually reproduce.
//
// CALIBRATION — this is a GARBAGE DETECTOR, not a ULP gate. The baked logits are
// bf16-rounded while the browser keeps f32, so the browser can NEVER reproduce
// the bake bit-for-bit: on the real 99-cell frame the benign noise floor is 3/99
// argmax disagreements and 73/99 reordered top-K lists. The f16 STORAGE bug, by
// contrast, produced tokens with ZERO overlap with truth. The two thresholds
// below sit comfortably between those regimes.
//
// The metric that makes this work is `worstPinDelta = |min(baked.ranks) −
// min(live.ranks)|` per pin, then MAX over pins — MIN OVER RANKS, never max over
// cells. A max-over-cells pin metric reads 39 on the same benign frame and would
// fire on every single load; the min-over-ranks form reads 2 (thin, but real).
//
// Pure — no worker, no `window`, no I/O. The caller performs the live readout and
// revives the baked frame; this module only compares two `LensReadoutRun`s.

import type { LensReadoutRun } from '../../src/inspector-types';

export type SelfTestVerdict = {
  ok: boolean;
  /** Fraction of cells whose top-1 token id agrees (layer-major, same order). */
  topOneAgreement: number;
  /** Max over pins of |min(baked.ranks) − min(live.ranks)| — the best-rank gap. */
  worstPinDelta: number;
  /** Human-readable failure explanation; absent when `ok`. */
  reason?: string;
};

/** top-1 must agree on >= 90% of cells; each pin's BEST rank within 3. */
export const TOP_ONE_AGREEMENT_FLOOR = 0.9;
export const PIN_BEST_RANK_TOLERANCE = 3;

/** Minimum over an array-like of ranks (empty → Infinity). Avoids spreading a
 *  typed array into `Math.min(...)` (stack-fragile on large inputs). */
function minRank(ranks: ArrayLike<number>): number {
  let m = Infinity;
  for (let i = 0; i < ranks.length; i++) {
    const r = ranks[i]!;
    if (r < m) m = r;
  }
  return m;
}

/**
 * The fuzzy metrics below ASSUME the live frame is shape-aligned with the baked
 * frame (same cell count, pin identities/order, layers, topK, promptLen). A trust
 * gate must not infer that from a mis-shaped envelope: a frame truncated by one
 * cell still scores ~0.99 top-1 agreement, and reordered/renamed pins can slip
 * past the min-over-ranks metric. So reject any structural mismatch OUTRIGHT,
 * before the fuzzy comparison. Returns a human-readable reason, or null if aligned.
 */
function structuralMismatch(live: LensReadoutRun, baked: LensReadoutRun): string | null {
  if (live.cells.length !== baked.cells.length) {
    return `cell count ${live.cells.length} != baked ${baked.cells.length}`;
  }
  if (live.pinned.length !== baked.pinned.length) {
    return `pin count ${live.pinned.length} != baked ${baked.pinned.length}`;
  }
  if (live.promptLen !== baked.promptLen) {
    return `promptLen ${live.promptLen} != baked ${baked.promptLen}`;
  }
  if (live.topK !== baked.topK) {
    return `topK ${live.topK} != baked ${baked.topK}`;
  }
  if (live.layers.length !== baked.layers.length) {
    return `layer count ${live.layers.length} != baked ${baked.layers.length}`;
  }
  for (let i = 0; i < baked.layers.length; i++) {
    if (live.layers[i] !== baked.layers[i]) return `layer[${i}] ${live.layers[i]} != baked ${baked.layers[i]}`;
  }
  for (let i = 0; i < baked.pinned.length; i++) {
    if (live.pinned[i]!.tokenId !== baked.pinned[i]!.tokenId) {
      return `pin[${i}] id ${live.pinned[i]!.tokenId} != baked ${baked.pinned[i]!.tokenId} (identity/order)`;
    }
    if (live.pinned[i]!.ranks.length !== baked.pinned[i]!.ranks.length) {
      return `pin[${i}] rank length ${live.pinned[i]!.ranks.length} != baked ${baked.pinned[i]!.ranks.length}`;
    }
  }
  for (let i = 0; i < baked.cells.length; i++) {
    if (live.cells[i]!.topKIds.length !== baked.cells[i]!.topKIds.length) {
      return `cell[${i}] topK width ${live.cells[i]!.topKIds.length} != baked ${baked.cells[i]!.topKIds.length}`;
    }
  }
  return null;
}

/**
 * Compare a LIVE readout against the baked oracle frame. Both must have been
 * produced at the SAME settings (same layers, topK, pins, prompt), so `cells`
 * and `pinned` are index-aligned and layer-major on both sides.
 *
 *   - `topOneAgreement` = fraction of cells where `topKIds[0]` matches.
 *   - `worstPinDelta`    = max over pins of the best-rank gap
 *                          `|min(baked.ranks) − min(live.ranks)|`.
 *   - `ok`               = `topOneAgreement >= 0.9 && worstPinDelta <= 3`.
 */
export function compareToBakedFrame(live: LensReadoutRun, baked: LensReadoutRun): SelfTestVerdict {
  // --- structural gate: a mis-shaped envelope is garbage, not a near miss ---
  const shape = structuralMismatch(live, baked);
  if (shape !== null) {
    return { ok: false, topOneAgreement: 0, worstPinDelta: Infinity, reason: `structural mismatch: ${shape}` };
  }

  // --- top-1 agreement over cells -----------------------------------------
  const nCells = baked.cells.length;
  let matches = 0;
  for (let i = 0; i < nCells; i++) {
    const bTop = baked.cells[i]!.topKIds[0];
    const lTop = live.cells[i]?.topKIds[0];
    if (lTop !== undefined && lTop === bTop) matches++;
  }
  const topOneAgreement = nCells === 0 ? 0 : matches / nCells;

  // --- worst pin best-rank delta (MIN over ranks, then MAX over pins) ------
  let worstPinDelta = 0;
  let pinReason: string | undefined;
  for (let pi = 0; pi < baked.pinned.length; pi++) {
    const bakedPin = baked.pinned[pi]!;
    const livePin = live.pinned[pi];
    if (!livePin) {
      worstPinDelta = Infinity;
      pinReason = `pinned token index ${pi} is missing from the live run`;
      break;
    }
    const delta = Math.abs(minRank(bakedPin.ranks) - minRank(livePin.ranks));
    if (delta > worstPinDelta) worstPinDelta = delta;
  }

  const agreementOk = topOneAgreement >= TOP_ONE_AGREEMENT_FLOOR;
  const pinsOk = worstPinDelta <= PIN_BEST_RANK_TOLERANCE;
  const ok = agreementOk && pinsOk;

  let reason: string | undefined;
  if (!ok) {
    const parts: string[] = [];
    if (!agreementOk) {
      parts.push(`top-1 agreement ${topOneAgreement.toFixed(4)} < ${TOP_ONE_AGREEMENT_FLOOR}`);
    }
    if (!pinsOk) {
      parts.push(pinReason ?? `worst pin best-rank delta ${worstPinDelta} > ${PIN_BEST_RANK_TOLERANCE}`);
    }
    reason = parts.join('; ');
  }

  return { ok, topOneAgreement, worstPinDelta, reason };
}
