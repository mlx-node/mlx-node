// Concept-threading opacity ramp for the argmax grid. rank 0 (argmax) keeps the
// legacy 0.18 tint exactly; deeper top-K hits fade linearly to ALPHA_MIN.
export const ALPHA_MAX = 0.18;
export const ALPHA_MIN = 0.045;
export function tintAlphaForRank(hitRank: number, K: number): number {
  const frac = K > 1 ? hitRank / (K - 1) : 0; // 0 at argmax … 1 at top-K tail
  return ALPHA_MAX - (ALPHA_MAX - ALPHA_MIN) * frac;
}
